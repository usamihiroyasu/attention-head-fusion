"""Frozen-model, fixed-prefix attention diagnostic. No training or generation."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    result = dict(state="RUNNING", phase="assets", native_forwards=0, projection_calls=0,
                  pid=os.getpid(), rows=[], worker_sha256=digest(__file__))
    def save():
        p = a.output.with_suffix(".tmp")
        p.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        os.replace(p, a.output)
    started = time.monotonic()
    try:
        ckpt = a.assets / "pytorch_model.bin"
        assert digest(ckpt) == "fd223b14932b8b66605c2a688e7d7193058ff2b40b2c6f6c1547c227d06d8417"
        assert digest(a.prompts) == "cc86af666e773c956163d638f3a1a46b29091509ac5e28c940481128719497d3"
        bank = [json.loads(x) for x in a.prompts.read_text().splitlines() if x.strip()]
        assert len(bank) == 64 and all("prompt" in x and "label" not in x for x in bank)
        result["rows"] = [dict(index=i, prompt_id=x["prompt_id"], status="PENDING") for i,x in enumerate(bank)]
        result["phase"] = "imports"; save()
        import numpy as np
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
        from peft import LoraConfig, get_peft_model
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        assert torch.cuda.is_available(), "No admitted CUDA device; CPU fallback forbidden"
        result["versions"] = {x: importlib.metadata.version(x) for x in ["torch","transformers","peft","numpy"]}
        result["timing_definition"] = "One CUDA-event interval brackets 20 consecutive kernel calls; total/20 is an amortized mean, not 20 independent timing samples."
        assert result["versions"]["transformers"] == "5.17.0" and result["versions"]["peft"] == "0.18.1"
        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                cfg = GPT2Config.from_json_file(str(a.assets / "config.json"))
                cfg.vocab_size = 50260
                cfg._attn_implementation = "eager"
                base = GPT2LMHeadModel(cfg)
                lc = LoraConfig(r=128, lora_alpha=32, lora_dropout=.05,
                                target_modules=["c_attn","c_proj","c_fc"],
                                fan_in_fan_out=True, bias="none", task_type="CAUSAL_LM",
                                use_dora=False)
                self.codi = get_peft_model(base, lc)
                self.prj = torch.nn.Sequential(torch.nn.Dropout(0), torch.nn.Linear(768,768),
                                               torch.nn.GELU(), torch.nn.Linear(768,768))
                self.prj.add_module("ln", torch.nn.LayerNorm(768))
        result["phase"] = "strict_load"; save()
        model = Wrapper().float()
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        before = model.state_dict()
        assert len(state) == len(before) == 251
        assert set(state) == set(before), {"missing":sorted(set(before)-set(state)), "unexpected":sorted(set(state)-set(before))}
        assert all(before[k].shape == state[k].shape for k in state)
        incompat = model.load_state_dict(state, strict=True)
        assert not incompat.missing_keys and not incompat.unexpected_keys
        loaded = model.state_dict()
        assert all(torch.equal(loaded[k], state[k].to(dtype=loaded[k].dtype)) for k in state)
        base = model.codi.base_model.model
        assert base.lm_head.weight.data_ptr() == base.transformer.wte.weight.data_ptr()
        result["load_gate"] = dict(keys=251, strict=True, all_loaded_values_equal=True, head_wte_tied=True)
        del before, loaded, state
        model.requires_grad_(False).eval().to("cuda")
        attn = base.transformer.h[0].attn
        assert base.config.n_head == 12 and base.config.n_embd == 768
        assert bool(attn.scale_attn_weights) and abs(float(attn.scaling)-0.125) < 1e-12
        tok = GPT2Tokenizer.from_pretrained(str(a.assets), local_files_only=True)
        result["phase"] = "native_queries"; save()
        def kernel(q, k, v):
            z = torch.einsum("nhd,hmd->nhm", q, k)
            return torch.einsum("nhm,hm->n", torch.softmax(z, dim=-1), v)
        def metrics(pred, target):
            err = pred - target
            rms = float(torch.sqrt(torch.mean(target.square())))
            rmse = float(torch.sqrt(torch.mean(err.square())))
            return dict(max_abs_error=float(err.abs().max()), mean_abs_error=float(err.abs().mean()),
                        rmse=rmse, rms_target=rms, normalized_rmse=rmse/rms if rms >= 1e-8 else None,
                        target_degenerate=rms < 1e-8)
        with torch.inference_mode():
            for i, item in enumerate(bank):
                row = result["rows"][i]
                try:
                    ids = tok.encode(item["prompt"], add_special_tokens=False)[:96]
                    T = len(ids)
                    row.update(tokens=T, token_ids_sha256=hashlib.sha256(np.asarray(ids,dtype="<i8").tobytes()).hexdigest())
                    if T < 64:
                        row["status"] = "INELIGIBLE_SHORT_CONTEXT"; save(); continue
                    assert max(ids) < 50257
                    captured = []
                    handle = attn.c_attn.register_forward_hook(lambda mod, inp, out: captured.append(out.detach().clone()))
                    try:
                        result["native_forwards"] += 1
                        assert result["native_forwards"] <= 64
                        model.codi(input_ids=torch.tensor([ids],device="cuda"), use_cache=False, return_dict=True)
                    finally:
                        handle.remove()
                    assert len(captured) == 1 and captured[0].shape == (1,T,2304)
                    q, k, v = [x.reshape(T,12,64) for x in captured[0][0].split(768,dim=-1)]
                    q = (q[:, :2, :] / 8).contiguous()
                    kp = k[:16, :2, :].permute(1,0,2).contiguous()
                    vp = v[:16, :2, :].permute(1,0,2).contiguous()
                    iso = torch.zeros((33,768),device="cuda",dtype=torch.float32)
                    for h in range(2): iso[h*16:(h+1)*16,h*64:(h+1)*64] = vp[h]
                    projected = attn.c_proj(iso)
                    result["projection_calls"] += 1
                    sv = (projected[:32,0] - projected[32,0]).reshape(2,16)
                    qc, qt = q[16:48].double().cpu(), q[48:T].double().cpu()
                    kc, vc = kp.double().cpu(), sv.double().cpu()
                    assert all(bool(torch.isfinite(z).all()) for z in [qc,qt,kc,vc])
                    fk = torch.cat((kc[0,:,None,:].expand(16,16,64),
                                    kc[1,None,:,:].expand(16,16,64)),dim=-1).reshape(1,256,128).contiguous()
                    fv = (vc[0,:,None]+vc[1,None,:]).reshape(1,256).contiguous()
                    qcf, qtf = qc.reshape(32,1,128), qt.reshape(T-48,1,128)
                    target = kernel(qt,kc,vc)
                    full = kernel(qtf,fk,fv)
                    assert torch.isfinite(target).all() and torch.isfinite(full).all()
                    scale = 1 + float(target.abs().max())
                    row["cpu_fusion_max_abs_error"] = float((full-target).abs().max())
                    assert row["cpu_fusion_max_abs_error"] <= 1e-9*scale
                    imp = torch.softmax(torch.einsum("nhd,hmd->nhm",qcf,fk),dim=-1).mean(dim=0)[0].numpy()
                    top = np.argsort(-imp,kind="stable")
                    uniform = np.random.Generator(np.random.PCG64(20261101+i)).permutation(256)
                    variants = [("separate",qt,kc,vc),("fused_full",qtf,fk,fv)]
                    for family, order in [("topmass",top),("uniform",uniform)]:
                        for budget in [32,64]:
                            ix = torch.tensor(order[:budget].copy(),dtype=torch.long)
                            variants.append((f"{family}_{budget}",qtf,fk[:,ix,:].clone(),fv[:,ix].clone()))
                    methods = []
                    for name, qq, kk, vv in variants:
                        pred = kernel(qq,kk,vv)
                        assert torch.isfinite(pred).all()
                        entry = dict(method=name, queries=T-48, calibration_queries=32,
                                     head_atom_terms=int(kk.shape[0]*kk.shape[1]), heads=int(kk.shape[0]),
                                     fp64=metrics(pred,target))
                        qg, kg, vg = [z.float().to("cuda").contiguous() for z in [qq,kk,vv]]
                        gp = kernel(qg,kg,vg)
                        assert torch.isfinite(gp).all()
                        entry["gpu"] = metrics(gp.double().cpu(),target)
                        if name == "separate":
                            gpu_reference = gp.detach().clone()
                        if name == "fused_full":
                            row["gpu_fusion_pair_max_abs_error"] = float((gp-gpu_reference).abs().max())
                            assert row["gpu_fusion_pair_max_abs_error"] <= 2e-5*scale
                        if name in ["separate","fused_full"]:
                            assert entry["gpu"]["max_abs_error"] <= 2e-5*scale
                        for _ in range(3): kernel(qg,kg,vg)
                        torch.cuda.synchronize()
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record()
                        for _ in range(20): kernel(qg,kg,vg)
                        end.record(); end.synchronize()
                        entry.update(kernel_ms_mean20=begin.elapsed_time(end)/20,
                                     KV_bytes=kg.numel()*kg.element_size()+vg.numel()*vg.element_size(),
                                     key_shape=list(kg.shape),value_shape=list(vg.shape),gpu_dtype="float32")
                        methods.append(entry)
                        del qg, kg, vg, gp
                    row.update(status="COMPLETE", heldout_queries=T-48, methods=methods)
                    save()
                except Exception as exc:
                    row.update(status="FAILED",error=type(exc).__name__+":"+str(exc))
                    save(); raise
        result.update(state="COMPLETE", phase="done",
                      eligible=sum(r["status"]=="COMPLETE" for r in result["rows"]),
                      ineligible=sum(r["status"].startswith("INELIGIBLE") for r in result["rows"]))
    except Exception as exc:
        result.update(state="FAILED",error=type(exc).__name__+":"+str(exc))
    finally:
        result["elapsed_seconds"] = time.monotonic()-started
        save()
    print(json.dumps({k:result.get(k) for k in ["state","phase","native_forwards","eligible","ineligible","error"]}))
    return 0 if result["state"]=="COMPLETE" else 2

if __name__ == "__main__":
    raise SystemExit(main())
