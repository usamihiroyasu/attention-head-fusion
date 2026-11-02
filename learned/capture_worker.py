"""Frozen-model whole-vector QKV capture only. No fit, heldout target, SVD, generation or c_proj probe."""
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
    result = dict(state="RUNNING", phase="assets", native_forwards=0, extra_projection_probe_calls=0,
                  pid=os.getpid(), rows=[], worker_sha256=digest(__file__))
    def save():
        p = a.output.with_suffix(".tmp")
        p.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        os.replace(p, a.output)
    result.update(scope="H2/H4 native-prefix capture only", query_scaling="native q/8; no later head-dimension scale", heldout_targets_evaluated=False, heldout_statistics_evaluated=False)
    started = time.monotonic()
    try:
        expected_assets = {
            "config.json":"0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4",
            "merges.txt":"1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
            "tokenizer.json":"8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
            "tokenizer_config.json":"5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f",
            "vocab.json":"196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"}
        assert all(digest(a.assets / name)==sha for name,sha in expected_assets.items())
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
        root = a.output.parent / "capture"
        root.mkdir(mode=0o700)
        caldir, helddir = root / "cal", root / "held"
        caldir.mkdir(mode=0o700); helddir.mkdir(mode=0o700)
        result["capture_schema"] = dict(calibration="qcal[32,4,64],kprefix[4,16,64],vprefix[4,16,64]", heldout="qheld[T-48,4,64] only; no target/statistics", dtype="float32", array_format="numpy npz; no object arrays")
        result["input_sha256"] = {"checkpoint": digest(ckpt), "promptbank": digest(a.prompts), "config": digest(a.assets / "config.json")}
        def atomic_npz(path, **arrays):
            assert not path.exists()
            temp = path.with_suffix(".npz.tmp")
            with temp.open("xb") as f:
                np.savez(f, **arrays); f.flush(); os.fsync(f.fileno())
            os.replace(temp, path)
            return dict(relative_path=str(path.relative_to(a.output.parent)), sha256=digest(path), bytes=path.stat().st_size,
                        arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype),nbytes=v.nbytes) for k,v in arrays.items()})
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
                    q = (q[:, :4, :] / 8).contiguous()
                    kp = k[:16, :4, :].permute(1,0,2).contiguous()
                    vp = v[:16, :4, :].permute(1,0,2).contiguous()
                    assert all(bool(torch.isfinite(z).all()) for z in [q,kp,vp])
                    qc = q[16:48].cpu().numpy().copy()
                    qh = q[48:T].cpu().numpy().copy()
                    kc, vc = kp.cpu().numpy().copy(), vp.cpu().numpy().copy()
                    assert qc.shape == (32,4,64) and qh.shape == (T-48,4,64)
                    assert kc.shape == vc.shape == (4,16,64)
                    assert all(z.dtype == np.float32 for z in [qc,qh,kc,vc])
                    row["calibration_asset"] = atomic_npz(caldir / f"{i:03d}.npz",qcal=qc,kprefix=kc,vprefix=vc)
                    row["heldout_asset"] = atomic_npz(helddir / f"{i:03d}.npz",qheld=qh)
                    row.update(status="COMPLETE", calibration_queries=32, heldout_queries=T-48)
                    save()
                    del captured, q, k, v, kp, vp, qc, qh, kc, vc
                except Exception as exc:
                    row.update(status="FAILED",error=type(exc).__name__+":"+str(exc))
                    save(); raise
        result.update(state="COMPLETE", phase="done", eligible=sum(r["status"]=="COMPLETE" for r in result["rows"]),
                      ineligible=sum(r["status"].startswith("INELIGIBLE") for r in result["rows"]))
    except Exception as exc:
        result.update(state="FAILED",error=type(exc).__name__+":"+str(exc))
    finally:
        result["elapsed_seconds"] = time.monotonic()-started
        save()
    print(json.dumps({k:result.get(k) for k in ["state","phase","native_forwards","extra_projection_probe_calls","eligible","ineligible","error"]}))
    return 0 if result["state"]=="COMPLETE" else 2

if __name__ == "__main__":
    raise SystemExit(main())
