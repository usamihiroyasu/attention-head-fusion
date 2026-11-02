"""Frozen cache learner. Public-query capture is read-only; selectors precede heldout reads."""
import argparse, hashlib, importlib.util, json, os, time
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, obj):
    t=p.with_suffix('.tmp'); t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n'); os.replace(t,p)
def kmeans(pool,n,rng):
    chosen=[int(rng.integers(len(pool)))]; distance=((pool-pool[chosen[0]])**2).sum(1)
    while len(chosen)<n:
        if distance.sum()>0: j=int(rng.choice(len(pool),p=distance/distance.sum()))
        else: j=next(j for j in range(len(pool)) if j not in chosen)
        chosen.append(j); distance=np.minimum(distance,((pool-pool[j])**2).sum(1))
    centers=pool[chosen].copy()
    for _ in range(20):
        d=((pool[:,None,:]-centers[None,:,:])**2).sum(-1); labels=d.argmin(1)
        for j in range(n):
            mask=labels==j
            if mask.any(): centers[j]=pool[mask].mean(0)
    labels=((pool[:,None,:]-centers[None,:,:])**2).sum(-1).argmin(1)
    b=np.log(np.bincount(labels,minlength=n)+1e-3)
    return centers-centers.mean(0),b-b.mean()
def attention(q,k,v,b=None):
    z=q@k.T
    if b is not None: z=z+b
    z=z-z.max(-1,keepdims=True); a=np.exp(z); a/=a.sum(-1,keepdims=True)
    return a@v
def teacher(q,k,v,H):
    return np.concatenate([attention(q[:,h],k[h],v[h]) for h in range(H)],1)
def identity(q,k,v):
    out={}
    for dtype,tol in ((np.float64,1e-9),(np.float32,2e-5)):
        q=q.astype(dtype);kk=k.astype(dtype);vv=v.astype(dtype)
        fk=np.concatenate((np.repeat(kk[0],16,axis=0),np.tile(kk[1],(16,1))),1)
        fv=np.concatenate((np.repeat(vv[0],16,axis=0),np.tile(vv[1],(16,1))),1)
        y=teacher(q,kk,vv,2);pred=attention(q[:,:2].reshape(len(q),128),fk,fv)
        err=float(np.abs(pred-y).max());assert err<=tol*(1+float(np.abs(y).max()))
        out[str(dtype)]=err
    return out
def predict(q,K,V,b,H,separate):
    if separate: return np.concatenate([attention(q[:,h],K[h],V[h],b[h]) for h in range(H)],1)
    return attention(q[:,:H].reshape(len(q),64*H),K,V,b)
def metrics(pred,y):
    e=pred-y; den=float(np.sqrt(np.mean(y*y))); centered=float(np.sqrt(np.mean((y-y.mean(0))**2)))
    rmse=float(np.sqrt(np.mean(e*e)))
    return dict(rmse=rmse,nrmse=rmse/den if den>=1e-8 else None,
                centered_nrmse=rmse/centered if centered>=1e-8 else None,
                target_rms=den,centered_target_rms=centered,max_abs=float(np.abs(e).max()))
def main():
    global np
    import numpy as np
    import torch
    ap=argparse.ArgumentParser();ap.add_argument('--capture',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--fitter',type=Path,required=True);ap.add_argument('--fitter-sha',required=True);a=ap.parse_args()
    assert sha(a.fitter)==a.fitter_sha
    spec=importlib.util.spec_from_file_location('pinned_fitter',a.fitter);f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    result=dict(state='RUNNING',phase='math_smoke',model_forwards=0,conditions=[],worker_sha256=sha(__file__),fitter_sha256=sha(a.fitter))
    start=time.monotonic();write(a.output,result)
    try:
        result['math_smoke']=f.smoke_test();result['phase']='load_cal_only';write(a.output,result)
        cap=json.loads((a.capture/'capture_receipt.json').read_text());assert cap['state']=='COMPLETE' and cap['eligible']==64
        cal=[]
        for row in cap['rows']:
            p=a.capture/row['calibration_asset']['relative_path'];assert sha(p)==row['calibration_asset']['sha256']
            with np.load(p,allow_pickle=False) as z: cal.append({k:z[k].astype(np.float64) for k in z.files})
        outdir=a.output.parent/'records';outdir.mkdir(mode=0o700)
        result['calibration_identity']=[identity(c['qcal'],c['kprefix'],c['vprefix']) for c in cal]
        result['calibration_target_diagnostics']={}
        all_records={};selectors={};seeds=[20261101,20261102,20261103,20261104]
        for H in (2,4):
            ys=[teacher(c['qcal'],c['kprefix'],c['vprefix'],H) for c in cal]
            rs=[np.maximum(np.sqrt(np.mean(c['qcal'][:,:H]**2,axis=0)),1e-6) for c in cal]
            scales=[max(float(np.sqrt(np.mean(y*y))),1e-8) for y in ys]
            result['calibration_target_diagnostics'][str(H)]=[
                dict(uncentered_sv=np.linalg.svd(y,compute_uv=False).tolist(),centered_sv=np.linalg.svd(y-y.mean(0),compute_uv=False).tolist(),
                     per_head_uncentered_sv=[np.linalg.svd(y[:,h*64:(h+1)*64],compute_uv=False).tolist() for h in range(H)],
                     per_head_centered_sv=[np.linalg.svd(y[:,h*64:(h+1)*64]-y[:,h*64:(h+1)*64].mean(0),compute_uv=False).tolist() for h in range(H)]) for y in ys]
            result.setdefault('calibration_query_diagnostics',{})[str(H)]=[]
            for i,c in enumerate(cal):
                sv=np.linalg.svd(c['qcal'][:,:H].reshape(32,64*H),compute_uv=False)
                result['calibration_query_diagnostics'][str(H)].append(dict(uncentered_sv=sv.tolist(),effective_rank=int(np.sum(sv>1e-6*sv[0])) if sv[0]>0 else 0,
                    query_RMS_floor_coordinates=int(np.sum(np.sqrt(np.mean(c['qcal'][:,:H]**2,axis=0))<1e-6)),target_RMS_floor_active=bool(np.sqrt(np.mean(ys[i]**2))<1e-8)))
            for n in (4,8):
                for arm in ('separate','flat'):
                    separate=arm=='separate'; tag=f'H{H}_n{n}_{arm}'
                    result['phase']='fit_'+tag;write(a.output,result)
                    xs=[];targets=[];ks=[];bs=[]
                    for i,c in enumerate(cal):
                        for seed in seeds:
                            for h in (range(H) if separate else [H]):
                                rng=np.random.default_rng(np.random.SeedSequence([seed,i,H,h,n]))
                                if separate:
                                    pool=c['kprefix'][h]*rs[i][h];x=c['qcal'][:,h]/rs[i][h];y=ys[i][:,h*64:(h+1)*64]/scales[i]
                                else:
                                    ix=np.arange(256) if H==2 else rng.choice(65536,size=512,replace=False)
                                    digits=np.array([(ix//(16**j))%16 for j in range(H-1,-1,-1)]).T
                                    pool=np.concatenate([c['kprefix'][j,digits[:,j]]*rs[i][j] for j in range(H)],1)
                                    x=(c['qcal'][:,:H]/rs[i]).reshape(32,64*H);y=ys[i]/scales[i]
                                k,b=kmeans(pool,n,rng);xs.append(x);targets.append(y);ks.append(k);bs.append(b)
                    tensor=lambda z:torch.as_tensor(np.asarray(z),dtype=torch.float64,device='cuda')
                    def progress(step,trace):
                        result['inflight']=dict(tag=tag,completed_updates=step,final_state_exported=False,trace=trace);write(a.output,result)
                    t=time.monotonic();fit=f.fit_batch(tensor(xs),tensor(targets),tensor(ks),tensor(bs),steps=512,lam=1e-6,lr=.03,Dtotal=64*H,progress=progress)
                    torch.cuda.synchronize();elapsed=time.monotonic()-t
                    K=fit['K'].cpu().numpy();V=fit['V'].cpu().numpy();b=fit['b'].cpu().numpy()
                    if separate:
                        K=K.reshape(64,4,H,n,64);V=V.reshape(64,4,H,n,64);b=b.reshape(64,4,H,n)
                        for i in range(64): K[i]/=rs[i][None,:,None,:];V[i]*=scales[i]
                    else:
                        K=K.reshape(64,4,n,64*H);V=V.reshape(64,4,n,64*H);b=b.reshape(64,4,n)
                        for i in range(64): K[i]/=rs[i].reshape(1,1,64*H);V[i]*=scales[i]
                    K=K.astype(np.float32);V=V.astype(np.float32);b=b.astype(np.float32)
                    assert all(np.isfinite(z).all() for z in (K,V,b))
                    allocated=4*129*H*n;actual=(K[0,0].nbytes+V[0,0].nbytes+b[0,0].nbytes)
                    assert allocated-actual==(0 if separate else 4*n*(H-1))
                    padding=np.zeros((64,4,allocated-actual),dtype=np.uint8)
                    path=outdir/(tag+'.npz');np.savez(path,K=K,V=V,b=b,padding=padding)
                    cm=[];chosen=[]
                    for i,c in enumerate(cal):
                        vals=[float(np.mean((predict(c['qcal'],K[i,s].astype(float),V[i,s].astype(float),b[i,s].astype(float),H,separate)-ys[i])**2)) for s in range(4)]
                        cm.append(vals);chosen.append(int(np.argmin(vals)))
                    all_records[tag]=(K,V,b,chosen);selectors[tag]=chosen
                    result['conditions'].append(dict(tag=tag,H=H,n=n,arm=arm,updates=512,starts=4,prompts=64,allocated_bytes=allocated,array_bytes=actual,padding_bytes=allocated-actual,training_seconds=elapsed,calibration_mse_all_starts=cm,selected_starts=chosen,records_sha256=sha(path),diagnostics=fit['diag']))
                    result.pop('inflight',None);write(a.output,result);del fit,xs,targets,ks,bs
        selectorpath=a.output.parent/'FROZEN_SELECTORS.json';write(selectorpath,dict(selectors=selectors,selection='calibration_only_FP32_export_MSE'))
        result.update(phase='heldout_after_selector_freeze',selector_sha256=sha(selectorpath));write(a.output,result)
        evaluations=[]
        for i,row in enumerate(cap['rows']):
            hp=a.capture/row['heldout_asset']['relative_path'];assert sha(hp)==row['heldout_asset']['sha256']
            with np.load(hp,allow_pickle=False) as z: qh=z['qheld'].astype(float)
            c=cal[i]
            held_identity=identity(qh,c['kprefix'],c['vprefix'])
            for H in (2,4):
                y=teacher(qh,c['kprefix'],c['vprefix'],H);yc=y-y.mean(0)
                u,sv,vt=np.linalg.svd(yc,full_matrices=False);den=np.linalg.norm(y)
                calmean=teacher(c['qcal'],c['kprefix'],c['vprefix'],H).mean(0)
                d=dict(prompt=i,H=H,heldout_queries=len(qh),target_singular_values=sv.tolist(),svd_reconstruction_relative=float(np.linalg.norm((u*sv)@vt-yc)/max(1,np.linalg.norm(yc))),
                       calibration_mean_baseline=metrics(np.broadcast_to(calmean,y.shape),y),heldout_H2_identity=held_identity,rows=[])
                for n in (4,8):
                    floor=float(np.sqrt(np.sum(sv[n-1:]**2))/den) if float(np.sqrt(np.mean(y*y)))>=1e-8 else None
                    for arm in ('separate','flat'):
                        tag=f'H{H}_n{n}_{arm}';K,V,b,chosen=all_records[tag]
                        ms=[metrics(predict(qh,K[i,s].astype(float),V[i,s].astype(float),b[i,s].astype(float),H,arm=='separate'),y) for s in range(4)]
                        d['rows'].append(dict(tag=tag,selected_start=chosen[i],all_starts=ms,selected=ms[chosen[i]],flat_class_affine_rank_nrmse_floor=floor if arm=='flat' else None))
                evaluations.append(d)
        result.update(state='COMPLETE',phase='done',evaluations=evaluations,heldout_selection=False,scientific_model_forwards=0)
    except Exception as e: result.update(state='FAILED',error=type(e).__name__+':'+str(e))
    finally: result['elapsed_seconds']=time.monotonic()-start;write(a.output,result)
    print(json.dumps({k:result.get(k) for k in ('state','phase','error','elapsed_seconds')}));return 0 if result['state']=='COMPLETE' else 2
if __name__=='__main__': raise SystemExit(main())
