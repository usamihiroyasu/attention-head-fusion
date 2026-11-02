from pathlib import Path
from itertools import product
import numpy as np
import json, os, time, hashlib, platform
P=Path(__file__).parent; L=P/'owned_work.lock'; fd=os.open(L,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600); inode=os.fstat(fd).st_ino
os.write(fd,json.dumps(dict(pid=os.getpid(),inode=inode)).encode()); os.fsync(fd)
def kernel(q,K,V):
    z=np.einsum('nhb,hmb->nhm',q,K,optimize=True); z-=z.max(axis=2,keepdims=True)
    np.exp(z,out=z); z/=z.sum(axis=2,keepdims=True)
    return np.einsum('nhm,hm->n',z,V,optimize=True)
def arrays(d,m):
    b={'native_cube':d,'block_b2':2,'coordinate_b1':1,'mean_fixed2':d}[m]; H=d//b
    K=np.tile(np.array(list(product((0.,1.),repeat=b)),dtype=np.float64)[None,:,:],(H,1,1)).copy() if m!='mean_fixed2' else np.stack((np.zeros(d),np.ones(d)/d))[None,:,:].copy()
    V=K.sum(axis=2).copy() if m!='mean_fixed2' else np.array([[0.,float(d)]],dtype=np.float64)
    return K,V,H,b
rows=[dict(d=d,R=R,method=m,status='PENDING') for d in [4,8,12] for R in [.25,1.,4.] for m in ['native_cube','block_b2','coordinate_b1','mean_fixed2']]
out=dict(status='RUNNING',numpy=np.__version__,platform=platform.platform(),pid=os.getpid(),dtype='float64',rows=rows,input_hashes={},code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),spec_sha256=hashlib.sha256((P/'SPEC.json').read_bytes()).hexdigest(),threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']})
def save():
    t=P/'RESULTS.tmp'; t.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n'); os.replace(t,P/'RESULTS.json')
try:
    save(); inputs={}
    for d in [4,8,12]:
        Z=np.random.Generator(np.random.PCG64(20261101)).uniform(-1.,1.,(512,d)); C=np.array(list(product((-1.,1.),repeat=d)),dtype=np.float64)
        out['input_hashes'][str(d)]=dict(Z=hashlib.sha256(Z.tobytes()).hexdigest(),corners=hashlib.sha256(C.tobytes()).hexdigest())
        for R in [.25,1.,4.]:
            q=np.concatenate((Z,C),axis=0)*R; inputs[d,R]=(q,(1./(1.+np.exp(-q))).sum(axis=1))
    for row in rows:
        try:
            d,R,m=row['d'],row['R'],row['method']; q,target=inputs[d,R]; K,V,H,b=arrays(d,m); qb=q.reshape(len(q),H,b)
            assert K.dtype==V.dtype==np.float64 and K.flags.owndata and V.flags.owndata
            kernel(qb,K,V); trials=[]
            for _ in range(5):
                t=time.perf_counter(); pred=kernel(qb,K,V); trials.append(time.perf_counter()-t)
            if not np.isfinite(pred).all(): raise FloatingPointError('Nonfinite output')
            err=np.abs(pred-target); mx=float(err.max()); mean=float(err.mean())
            row.update(status='COMPLETE',queries=len(q),uniform_queries=512,corner_queries=2**d,max_abs_error=mx,mean_abs_error=mean,max_abs_error_div_d=mx/d,mean_abs_error_div_d=mean/d,numeric_identity_pass=bool(mx<=1e-10) if m!='mean_fixed2' else None,head_atom_terms=int(H*K.shape[1]),heads=H,atoms_per_head=K.shape[1],block_dim=b,K_shape=list(K.shape),V_shape=list(V.shape),K_bytes=K.nbytes,V_bytes=V.nbytes,KV_bytes=K.nbytes+V.nbytes,logit_cells=len(q)*H*K.shape[1],logit_activation_proxy_bytes=8*len(q)*H*K.shape[1],activation_proxy_scope='one FP64 logits array only; not peakRSS or totalpeak',kernel_seconds=trials,median_kernel_seconds=float(np.median(trials)))
        except Exception as e: row.update(status='FAILED',error=type(e).__name__+': '+str(e))
        save()
    out['status']='COMPLETE' if all(r['status']=='COMPLETE' for r in rows) else 'COMPLETED_WITH_FAILURES'; save()
finally:
    try: os.close(fd)
    finally:
        if L.exists() and L.stat().st_ino==inode and json.loads(L.read_text())['pid']==os.getpid(): L.unlink()
