"""Root corrected variable projection. Generated draft had a missing 1/N loss factor."""
import time
import torch

def fit_batch(x,y,K0,b0,steps=512,lam=1e-6,lr=.03,Dtotal=None,progress=None):
    assert x.dtype==y.dtype==K0.dtype==b0.dtype==torch.float64
    B,N,d=x.shape;M=K0.shape[1];Dtotal=y.shape[-1] if Dtotal is None else Dtotal
    assert y.shape[:2]==(B,N) and K0.shape==(B,M,d) and b0.shape==(B,M)
    K=K0.detach().clone().requires_grad_(True);b=b0.detach().clone().requires_grad_(True)
    opt=torch.optim.Adam([K,b],lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0)
    eye=torch.eye(M,dtype=x.dtype,device=x.device)[None];trace={};worst=0.;start=time.monotonic()
    with torch.enable_grad():
        for step in range(steps+1):
            A=torch.softmax(x@K.transpose(-1,-2)+b[:,None,:],-1)
            with torch.no_grad():
                G=A.transpose(-1,-2)@A+N*lam*eye;rhs=A.transpose(-1,-2)@y
                L,info=torch.linalg.cholesky_ex(G);assert bool((info==0).all()),'Cholesky'
                V=torch.cholesky_solve(rhs,L)
                rel=torch.linalg.vector_norm(G@V-rhs,dim=(1,2))/torch.linalg.vector_norm(rhs,dim=(1,2)).clamp_min(1)
                assert bool(torch.isfinite(rel).all()) and float(rel.max())<=1e-8,'stationarity'
                worst=max(worst,float(rel.max()))
            residual=((A@V-y)**2).sum((1,2))/(N*Dtotal)
            ridge=lam*(V**2).sum((1,2))/Dtotal;obj=residual+ridge
            assert bool(torch.isfinite(obj).all()),'objective finite'
            if step in (0,128,256,384,512) or step==steps:
                trace[str(step)]={'objective':obj.detach().cpu().tolist(),'residual':residual.detach().cpu().tolist(),'ridge':ridge.cpu().tolist()}
                if progress is not None: progress(step,trace[str(step)])
            if step==steps: break
            opt.zero_grad(set_to_none=True);obj.sum().backward()
            assert bool(torch.isfinite(K.grad).all()) and bool(torch.isfinite(b.grad).all()),'gradient finite'
            opt.step()
            with torch.no_grad(): K-=K.mean(1,keepdim=True);b-=b.mean(1,keepdim=True)
    with torch.no_grad():
        sv=torch.linalg.svdvals(A);rank=(sv>sv[:,:1]*1e-6).sum(-1)
        eigen=torch.linalg.eigvalsh(G);cond=eigen[:,-1]/eigen[:,0]
    return dict(K=K.detach(),b=b.detach(),V=V.detach(),diag=dict(trace=trace,total_updates=steps,solve_rounds=steps+1,
        stationarity_max=worst,final_design_rank=rank.cpu().tolist(),final_ridge_condition=cond.cpu().tolist(),seconds=time.monotonic()-start))

def smoke_test():
    x=(torch.arange(40,dtype=torch.float64).reshape(2,5,4)%11-5)/7
    y=(torch.arange(30,dtype=torch.float64).reshape(2,5,3)%13-6)/5
    K=(torch.arange(24,dtype=torch.float64).reshape(2,3,4)%7-3)/9
    b=torch.tensor([[.1,-.2,.1],[.2,.1,-.3]],dtype=torch.float64);lam=.003;N=5;D=6
    def conditional(k,bb,detached=False):
        A=torch.softmax(x@k.transpose(-1,-2)+bb[:,None,:],-1)
        G=A.transpose(-1,-2)@A+N*lam*torch.eye(3,dtype=torch.float64)
        V=torch.linalg.solve(G,A.transpose(-1,-2)@y)
        if detached: V=V.detach()
        return (((A@V-y)**2).sum()/(N*D)+lam*(V**2).sum()/D)
    ka=K.clone().requires_grad_(True);ba=b.clone().requires_grad_(True)
    conditional(ka,ba,True).backward();errors=[];eps=1e-5
    for kind,idx in [('K',(0,0,0)),('K',(1,2,3)),('b',(0,1)),('b',(1,2))]:
        kp=K.clone();km=K.clone();bp=b.clone();bm=b.clone()
        if kind=='K':kp[idx]+=eps;km[idx]-=eps;ag=float(ka.grad[idx])
        else:bp[idx]+=eps;bm[idx]-=eps;ag=float(ba.grad[idx])
        fd=float((conditional(kp,bp)-conditional(km,bm))/(2*eps));errors.append(abs(fd-ag));assert abs(fd-ag)<2e-6
    fit=fit_batch(x,y,K,b,steps=2,lam=lam,lr=.01,Dtotal=D)
    assert fit['diag']['solve_rounds']==3 and fit['diag']['stationarity_max']<1e-8
    assert float(fit['K'].mean(1).abs().max())<1e-14 and float(fit['b'].mean(1).abs().max())<1e-14
    return dict(state='PASS',central_FD_checks=4,max_abs_FD_error=max(errors),stationarity=fit['diag']['stationarity_max'])
