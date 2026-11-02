"""One fixed17x17 native minor; stdlib-only directed Decimal interval proof."""
import argparse,ast,decimal as dec,hashlib,json,math,os,struct,time,zipfile
from pathlib import Path
D=dec.Decimal
F=dec.Context(prec=80,rounding=dec.ROUND_FLOOR)
C=dec.Context(prec=80,rounding=dec.ROUND_CEILING)
H=dec.Context(prec=80,rounding=dec.ROUND_HALF_EVEN)
ZERO=(D(0),D(0));ONE=(D(1),D(1))
class Hold(Exception):pass
def add(x,y):return F.add(x[0],y[0]),C.add(x[1],y[1])
def sub(x,y):return F.subtract(x[0],y[1]),C.subtract(x[1],y[0])
def mul(x,y):return min(F.multiply(a,b) for a in x for b in y),max(C.multiply(a,b) for a in x for b in y)
def div(x,y):
    if y[0]<=0<=y[1]:raise Hold('division_interval_contains_zero')
    return min(F.divide(a,b) for a in x for b in y),max(C.divide(a,b) for a in x for b in y)
def exp(x):
    return H.exp(x[0]).next_minus(context=H),H.exp(x[1]).next_plus(context=H)
def neg(x):return x[1].copy_negate(),x[0].copy_negate()
def isum(xs):
    z=ZERO
    for x in xs:z=add(z,x)
    return z
def pair(x):return [str(x[0]),str(x[1])]
def midpoint_abs(x):return H.divide(H.add(x[0],x[1]),D(2)).copy_abs()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read_f32(z,name,shape):
    data=z.read(name+'.npy');assert data[:6]==b'\x93NUMPY'
    version=data[6:8];assert version in (b'\x01\x00',b'\x02\x00')
    size=2 if version==b'\x01\x00' else 4;length=int.from_bytes(data[8:8+size],'little');assert length<4096
    hdr=ast.literal_eval(data[8+size:8+size+length].decode('latin1').strip())
    assert hdr['descr']=='<f4' and not hdr['fortran_order'] and tuple(hdr['shape'])==shape
    raw=data[8+size+length:];assert len(raw)==4*math.prod(shape)
    out=[]
    for value, in struct.iter_unpack('<f',raw):
        assert math.isfinite(value);v=D.from_float(value);out.append((v,v))
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-npz',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    start=time.monotonic();r=dict(state='RUNNING',precision=80,context_index=0,query_rows=list(range(17)),columns=['constant1']+['head0coord'+str(i) for i in range(15)]+['head1coord0'],source_sha256=digest(Path(__file__)),pivots=[])
    def save():
        p=a.output.with_suffix('.tmp');p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');os.replace(p,a.output)
    try:
        p=a.input_npz;assert digest(p)=="5bf2b720d26de4a3911e2f67086492969d793f91e6ea2484892223f1f1cdfa1d"
        r.update(input_npz_sha256=digest(p),native_positions=list(range(16,33)),phase='matrix')
        with zipfile.ZipFile(p) as z:q=read_f32(z,'qearly',(17,2,64));k=read_f32(z,'kprefix',(2,16,64));v=read_f32(z,'vprefix',(2,16,64))
        matrix=[];logits=[]
        for t in range(17):
            outputs=[];head_logits=[]
            for h in range(2):
                ell=[isum(mul(q[(t*2+h)*64+j],k[(h*16+m)*64+j]) for j in range(64)) for m in range(16)]
                ee=[exp(x) for x in ell];den=isum(ee)
                if den[0]<=0:raise Hold('nonpositive_denominator_bound')
                for coordinate in (range(15) if h==0 else [0]):outputs.append(div(isum(mul(ee[m],v[(h*16+m)*64+coordinate]) for m in range(16)),den))
                head_logits.append([pair(x) for x in ell])
            matrix.append([ONE]+outputs);logits.append(head_logits)
        r.update(matrix_entry_intervals=[[pair(x) for x in row] for row in matrix],logit_intervals=logits,phase='elimination');save()
        A=[row.copy() for row in matrix];det=ONE;swaps=0
        for j in range(17):
            pivotrow=max(range(j,17),key=lambda i:(midpoint_abs(A[i][j]),-i))
            if pivotrow!=j:A[j],A[pivotrow]=A[pivotrow],A[j];swaps+=1
            pivot=A[j][j];r['pivots'].append(dict(column=j,selected_current_row=pivotrow,interval=pair(pivot)))
            if pivot[0]<=0<=pivot[1]:raise Hold('pivot_'+str(j)+'_contains_zero')
            det=mul(det,pivot)
            for i in range(j+1,17):
                factor=div(A[i][j],pivot)
                for c in range(j+1,17):A[i][c]=sub(A[i][c],mul(factor,A[j][c]))
                A[i][j]=ZERO
        if swaps%2:det=neg(det)
        if not all(x.is_finite() for x in det):raise Hold('nonfinite_determinant')
        r.update(determinant_interval=pair(det),row_swaps=swaps,phase='done',state='CERTIFIED_NONZERO_MINOR' if not(det[0]<=0<=det[1]) else 'HOLD_ZERO_IN_DETERMINANT_INTERVAL')
    except Exception as e:r.update(state='HOLD',error=type(e).__name__+':'+str(e))
    r['elapsed_seconds']=time.monotonic()-start;save();print(json.dumps({k:r.get(k) for k in ['state','error','determinant_interval','elapsed_seconds']}));return 0 if r['state']=='CERTIFIED_NONZERO_MINOR' else 2
if __name__=='__main__':raise SystemExit(main())
