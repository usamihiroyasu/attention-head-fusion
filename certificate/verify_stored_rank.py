"""Exact Fraction checker; no NumPy, models, or determinant implementation."""
import argparse,json
from fractions import Fraction as F
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--enclosure',type=Path,required=True);ap.add_argument('--preconditioner',type=Path,required=True);a=ap.parse_args()
c=json.loads(a.enclosure.read_text());r=json.loads(a.preconditioner.read_text())
A=[[(F(lo),F(hi)) for lo,hi in row] for row in c['matrix_entry_intervals']];R=[[F(x) for x in row] for row in r['R_exact']]
assert len(A)==len(R)==17 and all(len(row)==17 for row in A+R)
for i in range(17):
 e=[]
 for j in range(17):
  lo=sum(z*A[k][j][0 if z>=0 else 1] for k,z in enumerate(R[i]));hi=sum(z*A[k][j][1 if z>=0 else 0] for k,z in enumerate(R[i]));assert lo<=hi;e.append((lo,hi))
 lo,hi=e[i];diag=min(abs(lo),abs(hi)) if lo>0 or hi<0 else F(0)
 off=sum(max(abs(lo),abs(hi)) for j,(lo,hi) in enumerate(e) if j!=i)
 assert diag-off>F('0.9999999993'),i
print(json.dumps(dict(state='PASS_EXACT_RATIONAL_RANK17',rows=17,strict_dominance_margin_lower='0.9999999993',packages='standard_library_only')))
