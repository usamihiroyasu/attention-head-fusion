from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
D=Path(__file__).resolve().parent
im=Image.new('RGB',(1000,500),'white');d=ImageDraw.Draw(im)
ink='#17313C';teal='#008C9E';gold='#D17A1B'
def font(n,bold=False):
 p=Path('/System/Library/Fonts/Supplemental')/('Arial Bold.ttf' if bold else 'Arial.ttf')
 if not p.exists():p=Path('/usr/share/fonts/truetype/dejavu')/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')
 return ImageFont.truetype(str(p),n)
def text(x,y,s,n=20,c=ink,anchor='mm',bold=False):d.text((x,y),s,font=font(n,bold),fill=c,anchor=anchor)
text(215,36,'12 separately normalized heads',23,bold=True)
text(745,36,'One exact single-head readout',23,bold=True)
left=0
for j in range(12):
 y=139+j*18;text(134,y,f'H{j+1}',15,anchor='rm')
 for x in [188,242]:d.ellipse((x-5,y-5,x+5,y+5),fill=teal);left+=1
right=0
for row in range(64):
 for col in range(64):
  x=588+col*5;y=77+row*5;d.ellipse((x-1.2,y-1.2,x+1.2,y+1.2),fill=gold);right+=1
assert (left,right)==(24,4096)
d.line((356,238,494,238),fill=gold,width=4);d.polygon([(512,238),(491,228),(491,248)],fill=gold)
text(432,210,'exact fusion',17)
text(215,435,'24',54,teal,bold=True);text(745,435,'4096',54,gold,bold=True)
text(215,481,'head–atom terms',21);text(745,481,'single-head atoms',21)
im.save(D/'atom_cost.png')
