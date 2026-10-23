import sys; sys.path.insert(0,'/home/user/webapp/sdk')
import numpy as np, cv2, subprocess
from PIL import Image
from mograph.lineanim.panel_anim import *
im=np.array(Image.open('/tmp/sb/storyboard.png').convert('L'))
pan=im[666:896, 3:502]
A=tonal(pan,s=4); Amask=soft_lines(pan,s=4,lo=40,rng=30)
H,W=A.shape
order=geodesic_order(Amask,seed=(0,int(0.83*H)))
OW,OH=1920,1080; fps=30; T=7.0
girl_piv=(918,436); boy_piv=(614,332); girl_head=(930,320); boy_head=(640,190)
flame=(1360,600)
cut=np.zeros_like(A); cut[int(560):int(620),1320:1400]=1  # flame region
rng=np.random.default_rng(1)
proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','gray','-s',f'{OW}x{OH}','-r',str(fps),'-i','-',
  '-c:v','libx264','-pix_fmt','yuv420p','-crf','16','/tmp/sb/panel_anim.mp4'],stdin=subprocess.PIPE)
for f in range(int(T*fps)):
    t=f/fps
    # draw-on 0..2.6s
    prog=ease(t/2.6)*1.08
    reveal=np.clip((prog-order)/0.04,0,1)
    a=A*reveal
    # acting after 2.4s
    k=ease((t-2.6)/1.4)
    breath=np.sin(t*2*np.pi/3.2)*0.004
    rigs=[ (girl_piv, np.radians(-5)*k + breath*2, 150, 1.0, girl_head,(0,0)),
           (boy_piv,  np.radians(3)*k  - breath*2, 140, 1.0, boy_head,(0,0)),
           ((W/2,H), breath, 900,1.0,(W/2,H*0.7),(0,0)) ]
    a=warp(a,rigs)
    # flame flicker
    if t>2.0:
        fl=1+0.08*np.sin(t*23)+0.05*np.sin(t*37+1)
        ys,xs=flame[1],flame[0]
        sub=a[ys-40:ys+20, xs-30:xs+30].copy()
        sub=cv2.resize(sub,None,fx=1,fy=fl)
        sh=sub.shape[0]; a[ys+20-min(sh,60):ys+20, xs-30:xs+30]=np.maximum(a[ys+20-min(sh,60):ys+20, xs-30:xs+30]*0.0, sub[-min(sh,60):])
        glow=np.zeros_like(a); cv2.circle(glow,(xs,ys-12),60,1,-1); glow=cv2.GaussianBlur(glow,(0,0),40)*0.10*fl*ease(t-2)
        a=np.clip(a+glow,0,1)
    # camera push
    z=1.0+0.06*ease(t/T); cx,cy=W*0.47,H*0.5
    M=np.float32([[z*OW/W*0.98,0,0],[0,z*OW/W*0.98,0]])
    sc=z*OW/W
    M=np.float32([[sc,0,OW/2-cx*sc],[0,sc,OH/2-cy*sc+40]])
    out=cv2.warpAffine(a,M,(OW,OH),flags=cv2.INTER_CUBIC)
    out=out*0.93+rng.normal(0,0.012,out.shape)
    # text fade 5.2s+
    fr=np.clip(out*255+10,0,255).astype(np.uint8)
    if t>5.0:
        al=ease((t-5.0)/1.0)
        txt=np.zeros_like(fr); cv2.putText(txt,'Happy Birthday',(1340,330),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1.6,255,2,cv2.LINE_AA)
        cv2.putText(txt,'my little sister',(1380,390),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1.3,255,2,cv2.LINE_AA)
        # write-on wipe
        wx=int(1320+ (1750-1320)*al); txt[:, wx:]=0
        fr=np.maximum(fr,(txt*0.9).astype(np.uint8))
    proc.stdin.write(fr.tobytes())
    if f in (45,80,150,205): cv2.imwrite(f'/tmp/sb/fr/f{f}.png',fr)
proc.stdin.close(); proc.wait(); print('ok')
