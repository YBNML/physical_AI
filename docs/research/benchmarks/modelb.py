import torch, time, sys, math
torch.set_grad_enabled(False); dev="mps"

def stats(name, lat):
    lat.sort(); p=lambda q: lat[min(len(lat)-1,int(q*len(lat)))]
    print(f"{name:34s} n={len(lat):5d} p50={p(.5):7.2f} p95={p(.95):7.2f} p99={p(.99):7.2f} max={lat[-1]:7.2f} ms -> {1000/p(.5):6.1f} Hz", flush=True)

class Enc(torch.nn.Module):  # tiny ViT-ish encoder over a low-res crop
    def __init__(s, d, layers, heads, tok):
        super().__init__()
        s.tok=tok
        s.patch=torch.nn.Conv2d(3,d,16,16)
        s.tr=torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d,heads,4*d,batch_first=True,norm_first=True),layers)
    def forward(s,img):
        z=s.patch(img).flatten(2).transpose(1,2)
        return s.tr(z)

class ModelB(torch.nn.Module):
    def __init__(s, d=384, layers=6, heads=6, use_img=True, ncam=1, res=128, H=16, adim=10, ctx=32):
        super().__init__()
        s.use_img=use_img; s.ncam=ncam; s.res=res; s.H=H; s.adim=adim
        if use_img: s.enc=Enc(d,layers,heads,(res//16)**2)
        s.state=torch.nn.Linear(ctx,d)
        s.dec=torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d,heads,4*d,batch_first=True,norm_first=True),4)
        s.q=torch.nn.Parameter(torch.randn(1,H,d))
        s.out=torch.nn.Linear(d,adim)
    def forward(s,img,st):
        toks=[s.state(st).unsqueeze(1)]
        if s.use_img:
            for i in range(s.ncam): toks.append(s.enc(img))
        toks.append(s.q.expand(st.shape[0],-1,-1))
        z=s.dec(torch.cat(toks,1))
        return s.out(z[:,-s.H:])

def bench(m, args, secs, name, sleep=0.0):
    for _ in range(8): m(*args); torch.mps.synchronize()
    lat=[]; t=time.perf_counter()+secs
    while time.perf_counter()<t:
        t0=time.perf_counter(); m(*args); torch.mps.synchronize(); lat.append((time.perf_counter()-t0)*1e3)
        if sleep: time.sleep(sleep)
    stats(name, lat)

def nparams(m): return sum(p.numel() for p in m.parameters())

secs=float(sys.argv[1]) if len(sys.argv)>1 else 6.0

cfgs=[
 ("B: proprio-only (no img), d256 L4",      dict(d=256,layers=4,heads=4,use_img=False)),
 ("B: 1x 128px crop, d384 L6",              dict(d=384,layers=6,heads=6,use_img=True,ncam=1,res=128)),
 ("B: 2x 128px crop, d384 L6",              dict(d=384,layers=6,heads=6,use_img=True,ncam=2,res=128)),
 ("B: 1x 224px, d384 L6 (full RGB)",        dict(d=384,layers=6,heads=6,use_img=True,ncam=1,res=224)),
]
for name,kw in cfgs:
    m=ModelB(**kw).to(dev).half().eval()
    res=kw.get("res",128)
    img=torch.randn(1,3,res,res,device=dev,dtype=torch.half)
    st=torch.randn(1,32,device=dev,dtype=torch.half)
    bench(m,(img,st),secs,f"{name} [{nparams(m)/1e6:.1f}M]")
