import torch, time
torch.set_grad_enabled(False)
torch.set_num_threads(2)
exec(open('modelb.py').read().split('secs=float')[0].replace('dev="mps"','dev="cpu"').replace('torch.mps.synchronize()','None'))
for name,kw,res in [("B proprio-only d256L4",dict(d=256,layers=4,heads=4,use_img=False),128),
                    ("B 1x128 crop d384L6",dict(d=384,layers=6,heads=6,use_img=True,ncam=1,res=128),128),
                    ("B 2x128 crop d384L6",dict(d=384,layers=6,heads=6,use_img=True,ncam=2,res=128),128)]:
    m=ModelB(**kw).to("cpu").float().eval()
    img=torch.randn(1,3,res,res); st=torch.randn(1,32)
    bench(m,(img,st),5.0,f"CPU {name} [{nparams(m)/1e6:.1f}M]")
