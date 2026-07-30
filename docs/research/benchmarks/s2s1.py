import torch, time, sys, math
torch.set_grad_enabled(False); dev="mps"
def enc(d,heads,mlp,depth):
    l=torch.nn.TransformerEncoderLayer(d,heads,mlp,batch_first=True,norm_first=True,activation="gelu")
    return torch.nn.TransformerEncoder(l,depth).to(dev).half().eval()
def dec(d,heads,mlp,depth):
    l=torch.nn.TransformerDecoderLayer(d,heads,mlp,batch_first=True,norm_first=True,activation="gelu")
    return torch.nn.TransformerDecoder(l,depth).to(dev).half().eval()
def bench(name,fn,secs=12.0):
    for _ in range(25): fn()
    torch.mps.synchronize()
    lat=[];t=time.perf_counter()+secs
    while time.perf_counter()<t:
        t0=time.perf_counter(); fn(); torch.mps.synchronize(); lat.append((time.perf_counter()-t0)*1e3)
    lat.sort(); p=lambda q: lat[min(len(lat)-1,int(q*len(lat)))]
    print(f"{name:52s} n={len(lat):5d} p50={p(.5):8.2f} p95={p(.95):8.2f} p99={p(.99):8.2f} ms -> {1000/p(.5):7.1f} Hz")
    return p(.5)

which=sys.argv[1] if len(sys.argv)>1 else "all"
tot=0
if which in ("all","s2"):
    # SigLIP-so400m-class: d=1152, 27 layers, mlp 4304, 16 heads, 256 patches @224 w/ pixel shuffle
    v=enc(1152,16,4304,27); xv=torch.randn(1,256,1152,device=dev,dtype=torch.half)
    tot+=bench("SigLIP-so400m-class vis enc, 256 tok, 1 cam", lambda: v(xv))
    xv2=torch.randn(2,256,1152,device=dev,dtype=torch.half)
    tot+=0; bench("  same, batch=2 (2 cameras in one call)", lambda: v(xv2))
    del v,xv,xv2; torch.mps.empty_cache()
    # Gemma-2B-class LLM prefill: d=2048, 18 layers, mlp 16384, 8 heads, ~560 tokens (2x256 img + lang + state)
    L=enc(2048,8,16384,18); xl=torch.randn(1,560,2048,device=dev,dtype=torch.half)
    tot+=bench("Gemma-2B-class prefill, 560 tok (KV build)", lambda: L(xl))
    del L,xl; torch.mps.empty_cache()
if which in ("all","s1"):
    # action expert: d=1024, 18 layers, mlp 4096, 8 heads, 51 tokens, cross-attn to 560-tok prefix
    A=dec(1024,8,4096,18)
    xa=torch.randn(1,51,1024,device=dev,dtype=torch.half); mem=torch.randn(1,560,1024,device=dev,dtype=torch.half)
    s=bench("Action expert 300M, 1 flow step (H=50 chunk)", lambda: A(xa,mem))
    print(f"   -> 4 flow steps = {4*s:.1f} ms ({1000/(4*s):.1f} Hz chunk rate)")
    print(f"   -> 10 flow steps= {10*s:.1f} ms ({1000/(10*s):.1f} Hz chunk rate)")
    del A; torch.mps.empty_cache()
    # tiny proprio-only reflex net
    M=torch.nn.Sequential(*[torch.nn.Sequential(torch.nn.Linear(512,2048),torch.nn.GELU(),torch.nn.Linear(2048,512)) for _ in range(4)]).to(dev).half().eval()
    xm=torch.randn(1,1,512,device=dev,dtype=torch.half)
    bench("Tiny 12M proprio MLP (reflex/residual head)", lambda: M(xm))
