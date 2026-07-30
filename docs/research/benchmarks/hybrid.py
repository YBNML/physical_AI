import torch, time, sys, numpy as np
torch.set_grad_enabled(False)

def bench(fn, secs=4.0, warm=5):
    for _ in range(warm): fn()
    try: torch.mps.synchronize()
    except Exception: pass
    lat=[]; t_end=time.perf_counter()+secs
    while time.perf_counter()<t_end:
        t0=time.perf_counter(); fn()
        try: torch.mps.synchronize()
        except Exception: pass
        lat.append((time.perf_counter()-t0)*1e3)
    lat.sort(); p=lambda q: lat[min(len(lat)-1,int(q*len(lat)))]
    return len(lat), p(.5), p(.95), p(.99), lat[-1]

def show(name, r):
    n,p50,p95,p99,mx = r
    print(f"{name:44s} n={n:6d} p50={p50:8.2f}ms p95={p95:8.2f}ms p99={p99:8.2f}ms max={mx:8.2f}ms -> {1000/p50:7.1f} Hz")

dev="mps"
# ---- Model A part 1: SigLIP-B/16 vision tower @ 512px = 1024 patches (SmolVLM2 default)
enc=torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(768,12,3072,batch_first=True,norm_first=True),12).to(dev).half().eval()
show("A.vision SigLIP-B/16 @512 (1024 tok)", bench(lambda: enc(torch.randn(1,1024,768,device=dev,dtype=torch.half))))
show("A.vision SigLIP-B/16 @224 (196 tok)",  bench(lambda: enc(torch.randn(1,197,768,device=dev,dtype=torch.half))))
show("A.vision SigLIP-B/16 @224 x2 cams",    bench(lambda: enc(torch.randn(2,197,768,device=dev,dtype=torch.half))))

# ---- Model A part 2: LM prefill, SmolVLM2-500M-ish trunk: 16 layers d=960, ~256 tokens
lm=torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(960,15,2560,batch_first=True,norm_first=True),16).to(dev).half().eval()
show("A.LM 16L d=960 prefill 256 tok",  bench(lambda: lm(torch.randn(1,256,960,device=dev,dtype=torch.half))))
show("A.LM 16L d=960 prefill 640 tok",  bench(lambda: lm(torch.randn(1,640,960,device=dev,dtype=torch.half))))

# ---- Model A part 3: flow-matching action expert ~100M, d=512, 12L, 24 tok, K steps
ae=torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(512,8,2048,batch_first=True,norm_first=True),12).to(dev).half().eval()
def ae_run(K):
    x=torch.randn(1,24,512,device=dev,dtype=torch.half)
    for _ in range(K): x=ae(x)
for K in (4,10):
    show(f"A.action-expert 100M, {K} flow steps", bench(lambda K=K: ae_run(K)))

# ---- Bigger cerebrum option: Qwen2.5-VL-3B-ish trunk 36L d=2048, 900 tok
big=torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(2048,16,11008,batch_first=True,norm_first=True),36).to(dev).half().eval()
show("A-XL 3B trunk 36L d=2048 prefill 900tok", bench(lambda: big(torch.randn(1,900,2048,device=dev,dtype=torch.half)), secs=6))
del big; torch.mps.empty_cache()

# ---- Residual net: 2-layer GRU h=128 on 44-dim proprio+wrench, CPU, batch 1, 1 step
class Res(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.g=torch.nn.GRU(44,128,2,batch_first=True); s.h=torch.nn.Sequential(torch.nn.Linear(128,128),torch.nn.GELU(),torch.nn.Linear(128,6))
    def forward(s,x,h): y,h2=s.g(x,h); return s.h(y[:,-1]),h2
res=Res().eval()
print(f"residual params = {sum(p.numel() for p in res.parameters())/1e6:.3f} M")
xh=(torch.randn(1,1,44), torch.zeros(2,1,128))
torch.set_num_threads(1)
show("B.residual GRU 0.2M, CPU 1 thread, 1 step", bench(lambda: res(*xh), secs=3))

# ---- Damped-least-squares IK step, numpy, 7-DoF
J=np.random.randn(6,7); e=np.random.randn(6); lam=0.05
def dls():
    JJt=J@J.T+lam*lam*np.eye(6)
    return J.T@np.linalg.solve(JJt,e)
t0=time.perf_counter(); N=200000
for _ in range(N): dls()
print(f"DLS IK step (numpy 6x7): {(time.perf_counter()-t0)/N*1e6:.2f} us  -> {N/(time.perf_counter()-t0)/1000:.0f}k Hz")
