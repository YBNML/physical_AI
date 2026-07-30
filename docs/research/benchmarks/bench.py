import torch, time, sys, statistics as st
torch.set_grad_enabled(False)
dev="mps"
role=sys.argv[1]

def mk(d,layers):
    return torch.nn.Sequential(*[torch.nn.Sequential(torch.nn.Linear(d,4*d),torch.nn.GELU(),torch.nn.Linear(4*d,d)) for _ in range(layers)]).to(dev).half()

if role=="big":   # cerebrum-ish: big transformer-scale matmuls on a long visual token seq
    m=mk(2048,24); x=torch.randn(1,1024,2048,device=dev,dtype=torch.half)
else:             # cerebellum-ish: small fast policy
    m=mk(512,6);   x=torch.randn(1,64,512,device=dev,dtype=torch.half)

for _ in range(5): m(x); torch.mps.synchronize()

lat=[]
t_end=time.perf_counter()+float(sys.argv[2])
while time.perf_counter()<t_end:
    t0=time.perf_counter(); m(x); torch.mps.synchronize(); lat.append((time.perf_counter()-t0)*1e3)
    if role=="small": time.sleep(0.005)   # target ~ up to 200Hz duty
lat.sort()
p=lambda q: lat[min(len(lat)-1,int(q*len(lat)))]
print(f"{role:5s} n={len(lat):5d} p50={p(.5):8.2f}ms p95={p(.95):8.2f}ms p99={p(.99):8.2f}ms max={lat[-1]:8.2f}ms")
