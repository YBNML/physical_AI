import time, jax, jax.numpy as jp, mujoco
from mujoco import mjx
m = mujoco.MjModel.from_xml_path('mm/aloha/scene_objs.xml')
try:
    mx = mjx.put_model(m)
except Exception as e:
    print("put_model failed:", type(e).__name__, str(e)[:400]); raise SystemExit
B=32
dx = jax.vmap(lambda _: mjx.make_data(mx))(jp.arange(B))
step = jax.jit(jax.vmap(mjx.step, in_axes=(None,0)))
t=time.perf_counter(); dx=step(mx,dx); jax.block_until_ready(dx); print(f"compile {time.perf_counter()-t:.1f}s")
N=50
t=time.perf_counter()
for _ in range(N): dx=step(mx,dx)
jax.block_until_ready(dx); dt=time.perf_counter()-t
print(f"MJX CPU batch={B}: {N*B/dt:,.0f} total steps/s ({N*B} steps in {dt:.2f}s)")
