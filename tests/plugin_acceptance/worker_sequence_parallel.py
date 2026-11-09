import os
import sys
import importlib.util

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
dist.init_process_group("nccl", rank=rank, world_size=world)
torch.cuda.set_device(rank)

_here = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(_here, "..", "..", "hybrid_moa_vl_operator_suite",
                                   "deepspeed_megatron_physics_plugin.py"))
spec = importlib.util.spec_from_file_location("plugin_mod", PKG)
m = importlib.util.module_from_spec(spec)
sys.modules["plugin_mod"] = m
spec.loader.exec_module(m)

B, D = 2, 8
L_LOCAL = 16
torch.manual_seed(1234)
# 每个 rank 拿不同数据，但参数必须完全相同（同 seed）
x_local = (torch.randn(B, L_LOCAL, D, device="cuda", dtype=torch.float64) * 0.5).requires_grad_(True)

names = None
def build():
    blk = m.HybridPhysicalMoABlock(d_model=D, seed=101).cuda()
    sp = m.MegatronSequenceParallelPlugin(blk, num_sp_ranks=world)
    return blk, sp

blk, sp = build()
print(f"[r{rank}] built | cuda={torch.cuda.current_device()}", flush=True)

ok = False
try:
    out = sp.forward_distributed_step(x_local, rank=rank)
    y, bnd = out[0], out[1]
    torch.cuda.synchronize()
    print(f"[r{rank}] FWD ok y={tuple(y.shape)} q[0,:2]={bnd['q_bound'][0,:2].tolist()}", flush=True)
    ok = True
except Exception as e:  # noqa: BLE001
    print(f"[r{rank}] FWD FAIL {type(e).__name__}: {str(e)[:400]}", flush=True)

if ok:
    try:
        # 每个 rank 用不同权重求和，确保下游损失真的进入上游梯度
        w = torch.arange(1, y.numel() + 1, device="cuda", dtype=torch.float64).reshape(y.shape)
        (y * w).sum().backward()
        torch.cuda.synchronize()
        g = x_local.grad.detach().clone()
        print(f"[r{rank}] BWD ok |dx|={float(g.norm()):.10f}", flush=True)
        # 汇总所有 rank 的梯度范数，便于跨卡比对
        vec = torch.tensor([float(g.norm()), float(g.abs().max())], device="cuda", dtype=torch.float64)
        gathered = [torch.zeros_like(vec) for _ in range(world)]
        dist.all_gather(gathered, vec)
        if rank == 0:
            for r, t in enumerate(gathered):
                print(f"    rank{r}: |dx|={float(t[0]):.10f} max|dx|={float(t[1]):.10f}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[r{rank}] BWD FAIL {type(e).__name__}: {str(e)[:500]}", flush=True)

dist.barrier()
dist.destroy_process_group()
print(f"[r{rank}] done", flush=True)
