import os, sys, importlib.util
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
dist.init_process_group("gloo", rank=rank, world_size=world)

_here = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(_here, "..", "..", "hybrid_moa_vl_operator_suite",
                                   "deepspeed_megatron_physics_plugin.py"))
spec = importlib.util.spec_from_file_location("plugin_mod", PKG)
m = importlib.util.module_from_spec(spec); sys.modules["plugin_mod"] = m
spec.loader.exec_module(m)

D = 8
blk = m.HybridPhysicalMoABlock(d_model=D, seed=101)
z = m.DeepSpeedZeROCompatibleWrapper(blk, zero_stage=3)
total_full = sum(p.numel() for _, p in blk.named_parameters())

z.partition_parameters_across_ranks()
live = sum(p.data.numel() for _, p in blk.named_parameters())
print("[r%d] 1.SHRINK %d -> %d (%.1f%%) status=%s" % (
    rank, total_full, live, 100.0 * live / total_full,
    sorted({p.ds_status for _, p in blk.named_parameters()})), flush=True)

torch.manual_seed(2000 + rank)
x = torch.randn(2, 8, D, dtype=torch.float64)
try:
    y = z.forward(x)
    y.sum().backward()
    ng = sum(1 for _, p in blk.named_parameters() if p.grad is not None)
    print("[r%d] 2.TRAIN OK ngrads=%d" % (rank, ng), flush=True)
except Exception as e:
    print("[r%d] 2.TRAIN FAIL %s: %s" % (rank, type(e).__name__, str(e)[:300]), flush=True)
    dist.barrier(); dist.destroy_process_group(); sys.exit(1)

local_grads = {n: p.grad.detach().clone() for n, p in blk.named_parameters() if p.grad is not None}
ref_sum = {}
for n, g in local_grads.items():
    t = g.flatten().clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    ref_sum[n] = t

shards = z.reduce_scatter_gradients()
worst, checked = 0.0, 0
for n, sh in shards.items():
    meta = z._param_partitions[n]
    chunk, total = meta["chunk_size"], meta["numel"]
    st, en = rank * chunk, min(rank * chunk + chunk, total)
    if en <= st: continue
    got = sh.flatten()[: en - st]
    expect = ref_sum[n][st:en]
    worst = max(worst, float((got.double() - expect.double()).abs().max())); checked += 1
print("[r%d] 3.REDUCE_SCATTER %d shards worst=%.3e %s" % (
    rank, checked, worst, "OK" if worst < 1e-12 else "MISMATCH"), flush=True)

a = torch.tensor([sum(float(s.sum()) for s in shards.values())], dtype=torch.float64)
b = torch.tensor([sum(float(g.sum()) for g in local_grads.values())], dtype=torch.float64)
dist.all_reduce(a, op=dist.ReduceOp.SUM); dist.all_reduce(b, op=dist.ReduceOp.SUM)
if rank == 0:
    print("  4.SUM shards=%.8f grads=%.8f ratio=%.8f" % (
        float(a[0]), float(b[0]), float(a[0]) / max(abs(float(b[0])), 1e-12)), flush=True)

dist.barrier(); dist.destroy_process_group()
print("[r%d] done" % rank, flush=True)
