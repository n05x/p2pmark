# p2pmark

GPU peer-to-peer interconnect benchmark. Measures PCIe P2P bandwidth across
multi-GPU systems and produces two scores:

- **PCIe Link Score** — how close each GPU's P2P bandwidth gets to the
  theoretical PCIe 5.0 x16 limit (63 GB/s).
- **Dense Interconnect Score** — ratio of actual aggregate bandwidth (all GPUs
  reading from all peers simultaneously) to the ideal sum of isolated per-GPU
  bandwidths. 1.0 = perfect full-mesh with no contention, lower values indicate
  shared fabric bottlenecks (e.g. multi-switch PCIe topologies).

## Tests

1. **Sequential NxN** — baseline per-link bandwidth, one transfer at a time.
2. **Topology probe** — staggered reads by peer distance, reveals switch topology.
3. **Single reader** — one GPU reads from all peers concurrently (max inbound BW per GPU).
4. **All GPUs concurrent** — every GPU reads from all peers at once (system-wide throughput).

## Build & Run

```
make
./p2pmark
```

Requires CUDA toolkit and a C++20 compiler (`nvcc` with `-std=c++20`).

## Example Output

```
===========================================================
  PCIe LINK SCORE:           0.86
  (54.30 GB/s avg  /  63.0 GB/s PCIe 5.0 x16 theoretical)

  DENSE INTERCONNECT SCORE:  0.44
  (189.13 GB/s measured  /  434.71 GB/s ideal)

  1.00 = perfect, 0.00 = none
===========================================================
```
