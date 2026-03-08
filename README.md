# p2pmark

GPU peer-to-peer interconnect benchmark. Measures PCIe/NVLink P2P bandwidth and
latency across multi-GPU systems.

## Modes

### Bandwidth (default)

Measures P2P bandwidth using 64 MB `cudaMemcpyPeerAsync` transfers and produces
two scores:

- **PCIe Link Score** — how close each GPU's P2P bandwidth gets to the
  theoretical PCIe 5.0 x16 limit (63 GB/s).
- **Dense Interconnect Score** — ratio of actual aggregate bandwidth (all GPUs
  reading from all peers simultaneously) to the ideal sum of isolated per-GPU
  bandwidths. 1.0 = perfect full-mesh with no contention, lower values indicate
  shared fabric bottlenecks (e.g. multi-switch PCIe topologies).

### Latency (`--latency`)

Measures P2P latency using a custom CUDA kernel that performs direct remote
reads (single cacheline, 128 bytes) via P2P-mapped pointers. Single-thread, single-block launches isolate the
true hardware transfer latency from API overhead.

Produces a **Latency Score** — ratio of average 1:1 (isolated) latency to
average all-GPUs-concurrent latency. 1.0 = no degradation under full load.

## Tests

Both modes run the same four test patterns:

1. **Sequential NxN** — baseline per-link measurement, one transfer at a time.
2. **Topology probe** — staggered reads by peer distance, reveals switch topology.
3. **Single reader** — one GPU reads from all peers concurrently.
4. **All GPUs concurrent** — every GPU reads from all peers at once.

## Build & Run

```
make
./p2pmark            # bandwidth mode
./p2pmark --latency  # latency mode
```

Requires CUDA toolkit and a C++20 compiler (`nvcc` with `-std=c++20`).

## Example Output

### Bandwidth

```
===========================================================
  PCIe LINK SCORE:           0.86
  (54.30 GB/s avg  /  63.0 GB/s PCIe 5.0 x16 theoretical)

  DENSE INTERCONNECT SCORE:  0.44
  (189.13 GB/s measured  /  434.71 GB/s ideal)

  1.00 = perfect, 0.00 = none
===========================================================
```

### Latency

```
===========================================================
  AVG 1:1 LATENCY:         3.42 us
  AVG LOADED LATENCY:      6.81 us  (all GPUs concurrent)

  LATENCY SCORE:           0.50
  (1.00 = no degradation under load)
===========================================================
```
