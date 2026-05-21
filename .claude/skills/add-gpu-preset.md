# Add GPU Preset

Guide for adding a new GPU hardware preset to AutoParallel.

## When to Use
When you need to add support for a new GPU type (e.g., B200, B100).

## Steps

1. Open `autoparallel/__main__.py`
2. Find `GPU_PRESETS` dictionary
3. Add a new entry following the pattern:

```python
"B200": HardwareSpec(
    gpu_flops=2250e12,      # BF16 peak TFLOPS (from spec sheet)
    bw_nvlink=900e9,        # NVLink unidirectional bandwidth (GB/s → bytes/s)
    bw_ib=100e9,            # IB RDMA unidirectional bandwidth
    hbm_bw=8.0e12,          # HBM bandwidth (for decode roofline)
    gpu_memory_gb=192,      # HBM capacity
    host_memory_gb=2000,    # Typical host memory per node
),
```

4. (Optional) Add a preset profiling JSON in `autoparallel/profile_data/presets/`
   - File naming: `{GPU}_{NVLink_gen}_{IB_speed}.json`
   - Example: `B200_NVLink5_IB800.json`

## Key Fields

| Field | Source | Unit |
| --- | --- | --- |
| `gpu_flops` | GPU spec sheet, BF16 Tensor Core peak | FLOPS (e.g., 990e12) |
| `bw_nvlink` | NVLink spec, unidirectional per GPU | bytes/s |
| `bw_ib` | InfiniBand spec (HDR=25GB/s, NDR=50GB/s, XDR=100GB/s) | bytes/s |
| `hbm_bw` | HBM spec sheet | bytes/s |
| `gpu_memory_gb` | GPU HBM capacity | GB |
| `host_memory_gb` | Typical node CPU memory (DGX H100=1TB, DGX H200=1.5TB) | GB |

## Validation

```bash
# Test with the new GPU type
python -m autoparallel --model-path /path/to/model --n-gpus 8 --gpu-type B200
```
