# Add Engine Preset

Guide for adding a new training/inference engine cost model preset to AutoParallel.

## When to Use
When you need to add support for a new engine with different runtime optimization behaviors
(e.g., a new inference framework, or a custom training engine).

## Steps

1. Open `autoparallel/__main__.py`
2. Find `EngineConfig` dataclass and existing presets (`MEGATRON_ENGINE`, `FSDP_ENGINE`, `SGLANG_ENGINE`)
3. Add a new preset:

```python
MY_ENGINE = EngineConfig(
    overlap_ep_alltoall=True,   # Does the engine overlap EP AllToAll with FFN compute?
    tp_bw_degradation=False,    # Does NVLink BW degrade for TP > 4?
    act_factor_dense=10,        # Activation memory factor for dense layers
    act_factor_moe=14,          # Activation memory factor for MoE layers
)
```

4. Register in the CLI argument parser — find `choices=["megatron", "fsdp", "sglang"]` and add your engine
5. Register in the `engine_map` dictionaries (there are two: one for training mode, one for inference mode)

## Key Parameters

| Parameter | Effect on Cost Model | How to Determine |
| --- | --- | --- |
| `overlap_ep_alltoall` | `True`: cost = max(ep, ffn); `False`: cost = ep + ffn | Check if engine overlaps AllToAll with compute |
| `tp_bw_degradation` | `True`: BW = BW/sqrt(tp/4) for tp>4 | Benchmark NVLink BW with large TP groups |
| `act_factor_dense` | Multiplier for activation memory per token | Profile or estimate from engine source code |
| `act_factor_moe` | Higher than dense due to router/dispatch buffers | Profile or estimate from engine source code |
