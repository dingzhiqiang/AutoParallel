# Add Model Architecture

Guide for adding support for a new model architecture feature in AutoParallel.

## When to Use
When a new model has config.json fields that AutoParallel doesn't recognize, or uses
a novel attention/FFN pattern that affects parallelism constraints or memory estimation.

## Steps

### 1. Update `ModelSpec.from_hf_config()`

Add parsing for new config.json fields:

```python
# In from_hf_config():
my_new_field=cfg.get("my_new_field", 0) or 0,
```

### 2. Add Properties (if needed)

```python
@property
def has_my_feature(self) -> bool:
    return self.my_new_field > 0
```

### 3. Update Memory Estimation

In `estimate_memory()` and `estimate_inference_memory()`, add the memory impact
of the new architecture feature.

### 4. Update Throughput Estimation

In `estimate_throughput()` and `compute_inference_score()`, add the compute/comm
cost of the new feature.

### 5. Update Constraint Checking

In `search_strategies()` and `search_inference_strategies()`, add any new
parallelism constraints (e.g., divisibility requirements).

## Checklist

- [ ] `ModelSpec` dataclass: add field with default
- [ ] `from_hf_config()`: parse from config.json
- [ ] `estimate_memory()`: memory impact for training
- [ ] `estimate_inference_memory()`: memory impact for inference
- [ ] `estimate_throughput()`: compute/comm cost for training
- [ ] `compute_inference_score()`: cost for inference
- [ ] `search_strategies()`: constraint checks
- [ ] `_total_params()`: parameter count
- [ ] README: document in supported models table
