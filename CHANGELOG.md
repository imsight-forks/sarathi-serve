# Changelog (local patches)

This file documents **local changes** made to the `extern/tracked/sarathi-serve` submodule for the `gpu-simulate-test` repo workflow. These patches are not part of upstream Sarathi-Serve unless explicitly upstreamed.

## 2025-12-31

### Added

- Qwen3 model support for Sarathi-Serve model executor:
  - Added `Qwen3ForCausalLM` implementation to load and run HuggingFace `Qwen3ForCausalLM` checkpoints, including:
    - GQA (`num_key_value_heads`) support with `head_dim` (query/key/value projection dims derived from `head_dim` rather than `hidden_size / num_heads`)
    - `q_norm` / `k_norm` per-head RMSNorm matching Qwen3 weight keys
  - Registered the architecture name `Qwen3ForCausalLM` so `config.json` with `"architectures": ["Qwen3ForCausalLM"]` can be loaded.

### Changed

- `ModelConfig.get_head_size()` now prefers `hf_config.head_dim` (or `hf_config.head_size`) when present, falling back to `hidden_size / num_attention_heads`.
  - This is required for models like Qwen3 where `head_dim != hidden_size / num_attention_heads`.

### Files touched

- `sarathi/config/config.py`
- `sarathi/model_executor/model_loader.py`
- `sarathi/model_executor/models/__init__.py`
- `sarathi/model_executor/models/qwen3.py`

### Motivation / verification

- Motivation: enable `gpu-simulate-test`’s `real-bench` Sarathi backend to run Qwen3-0.6B and generate **real GPU timing** comparable to Vidur CPU-side simulation outputs.
- Verified in this repo by running `pixi run real-bench backend=sarathi model=qwen3_0_6b model.model_id=...` on an A100.

