# Production State Inventory — 2026-08-19 01:30 CEST

## TL;DR

Production is HEALTHY. Engine pool has 23 models, 3 loaded. Notable issues:
- lfm-2.6b-8bit is NOT pinned and NOT loaded (only 3 / 4 desired resident models loaded)
- qwen3.6-27b-optiq is in registry, public, not pinned, not loaded (retirement target)
- mlx-community/multilingual-e5-large is in registry, public, NOT loaded → still required by Honcho (1024-dim, 3794 embeddings stored)
- Tier-5 gemma-4-26B-A4B-it-official (48GB at hf-cache) — NOT to delete, revalidate separately

## Production Runtime Isolation

| Field | Value |
|-------|-------|
| PRODUCTION_PID | 35075 |
| PRODUCTION_LAUNCHD_PID | 35075 (parent 35093 = bash launcher) |
| PRODUCTION_PYTHON | `/Users/bot/Library/Application Support/oMLX-Prod/runtime/venv/bin/python3.11` |
| PRODUCTION_OMLX_INSTALL | `/Users/bot/Library/Application Support/oMLX-Prod/runtime/venv/lib/python3.11/site-packages/omlx/` |
| PRODUCTION_PORT | 8930 |
| PRODUCTION_HEALTH | healthy |
| PRODUCTION_LAUNCHAGENT | `~/Library/LaunchAgents/local.omlx-prod.server.plist` |
| PRODUCTION_BASE_PATH | `/Users/bot/Library/Application Support/oMLX-Prod` |
| PRODUCTION_ENDPOINT | `http://127.0.0.1:8930`, `http://192.168.40.7:8930` |
| API_KEY | `/Users/bot/Library/Application Support/oMLX-Prod/api.key` (mode 600) |

## Runtime Isolation Verified

- `omlx.__file__` resolves to **venv-installed omlx** when invoked with PYTHONSAFEPATH=1 and cwd=/tmp
- `/opt/homebrew/Cellar/omlx/0.6.0/libexec/...` exists but **NOT in the runtime path** (launcher uses dedicated venv)
- Both `homebrew` and `venv` versions have the same `prefix_cache.py` SHA256
  - `283cc97ddde4fa9454013983ad653d66a479624a77d50baf7db15c2f7310df91`
  - Single-state ArraysCache patch present in both
- `PYTHONSAFEPATH=1` enforced via launchd EnvironmentVariables

## ArraysCache Patch

| Field | Value |
|-------|-------|
| PATCH_PRESENT | YES |
| PATCH_SHA | `283cc97ddde4fa9454013983ad653d66a479624a77d50baf7db15c2f7310df91` |
| Source commit | `3bf0acc6693adf520515cb1eccd071425f0b1f26` |
| Source branch | `fix/arrays-cache-single-state-prefix-cache` |
| Patch title | "support single-state ArraysCache prefix reuse" |
| Files changed | 1 (prefix_cache.py) |
| Lines added/removed | +49 / -18 |

## Memory State

| Field | Value |
|-------|-------|
| MEMORY_TOTAL | 64 GB (68719476736) |
| MEMORY_CEILING | 55.66 GB (55662788608) |
| MEMORY_USED | 21.27 GB (22839145201) |
| MEMORY_FREE | ~13.65 GB (vm_stat) |
| Headroom | ~30 GB to ceiling |

## Engine Pool (current)

| Field | Value |
|-------|-------|
| model_count | 23 |
| loaded_count | 3 |
| ceiling | 55662788608 bytes |
| current_model_memory | 22839145201 bytes |

Loaded models:
- `gemma-4-26B-A4B-it-vlm-4bit` (alias: gemma-vision) — DEFAULT, PINNED, FAV
- `Qwen3-Embedding-4B-4bit-DWQ` (alias: qwen3-embedding-4b) — FAV, TYPE=embedding
- `Qwen3-Reranker-4B-mxfp8` (alias: qwen3-reranker-4b) — FAV, TYPE=reranker

## Public Models (9)

| Alias | Path | Status |
|-------|------|--------|
| gemma-vision | gemma-4-26B-A4B-it-vlm-4bit | LOADED, PINNED, DEFAULT |
| lfm-2.6b-8bit | 8bit | NOT LOADED, NOT PINNED |
| qwen3-embedding-4b | Qwen3-Embedding-4B-4bit-DWQ | LOADED |
| qwen3-reranker-4b | Qwen3-Reranker-4B-mxfp8 | LOADED |
| multilingual-e5-large | mlx-community/multilingual-e5-large | NOT LOADED, FAV |
| gemma-vision-12b | gemma-4-12B-it-vlm-4bit | NOT LOADED |
| gemma-text-26b-5bit | gemma-4-26B-A4B-it-5bit | NOT LOADED |
| gemma-vision-26b-5bit | gemma-4-26B-A4B-it-vlm-5bit | NOT LOADED |
| qwen3.6-27b-optiq | mlx-community--Qwen3.6-27B-OptiQ-4bit | NOT LOADED, RETIRE TARGET |

## Hidden models (legacy / retired)

- mlx-community/gemma-4-26b-a4b-it-4bit (HIDDEN)
- _hidden_qwen3_embedding_legacy (HIDDEN, TYPE=embedding)
- _hidden_qwen3_reranker_legacy (HIDDEN, TYPE=reranker)
- lfm-2.6b-{4bit,5bit,6bit,bf16,mxfp4,mxfp8,nvfp4} (HIDDEN)
- _hidden_bonsai (Ternary-Bonsai-27B, HIDDEN)
- _hidden_gemma_12b_official, _hidden_gemma_26b_official, _hidden_gemma_26b_assistant_official
- _hidden_drafter_assistant_bf16, _hidden_drafter_assistant_12b, _hidden_drafter_qat_assistant

## Retrieval Directory Topology

```
/Users/bot/mlx-models/retrieval/
├── Qwen3-Embedding-4B-4bit-DWQ/  (REAL DIR)
│   ├── added_tokens.json → .../snapshots/b5d88f1/added_tokens.json
│   ├── config.json → .../snapshots/b5d88f1/config.json
│   ├── generation_config.json → .../snapshots/b5d88f1/generation_config.json
│   ├── merges.txt → .../snapshots/b5d88f1/merges.txt
│   ├── model.safetensors → .../snapshots/b5d88f1/model.safetensors
│   ├── model.safetensors.index.json → .../snapshots/b5d88f1/model.safetensors.index.json
│   ├── special_tokens_map.json → .../snapshots/b5d88f1/special_tokens_map.json
│   ├── tokenizer_config.json → .../snapshots/b5d88f1/tokenizer_config.json
│   ├── tokenizer.json → .../snapshots/b5d88f1/tokenizer.json
│   └── vocab.json → .../snapshots/b5d88f1/vocab.json
│
└── Qwen3-Reranker-4B-mxfp8/  (REAL DIR)
    ├── config.json → .../snapshots/25f203a/config.json
    ├── generation_config.json → .../snapshots/25f203a/generation_config.json
    ├── model.safetensors → .../snapshots/25f203a/model.safetensors
    ├── model.safetensors.index.json → .../snapshots/25f203a/model.safetensors.index.json
    ├── special_tokens_map.json → .../snapshots/25f203a/special_tokens_map.json
    ├── tokenizer_config.json → .../snapshots/25f203a/tokenizer_config.json
    ├── tokenizer.json → .../snapshots/25f203a/tokenizer.json
    └── vocab.json → .../snapshots/25f203a/vocab.json
```

All 18 symlinks resolve and are valid. NONE broken. NO duplicated weights.

## HF Sources (preserve)

- `~/.cache/huggingface/hub/models--mlx-community--Qwen3-Embedding-4B-4bit-DWQ/snapshots/b5d88f1fe49b50d2ac01b4692ca2d387f14f9c72`
- `~/.cache/huggingface/hub/models--mlx-community--Qwen3-Reranker-4B-mxfp8/snapshots/25f203a237b822a90f38763843562b93a5baf82f`

## Sanity Check Stats

- `total_requests`: 173 (lifetime)
- `total_prompt_tokens`: 74375
- `total_completion_tokens`: 1654
- `total_cached_tokens`: 20480 (LFM prefix cache hits)
- Per-model requests: 8bit (61) — heavy LFM use, gemma-4-26B-A4B-it-vlm-4bit (44), mlx-community/gemma-4-26b-a4b-it-4bit (30)

## Cross-cutting Findings

1. **8bit (LFM) had 20480 cached tokens** — confirms prefix cache works
2. **mlx-community** prefix in stats is an alias path that oMLX records; needs scan to confirm no broken routing
3. **multilingual-e5-large has 329 prompt tokens / 20 requests in stats** — required by Honcho, keep
4. **Qwen3-Embedding-4B-4bit-DWQ has 11 prompt_tokens / 1 req** — light usage, but part of canonical stack
5. **qwen3.6-27b-optiq has 117 prompt_tokens / 7 req** — last uses 2026-08-18, before retirement
