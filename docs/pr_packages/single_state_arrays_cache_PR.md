# Fix prefix cache support for single-state ArraysCache

> **PR Package** — for upstream submission. NOT yet opened.

---

## Summary

Prefix cache recurrent checkpoint restore rejected valid Arrays-family caches exposing one state tensor (`state_count == 1`, e.g. LFM2.x liquid-style hybrid attention + short-conv).

Three guards in `BlockAwarePrefixCache` previously assumed state cardinality `>= 2`. This patch lowers the lower bound to `1` and routes single-state caches through the existing `__nstate__` marker that already round-trips N-tuple caches.

The 2-tuple legacy Mamba/SSM shape is preserved verbatim, and no new persisted format is introduced.

---

## Root cause

`BoundarySnapshotSSDStore` already correctly serialized single-state recurrent state (its encoder uses `len(state) >= 1` and writes `state_count` metadata). The deserializer already round-trips tuples of any length.

The `ArraysCacheHandler.reconstruct_cache` already builds `SizedArraysCache(size=len(states))` for any state count.

The three broken guards all lived in `omlx/cache/prefix_cache.py`:

1. **`_validated_gdn_snapshot_layers`** (`omlx/cache/prefix_cache.py:351`):
   rejected `len(state) < 2` — rejected single-state ArraysCache as "structurally invalid".

2. **`_extract_block_tensor_slice`** (`omlx/cache/prefix_cache.py:2338`):
   only handled `len > 2` (N-tuple) and `len >= 2` (legacy pair); single-state fell through to a placeholder and was lost on reconstruction.

3. **`_validate_block_cache_data`** (`omlx/cache/prefix_cache.py:4474`):
   rejected `len(elements) < 2` for the N-state marker, invalidating any restored single-state block.

The result was that LFM2.x GDN sidecars were correctly written to disk but rejected at restore time, forcing a 100 % re-prefill on warm requests.

---

## Fix

- `omlx/cache/prefix_cache.py` (49 insertions / 18 deletions across 3 guards):
  - `_validated_gdn_snapshot_layers`: `len < 2` → `len < 1`; add `len == 1` branch that produces `("__nstate__", type_name, list(state))`.
  - `_extract_block_tensor_slice`: `len > 2` → `len >= 1`; explicit branch for `len == 2` (legacy Mamba pair) preserved; `len == 1` and `len > 2` route through `__nstate__`.
  - `_validate_block_cache_data`: `len(elements) < 2` → `len(elements) < 1`; safe access to `elements[1]` guarded by length check.

- `len == 0` remains rejected at all three sites.
- `len == 2` legacy pair path is **byte-identical** to the prior implementation.
- `len >= 3` N-tuple path uses the same `__nstate__` marker as before.

---

## Real-world reproducer (LFM2.5-2.6B-MLX-8bit-official)

Same model, same prompts, two clean runs from a fresh SSD cache directory:

| Run         | `cached_tokens` | re-prefill | total duration |
|-------------|-----------------|------------|----------------|
| before patch | **0 / 9880**   | 9880       | 54.86 s        |
| after patch  | **8192 / 9880** | 1688       | 44.42 s        |

Logs before patch:

```
WARNING - Ignoring structurally invalid recurrent checkpoint for block 4
WARNING - Ignoring structurally invalid recurrent checkpoint for block 3
WARNING - Ignoring structurally invalid recurrent checkpoint for block 2
WARNING - Ignoring structurally invalid recurrent checkpoint for block 1
INFO  - Split GDN restore found no compatible recurrent checkpoint
INFO  - prefix cache: request … re-prefills 9880 of 9880 tokens (reused 0)
```

Logs after patch:

```
INFO - Using boundary cache snapshot for b1fa5091-…: storing 8192/12131 tokens
       (skipping trailing partial block, 3 intermediate snapshots)
INFO - Flushed 4 hot cache blocks to SSD
usage.prompt_tokens_details.cached_tokens = 8192
```

---

## Regression validation

### Automated suite

`omlx/cache/prefix_cache.py` modified paths covered by 982 tests, all passing:

```
$ /Users/bot/mlx-vlm-prod-v0.6.8/bin/python -m pytest \
    tests/test_arrays_cache_single_state.py \
    tests/test_arrays_cache_adversarial.py \
    tests/test_boundary_snapshot_store.py \
    tests/test_cache_factory.py tests/test_cache_ntuple_state.py \
    tests/test_cache_observability.py tests/test_cache_stats.py \
    tests/test_cache_type_handlers.py tests/test_cluster_kv_reservation.py \
    tests/test_gdn_sidecar_index.py tests/test_gdn_sidecar_quantization.py \
    tests/test_hybrid_cache.py tests/test_index_cache.py \
    tests/test_nested_nstate_serialization.py tests/test_paged_cache.py \
    tests/test_paged_ssd_cache.py tests/test_pooling_cache_append_inplace.py \
    tests/test_pooling_cache_delta.py tests/test_prefix_cache.py \
    tests/test_prefix_cache_cachelist_mixed.py \
    tests/test_prefix_cache_cachelist_per_member.py \
    tests/test_prefix_cache_dedup_backfill.py \
    tests/test_prefix_cache_gdn_split.py \
    tests/test_prefix_cache_rotating_tip_strip.py \
    tests/test_prefix_cache_v4_block_storage.py \
    tests/test_rotating_cache_contract.py \
    tests/test_runtime_cache_observability.py \
    tests/test_store_cache_gate.py tests/test_hot_cache.py
...
982 passed in 20.40s
```

Two new test files added:

- `tests/test_arrays_cache_single_state.py` (13 tests): the focused regression suite — validator + block extract + end-to-end round-trip for `state_count == 1`, `== 2`, `>= 3`.
- `tests/test_arrays_cache_adversarial.py` (24 tests, Cases A–R): adversarial matrix covering zero-state, malformed elements, KVCache / RotatingKVCache / CacheList unaffected, sidecar corruption / truncation, metadata mismatch, V2 polyfill, partial-block reuse, prefix divergence at start and middle.

One existing test updated in `tests/test_prefix_cache_gdn_split.py`:

- `test_split_restore_walks_back_from_structurally_invalid_sidecar`: the corruption marker was `len(state) == 1` (now considered valid); updated to use empty-tuple `()` which remains structurally invalid. The walk-back assertion (`hit_table.num_tokens == 8`) still holds.

A new positive test added in the same file:

- `test_split_restore_accepts_single_state_arrays_cache`: locks the new behavior — single-state sidecar round-trips without walk-back, `chosen_endpoint_tokens == 8`, `walkback_blocks == 0`.

### Live non-regression

| Model | Cold `cached_tokens` | Warm `cached_tokens` | Result |
|-------|----------------------|----------------------|--------|
| LFM2.5-2.6B-MLX-8bit (hybrid ArraysCache) | 0 / 9860 | **8192 / 9880** | FIXED |
| gemma-4-26B-A4B-it-vlm-4bit (KVCache + vision) | 0 / 2457 | **2048 / 2476** | unaffected |
| gemma-4-26B-A4B-it-5bit (pure KVCache) | 0 / 3024 | **2048 / 2570** | unaffected |

Live Gemma VLM smoke test: a 16-token text request returned the correct answer in 0.84 s with no cache warnings.

### Adversarial partial-prefix (LFM live)

| Request | Divergence | `cached_tokens` | Notes |
|---------|-----------|-----------------|-------|
| Q1 (cold) | — | 0 | baseline |
| Q2 (same prefix) | none | **8192** | full 4-block reuse |
| Q3 (early diverge ~ char 4500) | inside block 1 | **2048** | only 1 block reused; **no unsafe reuse past divergence** |
| Q4 (end diverge ~ char 23000) | last block | **8192** | full 4-block reuse |

### Persistence across server restart

```
INFO - SSD cache scan complete: scanned=4, indexed=4, errors=0,
       total_size=128.74 MB, gdn_sidecars=4, gdn_size=727.47 KB
INFO - PagedSSDCacheManager initialized: incompatible_files=0
INFO - PagedSSDCacheManager updated layer cache signature
       (30 layers, 2 unique types, turboquant_kv_bits=None, cachelist_subtypes=no)

[warm request after restart]
usage.prompt_tokens_details.cached_tokens = 8192
```

---

## Compatibility

Why nothing legitimate breaks:

- **`state_count == 0`** remains rejected at all three guards (`len < 1`).
- **`state_count == 1`** was previously rejected; now accepted and routed through `__nstate__`. The `SizedArraysCache` reconstructor already handled this size.
- **`state_count == 2`** legacy Mamba pair is **byte-identical** — the `len == 2` branch keeps the plain `(state[0], state[1])` unpack without consulting the marker.
- **`state_count >= 3`** continues to use `__nstate__` exactly as before.
- **KVCache** is not in the `is_arrays_family()` set; the validator returns `{}` for it and the slicing branch is guarded by `handler.supports_block_slicing`. The new branch is unreachable for KVCache layers.
- **RotatingKVCache** has its own slice branch (`elif is_rotating_family(...)`); the new branch is unreachable.
- **CacheList** has its own sub-cache handling; the new branch is unreachable for the top-level layer data.
- **TurboQuantKVCache**, **PoolingCache**, **MiniMaxM3KVCache**, **BatchKVCache** — all gated by `is_arrays_family()` or have their own branch; the new branch is unreachable for these.

No new persisted wire format: the sidecar format version stays at `1`. Existing 2-tuple and N-tuple sidecars remain readable (V2 polyfill path preserved).

---

## Limitations (non-blocking follow-ups)

1. **GDN quantization inactive for `state_count == 1`** — `_should_quantize_gdn_state` is gated on `state_index == 1`, which is never reached for LFM2.x single-state caches. Sidecars for `state_count == 1` are written in the model's native dtype (e.g. `bf16`) instead of `rht_int16`. The compression ratio for LFM sidecars is therefore lower than for 2-state caches. Propagating `state_count` to that selector is a small, separate change that does not affect this fix's correctness.

2. **Single-state `ArraysCache` nested inside `CacheList`** — not empirically covered by tests because no LFM-family model in the test corpus exposes that geometry today. The code path should handle it correctly (the `__nstate__` marker is generic on length and `CacheListHandler.reconstruct_cache` is length-agnostic), but adding a dedicated test if such a model emerges is reasonable.

3. **Corruption `2 → 1` element truncation is now silent at validation** — a sidecar whose ArraysCache layer's state was originally a 2-tuple but was corrupted to a 1-tuple would now be accepted (previously rejected). The resulting `SizedArraysCache(size=1)` would either crash downstream on `cache.cache[1]` access or produce a wrong-shape error. The corruption case is rare and surfaces loudly (not silently wrong) because MLX's `ArraysCache.__getitem__` raises `IndexError` for out-of-range access. Adding `state_count` to the cache signature would catch this corruption mode if desired — separate change, not part of this PR.

---

## Maintainer reproducer

Generic, independent of private paths:

1. Run an LFM2.x-class model that exposes single-state ArraysCache (e.g. `LFM2.5-2.6B-MLX-8bit-official`, 22/30 layers are single-state ArraysCache).
2. Start the server with `--paged-ssd-cache-dir /tmp/lfm-cache --paged-ssd-cache-max-size 4GB`.
3. Send a long prompt (~10 k tokens) that produces at least 2 block boundaries.
4. Send a second request that shares a block-aligned prefix.

| Before patch | After patch |
|--------------|-------------|
| `cached_tokens = 0` | `cached_tokens = 8192` |
| `WARNING - Ignoring structurally invalid recurrent checkpoint for block N` (× N blocks) | (no such warning) |
| `INFO - Split GDN restore found no compatible recurrent checkpoint` | `INFO - Using boundary cache snapshot for …: storing 8192/…` |
| `prefix cache: request … re-prefills 9880 of 9880 tokens (reused 0)` | `cached_tokens: 8192` |

---

## Files changed

- `omlx/cache/prefix_cache.py` — 49 insertions / 18 deletions (the fix)
- `tests/test_arrays_cache_single_state.py` — 548 insertions (new regression suite)
- `tests/test_arrays_cache_adversarial.py` — 863 insertions (new adversarial matrix)
- `tests/test_prefix_cache_gdn_split.py` — 124 insertions / 1 deletion (corruption-strategy update + new positive lock)
- `docs/reports/arrays_cache_single_state_fix.md` — initial technical report
- `docs/reports/lfm_single_state_validation.md` — live validation report

**Total**: 2133 insertions, 19 deletions across 6 files.

---

## Branch

- `fix/arrays-cache-single-state-prefix-cache`
- HEAD: `3bf0acc6693adf520515cb1eccd071425f0b1f26`
- Off: `f610c6a8` (main)
- Local commit only — **NOT pushed** as of this package.
