# Persistent exact-overlap runtime

`pierce_query_runtime` supplies the two-pass overlap implementation shared by
`pierce_overlap_two_pass` and the persistent worker. The benchmark CLI retains its
arguments, setup order, warmups, CSV export, and historical timing definitions.
The runtime's preflight memory policy is opt-in at the query/build helper level;
the CLI does not enable it. Other query applications retain their default behavior.

## Ownership and API

`pierce::QueryRuntime` owns one OptiX context and overlap pipeline, plus a registry
of independently owned `ResidentPair` objects. `loadPair(key, pathA, pathB)` returns
a stable pair reference and an opaque handle. The pair is registered only after
both datasets load, upload, and build successfully. Loading the same key/paths is
idempotent; changing paths under a resident key is rejected. Query and unload use
`queryPair(handle)` and `unloadPair(handle)`. Stale/unknown handles are rejected.

Each pair retains its load measurement, completed-query count, last successful
query metrics, and explicitly owned resident bytes. `RuntimeResult` returns the
full unique object-pair vector plus timings and raw result-buffer measurements.
Worker responses transmit only metrics. Every query constructs fresh parameters
and temporary buffers; the resident meshes, edges, BVHs, and pipeline are reused.
Failed query budgeting leaves previous successful metrics intact.

Calls must be serialized on one owning thread. Pairs remain at stable addresses
and are destroyed before their pipeline/context. No implicit warmup or eviction
occurs. Pipeline initialization happens on the worker's first Load.

## Worker protocol, version 2

Run `pierce_worker [--ptx path/to/mesh_overlap_edges.ptx]`. Send one UTF-8 JSON
object per line on stdin; read one response per request from stdout. Existing
native diagnostics are redirected to stderr, which clients must continuously drain.

Requests contain `version: 2`, a nonempty string `id`, and `command`:

| Command | Additional request fields | Behavior |
| --- | --- | --- |
| `load` | `key`, `mesh1`, `mesh2` | Atomically register a pair, or reuse the same key/paths |
| `query` | `pairHandle` | Query that pair's A/B collections |
| `unload` | `pairHandle` | Remove only that pair |
| `status` | none | Observe registry, counters, memory, and allocation checkpoints |
| `shutdown` | none | Release all resources and exit after replying |

Success replies contain `{ "version": 2, "id": "...", "ok": true, "data": ... }`.
Every success includes a complete status snapshot: `pairs`, lifetime lifecycle
counters, `runtimeInitializationUs`, explicitly owned dataset/runtime bytes,
`memory`, and `checkpoints`. Before initialization `memory` is null. After shutdown
pairs are empty and memory is null. Each pair has `key`, `pairHandle`,
`datasetLoadUs`, `datasetGpuBytes`, `completedQueries`, and nullable `lastResult`.
The latter contains `resultCount`, `queryExecutionUs`, `benchmarkPhaseSumUs`,
`rawHitCount`, and `resultBufferBytes`.

Recoverable errors have `ok: false`, `code`, an `error` message, and the surviving
status in `data`. Memory errors use `code: "memory_budget_exceeded"` with `details`
containing `stage`, `requiredBytes`, `availableBytes` (after reserve), and
`reservedBytes`. Other request/load errors use `native_error`. Unparseable requests
may have an empty ID. Fatal CUDA/OptiX failures terminate the process; clients must
invalidate every handle. EOF also releases resources and exits.

## Memory checks

`MemoryBudget` reserves `max(512 MiB, 10% of total CUDA memory)` and supports an
injected `MemoryProvider` for deterministic rejection tests. Upload guards cover
mesh/edge allocations; BVH guards use OptiX's exact reported output and temporary
build sizes. Queries check count/offset buffers before allocation, then check the
raw result bytes plus twice that size as estimated sorting workspace after hit
counting. Byte additions/multiplications are checked for overflow.

Allocation checkpoints record stage, requested bytes, and observed device memory.
Status reports device name, total/free/reserved bytes. Explicit accounting excludes
driver/context and internal OptiX allocations; temporary query buffers are zero
between commands. Checkpoints and snapshots are observations, not exact peak GPU
usage. External allocations can race a successful preflight check; unexpected
CUDA allocation errors retain the existing fatal-process behavior.

## Verification

`pierce_runtime_verify a.pre b.pre baseline.csv zero.pre` checks the fixed small
cube workload, 100 full-pair comparisons, reloads, failure paths, and zero results.

`pierce_multi_verify small-a small-b large-a large-b small.csv large.csv` checks
100 alternating full-pair comparisons, independent pair lifetimes, exactly four
initial reads/uploads/builds, injected upload/BVH/edge/count/result-budget
rejections, cleanup/recovery, stale handles, checked arithmetic, and reload cycles.
Global CUDA free-memory checks allow 16 MiB of variation; explicitly owned bytes
must return exactly to prior values. Run the verifiers on an otherwise idle GPU.
Explorer's `python -m backend.verify_gpu` captures CLI baselines and also runs 100
alternating real API/worker queries, recording results and memory observations.
