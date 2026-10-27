# Persistent exact-overlap runtime

The `pierce_query_runtime` CMake library contains the shared two-pass overlap
implementation and `pierce::QueryRuntime`. The original `pierce_overlap_two_pass`
CLI uses the shared query implementation while retaining its setup order, warmups,
CSV output, and benchmark measurements. Other query applications remain separate.

`QueryRuntime` owns its OptiX context, overlap pipeline, and at most one dataset
pair. `load(a, b)` reads prepared `.pre` files, uploads meshes and unique edges,
and builds both indexes. It commits the pair only after both loads succeed.
`query(handleA, handleB)` returns unique pairs, complete synchronized native
execution time, and the historical benchmark phase sum. Fresh launch parameters
and temporary buffers are used each time; datasets and the pipeline are reused.
`unload()` releases datasets. Destruction releases datasets before their pipeline
and context. Calls must be serialized on one owning thread; objects are not
thread-safe or copyable. No warmup is implicit.

## Worker protocol (version 1)

Run `pierce_worker [--ptx path/to/mesh_overlap_edges.ptx]`. Send one UTF-8 JSON
object per line on stdin; read one response line per request from stdout. All
native diagnostic output is redirected to stderr, which clients must drain.
The context and pipeline are initialized lazily on the first Load command.

Every request contains `version: 1`, a nonempty string `id`, and `command`.
Success responses contain `{ "version": 1, "id": "...", "ok": true, "data": {} }`.
Recoverable errors contain `ok: false` and an `error` string instead of `data`.
Malformed requests use an empty ID if it could not be parsed. Fatal CUDA/OptiX
errors terminate the worker; clients must invalidate all handles and restart.
EOF also shuts the worker down.

| Command | Additional request fields | Success data |
| --- | --- | --- |
| `load` | `mesh1`, `mesh2`: prepared paths | `handleA`, `handleB`, `datasetLoadUs`, `runtimeInitializationUs` |
| `query` | `handleA`, `handleB` | `resultCount`, `queryExecutionUs`, `benchmarkPhaseSumUs` |
| `unload` | none | `loaded: false` |
| `status` | none | loaded state, handles, lifecycle counters, explicitly owned GPU buffer bytes |
| `shutdown` | none | empty object, then clean exit |

Load while a pair is resident returns its original handles and measurements;
unload before selecting new files. Unload is idempotent. Handles change after
reload and between runtime instances. The worker sends counts and timing only;
the C++ API returns full object pairs for integration and verification tools.

`status` exposes `datasetReads`, `datasetUploads`, `indexBuilds`,
`pipelineCreations`, and `completedQueries` for the worker lifetime. Failed loads
can increment work counters for work performed before failure. It also reports
`datasetGpuBytes`, `runtimeGpuBytes` (explicit pipeline/SBT parameter buffers),
and `temporaryQueryGpuBytes` (zero between synchronous commands). Driver/context
and internal OptiX allocations are excluded from explicit byte accounting.

## Verification

Build the query project normally, then run:

```text
pierce_runtime_verify a.pre b.pre baseline.csv zero.pre
```

This compares all pairs with the baseline CSV and the analytical 1,000-by-1,000
cube workload over 100 queries, checks stable residency, exercises reload and
failure paths, and writes a spatially separated zero-result fixture. Memory
checks allow 16 MiB of global free-memory variation from caches/other GPU clients.
Explorer's `python -m backend.verify_gpu` additionally tests the CLI and worker
protocol end to end. Capture the old CLI's exported CSV before refactoring when
an independent previous-version comparison is required.
