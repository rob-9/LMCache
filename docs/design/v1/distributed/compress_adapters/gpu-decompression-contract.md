# Portable Record GPU Decompression Contract

## Status

This document defines the contract targeted by the first backend-interface
implementation. The portable record format and Python reference codec exist
under `lmcache/v1/distributed/compress_adapters/`, but no production serde, L2,
or retrieve path uses them yet.

The undeployed draft wire format now carries a checksum over compressed payload
bytes and enforces GPU-compatible payload alignment. One prerequisite remains
before implementing `NvcompBackend`: prove the resulting frozen records against
the selected nvCOMP and hipCOMP versions.

If compatibility with any existing draft record becomes necessary, the wire
change requires a new record version. Otherwise, version 1 may be revised while
it remains undeployed, with all frozen vectors updated together.

## Purpose

LMCache should store a backend-independent compressed record, copy its exact
bytes to an accelerator, and expand it into the existing contiguous KV staging
buffer:

```text
portable record in L1
  -> compatible device input buffer
  -> GpuDecompressBackend
  -> validated contiguous device KV staging
  -> existing paged-KV placement
```

The first implementation will use nvCOMP. A hipCOMP implementation must fit
the same shared interfaces without adding CUDA assumptions to generic code.

The portable record is the interoperability boundary: it identifies Deflate
and its framing, not the library that encoded or decodes it. The backend is a
runtime choice and is never written into the record.

## Goals

- Keep CUDA, HIP, nvCOMP, and hipCOMP types inside vendor implementations.
- Reject unsupported record/backend combinations before native submission.
- Submit independently compressed record chunks as a device batch.
- Verify compressed-payload integrity before native decompression.
- Verify every output chunk's status, exact size, and CRC before KV is usable.
- Preserve the existing contiguous staging and paged-placement path.
- Make buffer ownership, finalization, and asynchronous completion explicit.

## Non-goals

- Selecting the production store-side compressor, framing, or chunk size.
- Connecting the portable format to serde or L2 in the interface-only slice.
- Loading directly from L2 into device memory.
- Decompressing directly into non-contiguous paged KV blocks.
- Silently falling back to CPU decompression.
- Treating CRC-32 as authentication for attacker-controlled records.

## Integrity and trust boundary

Native Deflate/Gzip decompression cannot be treated as a memory-safe validator.
nvCOMP documents that corrupt input can cause undefined behavior and does not
guarantee a useful per-chunk error status. Therefore, arbitrary bytes from an
untrusted writer must not reach a native GPU decompressor.

The draft record carries a `compressed_payload_crc32` protected by the header
checksum. It covers every byte in `[header_size, record_size)`, including
deterministic zero alignment padding. The deferred L1 path verifies this
checksum on the host before H2D copy or native submission.

Full-record validation returns an immutable `ValidatedCompressedRecord`
containing the parsed header, immutable record bytes, and verified payload
checksum. Public validation requires immutable `bytes` and retains them without
copying; mutable or otherwise aliased input is rejected before parsing. This
avoids an unbounded defensive copy and gives the validated value a stable owner
that callers cannot release. A decompression item accepts this value rather
than a bare header. Header-only parsing remains available for size discovery,
but its result is not eligible for native submission.

The later zero-copy L1 path uses a guarded validated-record form that retains
the source `MemoryObj` read lock. A read-only memoryview alone is insufficient
because another alias could mutate its underlying owner after validation.

This checksum detects accidental storage or transport corruption. It does not
authenticate malicious input because an attacker can recompute it. A deployment
that permits untrusted writes must first authenticate the complete record or
pass it through a memory-safe bitstream-validation boundary. Otherwise GPU
decompression is disabled for that storage path.

After decompression, every backend computes CRC-32/IEEE over each output chunk
and compares it with the descriptor's `uncompressed_crc32`. Native status and
actual output size alone do not establish content integrity.

## Portable alignment policy

The production encoder uses a 16-byte portable payload alignment. The first
payload offset and every later compressed payload offset are rounded up to 16
bytes; padding bytes are zero and are included in the compressed-payload CRC.
The format requires the minimal aligned offset after the header or preceding
payload, so extra padding is not another valid encoding of the same record.
Complete-record validation rejects non-zero gap bytes.

Production uncompressed chunk size is also divisible by 16, except for the
final chunk. Given an aligned output base, prefix-sum output ranges are then
aligned for every non-final chunk. The final chunk has no following output
range whose alignment depends on its size.

Sixteen bytes exceeds nvCOMP raw Deflate's current four-byte requirement and
keeps records portable without encoding a vendor name. A backend still checks
resolved native addresses against its runtime requirements. Records that do not
satisfy a selected backend may be repacked explicitly or rejected; they are
never submitted on the strength of caller-asserted alignment metadata.

## Module ownership

Reusable accelerator values belong under `lmcache/v1/platform/base/`:

```text
DeviceIdentity
CompatibleDeviceBuffer
DeviceExecutionContext
```

Compression-specific values belong under
`lmcache/v1/distributed/compress_adapters/`:

```text
GpuDecompressExecutionPolicy
GpuDecompressCapabilities
ValidatedCompressedRecord
GpuDecompressItem
GpuDecompressRequest
GpuDecompressCompletion
GpuDecompressBackend
```

Completion validation is compression-specific because it owns per-chunk
status, actual-size, and CRC semantics. It does not belong in a generic
platform `DeviceCompletion`.

Vendor implementations live in separate modules:

```text
compress_adapters/nvcomp_backend.py
compress_adapters/hipcomp_backend.py
```

Generic modules do not import vendor modules directly. Resolution follows the
existing lazy `DeviceSpec` capability pattern so installations without an
optional native library can still import LMCache.

## Shared device values

The following sketches describe semantics, not final Python spelling.

### Device identity

```python
DeviceIdentity(platform_type, runtime, device_index)
```

`platform_type` follows LMCache's existing registration, where both CUDA and
ROCm use the `cuda` `DeviceSpec`. `runtime` distinguishes CUDA from HIP using
build/runtime evidence such as `torch.version.hip`; it does not introduce a
competing `hip` platform registration. `device_index` identifies the concrete
accelerator. Equality requires every field to match.

### Compatible device buffer

```python
CompatibleDeviceBuffer.from_owner(
    owner,
    byte_offset,
    byte_length,
)
```

The buffer factory derives device identity and physical capacity from a
supported public owner API, such as `torch.Tensor.data_ptr()`/`nbytes` or
`MemoryObj.data_ptr`/`get_physical_size()`. Callers do not provide capacity,
device, pointer, or alignment claims independently of the owner.

Construction validates `byte_offset + byte_length` with overflow-safe
arithmetic. A vendor backend maps the owner to its native address and repeats
capacity, device, and actual `(pointer + offset)` alignment validation before
submission. This second validation also detects aliasing between distinct
Python owners of one allocation.

The descriptor does not transfer allocation ownership. The represented range
must not be freed, resized, or reused until its completion is finalized.

### Device execution context

```python
DeviceExecutionContext.from_cache_context(cache_context)
```

The adapter captures the existing cache context's device and opaque stream
owner. It does not add another abstract property that every CPU or accelerator
cache context must implement. Only the matching vendor backend translates the
owner to `cudaStream_t`, `hipStream_t`, or an equivalent native type.

A backend submits on the supplied context and never switches silently to a
global or default stream. Input H2D work must be enqueued on the same context,
or that context must first wait on an explicit producer event/completion.

## Decompression request

One `GpuDecompressItem` represents one parsed portable record:

```python
GpuDecompressItem(
    validated_record,
    input_buffer,
    output_buffer,
    expected_uncompressed_size,
)
```

- `validated_record` proves that the complete host record has exact length and
  a valid header plus compressed-payload checksum. Its parsed header supplies
  the metadata below.
- `input_buffer` begins at record byte zero and has a logical byte length equal
  to `validated_record.header.record_size`; the H2D copy source is the exact
  byte view retained by `validated_record`.
- `expected_uncompressed_size` comes from the caller's logical KV layout and
  must equal `validated_record.header.uncompressed_size`.
- `output_buffer.byte_length` must equal `expected_uncompressed_size`; its
  physical owner capacity may be larger.
- Chunk input ranges come from descriptor payload offsets and compressed sizes.
- Chunk output ranges are consecutive prefix-sum ranges in descriptor order.
- The backend modifies no byte outside an item's exact output range.

One `GpuDecompressRequest` contains a non-empty tuple of items. Constructors
validate only backend-independent structure, exact-size equality, checked
aggregate arithmetic, and one homogeneous `(version, codec, framing,
transform)` across the request.

The request also caps total compressed bytes, total uncompressed bytes, record
count, and flattened compression-chunk count. These are separate concepts and
must be named separately in errors and metrics.

Empty records are valid items. They contribute no native chunks; an all-empty
request returns an already successful completion without calling a vendor API.

## Backend validation and capabilities

Backend and context validation cannot occur in request construction because a
request contains neither. The backend exposes:

```python
backend.validate_request(request, execution_context, execution_policy)
```

It validates:

- format/version, runtime, device, library, and execution-policy support;
- input/output/context device identity;
- owner types and resolved native ranges;
- actual native pointer alignment;
- input/output and output/output non-aliasing;
- chunk, record, aggregate-byte, and active-submission limits;
- backend- and engine-specific chunk-size limits.

Input/input overlap is permitted because inputs are read-only. Every output
range is disjoint from all input and output ranges.

`GpuDecompressCapabilities` is immutable and reports:

- backend and installed library versions plus stability level;
- supported `(record version, codec, framing, transform)` combinations;
- supported platform/runtime/device identities;
- maximum compressed and uncompressed chunk sizes;
- maximum compression chunks, records, and aggregate bytes per submission;
- maximum active submissions;
- input, output, and workspace alignment;
- asynchronous behavior and supported execution policies.

Limits are conservative values for the selected device, native library, and
execution engine. They are not universal constants copied from one vendor.

## Execution policy

The initial policies are:

```text
any_gpu_engine
software_gpu_required
fixed_function_required
```

nvCOMP may use CUDA kernels or Blackwell's fixed-function Decompression Engine.
Its default selection may fall back to CUDA kernels. When fixed-function work
is required, `NvcompBackend` selects the strict hardware mode and validates
device support, dynamically queried chunk limits, and allocation provenance for
every item. It fails instead of falling back.

hipCOMP currently uses HIP compute kernels and therefore does not advertise
`fixed_function_required`.

## Workspace ownership and concurrency

Native workspace is backend-owned because its layout and lifetime are
vendor-specific. `required_workspace_size()` remains a pure diagnostic and
planning operation; callers do not pass the returned allocation to
`submit_batch()`.

The backend reserves a distinct aligned workspace slot and native descriptor /
status arrays before its first enqueue. It never reuses them while their
completion is active. Allocation or backpressure failure occurs before native
submission.

Backend methods are thread-safe. Capabilities report the maximum active
submissions; exhausting it raises a deterministic busy error rather than
reusing active resources. `required_workspace_size()` is pure and reentrant.

## Backend and completion interfaces

The shared operation has this shape:

```python
backend.validate_request(request, execution_context, execution_policy)
workspace_size = backend.required_workspace_size(
    request,
    execution_context,
    execution_policy,
)
completion = backend.submit_batch(
    request,
    execution_context,
    execution_policy,
)
try:
    completion.wait_and_validate()
finally:
    completion.wait_and_discard()
```

`submit_batch()` repeats backend validation to close races between advisory
validation/sizing and submission.

Submission has a strict ownership boundary:

```text
submit raises  -> no device work remains; caller can immediately reuse buffers
submit returns -> completion controls lifetime until finalization
```

If an error occurs after any work may have been enqueued, the backend drains
that work and raises only afterward, or returns a terminal-failure completion.
It never raises while orphaned work can still access caller buffers.

`GpuDecompressCompletion` exposes:

```python
class GpuDecompressCompletion(Protocol):
    def query(self) -> bool: ...
    def wait_and_validate(self) -> None: ...
    def wait_and_discard(self) -> None: ...
```

Native work is not cancellable initially. `wait_and_discard()` drains work and
releases resources without making output usable. Every returned completion is
finalized in a `finally` block, including cancellation and validation failure.

`wait_and_validate()` finalizes and releases resources after reaching either a
stable success or stable validation failure. The `finally` call to
`wait_and_discard()` is an idempotent safety net for host interruption while a
wait is in progress.

The completion retains strong references to the request, all buffer owners,
the execution-context owner, backend, workspace, descriptors, status arrays,
actual-size arrays, and CRC arrays until finalization.

### Completion states

| State | `query()` | `wait_and_validate()` | `wait_and_discard()` |
|---|---|---|---|
| pending | `False` | waits, then validates | waits, then discards |
| device complete, unvalidated | `True` | validates | discards |
| validated success | `True` | returns again | no-op |
| validated failure | `True` | replays stable failure | no-op |
| discarded | `True` | raises discarded-result error | no-op |

`query()` never releases resources and does not imply usable output. Concurrent
waits produce one stable terminal state. Backend shutdown atomically stops new
submissions, finalizes active work, and then releases resources; repeated
shutdown is safe.

## Submission and visibility flow

```text
1. Load the exact record and verify header plus compressed-payload checksums.
2. Enqueue an exact-length H2D copy on the selected execution context.
3. Build a homogeneous request from headers, buffers, and logical KV sizes.
4. Resolve owners and validate devices, ranges, aliasing, alignment, and limits.
5. Reserve workspace and native metadata before the first enqueue.
6. Upload descriptors and submit all non-empty chunk streams on that context.
7. Compute output CRCs on the same context.
8. Make status, actual-size, and CRC results host-readable.
9. Record completion only after every validation result is ready.
10. Validate every status, output size, and CRC.
11. Only then enqueue or expose paged-KV placement.
```

The initial implementation may synchronize the host during validation for
correctness. A later implementation may make validation stream-ordered, but it
cannot weaken the visibility rule.

## Failure contract

Failures before submission enqueue no work. Failures after ownership transfer
are represented or drained before caller resources become reusable.

Typed failures distinguish invalid requests, unsupported formats or policies,
backpressure/submission failures, native execution failures, and validation
failures.

Any item failure invalidates the complete staging request. Failed or discarded
output is indeterminate and never placed. CPU fallback is permitted only
through an explicit policy outside this backend interface.

A `GpuDecompressRequest` corresponds to one reuse-safe staging batch, not an
entire retrieve. A retrieve can contain several staging batches and a mixture
of raw and compressed records. If a later batch fails after earlier placement,
the overall retrieve returns failure and its exported completion event does not
authorize use of any partially written blocks; the caller recomputes. This is
logical atomicity, not physical rollback of prior device writes.

## L1 and L2 integration prerequisites

These changes are later milestones, but their contracts must exist before the
backend is connected to production.

### Variable-size loads

An L2 load destination is a capacity, not a required exact stored length. An
adapter succeeds when the stored object fits, writes the exact bytes, and calls
`MemoryObj.set_used_size(actual_bytes)`. It fails without truncation when the
stored object exceeds physical capacity. Existing fixed-size loads continue to
set an actual size equal to capacity.

This preserves the existing load bitmap while allowing a serde wrapper to
allocate an upper bound, load a smaller compressed record, and then parse only
its exact used bytes. Adapters such as filesystem storage that currently
require file size to equal destination length must adopt the capacity/actual
size rule.

### Representation identity

L1 state gains a public representation value, initially:

```text
raw_kv
portable_compressed_record
```

Representation metadata is stored with the L1 object state and exposed through
a public accessor; retrieve code never reads private metadata. Raw and
compressed values do not coexist under one logical `ObjectKey` initially. An
existing raw L1 hit wins; an object loaded from compressed L2 is tagged as a
portable record before becoming read-ready.

The record becomes read-ready only after its exact length, header, and
compressed-payload checksum are validated. A failed record is removed. Existing
read locks, eviction, cancellation, and stream-ordered release semantics remain
unchanged.

## Relationship to existing LMCache components

- `GPUCacheContext` already owns an opaque stream and flat per-object-group
  staging tensors. The output buffer wrapper derives from those tensor views.
- A later context-owned compressed-input pool supplies exact record-sized input
  ranges; output reuses existing raw-KV-sized staging.
- `SerdeL2AdapterWrapper` later preserves a validated portable record in L1
  instead of always CPU-materializing it.
- The multiprocess retrieve path classifies raw and deferred representations,
  orders compressed H2D work, invokes the backend, validates completion, and
  then uses the existing paged-placement operation.
- Existing stream callbacks retain and release L1 locks only after all
  decompression, validation, and placement work is ordered correctly.

## First implementation slice

With the wire integrity/alignment update complete, the interface slice is:

1. Add immutable platform identity, buffer, and execution-context adapters.
2. Add execution policy, capabilities, item, request, completion, typed errors,
   and backend ABC.
3. Add fake CUDA and HIP backends using only shared types.
4. Test intrinsic versus backend validation, owner-derived capacity/alignment,
   aliasing, device/context mismatch, homogeneous batching, exact logical size,
   output CRC mismatch, partial enqueue, finalization, concurrency, workspace
   isolation, shutdown races, and stable failure replay.
5. Keep real vendor imports and production call sites absent.

## Vendor constraints informing the contract

- nvCOMP warns that corrupt Deflate/Gzip input can produce undefined behavior;
  this motivates pre-native integrity and trust validation.
- nvCOMP raw Deflate currently requires four-byte alignment, motivating the
  stricter portable 16-byte policy.
- nvCOMP's default engine may fall back from fixed-function hardware to CUDA
  kernels, motivating explicit execution policy.
- hipCOMP 2.3 has compatible batched asynchronous Deflate/Gzip concepts but
  labels them experimental and unsuitable for production workloads. Initial
  HIP support remains feature-gated and reports that stability in capabilities.

Primary references:

- [nvCOMP Deflate/Gzip C API](https://docs.nvidia.com/cuda/archive/13.1.0/nvcomp/c_api.html)
- [NVIDIA Decompression Engine FAQ](https://docs.nvidia.com/cuda/nvcomp/decompression_engine_faq.html)
- [hipCOMP 2.3 algorithm support](https://github.com/ROCm/hipCOMP-core/blob/22cc762f54fba7cdfca74a4c50c00f2aac4ace7a/README.md#algorithm-support)
- [hipCOMP Deflate API](https://github.com/ROCm/hipCOMP-core/blob/22cc762f54fba7cdfca74a4c50c00f2aac4ace7a/include/hipcomp/deflate.h)
- [hipCOMP Gzip API](https://github.com/ROCm/hipCOMP-core/blob/22cc762f54fba7cdfca74a4c50c00f2aac4ace7a/include/hipcomp/gzip.h)

## Decisions deferred until compatibility probes

- Raw Deflate versus Gzip as the production framing.
- Production compression chunk size, subject to the 16-byte alignment policy.
- Initial CPU/QAT/IAA store-side encoder.
- Pinned nvCOMP and hipCOMP versions and exact common capability set.
- Whether the first AMD deliverable is interface conformance, an experimental
  backend, or complete hardware end-to-end coverage.

These choices do not weaken the integrity, ownership, submission, or completion
semantics above.
