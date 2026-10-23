# Video Ingestion and Frame Sampling

A CPU-first Python CLI for importing local video, recovering completed work after
process interruption, and extracting frames tied to source bytes and exact
presentation timestamps. Portfolio v0.1 covers the ingestion and frame tooling in
the unfinished Open-World Video Intelligence project.

## Quickstart

Requirements: Python 3.11 or 3.12, and existing `ffmpeg` / `ffprobe` executables on
`PATH`. The demo uses NumPy, OpenCV, PyAV and the project's core dependencies.
From the repository root, in your chosen Python environment:

```sh
python -m pip install -e ".[video]"
python scripts/fixture_ingest_demo.py --output artifacts/portfolio-demo
```

Installation is a setup step; the demo itself installs nothing and downloads no
models or videos. It creates a one-second **synthetic contract fixture**, imports
it in reference and managed modes, deliberately exits a child process after the
copy checkpoint, and resumes the job in a new process after its lease expires.
It then checks both frame-selection policies against sequential exact-PTS/RGB
decoding and saves one lossless PNG.

Expected final output: `status: passed`, with paths to `report.json` and `frame.png`.
The report retains every child command and exit code, including the intentional
`77` interruption. Use a fresh output directory for another run: existing output
is never overwritten. Generated media, databases and reports stay under
`artifacts/`. Reports contain local paths; keep them out of the public source slice.

Excerpt from the verified synthetic demo report:

```json
{
  "data_kind": "synthetic",
  "status": "passed",
  "checks": {"one_final_copy": true, "resumed_copy_checkpoint": true},
  "frames": {
    "exact_frame_matches": [true, true, true, true],
    "png_rgb_matches_cli": true
  }
}
```

For your own local video, use `import-video --mode reference` to keep the original
location or `--mode managed --managed-root ...` to create a managed copy.
`import-status` reports persisted state; `import-resume` revalidates bytes before
reuse. `inspect-frames` returns actual PTS, source/RGB hashes and seek error.
Run `python -m video_intelligence.cli <command> --help` for options.
See [local CLI examples and limits](docs/LOCAL_USAGE.md) for reference/managed import
and resume commands.

## Implementation

```text
local video → source identity → reference / managed copy → probe checkpoint
            → leased SQLite job → resume → exact-PTS frame selection → RGB / PNG
```

- Managed objects are content-addressed. No-clobber hard-link publication and
  lease fencing prevent partial or stale work from becoming a ready import.
- Registration, the final receipt and job success commit in one SQLite transaction.
  Resume verifies completed stages; invalid checkpoints are recomputed or rejected.
- Probe identity includes the pipeline/profile, implementation and FFprobe binary.
  A probe change does not require a second copy of unchanged source bytes.
- Frame selection supports `at_or_after` and `nearest`, using container PTS rather
  than nominal FPS. The public inspection command has a process deadline for codec hangs.

## Verification

The synthetic demo above checked four frame selections against a ten-frame
sequential reference and verified lossless PNG RGB. Its generated fixture is
marked `synthetic` / `contract`; no real-world quality metric is computed.

The prior local ingestion checkpoint passed **42 targeted CPU tests** across import,
worker, storage and affected observation contracts. Separate prior real-video checks
confirmed a byte-equal managed object and process restart without duplicate
registration. The decoder checkpoint matched 18 real-source requests to sequential
PTS/RGB references, plus eight CLI selections across both policies. The original
real video and raw local receipts are not distributed in this public slice.

Focused tests are in `tests/test_video_import.py`, `tests/test_decoder_forward.py`
and `tests/test_source_frames.py` (the `dev` extra is needed to run pytest).
Mock adapters in the wider contract tests are test doubles, not learned inference.
These results do not establish model accuracy, throughput or a passing full suite.

## Known limitations

- This slice does not provide a qualified learned video-search or semantic-QA product.
  Known false support remains unresolved; detection/QA quality and benchmark results
  are not claimed. Research model adapters are outside this demo.
- Crash recovery covers copy/probe. Shots, sampling, embeddings and index stage
  resume remain on the roadmap. Upload API and a polished application UI are not included.
- Fresh-prefix decoding can cost more than indexed seeking. Long-video throughput
  and host power-loss durability are unqualified; direct decoder time checks are
  cooperative, while `inspect-frames` uses an outer process deadline.
- The filesystem must support the existing atomic hard-link publication contract.
  Unsupported or denied publication fails explicitly; there is no overwrite fallback.
- Validation used an existing local dependency environment. Fresh installation,
  broad OS compatibility and the full repository quality gate are not qualified here.

The internal implementation roadmap and historical development evidence are
retained separately and are not included in this public source slice. The larger
project remains unfinished.

## Development and license status

Implementation and documentation used AI assistance. No root `LICENSE` file or
project license declaration is present in this checkout. The maintainer must
select the project license and confirm the rights and third-party notices for the
chosen public files; this preparation does not assign a license. Dependency and
FFmpeg distributions are not bundled and retain their own terms.
Existing upstream references are retained in [third-party notices](THIRD_PARTY_NOTICES.md).
