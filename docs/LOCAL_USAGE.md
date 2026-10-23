# Local video CLI

Run from the source root after installing the core and `video` dependencies in
your chosen Python environment. FFprobe must already be on `PATH`. These commands
configure the existing importer through CLI flags; no YAML file is auto-loaded.

For the verified synthetic workflow, use the README quickstart. To import your
own video, replace `INPUT.mp4` below with its local path and use a new output area:

```sh
python -m video_intelligence.cli import-video --video INPUT.mp4 --db artifacts/local/state.sqlite --video-id my_video --mode reference
```

Reference mode does not copy the input. For a managed copy, choose another video
ID and an explicit managed root:

```sh
python -m video_intelligence.cli import-video --video INPUT.mp4 --db artifacts/local/state.sqlite --video-id my_managed_video --mode managed --managed-root artifacts/local/managed
```

Keep the `job_id` returned by the importer. Replace `JOB_ID` with that value:

```sh
python -m video_intelligence.cli import-status --db artifacts/local/state.sqlite --job-id JOB_ID
python -m video_intelligence.cli import-resume --db artifacts/local/state.sqlite --job-id JOB_ID
```

Status describes historical checkpoints; resume revalidates the current bytes.
A live lease cannot be stolen. After an interrupted process, wait for its lease
to expire before resuming. Invalid input identity is rejected. Failed partials
are retained and are not ready imports; no cleanup is performed by this guide.

To inspect frames, replace `VIDEO_PATH` with the `video.source_path` in the import
receipt. The timestamp is relative to the first displayed frame:

```sh
python -m video_intelligence.cli inspect-frames --video VIDEO_PATH --timestamp 0.31 --selection-policy at_or_after --max-decode-frames 1000 --timeout-seconds 10
```

`nearest` is the other selection policy. The command returns exact PTS, source
and RGB hashes; an out-of-range request fails without an accepted partial list.
The synthetic demo also saves a PNG. Import does not run a learned model.

Default import limits are 512 MiB input, 60 seconds per attempt, 64 MiB free-space
reserve, 2,073,600 pixels and one hour duration. Use `import-video --help` for
explicit overrides. Immutable publication requires a filesystem supporting the
existing hard-link contract; a denied publication has no overwrite fallback.

The examples use placeholders for user-owned input. They are not additional
real-video or quality-evaluation results. Generated outputs may contain local
paths; keep them out of a public source distribution.
