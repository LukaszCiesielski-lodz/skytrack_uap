# skyhunt

[Polski](README.md) | **English**

## From the author

I write this code for all sky observers, not just for myself. I record the sky with an ordinary camera and a fast lens, and I want to know what really crossed the frame: which satellite it was (with its NORAD number), what is a meteor, what is an aircraft, and what remains unexplained. Everything should be computed and verifiable, with no guessing. If you have a camera, a tripod and some patience, you can do the same.

The code is open (MIT license): use it, change it, build on it. **I have one request: if you use this project in your observations, publications, videos or in your own code, please mention it** and link to the repository: <https://github.com/LukaszCiesielski-lodz/skytrack_uap>. I'd also be glad to see your results. Bug reports, ideas and fixes (issues, pull requests) are welcome.

## What it does

A pipeline for detecting **all** moving objects in 4K sky video, including those at the edge of the noise. Every object is measured and then, where possible, explained: satellite (TLE, NORAD ID), meteor, aircraft, or a nearby object (bird, bat, insect). Only what remains after rejecting the known classes is flagged as an anomaly, and only by criteria frozen before the analysis.

Full specification: [docs/HANDOFF_skyhunt.md](docs/HANDOFF_skyhunt.md) (in Polish). CPU baseline (results reference only): [baseline/skytracks.py](baseline/skytracks.py).

## Status

| Milestone | Scope | State |
|---|---|---|
| M0 | repo, config, Colab notebook, GPU decoding with benchmarks, manifest and resuming | works on Colab (L4) |
| Report | per-frame detector, tracks, plate solve, time sync from satellites, NORAD, PDFs with constellations and clips | works; first results below |
| M3 | GPU shift-and-stack (faint objects), FAR from shuffling, injection–recovery | – |
| M4 | non-linear tracks, distance from defocus blur, biological classes, meteors, aircraft (ADS-B) | – |
| M5 | anomaly scoring | – |

### First results: `DSCF4641.MOV` (27.09.2026, Cygnus at the zenith, 320 s)

| | |
|---|---|
| Plate solve | 5/5 epochs; field 25.8° × 14.7°, 24.7″/px, no 4K crop; epoch agreement 0.44 px |
| Detection | 4.04 million detections, 359 tracks; NVDEC ~150 fps (stack), ~50–60 fps (detection) |
| Clock correction | Δ = +26.85 ± 0.14 s, consistent across 24 satellite tracks (the camera clock was 9 min 33 s fast) |
| Catalog | 31,354 objects (CelesTrak + Space-Track) |
| Identified | 36 tracks, incl. Starlink, Kuiper, Hulianwang, Globalstar; the brightest object in the recording is STARLINK-2112 (NORAD 47391) |
| Position from satellite parallax | 0.24 km from the entered coordinates |
| Dark recording | ~112 false tracks per hour, 6 hot pixels |

Lesson from this recording: the first run gave 0 identifications because the config held a location ~2.2 km away from the real one. Satellites at ~500 km were shifted by ~0.25° as a result (parallax). That is why coordinates are now entered separately for each recording, and the pipeline checks them itself from parallax.

## Report

For each sky recording, in `out/<file>/report/`:

| file | contents |
|---|---|
| `summary.pdf` | full frame with constellations and all tracks; time correction and its agreement between satellites; track table; satellites predicted in the frame but not detected; FAR from the dark recording |
| `objects/sat_<NORAD>_t<id>.pdf` | identified satellite: star background, constellations (e.g. Cygnus), star names, track in green with ticks every 1 s, prediction from orbital elements; NORAD, name, COSPAR, range, height, illumination; page 2: strip of 12 frames from ≥ 1 s of video; MP4 clip attached to the PDF |
| `objects/unid_t<id>.pdf` | unidentified object: track **in red** on the star background; angular speed [°/s], transit duration, start and end in UTC (± time-correction uncertainty), RA/Dec and Az/Alt, class hint, unchecked hypotheses |
| `clips/t<id>.mp4` | clip ≥ 1 s around the track, object marked with a circle |

For color recordings the object PDF also has a **color** page (below), and `summary.pdf` has a color calibration page and a "color" column in the track table.

### Color

The `color` stage measures the color of every track.
- **Calibration:** once per minute of recording (3 epochs) I measure the color of catalog stars with a known B−V color index. They give a "stellar locus" on the log(R/G) × log(B/G) diagram. This removes the influence of white balance, film simulation and sky glow, and as a side effect reveals white-balance drift during the recording.
- **Object:** in each frame I subtract the background taken from frames in which the object has already moved away (stars and sky glow cancel out). Only non-saturated frames are used.
- **Result:** color temperature `T_eq` (position along the stellar locus), **green excess** (distance from the locus) and color change over time. Files: `color_calib.json`, `track_color.csv`, `track_color_points.parquet`.
- **Check:** identified satellites are reflected sunlight (B−V ≈ 0.6–0.9). Their mean color is the reference for other tracks.
- **Weak calibration:** when the stellar locus is nearly flat (slope < `color.min_locus_slope`, expected ~0.3 dex per 1 mag B−V), the codec has crushed the color of small points. The report then gives no kelvins, only "warmer/cooler than the satellites by X dex" (`d_sun_dex`). This happened in DSCF4651 (0.048 dex/mag). A temperature outside the B−V range −0.4…2.5 is shown as a limit, e.g. "T ≤ 2725 K (off scale)".

Hints (hypotheses, always with numbers):

| object | color | hint |
|---|---|---|
| meteor | green excess | Mg 517 nm / O 557.7 nm? |
| meteor | warm (< 3500 K) | Na/Fe? |
| meteor | hot (> 8000 K) | fast, Ca/Mg? |
| other | like satellites | sunlit |
| other | warm (< 3000 K) | city light (sodium)? |
| other | jumping R/G | navigation lights? |

**What RGB cannot tell you:** the camera has three broad bands, so the composition of a meteor (the ratios of the Na / Mg / Fe lines) cannot be read from it. That requires a diffraction grating in front of the lens (500–1000 lines/mm film). The meteor's spectrum then appears next to it.

**Camera settings for color:** fixed white balance (daylight or 5500 K, not auto), Standard/Provia film simulation without Color Chrome, **Color +4** (boosts chroma before H.264 crushes it; the star calibration takes it into account). Bright objects are saturated and have no color; the PDF states how many frames were rejected. Black-and-white recordings are detected and skipped.

### RAW photo sessions (in development)

Instead of video you can process RAW photo series (Fujifilm RAF) with AE bracketing: **one subfolder in `raw/` = one session** (e.g. `raw/deneb_0210/`), results in `out/<folder>/`. Dark: a folder with "dark" in its name. The observing site and hint star are entered in the "Nagrania" cell under the folder name.
- **Done (F1):** EXIF and sequence number (Fuji MakerNote), intervalometer cadence from whole-second EXIF times, plate solving, a deep stack of each exposure class (0, +1, −1 EV) aligned for sky rotation, a report with the map and the session timeline.
- **Next stages:** satellites as streaks with precise times and NORAD IDs (F2), brightness and glints along the streak, color (F3), asteroids in the stack (F4).
- **Camera settings (X-E3):** M, f/1.0, ISO 800, electronic shutter, RAW only (lossless compressed), DR100, WB 5600 K, AE BKT ±1 EV (1/2 s, 1 s, 1/4 s), long-exposure NR off, intervalometer.
- File diagnostics before the first session: [colab/raw_diagnostics.ipynb](colab/raw_diagnostics.ipynb).

### Asteroids, comets and NEOs

The `smallbodies` stage checks which known small bodies were in the frame and measures them in the recording.
- **List:** JPL Small-Body Identification API (`sb_ident`), positions from numerical orbit integration, V magnitude and motion in ″/h. A separate query for NEOs, including fainter ones. The observer position sent to JPL is rounded to 0.1° (~10 km): for an NEO at 0.01 au this changes the position by ~1″, and the exact location stays private.
- **Why not a track:** a main-belt asteroid moves ~0.1 px in 5 minutes, so in the video it looks like a star. The track detector needs ≥ ~0.1 px per frame, i.e. tens of thousands of ″/h. Only an NEO passing very close to Earth can produce a track; the report then labels the track with its name.
- **Measurement:** a stack of all frames in a small window that moves with the sky and the object. Noise drops as √N, so the reach is several magnitudes deeper than a single frame. The photometric zero point comes from catalog stars in the same stack, corrected for vignetting.
- **Verdict:** "wykryta" (detected) = SNR ≥ 5 at the predicted position, brightness consistent with the prediction (±1 mag) and no brighter background star (Gaia DR3 from VizieR) in the aperture. Otherwise "zlewa się z gwiazdą" (blended with a star), "za słaba (zasięg X mag)" (too faint, reach X mag) or "niewykryta" (not detected).
- **Report:** pink diamonds on the map (filled = detected), a table with predicted and measured magnitude, reach, motion and O−C offset, stack thumbnails. Files: `smallbodies.csv`, `smallbodies.json`, `smallbodies_tracks.csv`.

### `tracks_final.csv`: table of all tracks

The file is in `out/<file>/tracks_final.csv`, one row per track. The same table is next to it as `.parquet`. This is the most convenient place to find a specific object before opening the PDFs.

**Opening.**
- Colab: `pandas.read_csv`, examples below.
- Google Sheets / Excel: numbers are written with a decimal point. If your locale uses a decimal comma, first set the spreadsheet locale to "United States" (Sheets: File → Settings) or import with the `,` separator and English settings. Otherwise `0.771` turns into a date or text.

**Main columns.**

| column | meaning |
|---|---|
| `track_id` | track number, the same as in file names: `objects/sat_<NORAD>_t<id>.pdf`, `objects/unid_t<id>.pdf`, `clips/t<id>.mp4` |
| `kind` | `sat` (identified satellite) or `unid` (unidentified) |
| `norad`, `sat_name`, `confidence`, `match_reason` | identification: NORAD number, name, confidence (`high`/`medium`/`low`), reasoning |
| `tau0`, `tau1`, `dur_s` | start and end in seconds from the start of the video (≈ player counter) and duration |
| `utc_start`, `utc_end` | start and end in UTC, already corrected with the satellite clock correction |
| `omega_deg_s` | angular speed [°/s]; LEO overhead is ~0.5–1.1 °/s |
| `curv_arcsec` | deviation of the track from a great circle [″]; satellites usually < 20″ |
| `ra0`, `dec0`, `ra1`, `dec1` / `az0`, `alt0`, `az1`, `alt1` | position on the sky at the start and end of the track (ICRS RA/Dec and azimuth/altitude) |
| `n` | number of track points (frames with a detection) |
| `peak_snr_median` | brightness as SNR; the detection threshold is 5, clear objects > 15, saturated > 50 |
| `speed_px_frame`, `x0`, `y0`, `x1`, `y1` | motion and position in pixels (3840×2160 frame) |
| `cross_ratio` | object width ÷ star width; > 1.8 means out of focus (nearby), unless the object is very bright |
| `f_peak_hz`, `f_alias_hz`, `f_power` | brightness modulation (flashes, rotation); at 24 fps `f_peak_hz` cannot be distinguished from `f_alias_hz` |
| `starts_inside`, `ends_inside` | `False` means the object enters or leaves through the frame edge; `True` at the end of the track means it fades inside the frame (e.g. entering Earth's shadow) |
| `class_hint`, `class_reason` | class hint for unidentified tracks: `satelita?` (satellite?), `meteor?`, `samolot?` (aircraft?), `bliski obiekt?` (nearby object?) |
| `along_sigma_px`, `streak_ratio` | shape of the trace within a frame: elongation along the motion (meteor streak) and its ratio to the cross-track width |
| `flock_n` | number of parallel tracks with similar speed (bird migration); 0 = no group |

Track colors are in a separate file `track_color.csv` (same `track_id`): `T_eq_K`, `bv_eq`, `e_bv_eq`, `green_excess`, `d_sun_dex` (color relative to the satellites, "+" = warmer), `color_hint`, `n_color`, `n_saturated`.

**Examples (Colab cell).**

```python
import pandas as pd
t = pd.read_csv(f'{OUT}/DSCF4641/tracks_final.csv')

# identified satellites in order of appearance
t[t.kind == 'sat'].sort_values('tau0')[['track_id', 'tau0', 'utc_start', 'norad', 'sat_name', 'omega_deg_s']]

# what was in the frame at second 43 of the video
T = 43
t[(t.tau0 <= T + 2) & (t.tau1 >= T - 2)]

# unidentified tracks worth a look: longer tracks, brightest first
t[(t.kind == 'unid') & (t.n >= 15)].sort_values('peak_snr_median', ascending=False)
```

**Noise.** A track with `n` ≤ 8, `peak_snr_median` ≈ 5–6 and jumps of 25–48 px per frame is almost certainly noise detections linked by chance, not an object. In `DSCF4641` that is ~250 of 323 unidentified tracks.

### Verification and reporting observations

**Checking an identification.** In a `sat_…` PDF the green track (measurement) should overlap the dashed prediction from the orbital elements. The table next to it shows the cross-track residual (tens of ″), δ_j − Δ (< ~1 s), agreement of speed and direction, and solar illumination. A satellite "in Earth's shadow" would be invisible, so such a match is suspicious. You can check independently in Stellarium (Satellites plugin): just set the location and the time `utc_start` from `tracks_final.csv`.

**Satellites outside the public catalogs.** Besides CelesTrak and Space-Track, the pipeline downloads the `classfd` catalog (Mike McCants, <https://mmccants.org/tles/>). These are elements of satellites, mostly military, tracked by an amateur network of observers. The elements can be many days old, so a match to them is only a `low`-confidence candidate, shown in the unidentified object's PDF.

**Reporting.** Space-Track does not accept observations from amateurs. Satellite positions are reported to the observer community (SeeSat-L list, <https://www.satobs.org>) in IOD format. The pipeline writes them to `report/iod.txt`:
- 3 positions each (start, middle, end of track) for identified satellites and for unidentified tracks that are long, nearly straight and clearly above the noise (`iod` section in the config);
- J2000 RA/Dec in format 1, time uncertainty from the synchronization, position uncertainty from plate-solve epoch agreement.

Before sending:
- ask on SeeSat-L for a station number and enter it in `iod.station`, because 9999 means unassigned;
- state in the report that the time is calibrated on catalog satellites, not from GPS.

**Time synchronization.** Track positions on the sky are computed from the plate solve and a fixed-camera model: pixel ↔ fixed Alt/Az direction. They do not depend on the clock error. Straight tracks with LEO speeds are compared with passes from orbital elements (SGP4) in a ±5σ window around the metadata time and, if that is not enough, within ±2 h.

- The correction Δ is the **median δ of the agreeing satellites**: each satellite counts once, and amateur elements (classfd) are skipped when there are at least 3 public ones. The report still shows the first identified satellite as the reference.
- Previously Δ came from the first satellite. Two parts of one recording (DSCF4647/4648, same camera clock) then differed by ≥ 0.3 s, because the orbital-element error of a single Starlink propagated to the whole time base. Old mode: `identify.reference: first`.
- Δ is accepted when at least 2 independent tracks give consistent δ (±1 s). A single match gets `low` confidence, because parallel Starlink shells are easy to confuse.
- The remaining satellites are identified with Δ fixed, and the scatter of their δ is reported.

**Orbital elements.** CelesTrak only provides current elements, so the snapshot has to be frozen shortly after the recording (a notebook cell). It is saved in `cache/gp/` on Drive, and a copy of the elements used goes to `gp_elements.csv` next to the results. The format is CSV/OMM, because NORAD numbers ≥ 100000 do not fit into TLE. Space-Track (element history, full catalog including rocket bodies and debris) is optional: you enter the login in Colab Secrets.

Constellation and star-name data: [d3-celestial](https://github.com/ofrohn/d3-celestial) (BSD-3, © Olaf Frohn), in `skyhunt/data/d3celestial/`.

## Data

| | |
|---|---|
| Camera | Fujifilm X-E3, Fujinon XF 50mm F1.0; manual focus, just short of ∞ |
| Video | 3840×2160, H.264 in MOV, **no B-frames**. DSCF4641: 24000/1001 fps, shutter 1/24 s, B&W. Since 28.09.2026: 30000/1001 fps, shutter 1/30 s, ISO auto, color (Standard film simulation). fps and frame time are read from the file |
| Location | **separately for each recording**: the "Miejsce obserwacji" (observing location) notebook cell writes it to `MyDrive/skyhunt/sites.yaml` (outside the repo). The `site` section in `config.yaml` is only a default (Łódź). An error of ~2 km breaks satellite identification; the pipeline warns when parallax indicates an offset > 0.3 km |
| `DSCF4641.MOV` | 320 s, start 2026-09-27 18:26:35 UTC (after synchronization; the camera clock was 9 min 33 s fast); brightest star: Deneb |
| `dark_frames.MOV` | 320 s, lens capped, 2026-09-28 (after clock correction); for FAR and the hot-pixel map |

The metadata time is only a starting point (±60 s). The final clock correction comes from fitting satellite passes (M2). The Fuji EXIF date in `udta` is the start of the recording; `mvhd.creation_time` falls ~24 s after its end (checked on both files).

## Running (Colab)

Open [colab/run_skyhunt.ipynb](colab/run_skyhunt.ipynb) in Colab (A100 or L4 GPU) and run the cells from the top. Keep recordings in `MyDrive/skyhunt/raw/`; results go to `MyDrive/skyhunt/out/<file_name>/`.

For each new observation:
1. Enter its coordinates in the "Miejsce obserwacji" (observing location) cell.
2. Shortly after recording, run the "Snapshot elementów orbit" (orbital elements snapshot) cell (CelesTrak only has current elements).
3. Optionally add your Space-Track login in Colab Secrets (`SPACETRACK_USER`, `SPACETRACK_PASSWORD`).

Do not save the notebook with coordinates filled in back to a public repo.

CLI:

```bash
skyhunt probe RAW_DIR                      # metadata + initial time (no writes)
skyhunt bench-decode FILE --out OUT/_bench # decoding backend throughput
skyhunt run [RAW_DIR] [--out OUT]          # pipeline with resuming
skyhunt run FILE --stages stack --force stack
skyhunt status [RAW_DIR]                   # stage status from manifests
```

Every command accepts `--config` and any number of `--set key=value`, e.g. `--set decode.batch_frames=16`.

## Configuration and resuming

- `config.yaml` holds all parameters. The `files` section stores per-file overrides, e.g. a different clock correction for an older recording, or `role: dark`.
- Each stage declares which config sections it depends on. Its hash is computed from those sections, the stage's `rev` number and the hashes of the required stages.
- The per-file `manifest.json` records for each stage: status, hash, code version (including the git commit), timings, output files and metrics. A stage is skipped if it has status `done` with the same hash and its files exist.
- A config change recomputes only the dependent stages. A change to a stage's logic requires bumping `rev`.
- All JSON writes are atomic (temporary file + `os.replace`), so an interruption does not corrupt the manifest.

## Decoding

We work on the Y channel (luminance) at full resolution. Backends in the order tried:

| backend | where it decodes | Y | notes |
|---|---|---|---|
| `nvcodec` | NVDEC (PyNVVideoCodec) | exact | NV12 in GPU memory, we take the Y plane |
| `torchcodec` | NVDEC | ±1 DN | returns RGB; benchmark only, because `exact_luma_required: true` |
| `torchaudio` | NVDEC (`h264_cuvid`) | exact | `torchaudio.io` is going away in newer torchaudio versions |
| `pyav` | CPU, producer thread | exact | transfer to GPU via pinned memory |
| `ffmpeg` | CPU, subprocess | exact | `extractplanes=y`, dependency-free fallback |

Backend parity is checked by tests (`tests/test_decode.py`): on a real recording the "exact" backends must give Y bit-identical to PyAV.

### Throughput

Colab, L4 GPU, 4K H.264: `nvcodec` about 150–165 fps in the `stack` stage; detection with background, filtering and labeling on the GPU about 50–65 fps (320 s of video in ~2.5 min). Details: `skyhunt bench-decode`, output in `out/_bench/decode_bench.md`.

## M0 stage outputs

| file | description |
|---|---|
| `meta.json` | container metadata (GOP, keyframes, B-frames), time prior with candidates from all sources |
| `stack_mean.npy/.png` | mean of the whole recording (for plate solving in M2) |
| `stack_max.npy`, `stack_max_minus_mean.png` | maximum and max−mean (track preview) |
| `frame_stats.csv` | per frame: time, I/P, mean, median, p99.9 |

The `keyframe_pulse_dn` metric in the manifest is the mean brightness difference of I-frames relative to neighboring P-frames. A value clearly different from zero means pulsing every GOP, which has to be masked (section 5.3 of the handoff).

## Tests

```bash
python -m pytest -q
```

Tests marked `gpu` require CUDA. `video` tests require the `SKYHUNT_TEST_VIDEO` variable pointing to a real recording. The notebook sets it automatically.

## Interpretation rules

- The "anomaly" criteria are frozen in `config.yaml` before processing the full dataset. Changing the criteria requires a new version and recomputing everything.
- A single camera gives neither distance nor linear speed. The exception is the estimate from defocus blur for nearby objects. The report never gives km/s for in-focus objects without triangulation.
- Sensitivity without false-alarm control is worthless. The report always gives the detection limit (injection–recovery) and the expected number of false detections per hour of recording.
- Hardware recommendations:
  - a second camera 5–20 m away (parallax immediately tells whether an object is near or far),
  - 1080p/60 fps recordings alongside 4K/24 fps (resolves wing-beat aliasing),
  - periodically 1–2 min with the lens capped, preferably at night, at a temperature close to the session's.
