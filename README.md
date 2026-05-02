# Forward-Looking Imaging Sonar simulator

A physics-driven Python prototype of a forward-looking imaging sonar (FLS) — the kind of sensor that produces a polar fan image. Defaults are sized for a 120 kHz system (e.g. Kongsberg Mesotech 1071, Tritech SeaKing long-range) targeting ~300 m, but the same pipeline applies at higher frequencies for shorter-range / higher-resolution systems like Tritech Gemini, Sound Metrics ARIS, or BlueView. Aimed at accuracy over realtime, with a clear path to plug into your Unreal Engine pipeline.

## Files

| file | what it is |
|---|---|
| `sonar_sim.py` | Library: config, acoustics, ray sampling, pipeline, fan renderer, UE adapter |
| `demo_synthetic.py` | Standalone demo with an analytical scene (no UE needed) |
| `fls_demo.png` | 4-panel debug output (B-scan, fan, per-ray depth, cos θ) |
| `fls_fan.png` | Clean fan-display output |

Run `python demo_synthetic.py` to reproduce the images.

---

## Why a depth-image approach alone doesn't look like real sonar

Your UE setup is doing the right *first* step (you've got a depth render giving you per-pixel range), but a sonar image is fundamentally different from an optical image, in ways that aren't a postprocess shader you can bolt on. The depth image gives you geometry; sonar imagery is geometry **+ acoustics + sampling**.

Here's the gap, in roughly the order it matters for "looking right":

**1. Sonar is polar, not perspective.** A sonar image is sampled in `(range, azimuth)`, not `(x, y)` of a perspective camera. Two pixels on the same horizontal screen line in a depth image can be at very different ranges; conversely, points at very different elevations can land in the *same* range bin. So before anything else, you have to resample your depth/normal buffers into `(range, azimuth, elevation)` space.

**2. Elevation collapses.** This is the big one. A real FLS has a wide vertical beam (5°–20°). All the energy at all elevations within that beam, at the same slant range, sums into one range cell. That's why a small object on the seafloor and the seafloor under it can be indistinguishable in range — a depth image keeps them separate, the sonar can't. Modelling this requires sampling many elevation rays per beam and integrating, not just reading one depth value per beam.

**3. Brightness is angle-dependent (BRDF), not constant.** In a depth image, every pixel is "lit" the same. In sonar, backscatter strength depends strongly on the angle between the surface normal and the ray. Surfaces head-on to the sonar are bright; grazing surfaces are dark. The most-cited model in the FLS-sim literature (Cerqueira et al., KTH/Bjork & Folkesson) is a **Lambertian²** lobe (`cos²θ`) with an optional specular component for hard targets. This is what makes wreck faces flash and seafloor look smooth-textured.

**4. Speckle, not Gaussian noise.** The grainy texture of a sonar image is *coherent speckle* — a multiplicative random process with a Gamma/Rayleigh distribution, not additive Gaussian noise. You see it because each resolution cell is the coherent sum of many scatterers with random phase. If you add Gaussian noise on top of a clean depth image you get a noisy depth image, not a sonar image.

**5. Two-way transmission loss.** Sound at 120 kHz attenuates ~44 dB/km in seawater (Francois–Garrison formula), plus geometric spreading. So 300 m is ~26 dB of two-way absorption *plus* `40 log₁₀(300) ≈ 99 dB` of point-target spreading — operators counter this with TVG (time-varied gain). You need both pieces, otherwise far-range looks wrong. (Going to 400 kHz roughly triples the absorption rate, which is why high-frequency systems are short-range.)

**6. Beam pattern.** Each beam isn't a knife-edge — it has a main lobe (Gaussian-ish, ~3° wide at 120 kHz for typical apertures) and sidelobes (`sinc²`). Bright targets bleed into adjacent beams. Skipping this makes objects look unnaturally crisp.

**7. Pulse smearing in range.** Range resolution is `c·τ/2`. A bright target spans more than one range bin because the pulse has finite length.

The simulator in `sonar_sim.py` does all seven. Below is what each one looks like in the code.

---

## What the simulator does (pipeline)

```
ray grid in (az, el)              ← (1) polar sampling
  → trace into scene → depth, normal, material per ray
  → backscatter strength = μ_d · cos²θ + μ_s · cos^n(spec angle)   ← (3)
  → bin into (azimuth_beam, range_bin)                             ← (1, 2)
       (sums many elevation rays per beam-range cell)
  → multiply by 10^(-TL/10), TL = 40 log₁₀ R + 2αR                ← (5)
  → blur in azimuth with beamwidth Gaussian                        ← (6)
  → blur in range with pulse-length Gaussian                       ← (7)
  → multiply by Gamma(L, 1/L) speckle field                        ← (4)
  → add receiver noise floor
  → apply TVG, log-compress, render fan
```

Notable references / lineage:
- **Cerqueira et al., 2017/2020** ([github.com/IvisionLab/sonar-simulation](https://github.com/IvisionLab/sonar-simulation)) — GPU rasterisation+raytracing FLS sim, source of much of the modern formulation. Worth reading for the angle-dependent reflectivity model in particular.
- **HoloOcean** (BYU/CMU) — Potokar et al. 2022 used CPU octrees; HoloOcean 2.0 (Oct 2025) moved to UE5.3 hardware ray queries (RTX). The reason it doesn't quite look like real sonar is that it's still mostly a geometric model with simple noise and only weak BRDF.
- **Wang et al., 2023 IROS** (arXiv 2304.08146) — adds ground-multipath echoes for low-altitude AUVs; not in this prototype but trivial to add (extend the trace to second-bounce).
- **Bjork & Folkesson (KTH)** — empirical evidence that Lambert² (cos²) fits real sonar better than the optical Lambert (cos).
- **Francois & Garrison 1982** — the absorption formula (valid 200 Hz – 1 MHz). Implemented verbatim in `francois_garrison_absorption()`.

Default config: 400 kHz, 120° × 20° FoV, 256 beams × 1024 range bins, 300 m max range. Range bin is 29 cm (`c·τ/2` with τ ≈ 50 µs). Two-way absorption at 300 m: 63.7 dB. Rebuilds in roughly a second on CPU per frame; this is intended for offline / scripted simulation, not realtime.

---

## Plugging it into Unreal

Two integration paths, depending on how accurate you want to be.

### Path A — rasterised depth+normal, post-processed in Python (good, fast)

This is the upgrade to what you already have. Use a `SceneCaptureComponent2D` and a custom **post-process material** to output two render targets:

1. **Scene depth** — the axial distance along sensor +X (not NDC depth). In a post-process material this is `SceneTexture:SceneDepth`. Linear meters along view direction is what you want.
2. **World-space normals** — `SceneTexture:WorldNormal`, then transform into sensor frame on the CPU side (or just keep world-space and pass the sensor rotation through).

Configuration on the SceneCapture:
- FOV: match `cfg.fov_h_deg` (default 120°).
- Aspect ratio: pick to match `fov_v_deg / fov_h_deg`. For 120°×20° that's tall-and-thin (1:6 inverted, i.e. the *vertical* field is small). UE doesn't directly take "vertical FOV", so you control vertical FOV by setting horizontal FOV plus a viewport aspect.
- **Vertical resolution**: render at 1024 vertical pixels or more. The whole point of doing this is to sample many elevations per beam — undersample and you lose elevation collapse, which means objects pop off the seafloor instead of merging with it.
- Horizontal resolution: at least 4× number of beams (1024 px ⇒ 4 sub-rays per beam in azimuth).

Then in Python:
```python
from sonar_sim import SonarConfig, simulate, trace_from_ue_buffers, render_polar_fan

cfg = SonarConfig()       # tweak if needed
cfg.fov_h_deg = 120.0; cfg.fov_v_deg = 20.0
cfg.num_beams = 256;    cfg.num_range_bins = 1024

depth   = load_ue_depth_render_target(...)    # (H, W) meters along +X
normals = load_ue_normal_render_target(...)   # (H, W, 3) world-space, unit

scene_trace = trace_from_ue_buffers(depth, normals,
                                    cfg.fov_h_deg, cfg.fov_v_deg)
out = simulate(cfg, scene_trace,
               sensor_pose={'origin': sensor_world_pos,
                            'R_world_from_sensor': sensor_rotation_matrix})
fan = render_polar_fan(out['image'], out['range_centres'],
                       out['az_centres_deg'], (900, 1200))
```

`trace_from_ue_buffers()` is in `sonar_sim.py`. It bilinearly samples the depth/normal buffer at each of the simulator's `(az, el)` rays and converts axial depth to slant range.

**Material IDs.** A real shipwreck and a real seafloor have very different backscatter. The cheap way to encode this in UE is to write a *material ID* into one channel of a render target (e.g. via `Custom Stencil` or a separate material output). Then on the Python side, replace the constant `μ_d, μ_s, n_spec` in `trace_from_ue_buffers` with a lookup table.

**Limitations.** A rasteriser only sees first-hit geometry. That means no multipath, no second-bounce, no through-water ringing. For most surveys at ~300 m this is fine.

**Range precision at long range.** UE5 (DX12) uses reverse-Z by default, so depth precision at 300 m is good — sub-cm. Don't worry about this. *Do* worry about your scene scale — if your AUV is 30 m below the surface in the editor and your floor is at 200 m absolute depth, make sure your render target's near/far planes encompass the 300 m forward cone, not the world altitude.

### Path B — hardware ray queries (more accurate, what HoloOcean 2.0 does)

If you want true ray accuracy and the option of multipath, second-bounce on hard targets, etc., do ray queries instead of rasterising:

- **Inside UE**: use the [Hardware Ray Tracing](https://docs.unrealengine.com/5.3/en-US/hardware-ray-tracing-in-unreal-engine/) path, with ray queries from a compute shader fired in your beam pattern. Output `(t, normal, material_id)` per ray to a structured buffer, read it back and feed the same simulator pipeline. RTX cards (any Ampere+ GPU) handle a 256×192-ray batch trivially.
- **Outside UE**: export your scene mesh to Python (`trimesh` or `open3d`), use [Embree](https://www.embree.org/) via [`pyembree`](https://github.com/scopatz/pyembree) or [`trimesh.ray.ray_pyembree`](https://trimesh.org/trimesh.ray.html). This is what the Cerqueira reference implementation does. It's typically 50–200× faster than naive Python ray-AABB intersection and gets you offline raytracing without any UE work.

The simulator pipeline (`simulate(cfg, scene_trace, pose)`) doesn't care which trace backend you use — `scene_trace` is just a callable returning `(t, normal, μ_d, μ_s, n_spec)` per ray.

### Path C — hybrid (best practical accuracy)

What I'd actually recommend: **rasterise for first-hit + ray-query for shadow/second-bounce only at hot pixels**. UE5.3+ supports this natively via Lumen's hybrid pipeline. For sonar this means: do Path A for the bulk image, and for any beam-range cell where the geometry suggests a strong specular target, fire a few extra rays for the second bounce. This catches the multipath ghosts (Wang 2023) without ray-tracing the whole scene.

---

## Tuning to a specific real sonar

If you're matching a particular sensor (Tritech Gemini 720i, Oculus M1200, etc.):

| sensor parameter | `SonarConfig` field |
|---|---|
| Operating frequency | `freq_kHz` (drives absorption) |
| Number of beams | `num_beams` |
| Beam width (-3 dB) | `beamwidth_h_deg` |
| Vertical fan thickness | `fov_v_deg` |
| Pulse length | `pulse_length_m` (range resolution) |
| Max display range | `range_max_m` |
| Number of range bins | `num_range_bins` |
| Receiver noise floor | `additive_noise_db` |
| Speckle smoothness (ENL) | `speckle_looks` (1 = Rayleigh, 4-8 = "smoother") |

Most spec sheets give you the first six directly. For ENL, eyeball it from a real sample: count the average bright-cell size. Default 4 is typical for multi-look display.

For a calibrated sim (matching real images numerically, not just qualitatively), you'd also need:
- the exact transmit beam pattern (probably a measured one — manufacturers don't always publish it)
- the receive array's directivity (depends on the array layout)
- the actual TVG curve the device applies (often non-trivial, time-segmented)
- environmental sound speed profile (for refraction at long range)

Refraction matters past ~150 m if you have a thermocline; at 300 m in a well-mixed water column it's a sub-degree effect and probably below your other error sources.

---

## What this isn't

- Not realtime. ~1 s/frame on CPU. With ray queries on GPU it'll be a few ms but I haven't wired that up.
- Not a synthetic aperture sonar simulator. SAS needs a different processing chain (motion compensation, autofocus, k-space).
- Not a side-scan simulator. Same physics, different geometry — the pipeline would work, but the display conventions differ.
- Doesn't model frequency-dependent target strength. A real fish flashes differently at 400 kHz vs 100 kHz; this prototype uses one frequency at a time and assumes the BRDF is freq-independent.
