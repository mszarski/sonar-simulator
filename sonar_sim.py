"""
Forward-Looking Imaging Sonar Simulator
=======================================

Physics-based simulator for forward-looking multibeam imaging sonar (FLS).
Designed to run on depth + surface-normal maps (which can come from a synthetic
analytical scene, or from Unreal Engine SceneCapture exports).

Pipeline (this is what your current depth-image approach is missing):

  (1) For each direction (azimuth, elevation) in the sensor FoV, find the
      first surface hit: 3D point + outward surface normal.
  (2) Compute acoustic backscatter at that hit using a BRDF-like model
      driven by the incidence angle (cos² law + small specular term).
  (3) Project (range, azimuth, elevation) -> (range, azimuth). The vertical
      dimension COLLAPSES; multiple elevations summing into the same range
      bin is what gives FLS its characteristic look (and shadows).
  (4) Apply two-way transmission loss: spherical spreading + Francois-Garrison
      absorption.
  (5) Convolve in azimuth with the beam pattern (sinc²-like main lobe).
  (6) Apply Rayleigh-distributed multiplicative speckle (this is the noise
      that gives sonar its grainy texture - it's NOT additive Gaussian).
  (7) Apply TVG (time-varying gain), additive noise floor, and quantize for
      display.
  (8) Render as a polar/fan image.

References:
  Cerqueira et al. 2017/2020 (rasterized GPU pipeline, Computers & Graphics)
  Potokar et al. 2022 (HoloOcean sonar octree method)
  Wang et al. 2023 (FLS with ground echo / multipath, IROS)
  Francois & Garrison 1982 (seawater absorption)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable

# =============================================================================
# 1. SENSOR CONFIGURATION
# =============================================================================

@dataclass
class SonarConfig:
    """Configuration for a forward-looking imaging sonar.

    Defaults are sized for a long-range FLS at 120 kHz targeting ~500 m,
    in the spirit of long-range imaging / obstacle-avoidance sonars
    (e.g. Kongsberg Mesotech 120 kHz family, Tritech SeaKing long-range).
    Higher-frequency systems trade range for resolution: Tritech Gemini
    720i (720 kHz, ~120 m), Sound Metrics ARIS (~1.8 MHz, ~30 m).
    """
    # ---- Geometry ----
    range_min_m:     float = 1.0
    range_max_m:     float = 500.0
    num_range_bins:  int   = 2048     # → ~24 cm bin spacing, matches pulse cell
    fov_h_deg:       float = 120.0    # horizontal sector width (azimuth)
    fov_v_deg:       float = 20.0     # vertical aperture (elevation)
    num_beams:       int   = 256      # azimuth bins displayed

    # ---- Acoustic / signal ----
    freq_kHz:        float = 120.0    # 120 kHz: ~44 dB/km absorption, good to ~500 m
    pulse_length_s:  float = 300e-6   # → 22.5 cm range cell at c=1500
    sound_speed:     float = 1500.0

    # Per-beam directivity (3dB widths). Real arrays at this frequency band:
    #  Kongsberg Mesotech 1071 (~120 kHz): ~3° H × 20° V
    #  Tritech SeaKing 120 kHz long-range: ~3° H
    # For comparison higher-freq imaging arrays are much narrower:
    #  Tritech Gemini 720i (720 kHz):  0.5° H × 20° V
    #  ARIS 1200          (1.2 MHz):  ~0.3° H × 14° V
    beamwidth_h_deg: float = 3.0
    beamwidth_v_deg: float = 20.0

    # ---- Sampling resolution for ray grid ----
    # Each "beam" is sampled with multiple sub-rays in azimuth so we can
    # do proper beam-pattern integration rather than picking 1 ray/beam.
    az_samples_per_beam: int = 4
    el_samples:          int = 128    # vertical ray count

    # ---- Environment (for absorption) ----
    temperature_C:   float = 15.0
    salinity_psu:    float = 35.0
    depth_m:         float = 50.0
    pH:              float = 8.0

    # ---- Backscatter model ----
    # Diffuse coefficient (Lambert²): fraction of incident energy returned
    # in diffuse lobe. Typical seafloor: 0.1-0.5 depending on substrate.
    # Hard targets: higher.
    bs_diffuse_default:  float = 0.4
    # Specular peak (returns when surface roughly perpendicular to beam)
    bs_specular_default: float = 0.3
    bs_specular_power:   float = 16.0  # narrowness of specular lobe

    # ---- Noise ----
    speckle_enabled:        bool  = True
    speckle_looks:          int   = 4    # equivalent number of looks (ENL)
    additive_noise_db:      float = -55  # dB, ref full scale
    noise_seed:             int   = 42

    # ---- TVG (display gain, applied at end) ----
    # Real systems use either log-TVG (a*log10(r) + 2*alpha*r) or power TVG.
    tvg_log_coeff:          float = 30.0   # 30*log10(r) classic spreading TVG
    tvg_use_absorption:     bool  = True

    # ---- Display ----
    output_db_min:          float = -45.0
    output_db_max:          float = 0.0

    # ---- Derived ----
    @property
    def num_az_samples(self) -> int:
        return self.num_beams * self.az_samples_per_beam

    @property
    def range_resolution_m(self) -> float:
        return (self.range_max_m - self.range_min_m) / self.num_range_bins

    @property
    def pulse_length_m(self) -> float:
        return 0.5 * self.sound_speed * self.pulse_length_s


# =============================================================================
# 2. ACOUSTIC PHYSICS
# =============================================================================

def francois_garrison_absorption(freq_kHz: float, T_C: float, S_psu: float,
                                 depth_m: float, pH: float) -> float:
    """Francois-Garrison (1982) seawater absorption coefficient in dB/m.

    Valid 200 Hz – 1 MHz. Three contributions: boric acid, MgSO4, pure water.
    For typical FLS frequencies (200 kHz - 1 MHz) the MgSO4 term dominates.
    """
    f, T, S, D = freq_kHz, T_C, S_psu, depth_m
    c = 1412 + 3.21*T + 1.19*S + 0.0167*D     # sound speed approx

    # Boric acid (negligible above ~10 kHz)
    A1 = (8.86 / c) * 10**(0.78*pH - 5)
    P1 = 1.0
    f1 = 2.8 * np.sqrt(S / 35.0) * 10**(4 - 1245/(T + 273))

    # Magnesium sulfate (dominant 10 kHz - 1 MHz)
    A2 = 21.44 * (S / c) * (1 + 0.025*T)
    P2 = 1 - 1.37e-4*D + 6.2e-9*D**2
    f2 = (8.17 * 10**(8 - 1990/(T + 273))) / (1 + 0.0018*(S - 35))

    # Pure water (dominant > 1 MHz)
    if T <= 20:
        A3 = 4.937e-4 - 2.59e-5*T + 9.11e-7*T**2 - 1.50e-8*T**3
    else:
        A3 = 3.964e-4 - 1.146e-5*T + 1.45e-7*T**2 - 6.5e-10*T**3
    P3 = 1 - 3.83e-5*D + 4.9e-10*D**2

    alpha_db_per_km = (A1*P1*f1*f**2 / (f1**2 + f**2)
                       + A2*P2*f2*f**2 / (f2**2 + f**2)
                       + A3*P3*f**2)
    return alpha_db_per_km / 1000.0   # dB/m


def two_way_transmission_loss_db(r_m: np.ndarray, alpha_db_m: float) -> np.ndarray:
    """Two-way (round trip) TL. Spherical spreading 40 log r + 2 alpha r."""
    r = np.maximum(r_m, 1.0)
    return 40.0 * np.log10(r) + 2.0 * alpha_db_m * r


def backscatter_strength(cos_theta: np.ndarray,
                         mu_diff: float, mu_spec: float, n_spec: float) -> np.ndarray:
    """
    Per-element backscatter coefficient as a function of incidence angle.

    cos_theta: cosine between -ray_dir and outward surface normal at hit.
               1.0 = normal incidence (perpendicular to surface),
               0.0 = grazing incidence.

    Two-component model:
      diffuse  = mu_diff * cos²(θ)     # Lambert² law (popular for seafloor)
      specular = mu_spec * cos^n(θ)    # peak at normal, narrow for hard targets

    Lambert² (cos²) is a much better fit for sonar than the optical Lambert
    (cos), as shown by Bjork & Folkesson (KTH) - the "cotangent law" paper.
    """
    c = np.clip(cos_theta, 0.0, 1.0)
    return mu_diff * c**2 + mu_spec * c**n_spec


# =============================================================================
# 3. SCENE SAMPLING
# =============================================================================
#
# This is the part that's intentionally pluggable. The simulator only
# needs a function that, for an array of ray directions in sensor frame,
# returns:
#   range:  (N,) float - first-hit distance, np.inf for misses
#   normal: (N, 3) float - outward surface normal at hit (sensor frame)
#   material_id: (N,) int - look-up index into material table (optional)
#
# Two implementations below: (a) AnalyticalScene for the demo, (b) a
# stub showing how to wire UE-exported buffers.

def make_ray_grid(cfg: SonarConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a (azimuth × elevation) grid of unit ray directions in sensor frame.

    Sensor frame convention used here:
      +X forward (along boresight)
      +Y starboard / right
      +Z up
    Azimuth = angle from +X about Z, positive to starboard (right).
    Elevation = angle from horizontal plane, positive up.

    Returns:
      az_grid: (n_az,) azimuth bin centres in radians
      el_grid: (n_el,) elevation sample angles in radians
      dirs:    (n_az, n_el, 3) unit direction vectors
    """
    n_az = cfg.num_az_samples
    n_el = cfg.el_samples
    az = np.deg2rad(np.linspace(-cfg.fov_h_deg/2, cfg.fov_h_deg/2, n_az))
    el = np.deg2rad(np.linspace(-cfg.fov_v_deg/2, cfg.fov_v_deg/2, n_el))
    AZ, EL = np.meshgrid(az, el, indexing='ij')          # (n_az, n_el)
    dirs = np.stack([np.cos(EL) * np.cos(AZ),
                     np.cos(EL) * np.sin(AZ),
                     np.sin(EL)], axis=-1)               # (n_az, n_el, 3)
    return az, el, dirs


# ---- Analytical synthetic scene -------------------------------------------

@dataclass
class SceneObject:
    kind: str                           # 'sphere' | 'box'
    params: dict
    material: dict                      # {'mu_diff': .., 'mu_spec': .., 'n_spec': ..}


def _intersect_sphere(orig, dirs, centre, radius):
    """Vectorised ray-sphere. dirs: (...,3). Returns (t, normal_world)."""
    oc = orig - centre                                                # (3,)
    b = np.einsum('...i,i->...', dirs, oc)
    c = np.dot(oc, oc) - radius * radius
    disc = b*b - c
    hit = disc > 0
    sq = np.where(hit, np.sqrt(np.maximum(disc, 0.0)), 0.0)
    t = -b - sq
    t = np.where((t > 0) & hit, t, np.inf)
    p = orig + dirs * t[..., None]
    n = (p - centre) / radius
    return t, n


def _intersect_box(orig, dirs, centre, half):
    """Axis-aligned box (slab method). half = (3,) half-extents."""
    safe = np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
    inv = 1.0 / safe
    t1 = (centre - half - orig) * inv
    t2 = (centre + half - orig) * inv
    tmin = np.minimum(t1, t2)
    tmax = np.maximum(t1, t2)
    t_enter = np.max(tmin, axis=-1)
    t_exit  = np.min(tmax, axis=-1)
    hit = (t_enter < t_exit) & (t_exit > 0)
    t = np.where(hit, np.where(t_enter > 0, t_enter, np.inf), np.inf)
    # Compute normal: pick axis where tmin is largest at entry
    p = orig + dirs * t[..., None]
    local = p - centre
    a = np.abs(local) - half
    axis = np.argmax(a, axis=-1)                          # face axis
    nrm = np.zeros_like(local)
    idx = np.indices(axis.shape)
    nrm[(*idx, axis)] = np.sign(local[(*idx, axis)])
    return t, nrm


def _intersect_plane(orig, dirs, point, normal):
    """Single plane. Returns (t, normal_broadcast)."""
    denom = np.einsum('...i,i->...', dirs, normal)
    t = np.einsum('i,i->', point - orig, normal) / np.where(np.abs(denom) < 1e-9,
                                                             1e-9, denom)
    t = np.where((np.abs(denom) > 1e-9) & (t > 0), t, np.inf)
    n_full = np.broadcast_to(normal, dirs.shape).copy()
    return t, n_full


class AnalyticalScene:
    """A synthetic underwater scene. Sensor sits at origin looking down +X.

    Used to demonstrate the simulator without needing a 3D engine. In your
    pipeline you will replace this with the depth + normal buffer that
    SceneCapture2D writes in Unreal.
    """

    def __init__(self):
        # Sonar mounted ~12 m above seafloor (typical AUV survey altitude).
        # Slight slope so the return is not perfectly symmetric.
        self.seafloor_depth   = 12.0
        self.seafloor_slope_x = 0.004    # +x means floor goes down with range
        self.seafloor_slope_y = 0.001
        # Soft sediment seafloor: weak diffuse return, almost no specular.
        self.seafloor_material = dict(mu_diff=0.18, mu_spec=0.02, n_spec=4.0)

        self.objects = [
            # Small debris near field
            SceneObject('sphere',
                        dict(centre=np.array([55.0,   6.0, -10.5]), radius=1.0),
                        dict(mu_diff=0.85, mu_spec=0.7, n_spec=12)),
            SceneObject('sphere',
                        dict(centre=np.array([70.0,  -5.0, -10.8]), radius=0.8),
                        dict(mu_diff=0.85, mu_spec=0.7, n_spec=12)),

            # Wreck-like rectangular structure at ~125 m, sitting on seafloor,
            # raised off floor. Strong return.
            SceneObject('box',
                        dict(centre=np.array([125.0, -3.0, -8.5]),
                             half=np.array([5.0, 9.0, 3.5])),
                        dict(mu_diff=0.95, mu_spec=0.9, n_spec=20)),

            # Pipeline crossing the scene at ~200 m, 80 m long across azimuth
            SceneObject('box',
                        dict(centre=np.array([200.0, 0.0, -11.4]),
                             half=np.array([0.8, 40.0, 0.8])),
                        dict(mu_diff=1.0, mu_spec=1.0, n_spec=8)),

            # Boulder field at ~270 m
            SceneObject('sphere',
                        dict(centre=np.array([255.0, -38.0, -10.5]), radius=3.0),
                        dict(mu_diff=0.85, mu_spec=0.6, n_spec=10)),
            SceneObject('sphere',
                        dict(centre=np.array([270.0,  42.0, -10.0]), radius=3.5),
                        dict(mu_diff=0.85, mu_spec=0.6, n_spec=10)),
            SceneObject('sphere',
                        dict(centre=np.array([280.0,  12.0, -10.6]), radius=2.0),
                        dict(mu_diff=0.85, mu_spec=0.6, n_spec=10)),

            # Container-like long target at ~330 m, broadside-on
            SceneObject('box',
                        dict(centre=np.array([330.0,  18.0, -9.5]),
                             half=np.array([6.0, 1.2, 2.5])),
                        dict(mu_diff=0.9, mu_spec=0.8, n_spec=18)),

            # Distant cluster at ~430 m
            SceneObject('sphere',
                        dict(centre=np.array([425.0, -55.0, -9.0]), radius=5.0),
                        dict(mu_diff=0.85, mu_spec=0.6, n_spec=8)),
            SceneObject('box',
                        dict(centre=np.array([440.0,  65.0, -9.5]),
                             half=np.array([4.0, 7.0, 3.0])),
                        dict(mu_diff=0.9, mu_spec=0.8, n_spec=14)),

            # Far edge of operating range (~480 m)
            SceneObject('sphere',
                        dict(centre=np.array([480.0,   0.0, -9.0]), radius=4.0),
                        dict(mu_diff=0.85, mu_spec=0.6, n_spec=8)),
        ]

    def trace(self, origin: np.ndarray, dirs: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Trace rays and return first-hit info per ray.

        dirs: (n_az, n_el, 3). origin: (3,)
        Returns t (n_az,n_el), normal (n_az,n_el,3), and the three
        backscatter parameters per hit (mu_diff, mu_spec, n_spec).
        """
        shape = dirs.shape[:-1]
        t_best = np.full(shape, np.inf)
        n_best = np.zeros(dirs.shape)
        n_best[..., 2] = 1.0
        mu_d = np.full(shape, self.seafloor_material['mu_diff'])
        mu_s = np.full(shape, self.seafloor_material['mu_spec'])
        n_s  = np.full(shape, self.seafloor_material['n_spec'])

        # Seafloor plane: passes through (0,0,-h), normal ~ (-slope_x, -slope_y, 1)
        sf_n = np.array([-self.seafloor_slope_x, -self.seafloor_slope_y, 1.0])
        sf_n /= np.linalg.norm(sf_n)
        sf_p = np.array([0.0, 0.0, -self.seafloor_depth])
        t_p, n_p = _intersect_plane(origin, dirs, sf_p, sf_n)
        better = t_p < t_best
        t_best = np.where(better, t_p, t_best)
        n_best = np.where(better[..., None], n_p, n_best)
        # Seafloor material is already the default

        # Each object
        for obj in self.objects:
            if obj.kind == 'sphere':
                t_o, n_o = _intersect_sphere(origin, dirs,
                                             obj.params['centre'],
                                             obj.params['radius'])
            elif obj.kind == 'box':
                t_o, n_o = _intersect_box(origin, dirs,
                                          obj.params['centre'],
                                          obj.params['half'])
            else:
                continue
            better = t_o < t_best
            t_best = np.where(better, t_o, t_best)
            n_best = np.where(better[..., None], n_o, n_best)
            mu_d = np.where(better, obj.material['mu_diff'], mu_d)
            mu_s = np.where(better, obj.material['mu_spec'], mu_s)
            n_s  = np.where(better, obj.material['n_spec'],  n_s)

        return t_best, n_best, mu_d, mu_s, n_s


# =============================================================================
# 4. SONAR PIPELINE
# =============================================================================

def simulate(cfg: SonarConfig, scene_trace: Callable, sensor_pose: dict
             ) -> dict:
    """Run the full pipeline.

    sensor_pose: {'origin': (3,), 'R_world_from_sensor': (3,3)}
        Identity rotation puts the sensor frame == world frame.
    scene_trace: callable(origin_world, dirs_world) -> (t, normal_world,
                                                        mu_d, mu_s, n_s)

    Returns dict with intermediate arrays (handy for debugging / display).
    """
    rng = np.random.default_rng(cfg.noise_seed)

    # ---- Build ray grid in sensor frame, transform to world ----
    az_centres, el_grid, dirs_sensor = make_ray_grid(cfg)        # (n_az, n_el, 3)
    R = sensor_pose.get('R_world_from_sensor', np.eye(3))
    origin = sensor_pose['origin']
    dirs_world = dirs_sensor @ R.T

    # ---- Trace the scene ----
    t, normal_w, mu_d, mu_s, n_s = scene_trace(origin, dirs_world)
    # Bring normal back to sensor frame for incidence angle calc
    normal_s = normal_w @ R                                # (n_az, n_el, 3)

    # Cosine of incidence angle = max(-dir . normal, 0)
    cos_theta = -np.einsum('...i,...i->...', dirs_sensor, normal_s)
    cos_theta = np.maximum(cos_theta, 0.0)

    # ---- Per-ray backscatter strength ----
    sigma = backscatter_strength(cos_theta, mu_d, mu_s, n_s)
    # Out-of-range rays contribute nothing
    valid = (t >= cfg.range_min_m) & (t <= cfg.range_max_m) & np.isfinite(t)
    sigma = np.where(valid, sigma, 0.0)

    # ---- Footprint area: each ray represents a chunk of solid angle.
    # The contribution of a hit to the (range, az) cell scales with the
    # area subtended by the ray on the surface, ~ (r² dΩ)/cos(θ_grazing).
    # In our discretisation we just need each ray to carry an equal-Ω weight;
    # the (1/r²) part is in transmission loss further down.
    n_az_s, n_el = sigma.shape
    daz = np.deg2rad(cfg.fov_h_deg) / n_az_s
    dele = np.deg2rad(cfg.fov_v_deg) / n_el
    # Weight by elevation Jacobian (cos(el))
    el_weight = np.cos(el_grid)[None, :]                          # (1, n_el)
    omega_weight = daz * dele * el_weight                         # solid angle
    sigma_w = sigma * omega_weight

    # ---- Bin into (range, azimuth) image ----
    # For each ray, find its range-bin index and azimuth-beam index, accumulate.
    range_bin_w = cfg.range_resolution_m
    # Compute r_idx safely (avoid casting inf -> int undefined behaviour)
    r_idx = np.zeros(t.shape, dtype=np.int64)
    finite = np.isfinite(t)
    r_idx[finite] = np.clip(
        np.floor((t[finite] - cfg.range_min_m) / range_bin_w).astype(np.int64),
        0, cfg.num_range_bins - 1)
    # Azimuth bin: each beam covers az_samples_per_beam sub-rays
    az_idx_full = np.arange(n_az_s)
    az_idx = (az_idx_full // cfg.az_samples_per_beam).astype(np.int64)
    az_idx_2d = np.broadcast_to(az_idx[:, None], sigma.shape)

    img = np.zeros((cfg.num_beams, cfg.num_range_bins), dtype=np.float64)
    flat_idx = az_idx_2d * cfg.num_range_bins + r_idx
    np.add.at(img.ravel(), flat_idx[valid], sigma_w[valid])

    # ---- Apply two-way transmission loss in dB ----
    alpha = francois_garrison_absorption(cfg.freq_kHz, cfg.temperature_C,
                                         cfg.salinity_psu, cfg.depth_m,
                                         cfg.pH)
    r_centres = cfg.range_min_m + (np.arange(cfg.num_range_bins) + 0.5) * range_bin_w
    tl_db = two_way_transmission_loss_db(r_centres, alpha)
    tl_lin = 10**(-tl_db / 10.0)
    img *= tl_lin[None, :]

    # ---- Beam-pattern smoothing in azimuth ----
    # Approximate the in-beam directivity by Gaussian with sigma matched to
    # the -3dB beamwidth: sigma = bw/(2*sqrt(2 ln 2)) ≈ bw/2.355
    bw_az_rad = np.deg2rad(cfg.beamwidth_h_deg)
    az_bin_rad = np.deg2rad(cfg.fov_h_deg / cfg.num_beams)
    sigma_bins = (bw_az_rad / 2.355) / az_bin_rad
    if sigma_bins > 0.3:                       # only blur if it's worth it
        img = _gaussian_blur_1d_axis(img, sigma_bins, axis=0)

    # ---- Speckle (multiplicative, gamma-distributed) ----
    if cfg.speckle_enabled:
        # ENL=L looks → gamma(shape=L, scale=1/L). Mean=1, var=1/L.
        L = max(1, cfg.speckle_looks)
        speckle = rng.gamma(shape=L, scale=1.0/L, size=img.shape)
        img = img * speckle

    # ---- Additive noise floor ----
    img_max = img.max() if img.max() > 0 else 1.0
    noise_floor = img_max * 10**(cfg.additive_noise_db / 10.0)
    img = img + rng.exponential(scale=noise_floor, size=img.shape)

    # ---- TVG: amplify range to compensate for spreading + absorption ----
    if cfg.tvg_log_coeff > 0 or cfg.tvg_use_absorption:
        tvg_db = cfg.tvg_log_coeff * np.log10(np.maximum(r_centres, 1.0))
        if cfg.tvg_use_absorption:
            tvg_db = tvg_db + 2.0 * alpha * r_centres
        tvg_lin = 10**(tvg_db / 20.0)             # amplitude gain
        img = img * tvg_lin[None, :]

    return dict(image=img,
                range_centres=r_centres,
                az_centres_deg=np.rad2deg(az_centres[::cfg.az_samples_per_beam]),
                cos_theta=cos_theta,
                depth=t,
                normal=normal_s,
                absorption_db_m=alpha)


def _gaussian_blur_1d_axis(arr: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    """Plain 1-D Gaussian blur along one axis (no scipy dependency)."""
    if sigma <= 0:
        return arr
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-x*x / (2 * sigma * sigma))
    k /= k.sum()
    arr_t = np.moveaxis(arr, axis, -1)
    out = np.zeros_like(arr_t)
    pad = np.pad(arr_t, [(0,0)]*(arr_t.ndim-1) + [(radius, radius)], mode='edge')
    for i in range(2*radius + 1):
        out += k[i] * pad[..., i:i + arr_t.shape[-1]]
    return np.moveaxis(out, -1, axis)


# =============================================================================
# 5. DISPLAY
# =============================================================================

def to_db(image: np.ndarray, db_min: float, db_max: float) -> np.ndarray:
    """Log-compress the intensity image to a 0..1 dB-mapped value."""
    img = np.maximum(image, 1e-12)
    img_db = 10.0 * np.log10(img / img.max())
    return np.clip((img_db - db_min) / (db_max - db_min), 0.0, 1.0)


def render_polar_fan(image_polar: np.ndarray,
                     range_centres: np.ndarray,
                     az_centres_deg: np.ndarray,
                     output_size: Tuple[int, int] = (1024, 768)) -> np.ndarray:
    """Resample (azimuth, range) → cartesian (x_forward, y_starboard) fan image.

    This is the classic FLS sector display.
    """
    H, W = output_size
    r_min = range_centres.min()
    r_max = range_centres.max()
    az_min = np.deg2rad(az_centres_deg.min())
    az_max = np.deg2rad(az_centres_deg.max())

    # Cartesian extents to cover the fan (sensor at top, fan opening downward)
    half_width = r_max * np.sin(max(abs(az_min), abs(az_max)))
    # x = forward, y = lateral; we'll display with x going DOWN the screen.
    xs = np.linspace(0.0, r_max, H)              # forward distance
    ys = np.linspace(-half_width, half_width, W) # lateral
    Y, X = np.meshgrid(ys, xs)                   # X forward, Y lateral

    R = np.sqrt(X*X + Y*Y)
    A = np.arctan2(Y, X)                         # 0 forward, +ve right

    # Map (R, A) → indices into (az_centres, range_centres)
    a_idx = (A - np.deg2rad(az_centres_deg[0])) / np.deg2rad(
        az_centres_deg[1] - az_centres_deg[0])
    r_idx = (R - r_min) / (range_centres[1] - range_centres[0])

    a0 = np.floor(a_idx).astype(int)
    r0 = np.floor(r_idx).astype(int)
    af = a_idx - a0
    rf = r_idx - r0

    n_az = len(az_centres_deg)
    n_r  = len(range_centres)
    valid = (a0 >= 0) & (a0 < n_az - 1) & (r0 >= 0) & (r0 < n_r - 1)

    out = np.zeros_like(X, dtype=image_polar.dtype)
    a0c = np.clip(a0, 0, n_az - 2)
    r0c = np.clip(r0, 0, n_r - 2)
    v00 = image_polar[a0c,     r0c]
    v01 = image_polar[a0c,     r0c + 1]
    v10 = image_polar[a0c + 1, r0c]
    v11 = image_polar[a0c + 1, r0c + 1]
    interp = ((1-af)*(1-rf)*v00 + (1-af)*rf*v01
              + af*(1-rf)*v10 + af*rf*v11)
    out = np.where(valid, interp, 0.0)
    return out


# =============================================================================
# 6. UE INTEGRATION STUB
# =============================================================================

def trace_from_ue_buffers(depth_image: np.ndarray,
                          normal_image: np.ndarray,
                          fov_h_deg: float,
                          fov_v_deg: float,
                          material_id: Optional[np.ndarray] = None,
                          material_table: Optional[dict] = None) -> Callable:
    """Return a `scene_trace` callable backed by depth + normal images that
    you exported from a Unreal Engine SceneCapture2D.

    depth_image: (H, W) Z distance along sensor +X (NOT NDC depth)
    normal_image: (H, W, 3) world-space normals, sensor-frame normalised
    The image must be rendered with its principal axis along +X (boresight),
    spanning fov_h_deg horizontally and fov_v_deg vertically.

    Inside Unreal you'll typically:
      * Use a custom post-process material that writes scene depth and
        world-space normal into a render target (or two).
      * Make sure the SceneCaptureComponent2D's FOV matches (fov_h_deg).
      * Render at HIGH vertical resolution (1024+) so the elevation-axis
        integration is well-sampled.
    """
    H, W = depth_image.shape

    def trace(origin, dirs):
        # The simulator built a (n_az, n_el) grid covering the same FoV; we
        # bilinearly sample the UE buffers at those angles.
        # First recover (az, el) of each direction.
        az = np.arctan2(dirs[..., 1], dirs[..., 0])
        el = np.arcsin(np.clip(dirs[..., 2], -1, 1))

        u = (az + np.deg2rad(fov_h_deg)/2) / np.deg2rad(fov_h_deg) * (W - 1)
        v = (np.deg2rad(fov_v_deg)/2 - el) / np.deg2rad(fov_v_deg) * (H - 1)
        u = np.clip(u, 0, W - 1)
        v = np.clip(v, 0, H - 1)
        u0 = np.floor(u).astype(int); u1 = np.minimum(u0 + 1, W - 1)
        v0 = np.floor(v).astype(int); v1 = np.minimum(v0 + 1, H - 1)
        fu = u - u0; fv = v - v0

        def bilinear(img):
            return ((1-fu)*(1-fv)*img[v0,u0] + fu*(1-fv)*img[v0,u1]
                    + (1-fu)*fv*img[v1,u0] + fu*fv*img[v1,u1])

        z_along_x = bilinear(depth_image)               # axial depth
        # Convert to range (Euclidean distance along the ray)
        cos_off_axis = np.cos(az) * np.cos(el)
        cos_off_axis = np.where(np.abs(cos_off_axis) < 1e-6, 1e-6, cos_off_axis)
        rng = z_along_x / cos_off_axis

        if normal_image.ndim == 3 and normal_image.shape[-1] == 3:
            nx = bilinear(normal_image[..., 0])
            ny = bilinear(normal_image[..., 1])
            nz = bilinear(normal_image[..., 2])
            n  = np.stack([nx, ny, nz], axis=-1)
            n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
        else:
            n = np.zeros(dirs.shape); n[..., 2] = 1.0

        if material_id is not None and material_table is not None:
            # Nearest-neighbour sample of the material ID image
            u_nn = np.clip(np.round(u).astype(int), 0, W - 1)
            v_nn = np.clip(np.round(v).astype(int), 0, H - 1)
            ids = material_id[v_nn, u_nn]

            mu_d = np.full(ids.shape, 0.45)
            mu_s = np.full(ids.shape, 0.30)
            n_s  = np.full(ids.shape, 16.0)
            for mat_id, props in material_table.items():
                mask = ids == mat_id
                mu_d = np.where(mask, props['mu_diff'], mu_d)
                mu_s = np.where(mask, props['mu_spec'], mu_s)
                n_s  = np.where(mask, props['n_spec'],  n_s)
        else:
            mu_d = np.full(rng.shape, 0.45)
            mu_s = np.full(rng.shape, 0.30)
            n_s  = np.full(rng.shape, 16.0)
        return rng, n, mu_d, mu_s, n_s

    return trace


if __name__ == "__main__":
    # Quick self-check
    cfg = SonarConfig()
    print(f"FLS @ {cfg.freq_kHz} kHz")
    print(f"  range bin: {cfg.range_resolution_m*100:.1f} cm")
    print(f"  pulse cell: {cfg.pulse_length_m*100:.1f} cm")
    a = francois_garrison_absorption(cfg.freq_kHz, cfg.temperature_C,
                                     cfg.salinity_psu, cfg.depth_m, cfg.pH)
    print(f"  absorption: {a*1000:.2f} dB/km   "
          f"→ 2-way at 300 m: {2*a*300:.1f} dB")
