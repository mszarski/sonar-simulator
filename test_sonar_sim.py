"""Verify trace_from_ue_buffers honours material_id + material_table,
and that the rest of the pipeline still runs end-to-end."""
import numpy as np
from sonar_sim import (SonarConfig, AnalyticalScene, simulate,
                       trace_from_ue_buffers)


def test_analytical_regression():
    cfg = SonarConfig()
    cfg.num_beams = 64
    cfg.num_range_bins = 256
    cfg.el_samples = 32
    cfg.az_samples_per_beam = 2
    out = simulate(cfg, AnalyticalScene().trace,
                   dict(origin=np.zeros(3), R_world_from_sensor=np.eye(3)))
    assert out['image'].shape == (cfg.num_beams, cfg.num_range_bins)
    assert np.isfinite(out['image']).all()
    assert out['image'].max() > 0
    print(f"  analytical pipeline OK: img max={out['image'].max():.3e}, "
          f"abs={out['absorption_db_m']*1000:.1f} dB/km")


def test_material_table_used():
    H, W = 64, 128
    fov_h, fov_v = 120.0, 20.0
    # Constant depth, normal pointing back at sensor (+Z), two material zones
    depth = np.full((H, W), 50.0)
    normal = np.zeros((H, W, 3)); normal[..., 2] = 1.0
    mat_id = np.zeros((H, W), dtype=np.int32)
    mat_id[:, W//2:] = 1     # right half is material 1

    table = {
        0: dict(mu_diff=0.10, mu_spec=0.00, n_spec=4.0),    # quiet
        1: dict(mu_diff=0.90, mu_spec=0.80, n_spec=20.0),   # loud
    }

    # ---- With material table ----
    trace_with = trace_from_ue_buffers(depth, normal, fov_h, fov_v,
                                       material_id=mat_id, material_table=table)
    # Build a tiny ray grid spanning the same FoV so we can sample both halves
    az = np.deg2rad(np.linspace(-fov_h/2, fov_h/2, 32))
    el = np.deg2rad(np.linspace(-fov_v/2, fov_v/2, 8))
    AZ, EL = np.meshgrid(az, el, indexing='ij')
    dirs = np.stack([np.cos(EL)*np.cos(AZ),
                     np.cos(EL)*np.sin(AZ),
                     np.sin(EL)], axis=-1)
    rng, n, mu_d, mu_s, n_s = trace_with(np.zeros(3), dirs)
    # Left half az<0 should map to mat 0, right half az>0 to mat 1
    left_mu_d  = mu_d[AZ < 0].mean()
    right_mu_d = mu_d[AZ > 0].mean()
    print(f"  with table:    left mu_d={left_mu_d:.2f}, right mu_d={right_mu_d:.2f}")
    assert abs(left_mu_d  - 0.10) < 1e-6, f"expected 0.10 got {left_mu_d}"
    assert abs(right_mu_d - 0.90) < 1e-6, f"expected 0.90 got {right_mu_d}"
    assert abs(mu_s[AZ > 0].mean() - 0.80) < 1e-6
    assert abs(n_s [AZ > 0].mean() - 20.0) < 1e-6

    # ---- Without table: defaults preserved ----
    trace_default = trace_from_ue_buffers(depth, normal, fov_h, fov_v)
    rng2, n2, mu_d2, mu_s2, n_s2 = trace_default(np.zeros(3), dirs)
    print(f"  default:       mu_d={mu_d2.mean():.2f}, mu_s={mu_s2.mean():.2f}, "
          f"n_s={n_s2.mean():.1f}")
    assert (mu_d2 == 0.45).all()
    assert (mu_s2 == 0.30).all()
    assert (n_s2  == 16.0).all()

    # ---- Unknown id falls through to default ----
    mat_id_unknown = np.full((H, W), 99, dtype=np.int32)
    trace_u = trace_from_ue_buffers(depth, normal, fov_h, fov_v,
                                    material_id=mat_id_unknown,
                                    material_table=table)
    _, _, mu_du, mu_su, n_su = trace_u(np.zeros(3), dirs)
    print(f"  unknown id:    mu_d={mu_du.mean():.2f} (should be 0.45)")
    assert (mu_du == 0.45).all()
    assert (mu_su == 0.30).all()
    assert (n_su  == 16.0).all()


def test_full_simulate_with_ue_trace():
    """End-to-end: simulate() driving trace_from_ue_buffers with a material map."""
    cfg = SonarConfig()
    cfg.num_beams = 32
    cfg.num_range_bins = 128
    cfg.el_samples = 16
    cfg.az_samples_per_beam = 2
    # Strip the noise/TVG layers — we only want to verify material flow-through
    cfg.speckle_enabled = False
    cfg.additive_noise_db = -200
    cfg.tvg_log_coeff = 0.0
    cfg.tvg_use_absorption = False

    H, W = 96, 192
    depth = np.full((H, W), 40.0)
    normal = np.zeros((H, W, 3)); normal[..., 0] = -1.0   # facing sensor
    mat_id = np.zeros((H, W), dtype=np.int32)
    mat_id[:, W//2:] = 1
    table = {
        0: dict(mu_diff=0.05, mu_spec=0.00, n_spec=4.0),
        1: dict(mu_diff=0.95, mu_spec=0.80, n_spec=20.0),
    }
    trace = trace_from_ue_buffers(depth, normal, cfg.fov_h_deg, cfg.fov_v_deg,
                                  material_id=mat_id, material_table=table)
    out = simulate(cfg, trace,
                   dict(origin=np.zeros(3), R_world_from_sensor=np.eye(3)))
    img = out['image']
    assert np.isfinite(img).all()
    # Constant-depth scene → signal lands in one range bin, rest is noise floor.
    # Compare the per-beam max intensity across the two halves.
    left_peak  = img[:cfg.num_beams//2].max()
    right_peak = img[cfg.num_beams//2:].max()
    print(f"  end-to-end:    left peak={left_peak:.3e}, right peak={right_peak:.3e}, "
          f"ratio={right_peak/max(left_peak,1e-12):.1f}x")
    # Backscatter ratio is (0.95+0.80)/0.05 = 35× before speckle/noise.
    assert right_peak > 10 * left_peak, "louder material should be much brighter"


if __name__ == "__main__":
    print("test_analytical_regression")
    test_analytical_regression()
    print("test_material_table_used")
    test_material_table_used()
    print("test_full_simulate_with_ue_trace")
    test_full_simulate_with_ue_trace()
    print("\nall tests passed")
