"""
Demo: run the FLS simulator on a synthetic scene and produce display imagery.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sonar_sim import (SonarConfig, AnalyticalScene, simulate,
                       to_db, render_polar_fan)


# ---- Sonar colormap (similar to Tritech / Sound Metrics displays) ----
sonar_cmap = LinearSegmentedColormap.from_list(
    "sonar", [(0.00, "#000814"),
              (0.25, "#022c43"),
              (0.55, "#1e88a8"),
              (0.80, "#f9c74f"),
              (1.00, "#fff8d6")])


def main():
    cfg = SonarConfig()
    cfg.freq_kHz = 120.0
    cfg.range_max_m = 500.0
    cfg.num_range_bins = 2048
    cfg.num_beams = 256
    cfg.fov_h_deg = 120.0
    cfg.fov_v_deg = 20.0
    cfg.beamwidth_h_deg = 3.0
    cfg.pulse_length_s = 300e-6
    cfg.speckle_looks = 4
    cfg.az_samples_per_beam = 4
    cfg.el_samples = 192
    # TVG matched to TL gives uniform brightness across range (operator's view).
    # Use 40 to fully compensate spherical spreading.
    cfg.tvg_log_coeff = 40.0
    cfg.tvg_use_absorption = True
    cfg.additive_noise_db = -45.0
    cfg.output_db_min = -38.0
    cfg.output_db_max = 0.0

    scene = AnalyticalScene()

    # Sensor pose: at origin, level, looking +X
    sensor_pose = dict(origin=np.array([0.0, 0.0, 0.0]),
                       R_world_from_sensor=np.eye(3))

    print("Running simulation...")
    out = simulate(cfg, scene.trace, sensor_pose)
    img = out["image"]
    print(f"  raw image shape: {img.shape}")
    print(f"  range bins:      {len(out['range_centres'])}")
    print(f"  beams:           {len(out['az_centres_deg'])}")
    print(f"  abs:             {out['absorption_db_m']*1000:.1f} dB/km")

    # ---- Build the four display panels ----

    # (a) Raw log-compressed (azimuth, range) image - the "B-scan"
    img_db = to_db(img, cfg.output_db_min, cfg.output_db_max)

    # (b) Polar fan display - the classic FLS view
    fan = render_polar_fan(img,
                           out["range_centres"],
                           out["az_centres_deg"],
                           output_size=(900, 1200))
    fan_db = to_db(fan, cfg.output_db_min, cfg.output_db_max)

    # (c) Depth map (debug) - in sensor frame
    d = out["depth"].copy()
    d[~np.isfinite(d)] = 0
    # Average over elevation samples per beam to make a 2D image
    d_disp = d.reshape(cfg.num_beams, cfg.az_samples_per_beam,
                       cfg.el_samples).mean(axis=1)

    # (d) Cosine of incidence (debug)
    c = out["cos_theta"].reshape(cfg.num_beams, cfg.az_samples_per_beam,
                                 cfg.el_samples).mean(axis=1)

    # ---- Plot ----
    fig = plt.figure(figsize=(15, 11), facecolor='#0a0a0a')

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(img_db.T, origin="lower", cmap=sonar_cmap, aspect="auto",
               extent=[out["az_centres_deg"][0], out["az_centres_deg"][-1],
                       out["range_centres"][0], out["range_centres"][-1]])
    ax1.set_xlabel("Azimuth [deg]", color='w')
    ax1.set_ylabel("Range [m]",     color='w')
    ax1.set_title("(a) Polar B-scan  — log-compressed",
                  color='w', fontsize=11)
    ax1.tick_params(colors='w')
    ax1.set_facecolor('k')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(fan_db, origin="upper", cmap=sonar_cmap, aspect="equal")
    ax2.set_title("(b) Sector / fan display — what the operator sees",
                  color='w', fontsize=11)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_facecolor('k')

    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.imshow(d_disp.T, origin="lower", aspect="auto", cmap='magma',
                     extent=[out["az_centres_deg"][0],
                             out["az_centres_deg"][-1],
                             -cfg.fov_v_deg/2, cfg.fov_v_deg/2],
                     vmin=0, vmax=cfg.range_max_m)
    ax3.set_xlabel("Azimuth [deg]", color='w')
    ax3.set_ylabel("Elevation [deg]", color='w')
    ax3.set_title("(c) Per-ray range (debug)",
                  color='w', fontsize=11)
    ax3.tick_params(colors='w')
    ax3.set_facecolor('k')
    cb3 = plt.colorbar(im3, ax=ax3, fraction=0.04)
    cb3.ax.tick_params(colors='w'); cb3.set_label('m', color='w')

    ax4 = fig.add_subplot(2, 2, 4)
    im4 = ax4.imshow(c.T, origin="lower", aspect="auto", cmap='viridis',
                     extent=[out["az_centres_deg"][0],
                             out["az_centres_deg"][-1],
                             -cfg.fov_v_deg/2, cfg.fov_v_deg/2],
                     vmin=0, vmax=1)
    ax4.set_xlabel("Azimuth [deg]", color='w')
    ax4.set_ylabel("Elevation [deg]", color='w')
    ax4.set_title("(d) cos(incidence angle) — drives the brightness",
                  color='w', fontsize=11)
    ax4.tick_params(colors='w')
    ax4.set_facecolor('k')
    cb4 = plt.colorbar(im4, ax=ax4, fraction=0.04)
    cb4.ax.tick_params(colors='w')

    fig.suptitle(
        f"Forward-Looking Imaging Sonar simulation — "
        f"{cfg.freq_kHz:.0f} kHz, {cfg.fov_h_deg:.0f}° × {cfg.fov_v_deg:.0f}°, "
        f"{cfg.range_max_m:.0f} m, {cfg.num_beams} beams × "
        f"{cfg.num_range_bins} bins",
        color='w', fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = "fls_demo.png"
    plt.savefig(out_path, dpi=110, facecolor='#0a0a0a')
    print(f"  wrote {out_path}")

    # Also save a clean fan-only display
    fig2, ax = plt.subplots(figsize=(10, 7), facecolor='#0a0a0a')
    ax.imshow(fan_db, origin="upper", cmap=sonar_cmap, aspect="equal")
    ax.set_title("Forward-Looking Sonar — simulated output",
                 color='w', fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor('k')
    plt.tight_layout()
    fan_path = "fls_fan.png"
    plt.savefig(fan_path, dpi=120, facecolor='#0a0a0a')
    print(f"  wrote {fan_path}")


if __name__ == "__main__":
    main()
