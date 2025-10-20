import os
import math
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from src.constants import PC

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _band_center_and_span(band_limits_cm: Tuple[float, float]) -> Tuple[float, float]:
    lam_min, lam_max = band_limits_cm
    lam_c = math.sqrt(lam_min * lam_max)  
    span_ln = math.log(lam_max/lam_min) 
    return lam_c, span_ln

def plot_opacity_with_band_means(draine_data, band_opacities, BANDS, save = True, 
                                 outdir = "outputs/figures", run_dir: str | None = None):
    """
    Plot Draine with shaded B/V/K passbands and markers at band-averaged opacities.
    """
    lam_cm = np.asarray(draine_data['wavelength'])
    kappa = np.asarray(draine_data['kappa'])
    lam_um = lam_cm * 1e4

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(lam_um, kappa, lw=1.8, label='Draine opacity (RV=5.5)')

    # Shade bands + markers
    band_colors = {'B': (0.65,0.80,1.0,0.2), 'V': (0.80,1.0,0.65,0.2), 'K': (1.0,0.70,0.70,0.2)}
    center_defaults_um = {'B': 0.44, 'V': 0.55, 'K': 2.2}

    for b, (lam_min, lam_max) in BANDS.items():
        # shade
        ax.axvspan(lam_min*1e4, lam_max*1e4, color=band_colors.get(b, (0.8,0.8,0.8,0.2)), lw=0)
        # marker at center with provided kappa_bar
        if b in band_opacities:
            x = center_defaults_um.get(b, np.sqrt(lam_min*lam_max)*1e4)
            y = band_opacities[b]
            ax.scatter([x], [y], s=55, zorder=5)
            ax.text(x, y, f"  {b}", va='center')

    ax.set_xlabel('Wavelength (\u00b5m)')
    ax.set_ylabel(r'$\kappa\,(\mathrm{cm}^2\,\mathrm{g}^{-1})$')
    ax.set_title('Dust opacity with B/V/K band means')
    ax.grid(True, which='both', ls=':', alpha=0.35)
    ax.legend()

    if save:
        folder = _ensure_dir(os.path.join(outdir, run_dir) if run_dir else outdir)
        fig.savefig(os.path.join(folder, "opacity_validation.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)

    return fig, ax

def plot_band_sed_lambdaL(results, BANDS, bands = ('B','V','K'), save = True,
                          outdir = "outputs/figures", run_dir: str | None = None):
    """
    Plot band SED: intrinsic vs escaped, normalized
    """
    lam_um, L_in_proxy, L_esc_proxy = [], [], []

    for b in bands:
        if b not in BANDS or b not in results:
            continue
        lam_c, span_ln = _band_center_and_span(BANDS[b]) 
        lam_um.append(lam_c * 1e4)

        L_in = results[b].L_input  
        L_esc = results[b].L_escaped
        
        Li = L_in / max(span_ln, 1e-12)
        Le = L_esc / max(span_ln, 1e-12)
        L_in_proxy.append(Li)
        L_esc_proxy.append(Le)

    lam_um = np.array(lam_um)
    L_in_proxy = np.array(L_in_proxy)
    L_esc_proxy = np.array(L_esc_proxy)

    Lmax = max(float(np.max(L_in_proxy)), float(np.max(L_esc_proxy)), 1.0)
    L_in_n = L_in_proxy / Lmax
    L_esc_n = L_esc_proxy / Lmax

    fig, ax = plt.subplots(figsize=(7,5))
    ax.loglog(lam_um, L_in_n, marker='o', lw=1.8, label='Intrinsic (normalized)')
    ax.loglog(lam_um, L_esc_n, marker='s', lw=1.8, label='Escaped (normalized)')

    for x, b in zip(lam_um, bands):
        ax.text(x, np.interp(x, lam_um, L_esc_n), f" {b}", va='bottom')

    ax.set_xlabel('Wavelength (\u00b5m)')
    ax.set_ylabel(r'Normalized $\lambda L_\lambda$')
    ax.set_title('Band SED: intrinsic vs escaped (normalized)')
    ax.grid(True, which='both', ls=':', alpha=0.35)
    ax.legend()

    if save:
        folder = _ensure_dir(os.path.join(outdir, run_dir) if run_dir else outdir)
        fig.savefig(os.path.join(folder, "sed_comparison.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)

    return fig, ax

def plot_absorption_map_single(grid, band, save=False, outdir="outputs/figures", run_dir=None):
    """
    Single-band absorption map with white-on-black style.
    Orientation fixed (no transpose). Returns (fig, ax).
    """
    proj = np.sum(grid.L_absorbed, axis=2)
    extent = [grid.lower_bounds[0]/PC, grid.upper_bounds[0]/PC,
            grid.lower_bounds[1]/PC, grid.upper_bounds[1]/PC]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_facecolor('black')

    vmax = float(np.max(proj))
    if vmax <= 0:
        ax.imshow(np.zeros((10, 10)), origin='lower', extent=extent,
                  cmap='gray', vmin=0, vmax=1, aspect='equal')
        ax.text(0.5, 0.5, 'no absorption', color='w',
                ha='center', va='center', transform=ax.transAxes)
    else:
        proj_mask = np.ma.masked_less_equal(proj, 0)
        vmin = float(proj_mask.min())
        im = ax.imshow(proj_mask, origin='lower', extent=extent,
                       cmap='gray', norm=LogNorm(vmin=vmin, vmax=vmax),
                       aspect='equal')
        fig.colorbar(im, ax=ax, label='Absorbed Luminosity (log scale)')

    ax.set_xlabel('x (pc)')
    ax.set_ylabel('y (pc)')
    ax.set_title(f'{band} Band Absorption Map')

    if save:
        folder = _ensure_dir(os.path.join(outdir, run_dir) if run_dir else outdir)
        fig.savefig(os.path.join(folder, f"absorption_{band}.png"),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)

    return fig, ax

def create_rgb_composite(B_map, V_map, K_map, extent_cm=None,
                         save=False, outdir="outputs/figures", run_dir=None):
    """
    Combine B, V, K 2D arrays into an RGB image (R=K, G=V, B=B).
    Per-channel 99th-percentile normalization + asinh stretch.
    Adds axis labels so it matches the absorption maps. Returns (fig, ax).
    """
    def _prep_channel(img):
        img = np.clip(np.asarray(img, float), 0, None)
        if not np.any(img > 0):
            return np.zeros_like(img)
        p99 = np.percentile(img[img > 0], 99)
        x = img / (p99 + 1e-30)
        return np.arcsinh(2 * x) / np.arcsinh(2)

    R = _prep_channel(K_map)
    G = _prep_channel(V_map)
    B = _prep_channel(B_map)
    rgb = np.clip(np.dstack([R, G, B]), 0, 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor('black')
    if extent_cm is not None:
        extent_pc = [v/PC for v in extent_cm]
        ax.imshow(rgb, origin='lower', aspect='equal', extent=extent_pc)
    else:
        ax.imshow(rgb, origin='lower', aspect='equal')

    ax.set_title('RGB Composite')
    ax.set_xlabel('x (pc)')
    ax.set_ylabel('y (pc)')

    if save:
        folder = _ensure_dir(os.path.join(outdir, run_dir) if run_dir else outdir)
        fig.savefig(os.path.join(folder, "rgb_composite.png"),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)

    return fig, ax

def plot_absorption_rgb_grid(results_list, bands=('B','V','K'),
                             save=False, outdir="outputs/figures/final_plots"):
    """
    Build a montage: rows = different N, columns = B, V, K, RGB composite.
    Axes in pc; black plot backgrounds; only column headers at top.
    Per-band maps colored (B→Blues, V→Greens, K→Reds).
    Returns (fig, axes).
    """
    # figure layout
    n_rows = len(results_list)
    n_cols = len(bands) + 1  # + RGB
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.9*n_cols, 3.7*n_rows))
    if n_rows == 1:
        axes = np.atleast_2d(axes)

    # column titles once
    col_titles = list(bands) + ['RGB composite']
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, pad=8)

    # pick band colormaps
    band_cmaps = {'B': 'Blues', 'V': 'Greens', 'K': 'Reds'}

    for i, (N, res) in enumerate(results_list):
        # pc extents (box is always ~1 pc per your setup)
        g_any = res[bands[0]].grid
        xpc = np.array([g_any.lower_bounds[0], g_any.upper_bounds[0]]) / 3.085677581e18
        ypc = np.array([g_any.lower_bounds[1], g_any.upper_bounds[1]]) / 3.085677581e18
        extent_pc = [xpc[0], xpc[1], ypc[0], ypc[1]]

        # per-band panels
        rgb_maps = {}
        for j, b in enumerate(bands):
            ax = axes[i, j]
            ax.set_facecolor('black')
            proj = np.sum(res[b].grid.L_absorbed, axis=2)
            vmax = float(np.max(proj))
            if vmax <= 0:
                ax.imshow(np.zeros((10,10)), origin='lower', extent=extent_pc,
                          cmap=band_cmaps.get(b, 'gray'), vmin=0, vmax=1, aspect='equal')
            else:
                proj_mask = np.ma.masked_less_equal(proj, 0)
                vmin = float(proj_mask.min())
                ax.imshow(proj_mask, origin='lower', extent=extent_pc,
                          cmap=band_cmaps.get(b, 'gray'),
                          norm=LogNorm(vmin=vmin, vmax=vmax),
                          aspect='equal')
            # x/y labels only on outer edges
            if i == n_rows - 1:
                ax.set_xlabel('x (pc)')
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel('y (pc)')
            else:
                ax.set_yticklabels([])
            rgb_maps[b] = proj

        # RGB composite column
        ax_rgb = axes[i, n_cols - 1]
        ax_rgb.set_facecolor('black')
        Bm = np.sum(res['B'].grid.L_absorbed, axis=2) if 'B' in res else np.zeros_like(rgb_maps[bands[0]])
        Vm = np.sum(res['V'].grid.L_absorbed, axis=2) if 'V' in res else np.zeros_like(rgb_maps[bands[0]])
        Km = np.sum(res['K'].grid.L_absorbed, axis=2) if 'K' in res else np.zeros_like(rgb_maps[bands[0]])

        # build rgb (same normalization as single)
        def _prep(img):
            img = np.clip(np.asarray(img, float), 0, None)
            if not np.any(img > 0):
                return np.zeros_like(img)
            p99 = np.percentile(img[img > 0], 99)
            x = img / (p99 + 1e-30)
            return np.arcsinh(2 * x) / np.arcsinh(2)

        rgb = np.clip(np.dstack([_prep(Km), _prep(Vm), _prep(Bm)]), 0, 1)
        ax_rgb.imshow(rgb, origin='lower', extent=extent_pc, aspect='equal')

        # axis labels for RGB column
        if i == n_rows - 1:
            ax_rgb.set_xlabel('x (pc)')
        else:
            ax_rgb.set_xticklabels([])
        ax_rgb.set_yticklabels([])

        # vertical row label
        exp = int(np.floor(np.log10(int(N)))) if N > 0 else 0
        # place near left outside margin, vertically centered on the row
        bbox = axes[i,0].get_position()
        ymid = 0.5*(bbox.y0 + bbox.y1)
        fig.text(0.08, ymid, f"N=1e{exp}", va='center', ha='right',
                 rotation=90, fontsize=11)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.08,
                        wspace=0.08, hspace=0.10)

    if save:
        folder = _ensure_dir(outdir)  # overview spans multiple runs
        fig.savefig(os.path.join(folder, "absorption_rgb_grid.png"),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)

    return fig, axes

def plot_convergence(results_list, bands = ('B','V','K'), save = True, outdir = "outputs/figures/final_plots"):
    """
    Plots convergence curve: 
        f_esc(N) for each band + a reference C * N^-1/2 through the last-N point.
    """

    Ns = np.array([int(N) for (N, _) in results_list], dtype=float)
    order = np.argsort(Ns)
    Ns = Ns[order]

    fig, ax = plt.subplots(figsize=(7,5))

    for b in bands:
        f = np.array([results_list[i][1][b].escape_fraction if b in results_list[i][1] else np.nan
                      for i in order], dtype=float)
        ax.loglog(Ns, f, marker='o', lw=1.5, label=f'{b} band')

        if np.isfinite(f[-1]):
            C = f[-1] * Ns[-1]**0.5
            ax.loglog(Ns, C * Ns**(-0.5), ls='--', alpha=0.6)

    ax.set_xlabel('Packets per band, N')
    ax.set_ylabel(r'$f_{\rm esc}$')
    ax.set_title('Convergence of escape fraction')
    ax.grid(True, which='both', ls=':', alpha=0.35)
    ax.legend()

    if save:
        folder = _ensure_dir(outdir)
        fig.savefig(os.path.join(folder, "convergence_analysis.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)

    return fig, ax

def plot_time_test(n_packets, times, save=True, outdir="outputs/figures/final_plots", filename="time_test.png"):
    """
    Plot performance scaling of MCRT runs.

    Parameters
    ----------
    n_packets : array-like
        Packet counts for each test (e.g., [1e3, 1e4, 1e5, 1e6]).
    times : array-like
        Total runtime in seconds for each run.
    save : bool
        If True, saves plot to `outdir/filename`.
    outdir : str
        Output directory for figures.
    filename : str
        Output filename.

    Produces two plots:
    - Linear scale: N vs runtime
    - Log–log inset or overlay with slope guide
    """

    n_packets = np.asarray(n_packets, dtype=float)
    times = np.asarray(times, dtype=float)

    if np.max(times) < 1e-2:
        times_plot = times * 1e3
        time_label = "Runtime (ms)"
    elif np.max(times) < 60:
        times_plot = times
        time_label = "Runtime (s)"
    else:
        times_plot = times / 60
        time_label = "Runtime (min)"

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(n_packets, times_plot, 'o-', lw=1.8, markersize=6)
    ax.set_xlabel("Packets per band (N)")
    ax.set_ylabel(time_label)
    ax.set_title("MCRT Timing Analysis")
    ax.grid(True, which='both', ls=':', alpha=0.4)

    if save:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, filename)
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)

    return fig, ax

def summarize_tests(test_results):
    """
    Nicely print pass/fail summary from run_tests() results dict.

    Expected structure:
    {
        'Empty Box': True/False or (True/False, extra_info),
        'Opaque Box': ...,
        'Uniform Sphere': (bool, residual),
        'Energy Conservation': (bool, error)
    }
    """
    all_passed = True

    for name, result in test_results.items():
        if isinstance(result, tuple):
            passed = bool(result[0])
            extra = result[1] if len(result) > 1 else None
        else:
            passed = bool(result)
            extra = None

        if not passed:
            print(f"❌ {name}: FAILED", end="")
            if extra is not None:
                print(f" (value: {extra:.3g})", end="")
            print()
            all_passed = False

    if all_passed:
        print("🎉 All validation tests passed!")
    else:
        print("⚠️ Some validation tests failed — check above details.")

    return all_passed

def plot_convergence_error(results_list, bands=('B','V','K'), save=True, outdir="outputs/figures/final_plots",
                           filename="convergence_error.png"):
    """
    Plot |f_esc(N) - f_ref| vs N for each band on log-log axes.
    f_ref = median(f_esc) of the top-K largest N runs.
    Includes labeled N^{-1/2} reference lines.

    results_list : list of (N_packets, results_dict)
    """

    Ns_all = np.array([float(N) for (N, _) in results_list], dtype=float)
    order = np.argsort(Ns_all)
    Ns_all = Ns_all[order]
    R_sorted = [results_list[i][1] for i in order]

    fig, ax = plt.subplots(figsize=(7, 5))

    for b in bands:
        Ns_b, f_b = [], []
        for N_val, res in zip(Ns_all, R_sorted):
            if isinstance(res, dict) and (b in res):
                fesc = getattr(res[b], 'escape_fraction', np.nan)
                if np.isfinite(fesc):
                    Ns_b.append(float(N_val))
                    f_b.append(float(fesc))
        if len(Ns_b) == 0:
            continue

        Ns_b = np.array(Ns_b, dtype=float)
        f_b  = np.array(f_b, dtype=float)
        order_b = np.argsort(Ns_b)
        Ns_b, f_b = Ns_b[order_b], f_b[order_b]

        K = min(2, len(f_b))
        f_ref = np.median(f_b[-K:])

        err = np.abs(f_b - f_ref)
        err_plot = np.where(err > 0, err, 1e-16)  

        ax.loglog(Ns_b, err_plot, 'o-', lw=1.6, label=f'{b} band')

        nz = np.flatnonzero(err > 0)
        if nz.size > 0:
            j = nz[-1]
            N_anchor, E_anchor = Ns_b[j], err[j]
        else:
            N_anchor, E_anchor = Ns_b[-1], max(np.nanmedian(err), 1e-12)

        Ng = np.geomspace(Ns_b[0], Ns_b[-1], 100)
        guide = E_anchor * (Ng / N_anchor) ** (-0.5)
        ax.loglog(Ng, guide, '--', alpha=0.6)
        ax.text(Ng[len(Ng)//4], guide[len(Ng)//4]*1.2,
                r'$N^{-1/2}$', fontsize=9, alpha=0.7)

        print(f"[{b}] N range {Ns_b[0]:.0e}–{Ns_b[-1]:.0e}, "
              f"f_ref={f_ref:.6f}, err_min={np.min(err):.3e}, err_max={np.max(err):.3e}")

    ax.set_xlabel('Packets per band (N)')
    ax.set_ylabel(r'$|\,f_{\rm esc}(N) - f_{\rm ref}\,|$')
    ax.set_title('Convergence of Escape Fraction (Error vs N)')
    ax.grid(True, which='both', ls=':', alpha=0.35)
    ax.legend()
    ax.set_xlim(np.min(Ns_all)*0.9, np.max(Ns_all)*1.1)

    if save:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, filename)
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)

    return fig, ax
