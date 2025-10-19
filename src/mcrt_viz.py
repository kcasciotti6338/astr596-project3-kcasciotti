# mcrt_viz.py

import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

def plot_opacity_validation(draine_data, band_opacities, save=False, outdir="outputs/figures"):
    """
    Plot the dust opacity curve and band-averaged points.
    Returns (fig, ax).
    """
    lam_cm = np.asarray(draine_data['wavelength'])
    kappa = np.asarray(draine_data['kappa'])
    lam_um = lam_cm * 1e4

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(lam_um, kappa, lw=1.8, label='Draine opacity')

    defaults = {'B': 0.44, 'V': 0.55, 'K': 2.2}
    for b, kap in band_opacities.items():
        if b in defaults:
            ax.scatter(defaults[b], kap, s=60, zorder=5)
            ax.text(defaults[b], kap, f' {b}', va='center')

    ax.set_xlabel('Wavelength [μm]')
    ax.set_ylabel(r'$\kappa$ [cm$^2$ g$^{-1}$]')
    ax.set_title('Dust opacity with band averages')
    ax.grid(True, which='both', ls=':', alpha=0.35)
    ax.legend()

    if save:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, "opacity_validation.png"), dpi=200)

    return fig, ax


def plot_convergence_analysis(results_list, band='B', save=False, outdir="outputs/figures"):
    """
    Plot convergence diagnostics using results_list = [(N, results_dict), ...]
    where results_dict is from run_mcrt_jit.
    Returns (fig, (ax1, ax2)).
    """
    Ns = np.array([r[0] for r in results_list], dtype=float)
    f_escape = np.array([r[1][band].escape_fraction for r in results_list], dtype=float)

    order = np.argsort(Ns)
    Ns, f_escape = Ns[order], f_escape[order]

    f_inf = f_escape[-1]
    resid = np.abs(f_escape - f_inf)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), gridspec_kw={'height_ratios':[2,1.6]})
    plt.subplots_adjust(hspace=0.25)

    ax1.plot(Ns, f_escape, 'o-', lw=1.5)
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of packets, N')
    ax1.set_ylabel('Escape fraction')
    ax1.set_title(f'Convergence test – {band}-band')
    ax1.grid(True, ls=':', alpha=0.4)

    ax2.loglog(Ns, resid + 1e-30, 'o-', lw=1.5, label='|f(N) - f_max|')
    ref = (1.0/np.sqrt(Ns))
    ref *= (resid[0]/ref[0]) if resid[0]>0 else 1
    ax2.loglog(Ns, ref, '--', label=r'$\propto 1/\sqrt{N}$')
    ax2.set_xlabel('Number of packets, N')
    ax2.set_ylabel('Residual')
    ax2.grid(True, ls=':', alpha=0.4)
    ax2.legend()

    if save:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, f"convergence_{band}.png"), dpi=200)

    return fig, (ax1, ax2)


def plot_sed(results, bands, save=False, outdir="outputs/figures"):
    """
    Plot input vs escaped luminosity per band from run_mcrt_jit results.
    Returns (fig, ax).
    """
    x = np.arange(len(bands))
    L_in = [results[b].L_input for b in bands]
    L_esc = [results[b].L_escaped for b in bands]

    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.35
    ax.bar(x - width/2, L_in, width, label='Input (stellar)')
    ax.bar(x + width/2, L_esc, width, label='Escaped')
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel('Luminosity [erg s$^{-1}$]')
    ax.set_title('Input vs Escaped Luminosity by Band')
    ax.legend()
    ax.grid(alpha=0.3, ls=':')

    if save:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, "sed_comparison.png"), dpi=200)

    return fig, ax


def create_rgb_composite(B_map, V_map, K_map, save=False, outdir="outputs/figures"):
    """
    Combine B, V, K 2D arrays into RGB image. Returns (fig, ax).
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
    ax.imshow(rgb, origin='lower', aspect='equal')
    ax.set_title('RGB Composite (R=K, G=V, B=B)')
    ax.axis('off')

    if save:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, "rgb_composite.png"), dpi=200)

    return fig, ax


def plot_absorption_map(grid, band, save=False, outdir="outputs/figures"):
    """
    2D projection of absorbed luminosity (sum over z-axis).
    Returns (fig, ax).
    """
    proj = np.sum(grid.L_absorbed, axis=2)
    x_min, x_max = grid.lower_bounds[0], grid.upper_bounds[0]
    y_min, y_max = grid.lower_bounds[1], grid.upper_bounds[1]

    fig, ax = plt.subplots(figsize=(6,5))
    vmax = proj.max()
    if vmax <= 0:
        ax.imshow(np.zeros_like(proj).T, origin='lower', extent=[x_min,x_max,y_min,y_max],
                  aspect='equal', cmap='Greys', vmin=0, vmax=1)
        ax.set_title(f'{band} Band Absorption Map (no absorption)')
    else:
        proj_masked = np.ma.masked_less_equal(proj, 0)
        cmap = plt.get_cmap('Greys').copy()
        cmap.set_bad('black')
        vmin = proj_masked.min()
        im = ax.imshow(proj_masked.T, origin='lower',
                       extent=[x_min,x_max,y_min,y_max], aspect='equal',
                       cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
        fig.colorbar(im, ax=ax, label='Absorbed Luminosity (log scale)')
        ax.set_title(f'{band} Band Absorption Map')
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    plt.tight_layout()

    if save:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, f"absorption_map_{band}.png"), dpi=200)

    return fig, ax


def make_plots(results, draine_data=None, bands=['B','V','K'], save=True):
    """
    Generate and optionally save all standard plots from run_mcrt_jit results.
    Returns a dict of (fig, ax) tuples.
    """
    figs = {}

    if draine_data is not None:
        band_opacities = {b: results[b].grid.L_absorbed.sum() for b in bands if b in results}
        figs['opacity'] = plot_opacity_validation(draine_data, band_opacities, save=save)

    figs['sed'] = plot_sed(results, bands, save=save)

    for b in bands:
        figs[f'{b}_absorption'] = plot_absorption_map(results[b].grid, b, save=save)

    return figs




''' Starter code
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

def plot_opacity_validation(draine_data, band_opacities):
    """
    Plot opacity curve with band averages.
    """
    pass

def plot_convergence_analysis(n_packets_array, f_escape_array):
    """
    Plot escape fraction vs number of packets.
    Include 1/sqrt(N) reference line.
    """
    pass

def plot_sed(wavelength, L_input_by_band, L_output_by_band):
    """
    Plot input vs output SED.
    """
    pass

def create_rgb_composite(B_map, V_map, K_map):
    """
    Create RGB image from three bands.
    """
    pass

'''