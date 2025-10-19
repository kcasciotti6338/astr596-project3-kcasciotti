# mcrt_viz.py

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

def plot_opacity_validation(draine_data, band_opacities):
    """
    Plot the dust opacity curve and annotate Planck-mean, band-averaged opacities.

    Parameters
    ----------
    draine_data : dict
        Keys: 'wavelength' (cm), 'kappa' (cm^2/g), optionally 'albedo'.
    band_opacities : dict
        Mapping like {'B': kappa_B, 'V': kappa_V, 'K': kappa_K}.
        If Johnson band limits are available via src.constants.BANDS (in cm),
        the function will shade the bandpasses and place points at their
        geometric-mean wavelengths. Otherwise, it’ll place labeled points
        at common Johnson centers (0.44, 0.55, 2.2 μm).
    """
    lam_cm = np.asarray(draine_data['wavelength'])
    kappa = np.asarray(draine_data['kappa'])

    # Convert to microns for plotting
    lam_um = lam_cm * 1e4

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(lam_um, kappa, lw=1.8, label='Opacity curve')

    # Try to import band limits if available
    band_centers_um = {}
    band_spans_um = {}
    try:
        from src.constants import BANDS  # expects {'B': (lam_min_cm, lam_max_cm), ...}
        for b, (lmin_cm, lmax_cm) in BANDS.items():
            c_um = 1e4 * np.sqrt(lmin_cm * lmax_cm)  # geometric mean
            band_centers_um[b] = c_um
            band_spans_um[b] = (1e4 * lmin_cm, 1e4 * lmax_cm)
            # Shade band
            ax.axvspan(band_spans_um[b][0], band_spans_um[b][1],
                       color='0.9', zorder=0, alpha=0.5)
    except Exception:
        # Fallback: typical Johnson centers in μm
        defaults = {'B': 0.44, 'V': 0.55, 'K': 2.2}
        for b in band_opacities.keys():
            if b in defaults:
                band_centers_um[b] = defaults[b]

    # Plot the band-averaged points
    for b, kap in band_opacities.items():
        if b in band_centers_um:
            ax.scatter(band_centers_um[b], kap, s=45, zorder=5)
            ax.text(band_centers_um[b], kap, f'  {b}', va='center')

    ax.set_xlabel('Wavelength [μm]')
    ax.set_ylabel(r'$\kappa$  [cm$^2$ g$^{-1}$]')
    ax.set_title('Opacity curve with band-averaged points')
    ax.grid(True, which='both', ls=':', alpha=0.35)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_convergence_analysis(n_packets_array, f_escape_array):
    """
    Plot convergence diagnostics vs number of packets.

    Two panels:
    1) f_escape vs N (x-axis log scale).
    2) |f(N) - f(N_max)| vs N (log-log) with a 1/sqrt(N) reference line
       normalized to the residual at the smallest N provided.

    Parameters
    ----------
    n_packets_array : array-like of int
        Different packet counts used (monotonic increasing recommended).
    f_escape_array : array-like of float
        Corresponding escape fractions.
    """
    N = np.asarray(n_packets_array, dtype=float)
    fN = np.asarray(f_escape_array, dtype=float)

    if N.size != fN.size or N.size < 2:
        raise ValueError("Provide >=2 matching points for N and f_escape.")

    # Sort by N to be safe
    order = np.argsort(N)
    N = N[order]
    fN = fN[order]

    f_inf = fN[-1]  # best available estimate
    resid = np.abs(fN - f_inf)

    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1.6], hspace=0.2)

    # Top: f_escape vs N
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(N, fN, marker='o', lw=1.6)
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of packets, N')
    ax1.set_ylabel('Escape fraction')
    ax1.set_title('Convergence of escape fraction')
    ax1.grid(True, which='both', ls=':', alpha=0.35)

    # Bottom: residual vs N with 1/sqrt(N) reference
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.loglog(N, resid + 1e-30, marker='o', lw=1.6, label='|f(N) - f_max|')

    # 1/sqrt(N) normalized to match residual at smallest N
    ref = (1.0 / np.sqrt(N))
    if resid[0] > 0:
        ref *= (resid[0] / ref[0])
        ax2.loglog(N, ref, ls='--', label=r'$\propto 1/\sqrt{N}$ (ref.)')

    ax2.set_xlabel('Number of packets, N')
    ax2.set_ylabel('Residual')
    ax2.grid(True, which='both', ls=':', alpha=0.35)
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.show()


def plot_sed(wavelength, L_input_by_band, L_output_by_band):
    """
    Plot a simple SED comparison using band centers.

    Parameters
    ----------
    wavelength : array-like or None
        If provided (array of cm), a thin line will show a guide on the x-range.
        Typically you can pass your Draine wavelength grid here; only x-limits are used.
    L_input_by_band : dict
        {'B': L_B_in, 'V': L_V_in, 'K': L_K_in} in erg/s.
    L_output_by_band : dict
        {'B': L_B_out, 'V': L_V_out, 'K': L_K_out} in erg/s.
    """
    # Determine band centers from constants if available
    centers_cm = {}
    try:
        from src.constants import BANDS
        for b, (lmin, lmax) in BANDS.items():
            centers_cm[b] = np.sqrt(lmin * lmax)
    except Exception:
        # Fallback centers (μm): B=0.44, V=0.55, K=2.2
        fallback_um = {'B': 0.44, 'V': 0.55, 'K': 2.2}
        centers_cm = {b: 1e-4 * fallback_um[b] for b in L_input_by_band.keys() if b in fallback_um}

    # Build arrays
    bands = [b for b in ['B', 'V', 'K'] if b in centers_cm and b in L_input_by_band and b in L_output_by_band]
    x_um = np.array([centers_cm[b] * 1e4 for b in bands])
    y_in = np.array([L_input_by_band[b] for b in bands])
    y_out = np.array([L_output_by_band[b] for b in bands])

    fig, ax = plt.subplots(figsize=(7, 5))

    # Optional x-range guide from wavelength array
    if wavelength is not None:
        w = np.asarray(wavelength)
        if w.size > 1:
            ax.set_xlim(w.min() * 1e4, w.max() * 1e4)

    ax.loglog(x_um, y_in, marker='o', lw=1.6, label='Input (stellar)')
    ax.loglog(x_um, y_out, marker='s', lw=1.6, label='Output (escaped)')

    # Light band labels
    for i, b in enumerate(bands):
        ax.text(x_um[i], y_out[i], f'  {b}', va='center')

    ax.set_xlabel('Wavelength [μm]')
    ax.set_ylabel('Luminosity [erg s$^{-1}$]')
    ax.set_title('Band SED: input vs output')
    ax.grid(True, which='both', ls=':', alpha=0.35)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


def create_rgb_composite(B_map, V_map, K_map):
    """
    Combine B, V, K 2D arrays into an RGB image:
      R <- K, G <- V, B <- B

    Each channel is scaled by its 99th percentile and asinh-stretched
    to preserve dynamic range. Returns the RGB array shown.

    Parameters
    ----------
    B_map, V_map, K_map : 2D ndarray
        Per-band images (e.g., absorption or surface brightness).

    Returns
    -------
    rgb : (H, W, 3) float ndarray
        Values in [0, 1] suitable for imshow.
    """
    def _prep_channel(img):
        img = np.asarray(img, dtype=float)
        img = np.clip(img, a_min=0, a_max=None)
        if not np.any(img > 0):
            return np.zeros_like(img)
        p99 = np.percentile(img[img > 0], 99.0)
        if p99 <= 0:
            p99 = img.max()
        x = img / (p99 + 1e-30)
        # asinh stretch with soft knee
        alpha = 2.0
        return np.arcsinh(alpha * x) / np.arcsinh(alpha)

    R = _prep_channel(K_map)
    G = _prep_channel(V_map)
    B = _prep_channel(B_map)

    # Stack and clip
    rgb = np.dstack([R, G, B])
    rgb = np.clip(rgb, 0.0, 1.0)

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb, origin='lower', aspect='equal')
    plt.title('RGB composite (R=K, G=V, B=B)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return rgb

def plot_absorption_map(grid, band):
    """
    Create 2D projections of absorption.
    Sum along z-axis.
    """
    
    proj = np.sum(grid.L_absorbed, axis=2)

    x_min, x_max = grid.lower_bounds[0], grid.upper_bounds[0]
    y_min, y_max = grid.lower_bounds[1], grid.upper_bounds[1]

    # Handle the "no absorption at all" case
    vmax = proj.max()
    if vmax <= 0:
        plt.figure(figsize=(6, 5))
        # Show a black image as a placeholder
        plt.imshow(
            np.zeros_like(proj).T,
            origin='lower',
            extent=[x_min, x_max, y_min, y_max],
            aspect='equal',
            cmap='Greys',
            vmin=0, vmax=1
        )
        plt.colorbar(label='Absorbed Luminosity (none recorded)')
        plt.xlabel('x [cm]'); plt.ylabel('y [cm]')
        plt.title(f'{band} Band Absorption Map (no absorption yet)')
        plt.tight_layout(); plt.show()
        return

    # Mask zeros so they render as black
    proj_masked = np.ma.masked_less_equal(proj, 0)

    # Build a grayscale cmap where masked/NaN entries are black
    cmap = plt.get_cmap('Greys').copy()
    cmap.set_bad('black')

    # vmin = smallest positive value; vmax = max
    positive_min = proj_masked.min()  # works on masked array

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        proj_masked.T,
        origin='lower',
        extent=[x_min, x_max, y_min, y_max],
        aspect='equal',
        cmap=cmap,
        norm=LogNorm(vmin=positive_min, vmax=vmax)
    )
    plt.colorbar(im, label='Absorbed Luminosity (log scale)')
    plt.xlabel('x [cm]'); plt.ylabel('y [cm]')
    plt.title(f'{band} Band Absorption Map')
    plt.tight_layout(); plt.show()



'''
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