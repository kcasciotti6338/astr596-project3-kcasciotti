# Project 3: Monte Carlo Radiative Transfer in a Dusty Star Cluster

**ASTR 596: Modeling the Universe**
**Instructor:** Dr. Anna Rosen
**Due:** Friday October 17, 2025 by 11:59 PM  
**Working Mode:** Pair Programming

*Pair Programming Assignments:*

- **Pair 1:** Caden + Kasia
- **Pair 2:** Hy + Nodoka
- **Triple:** Paige + Aisling + Katie

Core algorithms must be independently implemented. You may share debugging strategies, test cases, and plotting code.

## Learning Objectives

Upon successful completion of this project, you will be able to:

**Core Physics Understanding:**

- **Explain** how Monte Carlo methods exactly solve the radiative transfer equation through statistical sampling rather than approximation.
- **Connect** discrete absorption to the exponential probability distributions inherent in radiative transfer.
- **Demonstrate** that photon packet propagation naturally samples Beer's law ($I = I_0 e^{-\tau}$) and reproduces the formal solution.
- **Analyze** wavelength-dependent extinction and connect it to dust grain physics (e.g., why $\kappa_B \gg \kappa_K$).

**Computational Skills:**

- **Implement** Monte Carlo radiative transfer algorithms for 3D homogeneous media with multiple sources.
- **Apply** inverse transform sampling to generate optical depths from the distribution $p(\tau) = e^{-\tau}$.
- **Design** efficient ray-marching algorithms through gridded density fields with proper boundary handling.
- **Construct** luminosity-weighted sampling schemes for multiple stellar sources across wavelength bands.

**Validation & Analysis:**

- **Validate** numerical codes against analytical solutions ($f_{\text{esc}} = e^{-\tau}$ for uniform media).

- **Quantify** Monte Carlo convergence and verify its $1/\sqrt{N}$ error scaling behavior.

- **Demonstrate** energy conservation to 0.1% precision through careful luminosity bookkeeping.

- **Compare** multi-band escape fractions to understand differential extinction $(K > V > B)$.

**Scientific Interpretation:**

- **Predict** observational signatures of embedded star clusters in color-magnitude diagrams.

- **Interpret** how dust geometry creates shadows, reddening, and differential extinction patterns.

- **Connect** computational results to real observations (e.g., JWST's ability to penetrate dust in IR).

- **Assess** how stellar position within the cluster affects individual escape probabilities.

## Project Overview \& Scientific Context

### Mission Brief

You're modeling a young star cluster embedded in a dusty molecular cloud. Your goal: build a Monte Carlo radiative transfer code to predict how dust extinction affects the observed starlight. This project connects microscopic physics (dust absorption) to macroscopic observables (colors and extinction).

### Required Background from Module 4

This project builds directly on Module 4's comprehensive radiative transfer framework. Please ensure you have thoroughly reviewed the following materials from the course website's **Statistical Thinking - Module 4: From Photons to Information**:

- **[Part I: Part I: The Hidden Physics in Every Astronomical Image](https://astrobytes-edu.github.io/astr596-modeling-universe/mod4-part1-light/)**
- **[Part II: Mathematical Foundations of Radiative Transfer](https://astrobytes-edu.github.io/astr596-modeling-universe/mod4-part2-rte/)**
- **[Part III: Monte Carlo Solutions](https://astrobytes-edu.github.io/astr596-modeling-universe/mod4-part3-mcrt/)**

### Key Concepts You Should Understand

Before starting this project, you should be comfortable with the following concepts:

- [ ] The radiative transfer equation and its statistical interpretation
- [ ] Why interaction distances follow an exponential distribution
- [ ] The difference between photon packets and individual photons
- [ ] How discrete absorption works (all or nothing energy deposition)
- [ ] Your ZAMS models from Project 1 (Tout et al. 1996 formulae)

### The Physical Picture: Monte Carlo Radiative Transfer

Monte Carlo Radiative Transfer (MCRT) solves the radiative transfer equation through a **stochastic process** â€” meaning it uses randomness and probability to represent physical reality. Each "photon packet" represents a bundle of photons carrying a fraction of the source luminosity. The stochastic nature manifests in two key ways: (1) the random sampling of interaction distances from an exponential distribution, and (2) the random emission directions. Through the law of large numbers, these random processes converge to the deterministic solution of the radiative transfer equation.

**Discrete Absorption**: In this project, you'll implement discrete absorption where each packet either:

1. **Deposits ALL its energy** at a single interaction point (when the sampled optical depth $\tau$ is reached), or
2. **Escapes with ALL its energy** (if it reaches the boundary first)

There is NO gradual energy loss - it's all or nothing! As $N \to \infty$, this stochastic Monte Carlo approach converges to the exact solution of the radiative transfer equation.

## The Physical System

### Grid Setup

- **Domain:** Cubic box extending from $(-L/2, -L/2, -L/2)$ to $(L/2, L/2, L/2)$ where $L = 1$ pc.

- **Resolution:** $128^3$ Cartesian cells (use CGS units: 1 pc = $3.086 \times 10^{18}$ cm)

- **Cell size:** $\Delta x = L/128$ in each dimension.

- **Cell indexing:** For position $(x, y, z)$, the cell indices are:
  - $i_x = \text{int}((x + L/2) / \Delta x)$
  - $i_y = \text{int}((y + L/2) / \Delta y)$
  - $i_z = \text{int}((z + L/2) / \Delta z)$
  - Where int() denotes integer division (floor function)

#### Star Cluster Configuration

**5 ZAMS stars** with the following properties ($Z=0.02$):

| Star | Mass ($M_\odot$) | $T_{\text{eff}}$ (K) | $L$ ($L_\odot$) | $R$ ($R_\odot$) | Spectral Type |
|------|------------------|------------------------|-----------------|-----------------|---------------|
| 1 | 30 | ~38,500 | ~119,000 | ~7.7 | O7V |
| 2 | 20 | ~34,000 | ~43,000 | ~6.0 | O9V |
| 3 | 10 | ~25,000 | ~5,500 | ~3.9 | B0V |
| 4 | 5 | ~17,000 | ~530 | ~2.6 | B5V |
| 5 | 2 | ~9,100 | ~16 | ~1.6 | A5V |

*Note: Use your ZAMS models from Project 1 to calculate exact values.*

- **Stellar positions:** Uniformly distributed random positions within a $0.75^3$ pc$^3$ sub-volume centered at origin (this keeps stars away from boundaries to avoid edge effects)
- **Luminosities and temperatures:** From ZAMS models

#### Dusty Medium

**Density Calculations:**

- **Number density:** $n_H = 1000$ cm$^{-3}$ (molecular hydrogen)

- **Molecular weight:** $\mu = 2.3$ (for H$_2$)

- **Proton mass:** $m_H = 1.67 \times 10^{-24}$ g

- **Gas density:** $\rho_{\text{gas}} = n_H \times \mu \times m_H = 3.84 \times 10^{-21}$ g/cm$^3$

- **Dust-to-gas mass ratio:** $f_{D/G} = 0.01$

- **Dust density:** $\rho_{\text{dust}} = \rho_{\text{gas}} \times f_{D/G} = 3.84 \times 10^{-23}$ g/cm$^3$

**Dust Model:** [Draine (2003a,b)](https://ui.adsabs.harvard.edu/abs/2003ApJ...598.1017D/abstract) Milky Way dust with $R_V = 5.5$

- Data file: [kext_albedo_WD_MW_5.5B_30](https://www.astro.princeton.edu/~draine/dust/extcurvs/kext_albedo_WD_MW_5.5B_30)

**Note on units:** Use CGS units as your default throughout this project.

**Reading the Draine opacity file:**

- Skip header lines (~80 lines) when reading the file, these contain metadata

- Look for the line containing "dust/H =" which gives the dust mass per H atom (typically ~1.4Ã—10â»Â²â¶ g)

- Data columns: wavelength(Î¼m) | C_ext/H | albedo | g | ...

- Convert to mass absorption coefficient: Îº_dust = (C_ext/H) / (dust_mass/H)

- Example: If C_ext/H = 5Ã—10â»Â²Â¹ cmÂ² and dust/H = 1.4Ã—10â»Â²â¶ g, then Îº = 3.6Ã—10âµ cmÂ²/g

### Band-Averaged Opacities

You'll calculate **Planck mean opacities** for each band by integrating the Draine data weighted by each star's Planck function:

$$\langle\kappa\rangle_{\text{band,star}} = \frac{\int_{\lambda_1}^{\lambda_2} \kappa(\lambda) B_\lambda(T_{\text{eff,star}}) d\lambda}{\int_{\lambda_1}^{\lambda_2} B_\lambda(T_{\text{eff,star}}) d\lambda}$$

**Why Planck mean opacity?** Each star emits radiation following its Planck function $B_Î»(T_\text{eff})$. The Planck mean weights the dust opacity by the actual spectrum of light emitted by the star. This gives the effective opacity "seen" by photons from that specific star. Since hot stars emit more blue light and cool stars emit more red light, each star experiences a different effective dust opacity in each band.

**Important:** Calculate separate opacities for each star-band combination. A 38,500 K star will have different band-averaged opacities than a 9,100 K star, even using the same dust model, because their emission spectra weight the wavelength-dependent $Îº(Î»)$ differently.

**Tip:** When creating each star, compute and store $\langle\kappa\rangle_\text{band}$ for each band in a dictionary or array as a `star` attribute for quick lookup during packet emission.

**Primary Bands:**

| Band | Wavelength Range | Central $\lambda$ | Notes |
|------|-----------------|-----------|-------|
| B | 390-500 nm | 445 nm | Blue optical |
| V | 500-600 nm | 551 nm | Visual |
| K | 1.95-2.40 $\mu$m | 2.19 $\mu$m | Near-infrared |

**Important:** Convert all wavelengths to cm for CGS consistency (1 nm = $10^{-7}$ cm, 1 $\mu$m = $10^{-4}$ cm)

**Conversion from Draine file:** The file provides $C_{\text{ext}}/H$ (extinction cross-section per H atom). Convert to mass absorption coefficient:
$$\kappa_{\text{dust}} = \frac{C_{\text{ext}}/H}{M_{\text{dust}}/H}$$
where $M_{\text{dust}}/H$ is given in the file header.

### Stellar Sources and Band Luminosities

Update your `star.py` class from Project 1 to calculate band-specific luminosities. Each star emits differently in each wavelength band based on its temperature:

$$\langle L \rangle_{\text{band}} = L_{\text{bol}} \times \frac{\int_{\lambda_1}^{\lambda_2} B_\lambda(T_{\text{eff}}) d\lambda}{\int_{0}^{\infty} B_\lambda(T_{\text{eff}}) d\lambda}$$

where the denominator equals $\sigma T_{\text{eff}}^4 / \pi$ (from integrating the Planck function over all wavelengths).

The Planck function is:
$$B_\lambda(T) = \frac{2hc^2}{\lambda^5} \frac{1}{e^{hc/(\lambda kT)} - 1}$$

**Important:** Each star's band luminosity depends on its temperature. Hot stars emit predominantly in B-band while cool stars emit more in K-band. Calculate $\langle L \rangle_{\text{band}}$ for each star individually.

## Key Physics You'll Implement

### Optical Depth and Absorption

The heart of Monte Carlo radiative transfer is determining where photons interact with dust:

- Sample interaction optical depth: $$\tau_{sample} = -\ln(\xi)$ where $\xi \sim \mathcal{U}(0,1)$ (uniform random number between 0 and 1).

- Use **discrete absorption**:
  - March packet through grid accumulating optical depth:

  $$ \tau_\text{accumulated} = \sum \kappa \rho_\text{dust} \Delta s$$

  where $\Delta s$ is the distance traveled in the current cell.

  - When $\tau_\text{accumulated} \geq \tau_\text{sample}$: deposit ALL packet luminosity at that point

  - If packet reaches boundary before interaction: escapes with ALL luminosity

- For each band (B, V, K), each packet carries:

$$
L_{\text{packet}} = \frac{L_{\text{band,total}}}{N_{\text{packets}}} 
\text{ where} L_{\text{band,total}} = \sum_{\text{stars}} L_{\text{star,band}}
$$

  (sum over all stars for THIS band only). This ensures uniform Monte Carlo statistics within each band's simulation - every packet in that band contributes equally to the error regardless of which star emits it.

**Critical Point**: There is no partial absorption! Each packet either deposits 100% of its energy or 0%. This binary outcome, averaged over millions of packets, reproduces the continuous radiation field.

### Packet Emission Geometry

While stellar radii are tiny compared to the box size ($R_{\text{star}}/L_{\text{box}} \sim 10^{-7}$), packets must start from the stellar surface:

**Initial position:** Uniformly sampled on stellar sphere

- Sample two uniform random numbers: $\xi_1, \xi_2 \in (0,1)$

- Convert to spherical coordinates on the unit sphere:

  - $\theta = \arccos(2\xi_1 - 1)$ (polar angle, uniform in $\cos\theta$)

  - $\phi = 2\pi\xi_2$ (azimuthal angle, uniform in $\phi$)

- Position on stellar surface. Let star position be $\vec{r}_* = (x_*, y_*, z_*)$ then:

  - $x = x_* + R_{\text{star}} \times \sin\theta \times \cos\phi$

  - $y = y_* + R_{\text{star}} \times \sin\theta \times \sin\phi$

  - $z = z_* + R_{\text{star}} \times \cos\theta$

**Initial direction:** Independently sampled isotropic direction

- Sample two NEW uniform random numbers: $\xi_3, \xi_4 \in (0,1)$ (different from position sampling!)

- Convert to direction on unit sphere:

  - $\theta_{\text{dir}} = \arccos(2\xi_3 - 1)$

  - $\phi_{\text{dir}} = 2\pi\xi_4$

- Unit direction vector:

$$\hat{n} = (\sin\theta_{\text{dir}}\cos\phi_{\text{dir}}, \sin\theta_{\text{dir}}\sin\phi_{\text{dir}}, \cos\theta_{\text{dir}})$$

***Important note:** Store initial position and $\hat{n}$ as photon attributes!*

### Packet Propagation

Since we're modeling pure absorption (no scattering), photon packets travel in straight lines:

- **Direction doesn't change:** Once emitted with direction $\hat{n}$, the packet maintains this direction

- **Position update:**

$$\vec{r}_{new} = \vec{r}_{old} + \Delta s \cdot \hat{n}$$

where $\Delta s$ is the distance to the next cell boundary or interaction point.

- **Boundary checking:** If $\vec{r}_{new}$ crosses any box boundary, the packet escapes.

- **Cell indexing:** Update cell indices after each step to determine local density and opacity.

- **Cell boundary crossing:** Calculate distance to next cell boundary in each dimension and take the minimum $\Delta s$ to step to the next cell.

- **Optical depth accumulation:** $\tau_{accumulated} = \tau_{accumulated} + \kappa \rho_{dust} \Delta s$

**Tip:** Handle the special case where a direction component is exactly zero (e.g., dx = 0) by setting the distance to that boundary to infinity rather than dividing by zero.

### Expected Opacity Values

| Band | $\langle\kappa\rangle$ (cm$^2$/g) | Why This Value? |
|------|--------------------------------|-----------------|
| B | 35,000-36,000 | Peak dust extinction near 220 nm |
| V | 30,000-31,000 | Still strong extinction |
| K | 3,400-3,500 | Wavelength $\gg$ grain size |

These apply to dust mass, not gas mass!

**Important Note:** Your exact escape fraction values will differ from others due to random stellar positions within the cluster. The relative ordering $(K > V > B)$ must always hold, but absolute values depend on where stars are placed relative to the box boundaries.

## Computational Strategy

### Grid Resolution Recommendations

- **Initial testing:** $64^3$ grid with $10^3$ packets (fast debugging)

- **Development:** $128^3$ grid with $10^4$ packets (see trends)

- **Analysis:** $128^3$ grid with $10^5$ packets (smooth statistics)

- **Final results:** $128^3$ or $256^3$ grid with $\geq 10^6$ packets (minimizes Monte Carlo noise to ~0.1%)

**Why this strategy:** Start with low resolution to debug your algorithm quickly. Once working, increase resolution to see physical trends emerge. For final science analysis, use high packet counts to minimize Monte Carlo noise. The $128^3$ resolution balances memory usage with spatial resolution of dust structures.

### Performance Measurement

Before running $10^6$ packets:

1. Time your code with 1000 packets
2. Calculate packets/second
3. Estimate time for full runs
4. If estimated time > 24 hours, profile your code to find bottlenecks (see Appendix A on using `cProfile`)

## Implementation Phases

### Phase 0: Core Infrastructure - Get Your Framework Working

Build your modular MCRT framework with proper project structure:

```bash
project3_mcrt/
├── src/
│   ├── __init__.py
│   ├── constants.py      # Copy from Project 1, ensure CGS units
│   ├── star.py           # Update from Project 1 with band luminosities
│   ├── zams.py           # Copy from Project 1 (Tout et al. formulae)
│   ├── utils.py          # Planck function, integration utilities
│   ├── dust.py           # Draine opacity processing
│   ├── grid.py           # 3D grid structure (CGS units)
│   ├── photon.py         # Packet propagation
│   ├── transport.py      # Main MCRT engine
│   ├── detectors.py      # Escape tracking, observables
│   └── mcrt_viz.py       # Visualization and plotting functions
├── data/
│   └── kext_albedo_WD_MW_5.5B_30
├── outputs/
│   ├── figures/
│   └── results/
├── docs/
│   ├── research_memo.md
│   └── growth_memo.md
├── tests/
│   └── test_validation.py
├── README.md
├── requirements.txt
└── project3_analysis.py  # Main analysis script
```

**Important:**

- Copy your `constants.py`, `star.py`, and `zams.py` from Project 1
- Update `star.py` to include band luminosity calculations
- Ensure all units are CGS throughout your code
- Use numpy.random for random number generation
- During debugging, set seed for reproducibility: `np.random.seed(42)`
- Remove seed setting for final runs to ensure proper Monte Carlo statistics

**Create `utils.py` with:**

- Planck function (Blackbody intensity):
$$B_\lambda(T) = \frac{2hc^2}{\lambda^5} \frac{1}{e^{hc/(\lambda kT)} - 1}$$

- Integration wrapper using `scipy.integrate.simpson`
- Band integration function to compute luminosity fractions and dust opacities

**Create `mcrt_viz.py` with:**

- Functions for all required plots
- 2D projection maps
- SED plotting
- Convergence analysis plots

**Validation:** Verify grid indexing and coordinate transformations are correct

### Phase 1: Single Band Implementation

Start with one band (suggest V-band), then expand:

**Part A - Single Star:**

- Place one $20~M_\odot$ star at box center
- Implement absorption-only transport for V-band only
- Verify: $f_{\text{esc}} = \exp(-\tau)$ for uniform medium
- Check energy conservation to 0.1%

**Part B - All Stars:**

- Add all 5 stars at random positions
- Implement luminosity-weighted packet emission (still V-band only)
- Verify energy conservation still holds
- Generate V-band absorption map

### Phase 2: Multi-Band Analysis

Expand to all three bands and increase packet counts:

- Add B-band and K-band calculations
- Run with $10^5$ packets minimum
- Create output SED showing reddening
- Generate absorption maps for all bands
- Analyze differential extinction

### Phase 3: Science Analysis and Extensions

- Perform convergence study with $N_\text{packets} = 10^3, 10^4, 10^5, 10^6$ (attempt $10^6$ if feasible)

Complete required outputs and implement extensions (see Grading Rubric).

## Quick Sanity Checks

Before diving into analysis, verify your code with these tests:

1. **Empty box test:** Set $\rho_{dust} = 0$ everywhere $\to$ 100% of packets should escape

2. **Extreme opacity test:** Set $\kappa \times \rho_{dust}$ very high $\to$ nearly 0% escape (Note: high opacity reduces runtime during debugging since packets absorb quickly)

3. **Energy conservation:** $|L_{in} - (L_{abs} + L_{esc})| / L_{in} < 0.001$ for all runs

4. **Statistical convergence:** Error should scale as $1/\sqrt{N_{packets}}$

5. **Single star centering:** Star at origin in uniform medium should give $f_{esc} = e^{-\tau}$

### Common Debug Checks

If your results seem wrong:

1. Verify $\tau = -\ln(\xi)$ not $+\ln(\xi)$

2. Check you're using $\rho_\text{dust}$ not $\rho_{gas}$

3. Confirm packets that escape are counted in $L_\text{escaped}$

4. Test with one packet and print every step

5. Verify your random number generator is working (should give uniform distribution)

### Required Outputs \& Analysis for Your Research Memo

1. **Opacity Validation Plot:**

Validate your opacity calculations from the Draine data:

- X-axis: Wavelength (0.3-3 $\mu$m, log scale)
- Y-axis: Mass absorption coefficient $\kappa$ (cm$^2$/g, log scale)
- Show the Draine opacity curve for $R_V = 5.5$
- Overlay vertical shaded bands for B, V, and K filters
- Include points (`plt.scatter`) or horizontal lines (`plt.axhline`) showing calculated band-averaged opacities

2. **Convergence Analysis:**

Plot $f_{\text{esc}}$ vs $N_{\text{packets}}$ for each band:

- X-axis: Number of packets ($10^3$ to $10^6$, log scale)

- Y-axis: Escape fraction

- Three curves: B-band, V-band, K-band
- Include $1/\sqrt{N}$ reference line

3. **Spectral Energy Distribution**

Show how dust reddens the star cluster:

- X-axis: Wavelength $\lambda$ (mark B, V, K centers)
- Y-axis: $\lambda L_\lambda$ / max($\lambda L_\lambda$)
- Two curves:
  - **Intrinsic**: Combined stellar emission (sum of all 5 stars' Planck functions)
  - **Observed**: Escaped light after dust extinction (from MCRT simulation)
- Divide both curves by the maximum value of the intrinsic curve
- The observed curve should be suppressed at blue wavelengths relative to red

4. Spatial Aborption Maps (2D Projections)

Create absorption maps for each band (B, V, K) by **integrating through the full box**:

$$\sum_z L_\text{absorbed, band}(x,y,z)$

Use log color scale to show dynamic range, choose your limits wisely.

5. **Escape Direction Map (optional)**

2D histogram showing the angular distribution of escaping light:

- Use spherical coordinates $(θ, φ)$ for packet escape directions
- Create 2D histogram with $θ~(0 \to π)$ and $φ~(0 \to 2π)$ bins
- Color shows escaped luminosity per solid angle
- Should Reveal anisotropies due to stellar positions within the box

#### 6. Data Table

| Quantity | B-band | V-band | K-band |
|----------|--------|--------|--------|
| Band-averaged opacity $\kappa$ (cm$^2$/g) | | | |
| Input luminosity ($L_\odot$) | | | |
| Escaped luminosity ($L_\odot$) | | | |
| Escape fraction | | | |
| Mean optical depth | | | |

### Analysis Discussion for Research Memo

Your memo should address both the physics and computational aspects. These topics are suggestions to help guide your analysis - pursue the aspects you find most interesting in your results.

**Physics Results:**

- How dust extinction creates spectral reddening in your SED
- Why escape fractions follow K > V > B (connect to dust opacity physics)
- Role of stellar position and luminosity in determining escape probabilities

**Numerical Analysis:**

- Convergence behavior and verification of $1/\sqrt{N}$ Monte Carlo error scaling
- Energy conservation accuracy achieved
- Computational performance and any optimizations implemented

**Interpretation:**

- What your results reveal about observing embedded clusters
- How different parameters (density, dust model, inhomogeneous medium) would affect outcomes

## Critical Implementation Notes

### Key Points for Success

- **Use dust density:** $\rho_{dust} = \rho_{gas} \times f_{dust-to-gas}$ where $f_{dust-to-gas} = 0.01$

- **Correct sign:** Sample $\tau = -\ln(\xi)$ (negative sign critical!)

- **Energy tracking:** Maintain separate counters for absorbed and escaped luminosity

- **Boundary checking:** Test all 6 box faces for packet escape

- **Packet luminosity:** All packets carry $L_{packet} = L_{total}/N_{packets}$ regardless of source

- **CGS units:** Ensure all calculations use CGS units consistently

### CGS Unit Conversions

- 1 pc = $3.086 \times 10^{18}$ cm
- 1 $R_\odot$ = $6.96 \times 10^{10}$ cm  
- 1 $L_\odot$ = $3.828 \times 10^{33}$ erg/s
- 1 nm = $10^{-7}$ cm, 1 $\mu$m = $10^{-4}$ cm

### Common Pitfalls to Avoid

1. Using gas density instead of dust density (100\% — error!)
2. Sign error in tau sampling (packets never absorb)
3. Lost packets at boundaries (energy not conserved)
4. Variable packet luminosities (breaks statistics)
5. Not integrating opacities properly (missing temperature weighting)
6. Wrong units in wavelength conversion (nm/$\mu$m to cm)

## Validation Requirements

Before analyzing results, your code MUST pass:

### Test 1: Uniform Sphere

Single star at center, uniform medium:

- Analytical: $f_{esc} = \exp(-\tau_{edge})$
- Numerical result must agree within $3\sigma$ Monte Carlo error
- Where $\sigma = \sqrt{f_{esc}(1-f_{esc})/N_{packets}}$

### Test 2: Energy Conservation

$$\left|\frac{L_{in} - (L_{abs} + L_{esc})}{L_{in}}\right| < 0.001$$

### Test 3: Convergence Scaling

Standard error $\propto N^{-1/2}$ for $N \in [10^3, 10^6]$

## Grading Rubric

| Component | Points | Focus |
|-----------|--------|-------|
| **Core Implementation** | 40 | Correct MCRT algorithms, multi-band treatment, energy conservation |
| **Code Design & Quality** | 20 | Modular structure, documentation, efficiency, readability |
| **Research Memo & Analysis** | 25 | Physics interpretation, required visualizations, quality of discussion |
| **Extension** | 10 | Implementation quality and comparison with baseline |
| **Validation Tests** | 5 | Energy conservation, uniform sphere test, convergence verification |

### Extension Requirement (Choose 1)

Implement ONE extension and compare results with the baseline. The list below provides ideas to get you started, but you're encouraged to propose your own extension that interests you. If you have an idea or want to discuss feasibility, come chat with me during office hours.

**Physics Extensions:**

- Isotropic scattering with albedo ω = 0.6 and compare escape fractions with/without scattering. (Note: this will increase runtime.)
- Different dust models: Compare $R_V = 3.1$ vs $R_V = 5.5$ using provided Draine files
- Temperature calculation: Compute dust temperature from absorbed energy using $\sigma T^4 = L_{abs}/A_{cell}$

**Computational Extensions:**

- Performance optimization using `numba` Just-in-Time (JIT) compilation (demonstrate speedup). Click [here](https://numba.readthedocs.io/en/stable/) for documentation and [here](https://numba.readthedocs.io/en/stable/user/5minguide.html) for a 5-minute quickstart guide.
- Parallel processing with `multiprocessing` (run bands in parallel). Click [here](https://docs.python.org/3/library/multiprocessing.html) for documentation and [here](https://www.geeksforgeeks.org/python/parallel-processing-in-python/) for a quick tutorial.
- Variance reduction: Implement forced first interaction or importance sampling

**Observational Extensions:**

- Extended wavelength coverage: Add U, R, I bands and analyze full optical SED
- Color-magnitude diagram: Plot (B-V) vs V for input and output stellar populations
- Escape anisotropy: Analyze directional dependence of escaped light

**Analysis Extensions:**

- Density parameter study: Vary $n_H$ from 10 to 10,000 cm$^{-3}$
- Non-uniform density: Implement $\rho(r) \propto r^{-2}$ profile or exponential disk
- Turbulent density field: Add lognormal density fluctuations to uniform medium (e.g., $\rho = \rho_0 \times 10^{\mathcal{N}(0,\sigma)}$)
- Statistical study: Multiple realizations with different random stellar positions

**Your own idea:** Propose something that connects to your research interests or explores an aspect of radiative transfer you find intriguing.

Your extension must include quantitative comparison with baseline and discussion of physical implications.

## Implementation Tips

Before you begin coding, these practical tips will save you hours of debugging:

**Performance & Efficiency:**

- When creating each star, compute and store $\langle\kappa\rangle_{\text{band,star}}$ for each band as a star attribute for quick lookup during packet emission - don't recalculate millions of times!

- Pre-calculate and store grid boundaries (`x_min`, `x_max`, etc.) and cell size (Î”x) as attributes to avoid recalculating them during packet propagation

- Save intermediate results (e.g., after every $10^4$ packets) to avoid losing progress if your code crashes during long runs - Monte Carlo results are additive

**Debugging Strategy:**

- Start with `kappa` as a simple constant (e.g., 30000 cm²/g) before implementing the full Draine integration to isolate algorithmic bugs from opacity calculation bugs
  
- During debugging, use `np.random.seed(42)` for reproducible results. Remove the seed for final runs to ensure proper Monte Carlo statistics

- If your calculated opacities differ significantly from expected ranges, check: (1) wavelength unit conversion to cm, (2) correct reading of dust/H mass from header, (3) proper Planck weighting with stellar temperature

**Numerical Robustness:**

- Use DIFFERENT random numbers for position sampling ($\xi_1$‚ $\xi_2$) and direction sampling $\xi_3$‚ $\xi_4$)  - reusing random numbers creates unwanted correlations

- Handle the special case where a direction component is exactly zero (e.g., dx = 0) by setting the distance to that boundary to infinity rather than dividing by zero

- Always check that `d_next` > 0 before moving packet - negative or zero distances indicate a bug in boundary calculations

## Getting Started

1. **Review** Module 4 lecture notes
2. **Copy** your `constants.py`, `star.py`, and `zams.py` from Project 1
3. **Start simple:** Single packet, single star, uniform medium
4. **Build incrementally:** Add complexity only after validation
5. **Test constantly:** Every new feature needs a test

## What Success Looks Like

- [ ] B-band escape fraction is lowest  
- [ ] K-band shows dramatically less extinction than B-band  
- [ ] Shadow patterns visible in absorption maps  
- [ ] Energy conserved to 0.1% precision  
- [ ] Results converge smoothly with increasing $N$  
- [ ] Opacity values match expected ranges  
- [ ] All sanity checks pass

---

## Appendix A: Using Python's cProfile for Performance Analysis

### What is cProfile?

cProfile is Python's built-in profiler that measures where your code spends time. It tracks every function call and measures execution time, helping identify bottlenecks.

### Basic Usage

**Method 1 - Command line:**

```bash
python -m cProfile -s cumtime project3_analysis.py > profile_output.txt
```

The `-s cumtime` sorts by cumulative time spent in each function.

**Method 2: In your script:**

```python
import cProfile
import pstats

# Profile a specific function
profiler = cProfile.Profile()
profiler.enable()

# Your MCRT code here
results = run_mcrt(n_packets=1000)

profiler.disable()

# Print statistics
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(20)  # Show top 20 functions
```

### Interpreting Output

```{bash}

ncalls  tottime  percall  cumtime  percall filename:lineno(function)
1000    0.234    0.000    45.678   0.046   transport.py:45(propagate_packet)
128000  12.345   0.000    12.345   0.000   grid.py:23(get_cell_index)
```

- **ncalls**: Number of times function was called
- **tottime**: Time spent in this function (excluding sub-functions)
- **cumtime**: Total time in function (including sub-functions)
- **percall**: Time per call

### What to Look For

1. Functions with high cumtime - these are your bottlenecks
2. Functions called millions of times with small percall time - consider vectorization
3. Unexpected functions taking significant time - may indicate bugs

### Example Optimization Workflow

```python
# Before optimization: profile your code
# Identify that get_cell_index takes 30% of runtime
# Optimize that specific function
# Re-profile to verify improvement
```

**Remember:** Profile before optimizing! Don't guess where the bottlenecks are.

---

## Appendix B: Parallel Processing for Bands

### Optional Speedup: Band Parallelization

Since B, V, and K bands are independent, you can run them simultaneously for ~3× speedup using Python's `multiprocessing` module ([click here for docs](https://docs.python.org/3/library/multiprocessing.html)).

**Key concept:** Each band runs as a separate process with its own random seed and absorption map. After all bands complete, combine the results.

**Considerations:**

- Each process needs a different random seed for independent Monte Carlo sampling
- Return results from each process rather than modifying shared objects
- For small test runs (<10^4 packets), parallelization overhead may make it slower

**Starting point:** Look into `multiprocessing.Pool` and its `map()` function to distribute bands across cores.

This optimization is entirely optional - focus on getting correct physics first.

---

## Appendix C: Troubleshooting Checklist

If your results seem incorrect, check these common issues:

- **f_esc = 100% for all bands:** You're using rho_gas instead of rho_dust
- **f_esc = 0% for all bands:** Sign error in tau sampling (should be -ln(xi))
- **Energy off by factor of 100:** Not applying dust-to-gas ratio (f_D/G = 0.01)
- **No convergence with N:** Variable packet luminosities (all packets must carry L_total/N)
- **Packets disappear:** Not tracking both absorbed AND escaped packets
- **Code extremely slow:** Profile first with cProfile before optimizing

*"Monte Carlo: Solving intractable integrals by rolling dice since 1946"*