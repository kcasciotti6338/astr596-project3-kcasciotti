# Project Description: 

This project was to simulate the radiative transfer of starlight through a dusty molecular cloud using the Monte Carlo method. The goal was to understand how dust extinction and absorption affect the observed emission from an embedded star cluster. Each photon packet emitted by a star either escapes the box or is absorbed by dust, following the discrete absorption method.

I modeled a cluster of five zero-age main sequence (ZAMS) stars, each with individually computed Planck-mean opacities in the B, V, and K bands based on the Draine (2003) dust model. The code tracks photon packets as they propagate through a one-parsec cube, interacting with dust according to its density and wavelength-dependent opacity. I also implemented a full validation suite, including empty box, opaque box, uniform sphere, and energy-conservation tests, to verify that the physical behavior matched analytical expectations before analyzing the final science results.

To accelerate performance, I used Numba to jit-compile the photon propagation and run each photon in parallel. Even at $10^7$ (100 million) packets per band, the code remained fast and stable.

# Installation instructions:

ipython

packages: numpy, matplotlib, scipy, numba

# Deviations from source code

transport & run_mcrt:

Since each packet does not influence another, I chose to implement a parallel MCRT instead of looping over each packet. So, instead of one run_mcrt function, I have run_packets_parallel, run_band_parallel, and run_mcrt_jit. 

Photon class:

Since Python classes are so computationally heavy and do not work with jit, I do not use the photon class at all. I left the base structure and moved the sub-functions to my old_code file to clean up the module. The photons are simply stored within lists or numpy arrays. 

EscapeTracker & results:

Since I am using jit, I could not use the escape tracker within my parallel processing. I save the results in lists or arrays and then save them to the escape tracker or results dictionary after. I also made escape tracker a dataclass instead, as that made it a lot easier to pull and assign values when running tests and plotting. 

BANDS:

Instead of creating a class just to hold the band min/max values, I put them into a dictionary and put that inside my constants module, since those are constants and needed to be referenced from most modules. 

## Plots and Graphs:

All plots and graphs are found in outputs/figures. Within each N_packet folder, the following plots can be found: an opacity validation plot, a spectral energy distribution (SED) plot, an absorption map per band, and a compiled RGB image. 

In the final_plots folder, there is a figure with subplots, displaying the absorption maps and RGB composites for each N_packet. There is also a convergence plot and a time test plot. 

# Usage examples:

ex 1.
ipython project3_analysis.py

Total run time: ~ 7 mins
This will run the main project analysis. Currently, it is set up to run 5 ZAMS stars over B, V, K bands in a 1pc box with $128^3$ resolution, over $10^4$ - $10^7$ packets. Analysis and time tests are also set to run with the main analysis. 

ex 2.
project3_analysis.run_simulation(stars, grid, bands, n_packets)

Total run time: ~ 1 min for $10^7$ packets
This runs only a single simulation. Will output a dictionary of results. 

# Key results summary:

All validation tests passed successfully. The empty-box test produced a 100% escape fraction, the opaque-box test trapped all packets, and the uniform-sphere result matched the analytical $\exp{-\tau}$ prediction within $3\sigma$ uncertainty. Energy conservation was consistently better than the 0.1% tolerance, and the escape fraction convergence followed the expected $N^{-1/2}$ trend. 

The physical results all behaved as expected. The higher the wavelength, the lower the escape fraction, which aligns with the observed reddening. The absorption maps also show that the most light is absorbed near the host star. The RGB composite images show very well how the lower mass stars peak in the red, rather than in the blue, like the more massive stars. 

Performance testing shows how the jit parallel implementation was very effective at running an efficient MCRT simulation over N packets. The parallel implementation was verified with a significantly slower code, consisting of heavy Python classes and O(N) time complexity. With only a $16^3$ grid and 1000 packets, that code was taking minutes (saved in tests/old_code.py). 

# Acknowledgment of collaborators:

Throughout the project, I collaborated with my classmates to help conceptualize things and bounce ideas around. In terms of sharing pseudo-code, I worked with Caden at the start of the project to figure out the distance-to-next-cell and propagate photon implementation. Otherwise, we all share offices and classes, so there was a constant stream of abstract communication and collaboration. We helped each other with conceptualization, debugging, formatting plots, understanding expected outcomes, etc. 

For packages, I used NumPy, Matplotlib, SciPy, and Numba. 

Stellar parameters were based on the Tout et al. (1996) ZAMS relations, and spectral classifications were found with the Pecaut & Mamajek (2013) table. For reference data, I used the Draine (2003) 5.5A dust model.

In terms of internet and AI usage, I used ChatGPT to help debug and format plots, and I used Grammarly to spell-check my written reports. For assistance with jit, I referenced the jit documentation, various online examples, and bounced ideas and issues off of notebook lm and ChatGPT when I got stuck.
