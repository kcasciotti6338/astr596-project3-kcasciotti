
Pecaut & Mamajek (2013) txt file for spectral classification

# Project Description: 

This project was to simulate the radiative transfer of starlight through a dusty molecular cloud using the Monte Carlo method. The goal was to understand how dust extinction and absorption affect the observed emission from an embedded star cluster. Each photon packet emitted by a star either escapes the box or is absorbed by dust, following the discrete absorption method.

I modeled a cluster of five zero-age main sequence (ZAMS) stars, each with individually computed Planck-mean opacities in the B, V, and K bands based on the Draine (2003) dust model. The code tracks photon packets as they propagate through a one-parsec cube, interacting with dust according to its density and wavelength-dependent opacity. I also implemented a full validation suite, including empty box, opaque box, uniform sphere, and energy-conservation tests, to verify that the physical behavior matched analytical expectations before analyzing the final science results.

To accelerate performance, I used Numba to jit-compile the photon propogation and run each photon in parallel. Even at 10e7 (100 million) packets per band, the code remained fast and stable.

# Installation instructions:

ipython

packages: numpy, matplotlib, scipy, numba

## Plots and Graphs:

All plots and graphs are found in outputs/figures. Within each N_packets folder, there are plots, such as the opacity validation plot, showing the Draine opacity curve with shaded regions for the B, V, K bands. The spectral energy distribution (SED) plot


# Usage examples:

body_2(Tt=10, dt=0.01, snaps=200, gif=True)

This runs the Earth-Sun system for 10 years, with a stepsize of 0.01 yrs, and saves 200 evenly spaced snapshots throughout the total time. It goes through each of the four integration methods, plotting euler's on its own standard diagnostic figure and the other three on the same diagnostic figure for meaningful comparisons. Then, a gif of the orbit over one year will be saved as well, for each integration method. 

Each N-body demonstration has a similar style of analysis. 

# Key results summary:

For the integrator behavior, each method acted generally as expected. Euler's was never accurate enough to make even half and orbit before drifting into an uncontrolled spiral. RK2 was better but showed drift even at small scales. RK4 was much better, it showed accurate orbits with a very slow drift that was unnoticeable until the number of bodies and number of steps got larger. Leapfrog, as a symplectic model, did the best. It was very stable in terms of energy, with some drift when I started to push into much higher body and step counts. However, this was mostly due to other factors, such as the ratios between step size, softening, the number of bodies, and radius. 

# Acknowledgment of collaborators:

Throughout the project, I collaborated with my classmates to help conceptualize things and bounce ideas around. We talked the most about what to vectorize and what to leave alone, how the Plummer sphere works with re-centering, and ways to improve plots. We all share offices and classes, so there was a constant stream of communication and collaboration, so I couldn't name anyone or anything too specific. We never shared physical code, but did share sections of pseudo code and wrote out equations together. 

I also went to office hours, where a parenthesis bug was found in my force calculations, and got advice on improving my plots. 

I used NumPy, Matplotlib, Imageio, and Pillow.

In terms of internet and AI usage, I used ChatGPT to help debug, I used a lot of matplotlib examples to build my plotting functions, and I used Grammarly to spell check my written reports.
