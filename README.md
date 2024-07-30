<p align="center">
  <img src="https://github.com/user-attachments/assets/68139de8-38d5-468a-a255-dd90d81e1ad4" width="50%">
</p>

# SSP CUB 2024 Orbital Determination Code
## Samuel Yuan

Super cool orbital determination (OD) code and report stuff are available here. OD code is in the `od` folder and the main od file is titled `od_lib_yuan.py`. Experiments on asteroid 699 Hela (aka A910 LC) are available in a Jupyter Notebook. The final report is titled `ssp_team08_final_report-1.pdf` and is in the main directory. Below is a schematic of the OD process:

<p align="center">
  <img src="https://github.com/user-attachments/assets/3794eadc-27d1-463f-ab6d-e3270e1d7ce0" width="90%">
</p>

Orbital elements of Mars-crossing asteroid 699 Hela were determined with Gauss' method. To do this, we observed 699 Hela over the course of $2.5$ weeks and determined three best observations from which we computed its orbital elements. The determined orbital elements of asteroid 699 Hela at Julian Date $2460500.68896$ were $a = 2.61 \pm 0.12 \mathrm{AU}$, $e = 0.410 \pm 0.027$, $i = 15.29 \pm 0.19^{\circ}$, $\Omega = 242.50 \pm 0.12^{\circ}$, $\omega = 91.4 \pm 2.2^{\circ}$, and $M = 321.1 \pm 4.1^{\circ}$ (uncertainties computed with Monte Carlo simulations). Empirically, we obtain low $\sim1.079\%$ error in right ascension and $\sim0.001\%$ error in declination on ephemeris generation consistency checks (comparing the predicted RA and DEC from these calculated orbital elements with the actual RA and DEC at the third observation time). 

Below is a plot of 11 simulated orbits of 699 Hela at Julian Date $2460500.68896$ (1 with exact calculated orbital elements and 10 clones with elements accounting for uncertainty—to create our clones, we sample new orbital elements from the calculated ones following $x' \sim \mathcal{N}(x, \delta x), x \in {a, e, i, \Omega, \omega, M}$):

<p align="center">
  <img src="https://github.com/user-attachments/assets/abff0395-aea0-4115-bf98-0ca09a395a8b" width="50%">
</p>

Our observations and simulations refine orbital element predictions for 699 Hela and further confirm that it is most likely on a stable orbit and will not hit earth :)! Huge thanks to all SSP CUB faculty, TAs, and peers!
