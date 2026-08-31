# Source code and data for
 
 Beyers, C., Villas Bôas, A. B., Marechal, G., Mazloff, M. R., Sun, R., and Torres, H. S. Spatial Variability of the Stokes Drift Across Scales: The Relative Roles of Wind Variability and Ocean Currents. Submitted to Journal of Geophysical Research, Oceans.

 # Abstract
The Stokes drift plays an important role in upper-ocean transport and mixing. However, its spatial variability is often represented using wave products forced by relatively smooth winds and without current effects on waves (CEW). As a result, variability in the Stokes drift at scales shorter than $\sim200~\mathrm{km}$ may be underestimated. Here, we use a suite of year-long WAVEWATCH III simulations in the California Current System region to quantify the relative roles that winds and currents play in driving the spatial variability of the Stokes drift across scales $\mathcal{O}$(1,000--10~km). Simulations are forced with either spatially filtered low-variability winds or high-variability winds, with and without surface-current forcing. Additional experiments disable individual CEW mechanisms to assess the sensitivity of the Stokes drift to refraction, changes in wavenumber, current advection, and the relative wind effect. We show that sub-synoptic wind variability $\mathcal{O}$(100--10~km) and CEW both contribute substantially to the spatial variability of the Stokes drift at scales shorter than $\sim200~\mathrm{km}$, but through distinct pathways. Sub-synoptic winds generate Stokes drift variability that remains locally coherent with the wind field, while CEW introduce local and non-local variability that reflects the underlying dynamics of the currents. The current-driven Stokes drift response is strongly associated with the divergent component of the currents and is dominated by changes in wavenumber. Our results demonstrate that the Stokes drift should be viewed not simply as a smooth wind-driven field but as a spatially structured field shaped by both atmospheric and oceanic variability, with potential consequences for upper-ocean transport and mixing.


# Authors

# Data

# Funding

# How to use this repository
All figures in the manuscript can be reproduced using the Python scripts from this repository and the data. To do so, follow these steps

1. Make a local copy of this repository by either cloning or downloading it.
Your directory tree should look like this:

```
stokes_drift_variability/
├── data/
├── figures/
├── notebooks/
├── src/
└── pixi.toml
```

2. If you would like to reproduce the figures from the manuscript using this repository, we recommend that you install [Pixi](https://pixi.sh/latest/). You can then run:
```
$ pixi shell
```
from the project root. This will build the dependencies and activate the Python environment for this project. You can then launch Jupyter Lab as you would normally:

```
$ jupyter lab
```

If you follow the steps above, you should be able to reproduce all figures by running the notebooks from the notebooks directory without having to adjust any paths.

# How to cite this code
DOI coming soon.
