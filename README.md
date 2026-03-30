# Haldane Model Study

A numerical study of the transport and topological properties of the Haldane model on a honeycomb lattice.

## Overview

This project investigates the Haldane model, a paradigm for topological insulators that demonstrates the Quantum Hall Effect in the absence of an external magnetic field. The study focuses on:

1.  **Transport Properties**: Calculating longitudinal semiclassical conductivity using the relaxation time approximation and comparing it with the Drude formula.
2.  **Topological Properties**: Verification of the Anomalous Hall Effect (AHE) through Berry curvature integration and Chern number calculation.

## Project Structure

-   `data/`: Contains pre-calculated eigenvalues and other simulation data.
-   `figures/`: Generated plots of band structures, Berry curvature, etc.
-   `report/`: LaTeX source files and the final PDF report.
-   `scripts/`: Python scripts for running simulations, data analysis, and plotting.
-   `tools/`: Core Python modules implementing the Haldane system Hamiltonian and utility functions.

## Dependencies

The project requires the following Python libraries:

-   `numpy`
-   `matplotlib`
-   `scipy` (likely used for integration and other numerical tasks)

## Usage

### Running Simulations

To calculate the band structure or other properties, use the scripts in the `scripts/` directory. For example:

```bash
python scripts/get_bands_kpath.py
```

### Plotting Results

To visualize the results, run the corresponding plotting script:

```bash
python scripts/plot_bands_kpath.py
```

### Analysis

Additional analysis such as Berry curvature or AHE can be performed with:

```bash
python scripts/get_berry_curvature.py
python scripts/get_ahe.py
```

## Report

The detailed results and theoretical background are available in the `report/` directory. You can find the compiled `main.pdf` or recompile it from `main.tex`.

---
**Author**: Víctor Martín Parra
