# flowgravity
This reposity contains the code to test FLOW-CM

# FLOW-CM

FLOW-CM is a phenomenological nodal-conductance law for galactic rotation curves.

The model constructs a deterministic baryon-derived network response from the observed gas, disk, and bulge contributions. It introduces nodal occupation, structural collectivity, inward radial memory, graph current, and a harmonic conductance between two geometric channels.

## Repository contents

- `code/`: Python scripts used to compute the FLOW-CM harmonic model and generate publication assets.
- `manuscript/`: English and Spanish versions of the manuscript.
- `figures/`: publication figures.
- `tables/`: final result tables used in the manuscript.

## Data

This project uses the SPARC/Rotmod galaxy rotation-curve dataset.

The original SPARC/Rotmod data are not redistributed in this repository. By default, the main script checks for `.dat` files in:

```text
data/sparc/
