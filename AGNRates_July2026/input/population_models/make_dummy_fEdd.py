import numpy as np

# Grids
z_grid = np.linspace(0.0, 10.5, 150)
M_grid = np.arange(5.0, 9.1, 0.1)   # log10(M/Msun)

# Dummy discrete fEdd support
fEdd_grid = np.array([0.01])

# Full conditional distribution p(fEdd | M, z)
# shape = (Nz, NM, Nf)
pfEdd_given_Mz = np.ones((len(z_grid), len(M_grid), len(fEdd_grid)), dtype=float)

np.savez(
    "input/population_models/nAGN_models/pfEdd_given_Mz_dummy.npz",
    scenario="delta_fEdd_0.01",
    z_grid=z_grid,
    M_grid=M_grid,
    fEdd_grid=fEdd_grid,
    pfEdd_given_Mz=pfEdd_given_Mz,
)