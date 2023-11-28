import numpy as np
import torch
import scf
from integrals import scf_step_simplified

np.random.seed(42)

n_atoms = 4
species = np.array([1., 1., 1., 1.]) * 14.0
positions = np.random.rand(n_atoms, 3)
C = np.random.rand(n_atoms * 9)

# test energy
# print(scf.scf_step(n_atoms, C, species, positions))

# torch test energy

with torch.inference_mode():
    C = torch.tensor(C)
    species = torch.tensor(species)
    positions = torch.tensor(positions)
    e = scf_step_simplified(C, positions)
    print(e)