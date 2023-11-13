torchSCF
========

Simple self-consistent field (SCF) implementation for PyTorch. It uses McMurchie-Davidson (MD) expansion for the two-electron integrals and the iterative Hartree-Fock (HF) method for the SCF procedure. 

Up to now, only restricted HF (RHF) is implemented, with cartesian Gaussian basis functions till p-type orbitals. Due to dengenracy, d-type and above will be added later. Basis function currently only support STO-3G for Si.

It uses Enzyme to generate forward and backward function, which will be bound to Python methods using Pybind11.

Based on: 
1. from Joshua Goings' https://github.com/jjgoings/McMurchie-Davidson, and https://joshuagoings.com/2017/04/28/integrals/
2. and myself pyhf
