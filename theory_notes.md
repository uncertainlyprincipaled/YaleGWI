# Theory Notes for SpecProj-UNet

## Microlocal sheaf theory & D-modules

The quotient category QP = HolΛ(X)/Nnull where:
- Objects: holonomic D-modules solving P
- Nnull: modules whose solution sheaf is invisible to surface sensors
- Verdier quotient ensures well-posed Cauchy problem modulo null space

The idempotents Π± are exact in QP; their Schwartz kernels are the FFT masks in proj_mask.py.

Embedding masks as fixed/learnable conv filters gives a representation-theoretic interpretation of channel splits in CNNs.

## References

1. OpenFWI dataset & baseline code:
   - H. Wang et al. "OpenFWI: Large-Scale Multi-Structural Benchmark Datasets for Full-Waveform Inversion." NeurIPS D–B 2022
   - https://openfwi.github.io

2. Spectral factorisation / minimum phase in seismology:
   - S. Rickett, PhD Thesis, Stanford 2001

3. Microlocal theory:
   - M. Kashiwara & P. Schapira, Sheaves on Manifolds, Springer 1990 — see Ch. 7 (microlocal)

4. FFT implementation:
   - L. Trefethen, Spectral Methods in MATLAB, SIAM 2000, §2 FFT grids

5. Waveform inversion background:
   - J. Virieux & S. Operto, "Overview of Full-Waveform Inversion." Geophysics 77, 3 (2012) 