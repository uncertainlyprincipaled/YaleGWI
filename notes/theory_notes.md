# Theory Notes for SpecProj-UNet

## Microlocal sheaf theory & D-modules

The quotient category QP = HolΛ(X)/Nnull where:
- Objects: holonomic D-modules solving P
- Nnull: modules whose solution sheaf is invisible to surface sensors
- Verdier quotient ensures well-posed Cauchy problem modulo null space

The idempotents Π± are exact in QP; their Schwartz kernels are the FFT masks in proj_mask.py.

Embedding masks as fixed/learnable conv filters gives a representation-theoretic interpretation of channel splits in CNNs.

## Joint Forward-Inverse Paradigm

The joint forward-inverse paradigm implemented in this codebase combines two complementary approaches to subsurface imaging:

1. **Category-Quotient View**: The physics-guided spectral projector (Π±) decomposes the wavefield into up-going and down-going components, effectively creating a quotient space that respects the underlying wave physics. This approach leverages domain knowledge to ensure physically meaningful decompositions.

2. **Latent-Manifold Assumption**: The IU-Net translator operates on the assumption that velocity models and wavefields lie on a shared latent manifold, allowing for bidirectional translation between these spaces. This is inspired by the ICLR25FWI paper's discussion in §3 about the existence of a common latent space for forward and inverse problems.

The combination of these approaches offers several advantages:
- The physics-guided splitter ensures physically meaningful decompositions
- The latent translator enables efficient exploration of the solution space
- The joint training objective helps regularize both forward and inverse problems

This implementation follows the mathematical formulation presented in the ICLR25FWI paper, particularly in §1 for the spectral projector and §3 for the latent space translation.

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