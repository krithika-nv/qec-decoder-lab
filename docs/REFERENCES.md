# References

- Oscar Higgott and Craig Gidney, **Sparse Blossom: correcting a million errors per core second with minimum-weight matching**, *Quantum* 9, 1600 (2025). DOI: [10.22331/q-2025-01-20-1600](https://doi.org/10.22331/q-2025-01-20-1600). PyMatching v2 reference and benchmark context.
- [PyMatching repository](https://github.com/oscarhiggott/PyMatching), including its [surface-code timing benchmark](https://github.com/oscarhiggott/PyMatching/tree/master/benchmarks/surface_codes) and numerical data. Author timing data are distinct from locally measured values.
- [Stim repository](https://github.com/quantumlib/Stim), used for stabilizer-circuit sampling and detector error models.
- Michael A. Nielsen and Isaac L. Chuang, *Quantum Computation and Quantum Information*, Cambridge University Press. General background.
- Daniel Gottesman, **An Introduction to Quantum Error Correction and Fault-Tolerant Quantum Computation**, arXiv:0904.2557. General QEC background.

Citation of a paper is not evidence that this project reproduces its full reported figure. Reproduction scope and deviations are given in [Results and limitations](RESULTS_AND_LIMITATIONS.md).

## Further performance-engineering context

- Daniel Báscones, Francisco Garcia-Herrero, Javier Valls, **Exploring the FPGA and ASIC design space of belief propagation and ordered statistics decoders for quantum error correction codes**, *EPJ Quantum Technology* 12, 140 (2025): https://doi.org/10.1140/epjqt/s40507-025-00446-y . Related literature, **not a reproduced experiment** in this project.
- **Localized statistics decoding for quantum low-density parity-check codes**, *Nature Communications* (2025): https://www.nature.com/articles/s41467-025-63214-7 . This discusses high costs of OSD-style matrix elimination; it is background for future optimization, not evidence of this project's performance.
- PyMatching documented `decode_batch` interface, including optional returned matching weights: https://github.com/oscarhiggott/PyMatching/blob/master/src/pymatching/matching.py . This repository does not presently use matching weights as a calibrated gating confidence measure.
