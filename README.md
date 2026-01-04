# fortunatebench:  A Series of (Un)Fortunate Plasma and Fusion Benchmarks

The goal of this repo is to collect a set of benchmark problems for
plasma physics, in particular, for fusion problems, with some initial
work on plasma astrophysics. Our goal is to provide guidance for
constructing new numerical methods and Scientific Machine Learning
(SciML) *foundation-models* (FM) for all of plasma physics as used in
fusion. Plasma physics, unlike fluid mechanics where one relies on the
Navier-Stokes equations, has a bewildering suite of widely-used
approximations. A proper foundation-model must incorporate all or a
good fraction of such approximations to be a useful tool to study and
design fusion power plants (FPP).

We aim to provide input files for each of the benchmark problems so
others can generate the data for their own models, or compare their
numerical methods on the benchmarks we provide.

There are two objectives:

1. To list the equations for major models used in plasma physics,
   along with some common extensions, conserved quantities, and a
   broad set of benchmark problems.

2. For each model, provide a list of the open-source codes that solve
   those equations, build instructions for the codes, and provide
   input files for each selected benchmark.

3. Provide access to our BEACONS-FM (Bounded Extrapolatory Composable
   Neural Surrogate - Foundation Model) code so others can use it for
   their research.
   
   
## Contributing to fortunatebench

This is an ambitious undertaking and we welcome contributions from
others. You can contribute in several ways:

- Add new model equations (and references) to the main TeX document,
  along with conserved quantities, benchmarks and links to code and
  input files for those benchmarks.
  
- Your own surrogates or Foundation Models for all or part of the
  models listed in this repo.
  
- Corrections and extension to existing text or code.

Please make a branch or fork the repo and submit a PR. We will examine
the PR, give you feedback and merge when ready. Our goal is to be err
on the side of inclusion, so please do not hesistate to contribute!

