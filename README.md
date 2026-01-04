# fortunatebench:  A Series of (Un)Fortunate Plasma and Fusion Benchmarks

The goal of this repo is to collect a set of benchmark problems for
plasma physics, in particular, for fusion problems. Our goal is to
provide guidance for constructing new numerical methods and Scientific
Machine Learning (SciML) *foundation-models* for all of plasma physics
as used in fusion. Plasma physics, unlike fluid mechanics where one
relies on the Navier-Stokes equations, has a bewildering suite of
widely-used approximations. A proper foundation-model must incorporate
all or a good fraction of such approximations to be a useful tool in
design of fusion power plants (FPP).

We aim to provide input files for each of the benchmark problems so
others can generate the data


There are two objectives:

1. To list the equations for major models used in plasma physics,
   along with some common extensions, conserved quantities and set of
   benchmark problems for each model

2. For each model, provide a list of the codes that solve those
   equations, build instructions for the codes, and provide input
   files for each selected benchmark.
