# Optimisation d'une n-body simulation en C
 1,6 à 41,5 GFLOP/s :déroulages, parallélisation, vectorisation, AoS vs SoA, alignements mémoires, moins d'instructions coûteuses... en C sur CPU Ryzen 5600G 16GB RAM RTX 3080 desktop


À l'attention de M. Bolloré et de M.Ibnamar, je vous ai envoyé mon rapport par e-mail, 

Bien à vous Rayane Hammoumi IATIC4

# Intro

-------

This code performs a simple nbody 3D simulation. 

Overall, the code is optimized for performance and can be compiled with various options to customize the level of optimization.

NB: The initial positions and speed of the particles are the same random numbers each run.
The final positions and speed of the first 3 particles are always printed so you can compare the results of different optimisations (they all output the same result)


# Compilation

-------------

"make" will compile the code and create 10 executables. Each one executes the initial code with 1 or several optimisations added.

Optional: to change compiler, add "CC=compiler_name" to the make command. GCC is used by default and performs best with my code.


# Running

---------

"export OMP_DISPLAY_ENV=TRUE" then "export OMP_NUM_THREADS=NB_THREADS" then "./nbody3D_VERSION" or "./nbody3D_VERSION NB_PARTICLES"


# Output

--------

- nbody3D_INITIAL_CODE: initial code

- nbody3D_SOA: SOA instead of AOS

- nbody3D_ALLIGNED: SOA + memory allignment

- nbody3D_PARALLELIZED: OPENMP parallelization instructions

- nbody3D_VECTORISED: OPENMP vectorisation instruction for the inner for loop in move_particle

- nbody3D_LESS_DIVISIONS: less divisions

- nbody3D_FISR: less divisions + Fast Inversed Square Root algorithm instead of 1/sqrtf()

- nbody3D_UNROLLED for manually unrolled loops 

- nbody3D_ALL_EXCEPT_UNROLL: all optimisations except unrolling (best performance)

- nbody3D_ALL: all optimisations added