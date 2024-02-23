CC ?= gcc

CFLAGS = -march=native -g3

ifeq ($(CC),gcc)
	OFLAGS = -O2 -fopenmp -fopt-info-all=nbody.gcc.optrpt
else ifeq ($(CC),clang)
	OFLAGS = -O2 -fopenmp -Rpass-analysis=loop-vectorize -Rpass=loop-vectorize -mllvm -force-vector-width=8
else
	OFLAGS = -O2 -fopenmp
endif

VERSIONS = SOA ALLIGNED PARALLELIZED VECTORISED LESS_DIVISIONS FISR UNROLLED ALL_EXCEPT_UNROLL ALL INITIAL_CODE

EXECUTABLES = $(addprefix nbody3D_, $(VERSIONS))

all: clean $(EXECUTABLES)

nbody3D_%: nbody.c
	$(CC) $(CFLAGS) $(OFLAGS) -D$* $< -o $@ -lm

clean:
	rm -Rf *~ $(EXECUTABLES) *.optrpt
