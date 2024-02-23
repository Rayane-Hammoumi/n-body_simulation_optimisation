#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

typedef float f32;
typedef double f64;
typedef unsigned long long u64;

typedef struct particle_s
{
    f32 x, y, z;
    f32 vx, vy, vz;
} particle_t;

#if defined(SOA) || defined(ALL) || defined(ALLIGNED) || defined(ALL_EXCEPT_UNROLL)

typedef struct particle_s_SOA
{
    f32 *x, *y, *z;
    f32 *vx, *vy, *vz;

} particle_t_SOA;

void init(particle_t_SOA *p, u64 n)
{
    for (u64 i = 0; i < n; i++)
    {
        u64 r1 = (u64)rand();
        u64 r2 = (u64)rand();
        f32 sign = (r1 > r2) ? 1 : -1;

        p->x[i] = sign * (f32)rand() / (f32)RAND_MAX;
        p->y[i] = (f32)rand() / (f32)RAND_MAX;
        p->z[i] = sign * (f32)rand() / (f32)RAND_MAX;

        p->vx[i] = (f32)rand() / (f32)RAND_MAX;
        p->vy[i] = sign * (f32)rand() / (f32)RAND_MAX;
        p->vz[i] = (f32)rand() / (f32)RAND_MAX;
    }
}

#else
void init(particle_t *p, u64 n)
{
    for (u64 i = 0; i < n; i++)
    {
        u64 r1 = (u64)rand();
        u64 r2 = (u64)rand();
        f32 sign = (r1 > r2) ? 1 : -1;

        p[i].x = sign * (f32)rand() / (f32)RAND_MAX;
        p[i].y = (f32)rand() / (f32)RAND_MAX;
        p[i].z = sign * (f32)rand() / (f32)RAND_MAX;

        p[i].vx = (f32)rand() / (f32)RAND_MAX;
        p[i].vy = sign * (f32)rand() / (f32)RAND_MAX;
        p[i].vz = (f32)rand() / (f32)RAND_MAX;
    }
}
#endif

#if defined(FISR) || defined(ALL) || defined(ALL_EXCEPT_UNROLL)
//"fast inverse square root" algorithm (approximation de la racine carrée inverse)
// fonction trouvée sur https://stackoverflow.com/questions/12923657/is-there-a-fast-c-or-c-standard-library-function-for-double-precision-inverse
f32 invSqrt(f32 number)
{
    union
    {
        f32 f;
        int i;
    } conv;

    f32 x2;
    const f32 threehalfs = 1.5F;

    x2 = number * 0.5F;
    conv.f = number;
    conv.i = 0x5f3759df - (conv.i >> 1);
    conv.f = conv.f * (threehalfs - (x2 * conv.f * conv.f));
    return conv.f;
}
#endif

#if defined(SOA) || defined(ALLIGNED) || defined(ALL) || defined(ALL_EXCEPT_UNROLL)
void move_particles(particle_t_SOA *p, const f32 dt, u64 n)
#else
void move_particles(particle_t *p, const f32 dt, u64 n)
#endif
{
    const f32 softening = 1e-20;
    u64 i, j;
#if defined(PARALLELIZED) || defined(ALL) || defined(ALL_EXCEPT_UNROLL)
#pragma omp parallel for schedule(static, (n / omp_get_num_threads()))
#endif
    for (i = 0; i < n; i++)
    {
        f32 fx = 0.0;
        f32 fy = 0.0;
        f32 fz = 0.0;

#if !defined(UNROLLED) && !defined(ALL)
//  23 floating-point operations
#if defined(VECTORISED) || defined(ALL) || defined(ALL_EXCEPT_UNROLL)
#pragma omp simd simdlen(8)
#endif
        for (j = 0; j < n; j++)
        {
// Newton's law
#if defined(SOA) || defined(ALLIGNED) || defined(ALL_EXCEPT_UNROLL)
            const f32 dx = p->x[j] - p->x[i]; // 1 (sub)
            const f32 dy = p->y[j] - p->y[i]; // 2 (sub)
            const f32 dz = p->z[j] - p->z[i]; // 3 (sub)
#else
            const f32 dx = p[j].x - p[i].x; // 1 (sub)
            const f32 dy = p[j].y - p[i].y; // 2 (sub)
            const f32 dz = p[j].z - p[i].z; // 3 (sub)
#endif
            const f32 d_2 = (dx * dx) + (dy * dy) + (dz * dz) + softening; // 9 (mul, add)

#if defined(FISR) || defined(ALL_EXCEPT_UNROLL)
            const f32 inverse_d_3_over_2 = invSqrt(d_2) * (1 / d_2); // x^3/2 <==> x^(1+1/2)
                                                                     //<=> x*x^1/2 <=> x*racine(x)
                                                                     // Au lieu d'effectuer plusieurs divisions par la suite,
                                                                     // on fait l'inverse de x*racine(x) puis on fait des multiplications
            // Net force
            fx += dx * inverse_d_3_over_2;
            fy += dy * inverse_d_3_over_2;
            fz += dz * inverse_d_3_over_2;
#elif defined(LESS_DIVISIONS)
            const f32 d_3_over_2 = 1 / (d_2 * sqrtf(d_2));

            // Net force
            fx += dx * d_3_over_2; // 13 (add, div)
            fy += dy * d_3_over_2; // 15 (add, div)
            fz += dz * d_3_over_2; // 17 (add, div)

#else
            const f32 d_3_over_2 = pow(d_2, 3.0 / 2.0); // 11 (pow, div)

            // Net force
            fx += dx / d_3_over_2; // 13 (add, div)
            fy += dy / d_3_over_2; // 15 (add, div)
            fz += dz / d_3_over_2; // 17 (add, div)
#endif
        }
#elif defined(UNROLLED)
        //  loop unrolling (4 fois moins d'itérations, chacune avec 4 fois plus d'instructions)
        //  23 floating-point operations

        for (j = 0; j < n; j += 4)
        {
            const f32 dx = p[j].x - p[i].x;
            const f32 dy = p[j].y - p[i].y;
            const f32 dz = p[j].z - p[i].z;
            const f32 d_2 = (dx * dx) + (dy * dy) + (dz * dz) + softening;
            const f32 d_3_over_2 = pow(d_2, 3.0 / 2.0);

            // Net force
            fx += dx / d_3_over_2;
            fy += dy / d_3_over_2;
            fz += dz / d_3_over_2;

            const f32 dx1 = p[j + 1].x - p[i].x;
            const f32 dy1 = p[j + 1].y - p[i].y;
            const f32 dz1 = p[j + 1].z - p[i].z;
            const f32 d_21 = (dx1 * dx1) + (dy1 * dy1) + (dz1 * dz1) + softening;
            const f32 d_3_over_21 = pow(d_21, 3.0 / 2.0);

            fx += dx1 / d_3_over_21;
            fy += dy1 / d_3_over_21;
            fz += dz1 / d_3_over_21;

            const f32 dx2 = p[j + 2].x - p[i].x;
            const f32 dy2 = p[j + 2].y - p[i].y;
            const f32 dz2 = p[j + 2].z - p[i].z;
            const f32 d_22 = (dx2 * dx2) + (dy2 * dy2) + (dz2 * dz2) + softening;
            const f32 d_3_over_22 = pow(d_22, 3.0 / 2.0);

            fx += dx2 / d_3_over_22;
            fy += dy2 / d_3_over_22;
            fz += dz2 / d_3_over_22;

            const f32 dx3 = p[j + 3].x - p[i].x;
            const f32 dy3 = p[j + 3].y - p[i].y;
            const f32 dz3 = p[j + 3].z - p[i].z;
            const f32 d_23 = (dx3 * dx3) + (dy3 * dy3) + (dz3 * dz3) + softening;
            const f32 d_3_over_23 = pow(d_23, 3.0 / 2.0);

            fx += dx3 / d_3_over_23;
            fy += dy3 / d_3_over_23;
            fz += dz3 / d_3_over_23;
        }
        j -= 4;

        if (j != n - 1)
        {
            // Traite les derniers éléments restants si n % 4 != 0
            for (; j < n; j++)
            {
                const f32 dx = p[j].x - p[i].x;
                const f32 dy = p[j].y - p[i].y;
                const f32 dz = p[j].z - p[i].z;
                const f32 d_2 = (dx * dx) + (dy * dy) + (dz * dz) + softening;
                const f32 d_3_over_2 = pow(d_2, 3.0 / 2.0);
                fx += dx / d_3_over_2; // 13 (add, div)
                fy += dy / d_3_over_2; // 15 (add, div)
                fz += dz / d_3_over_2; // 17 (add, div)
            }
        }

#else
        // MOST OPTIMISED
        //  loop unrolling (4 fois moins d'itérations, chacune avec 4 fois plus d'instructions)
        //  23 floating-point operations

        for (j = 0; j < n; j += 4)
        {
            const f32 dx = p->x[j] - p->x[i]; // 1 (sub)
            const f32 dy = p->y[j] - p->y[i]; // 2 (sub)
            const f32 dz = p->z[j] - p->z[i]; // 3 (sub)

            const f32 d_2 = (dx * dx) + (dy * dy) + (dz * dz) + softening;
            const f32 inverse_d_3_over_2 = invSqrt(d_2) * (1 / d_2);
            // x^3/2 <=> x^(1+1/2) <=> x*x^1/2 <=> x*racine(x)
            // Au lieu d'effectuer plusieurs divisions par la suite,
            // on fait l'inverse de x*racine(x) puis on fait des multiplications

            // Net force
            fx += dx * inverse_d_3_over_2;
            fy += dy * inverse_d_3_over_2;
            fz += dz * inverse_d_3_over_2;

            const f32 dx1 = p->x[j + 1] - p->x[i];
            const f32 dy1 = p->y[j + 1] - p->y[i];
            const f32 dz1 = p->z[j + 1] - p->z[i];

            const f32 d_21 = (dx1 * dx1) + (dy1 * dy1) + (dz1 * dz1) + softening;
            const f32 inverse_d_3_over_21 = invSqrt(d_21) * (1 / d_21);

            // Net force
            fx += dx1 * inverse_d_3_over_21;
            fy += dy1 * inverse_d_3_over_21;
            fz += dz1 * inverse_d_3_over_21;

            const f32 dx2 = p->x[j + 2] - p->x[i];
            const f32 dy2 = p->y[j + 2] - p->y[i];
            const f32 dz2 = p->z[j + 2] - p->z[i];

            const f32 d_22 = (dx2 * dx2) + (dy2 * dy2) + (dz2 * dz2) + softening;
            const f32 inverse_d_3_over_22 = invSqrt(d_22) * (1 / d_22);

            // Net force
            fx += dx2 * inverse_d_3_over_22;
            fy += dy2 * inverse_d_3_over_22;
            fz += dz2 * inverse_d_3_over_22;

            const f32 dx3 = p->x[j + 3] - p->x[i];
            const f32 dy3 = p->y[j + 3] - p->y[i];
            const f32 dz3 = p->z[j + 3] - p->z[i];

            const f32 d_23 = (dx3 * dx3) + (dy3 * dy3) + (dz3 * dz3) + softening;
            const f32 inverse_d_3_over_23 = invSqrt(d_23) * (1 / d_23);

            // Net force
            fx += dx3 * inverse_d_3_over_23;
            fy += dy3 * inverse_d_3_over_23;
            fz += dz3 * inverse_d_3_over_23;
        }
        j -= 4;

        if (j != n - 1)
        {
            // Traite les derniers éléments restants si n % 4 != 0
            for (; j < n; j++)
            {
                const f32 dx = p->x[j] - p->x[i];
                const f32 dy = p->y[j] - p->y[i];
                const f32 dz = p->z[j] - p->z[i];
                const f32 d_2 = (dx * dx) + (dy * dy) + (dz * dz) + softening;
                const f32 d_3_over_2 = pow(d_2, 3.0 / 2.0);
                fx += dx / d_3_over_2; // 13 (add, div)
                fy += dy / d_3_over_2; // 15 (add, div)
                fz += dz / d_3_over_2; // 17 (add, div)
            }
        }
#endif

#if defined(SOA) || defined(ALLIGNED) || defined(ALL) || defined(ALL_EXCEPT_UNROLL)
        p->vx[i] += dt * fx; // 19 (mul, add)
        p->vy[i] += dt * fy; // 21 (mul, add)
        p->vz[i] += dt * fz; // 23 (mul, add)
#else
        p[i].vx += dt * fx; // 19 (mul, add)
        p[i].vy += dt * fy; // 21 (mul, add)
        p[i].vz += dt * fz; // 23 (mul, add)
#endif
    }

#if defined(PARALLELIZED) || defined(ALL) || defined(ALL_EXCEPT_UNROLL)
#pragma omp parallel for schedule(static, (n / omp_get_num_threads()))
#endif

#if !defined(UNROLLED) && !defined(ALL)

// 3 floating-point operations
#if defined(SOA) || defined(ALLIGNED) || defined(ALL_EXCEPT_UNROLL)
    for (i = 0; i < n; i++)
    {
        p->x[i] += dt * p->vx[i];
        p->y[i] += dt * p->vy[i];
        p->z[i] += dt * p->vz[i];
    }

#else
    for (i = 0; i < n; i++)
    {
        p[i].x += dt * p[i].vx;
        p[i].y += dt * p[i].vy;
        p[i].z += dt * p[i].vz;
    }
#endif

#elif defined(UNROLLED)

    for (i = 0; i < n; i += 4)
    {
        p[i].x += dt * p[i].vx;
        p[i].y += dt * p[i].vy;
        p[i].z += dt * p[i].vz;

        p[i + 1].x += dt * p[i + 1].vx;
        p[i + 1].y += dt * p[i + 1].vy;
        p[i + 1].z += dt * p[i + 1].vz;

        p[i + 2].x += dt * p[i + 2].vx;
        p[i + 2].y += dt * p[i + 2].vy;
        p[i + 2].z += dt * p[i + 2].vz;

        p[i + 3].x += dt * p[i + 3].vx;
        p[i + 3].y += dt * p[i + 3].vy;
        p[i + 3].z += dt * p[i + 3].vz;
    }

    i -= 4;

    if (i != n - 1)
    {
        // Traite les derniers éléments restants si n % 4 != 0
        for (; i < n; i++)
        {
            p[i].x += dt * p[i].vx;
            p[i].y += dt * p[i].vy;
            p[i].z += dt * p[i].vz;
        }
    }

#else

    // VERSION=ALL

    for (i = 0; i < n; i += 4)
    {
        p->x[i] += dt * p->vx[i];
        p->y[i] += dt * p->vy[i];
        p->z[i] += dt * p->vz[i];

        p->x[i + 1] += dt * p->vx[i + 1];
        p->y[i + 1] += dt * p->vy[i + 1];
        p->z[i + 1] += dt * p->vz[i + 1];

        p->x[i + 2] += dt * p->vx[i + 2];
        p->y[i + 2] += dt * p->vy[i + 2];
        p->z[i + 2] += dt * p->vz[i + 2];

        p->x[i + 3] += dt * p->vx[i + 3];
        p->y[i + 3] += dt * p->vy[i + 3];
        p->z[i + 3] += dt * p->vz[i + 3];
    }

    i -= 4;
    if (i != n - 1)
    {
        // Traite les derniers éléments restants si n % 4 != 0
        for (; i < n; i++)
        {
            p->x[i] += dt * p->vx[i];
            p->y[i] += dt * p->vy[i];
            p->z[i] += dt * p->vz[i];
        }
    }
#endif
}

int main(int argc, char **argv)
{
    const u64 n = (argc > 1) ? atoll(argv[1]) : 16384;
    const u64 steps = 10;
    const f32 dt = 0.01;

    f64 rate = 0.0, drate = 0.0;

    // Steps to skip for warm up
    const u64 warmup = 3;

#if defined(SOA) || defined(ALLIGNED) || defined(ALL) || defined(ALL_EXCEPT_UNROLL)
    particle_t_SOA p;
#else
    particle_t *p = malloc(sizeof(particle_t) * n);
#endif

#if defined(SOA)
    p.x = malloc(n * sizeof(f32));
    p.y = malloc(n * sizeof(f32));
    p.z = malloc(n * sizeof(f32));
    p.vx = malloc(n * sizeof(f32));
    p.vy = malloc(n * sizeof(f32));
    p.vz = malloc(n * sizeof(f32));

#elif defined(ALLIGNED) || defined(ALL) || defined(ALL_EXCEPT_UNROLL)
    p.x = aligned_alloc(sizeof(f32), n * sizeof(f32));
    p.y = aligned_alloc(sizeof(f32), n * sizeof(f32));
    p.z = aligned_alloc(sizeof(f32), n * sizeof(f32));
    p.vx = aligned_alloc(sizeof(f32), n * sizeof(f32));
    p.vy = aligned_alloc(sizeof(f32), n * sizeof(f32));
    p.vz = aligned_alloc(sizeof(f32), n * sizeof(f32));

#endif

#if defined(SOA) || defined(ALLIGNED) || defined(ALL) || defined(ALL_EXCEPT_UNROLL)
    init(&p, n);
#else
    init(p, n);
#endif

    const u64 s = sizeof(particle_t) * n;

    printf("\n\033[1mTotal memory size:\033[0m %llu B, %llu KiB, %llu MiB\n\n", s, s >> 10, s >> 20);

    printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s", "GFLOP/s");
    fflush(stdout);

    for (u64 i = 0; i < steps; i++)
    {
        // Measure
        const f64 start = omp_get_wtime();

#if defined(SOA) || defined(ALLIGNED) || defined(ALL) || defined(ALL_EXCEPT_UNROLL)
        move_particles(&p, dt, n);
#else
        move_particles(p, dt, n);
#endif

        const f64 end = omp_get_wtime();

        // Number of interactions/iterations
        const f32 h1 = (f32)(n) * (f32)(n - 1);
        // GFLOPS
        const f32 h2 = (23.0 * h1 + 3.0 * (f32)n) * 1e-9;

        if (i >= warmup)
        {
            rate += h2 / (end - start);
            drate += (h2 * h2) / ((end - start) * (end - start));
        }

        printf("%5llu %10.3e %10.3e %8.1f %s\n",
               i,
               (end - start),
               h1 / (end - start),
               h2 / (end - start),
               (i < warmup) ? "*" : "");

        fflush(stdout);
    }

#if defined(SOA) || defined(ALLIGNED) || defined(ALL) || defined(ALL_EXCEPT_UNROLL)
    // pour voir si les différentes optimisations donnent les même positions quand elles sont initialiement fixées (pas aléatoire)
    for (int num_particule = 0; num_particule < 3; num_particule++)
    {
        printf("\nParticule %d:\n", num_particule);
        printf("x: %g, ", p.x[num_particule]);
        printf("y: %g, ", p.y[num_particule]);
        printf("z: %g, ", p.z[num_particule]);
        printf("vx: %g, ", p.vx[num_particule]);
        printf("vy: %g, ", p.vy[num_particule]);
        printf("vz: %g\n", p.vz[num_particule]);
    }
#else
    // pour voir si les différentes optimisations donnent les même positions quand elles sont initialiement fixées (pas aléatoire)
    for (int num_particule = 0; num_particule < 3; num_particule++)
    {
        printf("\nParticule %d:\n", num_particule);
        printf("x: %g, ", p[num_particule].x);
        printf("y: %g, ", p[num_particule].y);
        printf("z: %g, ", p[num_particule].z);
        printf("vx: %g, ", p[num_particule].vx);
        printf("vy: %g, ", p[num_particule].vy);
        printf("vz: %g\n", p[num_particule].vz);
    }
#endif

    rate /= (f64)(steps - warmup);
    drate = sqrt(drate / (f64)(steps - warmup) - (rate * rate));

    printf("-----------------------------------------------------\n");
    printf("\033[1m%s %4s \033[42m%10.1lf +- %.1lf GFLOP/s\033[0m\n",
           "Average performance:", "", rate, drate);
    printf("-----------------------------------------------------\n");

#if defined(SOA) || defined(ALLIGNED) || defined(ALL) || defined(ALL_EXCEPT_UNROLL)
    free(p.x);
    free(p.y);
    free(p.z);
    free(p.vx);
    free(p.vy);
    free(p.vz);
#else
    free(p);
#endif
    return 0;
}