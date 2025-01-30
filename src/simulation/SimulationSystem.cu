// Copyright (c) 2025. Shuojiang Liu.

#include <calculation/Kernels.cuh>
#include <simulation/SimulationSystem.cuh>


SimulationSystem::SimulationSystem(unsigned int otherN, unsigned int n0) :
    n(otherN), loc(std::make_unique<CudaSurface<float4>>(uint3{n, n, n})),
    vel(std::make_unique<CudaTexture<float4>>(uint3{n, n, n})),
    velNext(std::make_unique<CudaTexture<float4>>(uint3{n, n, n})),
    clr(std::make_unique<CudaTexture<float>>(uint3{n, n, n})),
    clrNext(std::make_unique<CudaTexture<float>>(uint3{n, n, n})),
    tmp(std::make_unique<CudaTexture<float>>(uint3{n, n, n})),
    tmpNext(std::make_unique<CudaTexture<float>>(uint3{n, n, n})),
    div(std::make_unique<CudaSurface<float>>(uint3{n, n, n})), pre(std::make_unique<CudaSurface<float>>(uint3{n, n, n}))
    //, preNext(std::make_unique<CudaSurface<float>>(uint3{n, n, n}))
    ,
    bound(std::make_unique<CudaSurface<char>>(uint3{n, n, n})) {
    // Immediately zeroes out the pressure field. Doing so ensures a clean start before any simulation steps
    // Use a typical “divide and round up by block size” pattern to cover all voxels
    fillzero_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(pre->accessSurface(), n);

    // This is building a V-cycle or F-cycle structure for multi-grid.
    // If n=128, we get levels {128,64,32,16} (and then stop)
    for (unsigned int tn = n; tn >= n0; tn /= 2) {
        // a (tn)^3 surface to store the residual
        res.push_back(std::make_unique<CudaSurface<float>>(uint3{tn, tn, tn}));
        // a (tn/2)^3 surface to store the restricted residual
        res2.push_back(std::make_unique<CudaSurface<float>>(uint3{tn / 2, tn / 2, tn / 2}));
        // also (tn/2)^3 for the error or correction
        err2.push_back(std::make_unique<CudaSurface<float>>(uint3{tn / 2, tn / 2, tn / 2}));
        // tracks each level’s resolution. Then tn /= 2; halving for the next coarser level
        sizes.push_back(tn);
    }
}

void SimulationSystem::smooth(CudaSurface<float> *v, CudaSurface<float> *f, unsigned int lev, int times) {
    unsigned int tn = sizes[lev];

    // Do 4 sweeps per call to smooth.
    // This is typical for a pre-/post-smoothing step in a V-cycle.
    for (int step = 0; step < times; step++) {
        /*
         * For a 3D Red-Black Gauss–Seidel, each iteration:
         * Splits the grid cells into two color sets (like a checkerboard).
         * Updates all red cells based on their black neighbors, then updates all black cells based on their newly
         * updated red neighbors.
         */
        // Update the “red” cells in the checkerboard
        rbgs_kernel<0><<<dim3((tn + 7) / 8, (tn + 7) / 8, (tn + 7) / 8), dim3(8, 8, 8)>>>(v->accessSurface(),
                                                                                          f->accessSurface(), tn);
        // Update the “black” cells in the checkerboard
        rbgs_kernel<1><<<dim3((tn + 7) / 8, (tn + 7) / 8, (tn + 7) / 8), dim3(8, 8, 8)>>>(v->accessSurface(),
                                                                                          f->accessSurface(), tn);
    }
}

void SimulationSystem::vcycle(unsigned int lev, CudaSurface<float> *v, CudaSurface<float> *f) {
    if (lev >= sizes.size()) { // Check if we are below the finest pyramid level
        // If lev >= sizes.size(), we have gone beyond the coarsest level
        unsigned int tn = sizes.back() / 2;
        smooth(v, f, lev);

        return;
    }

    // Retrieve surfaces allocated for the current level’s residual
    // and the next coarser level’s data
    auto *r = res[lev].get();
    auto *r2 = res2[lev].get();
    auto *e2 = err2[lev].get();
    unsigned int tn = sizes[lev];

    /*
     * We do a classic V-cycle in multi-grid:
     * 1. Smooth on the fine level,
     * 2. Compute residual,
     * 3. Restrict residual to coarser level,
     * 4. Solve or smooth on coarse level,
     * 5. Prolongate correction back up,
     * 6. Smooth again on the fine level.
     */

    // 1) Pre-smooth on current level
    smooth(v, f, lev); // This reduces high-frequency error at the current level

    // 2) Compute the residual
    // r = ∇^2(v) − f (the leftover error)
    residual_kernel<<<dim3((tn + 7) / 8, (tn + 7) / 8, (tn + 7) / 8), dim3(8, 8, 8)>>>(
            r->accessSurface(), v->accessSurface(), f->accessSurface(), tn);
    // 3) Restrict the residual to the next coarser level
    restrict_kernel<<<dim3((tn / 2 + 7) / 8, (tn / 2 + 7) / 8, (tn / 2 + 7) / 8), dim3(8, 8, 8)>>>(
            r2->accessSurface(), r->accessSurface(), tn / 2);
    // 4) Clear the error on the coarse level
    fillzero_kernel<<<dim3((tn / 2 + 7) / 8, (tn / 2 + 7) / 8, (tn / 2 + 7) / 8), dim3(8, 8, 8)>>>(e2->accessSurface(),
                                                                                                   tn / 2);
    // 5) Recurse: vcycle on the coarse level
    vcycle(lev + 1, e2, r2);
    // 6) Prolongate the correction from coarse to fine
    prolongate_kernel<<<dim3((tn / 2 + 7) / 8, (tn / 2 + 7) / 8, (tn / 2 + 7) / 8), dim3(8, 8, 8)>>>(
            v->accessSurface(), e2->accessSurface(), tn / 2);
    // 7) Post-Smooth again on fine
    smooth(v, f, lev);
}

void SimulationSystem::projection() {
    /*
     * Modifies the velocity field based on temperature and color.
     * We add an upward force proportional to (T − T_ambient) and a downward force proportional to the color density.
     *
     * Physical interpretation: warm cells cause velocity to rise; smoke color causes some negative buoyancy.
     * These numbers are chosen to give visually pleasing, stable motion.
     */
    heatup_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(
            vel->accessSurface(), tmp->accessSurface(), clr->accessSurface(), bound->accessSurface(), 0.05f, 0.018f,
            0.004f, n);

    /*
     * Computes ∇ ⋅ v and stores it in div.
     *
     * For incompressible flow, we want ∇ ⋅ v = 0.
     * But after adding buoyancy, we generally have nonzero divergence that must be corrected by the pressure solve.
     */
    divergence_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(
            vel->accessSurface(), div->accessSurface(), bound->accessSurface(), n);

    /*
     * Runs the multi-grid solve for the Poisson equation: ∇^2(p) = ∇ ⋅ v
     *
     * pre is the pressure, div is the right-hand side.
     * This call tries to converge the pressure field from the top (fine) level down to the coarsest, then back up.
     */
    vcycle(0, pre.get(), div.get());

    /*
     * Updates v ← v − ∇p to remove the divergence.
     * After this step, ∇ ⋅ v should be much closer to 0, enforcing incompressibility.
     */
    subgradient_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(
            pre->accessSurface(), vel->accessSurface(), bound->accessSurface(), n);
}

void SimulationSystem::advection() {
    /*
     * Backtraces each cell center based on the velocity field (Runge-Kutta integration).
     * Stores the “where did we come from?” location into loc.
     */
    advect_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(
            vel->accessTexture(), loc->accessSurface(), bound->accessSurface(), n);

    /*
     * We do a semi-Lagrangian approach: read each cell’s “backtrace” from loc,
     * sample from the old texture (velocity, color, temperature) at that location, and write into the next buffer (
     * velNext, clrNext, tmpNext).
     *
     * Then we do std::swap(...) to make the newly computed next fields become the current fields.
     */
    // Semi-Lagrangian resampling of velocity, color, temperature
    resample_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(
            loc->accessSurface(), vel->accessTexture(), velNext->accessSurface(), n);
    resample_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(
            loc->accessSurface(), clr->accessTexture(), clrNext->accessSurface(), n);
    resample_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(
            loc->accessSurface(), tmp->accessTexture(), tmpNext->accessSurface(), n);
    std::swap(vel, velNext);
    std::swap(clr, clrNext);
    std::swap(tmp, tmpNext);

    /*
     * Applies a simple diffusion/decay to temperature and color fields.
     * Each kernel call does an averaging with neighbors plus an exponential damping factor.
     *
     * Exponential factors exp(−0.5)≈0.6065, etc. are chosen to achieve a certain rate of diffusion/damping.
     * This is purely empirical for controlling how fast smoke/temperature dissipates.
     */
    // Decay/diffusion steps
    decay_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(
            tmp->accessSurface(), tmpNext->accessSurface(), bound->accessSurface(), std::exp(-0.5f), std::exp(-0.0003f),
            n);
    decay_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(
            clr->accessSurface(), clrNext->accessSurface(), bound->accessSurface(), std::exp(-0.05f), std::exp(-0.003f),
            n);
    std::swap(tmp, tmpNext);
    std::swap(clr, clrNext);
}

void SimulationSystem::step(int times) {
    /*
     * We repeatedly do:
     * 1. projection: Enforce incompressibility by solving for pressure and subtracting its gradient from velocity.
     * 2. advection: Move velocity, color, and temperature fields forward in time.
     */
    for (int step = 0; step < times; step++) {
        projection();
        advection();
    }
}

float SimulationSystem::calc_loss() {
    // Compute divergence again for the current velocity field
    divergence_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(
            vel->accessSurface(), div->accessSurface(), bound->accessSurface(), n);
    // Allocate a single float sum on the GPU
    float *sum;
    checkCudaErrors(cudaMalloc(&sum, sizeof(float)));

    // Call sumloss_kernel which accumulates ∑(divergence^2)
    sumloss_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(div->accessSurface(), sum, n);

    // Copy the result back to CPU in cpu
    float cpu;
    checkCudaErrors(cudaMemcpy(&cpu, sum, sizeof(float), cudaMemcpyDeviceToHost));
    // Free GPU memory for sum
    checkCudaErrors(cudaFree(sum));

    // Return the squared-divergence total.
    // This is a measure of how “divergence-free” (incompressible) the fluid is.
    // A perfect incompressible flow would have zero
    return cpu;
}
