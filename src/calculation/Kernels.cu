// Copyright (c) 2025. Shuojiang Liu.

#include <calculation/Kernels.cuh>

__global__ void advect_kernel(CudaTextureAccessor<float4> texVel, CudaSurfaceAccessor<float4> sufLoc,
                              CudaSurfaceAccessor<char> sufBound, unsigned int n) {
    // Thread indexing
    // We compute the global 3D index (x, y, z)
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n)
        return;

    // Inline function (lambda) for sampling the velocity from a 3D texture
    // Fetches a velocity vector (stored as a float4) from a 3D texture at location loc
    auto sample = [](CudaTextureAccessor<float4> tex, float3 loc) -> float3 {
        float4 vel = tex.sample(loc.x, loc.y, loc.z);
        return make_float3(vel.x, vel.y, vel.z);
    };

    // Construct the cell-center location
    // Typically, we store fluid values at cell centers. So the cell center is (x+0.5, y+0.5, z+0.5)
    float3 loc = make_float3(x + 0.5f, y + 0.5f, z + 0.5f);

    // This means “if this cell is a fluid cell (not a solid boundary)”.
    // A negative boundary marker might indicate solid or out of domain.
    if (sufBound.read(x, y, z) >= 0) {
        // Runge-Kutta 3-style integration. Advection can be described by the ordinary differential equation:
        // dx / dt = v(x,t)
        /*
         * We are stepping backward in time from x_current to find where the fluid came from (semi-Lagrangian
         * advection). That is, if in forward time you go from x_old to x_new, then for advection we do the inverse:
         * x_old = x_new - Δt * v(x,t)
         */
        // vel1, vel2, and vel3 are velocity samples used in a multi-stage integration:
        // Stage 1 of RK: k1 = v(x).
        float3 vel1 = sample(texVel, loc);
        // Stage 2 of RK, half step using vel1: k2 = v(x − 1/2 * Δt * k1).
        float3 vel2 = sample(texVel, loc - 0.5f * vel1);
        // Stage 3 of RK, 0.75 step using vel2: k3 = v(x − α * Δt * k2).
        float3 vel3 = sample(texVel, loc - 0.75f * vel2);
        // Then we compute a combined stages with RK coefficients:
        // x_new = x − Δt * (β1 * k1 + β2 * k2 + β3 * k3)
        loc -= (2.f / 9.f) * vel1 + (1.f / 3.f) * vel2 + (4.f / 9.f) * vel3;
    }

    // After computing the new location, we store it (with a dummy 4th component).
    // Another pass might then use these loc values to fetch data from the old time step.
    sufLoc.write(make_float4(loc.x, loc.y, loc.z, 0.f), x, y, z);
}

__global__ void decay_kernel(CudaSurfaceAccessor<float> sufTmp, CudaSurfaceAccessor<float> sufTmpNext,
                             CudaSurfaceAccessor<char> sufBound, float ambientRate, float decayRate, unsigned int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n)
        return;
    if (sufBound.read(x, y, z) < 0)
        return;

    // Reads six neighboring cells in the ±x,±y,±z±x,±y,±z directions
    float txp = sufTmp.read<cudaBoundaryModeClamp>(x + 1, y, z);
    float typ = sufTmp.read<cudaBoundaryModeClamp>(x, y + 1, z);
    float tzp = sufTmp.read<cudaBoundaryModeClamp>(x, y, z + 1);
    float txn = sufTmp.read<cudaBoundaryModeClamp>(x - 1, y, z);
    float tyn = sufTmp.read<cudaBoundaryModeClamp>(x, y - 1, z);
    float tzn = sufTmp.read<cudaBoundaryModeClamp>(x, y, z - 1);

    // Takes the average of these neighbors
    float tmpAvg = (txp + typ + tzp + txn + tyn + tzn) * (1 / 6.f);

    // Blends the current cell’s temperature tmpNext with this average:
    // tmpNext ← (tmpNext * ambientRate + tmpAvg * (1 − ambientRate)) × decayRate.
    float tmpNext = sufTmp.read(x, y, z);
    // ambientRate controls how strongly we keep the cell’s original value vs. the neighbors’ average.
    // decayRate is a multiplier that reduces (decays) the value in each iteration—useful to avoid indefinite build-up
    // of temperature or smoke.
    tmpNext = (tmpNext * ambientRate + tmpAvg * (1.f - ambientRate)) * decayRate;
    sufTmpNext.write(tmpNext, x, y, z);
}

__global__ void divergence_kernel(CudaSurfaceAccessor<float4> sufVel, CudaSurfaceAccessor<float> sufDiv,
                                  CudaSurfaceAccessor<char> sufBound, unsigned int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n)
        return;
    // If the cell is a boundary cell ( < 0 ), we set divergence to 0.
    if (sufBound.read(x, y, z) < 0) {
        sufDiv.write(0.f, x, y, z);
        return;
    }

    /*
     * Computes the finite-difference divergence of the velocity field:
     * ∇⋅v ≈ (v_x(x+1) − v_x(x−1)) / (2*Δx) + (v_y(y+1) − v_y(y−1)) / (2*Δy) + (v_z(z+1) − v_z(z−1)) / (2*Δz).
     *
     * We use a spacing of 1 in each dimension, hence the factor 0.5f (equivalent to 1 / (2*Δx) if Δx = 1).
     */
    float vxp = sufVel.read<cudaBoundaryModeClamp>(x + 1, y, z).x;
    float vyp = sufVel.read<cudaBoundaryModeClamp>(x, y + 1, z).y;
    float vzp = sufVel.read<cudaBoundaryModeClamp>(x, y, z + 1).z;
    float vxn = sufVel.read<cudaBoundaryModeClamp>(x - 1, y, z).x;
    float vyn = sufVel.read<cudaBoundaryModeClamp>(x, y - 1, z).y;
    float vzn = sufVel.read<cudaBoundaryModeClamp>(x, y, z - 1).z;
    float div = (vxp - vxn + vyp - vyn + vzp - vzn) * 0.5f;

    // Typically, an incompressible fluid requires ∇ ⋅ v = 0.
    // Any non-zero divergence that accumulates must be corrected by the pressure solver step.
    // The result is stored in sufDiv.
    sufDiv.write(div, x, y, z);
}

__global__ void sumloss_kernel(CudaSurfaceAccessor<float> sufDiv, float *sum, unsigned int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n)
        return;

    float div = sufDiv.read(x, y, z);
    // Sums up ∑(divergence^2) over all cells into a single GPU variable sum.
    // Used to measure how close we are to ∇ ⋅ v = 0. Minimizing ∥∇ ⋅ v∥ is part of the pressure projection step.
    atomicAdd(sum, div * div);
}

__global__ void heatup_kernel(CudaSurfaceAccessor<float4> sufVel, CudaSurfaceAccessor<float> sufTmp,
                              CudaSurfaceAccessor<float> sufClr, CudaSurfaceAccessor<char> sufBound, float tmpAmbient,
                              float heatRate, float clrRate, unsigned int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n)
        return;
    // Skip if solid/out-of-bound
    if (sufBound.read(x, y, z) < 0)
        return;

    // Fetch current velocity, temperature, color
    float4 vel = sufVel.read(x, y, z);
    float tmp = sufTmp.read(x, y, z);
    float clr = sufClr.read(x, y, z);

    // Upward force from excess temperature over ambient
    // A simplified buoyancy and vorticity effect for smoke is often introduced
    // by adding an upward velocity if the temperature is above ambient.
    // v_z ← v_z + α × (T − T_ambient), which is a typical buoyancy forcing term
    vel.z += heatRate * (tmp - tmpAmbient);

    // Downward force from color (e.g. heavy smoke)
    // Represent additional forces from smoke “heaviness”
    // or a combined effect of color/density acting as negative buoyancy.
    vel.z -= clrRate * clr;

    // Write updated velocity
    sufVel.write(vel, x, y, z);
}

__global__ void subgradient_kernel(CudaSurfaceAccessor<float> sufPre, CudaSurfaceAccessor<float4> sufVel,
                                   CudaSurfaceAccessor<char> sufBound, unsigned int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n)
        return;
    if (sufBound.read(x, y, z) < 0)
        return;

    /*
     * Once we compute the pressure p, the velocity is updated to remove its divergence:
     * v_new = v_old − ∇p.
     * Subtracts half the difference in pressure on each axis from velocity components.
     * The factor 0.5f is again from the (p_{i+1} − p_{i−1}) / 2 derivative approximation.
     */
    float pxn = sufPre.read<cudaBoundaryModeZero>(x - 1, y, z);
    float pyn = sufPre.read<cudaBoundaryModeZero>(x, y - 1, z);
    float pzn = sufPre.read<cudaBoundaryModeZero>(x, y, z - 1);
    float pxp = sufPre.read<cudaBoundaryModeZero>(x + 1, y, z);
    float pyp = sufPre.read<cudaBoundaryModeZero>(x, y + 1, z);
    float pzp = sufPre.read<cudaBoundaryModeZero>(x, y, z + 1);

    // Subtract pressure gradient from velocity
    float4 vel = sufVel.read(x, y, z);
    vel.x -= (pxp - pxn) * 0.5f;
    vel.y -= (pyp - pyn) * 0.5f;
    vel.z -= (pzp - pzn) * 0.5f;

    sufVel.write(vel, x, y, z);
}

__global__ void residual_kernel(CudaSurfaceAccessor<float> sufRes, CudaSurfaceAccessor<float> sufPre,
                                CudaSurfaceAccessor<float> sufDiv, unsigned int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n)
        return;

    /*
     * Computes the residual of the Poisson equation:
     * res = Δp − div = (p_{x+1} + p_{x−1} + p_{y+1} + p_{y−1} + p_{z+1} + p_{z−1} − 6p) − div.
     * If the residual is zero, we have solved ∇^2(p) = div∇2p exactly at that cell.
     */
    float pxp = sufPre.read<cudaBoundaryModeClamp>(x + 1, y, z);
    float pxn = sufPre.read<cudaBoundaryModeClamp>(x - 1, y, z);
    float pyp = sufPre.read<cudaBoundaryModeClamp>(x, y + 1, z);
    float pyn = sufPre.read<cudaBoundaryModeClamp>(x, y - 1, z);
    float pzp = sufPre.read<cudaBoundaryModeClamp>(x, y, z + 1);
    float pzn = sufPre.read<cudaBoundaryModeClamp>(x, y, z - 1);
    float pre = sufPre.read(x, y, z);
    float div = sufDiv.read(x, y, z);
    float res = pxp + pxn + pyp + pyn + pzp + pzn - 6.f * pre - div;

    sufRes.write(res, x, y, z);
}

__global__ void restrict_kernel(CudaSurfaceAccessor<float> sufPreNext, CudaSurfaceAccessor<float> sufPre,
                                unsigned int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n)
        return;

    /*
     * This is part of a multigrid hierarchy approach (MG or V-cycle approach).
     * Restriction: We take 8 neighboring cells at a fine level (the 2x2x2 block) and combine them into 1 cell at the
     * coarse level (the (x,y,z) in the coarser grid). The sum or average is used as the restricted value. Then we can
     * solve on a coarser grid more easily (fewer unknowns). In practice, we might do an average instead of a sum:
     * p_coarse(x,y,z) = [∑_{i=0..1, j=0..1, k=0..1}p_fine(2x+i, 2y+j, 2z+k)] / 8
     */
    // Read the 8 children in the finer level
    float ooo = sufPre.read<cudaBoundaryModeClamp>(x * 2, y * 2, z * 2);
    float ioo = sufPre.read<cudaBoundaryModeClamp>(x * 2 + 1, y * 2, z * 2);
    float oio = sufPre.read<cudaBoundaryModeClamp>(x * 2, y * 2 + 1, z * 2);
    float iio = sufPre.read<cudaBoundaryModeClamp>(x * 2 + 1, y * 2 + 1, z * 2);
    float ooi = sufPre.read<cudaBoundaryModeClamp>(x * 2, y * 2, z * 2 + 1);
    float ioi = sufPre.read<cudaBoundaryModeClamp>(x * 2 + 1, y * 2, z * 2 + 1);
    float oii = sufPre.read<cudaBoundaryModeClamp>(x * 2, y * 2 + 1, z * 2 + 1);
    float iii = sufPre.read<cudaBoundaryModeClamp>(x * 2 + 1, y * 2 + 1, z * 2 + 1);
    // Sum them (no division here)
    float preNext = (ooo + ioo + oio + iio + ooi + ioi + oii + iii);

    // Write out to the coarser grid
    sufPreNext.write(preNext, x, y, z);
}

__global__ void fillzero_kernel(CudaSurfaceAccessor<float> sufPre, unsigned int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n)
        return;

    // Zeros out the pressure or some field.
    // Used as initialization in a new coarser level for MG or anywhere we need a fresh slate.
    sufPre.write(0.f, x, y, z);
}

__global__ void prolongate_kernel(CudaSurfaceAccessor<float> sufPreNext, CudaSurfaceAccessor<float> sufPre,
                                  unsigned int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n)
        return;

    /*
     * The prolongation step in the multigrid method:
     * the coarse-grid correction is spread (or injected) back onto the fine grid.
     *
     * Here, we read the coarse cell (x, y, z) from sufPre,
     * then we distribute preDelta to the 8 child cells (2x+dx, 2y+dy, 2z+dz) in sufPreNext.
     *
     * Standard MG approach:
     * p_fine(2x+i, 2y+j, 2z+k) += w × p_coarse(x, y, z)
     */
    // Read the coarse value and compute the delta to add
    float preDelta = sufPre.read(x, y, z) * (0.5f / 8.f);

    // Distribute to 2x2x2 children
#pragma unroll
    for (int dz = 0; dz < 2; dz++) {
#pragma unroll
        for (int dy = 0; dy < 2; dy++) {
#pragma unroll
            for (int dx = 0; dx < 2; dx++) {
                float preNext = sufPreNext.read<cudaBoundaryModeZero>(x * 2 + dx, y * 2 + dy, z * 2 + dz);
                preNext += preDelta;
                sufPreNext.write<cudaBoundaryModeZero>(preNext, x * 2 + dx, y * 2 + dy, z * 2 + dz);
            }
        }
    }
}
