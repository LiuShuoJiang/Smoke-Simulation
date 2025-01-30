// Copyright (c) 2025. Shuojiang Liu.

#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <components/CudaArray.cuh>
#include <utils/HelperMath.cuh>

/**
 * \brief Backtrace cell centers to their origin position using a 3rd-order RK method,
 *        storing the traced location for subsequent resampling.
 *
 * \details
 * This kernel implements a semi-Lagrangian advection step for velocity or any scalar field
 * by computing where each cell center came from in the previous time step (a backtrace).
 *
 * Mathematically, if \f$\mathbf{x}\f$ is the current cell-center location, we want
 * \f[
 *  \mathbf{x}_{\mathrm{old}} \approx \mathbf{x} - \Delta t \cdot \mathbf{v}(\mathbf{x}, t).
 * \f]
 * We perform a 3-stage Runge-Kutta integration to improve accuracy:
 * \f[
 * \begin{aligned}
 *    \mathbf{k}_1 &= \mathbf{v}(\mathbf{x}), \\
 *    \mathbf{k}_2 &= \mathbf{v}\bigl(\mathbf{x} - 0.5 \,\mathbf{k}_1\bigr), \\
 *    \mathbf{k}_3 &= \mathbf{v}\bigl(\mathbf{x} - 0.75\, \mathbf{k}_2\bigr), \\
 *    \mathbf{x}_{\mathrm{old}} &\approx \mathbf{x} - \Bigl(\tfrac{2}{9}\mathbf{k}_1 + \tfrac{1}{3}\mathbf{k}_2 +
 * \tfrac{4}{9}\mathbf{k}_3\Bigr).
 * \end{aligned}
 * \f]
 *
 * \param[in]  texVel   A texture accessor for the velocity field (float4), where xyz are velocity.
 * \param[out] sufLoc   A surface to store the traced-back positions as float4 (only xyz used).
 * \param[in]  sufBound A surface accessor for boundary flags: negative indicates solid/off-domain.
 * \param[in]  n        The size of the cubic grid in each dimension (x, y, z).
 *
 * \note The cell-center is defined at \f$(x + 0.5, \, y + 0.5, \, z + 0.5)\f$ to avoid sampling artifacts.
 */
__global__ void advect_kernel(CudaTextureAccessor<float4> texVel, CudaSurfaceAccessor<float4> sufLoc,
                              CudaSurfaceAccessor<char> sufBound, unsigned int n);

/**
 * \brief Samples a scalar (or color) field at a backtraced location and writes it
 *        into the next timestep buffer.
 *
 * \details
 * After we've computed the backtraced location of each cell center in \c advect_kernel,
 * we fetch the old field value (e.g., color or density) from that location. This implements
 * the typical semi-Lagrangian resampling step:
 * \f[
 *   \phi^{n+1}(\mathbf{x}) = \phi^{n}(\mathbf{x}_{\mathrm{old}}).
 * \f]
 *
 * \tparam T             The type of the scalar field (e.g. float, float4, etc.).
 * \param[in]  sufLoc    A surface accessor holding the backtraced positions (x,y,z in float4).
 * \param[in]  texClr    A texture accessor for the old scalar/color data.
 * \param[out] sufClrNext The output surface to store the updated field (at current time).
 * \param[in]  n         The size of the cubic grid in each dimension.
 */
template<class T>
__global__ void resample_kernel(CudaSurfaceAccessor<float4> sufLoc, CudaTextureAccessor<T> texClr,
                                CudaSurfaceAccessor<T> sufClrNext, unsigned int n) {
    // Global indexing
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n)
        return;

    // Fetch the previously stored backtraced location
    float4 loc = sufLoc.read(x, y, z);
    // We fetch from old 3D texture texClr at position (loc.x, loc.y, loc.z).
    // This is the color field or some scalar quantity from the previous timestep.
    T clr = texClr.sample(loc.x, loc.y, loc.z);

    // We set the color/density in the “next” buffer (the updated field) to the old field's sample from the backtraced
    // location. This finalizes the semi-Lagrangian step for ϕ.
    sufClrNext.write(clr, x, y, z);
}

/**
 * \brief Diffuse and decay a scalar field (e.g. temperature) using simple neighbor averaging.
 *
 * \details
 * This kernel computes a new value for each cell by blending between the cell's current value
 * and the average of its 6 neighbors. The user can set how strongly the cell's own value is
 * preserved (\c ambientRate) and how strongly the field decays per iteration (\c decayRate).
 *
 * The update rule is:
 * \f[
 *   T_{\text{next}} = \bigl(T_{\text{curr}} \times \text{ambientRate} \;+\;
 *                     \bar{T}_{\text{neighbors}} \times (1 - \text{ambientRate})\bigr)
 *                     \times \text{decayRate}.
 * \f]
 *
 * \param[in]  sufTmp      Current scalar field surface accessor.
 * \param[out] sufTmpNext  Output scalar field surface accessor for the updated values.
 * \param[in]  sufBound    Surface accessor for boundary flags (negative => skip).
 * \param[in]  ambientRate Weight toward the cell's own value vs. neighbor average.
 * \param[in]  decayRate   Multiplicative factor for damping the field.
 * \param[in]  n           The size of the cubic grid in each dimension.
 */
__global__ void decay_kernel(CudaSurfaceAccessor<float> sufTmp, CudaSurfaceAccessor<float> sufTmpNext,
                             CudaSurfaceAccessor<char> sufBound, float ambientRate, float decayRate, unsigned int n);

/**
 * \brief Computes the finite-difference divergence of the velocity field and stores it.
 *
 * \details
 * In a staggered or collocated grid approach, the discrete divergence of velocity \f$\mathbf{v}\f$
 * is approximated by:
 * \f[
 *   \nabla \cdot \mathbf{v} \approx
 *   \frac{v_x(x+1) - v_x(x-1)}{2} +
 *   \frac{v_y(y+1) - v_y(y-1)}{2} +
 *   \frac{v_z(z+1) - v_z(z-1)}{2}.
 * \f]
 *
 * If \c sufBound is negative at a cell, we set the divergence to 0 for that cell (solid or invalid).
 *
 * \param[in]  sufVel   Velocity surface accessor (float4: xyz are velocity).
 * \param[out] sufDiv   Surface to store divergence of velocity for each cell.
 * \param[in]  sufBound Boundary flags (negative => skip).
 * \param[in]  n        The size of the grid in each dimension.
 */
__global__ void divergence_kernel(CudaSurfaceAccessor<float4> sufVel, CudaSurfaceAccessor<float> sufDiv,
                                  CudaSurfaceAccessor<char> sufBound, unsigned int n);

/**
 * \brief Accumulates the sum of squared divergence over the entire domain.
 *
 * \details
 * This kernel calculates \f$\sum (\text{divergence}^2)\f$ for all valid cells.
 * The atomic addition \c atomicAdd ensures thread safety.
 *
 * \param[in]  sufDiv  Surface accessor containing divergence values.
 * \param[out] sum     A pointer to a single float on device that accumulates the sum of squares.
 * \param[in]  n       The size of the grid in each dimension.
 */
__global__ void sumloss_kernel(CudaSurfaceAccessor<float> sufDiv, float *sum, unsigned int n);

/**
 * \brief Adds buoyancy-like effects to the z-velocity based on temperature and color.
 *
 * \details
 * This kernel modifies velocity in the z-direction by:
 * \f[
 *   v_z \leftarrow v_z + \text{heatRate} \times ( T - T_{\mathrm{ambient}} ),
 * \f]
 * and reduces it by a term proportional to color/density:
 * \f[
 *   v_z \leftarrow v_z - \text{clrRate} \times \phi.
 * \f]
 * This simulates warm fluid rising and heavy smoke pulling the fluid down.
 *
 * \param[in,out] sufVel     Velocity field surface accessor (float4: xyz => velocity).
 * \param[in]     sufTmp     Temperature field surface accessor.
 * \param[in]     sufClr     Color/density field surface accessor.
 * \param[in]     sufBound   Boundary flags (negative => skip).
 * \param[in]     tmpAmbient Ambient (base) temperature.
 * \param[in]     heatRate   Factor for how strongly temperature affects buoyancy.
 * \param[in]     clrRate    Factor for how strongly color/density reduces upward velocity.
 * \param[in]     n          The size of the grid in each dimension.
 */
__global__ void heatup_kernel(CudaSurfaceAccessor<float4> sufVel, CudaSurfaceAccessor<float> sufTmp,
                              CudaSurfaceAccessor<float> sufClr, CudaSurfaceAccessor<char> sufBound, float tmpAmbient,
                              float heatRate, float clrRate, unsigned int n);

/**
 * \brief Subtracts the gradient of pressure from velocity to enforce divergence-free flow.
 *
 * \details
 * We approximate:
 * \f[
 *   \mathbf{v}_{\text{new}} = \mathbf{v}_{\text{old}} - \nabla p,
 * \f]
 * using central differences for \f$\nabla p\f$:
 * \f[
 *   \frac{\partial p}{\partial x} \approx \frac{p_{x+1} - p_{x-1}}{2}, \quad
 *   \frac{\partial p}{\partial y} \approx \frac{p_{y+1} - p_{y-1}}{2}, \quad
 *   \frac{\partial p}{\partial z} \approx \frac{p_{z+1} - p_{z-1}}{2}.
 * \f]
 * We store the result back in \c sufVel.
 *
 * \param[in]     sufPre   Pressure field surface accessor.
 * \param[in,out] sufVel   Velocity field to be updated (xyz in float4).
 * \param[in]     sufBound Boundary flags (negative => skip).
 * \param[in]     n        The size of the grid in each dimension.
 */
__global__ void subgradient_kernel(CudaSurfaceAccessor<float> sufPre, CudaSurfaceAccessor<float4> sufVel,
                                   CudaSurfaceAccessor<char> sufBound, unsigned int n);

/**
 * \brief Performs one phase of Red-Black Gauss-Seidel relaxation for the Poisson equation.
 *
 * \details
 * For cells where \f$(x + y + z) \bmod 2 = \text{phase}\f$, we update:
 * \f[
 *   p_{\text{new}} = \frac{p_{x+1} + p_{x-1} + p_{y+1} + p_{y-1} + p_{z+1} + p_{z-1} - \text{div}}{6}.
 * \f]
 * The grid is colored in a checkerboard pattern so that we only update one color per kernel
 * call, then swap phases.
 *
 * \tparam phase  The color phase (0 or 1) to be updated in this kernel call.
 * \param[in,out] sufPre  Surface accessor for pressure field.
 * \param[in]     sufDiv  Surface accessor for divergence.
 * \param[in]     n       The size of the grid in each dimension.
 */
template<int phase>
__global__ void rbgs_kernel(CudaSurfaceAccessor<float> sufPre, CudaSurfaceAccessor<float> sufDiv, unsigned int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n)
        return;
    // Over multiple calls, we do one kernel call with phase=0 (update red cells),
    // then one kernel call with phase=1 (update black cells),
    // repeated many times to converge
    if ((x + y + z) % 2 != phase)
        return;

    // Red-Black Gauss-Seidel method for solving the Poisson equation
    /*
     * We split the grid cells into two sets (like a checkerboard, red/black):
     * (x + y + z) mod 2 == 0 (red), and (x + y + z) mod 2 == 1 (black)
     *
     * We only update one color (phase) per kernel call.
     * This approach has better convergence properties on parallel architectures than naive Gauss-Seidel.
     *
     * The iteration formula is:
     * p_new = (p_{x+1} + p_{x−1} + p_{y+1} + p_{y−1} + p_{z+1} + p_{z−1} − div) / 6
     */
    float pxp = sufPre.read<cudaBoundaryModeClamp>(x + 1, y, z);
    float pxn = sufPre.read<cudaBoundaryModeClamp>(x - 1, y, z);
    float pyp = sufPre.read<cudaBoundaryModeClamp>(x, y + 1, z);
    float pyn = sufPre.read<cudaBoundaryModeClamp>(x, y - 1, z);
    float pzp = sufPre.read<cudaBoundaryModeClamp>(x, y, z + 1);
    float pzn = sufPre.read<cudaBoundaryModeClamp>(x, y, z - 1);
    float div = sufDiv.read(x, y, z);
    float preNext = (pxp + pxn + pyp + pyn + pzp + pzn - div) * (1.f / 6.f);

    sufPre.write(preNext, x, y, z);
}

/**
 * \brief Computes the residual of \f$\nabla^2 p - \text{div}\f$ at each cell.
 *
 * \details
 * For each cell, we do:
 * \f[
 *   \text{res} = \bigl(p_{x+1} + p_{x-1} + p_{y+1} + p_{y-1} + p_{z+1} + p_{z-1}
 *             - 6 \cdot p_{\text{center}}\bigr) \;-\; \text{div}.
 * \f]
 * Writing this residual to \c sufRes helps measure convergence or feed into multigrid steps.
 *
 * \param[out] sufRes  Surface to store residuals.
 * \param[in]  sufPre  The current pressure solution.
 * \param[in]  sufDiv  The divergence field.
 * \param[in]  n       The size of the grid in each dimension.
 */
__global__ void residual_kernel(CudaSurfaceAccessor<float> sufRes, CudaSurfaceAccessor<float> sufPre,
                                CudaSurfaceAccessor<float> sufDiv, unsigned int n);

/**
 * \brief Restricts a fine-level 3D field to a coarser-level field by summing 2x2x2 blocks.
 *
 * \details
 * In a multigrid solver, we typically form a coarser grid of size \f$n/2\f$ from the finer grid
 * by some weighting of its 8 child cells. Here we do a simple sum:
 * \f[
 *   p_{\text{coarse}}(x,y,z) = \sum_{i=0}^{1}\sum_{j=0}^{1}\sum_{k=0}^{1}
 *       p_{\text{fine}}(2x+i,\; 2y+j,\; 2z+k).
 * \f]
 * Often one might store the average (divide by 8), but this code sums them.
 *
 * \param[out] sufPreNext Coarser-level surface accessor where we store restricted values.
 * \param[in]  sufPre     Finer-level pressure (or residual) surface accessor.
 * \param[in]  n          The size of the coarser grid in each dimension (the fine grid is 2n).
 */
__global__ void restrict_kernel(CudaSurfaceAccessor<float> sufPreNext, CudaSurfaceAccessor<float> sufPre,
                                unsigned int n);

/**
 * \brief Sets all cells in a 3D field to zero.
 *
 * \details
 * Often used to initialize or reset a field (such as pressure in a coarse grid)
 * before performing an iterative multigrid solve.
 *
 * \param[out] sufPre Surface accessor to be zeroed.
 * \param[in]  n      The size of the grid in each dimension.
 */
__global__ void fillzero_kernel(CudaSurfaceAccessor<float> sufPre, unsigned int n);

/**
 * \brief Prolongates a coarse-level correction back onto the fine grid in a multigrid solver.
 *
 * \details
 * Each coarse cell \f$(x, y, z)\f$ corresponds to an 2x2x2 block in the fine level:
 * \f[
 *   (2x + dx,\; 2y + dy,\; 2z + dz)\quad \text{with } dx,dy,dz \in \{0,1\}.
 * \f]
 * Here, we add a fraction of the coarse solution (\c preDelta) back onto each fine cell.
 *
 * \param[in,out] sufPreNext Fine-level surface (we add the coarse corrections to it).
 * \param[in]     sufPre     Coarse-level field with computed corrections.
 * \param[in]     n          The size of the coarse grid (the fine grid is 2n).
 */
__global__ void prolongate_kernel(CudaSurfaceAccessor<float> sufPreNext, CudaSurfaceAccessor<float> sufPre,
                                  unsigned int n);

#endif // KERNELS_CUH
