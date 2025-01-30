// Copyright (c) 2025. Shuojiang Liu.

/**************************************************************************************
 * \file SimulationSystem.cuh
 * \brief Defines the SmokeSim class for GPU-based smoke/fluid simulations using CUDA.
 *
 * This header declares a GPU-accelerated smoke solver that uses semi-Lagrangian
 * advection and multigrid pressure projection, among other features.
 **************************************************************************************/

#ifndef SIMULATIONSYSTEM_CUH
#define SIMULATIONSYSTEM_CUH

#include <components/CudaArray.cuh>
#include <memory>

/**
 * \class SimulationSystem
 * \brief A GPU-based 3D smoke simulation class.
 *
 * The SmokeSim class encapsulates a 3D CUDA-based smoke/fluid solver. It manages fields
 * such as velocity, density (referred to as "clr"), temperature ("tmp"), and pressure.
 * It provides methods for advecting these fields, enforcing incompressibility via a
 * multigrid Poisson solver, and applying buoyancy and other forces.
 *
 * ### Equations
 *
 * This solver implements a simplified incompressible Navier--Stokes system:
 * \f[
 *   \frac{\partial \mathbf{u}}{\partial t}
 *   + (\mathbf{u} \cdot \nabla)\mathbf{u}
 *   = -\frac{1}{\rho}\nabla p + \nu\,\nabla^2 \mathbf{u} + \mathbf{f},
 *   \quad
 *   \nabla \cdot \mathbf{u} = 0.
 * \f]
 *
 * Temperature (\f$ T \f$) and density (\f$ c \f$) advect according to:
 * \f[
 *   \frac{\partial \phi}{\partial t} + (\mathbf{u}\cdot\nabla)\phi = \kappa \nabla^2 \phi + \text{(source/decay)}.
 * \f]
 *
 * The projection step ensures \f$ \nabla \cdot \mathbf{u} = 0 \f$ by solving
 * \f[
 *   \nabla^2 p = \nabla \cdot \mathbf{u},
 *   \quad
 *   \mathbf{u} \leftarrow \mathbf{u} - \nabla p.
 * \f]
 */
class SimulationSystem : public CopyDisabler {
public:
    /**
     * \brief The grid resolution in each dimension (\f$ n \times n \times n\f$).
     */
    unsigned int n;

    /**
     * \brief Surface for storing particle locations or backtraced positions during advection.
     *
     * \note Typically used in a semi-Lagrangian scheme to store the position
     *       from which a voxel's value is sampled.
     */
    std::unique_ptr<CudaSurface<float4>> loc;
    /**
     * \brief 3D texture storing the current velocity field (\f$ \mathbf{u} \f$).
     *
     * Each voxel is a float4, typically storing (u, v, w, 0). Texture usage allows
     * hardware interpolation during semi-Lagrangian advection.
     */
    std::unique_ptr<CudaTexture<float4>> vel;
    /**
     * \brief 3D texture for the next-timestep velocity field.
     *
     * Used as a target to avoid read-write conflicts during advection steps.
     */
    std::unique_ptr<CudaTexture<float4>> velNext;
    /**
     * \brief 3D texture storing the smoke density (or "color").
     *
     * This is the field that is rendered visually as smoke. Values typically range
     * from 0 to 1, though the solver does not strictly limit the range.
     */
    std::unique_ptr<CudaTexture<float>> clr;
    /**
     * \brief 3D texture for next-timestep smoke density.
     *
     * Used for ping-pong updates to avoid overwriting data while it is still needed.
     */
    std::unique_ptr<CudaTexture<float>> clrNext;
    /**
     * \brief 3D texture for temperature values.
     *
     * Used in the buoyancy model to compute upward force based on temperature differences.
     */
    std::unique_ptr<CudaTexture<float>> tmp;
    /**
     * \brief 3D texture for next-timestep temperature.
     *
     * Ping-pong buffer for the temperature field updates.
     */
    std::unique_ptr<CudaTexture<float>> tmpNext;

    /**
     * \brief Surface holding boundary or obstacle information.
     *
     * A char-based signed distance or mask. Negative values often represent "inside" a solid,
     * positive or zero represent fluid regions.
     */
    std::unique_ptr<CudaSurface<char>> bound;
    /**
     * \brief Surface storing the velocity divergence (\f$ \nabla \cdot \mathbf{u} \f$).
     */
    std::unique_ptr<CudaSurface<float>> div;
    /**
     * \brief Surface storing the solved pressure field (\f$ p \f$).
     */
    std::unique_ptr<CudaSurface<float>> pre;
    // used in old Jacobi-based projection
    // std::unique_ptr<CudaSurface<float>> preNext;

    /**
     * \brief Array of surfaces to store residuals in the multigrid V-cycle.
     *
     * Each level has a corresponding resolution. For level \c i, \c res[i] is size
     * \f$ \text{sizes}[i]^3 \f$.
     */
    std::vector<std::unique_ptr<CudaSurface<float>>> res;
    /**
     * \brief Array of surfaces for restricted residual (coarser grid).
     *
     * Each is half the resolution in each dimension relative to \c res[i].
     */
    std::vector<std::unique_ptr<CudaSurface<float>>> res2;
    /**
     * \brief Array of surfaces to store the error/correction in the V-cycle.
     */
    std::vector<std::unique_ptr<CudaSurface<float>>> err2;
    /**
     * \brief Holds the resolution for each multigrid level, from finest to coarsest.
     *
     * For example, if \c n=128 and \c _n0=16, this might be {128, 64, 32, 16}.
     */
    std::vector<unsigned int> sizes;

    /**
     * \brief Constructs the SmokeSim object and allocates GPU buffers.
     * \param otherN    The size of the simulation grid (e.g. 128 means 128x128x128).
     * \param n0   The minimum resolution for the coarsest multigrid level (defaults to 16).
     *
     * This constructor initializes CUDA surfaces/textures for velocity, density, temperature,
     * boundary, pressure, and divergence. It also prepares data structures for a multigrid
     * solver, building levels down to a specified coarsest resolution.
     *
     * \note It also zeroes out the pressure field initially using \c fillzero_kernel.
     */
    explicit SimulationSystem(unsigned int otherN, unsigned int n0 = 16);

    /**
     * \brief Applies a smoothing operation (Red-Black Gauss--Seidel) on a multigrid level.
     * \param v     Surface of the solution variable (e.g. pressure).
     * \param f     Surface of the right-hand side (e.g. divergence).
     * \param lev   The multigrid level index (0 = finest).
     * \param times Number of smoothing iterations to apply.
     *
     * Each call does several passes of Red-Black Gauss--Seidel to reduce high-frequency
     * errors on the current grid level:
     * \f[
     *   v_{\mathrm{new}} \leftarrow \text{RBGS}(v_{\mathrm{old}}, f).
     * \f]
     */
    void smooth(CudaSurface<float> *v, CudaSurface<float> *f, unsigned int lev, int times = 4);

    /**
     * \brief Recursively applies a V-cycle on a specific multigrid level for the Poisson solver.
     * \param lev Index of the current multigrid level.
     * \param v   Surface of the solution variable (pressure) at this level.
     * \param f   Surface of the right-hand side (divergence) at this level.
     *
     * The V-cycle procedure:
     * 1. Pre-smooth the solution.
     * 2. Compute residual \f$ r = f - \nabla^2 v \f$.
     * 3. Restrict the residual to a coarser level.
     * 4. Solve on the coarser level (recursively).
     * 5. Prolongate the correction back to the finer level.
     * 6. Post-smooth the solution.
     *
     * This approach converges the Poisson equation more efficiently than naive iteration.
     */
    void vcycle(unsigned int lev, CudaSurface<float> *v, CudaSurface<float> *f);

    /**
     * \brief Performs the projection step of the fluid solver (making velocity divergence-free).
     *
     * 1. Adds buoyancy/heat forces to velocity (\c heatup_kernel).
     * 2. Computes \f$ \nabla \cdot \mathbf{u} \f$ into \c div.
     * 3. Solves \f$ \nabla^2 p = \nabla \cdot \mathbf{u} \f$ using multigrid V-cycle.
     * 4. Subtracts \f$ \nabla p \f$ from \mathbf{u} to enforce \f$ \nabla \cdot \mathbf{u} = 0 \f$.
     */
    void projection();

    /**
     * \brief Performs the advection step for velocity, density, and temperature.
     *
     * 1. Backtrace positions and store them in \c loc via \c advect_kernel.
     * 2. Resample velocity, density, and temperature from old positions into
     *    the "next" buffers.
     * 3. Swap the "next" fields with the current ones.
     * 4. Apply exponential decay to \c tmp (temperature) and \c clr (density)
     *    for gradual dissipation.
     *
     * Why do we do advection and then decay?
     * - Advection moves the fields along the velocity.
     * - Decay/diffusion is a simplified model so that smoke and heat do not remain forever in the domain and so that
     * they spread out more realistically.
     */
    void advection();

    /**
     * \brief Advances the simulation by a given number of sub-steps.
     * \param times Number of sub-steps to perform in each call (default 16).
     *
     * Each sub-step calls:
     * 1. \c projection() -- Enforce divergence-free velocity and apply buoyancy.
     * 2. \c advection()  -- Semi-Lagrangian advection of velocity, density, and temperature.
     */
    void step(int times = 16);

    /**
     * \brief Computes a "loss" metric based on the current velocity divergence.
     * \return A floating-point value representing the sum of divergence across the grid.
     *
     * This is a diagnostic function to measure how well the solver enforces
     * incompressibility (\f$ \nabla \cdot \mathbf{u} = 0 \f$). Lower values indicate
     * better enforcement.
     */
    float calc_loss();
};

#endif // SIMULATIONSYSTEM_CUH
