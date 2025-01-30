// Copyright (c) 2025. Shuojiang Liu.

#include <calculation/Kernels.cuh>
#include <components/CudaArray.cuh>
#include <iostream>
#include <simulation/SimulationSystem.cuh>
#include <thread>
#include <utility>
#include <vdb/VDBWriter.hpp>
#include <vector>

int main() {
    // Sets the resolution in each dimension
    // Total simulation domain is a cubic grid of size 128x128x128
    constexpr unsigned int n = 128;
    SimulationSystem sim(n);

    // Initializing the boundary (Signed Distance Field)
    {
        // Allocate a CPU-side buffer for n^3 cells of type char.
        // This will store boundary information in a discrete 3D grid
        std::vector<char> cpu(n * n * n);

        for (int z = 0; z < n; ++z) {
            for (int y = 0; y < n; ++y) {
                for (int x = 0; x < n; ++x) {
                    /*
                     * This effectively creates a sphere (or spherical region) of radius r1 = n / 12 centered at (n/2,
                     * n/2, n/4). If we are inside that sphere, SDF = −1. If you are outside, SDF = 1.
                     */
                    const char sdf1 = std::hypot(x - (int) n / 2, y - (int) n / 2, z - (int) n / 4) < n / 12 ? -1 : 1;
                    const char sdf2 =
                            std::hypot(x - (int) n / 2, y - (int) n / 2, z - (int) n * 3 / 4) < n / 6 ? 1 : -1;

                    // index = x + n * (y + n * z) for flattened indexing
                    cpu[x + n * (y + n * z)] = sdf1;
                }
            }
        }

        // Copy the boundary data from the CPU vector to the GPU buffer sim.bound
        sim.bound->copyIn(cpu.data());
    }

    // Initializing the density field
    {
        std::vector<float> cpu(n * n * n);

        for (int z = 0; z < n; ++z) {
            for (int y = 0; y < n; ++y) {
                for (int x = 0; x < n; ++x) {
                    /*
                     * Again check if a point (x, y, z) is within the same sphere as above (center=(n/2,n/2,n/4),
                     * radius=n/12). If inside, density = 1.0. Otherwise, density = 0.0.
                     */
                    const float den =
                            std::hypot(x - (int) n / 2, y - (int) n / 2, z - (int) n / 4) < n / 12 ? 1.f : 0.f;
                    cpu[x + n * (y + n * z)] = den;
                }
            }
        }

        sim.clr->copyIn(cpu.data());
    }

    // Initializing the temperature field
    {
        std::vector<float> cpu(n * n * n);

        for (int z = 0; z < n; ++z) {
            for (int y = 0; y < n; ++y) {
                for (int x = 0; x < n; ++x) {
                    // Similarly, if inside the sphere, temperature = 1.0; otherwise 0.0.
                    const float tmp =
                            std::hypot(x - (int) n / 2, y - (int) n / 2, z - (int) n / 4) < n / 12 ? 1.f : 0.f;
                    cpu[x + n * (y + n * z)] = tmp;
                }
            }
        }

        sim.tmp->copyIn(cpu.data());
    }

    // Initializing the velocity field
    {
        // The velocity field is stored in a 4D float vector (x, y, z, and possibly w). Often w is unused or stores
        // pressure, etc.
        std::vector<float4> cpu(n * n * n);

        for (int z = 0; z < n; ++z) {
            for (int y = 0; y < n; ++y) {
                for (int x = 0; x < n; ++x) {
                    /*
                     * Again testing if the point (x, y, z) is within the sphere. If yes, we set vel = 0.9f, else 0.
                     * Then the final velocity is (0, 0, 0.9f * 0.1f, 0) = (0, 0, 0.09, 0) for inside the sphere.
                     * This sets an initial upward velocity (in the z-direction) for the region near (n/2, n/2, n/4).
                     * This simulates rising smoke.
                     */
                    const float vel =
                            std::hypot(x - (int) n / 2, y - (int) n / 2, z - (int) n / 4) < n / 12 ? 0.9f : 0.f;
                    cpu[x + n * (y + n * z)] = make_float4(0.f, 0.f, vel * 0.1f, 0.f);
                }
            }
        }

        sim.vel->copyIn(cpu.data());
    }

    // A collection of std::thread objects for asynchronous data saving
    std::vector<std::thread> tpool;

    // simulate 200200 frames
    for (int frame = 1; frame <= 200; ++frame) {
        std::vector<float> cpuClr(n * n * n);
        std::vector<float> cpuTmp(n * n * n);

        // Copy the GPU data (density and temperature) back to the CPU for output
        sim.clr->copyOut(cpuClr.data());
        sim.tmp->copyOut(cpuTmp.data());

        // Launch a background thread that writes those fields to a .vdb file
        // We Capture the data in local cpuClr and cpuTmp by move semantics (std::move)
        // so the background thread can safely write it out without blocking the main simulation loop
        tpool.push_back(std::thread([cpuClr = std::move(cpuClr), cpuTmp = std::move(cpuTmp), frame, n] {
            VDBWriter writer;

            // Use VDBWriter to add grids “density” and “temperature” to an output VDB file named aXYZ.vdb
            writer.addGrid<float, 1>("density", cpuClr.data(), n, n, n);
            writer.addGrid<float, 1>("temperature", cpuTmp.data(), n, n, n);
            // zero-pad the frame number (e.g., frame 1 => “001”)
            writer.write("/tmp/a" + std::to_string(1000 + frame).substr(1) + ".vdb");
        }));

        printf("frame=%d, loss=%f\n", frame, sim.calc_loss());
        // Call sim.step() to advance the simulation by one time step on the GPU
        sim.step();
    }

    for (auto &t: tpool) {
        t.join();
    }

    printf("Execution Successful!!!\n");

    return 0;
}
