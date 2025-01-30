// Copyright (c) 2025. Shuojiang Liu.

#ifndef VDBWRITER_HPP
#define VDBWRITER_HPP

#include <memory>
#include <string>

/**
 * @class VDBWriter
 * @brief Provides an interface to create and manage OpenVDB grids from raw 3D data, then write them to a .vdb file.
 *
 * This class implements the PIMPL (Pointer to Implementation) idiom to hide internal details
 * and reduce compilation dependencies. It offers a templated interface to add a grid (either
 * scalar or vector) from raw memory, and then write all collected grids to disk.
 *
 * Typical usage:
 * @code
 *   VDBWriter writer;
 *   // Suppose you have some float* data pointer with a known size:
 *   writer.addGrid<float, 1>("density", data, sizeX, sizeY, sizeZ);
 *   writer.write("output.vdb");
 * @endcode
 */
class VDBWriter {
private:
    /**
     * @struct Impl
     * @brief Forward declaration of the private implementation (PIMPL).
     *
     * This structure is defined in the .cpp file to hide OpenVDB-specific details
     * and other internal data members from users of VDBWriter.
     */
    struct Impl;

    /**
     * @var impl_
     * @brief The unique pointer to the private implementation struct.
     *
     * Ensures that all implementation details are hidden and automatically
     * cleaned up when VDBWriter goes out of scope.
     */
    const std::unique_ptr<Impl> impl_;

    /**
     * @struct AddGridImpl
     * @tparam T The base numeric type for voxel values (e.g., float).
     * @tparam N Number of components per voxel (e.g., 1 for scalar, 3 for vector).
     * @brief A small functor that handles the creation of a Dense representation
     *        from raw data and the subsequent conversion to an OpenVDB grid.
     *
     * This struct is constructed with all necessary parameters for converting
     * raw memory data into a fully populated OpenVDB grid, then pushed into the
     * writer's internal grid list.
     */
    template<typename T, size_t N>
    struct AddGridImpl {
        /// Pointer back to the containing writer so we can access impl_.
        VDBWriter *that;
        /// Name of the grid to be stored as metadata.
        std::string name;
        /// Pointer to the raw data (float, float3, etc.).
        const void *base;

        /// Size (width) in the x dimension.
        uint32_t sizeX;
        /// Size (height) in the y dimension.
        uint32_t sizeY;
        /// Size (depth) in the z dimension.
        uint32_t sizeZ;

        /// The minimum X coordinate in the OpenVDB index space.
        int32_t minX;
        /// The minimum Y coordinate in the OpenVDB index space.
        int32_t minY;
        /// The minimum Z coordinate in the OpenVDB index space.
        int32_t minZ;

        /// Byte distance to move one step in the X dimension in the raw data.
        uint32_t pitchX;
        /// Byte distance to move one step in the Y dimension in the raw data.
        uint32_t pitchY;
        /// Byte distance to move one step in the Z dimension in the raw data.
        uint32_t pitchZ;

        /**
         * @brief The call operator that performs the main data loading into a Dense buffer
         *        and then converts that dense representation into a sparse OpenVDB grid.
         *
         * @details Iterates over all voxels in the specified size, calculates the correct
         *          memory location for each voxel in `base`, and assigns it in the Dense container.
         *          Finally, it copies that data into an OpenVDB grid and attaches metadata.
         */
        void operator()() const;
    };

public:
    /**
     * @brief Default constructor that sets up the internal private implementation.
     */
    VDBWriter();
    /**
     * @brief Destructor that performs any necessary cleanup of internal data.
     */
    ~VDBWriter();

    // Delete copy and move semantics to ensure unique ownership of the internal pointer.
    VDBWriter(const VDBWriter &) = delete;
    VDBWriter &operator=(const VDBWriter &) = delete;
    VDBWriter(VDBWriter &&) = delete;
    VDBWriter &operator=(VDBWriter &&) = delete;

    /**
     * @brief Add a new grid (scalar or vector) from raw data to the writer.
     *
     * @tparam T             Numeric type for voxels (e.g., float).
     * @tparam N             Number of components per voxel (1 for scalar, 3 for vector).
     * @tparam normalizedCoords If true, shifts the grid origin so it is centered around (0, 0, 0).
     * @param name           Name of the grid, stored as metadata.
     * @param base           Pointer to the raw 3D data.
     * @param sizeX          X dimension size.
     * @param sizeY          Y dimension size.
     * @param sizeZ          Z dimension size.
     * @param pitchX         Byte stride between adjacent X elements; 0 = auto-calc.
     * @param pitchY         Byte stride between adjacent Y rows; 0 = auto-calc.
     * @param pitchZ         Byte stride between adjacent Z slices; 0 = auto-calc.
     *
     * @note By default, if pitch parameters are zero, they are computed from `sizeof(T)*N`
     *       and the dimensions. This function immediately constructs a Dense representation,
     *       which is then converted to a sparse OpenVDB grid and stored in the writer.
     */
    template<typename T, size_t N, bool normalizedCoords = true>
    void addGrid(const std::string &name, const void *base, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ,
                 uint32_t pitchX = 0, uint32_t pitchY = 0, uint32_t pitchZ = 0) {
        // If pitchX, pitchY, pitchZ are not provided, automatically calculate them based on
        // the size of the data type and the number of components per voxel.
        if (pitchX == 0) {
            pitchX = sizeof(T) * N; ///< pitchX = bytes per single voxel across X
        }
        if (pitchY == 0) {
            pitchY = pitchX * sizeX; ///< pitchY = stride across one full row
        }
        if (pitchZ == 0) {
            pitchZ = pitchY * sizeY; ///< pitchZ = stride across one full 2D slice
        }

        // If normalizedCoords is true, shift the grid's coordinate origin so that the
        // grid is centered around (0,0,0) in each dimension.
        int32_t minX = normalizedCoords ? -(int32_t) sizeX / 2 : 0;
        int32_t minY = normalizedCoords ? -(int32_t) sizeY / 2 : 0;
        int32_t minZ = normalizedCoords ? -(int32_t) sizeZ / 2 : 0;

        // Create the AddGridImpl struct and invoke its operator() to build and store the grid.
        AddGridImpl<T, N>{this, name, base, sizeX, sizeY, sizeZ, minX, minY, minZ, pitchX, pitchY, pitchZ}();
    }

    /**
     * @brief Writes all accumulated grids to the specified .vdb file.
     *
     * @param path The filesystem path to the output .vdb file.
     *
     * @throws Throws if there is any issue writing to disk, e.g. permissions or I/O errors.
     */
    void write(const std::string &path);
};

#ifdef VDB_WRITER_IMPLEMENTATION

// OpenVDB headers: required for creating, manipulating, and writing VDB grids
#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>

/**
 * @struct VDBWriter::Impl
 * @brief Internal (hidden) implementation struct that stores the OpenVDB grids.
 *
 * This struct is defined here in the .hpp file under the `VDB_WRITER_IMPLEMENTATION`
 * guard so it is only visible to the translation unit that implements VDBWriter.
 */
struct VDBWriter::Impl {
    /**
     * @var grids
     * @brief Vector of OpenVDB grid pointers that accumulate as the user adds new data.
     *
     * When write() is called, all of these grids get serialized into the .vdb file.
     */
    openvdb::GridPtrVec grids;
};

//=============================================================================
// VDBWriter inline definitions (enabled by VDB_WRITER_IMPLEMENTATION guard)
//=============================================================================

/**
 * @brief Constructs the VDBWriter by allocating its hidden implementation struct.
 */
VDBWriter::VDBWriter() : impl_(std::make_unique<Impl>()) {}

/**
 * @brief Destructor. Defaulted because unique_ptr automatically cleans up Impl.
 */
VDBWriter::~VDBWriter() = default;

/**
 * @brief Writes all stored grids to a specified .vdb file.
 *
 * @param path Filesystem path to output file.
 *
 * Creates an OpenVDB file object, hands it the vector of grids, and calls write().
 * If there are no grids in impl_->grids, a file with no grids is written.
 */
void VDBWriter::write(const std::string &path) {
    // Create a File object targeting the specified path
    // and write out the entire grid vector in a single operation.
    openvdb::io::File(path).write(impl_->grids);
}

namespace {
    /**
     * @struct vdbtraits
     * @brief Template struct that maps (T, N) to the corresponding OpenVDB grid type.
     *
     * The default template is empty. We rely on partial specializations for actual usage.
     * For example, <float,1> maps to FloatGrid, <float,3> maps to Vec3fGrid.
     */
    template<typename T, size_t N>
    struct vdbtraits {};

    /**
     * @brief Specialization for scalar float data.
     */
    template<>
    struct vdbtraits<float, 1> {
        using type = openvdb::FloatGrid; ///< One float per voxel
    };

    /**
     * @brief Specialization for 3-component float data, e.g., velocity vectors.
     */
    template<>
    struct vdbtraits<float, 3> {
        using type = openvdb::Vec3fGrid; ///< Three floats per voxel
    };

    /**
     * @brief Helper function to construct an OpenVDB vector type from an array of T.
     *
     * @tparam VecT The resulting vector type (e.g., openvdb::Vec3f).
     * @tparam T    The numeric component type (float).
     * @tparam Is   The index pack for expansion via std::index_sequence.
     * @param ptr   Pointer to at least N contiguous elements of type T.
     *
     * @return A VecT constructed from [ptr[Is]...].
     */
    template<typename VecT, typename T, size_t... Is>
    VecT help_make_vec(const T *ptr, std::index_sequence<Is...>) {
        // For N=3, expands to: VecT(ptr[0], ptr[1], ptr[2])
        // For N=1, expands to: VecT(ptr[0])
        return VecT(ptr[Is]...);
    }
} // namespace

/**
 * @brief Templated call operator for AddGridImpl that creates a Dense, populates it from raw data,
 *        and transfers it into an OpenVDB sparse grid.
 */
template<typename T, size_t N>
void VDBWriter::AddGridImpl<T, N>::operator()() const {
    // Based on <T,N>, pick an appropriate OpenVDB grid type (FloatGrid or Vec3fGrid).
    using GridT = typename vdbtraits<T, N>::type;

    // Create a Dense container, specifying dimensions and the minimum coordinate.
    // sizeX/sizeY/sizeZ define the number of voxels along each axis,
    // while minX/minY/minZ define how the grid is positioned in index space.
    openvdb::tools::Dense<typename GridT::ValueType> dens(openvdb::Coord(sizeX, sizeY, sizeZ),
                                                          openvdb::Coord(minX, minY, minZ));

    // Triple nested loop to walk through the entire 3D volume.
    // For each (x, y, z) coordinate, compute the proper memory location in 'base'
    // using pitchX, pitchY, and pitchZ. Then set that value in the Dense container.
    for (uint32_t z = 0; z < sizeZ; ++z) {
        for (uint32_t y = 0; y < sizeY; ++y) {
            for (uint32_t x = 0; x < sizeX; ++x) {
                // Calculate the pointer to the voxel data at (x, y, z).
                auto ptr = reinterpret_cast<T const *>(reinterpret_cast<char const *>(base) + pitchX * x + pitchY * y +
                                                       pitchZ * z);
                // Transform the raw data pointer (which may hold 1 or 3 floats)
                // into the proper OpenVDB value type, e.g., float or Vec3f.
                dens.setValue(x, y, z, help_make_vec<typename GridT::ValueType>(ptr, std::make_index_sequence<N>{}));
            }
        }
    }

    // Create an empty OpenVDB grid of the appropriate type (FloatGrid or Vec3fGrid).
    auto grid = GridT::create();

    // Tolerance can define how closely to match the dense data.
    // For float data, 0 means exact copy.
    typename GridT::ValueType tolerance{0};

    // Copy from the Dense representation into the sparse VDB tree.
    // This is typically far more memory-efficient if the data has many uniform or zero regions.
    openvdb::tools::copyFromDense(dens, grid->tree(), tolerance);

    // Attach metadata with the grid name.
    openvdb::MetaMap &meta = *grid;
    meta.insertMeta(openvdb::Name("name"), openvdb::TypedMetadata<std::string>(name));

    // Push the newly created grid into the writer's internal vector of grids.
    that->impl_->grids.push_back(grid);
}

// Explicit template instantiations for the combinations we expect.
template struct VDBWriter::AddGridImpl<float, 1>;
template struct VDBWriter::AddGridImpl<float, 3>;

#endif

#endif // VDBWRITER_HPP
