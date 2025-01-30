// Copyright (c) 2025. Shuojiang Liu.

#ifndef CUDAARRAY_CUH
#define CUDAARRAY_CUH

#include <cuda_runtime.h>
#include <utils/HelperCuda.cuh>


/**
 * \class CopyDisabler
 * \brief A small utility class used to disable the copy constructor and copy assignment operator.
 *
 * I use a common C++ idiom to prevent accidental copying of classes
 * that manage resources, ensuring the resource is not freed twice.
 */
class CopyDisabler {
public:
    CopyDisabler() = default;
    CopyDisabler(const CopyDisabler &) = delete;
    CopyDisabler &operator=(const CopyDisabler &) = delete;
};

/**
 * \class CudaArray
 * \brief Manages a CUDA 3D array (cudaArray) resource.
 *
 * This class follows the RAII principle: it allocates a cudaArray upon construction,
 * and frees it upon destruction. The @c copyIn and @c copyOut methods allow me to
 * transfer data from host to device and from device to host, respectively.
 *
 * \tparam T The data type stored in the 3D array (e.g., float, int, float4, etc.)
 */
template<typename T>
class CudaArray : public CopyDisabler {
public:
    /**
     * \brief Constructor that allocates a 3D cudaArray resource.
     *
     * \param otherDim The 3D dimensions of the array, stored as (x, y, z).
     */
    explicit CudaArray(const uint3 &otherDim) : m_dim(otherDim) {
        // Create a cudaExtent with the same dimensions as m_dim.
        const cudaExtent extent = make_cudaExtent(m_dim.x, m_dim.y, m_dim.z);
        // Create a channel descriptor based on the type T.
        const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();

        // Allocate a 3D CUDA array with surface load/store capabilities.
        checkCudaErrors(cudaMalloc3DArray(&m_cudaArray, &channelDesc, extent, cudaArraySurfaceLoadStore));
    }

    /**
     * \brief Copies data from host memory into the allocated cudaArray on the device.
     *
     * \param data Pointer to the host data (must have at least @c m_dim.x * m_dim.y * m_dim.z elements).
     */
    void copyIn(const T *data) {
        cudaMemcpy3DParms copy3DParams{};

        copy3DParams.srcPtr = make_cudaPitchedPtr(const_cast<void *>(reinterpret_cast<const void *>(data)),
                                                  m_dim.x * sizeof(T), m_dim.x, m_dim.y);
        copy3DParams.dstArray = m_cudaArray;
        copy3DParams.extent = make_cudaExtent(m_dim.x, m_dim.y, m_dim.z);
        copy3DParams.kind = cudaMemcpyHostToDevice;

        checkCudaErrors(cudaMemcpy3D(&copy3DParams));
    }

    /**
     * \brief Copies data from the allocated cudaArray on the device back to host memory.
     *
     * \param data Pointer to the destination host buffer (must have enough space).
     */
    void copyOut(T *data) {
        cudaMemcpy3DParms copy3DParams{};

        copy3DParams.srcArray = m_cudaArray;
        copy3DParams.dstPtr = make_cudaPitchedPtr(const_cast<void *>(reinterpret_cast<const void *>(data)),
                                                  m_dim.x * sizeof(T), m_dim.x, m_dim.y);
        copy3DParams.extent = make_cudaExtent(m_dim.x, m_dim.y, m_dim.z);
        copy3DParams.kind = cudaMemcpyDeviceToHost;

        checkCudaErrors(cudaMemcpy3D(&copy3DParams));
    }

    /**
     * \brief Returns the underlying CUDA array handle.
     *
     * \return A pointer to the @c cudaArray used internally.
     */
    cudaArray *getArray() const { return m_cudaArray; }

    /**
     * @brief Destructor that frees the allocated cudaArray resource.
     */
    ~CudaArray() { checkCudaErrors(cudaFreeArray(m_cudaArray)); }

public:
    /**
     * \brief The raw pointer to the CUDA array resource.
     *
     * cudaArray is a special memory layout managed by the CUDA driver, used
     * for texture or surface references.
     */
    cudaArray *m_cudaArray{};

    /**
     * \brief Dimensions of the array (x, y, z).
     *
     * These dimensions are in terms of the number of elements in each dimension.
     */
    uint3 m_dim{};
};

/**
 * \class CudaSurfaceAccessor
 * \brief An accessor object for reading/writing from a cudaSurfaceObject_t in device code.
 *
 * This class encapsulates the logic for performing 3D reads and writes (surf3Dread, surf3Dwrite)
 * on a CUDA surface. It is meant to be used inside CUDA calculation or device functions.
 *
 * \tparam T The data type (float, int, etc.) associated with the surface.
 */
template<typename T>
class CudaSurfaceAccessor {
public:
    /**
     * \brief Read a value of type T from the 3D surface at (x, y, z).
     *
     * \tparam mode The boundary mode used for reads outside the valid range. Defaults to cudaBoundaryModeTrap.
     * \param x The X coordinate in element index.
     * \param y The Y coordinate in element index.
     * \param z The Z coordinate in element index.
     * \return The value of type T read from the surface.
     */
    template<cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap>
    __device__ __forceinline__ T read(const int x, const int y, const int z) const {
        return surf3Dread<T>(m_cudaSurface, x * sizeof(T), y, z, mode);
    }

    /**
     * \brief Write a value of type T into the 3D surface at (x, y, z).
     *
     * \tparam mode The boundary mode used for writes outside the valid range. Defaults to cudaBoundaryModeTrap.
     * \param val The value to be written.
     * \param x The X coordinate in element index.
     * \param y The Y coordinate in element index.
     * \param z The Z coordinate in element index.
     */
    template<cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap>
    __device__ __forceinline__ void write(T val, const int x, const int y, const int z) const {
        surf3Dwrite<T>(val, m_cudaSurface, x * sizeof(T), y, z, mode);
    }

public:
    /**
     * \brief The CUDA surface object handle.
     *
     * This is created from a cudaArray resource. The surface object
     * allows kernel-side read and write operations.
     */
    cudaSurfaceObject_t m_cudaSurface;
};

/**
 * \class CudaSurface
 * \brief A wrapper around a 3D cudaArray that also creates a cudaSurfaceObject_t.
 *
 * Inherits from @c CudaArray<T> and manages creation/destruction of the surface object.
 * The underlying cudaArray resource is used to back the surface object.
 *
 * \tparam T The data type (float, int, etc.) of the surface.
 */
template<typename T>
class CudaSurface : public CudaArray<T> {
public:
    /**
     * \brief Constructor that allocates the cudaArray (via @c CudaArray<T>)
     *        and then creates a surface object from it.
     *
     * \param otherDim The 3D dimensions (x, y, z) to allocate and use.
     */
    explicit CudaSurface(const uint3 &otherDim) : CudaArray<T>(otherDim) {
        cudaResourceDesc resourceDesc{};

        resourceDesc.resType = cudaResourceTypeArray;
        // Link the resource descriptor to the underlying CUDA array
        resourceDesc.res.array.array = CudaArray<T>::getArray();

        // Create the surface object
        checkCudaErrors(cudaCreateSurfaceObject(&m_cudaSurface, &resourceDesc));
    }

    /**
     * \brief Returns the CUDA surface object handle.
     */
    cudaSurfaceObject_t getSurface() const { return m_cudaSurface; }

    /**
     * \brief Creates and returns an accessor object that can be used in device code
     *        for 3D reads/writes (surf3Dread, surf3Dwrite).
     */
    CudaSurfaceAccessor<T> accessSurface() const { return {m_cudaSurface}; }

    /**
     * \brief Destructor that destroys the surface object.
     *        The base class destructor frees the cudaArray itself.
     */
    ~CudaSurface() { checkCudaErrors(cudaDestroySurfaceObject(m_cudaSurface)); }

public:
    /**
     * \brief Handle to the CUDA surface object (cudaSurfaceObject_t).
     */
    cudaSurfaceObject_t m_cudaSurface{};
};

/**
 * \class CudaTextureAccessor
 * \brief An accessor object for sampling from a cudaTextureObject_t in device code.
 *
 * Similar to CudaSurfaceAccessor but specialized for tex3D sampling.
 *
 * \tparam T The data type to be sampled from the texture.
 */
template<typename T>
class CudaTextureAccessor {
public:
    /**
     * \brief Sample the texture at (x, y, z) in normalized or unnormalized coordinates (depends on texture descriptor).
     *
     * \param x The X coordinate (in floating-point, can be normalized or not).
     * \param y The Y coordinate (in floating-point).
     * \param z The Z coordinate (in floating-point).
     * \return The value of type T read from the 3D texture.
     */
    __device__ __forceinline__ T sample(const float x, const float y, const float z) const {
        return tex3D<T>(m_cudaTexture, x, y, z);
    }

public:
    /**
     * \brief Handle to the texture object.
     */
    cudaTextureObject_t m_cudaTexture;
};

/**
 * \class CudaTexture
 * \brief A wrapper around a 3D cudaArray that also creates a cudaSurfaceObject_t (via @c CudaSurface<T>)
 *        and a cudaTextureObject_t for sampling.
 *
 * We can copy data into it with @c copyIn (inherited from @c CudaArray<T>),
 * and then sample from it in device code using the @c CudaTextureAccessor.
 *
 * \tparam T The data type (float, int, etc.) of the texture elements.
 */
template<typename T>
class CudaTexture : public CudaSurface<T> {
public:
    /**
     * \brief Parameters that control how the texture is addressed, filtered, and read.
     */
    struct Parameters {
        cudaTextureAddressMode addressMode{cudaAddressModeBorder}; ///< How out-of-bounds lookups are handled.
        cudaTextureFilterMode filterMode{cudaFilterModeLinear}; ///< How data is filtered (nearest, linear).
        cudaTextureReadMode readMode{cudaReadModeElementType}; ///< Read mode (element type or normalized float).
        bool normalizedCoords{false}; ///< If true, (0,1) range used for coordinates.
    };

    /**
     * \brief Constructor that allocates the array (via @c CudaSurface<T>) and creates a texture object from it.
     *
     * \param otherDim The dimensions of the underlying array (x, y, z).
     * \param otherArgs A structure of texture parameters controlling addressing, filtering, etc.
     */
    explicit CudaTexture(const uint3 &otherDim, const Parameters &otherArgs = {}) : CudaSurface<T>(otherDim) {
        // Resource descriptor: reference the same cudaArray from the base CudaSurface
        cudaResourceDesc resourceDesc{};
        resourceDesc.resType = cudaResourceTypeArray;
        resourceDesc.res.array.array = CudaSurface<T>::getArray();

        // Texture descriptor: specify how the texture is addressed and filtered
        cudaTextureDesc textureDesc{};
        textureDesc.addressMode[0] = otherArgs.addressMode;
        textureDesc.addressMode[1] = otherArgs.addressMode;
        textureDesc.addressMode[2] = otherArgs.addressMode;
        textureDesc.filterMode = otherArgs.filterMode;
        textureDesc.readMode = otherArgs.readMode;
        textureDesc.normalizedCoords = otherArgs.normalizedCoords;

        checkCudaErrors(cudaCreateTextureObject(&m_cudaTexture, &resourceDesc, &textureDesc, NULL));
    }

    /**
     * \brief Retrieve the CUDA texture object handle.
     *
     * \return A @c cudaTextureObject_t handle that can be sampled in device code.
     */
    cudaTextureObject_t getTexture() const { return m_cudaTexture; }

    /**
     * \brief Return an accessor for sampling from the texture in CUDA device code.
     *
     * \return A @c CudaTextureAccessor<T> object that wraps tex3D calls.
     */
    CudaTextureAccessor<T> accessTexture() const { return {m_cudaTexture}; }

    /**
     * \brief Destructor that destroys the texture object. The base class destructor frees
     *        the surface object and underlying cudaArray.
     */
    ~CudaTexture() { checkCudaErrors(cudaDestroyTextureObject(m_cudaTexture)); }

public:
    /**
     * \brief The handle to the CUDA texture object.
     */
    cudaTextureObject_t m_cudaTexture{};
};


#endif // CUDAARRAY_CUH
