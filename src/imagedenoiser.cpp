/**
 * @file imagedenoiser.cpp
 * @author E. Denisova
 * @date 29/2/2024
 * @version 1.0
**/

#include "imagedenoiser.h"
#include <stdexcept>
#include <iostream>

#include <OpenImageDenoise/oidn.hpp>
#include <optix.h>
#include <optix_denoiser_tiling.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <optix_function_table_definition.h>

struct OidnData
{
    oidn::DeviceRef device = nullptr;
    oidn::BufferRef colorBuf = nullptr;
    oidn::BufferRef albedoBuf = nullptr;
    oidn::BufferRef normalBuf = nullptr;
    oidn::BufferRef outputBuf = nullptr;
    oidn::FilterRef filter = nullptr;
    oidn::FilterRef albedoFilter = nullptr;
    oidn::FilterRef normalFilter = nullptr;
};

struct OptiXData
{
    OptixDeviceContext context = nullptr;
    OptixDenoiser denoiser = nullptr;
    CUcontext cuCtx = nullptr;  // use current context
    OptixDenoiserLayer layer = {};
    OptixDenoiserGuideLayer guideLayer = {};
    CUdeviceptr scratch = 0;
    size_t scratchSize  = 0;
    CUdeviceptr state   = 0;
    size_t stateSize    = 0;
    size_t overlap      = 0;
};

// Initialize static member instance
ImageDenoiser *ImageDenoiser::m_instance = nullptr;

// Constructor
ImageDenoiser::ImageDenoiser()
{
    memset(m_cpuData, 0, 3 * sizeof(void *));
    memset(m_gpuData, 0, 3 * sizeof(void *));
    memset(m_optiXData, 0, 3 * sizeof(void *));
}

// Destructor or cleanup function
void ImageDenoiser::release()
{
    // Cleanup resources
    for(int i = 0; i < 3; i++)
    {
        OptiXData *data = static_cast<OptiXData *>(m_optiXData[i]);
        if(!data)
            continue;

        optixDenoiserDestroy(data->denoiser);
        optixDeviceContextDestroy(data->context);

        cudaFree(reinterpret_cast<void *>(data->scratch));
        cudaFree(reinterpret_cast<void *>(data->state));
        cudaFree(reinterpret_cast<void *>(data->guideLayer.albedo.data));
        cudaFree(reinterpret_cast<void *>(data->guideLayer.normal.data));
        cudaFree(reinterpret_cast<void *>(data->layer.input.data));
        cudaFree(reinterpret_cast<void *>(data->layer.output.data));

        delete data;
        m_optiXData[i] = nullptr;
    }
}

// Singleton instance creation
ImageDenoiser *ImageDenoiser::instance()
{
    if(!m_instance)
        m_instance = new ImageDenoiser();

    return m_instance;
}

// Initialization function
bool ImageDenoiser::init()
{
    for(int i = 0; i < 3; i++)
    {
        if(m_gpuData[i] || m_cpuData[i])
            return m_gpuData[0] || m_cpuData[0];
    }
    for(int k = 0; k < 2; k++)
    {
        OidnData *data = new OidnData();
        for(int i = 0; i < 3; i++)
        {
            data->device = oidn::newDevice(k == 0
                                           ? oidn::DeviceType::CUDA
                                           : oidn::DeviceType::CPU); // CPU or GPU if available
            const char *errorMessage;
            if(data->device.getError(errorMessage) != oidn::Error::None)
            {
                std::cerr << "Denoiser:" << errorMessage << std::endl;
                delete data;
                continue;
            }
            data->device.commit();
            if(k == 0)
                m_gpuData[i] = data;
            else
                m_cpuData[i] = data;
        }
    }
    bool ok = _createOptiXContext();
    if(ok)
    {
        for(int i = 0; i < 3; i++)
            _createOptiXDenoiser(i);
    }
    return m_gpuData[0] || m_cpuData[0] || ok;
}

// Run function
bool ImageDenoiser::run(const std::vector<float> &input, int w, int h,
                        std::vector<float> &output, bool optiX, bool hdr,
                        bool cleanAux, bool cpu) const
{
    if(optiX)
        return _runOptiX(input, w, h, output, hdr);

    bool useAlb = input.size() >= 6 * size_t(w * h);
    bool useNor = input.size() == 9 * size_t(w * h);
    OidnData *data = static_cast<OidnData *>(cpu ? m_cpuData[useAlb + useNor]
                                                 : m_gpuData[useAlb + useNor]);
    if(!data)
        return false;

    size_t sz = size_t(w * h * 3) * sizeof(float);
    if(!data->colorBuf || data->colorBuf.getSize() != sz)
    {
        data->colorBuf = data->device.newBuffer(sz);
        data->outputBuf = data->device.newBuffer(sz);
        if(useAlb)
        {
            data->albedoBuf = data->device.newBuffer(sz);
            if(useNor)
                data->normalBuf = data->device.newBuffer(sz);
        }
        data->filter = data->device.newFilter("RT"); // generic ray tracing filter
        data->filter.set("quality", OIDN_QUALITY_HIGH);
        data->filter.setImage("color", data->colorBuf, oidn::Format::Float3,
                              size_t(w), size_t(h)); // beauty
        if(useAlb)
        {
            data->filter.setImage("albedo", data->albedoBuf,
                                  oidn::Format::Float3, size_t(w), size_t(h));
            if(useNor)
                data->filter.setImage("normal", data->normalBuf,
                                      oidn::Format::Float3,
                                      size_t(w), size_t(h));
            data->filter.set("cleanAux", cleanAux);
        }
        data->filter.setImage("output", data->outputBuf, oidn::Format::Float3,
                              size_t(w), size_t(h)); // denoised beauty
        data->filter.set("hdr", hdr);
        data->filter.commit();
        if(useAlb && !cleanAux)
        {
            data->albedoFilter = data->device.newFilter("RT");
            data->albedoFilter.setImage("albedo", data->albedoBuf,
                                        oidn::Format::Float3,
                                        size_t(w), size_t(h));
            data->albedoFilter.setImage("output", data->albedoBuf,
                                        oidn::Format::Float3,
                                        size_t(w), size_t(h));
            data->albedoFilter.commit();
            if(useNor)
            {
                data->normalFilter = data->device.newFilter("RT");
                data->normalFilter.setImage("normal", data->normalBuf,
                                            oidn::Format::Float3,
                                            size_t(w), size_t(h));
                data->normalFilter.setImage("output", data->normalBuf,
                                            oidn::Format::Float3,
                                            size_t(w), size_t(h));
                data->normalFilter.commit();
            }
        }
    }
    if(data->filter.get<bool>("hdr") != hdr)
    {
//        std::cerr << "Change HDR to" << hdr << std::endl;
        data->filter.set("hdr", hdr);
        data->filter.commit();
    }
    const float *inputPtr = input.data();
    data->colorBuf.write(0, sz, inputPtr);
    if(useAlb)
    {
        data->albedoBuf.write(0, sz, inputPtr + w * h * 3);
        if(useNor)
            data->normalBuf.write(0, sz, inputPtr + w * h * 6);
    }
    if(useAlb && !cleanAux)
    {
        data->albedoFilter.execute();
        if(useNor)
            data->normalFilter.execute();
    }
    // Filter the beauty image
    data->filter.execute();

    // Check for errors
    const char *errorMessage;
    if(data->device.getError(errorMessage) != oidn::Error::None)
    {
        std::cerr << "Denoiser:" << errorMessage << std::endl;
        return false;
    }
//    std::cout << "Denoiser: done in" << e.elapsed() << "ms" << std::endl;
    output.resize(size_t(w * h) * 3);
    data->outputBuf.read(0, sz, output.data());
    return true;
}

// Error checking macro for CUDA
#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    const cudaError_t error = call;                                        \
    if(error != cudaSuccess)                                               \
    {                                                                      \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", "      \
                  << cudaGetErrorString(error) << std::endl;               \
        exit(1);                                                           \
    }                                                                      \
} while(0)

// Error checking macro for OptiX
#define OPTIX_CHECK(call)                                                  \
do {                                                                       \
    const OptixResult result = call;                                       \
    if(result != OPTIX_SUCCESS)                                            \
    {                                                                      \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", "      \
                  << optixGetErrorString(result) << std::endl;             \
        exit(1);                                                           \
    }                                                                      \
} while(0)

#define UNUSED(x) (void)(x)

namespace {
// OptiX log callback function
void _OptixLogCallback(unsigned int level, const char *tag, const char *message,
                       void *)
{
    UNUSED(level);
    UNUSED(tag);
    UNUSED(message);
//    std::cout << "[OptiX] " << tag << " (" << level << "): "
//              << message << std::endl;
}

// Helper functions for OptiX image handling
OptixImage2D CreateOptixImage2D(int w, int h)
{
    OptixImage2D oi;
    const size_t frameByteSize = size_t(w * h) * sizeof(float3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&oi.data), frameByteSize));
    oi.width              = uint32_t(w);
    oi.height             = uint32_t(h);
    oi.rowStrideInBytes   = uint32_t(w) * sizeof(float3);
    oi.pixelStrideInBytes = sizeof(float3);
    oi.format             = OPTIX_PIXEL_FORMAT_FLOAT3;
    return oi;
}

void CopyFromImage2D(int w, int h, const float *hmem, OptixImage2D &oi)
{
    const size_t frameByteSize = size_t(w * h) * sizeof(float3);
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(oi.data), hmem,
                          frameByteSize, cudaMemcpyHostToDevice));
}

void CopyToImage2D(int w, int h, float *hmem, const OptixImage2D &oi)
{
    const size_t frameByteSize = size_t(w * h) * sizeof(float3);
    CUDA_CHECK(cudaMemcpy(hmem, reinterpret_cast<void *>(oi.data),
                          frameByteSize, cudaMemcpyDeviceToHost));
}

void CudaSyncCheck()
{
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

// OptiX denoising implementation
bool ImageDenoiser::_runOptiX(const std::vector<float> &input, int w, int h,
                              std::vector<float> &output, bool hdr) const
{
    if(!hdr)
        return false;

    int idx = (input.size() == size_t(w * h) * 3)
            ? 0 : (input.size() == size_t(w * h) * 6) ? 1 : 2;

    OptiXData *data = static_cast<OptiXData *>(m_optiXData[idx]);
    if(data->layer.input.data == 0)
    {
        OptixDenoiserSizes denoiserSizes;
        OPTIX_CHECK(optixDenoiserComputeMemoryResources(
                        data->denoiser, uint32_t(w), uint32_t(h),
                        &denoiserSizes));
        data->scratchSize = denoiserSizes.withOverlapScratchSizeInBytes;
        data->stateSize = denoiserSizes.stateSizeInBytes;
        data->overlap = denoiserSizes.overlapWindowSizeInPixels;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data->scratch),
                              data->scratchSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data->state),
                              data->stateSize));
        data->layer.input  = CreateOptixImage2D(w, h);
        data->layer.output = CreateOptixImage2D(w, h);
        if(idx > 0)
            data->guideLayer.albedo = CreateOptixImage2D(w, h);

        if(idx > 1)
            data->guideLayer.normal = CreateOptixImage2D(w, h);

        OPTIX_CHECK(optixDenoiserSetup(data->denoiser, nullptr,
                                       uint32_t(w) + 2 * uint32_t(data->overlap),
                                       uint32_t(h) + 2 * uint32_t(data->overlap),
                                       data->state,
                                       data->stateSize,
                                       data->scratch,
                                       data->scratchSize));
    }
    else if(data->layer.input.width != uint32_t(w)
            || data->layer.input.height != uint32_t(h))
        return false;

    OptixDenoiserParams params = {};
    params.hdrIntensity    = 0;
    params.hdrAverageColor = 0;
    params.blendFactor     = 0.0f;
    params.temporalModeUsePreviousLayers = 0;

    CopyFromImage2D(w, h, input.data(), data->layer.input);
    if(idx > 0)
        CopyFromImage2D(w, h, input.data() + 3 * w * h,
                        data->guideLayer.albedo);

    if(idx > 1)
        CopyFromImage2D(w, h, input.data() + 6 * w * h,
                        data->guideLayer.normal);

    OPTIX_CHECK(optixUtilDenoiserInvokeTiled(data->denoiser, nullptr,
                                             &params,
                                             data->state, data->stateSize,
                                             &data->guideLayer,
                                             &data->layer, 1,
                                             data->scratch, data->scratchSize,
                                             uint32_t(data->overlap),
                                             uint32_t(w), uint32_t(h)));
    CudaSyncCheck();
    output.resize(size_t(w * h) * 3);
    CopyToImage2D(w, h, output.data(), data->layer.output);
    return true;
}

// OptiX context creation
bool ImageDenoiser::_createOptiXContext()
{
    CUDA_CHECK(cudaFree(nullptr));
    OPTIX_CHECK(optixInit());

    for(int i = 0; i < 3; i++)
    {
        OptiXData *optiXData = new OptiXData();
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &_OptixLogCallback;
        options.logCallbackLevel = 4;
        OPTIX_CHECK(optixDeviceContextCreate(optiXData->cuCtx, &options,
                                             &optiXData->context));
        m_optiXData[i] = optiXData;
    }
    return true;
}

// OptiX denoiser creation
bool ImageDenoiser::_createOptiXDenoiser(int idx)
{
    OptixDenoiserModelKind kind = OPTIX_DENOISER_MODEL_KIND_HDR;
    OptixDenoiserOptions options = {};
    options.guideAlbedo = idx > 0 ? 1 : 0;
    options.guideNormal = idx > 1 ? 1 : 0;
    options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
    OptiXData *data = static_cast<OptiXData *>(m_optiXData[idx]);

    OPTIX_CHECK(optixDenoiserCreate(data->context, kind, &options,
                                    &data->denoiser));
    return true;
}
