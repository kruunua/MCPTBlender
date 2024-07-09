cmake_minimum_required(VERSION 3.10)

project(MCPTBlender)

set(CMAKE_CXX_STANDARD 11)

# Add the source files
set(SOURCES
    src/curvepredictor.cpp
    src/imagedenoiser.cpp
    src/imageloader.cpp
    src/main.cpp
)

set(HEADERS
    include/curvepredictor.h
    include/imagedenoiser.h
    include/imageloader.h
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Include directories
set(OIDN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/oidn-2.0.1")
set(OPENEXR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/openexr-3.2.0")
set(OPTIX_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0/")
set(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/")

include_directories(
    include
    ${OIDN_DIR}/include
    ${OPENEXR_DIR}/include/OpenEXR
    ${OPENEXR_DIR}/include/Imath
    ${OPTIX_DIR}/include
    ${CUDA_DIR}/include
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OIDN_DIR}/lib/OpenImageDenoise.lib
    ${OPENEXR_DIR}/lib/OpenEXR-3_2.lib
    ${OPENEXR_DIR}/lib/Imath-3_2.lib
    ${OPENEXR_DIR}/lib/Iex-3_2.lib
    advapi32
    ${CUDA_DIR}/lib/x64/cuda.lib
    ${CUDA_DIR}/lib/x64/cudart.lib
)

# Add defines
add_definitions(-DQT_DEPRECATED_WARNINGS)

# Define installation directories
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(INSTALL_DIR "${CMAKE_BINARY_DIR}/debug")
else()
    set(INSTALL_DIR "${CMAKE_BINARY_DIR}/release")
endif()

# Set installation directory
set(CMAKE_INSTALL_PREFIX "${INSTALL_DIR}" CACHE PATH "Installation directory" FORCE)

# Installation of DLLs
install(FILES ${OIDN_DIR}/bin/OpenImageDenoise.dll
        ${OIDN_DIR}/bin/OpenImageDenoise_core.dll
        ${OIDN_DIR}/bin/OpenImageDenoise_device_cpu.dll
        ${OIDN_DIR}/bin/OpenImageDenoise_device_cuda.dll
        DESTINATION ${CMAKE_INSTALL_PREFIX})

install(FILES ${OPENEXR_DIR}/bin/OpenEXR-3_2.dll
        ${OPENEXR_DIR}/bin/OpenEXRCore-3_2.dll
        ${OPENEXR_DIR}/bin/Imath-3_2.dll
        ${OPENEXR_DIR}/bin/Iex-3_2.dll
        ${OPENEXR_DIR}/bin/IlmThread-3_2.dll
        DESTINATION ${CMAKE_INSTALL_PREFIX})

install(FILES ${CUDA_DIR}/bin/cudart64_12.dll
        DESTINATION ${CMAKE_INSTALL_PREFIX})

# Configure debug and release output directories
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/debug
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/release
)
