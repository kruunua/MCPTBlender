Project: MCPTBlender

Description:
-------------
MCPTBlender is a C++ application that implements progressive converging algorithm-agnostic image denoising based on curve prediction algorithm: https://doi.org/10.1145/3675384

Requirements:
-------------
- CMake version 3.10 or higher
- C++11 compatible compiler
- External libraries:
  - Intel Open Image Denoise (OIDN) 2.0.1
  - OpenEXR 3.2.0
  - NVIDIA OptiX SDK 8.0.0
  - NVIDIA CUDA Toolkit 12.1

Build Instructions:
-------------------
1. Make sure you have CMake installed on your system.
2. Clone the repository or download the source files.
3. Create a build directory and navigate into it:

mkdir build
cd build

4. Configure the project using CMake:

cmake ..

Optionally, you can specify the generator for your build system. For example, for Visual Studio:

cmake .. -G "Visual Studio 16 2019" -DCMAKE_BUILD_TYPE=Debug

Replace `"Visual Studio 16 2019"` with your specific generator. Replace `Debug` with `Release` for building the release version.

5. Build the project:

cmake --build . --config Debug

Replace `Debug` with `Release` for building the release version.

6. Install the project:

cmake --build . --target install

This will install the necessary DLLs and executables to the configured installation directory.

Usage:
------
- After building and installing the project, navigate to the installation directory specified during installation (default: C:/path/to/installation/directory).
- Run the executable `MCPTBlender` from the command line or using your preferred IDE.

Directories:
------------
- `src/`: Contains the source files for the project.
- `include/`: Contains the header files for the project.
- `build/`: Directory where CMake builds the project.
- `oidn-2.0.1/`, `openexr-3.2.0/`: External library directories.
- `C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0/`, `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/`: SDK directories.

External Libraries:
--------------------
- By default, the project looks for OpenEXR and OIDN libraries in the following directories relative to the project root:
- `oidn-2.0.1/`
- `openexr-3.2.0/`

If you have these libraries installed elsewhere, you can modify the paths in `CMakeLists.txt` accordingly.

Additional Notes:
-----------------
- Make sure to have the required DLLs (`*.dll`) accessible in your system's PATH or in the same directory as the executable.

Contact:
--------
For issues or inquiries, please contact denisova.lena@gmail.com.
