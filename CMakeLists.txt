cmake_minimum_required(VERSION 3.10)

# Set CUDA architectures before project command
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)

project(MyProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# Find CUDA package
#find_package(CUDA 12.1 REQUIRED)
find_package(CUDA REQUIRED)


# Add -O3 optimization flag for g++
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Add -O3 optimization flag for nvcc
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

# Find MPI package
set(MPI_ROOT /root/miniconda3/envs/fed_env)
set(MPI_CXX_LIBRARIES -L/root/miniconda3/envs/fed_env/lib -lmpicxx -lmpi)
set(MPI_CXX_INCLUDE_PATH /root/miniconda3/envs/fed_env/include)


# Find OpenMP package
find_package(OpenMP REQUIRED)

# Include MPI headers
include_directories(${MPI_CXX_INCLUDE_PATH} src)

# Let CMake automatically detect CUDA architecture

# Set source files
set(SRC_FILES
    src/main.cu
    src/util.cpp
    src/lsh.cu
    src/param.cu
)

# Add PSI source files
set(PSI_FILES
    src/psi.cu
    src/util.cpp
    src/param.cu
)

# Add executable
add_executable(main ${SRC_FILES})

# Add PSI test executable
add_executable(psi_test src/psi_test.cu ${PSI_FILES})

# Link MPI and OpenMP
target_link_libraries(main ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX)
target_link_libraries(psi_test ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX)
