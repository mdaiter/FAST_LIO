cmake_minimum_required(VERSION 3.14)
project(fast_lio_tests CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread")

# ──────────────────────────────────────────────
# Google Test via FetchContent
# ──────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Google Benchmark
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

# ──────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────
find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED COMPONENTS common)

find_package(OpenMP QUIET)
if(OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Root of FAST-LIO source tree
set(FAST_LIO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Include paths (same as main project, minus ROS)
include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${FAST_LIO_ROOT}/include
  ${FAST_LIO_ROOT}/include/ikd-Tree
)

# Define ROOT_DIR so DEBUG_FILE_DIR macro works
add_definitions(-DROOT_DIR="${FAST_LIO_ROOT}/")

# ikd-Tree source (template instantiations)
set(IKD_TREE_SRC ${FAST_LIO_ROOT}/include/ikd-Tree/ikd_Tree.cpp)

# ──────────────────────────────────────────────
# Metal Shader Compilation (macOS only)
# ──────────────────────────────────────────────
set(HAS_METAL OFF)
if(APPLE)
  find_library(METAL_FRAMEWORK Metal)
  find_library(FOUNDATION_FRAMEWORK Foundation)
  if(METAL_FRAMEWORK)
    set(HAS_METAL ON)

    set(METAL_SHADER_SRC "${FAST_LIO_ROOT}/include/compute/metal/kernels.metal")
    set(METAL_AIR "${CMAKE_CURRENT_BINARY_DIR}/kernels.air")
    set(METAL_LIB "${CMAKE_CURRENT_BINARY_DIR}/kernels.metallib")

    # Compile .metal → .air → .metallib
    add_custom_command(
      OUTPUT ${METAL_AIR}
      COMMAND xcrun metal -c ${METAL_SHADER_SRC} -o ${METAL_AIR}
      DEPENDS ${METAL_SHADER_SRC}
      COMMENT "Compiling Metal shaders"
    )
    add_custom_command(
      OUTPUT ${METAL_LIB}
      COMMAND xcrun metallib ${METAL_AIR} -o ${METAL_LIB}
      DEPENDS ${METAL_AIR}
      COMMENT "Linking Metal library"
    )
    add_custom_target(metal_shaders DEPENDS ${METAL_LIB})

    set(METAL_BACKEND_SRC "${FAST_LIO_ROOT}/include/compute/metal_backend.mm")
    set(METAL_LIBS ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})

    message(STATUS "Metal GPU backend: ENABLED")
  endif()
endif()

if(NOT HAS_METAL)
  message(STATUS "Metal GPU backend: DISABLED (not macOS or no Metal framework)")
endif()

# ──────────────────────────────────────────────
# CUDA Backend (Linux / Windows with NVIDIA GPU)
# ──────────────────────────────────────────────
set(HAS_CUDA OFF)
if(NOT HAS_METAL)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(HAS_CUDA ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    set(CUDA_BACKEND_SRCS
      "${FAST_LIO_ROOT}/include/compute/cuda/cuda_backend.cu"
      "${FAST_LIO_ROOT}/include/compute/cuda/kernels.cu"
    )

    message(STATUS "CUDA GPU backend: ENABLED (${CMAKE_CUDA_COMPILER})")
  else()
    message(STATUS "CUDA GPU backend: DISABLED (no CUDA compiler found)")
  endif()
else()
  message(STATUS "CUDA GPU backend: DISABLED (Metal backend active)")
endif()

# ──────────────────────────────────────────────
# Unit Tests
# ──────────────────────────────────────────────
enable_testing()
include(GoogleTest)

# Test: SO(3) math (Exp, Log, skew_sym_mat, RotMtoEuler)
add_executable(test_so3_math test_so3_math.cpp)
target_link_libraries(test_so3_math GTest::gtest_main Eigen3::Eigen)
gtest_discover_tests(test_so3_math)

# Test: Plane estimation (esti_plane, esti_normvector)
add_executable(test_plane_estimation test_plane_estimation.cpp)
target_link_libraries(test_plane_estimation GTest::gtest_main Eigen3::Eigen ${PCL_LIBRARIES})
gtest_discover_tests(test_plane_estimation)

# Test: ikd-Tree (build, k-NN, insert, delete, box operations)
# Each test runs in its own process because ikd-Tree spawns background threads
# that don't cleanly co-exist across multiple tree instances.
add_executable(test_ikd_tree test_ikd_tree.cpp ${IKD_TREE_SRC})
target_link_libraries(test_ikd_tree GTest::gtest_main Eigen3::Eigen ${PCL_LIBRARIES})
gtest_discover_tests(test_ikd_tree PROPERTIES TIMEOUT 30)

# Test: Compute backend (CPU reference implementation)
add_executable(test_compute_backend test_compute_backend.cpp)
target_link_libraries(test_compute_backend GTest::gtest_main Eigen3::Eigen)
gtest_discover_tests(test_compute_backend PROPERTIES TIMEOUT 30)

# Test: Metal compute backend (if available)
if(HAS_METAL)
  add_executable(test_metal_backend test_metal_backend.cpp ${METAL_BACKEND_SRC})
  add_dependencies(test_metal_backend metal_shaders)
  target_link_libraries(test_metal_backend GTest::gtest_main Eigen3::Eigen ${METAL_LIBS})
  target_compile_definitions(test_metal_backend PRIVATE HAS_METAL=1)
  # Obj-C++ flags for the .mm file
  set_source_files_properties(${METAL_BACKEND_SRC} PROPERTIES
    COMPILE_FLAGS "-fobjc-arc"
  )
  gtest_discover_tests(test_metal_backend PROPERTIES TIMEOUT 60)
endif()

# ──────────────────────────────────────────────
# Benchmarks
# ──────────────────────────────────────────────

# Benchmark: ikd-Tree k-NN search
add_executable(bench_ikd_tree bench_ikd_tree.cpp ${IKD_TREE_SRC})
target_link_libraries(bench_ikd_tree benchmark::benchmark benchmark::benchmark_main Eigen3::Eigen ${PCL_LIBRARIES})

# Benchmark: point transform + plane fitting + H^T*H
add_executable(bench_compute_kernels bench_compute_kernels.cpp)
target_link_libraries(bench_compute_kernels benchmark::benchmark benchmark::benchmark_main Eigen3::Eigen ${PCL_LIBRARIES})

# Benchmark: ComputeBackend (CPU baseline for GPU comparison)
add_executable(bench_compute_backend bench_compute_backend.cpp)
target_link_libraries(bench_compute_backend benchmark::benchmark Eigen3::Eigen)

# Benchmark: Metal backend vs CPU (if available)
if(HAS_METAL)
  add_executable(bench_metal_backend bench_metal_backend.cpp ${METAL_BACKEND_SRC})
  add_dependencies(bench_metal_backend metal_shaders)
  target_link_libraries(bench_metal_backend benchmark::benchmark Eigen3::Eigen ${METAL_LIBS})
  target_compile_definitions(bench_metal_backend PRIVATE HAS_METAL=1)
  set_source_files_properties(${METAL_BACKEND_SRC} PROPERTIES
    COMPILE_FLAGS "-fobjc-arc"
  )
endif()

# ──────────────────────────────────────────────
# CUDA Tests & Benchmarks (if available)
# ──────────────────────────────────────────────
if(HAS_CUDA)
  # Test: CUDA compute backend
  add_executable(test_cuda_backend test_cuda_backend.cpp ${CUDA_BACKEND_SRCS})
  target_link_libraries(test_cuda_backend GTest::gtest_main Eigen3::Eigen)
  target_compile_definitions(test_cuda_backend PRIVATE HAS_CUDA=1)
  set_target_properties(test_cuda_backend PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80;86;89;90"
  )
  gtest_discover_tests(test_cuda_backend PROPERTIES TIMEOUT 60)

  # Benchmark: CUDA backend vs CPU
  add_executable(bench_cuda_backend bench_cuda_backend.cpp ${CUDA_BACKEND_SRCS})
  target_link_libraries(bench_cuda_backend benchmark::benchmark Eigen3::Eigen)
  target_compile_definitions(bench_cuda_backend PRIVATE HAS_CUDA=1)
  set_target_properties(bench_cuda_backend PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80;86;89;90"
  )
endif()
