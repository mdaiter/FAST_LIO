cmake_minimum_required(VERSION 2.8.3)
project(fast_lio)

SET(CMAKE_BUILD_TYPE "Debug")

ADD_COMPILE_OPTIONS(-std=c++14 )
ADD_COMPILE_OPTIONS(-std=c++14 )
set( CMAKE_CXX_FLAGS "-std=c++14 -O3" )

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -std=c++0x -std=c++14 -fexceptions")

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
    message("core for MP: 3")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=2)
    message("core for MP: 2")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

# ─── Metal GPU backend (Apple Silicon / macOS) ───────────────────────
set(USE_METAL OFF)
if(APPLE)
  find_library(METAL_FRAMEWORK Metal)
  find_library(FOUNDATION_FRAMEWORK Foundation)
  if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
    set(USE_METAL ON)
    message(STATUS "Metal GPU backend: ENABLED")

    # Compile .metal → .metallib
    set(METAL_SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/include/compute/metal/kernels.metal")
    set(METAL_AIR "${CMAKE_CURRENT_BINARY_DIR}/kernels.air")
    set(METAL_LIB "${CMAKE_CURRENT_BINARY_DIR}/kernels.metallib")
    add_custom_command(
      OUTPUT ${METAL_AIR}
      COMMAND xcrun -sdk macosx metal -c ${METAL_SHADER_SRC} -o ${METAL_AIR}
      DEPENDS ${METAL_SHADER_SRC}
      COMMENT "Compiling Metal shaders → kernels.air"
    )
    add_custom_command(
      OUTPUT ${METAL_LIB}
      COMMAND xcrun -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
      DEPENDS ${METAL_AIR}
      COMMENT "Linking Metal library → kernels.metallib"
    )
    add_custom_target(metal_shaders DEPENDS ${METAL_LIB})
  else()
    message(STATUS "Metal GPU backend: DISABLED (Metal framework not found)")
  endif()
else()
  message(STATUS "Metal GPU backend: DISABLED (non-Apple platform)")
endif()

# ─── CUDA GPU backend (NVIDIA GPUs) ──────────────────────────────────
set(USE_CUDA OFF)
if(NOT USE_METAL)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(USE_CUDA ON)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    message(STATUS "CUDA GPU backend: ENABLED (${CMAKE_CUDA_COMPILER})")
  else()
    message(STATUS "CUDA GPU backend: DISABLED (no CUDA compiler found)")
  endif()
else()
  message(STATUS "CUDA GPU backend: DISABLED (Metal backend active)")
endif()

find_package(PythonLibs REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  livox_ros_driver
  message_generation
  eigen_conversions
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)

message(Eigen: ${EIGEN3_INCLUDE_DIR})

include_directories(
	${catkin_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  include)

add_message_files(
  FILES
  Pose6D.msg
)

generate_messages(
 DEPENDENCIES
 geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS
)

if(USE_METAL)
  # Metal backend needs Objective-C++ compilation
  set_source_files_properties(include/compute/metal_backend.mm PROPERTIES
    LANGUAGE OBJCXX
    COMPILE_FLAGS "-fobjc-arc -std=c++14"
  )
  add_executable(fastlio_mapping
    src/laserMapping.cpp
    include/ikd-Tree/ikd_Tree.cpp
    src/preprocess.cpp
    include/compute/metal_backend.mm
  )
  add_dependencies(fastlio_mapping metal_shaders)
  target_compile_definitions(fastlio_mapping PRIVATE HAS_METAL USE_GPU_BACKEND)
  target_link_libraries(fastlio_mapping
    ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES}
    ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK}
  )
  # Tell Metal backend where to find the compiled .metallib
  target_compile_definitions(fastlio_mapping PRIVATE
    METALLIB_PATH="${CMAKE_CURRENT_BINARY_DIR}/kernels.metallib"
  )
elseif(USE_CUDA)
  # CUDA backend — .cu files compiled by nvcc, rest by host compiler
  add_executable(fastlio_mapping
    src/laserMapping.cpp
    include/ikd-Tree/ikd_Tree.cpp
    src/preprocess.cpp
    include/compute/cuda/cuda_backend.cu
    include/compute/cuda/kernels.cu
  )
  target_compile_definitions(fastlio_mapping PRIVATE HAS_CUDA USE_GPU_BACKEND)
  target_link_libraries(fastlio_mapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES})
  # Ensure CUDA files can find our headers
  target_include_directories(fastlio_mapping PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  # Set CUDA architecture (auto-detect or reasonable defaults)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(fastlio_mapping PROPERTIES CUDA_ARCHITECTURES "60;70;75;80;86;89;90")
  endif()
  set_target_properties(fastlio_mapping PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
else()
  add_executable(fastlio_mapping src/laserMapping.cpp include/ikd-Tree/ikd_Tree.cpp src/preprocess.cpp)
  target_link_libraries(fastlio_mapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES})
endif()
target_include_directories(fastlio_mapping PRIVATE ${PYTHON_INCLUDE_DIRS})