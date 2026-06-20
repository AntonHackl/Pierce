cmake_minimum_required(VERSION 3.25)
project(PiercePreprocess LANGUAGES CXX)

add_compile_definitions(NOMINMAX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
# Conda's Boost packages do not ship per-component BoostConfig files. Force
# CMake's FindBoost module so it resolves headers and libraries from the
# active environment instead of mixing them with a system BoostConfig.
set(Boost_NO_BOOST_CMAKE ON)
find_package(CGAL REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# TinyObjLoader
find_package(tinyobjloader CONFIG QUIET)
if(NOT tinyobjloader_FOUND)
    message(FATAL_ERROR "TinyObjLoader not found. Install via conda: conda install tinyobjloader")
endif()

# Add common library
add_subdirectory(../common ${CMAKE_CURRENT_BINARY_DIR}/common)

# Global compiler definitions
add_compile_definitions(NOMINMAX)

# Find Eigen3
find_package(Eigen3 CONFIG REQUIRED)

# Include Eigen3 path
if(TARGET Eigen3::Eigen)
    link_libraries(Eigen3::Eigen)
elseif(DEFINED ENV{CONDA_PREFIX})
    # Conda installs headers/libs under different directories on Windows vs Linux
    if(WIN32)
        set(CONDA_INCLUDE_DIR "$ENV{CONDA_PREFIX}/Library/include")
        set(CONDA_LIB_DIR "$ENV{CONDA_PREFIX}/Library/lib")
    else()
        set(CONDA_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")
        set(CONDA_LIB_DIR "$ENV{CONDA_PREFIX}/lib")
    endif()
    include_directories(SYSTEM "${CONDA_INCLUDE_DIR}/eigen3")
endif()

# TDBase library sources
set(TDBASE_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tdbase_lib)
file(GLOB TDBASE_SPATIAL_SRCS ${TDBASE_LIB_DIR}/spatial/*.cpp)
file(GLOB TDBASE_STORAGE_SRCS ${TDBASE_LIB_DIR}/storage/*.cpp)
file(GLOB TDBASE_GEOMETRY_SRCS ${TDBASE_LIB_DIR}/geometry/*.cpp)
file(GLOB TDBASE_INDEX_SRCS ${TDBASE_LIB_DIR}/index/*.cpp)

set(TDBASE_SOURCES 
    ${TDBASE_SPATIAL_SRCS} 
    ${TDBASE_STORAGE_SRCS} 
    ${TDBASE_GEOMETRY_SRCS} 
    ${TDBASE_INDEX_SRCS}
)

# Dataset loader sources
set(DATASET_SOURCES
    src/EdgePreprocessor.cpp
    src/DatasetLoaderFactory.cpp
    src/ObjMeshDatasetLoader.cpp
    src/DtMeshDatasetLoader.cpp
    ${TDBASE_SOURCES}
)

# Preprocessing executable
add_executable(pierce_preprocess
    src/pierce_preprocess.cpp
    ${DATASET_SOURCES}
)

target_include_directories(pierce_preprocess PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TDBASE_LIB_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
    $<IF:$<BOOL:${CONDA_INCLUDE_DIR}>,${CONDA_INCLUDE_DIR},$ENV{CONDA_PREFIX}>
)

# Add CUDA library path even though we don't link CUDA at runtime
if(DEFINED CONDA_LIB_DIR)
    target_link_directories(pierce_preprocess PRIVATE ${CONDA_LIB_DIR})
elseif(DEFINED ENV{CONDA_PREFIX} AND NOT WIN32)
    # fallback for non-Windows conda where CONDA_LIB_DIR may not have been set
    target_link_directories(pierce_preprocess PRIVATE $ENV{CONDA_PREFIX}/lib)
endif()

target_link_libraries(pierce_preprocess PRIVATE
    pierce_common
    CGAL::CGAL
    tinyobjloader::tinyobjloader
    Boost::filesystem
    Boost::system
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

target_compile_definitions(pierce_preprocess PRIVATE
    BOOST_HAS_THREADS
    CGAL_HAS_THREADS
    CGAL_DONT_USE_LAZY_KERNEL
)

set_target_properties(pierce_preprocess PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
