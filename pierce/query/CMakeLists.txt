cmake_minimum_required(VERSION 3.18)

# Force system compiler on Linux to avoid conda gcc/glibc conflicts
if(UNIX AND NOT APPLE AND NOT WIN32)
    if(DEFINED ENV{CONDA_PREFIX} AND NOT DEFINED CMAKE_CXX_COMPILER)
        find_program(SYSTEM_CXX_COMPILER NAMES g++ PATHS /usr/bin /usr/local/bin NO_DEFAULT_PATH)
        if(SYSTEM_CXX_COMPILER)
            set(CMAKE_CXX_COMPILER "${SYSTEM_CXX_COMPILER}" CACHE FILEPATH "C++ compiler" FORCE)
            message(STATUS "Forcing system C++ compiler: ${SYSTEM_CXX_COMPILER}")
        endif()
    endif()
endif()

# CUDA must be selected before project() enables the CUDA language.
if(DEFINED ENV{CONDA_PREFIX})
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    if(WIN32)
        list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/Library")
        set(PIERCE_CUDA_ROOT "$ENV{CONDA_PREFIX}/Library")
    else()
        set(PIERCE_CUDA_ROOT "$ENV{CONDA_PREFIX}")
    endif()
    if(NOT DEFINED CMAKE_CUDA_COMPILER)
        find_program(PIERCE_NVCC NAMES nvcc.exe nvcc
            PATHS "${PIERCE_CUDA_ROOT}/bin" NO_DEFAULT_PATH)
        if(PIERCE_NVCC)
            set(CMAKE_CUDA_COMPILER "${PIERCE_NVCC}" CACHE FILEPATH "CUDA compiler")
        endif()
    endif()
    if(PIERCE_NVCC AND NOT DEFINED CUDAToolkit_ROOT)
        set(CUDAToolkit_ROOT "${PIERCE_CUDA_ROOT}" CACHE PATH "CUDA toolkit root")
    endif()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Include compute_75 PTX so CUDA kernels can JIT on newer GPU generations.
    set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA kernel architectures")
endif()

project(PierceQuery LANGUAGES CXX CUDA)

# Policy settings
cmake_policy(SET CMP0091 NEW)

# -----------------------------------------------------------------------
# Global Settings
# -----------------------------------------------------------------------

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" CACHE STRING "MSVC runtime" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIE")
    link_directories(/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(WIN32)
    add_definitions(-DNOMINMAX)
endif()

# -----------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------

include(../common/cmake/PierceBoost.cmake)
find_package(CGAL REQUIRED)
find_package(CUDAToolkit REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(cmake/FindOptiX.cmake)
include(cmake/nvcuda_compile_module.cmake)

# -----------------------------------------------------------------------
# Add Common Library
# -----------------------------------------------------------------------
add_subdirectory(../common ${CMAKE_CURRENT_BINARY_DIR}/common)

# -----------------------------------------------------------------------
# Source Definitions
# -----------------------------------------------------------------------

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# CUDA Sources
set(CUDA_DIR ${SRC_DIR}/cuda)
set(CUDA_SOURCES ${CUDA_DIR}/mesh_overlap_edges.cu ${CUDA_DIR}/mesh_intersection.cu ${CUDA_DIR}/mesh_intersection_anyhit.cu)
set(RESULT_COMPACTION_SOURCE ${CUDA_DIR}/result_compaction.cu)
set(MESH_QUERY_DEDUP_SOURCE ${CUDA_DIR}/mesh_query_deduplication.cu)
set(SCAN_UTILS_SOURCE ${CUDA_DIR}/scan_utils.cu)
set(ESTIMATED_INTERSECTION_SOURCE ${CUDA_DIR}/estimated_intersection.cu)
set(ESTIMATED_OVERLAP_SOURCE ${CUDA_DIR}/estimated_overlap.cu)
set(MESH_CONTAINMENT_SOURCE ${CUDA_DIR}/mesh_containment.cu)

# Application Sources
set(MESH_OVERLAP_APP_SOURCE ${SRC_DIR}/applications/pierce_overlap_two_pass_main.cpp)
set(ESTIMATED_INTERSECTION_APP_SOURCE ${SRC_DIR}/applications/pierce_intersection_main.cpp)
set(ESTIMATED_OVERLAP_APP_SOURCE ${SRC_DIR}/applications/pierce_overlap_main.cpp)
set(CONTAINMENT_APP_SOURCE ${SRC_DIR}/applications/pierce_containment_main.cpp)

# Modules
set(OPTIX_DIR ${SRC_DIR}/optix)
set(GEOMETRY_DIR ${SRC_DIR}/geometry)
set(RAYTRACING_DIR ${SRC_DIR}/raytracing)

set(OPTIX_SOURCES
    ${OPTIX_DIR}/OptixContext.cpp
    ${OPTIX_DIR}/OptixAccelerationStructure.cpp)

set(GEOMETRY_SOURCES
    ${GEOMETRY_DIR}/BoundingBox.cpp
    ${GEOMETRY_DIR}/GeometryUploader.cpp
    ${GEOMETRY_DIR}/PrecomputedEdgeData.cpp)

set(RAYTRACING_SOURCES
    ${RAYTRACING_DIR}/MeshOverlapEdgesLauncher.cpp
    ${RAYTRACING_DIR}/MeshIntersectionLauncher.cpp
    ${RAYTRACING_DIR}/MeshContainmentLauncher.cpp)

# Runtime sources
set(RUNTIME_SOURCES
    ${SRC_DIR}/runtime/GeometryIO.cpp ${SRC_DIR}/runtime/MemoryBudget.cpp)

# -----------------------------------------------------------------------
# PTX Compilation
# -----------------------------------------------------------------------

set(TIMER_SOURCE ${SRC_DIR}/timer.cpp)

set(PTX_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(OPTIX_MODULE_EXTENSION ".ptx")
set(OPTIX_PROGRAM_TARGET "--ptx")

NVCUDA_COMPILE_MODULE(
    SOURCES       ${CUDA_SOURCES} ${MESH_CONTAINMENT_SOURCE}
    TARGET_PATH   ${PTX_OUTPUT_DIR}
    EXTENSION     "${OPTIX_MODULE_EXTENSION}"
    GENERATED_FILES PTX_FILES
    NVCC_OPTIONS  "${OPTIX_PROGRAM_TARGET}" "--gpu-architecture=sm_75" "--relocatable-device-code=true" "--expt-relaxed-constexpr" "-I${OptiX_INCLUDE}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include"
)

# -----------------------------------------------------------------------
# Executables
# -----------------------------------------------------------------------

# Shared exact-overlap engine, also used by the original benchmark CLI.
add_library(pierce_query_runtime STATIC
    ${SRC_DIR}/runtime/OverlapQuery.cpp
    ${SRC_DIR}/runtime/QueryRuntime.cpp
    ${RUNTIME_SOURCES} ${TIMER_SOURCE} ${OPTIX_SOURCES} ${GEOMETRY_SOURCES}
    ${RAYTRACING_DIR}/MeshOverlapEdgesLauncher.cpp
    ${MESH_QUERY_DEDUP_SOURCE} ${SCAN_UTILS_SOURCE})
add_executable(pierce_overlap_two_pass ${MESH_OVERLAP_APP_SOURCE})
add_executable(pierce_worker ${SRC_DIR}/applications/pierce_worker_main.cpp)
add_executable(pierce_multi_verify ${SRC_DIR}/applications/pierce_multi_verify.cpp)
target_link_libraries(pierce_multi_verify PRIVATE pierce_query_runtime)
set_target_properties(pierce_multi_verify PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_executable(pierce_runtime_verify ${SRC_DIR}/applications/pierce_runtime_verify.cpp)
target_link_libraries(pierce_runtime_verify PRIVATE pierce_query_runtime)
set_target_properties(pierce_runtime_verify PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_link_libraries(pierce_overlap_two_pass PRIVATE pierce_query_runtime)
target_link_libraries(pierce_worker PRIVATE pierce_query_runtime)
set_target_properties(pierce_worker PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_link_libraries(pierce_query_runtime PUBLIC pierce_common CUDA::cudart CGAL::CGAL ${PIERCE_BOOST_LIBRARIES})
target_include_directories(pierce_query_runtime PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS} ${OptiX_INCLUDE} ${SRC_DIR}/runtime
    ${GEOMETRY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../common/include)

add_executable(pierce_intersection
    ${ESTIMATED_INTERSECTION_APP_SOURCE}
    ${RUNTIME_SOURCES}
    ${TIMER_SOURCE}
    ${OPTIX_SOURCES}
    ${GEOMETRY_SOURCES}
    ${RAYTRACING_SOURCES}
    ${RESULT_COMPACTION_SOURCE}
    ${MESH_QUERY_DEDUP_SOURCE}
    ${SCAN_UTILS_SOURCE}
    ${ESTIMATED_INTERSECTION_SOURCE})

add_executable(pierce_overlap
    ${ESTIMATED_OVERLAP_APP_SOURCE}
    ${RUNTIME_SOURCES}
    ${TIMER_SOURCE}
    ${OPTIX_SOURCES}
    ${GEOMETRY_SOURCES}
    ${RAYTRACING_SOURCES}
    ${RESULT_COMPACTION_SOURCE}
    ${MESH_QUERY_DEDUP_SOURCE}
    ${SCAN_UTILS_SOURCE}
    ${ESTIMATED_OVERLAP_SOURCE})

add_executable(pierce_containment
    ${CONTAINMENT_APP_SOURCE}
    ${RUNTIME_SOURCES}
    ${TIMER_SOURCE}
    ${OPTIX_SOURCES}
    ${GEOMETRY_SOURCES}
    ${RAYTRACING_SOURCES}
    ${RESULT_COMPACTION_SOURCE}
    ${MESH_QUERY_DEDUP_SOURCE}
    ${SCAN_UTILS_SOURCE}
    ${ESTIMATED_INTERSECTION_SOURCE})

foreach(TARGET_NAME pierce_multi_verify pierce_runtime_verify pierce_query_runtime pierce_worker pierce_overlap_two_pass pierce_overlap pierce_intersection pierce_containment)
    target_include_directories(${TARGET_NAME} PRIVATE 
        ${CUDAToolkit_INCLUDE_DIRS}
        ${OptiX_INCLUDE} 
        ${SRC_DIR}
        ${CUDA_DIR}
        ${OPTIX_DIR}
        ${GEOMETRY_DIR}
        ${RAYTRACING_DIR}
        ${SRC_DIR}/runtime
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/include)
endforeach()

target_link_libraries(pierce_overlap_two_pass PRIVATE 
    pierce_common
    CUDA::cudart
    CGAL::CGAL 
    ${PIERCE_BOOST_LIBRARIES})

target_link_libraries(pierce_intersection PRIVATE 
    pierce_common
    CUDA::cudart
    CGAL::CGAL 
    ${PIERCE_BOOST_LIBRARIES})

target_link_libraries(pierce_overlap PRIVATE 
    pierce_common
    CUDA::cudart
    CGAL::CGAL 
    ${PIERCE_BOOST_LIBRARIES})

target_link_libraries(pierce_containment PRIVATE 
    pierce_common
    CUDA::cudart
    CGAL::CGAL 
    ${PIERCE_BOOST_LIBRARIES})

target_compile_definitions(pierce_overlap_two_pass PRIVATE INCLUDE_OPTIX)
target_compile_definitions(pierce_intersection PRIVATE INCLUDE_OPTIX)
target_compile_definitions(pierce_overlap PRIVATE INCLUDE_OPTIX)
target_compile_definitions(pierce_containment PRIVATE INCLUDE_OPTIX)

set_target_properties(pierce_overlap_two_pass PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(pierce_intersection PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(pierce_overlap PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(pierce_containment PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# -----------------------------------------------------------------------
# Post-Build: Copy PTX
# -----------------------------------------------------------------------

set(MESH_OVERLAP_EDGES_PTX_FILE "${PTX_OUTPUT_DIR}/mesh_overlap_edges${OPTIX_MODULE_EXTENSION}")
set(MESH_INTERSECTION_PTX_FILE "${PTX_OUTPUT_DIR}/mesh_intersection${OPTIX_MODULE_EXTENSION}")
set(MESH_INTERSECTION_ANYHIT_PTX_FILE "${PTX_OUTPUT_DIR}/mesh_intersection_anyhit${OPTIX_MODULE_EXTENSION}")
set(MESH_CONTAINMENT_PTX_FILE "${PTX_OUTPUT_DIR}/mesh_containment${OPTIX_MODULE_EXTENSION}")

# Copy PTX artifacts into the runtime bin directory whenever PTX outputs change.
# Relying only on POST_BUILD copy commands is fragile because PTX can change
# without requiring the executable to relink, leaving stale PTX next to the binary.
set(RUNTIME_PTX_DIR "${CMAKE_BINARY_DIR}/bin")
add_custom_target(copy_ptx ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${RUNTIME_PTX_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MESH_OVERLAP_EDGES_PTX_FILE}"
        "${RUNTIME_PTX_DIR}/mesh_overlap_edges${OPTIX_MODULE_EXTENSION}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MESH_INTERSECTION_PTX_FILE}"
        "${RUNTIME_PTX_DIR}/mesh_intersection${OPTIX_MODULE_EXTENSION}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MESH_INTERSECTION_ANYHIT_PTX_FILE}"
        "${RUNTIME_PTX_DIR}/mesh_intersection_anyhit${OPTIX_MODULE_EXTENSION}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MESH_CONTAINMENT_PTX_FILE}"
        "${RUNTIME_PTX_DIR}/mesh_containment${OPTIX_MODULE_EXTENSION}"
    DEPENDS ${PTX_FILES}
    COMMENT "Copying PTX files to runtime bin directory..."
)
foreach(TARGET_NAME pierce_multi_verify pierce_runtime_verify pierce_query_runtime pierce_worker pierce_overlap_two_pass pierce_overlap pierce_intersection pierce_containment)
    add_dependencies(${TARGET_NAME} copy_ptx)
endforeach()
