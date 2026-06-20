cmake_minimum_required(VERSION 3.25)

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

if(NOT WIN32)
    link_directories(/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu")
endif()

project(PierceTDBaseExtensions LANGUAGES CXX)

set(TDBASE_PATCH_DIR "${CMAKE_CURRENT_LIST_DIR}/src")
set(TDBASE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../tdbase/src")
if(NOT EXISTS "${TDBASE_SOURCE_DIR}/tools")
    message(FATAL_ERROR "TDBase source directory not found: ${TDBASE_SOURCE_DIR}")
endif()

function(gather_sources SUBDIR RESULT_VAR)
    file(GLOB ORIGINAL_SRCS "${TDBASE_SOURCE_DIR}/${SUBDIR}/*.cpp" "${TDBASE_SOURCE_DIR}/${SUBDIR}/*.cu")
    set(FINAL_SRCS "")
    foreach(FILE_PATH ${ORIGINAL_SRCS})
        get_filename_component(FILE_NAME ${FILE_PATH} NAME)
        if(EXISTS "${TDBASE_PATCH_DIR}/${SUBDIR}/${FILE_NAME}")
            list(APPEND FINAL_SRCS "${TDBASE_PATCH_DIR}/${SUBDIR}/${FILE_NAME}")
        else()
            list(APPEND FINAL_SRCS ${FILE_PATH})
        endif()
    endforeach()
    
    # Also add any files that exist only in patch dir
    file(GLOB PATCH_SRCS "${TDBASE_PATCH_DIR}/${SUBDIR}/*.cpp" "${TDBASE_PATCH_DIR}/${SUBDIR}/*.cu")
    foreach(FILE_PATH ${PATCH_SRCS})
        get_filename_component(FILE_NAME ${FILE_PATH} NAME)
        # If it wasn't already added (meaning it's a new file)
        set(ALREADY_ADDED FALSE)
        foreach(ADDED_FILE ${FINAL_SRCS})
            get_filename_component(ADDED_NAME ${ADDED_FILE} NAME)
            if("${ADDED_NAME}" STREQUAL "${FILE_NAME}")
                set(ALREADY_ADDED TRUE)
                break()
            endif()
        endforeach()
        if(NOT ALREADY_ADDED)
            list(APPEND FINAL_SRCS ${FILE_PATH})
        endif()
    endforeach()

    set(${RESULT_VAR} ${FINAL_SRCS} PARENT_SCOPE)
endfunction()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CGAL REQUIRED)
find_package(GMP REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

if(Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIRS})
endif()

set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)

gather_sources("spatial" SPATIAL_SRCS)
gather_sources("geometry" GEOMETRY_SRCS)
gather_sources("index" INDEX_SRCS)
gather_sources("storage" STORAGE_SRCS)
gather_sources("join" JOIN_SRCS)
gather_sources("mymesh" MYMESH_SRCS)
gather_sources("tools" TDBASE_SRCS)
list(FILTER TDBASE_SRCS EXCLUDE REGEX "obj_to_dt\\.cpp$")
list(FILTER TDBASE_SRCS EXCLUDE REGEX "tdbase_embedded\\.cpp$")
list(FILTER TDBASE_SRCS EXCLUDE REGEX "obj_to_dt_config\\.cpp$")

# Filter sources by type as before
set(GEOMETRY_CUDA_SRCS "")
set(TMP_GEOMETRY_SRCS "")
foreach(FILE ${GEOMETRY_SRCS})
    if(FILE MATCHES ".*\\.cu$")
        list(APPEND GEOMETRY_CUDA_SRCS ${FILE})
    else()
        list(APPEND TMP_GEOMETRY_SRCS ${FILE})
    endif()
endforeach()
set(GEOMETRY_SRCS ${TMP_GEOMETRY_SRCS})

# Include patch headers first
include_directories("${TDBASE_PATCH_DIR}/include")
include_directories("${TDBASE_SOURCE_DIR}/include")
include_directories("${TDBASE_SOURCE_DIR}/geometry")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_HAS_THREADS -DCGAL_HAS_THREADS -DCGAL_DONT_USE_LAZY_KERNEL")

if(DEFINED ENV{USE_GPU} OR USE_GPU)
    message(STATUS "cuda")
    set(CMAKE_CUDA_ARCHITECTURES 86)
    # Find system CUDA compiler
    if(NOT CMAKE_CUDA_COMPILER)
        find_program(SYSTEM_NVCC nvcc
            PATHS /usr/local/cuda/bin /usr/local/cuda-11/bin /usr/local/cuda-12/bin /opt/cuda/bin /usr/bin
            NO_DEFAULT_PATH)

        if(SYSTEM_NVCC)
            set(CMAKE_CUDA_COMPILER "${SYSTEM_NVCC}")
            message(STATUS "Using system CUDA compiler: ${SYSTEM_NVCC}")
        else()
            message(FATAL_ERROR "Could not find nvcc. Please install CUDA toolkit.")
        endif()
    endif()
    enable_language("CUDA")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_GPU")
    add_executable(tdbase ${TDBASE_SRCS} ${SPATIAL_SRCS} ${GEOMETRY_SRCS} ${GEOMETRY_CUDA_SRCS} ${INDEX_SRCS} ${STORAGE_SRCS} ${JOIN_SRCS})
    target_link_libraries(tdbase PRIVATE gmp mpfr pthread OpenMP::OpenMP_CXX cuda)

else()
    message(STATUS "non-cuda")
    add_executable(tdbase ${TDBASE_SRCS} ${SPATIAL_SRCS} ${GEOMETRY_SRCS} ${INDEX_SRCS} ${STORAGE_SRCS} ${JOIN_SRCS})
    target_link_libraries(tdbase PRIVATE gmp mpfr pthread OpenMP::OpenMP_CXX)
endif()

add_executable(wrap "${TDBASE_SOURCE_DIR}/simplifier/wrap.cpp")
target_link_libraries(wrap PRIVATE gmp mpfr)

add_executable(mymesh ${MYMESH_SRCS})

# OBJ Validator for DT Conversion (Phase 1 & 2)
add_executable(obj_validator_for_dt "${CMAKE_CURRENT_LIST_DIR}/ObjValidatorForDt.cpp")
set_target_properties(obj_validator_for_dt PROPERTIES CXX_STANDARD 17)

# Legacy OBJ to DT Converter - Full library version with Tile aggregation# Exclude develop.cpp from MYMESH_SRCS since it has a main()
set(MYMESH_SRCS_NODEV ${MYMESH_SRCS})
list(FILTER MYMESH_SRCS_NODEV EXCLUDE REGEX "develop\\.cpp$")

add_executable(obj_to_dt
    "${TDBASE_PATCH_DIR}/tools/obj_to_dt.cpp")
target_link_libraries(obj_to_dt PRIVATE pthread)
set_target_properties(obj_to_dt PROPERTIES CXX_STANDARD 17)
