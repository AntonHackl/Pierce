cmake_minimum_required(VERSION 3.15)
project(PierceFaceBaselines CXX)

cmake_policy(SET CMP0167 NEW)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# Find CGAL and its dependencies
find_package(CGAL REQUIRED COMPONENTS Core)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GMP REQUIRED)
find_package(MPFR REQUIRED)

# Find OpenMP for parallel execution
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP found - parallel execution enabled")
endif()

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/common/include)

# Main executable
add_executable(face_overlap
    src/overlap_query.cpp
)

add_executable(touch_overlap
    src/touch_query.cpp
)

target_link_libraries(face_overlap CGAL::CGAL)
target_link_libraries(touch_overlap CGAL::CGAL)

if(OpenMP_CXX_FOUND)
    target_link_libraries(face_overlap OpenMP::OpenMP_CXX)
    target_link_libraries(touch_overlap OpenMP::OpenMP_CXX)
endif()
