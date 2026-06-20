cmake_minimum_required(VERSION 3.18)
project(PierceCommon LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Common source files
set(COMMON_SOURCES
    src/timer.cpp
    src/triangulation.cpp
    include/DatasetUtils.cpp
)

# Create a static library
add_library(pierce_common STATIC ${COMMON_SOURCES})

# Public include directories
target_include_directories(pierce_common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Find CGAL (for triangulation)
set(Boost_NO_BOOST_CMAKE ON)
find_package(CGAL REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)

target_link_libraries(pierce_common PUBLIC
    CGAL::CGAL
    Boost::filesystem
    Boost::system
)
