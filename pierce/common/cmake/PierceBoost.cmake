# Modern Windows Boost packages provide CMake targets and header-only System.
# Retain module discovery for the Linux artifact's older Conda packages.
if(WIN32)
    find_package(Boost CONFIG REQUIRED COMPONENTS filesystem)
    set(PIERCE_BOOST_LIBRARIES Boost::filesystem)
else()
    if(POLICY CMP0167)
        cmake_policy(SET CMP0167 OLD)
    endif()
    set(Boost_NO_BOOST_CMAKE ON)
    find_package(Boost REQUIRED COMPONENTS filesystem system)
    set(PIERCE_BOOST_LIBRARIES Boost::filesystem Boost::system)
endif()
