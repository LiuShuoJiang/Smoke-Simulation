# Copyright (c) 2025. Shuojiang Liu.

add_executable(Simulation main.cu calculation/Kernels.cu simulation/SimulationSystem.cu)

target_include_directories(Simulation PUBLIC ../include)

target_compile_options(Simulation PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(Simulation PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

CPMAddPackage(
        NAME OpenVDB
        GITHUB_REPOSITORY AcademySoftwareFoundation/openvdb
        GIT_TAG v12.0.0
        OPTIONS "USE_MAYA OFF" "USE_HOUDINI OFF"
)

if (OpenVDB_ADDED)
    # add_library(OpenVDB::openvdb ALIAS openvdb)
    message(STATUS "Successfully added OpenVDB!")
endif()

message(STATUS "Found OpenVDB ${OpenVDB_VERSION} at ${OpenVDB_LIBRARIES}")

target_link_libraries(Simulation PUBLIC openvdb)
target_sources(Simulation PUBLIC vdb/VDBWriter.cpp)
