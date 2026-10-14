cmake_minimum_required(VERSION 4.0)
project(StudentTeacherAI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LibTorch ships Release binaries built against /MD; Debug builds must use the same CRT.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

# LibTorch (CUDA build). Override with -DLIBTORCH_ROOT=... if it lives elsewhere.
set(LIBTORCH_ROOT "${CMAKE_SOURCE_DIR}/third_party/libtorch" CACHE PATH "LibTorch install root")
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")
find_package(Torch REQUIRED)

add_executable(StudentTeacherAI
        src/main.cpp
        src/HybridChaserNet.cpp
        src/TrajectoryDataset.cpp
        src/TrainLoop.cpp
        src/ExportWeights.cpp
)
target_include_directories(StudentTeacherAI PRIVATE src)
target_link_libraries(StudentTeacherAI PRIVATE ${TORCH_LIBRARIES})
target_compile_definitions(StudentTeacherAI PRIVATE
        STAI_DEFAULT_UNITY_DIR="${CMAKE_SOURCE_DIR}/Chase")

if (MSVC)
    target_compile_options(StudentTeacherAI PRIVATE /utf-8 /permissive-)

    # Torch needs this so torch_cuda.dll is not dropped by the linker as unused.
    target_link_options(StudentTeacherAI PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")

    # The LibTorch DLLs (~4 GB) are hard-linked next to the exe instead of copied.
    # Falls back to a copy if the build dir is on another volume.
    file(GLOB TORCH_DLLS "${LIBTORCH_ROOT}/lib/*.dll")
    foreach (dll IN LISTS TORCH_DLLS)
        get_filename_component(dll_name "${dll}" NAME)
        set(dst "${CMAKE_BINARY_DIR}/${dll_name}")
        if (NOT EXISTS "${dst}")
            file(CREATE_LINK "${dll}" "${dst}" COPY_ON_ERROR)
        endif ()
    endforeach ()
endif ()
