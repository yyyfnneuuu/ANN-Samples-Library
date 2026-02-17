cmake_minimum_required(VERSION 3.16)
project(knowhere_kernel_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(
    knowhere_kernel_core
    src/async_graph_searcher.cpp
    src/topk_reducer.cpp
)
target_include_directories(knowhere_kernel_core PUBLIC include)

add_executable(knowhere_kernel_demo_app src/demo.cpp)
target_link_libraries(knowhere_kernel_demo_app PRIVATE knowhere_kernel_core)
