cmake_minimum_required(VERSION 3.16)
project(opengauss_vector_engine_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(
    opengauss_vector_core
    src/opq_rabitq.cpp
    src/diskann_scheduler.cpp
    src/versioned_graph.cpp
)
target_include_directories(opengauss_vector_core PUBLIC include)

add_executable(opengauss_vector_demo src/demo.cpp)
target_link_libraries(opengauss_vector_demo PRIVATE opengauss_vector_core)
