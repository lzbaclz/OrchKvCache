cmake_minimum_required(VERSION 3.18)
project(OrchKvCache LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# Compile options
add_compile_options(-Wall -Wextra -Wno-unused-parameter)

# Log level: 0=off, 1=err, 2=warn, 3=info, 4=debug
# Override with -DORCHKV_LOG_LEVEL=4 for verbose builds
if(NOT DEFINED ORCHKV_LOG_LEVEL)
    set(ORCHKV_LOG_LEVEL 3)
endif()
add_definitions(-DORCHKV_LOG_LEVEL=${ORCHKV_LOG_LEVEL})

# ---------- Core static library ----------
add_library(orchkv STATIC
    src/core/kv_block.c
    src/core/kv_request.c
    src/core/address_map.c
    src/tiered_store/dram_tier.c
    src/tiered_store/gpu_tier.cu
    src/tiered_store/transfer.cu
    src/api/orchkv_api.c
)

target_include_directories(orchkv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(orchkv PUBLIC CUDA::cudart pthread)

set_target_properties(orchkv PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ---------- Tests ----------
enable_testing()

add_executable(test_kv_types test/test_kv_types.c)
target_include_directories(test_kv_types PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_kv_types orchkv)
add_test(NAME test_kv_types COMMAND test_kv_types)

add_executable(test_kv_block test/test_kv_block.c)
target_include_directories(test_kv_block PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_kv_block orchkv)
add_test(NAME test_kv_block COMMAND test_kv_block)

add_executable(test_address_map test/test_address_map.c)
target_include_directories(test_address_map PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_address_map orchkv)
add_test(NAME test_address_map COMMAND test_address_map)

add_executable(test_kv_request test/test_kv_request.c)
target_include_directories(test_kv_request PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_kv_request orchkv)
add_test(NAME test_kv_request COMMAND test_kv_request)
