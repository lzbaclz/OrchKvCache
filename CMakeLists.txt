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

# ---------- OrchFS detection (Phase B) ----------
set(ORCHFS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/OrchFS)
set(ORCHFS_LIB ${ORCHFS_DIR}/build/libOrchFS.so)
set(ORCHFS_INCLUDE ${ORCHFS_DIR}/LibFS)

if(EXISTS ${ORCHFS_LIB})
    set(HAS_ORCHFS TRUE)
    message(STATUS "OrchFS found: ${ORCHFS_LIB}")
else()
    set(HAS_ORCHFS FALSE)
    message(STATUS "OrchFS not found at ${ORCHFS_LIB} — Phase B features disabled")
endif()

# ---------- Core static library ----------
add_library(orchkv STATIC
    src/core/kv_block.c
    src/core/kv_request.c
    src/core/address_map.c
    src/tiered_store/dram_tier.cu
    src/tiered_store/gpu_tier.cu
    src/tiered_store/transfer.cu
    src/tiered_store/orchfs_tier.c
    src/tiered_store/io_worker.c
    src/scheduler/attention_tracker.c
    src/scheduler/hotcold_classifier.c
    src/scheduler/adaptive_threshold.c
    src/scheduler/eviction_policy.c
    src/api/orchkv_api.cu
)

target_include_directories(orchkv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(orchkv PUBLIC CUDA::cudart pthread)

if(HAS_ORCHFS)
    target_compile_definitions(orchkv PUBLIC ORCHKV_HAS_ORCHFS=1)
    target_include_directories(orchkv PUBLIC ${ORCHFS_INCLUDE})
    target_link_libraries(orchkv PUBLIC ${ORCHFS_LIB})
    find_library(PMEM_LIB pmem)
    if(PMEM_LIB)
        target_link_libraries(orchkv PUBLIC ${PMEM_LIB})
    endif()
else()
    target_compile_definitions(orchkv PUBLIC ORCHKV_HAS_ORCHFS=0)
endif()

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

add_executable(test_gpu_dram test/test_gpu_dram.cu)
target_include_directories(test_gpu_dram PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_gpu_dram orchkv)
add_test(NAME test_gpu_dram COMMAND test_gpu_dram)

add_executable(test_e2e test/test_e2e.cu)
target_include_directories(test_e2e PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_e2e orchkv)
add_test(NAME test_e2e COMMAND test_e2e)

add_executable(test_orchfs_tier test/test_orchfs_tier.c)
target_include_directories(test_orchfs_tier PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_orchfs_tier orchkv)
add_test(NAME test_orchfs_tier COMMAND test_orchfs_tier)

add_executable(test_e2e_4tier test/test_e2e_4tier.cu)
target_include_directories(test_e2e_4tier PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_e2e_4tier orchkv)
add_test(NAME test_e2e_4tier COMMAND test_e2e_4tier)

# Phase C tests
add_executable(test_attention_tracker test/test_attention_tracker.c)
target_include_directories(test_attention_tracker PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_attention_tracker orchkv m)
add_test(NAME test_attention_tracker COMMAND test_attention_tracker)

add_executable(test_hotcold_classifier test/test_hotcold_classifier.c)
target_include_directories(test_hotcold_classifier PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_hotcold_classifier orchkv m)
add_test(NAME test_hotcold_classifier COMMAND test_hotcold_classifier)

add_executable(test_adaptive_threshold test/test_adaptive_threshold.c)
target_include_directories(test_adaptive_threshold PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_adaptive_threshold orchkv m)
add_test(NAME test_adaptive_threshold COMMAND test_adaptive_threshold)

add_executable(test_eviction_policy test/test_eviction_policy.c)
target_include_directories(test_eviction_policy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_eviction_policy orchkv m)
add_test(NAME test_eviction_policy COMMAND test_eviction_policy)
