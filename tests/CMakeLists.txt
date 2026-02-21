# ==================================================================
# Neuro-Fabric — Tests
# ==================================================================

add_executable(nf_smoke_test smoke_test.cpp)
target_link_libraries(nf_smoke_test PRIVATE nf_core)
target_include_directories(nf_smoke_test PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
)
add_test(NAME smoke COMMAND nf_smoke_test)

add_executable(nf_buffer_test buffer_test.cpp)
target_link_libraries(nf_buffer_test PRIVATE nf_core)
target_include_directories(nf_buffer_test PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
)
add_test(NAME buffer COMMAND nf_buffer_test)

add_executable(nf_scheduler_test scheduler_test.cpp)
target_link_libraries(nf_scheduler_test PRIVATE nf_core)
target_include_directories(nf_scheduler_test PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
)
add_test(NAME scheduler COMMAND nf_scheduler_test)

add_executable(nf_e2e_test e2e_pipeline_test.cpp)
target_link_libraries(nf_e2e_test PRIVATE nf_core)
target_include_directories(nf_e2e_test PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
)
add_test(NAME e2e COMMAND nf_e2e_test)

add_executable(nf_split_inference_test split_inference_test.cpp)
target_link_libraries(nf_split_inference_test PRIVATE nf_core)
target_include_directories(nf_split_inference_test PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
)
add_test(NAME split_inference COMMAND nf_split_inference_test)

add_executable(nf_ir_loader_test
    ir_loader_test.cpp
    ${CMAKE_SOURCE_DIR}/core/src/graph/mmap_buffer.cpp
    ${CMAKE_SOURCE_DIR}/core/src/graph/GraphBuilder.cpp
)
target_link_libraries(nf_ir_loader_test PRIVATE nf_core)
target_include_directories(nf_ir_loader_test PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
    ${CMAKE_SOURCE_DIR}/core/src
)
add_test(NAME ir_loader COMMAND nf_ir_loader_test)

add_executable(nf_nfir_e2e_test
    nfir_e2e_test.cpp
    ${CMAKE_SOURCE_DIR}/core/src/graph/mmap_buffer.cpp
    ${CMAKE_SOURCE_DIR}/core/src/graph/GraphBuilder.cpp
)
target_link_libraries(nf_nfir_e2e_test PRIVATE nf_core)
target_include_directories(nf_nfir_e2e_test PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
    ${CMAKE_SOURCE_DIR}/core/src
)
add_test(NAME nfir_e2e COMMAND nf_nfir_e2e_test)

add_executable(nf_context_hub_test context_hub_test.cpp)
target_link_libraries(nf_context_hub_test PRIVATE nf_core)
target_include_directories(nf_context_hub_test PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
)
add_test(NAME context_hub COMMAND nf_context_hub_test)

add_executable(nf_split_llama_mock_test
    split_llama_mock_test.cpp
    ${CMAKE_SOURCE_DIR}/core/src/graph/mmap_buffer.cpp
    ${CMAKE_SOURCE_DIR}/core/src/graph/GraphBuilder.cpp
)
target_link_libraries(nf_split_llama_mock_test PRIVATE nf_core)
target_include_directories(nf_split_llama_mock_test PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
    ${CMAKE_SOURCE_DIR}/core/src
)
add_test(NAME split_llama_mock COMMAND nf_split_llama_mock_test)

add_executable(nf_profiling_test profiling_test.cpp)
target_link_libraries(nf_profiling_test PRIVATE nf_core)
target_include_directories(nf_profiling_test PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
)
add_test(NAME profiling COMMAND nf_profiling_test)

if(NF_PLUGIN_METAL)
    add_executable(nf_silicon_test silicon_test.cpp)
    target_link_libraries(nf_silicon_test PRIVATE nf_core nf_plugin_metal)
    target_include_directories(nf_silicon_test PRIVATE
        ${CMAKE_SOURCE_DIR}/core/include
    )
    add_test(NAME silicon COMMAND nf_silicon_test)
endif()

if(NF_PLUGIN_RKNN)
    add_executable(nf_rknn_silicon_test
        rknn_silicon_test.cpp
        ${CMAKE_SOURCE_DIR}/core/src/graph/mmap_buffer.cpp
        ${CMAKE_SOURCE_DIR}/core/src/graph/GraphBuilder.cpp
    )
    target_link_libraries(nf_rknn_silicon_test PRIVATE nf_core nf_plugin_rknn)
    target_include_directories(nf_rknn_silicon_test PRIVATE
        ${CMAKE_SOURCE_DIR}/core/include
        ${CMAKE_SOURCE_DIR}/core/src
    )
    if(NF_HAS_RKNN_SDK)
        target_compile_definitions(nf_rknn_silicon_test PRIVATE NF_HAS_RKNN_SDK)
        target_link_libraries(nf_rknn_silicon_test PRIVATE ${RKNN_RT})
    endif()
    add_test(NAME rknn_silicon COMMAND nf_rknn_silicon_test)
endif()
