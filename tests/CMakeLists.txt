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

add_executable(nf_session_test
    session_test.cpp
    ${CMAKE_SOURCE_DIR}/core/src/graph/mmap_buffer.cpp
    ${CMAKE_SOURCE_DIR}/core/src/graph/GraphBuilder.cpp
)
target_link_libraries(nf_session_test PRIVATE nf_core)
target_include_directories(nf_session_test PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
    ${CMAKE_SOURCE_DIR}/core/src
)
add_test(NAME session COMMAND nf_session_test)

add_executable(nf_push_constants_test
    push_constants_test.cpp
    ${CMAKE_SOURCE_DIR}/core/src/graph/mmap_buffer.cpp
    ${CMAKE_SOURCE_DIR}/core/src/graph/GraphBuilder.cpp
)
target_link_libraries(nf_push_constants_test PRIVATE nf_core)
target_include_directories(nf_push_constants_test PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
    ${CMAKE_SOURCE_DIR}/core/src
)
add_test(NAME push_constants COMMAND nf_push_constants_test)

# ---- Tokenizer test (platform-independent, no Metal/RKNN dependency) ----
add_executable(nf_tokenizer_test tokenizer_test.cpp)
target_include_directories(nf_tokenizer_test PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
    ${CMAKE_SOURCE_DIR}/tools
)
target_link_libraries(nf_tokenizer_test PRIVATE nf_core)
add_test(NAME tokenizer COMMAND nf_tokenizer_test)

# ---- Architecture registry test (platform-independent) ----
add_executable(nf_arch_registry_test arch_registry_test.cpp)
target_include_directories(nf_arch_registry_test PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
    ${CMAKE_SOURCE_DIR}/tools
)
target_link_libraries(nf_arch_registry_test PRIVATE nf_core)
add_test(NAME arch_registry COMMAND nf_arch_registry_test)

# ---- Phase 28: PSO registry table validation (platform-independent) ----
add_executable(nf_pso_registry_test pso_registry_test.cpp)
target_include_directories(nf_pso_registry_test PRIVATE
    ${CMAKE_SOURCE_DIR}/plugins/metal/src
)
add_test(NAME pso_registry COMMAND nf_pso_registry_test)

# ---- Chrome trace export test (platform-independent, mocks Metal timing) ----
add_executable(nf_chrome_trace_test chrome_trace_test.cpp)
target_include_directories(nf_chrome_trace_test PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
    ${CMAKE_SOURCE_DIR}/tools
)
target_link_libraries(nf_chrome_trace_test PRIVATE nf_core)
add_test(NAME chrome_trace COMMAND nf_chrome_trace_test)

if(NF_PLUGIN_METAL)
    add_executable(nf_silicon_test silicon_test.cpp)
    target_link_libraries(nf_silicon_test PRIVATE nf_core nf_plugin_metal)
    target_include_directories(nf_silicon_test PRIVATE
        ${CMAKE_SOURCE_DIR}/core/include
    )
    add_test(NAME silicon COMMAND nf_silicon_test)

    add_executable(nf_dequant_test dequant_test.cpp)
    target_link_libraries(nf_dequant_test PRIVATE nf_core nf_plugin_metal)
    target_include_directories(nf_dequant_test PRIVATE
        ${CMAKE_SOURCE_DIR}/core/include
    )
    add_test(NAME dequant COMMAND nf_dequant_test)

    add_executable(nf_kquant_dequant_test kquant_dequant_test.cpp)
    target_link_libraries(nf_kquant_dequant_test PRIVATE nf_core nf_plugin_metal)
    target_include_directories(nf_kquant_dequant_test PRIVATE
        ${CMAKE_SOURCE_DIR}/core/include
    )
    add_test(NAME kquant_dequant COMMAND nf_kquant_dequant_test)

    add_executable(nf_transformer_test transformer_test.cpp)
    target_link_libraries(nf_transformer_test PRIVATE nf_core nf_plugin_metal)
    target_include_directories(nf_transformer_test PRIVATE
        ${CMAKE_SOURCE_DIR}/core/include
    )
    add_test(NAME transformer COMMAND nf_transformer_test)

    add_executable(nf_multi_layer_test multi_layer_test.cpp)
    target_link_libraries(nf_multi_layer_test PRIVATE nf_core nf_plugin_metal)
    target_include_directories(nf_multi_layer_test PRIVATE
        ${CMAKE_SOURCE_DIR}/core/include
    )
    add_test(NAME multi_layer COMMAND nf_multi_layer_test)

    add_executable(nf_dag_inference_test dag_inference_test.cpp)
    target_link_libraries(nf_dag_inference_test PRIVATE nf_core nf_plugin_metal)
    target_include_directories(nf_dag_inference_test PRIVATE
        ${CMAKE_SOURCE_DIR}/core/include
    )
    add_test(NAME dag_inference COMMAND nf_dag_inference_test)

    add_executable(nf_real_model_test real_model_test.cpp)
    target_link_libraries(nf_real_model_test PRIVATE nf_core nf_plugin_metal)
    target_include_directories(nf_real_model_test PRIVATE
        ${CMAKE_SOURCE_DIR}/core/include
        ${CMAKE_SOURCE_DIR}/tools
    )
    add_test(NAME real_model COMMAND nf_real_model_test)

    add_executable(nf_matmul_bench_test matmul_bench_test.cpp)
    target_link_libraries(nf_matmul_bench_test PRIVATE nf_core nf_plugin_metal)
    target_include_directories(nf_matmul_bench_test PRIVATE
        ${CMAKE_SOURCE_DIR}/core/include
    )
    add_test(NAME matmul_bench COMMAND nf_matmul_bench_test)

    # Phase 27: FP16 kernel unit tests
    add_executable(nf_fp16_kernel_test fp16_kernel_test.cpp)
    target_link_libraries(nf_fp16_kernel_test PRIVATE nf_core nf_plugin_metal)
    target_include_directories(nf_fp16_kernel_test PRIVATE
        ${CMAKE_SOURCE_DIR}/core/include
    )
    add_test(NAME fp16_kernel COMMAND nf_fp16_kernel_test)

    # Phase 27: FP16 E2E inference test
    add_executable(nf_fp16_e2e_test fp16_e2e_test.cpp)
    target_link_libraries(nf_fp16_e2e_test PRIVATE nf_core nf_plugin_metal)
    target_include_directories(nf_fp16_e2e_test PRIVATE
        ${CMAKE_SOURCE_DIR}/core/include
        ${CMAKE_SOURCE_DIR}/tools
    )
    add_test(NAME fp16_e2e COMMAND nf_fp16_e2e_test)

    # Phase 28: F16 vs F32 micro-benchmark (built, NOT added to ctest)
    add_executable(nf_benchmark_kernels benchmark_kernels.cpp)
    target_link_libraries(nf_benchmark_kernels PRIVATE nf_core nf_plugin_metal)
    target_include_directories(nf_benchmark_kernels PRIVATE
        ${CMAKE_SOURCE_DIR}/core/include
    )

    # Phase 29: Fused dequant+linear correctness test
    add_executable(nf_fused_matmul_test fused_matmul_test.cpp)
    target_link_libraries(nf_fused_matmul_test PRIVATE nf_core nf_plugin_metal)
    target_include_directories(nf_fused_matmul_test PRIVATE
        ${CMAKE_SOURCE_DIR}/core/include
    )
    add_test(NAME fused_matmul COMMAND nf_fused_matmul_test)

    # Phase 30: Sliding window attention test
    add_executable(nf_sliding_window_test sliding_window_test.cpp)
    target_link_libraries(nf_sliding_window_test PRIVATE nf_core nf_plugin_metal)
    target_include_directories(nf_sliding_window_test PRIVATE
        ${CMAKE_SOURCE_DIR}/core/include
    )
    add_test(NAME sliding_window COMMAND nf_sliding_window_test)

    # Phase 30: Fused op DAG integration test (requires NF_TEST_GGUF_PATH)
    add_executable(nf_fused_warmup_test fused_warmup_test.cpp)
    target_link_libraries(nf_fused_warmup_test PRIVATE nf_core nf_plugin_metal)
    target_include_directories(nf_fused_warmup_test PRIVATE
        ${CMAKE_SOURCE_DIR}/core/include
        ${CMAKE_SOURCE_DIR}/tools
    )
    add_test(NAME fused_warmup COMMAND nf_fused_warmup_test)
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
