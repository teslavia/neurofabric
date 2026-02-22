# ==================================================================
# Neuro-Fabric — Tests
# ==================================================================

# Helper: nf_add_test(NAME <exe> TEST_NAME <ctest_name> SOURCES <files...>
#                     [LINKS <libs...>] [INCLUDES <dirs...>]
#                     [GRAPH] [MODEL] [DEFS <defs...>])
#   GRAPH  → adds nf_graph_objects + core/src include
#   MODEL  → links nf_model_headers (gives "model/xxx.hpp")
function(nf_add_test)
    cmake_parse_arguments(T "GRAPH;MODEL" "NAME;TEST_NAME" "SOURCES;LINKS;INCLUDES;DEFS" ${ARGN})
    set(_srcs ${T_SOURCES})
    if(T_GRAPH)
        list(APPEND _srcs $<TARGET_OBJECTS:nf_graph_objects>)
    endif()
    add_executable(${T_NAME} ${_srcs})
    target_include_directories(${T_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/core/include
        ${T_INCLUDES}
    )
    if(T_GRAPH)
        target_include_directories(${T_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/core/src)
    endif()
    set(_libs nf_core ${T_LINKS})
    if(T_MODEL)
        list(APPEND _libs nf_model_headers)
    endif()
    target_link_libraries(${T_NAME} PRIVATE ${_libs})
    foreach(d ${T_DEFS})
        target_compile_definitions(${T_NAME} PRIVATE ${d})
    endforeach()
    if(T_TEST_NAME)
        add_test(NAME ${T_TEST_NAME} COMMAND ${T_NAME})
    endif()
endfunction()

# ------------------------------------------------------------------
# Platform-independent tests
# ------------------------------------------------------------------
nf_add_test(NAME nf_smoke_test          TEST_NAME smoke            SOURCES smoke_test.cpp)
nf_add_test(NAME nf_buffer_test         TEST_NAME buffer           SOURCES buffer_test.cpp)
nf_add_test(NAME nf_scheduler_test      TEST_NAME scheduler        SOURCES scheduler_test.cpp)
nf_add_test(NAME nf_e2e_test            TEST_NAME e2e              SOURCES e2e_pipeline_test.cpp)
nf_add_test(NAME nf_split_inference_test TEST_NAME split_inference  SOURCES split_inference_test.cpp)
nf_add_test(NAME nf_context_hub_test     TEST_NAME context_hub     SOURCES context_hub_test.cpp)
nf_add_test(NAME nf_profiling_test       TEST_NAME profiling       SOURCES profiling_test.cpp)

# GRAPH tests (need mmap_buffer + GraphBuilder objects)
nf_add_test(NAME nf_ir_loader_test       TEST_NAME ir_loader        SOURCES ir_loader_test.cpp       GRAPH)
nf_add_test(NAME nf_nfir_e2e_test        TEST_NAME nfir_e2e         SOURCES nfir_e2e_test.cpp        GRAPH)
nf_add_test(NAME nf_split_llama_mock_test TEST_NAME split_llama_mock SOURCES split_llama_mock_test.cpp GRAPH)
nf_add_test(NAME nf_session_test         TEST_NAME session          SOURCES session_test.cpp         GRAPH)
nf_add_test(NAME nf_push_constants_test  TEST_NAME push_constants   SOURCES push_constants_test.cpp  GRAPH)

# MODEL tests (need tools/model/ headers)
nf_add_test(NAME nf_tokenizer_test       TEST_NAME tokenizer        SOURCES tokenizer_test.cpp       MODEL)
nf_add_test(NAME nf_arch_registry_test   TEST_NAME arch_registry    SOURCES arch_registry_test.cpp   MODEL)
nf_add_test(NAME nf_model_config_test    TEST_NAME model_config     SOURCES model_config_test.cpp    MODEL)
nf_add_test(NAME nf_paged_kv_test        TEST_NAME paged_kv         SOURCES paged_kv_test.cpp        MODEL)
nf_add_test(NAME nf_batch_scheduler_test TEST_NAME batch_scheduler  SOURCES batch_scheduler_test.cpp MODEL)
nf_add_test(NAME nf_speculative_decode_test TEST_NAME speculative_decode SOURCES speculative_decode_test.cpp MODEL)
nf_add_test(NAME nf_chrome_trace_test    TEST_NAME chrome_trace     SOURCES chrome_trace_test.cpp    MODEL)
nf_add_test(NAME nf_qwen2_arch_test     TEST_NAME qwen2_arch       SOURCES qwen2_arch_test.cpp      MODEL)
nf_add_test(NAME nf_gemma_arch_test     TEST_NAME gemma_arch       SOURCES gemma_arch_test.cpp      MODEL)
nf_add_test(NAME nf_mixtral_arch_test   TEST_NAME mixtral_arch     SOURCES mixtral_arch_test.cpp    MODEL)

# ---- Standalone tests (no nf_core, custom includes) ----
add_executable(nf_pso_registry_test pso_registry_test.cpp)
target_include_directories(nf_pso_registry_test PRIVATE ${CMAKE_SOURCE_DIR}/plugins/metal/src)
add_test(NAME pso_registry COMMAND nf_pso_registry_test)

add_executable(nf_kv_cache_policy_test kv_cache_policy_test.cpp)
target_include_directories(nf_kv_cache_policy_test PRIVATE ${CMAKE_SOURCE_DIR}/tools)
add_test(NAME kv_cache_policy COMMAND nf_kv_cache_policy_test)
# ------------------------------------------------------------------
# Metal-dependent tests
# ------------------------------------------------------------------
if(NF_PLUGIN_METAL)
    nf_add_test(NAME nf_mistral_arch_test  TEST_NAME mistral_arch   SOURCES mistral_arch_test.cpp  LINKS nf_plugin_metal MODEL)
    nf_add_test(NAME nf_phi3_arch_test     TEST_NAME phi3_arch      SOURCES phi3_arch_test.cpp     LINKS nf_plugin_metal MODEL)
    nf_add_test(NAME nf_silicon_test       TEST_NAME silicon        SOURCES silicon_test.cpp       LINKS nf_plugin_metal)
    nf_add_test(NAME nf_dequant_test       TEST_NAME dequant        SOURCES dequant_test.cpp       LINKS nf_plugin_metal)
    nf_add_test(NAME nf_kquant_dequant_test TEST_NAME kquant_dequant SOURCES kquant_dequant_test.cpp LINKS nf_plugin_metal)
    nf_add_test(NAME nf_transformer_test   TEST_NAME transformer    SOURCES transformer_test.cpp   LINKS nf_plugin_metal)
    nf_add_test(NAME nf_multi_layer_test   TEST_NAME multi_layer    SOURCES multi_layer_test.cpp   LINKS nf_plugin_metal)
    nf_add_test(NAME nf_dag_inference_test TEST_NAME dag_inference  SOURCES dag_inference_test.cpp LINKS nf_plugin_metal)
    nf_add_test(NAME nf_real_model_test    TEST_NAME real_model     SOURCES real_model_test.cpp    LINKS nf_plugin_metal MODEL)
    nf_add_test(NAME nf_matmul_bench_test  TEST_NAME matmul_bench   SOURCES matmul_bench_test.cpp  LINKS nf_plugin_metal)
    nf_add_test(NAME nf_fp16_kernel_test   TEST_NAME fp16_kernel    SOURCES fp16_kernel_test.cpp   LINKS nf_plugin_metal)
    nf_add_test(NAME nf_fp16_e2e_test      TEST_NAME fp16_e2e       SOURCES fp16_e2e_test.cpp      LINKS nf_plugin_metal MODEL)

    # Phase 28: F16 vs F32 micro-benchmark (built, NOT added to ctest)
    nf_add_test(NAME nf_benchmark_kernels  SOURCES benchmark_kernels.cpp LINKS nf_plugin_metal)

    nf_add_test(NAME nf_fused_matmul_test  TEST_NAME fused_matmul   SOURCES fused_matmul_test.cpp  LINKS nf_plugin_metal)
    nf_add_test(NAME nf_sliding_window_test TEST_NAME sliding_window SOURCES sliding_window_test.cpp LINKS nf_plugin_metal)
    nf_add_test(NAME nf_fused_warmup_test  TEST_NAME fused_warmup   SOURCES fused_warmup_test.cpp  LINKS nf_plugin_metal MODEL)
    nf_add_test(NAME nf_paged_attn_test    TEST_NAME paged_attn     SOURCES paged_attn_test.cpp    LINKS nf_plugin_metal)
endif()

# ------------------------------------------------------------------
# RKNN-dependent tests
# ------------------------------------------------------------------
if(NF_PLUGIN_RKNN)
    nf_add_test(NAME nf_rknn_silicon_test TEST_NAME rknn_silicon SOURCES rknn_silicon_test.cpp
        GRAPH LINKS nf_plugin_rknn)
    if(NF_HAS_RKNN_SDK)
        target_compile_definitions(nf_rknn_silicon_test PRIVATE NF_HAS_RKNN_SDK)
        target_link_libraries(nf_rknn_silicon_test PRIVATE ${RKNN_RT})
    endif()
endif()
