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
# ddi — ABI / DDI / pipeline tests
# ------------------------------------------------------------------
nf_add_test(NAME nf_smoke_test          TEST_NAME smoke            SOURCES ddi/smoke_test.cpp)
nf_add_test(NAME nf_buffer_test         TEST_NAME buffer           SOURCES ddi/buffer_test.cpp)
nf_add_test(NAME nf_e2e_test            TEST_NAME e2e              SOURCES ddi/e2e_pipeline_test.cpp)
nf_add_test(NAME nf_split_inference_test TEST_NAME split_inference  SOURCES ddi/split_inference_test.cpp)
nf_add_test(NAME nf_streaming_callback_test TEST_NAME streaming_callback SOURCES ddi/streaming_callback_test.cpp)
nf_add_test(NAME nf_ddi_async_test        TEST_NAME ddi_async         SOURCES ddi/ddi_async_test.cpp)
nf_add_test(NAME nf_ddi_pipeline_async_test TEST_NAME ddi_pipeline_async SOURCES ddi/ddi_pipeline_async_test.cpp)

# GRAPH tests (need mmap_buffer + GraphBuilder objects)
nf_add_test(NAME nf_ir_loader_test       TEST_NAME ir_loader        SOURCES ddi/ir_loader_test.cpp       GRAPH)
nf_add_test(NAME nf_nfir_e2e_test        TEST_NAME nfir_e2e         SOURCES ddi/nfir_e2e_test.cpp        GRAPH)
nf_add_test(NAME nf_split_llama_mock_test TEST_NAME split_llama_mock SOURCES ddi/split_llama_mock_test.cpp GRAPH)

# kernel — scheduler / engine tests
nf_add_test(NAME nf_scheduler_test      TEST_NAME scheduler        SOURCES kernel/scheduler_test.cpp)
nf_add_test(NAME nf_context_hub_test     TEST_NAME context_hub     SOURCES kernel/context_hub_test.cpp)
nf_add_test(NAME nf_profiling_test       TEST_NAME profiling       SOURCES kernel/profiling_test.cpp)
nf_add_test(NAME nf_session_test         TEST_NAME session          SOURCES kernel/session_test.cpp         GRAPH)
nf_add_test(NAME nf_push_constants_test  TEST_NAME push_constants   SOURCES kernel/push_constants_test.cpp  GRAPH)

# model — model layer tests
nf_add_test(NAME nf_tokenizer_test       TEST_NAME tokenizer        SOURCES model/tokenizer_test.cpp       MODEL)
nf_add_test(NAME nf_arch_registry_test   TEST_NAME arch_registry    SOURCES model/arch_registry_test.cpp   MODEL)
nf_add_test(NAME nf_model_config_test    TEST_NAME model_config     SOURCES model/model_config_test.cpp    MODEL)
nf_add_test(NAME nf_paged_kv_test        TEST_NAME paged_kv         SOURCES kernel/paged_kv_test.cpp        MODEL)
nf_add_test(NAME nf_batch_scheduler_test TEST_NAME batch_scheduler  SOURCES kernel/batch_scheduler_test.cpp MODEL)
nf_add_test(NAME nf_speculative_decode_test TEST_NAME speculative_decode SOURCES kernel/speculative_decode_test.cpp MODEL)
nf_add_test(NAME nf_chrome_trace_test    TEST_NAME chrome_trace     SOURCES model/chrome_trace_test.cpp    MODEL)
nf_add_test(NAME nf_qwen2_arch_test     TEST_NAME qwen2_arch       SOURCES model/qwen2_arch_test.cpp      MODEL)
nf_add_test(NAME nf_gemma_arch_test     TEST_NAME gemma_arch       SOURCES model/gemma_arch_test.cpp      MODEL)
nf_add_test(NAME nf_mixtral_arch_test   TEST_NAME mixtral_arch     SOURCES model/mixtral_arch_test.cpp    MODEL)

# kernel — moe routing
nf_add_test(NAME nf_moe_routing_test    TEST_NAME moe_routing      SOURCES kernel/moe_routing_test.cpp)

# ---- Standalone tests (no nf_core, custom includes) ----
add_executable(nf_pso_registry_test driver/pso_registry_test.cpp)
target_include_directories(nf_pso_registry_test PRIVATE ${CMAKE_SOURCE_DIR}/plugins/metal/src)
add_test(NAME pso_registry COMMAND nf_pso_registry_test)

add_executable(nf_kv_cache_policy_test driver/kv_cache_policy_test.cpp)
target_include_directories(nf_kv_cache_policy_test PRIVATE ${CMAKE_SOURCE_DIR}/model)
add_test(NAME kv_cache_policy COMMAND nf_kv_cache_policy_test)

# ------------------------------------------------------------------
# kernel — NeuralOS vMMU / CFS / SpecEngine / VirtualBus tests
# ------------------------------------------------------------------

nf_add_test(NAME nf_vmmu_cow_test         TEST_NAME vmmu_cow          SOURCES kernel/vmmu_cow_test.cpp          MODEL
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)
nf_add_test(NAME nf_vmmu_pageout_test     TEST_NAME vmmu_pageout      SOURCES kernel/vmmu_pageout_test.cpp      MODEL
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)
nf_add_test(NAME nf_vmmu_radix_share_test TEST_NAME vmmu_radix_share  SOURCES kernel/vmmu_radix_share_test.cpp  MODEL
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)

nf_add_test(NAME nf_cfs_vtc_test          TEST_NAME cfs_vtc           SOURCES kernel/cfs_vtc_test.cpp           MODEL
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)
nf_add_test(NAME nf_cfs_preempt_test      TEST_NAME cfs_preempt       SOURCES kernel/cfs_preempt_test.cpp       MODEL
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)

nf_add_test(NAME nf_spec_tree_test        TEST_NAME spec_tree         SOURCES kernel/spec_tree_test.cpp         MODEL
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)
nf_add_test(NAME nf_spec_rollback_test    TEST_NAME spec_rollback     SOURCES kernel/spec_rollback_test.cpp     MODEL
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)

nf_add_test(NAME nf_virtual_bus_test      TEST_NAME virtual_bus       SOURCES kernel/virtual_bus_test.cpp
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)

nf_add_test(NAME nf_cfs_vmmu_test         TEST_NAME cfs_vmmu          SOURCES kernel/cfs_vmmu_integration_test.cpp  MODEL
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)
nf_add_test(NAME nf_spec_vmmu_test        TEST_NAME spec_vmmu         SOURCES kernel/spec_vmmu_integration_test.cpp MODEL
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)
nf_add_test(NAME nf_cfs_migration_test    TEST_NAME cfs_migration     SOURCES kernel/cfs_migration_test.cpp         MODEL
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)

# ------------------------------------------------------------------
# kernel — Phase 43: NeuralOSRuntime / BatchInferenceLoop tests
# ------------------------------------------------------------------

nf_add_test(NAME nf_runtime_basic_test     TEST_NAME runtime_basic      SOURCES kernel/runtime_basic_test.cpp      MODEL
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)
nf_add_test(NAME nf_runtime_pressure_test  TEST_NAME runtime_pressure   SOURCES kernel/runtime_pressure_test.cpp   MODEL
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)
nf_add_test(NAME nf_runtime_spec_test      TEST_NAME runtime_spec       SOURCES kernel/runtime_spec_test.cpp       MODEL
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)
nf_add_test(NAME nf_runtime_prefix_test    TEST_NAME runtime_prefix     SOURCES kernel/runtime_prefix_test.cpp     MODEL
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)
nf_add_test(NAME nf_batch_loop_test        TEST_NAME batch_loop         SOURCES kernel/batch_loop_test.cpp         MODEL
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)
nf_add_test(NAME nf_batch_loop_pressure_test TEST_NAME batch_loop_pressure SOURCES kernel/batch_loop_pressure_test.cpp MODEL
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)
nf_add_test(NAME nf_generate_neuralOS_test TEST_NAME generate_neuralOS  SOURCES kernel/generate_neuralOS_test.cpp  MODEL
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)
nf_add_test(NAME nf_serve_neuralOS_test    TEST_NAME serve_neuralOS     SOURCES kernel/serve_neuralOS_test.cpp     MODEL
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)

# ------------------------------------------------------------------
# compiler — NFIR / fusion / memory plan / ONNX tests
# ------------------------------------------------------------------

nf_add_test(NAME nf_nfir_multilevel_test  TEST_NAME nfir_multilevel   SOURCES compiler/nfir_multilevel_test.cpp
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)

nf_add_test(NAME nf_fusion_pass_test      TEST_NAME fusion_pass       SOURCES compiler/fusion_pass_test.cpp
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)

nf_add_test(NAME nf_memory_plan_test      TEST_NAME memory_plan       SOURCES compiler/memory_plan_test.cpp
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)

nf_add_test(NAME nf_onnx_parser_test      TEST_NAME onnx_parser       SOURCES compiler/onnx_parser_test.cpp
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include ${CMAKE_SOURCE_DIR}/model/nf_compiler)

nf_add_test(NAME nf_dce_pass_test         TEST_NAME dce_pass          SOURCES compiler/dce_pass_test.cpp
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)
nf_add_test(NAME nf_cse_pass_test         TEST_NAME cse_pass          SOURCES compiler/cse_pass_test.cpp
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)
nf_add_test(NAME nf_chain_fusion_test     TEST_NAME chain_fusion      SOURCES compiler/chain_fusion_test.cpp
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)
nf_add_test(NAME nf_shape_inference_test  TEST_NAME shape_inference   SOURCES compiler/shape_inference_test.cpp
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)

nf_add_test(NAME nf_compiler_pipeline_test TEST_NAME compiler_pipeline SOURCES compiler/compiler_pipeline_test.cpp
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)
nf_add_test(NAME nf_app_compiler_test      TEST_NAME app_compiler      SOURCES compiler/app_compiler_test.cpp
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)

# ------------------------------------------------------------------
# driver — transport abstraction test
# ------------------------------------------------------------------

add_executable(nf_transport_abstraction_test driver/transport_abstraction_test.cpp)
target_include_directories(nf_transport_abstraction_test PRIVATE
    ${CMAKE_SOURCE_DIR}/plugins/network/src
    ${CMAKE_SOURCE_DIR}/core/include)
add_test(NAME transport_abstraction COMMAND nf_transport_abstraction_test)

nf_add_test(NAME nf_driver_registry_populate_test TEST_NAME driver_registry_populate
    SOURCES driver/driver_registry_populate_test.cpp
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)

# ------------------------------------------------------------------
# mesh — topology / mesh / dataflow / migration / CXL tests
# ------------------------------------------------------------------

nf_add_test(NAME nf_topology_routing_test TEST_NAME topology_routing  SOURCES mesh/topology_routing_test.cpp
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)

nf_add_test(NAME nf_mesh_coordinator_test TEST_NAME mesh_coordinator  SOURCES mesh/mesh_coordinator_test.cpp
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)

nf_add_test(NAME nf_async_dataflow_test   TEST_NAME async_dataflow    SOURCES mesh/async_dataflow_test.cpp
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)

nf_add_test(NAME nf_kv_migration_test     TEST_NAME kv_migration      SOURCES mesh/kv_migration_test.cpp      MODEL
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)

nf_add_test(NAME nf_cxl_memory_test       TEST_NAME cxl_memory        SOURCES mesh/cxl_memory_test.cpp
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)

nf_add_test(NAME nf_dataplane_transport_test TEST_NAME dataplane_transport SOURCES mesh/dataplane_transport_test.cpp
    INCLUDES ${CMAKE_SOURCE_DIR}/core/include)

# ------------------------------------------------------------------
# driver — Metal-dependent tests
# ------------------------------------------------------------------
if(NF_PLUGIN_METAL)
    nf_add_test(NAME nf_mistral_arch_test  TEST_NAME mistral_arch   SOURCES model/mistral_arch_test.cpp  LINKS nf_plugin_metal MODEL)
    nf_add_test(NAME nf_phi3_arch_test     TEST_NAME phi3_arch      SOURCES model/phi3_arch_test.cpp     LINKS nf_plugin_metal MODEL)
    nf_add_test(NAME nf_silicon_test       TEST_NAME silicon        SOURCES driver/silicon_test.cpp       LINKS nf_plugin_metal)
    nf_add_test(NAME nf_dequant_test       TEST_NAME dequant        SOURCES driver/dequant_test.cpp       LINKS nf_plugin_metal)
    nf_add_test(NAME nf_kquant_dequant_test TEST_NAME kquant_dequant SOURCES driver/kquant_dequant_test.cpp LINKS nf_plugin_metal)
    nf_add_test(NAME nf_transformer_test   TEST_NAME transformer    SOURCES driver/transformer_test.cpp   LINKS nf_plugin_metal)
    nf_add_test(NAME nf_multi_layer_test   TEST_NAME multi_layer    SOURCES driver/multi_layer_test.cpp   LINKS nf_plugin_metal)
    nf_add_test(NAME nf_dag_inference_test TEST_NAME dag_inference  SOURCES driver/dag_inference_test.cpp LINKS nf_plugin_metal)
    nf_add_test(NAME nf_real_model_test    TEST_NAME real_model     SOURCES driver/real_model_test.cpp    LINKS nf_plugin_metal MODEL)
    nf_add_test(NAME nf_matmul_bench_test  TEST_NAME matmul_bench   SOURCES driver/matmul_bench_test.cpp  LINKS nf_plugin_metal)
    nf_add_test(NAME nf_fp16_kernel_test   TEST_NAME fp16_kernel    SOURCES driver/fp16_kernel_test.cpp   LINKS nf_plugin_metal)
    nf_add_test(NAME nf_fp16_e2e_test      TEST_NAME fp16_e2e       SOURCES driver/fp16_e2e_test.cpp      LINKS nf_plugin_metal MODEL)

    # F16 vs F32 micro-benchmark (built, NOT added to ctest)
    nf_add_test(NAME nf_benchmark_kernels  SOURCES driver/benchmark_kernels.cpp LINKS nf_plugin_metal)

    nf_add_test(NAME nf_fused_matmul_test  TEST_NAME fused_matmul   SOURCES driver/fused_matmul_test.cpp  LINKS nf_plugin_metal)
    nf_add_test(NAME nf_sliding_window_test TEST_NAME sliding_window SOURCES driver/sliding_window_test.cpp LINKS nf_plugin_metal)
    nf_add_test(NAME nf_fused_warmup_test  TEST_NAME fused_warmup   SOURCES driver/fused_warmup_test.cpp  LINKS nf_plugin_metal MODEL)
    nf_add_test(NAME nf_paged_attn_test    TEST_NAME paged_attn     SOURCES driver/paged_attn_test.cpp    LINKS nf_plugin_metal)

    nf_add_test(NAME nf_gqa_attention_test  TEST_NAME gqa_attention  SOURCES driver/gqa_attention_test.cpp  LINKS nf_plugin_metal)
endif()

# ------------------------------------------------------------------
# driver — RKNN-dependent tests
# ------------------------------------------------------------------
if(NF_PLUGIN_RKNN)
    nf_add_test(NAME nf_rknn_silicon_test TEST_NAME rknn_silicon SOURCES driver/rknn_silicon_test.cpp
        GRAPH LINKS nf_plugin_rknn)
    if(NF_HAS_RKNN_SDK)
        target_compile_definitions(nf_rknn_silicon_test PRIVATE NF_HAS_RKNN_SDK)
        target_link_libraries(nf_rknn_silicon_test PRIVATE ${RKNN_RT})
    endif()
endif()
