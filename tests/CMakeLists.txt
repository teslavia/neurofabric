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
