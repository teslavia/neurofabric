/**
 * @file nf_c_api.h
 * @brief Neuro-Fabric C API — Python FFI Gateway
 *
 * Pure C header. Opaque handles, zero C++ leakage.
 * This is the sole exported symbol surface for external consumers.
 */

#ifndef NF_C_API_H
#define NF_C_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Export Macro                                                        */
/* ------------------------------------------------------------------ */

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef NF_C_API_BUILDING
    #define NF_C_API_EXPORT __declspec(dllexport)
  #else
    #define NF_C_API_EXPORT __declspec(dllimport)
  #endif
#elif defined(__GNUC__) || defined(__clang__)
  #define NF_C_API_EXPORT __attribute__((visibility("default")))
#else
  #define NF_C_API_EXPORT
#endif

/* ------------------------------------------------------------------ */
/*  Opaque Handles                                                     */
/* ------------------------------------------------------------------ */

typedef struct nf_engine_s*  nf_engine_t;
typedef struct nf_session_s* nf_session_t;

/* ------------------------------------------------------------------ */
/*  Engine Lifecycle                                                    */
/* ------------------------------------------------------------------ */

NF_C_API_EXPORT nf_engine_t  nf_create_engine(uint32_t n_threads);
NF_C_API_EXPORT void         nf_destroy_engine(nf_engine_t engine);

/* ------------------------------------------------------------------ */
/*  Session Lifecycle                                                   */
/* ------------------------------------------------------------------ */

NF_C_API_EXPORT nf_session_t nf_create_session(nf_engine_t engine,
                                                const char* nfir_path);
NF_C_API_EXPORT void         nf_destroy_session(nf_session_t session);

/* ------------------------------------------------------------------ */
/*  Session Operations                                                 */
/* ------------------------------------------------------------------ */

NF_C_API_EXPORT int nf_session_set_input(nf_session_t s,
                                          uint32_t tensor_id,
                                          const void* data,
                                          uint64_t size);

NF_C_API_EXPORT int nf_session_step(nf_session_t s);

NF_C_API_EXPORT int nf_session_get_output(nf_session_t s,
                                           uint32_t tensor_id,
                                           void* data,
                                           uint64_t size);

NF_C_API_EXPORT int nf_session_set_push_constants(nf_session_t s,
                                                    const char* node_name,
                                                    const void* data,
                                                    uint32_t size);

/* ------------------------------------------------------------------ */
/*  Session Queries                                                    */
/* ------------------------------------------------------------------ */

NF_C_API_EXPORT double   nf_session_get_last_step_us(nf_session_t s);
NF_C_API_EXPORT uint32_t nf_session_num_tensors(nf_session_t s);
NF_C_API_EXPORT uint32_t nf_session_num_nodes(nf_session_t s);

#ifdef __cplusplus
}
#endif

#endif /* NF_C_API_H */
