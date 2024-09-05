#ifndef GGML_COMMON_DECL

#if defined(GGML_COMMON_DECL_C)
#include <stdint.h>

typedef uint16_t ggml_half;
typedef uint32_t ggml_half2;

#define GGML_COMMON_AGGR

#define GGML_COMMON_DECL
#elif defined(GGML_COMMON_DECL_METAL)
#include <metal_stdlib>

typedef half  ggml_half;
typedef half2 ggml_half2;

#define GGML_COMMON_AGGR

#define GGML_COMMON_DECL
#elif defined(GGML_COMMON_DECL_CUDA)
#if defined(GGML_COMMON_DECL_MUSA)
#include <musa_fp16.h>
#else
#include <cuda_fp16.h>
#endif
#include <cstdint>

typedef half  ggml_half;
typedef half2 ggml_half2;

#define GGML_COMMON_AGGR data

#define GGML_COMMON_DECL
#elif defined(GGML_COMMON_DECL_HIP)
#include <hip/hip_fp16.h>
#include <cstdint>

typedef half  ggml_half;
typedef half2 ggml_half2;

#define GGML_COMMON_AGGR data

#define GGML_COMMON_DECL
#elif defined(GGML_COMMON_DECL_SYCL)
#include <sycl/half_type.hpp>
#include <cstdint>

typedef sycl::half  ggml_half;
typedef sycl::half2 ggml_half2;

#define GGML_COMMON_AGGR data

#define GGML_COMMON_DECL
#endif

#if defined(GGML_COMMON_DECL)

#ifndef __cplusplus
#ifndef static_assert
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
#define static_assert(cond, msg) _Static_assert(cond, msg)
#else
#define static_assert(cond, msg) struct global_scope_noop_trick
#endif
#endif
#endif // __cplusplus

// QK = number of values after dequantization
// QK_K = super-block size

#define QK_K 256
#define K_SCALE_SIZE 12

#if defined(GGML_COMMON_DECL_CUDA) || defined(GGML_COMMON_DECL_HIP) || defined(GGML_COMMON_DECL_SYCL)
// QR = QK / number of values before dequantization
// QI = number of 32 bit integers before dequantization

#define QI4_0 (QK4_0 / (4 * QR4_0))
#define QR4_0 2

#define QI4_1 (QK4_1 / (4 * QR4_1))
#define QR4_1 2

#define QI5_0 (QK5_0 / (4 * QR5_0))
#define QR5_0 2

#define QI5_1 (QK5_1 / (4 * QR5_1))
#define QR5_1 2

#define QI8_0 (QK8_0 / (4 * QR8_0))
#define QR8_0 1

#define QI8_1 (QK8_1 / (4 * QR8_1))
#define QR8_1 1

#define QI2_K (QK_K / (4*QR2_K))
#define QR2_K 4

#define QI3_K (QK_K / (4*QR3_K))
#define QR3_K 4

#define QI4_K (QK_K / (4*QR4_K))
#define QR4_K 2

#define QI5_K (QK_K / (4*QR5_K))
#define QR5_K 2

#define QI6_K (QK_K / (4*QR6_K))
#define QR6_K 2

#define QI2_XXS (QK_K / (4*QR2_XXS))
#define QR2_XXS 4

#define QI2_XS (QK_K / (4*QR2_XS))
#define QR2_XS 4

#define QI2_S (QK_K / (4*QR2_S))
#define QR2_S 4

#define QI3_XXS (QK_K / (4*QR3_XXS))
#define QR3_XXS 4

#define QI3_XS (QK_K / (4*QR3_XS))
#define QR3_XS 4

#define QI1_S (QK_K / (4*QR1_S))
#define QR1_S 8

#define QI1_M (QK_K / (4*QR1_M))
#define QR1_M 8

#define QI4_NL (QK4_NL / (4*QR4_NL))
#define QR4_NL 2

#define QI4_XS (QK_K / (4*QR4_XS))
#define QR4_XS 2

#define QI3_S (QK_K / (4*QR3_S))
#define QR3_S 4

#endif // GGML_COMMON_DECL_CUDA || GGML_COMMON_DECL_HIP

#define QK4_0 32
typedef struct {
    ggml_half d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2, "wrong q4_0 block size/padding");

// [Line:186]
#define QK8_0 32
typedef struct {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0, "wrong q8_0 block size/padding");

// [Line:218]
typedef struct {
    ggml_half d[4];        // deltas for 4 q8_0 blocks
    int8_t qs[QK8_0 * 4];  // quants for 4 q8_0 blocks
} block_q8_0x4;
static_assert(sizeof(block_q8_0x4) == 4 * sizeof(ggml_half) + QK8_0 * 4, "wrong q8_0x4 block size/padding");

// [Line:401]
#endif // GGML_COMMON_DECL
#endif // GGML_COMMON_DECL

////////////////////////////////////////////////////////////////////////////////
