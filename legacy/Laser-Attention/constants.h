#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <stdint.h>
#include <math.h>

// Ascend CCE
using bf16 = __bf16;
using fp32 = float;
using half = __fp16;

// Flag & Enum
constexpr int AICFLAGID      = 0;
constexpr int AIVFLAGID      = 1;
constexpr int AIC2AIVFLAGID  = 2;
constexpr int AIV2AICFLAGID  = 3;

constexpr int TRI_MATRIX_NONE           = 0;
constexpr int TRI_MATRIX_TAIL           = 1;
constexpr int TRI_MATRIX_HEAD           = 2;
constexpr int TRI_MATRIX_HEAD_AND_TAIL  = 3;

// Data Types 输出/工作区精度
using T_OUTPUT        = fp32;   // 归一化后输出
using T_DATA_TYPE     = bf16;   // 中间 FP16
using WORKSPACE_DTYPE = bf16;   // Scores

// Buffer 元素数
// 以“元素数”为单位，具体字节 = 元素数 * sizeof(dtype)
constexpr int32_t L0AB_PINGPONG_BUFFER_LEN = 16384; // dtype=bf16 => 32KB
constexpr int64_t L1_PINGPONG_BUFFER_LEN   = 16384; // dtype=bf16 => 32KB
constexpr int64_t L0C_PINGPONG_BUFFER_LEN  = 16384; // dtype=fp32 => 64KB

// Block & Matrix Sizes
constexpr int32_t BLOCK_SIZE           = 16;
constexpr int32_t CUBE_MATRIX_SIZE     = BLOCK_SIZE * BLOCK_SIZE; // 256
constexpr int32_t BASE_BLOCK_SIDE_LEN  = 128;
constexpr int32_t HEAD_DIM_DEFAULT     = 128;

constexpr int32_t BASE_BLOCK_DATA_NUM  = BASE_BLOCK_SIDE_LEN * BASE_BLOCK_SIDE_LEN;
constexpr int32_t BASE_BLOCK_SIZE      = 128 * 128;   // == BASE_BLOCK_DATA_NUM
constexpr int32_t BASE_BLOCK_SIZE_192  = 128 * 192;
constexpr int32_t BASE_BLOCK_SIZE_256  = 128 * 256;

// Vector Processing
constexpr int MAX_LENG_PER_UB_PROC     = 8192;
constexpr int MAX_BLOCK_PER_ONE_PROC   = MAX_LENG_PER_UB_PROC / BASE_BLOCK_SIDE_LEN;
constexpr int BLOCK_NUM_FOR_VMAX       = 16;
constexpr int SHORT_SEQ_THRESHOLD      = 8192;
constexpr int MDDIUM_SEQ_THRESHOLD     = 32768;

// Math
constexpr float PADDING_FOR_MAX = -1e30f;

// 动态 scale：按 head_dim 计算 1/sqrt(dk)
__aicore__ __inline__ static float attn_scale_from_D(int D) {
    return 1.0f / sqrtf(static_cast<float>(D));
}

// 物理地址结构
namespace Address {
    template <typename TYPE>
    struct Phy_Addr {
        __gm__ TYPE *left;
        __gm__ TYPE *right;
        __gm__ TYPE *out;
        int32_t k = 0;
    };

    template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
    struct Phy_Addr_forward_cube1 {
        __gm__ T_LEFT  *left;   // Q block
        __gm__ T_RIGHT *right;  // K^T block
        __gm__ T_OUTPUT*out;    // Scores block
        int32_t k = 0;          // k-tile idx
        // 扩展：ld* stride
    };

    template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
    struct Phy_Addr_forward_cube2_rowsum {
        __gm__ T_LEFT  *left;        // Scores block
        __gm__ T_RIGHT *right;       // V block
        __gm__ T_OUTPUT*out;         // O (未归一化) block
        __gm__ T_OUTPUT*rowsum_out;  // rowsum（fp32）
        int32_t k = 0;
    };
} // namespace Address

#endif // CONSTANTS_H