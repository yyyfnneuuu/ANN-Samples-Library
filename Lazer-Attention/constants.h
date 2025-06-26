#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <stdint.h>

// --- Flags and Enums ---
constexpr int AICFLAGID = 0;
constexpr int AIVFLAGID = 1;
constexpr int AIC2AIVFLAGID = 2;
constexpr int AIV2AICFLAGID = 3;

constexpr int TRI_MATRIX_NONE = 0;
constexpr int TRI_MATRIX_TAIL = 1;
constexpr int TRI_MATRIX_HEAD = 2;
constexpr int TRI_MATRIX_HEAD_AND_TAIL = 3;

// --- Data Types ---
using T_OUTPUT = float;
// using T_DATA_TYPE = __bf16;
// using WORKSPACE_DTYPE = __bf16;

// --- Memory Buffer Sizes ---
constexpr int32_t L0AB_PINGPONG_BUFFER_LEN = 16384; // 32 KB
constexpr int64_t L1_PINGPONG_BUFFER_LEN = 16384;  // 32 KB
constexpr int64_t L0C_PINGPONG_BUFFER_LEN = 16384; // 64 KB

// --- Block & Matrix Sizes ---
constexpr int32_t BLOCK_SIZE = 16;
constexpr int32_t CUBE_MATRIX_SIZE = 256;		   // 16 * 16
constexpr int32_t BASE_BLOCK_SIDE_LEN = 128;
constexpr int32_t HEAD_DIM = 128;
constexpr int32_t BASE_BLOCK_DATA_NUM = BASE_BLOCK_SIDE_LEN * BASE_BLOCK_SIDE_LEN;
constexpr int32_t BASE_BLOCK_SIZE_LEN_BACKWARD = 128;

// --- Specialized Block Sizes ---
constexpr int32_t BASE_BLOCK_SIZE = 16384;	   // [128 * 128]
constexpr int32_t BASE_BLOCK_SIZE_192 = 24576; // [128 * 192]
constexpr int32_t BASE_BLOCK_SIZE_256 = 32768; // [128 * 256]

// --- Vector Processing Constants ---
constexpr int MAX_LENG_PER_UB_PROC = 8192;
constexpr int MAX_BLOCK_PER_ONE_PROC = MAX_LENG_PER_UB_PROC / BASE_BLOCK_SIDE_LEN;
constexpr int BLOCK_NUM_FOR_VMAX = 16;
constexpr int SHORT_SEQ_THRESHOLD = 8192;
constexpr int MDDIUM_SEQ_THRESHOLD = 32768;

// --- Math Constants ---
constexpr float SCALE = 0.07216878364870323; // 1/sqrt(128)
constexpr float PADDING_FOR_MAX = -1e30;

// --- Physical Address Structs ---
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
        __gm__ T_LEFT *left;
        __gm__ T_RIGHT *right;
        __gm__ T_OUTPUT *out;
        int32_t k = 0;
    };

    template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
    struct Phy_Addr_forward_cube2_rowsum {
        __gm__ T_LEFT *left;
        __gm__ T_RIGHT *right;
        __gm__ T_OUTPUT *out;
        __gm__ T_OUTPUT *rowsum_out;
        int32_t k = 0;
    };
} // namespace Address

#endif // CONSTANTS_H