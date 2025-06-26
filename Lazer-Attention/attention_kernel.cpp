#include "kernel_operator.h"
#include "cube_operations.h"
#include "vector_operations.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void ascend_laser_attention(
        __gm__ uint8_t *__restrict__ q_gm,
__gm__ uint8_t *__restrict__ k_gm,
__gm__ uint8_t *__restrict__ v_gm,
__gm__ uint8_t *__restrict__ atten_mask_gm,
__gm__ uint8_t *__restrict__ alibi_mask_gm,
__gm__ uint8_t *__restrict__ drop_mask_gm,
__gm__ uint8_t *__restrict__ softmax_log_max_sum_gm,
__gm__ uint8_t *__restrict__ attention_out_gm,
__gm__ uint8_t *__restrict__ workspace,
__gm__ uint8_t *__restrict__ tiling_para_gm)
{
GET_TILING_DATA(tiling_data_in, tiling_para_gm);
const AscendLaserAttentionTilingData *__restrict tiling_data = &tiling_data_in;
SetSysWorkspace(workspace);
__gm__ uint8_t *user = GetUserWorkspace(workspace);

// --- Extract Tiling Parameters ---
int32_t Y = tiling_data->coreNumPerGroup;
int32_t F = tiling_data->coreGroupNum;
int32_t B = tiling_data->batchSize;
int32_t N = tiling_data->headNum;
int32_t S1 = tiling_data->qSeqLength;
int32_t S2 = tiling_data->kSeqLength;
int32_t D = tiling_data->headDim;
int32_t G = tiling_data->headGroupSize;
int32_t qk_triangle = tiling_data->isTriangle;
int32_t sparseMode = tiling_data->sparseMode;
int32_t windowLen = tiling_data->windowLen;
bool isHighPrecision = true; // Assuming this might be a tiling parameter too
int32_t maskSeqLength = tiling_data->maskSeqLength;

// --- Workspace Allocation ---
__gm__ float *__restrict__ c_cube2 = (__gm__ float *__restrict__)attention_out_gm;
__gm__ float *__restrict__ log_sum_gm = (__gm__ float *__restrict__)softmax_log_max_sum_gm;
__gm__ uint8_t *__restrict__ const_ones_gm = (__gm__ uint8_t *__restrict__)user;
__gm__ float *__restrict__ const_zero_gm = (__gm__ float *__restrict__)(const_ones_gm + 128 * 128 * 2);
__gm__ float *__restrict__ gm_rowsum = (__gm__ float *__restrict__)(const_zero_gm + 32 * 128);
__gm__ uint8_t *__restrict__ score_gm = (__gm__ uint8_t *__restrict__)(gm_rowsum + B * N * S1);

// --- Kernel Dispatch ---
if (TILING_KEY_IS(0)) // FP16
{
#ifdef __DAV_C220_CUBE__
CubeForward<half, false, float> op;
            op.Init(q_gm, k_gm, v_gm, score_gm, c_cube2, const_ones_gm, gm_rowsum, Y, F, B, N, S1, S2, D, G, qk_triangle, sparseMode, windowLen);
            op.SetHighPrecision(isHighPrecision);
            op.Run();
#elif __DAV_C220_VEC__
VectorForward<half, false, float> op;
            op.Init(q_gm, k_gm, v_gm, atten_mask_gm, const_ones_gm, const_zero_gm, score_gm, c_cube2, log_sum_gm, gm_rowsum, S1, S2, N, B, Y, qk_triangle, windowLen / BASE_BLOCK_SIDE_LEN, maskSeqLength);
            op.SetHighPrecision(isHighPrecision);
            op.Run();
#endif
}
else if (TILING_KEY_IS(1)) // BF16
{
#ifdef __DAV_C220_CUBE__
CubeForward<__bf16, true, float> op;
            op.Init(q_gm, k_gm, v_gm, score_gm, c_cube2, const_ones_gm, gm_rowsum, Y, F, B, N, S1, S2, D, G, qk_triangle, sparseMode, windowLen);
            op.SetHighPrecision(isHighPrecision);
            op.Run();
#elif __DAV_C220_VEC__
VectorForward<__bf16, true, float> op;
            op.Init(q_gm, k_gm, v_gm, atten_mask_gm, const_ones_gm, const_zero_gm, score_gm, c_cube2, log_sum_gm, gm_rowsum, S1, S2, N, B, Y, qk_triangle, windowLen / BASE_BLOCK_SIDE_LEN, maskSeqLength);
            op.SetHighPrecision(isHighPrecision);
            op.Run();
#endif
}
}