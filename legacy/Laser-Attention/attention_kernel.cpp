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

    // 提取 tiling
    int32_t Y = tiling_data->coreNumPerGroup;
    int32_t F = tiling_data->coreGroupNum;
    int32_t B = tiling_data->batchSize;
    int32_t N = tiling_data->headNum;
    int32_t S1 = tiling_data->qSeqLength;
    int32_t S2 = tiling_data->kSeqLength;
    int32_t D  = tiling_data->headDim;
    int32_t G  = tiling_data->headGroupSize;
    int32_t qk_triangle = tiling_data->isTriangle;
    int32_t sparseMode  = tiling_data->sparseMode;
    int32_t windowLen   = tiling_data->windowLen;
    bool    isHighPrecision = true;
    int32_t maskSeqLength = tiling_data->maskSeqLength;

    // 动态 scale
    float scale = attn_scale_from_D(D);

    // Workspace 切片：按实际尺寸计算
    // [常量 ones (bf16) : 128*128] |
    // [rowsum (fp32) : B*N*S1]     |
    // [scores (bf16/half) : B*N*S1*S2]  —— 若 Vector 阶段需要 fp32，请改为 fp32
    // [logsum (fp32) : B*N*S1]

    size_t off = 0;

    __gm__ T_DATA_TYPE *__restrict__ const_ones_gm =
        reinterpret_cast<__gm__ T_DATA_TYPE*>(user + off);
    off += BASE_BLOCK_SIZE * sizeof(T_DATA_TYPE); // 128x128

    __gm__ float *__restrict__ gm_rowsum =
        reinterpret_cast<__gm__ float*>(user + off);
    off += static_cast<size_t>(B) * N * S1 * sizeof(float);

    __gm__ WORKSPACE_DTYPE *__restrict__ score_gm =
        reinterpret_cast<__gm__ WORKSPACE_DTYPE*>(user + off);
    off += static_cast<size_t>(B) * N * S1 * S2 * sizeof(WORKSPACE_DTYPE);

    // 输出/对数和
    __gm__ float *__restrict__ c_cube2 = reinterpret_cast<__gm__ float*>(attention_out_gm);
    __gm__ float *__restrict__ log_sum_gm = reinterpret_cast<__gm__ float*>(softmax_log_max_sum_gm);

    // 判断 off 未超过 userworkspace 容量

    // Kernel Dispatch：CUBE / VECTOR 分开
    if (TILING_KEY_IS(0)) // FP16
    {
    #ifdef __DAV_C220_CUBE__
        CubeForward<half, false, WORKSPACE_DTYPE> op;
        op.Init(q_gm, k_gm, v_gm, reinterpret_cast<__gm__ uint8_t*>(score_gm), c_cube2,
                reinterpret_cast<__gm__ uint8_t*>(const_ones_gm), gm_rowsum,
                Y, F, B, N, S1, S2, D, G, qk_triangle, sparseMode, windowLen);
        op.SetHighPrecision(isHighPrecision);
        op.Run();
    #elif defined(__DAV_C220_VEC__)
        VectorForward<half, false, WORKSPACE_DTYPE> op;
        op.Init(q_gm, k_gm, v_gm,
                atten_mask_gm, reinterpret_cast<__gm__ uint8_t*>(const_ones_gm),
                nullptr,
                reinterpret_cast<__gm__ uint8_t*>(score_gm), c_cube2, log_sum_gm, gm_rowsum,
                S1, S2, N, B, Y, qk_triangle, windowLen / BASE_BLOCK_SIDE_LEN, maskSeqLength);
        op.SetHighPrecision(isHighPrecision);
        op.SetScale(scale);
        op.SetHeadDim(D);
        op.Run();
    #endif
    }
    else if (TILING_KEY_IS(1)) // BF16
    {
    #ifdef __DAV_C220_CUBE__
        CubeForward<bf16, true, WORKSPACE_DTYPE> op;
        op.Init(q_gm, k_gm, v_gm, reinterpret_cast<__gm__ uint8_t*>(score_gm), c_cube2,
                reinterpret_cast<__gm__ uint8_t*>(const_ones_gm), gm_rowsum,
                Y, F, B, N, S1, S2, D, G, qk_triangle, sparseMode, windowLen);
        op.SetHighPrecision(isHighPrecision);
        op.Run();
    #elif defined(__DAV_C220_VEC__)
        VectorForward<bf16, true, WORKSPACE_DTYPE> op;
        op.Init(q_gm, k_gm, v_gm,
                atten_mask_gm, reinterpret_cast<__gm__ uint8_t*>(const_ones_gm),
                nullptr,
                reinterpret_cast<__gm__ uint8_t*>(score_gm), c_cube2, log_sum_gm, gm_rowsum,
                S1, S2, N, B, Y, qk_triangle, windowLen / BASE_BLOCK_SIDE_LEN, maskSeqLength);
        op.SetHighPrecision(isHighPrecision);
        op.SetScale(scale);
        op.SetHeadDim(D);
        op.Run();
    #endif
    }
}