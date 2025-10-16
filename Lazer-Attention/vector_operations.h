#ifndef VECTOR_OPERATIONS_H
#define VECTOR_OPERATIONS_H

#include "constants.h"
#include "kernel_operator.h"

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
class VectorForward {
public:
    __aicore__ inline VectorForward() : high_precision_(true), scale_(1.f) {}

    __aicore__ inline void Init(
        __gm__ uint8_t *__restrict__ a_cube1,   // Q
        __gm__ uint8_t *__restrict__ b_cube1,   // K
        __gm__ uint8_t *__restrict__ b_cube2,   // V
        __gm__ uint8_t *__restrict__ mask_gm,   // attention mask
        __gm__ uint8_t *__restrict__ ones_gm,   // 常量 1（bf16/half）
        __gm__ float   *__restrict__ zeros_gm,  // 可选的 0 常量区
        __gm__ uint8_t *__restrict__ score_gm,  // Scores（bf16/half）
        __gm__ float   *__restrict__ c_cube2,   // O（fp32，未归一化）
        __gm__ float   *__restrict__ log_sum_gm,// logsum（fp32）
        __gm__ float   *__restrict__ gm_rowsum, // rowsum（fp32）
        int32_t qSeqLength, int32_t kSeqLength,
        int32_t H, int32_t B, int32_t Y,
        int32_t qk_triangle, int32_t windows_block_num, int32_t maskSeqLength)
    {
        (void)a_cube1; (void)b_cube1; (void)b_cube2; (void)zeros_gm;
        gm_scores_   = reinterpret_cast<__gm__ WORKSPACE_T*>(score_gm);
        gm_o_        = c_cube2;
        gm_rowsum_   = gm_rowsum;
        gm_logsum_   = log_sum_gm;
        gm_mask_     = reinterpret_cast<__gm__ INPUT_T*>(mask_gm);
        gm_ones_     = reinterpret_cast<__gm__ INPUT_T*>(ones_gm);

        S1_ = qSeqLength; S2_ = kSeqLength;
        N_  = H; B_ = B; Y_ = Y;
        qk_triangle_ = qk_triangle;
        window_blk_num_ = windows_block_num;
        mask_seq_len_   = maskSeqLength;

        // head_dim 需要从外部传入或通过tiling
    }

    __aicore__ inline void SetHighPrecision(bool on) { high_precision_ = on; }
    __aicore__ inline void SetScale(float s) { scale_ = s; }

    __aicore__ inline void Run()
    {
        // 稳定 Softmax 未归一化，产出 rowsum 与 logsum
        // 目标：对 gm_scores_（已是 QKᵀ）进行：
        // 1) scaled = scores * scale_
        // 2) apply mask（-inf 或大负数填充）
        // 3) rowwise max
        // 4) exp(scores - max)
        // 5) rowsum = sum(exp)
        // 6) log_sum = log(rowsum) + max（供反向）
        //
        // 分块（UB/L1/L0 pipeline）

        int64_t total_heads = B_ * N_;
        for (int64_t bh = 0; bh < total_heads; ++bh) {
            // 起点
            __gm__ WORKSPACE_T* scores_row = gm_scores_ + (bh * S1_ * S2_);
            __gm__ float*       rowsum_row = gm_rowsum_ + (bh * S1_);
            __gm__ float*       logsum_row = gm_logsum_ + (bh * S1_);

            for (int64_t r = 0; r < S1_; ++r) {
                // (1)(2) scale + mask；(3) 求 max
                float row_max = -INFINITY;
                // 向量化加载 + 掩码应用
                for (int64_t c = 0; c < S2_; ++c) {
                    // 读 bf16/half 到 fp32
                    float v = static_cast<float>(scores_row[r*S2_ + c]);
                    v = v * scale_;
                    // mask
                    if (gm_mask_) {
                        v = gm_mask_[bh*S1_*S2_ + r*S2_ + c] == 0 ? -INFINITY : v;
                    }
                    if (v > row_max) row_max = v;
                    // 回写临时：这里直接复用 gm_scores_ 存 fp32 结果并不合适（dtype 不同）
                }

                // (4)(5) exp & rowsum
                float sumv = 0.f;
                for (int64_t c = 0; c < S2_; ++c) {
                    float v = static_cast<float>(scores_row[r*S2_ + c]);
                    v = v * scale_;
                    float e = expf(v - row_max);
                    sumv += e;
                    //Softmax 未归一化= e^(scaled - max)）
                }
                rowsum_row[r] = sumv;
                logsum_row[r] = logf(sumv) + row_max;
            }
        }

        // 阶段四：归一化 O / rowsum
        // O 由 Cube 阶段的融合路径写入 gm_o_（未归一化），逐行 O /= rowsum
        for (int64_t bh = 0; bh < total_heads; ++bh) {
            __gm__ float* o_row     = gm_o_      + (bh * S1_ * D_stub_); // 注意：需要 head_dim，这里先用占位
            __gm__ float* rowsum_row= gm_rowsum_ + (bh * S1_);
            for (int64_t r = 0; r < S1_; ++r) {
                float rs = rowsum_row[r] + 1e-6f;
                for (int64_t d = 0; d < D_stub_; ++d) {
                    o_row[r*D_stub_ + d] /= rs;
                }
            }
        }
    }
    __aicore__ inline void SetHeadDim(int D) { D_stub_ = D; }

private:
    // 形参/状态
    int32_t S1_, S2_, N_, B_, Y_;
    int32_t qk_triangle_, window_blk_num_, mask_seq_len_;
    bool    high_precision_;
    float   scale_;
    int32_t D_stub_ {HEAD_DIM_DEFAULT}; // 暂存 head_dim，供最终 O 归一化

    // GM 指针
    __gm__ WORKSPACE_T *gm_scores_ {nullptr}; // 在 Vector 阶段用 fp32 工作区
    __gm__ float       *gm_o_ {nullptr};
    __gm__ float       *gm_rowsum_ {nullptr};
    __gm__ float       *gm_logsum_ {nullptr};
    __gm__ INPUT_T     *gm_mask_ {nullptr};
    __gm__ INPUT_T     *gm_ones_ {nullptr};
};

#endif // VECTOR_OPERATIONS_H