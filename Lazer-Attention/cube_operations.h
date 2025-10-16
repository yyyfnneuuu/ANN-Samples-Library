#ifndef CUBE_OPERATIONS_H
#define CUBE_OPERATIONS_H

#include "constants.h"
#include "addressing.h"
#include "kernel_operator.h"

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
class CubeForward {
public:
    __aicore__ inline CubeForward() : high_precision_(true), scale_(1.f) {}

    __aicore__ inline void Init(
        __gm__ uint8_t *__restrict__ a_cube1,   // Q base
        __gm__ uint8_t *__restrict__ b_cube1,   // K base
        __gm__ uint8_t *__restrict__ b_cube2,   // V base
        __gm__ uint8_t *__restrict__ c_cube1,   // Scores base (bf16)
        __gm__ float   *__restrict__ c_cube2,   // O(未归一化) base (fp32)
        __gm__ uint8_t *__restrict__ ones_rowsum, // 常量 1 向量（bf16）
        __gm__ float   *__restrict__ gm_rowsum,   // rowsum (fp32)
        int32_t Y, int32_t F, int32_t B, int32_t N,
        int32_t S1, int32_t S2, int32_t D, int32_t nG,
        int32_t qk_triangle, int32_t sparseMode, int32_t window_length)
    {
        gm_q_          = reinterpret_cast<__gm__ TYPE*>(a_cube1);
        gm_k_          = reinterpret_cast<__gm__ TYPE*>(b_cube1);
        gm_v_          = reinterpret_cast<__gm__ TYPE*>(b_cube2);
        gm_scores_     = reinterpret_cast<__gm__ WORKSPACE_TYPE*>(c_cube1);
        gm_o_          = c_cube2;
        gm_rowsum_     = gm_rowsum;
        gm_ones_       = reinterpret_cast<__gm__ TYPE*>(ones_rowsum);

        Y_ = Y; F_ = F; B_ = B; N_ = N; S1_ = S1; S2_ = S2; D_ = D; G_ = nG;
        qk_triangle_ = qk_triangle;
        sparse_mode_ = sparseMode;
        window_len_  = window_length;

        scale_ = attn_scale_from_D(D_);

        // 需要拿到核组索引，占位使用
        int64_t cur_core_index = 0, core_group_index = 0, local_block_index = 0;

        address_.init(B_, N_, G_, S1_, S2_, D_, qk_triangle_ != 0);
        address_.set_core_info(Y_*F_, F_, Y_, cur_core_index, core_group_index, local_block_index);
    }

    __aicore__ inline void SetHighPrecision(bool on) { high_precision_ = on; }

    __aicore__ inline void Run()
    {
        // == 阶段一：Q x Kᵀ -> Scores(bf16/half)，需要按 k-tile 做累加 ==
        int64_t total_rounds = address_.get_total_round();
        for (int64_t rid = 0; rid < total_rounds; ++rid) {
            if (!address_.is_running(rid)) continue;

            Address::Phy_Addr_forward_cube1<TYPE, TYPE, WORKSPACE_TYPE> src{}, rm{};
            address_.template addrMapping_cube1<TYPE, TYPE, WORKSPACE_TYPE>(
                gm_q_, gm_k_, gm_scores_, rid, &src, &rm);

            // 在这里做 tile-k 循环与 mmad：
            // for (k0 = 0; k0 < S2_; k0 += BASE_BLOCK_SIDE_LEN) {
            //   load Q[rowblk, k0..k0+128], load K^T[k0..k0+128, colblk]
            //   C += Q * K^T
            // }
            // 写回 gm_scores_ 的该行块；dtype=WORKSPACE_TYPE(bf16/half)

            // 这里仅寻址，实际GEMM省略
        }

        // == 阶段三融合：Scores x V -> O（fp32） + rowsum（fp32） ==
        for (int64_t rid = 0; rid < total_rounds; ++rid) {
            if (!address_.is_running(rid)) continue;

            Address::Phy_Addr_forward_cube2_rowsum<WORKSPACE_TYPE, TYPE, float> src2{}, rm2{};
            int64_t src_len=0, rm_len=0;
            address_.template addrMapping_cube2_rowsum<WORKSPACE_TYPE, TYPE, float>(
                gm_scores_, gm_v_, gm_o_, gm_rowsum_, rid, src_len, rm_len, &src2, &rm2);

            // 与上类似，沿 k-tile 做 GEMM：
            // O[rowblk, :] += Scores[rowblk, k0..k0+128] @ V[k0..k0+128, :]
            // rowsum[rowblk] += sum(Scores[rowblk, k0..k0+128])
            // O/rowsum 用 fp32 累加
        }
    }

private:
    int32_t Y_, F_, B_, N_, S1_, S2_, D_, G_;
    int32_t qk_triangle_, sparse_mode_, window_len_;
    bool     high_precision_;
    float    scale_;

    // GM 指针
    __gm__ TYPE           *gm_q_ {nullptr};
    __gm__ TYPE           *gm_k_ {nullptr};
    __gm__ TYPE           *gm_v_ {nullptr};
    __gm__ WORKSPACE_TYPE *gm_scores_ {nullptr}; // bf16/half scores
    __gm__ float          *gm_o_ {nullptr};      // 未归一化 O（fp32）
    __gm__ float          *gm_rowsum_ {nullptr}; // rowsum（fp32）
    __gm__ TYPE           *gm_ones_ {nullptr};   // 常量 1（bf16/half）

    Address::AddrMapping_forward<TYPE> address_;
};

#endif // CUBE_OPERATIONS_H