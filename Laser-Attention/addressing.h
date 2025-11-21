#ifndef ADDRESSING_H
#define ADDRESSING_H

#include "constants.h"
#include "kernel_operator.h"

#define SIZE_128 128
#define SIZE_256 256
#define QUERY_BLOCK_SIZE        (128*128)
#define KEY_BLOCK_SIZE          (128*256)
#define VALUE_BLOCK_SIZE        (128*128)
#define OUTPUT_BLOCK_SIZE       (128*128)
#define ROWSUM_BLOCK_SIZE       128
#define ATTENTION_SCORE_BLOCK_SIZE (128*128)

namespace Address {
    template <typename TYPE>
    class AddrMapping_forward {
    public:
        // B N S D 格式的基础信息
        int64_t batch_size_;			 // batch批次的大小
        int64_t head_num_;				 // head数量
        int64_t gqa_group_num_;			 // GQA场景的组数
        int64_t query_sequence_len_;	 // query序列长度
        int64_t key_value_sequence_len_; // key、value序列长度
        int64_t output_sequence_len_;	 // 输出O的序列长度
        int64_t row_sum_sequence_len_;	 // row_sum的序列长度
        int64_t head_dim_;				 // head_dim的长度

        // 核数、核组的信息
        int64_t core_num_;			 // 使用的核心数量
        int64_t core_group_num_;	 // 核数数量
        int64_t core_num_per_group_; // 每一核组的核心数量
        int64_t cur_core_index_;	 // 当前核心的序号
        int64_t cur_group_index_;	 // 当前组的序号
        int64_t group_local_index_;	 // 组内核心的序号

        // 偏移相关信息
        bool is_triangle_;					   // 倒三角mask的标志
        int64_t forward_block_num_per_col_;	   // 负载均衡后attention的行数
        int64_t forward_block_num_per_row_;	   // 负载均衡后attention的列数
        int64_t forward_block_rows_per_head_;  // 前向每个head的基本行块数量
        int64_t forward_block_rows_per_batch_; // 前向每个batch的基本行块数量
        int64_t forward_total_rows_;		   // 前向计算的总行块数
        int64_t forward_total_rounds_;		   // 前向总共的轮次
        int64_t forward_block_num_per_core_;   // 前向每个核心平均处理的基本块数量
        int64_t forward_remain_block_num_;	   // 前向尾块的数量

public:
    __aicore__ __inline__ void init(
        int64_t batch_size, int64_t head_num, int64_t gqa_group_num,
        int64_t query_sequence_len, int64_t key_value_sequence_len,
        int64_t head_dim, bool is_triangle);

    __aicore__ __inline__ void set_core_info(
        int64_t core_num, int64_t core_group_num, int64_t core_num_per_group,
        int64_t cur_core_index, int64_t cur_group_index, int64_t group_local_index);

    __aicore__ __inline__ int64_t get_total_round() { return forward_total_rounds_; }

    // 当前 round 是否由本 core 执行
    __aicore__ __inline__ bool is_running(int64_t round_id) {
        return (round_id % core_num_) == cur_core_index_;
    }

    // ==无 mask 的基础块寻址==
    // cube1: Q [S1 x D]  x  K^T [D x S2]  -> Scores [S1 x S2]
    template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
    __aicore__ __inline__ void addrMapping_cube1(
        __gm__ T_LEFT  *__restrict__ left,      // Q base
        __gm__ T_RIGHT *__restrict__ right,     // K base (按 K^T 访问)
        __gm__ T_OUTPUT*__restrict__ out,       // Scores base
        int64_t round_id,
        Phy_Addr_forward_cube1<T_LEFT, T_RIGHT, T_OUTPUT> *src,
        Phy_Addr_forward_cube1<T_LEFT, T_RIGHT, T_OUTPUT> *remain);

    // cube2_rowsum: Scores [S1 x S2] x V [S2 x D] -> O [S1 x D] + rowsum[S1]
    template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
    __aicore__ __inline__ void addrMapping_cube2_rowsum(
        __gm__ T_LEFT  *__restrict__ left,      // Scores base
        __gm__ T_RIGHT *__restrict__ right,     // V base
        __gm__ T_OUTPUT*__restrict__ out,       // O base
        __gm__ T_OUTPUT*__restrict__ rowsum_out,// rowsum base
        int64_t round_id, int64_t &src_length, int64_t &remain_length,
        Phy_Addr_forward_cube2_rowsum<T_LEFT, T_RIGHT, T_OUTPUT> *src,
        Phy_Addr_forward_cube2_rowsum<T_LEFT, T_RIGHT, T_OUTPUT> *remain);

private:
    __aicore__ __inline__ void set_balance_info();
};

// 实现
template <typename TYPE>
__aicore__ __inline__ void AddrMapping_forward<TYPE>::init(
    int64_t batch_size, int64_t head_num, int64_t gqa_group_num,
    int64_t query_sequence_len, int64_t key_value_sequence_len,
    int64_t head_dim, bool is_triangle)
{
    batch_size_            = batch_size;
    head_num_              = head_num;
    gqa_group_num_         = gqa_group_num;
    query_sequence_len_    = query_sequence_len;
    key_value_sequence_len_= key_value_sequence_len;
    head_dim_              = head_dim;
    is_triangle_           = is_triangle;

    output_sequence_len_   = query_sequence_len_;
    row_sum_sequence_len_  = query_sequence_len_;

    set_balance_info();
}

template <typename TYPE>
__aicore__ __inline__ void AddrMapping_forward<TYPE>::set_core_info(
    int64_t core_num, int64_t core_group_num, int64_t core_num_per_group,
    int64_t cur_core_index, int64_t cur_group_index, int64_t group_local_index)
{
    core_num_           = core_num;
    core_group_num_     = core_group_num;
    core_num_per_group_ = core_num_per_group;
    cur_core_index_     = cur_core_index;
    cur_group_index_    = cur_group_index;
    group_local_index_  = group_local_index;
}

template <typename TYPE>
__aicore__ __inline__ void AddrMapping_forward<TYPE>::set_balance_info()
{
    // 基于 128x128 block 的行/列块数（忽略尾块，由remain返回）
    forward_block_num_per_row_ = (query_sequence_len_ + BASE_BLOCK_SIDE_LEN - 1) / BASE_BLOCK_SIDE_LEN; // Q 的 block 行数
    forward_block_num_per_col_ = (key_value_sequence_len_ + BASE_BLOCK_SIDE_LEN - 1) / BASE_BLOCK_SIDE_LEN; // K 的 block 列数 (以 128 对齐)

    // 每个 Head 的 block 行数
    forward_block_rows_per_head_  = forward_block_num_per_row_;
    // 每个 Batch 的 block 行数（所有 head 累加）
    forward_block_rows_per_batch_ = head_num_ * forward_block_rows_per_head_;

    // 总 block 行数 跨 Batch
    forward_total_rows_ = batch_size_ * forward_block_rows_per_batch_;

    // 轮次：以“块行”为粒度，列方向由 k tile 累加覆盖
    forward_total_rounds_ = forward_total_rows_; // 一行一个 round
    forward_block_num_per_core_ = (forward_total_rounds_ + core_num_ - 1) / core_num_;
    forward_remain_block_num_  = forward_total_rounds_ - forward_block_num_per_core_ * core_num_;
}

// == 基础 cube1 寻址：给出某个 round 的 (b,h,row_block) 并映射到 Q/K^T/Scores 的起始指针 ==
template <typename TYPE> template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
__aicore__ __inline__ void AddrMapping_forward<TYPE>::addrMapping_cube1(
    __gm__ T_LEFT  *__restrict__ q_base,
    __gm__ T_RIGHT *__restrict__ k_base,      // 按 K^T 访问
    __gm__ T_OUTPUT*__restrict__ score_base,
    int64_t round_id,
    Phy_Addr_forward_cube1<T_LEFT, T_RIGHT, T_OUTPUT> *src,
    Phy_Addr_forward_cube1<T_LEFT, T_RIGHT, T_OUTPUT> *remain)
{
    // 以“块行”为 round
    int64_t row_blk_global = round_id; // 0..forward_total_rows_-1
    int64_t b  = row_blk_global / forward_block_rows_per_batch_;
    int64_t t  = row_blk_global % forward_block_rows_per_batch_;
    int64_t h  = t / forward_block_rows_per_head_;
    int64_t rb = t % forward_block_rows_per_head_;      // row block idx within this head

    int64_t row0 = rb * BASE_BLOCK_SIDE_LEN;           // 该轮的 Q 行起点
    // 列方向从 0 开始，K^T 的列 block 在后续 k 累加中处理（调用者循环 k）
    int64_t col0 = 0;

    // 线性化：假定布局为 [B, N, S, D]，D 连续
    int64_t stride_q  = head_dim_;
    int64_t stride_s  = key_value_sequence_len_;   // for scores
    int64_t stride_kv = head_dim_;                 // for K/V base

    // 起始指针（仅本 row block 的起点；k tile 在调用时滚动）
    src->left  = q_base + (((b*head_num_ + h) * query_sequence_len_) + row0) * head_dim_ + 0;
    src->right = k_base;    // 访问时应在 k-tile 中偏移到 (k0, col)
    src->out   = score_base + (((b*head_num_ + h) * query_sequence_len_) + row0) * key_value_sequence_len_ + col0;
    src->k     = 0;

    *remain = *src; // 暂不单独处理尾块（调用处需根据实际长度裁剪）
}

// == 基础 cube2_rowsum 寻址：Scores 行块 x V -> O 行块 + rowsum 行 ==
template <typename TYPE> template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
__aicore__ __inline__ void AddrMapping_forward<TYPE>::addrMapping_cube2_rowsum(
    __gm__ T_LEFT  *__restrict__ scores_base,
    __gm__ T_RIGHT *__restrict__ v_base,
    __gm__ T_OUTPUT*__restrict__ o_base,
    __gm__ T_OUTPUT*__restrict__ rowsum_base,
    int64_t round_id, int64_t &src_length, int64_t &remain_length,
    Phy_Addr_forward_cube2_rowsum<T_LEFT, T_RIGHT, T_OUTPUT> *src,
    Phy_Addr_forward_cube2_rowsum<T_LEFT, T_RIGHT, T_OUTPUT> *remain)
{
    int64_t row_blk_global = round_id;
    int64_t b  = row_blk_global / forward_block_rows_per_batch_;
    int64_t t  = row_blk_global % forward_block_rows_per_batch_;
    int64_t h  = t / forward_block_rows_per_head_;
    int64_t rb = t % forward_block_rows_per_head_;

    int64_t row0 = rb * BASE_BLOCK_SIDE_LEN; // S1 方向
    int64_t col0 = 0;                        // D 方向从 0 开始，k-tile 决定 V 的切块

    // 基本布局：Scores [B,N,S1,S2]，V [B,N,S2,D]，O [B,N,S1,D]，rowsum [B,N,S1]
    src->left       = scores_base + (((b*head_num_ + h) * query_sequence_len_) + row0) * key_value_sequence_len_ + 0;
    src->right      = v_base;    // 具体 k-tile 偏移在调用时滚动
    src->out        = o_base + (((b*head_num_ + h) * query_sequence_len_) + row0) * head_dim_ + col0;
    src->rowsum_out = rowsum_base + ((b*head_num_ + h) * query_sequence_len_) + row0;
    src->k          = 0;

    // 返回本次处理的有效长度（行长度 / 尾部长度）
    int64_t remain_rows = query_sequence_len_ - row0;
    src_length    = remain_rows >= BASE_BLOCK_SIDE_LEN ? BASE_BLOCK_SIDE_LEN : remain_rows;
    remain_length = 0; // 简化：调用处据 src_length 裁剪
    *remain = *src;
}

} // namespace Address

#endif // ADDRESSING_H