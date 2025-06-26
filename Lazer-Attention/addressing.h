#ifndef ADDRESSING_H
#define ADDRESSING_H

#include "constants.h"
#include "kernel_operator.h"

#define SIZE_128 128
#define SIZE_256 256
#define QUERY_BLOCK_SIZE (128*128)
#define KEY_BLOCK_SIZE (128*256)
#define VALUE_BLOCK_SIZE (128*128)
#define OUTPUT_BLOCK_SIZE (128*128)
#define ROWSUM_BLOCK_SIZE 128
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
        __aicore__ __inline__ void init(int64_t batch_size, int64_t head_num, int64_t gqa_group_num, int64_t query_sequence_len,
                                        int64_t key_value_sequence_len, int64_t head_dim, bool is_triangle);

        __aicore__ __inline__ void set_core_info(int64_t core_num, int64_t core_group_num, int64_t core_num_per_group, int64_t cur_core_index,
                                                 int64_t cur_group_index, int64_t group_local_index);

        __aicore__ __inline__ int64_t get_total_round();

        __aicore__ __inline__ bool is_running(int64_t round_id);

        template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
        __aicore__ __inline__ void addrMapping_cube1(__gm__ T_LEFT *__restrict__ left,
        __gm__ T_RIGHT *__restrict__ right, __gm__ T_OUTPUT *__restrict__ out,
        int64_t round_id,
                Phy_Addr_forward_cube1<T_LEFT, T_RIGHT, T_OUTPUT> *src,
        Phy_Addr_forward_cube1<T_LEFT, T_RIGHT, T_OUTPUT> *remain);

        template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
        __aicore__ __inline__ void addrMapping_cube2_rowsum(__gm__ T_LEFT *__restrict__ left,
        __gm__ T_RIGHT *__restrict__ right, __gm__ T_OUTPUT *__restrict__ out,
        __gm__ T_OUTPUT *__restrict__ rowsum_out,
        int64_t round_id, int64_t &src_length,
        int64_t &remain_length,
                Phy_Addr_forward_cube2_rowsum<T_LEFT, T_RIGHT, T_OUTPUT> *src,
        Phy_Addr_forward_cube2_rowsum<T_LEFT, T_RIGHT, T_OUTPUT> *remain);

    private:
        __aicore__ __inline__ void set_balance_info();

        template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
        __aicore__ __inline__ void addrMapping_cube1_nomask(__gm__ T_LEFT *__restrict__ left,
        __gm__ T_RIGHT *__restrict__ right, __gm__ T_OUTPUT *__restrict__ out,
        int64_t round_id,
                Phy_Addr_forward_cube1<T_LEFT, T_RIGHT, T_OUTPUT> *src,
        Phy_Addr_forward_cube1<T_LEFT, T_RIGHT, T_OUTPUT> *remain);

        template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
        __aicore__ __inline__ void addrMapping_cube1_mask(__gm__ T_LEFT *__restrict__ left,
        __gm__ T_RIGHT *__restrict__ right, __gm__ T_OUTPUT *__restrict__ out,
        int64_t round_id,
                Phy_Addr_forward_cube1<T_LEFT, T_RIGHT, T_OUTPUT> *src,
        Phy_Addr_forward_cube1<T_LEFT, T_RIGHT, T_OUTPUT> *remain);

        template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
        __aicore__ __inline__ void addrMapping_cube2_rowsum_nomask(__gm__ T_LEFT *__restrict__ left,
        __gm__ T_RIGHT *__restrict__ right, __gm__ T_OUTPUT *__restrict__ out,
        __gm__ T_OUTPUT *__restrict__ rowsum_out,
        int64_t round_id, int64_t &src_length,
        int64_t &remain_length,
                Phy_Addr_forward_cube2_rowsum<T_LEFT, T_RIGHT, T_OUTPUT> *src,
        Phy_Addr_forward_cube2_rowsum<T_LEFT, T_RIGHT, T_OUTPUT> *remain);

        template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
        __aicore__ __inline__ void addrMapping_cube2_rowsum_mask(__gm__ T_LEFT *__restrict__ left,
        __gm__ T_RIGHT *__restrict__ right, __gm__ T_OUTPUT *__restrict__ out,
        __gm__ T_OUTPUT *__restrict__ rowsum_out,
        int64_t round_id, int64_t &src_length,
        int64_t &remain_length,
                Phy_Addr_forward_cube2_rowsum<T_LEFT, T_RIGHT, T_OUTPUT> *src,
        Phy_Addr_forward_cube2_rowsum<T_LEFT, T_RIGHT, T_OUTPUT> *remain);
    };

    template <typename TYPE>
    __aicore__ __inline__ void
            AddrMapping_forward<TYPE>::init(int64_t batch_size, int64_t head_num, int64_t gqa_group_num, int64_t query_sequence_len,
            int64_t key_value_sequence_len, int64_t head_dim, bool is_triangle)
{
    this->batch_size_ = batch_size;
    this->head_num_ = head_num;
    this->gqa_group_num_ = gqa_group_num;
    this->query_sequence_len_ = query_sequence_len;
    this->key_value_sequence_len_ = key_value_sequence_len;
    this->head_dim_ = head_dim;
    this->is_triangle_ = is_triangle;

    this->output_sequence_len_ = this->query_sequence_len_;
    this->row_sum_sequence_len_ = this->query_sequence_len_;
}

// rest of the implementations ...

} // namespace Address

#endif // ADDRESSING_H