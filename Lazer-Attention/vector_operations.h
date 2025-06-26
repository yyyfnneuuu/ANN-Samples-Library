#ifndef VECTOR_OPERATIONS_H
#define VECTOR_OPERATIONS_H

#include "constants.h"
#include "kernel_operator.h"

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
class VectorForward {
public:
    __aicore__ inline VectorForward(){};
    __aicore__ inline void Init(
            __gm__ uint8_t *__restrict__ a_cube1,
    __gm__ uint8_t *__restrict__ b_cube1,
    __gm__ uint8_t *__restrict__ b_cube2,
    __gm__ uint8_t *__restrict__ mask_gm,
    __gm__ uint8_t *__restrict__ ones_gm,
    __gm__ float *__restrict__ zeros_gm,
            __gm__ uint8_t *__restrict__ score_gm,
            __gm__ float *__restrict__ c_cube2,
            __gm__ float *__restrict__ log_sum_gm,
            __gm__ float *__restrict__ gm_rowsum,
            int32_t qSeqLength,
    int32_t kSeqLength,
            int32_t H,
    int32_t B,
            int32_t Y,
    int32_t qk,
            int32_t windows_block_num,
    int32_t maskSeqLength);
    __aicore__ inline void Run();

    // rest of public methods and structs ...

private:
    // --- Private Methods ---
    __aicore__ __inline__ void initWorkSpace();
    __aicore__ __inline__ void attention_score_normalize(int32_t max_proc_len, int32_t cur_core_process_lines, int32_t cur_core_offset_lines,
                                                         UB_FOR_NORMALIZE ub_norm, __gm__ float *__restrict__ gm_c_cube2, __gm__ float *__restrict__ rowsum_gm);
    // all other private helper functions ...

    // --- Member Variables ---
    __gm__ INPUT_T *__restrict__ gm_a_cube1;
    // all member variables ...
};

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ inline void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::Init(
        /* ... parameters ... */)
{
    // ... implementation ...
}


template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ inline void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::Run()
{
    // ... implementation from original Run() ...
}

// rest of the implementations ...

#endif // VECTOR_OPERATIONS_H