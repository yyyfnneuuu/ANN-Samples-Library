#ifndef CUBE_OPERATIONS_H
#define CUBE_OPERATIONS_H

#include "constants.h"
#include "addressing.h"
#include "kernel_operator.h"

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
class CubeForward {
public:
    __aicore__ inline CubeForward(){};
    __aicore__ inline void Init(__gm__ uint8_t *__restrict__ a_cube1, __gm__ uint8_t *__restrict__ b_cube1,
    __gm__ uint8_t *__restrict__ b_cube2, __gm__ uint8_t *__restrict__ c_cube1, __gm__ float *__restrict__ c_cube2,
            __gm__ uint8_t *__restrict__ ones_rowsum, __gm__ float *__restrict__ gm_rowsum, int32_t Y, int32_t F, int32_t B,
    int32_t N, int32_t S1, int32_t S2, int32_t D, int32_t nG, int32_t qk_triangle, int32_t sparseMode,
            int32_t window_length);
    __aicore__ inline void Run();

    // ... (rest of public methods) ...

private:
    // --- Private Methods ---
    __aicore__ __inline__ void mix_cube2_rowsum(int32_t cube2_roundId);
    __aicore__ __inline__ void cube2_rowsum_mix_only(const Address::Phy_Addr_forward_cube2_rowsum<WORKSPACE_TYPE, TYPE, float> *cube_addr,
                                                     int32_t cube_length, int32_t m_length, int32_t vcore_num_per_head);
    __aicore__ __inline__ void cube1_headDim192(
            Address::Phy_Addr_forward_cube1<TYPE, TYPE, WORKSPACE_TYPE> *src, Address::Phy_Addr_forward_cube1<TYPE, TYPE, WORKSPACE_TYPE> *remain);
    __aicore__ __inline__ void base_block_mad_headDim192(Address::Phy_Addr_forward_cube1<TYPE, TYPE, WORKSPACE_TYPE> addr_1,
                                                         Address::Phy_Addr_forward_cube1<TYPE, TYPE, WORKSPACE_TYPE> addr_2,
                                                         __cbuf__ TYPE *l1_base_a_cube1, int32_t l0a_offset_remain = -1);
    // ... (rest of private methods) ...

    // --- Member Variables ---
    __gm__ TYPE *__restrict__ gm_a_cube1;
    // ... (all member variables from the original CubeForward class) ...
    Address::AddrMapping_forward<TYPE> address;
};

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ inline void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::Init(
        /* ... parameters ... */)
{
    // ... implementation ...
    // Init and set_core_info for the address module
    address.init(this->B, this->N, this->G, this->S1, this->S2, this->D, this->qk_triangle);
    address.set_core_info(this->Y * this->F, this->F, this->Y, this->cur_core_index, this->core_group_index, this->local_block_index);
}


template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ inline void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::Run()
{
    // ... implementation from original Run() ...
}

// rest of the implementations ...

#endif // CUBE_OPERATIONS_H