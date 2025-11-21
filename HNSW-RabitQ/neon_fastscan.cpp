#pragma once
#include <arm_neon.h>

// 准备用于FastScan的查询向量 (lut)
void NeonFastScanPrepareQuery(const float* query_vec, int padded_dim, int total_bits, int8_t* lut);

// 使用NEON指令批量计算估算距离
void NeonFastScanAccumulate(
        const uint8_t* packed_codes, // 8个或16个向量的打包码字
        const int8_t* lut,           // 查询向量的查找表
        int padded_dim,
        int total_bits,
        uint16_t* results            // 累加结果
);

// rabitq码字打包，适配NEON的128位处理方式
void NeonPackCodes(const uint8_t* codes, size_t num_vecs, size_t code_bytes_per_vec, uint8_t* packed_blocks);

#include "neon_fastscan.h"

// accumulate核心逻辑的NEON改写
// 假设 total_bits = 4, 每个dim用4 bit表示, 2个dim一个byte
void NeonFastScanAccumulate(const uint8_t* packed_codes, const int8_t* lut, int padded_dim, int total_bits, uint16_t* results) {
    // NEON一次处理16个byte, 对应16个向量的某2个维度
    uint16x8_t-D- accu_lo_0 = vdupq_n_u16(0);
    uint16x8_t accu_lo_1 = vdupq_n_u16(0);

    const uint8x16_t lo_mask = vdupq_n_u8(0x0F);

    for (size_t i = 0; i < padded_dim / 2; i += 16) {
        // 加载16个向量的码字 (16 bytes)
        uint8x16_t c = vld1q_u8(&packed_codes[i * 16]);

        // 加载查询查找表, lut有 16 * (padded_dim / 2)
        // 每个维度的lut有16个值，这里一次加载一个维度的lut
        int8x16_t lut_part = vld1q_s8(&lut[i * 16]);

        // 拆分高低4 bit
        uint8x16_t lo_indices = vandq_u8(c, lo_mask); // 对应第一个维度的码字
        uint8x16_t hi_indices = vshrq_n_u8(c, 4);    // 对应第二个维度的码字

        // 查表 (Table-lookup)
        // vqtbl1q_s8(lut_part, lo_indices)
        int8x16_t res_lo = vqtbl1q_s8(lut_part, lo_indices);
        int8x16_t res_hi = vqtbl1q_s8(lut_part, hi_indices);

        // 累加
        // 将s8扩展为s16进行累加防止溢出
        accu_lo_0 = vaddw_s8(accu_lo_0, vget_low_s8(res_lo));
        accu_lo_1 = vaddw_s8(accu_lo_1, vget_high_s8(res_lo));
    }

    // 将所有累加器结果合并到 results
}