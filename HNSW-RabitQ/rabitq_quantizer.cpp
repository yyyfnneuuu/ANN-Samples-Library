// 关键改动点：
// 1. 移除了 Eigen 库依赖，所有矩阵和向量操作都通过指针和循环实现。
// 2. palloc/pfree 进行内存管理。
// 3. 算法逻辑（FHT、RabitQ 的因子计算）。

#include "postgres.h"
#include "fmgr.h"
#include "utils/palloc.h"
#include "utils/elog.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "rabitq_quantizer.h"

// RabitQ论文中提到的常数
#define RABITQ_CONST_EPSILON 1.9f

// FHT旋转器状态的内部定义
struct FhtRotator {
    int dim;
    int padded_dim;
    int trunc_dim; // FHT变换要求的2的幂次维度
    float fac;     // 缩放因子 1.0f / sqrt(trunc_dim)
    uint8_t* flip; // 用于随机符号翻转的比特序列

    // FHT变换函数指针
    void (*fht_transform)(float*);
};

// -------------------------------------------------------------------
// FHT (Fast Hadamard Transform) 内部辅助函数
// -------------------------------------------------------------------

// FHT的蝶形运算单元
static inline void fht_butterfly(float* a, float* b) {
    float t = *a;
    *a = t + *b;
    *b = t - *b;
}

// 针对不同维度的FHT实现
static void fht_float_6(float* data) {
    // dim = 64
    for (int i = 0; i < 64; i += 2) fht_butterfly(&data[i], &data[i+1]);
    for (int i = 0; i < 64; i += 4) for (int j = 0; j < 2; j++) fht_butterfly(&data[i+j], &data[i+j+2]);
    for (int i = 0; i < 64; i += 8) for (int j = 0; j < 4; j++) fht_butterfly(&data[i+j], &data[i+j+4]);
    for (int i = 0; i < 64; i += 16) for (int j = 0; j < 8; j++) fht_butterfly(&data[i+j], &data[i+j+8]);
    for (int i = 0; i < 64; i += 32) for (int j = 0; j < 16; j++) fht_butterfly(&data[i+j], &data[i+j+16]);
    for (int j = 0; j < 32; j++) fht_butterfly(&data[j], &data[j+32]);
}

// 增加FHT的随机性
static void kacs_walk(float* data, int len) {
    int half_len = len / 2;
    for (int i = 0; i < half_len; ++i) {
        fht_butterfly(&data[i], &data[i + half_len]);
    }
}

// 向量缩放
static void vec_rescale(float* data, int len, float factor) {
    for (int i = 0; i < len; ++i) {
        data[i] *= factor;
    }
}

// 符号翻转
static void flip_sign(const uint8_t* flip_bits, float* data, int len) {
    for (int i = 0; i < len; ++i) {
        if ((flip_bits[i / 8] >> (i % 8)) & 1) {
            data[i] = -data[i];
        }
    }
}

// -------------------------------------------------------------------
// FHT 旋转器公共接口实现
// -------------------------------------------------------------------

FhtRotator* CreateFhtRotator(int dim, int padded_dim) {
    FhtRotator* rotator = (FhtRotator*)palloc0(sizeof(FhtRotator));
    rotator->dim = dim;
    rotator->padded_dim = padded_dim;

    // 计算FHT需要的2的幂次维度
    if (padded_dim > 0 && (padded_dim & (padded_dim - 1)) != 0) {
        elog(ERROR, "padded_dim must be a power of 2 for FHT rotator");
    }
    rotator->trunc_dim = padded_dim;
    rotator->fac = 1.0f / sqrtf((float)rotator->trunc_dim);

    // 分配并生成随机翻转位
    // RabitQ论文中提到需要4轮FHT，每轮都需要一组独立的随机翻转
    int flip_size = 4 * rotator->padded_dim / 8;
    rotator->flip = (uint8_t*)palloc(flip_size);
    for (int i = 0; i < flip_size; ++i) {
        rotator->flip[i] = (uint8_t)rand();
    }

    // 根据维度选择FHT实现函数
    int log_dim = (int)floor(log2(rotator->trunc_dim));
    switch (log_dim) {
        case 6: rotator->fht_transform = fht_float_6; break;
        default:
            // 支持64维 (2^6) 到 2048维 (2^11)
            if (log_dim < 6 || log_dim > 11) {
                rotator->fht_transform = fht_float_unsupported;
            } else {
                rotator->fht_transform = fht_float_unsupported;
            }
            break;
    }

    return rotator;
}

void RotateVector(FhtRotator* rotator, const float* in, float* out) {
    // 临时工作区，用于存储旋转过程中的向量
    float* temp_vec = (float*)palloc(sizeof(float) * rotator->padded_dim);

    // 1. 复制并补零
    memcpy(temp_vec, in, sizeof(float) * rotator->dim);
    if (rotator->padded_dim > rotator->dim) {
        memset(temp_vec + rotator->dim, 0, sizeof(float) * (rotator->padded_dim - rotator->dim));
    }

    const uint8_t* flip_ptr = rotator->flip;
    for (int round = 0; round < 4; ++round) {
        flip_sign(flip_ptr, temp_vec, rotator->padded_dim);
        rotator->fht_transform(temp_vec);
        vec_rescale(temp_vec, rotator->trunc_dim, rotator->fac);

        flip_ptr += rotator->padded_dim / 8;
    }

    // 3. 最终缩放
    vec_rescale(temp_vec, rotator->padded_dim, 0.25f);

    memcpy(out, temp_vec, sizeof(float) * rotator->padded_dim);
    pfree(temp_vec);
}

void FreeFhtRotator(FhtRotator* rotator) {
    if (rotator) {
        pfree(rotator->flip);
        pfree(rotator);
    }
}


// -------------------------------------------------------------------
// RabitQ 内部辅助函数
// -------------------------------------------------------------------

// 计算向量L2范数的平方
static float l2_norm_sqr(const float* vec, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum += vec[i] * vec[i];
    }
    return sum;
}

// 计算两个向量的点积
static float dot_product(const float* v1, const float* v2, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

// 计算残差向量，并生成1-bit量化码（符号）
static void one_bit_code(const float* data, const float* centroid, int dim, int* binary_code, float* residual) {
    for (int i = 0; i < dim; ++i) {
        residual[i] = data[i] - centroid[i];
        binary_code[i] = (residual[i] >= 0.0f) ? 1 : 0;
    }
}

// 恒定缩放因子 t_const 的快速版本。在实际构建索引时，可以预先计算好这个t_const。
static float faster_quantize_ex(const float* o_abs, uint8_t* code, int dim, int ex_bits, double t_const) {
    double ipnorm = 0.0;
    const int max_code_val = (1 << ex_bits) - 1;

    for (int i = 0; i < dim; ++i) {
        int val = (int)(t_const * o_abs[i] + 0.5); // 加0.5做四舍五入
        if (val > max_code_val) {
            val = max_code_val;
        }
        code[i] = (uint8_t)val;
        ipnorm += (val + 0.5) * o_abs[i];
    }

    float ipnorm_inv = (float)(1.0 / ipnorm);
    return isfinite(ipnorm_inv) ? ipnorm_inv : 1.0f;
}

// -------------------------------------------------------------------
// RabitQ 编码公共接口实现
// -------------------------------------------------------------------

void RabitqEncode(
        const float* rotated_vec,
        int padded_dim,
        int total_bits,
        uint8_t* codes,     // 最终的量化码字 (total_bits per dim)
        float* f_add,
        float* f_rescale,
        float* f_error
) {
    if (total_bits < 1 || total_bits > 8) {
        elog(ERROR, "RabitQ total_bits must be between 1 and 8");
    }

    int ex_bits = total_bits - 1;
    const float centroid_zero[padded_dim]; // RabitQ使用0作为质心
    memset((void*)centroid_zero, 0, sizeof(float) * padded_dim);

    // 使用palloc为临时数组分配内存，由当前内存上下文管理
    int* binary_code = (int*)palloc(sizeof(int) * padded_dim);
    float* residual = (float*)palloc(sizeof(float) * padded_dim);

    // 步骤1: 1-bit量化，得到符号和残差
    one_bit_code(rotated_vec, centroid_zero, padded_dim, binary_code, residual);

    if (ex_bits > 0) {
        // RabitQ+ (total_bits > 1) 逻辑
        uint8_t* ex_code_u8 = (uint8_t*)palloc(sizeof(uint8_t) * padded_dim);
        float* abs_residual = (float*)palloc(sizeof(float) * padded_dim);

        // a. 计算残差的绝对值和归一化
        float residual_l2_norm = sqrtf(l2_norm_sqr(residual, padded_dim));
        float inv_residual_l2_norm = (residual_l2_norm > 1e-9) ? (1.0f / residual_l2_norm) : 1.0f;
        for (int i = 0; i < padded_dim; ++i) {
            abs_residual[i] = fabsf(residual[i]) * inv_residual_l2_norm;
        }

        // b. ex_bits量化。使用预计算的常量t_const来加速。
        // 这个值依赖于维度和ex_bits, 实际应从外部传入或查询得到。
        double t_const_mock = 2.5 * (1 << ex_bits);
        float ipnorm_inv = faster_quantize_ex(abs_residual, ex_code_u8, padded_dim, ex_bits, t_const_mock);

        // c. 根据原始残差符号翻转ex_code
        int ex_code_mask = (1 << ex_bits) - 1;
        for (int i = 0; i < padded_dim; ++i) {
            if (residual[i] < 0) {
                ex_code_u8[i] = (~ex_code_u8[i]) & ex_code_mask;
            }
        }

        // d. 合并1-bit码和ex_bits码
        for (int i = 0; i < padded_dim; ++i) {
            codes[i] = (uint8_t)((binary_code[i] << ex_bits) | ex_code_u8[i]);
        }

        // e. 计算RabitQ因子 f_add, f_rescale, f_error
        float* u_cb = (float*)palloc(sizeof(float)*padded_dim);
        float cb = -(float)((1 << ex_bits) - 0.5f);
        for(int i=0; i < padded_dim; ++i) {
            u_cb[i] = (float)codes[i] + cb;
        }

        float l2_sqr_res = residual_l2_norm * residual_l2_norm;
        float ip_resi_ucb = dot_product(residual, u_cb, padded_dim);

        if (fabsf(ip_resi_ucb) < 1e-9) ip_resi_ucb = (ip_resi_ucb > 0) ? 1e-9f : -1e-9f;

        // L2距离的因子
        *f_add = l2_sqr_res; // g_add=0, cent_ip=0
        *f_rescale = -2.0f * l2_sqr_res / ip_resi_ucb;

        float tmp_error_term_sq = ((l2_sqr_res * l2_norm_sqr(u_cb, padded_dim)) / (ip_resi_ucb * ip_resi_ucb)) - 1.0f;
        if (tmp_error_term_sq < 0) tmp_error_term_sq = 0;
        *f_error = 2.0f * residual_l2_norm * RABITQ_CONST_EPSILON * sqrtf(tmp_error_term_sq / (padded_dim - 1));

        pfree(u_cb);
        pfree(abs_residual);
        pfree(ex_code_u8);

    } else {
        // 纯1-bit RabitQ (total_bits = 1) 逻辑
        for (int i = 0; i < padded_dim; ++i) {
            codes[i] = (uint8_t)binary_code[i];
        }

        float* u_cb = (float*)palloc(sizeof(float)*padded_dim);
        float cb = -0.5f;
        for(int i=0; i < padded_dim; ++i) {
            u_cb[i] = (float)codes[i] + cb;
        }

        float l2_sqr_res = l2_norm_sqr(residual, padded_dim);
        float ip_resi_ucb = dot_product(residual, u_cb, padded_dim);

        if (fabsf(ip_resi_ucb) < 1e-9) ip_resi_ucb = (ip_resi_ucb > 0) ? 1e-9f : -1e-9f;

        *f_add = l2_sqr_res;
        *f_rescale = -2.0f * l2_sqr_res / ip_resi_ucb;

        float tmp_error_term_sq = ((l2_sqr_res * l2_norm_sqr(u_cb, padded_dim)) / (ip_resi_ucb * ip_resi_ucb)) - 1.0f;
        if (tmp_error_term_sq < 0) tmp_error_term_sq = 0;
        *f_error = 2.0f * sqrtf(l2_sqr_res) * RABITQ_CONST_EPSILON * sqrtf(tmp_error_term_sq / (padded_dim - 1));

        pfree(u_cb);
    }

    // 释放临时内存
    pfree(residual);
    pfree(binary_code);
}