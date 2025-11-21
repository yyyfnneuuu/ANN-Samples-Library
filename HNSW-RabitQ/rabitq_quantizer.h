// 将RabitQ的FHT和标量量化适配PG的数据类型
#pragma once
#include "vector.h"
#include "pg_rabitq.h"

// FHT旋转器
typedef struct FhtRotator FhtRotator;

FhtRotator* CreateFhtRotator(int dim, int padded_dim);
void RotateVector(FhtRotator* rotator, const float* in, float* out);
void FreeFhtRotator(FhtRotator* rotator);

// RabitQ编码函数
void RabitqEncode(
        const float* rotated_vec,
        int padded_dim,
        int total_bits,
        uint8_t* codes,     // output
        float* f_add,       // output
        float* f_rescale,   // output
        float* f_error      // output
);