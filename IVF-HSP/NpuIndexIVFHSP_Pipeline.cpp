#include "NpuIndexIVFHSP.h"

// L1 流水线：计算查询向量和L1码本的距离，并找到最近的nProbeL1个簇
APP_ERROR NpuIndexIVFHSP::SearchBatchImplL1(AscendTensor<float, DIMS_2>& queriesNpu,
                                            AscendTensor<float16_t, DIMS_2>& queryCodes,
                                            AscendTensor<uint16_t, DIMS_2>& l1KIndiceNpu) {
    APP_LOG_INFO("Running L1 Search Pipeline...");
    // ... 调用 RunL1DistOp 和 RunL1TopKOp 的具体实现 ...
    aclrtSynchronizeStream(defaultStream);
    aclrtSynchronizeStream(aiCpuStream);
    return APP_ERR_OK;
}

// L2 流水线：在L1选中的簇中，进一步计算和L2码本的距离，找到最近的nProbeL2个子簇
APP_ERROR NpuIndexIVFHSP::SearchBatchImplL2(AscendTensor<float16_t, DIMS_2>& queryCodesNpu,
                                            AscendTensor<uint16_t, DIMS_2>& l1KIndicesNpu,
                                            AscendTensor<uint64_t, DIMS_2>& addressOffsetOfBucketL3,
                                            AscendTensor<uint64_t, DIMS_2>& idAdressL3) {
    APP_LOG_INFO("Running L2 Search Pipeline...");
    // ... 调用 RunL2DistOp 和 RunL2TopKOp 的具体实现 ...
    aclrtSynchronizeStream(defaultStream);
    aclrtSynchronizeStream(aiCpuStream);
    return APP_ERR_OK;
}

// L2 流水线（带掩码）
APP_ERROR NpuIndexIVFHSP::SearchBatchImplL2(AscendTensor<uint8_t, DIMS_1>& maskBitNpu,
                                            AscendTensor<float16_t, DIMS_2>& queryCodesNpu,
                                            AscendTensor<uint16_t, DIMS_2>& l1KIndicesNpu,
                                            AscendTensor<uint64_t, DIMS_2>& addressOffsetOfBucketL3,
                                            AscendTensor<uint64_t, DIMS_2>& idAdressL3) {
    APP_LOG_INFO("Running L2 Search Pipeline with Mask...");
    // ... 调用 RunL2DistOp 和 RunL2TopKWithMaskOp 的具体实现 ...
    aclrtSynchronizeStream(defaultStream);
    aclrtSynchronizeStream(aiCpuStream);
    return APP_ERR_OK;
}


// L2 流水线（多索引）
APP_ERROR NpuIndexIVFHSP::SearchBatchImplMultiL2(AscendTensor<float16_t, DIMS_2>& queryCodesNpu,
                                                 AscendTensor<uint16_t, DIMS_2>& l1KIndicesNpu,
                                                 AscendTensor<uint64_t, DIMS_2>& indicesL2) {
    APP_LOG_INFO("Running Multi-Index L2 Search Pipeline...");
    // ... 调用 RunL2DistOp 和 RunMultiL2TopKOp 的具体实现 ...
    aclrtSynchronizeStream(defaultStream);
    aclrtSynchronizeStream(aiCpuStream);
    return APP_ERR_OK;
}


// L3 流水线：在L2选定的子簇中，计算最终距离并排序，得到TopK结果
APP_ERROR NpuIndexIVFHSP::SearchBatchImplL3(AscendTensor<float16_t, DIMS_2>& queryCodes,
                                            AscendTensor<uint64_t, DIMS_2>& addressOffsetOfBucketL3,
                                            AscendTensor<uint64_t, DIMS_2>& idAddressOfBucketL3,
                                            AscendTensor<float16_t, DIMS_2>& outDists,
                                            AscendTensor<int64_t, DIMS_2>& outIndices) {
    APP_LOG_INFO("Running L3 Search Pipeline...");
    // ... 根据 maskFlag 调用 RunL3DistOp/RunL3DistWithMaskOp 和 RunL3TopKOp 的具体实现 ...
    aclrtSynchronizeStream(defaultStream);
    aclrtSynchronizeStream(aiCpuStream);
    return APP_ERR_OK;
}

// L3 流水线（多索引）
APP_ERROR NpuIndexIVFHSP::SearchBatchImplMultiL3(const std::vector<NpuIndexIVFHSP*>& indexes, int i,
                                                 AscendTensor<float16_t, DIMS_2>& queryCodes,
                                                 AscendTensor<uint64_t, DIMS_3>& addressOffsetOfBucketL3,
                                                 AscendTensor<uint64_t, DIMS_3>& idAddressOfBucketL3,
                                                 AscendTensor<float16_t, DIMS_3>& distResult,
                                                 AscendTensor<float16_t, DIMS_3>& vcMinDistResult,
                                                 AscendTensor<uint16_t, DIMS_3>& opFlag) {
    APP_LOG_INFO("Running Multi-Index L3 Distance Calculation...");
    // ... 根据 maskFlag 调用 RunMultiL3DistOp/RunMultiL3DistWithMaskOp 的具体实现 ...
    return APP_ERR_OK;
}