#include "NpuIndexIVFHSP.h"
#include <numeric>

// 单索引实现
void NpuIndexIVFHSP::SearchImpl(int n, const float* x, int k, float* distances, int64_t* labels) {
    auto& mem = resources->getMemoryManager();
    this->maskFlag = false;

    AscendTensor<float, DIMS_2> queryNpu(mem, {n, dim}, defaultStream);
    auto ret = aclrtMemcpy(queryNpu.data(), queryNpu.getSizeInBytes(), x, n * dim * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    size_t searchCnt = 0;
    std::vector<float16_t> distHalf(n * k);
    for (auto batchSize : opAccessBatchList) {
        while (n - searchCnt >= batchSize) {
            AscendTensor<float, DIMS_2> queryTmpNpu(queryNpu.data() + searchCnt * dim, {batchSize, dim});
            ret = SearchBatchImpl(batchSize, queryTmpNpu, k, distHalf.data() + searchCnt * k, labels + searchCnt * k);
            ASCEND_THROW_IF_NOT(ret == APP_ERR_OK);
            searchCnt += batchSize;
        }
    }

    std::transform(distHalf.begin(), distHalf.end(), distances, [](float16_t temp) { return static_cast<float>(temp); /* Simplified conversion */ });
}

// 单索引带掩码实现
void NpuIndexIVFHSP::SearchImpl(int n, const uint8_t* mask, const float* x, int k, float* distances, int64_t* labels) {
    auto& mem = resources->getMemoryManager();
    this->maskFlag = true;

    AscendTensor<float, DIMS_2> queryNpu(mem, {n, dim}, defaultStream);
    aclrtMemcpy(queryNpu.data(), queryNpu.getSizeInBytes(), x, n * dim * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    AscendTensor<uint8_t, DIMS_1> maskBitNpu(mem, {static_cast<int>(n * ((ntotal + 7) / 8))}, defaultStream);
    aclrtMemcpy(maskBitNpu.data(), maskBitNpu.getSizeInBytes(), mask, n * ((ntotal + 7) / 8) * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);

    size_t searchCnt = 0;
    std::vector<float16_t> distHalf(n * k);
    int batchSize = 1; // Masking search may have different batching logic
    while (n - searchCnt >= batchSize) {
        AscendTensor<float, DIMS_2> queryTmpNpu(queryNpu.data() + searchCnt * dim, {batchSize, dim});
        AscendTensor<uint8_t, DIMS_1> maskBitTmpNpu(maskBitNpu.data() + searchCnt * ((ntotal + 7) / 8), {static_cast<int>(batchSize * (ntotal + 7) / 8)});
        auto ret = SearchBatchImpl(batchSize, maskBitTmpNpu, queryTmpNpu, k, distHalf.data() + searchCnt * k, labels + searchCnt * k);
        ASCEND_THROW_IF_NOT(ret == APP_ERR_OK);
        searchCnt += batchSize;
    }

    std::transform(distHalf.begin(), distHalf.end(), distances, [](float16_t temp) { return static_cast<float>(temp); /* Simplified conversion */ });
}

// 多索引实现
void NpuIndexIVFHSP::SearchImpl(const std::vector<NpuIndexIVFHSP*>& indexes, int n, const float* x, int k, float* distances, int64_t* labels, bool merge) {
    auto& mem = resources->getMemoryManager();
    this->maskFlag = false;

    AscendTensor<float, DIMS_2> queryNpu(mem, {n, dim}, defaultStream);
    aclrtMemcpy(queryNpu.data(), queryNpu.getSizeInBytes(), x, n * dim * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    size_t searchCnt = 0;
    size_t out_multiplier = merge ? 1 : indexes.size();
    std::vector<float16_t> distHalf(n * out_multiplier * k);

    for (auto batchSize : opAccessBatchList) {
        while (n - searchCnt >= batchSize) {
            AscendTensor<float, DIMS_2> queryTmpNpu(queryNpu.data() + searchCnt * dim, {batchSize, dim});
            auto ret = SearchBatchImpl(indexes, batchSize, queryTmpNpu, k,
                                       distHalf.data() + searchCnt * out_multiplier * k,
                                       labels + searchCnt * out_multiplier * k, merge);
            ASCEND_THROW_IF_NOT(ret == APP_ERR_OK);
            searchCnt += batchSize;
        }
    }

    std::transform(distHalf.begin(), distHalf.end(), distances, [](float16_t temp) { return static_cast<float>(temp); /* Simplified conversion */ });
}

// 多索引带掩码实现
void NpuIndexIVFHSP::SearchImpl(const std::vector<NpuIndexIVFHSP*>& indexes, int n, const uint8_t* mask, const float* x, int k, float* distances, int64_t* labels, bool merge) {
    auto& mem = resources->getMemoryManager();
    this->maskFlag = true;

    AscendTensor<float, DIMS_2> queryNpu(mem, {n, dim}, defaultStream);
    aclrtMemcpy(queryNpu.data(), queryNpu.getSizeInBytes(), x, n * dim * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    size_t searchCnt = 0;
    size_t out_multiplier = merge ? 1 : indexes.size();
    std::vector<float16_t> distHalf(n * out_multiplier * k);
    int batchSize = 1; // Masking search may have different batching logic

    while (n - searchCnt >= batchSize) {
        AscendTensor<float, DIMS_2> queryTmpNpu(queryNpu.data() + searchCnt * dim, {batchSize, dim});
        auto ret = SearchBatchImpl(indexes, batchSize, mask + searchCnt * ((ntotal + 7) / 8),
                                   queryTmpNpu, k, distHalf.data() + searchCnt * out_multiplier * k,
                                   labels + searchCnt * out_multiplier * k, merge);
        ASCEND_THROW_IF_NOT(ret == APP_ERR_OK);
        searchCnt += batchSize;
    }

    std::transform(distHalf.begin(), distHalf.end(), distances, [](float16_t temp) { return static_cast<float>(temp); /* Simplified conversion */ });
}