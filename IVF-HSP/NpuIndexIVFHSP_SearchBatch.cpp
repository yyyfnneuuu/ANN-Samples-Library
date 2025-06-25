#include "NpuIndexIVFHSP.h"

// 单索引批处理
APP_ERROR NpuIndexIVFHSP::SearchBatchImpl(int n, AscendTensor<float, DIMS_2>& queryNpu, int k, float16_t* distances, int64_t* labels) {
    auto& mem = resources->getMemoryManager();

    // 1. L1 Search
    AscendTensor<float16_t, DIMS_2> queryCodes(mem, {n, nList * subSpaceDimL1}, defaultStream);
    AscendTensor<uint16_t, DIMS_2> l1KIndicesNpu(mem, {n, searchParam->nProbeL1}, defaultStream);
    APPERR_RETURN_IF_NOT_FMT(SearchBatchImplL1(queryNpu, queryCodes, l1KIndicesNpu) == APP_ERR_OK, APP_ERR_INNER_ERROR, "L1 search failed");

    // 2. L2 Search
    AscendTensor<uint64_t, DIMS_2> addressOffsetL3(mem, {n, searchParam->nProbeL2 * 6}, defaultStream);
    AscendTensor<uint64_t, DIMS_2> idAdressL3(mem, {n, searchParam->nProbeL2 * 2}, defaultStream);
    APPERR_RETURN_IF_NOT_FMT(SearchBatchImplL2(queryCodes, l1KIndicesNpu, addressOffsetL3, idAdressL3) == APP_ERR_OK, APP_ERR_INNER_ERROR, "L2 search failed");

    // 3. L3 Search
    AscendTensor<float16_t, DIMS_2> outDists(mem, {n, k}, defaultStream);
    AscendTensor<int64_t, DIMS_2> outlabels(mem, {n, k}, defaultStream);
    APPERR_RETURN_IF_NOT_FMT(SearchBatchImplL3(queryCodes, addressOffsetL3, idAdressL3, outDists, outlabels) == APP_ERR_OK, APP_ERR_INNER_ERROR, "L3 search failed");

    // 4. Copy results back to host
    aclrtMemcpy(distances, n * k * sizeof(float16_t), outDists.data(), outDists.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(labels, n * k * sizeof(int64_t), outlabels.data(), outlabels.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);

    return APP_ERR_OK;
}

// 单索引带掩码批处理
APP_ERROR NpuIndexIVFHSP::SearchBatchImpl(int n, AscendTensor<uint8_t, DIMS_1>& maskBitNpu, AscendTensor<float, DIMS_2>& queryNpu, int k, float16_t* distances, int64_t* labels) {
    auto& mem = resources->getMemoryManager();

    // 1. L1 Search
    AscendTensor<float16_t, DIMS_2> queryCodes(mem, {n, nList * subSpaceDimL1}, defaultStream);
    AscendTensor<uint16_t, DIMS_2> l1KIndicesNpu(mem, {n, searchParam->nProbeL1}, defaultStream);
    APPERR_RETURN_IF_NOT_FMT(SearchBatchImplL1(queryNpu, queryCodes, l1KIndicesNpu) == APP_ERR_OK, APP_ERR_INNER_ERROR, "L1 search failed");

    // 2. L2 Search with Mask
    AscendTensor<uint64_t, DIMS_2> addressOffsetL3(mem, {n, searchParam->nProbeL2 * 6}, defaultStream);
    AscendTensor<uint64_t, DIMS_2> idAdressL3(mem, {n, searchParam->nProbeL2 * 2}, defaultStream);
    APPERR_RETURN_IF_NOT_FMT(SearchBatchImplL2(maskBitNpu, queryCodes, l1KIndicesNpu, addressOffsetL3, idAdressL3) == APP_ERR_OK, APP_ERR_INNER_ERROR, "L2 search with mask failed");

    // 3. L3 Search (L3 op handles mask internally)
    AscendTensor<float16_t, DIMS_2> outDists(mem, {n, k}, defaultStream);
    AscendTensor<int64_t, DIMS_2> outlabels(mem, {n, k}, defaultStream);
    APPERR_RETURN_IF_NOT_FMT(SearchBatchImplL3(queryCodes, addressOffsetL3, idAdressL3, outDists, outlabels) == APP_ERR_OK, APP_ERR_INNER_ERROR, "L3 search failed");

    // 4. Copy results & Post-process
    aclrtMemcpy(distances, n * k * sizeof(float16_t), outDists.data(), outDists.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(labels, n * k * sizeof(int64_t), outlabels.data(), outlabels.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);

#pragma omp parallel for
    for (int i = 0; i < n * k; ++i) { if (distances[i] == 100.0) { labels[i] = -1; } }

    return APP_ERR_OK;
}


// 多索引批处理
APP_ERROR NpuIndexIVFHSP::SearchBatchImpl(const std::vector<NpuIndexIVFHSP*>& indexes, int n, AscendTensor<float, DIMS_2>& queryNpu, int k, float16_t* distances, int64_t* labels, bool merge) {
    auto& mem = resources->getMemoryManager();
    int indexSize = static_cast<int>(indexes.size());

    // 1. L1 Search (Common for all indexes)
    AscendTensor<float16_t, DIMS_2> queryCodes(mem, {n, nList * subSpaceDimL1}, defaultStream);
    AscendTensor<uint16_t, DIMS_2> l1KIndicesNpu(mem, {n, searchParam->nProbeL1}, defaultStream);
    SearchBatchImplL1(queryNpu, queryCodes, l1KIndicesNpu);

    // 2. L2 Search (Multi-index)
    AscendTensor<uint64_t, DIMS_2> labelL2(mem, {n, searchParam->nProbeL2}, defaultStream);
    SearchBatchImplMultiL2(queryCodes, l1KIndicesNpu, labelL2);

    std::vector<uint64_t> labelL2Cpu(n * searchParam->nProbeL2);
    aclrtMemcpy(labelL2Cpu.data(), labelL2Cpu.size() * sizeof(uint64_t), labelL2.data(), labelL2.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);

    // 3. L3 Search (Host Calculation + Asynchronous Device Execution)
    // ... 此处省略了复杂的地址计算、多线程、异步算子下发和结果合并的详细逻辑 ...

    // 4. Post-processing and Merging Results

    return APP_ERR_OK;
}

// 多索引带掩码批处理
APP_ERROR NpuIndexIVFHSP::SearchBatchImpl(const std::vector<NpuIndexIVFHSP*>& indexes, int n, const uint8_t* mask, AscendTensor<float, DIMS_2>& queryNpu, int k, float16_t* distances, int64_t* labels, bool merge) {
    // 结构与不带掩码的多索引版本类似，但在地址计算时会传入掩码

    return APP_ERR_OK;
}