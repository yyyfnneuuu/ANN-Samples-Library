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

    // 1. L1 Search
    AscendTensor<float16_t, DIMS_2> queryCodes(mem, {n, nList * subSpaceDimL1}, defaultStream);
    AscendTensor<uint16_t, DIMS_2> l1KIndicesNpu(mem, {n, searchParam->nProbeL1}, defaultStream);
    SearchBatchImplL1(queryNpu, queryCodes, l1KIndicesNpu);

    // 2. L2 Search
    AscendTensor<uint64_t, DIMS_2> labelL2(mem, {n, searchParam->nProbeL2}, defaultStream);
    SearchBatchImplMultiL2(queryCodes, l1KIndicesNpu, labelL2);
    std::vector<uint64_t> labelL2Cpu(n * searchParam->nProbeL2);
    aclrtMemcpy(labelL2Cpu.data(), labelL2Cpu.size() * sizeof(uint64_t), labelL2.data(), labelL2.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtSynchronizeStream(defaultStream); // 等待L2结果拷贝完成

    // 3. L3 Search (Host Calculation + Asynchronous Device Execution)

    // 为每个索引的L3输入/输出分配Device内存
    AscendTensor<uint64_t, DIMS_3> addressOffsetL3(mem, {indexSize, n, searchParam->nProbeL2 * 6}, defaultStream);
    AscendTensor<uint64_t, DIMS_3> idAdressL3(mem, {indexSize, n, searchParam->nProbeL2 * 2}, defaultStream);

    // 分配每个索引的原始TopK结果内存
    size_t out_multiplier = merge ? 1 : indexSize;
    AscendTensor<float16_t, DIMS_3> distResults(mem, {indexSize, n, k}, defaultStream);
    AscendTensor<int64_t, DIMS_3> labelResults(mem, {indexSize, n, k}, defaultStream);

    // 使用线程池并行计算每个索引的L3地址偏移
    std::vector<std::future<bool>> futures;
    for (int i = 0; i < indexSize; ++i) {
        futures.emplace_back(pool->enqueue([this, &indexes, n, i, &labelL2Cpu, &addressOffsetL3, &idAdressL3]() {
            // 在CPU端计算地址偏移
            std::vector<uint64_t> outOffset, outIdsOffset;
            CalculateOffsetL3(indexes, n, i, labelL2Cpu, outOffset, outIdsOffset);

            // 将计算好的地址偏移拷贝到Device上对应的Tensor slice
            AscendTensor<uint64_t, DIMS_2> addrSlice(addressOffsetL3.data() + i * n * ..., {n, ...});
            AscendTensor<uint64_t, DIMS_2> idSlice(idAdressL3.data() + i * n * ..., {n, ...});
            aclrtMemcpy(addrSlice.data(), ..., outOffset.data(), ..., ACL_MEMCPY_HOST_TO_DEVICE);
            aclrtMemcpy(idSlice.data(), ..., outIdsOffset.data(), ..., ACL_MEMCPY_HOST_TO_DEVICE);
            return true;
        }));
    }
    for(auto&& f : futures) f.get(); // 等待所有CPU计算和H2D拷贝完成

    // 异步下发所有索引的L3距离计算任务
    for (int i = 0; i < indexSize; ++i) {
        AscendTensor<uint64_t, DIMS_2> addrSlice(addressOffsetL3.data() + i * n * ..., {n, ...});
        AscendTensor<uint64_t, DIMS_2> idSlice(idAdressL3.data() + i * n * ..., {n, ...});
        AscendTensor<float16_t, DIMS_2> distSlice(distResults.data() + i * n * k, {n, k});
        AscendTensor<int64_t, DIMS_2> labelSlice(labelResults.data() + i * n * k, {n, k});

        SearchBatchImplL3(queryCodes, addrSlice, idSlice, distSlice, labelSlice);
    }

    if (merge) {
        // 调用NPU上的归并排序算子
        AscendTensor<float16_t, DIMS_2> finalDists(mem, {n, k}, defaultStream);
        AscendTensor<int64_t, DIMS_2> finalLabels(mem, {n, k}, defaultStream);
        RunMergeTopKOp(distResults, labelResults, finalDists, finalLabels);

        // 将合并后的结果拷贝回Host
        aclrtMemcpy(distances, ..., finalDists.data(), ..., ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(labels, ..., finalLabels.data(), ..., ACL_MEMCPY_DEVICE_TO_HOST);
    } else {
        aclrtMemcpy(distances, ..., distResults.data(), ..., ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(labels, ..., labelResults.data(), ..., ACL_MEMCPY_DEVICE_TO_HOST);
    }

    aclrtSynchronizeStream(defaultStream);
    return APP_ERR_OK;
}