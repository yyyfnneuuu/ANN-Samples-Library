#ifndef NPU_INDEX_IVFHSP_H
#define NPU_INDEX_IVFHSP_H

#include "common_types.h"

class NpuIndexIVFHSP {
public:

    /**
     * @brief 构造函数
     */
    NpuIndexIVFHSP(int dim, int ntotal);

    /**
     * @brief 对单个索引进行向量检索
     */
    APP_ERROR Search(size_t nq, float* queryData, int topK, float* dists, int64_t* labels) const;

    /**
     * @brief 使用C++ vector容器进行向量检索
     */
    APP_ERROR Search(std::vector<float>& queryData, int topK, std::vector<float>& dists, std::vector<int64_t>& labels) const;

    /**
     * @brief 对单个索引进行带掩码的向量检索
     */
    APP_ERROR Search(size_t nq, uint8_t* mask, float* queryData, int topK, float* dists, int64_t* labels) const;

    /**
     * @brief 对多个索引进行向量检索，可选合并结果
     */
    static APP_ERROR Search(const std::vector<NpuIndexIVFHSP*>& indexes, size_t nq, float* queryData, int topK, float* dists, int64_t* labels, bool merge = false);

    /**
     * @brief 对多个索引进行带掩码的向量检索，可选合并结果
     */
    static APP_ERROR Search(const std::vector<NpuIndexIVFHSP*>& indexes, size_t nq, uint8_t* mask, float* queryData, int topK, float* dists, int64_t* labels, bool merge = false);


private:
    int dim;
    int ntotal;
    int nList;
    int nListL2;
    int subSpaceDimL1;
    bool maskFlag;
    bool isAddWithIds;
    void* defaultStream;
    void* aiCpuStream;
    std::vector<int> opAccessBatchList;

    Resources* resources;
    SearchParam* searchParam;
    ThreadPool* pool;
    AscendTensor<float, DIMS_2>* codeBooksShapedL1Npu;
    AscendTensor<float, DIMS_2>* codeBooksShapedL2Npu;
    OpAttrs* searchL1OpAttrs;
    OpAttrs* searchL2OpAttrs;
    OpAttrs* searchL3OpAttrs;
    AscendTensor<uint8_t, DIMS_1>* maskByteNpu;
    std::vector<uint8_t> maskByteCpu;

    // --- 辅助函数 ---
    size_t GetSearchPagedSize(size_t nq, int topK) const;
    static APP_ERROR ResetMultiL3TopKOp(int indexSize);
    std::vector<int64_t> GetIdMap() const; // 占位符

    // --- 实现函数 ---
    void SearchImpl(int n, const float* x, int k, float* distances, int64_t* labels);
    void SearchImpl(int n, const uint8_t* mask, const float* x, int k, float* distances, int64_t* labels);

    // 静态成员函数需要一个实例来访问非静态成员，这里将其改为非静态
    void SearchImpl(const std::vector<NpuIndexIVFHSP*>& indexes, int n, const float* x, int k, float* distances, int64_t* labels, bool merge);
    void SearchImpl(const std::vector<NpuIndexIVFHSP*>& indexes, int n, const uint8_t* mask, const float* x, int k, float* distances, int64_t* labels, bool merge);

    // --- 批处理核心逻辑 ---
    APP_ERROR SearchBatchImpl(int n, AscendTensor<float, DIMS_2>& queryNpu, int k, float16_t* distances, int64_t* labels);
    APP_ERROR SearchBatchImpl(int n, AscendTensor<uint8_t, DIMS_1>& maskBitNpu, AscendTensor<float, DIMS_2>& queryNpu, int k, float16_t* distances, int64_t* labels);
    APP_ERROR SearchBatchImpl(const std::vector<NpuIndexIVFHSP*>& indexes, int n, AscendTensor<float, DIMS_2>& queryNpu, int k, float16_t* distances, int64_t* labels, bool merge);
    APP_ERROR SearchBatchImpl(const std::vector<NpuIndexIVFHSP*>& indexes, int n, const uint8_t* mask, AscendTensor<float, DIMS_2>& queryNpu, int k, float16_t* distances, int64_t* labels, bool merge);

    // --- L1/L2/L3 流水线阶段 ---
    APP_ERROR SearchBatchImplL1(AscendTensor<float, DIMS_2>& queriesNpu, AscendTensor<float16_t, DIMS_2>& queryCodes, AscendTensor<uint16_t, DIMS_2>& l1KIndiceNpu);
    APP_ERROR SearchBatchImplL2(AscendTensor<float16_t, DIMS_2>& queryCodesNpu, AscendTensor<uint16_t, DIMS_2>& l1KIndicesNpu, AscendTensor<uint64_t, DIMS_2>& addressOffsetOfBucketL3, AscendTensor<uint64_t, DIMS_2>& idAdressL3);
    APP_ERROR SearchBatchImplL2(AscendTensor<uint8_t, DIMS_1>& maskBitNpu, AscendTensor<float16_t, DIMS_2>& queryCodesNpu, AscendTensor<uint16_t, DIMS_2>& l1KIndicesNpu, AscendTensor<uint64_t, DIMS_2>& addressOffsetOfBucketL3, AscendTensor<uint64_t, DIMS_2>& idAdressL3);
    APP_ERROR SearchBatchImplMultiL2(AscendTensor<float16_t, DIMS_2>& queryCodesNpu, AscendTensor<uint16_t, DIMS_2>& l1KIndicesNpu, AscendTensor<uint64_t, DIMS_2>& indicesL2);
    APP_ERROR SearchBatchImplL3(AscendTensor<float16_t, DIMS_2>& queryCodes, AscendTensor<uint64_t, DIMS_2>& addressOffsetOfBucketL3, AscendTensor<uint64_t, DIMS_2>& idAddressOfBucketL3, AscendTensor<float16_t, DIMS_2>& outDists, AscendTensor<int64_t, DIMS_2>& outIndices);
    APP_ERROR SearchBatchImplMultiL3(const std::vector<NpuIndexIVFHSP*>& indexes, int i, AscendTensor<float16_t, DIMS_2>& queryCodes, AscendTensor<uint64_t, DIMS_3>& addressOffsetOfBucketL3, AscendTensor<uint64_t, DIMS_3>& idAddressOfBucketL3, AscendTensor<float16_t, DIMS_3>& distResult, AscendTensor<float16_t, DIMS_3>& vcMinDistResult, AscendTensor<uint16_t, DIMS_3>& opFlag);

    // --- Host端并行计算任务 ---
    bool CalculateOffsetL3(const std::vector<NpuIndexIVFHSP*>& indexes, int n, int indexId, const std::vector<uint64_t>& labelL2Cpu, std::vector<uint64_t>& outOffset, std::vector<uint64_t>& outIdsOffset);
    bool CalculateOffsetL3WithMask(const std::vector<NpuIndexIVFHSP*>& indexes, int n, const uint8_t* mask, int indexId, const std::vector<uint64_t>& labelL2Cpu, std::vector<uint64_t>& outOffset, std::vector<uint64_t>& outIdsOffset);

    // --- NPU算子执行的封装 (占位符) ---
    void RunL1DistOp(int, AscendTensor<float, DIMS_2>&, AscendTensor<float, DIMS_2>&, AscendTensor<float16_t, DIMS_2>&, AscendTensor<float16_t, DIMS_2>&, AscendTensor<uint16_t, DIMS_2>&);
    // ... 其他 Run*Op 方法声明
};

#endif // NPU_INDEX_IVFHSP_H