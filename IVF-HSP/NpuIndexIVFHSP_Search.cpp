#include "NpuIndexIVFHSP.h"
#include <algorithm>

// 单索引搜索
APP_ERROR NpuIndexIVFHSP::Search(size_t nq, float* queryData, int topK, float* dists, int64_t* labels) const {
    size_t totalSize = nq * static_cast<size_t>(dim) * sizeof(float);
    size_t totalOutSize = nq * static_cast<size_t>(topK) * (sizeof(uint16_t) + sizeof(uint64_t));

    if (totalSize > SEARCH_PAGE_SIZE || nq > SEARCH_VEC_SIZE || totalOutSize > SEARCH_PAGE_SIZE) {
        size_t tileSize = GetSearchPagedSize(nq, topK);
        for (size_t i = 0; i < nq; i += tileSize) {
            size_t curNum = std::min(tileSize, nq - i);
            const_cast<NpuIndexIVFHSP&>(*this).SearchImpl(curNum, queryData + i * static_cast<size_t>(dim), topK,
                                                          dists + i * static_cast<size_t>(topK),
                                                          labels + i * static_cast<size_t>(topK));
        }
    } else {
        const_cast<NpuIndexIVFHSP&>(*this).SearchImpl(nq, queryData, topK, dists, labels);
    }
    return APP_ERR_OK;
}

// 单索引带掩码搜索
APP_ERROR NpuIndexIVFHSP::Search(size_t nq, uint8_t* mask, float* queryData, int topK, float* dists, int64_t* labels) const {
    size_t totalSize = static_cast<size_t>(nq) * static_cast<size_t>(dim) * sizeof(float);
    size_t totalOutSize = nq * static_cast<size_t>(topK) * (sizeof(uint16_t) + sizeof(uint64_t));

    if (totalSize > SEARCH_PAGE_SIZE || nq > SEARCH_VEC_SIZE || totalOutSize > SEARCH_PAGE_SIZE) {
        size_t tileSize = GetSearchPagedSize(nq, topK);
        for (size_t i = 0; i < nq; i += tileSize) {
            size_t curNum = std::min(tileSize, nq - i);
            const_cast<NpuIndexIVFHSP&>(*this).SearchImpl(curNum, mask, queryData + i * static_cast<size_t>(dim), topK,
                                                          dists + i * static_cast<size_t>(topK),
                                                          labels + i * static_cast<size_t>(topK));
        }
    } else {
        const_cast<NpuIndexIVFHSP&>(*this).SearchImpl(nq, mask, queryData, topK, dists, labels);
    }
    return APP_ERR_OK;
}

// vector版本
APP_ERROR NpuIndexIVFHSP::Search(std::vector<float>& queryData, int topK, std::vector<float>& dists, std::vector<int64_t>& labels) const {
    size_t nq = queryData.size() / dim;
    APPERR_RETURN_IF(nq == 0, APP_ERR_OK);
    ASCEND_THROW_IF_NOT(nq * dim == queryData.size());
    ASCEND_THROW_IF_NOT(topK > 0);
    ASCEND_THROW_IF_NOT(nq * topK == dists.size());
    ASCEND_THROW_IF_NOT(nq * topK == labels.size());

    return Search(nq, queryData.data(), topK, dists.data(), labels.data());
}

// 多索引搜索
APP_ERROR NpuIndexIVFHSP::Search(const std::vector<NpuIndexIVFHSP*>& indexes, size_t nq, float* queryData, int topK, float* dists, int64_t* labels, bool merge) {
    if (indexes.empty()) return APP_ERR_OK;
    int indexSize = indexes.size();
    ACL_REQUIRE_OK(ResetMultiL3TopKOp(indexSize));

    size_t totalSize = nq * static_cast<size_t>(indexes[0]->dim) * sizeof(float);
    size_t totalOutSize = nq * static_cast<size_t>(topK) * (sizeof(uint16_t) + sizeof(uint64_t));

    // 使用第一个索引作为代理来调用非静态的 SearchImpl
    NpuIndexIVFHSP& proxy = *indexes[0];

    if (totalSize > SEARCH_PAGE_SIZE || nq > SEARCH_VEC_SIZE || totalOutSize > SEARCH_PAGE_SIZE) {
        size_t tileSize = proxy.GetSearchPagedSize(nq, topK);
        for (size_t i = 0; i < nq; i += tileSize) {
            size_t curNum = std::min(tileSize, nq - i);
            size_t out_offset = merge ? (i * static_cast<size_t>(topK)) : (i * indexSize * static_cast<size_t>(topK));
            proxy.SearchImpl(indexes, curNum, queryData + i * static_cast<size_t>(proxy.dim), topK,
                             dists + out_offset, labels + out_offset, merge);
        }
    } else {
        proxy.SearchImpl(indexes, nq, queryData, topK, dists, labels, merge);
    }
    return APP_ERR_OK;
}

// 多索引带掩码搜索
APP_ERROR NpuIndexIVFHSP::Search(const std::vector<NpuIndexIVFHSP*>& indexes, size_t nq, uint8_t* mask, float* queryData, int topK, float* dists, int64_t* labels, bool merge) {
    if (indexes.empty()) return APP_ERR_OK;
    int indexSize = indexes.size();
    ACL_REQUIRE_OK(ResetMultiL3TopKOp(indexSize));

    size_t totalSize = nq * static_cast<size_t>(indexes[0]->dim) * sizeof(float);
    size_t totalOutSize = nq * static_cast<size_t>(topK) * (sizeof(uint16_t) + sizeof(uint64_t));

    NpuIndexIVFHSP& proxy = *indexes[0];

    if (totalSize > SEARCH_PAGE_SIZE || nq > SEARCH_VEC_SIZE || totalOutSize > SEARCH_PAGE_SIZE) {
        size_t tileSize = proxy.GetSearchPagedSize(nq, topK);
        for (size_t i = 0; i < nq; i += tileSize) {
            size_t curNum = std::min(tileSize, nq - i);
            size_t out_offset = merge ? (i * static_cast<size_t>(topK)) : (i * indexSize * static_cast<size_t>(topK));
            proxy.SearchImpl(indexes, curNum, mask, queryData + i * static_cast<size_t>(proxy.dim), topK,
                             dists + out_offset, labels + out_offset, merge);
        }
    } else {
        proxy.SearchImpl(indexes, nq, mask, queryData, topK, dists, labels, merge);
    }
    return APP_ERR_OK;
}