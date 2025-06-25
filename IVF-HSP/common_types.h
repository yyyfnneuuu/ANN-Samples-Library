#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <vector>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <future>
#include <algorithm>

// 错误码定义
using APP_ERROR = int;
#define APP_ERR_OK 0
#define APP_ERR_INNER_ERROR -1

// 内存维度
enum Dims { DIMS_1, DIMS_2, DIMS_3 };

// ACL/Ascend NPU 相关API的占位符
#define ACL_MEMCPY_HOST_TO_DEVICE 1
#define ACL_MEMCPY_DEVICE_TO_HOST 2
#define ACL_SUCCESS 0
inline int aclrtMemcpy(void* dst, size_t destMax, const void* src, size_t count, int kind) { return 0; }
inline int aclrtSynchronizeStream(void* stream) { return 0; }
#define ACL_REQUIRE_OK(err) if(err != 0) throw std::runtime_error("ACL error");

// 日志和断言宏的占位符
#define APP_LOG_INFO(msg) std::cout << "INFO: " << msg << std::endl
#define APPERR_RETURN_IF(cond, err, msg) if(cond) return err
#define APPERR_RETURN_IF_NOT_FMT(cond, err, fmt, ...) if(!(cond)) return err
#define ASCEND_THROW_IF_NOT(cond) if(!(cond)) throw std::runtime_error("Ascend throw")
#define ASCEND_THROW_IF_NOT_FMT(cond, fmt, ...) if(!(cond)) throw std::runtime_error("Ascend throw fmt")

// 自定义数据结构和类的占位符
struct MemoryManager {};
struct Resources { MemoryManager& getMemoryManager() { static MemoryManager mm; return mm; }};
struct SearchParam { int nProbeL1; int nProbeL2; int l3SegmentNum; };
struct OpAttrs {}; // 算子属性
template<typename T, Dims D> class AscendTensor { /* ... implementation ... */ };
class ThreadPool {
public:
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;
};

// 常量占位符
const size_t SEARCH_PAGE_SIZE = 1024 * 1024;
const size_t SEARCH_VEC_SIZE = 4096;
const int CORE_NUM = 4;
const int FLAG_SIZE = 16;
const int BASE_SEG_SIZE = 1024;
const int VCMIN_SEG_SIZE = 256;
using float16_t = uint16_t; // 简化表示

#endif // COMMON_TYPES_H