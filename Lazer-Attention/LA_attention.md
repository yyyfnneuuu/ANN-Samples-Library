# Optimized Attention Forward On Ascend NPU

这是一个为华为昇腾 NPU 设计的高性能 Attention 前向传播 C++ 实现。代码利用了昇腾的 AI Core (Cube) 和 Vector 的硬件特性，通过内存分级、计算流水线和算法融合等实现了高效的计算。

## 核心逻辑

整个前向计算过程被设计为一个多阶段的流水线，由 AI Core 和 Vector Core 协同完成。

1. **阶段一：Q x Kᵀ (AI Core)**

   - 该阶段在 AI Core 上执行，利用矩阵乘法MAD能力。
   - Q 和 K 矩阵被切分为 `128x128` 或 `128x192` 的基本块Blocks。
   - 通过 `AddrMapping_forward` 模块计算每个核负责处理的块的精确内存地址。
   - 计算结果 Attention Scores 被写回 Global Memory 的 Workspace 中。
2. **阶段二：Stable Softmax (Vector Core)**

   - Vector Core 从 GM 中读取 Attention Scores。
   - 为了保证数值稳定性，Softmax 进行最大值稳定：

     1. **Find Max**：逐行找到 Attention Score 的最大值。
     2. **Scale & Subtract**：将所有分数乘以 `1/sqrt(d_k)` 并减去该行的最大值。
     3. **Exponentiate**：计算 `exp()`。
     4. **Calculate Sum**：计算 `exp()` 结果的行和（`row_sum`）。
3. **阶段三：Scores x V & Row Sum (AI Core)**

   - 在 AI Core 上执行的**融合计算**步骤。
   - AI Core 从 GM 读取经过 Softmax 但未归一化的 Attention Scores。
   - **同时**执行两个计算：
     - 矩阵乘法：`Attention_Scores @ V`
     - 行和累加：`sum(Attention_Scores)`
   - Scores 矩阵只需从 GM 读取一次，显著减少了内存带宽占用，提升了效率。`sum(Attention_Scores)` 的结果就是上一阶段的 `row_sum`。
4. **阶段四：Normalization & LogSum (Vector Core)**

   - Vector Core 读取阶段三计算出的 `O` 矩阵和 `rowsum` 向量。
   - 执行最终的归一化操作：`Output = O / rowsum`。
   - 同时，为了反向传播的需要，计算 `log(rowsum)` 并与之前得到的行最大值相加，存入 `softmax_log_max_sum_gm`。

## 关键改进与优化点

- **分块与流水线**：将大矩阵分解为小块，并使用 Ping-Pong Buffering 技术在不同内存层级（GM -> L1 -> L0）之间创建数据处理流水线，从而掩盖数据传输延迟。
- **操作融合**：将 `Scores @ V` 和 `row_sum` 的计算合并，显著减少了对 Global Memory 的读写次数。
- **自适应序列长度策略**：为短、中、长序列设计了不同的 UB 内存布局和计算流程，确保在任何场景下都能高效利用宝贵的片上缓存。
- **混合精度**：支持使用 `bfloat16` 进行计算和存储，但在需要更高精度的累加步骤中使用 `float32`，实现了性能和准确性的平衡。

## 文件结构

- `constants.h`: 所有常量、数据类型和共享的结构体。
- `addressing.h`: 用于地址计算和负载均衡的 `AddrMapping_forward` 类。
- `cube_operations.h`: 实现在 AI Core 上运行的 `CubeForward` 类，负责所有矩阵乘法和融合操作。
- `vector_operations.h`: 实现在 Vector Core 上运行的 `VectorForward` 类，负责 Softmax 和归一化。
- `attention_kernel.cpp`: 内核入口函数 `ascend_laser_attention`，根据硬件类型分派任务给 `CubeForward` 或 `VectorForward`。
