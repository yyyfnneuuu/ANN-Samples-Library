//定义新的页内数据结构和文件组织方式
#pragma once

#include "postgres.h"
#include "access/itemptr.h"

// RabitQ量化后的数据单元，代替原始向量存储
// 这个结构会被连续存储在一个或多个专门的datablock页面上
typedef struct RabitqItemData
{
    // RabitQ 码字 (1 bit + 3 bits = 4 bits per dim, uint8_t可以存2个维度的码字)
    uint8_t     codes[FLEXIBLE_ARRAY_MEMBER];
    // 后续紧跟3个float因子：f_add, f_rescale, f_error
} RabitqItemData;

#define RABITQ_ITEM_SIZE(dim, total_bits) \
    (MAXALIGN(sizeof(uint8_t) * ((dim) * (total_bits) + 7) / 8) + sizeof(float) * 3)

// HNSW图的邻居列表依然存储在独立的"graphblock"页面上
// 但其ItemPointer现在指向的是RabitqItemData所在的页面和偏移
typedef struct RabitqNeighborTupleData
{
    uint16      type; // 标识为邻居元组
    uint16      count; // 邻居总数
    ItemPointerData indextids[FLEXIBLE_ARRAY_MEMBER];
} RabitqNeighborTupleData;

// 元数据页，增加RabitQ相关信息
typedef struct RabitqMetaPageData
{
    uint32      magicNumber;
    uint32      version;
    uint16      dimensions;
    uint16      padded_dim; // FHT旋转后的对齐维度
    uint8       m;
    uint8       total_bits; // RabitQ总比特数
    // 其他元数据
    BlockNumber entryBlkno;
    OffsetNumber entryOffno;
    int32       entryLevel;

    BlockNumber dataStartBlkno;   // 量化数据块的起始页面
    BlockNumber graphStartBlkno;  // 图结构块的起始页面
} RabitqMetaPageData;