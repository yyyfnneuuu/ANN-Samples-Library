// hnswbuild_internal需要被重写，核心是CreateGraphPages的替代者。

static void FlushRabitqDataAndGraph(HnswBuildState* buildstate) {
    // ...
    // 1. 分配独立的 "data" 和 "graph" 页面
    BlockNumber data_blk = HNSW_METAPAGE_BLKNO + 1;
    // ... 计算需要多少数据页和图页 ...
    BlockNumber graph_blk = data_blk + num_data_pages;

    // 2. 顺序写入所有RabitqItemData
    // 遍历内存中的所有HnswElement
    HnswElementPtr iter = buildstate->graph->head;
    while (!HnswPtrIsNull(base, iter)) {
        HnswElement element = (HnswElement)HnswPtrAccess(base, iter);
        // 获取向量，旋转，编码
        // ...
        // 将RabitqItem写入当前数据页，如果页面满了就换下一页
        // 这种方式保证了物理存储的连续性
    }

    // 3. 顺序写入所有NeighborTuple
    // 遍历所有HnswElement，将其邻居列表写入图页面
    // ...

    // 4. 更新元数据页，记录data和graph的起始块号
    HnswUpdateMetaPage(..., data_blk, graph_blk);
}