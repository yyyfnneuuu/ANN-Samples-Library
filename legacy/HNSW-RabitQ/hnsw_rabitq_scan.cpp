List* HnswRabitqSearchLayer(int ef, int lc) {
// 初始化 W (candidates) 和 C (visited)

while (!pairingheap_is_empty(C)) {
//  从C中取出最近的节点c
//  判断是否终止

// 获取c的邻居列表 (随机I/O，读取一个图页面)
HnswLoadUnvisitedNeighbors();

// 核心改造点: 批量处理邻居

// 1. 将一批unvisited邻居的RabitqItemData所在页面预取到内存
//   因为RabitqItemData是连续存储的，I/O更友好

// 2. 将这批邻居的码字打包
NeonPackCodes();

// 3. 调用NeonFastScanAccumulate一次性计算所有邻居的累加值
uint16_t partial_distances[BATCH_SIZE];
NeonFastScanAccumulate(packed_codes, query_lut, partial_distances);

// 4. 计算最终的估算距离并更新W和C
for (int i = 0; i < unvisited_len; ++i) {
// 从对应的RabitqItemData中读取f_add, f_rescale等因子
float f_add = , f_rescale = ;

// 估算距离 = f_add + f_rescale * (query_ip_offset + partial_distances[i]);
float estimated_dist = ;

// 无重排: 直接使用估算距离进行比较和剪枝
if (estimated_dist < f->distance || wlen < ef) {
// 将新候选者加入 W 和 C
}
}
}
// 返回W
}