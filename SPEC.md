## 一、数据准备

**数据集选择**

使用论文中的15个数据集，按类型分三组：

```
数值型（Numerical）：
- Amazon Access Samples (AMZ) - UCI
- Bike Sharing (BKS) - UCI
- Census Income (CES) - UCI
- Spambase (SPM) - UCI
- Wikipedia Math (WKM) - Kaggle

多媒体型（Multimedia）：
- Fashion-MNIST (FMN) - Kaggle
- MNIST (MNI) - 可直接从sklearn获取
- Fruit Images (FRT) - Kaggle
- Rice Images (RIE) - Kaggle

文本型（Textual）：
- Brazilian E-Commerce (BEP) - Kaggle
- Twitter Sentiment (TSC) - Kaggle
- Health News Tweets (HNT) - Kaggle
- IMDb (IMD) - Kaggle
- Reuters Text (RTC) - UCI
- Infobox (INB) - Kaggle
```

**数据预处理**

每个数据集按以下步骤处理：

```
1. 读取原始数据
2. 转换成二进制 bit 流
   - 数值型：直接用 numpy 的 unpackbits
   - 图像型：像素值转 uint8 再 unpackbits
   - 文本型：UTF-8 编码转 bytes 再 unpackbits
3. 按 512 bits 顺序切块，不足 512 bits 的尾部丢弃
4. 前 50000 个 block 作为初始 stale blocks
5. 后 50000 个 block 作为新写请求序列
6. 写请求的时间间隔按泊松分布生成（lambda 参数参考论文）
```

---

## 二、基础框架

**PCM 参数（严格按论文 Table II）**

```
SLC PCM：
- WD 错误率 wordline：9.9%
- WD 错误率 bitline：11.5%
- 读延迟：100ns
- RESET 写延迟：100ns
- SET 写延迟：200ns
- 读能耗：1.075 nJ/cacheline
- RESET 能耗：0.0137 nJ/bit
- SET 能耗：0.0268 nJ/bit

MLC PCM（Exp#5用）：
- WD 错误率 '00'：12.3%
- WD 错误率 '01'：15.2%
- WD 错误率 '11'：27.6%
```

**WD-prone cells 计算**

```
输入：new_block（512bits），stale_block（512bits）
输出：wd_prone_count（整数）

对每个位置 i（0到511）：
    先用差分写确定实际需要写的位：
        changed_bits = new_block XOR stale_block
    
    对每个 changed_bit 为1且 new_block[i]=0 的位置
    （即这个位置发生了 1→0 的 RESET）：
        检查左邻居 i-1：
            如果 stale_block[i-1]=0 且 changed_bits[i-1]=0
            （邻居存'0'且是idle的）
            → wd_prone_count += 1
        检查右邻居 i+1：
            同上条件
            → wd_prone_count += 1
```

**WD 错误模拟**

```
输入：wd_prone_count
输出：actual_wd_errors

wordline_errors = Binomial(wd_prone_count, 0.099)
bitline_errors = Binomial(wd_prone_count, 0.115)
actual_wd_errors = wordline_errors + bitline_errors
```

**写延迟计算**

```
输入：new_block，stale_block，vnr_count（VnR校验次数）
输出：write_latency（ns）

reset_count = 被写成'0'的bit数
            = sum(stale_block[i]=1 且 new_block[i]=0 的位置数)
set_count = 被写成'1'的bit数
          = sum(stale_block[i]=0 且 new_block[i]=1 的位置数)

base_latency = reset_count × 100 + set_count × 200
vnr_latency = vnr_count × (读延迟 + 校验延迟)
write_latency = base_latency + vnr_latency
```

**写能耗计算**

```
输入：new_block，stale_block
输出：write_energy（nJ）

reset_bits = reset_count（同上）
set_bits = set_count（同上）
write_energy = reset_bits × 0.0137 + set_bits × 0.0268
```

**写耐久度（write cost）计算**

```
输入：new_block，stale_block
输出：write_cost

按论文的非对称 cost model：
write_cost = reset_count × 2 + set_count × 1
```

**Stale Table 结构**

```
每条 entry 包含：
- physical_address：stale block 的物理地址（用数组下标模拟）
- cluster_id：属于哪个簇（整数，0到k-1）
- minhash_values：h个哈希值（默认8个，每个3bits）

操作：
- insert(address, cluster_id, minhash_values)
- delete(address)
- query_by_cluster(cluster_id)：返回该簇所有entry
- update_page_table(old_addr, new_addr)：覆写后更新页表
```

**重训练触发机制**

```
维护一个计数器 overwrite_counter
每完成一次覆写 → overwrite_counter += 1
当 overwrite_counter >= 20000 → 触发重训练
    重训练使用当前 Stale Table 里所有 stale blocks
    重训练完成后 overwrite_counter 归零
重训练在后台异步执行，不阻塞写操作
```

---

## 三、编码方案实现

**DCW（Data Comparison Write）**

```
输入：new_block，stale_block
输出：encoded_block，actual_write_mask

逻辑：
    actual_write_mask = new_block XOR stale_block
    只写 actual_write_mask 中为1的那些位
    encoded_block = new_block（内容不变，只减少写操作）

WD-prone cells 计算时只考虑 actual_write_mask=1 的位置
```

**DMPart**

```
输入：new_block（512bits）
输出：encoded_block

参数：partition_size = 2bits（每个partition 2个bit）

步骤：
1. 把 new_block 按每2bits切成256个partition
2. 统计4种可能pattern（00,01,10,11）各出现多少次
3. 选出现频率最低的pattern作为mask_pattern
4. 用 mask_pattern 对每个partition做XOR编码
5. 在encoded_block头部加1个标志位记录用的是哪种mask_pattern
   （需要额外的元数据空间，论文编码粒度32bits）

WD-prone cells 计算：
    用 encoded_block 和 stale_block 计算
    而不是用原始 new_block
```

**MinWD**

```
输入：new_block（512bits），stale_block（512bits）
输出：encoded_block

步骤：
1. 生成4种候选编码：
   候选1：new_block 原始
   候选2：new_block 全部取反
   候选3：new_block 高低半部分交换
   候选4：new_block 高低半部分交换后取反
2. 对每种候选，结合 stale_block 计算 WD-prone cells 数量
3. 选 WD-prone cells 最少的那个候选作为 encoded_block
4. 记录用了哪种编码方式（需要2bits元数据）

读取时根据元数据做逆变换还原原始数据
```

**DIN**

```
输入：new_block（512bits）
输出：encoded_block

步骤：
1. 用 FPC（Frequent Pattern Compression）压缩 new_block
   FPC 的pattern表（按论文引用的原始FPC论文）：
   - 000：全零串
   - 001：4位符号扩展
   - 010：单字节符号扩展
   - 011：半字零扩展
   - 100：半字符号扩展
   - 101：两个连续半字相同
   - 110：单字重复4次
   - 111：不可压缩（原始数据）
   
2. 压缩后的数据如果比原始短，用'0'填充到固定长度
3. 添加20bits的BCH纠错码保护编码后的数据
4. 如果压缩后反而更长，退化为DCW

注意：DIN 是这里最复杂的，FPC 需要单独完整实现
```

---

## 四、LearnWD 四个组件

**① Model Trainer**

```
输入：stale_blocks（N个512bits的block）
输出：cluster_ids（每个block属于哪个簇），centroids（k个质心）
参数：k=16（默认）

步骤：
1. 对每个 stale_block 计算 disturbance vector：
   对位置 i（0到511）：
       如果 block[i] = 0：d[i] = 0
       如果 block[i] = 1：
           left = 1 if (i>0 and block[i-1]=0) else 0
           right = 1 if (i<511 and block[i+1]=0) else 0
           d[i] = left + right
   disturbance_vector = d（512维，每维取值0/1/2）

2. 对所有 disturbance vectors 做 k-means 聚类：
   使用标准 k-means，欧氏距离
   迭代直到收敛
   输出每个block的cluster_id和k个质心

3. 同时计算每个block的MinHash指纹：
   用h=8个哈希函数
   每个哈希函数对block做随机排列
   找排列后第一个'1'的位置
   只记录位置的低3bits（节省空间）
   得到 8×3=24bits 的指纹

4. 把结果写入 Stale Table
```

**② Aggressor Extractor**

```
输入：new_block（512bits）
输出：aggressor_vector（512维，每维0或1）

对位置 i（0到511）：
    如果 new_block[i] = 1：a[i] = 0（不会做RESET）
    如果 new_block[i] = 0：
        left_zero = (i>0 and new_block[i-1]=0)
        right_zero = (i<511 and new_block[i+1]=0)
        如果 left_zero 或 right_zero：a[i] = 1
        否则：a[i] = 0
```

**③ Cluster Selector**

```
输入：aggressor_vector（512维），centroids（k个512维质心）
输出：best_cluster_id

对每个簇 i（0到k-1）：
    P[i] = dot_product(aggressor_vector, centroids[i])

best_cluster_id = argmin(P)
```

**④ Similarity Estimator**

```
输入：new_block（512bits），best_cluster_id，Stale Table
输出：best_stale_address

步骤：
1. 计算 new_block 的 MinHash 指纹：
   用同样的 h=8 个哈希函数
   得到 8 个hash值

2. 从 Stale Table 里取出 best_cluster_id 对应的所有 entry

3. 对每个 entry，计算相似度：
   similarity = 8个hash值中相同的个数 / 8
   
4. 选 similarity 最高的 entry 对应的 physical_address

5. 如果该簇为空（极端情况），
   退化为从全部 stale blocks 里随机选一个
```

---

## 五、ECC 模拟（Exp#6用）

```
ECC-i 表示能纠正 i 个bit错误

模拟逻辑：
    发生 actual_wd_errors 个错误后：
    如果 actual_wd_errors <= i：
        ECC 直接纠正，不需要 VnR
        vnr_count = 0
    如果 actual_wd_errors > i：
        剩余 actual_wd_errors - i 个错误需要 VnR 处理
        vnr_count += 1
        重复写操作直到错误数 <= i 或达到上限
```

---

## 六、仿真主循环

```
对每个数据集：
    加载数据，打包成blocks
    前50000个block初始化Stale Table
    训练模型（Model Trainer）
    
    对后50000个新写请求（按泊松时序）：
        取出 new_block
        
        根据编码方案对 new_block 编码（DCW/DMPart/MinWD/DIN）
        
        根据选择策略选出目标 stale_address：
            RandSel：随机选
            LearnWD：走四个组件流程
        
        取出对应的 stale_block
        
        计算并记录：
            wd_prone_count
            actual_wd_errors（概率模拟）
            write_latency
            write_energy
            write_cost
            vnr_count
        
        用 new_block 覆写 stale_block（更新内存数组）
        更新 Stale Table（删除旧entry，旧地址变为新stale）
        
        overwrite_counter += 1
        如果 overwrite_counter >= 20000：
            触发重训练
    
    汇总统计该数据集的所有指标
    和基线（RandSel+DCW）做归一化对比
```

---

## 七、需要输出的指标

```
对应论文各个实验：

Exp#1：每次写操作的平均 WD 错误数
Exp#2：write cost（归一化到 RandSel+DCW）
Exp#3：write energy（归一化到 RandSel+DCW）
Exp#4：write latency（归一化到 RandSel+DCW）
Exp#5：MLC PCM 下的 WD 错误数
Exp#6：VnR 操作次数（归一化到 RandSel+ECC-0）
Exp#7：不同 k 值下的 WD 错误数和训练延迟
Exp#8：不同 hash 函数数量下的 WD 错误数
Exp#9：不同重训练频率下的 WD 错误数
Exp#10：不同 block size 下的 WD 错误数
Exp#11：不同聚类算法下的 WD 错误数
Exp#12：数据集切换场景下的 WD 错误数
```

---

## 八、实现顺序建议

```
第一步：实现基础框架
    block打包、WD-prone计算、概率模拟、指标统计

第二步：实现 RandSel + DCW
    跑通整个仿真循环，输出 Exp#1 的结果

第三步：实现 LearnWD 四个组件
    加入 LearnWD，对比 RandSel+DCW vs LearnWD+DCW
    对照论文 Figure 13 验证数字是否接近

第四步：实现 DMPart 和 MinWD
    补全另外两种编码方案

第五步：实现 DIN
    最后实现最复杂的 FPC 压缩

第六步：补全所有实验
    MLC、ECC、参数敏感性分析、全部数据集
```


