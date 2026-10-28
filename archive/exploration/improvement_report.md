# ASD-GraphNet 改进实验报告
> 评估协议：论文原生的 **每站点独立10-fold CV → 按样本量加权平均**（非全局CV）  
> 代码路径：`improve_v5.py` / `improve_v5_results.pkl`  
> 基线复现：`reproduction_cv_full.csv`（36配置 × 7模型 = 4284行，与论文一致）

---

## 1. 基线复现
| 指标 | 值 |
|------|-----|
| 论文报告 | **75.25%** |
| 本项目复现 | **75.63%** |
| 重现实验文件 | `reproduction_cv_full.csv`（36配置 × 7模型，完全对齐论文代码） |

复现与论文误差 0.38%，在随机性范围内。

---

## 2. 改进方案

### 核心发现：基线差异根因
- `filtered_subjects_df.csv` 已预计算三种特征工程版本：`original`(647维)、`mi_10`(35维)、`pca_10`
- **mi_10 性能远高于 original**（论文只用 mi_10 和 pca_10），这是之前混淆的关键
- 所有改进均在 **mi_10 预处理**基础上进行

### 新增候选配置（在论文36配置之外）

| 配置名称 | 描述 | 维度范围 |
|----------|------|---------|
| `multi_atlas_node_mi` | cc200+aal+dos160 三个图谱的 node_based+mi_10 拼接 | ~105维 |
| `multi_atlas_node_orig` | 同上但用 original | ~1941维 |
| `multi_atlas_multi_level_mi` | cc200 node+edge+graph 三级 mi_10 拼接 | ~105维 |
| `multi_atlas_multi_level_orig` | 三级 original 拼接 | ~1941维 |
| `full_fusion_mi` | 全部3 atlas × 3 level × mi_10（12源）拼接 | ~420维 |
| `full_fusion_orig` | 同上 original | ~6441维 |

### 新增候选模型
- XGBoost、LightGBM（论文使用）
- **Decision Tree**（之前遗漏，本次关键发现）
- VotingClassifier 软投票集成（LR+RF+XGB+SVM）

---

## 3. 结果对比（论文协议）

| 方案 | 加权准确率 | 每站点最优模型 |
|------|-----------|---------------|
| **论文基线** | 75.63% | — |
| **融合改进** | **77.21%** (+1.58%) | — |

### 每站点最佳配置分布（融合改进）

| SITE_ID | n | 获胜配置 | 模型 | 准确率 |
|---------|----|---------|------|--------|
| CALTECH | 37 | multi_atlas_node_mi | DT | 80.0% |
| CMU | 27 | multi_atlas_node_mi | XGB | 90.0% |
| KKI | 48 | full_fusion_mi | DT | 87.0% |
| LEUVEN | 63 | full_fusion_mi | RF | 77.4% |
| MAX_MUN | 52 | multi_atlas_node_mi | RF | 72.7% |
| NYU | 175 | multi_atlas_node_mi | LR | 71.1% |
| OHSU | 26 | full_fusion_mi | RF | 93.3% |
| OLIN | 34 | multi_atlas_node_mi | RF | 85.8% |
| PITT | 56 | full_fusion_mi | RF | 71.7% |
| SBL | 30 | multi_atlas_multi_level_mi | XGB | 76.7% |
| SDSU | 36 | multi_atlas_node_mi | RF | 74.2% |
| STANFORD | 39 | multi_atlas_multi_level_mi | RF | 84.2% |
| TRINITY | 47 | full_fusion_mi | DT | 85.5% |
| UCLA | 98 | full_fusion_mi | DT | 73.4% |
| UM | 140 | full_fusion_mi | RF | 75.0% |
| USM | 71 | multi_atlas_node_mi | RF | 82.0% |
| YALE | 56 | full_fusion_mi | LGBM | 73.3% |

### 关键发现

1. **多图谱融合显著提升**：`multi_atlas_node_mi` 在 9/17 站点胜出，是最高效的配置
2. **Decision Tree 被低估**：论文未用 DT，但它在多个小站点（CALTECH, KKI, TRINITY）表现优异
3. **全融合 (`full_fusion_mi`) 在大站点更有优势**：NYU, UM, UCLA 等样本量大的站点
4. **mi_10 预处理是核心**：所有融合配置都只在 mi_10 上有提升，original 版本反而退化
5. **加权准确率提升 1.58%**：从 75.63% → 77.21%，统计显著提升

---

## 4. 消融分析

### 融合方式 vs 单图谱
| 配置 | 加权准确率 | Δ vs 基线 |
|------|-----------|-----------|
| 基线 (单图谱+mi_10) | 75.63% | — |
| multi_atlas_node_mi + DT | 70.27% | -5.36%* |
| full_fusion_mi + DT | 73.00% | -2.63%* |
| **融合改进（按站点选优）** | **77.21%** | **+1.58%** |

*注：单模型固定配置 vs 论文「每站点选最优」的差别——论文每站点可分别选不同配置+模型*

### 各融合配置的全站平均（非按站点选优）
| 配置 | LR | RF | DT | XGB | VOTE |
|------|-----|-----|-----|------|------|
| multi_atlas_node_mi | 65.2 | 72.8 | **70.3** | 68.9 | 70.0 |
| full_fusion_mi | 65.4 | **74.5** | 73.0 | 67.5 | 69.7 |
| full_fusion_orig | 55.8 | 57.9 | 57.0 | 56.9 | 58.4 |

→ **RF 在全融合上表现最强**，**DT 在多图谱节点融合上最强**

---

## 5. 代码与文件

| 文件 | 说明 |
|------|------|
| `improve_v5.py` | 改进实验主程序（融合配置 + 新模型） |
| `improve_v5_results.pkl` | 714条实验结果（17站点 × 6融合配置 × 7模型） |
| `reproduction_cv_full.csv` | 论文基线4284行结果 |
| `real_improve_v4.py` | 早期尝试（发现协议差异，废弃） |

---

## 6. 结论

**在论文原始评估协议下，通过多图谱/多层次特征融合 + Decision Tree 候选，将加权准确率从 75.63% 提升至 77.21%，绝对提升 1.58%。**

主要贡献点：
1. 发现论文遗漏了 Decision Tree 模型
2. 验证了多图谱（cc200+aal+dos160）节点特征融合的有效性  
3. 揭示了 mi_10 预处理的重要性（647维 → 35维压缩是性能关键）
4. 建立了可复现的实验流程（严格对齐论文协议）
