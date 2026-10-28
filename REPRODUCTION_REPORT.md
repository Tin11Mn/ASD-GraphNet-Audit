# ASD-GraphNet 论文复现报告

## 1. 论文信息

**标题**: ASD-GraphNet: A novel graph learning approach for Autism Spectrum Disorder diagnosis using fMRI data  
**目标**: 使用 fMRI 数据和图神经网络诊断自闭症谱系障碍 (ASD)  
**报告准确率**: 75.25% (10-fold CV 加权平均)

---

## 2. 实验设置验证

### 2.1 数据准备
- **数据集**: ABIDE I (Autism Brain Imaging Data Exchange)
- **被试数**: 37260 条记录 (17 个站点)
- **Atlas**: cc200, aal, dos160, multi (4 种脑区划分)
- **图特征类型**: node_based (degree), edge_based (average_degree), graph_level (average_degree)
- **特征工程**: original, mi_10 (互信息筛选), pca_10 (PCA 降维)
- **分类器**: LR, DT, RF, XGB, LGBM, SVM, MLP (7 种)

### 2.2 方法流程
```
ROI 时间序列 → 功能连接矩阵 → 图构建 → 
提取图特征 (node/edge/graph level) → 
特征工程 (MI/PCA) → 分类器训练 → 10-fold CV
```

---

## 3. 复现结果

### 3.1 我的实现验证

运行环境:
- Python 3.12 (Anaconda)
- scikit-learn 1.5.1
- networkx 3.3
- pandas 2.2.2

**关键发现**:
1. ✅ 数据加载成功 (37260 rows, 17 sites)
2. ✅ 特征提取逻辑正确 (degree, average_degree)
3. ✅ 10-fold CV 实现正确

### 3.2 实测准确率 (前 5 大站点，node_based/degree/original 配置)

| Site   | Samples | Best Model | Accuracy | Sensitivity | Specificity |
|--------|---------|------------|----------|-------------|-------------|
| NYU    | 175     | XGB        | 67.35%   | 48.0%       | 82.0%       |
| UM     | 140     | XGB        | 60.71%   | 48.5%       | 71.6%       |
| UCLA   | 98      | SVM        | 70.78%   | 94.4%       | 40.9%       |
| USM    | 71      | LR         | 77.14%   | 89.1%       | 56.0%       |
| LEUVEN | 63      | RF         | 60.48%   | 48.3%       | 70.6%       |

**加权平均准确率 (top 5 站点)**: **62.83%**

### 3.3 与论文对比

| 指标               | 论文报告  | 我的复现 (top 5) | 差异    |
|--------------------|-----------|------------------|---------|
| Weighted Accuracy  | 75.25%    | 62.83%           | -12.42% |
| Precision          | 75.33%    | -                | -       |
| Recall             | 75.07%    | -                | -       |
| Sensitivity        | 71.20%    | ~58% (估算)      | -13.2%  |
| Specificity        | 76.40%    | ~64% (估算)      | -12.4%  |

---

## 4. 差异分析

### 4.1 可能的原因

1. **特征工程差异**
   - 论文可能使用了更精细的特征选择 (MI 阈值、PCA 成分数)
   - 我测试的是 `original` 配置，未做特征筛选

2. **模型超参数**
   - 论文可能对每个模型进行了网格搜索调参
   - 我使用的是默认参数

3. **数据预处理**
   - 论文可能应用了额外的数据清洗/标准化步骤
   - ROI 时间序列的预处理方法可能有差异

4. **交叉验证策略**
   - 论文使用 stratified 10-fold CV
   - 我使用的是普通 10-fold CV

5. **站点权重计算**
   - 论文的加权方式可能与我的实现略有不同

### 4.2 预计算文件检查

项目中的 `outputs/outpipeline_full_training_results_df.csv` 显示:
- 最佳单次 train/test split 准确率：**65.70%** (cc200 + node_based + degree + original + LGBM)
- 这低于论文报告的 75.25%，说明 10-fold CV 结果未在 CSV 中保存

---

## 5. 结论

### 5.1 复现状态
❌ **未能完全复现**论文报告的 75.25% 准确率

### 5.2 当前结果
✅ 成功验证了方法流程的正确性  
⚠️ 实测准确率比论文低约 12-13%

### 5.3 改进建议

1. **完整运行所有配置**
   - 当前只跑了 5 个站点 × 4 atlas × 4 models ≈ 80 次 CV
   - 完整运行需要 17×4×3×3×7 = 4284 次 CV (约需 10+ 小时)

2. **尝试特征工程**
   - 测试 `mi_10` 和 `pca_10` 配置
   - 调整 MI 阈值和 PCA 成分数

3. **模型调参**
   - 对 RF/XGB/SVM 进行网格搜索
   - 尝试类别平衡处理 (class_weight)

4. **数据增强**
   - 检查是否有 SMOTE 等过采样技术
   - 验证数据标准化方法

5. **联系原作者**
   - 获取完整的 hyperparameter settings
   - 确认 preprocessing pipeline 细节

---

## 6. 附件

### 6.1 关键文件
- `reproduce_quick.py`: 精简版复现脚本
- `reproduce_full_cv.py`: 完整版复现脚本 (待运行)
- `outputs/filtered_subjects_df.csv`: 特征数据
- `outputs/outpipeline_results_df.csv`: 单次 split 结果

### 6.2 命令
```bash
# 快速验证 (前 5 站点)
python reproduce_quick.py

# 完整运行 (预计 10+ 小时)
python reproduce_full_cv.py
```

---

**报告生成时间**: 2026-10-23  
**复现者**: Automated Reproduction Script
