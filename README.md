# ASD-GraphNet-Audit: ABIDE 图特征分类的严格评估

<div align="center">

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-green.svg)
![Stars](https://img.shields.io/github/stars/Tin11Mn/ASD-GraphNet-Audit?style=social)
![Issues](https://img.shields.io/github/issues/Tin11Mn/ASD-GraphNet-Audit)

**🚨 重要发现**: 主流 ABIDE 分类方法的 **75%** 准确率存在严重数据泄露，真实性能上限约为 **58%**。

</div>

---

## 📊 核心发现

| 评估协议 | 加权准确率 | 说明 |
|---------|-----------|------|
| 🔴 **论文原协议 (有泄露)** | **75.63%** | MI/PCA 预筛选导致虚假高性能 |
| 🟢 **严格嵌套 CV (无泄露)** | **58.12%** | 多图谱融合 (真实上限) |
| ⚪ **严格嵌套 CV (单图谱)** | **50.31%** | 接近随机猜测 |
| 🔍 **泄露虚增量** | **+17.51pp** | MI (+6.37pp) + PCA (+1.98pp) + 选优自由度 (+8.8pp) |

> 💡 **关键结论**: 在严格无泄露的嵌套 CV 下，多图谱融合仍能带来 **+7.81pp** 的显著提升 (p<0.001)，但整体性能远低于文献报道。

---

## 📖 目录

- [背景与动机](#背景与动机)
- [快速开始](#快速开始)
- [使用方法](#使用方法)
- [实验结果](#实验结果)
- [方法细节](#方法细节)
- [引用与贡献](#引用与贡献)

---

## 🧠 背景与动机

自闭症谱系障碍 (ASD) 的自动诊断是神经影像学界的重要挑战。基于 ABIDE 数据集的图特征机器学习方法常报告 **70-80%** 的分类准确率，被视为该领域的基准。

然而，这些高性能结果往往依赖于复杂的特征工程流水线，其中**特征选择步骤极易发生数据泄露**：
- **特征选择泄露**: 在划分训练/测试集之前，使用全数据集的标签筛选特征
- **预拟合泄露**: 在全数据集上拟合 PCA 或标准化参数
- **后果**: 测试集"提前看到答案"，导致准确率虚高，误导后续研究

本研究对当前主流的 ABIDE 图特征分类流水线进行了系统性审计，揭示了真实性能上限，并提供了首个无泄露的严格基准。

---

## 🚀 快速开始

### 1️⃣ 安装依赖

```bash
git clone https://github.com/Tin11Mn/ASD-GraphNet-Audit.git
cd ASD-GraphNet-Audit
pip install -r requirements.txt
```

### 2️⃣ 准备数据

下载 ABIDE I 数据集并放置到 `data/raw/` 目录：
- [ABIDE Preprocessed Data](https://fcon_1000.projects.nitrc.org/indi/retro/Abide_Pools.html)
- 参考 [`docs/data_preparation.md`](docs/data_preparation.md) 获取详细指南

### 3️⃣ 运行实验

```bash
# 复现论文基线 (有泄露协议)
python scripts/reproduce.py --output results/leaky/

# 运行严格评估 (无泄露嵌套 CV)
python scripts/nested_cv.py --output results/clean/

# 泄露归因分析
python scripts/leak_attribution.py --compare leaky/ clean/

# 统计分析与可视化
python scripts/analyze_results.py results/leaky/ results/clean/
python scripts/visualize.py --output figures/
```

### 4️⃣ 查看结果

- `results/clean/predictions.csv`: 逐被试预测 (1035 条)
- `results/clean/site_summary.csv`: 17 站点汇总
- `figures/leakage_attribution.png`: 泄露归因柱状图
- `figures/performance_comparison.png`: 性能对比热力图

---

## 🛠️ 使用方法

### 复现论文基线

```bash
python scripts/reproduce.py \
  --atlas cc200,aal,dos160 \
  --features node,edge,graph \
  --models LR,RF,XGB,LGBM,MLP,DT,KNN,SVM \
  --cv-fold 10 \
  --seed 42 \
  --output results/leaky/
```

### 运行严格嵌套 CV

```bash
python scripts/nested_cv.py \
  --outer-folds 10 \
  --inner-folds 3 \
  --atlas cc200,aal,dos160 \
  --feature-engineering original,mi_10 \
  --fusion-method fixed-weight \
  --threshold-learning oof \
  --output results/clean/
```

### 泄露归因分析

```bash
python scripts/leak_attribution.py \
  --protocols leaky_mi leaky_pca clean_orig clean_mi clean_pca \
  --sites all \
  --output results/attribution/
```

---

## 📈 实验结果

### 泄露归因分析

| 泄露来源 | 虚增准确率 | 统计显著性 | 受影响站点 |
|---------|-----------|------------|-----------|
| **MI 特征选择** | **+6.37%** | p < 0.001 | OHSU (+26.7%), CMU (+16.7%) |
| **PCA 预拟合** | **+1.98%** | p < 0.001 | STANFORD (+15.0%), TRINITY (+12.5%) |
| **配置选优自由度** | **+8.8pp** | - | 所有站点 |
| **合计** | **+17.51pp** | - | - |

### 多图谱融合策略对比

| 策略 | 加权准确率 | vs 单图谱 | 说明 |
|-----|-----------|----------|------|
| **负对数似然优化** | **58.12%** | **+7.81pp** | 最优权重学习方法 |
| 置信度加权 | 57.86% | +7.55pp | 接近最优 |
| 门控网络 | 55.90% | +5.59pp | 小样本过拟合 |
| Ridge Stacking | 49.45% | -0.86pp | 过拟合严重 |
| 等权融合 | 48.80% | -1.51pp | 退化 |
| 多数投票 | 48.82% | -1.49pp | 退化 |
| **最优单图谱** | **50.31%** | - | 基准 |

### 站点性能差异

| 站点 | 样本量 | 泄露前 | 泄露后 | 虚增量 |
|-----|-------|--------|--------|--------|
| CALTECH | 37 | 51.67% | 59.17% | +7.50pp |
| CMU | 27 | 51.67% | 70.00% | +18.33pp |
| NYU | 175 | 42.81% | 60.03% | +17.22pp |
| UCLA | 98 | 54.00% | 60.11% | +6.11pp |
| UM | 140 | 47.14% | 60.00% | +12.86pp |

---

## 🔬 方法细节

### 嵌套交叉验证框架

```
外层: 每站点 10-fold StratifiedKFold(seed=42)
      └─ 测试折从不参与任何拟合, 是唯一的报告来源
内层:  3-5 fold, 只选模型, 得到训练折 OOF 概率
融合:  权重 / 门控网络 / 决策阈值 全部由训练折 OOF 学习
```

### 泄露协议 vs 严格协议

| 步骤 | 泄露协议 | 严格协议 |
|-----|---------|---------|
| 特征选择 | 全数据集 MI 筛选 Top 10% | 仅训练折内 MI 筛选 |
| 降维 | 全数据集 PCA | 仅训练折内 PCA |
| 标准化 | 全数据集 Z-score | 仅训练折内 Z-score |
| 模型选择 | 全数据集调参 | 内层交叉验证 |
| 阈值学习 | 固定 0.5 | 训练折 OOF 学习 |

### 统计检验

- **McNemar 检验**: 比较融合方法与单图谱的显著性
- **Bootstrap CI**: 计算性能差异的 95% 置信区间
- **冲突子集分析**: 验证"冲突样本协同"效应

---

## 📚 引用与贡献

### 引用本文

如果本项目对您的研究有帮助，请使用以下格式引用：

```bibtex
@software{zhang2024asdgraphnetaudit,
  author = {Zhang, Sanbao},
  title = {ASD-GraphNet-Audit: Strict Evaluation of Graph-Based ASD Classification on ABIDE},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Tin11Mn/ASD-GraphNet-Audit},
  version = {1.0.0}
}
```

### 贡献指南

欢迎提交 Issue 或 Pull Request！请遵循以下步骤：

1. **Fork** 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-improvement`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-improvement`)
5. 创建 Pull Request

---

## ❓ 常见问题

### Q: 为什么论文报告的准确率这么高？
A: 论文代码在划分训练/测试集之前，就用全体被试的标签筛过特征了（MI 选 top 10%）。所以 75.63% 里有一部分是"提前看过答案"的收益。

### Q: 真实性能上限是多少？
A: 在严格无泄露的嵌套 CV 下，多图谱融合的上限约为 **58%**，单图谱约为 **50%**。

### Q: 能否进一步提升性能？
A: 目前基于传统 ML + 手工特征的极限已接近。突破需要引入：
- 图神经网络 (GNN)
- 跨模态数据融合
- 大规模预训练

### Q: 如何防止类似泄露？
A: 遵循以下原则：
1. **强制嵌套 CV**: 所有特征工程必须在训练折内完成
2. **开源逐被试预测**: 便于独立复核
3. **报告泄露风险**: 在方法部分明确声明预处理步骤

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

## 🙏 致谢

- **ABIDE Consortium**: 提供公开的自闭症脑影像数据库
- **ASD-GraphNet 原作者**: 开源的代码和预处理流水线
- **社区贡献者**: 帮助完善代码和测试

---

<div align="center">

**最后更新**: 2024-10-28  
**版本**: v1.0.0  
**维护者**: [@Tin11Mn](https://github.com/Tin11Mn)

⭐ 如果这个项目对你有帮助，请给个 Star!

</div>
