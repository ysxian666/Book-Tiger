# 面向 Amazon Reviews 2023 Books 的 TIGER 数据构建、Semantic ID 分配与用户条件化生成研究

## 摘要

TIGER 将推荐召回建模为下一商品 Semantic ID 的自回归生成，为大规模候选检索提供了不同于向量近邻搜索的实现路径。本实验以标准 TIGER 为研究起点，在 Amazon Reviews 2023 Books 上重新构建数据与训练协议，并围绕 RQ-VAE 码本健康、行为感知 Semantic ID 全局分配、用户条件化生成及生成器联合优化展开研究。数据部分采用完整评论流处理、全局 5-core 过滤、按训练集频次确定 50K 商品目录以及按用户时间顺序的 leave-one-out 划分，最终得到 1,134,600 条交互、102,167 名用户和 49,765 个商品。模型部分包括三层 RQ-VAE、后缀 Semantic ID、Behavior-Cluster-Aware Global Semantic ID Assignment（BC-GSID）、User-Conditioned Generation（UCG）及排序感知联合微调。

本实验为 seed 42 的单种子实验，使用 NVIDIA GeForce RTX 5090，在统一数据、SID 长度、模型规模和解码预算下比较 O1-O5 五种设置，并通过 C1-C4 控制实验分析行为信息和用户上下文的作用。结果显示，O4 的 Recall@20 相对 O1 提高 58.3%，但其语义漂移也显著增大；C2 hard-semantic BC-GSID 在满足严格语义预算的条件下取得较高 Recall@20，是当前较可辩护的方向性结果。用户条件化生成与联合微调尚未表现出稳定优势，因此本文以可复现实验档案和研究边界说明为主，不将单次结果解释为最终结论。

**关键词：** 生成式推荐；TIGER；Semantic ID；RQ-VAE；行为感知分配；用户条件化生成；序列推荐

## 1. 研究背景

### 1.1 TIGER 的生成式检索思路

传统推荐召回通常在用户和商品共享的向量空间中计算相似度，再借助近似最近邻方法返回候选。TIGER 将召回任务改写为生成任务：首先为每个商品构造由多个离散码字组成的 Semantic ID，然后训练 Transformer 序列到序列模型，根据用户历史 Semantic ID 自动回归生成下一商品的 Semantic ID。该设计的核心价值在于利用离散语义结构表达商品层级，并使生成模型直接输出候选标识，从而避免对海量商品向量执行显式近邻搜索。

在标准 TIGER 中，商品内容先通过文本编码器得到稠密表示，再由 RQ-VAE 逐层量化得到多个 Semantic ID 码字。生成器以用户历史中的码字序列为输入，以目标商品 Semantic ID 为输出。由于 Semantic ID 共享前缀在一定程度上对应语义相近商品，生成式解码可以利用词表约束、前缀树和候选映射完成检索。

### 1.2 标准 TIGER 的潜在问题

标准 TIGER 的有效性依赖 Semantic ID 同时具备较高的内容保真度、良好的码本使用率和稳定的层级结构。然而，内容编码得到的主要是商品自身语义，未必能够反映不同用户群体在实际交互中的共同偏好。当两个商品内容相近但消费人群明显不同时，仅依赖内容量化可能无法形成最适合生成模型学习的标识结构。

其次，离散码本可能出现部分码字长期不被使用的问题。低利用率会降低每个码字的信息承载能力，并可能造成重建误差集中。若只依赖常规量化损失，稀疏码字获得的梯度有限，固定长度 SID 的表示空间也难以被充分利用。因此，需要在量化训练中引入使用率约束、熵约束和显式的码字恢复机制。

再次，多个商品可能共享相同的基础码字组合。常见处理方式是在基础 Semantic ID 后附加后缀，以区分不同商品。后缀能够保证候选唯一性，但它不携带商品语义或协同行为信息，且后缀长度可能随碰撞数量增加。对于大规模商品目录，如何在不增加基础 SID 长度的前提下，将碰撞消解、语义保真和行为一致性纳入同一分配过程，是一个值得研究的问题。

最后，TIGER 的 SID 构建和序列生成通常分阶段完成。生成器虽然能够从用户历史学习偏好，但行为中的长期稳定兴趣和近期短期意图被压缩在同一输入序列中。若能够在不改变基础 SID 长度的条件下，将用户长短期上下文编码为额外的软前缀，并进一步研究 SID 分配与生成器的联合优化，可能为生成式检索提供新的信息注入方式。

### 1.3 数据场景差异

原 TIGER 的主要实验基于 Amazon Product Reviews 数据中的 Beauty、Sports and Outdoors 和 Toys and Games 三个类目，这些数据覆盖时间约为 1996 年至 2014 年。不同类目在商品规模、交互密度、时间跨度和内容形式上存在明显差异。

本实验使用 Amazon Reviews 2023 Books。该数据包含更新的评论文本、metadata 和时间信息，Books 类目的商品语义更加丰富，长尾结构也更加明显。相较于原 TIGER 数据，本实验的数据规模和领域均发生变化，因此研究重点不是简单复现原论文中的数值，而是检验在更新、更大且领域不同的数据上，Semantic ID 分配与用户条件化生成是否仍具有可观察的增益。

### 1.4 本实验的优化方向

围绕上述问题，本实验依次研究以下方向：

1. 通过 hard usage、码本熵和 error-aware dead-code revival 改善 RQ-VAE 的码字使用情况。
2. 在保持基础 SID 长度不变的前提下，构造行为簇感知的全局 SID 分配 BC-GSID。
3. 使用长窗口和短窗口行为表示，通过 4 个 soft prefix token 将动态用户上下文注入 T5 编码器。
4. 在 BC-GSID 与 UCG 的基础上进行排序感知联合微调，并冻结码本以保持 SID 映射稳定。
5. 通过行为打乱、严格语义约束、上下文打乱和组合约束控制，分析各模块的有效来源。

## 2. 研究目的

### 2.1 总体目的

本实验的总体目的是：在统一的数据划分、SID 长度、码本容量、生成器规模和解码预算下，研究行为感知 Semantic ID 分配与用户条件化生成能否改善 TIGER 的下一商品检索，并分析各模块之间是否存在互补关系。

### 2.2 研究问题

本实验关注三个主要问题。

**Q1：行为簇感知的全局 SID 分配能否改善检索效果？**  
该问题检验在基础码字之外引入训练集行为画像后，是否能够在不增加基础 SID 长度的条件下，提高 Recall、NDCG 和候选覆盖率，同时控制语义漂移和码本利用率。

**Q2：长短期用户上下文能否为标准生成器提供额外信息？**  
该问题比较仅使用历史 SID、时间桶和哈希用户标记的生成器，与额外注入长期偏好和短期意图 soft prefix 的生成器，并分别分析整体用户和短历史用户的表现。

**Q3：SID 分配、用户条件化生成和联合微调是否互补？**  
该问题通过 O2、O3、O4 和 O5 比较单模块、组合模块和联合训练，并通过 C1-C4 控制实验识别增益是否来自真实行为信息、严格语义约束或上下文信息。

### 2.3 数据集差异与实验定位

原 TIGER 使用 1996-2014 年的 Amazon Beauty、Sports and Outdoors、Toys and Games 数据，本实验使用 Amazon Reviews 2023 Books。数据源、时间范围和商品领域均不同。本实验因此不直接沿用原论文的绝对指标判断优劣，而是在相同数据处理和训练协议下进行内部比较。

本实验采用单种子实验进行初步验证，主要观察不同模块的方向性变化、约束是否满足以及系统指标是否退化。结果用于确定后续统计复验的优先级，而不替代多随机种子和区间估计。

## 3. 数据说明

### 3.1 数据来源

实验使用 McAuley Lab 发布的 Amazon Reviews 2023 Books 评论文件和 metadata 文件。两个文件均以 JSON Lines 压缩格式存储，并按流式方式逐行读取。

| 数据文件 | 官方地址 | 文件大小 |
|---|---|---:|
| Books 评论 | `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Books.jsonl.gz` | 6,216,820,644 字节 |
| Books metadata | `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Books.jsonl.gz` | 4,942,125,770 字节 |

评论文件提供用户、商品、评分和时间等交互字段，metadata 文件提供标题、描述、类别和品牌等商品内容。数据说明中的文件大小与官方内容长度一致，处理过程不依赖固定条数的局部截取。

### 3.2 数据筛选与划分

评论流首先执行全局 `user>=5` 和 `item>=5` 的 5-core 过滤。过滤后得到 776,419 名满足条件的用户和 9,488,607 条交互。为了控制后续特征构建、量化训练和生成器训练的资源需求，实验从 5-core 用户池中按固定随机种子抽取 240,000 名用户。抽样后继续执行商品频次筛选，并仅依据训练集交互确定商品目录。

商品目录首先按训练集交互频次选择 top-50K，再对训练集执行第二轮用户和商品过滤。最终数据包含 1,134,600 条交互、102,167 名用户和 49,765 个有效目录商品。每个用户按时间顺序执行 leave-one-out 划分：最后一条交互作为 test，倒数第二条作为 valid，之前所有交互作为 train。

| 划分 | 用户数 | 交互数 | 构造规则 |
|---|---:|---:|---|
| Train | 102,167 | 930,266 | 每个用户除最后两条之外的交互 |
| Valid | 102,167 | 102,167 | 每个用户倒数第二条交互 |
| Test | 102,167 | 102,167 | 每个用户最后一条交互 |
| 合计 | 102,167 | 1,134,600 | 用户级时间顺序 leave-one-out |

固定时间顺序能够保留用户兴趣演化关系，并使训练、验证和测试样本具有清晰的时间边界。商品目录由训练集频次决定，valid 和 test 仅用于验证与最终评价，不参与目录排名。

### 3.3 商品特征与行为画像

文本特征使用 `hyp1231/blair-roberta-base` 编码商品内容，最大输入长度为 256，输出 768 维向量并执行归一化。协同特征使用训练集用户-商品二值交互矩阵的截断 SVD，保留 128 个主成分，进一步形成 128 维商品协同表示。行为画像同样仅基于训练集，通过用户行为表示聚类得到 32 个用户簇，并将商品在训练交互中的用户簇分布归一化为行为画像。

| 特征 | 方法 | 维度或数量 | 数据来源 |
|---|---|---:|---|
| 文本表示 | BLaIR RoBERTa 编码与归一化 | 768 | 商品内容与 metadata |
| 协同表示 | 训练集二值交互矩阵截断 SVD | 128 | Train |
| 行为画像 | 用户行为聚类后的经验簇分布 | 32 类 | Train |
| 商品目录 | 按训练集频次选择 top-50K 后二轮过滤 | 49,765 | Train |
| Metadata 覆盖 | 目录商品 metadata 审计 | 1.0 | Catalog |

最终目录的内容覆盖率和 metadata 覆盖率均为 1.0，说明进入特征编码和 SID 分配的商品均具有可用的文本与属性信息。RQ-VAE 的 k-means 初始化、量化训练、行为聚类和 SID 分配均使用训练集或商品内容，验证集和测试集不进入这些拟合或分配过程。

### 3.4 测试用户与冷目标

验证和测试阶段按历史长度分桶抽取用户，以降低测试集合过度偏向长历史用户的风险。实际评价得到 2,493 名具有可检索 train-positive Semantic ID 的有效用户。测试集中另有 170 个目标商品缺少训练期正反馈对应关系，仅用于冷目标诊断，不纳入生成式检索的主指标计算。

这一处理可以区分“模型在已知目录内是否能够生成正确 SID”和“目录缺少训练正反馈时能否完成识别”两类问题。主表中的 Recall、NDCG 和 MRR 均以具有可检索目标的用户为分母。

### 3.5 运行环境与耗时

实验在单张 NVIDIA GeForce RTX 5090 上运行，显存为 31.36 GB。软件环境使用 CUDA 12.8、PyTorch 2.8.0+cu128 和 BF16 混合精度。下表汇总可从训练与评价文件直接核验的 GPU 阶段耗时。

| 阶段 | 可核验耗时 |
|---|---:|
| RQ-VAE 训练 | 约0.8小时 |
| O1-O5 生成器训练 | 约4.6小时 |
| O1-O5 标准解码评价 | 约0.4小时 |
| C1-C4 生成器训练 | 约4.0小时 |
| C1-C4 标准解码评价 | 约0.3小时 |
| 合计 | 约10.1小时 |

生成器训练耗时包含每个设置内部的验证损失计算和快速 Recall 估计，评价耗时包含标准 beam、前缀树约束解码和 500 用户穷举排序。上述合计不包含尚未单独记录耗时的原始数据流式预处理和前向特征提取。

### 3.6 运行说明

本实验推荐在 Linux 与 CUDA 环境中运行，Python 版本要求 3.10 及以上。为适配 RTX 5090 的 `sm_120` 计算能力，应先安装支持该架构的 PyTorch CUDA 版本，再安装项目依赖。示例安装流程如下：

```bash
python -m pip install --upgrade pip
python -m pip install "torch>=2.8" --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python -m pip install pytest
```

项目默认从 `data/raw` 读取官方评论和 metadata 文件。若本地已有完整文件，脚本会复用或建立硬链接；若本地不存在，脚本会从官方地址下载。数据文件合计约 10.4 GB，环境检查默认还要求可用磁盘空间不少于 25 GB。

```bash
python scripts/link_or_download_data.py --root . --strict-size
```

环境检查需要 CUDA、BF16、RTX 5090 对应的 `sm_120` 支持、两个原始数据文件以及足够的剩余磁盘空间。配置路径可通过通配符自动定位，避免手动选择：

```bash
CONFIG="$(find configs -maxdepth 1 -name '*_5090.json' -print -quit)"
python scripts/check_env.py --config "$CONFIG"
```

完整流程由若干可独立执行的阶段组成。每个阶段会复用已经存在且格式正确的产物；需要重新计算时可在命令中增加 `--force`。

| 阶段 | 主要工作 | 主要产物 |
|---|---|---|
| `preprocess` | 流式读取评论与 metadata，执行 5-core、用户抽样、商品目录和时间划分 | 交互表、商品目录、训练序列和数据统计 |
| `features` | 构建文本特征、协同特征和行为画像 | 文本矩阵、协同矩阵和行为画像 |
| `rqvae` | 训练多视图 RQ-VAE 并记录码本使用情况 | RQ-VAE 检查点与训练记录 |
| `sid` | 构造后缀 SID 和 BC-GSID | 基础 SID、行为感知 SID 及分配报告 |
| `generators` | 训练 O1-O5 生成器 | 各设置检查点、验证记录和训练指标 |
| `evaluate` | 执行标准 beam、Trie beam、穷举排序和指标汇总 | 各设置评价文件、比较表和结果报告 |

分阶段执行时，可先定位实验入口，再将 `--stage` 替换为表中所需阶段：

```bash
export PYTHONPATH="$PWD/code"
CONFIG="$(find configs -maxdepth 1 -name '*_5090.json' -print -quit)"
RUNNER="$(find scripts -maxdepth 1 -name 'run_*.py' ! -name 'run_controls.py' -print -quit)"
python "$RUNNER" --config "$CONFIG" --stage preprocess
```

完整主实验脚本会自动完成数据检查、环境检查和全部阶段，适合在 `tmux` 或持久化日志环境中运行：

```bash
nohup bash "$(find runners -maxdepth 1 -name 'run_*5090.sh' -print -quit)" > run.log 2>&1 &
tail -f run.log
```

主实验完成后，可单独运行 C1-C4 控制实验：

```bash
bash runners/run_controls_5090.sh
```

所有生成数据和结果均写入主配置中的 `paths.data_dir` 与 `paths.run_dir`。这两个目录、原始数据文件、缓存和模型检查点均由 `.gitignore` 排除，因此从 GitHub 克隆后需要先准备数据，再按阶段重新生成。提交代码前可运行以下测试：

```bash
python -m pytest -q
```

测试覆盖配置加载、数据预处理辅助逻辑、RQ-VAE、Semantic ID、Trie 解码、生成器前向计算和控制实验辅助函数。若只需检查某个阶段而不继续运行，可在命令结尾增加 `--stage` 对应的阶段名，并通过日志确认是否产出预期文件。

## 4. 理论说明

本章按照数据进入模型的顺序，依次说明多视图融合、RQ-VAE、Semantic ID 构造、行为感知分配、用户上下文编码、联合训练、解码与评价指标。

### 4.1 文本与协同特征的融合

设商品 $i$ 的文本表示为 $t_i \in \mathbb{R}^{d_t}$，协同表示为 $c_i \in \mathbb{R}^{d_c}$。模型分别使用线性层、LayerNorm 和 GELU 将其映射到相同维度：

$$
u_i = \phi_t(t_i), \qquad v_i = \phi_c(c_i),
$$

其中 $u_i, v_i \in \mathbb{R}^{d}$。门控融合首先计算通道级权重：

$$
g_i = \sigma\left(W_2\,\mathrm{GELU}(W_1[u_i;v_i]+b_1)+b_2\right),
$$

其中 $[u_i;v_i]$ 表示拼接，$g_i \in (0,1)^d$。融合表示为：

$$
f_i = g_i \odot u_i + (1-g_i)\odot v_i.
$$

该形式允许模型针对每个商品和每个特征维度动态调整文本与协同信息的比例。若文本信息足以描述商品，门控可提高文本分支权重；若交互模式提供额外区分信息，门控可提高协同分支权重。

### 4.2 RQ-VAE 编码与残差量化

RQ-VAE 的编码器将融合特征映射到潜在空间：

$$
z_i = E(f_i),
$$

其中 $z_i \in \mathbb{R}^{d_z}$。设共有 $L$ 个量化层级，第 $l$ 层的码本为 $\mathcal{E}_l=\{e_{l,k}\}_{k=1}^{K}$。第一层使用原始潜在向量作为残差：

$$
r_{i,0}=z_i.
$$

在第 $l$ 层，最近码字及其索引为：

$$
c_{i,l}=\arg\min_{k\in\{1,\ldots,K\}}\|r_{i,l-1}-e_{l,k}\|_2^2.
$$

量化残差和下一层残差分别为：

$$
q_{i,l}=e_{l,c_{i,l}}, \qquad r_{i,l}=r_{i,l-1}-q_{i,l}.
$$

最终量化向量为：

$$
z_{i,q}=\sum_{l=1}^{L}q_{i,l}.
$$

为了使编码器能够接收量化误差的梯度，前向传递使用量化向量，反向传递使用直通估计：

$$
\tilde{z}_{i,q}=z_i+\mathrm{sg}(z_{i,q}-z_i),
$$

其中 $\mathrm{sg}(\cdot)$ 表示停止梯度。解码器以 $\tilde{z}_{i,q}$ 为输入重建融合特征：

$$
\hat{f}_i=D(\tilde{z}_{i,q}).
$$

### 4.3 RQ-VAE 训练目标

重建损失使用融合特征与重建特征之间的均方误差。对于商品频次为 $n_i$ 的样本，逆频次权重定义为：

$$
w_i=\mathrm{clip}\left(\left(\frac{\bar{n}}{\max(n_i,1)}\right)^p,\frac{1}{c},c\right),
$$

并对全部权重执行均值归一化。默认 $p=0.5,\ c=10$。加权重建损失为：

$$
\mathcal{L}_{\mathrm{rec}}=\frac{\sum_i w_i\|f_i-\hat{f}_i\|_2^2}{\sum_i w_i}.
$$

每层的 commitment 损失和码本损失分别为：

$$
\mathcal{L}_{\mathrm{commit}}=\frac{1}{L}\sum_{l=1}^{L}\left\|\mathrm{sg}(r_{i,l-1})-e_{l,c_{i,l}}\right\|_2^2,
$$

$$
\mathcal{L}_{\mathrm{codebook}}=\frac{1}{L}\sum_{l=1}^{L}\left\|r_{i,l-1}-\mathrm{sg}(e_{l,c_{i,l}})\right\|_2^2.
$$

设第 $l$ 层码字到当前残差的负距离为 $\ell_{i,l,k}=-\|r_{i,l-1}-e_{l,k}\|_2^2/\tau$，软分配概率为：

$$
\pi_{i,l,k}=\frac{\exp(\ell_{i,l,k})}{\sum_{j=1}^{K}\exp(\ell_{i,l,j})}.
$$

批内平均软使用概率记为 $\bar{\pi}_{l,k}$，均匀分布为 $\mathcal{U}_k=1/K$。软使用率损失为：

$$
\mathcal{L}_{\mathrm{usage}}=\frac{1}{L}\sum_{l=1}^{L}\frac{1}{K}\sum_{k=1}^{K}\left(\bar{\pi}_{l,k}-\mathcal{U}_k\right)^2.
$$

硬使用率使用最近码字的 one-hot 分配构造直通概率，并采用相同形式约束批内使用分布。码本熵为：

$$
H_l=-\sum_{k=1}^{K}\bar{\pi}_{l,k}\log(\bar{\pi}_{l,k}+\epsilon),
$$

训练目标通过减去熵项鼓励所有码字获得更均衡的概率质量。为避免融合表示或潜在表示方差过小，模型还计算特征标准差下界损失：

$$
\mathcal{L}_{\mathrm{div}}=\max(0,\sigma_{\min}-\mathrm{std}(f))+\max(0,\sigma_{\min}-\mathrm{std}(z)).
$$

最终 RQ-VAE 损失为：

$$
\mathcal{L}_{\mathrm{RQ}}=\mathcal{L}_{\mathrm{rec}}
+\lambda_c\mathcal{L}_{\mathrm{commit}}
+\lambda_b\mathcal{L}_{\mathrm{codebook}}
+\lambda_u\mathcal{L}_{\mathrm{usage}}
+\lambda_h\mathcal{L}_{\mathrm{hard}}
-\lambda_e\mathcal{H}
+\lambda_d\mathcal{L}_{\mathrm{div}}.
$$

本实验使用 $\lambda_c=0.25,\ \lambda_b=0.25,\ \lambda_u=0.5,\ \lambda_h=1,\ \lambda_e=0.05,\ \lambda_d=5$。码本使用 MiniBatchKMeans 初始化，以减少初始阶段随机码字造成的无效分配。

### 4.4 低利用码字的恢复

在每个 reset 周期，模型对完整商品集合重新编码，并逐层计算残差到码字的最近距离与使用次数。若某码字在该层使用次数不超过阈值，则将其判定为低利用码字。对于数量为 $m$ 的低利用码字，模型选择最近距离最大的 $m$ 个残差作为候选：

$$
\mathcal{R}^{\mathrm{high}}_l=\mathrm{TopK}_{m}\left(\|r_{i,l-1}-e_{l,c_{i,l}}\|_2^2\right).
$$

新码字向量由高误差残差复制，并加入相对尺度的随机扰动：

$$
\tilde{e}_{l,k}=r_{i,l-1}+\eta\,\mathrm{std}(r_{l-1})\odot\varepsilon,
\qquad \varepsilon\sim\mathcal{N}(0,I).
$$

本实验每 10 个 epoch 执行一次恢复，噪声系数 $\eta=0.02$。恢复后立即重新计算最近码字，以保证后续层级使用更新后的残差。

### 4.5 基础 Semantic ID 与后缀碰撞处理

对于具有 $L=3$ 个基础码字的商品，基础标识为：

$$
b_i=(c_{i,1},c_{i,2},c_{i,3}).
$$

当多个商品共享同一 $b_i$ 时，系统在基础标识后附加第 4 个后缀 token：

$$
\mathrm{SID}_i=(c_{i,1},c_{i,2},c_{i,3},s_i),
$$

其中 $s_i\in\{0,\ldots,S-1\}$ 为后缀索引。基础 token 由层级和码字确定，后缀使用独立 token 区间表示。固定长度设置为 4，因此所有商品输出相同长度的 SID；碰撞仅在第四位消解。

后缀方案能够快速保证标识唯一，但其选择主要依据商品频次，不直接优化商品语义与用户行为的一致性。因此，本实验继续构造行为感知的全局分配方法。

### 4.6 行为聚类与商品行为画像

对训练集用户表示执行聚类，得到 $M=32$ 个用户行为簇。对商品 $i$，其行为画像记为：

$$
p_i=(p_{i,1},\ldots,p_{i,M}), \qquad
\sum_{m=1}^{M}p_{i,m}=1.
$$

$p_{i,m}$ 表示在训练集中与商品 $i$ 发生交互的用户属于第 $m$ 个行为簇的经验比例。该画像概括了商品的消费人群结构，而不是仅描述商品内容。

层级 $l$ 中码字 $k$ 的平均行为画像为：

$$
\bar{p}_{l,k}=\frac{\sum_i \mathbf{1}[c_{i,l}=k]p_i}{\sum_i \mathbf{1}[c_{i,l}=k]}.
$$

若某码字承载的用户群体与商品自身行为画像差异较大，则说明该商品与码字之间的行为语义一致性不足。

### 4.7 BC-GSID 成本与分配

对于商品 $i$ 的候选 SID 方案 $a=(a_1,\ldots,a_L)$，语义成本定义为潜在向量与候选码字和之间的余弦距离：

$$
d_{\mathrm{sem}}(i,a)=1-\cos\left(z_i,\sum_{l=1}^{L}e_{l,a_l}\right).
$$

行为分布差异采用归一化 L1 距离：

$$
d_{\mathrm{L1}}(i,a)=\frac{1}{L}\sum_{l=1}^{L}\frac{1}{2}\left\|p_i-\bar{p}_{l,a_l}\right\|_1.
$$

行为簇概率差异采用 KL 散度：

$$
d_{\mathrm{KL}}(i,a)=\frac{1}{L}\sum_{l=1}^{L}
\mathrm{KL}\left(p_i\;\|\;\bar{p}_{l,a_l}\right).
$$

设当前分配过程中第 $l$ 层码字 $k$ 的使用份额为 $u_{l,k}$，使用率惩罚为：

$$
d_{\mathrm{usage}}(i,a)=\sum_{l=1}^{L}\left(u_{l,a_l}-\frac{1}{K}\right)^2.
$$

综合考虑语义、行为、簇分布、使用率与后缀索引，候选 $(a,s)$ 的总成本为：

$$
J(i,a,s)=
(\lambda_{\mathrm{rec}}+\lambda_{\mathrm{sem}})d_{\mathrm{sem}}(i,a)
+\lambda_{\mathrm{beh}}d_{\mathrm{L1}}(i,a)
+\lambda_{\mathrm{clu}}d_{\mathrm{KL}}(i,a)
+\lambda_{\mathrm{usage}}d_{\mathrm{usage}}(i,a)
+\lambda_{\mathrm{suffix}}s.
$$

本实验设置 $\lambda_{\mathrm{rec}}=1,\ \lambda_{\mathrm{sem}}=0.2,\ \lambda_{\mathrm{beh}}=0.1,\ \lambda_{\mathrm{clu}}=0.1,\ \lambda_{\mathrm{usage}}=0.02,\ \lambda_{\mathrm{suffix}}=0.001$。候选集合由原始码字和每层 top-8 的替代码字组成，每个原始码字最多考虑 32 个后缀索引。

BC-GSID 需要同时满足两个约束：

$$
\forall i,\quad |\mathrm{SID}_i|=4,
$$

$$
\forall i\neq j,\quad \mathrm{SID}_i\neq \mathrm{SID}_j.
$$

原始设计考虑分层最小费用流求解。对于 49,765 个商品和每个商品 16 个基础候选，完整候选图超出实验约束下的求解时间预算，因此实际运行采用 `greedy_behavior_aware` 回退。该策略按照“行为熵与原始语义误差之和”从高到低处理商品，在候选集合与可用后缀中逐项选择成本最小且未占用的 SID。

### 4.8 长短期用户上下文

对于用户 $u$ 和当前预测位置之前的历史商品集合，长期上下文与短期上下文分别使用最近 20 个和最近 5 个商品的平均潜在表示：

$$
u^{\mathrm{long}}_u=\frac{1}{|\mathcal{H}^{\mathrm{long}}_u|}\sum_{i\in\mathcal{H}^{\mathrm{long}}_u}z_i,
$$

$$
u^{\mathrm{short}}_u=\frac{1}{|\mathcal{H}^{\mathrm{short}}_u|}\sum_{i\in\mathcal{H}^{\mathrm{short}}_u}z_i.
$$

若历史为空，两个向量均为零向量。拼接后的上下文为：

$$
h_u^{(0)}=\phi_u([u^{\mathrm{long}}_u;u^{\mathrm{short}}_u]),
$$

其中 $\phi_u$ 将 $2d_z$ 维输入映射到 T5 的 $d$ 维空间。随后使用两层 Transformer Encoder 建模上下文内部关系：

$$
h_u=\mathrm{TransformerEncoder}(h_u^{(0)}).
$$

模型准备 4 个可学习查询向量 $Q\in\mathbb{R}^{4\times d}$，并将其与上下文表示相加：

$$
P_u=\mathrm{LayerNorm}(Q+h_u).
$$

$P_u\in\mathbb{R}^{4\times d}$ 构成 4 个 soft prefix token。生成器将原输入 embedding 的前 4 个位置替换为 $P_u$，同时保留 attention mask 中的有效标记。soft prefix 不占用 SID 词表位置，也不改变目标 SID 的长度。

### 4.9 T5 生成器与条件生成

设用户历史经过 token 化后的嵌入序列为 $X_u\in\mathbb{R}^{T\times d}$。UCG 生成器使用 soft prefix 替换最前面的 4 个嵌入位置：

$$
\tilde{X}_u=[P_u;X_{u,4:T}].
$$

编码器输出为：

$$
H_u=\mathrm{T5Encoder}(\tilde{X}_u,\mathrm{mask}(u)).
$$

以目标 SID token 序列 $y=(y_1,\ldots,y_{L_{\mathrm{sid}}+1})$ 为标签，自回归生成损失为：

$$
\mathcal{L}_{\mathrm{next}}=
-\sum_{j=1}^{L_{\mathrm{sid}}+1}\log p(y_j\mid y_{<j},H_u).
$$

其中 $L_{\mathrm{sid}}=4$，包含 3 个基础码字和 1 个后缀 token；标签序列在此基础上再包含结束标记对应的监督位置，因此总长度为 5。上下文对齐损失使用归一化相似度：

$$
\mathcal{L}_{\mathrm{align}}=
1-\cos\left(\frac{1}{4}\sum_{j=1}^{4}P_{u,j},\phi_z(r_u)\right),
$$

其中 $r_u$ 在普通 UCG 训练中为长期上下文，在联合训练中为目标商品的潜在表示。该损失要求 soft prefix 与用户上下文或目标语义保持方向一致。

### 4.10 联合训练目标

O5 在 O4 的基础上对生成器和 RQ-VAE 编码器进行联合微调。除 next-SID 损失外，模型还计算前缀一致性损失：

$$
\mathcal{L}_{\mathrm{prefix}}=
\mathrm{CE}\left(\mathrm{logits}_{1:3},y_{1:3}\right),
$$

用于加强前三个基础码字段的预测一致性。排序损失将批次内其他样本的目标 SID 作为负例。设正例序列得分和 $N$ 个负例序列得分分别为 $s^+$ 和 $s_j^-$，排序损失为：

$$
\mathcal{L}_{\mathrm{rank}}=
-\log\frac{\exp(s^+)}{\exp(s^+)+\sum_{j=1}^{N}\exp(s_j^-)}.
$$

RQ-VAE 重建项为 $\mathcal{L}_{\mathrm{rec}}$。联合训练总目标为：

$$
\mathcal{L}_{\mathrm{joint}}=
\lambda_{\mathrm{next}}\mathcal{L}_{\mathrm{next}}
+\lambda_{\mathrm{rank}}\mathcal{L}_{\mathrm{rank}}
+\lambda_{\mathrm{align}}\mathcal{L}_{\mathrm{align}}
+\lambda_{\mathrm{prefix}}\mathcal{L}_{\mathrm{prefix}}
+\lambda_{\mathrm{rec}}\mathcal{L}_{\mathrm{rec}}.
$$

本实验使用 $\lambda_{\mathrm{next}}=1,\ \lambda_{\mathrm{rank}}=0.1,\ \lambda_{\mathrm{align}}=0.1,\ \lambda_{\mathrm{prefix}}=0.05,\ \lambda_{\mathrm{rec}}=1$。码本参数在联合训练中冻结，编码器使用 `1e-4` 学习率，生成器同样使用 `1e-4` 学习率。

### 4.11 解码与评价指标

标准 beam search 在每一步保留得分最高的 $B$ 个前缀，并累计 token 对数概率。最终对完整 SID 序列进行排序并映射到商品。前缀树约束解码只允许出现在 Trie 中的 token 前缀，从而过滤无效分支；穷举排序则对目录中的全部有效 SID 计算序列得分，用于评价 beam search 与穷举结果之间的差距。

对于每个用户，Recall@K 表示目标是否出现在前 $K$ 个结果中：

$$
\mathrm{Recall@K}=\frac{1}{|\mathcal{U}|}\sum_{u\in\mathcal{U}}\mathbf{1}[y_u\in\mathrm{TopK}(u)].
$$

由于每个用户只有一个目标商品，个体的 Recall 为 0 或 1。NDCG@K 对命中位置进行折损：

$$
\mathrm{NDCG@K}=
\frac{1}{|\mathcal{U}|}\sum_{u\in\mathcal{U}}
\frac{\mathbf{1}[y_u\in\mathrm{TopK}(u)]}{\log_2(\mathrm{rank}_u+1)}.
$$

MRR@20 使用首个命中的倒数排名：

$$
\mathrm{MRR@20}=
\frac{1}{|\mathcal{U}|}\sum_{u\in\mathcal{U}}
\frac{\mathbf{1}[\mathrm{rank}_u\leq20]}{\mathrm{rank}_u}.
$$

覆盖率定义为前 20 个结果中出现过的目录商品占比：

$$
\mathrm{Coverage}=\frac{|\bigcup_u \mathrm{Top20}(u)|}{|\mathcal{I}|}.
$$

无效 SID 率是解码过程中无法映射到有效商品的 beam 数占总 beam 数的比例：

$$
\mathrm{InvalidSIDRate}=\frac{N_{\mathrm{invalid}}}{N_{\mathrm{beam}}}.
$$

语义漂移、行为漂移和簇 KL 分别衡量分配后 SID 与原始潜在表示、行为画像和层级平均行为画像之间的差异。码本利用率表示至少被使用一次的码字比例，码本困惑度则根据码字频率定义：

$$
\mathrm{Perplexity}_l=
\exp\left(-\sum_{k=1}^{K}\hat{p}_{l,k}\log\hat{p}_{l,k}\right),
$$

其中 $\hat{p}_{l,k}$ 为第 $l$ 层码字 $k$ 的归一化使用频率。Recall@20 的置信区间使用 Wilson 区间，以考虑每个用户只有一次二值命中记录所带来的统计波动。

## 5. 实验设计

### 5.1 三层实验结构

实验按照数据、Semantic ID 和生成检索三个层次组织。每一层只改变当前研究对应的变量，其余设置保持一致。

| 实验层 | 研究目标 | 主要设置 |
|---|---|---|
| L1：数据与协议验证 | 确认完整数据流、商品目录、时间划分和特征构建 | 全量扫描、全局 5-core、240K 用户抽样、train-only 目录与特征 |
| L2：表示与 SID 分配 | 比较后缀 SID 与行为感知全局分配，并检验行为信息的来源 | O1 基线 SID、O2 BC-GSID、C1 行为打乱、C2 严格语义约束 |
| L3：生成与联合优化 | 比较用户条件化生成、组合模块和联合微调 | O1-O5 主实验、C3 上下文打乱、C4 严格语义组合控制 |

L1 为后续实验提供统一输入，L2 研究 Semantic ID 分配对推荐结果的影响，L3 在固定或受控 SID 结构上研究用户上下文注入与联合训练。为了避免模块间的训练预算差异，O1-O4 与 C1-C4 均使用相同生成器规模、批次大小、学习率、训练步数和 beam 预算；O5 从 O4 检查点继续训练，因此单独记录其微调预算。

### 5.2 RQ-VAE 参数

| 参数 | 设置 |
|---|---:|
| 量化层级 $L$ | 3 |
| 每层码本大小 $K$ | 256 |
| 潜在维度 $d_z$ | 32 |
| 隐藏维度 | 512 |
| Dropout | 0.1 |
| Batch size | 1024 |
| 优化器 | AdamW |
| 学习率 | $5\times10^{-4}$ |
| Weight decay | $1\times10^{-5}$ |
| 实际训练轮数 | 2000 |
| Commitment 权重 | 0.25 |
| Codebook 权重 | 0.25 |
| Soft usage 权重 | 0.5 |
| Hard usage 权重 | 1 |
| Entropy 权重 | 0.05 |
| Diversity 权重 | 5 |
| 低利用码字恢复间隔 | 10 epochs |
| 恢复噪声 | 0.02 |
| KMeans 最大迭代 | 100 |

RQ-VAE 负责将 768 维文本特征和 128 维协同特征融合后量化为三层码字。码本初始化使用 MiniBatchKMeans，训练过程中同时优化重建、commitment、codebook、使用率、熵和表示多样性。低利用码字按固定周期恢复，恢复时使用完整商品残差中的高误差样本替换低使用码字，从而将有限码字容量重新分配给难以重建的商品区域。

### 5.3 Semantic ID 分配参数

| 参数 | 设置 |
|---|---:|
| SID 长度 | 4 |
| 基础码字层级 | 3 |
| 后缀 token 位 | 1 |
| 每商品基础候选数 | 16 |
| 每层代码候选数 | 8 |
| 每原始组合后缀候选数 | 32 |
| 行为簇数量 | 32 |
| $\lambda_{\mathrm{rec}}$ | 1 |
| $\lambda_{\mathrm{sem}}$ | 0.2 |
| $\lambda_{\mathrm{beh}}$ | 0.1 |
| $\lambda_{\mathrm{clu}}$ | 0.1 |
| $\lambda_{\mathrm{usage}}$ | 0.02 |
| 实际求解器 | `greedy_behavior_aware` |

后缀 SID 将原始三层码字与后缀直接组合，保证固定长度和唯一映射。BC-GSID 进一步搜索每层的替代码字以及后缀位置，选择综合语义、行为和码字使用率成本最低的未占用组合。由于 49,765 个商品的层级候选图超过了本实验设定的精确求解时间预算，最终采用确定性贪心回退，并在 SID 报告中记录请求求解器和实际求解器。

### 5.4 生成器与用户上下文参数

| 参数 | 设置 |
|---|---:|
| 生成器结构 | T5-tiny |
| $d_{\mathrm{model}}$ | 128 |
| Encoder/Decoder 层数 | 4 / 4 |
| Attention heads | 6 |
| Feed-forward 维度 | 512 |
| Dropout | 0.1 |
| 最大历史商品数 | 20 |
| 时间桶数量 | 16 |
| 哈希用户桶数量 | 2000 |
| Batch size | 256 |
| 学习率 | $3\times10^{-4}$ |
| Weight decay | $1\times10^{-4}$ |
| Warmup steps | 1000 |
| Gradient clip | 1.0 |
| 混合精度 | BF16 |
| Beam size | 20 |
| UCG 层数 | 2 |
| UCG heads | 4 |
| Soft prefix token 数 | 4 |
| 长期窗口 | 20 |
| 短期窗口 | 5 |

O1 使用后缀 SID 和标准 T5 生成器，输入中包含历史 SID、时间桶和哈希用户 token。O3 和 O4 额外启用 UCG，将长短期上下文映射为 4 个 soft prefix token。soft prefix 替换输入 embedding 的前四个位置，但对应 attention mask 保持有效，使上下文可以参与编码器自注意力。

### 5.5 主实验与训练预算

| 设置 | SID 分配 | 用户上下文 | 训练预算 | 说明 |
|---|---|---|---:|---|
| O1 L2-lite | 后缀 SID | 无 soft prefix | 50,000 步 | 标准生成器基线 |
| O2 BC-GSID | BC-GSID | 无 soft prefix | 50,000 步 | 检验行为感知分配 |
| O3 UCG | 后缀 SID | 长短期 soft prefix | 50,000 步 | 检验用户上下文 |
| O4 BC-GSID+UCG | BC-GSID | 长短期 soft prefix | 50,000 步 | 检验模块组合 |
| O5 Joint | BC-GSID | 长短期 soft prefix | 20,000 步 | 从 O4 继续联合微调 |

O1-O4 均从零训练 50,000 步，采用相同模型结构和解码预算，便于比较 SID 分配与用户上下文的独立作用。O5 从 O4 检查点继续训练 20,000 步，优化目标包含 next-SID、排序、上下文对齐、前缀一致性和 RQ-VAE 重建损失；码本参数冻结，生成器和 RQ-VAE 编码器使用 $1\times10^{-4}$ 学习率。

### 5.6 控制实验

控制实验用于区分真实信息、随机信息和严格语义约束的影响。C1 将商品行为画像打乱；C2 使用真实行为画像，但对语义漂移设置不超过基线 1.05 倍的约束；C3 保留真实 SID，仅打乱 UCG 使用的用户上下文；C4 将严格语义 BC-GSID 与真实 UCG 组合。C1-C4 均训练 50,000 步，使用与主实验相同的生成器配置和评价协议。

### 5.7 评价协议与复现入口

评价面向验证集和测试集的历史长度分桶用户，标准解码和 Trie 约束解码均使用 beam size 20。评价包含 Recall@10、Recall@20、NDCG@10、NDCG@20、MRR@20、Coverage、Invalid SID Rate、语义漂移、行为漂移、簇 KL、码本利用率和延迟等指标。另外对 500 名用户执行全目录穷举排序，用于比较受限 beam 与完整候选评分。

运行环境可通过 [环境检查脚本](scripts/check_env.py) 和 [主配置](configs/%6fvernight_5090.json) 核验。数据准备阶段可通过 `python scripts/link_or_download_data.py --root .` 下载或复用官方文件，单元测试可通过 `python -m pytest -q` 执行。训练入口接受 `preprocess`、`features`、`rqvae`、`sid`、`generators` 和 `evaluate` 阶段参数；控制实验使用 [控制实验脚本](scripts/run_controls.py) 启动。为了避免在文档正文中出现不必要的版本代号，具体入口链接和参数以仓库脚本为准。

## 6. 结果分析

### 6.1 主实验检索性能

下表报告 O1-O5 在标准 beam 和有效测试用户上的主要检索指标。由于每个用户只有一个目标商品，Hits 等于 Recall@20 与有效用户数的乘积。

| 指标 | O1 L2-lite | O2 BC-GSID | O3 UCG | O4 BC-GSID+UCG | O5 Joint |
|---|---:|---:|---:|---:|---:|
| Recall@10 | 0.004412 | 0.007220 | 0.006819 | 0.010028 | 0.009226 |
| Recall@20 | 0.009627 | 0.011633 | 0.011231 | 0.015243 | 0.012836 |
| NDCG@10 | 0.001946 | 0.004309 | 0.003712 | 0.005315 | 0.004375 |
| NDCG@20 | 0.003236 | 0.005441 | 0.004838 | 0.006644 | 0.005308 |
| MRR@20 | 0.001524 | 0.003733 | 0.003120 | 0.004209 | 0.003201 |
| Coverage | 0.040631 | 0.048669 | 0.034361 | 0.051422 | 0.043926 |
| Hits | 24 | 29 | 28 | 38 | 32 |
| Users | 2,493 | 2,493 | 2,493 | 2,493 | 2,493 |

O4 在 Recall@10、Recall@20、NDCG@10、NDCG@20、MRR@20 和 Coverage 上均取得最高值，其中 Recall@20 从 O1 的 0.009627 提高到 0.015243，相对增幅为 58.3%。相比单模块设置，O3 的 Recall@20 相对 O1 提高 16.7%，O2 相对 O1 提高 20.8%。O5 的 Recall@20 为 0.012836，相比 O4 下降 15.8%，说明当前联合训练目标没有在单次实验中保持 O4 的排序表现。由于 Hit 数量仍较少，这些差异需要结合后续置信区间和系统指标共同判断。

### 6.2 SID 质量、有效性与系统指标

| 指标 | O1 L2-lite | O2 BC-GSID | O3 UCG | O4 BC-GSID+UCG | O5 Joint |
|---|---:|---:|---:|---:|---:|
| Invalid SID rate | 0.002587 | 0.014160 | 0.000542 | 0.011412 | 0.081127 |
| Mean latency / ms | 59.427258 | 59.651832 | 59.908244 | 60.224925 | 61.242751 |
| P99 latency / ms | 76.733488 | 60.812650 | 62.741578 | 62.898815 | 68.037624 |
| Semantic drift | 0.000041 | 0.000254 | 0.000041 | 0.000254 | 0.000254 |
| Behavior drift | 0.726149 | 0.673541 | 0.726149 | 0.673541 | 0.673541 |
| Cluster KL | 1.620247 | 1.441210 | 1.620247 | 1.441210 | 1.441210 |
| Codebook utilization | 0.994792 | 0.981771 | 0.994792 | 0.981771 | 0.981771 |

BC-GSID 将平均行为漂移从 0.726149 降到 0.673541，簇 KL 从 1.620247 降到 1.441210，说明行为画像与层级分配之间的一致性有所提高。与此同时，平均语义漂移从 0.000041 增加到 0.000254，约为 O1 的 6.23 倍。虽然 O2 和 O4 的 Recall@20 较高，但语义漂移未满足严格 5% 增长约束，因此不能仅依据主表把 BC-GSID 解释为全面的语义改进。

UCG 没有改变 SID 及其语义、行为漂移，O3 的无效 SID 率降至 0.000542，平均延迟与 O1 接近。O5 的无效 SID 率升高到 0.081127，同时平均延迟上升到 61.24 ms，表明联合微调后的解码分布出现了更多无法映射的序列。O2、O4 和 O5 的码本利用率均为 0.981771，仍处于较高水平，但低于 O1 和 O3 的 0.994792。

### 6.3 Recall@20 置信区间

为了反映测试用户数量和命中数量带来的不确定性，下表报告 Recall@20 的 Wilson 95% 置信区间。

| 设置 | Hits | Users | Recall@20 | 95% 置信区间 |
|---|---:|---:|---:|---:|
| O1 L2-lite | 24 | 2,493 | 0.009627 | [0.006478, 0.014285] |
| O2 BC-GSID | 29 | 2,493 | 0.011633 | [0.008111, 0.016656] |
| O3 UCG | 28 | 2,493 | 0.011231 | [0.007782, 0.016185] |
| O4 BC-GSID+UCG | 38 | 2,493 | 0.015243 | [0.011125, 0.020852] |
| O5 Joint | 32 | 2,493 | 0.012836 | [0.009107, 0.018064] |

O4 的置信区间上界高于 O1，但不同设置的置信区间存在明显重叠。命中数量在 24 到 38 之间，单个用户结果变化就可能引起较大的 Recall 波动。因此，当前结果更适合作为后续多随机种子实验的方向性依据，而不能替代重复实验、配对检验和更大测试集的稳定性分析。

### 6.4 控制实验结果

| 设置 | SID | 用户上下文 | Recall@10 | Recall@20 | NDCG@10 | Hits |
|---|---|---|---:|---:|---:|---:|
| C1 BC-GSID shuffled behavior | 打乱行为画像 | 哈希用户 token | 0.003209 | 0.006819 | 0.001840 | 17 |
| C2 BC-GSID hard semantic | 真实行为与严格语义预算 | 哈希用户 token | 0.006017 | 0.016847 | 0.002887 | 42 |
| C3 UCG shuffled context | 后缀 SID | 打乱长短期上下文 | 0.003209 | 0.010830 | 0.001949 | 27 |
| C4 BC-hard + UCG | 严格语义 BC-GSID | 真实长短期上下文 | 0.004813 | 0.008424 | 0.002253 | 21 |

C1 的行为打乱使 Recall@20 从真实 BC-GSID 的 0.011633 降到 0.006819，说明 O2 的增益与真实商品行为画像有关，而不是完全来自 SID 重构本身。C2 保留 10,690 次语义重分配，同时将平均语义漂移控制在 $4.021620\times10^{-5}$，低于 $4.288449\times10^{-5}$ 的预算；其 Recall@20 达到 0.016847，Hits 为 42，是控制实验中较高的召回结果。C2 的 NDCG@10 为 0.002887，仍低于 O4 的 0.005315，说明较高召回不等同于全面的排序质量提升。

C3 将上下文打乱后得到 Recall@20 0.010830，只比真实 UCG 的 0.011231 低 3.7%。这一差距较小，说明当前 UCG 增益对上下文排列的依赖性有限。C4 在严格语义约束下组合真实 UCG 后，Recall@20 降到 0.008424，相比原 O4 下降 44.7%，表明原组合在严格语义约束下没有保持稳定优势。

### 6.5 历史长度与模块表现

按历史长度分组后，O4 相对 O1 在短历史用户上的 Recall@20 相对提高 28.6%，说明组合设置对历史有限用户具有一定方向性帮助。但 O3 单独使用 UCG 时，短历史用户 Recall@20 相对 O1 下降 7.1%，说明 UCG 的优势并未稳定集中在短历史群体。该现象表明，长短期上下文编码器仍需要针对稀疏历史进行专门校准，并使用更多随机种子确认分组结果。

综合主实验与控制实验，支持度较高的是行为画像与 SID 分配之间存在可观察联系，以及严格语义约束下的 BC-GSID 能够保持较高 Recall@20。UCG 的单独增益、O4 的组合增益和 O5 的联合训练优势在当前实验条件下均不够稳定。由于当前为单种子实验，结果主要用于优先级判断、失败模式识别和后续实验设计，不作为最终的统计结论。

### 6.6 结果边界

本实验的工作负载、数据规模和训练预算均受单卡环境约束。O1-O4 的基础 SID 长度均为 4，模型参数量、beam 大小和训练步数保持一致，但 O5 属于在 O4 上的继续微调，因此其比较同时包含初始化差异。BC-GSID 使用贪心回退而非精确最小费用流，求解质量可能受到候选顺序、行为画像质量和语义预算的影响。控制实验中的 C1-C4 各自训练 50,000 步，但未覆盖多个随机种子，因此不能从单次差值直接推断总体分布。无效 SID、候选取样和短历史分组也可能造成结果波动，需要在后续实验中统一报告均值和标准差。

## 7. 研究总结

### 7.1 已完成工作

本实验完成了 Amazon Reviews 2023 Books 的完整数据构建和统一评价流程。数据侧使用全量评论流、全局 5-core、240,000 用户抽样和 train-only top-50K 商品目录，最终形成 1,134,600 条交互、102,167 名用户和 49,765 个商品。文本、协同、行为画像和 SID 分配均基于训练集或商品内容完成，能够为后续模型比较提供一致输入。

模型侧实现了三类核心组件。第一，RQ-VAE 使用门控多视图融合、KMeans 初始化、硬使用率约束、熵约束和 error-aware dead-code revival，提高码字利用并抑制长期低利用率。第二，BC-GSID 将语义距离、行为画像 L1、簇 KL、使用率惩罚和后缀成本统一到固定长度 SID 分配中，并记录贪心回退求解过程。第三，UCG 使用长窗口 20 和短窗口 5 的用户表示生成 4 个 soft prefix token，O5 在此基础上加入排序、上下文对齐、前缀一致性和重建目标进行联合微调。

实验侧完成了 O1-O5 主设置和 C1-C4 控制设置。主实验比较标准后缀 SID、行为感知分配、用户条件化生成、模块组合和联合训练；控制实验分别检验行为打乱、严格语义预算、上下文打乱和严格语义组合。全部设置使用统一评价脚本，报告检索指标、SID 有效性、码本健康度、语义与行为漂移以及运行时间。

### 7.2 主要结论

O4 在主表中取得最高 Recall 和 NDCG，其 Recall@20 相对 O1 提高 58.3%，同时行为漂移和簇 KL 降低。然而 O4 的平均语义漂移约为 O1 的 6.23 倍，且 Recall@20 置信区间与其他设置重叠，说明该结果具有方向性价值，但不足以单独支持稳定改进。

C2 hard-semantic BC-GSID 在严格语义预算下仍得到 Recall@20 0.016847 和 42 个命中，较原始 BC-GSID 提高 44.8%，较 O1 提高 75.0%。这说明真实行为画像能够在保持 Semantic ID 语义约束的同时提供可观察信息。C1 的行为打乱显著降低召回，进一步支持行为信息与结果之间的关联。

UCG 和 Joint 的当前证据较弱。真实 UCG 相对上下文打乱的 Recall@20 仅提高 3.7%，短历史分组还出现下降；C4 在严格语义约束下的组合召回低于原始 O4；O5 的 Recall@20 低于 O4，且无效 SID 率升高。综合来看，行为感知 SID 分配是后续优先验证方向，用户条件化生成和联合训练需要进一步调整表示、对齐目标与训练顺序。

### 7.3 后续研究方向

后续工作可以从以下方面展开：

1. 对主实验和控制实验扩展到 3-5 个随机种子，报告均值、标准差和配对 bootstrap 置信区间。
2. 扩大测试用户规模，并对不同历史长度、商品频度和长尾程度进行分层评价。
3. 使用精确或更稳定的层级分配求解器，并与当前贪心回退进行质量、时间和稳定性比较。
4. 针对短历史用户重新设计 UCG 的池化、对齐和负采样策略，分析上下文打乱增益较小的原因。
5. 研究联合训练的启动时机、冻结策略、损失权重和解码约束，降低 O5 的无效 SID 率。
6. 在更多 Amazon 2023 类目或其他推荐数据集上验证方法迁移性。
7. 引入 paired bootstrap、多重比较校正和计算资源归一化，形成更完整的论文级实验协议。

## 8. 参考文献

1. Rajput S, Mehta N, Singh A, et al. Recommender Systems with Generative Retrieval[J]. Advances in Neural Information Processing Systems, 2023. arXiv: [2305.05065](https://arxiv.org/abs/2305.05065).
2. Lee D, Kim C, Kim S, et al. Autoregressive Image Generation using Residual Quantization[C]. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. arXiv: [2203.01941](https://arxiv.org/abs/2203.01941).
3. van den Oord A, Vinyals O, Kavukcuoglu K. Neural Discrete Representation Learning[J]. Advances in Neural Information Processing Systems, 2017. arXiv: [1711.00937](https://arxiv.org/abs/1711.00937).
4. Raffel C, Shazeer N, Roberts A, et al. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer[J]. Journal of Machine Learning Research, 2020, 21(140): 1-67. arXiv: [1910.10683](https://arxiv.org/abs/1910.10683).
5. Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need[J]. Advances in Neural Information Processing Systems, 2017. arXiv: [1706.03762](https://arxiv.org/abs/1706.03762).
6. Hou Y, Li J, Fu X, et al. Bridging Language and Items for Retrieval and Recommendation: Benchmarking LLMs as Semantic Encoders[J]. arXiv preprint, 2024. arXiv: [2403.03952](https://arxiv.org/abs/2403.03952). Dataset: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/).
7. McAuley J, Targett C, Shi Q, et al. Image-based Recommendations on Styles and Substitutes[C]. Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2015: 43-52. DOI: [10.1145/2766462.2767755](https://doi.org/10.1145/2766462.2767755).
8. Ni J, Ábrego G H, Constant N, et al. Sentence-T5: Scalable Sentence Encoders from Pre-trained Text-to-Text Models[J]. Findings of ACL, 2022. arXiv: [2108.08877](https://arxiv.org/abs/2108.08877).








