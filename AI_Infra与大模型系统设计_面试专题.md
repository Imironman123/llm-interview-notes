------------------------------------------------------------------------

**来源与参考**

本文题目与思路整理自以下公开面经 / 题库 / 开源资料（按仓库名列出，便于自行查阅原文）：

- GitHub `712sir/ai-infra-career` —— AI Infra 求职路线与知识点地图，覆盖 GPU 架构、CUDA、并行训练、推理加速
- GitHub `wdndev/llm_interview_note` —— 大模型面试笔记，「04.分布式训练」「06.推理」章节
- GitHub `WeThinkIn/AIGC-Interview-Book` —— AIGC 面试宝典，「模型部署基础」「计算机基础」章节
- GitHub `km1994/LLMs_interview_notes` —— LLM 面试笔记（大模型基础、分布式、推理优化）
- NVIDIA 官方文档：CUDA C Programming Guide、A100/H100/H800 White Paper、Nsight Systems/Compute 用户手册、Megatron-LM 与 TensorRT-LLM 文档
- 经典论文：ZeRO (Rajbhandari et al., 2020)、Megatron-LM (Shoeybi et al., 2019)、GPipe、Pathways、Orca (continuous batching)、vLLM / PagedAttention (Kwon et al., 2023)、FlashAttention (Dao et al., 2022)、DistServe / Splitwise (PD 分离)

> 说明：文中的带宽、算力、耗时数字均为工业界常见量级估算，用于面试快速推导，实际以实测 profiling 为准，切勿当教条背诵。

------------------------------------------------------------------------

## 目录

1.  GPU 与硬件基础
2.  CUDA 编程基础
3.  训练系统
4.  推理系统
5.  数据与存储
6.  平台与调度
7.  系统设计大题
8.  性能调优方法论

------------------------------------------------------------------------

# 一、GPU 与硬件基础

## Q1：请从架构层面讲清 GPU 为什么适合深度学习，SM、warp、寄存器、共享内存、HBM 各是什么角色？

答：GPU 的设计哲学是 **throughput-oriented（吞吐优先）**，CPU 则是 latency-oriented（延迟优先）。CPU 把大量晶体管花在分支预测、乱序执行、大容量 cache 上，核心数少（几十个）；GPU 把晶体管花在 ALU 上，核心数成千上万，用海量线程的切换来隐藏访存延迟。

关键层级自顶向下：

- **SM（Streaming Multiprocessor，流多处理器）**：GPU 的基本调度单元，一个 A100 有 108 个 SM，H100 SXM 有 132 个 SM。每个 SM 内含：
  - **CUDA Core**：FP32/INT32 执行单元。A100 每 SM 64 个 FP32 core。
  - **Tensor Core**：矩阵乘加专用单元，做 `D = A*B + C` 的融合运算，是深度学习算力主力。
  - **Register File**：A100 每 SM 256KB 寄存器（65536 个 32-bit 寄存器），是所有存储中访问最快的（~1 cycle）。
  - **Shared Memory / L1**：A100 每 SM 可配置最多 164KB 的 shared memory + L1 组合。shared memory 是**片上可编程缓存**，由程序员显式管理，延迟约 20~30 cycle。
  - **Warp Scheduler**：每 SM 4 个，负责从驻留 warp 中挑选可发射的指令。
- **Warp（线程束）**：32 个线程为一组，是**硬件调度和指令发射的最小单位**。SM 上的线程以 warp 为单位被调度，同一 warp 的 32 个线程在同一时刻执行同一条指令（SIMT）。所以 **warp 内出现分支分歧（divergence）时，两条路径会串行化执行**，这是性能杀手。
- **HBM（High Bandwidth Memory）**：GPU 的显存（device memory / global memory），离 SM 最远但容量最大。A100 80GB HBM2e 带宽约 2039 GB/s；H100 SXM 80GB HBM3 约 3350 GB/s。GPU 所有核心共享这一条总线，**访存带宽往往是真正的瓶颈**。

层次化的访存速度（A100 量级）：寄存器（~1 cycle）→ shared memory（~30 cycle）→ L2 cache（~200 cycle）→ HBM（<sub>400</sub>600 cycle，2 TB/s 量级）。

**核心结论**：GPU 性能的评判不是”峰值算力多少 TFLOPS”，而是 **算术强度（Arithmetic Intensity, AI）**，即每字节访存能做的浮点运算量：

    AI = FLOPs / Bytes_moved          (单位: FLOP/Byte)
    理论峰值算力 P_peak (FLOP/s)
    峰值带宽 BW (Byte/s)
    拐点强度 AI_ridge = P_peak / BW

- 若 `AI < AI_ridge` → **memory-bound**，优化方向是减少访存（算子融合、量化、提高复用）。
- 若 `AI > AI_ridge` → **compute-bound**，优化方向是提高 Tensor Core 利用率（tiling、增大 batch）。

以 A100 FP16（312 TFLOPS，峰值带宽 2039 GB/s）为例，`AI_ridge = 312e12 / 2.039e12 ≈ 153 FLOP/Byte`。这意味着**只有算术强度超过 153 的 kernel 才可能打满算力**。这解释了为什么 Decode 阶段（每生成一个 token 要把整个模型权重读一遍，AI 极低）必然是 memory-bound，而 Prefill 阶段（大矩阵乘，AI 高）是 compute-bound —— 这正是 PD 分离架构的理论根基。

**追问：为什么 GPU 需要大量线程？它隐式地解决什么问题？** 答：为了**隐藏内存延迟**。HBM 延迟约 400~600 cycle，若一个 warp 发起 load 后阻塞等待，SM 就空转。硬件通过”多 warp 交替发射”（warp-level parallelism）把延迟藏起来：当 warp A 等待访存时，warp scheduler 立刻切到 warp B 发射指令。因此要隐藏延迟，**每个 SM 必须有足够多的驻留 warp**，这就是 occupancy（占用率）的意义 —— 若 occupancy 太低，访存延迟无法被隐藏，SM 利用率就上不去。

------------------------------------------------------------------------

## Q2：A100 / H100 / H800 / RTX 4090 关键参数对比，面试怎么答？

答：给一张可直接背的表（数字为该型号常见配置量级）：

| 参数 | A100 SXM 80GB | H100 SXM 80GB | H800 SXM 80GB | RTX 4090 |
|----|----|----|----|----|
| 架构 | Ampere | Hopper | Hopper | Ada Lovelace |
| SM 数 | 108 | 132 | 132 | 128 |
| FP32 (CUDA Core) | 19.5 TFLOPS | 67 TFLOPS | 67 TFLOPS | 82.6 TFLOPS |
| FP16/BF16 Tensor（含稀疏） | 624 / 312 TFLOPS | 1979 / 989 TFLOPS | 同 H100 算力基本保留 | ~330 TFLOPS（FP16 dense） |
| TF32 Tensor | 156 TFLOPS | 494 TFLOPS | — | — |
| FP8 Tensor | 不支持 | 1979 TFLOPS | 同 H100 | 支持 FP8（约 660 TFLOPS） |
| 显存 | 80GB HBM2e | 80GB HBM3 | 80GB HBM3 | 24GB GDDR6X |
| 显存带宽 | 2039 GB/s | 3350 GB/s | 3350 GB/s | 1008 GB/s |
| L2 Cache | 40MB | 50MB | 50MB | 72MB |
| NVLink | NVLink 3, 600 GB/s | NVLink 4, 900 GB/s | **约 400 GB/s（阉割）** | 无（仅 PCIe） |
| NVSwitch | 支持（DGX） | 支持 | 支持 | 不支持 |
| TDP | 400W | 700W | 700W | 450W |
| 典型用途 | 训练/推理 | 训练/推理 | 国内合规训练 | 单卡微调/推理/桌面 |

**H800 的关键点（国内面试高频）**：H800 是 H100 的合规版，**算力基本保留（FP16 dense ~989 TFLOPS 与 H100 相同），但 NVLink 带宽被砍到约 400 GB/s**（H100 为 900 GB/s）。这直接影响并行策略选型：**NVLink 变慢 → 通信密集型并行（张量并行 TP、专家并行 EP）的代价变大，应尽量减少 TP 通信量或增大通信与计算重叠**。很多团队因 H800 互联弱，会倾向”减少 TP 度数、增大 PP/DP 度数”并加强 PP 的 communication overlap。

**4090 的关键点**：消费卡，算力性价比极高，但： 1. **24GB 显存太小**，无法单卡放下大模型权重，且无 NVLink，多卡只能走 PCIe 4.0 x16（~32 GB/s 双向），**多卡通信极慢，几乎不能做 TP**； 2. **无 ECC 显存**，长跑训练可能静默数据损坏； 3. 驱动为 GeForce 版，**禁止数据中心许可**，且 P2P 被驱动限制。 所以 4090 适合：单卡 LoRA 微调、量化推理、边缘部署、中小模型实验，**不适合大规模分布式训练**。

**追问：为什么 H100 的 FP8 很重要？** FP8 (E4M3/E5M2) 相对 FP16 有 2 倍算力和一半访存。Transformer 训练中，前向的大部分矩阵乘、以及 attention 的 QK^T / PV 都可以用 FP8，配合 per-tensor / per-block scaling 精度损失可控。理论上限收益：训练吞吐 <sub>1.5</sub>1.9x。代价是要处理 scaling factor、amax 统计，以及部分层（LayerNorm、softmax、残差）必须保持高精度。这是 Hopper 世代相比 Ampere 在训练侧最大的单点增量。

------------------------------------------------------------------------

## Q3：NVLink / NVSwitch / InfiniBand 三种互联各解决什么问题？带宽数量级是多少？

答：三者属于**不同层级**的互联，面试要分层说清：

**1) 节点内（intra-node）—— NVLink + NVSwitch** - **NVLink**：GPU 之间的高速直连总线，绕过 PCIe。H100 每 GPU 18 条 link，单向 450 GB/s，**双向 900 GB/s**（A100 为 600 GB/s）。比 PCIe 4.0 x16（双向 ~64 GB/s）快一个数量级。 - **NVSwitch**：当 GPU 数量超过 NVLink 直连拓扑能力时，用 NVSwitch 芯片做**无阻塞 crossbar 交换**。DGX H100 内 4 颗 NVSwitch，实现 8 卡任意两卡 900 GB/s 全互联（all-to-all），聚合带宽 7.2 TB/s。没有 NVSwitch 时，8 卡之间是混合拓扑（部分走 NVLink、部分走 PCIe），通信性能不均。 - NVSwitch 的意义：**让 TP（张量并行）在 8 卡内几乎无通信惩罚**，这是 Megatron 8 卡 TP 能跑起来的前提。

**2) 节点间（inter-node）—— InfiniBand / RoCE** - **InfiniBand (IB)**：专用高性能网络。常见 HDR（200 Gb/s = 25 GB/s 单向）、NDR（400 Gb/s = 50 GB/s）、XDR（800 Gb/s）。配 **SHARP** 可在交换机内做 AllReduce 的规约，减少流量。 - **RoCEv2**（RDMA over Converged Ethernet）：在普通以太网上跑 RDMA，成本低但需 PFC/ECN 调优避免丢包。 - **RDMA 的意义**：零拷贝、内核旁路（kernel bypass）、CPU 卸载。NCCL 在 IB 上走 RDMA，通信不占用 CPU 和显存带宽（用 GPUDirect RDMA 让网卡直接读 GPU 显存）。

**3) 带宽金字塔（H100 集群典型）**：

    寄存器       ~19 TB/s (每 SM)
    Shared/L1    ~33 TB/s (全芯片聚合)
    HBM          ~3.35 TB/s (每卡，80GB)
    NVLink       ~900 GB/s (双向, 卡间, 节点内)
    IB NDR       ~50 GB/s (单向, 节点间)   ← 与 NVLink 差约 18x
    PCIe 4.0     ~32 GB/s (双向)
    以太 100G    ~12.5 GB/s

**关键推论**：**节点内带宽（900 GB/s）是节点间（50 GB/s）的约 18 倍**。因此并行策略的核心原则是：**把通信量最大的并行维度放在节点内（NVLink），把通信量小的放节点间（IB）**。这也是 TP 通常限制在单机 8 卡内、PP/DP 跨机的原因。

**追问：为什么跨机 AllReduce 会成为瓶颈？通信量怎么算？** AllReduce 的通信量用 **Ring AllReduce** 分析（N 个 rank，数据量 S 字节）：

    每卡发送量 = 2 * S * (N-1) / N  → 大 N 时趋近 2S
    总时间     ≈ 2S(N-1)/(N*BW) + latency 项

即 **每卡发送量约 2S**（Reduce-Scatter 一次 S(N-1)/N，All-Gather 再一次）。以 7B 模型、BF16 梯度 14GB、128 卡为例：每卡发送 ~28GB，IB NDR 50 GB/s 下约 0.56s。若单步计算只需 0.3s，通信完全无法隐藏 → 必须靠 ZeRO 分片、梯度压缩或增大 batch 摊薄。这就是 **DP 的通信墙**。

------------------------------------------------------------------------

## Q4：MFU 和 BFU 怎么算？给一个具体模型算一遍。

答： - **MFU（Model FLOPs Utilization）** = 实际达到的每秒有效模型浮点运算数 / 硬件峰值算力。分子是**理论上必须做的运算**（不考虑重算、padding）。 - **BFU（Hardware/Bridged FLOPs Utilization）** = 实际硬件执行的 FLOPs / 峰值算力，包含了重计算（activation recomputation）带来的额外算力。因此 **BFU ≥ MFU**，两者差距反映重计算开销。

**标准计算公式（Transformer，未含 attention 的二次项）**：

    单 token 前向 FLOPs ≈ 2 * N_params          (一次乘加算 2 FLOPs)
    训练含反向 ≈ 3 倍前向 ≈ 6 * N_params
    完整一步（含 attention 二次项, 序列长 s）:
      C ≈ 6*N*s  +  12 * L * s^2 * h      (后者为 attention 的 QK^T 与 PV)

**MFU 定义式**：

    MFU = (6 * N_params * tokens_per_step) / (step_time * P_peak * N_gpu)

**具体算例**：7B 模型（N=7e9），序列 4096，global batch 4M tokens，用 128 张 A100（P_peak FP16 = 312 TFLOPS），实测单步 8 秒。

    分子 = 6 * 7e9 * 4e6 = 1.68e17 FLOPs
    分母 = 8 * 312e12 * 128 = 3.19e17 FLOPs
    MFU  = 1.68e17 / 3.19e17 ≈ 52.6%

52.6% 是很健康的数字（工业界 40~55% 算优秀；含 attention 项后真实 MFU 会更低）。**若 MFU 低于 35%，通常说明：batch 太小、通信未重叠、或存在大量 padding/重计算。**

**追问：MFU 低怎么系统性归因？** 逐条排查： 1. **数据/流水线气泡**：PP bubble 占比 `(p-1)/(m+p-1)`（p=PP 阶段数，m=micro-batch 数），若 m 太小气泡巨大。 2. **通信未重叠**：DDP 的 AllReduce 是否与 backward 重叠；TP 的 AllReduce 是否 overlap。 3. **padding 浪费**：变长序列不做 packing 时，padding token 也在算。 4. **重计算开销**：开启 activation checkpointing 后 BFU 会显著高于 MFU。 5. **小算子与 kernel launch 开销**：kernel launch 每条约 5~10us，小算子密集时 CPU 成为瓶颈（timeline 中的 gap）。 6. **精度**：是否用 TF32/FP8/BF16，是否误用 FP32 fallback。

------------------------------------------------------------------------

## Q5：国产 AI 芯片（昇腾、寒武纪等）现状，面试怎么答不失分？

答：答题框架是 **“架构差异 → 软件栈成熟度 → 迁移成本 → 实际约束”**，不要只评论算力纸面参数。

**华为昇腾（Ascend）**： - 910B/910C 系列，达芬奇架构，含 Cube（矩阵）、Vector（向量）、Scalar 三类单元。HBM 带宽与 A100 同量级，FP16 算力标称可达 300+ TFLOPS（910B）。 - 软件栈 **CANN + MindSpore / torch_npu**。生态是最大痛点：算子覆盖不全、自定义算子需用 Ascend C 开发、部分动态 shape 与高级特性支持滞后。 - 集群方案 **Atlas 800/900**，互联用 **HCCS**（板内，对标 NVLink）+ **RoCE**（跨机）。超节点（CloudMatrix）用光互联做到高带宽全互联，是对 NVLink 域限制的一种绕法。 - 迁移现实：**CUDA → Ascend 不是改个 flag**，需要处理算子替换（`torch_npu`）、混合精度策略、通信库（HCCL 对标 NCCL）、以及大量性能调优。

**寒武纪（Cambricon）**： - MLU 系列（如 MLU370/590），自研 MLUarch 架构，软件栈 **Neuware / MagicMind**，支持 PyTorch 的适配层（`torch_mlu`）。 - 优势在推理与特定场景性价比；训练生态相对更弱。

**其他**：海光 DCU（类 ROCm/HIP 路线，迁移成本相对低，因为 HIP 与 CUDA API 高度相似）；壁仞、摩尔线程等。

**面试话术**：安全且专业的答法是——“国产芯片在**算力密度和互联**上已接近可用，真正的差距在**软件栈成熟度与算子生态**。评估落地要看的不是峰值 TFLOPS，而是：① 目标模型的关键算子覆盖率；② 通信库在大规模下的有效带宽；③ 与你现有训练框架（Megatron/DeepSpeed）的适配度；④ 迁移后 MFU 相比 A100/H100 打几折。经验上，同等规模迁移初期 MFU 打 5~7 折属正常，靠 profiling 逐层优化可收回一部分。”

------------------------------------------------------------------------

# 二、CUDA 编程基础

## Q6：讲清 CUDA 的线程层次与 SIMT 执行模型。

答：线程层次是三层嵌套：

    Grid
     ├── Block 0
     │    ├── Warp 0  (thread 0-31)   ← 硬件调度单位
     │    │    ├── thread 0
     │    │    └── ... thread 31
     │    ├── Warp 1  (thread 32-63)
     │    └── ...
     ├── Block 1
     └── ...

- **Thread**：最小执行单元，有自己的寄存器和局部内存（local memory，实际是 global 的缓存）。
- **Block（线程块）**：可在一个 SM 上完整驻留，**块内线程可通过** `__syncthreads()` **同步，并共享 shared memory**。同一 block 的线程一定在同一个 SM 上；不同 block 可能在不同 SM，**block 之间无法同步**（这是设计约束，也是为什么需要 grid 级协作时要用 cooperative groups 或多次 kernel 启动）。
- **Grid**：一次 kernel launch 的全部线程。

**SIMT（Single Instruction, Multiple Threads）**： - 32 个线程组成一个 warp，**共享一个 program counter**，同一时刻执行同一条指令。 - 每个线程有自己的寄存器，可以有不同的数据（这就是 “Multiple Threads” 与 SIMD 的区别）。 - **分支分歧（Divergence）**：若 warp 内线程走不同分支（如 `if (tid % 2)`），硬件会**串行执行两条路径**，并用 mask 屏蔽不活跃线程，代价 = 两条路径耗时之和。避免手段：让分支条件以 warp 为粒度（32 对齐），或用谓词化（predication）替代短分支。

**与 SIMD 的区别（面试爱问）**：SIMD 是数据级并行，指令对向量寄存器操作；SIMT 是线程级并行，程序员写标量代码，硬件负责把 32 个线程的指令”捆”成一条。SIMT 更灵活（每线程可有独立控制流，代价是分歧），SIMD 更高效但要求数据规整。

**追问：一个 block 应该开多大？occupancy 是什么？** - Block 大小常取 **128 或 256**（是 warp 大小 32 的整数倍，且为 SM 能容纳的驻留块数留下余量）。取 1024 会导致每 SM 驻留块数少，尾块浪费严重。 - **Occupancy = 实际驻留 warp 数 / SM 支持的最大 warp 数**。A100 每 SM 最大 64 个 warp（2048 线程）。 - 限制 occupancy 的三个资源：**寄存器数/线程**、**shared memory/block**、**block 数上限**。 `受寄存器限制的`` warp ``数`` = 65536 / (regs_per_thread * 32)   ``受`` smem ``限制的`` block ``数``  = smem_per_sm / smem_per_block   occupancy = ``min(上述约束)`` / 64` 例：每线程用 64 个寄存器 → `65536/(64*32) = 32` warp → occupancy = 32/64 = 50%。 - **occupancy 高不等于快**。过高会挤占寄存器导致 spill，且有些 kernel 靠 ILP（指令级并行）和 shared memory 复用就能隐藏延迟。经验：**先用 Nsight 看 stall reason，再决定是否调 occupancy**。

------------------------------------------------------------------------

## Q7：什么是内存合并访问（coalescing）？bank conflict 怎么产生和避免？

答：**内存合并访问**：warp 内 32 个线程的一次 global load/store，若地址落在**连续的 128 字节段**内，硬件会合并成少量事务（transaction）。若地址发散（stride 大或不连续），会拆成多个事务，**有效带宽按事务数打折**。

    理想: thread i 访问 addr_base + i*4  → 32 线程共 128B → 1 个 128B 事务
    糟糕: thread i 访问 addr_base + i*128 → 32 个不同 128B 段 → 32 个事务 → 带宽 1/32

**经典反例**：按列访问二维数组（`A[threadIdx][j]`，行优先存储），warp 内线程跨行 → stride = N\*4 字节，完全不合并。 **解法**：让 warp 内线程沿内存连续维度索引，或先做 shared memory 转置（tiled transpose）再用合并方式写回。矩阵转置的经典优化就是”读合并 + 写 shared + 读 shared + 写合并”。

**Bank Conflict（共享内存 bank 冲突）**： - shared memory 被划分为 **32 个 bank**，每个 bank 宽 4 字节。地址 `addr` 落在 `bank = (addr/4) % 32`。 - 若**同一 warp 内多个线程访问同一 bank 的不同地址** → 冲突，硬件串行化，**N-way conflict 使延迟变为 N 倍**。 - 关键特例：**多个线程访问同一 bank 的同一地址 → 广播（broadcast），不冲突**。

经典场景： 1. **行优先访问** `smem[ty][tx]` **且** `tx` **为连续索引** → 各线程落在不同 bank，无冲突。 2. **按列访问** `smem[tx][ty]`（stride = 行宽）→ 若行宽是 32 的倍数，则所有线程落在同一 bank → **32-way conflict（最坏）**。 3. **解法**：**padding**，把数组声明为 `smem[BLOCK][BLOCK+1]`，行宽变为 33，则第 i 行起始 bank 偏移 i，冲突被打散。 4. **另一个朴素例子**：`smem[threadIdx.x * 2]`（stride 2）→ 2-way conflict。

**追问：**`__syncthreads()` **的作用与陷阱？** 作用：块内线程栅栏，确保所有线程到达此点，且此点之前的 shared/global 写对之后所有线程可见。陷阱：**它必须被块内所有线程以相同次数执行**。若放在分歧的分支里（如 `if (tid < 32) __syncthreads();`），会导致**死锁或未定义行为**。

------------------------------------------------------------------------

## Q8：手撕场景题——用 CUDA 实现矩阵乘法，怎么优化到接近 cuBLAS？

答：这是 CUDA 面试最高频的手撕题。思路按**优化阶梯**递进。

**Level 0：naive**

    __global__ void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < M && col < N) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += A[row * K + k] * B[k * N + col];   // B 的访问 stride=N，极不合并
            C[row * N + col] = acc;
        }
    }

问题：A 按行连续（warp 内合并），**B 的访问 stride = N → 完全不合并**；且每次乘加都访存，算术强度 = 2 FLOP / 8 Byte = 0.25，远低于拐点。

**Level 1：Tiling（分块）** 把 A、B 的 tile 先搬到 shared memory，块内复用：

    #define TS 32
    __global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
        __shared__ float As[TS][TS];
        __shared__ float Bs[TS][TS];
        int tx = threadIdx.x, ty = threadIdx.y;
        int row = blockIdx.y * TS + ty, col = blockIdx.x * TS + tx;
        float acc = 0.f;
        for (int t = 0; t < (K + TS - 1) / TS; ++t) {
            As[ty][tx] = (row < M && t*TS+tx < K) ? A[row*K + t*TS + tx] : 0.f;
            Bs[ty][tx] = (t*TS+ty < K && col < N) ? B[(t*TS+ty)*N + col] : 0.f;
            __syncthreads();
            #pragma unroll
            for (int k = 0; k < TS; ++k)
                acc += As[ty][k] * Bs[k][tx];
            __syncthreads();
        }
        if (row < M && col < N) C[row*N + col] = acc;
    }

收益：每次 HBM 访存被复用了 TS 次，算术强度大幅上升；且读 A、B 都是合并的。注意 `Bs[k][tx]` 在 k 变化时同 bank 广播，无冲突。

**Level 2：Register Tiling / Thread Coarsening** 每个线程算 **TM x TN**（如 8x8）个输出，把 acc 放进寄存器，shared memory 读的次数减少：

    float acc[TM][TN] = {0};
    for (k tile) {
        load As, Bs; __syncthreads();
        float regA[TM], regB[TN];
        for (int k = 0; k < TS; ++k) {
            #pragma unroll
            for (int i=0;i<TM;i++) regA[i] = As[ty*TM+i][k];
            #pragma unroll
            for (int j=0;j<TN;j++) regB[j] = Bs[k][tx*TN+j];
            #pragma unroll
            for (int i=0;i<TM;i++)
              #pragma unroll
              for (int j=0;j<TN;j++) acc[i][j] += regA[i]*regB[j];
        }
        __syncthreads();
    }

每个线程的计算量从 1 次乘加变成 TM\*TN 次，而 shared memory 访存只增加 TM+TN 次 → **计算/访存比显著提升**。这是手写 SGEMM 达到 cuBLAS 80%+ 的关键一步。

**Level 3：向量化访存 + 双缓冲** - 用 `float4` 读 shared memory 与 global memory：`reinterpret_cast<float4*>(&As[...])[0]`，减少指令数、提升合并效率。 - **Double buffering（双缓冲 / software pipelining）**：用两块 shared buffer，在计算 tile t 的同时异步预取 tile t+1（`cp.async` / TMA），把访存延迟与计算重叠。

**Level 4：用 Tensor Core（WMMA / mma）** 手写 `mma.sync.aligned.m16n8k16` 或直接用 WMMA API，从 FP32 累加器切到 FP16/BF16 输入 + FP32 累加。这是 A100 上从 19.5 TFLOPS（CUDA Core FP32）跃迁到 312 TFLOPS（Tensor Core FP16）的必经之路。**Hopper 上还要用 TMA + wgmma + warp specialization**。

**分值判断**：面试若只要求”写个能跑的”，Level 1 足够；若要求”性能分析”，要能讲清 Level 2/3/4 的收益来源与 roofline 位置。

**追问：手撕 Softmax / Reduce / LayerNorm 的要点？** - **Reduce（求和/求最大值）**：分块归约。① 每线程先串行累加若干元素；② warp 内用 `__shfl_down_sync` **蝶式/shuffle 归约**（比 shared memory 快，免 bank conflict 和 syncthreads）；③ 每个 warp 的 leader 把结果写 shared memory；④ 第一个 warp 再归约这些部分和。注意用 `__syncwarp` 或 mask 处理尾部。 - **Softmax**：数值稳定必须减最大值（`exp(x - max)`）。**Online softmax**（FlashAttention 核心）：单遍扫描，维护 running max `m` 与 running sum `l`，遇到更大的 max 时按 `exp(m_old - m_new)` 重缩放 `l` 和累加结果。这样避免两次 pass 访存，是 attention 高性能实现的基础。 - **LayerNorm**：先 reduce 求 mean 和 var（`var = E[x^2] - E[x]^2`，或用 Welford 在线算法更稳），再归一化 + 仿射。**优化点**：用 one-pass 同时算 sum 和 sum of squares（或 Welford），减少一次全局访存；算子融合（LN + residual + dropout 合一个 kernel）。

------------------------------------------------------------------------

## Q9：Triton 是什么？为什么现在算子开发在往 Triton 转？

答：**Triton** 是 OpenAI 开源的 GPU kernel 编程语言/编译器（Python DSL）。它在 CUDA 之上抽象了一层：**程序员以 “block” 为单位操作张量，不用手写线程级代码**。

对比： \| 维度 \| CUDA C++ \| Triton \| \|—\|—\|—\| \| 抽象层级 \| 线程级（thread/warp/shared mem） \| 块级（`tl.load`/`tl.dot` 操作 tile） \| \| 需管理 \| shared memory、bank conflict、warp shuffle \| 自动（编译器做 swizzling、vectorization、pipelining） \| \| 开发效率 \| 低，易出错 \| 高，几十行一个 kernel \| \| 性能上限 \| 最高（可手写 wgmma/TMA） \| 接近手写（A100 上 matmul 可达 cuBLAS 90%+） \| \| 生态 \| 成熟 \| 快速成长（vLLM、FlashAttention、Liger-Kernel 大量使用） \|

典型 Triton matmul 结构：

    @triton.jit
    def matmul_kernel(A, B, C, M, N, K, ...):
        pid_m, pid_n = ...
        offs_m = pid_m*BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = A + offs_m[:, None]*K + offs_k[None, :]
        b_ptrs = B + offs_k[:, None]*N + offs_n[None, :]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs)      # 自动做 coalescing / swizzle
            b = tl.load(b_ptrs)
            acc += tl.dot(a, b)      # 自动映射到 Tensor Core
            a_ptrs += BLOCK_K; b_ptrs += BLOCK_K
        tl.store(C + offs_m[:, None]*N + offs_n[None, :], acc.to(tl.float16))

**为什么转 Triton**：① 大模型新增算子的迭代速度决定生死，Triton 把开发从”几天”缩到”几小时”；② 编译器自动处理了 90% 的访存优化；③ **自动调优（autotune）** 一个装饰器就能扫 BLOCK_SIZE/num_warps/num_stages 空间。**什么时候还得用 CUDA**：需要极致性能、需要 warp specialization、需要 TMA/特殊指令、或算子语义无法用 tile 表达时。

------------------------------------------------------------------------

# 三、训练系统

## Q10：训练一个模型，显存到底被什么占满？给出完整公式并用 7B / 70B 算一遍。

答：显存占用分四大块（以 **Adam + 混合精度（BF16 参数 + FP32 主权重）** 为准）：

    1) 模型参数（权重）
       - 纯 BF16 训练：       2N bytes
       - 混合精度（常见）：    BF16 权重副本 2N + FP32 master weight 4N = 6N bytes
    2) 梯度
       - FP32 梯度:           4N bytes（若梯度也压成 BF16 则 2N）
    3) 优化器状态（Adam）
       - FP32 momentum m:     4N
       - FP32 variance v:     4N
       - 合计 8N bytes
    4) 激活值（activations）
       - 与 batch、序列长、层数、hidden 成正比，不受 N 直接约束

**标准”18N”结论（混合精度 + Adam，不含激活）**：

    2N (bf16 param) + 4N (fp32 master) + 4N (grad) + 4N (m) + 4N (v) = 18N bytes

常被简称为 **<sub>16N</sub>18N bytes**（不同实现略有差异，DeepSpeed 论文口径为 16N：2+2+4+4+4，即梯度也存 BF16）。

**算例（先不含激活）**： - **7B**：18 \* 7e9 = 126 GB → 单张 80GB A100 放不下 → **必须切分**。 - **70B**：18 \* 70e9 = 1260 GB → 需要 16 张 80GB 卡（理论下限）。 - **若用纯 FP32 全精度 Adam**：4N(param)+4N(grad)+8N(opt) = 16N，7B 需 112GB。

**激活值估算公式**（每层，per-sample，序列长 s，hidden h，Transformer）：

    粗略：每层激活 ≈ s * h * (若干倍) * dtype_bytes
    更实用: Activation_total ≈ L * s * b * h * c * dtype
      L=层数, s=序列长, b=micro-batch, h=hidden, c≈ 10~20 (与实现相关)

以 7B（L=32, h=4096）、s=4096、b=1、BF16 为例，**不做重计算**时激活可轻松达到几十 GB；**开启 activation checkpointing（重计算）** 后，只保存每层输入（约 `L*s*b*h*2` bytes = 32*4096*1*4096*2 ≈ 1GB 量级），代价是前向多算一遍（训练时间 +30%~40%）。

**结论**：**参数+梯度+优化器状态是可以精确算出的常量，而激活是随配置爆炸的变量**。所有训练显存优化（ZeRO、TP、PP、重计算、sequence parallel）本质都是在处理这两类的不同切分方式。

**追问：ZeRO 的 3 个阶段分别切了什么？省多少？** 核心是 **ZeRO-1 切优化器状态（8N → 8N/Nd）、ZeRO-2 再切梯度（省 4N）、ZeRO-3 再切参数（省全部 16N~18N）**。注意切分带来的是**通信换显存**，详见下一题。

------------------------------------------------------------------------

## Q11：ZeRO 一、二、三阶段的区别、通信量、以及为什么 ZeRO-3 不一定比 ZeRO-2 快？

答：**ZeRO (Zero Redundancy Optimizer)** 的核心思想：数据并行中，每张卡都完整保存一份参数/梯度/优化器状态，是**纯粹冗余**。ZeRO 把这些状态按 DP 组**分片（partition）**，每卡只存 1/Nd。

设 DP 度数为 `Nd`，参数量 N（单位元素）。

| 阶段 | 切分对象 | 每卡显存（混合精度，单位 N 元素） | 相对 DP 节省 | 额外通信 |
|----|----|----|----|----|
| 基线 DDP | 都不切 | 18N | — | 梯度 AllReduce ≈ 2S |
| **ZeRO-1** (P_os) | 优化器状态 | 2N + 4N + 4N + 8N/Nd | 约 4x（Nd=8） | 无额外（仍 AllReduce 梯度） |
| **ZeRO-2** (P_g) | \+ 梯度 | 2N + 4N + 4N/Nd + 8N/Nd | 约 8x | 无额外（用 Reduce-Scatter 替代 AllReduce） |
| **ZeRO-3** (P_p) | \+ 参数 | (2N+4N+4N+8N)/Nd ≈ 18N/Nd | 约 Nd 倍 | **+ All-Gather 参数（前向+反向各一次）** |

**通信量对比（每步，每卡，S 为总数据量）**：

    DDP:        2S            (AllReduce)
    ZeRO-1:     2S            (梯度 AllReduce，仅 optimizer 状态分片)
    ZeRO-2:     2S            (Reduce-Scatter 产生分片梯度)
    ZeRO-3:     2S + 3S ≈ 5S (额外: 前向 All-Gather S, 反向 All-Gather S, 反向 Reduce-Scatter S)

**ZeRO-3 的通信量约为 ZeRO-2 的 2.5 倍**。

**为什么 ZeRO-3 不一定更快（甚至显著更慢）**： 1. **通信量翻倍以上**（5S vs 2S），跨机 IB 场景下通信时间可能超过省显存带来的收益； 2. **参数 All-Gather 在关键路径上**：前向每层前都要先 gather 参数，形成”通信-计算-通信”的串行链，难以像 DDP 那样把 AllReduce 完全藏在 backward 后面； 3. **碎片化**：参数按层 gather，产生大量小消息，**latency 主导**而非带宽主导； 4. **实现复杂度与显存峰值**：gather 过程中有临时 buffer，显存峰值高于理论值。

**选型经验**： - **显存够 → ZeRO-2**（或 DDP + 重计算），通信最省； - **显存紧 → ZeRO-3**，代价是通信和速度； - **超大规模（70B+）** → ZeRO-3 配合 TP/PP 混用，并开启 **communication overlap**（DeepSpeed 的 `overlap_comm`、`contiguous_gradients`、bucket size 调优）； - **注意**：ZeRO-3 的通信瓶颈在跨机时更严重，因此**优先在节点内做 TP、把 ZeRO 用在跨机 DP**。

**追问：ZeRO-Offload / ZeRO-Infinity 是什么？** 把优化器状态、梯度甚至参数**卸载到 CPU 内存或 NVMe SSD**。原理是利用 CPU 内存（可到 TB 级）做参数的”后备存储”，GPU 只在需要时把当前层拉进来算。 - ZeRO-Offload：优化器状态 + 梯度放 CPU，optimizer step 在 CPU 上做。 - ZeRO-Infinity：进一步用 NVMe，支持单机跑 100B+ 模型。 - 代价：**PCIe 带宽（~32 GB/s 双向）远低于 HBM（2~3 TB/s）**，所以 offload 只适合”计算量/通信量比”高的场景，且需要 overlap 来掩盖延迟。

------------------------------------------------------------------------

## Q12：混合并行策略怎么设计？给定集群规模，怎么选 TP/PP/DP/EP 的度数？

答：先讲清四种并行的**通信特性**（这是选型的唯一标尺）：

| 并行 | 切什么 | 通信模式 | 通信频率 | 通信量级 | 适合放置 |
|----|----|----|----|----|----|
| **DP** (含 ZeRO) | batch 维度复制 | AllReduce / Reduce-Scatter | 每步 1~2 次 | ~2S（S=梯度） | **跨机**（最不敏感） |
| **TP** | 权重矩阵切分 | AllReduce / All-Gather | **每层 2 次（前向+反向各 2）** | 大（激活量级） | **节点内 NVLink** |
| **PP** | 按层切分 | P2P Send/Recv | 每 micro-batch 边界 | 小（只传边界激活） | 节点内或跨机（敏感度中） |
| **EP** | MoE 专家切分 | All-to-All | 每 MoE 层 2 次 | 大（token 路由） | **节点内优先** |

**关键：TP 通信最频繁、量最大 → 必须限制在 NVLink 域内（单机 8 卡）**。这就是”**TP ≤ 8，跨机靠 PP/DP**”铁律的来源。

**设计流程（面试按此步骤答）**：

Step 1 — **由显存反推最小并行度**：

    单卡可容纳的参数量上限 = (VRAM - 激活预留) / 18 bytes
    需要的总卡数下限 ≈ 18 * N_params / (可用显存)

例：70B 模型，80GB A100，预留激活与碎片后按 60GB 可用：`18 * 70e9 / 60e9 = 21` 卡 → 取整到 32 卡（8 卡 × 4 机）或 16 卡 + ZeRO-3。

Step 2 — **由带宽反推 TP 度数**：TP 只在 NVLink 内，`TP ∈ {1,2,4,8}`，一般取 8 打满节点。

Step 3 — **由集群规模算剩余维度**：`TP * PP * DP = ``总卡数`。例：1024 卡 = TP 8 × PP 8 × DP 16。

Step 4 — **PP 度数受气泡约束**：

    PP bubble 占比 ≈ (PP - 1) / (micro_batch_num + PP - 1)

PP=8 时若 micro-batch 数为 16，气泡 ≈ 7/23 ≈ 30%（太高）。**解法**：增大 micro-batch 数（global batch / (micro_batch_size \* DP)），或用 **interleaved / virtual pipeline（1F1B + 交错）** 把气泡降到 `(PP-1)/(m*PP)` 量级。

Step 5 — **验证通信可隐藏性**：估算 TP 的 AllReduce 时间 vs 每层计算时间，若通信 \> 计算的 40%，说明 TP 太大或需要 overlap。

**典型配置参考**：

    7B   / 8卡  : TP=1, PP=1, DP=8 (或 ZeRO-2)
    13B  / 8卡  : TP=2, PP=1, DP=4
    70B  / 64卡 : TP=8, PP=4, DP=2  (+ ZeRO-1)   [Megatron 官方推荐]
    175B / 128卡: TP=8, PP=8, DP=2  (+ ZeRO-1)
    MoE (万亿级) : TP + EP(专家并行) + DP + PP 混合

**追问：为什么要 PP + DP 而不是纯 DP 加 ZeRO-3？** 因为 ZeRO-3 的通信量是 PP 的很多倍。PP 只在层边界传一次激活（P2P，量小且可 overlap），而 ZeRO-3 每层都要 All-Gather 参数。**当模型大到单卡放不下、且集群跨机时，PP 的通信效率显著优于 ZeRO-3**。工业界（Megatron）因此主推 **TP+PP+DP 的三维并行**，ZeRO 只用于切 DP 组内的冗余状态（ZeRO-1）。

------------------------------------------------------------------------

## Q13：通信原语有哪些？各自通信量怎么算？怎么和计算重叠？

答：**基础集合通信原语**（以 N 个 rank、每 rank 数据量 S 字节）：

| 原语 | 语义 | 每卡发送量 | 每卡接收量 | 总通信量 |
|----|----|----|----|----|
| **Broadcast** | 1 → N | S | S | S |
| **Scatter** | 1 → N（分片） | S/N | S/N | S |
| **Gather** | N → 1（分片） | S/N | S/N | S |
| **All-Gather** | 全员互相收集 | S(N-1)/N → **S** | S | S\*N |
| **Reduce** | 归约到 1 个 | S(N-1)/N | — | S\*N |
| **Reduce-Scatter** | 归约 + 分片分发 | S(N-1)/N → **S** | S/N | S\*N |
| **All-Reduce** | Reduce + All-Gather | **2S(N-1)/N → 2S** | 同 | 2S\*N |
| **All-to-All** | 全员互相交换 | S(N-1)/N → **S** | S | S\*N |

**Ring AllReduce 的时间模型**（Bandwidth-optimal 实现）：

    T_allreduce ≈ 2 * (N-1)/N * S / BW + 2*(N-1) * latency
                ≈ 2S/BW   (N 较大时)

**注意 latency 项随 N 线性增长**——万卡时 latency 不可忽略，这是 Ring 算法的缺陷。**Tree / 递归倍增（Recursive Halving-Doubling）** 的 latency 项是 `2*log2(N)*latency`，更适合大 N 但带宽利用率略低。NCCL 会自动根据拓扑和 size 选择算法（Ring / Tree / CollNet / NVLS）。

**通信与计算重叠（overlap）的三种手段**：

1.  **梯度分桶（bucketing）**：DDP/ZeRO 不等到所有梯度算完再 AllReduce，而是**梯度一算完就立刻启动通信**。PyTorch DDP 用 `bucket_cap_mb`（默认 25MB）控制桶大小。桶太大 → 通信启动晚；桶太小 → 消息碎片、latency 占比高。
2.  **反向传播的天然流水线**：backward 是**从后往前**逐层产生梯度的，而各层的梯度 AllReduce 互相独立 → 可以做到”算第 k 层 backward 时，第 k+1 层的梯度正在通信”。理论重叠率可达 ~90%。
3.  **异步/多流（multi-stream）**：把 TP 的 AllReduce 放到独立 CUDA stream，与后续层的计算并行。Megatron 的 `--tp-comm-overlap`、DeepSpeed 的通信 overlap 开关。

**重叠效果的量级估算**：

    设单步计算时间 T_c，通信时间 T_comm
    无重叠:   T = T_c + T_comm
    完美重叠: T = max(T_c, T_comm)
    理想加速比 = (T_c + T_comm) / max(T_c, T_comm)

若 T_comm/T_c = 0.5，理想加速 1.5x。**实际只能做到 60~80% 重叠率**，因为首尾 bucket 无法隐藏、以及 SM 资源竞争（通信 kernel 也占 SM）。

**追问：为什么生产环境必须设置** `NCCL_IB_HCA`**、**`NCCL_SOCKET_IFNAME`**？** 因为多网卡/多链路环境下，NCCL 默认可能选错网卡或走 socket 而非 RDMA，导致通信性能断崖式下降。生产环境必须显式绑定 IB 网卡，并设置 `NCCL_IB_DISABLE=0`、`NCCL_NET_GDR_LEVEL`（GPUDirect RDMA 层级）、`NCCL_DEBUG=INFO` 来验证是否走了 RDMA。这是一个非常典型的”集群配置不对导致 MFU 腰斩”的现场问题。

------------------------------------------------------------------------

## Q14：PyTorch FSDP / Megatron-LM / DeepSpeed / Colossal-AI 怎么选？

答：**一句话总结**：FSDP 是 PyTorch 原生的 ZeRO-3；Megatron-LM 是 NVIDIA 的三维并行标杆；DeepSpeed 是微软的 ZeRO 全家桶 + 易用性；Colossal-AI 是学术派的多维并行 + 异构。

**PyTorch FSDP（Fully Sharded Data Parallel）**： - 本质是 **ZeRO-3**：参数、梯度、优化器状态全部分片。 - 关键机制：**按 layer 为单位**（`auto_wrap_policy`，用 TransformerBlock 做 wrapping）做 All-Gather → 计算 → 释放。`ForwardPrefetch`/`BackwardPrefetch` 预取下一层参数以隐藏通信。 - 优点：**PyTorch 原生，无第三方依赖，与 HuggingFace Trainer 无缝**，维护活跃。 - 缺点：缺少 TP 和 PP 的原生支持（只解决 DP 维度的切分），超大模型仍需外挂 TP（`DTensor` 在做，但不如 Megatron 成熟）；通信量比 ZeRO-2 大。 - 适用：**中小规模（≤70B）、想少引入依赖、DP 维度切分**。

**Megatron-LM**： - NVIDIA 出品，**TP + PP + DP 三维并行的工业级实现**，也是 TP 思想的发源地。 - 强项：**极致性能**。TP 的 fused kernel（ColumnParallelLinear / RowParallelLinear）、sequence parallel、selective activation recomputation、interleaved pipeline、FP8 训练，以及 Megatron-Core 对 MoE/EP 的支持。 - 缺点：**代码复杂、学习曲线陡、对模型结构有假设**（改模型需改代码）、配置项多。 - 适用：**百 B 以上、千卡以上、追求 MFU 的严肃训练**。

**DeepSpeed**： - 微软出品，**ZeRO 1/2/3 + Offload + Infinity + 推理/量化等**。 - 强项：**配置驱动（JSON），上手快**；ZeRO-Offload/Infinity 能单机跑大模型；与 HF 生态集成好。 - 缺点：ZeRO-3 性能不如 Megatron 的并行组合；纯 DP 路线在超大集群通信墙明显；部分高级功能生产稳定性需验证。 - 适用：**快速实验、资源受限（要 offload）、中等规模训练**。

**Colossal-AI**： - 新加坡 HPC-AI Tech，覆盖 **数据/张量/流水线/序列并行 + 异构（CPU offload）+ 自动并行搜索（Gemini/Auto-Parallel）**。 - 特色：`Gemini` 的 chunk-based 内存管理、`Booster` API 一行切换并行策略、对长序列支持较好。 - 缺点：社区规模和工业验证不如前三者。 - 适用：**研究、快速验证并行策略、长序列训练**。

**选型决策树（面试可直接背）**：

    模型 ≤ 13B, 单机或多机少卡      → DDP / FSDP (最简)
    模型 ≤ 70B, 有 NVLink 8 卡节点  → FSDP 或 DeepSpeed ZeRO-2/3
    模型 > 70B, 千卡以上            → Megatron-LM (TP+PP+DP)
    资源紧张, 想 offload 到 CPU/SSD → DeepSpeed ZeRO-Offload/Infinity
    需要 FP8 / MoE / 极致 MFU       → Megatron-Core / Megatron-DeepSpeed

**面试加分点**：指出 **Megatron-DeepSpeed 是两者的融合**（Megatron 提供 TP/PP，DeepSpeed 提供 ZeRO-1 切 DP 组的优化器状态），这是 175B 级训练的主流方案。

------------------------------------------------------------------------

## Q15：Checkpoint 与容错怎么做？故障恢复的工程细节？

答：大模型训练”三天两头挂”是常态。统计上，千卡集群的 **MTBF（平均无故障时间）可能低至几小时**，若 checkpoint 间隔 1 小时、保存/加载耗时 10 分钟，**有效算力利用率可能被吃掉 20%+**。容错体系分四层：

**1) 定期 Checkpoint** - 保存内容：模型参数、优化器状态（momentum/variance）、学习率调度器状态、**数据加载器的位置（data sampler state）**、随机数状态（RNG）、以及 step 计数。 - **分片保存（sharded checkpoint）**：每个 rank 只存自己那份分片，避免 rank 0 汇聚（会 OOM 且慢）。Megatron 的 distributed checkpoint；DeepSpeed 的 `--checkpoint` 自动分片。 - **异步保存（async checkpointing）**：把参数从 GPU 拷到 CPU pinned memory 后立即返回训练，后台线程再写盘。**把 checkpoint 的阻塞时间从”分钟级”降到”秒级”**。这是当前标配。

**2) 存储优化** - **Checkpoint 写盘量** = 18N bytes（含优化器状态）。70B 模型 = 1260 GB，写一次对存储是巨大压力。 - 优化手段：① **分层存储**（本地 NVMe 热存最近 N 个 + 远端对象存储冷存）；② **压缩**（参数可量化/稀疏化，但要注意恢复精度）；③ **只存模型参数**（用于推理发布，不用于续训）；④ 减少保留份数（`keep_last_n`）。 - 存储选型：**并行文件系统（Lustre / GPFS / JuiceFS）** 或对象存储 + 本地缓存。写入必须能打满，否则 checkpoint 时间随规模线性增长。

**3) 快速恢复** - **分层恢复**不现实（无法重放梯度），实际做法是 **降低 checkpoint 间隔 + 异步保存**，把”重算损失”控制在 10 分钟内。 - **弹性训练（elastic training）**：节点数变化时自动重分片。PyTorch 的 `torchrun --rdzv` + elastic。恢复时需重新构建通信组、重算 ZeRO 分片、部分实现会做 LR 的 warm restart。

**4) 故障检测与自愈** - **健康检查**：NCCL 通信 watchdog、GPU ECC 错误监控（`nvidia-smi -q` 的 ECC error count）、Xid 错误日志、NVLink 错误计数。 - **常见故障类型**：GPU 掉卡（Xid 48/79）、NVLink 降速、IB 链路 flapping、ECC uncorrectable error、节点掉电、**静默数据损坏（SDC）**。 - **自愈流程**：检测故障 → 标记节点不健康（`kubectl cordon`）→ 从任务中摘除 → 重新拉起训练（from latest checkpoint）→ 事后运维检修。 - **工程痛点**：**一个坏节点会让整个 1000 卡任务重启**。缓解：① 训练脚本要能”**换机器重启**”（幂等的 rendezvous）；② 快速节点健康预检（NCCL all-reduce smoke test）；③ 预留热备节点。

**追问：为什么”静默数据损坏（SDC）“很可怕？** 因为 ECC 无法检测的位翻转会让梯度悄悄变错，训练 loss 可能只是”轻微异常”甚至看起来正常，但模型最终质量下降。检测手段：定期做数值一致性校验（同一 batch 重算对比）、监控 loss/grad-norm 的异常尖刺、以及使用带 ECC 的显存（**这是 4090 不适合正式训练的原因之一**）。

------------------------------------------------------------------------

# 四、推理系统

> 边界说明：本章侧重**系统架构**（调度、批处理、多副本、PD 分离、KV 分布式、边缘部署），单点算子优化（量化、FlashAttention、投机解码）见部署专题文档。

## Q16：推理系统的核心指标与瓶颈在哪？和单点算子优化的边界是什么？

答：**边界划分**（面试要主动说清，体现系统视角）： - **单点优化（部署侧）**：量化（INT8/INT4/AWQ/GPTQ）、FlashAttention、算子融合、CUDA Graph、投机解码。目标是”**同一个请求跑得更快**”。 - **系统架构（本章重点）**：请求调度、批处理策略、多副本编排、KV Cache 的分布式管理、PD 分离、弹性伸缩、SLO 保障。目标是”**在给定硬件下，让整体吞吐和 SLO 最优**”。

**核心指标**： \| 指标 \| 含义 \| 关注方 \| \|—\|—\|—\| \| **TTFT** (Time To First Token) \| 首 token 延迟 = 排队 + prefill 时间 \| 交互体验 \| \| **TPOT / ITL** (Time Per Output Token) \| 每个输出 token 的间隔 \| 流式体验 \| \| **E2E Latency** \| 端到端总延迟 ≈ TTFT + (n_out-1)\*TPOT \| 用户感知 \| \| **Throughput** \| tokens/s（总吞吐） \| 成本 \| \| **QPS** \| 每秒请求数 \| 业务 \| \| **Goodput** \| **满足 SLO 前提下的有效吞吐** \| **系统设计真正的目标** \|

**关键矛盾：延迟 vs 吞吐**。批越大，GPU 利用率越高、吞吐越大，但**每个请求的 TPOT 变长**（batch 内要等），TTFT 也变长（排队）。**系统的艺术就是在 SLO 约束下把 batch 推到最大**。

**两阶段的瓶颈完全不同**（这是理解一切推理优化的钥匙）： - **Prefill**：一次处理整个 prompt，是大矩阵乘（M=prompt_len, K/N=hidden）→ **compute-bound**，GPU 算力吃满，TTFT 主要由它决定。 - **Decode**：每次只生成 1 个 token，矩阵乘退化成矩阵-向量乘（GEMV）→ **memory-bound**，时间 ≈ 权重字节数 / 显存带宽。

**Decode 的时间下界（必背公式）**：

    T_decode_per_token ≈ (2 * N_params * bytes_per_param) / HBM_BW

例：7B FP16 模型，`2 * 7e9 * 2 = 28 GB` 权重，A100 带宽 2039 GB/s：`28 / 2039 ≈ 13.7 ms` → **单请求理论极限约 13.7 ms/token（约 73 tokens/s）**。 这解释了：① 为什么 **batch 很重要**——batch 内多请求共享同一次权重读取，吞吐可线性提升而 TPOT 几乎不变；② 为什么 **量化（INT8/INT4）直接提速**——权重字节数减半/减四分之一；③ 为什么 **MoE 稀疏激活**能提速——只读激活的专家。

实测通常比理论下界差 1.5~3 倍（kernel launch、attention 开销、KV Cache 读取等）。

**追问：怎么判断我的推理服务是 prefill-bound 还是 decode-bound？** 看负载形态：① 长 prompt + 短输出（如文档摘要、RAG）→ prefill 主导，优化重点是 **prefill 吞吐 + chunked prefill**；② 短 prompt + 长输出（如代码生成、对话）→ decode 主导，优化重点是 **KV Cache 管理 + 大 batch + 量化**。两种负载混在一起时，**用一个统一 batch 会互相干扰**（prefill 会打断 decode 的稳定节奏），这正是 PD 分离的动机。

------------------------------------------------------------------------

## Q17：什么是 Continuous Batching？和静态批处理比好在哪？

答：**静态批处理（Static Batching，早期方案）**：攒够 B 个请求组成一个 batch，**整个 batch 一起 prefill、一起 decode，直到批次内最长的那个请求生成完，才释放整个 batch**。

问题（**padding 浪费**）：

    设 batch 内请求的输出长度差异大（如 10 到 500 token）
    静态 batch 必须等最长的 500 token 生成完
    短请求早已结束，但它的 slot 被"占着"不干活
    有效利用率 ≈ 平均长度 / 最大长度 = 255/500 ≈ 51%

**Continuous Batching（连续批处理 / iteration-level scheduling，Orca 提出，vLLM/TGI 采用）**： - **调度粒度从”请求级”细化到”迭代级（每生成一个 token 是一个调度点）“**。 - 每个 decode step 结束后，**已完成的请求立刻退出 batch，新到的请求立刻插入**。 - 效果：batch 总是”满的”，**没有头部阻塞（head-of-line blocking）**，吞吐可提升 **2~10 倍**（实测在混合长度负载下常提升 3~8 倍）。

    静态:  [R1======][R2==][R3========]    整个 batch 绑定生命周期
            \_______ 短请求白白等待 _______/

    连续:  step1: [R1 R2 R3]
           step2: [R1 R3 R4]   <- R2 完成退出，R4 补位
           step3: [R1 R3 R4 R5]

**工程实现要点**： 1. **两阶段调度**：每个 iteration 分为 prefill 队列（新请求、被抢占的请求重算）和 decode 队列。 2. **chunked prefill**：把长 prompt 切成若干 chunk，与 decode 混合调度，避免长 prefill 阻塞所有 decode 请求（**降低 TPOT 抖动**）。代价是单个长请求的 TTFT 略增。 3. **抢占（preemption）**：显存不足时，vLLM 会**驱逐（evict）**某些请求的 KV Cache，被驱逐的请求需要**重算（recompute）** 或 **swap 到 CPU**。vLLM 默认策略是 recompute（因为重算 prefill 比 PCIe 传输快）。 4. **公平性**：要防止超长请求饿死短请求，常用 **FCFS + 优先级队列**。

**追问：vLLM 的 PagedAttention 解决了什么？** 解决 **KV Cache 的内存碎片**。传统实现为每个请求预留 `max_seq_len` 的连续显存，而实际使用长度远小于上限：

    传统: 浪费率 = 1 - 实际平均长度/预留最大长度  → 常达 60~80%

PagedAttention 借鉴 OS 虚拟内存分页：把 KV Cache 切成固定大小的 **block（如 16 个 token）**，用 **block table（页表）** 映射逻辑位置到物理块。收益： - **碎片率降到 \<4%**； - **支持前缀共享（prefix sharing）**：多个请求共享同一 system prompt 的 KV block（**copy-on-write**），对 few-shot / 多轮对话场景吞吐提升巨大； - 支持 beam search 的分支共享。

**核心数字**：PagedAttention 让 vLLM 在相同硬件上吞吐比 HuggingFace `generate()` 高 **14~24 倍**（论文数据），其中大部分来自 batching + 分页。

------------------------------------------------------------------------

## Q18：PD 分离（Prefill-Decode Disaggregation）架构讲一讲，为什么要做？挑战是什么？

答：**动机**：Prefill 和 Decode 是两种性质完全不同的负载（前面已述：compute-bound vs memory-bound），放在同一个 GPU 上会**互相干扰**： 1. **资源竞争**：一个 2000 token 的 prefill 会占用 GPU 数秒，期间所有 decode 请求的 TPOT 出现尖刺（“**prefill 抖动**”）； 2. **最优并行策略不同**：prefill 适合大 TP（计算密集，切分收益高），decode 适合小 TP + 大 batch（通信敏感）； 3. **最优硬件配比不同**：prefill 需要算力，decode 需要显存带宽和容量，同一张卡无法同时最优。

**架构**：

                        ┌──────────────────────────────┐
       Client ──► API Gateway / Router (SLO 感知调度)
                        └───────────┬──────────────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
            ┌─────────────────┐          ┌─────────────────┐
            │  Prefill 集群    │  KV传输  │   Decode 集群    │
            │  (计算密集)      │ ───────► │  (访存密集)      │
            │  大 TP / FP8     │  高带宽  │  小 TP / 大batch │
            │  可弹性扩缩      │  低延迟  │  KV Cache 存储   │
            └─────────────────┘          └─────────────────┘
                                              │
                                        ┌─────┴──────┐
                                        │ KV Store / │
                                        │ 分布式缓存  │
                                        └────────────┘

**关键设计点**： 1. **KV Cache 传输**：prefill 结束后要把 KV Cache 从 prefill 节点传到 decode 节点。传输量公式：

    KV_bytes_per_token = 2 * L * n_kv_heads * d_head * bytes_per_elem
                       （2 是因为 K 和 V）

例：Llama-3-70B（L=80, n_kv_heads=8, d_head=128, FP16），`2 * 80 * 8 * 128 * 2 = 327,680 bytes ≈ 320 KB/token`。 一个 2000 token 的 prompt → **640 MB** 需传输！用 IB NDR（50 GB/s）需 **~13 ms**，加上 decode 侧接收，是可接受的；但若走 PCIe/以太网就完全不可行。**所以 PD 分离强依赖 RDMA/高带宽网络**。 2. **缓存复用**：用 **prefix caching** 减少重复计算的 prefill；用 KV Cache 的层次化存储（GPU HBM → CPU DRAM → NVMe / 分布式 KV Store）做**跨请求、跨节点复用**。 3. **调度策略**：Router 要感知两侧负载，做 **SLO 感知路由**（如 DistServe 的 goodput 优化：对 TTFT 敏感的路由到大 prefill 池，对 TPOT 敏感的给稳定 decode）。 4. **资源配比**：prefill:decode 的 GPU 数量比要按实际负载的 **prompt:output token 比** 动态调整。典型比值 1:2 ~ 1:8（输出长的场景 decode 占多数）。

**挑战（面试要点）**： - **KV 传输的延迟与带宽**成为新的关键路径，网络抖动直接影响 TTFT； - **调度复杂度**大幅上升（两套调度器 + 全局路由 + 缓存亲和性）； - **弹性伸缩更复杂**（两侧要独立扩缩容）； - **故障域变大**（prefill 挂了，decode 侧的 KV 变孤儿）； - **小模型/短请求下收益为负**（传输开销 \> 干扰节省）。实践中 **只有在 prompt 长、并发高、模型大、且网络好时才值得上 PD 分离**。

**追问：什么是 Chunked Prefill？它和 PD 分离的关系？** Chunked prefill 在**单实例内**把长 prompt 切成 chunk，与 decode 混合调度，缓解 prefill 对 decode 的阻塞 —— 是 PD 分离的**轻量替代方案**（不用改架构即可获得部分收益）。局限是它无法解决”两侧最优并行策略/硬件配比不同”的问题。**工程选择**：中小规模先上 chunked prefill；大规模、SLO 严苛再上 PD 分离。

------------------------------------------------------------------------

## Q19：多副本部署与负载均衡怎么做？KV Cache 怎么分布式存储与传输？

答：**多副本 + 负载均衡的基本架构**：

    LB (一致性哈希/最少连接/加权)
       │
       ├─► Model Replica 1 (多卡 TP)
       ├─► Model Replica 2
       └─► ...

**负载均衡的挑战（LLM 特有）**： 1. **请求成本不均**：一个请求可能 10 token 输出，也可能 10000 token。**按请求数轮询（round-robin）等于随机分配负载**。 2. **会话粘性（session affinity）**：多轮对话中，同一 session 的请求最好落到同一副本，以**复用 KV Cache**（否则每轮都要重算历史）。 3. **KV Cache 亲和性**：若副本 A 已缓存了某 system prompt 的 KV，新请求路由到 A 能显著降低 TTFT。

**解决手段（按优先级）**： - **一致性哈希 + 会话 ID 作为 key**：保证同 session 落到同副本； - **负载感知调度**：LB 要收集各副本的实时负载指标（**running/waiting 请求数、KV Cache 使用率、GPU 利用率**），而非均匀分发。vLLM 支持按 waiting 队列长度路由； - **前缀感知路由（prefix-aware routing）**：如 SGLang 的 RadixAttention / Router，按**最长公共前缀**路由到已缓存的副本； - **最少批（least-batch）策略**：让所有副本的 batch 大小趋于一致，最大化总吞吐。

**KV Cache 的分布式管理**：

1.  **单副本内的分层**：

<!-- -->

    GPU HBM (最快，容量小)
      ↓ 溢出
    CPU DRAM (PCIe 传输，容量大)
      ↓ 溢出
    本地 NVMe SSD (容量巨大，延迟高)
      ↓
    分布式 KV Store (跨节点复用)

2.  **跨副本复用（KV Cache 池化）**：

- **动机**：企业场景中大量请求共享同一 system prompt / 知识库文档。若能跨副本共享 KV，可省掉重复 prefill。
- **实现**：把 KV Cache 的 block（PagedAttention 的粒度）作为**可寻址对象**，用 **内容哈希 / 前缀哈希** 做 key，存入分布式缓存（如 Redis + RDMA、Mooncake 的 Transfer Engine、LMCache）。
- **传输优化**：用 **RDMA one-sided write**（接收方无需 CPU 参与）、**GPUDirect RDMA**（网卡直接读写 GPU 显存）、以及**分层预取**（预测下一步会用到哪个 block，提前拉取）。

3.  **关键数字**：

- LLM 的 KV Cache 大小（前面公式）：Llama-3-70B 用 GQA（n_kv_heads=8）后约 **320 KB/token**；若用 MHA（n_kv_heads=64）则 **2.5 MB/token** —— **GQA 把 KV Cache 缩小了 8 倍**，这是 MQA/GQA 在推理侧的核心商业价值（省显存 = 能放更大 batch = 更高吞吐）。
- 一个 8K 上下文的请求，70B MHA 模型的 KV = 20 GB —— **几乎吃掉半张卡**。所以要上 GQA + 分层存储。

**追问：KV Cache 量化（KV8/KV4）值得做吗？** 通常值得，但要注意： - **收益**：KV 显存减半/减四分之三 → 能支持更大 batch 或更长上下文 → 吞吐提升；同时减少 KV 的带宽读取（decode 是 memory-bound）→ 直接降 TPOT。 - **代价**：KV 的分布有异常值（outlier channels），朴素 per-tensor INT8 会导致明显精度损失。**解法**：per-channel / per-token 量化、**KIVI** 的 per-channel key + per-token value、或对 outlier channel 保留高精度（SmoothQuant 式通道均衡）。 - **经验**：INT8 KV 通常**几乎无损**；INT4 KV 需要仔细调，且在长上下文、检索类任务上损失更明显，建议按业务评测集回归。

------------------------------------------------------------------------

## Q20：边缘 / 端侧部署大模型的工程要点？

答：边缘场景的约束是 **算力小、显存小、功耗严、无稳定网络**，工程路径与云端完全不同：

**1) 模型压缩（必做）** - **量化**：INT4（AWQ / GPTQ / GGUF Q4_K_M）是端侧主流；权重 + 激活都量化（W4A16 常见，即权重 4bit、激活 16bit）。 - **剪枝**：结构化剪枝（去掉整个 head/channel）才能真正加速；非结构化稀疏在通用硬件上难加速。 - **蒸馏**：用小模型（1B~7B）蒸馏大模型能力，端侧通常直接选小模型（Qwen-1.8B、Phi-3-mini、Gemma-2B）。 - **稀疏化 / 早退**：MoE 端侧版本、layer skipping。

**2) 运行时与推理引擎** - 手机：**MNN / NCNN / TFLite / llama.cpp（移动端）/ ExecuTorch**；高通平台用 **QNN / SNPE** 走 NPU；苹果用 **Core ML / ANE**。 - PC/边缘盒子：**llama.cpp (GGUF) / Ollama / TensorRT-LLM / MLX（Apple Silicon）**。 - 关键：**必须用厂商 NPU 的量化格式**，通用 CUDA 生态在端侧不存在。

**3) 显存/内存优化** - 端侧内存是**共享**的（CPU/GPU 共用），KV Cache 的占用是长上下文的主要矛盾。**INT8/INT4 KV Cache** 必做。 - 使用 **滑窗注意力（sliding window attention）** 或 **RoPE scaling + 分块** 限制 KV 增长。 - **前缀缓存持久化**：把 system prompt 的 KV 存到磁盘，启动时加载，避免每次冷启动重算。

**4) 系统架构** - **端云协同**：简单请求端侧处理（省流量、低延迟、保隐私），复杂请求上云（fallback）。路由依据：输入长度、置信度、模型能力边界。 - **首包优化**：端侧 TTFT 敏感，用**投机解码（speculative decoding）** 小模型 draft + 大模型 verify，或 **Medusa / lookahead decoding**。 - **内存受限下的流式**：分页/分块的权重加载（mmap），避免一次性全部载入。

**追问：端侧 7B 模型 INT4 大概多少内存？** - 权重：`7e9 * 0.5 bytes = 3.5 GB`（INT4）。 - KV Cache（4K 上下文，GQA，INT8）：约 `2 * ``32层`` * 8 heads * 128 * 4096 * 1 byte ≈ 268 MB`。 - 运行时开销（激活、临时 buffer）：几百 MB。 - **总计约 4~5 GB** → 主流旗舰手机（8~12GB RAM）可用，但需注意与其他 App 的内存争抢；中低端机（4~6GB）基本放不下，需降到 1.8B~3B 模型。

------------------------------------------------------------------------

# 五、数据与存储

## Q21：训练数据管道怎么设计？数据加载为什么成为瓶颈？怎么优化？

答：**数据管道的三个环节**：

    原始数据 (对象存储/S3/OSS)
       ↓ 清洗/去重/质量过滤 (离线, Spark/Ray/自研)
    Tokenized 数据集 (二进制分片, 如 .bin + .idx / parquet / webdataset tar)
       ↓ 采样与打包 (packing, shuffle, curriculum)
    GPU 训练 (需要持续供数)

**为什么数据加载会成为瓶颈**： 1. **单步消费的 token 量极大**：global batch 4M tokens，若 8 秒/step，需要 **500K tokens/s 的供给速率**。 2. **CPU 预处理成为串行瓶颈**：tokenize、拼接、pad、mask 生成都在 CPU 上，Python GIL 和 DataLoader worker 的效率直接决定吞吐。 3. **随机读取导致 IO 抖动**：shuffle 后是随机读，若有 10 万个 100MB 分片，机械盘/网络存储的随机读性能崩掉。 4. **存储带宽**：4M token/step × 4 bytes/token(INT32) ≈ 16 MB/step，看似不大，但**多节点同时读 + 随机访问 + 元数据开销**会放大成百倍压力。

**优化手段**： 1. **预 tokenize 并落盘为二进制分片**：训练时只做”读 + 拼接”，不做 tokenize。分片大小取**几百 MB**（与文件系统块/顺序读匹配，且利于并行）。 2. **顺序读 + 全局 shuffle**：把数据预打散写入分片内（**片内 shuffle，片间顺序**），或维护”**随机种子 + 索引表**”来做逻辑 shuffle，物理读仍是顺序的。 3. **多 worker + prefetch**：PyTorch DataLoader `num_workers` 调大、`prefetch_factor`、`pin_memory=True`（用 pinned memory 让 H2D 拷贝异步化）。 4. **本地缓存（预热）**：把当前 epoch 用到的分片**提前拷到计算节点本地 NVMe**，训练时读本地。这是千卡训练的标准做法（“**数据本地化**”）。 5. **流式格式**：**WebDataset（tar 分片）/ Mosaic MDS / Megatron 的 indexed dataset**，支持流式顺序读 + 随机访问结合。 6. **样本打包（sequence packing）**：把多条短样本拼进一个 `max_seq_len` 的序列，用 attention mask（或 varlen attention）隔开。**消除 padding 浪费**，官方报告可提升训练吞吐 **20%~100%**（取决于原始长度分布）。

**追问：变长序列怎么处理最省算力？** 三种策略对比： - **Pad 到最大长度**：最简单，但短样本浪费算力。长度分布方差大时浪费可达数倍。 - **Pad 到 batch 内最大长度（dynamic batching）**：按长度排序分桶，同桶内 pad。改善明显但仍有浪费。 - **Packing（推荐）**：拼接到 `max_seq_len` 无 padding。**代价**：需要正确的 attention mask（不能让样本 A 看到样本 B），要么用 block-diagonal mask（效率略低），要么用 **varlen / flash-attn 的** `cu_seqlens`（效率最高）。这是当前大模型训练的标准配置。

------------------------------------------------------------------------

## Q22：大规模数据存储与 IO 优化、Checkpoint 存储、向量存储怎么选？

答：分三类存储讲。

**1) 训练数据存储** \| 方案 \| 特点 \| 适用 \| \|—\|—\|—\| \| 对象存储 (S3/OSS/MinIO) \| 便宜、无限扩容、高吞吐顺序读，但**延迟高、元数据操作贵** \| 数据湖，冷数据 \| \| 并行文件系统 (Lustre/GPFS) \| 高吞吐、POSIX 语义、支持小文件，**成本高、运维复杂** \| 大规模训练集群标配 \| \| JuiceFS / Alluxio \| **基于对象存储做缓存层**，POSIX 接口 + 本地缓存加速 \| 云上性价比方案 \| \| 本地 NVMe \| 最快，容量小 \| 热数据缓存、checkpoint 热存 \|

**优化原则**：**大文件顺序读 + 本地缓存 + 分层**。“**能顺序读就不要随机读，能读本地就不要读远端**”。

**2) Checkpoint 存储** - **写入量**：18N bytes（70B → 1260 GB/次）。若 1024 卡且每卡分片，单卡只需写 ~1.2 GB，**并行写**是关键（不能 rank0 汇聚）。 - **写路径**：GPU → (异步拷) → CPU pinned → 本地 NVMe（热存）→ 后台异步同步到远端对象存储（冷存）。 - **保留策略**：`keep_last_n=3` + 关键里程碑永久保留。**不要无限保留**，否则存储成本爆炸且写盘变慢。 - **元数据**：分片 checkpoint 需要一份元数据（哪些 rank 存了哪部分），恢复时据此重组。Megatron 的 distributed checkpoint 会在目录下放 `metadata.json`。

**3) 向量存储（给 RAG / 特征检索用）** \| 方案 \| 特点 \| 适用 \| \|—\|—\|—\| \| FAISS \| 库，非服务，GPU 加速强，需自管持久化 \| 离线/单机大规模检索 \| \| Milvus / Qdrant / Weaviate \| 分布式向量数据库，支持标量过滤 + 向量混合查询、水平扩展 \| 生产 RAG \| \| pgvector \| PostgreSQL 扩展，与业务库同源，事务一致 \| 中小规模、已有 PG \| \| Redis / Valkey (向量模块) \| 内存，极低延迟 \| 在线特征、实时推荐 \| \| HNSW vs IVF vs DiskANN \| 图索引精度高但内存大 / 倒排省内存 / **DiskANN 磁盘索引省内存** \| 按内存预算选 \|

**选型要点**： - **规模**：\<1000 万向量，单机 FAISS/HNSW 足够；\>1 亿，需要分片 + 分布式（Milvus）。 - **内存**：HNSW 的内存占用 ≈ `N * (M * 2 * 4 + dim * 4)` bytes（M 为图度数），1 亿条 768 维 FP32 ≈ 300 GB+。**降内存手段**：PQ 量化（IVF-PQ）、DiskANN、INT8 量化。 - **过滤**：RAG 场景常要”先按 tenant_id 过滤再检索”，**标量过滤 + 向量检索的混合查询**能力很关键（Milvus 的 partition、Qdrant 的 payload index 是在解决这个）。 - **一致性**：向量索引的增量更新有代价（HNSW 删改需重建或标记删除），**高频更新场景要考虑定时重建 + 双写**。

------------------------------------------------------------------------

# 六、平台与调度

## Q23：K8s 上怎么调度 GPU？GPU 虚拟化和切分有哪些方案？

答：**基础层：Device Plugin** - K8s 通过 **NVIDIA Device Plugin**（DaemonSet）把 GPU 作为 **extended resource**（`nvidia.com/gpu: 1`）暴露给调度器。 - `kubelet` 调用 device plugin 的 `Allocate()`，插件返回设备 ID，容器启动时通过 `NVIDIA_VISIBLE_DEVICES` 注入。 - 局限：**只能整卡分配，不感知拓扑（NVLink/NVSwitch 亲和性）、不感知显存**。

**调度增强**： - **拓扑感知调度**：需要把同一 NVLink 域（同机 8 卡）的任务放一起，否则 TP 通信跨机。方案：**Topology Aware Scheduler（TAS，Volcano 内置）**、或自研 scheduler extender。 - **Gang Scheduling（成组调度）**：分布式训练要求”**所有 worker 同时启动**”，否则已启动的 worker 会空等甚至超时。K8s 默认调度器不支持 → 用 **Volcano / Kueue / Yunikorn**。 - **队列与优先级**：多团队共享集群需要 quota、priority class、抢占（preemption）、fair-share。Volcano 的 **queue + podgroup + proportion plugin** 是标准方案。

**GPU 虚拟化 / 切分方案对比**：

| 方案 | 原理 | 隔离性 | 显存切分 | 算力切分 | 适用 |
|----|----|----|----|----|----|
| **MIG** (Multi-Instance GPU) | 硬件级切分，A100/H100 支持，最多 7 实例 | **强**（显存、L2、带宽都隔离） | 是（按 profile，如 3g.20gb） | 是 | 多租户推理、小模型独立服务 |
| **时间片（time-slicing）** | 多个进程分时复用同一 GPU | **无**（显存共享，会 OOM 互踩） | 否 | 分时 | 开发调试、低负载共享 |
| **MPS** (Multi-Process Service) | 多进程共享同一 context，并发执行 kernel | 弱（可配 SM 上限和显存） | 部分 | 可限 SM% | 小任务并发、提升利用率 |
| **vGPU** (NVIDIA vGPU / 阿里 cGPU) | 驱动层虚拟化，把卡切给多个容器 | 中 | 是 | 是 | 云厂商多租户售卖 |
| **软件切分** (HAMi / 腾讯 qGPU) | 拦截 CUDA API，限制显存/SM | 中 | 是 | 是 | K8s 内细粒度共享 |

**选型结论**： - **训练** → **必须整卡 + 拓扑亲和 + gang scheduling**，绝不用 MIG/时间片（会严重拖慢且不稳定）。 - **推理多租户** → **MIG**（强隔离、可预测）或 **HAMi 式软切分**（灵活、成本低）。 - **开发机/实验** → 时间片或 MPS。

**追问：MIG 有什么坑？** 1. **profile 固定**：切分比例是预设的（如 1g.5gb、2g.10gb、3g.20gb、7g.80gb），不能任意指定显存数； 2. **重新配置需要排空**：改 MIG 配置要停止该 GPU 上所有任务； 3. **算力按比例切**：3g 实例只有约 3/7 的 SM，稍大模型就慢； 4. **不支持 NVLink P2P 跨实例**：MIG 实例之间不能做 P2P，**无法做 TP**； 5. **不是所有卡都支持**：消费卡（4090）不支持 MIG。

------------------------------------------------------------------------

## Q24：集群资源利用率怎么优化？任务队列与优先级、故障自愈怎么设计？

答：**利用率是 AI 平台的核心 KPI**。典型问题：**训练任务占着卡但 GPU 利用率只有 30%（数据加载瓶颈 / 通信等待），而排队任务拿不到卡**。

**提利用率的手段**： 1. **提高单任务 MFU**：这是根本。profiling 定位瓶颈（见第八章）。利用率低时优先怀疑 data loader 和通信。 2. **混部（co-location）**：把**推理任务和训练任务混部** —— 训练对延迟不敏感、推理对吞吐敏感，可错峰互补。风险是资源争抢，需要 cgroup / MPS / 显存隔离。 3. **超卖（oversubscription）**：允许提交 \> 物理卡的资源请求，靠”实际不会同时用满”的统计规律。**风险高**，需要严格监控 + 驱逐机制。 4. **分时复用**：白天推理、夜间训练（很多公司实际在做）。 5. **抢占 + 弹性**：高优任务可抢占低优任务的卡，低优任务被抢占后**优雅退出并保存 checkpoint**。 6. **打破”占坑”**：给任务设 **最大运行时长 + 自动回收闲置任务**（如 GPU 利用率连续 30 分钟 \< 5% 则告警/回收）。

**任务队列与优先级设计**：

    # 概念模型
    Queue:
      - name: high-priority-research   # 权重高，可抢占
        weight: 10
        reclaimable: false
      - name: batch-training
        weight: 3
        reclaimable: true              # 可被高优抢占
      - name: dev-sandbox
        weight: 1
        maxRuntime: 4h

- **调度算法**：FIFO + 优先级 + **公平份额（fair-share / DRF）** + 回填（backfill，小任务填补空档）。
- **Gang scheduling 必须**：否则分布式任务死锁。
- **配额（quota）**：按团队分配 GPU 配额，防止一个团队吃满整个集群。

**故障自愈（Self-Healing）设计**：

    ┌────────────────────────────────────────────────┐
    │  监控层: DCGM / Prometheus / 节点心跳            │
    │   采集: GPU 利用率、显存、温度、ECC、Xid、        │
    │         NVLink 误码、IB 链路状态、进程存活        │
    ├────────────────────────────────────────────────┤
    │  判定层: 规则 + 阈值                             │
    │   节点不健康 = Xid 错误 / ECC 不可纠错 /         │
    │              心跳丢失 / NCCL 超时 X 次           │
    ├────────────────────────────────────────────────┤
    │  动作层:                                        │
    │   1. cordon 节点 (禁止新任务调度)                │
    │   2. drain 或驱逐受影响的任务                    │
    │   3. 触发任务重启 (from latest checkpoint)      │
    │   4. 通知运维 + 自动工单                        │
    ├────────────────────────────────────────────────┤
    │  预防层:                                        │
    │   - 每日节点健康巡检 (跑 NCCL smoke test)       │
    │   - 上线前 burn-in 测试 (GPU 冒烟 24h)          │
    │   - 预留热备节点池                              │
    └────────────────────────────────────────────────┘

**关键点**：**重启要幂等且快**。训练脚本必须支持”任意机器上从 checkpoint 恢复”，rendezvous 用稳定地址（`--rdzv_endpoint` 指向 svc，而非固定 pod IP）。这是很多团队踩过的坑：pod 重建后 IP 变了，任务起不来。

------------------------------------------------------------------------

# 七、系统设计大题

## Q25（大题）：设计一个支持千卡规模的大模型训练平台

答：分五层作答。

### 1. 需求与假设

- 目标：支持 1B~10B 到 100B+ 参数模型的预训练与微调；
- 规模：单集群 1024 张 GPU（如 A100/H100 80GB），8 卡/节点 = 128 节点；
- 网络：节点内 NVSwitch（900 GB/s），节点间 IB NDR 400 Gb/s × 8 网卡（聚合约 400 GB/s per node）；
- 存储：并行文件系统（Lustre/GPFS）提供 ~1 TB/s 聚合读带宽 + 大容量对象存储。

### 2. 分层架构

    ┌───────────────────────────────────────────────────────────────┐
    │  用户层: CLI / Web / SDK  |  实验管理 (MLflow/W&B)             │
    ├───────────────────────────────────────────────────────────────┤
    │  调度层: Volcano / Kueue (队列·优先级·Gang·抢占·配额)           │
    │         拓扑感知调度 (NVLink 域亲和 · 机架感知)                  │
    ├───────────────────────────────────────────────────────────────┤
    │  编排层: K8s + GPU Operator + Device Plugin + DCGM 监控        │
    ├───────────────────────────────────────────────────────────────┤
    │  训练框架层:                                                    │
    │   Megatron-LM (TP+PP+DP) / DeepSpeed ZeRO / FSDP               │
    │   NCCL 通信库 (RDMA/GPUDirect, SHARP)                           │
    ├───────────────────────────────────────────────────────────────┤
    │  数据层: 数据管道 (离线清洗→tokenize→打包) | 并行FS + 本地NVMe  │
    │          Checkpoint 服务 (异步保存 + 分层存储)                  │
    ├───────────────────────────────────────────────────────────────┤
    │  可观测性: DCGM + Prometheus + Grafana | 训练日志/loss 看板      │
    │           故障自愈控制器 (Health Controller)                    │
    └───────────────────────────────────────────────────────────────┘

### 3. 关键技术选型与理由

| 环节 | 选型 | 理由 |
|----|----|----|
| 并行框架 | Megatron-LM（TP+PP+DP） | 千卡规模下通信效率优于纯 ZeRO-3；MFU 更高 |
| DP 状态切分 | ZeRO-1（Megatron-DeepSpeed） | 只切优化器状态，通信开销小，同时省显存 |
| 互联 | NVLink 节点内 + IB 节点间 | TP 限制在机内 8 卡，跨机用 PP/DP |
| 通信库 | NCCL + SHARP + GPUDirect RDMA | 交换机内归约，减少跨机流量 |
| 调度 | Volcano（Gang + 拓扑感知 + 抢占） | 分布式训练必须成组调度 |
| 存储 | Lustre 热 + 对象存储冷，节点本地 NVMe 缓存 | 数据本地化，避免远端随机读 |
| Checkpoint | 异步保存 + 本地 NVMe 热存 + 异步上云 | 把阻塞时间从分钟级降到秒级 |
| 容错 | DCGM 监控 + 自动 cordon/重启 + 幂等 rendezvous | MTBF 低，必须自动化 |

### 4. 并行策略的具体配置（示例）

    模型规模     总卡数   TP  PP  DP   ZeRO  micro-batch  global batch
    7B          64      1   1   64   1     8             2M tokens
    13B         128     2   1   64   1     4             4M tokens
    70B         512     8   4   16   1     2             4M tokens
    175B        1024    8   8   16   1     1             2M tokens

**PP 气泡验证（70B, PP=4）**：micro-batch 数 = `4M / (2048 * 16) = 122`，气泡 ≈ `(4-1)/(122+4-1) = 2.4%` —— 可接受。

### 5. 容量与成本估算

**网络带宽需求验证（70B, DP=16）**： - 梯度总量 S = 2N = 140 GB（BF16），ZeRO-1 下每卡 AllReduce 的梯度量 ≈ `2S/Nd = 2*140/16 = 17.5 GB`。 - 时间 = `17.5 GB / 50 GB/s ≈ 350 ms`（单 IB 口）。用 8 网卡聚合 + overlap，实际可压到单步的 10% 以内。

**存储需求**： - Checkpoint：70B × 18 bytes = 1.26 TB/次 × 保留 3 份；512 卡并行写 → 单卡仅写 2.5 GB。 - 训练数据：假设 5T tokens，INT32 存储 = 20 TB，加索引/副本约 60 TB。

**成本量级**：128 个 8 卡 H100 节点；按云上约 ¥25/卡/小时估算，**每小时 ¥25,600，一个月（720h）≈ ¥1,840 万**。这一数字解释了为什么**MFU 每提升 5% 就等于每月省近百万**，也解释了平台侧一切优化（调度、容错、数据管道）的经济动机。

### 6. 关键风险与对策

- **慢节点/坏节点**：定期 smoke test，检测到降速节点主动摘除；
- **通信热点**：机架感知调度，避免同一 AllReduce 组跨 spine；
- **数据饥饿**：data loader 必须有独立监控指标（`data_wait_time`），并设置预取水位告警；
- **存储打满**：checkpoint 保留策略 + 配额 + 写带宽限速；
- **任务抢占引发的 checkpoint 风暴**：错峰保存（给每个任务分配不同的 save 时间偏移）。

------------------------------------------------------------------------

## Q26（大题）：设计一个支撑百万 QPS 的大模型推理服务

答：先做**关键澄清**——“百万 QPS”对 LLM 是极高的要求，必须先明确口径。

### 1. 需求澄清与假设（面试第一步必须做）

- QPS 是**什么粒度**？LLM 的 QPS 通常指”请求数/s”，但更该看 **token 吞吐**。
- 假设：**百万 QPS 是轻量场景**（如分类/意图识别/短文本改写，输入 \< 50 token，输出 \< 20 token）；若是通用对话（输入 500、输出 300），百万 QPS 意味着 **3 亿 tokens/s 的输出**，这需要天文数字的 GPU，不现实。
- 因此本题按 **“平均输入 30 token、输出 10 token 的高并发轻量推理”** 设计，并给出通用对话场景的推算。

### 2. 容量估算（关键推导）

**单请求计算量**：

    Prefill: 2 * N * L_in  = 2 * 7e9 * 30  = 4.2e11 FLOPs
    Decode:  2 * N * L_out = 2 * 7e9 * 10  = 1.4e11 FLOPs
    合计 ≈ 5.6e11 FLOPs/请求

**GPU 需求（7B 模型, FP16, A100 312 TFLOPS, 假设 MFU 30%）**：

    单卡有效算力 = 312e12 * 0.3 = 9.36e13 FLOPs
    单卡支撑 QPS = 9.36e13 / 5.6e11 ≈ 167 QPS
    百万 QPS 需要 = 1e6 / 167 ≈ 6000 张 A100

**显存侧验证（Decode 是 memory-bound）**：

    单 token decode 时间下界 = 28 GB / 2039 GB/s = 13.7 ms
    批量 = 100 时，吞吐 = 100 / 13.7ms ≈ 7300 tokens/s
    batch 256 → 256 * 10 输出 token / (13.7ms * 10) ≈ 1868 请求/s（每请求 10 token）

取整：**单卡约 1500 QPS（该场景）** → 百万 QPS 需 **~700 张 A100**（算力侧给的 6000 张是无 batching 的粗算；实际 batching 后 memory-bound 场景的算力并非瓶颈，**真正的约束是显存容量和带宽**）。

**核心结论：LLM 推理的容量瓶颈不在 FLOPS，而在显存带宽 + 显存容量（决定 batch 上限）**。这个洞察是本题的得分点。

### 3. 分层架构

    ┌──────────────────────────────────────────────────────────────┐
    │  L0 接入层: DNS/GSLB → LVS/四层LB → Nginx/Envoy (七层)        │
    │      · 多地域接入  · TLS 卸载  · 限流(令牌桶)  · 鉴权          │
    ├──────────────────────────────────────────────────────────────┤
    │  L1 路由层: 智能路由                                            │
    │      · 会话粘性(一致性哈希 by session_id)                       │
    │      · 前缀感知路由(prefix cache 亲和)                          │
    │      · 负载感知(按 waiting 队列长度/GPU 利用率)                  │
    ├──────────────────────────────────────────────────────────────┤
    │  L2 缓存层:                                                    │
    │      · 精确缓存(相同 query → 相同 answer, Redis, 命中 10~40%)   │
    │      · 语义缓存(向量相似度, 可选)                               │
    │      · Prompt/KV Cache 复用(system prompt 共享)                │
    ├──────────────────────────────────────────────────────────────┤
    │  L3 推理层:   [Prefill 池]  ──KV(RDMA)──►  [Decode 池]         │
    │      · 每池多副本, 副本内 TP=8(NVLink)                          │
    │      · 连续批处理 + chunked prefill + PagedAttention           │
    │      · 量化: W8A8 / FP8;  KV Cache INT8                       │
    │      · 弹性伸缩: 按队列长度 HPA                                │
    ├──────────────────────────────────────────────────────────────┤
    │  L4 存储层: KV Cache 分层(GPU→CPU→NVMe→分布式KV) | 模型仓库     │
    ├──────────────────────────────────────────────────────────────┤
    │  L5 可观测: QPS/TTFT/TPOT/Goodput | GPU 利用率 | 队列深度告警   │
    └──────────────────────────────────────────────────────────────┘

### 4. 关键技术选型与理由

| 问题 | 方案 | 理由 |
|----|----|----|
| 高并发下的批效率 | **连续批处理（vLLM/TGI）** | 吞吐比静态 batch 高 3~8x |
| 显存碎片 | **PagedAttention** | 碎片率 \<4%，支持前缀共享 |
| Prefill 干扰 Decode | **chunked prefill**，或 **PD 分离** | 降低 TPOT 抖动；PD 分离可独立扩缩 |
| 显存带宽瓶颈 | **W8A8 量化 + KV INT8** | 权重减半，decode 直接提速 |
| 负载不均 | **负载感知 + 会话哈希路由** | 避免长请求集中到单副本 |
| 重复请求 | **精确 + 语义缓存** | 高重复场景命中率可达 30%+ |
| 长尾延迟 | **按 SLO 分级 + 超时降级** | 保证 P99 |

### 5. 资源规划

    假设 7B 模型、平均 30 in / 10 out、目标 100 万 QPS：
    - 缓存命中 30% → 实际到达推理层 70 万 QPS
    - 单卡（A100 80G, FP16, batch 256）约 1500 QPS → 需 ~470 张推理卡
    - 冗余 30%（可用性 + 峰值）→ ~600 张
    - 加 Prefill 池（若 PD 分离, 按 1:3 配比）→ 总计 ~800 张 A100
    - 按云价 ¥12/卡/小时（推理卡）→ ¥9,600/h → 月约 ¥690 万

**成本优化杠杆**（按收益排序）：① 缓存命中率（直接线性省卡）；② 量化（吞吐 ×2，成本减半）；③ 路由与批处理调优（提升 GPU 利用率）；④ 混部/错峰。

### 6. 关于”通用对话百万 QPS”的现实结论

若输入 500 / 输出 300：

    单请求 FLOPs = 2 * 7e9 * (500+300) = 1.12e13
    单卡(30% MFU) QPS = 9.36e13 / 1.12e13 ≈ 8.4 QPS
    百万 QPS 需 ≈ 12 万张 A100

**这在工程上不可行** → 面试中必须指出：**百万 QPS 的通用对话必须通过”模型分级（小模型兜底 + 大模型升级）+ 缓存 + 请求合并”来降本**，而不是硬堆卡。能主动做这个判断是加分项。

------------------------------------------------------------------------

## Q27（大题）：设计一套大模型训练数据的清洗与去重流水线

答：数据质量是模型效果的第一性因素（业界共识是数据质量比数据量更重要，高质量的 1T token 常胜过脏的 10T）。

### 1. 流水线总体架构

    ┌──────────────────────────────────────────────────────────────┐
    │ Stage 0  采集: Web爬取 / 书籍 / 代码仓库 / 论文 / 私有语料      │
    │          → 原始快照落对象存储 (WARC/HTML/PDF/JSONL)            │
    ├──────────────────────────────────────────────────────────────┤
    │ Stage 1  解析与抽取 (Extraction)                               │
    │    HTML→文本(trafilatura/readability) | PDF→文本 | 代码→AST    │
    │    → 统一 JSONL {id, text, meta(url, ts, lang, source)}        │
    ├──────────────────────────────────────────────────────────────┤
    │ Stage 2  粗过滤 (Heuristic Filtering)                          │
    │    · 长度: 太短(<50词)/太长截断  · 字符重复率 > 0.3 丢弃        │
    │    · 词重复率 (gopher rules)     · 特殊符号/乱码比例            │
    │    · 停用词比例、是否含大量数字/模板文本                        │
    │    · 语言识别 (fastText/CLD3) → 只保留目标语言                 │
    ├──────────────────────────────────────────────────────────────┤
    │ Stage 3  去重 (Dedup) —— 本设计的重点                           │
    │    L1 精确去重: 全文 SHA256 哈希 (段级 + 文档级)                │
    │    L2 近重复:   MinHash + LSH (文档级)                          │
    │    L3 子串去重: Suffix Array / Suffix Automaton (跨文档)        │
    │    L4 语义去重: 向量聚类，去除语义冗余 (可选)                    │
    ├──────────────────────────────────────────────────────────────┤
    │ Stage 4  质量模型打分 (Quality Classifier)                     │
    │    · 训练分类器 (fastText/BERT) 区分"高质量"vs"低质量"          │
    │      (正例: 维基/书籍/论文; 负例: 垃圾/广告页)                  │
    │    · 困惑度过滤: 用参考模型算 PPL，异常高/低都丢                │
    │    · 毒性/隐私/合规过滤 (PII 检测、正则 + NER)                  │
    ├──────────────────────────────────────────────────────────────┤
    │ Stage 5  领域配比与混合 (Mixing)                               │
    │    · 按目标配比 (如 网页 60% / 代码 20% / 书籍 10% / 论文 10%)  │
    │    · 温度采样 / 数据课程 (先易后难 or 先难后易)                 │
    ├──────────────────────────────────────────────────────────────┤
    │ Stage 6  Tokenize 与打包 → 训练格式                            │
    │    · 分片写 .bin/.idx (Megatron) 或 parquet/webdataset         │
    │    · 定长打包 + attention mask + 全局 shuffle 索引              │
    └──────────────────────────────────────────────────────────────┘

### 2. 去重的技术细节（面试重点，给数字）

**为什么必须去重**： 1. 重复数据让模型**记忆而非泛化**（生成时吐出训练集原文的”复读”现象）； 2. 重复样本在训练中被**多次加权**，等效于改变了数据分布； 3. 浪费算力：研究表明 CommonCrawl 中近重复率极高，**去重可减少 20%~40% 的数据量而不损效果**。

**各级去重的方法与代价**：

| 级别 | 算法 | 复杂度 | 召回 | 精度 |
|----|----|----|----|----|
| L1 精确 | SHA256 哈希比对 | O(N)，可用 set | 仅完全相同 | 100% |
| L2 近重复 | **MinHash + LSH** | O(N·k) 签名 + 分桶 | 高（Jaccard 相似） | 可调 |
| L3 子串 | Suffix Array + 后缀自动机 | O(N log N) 或 O(N) | **能发现”文档 A 是 B 的一段”** | 精确 |
| L4 语义 | Embedding + 聚类/ANN | 贵（需过模型） | 语义级 | 依赖模型 |

**MinHash + LSH 具体做法（要能讲清）**：

    1. 把文档切成 n-gram (如 token 5-gram) 集合
    2. 用 k 个哈希函数 (如 k=128) 对每个 n-gram 哈希，取每支哈希的最小值
       → 得到该文档的 MinHash 签名 sig = [h1_min, h2_min, ..., hk_min]
       性质: P(h_i(A) == h_i(B)) = Jaccard(A, B)
    3. LSH: 把 k 个签名分成 b 个 band，每 band r 个 (k = b*r)
       两文档只要有一个 band 完全相同 → 候选对
       → 候选对的召回概率 = 1 - (1 - s^r)^b     (s 为真实 Jaccard 相似度)
    4. 对候选对做精确 Jaccard 校验，超过阈值(如 0.8)则去重

**参数选择例**：k=128, b=16, r=8。当 s=0.8 时召回 = `1-(1-0.8^8)^16 = 1-(1-0.168)^16 ≈ 94.7%`。调 b/r 就是在召回和计算量之间权衡。

**L3 子串去重（Suffix Array）为什么重要**： MinHash 是”整篇相似”，但很多污染是”**文档 A 完整包含在文档 B 里**”（如转载、拼接）。做法：把所有文档拼成一个大串（中间插分隔符），**先对段落做精确去重**（段落级 SHA256），再对段落建 **后缀数组（Suffix Array）**，扫描找到长度 \> 50 token 的重复子串并标记 → 删除出现在**较晚文档**中的重复段落（保留最早的，通常是权威源）。这是 **CCNet / RefinedWeb** 的核心做法之一。

### 3. 工程实现与性能

- **计算框架**：Spark / Ray Data / Dask（分布式批处理），或自研基于 Arrow 的流水线。
- **吞吐目标**：处理 10T token 原始数据（约 40 TB 文本）。以 100 节点 × 64 核 = 6400 核估算：

<!-- -->

    假设每核处理 2 MB/s 文本（含解析+过滤），6400 核 → 12.8 GB/s
    40 TB / 12.8 GB/s ≈ 3125 s ≈ 52 分钟（仅 Stage 1-2）
    去重（MinHash+LSH）通常慢 3~10 倍 → 实际全流程 1~3 天

- **去重的内存挑战**：全局去重需要把所有哈希/签名放内存。10 亿文档 × 128 个 4 字节签名 = **512 GB** → 必须分桶（按哈希前缀分片到多机）或外部排序。
- **增量更新**：新数据到来时不能全量重算 → 维护一份**全局哈希索引**（命中即丢）+ 对新增数据做增量 LSH。

### 4. 质量评估与迭代

- **上游指标**：去重率、过滤率、各来源占比、平均文档长度、语言分布。
- **下游验证**：**唯一可信的评估是”用清洗后的数据训一个小模型，看下游 benchmark”**。常用做法：固定 1B/3B 模型 + 固定 token 数，对比不同数据配比的效果（ablation）。
- **数据溯源**：每条数据保留来源 URL/时间/处理版本，便于问题回溯和合规审计。

------------------------------------------------------------------------

## Q28（大题）：企业知识问答系统的端到端资源规划与成本估算

答：（RAG 算法链路细节见《RAG 检索增强生成 面试专题》，本题侧重**系统与成本**。）

### 1. 业务需求

- 场景：企业内部知识问答，覆盖 10 万份文档（PDF/Word/Confluence）；
- 用户：5000 内部员工，日均活跃 1000 人，**人均 20 次提问** → **2 万次/天**；
- 峰值：工作时段 10 小时集中，峰值系数 5 → **峰值约 3 QPS**（很低，但要求低延迟）；
- SLO：TTFT \< 1s，端到端 \< 5s，可用性 99.5%；
- 约束：**数据不出内网**，需私有化部署。

### 2. 系统架构

    ┌────────────── 前端: Web / 企微 / 钉钉机器人 ─────────────────┐
    ├──────────────────────────────────────────────────────────────┤
    │  接入层: Nginx / APISIX  (鉴权 SSO · 限流 · 审计日志)          │
    ├──────────────────────────────────────────────────────────────┤
    │  应用层: RAG Orchestrator                                     │
    │    查询理解 → 检索 → Rerank → Prompt 组装 → 生成 → 引用回填    │
    ├──────────────────────────────────────────────────────────────┤
    │  模型层:                                                       │
    │    LLM: Qwen-32B / DeepSeek-32B (vLLM, TP=4, 私有化)          │
    │    Embedding: BGE-M3 (0.6B, 单卡)                             │
    │    Rerank: bge-reranker-v2-m3 (单卡)                          │
    ├──────────────────────────────────────────────────────────────┤
    │  检索层: Milvus (向量) + Elasticsearch (BM25) 混合检索         │
    ├──────────────────────────────────────────────────────────────┤
    │  存储层: 对象存储(原始文档) + PostgreSQL(元数据/会话/权限)      │
    ├──────────────────────────────────────────────────────────────┤
    │  离线管道: 文档解析 → 分块 → Embedding → 入库 (每日增量)        │
    └──────────────────────────────────────────────────────────────┘

### 3. 关键参数设计

| 环节 | 设计 | 理由 |
|----|----|----|
| 分块 | 512 token，overlap 64 | 平衡召回粒度与上下文完整性 |
| 检索 | 向量 top-50 + BM25 top-50 → 融合取 top-20 | 混合检索召回率显著优于单路 |
| Rerank | top-20 → 取 top-5 入 Prompt | 精度提升最明显的单点 |
| 上下文 | 5 chunk × 512 ≈ 2560 token + system 300 | 控制 prefill 成本 |
| 输出 | 平均 300 token | — |

### 4. 资源规划（关键计算）

**A. 离线索引构建**

    10 万文档 × 平均 20 页 ≈ 200 万页
    每页 ~500 词 ≈ 700 token → 总 token ≈ 14 亿 token
    分块后 chunk 数 ≈ 14亿 / 512 ≈ 273 万个 chunk
    Embedding: BGE-M3 处理 273 万 chunk × 512 token
      单卡 A100 吞吐约 1000 chunk/s → 273万/1000 = 2730 s ≈ 46 分钟
      （可用 2 卡并行 → ~25 分钟）
    向量存储: 273万 × 1024 维 × 4 bytes = 11.2 GB
      加 HNSW 索引开销(×1.5) ≈ 17 GB → 单机 Milvus 绰绰有余

**B. 在线推理容量**

    单次提问的 token 消耗:
      输入 = system(300) + 检索上下文(2560) + 用户问题(50) ≈ 2900 token
      输出 ≈ 300 token
    GPU 需求 (Qwen-32B, FP16, 2张 A100-80G 做 TP=2):
      Prefill: 2 * 32e9 * 2900 = 1.86e14 FLOPs
      Decode:  2 * 32e9 * 300  = 1.92e13 FLOPs
      单请求总 ≈ 2.05e14 FLOPs
      单卡有效算力(30% MFU) 9.36e13 → 2 卡 = 1.87e14 FLOPs/s
      → 单请求串行需 ~1.1 s（假设完美利用）

**现实配置**：32B 模型 FP16 权重 64 GB，**至少需要 2 张 80GB 卡（TP=2）**；为满足 3 QPS 峰值和 99.5% 可用性：

    配置: 2 个副本 × (TP=2) = 4 张 A100-80G 跑 LLM
          + 1 张卡跑 Embedding + Rerank（可共享）
          + 1 张备用/灰度
          合计 6 张 A100-80G

**C. 成本明细（自建 vs 云）**

    【自建私有化】
     GPU 服务器 (8×A100-80G) × 1 台 ≈ ¥100~120 万
       → 3 年折旧 ≈ ¥3.0 万/月
     其他服务器/存储/网络 ≈ ¥20 万 → ¥0.6 万/月
     人力 (1 名算法 + 0.5 名运维) ≈ ¥5 万/月
     电费/机房 ≈ ¥0.5 万/月
     合计 ≈ ¥9 万/月  → 单次提问成本 ≈ 9万/(30*2万) ≈ ¥0.15/次

    【云上租用】
     4 张 A100-80G 按需 ¥12/卡/小时 × 720h = ¥3.46 万/月
     + Embedding/Rerank 2 卡 ¥1.7 万/月
     + 存储/网络 ≈ ¥0.5 万/月
     合计 ≈ ¥5.7 万/月 → 约 ¥0.095/次

**结论**：低并发场景**云上更划算**（自建的人力与折旧摊不平）；只有当 QPS 上升（或需严格数据隔离）时自建才合算。**盈亏平衡点大致在 10~15 QPS 持续负载**。

**D. 优化杠杆（按 ROI 排序）** 1. **缓存**：企业知识问答重复率高（HR/IT 政策类问题重复可达 40%）→ Redis 精确缓存直接省卡； 2. **小模型兜底**：用 7B 模型处理”简单问题”（分类 + 路由），只把复杂问题交给 32B → 成本可降 50%+； 3. **量化**：32B W8A8 / FP8 → 显存减半，单张 80G 可放下 → 少一张卡； 4. **KV Cache 复用**：system prompt + 常用文档的前缀缓存； 5. **检索优化**：减少 top-k 数量直接降低 prefill token（**prefill 成本与上下文长度线性相关**）。

------------------------------------------------------------------------

# 八、性能调优方法论

## Q29：遇到”训练/推理变慢了”，你的排查流程是什么？

答：**自上而下（Top-Down）**，四步法。

### Step 1 — 定义问题与基准（先量化，别猜）

    必须回答：
    - 是吞吐下降（tokens/s）还是延迟上升（step time）？
    - 从什么时候开始的？改了什么东西？(用 git/配置 diff 回溯)
    - 影响范围：单卡/单节点/整个集群？
    - 对比基准是什么？(历史最佳 MFU / 官方 benchmark)

**先算 MFU/BFU，与理论值对比** —— 这是最快定位”到底是计算问题还是通信问题”的手段。

### Step 2 — 分层二分定位

    1) 全局视角: Nsight Systems / torch profiler 看时间线
       · 找 "gap" —— kernel 之间的空隙意味着 CPU 或同步等待
       · 看 NCCL kernel 占比 → 通信是否成瓶颈
       · 看各 kernel 的时长分布 → 有没有异常长的 kernel
       ↓
    2) 单 kernel 视角: Nsight Compute
       · 看 Memory Throughput / Compute Throughput 占比 → roofline 定位
       · 看 warp stall reason:
         - "Long Scoreboard"  → 全局访存延迟（提升 occupancy/prefetch）
         - "Short Scoreboard" → shared memory 延迟（bank conflict?）
         - "Barrier"          → __syncthreads 等待（负载不均）
         - "Not Selected"     → 说明是好事，warp 够多
       · 看 achieved occupancy 与理论 occupancy
       ↓
    3) CPU 视角: py-spy / cProfile
       · py-spy dump --pid <pid>  看 Python 栈，定位卡在哪个函数
       · 看 DataLoader worker 是否成为瓶颈
       ↓
    4) 系统视角: nvidia-smi dmon / DCGM
       · GPU 利用率、显存、功耗、NVLink/IB 吞吐、ECC 错误
       · nvidia-smi topo -m 看拓扑是否符合预期

### Step 3 — 常见问题与解法对照表

| 症状 | 可能原因 | 解法 |
|----|----|----|
| GPU 利用率周期性掉到 0 | **数据加载慢**（data starvation） | 增 DataLoader worker、prefetch、本地缓存分片 |
| GPU 利用率不高但有 kernel 在跑 | 通信未重叠 / 小算子过多 | 梯度分桶调参、op fusion、CUDA Graph |
| 时间线中 NCCL 占比 \>40% | 通信瓶颈 | 检查网卡绑定（NCCL_IB_HCA）、换 Ring/Tree、加 TP overlap、减小 TP |
| 单卡很快，多卡就慢 | 通信配置错 / 拓扑不对 | `NCCL_DEBUG=INFO` 看是否走了 RDMA；`nvidia-smi topo -m` |
| 显存 OOM | batch 大 / 激活爆炸 / 碎片 | 开重计算、ZeRO、减少 micro-batch、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments` |
| loss 有 NaN / 尖刺 | 精度问题（FP16 overflow） | 换 BF16、加 loss scaling、梯度裁剪、检查脏样本 |
| 推理 TPOT 抖动大 | prefill 打断 decode | chunked prefill 或 PD 分离 |
| 训练中个别 step 特别慢 | **慢节点 / 坏卡** | 监控每卡 step 时间，主动摘除慢节点 |
| 首次迭代特别慢 | CUDA 编译/cuDNN autotune/模型加载 | 预热（warmup）若干 step 后再计时 |

### Step 4 — 验证与固化

- **改动后必须重测**，并记录 MFU/吞吐前后对比；
- 把配置**固化进版本管理**（NCCL 环境变量、并行配置、batch size 都该进配置）；
- 建立**性能回归测试**（每次框架升级跑一次 benchmark）。

**追问：举例说明一个真实的排查案例？** “训练 512 卡时 MFU 只有 28%，单机 8 卡时有 45%。排查： 1. `Nsight Systems` 时间线显示 NCCL 占比 45%，远超单机时的 8% → 定位到跨机通信； 2. `NCCL_DEBUG=INFO` 发现走了 **socket 而非 RDMA** —— 容器里 `NCCL_SOCKET_IFNAME` 没设，NCCL 选到了 docker0 虚拟网卡； 3. 设置 `NCCL_SOCKET_IFNAME=eth0`、`NCCL_IB_HCA=mlx5` 后，跨机带宽从 3 GB/s 提升到 45 GB/s； 4. MFU 回升到 41%。**同时还发现**：即使通信正常，ring 拓扑让部分 AllReduce 跨 spine，用拓扑感知调度把同组任务放到同 leaf 下，再提升到 44%。”

------------------------------------------------------------------------

## Q30：谈谈你对”AI 系统性能优化”的整体方法论理解（开放性总结题）

答：可以用一个**五层模型**收尾，这类问题最能体现系统性。

**L1 指标层：先定义”好”是什么** - 训练：MFU/BFU、step time、扩展效率（scaling efficiency = 实际加速比 / 线性加速比）、tokens/GPU/day； - 推理：**Goodput（SLO 内有效吞吐）**、TTFT/TPOT 的 P50/P99、每百万 token 成本； - **没有指标就没有优化**。第一步永远是建立可复现的 benchmark。

**L2 建模层：用 roofline 和通信模型预判瓶颈** - 算 `AI = FLOPs/Bytes`，与 `AI_ridge = P_peak/BW` 比较 → 判断 memory-bound / compute-bound； - 算通信量（`2S(N-1)/N` 等）与计算量之比 → 判断通信是否可隐藏； - **能在动手之前就预测出瓶颈在哪，是资深工程师和新手的区别**。

**L3 定位层：profiling 工具链** - 宏观：Nsight Systems / torch profiler → 时间线、gap、占比； - 微观：Nsight Compute → roofline、stall reason、occupancy； - 语言层：py-spy → Python 侧热点； - 系统层：DCGM / nvidia-smi → 硬件健康与吞吐。

**L4 优化层：按”瓶颈类型”选武器**

    memory-bound   → 算子融合、量化、提高数据复用(tiling/shared mem)、减少中间张量
    compute-bound  → Tensor Core 精度(FP16/FP8)、更大 tile、算法层面减少 FLOPs(MoE、线性注意力)
    communication  → 分桶、overlap、拓扑感知、换算法(Ring↔Tree)、降低并行度、压缩(FP16 通信)
    launch-bound   → CUDA Graph、算子融合、增大 batch
    data-bound     → DataLoader 优化、本地缓存、packing、预 tokenize

**L5 系统层：从单点优化到全局权衡** - **Amdahl 定律**：优化占比小的部分收益有限。先优化占比最大的（**用 profiling 数据说话，不要凭直觉**）； - **全局最优 ≠ 局部最优之和**：减小 batch 能让单请求更快，但总吞吐下降。**永远以业务 SLO 下的全局最优为目标**； - **成本是终极指标**：所有优化最终要换算成”每百万 token 的成本”或”每月的 GPU 小时开销”，这样才和业务对话。

**一句话总结**：**“先量化、再建模、后定位、最后才动手优化；每一步都用数字说话，优化前先想清楚收益从哪来。”**

------------------------------------------------------------------------

## 附录：高频数字速查表

    【显存】
    混合精度 + Adam 训练显存 ≈ 18N bytes (不含激活)
      = 2N(bf16 param) + 4N(fp32 master) + 4N(grad) + 4N(m) + 4N(v)
    推理显存 ≈ 2N (FP16) / 1N (INT8) / 0.5N (INT4)
    激活 ≈ L * s * b * h * c * bytes (c ≈ 10~20, 开重计算后降一个量级)

    【通信】
    AllReduce 每卡发送量 ≈ 2S(N-1)/N → 2S
    TP 每层 2 次 AllReduce；PP 每 micro-batch 一次 P2P
    ZeRO-3 通信量 ≈ 5S vs ZeRO-2 的 2S
    PP 气泡率 ≈ (PP-1)/(m+PP-1)

    【算力】
    训练单步 FLOPs ≈ 6*N*tokens + 12*L*s²*h
    MFU = 6*N*tokens_per_step / (step_time * P_peak * N_gpu)

    【推理】
    Decode 单 token 下界 ≈ 2*N*bytes_per_param / HBM_BW
    KV per token = 2 * L * n_kv_heads * d_head * bytes
      Llama-3-70B (GQA, FP16) ≈ 320 KB/token
    PagedAttention 碎片率 <4%

    【带宽金字塔】
    HBM(A100) 2039 GB/s | HBM(H100) 3350 GB/s
    NVLink4 900 GB/s(双向) | IB NDR 50 GB/s(单向) | PCIe4 x16 32 GB/s
    → 节点内 / 节点间 ≈ 18x

    【Roofline 拐点 (A100 FP16)】
    AI_ridge = 312e12 / 2.039e12 ≈ 153 FLOP/Byte

    【并行配置经验】
    TP ≤ 8 (限制在 NVLink 域内)
    PP 度数使气泡 < 5%
    DP 与 ZeRO 放跨机
    ZeRO-1 用于切 DP 组优化器状态

------------------------------------------------------------------------

本文件整理日期：2026-09-12
