# 大模型面试面经合集

面向大模型 **算法研究 / 应用开发 / AI Infra** 岗位的系统化面试准备资料。

**10 份文档 · 345 道题 · 约 36 万字** —— 全部内容已转成 Markdown，可在 GitHub 上直接在线阅读和全文搜索，无需下载。

---

## 这份资料是什么

不是零散面经的堆砌。每份文档都按**知识体系**（而非来源仓库）重新组织，题目按模块串联，答案做了系统化的工程视角重写与补充：

- **八股成体系** —— 从 Transformer 架构、位置编码、注意力变体，到分布式训练、PEFT、RLHF、推理优化，公式尽量给全并可推导
- **面经带复盘** —— 字节 11 篇真实面经实录按时间线整理，附岗位版图、流程轮次、考察偏好、常见失败原因
- **大厂可横向对比** —— 12 家厂商的面试风格、高频真题、跨公司通用题 Top 榜
- **专题能落地** —— RAG / Agent / 推理部署 / Infra 四个方向各成专题，含手撕题、系统设计题、口述训练和速查卡

## 内容导航

| 文档 | 方向 | 覆盖内容 | 题量 |
|---|---|---|---|
| [**面经总索引**](00_面经总索引.md) | 总览 | 阅读路线 · 准备时间线 · 使用建议 | — |
| [字节跳动大模型面试复习手册 2025–2026](字节跳动大模型面试复习手册%202025至2026.md) | 字节专场 | **11 篇面经实录**：豆包 Seed、AI 应用研发、搜索推荐、Agent 平台、TikTok、飞书、AI Infra、推理优化等 | 97 |
| [字节跳动大模型面试_面经与真题复盘](字节跳动大模型面试_面经与真题复盘.md) | 字节专场 | 岗位版图与选岗 · 5 轮流程详解 · 字节面试风格 · 真题分类 · 避坑清单 | 23 |
| [大模型八股面试题_上_基础与架构](大模型八股面试题_上_基础与架构.md) | 八股 · 基础 | Transformer · 位置编码 RoPE/ALiBi · MHA/MQA/GQA · FlashAttention · MoE · 分词 · 采样 · Scaling Law | 27 |
| [大模型八股面试题_下_训练与微调](大模型八股面试题_下_训练与微调.md) | 八股 · 训练对齐 | 预训练与 Loss Spike · 混合精度 · ZeRO · 3D 并行 · LoRA/QLoRA · RLHF/PPO/GRPO · 蒸馏 | 34 |
| [RAG检索增强生成_面试专题](RAG检索增强生成_面试专题.md) | RAG | PDF 解析与分块 · Embedding 选型 · HNSW/IVF/PQ · BM25 与混合检索 · Rerank · 幻觉与评估 · 系统设计 | 35 |
| [AI_Agent开发_面试专题](AI_Agent开发_面试专题.md) | Agent | ReAct · Plan-and-Execute · Function Calling · **MCP** · 记忆系统与上下文工程 · 多 Agent 协作 · 评估与可观测 · 安全 · 系统设计 | 22 |
| [大模型推理优化与部署_面试专题](大模型推理优化与部署_面试专题.md) | 推理 / 部署 | Prefill/Decode · KV Cache 计算与优化 · PagedAttention · PD 分离 · GPTQ/AWQ/FP8 · 投机解码 · 服务化与成本 | 33 |
| [AI_Infra与大模型系统设计_面试专题](AI_Infra与大模型系统设计_面试专题.md) | Infra | GPU 架构与互联 · CUDA 编程与手撕算子 · 显存公式 · 并行策略 · 通信原语 · 推理系统 · 调度 · 性能调优 | 30 |
| [各大厂大模型面经合集_腾讯阿里百度快手等](各大厂大模型面经合集_腾讯阿里百度快手等.md) | 大厂横向 | 腾讯 / 阿里 / 百度 / 快手 / 美团 / 京东 / 小红书 / 华为 / 小米 / 讯飞 / 商汤 / 月之暗面 / 智谱 / MiniMax / DeepSeek —— 流程 · 真题 · 横向对比表 | 44 |

合计 **345 道题**。

## 按岗位方向的阅读路线

| 目标岗位 | 推荐路线 |
|---|---|
| **算法研究岗**（Seed / 通义 / 混元等） | 八股上 → 八股下 → 字节/大厂面经 → 推理优化（了解即可） |
| **应用 / Agent 开发岗** | 八股上（架构部分）→ RAG 专题 → Agent 专题 → 推理优化（成本与延迟部分） |
| **AI Infra / 训练推理工程岗** | 八股下（分布式训练）→ 推理优化与部署 → AI Infra 与系统设计 → 手撕 CUDA |
| **时间紧张（一周冲刺）** | 八股上 → 八股下 → 目标公司面经 → 手撕代码清单 |

## 建议的准备节奏

| 阶段 | 时间 | 任务 |
|---|---|---|
| 打基础 | 2–3 周 | 通读八股上下两篇，每个公式自己能推一遍 |
| 项目打磨 | 1 周 | 按 STAR + 技术决策梳理 2 个项目，准备被追问 3 层 |
| 方向深入 | 1–2 周 | 按岗位方向选读 RAG / Agent / 推理 / Infra 专题 |
| 手撕专项 | 1 周 | Attention、LayerNorm、KV Cache、采样、CUDA kernel 手写 |
| 模拟与复盘 | 3–5 天 | 用目标公司面经自测，限时口述 |

## 使用建议

1. **不要只读不写** —— 每道题合上文档，自己口述一遍，卡住的地方做标记。
2. **公式要能手推** —— Attention、DPO、ZeRO 显存、KV Cache 显存这几处是面试重灾区。
3. **项目是胜负手** —— 八股答得再好，项目讲不清也会挂。每个技术选型都要能回答「为什么不用另一个」。
4. **按公司微调** —— 确定目标公司后，优先精读对应面经文档，注意其考察偏好。
5. **结合真实数据** —— 讲工程实践时带上自己项目的 QPS、成功率、P99 延迟、成本，可信度高很多。

## 仓库结构

```
.
├── *.md            # 正文（10 份），GitHub 上可直接在线阅读
├── docx/           # 原始 Word 文件（10 份），排版更精确
└── README.md
```

根目录的 `.md` 就是正文，点开即可阅读，无需下载。Markdown 由 pandoc 转换生成（`pandoc -f docx -t gfm --wrap=none`），公式、表格、代码块已尽量保留；如需精确排版请参考 `docx/` 下的原件。

## 资料来源与致谢

内容整理自以下公开仓库与文档（按 star 数排序），并在各文档开头的「来源与参考」章节逐一署名：

| 仓库 | Star | 说明 |
|---|---|---|
| [wdndev/llm_interview_note](https://github.com/wdndev/llm_interview_note) | 15.1k | 大语言模型算法工程师知识及面试题，分 10 大模块 |
| [liyupi/mianshiya](https://github.com/liyupi/mianshiya) | 5.9k | 企业面试题库网站，含 AI 大模型 / Agent / RAG 面试题 |
| [WeThinkIn/AIGC-Interview-Book](https://github.com/WeThinkIn/AIGC-Interview-Book) | 4.7k | AIGC / LLM / AI Agent 面试资源平台 |
| [datawhalechina/daily-interview](https://github.com/datawhalechina/daily-interview) | 3.8k | Datawhale 面经合集 |
| [315386775/DeepLearing-Interview-Awesome-2024](https://github.com/315386775/DeepLearing-Interview-Awesome-2024) | 2.9k | AIGC / CV / LLMs 面试问题与答案集合 |
| [km1994/LLMs_interview_notes](https://github.com/km1994/LLMs_interview_notes) | 2.6k | 大模型算法工程师面试题整理 |
| [llmgenai/LLMInterviewQuestions](https://github.com/llmgenai/LLMInterviewQuestions) | 1.9k | 国外大厂（Google / NVIDIA 等）LLM 面试题 |
| [Devinterview-io/llms-interview-questions](https://github.com/Devinterview-io/llms-interview-questions) | 1.1k | LLM 面试问答（英文） |
| [KalyanKS-NLP/LLM-Interview-Questions-and-Answers-Hub](https://github.com/KalyanKS-NLP/LLM-Interview-Questions-and-Answers-Hub) | 1.0k | 100+ LLM 面试题与答案（英文） |
| [jingtian11/EasyOffer](https://github.com/jingtian11/EasyOffer) | 822 | 《大模型面经合集》，按公司分目录 |
| [Junvate/LLM-Algorithm-Intern-Guide](https://github.com/Junvate/LLM-Algorithm-Intern-Guide) | 678 | 2026 届大模型算法岗实习面经 |
| [luxuantao/advanced_LLM_interview_notes](https://github.com/luxuantao/advanced_LLM_interview_notes) | 158 | 大模型进阶面经 |
| [DolbyUUU/Awesome-LLM-Interview-Questions-and-Answers](https://github.com/DolbyUUU/Awesome-LLM-Interview-Questions-and-Answers) | 52 | 大模型算法 / Agent 开发常见题 |
| [MisterBooo/llm-interview-questions](https://github.com/MisterBooo/llm-interview-questions) | 53 | 100 道大模型面试题与 100 张 SVG 图解 |
| [712sir/ai-infra-career](https://github.com/712sir/ai-infra-career) | 27 | 推理引擎 / 分布式训练求职路线图 |

此外还参考了 Anthropic *Building Effective Agents*、OpenAI Function Calling / Agents SDK 文档、Model Context Protocol 规范、LangGraph / LlamaIndex / AutoGen / vLLM / SGLang 等官方文档，以及 ReAct、Reflexion、MemGPT、FlashAttention、ZeRO、LoRA、DPO 等原始论文。

> `jingtian11/EasyOffer` 与 `Junvate/LLM-Algorithm-Intern-Guide` 更新较频繁，建议定期直接浏览原仓库获取最新真题。

## 免责声明

- 面经类内容（尤其是**流程、轮次、题目细节**）来自公开经验分享与个人整理，**实际面试会有差异**，请以自己收到的通知为准。
- 答案中的工程实践数据为综合整理的**典型值**，用于理解量级与权衡，不代表任何具体线上系统。
- 本文档仅供学习交流，请勿用于商业用途；转载请注明来源。

---

如果这份资料对你有帮助，欢迎 Star ⭐
