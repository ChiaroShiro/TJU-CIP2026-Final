# 🔬 Deep Research Agent

<div align="center">

**基于多智能体架构的论文研究方向搜索与深度研究系统**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[功能特性](#-功能特性) • [快速开始](#-快速开始) • [使用指南](#-使用指南) • [架构设计](#-架构设计) • [贡献指南](#-贡献指南)

</div>

---

## ✨ 功能特性

### 🎯 核心能力

- **🔍 研究方向评估** - 智能评估研究方向的可行性、新颖性和影响力
- **📚 论文搜索** - 集成 arXiv 和 Semantic Scholar，全面覆盖学术资源
- **📖 论文深度分析** - 自动提取核心贡献、方法论、局限性和未来方向
- **📝 研究报告生成** - 自动生成结构化、引用完整的研究报告
- **🧠 记忆系统** - 基于 TF-IDF 的 RAG 检索，支持上下文关联
- **🔄 上下文管理** - 智能管理对话历史，防止 token 溢出
- **🔌 Tool 系统** - 12 个内置工具 + 可扩展的 Tool 注册表

### 🤖 Multi-Agent 架构

```
┌─────────────────────────────────────────────────────────┐
│                  ResearchOrchestrator                   │
│                     (核心编排器)                          │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Direction    │ │ Paper        │ │ Research     │
│ Evaluator    │ │ Analyzer     │ │ Planner      │
│ (方向评估)    │ │ (论文分析)    │ │ (研究规划)    │
└──────────────┘ └──────────────┘ └──────────────┘
        │            │            │
        └────────────┼────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────┐         ┌──────────────┐
│ RAG Memory   │         │ Paper Search │
│ (记忆检索)    │         │ (论文搜索)    │
└──────────────┘         └──────────────┘
```

---

## 🚀 快速开始

### 环境要求

- Python 3.11+
- Conda 或 venv
- OpenAI API 密钥 (或兼容的 API)

### 安装

#### 方式 1: 使用 Conda

```bash
# 克隆仓库
git clone https://github.com/mikaduu/deepresearch_agent.git
cd deepresearch_agent/deep_research_agent

# 创建环境（按你的实际环境名）
conda create -n hello_agent python=3.11 -y
conda activate hello_agent

# 安装依赖
pip install -r requirements.txt
```

#### 方式 2: 使用 venv / 系统 Python

```bash
git clone https://github.com/mikaduu/deepresearch_agent.git
cd deepresearch_agent/deep_research_agent

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

当前项目以 `requirements.txt` 作为依赖安装入口，不再要求 `environment.yml`。

### 配置

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件
nano .env
```

配置示例：

```env
LLM_API_KEY=sk-your-api-key-here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_ID=gpt-4o-mini
```

> 💡 **提示**: 支持任意 OpenAI 兼容 API，如 DeepSeek、Qwen、本地 Ollama 等

---

## 📖 使用指南

### 1️⃣ 评估研究方向

评估一个研究想法是否值得深入：

```bash
python main.py evaluate "使用强化学习优化大语言模型的推理能力"
```

**输出示例**：

```
┏━━━━━━━━━━┳━━━━━━━┓
┃ 维度     ┃ 得分  ┃
┡━━━━━━━━━━╇━━━━━━━┩
│ 可行性   │ 0.85  │
│ 新颖性   │ 0.72  │
│ 影响力   │ 0.90  │
└──────────┴───────┘

详细分析:
该方向结合了强化学习和大语言模型两个热门领域...
建议从以下切入点开始: 1) 设计合适的奖励函数...
```

### 2️⃣ 搜索论文

快速搜索相关学术论文：

```bash
python main.py search "reinforcement learning for LLM reasoning"
```

### 3️⃣ 深度分析论文

项目里现在有两条论文阅读命令：

- `analyze`
  - 保守版，默认给调研流程使用
  - 面向 `arXiv ID / arXiv URL`
  - 优先做稳定的文本分析，不主动走图片理解
  - 更适合批量调研、方向评估、`research` 链路

- `read-paper`
  - 细读版
  - 支持 `本地 PDF` 和 `arXiv ID / arXiv URL`
  - `arXiv` 来源会走多模态细读，支持结合论文图片做额外解读
  - `本地 PDF` 来源目前走保守版文本细读，不强依赖图片
  - 更适合你已经决定要认真读某一篇论文时使用

可以直接按下面理解：

| 命令 | 支持输入 | 默认风格 | 是否看图 | 适合场景 |
| --- | --- | --- | --- | --- |
| `analyze` | arXiv ID / arXiv URL | 保守版 | 默认不看图 | 调研、综述、批量筛论文 |
| `read-paper` | 本地 PDF / arXiv ID / arXiv URL | 细读版 | arXiv 会看图；本地 PDF 默认不看图 | 精读单篇论文、沉淀详细笔记 |

#### `analyze`：保守版 arXiv 分析

分析单篇论文的核心内容：

```bash
python main.py analyze "2301.00234" --focus "方法论"
```

特点：

- 输出更克制，适合调研场景
- 默认中文总结
- 会生成论文笔记，但不会像 `read-paper` 那样强调多模态细读
- `research` 默认调用的是这一条链路

#### `read-paper`：细读论文

如果你想认真读一篇论文，使用 `read-paper`：

```bash
python main.py read-paper "https://arxiv.org/abs/2301.00234" --focus "方法论"
```

也可以直接阅读本地 PDF：

```bash
python main.py read-paper "D:\path\to\paper.pdf" --focus "方法论"
```

如果 PDF metadata 里的标题不准，可以手动指定：

```bash
python main.py read-paper "D:\path\to\paper.pdf" --title "Direct Preference Optimization" --focus "实验设计"
```

特点：

- `arXiv` 版本会走多模态细读，包含图表补充解读
- `本地 PDF` 版本目前更偏保守版文本细读
- 会生成更完整的论文笔记，适合放进 `workspace/paper_notes/`

**输出包含**：
- 核心问题
- 主要贡献
- 方法论
- 实验结果
- 局限性
- 未来方向

### 4️⃣ 深度研究

对主题进行全面研究并生成报告：

```bash
python main.py research "大语言模型的可解释性研究"
```

**生成报告包含**：
- 📋 执行摘要
- 🔍 详细发现
- ⚠️ 风险和不确定性
- 💡 可行建议
- 📚 完整参考文献

报告保存在 `workspace/reports/` 目录。

### 5️⃣ 本地 GUI 工作台

如果你希望更方便地使用论文搜索、阅读和图谱查看，可以直接启动本地 GUI：

```bash
python main.py gui
```

如果不想自动打开浏览器：

```bash
python main.py gui --no-browser
```

GUI 当前支持：

- 搜索论文并一键填入阅读框
- 调用 `analyze` 保守版分析
- 调用 `read-paper` 细读模式
- 浏览 `workspace/paper_notes/` 里的已读笔记
- 可视化 `paper graph memory` 中的论文节点和关系边
- 点击图节点查看摘要并快速打开对应笔记

---

## 🏗️ 架构设计

### 项目结构

```
deep_research_agent/
├── src/
│   ├── orchestrator.py              # 核心编排器
│   ├── core/                        # 基础设施层
│   │   ├── config.py                # Settings 配置
│   │   ├── context_manager.py       # 会话记忆 Layer 1
│   │   ├── llm.py                   # LLM 客户端 (invoke + invoke_with_tools)
│   │   ├── models.py                # 数据模型
│   │   ├── prompts.py               # 提示词模板
│   │   └── utils.py                 # JSON 解析工具
│   ├── tools/                       # 工具抽象层（autonomous 基础）
│   │   ├── tool.py                  # Tool / ToolResult / ToolRegistry
│   │   ├── search_tools.py          # arxiv + s2 搜索 tool
│   │   ├── paper_tools.py           # fetch 全文 + analyze tool
│   │   └── memory_tools.py          # retrieve / save_note / save_episode tool
│   ├── agents/                      # Agent 层
│   │   ├── base_agent.py            # BaseAgent (autonomous ReAct loop 基类)
│   │   ├── conversational_agent.py  # 对话路由 Agent
│   │   ├── direction_evaluator.py   # 方向评估 Agent
│   │   ├── paper_analyzer.py        # 论文深度分析 Agent
│   │   ├── critic.py                # 评审 Agent
│   │   └── reviser.py               # 修订 Agent
│   ├── memory/                      # 三层记忆 + 语义索引
│   │   ├── memory_manager.py        # MemoryManager 统一入口
│   │   ├── episodic_memory.py       # 情节记忆 (SQLite+FTS5)
│   │   ├── skill_memory.py          # 技能记忆 (SQLite+FTS5)
│   │   ├── vector_store.py          # Chroma 向量存储
│   │   ├── reranker.py              # Cross-Encoder 精排
│   │   └── store.py                 # Markdown 笔记落盘
│   ├── learning/                    # 自我学习
│   │   └── reflection.py            # ReflectionEngine (反思闭环)
│   └── services/                    # 外部服务
│       ├── paper_search.py          # arXiv + Semantic Scholar
│       └── paper_fetcher.py         # PDF 下载 + 章节提取
├── workspace/                       # 运行时数据 (.gitignore 忽略)
│   ├── memory/ notes/ reports/ pdf_cache/ vector_db/
├── main.py                          # CLI 入口
├── requirements.txt
└── docs/                            # 详细文档（不发布，本地查看）
    ├── ARCHITECTURE.md              # 完整架构说明
    ├── TOOLS.md                     # 工具抽象 + BaseAgent 详解
    ├── MEMORY.md                    # 三层记忆系统详解
    ├── LEARNING.md                  # 反思引擎详解
    └── ROADMAP.md                   # 发展路线图
```

### 核心组件

#### 🎯 DirectionEvaluator (方向评估器)

评估研究方向，提供：
- ✅ 可行性分析
- 🆕 创新性评估
- 📈 影响力预测
- 💡 研究建议

#### 📖 PaperAnalyzer (论文分析器)

深度分析论文：
- 🎯 核心问题识别
- 🏆 主要贡献提取
- 🔬 方法论解析
- 📊 实验结果总结
- ⚠️ 局限性分析
- 🔮 未来方向预测

#### 🎼 ResearchOrchestrator (研究编排器)

协调整个研究流程：
1. 📋 制定研究计划
2. 🔍 搜索相关论文
3. 📝 分析和总结
4. 📄 生成最终报告

---

## 🔧 高级配置

### 自定义 LLM 提供商

支持任意 OpenAI 兼容 API：

```env
# DeepSeek
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL_ID=deepseek-chat

# 本地 Ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL_ID=llama3
```

### 调整搜索参数

```env
SEARCH_TOP_K=10          # 每次搜索返回论文数
MEMORY_TOP_K=5           # RAG 检索返回记忆条数
MAX_PLAN_ITEMS=7         # 研究计划最多任务数
CONTEXT_MAX_CHARS=10000  # 上下文最大字符数
```

---

## 🛠️ 扩展开发

### 添加自定义 Agent

```python
from src.llm import LLMClient
from src.config import Settings

class CustomAgent:
    def __init__(self, llm_client: LLMClient, settings: Settings):
        self.llm = llm_client
        self.settings = settings
    
    def process(self, input_data):
        prompt = f"处理: {input_data}"
        return self.llm.invoke([{"role": "user", "content": prompt}], 0.3)
```

### 自定义工具（Tool 系统）

```python
from src.tools import Tool, ToolRegistry, ToolResult

my_tool = Tool(
    name="my_custom_tool",
    description="What this tool does (LLM reads this to decide when to use it)",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    run=lambda args: ToolResult(success=True, content=f"Result for {args['query']}"),
)

# 注册到 Manager 的工具箱
registry = ToolRegistry()
registry.register(my_tool)
```

---

## 📚 文档

- [📐 架构详解](docs/ARCHITECTURE.md) - 分层架构、数据流、执行模式
- [🔧 工具层 + BaseAgent](docs/TOOLS.md) - autonomous 基础抽象详解
- [🧠 记忆系统详解](docs/MEMORY.md) - 三层记忆 + rerank 实现
- [🎓 学习引擎详解](docs/LEARNING.md) - ReflectionEngine + Skill-Creator 范式
- [🗺️ 发展路线图](docs/ROADMAP.md) - 未来功能规划和优先级

### 当前开发重点（见 ROADMAP）

- ✅ **Step 1-6 全部完成** — 自主研究架构已就绪
  - Tool 抽象 + BaseAgent + 12 个工具
  - ResearchManager（主脑 autonomous agent）
  - CriticWorker / ReviserWorker（autonomous workers）
  - web_search / web_fetch（突破学术信息墙）
- 📋 **下一步** — LangGraph 对照版本 / GitHub 搜索 / Web UI

---

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 开发规范

- 使用 `black` 格式化代码
- 添加类型注解
- 编写文档字符串
- 添加单元测试

---

## 📄 License

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- [arXiv](https://arxiv.org/) - 开放获取的预印本论文库
- [Semantic Scholar](https://www.semanticscholar.org/) - 学术搜索引擎
- [OpenAI](https://openai.com/) - LLM API 提供商
- [DeepSeek](https://www.deepseek.com/) - 高性价比 LLM 服务，本项目默认使用
- [Anthropic Agent Skills](https://docs.anthropic.com/) - 项目中的 skill-creator 范式参考
- [dailypaper-skills](https://github.com/) - 论文阅读模块的多源图片 fallback、零遗漏原则、概念库联动等设计参考自该项目的 `paper-reader` skill；具体借鉴点包括：
  - 多源图片获取流程（arXiv HTML → 项目主页 → PDF 提取）
  - 公式 5 类必检规范（变量冲突 / 文本-公式不一致 / 符号约定不一致 / 求和范围错 / 缺算子）
  - URL 规范化（防止 arxiv_id 路径重复 bug）
  - 图片可达性检查 + 选择性本地化（不可达自动下载到 `assets/`）
  - 笔记模板结构（YAML frontmatter + 一句话总结 + 核心贡献 + 方法详解 + 关键公式 + 关键图表 + 批判性思考 + 速查卡片）

---

## 📧 联系方式

- GitHub: [@mikaduu](https://github.com/mikaduu)
- 项目链接: [https://github.com/mikaduu/deepresearch_agent](https://github.com/mikaduu/deepresearch_agent)

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个 Star！**

Made with ❤️ by the Deep Research Agent Team

</div>
