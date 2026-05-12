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
- **🔌 MCP 支持** - 可扩展的外部工具集成协议

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
- Conda (推荐) 或 pip
- OpenAI API 密钥 (或兼容的 API)

### 安装

#### 方式 1: 使用 Conda (推荐)

```bash
# 克隆仓库
git clone https://github.com/mikaduu/deepresearch_agent.git
cd deepresearch_agent/deep_research_agent

# 创建环境
conda env create -f environment.yml
conda activate deepresearch
```

#### 方式 2: 使用 pip

```bash
git clone https://github.com/mikaduu/deepresearch_agent.git
cd deepresearch_agent/deep_research_agent

pip install -r requirements.txt
```

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

分析单篇论文的核心内容：

```bash
python main.py analyze "2301.00234" --focus "方法论"
```

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

---

## 🏗️ 架构设计

### 项目结构

```
deep_research_agent/
├── src/
│   ├── config.py              # 配置管理
│   ├── models.py              # 数据模型
│   ├── llm.py                 # LLM 客户端
│   ├── prompts.py             # 提示词模板
│   ├── orchestrator.py        # 核心编排器
│   ├── direction_evaluator.py # 方向评估 Agent
│   ├── paper_analyzer.py      # 论文分析 Agent
│   ├── context_manager.py     # 上下文管理
│   ├── mcp_client.py          # MCP 客户端
│   ├── memory/
│   │   ├── retriever.py       # TF-IDF RAG 检索
│   │   └── store.py           # 笔记持久化
│   └── services/
│       └── paper_search.py    # 论文搜索服务
├── workspace/
│   ├── notes/                 # 研究笔记
│   └── reports/               # 生成的报告
├── main.py                    # CLI 入口
├── requirements.txt
├── environment.yml
└── docs/                      # 详细文档（不发布，本地查看）
    ├── ARCHITECTURE.md        # 完整架构说明
    ├── MEMORY.md              # 三层记忆系统详解
    ├── LEARNING.md            # 反思引擎详解
    └── ROADMAP.md             # 未来发展路线图
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

### 集成 MCP 工具

```python
from src.mcp_client import MCPClient

mcp = MCPClient()
mcp.register_tool(
    name="custom_tool",
    description="自定义工具描述",
    parameters={"param1": "string", "param2": "number"}
)

result = mcp.call_tool("custom_tool", {"param1": "value", "param2": 42})
```

---

## 📚 文档

- [📐 架构详解](docs/ARCHITECTURE.md) - 详细的代码架构和设计文档
- [🧠 记忆系统详解](docs/MEMORY.md) - 三层记忆 + rerank 实现
- [🎓 学习引擎详解](docs/LEARNING.md) - ReflectionEngine 反思闭环
- [🗺️ 发展路线图](docs/ROADMAP.md) - 未来功能规划和优先级

### 当前开发重点（见 ROADMAP）

- 🚧 **优先级 1** - 自编排式 Agent 重构（把固定管线换成 ReAct-style autonomous loop）
- 📋 **优先级 2** - Summarizer 加全文阅读能力（让 research 不再只看 abstract）
- 📋 **优先级 3** - 接入 WebSearch，突破 arXiv 的学术信息墙

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

---

## 📧 联系方式

- GitHub: [@mikaduu](https://github.com/mikaduu)
- 项目链接: [https://github.com/mikaduu/deepresearch_agent](https://github.com/mikaduu/deepresearch_agent)

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个 Star！**

Made with ❤️ by the Deep Research Agent Team

</div>
