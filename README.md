# Deep Research Agent

面向科研文献调研的 Agent 原型。当前版本支持论文检索、GitHub 代码关联、保守版综述生成、算法演进图生成、SVG 海报生成，以及配置 API key 后的 LLM 驱动深度研究。

## 功能概览

- 自动检索设定主题相关论文，优先覆盖 arXiv 和 Semantic Scholar。
- 自动搜索论文/方法对应的 GitHub 公开代码，并优先展示有代码的论文。
- 生成文献综述草稿，避免在没有全文或 API key 时编造细节。
- 自动生成算法发展演进图，按论文发布日期排序，降低时间线错误。
- 自动生成 SVG 综述海报，可直接用浏览器打开或放入报告/幻灯片。
- 配置 OpenAI 兼容 API 后，可运行 `research` / `analyze` / `read-paper` 做深度分析。

## 环境要求

- Python 3.11 或更高版本，建议 3.11/3.12。
- Windows PowerShell、macOS Terminal 或 Linux shell。
- 可选：OpenAI 兼容 LLM API key。没有 key 也可以运行 `search`、`discover`、`survey`。

## 安装流程

进入项目目录：

```powershell
cd TJU-CIP2026-Final
```

创建虚拟环境：

```powershell
python -m venv .venv
```

激活虚拟环境：

```powershell
.\.venv\Scripts\activate
```

安装依赖：

```powershell
pip install -r requirements.txt
```

如果网络较慢，`sentence-transformers` / `torch` 安装时间会比较长，这是正常的。

## 配置 API Key

复制配置模板：

```powershell
copy .env.example .env
```

编辑 `.env`。如果使用 OpenAI 官方 API：

```env
LLM_API_KEY=你的_API_KEY
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_ID=gpt-4o-mini
```

如果使用 DeepSeek、Qwen、OpenRouter、硅基流动等 OpenAI 兼容平台，只需要替换：

```env
LLM_API_KEY=对应平台的_API_KEY
LLM_BASE_URL=对应平台的_OpenAI兼容地址
LLM_MODEL_ID=对应模型名
```

如果暂时没有 API key，可以保持 `LLM_API_KEY=` 为空。此时不要运行需要 LLM 的命令，例如 `research`、`analyze`、`evaluate`，但可以运行 `search`、`discover`、`survey`。

建议保守配置：

```env
SEARCH_TOP_K=4
MAX_PLAN_ITEMS=3
ENABLE_RERANK=false
RESEARCH_WORKSPACE_DIR=workspace
```

这样可以降低 arXiv 访问压力和首次模型下载压力。

## 工作流程

### 1. 快速搜索论文

只搜索论文，不查代码：

```powershell
python main.py search "diffusion models"
```

输出内容包括论文标题、作者、摘要片段和论文链接。

### 2. 搜索论文并关联 GitHub 代码

推荐先用这个命令做选题材料收集：

```powershell
python main.py discover "diffusion models" --max-papers 10
```

这个流程会：

- 搜索主题相关论文。
- 自动搜索 GitHub 公开代码。
- 将有公开代码的论文排在前面。
- 输出论文年份、标题、代码链接和论文链接。

### 3. 生成综述、演进图和海报

无 API key 也能运行：

```powershell
python main.py survey "diffusion models" --max-papers 12
```

生成文件位于：

```text
workspace/surveys/<时间戳>_<主题>/
```

目录内包含：

- `survey_report.md`：文献综述草稿。
- `algorithm_timeline.svg`：算法发展演进图。
- `survey_poster.svg`：综述海报。
- `papers.json`：检索到的论文和代码原始数据。

### 4. 配置 API key 后做深度研究

配置好 `.env` 后，可以运行：

```powershell
python main.py research "large language model reasoning"
```

该流程会调用 LLM Agent，自动规划、搜索、整理证据、生成报告，并尽量调用 GitHub 搜索工具优先纳入有公开代码的论文。

报告会保存到：

```text
workspace/reports/
```

### 5. 分析单篇论文

保守版分析 arXiv 论文：

```powershell
python main.py analyze "2301.00234" --focus "methodology"
```

细读论文：

```powershell
python main.py read-paper "https://arxiv.org/abs/2301.00234" --focus "experiments"
```

本地 PDF：

```powershell
python main.py read-paper "D:\path\to\paper.pdf" --title "Paper Title"
```

分析笔记会保存到：

```text
workspace/paper_notes/
```

## 推荐使用顺序

1. 用 `discover` 快速确认主题下有哪些论文有公开代码。
2. 用 `survey` 生成第一版综述、演进图和海报。
3. 人工检查 `papers.json` 中的论文和代码链接，排除误匹配。
4. 配置 API key。
5. 对核心论文运行 `analyze` 或 `read-paper`。
6. 运行 `research` 生成完整研究报告。
7. 将 `workspace/surveys/`、`workspace/paper_notes/`、`workspace/reports/` 中的产物整理进课题报告。

## 性能与质量约束

当前实现针对以下要求做了约束：

- 论文主题准确率目标不低于 85%：通过标题/摘要关键词匹配、去重和人工可复核链接支持，不自动宣称最终准确率。
- 综述文字无明显幻觉：无 API key 时只基于检索元数据写保守综述，不生成超出证据的细节结论。
- 算法演进图无明显时间错误：演进图按论文 `published` / `updated` 日期排序生成。
- 优先考虑公开代码：`discover` 和 `survey` 会把有 GitHub 代码线索的论文优先展示。

## 常见问题

### 没有 API key 能做什么？

可以运行：

```powershell
python main.py search "topic"
python main.py discover "topic"
python main.py survey "topic"
```

不能完整运行：

```powershell
python main.py research "topic"
python main.py analyze "arxiv_id"
python main.py evaluate "idea"
```

这些命令需要 LLM。

### arXiv 被限流怎么办？

- 降低 `SEARCH_TOP_K`。
- 避免短时间重复运行同一主题。
- 优先用 `survey --max-papers 6` 做小规模测试。
- 被限流后等待一段时间再运行。

### GitHub 代码关联不准怎么办？

自动关联依赖公开网页搜索，可能出现同名仓库误匹配。正式提交前应人工检查 `papers.json` 和报告中的 GitHub 链接。

## 主要命令汇总

```powershell
python main.py --help
python main.py search "diffusion models"
python main.py discover "diffusion models" --max-papers 10
python main.py survey "diffusion models" --max-papers 12
python main.py analyze "2301.00234" --focus "methodology"
python main.py read-paper "https://arxiv.org/abs/2301.00234"
python main.py research "large language model reasoning"
python main.py gui
```
