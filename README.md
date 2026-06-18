# Deep Research Agent

面向科研文献调研的 Agent 系统。支持论文检索、GitHub 代码关联、综述生成、多分支算法演进图、内嵌原文图的 SVG 海报、论文知识图谱，以及配置 API key 后的 LLM 驱动深度研究。除命令行外，还提供一个覆盖全部能力的本地 GUI 工作台。

## 功能概览

- 自动检索主题相关论文，覆盖 arXiv 和 Semantic Scholar，并对结果去重、按相关度与时间排序。
- 自动搜索论文/方法对应的 GitHub 公开代码，将有代码的论文优先排序。
- 生成文献综述草稿；无 API key 时只基于检索元数据写保守综述，不编造超出证据的细节。
- 自动生成**算法演进图**：纵轴按发布时间从早到晚排列（避免时间线错误），横向按技术分支分泳道。配置 LLM key 后由 LLM 在论文集合内部划分技术分支，并标注 `builds_on`（继承/改进，实线）/ `compares_with`（对比，虚线）关系；无 key 时按 arXiv 主类目确定性分泳道。
- 自动生成 SVG 综述海报，尽力内嵌论文原图，可直接用浏览器打开或放进报告/幻灯片。
- 维护一个**论文知识图谱**：精读论文、综述/研究检索发现的论文、引用占位三层，GUI 中可交互查看并按层次筛选。
- 配置 OpenAI 兼容 API 后，可运行 `research` / `analyze` / `read-paper` / `evaluate` / `chat` 做 LLM 驱动的深度分析。
- 提供本地 GUI（`python main.py gui`），把上述能力集中到一个网页工作台，检索/分析/研究过程实时流式展示。

## 环境要求

- Python 3.11 或更高版本，建议 3.11/3.12。
- Windows PowerShell、macOS Terminal 或 Linux shell。
- 可选：OpenAI 兼容 LLM API key。没有 key 也可以运行 `search`、`discover`、`survey`，以及 GUI 中对应的视图。

## 安装流程

进入项目目录：

```powershell
cd TJU-CIP2026-Final
```

创建并激活虚拟环境（venv 或 conda 均可）：

```powershell
python -m venv .venv
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

如果使用 DeepSeek、Qwen、OpenRouter、硅基流动等 OpenAI 兼容平台，只需要替换为对应平台的 key、OpenAI 兼容地址与模型名。

`read-paper` 的多模态图表解读会用到视觉模型，可选配 `VISION_API_KEY` / `VISION_BASE_URL` / `VISION_MODEL_ID`（缺省时自动复用上面的 LLM 配置）。

如果暂时没有 API key，可以保持 `LLM_API_KEY=` 为空。此时不要运行需要 LLM 的命令（`research`、`analyze`、`read-paper`、`evaluate`、`chat`），但可以运行 `search`、`discover`、`survey`。

建议保守配置（降低 arXiv 访问压力和首次模型下载压力）：

```env
SEARCH_TOP_K=4
MAX_PLAN_ITEMS=3
ENABLE_RERANK=false
RESEARCH_WORKSPACE_DIR=workspace
```

## 命令行工作流程

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
- 输出论文年份、标题、公开代码链接和论文链接（GUI 的「发现 · 代码」视图还会显示代码置信度）。

### 3. 评估研究方向

需要 API key：

```powershell
python main.py evaluate "用扩散模型为机器人操作生成多模态动作策略"
```

输出可行性 / 新颖性 / 影响力评分，以及现状分析、研究建议和相关 benchmark。

### 4. 生成综述、演进图和海报

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
- `algorithm_evolution.svg`：算法演进图（多泳道 + 演进关系）。
- `survey_poster.svg`：综述海报（尽力内嵌论文原图）。
- `papers.json`：检索到的论文与代码原始数据。

演进图行为：无 API key 时按 arXiv 主类目确定性分泳道、按发布时间纵向排列；配置 key 后升级为 LLM 划分技术分支并标注 `builds_on` / `compares_with` 关系。两种模式下边的方向都强制为"较早 → 较晚"，不会出现时间错误。

### 5. 分析单篇论文

保守版分析 arXiv 论文：

```powershell
python main.py analyze "2301.00234" --focus "methodology"
```

细读论文（arXiv 走多模态细读版，本地 PDF 走保守版）：

```powershell
python main.py read-paper "https://arxiv.org/abs/2301.00234" --focus "experiments"
python main.py read-paper "D:\path\to\paper.pdf" --title "Paper Title"
```

分析笔记会保存到：

```text
workspace/paper_notes/
```

`paper-reader` 是 `read-paper` 的别名。

### 6. 深度研究

配置好 `.env` 后：

```powershell
python main.py research "large language model reasoning"
```

默认走**自主 Agent 模式**：Agent 通过 function calling 自行规划，自动检索、关联代码、精读核心论文、委派独立 Critic 评审、反思沉淀记忆，最后生成报告。可选参数：

- `--max-steps` / `-s`：最大决策步数（默认 30）。
- `--max-tokens` / `-t`：总 token 预算（默认 200000）。
- `--legacy`：改用编排式流水线（规划 → 检索 → 总结 → 评审/修改 → 报告）。

报告会保存到：

```text
workspace/reports/
```

> 提示：在 GUI 中运行深度研究会额外生成算法演进图与海报，并把论文并入图谱；命令行 `research` 仅生成报告。

### 7. 对话模式

需要 API key：

```powershell
python main.py chat
```

用自然语言描述研究想法，系统会自动识别意图并调用搜索 / 评估 / 精读 / 研究等能力。

## 图形界面（GUI）

启动本地工作台：

```powershell
python main.py gui
```

加 `--no-browser` 只启动服务、不自动打开浏览器。启动后终端会打印本地访问地址（形如 `http://127.0.0.1:<port>`），按 `Ctrl+C` 关闭服务。

GUI 左侧导航包含以下视图：

| 视图 | 说明 |
| --- | --- |
| 搜索 | 从 arXiv / Semantic Scholar 检索论文，结果可一键填入精读 |
| 发现 · 代码 | 检索论文并实时关联 GitHub 代码，有代码的论文优先 |
| 方向评估 | 对研究方向打分（可行性 / 新颖性 / 影响力）并给出现状分析（需 LLM） |
| 论文精读 | 输入 arXiv ID / 链接 / 本地 PDF，逐章节流式精读（需 LLM） |
| 综述生成 | 生成综述草稿、算法演进图、海报；并把论文以"检索发现"层并入论文图谱 |
| 深度研究 | 自主 / 编排式研究，实时显示决策轨迹与逐字报告；完成后追加演进图与海报，并并入图谱（需 LLM） |
| 对话 | 自然语言驱动，自动识别意图并调用各能力，回复逐字流式（需 LLM） |
| 论文图谱 | 交互式知识图：精读（青）/ 检索发现（紫）/ 引用占位（橙）三层着色，可用层次开关筛选；边为 builds_on / compares_with / similar_to |
| 笔记 | 浏览精读笔记，支持关键词筛选与 Markdown 阅读 |

需要 LLM 的视图（方向评估 / 论文精读 / 深度研究 / 对话）在未配置 key 时会直接给出提示。

## 接口说明（GUI HTTP API）

GUI 后端是一个本地 HTTP 服务，前端通过下列接口通信。同步接口返回 `{ "ok": true, "data": ... }`；流式接口为 `text/event-stream`，逐条推送 `phase` / `log` / `step` / `token` / `intent` / `result` / `error` / `done` 事件。

| 方法 | 路径 | 形式 | 参数 / 请求体 | 说明 |
| --- | --- | --- | --- | --- |
| GET | `/api/health` | JSON | — | 健康检查 |
| GET | `/api/stats` | JSON | — | 记忆库统计 |
| GET | `/api/capabilities` | JSON | — | 是否已配置 LLM key |
| GET | `/api/graph` | JSON | `?q=` | 论文图谱快照（节点 / 边 / 分层） |
| GET | `/api/notes` | JSON | `?q=` | 笔记列表（最多 50 条） |
| GET | `/api/note` | JSON | `?path=` | 单篇笔记内容 |
| GET | `/api/asset` | 文件 | `?path=` | workspace 内的图片 / SVG |
| POST | `/api/search` | JSON | `{query}` | 搜索论文 |
| POST | `/api/survey` | JSON | `{topic, max_papers}` | 生成综述 + 演进图 + 海报 |
| POST | `/api/figure` | JSON | `{mode, title, spec}` | 统一图产物契约示例 |
| POST | `/api/open-note` | JSON | `{path}` | 读取笔记 |
| POST | `/api/chat-clear` | JSON | `{}` | 清空对话历史 |
| POST | `/api/discover` | SSE | `{topic, max_papers}` | 发现 + 代码关联（流式） |
| POST | `/api/evaluate` | SSE | `{direction}` | 方向评估（流式） |
| POST | `/api/analyze` | SSE | `{source, mode, focus, title}` | 论文精读（流式） |
| POST | `/api/research` | SSE | `{topic, mode, max_steps, max_tokens}` | 深度研究（流式） |
| POST | `/api/chat` | SSE | `{message}` | 对话（流式） |

`/api/asset`、`/api/note` 等文件接口都限制在 workspace 目录内，防止路径穿越。

## 产物与目录

```text
workspace/
├── surveys/<时间戳>_<主题>/   # survey 产物：survey_report.md、algorithm_evolution.svg、survey_poster.svg、papers.json
├── reports/                   # 深度研究报告
├── paper_notes/               # 精读笔记（含 assets/ 图片）
├── memory/                    # episodic.db / skills.db / paper_graph.db（情节 / 技能 / 论文图谱）
└── vector_db/                 # 向量库
```

## 推荐使用顺序

1. 用 `discover` 快速确认主题下有哪些论文有公开代码。
2. 用 `survey` 生成第一版综述、算法演进图和海报。
3. 人工检查 `papers.json` 中的论文和代码链接，排除误匹配。
4. 配置 API key。
5. 对核心论文运行 `analyze` 或 `read-paper`。
6. 运行 `research` 生成完整研究报告，或在 GUI 中运行以同时得到演进图、海报与图谱。
7. 将 `workspace/surveys/`、`workspace/paper_notes/`、`workspace/reports/` 中的产物整理进课题报告。

> 也可以直接 `python main.py gui`，在一个网页里完成上述全部步骤。

## 性能与质量约束

当前实现针对以下要求做了约束：

- 论文主题准确率目标不低于 85%：通过标题/摘要关键词匹配、去重和人工可复核链接支持，不自动宣称最终准确率。
- 综述文字无明显幻觉：无 API key 时只基于检索元数据写保守综述，不生成超出证据的细节结论。
- 算法演进图无明显时间错误：纵轴按论文 `published` / `updated` 日期排序，边只能由较新论文指向较早论文；有 key 时 LLM 标注的继承/对比关系限定在检索到的论文集合内，并强制时间方向。
- 优先考虑公开代码：`discover` 和 `survey` 会把有 GitHub 代码线索的论文优先展示。

## 常见问题

### 没有 API key 能做什么？

可以运行：

```powershell
python main.py search "topic"
python main.py discover "topic"
python main.py survey "topic"
```

GUI 中对应的「搜索」「发现 · 代码」「综述生成」「论文图谱」「笔记」视图也可用。

不能运行（需要 LLM）：

```powershell
python main.py research "topic"
python main.py analyze "arxiv_id"
python main.py read-paper "arxiv_id_or_pdf"
python main.py evaluate "idea"
python main.py chat
```

### arXiv 被限流怎么办？

- 降低 `SEARCH_TOP_K`。
- 避免短时间重复运行同一主题。
- 优先用 `survey --max-papers 6` 做小规模测试。
- 被限流后等待一段时间再运行。

### GitHub 代码关联不准怎么办？

自动关联依赖公开网页搜索，可能出现同名仓库误匹配。正式提交前应人工检查 `papers.json` 和报告中的 GitHub 链接。

### 演进图里为什么有的主题只有一两条分支？

技术分支来自 LLM 在检索到的论文集合内部的判断；论文较少或同质时分支自然较少。图中默认展示与主题最相关的若干篇（上限 22 篇）以保证可读性，其余论文仍会出现在综述报告和 `papers.json` 里。

## 主要命令汇总

```powershell
python main.py --help
python main.py search "diffusion models"
python main.py discover "diffusion models" --max-papers 10
python main.py evaluate "diffusion policy for manipulation"
python main.py survey "diffusion models" --max-papers 12
python main.py analyze "2301.00234" --focus "methodology"
python main.py read-paper "https://arxiv.org/abs/2301.00234"
python main.py research "large language model reasoning"
python main.py chat
python main.py gui
```
