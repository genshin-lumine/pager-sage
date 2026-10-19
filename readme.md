# Paper Sage

通过亲手构建论文问答工作台，逐步学习 LLM 应用、RAG 和 Agent 开发。

最终目标是基于 **Streamlit + LangChain + LangGraph**，实现按项目组织论文、限定检索范围、自动选择工作流程，以及可追溯到论文原文的证据回答。检索层计划采用 **Dense + BM25 双路召回 → RRF 排名融合 → Rerank 精排**。

**当前阶段：已完成 Day 12，正在构建论文 RAG。现有聊天页面支持手动提供摘录、结构化回答与异常处理。本地流程已能读取 PDF、逐页分块，将 800 / 100 配置的 51 个论文片段编码为 384 维向量，并保存 FAISS 索引、原文、来源及编码配置。正式模块 `retriever.py` 已能把中文问题编码后查询已有索引，返回带名次、分数和完整片段记录的候选列表。四组问题已检查，保留了排序不佳、漏检和相关但不足以回答的案例；自动检索尚未接入聊天页面，项目管理、混合检索和 Agent 工作流仍待实现。**

## 1. 学习路径

学习主线：

**模型调用 → 论文 RAG → 项目与检索范围 → 混合检索 → Agent 工作流 → 证据校验与评估。**

| 阶段 | 学习日 | 学习重点 | 阶段成果 |
| --- | --- | --- | --- |
| 第一阶段：模型与应用基础 | Day 01–07 | Python 环境、模型调用、消息、提示词、Streamlit、LangChain 基础、模块划分 | 能连续对话、可复现运行的基础应用 |
| 第二阶段：论文 RAG | Day 08–14 | PDF 解析、分块、Embedding、向量检索、上下文组装、基础引用 | 可以针对一篇论文提问，并查看回答依据 |
| 第三阶段：项目与范围管理 | Day 15–21 | 数据模型、持久化、文档生命周期、检索过滤、会话隔离 | 按项目管理论文，并限定参与问答的文档 |
| 第四阶段：混合检索 | Day 22–28 | Dense、BM25、RRF、Rerank、对照实验 | 能比较和解释各检索环节的效果与耗时 |
| 第五阶段：Agent 工作流 | Day 29–35 | 工具调用、LangGraph 状态与节点、路由、条件分支、循环终止 | 自动选择问答、总结、对比等流程 |
| 第六阶段：证据与工程完善 | Day 36–42 | 结构化证据、引用校验、原文定位、持久化、缓存、日志、评估 | 可演示、可维护的论文问答工作台 |

计划按每天约 **45–60 分钟**安排。这里的“天”是学习日，不是连续自然日；遇到基础知识缺口或调试问题，可以把一天拆成多次学习。

教学方式：学习者编写关键代码，助手负责讲解、拆解任务、检查和调试。每天开始时明确目标与验收标准，完成当天验收后收尾。

讲题约定：学习者表示不明白题意时，先解释场景与术语，再逐步给提示，保留学习者自行作答的机会；作答后再纠错和讲解。

建议节奏：10 分钟复盘、15 分钟讲解、25 分钟实践、10 分钟验证与记录。

## 2. 每日学习任务与状态

- `[x]`：已完成，已有代码、学习记录或验收结果支撑。
- `[ ]`：未完成，包括尚未开始的任务。
- 做了一部分但尚未达到验收标准的任务仍保留 `[ ]`，把已完成部分写入当天记录。

**当前进度：已完成 12 / 42 个学习日。下一次从 Day 13 开始。** 此数字只表示学习日进度，不代表项目功能完成比例。

Day 01 实际完成内容超过最初的“跑通页面”目标。下面已按实际进度调整，已学会的模型调用、会话状态和函数拆分统一记入 Day 01，后续安排在此基础上推进。

### 第一阶段：模型与应用基础

- [x] **Day 01｜从页面到多轮模型对话**：创建 Python 虚拟环境；运行 Streamlit；接入 DeepSeek；理解消息角色和返回值；实现聊天历史与清空；封装函数并拆分模块。验收：基础聊天与清空由学习者验证，模块拆分后的语法与引用关系检查通过。
- [x] **Day 02｜用 LangChain 接入 DeepSeek**：复验现有功能；理解模型对象、消息对象和调用接口；将 `llm.py` 中的模型调用改为 LangChain 实现。验收：页面仍能完成单轮问答、多轮追问和清空，能解释与直接 HTTP 调用的对应关系。
- [x] **Day 03｜提示词与上下文实验**：整理系统提示词；手动给模型一小段论文原文；比较有无原文时的回答。验收：保存一组对照结果，并解释提示词、外部资料和聊天历史分别起什么作用。
- [x] **Day 04｜结构化输出**：学习字典、JSON 和数据校验；定义包含回答、依据和证据不足说明的结果结构。验收：程序可以读取各字段，并识别缺字段或格式错误的结果。
- [x] **Day 05｜调用失败与输出长度**：理解异常传播；区分认证、限流、超时等常见失败；处理输出被截断的情况。验收：用模拟失败验证提示与历史记录状态，能区分“请求成功”和“回答生成完整”。
- [x] **Day 06｜让项目可复现运行**：整理依赖文件、配置示例和启动步骤；学习 Git 的基本提交方式。验收：独立环境的依赖检查、问答与清空均已通过，依赖与配置有明确记录；已完成本地提交 `f5d99e0` 并核对提交内容。
- [x] **Day 07｜基础阶段复盘与样例准备**：复盘输入到输出的数据流；选择可提取文字的论文；准备 3 个有原文依据的问题和 1 个论文未提供答案的问题。验收：固定 NeurIPS 2017 的 11 页版本，保存 4 条 JSON 样例及依据与页码；问答、追问、资料不足和清空会话均由学习者实际验证通过。

### 第二阶段：论文 RAG

- [x] **Day 08｜读取 PDF 并保留来源**：使用 `pypdf` 读取文本型 PDF，封装逐页读取函数；在页面记录中保留文档标识、来源和 PDF 页序号，学习其与印刷页码的区别。验收：11 页记录完整，三条样例原文与页码匹配；空白、混合页面及仅空白字符场景检查通过，空白页保留在结果中。
- [x] **Day 09｜文档分块与元数据**：学习分块长度与目标重叠，封装逐页分块函数，为片段保留项目、文档、来源、页码和标识。验收：对比 400 / 100 与 800 / 100 两套配置，得到 113 与 51 个片段；来源、长度、三条样例依据及空页面等边界检查通过。跨配置标识重复问题已在 Day 11 加入参数区分后处理。
- [x] **Day 10｜Embedding 与语义相似度**：区分聊天模型与向量模型；在 CPU 上生成中英文文本向量并比较余弦相似度。验收：理解 `(4, 384)` 的含义与归一化后的点积；完成三组问题对照并保留排序失败案例，能解释语义相关不等于有回答依据。
- [x] **Day 11｜建立向量索引**：片段标识加入分块长度和重叠量，保存八个片段字段；用 FAISS + JSON 保存向量索引与原文记录。验收：三套参数的标识无交叉重复；51 个 384 维向量可在新进程中重新加载并查询到原片段；保存数量检查和读取版本、维度、数量检查通过。
- [x] **Day 12｜完成 Dense 检索**：按索引保存的编码配置生成问题向量，封装 `retrieve_dense()`，返回带名次、分数和完整来源的候选列表。验收：四组问题的封装前后结果一致；三条有答案问题分别记录 Top-1、Top-3 命中情况，保留漏检和无答案案例；空问题、非法数量、空索引及返回数量限制检查通过。
- [ ] **Day 13｜跑通基础 RAG**：将检索片段、问题和回答要求组合后发送给模型；展示回答及对应片段。验收：论文问题能附上可核对的来源，证据不足时有明确说明。
- [ ] **Day 14｜建立 RAG 评估基线**：逐步扩充到约 20 个问题，覆盖事实问答、跨段落问题和无答案问题。验收：记录检索命中、回答正确性、引用支持情况和典型失败案例。

### 第三阶段：项目与范围管理

- [ ] **Day 15｜设计持久化数据模型**：整理 Project、Document、Chunk 的关系；为项目和文档元数据引入本地持久化，按项目、文档与索引版本规范索引目录命名。验收：能说明每个标识的用途，重启后数据关系仍可读取，索引目录不再依赖学习日命名。
- [ ] **Day 16｜创建与切换项目**：实现项目创建、列表与切换；展示当前项目的文档。验收：至少两个项目可以独立管理各自的论文。
- [ ] **Day 17｜文档导入生命周期**：将上传、解析、分块和索引串起来；处理重复上传和失败状态。验收：同一文件重复导入不会无意产生重复记录，失败可以定位到具体环节。
- [ ] **Day 18｜限定检索文档范围**：支持当前项目全部论文或选定论文；在检索层执行范围约束。验收：返回片段全部属于选定范围，并覆盖空选择的处理。
- [ ] **Day 19｜隔离项目与会话上下文**：设计切换项目或文档范围时的历史处理规则。验收：旧项目的对话与证据不会误用于新项目的问答。
- [ ] **Day 20｜删除文档与同步索引**：处理文档、片段、索引及缓存之间的更新关系。验收：删除文档后检索不到旧片段，其他文档仍可正常查询。
- [ ] **Day 21｜项目管理阶段验收**：构造内容相近但属于不同项目的论文样例；覆盖范围切换和删除场景。验收：能够复现并验证项目隔离与检索范围约束。

### 第四阶段：混合检索

- [ ] **Day 22｜BM25 关键词检索**：理解词项匹配；处理论文术语、模型名和分词；实现 BM25 基线。验收：展示它与 Dense 在同一组问题上的不同候选结果。
- [ ] **Day 23｜统一双路召回接口**：让 Dense 和 BM25 使用相同的文档范围，返回统一的片段标识和来源结构。验收：两路结果都满足范围约束，并可按标识去重。
- [ ] **Day 24｜实现 RRF 排名融合**：理解按名次融合而非直接混加原始分数；实现去重与融合排序。验收：用两份小型排名手算结果，并与程序输出核对。
- [ ] **Day 25｜加入 Rerank 精排**：为融合后的候选计算相关性；控制候选数量和最终返回数量。验收：展示精排前后顺序，并记录额外耗时。
- [ ] **Day 26｜观察完整检索流程**：记录召回、融合和精排各阶段的候选、排名及耗时。验收：能追踪一个最终证据片段如何进入答案上下文。
- [ ] **Day 27｜完成检索对照实验**：在同一评估集上比较 Dense、BM25、RRF 和加入 Rerank 后的结果；学习 Recall@K、MRR 等指标。验收：形成结果表，指出改善与退化的具体案例。
- [ ] **Day 28｜调整检索参数并复盘**：依据失败案例调整分块、候选数量等参数；保留独立问题用于复验。验收：能解释选择的配置及其效果、速度和资源开销之间的取舍。

### 第五阶段：Agent 工作流

- [ ] **Day 29｜把检索封装成工具**：学习工具名称、描述、参数与返回值；区分模型请求调用工具和程序实际执行工具。验收：模型可发出合法调用，执行结果能返回给模型继续回答。
- [ ] **Day 30｜LangGraph 的状态、节点和边**：定义最小状态结构；把已有检索问答流程拆成节点并连成图。验收：能运行图，并解释各节点读取、更新了哪些状态。
- [ ] **Day 31｜结构化意图路由**：区分论文问答、总结、对比及需要澄清的请求；把路由结果连接到条件分支。验收：用一组样例检查实际执行路径。
- [ ] **Day 32｜实现论文问答分支**：在图中接入范围受控的检索工具、证据整理与回答生成。验收：从请求到答案可查看完整执行路径与证据来源。
- [ ] **Day 33｜实现论文总结分支**：处理较长论文的分段总结与汇总；考虑全文覆盖。验收：总结包含研究问题、方法、结果和局限，并能定位相关来源。
- [ ] **Day 34｜实现论文对比分支**：按指定维度分别收集各论文证据，再组织比较结果。验收：对比双方都有来源，缺失信息会被标明。
- [ ] **Day 35｜受控重试与终止条件**：在证据不足时尝试改写问题再检索；设置次数上限与退出结果。验收：成功、无证据和工具失败三类路径都能结束，不出现无限循环。

### 第六阶段：证据与工程完善

- [ ] **Day 36｜统一结构化证据**：定义回答、结论与证据标识的对应关系，保留原文、页码、文档及片段标识。验收：回答引用只能关联到本次实际取回的证据。
- [ ] **Day 37｜校验引用与结论支持关系**：分别检查引用是否存在、页码是否正确、原文是否支持结论。验收：用错误引用和无依据结论样例验证检查方式，记录人工复核结果。
- [ ] **Day 38｜在界面查看原文证据**：展示证据卡片、引用片段及对应 PDF 页面入口。验收：从回答可以找到对应论文页和原文，能够人工核对。
- [ ] **Day 39｜会话与工作流持久化**：区分 Streamlit 会话状态和 LangGraph 检查点；设计会话、项目与工作流标识的关系。验收：重启后能恢复预期状态，恢复时仍遵守项目范围。
- [ ] **Day 40｜缓存、日志与耗时记录**：缓存适合复用的解析或模型资源；记录检索与模型调用的关键事件和耗时。验收：能解释缓存何时失效，并依据日志定位一项失败。
- [ ] **Day 41｜完成关键场景回归**：覆盖空文档、无答案、范围切换、重复上传、删除、接口失败和循环终止。验收：形成可重复执行的检查清单或必要测试，并修复发现的问题。
- [ ] **Day 42｜项目演示与复盘**：用固定论文和问题集完成端到端演示；更新架构、运行说明和评估结果；总结后续改进。验收：他人可按 README 运行，并理解项目能力及当前限制。

## 3. 当前项目与运行方式

### 当前文件结构（含本地练习文件）

```text
paper-sage/
├── app.py                    # Streamlit 页面、消息组装与聊天记录
├── llm.py                    # 调用 DeepSeek、检查结束原因，返回校验后的字典
├── answer_parser.py          # JSON 解析、必需字段与数据类型校验
├── pdf_loader.py             # PDF 逐页读取，返回保留来源的页面记录
├── chunker.py                # 逐页分块，保留项目、文档、页码及片段标识
├── index_store.py            # FAISS 索引与 JSON 元数据的保存、读取和基础检查
├── retriever.py              # 问题向量化、Dense 检索，返回分数与完整片段记录
├── practice_json.py          # 本地 JSON 解析练习，不调用模型
├── practice_errors.py        # 本地异常传播与历史记录练习，不调用模型
├── practice_pdf.py           # 打印 PDF 页面摘要，检查未提取到文字的页码
├── practice_chunks.py        # 比较两套分块配置，查看 GPU 样例所在片段
├── practice_embeddings.py    # 本地向量与相似度练习，当前保留耗电量问题
├── practice_index.py         # 为 51 个论文片段生成向量，建立并保存索引
├── practice_load_index.py    # 在新进程读取索引，检查向量与来源对应关系
├── practice_retrieval.py     # 调用正式检索函数，打印中文问题的候选与来源
├── requirements.txt          # 当前 Windows / Python 3.12 环境的依赖版本快照
├── readme.md                 # 学习路线、任务状态与每日学习记录
├── data/
│   ├── sample_questions.json # 3 条有答案样例与 1 条资料不足样例
│   ├── papers/
│   │   └── attention_is_all_you_need_neurips2017.pdf
│   └── indexes/              # 本地生成文件，不提交到 Git
│       └── day11_s800_o100/
│           ├── index.faiss
│           └── metadata.json
├── .gitignore
└── .streamlit/
    ├── secrets.toml.example  # 配置模板，仅含占位符，提交到 Git
    └── secrets.toml          # 本地 API Key 配置，不提交到 GitHub
```

当前使用 Python 3.12.6，通过 `langchain-deepseek` 提供的 `ChatDeepSeek` 调用 `deepseek-flash`。`ask_deepseek(messages, api_key)` 接收完整消息列表和密钥，使用 `response_format={"type": "json_object"}` 请求 JSON 输出。拿到响应后先检查 `finish_reason`，仅在其为 `stop` 时调用 `parse_answer()` 解析并校验，向页面返回字典。

当前客户端也可能在 `invoke()` 返回之前抛出长度限制或内容过滤异常，`llm.py` 将这两类异常转换为 `IncompleteAnswerError`，由页面显示提示。认证失败、限流、超时、连接失败、JSON 格式错误和字段校验失败也有相应处理。只有模型调用、结束状态检查和字段校验均通过后，才追加本轮聊天记录。

`max_tokens=1024` 限制输出 token 数，不等于整个上下文窗口的长度。`finish_reason="stop"` 表示正常结束或遇到指定停止序列，仍不能证明答案事实正确或证据充分。

当前结果结构：

| 字段 | Python 类型 | 用途 |
| --- | --- | --- |
| `answer` | `str` | 回答正文 |
| `evidence` | `list[str]` | 原文依据列表；资料不足时按提示词要求返回空列表 |
| `insufficient_reason` | `str` | 资料不足的原因；资料足够时按提示词要求返回空字符串 |

解析函数检查最外层是否为字典、三个字段是否存在，以及字段和依据列表元素的类型。页面读取字典后组装成 Markdown 字符串，用于展示和保存聊天记录；会话中目前保存的是展示文字，尚未单独保存结构化字段。字段之间的语义关系、引用真实性与结论支持情况仍需后续校验。

当前主要包版本如下。Day 08 新增 `pypdf`，Day 09 新增 `langchain-text-splitters`，Day 10 新增本地向量模型相关依赖，Day 11 新增 `faiss-cpu`；Day 10 保存的 95 个包版本未变：

| 包 | 当前版本 | 用途 |
| --- | --- | --- |
| `streamlit` | 1.65.0 | 页面交互与会话状态 |
| `langchain-deepseek` | 1.1.1 | DeepSeek 模型集成 |
| `langchain-core` | 1.6.10 | LangChain 消息与模型基础接口 |
| `langchain-text-splitters` | 1.1.3 | 按字符长度与分隔符递归分块 |
| `langchain-openai` | 1.7.2 | 当前 DeepSeek 集成使用的模型客户端基础实现 |
| `openai` | 3.31.0 | 兼容协议客户端及页面捕获的 API 异常类型 |
| `pypdf` | 6.21.1 | 本地 PDF 逐页读取与文本提取 |
| `sentence-transformers` | 6.1.0 | 加载向量模型并编码文本 |
| `torch` | 2.14.1+cpu | 本地 CPU 模型计算 |
| `transformers` | 5.20.1 | 向量模型底层实现与分词 |
| `numpy` | 2.5.4 | 向量数组与数值计算 |
| `faiss-cpu` | 1.15.1 | CPU 向量索引、相似度查询与索引文件读写 |

`ChatDeepSeek` 在本项目中连接 `https://api.deepseek.com`。代码直接使用的第三方包为 `streamlit`、`langchain-deepseek`、`openai`、`pypdf`、`langchain-text-splitters`、`sentence-transformers` 和 `faiss-cpu`（导入名为 `faiss`）；`json`、`pathlib` 属于 Python 标准库，无需额外安装。向量与索引练习使用本地计算，不调用 DeepSeek API。

`requirements.txt` 最初由项目 `.venv` 中的 `pip freeze` 导出，Day 08、09 补入 PDF 和分块依赖，Day 10 补入向量模型相关的 27 个包，Day 11 补入 `faiss-cpu==1.15.1`。目前共有 96 条指定版本记录，与当前环境快照一致，包括间接依赖及环境中的其他已安装包。它不是自动求解生成的锁文件，也不代表最小依赖集合。Day 06 的 66 条依赖已在本机新建的 Windows / Python 3.12.6 环境中验证；Day 08～11 在现有项目环境中验证新增功能，并通过 `pip check`，未重新创建独立环境。其他操作系统和 Python 版本尚未验证。

当前锁定的是 `torch==2.14.1+cpu`。新环境先从 PyTorch 官方 CPU 源安装这一版本，再安装完整快照；`pip freeze` 本身不记录下载源。以下三处环境准备命令均包含这一步。

### 本地启动（Windows / Git Bash）

先安装 Python 3.12，并准备自己的 DeepSeek API Key。以下命令均在项目根目录执行，逐条执行成功后再继续。首次准备环境时运行：

```bash
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install "torch==2.14.1+cpu" --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m pip check
```

`-r` 表示从文件读取依赖清单；`pip check` 检查声明的依赖是否缺失或版本不兼容，预期输出 `No broken requirements found.`。直接指定环境中的 Python 路径，无需先激活环境。

首次配置时，从模板创建本地配置。下面的 `-n` 表示已有目标文件时不覆盖：

```bash
cp -n .streamlit/secrets.toml.example .streamlit/secrets.toml
```

在编辑器中打开 `.streamlit/secrets.toml`，将占位符替换为自己的 API Key：

```toml
DEEPSEEK_API_KEY = "your-deepseek-api-key"
```

模板 `secrets.toml.example` 用于说明配置格式；应用通过 `st.secrets["DEEPSEEK_API_KEY"]` 读取实际配置。已有可用配置时沿用原文件。

启动页面：

```bash
./.venv/Scripts/python.exe -m streamlit run app.py
```

打开终端显示的 `Local URL`。已有虚拟环境和配置时，只需运行启动命令。

如果使用 PowerShell，安装和启动命令也可按以下形式执行；配置文件可在编辑器中从模板复制并填写：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch==2.14.1+cpu" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m streamlit run app.py
```

当前 `.gitignore` 已排除 `.venv/`、`__pycache__/`、`.streamlit/secrets.toml`、`practice_*` 和 `data/indexes/`。八个练习脚本均只在本地保留、未被 Git 跟踪，因此克隆仓库后不会自动获得它们；下方练习命令以本地已有对应文件为前提，Day 10～12 附完整示例便于重建。索引文件由练习生成，克隆后按下方步骤重新构建。忽略文件不等于删除文件，原有练习仍按维护约定保留。提交配置模板、依赖文件及正式模块，真实密钥文件和虚拟环境保留在本地；手动通过网页选择文件上传时也应排除它们。

### 在独立环境中验证（Windows / Git Bash）

项目已有可用的 `.streamlit/secrets.toml` 时，可以新建 `.venv-check` 验证依赖清单。此处独立的是 Python 包环境，运行时使用当前项目源码与本地配置。

Day 06 使用的 `.venv-check` 已在练习后清理，当前 `.gitignore` 中也已移除它的规则。若再次执行下面的验证，先在 `.gitignore` 中加入一行 `.venv-check/`，再新建环境。

```bash
py -3.12 -m venv .venv-check
./.venv-check/Scripts/python.exe -m pip install "torch==2.14.1+cpu" --index-url https://download.pytorch.org/whl/cpu
./.venv-check/Scripts/python.exe -m pip install -r requirements.txt
./.venv-check/Scripts/python.exe -m pip check
./.venv-check/Scripts/python.exe -m streamlit run app.py --server.port=8502
```

打开 `http://localhost:8502`，清空会话，在参考资料中粘贴：

```text
Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task
```

提交问题“参考资料中的模型在什么翻译任务上取得了多少 BLEU？”。预期回答包含 WMT 2014、英语到德语、28.4 BLEU，并展示原文依据；之后点击“清空会话”，确认聊天记录消失。Day 06 学习者已确认依赖检查、问答和清空均正常。

### 更新依赖记录

在项目环境中有意调整依赖并验证功能后，再更新快照。Git Bash 命令为：

```bash
./.venv/Scripts/python.exe -m pip freeze > requirements.txt
```

PowerShell 中使用明确的文件编码：

```powershell
.\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding utf8 requirements.txt
```

导出后检查文件差异，并在新环境中验证。若仍锁定 CPU 版 PyTorch，应保留上述从官方 CPU 源预先安装的步骤；依赖快照不包含模型权重，首次运行向量练习时还需下载模型。`Set-Content` 是 PowerShell 命令，不能直接在 Bash 中使用。

### Git 基本提交流程

编辑文件后，用 `git status --short` 查看状态；`git add <文件路径>` 把指定文件的当前内容加入暂存区；`git diff --cached` 查看待提交差异；`git commit -m "提交说明"` 把暂存区内容记录为本地版本。`git log -1 --oneline` 可查看最新提交。修改已暂存的文件后，需要再次 `git add` 才能把新修改纳入提交。

本地提交与上传 GitHub 是两个步骤，上传远端使用 `git push`。Day 06 学习者已完成本地提交 `f5d99e0`（完善异常处理并添加可复现运行配置），提交内容经助手核对。本次验收未检查远端上传状态。

### 当前能力与本地练习

当前聊天记录保存在 Streamlit 会话内存中，尚未实现长期持久化。页面支持手动粘贴论文摘录，并将资料与问题一起放入本次用户消息；系统提示词要求根据本次资料回答并给出原文依据。Day 03 已通过实际对照实验与修正后的复验：无资料时说明依据不足，有资料时正确回答并引用摘录，具体结果见文末。Day 08 的 PDF 读取函数与 Day 09 的分块函数目前由本地练习调用，尚未接入页面或自动检索；当前页面引用由模型生成，未经过程序校验。

Day 04 接入结构化输出后，学习者完成了有无资料两组实际问答。Day 05 通过本地模拟进一步验证：调用失败、输出未正常结束、JSON 或字段校验失败时显示对应提示，保留已有历史；正常回答只追加一问一答。模拟验证使用占位密钥，不代表重新验证了远端 API 的实时可用性。

运行本地解析练习：

```bash
./.venv/Scripts/python.exe practice_json.py
```

练习文件目前保留正常样例；缺字段和格式错误的实验操作、实际报错保存在 Day 04 学习记录中。

运行本地异常传播练习：

```bash
./.venv/Scripts/python.exe practice_errors.py
```

文件当前保留模拟正常返回的版本。将模拟函数中的 `return "模拟回答"` 替换为 `raise TimeoutError("模拟请求超时")`，可以重现失败时跳过历史追加的行为；每次重新运行脚本都从两条历史消息开始。该练习不会调用 API。

### 论文与问答样例（Day 07）

当前选用《Attention Is All You Need》的 [NeurIPS 2017 官方 PDF](https://papers.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)，共 11 页。如本地没有该文件，从这个地址下载，并保存为 `data/papers/attention_is_all_you_need_neurips2017.pdf`。

本地文件已与官方文件进行内容校验，大小为 569,417 字节，SHA-256 为：

```text
d87d482d5ae7960e2e43d7dd6d21377e60e73e8fce1bf2a01aff7aca8a08c537
```

本次样例的 `pdf_page` 表示从 1 开始的 PDF 页序号。不要直接套用其他版本的页码：例如，编码器层数的原句在这份文件的第 2 页，而在 [arXiv v7 的 15 页版本](https://arxiv.org/pdf/1706.03762v7)中位于第 3 页。

`data/sample_questions.json` 保存论文路径、官方来源及 `cases` 列表。每条样例包含问题、是否可回答、预期答案、原文摘录、页码和不足原因。预期答案可以概括，`evidence` 保存可对照原文的连续摘录。无答案样例使用 `answerable: false`、`evidence: []` 和 `pdf_page: null`；检查过哪些相关页写在不足原因中。

| 样例 | 问题 | 预期答案 | 支持答案的 PDF 页码 |
| --- | --- | --- | --- |
| q1 | 编码器由多少个相同层组成？ | 6 层 | 2 |
| q2 | 训练使用多少块 GPU，型号是什么？ | 8 块 NVIDIA P100 GPU | 7 |
| q3 | 位置编码使用哪两种函数？ | 正弦和余弦函数 | 6 |
| q4 | 模型训练实际消耗了多少 kWh 的电能？ | 无法依据论文确定；未报告实测能耗或足以确定耗电量的实际平均功率等数据。 | 无；相关内容核对位置为第 7～8 页 |

这些是人工核对的基础样例，供后续 PDF 解析与检索对照使用。它们尚未被应用自动加载，也不构成完整的 RAG 评估集；页面目前仍读取手动粘贴的资料。样例文件的字段结构与模型回答不同，不应直接传入 `parse_answer()`。

Day 07 页面回归使用 q2 的 GPU 原文摘录：先问 GPU 数量，再保留资料与历史追问型号；随后清空历史、保留资料，询问实际耗电量；最后再次清空。学习者确认回答正确、追问后保留两轮记录、资料不足时依据显示“无”，以及清空后聊天记录消失。清空按钮只清理历史，参考资料输入框仍保留内容。

### PDF 逐页读取与空文本检查（Day 08）

准备好上述 PDF 并安装依赖后，在项目根目录运行：

```bash
./.venv/Scripts/python.exe practice_pdf.py
```

练习使用 `load_pdf_pages(pdf_path, document_id)` 读取文件。该函数位于 `pdf_loader.py`，接收文件路径字符串与文档标识，返回一个列表，每条记录包含：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `document_id` | `str` | 由调用方传入的文档标识，本次练习使用固定名称。 |
| `source` | `str` | 由调用方传入的 PDF 路径。 |
| `pdf_page` | `int` | 从 1 开始的 PDF 页面序号。 |
| `text` | `str` | 本页提取到的文字，可能为空字符串。 |

页面记录当前保存在内存中，没有写入数据库或解析结果文件。函数保留全部页面，练习脚本负责打印摘要及空文本提示。

当前论文应得到 11 条页面记录，页码从 1 到 11，空页检查输出：

```text
未提取到可用文字的页码： []
```

`empty_pages` 收集的是去除首尾空白后文本为空的页面编号。因此 `[]` 表示没有发现这类页面，不表示“没有编码”或“没有信息”。有非空文本也不能证明整页内容、图表、公式和阅读顺序都提取完整。若所有页面都没有可用文字，练习脚本会打印检查原文件的提示；当前检查不能自动判断是扫描件、空白文档还是其他解析问题，尚未接入 OCR。

需要查看具体文本时，可在练习脚本获得 `pages` 后打印 `pages[1]["text"]`，它对应 PDF 第 2 页。`pdf_page` 记录文件中的页面位置；页面上印刷的页码可能因封面、前言或编号方式而不同。即使某页为空，也保留它的记录与原页序号。

三条有答案样例的原文已在第 2、7、6 页的提取结果中找到；核对时仅忽略空格与换行差异。临时 PDF 检查还覆盖全空白文件、空白页与正文页混合的文件，以及单独构造的仅空白字符记录。检查脚本与临时 PDF 未保存为项目文件。

### 文档分块与来源核对（Day 09）

在项目根目录运行分块对照练习：

```bash
./.venv/Scripts/python.exe practice_chunks.py
```

`chunker.py` 提供 `split_pages(pages, project_id, chunk_size=800, chunk_overlap=100)`。输入是 PDF 读取函数返回的页面列表，以及调用方提供的项目标识。函数逐页调用 `RecursiveCharacterTextSplitter`，返回字典列表；每个片段只属于一个 PDF 页面，并包含：

| 字段 | 用途 |
| --- | --- |
| `project_id` | 调用方传入的项目标识，当前练习固定为 `demo_project`。 |
| `document_id` | 沿用来源页面的文档标识。 |
| `chunk_size` | Day 11 新增，本次调用传入的分块长度。 |
| `chunk_overlap` | Day 11 新增，本次调用传入的目标重叠量。 |
| `source` | 沿用来源页面的 PDF 路径。 |
| `pdf_page` | 沿用从 1 开始的 PDF 页序号。 |
| `chunk_id` | 由项目、文档、分块长度、重叠量、页码和页内片段编号组合的标识。 |
| `text` | 当前片段的字符串内容。 |

`length_function=len` 按 Python 字符数计算长度。`chunk_size` 是上限，片段可以更短；`chunk_overlap` 是目标重叠量，实际不保证相邻片段恰好重复指定字符数。当前使用默认分隔符顺序：段落、换行、空格，最后按字符切分，具体行为见 [LangChain 官方分块说明](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter)。

当前固定论文与依赖版本下的结果：

| 配置 | 分块长度 | 目标重叠 | 片段总数 | 最短字符数 | 最长字符数 | 第 7 页包含 `P100` 的片段数 |
| --- | --- | --- | --- | --- | --- | --- |
| A | 400 | 100 | 113 | 112 | 400 | 1 |
| B | 800 | 100 | 51 | 198 | 797 | 2 |

本次仅改变分块长度。A 组总片段数更多；B 组的两个相邻片段都包含 GPU 原句，是重叠造成的证据重复。两组都完整保留 GPU 数量和型号，不能仅凭总片段数或关键词匹配数判断检索效果。此练习中的 `P100` 筛选用于人工核对；Day 11 另用 B 组建立了向量索引。

两套配置的每个片段都能在标注页的提取文本中直接找到，三条有答案样例也都在相应页的片段中保留。空白或仅含空白字符的页不生成片段，但输入页面记录不被修改，后续片段仍沿用原页码。项目字段目前只是元数据，尚未实现项目创建、切换或检索过滤。

**当前片段标识的适用范围**：Day 11 已将标识改为 `项目:文档:s长度:o重叠:p页码:c页内编号`。400 / 100、800 / 100、800 / 50 三套配置的标识分别无重复，彼此也无交集。新增配置字段直接读取函数参数，而非从 PDF 页面字典取值。此标识仍以固定文档内容与分块规则为前提；文档内容或算法变化时需进一步引入版本。FAISS 当前按向量位置对应片段记录，不会自动按字符串 `chunk_id` 去重或管理更新。

### 文本向量与相似度练习（Day 10）

使用 [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) 处理中文问题和中英文候选文本，通过 `SentenceTransformer` 在 CPU 上运行。它输出 384 维向量：384 表示数值数量，不是单词或 token 数量；模型将整段文本编码为一个向量。聊天模型负责生成回答，向量模型在这个练习中负责生成用于比较的数值表示。

首次运行需要联网下载模型文件，后续可复用本机缓存；向量计算无需新增 API Key。`device="cpu"` 指定计算设备，当前安装的是 CPU 版 PyTorch，不能仅改为 `"cuda"` 就使用 NVIDIA GPU，还需兼容硬件、驱动及 CUDA 版 PyTorch。少量文本先使用 CPU；本次没有进行 GPU 性能测试。

`practice_embeddings.py` 当前仅保留耗电量问题。若克隆仓库后没有该文件，可在根目录新建同名文件，写入下面的完整练习代码：

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "intfloat/multilingual-e5-small",
    device="cpu",
)

texts = [
    "query: 模型训练实际消耗了多少千瓦时的电能？",
    "passage: We trained our models on one machine with 8 NVIDIA P100 GPUs.",
    "passage: The encoder is composed of a stack of N = 6 identical layers.",
    "passage: 今天晚饭准备吃番茄鸡蛋面。",
]

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

print("结果类型：", type(embeddings))
print("结果形状：", embeddings.shape)
print("第一个向量的前 5 个数：", embeddings[0][:5])

question_vector = embeddings[0]
for index in range(1, len(texts)):
    passage_vector = embeddings[index]
    score = question_vector @ passage_vector
    print("候选片段：", texts[index])
    print("相似度：", round(float(score), 4))
```

运行命令：

```bash
./.venv/Scripts/python.exe practice_embeddings.py
```

输入列表第一条是带 `query: ` 前缀的问题，其余是带 `passage: ` 前缀的候选；这两个前缀是本次模型的检索约定，中文也保留英文前缀。返回 `numpy.ndarray`，四条文本对应形状 `(4, 384)`。归一化把每个向量的数学长度变为 1，不改变维度；此时一维数组的点积 `@` 等于余弦相似度。循环跳过问题自身，只比较问题与三个候选。

对照时保持候选不变，每次只替换第一条的问题，依次使用“训练这个模型使用了什么硬件？”、“编码器由多少个相同的层堆叠而成？”和“模型训练实际消耗了多少千瓦时的电能？”。三组实际结果保存在文末 Day 10 记录中，硬件与层数问题的第一名符合预期，耗电量问题的第一名却是晚饭片段。

相似度表示模型计算出的语义匹配程度，不是答案正确率，也不能单独判断证据是否充分。即使手动选回 GPU 片段，它仍未提供实际耗电量，无法回答 kWh。保留这个失败案例，后续扩大样例检查；此练习只使用三个手写候选，Day 11 另行建立整篇论文的索引，仍未实现自动判断可回答性。模型输入上限为 512 tokens，不能把分块的字符数上限直接视为 token 上限。

### 向量索引保存与重新加载（Day 11）

本地方案使用 FAISS 的 `IndexFlatIP` 保存向量，以 JSON 保存按相同顺序排列的片段记录。它逐一比较索引中的向量，按内积查询；问题和片段向量均归一化时，内积等于余弦相似度。当前索引包含 51 个 384 维、`float32` 类型的向量，来源为 800 / 100 配置的整篇论文片段。

编码前已用该模型的 tokenizer 检查全部 51 个输入，包括 `passage: ` 前缀和特殊标记：长度为 63～268 tokens，没有超过 512 tokens。这个检查针对当前固定论文和配置，不能代替后续导入新文档时的长度检查。

`index_store.py` 提供两个接口：

| 接口 | 行为 |
| --- | --- |
| `save_index(index, chunks, output_dir, embedding_config)` | 先检查向量与片段数量相同，创建目录，写入 `index.faiss` 和 `metadata.json`。 |
| `load_index(output_dir)` | 读取 JSON 和 FAISS 文件，检查格式版本为 1、维度一致、向量与片段数量一致，返回 `(index, metadata)`。 |

JSON 保存 `schema_version`、`dimension`、`embedding_config`、`chunks`。编码配置包含模型名、归一化开关及问题和资料前缀；`chunks` 保留八个片段字段和原列表顺序。FAISS 中位置 `i` 的向量对应 `chunks[i]`，不能独立重排其中一份。

克隆仓库后，先按前文安装依赖、准备论文，再在根目录建立以下两个本地练习文件。模型首次使用需要下载，之后可复用缓存。保存目录固定为 `data/indexes/day11_s800_o100`。

将下面代码保存为 `practice_index.py`：

```python
import faiss
from sentence_transformers import SentenceTransformer
from index_store import save_index

from pdf_loader import load_pdf_pages
from chunker import split_pages


pdf_path = "data/papers/attention_is_all_you_need_neurips2017.pdf"
document_id = "attention_is_all_you_need_neurips2017"

pages = load_pdf_pages(pdf_path, document_id)

chunks = split_pages(
    pages,
    project_id="demo_project",
    chunk_size=800,
    chunk_overlap=100,
)

passage_texts = []

for chunk in chunks:
    passage_texts.append("passage: " + chunk["text"])

model = SentenceTransformer(
    "intfloat/multilingual-e5-small",
    device="cpu",
)

embeddings = model.encode(
    passage_texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)

embeddings = embeddings.astype("float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

print("片段数量：", len(chunks))
print("向量形状：", embeddings.shape)
print("索引维度：", index.d)
print("索引中的向量数量：", index.ntotal)

embedding_config = {
    "model_name": "intfloat/multilingual-e5-small",
    "normalize_embeddings": True,
    "passage_prefix": "passage: ",
    "query_prefix": "query: ",
}

output_dir = "data/indexes/day11_s800_o100"

save_index(
    index=index,
    chunks=chunks,
    output_dir=output_dir,
    embedding_config=embedding_config,
)

print("索引和片段记录已保存到：", output_dir)
```

运行一次以构建并保存索引：

```bash
./.venv/Scripts/python.exe practice_index.py
```

预期输出片段数 51、向量形状 `(51, 384)`、索引维度 384、索引数量 51，并生成 `index.faiss` 与 `metadata.json`。重复运行会重新编码并覆盖该目录内的两个同名文件；不同配置需选择不同保存目录，片段标识不同并不能防止文件被同路径覆盖。

将下面代码保存为 `practice_load_index.py`：

```python
from index_store import load_index


output_dir = "data/indexes/day11_s800_o100"

index, metadata = load_index(output_dir)
chunks = metadata["chunks"]

print("读取的向量数量：", index.ntotal)
print("读取的片段数量：", len(chunks))
print("读取的索引维度：", index.d)

# 取出索引中的第三个向量，作为这次检查的查询向量。
probe_position = 2
probe_vector = index.reconstruct(probe_position)

# FAISS 接收二维数组：一行代表一条查询。
query_vectors = probe_vector.reshape(1, -1)

scores, positions = index.search(query_vectors, k=1)

matched_position = int(positions[0][0])
matched_chunk = chunks[matched_position]

print("用于查询的片段：", chunks[probe_position]["chunk_id"])
print("匹配到的片段：", matched_chunk["chunk_id"])
print("相似度：", round(float(scores[0][0]), 4))
print("来源：", matched_chunk["source"])
print("PDF 页码：", matched_chunk["pdf_page"])
```

在新的 Python 进程中运行：

```bash
./.venv/Scripts/python.exe practice_load_index.py
```

这个脚本直接读取已保存的索引与元数据，不重新读取 PDF 或调用 Embedding 模型。`reconstruct(2)` 取出第三个向量，`reshape(1, -1)` 将其变为一行查询，`search(..., k=1)` 返回分数与位置，再据位置读取片段记录。

实际结果为 51 个向量、51 条记录、384 维，查询与匹配标识同为 `demo_project:attention_is_all_you_need_neurips2017:s800:o100:p1:c3`，相似度显示 1.0，来源是固定论文的 PDF 第 1 页。这验证索引重新加载后可搜索，并能连接原文记录；中文用户问题的检索与答案所在片段命中检查见下节 Day 12。

**当前边界**：读取函数只做格式版本、维度和数量检查，不能发现同数量记录被重新排列或替换，也不自动核验模型配置与向量是否匹配。两个文件须配套保存，目前没有事务写入或完整性校验；模型仓库 revision 尚未固定。当前示例只有一份论文的索引，没有增量更新、删除同步、范围过滤或页面集成。

### 中文问题的 Dense 检索（Day 12）

`retriever.py` 提供 `retrieve_dense(question, model, index, metadata, top_k=3)`。调用方先通过 `load_index()` 加载配套文件，再根据 `metadata["embedding_config"]["model_name"]` 初始化模型；函数复用传入的模型与索引，每次只编码新的问题。查询使用保存的 `query_prefix` 和归一化配置，生成二维 `float32` 数组，通过 FAISS 返回的位置取出原文记录。

返回值是按相似度从高到低排列的列表，每条结果包含：

| 字段 | 含义 |
| --- | --- |
| `rank` | 从 1 开始的名次。 |
| `score` | Python 浮点数，保留原始精度；展示时才取小数位。 |
| `chunk` | 完整片段字典，包含项目、文档、分块参数、来源、PDF 页码、片段标识和原文。 |

`question` 约定为字符串，去除首尾空白后不能为空；`top_k` 必须为正整数，布尔值也被拒绝。空索引返回空列表；请求数量超过索引数量时按实际数量返回。函数负责返回数据，打印由练习脚本负责，后续页面可复用相同接口。

先按 Day 11 步骤生成本地索引，再将以下代码保存为 `practice_retrieval.py`：

```python
from sentence_transformers import SentenceTransformer
from retriever import retrieve_dense

from index_store import load_index


output_dir = "data/indexes/day11_s800_o100"
index, metadata = load_index(output_dir)

embedding_config = metadata["embedding_config"]

model = SentenceTransformer(
    embedding_config["model_name"],
    device="cpu",
)

question = "模型训练实际消耗了多少千瓦时的电能？"

results = retrieve_dense(
    question=question,
    model=model,
    index=index,
    metadata=metadata,
    top_k=3,
)

print("结果类型：", type(results))
print("结果数量：", len(results))

for result in results:
    chunk = result["chunk"]

    print()
    print("排名：", result["rank"])
    print("相似度：", round(result["score"], 4))
    print("片段标识：", chunk["chunk_id"])
    print("来源：", chunk["source"])
    print("PDF 页码：", chunk["pdf_page"])
    print("片段原文：", chunk["text"])
```

运行：

```bash
./.venv/Scripts/python.exe practice_retrieval.py
```

当前耗电量问题返回 3 条记录，依次为 `p7:c4`、`p8:c3`、`p8:c1`，分数显示为 0.8284、0.8238、0.8130。这里的短标识省略了共同前缀 `demo_project:attention_is_all_you_need_neurips2017:s800:o100:`。这些片段涉及硬件、训练时间和 FLOPs，人工核对后仍无法确定实际耗电量；检索函数本身不判断能否回答。

本次实际使用的问题原句如下，与 `data/sample_questions.json` 中的问法不完全相同，复测时应固定表述；人工依据沿用样例集。

| 问题原句 | Top-1 命中 | Top-3 命中 | 核对结果 |
| --- | --- | --- | --- |
| 训练这个模型使用了什么硬件？ | 否 | 是 | 第 2 名 `p7:c4` 包含 8 块 NVIDIA P100 GPU 的依据。 |
| 编码器由多少个相同的层堆叠而成？ | 是 | 是 | 第 1 名 `p2:c7` 包含 6 层的依据。 |
| 论文中的位置编码使用了什么函数？ | 否 | 否 | 前三条均为参考文献；已核对的原文在第 6 页，属于检索漏检。 |
| 模型训练实际消耗了多少千瓦时的电能？ | 不适用 | 不适用 | 已知无答案样例；相关候选不提供实际耗电量或足够的计算依据。 |

Top-3 命中指前三条中至少一条含有直接依据；这三条有答案样例只能作为当前小样本检查，不能代表整体检索质量。无答案样例不计入有答案问题的命中统计。相似度不是答案正确的概率，当前函数没有分数阈值、自动拒答或引用校验。

Day 12 没有新增依赖，现有环境的 `pip check` 通过。真实问题复核与边界检查见文末学习记录。调用方仍须保证模型、索引和元数据配套；目前未实现额外的模型身份校验、长问题输入检查或元数据完整性校验。自动检索接入问答页面安排在 Day 13；索引目录按项目、文档与版本命名安排在 Day 15，当前目录仍为 `day11_s800_o100`。

### 学习参考

- [Streamlit 安装与启动](https://docs.streamlit.io/get-started/installation/command-line)
- [Streamlit Session State](https://docs.streamlit.io/develop/concepts/architecture/session-state)
- [Streamlit 按钮行为](https://docs.streamlit.io/develop/concepts/design/buttons)
- [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
- [Streamlit 本地配置与密钥](https://docs.streamlit.io/develop/concepts/connections/secrets-management)
- [Python 3.12 虚拟环境](https://docs.python.org/3.12/library/venv.html)
- [pip freeze](https://pip.pypa.io/en/stable/cli/pip_freeze/)
- [pip check](https://pip.pypa.io/en/stable/cli/pip_check/)
- [Git 暂存文件](https://git-scm.com/docs/git-add)
- [Git 本地提交](https://git-scm.com/docs/git-commit)
- [DeepSeek API 入门](https://api-docs.deepseek.com/)
- [DeepSeek JSON Output](https://api-docs.deepseek.com/zh-cn/guides/json_mode/)
- [DeepSeek 错误码](https://api-docs.deepseek.com/quick_start/error_codes/)
- [DeepSeek Chat Completions 与结束原因](https://api-docs.deepseek.com/api/create-chat-completion/)
- [Python 3.12 JSON 解析](https://docs.python.org/3.12/library/json.html)
- [pypdf 安装](https://pypdf.readthedocs.io/en/stable/user/installation.html)
- [pypdf 文本提取及限制](https://pypdf.readthedocs.io/en/stable/user/extract-text.html)
- [LangChain 递归字符分块](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter)
- [Sentence Transformers 安装](https://sbert.net/docs/installation.html)
- [SentenceTransformer 参数与编码接口](https://sbert.net/docs/package_reference/sentence_transformer/model.html)
- [multilingual-e5-small 模型、前缀与相似度说明](https://huggingface.co/intfloat/multilingual-e5-small)
- [PyTorch 安装与 CPU / CUDA 环境](https://pytorch.org/get-started/locally/)
- [FAISS 向量格式、索引添加与查询](https://github.com/facebookresearch/faiss/wiki/Getting-started)
- [FAISS 索引类型与 IndexFlatIP](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [FAISS 内积与余弦相似度](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)
- [FAISS 索引文件读写](https://github.com/facebookresearch/faiss/wiki/Index-IO,-cloning-and-hyper-parameter-tuning)
- [Attention Is All You Need：本项目使用的 NeurIPS 2017 版本](https://papers.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [LangChain 官方学习入口](https://docs.langchain.com/oss/python/learn)
- [LangChain 的 ChatDeepSeek 集成](https://docs.langchain.com/oss/python/integrations/chat/deepseek)
- [LangGraph 工作流与 Agent](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [RRF 排名融合](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)
- [检索与 Rerank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)

## 4. README 长期维护约定

这份 README 同时作为项目说明和持续学习记录。每次学习开始时先阅读当前进度与最后一条学习记录，结束时更新以下内容：

1. **更新任务状态**：完成验收的学习日改为 `[x]`，未完成的保持 `[ ]`；同步修改完成数量和下一学习日。
2. **更新当前能力**：只有代码已经实现并有验证依据的功能，才写入“当前已完成”；计划中的能力继续保留为目标。
3. **追加当天记录**：在文末按时间顺序追加新条目，写明日期、任务、改动、理解的概念、问题及解决过程、验证方式和下一步。
4. **保留学习过程**：不要用新一天的记录覆盖旧记录；同一学习日分多次完成时，保留每次日期和进展。
5. **区分验证方式**：明确是学习者实际运行确认、代码检查，还是测试结果；未验证的事项如实写出。
6. **按实际调整计划**：提前完成或临时补课时，调整尚未开始的安排，并在当天记录说明原因。每次课明确结束条件，避免不断追加任务。
7. **同步运行说明**：变更依赖、配置、目录结构或启动命令后，同步修改本文件。示例配置只使用占位符。
8. **练习文件清理提醒（2026-10-18 约定）**：暂时保留 `practice_json.py`、`practice_errors.py` 和 `practice_pdf.py`。后续相关功能迁移或阶段验收时，检查它们是否仍有练习、验证或调用用途；确认不再需要时，主动提醒学习者删除，由学习者执行清理，不自动删除。尤其要先确认 `practice_pdf.py` 中的空文本检查已有正式功能或测试承接。实际清理后同步更新文件结构和运行说明，保留历史学习记录。

后续协作时可以直接说明：“先阅读 `readme.md`，从下一项未完成任务继续；今天结束后更新任务状态，并在文末追加学习记录。”

### 后续每日记录模板

复制以下模板到学习记录末尾。未完成或未验证的内容保留未完成状态，并说明下一次从哪里继续。

```markdown
### Day XX｜YYYY-MM-DD｜学习主题

- 当天目标：
- 本次状态：已完成 / 未完成
- 实际用时：未记录时注明“未记录”

#### 完成的任务

- [x] 已完成并有依据的任务
- [ ] 尚未完成的任务

#### 代码与文档改动

- 涉及的文件、改动及原因。

#### 学到的概念

- 用自己的话说明关键概念。

#### 问题与解决过程

- 现象 → 原因 → 修改 → 验证结果。

#### 验证与遗留事项

- 记录实际运行、代码检查或测试结果。
- 未完成、未验证的内容及下一次的起点。

#### 下一次学习

- 下一项任务、预期成果与验收标准。
```

## 5. 每日学习记录

### Day 01｜2026-10-13｜从第一个页面到多轮聊天应用

- **当天目标**：配置环境，亲手运行第一个 Streamlit 页面，理解输入到输出的基本过程。
- **本次状态**：已完成。
- **实际用时**：未记录。
- **计划调整**：当天继续完成了真实模型调用、多轮对话和函数拆分，已将这些成果统一计入 Day 01；Day 02 从 LangChain 模型调用开始。
- **当天剩余必做任务**：0。

#### 完成的任务

- [x] 使用 Python 3.12 创建项目虚拟环境，并安装、运行 Streamlit。
- [x] 创建标题、问题输入框、提交按钮、空输入提示和输入展示；完成字符数统计练习。
- [x] 将单纯显示输入升级为请求 DeepSeek，并展示模型回答。
- [x] 在本地 `secrets.toml` 中配置 API Key，通过 `.gitignore` 排除密钥文件与虚拟环境。
- [x] 理解 `system`、`user`、`assistant` 三种消息角色和提示词的作用。
- [x] 理解 `choices[0]["message"]["content"]` 如何提取一条回答的正文。
- [x] 用 `st.session_state.messages` 保存历史，并把历史与新问题一起发送给模型。
- [x] 展示多轮聊天记录，实现并修复清空会话按钮。
- [x] 理解函数参数、返回值与调用次数，封装 `ask_deepseek(messages, api_key)`。
- [x] 将模型请求函数移动到 `llm.py`，在 `app.py` 中导入并调用。
- [x] 创建本 README，整理学习路线、每日任务状态与长期维护方式。

#### 代码与文档改动

| 文件 | 当前职责 |
| --- | --- |
| `app.py` | 页面交互、输入检查、系统提示词与历史消息组装、聊天记录保存与展示、错误提示 |
| `llm.py` | 接收完整消息和 API Key，发送 DeepSeek 请求，检查响应并返回回答字符串 |
| `.gitignore` | 排除虚拟环境、Python 缓存和本地密钥配置 |
| `.streamlit/secrets.toml` | 仅在本地保存模型 API 配置，不提交到仓库 |
| `readme.md` | 记录项目目标、学习计划、任务状态和每日学习过程 |

当前一次请求的数据流：

**页面输入 → 组装系统提示词、历史和新问题 → 调用 `ask_deepseek` → 解析响应 → 返回回答 → 保存本轮消息 → 展示对话。**

#### 学到的概念

- **界面、模型调用与 RAG**：输入框和输出展示只构成交互界面；模型调用增加了生成能力；RAG 还需要读取资料、检索证据并把相关内容提供给模型。
- **消息角色**：`system` 设置整体回答要求，`user` 表示用户请求，`assistant` 表示模型已有回答。它们共同构成本次请求的上下文。
- **回答提取**：`choices[0]` 是结果列表中的第一项，不是回答的第一句话；`content` 是这条消息实际返回的正文。触及生成长度限制时，正文仍可能被截断。
- **多轮对话**：保存聊天记录、把记录发送给模型、在页面展示记录是三件独立的事。模型需要在本次请求中收到相关历史。
- **Streamlit 状态与执行顺序**：交互通常会触发脚本重新运行；会话状态用于跨重新运行保存数据；按钮只在自身点击触发的运行中返回 `True`。
- **函数与模块**：函数通过参数接收数据，通过 `return` 交还结果；模块让模型请求与页面职责分开。`llm.py` 中的函数不直接读写 Streamlit 状态。

#### 问题与解决过程

| 遇到的问题 | 原因与处理 |
| --- | --- |
| 首次启动要求输入邮箱 | 属于可选的邮件订阅提示，留空按回车即可继续。 |
| `Ctrl+C` 后停留在 `Stopping...` | 检查发现默认端口已关闭，但对应进程仍存活；结束了这次已核实的 Streamlit 进程。具体阻塞在哪个退出步骤尚未定位。 |
| 页面只把输入重新显示出来 | 最初版本仅实现界面；随后加入 API 请求和返回正文提取，接通真实模型回答。 |
| 模型不能利用上一轮问题 | 请求只包含系统提示词和本次问题；改为保存历史，并在下一次请求中一并发送。 |
| 清空会话后旧记录还在 | 清空按钮嵌套在提交按钮分支里，点击清空时外层分支不执行；将清空按钮移到顶层，并放在历史展示循环之前。 |
| 封装函数后返回结果异常 | 函数缺少 `return`，且一次提交调用两次，之后把历史覆盖成了 `[None]`；改为一次调用接收返回值，再由页面追加完整的一轮消息。 |
| 函数传入参数却仍依赖页面 | 原函数没有使用 `messages` 参数，而是读取外部问题和会话状态；改为显式传入 `messages` 与 `api_key`。 |

#### 验证与遗留事项

- 学习者已确认 DeepSeek 能正常回答，页面可以显示两轮记录，并在修正按钮位置后完成清空会话练习。
- 函数封装检查确认：存在返回值，每次有效提交的代码路径只调用一次，函数内部不再引用 Streamlit。
- 最后的模块拆分检查确认：`app.py` 与 `llm.py` 语法解析通过，页面通过 `from llm import ask_deepseek` 导入函数，模型函数只定义在 `llm.py` 中。
- 当前没有阻塞后续学习的问题。模块拆分后的完整交互复验安排在 Day 02 开始时进行；现有代码检查不等同于重新发起真实 API 调用。
- 论文读取、RAG、长期会话保存和 Agent 工作流属于后续未完成任务，状态已在每日任务中列明。

#### 下一次学习

**Day 02：用 LangChain 接入 DeepSeek。**

先简短复盘并复验现有应用，再学习 LangChain 的模型与消息接口，修改 `llm.py` 中的调用实现。

验收标准：单轮问答、多轮追问和清空会话仍然可用；能够解释 LangChain 调用与目前 HTTP 请求的对应关系。完成验收并追加当天记录后结束当次学习。

### Day 02｜2026-10-14｜用 LangChain 接入 DeepSeek

- **当天目标**：理解 LangChain 的模型对象、消息对象和调用接口，替换模型调用实现，并保留单轮问答、多轮追问与清空会话功能。
- **本次状态**：已完成。
- **实际用时**：未记录。
- **当天剩余必做任务**：0。

#### 完成的任务

- [x] 复验 Day 01 模块拆分后的应用，学习者确认原版功能正常。
- [x] 在项目虚拟环境中安装 `langchain-deepseek`，验证 `ChatDeepSeek` 可以导入。
- [x] 使用 `ChatDeepSeek(...)` 配置 DeepSeek 模型、接口地址、密钥和调用参数。
- [x] 将 `llm.py` 中的 Requests 请求替换为 `model.invoke(messages)`。
- [x] 理解返回的 `AIMessage` 对象，通过 `response.content` 向页面返回正文字符串。
- [x] 保留 `ask_deepseek(messages, api_key)` 的函数接口，继续由页面组装历史、保存消息和展示回答。
- [x] 将页面的异常处理改为 `APITimeoutError` 和 `APIError`，先捕获具体的超时错误。
- [x] 按单轮回答、多轮追问和清空会话的验收步骤验证，学习者确认运行通过。
- [x] 完成模型初始化、实际调用、消息对象及函数返回值的理解检查。
- [x] 更新 README 的任务状态、当前依赖、安装命令，并追加 Day 02 学习记录。

#### 代码与文档改动

| 文件 | 改动及原因 |
| --- | --- |
| `llm.py` | 使用 `ChatDeepSeek` 和 `invoke` 完成模型调用，通过消息对象读取正文；对页面继续提供相同的函数参数与字符串返回值。 |
| `app.py` | 把 Requests 异常类替换为客户端 SDK 的 API 异常类，使请求失败时仍能显示错误提示。 |
| `readme.md` | 将 Day 02 标记为完成，进度更新为 2 / 42；同步当前技术栈、安装依赖与下一学习日。 |

当前模型调用的数据流：

**页面组装消息 → `ask_deepseek(messages, api_key)` → 创建 `ChatDeepSeek` 对象 → `model.invoke(messages)` → 返回 `AIMessage` → 提取 `response.content` → 页面保存并展示。**

#### 学到的概念

| 原来的 HTTP 调用 | LangChain 实现 | 对应含义 |
| --- | --- | --- |
| URL、请求头中的密钥及 JSON 中的模型参数 | `ChatDeepSeek(...)` 的配置参数 | 创建并配置模型客户端对象 |
| `requests.post(...)` | `model.invoke(messages)` | 发起模型调用并等待结果 |
| 解析后的响应 JSON | `AIMessage` | 表示模型返回消息的数据结构 |
| `data["choices"][0]["message"]["content"]` | `response.content` | 读取本次纯文本回答的正文 |

- 创建 `ChatDeepSeek` 对象用于初始化客户端及调用配置；本次代码中真正发送模型请求的是 `invoke`。
- 原有的 `role` / `content` 字典列表可以作为输入，LangChain 会转换为对应的消息对象。
- `invoke` 返回 `AIMessage`，而我们自己的 `ask_deepseek` 返回其中的正文字符串，两者的返回类型不同。
- 多轮上下文仍由页面通过 `messages` 传入。接入 LangChain 后，历史记录的保存、传递和展示仍各有职责。
- 当前集成底层使用兼容协议客户端请求 DeepSeek，API 异常类型来自 `openai` 包；`APITimeoutError` 属于 `APIError` 的一种，应先处理更具体的超时错误。
- `max_retries=0` 关闭自动重试；`extra_body` 用来传递 DeepSeek 的非思考模式配置。

#### 问题与处理

本次学习者未报告新的运行故障。迁移时检查并同步处理了以下差异：

- 返回值由原始响应 JSON 变为 `AIMessage`，相应改用 `.content` 提取正文。
- 新客户端的异常不再由 `requests.Timeout` / `requests.RequestException` 捕获，页面需同步更换异常类型。
- README 的旧安装命令只包含 Streamlit 与 Requests，需补上当前模型集成及客户端依赖。

#### 验证与遗留事项

- **学习者实际运行确认**：原版功能复验正常，LangChain 安装与导入成功；修改后按问答、追问、清空的验收步骤确认“运行通过”。
- **本地接口检查**：使用占位密钥和模拟响应检查构造参数、字典消息转换及返回的 `AIMessage` 正文类型，该检查未向远端模型发送请求。
- **代码检查**：`app.py` 与 `llm.py` 语法解析通过；函数中只有一次 `invoke`，页面提交分支只调用一次 `ask_deepseek`；异常捕获顺序为 `APITimeoutError`、`APIError`。
- **理解检查**：学习者正确说明了 `ChatDeepSeek` 用于初始化配置、`invoke` 发起请求并返回 `AIMessage`，以及函数最终返回模型回答正文。
- **后续验证边界**：异常类型与捕获顺序已经核对，主动模拟认证失败、限流和超时等场景仍属于 Day 05 的未完成任务。
- 当前没有阻塞下一次学习的问题。论文读取、检索与持久化仍按后续计划推进。

#### 下一次学习

**Day 03：提示词与上下文实验。**

手动提供一小段论文原文，对同一个问题比较有无原文时的回答；实验时控制聊天历史，避免上一轮内容干扰对照。

验收标准：保存至少一组对照结果，并能解释系统提示词、外部资料和聊天历史分别起什么作用。完成后更新任务状态和学习记录。

### Day 03｜2026-10-14｜提示词与上下文实验

- **当天目标**：手动提供论文摘录，对比有无原文时的回答，理解系统提示词、外部资料和聊天历史的作用。
- **本次状态**：已完成；首次对照、代码修正后的复验和理解检查均已通过。
- **实际用时**：未记录。
- **学习安排**：学习者在完成 Day 02 后主动继续下一项，因此两个学习日记录使用同一自然日期。
- **当天剩余必做任务**：0。

#### 完成的任务

- [x] 在页面添加参考资料输入框，把手动粘贴的资料和问题一起组装到用户消息。
- [x] 调整系统提示词：论文事实依据本次资料，资料不足时说明不足，回答包含“结论”和“原文依据”。
- [x] 学习者提供了无资料与有资料两次实际回答，保存如下。
- [x] 补齐用户消息中的 `<reference>` 开始标记，代码检查确认标记成对存在且语法通过。
- [x] 用自己的话解释系统提示词、参考资料和聊天历史各自的作用，以及对照实验为何清空历史。
- [x] 按每组提交前清空历史的步骤完成修正后的两组复验，学习者提供实际回答并确认符合预期。

#### 代码与文档改动

- `app.py`：新增 `st.text_area`，使用格式化字符串组装参考资料和问题，调整系统提示词。
- `llm.py`：本次练习继续复用已有模型调用函数。
- `readme.md`：保存首次实验、标记修正、理解检查与复验结果；将 Day 03 标记为完成，进度更新为 3 / 42，下一学习日为 Day 04。

#### 首次对照实验

实验问题：**参考资料中的模型在什么翻译任务上取得了多少 BLEU？**

实验要求：使用同一问题、同一套系统提示词，每组开始前清空聊天历史，只改变本次参考资料。首次返回结果时，学习者未单独说明是否逐组清空历史；随后按明确的操作步骤完成了复验，结果补录在后文。

参考资料来源：[Attention Is All You Need 的摘要](https://arxiv.org/abs/1706.03762)。有资料组使用的输入为：

```text
文献：Attention Is All You Need
位置：摘要
原文摘录：Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task
```

**A 组：参考资料为空。学习者提供的实际回答：**

```text
结论：本次用户消息中的 <reference> 内没有提供任何资料，因此无法回答该模型在什么翻译任务上取得了多少 BLEU。

原文依据：无可用原文依据。
```

**B 组：提供上述摘录。学习者提供的实际回答：**

```text
结论：参考资料中的模型在 WMT 2014 英语到德语翻译任务上取得了 28.4 BLEU。

原文依据：摘要中明确写道：“Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task”。
```

观察：A 组说明资料不足，B 组的翻译方向和分数与摘录一致，并引用了对应原文。两次实际回答符合本次预期；这一组样例不能证明模型在所有情况下都遵守提示词。

#### 问题与处理

首次代码检查发现，用户消息的 `content` 直接以 `reference_text.strip()` 开头，后面有 `</reference>`，但缺少前面的 `<reference>`。这不会造成 Python 语法错误，模型本次也给出了符合预期的回答，但资料边界没有按设计完整传入。

学习者随后已在资料正文前补上 `f"<reference>\n"`。再次代码检查确认开始与结束标记均存在，`app.py` 语法解析通过；修正后的真实模型运行结果也已由学习者复验并确认符合预期。这些标记是提示词中的文本分隔符，不能代替检索或程序校验。

复验说明最初过于概括，学习者要求具体操作流程。随后明确为：第一次清空会话、手动清空参考资料并提交问题；第二次再次清空会话、粘贴摘录并提交相同问题。“清空会话”清除聊天历史，参考资料输入框的内容需要单独编辑。

#### 理解检查（2026-10-14）

学习者说明：系统提示词用于约束 AI；参考资料告诉 AI 从哪里找答案；聊天历史让模型知道之前的交流并保持上下文连贯；不清空历史会干扰对照实验。

理解检查通过，并补充明确：当前的参考资料直接提供回答所用的原文依据，后续检索功能才负责自动寻找相关片段。聊天历史中如果已经包含原文或答案，就可能影响“本次未提供资料”这一组的回答，因此对照实验需要控制历史。

#### 修正后复验（2026-10-14）

使用与首次实验相同的问题、论文摘录和系统提示词，按每组提交前清空会话的步骤操作。学习者返回了以下实际回答，并明确确认“符合上述预期”。

**A 组：参考资料为空。**

```text
结论：本次用户消息中的 <reference> 资料为空，没有提供任何模型或 BLEU 分数的信息，因此无法回答该模型在什么翻译任务上取得了多少 BLEU。

原文依据：无。参考资料为空，不能据此推测或补充事实。
```

**B 组：提供论文摘录。**

```text
结论：参考资料中的模型在 WMT 2014 英语到德语（English-to-German）翻译任务上取得了 28.4 BLEU。

原文依据：原文摘录为“Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task”。
```

复验结论：无资料时说明依据不足；有资料时，翻译任务、BLEU 分数与原文引用均与提供的摘录一致。本次对照实验验收通过。

#### 验证与遗留事项

- **学习者实际运行结果**：首次对照和修正后的两组复验回答均由学习者提供，复验结果已明确确认符合预期；助手未重新调用远端模型。
- **代码检查结果**：参考资料和问题都已放入用户消息，系统提示词已更新；资料开始标记已补齐，`app.py` 语法解析通过。
- **理解检查结果**：学习者已说明三类上下文的作用及清空历史的实验目的。
- **当天遗留事项**：无；Day 03 验收已完成。
- **验证边界**：本次样例通过不代表提示词在所有情况下都能约束模型。自动检索、结构化输出与引用程序校验仍按后续计划推进。

#### 下一次学习

**Day 04：结构化输出。**

学习字典、JSON 和数据校验，将回答、依据和证据不足说明整理为程序可以分别读取的字段。

验收标准：程序能读取各字段，并识别缺字段或格式错误的结果。完成验收并更新记录后结束当次学习。

### Day 04｜2026-10-16｜结构化输出与数据校验

- **当天目标**：理解 JSON 文本与 Python 字典的区别；定义并读取回答字段；识别缺字段和 JSON 格式错误；将解析、校验接入真实问答页面。
- **本次状态**：已完成。
- **实际用时**：未记录。
- **当天剩余必做任务**：0。

#### 完成的任务

- [x] 编写 `practice_json.py`，用模拟 JSON 文本观察解析前的 `str` 和解析后的 `dict`。
- [x] 通过 `result["answer"]` 读取回答，通过 `result["evidence"][0]` 读取第一条原文依据。
- [x] 编写 `parse_answer()`，检查最外层类型、必需字段、字段类型及依据列表中的元素类型。
- [x] 运行正常样例，并分别构造缺少 `answer` 和缺少逗号的输入，观察两类错误。
- [x] 在 `llm.py` 中使用 `bind(response_format={"type": "json_object"})`，并返回解析、校验后的字典。
- [x] 更新系统提示词，明确三个字段的格式、资料不足时的行为及 JSON 示例。
- [x] 页面分别读取字段，组装展示文字，并添加 `JSONDecodeError`、`ValueError` 错误处理分支。
- [x] 定位并修正 JSON 模式参数拼写导致的 HTTP 422 错误。
- [x] 学习者提供修正后有无资料两组实际页面结果，均符合预期。
- [x] 更新 README 的文件结构、函数返回值说明、任务状态与学习记录。

#### 代码与文档改动

| 文件 | 改动及原因 |
| --- | --- |
| `answer_parser.py` | 新增解析函数，将 JSON 文本转换为字典并检查必需字段和类型。 |
| `practice_json.py` | 新增本地练习，通过模拟回答学习解析、字段读取和错误识别；当前文件已恢复为正常样例。 |
| `llm.py` | 开启 JSON 输出模式，调用解析函数；`ask_deepseek()` 的返回值由正文字符串改为校验后的字典。 |
| `app.py` | 更新系统提示词；读取字典字段并组装展示文字；补充格式与字段校验失败的错误提示。 |
| `readme.md` | 将 Day 04 标记为完成，进度更新为 4 / 42，下一学习日为 Day 05。 |

当前一次请求的数据流：

**页面组装消息 → JSON 模式调用 DeepSeek → `AIMessage.content` 中的 JSON 文本 → `parse_answer()` 解析与校验 → 返回字典 → 页面读取字段、组装展示文字 → 保存并展示本轮消息。**

#### 学到的概念

- JSON 文本可以保存在 Python 字符串中，外观类似字典不代表它已经是字典。
- 学习者说明 `json.loads()` 可以将文本解析后按字段读取；讲解中进一步明确：JSON 对象对应 Python 字典，JSON 数组对应列表，解析结果不一定总是字典。
- JSON 格式正确与满足业务字段要求是两项不同检查。缺少 `answer` 的 JSON 仍可能被正常解析，因此需要额外校验。
- `isinstance()` 用于检查数据类型；`raise ValueError(...)` 用于主动报告不符合要求的数据，并中断当前函数。
- `bind()` 设置后续请求使用的参数，实际发送请求的仍是 `invoke()`。
- 页面中的 `result` 是供程序读取的字典，`answer` 是用于展示和保存聊天历史的字符串。
- `JSONDecodeError` 是 `ValueError` 的子类，应先捕获更具体的 JSON 格式错误。
- 字段和类型检查不等于事实核验。当前解析函数不会自动判断依据是否出自资料、是否足以支持结论。

#### 本地解析练习结果

以下结果由学习者实际运行并反馈：

| 输入与操作 | 实际结果 |
| --- | --- |
| 正常 JSON 样例 | 解析前为 `str`，解析后为 `dict`；成功读取回答和第一条原文，学习者确认“正常通过”。 |
| 将模拟 JSON 中的 `"answer":` 改为 `"answer_typo":` | `ValueError: 回答缺少字段：answer` |
| 恢复字段名后删除回答字段一行末尾的逗号 | `json.decoder.JSONDecodeError: Expecting ',' delimiter: line 4 column 5 (char 61)` |

练习结束后已恢复正确字段名和逗号。类型检查分支已实现并完成代码检查，本次没有逐一构造所有错误类型的运行样例。

#### 问题与解决过程

接入页面后，有无资料两次调用都返回 HTTP 422。错误信息指出 `response_format.type` 收到了 `json.object`。检查发现 `llm.py` 将固定参数值 `json_object` 中的下划线误写成点号。

学习者将其修正为 `response_format={"type": "json_object"}` 并保存，随后两组真实问答通过。此次错误发生在服务端校验请求参数时，尚未执行到本地的 `parse_answer()`。

#### 修正后的实际页面结果

测试问题：**参考资料中的模型在什么翻译任务上取得了多少 BLEU？**

两组操作均从清空会话开始：第一组手动清空参考资料，第二组提供以下摘录，再提交相同问题：

```text
Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task
```

摘录来源为此前使用的 [Attention Is All You Need 摘要](https://arxiv.org/abs/1706.03762)。以下保留学习者反馈的页面文字，而非推测或重建模型返回的原始 JSON。

**A 组：参考资料为空。**

```text
资料不足，无法回答。

原文依据

无。

资料不足说明

用户消息中的 <reference> 内没有提供任何参考资料内容，因此无法依据资料回答该模型在什么翻译任务上取得了多少 BLEU。
```

**B 组：提供论文摘录。**

```text
该模型在 WMT 2014 英语到德语翻译任务上取得了 28.4 BLEU。

原文依据

Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task
```

结果：无资料时显示不足说明与“无”依据；有资料时，任务方向、分数和原文引用与输入摘录一致，未显示资料不足说明。

#### 验证与遗留事项

- **学习者实际运行确认**：本地正常样例、缺字段、格式错误三种情况均有反馈；修正参数拼写后，两组真实页面问答结果符合预期。
- **本地接口检查**：使用占位密钥检查 JSON 模式和非思考模式的请求参数组装，未发送远端请求。
- **最终代码检查**：四个 Python 文件语法解析通过；`llm.py` 使用正确的 `json_object` 参数，每次调用路径只有一次 `invoke`，返回 `parse_answer(response.content)`；页面先捕获 `JSONDecodeError` 再捕获 `ValueError`。
- **验证边界**：此次没有主动在页面中注入异常并验证历史状态，也没有全面复测认证、限流、超时和输出截断；这些属于 Day 05。当前会话仍保存展示文字，尚未持久化结构化结果，也尚未实现自动引用核验。
- **当天遗留事项**：无；Day 04 验收已完成。

#### 下一次学习

**Day 05：调用失败与输出长度。**

理解异常如何从模型调用传到页面，区分认证、限流、超时等失败，并学习识别回答被截断的情况。

验收标准：使用模拟失败验证页面提示和聊天历史状态，能够区分“请求成功”与“回答生成完整”。完成验收并追加记录后结束当次学习。

### Day 05｜2026-10-16｜调用失败、异常传播与输出长度

- **当天目标**：理解异常传播和捕获顺序，细分调用失败提示，识别未正常结束的回答，并用本地模拟验证失败时的聊天历史状态。
- **本次状态**：已完成。
- **实际用时**：未记录。
- **当天剩余必做任务**：0。

#### 完成的任务

- [x] 编写 `practice_errors.py`，分别运行模拟超时与正常返回两种情况，观察历史消息数量为 2 和 4。
- [x] 理解异常会向外传播，由匹配的 `except` 处理；同一 `try` 中尚未执行的保存代码会被跳过。
- [x] 为认证失败、限流、超时和连接失败增加专用提示，保留通用 API、JSON 格式与字段校验错误处理。
- [x] 完成捕获顺序理解检查，能说明通用 `APIError` 放在前面会遮住专用提示。
- [x] 定义 `IncompleteAnswerError`，在 JSON 解析之前检查 `finish_reason`。
- [x] 处理客户端在 `invoke()` 返回前抛出的 `LengthFinishReasonError` 和 `ContentFilterFinishReasonError`，保留原始异常原因。
- [x] 使用本地模拟检查失败提示、已有历史保留，以及正常回答只追加一问一答。
- [x] 理解合法 JSON 不代表回答已完整生成，区分输出 token 限制与上下文窗口长度。
- [x] 完成最终模拟验证，更新任务状态、当前功能说明和学习记录。

#### 代码与文档改动

| 文件 | 改动及原因 |
| --- | --- |
| `practice_errors.py` | 新增本地练习，通过主动抛出异常或正常返回观察控制流与历史数量变化。 |
| `app.py` | 添加认证、限流、连接失败与回答未正常结束的专用错误提示，并保持具体异常先于通用异常的顺序。 |
| `llm.py` | 定义自有异常，转换客户端的提前终止异常，检查响应结束原因，再执行 JSON 解析。 |
| `readme.md` | 将 Day 05 标记为完成，进度更新为 5 / 42；记录测试层次、结果和下一学习日。 |

正常路径：

**调用模型 → 检查结束原因 → 解析并校验 JSON → 页面组装展示文字 → 追加一问一答。**

失败路径：

**调用或检查中抛出异常 → 页面匹配相应错误分支 → 显示提示 → 保留已有聊天历史。**

#### 学到的概念

- 函数内没有被处理的异常会传到调用方；匹配的 `except` 执行后，程序可以继续执行异常处理块后面的语句。
- `except` 从上往下匹配，执行第一个匹配分支。认证和限流等专用异常应放在 `APIError` 前，超时应放在连接失败前，JSON 格式错误应放在 `ValueError` 前。
- `IncompleteAnswerError` 是应用自己的异常类型，用于统一表示回答未正常结束；`raise ... from error` 保留底层异常以便排查。
- `AIMessage.response_metadata` 中的 `finish_reason` 描述生成结束原因。当前流程仅接受 `stop`；`length` 或其他结束状态、缺失状态均不会作为成功回答保存。
- `length` 表示生成触及长度限制，即使正文恰好是合法 JSON，也不能据此把它视为完整回答。
- 当前 `max_tokens=1024` 限制输出 token 数。学习者最初将其表述为上下文长度，讲解中补充区分了输出限制与整个上下文窗口。
- 请求成功、正常结束、JSON 格式正确、字段符合要求和事实有证据支持，是不同层面的判断；本次仍未实现自动事实与引用核验。

#### 学习者本地练习结果

模拟超时的实际输出：

```text
准备调用
进入模拟调用函数
捕获到错误： 模拟请求超时
历史消息数量： 2
```

将抛出异常改为正常返回后的实际输出：

```text
准备调用
进入模拟调用函数
本轮对话已保存
历史消息数量： 4
```

学习者解释：第一次捕获到错误，第二次正常执行。补充说明了第一次跳过保存代码后仍会执行最后的数量打印。当前练习文件保留正常返回版本。

#### 问题与解决过程

首版教学示例只在 `invoke()` 返回后检查 `finish_reason`。检查当前安装的客户端实现，并用模拟 HTTP 响应运行实际 SDK 与 LangChain 调用后发现：JSON 模式可能在返回 `AIMessage` 前就抛出 `LengthFinishReasonError` 或 `ContentFilterFinishReasonError`。

这两个异常不是 `APIError` 的子类，原有页面处理也不会将其识别为 `IncompleteAnswerError`，因此只检查响应元数据存在遗漏。此遗漏来自助手给出的首版示例。

学习者随后在 `llm.py` 的 `invoke()` 周围添加专用捕获，将它们转换成 `IncompleteAnswerError`，并用 `from error` 保留原始原因。响应元数据检查继续保留，以覆盖已经返回响应但未正常结束的情况。修正后最终验证通过。

#### 助手执行的本地模拟验证

**第一轮：8 项页面检查全部通过。** 使用 AppTest 执行页面，将 `ask_deepseek()` 替换成模拟返回或异常，并预置两条历史消息。

| 场景 | 实际结果 |
| --- | --- |
| 认证失败、限流、超时、连接失败、通用 API 错误 | 5 项均显示对应错误，历史保持原来的两条，未追加失败轮次。 |
| JSON 格式错误、字段校验错误 | 2 项均显示对应错误，历史保持不变。 |
| 正常返回 | 只追加一问一答，历史由 2 条变为 4 条。 |

**最终补验：9 项全部通过。** 其中 7 项使用模拟 HTTP 响应运行实际安装的 SDK、LangChain、`llm.py` 和页面；另 2 项直接构造 `AIMessage`，验证返回响应后的检查分支。

| 场景 | 实际结果 |
| --- | --- |
| `stop` 且 JSON、字段有效 | 正常解析、展示，历史由 2 条变为 4 条。 |
| `length` 且正文为合法 JSON | SDK 的提前异常被转换，页面显示长度限制提示；未进入 JSON 解析，历史不变。 |
| `length` 且正文为残缺 JSON | 优先显示长度限制提示，未把截断误报为普通 JSON 格式问题；历史不变。 |
| `content_filter` | SDK 的提前异常被转换，页面显示内容过滤提示；历史不变。 |
| `aborted` | 结束状态检查拒绝该响应，显示未正常结束；历史不变。 |
| `stop` 但 JSON 格式错误 | 页面显示 JSON 错误；历史不变。 |
| `stop` 但缺少必需字段 | 页面显示结构校验错误；历史不变。 |
| 直接返回带 `length` 元数据的 `AIMessage` | 响应检查在 JSON 解析前拦截，历史不变。 |
| 直接返回缺少结束原因的 `AIMessage` | 响应检查在 JSON 解析前拦截，历史不变。 |

全部模拟使用占位密钥。模拟 HTTP 由本地 `MockTransport` 提供，未向外部模型发送请求；页面测试预置的历史和密钥与实际浏览器会话分开。自动化检查通过终端临时脚本执行，本次未另存为仓库中的测试文件。

#### 验证与遗留事项

- **学习者实际运行**：提供本地异常练习的两组输出，并完成捕获顺序和长度限制的理解检查；保存了最终代码修正。
- **助手代码检查**：`app.py`、`llm.py`、`answer_parser.py`、`practice_json.py`、`practice_errors.py` 五个文件语法解析通过。
- **助手模拟验证**：第一轮 8 项页面检查和最终 9 项补验通过，具体覆盖范围见上表。
- **验证边界**：本次验收针对本地错误处理、结束状态检查和历史状态，未重新调用真实 DeepSeek API，也未模拟完整的网络等待过程。自动重试、事实核验及持久化仍按后续计划推进。
- **当天遗留事项**：无；Day 05 验收已完成。

#### 下一次学习

**Day 06：让项目可复现运行。**

整理依赖文件、配置示例与启动步骤，学习 Git 基本提交方式，并在独立环境中验证说明是否足够完整。

验收标准：可以按文档准备配置和依赖并启动应用，依赖版本与必要配置有明确记录。完成验收并更新学习记录后结束当次学习。

### Day 06｜2026-10-16｜依赖快照、配置模板与独立环境验证

- **当天目标**：记录依赖与配置，在独立环境中验证启动步骤，并练习 Git 本地提交。
- **本次状态**：已完成。
- **实际用时**：未记录。
- **当天剩余必做任务**：0。

#### 完成的任务

- [x] 理解项目虚拟环境与外部 Python 环境的区别，明确使用项目环境中的 Python 导出依赖。
- [x] 生成 `requirements.txt`，保存当前环境的 66 条依赖版本记录。
- [x] 新建 `.streamlit/secrets.toml.example`，仅填写配置键名和占位符。
- [x] 在 `.gitignore` 中增加 `.venv-check/`，确认真实配置被忽略、模板可被跟踪。
- [x] 新建 `.venv-check`，根据依赖清单安装，并通过 `pip check`。
- [x] 用新环境在 8502 端口启动应用，完成带参考资料的问答与清空会话验证。
- [x] 更新 README 的依赖说明、配置方式、Bash 与 PowerShell 命令及验证步骤。
- [x] 学习者完成 Git 本地提交 `f5d99e0` 并反馈最新提交记录，助手核对提交文件与工作区状态。

#### 代码与文档改动

| 文件 | 改动及原因 |
| --- | --- |
| `requirements.txt` | 新增当前 Windows / Python 3.12 环境的已安装包版本快照。 |
| `.streamlit/secrets.toml.example` | 新增只含占位符的配置模板，说明需要填写的键名与格式。 |
| `.gitignore` | 增加独立验证环境目录，避免将其加入版本管理。 |
| `readme.md` | 更新运行步骤与独立环境验证方法；完成提交核对后，将 Day 06 标记为完成，进度更新为 6 / 42，并补充本次记录。 |

`app.py`、`llm.py` 和 `practice_errors.py` 的变化来自 Day 05 的异常处理与练习；本次环境整理没有修改这些文件。提交 `f5d99e0` 将这些尚未提交的代码与本次依赖、配置及文档改动一并保存。

#### 学到的概念

- 虚拟环境拥有独立的包安装目录。明确指定 `.venv/Scripts/python.exe`，可以避免导出其他 Python 环境中的包。
- `python -m pip` 使用所指定 Python 环境中的 pip；`pip freeze` 导出已安装包，`pip install -r` 按文件中的要求安装。
- `==` 指定包版本。当前快照包含直接依赖、间接依赖及环境中的其他已安装包，不是最小依赖集合，也不是自动求解生成的锁文件。
- 配置模板记录字段与占位符，实际配置保存个人密钥；应用默认读取 `.streamlit/secrets.toml`，不会自动读取 `.example` 模板。
- `pip check` 检查包声明的依赖关系；页面问答与清空验证用于检查应用实际行为，两者验证范围不同。
- 本次复现范围为同一台 Windows 电脑上的 Python 3.12.6 独立包环境，源码与现有本地配置共用；跨平台运行尚未验证。
- `git add` 将指定文件的当前内容加入暂存区，`git diff --cached` 查看待提交差异，`git commit` 将暂存内容保存为本地版本；上传远端属于另一步操作。

#### 问题与解决过程

学习者在 Bash 中执行助手最初按 PowerShell 提供的导出命令，收到 `bash: Set-Content: command not found`。原因是终端类型与命令语法不匹配。

改用 Bash 的正斜杠路径与输出重定向：

```bash
./.venv/Scripts/python.exe -m pip freeze > requirements.txt
```

随后成功生成文件。后续练习按学习者当前的 Bash 终端给出命令，README 也分别说明两种终端的写法。

#### 验证与遗留事项

- **学习者实际运行确认**：依赖检查通过、问答正常、清空正常。验证使用新环境启动的页面与 WMT 2014 英语到德语、28.4 BLEU 的参考摘录。
- **助手文件检查**：依赖文件为 66 条 `包名==版本` 记录，与原项目环境导出一致；配置模板 TOML 格式正确且使用预期占位符；真实配置与验证环境已被 Git 忽略，未被跟踪。
- **助手新环境检查**：`.venv-check` 的 Python 为 3.12.6，未启用系统包共享；已安装包快照与 `requirements.txt` 一致，主要依赖版本与原环境相同。
- **Git 提交核对**：学习者反馈 `f5d99e0 (HEAD -> main) 完善异常处理并添加可复现运行配置`。助手确认该提交包含 `.gitignore`、配置模板、`app.py`、`llm.py`、`practice_errors.py`、`readme.md` 和 `requirements.txt` 七个文件；核对时 `git status --short` 无输出，真实配置和两个虚拟环境均未被跟踪。
- **验证边界**：问答与清空结果来自学习者实际操作反馈；助手没有重新调用远端模型，也没有读取真实密钥。本次没有重复 Day 05 的完整异常模拟测试。
- **当天遗留事项**：无；Day 06 学习验收已完成。本段完成状态是在核对 `f5d99e0` 后补写，属于该提交之后新增的 README 修改，可单独提交保存。

#### 下一次学习

**Day 07：基础阶段复盘与样例准备。**

解释输入到输出的数据流，选择可提取文字的论文，准备 3 个有原文依据的问题和 1 个无答案问题，并记录对应页码。验收标准：能说明各模块的职责，保存问题与页码清单，并完成聊天功能回归检查。

### Day 07｜2026-10-16｜基础复盘、论文证据与问答样例

- **当天目标**：复盘问答数据流，准备固定版本的论文与 3 个有答案、1 个资料不足的样例，并检查现有页面功能。
- **本次状态**：已完成；第一阶段验收通过，总进度为 7 / 42。
- **实际用时**：未记录。
- **当天剩余必做任务**：0。

#### 完成的任务

- [x] 复盘页面、模型调用和解析模块的职责，以及各步骤的数据类型。
- [x] 通过练习区分 JSON 语法解析与 Python 字典结构校验。
- [x] 选择可提取文字的论文，核对本地 PDF 的来源、版本和页码。
- [x] 手动找到三条直接支持答案的原文，记录对应 PDF 页码。
- [x] 设计一个与论文相关、但原文资料不足的问题，并说明无法回答的原因。
- [x] 编写并修正 `data/sample_questions.json`，完成字段、类型、引用和页码检查。
- [x] 在页面验证 GPU 数量问答、型号追问和耗电量资料不足的情况。
- [x] 确认两轮历史保留、资料不足时原文依据显示“无”、清空后历史消失。
- [x] 更新 README 的任务状态、样例说明和每日学习记录。

#### 文件改动

| 文件 | 改动及原因 |
| --- | --- |
| `data/papers/attention_is_all_you_need_neurips2017.pdf` | 保存并核对 NeurIPS 2017 官方论文；将原先带有误导性 `v7` 后缀的文件名改为与实际版本对应的名称。 |
| `data/sample_questions.json` | 学习者编写四条样例，记录答案、原文、页码和资料不足原因；助手协助检查并统一两处标点。 |
| `readme.md` | 将 Day 07 标记为完成，增加论文来源、样例说明、验收结果与本次学习记录。 |

本次应用代码和依赖没有变化。样例文件已保存，应用尚未自动加载这些样例或解析 PDF。

#### 学到的概念

成功调用时，数据依次经过以下步骤：

```text
页面组织系统要求、聊天历史、当前参考资料与问题
→ invoke(messages) 返回 AIMessage
→ response.content 提供 JSON 文本
→ parse_answer() 内的 json.loads() 解析为 Python 值
→ 校验字典、必需字段与字段类型
→ ask_deepseek() 返回校验通过的 dict
→ 页面展示结果并保存本轮问答
```

- `invoke()` 负责模型调用；将 JSON 文本解析为 Python 值的是 `json.loads()`。
- JSON 数组也符合 JSON 语法。例如 `'["你好"]'` 能成功解析，但结果是 `list`，会被当前解析器的字典类型检查拒绝。
- `not False` 的结果是布尔值 `True`，因此相应 `if` 分支会执行，不应把表达式结果说成整数 `1`。
- 答案可以概括原文；证据字段应保留直接支持答案的连续原文，不能用自行改写的句子冒充引用。
- 页码必须对应具体文档版本。本次 `pdf_page` 使用从 1 开始的 PDF 页面序号。
- FLOPs 与 kWh 分别表示浮点运算量与电能，不能将论文中的计算成本直接当作实际耗电量。
- 无答案样例使用 `answerable: false`、`evidence: []`、`pdf_page: null`，另写资料不足原因；检查过哪些页不等于这些页提供了答案证据。

#### 问题与解决过程

**解析和校验的区别。** 学习者起初认为 JSON 数组会在 `loads()` 处失败。通过检查解析结果类型，明确语法合法不代表满足应用要求：当前应用只接受含指定字段的字典。以后学习者表示“不明白题目”时，先解释场景、术语并给提示，待其尝试后再讲解答案。

**论文版本与页码。** 学习者找到编码器层数的原文位于第 2 页，与先前参考的 arXiv v7 第 3 页不同。检查确认，本地文件实际是 NeurIPS 2017 的 11 页版本，其 SHA256 与官方文件一致；学习者记录的第 2 页正确。因此保留这份论文，并修正文件名与来源说明。

**证据是否直接回答问题。** 位置编码练习中，先找到的第 5 页段落解释了为什么需要位置信息；继续阅读后，在第 6 页找到正弦和余弦函数的直接依据。写入 JSON 时又将原文改写为解释句，随后恢复原文。引用省略句末冒号仍可构成有效的连续摘录；助手最初的整句严格相等检查过于严格，最终仅为统一记录补回冒号。

**资料不足与无关问题。** “明天吃什么？”与论文无关，因此改用论文训练实际耗电量作为样例。核对训练描述和成本表后确认，论文给出了硬件、训练时间及 FLOPs，但缺少实际耗电量和确定它所需的实际功率等数据，不能据此给出 kWh 数值。

#### 保存的四条样例

| 编号 | 问题主题 | 预期结果 | PDF 页码 |
| --- | --- | --- | --- |
| q1 | 编码器层数 | 6 层，有直接原文依据。 | 2 |
| q2 | 训练 GPU 数量与型号 | 8 块 NVIDIA P100 GPU，有直接原文依据。 | 7 |
| q3 | 位置编码使用的函数 | 正弦和余弦函数，有直接原文依据。 | 6 |
| q4 | 训练实际耗电量（kWh） | 资料不足，无法依据论文确定。 | `null` |

q4 的资料不足说明记录了相关检查位置第 7～8 页，证据列表仍为空。完整原文与字段保存在 `data/sample_questions.json`，后续可用于检查 PDF 解析与检索结果。

#### 验证与遗留事项

- **样例检查**：JSON 可解析，四条样例的编号、字段类型、可回答标记、页码和引用符合约定；本地论文与官方 NeurIPS PDF 的 SHA256 一致。
- **学习者实际页面验证**：手动粘贴训练 GPU 的原文后，首轮回答“训练使用了 8 块 NVIDIA P100 GPU。”；追问型号时回答“型号是 NVIDIA P100 GPU。”。清空历史并保留同一段参考资料，再问实际耗电量，页面说明资料不足，未编造数值。
- **页面状态确认**：学习者确认两轮问答记录均保留、资料不足时原文依据显示“无”、再次清空后全部聊天历史消失。
- **验证边界**：页面使用手动摘录，尚未自动读取整篇 PDF；四条样例的数据检查不代表已经完成四条自动问答或检索评测。助手核对 PDF 使用临时只读脚本，没有将其作为项目解析功能保存，也没有新增依赖或重复调用远端模型。
- **当天遗留事项**：无；Day 07 验收完成。

#### 下一次学习

**Day 08：读取 PDF 并保留页码来源。**

学习使用 PDF 解析库逐页提取文本，保留文档标识与从 1 开始的 PDF 页码，理解 PDF 页面序号与正文印刷页码的区别。

验收标准：能读取本次固定版本论文，展示提取到的文本及其来源页码，并用已有样例核对对应页面。完成验收后更新依赖与学习记录。

### Day 08｜2026-10-18｜PDF 逐页读取、来源记录与空文本检查

- **当天目标**：读取文本型 PDF，逐页保留文档来源和页码，核对样例原文，并识别未提取到可用文字的页面。
- **本次状态**：已完成，总进度为 8 / 42。
- **实际用时**：未记录。
- **当天剩余必做任务**：0。

#### 完成的任务

- [x] 在项目虚拟环境中安装 `pypdf`，实际版本为 6.21.1。
- [x] 使用 `PdfReader` 读取论文，确认共 11 页，第一页提取结果为 `str`。
- [x] 使用 `enumerate(..., start=1)` 逐页处理，并构造包含四个字段的页面记录。
- [x] 修正文档标识的字段名，确认文档标识和路径能通过固定键读取。
- [x] 查看第 2、6、7 页提取文本，对照已有三条有答案样例。
- [x] 将读取逻辑封装为 `load_pdf_pages(pdf_path, document_id)`，练习脚本负责调用和打印。
- [x] 修正循环缩进与重复打印，确认函数返回全部 11 页。
- [x] 编写空文本判断，收集需要检查的页码，并在整篇没有可用文字时提示。
- [x] 通过原论文、全空白 PDF、混合页面 PDF 和仅空白字符记录四组检查。
- [x] 更新依赖文件、当前能力说明和本次学习记录。

#### 代码与文档改动

| 文件 | 改动及原因 |
| --- | --- |
| `pdf_loader.py` | 学习者新增逐页读取函数，返回含 `document_id`、`source`、`pdf_page`、`text` 的字典列表，保留空白页。 |
| `practice_pdf.py` | 学习者新增读取练习，打印页面数、各页字符数和空文本页码，整篇无可用文字时打印提示。 |
| `requirements.txt` | 助手补入已安装并验证的 `pypdf==6.21.1`，其余 66 条记录不变。 |
| `readme.md` | 更新为 Day 08 完成、进度 8 / 42，补充读取方式、数据字段、验证边界与学习记录。 |

关键代码由学习者编写和修改。助手负责讲解、检查、临时验证及依赖与文档维护；本次未修改聊天页面和模型调用代码。

#### 学到的概念

- `PdfReader` 提供页面集合，页面对象通过 `extract_text()` 返回提取文本。
- 列表下标从 0 开始；学习者正确回答 PDF 第 7 页对应下标 6。对全部页面按原顺序保留的列表，可以用 `pages[pdf_page - 1]` 访问对应记录。
- `enumerate(reader.pages, start=1)` 同时提供从 1 开始的编号和页面对象，编号用于记录来源。
- 字典中的字符串键与变量含义不同：`"document_id": document_id` 左边是固定字段名称，右边是传入的值。
- 创建本页字典和追加记录都要放在循环内；初始化列表和最终返回放在循环外、函数内。
- 函数接收路径和标识，返回结构化数据；练习脚本负责展示，便于以后复用读取逻辑。
- PDF 页序号是文件中的位置，印刷页码是页面上写出的编号；两者可能不同。本次来源字段统一使用 PDF 页序号。
- `.strip()` 去掉字符串首尾的空白；`not record["text"].strip()` 可以判断空字符串或仅含空白的文本。
- `empty_pages` 是页码列表，`[]` 表示没有发现文本为空的页面；它与编码是否存在无关，也不能证明提取结果完整。

#### 问题与解决过程

**字段名使用了变量。** 初稿写成 `document_id: document_id`，导致键实际变成论文标识的值。页码和字符数仍可打印，因此仅检查这些输出没有发现问题。学习者改成 `"document_id": document_id` 后，成功按固定字段名读取文档标识和路径。

**封装函数时缩进改变了执行次数。** 首次封装后输出两次“页面记录数：1”，且只显示第 11 页。检查发现，循环内只有提取文本，创建字典与追加操作在循环外，因此只保存了最后一页。学习者将两个操作移回循环内，并删除函数中的打印；练习脚本随后只打印一次总数，保留第 1～11 页。无作用的 `pdf_path = pdf_path` 也已删除。

**空列表含义的解释。** 学习者正确实现了空文本判断，但将输出 `[]` 解释为“没有编码”。助手补充区分了文本与页码清单：列表收集的是异常页面的编号，空列表表示没有发现这类页面；当前 11 页均提取到了非空白文本。文本存在与内容是否完整是两项不同的检查。

#### 学习者实际运行结果

- 第一次读取显示 PDF 共 11 页、提取类型为 `<class 'str'>`，第一页开头包含论文标题和作者信息。
- 逐页记录及最终函数调用均得到 11 条记录，页码连续为 1～11；第 1 页字符数为 2908，第 11 页为 2328，封装前后各页字符数一致。
- 修正字段名后，打印出文档标识 `attention_is_all_you_need_neurips2017` 和预期 PDF 路径。
- 在练习中打印第 2、6、7 页全文后，学习者反馈对应页码正确；助手进一步核对三条样例原文。
- 空文本检查实际输出 `未提取到可用文字的页码： []`。

#### 助手验证结果

两个新文件的 Python 语法检查通过。以下四组检查均通过，使用当前安装的 `pypdf` 与学习者保存的代码：

| 场景 | 验证结果 |
| --- | --- |
| 原论文 | 函数返回 11 条记录，字段、来源、文档标识和连续页码正确；三条样例原文在对应页中匹配，比较时仅忽略空白字符；练习脚本得到空页清单 `[]`，不显示整篇无文字提示。 |
| 单页空白 PDF | 函数保留第 1 页，文本为空；练习脚本得到 `[1]`，并提示整篇未提取到可用文字。 |
| 空白页与正文页混合的 PDF | 临时文件第 1 页为空白，第 2 页复制自原论文的 GPU 训练页面；两条记录均保留，正文页的 `pdf_page` 为 2，空页清单为 `[1]`，不误报整篇没有文字。 |
| 仅空白字符的记录 | 单独构造文本为若干空格、换行和制表符的页面记录，练习脚本将其识别为空文本并显示整篇无文字提示。 |

前三组通过实际 PDF 文件调用读取函数，再将返回记录提供给练习脚本检查其实际输出分支；最后一组直接构造页面记录，验证 `.strip()` 判断。测试通过终端临时脚本执行，两份临时 PDF 已清理，没有新增仓库测试文件，也没有调用模型。

依赖检查输出 `No broken requirements found.`。与 Day 06 保存的快照相比，现有环境仅新增 `pypdf==6.21.1`，共 67 条记录；本次未重新创建独立虚拟环境。

#### 验证与遗留事项

- **当前能力**：可从本地文本型 PDF 提取逐页文本并保留来源，练习脚本可列出空文本页面；页面记录保存在内存中。
- **验证边界**：没有实现 OCR，也没有使用真实扫描件检验识别流程。空文本检查只报告未提取到可用文字，不能确定其原因；三条原文匹配不等于全文、图表或公式均已完整提取。
- **应用接入状态**：PDF 读取尚未接入 Streamlit 问答页面；本次未重复调用远端 API 或执行聊天页面回归。
- **当天遗留事项**：无；Day 08 学习验收完成。

#### 下一次学习

**Day 09：文档分块与元数据。**

把逐页文本切成适合后续检索的小段，学习分块长度和重叠，为片段保留标识及来源信息。

验收标准：比较两种分块设置产生的结果，抽查片段能定位到原文和对应 PDF 页码，并为后续向量化准备数据。

### Day 09｜2026-10-18｜递归分块、片段元数据与配置对照

- **当天目标**：理解分块长度与重叠，逐页生成带来源的片段，并比较两套配置。
- **本次状态**：已完成，总进度为 9 / 42。
- **实际用时**：未记录。
- **当天剩余必做任务**：0。

#### 完成的任务

- [x] 用字母串手算分块，正确给出第二个片段从 E 开始、内容为 EFGHIJ。
- [x] 安装 `langchain-text-splitters==1.1.3`，使用字符分块例子验证长度和重叠。
- [x] 理解分块长度是上限，最后一个片段可以更短，不补入原文没有的字符。
- [x] 学习递归分块的默认分隔符，以及目标重叠量不等于每次精确重复的字符数。
- [x] 编写 `chunker.py` 中的逐页分块函数，保存六个片段字段。
- [x] 运行真实论文分块，默认配置得到 51 个片段，第一条记录字段完整。
- [x] 固定目标重叠为 100，对比 400 与 800 两种分块长度，并检查 GPU 原文片段。
- [x] 核对全部片段的来源、长度和标识，以及三条有答案样例。
- [x] 完成空输入、空页面、来源编号及参数边界检查，并通过依赖检查。
- [x] 更新依赖、学习记录，并把跨配置标识重复问题纳入 Day 11 的入库准备任务。

#### 代码与文档改动

| 文件 | 改动及原因 |
| --- | --- |
| `chunker.py` | 学习者新增 `split_pages()`，逐页分块并返回含项目、文档、文件路径、PDF 页码、片段标识和文本的字典列表。 |
| `practice_chunks.py` | 学习者先运行字母串练习，随后改为真实论文的两组配置对照；助手在收尾时移除循环前一次结果未被使用的重复分块调用。 |
| `requirements.txt` | 助手补入 `langchain-text-splitters==1.1.3`，当前共 68 条记录，其余依赖版本不变。 |
| `readme.md` | 更新 Day 09 完成状态、分块接口与结果表，补充标识适用范围和后续入库要求。 |

分块和元数据代码由学习者编写；助手负责讲解、核对、临时检查及上述收尾整理。本次未修改聊天页面和模型调用逻辑。

#### 学到的概念

- `chunk_size` 控制长度上限；本次 `length_function=len` 按 Python 字符数计量，不是按单词数或 token 数计量。
- `chunk_overlap` 用于保留相邻片段的部分上下文。按字符切分的字母串例子可直接手算；真实论文使用默认分隔符时，实际重叠取决于边界，是一个目标值。
- `split_text()` 返回字符串列表；项目分块函数进一步把每段文字与来源字段放入字典，返回片段记录列表。
- 外层循环处理页面，内层循环处理该页面的片段；页码沿用原页面记录，片段编号在每页内从 1 开始。
- 元数据用于描述片段属于哪个项目、哪份文档和哪一页；当前 `demo_project` 是练习参数，尚未实现项目管理界面。
- 列表推导式 `[len(chunk["text"]) for chunk in chunks]` 收集各片段长度，`min()` 和 `max()` 用于观察结果范围。
- 同一句话可能因重叠出现在相邻片段中。关键词命中的片段数量，与整份论文的总片段数不同。
- 片段标识需要明确适用范围。当前组合标识可区分同一套配置中的片段，尚不能作为跨配置混存的唯一标识。

#### 字符分块练习

输入 `ABCDEFGHIJKLMNO`，设置 `chunk_size=6`、`chunk_overlap=2`、`separators=[""]`。学习者运行得到一个包含四个字符串的列表：

```text
ABCDEF   6
EFGHIJ   6
IJKLMN   6
MNO      3
```

学习者正确解释最后一块不足 6 个字符是因为原文已到末尾，不能加入原文没有的内容。

#### 真实论文对照结果

两组均使用同一份 NeurIPS 2017 论文、相同默认分隔符和目标重叠量 100，仅改变分块长度。

| 配置 | 分块长度 | 总片段数 | 最短字符数 | 最长字符数 | GPU 原文所在页与片段 |
| --- | --- | --- | --- | --- | --- |
| A | 400 | 113 | 112 | 400 | 第 7 页，c7。 |
| B | 800 | 51 | 198 | 797 | 第 7 页，c3 与 c4。 |

两组对应片段都完整保留了训练使用 8 块 NVIDIA P100 GPU 的原句。B 组的两个相邻片段重复包含这条依据，与目标重叠有关；最长片段为 797 字符，符合上限 800。

学习者正确判断两组都包含 GPU 数量与型号，但最初把总片段数判断为 B 组更多。助手据实际输出纠正：A 组 113 个，大于 B 组 51 个；B 组打印两段 GPU 内容，不代表整篇片段更多。本次对照说明了长度设置对该论文分块结果的影响，还不能据此确定哪组检索效果更好。

#### 助手验证结果

| 检查范围 | 结果 |
| --- | --- |
| 两套真实论文配置 | 数量、最短和最长长度与学习者输出一致；所有片段非空、不超长，六个字段完整，标识在各自结果中不重复；片段文字均能在标注页中直接找到，输入页面记录未被修改。 |
| 三条有答案样例 | 忽略空白字符差异后，两组都保留全部三条原文。400 / 100 分别位于 q1：p2:c14、q2：p7:c7、q3：p6:c3；800 / 100 分别位于 q1：p2:c7、q2：p7:c3 与 p7:c4、q3：p6:c1 与 p6:c2。 |
| 空输入与空页面 | 空列表、空字符串页面和仅空白字符页面均不生成片段。 |
| 来源编号与标识 | 构造含空页、不同文档且同为第 8 页的记录，生成片段仍保留第 8 页和各自文档来源；更换传入项目标识后，片段项目字段与标识随之区分。 |
| 参数边界 | 底层库对分块长度为 0、重叠为负数、重叠大于分块长度的情况抛出 `ValueError`。 |
| 保存的对照脚本 | 移除重复调用后执行成功，输出两组统计与 GPU 片段。 |

两个新文件语法检查通过。验证通过终端临时脚本执行，使用本地论文和内存中构造的页面记录，没有新增持久化测试文件，也没有调用模型。

`pip check` 输出 `No broken requirements found.`。环境相较 Day 08 仅新增 `langchain-text-splitters==1.1.3`，`langchain-core` 仍为 1.6.10；更新后的 68 条依赖记录与项目环境快照一致。本次未重新创建独立虚拟环境。

#### 当前限制与后续安排

- **跨配置标识**：实测两组配置共有 51 个相同 `chunk_id`，对应文本全部不同，因为当前标识不包含配置或版本。当前结果只在内存中分别观察；Day 11 入库前需调整标识，避免混存时覆盖。
- **验证边界**：当前检查证明片段与提取文本的来源关系，以及三条已知依据被保留；没有开展向量检索、模型回答或完整评估集测试，也不能据此宣称已经选出最优分块参数。
- **练习文件**：原有三个练习文件按约定继续保留，`practice_pdf.py` 的空文本提示尚未迁移；新增 `practice_chunks.py` 作为当前配置对照入口保留。
- **当天完成情况**：Day 09 必做任务已完成；标识升级已明确列入后续入库任务。

#### 下一次学习

**Day 10：Embedding 与语义相似度。**

区分聊天模型和向量模型，把文本转换成向量，用少量相近及不相关文本观察相似度，为后续 Dense 检索做准备。

验收标准：能获得文本向量并比较相似度，用自己的话解释语义相似度与关键词匹配的区别，并记录实际使用的模型和运行方式。

### Day 10｜2026-10-18｜中英文文本向量、余弦相似度与证据边界

- **当天目标**：区分聊天模型和向量模型，在本机生成文本向量，比较相关与不相关候选，并理解相似度的局限。
- **本次状态**：已完成，总进度为 10 / 42。
- **实际用时**：未记录。
- **当天剩余必做任务**：0。

#### 完成的任务

- [x] 用中文问题与英文原文判断相关片段，理解只检查原词出现可能漏掉跨语言匹配。
- [x] 在项目虚拟环境中安装 CPU 版 PyTorch 与 Sentence Transformers，运行本地多语言模型。
- [x] 编写并运行 `practice_embeddings.py`，获得 NumPy 数组和 384 维文本向量。
- [x] 区分向量数量、维度、词数和数值，正确判断两条文本的结果形状为 `(2, 384)`。
- [x] 了解 `device` 的含义，以及使用 GPU 所需的硬件与依赖条件。
- [x] 在向量归一化后通过点积计算余弦相似度，逐一比较问题和候选片段。
- [x] 固定候选，完成硬件、编码器层数和实际耗电量三组问题对照。
- [x] 保留晚饭片段错误排第一的案例，区分排序相关性与答案证据充分性。
- [x] 核对本地执行结果与依赖，更新运行说明、任务状态及本次学习记录。

#### 代码与文档改动

| 文件 | 改动及原因 |
| --- | --- |
| `practice_embeddings.py` | 学习者按讲解编写向量练习，逐次修改问题并运行；最终保存耗电量问题与三个候选的相似度输出代码。 |
| `requirements.txt` | 补入当前环境实际新增的 27 个包，共 95 条指定版本记录；此前 68 条版本不变。 |
| `readme.md` | 更新为 Day 10 完成、进度 10 / 42，补充模型、CPU 安装步骤、完整练习示例、三组结果与能力边界。 |

助手收尾时没有修改学习者的练习代码或聊天应用，也没有执行 Git 提交。当前 `.gitignore` 包含忽略 `practice_*` 的规则，检查确认五个练习脚本均未被 Git 跟踪；本次保留该设置与本地文件，在 README 提供 Day 10 示例以便克隆后重建。原有练习文件仍按维护约定保留，不自动删除。

#### 模型与运行配置

- 模型：`intfloat/multilingual-e5-small`，输出维度 384，输入上限 512 tokens。
- 运行方式：Windows / Python 3.12.6，`SentenceTransformer(..., device="cpu")`；本地模型计算，不调用聊天 API。
- 主要新增依赖：`sentence-transformers==6.1.0`、`torch==2.14.1+cpu`、`transformers==5.20.1`；已有 `numpy==2.5.4` 版本未变。
- 输入约定：问题使用 `query: `，候选使用 `passage: `，包括中文文本；问题和候选由同一个模型编码。
- 编码设置：`convert_to_numpy=True`、`normalize_embeddings=True`，用问题与候选的一维数组点积作为余弦相似度。
- PyTorch 检查结果为 `cuda_build: None`、`cuda_available: False`；这说明当前安装不支持 CUDA，不能据此断定电脑没有显卡。

#### 学到的概念与纠正

聊天模型接收消息并生成回答；向量模型把整段文本转换为数值表示，用于比较语义。中文问题与英文原文可能没有相同字面词汇，纯粹检查原词出现会漏掉相关内容。

学习者首次运行得到 `numpy.ndarray`，形状 `(4, 384)`，第一条向量的前五个数为 `[0.08890866, -0.01363697, -0.04262213, -0.03343075, 0.11417024]`。最初把 384 理解为词数，讲解后正确回答两条文本对应 `(2, 384)`：每个向量由 384 个数值组成，各维度不逐一对应输入词语。

`device="cpu"` 指定计算设备。GPU 对批量数值计算通常有优势，但本次未做性能对照，少量文本先使用 CPU；模型文件下载主要受网络影响。切换 CUDA 需要兼容硬件、驱动与对应 PyTorch 构建，不能只改字符串。

归一化把向量的数学长度变为 1，不改变维度。对归一化的一维数组执行 `@`，相当于对应元素相乘后相加，得到余弦相似度。循环从索引 1 开始，避开问题与自身的比较；分数仅显示到四位小数。

#### 三组实际实验结果

三个候选固定如下，每组只替换 `texts[0]` 中的问题：

- A：`We trained our models on one machine with 8 NVIDIA P100 GPUs.`
- B：`The encoder is composed of a stack of N = 6 identical layers.`
- C：`今天晚饭准备吃番茄鸡蛋面。`

| 中文问题 | A：GPU 硬件 | B：编码器层数 | C：晚饭 | 实际第一名及解释 |
| --- | --- | --- | --- | --- |
| 训练这个模型使用了什么硬件？ | **0.8107** | 0.7666 | 0.7980 | A，包含硬件依据，第一名符合预期；C 仍高于 B，不能据此认定整体排序可靠。 |
| 编码器由多少个相同的层堆叠而成？ | 0.7220 | **0.8100** | 0.7616 | B，包含 6 层的依据，第一名符合预期。 |
| 模型训练实际消耗了多少千瓦时的电能？ | 0.7871 | 0.7532 | **0.8000** | C，无关候选排第一；三个候选都没有实际耗电量答案。 |

以上为学习者实际运行并提供的结果，助手收尾时用本地缓存模型复跑，四位小数结果一致。保留第三组作为失败案例，没有更换措辞或候选来隐藏错误。

相似度不等于答案正确概率，不能将 `0.8` 直接设为“存在答案”的标准。检索排序可能出错；即使手动选回主题相关的 A，它也只有 GPU 数量和型号，没有实际 kWh。学习者回答“不能，因为虽然可能正相关，但是这个资料回答不了问题”；助手补充这里更准确的用语是“语义相关”，而非统计学的正相关。

#### 助手验证结果

- 阅读保存的脚本，确认问题、候选、前缀、CPU 设置、归一化和点积索引正确。
- 使用缓存模型在离线模式执行当前练习，验证结果形状为 `(4, 384)`，全部数值有限，四个向量的范数在误差范围内为 1，问题向量与自身的点积接近 1。
- 复用该模型另行编码硬件与层数问题，得到 `(2, 384)`，三组相似度与学习者报告一致；运行时最大输入长度为 512 tokens。
- `pip check` 输出 `No broken requirements found.`，95 条依赖记录与当前 `pip freeze` 快照一致，原有 68 个包未更改版本。
- 验证通过终端临时脚本执行，没有新增测试文件，没有调用远端聊天 API，也没有重新建立独立虚拟环境或重复验收聊天页面。

#### 当前限制与下一次学习

Day 10 学习任务已完成。当前仅比较三个手写候选，不能用这几次结果概括模型整体质量；尚未批量编码论文、持久化向量、建立完整检索流程或自动校验回答依据。模型名称和库版本已记录，模型仓库 revision 尚未固定，不保证未来不同模型快照的结果逐位一致。

**下一次从 Day 11：建立向量索引开始。** 先解决不同分块配置的 `chunk_id` 重复问题，再把论文片段的向量与来源信息一同保存，检查重新加载后仍可查询。入库时应记录模型及向量化配置，并注意 512 tokens 的限制与字符分块上限不是同一单位。

### Day 11｜2026-10-18｜分块配置标识、FAISS 索引与本地保存

- **当天目标**：区分不同分块参数的片段，生成整篇论文向量，保存索引和原文记录，再从新进程加载并查询。
- **本次状态**：已完成，总进度为 11 / 42。
- **实际用时**：未记录。
- **当天剩余必做任务**：0。

#### 完成的任务

- [x] 学习者提出把分块长度和重叠量加入片段标识，并完成对应修改。
- [x] 在片段字典中保存 `chunk_size`、`chunk_overlap`，修正从页面字典错误取值的问题。
- [x] 验证 400 / 100、800 / 100 和 800 / 50 配置的标识互不重复，来源字段与原页面仍匹配。
- [x] 安装 `faiss-cpu==1.15.1`，确认导入版本与依赖检查通过。
- [x] 理解文件名只能定位文档，还需保存片段原文、页码、标识以及向量与记录的对应顺序。
- [x] 补齐带 `passage: ` 前缀的文本列表，将 51 个片段编码成 `(51, 384)` 数组。
- [x] 用 `IndexFlatIP` 建立索引，转换为 `float32` 后加入归一化向量。
- [x] 编写 `save_index()` 与 `load_index()`，保存索引、完整片段记录和编码配置，并进行基础检查。
- [x] 在独立读取脚本中恢复 51 个向量和记录，用第三个已存向量检索回自身，读取来源与页码。
- [x] 完成实际文件、临时往返和异常分支验证，更新依赖、忽略规则、重建说明和本次学习记录。

#### 代码与文档改动

| 文件 | 改动及原因 |
| --- | --- |
| `chunker.py` | 学习者在标识中加入 `s{chunk_size}:o{chunk_overlap}`，并把两个参数写入片段字典；当前记录包含八个字段。 |
| `practice_chunks.py` | 学习者增加每套配置的第一条片段标识输出，便于观察参数如何进入标识。 |
| `practice_index.py` | 学习者按框架补齐文本列表构造，串联 PDF 读取、分块、CPU 向量编码、FAISS 索引与保存。 |
| `index_store.py` | 学习者根据讲解编写可复用的保存与读取函数，检查数量、格式版本和维度。 |
| `practice_load_index.py` | 学习者新增独立读取与查询检查脚本，通过位置找回对应片段记录。 |
| `requirements.txt` | 助手补入 `faiss-cpu==1.15.1`，共 96 条指定版本记录，其余 95 条不变。 |
| `.gitignore` | 助手新增 `data/indexes/`，保留本地索引文件并忽略提交，已有 `practice_*` 规则不变。 |
| `readme.md` | 助手更新 Day 11 完成状态、当前能力、八个片段字段、文件结构及两个完整练习脚本的重建步骤。 |

核心代码由学习者编写和运行；助手负责讲解、审阅、临时验证及依赖和文档整理。七个练习脚本继续保留在本地，不自动删除；正式保存模块不受 `practice_*` 忽略规则影响，应随项目源码提交。本次未执行 Git 提交或推送。

#### 学到的概念与调试过程

**参数与页面记录是不同的数据来源。** 新的 `chunk_id` 拼接正确，但学习者最初写成 `"chunk_size": page_record["chunk_size"]`，触发 `KeyError: 'chunk_size'`。PDF 页面字典只有文档标识、路径、页码和文本；分块长度与重叠量来自 `split_pages()` 的参数。改为 `"chunk_size": chunk_size`、`"chunk_overlap": chunk_overlap` 后，片段生成正常。

**片段标识与索引位置各有用途。** 学习者最初认为需要保存文件名。进一步讲解了同一篇论文含多个片段，必须保留原文、页码和片段标识。当前 FAISS 索引按加入顺序使用从 0 开始的位置，与 JSON 中的 `chunks[i]` 对应；字符串标识用于追溯，FAISS 不会自动据此去重。构建后独立重排记录会破坏对应关系。

**向量模型和 FAISS 分工不同。** 模型负责将文本编码为 384 维向量，FAISS 接收向量并组织查询。`IndexFlatIP` 使用内积；当前向量归一化后，内积等于余弦相似度。构造索引时传入的是维度，`ntotal` 表示已加入的向量数量。

**保存与读取。** `Path` 处理路径，`mkdir()` 创建目录，`faiss.write_index()` 与 `read_index()` 处理索引文件。`json.dump()` 和 `json.load()` 面向文件，之前的 `json.loads()` 面向字符串。JSON 还记录格式版本、维度和编码配置，便于后续查询使用一致的模型、前缀及归一化方式。

**查询结果到原文。** `reconstruct(2)` 取出第三个向量，`reshape(1, -1)` 整理成一行查询，`search(..., k=1)` 返回分数与位置。`positions[0][0]` 表示第一条查询的第一个结果位置，由它读取片段的来源和页码。

#### 实际结果

| 检查项目 | 结果 |
| --- | --- |
| 400 / 100 分块 | 113 个片段，第一条标识含 `:s400:o100:p1:c1`。 |
| 800 / 100 分块 | 51 个片段，第一条标识含 `:s800:o100:p1:c1`。 |
| 仅改变重叠的 800 / 50 对照 | 50 个片段，第一条标识含 `:s800:o50:p1:c1`；三套标识两两无交集，配置字段正确。 |
| 本次编码输入长度 | 用模型 tokenizer 检查 51 个输入，包含前缀和特殊标记后为 63～268 tokens，均不超过 512。 |
| 学习者构建索引 | 片段数 51，向量形状 `(51, 384)`，索引维度 384，索引内向量数 51。 |
| 保存目录 | `data/indexes/day11_s800_o100/`，包含 `index.faiss` 和 `metadata.json`。 |
| 学习者重新读取 | 读取 51 个向量、51 条片段记录、384 维；没有重新生成向量。 |
| 第三个向量查询 | 输入与结果标识均为 `demo_project:attention_is_all_you_need_neurips2017:s800:o100:p1:c3`，相似度显示 1.0。 |
| 来源 | `data/papers/attention_is_all_you_need_neurips2017.pdf`，PDF 页序号为 1。 |

第三套分块参数和输入 token 长度由助手检查；构建、保存、新脚本读取与单条查询由学习者实际运行并报告，助手随后复核。相似度 1.0 是已存向量查询自身的结果，不是论文问答正确率。

#### 助手验证结果

- 保存文件的 51 条记录与重新读取 PDF、分块得到的记录逐条一致，包括顺序、八个字段与原文。
- 恢复的索引为 `IndexFlatIP`，维度和数量正确，编码配置与构建脚本一致；全部向量为有限的 `float32` 数值，范数接近 1。
- 用全部 51 个已存向量查询，每条最高相似度均接近 1，匹配向量与查询向量在浮点误差范围内相同。
- 在新的 Python 进程中运行学习者的读取脚本，输出与报告一致，读取阶段不调用 Embedding 模型。
- 在临时目录中保存并读回三条合成单位向量，向量、包含非 ASCII 字符的片段记录和编码配置均保持一致。
- 保存时向量与记录数量不一致，会在创建目标目录前抛出 `ValueError`；读取时不支持的版本、维度不一致、记录数量不一致三种情况也均被拒绝。
- 原有索引文件哈希在验证前后保持一致；临时文件已清理，没有新增持久化测试文件。
- `pip check` 通过；96 条依赖与当前环境快照一致，仅新增 FAISS。本次未重新创建独立虚拟环境，也未重复验收聊天页面或调用远端 API。

#### 当前边界与下一次学习

Day 11 必做任务已完成。标识中的分块参数解决了当前固定论文与算法下的跨参数重复，尚未表达文档内容或分块规则版本。改变配置时需要换保存目录；保存函数会覆盖同路径文件，未实现增量更新和事务写入。

读取时的版本、维度、数量检查不是完整的文件一致性验证，无法发现同数量记录被打乱或同维度模型向量被替换。当前检查针对本项目生成的索引，保留向量与记录的配套关系仍是调用方责任。输入长度检查是本次验收中的临时检查，尚未封装进构建脚本；模型仓库 revision 也尚未固定。

**下一次从 Day 12：完成 Dense 检索开始。** 使用保存的编码配置，把中文问题转换成向量，查询已保存的索引，按返回位置取回片段原文和来源，用已准备的问题检查依据片段是否进入候选。Day 10 的无答案与排序失败现象仍需关注，不能把相似度当作证据充分性的判断。

### Day 12｜2026-10-19｜中文问题检索、命中检查与函数封装

- **当天目标**：把用户问题编码后查询已保存的索引，核对候选原文与来源，再将检索逻辑封装成可供项目调用的函数。
- **本次状态**：已完成，总进度为 12 / 42。
- **实际用时**：未记录。
- **当天剩余必做任务**：0。

#### 完成的任务

- [x] 使用保存的模型名、问题前缀与归一化配置，生成形状为 `(1, 384)`、类型为 `float32` 的问题向量。
- [x] 用 `k=3` 查询本地索引，通过返回位置取回片段原文、标识、来源与 PDF 页码。
- [x] 区分 Top-1 与 Top-3 命中，对三条有答案问题逐条核对依据，保留排序不佳和漏检结果。
- [x] 检查实际耗电量这一无答案样例，区分语义相关与足以回答。
- [x] 新建正式模块 `retriever.py`，封装 `retrieve_dense()`，返回包含 `rank`、`score`、`chunk` 的列表。
- [x] 修正循环内用字典覆盖结果列表和名次从 0 开始的问题，使用列表追加并从 1 记录名次。
- [x] 将 `practice_retrieval.py` 改为调用正式函数，验证耗电量问题的候选顺序、分数与来源保持一致。
- [x] 完成四组真实问题复核、边界检查及依赖检查，更新 README，并总结上次同步后的工作。

#### 代码与文档改动

| 文件 | 改动及原因 |
| --- | --- |
| `retriever.py` | 学习者新增正式 Dense 检索函数，复用调用方传入的模型与索引，验证问题和数量，返回结构化候选。 |
| `practice_retrieval.py` | 学习者新增问题向量化与检索练习，随后改为调用正式函数并负责打印结果；当前保留耗电量问题。 |
| `readme.md` | 助手更新 Day 12 状态、当前能力、接口与完整运行示例，追加本次结果及待提交工作范围；将已讨论的索引目录命名安排写入 Day 15。 |

今天没有新增依赖或重建论文索引。八个练习文件继续按既有 `practice_*` 规则保留在本地；新增正式模块不受此规则影响。原有三个练习文件仍按维护约定保留，本次未执行清理、Git 提交或推送。

#### 学到的概念与调试过程

**查询向量。** 问题加上保存配置中的 `query: ` 前缀后编码，`(1, 384)` 表示一条向量、384 个数值维度。将问题放在列表中传给 `encode()`，得到供 FAISS 查询使用的二维数组；文档向量继续使用 Day 11 的保存结果。

**名次与片段位置。** `scores[0]` 和 `positions[0]` 分别是第一条查询的候选分数与索引位置。名次用于显示先后顺序，索引位置用于读取 `chunks[position]`；记录中的 `pdf_page` 才是论文来源页码。

**Top-1 与 Top-3。** 硬件问题的第一名是参考文献，第二名包含直接依据，因此 Top-1 未命中、Top-3 命中。学习者最初将二者都判断为未命中，经原文对照后明确：前三条中至少一条有依据就算 Top-3 命中。编码器问题的依据位于第一名，两个检查项均命中。其他候选无关不影响这一命中判断。

**检索漏检与资料不足。** 位置编码问题在论文第 6 页已有依据，但前三候选均为参考文献，属于漏检。耗电量问题虽然检索到硬件、训练时间和 FLOPs，但没有实际电量或足够的功率信息；学习者判断这些片段无法回答。保留原始结果，没有调大候选数或改写问题来掩盖失败。

**收集结果与返回结果。** 初版在循环内执行 `results = {...}`，使列表变成字典，每轮覆盖前一条，最终只返回最后一条记录；初版还直接保存从 0 开始的 `rank`。改为构建单条字典、`results.append(result_records)` 和 `rank + 1`，循环结束后再返回列表。打印逻辑放在调用脚本中，正式函数可供后续页面复用。

#### 四组实际结果

保持同一模型、800 / 100 分块索引与 `k=3`。以下标识省略共同前缀 `demo_project:attention_is_all_you_need_neurips2017:s800:o100:`；分数展示四位小数，函数返回值保留原始精度。

| 本次问题原句 | 第 1 名 | 第 2 名 | 第 3 名 | 依据核对 |
| --- | --- | --- | --- | --- |
| 训练这个模型使用了什么硬件？ | `p11:c4`，0.8250 | `p7:c4`，0.8243 | `p10:c3`，0.8241 | Top-1 未命中；Top-3 命中，第二条包含 8 块 NVIDIA P100 GPU 的原文。 |
| 编码器由多少个相同的层堆叠而成？ | `p2:c7`，0.8109 | `p10:c3`，0.7878 | `p5:c2`，0.7837 | Top-1、Top-3 均命中，第一条明确写出 6 层。 |
| 论文中的位置编码使用了什么函数？ | `p10:c1`，0.8115 | `p11:c2`，0.8016 | `p11:c3`，0.7982 | Top-1、Top-3 均未命中；依据仍在索引中的第 6 页片段里。 |
| 模型训练实际消耗了多少千瓦时的电能？ | `p7:c4`，0.8284 | `p8:c3`，0.8238 | `p8:c1`，0.8130 | 相关片段仍不足以确定实际耗电量；不计入有答案问题的命中统计。 |

这些问题沿用 `data/sample_questions.json` 的参考依据，但问法并非逐字相同；本次没有修改原样例集。后续复测应记录具体问法和配置，不能将不同表述的结果直接视为同一次实验。

#### 助手验证结果

- 在本地缓存、CPU 环境中直接调用正式函数，复核四条本次问题原句；每组均返回列表、3 条记录、从 1 开始的名次及降序分数，候选标识和四位小数分数均与学习者输出一致。
- 三条有答案问题的依据按空白归一化后与片段原文核对，同时检查来源页码；复现硬件问题第二名命中、编码器问题第一名命中、位置编码问题前三未命中。确认位置编码依据仍存在于已保存片段中。
- 核对每条结果的完整片段记录与保存的元数据一致，查询前后元数据未被修改；索引和 JSON 文件的哈希保持不变。
- 用两条合成向量检查请求 9 条时只返回 2 条，名次和对应记录正确；请求 1 条时只返回首条。检查问题首尾空白会被去除，并使用配置中的前缀及归一化开关。
- 空字符串、纯空白问题，以及 `top_k` 为 0、负数、小数、布尔值、字符串或 `None` 时均抛出 `ValueError`，未调用模型；空索引返回空列表，同样未调用模型。
- `pip check` 通过。检查以临时终端脚本执行，没有新增持久化测试文件；没有重新生成论文向量、调用 DeepSeek、复测聊天页面或重建虚拟环境。

#### 上次同步后的工作范围

收尾时，本地 `main` 与远端跟踪引用 `origin/main` 均指向 `fefc9f6`（添加读取 pdf 论文逻辑，使用 langchain 工具进行文本切割），领先与落后提交数均为 0；该提交中的学习进度为 Day 09。以下总结以本地记录的同步点为基准，Day 10～12 的改动尚未提交。

| 范围 | 完成内容 |
| --- | --- |
| Day 10：文本向量 | 使用 `multilingual-e5-small` 在 CPU 上编码中英文文本，理解 384 维向量、前缀、归一化和点积，记录三组排序结果及失败案例。 |
| Day 11：索引保存 | 片段标识和记录加入分块长度与重叠量；为 51 个片段建立 FAISS 内积索引；新增 `index_store.py` 保存、读取配套向量与 JSON 元数据，检查版本、维度及数量。 |
| Day 12：问题检索 | 新增 `retriever.py`，完成中文问题向量化、候选返回和来源追溯；核对四组问题并保留真实失败结果，验证数量与空输入边界。 |
| 运行与记录 | `requirements.txt` 相较同步点新增 28 条依赖记录，共 96 条；补充 CPU 安装源与运行示例，忽略生成的 `data/indexes/`，维护 Day 10～12 的学习日志。 |

本次待提交的正式文件为 `.gitignore`、`chunker.py`、`index_store.py`、`retriever.py`、`requirements.txt` 和 `readme.md`。练习脚本与生成索引按既有忽略规则保留在本地，README 提供相应重建示例。

#### 当前边界与下一次学习

Day 12 必做任务已完成，完成标准是检索流程、来源对应和实验记录正确，并不意味着每个问题的答案都能被召回。当前函数不判断证据是否充分，也未校验模型身份、元数据完整性或长问题输入；调用方须提供配套的模型、索引与元数据。尚未接入 Streamlit 问答页面，也没有 BM25、RRF、Rerank 或 Agent 路由。

**下一次从 Day 13：跑通基础 RAG 开始。** 把检索片段与用户问题交给现有回答模块，在页面展示回答和对应来源，并检查证据不足时的说明。索引目录的正式命名按已讨论安排放在 Day 15，当前目录继续用于学习验收。
