# Paper Sage

通过亲手构建论文问答工作台，逐步学习 LLM 应用、RAG 和 Agent 开发。

最终目标是基于 **Streamlit + LangChain + LangGraph**，实现按项目组织论文、限定检索范围、自动选择工作流程，以及可追溯到论文原文的证据回答。检索层已实现 **Dense + BM25 双路召回 → RRF 排名融合 → Rerank 精排**。

**当前阶段：Day 28 已完成参数对照与复验，工作台采用质量优先的 B 配置；下一步学习把检索封装为 Agent 工具。** 工作台支持项目创建与切换、PDF 导入查重、解析分块、索引保存、文档删除与失败清理重试，以及会话和文档检索范围隔离。当前检索流程是同一范围内 Dense 与 BM25 各召回最多 6 条，去重后 RRF 保留最多 6 条交给本地 BGE 精排，最终最多选 3 条供 DeepSeek 回答。新消息记录六个阶段的数量、耗时与快照，按片段标识关联各路排名，并保留未入选候选；只有成功回答才保存本轮聊天与追踪记录。Day 28 固定新题复验中，四道有答案题的最终平均 Recall@3 / MRR@3 从 0.50 提升到 0.75，五题平均耗时从约 1.24 秒增加到 1.84 秒；这是小样本检索结果，不是回答正确率。Day 27 的 20 题检索报告及 Day 14 的历史回答评估继续保留。Agent 工作流、自动引用校验和长期会话保存待实现。

Day 21 已保存两条用户回传的实际回答、磁盘来源核对与四组隔离验收结果。学习者编写的归属断言用于检查每条候选；自动检查采用真实页面和 FAISS、控制向量与模拟回答，验证项目切换、文档选择、空选择和跨项目删除隔离，不作为语义召回或模型回答质量评估。

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

**当前进度：已完成 28 / 42 个学习日。Day 28 完成参数传递、开发题对照、复验前选择 B、五道固定新题复验与结果复盘，工作台已采用每路 6 条、RRF 最多 6 条、最终最多 3 条的配置。16 个不同页面场景经检查和目标复验验证，原有数据与实验报告保留。下一次学习 Day 29：把检索封装为模型可以请求调用的工具。** 此数字只表示学习日进度，不代表项目功能完成比例；旧评估与每日学习记录继续保留。

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
- [x] **Day 13｜跑通基础 RAG**：将检索片段、问题和回答要求组合后发送给模型；展示回答及对应片段。验收：真实页面回答编码器为 6 层并引用原文，耗电量问题说明资料不足；针对模型仍附引用的违规结果补充字段组合校验。固定数据与页面模拟验证合法结果、违规拦截、两轮来源保留及清空通过。
- [x] **Day 14｜建立 RAG 评估基线**：完成 20 题，覆盖单主题事实、多处依据组合和无答案问题，保存首轮结果、检索命中、回答正确性与引用支持。验收：汇总为 20 次提交、16 次正常返回、13 题回答正确；保留漏检与部分作答触发校验的案例，标准题、原始观测、核对和汇总分开保存。

### 第三阶段：项目与范围管理

- [x] **Day 15｜设计持久化数据模型**：整理 Project、Document、Chunk 的关系；为项目和文档元数据引入本地持久化，按项目、文档与索引版本规范索引目录命名。验收：学习者完成路径函数、JSON 保存读取及页面接入；新进程读到项目字典和 51 条片段，新旧索引文件完全一致，重启页面后显示可检索片段 51。
- [x] **Day 16｜创建与切换项目**：实现项目创建、列表与切换；展示当前项目的文档。验收：学习者确认页面创建、自动选中、空名称拒绝及两个项目切换；新进程读到三个已保存项目。修复提前使用变量后，AppTest 验证文件名随项目变化、空项目停止加载，以及切换时聊天与来源清空；未调用模型 API。文档上传与索引构建在 Day 17 接入。
- [x] **Day 17｜文档导入生命周期**：将上传、解析、分块和索引串起来；处理重复上传和失败状态。验收：学习者在 test 项目完成真实导入，文档数为 1、片段数为 51；再次上传提示已有。助手在隔离目录完成 8 组检查，覆盖改名查重、完整导入、保存与解析失败、无可用文字、索引与登记写入失败及重试；没有调用 DeepSeek。
- [x] **Day 18｜限定检索文档范围**：支持当前项目全部论文或选定论文，仅加载所选索引并在内存中合并检索。验收：学习者在同一项目完成两份不同 PDF 的三组问答，片段数量为 1、51、52；助手核对完整片段归属，并用实际索引验证全部模式、空选择、历史清理和等价范围缓存复用，未新增 DeepSeek 调用。
- [x] **Day 19｜隔离项目与会话上下文**：定义带编号与范围归属的会话记录，集中处理继续、切换、清空和历史校验。验收：学习者完成三个核心函数；实际页面配合真实索引、回放向量和模拟回答通过九组检查，确认历史与候选来源不跨范围、清空与错误处理正确，真实 DeepSeek 调用为 0。
- [x] **Day 20｜删除文档与同步索引**：记录移除、所属 PDF 与全部版本索引清理、失败待办和重试，以及页面选择、缓存和会话同步均已接入。验收：10 项文件检查与 8 组页面检查通过；学习者实际删除测试 PDF 后回传 1 份文档、51 个片段及输入框清空，助手核对记录、文件与索引一致，素材原件保留。
- [x] **Day 21｜项目管理阶段验收**：准备同名、相近内容的人工 PDF，学习者实现候选归属断言并完成 A、B 两次实际问答及切换清空确认。验收：磁盘来源核对和项目切换、范围选择、空选择、跨项目删除四组自动检查通过；保存人工回传与自动检查的独立记录，助手未调用模型。

### 第四阶段：混合检索

- [x] **Day 22｜BM25 关键词检索**：完成英文与数字词项处理、BM25 打分和候选排名，保留完整片段来源。验收：真实论文检索通过，四项边界检查通过；保存三条固定查询的 Dense 与 BM25 对比，明确当前分词的中文限制及有候选不等于有答案。
- [x] **Day 23｜统一双路召回接口**：统一入口、候选去重、两路排名与来源保存、页面接入均已完成；12 项检查通过，覆盖范围、失败拦截和空匹配。学习者实际页面确认 Dense 与 BM25 各 3 条、去重后 4 个片段，同一片段保留两路独立排名和分数。
- [x] **Day 24｜实现 RRF 排名融合**：手算与程序对照通过，完成打分、排序、同分规则、数量限制、来源保留及页面参考资料切换，17 项检查已验证。学习者确认实际页面使用三个融合片段、常数为 60，按顺序显示的 RRF 分数为 0.032522、0.032002、0.016393。
- [x] **Day 25｜加入 Rerank 精排**：输入对、模型打分、排序、真实候选对比与页面接入已完成，分别限制融合候选和最终返回数量，保留前后排名、分数、来源与耗时。21 项检查通过；学习者实际页面确认 4 条精排后选出 3 条、原 RRF 名次为 1 / 3 / 2，本次精排耗时 1.212 秒（不含模型加载）。
- [x] **Day 26｜观察完整检索流程**：计时与片段标识关联练习、六阶段记录及页面接入已完成，24 个不同场景经检查与复验验证通过。学习者回传六个阶段的数量、耗时和四条排名路径，确认 `p10:c3` 在最终截取时未入选；能够判断入选参考资料不保证支持回答。
- [x] **Day 27｜完成检索对照实验**：完成 Recall、RR、MRR 计算、全范围标注、输入检查和四种方法对照。学习者正确判断汇总指标的不同变化，并在纠正讲解后确认理解 q3 的首个命中提前、q17 的相关片段被排到第四名后落选，以及 Recall 分母和 Top-K 范围。报告保留 20 题的候选、排名和阶段记录，15 道有答案题参与指标平均。
- [x] **Day 28｜调整检索参数并复盘**：完成每路召回 3 / 6 的单参数对照，学习者明确质量优先并在复验前选择 B。五道新题复验中，四道有答案题的最终平均 Recall@3 / MRR@3 从 0.50 增至 0.75，平均耗时增加约 0.60 秒；v4 仍遗漏。工作台采用 B，16 个页面场景经检查与目标复验验证，保留原基线和全部报告，不继续在复验题上挑参数。

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
├── app.py                    # 项目管理、混合检索问答、来源及完整检索流程展示
├── chat_session.py           # 会话创建、范围切换及模型历史的归属检查
├── llm.py                    # 调用 DeepSeek、检查结束原因，返回校验后的字典
├── answer_parser.py          # JSON、字段类型及资料不足时依据为空的校验
├── pdf_loader.py             # PDF 逐页读取，返回保留来源的页面记录
├── chunker.py                # 逐页分块，保留项目、文档、页码及片段标识
├── document_importer.py      # 文件摘要、项目内查重、PDF 保存及正式索引构建函数
├── document_deleter.py       # 文档移除、路径归属检查、清理待办与重试、导入登记协调
├── index_store.py            # FAISS 索引与 JSON 元数据的保存、读取和基础检查
├── project_store.py          # 项目创建与列表、JSON 临时写入后替换、规范索引路径
├── retriever.py              # 问题向量化、Dense 检索，返回分数与完整片段记录
├── bm25_retriever.py          # 英文与数字词项处理、BM25 排名及完整片段返回
├── dual_retriever.py         # 双路统一入口、结果归属检查与按标识合并候选
├── rank_fusion.py            # RRF 打分、排序、数量限制及来源保留
├── reranker.py               # 本地精排模型加载、候选打分与排序，保留原 RRF 信息
├── retrieval_trace.py        # 阶段计时、独立快照与片段排名路径汇总
├── config.example.toml       # 本机模型缓存目录配置模板
├── config.local.toml         # 本机缓存位置，不提交 Git
├── compare_retrievers.py     # 在相同范围比较三条固定查询，保存独立检索报告
├── retrieval_scope.py        # 选择项目内文档，校验并合并所选索引及片段
├── rag_context.py            # 将检索片段与来源组装为模型参考资料字符串
├── practice_json.py          # 本地 JSON 解析练习，不调用模型
├── practice_errors.py        # 本地异常传播与历史记录练习，不调用模型
├── practice_pdf.py           # 打印 PDF 页面摘要，检查未提取到文字的页码
├── practice_chunks.py        # 比较两套分块配置，查看 GPU 样例所在片段
├── practice_embeddings.py    # 本地向量与相似度练习，当前保留耗电量问题
├── practice_index.py         # 为 51 个论文片段生成向量，建立并保存索引
├── practice_load_index.py    # 在新进程读取索引，检查向量与来源对应关系
├── practice_retrieval.py     # 打印检索候选，并查看组装后的参考资料字符串
├── practice_bm25.py          # 真实论文的 BM25 检索练习，打印分数与完整来源
├── practice_dual_retrieval.py # 回放 Day 22 候选，验证去重及两路排名保存
├── practice_rrf.py           # 两组手算对照及已保存真实候选的 RRF 回放
├── practice_rerank.py        # 输入对、本地精排打分及后续排序练习
├── practice_rerank_candidates.py # 四条已保存真实候选的 RRF 与精排对比
├── practice_retrieval_trace.py # 阶段输入输出数量、耗时与结果快照练习
├── retrieval_metrics.py      # 复用学习者的指标函数与已完成的片段标识读取
├── evaluate_retrieval.py     # 四种方法固定题集对照；支持 --check 输入检查
├── compare_recall_limits.py  # 三道开发题的召回上限 3 / 6 对照，记录阶段命中与耗时
├── validate_recall_limits.py # 校验事先选择，在固定五道新题上复验 A / B
├── practice_retrieval_metrics.py # 从正式模块导入指标，保留原教学样例
├── practice_relevance_labels.py # 展示三个真实片段与助手已记录的依据判断
├── practice_retrieval_tuning.py # 模拟检查每路召回参数传递，不加载模型
├── practice_evaluation.py    # 汇总五批评估记录，统计总数、正常返回与回答正确数量
├── requirements.txt          # 当前 Windows / Python 3.12 环境的依赖版本快照
├── readme.md                 # 学习路线、任务状态与每日学习记录
├── tests/
│   ├── test_document_deletion.py # 临时目录中的删除与失败恢复检查，不接触实际项目
│   ├── test_project_scope.py # 复用学习者归属断言，四组可重复运行的隔离验收
│   ├── test_bm25_retriever.py # 排名与来源、空匹配、输入边界及零分/负分匹配检查
│   ├── test_dual_retriever.py # 接口与合并检查、可复用的双路页面范围验收
│   ├── test_rank_fusion.py   # 八项融合检查、九组范围与参考资料页面验收
│   ├── test_reranker.py      # 七项精排与配置检查、十四组页面与历史验收
│   └── test_retrieval_trace.py # 阶段快照、完整路径、失败与旧历史兼容检查
├── data/
│   ├── sample_questions.json # 3 条有答案样例与 1 条资料不足样例
│   ├── evaluation_questions.json # 固定 20 题的标准答案、原文依据和不足原因
│   ├── fixtures/
│   │   ├── day18_scope_note.pdf # 明确标注为人工测试资料，用于范围验收
│   │   └── day21/
│   │       ├── manifest.json # 测试用途、固定问题、预期校验码及文件摘要
│   │       ├── project_a/scope_note.pdf # 项目 A 测试素材，1 页、1 个片段
│   │       └── project_b/scope_note.pdf # 同名相近内容，校验码与 A 不同
│   ├── evaluations/
│   │   ├── day14_baseline.json # q1～q4 人工回传，保留运行失败与未获取字段
│   │   ├── day14_q5_q8.json   # q5～q8 真实自动调用，保存候选、回答、引用与核对结果
│   │   ├── day14_q9.json      # q9 人工回传，核对两项依据及不同片段中的等价引用
│   │   ├── day14_q10.json     # q10 人工回传，记录 CPU 型号问题的正常资料不足回答
│   │   ├── day14_q11_q20.json # 最后 10 题真实代跑，保留原始响应、失败与核对结果
│   │   ├── day14_summary.json # 全部 20 题的指标、逐题判定、题集快照及报告哈希
│   │   ├── day21_project_management.json # 人工回答、磁盘核对及四组项目范围验收
│   │   ├── day22_retrieval_comparison.json # 同范围、同查询的 Dense 与 BM25 候选
│   │   ├── day23_dual_retrieval.json # 去重回传、自动检查摘要及实际页面确认
│   │   ├── day24_rrf.json     # 手算回传、候选回放、两次检查记录及实际页面确认
│   │   ├── day25_rerank.json # 用户精排回传、自动检查摘要与实际页面确认
│   │   ├── day26_retrieval_trace.json # 排名路径回传、阶段快照测试与已完成页面确认
│   │   ├── day27_relevance_labels.json # 固定题集的片段依据标注、引用位置与范围摘要
│   │   ├── day27_retrieval_comparison.json # 四种方法的真实检索结果、指标与阶段快照
│   │   ├── day28_tuning_plan.json # 实验前固定的配置、开发题与新复验题标签
│   │   ├── day28_development_comparison.json # 三道开发题的两组实际结果、阶段命中与耗时
│   │   ├── day28_selection.json # 复验前暂选 B、用户的质量优先偏好与来源摘要
│   │   ├── day28_validation_comparison.json # 五道固定新题的实际 A/B 复验
│   │   └── day28_conclusion.json # 采用 B 的依据、未解决问题与页面检查记录
│   ├── papers/
│   │   ├── attention_is_all_you_need_neurips2017.pdf # 原始演示论文
│   │   └── project_<uuid>/
│   │       └── doc_<sha256>/
│   │           └── <文件名>.pdf # 上传后按项目保存的副本
│   ├── projects/
│   │   ├── demo_project/
│   │   │   └── project.json  # 项目名称、文档来源及当前索引版本
│   │   └── project_<uuid>/   # 新建项目，每个项目各有一个目录
│   │       └── project.json  # 新项目的 documents 初始为空列表
│   └── indexes/              # 本地生成文件，不提交到 Git
│       ├── demo_project/
│       │   └── attention_is_all_you_need_neurips2017/
│       │       └── v1/
│       │           ├── index.faiss
│       │           └── metadata.json
│       ├── project_<uuid>/
│       │   └── doc_<sha256>/
│       │       └── v1/       # 上传构建的 index.faiss 与 metadata.json
│       └── day11_s800_o100/   # 保留的旧目录，当前工作台与练习已不使用
├── .gitignore
└── .streamlit/
    ├── config.toml           # 关闭触发可选模块导入的文件扫描；改代码后重启服务
    ├── secrets.toml.example  # 配置模板，仅含占位符，提交到 Git
    └── secrets.toml          # 本地 API Key 配置，不提交到 GitHub
```

当前使用 Python 3.12.6，通过 `langchain-deepseek` 提供的 `ChatDeepSeek` 调用 `deepseek-flash`。`ask_deepseek(messages, api_key)` 接收完整消息列表和密钥，使用 `response_format={"type": "json_object"}` 请求 JSON 输出。拿到响应后先检查 `finish_reason`，仅在其为 `stop` 时调用 `parse_answer()` 解析并校验，向页面返回字典。

当前客户端也可能在 `invoke()` 返回之前抛出长度限制或内容过滤异常，`llm.py` 将这两类异常转换为 `IncompleteAnswerError`，由页面显示提示。认证失败、限流、超时、连接失败、JSON 格式错误和字段校验失败也有相应处理。只有模型调用、结束状态检查和解析校验均通过后，才追加本轮聊天记录及检索片段。

`max_tokens=1024` 限制输出 token 数，不等于整个上下文窗口的长度。`finish_reason="stop"` 表示正常结束或遇到指定停止序列，仍不能证明答案事实正确或证据充分。

当前结果结构：

| 字段 | Python 类型 | 用途 |
| --- | --- | --- |
| `answer` | `str` | 回答正文 |
| `evidence` | `list[str]` | 原文依据列表；资料不足时要求为空，非空不足原因与非空依据同时出现会被程序拒绝 |
| `insufficient_reason` | `str` | 资料不足的原因；资料足够时按提示词要求返回空字符串 |

解析函数检查最外层是否为字典、三个字段是否存在，以及字段和依据列表元素的类型。Day 13 新增组合约束：当 `insufficient_reason.strip()` 非空且 `evidence` 非空时，抛出 `ValueError`。这条检查只拒绝该违规组合，尚不检查资料足够时是否一定有依据、答案是否与不足原因一致，也不验证引用真实性或结论支持情况。

页面将结果组装成 Markdown 字符串保存，并在助手消息中另存本轮 `retrieval_results`。Day 19 起这些消息统一位于 `st.session_state['chat_session']['messages']`，会话同时保存 `session_id` 与 `scope_key`。三个模型输出字段尚未分别存入会话；发送历史前先检查范围归属，再只提取 `role` 和 `content`，当前问题的检索片段则放进本次用户消息的 `<reference>`。来源区域展示的是候选片段，尚未自动把每条原文引用绑定到唯一片段。

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

`ChatDeepSeek` 在本项目中连接 `https://api.deepseek.com`。代码直接使用的第三方包为 `streamlit`、`langchain-deepseek`、`openai`、`pypdf`、`langchain-text-splitters`、`sentence-transformers` 和 `faiss-cpu`（导入名为 `faiss`）；`json`、`pathlib`、`uuid`、`hashlib`、`tempfile` 属于 Python 标准库，无需额外安装。向量与索引构建使用本地计算，不调用 DeepSeek API。

`requirements.txt` 最初由项目 `.venv` 中的 `pip freeze` 导出，Day 08、09 补入 PDF 和分块依赖，Day 10 补入向量模型相关的 27 个包，Day 11 补入 `faiss-cpu==1.15.1`，Day 22 补入 `rank-bm25==0.2.2`。目前共有 97 条指定版本记录，包括间接依赖及环境中的其他已安装包。它不是自动求解生成的锁文件，也不代表最小依赖集合。Day 06 的 66 条依赖已在本机新建的 Windows / Python 3.12.6 环境中验证；后续新增依赖在现有项目环境中验证并通过 `pip check`，未重新创建独立环境。Day 22 已确认 BM25 库可导入，学习者完成三条短文本打分。其他操作系统和 Python 版本尚未验证。

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

Day 25 起，问答还需要本地精排模型。首次配置时执行 `cp -n config.example.toml config.local.toml`，然后在编辑器中设置缓存位置。本机已配置为 `E:/hf-cache/hub`，并有完整模型缓存，可直接启动；其他机器按实际目录填写，空字符串沿用 Hugging Face 默认缓存。`config.local.toml` 已加入忽略规则。

```toml
[rerank]
cache_folder = "E:/hf-cache/hub"
```

若选定目录尚无精排模型，先单独准备缓存：

```bash
./.venv/Scripts/python.exe -X utf8 -c "from reranker import load_reranker, get_rerank_cache_folder; load_reranker(cache_folder=get_rerank_cache_folder())"
```

该命令首次下载约 1.11 GB 的 `BAAI/bge-reranker-base` 权重及配套文件，固定模型版本，后续复用缓存。页面使用 `local_files_only=True`，不会在提交问题时下载权重；缺少模型时显示精排资源提示，且不调用回答模型。缓存不会因关闭程序自动删除；后续精排仍需使用，先保留。向量模型继续使用原有缓存位置，不会因这项配置自动迁移。

启动页面：

```bash
./.venv/Scripts/python.exe -m streamlit run app.py
```

打开终端显示的 `Local URL`。已有虚拟环境和配置时，只需运行启动命令。

页面从 `data/projects/*/project.json` 读取项目列表，通过侧栏创建或选择项目，上传 PDF 后点击“导入论文”。上传入口在空项目判断之前，空项目也能导入；成功后保存项目记录，再按照“全部文档”或“指定文档”加载对应索引。未选文档时提示至少选择一篇并停止问答。Demo project 的索引目录为 `data/indexes/demo_project/attention_is_all_you_need_neurips2017/v1/`；新克隆的仓库须先按下方 Day 11 的更新示例构建这份索引。首次加载模型可能较慢，成功后显示参与检索的文档及片段数量。数据格式和页面操作见下方 Day 15～18 小节。

索引目录不提交到 Git。若克隆后已登记项目缺少索引，需要按各篇文档的来源与版本逐份重建，对应 PDF 必须存在；旧的 `practice_index.py` 默认只处理第一篇文档。重新上传已登记的相同文件会命中查重，不会替代重建。以下是项目根目录中 Git Bash 的按需重建示例，改动 `load_project()` 的参数可选择目标项目：

```bash
./.venv/Scripts/python.exe - <<'PY'
from project_store import load_project, get_index_dir
from pdf_loader import load_pdf_pages
from document_importer import build_document_index

project = load_project("demo_project")
for document in project["documents"]:
    pages = load_pdf_pages(document["source"], document["document_id"])
    output_dir = get_index_dir(
        project["project_id"], document["document_id"], document["index_version"]
    )
    result = build_document_index(pages, project["project_id"], output_dir)
    print(document["document_id"], result)
PY
```

该命令会重新计算并覆盖目标项目各文档当前版本的索引，不修改项目记录，也不调用 DeepSeek；已有正常索引时无需执行。重建后重启工作台，使内存缓存重新加载。

项目新增 `.streamlit/config.toml`，设置 `[server]` 下的 `fileWatcherType = "none"`，规避后台文件监视器扫描 Transformers 模块时触发的可选视觉依赖导入错误。该配置会关闭文件监视；修改代码或配置后，停止并重新启动 Streamlit，确保变更生效。按钮交互仍会触发页面运行，`st.cache_resource` 缓存仍可复用；重启服务会重新初始化资源。[Streamlit 配置说明](https://docs.streamlit.io/develop/api-reference/configuration/config.toml)

如果使用 PowerShell，安装和启动命令也可按以下形式执行；配置文件可在编辑器中从模板复制并填写：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch==2.14.1+cpu" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m streamlit run app.py
```

当前 `.gitignore` 已排除 `.venv/`、`__pycache__/`、`.streamlit/secrets.toml`、`practice_*` 和 `data/indexes/`。练习脚本均只在本地保留、未被 Git 跟踪，因此克隆仓库后不会自动获得它们；下方练习命令以本地已有对应文件为前提，Day 10～12 及 Day 14 附完整示例便于重建。索引文件由练习生成，克隆后按下方步骤重新构建。忽略文件不等于删除文件，原有练习仍按维护约定保留。提交配置模板、依赖文件及正式模块，真实密钥文件和虚拟环境保留在本地；手动通过网页选择文件上传时也应排除它们。

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

当前聊天记录保存在 Streamlit 会话内存中的 `chat_session` 记录里，尚未实现长期持久化；项目和文档信息则已保存到本地 JSON。切换项目或实际范围会新建会话，切回不会恢复旧聊天；点击“清空会话”也会新建编号并清空问题输入框，范围和资源缓存继续保留。页面按项目记录加载现有索引，自动检索论文片段，并将资料与问题一起放入本次用户消息。手动粘贴摘录是 Day 03 的历史练习，当前页面已使用自动检索流程。Day 17 已将 Day 08 的读取函数与 Day 09 的分块函数接入上传导入，生成并保存本地索引；当前回答中的引用由模型生成，未经过程序自动校验。

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

`empty_pages` 收集的是去除首尾空白后文本为空的页面编号。因此 `[]` 表示没有发现这类页面，不表示“没有编码”或“没有信息”。有非空文本也不能证明整页内容、图表、公式和阅读顺序都提取完整。若所有页面都没有可用文字，练习脚本会打印检查原文件的提示；Day 17 的正式构建函数也会在片段列表为空时拒绝建索引。逐页空文本清单仍由练习负责，当前不能自动判断是扫描件、空白文档还是其他解析问题，尚未接入 OCR。

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

克隆仓库后，先按前文安装依赖、准备论文，并确认 `data/projects/demo_project/project.json` 存在（格式见 Day 15 小节），再在根目录建立以下两个本地练习文件。模型首次使用需要下载，之后可复用缓存。以下示例已按 Day 15 更新：从项目记录取得来源与索引版本，通过 `get_index_dir()` 确定保存目录，当前为 `data/indexes/demo_project/attention_is_all_you_need_neurips2017/v1/`。

将下面代码保存为 `practice_index.py`：

```python
import faiss
from sentence_transformers import SentenceTransformer
from index_store import save_index
from project_store import load_project, get_index_dir

from pdf_loader import load_pdf_pages
from chunker import split_pages


project = load_project("demo_project")
document = project["documents"][0]
pdf_path = document["source"]
document_id = document["document_id"]

pages = load_pdf_pages(pdf_path, document_id)

chunks = split_pages(
    pages,
    project_id=project["project_id"],
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

output_dir = get_index_dir(
    project["project_id"],
    document["document_id"],
    document["index_version"],
)

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

预期输出片段数 51、向量形状 `(51, 384)`、索引维度 384、索引数量 51，并生成 `index.faiss` 与 `metadata.json`。重复运行会重新编码并覆盖当前版本目录内的两个同名文件；不同配置或文档内容需使用新的索引版本，片段标识不同并不能防止文件被同路径覆盖。当前练习直接使用项目记录中的版本，没有自动分配版本或构建完成后切换版本的流程；调整配置与版本后须成功完成构建，再重启工作台。

将下面代码保存为 `practice_load_index.py`：

```python
from index_store import load_index
from project_store import load_project, get_index_dir


project = load_project("demo_project")
document = project["documents"][0]
output_dir = get_index_dir(
    project["project_id"],
    document["document_id"],
    document["index_version"],
)

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

**当前边界**：读取函数只做格式版本、维度和数量检查，不能发现同数量记录被重新排列或替换，也不自动核验模型配置与向量是否匹配。两个文件须配套保存，目前没有事务写入或完整性校验；模型仓库 revision 尚未固定。当前示例只有一份论文的索引，没有增量更新、删除同步或范围过滤；Day 13 已接入页面，Day 15 改为按项目记录定位索引。

### 中文问题的 Dense 检索（Day 12）

`retriever.py` 提供 `retrieve_dense(question, model, index, metadata, top_k=3)`。调用方先通过 `load_index()` 加载配套文件，再根据 `metadata["embedding_config"]["model_name"]` 初始化模型；函数复用传入的模型与索引，每次只编码新的问题。查询使用保存的 `query_prefix` 和归一化配置，生成二维 `float32` 数组，通过 FAISS 返回的位置取出原文记录。

返回值是按相似度从高到低排列的列表，每条结果包含：

| 字段 | 含义 |
| --- | --- |
| `rank` | 从 1 开始的名次。 |
| `score` | Python 浮点数，保留原始精度；展示时才取小数位。 |
| `chunk` | 完整片段字典，包含项目、文档、分块参数、来源、PDF 页码、片段标识和原文。 |

`question` 约定为字符串，去除首尾空白后不能为空；`top_k` 必须为正整数，布尔值也被拒绝。空索引返回空列表；请求数量超过索引数量时按实际数量返回。函数负责返回数据，打印由练习脚本负责；Day 13 页面已复用相同接口。

先按 Day 11 步骤生成本地索引，再将以下代码保存为 `practice_retrieval.py`：

```python
from sentence_transformers import SentenceTransformer
from retriever import retrieve_dense

from index_store import load_index
from project_store import load_project, get_index_dir


project = load_project("demo_project")
document = project["documents"][0]
output_dir = get_index_dir(
    project["project_id"],
    document["document_id"],
    document["index_version"],
)
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

Day 12 没有新增依赖，当天环境的 `pip check` 通过。真实问题复核与边界检查见文末学习记录。调用方仍须保证模型、索引和元数据配套；目前未实现额外的模型身份校验、长问题输入检查或元数据完整性校验。Day 13 已完成自动检索接入问答页面；Day 15 已将索引目录改为按项目、文档与版本命名，上述练习同步使用项目记录。

### 基础 RAG 工作台（Day 13）

当前流程：

```text
用户问题 → retrieve_dense(top_k=3) → build_reference_text()
        → 本次 <reference> + 问题 + 系统要求 + 聊天历史
        → ask_deepseek() → 结束状态与 parse_answer() 校验
        → 保存问题、回答及本轮候选 → 展示引用与来源
```

`rag_context.py` 中的 `build_reference_text(results)` 按检索顺序组装字符串，每条包括名次、片段标识、来源路径、PDF 页码和原文，片段之间用空行分隔。空列表返回空字符串；页面遇到空候选时会提前提示，不调用聊天模型。

若要在 Day 12 练习中查看资料组装结果，在导入区增加：

```python
from rag_context import build_reference_text
```

在已有 `results` 生成之后增加：

```python
reference_text = build_reference_text(results)
print("参考资料类型：", type(reference_text))
print("参考资料前 500 个字符：")
print(reference_text[:500])
```

前 500 个字符只用于终端查看；页面会将三条候选组装后的完整字符串交给模型。

页面中三种消息数据的用途：

| 数据 | 用途 |
| --- | --- |
| `user_message` | 本次发送给模型的问题与完整 `<reference>`。 |
| `display_user_message` | 保存并显示用户问题，不包含整段参考资料。 |
| `assistant_message` | 保存回答展示文字及本轮 `retrieval_results`；重运行时仍能展示这一轮的原文与来源。 |

`render_message()` 同时用于历史消息与新消息。回答下的“查看本轮检索片段”显示 Dense 数量、BM25 数量和去重数量。Day 25 的新消息按精排后的排名展示选入参考资料的片段，同时保存并展示原 RRF 排名、两种分数、原文、来源、PDF 页码及各路独立排名和分数，另显示参与精排数量和不含模型加载的精排耗时。旧消息继续按其当时的 Dense 或 RRF 规则展示。页面标注本轮实际使用的参考资料方式。候选中可能混有无关内容，不能把每条候选都当作支持结论的依据。折叠区域不会建立引用与片段之间的自动校验关系。

从 Day 18 起，`load_retrieval_resources(project, selected_document_ids)` 按项目记录（包含索引版本）和排序后的所选标识缓存范围索引与元数据；`load_embedding_model(model_name)` 单独缓存问答向量模型，同一模型可供不同范围复用。清空会话会移除聊天及其来源记录，已加载资源继续复用。Day 20 的页面删除、清理完成和导入成功会显式清除范围索引缓存，保留模型缓存；同一文档删除后重新导入也会加载新索引。手动替换同一路径、同一版本的索引仍需重启服务，尚未自动监测外部文件变更。

Day 25 的 `load_rerank_resources(cache_folder)` 单独缓存精排模型，仅在有候选的问题提交时加载。切换范围、清空会话和删除文档继续复用同一模型；普通页面重运行只重绘已保存的结果，不重新精排。精排加载、运行或分数校验失败时停止本轮，不调用 DeepSeek、不保存半轮对话，也不自动切回其他检索规则。

验收时先清空旧会话，输入“编码器由多少个相同的层堆叠而成？”，核对回答与第 2 页原文；随后输入“模型训练实际消耗了多少千瓦时的电能？”，检查是否说明当前资料不足。若模型返回非空不足原因和非空依据，页面会显示“回答结构不符合要求：资料不足时，evidence 必须为空列表”，且不保存该轮。程序目前不会自动重试或改写这类结果。

真实调用与本地模拟结果分别记录在文末。新增校验后的合规资料不足展示、违规拦截及会话状态由固定数据验证；尚未再次调用真实 DeepSeek 验证修正后的同一问题。后续真实调用仍可能违反提示词，校验不会让模型自动遵守要求。

### RAG 评估记录（Day 14 已完成）

固定问题集保存在 `data/evaluation_questions.json`，共 20 题：11 道单主题题、4 道需要组合多处依据的题，以及 5 道无答案题。标准答案和出处在各批实测前准备；没有为得到正确结果而重试或调整问法。该次历史评估使用同一篇 11 页论文、800 / 100 分块、51 个向量、`multilingual-e5-small` 和 Top-3 Dense 检索。

实际记录按批次保存：

| 文件 | 内容与采集方式 |
| --- | --- |
| `day14_baseline.json` | q1～q4，学习者回传页面结果；保留 q4 校验失败与未获取字段。 |
| `day14_q5_q8.json` | q5～q8，经用户授权通过真实工作台自动调用。 |
| `day14_q9.json` | q9，学习者回传，两处原文共同支持答案。 |
| `day14_q10.json` | q10，学习者回传 CPU 型号问题的正常资料不足回答。 |
| `day14_q11_q20.json` | 本次授权代跑的 10 题，保存真实候选、请求、原始返回、校验结果和助手核对结论。 |
| `day14_summary.json` | 全部 20 题的汇总、逐题判定、题集快照及各报告哈希；不是额外一批运行记录。 |

上述文件均在 `data/evaluations/` 下。各批报告中的题集哈希对应各自运行时版本，历史记录保留原样。前四题样例仍单独保存在 `data/sample_questions.json`。手工回传记录不冒充原始 API 响应，未采集到的数据使用 `null` 并说明原因。

#### 汇总结果

| 指标 | 结果 | 分母含义 |
| --- | --- | --- |
| 正常返回 | 16 / 20（80%） | 全部提交，包含正确和错误的正常回答。 |
| 回答正确 | 13 / 20（65%） | 全部提交；输出校验失败没有计入正确数量。 |
| Top-1 完整支持 | 4 / 15（26.7%） | 仅论文有答案的题，第一候选需支持全部所问信息。 |
| Top-3 完整支持 | 9 / 15（60%） | 仅论文有答案的题，前三候选合起来需支持全部所问信息。 |
| 引用支持 | 9 / 9 | 仅有效页面回答中带引用的 9 题；无引用或回答被拒绝的题不计成功。 |
| 无答案题正确说明不足 | 4 / 5 | 论文确实没有答案的题；q4 因校验失败未计成功。 |

四次输出校验失败的回答正确性为 `null`，另外三次正常返回未答出论文已有答案的正确性为 `false`。只看已取得有效回答的 16 题，正确为 13 / 16，但不能用该比例替代包含全部失败的 13 / 20。这是单篇论文上的教学开发集，每题仅保留一次结果，不代表独立测试集上的泛化表现。

#### 典型结果与判定规则

- **漏检后正常说明不足**：q5（注意力头）、q12（前馈网络维度）、q13（束搜索参数）正常返回，但没有答出论文已有答案。模型依据当前片段说明不足可以合理，整体问答仍判为未完成。
- **部分依据与校验冲突**：q15 只找到优化器，q16 只找到预热步数，q17 只找到 BLEU。模型返回了有依据的部分答案，同时提供非空引用和不足说明，被现有“资料不足时 evidence 必须为空列表”规则拦截。原始返回可供诊断，页面没有保存这些轮次。当前回答结构尚不能接受这种部分作答形式。
- **正常资料不足**：q10（CPU 型号）、q18（RAM 容量）、q19（CUDA 版本）、q20（随机种子）符合预期。它们的检索命中不适用，没有引用也不计引用支持成功。
- **等价依据**：q9 的预选原文来自第 2、7 页，实际两项依据都来自第 2 页不同片段，仍然完整支持答案。不能仅以页码或引用文字是否等于标准摘录判断命中。q9 是跨片段问答，不能据此声称必须跨页才能回答。
- **完整支持与数值线索**：q12 的第三候选 Table 3 含有 512、2048 及参数名，但缺少将参数对应到所问前馈网络输入、输出和内部层的定义。按完整支持标准未计命中，详细理由保存在实测记录中。

无答案题的 `top1_hit`、`top3_hit` 为 `null`。有答案的组合题必须覆盖所有部分，不能只找到一个信息点就计完整命中。跨页标准引用使用 `evidence_pages` 与 `evidence` 逐条对应；单页字段 `pdf_page` 为 `null` 时，应结合 `answerable` 与对应页码列表理解，不能直接推断无答案。

#### 本次 q11～q20 实测

| 题号 | 结果 | 核对结论 |
| --- | --- | --- |
| q11 | 正常、正确 | big 模型 300,000 步、约 3.5 天；补充的 GPU 信息和两条引用也有依据。 |
| q12 | 正常、未答出 | 前馈网络维度所需的完整依据未被召回。 |
| q13 | 正常、未答出 | 三个候选均为参考文献，未召回束宽与长度惩罚参数。 |
| q14 | 正常、正确 | 标签平滑参数为 0.1，第二候选支持。 |
| q15 | 输出校验失败 | 有 Adam 依据，缺少注意力头数量，部分作答触发字段组合校验。 |
| q16 | 输出校验失败 | 有 4000 步依据，缺少位置编码函数，部分作答触发校验。 |
| q17 | 输出校验失败 | 有 base 的 27.3 BLEU 依据，缺少训练步数，部分作答触发校验。 |
| q18 | 正常、正确 | RAM 容量未报告，说明不足。 |
| q19 | 正常、正确 | CUDA 版本未报告，说明不足。 |
| q20 | 正常、正确 | 随机种子未报告，说明不足。 |

本批通过 Streamlit AppTest 驱动原 `app.py`，使用真实向量模型、FAISS 与 DeepSeek。每题新会话，核对实际请求只有系统要求与本题检索资料，没有聊天历史或标准答案；实际调用 10 次，`max_retries=0`，没有额外模型评分。测试脚本最初的请求核对因 PowerShell 管道替换中文标点失败，尚未发出模型请求；改用字符码构造标点后完成正式运行，前置诊断保存在 `preflight_diagnostics` 中，不混入模型评估样本。

#### 复现当前统计

本地 `practice_evaluation.py` 已加入最后一批路径，运行输出为 20、16、13。核心计数逻辑由学习者编写，本次助手仅补入数据文件路径：

```python
import json

report_paths = [
    "data/evaluations/day14_baseline.json",
    "data/evaluations/day14_q5_q8.json",
    "data/evaluations/day14_q9.json",
    "data/evaluations/day14_q10.json",
    "data/evaluations/day14_q11_q20.json",
]

all_records = []
for report_path in report_paths:
    with open(report_path, "r", encoding="utf-8") as file:
        report = json.load(file)
    all_records.extend(report["results"])

ok_count = 0
correct_count = 0
for record in all_records:
    if record["status"] == "ok":
        ok_count += 1
    if record["status"] == "ok" and record["answer_correct"]:
        correct_count += 1

print("记录总数：", len(all_records))
print("正常返回数量：", ok_count)
print("回答正确数量：", correct_count)
```

运行 `./.venv/Scripts/python.exe practice_evaluation.py`。`json.load(file)` 从文件对象读取 JSON，`extend()` 将每条结果加入列表。`answer_correct` 来自已保存的人工或助手核对，计数脚本不会自行判断答案。显式列出五个运行报告，避免将汇总文件或诊断记录当成新增样本。

Day 14 已完成固定 20 题的基线任务，首次失败结果均保留。Day 15 仅调整项目数据存储与索引定位，不重写这些报告中的旧运行路径或历史结果。

### 项目记录与索引目录（Day 15 已完成）

一个 Project 包含多个 Document，一个 Document 经分块后对应多个 Chunk。目前项目和文档记录放在同一个 JSON 中，片段及编码配置继续存放在索引旁的 `metadata.json` 中。

`data/projects/demo_project/project.json` 的内容如下，作为可提交的示例配置保留；新克隆若缺少此文件，可按相同目录和内容创建：

```json
{
  "project_id": "demo_project",
  "name": "Demo project",
  "documents": [
    {
      "document_id": "attention_is_all_you_need_neurips2017",
      "source": "data/papers/attention_is_all_you_need_neurips2017.pdf",
      "index_version": "v1"
    }
  ]
}
```

| 字段 | 用途 |
| --- | --- |
| `project_id` | 项目标识，关联项目记录、片段归属及索引目录。 |
| `document_id` | 文档标识，在项目内关联文档记录及其片段。 |
| `source` | PDF 来源路径，构建索引时用来读取论文，片段中保留用于追溯。 |
| `index_version` | 当前使用的索引构建版本，本次为 `v1`；不是文件格式的 `schema_version`。 |
| `chunk_id` | 具体片段的标识，现有格式含项目、文档、分块参数、页码及页内序号。 |

`project_store.py` 在 Day 15 实现三个基础函数：`save_project(project)` 将字典写入项目 JSON；`load_project(project_id)` 从文件读取并返回字典；`get_index_dir(project_id, document_id, index_version)` 返回 `Path` 对象，路径为 `data/indexes/{project_id}/{document_id}/{index_version}/`。Day 16 又加入项目列表和创建函数，见下节。生成路径不会创建目录，保存函数负责创建所需目录。所有相对路径以项目根目录为运行位置。

工作台在 Day 16 开始读取用户选择的项目，Day 18 改为按所选文档范围加载索引。本地索引练习默认使用 Demo project，页面上传与构建已在 Day 17 接入。项目 JSON 更新先写同目录的唯一临时文件，关闭后替换正式文件；失败时清理临时文件。Day 20 在导入与删除流程增加标识、路径归属及相关记录检查，但底层通用存储函数尚无统一完整模式校验或并发更新协调；FAISS 与索引元数据也未做跨文件事务，`chunk_id` 不作为跨构建版本的唯一标识。

本次将旧目录中的 `index.faiss` 和 `metadata.json` 一起复制到规范目录，未重新生成向量。助手核对两份文件的 SHA-256 完全一致；新进程加载得到 51 个向量、51 条记录、384 维，各片段项目、文档和来源与项目记录一致。学习者重启页面后确认显示“可检索片段：51”。旧目录暂留作迁移对照，当前工作台和三个索引练习均不再引用它。

在新进程中检查项目记录与索引：

```bash
./.venv/Scripts/python.exe -c "from project_store import load_project; project = load_project('demo_project'); print(type(project)); print(project['name']); print(project['documents'][0]['document_id'])"
./.venv/Scripts/python.exe practice_load_index.py
```

现有索引已经迁移完成，无需为了路径变更重跑构建练习。新克隆按前文更新后的构建示例生成索引；构建、读取和检索练习都从同一份项目记录取路径。Day 15 没有新增依赖或模型评估调用，Day 14 基线保持不变。

### 页面项目管理（Day 16 已完成）

`list_projects()` 按路径排序扫描 `data/projects/*/project.json`，复用 `load_project()` 返回项目字典列表；`create_project(name)` 去除名称两端空白，拒绝空名称，用 `project_` 加 UUID 生成标识，并保存 `documents: []` 的新项目。项目名称用于显示，项目标识用于存储和选择，两者用途不同。

侧栏创建表单位于项目列表及选择框之前。提交成功后，代码先保存项目，再把新标识写入 `st.session_state['selected_project_id']`；随后读取最新列表并创建同名 `key` 的选择框，使它自动选中新项目。输入空名称时显示错误提示，不新增项目。项目记录在磁盘上持久化，当前选择和聊天记录仍属于会话状态。

选择后先读取项目，再比较 `active_project_id`，发生切换就清空当前 `messages`。随后显示文档数量：空项目提示“当前项目还没有论文”并停止；有文档时循环显示 `Path(document_record['source']).name`，最后加载第一篇文档的索引。切回先前项目不会恢复已经清空的聊天，按项目分别保存会话尚未实现。

学习者在 Day 16 通过真实页面确认创建成功后自动选中、空名称被拒绝，以及已有论文项目与空项目切换。助手当时在新进程中读到 `Demo project`、`My second project` 和页面创建的 `test`，对应文档数量为 1、0、0。

收尾时发现新增文档代码曾提前使用 `project`，运行触发 `NameError`。学习者将循环移到正确位置后，助手删去残留的提前取文档语句，保留其文档循环。正确顺序是：列出项目 → 选择项目 → 读取项目 → 检查文档 → 展示文档 → 加载索引。

修复后的 AppTest 使用真实应用、项目 JSON 和 FAISS 索引，将向量模型替换为测试对象，禁止检索编码与 DeepSeek 调用。检查通过：Demo project 显示文件名和 51 条片段；同项目重运行保留模拟聊天与来源；切到空项目清空聊天、隐藏旧文件名和问答框且不加载索引；切回后恢复文档与问答入口但不恢复旧聊天；空名称提交显示错误且不增加项目。项目及索引文件内容在检查前后一致。本次验证页面状态与加载流程，不作为新的真实模型问答评估。

Day 16 验收时只有 Demo project 有论文。Day 17 已完成向 test 项目导入论文，主区标题改为显示所选项目名称和实际 PDF 文件名；检索范围选择在 Day 18 实现。没有新增依赖，原练习文件继续保留。

### PDF 导入与失败处理（Day 17 已完成）

在侧栏选择项目，上传 PDF，再点击“导入论文”。文件选择本身只提供字节数据；点击按钮后才保存和建索引。上传控件的 key 包含项目标识，避免不同项目共用所选文件。

```text
上传字节 → SHA-256 摘要 → 当前项目内查重
未重复 → 保存 PDF → 解析页面 → 分块、向量化并保存 FAISS 索引
       → 写入项目文档记录 → 展示可检索片段与问答入口
```

| `document_importer.py` 接口 | 行为 |
| --- | --- |
| `get_file_hash(file_bytes)` | 返回文件内容的 SHA-256 十六进制摘要。 |
| `find_existing_document(project, file_hash)` | 在当前项目查找相同摘要，返回文档字典或 `None`；旧记录没有摘要时读取原文件计算。 |
| `save_uploaded_pdf(project_id, document_id, file_name, file_bytes)` | 保存到项目、文档目录，返回 PDF 的 `Path`。 |
| `build_document_index(pages, project_id, output_dir)` | 使用 800 / 100 分块、本地 E5 small 模型和 FAISS，保存索引并返回片段数、维度。没有可用片段时提前报错。 |

新文档标识为 `doc_{file_hash}`；PDF 保存到 `data/papers/{project_id}/{document_id}/{文件名}`，索引保存到 `data/indexes/{project_id}/{document_id}/v1/`。同一内容改名后仍会命中查重，不同项目可以分别导入。新文档记录保存 `document_id`、`source`、`index_version`、`file_hash`、`page_count`、`chunk_count`；旧的 Demo project 记录保持原格式，由查重函数兼容。

索引保存成功后才登记文档。页面先复制项目字典并生成新的文档列表，`save_project()` 成功后再使用新记录，避免登记失败时页面显示尚未保存的文档。项目 JSON 先写同目录临时文件，再替换正式记录；模拟写到一半失败的检查确认旧 JSON 内容保持完整。

页面通过 `import_stage` 区分保存 PDF、解析 PDF、构建索引和登记文档，显示对应失败提示；查重读取失败也有独立提示。常见 PDF 读取错误通过 `PyPdfError` 纳入处理，详见 [pypdf 异常说明](https://pypdf.readthedocs.io/en/stable/modules/errors.html)。学习时的摘要、字节类型、向量维度等临时输出已移除，保留文件名、导入页数、片段数和状态。

学习者实际将固定论文导入 test 项目，得到 11 页、51 个片段、384 维索引，文档数为 1；再次选择同一 PDF 显示已存在，数量仍为 1。助手重新读取保存的索引与项目 JSON，确认片段归属、来源和配置匹配。

收尾检查使用真实 Streamlit AppTest 上传控件、PDF 解析、分块、FAISS 和文件写入，在工作区临时副本中运行；向量模型使用已保存向量的回放对象，要求输入片段原文完全一致，没有重新执行神经网络编码。8 组结果如下：

| 检查 | 结果 |
| --- | --- |
| 已有记录与旧格式记录，上传改名但内容相同的 PDF | 提示已存在，数量不变，不编码。 |
| 空项目完成导入，再启动新的测试会话 | 保存 1 条记录，加载 51 条片段，再次上传不新增。 |
| 模拟 PDF 写入失败 | 提示保存环节，项目保持为空。 |
| 无效 PDF 字节 | 提示解析环节，不登记。 |
| 可解析但没有文字的空白 PDF | 在编码前拒绝，项目保持为空。 |
| 模拟索引写入失败 | 提示构建索引环节，不登记。 |
| 模拟项目 JSON 仅写入开头后失败 | 提示登记环节，磁盘旧记录完整，页面仍无文档。 |
| 登记失败后重试 | 正常完成导入。 |

检查中 DeepSeek 和问题检索调用均为 0，使用已保存向量回放 4 次；实际项目、PDF、索引和 Day 14 报告内容均未改变，临时副本已清理。这些检查验证导入流程和错误处理，不是新增的模型问答评估。

Day 17 结束时仍查询所选项目的第一篇文档，Day 18 开始接入文档范围选择。Day 20 已补充新导入的清理待办：写 PDF 前保存待办，成功登记文档时移除待办，失败则清理；清理不成功时可在页面重试。既有版本产生、且没有待办记录的历史残留不做自动扫描删除。尚无并发导入协调或跨文件事务，新上传内容的索引版本仍为 `v1`。索引构建和问答资源加载各自初始化模型；Day 18 的模型缓存只复用于不同问答范围，导入与问答共用模型仍留待后续工程阶段。

原有 `practice_json.py`、`practice_errors.py`、`practice_pdf.py` 按约定保留；逐页空文本检查仍由 PDF 练习承接，本日不触发删除提醒。本日没有新增依赖，也未重跑 Day 14 的 20 道题。

### Day 18：限定检索范围（已完成）

`retrieval_scope.py` 提供两个正式接口：

| 接口 | 行为 |
| --- | --- |
| `get_selected_documents(project, selected_document_ids)` | 返回当前项目中所选文档的完整记录，未知标识报错，空选择返回空列表。 |
| `load_scope_index(project, selected_document_ids)` | 拒绝空选择，仅加载所选索引；检查类型、维度、编码配置和片段归属后，在内存中合并向量与片段，返回合并索引及元数据。 |

学习者使用 `combined_index.add(vectors)` 与 `combined_chunks.extend(chunks)` 按相同顺序加入向量和片段。当前仅支持已使用的 `IndexFlatIP`，通过 `reconstruct_n()` 取回原向量，不重新编码论文，不创建新的索引目录。编码配置检查基于已保存字段，目前尚未固定模型 revision。

Day 18 将当前项目标识与排序后的所选文档标识、索引版本、来源组合为范围标识，Day 19 将其统一存入会话的 `scope_key`，替代原先独立的 `active_scope`。实际范围变化先新建空会话，再处理空选择；同一集合只改变选择顺序，或切换全部／指定模式但文档相同，均保留会话。查询复用原来的 `retrieve_dense()`，在所选文档合并后的索引上统一取最多三个候选。

已检查真实 Demo 索引的 51 条向量和片段顺序一致。隔离的两文档索引夹具检查了单选、多选、空选择、范围变更、版本变更、缓存复用及来源约束；该阶段真实 AppTest 中四轮回答使用测试替身，未调用 DeepSeek。随后学习者在 test 项目真实导入第二份 PDF 并完成三组问答，助手使用两份实际索引补查完整片段归属与页面状态，Day 18 验收通过。

双 PDF 验收素材为 [day18_scope_note.pdf](data/fixtures/day18_scope_note.pdf)，它是助手生成的人工测试资料，页内明确标注并非研究论文。用现有解析和 800 / 100 分块配置检查得到 1 页、361 个字符、1 个片段；原文包含 `The verification code of this test document is SCOPE-B-742.`。文件 SHA-256 为 `3dd65aff86fe7cd8dfb4215beb9e7eef645bc2bc13f54b8c00ca2de7163506a6`。

学习者已通过页面将其导入 test 项目，与现有论文处于同一项目。固定问题为“检索范围测试文档的校验码是什么？”，三组实际回传如下：

| 组别 | 范围 | 实际片段总数 | 实际回传 |
| --- | --- | --- | --- |
| A | 仅测试 PDF | 1 | 正确返回 `SCOPE-B-742`，唯一候选来自测试 PDF 第 1 页。 |
| B | 仅原论文 | 51 | 明确说明资料不足，原文依据为空；没有继续回答 A 组的校验码。 |
| C | 指定两份 PDF | 52 | 第 1 名命中测试 PDF，返回 `SCOPE-B-742`，并引用完整原句。 |

用户未单独回传“全部文档”或 B、C 组全部候选列表。助手读取实际两份索引，逐条检查所有片段的项目、文档、来源和数量；用真实页面与真实索引确认全部文档为 52 个片段，单选为 1 或 51，指定两份为 52。此次页面检查替换了向量模型，未提交问题，模型推理和 DeepSeek 调用均为 0。切换范围清空历史、空选择停止问答，以及等价范围保留历史和复用资源均通过；23 个数据文件的路径与内容保持一致。

全部／多选只表示这些文档有资格参与检索，不保证前三候选中每篇都出现。范围约束、语义召回和回答质量分别核对；本次测试资料及两份文档的索引保留，供后续会话隔离和删除同步练习使用。

### Day 19：隔离项目与会话上下文（已完成）

`chat_session.py` 由学习者实现三个核心函数：

| 接口 | 行为 |
| --- | --- |
| `create_chat_session(scope_key)` | 返回带新 UUID 编号、范围标识和独立空消息列表的会话字典。 |
| `ensure_chat_session(session, scope_key)` | 无会话或范围不匹配时新建，相同范围返回原会话。 |
| `build_history_messages(session, scope_key)` | 范围不匹配时抛出 `ValueError`；匹配时返回只含 `role`、`content` 的新列表与新字典，不改写原记录。 |

应用中的聊天状态统一保存在 `st.session_state['chat_session']`，不再使用分散的 `messages`、`active_project_id`、`active_scope`。项目选择改变时先解除旧项目会话，再读取选中项目；空项目使用该项目的空范围会话并停止问答。范围选定后调用 `ensure_chat_session()`，空选择也先更新归属，再停止。成功回答及其候选保存到同一个会话的 `messages` 中。

| 操作 | 会话与页面行为 |
| --- | --- |
| 同范围继续提问 | 编号与消息保留，模型请求包含本会话的已有消息。 |
| 选择顺序改变，或全部／指定模式对应同一集合 | 保留原会话和输入框内容。 |
| 切换项目、改变实际文档范围 | 新会话，旧聊天、来源和问题输入不再显示；切回也不恢复。 |
| 点击清空会话 | 在相同范围创建新编号，清空聊天、来源和问题输入，资源缓存复用。 |
| 没有选择文档、项目为空 | 保持对应的空范围状态，停止问答，不调用模型。 |
| 会话归属检查不通过 | 展示“会话检查失败”，在渲染聊天、检索和模型请求前停止。 |
| 模型调用或返回检查失败 | 当前会话编号与已有消息保持，不保存失败的一轮。 |

问题输入框的 key 包含会话编号，新会话会得到新的输入控件，相同会话继续保留输入值。[Streamlit 控件状态说明](https://docs.streamlit.io/develop/concepts/architecture/widget-behavior)

助手用真实 `app.py`、项目和 FAISS 索引执行九组 AppTest 检查，覆盖连续追问、清空、文档范围、项目及空项目、等价范围、独立页面会话、失败返回、故意传入错误范围和选中项目读取失败。捕获实际组装的请求，确认历史只有 `role` 与 `content`，候选属于本范围，页面仍保留原始候选供展示。一次成功检查使用八次模拟模型请求（含一次模拟失败），查询向量由已保存向量回放；真实模型推理和 DeepSeek 调用均为 0，23 个数据文件未改动。

本日完成内存中的会话管理与隔离。聊天落盘、跨重启恢复及 LangGraph 检查点仍安排在 Day 39。重启 Streamlit 后加载新代码，已有练习与测试 PDF 继续保留。

### Day 20：删除文档与同步索引（已完成）

侧栏“文档管理”提供文档选择和“删除所选文档”。学习者编写的 `remove_document_record()` 负责准备项目副本，`get_document_paths()` 负责定位 PDF 与该文档全部版本的索引目录；实际操作入口是 `delete_document(project_id, document_id)`。

```text
读取最新项目 → 验证路径归属 → 生成移除文档后的记录
    → 保存 documents 与 pending_cleanup → 清理 PDF 和索引
    → 移除已完成的待办 → 刷新页面选择、索引缓存与会话
```

`pending_cleanup` 是可选的文档记录列表，没有待办时不保留该字段。移除可检索记录与增加待办在同一次项目 JSON 保存中完成；第一步保存失败时不动文件。文件被占用、部分清理失败，或清理后更新待办失败，均保留待办供 `retry_document_cleanup()` 重试。已经不存在的文件视为已清理，项目即使为空仍能看到重试入口。待清理文档不能重新导入，避免旧待办删除新文件。

上传副本仅清理 `data/papers/{project_id}/{document_id}/` 中登记的 PDF，索引清理 `data/indexes/{project_id}/{document_id}/` 下的全部版本。早期 Demo 位于 `data/papers/` 根目录的公共 PDF 保留；其他项目目录、测试素材目录与越界路径拒绝处理。检查原始路径的每一级，并拒绝索引树中的符号链接与 Windows 目录联接，再进行递归清理。[Python 路径检查](https://docs.python.org/3.12/library/pathlib.html#pathlib.Path.is_junction)、[目录清理接口](https://docs.python.org/3.12/library/shutil.html#shutil.rmtree)

删除成功保存后，先从多选控件状态移除目标，再重新创建页面控件；清空当前会话及问题输入，刷新范围索引缓存。模型缓存继续复用，其他项目的数据文件不变；缓存清除作用于当前进程，各页面在下次运行时重新加载。应用使用函数级 `clear()` 清除索引资源缓存。[Streamlit 缓存接口](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource)

本次持久化的是清理待办，不是聊天记录。JSON 替换与文件清理仍是分步操作，没有跨文件事务、并发写入锁或后台清理任务；当前按本机单人串行操作设计。历史未知残留不自动扫描，手动替换索引需重启服务。

助手已完成 10 项文件处理检查，覆盖保存失败不改动文件、部分清理与重试、待办保存失败、公共原件保留、目录归属、重复及未知标识、导入待办、文件已不存在和真实目录联接拒绝。运行命令如下，检查只创建和删除系统临时目录中的模拟数据：

```bash
./.venv/Scripts/python.exe -m unittest discover -s tests -p test_document_deletion.py -v
```

另以真实 `app.py`、PDF 与已保存索引的隔离副本完成 8 组页面检查：删除后 52 → 51、重新导入后 52、记录保存失败、清理失败与新会话重试、最后一份文档删除、导入解析失败、导入登记失败及公共 PDF 保留。查询与重新建索引使用保存向量回放，3 次回答均为模拟返回；共 5 次向量回放，真实模型推理和 DeepSeek 调用均为 0。实际工作区 23 个数据文件的路径和内容未改变，临时副本已清理，Day 14 基线未改动。

学习者已在 test 项目完成实际删除，回传“当前项目：test · 检索文档：1 / 1 · 可检索片段：51”，并确认输入框清空。助手只读核对项目 JSON 仅剩原论文且无清理待办，测试 PDF 的上传副本和文档索引目录均已不存在，剩余论文 PDF、索引及 51 条片段记录保留。聊天清理以此前页面自动检查为依据，本次未追加模型问答。素材原件 `data/fixtures/day18_scope_note.pdf` 保留，可在后续需要时再次导入。本日没有新增依赖或自动提交 Git，约定保留的练习文件继续保留。

### Day 21：项目管理阶段验收（已完成）

学习者已在 `tests/test_project_scope.py` 实现 `assert_results_in_scope()`，逐条检查候选的 `project_id` 等于当前项目、`document_id` 属于所选列表。合法候选通过；来自其他项目、但文档标识相同的候选被拒绝；当前项目内未选中文档的候选也被拒绝。该检查只判断已返回候选是否越界，不保证非空、回答正确或引用充分。

准备的两份 `scope_note.pdf` 位于不同素材目录，均标注人工测试用途、不属于研究论文。正文除了校验码之外相同，解析后各为 1 页、367 字符、1 个片段，文件摘要与预期值保存在 `data/fixtures/day21/manifest.json`。这些预期值供验收比较，模型应从导入后的 PDF 检索资料取得答案。

| 实际项目名称 | 导入素材 | 预期校验码 |
| --- | --- | --- |
| Day21 A | `data/fixtures/day21/project_a/scope_note.pdf` | `PROJECT-A-217` |
| Day21 B | `data/fixtures/day21/project_b/scope_note.pdf` | `PROJECT-B-963` |

学习者已通过页面新建两个项目并分别导入，在 A、B 各问“这份测试文档的校验码是什么？”。回传答案分别为 `PROJECT-A-217`、`PROJECT-B-963`，原文依据与各自 PDF 完全匹配，并确认切换后聊天全部清空。用户没有回传候选片段标识，报告中的人工候选字段保留为 `null`；助手另行只读核对实际项目 JSON、文件摘要与索引，确认各有 1 条 384 维向量，片段项目、文档、来源与页码一致，将这些结果单独记录为磁盘观察。

`tests/test_project_scope.py` 保留学习者的两个断言，扩展为四个 `unittest` 用例。每个用例创建独立临时目录，从两份真实 PDF 解析、分块并建立控制向量索引，运行真实 `app.py`。查询向量故意更匹配 B 文档，检查只选 A 时仍只能返回 A；每次正向查询先确认候选非空，再调用归属断言并核对来源路径。

| 场景 | 验收结果 |
| --- | --- |
| 项目切换 | A → B → A 的来源均属于当前项目；会话编号更新，旧聊天和输入不恢复。相同文档标识也不能绕过项目检查。 |
| 文档范围 | A 项目单选 A、单选 B、同时选择两份时，候选仅来自所选范围；换范围清空历史，仅改变选择顺序时保留。 |
| 空选择 | 隐藏问题入口、清空会话，不增加查询或模拟回答次数；后端拒绝空范围，重新选择后恢复。 |
| 删除隔离 | 同一份 B 文件分别放入两个项目，文档标识相同；删除 A 项目内的副本后，其文件与索引消失，B 项目的记录、PDF、索引哈希不变，两个项目的剩余文档均可查询。 |

运行四组检查的命令：

```bash
./.venv/Scripts/python.exe -m unittest discover -s tests -p test_project_scope.py -v
```

四组检查通过，共 11 次控制向量查询和 11 次模拟回答，真实模型推理与助手发起的 DeepSeek 调用均为 0。控制向量不代表真实语义检索效果，模拟回答不代表模型质量；本日模型实际表现以两条用户回传为依据。测试期间已有 31 个数据文件的路径和内容不变，临时目录已清理；新增验收报告位于 [day21_project_management.json](data/evaluations/day21_project_management.json)，未修改 Day 14 基线。

Day 21 验收按本机单进程串行操作完成，不覆盖并发写入。两个真实 Day21 项目、PDF、索引与素材保留；已有练习文件按约定继续保留，没有新增依赖或自动提交 Git。今天无需追加模型问题，下一次进入 BM25 学习。

### BM25 基线与 Dense 对比（Day 22）

`bm25_retriever.py` 提供 `tokenize_bm25(text)` 与 `retrieve_bm25(question, chunks, top_k=3)`。后者接收当前范围内的完整片段列表，返回与 Dense 相同的 `rank`、`score`、`chunk` 字段。当前基线在每次调用时统计传入片段的词频和长度，使用 `BM25Okapi` 默认参数；尚未增加 BM25 资源缓存。Day 23 已通过统一入口接入页面，Day 24 开始将两路候选按 RRF 融合后选入回答参考资料。

分词规则为先转小写，再按 `[a-z0-9]+` 提取连续英文字母与数字，保留 `P100`、`2014` 等词项；连字符被拆开。该规则忽略中文，不执行翻译、词形还原、同义词扩展或停用词过滤，`GPU` 与 `GPUs` 仍是不同词项。问题和片段均使用同一函数，也不为 BM25 添加向量模型的 `query:` / `passage:` 前缀。

候选至少要与查询重合一个词项，因此允许部分匹配，最多返回 `top_k` 条。没有可用查询词、没有片段或没有任何匹配时返回空列表；空白问题和非法 `top_k` 报错。不要把 `score > 0` 当作通用的匹配判断：本库在特定小语料上会为确实匹配的词项给出零分或负分，测试中保留了这两个案例。BM25 分数不是概率，也不能与 Dense 原始分数直接比较或相加。

学习者完成 `NVIDIA P100` 查询，51 个片段中前三名为 `p7:c3`、`p7:c4`、`p2:c3`，分数约为 4.9147、4.4716、1.9785；前两条包含完整硬件说明，第三条只匹配到 `P100`。候选字典保留项目、文档、片段标识、原文、来源路径及 PDF 页码。

助手随后在 Demo project 的同一份 51 片段索引上运行三条固定查询。每条查询原样交给两路检索，Dense 使用本地已缓存的 `intfloat/multilingual-e5-small`、CPU 和已保存编码配置，BM25 使用相同片段文本。下表用片段标识末尾的 `p页码:c片段序号` 简写，完整标识、分数、原文和来源保存在 [对比报告](data/evaluations/day22_retrieval_comparison.json)。

| 查询 | Dense 前三名 | BM25 前三名 |
| --- | --- | --- |
| `NVIDIA P100` | `p7:c4`、`p2:c3`、`p10:c3` | `p7:c3`、`p7:c4`、`p2:c3` |
| 编码器由多少个相同的层堆叠而成？ | `p2:c7`、`p10:c3`、`p5:c2` | 空列表 |
| `How much electrical energy in kWh was consumed during training?` | `p8:c3`、`p7:c4`、`p7:c2` | `p7:c5`、`p8:c2`、`p8:c4` |

本次观察：硬件型号查询两路均找到硬件说明，但次序和候选不同；纯中文问题在当前 BM25 分词下没有词项，Dense 找到含编码器 6 层说明的片段。英文耗电量问题两路都返回候选，助手核对六条原文后未见实际 kWh 数值，词项匹配或语义相关不能代替答案依据。三例只用于学习观察，不代表总体性能优劣；没有生成模型回答，也不计入 Day 14 的 20 题基线。

已完成四项自动检查，覆盖候选排序和来源映射、返回数量与无匹配排除、空输入及非法数量，以及零分或负分仍有匹配的情况。运行命令：

```bash
./.venv/Scripts/python.exe -m unittest discover -s tests -p test_bm25_retriever.py -v
```

对比脚本与报告均可纳入 Git。脚本须从项目根目录运行，读取已构建的 Demo 索引及本地已缓存的向量模型；首次安装后若尚无这些资源，先按前述索引构建步骤准备。已有结果文件不会覆盖，重新观察时另指定输出路径：

```bash
./.venv/Scripts/python.exe compare_retrievers.py --output data/evaluations/day22_retrieval_comparison_rerun.json
```

本次实际执行 3 次 Dense 查询和 3 次 BM25 查询，DeepSeek 调用为 0；原有 32 个数据文件路径和内容保持一致。报告包含输入索引摘要、依赖版本、参数和执行范围；原索引未固定模型 revision，查询使用本地已缓存的同名模型。后续跨项目双路接入与归属验证按 Day 23 继续。

### 双路入口与候选去重（Day 23）

`dual_retriever.py` 的 `retrieve_dual(question, model, index, metadata, top_k=3)` 复用调用方从 `load_scope_index()` 取得的配套索引与片段，分别返回 `dense` 和 `bm25` 两个候选列表。每路最多 `top_k` 条，原始名次和分数保持各自含义。入口先检查索引数量与片段数量、片段标识有效且唯一，再核对每条返回记录与当前范围内的原文、来源及其他字段一致；范围外或被改写的候选会在进入页面和模型请求前被拒绝。

`merge_candidates(dual_results)` 按 `chunk_id` 合并，每个候选包含完整 `chunk` 与 `retrievals`。后者以检索方式为键，保留对应的 `rank`、`score`。同一标识对应不同记录时拒绝合并，单路重复标识也报错；文字相同但片段标识不同的记录分别保留。列表按 Dense 后 BM25 的首次出现顺序返回，不代表融合后的相关性排名。

学习者回放 Day 22 已保存的 `NVIDIA P100` 候选，两路各 3 条，去重后为 4 个片段：`p7:c4` 保留 Dense 第 1、BM25 第 2；`p2:c3` 保留 Dense 第 2、BM25 第 3；`p10:c3` 仅来自 Dense；`p7:c3` 仅来自 BM25。这是旧结果的合并练习，没有产生新的模型请求。

页面已调用统一入口并保存各阶段候选。`retrieval_results` 保存实际提供给回答的片段，Day 23 为 Dense，Day 24 的新消息为 RRF 选中的结果；`retrieval_channels` 保存两路原始列表，`retrieval_candidates` 保存去重结果。新消息额外保存 `reference_method` 与融合参数，历史展示据此选择对应规则。组装模型历史时仍只取 `role` 和 `content`；切项目、换范围、清空或删除导致新会话时，候选随消息一起清除。页面兼容更新前仅有 `retrieval_results` 的旧消息。

BM25 无匹配时，页面显示当前分词只处理英文字母与数字的提示，Dense 可继续提供候选。Day 23 时模型只接收 Dense 片段，BM25 独有候选用于页面观察；Day 24 接入 RRF 后，任一路候选只要进入最终融合前三名，就可以成为本轮参考资料。

Day 23 时，`tests/test_dual_retriever.py` 的 12 项检查全部通过：五项检查覆盖真实保存结果的合并、空路、不同标识的相同文字、重复或冲突记录、输入不匹配与返回记录越界；七组页面检查覆盖项目切换、文档选择、空选择、删除隔离、BM25 无匹配及清空、BM25 独有候选展示和越界失败后的历史保留。Day 24 的页面检查已更新为当前 RRF 参考资料规则，并由 RRF 测试文件继承复用；原 Day 23 报告仍保留当时结果和代码摘要。

```bash
./.venv/Scripts/python.exe -m unittest discover -s tests -p test_dual_retriever.py -v
```

Day 23 页面检查复用 Day 21 的人工 PDF，在独立临时目录中运行真实页面、FAISS 和 BM25，查询向量受控，回答为模拟数据。当时的 BM25 独有候选场景把 Dense 数量控制为 1，检查该候选可见但没有进入参考资料。实际向量模型推理与 DeepSeek 调用均为 0，实际工作区原有 33 个数据文件不变；这些检查不代表语义召回或回答质量。检查摘要与旧候选回放分别保存在 [Day 23 报告](data/evaluations/day23_dual_retrieval.json)，未记录逐次页面请求轨迹，也未改动 Day 14、Day 22 报告。

**实际页面验收已通过**：学习者回传 `Dense：3 条 · BM25：3 条 · 去重后：4 个片段`。第一个候选为 Demo 原论文的 `p7:c4`，PDF 第 7 页，Dense 排名 1、显示分数 0.8396，BM25 排名 2、显示分数 4.4716；页面同时说明参考资料使用 Dense，候选按首次出现顺序展示。回传记录与自动检查分别保存，只记录实际提供的数量和首条候选，不扩写为其余候选、答案质量或原始 API 响应。Day 23 已完成，下一次进入 RRF 排名融合。

### RRF 排名融合与参考资料（Day 24）

`rank_fusion.py` 中的 `calculate_rrf_score(retrievals, rank_constant=60)` 使用学习者编写的 `total += 1.0 / (rank_constant + rank)`，累加各路实际召回排名的贡献；原始 Dense 与 BM25 分数不参与计算。`fuse_rrf(candidates, top_k=3, rank_constant=60)` 接收已去重候选，生成独立结果记录，按融合分数降序、同分时按 `chunk_id` 升序排列，最后截取前 `top_k` 条并从 1 重新编号。

返回的 `score` 是 RRF 分数，`chunk` 保留完整来源，`retrievals` 保留各路原始排名和分数，输入记录不会被改写。空候选返回空列表，单路有结果时按该路排名计算贡献；正整数参数、有效排名、唯一片段标识和至少一路排名均有检查。融合分数用于排序，不是回答正确的概率。

学习者完成两组手算对照：常数为 1 时 A=0.5、B≈0.833333、C≈0.333333；常数为 2 时 A≈0.333333、B≈0.583333、C=0.25，两组顺序都是 B、A、C。本地 `practice_rrf.py` 保留这些样例，并可读取 Day 22 已保存的真实候选进行回放。

Day 24 页面每路最多召回 3 条，RRF 常数为 60，最终最多选 3 条。对已保存的 `NVIDIA P100` 候选进行回放，得到：

| 最终排名 | 片段标识末尾 | Dense 排名 | BM25 排名 | RRF 分数（约） |
| --- | --- | --- | --- | --- |
| 1 | `p7:c4` | 1 | 2 | 0.032522 |
| 2 | `p2:c3` | 2 | 3 | 0.032002 |
| 3 | `p7:c3` | 未出现 | 1 | 0.016393 |

原 Dense 第 3 名 `p10:c3` 未进入最终前三名，BM25 独有的硬件说明片段 `p7:c3` 被选入。本次回放只重算旧候选的融合分数，没有执行新的语义检索，也不能据此推断整体问答质量提升。常数 60 和召回数量尚未调优；Day 25 在此基础上接入 Rerank，见下一节。

Day 24 问答保存 `reference_method='rrf'`、`fusion_config` 和最终候选；模型的本轮 `<reference>` 仅包含融合选中的片段，片段编号跟随最终排名。页面显示“本轮回答使用 RRF 融合后的 N 个片段作为参考资料”，并展示其融合排名、RRF 分数和各路原始信息。该规则继续用于已有 RRF 历史，新问答从 Day 25 起使用下述精排规则；候选与配置不会进入模型的聊天历史字段。

`tests/test_rank_fusion.py` 包含八项打分与融合检查和九组页面检查。首次运行 17 项，其中 16 项通过；一个页面用例因只用四条候选重建 BM25，改变了统计语料和排名，与预期的完整论文结果不一致。修正为解析完整论文的 51 个片段后，只重跑该用例并通过，生产代码未因这次样例问题修改，其余通过项没有重复运行。

检查覆盖手算、原始分数不影响融合、单路与空候选、同分顺序、数量限制、参数与重复标识、输入不变性，以及项目切换、文档选择、删除、越界拦截、最终参考资料截取和旧历史兼容。页面测试使用真实 FAISS 和 BM25、控制向量与模拟回答，实际模型推理及 DeepSeek 调用为 0，工作区原有 34 个数据文件不变。完整论文用例验证模型只收到最终三个片段，未选中的参考文献片段不在本轮请求中。

```bash
./.venv/Scripts/python.exe -m unittest discover -s tests -p test_rank_fusion.py -v
```

[Day 24 报告](data/evaluations/day24_rrf.json) 分别保存用户手算回传、三个旧查询的融合回放、首次检查的失败原因与针对性复验、复验页面观察以及学习者实际页面确认。原 Day 14、Day 22、Day 23 报告保持不变。

**实际页面验收已完成**：按 Demo project 查询 `NVIDIA P100` 的验收步骤，学习者回传“本轮回答使用 RRF 融合后的 3 个片段作为参考资料”、排名常数 60，以及按顺序显示的分数 0.032522、0.032002、0.016393，符合上表的六位小数结果。本次回传未包含片段标识、两路数量、回答正文或原始 API 响应，不将这些内容记为新增观察或回答质量结论；无需重复验收。

### Rerank 精排与页面接入（Day 25）

`reranker.py` 使用 `BAAI/bge-reranker-base`，固定 revision 为 `2cfc18c9415c912f9d8155881c133215df768a70`。模型在 CPU 上处理问题与候选原文的输入对，批量大小 3，最大长度 512 token，显式使用 Identity 保留原始相关性分数；负数属于正常输出，不能把分数当成回答正确率。超长输入可能被截断，该长度并非字符数。模型与调用方式参见 [官方模型卡](https://huggingface.co/BAAI/bge-reranker-base) 和 [CrossEncoder 文档](https://sbert.net/docs/cross_encoder/usage/usage.html)。

当前流程为 **两路各取最多 6 条 → 去重 → RRF 排序并保留最多 6 条 → 精排 → 最终最多 3 条作为参考资料**。Day 25 初接入时每路最多召回 3 条，Day 28 对照与复验后更新为 6；下方 Day 25 的回传仍是当时配置的历史记录。RRF 常数为 60；融合候选数量和最终参考数量分开控制。精排函数只能对传入候选重新排序和截取；完整 `chunk`、各路 `retrievals` 保留，结果深复制后保存原 `rrf_rank`、`rrf_score`，新 `rank` 从 1 编号，`score` 与 `rerank_score` 均为精排分数。同分沿用输入的 RRF 顺序。

学习者用三条短样例完成输入对、模型打分、降序排序和前两条截取。随后读取 Day 22 保存的四条真实候选，仅重新执行精排，回传以下结果：

| 精排排名 | 原 RRF 排名 | 片段标识末尾 | RRF 分数 | 精排分数 |
| --- | --- | --- | --- | --- |
| 1 | 1 | `p7:c4` | 0.032522 | 0.1071 |
| 2 | 3 | `p7:c3` | 0.016393 | -0.6970 |
| 3 | 2 | `p2:c3` | 0.032002 | -3.4262 |

这是 `NVIDIA P100` 查询的旧候选精排回放，不是新的召回对比或整个评估集的质量结论。短样例的单次打分耗时为 0.252 秒；上述四候选回放脚本的总耗时未回传，报告保留为空。后续实际页面回传的 1.212 秒另行记录，不回填为先前脚本的耗时，两者也不作为稳定性能基准。

Day 25 新消息保存 `reference_method='rerank'`、`rrf_results`、`rerank_config` 和 `rerank_elapsed_seconds`。DeepSeek 本轮只接收精排最终选出的参考资料；历史仍只传角色与正文。页面展示最终片段的新旧排名及分数，精排耗时包括输入准备、模型打分与结果整理，不含模型加载和回答生成。完整链路的阶段记录在下方 Day 26 中补充。

`tests/test_reranker.py` 的七项逻辑与配置检查、十四组页面检查共 21 项通过，用时 48.456 秒。覆盖项目与文档范围、空选择、清空与缓存复用、删除隔离、越界拦截、精排改序、最终参考截取、缺少缓存、异常分数、模型失败以及旧 Dense / RRF 历史展示。页面使用真实 FAISS 和 BM25、控制向量、控制精排分数和模拟回答；助手未调用真实模型或 DeepSeek，原有 35 个数据文件不变，临时目录已清理。共享页面测试已适配当前精排规则，旧报告继续记录各阶段当时的测试结果。

```bash
./.venv/Scripts/python.exe -X utf8 -m unittest discover -s tests -p test_reranker.py -v
```

[Day 25 报告](data/evaluations/day25_rerank.json) 分别记录学习者的短样例与真实候选回传、自动检查及其边界，以及已完成的实际页面确认。**页面验收已通过**：学习者回传参与精排 4 条、最终参考 3 条、精排耗时 1.212 秒（不含模型加载）；最终片段依次为 `p7:c4`、`p7:c3`、`p2:c3`，原 RRF 名次为 1 / 3 / 2，PDF 页码为 7 / 7 / 2，两种分数与上表一致。此次未回传回答正文或原始 API 响应，不新增回答正确性结论；无需追加验收。

### 完整检索阶段记录（Day 26，已完成）

`retrieval_trace.py` 的 `run_stage()` 执行一次处理函数，记录输入数量、输出数量、耗时与深复制的结果快照。`retrieve_dual()` 可接收 `stage_records`，保持原有两路返回结构；通过范围校验后才交付召回记录。页面对全部 RRF 候选精排一次，保留完整排序后再截取前三条，未入选的片段仍在记录中。

新消息的 `retrieval_trace` 包含本轮问题、范围、六个阶段、按完整 `chunk_id` 汇总的排名路径及单独的资源获取和回答耗时。“查看本轮检索流程”面板展示两张表，并可展开原始 JSON 查看完整原文与分数。记录跟随成功的聊天消息，普通页面重运行不重新检索或打分；旧消息没有追踪字段时按原方式显示，不补造历史耗时。当前记录保存在 Streamlit 会话中，长期持久化留到 Day 39。

学习者按 Demo 原论文查询 `NVIDIA P100` 的验收步骤，回传以下阶段记录，数量与预期一致：

| 阶段 | 输入数量 | 输出数量 | 本次耗时（秒） |
| --- | ---: | ---: | ---: |
| Dense | 51 | 3 | 0.158026 |
| BM25 | 51 | 3 | 0.006838 |
| 去重 | 6 | 4 | 0.000015 |
| RRF | 4 | 4 | 0.000117 |
| Rerank | 4 | 4 | 1.226635 |
| 最终截取 | 4 | 3 | 0.000002 |

四条候选路径保留了同一来源、PDF 页码与前后排名。`p10:c3` 的 Dense 排名为 3，BM25 无返回，RRF 和精排排名均为 4，最终未入选；其他三条按精排名次依次为 `p7:c4`、`p7:c3`、`p2:c3`。本次耗时只代表这次运行，不与早先练习的去重计时混用，也不作为稳定性能基准。页面验收已通过，无需补做本轮未回传的回答内容或分数。

Dense 的耗时包含问题编码与索引查询；BM25 包含当前语料分词、统计构建与排序。各阶段计时不含额外的快照复制、模型资源加载、页面展示和回答生成；精排资源获取单列，包含缓存查找及缓存未命中时的加载，回答调用与校验也单列。页面不把这些值误称为端到端总耗时。

路径表按去重后的原顺序展示，缺少排名显示 `—`，原记录对应 `None`。入选状态只说明片段是否发送给回答模型，不能自动证明它支持回答；原文引用的事实支持仍需核对。失败时保留已有聊天，不追加半轮记录。

```bash
./.venv/Scripts/python.exe -X utf8 -m unittest discover -s tests -p test_retrieval_trace.py -v
```

学习者的去重计时与四条排名路径回传、控制模型的检查结果和实际页面确认状态分别记录在 [Day 26 报告](data/evaluations/day26_retrieval_trace.json)。旧报告保持不变。

自动检查分两次：首次 24 项中 11 项通过，13 项因 AppTest 将 Arrow 表格转回 pandas 时的类型错误中断；测试改为直接读取页面发送的 Arrow 表格后，16 组页面场景全部通过，用时 73.876 秒。结合首次通过的 8 项逻辑检查，24 个不同场景均已验证；不是一次运行全部通过。检查使用临时项目、真实 FAISS / BM25、控制向量与精排分数及模拟回答，真实模型与 DeepSeek 调用均为 0，原有 36 个数据文件未变。

### 检索评估标注与对照结果（Day 27，已完成）

[day27_relevance_labels.json](data/evaluations/day27_relevance_labels.json) 保存助手对 Demo 原论文固定 20 题与 51 个片段的判断。范围是 `s800:o100`、索引 `v1`，文件记录题集、PDF、索引元数据的 SHA-256，以及各片段的文本摘要；变更分块或来源后应建立新版本，不沿用旧标签。

本轮采用 `direct_evidence_v1`：一个片段自身须能明确支持至少一个预先定义的完整证据单元。单元及支持片段保存在 `facts` 中，每题的 `required_fact_ids` 列明需要哪些单元，`relevant_chunk_ids` 是这些单元的支持片段并集，`non_relevant_chunk_ids` 显式记录其余片段。共有 13 个证据单元、23 条带字符起止位置的原文引用。

标注规则包括：

- 仅有主题词、相同数值或参考文献条目不足以作为直接依据；不跨片段补全缺失的主语、表头、符号解释或量纲。重要边界在各题 `review_notes` 中解释。
- 单元可要求成组事实，例如 q8 的 base 训练安排须同时说明步数与时长；复合题则显式分成多个单元。命中一个单元不表示整道复合题已经具备全部证据。
- 按唯一 `chunk_id` 计数，重叠分块即使含有同一句原文，也分别计作片段。因此片段 Recall 与首个命中 MRR 不代表独立事实覆盖率。
- 15 道有答案题用于 Recall / MRR；5 道资料不足题的相关集合为空，单独记录。存在依据但系统未命中的问题仍以 0 参与平均。
- 这些是助手按固定规则完成的学习评估标注，尚无独立人工复核，也不是盲测基准。它们不是穷尽所有语义关联的标签，更不是模型回答正确性判定。

q3 的教学材料已由助手按学习者请求判断：`p5:c5` 只解释位置编码的用途，不能提供两个函数名称；`p6:c1`、`p6:c2` 均包含 “In this work, we use sine and cosine functions of different frequencies:”，能直接支持回答。完整范围内按本规则认定的 q3 相关片段数为 2。运行 `practice_relevance_labels.py` 可以同时查看三个片段及其判断，不会修改标签或调用模型。

本轮检查了 20 × 51 = 1020 个题目与片段标签的覆盖、正负集合互斥、全部原文引用位置和输入摘要；这些检查确认记录一致性，语义判断仍来自助手审阅。

`retrieval_metrics.py` 保留学习者完成的三个指标函数，练习脚本改为导入它们，原教学输出不变。学习者已在 `evaluate_retrieval_results()` 使用 `result.get("chunk", {}).get("chunk_id")` 读取完整标识，通过连接检查。项目根目录的输入检查命令为：

```bash
./.venv/Scripts/python.exe -X utf8 evaluate_retrieval.py --check
```

学习者已确认显示 20 题、15 道有答案题、5 道资料不足题和 51 个片段。该命令先验证指标连接，再核对题集、标注、项目与索引的一致性；不加载模型、不执行检索、不生成报告。缺少有效片段标识时会报数据校验错误。

2026-10-25 已执行一次完整对照。对同一原始问题共享 Dense / BM25 各前三条召回，RRF 最多保留六条交给精排，四种方法均评价各自前三条。只用 15 道有答案题计算 Mean Recall@3 与 MRR@3，保留未命中的 0；另 5 题记录候选但不参与均值。

| 方法 | 参与平均的题数 | Mean Recall@3 | MRR@3 |
| --- | ---: | ---: | ---: |
| Dense | 15 | 0.5133 | 0.5444 |
| BM25 | 15 | 0.1800 | 0.2444 |
| RRF | 15 | 0.5156 | 0.5111 |
| RRF 后加入 Rerank | 15 | 0.5044 | 0.7333 |

[对照报告](data/evaluations/day27_retrieval_comparison.json) 保存每题候选、命中标识、原始分数、阶段快照与输入摘要。使用本地缓存的 E5 和 BGE 模型，分别执行 20 次查询与精排，DeepSeek 调用为 0；模型加载分别耗时约 6.99 秒与 11.73 秒，检索对照部分约 36.90 秒（不含模型加载及 Python 导入开销）。这些是单次运行记录。

独立复核了全部 80 组方法结果、阶段数量、相关标识交集、首次命中名次、15 题均值与输入摘要。原有 38 个数据文件未变，只新增本次报告。当前 BM25 只提取英文字母与数字，20 题中有 14 题返回空结果；本次保留原题文本，不做翻译或改写，不能将这组结果概括为 BM25 在其他语料与查询上的表现。片段指标也不等于回答正确率。

脚本通过 `./.venv/Scripts/python.exe -X utf8 evaluate_retrieval.py` 执行完整实验。默认报告已存在，重复执行会停止；后续有明确对照目的时可用 `--output` 指定新报告路径。本日实验和结果解读已完成，后续参数实验以该报告为基线，使用新报告保存结果。

### 召回数量对照（Day 28，已完成）

学习者已将 `evaluate_case()` 的每路召回参数传给 `retrieve_dual()`，默认仍为 3。`practice_retrieval_tuning.py` 确认分别传入 3 和 6 时底层收到相同值；RRF 上限保持 6，最终评价 K 保持 3。

[day28_tuning_plan.json](data/evaluations/day28_tuning_plan.json) 在运行新配置前保存实验约定与输入摘要。开发题使用已知失败的 q5、q12、q13；另行固定五道新问题，四道有答案、一道资料不足，助手按同一直接依据标准检查 51 个片段，保存正负集合及四条精确引用。这些是同论文的新复验问题，没有使用它们运行检索或选择参数，不是跨文档或独立人工盲测基准。开发结果用于暂选配置，后续复验前记录选择，避免根据复验结果反复尝试新参数。

运行开发对照：

```bash
./.venv/Scripts/python.exe -X utf8 compare_recall_limits.py
```

脚本使用本地缓存的向量与精排模型，各加载一次，在三个开发问题上分别运行 A（每路最多 3 条）和 B（每路最多 6 条），共六次流程；按题交替 A/B 顺序。它打印召回池命中数、交给精排的候选命中数、最终 Recall@3 / RR@3 和耗时，再汇总两组结果。阶段完整候选集合的命中率与最终前三条的指标分别记录。

学习者已运行并保存 [开发题对照报告](data/evaluations/day28_development_comparison.json)，已有报告时脚本拒绝覆盖；原 Day 27 报告保留。脚本未运行五道新复验题，未生成 DeepSeek 回答。单次计时包含流程和指标计算、不包含模型加载与 Python 导入，未专门预热，不作为稳定性能基准。

准备阶段通过了计划、范围、引用和摘要检查，以及使用合成向量、固定精排分数的受控流程检查；当时原有 39 个数据文件未变，只新增计划文件。随后学习者完成真实运行并回传六行结果，助手核对报告中的输入及代码摘要、全部阶段命中、24 组方法指标和汇总，均一致。

| 组别 | 每路召回上限 | 最终平均 Recall@3 | 最终 MRR@3 | 平均流程耗时（秒） |
| --- | ---: | ---: | ---: | ---: |
| A | 3 | 0.0000 | 0.0000 | 1.076 |
| B | 6 | 0.6667 | 0.6667 | 1.943 |

q12 的依据 `p5:c4` 在 Dense / RRF 中排第 6，q13 的依据 `p8:c3` 排第 5；B 组让它们进入精排后都升至第 1，A 组则在召回时就截掉了它们。q5 在 B 组完整候选集合中仍没有标注依据。平均耗时本次增加约 0.87 秒，约为 A 的 1.8 倍。结果只针对三道特意选择的已知失败题，尚不能推断全部问题的表现。

学习者明确愿意为检索质量接受额外耗时，在复验前暂选 B，记录在 [day28_selection.json](data/evaluations/day28_selection.json)，其中包含选择时间、用户原话和开发报告摘要。此处观测到的是片段召回和排名改善，不是回答准确率提高。选择记录保留当时尚未切换页面的状态，复验后的实际采用结果另存于结论报告。

运行固定新题复验：

```bash
./.venv/Scripts/python.exe -X utf8 validate_recall_limits.py
```

脚本先核对计划、开发报告、选择记录和实际检索模块版本，再使用原有本地模型缓存运行五道新题的 A/B 配置，共十次流程。四道有答案题参与 Recall / MRR 平均；一道资料不足题显示“不适用”、在 JSON 中保留 `null`，不计入质量指标均值。五道题都参与耗时平均。返回候选不表示已经生成回答，也不据此评价拒答是否正确。

共用运行函数仅增加了 `None` 指标的显示处理，检索与指标算法不变，原开发报告保留当时的代码摘要。准备阶段通过了选择一致性和受控数据检查，包括未命中的 0 仍参与平均、资料不足题不参与平均及打印兼容；当时原有 41 个数据文件保持不变，只新增选择记录。

学习者已完成 [五道新题复验](data/evaluations/day28_validation_comparison.json)。复核十条流程、40 组方法指标、阶段命中、输入与代码摘要，以及“先固定题目、再开发、再选择、最后复验”的时间顺序，均一致。最终精排结果如下：

| 组别 | 每路召回上限 | 平均 Recall@3（4 题） | MRR@3（4 题） | 平均流程耗时（5 题，秒） |
| --- | ---: | ---: | ---: | ---: |
| A | 3 | 0.5000 | 0.5000 | 1.242 |
| B | 6 | 0.7500 | 0.7500 | 1.840 |

v1 的依据 `p7:c3` 在 B 组 Dense / RRF 中排第 6，精排后升到第 1，A 组没有召回它；v2、v3 两组都命中且首条相关。v4 的代码仓库依据仍未进入任一组候选。v5 是资料不足题，两组指标均为 `null`，未参加质量平均，也没有生成回答来测量拒答质量。本次平均耗时增加约 0.60 秒、约为 A 的 1.48 倍，只是单次观测。

结合开发结果、新题复验和用户的质量优先偏好，工作台 `app.py` 已采用 **每路召回 6、RRF 上限 6、最终返回 3**。实验脚本保留原默认 3 作为 A 组基线，并通过参数显式传入 B=6，便于重现。重启 Streamlit 后，新提交使用新配置，已有消息按当时保存的参数展示。

[day28_conclusion.json](data/evaluations/day28_conclusion.json) 保存采用依据、原报告摘要、未解决问题和页面检查记录。16 个不同页面场景已验证：首次 16 项中 15 项通过，1 项仍预期旧 BM25 数量；更新控制数据与数量/排名断言后，受影响的 2 项目标复验通过。检查确认两路各 6 条合并为 9 条、RRF 截到 6 条、最终取 3 条，并覆盖未进入精排的候选追踪、原第六名提升、旧历史和范围隔离。使用临时项目、真实 FAISS/BM25 与控制模型，真实模型和 DeepSeek 调用均为 0，原有 43 个数据文件保持不变，只新增结论报告。此小样本结果支持当前选择，不代表全局最优；开发题 q5 和复验题 v4 留作后续问题，不在本次复验集上继续尝试更多参数。

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
- [FAISS 已存向量读取与索引操作](https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes)
- [Rank-BM25 用法与预处理约定](https://github.com/dorianbrown/rank_bm25)
- [Rank-BM25 打分实现](https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py)
- [Streamlit 资源缓存](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource)
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

### Day 13｜2026-10-20｜参考资料组装与页面资源加载（进行中）

- **当前阶段状态**：尚未完成 Day 13，已完成的学习日仍为 12 / 42。
- **开始时的仓库状态**：本地 `main` 与跟踪引用 `origin/main` 均指向 `1fabbc0`（实现论文向量化、FAISS 索引持久化与 Dense 检索），工作区干净；Day 10～12 的正式文件已进入该提交。
- **学习者已完成**：新增 `rag_context.py` 的 `build_reference_text(results)`，逐条收集带标识、来源、PDF 页码和原文的字符串，再用空行连接。练习输出为 `str`，首条是耗电量问题检索到的 `p7:c4` 片段。
- **页面加载**：在 `app.py` 增加 `@st.cache_resource` 装饰的 `load_retrieval_resources(output_dir)`，返回模型、索引和元数据。学习者确认页面能显示可检索片段数量 51；交互后的缓存复用仍待确认。
- **遇到的问题**：终端出现 ZoeDepth 图像处理模块导入 `torchvision` 失败。学习者补充 traceback 起点为 `streamlit/watcher/local_sources_watcher.py` 的 `get_module_paths()`，同时页面仍能显示 51 条片段，定位为后台模块扫描报错。
- **助手复核**：项目 `.venv` 中未安装 `torchvision`，但文本模型可独立导入、加载，51 条索引可读取，并成功生成 `(1, 384)` 查询向量。堆栈中的 `C:\Python312\Lib` 是虚拟环境所依托的 Python 标准库路径，不能仅据此判断使用了错误的环境。
- **处理方式**：助手新增 `.streamlit/config.toml`，关闭 `server.fileWatcherType`，并同步启动说明。未安装视觉依赖，也未改变模型、索引或依赖版本；此为绕开扫描路径的配置处理，没有修改上游库代码。用户仍需重启现有 Streamlit 服务，确认终端不再出现该扫描异常。
- **当前尚未接通**：页面仍从手工输入框获得参考资料，尚未在提交问题时调用检索与资料组装函数，也未完成回答、来源展示和资料不足场景的端到端验收。
- **下一步**：重启后确认页面加载与缓存表现，再把检索结果接入本次请求的 `<reference>`，完成 Day 13 剩余学习任务。原有练习文件继续保留。

### Day 13｜2026-10-20｜基础 RAG 接入、每轮来源与输出约束（完成）

- **当天目标**：把已经学过的检索接入实际问答页面，展示可核对的来源，并验证资料不足的行为。
- **本次状态**：已完成，总进度为 13 / 42；上面的阶段记录保留为当时进展。
- **实际用时**：未完整记录。
- **当天剩余必做任务**：0。

#### 完成的任务

- [x] 在提交问题时执行 Dense 检索，将三条候选与元数据组装为本次参考资料。
- [x] 复用现有 DeepSeek 调用、结构化解析和异常处理，跑通真实页面问答。
- [x] 分开模型请求和用户聊天展示，避免整段参考资料出现在用户气泡中。
- [x] 将本轮检索结果保存到助手消息，统一渲染历史与新消息，展示可展开的来源区域。
- [x] 发送历史前只提取 `role`、`content`，让页面附加的检索记录留在会话展示层。
- [x] 发现真实模型未遵守“资料不足时依据为空”的约定，新增程序校验并验证拒绝路径。
- [x] 验证两轮来源保留、清空会话、缓存复用和失败不写入历史；更新运行说明与学习记录。

#### 代码与文档改动

| 文件 | 改动及原因 |
| --- | --- |
| `rag_context.py` | 学习者新增 `build_reference_text()`，将候选和来源整理为模型参考资料字符串。 |
| `app.py` | 学习者接入资源缓存、自动检索、资料组装、历史投影和来源展示；助手收尾时清理两处空白行中的尾随空格。 |
| `answer_parser.py` | 学习者在类型检查之后增加组合约束，拒绝非空不足原因与非空依据同时出现。 |
| `.streamlit/config.toml` | 助手新增禁用文件监视的配置，绕开可选视觉模块的后台扫描问题；修改代码后需要重启服务。 |
| `practice_retrieval.py` | 学习者追加参考资料类型与前 500 字符的打印，继续按既有规则保留在本地。 |
| `readme.md` | 助手同步当前能力、Day 13 完成状态、页面数据流、校验边界和下一次学习安排。 |

今天没有新增依赖或重建索引；没有执行 Git 提交、推送或练习文件清理。原有三个练习文件继续保留，PDF 空文本检查仍未完全迁移到正式导入流程。

#### 真实页面结果与发现的问题

**缓存复用。** 学习者确认点击清空会话后，终端没有再次输出“正在初始化检索资源”。这是当前交互复用了缓存的观察；不代表已经实现索引内容变化时的自动失效。

**有依据的问题。** 编码器问题返回“编码器由 N = 6 个相同的层堆叠而成。”，引用 `Encoder: The encoder is composed of a stack of N = 6 identical layers.`，与第 2 页候选原文一致。初次页面还展示了整段 `<reference>`，随后将用户展示消息与模型请求分开，并为每轮回答加入来源区域。

**资料不足的问题。** 耗电量问题明确说明资料只提供训练步数、时长和 FLOPs，不能回答实际 kWh，没有编造电量数值。但同一次结果仍列出训练时长和成本表引用，违反了本项目“资料不足时 `evidence` 为空”的约定。该结果作为真实失败案例保留，没有记作完全符合输出要求。

**字段组合校验。** 原解析器只检查 JSON、字段和类型，因此上述组合能够通过。学习者尝试用 `evidence.empty()` 判断，经讲解后改为检查整个 `result["evidence"]` 列表：空列表在条件中为假，非空列表为真；循环变量 `evidence` 代表单条依据。最终在所有类型检查之后加入：

```python
if result["insufficient_reason"].strip() and result["evidence"]:
    raise ValueError("资料不足时，evidence 必须为空列表")
```

提示词负责提出要求，程序负责检查已明确实现的约束。此处检查字段组合，不判断论文是否足以回答，也不自动检查引用是否真实或是否支持结论。

#### 助手验证结果

- 固定 JSON 数据验证：有依据且不足原因为空、依据为空且有不足原因，两种合法结果均通过；两字段同时非空的结果抛出预期 `ValueError`。原有缺字段、错误类型、依据元素类型错误和无效 JSON 仍被拒绝。
- 资料组装检查：三条实际保存的候选原文与来源均完整进入字符串，顺序一致；空列表返回空字符串。
- 使用 Streamlit AppTest 加载真实 `app.py`，模拟向量资源、检索和模型返回，调用真实 `build_reference_text()` 与 `parse_answer()`。两轮合法回答生成四条消息和两个来源区域，重运行后仍分别保留本轮原文；API 消息仅包含 `role` 和 `content`。
- 模拟这次违规模型输出，页面显示校验错误，先前四条消息和两轮来源保持不变。模拟未完整结束的回答、空检索和检索异常，也不追加历史；空问题与空检索/检索异常均不调用聊天模型。
- 清空操作移除聊天记录、来源区域和原文；整个模拟会话中资源加载函数及向量模型构造均只执行一次。
- 这些检查使用占位密钥和固定返回值，没有读取真实密钥或调用 DeepSeek；索引和元数据文件哈希不变，候选元数据也未被修改。检查脚本只在临时终端中执行，没有新增持久化测试文件。
- 新增校验后的合规不足回答由模拟数据验证，尚未进行新的真实模型复验。后台扫描异常是否完全消失未收到单独确认；已确认页面问答可运行和缓存可复用。

#### 当前边界与下一次学习

当前是固定单篇论文、`top_k=3` 的 Dense RAG。来源区域支持人工核对，尚未实现逐条引用自动绑定、真实性检查或 PDF 定位。原有位置编码漏检案例仍保留，未通过改问法或增大候选数掩盖问题；多轮历史不会自动改写下一轮检索问题。

新增约束拒绝违规结果，不会修改结果、自动重试或保证下次生成合规。其他字段组合、引用与答案支持关系仍需后续完善。会话当前保存在 Streamlit 状态中，尚无跨服务重启的持久化。

**下一次从 Day 14：建立 RAG 评估基线开始。** 逐步扩充固定问题集，分别记录检索命中、答案正确性、引用支持及无答案行为。索引目录命名仍按计划在 Day 15 整理，当前继续使用 `data/indexes/day11_s800_o100`。

### Day 14｜2026-10-20｜首批评估记录与失败分类（进行中）

- **当前状态**：已完成学习日仍为 13 / 42；Day 14 已记录 4 道题，尚未完成约 20 题的扩充与汇总任务。
- **开始时版本**：工作区干净，`HEAD` 为 `5b9ebab`（检索、模型回答、来源展示和异常处理接入 RAG 工作台）；本次未确认或更改远端状态。
- **操作要求**：使用 `sample_questions.json` 的原句，保持现有索引与 `top_k=3`，每题提交前清空会话，保留第一次运行结果。未自动采集会话状态或调用时间。
- **学习重点**：分开判断检索命中、回答正确和引用支持；区分正常的资料不足回答与输出校验失败。

#### 学习者回传结果与核对

下表的片段标识省略共同前缀 `demo_project:attention_is_all_you_need_neurips2017:s800:o100:`。完整问题、标识、回答、引用与错误提示保存在 `data/evaluations/day14_baseline.json`。

| 题目 | 前三候选 | Top-1 | Top-3 | 页面结果 |
| --- | --- | --- | --- | --- |
| q1：编码器层数 | `p2:c7`、`p10:c3`、`p9:c1` | 命中 | 命中 | 回答 6 层并引用对应原文，回答与引用通过人工核对。 |
| q2：GPU 数量与型号 | `p7:c4`、`p10:c3`、`p8:c4` | 命中 | 命中 | 回答 8 块 NVIDIA P100 GPU，依据同时覆盖数量与型号。 |
| q3：位置编码函数 | `p5:c5`、`p10:c1`、`p6:c2` | 未命中 | 命中 | 回答正弦、余弦及公式，直接依据在第三条；第一条只有位置编码动机。 |
| q4：实际耗电量 | 未记录 | 不适用 | 不适用 | 输出校验失败，页面未生成有效回答；错误为“资料不足时，evidence 必须为空列表”。 |

助手核对了保存的候选原文。q3 回传引用在去除空格、换行差异后可匹配 `p6:c2`；公式排版同时参考[原论文第 6 页](https://papers.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf#page=6)。当前问法与 Day 12 位置编码题不同，命中变化不能直接解释为算法改进。

q4 的错误发生在已取得候选并调用模型之后。当前页面在回答通过校验后才保存及展示本轮候选，因此本次未取得候选标识和原始回答。记录中用 `null` 表达缺失，不伪装成空检索，也不补用旧结果。这次失败保留为基线的一部分，程序校验成功拦截不等于问答任务成功完成。

#### 本次保存与后续任务

- 助手新增 `data/evaluations/day14_baseline.json`，按学习者回传内容保存四次结果、配置及核对结论；预期答案文件保持不变。
- 没有调整检索、提示词或校验代码，没有重新调用模型、重跑检索、提交 Git 或删除练习文件。
- 接下来由学习者读取评估 JSON，统计记录总数和正常返回数量，再继续扩充并验证问题集。当前仅有首批记录，不将 Day 14 标为完成。

### Day 14｜2026-10-20｜状态统计与 q5～q8 标准答案准备（进行中）

- **当前状态**：已完成学习日仍为 13 / 42；评估题集已有 8 题，实际页面记录仍为前 4 题。
- **学习者完成**：新增 `practice_evaluation.py`，用 `json.load(file)` 读取评估记录，在 `status == "ok"` 时累加 `ok_count`，实际输出记录总数 4、正常返回数量 3。此比例表示正常返回情况，回答正确性和引用支持仍需独立判断。
- **本次分工**：学习者明确请求助手完成 q5～q8 的标准答案准备。助手从本地固定版本 PDF 的第 5、7 页提取原文，核对答案和页码，并保存扩充题集；本次仅完成这四题的资料准备，没有代跑模型问答。

| 编号 | 标准答案 | PDF 页码 |
| --- | --- | --- |
| q5 | 8 个并行注意力头。 | 5 |
| q6 | Adam 优化器。 | 7 |
| q7 | 4000 个预热训练步。 | 7 |
| q8 | base 模型训练 100,000 步，耗时 12 小时。 | 7 |

完整问题和英文依据保存在 `data/evaluation_questions.json`。其中 q1～q4 沿用原样例，新增四题沿用相同字段结构；长句中提取造成的换行按空白归一化核对，不改写原文的数值或含义。q8 的同一句依据同时覆盖步数和时长。

原 `sample_questions.json` 和 `day14_baseline.json` 保持不变，未把标准答案写成模型实测结果，也没有改变检索、索引或提示词。下一步由学习者按 q5～q8 的问题原句测试工作台、回传真实结果，再继续扩充到约 20 题并补充其他题型。

### Day 14｜2026-10-20｜代跑 q5～q8 与真实结果核对（进行中）

- **用户请求**：学习者明确请求助手代为测试并回传 q5～q8。
- **执行方式**：通过 Streamlit AppTest 驱动现有 `app.py`，实际加载 CPU 向量模型和保存的 FAISS 索引，沿用原提示词、`top_k=3`、DeepSeek 配置与解析校验。观测包装调用原函数并记录输入输出，不替换模型或检索返回值。
- **隔离与次数**：每题使用新的测试会话，提交前验证历史为空；每题仅提交一次，合计 4 次真实 DeepSeek 调用，`max_retries=0`，未重试或额外调用模型评分。
- **保存位置**：新增 `data/evaluations/day14_q5_q8.json`。保留完整候选及分数、实际请求消息、模型解析前内容、校验后字段、页面显示结果、时间、文件哈希与助手核对结论；不记录 API 密钥。
- **当前进度**：8 道标准题已有 8 条实测记录；约 20 题的扩充及其他题型仍未完成，Day 14 保持未完成状态。

#### 本批真实结果

片段标识省略共同前缀 `demo_project:attention_is_all_you_need_neurips2017:s800:o100:`。

| 编号 | 前三候选 | Top-1 | Top-3 | 实际结果与核对 |
| --- | --- | --- | --- | --- |
| q5 | `p10:c5`、`p11:c4`、`p1:c2` | 未命中 | 未命中 | 返回“资料不足，无法回答”，说明候选未提供并行注意力头数量；依据为空。流程正常，但未回答出论文已有答案。 |
| q6 | `p10:c3`、`p7:c4`、`p10:c1` | 未命中 | 命中 | 返回 Adam 及参数 β1=0.9、β2=0.98、ε=10⁻⁹，并说明学习率调整；第二条原文支持回答和附加信息。 |
| q7 | `p7:c4`、`p7:c5`、`p10:c3` | 未命中 | 命中 | 返回 4000 个训练步；第二条提供明确数值，第一条只提及预热变量而未给出其值。 |
| q8 | `p7:c4`、`p11:c4`、`p10:c3` | 命中 | 命中 | 返回 100,000 步、12 小时；第一条原文同时支持两个信息点。 |

q6～q8 的引用经空白归一化后均能在实际候选中找到，语义也支持回答。q5 的原文依据仍存在于当前索引的 `p5:c1`，未进入前三候选，因此判为检索漏检。模型根据当前候选说明不足是合理行为，但题目在论文中有答案，故 `status` 为 `ok`、`answer_correct` 为 `false`；没有引用，`citation_supported` 记为 `null`（不适用）。

本批正常返回 4 / 4，回答正确 3 / 4；Top-1 命中 1 / 4、Top-3 命中 3 / 4；有引用的三条回答均获得原文支持。候选相关性、输出格式合规和答对问题是不同检查项，不用“正常返回比例”代替回答正确率。

#### 验证与保留事项

- 四题均只产生一次检索调用和一次模型调用，实际请求仅含系统要求与本题用户消息，没有历史或评估标准答案。
- 每题都产生两条页面消息和一个来源区域，没有页面错误。原始观测内容在补充助手判定时保持不变。
- 验证 q5 的标准依据存在于完整索引；核对所有候选与已保存元数据一致，保留首次漏检结果，未修改问法、候选数或排序。
- 源码、索引、元数据、标准题集及首批评估记录的文件哈希在运行前后保持不变。没有更改依赖、提示词或应用逻辑，也未执行 Git 提交或清理练习文件。
- 首次运行耗时包含向量资源初始化，记录中的 `total_seconds` 不宜直接与后续请求用来比较模型速度；检索和模型调用另有独立计时。
- 当前 `practice_evaluation.py` 仍读取首批文件；两批结果分别保存，后续再学习合并汇总。

下一步继续扩充问题集，补充事实、跨段落和无答案题，并保留 q4 校验失败与 q5 检索漏检，作为后续改进的对照案例。

### Day 14｜2026-10-21｜合并评估记录与回答正确计数（进行中）

- **学习者完成**：用 `extend(report["results"])` 合并两批记录；分别统计正常返回数量和正常返回且回答正确的数量，实际输出为 8、7、6。
- **排错过程**：用 `append()` 加入整批结果导致嵌套列表，按字符串键读取列表时触发 `TypeError`；修正后又因使用不存在的 `is_correct` 字段触发 `KeyError`，改为实际字段 `answer_correct` 后通过。
- **结果含义**：q4 输出校验失败且正确性未评估；q5 正常返回资料不足，但漏检了论文已有依据，回答正确性为假。其余六题已标记回答正确，正常返回数量与回答正确数量分别计数。
- **助手核对**：读取保存的脚本和两批 JSON，并在项目虚拟环境中运行统计脚本，复现记录总数 8、正常返回数量 7、回答正确数量 6；本次没有调用模型。
- **当前状态**：Day 14 继续保持未完成，已完成学习日仍为 13 / 42。下一步准备需要多处原文支持的题目，再扩充并实测问题集；尚未生成 q9 的标准答案或运行记录。

### Day 14｜2026-10-21｜q9 跨页问题标准答案准备（进行中）

- **用户请求**：学习者请助手完成 q9 的标准答案与原文依据准备。
- **标准问题**：编码器由多少个相同的层堆叠而成？训练时使用了多少块 GPU，型号是什么？
- **预期答案**：编码器由 6 个相同的层堆叠而成；训练时使用了 8 块 NVIDIA P100 GPU。
- **依据与页码**：`The encoder is composed of a stack of N = 6 identical layers.` 位于 PDF 第 2 页；`We trained our models on one machine with 8 NVIDIA P100 GPUs.` 位于 PDF 第 7 页。页码对应项目中的 11 页版本。
- **数据保存**：在 `data/evaluation_questions.json` 追加 q9，保留 q1～q8；用 `evidence_pages` 将两条原文分别对应到第 2、7 页，原单页字段在本题为 `null`，`answerable` 为 `true`。
- **核对方式**：从本地 PDF 提取第 2、7 页文字，核对两条连续原文；检查保存的索引元数据，层数依据位于 `p2:c7`，硬件依据位于 `p7:c3`、`p7:c4`。这证明完整索引包含所需依据，不代表问题检索时一定能找齐。
- **当前状态**：已有 9 道标准题和 8 条实测记录；本次只完成 q9 的资料准备，尚未调用模型实测 q9。已有统计仍为记录总数 8、正常返回数量 7、回答正确数量 6。Day 14 继续保持未完成。

### Day 14｜2026-10-21｜q9 实测与等价依据核对（进行中）

- **学习者回传**：前三片段依次为 `p10:c3`、`p2:c3`、`p2:c7`，判断层数和硬件两项均命中；页面回答为“编码器由 N=6 个相同的层堆叠而成。训练时使用了 8 块 P100 GPU。”，附两条引用。
- **引用核对**：第 3 名包含 `Encoder: The encoder is composed of a stack of N = 6 identical layers.`；第 2 名包含 `translation quality after being trained for as little as twelve hours on eight P100 GPUs.`。两段文字均可在保存的候选和本地 PDF 第 2 页找到；核对该页完整上下文，硬件描述对应 Transformer。
- **评估结论**：第一名参考文献没有支持本题；第二、三名共同覆盖所需信息。Top-1 未命中、Top-3 完整命中，正常返回、回答正确、引用支持。回答中的 P100 GPU 与预期型号一致，本题未另行要求厂商名称。
- **认识修正**：标准引用不是唯一有效依据。原先准备的是第 2、7 页的一组依据，实际检索找到了第 2 页的等价硬件表述，所以本次是同页跨片段问答，不是必须跨页才能回答的问题。保留原标准答案和历史记录，在新记录中保存实际引用映射。
- **保存与验证**：新增 `data/evaluations/day14_q9.json`，逐字保存回传回答和引用，记录按保存的候选及 PDF 核对的结果。没有重跑检索或调用模型，也未将人工回传内容写成原始 API 响应。助手合并三批记录核对得到 9、8、7；本地统计练习待学习者补充 q9 文件路径。
- **当前状态**：9 道标准题已有 9 条实测记录；Day 14 仍未完成约 20 题的扩充与汇总。继续补充不同问题及无答案案例，保留 q4 校验失败和 q5 检索漏检作为对照。

### Day 14｜2026-10-21｜三批统计通过与 q10 无答案题准备（进行中）

- **学习者完成**：在 `report_paths` 中加入 `data/evaluations/day14_q9.json`，回传记录总数 9、正常返回数量 8、回答正确数量 7。
- **助手核对**：检查保存的统计脚本，并用项目虚拟环境运行，复现 9、8、7；本次未改动学习者代码。
- **下一题准备**：在标准题集中追加 q10：“论文训练模型所用机器的 CPU 型号是什么？”核对本地 PDF 第 7 页硬件说明，文中给出了 GPU 数量、型号和训练计划，没有提供 CPU 型号；全文提取文本中的 CPU、CPUs、processor、Xeon 关键词检索均未命中。
- **评估重点**：CPU 型号无法从 GPU 型号推出。预期行为为明确说明资料不足、依据列表为空，并解释缺少 CPU 信息。无答案题不评检索命中，正常拒答与输出校验失败应分别记录。
- **当前状态**：10 道标准题、9 条实测记录；q10 尚未提交工作台，没有新增模型调用或实际回答记录。Day 14 继续保持未完成。

### Day 14｜2026-10-21｜q10 正常资料不足回答验收（进行中）

- **学习者回传**：候选依次为 `p11:c4`、`p1:c2`、`p9:c1`；回答为“无法依据资料回答。提供的参考片段未提及训练模型所用机器的 CPU 型号。”，原文依据显示为“无。”，另附资料不足说明。
- **助手核对**：三个候选分别是参考文献、摘要中的架构与实验描述、架构变体实验表，没有 CPU 型号信息。资料不足判断与已核对的标准题一致，没有用 GPU 型号替代 CPU 型号。
- **评估结论**：正常返回且回答正确，`actual_evidence` 整理为空列表，并保留原始页面显示。无答案题不评检索命中，没有引用则不评引用支持，相关指标为 `null`，不记为命中失败或引用支持成功。
- **保存位置**：新增 `data/evaluations/day14_q10.json`，逐字保存页面回答和不足说明，明确区分人工回传与原始 API 响应；没有重跑检索或调用模型。
- **对照案例**：q4 是输出校验失败；q5 是论文有答案但检索漏掉依据；q10 是论文没有所问信息，正常说明资料不足。这三种情况分别记录。
- **汇总与后续**：助手核对四批记录合计为记录总数 10、正常返回数量 9、回答正确数量 8；本地统计脚本待加入 q10 路径。Day 14 仍在进行，约 20 题的扩充及整体汇总尚未完成。

### Day 14｜2026-10-21｜资料不足判断澄清与 q11～q12 准备（进行中）

- **理解修正**：学习者最初将 q5 与 q10 的差别理解为问题和论文的相关性。讲解后，学习者对“论文明确提供 CPU 型号，但检索漏掉，最终回答资料不足”的情形正确判断为 `false`。判断依据是论文是否有答案，以及系统是否完成题目；模型根据缺少依据的当前片段说明不足仍可合理，但整体问答可能因检索漏检而失败。
- **统计脚本**：确认学习者已加入 q10 路径，助手运行当前脚本得到记录总数 10、正常返回数量 9、回答正确数量 8。
- **下一组标准题**：追加 q11 的 big 模型训练步数和时长，以及 q12 在逐位置前馈网络小节中的输入、输出、内部层维度。分别对照本地 PDF 第 7、5 页的连续原文，并核对保存的分块中存在相应依据。
- **测试安排**：两题作为一组，每题前清空会话，沿用当前 Top-3，回传首次候选、回答和引用，再统一记录与汇总。本次未运行这两题的检索或模型问答。
- **当前状态**：12 道标准题、10 条实测记录；q11～q12 待实测，Day 14 保持未完成。

### Day 14｜2026-10-21｜代跑剩余 10 题并完成评估基线（已完成）

- **用户请求**：学习者明确请助手完成剩余 q11～q20 的测试；Day 14 按固定 20 题收尾。
- **资料准备**：保留已有 q1～q12，补齐 q13～q20 的标准答案、原文与页码；新增束搜索、标签平滑、多项事实组合，以及 RAM、CUDA、随机种子无答案题。答案在正式测试前固定，并与本地 PDF 核对。
- **执行方式**：Streamlit AppTest 驱动原工作台，实际使用 CPU 向量模型、FAISS 和 DeepSeek；每题使用新会话，实际请求无历史或标准答案，10 道题各调用一次，未对失败结果进行付费重试或另用模型评分。
- **测试脚本排错**：首次请求核对遇到 PowerShell 管道将中文标点替换为问号，导致测试脚本断言失败，模型调用数为 0。修正字符构造后完成正式测试，原诊断完整保留，不作为模型失败样本。应用代码与已保存索引未因此改动。
- **本批结果**：正常返回 7 / 10，回答正确 5 / 10；q11、q14 和三道无答案题通过。q12、q13 漏检后正常说明不足；q15～q17 只召回部分依据，模型同时给出引用和不足说明，被现有字段组合校验拦截。
- **全部 20 题**：正常返回 16 次、回答正确 13 次；3 次正常但未答出、4 次输出校验失败。有答案的 15 题中 Top-1 完整支持 4 / 15、Top-3 完整支持 9 / 15；带引用的 9 条有效回答均获得来源支持；5 道无答案题中 4 道正常且正确说明不足。
- **保存与统计**：新增 `data/evaluations/day14_q11_q20.json` 和 `day14_summary.json`；补入统计练习的最后一批路径，保留学习者的计数逻辑；README 同步当前进度、指标分母、失败分类与复现方式。旧题、旧报告和以往学习日志保留原样。
- **验收边界**：已完成的是可追溯的评估基线，未将失败改写为成功；这些是单篇论文的开发样例，不作为独立泛化评估。当前回答结构对部分作答的限制已记录，留作后续改进依据。
- **下一次学习**：Day 15，整理 Project、Document、Chunk 的持久化模型，并将索引目录从学习日命名改为项目、文档与索引版本命名。原有练习文件继续按约定保留。

### Day 15｜2026-10-22｜数据关系与索引目录规范（开始）

- **当天目标**：按三个步骤完成规范路径生成、项目与文档元数据的 JSON 保存和读取、工作台按元数据加载索引。收尾时在新进程中仍能读取项目与文档记录，并从规范目录加载现有 51 条片段。
- **当前检查**：片段已有 `project_id`、`document_id`、来源和分块配置；正式的 `index_store.py` 已负责索引与片段元数据的保存、加载。项目和文档尚无独立记录，工作台与索引练习仍使用 `data/indexes/day11_s800_o100`。
- **拟采用的目录**：`data/indexes/{project_id}/{document_id}/{index_version}/`。目录对应项目、文档和某次索引构建版本；向量模型与分块等具体配置继续由元数据说明，索引构建版本与文件格式的 `schema_version` 区分。
- **第一步练习**：由学习者在正式模块 `project_store.py` 编写 `get_index_dir(project_id, document_id, index_version)`，返回 `Path` 对象。先验证路径拼接，再逐步接入保存、加载与工作台。
- **当前状态**：本次仅完成现状检查和练习安排，尚未创建上述模块、迁移索引或修改工作台；Day 15 保持未完成，完成进度仍为 14 / 42。

### Day 15｜2026-10-22｜项目记录持久化与索引目录接入（已完成）

- **当天目标**：完成规范路径生成、项目与文档 JSON 保存读取、工作台按记录定位索引。实际用时未记录。
- **学习者完成**：编写 `project_store.py` 中的三个函数；使用 `Path` 拼接路径，用 `json.dump()` 写文件、`json.load()` 读取文件；生成 `data/projects/demo_project/project.json`，保存项目名称及文档来源、索引版本。
- **关系与标识**：项目包含文档列表，片段已有项目、文档及来源标识；通过 `project_id`、`document_id`、`index_version` 确定索引目录。区分路径对象与字符串、JSON 文件与 Python 字典，以及索引构建版本与文件格式版本。
- **迁移与接入**：学习者将旧索引复制到 `data/indexes/demo_project/attention_is_all_you_need_neurips2017/v1/`，在 `app.py` 读取项目和第一篇文档，补齐路径函数实参，再传给现有缓存加载函数；重启页面后回传可检索片段为 51。
- **助手核对**：在新进程中实际读到项目字典、新目录的 51 个向量及 51 条片段，维度为 384；所有片段的项目、文档与来源均匹配项目记录。新旧两个索引文件分别进行 SHA-256 对比，内容完全一致。
- **收尾检查**：更新后的 `practice_load_index.py` 在新进程中运行通过，第三个向量匹配回原片段，相似度显示 1.0；涉及的五个 Python 文件语法检查通过，README 的构建与读取示例与本地脚本一致。核对确认学习者的 `app.py` 实现及此前每日记录保持原样。
- **收尾维护**：助手同步三个本地索引练习的路径来源与构建时的文档信息，更新 README 的当前能力、目录、启动及重建示例；保留学习者核心实现、旧索引和历史评估记录。练习文件继续遵守原保留约定。
- **验收边界**：本日完成数据关系与索引定位；页面仍固定读取一个项目的第一篇文档，尚未实现项目选择、文档导入、记录格式校验或并发写入。本次没有新增 DeepSeek 调用，也没有重跑 Day 14 的 20 道题。
- **当前状态**：Day 15 已完成，进度为 15 / 42；下一次从 Day 16：创建与切换项目开始。

### Day 16｜2026-10-22｜创建与切换项目（开始）

- **现状检查**：确认上次提交为 `415f954`，开始前工作区干净。`project_store.py` 已有路径生成及单项目保存读取函数，当前只有 `demo_project`；工作台仍固定读取该项目和 `documents[0]`。
- **当天安排**：先实现项目列表和新建空项目，再接入页面创建与选择，最后处理文档展示、空项目状态及切换时清空聊天。按现有 45–60 分钟的学习节奏分步完成，由学习者编写关键代码。
- **验收范围细化**：文档上传与索引构建安排在 Day 17，本日以已有论文项目和新建空项目验证创建、切换、持久化及文档列表归属。空项目应有正常提示，不能继续访问 `documents[0]`。切换时采用清空当前聊天的基础规则，完整会话隔离仍在 Day 19 学习。
- **第一步练习**：在 `project_store.py` 新增 `list_projects()`，扫描 `data/projects/*/project.json`，复用 `load_project()` 读取每个项目并返回字典列表。学习 `Path.glob()`、父目录名称与稳定排序，先用终端验证已有项目能被列出。
- **当前状态**：本次仅检查代码并安排练习，尚未实现列表、创建或页面切换；Day 16 保持未完成，进度为 15 / 42。

### Day 16｜2026-10-23｜项目创建函数与页面切换通过（进行中）

- **学习者完成**：实现 `list_projects()`，扫描各项目 JSON 并返回项目字典列表；实现 `create_project(name)`，去除名称首尾空白、拒绝空名称，以 UUID 生成项目标识，保存并返回文档列表为空的新项目。
- **持久化核对**：终端创建 `My second project` 后项目总数为 2，助手在新进程中读到已有论文项目和新建空项目；空白名称触发 `ValueError`，项目文件数量与内容均未改变。
- **页面接入**：学习者补齐 `load_project(selected_project_id)`，通过侧栏下拉框选择项目；代码在项目变化时清空当前聊天，并在访问第一篇文档前处理空列表。
- **人工回传**：学习者确认切换结果符合预期：Demo project 的文档数为 1、可检索片段为 51；My second project 的文档数为 0，显示空项目提示且没有问答输入框。此回传验证页面选择与空项目状态，尚未单独验收带历史消息时的切换清空。
- **下一步**：在项目列表加载之前加入创建表单，创建成功后通过下拉框对应的会话状态选中新项目；补齐文档列表展示后统一收尾。Day 16 仍未完成，进度为 15 / 42。

### Day 16｜2026-10-23｜页面创建、文档展示与切换验收（已完成）

- **完成情况**：学习者实现页面创建表单，补齐名称传参及新项目自动选中；回传成功创建与空名称拒绝均符合预期。新进程确认页面创建的 `test` 已保存，当前共三个项目。
- **文档展示**：学习者编写 `Path(document_record['source']).name` 并用侧栏显示文件名。初次插入位置早于 `project = load_project(...)`，出现变量尚未定义的错误；随后学习者移动循环，助手删除残留的提前取第一篇文档语句，保留后面正确位置的赋值。
- **学习要点**：`projects` 是项目列表，`project` 是选中后读取的单个项目字典；调用列表函数不会自动在页面定义单个项目变量。使用变量前必须先赋值，访问第一篇文档前必须先检查列表非空。
- **验证方式**：助手以 Streamlit AppTest 运行修复后的真实页面，使用真实项目和索引文件，以测试对象替代向量模型，并预置一轮模拟聊天和来源。验证同项目保留、切换清空、切回不恢复、文档列表归属、空项目停止加载及空名称表单提示均通过。DeepSeek 与检索调用次数均为 0，真实索引加载一次。
- **保存与边界**：检查未改动项目或索引数据；同步 README 当前能力、进度、运行方式及本记录，并清除代码空行上的多余空格。本日未重跑 Day 14 评估，没有新增依赖；项目分别保留会话、上传文档与检索范围仍按后续计划实现。
- **当前状态**：Day 16 已完成，进度为 16 / 42，实际用时未记录。页面关闭了文件监听，学习者需重启 Streamlit 加载本次修复；下一次从 Day 17：文档导入生命周期开始。

### Day 17｜2026-10-23｜文档导入生命周期（开始）

- **现状检查**：上次提交为 `4a13cc5`，开始前工作区干净；学习者已确认 Day 16 修复后页面符合预期。现有三个项目中 Demo project 有一篇论文，另外两个项目为空。
- **可复用代码**：PDF 逐页读取、分块、索引保存读取已有正式模块；生成向量和建立 FAISS 索引的流程仍在本地 `practice_index.py`。本日将这些环节逐步接入页面导入流程。
- **当天安排**：上传并识别重复文件；按项目保存 PDF，串联解析、分块和索引；成功后登记文档，失败时提示具体环节。问答沿用当前首篇文档方式，检索范围选择仍在 Day 18 学习。
- **收尾目标**：在现有空项目中导入固定论文并加载索引；同一项目重复导入相同文件不增加记录；解析或无可用文字等失败有明确提示，未完成导入不作为成功文档使用。已有示例文档也应纳入重复识别考虑。
- **第一步练习**：在读取所选项目之后、空项目停止执行之前加入单 PDF 上传控件，按项目设置控件 key；由学习者读取上传对象的字节，临时显示文件名、数据类型和字节数。区分选中文件、获得文件字节、保存到磁盘和完成索引导入。
- **当前状态**：本次仅检查代码、核对上传接口并安排练习，尚未修改应用或导入文件；Day 17 保持未完成，进度为 16 / 42。原有练习文件继续保留。

### Day 17｜2026-10-23｜上传查重与按项目保存解析通过（进行中）

- **学习者完成**：在空项目停止执行之前加入上传控件，使用包含项目标识的 key；通过 `getvalue()` 取得 569417 字节的 PDF 内容，在正式模块 `document_importer.py` 中实现 SHA-256 摘要、查重和文件保存函数。
- **摘要与兼容**：固定论文的摘要为 `d87d482d5ae7960e2e43d7dd6d21377e60e73e8fce1bf2a01aff7aca8a08c537`。查重只遍历当前项目的文档；旧记录缺少 `file_hash` 时，从原文件计算。学习者回传 Demo project 识别为已有论文，test 识别为尚未导入，助手只读复核一致。
- **保存与解析**：点击按钮后才写文件并调用已有 `load_pdf_pages()`。上传副本位于 `data/papers/{project_id}/doc_{file_hash}/{文件名}`；学习者正确使用 `write_bytes()` 保存，并把路径字符串和文档标识传给解析函数，页面显示 11 页。
- **助手复核**：保存副本的字节数和 SHA-256 与原论文一致，解析得到 11 页，无未提取到文字的页；用已有 800 / 100 参数只读检查得到 51 条片段。此检查尚未生成向量或保存新索引。
- **下一步**：将已有分块、向量编码、FAISS 建立和保存逻辑整理为正式函数，接入导入按钮。当前 test 项目尚未登记这份文档，页面仍显示空项目提示；Day 17 保持未完成，进度为 16 / 42。

### Day 17｜2026-10-23｜索引构建、文档登记与失败检查（已完成）

- **学习者完成**：将分块、向量编码和 FAISS 保存逻辑整理为正式 `build_document_index()`，正确添加向量并回传 51 条片段、384 维。随后在建索引成功后追加文档并保存项目，test 的文档数变为 1；再次上传显示已存在，数量不增加。主区标题已改为当前项目和实际文件名。
- **助手核对**：在新进程中加载实际新索引，读取 51 个向量、51 条片段和 384 维；项目、文档、来源及编码配置均正确。项目 JSON 保存了摘要、11 页和 51 条片段信息。
- **收尾改动**：为查重与导入各环节增加失败提示，捕获常见 PDF 读取错误；移除临时摘要、类型等调试输出。登记改为先准备更新记录，保存成功后再更新页面；项目 JSON 改为同目录临时写入后替换，失败时清理临时文件，防止直接截断旧记录。
- **验收方式**：在隔离副本中用真实 AppTest 上传控件执行 8 组检查，覆盖完整导入、重启会话、改名重复、文件写入失败、无效 PDF、空白 PDF、索引写入失败、部分 JSON 写入失败和重试。重复及旧格式兼容合为一组，完整导入与新会话合为一组；全部通过。
- **验证边界**：学习者已真实执行本地向量编码；助手收尾检查使用相同片段的已保存向量回放，未再次运行向量模型推理，DeepSeek 与问题检索调用均为 0。临时副本已清理，实际项目、PDF、索引及评估报告内容未改动。
- **文档与保留事项**：更新当前能力、目录、运行与重建说明、接口及失败边界；旧日志和 Day 14 基线保留，指定的三个练习文件继续保留。没有新增依赖，实际用时未记录。
- **当前状态**：Day 17 已完成，进度为 17 / 42。重启 Streamlit 可加载收尾修改；下一次从 Day 18：限定检索文档范围开始。

### Day 18｜2026-10-23｜限定检索文档范围（开始）

- **现状检查**：Day 17 的改动仍在工作区，最近提交为 `4a13cc5`。Demo project 和 test 各有一篇论文，My second project 为空；页面仍用 `project['documents'][0]` 加载第一篇文档，检索函数只接收一个索引。
- **当天安排**：实现所选文档记录的获取；接入当前项目全部文档或指定文档的选择；让检索实际使用所选文档的索引与片段，并验收单选、多选、全部及空选择。范围改变时先采用清空当前聊天的基础规则，完整会话策略仍在 Day 19 学习。
- **关键约束**：只展示选择框不足以限定检索；后端需要拒绝不属于当前项目的文档标识，空选择不能自动退回全部文档或第一篇文档。合并检索前需要核对索引归属、维度及编码配置。
- **第一步练习**：由学习者创建正式模块 `retrieval_scope.py`，编写 `get_selected_documents(project, selected_document_ids)`；输入项目记录和文档标识列表，返回所选文档字典列表，对未知标识报错。先用两份模拟文档记录检查单选和空选择，不读取实际 PDF。
- **验收资料**：目前单个项目只有一篇论文；后续需要在同一项目准备第二份内容可区分、明确标注用途的测试 PDF，验证不同选择确实改变候选来源。模拟记录练习不计作多文档检索验收。
- **当前状态**：本次只检查代码并安排练习，尚未新增范围模块、修改检索或改变项目数据；Day 18 保持未完成，进度为 17 / 42。

### Day 18｜2026-10-23｜按所选标识取得文档记录（进行中）

- **学习者完成**：在正式模块 `retrieval_scope.py` 中实现 `get_selected_documents()`，使用 `selected_documents.append(documents_by_id[document_id])` 完成查找和加入列表；单选 `paper_b` 返回一条对应记录，空选择返回空列表。
- **助手核对**：读取实际代码确认与回传一致，函数保留了未知文档标识的拒绝分支；本步未运行模型、检索或新增自动测试。
- **下一步**：在文档列表之后、原有第一篇文档赋值之前加入全部／指定文档选择控件，并把标识列表传给本函数。练习阶段在展示所选记录后暂时停止页面执行，避免选择控件与旧的首篇检索行为不一致；完成范围索引加载后再恢复问答。
- **当前状态**：仅文档记录选择函数完成，页面控件与实际范围检索尚未接入；Day 18 保持未完成，进度为 17 / 42。

### Day 18｜2026-10-23｜范围选择控件通过，准备合并索引（进行中）

- **学习者完成**：把 `selected_document_ids` 传入文档选择函数；回传 Demo project 的全部、指定但未选、指定一篇三个场景分别为 1、0、1。页面仍保留临时 `st.stop()`，尚未恢复问答。
- **助手核对与修正**：读取实际页面确认参数正确，将 `document_names = {}` 从文档展示循环内移到循环后。只读检查 Demo 索引，通过 `reconstruct_n()` 取回 `(51, 384)` 的 `float32` 向量，对应 51 条片段，未调用向量模型或 DeepSeek。
- **下一步框架**：在 `retrieval_scope.py` 加入 `load_scope_index()` 教学框架，按所选记录读取原索引，检查非空范围、索引类型、向量维度、编码配置、片段归属及重复标识；合并对象只存在于内存，不保存新索引目录。
- **学习者待完成**：替换函数内明确标注的 `NotImplementedError`，补齐添加向量和展开合并片段列表的两句代码。当前框架有意保持未完成，尚未接入页面；接入前还需检查合并结果与向量、片段顺序对应关系。
- **当前状态**：页面选择通过，实际范围检索尚未完成。后续还包括范围变更清空历史、空选择停止检索及多文档来源验收；Day 18 保持未完成，进度为 17 / 42。

### Day 18｜2026-10-23｜索引合并通过并接回页面（进行中）

- **学习者完成**：使用 `combined_index.add(vectors)` 和 `combined_chunks.extend(chunks)` 完成两句合并代码，回传向量 51 条、片段 51 条、384 维。助手逐项比对真实原索引与合并索引，向量值、顺序和片段列表一致。
- **页面接入**：移除教学暂停和固定第一篇文档的加载方式，改为加载所选范围；页面显示所选数量、文件名和范围内片段数。范围改变先清空聊天和来源，再在空选择时停止；所选标识排序后用于缓存，问答模型按模型名单独复用。
- **检查结果**：真实 AppTest 配合两个文档的控制向量索引通过单选、多选、空选择、等价范围与选择顺序、切换项目及索引版本变化检查。参考资料与保存的候选均来自所选范围，旧范围历史不会进入新请求；不兼容维度、配置、类型、归属和重复片段均被拒绝。
- **验证边界**：四次回答均为模拟结果，真实 DeepSeek 调用为 0；控制向量用于检查范围与排序，不用于判断语义检索质量。本次没有新增依赖，也没有重新执行 Day 14 基线。
- **测试清理**：首次测试脚本按控件位置定位，误触创建项目表单，产生一条空的测试项目。已核验并删除该条测试记录，改用标签定位，并阻止测试脚本写入真实项目。重跑通过；最后比较原有 19 个数据文件的路径与哈希，全部一致，临时索引夹具已清理。
- **当前状态**：范围选择与页面检索已接通，README 同步当前实现及缓存限制。待学习者确认页面恢复，并用同一项目的两份不同 PDF 完成实际来源验收；Day 18 保持未完成，进度为 17 / 42。

### Day 18｜2026-10-23｜页面确认通过，准备双 PDF 验收（进行中）

- **学习者回传**：Demo project 全部文档显示 51 个片段与问答入口；指定但未选择时提示并停止问答；重新选中论文后恢复，三项符合预期。
- **助手准备**：生成 `data/fixtures/day18_scope_note.pdf`，明确标注人工测试资料，含独有校验码；用项目现有 PDF 解析和分块函数检查为 1 页、1 个片段。原有 19 个数据文件的路径和内容保持一致，尚未导入、生成新索引或调用 DeepSeek。
- **验收安排**：由学习者在 test 项目导入素材，与原论文组成两份不同文档。仅测试 PDF、仅原论文、指定两份各问一次校验码，记录片段总数、候选来源与实际回答；最后切换全部文档只核对 52 个片段，无需再问。
- **判断边界**：范围约束要求返回片段全部属于所选文档，不要求每份文档都进入前三候选。回传后区分来源越界、召回遗漏与回答问题，按实际情况收尾。
- **当前状态**：Day 18 的页面确认完成，剩余双 PDF 实际验收；进度保持 17 / 42，未提前勾选完成。

### Day 18｜2026-10-23｜双 PDF 验收与收尾（已完成）

- **学习者回传**：test 项目已登记原论文与测试 PDF 两份文档。A 组仅选测试资料，显示 1 个片段，正确返回 `SCOPE-B-742`，来源为测试 PDF 第 1 页；B 组仅选原论文，显示 51 个片段，明确资料不足且依据为空；C 组选择两份，显示 52 个片段，第 1 名为测试 PDF，正确返回校验码并引用 `The verification code of this test document is SCOPE-B-742.`。
- **实际索引核对**：助手只读加载两份正式索引，确认 1、51、52 条向量与片段均为 384 维，全部片段的项目、文档与来源归属正确；两份已保存 PDF 的 SHA-256 与登记记录一致。
- **页面补查**：用户未单独回传全部模式，助手以真实项目和索引运行 AppTest，确认全部模式为 52，单选为 1 或 51，多选为 52；预置模拟聊天验证范围改变会清空，空选择停止问答，选择顺序及等价模式变化保留历史。三个范围仅初始化一次测试模型对象。
- **验证边界**：本轮自动检查未提交问题，向量模型为不可推理的测试替身，模型推理与 DeepSeek 调用均为 0；三组实际回答以学习者回传为依据，未补造未提供的完整候选列表或 A 组原文引用。检查前后 23 个数据文件路径与哈希一致，没有改写项目、PDF、索引或 Day 14 评估结果。
- **文档收尾**：更新当前能力、进度、验收表和按项目逐份重建索引的说明；保留旧日志、测试资料和约定保留的练习文件，没有新增依赖，也未自动提交 Git。测试资料继续用于后续隔离和删除同步学习。
- **当前状态**：Day 18 已完成，进度为 18 / 42，实际用时未记录。今天无需继续测试；下一次从 Day 19：隔离项目与会话上下文开始。

### Day 19｜2026-10-24｜隔离项目与会话上下文（开始）

- **现状检查**：学习者已提交 Day 17–18 的改动，当前 HEAD 为 `0d27611`，开始时工作区干净。页面已在项目或实际范围变化时清空 `messages`，相同范围保留；会话状态仍分散在 `messages`、`active_project_id` 与 `active_scope` 中，消息列表没有独立会话标识。
- **当天安排**：先定义包含 `session_id`、`scope_key`、`messages` 的会话记录；再集中处理相同范围继续、项目或范围变化重建、手动清空重建等规则；最后在组装模型历史前核对会话归属，并验证聊天和来源不会跨范围使用。
- **范围说明**：沿用 Day 18 的项目标识与排序后文档标识、版本、来源作为范围依据；本日会话仍保存在内存，长期保存与重启恢复按 Day 39 安排。已有检索和导入能力继续复用，测试 PDF 与约定保留的练习文件继续保留。
- **第一步练习**：由学习者创建 `chat_session.py`，实现 `create_chat_session(scope_key)`。使用已有的 UUID 思路生成新会话标识，补齐范围归属和空消息列表两个字段；用模拟范围验证相同范围可建立两次独立会话，修改其中一份消息不影响另一份。
- **当前状态**：仅检查代码、更新进度并安排首个练习，尚未新增会话模块或修改应用，没有调用模型、改动项目数据或重跑既有评估。Day 19 保持未完成，进度为 18 / 42。

### Day 19｜2026-10-24｜独立会话记录创建通过（进行中）

- **学习者完成**：在 `chat_session.py` 中实现 `create_chat_session(scope_key)`，保存传入的范围标识，每次调用创建新的 UUID 会话编号和独立的空消息列表。回传范围匹配、初始空列表、编号不同均符合预期；向 A 添加一条消息后，A 为 1 条、B 仍为 0 条。
- **助手核对**：读取实际文件确认两个字段及空列表创建位置正确，没有修改学习者代码或重跑模型。
- **下一步练习**：新增纯 Python 函数 `ensure_chat_session(session, scope_key)`；首次无会话或完整范围改变时新建，相同范围返回原会话。用模拟记录验证继续、切换及切回时的编号和消息数量，沿用切换后不恢复旧聊天的规则。
- **当前状态**：会话创建函数完成，切换函数与页面接入尚未完成；当前应用仍使用 Day 18 的状态管理，Day 19 保持未完成，进度为 18 / 42。

### Day 19｜2026-10-24｜会话继续与切换规则通过（进行中）

- **学习者完成**：在 `ensure_chat_session()` 中用 `session['scope_key'] != scope_key` 判断完整范围是否改变。首次创建后添加一条消息，同范围继续时保留原编号与 1 条消息；换范围、切回原范围均为新编号和 0 条消息，三组回传符合预期。
- **助手核对**：读取实际代码确认首次无会话分支、范围不匹配分支及返回原会话的分支正确；本次未修改应用或调用模型。
- **下一步练习**：由学习者编写 `build_history_messages(session, scope_key)`，先拒绝范围不匹配的会话，再从消息记录提取 `role` 与 `content`，生成独立的列表和字典。原会话及 `retrieval_results` 保留，供页面展示；本轮候选仍单独放入本轮参考资料。
- **验收安排**：本地检查历史字段、原记录保留与范围不匹配时的 `ValueError`。三个核心函数完成后统一接入页面，并检查项目切换、范围切换、清空、来源展示及模型请求中的历史归属。
- **当前状态**：创建与切换函数已完成，历史校验函数及页面接入待完成；Day 19 保持未完成，进度为 18 / 42。

### Day 19｜2026-10-24｜历史校验、页面接入与验收（已完成）

- **学习者完成**：实现 `build_history_messages()`，范围不匹配时抛出 `ValueError`，逐条创建只含 `role`、`content` 的新字典。回传原候选仍保留、修改整理后的消息不影响原问题，以及错误范围被拒绝；助手读取实际代码并在页面请求中核对字段正确。
- **页面接入**：导入三个函数，以 `chat_session` 统一保存编号、范围与消息。切项目时先解除旧归属，再读取目标项目；空项目与空选择在停止前更新会话。清空按钮创建同范围的新会话，输入框 key 关联编号；展示与组装历史前调用归属检查，成功问答及来源写回当前会话。
- **验收结果**：九组本地检查通过，覆盖同范围追问、清空、范围及项目切换、空范围与空项目、选择顺序与等价模式、两个独立页面、模型失败、错范围拦截及目标项目读取失败。旧范围的消息与来源不会进入新请求，同范围候选仍可正常展示。
- **测试边界**：使用真实页面与索引，八次模拟模型请求中含一次模拟失败；向量查询使用已保存向量回放，没有执行真实模型推理或 DeepSeek 调用。错误会话注入在渲染、检索及调用前被拦截；23 个数据文件路径与哈希保持一致。
- **文档与保存**：更新 README 当前能力、接口、会话规则与任务状态；保留旧学习日志、既有评估和约定的练习文件。没有新增依赖或自动提交 Git，会话仍只保存在内存中。
- **当前状态**：Day 19 已完成，进度为 19 / 42，实际用时未记录。重启工作台即可加载本次改动，今天无需追加模型测试；下一次从 Day 20：删除文档与同步索引开始。

### Day 20｜2026-10-24｜删除文档与同步索引（开始）

- **现状检查**：当前 HEAD 为 `0d27611`，Day 19 的 `app.py`、`chat_session.py` 与 README 改动尚未提交，继续保留。三个项目分别有 1、2、0 份文档；test 项目有原论文副本与测试 PDF 副本，源测试素材另存于 `data/fixtures/`。尚未实现文档删除入口。
- **当天安排**：准备移除指定文档后的新项目记录；连接记录保存、所属文件及索引清理，并处理失败；接入页面删除操作，更新所选范围、索引缓存和会话，验证目标文档不再进入检索且其他文档正常。
- **实现依据**：项目 JSON 是当前可选文档的登记来源，PDF 与索引各自按项目和文档定位，问答还会复用内存中的范围索引。旧版 Demo PDF 位于 `data/papers/` 根目录，上传副本则位于项目专属目录；后续文件清理需要依据实际归属检查路径，区别处理这些位置。
- **第一步练习**：由学习者创建 `document_deleter.py`，编写 `remove_document_record(project, document_id)`，返回项目副本与找到的目标文档记录。使用新的文档列表替换副本中的 `documents`，保留原项目；未知或重复目标标识报错。先以模拟的 A、B 两条记录检查移除 B、原记录保持及未知标识处理。
- **当前状态**：只完成现状检查和练习安排，尚未创建删除模块、保存新项目记录或删除任何文件；没有调用模型。约定保留的练习文件继续保留，Day 20 未完成，进度保持 19 / 42。

### Day 20｜2026-10-24｜准备移除后的项目记录通过（进行中）

- **学习者实现**：创建 `document_deleter.py`，在 `else` 分支将其他文档加入保留列表，并返回项目副本和目标记录。目标判断的字段名先写为 `index`，后改为 `id`，与记录中的实际键名不符，触发 `KeyError`。
- **修正与要点**：助手核对当前文件，将条件改为 `document['document_id'] == document_id`。方括号中的字符串必须与字典实际键名一致；右侧参数表示此次要移除的目标标识。
- **检查结果**：运行已有的模拟字典练习，更新后的文档为 `['paper_a']`，目标记录为 `paper_b`，原项目仍为 `['paper_a', 'paper_b']`；未知的 `paper_c` 正确抛出 `ValueError`。未保存项目记录、删除文件或调用模型。
- **当前状态**：纯记录处理函数完成，实际保存、关联文件与索引清理、页面及缓存同步尚未接入；Day 20 保持未完成，进度为 19 / 42。

### Day 20｜2026-10-24｜定位 PDF 与文档索引目录（进行中）

- **现状核对**：test 项目仍有原论文和测试资料两份文档，PDF 副本位于各自的项目／文档目录，索引登记版本均为 `v1`。Demo 原始论文位于 `data/papers/` 根目录，原测试素材位于 `data/fixtures/`，与 test 项目的上传副本是不同路径。
- **本步练习**：由学习者在 `document_deleter.py` 中新增 `get_document_paths(project_id, document)`，从记录的 `source` 取得 PDF 路径，并用 `get_index_dir()` 定位当前索引版本，再取父目录得到该文档的索引目录；两个路径均转为绝对路径后返回。
- **学习要点**：文档删除需考虑该文档的全部索引版本，因此预览的索引目标是 `data/indexes/{project_id}/{document_id}/`，不是其中某个版本目录。`resolve()` 用于明确路径位置，本步定位结果尚不代表已经完成文件归属检查。
- **后续接入**：助手在实际清理前补齐路径归属检查、原始公共 PDF 的处理及失败处理，再连接项目保存和页面缓存同步。本步只打印路径，不调用保存、移动或删除操作。
- **当前状态**：路径定位练习已安排，删除功能仍未接入，Day 20 保持未完成，进度为 19 / 42。

### Day 20｜2026-10-24｜路径定位通过，准备删除流程（进行中）

- **学习者完成**：从文档记录的 `source` 字段取得 PDF 路径，从 `document_id` 与 `index_version` 定位索引目录。回传测试 PDF 与文档索引目录均存在，索引目标止于文档目录，包含其下各版本。
- **字段查找**：通过 `app.py` 中创建文档记录的位置、项目 JSON 及 `document.keys()` 确认字典键名；`source_path`、`path` 均不是当前记录中的来源字段。
- **助手核对**：读取实际函数确认字段已修正，未重新运行路径检查或改写项目数据。本次回传仅证明路径定位和存在性，后续清理仍需核对目录归属；项目专属副本与公共原始 PDF 分别处理。
- **下一步**：先理解项目记录与文件分开保存时的失败情况，再连接保存与清理；以“先删文件、后保存记录，但保存失败”的场景判断会留下什么状态。实际删除前由助手补齐归属检查和失败处理。
- **当前状态**：两项基础函数已完成，文件清理、保存协调与页面同步尚未实现；Day 20 保持未完成，进度为 19 / 42。

### Day 20｜2026-10-24｜删除接入、失败恢复与本地验收（待页面确认）

- **理解与纠正**：学习者起初认为保存失败后磁盘记录也已移除；讲解后明确，修改 Python 字典只影响内存，成功保存 JSON 才改变磁盘记录。因此采用先保存退出检索与待清理事项、再清理文件的顺序。
- **正式实现**：复用学习者的记录移除和路径定位函数，加入路径归属、标识及目录联接检查；清理项目内的 PDF 副本与该文档全部索引版本，保留公共原始 PDF。待办随项目 JSON 保存，部分清理或末次保存失败后可重试；相同文档清理完成前禁止重新导入。
- **页面与导入**：增加文档管理、删除与重试入口。删除后修正所选标识、清空当前会话和输入、清除范围索引缓存并重新运行页面，模型缓存保留。新导入在写文件前登记待办，成功时转为正式文档，失败时清理本次残留，空项目也能处理待办。
- **文件检查**：新增 `tests/test_document_deletion.py`，10 项检查通过，所有写入与删除均位于临时工作区。覆盖保存失败、部分清理、最后保存失败、公共原件、非法路径、重复与未知记录、导入待办、缺失文件和真实目录联接；项目内其他文档、其他项目及素材不受影响。
- **页面检查**：8 组 AppTest 检查通过，使用真实页面、PDF 与已保存索引的隔离副本，验证 52 → 51、重新导入后 52、选择与会话同步、缓存复用、失败及重试、空项目和公共原件保留。3 次模拟回答、5 次向量回放，真实推理与 DeepSeek 调用均为 0；实际工作区 23 个数据文件路径和哈希一致。
- **边界与保留**：不自动清扫没有待办的历史残留；当前为本机串行操作，未实现并发写入协调或跨文件事务。没有新增依赖、提交 Git、修改既有评估或删除约定保留的练习文件；所有实际论文与测试素材保留给学习者操作。
- **当前状态**：代码与自动检查完成，剩一次学习者页面确认：在 test 项目全部文档范围删除测试 PDF，核对 1 份文档、51 个片段及空会话。Day 20 暂不勾选，进度保持 19 / 42。

### Day 20｜2026-10-24｜实际删除确认与收尾（已完成）

- **学习者回传**：页面显示“当前项目：test · 检索文档：1 / 1 · 可检索片段：51”，并明确输入框已清空，符合删除测试 PDF 后的预期。
- **磁盘核对**：助手只读检查当前项目 JSON，确认仅保留原论文、无 `pending_cleanup`；测试 PDF 的上传副本与文档索引目录均已删除。原论文 PDF、索引与 51 条片段记录仍在，公共原始 PDF 和 `data/fixtures/day18_scope_note.pdf` 素材原件保留。
- **验证边界**：本次核对没有重新执行删除、自动测试、向量推理或 DeepSeek 调用；聊天与候选清理、失败恢复等行为沿用前一轮已通过的自动检查，不把本次用户回传扩写为新的模型问答结果。
- **文档维护**：更新当前能力、任务勾选、进度与 Day 20 验收说明，保留先前各阶段日志。用户通过页面产生的项目记录修改与测试副本删除保持原样；没有提交 Git，也没有删除约定保留的练习文件。
- **当前状态**：Day 20 已完成，进度为 20 / 42，实际用时未记录。今天无需追加测试；下一次从 Day 21：项目管理阶段验收开始。

### Day 21｜2026-10-24｜项目管理阶段验收（开始）

- **现状检查**：当前 HEAD 仍为 `0d27611`，Day 19–20 的代码、README 与用户页面删除产生的数据变更尚未提交，继续保留。三个项目的文档数分别为 1、1、0，均无待清理记录；Day 20 已有 10 项文件检查和 8 组页面检查，本次不重复执行这些检查。
- **当天安排**：学习用断言检查每条候选的项目与所选文档归属；准备明确标注为人工测试资料的同名、相近内容样例；固定验证项目切换、文档范围切换、空选择和删除后的隔离四类场景，并保存结果。断言属于验收代码，应用中的输入校验继续使用明确的异常分支。
- **第一步练习**：由学习者新建 `tests/test_project_scope.py`，实现 `assert_results_in_scope(results, project_id, selected_document_ids)`，补齐项目匹配与文档属于所选集合两个断言。用合法候选、其他项目的同标识候选、当前项目内未选文档的候选三个模拟场景验证检查器本身。
- **判断边界**：归属检查针对已返回的每条候选，不能仅依据答案正确或相似度高判断范围正确；空结果没有越界片段，但也不能据此认为检索成功。非空结果、回答内容及证据支持在对应场景另行检查。
- **当前状态**：本次只读检查代码和项目记录，更新学习安排；尚未创建验收脚本、生成新样例、导入文档或调用模型。Day 21 保持未完成，进度为 20 / 42，原有练习文件和测试素材继续保留。

### Day 21｜2026-10-24｜归属断言通过，准备跨项目资料（进行中）

- **学习者完成**：使用 `chunk['project_id'] == project_id` 与 `chunk['document_id'] in selected_document_ids` 完成两个断言，回传合法候选通过、其他项目和未选文档分别被拒绝。助手读取文件确认实现，只整理空行，不重复执行这三个模拟场景。
- **样例准备**：新增 `data/fixtures/day21/project_a/scope_note.pdf`、`project_b/scope_note.pdf` 与 `manifest.json`。PDF 文件名相同，正文只在 `PROJECT-A-217` 与 `PROJECT-B-963` 校验码上不同，明确标注人工测试用途。用现有解析与分块函数确认各为 1 页、367 字符和 1 个片段；生成前已有 20 个数据文件的内容保持一致。
- **下一步练习**：学习者在页面创建 Day21 A 与 Day21 B，分别导入对应素材，各问一次相同的校验码问题，回传回答、原文依据和候选标识。切回 A 时仅检查旧聊天未恢复，不再追加问题。
- **判断依据**：同时核对回答值和候选的实际项目归属；文件名相同、文档标识相同或回答看似合理，都不能替代项目与所选文档的检查。资料生成不计作检索或问答验收。
- **当前状态**：归属断言练习完成，跨项目资料已准备；尚未创建新项目、导入资料、生成向量或调用 DeepSeek。四类场景的完整验收待继续，Day 21 未完成，进度为 20 / 42。

### Day 21｜2026-10-24｜实际问答、四组验收与收尾（已完成）

- **学习者回传**：Day21 A 回答 `PROJECT-A-217`，Day21 B 回答 `PROJECT-B-963`，均引用对应 PDF 的完整校验码原句，并确认切换后聊天“全清空了”。用户未提供候选片段标识，未将其补写为人工观察。
- **实际资料核对**：读取新建项目记录及索引，确认各项目仅登记对应 PDF，上传副本摘要与素材一致，均为 1 条 384 维向量；唯一片段的项目、文档、来源和页码匹配。该结果与用户回传分开保存，不冒充实际 API 请求或页面候选日志。
- **验收脚本**：保留学习者的归属断言，扩展 `tests/test_project_scope.py` 为四组可重复运行的检查。临时项目使用真实 PDF、解析、分块、FAISS 和页面，通过控制向量与模拟回答验证项目切换、单选及多选、空选择和删除隔离。
- **关键覆盖**：未选中文档的控制相似度更高时仍不能进入候选；另一项目存在同一文档标识时仍检查项目归属；删除一个项目中的副本不会影响另一项目的同内容文件。范围改变清空历史，空选择停止请求，等价选择顺序保留会话，删除后模型缓存复用。
- **检查结果**：四组全部通过，共 11 次控制向量查询和 11 次模拟回答，真实推理与助手发起的 DeepSeek 调用均为 0。测试前已有 31 个数据文件哈希保持一致，临时目录已清理；没有再次运行 Day 20 的既有检查。
- **保存与边界**：新增 `data/evaluations/day21_project_management.json`，分别记录两条人工回答、会话清空回传、磁盘来源和自动用例；明确缺少用户候选标识及控制向量的限制。README 更新任务状态、复现命令与进度，实际项目和素材继续保留，没有新增依赖、改动应用逻辑或提交 Git。
- **当前状态**：Day 21 已完成，进度为 21 / 42，实际用时未记录。项目与范围管理阶段收尾，今天无需追加测试；下一次从 Day 22：BM25 关键词检索开始。

### Day 22｜2026-10-25｜BM25 关键词检索（开始）

- **现状检查**：学习者已提交 Day 19–21 的改动，当前 HEAD 为 `2a08fc1`，开始时工作区干净。正式检索仍为 Dense，返回 `rank`、`score`、`chunk`；项目虚拟环境尚未安装 `rank_bm25`。
- **当天安排**：先将英文论文文本整理为词项列表；再学习词频、词项稀有程度与片段长度如何影响 BM25 排名，实现独立基线；最后在同一论文范围、同一组问题上比较 Dense 与 BM25 候选。双路统一接口与页面接入按后续学习步骤推进。
- **第一步练习**：由学习者创建 `bm25_retriever.py`，实现 `tokenize_bm25(text)`，将文本转为小写，再用 `re.findall(r"[a-z0-9]+", normalized_text)` 提取连续英文字母和数字，返回字符串列表。在 `practice_bm25.py` 中检查大小写、标点与型号中的数字；本步仅使用 Python 标准库。
- **处理边界**：这是英文与数字词项的起步规则，中文会被忽略，不做翻译、词形还原或同义词扩展；`GPU` 与 `GPUs` 不会自动合并。问题与片段必须使用同一预处理函数，纯中文输入在本规则下没有词项，需要在后续检索函数中明确处理，不能把任意候选当作匹配结果。
- **当前状态**：已检查代码与依赖、更新学习安排；尚未创建 BM25 模块、安装依赖或运行检索。Day 22 保持未完成，进度仍为 21 / 42，既有评估、数据与约定保留的练习文件继续保留。

### Day 22｜2026-10-25｜分词通过，准备 BM25 打分（进行中）

- **学习者完成**：使用 `text.lower()` 完成小写转换。两条大小写不同的型号文本均得到 `['nvidia', 'p100', 'gpus']`，第三条得到 `['wmt', '2014', 'english', 'to', 'german']`，结果类型均为列表。助手读取两个实际文件确认实现正确，没有重复运行分词练习。
- **依赖准备**：助手在项目 `.venv` 安装 `rank-bm25==0.2.2`，只在依赖清单中新增这一条。库导入及 `pip check` 通过，已有 NumPy 满足依赖，无需新增模型权重。
- **本步要点**：BM25 结合查询词在片段中的出现次数、在片段集合中的稀有程度和片段长度计算分数。`BM25Okapi` 接收每条文本的词项列表，`get_scores()` 返回按原文本顺序对应的分数，尚未自动排序；分数不是概率，也不能与 Dense 原始分数直接相加。
- **下一步练习**：由学习者将 `practice_bm25.py` 改为三条英文短文本的打分示例，分别补齐片段与查询的分词调用。查询为 `NVIDIA P100`，打印查询词项和每条原文的 BM25 分数；先观察分数，再进入排序与真实论文检索。
- **当前状态**：分词函数和依赖准备完成，尚未运行 BM25 打分、接入页面或调用模型。Day 22 未完成，进度保持 21 / 42；既有问答评估与项目数据保持原样。

### Day 22｜2026-10-25｜短文本打分通过，准备候选排名（进行中）

- **学习者完成**：对片段使用 `tokenize_bm25(text)`、对查询使用 `tokenize_bm25(question)`。查询 `NVIDIA P100` 得到 `['nvidia', 'p100']`，三条文本依次返回约 `1.0217`、`0.0`、`0.0`；查询词仅出现在第一条硬件说明中，符合本组样例预期。
- **助手核对**：读取实际练习与分词函数确认调用正确，未重复执行打分。当前打印按 `corpus` 原始顺序输出，还没有对结果排序；本例第一条恰好也是最高分，不能仅凭此验证排序。
- **下一步练习**：学习者把查询改成 `encoder`，按分数从高到低排序原文位置，再使用位置取得原文与对应分数，打印排名、原始位置、原文和分数。保留三条用于观察排序，理解排名从 1 开始、列表位置从 0 开始，以及位置与来源记录必须保持对应。
- **当前状态**：短文本分词与打分已完成，排名展示、正式 BM25 检索函数及真实论文候选对比待完成。Day 22 仍未完成，进度为 21 / 42；没有改动应用、项目数据或既有评估。

### Day 22｜2026-10-25｜排名通过，准备真实论文候选（进行中）

- **学习者完成**：通过 `corpus[position]` 取回排序位置对应的原文。查询 `encoder` 时，原始位置 1 的编码器说明排第 1，分数约 `0.5108`；位置 0、2 分别排第 2、3，分数均为 0。助手读取文件确认位置映射正确，未重复执行这组打分。
- **函数骨架**：助手在 `bm25_retriever.py` 中准备 `retrieve_bm25(question, chunks, top_k=3)`，复用已学的分词、BM25 打分和按位置排序。验证空问题与 `top_k`；没有可用查询词、没有片段或没有词项重合时返回空列表；以词项重合筛选候选，不把正分作为通用的匹配判断。
- **范围与来源**：练习脚本改为读取 Demo project 的已登记文档，通过 `load_scope_index()` 取得经范围校验的完整片段，输出候选标识、来源、页码和原文。BM25 用本次传入的全部片段统计词频与长度，返回最多 `top_k` 条有词项重合的记录。
- **留给学习者**：补齐每条候选的 `rank`、`score`、`chunk` 字段并加入结果列表。函数内暂保留明确的 `NotImplementedError`，须完成此处后再运行；完整片段字典必须保留，供后续追溯来源。
- **准备检查**：两个 Python 文件语法检查通过，已有索引可读取，共 51 个片段、51 条向量。本次未执行尚未完成的检索函数，没有生成向量或调用模型。
- **当前状态**：短文本排名已完成，函数候选组装和真实论文检索待学习者完成，随后比较同一组问题的 Dense 与 BM25 候选。Day 22 未完成，进度保持 21 / 42；页面仍使用原 Dense 流程。

### Day 22｜2026-10-25｜真实论文、同范围对比与收尾（已完成）

- **学习者完成**：将 `rank`、`float(scores[position])`、`chunks[position]` 保存为候选字段。真实论文的 `NVIDIA P100` 查询返回 3 条记录，标识依次为 `p7:c3`、`p7:c4`、`p2:c3`，分数约 4.9147、4.4716、1.9785；原文、页码及来源完整。助手读取函数确认实现，只更新已过时的未完成说明。
- **边界验证**：新增 `tests/test_bm25_retriever.py`，四项检查全部通过。排序后来源保持对应，输入不被改写，没有匹配时不凑满候选；空白问题和非法数量被拒绝；确有词项匹配的零分或负分记录不会被误删。
- **对比实现**：新增可提交的 `compare_retrievers.py`，固定 Demo 范围和三条查询，两路读取同一批 51 个片段，保存独立报告；已有结果文件拒绝覆盖。查询分别覆盖硬件型号、中文层数与英文耗电量，只做检索，不生成回答。
- **实测结果**：硬件型号的两路第一名分别为 `p7:c4` 与 `p7:c3`，均含硬件说明；中文层数问题 Dense 第一名为 `p2:c7`，BM25 为空；英文耗电量问题两路各有 3 条候选，助手核对原文后未见实际 kWh 数值。词项匹配、语义相关与可回答性分别判断。
- **执行与保存**：实际运行 3 次本地 CPU 向量查询和 3 次 BM25 查询，DeepSeek 调用为 0，原有 32 个数据文件哈希保持一致。新增 `data/evaluations/day22_retrieval_comparison.json` 保存参数、输入摘要、依赖版本及完整候选，不修改或混入既有评估。模型只使用本地缓存，本次没有重新构建索引。
- **当前状态**：Day 22 已完成，进度为 22 / 42，实际学习用时未记录。README 更新能力、任务勾选、文件结构、复现命令及观察结论；既有练习文件保留，没有自动提交 Git。页面仍使用 Dense，下一次从 Day 23：统一双路召回接口开始，今天无需再追加测试。

### Day 23｜2026-10-25｜统一双路召回接口（开始）

- **现状检查**：当前 HEAD 仍为 `2a08fc1`，Day 22 的模块、依赖、对比脚本、报告与学习记录尚未提交，继续保留。Dense 与 BM25 已分别返回 `rank`、`score`、`chunk`，对比脚本分别调用两者；页面仍只调用 Dense。
- **当天安排**：集中两路调用，复用同一份已按项目和所选文档范围加载的索引与片段；按片段标识组织去重候选，保留各路的原始排名和分数；完成接入与范围验证，检查单路无结果、重复候选和未选中文档等情况。RRF 排名融合仍安排在 Day 24。
- **第一步练习**：由学习者新建 `dual_retriever.py`，实现 `retrieve_dual(question, model, index, metadata, top_k=3)`。Dense 接收配套的 `index` 和 `metadata`，BM25 从同一个 `metadata` 取得片段列表，结果以 `dense`、`bm25` 两个键分别返回。本步补齐 BM25 的片段参数与返回字典对应值。
- **接口约定**：范围由调用前的 `load_scope_index()` 确定，统一入口复用其返回值，不重新读取其他项目或所有文档。两路分数保持各自含义；同一片段在两路出现属于重复候选，需要按 `chunk_id` 识别，不能因分数不同当作两个片段，也不直接相加原始分数。
- **当前状态**：只读核对代码并更新学习安排，尚未创建统一入口、修改页面或运行新检索。Day 23 保持未完成，进度为 22 / 42；既有学习记录、评估、项目数据与练习文件继续保留。

### Day 23｜2026-10-25｜统一入口通过，准备候选去重（进行中）

- **学习者完成**：在 `dual_retriever.py` 中实现 `retrieve_dual()`，BM25 的 `chunks` 使用同一个 `metadata['chunks']`，返回字典中的 `bm25` 对应 `bm25_results`。Dense 继续接收配套索引和同一份元数据，两路保持各自列表。
- **接口核对**：助手读取实际代码，并以 Demo 的真实索引、片段及一条已存向量回放调用统一入口。两路各返回 3 条记录，项目、所选文档及完整来源与输入记录一致；实际向量模型推理和 DeepSeek 调用均为 0。该检查用于接口与来源对应，不作为新的语义检索质量观察。
- **下一步练习**：学习者在同一模块追加 `merge_candidates(dual_results)`，以 `chunk_id` 为键去重，保留完整 `chunk`，在 `retrievals` 中分别记录 `dense`、`bm25` 的原始排名和分数。同一标识对应不同内容或来源时拒绝合并；单路重复标识也应明确报错，避免覆盖排名。
- **练习准备**：助手新增本地 `practice_dual_retrieval.py`，读取 Day 22 报告中 `NVIDIA P100` 的真实两路候选，打印去重数量、来源和各路排名。脚本依赖尚待学习者编写的去重函数，因此本轮未运行，也不需要重新调用模型。
- **判断边界**：去重后的列表按片段首次出现顺序排列，尚未计算综合排名；保留各路分数及排名供 Day 24 的 RRF 使用。文字相同但标识不同的片段仍分别保留其来源。
- **当前状态**：统一入口已完成，候选去重、页面接入和完整范围验收待继续。Day 23 未完成，进度为 22 / 42；应用、项目数据和旧评估均未修改。

### Day 23｜2026-10-25｜候选合并、页面接入与自动验收（待页面确认）

- **学习者完成**：使用 `result['chunk']['chunk_id']` 提取片段标识，以 `method` 保存各路排名与分数。回传两路各 3 条、去重后 4 条；共同片段保留两份排名，独有片段保留所属单路信息，符合预期。
- **正式接入**：保留学习者函数，补充配套数量、唯一标识及返回记录归属检查。页面通过统一入口检索并合并候选，在成功问答中保存两路结果和去重记录；折叠区展示数量、来源、原文及各路分数，空 BM25 有明确提示。参考资料继续使用 Dense，RRF 排序按 Day 24 推进。
- **自动检查**：新增 `tests/test_dual_retriever.py`，五项接口与合并检查、七组页面范围检查全部通过，运行用时 22.791 秒。范围切换与删除后两路来源均正确；BM25 独有候选可显示但不进入当前参考资料；注入越界候选时拦截模型请求并保留旧历史。
- **执行边界**：页面使用临时项目、真实 PDF、FAISS 和 BM25，配合控制向量及模拟回答，未执行真实模型推理或 DeepSeek 调用。工作区原有 33 个数据文件未改变，临时目录已清理；没有重复运行 Day 22 的真实向量对比或其他未受影响的检查。
- **记录保存**：新增 `data/evaluations/day23_dual_retrieval.json`，区分学习者的旧结果回放、自动检查摘要与待完成的页面确认；记录验证代码摘要和测试边界，不将模拟回答视为实际问答质量。README 更新能力、数据流和复现步骤，原学习日志及练习文件保留，没有自动提交 Git。
- **当前状态**：代码与自动检查已完成，Day 23 只剩一次学习者页面确认：重启后在 Demo project 查询 `NVIDIA P100`，核对两路数量和独立排名。任务暂不勾选，进度保持 22 / 42，无需重新执行其余场景。

### Day 23｜2026-10-25｜实际页面确认与收尾（已完成）

- **学习者回传**：页面显示 Dense 与 BM25 各 3 条、去重后 4 个片段。首条候选为 `demo_project:attention_is_all_you_need_neurips2017:s800:o100:p7:c4`，来源为原论文 PDF、第 7 页；Dense 第 1、分数 0.8396，BM25 第 2、分数 4.4716，两路信息均保留。
- **页面说明核对**：回传包含“本轮回答使用 Dense 候选作为参考资料”以及按首次出现顺序展示的说明，符合当前两路展示、尚未融合排序的阶段设计。
- **记录维护**：将 `day23_dual_retrieval.json` 的页面确认与整体状态更新为通过和完成，保存用户提供的数量、说明及首条候选，与旧结果回放、自动检查分开记录。用户未提供回答正文或原始 API 响应，不据此新增回答正确性结论。
- **当前状态**：Day 23 已完成，进度为 23 / 42，实际学习用时未记录。本轮仅维护报告和 README，没有重跑模型或自动测试，没有修改应用、项目数据或提交 Git；今天无需追加验收，下一次从 Day 24：实现 RRF 排名融合开始。

### Day 24｜2026-10-25｜RRF 排名融合（开始）

- **现状检查**：当前 HEAD 仍为 `2a08fc1`，Day 22–23 的改动尚未提交，继续保留。两路统一检索和按标识去重已完成，每个合并候选包含完整 `chunk` 及各路 `retrievals`；页面按首次出现顺序展示，回答参考资料仍取 Dense 候选。
- **当天安排**：先手算小型排名表，理解每一路的排名贡献；再实现 RRF 打分、排序和最终数量限制，保留片段来源及各路排名；最后把融合候选用于回答参考资料，并检查单路为空、去重、来源与历史展示等行为。
- **公式依据**：参考 [RRF 官方说明](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)，每个召回该片段的检索方式贡献 `1 / (rank_constant + rank)`，各路贡献相加。名次从 1 开始，未返回该片段的检索方式贡献 0；不直接使用 Dense 或 BM25 的原始分数。
- **第一步练习**：为便于手算，暂设 `rank_constant=1`。Dense 排名为 A 第 1、B 第 2；BM25 排名为 B 第 1、C 第 2。学习者计算 A、B、C 的融合分数并给出从高到低的顺序，作答前不提供结果。
- **参数说明**：`rank_constant` 是控制排名差异影响的常数，不是候选数量 `top_k`。后续正式基线计划从常见的 60 开始，该值尚未在本项目中调优；两路权重暂按相同处理。
- **当前状态**：本次只读核对代码、查阅公式并更新学习安排，尚未新增 RRF 模块、修改页面或运行检索。Day 24 保持未完成，进度为 23 / 42；既有报告、项目数据与练习文件继续保留。

### Day 24｜2026-10-25｜手算纠正，准备 RRF 打分函数（进行中）

- **手算与纠正**：学习者首次给出 B=0.75、C=0.5、A=0.25。讲解后明确，常数为 1 时 A 为 `1/2`、B 为 `1/3 + 1/2`、C 为 `1/3`，顺序是 B、A、C；只对实际召回该片段的检索方式累加贡献。
- **学习者复验**：保持排名不变、常数改为 2 后，正确写出 B 的计算式 `1/(2+2) + 1/(2+1)`，能够把常数与各路排名分别代入。
- **下一步练习**：由学习者新建 `rank_fusion.py`，实现 `calculate_rrf_score(retrievals, rank_constant=60)`，遍历各路信息，从 `rank` 取得名次，补齐累计贡献的表达式，返回浮点总分。常数和名次均要求为正整数，原始 `score` 不参与计算。
- **练习准备**：助手新增本地 `practice_rrf.py`，使用 A、B、C 的小型排名表，分别计算常数为 1 和 2 的分数并打印排序。脚本依赖尚待学习者创建的打分函数，本轮仅检查语法，没有执行脚本。
- **当前状态**：手算中的参数代入已纠正，RRF 打分函数、正式候选融合及页面接入仍待完成。Day 24 未完成，进度为 23 / 42；本轮没有修改应用或旧报告，没有调用模型。

### Day 24｜2026-10-25｜RRF 打分、排序与页面接入（待页面确认）

- **学习者完成**：使用 `total += 1.0 / (rank_constant + rank)` 实现逐路累加。常数 1 与 2 的 A、B、C 分数均符合手算，两组排序都是 B、A、C；助手核对实际文件后保留核心表达式。
- **正式实现**：补齐打分输入校验和 `fuse_rrf()`，保留完整片段与各路信息，按 RRF 分数排序、同分按片段标识固定顺序，最后限制数量并重新编号。页面每路最多召回 3 条，常数 60，最终最多 3 条，参考资料已切换为融合结果。
- **历史与展示**：新消息保存参考资料方式、融合参数和最终片段，展示 RRF 排名与分数；旧 Dense 消息仍显示原规则。历史传入模型时继续只包含角色与正文，RRF 元数据保留在页面记录中。
- **真实候选回放**：复用 Day 22 的固定候选，硬件查询的融合前三名为 `p7:c4`、`p2:c3`、`p7:c3`，分数约为 0.032522、0.032002、0.016393。BM25 独有片段进入最终列表，原 Dense 参考文献候选退出；没有为此重新运行向量检索或调用 DeepSeek。
- **验证与修正**：八项融合检查、九组页面检查首次运行 16 项通过、1 项失败。失败源于页面测试仅用四条候选重建 BM25，导致语料统计变化；改为完整 51 片段后，只重跑该项并通过，首次用时 31.034 秒、复验 1.319 秒。17 项现已完成验证，没有重复运行其余通过项。
- **执行与记录**：自动页面检查使用控制向量与模拟回答，真实模型推理和 DeepSeek 调用为 0；原有 34 个数据文件不变，临时目录已清理。新增 `day24_rrf.json` 区分手算、旧候选回放和两次检查过程，保留测试边界；Day 23 的可复用页面断言更新为当前 RRF 规则，旧报告不改写。
- **当前状态**：代码与自动检查已完成，Day 24 只剩一次学习者实际页面确认。重启后在 Demo project 查询 `NVIDIA P100`，核对使用 RRF 的说明、三个融合片段及分数；任务暂不勾选，进度仍为 23 / 42。本轮没有新增依赖、自动提交 Git或删除约定保留的练习文件。

### Day 24｜2026-10-25｜实际页面确认与收尾（已完成）

- **学习者回传**：页面说明本轮回答使用 RRF 融合后的 3 个片段作为参考资料，排名常数为 60，按融合排名展示并保留各路原始排名与分数。三个 RRF 分数依次为 0.032522、0.032002、0.016393，与候选回放的六位小数结果一致。
- **记录维护**：将 `day24_rrf.json` 的页面确认与整体状态更新为通过和完成，保存回传说明、常数、数量及分数，与候选回放和自动检查分开记录。本次未回传片段标识、两路数量、回答正文或原始 API 响应，不补造这些观察，也不新增回答正确性结论。
- **当前状态**：Day 24 已完成，进度为 24 / 42，实际学习用时未记录。本轮仅更新报告和 README，没有重跑模型或自动测试，没有修改应用、项目数据或提交 Git；约定保留的练习文件继续保留。
- **下一次学习**：Day 25，加入 Rerank 精排，学习根据问题与候选原文计算相关性，观察精排前后的顺序与额外耗时。

### Day 25｜2026-10-25｜Rerank 精排与输入对（开始）

- **现状检查**：页面仍使用 Day 24 的 RRF 结果作为参考资料，每路召回最多 3 条、融合后最多 3 条、排名常数为 60；已有改动继续保留。当前 `calculate_rrf_score()` 只使用各路排名，尚未接入精排模型。
- **当天安排**：先理解 CrossEncoder 如何同时读取问题与候选原文，并组织输入对；再运行模型，观察分数和排序；最后将精排接入当前范围内的融合候选，保留 RRF 信息及来源，记录耗时并验证页面。待精排的候选数量与最终交给回答模型的数量分别控制。
- **资料依据**：核对 [Sentence Transformers 精排流程](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html) 和 [CrossEncoder 使用说明](https://sbert.net/docs/cross_encoder/usage/usage.html)，并只读查看当前虚拟环境的 `cross_encoder/model.py` 接口。文本精排使用问题与片段的输入对，每对得到一个相关性分数；分数尺度由具体模型及输出设置决定，不能直接当作回答正确率。
- **第一步练习**：新增本地 `practice_rerank.py`，提供三个仅含练习所需字段的候选样例。学习者补齐 `pairs.append([..., ...])` 的两项，使每组输入包含同一个问题和当前候选原文，再回传补齐的代码、输入对数量及第一组输入。样例不冒充真实检索记录，当前步骤不加载模型。
- **当前状态**：Day 25 保持未完成，进度为 24 / 42；尚未选择或下载精排模型，也未修改应用、索引、依赖或旧报告。约定保留的练习文件继续保留，没有自动提交 Git。

### Day 25｜2026-10-25｜输入对通过，准备模型打分（进行中）

- **学习者完成**：补齐 `pairs.append([question, candidate["chunk"]["text"]])`，回传输入对数量为 3，第一组包含中文硬件问题与英文 GPU 原文，结构正确。助手核对本地文件后保留学习者写法。
- **模型选择**：本次使用支持中英文的 `BAAI/bge-reranker-base`，通过已安装的 `sentence-transformers` 调用，在 CPU 上进行小批量练习。参考 [官方模型卡](https://huggingface.co/BAAI/bge-reranker-base) 与 [BGE 模型说明](https://bge-model.com/bge/bge_reranker.html)；权重约 1.11 GB，首次运行需下载，后续使用本地缓存。本项目上的实际排序质量仍待观察。
- **调用准备**：在 `practice_rerank.py` 补充 CrossEncoder 初始化与 `predict(pairs)`，固定模型版本 `2cfc18c9415c912f9d8155881c133215df768a70`，使用 CPU、最大输入长度 512 token、批量 3、safetensors 权重。显式使用 Identity 保留原始相关性分数，打印数组类型、形状、各片段分数和本次打分耗时；下载加载计时与打分计时分开，首次打分不作为稳定性能基准。
- **下一步练习**：学习者运行脚本，回传分数类型、形状、三个片段的分数及打分耗时。当前输出保持输入顺序，待理解分数与片段的对应关系后再练习排序。相关性分数不能作为回答正确率或跨模型通用阈值。
- **验证边界**：本轮仅核对官方资料、远程模型版本和已安装接口，并检查练习文件语法；助手未下载权重或执行推理，尚无实际模型分数。正式应用、索引、依赖和旧报告保持原样，Day 25 未完成，进度仍为 24 / 42。

### Day 25｜2026-10-25｜模型打分通过，练习排序与数量限制（进行中）

- **缓存位置**：学习者考虑系统盘空间，在 CrossEncoder 参数中加入 `cache_folder="E:/hf-cache/hub"`，助手已核对本地代码。后续正式精排接入沿用这个缓存目录；模型仍需使用，先保留，旧 C 盘向量模型缓存未移动或删除。
- **学习者回传**：分数类型为 `numpy.ndarray`，形状为 `(3,)`，本次打分耗时 0.252 秒。`practice:a` 为 -1.5961，`practice:b` 为 -10.1963，`practice:c` 为 -10.1873；学习者正确判断包含训练硬件原文的 `practice:a` 得分最高。
- **结果理解**：三个输入对各得到一个相关性分数，`predict()` 返回顺序与输入一致，尚未排序。Identity 配置下负分属于正常输出；比较本次候选时按数值大小排序，不能把分数当成回答正确率。0.252 秒仅为三条短样例的一次打分耗时，不包含模型加载，也不代表真实论文检索的稳定端到端耗时。
- **下一步练习**：由学习者在脚本末尾将每个分数与对应 `chunk` 组成记录，补齐排序键和降序开关，排序后取前 2 条并打印重新编号的排名、片段标识和分数。这里的 2 只用于练习数量限制，不改正式页面参数；作答前不提供完整排序结果。
- **当前状态**：Day 25 尚未完成，进度保持 24 / 42。此次记录来自学习者回传，助手未重复执行模型推理，也未修改正式应用、索引、依赖或旧报告；排序函数封装、真实融合候选精排及页面验收仍待继续。

### Day 25｜2026-10-25｜排序通过，封装精排并准备真实候选对比（进行中）

- **学习者完成**：使用 `key=lambda item: item["rerank_score"]` 和 `reverse=True` 完成降序排序；前两条为 `practice:a`（-1.5961）和 `practice:c`（-10.1873），符合预期，助手已核对脚本中的实际写法。
- **函数封装**：新增 `reranker.py`，`load_reranker()` 使用既定 BGE 模型与版本；`rerank_candidates(question, candidates, model, top_k=3)` 接收按 RRF 排好的候选，沿用学习者的输入对与排序逻辑，校验分数形状、数量和有限值，排序后截取并重新编号。同分时沿用输入的 RRF 顺序，空候选直接返回，不调用模型。
- **来源与分数**：输出深复制原候选，完整保留 `chunk` 和各路 `retrievals`；原融合排名和分数另存为 `rrf_rank`、`rrf_score`，新排名写入 `rank`，精排分数写入 `rerank_score`，通用 `score` 与精排分数一致。这样能对比排序前后变化，且不会改坏原 RRF 列表。
- **逻辑验证**：`tests/test_reranker.py` 六项检查全部通过，覆盖排序后再限量、来源与两路信息保留、原记录不被修改、同分顺序、空输入、无效参数、异常分数及模型失败传播。使用模拟分数，未加载模型；可通过 `./.venv/Scripts/python.exe -X utf8 -m unittest discover -s tests -p test_reranker.py -v` 复现。
- **真实候选练习准备**：新增本地 `practice_rerank_candidates.py`，读取 Day 22 的 `NVIDIA P100` 双路记录，去重后将四条全部交给 RRF 和精排，再取精排前三条，打印前后排名、两种分数、来源、页码与耗时。模型只从 `E:/hf-cache/hub` 已有缓存加载，`local_files_only=True`；助手已确认固定版本的配置和权重文件存在。
- **验证边界与下一步**：本轮只执行六项逻辑检查和脚本语法检查，真实候选精排尚待学习者运行回传。候选来自旧报告，不算新增召回实验；没有调用 DeepSeek、修改页面或项目数据，旧报告和原练习继续保留。Day 25 未完成，进度仍为 24 / 42。

### Day 25｜2026-10-25｜真实候选回传、页面接入与自动验收（待页面确认）

- **学习者回传**：精排顺序为 `p7:c4`、`p7:c3`、`p2:c3`，原 RRF 排名依次为 1、3、2；精排分数分别为 0.1071、-0.697、-3.4262，来源为 Demo 原论文的第 7、7、2 页。学习者正确指出后两个片段交换名次；助手已询问本次脚本总耗时，当前尚未回传。
- **正式接入**：页面两路各召回最多 3 条，RRF 保留最多 6 条供精排，最终最多 3 条进入参考资料。新消息保存精排配置、耗时及精排前后的结果，页面显示对应排名和分数；旧 Dense 与 RRF 消息继续使用原有说明。
- **缓存配置**：新增 `config.example.toml` 与已忽略的本机 `config.local.toml`，本机指向学习者已使用的 `E:/hf-cache/hub`；未配置时使用 Hugging Face 默认缓存。页面精排模型独立缓存，使用 `local_files_only=True`，仅有候选时加载；已有 C 盘向量缓存保持原样。
- **异常与历史**：模型缓存缺失、精排失败或分数校验失败时展示对应提示，停止回答生成并保留历史；不会自动切回 RRF。空范围或无候选时不加载精排模型，普通重运行和清空会话不重复打分，删除与范围切换继续复用模型。
- **自动验收**：七项逻辑与配置检查、十四组页面检查共 21 项一次通过，用时 48.456 秒。控制分数的专门场景将 RRF 第 4 名提升到最终列表，验证参考资料确实按精排选择；这不是模型质量结论。测试使用临时项目与模拟模型，助手真实推理和 DeepSeek 调用均为 0，原有 35 个数据文件不变。
- **记录与剩余步骤**：新增 `day25_rerank.json` 保存回传和测试观察，原学习日志及旧评估继续保留。本日任务暂不勾选，进度保持 24 / 42；只剩学习者在 Demo project 查询一次 `NVIDIA P100`，确认精排说明、三个片段的新旧排名及页面耗时。本轮没有新增依赖、自动提交 Git或删除练习文件。

### Day 25｜2026-10-25｜实际页面确认与收尾（已完成）

- **学习者回传**：页面显示参与精排 4 条、最终参考 3 条、精排耗时 1.212 秒（不含模型加载）。最终片段为 `p7:c4`、`p7:c3`、`p2:c3`，新排名为 1、2、3，原 RRF 排名为 1、3、2；来源均为 Demo 原论文，PDF 页码依次为 7、7、2。
- **分数核对**：页面 RRF 分数依次为 0.032522、0.016393、0.032002，精排分数为 0.1071、-0.6970、-3.4262，与练习回传的顺序及显示精度一致。精排接入及来源展示验收通过；未提供回答正文或原始 API 响应，不据此追加回答质量结论。
- **记录维护**：将 `day25_rerank.json` 的页面确认与整体状态改为通过和完成，保存实际回传的数量、完整标识、来源、页码、两种排名与分数及页面耗时。早先脚本的未回传耗时保留为空，不以这次页面耗时替代，也无需补做该次运行。
- **当前状态**：Day 25 已完成，进度为 25 / 42，实际学习用时未记录。本轮仅维护报告和 README，没有重跑模型或自动测试，没有修改应用、项目数据或提交 Git；既有练习文件与 E 盘模型缓存继续保留。
- **下一次学习**：Day 26，观察完整检索流程，记录召回、融合和精排各阶段的候选、排名与耗时，追踪最终参考片段的来源。

### Day 26｜2026-10-25｜检索阶段记录与计时（开始）

- **现状检查**：当前页面保存双路结果、去重候选、RRF 候选与精排最终结果，已展示精排前后排名和单独的精排耗时；Dense、BM25、去重及 RRF 尚无独立耗时记录。精排函数目前仅返回最终截取结果，完整精排候选的打分与未入选状态留待本日补充。
- **当天安排**：先为一个阶段记录输入数量、输出数量、结果快照和耗时；再逐步记录各阶段，并按 `chunk_id` 汇总片段经过的路径；最后接入页面，观察入选及未入选候选。保持当前检索参数，区分模型加载、检索处理和回答生成的计时范围。
- **第一步练习**：新增本地 `practice_retrieval_trace.py`，读取 Day 22 的固定关键词候选，只对当前 `merge_candidates()` 调用计时。学习者补齐 `elapsed_seconds` 和 `output_count` 两处，再回传代码及阶段输出。输入数量统计两路返回的记录总数，输出数量统计去重后的片段；结果快照保留完整来源与各路排名。
- **计时依据**：使用 [Python 3.12 的 perf_counter](https://docs.python.org/3.12/library/time.html#time.perf_counter)，取两次计时值的差得到经过的秒数。当前计时不包含文件读取、结果快照复制、模型加载或其他检索阶段，不从旧报告推算历史耗时。
- **当前状态**：Day 26 未完成，进度保持 25 / 42。本轮仅准备练习并检查语法，未执行模型或新增检索实验，应用、原有索引、依赖与旧报告未修改；E 盘模型缓存和约定保留的练习文件继续保留，没有提交 Git。

### Day 26｜2026-10-25｜阶段记录通过，关联同一片段的排名（进行中）

- **学习者完成**：使用 `elapsed_seconds = ended - started` 和 `"output_count": len(merged)`，回传 `NVIDIA P100` 的去重输入记录 6 条、输出片段 4 条、快照 4 条，本次去重耗时为 0.000011 秒；助手核对实际文件后保留写法。该耗时仅属于这次去重调用，不代表完整检索耗时。
- **下一步练习**：在 `practice_retrieval_trace.py` 末尾补充 RRF 回放与排名路径表。学习者从 RRF 记录中取完整 `chunk_id` 建立映射，再用当前候选的同一标识查找 RRF 记录；表中保留来源和 PDF 页码，并展示 Dense、BM25、RRF 三种排名。RRF 常数 60、最多保留 6 条，与当前页面配置一致。
- **身份与缺失值**：片段经过排序后名次和列表位置会变化，用完整 `chunk_id` 关联同一片段。某一路没有返回该片段时，其排名记为 `None`，不能把它解释为第 0 名，也不能据此判断该路未执行。
- **当前状态**：只准备了两处待补的映射练习并检查语法，尚未执行新的路径汇总或接入完整阶段记录；没有重新运行模型，也没有改动应用、项目数据或旧报告。Day 26 仍未完成，进度为 25 / 42。

### Day 26｜2026-10-25｜排名关联通过与完整流程接入（待页面确认）

- **学习者完成**：正确填写 `chunk_id = result["chunk"]["chunk_id"]` 与 `rrf_result = rrf_by_id.get(chunk_id)`。四个候选按去重原顺序为 `p7:c4`、`p2:c3`、`p10:c3`、`p7:c3`，Dense / BM25 / RRF 名次分别是 `1 / 2 / 1`、`2 / 3 / 2`、`3 / None / 4`、`None / 1 / 3`，与固定候选一致。缺失名次表示该阶段没有返回这个片段。
- **正式实现**：新增 `retrieval_trace.py`，把一次阶段调用的输入输出数量、耗时和独立结果快照集中记录；双路接口仍返回原有两路字典，范围校验通过后交付召回记录。页面记录 Dense、BM25、去重、RRF、精排、最终截取六个阶段，并以完整片段标识关联前后排名、分数、来源与入选状态。
- **精排与参考资料**：每轮仅调用一次精排预测，先保留全部精排结果，再选前三条构建实际参考资料。被截去的候选仍可在路径表与原始快照中查看；入选仅表示已发送给模型，不能据此认定支持回答或引用已通过事实校验。
- **计时与生命周期**：各阶段独立计时，不包括额外快照复制；资源获取和回答调用单独记录。追踪记录跟随成功的问答保存在会话中，普通重运行不重算；清空和范围切换沿用现有会话机制。检索、精排或回答失败时不追加半轮记录，旧消息没有追踪字段时不补造历史数据。
- **自动检查与修正**：首次 24 项用时 66.327 秒，11 项通过、13 项在测试读取表格时遇到 pandas / Arrow 类型错误，页面本身未抛异常。改为直接读取页面发送的 Arrow 内容后，16 组页面场景全部通过，用时 73.876 秒；结合此前通过的 8 项逻辑检查，24 个不同场景均已验证。包括完整四候选及未入选状态、改序后的实际参考资料、空匹配、范围与删除隔离、清空、缓存复用、越界拦截、精排及回答失败、旧 Dense / RRF / Rerank 历史兼容。
- **证据边界**：测试使用控制模型与模拟回答，真实推理及 DeepSeek 调用为 0，不产生新的模型质量或性能结论。原有 36 个数据文件不变，临时目录已清理；新增 `day26_retrieval_trace.json` 分别保存学习者回传、两次检查及受控完整快照，旧报告不改动。
- **剩余验收**：重启 Streamlit，在 Demo project 选择原论文并查询一次 `NVIDIA P100`，展开“查看本轮检索流程”。核对六个阶段的输入输出数量，回传未入选候选的排名路径，并说明入选与回答依据的区别。Day 26 暂不勾选，进度保持 25 / 42；本轮没有安装依赖、下载模型、删除练习文件或提交 Git，E 盘模型缓存继续保留。

### Day 26｜2026-10-25｜实际页面确认与收尾（已完成）

- **页面数量与耗时**：学习者回传 Dense `51 → 3`、BM25 `51 → 3`、去重 `6 → 4`、RRF `4 → 4`、精排 `4 → 4`、最终截取 `4 → 3`，耗时依次为 0.158026、0.006838、0.000015、0.000117、1.226635、0.000002 秒，六阶段数量全部符合预期。这是单次运行观测，不代表端到端总耗时或稳定性能。
- **完整候选路径**：四条记录均来自 Demo 原论文。按去重原顺序，`p7:c4` 的 Dense / BM25 / RRF / 精排名次为 `1 / 2 / 1 / 1`，`p2:c3` 为 `2 / 3 / 2 / 3`，`p10:c3` 为 `3 / None / 4 / 4`，`p7:c3` 为 `None / 1 / 3 / 2`。其中 `p10:c3` 显示“未进入精排前 K 条”，其他三条已入选，说明完整精排记录与最终参考资料的截取结果可以分别追踪。
- **理解核对**：学习者回答“不一定，只是代表语义方面有关联”，正确判断入选不保证支持回答。助手补充：入选表示经过当前排序与截取后被送入参考资料，不能只凭入选认定语义相关或事实支持，还需核对原文与具体结论。
- **记录与边界**：将 `day26_retrieval_trace.json` 更新为完成，保存实际回传的六组数量和耗时、四条完整片段标识、来源、页码、排名及入选状态；自动检查与早先练习数据保持原样。此次未回传原始追踪 JSON、候选分数、资源获取和回答耗时、回答正文或引用，不从旧运行补填，也不新增回答质量结论。
- **当前状态**：Day 26 已完成，进度为 26 / 42，实际学习用时未记录。本轮仅维护报告和 README，未修改程序、重跑测试或模型、删除练习文件或提交 Git；约定保留的练习及 E 盘缓存继续保留。
- **下一次学习**：Day 27，在固定问题与相同范围上对比 Dense、BM25、RRF 和 Rerank 的检索结果，学习 Recall@K、MRR，并观察改善与退化案例。

### Day 27｜2026-10-25｜检索对照实验与 Recall@K（开始）

- **现状核对**：`evaluation_questions.json` 包含 20 道固定题，15 道有答案、5 道资料不足，保存了标准答案、原文依据与页码，但尚无完整的相关片段标识集合。Day 26 已能保存各阶段候选、排名和耗时；Day 22 的三查询对比与 Day 14 回答评估继续保留，不把旧回答正确率当成新检索指标。
- **当天安排**：先用固定标注的小例子学习 Recall@K，再实现指标计算并学习首个命中排名与 MRR；随后为真实问题建立可核对的片段标注，在相同文档、分块、查询及 K 下比较 Dense、BM25、RRF、Rerank。仅执行检索对比，不需要为指标再次生成模型回答。
- **指标口径**：本日按唯一 `chunk_id` 统计片段。Recall@K 为前 K 条中命中的相关片段数，除以该问题标注的全部相关片段数；分母不是返回数量。真实实验应先明确相关性标准与标注覆盖范围，不能仅把某次系统返回的候选当作全部相关片段。无相关答案的题需单独记录，不把分母为零的情况直接算作正常召回率。
- **第一步练习**：假定已完整确认某问题只有 A、C 两个相关片段，其他片段均无关。方法甲的前三条为 `B → A → D`，方法乙为 `C → B → A`。学习者分别手算两者的 Recall@3，说明分母代表什么；先等待计算过程，再给反馈与代码练习，不提前填入答案。
- **参考依据**：核对 [Stanford《Introduction to Information Retrieval》的召回率定义](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-unranked-retrieval-sets-1.html)、[排名结果与前 K 条评估](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html)，以及 [Sentence Transformers 检索评估接口](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html) 中查询、语料、相关性标注和 Recall / MRR 参数的区分。
- **当前状态**：Day 27 未完成，进度保持 26 / 42。本轮只读取现有代码与题集并更新 README，尚未生成新的实验结果或标注；没有运行模型、改动应用与依赖、覆盖旧报告、删除练习文件或提交 Git。

### Day 27｜2026-10-25｜Recall 手算纠正与函数练习（进行中）

- **学习者回传与讲解**：首次回答“4，5，知识库中的片段”。助手说明 Recall 是 0～1 的比例，分母应为知识库中与当前问题相关的全部片段数量，并用方法甲演示命中 A、计算 `1 ÷ 2 = 0.5`。随后学习者回答“1”，按方法乙的最终 Recall@3 理解是正确的；补充区分其命中数量为 2，最终比例为 `2 ÷ 2 = 1`。
- **代码练习**：新增本地 `practice_retrieval_metrics.py`，准备 `calculate_recall_at_k(retrieved_ids, relevant_ids, k=3)`。先按排名截取前 K 条，再收集其中命中的不同相关片段，学习者填写命中数量与 Recall 比例两处表达式。`set` 用于避免重复片段被重复统计；相关集合为空时返回 `None` 表示本指标不适用，非法 K 提示错误。
- **当前状态**：脚本保留两处 `None` 等待填写，目前只检查语法，尚未执行完成后的指标函数。没有生成新检索结果或修改应用、依赖、索引与旧报告；未调用模型、删除练习文件或提交 Git。Day 27 未完成，进度保持 26 / 42。

### Day 27｜2026-10-25｜Recall 函数通过，学习 RR 与 MRR（进行中）

- **学习者完成**：填写 `hit_count = len(hit_ids)` 与 `recall = hit_count / len(relevant_ids)`，回传方法甲 Recall@3 为 0.5、方法乙为 1.0。助手核对实际文件，并在项目虚拟环境运行同一练习，输出一致；保留学习者写法，更新已过期的待填提示。
- **新指标**：RR@K 取前 K 条中第一个相关片段名次的倒数；本来存在相关片段但前 K 条没有命中时，RR@K 为 0。MRR@K 对同一检索方法在多个评估问题上的 RR@K 求平均。它关注首个命中的位置，不衡量全部证据是否完整；无相关片段的题仍需单独处理，并明确汇总范围。
- **手算练习**：同一检索方法处理三个假设问题。问题 1 的全部相关片段为 A、C，前三条为 `B → A → D`；问题 2 的全部相关片段为 E、G，前三条为 `E → F → H`；问题 3 的全部相关片段为 J、L，前三条为 `I → K → M`。学习者计算各题 RR@3，再求三题的 MRR@3，等待回传后核对。
- **参考依据**：核对 [TorchMetrics 的 MRR 文档](https://lightning.ai/docs/torchmetrics/stable/retrieval/mrr.html)，区分每个查询的倒数排名、Top-K 截取、按查询求平均及没有正例时的处理规则；本练习无需安装新库。
- **当前状态**：本轮只运行已有 Recall 教学样例并维护练习说明与 README，没有执行真实检索或模型调用、改动应用与旧报告、删除文件或提交 Git。Day 27 未完成，进度保持 26 / 42。

### Day 27｜2026-10-25｜RR / MRR 手算通过，准备指标代码（进行中）

- **学习者回传**：三个问题的 RR@3 为 `1/2 = 0.5`、`1/1 = 1`、`0`，最终 MRR@3 为 0.5，数值正确。助手提醒平均值的完整算式应写为 `(0.5 + 1 + 0) / 3`，避免按运算优先级只对最后一项做除法。
- **代码练习**：在 `practice_retrieval_metrics.py` 保留已完成的 Recall 函数，补充 `calculate_rr_at_k()`、`calculate_mrr()` 和三个教学问题。学习者填写首次命中时的倒数名次、前 K 条未命中的返回值、跨问题平均值三处核心计算。`enumerate(..., start=1)` 提供从 1 开始的名次，找到首个相关片段后立即返回。
- **汇总范围**：已有相关片段却没有检索到时，RR 为 0，仍参加平均；没有相关片段的题在本练习中返回 `None`，单独处理并排除在 MRR 分母之外。`applicable_values` 仅排除 `None`，不能用真假判断把 0 也排除。
- **当前状态**：新增部分仍保留三处 TODO，本轮只检查脚本语法，没有执行未完成的 RR / MRR 计算，也没有运行真实检索或模型。应用、依赖、索引及旧报告保持原样，未删除练习文件或提交 Git。Day 27 未完成，进度保持 26 / 42。

### Day 27｜2026-10-25｜指标计算通过，开始真实片段标注（进行中）

- **学习者完成**：正确填写首次命中时的 `return 1.0 / rank`、未命中的 `return 0.0`，以及 `mrr = sum(applicable_values) / len(applicable_values)`。回传三个 RR@3 为 0.5、1.0、0.0，MRR@3 为 0.5，与手算一致；助手核对实际文件并保留写法。
- **轻量检查**：检查重复标识不重复计数、RR 使用首次命中、K 之外不计命中、空召回但存在相关片段时记 0、无相关片段时返回 `None`、MRR 保留 0 并排除 `None`，均通过。没有调用向量模型、精排模型或 DeepSeek。
- **真实材料准备**：新增 `practice_relevance_labels.py`，读取 Demo 原论文的已存索引与固定题 q3“位置编码使用哪两种函数？”，展示 `p5:c5`、`p6:c1`、`p6:c2` 三个片段的完整标识、来源、PDF 页码与原文。学习者逐条判断该段是否直接提供两个函数名称，并摘录依据。
- **标注范围**：这三个片段是预先选定的教学材料，不是新的检索结果，也不是全库的完整相关标签。正式实验前还需按明确的依据标准检查完整范围，不能仅凭这三段或某一路 Top-3 决定整个知识库的相关片段集合。
- **当前状态**：指标练习已完成核心计算，当前等待真实片段的判断回传；尚未生成正式标注或四种方法的对照报告。应用、依赖、索引与旧报告保持原样，未删除练习文件或提交 Git，Day 27 未完成，进度保持 26 / 42。

### Day 27｜2026-10-25｜助手完成依据判断与评估标注（进行中）

- **学习者请求**：由助手代为判断三个真实片段。助手核对原文后确认 `p5:c5` 不直接支持 q3；`p6:c1`、`p6:c2` 都明确写出 sine 与 cosine，支持回答。两条相关片段因标识不同分别计数，虽然共享同一句证据。
- **完整标注**：继续审阅固定索引的 51 个片段，按 `direct_evidence_v1` 为原有 20 道题整理正负片段集合与必需证据单元，保存为 `data/evaluations/day27_relevance_labels.json`。覆盖 15 道有答案题、5 道资料不足题，共 1020 个题目与片段组合；记录 13 个证据单元和 23 条精确原文位置，不依据检索分数判断支持关系。
- **边界与复合题**：主题相近、缺少具体事实、其他对象的数字和符号定义不完整等情况有单独说明。表 3 的 base 训练步数还通过本地 PDF 第 9 页的 layout 提取核对列对应；该片段的开发集 BLEU 不可作为 q17 的测试集 BLEU。复合题的正例可以只支持其中一个完整证据单元，不能由 RR / MRR 推断整题可答。
- **核对与展示**：校验片段归属、题集与语料摘要、正负集合覆盖及所有引用位置；保留原题集和旧报告。练习脚本改为显示原文及已完成的助手判断，并在元数据摘要变化时提示复核标注版本。标注来自助手审阅，未经过独立人工复核，不作为盲测或回答质量结论。
- **当前状态**：直接依据标注已完成，Day 27 的指标模块整合、四种方法检索对照和结果解读仍待继续，进度保持 26 / 42。本轮没有调用向量、精排或 DeepSeek 模型，没有修改应用、依赖与索引、删除练习文件或提交 Git。

### Day 27｜2026-10-25｜准备指标模块与四种方法对照脚本（进行中）

- **函数复用**：将学习者完成的 Recall、RR、MRR 函数移入 `retrieval_metrics.py`，函数语法树摘要与迁移前一致；原练习改为导入正式模块，教学样例输出仍为 Recall 0.5 / 1.0、RR 0.5 / 1.0 / 0.0、MRR 0.5。
- **连接练习**：准备 `evaluate_retrieval_results()`，由学习者填写一处从完整检索记录读取片段标识的表达式；其他转换、重复标识检查和指标调用已接好。未填写时抛出明确提示，模型加载前即可发现。
- **对照脚本**：新增 `evaluate_retrieval.py`，固定原题、原索引与当前 3 / 6 / 3 参数，复用双路召回、去重、RRF 和精排模块。记录输入摘要、标注口径、模型配置、各题完整候选及阶段快照，四种方法统一按前三条计算指标。15 道有答案题参加均值，5 道资料不足题单独记录候选；不生成模型回答。
- **轻量验证**：三个脚本语法与原教学输出通过，真实输入检查确认 20 题、15 道有答案题、5 道资料不足题及 51 个片段一致。受控汇总检查确认未命中的 0 保留、资料不足题排除、有答案题缺失指标时拒绝汇总；确认待填位置在模型加载前阻止执行。
- **当前状态**：下一步由学习者补齐片段标识读取，并执行 `--check` 回传。本轮未加载向量或精排模型，未调用 DeepSeek，未生成新的检索对照报告，未删除练习或提交 Git。Day 27 尚未完成，进度保持 26 / 42。

### Day 27｜2026-10-25｜指标连接通过与真实对照运行（进行中）

- **学习者完成**：使用 `chunk_id = result.get("chunk", {}).get("chunk_id")` 读取片段标识，回传输入检查通过：20 题、15 道有答案题、5 道资料不足题、51 个片段。保留学习者写法，将已过期的“练习未完成”异常改为正常的缺少有效标识校验。
- **连接验证**：检查排名顺序、命中标识、Recall / RR、空返回、空相关集合以及缺失和重复标识，均通过。随后由助手在项目虚拟环境运行一次真实对照，使用本地 E5 与 E 盘缓存的 BGE，未下载模型或调用 DeepSeek。
- **实际结果**：15 道有答案题的 Mean Recall@3 / MRR@3 分别为 Dense `0.5133 / 0.5444`、BM25 `0.1800 / 0.2444`、RRF `0.5156 / 0.5111`、Rerank `0.5044 / 0.7333`。保留 5 道资料不足题的候选与不适用指标，未将它们算入平均。20 题检索部分耗时约 36.90 秒，不含模型加载。
- **保存与复核**：新增 `day27_retrieval_comparison.json`，独立重算全部 80 组结果的命中与指标，检查阶段记录、15 题汇总和所有输入摘要；原有 38 个数据文件保持不变。报告不是回答生成实验，未新增回答正确率结论。
- **下一步学习**：先由学习者比较 Dense 与 Rerank 的两个汇总指标，再查看具体问题中的排名变化，解释改善与退化。Day 27 仍待结果解读，进度保持 26 / 42；本轮未删除练习、调整检索参数或提交 Git。

### Day 27｜2026-10-25｜汇总指标理解通过，进入案例解读（进行中）

- **学习者判断**：回答“recall@3 减小了，mrr@3 增大了，不能这么说”，正确识别本次 Dense 与 Rerank 的不同指标变化。助手补充：平均相关片段覆盖比例降低，而首个相关片段的倒数名次均值提高；不能由汇总值认定每一道题都有改善，也不能将检索指标当作回答正确率。
- **案例材料**：只读取已保存的对照报告。q3 的全部相关片段为 `p6:c1`、`p6:c2`，RRF 顺序是 `p5:c5 → p10:c1 → p6:c2`，精排后为 `p6:c2 → p5:c5 → p10:c1`。q17 的全部相关片段为 `p7:c4`、`p8:c1`、`p9:c1`，四条完整 RRF 候选为 `p1:c2 → p8:c1 → p8:c3 → p8:c2`，精排后为 `p8:c3 → p1:c2 → p8:c2 → p8:c1`，最终仅取前三条。这里用短标识便于阅读，报告保留完整标识和来源。
- **最后一项练习**：学习者分别计算两题精排前后的 Recall@3、RR@3，并判断 q17 中原本已找到的相关片段为何没有进入最终参考资料。单题使用 RR，跨题汇总才称 MRR；等待作答后核对，不提前填入结果。
- **当前状态**：Day 27 剩具体案例解读，进度仍为 26 / 42。本轮仅读取已有报告和维护学习记录，没有运行模型、改动代码与指标结果或提交 Git。

### Day 27｜2026-10-25｜澄清 Recall 分母与前三条截取（进行中）

- **学习者回传**：q3 两个 RR 为 `1/3`、`1/1`，计算正确；Recall 分母及 q17 的指标存在混淆，对相关片段为何落选尚不清楚。不将这次作答记录为案例验收通过。
- **计算说明**：Recall@3 的分母是本题全部标注相关片段数量，q3 固定为 2、q17 固定为 3，与候选条数和排名无关。q3 精排前后都命中一个相关片段，因此 Recall 都为 `1/2`；首个相关片段由第 3 名升到第 1 名，RR 从 `1/3` 变为 1。q17 的 RRF 前三条命中一个相关片段，Recall 为 `1/3`、RR 为 `1/2`；精排后的前三条没有标注相关片段，两项均为 0。
- **落选环节**：核对原报告，q17 的 `p8:c1` 已在融合候选中，精排将其放到第 4 名，随后最终前三条截取将其排除。它仍是标注相关片段，排序分数不保证真实依据支持关系；RR@3 不继续寻找第 4 条，因此不能记为 `1/4`。本次记录能确认排名与截取过程，不能据此断言模型内部为何给出该分数。
- **当前状态**：先解释已尝试的两个案例，不新增模型运行或调参实验。Day 27 继续保持未完成，待学习者确认分母与截取规则的理解；本轮只维护学习记录，原报告和程序不变。

### Day 27｜2026-10-25｜理解确认与收尾（已完成）

- **学习者确认**：在纠正讲解 q3、q17 后回复“明白了”。本次记录为讲解后的理解确认，没有将之前的错误计算改写为独立作答通过，也没有新增一轮测试。
- **当天成果**：学习者完成 Recall、RR、MRR 的核心代码与片段标识读取；助手完成固定题集的片段依据标注、四种方法对照脚本、一次真实模型检索及报告复核。15 道有答案题的 Mean Recall@3 / MRR@3 为 Dense `0.5133 / 0.5444`、BM25 `0.1800 / 0.2444`、RRF `0.5156 / 0.5111`、Rerank `0.5044 / 0.7333`，另 5 道资料不足题单独记录。
- **案例结论**：q3 精排前后 Recall 都为 `1/2`，RR 从 `1/3` 升至 1；q17 的相关片段 `p8:c1` 已进入候选，但精排后排第四，最终前三条没有命中，Recall@3 与 RR@3 均为 0。一次排序改善不能保证所有题、所有指标都改善；检索指标不等于回答正确率。
- **当前状态**：Day 27 标记完成，学习进度更新为 27 / 42。本轮仅维护 README，不修改报告、程序或检索参数，不重复运行模型，不删除练习文件或提交 Git；实际学习用时未记录。
- **下一次学习**：Day 28，依据失败案例选择一个检索参数做对照，优先观察召回候选数量对命中、最终排名与耗时的影响；保留本日基线和独立问题用于复验，再讨论配置取舍。当前尚未开始调参。

### Day 28｜2026-10-25｜区分召回遗漏与排序落选（开始）

- **现状核对**：现有评估脚本的每路召回上限为 3、RRF 上限为 6、最终评价 K 为 3，排名常数为 60。Day 27 的 q5、q12、q13 在当前各阶段最终前三条中均未命中；q17 则已有相关片段进入候选，但在精排后落到第四名。两类失败需要分别观察。
- **第一步材料**：q5“论文采用了多少个并行注意力头？”的标注相关片段为 `p5:c1`；Dense 返回 `p10:c5 → p11:c4 → p1:c2`，BM25 返回空，因此交给精排的候选也只有这三条。学习者先判断仅改变这些候选的精排顺序能否找到缺失依据，并解释原因；暂不提前提供作答结论。
- **单参数对照计划**：A 组每路召回最多 3 条，B 组最多 6 条；两组都保持 RRF 最多 6 条、最终最多 3 条以及同一分块、模型、原始问题和标注。去重前记录数的上限分别为 6 和 12，实际数量取决于匹配与重叠。增加召回数量只是待验证的假设，不预设最终指标一定提高；需要跟踪相关片段是否进入融合、精排和最终参考资料。
- **评价与复验计划**：先用已知失败题定位问题，再按统一 K 比较指标和耗时，记录本地模型加载与检索阶段的区别。原 20 题已经用于观察和分析，只作为开发材料；在运行新配置前另行固定复验问题及其标签，不用复验结果反复挑参数。具体复验题尚待准备。
- **当前状态**：Day 28 未完成，进度保持 27 / 42。本轮只读取代码、旧报告并维护学习记录，没有运行模型、修改检索参数或页面、生成新实验报告、删除练习文件或提交 Git。

### Day 28｜2026-10-25｜确认召回遗漏，准备参数传递练习（进行中）

- **学习者判断**：回答“找不到，因为 p5:c1 就不在 dense 候选里面”，针对本题判断正确。助手补充范围：精排面对的是两路合并后的候选，本题 BM25 为空，所以该片段完全没有进入精排输入；其他题若 Dense 漏掉而 BM25 找到，仍可能进入后续流程。
- **参数入口**：在 `evaluate_retrieval.py` 的 `evaluate_case()` 新增仅限关键字传入的 `recall_top_k`，默认沿用原常数 3，并检查其为正整数。学习者需要将调用 `retrieve_dual()` 时 `top_k` 右侧的固定常数改为本次收到的参数；当前该行保留 TODO，尚未完成传递。默认 CLI 配置与正式页面配置仍为原值。
- **检查脚本**：新增本地 `practice_retrieval_tuning.py`，分别传入 3 和 6，临时用返回空候选的模拟函数替换双路检索，检查底层实际收到的参数。同时检查原默认配置保持 3 / 6 / 3；无需模型、网络或输出报告。这只检查参数连接，不代表真实召回效果。
- **当前状态**：两个脚本语法检查通过，尚未运行待补齐的参数检查。下一步由学习者填写一行并回传模拟输出，再准备固定复验材料和真实实验。Day 28 未完成，进度保持 27 / 42；未修改页面、运行模型、覆盖原报告、删除练习或提交 Git。

### Day 28｜2026-10-25｜参数传递通过，固定对照与复验材料（进行中）

- **学习者完成**：填写 `top_k=recall_top_k`，回传两组传入与收到的数量分别为 3 / 3、6 / 6，RRF 上限为 6、最终 K 为 3。助手核对实际代码并运行同一模拟检查，结果一致；保留学习者写法，更新已过期的练习注释。
- **固定计划**：在真实运行新配置前新增 `day28_tuning_plan.json`，保存 A=3、B=6 的配置，开发题 q5 / q12 / q13，以及五道新复验题的标签。复验题覆盖训练句对数量、base dropout、检查点平均、代码仓库路径和未提供的操作系统版本；按 51 个片段审阅 255 个题目与片段组合，四道有答案、一道资料不足，记录四条精确引用和边界说明。题集来自同一篇论文，标签由助手审阅，没有独立人工复核。
- **实验入口**：新增 `compare_recall_limits.py`，复用学习者参数化的 `evaluate_case()`，同一进程加载一次模型，对三道开发题交替运行 A/B 共六次流程。记录完整阶段结果、各阶段相关片段是否进入、最终前三条指标与耗时，并保存独立报告。五道复验题本轮只校验数据，不运行模型。
- **轻量验证**：计划与输入一致性、原文引用位置及摘要检查通过；用合成向量、固定精排分数验证六次调用、交替顺序、每路 3 / 6 上限、最终 K=3、阶段命中和分组汇总。实际数据文件的前后摘要确认原有 39 个文件未变，只新增计划；未运行真实论文模型检索。
- **下一步**：学习者运行开发对照脚本，回传六行逐题明细和两行汇总，再判断候选数量改变后的命中与耗时。新复验题保留到暂选配置后使用。Day 28 未完成，进度仍为 27 / 42；正式页面和旧报告未修改，未下载模型、调用 DeepSeek、删除练习或提交 Git。

### Day 28｜2026-10-25｜开发题真实对照完成，分析收益与开销（进行中）

- **实际回传**：学习者运行六次流程并回传结果。A 组 q5 / q12 / q13 均未命中，平均 Recall@3 和 MRR@3 均为 0；B 组 q12、q13 都命中且首条相关，q5 仍未命中，两项均值均为 `2/3 = 0.6667`。A、B 平均耗时分别为 1.076、1.943 秒。
- **排名路径核对**：报告显示 q12 的 `p5:c4` 在 B 组 Dense / RRF 中均为第 6，精排后第 1；q13 的 `p8:c3` 原为第 5，精排后第 1。扩充召回数量使这两段进入精排输入，精排再将其移入最终前三条。q5 的依据没有进入 B 组候选，增加到 6 条仍未解决本题。
- **开销与边界**：按报告未四舍五入的计时，B 平均增加约 0.866 秒、约为 A 的 1.805 倍。该计时不含模型加载及 Python 导入，仅为三道开发题的单次观测。开发题已知失败且数量很少，不能据此认定所有题都改善，也不能将候选命中解释为最终模型回答正确。
- **报告验证**：只读复核六条流程记录、24 组方法结果、完整阶段命中及组内平均；输入、实验计划与代码摘要一致。确认计划保存早于实际实验，五道新复验题运行次数和 DeepSeek 调用次数均为 0。保留原始报告，没有补跑模型或修改实验数据。
- **下一步学习**：由学习者基于开发结果暂选 A 或 B，并说明检索收益与耗时代价；记录选择后再进行固定新问题复验。Day 28 尚未完成，进度保持 27 / 42，正式页面配置未调整；本轮只维护 README，未提交 Git。

### Day 28｜2026-10-25｜按质量优先暂选 B，准备固定新题复验（进行中）

- **学习者选择**：选择 B，说明愿意接受时间增加，个人更看重质量。助手澄清本次提高的是相关片段召回与排名，尚未测量模型回答正确率；保留该偏好用于后续配置取舍。
- **事先记录**：在运行新题之前新增 `day28_selection.json`，保存用户原话、暂选 B 的参数、开发结果与输入摘要，并记录当时复验运行次数为 0。原计划、题目、标签、开发报告和正式页面配置保持不变。
- **复验入口**：新增 `validate_recall_limits.py`，复用原对照函数，在固定 v1～v5 上分别运行 A/B 共十次流程。运行前核对暂选配置和来源摘要，质量均值仅包含四道有答案题，耗时均值包含全部五道题；资料不足题保留候选并将指标记为不适用。共用驱动只补充 `None` 指标打印，不改变检索算法。
- **验证边界**：`--check` 通过，受控检查确认空相关集合显示与排除规则、未命中的 0 保留、最终 K=3，以及不一致的选择会被拦截。未加载真实模型，未执行新问题检索，未生成复验报告；原有 41 个数据文件摘要未变，只新增选择记录。
- **下一步**：学习者运行复验脚本，保存完成后由助手直接读取报告，一起比较新题结果并形成最终配置结论。Day 28 未完成，进度保持 27 / 42；未下载模型、调用 DeepSeek、删除练习文件或提交 Git。

### Day 28｜2026-10-25｜新题复验、采用 B 与收尾（已完成）

- **实际复验**：学习者完成运行并回复“保存好了”。助手读取报告：五道新题分别运行 A/B，共十次流程；四道有答案题的最终平均 Recall@3 与 MRR@3 均从 A 的 0.50 提升到 B 的 0.75，五题平均耗时由 1.241940 秒增至 1.840414 秒。资料不足题 v5 的指标保留 `null`，不计入质量均值。
- **逐题复盘**：v1 的 `p7:c3` 被 B 组以 Dense / RRF 第 6 名召回，经精排升到第 1；v2、v3 两组均命中，v4 两组仍未命中。新题中未观察到最终 Recall / RR 的退化，但仅有四道有答案题，不能推广为所有问题都改善。未调用 DeepSeek，因此不新增回答正确率或拒答质量结论。
- **采用配置**：依据学习者先前明确的质量优先偏好及本次复验，工作台每路召回上限从 3 调整为 6，RRF 上限保持 6、最终返回保持 3。实验脚本的 A 组默认值与原报告不变，选择记录保留复验前状态，新增 `day28_conclusion.json` 保存最终采用依据。
- **页面检查**：沿用现有 16 组页面场景，扩展控制候选来覆盖两路各 6 条、合并 9 条、融合 6 条和最终 3 条，以及未进入精排的候选。首次运行 72.113 秒，15 项通过、1 项因旧 BM25 数量断言失败；定位后更新控制排序与断言，受影响的两项复验耗时 6.397 秒并通过。16 个不同场景均已验证，不表述为一次运行全部通过。原有 43 个数据文件摘要未变，测试未加载真实模型或调用 DeepSeek。
- **当天完成**：Day 28 标记完成，学习进度更新为 28 / 42。重启 Streamlit 后新问题采用 B 配置。保留开发题 q5、复验题 v4 的遗漏记录，不继续用本次新题调整更多参数；没有删除练习、缓存、旧报告或提交 Git，实际学习用时未记录。
- **下一次学习**：Day 29，把现有检索封装成带名称、描述和参数的工具，学习模型如何请求调用工具，以及程序如何执行并把结果交回模型。当前尚未开始 Agent 工具改造。
