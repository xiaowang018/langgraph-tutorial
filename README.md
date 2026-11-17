# LangGraph 教程（01 · 02 · 03）

跟着 [LangGraph](https://github.com/langchain-ai/langgraph) 一步一步写出来的实战代码：从「最小聊天机器人」起步，逐步加上工具调用与多轮记忆，最后落地一个带私有知识库的 RAG 检索问答。

## 当前进度

| 目录 | 主题 | 状态 |
|---|---|---|
| [`01基础聊天机器人/`](01基础聊天机器人/) | 最小可运行的聊天机器人（单向图） | ✅ |
| [`02带记忆的机器人/`](02带记忆的机器人/) | 工具调用 + 多轮记忆（ReAct 循环 + checkpointer） | ✅ |
| [`03RAG检索/`](03RAG检索/) | RAG 检索增强问答：PDF 知识库 → FAISS 向量检索 → 条件分支图 | ✅ |

## 目录结构

```
.
├── 01基础聊天机器人/
│   └── basic-chatbot.py              # 最小聊天机器人
├── 02带记忆的机器人/
│   ├── Memory-Tool-Chatbots.py       # 带记忆 + 工具调用的机器人
│   ├── README.md                     # 代码详解、图结构、案例演示
│   ├── graph-structure.png           # 图结构
│   └── graph-compare.png             # 01 / 02 图结构对比
├── 03RAG检索/
│   ├── RAG.py                        # RAG 主脚本（环境自检 / 索引缓存 / --mock 离线演示）
│   ├── RAG.py代码详解.md             # 逐段讲解：整体架构、六步流程、设计决策、FAQ
│   ├── 环境修复说明.md               # 依赖、.env、模型回退、PDF / 索引等环境说明
│   ├── rag_graph.png                 # 运行流程图（离线建库 + 在线问答 + 共享状态）
│   ├── test.pdf                      # 示例知识库文档
│   └── .gitignore                    # 排除 .env / 向量索引 / 缓存
├── requirements.txt
├── .env.example
└── .gitignore
```

## 环境要求

- Python 3.10+（推荐 3.11 / 3.12）
- **01 / 02**：一个支持 OpenAI 兼容协议的模型服务（DeepSeek 官方、中转网关、本地 vLLM / Ollama 均可）
- **03**：
  - 大模型：任意 OpenAI 兼容网关（默认 `deepseek-v4.1-flash-1017`）
  - 向量模型：火山方舟 Ark 的豆包向量（默认 `doubao-embedding-text-240715`，未开通时会自动回退到可用的豆包向量模型）
  - 两者都通过 `.env` 配置；**没有 Key 也可以加 `--mock` 离线跑通整条流程**

## 快速开始

```bash
git clone <this-repo>
cd <this-repo>

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

# 复制环境变量模板，填入自己的密钥
copy .env.example .env        # Windows
cp .env.example .env          # macOS / Linux

python 01基础聊天机器人/basic-chatbot.py
python 02带记忆的机器人/Memory-Tool-Chatbots.py
```

两个脚本都进入命令行对话循环：`01` 输入 `quit` / `exit` / `q` 退出，`02` 输入 `exit` / `quit` / `退出` 退出。

### 03 RAG 检索

```bash
# 环境自检：解释器 / 依赖 / Key / 大模型 / 向量模型 / PDF / 索引
python 03RAG检索/RAG.py --doctor

# 首次运行：构建向量索引 + 跑内置测试案例（之后自动复用索引）
python 03RAG检索/RAG.py

# 直接提问（可传多个问题）
python 03RAG检索/RAG.py "这份文档主要讲了什么？"

# 换了 PDF / 换了向量模型后强制重建索引
python 03RAG检索/RAG.py --rebuild

# 离线模拟模式：不调用任何 API，用假模型 + 假向量演示整条 RAG 流程
python 03RAG检索/RAG.py --mock
```

03 章 `.env` 里至少要有这些配置（模板见 `.env.example`）：

```dotenv
# 大模型（OpenAI 兼容网关）
LLM_BASE_URL=https://tokenrhythm.studio/v1
LLM_MODEL=deepseek-v4.1-flash-1017
deepseek_API_KEY=你的大模型 Key

# 向量模型（火山方舟 Ark）
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_API_KEY=你的方舟 Key
EMBEDDING_MODEL=doubao-embedding-text-240715
EMBEDDING_FALLBACKS=doubao-embedding-large-text-250515,doubao-embedding-vision-251215
```

PDF 放在 `03RAG检索/test.pdf`（与 `RAG.py` 同级），换成自己的文档时覆盖同名文件后加 `--rebuild` 即可。

## 三个脚本的差别

- **01 基础聊天机器人**：`START → chatbot → END` 的单向图，每次问答都是独立的，不记历史、不能调工具。
- **02 带记忆的机器人**：`chat ⇄ tools` 的循环图，`tools_condition` 决定要不要调工具，`MemorySaver` 按 `thread_id` 保存历史，所以它能记住你上一轮说的名字，也能查当前时间。
- **03 RAG 检索**：`judge_retrieve →（条件边）retrieve / generate_answer → END` 的状态图。先让大模型判断问题是否需要查私有文档：需要就检索 PDF 文本块、把上下文交给大模型回答，不需要就直接用通用知识回答；带索引缓存、环境自检、失败重试和离线模拟。

更详细的内容：

- 02 章的图结构、逐行代码解析与完整对话案例见 [`02带记忆的机器人/README.md`](02带记忆的机器人/README.md)
- 03 章的代码结构与设计决策见 [`03RAG检索/RAG.py代码详解.md`](03RAG检索/RAG.py代码详解.md)
- 03 章的运行流程图见 [`03RAG检索/rag_graph.png`](03RAG检索/rag_graph.png)
- 03 章的依赖 / `.env` / 模型回退说明见 [`03RAG检索/环境修复说明.md`](03RAG检索/环境修复说明.md)

## ⚠️ 关于密钥安全

- 仓库中**不包含**任何真实密钥，所有密钥通过 `.env` 注入，`.env` 已被 `.gitignore` 排除。
- 03 章额外做了两层保护：`03RAG检索/.gitignore` 排除 `.env`、`faiss_rag_index*/`（自动生成的向量索引）和 `__pycache__/`；文档里的 Key 只保留脱敏后的片段（如 `sk_tr_****`）。
- 提交前请用 `git status` / `git add -n .` 确认待提交内容里没有 `.env` 或任何 `sk-` 开头的字符串；也可以扫一遍暂存内容：

  ```bash
  git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{16,}|ark-[a-f0-9]{8}-|tvly-|lsv2_|gho_'
  ```

- 如果曾把含密钥的 `.env` 提交过，请立即到服务商后台**吊销并轮换**密钥——从 Git 历史里删掉文件并不能阻止已泄露的密钥被使用。