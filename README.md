# LangGraph 教程（01 · 02）

跟着 [LangGraph](https://github.com/langchain-ai/langgraph) 一步一步写出来的实战代码，从「最小聊天机器人」起步，逐步加上工具调用与多轮记忆。

## 当前进度

| 目录 | 主题 | 状态 |
|---|---|---|
| [`01基础聊天机器人/`](01基础聊天机器人/) | 最小可运行的聊天机器人（单向图） | ✅ |
| [`02带记忆的机器人/`](02带记忆的机器人/) | 工具调用 + 多轮记忆（ReAct 循环 + checkpointer） | ✅ |

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
├── requirements.txt
├── .env.example
└── .gitignore
```

## 环境要求

- Python 3.10+（推荐 3.11 / 3.12）
- 一个支持 OpenAI 兼容协议的模型服务（DeepSeek 官方、中转网关、本地 vLLM / Ollama 均可）

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

## 两个脚本的差别

- **01 基础聊天机器人**：`START → chatbot → END` 的单向图，每次问答都是独立的，不记历史、不能调工具。
- **02 带记忆的机器人**：`chat ⇄ tools` 的循环图，`tools_condition` 决定要不要调工具，`MemorySaver` 按 `thread_id` 保存历史，所以它能记住你上一轮说的名字，也能查当前时间。

图结构、逐行代码解析与完整对话案例见 [`02带记忆的机器人/README.md`](02带记忆的机器人/README.md)。

## ⚠️ 关于密钥安全

- 仓库中**不包含**任何真实密钥，所有密钥通过 `.env` 注入，`.env` 已被 `.gitignore` 排除。
- 提交前请用 `git status` / `git add -n .` 确认待提交内容里没有 `.env` 或任何 `sk-` 开头的字符串。
- 如果曾把含密钥的 `.env` 提交过，请立即到服务商后台**吊销并轮换**密钥——从 Git 历史里删掉文件并不能阻止已泄露的密钥被使用。
