"""
linear-workflow.py — LangGraph 线性工作流（Linear Workflow）最小示例

流程：文本清洗(process_doc) → LLM 生成摘要(generate_summary) → 格式优化(format_summary) → END
特点：节点之间用固定边按顺序连接，没有分支、没有循环，是最基础的工作流形态；
      适合"输入 → 多步顺序加工 → 输出"的确定性流水线。

运行方式：
    d:/langgraph-tutorial/.venv/Scripts/python.exe "d:/langgraph-tutorial/04智能体工作流/linear-workflow.py"

依赖（见项目根目录 requirements.txt）：langgraph、langchain-core、langchain-openai、python-dotenv
环境变量：在项目根目录 .env 中配置 deepseek_API_KEY
"""

# ============================================================
# 1. 导入依赖
# ============================================================
from typing import TypedDict, Optional  # TypedDict: 声明状态字典的字段结构；Optional: 允许字段为 None

from dotenv import load_dotenv  # 从 .env 文件加载环境变量到 os.environ
import os  # 用 os.getenv 读取环境变量（API Key）

from langchain_openai import ChatOpenAI  # OpenAI 兼容接口的聊天模型封装（这里指向 DeepSeek 中转服务）
from langchain_core.prompts import ChatPromptTemplate  # 聊天提示词模板：由 system / human 多条消息组成
from langchain_core.output_parsers import StrOutputParser  # 把 LLM 返回的 AIMessage 解析为纯字符串
from langgraph.graph import StateGraph, END  # StateGraph: 状态图构建器；END: 内置的结束节点常量

# ============================================================
# 2. 加载环境变量
# ============================================================
load_dotenv()  # 读取项目根目录 .env 并写入进程环境变量；必须在读取 API Key 之前调用

# ============================================================
# 3. 定义工作流状态
# ============================================================
# 定义状态结构：存储原始文档、预处理后的文本、LLM摘要结果
# 注意：函数注解在 def 执行时就会求值，因此 TypedDict 必须定义在使用它的函数之前
class SummaryState(TypedDict):  # 状态契约：所有节点共享同一份状态字典，按 key 读写
    raw_document: str  # 输入：原始文档（invoke 时必须提供）
    processed_text: Optional[str]  # 中间结果：预处理（清洗）后的文本，由节点1写入
    summary: Optional[str]  # 输出：LLM生成的摘要，随后被节点3改写为格式化摘要


# ============================================================
# 4. 初始化 LLM
# ============================================================
## 初始化 LLM
llm = ChatOpenAI(  # 创建聊天模型客户端；三个节点复用同一个实例
    model="deepseek-v4.1-flash-1017",  # 模型名称
    api_key=os.getenv("deepseek_API_KEY"),  # 从 .env 读取，避免把密钥硬编码进代码
    base_url="https://tokenrhythm.studio/v1",  # OpenAI 兼容的中转地址（DeepSeek 网关）
    temperature=0  # 温度为 0：输出尽量确定，适合摘要这类不希望发挥创意的任务
)

# ============================================================
# 5. 提示词模板与处理链
# ============================================================
# 定义摘要提示词模板（指令+上下文）
summary_prompt = ChatPromptTemplate.from_messages([  # 由多条对话消息组成的模板对象
    # 系统消息：设定角色与约束（长度、忠实于原文、不添加额外信息）
    ("system", "你是专业的文本摘要助手，需基于以下文本生成简洁、准确的摘要，不添加额外信息，长度控制在300字以内。"),
    # 人类消息：{processed_text} 是占位符，invoke 时用输入字典中的同名 key 填充
    ("human", "文本内容：{processed_text}")
])

# 构建摘要链（提示词+LLM+输出解析器）
summary_chain = summary_prompt | llm | StrOutputParser()  # LCEL 管道：填模板 → 调模型 → 只取文本内容


# ============================================================
# 6. 定义节点函数（每个节点：读状态 → 计算 → 返回状态更新）
# ============================================================
def generate_summary(state: SummaryState) -> SummaryState:  # 节点2：调用 LLM 生成摘要
    """调用DeepSeek LLM生成文本摘要"""
    processed_text = state["processed_text"]  # 取出节点1清洗后的文本
    # 执行摘要链
    summary = summary_chain.invoke({"processed_text": processed_text})  # 同步调用链，返回纯字符串摘要
    # 返回更新后的状态
    return {"summary": summary}  # 只返回被修改的 key；LangGraph 会把它合并进全局状态


def format_summary(state: SummaryState) -> SummaryState:  # 节点3：对摘要做格式后处理
    """后处理摘要结果：优化格式"""
    summary = state["summary"]  # 取出节点2生成的原始摘要
    # 格式优化逻辑：添加标题、分段
    formatted_summary = f"### 文档摘要\n\n{summary.replace('. ', '.\n')}"  # 加 Markdown 标题，并把英文句号后换行分段
    # 返回最终状态
    return {"summary": formatted_summary}  # 用格式化后的文本覆盖 summary 字段


## 节点 文本清洗
def process_document(state: SummaryState) -> SummaryState:  # 节点1：清洗原始文档（工作流入口节点）
    # 实现文本清洗逻辑
    cleaned_text = state["raw_document"].replace("\n", " ")  # 把换行符替换为空格，压成单行文本
    state["processed_text"] = cleaned_text  # 直接原地写入状态字典（也可像其他节点一样返回新字典）
    return state  # 返回整个状态；其中的 raw_document / processed_text 都会被合并回全局状态


# ============================================================
# 7. 构建线性工作流图
# ============================================================
## 构建线性工作流图
# 1. 初始化状态图（绑定状态结构）
graph_builder = StateGraph(SummaryState)  # 创建状态图，并声明它使用 SummaryState 这套字段

# 2. 添加节点（每个节点映射到对应的处理函数）
graph_builder.add_node("process_doc", process_document)  # 节点1：预处理（注册名为 process_doc）
graph_builder.add_node("generate_summary", generate_summary)  # 节点2：LLM摘要
graph_builder.add_node("format_summary", format_summary)  # 节点3：后处理

# 3. 定义线性边（顺序执行：预处理→摘要生成→格式优化→结束）
graph_builder.add_edge("process_doc", "generate_summary")  # 固定边：节点1执行完必走节点2
graph_builder.add_edge("generate_summary", "format_summary")  # 固定边：节点2执行完必走节点3
graph_builder.add_edge("format_summary", END)  # 最后一个节点指向END（工作流结束）

# 4. 设置入口节点（工作流起始点）
graph_builder.set_entry_point("process_doc")  # 指定 invoke 时第一个执行的节点

# 5. 编译生成工作流实例
summary_graph = graph_builder.compile()  # 编译成可执行对象（会做结构校验，如入口/孤立节点）

# ============================================================
# 8. 测试数据与执行
# ============================================================
# 测试用原始文档（可替换为实际文档）
# 下面这段多行字符串是节点1的输入 raw_document，会原样进入工作流，因此内部不能写注释：
#   - 第1句：介绍 LangGraph 的定位（LangChain 生态中构建有状态、可循环工作流的框架）
#   - 第2句：说明其实现思想（基于状态机定义节点与边，用状态对象管理数据流转）
#   - 第3句：与传统线性脚本对比，强调可扩展性、可观测性及典型适用场景
#   - 第4句：列举核心组件 StateGraph / State / Node / Edge
raw_document = """
LangGraph 是 LangChain 生态系统中的一个框架，专门用于构建状态ful、可循环的工作流。
它基于状态机的思想，允许开发者定义节点和边，并通过状态对象管理整个工作流的数据流转。
与传统的线性脚本相比，LangGraph 提供了更好的可扩展性和可观测性，特别适合 LLM 应用中的复杂流程编排，
例如多轮对话、文档分析、工具调用链等场景。LangGraph 的核心组件包括 StateGraph、State、Node 和 Edge，
这些组件共同构成了灵活且强大的工作流系统。
"""

# 执行工作流：输入初始状态（仅包含raw_document）
result = summary_graph.invoke({  # 同步执行整张图，返回合并后的最终状态
    "raw_document": raw_document  # 只需提供输入字段；processed_text / summary 由节点逐步写入
})

# 输出结果
print("最终摘要：")  # 打印提示标题
print(result["summary"])  # 打印节点3格式化后的摘要（含 "### 文档摘要" 标题）
