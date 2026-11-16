from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# from langchain_tavily import TavilySearch
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
import os

load_dotenv()
llm = ChatOpenAI(
    model="deepseek-v4.1-flash-1017",
    api_key=os.getenv("deepseek_API_KEY"),  # 从 .env 读取
    base_url="https://tokenrhythm.studio/v1",
    temperature=0.7
)

def get_date(local_date: str) -> str:
    """获取当前时间"""
    import time
    local_time = time.localtime()
    local_date=time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    return f"当前时间为 {local_date}!"

# search_tool = TavilySearch(max_results=2)

# tools = [search_tool, get_date]
tools = [get_date]
llm_with_tools = llm.bind_tools(tools)
state = MessagesState

workflow = StateGraph(state)
workflow.add_node("chat", lambda state: {"messages": llm_with_tools.invoke(state["messages"])})
workflow.add_node("tools", ToolNode(tools=tools))
#tools_condition是一个预置的条件路由函数，专门用于实现 ReAct 风格 Agent 的“思考-行动”循环。
# 它的核心作用是自动判断是否需要调用外部工具，并据此决定工作流的下一步走向。
workflow.add_conditional_edges("chat", tools_condition)
workflow.add_edge("tools", "chat")
workflow.set_entry_point("chat")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


def chat_with_memory():
    print("智能对话机器人（支持记忆和搜索）")

    print("输入 exit 退出对话")
    thread_id = "user_session_1"
    config = {"configurable": {"thread_id": thread_id}}
    while True:
        try:
            user_input = input("\n你：").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "退出"]:
                print("机器人：再见！")
                break

            result = app.invoke(
                {"messages": [("user", user_input)]},
                config=config
            )

            if result["messages"]:
                last_message = result["messages"][-1]
                print("机器人：", last_message.content)

        except KeyboardInterrupt:
            print("\n对话已结束")
            break
        except Exception as e:
            print(f"出错：{str(e)}")

if __name__ == "__main__":
    chat_with_memory()

