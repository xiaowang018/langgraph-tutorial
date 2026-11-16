# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 15:24:06 2025

@author: liguo
"""
from typing import Annotated
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

# 启用 dotenv 读取环境变量
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# 从环境变量读取 API Key
llm = ChatOpenAI(
    model="deepseek-v4.1-flash-1017",
    api_key=os.getenv("deepseek_API_KEY"),  # 从 .env 读取
    base_url="https://tokenrhythm.studio/v1",
    temperature=0
)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# 添加节点和边
#向图中添加一个名为"chatbot"的节点，这个节点的执行逻辑由我们定义的chatbot函数来实现。
graph_builder.add_node("chatbot", chatbot)
#定义了一条从START节点到"chatbot"节点的边。这告诉图执行引擎：​“工作流开始时，请首先执行'chatbot'节点”​。
graph_builder.add_edge(START, "chatbot")
#这是最后一步，也是关键的一步。它将graph_builder对象编译成一个最终可执行的graph对象。在这个阶段，LangGraph会进行一些内部优化和验证，确保图的结构是正确的。
graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    # 调用graph对象的stream方法来执行图。与invoke方法一次性获取所有结果不同，stream会以流式生成的方式逐步返回执行过程中的更新
    # {"messages"：​[{"role"："user"，"content"：user_input}]}：这是启动图时传入的初始状态。它是一个State类型的字典，包含用户的最新输入。
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        # 遍历流式返回的事件(event)。每个event代表图执行过程中的一个状态更新。
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break

