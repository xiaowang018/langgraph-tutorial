"""
non-linear-workflow.py —— LangGraph 非线性（条件路由）工作流示例

作用：
    演示一个「智能客服」工作流如何用 LangGraph 表达非线性流程：
        分类（classify_query）→ 条件路由（route_after_classify）
        → 订单处理 / 产品咨询 / 投诉记录（→ 投诉转人工）
        → 校验（validate_response）→ 条件路由（route_after_validation）
        → 结束（END）或回到原处理节点重试。

    与线性示例的区别：本示例不是 "A→B→C" 的固定顺序，而是用条件边按用户意图分流，
    并用校验节点 + 重试计数器形成"可控循环"，直到响应完整或达到重试上限。

运行方式：
    d:/langgraph-tutorial/.venv/Scripts/python.exe "d:/langgraph-tutorial/04智能体工作流/non-linear-workflow.py"

说明：
    脚本会真实调用 LLM（配置见下方「环境变量与 LLM 初始化」），
    运行 run_all_tests() 里的 5 个测试用例，耗时可能数分钟。
"""

# ============================================================
# 1. 导入依赖
# ============================================================
from dotenv import load_dotenv  # 导入 load_dotenv：从 .env 文件加载环境变量到进程环境
import os  # 导入 os 模块：用于 os.getenv 读取环境变量
from langchain_openai import ChatOpenAI  # 导入 ChatOpenAI：调用 OpenAI 兼容接口的聊天模型封装
# 类型标注：TypedDict 定义状态结构；Optional 表示可选值；Literal 用于字面量标注（本文件未实际使用）
from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, END  # StateGraph：状态图构建器；END：表示终止的哨兵节点值

# ============================================================
# 2. 环境变量与 LLM 初始化
# ============================================================
# 加载环境变量
# 说明：读取项目根目录 .env 中的 deepseek_API_KEY 等配置；找不到 .env 时不会报错
load_dotenv()  # 执行加载，把 .env 里的键值写入当前进程的环境变量

## 初始化 LLM
# 说明：整个工作流只创建一个 llm 实例，所有节点共用，避免重复建连
llm = ChatOpenAI(  # 构造 LangChain 的聊天模型对象
    model="deepseek-v4.1-flash-1017",  # 模型名称：由 base_url 指向的服务端提供
    api_key=os.getenv("deepseek_API_KEY"),  # 从 .env 读取
    base_url="https://tokenrhythm.studio/v1",  # 服务端地址：OpenAI 兼容的 /v1 端点
    temperature=0  # 采样温度 0：分类、校验这类判断型任务需要稳定、可复现的输出
)

# ============================================================
# 3. 工作流状态定义
# ============================================================
# 定义工作流状态结构
# 说明：状态是节点之间唯一的通信方式，每个节点都返回"旧状态 + 本次局部更新"的完整字典
class SupportState(TypedDict):  # 用 TypedDict 描述共享状态；仅做类型提示，运行时不校验
    """工作流共享状态（图内流转的"内存"）。

    字段含义：
        user_query   用户原始查询文本，全程不变
        query_type   分类结果：订单问题 / 产品咨询 / 投诉建议
        order_id     订单号（只有订单问题且模型抽取成功时才有值）
        response     给用户的响应文本（中间结果或最终结果）
        is_complete  响应是否已完整，供 route_after_validation 决定结束还是重试
        retry_count  已重试次数，供 validate_response 判断是否达到上限（>=2 强制结束）
    """

    user_query: str  # 用户原始查询
    query_type: Optional[str]  # 问题类型（订单/产品/投诉）
    order_id: Optional[str]  # 订单号（仅订单问题需要）
    response: Optional[str]  # 最终响应
    is_complete: bool  # 响应是否完整（用于动态路由）
    retry_count: int  # 添加重试计数器

# ============================================================
# 4. 各节点函数（每个节点 = 图中的一个"步骤"）
# ============================================================
# 节点约定：签名固定为 (state) -> dict（这里写成 -> SupportState 只是类型提示）
# 返回值统一用 {**state, ...}：先保留全部旧字段，再覆盖本次改动的字段


# ------------------------------------------------------------
# 4.1 分类节点：classify_query
# ------------------------------------------------------------
def classify_query(state: SupportState) -> SupportState:  # 节点 1：判断用户问题属于哪一类
    """分类用户查询：订单问题/产品咨询/投诉建议

    输入：
        state["user_query"] —— 用户原始查询文本
    输出：
        新的状态字典，写入 query_type（分类结果）与 order_id（订单号；非订单问题为 None）
    流程位置：
        工作流入口节点；其 query_type 由 route_after_classify 决定后续走哪条分支
    """
    user_query = state["user_query"]  # 取出用户原始查询，供下面两个 prompt 插值

    # 调用Deepseek进行分类
    # 下面分类 prompt 的分段含义：
    #   第 1 段：给出"三选一"的分类任务，把输出空间限制为三类
    #   第 2 段：标题"分类规则："，告诉模型下面是判断依据
    #   第 3-5 段：三类问题各自的关键词特征（订单 / 产品 / 投诉）
    #   第 6 段：用 f-string 把用户查询原文插进去（{user_query}）
    #   第 7 段：约束只返回分类结果，便于后面用 == 精确匹配
    # 注意：字符串内部不要加 # 注释，否则会改变真正发给 LLM 的内容
    prompt = f"""
    请将用户查询分类为以下三类之一：订单问题、产品咨询、投诉建议
    分类规则：
    - 订单问题：包含订单号、查询物流、修改订单、取消订单等关键词
    - 产品咨询：包含产品功能、使用方法、规格参数、兼容性等关键词
    - 投诉建议：包含投诉、不满、建议、改进等关键词

    用户查询：{user_query}
    仅返回分类结果，无需额外说明。
    """

    # .invoke() 同步调用模型；.content 取回复正文；.strip() 去掉首尾空白/换行
    query_type = llm.invoke(prompt).content.strip()  # 得到分类结果字符串
    print(f"问题分类结果：{query_type}")  # 打印分类结果，便于观察运行过程

    # 提取订单号（仅订单问题）
    order_id = None  # 默认无订单号：非订单问题保持 None
    if query_type == "订单问题":  # 只有订单类问题才需要再抽一次订单号
        # 订单号抽取 prompt 的分段含义：
        #   第 1 段：说明订单号的形态（6-12 位数字，或字母+数字组合），给出抽取标准
        #   第 2 段：约定"未提及则返回无"，让"无"成为一个可显式判断的值
        #   第 3 段：插入用户查询原文
        #   第 4 段：约束输出只有订单号本身或"无"，避免模型附加解释
        extract_prompt = f"""
        从用户查询中提取订单号（通常为6-12位数字或字母+数字组合），
        若未提及订单号，返回"无"。

        用户查询：{user_query}
        仅返回订单号或"无"。
        """
        order_id = llm.invoke(extract_prompt).content.strip()  # 抽取到的订单号或"无"
        print(f"提取订单号：{order_id}")  # 打印抽取结果

    # 返回值：{**state} 表示"先展开旧状态、保留全部旧字段"，后面的键再覆盖同名字段
    return {  # 返回更新后的完整状态字典
        **state,  # 保留旧状态字段（如 user_query）
        "query_type": query_type,  # 写入分类结果，供路由函数读取
        "order_id": order_id,  # 写入订单号（可能为 None）
        "is_complete": False,  # 初始响应未完成
        "retry_count": 0  # 初始化重试计数器
    }


# ------------------------------------------------------------
# 4.2 订单处理节点：handle_order
# ------------------------------------------------------------
def handle_order(state: SupportState) -> SupportState:  # 节点 2：处理"订单问题"分支
    """处理订单问题：查询订单状态（模拟工具调用）

    输入：
        state["order_id"] —— 分类节点抽取到的订单号（可能是 None 或"无"）
        state["retry_count"] —— 已重试次数，决定用哪种话术
    输出：
        新状态：写入 response，并把 retry_count 加 1
    流程位置：
        classify_query 的条件边目标之一；执行完进入 validate_response 校验
    """
    order_id = state.get("order_id")  # 取订单号；缺该键时返回 None，不会抛 KeyError
    user_query = state["user_query"]  # 取用户原始查询（本节点未直接使用，保留原变量）
    retry_count = state.get("retry_count", 0)  # 取重试次数，缺省 0 表示首次进入

    if order_id == "无" or not order_id:  # 没有可用订单号：模型回"无"或值为空
        if retry_count > 0:  # 已重试过一次仍拿不到订单号 → 走兜底话术
            # 兜底回复：说明仍查不到，并引导联系人工，避免无限重试
            response = "抱歉，我仍然无法找到您的订单号。请确认订单号是否正确，或联系人工客服进一步协助。"
        else:  # 首次缺订单号：请用户补充
            response = "为了帮你查询订单状态，请提供你的订单号（6-12位数字或字母+数字组合）。"
    else:  # 有订单号：走"查询订单状态"的模拟逻辑
        # 模拟调用订单查询API
        mock_order_status = f"订单{order_id}当前状态：已发货，物流单号：SF123456789，预计明日送达。"
        response = f"你好！{mock_order_status} 若有其他需求，请随时告知。"  # 包成客服口吻
        # mock_order_status 是模拟的第三方接口返回，response 把它拼进完整回复

    print(f"订单问题响应：{response}")  # 打印本次响应，便于跟踪
    # 返回：写回 response，并把 retry_count +1，供校验节点判断是否已达重试上限
    return {**state, "response": response, "retry_count": retry_count + 1}


# ------------------------------------------------------------
# 4.3 产品咨询节点：handle_product
# ------------------------------------------------------------
def handle_product(state: SupportState) -> SupportState:  # 节点 3：处理"产品咨询"分支
    """处理产品咨询：生成产品说明

    输入：
        state["user_query"] —— 用户咨询内容
        state["retry_count"] —— 已重试次数，>1 时改用一句话简答
    输出：
        新状态：写入 response，并把 retry_count 加 1
    流程位置：
        分类节点的产品分支目标；也是"分类兜底"和产品分支重试的落点
    """
    user_query = state["user_query"]  # 取用户咨询内容
    retry_count = state.get("retry_count", 0)  # 取重试次数，缺省 0

    if retry_count > 1:  # 如果已经重试过，使用更简洁的回复
        # 简答 prompt 的分段含义：
        #   第 1 段：点明"用户再次咨询"，暗示这是重试轮次
        #   第 2 段：要求一句话简答，压低输出长度，提高"完整"通过率
        prompt = f"""
        用户再次咨询：{user_query}
        请用一句话简明扼要地回答这个问题。
        """
    else:  # 第一轮（或第一次重试）：给详细回答
        # 详情 prompt 的分段含义：
        #   第 1 段：设定"产品顾问"角色，并给出语言风格要求
        #   第 2 段：插入用户咨询原文
        #   第 3-5 段：三条回答结构要求（点明核心问题 / 分点说明 / 结尾引导）
        prompt = f"""
        作为产品顾问，请详细解答用户的产品咨询，语言通俗易懂，结构清晰。

        用户咨询：{user_query}
        回答要求：
        1. 先明确核心问题
        2. 分点说明（不超过3点）
        3. 结尾提供进一步帮助引导
        """

    response = llm.invoke(prompt).content  # 调用模型生成产品回答（此处保留首尾空白）
    print(f"产品咨询响应：{response}")  # 打印本次回答
    # 返回：写回 response，并把 retry_count +1
    return {**state, "response": response, "retry_count": retry_count + 1}


# ------------------------------------------------------------
# 4.4 投诉记录节点：handle_complaint_record
# ------------------------------------------------------------
def handle_complaint_record(state: SupportState) -> SupportState:  # 节点 4：投诉分支第 1 步
    """处理投诉建议：记录投诉内容（并行节点1）

    输入：
        state["user_query"] —— 用户投诉内容
    输出：
        原状态不变（只打印"已记录"，模拟写入数据库/工单系统）
    流程位置：
        classify_query 的投诉分支入口；本示例通过 add_edge 顺序接到
        handle_complaint_forward（原注释写作"并行节点1"，实际是串行执行）
    """
    user_query = state["user_query"]  # 取用户投诉原文
    # 模拟写入数据库
    print(f"已记录投诉建议：{user_query}")  # 打印"已记录"，模拟落库与建单
    return {**state}  # 只保留原状态：记录动作本身不修改任何状态字段


# ------------------------------------------------------------
# 4.5 投诉转人工节点：handle_complaint_forward
# ------------------------------------------------------------
def handle_complaint_forward(state: SupportState) -> SupportState:  # 节点 5：投诉分支第 2 步
    """处理投诉建议：转人工处理（并行节点2）

    输入：
        state["retry_count"] —— 已重试次数，>0 时用"升级处理"话术
    输出：
        新状态：写入 response（安抚话术），并把 retry_count 加 1
    流程位置：
        handle_complaint_record 的下游；也是投诉分支的重试落点
    """
    user_query = state["user_query"]  # 取用户投诉原文（本节点未直接使用，保留原变量）
    retry_count = state.get("retry_count", 0)  # 取重试次数，缺省 0

    if retry_count > 0:  # 重试轮次：话术升级，承诺主管跟进
        response = f"您好，您的投诉已升级处理。我们的高级客服主管将亲自跟进您的问题，并在6小时内联系您。"
    else:  # 首轮：标准安抚话术，承诺 24 小时内联系
        response = f"你好！你的反馈已收到，我们将在24小时内安排专属客服与你联系。感谢你的支持！"

    print(f"投诉转人工响应：{response}")  # 打印转人工话术
    # 返回：写回 response，并把 retry_count +1
    return {**state, "response": response, "retry_count": retry_count + 1}


# ------------------------------------------------------------
# 4.6 响应校验节点：validate_response
# ------------------------------------------------------------
def validate_response(state: SupportState) -> SupportState:  # 节点 6：判断响应是否可用
    """校验响应是否完整，决定是否重试

    输入：
        state["response"] —— 上游处理节点生成的回复
        state["user_query"] —— 用户原始查询（作为评判依据）
        state["query_type"] —— 问题类型（投诉建议直接放行）
        state["retry_count"] —— 已重试次数（>=2 强制放行）
    输出：
        新状态：写入 is_complete（True 结束 / False 触发重试）
    流程位置：
        三个处理节点的汇合点；其 is_complete 由 route_after_validation 决定走向
    """
    response = state.get("response", "")  # 取回复文本，缺省空字符串
    user_query = state["user_query"]  # 取用户原始查询
    query_type = state.get("query_type", "")  # 取问题类型，缺省空字符串
    retry_count = state.get("retry_count", 0)  # 取重试次数，缺省 0

    # 对于投诉建议，默认认为已处理
    if query_type == "投诉建议":  # 投诉分支不走 LLM 校验：记录 + 转人工即视为完成
        print(f"投诉建议已记录并转人工，跳过校验")  # 打印"跳过校验"的说明
        return {  # 直接返回完成状态，避免投诉分支进入重试循环
            **state,  # 保留旧字段（含 response / retry_count）
            "is_complete": True  # 标记完成 → 路由到 END
        }

    # 如果重试次数超过2次，强制结束
    if retry_count >= 2:  # 达到重试上限：即使模型判"不完整"也结束，防止死循环
        print(f"已达到最大重试次数({retry_count})，强制结束流程")  # 打印强制结束原因
        return {  # 返回完成状态
            **state,  # 保留旧字段
            "is_complete": True  # 标记完成 → 路由到 END
        }

    # 校验 prompt 的分段含义：
    #   第 1 段：说明任务是"判断响应是否完整解答了用户查询"
    #   第 2 段：给出判定为"完整"的条件（正面清单，含"要求补充信息"也算完整）
    #   第 3 段：给出判定为"不完整"的条件（模糊、答非所问、遗漏关键信息）
    #   第 4-5 段：插入用户查询与系统响应，作为评判材料
    #   第 6 段：约束只返回"完整"或"不完整"，便于用 == 判断
    prompt = f"""
    请判断以下响应是否完整解答了用户的查询：
    1. 若响应直接回答了核心问题，或提供了明确的下一步操作指引（如要求补充信息），则判定为完整，返回"完整"
    2. 若响应模糊、未解答核心问题、遗漏关键信息，则判定为不完整，返回"不完整"

    用户查询：{user_query}
    系统响应：{response}
    仅返回"完整"或"不完整"。
    """

    try:  # 校验调用可能因网络/接口问题失败，这里做异常兜底
        validation_result = llm.invoke(prompt).content.strip()  # 模型给出的校验结论
        print(f"响应校验结果：{validation_result}")  # 打印校验结论
    except Exception as e:  # 捕获任何异常（超时、鉴权失败、连接错误等）
        print(f"校验过程出错，自动判定为完整：{e}")  # 打印出错原因
        validation_result = "完整"  # 出错即放行，避免因接口问题陷入无意义的重试

    # 用布尔表达式把字符串结论映射成 is_complete：只有恰好"完整"才算完成
    return {  # 返回更新后的状态
        **state,  # 保留旧字段
        "is_complete": validation_result == "完整"  # False 时由路由函数安排重试
    }


# ============================================================
# 5. 构建工作流图：初始化与节点注册
# ============================================================
# 初始化状态图
graph = StateGraph(SupportState)  # 以 SupportState 为状态类型创建图构建器（尚不可执行）

# 添加核心节点
# add_node(节点名, 节点函数)：节点名用于定义边和调试显示，函数是真正的执行体
graph.add_node("classify_query", classify_query)  # 问题分类
graph.add_node("handle_order", handle_order)  # 订单处理
graph.add_node("handle_product", handle_product)  # 产品咨询处理
graph.add_node("handle_complaint_record", handle_complaint_record)  # 投诉记录
graph.add_node("handle_complaint_forward", handle_complaint_forward)  # 投诉转人工
graph.add_node("validate_response", validate_response)  # 响应校验


# ============================================================
# 6. 路由函数与条件边
# ============================================================
# 路由函数只读取状态、返回"下一个节点名"（或 END）；它不是节点，不修改状态


# ------------------------------------------------------------
# 6.1 分类后的路由：route_after_classify
# ------------------------------------------------------------
# 从分类节点根据query_type分流到对应处理节点
def route_after_classify(state: SupportState) -> str:  # 返回下一个节点的名字
    """按 query_type 决定分类节点的后继节点。

    输入：
        state["query_type"] —— classify_query 写入的分类结果
    输出：
        节点名 "handle_order" / "handle_product" / "handle_complaint_record"
    流程位置：
        作为 add_conditional_edges 的路由函数，在 classify_query 执行后被调用
    """
    query_type = state.get("query_type", "")  # 取分类结果；缺键时给空字符串兜底
    if query_type == "订单问题":  # 订单类 → 订单处理节点
        return "handle_order"
    elif query_type == "产品咨询":  # 产品类 → 产品咨询节点
        return "handle_product"
    elif query_type == "投诉建议":  # 投诉类 → 先记录
        return "handle_complaint_record"  # 先执行记录，再由顺序边转人工（原注释称"并行"，实际是串行）
    else:  # 分类异常（模型返回了预期外文本）→ 兜底
        return "handle_product"  # 默认走产品咨询


# 添加条件边：分类节点 -> 处理节点
# add_conditional_edges(源节点, 路由函数)：运行时先调用路由函数，再跳到它返回的节点
graph.add_conditional_edges(
    "classify_query",  # 源节点：分类完成之后
    route_after_classify  # 路由函数：根据 query_type 返回目标节点名
)

# 投诉记录后，指向转人工节点
graph.add_edge("handle_complaint_record", "handle_complaint_forward")  # 记录 → 转人工（顺序边）

# 所有处理节点都指向校验节点
graph.add_edge("handle_order", "validate_response")  # 订单处理完 → 校验
graph.add_edge("handle_product", "validate_response")  # 产品咨询完 → 校验
graph.add_edge("handle_complaint_forward", "validate_response")  # 投诉转人工后 → 校验


# ------------------------------------------------------------
# 6.2 校验后的路由：route_after_validation
# ------------------------------------------------------------
# 校验节点的动态路由：完整则结束，不完整则重试对应处理节点
def route_after_validation(state: SupportState) -> str:  # 返回 END 或重试节点名
    """按 is_complete 决定校验节点的后继：END 或回到原处理节点重试。

    输入：
        state["is_complete"] —— validate_response 写入的完成标记
        state["query_type"] —— 用于决定重试时应回到哪个处理节点
    输出：
        END（结束整个工作流）或对应处理节点名（重试）
    流程位置：
        作为 add_conditional_edges 的路由函数，在 validate_response 执行后被调用
    """
    if state.get("is_complete", False):  # 响应完整 → 结束整个工作流
        return END
    else:  # 不完整 → 回到该类型对应的处理节点再跑一遍
        # 不完整则重试原处理节点
        query_type = state.get("query_type", "")  # 重新取分类，决定重试哪个节点
        if query_type == "订单问题":  # 订单类重试订单节点
            return "handle_order"
        elif query_type == "产品咨询":  # 产品类重试产品节点
            return "handle_product"
        elif query_type == "投诉建议":  # 投诉类重试转人工节点
            return "handle_complaint_forward"
        else:  # 兜底：走产品节点（与分类兜底保持一致）
            return "handle_product"


# 添加条件边：校验节点 -> 结束或重试
graph.add_conditional_edges(
    "validate_response",  # 源节点：校验完成之后
    route_after_validation  # 路由函数：完整→END，不完整→对应处理节点
)


# ============================================================
# 7. 设置入口点并编译图
# ============================================================
graph.set_entry_point("classify_query")  # 指定入口：一次 invoke 从分类节点开始执行

# 编译图（生成可执行工作流）
# compile() 会检查边的完整性（例如节点有进无出会在编译期报错），返回可 invoke 的对象
app = graph.compile()  # 编译后得到可执行对象 app


# ============================================================
# 8. 测试函数与执行入口
# ============================================================
# 测试运行：输入不同类型的用户查询
def test_workflow(user_query: str):
    """执行单个用户查询并打印结果。

    输入：
        user_query —— 用户查询文本
    输出：
        app.invoke 返回的最终状态字典（含 response / retry_count 等字段）
    流程位置：
        测试辅助函数，内部调用编译好的 app，真正跑完整张图
    """
    print(f"\n{'=' * 60}")  # 打印 60 个 "=" 组成的分隔线
    print(f"处理用户查询：{user_query}")  # 打印本轮输入
    print(f"{'=' * 60}")  # 结束分隔线
    # 运行工作流
    # invoke 传入初始状态：未分类、无订单号、无响应、未完成、重试 0 次
    result = app.invoke({  # 同步执行整张图，返回最终状态
        "user_query": user_query,  # 用户查询
        "query_type": None,  # 初始未分类
        "order_id": None,  # 初始无订单号
        "response": None,  # 初始无响应
        "is_complete": False,  # 初始未完成
        "retry_count": 0  # 初始重试计数为 0
    })
    print(f"\n最终响应：{result.get('response')}")  # 打印最终响应（缺键则为 None）
    print(f"重试次数：{result.get('retry_count', 0)}")  # 打印重试次数，缺省 0
    print(f"{'=' * 60}")  # 收尾分隔线
    return result  # 返回最终状态，供 run_all_tests 统计


# 测试函数，运行所有测试用例
def run_all_tests():
    """运行所有测试用例（5 个，覆盖三条分支与边界情况）。

    输入：
        无（用例写死在下方 test_cases 列表里）
    输出：
        无返回值；逐个打印中间过程，最后打印成功用例汇总
    流程位置：
        脚本入口调用的测试驱动，内部循环调用 test_workflow
    """
    # 下面 5 个用例分别覆盖：
    #   1) 订单问题且带订单号 → 走 handle_order 的"有订单号"分支
    #   2) 产品咨询 → 走 handle_product 的详情版 prompt
    #   3) 投诉建议 → 依次跑 record + forward，validate 直接放行
    #   4) 订单问题但没带订单号 → 走"缺订单号"分支，可能触发重试
    #   5) 简短产品咨询 → 走产品分支（模型可能判不完整而重试）
    # 每个用例是 (查询文本, 用例说明) 二元组
    test_cases = [
        ("我的订单号是OD123456，请问什么时候发货？", "订单问题"),
        ("你们的智能音箱支持蓝牙5.0吗？怎么连接手机？", "产品咨询"),
        ("我对昨天收到的商品不满意，质量有问题，希望退款或换货。", "投诉建议"),
        ("我的订单什么时候能到？", "订单问题但没提供订单号"),
        ("我想问一下这个产品怎么用", "简短的产品咨询")
    ]

    results = []  # 收集成功执行的用例结果（失败的不会追加进来）
    for i, (query, description) in enumerate(test_cases, 1):  # 从 1 开始编号遍历用例
        print(f"\n{'#' * 60}")  # 打印 60 个 "#" 的用例分隔线
        print(f"测试 {i}: {description}")  # 打印用例编号与说明
        print(f"查询内容: {query}")  # 打印该用例的查询内容
        print(f"{'#' * 60}")  # 结束分隔线

        try:  # 单个用例异常不应中断整个测试套件
            result = test_workflow(query)  # 跑完整工作流
            results.append((query, description, result))  # 记录成功结果
        except Exception as e:  # 捕获该用例的任何异常（含 LLM 接口错误）
            print(f"测试失败: {e}")  # 打印异常信息
            import traceback  # 函数内导入 traceback：保持原代码结构
            traceback.print_exc()  # 打印完整堆栈，便于排查

    print(f"\n{'=' * 60}")  # 汇总分隔线
    print(f"所有测试完成，共运行 {len(results)} 个测试用例")  # 打印成功用例数
    print(f"{'=' * 60}")  # 收尾分隔线


# 测试用例
if __name__ == "__main__":  # 只有直接运行本文件时才执行（被 import 时不执行）
    run_all_tests()  # 依次运行 5 个测试用例
    # 也可以单独测试
    # test_workflow("我的订单号是OD123456，请问什么时候发货？")
    # test_workflow("你们的智能音箱支持蓝牙5.0吗？怎么连接手机？")
    # test_workflow("我对昨天收到的商品不满意，质量有问题，希望退款或换货。")
    # test_workflow("我的订单什么时候能到？")
    # test_workflow("我想问一下这个产品怎么用")