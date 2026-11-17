# -*- coding: utf-8 -*-
r"""
03RAG检索/RAG.py —— LangGraph + FAISS 的 RAG（检索增强生成）完整案例

       + 本章（03RAG检索）可用的真实模型配置（DeepSeek 网关 + 火山方舟豆包向量）
       + 工程化补强（依赖自检、打印编码、失败重试、索引缓存、离线模拟模式、环境自检）

【模型配置】
  大模型   : deepseek-v4.1-flash-1017   （OpenAI 兼容网关，见 .env 的 LLM_BASE_URL / LLM_MODEL）
  向量模型 : doubao-embedding-text-240715（火山方舟豆包文本向量；若当前账号未开通该模型，
             会自动按 EMBEDDING_FALLBACKS 回退到可用的豆包向量模型，日志里会明确提示）

【运行方式】（在 03RAG检索 目录下执行；解释器需已安装依赖，脚本会自检并提示）
  python RAG.py --doctor     # 环境自检：解释器 / 依赖 / Key / 大模型 / 向量模型 / PDF / 索引
  python RAG.py              # 构建（或复用）向量库 + 跑内置测试案例
  python RAG.py "你的问题"   # 直接问一个问题（可传多个问题）
  python RAG.py --rebuild    # 忽略缓存，强制重建 FAISS 索引
  python RAG.py --mock       # 离线模拟模式：不调用任何 API，用假模型 + 假向量演示整条 RAG 流程
  python RAG.py --top-k 5 "问题"   # 每次检索的文本块数量（默认 3）

【整体流程】步骤 1 配置 -> 步骤 2 大模型 -> 步骤 3 向量模型 -> 步骤 4 私有知识库
          -> 步骤 5 状态与节点 -> 步骤 6 流程图与测试
"""

from __future__ import annotations

# ============================================================
# 步骤 0：环境兜底（必须在导入第三方库之前执行）
# ============================================================
import argparse  # 解析命令行参数（--doctor / --rebuild / --mock / 问题列表）
import hashlib  # 生成模拟向量时用的哈希算法（保证同样的文本得到同样的向量）
import importlib  # 动态导入模块，用来自检依赖是否装齐
import importlib.metadata  # 读取已安装包的版本号，自检时打印
import json  # 读写索引元信息 meta.json
import os  # 读取环境变量
import sys  # 拿到解释器路径、标准输出流、退出码
import time  # 计时、重试等待、生成时间戳
import warnings  # 屏蔽第三方库的弃用警告
from pathlib import Path  # 跨平台路径处理，避免相对路径受工作目录影响
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict  # 类型标注


def _fix_console_encoding() -> None:
    """Windows 控制台默认 GBK，打印 ✅📄🔍 这类 emoji 会抛 UnicodeEncodeError，这里放宽错误处理。"""
    for stream_name in ("stdout", "stderr"):  # 标准输出、标准错误都处理
        stream = getattr(sys, stream_name, None)  # 取流对象（某些环境可能是 None）
        if stream is None:  # pythonw / 部分 IDE 没有标准流，直接跳过
            continue
        try:
            stream.reconfigure(errors="replace")  # 保留原编码，只把「无法编码」的字符替换掉，避免崩溃
        except Exception:  # noqa: BLE001 有些流对象不支持 reconfigure，忽略即可
            pass


_fix_console_encoding()  # 尽早执行，保证后面所有 print 都不会因为编码崩掉
warnings.filterwarnings("ignore", category=DeprecationWarning)  # 屏蔽 langchain-community 的 sunset 提示


# ------------------------------------------------------------
# 步骤 0.1：依赖检查
# 缺依赖时不要抛一长串 ImportError，而是直接告诉用户「哪个解释器 + 怎么装」
# ------------------------------------------------------------
_REQUIRED_IMPORTS = {  # 模块名 -> pip 包名（两者不一定相同）
    "dotenv": "python-dotenv",  # 读取 .env
    "openai": "openai",  # 调用 OpenAI 兼容网关
    "httpx": "httpx",  # 调方舟多模态向量接口（SDK 没覆盖）
    "langchain_community": "langchain-community",  # PDF 加载器、FAISS 向量库
    "langchain_core": "langchain-core",  # Embeddings 基类
    "langchain_text_splitters": "langchain-text-splitters",  # 文本分割器
    "langgraph": "langgraph",  # 状态图
    "faiss": "faiss-cpu",  # FAISS 向量检索库
}

_missing: List[str] = []  # 收集缺失的 pip 包名
for _mod, _pkg in _REQUIRED_IMPORTS.items():  # 逐个尝试导入
    try:
        importlib.import_module(_mod)  # 能导入说明已安装
    except Exception:  # noqa: BLE001 任何导入异常都算缺失
        _missing.append(_pkg)  # 记录缺失的包名

if _missing:  # 有缺失就把安装方法打印出来，然后退出
    _pkgs = " ".join(sorted(set(_missing)))  # 去重后的包名，拼成一行命令参数
    print("❌ 缺少依赖：" + ", ".join(sorted(set(_missing))))  # 告诉用户缺什么
    print(f"   当前解释器：{sys.executable}")  # 明确是哪个 python 缺依赖
    print("   方式一：装到当前解释器（复制这一行执行）")  # 方案一：就地安装
    print(
        f'   "{sys.executable}" -m pip install --index-url https://mirrors.aliyun.com/pypi/simple'
        f' --trusted-host mirrors.aliyun.com {_pkgs}'
    )
    print("   方式二：或改用另一个已装齐依赖的解释器运行 RAG.py")  # 方案二：换解释器
    raise SystemExit(1)  # 缺依赖无法继续，退出码 1


# 依赖确认存在后再正式导入
from dotenv import load_dotenv  # 从 .env 文件加载环境变量到 os.environ

from openai import OpenAI  # OpenAI SDK，用兼容模式调用大模型网关

import httpx  # HTTP 客户端，调用方舟的多模态向量接口

from langchain_community.document_loaders import PyPDFLoader  # 把 PDF 一页页读成 Document

from langchain_text_splitters import RecursiveCharacterTextSplitter  # 按字符递归切分长文本

from langchain_community.vectorstores import FAISS  # FAISS 向量库（支持相似度检索）

from langchain_core.embeddings import Embeddings  # LangChain 嵌入模型抽象基类

from langgraph.graph import StateGraph, END  # LangGraph 状态图 + 终止节点


# ============================================================
# 步骤 1：路径与配置（导入依赖后的初始化配置）
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent  # 当前脚本所在目录：03RAG检索/
PROJECT_ROOT = SCRIPT_DIR.parent  # 项目根目录：langgraph-tutorial/
DEFAULT_INDEX_DIR = SCRIPT_DIR / "faiss_rag_index"  # 真实模型的索引目录
MOCK_INDEX_DIR = SCRIPT_DIR / "faiss_rag_index_mock"  # 模拟模式的索引目录（和真实索引隔离，避免维度冲突）

# 先加载仓库根目录 .env，再加载本章 .env；本章 .env 覆盖根目录的同名变量
for _env_file in (PROJECT_ROOT / ".env", SCRIPT_DIR / ".env"):  # 两个位置都尝试加载
    if _env_file.exists():  # 文件存在才加载
        load_dotenv(_env_file, override=(_env_file == SCRIPT_DIR / ".env"))  # 本章配置优先级更高


def _env(*names: str, default: Optional[str] = None) -> Optional[str]:
    """按顺序取第一个非空环境变量（兼容 DASHSCOPE_API_KEY / QWEN_API_KEY 等多种历史命名）。"""
    for name in names:  # 按优先级逐个尝试
        value = os.getenv(name)  # 读取环境变量
        if value and value.strip():  # 非空才算有效
            return value.strip()  # 去掉首尾空白后返回
    return default  # 全都没有则返回默认值


# ---------- 大模型配置 ----------
LLM_BASE_URL = _env("LLM_BASE_URL", "OPENAI_BASE_URL", default="https://tokenrhythm.studio/v1")  # 网关地址
LLM_MODEL = _env("LLM_MODEL", "OPENAI_MODEL", default="deepseek-v4.1-flash-1017")  # 模型名称
LLM_API_KEY = _env(  # 大模型 Key：兼容多种历史变量名
    "LLM_API_KEY",  # 本脚本推荐的名字
    "deepseek_API_KEY",  # 仓库根目录 .env 里已有的名字（小写 deepseek）
    "DEEPSEEK_API_KEY",  # 章节模板里的写法
    "OPENAI_API_KEY",  # OpenAI SDK 通用名字
)

# ---------- 向量模型配置（火山方舟 Ark） ----------
ARK_BASE_URL = _env("ARK_BASE_URL", default="https://ark.cn-beijing.volces.com/api/v3")  # 方舟接口地址
ARK_API_KEY = _env("ARK_API_KEY", "VOLC_ARK_API_KEY")  # 方舟 Key
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", default="doubao-embedding-text-240715")  # 首选豆包文本向量
EMBEDDING_FALLBACKS: Tuple[str, ...] = tuple(  # 备用向量模型（逗号分隔）
    name.strip()  # 去掉每项首尾空白
    for name in (
        _env("EMBEDDING_FALLBACKS", default="doubao-embedding-large-text-250515,doubao-embedding-vision-251215") or ""
    ).split(",")  # 按逗号拆开
    if name.strip()  # 过滤空项
)
EMBEDDING_BATCH_SIZE = int(_env("EMBEDDING_BATCH_SIZE", default="10") or 10)  # 文本向量接口每批条数

# ---------- 文本切分参数 ----------
CHUNK_SIZE = 1000  # 每个文本块最大字符数
CHUNK_OVERLAP = 200  # 相邻块重叠字符数，避免一句话被切断导致语义丢失

# ---------- 内置测试案例（覆盖「需要检索」和「不需要检索」两条分支） ----------
TEST_QUESTIONS = [  # 默认跑这 4 个问题，覆盖「需要检索」和「不需要检索」两种分支
    "这份文档主要讲了什么？",  # 需要检索：文档主题
    "大会在什么时间、什么地点举办？",  # 需要检索：文档细节
    "Python 是什么类型的编程语言？",  # 不需要检索：通用知识（示例测试案例 2）
    "LangGraph 的核心组件有哪些？",  # 示例测试案例 1（用来验证「知识库无相关信息」的兜底）
]

# 模型判断返回空时用的兜底关键词：命中说明大概率要查私有文档
PRIVATE_DOC_KEYWORDS = ("文档", "本文", "报告", "PDF", "pdf", "资料", "知识库", "文件", "附件", "手册", "通知")

# 模拟模式（无 PDF / 无 Key 时）使用的示例文本块
MOCK_CHUNKS = [
    "LangChain 是一个用于开发语言模型应用的框架。",  # 示例块 1
    "RAG（检索增强生成）技术结合了信息检索和文本生成。",  # 示例块 2
    "通义千问是阿里巴巴推出的大型语言模型。",  # 示例块 3
    "本系统使用 FAISS 向量数据库存储文档嵌入。",  # 示例块 4
    "LangGraph 用于构建有状态的多智能体工作流。",  # 示例块 5
]


def mask(secret: Optional[str]) -> str:
    """打印日志时只显示 Key 的头尾，避免密钥泄露。"""
    if not secret:  # 没配置
        return "<未配置>"  # 返回醒目的占位符
    if len(secret) <= 10:  # 太短的 Key 只露前两位
        return secret[:2] + "***"
    return f"{secret[:6]}...{secret[-4:]}"  # 常规情况：头 6 位 + 尾 4 位


# ============================================================
# 步骤 2：大模型封装（OpenAI 兼容 chat.completions）
# ============================================================
class ChatModel:
    """真实大模型：对 OpenAI SDK 的 chat.completions 做一层薄封装，带失败重试。"""

    RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}  # 这些状态码值得重试（限流/服务端抖动）

    def __init__(
        self,
        api_key: str,  # 模型密钥
        base_url: str = LLM_BASE_URL,  # 接口地址
        model: str = LLM_MODEL,  # 模型名
        timeout: float = 120.0,  # 单次请求超时（秒）
        max_retries: int = 3,  # 最多尝试次数
    ) -> None:
        self.model = model  # 保存模型名
        self.max_retries = max(1, max_retries)  # 至少重试 1 次
        self.base_url = base_url.rstrip("/")  # 去掉末尾斜杠，避免拼出双斜杠
        # SDK 自带重试关掉，统一由下面的 _should_retry 控制，日志更清楚
        self.client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=timeout, max_retries=0)

    @classmethod
    def _should_retry(cls, exc: BaseException) -> bool:
        """判断异常是否值得重试（限流 / 5xx / 网络类）。"""
        status = getattr(exc, "status_code", None)  # 有些异常带 HTTP 状态码
        if isinstance(status, int) and status in cls.RETRYABLE_STATUS:  # 命中可重试状态码
            return True
        name = type(exc).__name__  # 退而求其次：按异常类名判断
        return name in {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError", "TimeoutError", "ConnectionError"}

    def invoke(
        self,
        prompt: str,  # 提示词
        *,
        temperature: float = 0.1,  # 温度：越低越稳定
        system_prompt: Optional[str] = None,  # 可选的 system 提示词
        max_tokens: Optional[int] = None,  # 可选的最大输出 token 数
    ) -> str:
        """输入提示词，返回模型生成的文本。"""
        messages: List[Dict[str, str]] = []  # 组装 OpenAI 格式的 messages
        if system_prompt:  # 有 system 提示词就先放进去
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})  # 用户消息
        kwargs: Dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}  # 请求参数
        if max_tokens:  # 指定了输出上限才带上，避免未知模型不认
            kwargs["max_tokens"] = max_tokens

        last_exc: Optional[BaseException] = None  # 记录最后一次异常，循环结束后统一抛出
        for attempt in range(1, self.max_retries + 1):  # 最多尝试 max_retries 次
            try:
                response = self.client.chat.completions.create(**kwargs)  # 真正发起 /chat/completions 请求
                return (response.choices[0].message.content or "").strip()  # 取出回答文本并去掉首尾空白
            except Exception as exc:  # noqa: BLE001 统一在这里转成可读错误
                last_exc = exc  # 记住异常
                if attempt >= self.max_retries or not self._should_retry(exc):  # 到次数上限 / 不该重试
                    break  # 跳出循环，抛错
                wait = 2 ** (attempt - 1)  # 指数退避：1s、2s、4s
                print(f"⚠️ 大模型调用失败（第 {attempt} 次）：{type(exc).__name__}: {str(exc)[:120]}，{wait}s 后重试")
                time.sleep(wait)  # 等待后重试
        raise RuntimeError(f"大模型调用失败：{type(last_exc).__name__}: {str(last_exc)[:300]}")  # 最终失败

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """支持 llm("...") 的调用写法。"""
        return self.invoke(prompt, **kwargs)  # 直接转给 invoke


class MockChatModel:
    """模拟大模型：不联网，按提示词关键词返回固定答案，用于离线演示整条流程。"""

    def __init__(self, model: str = "mock-llm") -> None:
        self.model = model  # 假的模型名，仅用于打印

    @staticmethod
    def _question_of(prompt: str) -> str:
        """从提示词里取出真正的「用户问题」部分（判断节点 / 生成节点的提示词都用「问题：」分隔）。"""
        marker = "问题："  # 提示词里的分隔标记
        if marker in prompt:  # 有标记就取最后一段（生成节点会把知识库上下文放在前面）
            return prompt.rsplit(marker, 1)[1].strip()
        return prompt.strip()  # 没有标记就把整个提示词当问题

    def invoke(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,  # 模拟模式忽略温度
        system_prompt: Optional[str] = None,  # 模拟模式忽略 system 提示词
        max_tokens: Optional[int] = None,  # 模拟模式忽略 token 上限
    ) -> str:
        """按「用户问题」里的关键词返回预设答案（离线演示逻辑）。"""
        question = self._question_of(prompt)  # 只分析用户问题，避免被系统提示词里的示例词干扰
        if "判断助手" in prompt:  # 判断节点：按问题里是否出现文档类关键词决定 True/False
            needs_doc = any(key in question for key in ("文档", "PDF", "报告", "资料", "知识库", "文件", "大会", "LangGraph"))
            return "True" if needs_doc else "False"
        if "这份文档主要讲了什么" in question:  # 文档主题问题
            return "这是一份关于 RAG 检索增强生成系统的演示文档，介绍了知识库构建、向量检索与 LangGraph 工作流。"
        if "大会" in question and ("时间" in question or "地点" in question or "举办" in question):  # 文档细节问题
            return "根据知识库，大会于 2025 年 11 月 28—30 日在厦门举办，地点是厦门国际会议中心。"
        if "LangGraph" in question and "核心组件" in question:  # 框架组件问题（模拟知识库里有 LangGraph 例句）
            return "LangGraph 的核心组件包括 StateGraph（状态图）、节点（Node）、边（Edge）与条件边（Conditional Edge），以及编译后的可执行图（CompiledGraph）。"
        if "Python" in question and "编程语言" in question:  # 通用知识问题
            return "Python 是一种高级、解释型、动态类型的通用编程语言，广泛用于数据分析、人工智能、Web 开发和自动化脚本。"
        return "（模拟模式）已收到问题，接入真实大模型后即可给出完整回答。"  # 其他问题的兜底回答

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """支持 llm("...") 的调用写法。"""
        return self.invoke(prompt, **kwargs)  # 转给 invoke


def build_llm(use_mock: bool) -> Any:
    """按运行模式创建大模型对象：模拟模式用 MockChatModel，否则用真实 ChatModel。"""
    if use_mock:  # 离线模式
        print("🤖 大模型：MockChatModel（模拟模式，不联网）")  # 告知用户当前是假模型
        return MockChatModel()  # 返回模拟对象
    print(f"🤖 大模型：{LLM_MODEL} @ {LLM_BASE_URL}（key={mask(LLM_API_KEY)}）")  # 打印真实配置
    return ChatModel(LLM_API_KEY or "")  # 返回真实对象


# ============================================================
# 步骤 3：向量模型封装（火山方舟豆包向量）
# ============================================================
class EmbeddingError(RuntimeError):
    """向量接口返回的业务错误（Key 无效、模型未开通、参数错误等）。"""


class ArkEmbeddings(Embeddings):
    """
    豆包向量模型封装（继承 LangChain 的 Embeddings 基类，实现 embed_documents / embed_query）。

    火山方舟有两套向量接口：
      - 文本向量：POST {base_url}/embeddings                    如 doubao-embedding-text-240715（一批多条）
      - 多模态向量：POST {base_url}/embeddings/multimodal        如 doubao-embedding-vision-251215（一次一条）
    本类会自动探测当前账号能用哪一个，并把结果缓存下来，之后所有请求都走同一个接口。
    """

    TEXT_API = "/embeddings"  # 文本向量接口路径
    MULTIMODAL_API = "/embeddings/multimodal"  # 多模态向量接口路径
    RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}  # 这些状态码值得重试

    def __init__(
        self,
        api_key: str,  # 方舟 Key
        base_url: str = ARK_BASE_URL,  # 方舟接口地址
        model: str = EMBEDDING_MODEL,  # 首选向量模型
        fallbacks: Sequence[str] = EMBEDDING_FALLBACKS,  # 备用向量模型
        batch_size: int = EMBEDDING_BATCH_SIZE,  # 文本向量每批条数
        timeout: float = 60.0,  # 单次请求超时
        max_retries: int = 4,  # 最多尝试次数
    ) -> None:
        super().__init__()  # 调用父类初始化
        self.base_url = base_url.rstrip("/")  # 去掉末尾斜杠
        self.model = model  # 首选模型名
        self.fallbacks = tuple(fallbacks)  # 备用模型（去重前先转元组）
        self.batch_size = max(1, batch_size)  # 每批至少 1 条
        self.max_retries = max(1, max_retries)  # 至少尝试 1 次
        self._client = httpx.Client(  # 自建 HTTP 客户端（多模态向量接口 OpenAI SDK 没有）
            base_url=self.base_url,  # 基准地址，后面只写路径
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},  # 认证头
            timeout=httpx.Timeout(timeout, connect=15.0),  # 总超时 + 连接超时
        )
        self.active_model: Optional[str] = None  # 实际生效的模型名（首次探测后确定）
        self.active_mode: Optional[str] = None  # 实际生效的接口模式：text / multimodal
        self.dimension: Optional[int] = None  # 向量维度（首次探测后确定）
        self._errors: List[str] = []  # 探测失败原因，报错时一起打印，方便定位

    # ---------- 3.1 底层 HTTP 请求 ----------
    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """带重试的 POST；成功返回 JSON，业务错误抛 EmbeddingError。"""
        last_exc: Optional[BaseException] = None  # 记录最后一次异常
        for attempt in range(1, self.max_retries + 1):  # 最多尝试 max_retries 次
            try:
                response = self._client.post(path, json=payload)  # 发请求
            except Exception as exc:  # noqa: BLE001 网络异常也重试
                last_exc = exc  # 记住异常
                if attempt >= self.max_retries:  # 最后一次仍失败
                    break  # 退出循环
                time.sleep(2 ** (attempt - 1))  # 指数退避
                continue  # 进入下一次尝试

            if response.status_code == 200:  # 成功
                return response.json()  # 返回解析后的 JSON

            # 失败：解析方舟的错误结构，给出人话提示
            try:
                error = response.json().get("error", {})  # 方舟错误体是 {"error": {...}}
            except Exception:  # noqa: BLE001 返回体不是 JSON
                error = {"code": "", "message": response.text[:200]}  # 退化成文本
            code = str(error.get("code", ""))  # 错误码
            message = str(error.get("message", ""))  # 错误描述
            hint = ""  # 给用户的额外提示
            if response.status_code == 401:  # 认证失败
                hint = "（ARK_API_KEY 无效，请检查火山方舟控制台的 API Key）"
            elif code == "InvalidEndpointOrModel.NotFound":  # 模型不存在 / 无权限
                hint = "（当前方舟账号未开通该模型/没有访问权限）"
            elif response.status_code == 400 and "not support this api" in message:  # 接口用错了
                hint = "（该模型不支持文本向量接口，需要走 /embeddings/multimodal）"

            if response.status_code in self.RETRYABLE_STATUS and attempt < self.max_retries:  # 可重试
                wait = 2 ** (attempt - 1)  # 指数退避
                print(f"⚠️ 向量接口限流/服务端异常（{response.status_code}），{wait}s 后重试：{message[:80]}")
                time.sleep(wait)  # 等待
                last_exc = EmbeddingError(f"{response.status_code} {code}: {message}")  # 记下本次错误
                continue  # 进入下一次尝试
            raise EmbeddingError(f"{response.status_code} {code}: {message}{hint}")  # 不可重试，直接抛

        raise EmbeddingError(f"向量接口请求失败：{type(last_exc).__name__}: {str(last_exc)[:200]}")  # 重试耗尽

    # ---------- 3.2 模型能力探测 ----------
    @staticmethod
    def _mode_of(model: str) -> str:
        """带 vision 的模型只能走多模态向量接口，其余按文本向量处理。"""
        return "multimodal" if "vision" in model.lower() else "text"  # 按模型名判断接口

    def _probe_model(self, model: str) -> Tuple[str, int]:
        """试跑一次单条向量，成功返回 (模式, 维度)。"""
        mode = self._mode_of(model)  # 判断该用哪个接口
        if mode == "text":  # 文本向量接口
            payload = {"model": model, "input": ["向量维度探测"], "encoding_format": "float"}  # 请求体
            data = self._post(self.TEXT_API, payload)  # 发请求
            embedding = data["data"][0]["embedding"]  # 文本接口返回数组
        else:  # 多模态向量接口
            payload = {"model": model, "input": [{"type": "text", "text": "向量维度探测"}]}  # 请求体
            data = self._post(self.MULTIMODAL_API, payload)  # 发请求
            embedding = data["data"]["embedding"]  # 多模态接口的 data 是对象而不是数组
        return mode, len(embedding)  # 返回接口模式与向量维度

    def ensure_active(self) -> None:
        """确认可用的向量模型（只探测一次）；全部失败时给出可操作的错误信息。"""
        if self.active_model:  # 已经探测过就直接返回
            return
        candidates = [name for name in (self.model, *self.fallbacks) if name]  # 首选 + 备用
        self._errors.clear()  # 清空上次的探测记录
        for name in dict.fromkeys(candidates):  # 去重但保留顺序
            try:
                mode, dim = self._probe_model(name)  # 试跑一次
            except EmbeddingError as exc:  # 该模型不可用
                self._errors.append(f"  - {name} -> {exc}")  # 记录原因（可能是 Key/权限问题）
                continue  # 换下一个候选
            self.active_model, self.active_mode, self.dimension = name, mode, dim  # 记住可用配置
            note = "" if name == self.model else "（首选模型不可用，已自动回退）"  # 回退时特别提示
            print(f"🧮 向量模型就绪：{name}｜模式={mode}｜维度={dim}{note}")
            return  # 探测成功，结束
        detail = "\n".join(self._errors) if self._errors else "  - 未配置任何向量模型"  # 汇总失败原因
        raise EmbeddingError(  # 全部不可用时给出排查步骤
            "❌ 没有可用的豆包向量模型，请检查：\n"
            "   1) .env 里的 ARK_API_KEY 是否正确、该账号是否已开通向量模型；\n"
            "   2) 火山方舟控制台 -> 开通管理 里开通 doubao-embedding-text-240715；\n"
            "   3) 或把 EMBEDDING_FALLBACKS 换成账号已开通的向量模型。\n"
            f"探测记录：\n{detail}"
        )

    # ---------- 3.3 单条 / 批量向量化 ----------
    def _embed_one(self, text: str) -> List[float]:
        """单条文本向量化（两种接口都能走）。"""
        self.ensure_active()  # 确保已经确定可用模型
        if self.active_mode == "text":  # 文本向量接口
            data = self._post(self.TEXT_API, {"model": self.active_model, "input": [text], "encoding_format": "float"})
            return data["data"][0]["embedding"]  # 返回向量
        data = self._post(self.MULTIMODAL_API, {"model": self.active_model, "input": [{"type": "text", "text": text}]})
        return data["data"]["embedding"]  # 多模态接口返回对象

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量把文本块转成向量，返回值顺序与输入完全一致（LangChain 要求）。"""
        self.ensure_active()  # 确保已经确定可用模型
        if not texts:  # 空列表直接返回
            return []

        if self.active_mode == "multimodal":  # 多模态接口一次只返回一条向量
            vectors: List[List[float]] = []  # 收集结果
            for index, text in enumerate(texts, 1):  # 逐条处理
                vectors.append(self._embed_one(text))  # 单条向量化
                if index == len(texts) or index % 10 == 0:  # 每 10 条或最后一条打印进度
                    print(f"📄 已向量化 {index}/{len(texts)} 个文本块")
            return vectors  # 返回顺序一致的向量列表

        vectors = []  # 文本向量接口：按批处理
        for start in range(0, len(texts), self.batch_size):  # 每 batch_size 条一批
            batch = texts[start:start + self.batch_size]  # 取出一批
            data = self._post(  # 发一批
                self.TEXT_API,
                {"model": self.active_model, "input": batch, "encoding_format": "float"},
            )
            items = sorted(data["data"], key=lambda item: item.get("index", 0))  # 按 index 排序保证顺序
            vectors.extend(item["embedding"] for item in items)  # 收集本批向量
            print(f"📄 已向量化 {min(start + self.batch_size, len(texts))}/{len(texts)} 个文本块")  # 进度
        return vectors  # 返回全部向量

    def embed_query(self, query: str) -> List[float]:
        """把用户查询转成向量（检索时用）。"""
        return self._embed_one(query)  # 复用单条向量化

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain 异步接口，复用同步实现。"""
        return self.embed_documents(texts)

    async def aembed_query(self, query: str) -> List[float]:
        """LangChain 异步接口，复用同步实现。"""
        return self.embed_query(query)


class MockEmbeddings(Embeddings):
    """模拟向量模型：用 sha256 哈希生成确定性向量，不需要联网，用于 --mock 离线演示。"""

    DIMENSION = 256  # 模拟向量维度（比真实模型小，纯本地计算很快）

    def _vector(self, text: str) -> List[float]:
        """把文本映射成固定维度的向量：同样的文本永远得到同样的向量。"""
        digest = hashlib.sha256(text.encode("utf-8")).digest()  # 32 字节摘要
        values = [b / 255.0 for b in digest]  # 归一化到 0~1
        return (values * (self.DIMENSION // len(values) + 1))[: self.DIMENSION]  # 循环铺满目标维度

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量向量化。"""
        return [self._vector(text) for text in texts]  # 逐条映射

    def embed_query(self, query: str) -> List[float]:
        """查询向量化。"""
        return self._vector(query)  # 同样映射

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步接口复用同步实现。"""
        return self.embed_documents(texts)

    async def aembed_query(self, query: str) -> List[float]:
        """异步接口复用同步实现。"""
        return self.embed_query(query)


def build_embeddings(use_mock: bool) -> Any:
    """按运行模式创建向量模型对象。"""
    if use_mock:  # 离线模式
        print("🧮 向量模型：MockEmbeddings（模拟模式，本地哈希向量）")  # 告知用户
        return MockEmbeddings()  # 返回模拟对象
    print(f"🧮 向量模型：{EMBEDDING_MODEL} @ {ARK_BASE_URL}（key={mask(ARK_API_KEY)}）")  # 打印真实配置
    return ArkEmbeddings(ARK_API_KEY or "")  # 返回真实对象


# ============================================================
# 步骤 4：构建私有知识库（PDF -> 文本块 -> 向量 -> FAISS）
# ============================================================
def find_pdf() -> Path:
    """查找测试 PDF：固定使用与脚本同目录的 test.pdf。"""
    candidates = [  # 候选路径列表
        SCRIPT_DIR / "test.pdf",  # 与脚本同目录的 test.pdf（把 PDF 放这里即可）
    ]
    for path in candidates:  # 按优先级检查
        if path.exists():  # 找到就返回
            return path
    raise FileNotFoundError(  # 都没找到就报错，并列出应该放哪
        f"❌ 未找到 PDF，请把 test.pdf 放到以下任一位置：\n" + "\n".join(f"   - {p}" for p in candidates)
    )


def _pdf_fingerprint(pdf_path: Path) -> Dict[str, Any]:
    """PDF 指纹：文件名 + 大小 + 修改时间，用来判断索引能不能复用。"""
    stat = pdf_path.stat()  # 取文件状态
    return {"name": pdf_path.name, "size": stat.st_size, "mtime": int(stat.st_mtime)}  # 三元组指纹


def _read_meta(index_dir: Path) -> Optional[Dict[str, Any]]:
    """读取索引元信息 meta.json；不存在或损坏时返回 None。"""
    meta_file = index_dir / "meta.json"  # 元信息文件路径
    if not meta_file.exists():  # 不存在
        return None
    try:
        return json.loads(meta_file.read_text(encoding="utf-8"))  # 读 JSON
    except Exception:  # noqa: BLE001 文件损坏就当没有缓存
        return None


def _embedding_tag(embeddings: Any) -> str:
    """取向量模型的「标识名」，用来判断缓存索引是不是同一个模型建的。"""
    return getattr(embeddings, "active_model", None) or getattr(embeddings, "model", "mock")  # 真实模型用 active_model，模拟模型固定 mock


def _index_is_usable(pdf_path: Path, embeddings: Any, index_dir: Path) -> bool:
    """索引存在、且 PDF / 向量模型 / 切分参数都没变，才复用。"""
    meta = _read_meta(index_dir)  # 先读元信息
    if not meta or not (index_dir / "index.pickle").exists():  # 元信息或索引文件缺失
        return False
    if meta.get("pdf") != _pdf_fingerprint(pdf_path):  # PDF 变了
        return False
    if meta.get("embedding", {}).get("model") != _embedding_tag(embeddings):  # 向量模型变了
        return False
    if meta.get("chunk_size") != CHUNK_SIZE or meta.get("chunk_overlap") != CHUNK_OVERLAP:  # 切分参数变了
        return False
    return True  # 三项都一致，可以复用


def _save_store(vector_store: FAISS, index_dir: Path) -> None:
    """
    保存向量库：用 serialize_to_bytes 写成字节文件。

    不能直接调 vector_store.save_local(中文目录)：faiss 在 Windows 上用 C 的 fopen 写文件，
    路径里带中文（本目录名是「03RAG检索」）会报 “could not open ... for writing”。
    改成先 pickle 成 bytes，再用 Python 的 open 写文件，就绕开了编码问题。
    """
    index_dir.mkdir(parents=True, exist_ok=True)  # 目录必须先存在
    (index_dir / "index.pickle").write_bytes(vector_store.serialize_to_bytes())  # 写字节文件


def _load_store(embeddings: Any, index_dir: Path) -> FAISS:
    """读取 _save_store 保存的向量库。"""
    return FAISS.deserialize_from_bytes(  # 从字节反序列化
        (index_dir / "index.pickle").read_bytes(),  # 读回字节
        embeddings,  # 需要同一个向量模型对象（检索查询向量时才用）
        allow_dangerous_deserialization=True,  # 读取本地 pickle 必须显式允许
    )


def build_or_load_index(
    pdf_path: Path,  # PDF 路径
    embeddings: Any,  # 向量模型对象
    index_dir: Path = DEFAULT_INDEX_DIR,  # 索引导出目录
    rebuild: bool = False,  # 是否强制重建
    fallback_chunks: Sequence[str] = (),  # PDF 不存在时用的兜底文本块（离线示例数据）
) -> FAISS:
    """构建知识库：PDF -> 文本块 -> 向量 -> FAISS；优先复用本地缓存。"""
    if hasattr(embeddings, "ensure_active"):  # 真实向量模型需要先探测可用接口
        embeddings.ensure_active()

    if not rebuild and _index_is_usable(pdf_path, embeddings, index_dir):  # 缓存可用
        try:
            vector_store = _load_store(embeddings, index_dir)  # 读缓存
            meta = _read_meta(index_dir) or {}  # 读元信息
            print(f"♻️ 复用已保存的向量库：{index_dir}（{meta.get('chunks', '?')} 个文本块，模型 {meta.get('embedding', {}).get('model')}）")
            print("   如需重建请加参数 --rebuild")  # 提示怎么强制重建
            return vector_store  # 直接返回
        except Exception as exc:  # noqa: BLE001 缓存损坏 / faiss 版本不一致
            print(f"⚠️ 读取缓存索引失败（{type(exc).__name__}: {str(exc)[:160]}），改为重新构建")

    # ---------- 4.1 加载 PDF ----------
    if not pdf_path.exists():  # PDF 不存在：用兜底文本块（模拟模式）
        chunks = list(fallback_chunks) or list(MOCK_CHUNKS)  # 优先用调用方给的兜底文本
        print(f"⚠️ 未找到 PDF，改用内置示例文本 {len(chunks)} 条")  # 提示用户
        vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)  # 直接由文本建库（无 PDF 时的兜底）
        _save_store(vector_store, index_dir)  # 落盘缓存
        return vector_store  # 返回

    loader = PyPDFLoader(str(pdf_path))  # 创建 PDF 加载器（每页一个 Document）
    documents = loader.load()  # 真正读取 PDF
    print(f"✅ 加载 PDF：{pdf_path.name}，共 {len(documents)} 页")  # 日志

    # ---------- 4.2 切分文本块 ----------
    text_splitter = RecursiveCharacterTextSplitter(  # 递归字符分割器
        chunk_size=CHUNK_SIZE,  # 每块最大字符数
        chunk_overlap=CHUNK_OVERLAP,  # 相邻块重叠字符数
        separators=["\n\n", "\n", " ", ""],  # 优先按段落、换行、空格、字符分割
        length_function=len,  # 用字符数衡量长度
    )
    splits = text_splitter.split_documents(documents)  # 切分
    print(f"✅ 文本分割完成，共 {len(splits)} 个文本块")  # 日志

    # ---------- 4.3 向量化 + 建 FAISS 索引 ----------
    started = time.perf_counter()  # 开始计时
    vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)  # 批量向量化并建索引
    cost = time.perf_counter() - started  # 统计耗时

    # ---------- 4.4 保存索引 + 元信息 ----------
    _save_store(vector_store, index_dir)  # 保存索引文件
    meta: Dict[str, Any] = {  # 元信息：决定下次能不能复用
        "pdf": _pdf_fingerprint(pdf_path),  # PDF 指纹
        "embedding": {  # 向量模型信息
            "model": _embedding_tag(embeddings),  # 模型标识（真实模型是 active_model，模拟模型是 mock）
            "mode": getattr(embeddings, "active_mode", "mock"),  # 接口模式
            "dimension": getattr(embeddings, "dimension", getattr(embeddings, "DIMENSION", None)),  # 向量维度
        },
        "chunk_size": CHUNK_SIZE,  # 切分参数
        "chunk_overlap": CHUNK_OVERLAP,
        "chunks": len(splits),  # 文本块数量
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),  # 创建时间
    }
    (index_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")  # 写元信息
    print(f"✅ 向量库构建完成（耗时 {cost:.1f}s），已保存到：{index_dir}")  # 日志
    return vector_store  # 返回向量库


def simple_retrieve(vector_db: Optional[FAISS], query: str, k: int = 3, fallback_chunks: Sequence[str] = ()) -> List[str]:
    """简单检索：返回与问题最相关的 k 段文本。"""
    if vector_db is None:  # 没有向量库（理论上不会发生，防一手）
        return list(fallback_chunks[:k]) or ["无可用文档"]  # 退化成前 k 条兜底文本
    try:
        docs = vector_db.similarity_search(query, k=k)  # FAISS 相似度检索
        return [doc.page_content for doc in docs]  # 只要文本内容
    except Exception as exc:  # noqa: BLE001 检索失败时退化，避免整条流程中断
        print(f"⚠️ 检索失败：{exc}")
        return list(fallback_chunks[:k]) or ["无可用文档"]  # 兜底


# ============================================================
# 步骤 5：定义状态与节点（状态 + 判断/检索/生成三个节点）
# ============================================================
class AgentState(TypedDict):  # LangGraph 在节点之间传递的状态结构
    question: str  # 用户原始问题
    retrieved_docs: List[str]  # 检索到的文本块
    answer: str  # 最终回答
    need_retrieve: bool  # 是否需要检索


def build_agent(vector_db: FAISS, llm: Any, top_k: int = 3) -> Any:
    """根据向量库和大模型构建 RAG 代理图：判断 ->（检索）-> 生成。"""

    def judge_retrieve_node(state: AgentState) -> AgentState:
        """节点 1：判断问题是否需要查询私有知识库。"""
        prompt = f"""你是检索判断助手。请判断以下问题是否需要查询私有文档才能回答。
规则：
1. 问题涉及具体文档内容、文件、报告、PDF、私有资料时，返回 True
2. 通用知识、闲聊、常识问题，返回 False
3. 只返回 True 或 False，不要输出其他文字

问题：{state["question"]}"""
        question = state["question"]  # 取出问题，兜底时要用
        try:
            # 注意：不要限制 max_tokens，该模型会先用 token 做思考，限制太小会返回空字符串
            result = llm.invoke(prompt, temperature=0).strip().lower()  # 温度 0，输出更稳定
            if result:  # 模型给了明确答案
                state["need_retrieve"] = result.startswith("true")  # 以 True 开头才算需要检索
            else:  # 模型返回空时用关键词兜底，避免漏检索
                state["need_retrieve"] = any(key in question for key in PRIVATE_DOC_KEYWORDS)
                print("⚠️ 检索判断返回空，改用关键词兜底")
        except Exception as exc:  # noqa: BLE001 判断失败时保守处理
            print(f"⚠️ 检索判断异常：{exc}，默认执行检索")
            state["need_retrieve"] = True  # 宁可多检索，也不要漏检索
        print(f"📌 是否需要检索：{state['need_retrieve']}")  # 日志
        return state  # 返回更新后的状态

    def retrieve_node(state: AgentState) -> AgentState:
        """节点 2：从 FAISS 向量库检索相关文本块（调用 simple_retrieve）。"""
        state["retrieved_docs"] = simple_retrieve(vector_db, state["question"], k=top_k)  # 返回最相关的 k 段文本
        print(f"🔍 检索到 {len(state['retrieved_docs'])} 个文本块")  # 日志
        return state  # 返回状态

    def generate_node(state: AgentState) -> AgentState:
        """节点 3：基于检索结果（或通用知识）生成回答。"""
        docs = state.get("retrieved_docs") or []  # 兼容未检索的情况
        if docs:  # 有检索结果：严格基于知识库回答
            context = "\n\n".join(docs)  # 把所有文本块拼成上下文
            prompt = f"""你是专业助手，请严格依据以下知识库内容回答问题。
如果知识库中没有相关信息，请回答“根据知识库，暂无相关信息”。

知识库内容：
{context}

问题：{state["question"]}"""
        else:  # 没有检索结果：用通用知识回答
            prompt = f"""你是专业助手，请根据通用知识回答问题；如果不知道，请如实说明。

问题：{state["question"]}"""
        try:
            state["answer"] = llm.invoke(prompt)  # 调用大模型生成
        except Exception as exc:  # noqa: BLE001 生成失败也不让图崩掉
            state["answer"] = f"生成失败：{exc}"
        print("📝 回答生成完成")  # 日志
        return state  # 返回状态

    def route_after_judge(state: AgentState) -> str:
        """条件边路由函数：按 need_retrieve 决定走检索还是直接生成。"""
        return "retrieve" if state["need_retrieve"] else "generate_answer"  # 需要检索走 retrieve，否则直接生成

    workflow = StateGraph(AgentState)  # 1. 创建状态图（绑定 AgentState）
    workflow.add_node("judge_retrieve", judge_retrieve_node)  # 2. 添加判断节点
    workflow.add_node("retrieve", retrieve_node)  # 2. 添加检索节点
    workflow.add_node("generate_answer", generate_node)  # 2. 添加生成节点
    workflow.set_entry_point("judge_retrieve")  # 3. 入口节点：先判断
    workflow.add_conditional_edges(  # 4. 条件边：按路由函数决定下一步
        source="judge_retrieve",  # 从判断节点出发
        path=route_after_judge,  # 路由函数
        path_map={"retrieve": "retrieve", "generate_answer": "generate_answer"},  # 返回值 -> 节点名
    )
    workflow.add_edge("retrieve", "generate_answer")  # 5. 普通边：检索后一定进入生成
    workflow.add_edge("generate_answer", END)  # 5. 普通边：生成后结束
    return workflow.compile()  # 6. 编译成可执行的 RAG 代理


# ============================================================
# 步骤 6：运行代理并测试（构建图 -> 执行 -> 打印结果）
# ============================================================
def run_rag_agent(agent: Any, question: str) -> str:
    """运行一次 RAG 问答，返回最终回答字符串。"""
    initial_state: AgentState = {  # 初始状态
        "question": question,  # 用户问题
        "retrieved_docs": [],  # 还没有检索结果
        "answer": "",  # 还没有回答
        "need_retrieve": False,  # 由判断节点决定
    }
    final_state = agent.invoke(initial_state)  # 同步执行整张图
    return final_state["answer"]  # 只把回答返回给调用方


def run_doctor() -> int:
    """环境自检：解释器 / 依赖 / Key / 大模型 / 向量模型 / PDF / 索引。"""
    ok = True  # 整体是否通过

    def report(flag: bool, title: str, detail: str) -> None:
        """打印一行带 ✅/❌ 的检查结果。"""
        nonlocal ok  # 修改外层的 ok
        if not flag:  # 有一项失败整体就失败
            ok = False
        print(f"{'✅' if flag else '❌'} {title}：{detail}")  # 输出结果

    print("=" * 70)  # 分隔线
    print("🩺 环境自检")  # 标题
    print("=" * 70)

    report(sys.version_info >= (3, 10), "Python 解释器", f"{sys.version.split()[0]}（{sys.executable}）")  # 解释器版本
    if ".venv" not in sys.executable.lower():  # 用的是系统解释器
        print("   ℹ️ 当前用的是系统解释器（不是项目 .venv），依赖已检查通过，不影响运行")

    missing_pkgs: List[str] = []  # 缺版本信息的包
    package_versions: List[str] = []  # 已安装包的版本
    for mod, pkg in _REQUIRED_IMPORTS.items():  # 逐个检查
        try:
            package_versions.append(f"{pkg} {importlib.metadata.version(pkg)}")  # 记录版本
        except Exception:  # noqa: BLE001 装的是源码包 / 名字对不上
            missing_pkgs.append(pkg)  # 记为缺失
    if missing_pkgs:  # 有缺失
        report(False, "依赖包", "缺少：" + "、".join(sorted(set(missing_pkgs))))
    else:  # 全部齐了
        report(True, "依赖包", "全部已安装（" + "，".join(package_versions[:4]) + " 等）")

    report(bool(LLM_API_KEY), "大模型 Key", f"{LLM_BASE_URL}｜{LLM_MODEL}｜key={mask(LLM_API_KEY)}")  # Key 是否存在
    report(bool(ARK_API_KEY), "方舟 Key", f"{ARK_BASE_URL}｜key={mask(ARK_API_KEY)}")  # Key 是否存在

    if LLM_API_KEY:  # 有大模型 Key 才测连通性
        try:
            started = time.perf_counter()  # 开始计时
            answer = ChatModel(LLM_API_KEY).invoke("只回复 OK 两个字母，不要输出其他内容。", temperature=0, max_tokens=64)
            report(True, "大模型连通", f"耗时 {time.perf_counter() - started:.1f}s，返回 {answer[:30]!r}")
        except Exception as exc:  # noqa: BLE001 打印失败原因
            report(False, "大模型连通", str(exc)[:200])
    else:  # 没有 Key
        report(False, "大模型连通", "缺少 LLM_API_KEY / deepseek_API_KEY")

    if ARK_API_KEY:  # 有方舟 Key 才测向量模型
        try:
            started = time.perf_counter()  # 开始计时
            embeddings = ArkEmbeddings(ARK_API_KEY)  # 创建向量模型
            vector = embeddings.embed_query("环境自检：花椰菜是一种蔬菜")  # 试算一条
            report(True, "向量模型连通", f"{embeddings.active_model}｜维度 {len(vector)}｜耗时 {time.perf_counter() - started:.1f}s")
        except Exception as exc:  # noqa: BLE001 打印失败原因
            report(False, "向量模型连通", str(exc)[:400])
    else:  # 没有 Key
        report(False, "向量模型连通", "缺少 ARK_API_KEY")

    try:  # PDF 是否存在
        pdf_path = find_pdf()  # 查找 PDF
        report(True, "知识库 PDF", f"{pdf_path}（{pdf_path.stat().st_size / 1024:.0f} KB）")
    except Exception as exc:  # noqa: BLE001 PDF 缺失
        report(False, "知识库 PDF", str(exc))

    meta = _read_meta(DEFAULT_INDEX_DIR)  # 读取真实索引的元信息
    if meta:  # 有缓存
        print(f"ℹ️ 已缓存向量库：{DEFAULT_INDEX_DIR}")
        print(f"   文本块 {meta.get('chunks')} 个｜向量模型 {meta.get('embedding', {}).get('model')}｜创建于 {meta.get('created_at')}")
    else:  # 没有缓存
        print("ℹ️ 还没有缓存向量库，首次运行会自动构建")

    print("=" * 70)  # 分隔线
    if ok:  # 全部通过
        print(f'🎉 自检全部通过，可以运行："{sys.executable}" 03RAG检索\\RAG.py')
    else:  # 有失败项
        print("⚠️ 上面标 ❌ 的项需要先处理")
    return 0 if ok else 1  # 返回退出码


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="LangGraph + FAISS 的 RAG 完整案例")  # 创建解析器
    parser.add_argument("questions", nargs="*", help="要问的问题（不传则跑内置测试案例）")  # 位置参数：问题列表
    parser.add_argument("--doctor", action="store_true", help="只做环境自检，不建索引")  # 环境自检
    parser.add_argument("--rebuild", action="store_true", help="强制重建 FAISS 向量库")  # 强制重建
    parser.add_argument("--mock", action="store_true", help="离线模拟模式：不调用任何 API")  # 离线演示
    parser.add_argument("--top-k", type=int, default=3, help="每次检索的文本块数量（默认 3）")  # 检索条数
    return parser.parse_args(argv)  # 返回解析结果


def main(argv: Optional[Sequence[str]] = None) -> int:
    """主流程：配置 -> 知识库 -> 大模型 -> 图 -> 逐个问题测试。"""
    args = parse_args(argv)  # 解析参数

    if args.doctor:  # 只做自检
        return run_doctor()

    # ---------- 6.1 决定运行模式 ----------
    use_mock = bool(args.mock)  # 命令行强制模拟模式
    if not use_mock and (not LLM_API_KEY or not ARK_API_KEY):  # 没有任何 Key 时自动降级为模拟模式
        print("⚠️ 未检测到完整的大模型 / 方舟 Key，自动切换到 --mock 模拟模式（只演示流程，不调用 API）")
        use_mock = True  # 自动降级

    index_dir = MOCK_INDEX_DIR if use_mock else DEFAULT_INDEX_DIR  # 两种模式索引目录隔离

    # ---------- 6.2 初始化组件 ----------
    llm = build_llm(use_mock)  # 大模型
    embeddings = build_embeddings(use_mock)  # 向量模型
    if hasattr(embeddings, "ensure_active"):  # 真实向量模型先探测接口
        embeddings.ensure_active()

    # ---------- 6.3 构建（或复用）知识库 ----------
    try:
        pdf_path = find_pdf()  # 找 PDF
    except FileNotFoundError as exc:  # 没有 PDF
        if not use_mock:  # 真实模式必须有 PDF
            print(str(exc))
            return 1
        print(f"⚠️ {exc}")  # 模拟模式：用内置示例文本
        pdf_path = SCRIPT_DIR / "test.pdf"  # 给一个不存在的路径，交给 build_or_load_index 走兜底分支
    vector_db = build_or_load_index(pdf_path, embeddings, index_dir=index_dir, rebuild=args.rebuild)  # 建库/复用

    # ---------- 6.4 构建 RAG 代理图 ----------
    # 也可以写成：retriever = vector_db.as_retriever(search_kwargs={"k": args.top_k})
    # 这里直接传 vector_db + top_k，检索节点内部用 simple_retrieve
    rag_agent = build_agent(vector_db, llm, top_k=max(1, args.top_k))  # 编译状态图

    # ---------- 6.5 跑测试案例 ----------
    questions = args.questions or TEST_QUESTIONS  # 命令行传了问题就用命令行的
    for index, question in enumerate(questions, 1):  # 逐个问题执行
        print(f"\n{'=' * 70}\n测试 {index}：{question}\n{'=' * 70}")  # 分隔与标题
        started = time.perf_counter()  # 开始计时
        answer = run_rag_agent(rag_agent, question)  # 执行一次 RAG
        print(f"最终回答：{answer}")  # 打印回答
        print(f"耗时：{time.perf_counter() - started:.1f}s")  # 打印耗时
    return 0  # 正常结束


if __name__ == "__main__":  # 只有直接运行脚本时才执行
    try:
        raise SystemExit(main())  # 用返回值作为退出码
    except KeyboardInterrupt:  # 用户 Ctrl+C
        print("\n已中断")
        raise SystemExit(130)  # 130 是常见的「被中断」退出码
