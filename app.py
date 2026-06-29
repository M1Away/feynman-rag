"""
费曼三层解释器 - Day 7 混合检索版
新增：BM25 关键词检索 + BGE 语义检索 = 混合检索
"""

import os
import json
import streamlit as st
from openai import OpenAI
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── 本地模型路径 ──────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "BAAI", "bge-small-zh-v1___5")

# ── 页面配置 ─────────────────────────────────────────────
st.set_page_config(page_title="费曼三层解释器", page_icon="🧠", layout="wide")
st.title("🧠 费曼三层解释器")
st.caption("输入任何概念，AI 用三种深度为你解释 | Day 7 · 混合检索")

# ── 侧边栏 ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ API 配置")
    api_key = st.text_input(
        "DeepSeek API Key", type="password", value="",
        help="从 platform.deepseek.com 获取",
    )
    model = st.selectbox("模型", ["deepseek-chat", "deepseek-reasoner"], index=0)

    st.divider()
    st.header("🎛️ 参数调节")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    view_mode = st.radio("展示模式", ["📊 三栏并排", "📋 单栏依次"])

    st.divider()
    st.header("📚 知识库")
    use_rag = st.toggle("启用知识库检索", value=True,
                        help="关闭后只靠 AI 自身知识回答，不查文档")
    top_k = st.slider("检索片段数", 1, 5, 3,
    help="从知识库取几个最相关的片段给 AI 参考")
    bm25_weight = st.slider(
        "BM25 关键词权重", 0.0, 1.0, 0.5, 0.1,
        help="0=纯语义检索，1=纯关键词匹配。专有名词搜不到时拉高这个"
    )
    st.caption("💡 先用 `python ingest.py` 构建知识库")

# ── 加载向量库 + BM25 ─────────────────────────────────────

@st.cache_resource
def load_vectordb():
    """加载 ChromaDB + 本地 BGE 嵌入模型"""
    chroma_dir = "chroma_db"
    if not os.path.exists(chroma_dir) or not os.path.exists(MODEL_PATH):
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings,
    )


@st.cache_resource
def load_bm25():
    """加载所有片段文本，构建 BM25 关键词检索引擎"""
    json_path = os.path.join("chroma_db", "chunks.json")
    if not os.path.exists(json_path):
        return None, []

    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # 字符级分词（中文友好，不依赖 jieba）
    tokenized = [list(c["content"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25, chunks


vectordb = load_vectordb()
bm25, bm25_chunks = load_bm25()

if vectordb is None and use_rag:
    st.warning("⚠️ 知识库尚未构建。请在终端运行 `python ingest.py` 先摄取文档。")

# ── 提示词模板（Day 4：加入知识库上下文） ─────────────────

SYSTEM_LAYER1 = """你是一位小学科学老师，正在给一群12岁的孩子上课。

严格遵循以下规则——违反任何一条都会让解释失败：
1. 【强制】开头第一句必须是类比，格式为"想象一下……"或"就像……"
2. 【强制】全文不得出现任何英文缩写或专业术语。如果不可避免，用中文口语描述代替
3. 【强制】至少包含一个具体的、孩子见过的生活场景（比如做饭、玩游戏、养宠物）
4. 用"你"来跟孩子对话，不要用"我们"
5. 结尾用"🌟 简单说就是：……"总结
6. 总共不超过300字

反面示例（绝对不要这样写）：
❌ "机器学习是一种人工智能技术" → 这是定义，不是类比
❌ "通过梯度下降优化损失函数" → 全是术语
✅ "想象一下你教小狗认字……" → 这才是我们要的"""

SYSTEM_LAYER2 = """你是一位资深后端工程师，在给团队做技术分享。

严格遵循以下规则：
1. 【强制】第一句给出精准定义："{概念} 是……"
2. 【强制】用 2-4 个要点拆解核心机制，每个要点一行，格式为 "• ……"
3. 可以且应该使用技术术语（如 embedding、token、attention），但每个术语首次出现时用括号给一句话解释
4. 提及至少一种具体的算法/框架/实现方式
5. 语气像内部技术文档，不卖关子，不煽情
6. 总共不超过300字

反面示例：
❌ "机器学习很有趣，它改变了世界" → 没有原理
❌ 用讲故事的方式 → 这是Layer 1的事"""

SYSTEM_LAYER3 = """你是一位在AI领域工作了十年的研究员，正在和同事在咖啡机旁闲聊。

严格遵循以下规则：
1. 【强制】跳过所有基础定义——假设对方已经知道机器学习是什么
2. 【强制】给出一个"外行人不会想到"的洞察或反直觉观点
3. 用第一性原理切入：这个东西之所以work，本质上是因为……
4. 提及至少一个常见的误解并纠正它
5. 语气随意但有深度，像在说"其实这玩意儿没大家想的那么玄……"
6. 总共不超过300字

反面示例：
❌ "机器学习是人工智能的一个分支……" → 这是定义，不是洞察
❌ 用要点列表 → 闲聊不用列表"""

PROMPTS = {
    "🐣 给12岁孩子": (SYSTEM_LAYER1, "请用类比解释这个概念，优先参考提供的资料："),
    "📘 给技术同事": (SYSTEM_LAYER2, "请用技术原理层拆解这个概念，优先参考提供的资料："),
    "🧪 给领域专家": (SYSTEM_LAYER3, "从专家视角闲聊这个概念，优先参考提供的资料："),
}

# ── 混合检索（Day 7：BM25 + BGE） ─────────────────────────

def retrieve_context(question: str, k: int, bm25_w: float):
    """
    混合检索 = BM25关键词 + BGE语义
    bm25_w=0 → 纯语义；bm25_w=1 → 纯关键词；默认0.5各一半
    """
    # ── 1. 语义检索（向量） ──
    vector_results = {}  # content_hash → {score, source, content, method}
    docs_with_scores = vectordb.similarity_search_with_score(question, k=k * 3)
    for doc, score in docs_with_scores:
        sim = max(0.0, min(1.0, 1.0 - score / 2.0))
        key = doc.page_content[:100]  # 用前100字做去重key
        vector_results[key] = {
            "similarity": sim,
            "source": doc.metadata.get("source", "未知"),
            "content": doc.page_content,
            "method": "语义",
        }

    # ── 2. 关键词检索（BM25） ──
    bm25_results = {}
    if bm25 and bm25_chunks and bm25_w > 0:
        tokenized_query = list(question)
        bm25_scores = bm25.get_scores(tokenized_query)
        # 取 BM25 分数最高的 k*3 个，归一化
        top_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[: k * 3]
        if top_indices:
            max_bm25 = bm25_scores[top_indices[0]]
            for idx in top_indices:
                norm_score = bm25_scores[idx] / max_bm25 if max_bm25 > 0 else 0
                key = bm25_chunks[idx]["content"][:100]
                bm25_results[key] = {
                    "similarity": round(norm_score, 4),
                    "source": bm25_chunks[idx]["source"],
                    "content": bm25_chunks[idx]["content"],
                    "method": "关键词",
                }

    # ── 3. 合并 ──
    merged = {}
    for key, info in vector_results.items():
        merged[key] = {
            **info,
            "final_score": info["similarity"] * (1 - bm25_w),
            "methods": [info["method"]],
        }
    for key, info in bm25_results.items():
        if key in merged:
            merged[key]["final_score"] += info["similarity"] * bm25_w
            merged[key]["methods"].append(info["method"])
        else:
            merged[key] = {
                **info,
                "final_score": info["similarity"] * bm25_w,
                "methods": [info["method"]],
            }

    # ── 4. 排序取 top_k ──
    sorted_items = sorted(merged.items(), key=lambda x: x[1]["final_score"], reverse=True)
    scored_docs = []
    parts = []
    for i, (_, info) in enumerate(sorted_items[:k]):
        method_tag = "+".join(info["methods"])
        scored_docs.append({
            "index": i + 1,
            "source": info["source"],
            "similarity": info["final_score"],
            "content": info["content"],
            "method": method_tag,
        })
        parts.append(
            f"[片段{i+1} · {method_tag} · 来源：{info['source']} · "
            f"得分：{info['final_score']:.2%}]\n{info['content']}"
        )

    context = "\n\n---\n\n".join(parts) if parts else ""
    return context, scored_docs


# ── 流式调用 ─────────────────────────────────────────────

def explain_layer_stream(client, model, system_prompt, user_prefix,
    question, context, temp):
    """流式调用 DeepSeek API"""
    user_content = f"{user_prefix}\n\n{question}"
    if context:
        user_content = (
            f"以下是参考资料的片段，请基于这些内容回答问题。"
            f"如果资料不足，可以用你的知识补充，但优先使用资料中的信息。\n\n"
            f"=== 参考资料 ===\n{context}\n=== 资料结束 ===\n\n"
            f"{user_prefix}\n\n{question}"
        )

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=temp,
        max_tokens=600,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ── 主界面 ───────────────────────────────────────────────

question = st.text_input(
    "🔍 输入你想理解的概念",
    placeholder="例如：什么是机器学习？Transformer 为什么有效？什么是梯度下降？",
)

if st.button("✨ 生成三层解释", type="primary", use_container_width=True):
    if not api_key:
        st.error("⚠️ 请先在侧边栏输入 DeepSeek API Key")
    elif not question.strip():
        st.warning("⚠️ 请输入一个问题")
    else:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        # ── 检索 ──
        context = ""
        scored_docs = []
        if use_rag and vectordb:
            with st.spinner("🔍 检索知识库..."):
                context, scored_docs = retrieve_context(question, top_k, bm25_weight)

        # ── 检索质量调试面板（Day 6 新增） ──
        if scored_docs:
            with st.expander("📎 检索质量调试面板", expanded=True):
                st.caption(f"🔍 问题：{question}")
                st.divider()
                for doc in scored_docs:
                    sim = doc["similarity"]
                    # 颜色编码：≥70% 绿，40-70% 黄，<40% 红
                    if sim >= 0.70:
                        color = "#10b981"
                        label = "🟢 高相关"
                    elif sim >= 0.40:
                        color = "#f59e0b"
                        label = "🟡 中相关"
                    else:
                        color = "#ef4444"
                        label = "🔴 低相关"

                    method_tag = doc.get("method", "语义")
                    st.markdown(
                        f"**片段 {doc['index']}** · {label} · "
                        f"`{method_tag}` · "
                        f"得分 `{sim:.1%}` · 来源 `{doc['source']}`"
                    )
                    # 相似度进度条
                    st.progress(sim)
                    # 片段原文（截取前300字）
                    preview = doc["content"][:300]
                    if len(doc["content"]) > 300:
                        preview += "..."
                    st.text(preview)
                    st.divider()

        # ── 生成解释 ──
        if view_mode == "📊 三栏并排":
            col1, col2, col3 = st.columns(3)
            for col, (label, (sys_prompt, user_prefix)) in zip(
                [col1, col2, col3], PROMPTS.items()
            ):
                with col:
                    st.subheader(label)
                    placeholder = st.empty()
                    full_text = ""
                    try:
                        for token in explain_layer_stream(
                            client, model, sys_prompt, user_prefix,
                            question, context, temperature
                        ):
                            full_text += token
                            placeholder.markdown(full_text + "▌")
                        placeholder.markdown(full_text)
                    except Exception as e:
                        st.error(f"出错了：{e}")
        else:
            for label, (sys_prompt, user_prefix) in PROMPTS.items():
                st.subheader(label)
                placeholder = st.empty()
                full_text = ""
                try:
                    for token in explain_layer_stream(
                        client, model, sys_prompt, user_prefix,
                        question, context, temperature
                    ):
                        full_text += token
                        placeholder.markdown(full_text + "▌")
                    placeholder.markdown(full_text)
                except Exception as e:
                    st.error(f"出错了：{e}")
                st.divider()

        st.divider()
        if context:
            st.caption("💡 本次回答基于知识库中的资料片段。点击上方「检索到的资料片段」查看原文。")
        else:
            st.caption("💡 本次回答仅基于 AI 自身知识（未启用知识库或知识库为空）。")

st.divider()
st.caption("Day 7 · 混合检索(BM25+BGE) · 费曼项目 · Away")
