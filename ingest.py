"""
文档摄取器 — AI 嵌入版
用本地 BGE 中文模型做语义向量化
用法：python ingest.py
"""

import os

# ── 本地模型路径（由 modelscope 下载） ─────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "BAAI", "bge-small-zh-v1___5")

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── 配置 ──────────────────────────────────────────────────
DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ── 加载嵌入模型（本地，不联网） ─────────────────────────
print("⏳ 加载 BGE 中文嵌入模型...")
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_PATH,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("✅ 模型加载完成")

# ── 加载文档 ─────────────────────────────────────────────
all_docs = []

if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)
    print(f"📁 已创建 {DOCS_DIR}/ 目录，请放入文档后重新运行")

for filename in os.listdir(DOCS_DIR):
    filepath = os.path.join(DOCS_DIR, filename)

    if filename.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
        print(f"📄 加载 PDF: {filename}")
    elif filename.endswith(".txt") or filename.endswith(".md"):
        loader = TextLoader(filepath, encoding="utf-8")
        print(f"📝 加载文本: {filename}")
    else:
        print(f"⏭️  跳过: {filename}（不支持的格式）")
        continue

    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = filename
    all_docs.extend(docs)

if not all_docs:
    print("⚠️  docs/ 目录下没有支持的文档（PDF/TXT/MD），放一篇后再跑。")
    exit(1)

print(f"✅ 共加载 {len(all_docs)} 页/段落")

# ── 切分 ─────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
)
chunks = splitter.split_documents(all_docs)
print(f"✂️  切分为 {len(chunks)} 个片段")

# ── 存入 ChromaDB ────────────────────────────────────────
print("💾 向量化并存入 ChromaDB...")
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
)
print(f"🎉 完成！知识库已就绪，共 {len(chunks)} 个片段存入 {CHROMA_DIR}/")
print(f"   嵌入模型: BGE-small-zh | 检索方式: 语义相似度")
