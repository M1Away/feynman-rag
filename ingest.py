"""
文档摄取器 — AI 嵌入版
用本地 BGE 中文模型做语义向量化

用法：
  python ingest.py                          # 默认：500字切片，100字重叠
  python ingest.py --size 300 --overlap 50  # 自定义切片参数
"""

import os
import sys

# ── 命令行参数 ──────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

args = sys.argv[1:]
for i, arg in enumerate(args):
    if arg == "--size" and i + 1 < len(args):
        CHUNK_SIZE = int(args[i + 1])
    elif arg == "--overlap" and i + 1 < len(args):
        CHUNK_OVERLAP = int(args[i + 1])

# ── 本地模型路径（由 modelscope 下载） ─────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "BAAI", "bge-small-zh-v1___5")

import docx2txt
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── 配置 ──────────────────────────────────────────────────
DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"

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
        docs = loader.load()
    elif filename.endswith(".docx"):
        text = docx2txt.process(filepath)
        if not text or not text.strip():
            print(f"⚠️  Word 文档为空: {filename}")
            continue
        docs = [Document(page_content=text)]
        print(f"📝 加载 Word: {filename}")
    elif filename.endswith(".txt") or filename.endswith(".md"):
        loader = TextLoader(filepath, encoding="utf-8")
        print(f"📝 加载文本: {filename}")
        docs = loader.load()
    else:
        print(f"⏭️  跳过: {filename}（不支持的格式）")
        continue

    for doc in docs:
        doc.metadata["source"] = filename
    all_docs.extend(docs)

if not all_docs:
    print("⚠️  docs/ 目录下没有支持的文档（PDF/TXT/MD），放一篇后再跑。")
    exit(1)

print(f"✅ 共加载 {len(all_docs)} 页/段落")
print(f"🔧 切片策略: {CHUNK_SIZE}字/片, 重叠{CHUNK_OVERLAP}字")

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

# ── 保存原始片段（给 BM25 关键词检索用） ─────────────────
import json
chunk_data = []
for chunk in chunks:
    chunk_data.append({
        "content": chunk.page_content,
        "source": chunk.metadata.get("source", "未知"),
    })
with open(os.path.join(CHROMA_DIR, "chunks.json"), "w", encoding="utf-8") as f:
    json.dump(chunk_data, f, ensure_ascii=False)

print(f"🎉 完成！共 {len(chunks)} 个片段 → {CHROMA_DIR}/")
print(f"   切片 {CHUNK_SIZE}字 | 重叠 {CHUNK_OVERLAP}字 | 嵌入 BGE-small-zh")
print(f"   混合检索: 语义(BGE) + 关键词(BM25)")
