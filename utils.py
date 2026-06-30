"""共享工具函数"""
import requests
from bs4 import BeautifulSoup


def fetch_url(url: str) -> tuple[str, str] | None:
    """
    抓取网页文本内容
    返回 (正文, 标题) 或 None
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.encoding = resp.apparent_encoding or "utf-8"
    except Exception as e:
        print(f"❌ 抓取失败: {e}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # 去掉非正文标签
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title else url

    # 提取纯文本
    lines = [line.strip() for line in soup.get_text().splitlines() if line.strip()]
    text = "\n".join(lines)

    if len(text) < 200:
        print(f"⚠️  网页正文太短 ({len(text)}字)，可能抓取不完整")
        return None

    print(f"🌐 抓取成功: {title} ({len(text)}字)")
    return text, title
