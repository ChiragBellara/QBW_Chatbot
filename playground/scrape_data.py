import re
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md


def html_to_markdown(url: str) -> str:
    main_html = requests.get(url, timeout=30).text
    soup = BeautifulSoup(main_html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    markdown = md(str(soup), heading_style="ATX")
    markdown = re.sub(r'\bImage\b(\s*\bImage\b)+', ' ', markdown)
    markdown = f"# Source\n{url}\n\n---\n\n{markdown}\n"
    return markdown


if __name__ == "__main__":
    urls = [
        "https://thequantumbodyworks.com/about/",
        "https://thequantumbodyworks.com/our-team/",
        "https://thequantumbodyworks.com/what-we-treat/",
        "https://thequantumbodyworks.com/faq/",
        "https://thequantumbodyworks.com/contact-us/"
    ]

    for u in urls:
        filename = u.rstrip("/").split("/")[-1] or "home"
        with open(f"tqb_{filename}.md", "w", encoding="utf-8") as f:
            f.write(html_to_markdown(u))
