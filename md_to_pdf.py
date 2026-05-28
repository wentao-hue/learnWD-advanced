"""
Convert RESULTS.md and CODE_GUIDE.md to PDF using markdown + weasyprint.
Supports Chinese characters via bundled STHeiti font.

Usage:
    .venv/bin/python3 md_to_pdf.py
"""

import re
import sys
from pathlib import Path

import markdown
from weasyprint import HTML, CSS

# ──────────────────────────────────────────────────────────────────── #
# CSS stylesheet — A4, Chinese font, tables, code blocks               #
# ──────────────────────────────────────────────────────────────────── #

HEITI_PATH = "/System/Library/Fonts/STHeiti Light.ttc"
HEITI_MED  = "/System/Library/Fonts/STHeiti Medium.ttc"

CSS_TEMPLATE = f"""
@font-face {{
    font-family: 'STHeiti';
    src: url('file://{HEITI_PATH}');
    font-weight: normal;
}}
@font-face {{
    font-family: 'STHeiti';
    src: url('file://{HEITI_MED}');
    font-weight: bold;
}}

@page {{
    size: A4;
    margin: 20mm 18mm 20mm 18mm;
    @bottom-center {{
        content: counter(page) " / " counter(pages);
        font-family: 'STHeiti', sans-serif;
        font-size: 9pt;
        color: #888;
    }}
}}

body {{
    font-family: 'STHeiti', 'Helvetica Neue', Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.6;
    color: #1a1a1a;
    word-wrap: break-word;
}}

/* Headings */
h1 {{
    font-size: 20pt;
    font-weight: bold;
    color: #1a1a1a;
    margin: 0 0 14pt 0;
    padding-bottom: 6pt;
    border-bottom: 2pt solid #2c3e50;
    page-break-after: avoid;
}}
h2 {{
    font-size: 14pt;
    font-weight: bold;
    color: #2c3e50;
    margin: 18pt 0 6pt 0;
    padding-bottom: 3pt;
    border-bottom: 1pt solid #bdc3c7;
    page-break-after: avoid;
}}
h3 {{
    font-size: 11pt;
    font-weight: bold;
    color: #34495e;
    margin: 12pt 0 4pt 0;
    page-break-after: avoid;
}}
h4 {{
    font-size: 10pt;
    font-weight: bold;
    color: #555;
    margin: 10pt 0 4pt 0;
    page-break-after: avoid;
}}

/* Paragraphs & lists */
p {{
    margin: 0 0 6pt 0;
}}
ul, ol {{
    margin: 0 0 6pt 0;
    padding-left: 18pt;
}}
li {{
    margin-bottom: 3pt;
}}

/* Blockquote */
blockquote {{
    margin: 6pt 0 6pt 12pt;
    padding: 4pt 10pt;
    border-left: 3pt solid #3498db;
    background: #f4f8fb;
    color: #444;
    font-size: 9.5pt;
}}

/* Code — inline */
code {{
    font-family: 'Courier New', 'Menlo', monospace;
    font-size: 8.5pt;
    background: #f4f4f4;
    border: 0.5pt solid #ddd;
    border-radius: 2pt;
    padding: 0.5pt 3pt;
    color: #c0392b;
}}

/* Code — fenced block */
pre {{
    background: #f8f8f8;
    border: 0.5pt solid #ddd;
    border-left: 3pt solid #95a5a6;
    border-radius: 3pt;
    padding: 8pt 10pt;
    margin: 6pt 0 8pt 0;
    overflow-x: auto;
    page-break-inside: avoid;
    font-size: 7.5pt;
    line-height: 1.5;
}}
pre code {{
    font-family: 'Courier New', 'Menlo', monospace;
    font-size: 7.5pt;
    background: none;
    border: none;
    padding: 0;
    color: #1a1a1a;
}}

/* Tables */
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 8pt 0 10pt 0;
    font-size: 8pt;
    page-break-inside: auto;
}}
thead {{
    background-color: #2c3e50;
    color: #fff;
}}
thead tr th {{
    padding: 5pt 7pt;
    text-align: left;
    font-weight: bold;
    border: 0.5pt solid #2c3e50;
    white-space: nowrap;
}}
tbody tr:nth-child(even) {{
    background-color: #f2f6f9;
}}
tbody tr:nth-child(odd) {{
    background-color: #ffffff;
}}
tbody tr td {{
    padding: 4pt 7pt;
    border: 0.5pt solid #bdc3c7;
    vertical-align: top;
}}
tbody tr:hover {{
    background-color: #eaf2ff;
}}

/* Horizontal rule */
hr {{
    border: none;
    border-top: 1pt solid #bdc3c7;
    margin: 12pt 0;
}}

/* Bold / emphasis */
strong {{
    font-weight: bold;
    color: #1a1a1a;
}}
em {{
    font-style: italic;
    color: #555;
}}

/* Links */
a {{
    color: #2980b9;
    text-decoration: none;
}}
"""


# ──────────────────────────────────────────────────────────────────── #
# Markdown pre-processing                                              #
# ──────────────────────────────────────────────────────────────────── #

def preprocess_md(text: str) -> str:
    """Fix markdown quirks before conversion."""
    # Escape raw < / > characters that aren't HTML tags (e.g. in code text)
    # (markdown library handles most of this; just normalise line endings)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


# ──────────────────────────────────────────────────────────────────── #
# Conversion                                                           #
# ──────────────────────────────────────────────────────────────────── #

def md_to_pdf(md_path: Path, pdf_path: Path) -> None:
    print(f"  Converting  {md_path.name}  →  {pdf_path.name} …", end=" ", flush=True)

    md_text = md_path.read_text(encoding="utf-8")
    md_text = preprocess_md(md_text)

    # Convert markdown → HTML
    html_body = markdown.markdown(
        md_text,
        extensions=[
            "tables",           # pipe tables
            "fenced_code",      # ```python ... ```
            "codehilite",       # syntax colouring (degrades gracefully)
            "toc",              # auto TOC anchors
            "nl2br",            # newline → <br> in paragraphs
            "sane_lists",       # better list handling
            "smarty",           # smart quotes / dashes
        ],
        extension_configs={
            "codehilite": {
                "guess_lang": False,
                "noclasses": True,   # inline styles, no external CSS needed
            },
        },
    )

    title = md_path.stem.replace("_", " ")
    full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
</head>
<body>
{html_body}
</body>
</html>"""

    # Render to PDF
    html_obj = HTML(string=full_html, base_url=str(md_path.parent))
    css_obj  = CSS(string=CSS_TEMPLATE)
    html_obj.write_pdf(str(pdf_path), stylesheets=[css_obj])

    size_kb = pdf_path.stat().st_size // 1024
    print(f"done  ({size_kb} KB)")


# ──────────────────────────────────────────────────────────────────── #
# Main                                                                 #
# ──────────────────────────────────────────────────────────────────── #

def main():
    base = Path("/Users/pudding/Desktop/learnWD")
    targets = [
        (base / "RESULTS.md",    base / "RESULTS.pdf"),
        (base / "CODE_GUIDE.md", base / "CODE_GUIDE.pdf"),
    ]

    print("\n=== Markdown → PDF conversion ===\n")
    for md_path, pdf_path in targets:
        if not md_path.exists():
            print(f"  SKIP (not found): {md_path}")
            continue
        try:
            md_to_pdf(md_path, pdf_path)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()

    print("\nAll done.")


if __name__ == "__main__":
    main()
