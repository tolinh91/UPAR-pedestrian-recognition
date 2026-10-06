"""Inline xai_data.json into page_template.html -> attention.html (self-contained page)."""
from pathlib import Path
HERE = Path(__file__).parent
data = (HERE / "xai_data.json").read_text().replace("</", "<\\/")
(HERE / "attention.html").write_text((HERE / "page_template.html").read_text().replace("__DATA__", data))
print("attention.html", round((HERE / "attention.html").stat().st_size / 1e6, 2), "MB")
