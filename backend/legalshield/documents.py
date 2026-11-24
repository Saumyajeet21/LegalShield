"""Extract immutable page text with Unicode offsets and normalized PDF boxes."""
import io
import re
import zipfile
from dataclasses import dataclass, field

import pymupdf
from defusedxml.ElementTree import fromstring

MAX_BYTES = 20 * 1024 * 1024
MAX_PAGES = 300
MAX_TEXT = 200_000
Box = list[float]


@dataclass
class Page:
    number: int
    text: str
    boxes: dict[int, Box] = field(default_factory=dict)

    def rectangles(self, start: int, end: int) -> list[Box]:
        lines: list[Box] = []
        current = None
        for index in range(start, end):
            if self.text[index] == "\n":
                current = None
            box = self.boxes.get(index)
            if box is None:
                if self.boxes and not self.text[index].isspace():
                    return []
                continue
            if current is None:
                current = box.copy()
                lines.append(current)
            else:
                current[0] = min(current[0], box[0])
                current[1] = min(current[1], box[1])
                current[2] = max(current[2], box[2])
                current[3] = max(current[3], box[3])
        return lines


def pdf_pages(data: bytes) -> list[Page]:
    pages: list[Page] = []
    total = 0
    with pymupdf.open(stream=data, filetype="pdf") as source:
        if source.needs_pass:
            raise ValueError("Encrypted PDFs require an unlocked copy.")
        if len(source) > MAX_PAGES:
            raise ValueError(f"A PDF can contain at most {MAX_PAGES} pages.")
        for page in source:
            pieces: list[str] = []
            boxes: dict[int, Box] = {}
            offset = 0
            for block in page.get_text("rawdict", flags=pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_PRESERVE_IMAGES, sort=True)["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        for char in span["chars"]:
                            value = char["c"]
                            rect = pymupdf.Rect(char["bbox"]) * page.rotation_matrix
                            box = [rect.x0 / page.rect.width, rect.y0 / page.rect.height,
                                   rect.x1 / page.rect.width, rect.y1 / page.rect.height]
                            for index, character in enumerate(value):
                                if not character.isspace() and all(0 <= value <= 1 for value in box):
                                    boxes[offset + index] = box
                            pieces.append(value)
                            offset += len(value)
                            if total + offset > MAX_TEXT:
                                raise ValueError("The document exceeds the extracted-text limit.")
                    pieces.append("\n")
                    offset += 1
            total += offset
            pages.append(Page(page.number + 1, "".join(pieces), boxes))
    return pages


def docx_text(data: bytes) -> str:
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        body = archive.getinfo("word/document.xml")
        if body.file_size > 8 * 1024 * 1024:
            raise ValueError("The DOCX main document is too large to process.")
        root = fromstring(archive.read(body), forbid_dtd=True)
    paragraphs = []
    for paragraph in root.iter(namespace + "p"):
        pieces = []
        for item in paragraph.iter():
            if item.tag == namespace + "t":
                pieces.append(item.text or "")
            elif item.tag in {namespace + "br", namespace + "cr"}:
                pieces.append("\n")
            elif item.tag == namespace + "tab":
                pieces.append("\t")
        paragraphs.append("".join(pieces))
    return "\n".join(paragraphs)


def extract(filename: str, data: bytes) -> tuple[str, list[Page]]:
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Choose a nonempty file of at most 20 MiB.")
    suffix = filename.rsplit(".", 1)[-1].lower()
    if suffix == "pdf" and data[:5] == b"%PDF-":
        pages = pdf_pages(data)
        kind = "application/pdf"
    elif suffix == "docx" and data[:4] == b"PK\x03\x04":
        pages = [Page(1, docx_text(data))]
        kind = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif suffix == "txt" and b"\x00" not in data:
        pages = [Page(1, data.decode("utf-8-sig"))]
        kind = "text/plain"
    else:
        raise ValueError("Choose a valid PDF, DOCX, or UTF-8 TXT file.")
    if sum(len(page.text) for page in pages) > MAX_TEXT:
        raise ValueError("The document exceeds the extracted-text limit.")
    if not any(page.text.strip() for page in pages):
        raise ValueError("No selectable text was found. Run OCR on scanned files before uploading them.")
    return kind, pages


HEADING = re.compile(r"(?im)^\s*(?P<label>(?:article|section|clause|schedule|chapter)\s+[\w.-]+|\d+(?:\.\d+)*[.)])(?=\s)")


def chunk_ranges(text: str, limit: int = 950):
    headings = list(HEADING.finditer(text))
    boundaries = [(0, None)]
    boundaries.extend((match.start(), match.group("label").strip()) for match in headings)
    for index, (start, clause) in enumerate(boundaries):
        end = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(text)
        while start < end:
            stop = min(end, start + limit)
            if stop < end:
                window = text[start:stop]
                cut = max(window.rfind("\n"), window.rfind(". "), window.rfind(" "))
                if cut >= limit // 2:
                    stop = start + cut + 1
            if text[start:stop].strip():
                yield start, stop, clause
            start = stop
