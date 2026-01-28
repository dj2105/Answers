from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

OUTPUT_PATH = Path("answers_exam_style.pdf")


@dataclass
class Line:
    text: str
    font_key: str
    size: float
    x: float
    y: float


@dataclass
class Style:
    font_key: str
    size: float
    leading: float
    indent: float
    space_before: float
    space_after: float
    align: str = "left"


class TrueTypeFont:
    def __init__(self, path: Path, name: str) -> None:
        self.path = path
        self.name = name
        self.data = path.read_bytes()
        self.tables = self._read_tables()
        self.units_per_em, self.bbox = self._read_head()
        self.ascender, self.descender, self.num_hmetrics = self._read_hhea()
        self.num_glyphs = self._read_maxp()
        self.advance_widths = self._read_hmtx()
        self.cmap = self._read_cmap()
        self.used_gids: set[int] = set()

    def _read_tables(self) -> Dict[str, Tuple[int, int]]:
        num_tables = struct.unpack(">H", self.data[4:6])[0]
        tables: Dict[str, Tuple[int, int]] = {}
        offset = 12
        for _ in range(num_tables):
            tag = self.data[offset : offset + 4].decode("ascii")
            _, table_offset, length = struct.unpack(">III", self.data[offset + 4 : offset + 16])
            tables[tag] = (table_offset, length)
            offset += 16
        return tables

    def _read_head(self) -> Tuple[int, Tuple[int, int, int, int]]:
        offset, _ = self.tables["head"]
        units_per_em = struct.unpack(">H", self.data[offset + 18 : offset + 20])[0]
        x_min, y_min, x_max, y_max = struct.unpack(">hhhh", self.data[offset + 36 : offset + 44])
        return units_per_em, (x_min, y_min, x_max, y_max)

    def _read_hhea(self) -> Tuple[int, int, int]:
        offset, _ = self.tables["hhea"]
        ascender, descender = struct.unpack(">hh", self.data[offset + 4 : offset + 8])
        num_hmetrics = struct.unpack(">H", self.data[offset + 34 : offset + 36])[0]
        return ascender, descender, num_hmetrics

    def _read_maxp(self) -> int:
        offset, _ = self.tables["maxp"]
        return struct.unpack(">H", self.data[offset + 4 : offset + 6])[0]

    def _read_hmtx(self) -> Dict[int, int]:
        offset, _ = self.tables["hmtx"]
        widths: Dict[int, int] = {}
        for gid in range(self.num_hmetrics):
            advance = struct.unpack(">H", self.data[offset + gid * 4 : offset + gid * 4 + 2])[0]
            widths[gid] = advance
        last_advance = widths[self.num_hmetrics - 1]
        for gid in range(self.num_hmetrics, self.num_glyphs):
            widths[gid] = last_advance
        return widths

    def _read_cmap(self) -> Dict[int, int]:
        offset, _ = self.tables["cmap"]
        num_tables = struct.unpack(">H", self.data[offset + 2 : offset + 4])[0]
        chosen_offset = None
        chosen_format = None
        for i in range(num_tables):
            sub_offset = offset + 4 + i * 8
            platform_id, encoding_id, table_offset = struct.unpack(">HHI", self.data[sub_offset : sub_offset + 8])
            table_abs_offset = offset + table_offset
            fmt = struct.unpack(">H", self.data[table_abs_offset : table_abs_offset + 2])[0]
            if platform_id == 3 and encoding_id in (1, 10) and fmt in (4, 12):
                chosen_offset = table_abs_offset
                chosen_format = fmt
                break
            if platform_id == 0 and fmt in (4, 12):
                chosen_offset = table_abs_offset
                chosen_format = fmt
        if chosen_offset is None:
            raise ValueError("No usable cmap subtable found.")
        if chosen_format == 4:
            return self._parse_cmap_format4(chosen_offset)
        return self._parse_cmap_format12(chosen_offset)

    def _parse_cmap_format4(self, offset: int) -> Dict[int, int]:
        seg_count_x2 = struct.unpack(">H", self.data[offset + 6 : offset + 8])[0]
        seg_count = seg_count_x2 // 2
        end_codes = struct.unpack(">" + "H" * seg_count, self.data[offset + 14 : offset + 14 + seg_count * 2])
        start_offset = offset + 16 + seg_count * 2
        start_codes = struct.unpack(">" + "H" * seg_count, self.data[start_offset : start_offset + seg_count * 2])
        id_delta_offset = start_offset + seg_count * 2
        id_deltas = struct.unpack(">" + "h" * seg_count, self.data[id_delta_offset : id_delta_offset + seg_count * 2])
        id_range_offset_offset = id_delta_offset + seg_count * 2
        id_range_offsets = struct.unpack(">" + "H" * seg_count, self.data[id_range_offset_offset : id_range_offset_offset + seg_count * 2])
        glyph_array_offset = id_range_offset_offset + seg_count * 2
        cmap: Dict[int, int] = {}
        for i in range(seg_count):
            start = start_codes[i]
            end = end_codes[i]
            if start == 0xFFFF and end == 0xFFFF:
                continue
            for code in range(start, end + 1):
                if id_range_offsets[i] == 0:
                    gid = (code + id_deltas[i]) % 65536
                else:
                    roffset = id_range_offsets[i]
                    glyph_index_offset = id_range_offset_offset + i * 2 + roffset + (code - start) * 2
                    glyph_index = struct.unpack(">H", self.data[glyph_index_offset : glyph_index_offset + 2])[0]
                    gid = (glyph_index + id_deltas[i]) % 65536 if glyph_index != 0 else 0
                if gid != 0:
                    cmap[code] = gid
        return cmap

    def _parse_cmap_format12(self, offset: int) -> Dict[int, int]:
        n_groups = struct.unpack(">I", self.data[offset + 12 : offset + 16])[0]
        cmap: Dict[int, int] = {}
        pos = offset + 16
        for _ in range(n_groups):
            start_char, end_char, start_gid = struct.unpack(">III", self.data[pos : pos + 12])
            for code in range(start_char, end_char + 1):
                cmap[code] = start_gid + (code - start_char)
            pos += 12
        return cmap

    def glyph_width(self, gid: int) -> float:
        return self.advance_widths.get(gid, 0) * 1000 / self.units_per_em

    def encode_text(self, text: str) -> str:
        hex_bytes = []
        for ch in text:
            gid = self.cmap.get(ord(ch), 0)
            self.used_gids.add(gid)
            hex_bytes.append(f"{gid:04X}")
        return "<" + "".join(hex_bytes) + ">"

    def text_width(self, text: str, size: float) -> float:
        total = 0.0
        for ch in text:
            gid = self.cmap.get(ord(ch), 0)
            total += self.glyph_width(gid)
        return total * size / 1000


class PDFWriter:
    def __init__(self) -> None:
        self.objects: List[bytes] = []

    def add_object(self, content: bytes) -> int:
        self.objects.append(content)
        return len(self.objects)

    def build(self, root_obj: int) -> bytes:
        output = [b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"]
        offsets = [0]
        for idx, obj in enumerate(self.objects, start=1):
            offsets.append(sum(len(chunk) for chunk in output))
            output.append(f"{idx} 0 obj\n".encode("ascii"))
            output.append(obj)
            output.append(b"\nendobj\n")
        xref_offset = sum(len(chunk) for chunk in output)
        output.append(f"xref\n0 {len(offsets)}\n".encode("ascii"))
        output.append(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            output.append(f"{off:010d} 00000 n \n".encode("ascii"))
        output.append(
            f"trailer\n<< /Size {len(offsets)} /Root {root_obj} 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode(
                "ascii"
            )
        )
        return b"".join(output)


def font_objects(font: TrueTypeFont, alias: str) -> Tuple[int, int, int, int, int, List[bytes]]:
    writer = PDFWriter()
    font_file_stream = (
        b"<< /Length "
        + str(len(font.data)).encode("ascii")
        + b" >>\nstream\n"
        + font.data
        + b"\nendstream"
    )
    font_file_obj = writer.add_object(font_file_stream)

    x_min, y_min, x_max, y_max = font.bbox
    font_descriptor = (
        b"<< /Type /FontDescriptor /FontName /"
        + alias.encode("ascii")
        + b" /Flags 4 /FontBBox ["
        + f"{x_min} {y_min} {x_max} {y_max}".encode("ascii")
        + b"] /ItalicAngle 0 /Ascent "
        + str(font.ascender).encode("ascii")
        + b" /Descent "
        + str(font.descender).encode("ascii")
        + b" /CapHeight "
        + str(font.ascender).encode("ascii")
        + b" /StemV 80 /FontFile2 "
        + f"{font_file_obj} 0 R".encode("ascii")
        + b" >>"
    )
    font_descriptor_obj = writer.add_object(font_descriptor)

    used = sorted(gid for gid in font.used_gids if gid != 0)
    widths_parts: List[str] = []
    if used:
        start = prev = used[0]
        current_widths = [f"{font.glyph_width(start):.2f}"]
        for gid in used[1:]:
            if gid == prev + 1:
                current_widths.append(f"{font.glyph_width(gid):.2f}")
            else:
                widths_parts.append(f"{start} [" + " ".join(current_widths) + "]")
                start = gid
                current_widths = [f"{font.glyph_width(gid):.2f}"]
            prev = gid
        widths_parts.append(f"{start} [" + " ".join(current_widths) + "]")
    widths_value = " ".join(widths_parts)

    cid_font = (
        b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /"
        + alias.encode("ascii")
        + b" /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >>"
        + b" /FontDescriptor "
        + f"{font_descriptor_obj} 0 R".encode("ascii")
        + b" /W ["
        + widths_value.encode("ascii")
        + b"] /DW 1000 /CIDToGIDMap /Identity >>"
    )
    cid_font_obj = writer.add_object(cid_font)

    to_unicode_stream = build_to_unicode(font)
    to_unicode_obj = writer.add_object(to_unicode_stream)

    type0_font = (
        b"<< /Type /Font /Subtype /Type0 /BaseFont /"
        + alias.encode("ascii")
        + b" /Encoding /Identity-H /DescendantFonts ["
        + f"{cid_font_obj} 0 R".encode("ascii")
        + b"] /ToUnicode "
        + f"{to_unicode_obj} 0 R".encode("ascii")
        + b" >>"
    )
    type0_font_obj = writer.add_object(type0_font)

    return (
        type0_font_obj,
        cid_font_obj,
        font_descriptor_obj,
        font_file_obj,
        to_unicode_obj,
        writer.objects,
    )


def build_to_unicode(font: TrueTypeFont) -> bytes:
    entries = []
    for gid in sorted(gid for gid in font.used_gids if gid != 0):
        chars = [code for code, mapped_gid in font.cmap.items() if mapped_gid == gid]
        if not chars:
            continue
        codepoint = chars[0]
        entries.append((gid, codepoint))
    blocks = []
    for i in range(0, len(entries), 100):
        chunk = entries[i : i + 100]
        block = f"{len(chunk)} beginbfchar\n"
        for gid, codepoint in chunk:
            block += f"<{gid:04X}> <{codepoint:04X}>\n"
        block += "endbfchar\n"
        blocks.append(block)
    cmap = (
        "/CIDInit /ProcSet findresource begin\n"
        "12 dict begin\n"
        "begincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> def\n"
        "/CMapName /Identity-H def\n"
        "/CMapType 2 def\n"
        "1 begincodespacerange\n"
        "<0000> <FFFF>\n"
        "endcodespacerange\n"
        + "".join(blocks)
        + "endcmap\n"
        "CMapName currentdict /CMap defineresource pop\n"
        "end\n"
        "end\n"
    ).encode("ascii")
    return b"<< /Length " + str(len(cmap)).encode("ascii") + b" >>\nstream\n" + cmap + b"endstream"


def layout_lines(
    blocks: Iterable[Tuple[str, Style]],
    fonts: Dict[str, TrueTypeFont],
    page_width: float,
    page_height: float,
    margin: float,
) -> List[List[Line]]:
    pages: List[List[Line]] = [[]]
    y = page_height - margin
    for text, style in blocks:
        y -= style.space_before
        for raw_line in text.split("\n"):
            line = raw_line
            if y - style.leading < margin:
                pages.append([])
                y = page_height - margin
            x = margin + style.indent
            if style.align == "center":
                width = fonts[style.font_key].text_width(line, style.size)
                x = (page_width - width) / 2
            pages[-1].append(Line(line, style.font_key, style.size, x, y))
            y -= style.leading
        y -= style.space_after
    return pages


def build_content_stream(pages: List[List[Line]], fonts: Dict[str, TrueTypeFont]) -> List[bytes]:
    streams: List[bytes] = []
    for page in pages:
        parts = ["BT"]
        for line in page:
            font_key = line.font_key
            font_alias = "F1" if font_key == "regular" else "F2"
            encoded = fonts[font_key].encode_text(line.text)
            parts.append(f"/{font_alias} {line.size:.2f} Tf")
            parts.append(f"1 0 0 1 {line.x:.2f} {line.y:.2f} Tm")
            parts.append(f"{encoded} Tj")
        parts.append("ET")
        data = "\n".join(parts).encode("ascii")
        streams.append(b"<< /Length " + str(len(data)).encode("ascii") + b" >>\nstream\n" + data + b"\nendstream")
    return streams


def main() -> None:
    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    fonts = {
        "regular": TrueTypeFont(font_dir / "DejaVuSans.ttf", "DejaVuSans"),
        "bold": TrueTypeFont(font_dir / "DejaVuSans-Bold.ttf", "DejaVuSans-Bold"),
    }

    styles = {
        "title": Style("bold", 18, 24, 0, 0, 8, "center"),
        "question": Style("bold", 13, 18, 0, 4, 4),
        "part": Style("bold", 11, 15, 0, 2, 2),
        "body": Style("regular", 11, 15, 12, 0, 2),
        "sub": Style("regular", 11, 15, 24, 0, 2),
        "separator": Style("regular", 11, 15, 0, 2, 2, "center"),
    }

    blocks: List[Tuple[str, Style]] = [
        ("Answers", styles["title"]),
        ("⸻", styles["separator"]),
        ("Question 1", styles["question"]),
        ("A. Write each ratio in its simplest form:", styles["part"]),
        ("i) 35 : 15", styles["body"]),
        ("35 ÷ 5 : 15 ÷ 5\n= 7 : 3", styles["sub"]),
        ("ii) 1/3 : 3/4", styles["body"]),
        ("Multiply both terms by 12:\n(1/3 × 12) : (3/4 × 12)\n= 4 : 9", styles["sub"]),
        ("⸻", styles["separator"]),
        ("B. €58.50 is divided between Ann and Barry in the ratio 8 : 5.", styles["part"]),
        ("How much does each person receive?", styles["body"]),
        ("Total parts = 8 + 5 = 13", styles["sub"]),
        ("Ann’s share = (8/13) × 58.50 = €36.00\nBarry’s share = (5/13) × 58.50 = €22.50", styles["sub"]),
        ("⸻", styles["separator"]),
        ("C. Sam and Tina share a bag of sweets in the ratio 2 : 3.", styles["part"]),
        ("If Sam receives 18 sweets, how many sweets will Tina receive?", styles["body"]),
        ("2 parts = 18 sweets\n1 part = 9 sweets", styles["sub"]),
        ("Tina = 3 × 9 = 27 sweets", styles["sub"]),
        ("⸻", styles["separator"]),
        ("Question 2", styles["question"]),
        ("12 men can paint a school building in 10 days.", styles["body"]),
        ("Total work = 12 × 10 = 120 man-days", styles["sub"]),
        ("a) How long would it take one man to paint the same school by himself?", styles["part"]),
        ("= 120 days", styles["sub"]),
        ("b) How many men would it take to paint the same school in 15 days?", styles["part"]),
        ("120 ÷ 15 = 8 men", styles["sub"]),
        ("⸻", styles["separator"]),
        ("Question 4", styles["question"]),
        ("A. Find the angles x and y in the diagram, giving reasons for each answer.", styles["part"]),
        ("In triangle ACB:\n55° + 80° + y = 180°\ny = 45°", styles["sub"]),
        ("Since AC ∥ BE, corresponding angles are equal:\nx = 55°", styles["sub"]),
        ("⸻", styles["separator"]),
        ("B. Find the value of x and the value of y.", styles["part"]),
        ("Triangle angles:\n72° + 2x° + 4y° = 180°\n→ x + 2y = 54", styles["sub"]),
        ("Straight line angles:\n4y + 5x = 180", styles["sub"]),
        ("Solving simultaneously:\nx = 24°, y = 15°", styles["sub"]),
        ("⸻", styles["separator"]),
        ("Question 5", styles["question"]),
        ("A. Simplify:", styles["part"]),
        ("4(2x + 1) + 3(5x − 2)", styles["body"]),
        ("= 8x + 4 + 15x − 6\n= 23x − 2", styles["sub"]),
        ("⸻", styles["separator"]),
        ("B. Simplify:", styles["part"]),
        ("−2a(a − 3y) − a(a + 4y)", styles["body"]),
        ("= −2a² + 6ay − a² − 4ay\n= −3a² + 2ay", styles["sub"]),
        ("⸻", styles["separator"]),
        ("C. Simplify:", styles["part"]),
        ("(x + 4)(x − 3)", styles["body"]),
        ("= x² + x − 12", styles["sub"]),
        ("⸻", styles["separator"]),
        ("Question 6", styles["question"]),
        ("If t = 4 and p = −3, find the value of:", styles["part"]),
        ("2t − 3p²", styles["body"]),
        ("= 2(4) − 3(9)\n= 8 − 27\n= −19", styles["sub"]),
        ("⸻", styles["separator"]),
        ("Question 7", styles["question"]),
        ("A. Solve for x:", styles["part"]),
        ("2x + 7 = 4x − 5", styles["body"]),
        ("2x = 12\nx = 6", styles["sub"]),
        ("⸻", styles["separator"]),
        ("B. Solve for y:", styles["part"]),
        ("5(y − 2) + 12 = 2(y − 5)", styles["body"]),
        ("5y + 2 = 2y − 10\n3y = −12\ny = −4", styles["sub"]),
        ("⸻", styles["separator"]),
        ("Question 8", styles["question"]),
        ("Bart has x euro.\nLisa has x + 12 euro.\nMaggie has 4x euro.", styles["body"]),
        ("Equation:\nx + (x + 12) = 4x", styles["sub"]),
        ("2x + 12 = 4x\nx = 6", styles["sub"]),
        ("Bart has €6\n(Lisa €18, Maggie €24)", styles["sub"]),
        ("⸻", styles["separator"]),
        ("Question 9", styles["question"]),
        ("The length of a rectangle is 3 cm longer than its width.", styles["body"]),
        ("Let width = x cm", styles["sub"]),
        ("a) Length = x + 3 cm", styles["part"]),
        ("b) Perimeter", styles["part"]),
        ("= 2(x + x + 3)\n= 4x + 6 cm", styles["sub"]),
        ("c) If the perimeter is 26 cm:", styles["part"]),
        ("4x + 6 = 26\nx = 5 cm", styles["sub"]),
        ("⸻", styles["separator"]),
        ("Question 10", styles["question"]),
        ("A. Solve:", styles["part"]),
        ("5x − 7 > 3, x ∈ ℕ", styles["body"]),
        ("5x > 10\nx > 2", styles["sub"]),
        ("x ≥ 3", styles["sub"]),
        ("⸻", styles["separator"]),
        ("B. Solve:", styles["part"]),
        ("7x + 1 ≤ 3x − 15, x ∈ ℝ", styles["body"]),
        ("4x ≤ −16\nx ≤ −4", styles["sub"]),
        ("⸻", styles["separator"]),
        ("Question 11", styles["question"]),
        ("A. Solve:", styles["part"]),
        ("x + y = 5\nx − y = −7", styles["body"]),
        ("2x = −2\nx = −1\ny = 6", styles["sub"]),
        ("⸻", styles["separator"]),
        ("B. Solve:", styles["part"]),
        ("3x + 4y = 5\n5x − 6y = 2", styles["body"]),
        ("x = 1, y = ½", styles["sub"]),
        ("⸻", styles["separator"]),
        ("Question 12", styles["question"]),
        ("A. Use Pythagoras’ Theorem to find side length a.", styles["part"]),
        ("a² = 10² + 12²\na = √244\na = 2√61", styles["sub"]),
        ("⸻", styles["separator"]),
        ("B. Find side length f.", styles["part"]),
        ("f² = 5² − 2²\nf = √21", styles["sub"]),
        ("⸻", styles["separator"]),
        ("Question 13", styles["question"]),
        ("Area = base × height = 88 cm²\nBase = 11 cm", styles["body"]),
        ("h = 88 ÷ 11\nh = 8 cm", styles["sub"]),
        ("⸻", styles["separator"]),
        ("Question 14", styles["question"]),
        ("Find the circumference of a circle with diameter 18 cm.", styles["body"]),
        ("C = πd\n= 18π\n≈ 56.5 cm (to 1 d.p.)", styles["sub"]),
        ("⸻", styles["separator"]),
        ("Question 15", styles["question"]),
        ("a) Work out the area of the front face of the shed.", styles["part"]),
        ("Rectangle area = 8 × 7 = 56 m²\nTriangle area = ½ × 8 × 3 = 12 m²", styles["sub"]),
        ("Total area = 68 m²", styles["sub"]),
        ("⸻", styles["separator"]),
        ("b) Hence work out the capacity of the shed in litres", styles["part"]),
        ("(1 m³ = 1,000 litres)", styles["body"]),
        ("Volume = 68 × 20 = 1,360 m³\n= 1,360,000 litres", styles["sub"]),
        ("⸻", styles["separator"]),
        ("Question 16", styles["question"]),
        ("A cylinder with radius 15 cm and height 24 cm", styles["body"]),
        ("Volume = πr²h\n= π × 15² × 24\n= 5400π cm³", styles["sub"]),
        ("⸻", styles["separator"]),
        ("Question 17", styles["question"]),
        ("A sphere of radius 7 cm fits exactly into a cube.", styles["body"]),
        ("A. Volume of the sphere", styles["part"]),
        ("= (4/3)πr³\n= (4/3)π(7³)\n= (1372/3)π cm³", styles["sub"]),
        ("⸻", styles["separator"]),
        ("B. Volume of the box", styles["part"]),
        ("Side length = 14 cm", styles["body"]),
        ("14³ = 2744 cm³", styles["sub"]),
        ("⸻", styles["separator"]),
        ("C. Volume not occupied by the sphere", styles["part"]),
        ("2744 − (1372/3)π\n≈ 1307 cm³ (nearest cm³)", styles["sub"]),
        ("⸻", styles["separator"]),
        ("D. Percentage of the box not occupied", styles["part"]),
        ("(1307 ÷ 2744) × 100\n≈ 47.6%", styles["sub"]),
    ]

    page_width = 595.28
    page_height = 841.89
    margin = 56.7

    pages = layout_lines(blocks, fonts, page_width, page_height, margin)
    content_streams = build_content_stream(pages, fonts)

    writer = PDFWriter()

    reg_objs = font_objects(fonts["regular"], "DejaVuSans")
    bold_objs = font_objects(fonts["bold"], "DejaVuSans-Bold")

    base_index = len(writer.objects)
    writer.objects.extend(reg_objs[5])
    reg_font_obj = base_index + reg_objs[0]

    base_index = len(writer.objects)
    writer.objects.extend(bold_objs[5])
    bold_font_obj = base_index + bold_objs[0]

    content_obj_ids = []
    for stream in content_streams:
        content_obj_ids.append(writer.add_object(stream))

    pages_kids = []
    for content_obj in content_obj_ids:
        page_obj = (
            b"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 595.28 841.89]"
            b" /Resources << /Font << /F1 "
            + f"{reg_font_obj} 0 R".encode("ascii")
            + b" /F2 "
            + f"{bold_font_obj} 0 R".encode("ascii")
            + b" >> >> /Contents "
            + f"{content_obj} 0 R".encode("ascii")
            + b" >>"
        )
        pages_kids.append(writer.add_object(page_obj))

    kids_refs = " ".join(f"{kid} 0 R" for kid in pages_kids)
    pages_obj = writer.add_object(
        f"<< /Type /Pages /Kids [{kids_refs}] /Count {len(pages_kids)} >>".encode("ascii")
    )

    updated_pages = []
    for kid in pages_kids:
        page_content = writer.objects[kid - 1]
        updated = page_content.replace(b"/Parent 0 0 R", f"/Parent {pages_obj} 0 R".encode("ascii"))
        updated_pages.append(updated)
    for idx, kid in enumerate(pages_kids):
        writer.objects[kid - 1] = updated_pages[idx]

    catalog_obj = writer.add_object(f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode("ascii"))

    OUTPUT_PATH.write_bytes(writer.build(catalog_obj))

    if not OUTPUT_PATH.exists() or OUTPUT_PATH.stat().st_size == 0:
        raise SystemExit("Failed to write answers_exam_style.pdf")
    print("Wrote answers_exam_style.pdf")


if __name__ == "__main__":
    main()
