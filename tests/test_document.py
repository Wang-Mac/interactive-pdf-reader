from pathlib import Path

import pytest

from pdf_reader.document import PDFLoadError, load_pdf


def create_pdf(path: Path, pages: list[str]) -> None:
    font_object = 3 + 2 * len(pages)
    page_object_ids = [3 + 2 * index for index in range(len(pages))]
    stream_object_ids = [4 + 2 * index for index in range(len(pages))]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_object_ids)

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode()),
    ]

    for index, (_page_id, stream_id) in enumerate(
        zip(page_object_ids, stream_object_ids, strict=True)
    ):
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 "
            + str(font_object).encode()
            + b" 0 R >> >> /Contents "
            + str(stream_id).encode()
            + b" 0 R >>"
        )

        text = pages[index]
        text_bytes = (
            b"BT /F1 12 Tf 72 720 Td ("
            + text.replace("(", r"\(").replace(")", r"\)").encode()
            + b") Tj ET"
        )
        objects.append(
            b"<< /Length "
            + str(len(text_bytes)).encode()
            + b" >>\nstream\n"
            + text_bytes
            + b"\nendstream"
        )

    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_number} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode()
        + b" /Root 1 0 R >>\n"
        + b"startxref\n"
        + str(xref_offset).encode()
        + b"\n%%EOF\n"
    )
    path.write_bytes(pdf)


def test_load_pdf_extracts_text_and_page_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    create_pdf(pdf_path, ["First page", "Second page"])

    documents = load_pdf(pdf_path)

    assert len(documents) == 2
    assert "First page" in documents[0].page_content
    assert documents[0].metadata["source"] == str(pdf_path.resolve())
    assert documents[0].metadata["file_name"] == "sample.pdf"
    assert documents[0].metadata["page"] == 1
    assert "Second page" in documents[1].page_content
    assert documents[1].metadata["page"] == 2


def test_load_pdf_rejects_missing_directory_and_non_pdf(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "missing.pdf")

    with pytest.raises(IsADirectoryError):
        load_pdf(tmp_path)

    text_path = tmp_path / "notes.txt"
    text_path.write_text("not a pdf")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_pdf(text_path)


def test_load_pdf_rejects_invalid_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"not a PDF")

    with pytest.raises(PDFLoadError, match="Unable to read PDF"):
        load_pdf(pdf_path)
