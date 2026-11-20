from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PdfReadError


class PDFLoadError(Exception):
    """Raised when a PDF cannot be loaded or read."""


def load_pdf(source: str | Path) -> list[Document]:
    path = Path(source).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"PDF source must be a file: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Unsupported file type: {path.suffix or '(none)'}")

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise PDFLoadError(f"Encrypted PDF is not supported: {path}")

        documents: list[Document] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "file_name": path.name,
                        "page": page_number,
                    },
                )
            )
        return documents
    except PDFLoadError:
        raise
    except (PdfReadError, FileNotDecryptedError) as error:
        raise PDFLoadError(f"Unable to read PDF: {path}") from error
    except Exception as error:
        raise PDFLoadError(f"Failed to extract text from PDF: {path}") from error
