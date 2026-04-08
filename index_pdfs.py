from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

from rag_utils import save_vector_store_id

PDF_FOLDER = "docs"


def main():
    load_dotenv()
    client = OpenAI()

    pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError("No PDFs found in docs/")

    # 1. Create vector store
    vs = client.vector_stores.create(name="simple-pdf-rag")

    # 2. Upload PDFs and wait until done
    client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vs.id,
        files=[open(p, "rb") for p in pdf_files],
    )

    # 3. Save vector store ID
    save_vector_store_id(vs.id)

    print("Indexing complete")
    print("Vector Store ID:", vs.id)


if __name__ == "__main__":
    main()
