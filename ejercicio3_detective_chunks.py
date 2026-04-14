import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_ollama import ChatOllama


def _strip_llm_fuente(answer: str) -> str:
    # Evitamos duplicar la línea "Fuente:" porque nosotros ya la calculamos con metadatos.
    lines = answer.splitlines()
    out = []
    for line in lines:
        if line.strip().lower().startswith("fuente:"):
            continue
        out.append(line)
    return "\n".join(out).rstrip()


def _resolve_pdf_paths(pdf_arg: str | None) -> list[str]:
    # Si el usuario pasa --pdf, puede ser un archivo o una carpeta.
    if pdf_arg:
        path = Path(pdf_arg)
        if path.is_file():
            if path.suffix.lower() != ".pdf":
                raise FileNotFoundError(f"El archivo indicado no es PDF: {pdf_arg}")
            return [str(path)]
        if path.is_dir():
            pdfs = sorted(path.rglob("*.pdf"))
            if not pdfs:
                raise FileNotFoundError(
                    f"No se encontraron archivos .pdf en la carpeta: {pdf_arg}"
                )
            return [str(p) for p in pdfs]
        raise FileNotFoundError(f"No existe la ruta indicada en --pdf: {pdf_arg}")

    # Sin --pdf: usar el PDF específico del ejercicio 3.
    default_pdf = Path("Rivas_Guia_basica_uso_inteligencia_artificial_generativa_2025.pdf")
    if not default_pdf.is_file():
        raise FileNotFoundError(
            "No se encontró el PDF por defecto del ejercicio 3: "
            "'Rivas_Guia_basica_uso_inteligencia_artificial_generativa_2025.pdf'. "
            "Pásalo con --pdf o coloca ese archivo en la raíz del proyecto."
        )
    return [str(default_pdf)]


def run_rag(pdf_paths: list[str], chunk_size: int, chunk_overlap: int) -> None:
    load_dotenv()
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    ollama_model = os.getenv("OLLAMA_MODEL", "mistral:7b")

    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"No existe el PDF: {pdf_path}")

    if ollama_base_url:
        llm = ChatOllama(base_url=ollama_base_url, model=ollama_model, temperature=0)
    else:
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError(
                "Configura OLLAMA_BASE_URL (local) o GOOGLE_API_KEY (Gemini) en el .env"
            )
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0,
            max_output_tokens=500,
            max_retries=2,
        )
    embeddings = FastEmbedEmbeddings()

    docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)

    prompt = ChatPromptTemplate.from_template(
        "Usa el siguiente contexto para responder. Si no está en el contexto, dilo claramente.\n\n"
        "Contexto:\n{context}\n\n"
        "Pregunta: {input}\n\n"
        "Además, al final incluye una línea 'Fuente:' indicando la(s) página(s) más relevantes."
    )

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(vector_store.as_retriever(), combine_docs_chain)

    pregunta = "¿Qué es un KPI y pon un ejemplo del documento?"
    response = rag_chain.invoke({"input": pregunta})

    paginas = []
    for doc in response.get("context", []):
        p = doc.metadata.get("page")
        if isinstance(p, int):
            paginas.append(p + 1)  # 0-indexed -> 1-indexed

    paginas_unicas = ", ".join(str(p) for p in sorted(set(paginas))) or "N/A"
    print(f"PDF(s) cargados: {len(pdf_paths)}")
    for pdf_path in pdf_paths:
        print(f"- {pdf_path}")
    print(f"Chunk size: {chunk_size} | Overlap: {chunk_overlap}")
    print("-" * 60)
    print("Pregunta:", pregunta)
    print("-" * 60)
    print(_strip_llm_fuente(response["answer"]))
    print(f"\nFuente: páginas {paginas_unicas}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejercicio 3: comparar chunking en RAG."
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help=(
            "Ruta a un PDF o carpeta con PDFs. "
            "Si no se indica, se usa "
            "'Rivas_Guia_basica_uso_inteligencia_artificial_generativa_2025.pdf'."
        ),
    )
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--chunk-overlap", type=int, required=True)
    args = parser.parse_args()

    pdf_paths = _resolve_pdf_paths(args.pdf)
    run_rag(pdf_paths, args.chunk_size, args.chunk_overlap)


if __name__ == "__main__":
    main()
