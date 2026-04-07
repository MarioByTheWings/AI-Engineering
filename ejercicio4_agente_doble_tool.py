import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import create_retriever_tool
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain


def consultar_calendario_examenes_fn(modulo: str) -> str:
    """Devuelve fechas de exámenes por módulo."""
    calendario = {
        "proyecto web": "15 de junio",
        "despliegue": "22 de mayo",
        "bases de datos": "3 de junio",
        "programación": "27 de mayo",
    }
    key = modulo.strip().lower()
    if key in calendario:
        return f"Examen de {modulo}: {calendario[key]}."
    return (
        f"No tengo fecha para '{modulo}'. Módulos con fecha: "
        + ", ".join(sorted(calendario.keys()))
        + "."
    )


@tool
def consultar_calendario_examenes(modulo: str) -> str:
    """Tool wrapper para consultar fechas de exámenes."""
    return consultar_calendario_examenes_fn(modulo)


@dataclass
class AsistenteLocal:
    rag_chain: object
    llm_general: object
    horas_modulos: list[tuple[str, str, str, int]]  # (nombre_modulo_norm, horas_totales, horas_semanales, pagina_1_indexed)
    last_modulo: str | None = None


def _extraer_modulo(texto: str) -> str | None:
    m = re.search(r"examen\s+de\s+(.+)$", texto.strip(), flags=re.IGNORECASE)
    if not m:
        return None
    modulo = m.group(1).strip().strip("?.!")
    return modulo or None


def _extraer_modulo_de_pregunta_horas(texto: str) -> str | None:
    """
    Intenta extraer el nombre del módulo desde frases tipo:
    - "cuántas horas tiene el módulo de entornos de desarrollo?"
    - "horas de bases de datos"
    """
    t = texto.strip().lower()
    m = re.search(r"m[oó]dulo\s+de\s+(.+)$", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().strip("?.!")
    m = re.search(r"m[oó]dulo\s+(.+)$", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().strip("?.!")
    m = re.search(r"horas\s+de\s+(.+)$", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().strip("?.!")
    return None


def _actualizar_ultima_mencion_modulo(asistente: AsistenteLocal, texto: str) -> None:
    """
    Para validar memoria: si el usuario menciona un módulo y luego pregunta
    '¿y cuándo es el examen?', recordamos el módulo.
    """
    m = re.search(r"m[oó]dulo\s+de\s+(.+)$", texto.strip(), flags=re.IGNORECASE)
    if m:
        asistente.last_modulo = m.group(1).strip().strip("?.!")


def _extraer_horas_literal(context_docs: list, modulo: str | None) -> tuple[str | None, str]:
    """
    Busca una cifra de horas literal en el contexto recuperado.
    Devuelve (horas, paginas_csv).
    """
    paginas = []
    blobs = []
    for d in context_docs:
        p = d.metadata.get("page")
        if isinstance(p, int):
            paginas.append(p + 1)
        blobs.append(d.page_content or "")
    text = "\n".join(blobs)

    # Si sabemos el módulo, intentamos la fila típica de tabla:
    # "0487. Entornos de desarrollo. 96 3"
    if modulo:
        mod_escaped = re.escape(modulo)
        # Solo aceptamos números si aparecen en una fila de módulo con código 0XXX.
        # Capturamos la columna de horas (2º número), no el código.
        m = re.search(
            rf"\b0\d{{3}}\.\s*{mod_escaped}.*?\b(\d{{2,4}})\b\s+\d+\b",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            horas = m.group(1)
            pages = ", ".join(str(p) for p in sorted(set(paginas))) or "N/A"
            return horas, pages

        # Alternativa: línea explícita de duración para ese módulo.
        m = re.search(
            rf"{mod_escaped}.*?Duraci[oó]n:\s*(\d{{2,4}})\s*horas?",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            horas = m.group(1)
            pages = ", ".join(str(p) for p in sorted(set(paginas))) or "N/A"
            return horas, pages

    # Fallback SOLO si no buscamos un módulo concreto
    if not modulo:
        m = re.search(r"Duraci[oó]n:\s*(\d{2,4})\s*horas", text, flags=re.IGNORECASE)
        if m:
            horas = m.group(1)
            pages = ", ".join(str(p) for p in sorted(set(paginas))) or "N/A"
            return horas, pages

    pages = ", ".join(str(p) for p in sorted(set(paginas))) or "N/A"
    return None, pages


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _extraer_tabla_horas_por_modulo(docs_por_pagina: list) -> list[tuple[str, str, int]]:
    """
    Extrae entradas tipo:
    '0487. Entornos de desarrollo. 96 3'
    Devuelve (nombre_modulo_normalizado, horas, pagina_1_indexed)
    """
    entries: list[tuple[str, str, str, int]] = []
    # Ejemplo de fila:
    # 0485. Programación. 256 8
    row_re = re.compile(r"\b0\d{3}\.\s*(.+?)\.\s+(\d{2,4})\s+(\d{1,2})\b")
    for d in docs_por_pagina:
        text = d.page_content or ""
        page0 = d.metadata.get("page")
        page1 = (page0 + 1) if isinstance(page0, int) else None
        for m in row_re.finditer(text):
            nombre = _norm(m.group(1))
            horas_totales = m.group(2)
            horas_semanales = m.group(3)
            if page1 is None:
                continue
            entries.append((nombre, horas_totales, horas_semanales, page1))
    return entries


def _buscar_horas_modulo(
    entries: list[tuple[str, str, str, int]], modulo_query: str, tipo: str
) -> tuple[str | None, int | None]:
    q = _norm(modulo_query)
    # Match por inclusión (el nombre del módulo suele estar completo en la tabla)
    for nombre, horas_totales, horas_semanales, page in entries:
        if q in nombre or nombre in q:
            if tipo == "semanales":
                return horas_semanales, page
            return horas_totales, page
    return None, None


def configurar() -> tuple[object, AsistenteLocal | None]:
    load_dotenv()
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    ollama_model = os.getenv("OLLAMA_MODEL", "mistral:7b")

    if not os.path.exists("normativa"):
        os.makedirs("normativa")
        raise RuntimeError(
            "He creado la carpeta 'normativa/'. Mete ahí uno o más PDFs y vuelve a ejecutar."
        )

    loader = PyPDFDirectoryLoader("normativa/")
    docs = loader.load()
    if not docs:
        raise RuntimeError("La carpeta 'normativa/' está vacía (no hay PDFs cargables).")

    horas_modulos = _extraer_tabla_horas_por_modulo(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Embeddings en español para mejorar la recuperación en preguntas en castellano.
    embeddings = FastEmbedEmbeddings(model_name="jinaai/jina-embeddings-v2-base-es")
    vector_db = FAISS.from_documents(chunks, embeddings)

    retriever = vector_db.as_retriever(search_kwargs={"k": 8})
    rag_prompt = ChatPromptTemplate.from_template(
        "Responde usando SOLO el contexto recuperado.\n"
        "- Si el número de horas NO aparece literal en el contexto, di: \"No encontrado en el contexto\".\n"
        "- NO sumes, NO extrapoles, NO deduzcas.\n\n"
        "Contexto:\n{context}\n\n"
        "Pregunta: {input}\n"
        "Respuesta:\n"
    )

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
            max_output_tokens=600,
            max_retries=2,
        )

    if ollama_base_url:
        combine_docs_chain = create_stuff_documents_chain(llm, rag_prompt)
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
        return llm, AsistenteLocal(rag_chain=rag_chain, llm_general=llm, horas_modulos=horas_modulos)

    buscador_normativa = create_retriever_tool(
        retriever=retriever,
        name="buscador_normativa",
        description=(
            "Úsala para buscar información en PDFs oficiales sobre ciclo, módulos, horas, "
            "evaluación, normativa y requisitos."
        ),
    )
    tools = [buscador_normativa, consultar_calendario_examenes]

    system_msg = (
        "Eres un asistente educativo. Tienes dos herramientas:\n"
        "- buscador_normativa: para dudas que estén en los PDFs.\n"
        "- consultar_calendario_examenes: SOLO para fechas de exámenes.\n\n"
        "Reglas:\n"
        "- Si la pregunta pide una fecha de examen, usa consultar_calendario_examenes.\n"
        "- Si la pregunta es sobre módulos/horas/normativa, usa buscador_normativa.\n"
        "- Si no encaja en ninguna, responde con conocimiento general y sé claro.\n"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=6,
    )
    return agent_executor, None


def responder_local(asistente_local: AsistenteLocal, user: str) -> str:
    """
    Mismo comportamiento que el modo local del chat, pero devolviendo un string.
    Útil para tests automáticos.
    """
    txt = user.strip()
    _actualizar_ultima_mencion_modulo(asistente_local, txt)

    if "examen" in txt.lower():
        modulo = _extraer_modulo(txt) or asistente_local.last_modulo
        if not modulo:
            return "¿De qué módulo quieres saber la fecha del examen?"
        asistente_local.last_modulo = modulo
        return consultar_calendario_examenes_fn(modulo)

    if any(
        k in txt.lower()
        for k in [
            "módulo",
            "modulo",
            "horas",
            "normativa",
            "evaluación",
            "evaluacion",
            "kpi",
            "superficie",
            "m2",
            "m²",
            "laboratorio",
            "aula",
            "taller",
            "instalación",
            "instalaciones",
        ]
    ):
        retrieval_input = txt
        if "hora" in txt.lower() or "horas" in txt.lower():
            modulo = _extraer_modulo_de_pregunta_horas(txt)
            if modulo:
                retrieval_input = (
                    f"{txt}\n"
                    f"Busca literalmente una fila tipo '0XXX. {modulo}. NNN' o 'Duración: NNN horas'."
                )
            else:
                retrieval_input = f"{txt} Duración: horas totales código del módulo (ej: 0487.)"

        out = asistente_local.rag_chain.invoke({"input": retrieval_input})

        if "hora" in txt.lower() or "horas" in txt.lower():
            modulo = _extraer_modulo_de_pregunta_horas(txt)
            if modulo:
                tipo = "semanales" if "semanal" in txt.lower() else "totales"
                horas_exactas, page = _buscar_horas_modulo(
                    asistente_local.horas_modulos, modulo, tipo
                )
                if horas_exactas and page:
                    sufijo = "" if tipo == "semanales" else " horas"
                    return f"{horas_exactas}{sufijo}\nFuente: página {page}"

            horas, paginas_unicas = _extraer_horas_literal(out.get("context", []), modulo)
            if horas:
                return f"{horas} horas\nFuente: páginas {paginas_unicas}"
            return (
                "No encontrado en el contexto (necesito que el PDF contenga la cifra literal).\n"
                f"Fuente: páginas {paginas_unicas}"
            )

        _, paginas_unicas = _extraer_horas_literal(out.get("context", []), None)
        return f"{out.get('answer','')}\nFuente: páginas {paginas_unicas}"

    p = ChatPromptTemplate.from_messages(
        [("system", "Responde de forma breve y clara."), ("human", "{q}")]
    )
    resp = (p | asistente_local.llm_general).invoke({"q": txt})
    return resp.content


def chat() -> None:
    runtime, asistente_local = configurar()

    print("=" * 44)
    print("AGENTE EDUCATIVO (normativa + calendario)")
    print("Escribe 'salir' para finalizar")
    print("=" * 44)

    while True:
        user = input("Tú: ").strip()
        if user.lower() in {"salir", "exit", "quit"}:
            break

        if asistente_local:
            respuesta = responder_local(asistente_local, user)
            print(f"Asistente: {respuesta}\n")
            continue

        out = runtime.invoke({"input": user})
        print(f"Asistente: {out['output']}\n")


if __name__ == "__main__":
    chat()
