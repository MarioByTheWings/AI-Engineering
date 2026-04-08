import os
import re
import unicodedata
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    key = _norm(modulo)
    key_tokens = {t[:-1] if t.endswith("s") else t for t in key.split() if len(t) > 2}
    for nombre, fecha in calendario.items():
        n = _norm(nombre)
        if key == n or key in n or n in key:
            return f"Examen de {nombre}: {fecha}."
        n_tokens = {t[:-1] if t.endswith("s") else t for t in n.split() if len(t) > 2}
        if key_tokens and n_tokens:
            overlap = len(key_tokens & n_tokens) / len(key_tokens)
            if overlap >= 0.66:
                return f"Examen de {nombre}: {fecha}."
    return (
        f"No tengo fecha para '{modulo}'. Módulos con fecha: "
        + ", ".join(sorted(calendario.keys()))
        + "."
    )


@tool
def consultar_calendario_examenes(modulo: str) -> str:
    """
    Devuelve la fecha de examen de un módulo concreto.

    CUÁNDO USAR:
    - Preguntas sobre fechas: "¿cuándo es el examen de X?", "fecha de examen", "siguiente examen".

    CUÁNDO NO USAR:
    - Preguntas de normativa, horas, contenidos o requisitos del ciclo.
    """
    return consultar_calendario_examenes_fn(modulo)


@dataclass
class AsistenteLocal:
    rag_chain: object
    llm_general: object
    horas_modulos: list[tuple[str, str, str, int]]
    tools: dict[str, object] | None = None
    last_modulo: str | None = None
    last_horas_tipo: str | None = None
    last_exam_modulo: str | None = None
    chat_history: list[tuple[str, str]] | None = None


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )
    s = re.sub(r"[^a-z0-9\\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _extraer_tabla_horas_por_modulo(docs_por_pagina: list) -> list[tuple[str, str, str, int]]:
    entries: list[tuple[str, str, str, int]] = []
    row_re = re.compile(r"\b0\d{3}\.\s*(.+?)\.\s+(\d{2,4})\s+(\d{1,2})\b")
    row_no_weekly_re = re.compile(r"\b0\d{3}\.\s*(.+?)\.\s+(\d{2,4})\b")
    for d in docs_por_pagina:
        text = d.page_content or ""
        # Recompone cortes típicos OCR/PDF: "apli - caciones" -> "aplicaciones"
        text = re.sub(r"([a-zA-Z])\s*-\s*([a-zA-Z])", r"\1\2", text)
        text = re.sub(r"\s+", " ", text)
        page0 = d.metadata.get("page")
        page1 = (page0 + 1) if isinstance(page0, int) else None
        if page1 is None or not (120 <= page1 <= 130):
            continue
        for m in row_re.finditer(text):
            nombre = _norm(m.group(1))
            horas_totales = m.group(2)
            horas_semanales = m.group(3)
            if page1 is None:
                continue
            if len(nombre) > 70 or re.search(r"\b0\d{3}\b", nombre):
                continue
            entries.append((nombre, horas_totales, horas_semanales, page1))
        for m in row_no_weekly_re.finditer(text):
            nombre = _norm(m.group(1))
            horas_totales = m.group(2)
            if page1 is None:
                continue
            if len(nombre) > 70 or re.search(r"\b0\d{3}\b", nombre):
                continue
            # Evita duplicar entradas ya capturadas con horas semanales.
            if any(x[0] == nombre and x[1] == horas_totales and x[3] == page1 for x in entries):
                continue
            entries.append((nombre, horas_totales, "", page1))
    return entries


def _buscar_horas_modulo(
    entries: list[tuple[str, str, str, int]], modulo_query: str, tipo: str
) -> tuple[str | None, int | None]:
    q = _norm(modulo_query)
    q_tokens = {t for t in q.split() if len(t) > 2}
    best = (None, None, 0.0)  # horas, page, score
    for nombre, horas_totales, horas_semanales, page in entries:
        if q and (q in nombre or nombre in q):
            return (horas_semanales, page) if tipo == "semanales" else (horas_totales, page)
        n_tokens = {t for t in nombre.split() if len(t) > 2}
        if not q_tokens or not n_tokens:
            continue
        overlap = len(q_tokens & n_tokens) / len(q_tokens)
        if overlap > best[2]:
            horas = horas_semanales if tipo == "semanales" else horas_totales
            best = (horas, page, overlap)
    if best[2] >= 0.5:
        return best[0], best[1]
    return None, None


def _extraer_modulo_de_pregunta_horas(texto: str) -> str | None:
    t = _norm(texto)
    m = re.search(r"m[oó]dulo\s+de\s+(.+)$", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().strip("?.!")
    m = re.search(r"m[oó]dulo\s+(.+)$", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().strip("?.!")
    m = re.search(r"horas(?:\s+semanales)?\s+de\s+(.+)$", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().strip("?.!")
    m = re.search(r"horas\s+totales?\s+tiene\s+(.+)$", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().strip("?.!")
    m = re.search(r"horas\s+tiene\s+(.+)$", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().strip("?.!")
    return None


def _extraer_modulo_examen(texto: str) -> str | None:
    m = re.search(r"examen\s+de\s+(.+)$", texto.strip(), flags=re.IGNORECASE)
    if not m:
        return None
    modulo = m.group(1).strip().strip("?.!")
    return modulo or None


def _extraer_nombre_modulo_calendario(texto_respuesta: str) -> str | None:
    m = re.search(r"Examen de (.+?):", texto_respuesta)
    if not m:
        return None
    return _norm(m.group(1))


def _siguiente_modulo_calendario(modulo_actual: str | None) -> str | None:
    # Orden cronológico del calendario simulado (mayo -> junio)
    orden = ["despliegue", "programación", "bases de datos", "proyecto web"]
    if not modulo_actual:
        return orden[0]

    actual_norm = _norm(modulo_actual)
    idx_actual = None
    for i, nombre in enumerate(orden):
        n = _norm(nombre)
        if actual_norm == n or actual_norm in n or n in actual_norm:
            idx_actual = i
            break
    if idx_actual is None:
        return orden[0]
    if idx_actual + 1 >= len(orden):
        return None
    return orden[idx_actual + 1]


def _extraer_modulo_seguimiento(texto: str) -> str | None:
    """
    Detecta preguntas de continuidad tipo:
    - "Y el de base de datos?"
    - "y el de bases de datos"
    """
    t = _norm(texto)
    m = re.search(r"^y\s+(?:el|la)?\s*de\s+(.+)$", t)
    if not m:
        return None
    modulo = m.group(1).strip().strip("?.!")
    return modulo or None


def configurar_asistente() -> AsistenteLocal:
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

    embeddings = FastEmbedEmbeddings(model_name="jinaai/jina-embeddings-v2-base-es")
    vector_db = FAISS.from_documents(chunks, embeddings)

    retriever = vector_db.as_retriever(search_kwargs={"k": 8})
    rag_prompt = ChatPromptTemplate.from_template(
        "Responde usando SOLO el contexto recuperado.\n"
        "- Si el dato no aparece literal, di: 'No encontrado en el contexto'.\n"
        "- No inventes cifras.\n\n"
        "Contexto:\n{context}\n\n"
        "Pregunta: {input}\n"
        "Respuesta:\n"
    )

    if ollama_base_url:
        llm = ChatOllama(base_url=ollama_base_url, model=ollama_model, temperature=0)
    else:
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("Configura OLLAMA_BASE_URL o GOOGLE_API_KEY en el .env")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0,
            max_output_tokens=600,
            max_retries=2,
        )

    combine_docs_chain = create_stuff_documents_chain(llm, rag_prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    @tool
    def buscador_normativa(pregunta: str) -> str:
        """
        Busca información oficial en el PDF de normativa del ciclo.

        CUÁNDO USAR:
        - Dudas de normativa: módulos, horas, contenidos, evaluación, FCT, requisitos.
        - Preguntas que necesitan cita de fuente del PDF.

        CUÁNDO NO USAR:
        - Fechas de examen o calendario académico (usar consultar_calendario_examenes).
        """
        out = rag_chain.invoke({"input": pregunta})
        pages = sorted(
            {
                (d.metadata.get("page") + 1)
                for d in out.get("context", [])
                if isinstance(d.metadata.get("page"), int)
            }
        )
        pages_txt = ", ".join(str(p) for p in pages) if pages else "N/A"
        return f"{out.get('answer', '')}\nFuente: páginas {pages_txt}"

    tools = {
        "buscador_normativa": buscador_normativa,
        "consultar_calendario_examenes": consultar_calendario_examenes,
    }

    return AsistenteLocal(
        rag_chain=rag_chain,
        llm_general=llm,
        horas_modulos=horas_modulos,
        tools=tools,
        chat_history=[],
    )


def _registrar_turno(asistente_local: AsistenteLocal, user: str, assistant: str) -> str:
    if asistente_local.chat_history is None:
        asistente_local.chat_history = []
    asistente_local.chat_history.append((user, assistant))
    # Evita crecimiento infinito de memoria en sesiones largas.
    asistente_local.chat_history = asistente_local.chat_history[-20:]
    return assistant


def _ultimo_user(asistente_local: AsistenteLocal) -> str | None:
    if not asistente_local.chat_history:
        return None
    return asistente_local.chat_history[-1][0]


def _es_repregunta_corta(texto_norm: str) -> bool:
    return bool(
        re.match(
            r"^(y|entonces|vale|ok)\b|^(el|la|los|las)\b|^siguiente\b|^cuando\b|^cuanto\b",
            texto_norm,
        )
    )


def responder(asistente_local: AsistenteLocal, user: str) -> str:
    txt = user.strip()
    txt_norm = _norm(txt)
    last_user = _ultimo_user(asistente_local)

    # Reglas de decisión entre fuentes:
    # 1) Calendario de exámenes -> consultar_calendario_examenes
    # 2) Normativa / horas / contenidos -> buscador_normativa (y extractor de tabla para horas)
    # 3) Otros temas -> LLM general con historial

    # Conserva contexto: repreguntas cortas heredan el tema del turno anterior.
    if _es_repregunta_corta(txt_norm) and last_user:
        txt_contextual = f"Contexto previo del usuario: {last_user}\nPregunta actual: {txt}"
    else:
        txt_contextual = txt
    if "siguiente examen" in txt_norm or "proximo examen" in txt_norm or "próximo examen" in txt.lower():
        siguiente = _siguiente_modulo_calendario(asistente_local.last_exam_modulo)
        if not siguiente:
            return _registrar_turno(
                asistente_local,
                user,
                "No hay más exámenes posteriores en el calendario que tengo cargado.",
            )
        asistente_local.last_exam_modulo = siguiente
        return _registrar_turno(asistente_local, user, consultar_calendario_examenes_fn(siguiente))

    if "examen" in txt.lower():
        modulo = _extraer_modulo_examen(txt) or asistente_local.last_modulo
        if not modulo:
            return _registrar_turno(
                asistente_local, user, "¿De qué módulo quieres saber la fecha del examen?"
            )
        asistente_local.last_modulo = modulo
        respuesta = consultar_calendario_examenes_fn(modulo)
        modulo_cal = _extraer_nombre_modulo_calendario(respuesta)
        if modulo_cal:
            asistente_local.last_exam_modulo = modulo_cal
        return _registrar_turno(asistente_local, user, respuesta)

    # Soporte de memoria conversacional para repreguntas cortas.
    # Si venimos de una consulta de horas, "Y el de X?" mantiene el mismo tipo.
    modulo_seguimiento = _extraer_modulo_seguimiento(txt)
    if modulo_seguimiento and asistente_local.last_horas_tipo in {"totales", "semanales"}:
        horas_exactas, page = _buscar_horas_modulo(
            asistente_local.horas_modulos, modulo_seguimiento, asistente_local.last_horas_tipo
        )
        if horas_exactas and page:
            asistente_local.last_modulo = modulo_seguimiento
            sufijo = "" if asistente_local.last_horas_tipo == "semanales" else " horas"
            return _registrar_turno(
                asistente_local, user, f"{horas_exactas}{sufijo}\nFuente: página {page}"
            )

    if any(k in txt.lower() for k in ["módulo", "modulo", "horas", "normativa", "evaluación", "evaluacion"]):
        if "hora" in txt.lower() or "horas" in txt.lower():
            modulo = _extraer_modulo_de_pregunta_horas(txt)
            if modulo:
                modulo = re.sub(r"^(el|la)\\s+", "", modulo)
                modulo = re.sub(r"^modulo\\s+de\\s+", "", modulo)
                modulo = re.sub(r"^modulo\\s+", "", modulo)
                tipo = "semanales" if "semanal" in txt.lower() else "totales"
                horas_exactas, page = _buscar_horas_modulo(asistente_local.horas_modulos, modulo, tipo)
                if horas_exactas and page:
                    asistente_local.last_modulo = modulo
                    asistente_local.last_horas_tipo = tipo
                    sufijo = "" if tipo == "semanales" else " horas"
                    return _registrar_turno(
                        asistente_local, user, f"{horas_exactas}{sufijo}\nFuente: página {page}"
                    )

        out = asistente_local.rag_chain.invoke({"input": txt_contextual})
        pages = sorted(
            {
                (d.metadata.get("page") + 1)
                for d in out.get("context", [])
                if isinstance(d.metadata.get("page"), int)
            }
        )
        pages_txt = ", ".join(str(p) for p in pages) if pages else "N/A"
        return _registrar_turno(
            asistente_local, user, f"{out.get('answer', '')}\nFuente: páginas {pages_txt}"
        )

    # También cubre frases sin "horas", pero claramente de continuidad.
    if txt_norm.startswith("y ") and asistente_local.last_horas_tipo in {"totales", "semanales"}:
        modulo = re.sub(r"^y\s+(?:el|la)?\s*", "", txt_norm).strip("?.! ")
        if modulo:
            horas_exactas, page = _buscar_horas_modulo(
                asistente_local.horas_modulos, modulo, asistente_local.last_horas_tipo
            )
            if horas_exactas and page:
                sufijo = "" if asistente_local.last_horas_tipo == "semanales" else " horas"
                return _registrar_turno(
                    asistente_local, user, f"{horas_exactas}{sufijo}\nFuente: página {page}"
                )

    p = ChatPromptTemplate.from_messages(
        [
            ("system", "Responde de forma breve y clara usando el historial reciente como contexto."),
            ("human", "Historial reciente:\n{historial}\n\nPregunta actual:\n{q}"),
        ]
    )
    historial = "\n".join(
        [f"Usuario: {u}\nAsistente: {a}" for u, a in (asistente_local.chat_history or [])[-6:]]
    )
    resp = (p | asistente_local.llm_general).invoke({"q": txt, "historial": historial})
    return _registrar_turno(asistente_local, user, resp.content)


def chat_asistente():
    asistente = configurar_asistente()

    print("\n" + "=" * 50)
    print("SISTEMA DE ASISTENCIA DUAL (Normativa + Calendario)")
    print("   Escribe 'salir' para finalizar")
    print("=" * 50 + "\n")

    while True:
        usuario = input("Tú: ")
        if usuario.lower() in ["salir", "exit"]:
            break
        try:
            print(f"\nAsistente: {responder(asistente, usuario)}\n")
        except Exception as e:
            print(f"Error en la comunicación: {e}")


if __name__ == "__main__":
    chat_asistente()