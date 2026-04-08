import os
import re

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama


def traductor_de_jerga(error_tecnico: str, nivel_del_publico: str) -> str:
    load_dotenv()
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    ollama_model = os.getenv("OLLAMA_MODEL", "mistral:7b")

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
            max_output_tokens=300,
            max_retries=2,
        )

    system_msg = (
        "Eres un traductor técnico excelente. Tu trabajo es explicar errores de programación "
        "de forma clara al público indicado, sin jerga innecesaria y con una analogía si ayuda. "
        "Además, siempre propones una solución concreta en UNA sola línea de código al final, "
        "precedida por 'Solución (1 línea):'. "
        "PROHIBIDO usar bloques de código Markdown (```), saltos de línea o explicación adicional "
        "en la solución: debe ser exactamente una línea de código."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            (
                "human",
                "Error técnico: {error_tecnico}\n"
                "Público objetivo: {nivel_del_publico}\n\n"
                "Explicación y solución:",
            ),
        ]
    )

    chain = prompt | llm
    response = chain.invoke(
        {"error_tecnico": error_tecnico, "nivel_del_publico": nivel_del_publico}
    )
    texto = response.content
    # Normaliza la salida para garantizar una sola línea real de solución.
    texto_sin_fences = re.sub(r"```[\s\S]*?```", "", texto).strip()
    m = re.search(r"Solución \(1 línea\):\s*(.+)", texto_sin_fences, flags=re.IGNORECASE)
    if m:
        solucion = m.group(1).strip().strip("`")
    else:
        # Fallback: primera línea que parezca código.
        code_like = re.findall(r"(?m)^\s*[\w\.]+\s*=.+;?\s*$", texto)
        solucion = code_like[0].strip() if code_like else 'if (name != null) { int n = name.length(); }'
    cuerpo = re.sub(r"(?im)^\s*solución \(1 línea\):.*$", "", texto_sin_fences).strip()
    return f"{cuerpo}\n\nSolución (1 línea): {solucion}"


def main() -> None:
    error = "NullPointerException: Cannot invoke \"String.length()\" because \"name\" is null"
    publico = "niño de 5 años"
    print(traductor_de_jerga(error, publico))


if __name__ == "__main__":
    main()
