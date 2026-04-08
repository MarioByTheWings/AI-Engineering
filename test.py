import re
from dataclasses import dataclass

from ejercicio4_agente_doble_tool import configurar_asistente, responder


@dataclass
class TestCase:
    pregunta: str
    patrones_ok: list[str]
    descripcion: str


def run_tests() -> int:
    asistente = configurar_asistente()

    # Batería variada:
    # - Horas de módulos (tabla PDF)
    # - Normativa (RAG en PDF)
    # - Calendario de exámenes
    # - Memoria conversacional (seguimiento y "siguiente examen")
    tests = [
        TestCase(
            pregunta="¿Cuántas horas tiene el módulo de Programación?",
            patrones_ok=[r"\b256\b", r"Fuente:\s*página\s*125"],
            descripcion="Horas módulo Programación",
        ),
        TestCase(
            pregunta="¿Cuántas horas tiene Entornos de desarrollo?",
            patrones_ok=[r"\b96\b", r"Fuente:\s*página\s*125"],
            descripcion="Horas módulo Entornos",
        ),
        TestCase(
            pregunta="¿Cuántas horas tiene Formación en centros de trabajo?",
            patrones_ok=[r"\b370\b", r"Fuente:\s*página\s*125"],
            descripcion="Horas módulo FCT",
        ),
        TestCase(
            pregunta="¿Cuándo es el examen de despliegue?",
            patrones_ok=[r"22 de mayo", r"despliegue"],
            descripcion="Calendario examen despliegue",
        ),
        TestCase(
            pregunta="¿Y el siguiente examen?",
            patrones_ok=[r"27 de mayo", r"programaci[oó]n"],
            descripcion="Memoria + siguiente examen",
        ),
        TestCase(
            pregunta="Y el de base de datos?",
            patrones_ok=[r"\b192\b", r"Fuente:\s*página\s*125"],
            descripcion="Seguimiento horas por contexto",
        ),
        TestCase(
            pregunta="¿Y cuándo es el examen?",
            patrones_ok=[r"3 de junio", r"bases de datos"],
            descripcion="Memoria del último módulo para examen",
        ),
        TestCase(
            pregunta="¿Qué son las horas de libre configuración en el ciclo?",
            patrones_ok=[r"Fuente:\s*páginas?\s*"],
            descripcion="Normativa general por RAG con fuente",
        ),
    ]

    ok = 0
    print("=" * 72)
    print("BATERIA DE TESTS - ejercicio4_agente_doble_tool.py")
    print("=" * 72)

    for i, t in enumerate(tests, 1):
        respuesta = responder(asistente, t.pregunta)
        passed = all(re.search(p, respuesta, re.IGNORECASE) for p in t.patrones_ok)
        estado = "OK" if passed else "FALLO"
        print(f"\n[{i}] {estado} - {t.descripcion}")
        print(f"Q: {t.pregunta}")
        print(f"A: {respuesta}")
        if passed:
            ok += 1

    total = len(tests)
    print("\n" + "-" * 72)
    print(f"RESULTADO FINAL: {ok}/{total} tests OK")
    print("-" * 72)
    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(run_tests())

