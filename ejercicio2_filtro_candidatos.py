from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings


def filtro_candidatos() -> None:
    # Modelo multilingüe para que el ejemplo funcione bien en español.
    embeddings = FastEmbedEmbeddings(model_name="jinaai/jina-embeddings-v2-base-es")

    perfiles = [
        "Desarrollador backend en Python con 5 años: APIs REST, bases de datos y despliegues.",
        "Diseñador UX/UI especializado en interfaces móviles y sistemas de diseño.",
        "Ingeniero de datos: ETL, SQL, pipelines y orquestación.",
        "Programador JavaScript/Node.js: servicios web, colas y microservicios.",
        "Camarero con experiencia sirviendo mesas en eventos y restaurante (candidato trampa).",
    ]

    db = FAISS.from_texts(perfiles, embeddings)

    query_vacante = "Buscamos un programador para crear servidores y lógica de negocio"
    resultados = db.similarity_search_with_score(query_vacante, k=3)

    print(f"Vacante: {query_vacante}\n")
    print("Top 3 candidatos (menor distancia = mejor):")
    print("-" * 60)
    for i, (doc, score) in enumerate(resultados, start=1):
        print(f"{i}. Distancia: {score:.4f}")
        print(f"   Perfil: {doc.page_content}\n")


if __name__ == "__main__":
    filtro_candidatos()
