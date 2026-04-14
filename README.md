# AI Engineering con Python — Entrega
## Requisitos

- Python 3.10+ recomendado

Instalar dependencias:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

## Ollama en local (recomendado) / API Key

Puedes usar **Ollama en local** (sin API key) o **Gemini** (con API key).

### Opción A: Ollama (local, sin key)

En `.env`:

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b
```

### Opción B: Gemini (con key)

Crea un archivo `.env` con:

```
GOOGLE_API_KEY=TU_API_KEY
```

## Ejercicio 1 — Traductor de jerga técnica

```bash
python3 ejercicio1_traductor_jerga.py
```

## Ejercicio 2 — Filtro de candidatos (embeddings + FAISS)

```bash
python3 ejercicio2_filtro_candidatos.py
```

## Ejercicio 3 — Detective de chunks (RAG)

PDF por defecto: `Rivas_Guia_basica_uso_inteligencia_artificial_generativa_2025.pdf`

Prueba A (chunks minúsculos):

```bash
python3 ejercicio3_detective_chunks.py --chunk-size 100 --chunk-overlap 0
```

Prueba B (chunks enormes):

```bash
python3 ejercicio3_detective_chunks.py --chunk-size 5000 --chunk-overlap 500
```

Otro PDF o carpeta con PDFs:

```bash
python3 ejercicio3_detective_chunks.py --pdf "ruta/al/archivo.pdf" --chunk-size 1000 --chunk-overlap 100
```

## Ejercicio 4 — Agente con doble tool

PDF usado por el asistente: `normativa/guia_normativa_ejemplo.pdf`

```bash
python3 ejercicio4_agente_doble_tool.py
```