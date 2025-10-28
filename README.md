# 🤖 Sistema RAG con FAISS y LLMs

Sistema de Recuperación Aumentada por Generación (RAG) que permite hacer preguntas sobre documentos técnicos utilizando embeddings vectoriales y modelos de lenguaje.

## 🚀 Descarga Rápida (Windows)

**[📥 Descargar Sistema_RAG_Portable.zip](../../releases/latest)**

### Requisitos
- Windows 10/11 (64-bit)
- **¡NO requiere Python ni instalaciones!**

### ¿Cómo usar?
1. Descomprime `Sistema_RAG_Portable.zip`
2. Doble clic en `Sistema_RAG.exe`
3. Se abrirá tu navegador automáticamente
4. ¡Haz preguntas sobre tus documentos!

---

## 🎯 ¿Qué hace este sistema?

Este sistema implementa RAG (Retrieval-Augmented Generation), una técnica que:

1. **Vectoriza** tus documentos PDF en una base de datos semántica
2. **Busca** los fragmentos más relevantes cuando haces una pregunta
3. **Genera** una respuesta usando IA basándose en el contexto encontrado

### Ejemplo Visual

```
┌─────────────────────────────────────────────────────────┐
│  "¿Qué parámetros se usan en IPC-7525A?"               │
│                    (Tu pregunta)                         │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  Vectorización         │
         │  [0.23, -0.45, 0.12...]│
         └────────────┬───────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  FAISS busca similares │
         │  Top 4 fragmentos      │
         └────────────┬───────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  LLM genera respuesta  │
         │  (OpenAI/Claude/etc)   │
         └────────────┬───────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  "Según IPC-7525A, los parámetros principales son..."   │
│                  (Respuesta IA)                          │
└─────────────────────────────────────────────────────────┘
```

---

## 📚 ¿Cómo funciona por dentro? (Explicación técnica)

### 1. Carga y Procesamiento de Documentos

```python
# api_rag.py - Líneas 120-140 (aproximado)

def cargar_documentos():
    """
    Carga PDFs desde uploaded_docs/ y los divide en chunks.
    Cada chunk es un fragmento de ~500 caracteres con overlap
    para mantener contexto entre fragmentos.
    """
    docs_dir = Path("uploaded_docs")
    documentos = []
    
    for pdf_file in docs_dir.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        documentos.extend(loader.load())
    
    # Dividir en chunks manejables
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # Tamaño de cada fragmento
        chunk_overlap=50     # Overlap para mantener contexto
    )
    
    return text_splitter.split_documents(documentos)
```

**¿Por qué chunks?** Los LLMs tienen límite de tokens. Dividir el documento permite:
- Buscar solo las partes relevantes
- Procesar documentos de cualquier tamaño
- Mantener contexto con el overlap

---

### 2. Vectorización con FAISS

```python
# api_rag.py - Líneas 145-165 (aproximado)

def crear_vectorstore(chunks):
    """
    Convierte texto en vectores matemáticos.
    Textos similares tienen vectores cercanos en el espacio.
    
    Ejemplo:
    "stencil design"    → [0.23, -0.45,  0.12, ...]
    "diseño de stencil" → [0.21, -0.43,  0.14, ...]  ← Similar!
    "perro gato"        → [-0.80, 0.32, -0.67, ...]  ← Diferente!
    """
    
    # Usar embeddings fake para demo (384 dimensiones)
    embeddings = FakeEmbeddings()
    
    # FAISS crea índice optimizado para búsqueda rápida
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Guardar para uso futuro
    vectorstore.save_local("vectorstore_faiss")
    
    return vectorstore
```

**¿Qué son embeddings?** Son representaciones numéricas del significado del texto. Palabras con significado similar tienen vectores similares.

---

### 3. Búsqueda Semántica (Retrieval)

```python
# api_rag.py - SimpleRetrievalQA.__call__

def buscar_contexto(pregunta):
    """
    Busca los chunks más relevantes usando similitud coseno.
    
    Matemáticamente:
    similitud = cos(θ) = (A·B) / (||A|| × ||B||)
    
    Donde A es el vector de la pregunta y B es cada chunk.
    """
    
    # Vectorizar pregunta
    query_vector = embeddings.embed_query(pregunta)
    # query_vector = [0.23, -0.45, 0.12, ...]
    
    # FAISS busca los 4 vectores más cercanos (k=4)
    docs_relevantes = vectorstore.similarity_search(pregunta, k=4)
    
    # Combinar contexto
    contexto = "\n\n".join([doc.page_content for doc in docs_relevantes])
    
    return contexto
```

**¿Por qué k=4?** Balance entre:
- Suficiente contexto para responder
- No saturar el límite de tokens del LLM
- Tiempo de procesamiento razonable

---

### 4. Generación con LLM

```python
# api_rag.py - SimpleRetrievalQA.__call__ (continuación)

def generar_respuesta(pregunta, contexto):
    """
    Usa el LLM para generar una respuesta coherente
    basándose SOLO en el contexto proporcionado.
    """
    
    # Crear prompt estructurado
    prompt = f"""Contexto:
{contexto}

Pregunta: {pregunta}

Respuesta (basándote SOLO en el contexto):"""
    
    # Llamar al LLM (OpenAI, Claude, etc.)
    messages = [
        SystemMessage(content="Eres un experto que responde basándote en el contexto."),
        HumanMessage(content=prompt)
    ]
    
    respuesta = llm.invoke(messages)
    
    return respuesta.content
```

**Ventaja del RAG:** El LLM no "inventa" respuestas, se basa en tus documentos reales.

---

### 5. Configuración Flexible de LLMs

```python
# api_rag.py - Líneas 200-250 (aproximado)

def inicializar_llm():
    """
    Soporta múltiples LLMs según variable de entorno LLM_TYPE
    """
    llm_type = os.getenv("LLM_TYPE", "fake").lower()
    
    if llm_type == "openai":
        # GPT-3.5/GPT-4 - Rápido y preciso
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,  # 0 = determinista, 1 = creativo
        )
    
    elif llm_type == "anthropic":
        # Claude - Muy bueno con documentos largos
        return ChatAnthropic(
            model="claude-3-sonnet-20240229",
            temperature=0,
        )
    
    elif llm_type == "huggingface":
        # Modelos open-source offline
        return HuggingFacePipeline.from_model_id(
            model_id="microsoft/DialoGPT-medium",
            task="text-generation",
        )
    
    else:
        # Modo fake para demos sin API
        return FakeChatModel()
```

---

## 🛠️ Arquitectura Completa

```python
# Flujo completo de una pregunta

async def endpoint_preguntar(pregunta: str):
    """
    /preguntar endpoint - Flujo completo
    """
    
    # 1. Cargar vectorstore
    vectorstore = FAISS.load_local("vectorstore_faiss", embeddings)
    
    # 2. Crear retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # 3. Crear cadena RAG
    qa_chain = SimpleRetrievalQA(
        llm=llm,           # Modelo de lenguaje
        retriever=retriever # Motor de búsqueda
    )
    
    # 4. Ejecutar consulta
    resultado = qa_chain({"query": pregunta})
    
    # 5. Retornar respuesta + fuentes
    return {
        "respuesta": resultado["result"],
        "fuentes": [doc.page_content[:200] for doc in resultado["source_documents"]]
    }
```

---

## 🔧 Configuración Avanzada (Opcional)

### Usar con OpenAI (recomendado para producción)

```powershell
# Windows PowerShell
set LLM_TYPE=openai
set OPENAI_API_KEY=tu-api-key-aqui
Sistema_RAG.exe
```

**Obtener API Key:** https://platform.openai.com/api-keys

### Comparación de Costos

| Pregunta típica | Tokens | Costo OpenAI (GPT-3.5) |
|----------------|--------|------------------------|
| Simple         | ~500   | $0.0005 (~$0.5/1000)  |
| Con contexto   | ~2000  | $0.002 (~$2/1000)     |
| Compleja       | ~4000  | $0.004 (~$4/1000)     |

---

## 📊 Documentos Incluidos

- **IPC-7525A - Stencil Design Guidelines** (ejemplo)
- Puedes agregar tus propios documentos copiándolos a la carpeta `uploaded_docs/`

---

## 🎨 Interfaz Web

La interfaz incluida (`interfaz_web.html`) ofrece:

- ✅ Diseño moderno y responsive
- ✅ Burbujas de chat estilo WhatsApp
- ✅ Muestra documentos fuente usados
- ✅ Indicador de estado del servidor
- ✅ Sin dependencias externas

```html
<!-- interfaz_web.html - Fragmento -->

<script>
async function enviarPregunta() {
    const pregunta = document.getElementById('pregunta').value;
    
    // Llamar a la API
    const response = await fetch('http://localhost:8000/preguntar', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({pregunta, k: 4})
    });
    
    const data = await response.json();
    
    // Mostrar respuesta en chat
    mostrarMensaje(data.respuesta, 'asistente');
    mostrarFuentes(data.source_documents);
}
</script>
```

---

## 🔐 Seguridad y Privacidad

- ✅ **Procesamiento local:** Tus documentos NO se suben a ningún servidor
- ✅ **API Keys seguras:** Se usan variables de entorno (nunca en código)
- ✅ **Open Source:** Puedes auditar el código completo

**Nota:** Al usar OpenAI/Anthropic, las preguntas SÍ se envían a sus APIs (no los documentos completos, solo fragmentos relevantes).

---

## 📞 Soporte y Ayuda

### ¿Problemas con el ejecutable?

1. **Error "Puerto 8000 en uso":**
   - Otro programa está usando el puerto
   - Cierra otras aplicaciones o reinicia

2. **No se abre el navegador:**
   - Abre manualmente: http://localhost:8000

3. **Error 429 (OpenAI):**
   - Sin créditos en tu cuenta
   - Agrega créditos o usa modo fake

### ¿Quieres el código fuente?

Este proyecto está disponible como ejecutable standalone. Si eres desarrollador y quieres el código fuente para aprender o modificar, contáctame.

---

## 🙏 Créditos

Construido con:
- [LangChain](https://langchain.com/) - Framework RAG
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search (Meta AI)
- [FastAPI](https://fastapi.tiangolo.com/) - API moderna
- [OpenAI](https://openai.com/) / [Anthropic](https://anthropic.com/) - LLMs

---

## 📄 Licencia

Este software se proporciona "tal cual" para uso personal y educativo.

---

**⭐ ¿Te resultó útil? ¡Dale una estrella al repositorio!**

**💬 ¿Preguntas?** Abre un [Issue](../../issues) y te ayudo.
