"""
API REST para Sistema RAG con FastAPI
Permite a usuarios hacer preguntas a través de endpoints HTTP
Soporta múltiples LLMs: Hugging Face (offline), OpenAI, Anthropic
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import os
import shutil
from datetime import datetime

# LangChain
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import Dict, Any
import numpy as np

# LLM reales (imports opcionales)
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from langchain_community.llms import HuggingFacePipeline
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Implementaciones fake offline
class FakeChatModel(BaseChatModel):
    """ChatModel fake para pruebas offline"""
    def _generate(self, messages, **kwargs):
        # Simula una respuesta básica
        text = "Esta es una respuesta de prueba generada offline. Los datos se basaron en el contexto proporcionado."
        message = AIMessage(content=text)
        return self.create_llm_result([message])
    
    @property
    def _llm_type(self):
        return "fake"

class FakeEmbeddings(Embeddings):
    """Embeddings fake para pruebas offline"""
    def embed_documents(self, texts):
        # Genera embeddings aleatorios de dimensión 384
        return [np.random.rand(384).tolist() for _ in texts]
    
    def embed_query(self, text):
        # Genera un embedding aleatorio de dimensión 384
        return np.random.rand(384).tolist()

# Clase simple para simular RetrievalQA
class SimpleRetrievalQA:
    """Implementación simple de RetrievalQA para pruebas offline"""
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    def __call__(self, query_dict: dict) -> Dict[str, Any]:
        query = query_dict.get('query', '')
        
        try:
            # Intentar obtener el vectorstore del retriever
            if hasattr(self.retriever, 'vectorstore'):
                vectorstore = self.retriever.vectorstore
                docs = vectorstore.similarity_search(query, k=4)
            elif hasattr(self.retriever, 'get_relevant_documents'):
                docs = self.retriever.get_relevant_documents(query)
            else:
                # Intentar con el método invoke si existe
                docs = self.retriever.invoke(query)
        except Exception as e:
            print(f"Error recuperando documentos: {e}")
            docs = []
        
        # Combinar contexto
        if docs:
            context = "\n\n".join([doc.page_content for doc in docs])
        else:
            context = "No se encontró información relevante en los documentos."
        
        # Generar respuesta usando el LLM real
        try:
            # Crear prompt con contexto RAG
            prompt = f"""Contexto:
{context}

Pregunta: {query}

Respuesta (basándote en el contexto):"""
            
            # Intentar usar el LLM real
            if hasattr(self.llm, 'invoke'):
                # LangChain ChatModel (OpenAI, Anthropic, Hugging Face)
                messages = [
                    SystemMessage(content="Eres un asistente experto que responde preguntas basándote en el contexto proporcionado. Responde de forma clara y concisa."),
                    HumanMessage(content=prompt)
                ]
                result = self.llm.invoke(messages)
                answer = result.content if hasattr(result, 'content') else str(result)
            elif hasattr(self.llm, 'generate'):
                # Hugging Face Pipeline
                result = self.llm.generate([prompt])
                answer = result.generations[0][0].text
            else:
                # Fallback: respuesta simple
                answer = f"Basándome en el contexto proporcionado sobre: {query}\n\n{context[:200]}..."
        
        except Exception as e:
            print(f"Error generando respuesta: {e}")
            answer = f"Basándome en el contexto: {context[:200]}... (Error al generar respuesta: {str(e)})"
        
        return {
            'result': answer,
            'source_documents': docs
        }

# =============================================================================
# CONFIGURACIÓN Y FUNCIONES AUXILIARES
# =============================================================================

# Variables globales
vectorstore = None
qa_chain = None
VECTORSTORE_PATH = "vectorstore_faiss"
UPLOAD_DIR = "uploaded_docs"
LLM_TYPE = os.getenv("LLM_TYPE", "fake")  # fake, huggingface, openai, anthropic
LLM_MODEL = os.getenv("LLM_MODEL", "microsoft/DialoGPT-medium")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Funciones auxiliares
def inicializar_embeddings():
    """Inicializa los embeddings según el tipo configurado"""
    if LLM_TYPE in ["huggingface", "fake"]:
        try:
            # Usar embeddings reales de Hugging Face
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
        except Exception as e:
            print(f"[ADVERTENCIA] No se pudo cargar Hugging Face embeddings: {e}")
            print("[INFO] Usando embeddings fake")
            return FakeEmbeddings()
    return FakeEmbeddings()

def inicializar_llm():
    """Inicializa el LLM según el tipo configurado"""
    if LLM_TYPE == "openai":
        if not OPENAI_AVAILABLE:
            print("[ERROR] langchain_openai no está instalado. Usando LLM fake")
            return FakeChatModel()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY no configurada")
        return ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            api_key=api_key
        )
    
    elif LLM_TYPE == "anthropic":
        if not ANTHROPIC_AVAILABLE:
            print("[ERROR] langchain_anthropic no está instalado. Usando LLM fake")
            return FakeChatModel()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY no configurada")
        return ChatAnthropic(
            model="claude-3-sonnet-20240229",
            temperature=0.7,
            api_key=api_key
        )
    
    elif LLM_TYPE == "huggingface":
        if not HUGGINGFACE_AVAILABLE:
            print("[ERROR] transformers no está instalado. Usando LLM fake")
            return FakeChatModel()
        try:
            print(f"[INFO] Cargando modelo Hugging Face: {LLM_MODEL}")
            tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                device_map="auto",
                torch_dtype="auto"
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=500,
                temperature=0.7,
                top_p=0.95,
                return_full_text=False
            )
            
            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            print(f"[ERROR] No se pudo cargar modelo Hugging Face: {e}")
            print("[INFO] Usando LLM fake")
            return FakeChatModel()
    
    else:  # fake
        return FakeChatModel()

def crear_qa_chain(vectorstore_obj):
    llm = inicializar_llm()
    retriever = vectorstore_obj.as_retriever(search_kwargs={"k": 4})
    chain = SimpleRetrievalQA(llm=llm, retriever=retriever)
    return chain

def cargar_vectorstore_global():
    global vectorstore, qa_chain
    try:
        if os.path.exists(VECTORSTORE_PATH):
            embeddings = inicializar_embeddings()
            vectorstore = FAISS.load_local(
                VECTORSTORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            qa_chain = crear_qa_chain(vectorstore)
            return True
        return False
    except Exception as e:
        print(f"Error cargando vectorstore: {e}")
        return False

# Lifespan event (debe estar antes de crear FastAPI)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[INICIO] Iniciando API RAG Offline...")
    if cargar_vectorstore_global():
        print("[OK] Vectorstore cargado correctamente")
    else:
        print("[ADVERTENCIA] No se encontro vectorstore. Sube documentos primero.")
    yield
    # Shutdown
    print("[SHUTDOWN] Cerrando API RAG Offline...")

# Crear la app FastAPI
app = FastAPI(
    title="RAG API Offline",
    description="API para hacer pruebas RAG con documentos locales (offline)",
    version="1.0.0",
    lifespan=lifespan
)

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# MODELOS PYDANTIC
# =============================================================================

class PreguntaRequest(BaseModel):
    pregunta: str
    k: Optional[int] = 4

class RespuestaResponse(BaseModel):
    respuesta: str
    documentos_fuente: List[dict]
    timestamp: str

class EstadoResponse(BaseModel):
    estado: str
    vectorstore_cargado: bool
    total_documentos: Optional[int]
    mensaje: str

class DocumentoResponse(BaseModel):
    mensaje: str
    archivo: str
    chunks_creados: int
    timestamp: str

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", tags=["General"])
async def root():
    return {
        "mensaje": "API RAG Offline activa",
        "version": "1.0.0",
        "endpoints": {
            "estado": "/estado",
            "preguntar": "/preguntar (POST)",
            "subir_documento": "/subir-documento (POST)",
            "recargar": "/recargar (POST)"
        }
    }

@app.get("/estado", response_model=EstadoResponse, tags=["General"])
async def obtener_estado():
    vectorstore_existe = vectorstore is not None
    total_docs = getattr(vectorstore.index, "ntotal", None) if vectorstore_existe else None

    return EstadoResponse(
        estado="activo" if vectorstore_existe else "sin_documentos",
        vectorstore_cargado=vectorstore_existe,
        total_documentos=total_docs,
        mensaje="Sistema listo para recibir preguntas" if vectorstore_existe
                else "Sube documentos para comenzar"
    )

@app.post("/preguntar", response_model=RespuestaResponse, tags=["RAG"])
async def hacer_pregunta(request: PreguntaRequest):
    if qa_chain is None:
        raise HTTPException(
            status_code=503,
            detail="Sistema no inicializado. Sube documentos primero o recarga el vectorstore."
        )
    try:
        resultado = qa_chain({"query": request.pregunta})
        docs_fuente = [
            {
                "contenido": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "metadata": doc.metadata
            }
            for doc in resultado['source_documents'][:request.k]
        ]
        return RespuestaResponse(
            respuesta=resultado['result'],
            documentos_fuente=docs_fuente,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar pregunta: {str(e)}"
        )

@app.post("/subir-documento", response_model=DocumentoResponse, tags=["Documentos"])
async def subir_documento(archivo: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    global vectorstore, qa_chain

    if not archivo.filename.endswith(('.txt', '.pdf')):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .txt o .pdf")

    try:
        file_path = os.path.join(UPLOAD_DIR, archivo.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(archivo.file, buffer)

        loader = PyPDFLoader(file_path) if archivo.filename.endswith('.pdf') else TextLoader(file_path, encoding='utf-8')
        documentos = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_documents(documentos)

        embeddings = inicializar_embeddings()

        if vectorstore is None:
            vectorstore = FAISS.from_documents(chunks, embeddings)
        else:
            new_vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.merge_from(new_vectorstore)

        vectorstore.save_local(VECTORSTORE_PATH)
        qa_chain = crear_qa_chain(vectorstore)

        return DocumentoResponse(
            mensaje="Documento procesado correctamente",
            archivo=archivo.filename,
            chunks_creados=len(chunks),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar documento: {str(e)}")

@app.post("/recargar", tags=["Admin"])
async def recargar_vectorstore():
    if cargar_vectorstore_global():
        return {"mensaje": "Vectorstore recargado correctamente", "timestamp": datetime.now().isoformat()}
    else:
        raise HTTPException(status_code=404, detail="No se encontró vectorstore para recargar")

@app.delete("/limpiar", tags=["Admin"])
async def limpiar_documentos():
    global vectorstore, qa_chain
    try:
        if os.path.exists(VECTORSTORE_PATH):
            shutil.rmtree(VECTORSTORE_PATH)
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)

        vectorstore = None
        qa_chain = None

        return {"mensaje": "Todos los documentos han sido eliminados", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al limpiar: {str(e)}")

# =============================================================================
# EJECUTAR
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("="*70)
    print("[INICIO] Iniciando servidor RAG Offline")
    print("="*70)
    print(f"[INFO] Documentacion: http://localhost:8000/docs")
    print("="*70)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)