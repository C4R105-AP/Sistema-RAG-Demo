"""
Cliente Python para interactuar con la API RAG
"""

import requests
import json
from typing import Optional

class RAGClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def verificar_estado(self):
        """Verifica el estado del sistema"""
        response = requests.get(f"{self.base_url}/estado")
        return response.json()
    
    def hacer_pregunta(self, pregunta: str, k: int = 4):
        """Hace una pregunta al sistema RAG"""
        data = {
            "pregunta": pregunta,
            "k": k
        }
        response = requests.post(f"{self.base_url}/preguntar", json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json()}
    
    def subir_documento(self, ruta_archivo: str):
        """Sube un documento al sistema"""
        with open(ruta_archivo, 'rb') as f:
            files = {'archivo': f}
            response = requests.post(f"{self.base_url}/subir-documento", files=files)
        
        return response.json()
    
    def recargar_sistema(self):
        """Recarga el vectorstore"""
        response = requests.post(f"{self.base_url}/recargar")
        return response.json()
    
    def limpiar_sistema(self):
        """Limpia todos los documentos"""
        response = requests.delete(f"{self.base_url}/limpiar")
        return response.json()


# =============================================================================
# EJEMPLOS DE USO
# =============================================================================

def ejemplo_basico():
    """Ejemplo básico de uso"""
    client = RAGClient()
    
    # 1. Verificar estado
    print("📊 Estado del sistema:")
    estado = client.verificar_estado()
    print(json.dumps(estado, indent=2, ensure_ascii=False))
    
    # 2. Hacer pregunta
    print("\n❓ Haciendo pregunta...")
    resultado = client.hacer_pregunta("¿De qué trata el documento?")
    
    if "error" not in resultado:
        print(f"\n✅ Respuesta: {resultado['respuesta']}")
        print(f"\n📚 Documentos fuente ({len(resultado['documentos_fuente'])}):")
        for i, doc in enumerate(resultado['documentos_fuente'], 1):
            print(f"\n{i}. {doc['contenido'][:150]}...")
    else:
        print(f"❌ Error: {resultado['error']}")


def ejemplo_subir_documento():
    """Ejemplo de subida de documento"""
    client = RAGClient()
    
    # Subir documento
    print("📤 Subiendo documento...")
    resultado = client.subir_documento("mi_documento.txt")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


def ejemplo_interactivo():
    """Modo interactivo para hacer preguntas"""
    client = RAGClient()
    
    print("🤖 Cliente RAG Interactivo")
    print("=" * 50)
    
    # Verificar estado
    estado = client.verificar_estado()
    if not estado['vectorstore_cargado']:
        print("⚠️  El sistema no tiene documentos cargados.")
        print("   Sube un documento primero.")
        return
    
    print(f"✅ Sistema listo ({estado['total_documentos']} documentos)")
    print("\nEscribe 'salir' para terminar\n")
    
    while True:
        pregunta = input("Tu pregunta: ").strip()
        
        if pregunta.lower() in ['salir', 'exit', 'quit']:
            print("👋 ¡Hasta luego!")
            break
        
        if not pregunta:
            continue
        
        print("\n⏳ Procesando...")
        resultado = client.hacer_pregunta(pregunta)
        
        if "error" not in resultado:
            print(f"\n💬 Respuesta:\n{resultado['respuesta']}")
            print(f"\n📄 Basado en {len(resultado['documentos_fuente'])} documentos")
        else:
            print(f"\n❌ Error: {resultado['error']}")
        
        print("\n" + "-" * 50 + "\n")


# =============================================================================
# EJEMPLO CON CURL
# =============================================================================

def mostrar_ejemplos_curl():
    """Muestra ejemplos de uso con curl"""
    ejemplos = """
    🌐 EJEMPLOS DE USO CON CURL
    ═══════════════════════════════════════════════════════════════
    
    1️⃣  Verificar estado:
    curl http://localhost:8000/estado
    
    2️⃣  Hacer una pregunta:
    curl -X POST http://localhost:8000/preguntar \\
      -H "Content-Type: application/json" \\
      -d '{"pregunta": "¿Cuál es el tema principal?", "k": 4}'
    
    3️⃣  Subir un documento:
    curl -X POST http://localhost:8000/subir-documento \\
      -F "archivo=@mi_documento.pdf"
    
    4️⃣  Recargar sistema:
    curl -X POST http://localhost:8000/recargar
    
    5️⃣  Limpiar todo:
    curl -X DELETE http://localhost:8000/limpiar
    
    ═══════════════════════════════════════════════════════════════
    📝 Documentación interactiva: http://localhost:8000/docs
    """
    print(ejemplos)


# =============================================================================
# EJECUTAR
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        comando = sys.argv[1]
        
        if comando == "estado":
            client = RAGClient()
            print(json.dumps(client.verificar_estado(), indent=2, ensure_ascii=False))
        
        elif comando == "pregunta" and len(sys.argv) > 2:
            client = RAGClient()
            pregunta = " ".join(sys.argv[2:])
            resultado = client.hacer_pregunta(pregunta)
            print(json.dumps(resultado, indent=2, ensure_ascii=False))
        
        elif comando == "subir" and len(sys.argv) > 2:
            client = RAGClient()
            archivo = sys.argv[2]
            resultado = client.subir_documento(archivo)
            print(json.dumps(resultado, indent=2, ensure_ascii=False))
        
        elif comando == "interactivo":
            ejemplo_interactivo()
        
        elif comando == "curl":
            mostrar_ejemplos_curl()
        
        else:
            print("Uso: python cliente_rag.py [comando]")
            print("Comandos: estado | pregunta | subir | interactivo | curl")
    
    else:
        # Modo por defecto: interactivo
        ejemplo_interactivo()