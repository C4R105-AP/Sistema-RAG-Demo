"""
Punto de entrada del Sistema RAG universal.
Fuerza el venv del proyecto y arranca uvicorn en el hilo principal.
"""
import os
import sys
import socket
import threading
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
HOST = os.getenv("RAG_HOST", "127.0.0.1")
PORT = int(os.getenv("RAG_PORT", "8000"))


def ensure_project_venv() -> None:
    if not VENV_PYTHON.exists():
        print(f"[AVISO] No se encontró venv en: {VENV_PYTHON}")
        return
    if Path(sys.executable).resolve() != VENV_PYTHON.resolve():
        print(f"[INFO] Cambiando a venv: {VENV_PYTHON}")
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])


def puerto_ocupado(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def abrir_navegador_cuando_listo() -> None:
    time.sleep(2)
    webbrowser.open(f"http://localhost:{PORT}/app")


def main() -> None:
    ensure_project_venv()
    os.chdir(PROJECT_ROOT)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    if puerto_ocupado(PORT):
        print(f"[ERROR] El puerto {PORT} ya está en uso.")
        print("Cierra la instancia anterior o define RAG_PORT en .env")
        sys.exit(1)

    llm_type = os.getenv("LLM_TYPE", "fake")
    print("=" * 70)
    print("  SISTEMA RAG UNIVERSAL")
    print("=" * 70)
    print(f"  Python:   {sys.executable}")
    print(f"  LLM:      {llm_type}")
    if llm_type.lower() == "fake":
        print("  (modo demo: búsqueda sí, respuestas reales no)")
        print("  Para generar: instala Ollama + `ollama pull llama3.2` y LLM_TYPE=ollama")
    print(f"  Bind:     {HOST}:{PORT}")
    print(f"  Interfaz: http://localhost:{PORT}/app")
    print(f"  API docs: http://localhost:{PORT}/docs")
    print("=" * 70)
    print("  Ctrl+C para detener")
    print("=" * 70)

    threading.Thread(target=abrir_navegador_cuando_listo, daemon=True).start()

    import uvicorn
    from api_rag import app

    try:
        uvicorn.run(app, host=HOST, port=PORT, log_level="info")
    except KeyboardInterrupt:
        print("\nServidor detenido.")


if __name__ == "__main__":
    main()
