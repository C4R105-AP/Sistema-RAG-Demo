"""Punto de compatibilidad. El código vive en `rag.api`."""
from rag.api import app

if __name__ == "__main__":
    from launcher import main

    main()
