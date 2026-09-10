"""Errores de dominio (ingesta y retrieval) sin acoplar FastAPI."""


class IngestError(Exception):
    def __init__(self, detail: str, status_code: int = 400):
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)


class RetrievalPipelineError(Exception):
    """Fallo crítico en retrieval (p. ej. índice vacío)."""
