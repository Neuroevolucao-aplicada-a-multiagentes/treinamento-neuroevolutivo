"""Substituto do TrainingPersistenceService para treinar sem o Supabase.

Mesma interface publica, sem chamadas de rede; os arquivos locais da run
continuam sendo gravados pelo Logger.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4


def _registro(**campos: Any) -> dict[str, Any]:
    return {"id": str(uuid4()), **campos}


class PersistenciaLocal:
    def start_experiment(self, **campos: Any) -> dict[str, Any]:
        return _registro(**campos)

    def start_run(self, **campos: Any) -> dict[str, Any]:
        return _registro(**campos)

    def save_run_config_snapshot(self, **campos: Any) -> dict[str, Any]:
        return _registro()

    def start_generation(self, **campos: Any) -> dict[str, Any]:
        return _registro(**campos)

    def finish_generation(self, **campos: Any) -> dict[str, Any]:
        return _registro()

    def save_individual(self, **campos: Any) -> dict[str, Any]:
        return _registro()

    def save_checkpoint(self, **campos: Any) -> dict[str, Any]:
        return _registro()

    def update_checkpoint(self, checkpoint_id: UUID | str, **campos: Any) -> dict[str, Any]:
        return {"id": str(checkpoint_id)}

    def complete_run(self, **campos: Any) -> dict[str, Any]:
        return _registro()

    def fail_run(self, **campos: Any) -> dict[str, Any]:
        return _registro()
