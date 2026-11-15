"""Persistence layer for the Individual entity using Supabase Data API."""

from __future__ import annotations

from typing import Any, Mapping, cast
from uuid import UUID, uuid4

from supabase import Client

from database.connection import get_supabase_client


Numeric = int | float


class IndividualRepositoryError(RuntimeError):
    """Base error for persistence failures in IndividualRepository."""


class IndividualRepository:
    """Repository responsible for persisting Individual records in Supabase."""

    _SCHEMA = "core"
    _TABLE = "individual"
    _UNSET = object()

    def __init__(self, client: Client | None = None) -> None:
        """Initialize the repository with a Supabase client."""

        self._client = client if client is not None else get_supabase_client()

    def create(
        self,
        *,
        generation_id: UUID | str,
        individual_index: int,
        parent_1_id: UUID | str | None = None,
        parent_2_id: UUID | str | None = None,
        fitness: Numeric | None = None,
        rank: int | None = None,
        is_elite: bool = False,
        genome: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an Individual record and return the persisted row."""

        normalized_generation_id = self._normalize_uuid_like(generation_id, "generation_id")
        normalized_parent_1_id = self._normalize_nullable_uuid_like(parent_1_id, "parent_1_id")
        normalized_parent_2_id = self._normalize_nullable_uuid_like(parent_2_id, "parent_2_id")
        self._validate_required_int(individual_index, "individual_index", minimum=1)
        self._validate_optional_number(fitness, "fitness")
        self._validate_optional_int(rank, "rank", minimum=1)
        self._validate_bool(is_elite, "is_elite")
        self._validate_optional_mapping(genome, "genome")
        self._validate_optional_mapping(metadata, "metadata")

        payload: dict[str, Any] = {
            "id": str(uuid4()),
            "generation_id": normalized_generation_id,
            "individual_index": individual_index,
            "parent_1_id": normalized_parent_1_id,
            "parent_2_id": normalized_parent_2_id,
            "fitness": fitness,
            "rank": rank,
            "is_elite": is_elite,
            "genome": dict(genome) if genome is not None else {},
            "metadata": dict(metadata) if metadata is not None else {},
        }

        created_id = payload["id"]

        try:
            response = (
                self._client.schema(self._SCHEMA)
                .table(self._TABLE)
                .insert(payload)
                .execute()
            )
            created = self._first_or_none(response.data)
            if created is None:
                created = self.get_by_id(created_id)
        except IndividualRepositoryError:
            raise
        except Exception as exc:
            raise IndividualRepositoryError("Failed to create individual in Supabase.") from exc

        if created is None:
            raise IndividualRepositoryError(
                "Individual was inserted but could not be retrieved afterward."
            )

        return created

    def get_by_id(self, individual_id: UUID | str) -> dict[str, Any] | None:
        """Fetch one Individual by its UUID identifier."""

        identifier = self._normalize_uuid_like(individual_id, "individual_id")

        try:
            response = (
                self._client.schema(self._SCHEMA)
                .table(self._TABLE)
                .select("*")
                .eq("id", identifier)
                .limit(1)
                .execute()
            )
        except Exception as exc:
            raise IndividualRepositoryError("Failed to fetch individual by id.") from exc

        return self._first_or_none(response.data)

    def list_by_generation(self, generation_id: UUID | str) -> list[dict[str, Any]]:
        """List Individuals associated with a generation."""

        identifier = self._normalize_uuid_like(generation_id, "generation_id")

        try:
            response = (
                self._client.schema(self._SCHEMA)
                .table(self._TABLE)
                .select("*")
                .eq("generation_id", identifier)
                .order("individual_index", desc=False)
                .execute()
            )
        except Exception as exc:
            raise IndividualRepositoryError("Failed to list individuals by generation.") from exc

        rows = response.data or []
        return [row for row in rows if isinstance(row, dict)]

    def list_all(self) -> list[dict[str, Any]]:
        """List all Individual rows ordered by creation timestamp."""

        try:
            response = (
                self._client.schema(self._SCHEMA)
                .table(self._TABLE)
                .select("*")
                .order("created_at", desc=False)
                .execute()
            )
        except Exception as exc:
            raise IndividualRepositoryError("Failed to list individuals.") from exc

        rows = response.data or []
        return [row for row in rows if isinstance(row, dict)]

    def update(
        self,
        individual_id: UUID | str,
        *,
        generation_id: UUID | str | object = _UNSET,
        individual_index: int | object = _UNSET,
        parent_1_id: UUID | str | None | object = _UNSET,
        parent_2_id: UUID | str | None | object = _UNSET,
        fitness: Numeric | None | object = _UNSET,
        rank: int | None | object = _UNSET,
        is_elite: bool | object = _UNSET,
        genome: Mapping[str, Any] | None | object = _UNSET,
        metadata: Mapping[str, Any] | None | object = _UNSET,
    ) -> dict[str, Any] | None:
        """Update an Individual and return the updated row."""

        identifier = self._normalize_uuid_like(individual_id, "individual_id")
        payload: dict[str, Any] = {}

        if generation_id is not self._UNSET:
            payload["generation_id"] = self._normalize_uuid_like(generation_id, "generation_id")
        if individual_index is not self._UNSET:
            self._validate_required_int(individual_index, "individual_index", minimum=1)
            payload["individual_index"] = individual_index
        if parent_1_id is not self._UNSET:
            payload["parent_1_id"] = self._normalize_nullable_uuid_like(parent_1_id, "parent_1_id")
        if parent_2_id is not self._UNSET:
            payload["parent_2_id"] = self._normalize_nullable_uuid_like(parent_2_id, "parent_2_id")
        if fitness is not self._UNSET:
            self._validate_optional_number(fitness, "fitness")
            payload["fitness"] = fitness
        if rank is not self._UNSET:
            self._validate_optional_int(rank, "rank", minimum=1)
            payload["rank"] = rank
        if is_elite is not self._UNSET:
            self._validate_bool(is_elite, "is_elite")
            payload["is_elite"] = is_elite
        if genome is not self._UNSET:
            self._validate_optional_mapping(genome, "genome")
            payload["genome"] = {} if genome is None else dict(cast(Mapping[str, Any], genome))
        if metadata is not self._UNSET:
            self._validate_optional_mapping(metadata, "metadata")
            payload["metadata"] = {} if metadata is None else dict(cast(Mapping[str, Any], metadata))

        if not payload:
            raise ValueError("At least one field must be provided for update.")

        try:
            response = (
                self._client.schema(self._SCHEMA)
                .table(self._TABLE)
                .update(payload)
                .eq("id", identifier)
                .execute()
            )
            updated = self._first_or_none(response.data)
            if updated is not None:
                return updated
        except Exception as exc:
            raise IndividualRepositoryError("Failed to update individual.") from exc

        return self.get_by_id(identifier)

    def delete(self, individual_id: UUID | str) -> bool:
        """Delete an Individual by UUID."""

        identifier = self._normalize_uuid_like(individual_id, "individual_id")

        try:
            response = (
                self._client.schema(self._SCHEMA)
                .table(self._TABLE)
                .delete()
                .eq("id", identifier)
                .execute()
            )

            if isinstance(response.data, list):
                return len(response.data) > 0

            return self.get_by_id(identifier) is None
        except Exception as exc:
            raise IndividualRepositoryError("Failed to delete individual.") from exc

    @staticmethod
    def _first_or_none(rows: Any) -> dict[str, Any] | None:
        """Return the first row from a list-like API response or None."""

        if isinstance(rows, list) and rows:
            first_row = rows[0]
            if isinstance(first_row, dict):
                return first_row
        return None

    @staticmethod
    def _validate_required_int(value: object, field_name: str, *, minimum: int | None = None) -> None:
        """Validate a required integer field."""

        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{field_name} must be an int.")
        if minimum is not None and value < minimum:
            raise ValueError(f"{field_name} must be greater than or equal to {minimum}.")

    @staticmethod
    def _validate_optional_int(value: object, field_name: str, *, minimum: int | None = None) -> None:
        """Validate an optional integer field."""

        if value is None:
            return
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{field_name} must be an int or None when provided.")
        if minimum is not None and value < minimum:
            raise ValueError(f"{field_name} must be greater than or equal to {minimum}.")

    @staticmethod
    def _validate_optional_number(value: object, field_name: str) -> None:
        """Validate an optional numeric field."""

        if value is None:
            return
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{field_name} must be an int, float, or None when provided.")

    @staticmethod
    def _validate_bool(value: object, field_name: str) -> None:
        """Validate a boolean field."""

        if isinstance(value, bool):
            return
        raise TypeError(f"{field_name} must be a bool.")

    @staticmethod
    def _validate_optional_mapping(value: object, field_name: str) -> None:
        """Validate an optional JSON mapping field."""

        if value is None:
            return
        if not isinstance(value, Mapping):
            raise TypeError(f"{field_name} must be a mapping or None when provided.")

    @staticmethod
    def _normalize_nullable_uuid_like(value: object, field_name: str) -> str | None:
        """Normalize an optional UUID-like field to canonical string form."""

        if value is None:
            return None
        return IndividualRepository._normalize_uuid_like(value, field_name)

    @staticmethod
    def _normalize_uuid_like(value: object, field_name: str) -> str:
        """Normalize a UUID-like field to canonical string form after validation."""

        if isinstance(value, UUID):
            return str(value)
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a UUID or str when provided.")
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError(f"{field_name} must not be empty or whitespace.")

        try:
            return str(UUID(normalized_value))
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a valid UUID.") from exc
