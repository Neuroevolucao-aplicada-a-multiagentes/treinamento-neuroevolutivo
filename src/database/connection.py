"""Supabase connection management for the project."""

from __future__ import annotations

import os
from threading import Lock

from dotenv import load_dotenv
from supabase import Client, create_client


class SupabaseConfigurationError(RuntimeError):
	"""Raised when Supabase environment variables are missing or invalid."""


_supabase_client: Client | None = None
_client_lock = Lock()


def _load_environment() -> None:
	"""Load environment variables from .env into process environment."""

	load_dotenv()


def _get_supabase_credentials() -> tuple[str, str]:
	"""Read and validate Supabase credentials from environment variables."""

	supabase_url = (os.getenv("SUPABASE_URL") or "").strip()
	supabase_key = (os.getenv("SUPABASE_KEY") or "").strip()

	if not supabase_url or not supabase_key:
		raise SupabaseConfigurationError(
			"Supabase configuration is missing. Set SUPABASE_URL and "
			"SUPABASE_KEY in your environment or .env file."
		)

	return supabase_url, supabase_key


def get_supabase_client() -> Client:
	"""Return a shared Supabase client instance."""

	global _supabase_client

	if _supabase_client is not None:
		return _supabase_client

	with _client_lock:
		if _supabase_client is None:
			_load_environment()
			supabase_url, supabase_key = _get_supabase_credentials()
			try:
				_supabase_client = create_client(supabase_url, supabase_key)
			except Exception as exc:
				raise SupabaseConfigurationError(
					"Failed to create Supabase client. Verify SUPABASE_URL and "
					"SUPABASE_KEY values and your supabase-py configuration."
				) from exc

	assert _supabase_client is not None
	return _supabase_client


def reset_supabase_client() -> None:
	"""Reset the cached client instance."""

	global _supabase_client

	with _client_lock:
		_supabase_client = None
