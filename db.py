import os
from functools import lru_cache

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()


@lru_cache
def get_supabase_client() -> Client:
    """Return a configured Supabase client for opportunity lookups."""
    supabase_url = os.getenv("SUPABASE_URL") or os.getenv(
        "NEXT_PUBLIC_SUPABASE_URL"
    )
    supabase_key = (
        os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
        or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    )

    if not supabase_url or not supabase_key:
        raise RuntimeError(
            "Set SUPABASE_URL and SUPABASE_KEY in .env before querying opportunities."
        )

    return create_client(supabase_url, supabase_key)
