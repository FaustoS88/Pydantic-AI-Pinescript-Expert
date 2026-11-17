"""Shared test setup: dotenv is switched off, so no test reads a developer's .env."""

import dotenv

dotenv.load_dotenv = lambda *args, **kwargs: False
