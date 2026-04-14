"""Custom exceptions for the LightOn SDK."""

from __future__ import annotations


class LightOnError(Exception):
    """Base exception for all LightOn API errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: dict | None = None,
    ):
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message)


class AuthenticationError(LightOnError):
    """Raised when the API key is invalid or missing."""


class NotFoundError(LightOnError):
    """Raised when the requested resource does not exist."""


class RateLimitError(LightOnError):
    """Raised when the API rate limit has been exceeded."""


class ValidationError(LightOnError):
    """Raised when the request parameters are invalid."""
