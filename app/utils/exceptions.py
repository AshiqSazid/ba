from fastapi import HTTPException, status
from typing import Any, Dict, Optional


class TheramuseException(Exception):
    """
    Base exception for Theramuse application.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(TheramuseException):
    """
    Exception raised for validation errors.
    """

    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        super().__init__(message)
        self.field = field
        self.value = value


class DatabaseError(TheramuseException):
    """
    Exception raised for database-related errors.
    """

    def __init__(self, message: str, operation: Optional[str] = None):
        super().__init__(message)
        self.operation = operation


class MLModelError(TheramuseException):
    """
    Exception raised for machine learning model errors.
    """

    def __init__(self, message: str, model_name: Optional[str] = None):
        super().__init__(message)
        self.model_name = model_name


class FileNotFoundError(TheramuseException):
    """
    Exception raised when a required file is not found.
    """

    def __init__(self, message: str, file_path: Optional[str] = None):
        super().__init__(message)
        self.file_path = file_path


class AuthenticationError(TheramuseException):
    """
    Exception raised for authentication errors.
    """

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message)


class AuthorizationError(TheramuseException):
    """
    Exception raised for authorization errors.
    """

    def __init__(self, message: str = "Access denied"):
        super().__init__(message)


class RateLimitError(TheramuseException):
    """
    Exception raised when rate limit is exceeded.
    """

    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


# HTTP Exception helpers
def create_http_exception(
    status_code: int,
    detail: str,
    headers: Optional[Dict[str, str]] = None
) -> HTTPException:
    """
    Create an HTTPException with consistent formatting.
    """
    return HTTPException(
        status_code=status_code,
        detail={
            "error": detail,
            "status_code": status_code,
            "timestamp": "2024-01-01T00:00:00Z"  # This should be dynamic
        },
        headers=headers
    )


def handle_validation_error(exc: ValidationError) -> HTTPException:
    """
    Handle ValidationError and return appropriate HTTPException.
    """
    detail = exc.message
    if exc.field:
        detail = f"Validation error for field '{exc.field}': {exc.message}"
    if exc.value:
        detail += f" (value: {exc.value})"

    return create_http_exception(status.HTTP_422_UNPROCESSABLE_ENTITY, detail)


def handle_database_error(exc: DatabaseError) -> HTTPException:
    """
    Handle DatabaseError and return appropriate HTTPException.
    """
    detail = "A database error occurred"
    if exc.operation:
        detail += f" during {exc.operation}"
    detail += f": {exc.message}"

    return create_http_exception(status.HTTP_500_INTERNAL_SERVER_ERROR, detail)


def handle_ml_model_error(exc: MLModelError) -> HTTPException:
    """
    Handle MLModelError and return appropriate HTTPException.
    """
    detail = "Machine learning model error"
    if exc.model_name:
        detail += f" with model '{exc.model_name}'"
    detail += f": {exc.message}"

    return create_http_exception(status.HTTP_500_INTERNAL_SERVER_ERROR, detail)