import structlog
import logging
import sys
from app.core.config import settings


def configure_logging() -> None:
    """
    Configure structured logging.
    """
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
            if settings.LOG_FORMAT == "json"
            else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Custom formatter to replace SQLAlchemy with PostgreSQL in log messages
    class PostgreSQLFormatter(logging.Formatter):
        def format(self, record):
            # Format the record first
            formatted = super().format(record)
            # Replace SQLAlchemy references with PostgreSQL (order matters!)
            formatted = formatted.replace('SQLAlchemy', 'PostgreSQL')
            formatted = formatted.replace('sqlalchemy', 'postgresql')
            return formatted

    # Configure standard logging with PostgreSQL formatter
    # Send framework logs to stderr so CLI utilities can emit clean JSON on stdout.
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(PostgreSQLFormatter("%(message)s"))

    logging.basicConfig(
        handlers=[handler],
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        force=True,
    )

    # Set up loggers for specific libraries
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)

    # Create custom PostgreSQL logger to show PostgreSQL instead of SQLAlchemy
    postgresql_logger = logging.getLogger("postgresql.engine")
    sqlalchemy_logger = logging.getLogger("sqlalchemy.engine")

    # Set levels - completely silence SQLAlchemy engine logs
    sqlalchemy_logger.setLevel(logging.ERROR)  # Only show errors, silence everything else
    postgresql_logger.setLevel(logging.ERROR)   # Only show errors, silence everything else

    # Add custom filter to replace SQLAlchemy with PostgreSQL in log messages
    class PostgreSQLLogFilter(logging.Filter):
        def filter(self, record):
            if hasattr(record, 'msg') and isinstance(record.msg, str):
                record.msg = record.msg.replace('SQLAlchemy', 'PostgreSQL')
                record.msg = record.msg.replace('sqlalchemy', 'postgresql')
            if hasattr(record, 'name'):
                record.name = record.name.replace('sqlalchemy', 'postgresql')
            return True

    # Apply the filter to SQLAlchemy logger
    sqlalchemy_logger.addFilter(PostgreSQLLogFilter())

    # Suppress all database-related logs completely
    logging.getLogger("sqlalchemy.dialects.postgresql").setLevel(logging.ERROR)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.ERROR)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.ERROR)
    logging.getLogger("psycopg2").setLevel(logging.ERROR)
    logging.getLogger("psycopg2.pool").setLevel(logging.ERROR)
    logging.getLogger("alembic").setLevel(logging.ERROR)

    # Suppress specific SQLAlchemy connection errors
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)


# Configure logging on import
configure_logging()
