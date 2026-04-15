import os
from urllib.parse import urlparse

import openlit


def _as_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_placeholder_endpoint(value: str) -> bool:
    return "<" in value or "your-openlit-host" in value


def _is_http_endpoint(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def init_openlit(app_name: str) -> None:
    """Initialize OpenLit before ADK imports create model/tool clients."""
    if not _as_bool(os.getenv("OPENLIT_ENABLED"), default=True):
        return

    otel_endpoint = os.getenv("OTEL_ENDPOINT") or os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    if not otel_endpoint:
        return
    if _is_placeholder_endpoint(otel_endpoint):
        return
    if not _is_http_endpoint(otel_endpoint):
        return

    openlit.init(
        otlp_endpoint=otel_endpoint,
        application_name=os.getenv("OPENLIT_APP_NAME", app_name),
        environment=os.getenv("OPENLIT_ENVIRONMENT", "production"),
        disable_batch=False,
    )
