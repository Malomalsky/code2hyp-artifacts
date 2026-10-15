"""Original D0 source normalization; extracted without changes."""
import io, tokenize
from dataclasses import dataclass

@dataclass(frozen=True)
class CanonicalSource:
    text: str
    encoding: str
    decode_ok: bool
    decode_error: str | None

def normalize_python_source(raw: bytes) -> CanonicalSource:
    """Decode a Python file and apply the preregistered D0 normalization."""

    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
        text = raw.decode(encoding, errors="strict")
    except (LookupError, SyntaxError, UnicodeDecodeError) as exc:
        return CanonicalSource(
            text="",
            encoding="unknown",
            decode_ok=False,
            decode_error=type(exc).__name__,
        )
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip(" \t") for line in text.split("\n")]
    normalized = "\n".join(lines).rstrip("\n")
    if normalized:
        normalized += "\n"
    return CanonicalSource(
        text=normalized,
        encoding=encoding,
        decode_ok=True,
        decode_error=None,
    )
