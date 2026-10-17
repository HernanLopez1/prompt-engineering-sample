import argparse
import os
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from openai import OpenAI

__all__ = ["get_chat_completion", "sanitize_pii"]

BASE_DIR = Path(__file__).resolve().parent


def _read_text_flexible(file_path: Path) -> str:
    """Read a text file handling UTF-8, UTF-8-SIG, and PowerShell's default UTF-16."""
    raw_bytes = file_path.read_bytes()
    if raw_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw_bytes.decode("utf-16")
    return raw_bytes.decode("utf-8-sig", errors="replace").replace("\x00", "")


def _load_env_file(env_path: Path) -> None:
    """Load environment variables from a local .env file without exposing secrets."""
    if not env_path.is_file():
        return
    for raw_line in _read_text_flexible(env_path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ[key] = value


def _get_openai_client() -> OpenAI:
    """Authenticate safely without exposing the API key in logs or tracebacks."""
    _load_env_file(BASE_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "tu-api-key-aqui":
        raise RuntimeError(
            "No se encontró una OPENAI_API_KEY válida. "
            "Configúrala en el archivo .env (basado en .env.example) o en tus variables de entorno."
        )
    return OpenAI(api_key=api_key)


# Load settings file
settings_path = BASE_DIR / "settings.toml"
with settings_path.open("rb") as settings_file:
    SETTINGS = tomllib.load(settings_file)


def sanitize_pii(content: str) -> str:
    """Sanitize personally identifiable information (PII) locally before sending to external APIs."""
    sanitized = content

    # 1. Replace [support_...] and customer usernames in brackets
    sanitized = re.sub(r"\[support_[^\]]+\]", "[Agent]", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(
        r"\[(?!Agent\]|Client\]|Customer\])[^\]]+\]",
        "[Client]",
        sanitized,
    )

    # 2. Keep only the date (YYYY-MM-DD) from ISO timestamps
    sanitized = re.sub(
        r"(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}:\d{2}(?:\+\d{2}:\d{2}|Z)?",
        r"\1",
        sanitized,
    )

    # 3. Mask email addresses
    sanitized = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "********",
        sanitized,
    )

    # 4. Mask order numbers (6 or more digits)
    sanitized = re.sub(r"\b\d{6,}\b", "********", sanitized)

    # 5. Mask common conversational name salutations (e.g., "Hi Lisa,", "No problem, Greg.")
    non_names = {"there", "everyone", "all", "team", "support", "again"}

    def _mask_hi_name(match: re.Match) -> str:
        greeting, quote1, name, quote2 = match.groups()
        if name.lower() in non_names:
            return match.group(0)
        return f"{greeting} {quote1}********{quote2}"

    sanitized = re.sub(
        r"\b([Hh]i|[Hh]ello|[Hh]ey)\s+(\"?)([A-Z][a-z]+)(\"?)(?=[,!.\s])",
        _mask_hi_name,
        sanitized,
    )
    sanitized = re.sub(
        r"\b(No problem|sorry to hear that|Thank you|Sure|Of course|Alright|trouble|inconvenience|with this|Certainly),\s+([A-Z][a-z]+)(?=[.!?,])",
        r"\1, ********",
        sanitized,
    )

    # 6. Replace swear/expletive words used in the dataset with 😤
    swear_words = r"\b(blasted|blast|darn|friggin|fudge|crikey)\b"
    sanitized = re.sub(swear_words, "😤", sanitized, flags=re.IGNORECASE)

    return sanitized


def parse_args() -> argparse.Namespace:
    """Parse command-line input."""
    parser = argparse.ArgumentParser(
        description="Sanitize and classify customer support chats safely."
    )
    parser.add_argument("file_path", type=Path, help="Path to the input file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the JSON output in UTF-8 (prevents UTF-16LE encoding issues)",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    import sys
    from openai import APIError, AuthenticationError, RateLimitError

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    file_content = _read_text_flexible(args.file_path)
    try:
        result = get_chat_completion(file_content)
    except RateLimitError as err:
        sys.exit(
            "Error de cuota de OpenAI (429 - insufficient_quota): "
            "La cuenta asociada a tu OPENAI_API_KEY no tiene saldo/créditos disponibles "
            "en https://platform.openai.com/settings/organization/billing/.\n"
            f"Detalle: {err.message}"
        )
    except AuthenticationError:
        sys.exit(
            "Error de autenticación (401): La OPENAI_API_KEY en tu archivo .env no es válida."
        )
    except APIError as err:
        sys.exit(f"Error al consultar la API de OpenAI: {err.message}")

    if args.output is not None:
        if args.output.resolve() == args.file_path.resolve():
            raise ValueError("El archivo de salida (--output) no puede ser el mismo que el de entrada.")
        args.output.write_text(result + "\n", encoding="utf-8")
        print(f"Resultado guardado de forma segura (UTF-8) en: {args.output}")
    else:
        print(result)


def get_chat_completion(content: str) -> str:
    """Sanitize input locally and send a request to the /chat/completions endpoint."""
    client = _get_openai_client()
    safe_content = sanitize_pii(content)
    response = client.chat.completions.create(
        model=SETTINGS["general"]["model"],
        messages=_assemble_chat_messages(safe_content),
        temperature=SETTINGS["general"]["temperature"],
        seed=12345,  # Doesn't do anything for older models
    )
    return response.choices[0].message.content or "{}"


def _assemble_chat_messages(content: str) -> list[dict]:
    """Combine all messages into a well-formatted list of dicts."""
    messages = [
        {"role": "system", "content": SETTINGS["prompts"]["role_prompt"]},
        {"role": "user", "content": SETTINGS["prompts"]["negative_example"]},
        {
            "role": "system",
            "content": SETTINGS["prompts"]["negative_reasoning"],
        },
        {
            "role": "assistant",
            "content": SETTINGS["prompts"]["negative_output"],
        },
        {"role": "user", "content": SETTINGS["prompts"]["positive_example"]},
        {
            "role": "system",
            "content": SETTINGS["prompts"]["positive_reasoning"],
        },
        {
            "role": "assistant",
            "content": SETTINGS["prompts"]["positive_output"],
        },
        {"role": "user", "content": f">>>>>\n{content}\n<<<<<"},
        {"role": "user", "content": SETTINGS["prompts"]["instruction_prompt"]},
    ]
    return messages


if __name__ == "__main__":
    main(parse_args())
