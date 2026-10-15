"""Run the document path without voice: PDF -> parse -> command -> translation -> summary.

    uv run python -m scripts.translate_cli samples/sample.pdf "translate this document into English"
"""

import asyncio
import sys
from pathlib import Path

from app.commands.parser import Intent, parse_command
from app.documents.models import DocumentStatus
from app.documents.store import get_store
from app.translation.service import TranslationService


async def main(path: str, command_text: str):
    store = get_store()
    doc = await store.ingest(session_id="cli", filename=Path(path).name, data=Path(path).read_bytes())
    while doc.status == DocumentStatus.PROCESSING:
        await asyncio.sleep(0.1)
    print("document:", doc.summary())
    if doc.status != DocumentStatus.READY:
        return

    command = await parse_command(command_text)
    print("command:", command)
    if command.intent != Intent.TRANSLATE:
        return

    async def progress(done, total):
        print(f"  progress {done}/{total}")

    result = await TranslationService().translate_document(doc, command.target_language or "en", progress)
    print("\nSUMMARY (spoken):", result.summary_text)
    print("warnings:", result.warnings, "| elapsed:", result.elapsed_secs, "s")
    print("\nTRANSLATED TEXT:\n" + result.translated_text)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "translate this document into English"))
