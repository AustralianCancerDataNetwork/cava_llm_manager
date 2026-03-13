import asyncio
import httpx
import logging
from ..models.registry import get_model, get_system_prompt, get_prompt
from ..utils.json_utils import extract_json
from ..utils.config import OLLAMA_URL

logger = logging.getLogger(__name__)

def extract_ollama_message(data: dict) -> str:

    if "error" in data:
        logger.error("Ollama returned error: %s", data["error"])
        raise RuntimeError(f"Ollama error: {data['error']}")

    if "message" not in data:
        logger.error("Unexpected Ollama response: %s", data)
        raise RuntimeError(
            f"Unexpected Ollama response structure:\n{data}"
        )

    msg = data["message"]

    if "content" not in msg:
        logger.error("Ollama message missing content: %s", data)
        raise RuntimeError(
            f"Ollama response missing message content:\n{data}"
        )

    return msg["content"]

async def retry_async(fn, retries=3):

    for attempt in range(retries):

        try:
            return await fn()

        except Exception as e:
            if attempt == retries - 1:
                logger.exception(
                    "Retry failed after %d attempts",
                    retries
                )
                raise
            logger.warning(
                "Retrying async call (attempt %d/%d): %s",
                attempt + 1,
                retries,
                str(e),
            )
            await asyncio.sleep(1)
async def run_pipeline_batch(pipeline, items, client):

    model = pipeline.model

    logger.debug(
        "Running pipeline batch | pipeline=%s | model=%s | batch_size=%d",
        pipeline.name,
        model.id,
        len(items),
    )

    system_prompt = pipeline.system_prompt_text
    prompt = pipeline.build_prompt(items)

    payload = {
        "model": model.server_label,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }

    async def call():

        logger.debug(
            "Sending request to Ollama | model=%s | batch_size=%d",
            model.server_label,
            len(items),
        )

        r = await client.post(OLLAMA_URL, json=payload, timeout=600)

        data = r.json()

        raw = extract_ollama_message(data)

        try:
            parsed = extract_json(raw)

        except Exception:
            logger.error(
                "Failed to extract JSON from model output | model=%s | preview=%s",
                model.server_label,
                raw[:300],
            )
            raise

        result = pipeline.return_schema.model_validate(parsed)

        logger.debug(
            "Batch completed successfully | model=%s | batch_size=%d",
            model.server_label,
            len(items),
        )

        return result

    return await retry_async(call)

async def run_batch_query(client, payload, retries=4):

    for attempt in range(retries):

        try:
            r = await client.post(OLLAMA_URL, json=payload["data"], timeout=600)

            return r.json()

        except Exception:

            if attempt == retries - 1:
                raise


async def process_batches(all_reports, batch_size=5, workers=4):

    batches = [
        all_reports[i:i + batch_size]
        for i in range(0, len(all_reports), batch_size)
    ]

    sem = asyncio.Semaphore(workers)

    async with httpx.AsyncClient() as client:

        async def task(batch):

            async with sem:
                return await run_batch_query(client, batch)

        tasks = [task(b) for b in batches]

        return await asyncio.gather(*tasks)
    
async def run_pipeline(
    pipeline,
    items,
    batch_size=5,
    workers=4
):

    logger.info(
        "Starting pipeline | pipeline=%s | items=%d | batch_size=%d | workers=%d",
        pipeline.name,
        len(items),
        batch_size,
        workers,
    )

    batches = [
        items[i:i + batch_size]
        for i in range(0, len(items), batch_size)
    ]

    logger.debug("Created %d batches", len(batches))

    sem = asyncio.Semaphore(workers)

    async with httpx.AsyncClient() as client:

        async def task(batch_id, batch):

            async with sem:

                logger.debug(
                    "Processing batch %d | size=%d",
                    batch_id,
                    len(batch),
                )

                return await run_pipeline_batch(
                    pipeline,
                    batch,
                    client
                )

        tasks = [task(i, b) for i, b in enumerate(batches)]

        results = await asyncio.gather(*tasks)

    logger.info(
        "Pipeline completed | pipeline=%s | batches=%d",
        pipeline.name,
        len(batches),
    )

    return results