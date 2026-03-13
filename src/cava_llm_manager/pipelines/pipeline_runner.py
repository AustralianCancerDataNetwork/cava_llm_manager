import asyncio
import httpx
from cava_llm_manager.utils.prompt_utils import build_prompt, schema_to_prompt
from cava_llm_manager.models.registry import get_model
from cava_llm_manager import PROMPT_REGISTRY, SYSTEM_PROMPT_REGISTRY
from cava_llm_manager.utils.json_utils import extract_json
from ..utils.config import OLLAMA_URL

def extract_ollama_message(data: dict) -> str:

    if "error" in data:
        raise RuntimeError(f"Ollama error: {data['error']}")

    if "message" not in data:
        raise RuntimeError(
            f"Unexpected Ollama response structure:\n{data}"
        )

    msg = data["message"]

    if "content" not in msg:
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
                raise

            await asyncio.sleep(1)

async def run_pipeline_batch(pipeline, reports, client):

    model = get_model(pipeline.model_id)

    fewshot = None
    if pipeline.fewshot_id:
        fewshot = PROMPT_REGISTRY[pipeline.fewshot_id]["examples"]

    system_prompt = SYSTEM_PROMPT_REGISTRY[pipeline.system_prompt]

    if pipeline.inject_schema and pipeline.return_schema:
        system_prompt = (
            system_prompt
            + "\n\n"
            + schema_to_prompt(pipeline.return_schema)
        )

    prompt = build_prompt(reports, fewshot)

    payload = {
        "model": model.server_label,
        "format": "json",
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }

    async def call():

        r = await client.post(OLLAMA_URL, json=payload, timeout=600)

        data = r.json()

        raw = extract_ollama_message(data)

        parsed = extract_json(raw)

        for i in parsed["reports"]:
            idx = i["report_id"] - 1
            i["report"] = reports[idx]

        return pipeline.return_schema.model_validate(parsed)

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
    reports,
    batch_size=5,
    workers=4
):

    batches = [
        reports[i:i+batch_size]
        for i in range(0, len(reports), batch_size)
    ]

    sem = asyncio.Semaphore(workers)

    async with httpx.AsyncClient() as client:

        async def task(batch):

            async with sem:
                return await run_pipeline_batch(
                    pipeline,
                    batch,
                    client
                )

        tasks = [task(b) for b in batches]

        return await asyncio.gather(*tasks)