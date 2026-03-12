from pathlib import Path
from .loader import load_models, load_prompts, load_system_prompts

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

load_models(ARTIFACTS_DIR / "models")
load_prompts(ARTIFACTS_DIR / "prompts" / "fewshot")
load_system_prompts(ARTIFACTS_DIR / "prompts" / "system")