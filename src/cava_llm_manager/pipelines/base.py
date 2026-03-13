from dataclasses import dataclass
from typing import Type, Any
from pydantic import BaseModel
import json
from ..models.registry import get_model, get_system_prompt, get_prompt



@dataclass
class PipelineSpec:
    name: str
    model_id: str
    system_prompt_id: str
    fewshot_id: str | None = None
    return_schema: Type[BaseModel] | None = None
    inject_schema: bool = False

    @property
    def model(self):
        return get_model(self.model_id)

    @property
    def fewshot_examples(self) -> list[dict[str, Any]] | None:
        if not self.fewshot_id:
            return None
        return get_prompt(self.fewshot_id)["examples"]

    @property
    def root_collection_name(self) -> str:
        if self.return_schema is None:
            raise ValueError(
                f"Pipeline '{self.name}' has no return_schema"
            )

        fields = list(self.return_schema.model_fields.keys())

        if len(fields) != 1:
            raise ValueError(
                f"{self.return_schema.__name__} must have exactly one top-level field"
            )

        return fields[0]

    @property
    def item_label(self) -> str:
        root = self.root_collection_name
        if root.endswith("s"):
            return root[:-1].capitalize()
        return root.capitalize()

    @property
    def system_prompt_text(self) -> str:
        base = get_system_prompt(self.system_prompt_id)
        sections = [base]

        if self.fewshot_examples:
            sections.append("\nExamples:\n")

            for example in self.fewshot_examples:
                sections.append("INPUT\n")

                inputs = example.get("inputs")
                if inputs is None:
                    single_input = example.get("input")
                    if single_input is None:
                        raise ValueError(
                            f"Few-shot example must contain 'input' or 'inputs': {example}"
                        )
                    inputs = [single_input]

                for i, item in enumerate(inputs, start=1):
                    sections.append(f"{self.item_label} {i}:")
                    sections.append(str(item).strip())
                    sections.append("")

                sections.append("OUTPUT\n")
                sections.append(json.dumps(example["output"], indent=2))
                sections.append("")

        return "\n".join(sections).strip()

    def build_prompt(self, items: list[str]) -> str:
        sections = []

        for i, item in enumerate(items, start=1):
            sections.append(f"{self.item_label} {i}:")
            sections.append(item)
            sections.append("")

        return "\n".join(sections).strip()

    def render_prompt(self, items: list[str]) -> str:
        system = self.system_prompt_text
        user = self.build_prompt(items)

        return (
            "===== SYSTEM PROMPT =====\n\n"
            + system
            + "\n\n===== USER PROMPT =====\n\n"
            + user
        )
    
    def preview(self, items: list[str]) -> None:
        """
        Pretty-print the full prompt.
        """

        print(self.render_prompt(items))

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "resolved_model": self.model.id,
            "server_label": self.model.server_label,
            "system_prompt": self.system_prompt_text,
            "fewshot_id": self.fewshot_id,
            "return_schema": (
                self.return_schema.__name__ if self.return_schema else None
            ),
            "inject_schema": self.inject_schema,
            "root_collection_name": (
                self.root_collection_name if self.return_schema else None
            ),
        }