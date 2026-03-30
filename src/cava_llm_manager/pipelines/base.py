from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type, Any, Literal, get_args, get_origin
from pydantic import BaseModel
import json
from ..models.registry import get_model, get_system_prompt, get_prompt
from ..schemas.base import LLMOutputModel


@dataclass
class ConfidenceConfig:
    enabled: bool = False
    review_pipeline: PipelineSpec | None = None
    revision_pipeline: PipelineSpec | None = None
    attempt_revision: bool = False
    max_revision_rounds: int = 1
    acceptance_policy: str = "revise_once_then_annotate"
    return_sidecar: bool = True


@dataclass
class ConfidenceBatchSummary:
    reviewed_reports: int = 0
    accepted_reports: int = 0
    revised_reports: int = 0
    unresolved_reports: int = 0
    review_failures: int = 0


@dataclass
class ReportConfidenceTrace:
    report_id: int
    input_text: str
    original_report: dict[str, Any]
    final_report: dict[str, Any]
    initial_review: dict[str, Any] | None = None
    revision_review: dict[str, Any] | None = None
    revised: bool = False
    unresolved: bool = False
    revision_error: str | None = None
    review_error: str | None = None


@dataclass
class ConfidenceBatchResult:
    result: BaseModel | dict[str, Any] | None
    original_result: BaseModel | dict[str, Any] | None
    report_confidence: list[ReportConfidenceTrace] = field(default_factory=list)
    summary: ConfidenceBatchSummary = field(default_factory=ConfidenceBatchSummary)


@dataclass
class PipelineSpec:
    name: str
    model_id: str
    system_prompt_id: str
    fewshot_id: str | None = None
    return_schema: Type[BaseModel] | None = None
    inject_schema: bool = False
    host_type: Literal["ollama", "llama_cpp"] = "ollama"
    host_url: str | None = None
    host_options: dict[str, Any] = field(default_factory=dict)
    prompt_template: str | None = None
    confidence: ConfidenceConfig | None = None

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
        # filter out fields that have only been included on the model
        # for LLM result tolerance and are not part of the actual output schema
        helper_fields = set(LLMOutputModel.model_fields.keys())
        fields = [
            name
            for name in self.return_schema.model_fields.keys()
            if name not in helper_fields
        ]

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
    def report_model(self) -> Type[BaseModel]:
        if self.return_schema is None:
            raise ValueError(
                f"Pipeline '{self.name}' has no return_schema"
            )

        field_info = self.return_schema.model_fields[self.root_collection_name]
        annotation = field_info.annotation
        origin = get_origin(annotation)

        if origin is list:
            args = get_args(annotation)
            if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                return args[0]

        raise ValueError(
            f"{self.return_schema.__name__}.{self.root_collection_name} "
            "must be typed as list[BaseModel]"
        )

    @property
    def serialized_return_schema(self) -> str:
        if self.return_schema is None:
            raise ValueError(
                f"Pipeline '{self.name}' has no return_schema"
            )

        return json.dumps(
            self.return_schema.model_json_schema(),
            indent=2,
        )

    @property
    def serialized_report_schema(self) -> str:
        return json.dumps(
            self.report_model.model_json_schema(),
            indent=2,
        )

    @property
    def system_prompt_text(self) -> str:
        base = get_system_prompt(self.system_prompt_id)
        sections = [base]

        if self.inject_schema:
            sections.append("\nTarget JSON schema:\n")
            sections.append(self.serialized_return_schema)
            sections.append("")

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
            "host_type": self.host_type,
            "host_url": self.host_url,
            "confidence_enabled": (
                self.confidence.enabled if self.confidence else False
            ),
            "root_collection_name": (
                self.root_collection_name if self.return_schema else None
            ),
        }
