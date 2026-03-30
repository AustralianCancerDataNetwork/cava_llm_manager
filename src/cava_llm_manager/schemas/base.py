import json
from typing import Any, get_args, get_origin

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from .soft_enum import SoftEnum

class LLMOutputModel(BaseModel):
    """
    Base class for tolerant LLM return models.

    Use this for any schema that validates model output which may contain:
    - extra keys not present in the schema
    - `SoftEnum` values that need normalisation or fallback
    - nested model collections returned as a single object instead of a list
    - loosely typed list-like values
    - partial non-conforming content that should be preserved for review

    Unknown fields are collected into `extra_string_results` rather than
    causing validation to fail, which lets downstream code keep valid
    structured fields while still surfacing unexpected model output. Any
    fields typed as `SoftEnum` are also normalised automatically before
    field-level validation runs.
    """

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )

    extra_string_results: list[str] | None = Field(
        default=None,
        description=(
            "Unexpected LLM outputs coerced into strings so validation can"
            " succeed without dropping non-conforming content entirely."
        ),
    )

    @staticmethod
    def coerce_list(value: Any) -> list[Any]:
        """
        Normalise common LLM list-shape mistakes into a Python list.

        Helpful when a model returns:
        - `null` for an empty collection
        - a single object instead of a list of objects
        - a tuple-like structure from upstream preprocessing
        """
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, dict):
            return [value]
        return []

    @classmethod
    def _coerce_extra_strings(cls, value: Any) -> list[str]:
        """
        Convert non-conforming values into reviewable string payloads.

        This is intentionally lossy for complex types: the goal is to avoid
        validation failure while preserving enough detail for debugging or
        auditing unusual model output.
        """
        if value is None:
            return []

        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []

        if isinstance(value, (int, float, bool)):
            return [str(value)]

        if isinstance(value, dict):
            return [json.dumps(value, ensure_ascii=True, sort_keys=True)]

        if isinstance(value, (list, tuple, set)):
            results: list[str] = []
            for item in value:
                results.extend(cls._coerce_extra_strings(item))
            return results

        text = str(value).strip()
        return [text] if text else []

    @classmethod
    def _resolve_soft_enum_type(cls, annotation: Any) -> type[SoftEnum] | None:
        """
        Resolve a model field annotation to a concrete SoftEnum subclass.

        Supports direct `MyEnum` annotations as well as wrappers like
        `MyEnum | None` / `Optional[MyEnum]`.
        """
        try:
            if isinstance(annotation, type) and issubclass(annotation, SoftEnum):
                return annotation
        except TypeError:
            pass

        for arg in get_args(annotation):
            resolved = cls._resolve_soft_enum_type(arg)
            if resolved is not None:
                return resolved

        return None

    @classmethod
    def _is_model_list_field(cls, annotation: Any) -> bool:
        """
        Identify fields typed as lists of nested Pydantic models.

        This is used for tolerant coercion of batch/report collections like
        `reports`, `tests`, or similar child-object arrays. It intentionally
        does not apply to primitive list fields such as `enum_errors`.
        """
        origin = get_origin(annotation)
        if origin is list:
            args = get_args(annotation)
            if not args:
                return False

            item_type = args[0]
            try:
                return isinstance(item_type, type) and issubclass(
                    item_type,
                    BaseModel,
                )
            except TypeError:
                return False

        for arg in get_args(annotation):
            if cls._is_model_list_field(arg):
                return True

        return False

    @model_validator(mode="before")
    @classmethod
    def collect_extra_results(cls, value: Any) -> Any:
        """
        Pull unknown keys out of incoming model data before field validation.

        Any unexpected fields are removed from the payload and stored in
        `extra_string_results`. This allows child schemas to stay specific
        without being brittle when the LLM emits extra text or objects.
        """
        if not isinstance(value, dict):
            return value

        data = dict(value)
        known_fields = set(cls.model_fields)
        extra_results = cls._coerce_extra_strings(
            data.get("extra_string_results")
        )

        for key in list(data):
            if key in known_fields:
                continue
            extra_results.extend(cls._coerce_extra_strings(data.pop(key)))

        if extra_results:
            data["extra_string_results"] = extra_results
        elif "extra_string_results" in data:
            data["extra_string_results"] = None

        return data

    @model_validator(mode="before")
    @classmethod
    def normalise_soft_enum_fields(cls, value: Any) -> Any:
        """
        Normalise all SoftEnum-typed fields declared on the model.

        If a raw value cannot be matched cleanly, the enum falls back to its
        default and the original unmatched value is appended to
        `enum_errors`, when that field exists on the model.
        """
        if not isinstance(value, dict):
            return value

        data = dict(value)
        enum_errors = list(cls.coerce_list(data.get("enum_errors")))

        for field_name, field_info in cls.model_fields.items():
            enum_type = cls._resolve_soft_enum_type(field_info.annotation)
            if enum_type is None or field_name not in data:
                continue

            parsed, error = enum_type.parse(data.get(field_name))
            data[field_name] = parsed

            if error:
                enum_errors.append(error)

        if "enum_errors" in cls.model_fields:
            data["enum_errors"] = enum_errors

        return data

    @model_validator(mode="before")
    @classmethod
    def coerce_model_list_fields(cls, value: Any) -> Any:
        """
        Coerce nested model collections into lists before validation.

        This handles a common LLM failure mode where a field declared as
        `list[ChildModel]` is returned as a single object or `null`.
        """
        if not isinstance(value, dict):
            return value

        data = dict(value)

        for field_name, field_info in cls.model_fields.items():
            if field_name not in data:
                continue
            if not cls._is_model_list_field(field_info.annotation):
                continue

            data[field_name] = cls.coerce_list(data.get(field_name))

        return data

    @field_validator("extra_string_results", mode="before")
    @classmethod
    def coerce_extra_string_results(
        cls,
        value: Any,
    ) -> list[str] | None:
        results = cls._coerce_extra_strings(value)
        return results or None


class LLMReportModel(LLMOutputModel):
    """
    Base class for per-report items nested inside a batch result.

    Inherit from this when each validated object should carry:
    - `report_id`: the 1-based batch position
    - `input_text`: the original source text for traceability

    Because this inherits from `LLMOutputModel`, report-level schemas also
    benefit from tolerant extra-field collection via `extra_string_results`.
    """

    report_id: int = Field(
        default=0,
        description="1-based position of the source item in the batch",
    )
    input_text: str = Field(
        default="",
        description="Exact input text sent to the LLM for this report",
    )

    @field_validator("report_id", mode="before")
    @classmethod
    def coerce_report_id(cls, value: Any) -> int:
        if value in (None, ""):
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
