from enum import Enum
from typing import Any, ClassVar


class SoftEnum(str, Enum):
    """
    Enum that tolerates invalid inputs by normalising or
    falling back to a default value instead of raising.
    """

    @classmethod
    def fallback(cls) -> str:
        return list(cls)[0].value

    @classmethod
    def normalisations(cls) -> dict[str, str]:
        return {}

    @classmethod
    def parse(cls, value: Any) -> tuple[str, str | None]:
        if value is None:
            return cls.fallback(), None

        v = str(value).strip().lower()
        valid = {e.value for e in cls}
        if v in valid:
            return v, None

        norms = cls.normalisations()

        if v in norms:
            return norms[v], None

        for phrase, canonical in norms.items():
            if phrase in v:
                return canonical, None

        return cls.fallback(), v