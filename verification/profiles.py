from __future__ import annotations

import json
from pydantic import BaseModel, Field


class VerificationProfile(BaseModel):
    name: str
    min_faithfulness_score: float = Field(default=1.0, ge=0.0, le=1.0)
    allow_partial: bool = False
    require_evidence: bool = True
    deterministic_conflicts_fail: bool = True


class VerificationProfileRegistry:
    def __init__(self, profiles: list[VerificationProfile] | None = None, *, default_profile: str = "strict"):
        builtins = profiles or [
            VerificationProfile(name="strict", min_faithfulness_score=1.0, allow_partial=False),
            VerificationProfile(name="standard", min_faithfulness_score=0.9, allow_partial=True),
            VerificationProfile(name="creative", min_faithfulness_score=0.75, allow_partial=True),
        ]
        self._profiles = {profile.name: profile for profile in builtins}
        self.default_profile = default_profile if default_profile in self._profiles else "strict"

    @classmethod
    def from_json(cls, raw: str | None, *, default_profile: str = "strict") -> "VerificationProfileRegistry":
        if not raw:
            return cls(default_profile=default_profile)
        data = json.loads(raw)
        if isinstance(data, dict):
            data = [{"name": name, **dict(value)} for name, value in data.items()]
        profiles = [VerificationProfile.model_validate(item) for item in data]
        return cls(profiles, default_profile=default_profile)

    def resolve(self, name: str | None) -> VerificationProfile:
        key = (name or self.default_profile).strip().lower()
        return self._profiles.get(key, self._profiles[self.default_profile])

    def describe(self) -> dict[str, dict]:
        return {name: profile.model_dump(mode="json") for name, profile in self._profiles.items()}
