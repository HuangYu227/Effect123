from __future__ import annotations

from effectcma_flow.data.effects import EffectSpec


def spec_to_text(spec: EffectSpec, length: int) -> tuple[str, list[str]]:
    strength_phrase = strength_to_phrase(spec)
    effect_phrase = effect_to_phrase(spec.effect_type)
    time_phrase = time_to_phrase(spec.start, spec.end, length)
    channel_phrase = channels_to_phrase(spec.channels)
    full_text = f"{channel_phrase} shows {strength_phrase} {effect_phrase} in the {time_phrase}."
    slot = f"{strength_phrase} {effect_phrase} in the {time_phrase} for {channel_phrase}"
    return full_text, [slot]


def effect_to_phrase(effect_type: str) -> str:
    phrases = {
        "level_shift": "level shift",
        "trend_up": "upward trend",
        "trend_down": "downward trend",
        "volatility_up": "volatility increase",
        "volatility_down": "volatility decrease",
        "spike": "local spike",
        "drop": "local drop",
    }
    return phrases.get(effect_type, effect_type.replace("_", " "))


def strength_to_phrase(spec: EffectSpec) -> str:
    value = abs(float(spec.strength))
    if spec.effect_type == "volatility_down":
        value = 1.0 - min(value, 1.0)
    elif spec.effect_type == "volatility_up":
        value = max(0.0, value - 1.0)
    if value < 0.35:
        return "weak"
    if value < 0.8:
        return "moderate"
    return "strong"


def time_to_phrase(start: int, end: int, length: int) -> str:
    mid = (start + end) / 2.0
    rel = mid / max(float(length), 1.0)
    span = end - start
    if span >= 0.8 * length:
        return "whole window"
    if rel < 0.3:
        return "early segment"
    if rel > 0.7:
        return "late segment"
    return "middle segment"


def channels_to_phrase(channels: tuple[int, ...]) -> str:
    labels = [f"channel {c}" for c in channels]
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f" and {labels[-1]}"

