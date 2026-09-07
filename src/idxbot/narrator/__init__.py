"""Narrator layer — never makes trading decisions."""

from idxbot.narrator.gemini import GeminiNarrator, narrate_signal

__all__ = ["GeminiNarrator", "narrate_signal"]
