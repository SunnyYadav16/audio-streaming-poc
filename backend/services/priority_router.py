"""
Priority router for multi-participant conversation rooms.

Determines TTS output priority based on speaker role.
The creator (role "a") always gets HIGH priority — their translated
speech interrupts any in-progress TTS on recipients.

All other participants get NORMAL priority — their TTS is queued.
"""

from typing import Literal

Priority = Literal["high", "normal"]


class PriorityRouter:
    """
    Stateless role → priority mapper.

    Unlike the previous ``TurnStateMachine``, this class does **not** gate
    speech input — everyone can speak simultaneously.  It only controls
    output priority so the creator's TTS is always heard first.
    """

    @staticmethod
    def get_priority(role: str) -> Priority:
        """Return TTS priority for a given participant role."""
        return "high" if role == "a" else "normal"

    @staticmethod
    def get_speaker_role(role: str) -> str:
        """Return a human-readable role label for transcript messages."""
        return "creator" if role == "a" else "participant"
