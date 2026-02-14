"""
Turn-taking state machine for bidirectional conversation rooms.

Manages who has the "floor" (is allowed to speak) and applies
echo-suppression lockouts after TTS playback.

State transitions
-----------------

    IDLE ─── speech_start(A) ──► A_SPEAKING
    A_SPEAKING ─── speech_end(A) ──► A_PROCESSING  (grace timer starts)
    A_PROCESSING ─── grace expires ──► IDLE
    A_PROCESSING ─── speech_start(A) ──► A_SPEAKING (grace cancelled, same user continues)

    Same transitions for B, but with a *shorter* grace period.

    During A_SPEAKING / A_PROCESSING:
        B's speech_start is REJECTED (A holds the floor).
    After TTS is sent to B:
        B is LOCKED for (TTS duration + LOCKOUT_BUFFER_MS).
        lock_user(B) is a no-op if B currently holds the floor.

Asymmetric grace periods
------------------------
Courtroom conversations are often asymmetric: the official (role A)
asks long, multi-part questions with natural pauses, while the
respondent (role B) gives shorter answers.

    Role A (creator / official):  grace_a_ms = 2 000 ms (default)
    Role B (joiner / respondent): grace_b_ms = 1 000 ms (default)

Both values are configurable via the constructor.
"""

import enum
import time
from typing import Optional


class FloorState(str, enum.Enum):
    """Observable state of the conversation turn."""
    IDLE = "idle"
    A_SPEAKING = "a_speaking"
    A_PROCESSING = "a_processing"
    B_SPEAKING = "b_speaking"
    B_PROCESSING = "b_processing"


class TurnStateMachine:
    """
    Lightweight turn-taking + echo-suppression for a 2-party room.

    Roles are ``'a'`` (room creator / official) and ``'b'`` (room joiner
    / respondent).

    The grace period is **per-role** to support asymmetric conversations
    like courtroom proceedings where the official asks long, multi-part
    questions (needs a longer pause allowance) while the respondent gives
    shorter answers (quicker turn release).

    Parameters
    ----------
    lockout_buffer_ms : float
        Extra silence appended to the TTS duration before unlocking a
        user's mic.  Prevents the tail-end of speaker audio from
        triggering VAD.
    grace_a_ms : float
        After speech_end, how long the floor stays reserved for role A
        (creator / official).  Default 2 000 ms — long enough for a
        natural pause between sub-questions.
    grace_b_ms : float
        Same, for role B (joiner / respondent).  Default 1 000 ms —
        shorter answers release the floor sooner.
    """

    def __init__(
        self,
        lockout_buffer_ms: float = 200,
        grace_a_ms: float = 2000,
        grace_b_ms: float = 1000,
    ):
        self.lockout_buffer_ms = lockout_buffer_ms
        self.grace_ms: dict[str, float] = {"a": grace_a_ms, "b": grace_b_ms}

        self.state: FloorState = FloorState.IDLE
        self._floor_holder: Optional[str] = None
        self._lockout: dict[str, float] = {"a": 0.0, "b": 0.0}
        self._grace_expiry: float = 0.0

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _check_grace(self):
        """Auto-release the floor if the grace period has expired."""
        if (
            self._floor_holder is not None
            and self._grace_expiry > 0
            and time.monotonic() >= self._grace_expiry
        ):
            self._floor_holder = None
            self._grace_expiry = 0.0
            self.state = FloorState.IDLE

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def is_locked(self, role: str) -> bool:
        """True if the user's mic is echo-locked."""
        return time.monotonic() < self._lockout.get(role, 0.0)

    def holds_floor(self, role: str) -> bool:
        """True if *role* currently owns the conversational floor."""
        self._check_grace()
        return self._floor_holder == role

    def try_speech_start(self, role: str) -> bool:
        """
        Called on VAD ``speech_start``.

        Returns True if the user is allowed to speak (floor granted or
        already held).  Returns False if locked or the other user holds
        the floor.
        """
        self._check_grace()

        if self.is_locked(role):
            return False

        if self._floor_holder is None:
            # Floor is free — claim it
            self._floor_holder = role
            self._grace_expiry = 0.0
            self.state = (
                FloorState.A_SPEAKING if role == "a" else FloorState.B_SPEAKING
            )
            return True

        if self._floor_holder == role:
            # Already holding the floor (continuing after a brief pause)
            self._grace_expiry = 0.0
            self.state = (
                FloorState.A_SPEAKING if role == "a" else FloorState.B_SPEAKING
            )
            return True

        # Other user holds the floor
        return False

    def on_speech_end(self, role: str) -> bool:
        """
        Called on VAD ``speech_end``.

        Transitions to *PROCESSING* and starts the role-specific grace
        period (longer for the official, shorter for the respondent).
        Returns True if this was the active speaker.
        """
        if self._floor_holder != role:
            return False

        self.state = (
            FloorState.A_PROCESSING if role == "a" else FloorState.B_PROCESSING
        )
        self._grace_expiry = (
            time.monotonic() + self.grace_ms[role] / 1000.0
        )
        return True

    def lock_user(self, role: str, duration_ms: float):
        """
        Lock a user's mic for echo suppression after TTS.

        Has **no effect** if *role* currently holds the floor (they are
        actively speaking and should not be interrupted).
        """
        if self._floor_holder == role:
            return  # don't lock the active speaker

        total_ms = duration_ms + self.lockout_buffer_ms
        self._lockout[role] = time.monotonic() + total_ms / 1000.0

    def on_interrupt(self, interrupter_role: str):
        """
        Handle barge-in: give the floor to the interrupter immediately.

        Called when a user starts speaking while TTS is playing on their
        end.  Clears their echo-suppression lockout and grants them the
        floor so that the in-progress ``speech_end`` event from the VAD
        will be properly accepted and routed through ASR → MT → TTS.
        """
        # Clear interrupter's echo lockout
        self._lockout[interrupter_role] = 0.0

        # Give the floor directly to the interrupter
        self._floor_holder = interrupter_role
        self._grace_expiry = 0.0
        self.state = (
            FloorState.A_SPEAKING if interrupter_role == "a"
            else FloorState.B_SPEAKING
        )

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                         #
    # ------------------------------------------------------------------ #

    def get_status(self) -> dict:
        """Snapshot of current state (for logging / REST / frontend)."""
        self._check_grace()
        now = time.monotonic()
        return {
            "state": self.state.value,
            "floor_holder": self._floor_holder,
            "grace_ms": dict(self.grace_ms),
            "a_locked": now < self._lockout.get("a", 0.0),
            "b_locked": now < self._lockout.get("b", 0.0),
            "a_lock_remaining_ms": max(
                0, int((self._lockout.get("a", 0.0) - now) * 1000)
            ),
            "b_lock_remaining_ms": max(
                0, int((self._lockout.get("b", 0.0) - now) * 1000)
            ),
        }

    def __repr__(self) -> str:
        s = self.get_status()
        parts = [f"state={s['state']}", f"floor={s['floor_holder']}"]
        if s["a_locked"]:
            parts.append(f"A_locked({s['a_lock_remaining_ms']}ms)")
        if s["b_locked"]:
            parts.append(f"B_locked({s['b_lock_remaining_ms']}ms)")
        return f"TurnStateMachine({', '.join(parts)})"
