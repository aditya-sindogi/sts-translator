"""Stop listening while the bot is busy.

``MicGateProcessor`` sits right after the transport input. While the gate is closed it
drops the user's microphone audio, so nothing downstream hears it: no VAD, no transcript,
no interruption. The assistant closes the gate when it starts handling a command and
reopens it once any translation job has finished and the bot has stopped speaking.
"""

from pipecat.frames.frames import Frame, InputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class MicGate:
    """Shared switch between the gate processor and the assistant."""

    def __init__(self):
        self.closed = False


class MicGateProcessor(FrameProcessor):
    def __init__(self, gate: MicGate, **kwargs):
        super().__init__(**kwargs)
        self._gate = gate

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self._gate.closed and isinstance(frame, InputAudioRawFrame):
            return  # the bot is busy: drop mic audio
        await self.push_frame(frame, direction)
