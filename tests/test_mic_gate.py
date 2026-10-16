import asyncio

from pipecat.frames.frames import InputAudioRawFrame, TextFrame
from pipecat.processors.frame_processor import FrameDirection

from app.session.mic_gate import MicGate, MicGateProcessor


def test_gate_drops_only_mic_audio_while_closed():
    gate = MicGate()
    proc = MicGateProcessor(gate)
    passed = []

    async def fake_push(frame, direction=FrameDirection.DOWNSTREAM):
        passed.append(type(frame).__name__)

    async def no_op(*args, **kwargs):
        return None

    proc.push_frame = fake_push
    proc._FrameProcessor__process_frame = no_op  # skip lifecycle checks in this unit test
    import pipecat.processors.frame_processor as fp

    async def run():
        orig = fp.FrameProcessor.process_frame
        fp.FrameProcessor.process_frame = no_op
        try:
            audio = InputAudioRawFrame(audio=b"\0\0" * 160, sample_rate=16000, num_channels=1)
            await proc.process_frame(audio, FrameDirection.DOWNSTREAM)
            gate.closed = True
            await proc.process_frame(audio, FrameDirection.DOWNSTREAM)
            await proc.process_frame(TextFrame("still flows"), FrameDirection.DOWNSTREAM)
            gate.closed = False
            await proc.process_frame(audio, FrameDirection.DOWNSTREAM)
        finally:
            fp.FrameProcessor.process_frame = orig

    asyncio.run(run())
    assert passed == ["InputAudioRawFrame", "TextFrame", "InputAudioRawFrame"]
