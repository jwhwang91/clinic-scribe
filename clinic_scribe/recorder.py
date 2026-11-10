"""마이크 녹음 (sounddevice / PortAudio, macOS CoreAudio).

장치가 지원하는 샘플레이트로 float32 녹음한다. 16 kHz 변환은 전사 직전에 한다.
오디오 콜백은 블록을 큐에 넣기만 하고, 누적·레벨 계산은 UI 타이머가 한다.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass

import numpy as np
import sounddevice as sd


class RecorderError(Exception):
    pass


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    channels: int
    default_samplerate: int
    is_default: bool

    def label(self) -> str:
        default = " (시스템 기본)" if self.is_default else ""
        return f"{self.name}{default} · {self.default_samplerate / 1000:g} kHz"


def list_input_devices(refresh: bool = False) -> list[InputDevice]:
    """입력 채널이 있는 장치 목록. refresh=True 면 새로 연결된 장치까지 다시 읽는다(녹음 중에는 금지)."""
    try:
        if refresh:
            # PortAudio 는 초기화 시점의 장치 목록을 유지하므로 다시 초기화해야 새 장치가 보인다.
            sd._terminate()
            sd._initialize()
        devices = sd.query_devices()
    except Exception as exc:
        raise RecorderError(f"오디오 장치를 조회하지 못했습니다: {exc}") from exc
    default_index = _default_input_index()
    result = []
    for index, info in enumerate(devices):
        if int(info["max_input_channels"]) > 0:
            result.append(
                InputDevice(
                    index=index,
                    name=str(info["name"]),
                    channels=int(info["max_input_channels"]),
                    default_samplerate=int(info["default_samplerate"]),
                    is_default=index == default_index,
                )
            )
    return result


def _default_input_index() -> int:
    try:
        index = sd.default.device[0]
        if index is not None and int(index) >= 0:
            return int(index)
        return int(sd.query_devices(kind="input")["index"])
    except Exception:
        return -1


class Recorder:
    def __init__(self) -> None:
        self._stream: sd.InputStream | None = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._chunks: list[np.ndarray] = []
        self._frames = 0
        self._peak = 0.0
        self.samplerate = 0
        self.channels = 0
        self.device: InputDevice | None = None

    @property
    def active(self) -> bool:
        return self._stream is not None

    @property
    def elapsed(self) -> float:
        return self._frames / self.samplerate if self.samplerate else 0.0

    def start(self, device: InputDevice) -> None:
        if self._stream is not None:
            raise RecorderError("이미 녹음 중입니다.")
        channels = max(1, min(2, device.channels))
        samplerate = self._pick_samplerate(device, channels)
        self._reset_buffers()
        try:
            stream = sd.InputStream(
                device=device.index,
                samplerate=samplerate,
                channels=channels,
                dtype="float32",
                callback=self._callback,
            )
            stream.start()
        except Exception as exc:  # PortAudioError, OSError 등
            raise RecorderError(f"마이크 입력 스트림을 열지 못했습니다 ({device.name}): {exc}") from exc
        self._stream = stream
        self.samplerate = samplerate
        self.channels = channels
        self.device = device

    def _pick_samplerate(self, device: InputDevice, channels: int) -> int:
        for rate in dict.fromkeys((device.default_samplerate, 48000, 44100, 32000, 16000)):
            try:
                sd.check_input_settings(device=device.index, channels=channels, samplerate=rate, dtype="float32")
                return int(rate)
            except Exception:
                continue
        raise RecorderError(f"'{device.name}' 이(가) 지원하는 샘플레이트를 찾지 못했습니다.")

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        self._queue.put(indata.copy())

    def drain(self) -> list[np.ndarray]:
        """콜백이 쌓아 둔 새 블록을 누적하고 돌려준다 (UI 스레드에서 호출)."""
        blocks: list[np.ndarray] = []
        while True:
            try:
                blocks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        for block in blocks:
            self._chunks.append(block)
            self._frames += len(block)
            if block.size:
                self._peak = max(self._peak, float(np.max(np.abs(block))))
        return blocks

    def stream_alive(self) -> bool:
        return self._stream is not None and bool(self._stream.active)

    def looks_digitally_silent(self) -> bool:
        """1.5초 넘게 샘플이 전부 정확히 0 이면 True.

        실제 마이크는 잡음 때문에 0 이 이어지지 않는다. macOS 에서 마이크 권한이 거부되면
        스트림은 열리지만 0 만 들어오므로 이것으로 권한 문제를 알린다.
        """
        return self.samplerate > 0 and self._frames > 1.5 * self.samplerate and self._peak == 0.0

    def stop(self) -> tuple[np.ndarray, int]:
        """녹음을 끝내고 마이크를 놓는다. (프레임 수, 채널) float32 와 샘플레이트를 돌려준다."""
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
        self.drain()
        if self._chunks:
            audio = np.concatenate(self._chunks)
        else:
            audio = np.zeros((0, max(1, self.channels)), dtype=np.float32)
        self._chunks = []
        return audio, self.samplerate

    def abort(self) -> None:
        """녹음을 버리고 마이크를 놓는다 (취소·앱 종료)."""
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._chunks = []
        self._frames = 0
        self._peak = 0.0
