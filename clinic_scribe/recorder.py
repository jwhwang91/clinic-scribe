"""마이크 녹음 (sounddevice / PortAudio: macOS Core Audio, Windows MME, Linux ALSA).

내장·USB·블루투스·오디오 인터페이스 등 어떤 입력 장치든 그 장치가 지원하는 샘플레이트·채널(최대 2)로 float32 녹음하고,
16 kHz mono 변환은 전사 직전에 한다. 장치는 이름으로 기억하므로 다른 컴퓨터에서 같은 이름이 없으면 시스템 기본 입력을 쓴다.
오디오 콜백은 블록을 큐에 넣기만 하고, 누적·레벨 계산은 UI 타이머가 한다.
데모 입력 장치는 오디오 파일을 마이크처럼 실시간 속도로 흘려 보낸다(마이크 없이 대시보드 테스트).
"""

from __future__ import annotations

import bisect
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


class RecorderError(Exception):
    pass


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    channels: int
    default_samplerate: int
    is_default: bool
    demo_file: str | None = None  # 데모 입력: 이 파일을 마이크처럼 재생
    demo_speed: float = 1.0

    def label(self) -> str:
        if self.demo_file:
            return f"데모 입력 · {Path(self.demo_file).name} (가상 대화를 마이크처럼 실시간 재생)"
        default = " (시스템 기본)" if self.is_default else ""
        return f"{self.name}{default} · {self.default_samplerate / 1000:g} kHz"


def demo_device(path: Path, speed: float = 1.0) -> InputDevice:
    info = sf.info(str(path))
    return InputDevice(
        index=-100, name=f"데모 입력 · {path.name}", channels=info.channels,
        default_samplerate=int(info.samplerate), is_default=False, demo_file=str(path), demo_speed=speed,
    )


class FileInputStream:
    """sounddevice.InputStream 처럼 동작하는 데모 입력. 파일을 실시간 속도로 콜백에 흘리고, 끝나면 무음을 보낸다."""

    blocksize = 1024

    def __init__(self, path: str, callback, speed: float = 1.0) -> None:
        self.data, self.samplerate = sf.read(path, dtype="float32", always_2d=True)
        self.channels = self.data.shape[1]
        self.callback = callback
        self.speed = max(0.1, speed)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="clinic-scribe-demo-input", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        started = time.monotonic()
        position = sent = 0
        silence = np.zeros((self.blocksize, self.channels), np.float32)
        while not self._stop.is_set():
            block = self.data[position : position + self.blocksize]
            if len(block) < self.blocksize:
                block = np.concatenate([block, silence[: self.blocksize - len(block)]])
            self.callback(block, self.blocksize, None, None)
            position += self.blocksize
            sent += self.blocksize
            delay = started + sent / (self.samplerate * self.speed) - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    abort = stop

    def close(self) -> None:
        pass


# ALSA 가 입력 장치처럼 내보이지만 사람이 고를 일이 없는 가상 장치. 'default'·'pulse'·'pipewire' 와 실제 하드웨어만 남긴다.
_ALSA_HIDDEN = {"sysdefault", "dsnoop", "dmix", "lavrate", "samplerate", "speexrate", "upmix", "vdownmix", "jack",
                "oss", "null", "pcm"}
_MME_NAME_LIMIT = 30  # Windows MME 는 장치 이름을 31자로 자른다 (끝 공백이 빠지면 30자)


def list_input_devices(refresh: bool = False) -> list[InputDevice]:
    """입력 채널이 있는 장치 목록. refresh=True 면 새로 연결된 장치까지 다시 읽는다(녹음 중에는 금지)."""
    try:
        if refresh:
            # PortAudio 는 초기화 시점의 장치 목록을 유지하므로 다시 초기화해야 새 장치가 보인다.
            sd._terminate()
            sd._initialize()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception as exc:
        raise RecorderError(f"오디오 장치를 조회하지 못했습니다: {exc}") from exc
    return select_input_devices(list(devices), list(hostapis), _default_input_index(), _default_hostapi())


def select_input_devices(devices: list, hostapis: list, default_index: int, default_api: int | None,
                         platform: str | None = None) -> list[InputDevice]:
    """PortAudio 장치 정보 → 고를 수 있는 입력 장치.

    Windows 는 같은 마이크가 오디오 방식(MME·DirectSound·WASAPI·WDM-KS)마다 겹쳐 나오므로 기본 방식(MME, 어떤 샘플레이트든
    Windows 가 변환해 주어 가장 잘 열린다) 것만 보여 주고, MME 가 31자로 자른 이름은 WASAPI 의 전체 이름으로 바꾼다.
    Linux 는 ALSA 가상 장치(sysdefault, dsnoop 등)를 뺀다. 시스템 기본 입력 장치는 언제나 남긴다.
    """
    platform = platform or sys.platform
    wasapi = [i for i, api in enumerate(hostapis) if "WASAPI" in str(api.get("name", ""))]
    full_names = [str(d["name"]) for d in devices
                  if d.get("hostapi") in wasapi and int(d["max_input_channels"]) > 0] if platform == "win32" else []
    result = []
    for index, info in enumerate(devices):
        if int(info["max_input_channels"]) <= 0:
            continue
        name = str(info["name"])
        default = index == default_index
        if default_api is not None and info.get("hostapi", default_api) != default_api and not default:
            continue
        if platform.startswith("linux") and name.split(":")[0].strip().lower() in _ALSA_HIDDEN and not default:
            continue
        if len(name) >= _MME_NAME_LIMIT:
            longer = [full for full in full_names if full.startswith(name)]
            if len(longer) == 1:
                name = longer[0]
        result.append(InputDevice(index=index, name=name, channels=int(info["max_input_channels"]),
                                  default_samplerate=int(info["default_samplerate"]), is_default=default))
    return result


def _default_hostapi() -> int | None:
    try:
        return int(sd.default.hostapi)
    except Exception:
        return None


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
        self._stream: sd.InputStream | FileInputStream | None = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._chunks: list[np.ndarray] = []
        self._offsets: list[int] = []  # 각 블록의 시작 프레임 (snapshot 용)
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
        if device.demo_file:
            self._reset_buffers()
            try:
                stream = FileInputStream(device.demo_file, self._callback, device.demo_speed)
                stream.start()
            except Exception as exc:
                raise RecorderError(f"데모 입력 파일을 열지 못했습니다 ({device.demo_file}): {exc}") from exc
            self._stream, self.samplerate, self.channels, self.device = stream, stream.samplerate, stream.channels, device
            return
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
            self._offsets.append(self._frames)
            self._chunks.append(block)
            self._frames += len(block)
            if block.size:
                self._peak = max(self._peak, float(np.max(np.abs(block))))
        return blocks

    def discard_before(self, seconds: float) -> None:
        """이미 넘겨준 앞부분 블록을 메모리에서 지운다. 시각(프레임 위치)은 녹음 시작 기준 그대로 유지된다."""
        if not self.samplerate:
            return
        frame = int(seconds * self.samplerate)
        keep = max(0, bisect.bisect_right(self._offsets, frame) - 1)
        if keep:
            del self._chunks[:keep]
            del self._offsets[:keep]

    def snapshot(self, start_s: float, end_s: float) -> np.ndarray:
        """지금까지 녹음된 소리 중 [start_s, end_s] 구간 (프레임, 채널). UI 스레드에서만 부른다."""
        empty = np.zeros((0, max(1, self.channels)), np.float32)
        if not self.samplerate or not self._chunks:
            return empty
        a = max(0, int(start_s * self.samplerate))
        b = min(self._frames, int(end_s * self.samplerate))
        if b <= a:
            return empty
        i = max(0, bisect.bisect_right(self._offsets, a) - 1)
        parts = []
        while i < len(self._chunks) and self._offsets[i] < b:
            chunk, offset = self._chunks[i], self._offsets[i]
            parts.append(chunk[max(0, a - offset) : max(0, b - offset)])
            i += 1
        return np.concatenate(parts) if parts else empty

    def stream_alive(self) -> bool:
        return self._stream is not None and bool(self._stream.active)

    def looks_digitally_silent(self) -> bool:
        """1.5초 넘게 샘플이 전부 정확히 0 이면 True.

        실제 마이크는 잡음 때문에 0 이 이어지지 않는다. macOS 에서 마이크 권한이 거부되면
        스트림은 열리지만 0 만 들어오므로 이것으로 권한 문제를 알린다.
        """
        return self.samplerate > 0 and self._frames > 1.5 * self.samplerate and self._peak == 0.0

    def stop(self, keep_audio: bool = True) -> tuple[np.ndarray, int]:
        """녹음을 끝내고 마이크를 놓는다. (프레임 수, 채널) float32 와 샘플레이트를 돌려준다.

        keep_audio=False 면 남은 소리를 이어 붙이지 않는다(이미 snapshot 으로 가져간 경우).
        """
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
        self.drain()
        if self._chunks and keep_audio:
            audio = np.concatenate(self._chunks)
        else:
            audio = np.zeros((0, max(1, self.channels)), dtype=np.float32)
        self._chunks = []
        self._offsets = []
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
        self._offsets = []
        self._frames = 0
        self._peak = 0.0
