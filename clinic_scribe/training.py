"""따라 읽기 학습 (약 1분): 의사가 화면에 나오는 짧은 문장을 하나씩 따라 읽는다.

문장은 의사의 진료과에 맞춰 고른다 (SPECIALTY_PROMPTS): 그 과 의사가 진료실에서 실제로 하는 말.

읽을 문장(정답)을 알고 있으므로 같은 녹음으로
- 목소리를 등록하고(말한 사람 표시),
- Whisper 가 받아 적은 글을 정답과 맞춰 이 의사의 말이 어떻게 잘못 들리는지 찾아 딕션 사전(교정·용어)을 만든다.
학습 전과 후에 정확히 받아 적은 문장 수를 비교해 보여 준다. 녹음은 분석 뒤 버리고, Whisper 모델은 다시 학습시키지 않는다.
"""

from __future__ import annotations

import difflib
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from clinic_scribe.diction import Correction, DictionBook, DictionError, apply_corrections, build_prompt, check_correction
from clinic_scribe.speakers import SAMPLE_RATE, SpeakerEncoder, enrollment_embeddings, speech_only
from clinic_scribe.whisper import TranscriptionResult

# 진료과별 따라 읽기 문장 (과마다 12개): 그 과 의사가 진료실에서 실제로 하는 말 — 증상 묻기, 진찰 안내, 검사·약 설명,
# 생활 지도. 의사는 환자에게 약을 성분명보다 "위산 줄이는 약", "해열제"처럼 말하므로 성분명은 넣지 않고,
# 환자에게도 그대로 쓰는 이름(타이레놀, 타미플루 등)만 둔다. 숫자·단위는 Whisper 가 받아 적는 대로(3번, 30분, 4mm) 쓴다.
# 말을 많이 하는 과(내과·가정의학과·소아청소년과·정신건강의학과)를 앞에 둔다.
GENERAL = "내과"
DEFAULT_PROMPTS = (
    "어디가 불편해서 오셨어요?",
    "언제부터 그러셨어요? 열이나 기침은 없으세요?",
    "명치 쪽이 아프세요, 아니면 배 전체가 아프세요?",
    "밥 먹고 나면 속이 더 쓰리고 신물이 올라오세요?",
    "변은 잘 보세요? 검은 변이나 피는 없었고요?",
    "누워 보세요. 배를 한번 눌러 볼게요.",
    "숨 크게 쉬어 보세요. 가슴 소리 좀 들어 볼게요.",
    "역류성 식도염 같아요. 위산 줄이는 약을 드릴게요.",
    "혈압이 좀 높네요. 집에서도 아침저녁으로 재 보세요.",
    "당화혈색소가 높아서 당뇨약을 조금 늘릴게요.",
    "증상이 계속되면 위내시경을 한번 해 보죠.",
    "피검사 결과 나오면 다음 주에 같이 봐요.",
)
SPECIALTY_PROMPTS: dict[str, tuple[str, ...]] = {
    GENERAL: DEFAULT_PROMPTS,
    "가정의학과": (
        "오늘은 어떤 일로 오셨어요?",
        "건강검진 결과 보니까 콜레스테롤이 조금 높아요.",
        "간 수치가 올라가 있어서 지방간이 있는 것 같아요.",
        "술은 일주일에 몇 번, 얼마나 드세요?",
        "담배는 하루에 몇 개비 피우세요?",
        "요즘 잠은 잘 주무세요? 피곤하지는 않으시고요?",
        "체중이 늘면 혈압이랑 혈당도 같이 올라가요.",
        "하루에 30분씩 빠르게 걷는 것부터 해 보세요.",
        "짜게 드시지 말고 국물은 조금만 드세요.",
        "몸살 기운이 있으면 수액 맞고 푹 쉬세요.",
        "독감 예방접종은 맞으셨어요?",
        "3개월 뒤에 피검사 다시 해서 비교해 볼게요.",
    ),
    "소아청소년과": (
        "아이가 언제부터 열이 났어요? 몇 도까지 올랐어요?",
        "밥은 잘 먹어요? 소변은 평소만큼 보고요?",
        "기침은 밤에 더 심해요? 가래 끓는 소리도 나요?",
        "아 해 볼까? 목 좀 보자.",
        "목이 많이 부었고 귀에 중이염이 조금 있네요.",
        "해열제는 타이레놀이랑 부루펜을 번갈아 먹여도 돼요.",
        "열이 39도 넘거나 많이 처지면 바로 다시 오세요.",
        "독감 검사가 양성이라 타미플루를 먹여야 해요.",
        "수족구는 입안이랑 손발에 물집이 생겨요.",
        "탈수가 되지 않게 물을 조금씩 자주 먹이세요.",
        "콧물이 많아서 코를 좀 빼 주고 갈게요.",
        "다음 달에 영유아 검진이랑 예방접종 하러 오세요.",
    ),
    "정신건강의학과": (
        "요즘 잠은 몇 시간 정도 주무세요?",
        "잠드는 데 얼마나 걸리고, 밤에 몇 번 깨세요?",
        "기분이 처지고 흥미가 없는 게 2주 넘게 계속됐어요?",
        "지난번 오셨을 때보다 기분은 좀 어떠세요?",
        "식욕은 어떠세요? 체중이 변하지는 않았고요?",
        "불안할 때 가슴이 두근거리거나 숨이 차기도 해요?",
        "약 드시고 나서 불편한 점은 없으셨어요?",
        "처음 며칠은 속이 울렁거리거나 졸릴 수 있어요.",
        "약은 효과가 나기까지 2주에서 4주 정도 걸려요.",
        "약을 갑자기 끊지 말고 꼭 상의하고 줄이세요.",
        "혹시 죽고 싶다는 생각이 든 적도 있으세요?",
        "다음 진료는 2주 뒤로 잡을게요.",
    ),
    "이비인후과": (
        "코막힘이랑 콧물은 언제부터 있었어요?",
        "콧물은 맑아요, 아니면 누런 색이에요?",
        "코 안을 내시경으로 볼게요. 조금 불편할 수 있어요.",
        "귀가 먹먹하거나 소리가 울리지는 않아요?",
        "편도가 많이 부어서 침 삼킬 때 아프실 거예요.",
        "알레르기 비염이라 코 스프레이를 같이 쓸게요.",
        "축농증이 있어서 항생제를 일주일 정도 드셔야 해요.",
        "고막이 빨갛게 부어 있어서 중이염이 맞네요.",
        "청력 검사 결과는 양쪽 다 정상이에요.",
        "이석증 같아서 머리 위치를 바꾸는 치료를 할게요.",
        "생리식염수로 코 세척을 하루 2번 해 주세요.",
        "목을 많이 쓰시면 목소리가 더 쉴 수 있어요.",
    ),
    "정형외과": (
        "어디가 아프세요? 허리예요, 다리예요?",
        "언제부터 아팠고, 다치신 적이 있으세요?",
        "앉아 있을 때랑 걸을 때 중에 언제 더 아프세요?",
        "다리를 들어 올려 볼게요. 저리면 말씀하세요.",
        "엑스레이 찍어 보고 필요하면 MRI도 볼게요.",
        "다리가 저린 걸 보니 허리 디스크일 수 있어요.",
        "소염진통제랑 근육 이완제를 같이 드릴게요.",
        "어깨가 굳는 오십견이라 물리치료를 같이 할게요.",
        "무릎 관절염이라 연골 주사를 맞아 보죠.",
        "체외충격파 치료는 일주일에 한 번씩 받으세요.",
        "골밀도 검사에서 골다공증이 나왔어요.",
        "2주 동안 무리하지 말고 찜질해 주세요.",
    ),
    "피부과": (
        "언제부터 가렵고 붉게 올라왔어요?",
        "새로 바꾼 화장품이나 드신 음식이 있으세요?",
        "아토피 피부염이라 보습제를 하루 2번 이상 바르세요.",
        "스테로이드 연고는 얇게 하루 1번만 바르세요.",
        "두드러기라서 알레르기 약을 아침저녁으로 드세요.",
        "여드름 약 드시는 동안은 임신하시면 안 돼요.",
        "대상포진이라 항바이러스제를 7일 동안 드세요.",
        "발톱 무좀은 약을 3개월 정도 드셔야 해요.",
        "사마귀는 냉동치료를 2주 간격으로 할게요.",
        "기미는 레이저 치료랑 미백 크림을 같이 쓸게요.",
        "자외선 차단제는 외출 30분 전에 바르세요.",
        "긁으면 더 번지니까 손톱은 짧게 깎으세요.",
    ),
    "산부인과": (
        "마지막 생리 시작일이 언제였어요?",
        "생리 주기는 규칙적인 편이세요?",
        "초음파로 아기 심장 소리를 확인해 볼게요.",
        "임신 초기에는 엽산을 꼭 챙겨 드세요.",
        "냉이 많아지고 가려운 건 질염 때문이에요.",
        "질염 약 드시는 동안은 술을 드시면 안 돼요.",
        "자궁경부암 검사 결과는 일주일 뒤에 나와요.",
        "자궁근종이 3cm 정도 보여서 지켜볼게요.",
        "생리통이 심하면 진통제를 미리 드세요.",
        "피임약은 매일 같은 시간에 드세요.",
        "철분제 드시면 변이 검게 나올 수 있어요.",
        "출혈이 많거나 배가 심하게 아프면 바로 오세요.",
    ),
    "비뇨의학과": (
        "소변을 하루에 몇 번 정도 보세요? 밤에도 깨세요?",
        "소변 볼 때 아프거나 피가 섞여 나오지는 않아요?",
        "소변 검사에서 염증 수치가 높게 나왔어요.",
        "방광염이라 항생제를 3일 동안 드세요.",
        "전립선이 커져서 소변 줄기가 약해진 거예요.",
        "이 약은 어지러울 수 있으니 천천히 일어나세요.",
        "초음파로 잔뇨량을 확인해 볼게요.",
        "PSA 수치는 정상 범위예요.",
        "요로결석이 4mm라서 물을 많이 드시면 빠질 수 있어요.",
        "통증이 심하면 진통제를 드시고 바로 오세요.",
        "커피랑 술은 방광을 자극하니까 줄이세요.",
        "3개월 뒤에 소변 검사를 다시 해 볼게요.",
    ),
    "안과": (
        "눈이 언제부터 빨갛고 가려웠어요?",
        "눈곱이 많이 끼거나 눈이 부시지는 않으세요?",
        "시력 검사랑 안압 검사부터 할게요.",
        "알레르기 결막염이라 안약을 하루 4번 넣으세요.",
        "안구건조증이라 인공눈물을 자주 넣으세요.",
        "항생제 안약은 하루 4번, 일주일 넣으세요.",
        "안압이 조금 높아서 녹내장 검사를 해 볼게요.",
        "산동 검사를 하면 몇 시간 동안 눈이 부셔요.",
        "백내장이 진행돼서 수술을 고려해 보죠.",
        "다래끼는 따뜻한 찜질을 하루 3번 해 주세요.",
        "콘택트렌즈는 당분간 끼지 마세요.",
        "망막 사진은 이상 없어요.",
    ),
    "신경과": (
        "두통은 어느 쪽이 어떻게 아프세요?",
        "머리가 아플 때 속이 메스껍거나 빛이 눈부시세요?",
        "편두통이라 아플 때 바로 드시는 약을 드릴게요.",
        "두통이 자주 오면 예방약을 저녁에 드세요.",
        "어지럼증이 계속되면 뇌 MRI를 찍어 볼게요.",
        "손발 저림은 신경전도 검사로 확인해 볼게요.",
        "아스피린은 매일 빠짐없이 드세요.",
        "말이 어눌해지거나 한쪽에 힘이 빠지면 119를 부르세요.",
        "기억력 검사에서 경도 인지장애가 의심돼요.",
        "치매 약은 자기 전에 한 알 드세요.",
        "뇌파 검사는 잠을 조금 덜 자고 오세요.",
        "어지러울 때는 천천히 일어나세요.",
    ),
}
# 예전 이름으로 저장된 진료과 → 지금 이름
RENAMED_SPECIALTIES = {"내과·가정의학과": GENERAL}
SPECIALTIES = tuple(SPECIALTY_PROMPTS)
# 그 의사 사전의 용어로 만드는 문장
TERM_TEMPLATES = ("{} 처방해 드릴게요.", "오늘은 {} 이야기를 해 볼게요.")
GAP_S = 1.2  # 문장별 녹음을 이어 붙일 때 사이에 넣는 무음 (한 번에 전사하고 문장별로 다시 나눈다)
# 이 글자 수 이하로 잘못 들린 말은 흔한 낱말일 수 있어("발에"→"가래", "기온"→"기운") 다른 문장의 같은 말까지 바꾸지 않게
# 바로 뒤(없으면 앞) 낱말과 묶어 교정한다 ("수행 맞고 → 수액 맞고"). 바른 말("수액")은 그대로 Whisper 힌트에 넣는다.
SHORT_WRONG_CHARS = 3
_IGNORED = re.compile(r"[\s.,?!·…'\"“”‘’()\[\]~-]")
_EDGE = ".,?!… "  # 교정 앞뒤에서 떼어 낼 문장부호


def _norm(text: str) -> str:
    """띄어쓰기·문장부호·영문 대소문자를 무시한 비교용 글."""
    return _IGNORED.sub("", text).lower()


def build_prompts(vocabulary: list[str], count: int = len(DEFAULT_PROMPTS), specialty: str = GENERAL) -> list[str]:
    """그 의사 사전의 용어로 만든 문장을 먼저(최대 절반), 나머지는 그 진료과 문장으로 채운다."""
    terms = [t for t in vocabulary if _norm(t)][: count // 2]
    own = [TERM_TEMPLATES[i % len(TERM_TEMPLATES)].format(term) for i, term in enumerate(terms)]
    base = SPECIALTY_PROMPTS.get(specialty, DEFAULT_PROMPTS)
    return (own + [p for p in base if p not in own])[:count]


def exact(expected: str, heard: str) -> bool:
    """띄어쓰기·문장부호를 빼고 같으면 정확히 받아 적은 것으로 본다."""
    return _norm(expected) == _norm(heard)


@dataclass(frozen=True)
class Finding:
    heard: str  # Whisper 가 받아 적은 말
    expected: str  # 화면에 나온 말 (의사가 읽은 말)
    prompt: str  # 그 문장
    term: str = ""  # 용어 목록(Whisper 힌트)에 넣을 바른 말. 비면 expected (짧은 말은 앞뒤 낱말 없이)

    @property
    def hint(self) -> str:
        return self.term or self.expected


def _same_run(words: list[str], target: str, from_end: bool) -> int:
    """words 의 끝(또는 앞)에서 몇 낱말을 이어 붙여야 target 과 같아지는지. 같아지지 않으면 0."""
    joined, count = "", 0
    for word in reversed(words) if from_end else words:
        joined = _norm(word) + joined if from_end else joined + _norm(word)
        count += 1
        if joined == target:
            return count
        if len(joined) >= len(target):
            return 0
    return 0


def _trim_spacing_only_edges(got: list[str], want: list[str]) -> tuple[list[str], list[str]]:
    """바뀐 낱말 묶음의 앞뒤에서 띄어쓰기만 다른 낱말("속쓰릴" / "속 쓰릴")을 떼어 낸다."""
    for from_end in (True, False):
        while got and want:
            edge_got, edge_want = (got[-1], want[-1]) if from_end else (got[0], want[0])
            n_want = _same_run(want, _norm(edge_got), from_end)
            n_got = _same_run(got, _norm(edge_want), from_end) if not n_want else 0
            if n_want:
                got, want = (got[:-1], want[:-n_want]) if from_end else (got[1:], want[n_want:])
            elif n_got:
                got, want = (got[:-n_got], want[:-1]) if from_end else (got[n_got:], want[1:])
            else:
                break
    return got, want


def _trim_shared_ending(heard: str, expected: str) -> tuple[str, str]:
    """'현타객은 → 현탁액은'처럼 끝 글자(조사)가 같으면 떼어 '현타객 → 현탁액'으로 (다른 조사가 붙어도 고쳐지게)."""
    if len(heard) > 2 and len(expected) > 2 and heard[-1] == expected[-1] and not heard[-2].isspace() and not expected[-2].isspace():
        return heard[:-1], expected[:-1]
    return heard, expected


def find_misheard(expected: str, heard: str) -> list[Finding]:
    """화면 문장과 받아 적은 글을 낱말 단위로 맞춰 다르게 들린 부분을 찾는다 (띄어쓰기·문장부호 차이는 무시).

    빠지거나 더 들린 말은 고쳐 쓸 대상이 없어 넣지 않는다.
    """
    want = [w for w in expected.split() if _norm(w)]
    got = [w for w in heard.split() if _norm(w)]
    matcher = difflib.SequenceMatcher(a=[_norm(w) for w in got], b=[_norm(w) for w in want], autojunk=False)
    findings = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        heard_words, read_words = _trim_spacing_only_edges(got[i1:i2], want[j1:j2])
        wrong = " ".join(heard_words).strip(_EDGE)
        right = " ".join(read_words).strip(_EDGE)
        if not wrong or not right or _norm(wrong) == _norm(right):
            continue  # 띄어쓰기만 다르다
        core_wrong, core_right = _trim_shared_ending(wrong, right)
        if len(_norm(core_wrong)) > SHORT_WRONG_CHARS:
            findings.append(Finding(core_wrong, core_right, expected))
            continue
        term = right  # 힌트에는 낱말 그대로 ("식도염": 끝 글자가 같아도 조사가 아닐 수 있다)
        start = j1 + next(k for k in range(j2 - j1) if want[j1 + k : j1 + k + len(read_words)] == read_words)
        end = start + len(read_words)
        if end < len(want):  # 짧은 말은 바로 뒤 낱말과 묶는다 (문장 끝이면 바로 앞 낱말)
            near = want[end].strip(_EDGE)
            wrong, right = f"{wrong} {near}", f"{right} {near}"
        elif start > 0:
            near = want[start - 1].strip(_EDGE)
            wrong, right = f"{near} {wrong}", f"{near} {right}"
        findings.append(Finding(wrong, right, expected, term))
    return findings


def join_clips(clips: list[np.ndarray], gap_s: float = GAP_S) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """문장별 녹음(16 kHz)을 무음을 사이에 두고 이어 붙인다. 각 문장의 (시작, 끝) 시각도 돌려준다."""
    gap = np.zeros(int(gap_s * SAMPLE_RATE), np.int16)
    parts, spans, cursor = [gap], [], gap_s
    for clip in clips:
        clip = clip.astype(np.int16)
        spans.append((cursor, cursor + len(clip) / SAMPLE_RATE))
        cursor += len(clip) / SAMPLE_RATE + gap_s
        parts += [clip, gap]
    return np.concatenate(parts), spans


def split_by_prompts(prompts: list[str], heard: str) -> list[str]:
    """받아 적은 글 전체를 화면 문장들과 글자 단위로 맞춰 문장별로 나눈다.

    Whisper 의 낱말 시각은 쉼 뒤에서 1초 넘게 앞당겨지기도 해서, 시각 대신 정답 문장과의 맞춤으로 나눈다.
    글자 단위로 맞추므로 띄어쓰기가 달라도("확인해볼게요") 바로 옆의 잘못 들린 말이 다른 문장으로 밀리지 않는다.
    받아 적은 낱말마다 그 글자들이 가장 많이 맞춰진 문장으로 보낸다(더 들린 말은 바로 앞 문장으로).
    """
    owner = [index for index, prompt in enumerate(prompts) for _ in _norm(prompt)]  # 정답 글자마다 문장 번호
    if not owner:
        return ["" for _ in prompts]
    got = [w for w in heard.split() if _norm(w)]
    word_of = [i for i, w in enumerate(got) for _ in _norm(w)]  # 받아 적은 글자마다 낱말 번호
    votes: list[dict[int, int]] = [{} for _ in got]
    matcher = difflib.SequenceMatcher(a="".join(_norm(w) for w in got), b="".join(_norm(p) for p in prompts),
                                      autojunk=False)
    for _tag, i1, i2, j1, j2 in matcher.get_opcodes():
        for i in range(i1, i2):
            j = j1 + (i - i1) * (j2 - j1) // (i2 - i1) if j2 > j1 else max(j1 - 1, 0)
            k = owner[min(j, len(owner) - 1)]
            votes[word_of[i]][k] = votes[word_of[i]].get(k, 0) + 1
    texts: list[list[str]] = [[] for _ in prompts]
    last = 0
    for word, vote in zip(got, votes):
        last = max(last, max(vote, key=vote.__getitem__))  # 문장 순서는 거꾸로 가지 않는다
        texts[last].append(word)
    return [" ".join(words) for words in texts]


@dataclass
class TrainingResult:
    prompts: list[str]
    heard_before: list[str]  # 기본 힌트로 받아 적은 글 (학습 전)
    heard_after: list[str]  # 찾은 교정·용어를 넣어 다시 받아 적은 글 (학습 후)
    findings: list[Finding]
    embeddings: list[np.ndarray]  # 목소리 특징값 (화자 인식 모델이 없으면 빈 목록)
    speech_seconds: float
    voice_issue: str | None = None  # 목소리를 등록하지 못한 이유

    @property
    def correct_before(self) -> int:
        return sum(exact(p, h) for p, h in zip(self.prompts, self.heard_before))

    @property
    def correct_after(self) -> int:
        return sum(exact(p, h) for p, h in zip(self.prompts, self.heard_after))


def corrections_from(findings: list[Finding]) -> list[Correction]:
    rules = []
    for finding in findings:
        try:
            wrong, right = check_correction(finding.heard, finding.expected)
        except DictionError:
            continue  # 한 글자짜리 등은 교정으로 쓸 수 없다
        rules.append(Correction(wrong, right))
    return rules


def analyze(prompts: list[str], clips: list[np.ndarray], transcribe: Callable[[np.ndarray, str | None], TranscriptionResult],
            encoder: Callable[[], SpeakerEncoder], base_prompt: str,
            cancel: threading.Event | None = None) -> TrainingResult:
    """따라 읽은 녹음을 분석한다: 학습 전 전사 → 다르게 들린 말 찾기 → 그 교정·용어로 다시 전사(학습 후) → 목소리 특징값."""
    audio, _spans = join_clips(clips)

    def heard(result: TranscriptionResult) -> list[str]:
        return split_by_prompts(prompts, " ".join(seg.text for seg in result.segments))

    before = heard(transcribe(audio, None))
    findings = [f for prompt, said in zip(prompts, before) for f in find_misheard(prompt, said)]

    doctor = "따라 읽기"  # 학습 후 확인용 임시 사전의 이름
    rules = corrections_from(findings)
    trial = DictionBook(Path(), {})
    trial.set_entry(doctor, [f.hint for f in findings], rules)
    after_result = transcribe(audio, build_prompt(base_prompt, trial, doctor) if rules else None)
    for seg in after_result.segments:
        seg.speaker = doctor  # 모두 이 의사가 읽은 말이다
    after_result.segments = apply_corrections(after_result.segments, trial)
    after = heard(after_result)

    speech = speech_only(audio)
    embeddings: list[np.ndarray] = []
    issue = None
    if cancel is None or not cancel.is_set():
        try:
            embeddings = enrollment_embeddings(encoder(), speech)
        except Exception as exc:  # 화자 인식 모델이 없어도 딕션 학습은 쓸 수 있다
            issue = str(exc)
    return TrainingResult(list(prompts), before, after, findings, embeddings, len(speech) / SAMPLE_RATE, issue)
