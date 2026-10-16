"""音声でコードに質問できるローカルボット。

マイク → Silero VAD → faster-whisper(音声認識) → LLM → Windows標準の日本語音声 → ヘッドホン
LLMは OpenCode Go(既定)か Claude(LLM_PROVIDER=claude)。必要に応じて search_code / read_file で
指定フォルダのコードを調べて答える。音声認識と音声合成はPC内で動き、外に出るのはLLMへのテキストだけ。

    python bot.py C:\\path\\to\\code    起動(ヘッドホン必須)
    python bot.py --check             キー・モデル・LLMの応答速度・音声認識(GPU)・日本語音声を確認
"""

import argparse
import asyncio
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

try:
    load_dotenv(encoding="utf-8")
except UnicodeDecodeError:  # 古いメモ帳はcp932で保存することがある
    load_dotenv(encoding="cp932")

# 回答するLLM。opencode(既定) か claude
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "opencode").strip().lower()
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
OPENCODE_MODEL = os.getenv("OPENCODE_MODEL", "mimo-v2.6-flash")
OPENCODE_BASE_URL = os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/inference/go/v1")
# OpenCode Goは、セッションIDとクライアント名のヘッダを付けないリクエストを400で拒否する
OPENCODE_HEADERS = {"x-opencode-session": uuid.uuid4().hex, "User-Agent": "voice-code-qa/1.0"}
# 思考(reasoning)を切る。切らないと本文が出るまで数秒かかったり、思考で文字数を使い切って無言になったりする
NO_THINKING = {"extra_body": {"thinking": {"type": "disabled"}}}
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")  # 手元のfaster-whisperモデルのフォルダも指定可
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")  # GPUで落ちるときは cpu
TTS_VOICE = os.getenv("TTS_VOICE", "")  # Windowsの音声名の一部。空なら日本語の音声を自動選択
TTS_RATE = int(os.getenv("TTS_RATE", "2"))  # -10(遅い)〜10(速い)

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", "target", ".next", ".idea"}
MAX_FILE_BYTES = 1_000_000
MAX_HITS = 25
MAX_READ_LINES = 150
MAX_TREE_FILES = 400

def _t() -> str:
    """画面に出す時刻(どこで時間がかかったかを見るため)"""
    return time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"


ROOT = Path(".")  # main() で対象フォルダに設定する

SYSTEM_PROMPT = """あなたは、ユーザーとリアルタイムで音声会話をしながら、指定されたコードについての質問に答えるアシスタントです。

話し方のルール(出力はそのまま音声合成されます):
- 日本語で、最大2文、合計80文字程度の短い話し言葉で答える。結論から話す。
- Markdown、箇条書き、コードブロック、URL、記号の羅列は使わない。コードは読み上げず、何をしているかを言葉で説明する。
- ファイル名や関数名は必要なものだけをそのまま言う。行番号は原則言わない。
- 説明が長くなるときは要点だけ話し、「続きを聞きますか」と尋ねる。ユーザーが詳しく求めたときだけ、長く話してよい。

コードの調べ方:
- 質問がコードに関わるときは、推測で答えず、search_code や read_file で確認してから答える。
- ツールを呼ぶ前に、「確認しますね」のような一言だけ先に話す。
- 調べるのはツール呼び出し4回までにする。広い質問は、まず search_code で絞ってから、必要な所だけ100行ほど読む。
- 4回で足りなければ、分かった範囲で答えて、「もっと調べますか」と尋ねる。
- 見つからなければ、見つからなかったと正直に言う。
- 音声認識の誤変換がありうる。固有名詞は、近い名前を検索して確かめる。

コードフォルダのファイル一覧:
"""


# ---------------------------------------------------------------- コード参照ツール

def _iter_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            yield Path(dirpath) / name


def _read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > MAX_FILE_BYTES or b"\0" in data[:2048]:
        return None
    for enc in ("utf-8", "cp932"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace")


def _search(pattern: str, file_glob: str) -> str:
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error:
        rx = re.compile(re.escape(pattern), re.IGNORECASE)
    hits: list[str] = []
    for path in _iter_files():
        if file_glob and not path.match(file_glob):
            continue
        text = _read_text(path)
        if text is None:
            continue
        rel = path.relative_to(ROOT).as_posix()
        for n, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append(f"{rel}:{n}: {line.strip()[:200]}")
                if len(hits) >= MAX_HITS:
                    return "\n".join(hits) + f"\n(上位{MAX_HITS}件で打ち切り。パターンを絞ってください)"
    return "\n".join(hits) or "一致なし"


def _read(path: str, start: int, end: int) -> str:
    p = (ROOT / path).resolve()
    if not (p.is_file() and p.is_relative_to(ROOT)):
        return "ファイルが見つかりません。パスを確認してください。"
    text = _read_text(p)
    if text is None:
        return "このファイルは読めません(バイナリか大きすぎます)。"
    lines = text.splitlines()
    start = max(1, start)
    if start > len(lines):
        return f"{path} は {len(lines)} 行しかありません。"
    end = min(end, start + MAX_READ_LINES - 1, len(lines))
    body = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
    return f"{path} ({start}-{end}行 / 全{len(lines)}行)\n{body}"


def _tree() -> str:
    files = [p.relative_to(ROOT).as_posix() for p in _iter_files()]
    out = "\n".join(files[:MAX_TREE_FILES])
    if len(files) > MAX_TREE_FILES:
        out += f"\n...ほか{len(files) - MAX_TREE_FILES}ファイル"
    return out


# Pipecatは関数名・docstring・型注釈からLLM用のツール定義を自動生成する
async def search_code(params, pattern: str, file_glob: str = ""):
    """コード全体を正規表現で検索し、一致した行を「相対パス:行番号: 内容」の形で返す。

    Args:
        pattern: 検索する正規表現。大文字小文字は区別しない。関数名やキーワードを指定する。
        file_glob: 対象ファイルを絞るglob。例は "*.py"。空なら全ファイルが対象。
    """
    print(f"{_t()}   [調査] search_code {pattern!r} {file_glob}", flush=True)
    await params.result_callback(await asyncio.to_thread(_search, pattern, file_glob))


async def read_file(params, path: str, start: int = 1, end: int = 200):
    """ファイルの指定した行範囲を、行番号つきで読む。一度に読めるのは最大150行。

    Args:
        path: コードフォルダからの相対パス。
        start: 開始行。1から数える。
        end: 終了行。
    """
    print(f"{_t()}   [調査] read_file {path} {start}-{end}", flush=True)
    await params.result_callback(await asyncio.to_thread(_read, path, start, end))


# ---------------------------------------------------------------- 音声認識(faster-whisper)

def _add_cuda_dll_dirs() -> None:
    """Windowsで pip の nvidia-cublas-cu12 / nvidia-cudnn-cu12 のDLLを見つけられるようにする。"""
    if sys.platform != "win32":
        return
    import importlib.util

    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            spec = importlib.util.find_spec(pkg)
        except (ImportError, ValueError):
            continue
        for base in (spec.submodule_search_locations or []) if spec else []:
            d = Path(base) / "bin"
            if d.is_dir():
                os.add_dll_directory(str(d))
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def make_stt():
    """Whisperを読み込み、実際に1回動かして確認する。GPUで失敗したらCPUに切り替える。"""
    import numpy as np
    from pipecat.services.whisper.stt import WhisperSTTService
    from pipecat.transcriptions.language import Language

    _add_cuda_dll_dirs()

    def build(device: str, compute_type: str):
        return WhisperSTTService(
            device=device,
            compute_type=compute_type,
            settings=WhisperSTTService.Settings(model=WHISPER_MODEL, language=Language.JA, no_speech_prob=0.5),
        )

    try:
        stt = build(WHISPER_DEVICE, "default")
        t0 = time.perf_counter()
        segments, _ = stt._model.transcribe(np.zeros(16000, dtype=np.float32), language="ja")
        list(segments)  # 遅延評価なので、消費して初めて推論が走る
        logger.info(f"音声認識: {WHISPER_MODEL} / {WHISPER_DEVICE}(ウォームアップ{time.perf_counter() - t0:.1f}秒)")
        return stt
    except Exception as e:  # noqa: BLE001
        logger.warning(f"音声認識を {WHISPER_DEVICE} で動かせませんでした({type(e).__name__}: {e})。CPUで続行します。遅い場合は WHISPER_MODEL=small を試してください")
        return build("cpu", "int8")


# ---------------------------------------------------------------- 音声合成(Windows標準 SAPI)

class _Sapi:
    """SAPIは作ったスレッドでしか使えないので、専用の1スレッドに閉じ込める。"""

    def __init__(self, voice_hint: str, rate: int):
        self._voice_hint = voice_hint
        self._rate = rate
        self.voice_name = ""
        self._ex = ThreadPoolExecutor(max_workers=1)
        self._ex.submit(self._init).result()  # 以後の合成も同じスレッドで行う。失敗はここで例外になる

    def _init(self):
        import pythoncom
        import win32com.client as wc

        pythoncom.CoInitialize()
        self._wc = wc
        self._sp = wc.Dispatch("SAPI.SpVoice")
        voices = self._sp.GetVoices()
        names = [voices.Item(i).GetDescription() for i in range(voices.Count)]
        hint = self._voice_hint.lower()
        for i, name in enumerate(names):
            low = name.lower()
            if (hint and hint in low) or (not hint and ("japanese" in low or "日本" in name)):
                self._sp.Voice = voices.Item(i)
                self.voice_name = name
                break
        else:
            self.voice_name = self._sp.Voice.GetDescription()
            logger.warning(f"日本語の音声が見つかりません。既定の音声({self.voice_name})を使います。入っている音声: {names}")
        self._sp.Rate = self._rate

    def _synth(self, text: str) -> bytes:
        stream = self._wc.Dispatch("SAPI.SpMemoryStream")
        fmt = self._wc.Dispatch("SAPI.SpAudioFormat")
        fmt.Type = 18  # 16kHz / 16bit / モノラル
        stream.Format = fmt
        self._sp.AudioOutputStream = stream
        self._sp.Speak(text)
        return bytes(stream.GetData())

    async def synth(self, text: str) -> bytes:
        return await asyncio.get_running_loop().run_in_executor(self._ex, self._synth, text)


def make_tts():
    from pipecat.audio.utils import create_stream_resampler
    from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame
    from pipecat.services.settings import TTSSettings
    from pipecat.services.tts_service import TTSService
    from pipecat.transcriptions.language import Language

    try:
        sapi = _Sapi(TTS_VOICE, TTS_RATE)
        logger.info(f"音声合成: {sapi.voice_name}")
    except Exception as e:  # noqa: BLE001
        sapi = None
        logger.warning(f"Windowsの音声合成を使えません({type(e).__name__}: {e})。返答は文字だけで表示します(pip install pywin32)")

    class SapiTTSService(TTSService):
        """返答を画面に出し、使えればWindows標準の音声で読み上げる。"""

        def __init__(self, **kwargs):
            super().__init__(
                push_start_frame=True,
                push_stop_frames=True,
                settings=TTSSettings(model=None, voice=None, language=Language.JA),
                **kwargs,
            )
            self._resampler = create_stream_resampler()
            self._idle_since = 0.0  # 直前の音声を再生し終える見込みの時刻

        def can_generate_metrics(self) -> bool:
            return True

        async def run_tts(self, text: str, context_id: str):
            print(f"{_t()}   ボット: {text}", flush=True)
            if sapi is None:
                return
            try:
                pcm = await sapi.synth(text)
                audio = await self._resampler.resample(pcm, 16000, self.sample_rate)
                # しばらく無音だったあとは、出力機器(特にBluetooth)が起き上がる間に頭が欠けることがある。
                # 先頭に無音を足して、その欠けを無音のほうに吸わせる
                now = time.monotonic()
                if now - self._idle_since > 2.0:
                    audio = bytes(int(self.sample_rate * 0.35) * 2) + audio
                self._idle_since = max(now, self._idle_since) + len(audio) / 2 / self.sample_rate
                yield TTSAudioRawFrame(audio=audio, sample_rate=self.sample_rate, num_channels=1, context_id=context_id)
            except Exception as e:  # noqa: BLE001
                yield ErrorFrame(error=f"音声合成に失敗: {e}")

    return SapiTTSService()


# ---------------------------------------------------------------- LLM

def make_llm():
    """回答するLLMのサービスを作る。OpenCode Go(OpenAI互換)か、Claude(Anthropic API)。"""
    system = SYSTEM_PROMPT + _tree()
    if LLM_PROVIDER == "claude":
        from pipecat.services.anthropic.llm import AnthropicLLMService

        return AnthropicLLMService(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            settings=AnthropicLLMService.Settings(
                model=CLAUDE_MODEL,
                system_instruction=system,
                temperature=0.3,
                max_tokens=400,
                enable_prompt_caching=True,  # ファイル一覧つきの長い指示を毎回送り直さない
            ),
        )

    from pipecat.services.openai.llm import OpenAILLMService

    return OpenAILLMService(
        api_key=os.environ["OPENCODE_API_KEY"],
        base_url=OPENCODE_BASE_URL,
        default_headers=OPENCODE_HEADERS,
        settings=OpenAILLMService.Settings(
            model=OPENCODE_MODEL,
            system_instruction=system,
            temperature=0.3,
            max_tokens=400,
            extra=NO_THINKING,
        ),
    )


# ---------------------------------------------------------------- 接続確認

def _llm_model() -> str:
    return CLAUDE_MODEL if LLM_PROVIDER == "claude" else OPENCODE_MODEL


async def _check_llm(client=None) -> bool:
    """LLMのキー・モデル名・応答速度を確認する。問題があればFalse。"""
    fmt = lambda v: "-" if v is None else f"{v:.2f}s"  # noqa: E731
    model = _llm_model()

    if LLM_PROVIDER == "claude":
        from anthropic import AsyncAnthropic

        client = client or AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        ids = sorted([m.id async for m in client.models.list()])
        print("利用できるモデル:", ", ".join(ids))
        if model not in ids:
            print(f"!! CLAUDE_MODEL={model} は一覧にありません。.env を直してください。")
            return False
        print(f"\n{model} の応答速度(3回。最初の出力 / 最初の本文までの秒数):")
        for _ in range(3):
            t0 = time.perf_counter()
            first_any = first_text = None
            stream = await client.messages.create(
                model=model, max_tokens=60, stream=True,
                messages=[{"role": "user", "content": "一言で挨拶して。"}],
            )
            async for ev in stream:
                now = time.perf_counter() - t0
                first_any = first_any or now
                if ev.type == "content_block_delta" and getattr(ev.delta, "text", None):
                    first_text = now
                    break
            print(f"  {fmt(first_any)} / {fmt(first_text)}")
        return True

    from openai import AsyncOpenAI

    client = client or AsyncOpenAI(
        api_key=os.environ["OPENCODE_API_KEY"], base_url=OPENCODE_BASE_URL, default_headers=OPENCODE_HEADERS
    )
    ids = sorted([m.id async for m in await client.models.list()])
    print("利用できるモデル:", ", ".join(ids))
    if model not in ids:
        print(f"!! OPENCODE_MODEL={model} は一覧にありません。.env を直してください。")
        return False
    print(f"\n{model} の応答速度(3回。最初の出力 / 最初の本文までの秒数):")
    for _ in range(3):
        t0 = time.perf_counter()
        first_any = first_text = None
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "一言で挨拶して。"}],
            max_tokens=60,
            stream=True,
            **NO_THINKING,
        )
        async for chunk in stream:
            now = time.perf_counter() - t0
            first_any = first_any or now
            if chunk.choices and chunk.choices[0].delta.content:
                first_text = now
                break
        print(f"  {fmt(first_any)} / {fmt(first_text)}")
    return True


async def check(client=None) -> None:
    """LLMのキー・モデル一覧・応答速度、音声認識、日本語音声を確認する。"""
    if not await _check_llm(client):
        return
    print("本文までが1秒を超えるなら、より軽いモデルに変えるのが近道です。\n")

    print(f"音声認識を読み込み中: {WHISPER_MODEL}(初回だけモデルのダウンロードあり)")
    make_stt()
    print("\n日本語の音声合成を確認中:")
    make_tts()
    print("\n問題なければ python bot.py <コードのフォルダ> で起動できます。")


# ---------------------------------------------------------------- 音声ボット

async def run_bot() -> None:
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.frames.frames import TTSSpeakFrame
    from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
    from pipecat.workers.runner import WorkerRunner

    stt = make_stt()
    tts = make_tts()
    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )
    llm = make_llm()

    context = LLMContext(tools=[search_code, read_file])
    # 発話の検出はSilero VAD、話し終わりの判定はPipecat同梱のSmart Turn(どちらもローカル)
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
    )

    @user_agg.event_handler("on_user_turn_started")
    async def _show_start(aggregator, strategy):
        print(f"{_t()} (話し始めを検出)", flush=True)

    @user_agg.event_handler("on_user_turn_stopped")
    async def _show_user_text(aggregator, strategy, message):
        if message.content:
            print(f"{_t()} あなた: {message.content}", flush=True)
        # 前の質問で読んだコードは、LLMへの入力を重くして返事を遅くするので、頭の部分だけ残す
        for m in context.messages:
            if isinstance(m, dict) and m.get("role") == "tool" and isinstance(m.get("content"), str) and len(m["content"]) > 400:
                m["content"] = m["content"][:400] + "\n...(古い結果のため省略)"

    pipeline = Pipeline(
        [transport.input(), stt, user_agg, llm, tts, transport.output(), assistant_agg]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True),
        observers=[MetricsLogObserver()],
        idle_timeout_secs=None,  # 無言が続いても終了しない
    )

    @worker.event_handler("on_pipeline_started")
    async def _greet(worker, frame):
        await worker.queue_frames([TTSSpeakFrame("準備できました。コードについて、何でも聞いてください。")])

    logger.info(f"コードフォルダ: {ROOT} / LLM: {LLM_PROVIDER} {_llm_model()}。Ctrl+Cで終了")
    runner = WorkerRunner()
    await runner.add_workers(worker)
    await runner.run()


def main() -> None:
    ap = argparse.ArgumentParser(description="音声でコードに質問できるローカルボット")
    ap.add_argument("code_dir", nargs="?", help="質問したいコードのフォルダ")
    ap.add_argument("--check", action="store_true", help="キー・モデル・音声認識・日本語音声を確認して終了")
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))

    if LLM_PROVIDER not in ("opencode", "claude"):
        sys.exit(f"LLM_PROVIDER は opencode か claude を指定してください(現在: {LLM_PROVIDER})")
    key_name = "ANTHROPIC_API_KEY" if LLM_PROVIDER == "claude" else "OPENCODE_API_KEY"
    if not os.getenv(key_name):
        sys.exit(f".env に {key_name} を設定してください(.env.example を参照)")

    if args.check:
        asyncio.run(check())
        return
    if not args.code_dir or not Path(args.code_dir).is_dir():
        sys.exit("コードのフォルダを指定してください: python bot.py C:\\path\\to\\code")

    global ROOT
    ROOT = Path(args.code_dir).resolve()
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
