"""
Ириска — бот-помощник по командам Iris | Чат-менеджер (@iris_cm_bot).

Запуск: python bot.py
Нужные переменные окружения:
    BOT_TOKEN       — токен телеграм-бота (от @BotFather)
    GROQ_API_KEY    — один или несколько ключей Groq, через запятую:
                       "gsk_1,gsk_2,gsk_3".
    GROQ_MODELS     — необязательно, каскад model ID через запятую,
                       от основной модели к более лёгким резервным.
    GROQ_MODEL      — устаревший одиночный параметр; поддерживается.
    MAX_RUNTIME_SEC — необязательно, через сколько секунд бот сам
                      завершится (нужно для GitHub Actions, см. bot.yml)
"""

import asyncio
import hashlib
import logging
import os
import re
import sys
import threading
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQuery,
    InlineQueryResultArticle,
    InputTextMessageContent,
    Message,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("iriska")

BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    raise RuntimeError("Не задан BOT_TOKEN — добавьте токен Telegram-бота в переменные окружения")


def _split_env_list(value: str) -> list[str]:
    """Читает список из env: разделители — запятая, точка с запятой или новая строка."""
    return [item.strip() for item in re.split(r"[,;\n]+", value) if item.strip()]


# --- Ключи и каскад моделей Groq --------------------------------------------
# GROQ_API_KEY принимает несколько ключей через запятую. При превышении
# лимита на конкретной модели бот пробует более лёгкую модель на том же ключе,
# а после исчерпания каскада — следующий ключ.
GROQ_API_KEYS = _split_env_list(os.environ.get("GROQ_API_KEY", ""))
if not GROQ_API_KEYS:
    raise RuntimeError("GROQ_API_KEY пуст или не содержит ни одного валидного ключа")

_DEFAULT_GROQ_MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
_raw_models = os.environ.get("GROQ_MODELS", "").strip()
if _raw_models:
    GROQ_MODELS = list(dict.fromkeys(_split_env_list(_raw_models)))
else:
    # GROQ_MODEL оставлен для совместимости со старой конфигурацией. Для него
    # автоматически добавляем резервную модель, чтобы одиночная переменная
    # тоже получила базовый каскад.
    _legacy_model = os.environ.get("GROQ_MODEL", "").strip()
    GROQ_MODELS = list(dict.fromkeys(
        ([_legacy_model] if _legacy_model else [])
        + [model for model in _DEFAULT_GROQ_MODELS if model != _legacy_model]
    ))
if not GROQ_MODELS:
    raise RuntimeError("GROQ_MODELS пуст: укажите хотя бы одну активную модель Groq")

# Для старого кода/логов оставляем синоним с основной моделью.
GROQ_MODEL = GROQ_MODELS[0]


def _integer_env(name: str, default: int, minimum: int) -> int:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default
    try:
        return max(minimum, int(raw_value))
    except ValueError as exc:
        raise RuntimeError(f"{name} должен быть целым числом") from exc


# 429 кешируется отдельно для пары «ключ + модель», чтобы при следующих
# запросах не стучать в уже исчерпанный лимит и не терять время.
RATE_LIMIT_COOLDOWN_SEC = _integer_env("GROQ_RATE_LIMIT_COOLDOWN_SEC", 60, 1)
GROQ_REQUEST_TIMEOUT_SEC = _integer_env("GROQ_REQUEST_TIMEOUT_SEC", 30, 1)
AI_TOTAL_TIMEOUT_SEC = _integer_env("AI_TOTAL_TIMEOUT_SEC", 60, 5)
INLINE_AI_TIMEOUT_SEC = _integer_env("INLINE_AI_TIMEOUT_SEC", 22, 5)
MAX_RUNTIME_SEC = _integer_env("MAX_RUNTIME_SEC", 0, 0)  # 0 = без лимита

_CURRENT_KEY_LOCK = threading.Lock()
_STATE_LOCK = threading.Lock()
CURRENT_KEY_INDEX = 0
_RATE_LIMIT_UNTIL: dict[tuple[int, str], float] = {}
_UNAVAILABLE_MODELS: set[tuple[int, str]] = set()
_INVALID_KEYS: set[int] = set()


def rotate_key() -> None:
    """Переключает текущий ключ на следующий по кругу."""
    global CURRENT_KEY_INDEX
    with _CURRENT_KEY_LOCK:
        CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(GROQ_API_KEYS)


def _get_current_key_index() -> int:
    with _CURRENT_KEY_LOCK:
        return CURRENT_KEY_INDEX % len(GROQ_API_KEYS)


def _set_current_key_index(index: int) -> None:
    global CURRENT_KEY_INDEX
    with _CURRENT_KEY_LOCK:
        CURRENT_KEY_INDEX = index % len(GROQ_API_KEYS)


def _key_is_invalid(key_index: int) -> bool:
    with _STATE_LOCK:
        return key_index in _INVALID_KEYS


def _disable_key(key_index: int) -> None:
    with _STATE_LOCK:
        _INVALID_KEYS.add(key_index)


def _model_is_unavailable(key_index: int, model: str) -> bool:
    with _STATE_LOCK:
        return (key_index, model) in _UNAVAILABLE_MODELS


def _disable_model_for_key(key_index: int, model: str) -> None:
    with _STATE_LOCK:
        _UNAVAILABLE_MODELS.add((key_index, model))


def _model_cooldown_remaining(key_index: int, model: str) -> float:
    now = time.monotonic()
    with _STATE_LOCK:
        until = _RATE_LIMIT_UNTIL.get((key_index, model), 0.0)
        if until <= now:
            _RATE_LIMIT_UNTIL.pop((key_index, model), None)
            return 0.0
        return until - now


def _mark_model_rate_limited(key_index: int, model: str, cooldown_sec: float) -> None:
    with _STATE_LOCK:
        _RATE_LIMIT_UNTIL[(key_index, model)] = time.monotonic() + max(1.0, cooldown_sec)


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

KNOWLEDGE_BASE_PATH = Path(__file__).with_name("knowledge_base.md")
KNOWLEDGE_BASE_RAW = KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")

BASE_SYSTEM_PROMPT = """Ты — «Ириска», дружелюбный и знающий помощник в Telegram-чатах,
который подробно объясняет пользователям, как пользоваться ботом-модератором
Iris | Чат-менеджер (@iris_cm_bot). Ты не сам Iris и не выполняешь его команды —
ты объясняешь, как устроена нужная функция, и подсказываешь, какую именно
команду и в каком формате написать.

Правила ответа:
- Отвечай по-русски, содержательно, но по делу — не воды ради воды. Структура
  для не самых очевидных функций (триггеры, кланы, дуэли, отношения и т.п.):
  1) коротко, что это за функция и зачем она нужна;
  2) готовая команда(ы) в блоке кода с подставленными по контексту значениями
     (например, если человек написал "замутить на 7 дней" — покажи
     "Мут 7 дней @username");
  3) хотя бы один конкретный пример использования "из жизни" — в какой
     ситуации это пригодится модератору/участнику чата.
  Для простых прямых вопросов ("как забанить", "как удалить сообщение") не
  нужно расписывать теорию — достаточно команды и короткого пояснения.
- Если в вопросе не хватает данных (кого именно, на какой срок) — спроси
  коротко или покажи общий шаблон команды с плейсхолдерами, и всё равно
  объясни, как она работает.
- Если вопрос не связан с Iris (общие вопросы, не про бота) — вежливо
  скажи, что помогаешь только по функциям Iris | Чат-менеджер.
- Ниже приведены только НАИБОЛЕЕ РЕЛЕВАНТНЫЕ разделы базы знаний по этому
  конкретному вопросу (полная база гораздо больше и не помещается в один
  запрос). ЭТО ЕДИНСТВЕННЫЙ источник фактов, которым тебе разрешено
  пользоваться про функции Iris. СТРОГО ЗАПРЕЩЕНО дополнять, додумывать
  или "красиво объяснять" механику, которой нет в тексте ниже — даже если
  предположение звучит правдоподобно (например: "мешок — это как сундук,
  можно что-то найти" — это выдумка, ничего подобного в Iris нет). Если в
  приведённых разделах нет ответа на конкретную деталь вопроса — так и
  скажи: "не уверена насчёт этого" / "в базе нет точной информации" — и
  посоветуй написать в чате "Команды" или посмотреть официальный список:
  teletype.in/@iris_cm/commands. Лучше честно признать, что не знаешь,
  чем один раз соврать.
- Форматирование — используется старый Telegram Markdown, а НЕ обычный
  CommonMark, поэтому:
  * жирный текст — ОДНА звёздочка с каждой стороны: *текст* — а не
    **текст** (двойные звёздочки Telegram не понимает и сломает всё
    сообщение);
  * код/команда — обратные кавычки: `текст`;
  * блок кода — тройные обратные кавычки БЕЗ названия языка после них
    (не ```text и не ```python, а просто ``` на отдельной строке);
  * курсив — _текст_ (одно нижнее подчёркивание с каждой стороны).
  Каждый символ *, _, ` обязательно должен быть закрыт парой в пределах
  одного сообщения. Если не уверена, что разметка выйдет корректной —
  лучше вообще без неё, простым текстом: это надёжнее, чем сломанное
  форматирование.
- Не пиши никаких служебных пометок о своих рассуждениях (никаких "думаю",
  "Thinking", "<think>" и т.п.) — в ответе должен быть только финальный
  текст для пользователя.
"""

# --- Разбиение базы знаний на разделы для RAG-подобного поиска -------------
# Вся knowledge_base.md слишком большая, чтобы отправлять её целиком в
# каждом запросе (упирается в лимиты Groq по размеру запроса/токенов в
# минуту). Поэтому под каждый вопрос выбираем несколько наиболее подходящих
# разделов (по совпадению ключевых слов) и отправляем только их + общие
# правила синтаксиса, которые нужны всегда.

_STOPWORDS = {
    "как", "что", "это", "для", "или", "она", "они", "нее", "него", "мне",
    "мой", "моя", "мои", "его", "нам", "вас", "тебя", "если", "чтобы",
    "можно", "нужно", "пожалуйста", "привет", "ириска", "ирис", "бот",
    "команда", "команду", "команды", "напиши", "подскажи", "помоги",
    "написать", "который", "которая", "которое",
}


def _normalize_words(text: str) -> set[str]:
    words = re.findall(r"[а-яёa-z0-9]+", text.lower())
    result = set()
    for w in words:
        if len(w) <= 2 or w in _STOPWORDS:
            continue
        # Грубое «усечение окончаний»: сравниваем первые 4 буквы слова,
        # чтобы разные словоформы (мут/мута/мутить, биржа/бирже, дуэль/дуэли)
        # всё равно совпадали при простом поиске по пересечению множеств.
        result.add(w[:4] if len(w) > 4 else w)
    return result


def _split_into_sections(markdown_text: str) -> list[tuple[str, str]]:
    """Делит markdown на секции по заголовкам ## (сохраняя интро до первого ##)."""
    parts = re.split(r"(?m)^(## .+)$", markdown_text)
    sections: list[tuple[str, str]] = []
    if parts and parts[0].strip():
        sections.append(("Общие правила синтаксиса", parts[0].strip()))
    for i in range(1, len(parts), 2):
        title = parts[i].lstrip("# ").strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append((title, f"{parts[i]}\n{body}".strip()))
    return sections


KB_SECTIONS = _split_into_sections(KNOWLEDGE_BASE_RAW)
# Первая секция ("Общие правила синтаксиса") нужна почти всегда — держим её
# отдельно и добавляем в каждый запрос вне конкурса по релевантности.
_INTRO_SECTION = KB_SECTIONS[0][1] if KB_SECTIONS else ""
_SEARCHABLE_SECTIONS = KB_SECTIONS[1:] if len(KB_SECTIONS) > 1 else KB_SECTIONS
_SECTION_WORDS = [(title, body, _normalize_words(title + " " + body)) for title, body in _SEARCHABLE_SECTIONS]

MAX_KB_CHARS = 6000  # ограничиваем объём базы, отправляемой в одном запросе
TOP_SECTIONS = 4


def build_knowledge_excerpt(question: str) -> str:
    """Возвращает интро + до TOP_SECTIONS наиболее релевантных вопросу разделов,
    суммарно не длиннее MAX_KB_CHARS символов."""
    q_words = _normalize_words(question)
    scored = []
    if q_words:
        for title, body, words in _SECTION_WORDS:
            overlap = len(q_words & words)
            if overlap:
                scored.append((overlap, title, body))
    scored.sort(key=lambda x: x[0], reverse=True)

    chosen: list[str] = []
    budget = MAX_KB_CHARS - len(_INTRO_SECTION)
    for _, _title, body in scored[:TOP_SECTIONS]:
        if budget <= 0:
            break
        chosen.append(body[:budget])
        budget -= len(body)

    if not chosen:
        # Ничего конкретного не нашли — даём небольшую подсказку модели,
        # чтобы она не выдумывала синтаксис, а предложила официальный список.
        return _INTRO_SECTION

    return _INTRO_SECTION + "\n\n" + "\n\n".join(chosen)


# История диалога по chat_id, чтобы Ириска помнила контекст переписки
# (простое хранение в памяти процесса; сбрасывается при перезапуске бота).
# Ключи в GROQ_API_KEYS могут переключаться посреди разговора — история
# от chat_id никак с этим не связана и никогда не теряется при ротации.
HISTORY: dict[object, list[dict]] = {}
MAX_HISTORY_MESSAGES = 6  # 3 сообщения пользователя + 3 ответа бота

TRIGGER_RE = re.compile(r"^\s*[!./]?\s*ирис(ка)?\b[,:]?\s*(.*)$", re.IGNORECASE | re.DOTALL)

# Ошибки, при которых имеет смысл повторить запрос с ТЕМ ЖЕ ключом
# (временная проблема на стороне сервера Groq, а не с самим ключом).
RETRYABLE_5XX = {500, 502, 503, 504}
MAX_RETRIES = 2  # повторов на один ключ при 5xx / сетевых ошибках

# Модели, которые умеют "думать" перед ответом и по умолчанию норовят
# засунуть эти рассуждения прямо в тело ответа, если явно не попросить
# формат "hidden".
_REASONING_MODEL_HINTS = ("qwen3", "qwen-qwq", "gpt-oss", "deepseek-r1", "r1-distill")


def _is_reasoning_model(model_name: str) -> bool:
    name = model_name.lower()
    return any(hint in name for hint in _REASONING_MODEL_HINTS)


def _reasoning_effort_for(model_name: str) -> str | None:
    """Для FAQ-бота глубокое 'размышление' не нужно и только съедает
    max_tokens (модель может потратить весь лимит на скрытые рассуждения и
    не оставить места на сам ответ — из-за этого Telegram получал пустое
    сообщение). Поэтому по возможности отключаем/урезаем reasoning явно."""
    name = model_name.lower()
    if "qwen3" in name or "qwen-qwq" in name:
        # У моделей семейства Qwen3 (в т.ч. qwen3.8) reasoning_effort
        # принимает только "none" (выключить) или "default" (думать).
        return "none"
    if "gpt-oss" in name:
        # У GPT-OSS нет "none", но можно снизить до "low".
        return "low"
    return None


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def _strip_reasoning(text: str) -> str:
    """Подчищает ответ модели на случай, если рассуждения всё же просочились
    в content (известный баг reasoning_format=hidden у некоторых моделей
    Groq — параметр не всегда отрабатывает на 100%)."""
    return _THINK_BLOCK_RE.sub("", text).strip()


_CODE_FENCE_LANG_RE = re.compile(r"```[ \t]*[a-zA-Z0-9_+-]+[ \t]*\n")
_DOUBLE_ASTERISK_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)


def _sanitize_markdown(text: str) -> str:
    """Модель (несмотря на инструкцию) периодически пишет обычный CommonMark
    вместо старого Telegram-Markdown: **жирный** двумя звёздочками и
    ```language блоки кода с языковым тегом. Старый parse_mode="Markdown" в
    Telegram такое не понимает — сообщение падает с ошибкой и раньше
    целиком уходило в сырой plain-text fallback (пользователь видел кучу
    голых `**`/```` ``` ````). Приводим самые частые случаи к безопасному
    виду сами, не полагаясь только на послушность модели."""
    text = _CODE_FENCE_LANG_RE.sub("```\n", text)
    text = _DOUBLE_ASTERISK_RE.sub(r"*\1*", text)
    return text


def _markdown_is_balanced(text: str) -> bool:
    """Грубая проверка: количество *, _ и ` в сообщении должно быть чётным,
    иначе Telegram откажется парсить разметку. Не идеальная (не учитывает
    экранирование), но отлавливает подавляющее большинство поломок ещё ДО
    отправки — вместо того чтобы узнавать о них по ошибке TelegramBadRequest
    и показывать пользователю уродливый сырой текст."""
    return all(text.count(ch) % 2 == 0 for ch in ("*", "_", "`"))


def _build_payload(model: str, messages: list[dict]) -> dict:
    """Готовит Groq payload с параметрами, подходящими конкретной модели."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.4,
        # max_tokens устарел у Groq; max_completion_tokens работает и для
        # Qwen, и для актуальных моделей GPT-OSS.
        "max_completion_tokens": 1024,
    }
    if model.startswith("groq/compound"):
        # Не используем веб-поиск/код: ответы должны опираться на локальную БЗ.
        payload["compound_custom"] = {"tools": {"enabled_tools": []}}
    if _is_reasoning_model(model):
        payload["reasoning_format"] = "hidden"
        effort = _reasoning_effort_for(model)
        if effort:
            payload["reasoning_effort"] = effort
    return payload


def _duration_to_seconds(value: str) -> float | None:
    """Разбирает Retry-After и Groq reset-значения вроде `2m59.5s`."""
    text = value.strip().lower()
    if not text:
        return None

    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)

    clock_match = re.fullmatch(r"(?:(\d+):)?(\d{1,2}):(\d{2})(?:\.(\d+))?", text)
    if clock_match:
        hours = int(clock_match.group(1) or 0)
        minutes = int(clock_match.group(2))
        seconds = float(clock_match.group(3) + ("." + clock_match.group(4) if clock_match.group(4) else ""))
        return hours * 3600 + minutes * 60 + seconds

    unit_seconds = {
        "ms": 0.001, "msec": 0.001, "msecs": 0.001,
        "millisecond": 0.001, "milliseconds": 0.001,
        "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
        "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
        "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
        "d": 86400, "day": 86400, "days": 86400,
    }
    matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*(milliseconds?|msecs?|ms|seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d)",
        text,
    )
    if matches:
        total = 0.0
        for amount, unit in matches:
            total += float(amount) * unit_seconds[unit]
        return total

    # Retry-After иногда приходит как HTTP-date вместо числа секунд.
    try:
        retry_at = parsedate_to_datetime(value)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def _retry_after_for(resp: httpx.Response) -> float:
    headers = resp.headers
    value = headers.get("retry-after")
    if value:
        parsed = _duration_to_seconds(value)
        if parsed is not None and parsed > 0:
            return min(parsed, 7 * 24 * 60 * 60)

    # В сообщении Groq обычно указан именно тот лимит, который сработал
    # («try again in 1m...»); используем его до общих reset-заголовков.
    body = getattr(resp, "text", "") or ""
    match = re.search(
        r"(?:try again|retry|reset)[^.!]{0,80}?((?:\d+(?:\.\d+)?\s*(?:milliseconds?|msecs?|ms|seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d)\s*)+)",
        body,
        re.IGNORECASE,
    )
    if match:
        parsed = _duration_to_seconds(match.group(1))
        if parsed is not None and parsed > 0:
            return min(parsed, 7 * 24 * 60 * 60)

    # Если API не приложил точное время, берём ближайшее известное окно.
    reset_values = []
    for header_name in ("x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        value = headers.get(header_name)
        parsed = _duration_to_seconds(value) if value else None
        if parsed is not None and parsed > 0:
            reset_values.append(parsed)
    if reset_values:
        return min(min(reset_values), 7 * 24 * 60 * 60)
    return float(RATE_LIMIT_COOLDOWN_SEC)


def _groq_error(resp: httpx.Response) -> tuple[str, str]:
    """Возвращает код и безопасный для логов фрагмент ошибки API."""
    try:
        data = resp.json()
    except (ValueError, TypeError):
        data = {}
    error = data.get("error", {}) if isinstance(data, dict) else {}
    if not isinstance(error, dict):
        error = {}
    code = str(error.get("code") or "")
    message = str(error.get("message") or getattr(resp, "text", "") or "без описания")
    return code, message[:500]


def _is_model_unavailable_error(status_code: int, code: str, message: str) -> bool:
    text = f"{code} {message}".lower()
    if status_code == 404:
        return True
    if status_code == 403:
        return "model" in text and any(marker in text for marker in (
            "not authorized", "not authorised", "not allowed", "access denied",
            "permission", "does not have access",
        ))
    if status_code not in (400, 422):
        return False
    return any(marker in text for marker in (
        "model_not_found", "model_decommissioned", "unknown model", "model not found",
        "model does not exist", "no such model", "model unavailable", "model is not available",
        "invalid model name",
    ))


def _remove_pending_history(history: list[dict], pending: dict) -> None:
    for index in range(len(history) - 1, -1, -1):
        if history[index] is pending:
            del history[index]
            return


def _format_wait(seconds: float) -> str:
    seconds = max(1, int(seconds + 0.999))
    if seconds >= 3600:
        return f"примерно {max(1, (seconds + 3599) // 3600)} ч"
    if seconds >= 60:
        return f"примерно {max(1, (seconds + 59) // 60)} мин"
    return f"примерно {seconds} сек"


async def ask_ai(
    history_key: object,
    question: str,
    *,
    request_timeout: float | None = None,
) -> str:
    """Отвечает через каскад моделей (сильная → резервные) и ключей Groq.

    Превышение лимита запоминается отдельно для пары ключ/модель. Пока лимит
    не сбросился, следующие запросы сразу переходят на следующую ступень —
    не тратят время на повторный заведомо ограниченный запрос.
    """
    history = HISTORY.setdefault(history_key, [])
    pending_user_message = {"role": "user", "content": question}
    history.append(pending_user_message)
    history[:] = history[-MAX_HISTORY_MESSAGES:]

    excerpt = build_knowledge_excerpt(question)
    system_prompt = (
        f"{BASE_SYSTEM_PROMPT}\nРелевантные разделы базы знаний по командам "
        f"Iris:\n---\n{excerpt}\n---\n"
    )
    messages = [{"role": "system", "content": system_prompt}, *history]

    total_keys = len(GROQ_API_KEYS)
    start_key_index = _get_current_key_index()
    saw_429 = False
    saw_5xx = False
    saw_network_error = False
    saw_empty_response = False
    saw_model_error = False
    saw_bad_response = False
    last_status: int | None = None
    last_error_text = ""
    cooldown_waits: list[float] = []
    successful = False

    try:
        async with httpx.AsyncClient(timeout=request_timeout or GROQ_REQUEST_TIMEOUT_SEC) as client:
            for key_offset in range(total_keys):
                key_index = (start_key_index + key_offset) % total_keys
                key_num = key_index + 1
                if _key_is_invalid(key_index):
                    continue

                key_was_rejected = False
                for model in GROQ_MODELS:
                    if key_was_rejected:
                        break
                    if _model_is_unavailable(key_index, model):
                        saw_model_error = True
                        continue

                    remaining = _model_cooldown_remaining(key_index, model)
                    if remaining > 0:
                        saw_429 = True
                        cooldown_waits.append(remaining)
                        log.debug(
                            "Пропускаю модель %s на ключе #%s: лимит сбросится через %s",
                            model, key_num, _format_wait(remaining),
                        )
                        continue

                    payload = _build_payload(model, messages)
                    headers = {"Authorization": f"Bearer {GROQ_API_KEYS[key_index]}"}
                    for attempt in range(MAX_RETRIES + 1):
                        try:
                            resp = await client.post(GROQ_URL, headers=headers, json=payload)
                        except (httpx.TimeoutException, httpx.RequestError) as exc:
                            log.warning(
                                "Сетевая ошибка Groq (ключ #%s/%s, модель %s, попытка %s): %s",
                                key_num, total_keys, model, attempt + 1, exc,
                            )
                            saw_network_error = True
                            last_error_text = str(exc)[:300]
                            if attempt < MAX_RETRIES:
                                await asyncio.sleep(2 * (attempt + 1))
                                continue
                            break

                        if resp.status_code == 200:
                            try:
                                data = resp.json()
                            except (ValueError, TypeError):
                                log.error("Groq вернул некорректный JSON (ключ #%s, модель %s)", key_num, model)
                                saw_bad_response = True
                                break

                            try:
                                choices = data["choices"]
                                raw_answer = choices[0]["message"]["content"]
                            except (KeyError, IndexError, TypeError):
                                log.error("Неожиданный формат ответа Groq (ключ #%s, модель %s)", key_num, model)
                                saw_bad_response = True
                                break

                            clean_answer = _sanitize_markdown(_strip_reasoning(str(raw_answer or "")))
                            if not clean_answer:
                                saw_empty_response = True
                                log.warning(
                                    "Пустой ответ Groq (ключ #%s, модель %s, попытка %s), finish_reason=%s",
                                    key_num, model, attempt + 1,
                                    choices[0].get("finish_reason") if isinstance(choices[0], dict) else None,
                                )
                                if attempt < MAX_RETRIES:
                                    continue
                                break

                            _set_current_key_index(key_index)
                            history.append({"role": "assistant", "content": clean_answer})
                            history[:] = history[-MAX_HISTORY_MESSAGES:]
                            successful = True
                            log.info("Ответ Groq: ключ #%s/%s, модель %s", key_num, total_keys, model)
                            return clean_answer

                        status_code = resp.status_code
                        code, error_text = _groq_error(resp)
                        last_status = status_code
                        last_error_text = error_text
                        log.warning(
                            "Groq error %s (ключ #%s/%s, модель %s, попытка %s): %s",
                            status_code, key_num, total_keys, model, attempt + 1, error_text[:300],
                        )

                        if status_code == 429:
                            saw_429 = True
                            cooldown = _retry_after_for(resp)
                            cooldown_waits.append(cooldown)
                            _mark_model_rate_limited(key_index, model, cooldown)
                            log.info(
                                "Лимит модели %s на ключе #%s; переключаюсь на следующую ступень каскада",
                                model, key_num,
                            )
                            break

                        if status_code in (401, 403):
                            if status_code == 403 and _is_model_unavailable_error(status_code, code, error_text):
                                saw_model_error = True
                                _disable_model_for_key(key_index, model)
                                log.warning(
                                    "Модель %s недоступна для ключа #%s; пробую резервную модель",
                                    model, key_num,
                                )
                                break
                            _disable_key(key_index)
                            key_was_rejected = True
                            break

                        if status_code in RETRYABLE_5XX:
                            saw_5xx = True
                            if attempt < MAX_RETRIES:
                                await asyncio.sleep(2 * (attempt + 1))
                                continue
                            break

                        if status_code in (400, 404, 422):
                            saw_model_error = True
                            if _is_model_unavailable_error(status_code, code, error_text):
                                _disable_model_for_key(key_index, model)
                                log.warning(
                                    "Модель %s недоступна для ключа #%s; отключаю её до перезапуска",
                                    model, key_num,
                                )
                            # Ошибка конкретной модели/параметров не должна
                            # блокировать резервную модель и остальные ключи.
                            break

                        # Остальные ошибки тоже не повторяем с тем же payload;
                        # пробуем следующую модель и, если нужно, другой ключ.
                        break

        # Ни один вариант из каскада не сработал — объясняем причину, не
        # сохраняя безответный вопрос в контексте диалога.
        if all(_key_is_invalid(index) for index in range(total_keys)):
            return "Ошибка авторизации: все указанные API-ключи Groq недействительны или заблокированы."

        if saw_429:
            wait_text = _format_wait(min(cooldown_waits)) if cooldown_waits else "немного"
            return (
                "Сейчас лимит запросов или токенов Groq исчерпан на доступных "
                f"моделях и ключах. Попробуй ещё раз через {wait_text}."
            )
        if saw_network_error:
            return "Сетевая ошибка или таймаут при подключении к Groq. Попробуй ещё раз чуть позже."
        if saw_5xx:
            return "Сервер Groq временно недоступен или перегружен. Попробуй позже."
        if saw_empty_response:
            return "Модель вернула пустой ответ. Попробуй задать вопрос короче и конкретнее."
        if saw_model_error:
            detail = f" Последняя ошибка API: {last_error_text}" if last_error_text else ""
            return (
                "Не удалось использовать настроенные модели Groq. Проверь GROQ_MODELS "
                "и активные model ID в Groq Console." + detail
            )
        if saw_bad_response:
            return "Groq вернул ответ в неожиданном формате. Попробуй ещё раз позже."
        if last_status is not None:
            return f"Ошибка API Groq [{last_status}]: {last_error_text or 'без описания'}"
        return "Не смогла обратиться к ИИ. Попробуй ещё раз чуть позже 🙏"
    finally:
        if not successful:
            _remove_pending_history(history, pending_user_message)


_INLINE_CACHE_TTL_SEC = 90
_INLINE_CACHE_MAX_ITEMS = 256
_INLINE_ANSWER_CACHE: dict[tuple[int, str], tuple[float, str]] = {}
_CODE_BLOCK_RE = re.compile(r"```(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_CODE_LANGUAGE_TAGS = {
    "text", "plain", "txt", "python", "py", "bash", "sh", "json",
    "javascript", "js", "html", "yaml", "yml",
}


def _inline_question(query: str) -> str:
    """Принимает как «ириска вопрос», так и просто «вопрос» после @бота."""
    query = query.strip()
    match = TRIGGER_RE.match(query)
    return match.group(2).strip() if match else query


def extract_suggested_command(answer: str) -> str | None:
    """Достаёт первую команду из кодового блока/inline-кода для отдельного результата."""
    for block in _CODE_BLOCK_RE.findall(answer):
        command = block.strip()
        if not command:
            continue
        lines = command.splitlines()
        if len(lines) > 1 and lines[0].strip().lower() in _CODE_LANGUAGE_TAGS:
            command = "\n".join(lines[1:]).strip()
        if command:
            return command
    for inline_code in _INLINE_CODE_RE.findall(answer):
        command = inline_code.strip()
        if command:
            return command
    return None


def _truncate_telegram_text(text: str, limit: int = 3900) -> str:
    if len(text) <= limit:
        return text
    suffix = "\n\n… (ответ сокращён)"
    keep = max(1, limit - len(suffix))
    return text[:keep].rstrip() + suffix


def _inline_preview(text: str, limit: int = 220) -> str:
    preview = re.sub(r"```", "", text)
    preview = re.sub(r"[*_`]", "", preview)
    preview = re.sub(r"\s+", " ", preview).strip()
    if len(preview) > limit:
        preview = preview[: limit - 1].rstrip() + "…"
    return preview


def _inline_again_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(
            text="💬 Спросить Ириску ещё",
            switch_inline_query_current_chat="ириска ",
        )
    ]])


def _make_inline_results(question: str, answer: str, *, plain_text: bool = False) -> list[InlineQueryResultArticle]:
    digest = hashlib.sha1(f"{question}\0{answer}".encode()).hexdigest()[:16]
    message_text = _truncate_telegram_text(answer)
    parse_mode = None
    if not plain_text and _markdown_is_balanced(message_text):
        parse_mode = ParseMode.MARKDOWN

    results = [InlineQueryResultArticle(
        id=f"answer-{digest}",
        title="🍬 Ответ Ириски",
        description=_inline_preview(answer),
        input_message_content=InputTextMessageContent(
            message_text=message_text,
            parse_mode=parse_mode,
        ),
        reply_markup=_inline_again_keyboard(),
    )]

    command = extract_suggested_command(answer)
    if command:
        command_text = _truncate_telegram_text(command, limit=3500)
        # Не отправляем голую команду: Iris мог бы выполнить её сразу после
        # вставки в чат. В результате она только показывается для копирования.
        copy_text = (
            "Готовая команда для копирования (не отправляй это сообщение как команду):\n\n"
            f"{command_text}\n\nСкопируй команду и отправь её отдельно."
        )
        results.append(InlineQueryResultArticle(
            id=f"command-{digest}",
            title=f"📋 Команда: {_inline_preview(command, 48)}"[:64],
            description="Нажми, чтобы показать готовую команду отдельно",
            input_message_content=InputTextMessageContent(
                message_text=copy_text,
                parse_mode=None,
            ),
            reply_markup=_inline_again_keyboard(),
        ))
    return results


def _make_inline_hint_result() -> InlineQueryResultArticle:
    return InlineQueryResultArticle(
        id="inline-hint",
        title="Спроси Ириску про команду Iris",
        description="Например: «ириска как заключить брак?»",
        input_message_content=InputTextMessageContent(
            message_text=(
                "Напиши вопрос после имени бота, например:\n"
                "@имя_бота ириска как заключить брак?\n\n"
                "Потом выбери «Ответ Ириски» или отдельный результат с командой."
            ),
            parse_mode=None,
        ),
    )


def _get_inline_cached_answer(user_id: int, question: str) -> str | None:
    cache_key = (user_id, re.sub(r"\s+", " ", question).strip().casefold())
    cached = _INLINE_ANSWER_CACHE.get(cache_key)
    if not cached:
        return None
    expires_at, answer = cached
    if expires_at <= time.monotonic():
        _INLINE_ANSWER_CACHE.pop(cache_key, None)
        return None
    return answer


def _cache_inline_answer(user_id: int, question: str, answer: str) -> None:
    cache_key = (user_id, re.sub(r"\s+", " ", question).strip().casefold())
    now = time.monotonic()
    # Подчищаем протухшие записи и ограничиваем кэш по размеру.
    for key, (expires_at, _) in list(_INLINE_ANSWER_CACHE.items()):
        if expires_at <= now:
            _INLINE_ANSWER_CACHE.pop(key, None)
    if len(_INLINE_ANSWER_CACHE) >= _INLINE_CACHE_MAX_ITEMS:
        oldest_key = min(_INLINE_ANSWER_CACHE, key=lambda key: _INLINE_ANSWER_CACHE[key][0])
        _INLINE_ANSWER_CACHE.pop(oldest_key, None)
    _INLINE_ANSWER_CACHE[cache_key] = (now + _INLINE_CACHE_TTL_SEC, answer)


def _is_transient_ai_error(answer: str) -> bool:
    return answer.startswith((
        "Сейчас лимит", "Сетевая ошибка", "Сервер Groq", "Ошибка API Groq",
        "Не удалось использовать настроенные модели", "Не смогла обратиться",
        "Ошибка авторизации", "Модель вернула", "Groq вернул",
        "Что-то пошло не так", "Запрос занял слишком много времени",
    ))


async def main() -> None:
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
    dp = Dispatcher()

    @dp.message(Command("start", "help"))
    async def cmd_start(message: Message) -> None:
        await message.answer(
            "Привет! Я *Ириска* 🍬 — помогаю разобраться с командами бота "
            "*Iris | Чат-менеджер*.\n\n"
            "В *группе* обращайся ко мне с приставкой: "
            "`!ириска помоги замутить человека на 7 дней`.\n"
            "В *личных сообщениях* приставка не нужна — просто пиши вопрос "
            "как есть.\n"
            "Есть и *инлайн-режим*: набери `@имя_бота ириска как заключить брак?` "
            "и выбери результат «Ответ Ириски» или отдельную готовую команду.\n\n"
            "Я знаю про модерацию, баны/муты/варны, ранги, чистку чата, "
            "профиль, экономику ирисок, РП и другие модули Iris."
        )

    @dp.inline_query()
    async def handle_inline_query(inline_query: InlineQuery) -> None:
        question = _inline_question(inline_query.query)
        if len(question.strip()) < 3:
            await inline_query.answer(
                [_make_inline_hint_result()],
                cache_time=1,
                is_personal=True,
            )
            return

        user_id = inline_query.from_user.id
        answer = _get_inline_cached_answer(user_id, question)
        if answer is None:
            try:
                answer = await asyncio.wait_for(
                    ask_ai(
                        f"inline:{user_id}",
                        question,
                        request_timeout=min(GROQ_REQUEST_TIMEOUT_SEC, INLINE_AI_TIMEOUT_SEC),
                    ),
                    timeout=INLINE_AI_TIMEOUT_SEC,
                )
            except TimeoutError:
                answer = "Запрос занял слишком много времени. Попробуй повторить его в личке с ботом."
            except Exception:
                log.exception("Failed to answer inline query")
                answer = "Что-то пошло не так при обращении к ИИ. Попробуй ещё раз."

            if answer and not _is_transient_ai_error(answer):
                _cache_inline_answer(user_id, question, answer)

        results = _make_inline_results(question, answer)
        try:
            # Персональные ответы и нулевой Telegram-кэш: в ответах может быть
            # контекст переписки конкретного пользователя.
            await inline_query.answer(results, cache_time=0, is_personal=True)
        except TelegramBadRequest:
            log.warning("Inline result rejected by Telegram; retrying without Markdown")
            await inline_query.answer(
                _make_inline_results(question, answer, plain_text=True),
                cache_time=0,
                is_personal=True,
            )

    @dp.message()
    async def handle_message(message: Message) -> None:
        if not message.text:
            return

        is_private = message.chat.type == "private"
        match = TRIGGER_RE.match(message.text)

        if is_private:
            # В личных сообщениях обращение "ириска" не обязательно —
            # отвечаем на любой текст. Если человек всё же написал с
            # обращением, вопросом считаем то, что идёт после него.
            question = match.group(2).strip() if match else message.text.strip()
        else:
            # В группах/супергруппах реагируем только на явное обращение.
            if not match:
                return
            question = match.group(2).strip()

        if not question:
            await message.reply("Слушаю! Напиши, с чем помочь по Iris.")
            return

        await message.bot.send_chat_action(message.chat.id, "typing")
        try:
            answer = await asyncio.wait_for(
                ask_ai(message.chat.id, question),
                timeout=AI_TOTAL_TIMEOUT_SEC,
            )
        except TimeoutError:
            answer = "ИИ отвечает слишком долго. Попробуй повторить вопрос чуть позже."
        except Exception:
            log.exception("Failed to get answer from AI")
            answer = "Что-то пошло не так при обращении к ИИ. Попробуй ещё раз."

        if not answer or not answer.strip():
            # Подстраховка: ask_ai не должен возвращать пустую строку, но
            # если это всё же случится — Telegram откажется отправлять
            # пустое сообщение ("message text is empty"), а бот молча
            # свалится. Лучше явно ответить пользователю.
            log.error("ask_ai вернул пустой ответ для вопроса: %r", question)
            answer = "Не получилось сформулировать ответ. Попробуй переформулировать вопрос."

        # Ответ Ириски может содержать "битую" markdown-разметку (например,
        # непарные * или _). Раньше это ловилось только по факту ошибки от
        # Telegram, и в аварийном fallback'е пользователь видел сырые
        # `**`/```` ``` ```` — теперь проверяем баланс символов ЗАРАНЕЕ и,
        # если он не сходится, сразу шлём обычным текстом (без parse_mode),
        # не дожидаясь ошибки. Try/except ниже — просто дополнительная
        # подстраховка на случай более хитрой поломки разметки.
        answer = _truncate_telegram_text(answer, limit=3900)
        use_markdown = _markdown_is_balanced(answer)
        try:
            if use_markdown:
                await message.reply(answer)
            else:
                await message.reply(answer, parse_mode=None)
        except TelegramBadRequest:
            log.warning("Markdown parse failed, retrying as plain text")
            try:
                await message.reply(answer, parse_mode=None)
            except Exception:
                log.exception("Failed to send reply even as plain text")
                await message.reply(
                    "Не смогла корректно отправить ответ (ошибка форматирования). "
                    "Попробуй переформулировать вопрос."
                )
        except Exception:
            log.exception("Failed to send reply")

    log.info(
        "Ириска запущена, каскад Groq: %s, ключей: %s",
        " -> ".join(GROQ_MODELS), len(GROQ_API_KEYS),
    )

    polling_task = asyncio.create_task(dp.start_polling(bot))

    if MAX_RUNTIME_SEC > 0:
        try:
            await asyncio.wait_for(polling_task, timeout=MAX_RUNTIME_SEC)
        except TimeoutError:
            log.info("Достигнут лимит времени работы (%s сек), завершаюсь для рестарта.", MAX_RUNTIME_SEC)
            await dp.stop_polling()
    else:
        await polling_task


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
