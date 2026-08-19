"""
Ириска — бот-помощник по командам Iris | Чат-менеджер (@iris_cm_bot).

Запуск: python bot.py
Нужные переменные окружения:
    BOT_TOKEN       — токен телеграм-бота (от @BotFather)
    GROQ_API_KEY    — бесплатный ключ Groq (console.groq.com/keys)
    GROQ_MODEL      — необязательно, по умолчанию openai/gpt-oss-120b
    MAX_RUNTIME_SEC — необязательно, через сколько секунд бот сам
                      завершится (нужно для GitHub Actions, см. bot.yml)
"""

import asyncio
import logging
import os
import re
import sys
from pathlib import Path

import httpx
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import Message

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("iriska")

BOT_TOKEN = os.environ["BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_RUNTIME_SEC = int(os.environ.get("MAX_RUNTIME_SEC", "0"))  # 0 = без лимита

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

KNOWLEDGE_BASE_PATH = Path(__file__).with_name("knowledge_base.md")
KNOWLEDGE_BASE_RAW = KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")

BASE_SYSTEM_PROMPT = """Ты — «Ириска», дружелюбный помощник в Telegram-чатах, который
объясняет пользователям, как пользоваться ботом-модератором Iris | Чат-менеджер
(@iris_cm_bot). Ты не сам Iris и не выполняешь его команды — ты только
подсказываешь, какую именно команду и в каком формате нужно написать.

Правила ответа:
- Отвечай по-русски, кратко и по делу (обычно 2-6 предложений или список).
- Всегда давай готовую команду Iris в блоке кода или явно выделенной строкой,
  с подставленными по контексту значениями, если это уместно (например,
  если человек написал "замутить на 7 дней" — покажи "Мут 7 дней @username").
- Если в вопросе не хватает данных (кого именно, на какой срок) — спроси
  коротко или покажи общий шаблон команды с плейсхолдерами.
- Если вопрос не связан с Iris (общие вопросы, не про бота) — вежливо
  скажи, что помогаешь только по функциям Iris | Чат-менеджер.
- Ниже приведены только НАИБОЛЕЕ РЕЛЕВАНТНЫЕ разделы базы знаний по этому
  конкретному вопросу (полная база гораздо больше и не помещается в один
  запрос). Если среди них нет точного ответа — НЕ выдумывай синтаксис.
  Скажи, что не уверена в точной команде, и посоветуй написать в чате
  "Команды" или заглянуть в официальный список: teletype.in/@iris_cm/commands
- Форматирование используй по минимуму и аккуратно: Telegram понимает
  *жирный*, _курсив_ и `код`, но каждый символ *, _ и ` обязательно должен
  быть закрыт парой. Если не уверена, что разметка получится корректной —
  лучше вообще без неё, простым текстом.
"""

# --- Разбиение базы знаний на разделы для RAG-подобного поиска -------------
# Вся knowledge_base.md слишком большая, чтобы отправлять её целиком в
# каждом запросе (упирается в лимиты Groq по размеру запроса/токенов в
# минуту). Поэтому под каждый вопрос выбираем несколько наиболее подходящих
# разделов (по совпадению ключевых слов) и отправляем только их + общие
# правила синтаксиса, которые нужны всегда.

_STOPWORDS = {
    "как", "что", "это", "для", "или", "она", "они", "нее", "него", "мне",
    "мой", "моя", "мои", "его", "мне", "нам", "вас", "тебя", "если", "чтобы",
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
# (простое хранение в памяти процесса; сбрасывается при перезапуске бота)
HISTORY: dict[int, list[dict]] = {}
MAX_HISTORY_MESSAGES = 12

TRIGGER_RE = re.compile(r"^\s*[!./]?\s*ирис(ка)?\b[,:]?\s*(.*)$", re.IGNORECASE | re.DOTALL)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 2


async def ask_ai(chat_id: int, question: str) -> str:
    history = HISTORY.setdefault(chat_id, [])
    history.append({"role": "user", "content": question})
    history[:] = history[-MAX_HISTORY_MESSAGES:]

    # Под каждый вопрос собираем компактную выжимку базы знаний (см.
    # build_knowledge_excerpt) вместо того, чтобы слать её целиком — так
    # запрос остаётся маленьким независимо от того, насколько выросла
    # knowledge_base.md.
    excerpt = build_knowledge_excerpt(question)
    system_prompt = (
        f"{BASE_SYSTEM_PROMPT}\nРелевантные разделы базы знаний по командам "
        f"Iris:\n---\n{excerpt}\n---\n"
    )

    messages = [{"role": "system", "content": system_prompt}, *history]
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 700,
    }
    if GROQ_MODEL.startswith("groq/compound"):
        # Все ответы должны идти строго из базы знаний в системном промпте —
        # отключаем встроенные инструменты (веб-поиск, код), чтобы Ириска не
        # уходила гуглить и не тратила лишние токены/время на каждый вопрос.
        payload["compound_custom"] = {"tools": {"enabled_tools": []}}
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

    resp = None
    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(MAX_RETRIES + 1):
            resp = await client.post(GROQ_URL, headers=headers, json=payload)
            if resp.status_code == 200:
                break
            log.error("Groq error %s (попытка %s): %s", resp.status_code, attempt + 1, resp.text[:500])
            if resp.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
                await asyncio.sleep(2 * (attempt + 1))
                continue
            break

    if resp is None or resp.status_code != 200:
        history.pop()  # не сохраняем вопрос, на который не получили ответ
        status = resp.status_code if resp is not None else None
        if status == 429:
            return (
                "Сейчас превышен лимит запросов к ИИ (бесплатный тариф Groq). "
                "Подожди минуту-две и попробуй снова 🙏"
            )
        if status in (401, 403):
            return "Проблема с ключом ИИ (GROQ_API_KEY) — проверь, что он верный и активен."
        return "Не смогла обратиться к ИИ (ошибка API). Попробуй ещё раз чуть позже 🙏"

    data = resp.json()
    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        log.error("Unexpected Groq response: %s", data)
        history.pop()
        return "Хм, не получилось разобрать ответ ИИ. Попробуй переформулировать вопрос."

    history.append({"role": "assistant", "content": answer})
    history[:] = history[-MAX_HISTORY_MESSAGES:]
    return answer.strip()


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
            "как есть.\n\n"
            "Я знаю про модерацию, баны/муты/варны, ранги, чистку чата, "
            "профиль, экономику ирисок, РП и другие модули Iris."
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
            answer = await ask_ai(message.chat.id, question)
        except Exception:
            log.exception("Failed to get answer from AI")
            answer = "Что-то пошло не так при обращении к ИИ. Попробуй ещё раз."

        # Ответ Ириски может содержать "битую" markdown-разметку (например,
        # непарные * или _), из-за которой Telegram откажется отправлять
        # сообщение. Раньше это приводило к полному молчанию бота — теперь
        # при такой ошибке пробуем отправить тот же текст без разметки.
        try:
            await message.reply(answer)
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

    log.info("Ириска запущена, модель: %s", GROQ_MODEL)

    polling_task = asyncio.create_task(dp.start_polling(bot))

    if MAX_RUNTIME_SEC > 0:
        try:
            await asyncio.wait_for(polling_task, timeout=MAX_RUNTIME_SEC)
        except asyncio.TimeoutError:
            log.info("Достигнут лимит времени работы (%s сек), завершаюсь для рестарта.", MAX_RUNTIME_SEC)
            await dp.stop_polling()
    else:
        await polling_task


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
