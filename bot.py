"""
Ириска — бот-помощник по командам Iris | Чат-менеджер (@iris_cm_bot).

Запуск: python bot.py
Нужные переменные окружения:
    BOT_TOKEN       — токен телеграм-бота (от @BotFather)
    GROQ_API_KEY    — бесплатный ключ Groq (console.groq.com/keys)
    GROQ_MODEL      — необязательно, по умолчанию groq/compound-mini
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
GROQ_MODEL = os.environ.get("GROQ_MODEL", "groq/compound-mini")
MAX_RUNTIME_SEC = int(os.environ.get("MAX_RUNTIME_SEC", "0"))  # 0 = без лимита

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

KNOWLEDGE_BASE = Path(__file__).with_name("knowledge_base.md").read_text(encoding="utf-8")

SYSTEM_PROMPT = f"""Ты — «Ириска», дружелюбный помощник в Telegram-чатах, который
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
- Если в базе знаний ниже нет точного ответа — не выдумывай синтаксис.
  Скажи, что не уверена в точной команде, и посоветуй написать в чате
  "Команды" или заглянуть в официальный список: teletype.in/@iris_cm/commands
- Форматирование используй по минимуму и аккуратно: Telegram понимает
  *жирный*, _курсив_ и `код`, но каждый символ *, _ и ` обязательно должен
  быть закрыт парой. Если не уверена, что разметка получится корректной —
  лучше вообще без неё, простым текстом.

База знаний по командам Iris:
---
{KNOWLEDGE_BASE}
---
"""

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

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
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
