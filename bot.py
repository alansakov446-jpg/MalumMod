"""
Ириска — бот-помощник по командам Iris | Чат-менеджер (@iris_cm_bot).

Запуск: python bot.py
Нужные переменные окружения:
    BOT_TOKEN       — токен телеграм-бота (от @BotFather)
    GROQ_API_KEY    — один или несколько бесплатных ключей Groq
                       (console.groq.com/keys), через запятую:
                       "gsk_1,gsk_2,gsk_3". Бот сам переключается на
                       следующий ключ, если текущий упёрся в лимит или
                       оказался невалидным.
    GROQ_MODEL      — необязательно, по умолчанию qwen/qwen3.6-27b
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

# --- Ротация нескольких ключей Groq -----------------------------------------
# GROQ_API_KEY может содержать сразу несколько ключей через запятую. Это
# позволяет пережить лимиты бесплатного тарифа: если один ключ упирается в
# 429 (rate limit) или оказывается невалидным (401/403), бот автоматически
# переключается на следующий ключ из списка и повторяет запрос.
_raw_keys = os.environ["GROQ_API_KEY"]
GROQ_API_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]
if not GROQ_API_KEYS:
    raise RuntimeError("GROQ_API_KEY пуст или не содержит ни одного валидного ключа")

CURRENT_KEY_INDEX = 0


def rotate_key() -> None:
    """Переключает глобальный указатель на следующий ключ Groq по кругу."""
    global CURRENT_KEY_INDEX
    CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(GROQ_API_KEYS)


GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.6-27b")
MAX_RUNTIME_SEC = int(os.environ.get("MAX_RUNTIME_SEC", "0"))  # 0 = без лимита

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
  запрос). Если среди них нет точного ответа — НЕ выдумывай синтаксис.
  Скажи, что не уверена в точной команде, и посоветуй написать в чате
  "Команды" или заглянуть в официальный список: teletype.in/@iris_cm/commands
- Форматирование используй по минимуму и аккуратно: Telegram понимает
  *жирный*, _курсив_ и `код`, но каждый символ *, _ и ` обязательно должен
  быть закрыт парой. Если не уверена, что разметка получится корректной —
  лучше вообще без неё, простым текстом.
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
# (простое хранение в памяти процесса; сбрасывается при перезапуске бота).
# Ключи в GROQ_API_KEYS могут переключаться посреди разговора — история
# от chat_id никак с этим не связана и никогда не теряется при ротации.
HISTORY: dict[int, list[dict]] = {}
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
        # У моделей семейства Qwen3 (в т.ч. qwen3.6) reasoning_effort
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
        "max_tokens": 1024,
    }
    if GROQ_MODEL.startswith("groq/compound"):
        # Все ответы должны идти строго из базы знаний в системном промпте —
        # отключаем встроенные инструменты (веб-поиск, код), чтобы Ириска не
        # уходила гуглить и не тратила лишние токены/время на каждый вопрос.
        payload["compound_custom"] = {"tools": {"enabled_tools": []}}
    if _is_reasoning_model(GROQ_MODEL):
        # Некоторые модели (Qwen3, GPT-OSS, DeepSeek-R1 и т.п.) по умолчанию
        # пишут в content ещё и цепочку своих рассуждений ("<think>...</think>",
        # "Thinking Process" и т.д.) — без этого параметра пользователь в
        # Telegram получал бы этот внутренний монолог вместо чистого ответа.
        payload["reasoning_format"] = "hidden"
        effort = _reasoning_effort_for(GROQ_MODEL)
        if effort:
            # Дополнительно снижаем/выключаем сами рассуждения: иначе модель
            # может потратить весь max_tokens на скрытые "мысли" и не
            # оставить места на сам ответ (пустое сообщение в Telegram).
            payload["reasoning_effort"] = effort

    total_keys = len(GROQ_API_KEYS)
    keys_tried = 0

    # Диагностика: какие типы проблем встретились по ходу перебора ключей —
    # нужна, чтобы в конце дать пользователю точную и полезную причину.
    seen_statuses: list[int] = []
    saw_429 = False
    saw_401_403 = False
    saw_5xx = False
    saw_network_error = False
    saw_empty_response = False
    last_status: int | None = None
    last_error_text: str = ""

    async with httpx.AsyncClient(timeout=30) as client:
        while keys_tried < total_keys:
            key = GROQ_API_KEYS[CURRENT_KEY_INDEX]
            key_num = CURRENT_KEY_INDEX + 1
            headers = {"Authorization": f"Bearer {key}"}

            for attempt in range(MAX_RETRIES + 1):
                try:
                    resp = await client.post(GROQ_URL, headers=headers, json=payload)
                except (httpx.TimeoutException, httpx.RequestError) as exc:
                    log.error(
                        "Сетевая ошибка при обращении к Groq (ключ #%s/%s, попытка %s): %s",
                        key_num, total_keys, attempt + 1, exc,
                    )
                    saw_network_error = True
                    last_error_text = str(exc)
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    # исчерпали попытки на этом ключе из-за сети — переходим
                    # к следующему ключу (вдруг дело было не в сети, а в проблеме
                    # именно с этим соединением/ключом)
                    break

                if resp.status_code == 200:
                    data = resp.json()
                    try:
                        answer = data["choices"][0]["message"]["content"]
                    except (KeyError, IndexError):
                        log.error("Unexpected Groq response: %s", data)
                        history.pop()
                        return "Хм, не получилось разобрать ответ ИИ. Попробуй переформулировать вопрос."

                    clean_answer = _strip_reasoning(answer)
                    if not clean_answer:
                        # Модель вернула 200, но пустой (после очистки от
                        # рассуждений) ответ — обычно значит, что весь
                        # max_tokens ушёл на скрытые размышления и на сам
                        # текст не осталось места. Отправлять пустое
                        # сообщение нельзя (Telegram ответит ошибкой), так
                        # что считаем это неудачной попыткой и пробуем ещё раз.
                        saw_empty_response = True
                        log.warning(
                            "Groq вернул пустой ответ после очистки от рассуждений "
                            "(ключ #%s/%s, попытка %s), finish_reason=%s",
                            key_num, total_keys, attempt + 1,
                            data.get("choices", [{}])[0].get("finish_reason"),
                        )
                        if attempt < MAX_RETRIES:
                            continue
                        break

                    history.append({"role": "assistant", "content": clean_answer})
                    history[:] = history[-MAX_HISTORY_MESSAGES:]
                    return clean_answer

                # Не 200 — логируем и разбираемся, что делать дальше.
                body_preview = resp.text[:500]
                log.error(
                    "Groq error %s (ключ #%s/%s, попытка %s): %s",
                    resp.status_code, key_num, total_keys, attempt + 1, body_preview,
                )
                seen_statuses.append(resp.status_code)
                last_status = resp.status_code
                last_error_text = resp.text[:300]

                if resp.status_code == 429:
                    saw_429 = True
                    # На лимите нет смысла долбить тот же ключ повторно —
                    # сразу переключаемся на следующий.
                    break

                if resp.status_code in (401, 403):
                    saw_401_403 = True
                    # Ключ невалиден/заблокирован — повторять запрос с ним
                    # бессмысленно, сразу пробуем следующий ключ.
                    break

                if resp.status_code in RETRYABLE_5XX:
                    saw_5xx = True
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    break

                # Любая другая ошибка — тоже не повторяем на этом ключе.
                break

            keys_tried += 1
            if keys_tried < total_keys:
                rotate_key()
                log.warning(
                    "Переключаюсь на следующий ключ Groq (#%s из %s)",
                    CURRENT_KEY_INDEX + 1, total_keys,
                )

    # Прошли по всем доступным ключам и ни один не сработал.
    history.pop()  # не сохраняем вопрос, на который не получили ответ

    only_429 = bool(seen_statuses) and all(s == 429 for s in seen_statuses) and not saw_network_error
    only_auth = bool(seen_statuses) and all(s in (401, 403) for s in seen_statuses) and not saw_network_error

    if only_429:
        return (
            "Все доступные API-ключи исчерпали лимит токенов (Rate Limit). "
            "Подождите 2-3 минуты."
        )
    if only_auth:
        return "Ошибка авторизации: указанные API-ключи Groq недействительны или заблокированы."
    if saw_5xx:
        return "Сервер ИИ (Groq) временно недоступен или перегружен. Попробуйте позже."
    if saw_network_error:
        return "Сетевая ошибка или таймаут при подключении к ИИ. Попробуйте ещё раз через пару секунд."
    if saw_empty_response:
        return (
            "Модель не успела дописать ответ (ушло слишком много токенов на "
            "размышления). Попробуй задать вопрос короче и конкретнее."
        )
    if last_status is not None:
        return f"Ошибка API Groq [{last_status}]: {last_error_text or 'без описания'}"
    return "Не смогла обратиться к ИИ. Попробуй ещё раз чуть позже 🙏"


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

        if not answer or not answer.strip():
            # Подстраховка: ask_ai не должен возвращать пустую строку, но
            # если это всё же случится — Telegram откажется отправлять
            # пустое сообщение ("message text is empty"), а бот молча
            # свалится. Лучше явно ответить пользователю.
            log.error("ask_ai вернул пустой ответ для вопроса: %r", question)
            answer = "Не получилось сформулировать ответ. Попробуй переформулировать вопрос."

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

    log.info(
        "Ириска запущена, модель: %s, ключей Groq: %s",
        GROQ_MODEL, len(GROQ_API_KEYS),
    )

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
