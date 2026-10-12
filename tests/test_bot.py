import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("BOT_TOKEN", "123456:TEST")
os.environ.setdefault("GROQ_API_KEY", "gsk-test")

import bot


class FakeResponse:
    def __init__(self, status_code, data=None, headers=None):
        self.status_code = status_code
        self._data = data or {}
        self.headers = headers or {}
        self.text = json.dumps(self._data)

    def json(self):
        return self._data


class FakeAsyncClient:
    def __init__(self, responder, calls):
        self.responder = responder
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def post(self, url, *, headers, json):
        call = (headers["Authorization"].removeprefix("Bearer "), json["model"])
        self.calls.append(call)
        return self.responder(*call)


def success_response(text="Готово"):
    return FakeResponse(200, {"choices": [{"message": {"content": text}}]})


class BotHelpersTests(unittest.TestCase):
    def test_inline_query_strips_optional_trigger(self):
        self.assertEqual(bot._inline_question("Ириска, а как заключить брак?"), "а как заключить брак?")
        self.assertEqual(bot._inline_question("как заключить брак?"), "как заключить брак?")

    def test_command_result_is_safe_to_insert_into_chat(self):
        answer = "Предложи брак так:\n```\nБрак @username\n```"
        results = bot._make_inline_results("как вступить в брак", answer)
        self.assertEqual(len(results), 2)
        command_message = results[1].input_message_content.message_text
        self.assertIn("Брак @username", command_message)
        self.assertIn("не отправляй это сообщение как команду", command_message)
        self.assertEqual(results[1].input_message_content.parse_mode, None)

    def test_rate_limit_duration_parses_groq_reset_header(self):
        response = SimpleNamespace(headers={"x-ratelimit-reset-tokens": "2m59.5s"}, text="")
        self.assertAlmostEqual(bot._retry_after_for(response), 179.5)

    def test_explicit_retry_time_in_error_body_beats_generic_reset_header(self):
        response = SimpleNamespace(
            headers={"x-ratelimit-reset-tokens": "2s"},
            text="Rate limit reached. Please try again in 1m30s.",
        )
        self.assertAlmostEqual(bot._retry_after_for(response), 90)


class GroqCascadeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_keys = bot.GROQ_API_KEYS[:]
        self.original_models = bot.GROQ_MODELS[:]
        self.original_key_index = bot.CURRENT_KEY_INDEX
        bot.GROQ_API_KEYS[:] = ["key-1"]
        bot.GROQ_MODELS[:] = ["model-strong", "model-lite"]
        bot.CURRENT_KEY_INDEX = 0
        bot._RATE_LIMIT_UNTIL.clear()
        bot._UNAVAILABLE_MODELS.clear()
        bot._INVALID_KEYS.clear()
        bot.HISTORY.clear()

    def tearDown(self):
        bot.GROQ_API_KEYS[:] = self.original_keys
        bot.GROQ_MODELS[:] = self.original_models
        bot.CURRENT_KEY_INDEX = self.original_key_index
        bot._RATE_LIMIT_UNTIL.clear()
        bot._UNAVAILABLE_MODELS.clear()
        bot._INVALID_KEYS.clear()
        bot.HISTORY.clear()

    async def test_429_falls_back_on_same_key_and_remembers_cooldown(self):
        calls = []

        def responder(key, model):
            if model == "model-strong":
                return FakeResponse(429, {"error": {"message": "rate limited"}}, {"retry-after": "3600"})
            return success_response("Ответ с резервной модели")

        with patch.object(bot.httpx, "AsyncClient", side_effect=lambda **kwargs: FakeAsyncClient(responder, calls)):
            answer = await bot.ask_ai("chat", "вопрос")
            second_answer = await bot.ask_ai("chat", "ещё вопрос")

        self.assertEqual(answer, "Ответ с резервной модели")
        self.assertEqual(second_answer, "Ответ с резервной модели")
        self.assertEqual(calls, [
            ("key-1", "model-strong"),
            ("key-1", "model-lite"),
            ("key-1", "model-lite"),
        ])

    async def test_429_cascade_tries_next_key_after_all_models(self):
        bot.GROQ_API_KEYS[:] = ["key-1", "key-2"]
        calls = []

        def responder(key, model):
            if key == "key-1":
                return FakeResponse(429, {"error": {"message": "rate limited"}}, {"retry-after": "3600"})
            return success_response("Ответ со второго ключа")

        with patch.object(bot.httpx, "AsyncClient", side_effect=lambda **kwargs: FakeAsyncClient(responder, calls)):
            answer = await bot.ask_ai("chat", "вопрос")

        self.assertEqual(answer, "Ответ со второго ключа")
        self.assertEqual(calls, [
            ("key-1", "model-strong"),
            ("key-1", "model-lite"),
            ("key-2", "model-strong"),
        ])
        self.assertEqual(bot.CURRENT_KEY_INDEX, 1)

    async def test_deprecated_model_is_skipped_after_model_not_found(self):
        calls = []

        def responder(key, model):
            if model == "model-strong":
                return FakeResponse(404, {"error": {"code": "model_decommissioned", "message": "model is unavailable"}})
            return success_response()

        with patch.object(bot.httpx, "AsyncClient", side_effect=lambda **kwargs: FakeAsyncClient(responder, calls)):
            await bot.ask_ai("chat", "первый вопрос")
            await bot.ask_ai("chat", "второй вопрос")

        self.assertEqual(calls, [
            ("key-1", "model-strong"),
            ("key-1", "model-lite"),
            ("key-1", "model-lite"),
        ])

    async def test_model_access_403_falls_back_without_disabling_key(self):
        calls = []

        def responder(key, model):
            if model == "model-strong":
                return FakeResponse(403, {"error": {"code": "model_access_denied", "message": "No permission to use this model"}})
            return success_response("Резервная модель работает")

        with patch.object(bot.httpx, "AsyncClient", side_effect=lambda **kwargs: FakeAsyncClient(responder, calls)):
            answer = await bot.ask_ai("chat", "вопрос")

        self.assertEqual(answer, "Резервная модель работает")
        self.assertEqual(calls, [("key-1", "model-strong"), ("key-1", "model-lite")])
        self.assertNotIn(0, bot._INVALID_KEYS)

    async def test_auth_failure_skips_all_models_for_that_key(self):
        bot.GROQ_API_KEYS[:] = ["bad-key", "good-key"]
        calls = []

        def responder(key, model):
            if key == "bad-key":
                return FakeResponse(401, {"error": {"message": "invalid API key"}})
            return success_response("Ок")

        with patch.object(bot.httpx, "AsyncClient", side_effect=lambda **kwargs: FakeAsyncClient(responder, calls)):
            answer = await bot.ask_ai("chat", "вопрос")

        self.assertEqual(answer, "Ок")
        self.assertEqual(calls, [("bad-key", "model-strong"), ("good-key", "model-strong")])


if __name__ == "__main__":
    unittest.main()
