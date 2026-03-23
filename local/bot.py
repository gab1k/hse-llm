"""
bot.py — Telegram-бот для суммаризации текстов.

Запуск:
    export BOT_TOKEN="ваш_токен"
    python bot.py
"""

import asyncio
import logging
import os
import tempfile

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    Document,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 120
MAX_TEXT_LENGTH = 8000

MODE_LABELS = {
    "short":  "🔹 Краткое (~30-50 слов)",
    "medium": "🔸 Среднее (~60-80 слов)",
    "long":   "🔶 Развёрнутое (~100-130 слов)",
}


class SummaryState(StatesGroup):
    waiting_for_mode = State()


def make_mode_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(text=label, callback_data=f"mode:{mode}")]
        for mode, label in MODE_LABELS.items()
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


async def call_summarize_api(text: str, mode: str) -> dict:
    url = f"{SERVER_URL}/summarize"
    payload = {"text": text, "mode": mode}
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    return await resp.json()
                body = await resp.text()
                return {"error": f"HTTP {resp.status}: {body[:200]}"}
        except aiohttp.ServerTimeoutError:
            return {"error": "Таймаут запроса (сервер долго не отвечает)"}
        except aiohttp.ClientConnectorError:
            return {"error": "Не удалось подключиться к серверу. Проверьте SSH-туннель."}
        except Exception as e:
            return {"error": f"Неожиданная ошибка: {e}"}


async def check_server_health() -> bool:
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{SERVER_URL}/health") as resp:
                return resp.status == 200
    except Exception:
        return False


def format_summary_response(result: dict) -> str:
    mode_emoji = {"short": "🔹", "medium": "🔸", "long": "🔶"}.get(result["mode"], "📄")
    mode_name = {"short": "Краткое", "medium": "Среднее", "long": "Развёрнутое"}.get(
        result["mode"], result["mode"]
    )
    return (
        f"{mode_emoji} *{mode_name} резюме*\n"
        f"{'─' * 35}\n"
        f"{result['summary']}\n"
        f"{'─' * 35}\n"
        f"📊 *Статистика:*\n"
        f"  • Исходный текст: {result['input_words']} слов\n"
        f"  • Резюме: {result['output_words']} слов\n"
        f"  • Сжатие: {result['compression_ratio']:.1%}\n"
        f"  • Время генерации: {result['time_s']} с"
    )


bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())


@dp.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(
        "👋 Привет! Я бот для суммаризации текстов.\n\n"
        "📝 *Что я умею:*\n"
        "  • Кратко излагать длинные тексты\n"
        "  • Работать с тремя режимами длины\n"
        "  • Принимать .txt файлы\n\n"
        "🚀 Просто отправь мне текст или .txt файл!\n\n"
        "⌨️ /help — помощь, /status — статус сервера",
        parse_mode="Markdown",
    )


@dp.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "📖 *Инструкция*\n\n"
        "1️⃣ Отправь текст (минимум ~50 символов)\n"
        "2️⃣ Выбери режим длины резюме:\n"
        "   🔹 *Краткое* — только ключевые факты\n"
        "   🔸 *Среднее* — основные детали\n"
        "   🔶 *Развёрнутое* — полный контекст\n"
        "3️⃣ Получи готовое резюме!\n\n"
        "📎 Также поддерживаются .txt файлы.\n"
        f"⚠️ Максимальная длина текста: {MAX_TEXT_LENGTH} символов",
        parse_mode="Markdown",
    )


@dp.message(Command("status"))
async def cmd_status(message: Message):
    msg = await message.answer("🔍 Проверяю статус сервера...")
    is_ok = await check_server_health()
    if is_ok:
        await msg.edit_text(f"✅ *Сервер работает*\nURL: `{SERVER_URL}`", parse_mode="Markdown")
    else:
        await msg.edit_text(
            "❌ *Сервер недоступен*\n\n"
            "Возможные причины:\n"
            "  • SSH-туннель не активен\n"
            "  • Сервер выключен\n\n"
            "Команда для туннеля:\n"
            "`ssh -N -L 8000:localhost:8000 <your_server>`",
            parse_mode="Markdown",
        )


@dp.message(F.text, SummaryState.waiting_for_mode)
async def handle_text_during_mode_selection(message: Message):
    await message.answer(
        "⏳ Сначала выбери режим для предыдущего текста, "
        "или отправь /start чтобы начать заново."
    )


@dp.message(F.document)
async def handle_document(message: Message, state: FSMContext):
    doc: Document = message.document
    if not doc.file_name.endswith(".txt"):
        await message.answer("⚠️ Поддерживаются только *.txt* файлы", parse_mode="Markdown")
        return

    processing_msg = await message.answer("📥 Читаю файл...")
    try:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            await bot.download(doc, destination=tmp.name)
            tmp_path = tmp.name
        with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        os.unlink(tmp_path)
    except Exception as e:
        await processing_msg.edit_text(f"❌ Ошибка чтения файла: {e}")
        return

    if len(text) < 50:
        await processing_msg.edit_text("⚠️ Файл слишком короткий (минимум 50 символов).")
        return

    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
        await processing_msg.edit_text(
            f"⚠️ Файл обрезан до {MAX_TEXT_LENGTH} символов.\nВыбери режим резюме:",
            reply_markup=make_mode_keyboard(),
        )
    else:
        await processing_msg.edit_text(
            f"✅ Файл прочитан ({len(text)} символов).\nВыбери режим резюме:",
            reply_markup=make_mode_keyboard(),
        )

    await state.update_data(text=text)
    await state.set_state(SummaryState.waiting_for_mode)


@dp.message(F.text)
async def handle_text(message: Message, state: FSMContext):
    text = message.text.strip()
    if len(text) < 50:
        await message.answer("📝 Текст слишком короткий (минимум ~50 символов).")
        return

    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
        await message.answer(f"⚠️ Текст обрезан до {MAX_TEXT_LENGTH} символов.")

    await state.update_data(text=text)
    await state.set_state(SummaryState.waiting_for_mode)
    await message.answer(
        f"📄 Текст получен ({len(text)} символов).\nВыбери режим резюме:",
        reply_markup=make_mode_keyboard(),
    )


@dp.callback_query(F.data.startswith("mode:"), SummaryState.waiting_for_mode)
async def handle_mode_selection(callback: CallbackQuery, state: FSMContext):
    mode = callback.data.split(":")[1]
    data = await state.get_data()
    text = data.get("text", "")

    await state.clear()

    if not text:
        await callback.answer("Текст не найден, отправь его заново", show_alert=True)
        return

    await callback.answer()
    await callback.message.edit_text(
        f"⏳ Генерирую {MODE_LABELS[mode].lower()} резюме...\nЭто может занять 10-30 секунд."
    )

    result = await call_summarize_api(text, mode)

    if "error" in result:
        await callback.message.edit_text(f"❌ *Ошибка:*\n{result['error']}", parse_mode="Markdown")
        return

    response_text = format_summary_response(result)

    other_modes = [m for m in ["short", "medium", "long"] if m != mode]
    retry_buttons = [
        [InlineKeyboardButton(text=f"↩️ {MODE_LABELS[m]}", callback_data=f"retry:{m}")]
        for m in other_modes
    ]
    retry_keyboard = InlineKeyboardMarkup(inline_keyboard=retry_buttons)

    await state.update_data(text=text)
    await state.set_state(SummaryState.waiting_for_mode)

    await callback.message.edit_text(response_text, parse_mode="Markdown", reply_markup=retry_keyboard)


@dp.callback_query(F.data.startswith("retry:"), SummaryState.waiting_for_mode)
async def handle_retry(callback: CallbackQuery, state: FSMContext):
    callback.data = callback.data.replace("retry:", "mode:")
    await handle_mode_selection(callback, state)


async def main():
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        logger.error("Установите переменную окружения BOT_TOKEN")
        return

    await bot.set_my_commands([
        BotCommand(command="start", description="Начать работу"),
        BotCommand(command="help", description="Инструкция"),
        BotCommand(command="status", description="Статус сервера"),
    ])

    logger.info(f"Запуск бота, сервер: {SERVER_URL}")
    is_ok = await check_server_health()
    logger.info(f"Статус сервера: {'OK' if is_ok else 'Недоступен'}")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
