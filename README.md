# TL;DR по-русски: управляемая абстрактивная суммаризация текста

## Выполнил Габидуллин Камиль Маратович
### Курс "Большие языковые модели", HSE

**Презентация:** https://docs.google.com/presentation/d/1mfpCenbjwq5oISoyHzxNOnUpZ6it8dxF

---

## Задача

По входному тексту генерировать структурированное резюме на русском в выбранном режиме длины (short / medium / long). Решение оформлено как Telegram-бот с GPU-инференсом на удалённом сервере.

## Данные

**IlyaGusev/gazeta v2.0** — корпус русскоязычных новостных статей Газеты.ру с человеческими саммари:
- Train: 60 964 | Validation: 6 369 | Test: 6 793
- Средняя длина текста: ~633 слова, саммари: ~42 слова
- Compression ratio: ~7%

## Метрики

- **ROUGE-1/2/L** — перекрытие n-грамм с эталоном
- **BERTScore** — семантическое сходство (модель для русского)
- **LLM-as-a-judge** — оценка Qwen-7B по 5-балльной шкале (точность, полнота, связность)

## Быстрый старт

### 1. Скачивание моделей и данных (с интернетом)

```bash
cd server
pip install -r requirements.txt
python save_dataset.py --output_dir ../offline_assets
```

### 2. Перенос на удалённый сервер (без интернета)

```bash
rsync -avh --progress ./offline_assets/ user@gpu-server:~/project/offline_assets/
rsync -avh --progress ./server/ user@gpu-server:~/project/server/
```

### 3. Дообучение на GPU-сервере

```bash
cd ~/project/server
pip install -r requirements.txt

# Запустить server/experiments.ipynv
```

### 4. Запуск сервера инференса

```bash
python server.py \
    --model_path <model_path> \
    --model_type <model_type>
```
например

```bash
python server.py \
    --model_path ../checkpoints/qwen2.5-6b-lora/checkpoint-25 \
    --model_type causal_lora
```


Сервер слушает `http://127.0.0.1:8000`

### 5. SSH-туннель (на macOS)

```bash
ssh -N -L 8000:localhost:8000 gab1k-gpu-research
```

### 6. Запуск Telegram-бота (на macOS)

```bash
cd local
pip install -r requirements.txt
export BOT_TOKEN="ваш_токен_от_BotFather"
python bot.py
```

---

## API-эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | Статус сервера, GPU, VRAM |
| POST | `/summarize` | Суммаризация текста |
| GET | `/modes` | Доступные режимы длины |

**POST /summarize** — тело запроса:
```json
{"text": "Текст статьи...", "mode": "medium"}
```

**Ответ:**
```json
{
  "summary": "Сгенерированное резюме...",
  "mode": "medium",
  "input_words": 500,
  "output_words": 45,
  "compression_ratio": 0.09,
  "time_s": 2.5
}
```

---

## Deployment: техническая записка

**Способ инференса:** Real-time (синхронный HTTP)

**Стек:**
- FastAPI + Uvicorn (HTTP API)
- HuggingFace Transformers (загрузка и инференс моделей)
- PEFT (LoRA адаптеры)
- torch.float16 (half-precision для экономии VRAM)

**Ожидаемая нагрузка:** 1-5 RPS (учебный проект, один пользователь)

**Bottlenecks:**
1. GPU-инференс (основное время — генерация токенов)
2. SSH-туннель (добавляет ~1-5ms латентности)
3. Telegram API (внешняя зависимость)

**Примерная стоимость:**
- Tesla V100 32GB в облаке: ~$2-3/час
- При использовании 8 часов/день, 30 дней: ~$480-720/мес
