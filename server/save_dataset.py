"""
Запуск:
    python server/save_dataset.py --output_dir ./offline_assets
"""

import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS = {
    "ai-forever/FRED-T5-large":    "seq2seq",
    "Qwen/Qwen2.5-3B-Instruct":   "causal",
    "Qwen/Qwen2.5-7B-Instruct":   "causal",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./offline_assets")
    parser.add_argument(
        "--skip_dataset", action="store_true",
        help="Пропустить скачивание датасета (уже есть)"
    )
    return parser.parse_args()


def save_dataset(output_dir: str):
    from datasets import load_dataset

    logger.info("Скачиваем датасет IlyaGusev/gazeta...")
    dataset = load_dataset("IlyaGusev/gazeta", revision="v2.0")

    dataset_path = os.path.join(output_dir, "gazeta_dataset")
    dataset.save_to_disk(dataset_path)
    logger.info(f"✅ Датасет сохранён: {dataset_path}")


def save_model(model_name: str, model_type: str, output_dir: str):
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        AutoModelForCausalLM,
    )

    safe_name  = model_name.replace("/", "__")
    model_path = os.path.join(output_dir, safe_name)

    if os.path.exists(model_path):
        logger.info(f"⏭️  Уже скачана, пропускаем: {model_name}")
        return

    logger.info(f"Скачиваем [{model_type}] {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(model_path)

    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
    elif model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"Неизвестный тип: {model_type}")

    model.save_pretrained(model_path)
    logger.info(f"✅ Сохранена: {model_path}")

    del model
    import gc
    gc.collect()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.skip_dataset:
        dataset_path = os.path.join(args.output_dir, "gazeta_dataset")
        if os.path.exists(dataset_path):
            logger.info("⏭️  Датасет уже есть, пропускаем")
        else:
            save_dataset(args.output_dir)

    for model_name, model_type in MODELS.items():
        try:
            save_model(model_name, model_type, args.output_dir)
        except Exception as e:
            logger.error(f"❌ Ошибка {model_name}: {e}")

    logger.info("Готово! Содержимое папки:")
    for item in sorted(os.listdir(args.output_dir)):
        item_path = os.path.join(args.output_dir, item)
        total = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, files in os.walk(item_path)
            for f in files
        )
        logger.info(f"  {item:<45} {total/1e9:.2f} GB")

    logger.info("\Можно копировать на сервер:")
    logger.info(f"  rsync -avh --progress {args.output_dir}/ user@server:~/project/offline_assets/")


if __name__ == "__main__":
    main()
