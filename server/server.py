"""
server.py — FastAPI-сервер для суммаризации, поддержка seq2seq и causal моделей.

Запуск:
  python server.py --model_path ./checkpoints/fred-t5-large --model_type seq2seq
  python server.py --model_path Qwen/Qwen2.5-7B-Instruct --model_type causal_zeroshot
  python server.py --model_path ./checkpoints/qwen2.5-7b-lora --model_type causal_lora \
      --base_model_path ./offline_assets/Qwen__Qwen2.5-7B-Instruct
"""

import argparse
import logging
import os
import time
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEQ2SEQ_PREFIX = "Summarize: "
SEQ2SEQ_MAX_INPUT = 800

LENGTH_CONFIG = {
    "short": {
        "seq2seq": {"max_new_tokens": 60, "min_new_tokens": 20, "num_beams": 4},
        "causal": {"max_new_tokens": 120, "temperature": 0.3, "do_sample": False},
        "instruction": "Напиши КРАТКОЕ резюме статьи в 2-3 предложения.\n\n",
    },
    "medium": {
        "seq2seq": {"max_new_tokens": 120, "min_new_tokens": 50, "num_beams": 4},
        "causal": {"max_new_tokens": 220, "temperature": 0.3, "do_sample": False},
        "instruction": "Напиши резюме статьи в 4-5 предложений.\n\n",
    },
    "long": {
        "seq2seq": {"max_new_tokens": 200, "min_new_tokens": 80, "num_beams": 4},
        "causal": {"max_new_tokens": 350, "temperature": 0.4, "do_sample": True},
        "instruction": "Напиши ПОДРОБНОЕ резюме статьи в 6-8 предложений.\n\n",
    },
}


def build_causal_prompt(text: str, mode: str) -> str:
    instruction = LENGTH_CONFIG[mode]["instruction"]
    if len(text) > 3500:
        text = text[:3500] + "..."
    return (
        f"<|im_start|>system\n"
        f"Ты — эксперт по созданию резюме новостных статей на русском языке. "
        f"Пиши чётко, информативно, без воды.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{instruction}"
        f"Статья:\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


class SummarizationModel:
    def __init__(self, model_path: str, model_type: str, base_model_path: Optional[str] = None):
        """
        model_type: "seq2seq" | "causal_lora" | "causal_zeroshot"
        base_model_path: путь к базовой модели (нужен для causal_lora)
        """
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Загрузка модели [{model_type}]: {model_path}, device={self.device}")

        if model_type == "seq2seq":
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path, torch_dtype=torch.float16
            ).to(self.device)
        elif model_type == "causal_zeroshot":
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
            )
        elif model_type == "causal_lora":
            if not base_model_path:
                raise ValueError("Для causal_lora нужен --base_model_path с путём к базовой модели")
            from peft import PeftModel
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
            base = AutoModelForCausalLM.from_pretrained(
                base_model_path, torch_dtype=torch.float16, trust_remote_code=True
            )
            self.model = PeftModel.from_pretrained(base, model_path)
            self.model = self.model.merge_and_unload()
            self.model = self.model.to(self.device)
            logger.info("LoRA адаптер смержен с базовой моделью")
        else:
            raise ValueError(f"Неизвестный model_type: {model_type}")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info(f"Модель загружена ({n_params:.1f}M параметров)")
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"VRAM: {used:.1f} / {total:.1f} GB")

    def summarize(self, text: str, mode: str = "medium") -> dict:
        if mode not in LENGTH_CONFIG:
            raise ValueError(f"Неизвестный режим: {mode}")

        t0 = time.time()
        if self.model_type == "seq2seq":
            summary = self._generate_seq2seq(text, mode)
        else:
            summary = self._generate_causal(text, mode)

        elapsed = round(time.time() - t0, 3)
        input_words = len(text.split())
        output_words = len(summary.split())

        return {
            "summary": summary,
            "mode": mode,
            "input_words": input_words,
            "output_words": output_words,
            "compression_ratio": round(output_words / input_words, 3) if input_words else 0,
            "time_s": elapsed,
        }

    def _generate_seq2seq(self, text: str, mode: str) -> str:
        params = LENGTH_CONFIG[mode]["seq2seq"]
        inputs = self.tokenizer(
            SEQ2SEQ_PREFIX + text,
            max_length=SEQ2SEQ_MAX_INPUT,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                no_repeat_ngram_size=3,
                early_stopping=True,
                length_penalty=1.0,
                **params,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def _generate_causal(self, text: str, mode: str) -> str:
        params = LENGTH_CONFIG[mode]["causal"]
        prompt = build_causal_prompt(text, mode)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1500
        ).to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                no_repeat_ngram_size=3,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                pad_token_id=self.tokenizer.pad_token_id,
                **params,
            )

        generated = output_ids[0][prompt_len:]
        summary = self.tokenizer.decode(generated, skip_special_tokens=True)
        summary = summary.replace("<|im_end|>", "").strip()
        return summary


app = FastAPI(title="RuSum API", version="2.0.0")
summarization_model: Optional[SummarizationModel] = None


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=50, max_length=15000)
    mode: str = Field(default="medium")


class SummarizeResponse(BaseModel):
    summary: str
    mode: str
    input_words: int
    output_words: int
    compression_ratio: float
    time_s: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str]
    device: str
    gpu_name: Optional[str]
    vram_used_gb: Optional[float]
    vram_total_gb: Optional[float]


@app.get("/health", response_model=HealthResponse)
async def health():
    vram_used = vram_total = gpu_name = None
    if torch.cuda.is_available():
        vram_used = round(torch.cuda.memory_allocated() / 1e9, 2)
        vram_total = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        gpu_name = torch.cuda.get_device_name(0)

    return HealthResponse(
        status="ok",
        model_loaded=summarization_model is not None,
        model_type=summarization_model.model_type if summarization_model else None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        gpu_name=gpu_name,
        vram_used_gb=vram_used,
        vram_total_gb=vram_total,
    )


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    if summarization_model is None:
        raise HTTPException(status_code=503, detail="Модель загружается, повторите позже")
    if request.mode not in LENGTH_CONFIG:
        raise HTTPException(status_code=400, detail=f"Неизвестный режим '{request.mode}'")

    try:
        result = summarization_model.summarize(request.text, request.mode)
        logger.info(
            f"mode={result['mode']} | in={result['input_words']}w -> out={result['output_words']}w | t={result['time_s']}s"
        )
        return SummarizeResponse(**result)
    except Exception as e:
        logger.error(f"Ошибка генерации: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/modes")
async def modes():
    return {
        "short": "2-3 предложения",
        "medium": "4-5 предложений",
        "long": "6-8 предложений",
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=os.getenv("MODEL_PATH", "ai-forever/FRED-T5-large"))
    parser.add_argument("--model_type", type=str, default=os.getenv("MODEL_TYPE", "seq2seq"),
                        choices=["seq2seq", "causal_lora", "causal_zeroshot"])
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="Путь к базовой модели для causal_lora")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summarization_model = SummarizationModel(args.model_path, args.model_type, args.base_model_path)
    logger.info(f"Запуск на http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
