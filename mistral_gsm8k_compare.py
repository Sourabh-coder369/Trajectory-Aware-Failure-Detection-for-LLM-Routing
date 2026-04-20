import argparse
import csv
import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import error, request


api_key = "WhL4l5917Ko6ffOtCe6BtC12V6QlXTfu"
API_URL = "https://api.mistral.ai/v1/chat/completions"
DEFAULT_MODEL_A = "ministral-3b-2512"
DEFAULT_MODEL_B = "ministral-14b-2512"
DEFAULT_JUDGE_MODEL = "mistral-large-2411"


class PerModelRateLimiter:
    def __init__(self, min_interval_seconds: float = 2.0) -> None:
        self.min_interval_seconds = min_interval_seconds
        self.last_call_time: Dict[str, float] = {}

    def wait(self, model_name: str) -> None:
        last = self.last_call_time.get(model_name)
        now = time.monotonic()
        if last is not None:
            elapsed = now - last
            wait_time = self.min_interval_seconds - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
        self.last_call_time[model_name] = time.monotonic()


class MistralClient:
    def __init__(
        self,
        api_key: str,
        min_interval_seconds: float = 2.0,
        timeout_seconds: int = 90,
        max_retries: int = 12,
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.rate_limiter = PerModelRateLimiter(min_interval_seconds=min_interval_seconds)

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")

        last_exception: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            self.rate_limiter.wait(model)
            req = request.Request(
                API_URL,
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw)
                    content = data["choices"][0]["message"]["content"]
                    return _normalize_content(content)
            except error.HTTPError as http_err:
                detail = http_err.read().decode("utf-8", errors="replace")
                last_exception = RuntimeError(f"HTTP {http_err.code}: {detail}")
                if http_err.code in (429, 500, 502, 503, 504):
                    detail_type, detail_code = _extract_error_type_and_code(detail)
                    # Capacity errors (429/code 3505) often need a longer cooldown.
                    if http_err.code == 429 and (
                        detail_type == "service_tier_capacity_exceeded" or detail_code == "3505"
                    ):
                        backoff = min(20 * attempt, 180) + random.uniform(0.0, 2.0)
                    else:
                        backoff = min(2 ** attempt, 30) + random.uniform(0.0, 1.0)
                    print(
                        f"[warn] {model} request failed with HTTP {http_err.code}; "
                        f"retrying in {backoff:.1f}s (attempt {attempt}/{self.max_retries})"
                    )
                    time.sleep(backoff)
                    continue
                raise last_exception
            except Exception as ex:
                last_exception = ex
                backoff = min(2 ** attempt, 30) + random.uniform(0.0, 1.0)
                print(
                    f"[warn] {model} request failed with {type(ex).__name__}; "
                    f"retrying in {backoff:.1f}s (attempt {attempt}/{self.max_retries})"
                )
                time.sleep(backoff)

        if last_exception is None:
            raise RuntimeError("Mistral API call failed for unknown reason.")
        raise last_exception


def _normalize_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def _extract_error_type_and_code(detail: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        data = json.loads(detail)
        err_type = data.get("type")
        err_code = data.get("code")
        err_type = str(err_type) if err_type is not None else None
        err_code = str(err_code) if err_code is not None else None
        return err_type, err_code
    except Exception:
        return None, None


def load_gsm8k_csv(csv_path: Path, max_samples: Optional[int] = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not question:
                continue
            rows.append({"question": question, "answer": answer})
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def extract_ground_truth_final(answer_text: str) -> Optional[str]:
    match = re.search(r"####\s*([^\n\r]+)", answer_text)
    if match:
        return match.group(1).strip()
    return None


def generate_model_answer(client: MistralClient, model_name: str, question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a math assistant. Solve the problem carefully and provide the final answer "
                "on the last line in the format: #### <answer>."
            ),
        },
        {"role": "user", "content": question},
    ]
    return client.chat_completion(model=model_name, messages=messages, temperature=0.0, max_tokens=512)


def judge_both_answers(
    client: MistralClient,
    judge_model: str,
    question: str,
    ground_truth_answer: str,
    model_a_answer: str,
    model_b_answer: str,
) -> Tuple[int, int, str]:
    system_prompt = (
        "You are a strict GSM8K answer judge. Compare each candidate answer against the official ground truth. "
        "Return JSON only with this schema: "
        "{\"model_a_correct\":0_or_1,\"model_b_correct\":0_or_1,\"reason\":\"short reason\"}. "
        "Mark correct only if the final numerical answer is mathematically equivalent to the ground truth."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Ground truth answer (includes final after ####):\n{ground_truth_answer}\n\n"
        f"Model A answer:\n{model_a_answer}\n\n"
        f"Model B answer:\n{model_b_answer}\n"
    )

    raw = client.chat_completion(
        model=judge_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=160,
    )

    parsed = _parse_judge_json(raw)
    model_a_correct = 1 if int(parsed.get("model_a_correct", 0)) == 1 else 0
    model_b_correct = 1 if int(parsed.get("model_b_correct", 0)) == 1 else 0
    reason = str(parsed.get("reason", ""))
    return model_a_correct, model_b_correct, reason


def _parse_judge_json(raw_text: str) -> Dict[str, object]:
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    # Safe fallback if judge output is malformed.
    fallback_a = 1 if re.search(r"model_a_correct\s*[:=]\s*1", raw_text) else 0
    fallback_b = 1 if re.search(r"model_b_correct\s*[:=]\s*1", raw_text) else 0
    return {
        "model_a_correct": fallback_a,
        "model_b_correct": fallback_b,
        "reason": "Fallback parse used due to invalid judge JSON.",
    }


def count_existing_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def append_jsonl(path: Path, row: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def append_csv(path: Path, row: Dict[str, object], header_written: bool) -> bool:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not header_written:
            writer.writeheader()
            header_written = True
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())
    return header_written


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_dataset = script_dir.parent / "legacy_root_files" / "gsm8k_dataset.csv"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_jsonl = script_dir / f"mistral_gsm8k_compare_results_{run_id}.jsonl"
    default_csv = script_dir / f"mistral_gsm8k_compare_results_{run_id}.csv"

    parser = argparse.ArgumentParser(description="Compare two Mistral models on GSM8K and score correctness with Mistral judge.")
    parser.add_argument("--api-key", default=os.getenv("MISTRAL_API_KEY", api_key))
    parser.add_argument("--dataset-csv", type=Path, default=default_dataset)
    parser.add_argument("--model-a", default=DEFAULT_MODEL_A)
    parser.add_argument("--model-b", default=DEFAULT_MODEL_B)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--request-interval", type=float, default=2.0)
    parser.add_argument("--max-retries", type=int, default=12)
    parser.add_argument("--output-jsonl", type=Path, default=default_jsonl)
    parser.add_argument("--output-csv", type=Path, default=default_csv)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.api_key == "PASTE_YOUR_MISTRAL_API_KEY_HERE":
        raise ValueError("Set your key via --api-key or environment variable MISTRAL_API_KEY.")

    if not args.dataset_csv.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {args.dataset_csv}")

    rows = load_gsm8k_csv(args.dataset_csv, max_samples=args.max_samples)
    if not rows:
        print("No dataset rows loaded.")
        return

    processed = 0
    if args.resume:
        processed = count_existing_jsonl_rows(args.output_jsonl)

    if processed >= len(rows):
        print("All selected rows already processed. Nothing to do.")
        return

    print(f"Live JSONL output: {args.output_jsonl}")
    print(f"Live CSV output:   {args.output_csv}")

    header_written = args.output_csv.exists() and args.output_csv.stat().st_size > 0
    client = MistralClient(
        api_key=args.api_key,
        min_interval_seconds=args.request_interval,
        max_retries=args.max_retries,
    )

    summary_a = 0
    summary_b = 0

    for idx in range(processed, len(rows)):
        sample = rows[idx]
        question = sample["question"]
        ground_truth_answer = sample["answer"]
        ground_truth_final = extract_ground_truth_final(ground_truth_answer)

        while True:
            try:
                model_a_answer = generate_model_answer(client, args.model_a, question)
                model_b_answer = generate_model_answer(client, args.model_b, question)
                model_a_correct, model_b_correct, judge_reason = judge_both_answers(
                    client=client,
                    judge_model=args.judge_model,
                    question=question,
                    ground_truth_answer=ground_truth_answer,
                    model_a_answer=model_a_answer,
                    model_b_answer=model_b_answer,
                )
                break
            except RuntimeError as ex:
                msg = str(ex)
                if "HTTP 429" in msg or "service_tier_capacity_exceeded" in msg or '"code":"3505"' in msg:
                    cooldown = 60
                    print(f"[warn] capacity hit at row {idx}; waiting {cooldown}s then retrying same row")
                    time.sleep(cooldown)
                    continue
                raise

        row = {
            "index": idx,
            "question": question,
            "ground_truth_answer": ground_truth_answer,
            "ground_truth_final": ground_truth_final,
            "model_a": args.model_a,
            "model_b": args.model_b,
            "model_a_answer": model_a_answer,
            "model_b_answer": model_b_answer,
            "model_a_correct": model_a_correct,
            "model_b_correct": model_b_correct,
            "both_correct": 1 if model_a_correct == 1 and model_b_correct == 1 else 0,
            "judge_model": args.judge_model,
            "judge_reason": judge_reason,
        }

        append_jsonl(args.output_jsonl, row)
        header_written = append_csv(args.output_csv, row, header_written)

        summary_a += model_a_correct
        summary_b += model_b_correct

        done = idx + 1
        print(
            f"Processed {done}/{len(rows)} | "
            f"{args.model_a} correct so far: {summary_a} | "
            f"{args.model_b} correct so far: {summary_b}"
        )

    total_done = len(rows) - processed
    print("\nRun complete")
    print(f"Processed rows this run: {total_done}")
    print(f"Output JSONL: {args.output_jsonl}")
    print(f"Output CSV:   {args.output_csv}")


if __name__ == "__main__":
    main()
