import os
import json
import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Day 1 Advanced LLM Control with Groq"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question to ask the model. If omitted, interactive input is used."
    )
    parser.add_argument(
        "--system",
        type=str,
        default="You are a clear, concise, helpful AI assistant.",
        help="System prompt to control model behavior."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.1-8b-instant",
        help="Groq model to use."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature. Lower = more deterministic."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=300,
        help="Maximum completion tokens."
    )
    parser.add_argument(
        "--json_mode",
        action="store_true",
        help="Force JSON output."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save request/response log to logs/."
    )
    return parser.parse_args()


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY in environment or .env file.")
    return Groq(api_key=api_key)


def build_messages(system_prompt: str, question: str, json_mode: bool):
    if json_mode:
        system_prompt = (
            system_prompt.strip()
            + "\n\nReturn only valid JSON. No markdown. No explanation outside JSON."
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def call_model(
    client: Groq,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    json_mode: bool,
):
    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # JSON mode is supported on some Groq models, including llama-3.1-8b-instant.
    if json_mode:
        params["response_format"] = {"type": "json_object"}

    return client.chat.completions.create(**params)


def save_log(payload: dict):
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = logs_dir / f"run_{timestamp}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return out_file


def main():
    args = parse_args()
    client = get_client()

    question = args.question or input("Ask something: ").strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    messages = build_messages(args.system, question, args.json_mode)

    try:
        completion = call_model(
            client=client,
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            json_mode=args.json_mode,
        )

        answer = completion.choices[0].message.content or ""

        print("\n=== ANSWER ===\n")
        print(answer)

        if args.save:
            payload = {
                "timestamp": datetime.now().isoformat(),
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "json_mode": args.json_mode,
                "messages": messages,
                "answer": answer,
            }
            log_file = save_log(payload)
            print(f"\nSaved log to: {log_file}")

    except Exception as e:
        print("\nRequest failed.")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()