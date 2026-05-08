#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI


VALID_POSITIONS = {"front", "middle", "end"}
VALID_LENGTH_TAGS = {"256", "512", "1k", "1.5k", "2k"}


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")[:180] or "model"


def normalize_position(value: Any) -> str:
    pos = str(value).strip().lower()
    if pos not in VALID_POSITIONS:
        raise ValueError(f"Invalid position: {value!r}. Expected one of {sorted(VALID_POSITIONS)}")
    return pos


def infer_length_tag(path: Path) -> str:
    candidates = [path.name, path.stem]
    candidates.extend([p.name for p in path.parents])

    text = " ".join(candidates).lower()

    patterns = [
        (r"(?<![\w.])1\.5k(?![\w.])", "1.5k"),
        (r"(?<![\w.])2k(?![\w.])", "2k"),
        (r"(?<![\w.])1k(?![\w.])", "1k"),
        (r"(?<![\w.])512(?![\w.])", "512"),
        (r"(?<![\w.])256(?![\w.])", "256"),
    ]

    for pattern, tag in patterns:
        if re.search(pattern, text):
            return tag

    m = re.search(r"position[_ -]?(256|512|1k|1\.5k|2k)", text)
    if m:
        return m.group(1)

    raise ValueError(f"Cannot infer length tag from path: {path}")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            if "pid" not in obj:
                raise ValueError(f"{path} line {line_no}: missing required field 'pid'")
            if "position" not in obj:
                raise ValueError(f"{path} line {line_no}: missing required field 'position'")
            if "code" not in obj:
                raise ValueError(f"{path} line {line_no}: missing required field 'code'")

            obj["pid"] = int(obj["pid"])
            obj["position"] = normalize_position(obj["position"])
            obj["code"] = str(obj["code"])
            out.append(obj)

    return out


def load_done_pids(out_file: Path) -> Set[int]:
    done: Set[int] = set()

    if not out_file.exists():
        return done

    with out_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                if "pid" in obj:
                    done.add(int(obj["pid"]))
            except Exception:
                pass

    return done


def build_messages(code_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": code_text},
    ]


def iter_input_jsonls(path: Path) -> List[Path]:
    if path.is_file():
        if path.suffix != ".jsonl":
            raise ValueError(f"Input file must be .jsonl: {path}")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"--data_jsonl not found: {path}")

    files = []
    for fp in sorted(path.glob("*.jsonl")):
        if fp.suffix == ".jsonl" and ":Zone.Identifier" not in fp.name:
            files.append(fp)

    if not files:
        raise SystemExit(f"No valid .jsonl files found under: {path}")

    return files


def group_items_by_position(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {pos: [] for pos in sorted(VALID_POSITIONS)}

    for item in items:
        grouped[item["position"]].append(item)

    return {pos: rows for pos, rows in grouped.items() if rows}


@dataclass
class ModelArgs:
    max_tokens: int = 800
    temperature: float = 0.2
    top_p: float = 0.95
    enable_thinking: bool = False
    thinking_budget: int = 0
    force_temperature: bool = False


class CloseSourceLLM:
    def __init__(self, model_name: str, model_args: ModelArgs, base_url: str, api_key: str):
        self.model_name = model_name
        self.total_token_cnt = 0

        base_url = base_url[:-1] if base_url.endswith("/") else base_url
        self.client = OpenAI(base_url=base_url, api_key=api_key)

        self.kw = {
            "model": model_name,
            "max_tokens": int(model_args.max_tokens),
        }

        if any(k in model_name.lower() for k in ["qwen3", "r1"]):
            self.kw["extra_body"] = {
                "enable_thinking": bool(model_args.enable_thinking),
                "thinking_budget": int(model_args.thinking_budget),
            }

        if not any(k in model_name.lower() for k in ["o1", "o3", "o4", "claude", "gemini", "gpt"]):
            self.kw["temperature"] = float(model_args.temperature)
            self.kw["top_p"] = float(model_args.top_p)

        if model_args.force_temperature:
            self.kw["temperature"] = float(model_args.temperature)
            self.kw["top_p"] = float(model_args.top_p)

    def _extract(self, completion) -> Dict[str, Any]:
        out = {
            "formal_answer": "",
            "all_token_count": 0,
        }

        try:
            msg = completion.choices[0].message
            if getattr(msg, "content", None) is not None:
                out["formal_answer"] = str(msg.content).strip()

            usage = getattr(completion, "usage", None)
            if usage is not None:
                out["all_token_count"] = int(getattr(usage, "completion_tokens", 0) or 0)
        except Exception:
            pass

        return out

    def get_response(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 4,
        retry_sleep: float = 1.0,
    ) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        self.kw["messages"] = messages

        for attempt in range(1, max_retries + 1):
            try:
                completion = self.client.chat.completions.create(**self.kw)
                out = self._extract(completion)
                self.total_token_cnt += int(out.get("all_token_count", 0) or 0)
                return out
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    time.sleep(retry_sleep * attempt)
                else:
                    return {
                        "formal_answer": "",
                        "all_token_count": 0,
                        "error": repr(last_err),
                    }

    def get_token_count(self) -> int:
        return int(self.total_token_cnt)


def parse_target(s: str) -> Tuple[str, str, str, str]:
    parts = s.split("|")
    if len(parts) != 4:
        raise ValueError(f"Bad --target format: {s}")
    return parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()


def maybe_check_models(client: OpenAI, model_name: str) -> bool:
    try:
        models = client.models.list()
        names = set()

        for model in getattr(models, "data", []) or []:
            model_id = getattr(model, "id", None)
            if model_id:
                names.add(str(model_id))

        return (not names) or (model_name in names)
    except Exception:
        return True


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target", action="append", default=[], help="model|group|base_url|api_key_env")
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=800)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--check_models", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--thinking_budget", type=int, default=int(os.environ.get("THINKING_BUDGET", "2048")))
    ap.add_argument("--force_temperature", type=int, default=int(os.environ.get("FORCE_TEMPERATURE", "0")))
    return ap.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data_jsonl)
    out_root = Path(args.out_dir)
    in_files = iter_input_jsonls(data_path)

    if not args.target:
        raise SystemExit("No --target provided.")

    print(f"[INPUT] {len(in_files)} file(s) from: {data_path}")
    print(f"[OUTPUT] base output dir: {out_root}")

    for in_file in in_files:
        length_tag = infer_length_tag(in_file)
        out_len_root = out_root / length_tag
        out_len_root.mkdir(parents=True, exist_ok=True)

        dataset = read_jsonl(in_file)
        if args.limit and args.limit > 0:
            dataset = dataset[: args.limit]

        grouped = group_items_by_position(dataset)
        group_desc = ", ".join(f"{pos}={len(rows)}" for pos, rows in grouped.items())

        print(f"\n[DATA] {in_file.name} length={length_tag} samples={len(dataset)} positions=({group_desc})")

        for position_tag, position_rows in grouped.items():
            pos_dir = out_len_root / position_tag
            pos_dir.mkdir(parents=True, exist_ok=True)

            print(f"[POSITION] length={length_tag} position={position_tag} -> {pos_dir}")

            for target in args.target:
                model_name, group, base_url, api_key_env = parse_target(target)
                api_key = os.environ.get(api_key_env, "")

                if not api_key:
                    raise SystemExit(f"Missing env var for api key: {api_key_env}")

                margs = ModelArgs(
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    enable_thinking=(group.lower() == "thinking"),
                    thinking_budget=args.thinking_budget if group.lower() == "thinking" else 0,
                    force_temperature=bool(args.force_temperature),
                )

                llm = CloseSourceLLM(
                    model_name=model_name,
                    model_args=margs,
                    base_url=base_url,
                    api_key=api_key,
                )

                if args.check_models:
                    ok = maybe_check_models(llm.client, model_name)
                    if not ok:
                        print(f"[WARN] model not found in /models: {model_name}. Continue anyway.")

                out_file = pos_dir / f"{sanitize_filename(model_name)}__{sanitize_filename(group)}.jsonl"
                done = load_done_pids(out_file) if args.resume else set()

                print(f"[MODEL] {model_name} group={group} resume_skips={len(done)} -> {out_file}")

                with out_file.open("a", encoding="utf-8") as f_out:
                    for i, item in enumerate(position_rows, 1):
                        pid = int(item["pid"])
                        if pid in done:
                            continue

                        messages = build_messages(item["code"])
                        out = llm.get_response(
                            messages,
                            max_retries=args.retries,
                            retry_sleep=1.0,
                        )

                        rec: Dict[str, Any] = {
                            "src": in_file.name,
                            "length": length_tag,
                            "position": position_tag,
                            "pid": pid,
                            "answer": out.get("formal_answer", ""),
                        }

                        if out.get("all_token_count") is not None:
                            rec["completion_tokens"] = int(out.get("all_token_count", 0) or 0)

                        if out.get("error"):
                            rec["error"] = out["error"]

                        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f_out.flush()
                        done.add(pid)

                        if args.sleep > 0:
                            time.sleep(args.sleep)

                        if i % 25 == 0:
                            print(f"  progress {i}/{len(position_rows)} position={position_tag} last_pid={pid}")

                print(
                    f"[DONE] {model_name} length={length_tag} "
                    f"position={position_tag} completion_tokens_total={llm.get_token_count()}"
                )

    print(f"\n[OK] outputs in: {out_root}")


if __name__ == "__main__":
    main()