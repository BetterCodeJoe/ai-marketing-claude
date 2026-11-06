#!/usr/bin/env python3
"""
Model Comparison Harness — Utility script for AI Marketing Claude Code Skills
Runs the same /market skill prompt through Claude (Anthropic API) and Muse Spark
(Meta's Anthropic-compatible endpoint) and writes a side-by-side report.

Usage:
    export ANTHROPIC_API_KEY="..."   # for Claude
    export MODEL_API_KEY="..."       # for Muse Spark
    python3 scripts/compare_models.py --skill market-ads --task "Business: ..."
    python3 scripts/compare_models.py --skill market-copy --task-file brief.txt --runs 3
    python3 scripts/compare_models.py --skill market-ads --task "..." --dry-run

Both providers speak the Anthropic Messages API, so one request shape serves both.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PROVIDERS = {
    "claude": {
        "label": "Claude",
        "base_url": "https://api.anthropic.com",
        "model": "claude-opus-5-5",
        "key_env": "ANTHROPIC_API_KEY",
        "auth": "x-api-key",
    },
    "muse": {
        "label": "Muse Spark",
        "base_url": "https://api.meta.ai",
        "model": "muse-spark-1.3-contributor",
        "key_env": "MODEL_API_KEY",
        "auth": "bearer",
    },
}

# Criteria a human (or a judge model) scores 1-5 per output.
RUBRIC = [
    ("Instruction fidelity", "Follows the skill's phases, tables and output format"),
    ("Specificity", "Concrete to this business, not generic marketing filler"),
    ("Actionability", "Could be handed to a media buyer / writer today"),
    ("Accuracy", "No invented stats, features or platform specs"),
    ("Voice & copy quality", "Hooks, headlines and CTAs that would actually perform"),
]


def load_skill(name):
    path = REPO_ROOT / "skills" / name / "SKILL.md"
    if not path.exists():
        available = sorted(p.name for p in (REPO_ROOT / "skills").iterdir() if p.is_dir())
        sys.exit(f"Unknown skill '{name}'. Available: {', '.join(available)}")
    return path.read_text(encoding="utf-8")


def build_request(provider, system_prompt, task, max_tokens):
    cfg = PROVIDERS[provider]
    headers = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
    key = os.environ.get(cfg["key_env"], "")
    if cfg["auth"] == "bearer":
        headers["authorization"] = f"Bearer {key}"
    else:
        headers["x-api-key"] = key
    body = {
        "model": cfg["model"],
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": task}],
    }
    url = cfg["base_url"].rstrip("/") + "/v1/messages"
    return url, headers, body


def call_provider(provider, system_prompt, task, max_tokens, timeout):
    url, headers, body = build_request(provider, system_prompt, task, max_tokens)
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}",
                "latency_s": time.monotonic() - start}
    except (urllib.error.URLError, TimeoutError) as e:
        return {"error": f"Network error: {e}", "latency_s": time.monotonic() - start}

    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    usage = data.get("usage", {})
    return {
        "text": text,
        "latency_s": time.monotonic() - start,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "stop_reason": data.get("stop_reason"),
        "model": data.get("model", PROVIDERS[provider]["model"]),
    }


def summarize(runs):
    ok = [r for r in runs if "error" not in r]

    def avg(key):
        vals = [r[key] for r in ok if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    return {
        "success": f"{len(ok)}/{len(runs)}",
        "latency_s": avg("latency_s"),
        "output_tokens": avg("output_tokens"),
        "words": (sum(len(r["text"].split()) for r in ok) / len(ok)) if ok else None,
        "truncated": sum(1 for r in ok if r.get("stop_reason") == "max_tokens"),
    }


def fmt(v, digits=1):
    if v is None:
        return "—"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def render_report(skill, task, results):
    names = list(results)
    labels = [PROVIDERS[n]["label"] for n in names]
    lines = [
        f"# Model Comparison — `{skill}`",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## Task",
        "",
        "```",
        task.strip(),
        "```",
        "",
        "## Measured",
        "",
        "| Metric | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(names),
    ]
    stats = {n: summarize(results[n]) for n in names}
    for key, label in [("success", "Successful runs"), ("latency_s", "Avg latency (s)"),
                       ("output_tokens", "Avg output tokens"), ("words", "Avg words"),
                       ("truncated", "Hit max_tokens")]:
        lines.append(f"| {label} | " + " | ".join(fmt(stats[n][key]) for n in names) + " |")

    lines += ["", "## Score (fill in 1–5)", "",
              "| Criterion | What good looks like | " + " | ".join(labels) + " |",
              "|---|---|" + "---|" * len(names)]
    for crit, desc in RUBRIC:
        lines.append(f"| {crit} | {desc} | " + " | ".join(" " for _ in names) + " |")
    lines.append("| **Total** | | " + " | ".join(" " for _ in names) + " |")

    for n in names:
        lines += ["", "---", "", f"## {PROVIDERS[n]['label']} output"]
        for i, r in enumerate(results[n], 1):
            lines += ["", f"### Run {i}", ""]
            lines.append(f"> **Error:** {r['error']}" if "error" in r else r["text"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Compare Claude vs Muse Spark on a /market skill")
    parser.add_argument("--skill", required=True, help="Skill folder name, e.g. market-ads")
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", help="Business brief / instruction sent as the user message")
    task_group.add_argument("--task-file", help="File containing the brief")
    parser.add_argument("--providers", default="claude,muse", help="Comma list from: " + ",".join(PROVIDERS))
    parser.add_argument("--claude-model", default=PROVIDERS["claude"]["model"])
    parser.add_argument("--muse-model", default=PROVIDERS["muse"]["model"])
    parser.add_argument("--runs", type=int, default=1, help="Runs per provider (outputs vary run to run)")
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", default="MODEL-COMPARISON.md")
    parser.add_argument("--dry-run", action="store_true", help="Print the requests without sending them")
    args = parser.parse_args()

    PROVIDERS["claude"]["model"] = args.claude_model
    PROVIDERS["muse"]["model"] = args.muse_model
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    for p in providers:
        if p not in PROVIDERS:
            sys.exit(f"Unknown provider '{p}'. Choose from: {', '.join(PROVIDERS)}")

    system_prompt = load_skill(args.skill)
    task = Path(args.task_file).read_text(encoding="utf-8") if args.task_file else args.task
    # Skills normally fetch the site with tools; this harness is text-only, so say so.
    task += ("\n\nNote: you have no web or file tools in this run. Work only from the brief "
             "above and return the deliverable as Markdown in your reply.")

    if args.dry_run:
        for p in providers:
            url, headers, body = build_request(p, system_prompt, task, args.max_tokens)
            redacted = {k: ("<redacted>" if k in ("authorization", "x-api-key") else v) for k, v in headers.items()}
            key_set = bool(os.environ.get(PROVIDERS[p]["key_env"]))
            print(f"[{p}] POST {url}  model={body['model']}  {PROVIDERS[p]['key_env']} set={key_set}")
            print(f"       headers={redacted}  system_chars={len(body['system'])}  task_chars={len(task)}")
        return

    missing = [PROVIDERS[p]["key_env"] for p in providers if not os.environ.get(PROVIDERS[p]["key_env"])]
    if missing:
        sys.exit(f"Missing API key env var(s): {', '.join(missing)}")

    results = {p: [] for p in providers}
    for i in range(args.runs):
        for p in providers:
            print(f"Run {i + 1}/{args.runs} — {PROVIDERS[p]['label']}...", file=sys.stderr)
            r = call_provider(p, system_prompt, task, args.max_tokens, args.timeout)
            status = r["error"] if "error" in r else f"{r['latency_s']:.1f}s, {r.get('output_tokens')} tokens"
            print(f"  {status}", file=sys.stderr)
            results[p].append(r)

    Path(args.output).write_text(render_report(args.skill, task, results), encoding="utf-8")
    raw_path = Path(args.output).with_suffix(".json")
    raw_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} and {raw_path}")


if __name__ == "__main__":
    main()
