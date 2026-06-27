"""
pilot_report.py

Purpose:
    Reads data/pilot_logs.jsonl and prints a summary report of the
    pilot study to the terminal. Designed to be run by the analyst
    or project lead at any point during or after the pilot to check
    how the chatbot is performing.

What the report covers:
    1. Overview     — total messages, unique users, date range, channels
    2. Languages    — distribution of detected languages
    3. Escalation   — rate and count of escalated (crisis/diagnostic) queries
    4. Pipeline     — average chunks retrieved, distribution
    5. Performance  — response time percentiles (p50, p95)
    6. Daily volume — message counts for the last 14 days
    7. Recent logs  — last 10 interactions for spot-checking

Usage:
    python -m evaluation.pilot_report
    python -m evaluation.pilot_report --log data/pilot_logs.jsonl
    python -m evaluation.pilot_report --csv  (also writes report.csv)

Dependencies:
    pandas (already installed) for aggregation
    rich   (already installed) for terminal formatting
"""

import argparse
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.text import Text

from evaluation.logger import LOG_FILE as DEFAULT_LOG_FILE

console = Console()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_logs(log_path: Path) -> pd.DataFrame:
    """
    Reads the JSONL log file into a pandas DataFrame.
    Returns an empty DataFrame with the correct schema if the file
    does not exist or has no valid records.
    """
    _empty = pd.DataFrame(columns=[
        "timestamp", "user_id", "language", "message",
        "answer_preview", "escalated", "chunks_used",
        "response_time_ms", "channel",
    ])

    if not log_path.exists():
        return _empty

    try:
        df = pd.read_json(log_path, lines=True)
    except ValueError:
        return _empty

    if df.empty:
        return _empty

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.date
    return df


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _overview(df: pd.DataFrame) -> None:
    console.print(Panel("[bold cyan]1. OVERVIEW[/bold cyan]", expand=False))

    if df.empty:
        console.print("  [dim]No interactions logged yet.[/dim]\n")
        return

    total = len(df)
    unique_users = df["user_id"].nunique()
    first = df["timestamp"].min().strftime("%Y-%m-%d %H:%M UTC")
    last = df["timestamp"].max().strftime("%Y-%m-%d %H:%M UTC")
    whatsapp = (df["channel"] == "whatsapp").sum()
    api = (df["channel"] == "api").sum()

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column("Metric", style="bold")
    t.add_column("Value")
    t.add_row("Total messages",   str(total))
    t.add_row("Unique users",     str(unique_users))
    t.add_row("First message",    first)
    t.add_row("Last message",     last)
    t.add_row("WhatsApp",         str(whatsapp))
    t.add_row("Direct API",       str(api))
    console.print(t)


def _languages(df: pd.DataFrame) -> None:
    console.print(Panel("[bold cyan]2. LANGUAGE DISTRIBUTION[/bold cyan]", expand=False))

    if df.empty:
        console.print("  [dim]No data.[/dim]\n")
        return

    counts = df["language"].value_counts()
    t = Table(box=box.SIMPLE, padding=(0, 2))
    t.add_column("Language", style="bold")
    t.add_column("Messages", justify="right")
    t.add_column("Share", justify="right")

    lang_names = {"en": "English", "ha": "Hausa", "yo": "Yoruba", "ig": "Igbo"}
    for lang, count in counts.items():
        name = lang_names.get(lang, lang)
        share = f"{count / len(df) * 100:.1f}%"
        t.add_row(f"{name} ({lang})", str(count), share)
    console.print(t)


def _escalation(df: pd.DataFrame) -> None:
    console.print(Panel("[bold cyan]3. ESCALATION[/bold cyan]", expand=False))

    if df.empty:
        console.print("  [dim]No data.[/dim]\n")
        return

    total = len(df)
    escalated = df["escalated"].sum()
    rate = escalated / total * 100 if total else 0

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column("Metric", style="bold")
    t.add_column("Value")
    t.add_row("Total messages",    str(total))
    t.add_row("Escalated",         str(int(escalated)))
    t.add_row("Escalation rate",   f"{rate:.1f}%")
    t.add_row("Educational (RAG)", str(int(total - escalated)))
    console.print(t)


def _pipeline(df: pd.DataFrame) -> None:
    console.print(Panel("[bold cyan]4. PIPELINE — CHUNKS RETRIEVED[/bold cyan]", expand=False))

    if df.empty:
        console.print("  [dim]No data.[/dim]\n")
        return

    # Only count non-escalated messages for chunk stats (escalated = 0 chunks by design)
    rag_df = df[~df["escalated"]]
    if rag_df.empty:
        console.print("  [dim]No RAG interactions yet.[/dim]\n")
        return

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column("Metric", style="bold")
    t.add_column("Value")
    t.add_row("RAG messages",     str(len(rag_df)))
    t.add_row("Avg chunks",       f"{rag_df['chunks_used'].mean():.1f}")
    t.add_row("Median chunks",    f"{rag_df['chunks_used'].median():.0f}")
    t.add_row("Zero-chunk rate",  f"{(rag_df['chunks_used'] == 0).mean() * 100:.1f}%")

    # Distribution
    dist = rag_df["chunks_used"].value_counts().sort_index()
    dist_str = "  ".join(f"{k}x:{v}" for k, v in dist.items())
    t.add_row("Distribution",     dist_str)
    console.print(t)


def _performance(df: pd.DataFrame) -> None:
    console.print(Panel("[bold cyan]5. RESPONSE TIME (ms)[/bold cyan]", expand=False))

    if df.empty:
        console.print("  [dim]No data.[/dim]\n")
        return

    rt = df["response_time_ms"]
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column("Metric", style="bold")
    t.add_column("Value")
    t.add_row("Min",    f"{rt.min():.0f} ms")
    t.add_row("p50",    f"{rt.median():.0f} ms")
    t.add_row("p95",    f"{rt.quantile(0.95):.0f} ms")
    t.add_row("Max",    f"{rt.max():.0f} ms")
    t.add_row("Mean",   f"{rt.mean():.0f} ms")
    console.print(t)


def _daily_volume(df: pd.DataFrame, days: int = 14) -> None:
    console.print(Panel(f"[bold cyan]6. DAILY VOLUME (last {days} days)[/bold cyan]", expand=False))

    if df.empty:
        console.print("  [dim]No data.[/dim]\n")
        return

    today = datetime.now(timezone.utc).date()
    cutoff = today - timedelta(days=days - 1)

    # Build a complete date range so days with 0 messages still appear
    all_dates = pd.date_range(start=cutoff, end=today, freq="D").date
    daily = df[df["date"] >= cutoff].groupby("date").size().reindex(all_dates, fill_value=0)

    t = Table(box=box.SIMPLE, padding=(0, 2))
    t.add_column("Date", style="bold")
    t.add_column("Messages", justify="right")
    t.add_column("Bar")

    max_count = daily.max() if daily.max() > 0 else 1
    for date, count in daily.items():
        bar_len = int(count / max_count * 20)
        bar = "[green]" + "#" * bar_len + "[/green]" + "." * (20 - bar_len)
        t.add_row(str(date), str(count), bar)
    console.print(t)


def _recent_interactions(df: pd.DataFrame, n: int = 10) -> None:
    console.print(Panel(f"[bold cyan]7. RECENT INTERACTIONS (last {n})[/bold cyan]", expand=False))

    if df.empty:
        console.print("  [dim]No interactions logged yet.[/dim]\n")
        return

    recent = df.sort_values("timestamp", ascending=False).head(n)

    t = Table(box=box.SIMPLE, padding=(0, 1))
    t.add_column("Time (UTC)", style="dim", width=17)
    t.add_column("Ch", width=3)
    t.add_column("Lang", width=4)
    t.add_column("Esc", width=3)
    t.add_column("Cks", width=3, justify="right")
    t.add_column("ms", width=6, justify="right")
    t.add_column("Question", max_width=45, no_wrap=True)

    for _, row in recent.iterrows():
        esc_marker = "[red]Y[/red]" if row["escalated"] else "[green]N[/green]"
        ch = "WA" if row["channel"] == "whatsapp" else "API"
        t.add_row(
            row["timestamp"].strftime("%m-%d %H:%M"),
            ch,
            row["language"],
            esc_marker,
            str(int(row["chunks_used"])),
            f"{row['response_time_ms']:.0f}",
            row["message"][:45],
        )
    console.print(t)


# ---------------------------------------------------------------------------
# Optional CSV export
# ---------------------------------------------------------------------------

def _write_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Writes the full log DataFrame to a CSV for further analysis."""
    export = df.drop(columns=["answer_preview"], errors="ignore")
    export.to_csv(output_path, index=False)
    console.print(f"\n[dim]CSV written to {output_path}[/dim]")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_report(log_path: Path = None, write_csv: bool = False) -> None:
    """
    Loads the pilot log and prints the full report.

    Parameters:
        log_path  : Path to the JSONL log file. Defaults to
                    evaluation.logger.LOG_FILE.
        write_csv : If True, also writes a CSV summary alongside the log.
    """
    if log_path is None:
        log_path = DEFAULT_LOG_FILE

    console.print("[bold white]" + "=" * 60 + "[/bold white]")
    console.print("[bold white]  Abiyamo SRH Chatbot -- Pilot Report[/bold white]")
    console.print("[bold white]" + "=" * 60 + "[/bold white]")
    console.print(f"[dim]Log file: {log_path}[/dim]\n")

    df = load_logs(log_path)

    if df.empty:
        console.print(
            "[yellow]No interactions found in the log file.[/yellow]\n"
            "The log is populated automatically as the chatbot handles messages.\n"
            "Run the server and send some test messages to populate it."
        )
        return

    _overview(df)
    _languages(df)
    _escalation(df)
    _pipeline(df)
    _performance(df)
    _daily_volume(df)
    _recent_interactions(df)

    if write_csv:
        csv_path = log_path.with_suffix(".csv")
        _write_csv(df, csv_path)

    console.print("[bold white]" + "=" * 60 + "[/bold white]")


# ---------------------------------------------------------------------------
# Synthetic preview (used when no real log exists yet)
# ---------------------------------------------------------------------------

def _run_preview_report() -> None:
    """
    Generates a small synthetic dataset and runs the report on it.
    Used when no real log exists so the report output can be verified
    without needing live pilot data.
    """
    import tempfile, json, random
    from datetime import datetime, timezone, timedelta

    random.seed(42)
    languages = ["en", "en", "en", "ha", "yo", "ig"]
    messages = [
        "What is family planning?",
        "How do condoms work?",
        "I think I might be pregnant",
        "Menene hana haihuwa?",
        "Kini idena ibi?",
        "Gini bu nzere ime ime?",
        "What are the signs of STIs?",
        "How can I prevent HIV?",
        "I have been feeling sick for weeks",
        "What is the safe period method?",
    ]

    tmp = tempfile.NamedTemporaryFile(
        suffix=".jsonl", delete=False, mode="w", encoding="utf-8"
    )
    now = datetime.now(timezone.utc)
    for i in range(40):
        lang = random.choice(languages)
        escalated = random.random() < 0.15
        record = {
            "timestamp": (now - timedelta(
                days=random.randint(0, 13),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            )).isoformat(),
            "user_id": f"user_{random.randint(1, 12):02d}",
            "language": lang,
            "message": random.choice(messages),
            "answer_preview": "This is a synthetic preview answer.",
            "escalated": escalated,
            "chunks_used": 0 if escalated else random.randint(1, 3),
            "response_time_ms": round(random.uniform(500, 4500), 1),
            "channel": random.choice(["whatsapp", "whatsapp", "api"]),
        }
        tmp.write(json.dumps(record) + "\n")
    tmp.close()

    generate_report(log_path=Path(tmp.name))

    import os
    os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a pilot study report from the chatbot interaction log."
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help=f"Path to JSONL log file (default: {DEFAULT_LOG_FILE})",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write a CSV export of the log alongside the report.",
    )
    args = parser.parse_args()

    # If no real log exists yet, generate synthetic data so the report
    # output can be visually verified without needing live pilot data.
    if not args.log.exists():
        console.print(
            f"[yellow]Log file not found at {args.log}.[/yellow]\n"
            "[dim]Generating synthetic test data for report preview...[/dim]\n"
        )
        _run_preview_report()
        sys.exit(0)

    generate_report(log_path=args.log, write_csv=args.csv)
