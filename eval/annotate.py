"""
eval/annotate.py — Terminal annotator for paragraph and media evaluation.

Usage:
    python eval/annotate.py text                  # eval/out/text/     (processed paragraphs)
    python eval/annotate.py text_raw              # eval/out/text_raw/  (raw Docling elements)
    python eval/annotate.py json_figures          # figures only, from full variant (same across all)
    python eval/annotate.py json_tables_full      # tables from hybrid (Docling + TATR) detector
    python eval/annotate.py json_tables_docling   # tables from Docling-only detector
    python eval/annotate.py json_tables_docling_recon  # tables from Docling recon detector

Keys:
    y / →    Correct
    n / ←    Incorrect
    s        Skip
    b        Back one item
    r        Show metrics so far
    q        Quit and save

Progress is saved to eval/out/annotations_{mode}.json after every keypress.
"""
from __future__ import annotations

import json
import os
import sys
import termios
import textwrap
import tty
from dataclasses import dataclass, field
from pathlib import Path

HERE     = Path(__file__).parent
OUT_DIR  = HERE / "out"

MAX_PER_DOC = 200   # skip documents with more than this many items (0 = no limit)

TERM_WIDTH = 82
WRAP_WIDTH = TERM_WIDTH - 4

# ── ANSI helpers ───────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN    = "\033[36m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CLEAR  = lambda: os.system("clear")


def c(color: str, text: str) -> str:
    return f"{color}{text}{RESET}"


# ── Item dataclass (shared across all modes) ───────────────────────────────────
@dataclass
class Item:
    doc_id:  str
    label:   str          # section name, "figure", "table", etc.
    index:   int          # global index
    text:    str          # main display text
    meta:    dict = field(default_factory=dict)  # extra fields (page, image_path, …)


# ── Parsers ────────────────────────────────────────────────────────────────────
def parse_text(path: Path) -> list[Item]:
    """Parse eval/out/text/*_text.txt — [Section] headers + plain paragraphs."""
    doc_id  = path.stem.replace("_text", "")
    section = ""
    items   = []

    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            section = s[1:-1]
        elif set(s) <= {"=", "-"} and len(s) > 4:
            pass
        elif s.startswith("Document:"):
            pass
        elif s:
            items.append(Item(doc_id=doc_id, label=section, index=0, text=s))

    return items


def parse_text_raw(path: Path) -> list[Item]:
    """Parse eval/out/text_raw/*_raw.txt — [SECTION_HEADER] / [TEXT] tags."""
    doc_id  = path.stem.replace("_raw", "")
    section = ""
    items   = []

    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("[SECTION_HEADER]"):
            section = s[len("[SECTION_HEADER]"):].strip()
        elif s.startswith("[TEXT]"):
            text = s[len("[TEXT]"):].strip()
            if text:
                items.append(Item(doc_id=doc_id, label=section, index=0, text=text))

    return items


def parse_json(path: Path, only: str | None = None) -> list[Item]:
    """Parse eval/out/json/*_media.json — figure and table detections.

    Args:
        only: ``"figures"`` or ``"tables"`` to restrict output; ``None`` for both.
    """
    data  = json.loads(path.read_text(encoding="utf-8"))
    pmcid = data["pmcid"]
    items: list[Item] = []

    kinds = ("figures", "tables") if only is None else (only,)
    for kind in kinds:
        item_type = kind.rstrip("s")  # "figure" / "table"
        for entry in data.get(kind, []):
            caption = (entry.get("caption") or "").replace("\n", " ")
            items.append(
                Item(
                    doc_id=pmcid,
                    label=item_type,
                    index=0,
                    text=caption or "(no caption)",
                    meta={
                        "detected_label": entry.get("label", ""),
                        "page":           entry.get("page", ""),
                        "image_path":     entry.get("image_path", ""),
                    },
                )
            )

    return items


def load_items(mode: str, max_per_doc: int = 0) -> list[Item]:
    from functools import partial
    loaders = {
        "text":                       (OUT_DIR / "text",                   "*_text.txt",   parse_text),
        "text_raw":                   (OUT_DIR / "text_raw",               "*_raw.txt",    parse_text_raw),
        # figures are identical across variants — annotate once from full
        "json_figures":               (OUT_DIR / "json" / "full",          "*_media.json", partial(parse_json, only="figures")),
        # tables differ per detector — annotate each separately
        "json_tables_full":           (OUT_DIR / "json" / "full",          "*_media.json", partial(parse_json, only="tables")),
        "json_tables_docling":        (OUT_DIR / "json" / "docling",       "*_media.json", partial(parse_json, only="tables")),
        "json_tables_docling_recon":  (OUT_DIR / "json" / "docling_recon", "*_media.json", partial(parse_json, only="tables")),
    }
    folder, glob, parser = loaders[mode]
    all_items: list[Item] = []
    skipped_docs: list[str] = []
    for f in sorted(folder.glob(glob)):
        doc_items = parser(f)
        if max_per_doc and len(doc_items) > max_per_doc:
            skipped_docs.append(f"{f.stem} ({len(doc_items)} items)")
            continue
        all_items.extend(doc_items)
    if skipped_docs:
        print(f"Skipped {len(skipped_docs)} docs exceeding --max-per-doc {max_per_doc}:")
        for name in skipped_docs:
            print(f"  {name}")
        print()
    for i, item in enumerate(all_items):
        item.index = i
    return all_items


# ── Annotation persistence ─────────────────────────────────────────────────────
def ann_path(mode: str) -> Path:
    return OUT_DIR / f"annotations_{mode}.json"


def load_annotations(mode: str) -> dict[str, str]:
    p = ann_path(mode)
    return json.loads(p.read_text()) if p.exists() else {}


def save_annotations(mode: str, ann: dict[str, str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ann_path(mode).write_text(json.dumps(ann, indent=2))


def ann_key(item: Item) -> str:
    image_path = item.meta.get("image_path", "").strip()
    if image_path:
        return Path(image_path).name
    detected = item.meta.get("detected_label", "").strip()
    if detected:
        return f"{item.doc_id}::{detected}"
    return f"{item.doc_id}::{item.label}::{item.index}"


# ── Input helpers ──────────────────────────────────────────────────────────────
def read_label(prompt: str = "  Label: ") -> str:
    """Restore normal terminal mode, read a line of text, return it."""
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)  # ensure normal mode
        print(prompt, end="", flush=True)
        return input()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def getch() -> str:
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            if ch2 == "[":
                if ch3 == "C":
                    return "RIGHT"
                if ch3 == "D":
                    return "LEFT"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ── Display ────────────────────────────────────────────────────────────────────
def render(item: Item, total: int, ann: dict[str, str], mode: str) -> None:
    _known      = {"correct", "incorrect", "other", "skipped"}
    n_correct   = sum(1 for v in ann.values() if v == "correct")
    n_incorrect = sum(1 for v in ann.values() if v == "incorrect")
    n_other     = sum(1 for v in ann.values() if v == "other")
    n_skipped   = sum(1 for v in ann.values() if v == "skipped")
    n_custom    = sum(1 for v in ann.values() if v not in _known)
    annotated   = n_correct + n_incorrect + n_other + n_custom
    pct         = annotated / total * 100 if total else 0

    CLEAR()
    print(c(BOLD, "─" * TERM_WIDTH))

    # Progress bar
    bar_width = TERM_WIDTH - 32
    filled = int(bar_width * annotated / total) if total else 0
    bar = "█" * filled + "░" * (bar_width - filled)
    print(
        f"  {c(CYAN, f'{annotated}/{total}')}  [{c(GREEN, bar)}]  {pct:.1f}%  "
        f"{c(GREEN, f'✓{n_correct}')}  {c(RED, f'✗{n_incorrect}')}  "
        f"{c(MAGENTA, f'?{n_other}')}  {c(CYAN, f'#{n_custom}')}  {c(DIM, f'~{n_skipped}')}"
    )
    print(c(BOLD, "─" * TERM_WIDTH))

    # Item metadata
    print(f"  {c(DIM, 'mode')}    {c(YELLOW, mode)}")
    print(f"  {c(DIM, 'doc')}     {c(BOLD, item.doc_id)}")
    print(f"  {c(DIM, 'label')}   {c(CYAN, item.label or '(top level)')}")
    if item.meta:
        if item.meta.get("detected_label"):
            print(f"  {c(DIM, 'item')}    {item.meta['detected_label']}  "
                  f"(page {item.meta.get('page', '?')})")
        if item.meta.get("image_path"):
            print(f"  {c(DIM, 'image')}   {c(DIM, item.meta['image_path'])}")
    print(f"  {c(DIM, '#')}       {item.index + 1} / {total}")
    print(c(BOLD, "─" * TERM_WIDTH))
    print()

    # Main text (wrapped)
    clean = item.text.replace("\r", " ")
    for line in textwrap.fill(clean, width=WRAP_WIDTH).splitlines():
        print(f"    {line}")

    print()
    print(c(BOLD, "─" * TERM_WIDTH))

    # Current annotation
    key = ann_key(item)
    if key in ann:
        val    = ann[key]
        colour = GREEN if val == "correct" else (RED if val == "incorrect" else (MAGENTA if val == "other" else (YELLOW if val == "skipped" else CYAN)))
        print(f"  current: {c(colour, val)}")
    else:
        print(f"  {c(DIM, 'not yet annotated')}")

    print()
    print(
        f"  {c(GREEN, '[y/→]')} correct   "
        f"{c(RED, '[n/←]')} incorrect   "
        f"{c(MAGENTA, '[o]')} other   "
        f"{c(CYAN, '[l]')} label   "
        f"{c(YELLOW, '[s]')} skip   "
        f"{c(BLUE, '[b]')} back   "
        f"{c(DIM, '[space]')} next   "
        f"{c(DIM, '[r]')} metrics   "
        f"{c(DIM, '[q]')} quit"
    )
    print(c(BOLD, "─" * TERM_WIDTH))


def show_metrics(items: list[Item], ann: dict[str, str], mode: str) -> None:
    known     = {"correct", "incorrect", "other", "skipped"}
    n_correct   = sum(1 for v in ann.values() if v == "correct")
    n_incorrect = sum(1 for v in ann.values() if v == "incorrect")
    n_other     = sum(1 for v in ann.values() if v == "other")
    custom_labels = {v: sum(1 for x in ann.values() if x == v)
                     for v in ann.values() if v not in known}
    n_custom    = sum(custom_labels.values())
    annotated   = n_correct + n_incorrect + n_other + n_custom
    total       = len(items)
    accuracy    = n_correct / annotated if annotated else 0.0

    CLEAR()
    print(c(BOLD, "─" * TERM_WIDTH))
    print(c(BOLD, f"  METRICS — mode: {mode}"))
    print(c(BOLD, "─" * TERM_WIDTH))
    print(f"  Total items   : {total}")
    print(f"  Annotated     : {annotated}")
    print(f"  Correct       : {c(GREEN,   str(n_correct))}")
    print(f"  Incorrect     : {c(RED,     str(n_incorrect))}")
    print(f"  Other         : {c(MAGENTA, str(n_other))}")
    for lbl, cnt in sorted(custom_labels.items()):
        print(f"  {lbl:<14}: {c(CYAN, str(cnt))}")
    print(f"  Accuracy      : {c(BOLD, f'{accuracy:.1%}')}")
    print()

    # Per-document breakdown
    by_doc: dict[str, dict[str, int]] = {}
    for item in items:
        key = ann_key(item)
        if key not in ann:
            continue
        by_doc.setdefault(item.doc_id, {"correct": 0, "incorrect": 0})
        by_doc[item.doc_id][ann[key]] = by_doc[item.doc_id].get(ann[key], 0) + 1

    if by_doc:
        print(c(BOLD, "  Per-document accuracy:"))
        for doc, counts in sorted(by_doc.items()):
            tot = counts.get("correct", 0) + counts.get("incorrect", 0)
            acc = counts.get("correct", 0) / tot if tot else 0
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            print(f"  {doc:<52} [{bar}] {acc:.0%}")

    print(c(BOLD, "─" * TERM_WIDTH))
    print(f"  {c(DIM, 'Press any key to continue...')}")
    getch()


# ── Main loop ──────────────────────────────────────────────────────────────────
def main() -> None:
    VALID_MODES = (
        "text", "text_raw",
        "json_figures",
        "json_tables_full", "json_tables_docling", "json_tables_docling_recon",
    )
    if len(sys.argv) < 2 or sys.argv[1] not in VALID_MODES:
        print("Usage: python eval/annotate.py [text | text_raw | json_figures | json_tables_full | json_tables_docling | json_tables_docling_recon]")
        sys.exit(1)

    mode  = sys.argv[1]
    items = load_items(mode, max_per_doc=MAX_PER_DOC)
    ann   = load_annotations(mode)
    total = len(items)

    if not items:
        print(f"No items found for mode '{mode}' in {OUT_DIR / mode}")
        sys.exit(1)

    # Resume from first un-annotated item
    cursor = next(
        (i for i, item in enumerate(items) if ann_key(item) not in ann),
        0,
    )

    while 0 <= cursor < total:
        item = items[cursor]
        render(item, total, ann, mode)
        key = getch()

        if key in ("y", "Y", "RIGHT"):
            ann[ann_key(item)] = "correct"
            save_annotations(mode, ann)
            cursor += 1
        elif key in ("n", "N", "LEFT"):
            ann[ann_key(item)] = "incorrect"
            save_annotations(mode, ann)
            cursor += 1
        elif key in ("o", "O"):
            ann[ann_key(item)] = "other"
            save_annotations(mode, ann)
            cursor += 1
        elif key in ("l", "L"):
            label = read_label().strip()
            if label:
                ann[ann_key(item)] = label
                save_annotations(mode, ann)
                cursor += 1
        elif key in ("s", "S"):
            ann[ann_key(item)] = "skipped"
            save_annotations(mode, ann)
            cursor += 1
        elif key == " ":
            cursor += 1
        elif key in ("b", "B"):
            cursor = max(0, cursor - 1)
        elif key in ("r", "R"):
            show_metrics(items, ann, mode)
        elif key in ("q", "Q", "\x03"):
            save_annotations(mode, ann)
            show_metrics(items, ann, mode)
            print(f"\n  Saved to {ann_path(mode)}\n")
            return

    # ── Second pass: revisit skipped items ────────────────────────────────────
    skipped_indices = [i for i, it in enumerate(items)
                       if ann.get(ann_key(it)) == "skipped"]
    if skipped_indices:
        CLEAR()
        print(f"\n  {len(skipped_indices)} skipped item(s) to revisit. Press any key...")
        getch()
        sk = 0
        while 0 <= sk < len(skipped_indices):
            item = items[skipped_indices[sk]]
            render(item, total, ann, mode)
            key = getch()
            if key in ("y", "Y", "RIGHT"):
                ann[ann_key(item)] = "correct"
                save_annotations(mode, ann)
                sk += 1
            elif key in ("n", "N", "LEFT"):
                ann[ann_key(item)] = "incorrect"
                save_annotations(mode, ann)
                sk += 1
            elif key in ("s", "S"):
                sk += 1  # keep as skipped, move on
            elif key == " ":
                sk += 1
            elif key in ("b", "B"):
                sk = max(0, sk - 1)
            elif key in ("r", "R"):
                show_metrics(items, ann, mode)
            elif key in ("q", "Q", "\x03"):
                save_annotations(mode, ann)
                show_metrics(items, ann, mode)
                print(f"\n  Saved to {ann_path(mode)}\n")
                return

    save_annotations(mode, ann)
    show_metrics(items, ann, mode)
    print(f"\n  All items reviewed! Saved to {ann_path(mode)}\n")


if __name__ == "__main__":
    main()
