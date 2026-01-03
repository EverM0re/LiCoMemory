#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process multi-session Q&A dataset into per-item folders organized by question type:

For each item, based on question_type:
  outdir/longmem_ssu/lm_1/Corpus.json   # single-session-user
  outdir/longmem_ssa/lm_1/Corpus.json   # single-session-assistant  
  outdir/longmem_ms/lm_1/Corpus.json    # multi-session
  outdir/longmem_tr/lm_1/Corpus.json    # temporal-reasoning
  outdir/longmem_ku/lm_1/Corpus.json    # knowledge-update
  outdir/longmem_ssp/lm_1/Corpus.json   # single-session-preference

Input format (summary):
[
  {
    "question_id": "...",
    "question_type": "single-session-user",  # Used for folder classification
    "question": "What degree did I graduate with?",
    "answer": "Business Administration",
    "question_date": "2023/05/30 (Tue) 23:40",
    "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/20 (Sat) 02:57"],
    "haystack_session_ids": ["sharegpt_yywfIrx_0", "85a1be56_1"],
    "haystack_sessions": [
        [ {"role":"user","content":"..."},
          {"role":"assistant","content":"..."} ],
        [ {"role":"user","content":"..."},
          {"role":"assistant","content":"..."} ]
    ],
    "answer_session_ids": ["answer_280352e9"]
  },
  ...
]

Usage:
  python longmem.py --input data.json --outdir ./out
"""

import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

DATE_RE = re.compile(r"^(\d{4}/\d{2}/\d{2})")

# Question type to folder suffix mapping
QUESTION_TYPE_MAPPING = {
    "single-session-user": "ssu",
    "single-session-assistant": "ssa", 
    "multi-session": "ms",
    "temporal-reasoning": "tr",
    "knowledge-update": "ku",
    "single-session-preference": "ssp"
}

def get_folder_suffix(question_type: str) -> str:
    """Get folder suffix for given question type."""
    return QUESTION_TYPE_MAPPING.get(question_type, "unknown")

def get_next_folder_index(parent_dir: str) -> int:
    """Get the next available index for lm_x folders in parent directory."""
    if not os.path.exists(parent_dir):
        return 1
    
    existing_folders = []
    for item in os.listdir(parent_dir):
        if os.path.isdir(os.path.join(parent_dir, item)) and item.startswith("lm_"):
            try:
                idx = int(item[3:])  # Extract number after "lm_"
                existing_folders.append(idx)
            except ValueError:
                continue
    
    return max(existing_folders) + 1 if existing_folders else 1

def extract_date_prefix(dt: Optional[str]) -> Optional[str]:
    """Extract 'YYYY/MM/DD' from strings like '2023/05/20 (Sat) 02:21'."""
    if not dt or not isinstance(dt, str):
        return None
    m = DATE_RE.match(dt.strip())
    return m.group(1) if m else None

def build_context(session_messages: Iterable[Dict[str, Any]]) -> str:
    """
    Build the 'context' string by concatenating parts like:
      "user": "..." "assistant": "..."
    No separators (match the provided example).
    NOTE: Do not json.dumps() content here to avoid double-escaping \n.
    """
    parts: List[str] = []
    for msg in session_messages:
        role = str(msg.get("role", ""))
        content = str(msg.get("content", ""))
        parts.append(f"\"{role}\": \"{content}\"")
    return "".join(parts)

def process_item(item: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    From one Q&A item, produce:
      - corpus_records: list of per-session dicts
      - question_record: single dict (for this question)
    """
    haystack_dates = item.get("haystack_dates") or []
    haystack_session_ids = item.get("haystack_session_ids") or []
    haystack_sessions = item.get("haystack_sessions") or []

    # Align by the shortest length of the three parallel arrays
    n = min(len(haystack_dates), len(haystack_session_ids), len(haystack_sessions))

    corpus_records: List[Dict[str, Any]] = []
    for i in range(n):
        session_time = extract_date_prefix(haystack_dates[i])
        session_id = haystack_session_ids[i]
        session_msgs = haystack_sessions[i] or []
        context = build_context(session_msgs)
        corpus_records.append({
            "session_time": session_time,
            "context": context,
            "session_id": session_id,
        })

    # Build question record
    question_text = item.get("question", "")
    answer_text = item.get("answer", "")
    question_type = item.get("question_type", "")
    question_time = item.get("question_date", "")  # Extract question_date as string

    answer_session_ids = item.get("answer_session_ids", [])
    if isinstance(answer_session_ids, list):
        if len(answer_session_ids) == 1:
            origin: Any = answer_session_ids[0]  # single string (match example)
        else:
            origin = answer_session_ids         # keep as list if multiple
    else:
        origin = answer_session_ids            # unexpected shape; pass through

    question_record = {
        "question": question_text,
        "answer": answer_text,
        "question_type": question_type,
        "question_time": question_time,  # Add question_time field
        "origin": origin,
        "label": answer_text,
    }

    return {"corpus": corpus_records, "question": [question_record]}

def write_ndjson(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    """Write NDJSON file (one JSON object per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Path to input dataset JSON file")
    parser.add_argument("--outdir", "-o", required=True, help="Output directory to place per-item folders")
    parser.add_argument("--basename", default="longmem", help="Folder base name prefix (default: longmem)")
    args = parser.parse_args()

    in_path = args.input
    out_root = args.outdir
    base = args.basename

    os.makedirs(out_root, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of items.")

    total_items = 0
    total_sessions = 0
    type_counts = {}

    for item in data:
        if not isinstance(item, dict):
            continue

        # Get question type and determine folder structure
        question_type = item.get("question_type", "")
        folder_suffix = get_folder_suffix(question_type)
        
        if folder_suffix == "unknown":
            print(f"Warning: Unknown question type '{question_type}', skipping item")
            continue

        # Create type-specific parent folder
        type_folder = os.path.join(out_root, f"{base}_{folder_suffix}")
        os.makedirs(type_folder, exist_ok=True)
        
        # Get next available index for this type
        next_idx = get_next_folder_index(type_folder)
        
        # Create the lm_x folder
        item_folder = os.path.join(type_folder, f"lm_{next_idx}")
        os.makedirs(item_folder, exist_ok=True)

        result = process_item(item)
        corpus_records = result["corpus"]
        question_records = result["question"]

        corpus_out = os.path.join(item_folder, "Corpus.json")
        question_out = os.path.join(item_folder, "Question.json")

        write_ndjson(corpus_out, corpus_records)
        write_ndjson(question_out, question_records)

        total_items += 1
        total_sessions += len(corpus_records)
        type_counts[question_type] = type_counts.get(question_type, 0) + 1
        
        print(f"[OK] {item_folder} -> {len(corpus_records)} sessions, 1 question (type: {question_type})")

    print(f"\nDone. Processed {total_items} items with {total_sessions} sessions in total.")
    print(f"Base folder: {out_root}, prefix: '{base}_'")
    print("Items by type:")
    for qtype, count in type_counts.items():
        suffix = get_folder_suffix(qtype)
        print(f"  {qtype} ({base}_{suffix}): {count} items")

if __name__ == "__main__":
    main()
