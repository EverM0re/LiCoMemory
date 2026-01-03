#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process LOCOMO dataset into per-group folders with corpus and question files.

The LOCOMO dataset contains 10 groups, each with:
- qa: list of question-answer pairs
- conversation: multi-session dialogues between speakers

For each group, this script creates:
  outdir/group_1/Corpus.json   # session data in NDJSON format
  outdir/group_1/Question.json # Q&A data in NDJSON format

Corpus format:
  - session_time: extracted from session_x_date_time
  - session_id: Dx format (D1, D2, D3, etc.)
  - context: concatenated speaker+text pairs

Question format:
  - question: from qa
  - answer: from qa
  - label: same as answer
  - origin: first two characters of evidence (e.g., "D1:3" -> "D1")
  - question_type: category number as string

Usage:
  python locomo.py --input locomo10.json --outdir ./dataset
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List

# Regex to extract date from formats like "1:56 pm on 8 May, 2023"
# We'll try to convert it to a standard format
DATE_RE = re.compile(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})")

MONTH_MAP = {
    "january": "01", "jan": "01",
    "february": "02", "feb": "02",
    "march": "03", "mar": "03",
    "april": "04", "apr": "04",
    "may": "05",
    "june": "06", "jun": "06",
    "july": "07", "jul": "07",
    "august": "08", "aug": "08",
    "september": "09", "sep": "09",
    "october": "10", "oct": "10",
    "november": "11", "nov": "11",
    "december": "12", "dec": "12"
}


def parse_date(date_str: str) -> str:
    """
    Parse date string like "1:56 pm on 8 May, 2023" to "2023/05/08".
    If parsing fails, return the original string.
    """
    if not date_str:
        return ""
    
    match = DATE_RE.search(date_str.lower())
    if match:
        day = match.group(1).zfill(2)
        month_name = match.group(2).lower()
        year = match.group(3)
        month = MONTH_MAP.get(month_name, "00")
        return f"{year}/{month}/{day}"
    
    return date_str


def extract_evidence_prefix(evidence_list: List[str]) -> List[str]:
    """
    Extract session IDs from evidence list.
    E.g., ["D1:3", "D2:5"] -> ["D1", "D2"]
    Only keep first two characters.
    """
    prefixes = []
    for ev in evidence_list:
        if isinstance(ev, str) and len(ev) >= 2:
            prefix = ev[:2]  # Take first two characters (e.g., "D1")
            if prefix not in prefixes:
                prefixes.append(prefix)
    return prefixes


def build_context(session_messages: List[Dict[str, Any]]) -> str:
    """
    Build context string by concatenating speaker and text.
    Format: "speaker": "text""speaker": "text"...
    If blip_caption exists, append it as "(attached is ...)"
    """
    parts = []
    for msg in session_messages:
        speaker = msg.get("speaker", "")
        text = msg.get("text", "")
        
        # If there's a blip_caption, append it to the text
        blip_caption = msg.get("blip_caption", "")
        if blip_caption:
            text = f"{text} (attached is {blip_caption})"
        
        parts.append(f'"{speaker}": "{text}"')
    return "".join(parts)


def process_group(group_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process a single group from LOCOMO dataset.
    Returns dict with 'corpus' and 'question' lists.
    """
    conversation = group_data.get("conversation", {})
    qa_list = group_data.get("qa", [])
    
    # Process conversation sessions
    corpus_records = []
    session_num = 1
    
    while True:
        session_key = f"session_{session_num}"
        date_key = f"session_{session_num}_date_time"
        
        if session_key not in conversation:
            break
        
        session_messages = conversation.get(session_key, [])
        session_date = conversation.get(date_key, "")
        
        # Parse and format the date
        session_time = parse_date(session_date)
        
        # Build context from messages
        context = build_context(session_messages)
        
        # Session ID is Dx where x is the session number
        session_id = f"D{session_num}"
        
        corpus_records.append({
            "session_time": session_time,
            "context": context,
            "session_id": session_id,
        })
        
        session_num += 1
    
    # Process Q&A
    question_records = []
    for qa_item in qa_list:
        question = qa_item.get("question", "")
        
        # Check if answer exists and is not empty
        answer = qa_item.get("answer", "")
        if not answer:  # If answer is empty, None, or doesn't exist
            answer = "Context insufficient to answer"
        else:
            # Convert answer to string if it's not already
            if not isinstance(answer, str):
                answer = str(answer)
        
        evidence = qa_item.get("evidence", [])
        category = qa_item.get("category", "")
        
        # Extract origin from evidence
        origin = extract_evidence_prefix(evidence)
        
        # If only one origin, use string instead of list
        if len(origin) == 1:
            origin = origin[0]
        
        question_records.append({
            "question": question,
            "answer": answer,
            "question_type": str(category),
            "origin": origin,
            "label": answer,
        })
    
    return {
        "corpus": corpus_records,
        "question": question_records
    }


def write_ndjson(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write NDJSON file (one JSON object per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Process LOCOMO dataset into group folders with corpus and question files"
    )
    parser.add_argument(
        "--input", "-i", 
        required=True, 
        help="Path to input LOCOMO JSON file"
    )
    parser.add_argument(
        "--outdir", "-o", 
        required=True, 
        help="Output directory for group folders"
    )
    args = parser.parse_args()
    
    in_path = args.input
    out_root = args.outdir
    
    # Create output directory
    os.makedirs(out_root, exist_ok=True)
    
    # Load data
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of groups.")
    
    total_groups = 0
    total_sessions = 0
    total_questions = 0
    
    # Process each group
    for idx, group_data in enumerate(data, start=1):
        if not isinstance(group_data, dict):
            continue
        
        # Create group folder
        group_folder = os.path.join(out_root, f"group_{idx}")
        os.makedirs(group_folder, exist_ok=True)
        
        # Process the group
        result = process_group(group_data)
        corpus_records = result["corpus"]
        question_records = result["question"]
        
        # Write files
        corpus_out = os.path.join(group_folder, "Corpus.json")
        question_out = os.path.join(group_folder, "Question.json")
        
        write_ndjson(corpus_out, corpus_records)
        write_ndjson(question_out, question_records)
        
        total_groups += 1
        total_sessions += len(corpus_records)
        total_questions += len(question_records)
        
        print(f"[OK] {group_folder} -> {len(corpus_records)} sessions, {len(question_records)} questions")
    
    print(f"\nDone. Processed {total_groups} groups with {total_sessions} sessions and {total_questions} questions in total.")
    print(f"Output directory: {out_root}")


if __name__ == "__main__":
    main()

