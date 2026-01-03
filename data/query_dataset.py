import pandas as pd
from torch.utils.data import Dataset
import os
import json
from typing import List, Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger

class RAGQueryDataset(Dataset):

    def __init__(self, data_dir: str):
        super().__init__()

        self.corpus_path = os.path.join(data_dir, "Corpus.json")
        self.qa_path = os.path.join(data_dir, "Question.json")

        try:
            self.dataset = pd.read_json(self.qa_path, lines=True, orient="records")
            logger.info(f"Loaded QA dataset with {len(self.dataset)} questions")
        except Exception as e:
            logger.error(f"Failed to load QA dataset: {e}")
            self.dataset = pd.DataFrame()

    def get_corpus(self) -> List[Dict[str, Any]]:
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                corpus_data = json.load(f)

            corpus_list = []
            for i, doc in enumerate(corpus_data):
                corpus_list.append({
                    "context": doc.get("context", ""),
                    "session_time": doc.get("session_time", ""),
                    "session_id": doc.get("session_id", ""),
                    "doc_id": i,
                })

            logger.info(f"Loaded corpus with {len(corpus_list)} documents")
            return corpus_list

        except json.JSONDecodeError:
            try:
                corpus = pd.read_json(self.corpus_path, lines=True)
                corpus_list = []
                for i in range(len(corpus)):
                    row = corpus.iloc[i]
                    corpus_list.append({
                        "title": row.get("title", ""),
                        "content": row.get("context", ""),
                        "context": row.get("context", ""),
                        "session_time": row.get("session_time", ""),
                        "session_id": row.get("session_id", ""),
                        "doc_id": i,
                    })
                logger.info(f"Loaded corpus with {len(corpus_list)} documents (JSON Lines)")
                return corpus_list
            except Exception as e:
                logger.error(f"Failed to load corpus: {e}")
                return []

        except Exception as e:
            logger.error(f"Failed to load corpus: {e}")
            return []

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self.dataset):
            raise IndexError(f"Index {idx} out of range")

        question = self.dataset.iloc[idx]["question"]
        answer = self.dataset.iloc[idx]["answer"]

        other_attrs = self.dataset.iloc[idx].drop(["answer", "question"]).to_dict()

        return {
            "id": idx,
            "question": question,
            "answer": answer,
            **other_attrs
        }
