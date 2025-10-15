import glob
import pandas as pd
from typing import Dict, Union, List, Literal
import numpy as np
import os
import json
import re

from experiments.evaluator.datasets.base_dataset import BaseDataset, SwarmInput

def load_dataset(directory, span=1) -> Dict[str, List[str]]:
    data = {"question": [], "solution": [], "answer": []}
    with open(directory, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % span != 0:
                continue
            item = json.loads(line)
            question = item["question"]
            solution = item["answer"]

            # 使用正则表达式匹配 #### 后的最终答案
            match = re.search(r"####\s*(.+)", solution)
            final_answer = match.group(1).strip() if match else ""
            if final_answer == '':
                continue

            data["question"].append(question)
            data["solution"].append(solution)
            data["answer"].append(final_answer)

    return data

class Gsm8kDataset(BaseDataset):
    def __init__(self,
        split: Union[Literal['train'], Literal['val'], Literal['test']],
        ) -> None:

        self._split = split

        data_path = f"datasets/grade-school-math/grade_school_math/data/{self._split}.jsonl"
        self.data = load_dataset(data_path, span=10)
        self._total_df: pd.DataFrame = pd.DataFrame.from_dict(self.data)

        print("Total number of questions: ", len(self))

    @staticmethod
    def get_domain() -> str:
        return 'gsm8k'

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> Dict:
        record = self._total_df.iloc[index]
        assert isinstance(record, pd.DataFrame) or isinstance(record, pd.Series)
        return record

    @staticmethod
    def record_to_swarm_input(record: pd.DataFrame) -> SwarmInput:
        demo_question = (
            f"{record['question']}\n"
            )
        input_dict = {"task": demo_question}
        return input_dict
    
    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        # if len(answer) > 0:
        #     answer = answer[0] # Try to format the answer by taking the first letter

        matches = re.findall(r'\\boxed{([^}]*)}', answer)
        return matches[0] if matches else ""

    @staticmethod
    def record_to_target_answer(record: pd.Series) -> str:
        match = re.search(r"####\s*([^\n]+)", record["solution"])
        if match:
            return match.group(1).strip()
        raise ValueError("No final answer found in solution")
