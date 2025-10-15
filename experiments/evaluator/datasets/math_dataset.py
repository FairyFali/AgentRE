import glob
import pandas as pd
from typing import Dict, Union, List, Literal
import numpy as np
import os
import json
import re

from experiments.evaluator.datasets.base_dataset import BaseDataset, SwarmInput


def load_dataset(directory, span=1):
    data = {'question':[], 'level':[], 'type':[], 'solution':[]}  # List to store all data from JSON files
    i = 0
    # Walk through each directory and subdirectory
    for root, dirs, files in os.walk(directory):
        # layer by layer, find the dirs and files
        for file in files:
            if file.endswith('.json'):
                if i < span-1:
                    i += 1
                    continue
                i = 0
                file_path = os.path.join(root, file)  # Full path to the file
                # Open and load the JSON file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    data['question'].append(content.get('problem'))
                    data['level'].append(content.get('level'))
                    data['type'].append(content.get('type'))
                    data['solution'].append(content.get('solution'))

    return data

class MATHDataset(BaseDataset):
    def __init__(self,
        split: Union[Literal['train'], Literal['val'], Literal['test']],
        ) -> None:

        self._split = split

        data_path = f"datasets/MATH/{self._split}/"
        self.data = load_dataset(data_path, span=10)
        self._total_df: pd.DataFrame = pd.DataFrame.from_dict(self.data)

        print("Total number of questions: ", len(self))

    @staticmethod
    def get_domain() -> str:
        return 'math'

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
        return matches[-1] if matches else ""


    @staticmethod
    def record_to_target_answer(record: pd.DataFrame) -> str:
        correct_answer = record['solution']
        assert isinstance(correct_answer, str), (
            f"String expected but got {correct_answer} "
            f"of type {type(correct_answer)} (2)" \
            f" record={record}")
        matches = re.findall(r'\\boxed{([^}]*)}', correct_answer)
        return matches[0]
