#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from collections import defaultdict, Counter
from enum import Enum
from typing import List, Any, Optional
import random
import re
import asyncio

from swarm.llm.format import Message
from swarm.graph import Node
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry, PromptSet
from swarm.llm import LLMRegistry, LLM
from swarm.utils.log import logger, swarmlog
from swarm.utils.globals import Cost
from swarm.environment.operations.operation_registry import OperationRegistry
from swarm.environment.tools.coding.python_executor import PyExecutor

random.seed(0)

class MergingStrategy(Enum):
    OutputsAsReferences = 0
    MajorityVote = 1
    RandomChoice = 2
    SelfConsistency = 3
    SelectBest = 5
    Verifier = 6


@OperationRegistry.register("FinalDecision")
class FinalDecision(Node):
    def __init__(self, 
                 domain: str,
                 model_name: Optional[str],
                 strategy: MergingStrategy,
                 operation_description: str = "Refer to all answers and give a final answer.", 
                 id=None,
                 use_verifier=False):
        super().__init__(operation_description, id, True)
        self.strategy: MergingStrategy = strategy
        self.domain: str = domain
        self.llm: LLM = LLMRegistry.get('llama3.2-3b-longcontext:latest')
        self.prompt_set: PromptSet = PromptSetRegistry.get(domain)
        self.role: str = self.prompt_set.get_role()
        self.constraint: str = self.prompt_set.get_constraint()
        self.use_verifier=use_verifier
        self.model_name = 'llama3.2-3b-longcontext:latest'

    @property
    def node_name(self):
        return self.__class__.__name__

    async def _execute(self, inputs: List[Any] = [], 
                       **kwargs) -> None:

        node_inputs = self.process_input(inputs)
        prompt = None
        response = None

       
        if self.strategy == MergingStrategy.SelectBest:  
            # This is different from MajorityVote because it is prompt-based.
            if len(inputs) == 0:
                raise Exception("No inputs is not supported for MajorityVote")
            
            question = inputs[0]["task"]
            answers = [input.get("output") for input in inputs]
            verified_answers = [input.get("verified_answer") for input in inputs]
            if self.use_verifier == False:
                # answers = [ans for ans in answers if len(ans)<1500]
                answers = [ans for ans in answers]
                prompt = self.prompt_set.get_select_best(question=question, solutions=answers)
            elif len(answers) > 3:
                if all(isinstance(ans, tuple) and len(ans) == 3 for ans in verified_answers):
                    verified_answers.sort(key=lambda x: -x[0])
                    top4 = verified_answers[:3]
                    verified_answers = [f"Answer: {ans}\nReason: {reason}" for score, reason, ans in top4]
                    verified_answers = [ans for ans in verified_answers if len(ans) < 1500]
                    prompt = self.prompt_set.get_select_best(question=question, solutions=verified_answers)
                else:
                    print("Invalid verified_answers format. Falling back to random selection.")
                    # verified_answers = random.sample(answers, k=4)
                    verified_answers = [ans for ans in answers if len(ans) < 1500] # [ans for ans in verified_answers if len(ans) < 1500]
                    prompt = self.prompt_set.get_select_best(question=question, solutions=verified_answers)
            else:
                # answers = [ans for ans in answers if len(ans) < 1500]
                answers = [ans for ans in answers]
                prompt = self.prompt_set.get_select_best(question=question, solutions=answers)
            message = [Message(role="system", content=f"You are a {self.role}. {self.constraint}"),
                    Message(role="user", content=prompt)]
            response,_,_ = await self.llm.agen(message)
            print(f"{len(answers)=}")

        else:
            logger.error(f"Error: does not support \"{self.strategy}\"!")

        executions = {"operation": self.node_name,
                            "task": inputs[0]["task"], 
                            "files": inputs[0]["files"],
                            "input": inputs, 
                            "subtask": prompt,
                            "output": response,
                            "format": "natural language",
                            "cost": None}

        self.memory.add(self.id, executions)
        self.log()
        return executions
        
