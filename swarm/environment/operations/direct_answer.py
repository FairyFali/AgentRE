#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

import re
import random
import asyncio
import copy

from copy import deepcopy
from collections import defaultdict
from swarm.llm.format import Message
from swarm.graph import Node
from swarm.memory.memory import GlobalMemory
from typing import List, Any, Optional
from swarm.utils.log import logger, swarmlog
from swarm.utils.globals import Cost
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.llm.format import Message
from swarm.llm import LLMRegistry
from swarm.optimizer.node_optimizer import MetaPromptOptimizer
from swarm.environment.operations.final_decision import FinalDecision

class DirectAnswer(Node):
    '''
    直接回答是Node的一种类型
    '''
    def __init__(self, 
                 domain: str,
                 model_name: Optional[str],
                 operation_description: str = "Directly output an answer.",
                 max_token: int = 4096,
                 id=None,
                 use_constraint=False,
                 use_reviewer=False,
                 use_verifier=False):
        super().__init__(operation_description, id, True)
        self.domain = domain
        self.model_name = model_name
        self.llm = LLMRegistry.get(model_name)
        self.max_token = max_token
        self.prompt_set = PromptSetRegistry.get(domain)  # prompt_set也是提前注册好的
        self.role = self.prompt_set.get_role()
        self.constraint = self.prompt_set.get_constraint()
        self.use_constraint = use_constraint
        self.use_reviewer = use_reviewer
        self.role_idx = 0
        self.role_name = ''
        self.behavior_fn = self.io_behavior
        self.original_fn = self.io_behavior
        self.verifier = LLMRegistry.get('deepseek-v3')
        self.slm_rate = asyncio.Semaphore(1000000)
        self.llm_rate = asyncio.Semaphore(5)
        self.use_verifier = use_verifier



    @property
    def node_name(self):
        return self.__class__.__name__
    
    async def node_optimize(self, input, meta_optmize=False):
        task = input["task"]
        self.prompt_set = PromptSetRegistry.get(self.domain)
        role = self.prompt_set.get_role()
        constraint = self.prompt_set.get_constraint()

        if meta_optmize:
            update_role = role
            node_optmizer = MetaPromptOptimizer(self.domain, self.model_name, self.node_name)
            update_constraint = await node_optmizer.generate(init_prompt=task, init_role=role, init_constraint=constraint, tests=input)    #输入问题，role，constraint，input（测试样例）
            print('update_constraint',update_constraint)
            return update_role, update_constraint

        return role, constraint

    async def _execute(self,
                       inputs: List[Any] = [],
                       predecessor_outputs: List[Any] = [],
                       **kwargs):
        node_inputs = self.process_input(inputs)
        for input in node_inputs:  # 其实只有一个
            task = input["task"]
        previous_answers = [output.get('output') for output in predecessor_outputs]
        if self.use_verifier == False:
            # previous_answers = [ans for ans in previous_answers if len(ans)<1500]
            previous_answers = [ans for ans in previous_answers]
            return await self.behavior_fn(inputs,
                                        previous_answers,
                                        )
    
    async def io_behavior(self, inputs: List[Any] = [], predecessor_outputs: List[Any] = [], **kwargs):
        node_inputs = self.process_input(inputs)
        outputs = []
        has_predecessor = True if len(predecessor_outputs) > 0 else False
        for input in node_inputs:
            task = input["task"]
            previous_answers = predecessor_outputs
            if self.use_constraint:
                try:
                    role, constraint = await self.node_optimize(input, meta_optmize=False)
                except Exception as e:
                    print('❌ node_optimize 出错了:', e)
                    exit()
            else:
                role, constraint = await self.node_optimize(input, meta_optmize=False)
                if previous_answers: # 存在前置节点
                    if self.domain == 'math':
                        constraint = 'Make sure to put the answer (and only answer) inside \\boxed{}.'
                    else:
                        constraint = ""
                else: # 没有前置节点
                    if self.domain == 'humaneval':
                        constraint = "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
                    if self.domain == 'math' or self.domain == 'gsm8k' and self.use_constraint == False:
                        constraint = """You must provide the separator 'The answer is: ' before your final answer.
            # Make sure to put the answer (and only answer) inside \\boxed{}."""
                    else:
                        constraint = constraint
            if previous_answers:
                prompt = self.prompt_set.get_answer_prompt_refine_last_answers(task, previous_answers)
            else:
                prompt = self.prompt_set.get_answer_prompt(question=task)
            message = [Message(role="system", content=f"You are {role}. {constraint}"),
                       Message(role="user", content=prompt)]
            # print('### Log, message,', message, '\n')
            async with self.slm_rate:
                response, prompt_tokens, completion_tokens = await self.llm.agen(message, max_tokens=self.max_token)

            if self.use_verifier == False or self.use_constraint == False: # self.use_constraint == False
                verified_answer = '' #not use verifier ()


            if self.domain == 'humaneval':
                execution = {
                    "operation": self.node_name,
                    "task": task,
                    "files": input.get("files", []),
                    "input": task,
                    "role": role,
                    "constraint": constraint,
                    "prompt": prompt,
                    "output": response,#summary_response
                    "ground_truth": input.get("GT", []),
                    "format": "natural language",
                    "tests": input["tests"],
                    "verified_answer": verified_answer,
                    "cost": (prompt_tokens,completion_tokens)
                }
            else:
                execution = {
                    "operation": self.node_name,
                    "task": task,
                    "files": input.get("files", []),
                    "input": task,
                    "role": role,
                    "constraint": constraint,
                    "prompt": prompt,
                    "output": response,#summary_response
                    "ground_truth": input.get("GT", []),
                    "format": "natural language",
                    "verified_answer": verified_answer,
                    "cost": (prompt_tokens,completion_tokens)
                }
            outputs.append(execution)
            self.memory.add(self.id, execution)

        # self.log()
        return outputs 

    async def review_behavior(self, inputs: List[Any] = [], predecessor_outputs: List[Any] = [], **kwargs):
        role = self.role
        constraint = self.constraint
        node_inputs = self.process_input(inputs)
        outputs = []
        for input in node_inputs:  # 其实只有一个
            task = input["task"]
            response = predecessor_outputs
            prompt = self.prompt_set.get_reflect_prompt(question=task,answer_list=response)
            refine_message = [Message(role="system", content="You are a fusion agent in a multi-agent system. Your primary role is to carefully review , condense and synthesize the reasoning and answers produced by previous agents."),
                            Message(role="user", content=prompt)]
            async with self.slm_rate:
                refine_response, prompt_tokens, completion_tokens = await self.llm.agen(refine_message,  max_tokens=self.max_token)
            format_response = refine_response


            if self.use_verifier == False or self.use_constraint == False: # self.use_constraint == False
                verified_answer = '' #not use verifier ()
           
            if self.domain == 'humaneval':
                execution = {
                    "operation": self.node_name,
                    "task": task,
                    "files": input.get("files", []),
                    "input": task,
                    "role": role,
                    "constraint": constraint,
                    "prompt": prompt,
                    "output": format_response,#summary_response
                    "ground_truth": input.get("GT", []),
                    "format": "natural language",
                    "tests": input["tests"],
                    "verified_answer": verified_answer,
                    "cost": (prompt_tokens,completion_tokens)
                }
            else:
                execution = {
                    "operation": self.node_name,
                    "task": task,
                    "files": input.get("files", []),
                    "input": task,
                    "role": role,
                    "constraint": constraint,
                    "prompt": prompt,
                    "output": format_response,#summary_response
                    "ground_truth": input.get("GT", []),
                    "format": "natural language",
                    "verified_answer": verified_answer,
                    "cost": (prompt_tokens,completion_tokens)
                }
            outputs.append(execution)
            self.memory.add(self.id, execution)

        return outputs 
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k in ("prm_llm", "llm", "checker", "verifier"):
                setattr(new, k, v)  # 复用同一实例
            else:
                setattr(new, k, copy.deepcopy(v, memo))
        return new
            
        