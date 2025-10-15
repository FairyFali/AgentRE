import asyncio
from typing import Union, Literal, Optional
import argparse
import os, sys
import json
import random
sys.path.append('./Agenttts')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root) 
print("Current working directory:", os.getcwd())

from swarm.graph.swarm import Swarm
from swarm.environment.operations.final_decision import MergingStrategy
from experiments.evaluator.evaluator import Evaluator
from experiments.evaluator.llm_optimizer import LLM_optimizer
from experiments.evaluator.datasets.mmlu_dataset import MMLUDataset
from datasets.MMLU.download import download


def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument('--mode', type=str, default='TextgradSwarm',
                        choices=['OptimizedSwarm','OptimizedLLMSwarm','DepthParallelSwarm','WidthChainSwarm','RandomOptimizedSwarm','TextgradSwarm','MaaoSwarm','BOSwarm'],
                        help="Mode of operation. Default is 'OptimizedSwarm'.")

    parser.add_argument('--num-truthful-agents', type=int, default=1,
                        help="Number of truthful agents. The total will be N truthful and N adversarial.")

    parser.add_argument('--num-iterations', type=int, default=10,
                        help="Number of optimization iterations. Default 200.")

    parser.add_argument('--model_names', type=str, default="llama3.2-1b-longcontext:latest",
                        help="Model name, None runs the default ChatGPT4.")

    parser.add_argument('--domain', type=str, default="mmlus",
                        help="Domain (the same as dataset name), default ''")

    parser.add_argument('--debug', action='store_true', default=False,
                        help="Set for a quick debug cycle")
    
    parser.add_argument('--models_cfg',type=json.loads,default=None,help="JSON models cfg") # {"1b":1, "3b":2, "8b":0, "70b":0}

    parser.add_argument('--use_verifier',type=bool,default=False,help='use verifier or not')

    parser.add_argument('--budget',type=int,default=8,help="Base on 1b")

    parser.add_argument('--llm_rl',type=bool,default=True)

    args = parser.parse_args()
    return args


async def main():

    args = parse_args()
    domain: str = args.domain
    budget = args.budget
    enable_verifier = args.use_verifier
    enable_llm_rl = args.llm_rl
    models_cfg = args.models_cfg

    mode = Union[
        Literal['OptimizedSwarm'],
        Literal['OptimizedLLMSwarm'],
        Literal['DepthParallelSwarm'],
        Literal['WidthChainSwarm'],
        Literal['RandomOptimizedSwarm'],
        Literal['TextgradSwarm'],
        Literal['MaaoSwarm'],
        Literal['BOSwarm'],
    ]
    mode = args.mode

    if budget and mode in ("OptimizedSwarm","BOSwarm","TextgradSwarm","MaaoSwarm","RandomOptimizedSwarm"):
        models_cost = {"OFF":0, "llama3.2-1b-longcontext:latest":1,"llama3.2-3b-longcontext:latest":3, "llama3.1-8b-longcontext:latest":9, "llama3.1-70b-longcontext:latest":81,"gemma3-1b-longcontext:latest":1,"gemma1-2b-longcontext:latest":3,"gemma1-7b-longcontext:latest":9}
    elif budget and mode in ("OptimizedLLMSwarm"):
        models_cost = {"OFF":0, "llama3.2-1b-longcontext:latest":1,"llama3.2-3b-longcontext:latest":3, "llama3.1-8b-longcontext:latest":9, "llama3.1-70b-longcontext:latest":81}
    else:
        models_cost = None

    if enable_llm_rl:
        if mode=='OptimizedLLMSwarm' or mode=='OptimizedLatencySwarm':
            optimizer = LLM_optimizer(domain)
            print('Start model initialize with LLM')
            model_selection_list, agents_num = optimizer.model_initialize(budget,models_cost)
        if mode=='TextgradSwarm' or mode=='MaaoSwarm':
            optimizer = LLM_optimizer(domain)
            print('Start model initialize with LLM textgrad')
            model_selection, agents_num = optimizer.model_initialize_textgrad(budget,models_cost)
        enable_llm_rl = True


    debug: bool = args.debug
    model_names = [e.strip() for e in args.model_names.split(',')]

    strategy = MergingStrategy.SelectBest

    domain: str = args.domain
    if mode == 'RandomOptimizedSwarm':
        optimizer = LLM_optimizer(domain)
        model_selection = optimizer.enumerate_model_combos_multi(budget=budget,models_cost=models_cost)
    if mode == 'BOSwarm':
        optimizer = LLM_optimizer(domain)
        model_selection = optimizer.enumerate_model_combos_multi(budget=budget,models_cost=models_cost)
        model_selection = random.choice(model_selection)

    print(args)

    if budget and enable_llm_rl:
        N = agents_num
    elif budget:
        N = args.budget
    else:
        N = args.num_truthful_agents

    agent_name_list = N * ["IO"] #+ M * ["IO"]
    if len(model_names) == 1:
        model_names = model_names * (N+1)
    print('model names', model_names)

    if budget:
        swarm_name = f"{budget}true"
    else:
        swarm_name = f"{N}true"

    swarm = Swarm(
            agent_name_list,
            domain,
            model_names=model_names,
            final_node_class="FinalDecision",
            final_node_kwargs=dict(strategy=strategy,use_verifier=enable_verifier),
            edge_optimize=True,
            models_cost=models_cost,
            use_verifier=enable_verifier
        )

    tag = f"{domain}_{swarm_name}_{strategy.name}_{mode}"


    download()

    dataset_train = MMLUDataset('dev')
    dataset_val = MMLUDataset('val')
    print('len(dataset_train)', len(dataset_train))
    print('len(dataset_val)', len(dataset_val))

    evaluator = Evaluator(
        swarm,
        dataset_train,
        dataset_val,
        model_name=model_names[-1],
        enable_tensorboard = mode=='OptimizedSwarm',
        enable_artifacts=True,
        tensorboard_tag=tag,
        models_cfg = models_cfg,
        enable_budget=budget)

    limit_questions = 5 if debug else 100

    if mode == 'DepthParallelSwarm':
        score = await evaluator.evaluate_swarm(
            mode='depth_parallel_swarm',
            limit_questions=limit_questions,
            eval_batch_size=1)
    elif mode == 'WidthChainSwarm':
        score = await evaluator.evaluate_swarm(
            mode='width_chain_swarm',
            limit_questions=limit_questions,
            eval_batch_size=1)
    elif mode == 'OptimizedSwarm':

        num_iters = 5 if debug else args.num_iterations

        lr = 0.1

        edge_probs = await evaluator.optimize_swarm(num_iters=num_iters, lr=lr, budget=budget)

        score = await evaluator.evaluate_swarm(
            mode='external_edge_probs',
            edge_probs=edge_probs,
            limit_questions=limit_questions,
            )
    elif mode == 'OptimizedLLMSwarm':

        num_iters = 5 if debug else args.num_iterations

        if not enable_llm_rl:
            raise RuntimeError("mode=OptimizedLLMSwarm requires --llm_rl=True")

        graph = await evaluator.optimize_llm_swarm(
            num_iters=num_iters,
            model_selection_list=model_selection_list,
            optimizer=optimizer
        )

        score = await evaluator.evaluate_swarm(
            mode='llm_swarm',
            limit_questions=limit_questions,
            graph=graph
        )
    elif mode == 'RandomOptimizedSwarm':
        num_iters = 5 if debug else args.num_iterations

        graph = await evaluator.optimize_random_swarm(num_iters=num_iters, model_selection_list=model_selection)

        score = await evaluator.evaluate_swarm(
            mode='llm_swarm',
            limit_questions=limit_questions,
            graph=graph
            )
    elif mode == 'TextgradSwarm':

        num_iters = 5 if debug else args.num_iterations

        graph = await evaluator.optimize_textgrad_swarm(num_iters=num_iters, model_selection=model_selection,optimizer=optimizer)

        score = await evaluator.evaluate_swarm(
            mode='llm_swarm',
            limit_questions=limit_questions,
            graph=graph
            )
    elif mode == 'MaaoSwarm':

        num_iters = 5 if debug else args.num_iterations

        edge_probs = await evaluator.optimize_maao_swarm(num_iters=num_iters, budget=budget ,model_selection=model_selection,llm_optimizer=optimizer)

        score = await evaluator.evaluate_swarm(
            mode='external_edge_probs',
            edge_probs=edge_probs,
            limit_questions=limit_questions,
            )
    elif mode=='BOSwarm':

        num_iters = 3

        graph = await evaluator.optimize_bayes_swarm(num_iters=num_iters, model_selection=model_selection)

        score = await evaluator.evaluate_swarm(
            mode='bos_swarm',
            limit_questions=limit_questions,
            graph=graph
            )
    else:
        raise Exception(f"Unsupported mode {mode}")

    print(f"Score: {score}")


if __name__ == "__main__":
    asyncio.run(main())
