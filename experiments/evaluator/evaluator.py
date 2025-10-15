import os
import asyncio
import pandas as pd
from typing import Iterable, Optional, Iterator, Union, Literal, List, Dict, Any, Tuple
from tqdm import tqdm
import torch
import time
import datetime
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import json
import math
import shutil
from pathlib import Path
import random
import re
import optuna
from optuna.samplers import TPESampler

from swarm.graph import Graph
from swarm.environment.agents import IO
from swarm.graph.swarm import Swarm
from experiments.evaluator.datasets.base_dataset import BaseDataset
from experiments.evaluator.accuracy import Accuracy
from swarm.optimizer.role_optimizer.role import Role
from swarm.llm import LLMRegistry


class Evaluator():
    def __init__(
            self,
            swarm: Optional[Swarm],
            train_dataset: BaseDataset,
            val_dataset: BaseDataset,
            model_name: Optional[str] = None,
            enable_tensorboard: bool = False,
            enable_budget = False,
            enable_artifacts: bool = False,
            tensorboard_tag: Optional[str] = None,
            models_cfg = None
        ) -> None:

        self._swarm: Optional[Swarm] = swarm
        self._train_dataset: BaseDataset = train_dataset
        self._val_dataset: BaseDataset = val_dataset
        self._model_name: Optional[str] = model_name
        self._models_cfg = models_cfg
        self.enable_budget = enable_budget

        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        art_dir_name = (f"{datetime_str}" +
                        (f"_{tensorboard_tag}" if tensorboard_tag is not None else ""))

        if enable_artifacts or enable_tensorboard:
            self._art_dir_name = os.path.join("runs", art_dir_name)
            os.makedirs(self._art_dir_name, exist_ok=True)
        else:
            self._art_dir_name = None

        if enable_tensorboard:
            self._logger = SummaryWriter(log_dir=self._art_dir_name)
        else:
            self._logger = None

    async def evaluate_swarm(
            self,
            mode: Union[
                Literal['external_edge_probs'],
                Literal['llm_swarm'],
                Literal['depth_parallel_swarm'],
                Literal['wdith_chain_swarm'],
                Literal['ablation_role_swarm']
                ],
            edge_probs: Optional[torch.Tensor] = None,
            limit_questions: Optional[int] = None,
            eval_batch_size: int = 1,
            set_models = False,
            graph = None
            ) -> float:

        assert self._swarm is not None

        dataset = self._val_dataset
        domain = dataset.get_domain()

        print(f"Evaluating swarm on {dataset.__class__.__name__} split {dataset.split}")

        realized_graph: Optional[Graph]
        
        if mode == 'depth_parallel_swarm':
            realized_graph = self._swarm.connection_dist.realize_depth_parallel(self._swarm.composite_graph,depth=2)
            realized_graph.display(draw=True, file_name='depth_parallel_swarm.html')
        elif mode == 'width_chain_swarm':
            realized_graph = self._swarm.connection_dist.realize_width_chain(self._swarm.composite_graph,width=2)
            # realized_graph = self._swarm.connection_dist.realize_custom_topology(self._swarm.composite_graph)
            realized_graph.display(draw=True, file_name='width_chain_swarm.html')
        elif mode == 'external_edge_probs' and self.enable_budget:
            assert edge_probs is not None
            edge_mask = edge_probs > 0.55
            realized_graph = self._swarm.connection_dist.realize_mask(
                self._swarm.composite_graph, edge_probs > 0.00)

            ranks = self._swarm.connection_dist.realize_ranks(
                realized_graph)
            chosen_models, kept_ids = self._swarm.connection_dist.greedy_models_in_budget(
                realized_graph,
                ranks=ranks,
                budget_left=self.enable_budget, 
            )
            kept_ids.add(realized_graph.decision_method.id)

            for nid, midx in chosen_models.items():
                node = realized_graph.find_node(nid)
                node.model_idx  = midx
                node.model_name = self._swarm.connection_dist.models[midx]
                if hasattr(node, "llm"):
                    node.llm = LLMRegistry.get(node.model_name)

            useless_nodes = [node for node in realized_graph.nodes if node not in kept_ids]
            realized_graph = self._swarm.connection_dist.realize_mask(realized_graph, edge_mask)
            realized_graph.delete_nodes(useless_nodes)
            realized_graph = self._swarm.connection_dist.make_all_kept_nodes_useful(realized_graph,kept_ids)

            removed = realized_graph.ensure_within_budget(self.enable_budget)

            max_try = 2000
            for _ in range(max_try):
                sampled_roles, _ = self._swarm.connection_dist.sample_roles(realized_graph,kept_ids)
                # 把没在 kept_ids 里的节点强制 IO
                for nid in realized_graph.nodes:
                    if nid not in kept_ids:
                        sampled_roles[nid] = Role.IO
                if realized_graph.check_role_constraints(sampled_roles,
                                                        io_idx=Role.IO,
                                                        reviewer_idx=Role.REV):
                    realized_graph.set_roles(sampled_roles)
                    break
            else:
                fallback_roles = {nid: Role.IO for nid in realized_graph.nodes}
                realized_graph.set_IO(fallback_roles)
                print("[Warn] role-sampling failed, fallback to all-IO.")

            realized_graph.display(draw=True, file_name='example.html')

            from swarm.utils.const import GPTSWARM_ROOT
            
            vis_name   = "example.html"
            vis_origin = Path(GPTSWARM_ROOT) / "result" / vis_name   # 已经存在的 html

            result_path = Path(self._art_dir_name)
            os.makedirs(result_path, exist_ok=True)
            file_name = 'role_example.txt'
            file_path  = result_path / file_name
            with open(file_path, "w", encoding="utf-8") as fp:
                for nid, node in realized_graph.nodes.items():
                    line = f"[{nid}]  {node.__class__.__name__}  role={getattr(node,'role_name', None)}"
                    print(line)          # 仍然在控制台打印
                    fp.write(line + "\n")

            file_name = 'models_example.txt'
            file_path  = result_path / file_name
            with open(file_path, "w", encoding="utf-8") as fp:
                for nid, node in realized_graph.nodes.items():
                    line = f"[{nid}]  {node.__class__.__name__}  models={getattr(node,'model_name', None)}"
                    print(line)          # 仍然在控制台打印
                    fp.write(line + "\n")

            shutil.copy2(vis_origin, Path(result_path) / vis_origin.name)
        elif mode == 'external_edge_probs':
            assert edge_probs is not None
            edge_mask = edge_probs > 0.55
            realized_graph = self._swarm.connection_dist.realize_mask(self._swarm.composite_graph, edge_mask)
            realized_graph = self._swarm.connection_dist.make_all_nodes_useful(realized_graph)

            max_try = 2000
            for _ in range(max_try):
                sampled_roles, _ = self._swarm.connection_dist.sample_roles(realized_graph)
                if realized_graph.check_role_constraints(sampled_roles, io_idx=Role.IO, reviewer_idx=Role.REV):
                    realized_graph.set_roles(sampled_roles)
                    break
            else:  # 连续 max_try 次都失败 → 全 IO 回退
                fallback_roles = {nid: Role.IO for nid in realized_graph.nodes}
                realized_graph.set_IO(fallback_roles)
                print("[Warn] role-sampling failed, fallback to all-IO.")

            realized_graph.display(draw=True, file_name='example.html')

            from swarm.utils.const import GPTSWARM_ROOT
            
            vis_name   = "example.html"
            vis_origin = Path(GPTSWARM_ROOT) / "result" / vis_name   # 已经存在的 html

            result_path = Path(self._art_dir_name)
            os.makedirs(result_path, exist_ok=True)
            file_name = 'role_example.txt'
            file_path  = result_path / file_name
            with open(file_path, "w", encoding="utf-8") as fp:
                for nid, node in realized_graph.nodes.items():
                    line = f"[{nid}]  {node.__class__.__name__}  role={getattr(node,'role_name', None)}"
                    print(line)          # 仍然在控制台打印
                    fp.write(line + "\n")

            shutil.copy2(vis_origin, Path(result_path) / vis_origin.name)
        elif mode == 'ablation_role_swarm':
            realized_graph = graph
            # realized_graph = self._swarm.connection_dist.realize_custom_topology(self._swarm.composite_graph)
            realized_graph.set_IO_graph()
            realized_graph.display(draw=True, file_name='example.html')

            from swarm.utils.const import GPTSWARM_ROOT
            
            vis_name   = "example.html"
            vis_origin = Path(GPTSWARM_ROOT) / "result" / vis_name   # 已经存在的 html

            result_path = Path(self._art_dir_name)
            os.makedirs(result_path, exist_ok=True)
            file_name = 'role_example.txt'
            file_path  = result_path / file_name
            with open(file_path, "w", encoding="utf-8") as fp:
                for nid, node in realized_graph.nodes.items():
                    line = f"[{nid}]  {node.__class__.__name__}  role={getattr(node,'role_name', None)}"
                    print(line)          # 仍然在控制台打印
                    fp.write(line + "\n")

            file_name = 'models_example.txt'
            file_path  = result_path / file_name
            with open(file_path, "w", encoding="utf-8") as fp:
                for nid, node in realized_graph.nodes.items():
                    line = f"[{nid}]  {node.__class__.__name__}  models={getattr(node,'model_name', None)}"
                    print(line)          # 仍然在控制台打印
                    fp.write(line + "\n")

            shutil.copy2(vis_origin, Path(result_path) / vis_origin.name)
        elif mode == 'llm_swarm':
            realized_graph = graph
            realized_graph.display(draw=True, file_name='example.html')

            from swarm.utils.const import GPTSWARM_ROOT
            
            vis_name   = "example.html"
            vis_origin = Path(GPTSWARM_ROOT) / "result" / vis_name   # 已经存在的 html

            result_path = Path(self._art_dir_name)
            os.makedirs(result_path, exist_ok=True)
            file_name = 'role_example.txt'
            file_path  = result_path / file_name
            with open(file_path, "w", encoding="utf-8") as fp:
                for nid, node in realized_graph.nodes.items():
                    line = f"[{nid}]  {node.__class__.__name__}  role={getattr(node,'role_name', None)}"
                    print(line)          # 仍然在控制台打印
                    fp.write(line + "\n")

            file_name = 'models_example.txt'
            file_path  = result_path / file_name
            with open(file_path, "w", encoding="utf-8") as fp:
                for nid, node in realized_graph.nodes.items():
                    line = f"[{nid}]  {node.__class__.__name__}  models={getattr(node,'model_name', None)}"
                    print(line)          # 仍然在控制台打印
                    fp.write(line + "\n")

            shutil.copy2(vis_origin, Path(result_path) / vis_origin.name)
        elif mode == 'bos_swarm':
            realized_graph = graph
            removed = realized_graph.ensure_within_budget(self.enable_budget)
            realized_graph.display(draw=True, file_name='example.html')

            from swarm.utils.const import GPTSWARM_ROOT
            
            vis_name   = "example.html"
            vis_origin = Path(GPTSWARM_ROOT) / "result" / vis_name   # 已经存在的 html

            result_path = Path(self._art_dir_name)
            os.makedirs(result_path, exist_ok=True)
            file_name = 'role_example.txt'
            file_path  = result_path / file_name
            with open(file_path, "w", encoding="utf-8") as fp:
                for nid, node in realized_graph.nodes.items():
                    line = f"[{nid}]  {node.__class__.__name__}  role={getattr(node,'role_name', None)}"
                    print(line)          # 仍然在控制台打印
                    fp.write(line + "\n")

            file_name = 'models_example.txt'
            file_path  = result_path / file_name
            with open(file_path, "w", encoding="utf-8") as fp:
                for nid, node in realized_graph.nodes.items():
                    line = f"[{nid}]  {node.__class__.__name__}  models={getattr(node,'model_name', None)}"
                    print(line)          # 仍然在控制台打印
                    fp.write(line + "\n")

            shutil.copy2(vis_origin, Path(result_path) / vis_origin.name)
        else:
            realized_graph = None

        accuracy = Accuracy()

        if mode not in ('external_edge_probs') and self._models_cfg:
            realized_graph.random_set_models(self._models_cfg)
            realized_graph.set_IO_graph()
            realized_graph.display(draw=True, file_name='chain_swarm.html')
        
        def eval_loader(batch_size: int) -> Iterator[List[Any]]:
            records = []
            for i_record, record in enumerate(dataset):
                if limit_questions is not None:
                    if i_record >= limit_questions:
                        break
                records.append(record)
                if len(records) >= batch_size:
                    yield records
                    records = []
            if len(records) > 0:
                yield records
            return

        data_len = min(len(dataset), limit_questions) if limit_questions is not None else len(dataset)
        num_batches = int(math.ceil(data_len / eval_batch_size))
        infence_time = 0

        for i_batch, record_batch in tqdm(enumerate(eval_loader(batch_size=eval_batch_size)), total=num_batches):
            print(80*'-')

            start_ts = time.time()

            future_answers = []
            for record in record_batch:

                input_dict = dataset.record_to_swarm_input(record)
                print('### input_dict', input_dict, '\n')

                future_answer = self._swarm.arun(input_dict, realized_graph)
                future_answers.append(future_answer)

            raw_answers = await asyncio.gather(*future_answers)

            print(f"Batch time {time.time() - start_ts:.3f}")
            batch_time = time.time() - start_ts
            infence_time += batch_time

            for raw_answer, record in zip(raw_answers, record_batch):
                input_dict = dataset.record_to_swarm_input(record)
                if domain == 'humaneval':
                    print("Raw answer:", raw_answer)
                    answer = dataset.postprocess_answer(raw_answer, input_dict)
                else:
                    print("Raw answer:", raw_answer)
                    answer = dataset.postprocess_answer(raw_answer)
                print("Postprocessed answer:", answer)
                correct_answer = dataset.record_to_target_answer(record)
                print('Correct answer:', correct_answer)
                if mode == '10_random_swarm':
                    cur_acc.update(answer,correct_answer)
                    accuracy.update(answer, correct_answer)
                    accuracy.print()
                else:
                    accuracy.update(answer, correct_answer)
                    accuracy.print()

            accuracy.print()
            print("Done!")

            self._dump_eval_results(dict(
                per_graph_accuracy = per_graph_acc,
                mean_accuracy      = accuracy.get(),
                variance           = var_acc,
                per_graph_time     = per_graph_time,
                limit_questions    = limit_questions,
                infence_time       = infence_time,
            ))

            from swarm.utils.const import GPTSWARM_ROOT
            
            for i in range(10):
                i+=1
                vis_name   = f"random_swarm_{i}.html"
                vis_origin = Path(GPTSWARM_ROOT) / "result" / vis_name

                result_path = Path(self._art_dir_name)
                os.makedirs(result_path, exist_ok=True)

                shutil.copy2(vis_origin, Path(result_path) / vis_origin.name)

            return accuracy.get()
        
        accuracy.print()
        print("Done!")
        
        self._dump_eval_results(dict(
            accuracy=accuracy.get(),
            limit_questions=limit_questions,
            infence_time=infence_time))

        return accuracy.get()

    def _dump_eval_results(self, dct: Dict[str, Any]) -> None:
        if self._art_dir_name is not None:
            eval_json_name = os.path.join(self._art_dir_name, "evaluation.json")
            with open(eval_json_name, "w") as f:
                json.dump(dct, f)

    def _print_conns(self, edge_probs: torch.Tensor, save_to_file: bool = False):
        assert self._swarm is not None
        msgs = []
        for i_conn, (conn, prob) in enumerate(zip(
                self._swarm.connection_dist.potential_connections, edge_probs)):
            src_id, dst_id = conn
            src_node = self._swarm.composite_graph.find_node(src_id)
            dst_node = self._swarm.composite_graph.find_node(dst_id)
            msg = (f"{i_conn}: src={src_node.node_name}({src_node.id}), "
                    f"dst={dst_node.node_name}({dst_node.id}), prob={prob.item():.3f}")
            msgs.append(msg+"\n")
            print(msg)
        if save_to_file:
            if self._art_dir_name is not None:
                txt_name = os.path.join(self._art_dir_name, "connections.txt")
                with open(txt_name, "w") as f:
                    f.writelines(msgs)
        return msgs
    
    def _print_conns_textgrad(self, edge_probs: torch.Tensor, kept_ids, save_to_file: bool = False):
        assert self._swarm is not None
        msgs = []
        for i_conn, (conn, prob) in enumerate(zip(
                self._swarm.connection_dist.potential_connections, edge_probs)):
            src_id, dst_id = conn
            if src_id not in kept_ids or dst_id not in kept_ids:
                continue
            src_node = self._swarm.composite_graph.find_node(src_id)
            dst_node = self._swarm.composite_graph.find_node(dst_id)
            msg = (f"{i_conn}: src={src_node.node_name}({src_node.id}), "
                    f"dst={dst_node.node_name}({dst_node.id}), prob={0.500}")
            msgs.append(msg+"\n")
            print(msg)
        if save_to_file:
            if self._art_dir_name is not None:
                txt_name = os.path.join(self._art_dir_name, "connections.txt")
                with open(txt_name, "w") as f:
                    f.writelines(msgs)
        return msgs

    def _print_conns_llm(self, edge_probs: torch.Tensor, kept_ids, save_to_file: bool = False):
        assert self._swarm is not None
        msgs = []
        for i_conn, (conn, prob) in enumerate(zip(
                self._swarm.connection_dist.potential_connections, edge_probs)):
            src_id, dst_id = conn
            if src_id not in kept_ids or dst_id not in kept_ids:
                continue
            src_node = self._swarm.composite_graph.find_node(src_id)
            dst_node = self._swarm.composite_graph.find_node(dst_id)
            msg = (f"{i_conn}: src={src_node.node_name}({src_node.id}), "
                    f"dst={dst_node.node_name}({dst_node.id}), prob={0.000}")
            msgs.append(msg+"\n")
            print(msg)
        if save_to_file:
            if self._art_dir_name is not None:
                txt_name = os.path.join(self._art_dir_name, "connections.txt")
                with open(txt_name, "w") as f:
                    f.writelines(msgs)
        return msgs
    
    def _print_roles(self,
                    role_probs: torch.Tensor,):
        msgs = []
        conn   = self._swarm.connection_dist
        idx2nid = {idx: nid for nid, idx in conn.node_id2idx.items()}   # 反向映射

        for row in range(role_probs.size(0)):
            nid   = idx2nid[row]
            node  = self._swarm.composite_graph.find_node(nid)
            probs = role_probs[row]

            msg = (f"{row}: node={node.node_name}({nid}), "
                f"IO={probs[Role.IO].item():.3f}, "
                f"REV={probs[Role.REV].item():.3f}")
            msgs.append(msg + "\n")
            print(msg)

        return msgs


    def _print_model_probs(self,
                        model_probs: torch.Tensor,):
        msgs = []
        conn      = self._swarm.connection_dist
        idx2nid   = {idx: nid for nid, idx in conn.node_id2idx.items()}
        model_names = conn.models                      # list[str]

        for row in range(model_probs.size(0)):
            nid   = idx2nid[row]
            node  = self._swarm.composite_graph.find_node(nid)
            probs = model_probs[row]

            pairs = ", ".join(f"({mn},{p.item():.3f})"
                            for mn, p in zip(model_names, probs))

            msg = (f"{row}: node={node.node_name}({nid}), probs=[{pairs}]")
            msgs.append(msg + "\n")
            print(msg)

        return msgs

    def parse_llm_prob_blocks(self, raw_text: str, eps: float = 1e-6):

        conn        = self._swarm.connection_dist
        nid2idx     = conn.node_id2idx   # {node_id: row_idx}
        N           = len(nid2idx)
        M           = len(conn.models)
        E           = len(conn.potential_connections)

        def _blk(h):
            m = re.search(rf"{h}:\s*\[(.*)\]", raw_text, flags=re.S)
            if not m:
                raise ValueError(f"{h} block not found")
            return m.group(1)

        edge_blk  = _blk("Edge-probabilities")
        role_blk  = _blk("Role-probabilities")

        # ---------- edge_logits ----------
        edge_logits = torch.zeros(E)
        e_pat = re.compile(r"(\d+):[\s\S]*?prob=([0-9.]+)")
        for idx, p in e_pat.findall(edge_blk):
            p = max(min(float(p), 1 - eps), eps)
            edge_logits[int(idx)] = math.log(p / (1.0 - p))

        # ---------- role_logits ----------
        role_logits = torch.full((N, 2), -float("inf"))
        r_pat = re.compile(
            r"\d+:\s+node=[^(]+\(([^)]+)\),\s+IO=([0-9.]+),\s+REV=([0-9.]+)"
        )
        for nid, io, rev in r_pat.findall(role_blk):
            row = nid2idx[nid]
            io, rev = [max(min(float(x), 1.0), eps) for x in (io, rev)]
            role_logits[row] = torch.log(torch.tensor([io, rev]))

        with torch.no_grad():
            conn.edge_logits .data.copy_(edge_logits)
            conn.role_logits .data.copy_(role_logits)

        edge_p  = torch.sigmoid(edge_logits)      # → 0.525, 0.475 …
        role_p  = torch.softmax(role_logits, -1)  # → 0.45 / 0.55 …
        return edge_p,role_p
    
    async def optimize_swarm(
            self,
            num_iters: int,
            lr: float,
            budget: None,
            batch_size: int = 3,
            ) -> torch.Tensor:

        assert self._swarm is not None

        dataset = self._train_dataset
        domain = dataset.get_domain()

        print(f"Optimizing swarm on {dataset.__class__.__name__} split {dataset.split}")

        optimizer = torch.optim.Adam(self._swarm.connection_dist.parameters(), lr=lr)

        if self._art_dir_name is not None:
            hp_json_name = os.path.join(self._art_dir_name, "hp.json")
            with open(hp_json_name, "w") as f:
                json.dump(dict(lr=lr,
                               batch_size=batch_size,
                               num_iters=num_iters,
                               model_name=self._model_name
                               ), f)

        def infinite_data_loader() -> Iterator[pd.DataFrame]:
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    record = dataset[idx.item()]
                    yield record

        loader = infinite_data_loader()

        sem = asyncio.Semaphore(3)  # 控制图同时执行

        async def safe_arun(input_dict, realized_graph):
            async with sem:
                return await self._swarm.arun(input_dict, realized_graph)

        edge_probs = None
        for i_iter in range(num_iters):
            print(f"Iter {i_iter}", 80*'-')

            start_ts = time.time()

            future_answers = []
            log_probs = []
            correct_answers = []
            batch_records = []
            for i_record, record in zip(range(batch_size), loader):

                realized_graph, log_prob = self._swarm.connection_dist.realize(
                    self._swarm.composite_graph,budget=budget
                    # temperature=3.0, # DEBUG
                    )

                input_dict = dataset.record_to_swarm_input(record)
                # answer = self._swarm.arun(input_dict, realized_graph)
                answer = safe_arun(input_dict, realized_graph)
                future_answers.append(answer)
                log_probs.append(log_prob)
                correct_answer = dataset.record_to_target_answer(record)
                correct_answers.append(correct_answer)
                batch_records.append(record)

            raw_answers = await asyncio.gather(*future_answers)

            print(f"Batch time {time.time() - start_ts:.3f}")
            batch_time = time.time() - start_ts

            loss_list: List[torch.Tensor] = []
            utilities: List[float] = []
            for raw_answer, log_prob, correct_answer, record in zip(raw_answers, log_probs, correct_answers, batch_records):
                input_dict = dataset.record_to_swarm_input(record)
                if domain == 'humaneval':
                    answer = dataset.postprocess_answer(raw_answer, input_dict)
                else:
                    answer = dataset.postprocess_answer(raw_answer)
                    assert isinstance(correct_answer, str), \
                        f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
                accuracy = Accuracy()
                accuracy.update(answer, correct_answer)
                utility = accuracy.get()
                utilities.append(utility)
                single_loss = - log_prob * utility
                loss_list.append(single_loss)

            print("utilities:", utilities)
            mean_utility = np.mean(np.array(utilities))
            total_loss = torch.mean(torch.stack(loss_list))

            print("loss:", total_loss.item())
            optimizer.zero_grad()
            total_loss.backward()
            print("Grad:", self._swarm.connection_dist.edge_logits.grad)
            optimizer.step()

            print("edge_logits:", self._swarm.connection_dist.edge_logits)
            edge_probs = torch.sigmoid(self._swarm.connection_dist.edge_logits)
            role_probs = torch.softmax(self._swarm.connection_dist.role_logits, dim=-1)
            print("edge_probs:", edge_probs)
            print("role_probs:", role_probs)

            self._print_conns(edge_probs)

            if self._logger is not None:
                self._logger.add_scalar("train/loss", total_loss.item(), i_iter)
                self._logger.add_scalar("train/utility", mean_utility.item(), i_iter)
                self._logger.add_scalar("batch_time", float(batch_time), i_iter)
            if self._art_dir_name is not None:
                log_jsonl_name = os.path.join(self._art_dir_name, "training.jsonl")
                with open(log_jsonl_name, "a") as f:
                    json.dump(dict(iter=i_iter, train_loss=total_loss.item(), train_utility=mean_utility.item(), batch_time=batch_time), f)
                    f.write("\n")
            print("end of iteration")

        if edge_probs is not None:
            self._print_conns(edge_probs, save_to_file=True)
        
        if self._art_dir_name is not None and num_iters > 1:
            log_jsonl_name = os.path.join(self._art_dir_name, "training.jsonl")
            total_time = 0.0
            with open(log_jsonl_name, "r") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if "batch_time" in record:
                            total_time += record["batch_time"]
                    except json.JSONDecodeError:
                        pass   # 防止某些非 JSON 行报错

            # 在日志最后追加一行 training time
            with open(log_jsonl_name, "a") as f:
                json.dump(dict(training_time=total_time), f)
                f.write("\n")

        print("Done!")
        edge_probs = torch.sigmoid(self._swarm.connection_dist.edge_logits)
        return edge_probs
    
    async def optimize_llm_swarm(
            self,
            num_iters: int,
            model_selection_list: json,
            optimizer,
            budget,
            batch_size: int = 3,
            ) -> torch.Tensor:

        assert self._swarm is not None

        dataset = self._train_dataset
        domain = dataset.get_domain()

        print(f"Optimizing llm swarm on {dataset.__class__.__name__} split {dataset.split}")   

        def infinite_data_loader() -> Iterator[pd.DataFrame]:
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    record = dataset[idx.item()]
                    yield record

        loader = infinite_data_loader()

        sem = asyncio.Semaphore(3)  # 控制图同时执行

        async def safe_arun(input_dict, realized_graph):
            async with sem:
                return await self._swarm.arun(input_dict, realized_graph)

        edge_probs = None
        flag = 0
        history_graph_list = []
        acc_list = []
        edge_probs_list = []
        response_list = []
        initial_graph_list,pre_model_selection = self._swarm.connection_dist.model_pretest_initialize(self._swarm.composite_graph,model_selection_list)
        pre_results = []
        for graph,sel in zip(initial_graph_list,pre_model_selection):
            pre_future_answers = []
            pre_correct_answers = []
            pre_batch_records = []
            for i_record, record in zip(range(1), loader):
                input_dict = dataset.record_to_swarm_input(record)
                answer = safe_arun(input_dict, graph)
                pre_future_answers.append(answer)
                pre_correct_answer = dataset.record_to_target_answer(record)
                pre_correct_answers.append(pre_correct_answer)
                pre_batch_records.append(record)

            raw_answers = await asyncio.gather(*pre_future_answers)

            utilities: List[float] = []
            for raw_answer, correct_answer, record in zip(raw_answers, pre_correct_answers, pre_batch_records):
                input_dict = dataset.record_to_swarm_input(record)
                if domain == 'humaneval':
                    answer = dataset.postprocess_answer(raw_answer, input_dict)
                else:
                    answer = dataset.postprocess_answer(raw_answer)
                accuracy = Accuracy()
                accuracy.update(answer, pre_correct_answer)
                utility = accuracy.get()
                utilities.append(utility)

            mean_utility = np.mean(np.array(utilities))
            pre_results.append((mean_utility,sel))
        best_acc = max(acc for acc, sel in pre_results)
        eps = 1e-8
        candidates = [sel for acc, sel in pre_results if acc-best_acc<eps]
        best_sel = random.choice(candidates)
        print('best_model_selection',best_sel)
        model_selection = best_sel

        if self._art_dir_name is not None:
            hp_json_name = os.path.join(self._art_dir_name, "hp.json")
            with open(hp_json_name, "w") as f:
                json.dump(dict(model_selection=model_selection,
                               batch_size=batch_size,
                               num_iters=num_iters,
                               ), f)


        for i_iter in range(num_iters):
            print(f"Iter {i_iter}", 80*'-')

            start_ts = time.time()

            future_answers = []
            correct_answers = []
            batch_records = []

            if flag == 0:
                    realized_graph, kept_ids = self._swarm.connection_dist.initialize_realize(
                        self._swarm.composite_graph,model_selection,temperature=1.0)
                    removed = realized_graph.ensure_within_budget(budget)

                    for sk in removed:
                        if model_selection[sk] > 0:
                            model_selection[sk] -= 1

                    if removed:
                        realized_graph, kept_ids = self._swarm.connection_dist.initialize_realize(
                            self._swarm.composite_graph, model_selection, temperature=1.0)
            else:
                    if response_list:
                        his_response = response_list
                    else:
                        his_response = ''
                    realized_graph, edge_probs, response = self._swarm.connection_dist.llm_realize(self._swarm.composite_graph,model_selection,history_graph_list,acc_list,edge_probs_list,his_response,optimizer)
                    removed = realized_graph.ensure_within_budget(budget)

                    for sk in removed:
                        if model_selection[sk] > 0:
                            model_selection[sk] -= 1

                    if removed:
                        realized_graph, edge_probs, response = self._swarm.connection_dist.llm_realize(self._swarm.composite_graph,model_selection,history_graph_list,acc_list,edge_probs_list,his_response,optimizer)


            for i_record, record in zip(range(batch_size), loader):
                input_dict = dataset.record_to_swarm_input(record)
                answer = safe_arun(input_dict, realized_graph)
                future_answers.append(answer)
                correct_answer = dataset.record_to_target_answer(record)
                correct_answers.append(correct_answer)
                batch_records.append(record)

            raw_answers = await asyncio.gather(*future_answers)
            

            print(f"Batch time {time.time() - start_ts:.3f}")
            batch_time = time.time() - start_ts

            utilities: List[float] = []
            for raw_answer, correct_answer, record in zip(raw_answers, correct_answers, batch_records):
                input_dict = dataset.record_to_swarm_input(record)
                if domain == 'humaneval':
                    answer = dataset.postprocess_answer(raw_answer, input_dict)
                else:
                    answer = dataset.postprocess_answer(raw_answer)
                    assert isinstance(correct_answer, str), \
                        f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
                accuracy = Accuracy()
                accuracy.update(answer, correct_answer)
                utility = accuracy.get()
                utilities.append(utility)

            print("utilities:", utilities)
            mean_utility = np.mean(np.array(utilities))

            print("edge_logits:", self._swarm.connection_dist.edge_logits)
            history_graph = realized_graph.graph_str(realized_graph,kept_ids)
            mean_acc = mean_utility
            history_graph_list.append(history_graph)
            acc_list.append(mean_acc)
            
            if flag == 0:
                edge_probs = torch.sigmoid(self._swarm.connection_dist.edge_logits)
                probs = self._print_conns_llm(edge_probs,kept_ids)
                edge_probs_list.append(probs)
            else:
                edge_probs_list.append(edge_probs)
                response_list.append(response)


            if self._logger is not None:
                self._logger.add_scalar("train/utility", mean_utility.item(), i_iter)
                self._logger.add_scalar("batch_time", float(batch_time), i_iter)
            if self._art_dir_name is not None:
                log_jsonl_name = os.path.join(self._art_dir_name, "training.jsonl")
                with open(log_jsonl_name, "a") as f:
                    json.dump(dict(iter=i_iter, train_utility=mean_utility.item(), batch_time=batch_time), f)
                    f.write("\n")
            flag = 1
            print("end of iteration")

        if self._art_dir_name is not None:
            log_jsonl_name = os.path.join(self._art_dir_name, "training.jsonl")
            total_time = 0.0
            with open(log_jsonl_name, "r") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if "batch_time" in record:
                            total_time += record["batch_time"]
                    except json.JSONDecodeError:
                        pass   # 防止某些非 JSON 行报错

            # 在日志最后追加一行 training time
            with open(log_jsonl_name, "a") as f:
                json.dump(dict(training_time=total_time), f)
                f.write("\n")

        print("Done!")
        return realized_graph

    async def optimize_random_swarm(
            self,
            num_iters: int,
            model_selection_list: list,
            budget,
            batch_size: int = 3,
            ) -> torch.Tensor:

        assert self._swarm is not None

        dataset = self._train_dataset
        domain = dataset.get_domain()

        print(f"Optimizing random swarm on {dataset.__class__.__name__} split {dataset.split}")   

        if self._art_dir_name is not None:
            hp_json_name = os.path.join(self._art_dir_name, "hp.json")
            with open(hp_json_name, "w") as f:
                json.dump(dict(
                               batch_size=batch_size,
                               num_iters=num_iters,
                               ), f)

        def infinite_data_loader() -> Iterator[pd.DataFrame]:
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    record = dataset[idx.item()]
                    yield record

        loader = infinite_data_loader()

        sem = asyncio.Semaphore(3)  # 控制图同时执行

        async def safe_arun(input_dict, realized_graph):
            async with sem:
                return await self._swarm.arun(input_dict, realized_graph)

        history_graph_list = []
        acc_list = []
        
        for i_iter in range(num_iters):
            print(f"Iter {i_iter}", 80*'-')

            start_ts = time.time()

            future_answers = []
            correct_answers = []
            batch_records = []

            model_selection = random.choice(model_selection_list)

            realized_graph, kept_ids = self._swarm.connection_dist.initialize_realize(
                        self._swarm.composite_graph,model_selection,temperature=1.0)
            realized_graph = self._swarm.connection_dist.make_all_nodes_useful(realized_graph)

            removed = realized_graph.ensure_within_budget(budget)

            for sk in removed:
                if model_selection[sk] > 0:
                    model_selection[sk] -= 1

            if removed:
                realized_graph, kept_ids = self._swarm.connection_dist.initialize_realize(
                        self._swarm.composite_graph,model_selection,temperature=1.0)
                realized_graph = self._swarm.connection_dist.make_all_nodes_useful(realized_graph)
            
            for i_record, record in zip(range(batch_size), loader):
                input_dict = dataset.record_to_swarm_input(record)
                answer = safe_arun(input_dict, realized_graph)
                future_answers.append(answer)
                correct_answer = dataset.record_to_target_answer(record)
                correct_answers.append(correct_answer)
                batch_records.append(record)

            raw_answers = await asyncio.gather(*future_answers)
            

            print(f"Batch time {time.time() - start_ts:.3f}")
            batch_time = time.time() - start_ts

            utilities: List[float] = []
            for raw_answer, correct_answer, record in zip(raw_answers, correct_answers, batch_records):
                input_dict = dataset.record_to_swarm_input(record)
                if domain == 'humaneval':
                    answer = dataset.postprocess_answer(raw_answer, input_dict)
                else:
                    answer = dataset.postprocess_answer(raw_answer)
                    assert isinstance(correct_answer, str), \
                        f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
                accuracy = Accuracy()
                accuracy.update(answer, correct_answer)
                utility = accuracy.get()
                utilities.append(utility)

            print("utilities:", utilities)
            mean_utility = np.mean(np.array(utilities))

            print("edge_logits:", self._swarm.connection_dist.edge_logits)
            mean_acc = mean_utility
            history_graph_list.append(realized_graph)
            acc_list.append(mean_acc)

            if self._logger is not None:
                self._logger.add_scalar("train/utility", mean_utility.item(), i_iter)
                self._logger.add_scalar("batch_time", float(batch_time), i_iter)
            if self._art_dir_name is not None:
                log_jsonl_name = os.path.join(self._art_dir_name, "training.jsonl")
                with open(log_jsonl_name, "a") as f:
                    json.dump(dict(iter=i_iter, train_utility=mean_utility.item(), batch_time=batch_time), f)
                    f.write("\n")
            print("end of iteration")

        if self._art_dir_name is not None:
            log_jsonl_name = os.path.join(self._art_dir_name, "training.jsonl")
            total_time = 0.0
            with open(log_jsonl_name, "r") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if "batch_time" in record:
                            total_time += record["batch_time"]
                    except json.JSONDecodeError:
                        pass   # 防止某些非 JSON 行报错

            # 在日志最后追加一行 training time
            with open(log_jsonl_name, "a") as f:
                json.dump(dict(training_time=total_time), f)
                f.write("\n")

        print("Done!")

        max_val = np.max(acc_list)
        max_indices = np.where(acc_list == max_val)[0]
        best_idx = int(random.choice(max_indices))

        best_graph = history_graph_list[best_idx]
        best_score = acc_list[best_idx]
        print(f"\n[Random-Opt] best_acc={best_score:.4f}  at iter {best_idx}")

        return best_graph

    async def optimize_textgrad_swarm(
            self,
            num_iters: int,
            model_selection: json,
            optimizer,
            budget,
            batch_size: int = 3,
            ) -> torch.Tensor:

        assert self._swarm is not None

        dataset = self._train_dataset
        domain = dataset.get_domain()

        print(f"Optimizing textgrad swarm on {dataset.__class__.__name__} split {dataset.split}")   

        if self._art_dir_name is not None:
            hp_json_name = os.path.join(self._art_dir_name, "hp.json")
            with open(hp_json_name, "w") as f:
                json.dump(dict(
                               batch_size=batch_size,
                               num_iters=num_iters,
                               ), f)

        def infinite_data_loader() -> Iterator[pd.DataFrame]:
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    record = dataset[idx.item()]
                    yield record

        loader = infinite_data_loader()

        sem = asyncio.Semaphore(3)  # 控制图同时执行

        async def safe_arun(input_dict, realized_graph):
            async with sem:
                return await self._swarm.arun(input_dict, realized_graph)

        edge_probs = None
        flag = 0
        history_graph_list = []
        acc_list = []
        edge_probs_list = []
        response_list = []

        for i_iter in range(num_iters):
            print(f"Iter {i_iter}", 80*'-')

            start_ts = time.time()

            future_answers = []
            correct_answers = []
            batch_records = []

            if flag == 0:
                    realized_graph, kept_ids = self._swarm.connection_dist.initialize_realize(
                        self._swarm.composite_graph,model_selection,temperature=1.0)

                    removed = realized_graph.ensure_within_budget(budget)

                    for sk in removed:
                        if model_selection[sk] > 0:
                            model_selection[sk] -= 1

                    if removed:
                        realized_graph, kept_ids = self._swarm.connection_dist.initialize_realize(
                            self._swarm.composite_graph, model_selection, temperature=1.0)
            else:
                    if response_list:
                        his_response = response_list
                    else:
                        his_response = ''
                    realized_graph, edge_probs, response = self._swarm.connection_dist.textgrad_realize(self._swarm.composite_graph,model_selection,history_graph_list,acc_list,edge_probs_list,his_response,optimizer)

                    removed = realized_graph.ensure_within_budget(budget)

                    for sk in removed:
                        if model_selection[sk] > 0:
                            model_selection[sk] -= 1

                    if removed:
                        realized_graph, edge_probs, response = self._swarm.connection_dist.textgrad_realize(self._swarm.composite_graph,model_selection,history_graph_list,acc_list,edge_probs_list,his_response,optimizer)


            for i_record, record in zip(range(batch_size), loader):
                input_dict = dataset.record_to_swarm_input(record)
                answer = safe_arun(input_dict, realized_graph)
                future_answers.append(answer)
                correct_answer = dataset.record_to_target_answer(record)
                correct_answers.append(correct_answer)
                batch_records.append(record)

            raw_answers = await asyncio.gather(*future_answers)
            

            print(f"Batch time {time.time() - start_ts:.3f}")
            batch_time = time.time() - start_ts

            utilities: List[float] = []
            for raw_answer, correct_answer, record in zip(raw_answers, correct_answers, batch_records):
                input_dict = dataset.record_to_swarm_input(record)
                if domain == 'humaneval':
                    answer = dataset.postprocess_answer(raw_answer, input_dict)
                else:
                    answer = dataset.postprocess_answer(raw_answer)
                    assert isinstance(correct_answer, str), \
                        f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
                accuracy = Accuracy()
                accuracy.update(answer, correct_answer)
                utility = accuracy.get()
                utilities.append(utility)

            print("utilities:", utilities)
            mean_utility = np.mean(np.array(utilities))

            print("edge_logits:", self._swarm.connection_dist.edge_logits)
            history_graph = realized_graph.graph_str(realized_graph,kept_ids)
            mean_acc = mean_utility
            history_graph_list.append(history_graph)
            acc_list.append(mean_acc)
            
            if flag == 0:
                edge_probs = torch.sigmoid(self._swarm.connection_dist.edge_logits)
                probs = self._print_conns_textgrad(edge_probs,kept_ids)
                edge_probs_list.append(probs)
            else:
                edge_probs_list.append(edge_probs)
                response_list.append(response)


            if self._logger is not None:
                self._logger.add_scalar("train/utility", mean_utility.item(), i_iter)
                self._logger.add_scalar("batch_time", float(batch_time), i_iter)
            if self._art_dir_name is not None:
                log_jsonl_name = os.path.join(self._art_dir_name, "training.jsonl")
                with open(log_jsonl_name, "a") as f:
                    json.dump(dict(iter=i_iter, train_utility=mean_utility.item(), batch_time=batch_time), f)
                    f.write("\n")
            flag = 1
            print("end of iteration")

        if self._art_dir_name is not None:
            log_jsonl_name = os.path.join(self._art_dir_name, "training.jsonl")
            total_time = 0.0
            with open(log_jsonl_name, "r") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if "batch_time" in record:
                            total_time += record["batch_time"]
                    except json.JSONDecodeError:
                        pass   # 防止某些非 JSON 行报错

            # 在日志最后追加一行 training time
            with open(log_jsonl_name, "a") as f:
                json.dump(dict(training_time=total_time), f)
                f.write("\n")

        print("Done!")
        return realized_graph

    async def optimize_maao_llm_swarm(
            self,
            num_iters: int,
            llm_optimizer,
            model_selection,
            edge_probs,
            budget,
            batch_size: int = 3,
            flag=0
            ):
        assert self._swarm is not None

        dataset = self._train_dataset
        domain = dataset.get_domain()

        print(f"Optimizing textgrad swarm on {dataset.__class__.__name__} split {dataset.split}")   

        if self._art_dir_name is not None:
            hp_json_name = os.path.join(self._art_dir_name, "hp.json")
            with open(hp_json_name, "w") as f:
                json.dump(dict(
                               batch_size=batch_size,
                               num_iters=num_iters,
                               ), f)

        def infinite_data_loader() -> Iterator[pd.DataFrame]:
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    record = dataset[idx.item()]
                    yield record

        loader = infinite_data_loader()

        sem = asyncio.Semaphore(3)  # 控制图同时执行

        async def safe_arun(input_dict, realized_graph):
            async with sem:
                return await self._swarm.arun(input_dict, realized_graph)

        cur_edge_prob = self._print_conns(edge_probs)
        cur_role_prob = torch.softmax(self._swarm.connection_dist.role_logits,dim=-1)
        cur_role_prob = self._print_roles(cur_role_prob)


        for i_iter in range(num_iters):
            print(f"Iter {i_iter}", 80*'-')

            start_ts = time.time()

            future_answers = []
            log_probs = []
            correct_answers = []
            batch_records = []
            for i_record, record in zip(range(batch_size), loader):

                realized_graph, log_prob = self._swarm.connection_dist.realize(
                    self._swarm.composite_graph,budget=budget
                    # temperature=3.0, # DEBUG
                    )

                input_dict = dataset.record_to_swarm_input(record)
                # answer = self._swarm.arun(input_dict, realized_graph)
                answer = safe_arun(input_dict, realized_graph)
                future_answers.append(answer)
                log_probs.append(log_prob)
                correct_answer = dataset.record_to_target_answer(record)
                correct_answers.append(correct_answer)
                batch_records.append(record)

            raw_answers = await asyncio.gather(*future_answers)

            utilities: List[float] = []
            for raw_answer, log_prob, correct_answer, record in zip(raw_answers, log_probs, correct_answers, batch_records):
                input_dict = dataset.record_to_swarm_input(record)
                if domain == 'humaneval':
                    answer = dataset.postprocess_answer(raw_answer, input_dict)
                else:
                    answer = dataset.postprocess_answer(raw_answer)
                    assert isinstance(correct_answer, str), \
                        f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
                accuracy = Accuracy()
                accuracy.update(answer, correct_answer)
                utility = accuracy.get()
                utilities.append(utility)

            print("utilities:", utilities)
            mean_utility = np.mean(np.array(utilities))
            kept_ids = [node for node in realized_graph.nodes]
            graph_str = realized_graph.graph_str(realized_graph,kept_ids)

            response = llm_optimizer.maao_forward(graph_str,mean_utility,cur_edge_prob,cur_role_prob)
            edge_p,role_p = self.parse_llm_prob_blocks(response)

            print(f"Batch time {time.time() - start_ts:.3f}")
            batch_time = time.time() - start_ts
            ev = self._print_conns(edge_p)
            rv = self._print_roles(role_p)

            if self._logger is not None:
                self._logger.add_scalar("train/utility", mean_utility.item(), i_iter)
                self._logger.add_scalar("batch_time", float(batch_time), i_iter)
            if self._art_dir_name is not None:
                log_jsonl_name = os.path.join(self._art_dir_name, "training.jsonl")
                with open(log_jsonl_name, "a") as f:
                    json.dump(dict(iter=i_iter, train_utility=mean_utility.item(), batch_time=batch_time), f)
                    f.write("\n")
            print("end of iteration")

        edge_probs = torch.sigmoid(self._swarm.connection_dist.edge_logits)
        if edge_probs is not None and flag==1:
            self._print_conns(edge_probs, save_to_file=True)
        
        if self._art_dir_name is not None and flag==1:
            log_jsonl_name = os.path.join(self._art_dir_name, "training.jsonl")
            total_time = 0.0
            with open(log_jsonl_name, "r") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if "batch_time" in record:
                            total_time += record["batch_time"]
                    except json.JSONDecodeError:
                        pass   # 防止某些非 JSON 行报错

            # 在日志最后追加一行 training time
            with open(log_jsonl_name, "a") as f:
                json.dump(dict(training_time=total_time), f)
                f.write("\n")

        print("Done!")
        return edge_probs,realized_graph

    async def optimize_bayes_swarm(
            self,
            num_iters: int,
            model_selection,
            batch_size: int = 10,
            ) -> torch.Tensor:

        assert self._swarm is not None

        dataset = self._train_dataset
        domain = dataset.get_domain()

        print(f"Optimizing bayes swarm on {dataset.__class__.__name__} split {dataset.split}")

        if self._art_dir_name is not None:
            hp_json_name = os.path.join(self._art_dir_name, "hp.json")
            with open(hp_json_name, "w") as f:
                json.dump(dict(
                               batch_size=batch_size,
                               num_iters=num_iters
                               ), f)

        def infinite_data_loader() -> Iterator[pd.DataFrame]:
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    record = dataset[idx.item()]
                    yield record

        loader = infinite_data_loader()

        base_graph = self._swarm.connection_dist.realize_bayes_models(self._swarm.composite_graph,model_selection=model_selection)

        conn_dist = self._swarm.connection_dist
        fd_id     = base_graph.decision_method.id
        kept_ids  = list(base_graph.nodes.keys())

        edge_vars = [f"e_{s}_{d}" for (s,d) in conn_dist.potential_connections
                    if s in kept_ids and d in kept_ids]
        role_vars = [f"r_{nid}"   for nid in kept_ids if nid != fd_id]

        sem = asyncio.Semaphore(3)
        async def safe_run(inp, g):
            async with sem:
                return await self._swarm.arun(inp, g)

        # ---------- 1. 目标函数 ----------
        async def obj_async(trial):
            # 1-a 采样
            start_ts = time.time()

            edges = {v: trial.suggest_int(v, 0, 1) for v in edge_vars}
            roles = {v: trial.suggest_categorical(v, [0,1]) for v in role_vars}
            roles[f"r_{fd_id}"] = 0                    # 决策节点固定 IO

            g = conn_dist.realize_bayes(base_graph,
                                            edges_sample=edges, roles_sample=roles)

            accs = []
            for i_record, record in zip(range(batch_size), loader):
                inp  = dataset.record_to_swarm_input(record)
                ans  = await safe_run(inp, g)
                if domain == 'humaneval':
                    ans = dataset.postprocess_answer(ans,inp)
                else:
                    ans = dataset.postprocess_answer(ans)
                
                tgt  = dataset.record_to_target_answer(record)
                A    = Accuracy(); A.update(ans, tgt)
                accs.append(A.get())

            batch_time = time.time() - start_ts

            if self._art_dir_name is not None:
                log_jsonl_name = os.path.join(self._art_dir_name, "training.jsonl")
                with open(log_jsonl_name, "a") as f:
                    json.dump(dict(train_utility=float(np.mean(accs)), batch_time=batch_time), f)
                    f.write("\n")
            return float(np.mean(accs))          # Optuna 最大化此值

        # ---------- 2. 运行 BO ----------
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=0))

        for _ in range(num_iters):
            await asyncio.to_thread(
                study.optimize,
                lambda t: asyncio.run(obj_async(t)),
                n_trials=1, catch=(asyncio.CancelledError,)
            )
        
        
        # ---------- 3. 输出最优图 ----------
        best = study.best_trial
        best_edges = {k:v for k,v in best.params.items() if k.startswith("e_")}
        best_roles = {k:v for k,v in best.params.items() if k.startswith("r_")}
        best_roles[f"r_{fd_id}"]=0
        best_graph = conn_dist.realize_bayes(base_graph,
                                                edges_sample=best_edges,
                                                roles_sample=best_roles)
        print(f"BayesOpt 最优 trial#{best.number}, acc={best.value:.3%}")
        if self._art_dir_name is not None:
            log_jsonl_name = os.path.join(self._art_dir_name, "training.jsonl")
            total_time = 0.0
            with open(log_jsonl_name, "r") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if "batch_time" in record:
                            total_time += record["batch_time"]
                    except json.JSONDecodeError:
                        pass   # 防止某些非 JSON 行报错
            with open(log_jsonl_name, "a") as f:
                json.dump(dict(training_time=total_time), f)
                f.write("\n")


        return best_graph
    
    async def optimize_maao_swarm(
            self,
            num_iters: int,
            budget: None,
            model_selection,
            llm_optimizer,
            batch_size: int = 3,
            ) -> torch.Tensor:
        assert self._swarm is not None

        dataset = self._train_dataset
        domain = dataset.get_domain()

        print(f"Optimizing maao swarm on {dataset.__class__.__name__} split {dataset.split}")

        async def pg_update(n_step):
            edge_prob = await self.optimize_swarm(
                num_iters=n_step,
                lr=0.1,
                budget=budget,
                batch_size=batch_size,
            )
            return edge_prob

        # LLM-TextGrad 同理，包一层
        async def llm_update(n_step,edge_prob,budget,flag):
            edge_prob,_ = await self.optimize_maao_llm_swarm(
                num_iters=n_step,
                llm_optimizer=llm_optimizer,
                model_selection=model_selection,
                edge_probs=edge_prob,
                budget=budget,
                batch_size=batch_size,
                flag=flag
            )
            return edge_prob

        flag = 0
        for cyc in range(int(num_iters/2)):
            # ① policy-gradient 段
            edge_prob = await pg_update(1)
            # ② LLM-proposal 段
            if cyc == int(num_iters/2)-1:
                flag = 1
            edge_prob = await llm_update(1,edge_prob=edge_prob,budget=budget,flag=flag)
        return edge_prob
