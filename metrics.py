import torch
import numpy as np
from typing import List, Dict, Union, Any
from collections import defaultdict

class EvaluationEngine:
    def __init__(self, device: str = "cuda"):
        self.device = device

    def _compute_metrics_chunked(
        self, queries: torch.Tensor, candidates: torch.Tensor, 
        query_ids: List[str], candidate_ids: List[str], 
        k_list: List[int], chunk_size: int = 2048
    ) -> Dict[str, float]:
        Q = queries.shape[0]
        C = candidates.shape[0]
        max_k = max(k_list)
        
        q_ids_np = np.array(query_ids)
        c_ids_np = np.array(candidate_ids)

        total_recall = {k: 0.0 for k in k_list}
        total_mrr = 0.0
        total_ndcg = {k: 0.0 for k in k_list}
        valid_queries = 0

        ranks = torch.arange(1, max_k + 1, device=self.device).float()
        discounts = torch.log2(ranks + 1)

        for i in range(0, Q, chunk_size):
            end = min(i + chunk_size, Q)
            q_chunk = queries[i:end].to(self.device)
            sim_matrix = q_chunk @ candidates.T
            # O0 queries and O3 candidates are disjoint — no self-similarity masking needed
            target_mask_np = (q_ids_np[i:end, None] == c_ids_np[None, :])
            target_mask = torch.tensor(target_mask_np, device=self.device)
            has_target = target_mask.any(dim=1)
            
            sorted_indices = sim_matrix.argsort(dim=1, descending=True)
            sorted_targets = torch.gather(target_mask.int(), 1, sorted_indices)
            
            for k in k_list:
                hits_at_k = (sorted_targets[:, :k].sum(dim=1) > 0).float()
                total_recall[k] += hits_at_k[has_target].sum().item()

            first_hit_idx = (sorted_targets == 1).int().argmax(dim=1)
            mrr = 1.0 / (first_hit_idx.float() + 1.0)
            total_mrr += mrr[has_target].sum().item()

            num_actual_targets = target_mask.sum(dim=1)
            for k in k_list:
                dcg = (sorted_targets[:, :k] / discounts[:k]).sum(dim=1)
                idcg = torch.zeros_like(dcg)
                for j in range(end - i):
                    n_targets = min(k, int(num_actual_targets[j].item()))
                    if n_targets > 0:
                        idcg[j] = (1.0 / discounts[:n_targets]).sum()
                
                ndcg = torch.zeros_like(dcg)
                valid = idcg > 0
                ndcg[valid] = dcg[valid] / idcg[valid]
                total_ndcg[k] += ndcg[valid].sum().item()

            valid_queries += has_target.sum().item()

        if int(valid_queries) < Q:
            print(f"WARNING: {Q - int(valid_queries)}/{Q} queries had no matching candidate — excluded from all metrics. Check your data.")

        res = {"MRR": total_mrr / valid_queries if valid_queries > 0 else 0.0}
        for k in k_list:
            res[f"Recall@{k}"] = total_recall[k] / valid_queries if valid_queries > 0 else 0.0
            res[f"NDCG@{k}"] = total_ndcg[k] / valid_queries if valid_queries > 0 else 0.0

        return res

    def evaluate(
        self, results_dict: Dict[str, Any], 
        pool_sizes: List[Union[int, str]] =[50, 100, 200, 500, "global"],
        k_list: List[int] = [1, 5, 10], num_trials: int = 100
    ) -> Dict[str, Any]:
        
        all_embs = results_dict['embeddings'].to(self.device)
        all_ids = results_dict['ids']
        all_opts = results_dict['opts']

        o0_indices = [i for i, opt in enumerate(all_opts) if opt == 'O0']
        o3_indices = [i for i, opt in enumerate(all_opts) if opt == 'O3']

        o0_id_to_idx = {all_ids[i]: i for i in o0_indices}
        o3_id_to_idx = {all_ids[i]: i for i in o3_indices}

        paired_ids = list(set(o0_id_to_idx.keys()) & set(o3_id_to_idx.keys()))
        
        metrics_report = {}

        for pool_size in pool_sizes:
            if pool_size == "global":
                print("Running Global Evaluation (O0 queries vs ALL O3 candidates)...")
                q_idx = [o0_id_to_idx[fid] for fid in paired_ids]
                c_idx = o3_indices
                
                metrics = self._compute_metrics_chunked(
                    all_embs[q_idx], all_embs[c_idx],[all_ids[i] for i in q_idx], [all_ids[i] for i in c_idx], 
                    k_list
                )
                metrics_report["Global"] = metrics
                continue
                
            print(f"Running Pool Evaluation (Size: {pool_size}, Trials: {num_trials})...")
            trial_metrics = defaultdict(list)
            
            for trial in range(num_trials):
                if len(paired_ids) < pool_size:
                    sampled_ids = paired_ids
                else:
                    sampled_ids = np.random.choice(paired_ids, pool_size, replace=False)

                q_idx = [o0_id_to_idx[fid] for fid in sampled_ids]
                c_idx = [o3_id_to_idx[fid] for fid in sampled_ids]
                
                metrics = self._compute_metrics_chunked(
                    all_embs[q_idx], all_embs[c_idx], 
                    [all_ids[i] for i in q_idx],[all_ids[i] for i in c_idx], 
                    k_list
                )
                for k, v in metrics.items():
                    trial_metrics[k].append(v)
            
            metrics_report[f"Pool_{pool_size}"] = {k: float(np.mean(v)) for k, v in trial_metrics.items()}

        return metrics_report