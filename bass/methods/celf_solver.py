import heapq
import random

import torch


class CELFSelector:
    """
    Cost-aware lazy greedy selection (Algorithm 1) for the objective of Eq. (9),
        F_q(S) = F_rel(S) + lambda * F_reach(S),
        F_rel(S)   = sum_{v in S} Rel(v, q),
        F_reach(S) = sum_{u in V} Rel(u, q) * log(1 + sum_{v in S} Pi_{vu})   (Eq. 8).
    """

    def __init__(self, Pi, query_relevance, costs, lambda_param=1.0, use_relevance=True):
        """
        Args:
            Pi: (N, N) reachability matrix.
            query_relevance: (N,) Rel(v, q) >= 0.
            costs: (N,) positive visual-token cost c(v) of each event.
            lambda_param: weight of F_reach.
            use_relevance: include the F_rel term (disabled for the w/o F_rel ablation).
        """
        self.Pi = Pi.detach().double().cpu()
        self.rel = query_relevance.detach().double().cpu()
        self.costs = costs.detach().double().cpu()
        self.lambda_param = lambda_param
        self.rel_weight = 1.0 if use_relevance else 0.0
        self.N = len(self.costs)

    def objective_function(self, S):
        if not S:
            return 0.0
        idx = torch.tensor(S, dtype=torch.long)
        f_rel = self.rel_weight * self.rel[idx].sum()
        reach = self.Pi[idx].sum(dim=0)
        f_reach = (self.rel * torch.log1p(reach)).sum()
        return float(f_rel + self.lambda_param * f_reach)

    def _marginal_gain(self, v, reach):
        """F_q(S + {v}) - F_q(S), given reach = sum_{u in S} Pi_{u, :}."""
        delta_reach = torch.log1p(reach + self.Pi[v]) - torch.log1p(reach)
        return float(self.rel_weight * self.rel[v] + self.lambda_param * (self.rel * delta_reach).sum())

    def select(self, budget):
        S = []
        spent = 0.0
        reach = torch.zeros(self.N, dtype=torch.float64)

        # Lines 1-7: initialize the queue with singleton densities
        queue = []
        for v in range(self.N):
            c_v = float(self.costs[v])
            if c_v <= budget:
                density = self._marginal_gain(v, reach) / c_v
                heapq.heappush(queue, (-density, v))

        # Lines 8-23: lazy evaluation
        while spent < budget and queue:
            _, u = heapq.heappop(queue)
            c_u = float(self.costs[u])
            if spent + c_u > budget:
                continue

            density = self._marginal_gain(u, reach) / c_u
            if not queue or density >= -queue[0][0]:
                if density <= 0:
                    break
                S.append(u)
                spent += c_u
                reach += self.Pi[u]
            else:
                heapq.heappush(queue, (-density, u))

        return S


def random_selection(costs, budget, seed=0):
    """
    Cost-feasible random baseline (Sec. 5.3): visits the events once in a random
    order and adds an event whenever it fits in the remaining budget.
    """
    order = list(range(len(costs)))
    random.Random(seed).shuffle(order)
    S, spent = [], 0.0
    for v in order:
        c_v = float(costs[v])
        if spent + c_v <= budget:
            S.append(v)
            spent += c_v
    return S
