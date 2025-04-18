

from typing import List, Optional, Tuple, Union, Dict
import numpy as np
import torch
from torch import Tensor
from loguru import logger
from Nexus.modules.score import MLPScorer, CosineScorer, EuclideanScorer, InnerProductScorer
import torch.nn.functional as F



__all__=['UniformSampler']

def kmeans(X, K_or_center, max_iter=300, verbose=False):
    N = X.size(0)
    if isinstance(K_or_center, int):
        K = K_or_center
        C = X[torch.randperm(N)[:K]]
    else:
        K = K_or_center.size(0)
        C = K_or_center
    prev_loss = np.inf
    for iter in range(max_iter):
        dist = torch.sum(X * X, dim=-1, keepdim=True) - 2 * \
            (X @ C.T) + torch.sum(C * C, dim=-1).unsqueeze(0)
        assign = dist.argmin(-1)
        assign_m = X.new_zeros(N, K)
        assign_m[(range(N), assign)] = 1
        loss = torch.sum(torch.square(X - C[assign, :])).item()
        if verbose:
            print(f'step:{iter:<3d}, loss:{loss:.3f}')
        if (prev_loss - loss) < prev_loss * 1e-6:
            break
        prev_loss = loss
        cluster_count = assign_m.sum(0)
        C = (assign_m.T @ X) / cluster_count.unsqueeze(-1)
        empty_idx = cluster_count < .5
        ndead = empty_idx.sum().item()
        C[empty_idx] = X[torch.randperm(N)[:ndead]]
    return C, assign, assign_m, loss


def construct_index(cd01, K):
    cd01, indices = torch.sort(cd01, stable=True)
    cluster, count = torch.unique_consecutive(cd01, return_counts=True)
    cluster = cluster.type(torch.long)
    count_all = torch.zeros(K + 1, dtype=torch.long, device=cd01.device)
    count_all[cluster + 1] = count
    indptr = count_all.cumsum(dim=-1)
    return indices, indptr


def uniform_sample_masked_hist(num_items: int, num_neg: int, user_hist: Tensor, num_query_per_user: int = None):
    """Sampling from ``1`` to ``num_items`` uniformly with masking items in user history.

    Args:
        num_items(int): number of total items.
        num_neg(int): number of negative samples.
        user_hist(torch.Tensor): items list in user interacted history. The shape are required to be ``[num_user(or batch_size),max_hist_seq_len]`` with padding item(with index ``0``).
        num_query_per_user(int, optimal): number of queries of each user. It will be ``None`` when there is only one query for one user.

    Returns:
        torch.Tensor: ``[num_user(or batch_size),num_neg]`` or ``[num_user(or batch_size),num_query_per_user,num_neg]``, negative item index. If ``num_query_per_user`` is ``None``,  the shape will be ``[num_user(or batch_size),num_neg]``.
    """
    # N: num_users, M: num_neg, Q: n_q, L: hist_len
    n_q = 1 if num_query_per_user is None else num_query_per_user
    num_user, hist_len = user_hist.shape
    device = user_hist.device
    neg_float = torch.rand(num_user, n_q*num_neg, device=device) # O(NMQ)
    non_zero_count = torch.count_nonzero(user_hist, dim=-1) # O(NQ)
    # BxNeg ~ U[1,2,...]
    neg_items = (torch.floor(neg_float * (num_items - non_zero_count).view(-1, 1))).long() + 1
    sorted_hist, _ = user_hist.sort(dim=-1) # BxL
    offset = torch.arange(hist_len, device=device).repeat(num_user, 1) # BxL
    offset = offset - (hist_len - non_zero_count).view(-1, 1)
    offset[offset < 0] = 0
    sorted_hist = sorted_hist - offset
    masked_offset = torch.searchsorted(sorted_hist, neg_items, right=True) # BxNeg O(NMQlogL)
    padding_nums = hist_len - non_zero_count
    neg_items += (masked_offset - padding_nums.view(-1, 1))
    if num_query_per_user is not None:
        neg_items = neg_items.reshape(num_user, num_query_per_user, num_neg)
    return neg_items


def uniform_sampling(
        num_queries: int,
        num_items: int,
        num_neg: int,
        user_hist: Tensor = None,
        device='cpu',
        backend='multinomial',  # ['numpy', 'torch']
    ):
    if user_hist is None:
        neg_idx = torch.randint(1, num_items, size=(num_queries, num_neg), device=device)
        return neg_idx
    else:
        device = user_hist.device
        if backend == 'multinomial':
            weight = torch.ones(size=(num_queries, num_items), device=device)
            _idx = torch.arange(user_hist.size(0), device=device).view(-1, 1).expand_as(user_hist)
            weight[_idx, user_hist] = 0.0
            neg_idx = torch.multinomial(weight, num_neg, replacement=True)
        elif backend == 'numpy':
            user_hist_np = user_hist.cpu().numpy()
            neg_idx_np = np.zeros(shape=(num_queries * num_neg))
            isin_id = np.arange(num_queries * num_neg)
            while len(isin_id) > 0:
                neg_idx[isin_id] = np.random.randint(1, num_items, len(isin_id))
                isin_id = torch.tensor(
                    [id for id in isin_id if neg_idx[id] in user_hist_np[id // num_neg]])
            neg_idx = torch.tensor(neg_idx_np, dtype=torch.long, device=device)
        elif backend == 'torch':
            neg_idx = user_hist.new_zeros(size=(num_queries * num_neg))
            isin_id = torch.arange(neg_idx.size(0), device=device)
            while len(isin_id) > 0:
                neg_idx[isin_id] = torch.randint(1, num_items, size=(len(isin_id)), device=device)
                isin_id = torch.tensor(
                    [id for id in isin_id if neg_idx[id] in user_hist[id // num_neg]], device=device)
        return neg_idx



class Sampler(object):
    def __init__(self, num_items, scorer_fn=None):
        super(Sampler, self).__init__()
        self.num_items = num_items
        self.scorer = scorer_fn

    def update(self, item_embs, max_iter=30):
        pass

    def compute_item_p(self, query, pos_items):
        pass

    def __call__(self, *args, **kwargs):
        pass


class UniformSampler(Sampler):
    """
    For each user, sample negative items
    """

    def __call__(
            self,
            query: Union[Tensor, int],
            num_neg: int,
            pos_items: Optional[Tensor] = None,
            device: Optional[torch.device] = None, *args, **kwargs
        ):
        # query: B x L x D, pos_items: B x L || query: B x D, pos_item: B x L || assume padding=0
        # not deal with reject sampling
        if isinstance(query, int):
            num_queries = query
            device = pos_items.device if pos_items is not None else device
            shape = (num_queries, )
        elif isinstance(query, Tensor):
            query = query
            num_queries = np.prod(query.shape[:-1])
            device = query.device
            shape = query.shape[:-1]

        with torch.no_grad():
            neg_items = torch.randint(0, self.num_items, size=(num_queries, num_neg), device=device)  # padding with zero
            neg_items = neg_items.reshape(*shape, -1)  # B x L x Neg || B x Neg
            neg_prob = self.compute_item_p(None, neg_items)
            if pos_items is not None:
                pos_prob = self.compute_item_p(None, pos_items)
                return pos_prob, neg_items, neg_prob
            else:
                return neg_items, neg_prob

    def compute_item_p(self, query, pos_items):
        return torch.zeros_like(pos_items)
    
    
class RetrieverSampler(Sampler):    # use for IRGAN
    def __init__(self, num_items, retriever=None, item_loader=None, method='brute', t=1):
        super().__init__(num_items)
        self.retriever = retriever
        self.item_loader = item_loader
        self.method = method
        self.T = t
        
    @torch.no_grad()
    def update(self, item_embs):
        logger.info(f'Update item vectors...')
        self.retriever.eval()
        all_item_vectors, all_item_ids = [], []
        for item_batch in self.item_loader:
            item_vector = self.retriever.item_encoder(item_batch)
            all_item_vectors.append(item_vector)
            all_item_ids.append(item_batch[self.retriever.fiid])
        all_item_vectors = torch.cat(all_item_vectors, dim=0)
        all_item_ids = torch.cat(all_item_ids, dim=0).cpu()
        self.retriever.item_vectors = all_item_vectors
        self.retriever.all_item_ids = all_item_ids

    def __call__(self, query, num_neg: Union[int, List, Tuple], pos_items=None, *args, **kwargs):
        if pos_items is not None:
            pos_prob, neg_items, neg_prob = self.retriever.sampling(query=query, num_neg=num_neg, pos_items=pos_items,
                                method=self.method)
            return pos_prob.detach(), neg_items.detach(), neg_prob.detach()
        else:
            neg_items, neg_prob = self.retriever.sampling(query=query, num_neg=num_neg, pos_items=pos_items,
                                method=self.method)
            return neg_items.detach(), neg_prob.detach()

    
    

    
class MaskedUniformSampler(Sampler):
    """
    For each user, sample negative items
    """

    def __call__(self, query, num_neg, pos_items=None, user_hist=None, *args, **kwargs):
        # query: B x L x D, pos_items: B x L || query: B x D, pos_item: B x L || assume padding=0
        # return BxLxN or BxN
        with torch.no_grad():
            if query.dim() == 2:
                neg_items = uniform_sample_masked_hist(
                    num_query_per_user=None, num_items=self.num_items,
                    num_neg=num_neg, user_hist=user_hist)
            elif query.dim() == 3:
                neg_items = uniform_sample_masked_hist(num_query_per_user=query.size(1),
                                                        num_items=self.num_items, num_neg=num_neg, user_hist=user_hist)
            else:
                raise ValueError("`query` need to be 2-dimensional or 3-dimensional.")

            neg_prob = self.compute_item_p(query, neg_items)
            if pos_items is not None:
                pos_prob = self.compute_item_p(query, pos_items)
                return pos_prob, neg_items, neg_prob
            else:
                return neg_items, neg_prob

    def compute_item_p(self, query, pos_items):
        return - torch.log(torch.ones_like(pos_items))
    


class PopularSampler(Sampler, torch.nn.Module):
    def __init__(self, pop_count, mode=0):
        Sampler.__init__(self, pop_count.shape[0])
        with torch.no_grad():
            # pop_count[0] is the default value
            pop_count = torch.tensor(pop_count, dtype=torch.float)
            pop_count = torch.cat([torch.tensor([1.]), pop_count])
            if mode == 0:
                pop_count = torch.log(pop_count + 1)
            elif mode == 1:
                pop_count = torch.log(pop_count + 1) + 1e-6
            elif mode == 2:
                pop_count = pop_count**0.75

            
            self.pop_prob = (pop_count / pop_count.sum())
            self.table = torch.cumsum(self.pop_prob, dim=0)
            # self.pop_prob[-1] = 1.0 # default pop_prob

    def __call__(self, query, num_neg, pos_items=None, *args, **kwargs):
        
        num_queries = np.prod(query.shape[:-1])
        seeds = torch.rand(num_queries, num_neg, device=query.device)
        self.table = self.table.to(query.device)
        neg_items = torch.searchsorted(self.table, seeds)
        # B x L x Neg || B x Neg
        neg_items = neg_items.reshape(*query.shape[:-1], -1) - 1
        neg_prob = self.compute_item_p(query, neg_items)
        if pos_items is not None:
            pos_prob = self.compute_item_p(query, pos_items)
            return pos_prob, neg_items, neg_prob
        else:
            return neg_items, neg_prob


    def compute_item_p(self, query, pos_items):
        self.pop_prob = self.pop_prob.to(pos_items.device)
        pos_items = torch.where(pos_items >= self.num_items, torch.tensor(-1), pos_items) + 1 # padding position: 0
        return torch.log(self.pop_prob[pos_items])  # padding value with log(0)
    
    
class MIDXUniformSampler(Sampler):
    """
    Uniform sampling for the final items
    """

    def __init__(self, num_items, num_clusters, scorer_fn=None):
        assert scorer_fn is None or not isinstance(scorer_fn, MLPScorer)
        super(MIDXUniformSampler, self).__init__(num_items, scorer_fn)
        self.K = num_clusters

    def update(self, item_embs, max_iter=30):
        if isinstance(self.scorer, CosineScorer):
            item_embs = F.normalize(item_embs, dim=-1)
        embs1, embs2 = torch.chunk(item_embs, 2, dim=-1)
        self.c0, cd0, cd0m, _ = kmeans(
            embs1, self.c0 if hasattr(self, 'c0') else self.K, max_iter)
        self.c1, cd1, cd1m, _ = kmeans(
            embs2, self.c1 if hasattr(self, 'c1') else self.K, max_iter)
        # for retreival probability, considering padding
        self.c0_ = torch.cat(
            [self.c0.new_zeros(1, self.c0.size(1)), self.c0], dim=0)
        # for retreival probability, considering padding
        self.c1_ = torch.cat(
            [self.c1.new_zeros(1, self.c1.size(1)), self.c1], dim=0)
        # for retreival probability, considering padding
        self.cd0 = torch.cat([-cd0.new_ones(1), cd0], dim=0) + 1
        # for retreival probability, considering padding
        self.cd1 = torch.cat([-cd1.new_ones(1), cd1], dim=0) + 1
        cd01 = cd0 * self.K + cd1
        self.indices, self.indptr = construct_index(cd01, self.K**2)
        self._update(item_embs, cd0m, cd1m)

    def _update(self, item_embs, cd0m, cd1m):
        if not isinstance(self.scorer, EuclideanScorer):
            self.wkk = cd0m.T @ cd1m
        else:
            norm = torch.exp(-0.5*torch.sum(item_embs**2, dim=-1))
            self.wkk = cd0m.T @ (cd1m * norm.view(-1, 1))
            # this is similar, to avoid log 0 !!! in case of zero padding
            self.p = torch.cat([norm.new_ones(1), norm], dim=0)
            self.cp = norm[self.indices]
            for c in range(self.K**2):
                start, end = self.indptr[c], self.indptr[c+1]
                if end > start:
                    cumsum = self.cp[start:end].cumsum(0)
                    self.cp[start:end] = cumsum / cumsum[-1]

    def __call__(self, query, num_neg, pos_items=None, *args, **kwargs):
        with torch.no_grad():
            if isinstance(self.scorer, CosineScorer):
                query = F.normalize(query, dim=-1)
            q0, q1 = query.view(-1, query.size(-1)).chunk(2, dim=-1)
            r1 = q1 @ self.c1.T.to(q1.device)
            r1s = torch.softmax(r1, dim=-1)  # num_q x K1
            r0 = q0 @ self.c0.T.to(q0.device)
            r0s = torch.softmax(r0, dim=-1)  # num_q x K0
            self.wkk = self.wkk.to(r1s.device)
            s0 = (r1s @ self.wkk.T) * r0s  # num_q x K0 | wkk: K0 x K1
            k0 = torch.multinomial(
                s0, num_neg, replacement=True)  # num_q x neg
            p0 = torch.gather(r0, -1, k0)     # num_q * neg
            subwkk = self.wkk[k0, :]          # num_q x neg x K1
            s1 = subwkk * r1s.unsqueeze(1)     # num_q x neg x K1
            k1 = torch.multinomial(
                s1.view(-1, s1.size(-1)), 1).squeeze(-1).view(*s1.shape[:-1])  # num_q x neg
            p1 = torch.gather(r1, -1, k1)  # num_q x neg
            k01 = k0 * self.K + k1  # num_q x neg
            p01 = p0 + p1
            neg_items, neg_prob = self.sample_item(k01, p01)
            neg_items = neg_items - 1 # padding position: 0, so output index should minus 1
            if pos_items is not None:
                pos_prob = None if pos_items is None else self.compute_item_p(
                    query, pos_items)
                return pos_prob, neg_items.view(*query.shape[:-1], -1), neg_prob.view(*query.shape[:-1], -1)
            else:
                return neg_items.view(*query.shape[:-1], -1), neg_prob.view(*query.shape[:-1], -1)

    def sample_item(self, k01, p01, pos=None):
        # TODO: remove positive items
        if not hasattr(self, 'cp'):
            # num_q x neg, the number of items
            self.indptr = self.indptr.to(k01.device)
            item_cnt = self.indptr[k01 + 1] - self.indptr[k01]
            item_idx = torch.floor(
                item_cnt * torch.rand_like(item_cnt.float())).int()  # num_q x neg
            self.indices = self.indices.to(item_idx.device)
            neg_items = self.indices[item_idx + self.indptr[k01]] + 1
            neg_prob = p01
            return neg_items, neg_prob
        else:
            return self._sample_item_with_pop(k01, p01)

    def _sample_item_with_pop(self, k01, p01):
        # k01 num_q x neg, p01 num_q x neg
        self.indptr = self.indptr.to(k01.device)
        start = self.indptr[k01]
        last = self.indptr[k01 + 1] - 1
        count = last - start + 1
        maxlen = count.max()
        fullrange = start.unsqueeze(-1) + torch.arange(
            maxlen, device=start.device).reshape(1, 1, maxlen)  # num_q x neg x maxlen
        fullrange = torch.minimum(fullrange, last.unsqueeze(-1))
        # @todo replace searchsorted with torch.bucketize
        self.cp = self.cp.to(fullrange.device)
        item_idx = torch.searchsorted(self.cp[fullrange], torch.rand_like(
            start.float()).unsqueeze(-1)).squeeze(-1)  # num_q x neg
        item_idx = torch.minimum(item_idx, last)
        self.indices = self.indices.to(item_idx.device)
        self.p = self.p.to(item_idx.device)
        neg_items = self.indices[item_idx + self.indptr[k01]]
        # plus 1 due to considering padding, since p include num_items + 1 entries
        neg_probs = self.p[item_idx + self.indptr[k01] + 1]
        return neg_items, p01 + torch.log(neg_probs)

    def compute_item_p(self, query, pos_items):
        # query: B x L x D, pos_items: B x L || query: B x D, pos_item: B x L1 || assume padding=0
        pos_items = torch.where(pos_items >= self.num_items, torch.tensor(-1), pos_items) + 1 # padding position: 0
        if pos_items.dim() == 1:
            pos_items_ = pos_items.unsqueeze(1)
        else:
            pos_items_ = pos_items
        self.cd0 = self.cd0.to(pos_items_.device)
        self.cd1 = self.cd1.to(pos_items_.device)
        self.c0_ = self.c0_.to(pos_items_.device)
        self.c1_ = self.c1_.to(pos_items_.device)
        k0 = self.cd0[pos_items_]  # B x L || B x L1
        k1 = self.cd1[pos_items_]  # B x L || B x L1
        c0 = self.c0_[k0, :]  # B x L x D || B x L1 x D
        c1 = self.c1_[k1, :]  # B x L x D || B x L1 x D
        q0, q1 = query.chunk(2, dim=-1)  # B x L x D || B x D
        if query.dim() == pos_items_.dim():
            r = (torch.bmm(c0, q0.unsqueeze(-1)) +
                torch.bmm(c1, q1.unsqueeze(-1))).squeeze(-1)  # B x L1
        else:
            r = torch.bmm(q0, c0.transpose(1, 2)) + \
                torch.bmm(q1, c1.transpose(1, 2))
            pos_items_ = pos_items_.unsqueeze(1)
        if not hasattr(self, 'p'):
            return r.view_as(pos_items)
        else:
            self.p = self.p.to(pos_items_.device)
            log_p = torch.where(pos_items_ < self.p.shape[0], torch.log(self.p[pos_items_]), torch.tensor(0.0))
            return (r + log_p).view_as(pos_items)
        
        
class MIDXPopSampler(MIDXUniformSampler):
    """
    Popularity sampling for the final items
    """

    def __init__(self, pop_count, num_clusters, scorer=None, mode=1):
        super(MIDXPopSampler, self).__init__(
            pop_count.shape[0], num_clusters, scorer)
        self.cp = None
        pop_count = torch.tensor(pop_count, dtype=torch.float)
        pop_count = torch.cat([torch.tensor([1.]), pop_count])
        if mode == 0:
            pop_count = torch.log(pop_count + 1)
        elif mode == 1:
            pop_count = torch.log(pop_count + 1) + 1e-6
        elif mode == 2:
            pop_count = pop_count**0.75
        self.pop_count = torch.nn.Parameter(pop_count, requires_grad=False)

    def _update(self, item_embs, cd0m, cd1m):
        if not isinstance(self.scorer, EuclideanScorer):
            norm = self.pop_count[1:]
        else:
            norm = self.pop_count[1:] * \
                torch.exp(-0.5*torch.sum(item_embs**2, dim=-1))
        self.wkk = cd0m.T @ (cd1m * norm.view(-1, 1))
        # self.p = torch.from_numpy(np.insert(pop_count, 0, 1.0))
        # # this is similar, to avoid log 0 !!! in case of zero padding
        # this is similar, to avoid log 0 !!! in case of zero padding
        self.p = torch.cat([norm.new_ones(1), norm], dim=0)
        self.cp = norm[self.indices]
        for c in range(self.K**2):
            start, end = self.indptr[c], self.indptr[c+1]
            if end > start:
                cumsum = self.cp[start:end].cumsum(0)
                self.cp[start:end] = cumsum / cumsum[-1]
                
    def sample_item(self, k01, p01, pos=None):
        if self.cp is not None:
            # num_q x neg, the number of items
            self.indptr = self.indptr.to(k01.device)
            item_cnt = self.indptr[k01 + 1] - self.indptr[k01]
            item_idx = torch.floor(
                item_cnt * torch.rand_like(item_cnt.float())).int()  # num_q x neg
            self.indices = self.indices.to(item_idx.device)
            neg_items = self.indices[item_idx + self.indptr[k01]] + 1
            neg_prob = p01
            return neg_items, neg_prob
        else:
            return self._sample_item_with_pop(k01, p01)
                
                
class ClusterUniformSampler(MIDXUniformSampler):
    def __init__(self, num_items, num_clusters, scorer_fn=None):
        assert scorer_fn is None or not isinstance(scorer_fn, MLPScorer)
        super(ClusterUniformSampler, self).__init__(
            num_items, num_clusters, scorer_fn)
        self.K = num_clusters
        self.cp = None

    def update(self, item_embs, max_iter=30):
        if isinstance(self.scorer, CosineScorer):
            item_embs = F.normalize(item_embs, dim=-1)
        self.c, cd, cdm, _ = kmeans(item_embs, self.K, max_iter)
        # for retreival probability, considering padding
        self.c_ = torch.cat(
            [self.c.new_zeros(1, self.c.size(1)), self.c], dim=0)
        # for retreival probability, considering padding
        self.cd = torch.cat([-cd.new_ones(1), cd], dim=0) + 1
        self.indices, self.indptr = construct_index(cd, self.K)
        self._update(item_embs, cdm)

    def _update(self, item_embs, cdm):
        if not isinstance(self.scorer, EuclideanScorer):
            self.wkk = cdm.sum(0)
        else:
            norm = torch.exp(-0.5*torch.sum(item_embs**2, dim=-1))
            self.wkk = (cdm * norm.view(-1, 1)).sum(0)
            # this is similar, to avoid log 0 !!! in case of zero padding
            self.p = torch.cat([norm.new_ones(1), norm], dim=0)
            self.cp = norm[self.indices]
            for c in range(self.K):
                start, end = self.indptr[c], self.indptr[c+1]
                if end > start:
                    cumsum = self.cp[start:end].cumsum()
                    self.cp[start:end] = cumsum / cumsum[-1]

    def __call__(self, query, num_neg, pos_items=None, *args, **kwargs):
        with torch.no_grad():
            if isinstance(self.scorer, CosineScorer):
                query = F.normalize(query, dim=-1)
            q = query.view(-1, query.size(-1))
            self.c = self.c.to(q.device)
            r = q @ self.c.T
            rs = torch.softmax(r, dim=-1)   # num_q x K
            k = torch.multinomial(rs, num_neg, replacement=True)    # num_q x neg
            p = torch.gather(r, -1, k)
            neg_items, neg_prob = self.sample_item(k, p)
            neg_items = neg_items - 1 # padding position: 0, so output index should minus 1
            if pos_items is not None:
                pos_prob = self.compute_item_p(query, pos_items)
                return pos_prob, neg_items.view(*query.shape[:-1], -1), neg_prob.view(*query.shape[:-1], -1)
            else:
                return neg_items.view(*query.shape[:-1], -1), neg_prob.view(*query.shape[:-1], -1)

    def compute_item_p(self, query, pos_items):
        # query: B x L x D, pos_items: B x L || query: B x D, pos_item: B x L1 || assume padding=0
        pos_items = torch.where(pos_items >= self.num_items, torch.tensor(-1), pos_items) + 1 # padding position: 0
        shape = pos_items.shape
        if pos_items.dim() == 1:
            pos_items = pos_items.view(-1, 1)
        self.cd = self.cd.to(pos_items.device)
        self.c_ = self.c_.to(pos_items.device)
        k = self.cd[pos_items]  # B x L || B x L1
        c = self.c_[k, :]  # B x L x D || B x L1 x D
        if query.dim() == pos_items.dim():
            r = torch.bmm(c, query.unsqueeze(-1)).squeeze(-1)  # B x L1
        else:
            r = torch.bmm(query, c.transpose(1, 2))  # B x L x L1
            pos_items = pos_items.unsqueeze(1)
        r = r.reshape(*shape)
        if not hasattr(self, 'p'):
            return r
        else:
            self.p = self.p.to(pos_items.device)
            return r + torch.log(self.p[pos_items])

    def sample_item(self, k01, p01):
        if self.cp is not None:
            self.indptr = self.indptr.to(k01.device)
            item_cnt = self.indptr[k01 + 1] - self.indptr[k01]
            item_idx = torch.floor(
                item_cnt * torch.rand_like(item_cnt.float())).int()  # num_q x neg
            self.indices = self.indices.to(item_idx.device)
            neg_items = self.indices[item_idx + self.indptr[k01] - 1] + 1
            neg_prob = p01
            return neg_items, neg_prob
        else:
            return self._sample_item_with_pop(k01, p01)

    def _sample_item_with_pop(self, k01, p01):
        # k01 num_q x neg, p01 num_q x neg
        self.indptr = self.indptr.to(k01.device)
        start = self.indptr[k01]
        last = self.indptr[k01 + 1] - 1
        count = last - start + 1
        maxlen = count.max()
        fullrange = start.unsqueeze(-1) + torch.arange(
            maxlen, device=start.device).reshape(1, 1, maxlen)  # num_q x neg x maxlen
        fullrange = torch.minimum(fullrange, last.unsqueeze(-1))
        # @todo replace searchsorted with torch.bucketize
        self.cp = self.cp.to(fullrange.device)
        item_idx = torch.searchsorted(self.cp[fullrange], torch.rand_like(
            start.float()).unsqueeze(-1)).squeeze(-1)  # num_q x neg
        item_idx = torch.minimum(item_idx, last)
        self.indices = self.indices.to(item_idx.device)
        self.p = self.p.to(item_idx.device)
        neg_items = self.indices[item_idx + self.indptr[k01]]
        # plus 1 due to considering padding, since p include num_items + 1 entries
        neg_probs = self.p[item_idx + self.indptr[k01] + 1]
        return neg_items, p01 + torch.log(neg_probs)
    
    
class ClusterPopSampler(ClusterUniformSampler):
    def __init__(self, pop_count, num_clusters, scorer=None, mode=1):
        super(ClusterPopSampler, self).__init__(
            pop_count.shape[0], num_clusters, scorer)
        pop_count = torch.tensor(pop_count, dtype=torch.float)
        pop_count = torch.cat([torch.tensor([1.]), pop_count])
        if mode == 0:
            pop_count = torch.log(pop_count + 1)
        elif mode == 1:
            pop_count = torch.log(pop_count + 1) + 1e-6
        elif mode == 2:
            pop_count = pop_count**0.75
        self.pop_count = torch.nn.Parameter(pop_count, requires_grad=False)

    def _update(self, item_embs, cdm):
        if not isinstance(self.scorer, EuclideanScorer):
            norm = self.pop_count[1:]
        else:
            norm = self.pop_count[1:] * \
                torch.exp(-0.5*torch.sum(item_embs**2, dim=-1))
        self.wkk = (cdm * norm.view(-1, 1)).sum(0)
        # this is similar, to avoid log 0 !!! in case of zero padding
        self.p = torch.cat([norm.new_ones(1), norm], dim=0)
        self.cp = norm[self.indices]
        for c in range(self.K):
            start, end = self.indptr[c], self.indptr[c+1]
            if end > start:
                cumsum = self.cp[start:end].cumsum(0)
                self.cp[start:end] = cumsum / cumsum[-1]
                
                
class LSHSampler(UniformSampler):

    def __init__(
            self, 
            num_items: int,
            n_dims: int,
            n_bits: int = 4,
            n_table: int = 16, 
            device: Union[str, torch.device]="cuda", 
            scorer_fn=None
        ):
        """
        LSH-based negative sampler proposed in 
        "A New Unbiased and Efficient Class of LSH-Based Samplers and Estimators for Partition Function Computation in Log-Linear Models".

        Args:
            num_items (int): number of items
            n_dims (int): dimension of item embedding vectors
            n_bits (int): number of hash functions in each hash table, i.e., K in the paper
            n_table (int): number of hash tables, i.e., L in the paper
            device (str or torch.device): device to save the hash functions, must be consistent with item embedding vectors
        """
        super().__init__(num_items, scorer_fn)
        if self.scorer is None:
            print("Scorer should be `InnerProductScorer` for LSHSampler, while got None.")
        else:
            assert type(self.scorer) == InnerProductScorer, f"Scorer should be `InnerProductScorer` for LSHSampler, while got {type(self.scorer)}."
        self.n_dims = n_dims
        self.n_bits = n_bits
        self.n_table = n_table
        self.device = device
        self.weight_vectors = self._generate_random_vectors(self.n_dims, self.n_bits, self.n_table) # DxKxL, normalized
        K_base_vec = torch.from_numpy(1 << np.arange(n_bits - 1, -1, -1)).type(torch.float).to(self.device)  # [K]
        self.K_base_vec = torch.nn.parameter.Parameter(K_base_vec, requires_grad=False)
        self.indptr, self.indices = None, None
        self.item_embs = None

    @torch.no_grad()
    def update(self, item_embs: torch.Tensor):
        norm_item_embs = item_embs / (torch.norm(item_embs, dim=1, keepdim=True) + 1e-10)
        y = torch.matmul(norm_item_embs.to(self.weight_vectors.device), self.weight_vectors.view(self.n_dims, -1)).view(item_embs.size(0), self.n_bits, -1)    # NxKxL
        y = (y > 0).type(torch.float)
        code = torch.matmul(y.transpose(1,2), self.K_base_vec)   # N, L
        self.item_embs = item_embs.clone().detach() 
        self.indices, self.indptr = self._construct_inverted_index(code)   # indptr: Lx(K+1); indices: LxN

    @torch.no_grad()
    def __call__(self, query, num_neg, pos_items=None, *args, **kwargs):
        """
        Sample negative items and calculate sampling probablity for correction.

        Args:
            query (torch.Tensor): shape of (B,D), query embedding. 
            num_neg (int): number of negative items to be sampled
            pos_items (torch.Tensor): shape of (B), positive item indexes.

        Returns:
            log_pos_prob (torch.Tensor): log sampling probability of positive items
            neg_id (torch.Tensor): sampled negative item indexes
            log_neg_prob (torch.Tensor): log sampling probability of negative items
        """
        # get hash code
        norm_query = query / (torch.norm(query, dim=-1, keepdim=True) + 1e-10)
        y = torch.matmul(norm_query, self.weight_vectors.view(self.n_dims, -1)).view(query.size(0), self.n_bits, -1)    # BxKxL
        y = (y > 0).type(torch.float)
        code = torch.matmul(y.transpose(1,2), self.K_base_vec).transpose(0, 1).type(torch.long)   # LxB
        start_idx = torch.gather(self.indptr, dim=1, index=code)    # LxB
        end_idx = torch.gather(self.indptr, dim=1, index=code+1)    # LxB
        num_candidates = (end_idx - start_idx)
        len_item = num_candidates.sum(dim=0) # B

        # for empty candidates, use uniform sampling
        empty_flag = (len_item == 0)
        if empty_flag.any():
            neg_id_empty, log_neg_prob_empty = super().__call__(query[empty_flag], num_neg, pos_items=None)

        cum_len = num_candidates.cumsum(dim=0).T.contiguous()
        rand_idx = torch.floor(torch.rand((query.size(0), num_neg), device=query.device) * len_item.view(-1, 1)).type(torch.long)   # B x neg
        rand_idx[rand_idx==len_item.view(-1,1)] = rand_idx[rand_idx==len_item.view(-1,1)] - 1   # in case of numerically unstable
        table_id = torch.searchsorted(cum_len, rand_idx, right=True)    # B x neg
        _table_id = table_id - 1
        flag = _table_id < 0
        _table_id[flag] = 0
        offset = torch.gather(cum_len, dim=1, index=_table_id)
        offset[flag] = 0
        offset = rand_idx - offset
        indices = torch.gather(start_idx.transpose(0,1), dim=1, index=table_id) + offset    # B x neg
        item_id = self.indices[table_id, indices]   # B x neg

        # cal probablity
        sampling_prob = 1.0 / (len_item) # B
        self.item_embs = self.item_embs.to(query.device)
        sampled_item_emb = F.embedding(item_id, self.item_embs, padding_idx=0)  # B x neg x D
        cosine_theta = F.cosine_similarity(query.view(query.size(0), 1, query.size(1)), sampled_item_emb, dim=-1)   # B x neg
        theta = torch.acos(cosine_theta)
        collision_p = 1 - theta / torch.pi
        weight = (1 - (1 - collision_p ** self.n_bits) ** self.n_table)
        neg_prob = sampling_prob.view(-1, 1) * weight
        neg_id = item_id + 1    # item_id denotes item index without padding

        eps = 1e-12
        log_neg_prob = torch.log(neg_prob + eps)
        if empty_flag.any():
            neg_id[empty_flag] = neg_id_empty
            log_neg_prob[empty_flag] = log_neg_prob_empty.type_as(log_neg_prob)
            
        neg_id = neg_id - 1 # padding position: 0, so output index should minus 1
        
        if pos_items is not None:   # get correction for positive items
            # pos_item_emb = F.embedding(pos_items-1, self.item_embs)  # B x D, remove padding index
            # pos_cosine_theta = F.cosine_similarity(query, pos_item_emb, dim=-1)   # B
            # pos_theta = torch.acos(pos_cosine_theta)
            # pos_p = 1 - pos_theta / torch.pi
            # pos_weight = (1 - (1 - pos_p ** self.n_bits) ** self.n_table)
            # pos_prob = sampling_prob * pos_weight
            # return torch.log(pos_prob+eps), neg_id, log_neg_prob
            return torch.zeros_like(pos_items), neg_id, log_neg_prob    
        else:
            return item_id + 1, torch.log(neg_prob)


    def _generate_random_vectors(self, n_dims, n_hash, n_table):
        random_vectors = torch.rand(n_dims, n_hash, n_table)    # DxKxL
        norm_random_vectors = random_vectors / (torch.norm(random_vectors, dim=0, keepdim=True))
        return torch.nn.Parameter(norm_random_vectors.to(self.device), requires_grad=False)


    def _construct_inverted_index(self, code):
        """
        Construct inverted index for each hash table with a csr sparse data structure.

        Args:
            idx (np.ndarray): NxL, where N is the number of data points, L is the number of tables. Each entry ranges from 0 to K-1. 
                              It indicates each data point's hash code.
        """
        table_indptr = []
        table_indices = []
        for i in range(self.n_table):
            indptr, indices = construct_index(code[:, i], 2 ** self.n_bits)
            table_indptr.append(indptr)
            table_indices.append(indices)
        table_indptr = torch.stack(table_indptr)
        table_indices = torch.stack(table_indices)

        # check legacy
        # if there are a lot of empty buckets in each table, it's possible that the number of bits is too large.


        return table_indptr, table_indices
    
    
# TODO(@AngusHuang17): avoid sampling pos items in MIDX and Cluster
# TODO(@AngusHuang17) aobpr sampler

def test():
    num_items = 1000
    n_dims = 64
    batch_size = 4
    num_negs = 10
    item_emb = torch.rand(num_items, n_dims).cuda()
    query = torch.rand(batch_size, n_dims).cuda()
    pos_id = torch.randint(1, num_items, size=(batch_size,)).cuda()

    #=== LSH Sampler ===
    n_bits = 4
    n_table = 5
    lsh_sampler = LSHSampler(num_items, n_dims, n_bits, n_table, device="cuda")
    lsh_sampler.update(item_emb)
    pos_prob, neg_id, neg_prob = lsh_sampler.__call__(query, num_negs, pos_id)
    print(pos_prob.shape, neg_id.shape, neg_prob.shape)

    print('Test Passed')

if __name__ == "__main__":
    test()