from easydict import EasyDict as edict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def piecewise_index(relative_position, alpha, beta, gamma):
    ''' piecewise index function
    defined in Eq. (18) in our paper
    returns:
        [-beta, beta]
    '''
    rp_abs = relative_position.abs()
    mask = rp_abs <= alpha
    not_mask = ~mask
    rp_out = relative_position[not_mask]
    rp_abs_out = rp_abs[not_mask]
    y_out = (torch.sign(rp_out) * (alpha + 
                                      torch.log(rp_abs_out / alpha) /
                                      math.log(gamma / alpha) *
                                      (beta - alpha)).round().clip(max=beta)).long()

    idx = relative_position.clone()
    if idx.dtype in [torch.float32, torch.float64]:
        idx = idx.round().long()

    idx[not_mask] = y_out
    return idx

@torch.no_grad()
def _get_pos(E):
    '''
    return: (E, E, 2), 2 is [row, col]

    rows:
    1 1 1
    2 2 2
    3 3 3

    cols:
    1 2 3
    1 2 3
    1 2 3
    '''
    rg = torch.arange(E)
    rows = rg.view(E, 1).repeat(1, E) 
    cols = rows.t()
    return torch.stack([rows, cols], 2)

@torch.no_grad()
def quantize_idx(ids):
    # quantize and re-assign bucket id
    res = torch.empty_like(ids)
    uq = ids.unique()
    cnt = 0
    for (tid, v) in enumerate(uq):
        mask = (ids == v)
        cnt += torch.count_nonzero(mask)
        res[mask] = tid
    assert cnt == ids.numel()
    return res, uq.numel() 


class METHOD:
    EUCLIDEAN=0
    QUANT=1
    PRODUCT=3
    CROSS=4
    CROSS_ROWS=41
    CROSS_COLS=42

@torch.no_grad()
def _rp_2d_euclidean(diff, **kwargs):
    '''Euclidean method
    diff: torch.Tensor (E*E, E*E, 2)
    '''
    dis = diff.square().sum(2).float().sqrt().round()
    g = piecewise_index(dis, **kwargs)
    return quantize_idx(g)[0]

@torch.no_grad()
def _rp_2d_quant(diff, **kwargs):
    '''Quantization method
    '''
    dis = diff.square().sum(2)
    g = piecewise_index(dis, **kwargs)
    return quantize_idx(g)[0]

@torch.no_grad()
def _rp_2d_product(diff, **kwargs):
    '''Product method
    '''
    r, r_num = quantize_idx(piecewise_index(diff[:, :, 0], **kwargs))
    c, c_num = quantize_idx(piecewise_index(diff[:, :, 1], **kwargs))
    pid = r * c_num + c
    return pid

@torch.no_grad()
def _rp_2d_cross_rows(diff, **kwargs):
    '''
    diff: torch.Tensor (E*E, E*E, 2)
    '''
    dis = diff[:, :, 0]
    g = piecewise_index(dis, **kwargs)
    return quantize_idx(g)[0]

@torch.no_grad()
def _rp_2d_cross_cols(diff, **kwargs):
    '''
    diff: torch.Tensor (E*E, E*E, 2)
    '''
    dis = diff[:, :, 1]
    g = piecewise_index(dis, **kwargs)
    return quantize_idx(g)[0]

_METHOD_FUNC = {
    METHOD.EUCLIDEAN: _rp_2d_euclidean,
    METHOD.QUANT: _rp_2d_quant,
    METHOD.PRODUCT: _rp_2d_product,
    METHOD.CROSS_ROWS: _rp_2d_cross_rows,
    METHOD.CROSS_COLS: _rp_2d_cross_cols,
}

@torch.no_grad()
def _get_bucket_ids_2d(method, E, skip, alpha, beta, gamma):
    '''
    num_buckets: buckets including `skip`
    return:
        (skip + E * E, skip + E * E)
    '''
    assert skip in [0, 1]
    func = _METHOD_FUNC.get(method, None)
    if func is None:
        raise NotImplementedError(method)
    # (E, E, 2)
    pos = _get_pos(E)
    pos1 = pos.view((E * E, 1, 2))
    pos2 = pos.view((1, E * E, 2))
    # (E*E, E*E, 2)
    diff = pos1 - pos2
    # (E*E, E*E)
    bucket_ids = func(diff, alpha=alpha, beta=beta, gamma=gamma)
    num_buckets = bucket_ids.unique().numel()

    # add an extra encoding (id = num_buckets) for the classification token
    if skip > 0:
        assert skip == 1
        new_bids = bucket_ids.new_full(size=(skip + E * E, skip + E * E), fill_value=num_buckets)
        new_bids[skip:, skip:] = bucket_ids
        bucket_ids = new_bids
    return bucket_ids, num_buckets + skip


class RelativePositionBias2D(nn.Module):
    def __init__(self, head_dim, num_heads=8, mode=None, method=None, transposed=True, num_buckets=None, initializer=None, rpe_config=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        # relative position
        assert mode in [None, 'bias', 'contextual']
        self.mode = mode
        assert method is not None
        self.method = method
        self.transposed = transposed
        self.num_buckets = num_buckets

        self.initializer = initializer
        if initializer is None:
            initializer = lambda x: None

        if transposed:
            if mode == 'bias':
                self.lookup_table = nn.Parameter(torch.zeros(num_heads, num_buckets))
                initializer(self.lookup_table)
            elif mode == 'contextual':
                self.lookup_table_weight = nn.Parameter(torch.zeros(num_heads, head_dim, num_buckets))
                initializer(self.lookup_table_weight)
        else:
            if mode == 'bias':
                raise NotImplementedError()
            elif mode == 'contextual':
                self.lookup_table_weight = nn.Parameter(torch.zeros(num_heads, num_buckets, head_dim))
                initializer(self.lookup_table_weight)

        self.rpe_config = rpe_config

        # store bucket index
        self._rp_bucket = (None, None)  # ((L, device), rp_bucket) 

    def forward(self, x):
        '''
        Inputs
        ------
        x: (B, H, L, head_dim)
        '''
        rp_bucket = self._get_rp_bucket(x)
        if self.transposed:
            return self._get_bias_from_lookup_table_transpose(x, rp_bucket)
        return self._get_bias_from_lookup_table_no_transpose(x, rp_bucket)

    def _get_rp_bucket(self, x):
        B, H, L, D = x.shape
        device = x.device
        key = (L, device)
        if self._rp_bucket[0] == key:
            return self._rp_bucket[1]
        E = int(math.sqrt(L))
        skip = L - E * E
        config = self.rpe_config
        rp_bucket, num_buckets = _get_bucket_ids_2d(method=self.method, E=E, skip=skip, alpha=config.alpha, beta=config.beta, gamma=config.gamma)
        rp_bucket = rp_bucket.to(device)
        assert num_buckets == self.num_buckets

        # transpose contextual
        if self.mode == 'contextual' and self.transposed:
            offset = torch.arange(0, L * self.num_buckets, self.num_buckets,
                                  dtype=rp_bucket.dtype, device=rp_bucket.device).view(-1, 1)
            self._ctx_rp_bucket_flatten = (rp_bucket + offset).flatten()
        self._rp_bucket = (key, rp_bucket)
        return rp_bucket

    def _get_bias_from_lookup_table_no_transpose(self, attn, rp_bucket):
        '''no transpose version
        Inputs
        ------
        attn: (B, H, L_query, L_mem)
        rp_bucket: (L_query, L_mem)
        lookup_table_weight: (H or 1, num_buckets, head_dim)
        weight: (H, L_query, L_mem, D)

        Output
        ------
        return: (B, H, L_query, D)
        '''
        B = len(attn)  # batch_size
        L_query, L_mem = rp_bucket.shape
        assert  self.mode == 'contextual'
        # (H, head_dim, L_query, L_mem)
        # reduce L_mem
        weight = self.lookup_table_weight[:, rp_bucket.flatten()].view(self.num_heads, L_query, L_mem, self.head_dim)
        # (H, L_query, B, L_mem) @ (H, L_query, L_mem, D) = (H, L_query, B, D) -> (B, H, L_query, D)
        return torch.matmul(attn.permute(1, 2, 0, 3), weight).permute(2, 0, 1, 3)

    def _get_bias_from_lookup_table_transpose(self, q, rp_bucket):
        '''
        Inputs
        ------
        q: (B, H, L, head_dim)
        rp_bucket: (L_query, L_mem)
        '''
        B = len(q)  # batch_size
        L_query, L_mem = rp_bucket.shape
        if self.mode == 'bias':
            '''
            lookup_table: (num_heads, num_buckets)
            ret: (1, num_heads, L_query, L_mem)
            '''
            return self.lookup_table[:, rp_bucket.flatten()].view(1, self.num_heads, L_query, L_mem)

        elif self.mode == 'contextual':
            '''
            q: (B, H, L_query, head_dim)
            rp_bucket: (L_query, L_mem)
            lookup_table_weight: (H or 1, head_dim, num_buckets)
            lookup_table: (B, H or 1, L_query, num_buckets)

            return value: (B, H, L_query, L_mem)

            ret[b, h, i, j] = lookup_table[b, h, i, rp_bucket[i, j]]

            ret[b, h, i * L_mem + j] = \
               lookup_table[b, h, i * num_buckets + rp_buckets[i, j]]

            computational cost
            matmul: B * H * L_query * head_dim * num_buckets
            index: L_query + L_query * L_mem + B * H * L_query * L_mem
            ------
            total: O(B * H * L_query * (head_dim * num_buckets + L_mem))
            '''
            # reduce head_dim
            lookup_table = torch.matmul(
                q.transpose(0, 1).reshape(-1, B * L_query, self.head_dim),
                self.lookup_table_weight).view(-1, B, L_query, self.num_buckets).transpose(0, 1)
            return lookup_table.flatten(2)[:, :, self._ctx_rp_bucket_flatten].view(B, -1, L_query, L_mem)
        else:
            raise NotImplementedError(self.mode)

    def __repr__(self):
        return 'RelativePositionBias2D(head_dim={rpe.head_dim}, num_heads={rpe.num_heads}, mode="{rpe.mode}", method={rpe.method}, transposed={rpe.transposed}, num_buckets={rpe.num_buckets}, initializer={rpe.initializer}, rpe_config={rpe.rpe_config})'.format(rpe=self)

class RelativePositionBias2D_Cross(nn.Module):
    def __init__(self, method, **kwargs): 
        super().__init__()
        assert method == METHOD.CROSS
        self.rp_rows = RelativePositionBias2D(**kwargs, method=METHOD.CROSS_ROWS)
        self.rp_cols = RelativePositionBias2D(**kwargs, method=METHOD.CROSS_COLS)

    def forward(self, q):
        rows = self.rp_rows(q)
        cols = self.rp_cols(q)
        return rows + cols

    def __repr__(self):
        return 'RelativePositionBias2D_Cross(head_dim={rpe.head_dim}, num_heads={rpe.num_heads}, mode="{rpe.mode}", method={rpe.method}, transposed={rpe.transposed}, num_buckets={rpe.num_buckets}, initializer={rpe.initializer}, rpe_config={rpe.rpe_config})'.format(rpe=self.rp_rows)

def _get_single_rpe_config(ratio=1.9, method=METHOD.PRODUCT, mode='contextual', shared_head=True):
    config = edict()
    # whether to share encodings across different heads
    config.shared_head = shared_head
    # mode: None, bias, contextual
    config.mode = mode
    # method: None, Bias, Quant, Cross, Product
    config.method = method
    config.alpha = 1 * ratio
    config.beta = 2 * ratio
    config.gamma = 8 * ratio
    E = 14
    skip = 1
    if method == METHOD.CROSS:
        config.num_buckets = _get_bucket_ids_2d(method=METHOD.CROSS_ROWS, E=E, skip=skip, alpha=config.alpha, beta=config.beta, gamma=config.gamma)[1]
    else:
        config.num_buckets = _get_bucket_ids_2d(method=method, E=E, skip=skip, alpha=config.alpha, beta=config.beta, gamma=config.gamma)[1]
    return config

def get_rpe_config(ratio=1.9, method=METHOD.PRODUCT, mode='contextual', shared_head=True, rpe_on='k'):
    # alias
    if isinstance(method, str):
        method_mapping = dict(
            euc=METHOD.EUCLIDEAN,
            quant=METHOD.QUANT,
            cross=METHOD.CROSS,
            product=METHOD.PRODUCT,
        )
        method = method_mapping[method]
    if mode == 'ctx':
        mode = 'contextual'
    config = edict()
    # relative position encoding on queries, keys and values
    kwargs = dict(
        ratio=ratio,
        method=method,
        mode=mode,
        shared_head=shared_head,
    )
    config.rpe_q = _get_single_rpe_config(**kwargs) if 'q' in rpe_on else None
    config.rpe_k = _get_single_rpe_config(**kwargs) if 'k' in rpe_on else None
    config.rpe_v = _get_single_rpe_config(**kwargs) if 'v' in rpe_on else None
    return config

def build_rpe(config, head_dim, num_heads):
    if config is None:
        return None, None, None
    rpes = [config.rpe_q, config.rpe_k, config.rpe_v]
    transposeds = [True, True, False]
    def _build_single_rpe(rpe, transposed):
        if rpe is None:
            return None
        num_heads = 1 if rpe.shared_head else rpe.num_heads
        rpe_cls = RelativePositionBias2D if rpe.method != METHOD.CROSS else RelativePositionBias2D_Cross
        return rpe_cls(
            head_dim=head_dim,
            num_heads=num_heads,
            mode=rpe.mode,
            method=rpe.method,
            transposed=transposed,
            num_buckets=rpe.num_buckets,
            rpe_config=rpe,
        )
    return [_build_single_rpe(rpe, transposed) \
            for rpe, transposed in zip(rpes, transposeds)]


if __name__ == '__main__':
    config = get_rpe_config(head_dim=8, num_heads=2)
    rpe = build_rpe(config)
    print(rpe)
