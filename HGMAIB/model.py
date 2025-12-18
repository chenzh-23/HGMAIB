import torch
import torch.nn as nn
import torch.nn.functional as F


class Op(nn.Module):
    def __init__(self):
        super(Op, self).__init__()

    def forward(self, x, adjs, idx):
        return torch.spmm(adjs[idx], x)


class Cell(nn.Module):

    def __init__(self, n_step, n_hid_prev, n_hid, use_norm=True, use_nl=True):
        super(Cell, self).__init__()

        self.affine = nn.Linear(n_hid_prev, n_hid)  #一个线性变换层，将输入维度从 n_hid_prev 变为 n_hid
        self.n_step = n_step    #步数，即计算图神经网络层的次数
        self.norm = nn.BatchNorm1d(n_hid) if use_norm is True else lambda x: x
        self.use_nl = use_nl

        self.ops_seq = nn.ModuleList()
        self.ops_res = nn.ModuleList() #分别用于顺序操作和残差操作的 Op 实例的列表。
        self.res_weights = nn.ParameterList()

        for i in range(n_step):
            self.ops_seq.append(Op())
        for i in range(1, n_step):
            for j in range(i):
                self.ops_res.append(Op())
                self.res_weights.append(nn.Parameter(torch.Tensor(1)))
                nn.init.constant_(self.res_weights[-1], 1.0)

    def forward(self, x, adjs, idxes_seq, idxes_res):

        x = self.affine(x)
        states = [x] #初始化状态列表，初始状态为线性变换后的 x
        offset = 0
        for i in range(self.n_step):
            seqi = self.ops_seq[i](states[i], adjs[:-1], idxes_seq[i])  # ! exclude zero Op
            resi = sum(self.res_weights[offset + j] * self.ops_res[offset + j](h, adjs, idxes_res[offset + j])
                       for j, h in enumerate(states[:i]))
            offset += i
            states.append(seqi + resi) #更新状态列表。

        output = self.norm(states[-1])
        if self.use_nl:
            output =F.silu(output)
        return output


class Model(nn.Module):
    def __init__(self, in_dims, n_hid, n_steps, dropout=True, attn_dim=64, use_norm=True, out_nl=True, num_heads=4):
        super(Model, self).__init__()
        self.n_hid = n_hid
        self.num_heads = num_heads

        self.ws = nn.ModuleList()
        assert isinstance(in_dims, list)
        for i in range(len(in_dims)):
            self.ws.append(nn.Linear(64, n_hid))

        assert isinstance(n_steps, list)
        self.metas = nn.ModuleList()
        for i in range(len(n_steps)):
            self.metas.append(Cell(n_steps[i], n_hid, n_hid, use_norm=use_norm, use_nl=out_nl))

        self.attn_fc1 = nn.ModuleList([nn.Linear(n_hid, attn_dim) for _ in range(num_heads)])
        self.attn_fc2 = nn.ModuleList([nn.Linear(attn_dim, 1) for _ in range(num_heads)])
        self.feats_drop = nn.Dropout(dropout) if dropout is not None else lambda x: x

        # 信息瓶颈模块
        self.fc_mu = nn.Linear(n_hid, n_hid)
        self.fc_logvar = nn.Linear(n_hid, n_hid)

    def forward(self, node_feats, node_types, adjs, idxes_seq, idxes_res):
        hid = torch.zeros((node_types.size(0), self.n_hid)).cuda()
        for i in range(len(node_feats)):
            hid[node_types == i] = self.ws[i](node_feats[i])
        hid = self.feats_drop(hid)

        # 多元路径输出
        temps = []
        for i, meta in enumerate(self.metas):
            hidi = meta(hid, adjs, idxes_seq[i], idxes_res[i])
            temps.append(hidi)

        # 路径表示堆叠 (B, n_steps, n_hid)
        hids = torch.stack(temps, dim=0).transpose(0, 1)

        # 层级注意力机制
        per_head_outs = []
        for head in range(self.num_heads):
            # 第1层注意力（每个 head 内部对路径加权）
            score = torch.tanh(self.attn_fc1[head](hids))           # (B, n_steps, attn_dim)
            score = self.attn_fc2[head](score)                      # (B, n_steps, 1)
            weight = F.softmax(score, dim=1)                        # (B, n_steps, 1)
            head_out = (weight * hids).sum(dim=1)                   # (B, n_hid)
            per_head_outs.append(head_out)

        # 堆叠所有 head 的输出 (B, num_heads, n_hid)
        head_stack = torch.stack(per_head_outs, dim=1)

        # 第2层注意力：head 间加权（使用 sigmoid 门控）
        gate = torch.sigmoid(torch.mean(head_stack, dim=-1, keepdim=True))  # (B, num_heads, 1)
        out = (gate * head_stack).sum(dim=1)  # (B, n_hid)

        # 信息瓶颈输出
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        return mu, logvar
