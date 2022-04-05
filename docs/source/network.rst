###############
Neural Networks
###############

.. default-role:: code

Layer Normalization
===================

- It computes the mean and variance used for normalization from **all** of the
  summed inputs to the neurons in a layer on a single training case. [#f2]_

- It is different from the **batch normalization**, which computes a mean and
  variance over a mini-batch

- It is like a **transposed** batch normalization

- Like batch normalization it has trainable bias and gain, which are applied
  after the normalization.

Feed-Forward
============

CONV1D Layer
------------

The `Conv1D` [#f3]_ is actually a linear layer, changing the dimension of input tensor
from `nx` to `nf`.

.. code-block:: python

    class Conv1D(nn.Module):
        def __init__(self, nf, nx):
            super(Conv1D, self).__init__()
            self.nf = nf
            w = torch.empty(nx, nf)
            nn.init.normal_(w, std=0.02)
            self.weight = Parameter(w)
            self.bias = Parameter(torch.zeros(nf))

        def forward(self, x):
            size_out = x.size()[:-1] + (self.nf,)
            # x.size(-1) should be equal to nx
            x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
            x = x.view(*size_out)
            return x

Feed-Forward Layer
------------------

- accept outputs from the attention layer

- cast from `d_model` dimension to `nx` dimension. As mentioned in the
  :doc:`model` part, the output dimension is **four times** the
  :math:`d_{model}`

- activate with `GELU`

- cast back to `d_model` dimension

- final dropout

.. code-block:: python

    D_MODEL = 768


    class FeedForward(nn.Module):
        def __init__(self, dropout, d_model=D_MODEL, nx=D_MODEL*4):
             super().__init__()
             self.c_fc    = Conv1D(d_model, nx)
             self.c_proj  = Conv1D(nx, d_model)
             self.act     = F.gelu
             self.dropout = nn.Dropout(dropout)

        def forward(self, x):
             return self.dropout(self.c_proj(self.act(self.c_fc(x))))

Attention Layer
===============

By combining `MultiHeadedAttention` from `The Annotated Transformer
<https://nlp.seas.harvard.edu/2018/04/03/attention.html>`_ and `Attention` from
`GPT-2 implementation <https://github.com/graykode/gpt-2-Pytorch>`_ we have a
new annotation:

.. code-block:: python

    import math

    import torch
    from torch import nn


    class Attention(nn.Module):

        def __init__(
            self,
            d_model=768,
            n_head=12,
            n_ctx=1024,
            d_head=64,
            bias=True,
            scale=False,
        ):
            super().__init__()
            self.n_head = n_head
            self.d_model = d_model
            self.c_attn = Conv1D(d_model, d_model * 3)
            self.scale = scale
            self.softmax = nn.Softmax(dim=-1)
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
            self.dropout = nn.Dropout(0.1)
            self.c_proj = Conv1D(d_model, d_model)

        def split_heads(self, x):
            "return shape [`batch`, `head`, `sequence`, `features`]"
            new_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
            x = x.view(*new_shape)
            return x.permute(0, 2, 1, 3)

        def _attn(self, q, k, v, attn_mask=None):
            scores = torch.matmul(q, k.transpose(-2, -1))
            if self.scale:
                scores = scores / math.sqrt(v.size(-1))
            nd, ns = scores.size(-2), scores.size(-1)
            if attn_mask is not None:
                scores = scores + attn_mask
            scores = self.softmax(scores)
            scores = self.dropout(scores)
            outputs = torch.matmul(scores, v)
            return outputs

        def merge_heads(self, x):
            x = x.permute(0, 2, 1, 3).contiguous()
            new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1), )
            return x.view(*new_shape)

        def forward(self, x):
            # original x shape: ..., n_batch, d_model
            # x shape: ..., n_batch, 3 * d_model
            x = self.c_attn(x)
            # q, k, v shape: ..., n_batch, d_model
            q, k, v = x.split(self.d_model, dim=-1)
            # q, k, v shape: ..., n_head, n_batch, d_key
            q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
            # out shape: ..., n_head, n_batch, d_key
            out = self._attn(q, k, v)
            # out shape: ..., n_batch, d_model
            out = self.merge_heads(out)
            # out shape: ..., n_batch, d_model
            out = self.c_proj(out)
            return out

Reference
=========

.. [#f2] https://arxiv.org/abs/1607.06450

.. [#f3] https://github.com/graykode/gpt-2-Pytorch/blob/master/GPT2/model.py#L30

Back to :doc:`index`.
