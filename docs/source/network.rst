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

Reference
=========

.. [#f2] https://arxiv.org/abs/1607.06450

.. [#f3] https://github.com/graykode/gpt-2-Pytorch/blob/master/GPT2/model.py#L30

Back to :doc:`index`.
