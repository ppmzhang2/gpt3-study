##############
Input Encoding
##############

The first step is an unsupervised pre-training process.

- input text string

  .. code-block:: python

      "python primer plus"

- tokenized strings

  .. code-block:: python

      ["python", "prim", "er", "plus"]

- each token will be mapped to a token ID, ranging from **0** to **50256**

  .. code-block:: python

      [2, 4, 6, 101]

- each token ID will be mapped to a 12288-dimension vector (GPT-3)

Embedding
=========

For each of the **50257** tokens there is a pre-trained **12288-dimension**
vector, and the whole vocabulary is a **50257-by-12288** matrix.

The GPT-3 has 2048 input slots and after tokenisation there the input will be
a **2048-by-50257** matrix. By multiplying the input matrix with the vocabulary
matrix mentioned above we have a **2048-by-12288** embedded matrix.

Positional Encoding
===================

For each token in the 2048 input slots, its position is encoded passing the
index (from 0 to 2047) to **12288** sinusoidal functions of different frequencies.
The output is a **2048-by-12288** position encoded matrix.


Encoding Combination
====================

Both the embedding matrix and position encoded matrix are **2048-by-12288**
matrices, and the final input to the model will be the sum of them.

Back to :doc:`index`.
