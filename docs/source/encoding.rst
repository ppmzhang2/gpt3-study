##############
Input Encoding
##############

The input text will go through an embedding / encoding process, which will map
the string into a matrix:

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

Once tokenized, each of the input tokens will have two groups of properties:

- semantic properties as a sub-word, indicated by its token index ranging from
  **0** to **50256**

- positional properties as an element in a sequence, indicated by its
  positional index ranging from **0** to **2047**

The semantic properties will be interpreted by a "dictionary" in the form of a
matrix, and

Vocabulary Embedding
====================

To interpret a word we could use a dictionary and in the case of the GPT-3
training the "dictionary" is actually a **50257-by-12288** matrix, each row of
which is a **12288-dimension** vector, containing distance information with
other tokens

The GPT-3 has 2048 input slots and after tokenisation there the input will be
a **2048-by-50257** matrix. By multiplying the input matrix with the vocabulary
matrix mentioned above we have a **2048-by-12288** embedded matrix.

Questions
---------

- does the token entries obey the additive law?

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
