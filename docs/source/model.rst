############
Model Design
############

.. _ref-model-parameters:

Parameters
==========

.. default-role:: math

The input text will go through an embedding / encoding process, which will map
the string into a matrix:

+--------------+-----------------+-------------+-------------+------------+------------+
| Model Name   | `n_{parameter}` | `n_{layer}` | `d_{model}` | `n_{head}` | `d_{head}` |
+==============+=================+=============+=============+============+============+
| GPT-2 Small  | 117M            | 12          | 768         | 12         | 64         |
+--------------+-----------------+-------------+-------------+------------+------------+
| GPT-3 [#f1]_ | 175B            | 96          | 12288       | 96         | 128        |
+--------------+-----------------+-------------+-------------+------------+------------+

- `n_{parameter}` - total number of trainable parameters

- `n_{layer}` - number of the decoder-only transformer layers

- `d_{model}` - the original explain is "number of units in each bottleneck layer".
  It first denotes the vector length of input embedding / encoding.

- `d_{ff}` - number of unit in the hidden states of the feed-forward layer. It
  is designed to be **four times** of `d_{model}`.

- `d_{head}` - dimension of each attention head

.. default-role:: code

Workflow Chart
==============

.. code-block:: none

    diagram of the GPT-3 process

                                    <---------- 50257 ---------->

                                    ============= 1 =============
                                    ============= 2 =============
                                    ============ ... ============
                                    ============ 2048 ===========

                                   |||                         |||
                                   |||                         |||
                                  \|||/                       \|||/
                                   \|/                         \|/

                             token embedding           positional encoding
                                  (WTE)                       (WPE)

                             <--- 12288 --->             <--- 12288 --->

                             ====== 1 ======             ====== 1 ======
                             ====== 2 ======             ====== 2 ======
                             ===== ... =====             ===== ... =====
                             ===== 2048 ====             ===== 2048 ====

                                    |                           |
                                    |                           |
                                    -----------------------------
                                                 |||
                                                 |||
                                                \|||/
                                                 \|/

                                               plus (+)

                                           <--- 12288 --->

                                           ====== 1 ======
                                           ====== 2 ======
                                           ===== ... =====
                                           ===== 2048 ====

                                                 |||
                                                 |||
                                                \|||/
                                                 \|/

                                       1st Decoder Transformer

                                           <--- 12288 --->

                                           ====== 1 ======
                                           ====== 2 ======
                                           ===== ... =====
                                           ===== 2048 ====

                                                 |||
                                                 |||
                                                \|||/
                                                 \|/

                                                 ...

                                                 |||
                                                 |||
                                                \|||/
                                                 \|/

                                       96th Decoder Transformer

                                           <--- 12288 --->

                                           ====== 1 ======
                                           ====== 2 ======
                                           ===== ... =====
                                           ===== 2048 ====

                                                 |||
                                                 |||
                                                \|||/
                                                 \|/

                                       inverse token embedding
                                         (WTE^(-1) + softmax)

                                    <---------- 50257 ---------->

                                    ============= 1 =============
                                    ============= 2 =============
                                    ============ ... ============
                                    ============ 2048 ===========


Reference
=========

.. [#f1] https://arxiv.org/abs/2005.14165

Back to :doc:`index`.
