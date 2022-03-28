############
Pre-Training
############

The first step is an unsupervised pre-training process.

- model: autoregressive, several blocks of transformer decoders

- objective: predict the next word (maximize the probability of the next word)

Meta-Learning
=============

The **meta-learning** is the inner-loop / outer-loop structure in the learning
process. The **in-context learning** refers to the inner loop of meta-learning.

.. code-block:: none

    sequence 1 inner loop (arithmetic)
    3+5=13, 7+2=9, ..., 9+8=17

    sequence 2 inner loop (correct words)
    gaot=>goat, sakne=>snake, dcku=>duck

    ...

    (the end of the outer loop)

- Each inner-loop represents a task to train a specific set of skills, and
  repeated sub-tasks can be embedded within a single sequence

- Meta-learning develops a broad set of skills and pattern recognition
  abilities at training time within the same model

- With these abilities the model can adapt rapidly to desired tasks

- Some of the results are still far inferior to fine-tuning

- The efficiency of meta-learning improves with scale

Pre-Training
============

.. code-block:: none

   1. input: "The quick brown fox jumps", label: "over"
   2. input: "The quick brown fox jumps over", label: "the"
   3. input: "The quick brown fox jumps over the", label: "lazy"
   4. input: "The quick brown fox jumps over the lazy", label: "dog"

- dataset: a total of 300 billion tokens

Back to :doc:`index`.
