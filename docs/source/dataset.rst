#######
Dataset
#######

.. default-role:: code

Challenges
==========

Summary from the paper:

- model scale

  - larger models can typically use a larger batch size, but require a smaller
    learning rate

  - potential contamination of downstream tasks

    - test / development sets inadvertently seen during pre-training

    - remove overlaps

- spurious features

  - these spurious correlations can occur due to biased sampling or artifacts
    in crowd-sourcing. For example, we may have a labeled dataset for
    recidivism prediction where race correlates with recurrence of crime due to
    sample selection bias, but this correlation does not hold on the
    population. Models which learn spurious correlations can generalize poorly
    on population data which does not have these biases. [#f1]_

Reference
=========

.. [#f1] https://arxiv.org/abs/2006.10032

Back to :doc:`index`.
