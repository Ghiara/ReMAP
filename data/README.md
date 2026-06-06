# ReMAP Data

This directory contains the source CSV data used for the experimental results
reported in **Knowledge Reutilization in Meta-Reinforcement Learning**.

## Structure

- `ReMAP/`: results from the proposed ReMAP method, including DPMM inference
  results for Ant, Cheetah, Hopper, and Walker, plus simplified-agent results.
- `baseline/`: baseline experiment data for CEMRL, MELTS, PEARL, and RL2, as
  well as tracking MSE comparisons for goal and velocity tasks.
- `ablation_study/`: ablation data comparing GMM, single-Gaussian, and
  stick-breaking variants.
- `reward/`: reward statistics used in the paper analysis.
- `single_task_case_study/`: per-task case-study CSV files for Ant, Cheetah,
  Hopper, and Walker.


