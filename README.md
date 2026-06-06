# ReMAP




<div align="center">

Official implementation of the paper: **Knowledge Reutilization in Meta-Reinforcement Learning**

Yuan Meng<sup>1,2,*</sup>, Bo Wang<sup>1,2,*</sup>, Juan de los Rios Ruiz<sup>1</sup>, Xiangtong Yao<sup>1</sup>, 
Zhenshan Bing<sup>2,&dagger;</sup>, Fuchun Sun<sup>3</sup>, and Alois Knoll<sup>1</sup> 

<sup>1</sup>School of Computation, Information and Technology, Technical University of Munich, Munich, Germany  
<sup>2</sup>State Key Laboratory for Novel Software Technology, Nanjing University, Suzhou, China  
<sup>3</sup>Department of Computer Science and Technology, Tsinghua University, Beijing, China

<sup>*</sup>Equal contribution. The work was done during the research visit at Nanjing University.  
<sup>&dagger;</sup>Corresponding author: bing@nju.edu.cn

</div>

## Abstract

Meta-reinforcement learning enables fast adaptation by extracting shared structure from related tasks, but existing end-to-end methods often couple task inference with embodiment-specific control.
This coupling can obscure non-parametric task semantics, reduce sample efficiency, and limit cross-agent reuse.
We propose a meta-knowledge reutilization framework that learns task-level knowledge on a dynamics-simplified agent and transfers it to heterogeneous agents.
The framework uses a Bayesian non-parametric prior to organize latent task modes and a high-level policy to generate task-level magnitude guidance.
To bridge reusable task knowledge with different embodiments, we introduce a semantic-magnitude interface and a lightweight temporal adaptor, which convert frozen meta-knowledge into temporally aligned subgoals for embodiment-specific low-level controllers.
Experiments on multiple locomotion agents show that our framework reduces final-step tracking error by \(94.75\%\)--\(99.79\%\) compared with recent state-of-the-art baselines and achieves comparable deployment performance with about \(23.8\%\) of their interaction data.

## Introduction to Repository

Code coming soon.

## Data

The source data for the paper experiments are provided in [`data/`](data/).
They include CSV files for ReMAP results, baseline comparisons, ablation
studies, reward statistics, and single-task case studies used in the main paper
and supplementary material.
