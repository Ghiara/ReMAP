# ReMAP

<div align="center">

Official implementation of the paper:

**Knowledge Reutilization in Meta-Reinforcement Learning**


Yuan Meng<sup>1,3,*</sup>, Bo Wang<sup>1,3,*</sup>, Juan de los Rios Ruiz<sup>1</sup>, Xiangtong Yao<sup>1</sup>, Kai Huang<sup>2</sup>, Yang Gao<sup>3</sup>, Fuchun Sun<sup>4</sup>, Alois Knoll<sup>1</sup>, and Zhenshan Bing<sup>3,&dagger;</sup>

<sup>1</sup>School of Computation, Information and Technology, Technical University of Munich, Munich, Germany  
<sup>2</sup>School of Computer Science and Engineering, Sun Yat-sen University, Guangzhou, China  
<sup>3</sup>State Key Laboratory for Novel Software Technology, Nanjing University, Suzhou, China  
<sup>4</sup>Department of Computer Science and Technology, Tsinghua University, Beijing, China

<sup>*</sup>Equal contribution. The work was done during the research visit at Nanjing University.  
<sup>&dagger;</sup>Corresponding author: bing@nju.edu.cn

</div>

## Abstract

Meta-reinforcement learning enables fast adaptation by extracting shared structure from related tasks, but existing end-to-end methods often couple task inference with embodiment-specific control. This coupling can obscure non-parametric task semantics, reduce sample efficiency, and limit cross-agent reuse. We propose ReMAP, a meta-knowledge reutilization framework that learns task-level meta-knowledge on a dynamics-simplified agent and transfers it to heterogeneous agents. ReMAP uses a DPMM-regularized task inference module to organize non-parametric task modes and a high-level policy to generate task-level magnitude guidance. To bridge reusable task knowledge with different embodiments, we introduce the Semantic-Magnitude Alignment Interface and a lightweight stride predictor, which convert frozen meta-knowledge into temporally aligned semantic-magnitude subgoals for embodiment-specific low-level controllers. Experiments on multiple locomotion agents show that ReMAP reduces final-step tracking MSE by \(94.75\%\)--\(99.79\%\) and achieves state-of-the-art deployment performance with about \(23.8\%\) of the interactions required by MELTS.

## Introduction to Repository

Code coming soon.



