# ReMAP

Official implementation of the paper:

**Knowledge Reutilization in Meta-Reinforcement Learning**

## Authors

Yuan Meng<sup>1,3,*</sup>, Bo Wang<sup>1,3,*</sup>, Juan de los Rios Ruiz<sup>1</sup>, Xiangtong Yao<sup>1</sup>, Kai Huang<sup>2</sup>, Yang Gao<sup>3</sup>, Fuchun Sun<sup>4</sup>, Alois Knoll<sup>1</sup>, and Zhenshan Bing<sup>3,&dagger;</sup>

<sup>1</sup>School of Computation, Information and Technology, Technical University of Munich, Munich, Germany  
<sup>2</sup>School of Computer Science and Engineering, Sun Yat-sen University, Guangzhou, China  
<sup>3</sup>State Key Laboratory for Novel Software Technology, Nanjing University, Suzhou, China  
<sup>4</sup>Department of Computer Science and Technology, Tsinghua University, Beijing, China

<sup>*</sup>Equal contribution. The work was done during the research visit at Nanjing University.  
<sup>&dagger;</sup>Corresponding author: bing@nju.edu.cn

## Abstract

Meta-reinforcement learning aims to enable fast adaptation to unseen tasks by extracting shared structure from a distribution of related tasks. However, existing end-to-end methods often learn this structure jointly with embodiment-specific control dynamics, which can entangle task semantics with agent-specific motion patterns and limit cross-agent reuse. We propose ReMAP, a meta-knowledge reutilization framework that learns task-level meta-knowledge on a dynamics-simplified agent and transfers it to heterogeneous agents. A DPMM-regularized task inference module captures non-parametric task semantics, while a high-level policy generates task-level magnitude guidance. To bridge this reusable knowledge with different robot embodiments, we introduce the Semantic-Magnitude Alignment Interface, which converts task semantics and magnitudes into shared subgoals for embodiment-specific low-level controllers. Experiments across multiple locomotion agents show that ReMAP reduces final-step tracking MSE by 94.75%-99.79% compared with Meta-RL baselines and achieves state-of-the-art deployment performance with about 23.8% of the interactions required by recent baselines.

## Introduction to Repository

Code coming soon.
