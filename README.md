# New Adversarial Attack Papers
**update at 2021-11-15**

[中文版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Resilient Consensus-based Multi-agent Reinforcement Learning**

cs.LG

**SubmitDate**: 2021-11-12    [paper-pdf](http://arxiv.org/pdf/2111.06776v1)

**Authors**: Martin Figura, Yixuan Lin, Ji Liu, Vijay Gupta

**Abstracts**: Adversarial attacks during training can strongly influence the performance of multi-agent reinforcement learning algorithms. It is, thus, highly desirable to augment existing algorithms such that the impact of adversarial attacks on cooperative networks is eliminated, or at least bounded. In this work, we consider a fully decentralized network, where each agent receives a local reward and observes the global state and action. We propose a resilient consensus-based actor-critic algorithm, whereby each agent estimates the team-average reward and value function, and communicates the associated parameter vectors to its immediate neighbors. We show that in the presence of Byzantine agents, whose estimation and communication strategies are completely arbitrary, the estimates of the cooperative agents converge to a bounded consensus value with probability one, provided that there are at most $H$ Byzantine agents in the neighborhood of each cooperative agent and the network is $(2H+1)$-robust. Furthermore, we prove that the policy of the cooperative agents converges with probability one to a bounded neighborhood around a local maximizer of their team-average objective function under the assumption that the policies of the adversarial agents asymptotically become stationary.



## **2. Learning to Break Deep Perceptual Hashing: The Use Case NeuralHash**

cs.LG

22 pages, 15 figures, 5 tables

**SubmitDate**: 2021-11-12    [paper-pdf](http://arxiv.org/pdf/2111.06628v1)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Daniel Neider, Kristian Kersting

**Abstracts**: Apple recently revealed its deep perceptual hashing system NeuralHash to detect child sexual abuse material (CSAM) on user devices before files are uploaded to its iCloud service. Public criticism quickly arose regarding the protection of user privacy and the system's reliability. In this paper, we present the first comprehensive empirical analysis of deep perceptual hashing based on NeuralHash. Specifically, we show that current deep perceptual hashing may not be robust. An adversary can manipulate the hash values by applying slight changes in images, either induced by gradient-based approaches or simply by performing standard image transformations, forcing or preventing hash collisions. Such attacks permit malicious actors easily to exploit the detection system: from hiding abusive material to framing innocent users, everything is possible. Moreover, using the hash values, inferences can still be made about the data stored on user devices. In our view, based on our results, deep perceptual hashing in its current form is generally not ready for robust client-side scanning and should not be used from a privacy perspective.



## **3. Characterizing and Improving the Robustness of Self-Supervised Learning through Background Augmentations**

cs.CV

Technical Report; Additional Results

**SubmitDate**: 2021-11-12    [paper-pdf](http://arxiv.org/pdf/2103.12719v2)

**Authors**: Chaitanya K. Ryali, David J. Schwab, Ari S. Morcos

**Abstracts**: Recent progress in self-supervised learning has demonstrated promising results in multiple visual tasks. An important ingredient in high-performing self-supervised methods is the use of data augmentation by training models to place different augmented views of the same image nearby in embedding space. However, commonly used augmentation pipelines treat images holistically, ignoring the semantic relevance of parts of an image-e.g. a subject vs. a background-which can lead to the learning of spurious correlations. Our work addresses this problem by investigating a class of simple, yet highly effective "background augmentations", which encourage models to focus on semantically-relevant content by discouraging them from focusing on image backgrounds. Through a systematic investigation, we show that background augmentations lead to substantial improvements in performance across a spectrum of state-of-the-art self-supervised methods (MoCo-v2, BYOL, SwAV) on a variety of tasks, e.g. $\sim$+1-2% gains on ImageNet, enabling performance on par with the supervised baseline. Further, we find the improvement in limited-labels settings is even larger (up to 4.2%). Background augmentations also improve robustness to a number of distribution shifts, including natural adversarial examples, ImageNet-9, adversarial attacks, ImageNet-Renditions. We also make progress in completely unsupervised saliency detection, in the process of generating saliency masks used for background augmentations.



## **4. Distributionally Robust Trajectory Optimization Under Uncertain Dynamics via Relative Entropy Trust-Regions**

eess.SY

**SubmitDate**: 2021-11-11    [paper-pdf](http://arxiv.org/pdf/2103.15388v3)

**Authors**: Hany Abdulsamad, Tim Dorau, Boris Belousov, Jia-Jie Zhu, Jan Peters

**Abstracts**: Trajectory optimization and model predictive control are essential techniques underpinning advanced robotic applications, ranging from autonomous driving to full-body humanoid control. State-of-the-art algorithms have focused on data-driven approaches that infer the system dynamics online and incorporate posterior uncertainty during planning and control. Despite their success, such approaches are still susceptible to catastrophic errors that may arise due to statistical learning biases, unmodeled disturbances, or even directed adversarial attacks. In this paper, we tackle the problem of dynamics mismatch and propose a distributionally robust optimal control formulation that alternates between two relative entropy trust-region optimization problems. Our method finds the worst-case maximum entropy Gaussian posterior over the dynamics parameters and the corresponding robust policy. Furthermore, we show that our approach admits a closed-form backward-pass for a certain class of systems. Finally, we demonstrate the resulting robustness on linear and nonlinear numerical examples.



## **5. Poisoning Knowledge Graph Embeddings via Relation Inference Patterns**

cs.LG

Joint Conference of the 59th Annual Meeting of the Association for  Computational Linguistics and the 11th International Joint Conference on  Natural Language Processing (ACL-IJCNLP 2021)

**SubmitDate**: 2021-11-11    [paper-pdf](http://arxiv.org/pdf/2111.06345v1)

**Authors**: Peru Bhardwaj, John Kelleher, Luca Costabello, Declan O'Sullivan

**Abstracts**: We study the problem of generating data poisoning attacks against Knowledge Graph Embedding (KGE) models for the task of link prediction in knowledge graphs. To poison KGE models, we propose to exploit their inductive abilities which are captured through the relationship patterns like symmetry, inversion and composition in the knowledge graph. Specifically, to degrade the model's prediction confidence on target facts, we propose to improve the model's prediction confidence on a set of decoy facts. Thus, we craft adversarial additions that can improve the model's prediction confidence on decoy facts through different inference patterns. Our experiments demonstrate that the proposed poisoning attacks outperform state-of-art baselines on four KGE models for two publicly available datasets. We also find that the symmetry pattern based attacks generalize across all model-dataset combinations which indicates the sensitivity of KGE models to this pattern.



## **6. Qu-ANTI-zation: Exploiting Quantization Artifacts for Achieving Adversarial Outcomes**

cs.LG

Accepted to NeurIPS 2021 [Poster]

**SubmitDate**: 2021-11-11    [paper-pdf](http://arxiv.org/pdf/2110.13541v2)

**Authors**: Sanghyun Hong, Michael-Andrei Panaitescu-Liess, Yiğitcan Kaya, Tudor Dumitraş

**Abstracts**: Quantization is a popular technique that $transforms$ the parameter representation of a neural network from floating-point numbers into lower-precision ones ($e.g.$, 8-bit integers). It reduces the memory footprint and the computational cost at inference, facilitating the deployment of resource-hungry models. However, the parameter perturbations caused by this transformation result in $behavioral$ $disparities$ between the model before and after quantization. For example, a quantized model can misclassify some test-time samples that are otherwise classified correctly. It is not known whether such differences lead to a new security vulnerability. We hypothesize that an adversary may control this disparity to introduce specific behaviors that activate upon quantization. To study this hypothesis, we weaponize quantization-aware training and propose a new training framework to implement adversarial quantization outcomes. Following this framework, we present three attacks we carry out with quantization: (i) an indiscriminate attack for significant accuracy loss; (ii) a targeted attack against specific samples; and (iii) a backdoor attack for controlling the model with an input trigger. We further show that a single compromised model defeats multiple quantization schemes, including robust quantization techniques. Moreover, in a federated learning scenario, we demonstrate that a set of malicious participants who conspire can inject our quantization-activated backdoor. Lastly, we discuss potential counter-measures and show that only re-training consistently removes the attack artifacts. Our code is available at https://github.com/Secure-AI-Systems-Group/Qu-ANTI-zation



## **7. Robust Deep Reinforcement Learning through Adversarial Loss**

cs.LG

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2008.01976v2)

**Authors**: Tuomas Oikarinen, Wang Zhang, Alexandre Megretski, Luca Daniel, Tsui-Wei Weng

**Abstracts**: Recent studies have shown that deep reinforcement learning agents are vulnerable to small adversarial perturbations on the agent's inputs, which raises concerns about deploying such agents in the real world. To address this issue, we propose RADIAL-RL, a principled framework to train reinforcement learning agents with improved robustness against $l_p$-norm bounded adversarial attacks. Our framework is compatible with popular deep reinforcement learning algorithms and we demonstrate its performance with deep Q-learning, A3C and PPO. We experiment on three deep RL benchmarks (Atari, MuJoCo and ProcGen) to show the effectiveness of our robust training algorithm. Our RADIAL-RL agents consistently outperform prior methods when tested against attacks of varying strength and are more computationally efficient to train. In addition, we propose a new evaluation method called Greedy Worst-Case Reward (GWC) to measure attack agnostic robustness of deep RL agents. We show that GWC can be evaluated efficiently and is a good estimate of the reward under the worst possible sequence of adversarial attacks. All code used for our experiments is available at https://github.com/tuomaso/radial_rl_v2.



## **8. Trustworthy Medical Segmentation with Uncertainty Estimation**

eess.IV

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05978v1)

**Authors**: Giuseppina Carannante, Dimah Dera, Nidhal C. Bouaynaya, Rasool Ghulam, Hassan M. Fathallah-Shaykh

**Abstracts**: Deep Learning (DL) holds great promise in reshaping the healthcare systems given its precision, efficiency, and objectivity. However, the brittleness of DL models to noisy and out-of-distribution inputs is ailing their deployment in the clinic. Most systems produce point estimates without further information about model uncertainty or confidence. This paper introduces a new Bayesian deep learning framework for uncertainty quantification in segmentation neural networks, specifically encoder-decoder architectures. The proposed framework uses the first-order Taylor series approximation to propagate and learn the first two moments (mean and covariance) of the distribution of the model parameters given the training data by maximizing the evidence lower bound. The output consists of two maps: the segmented image and the uncertainty map of the segmentation. The uncertainty in the segmentation decisions is captured by the covariance matrix of the predictive distribution. We evaluate the proposed framework on medical image segmentation data from Magnetic Resonances Imaging and Computed Tomography scans. Our experiments on multiple benchmark datasets demonstrate that the proposed framework is more robust to noise and adversarial attacks as compared to state-of-the-art segmentation models. Moreover, the uncertainty map of the proposed framework associates low confidence (or equivalently high uncertainty) to patches in the test input images that are corrupted with noise, artifacts or adversarial attacks. Thus, the model can self-assess its segmentation decisions when it makes an erroneous prediction or misses part of the segmentation structures, e.g., tumor, by presenting higher values in the uncertainty map.



## **9. Robust Learning via Ensemble Density Propagation in Deep Neural Networks**

cs.LG

submitted to 2020 IEEE International Workshop on Machine Learning for  Signal Processing

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05953v1)

**Authors**: Giuseppina Carannante, Dimah Dera, Ghulam Rasool, Nidhal C. Bouaynaya, Lyudmila Mihaylova

**Abstracts**: Learning in uncertain, noisy, or adversarial environments is a challenging task for deep neural networks (DNNs). We propose a new theoretically grounded and efficient approach for robust learning that builds upon Bayesian estimation and Variational Inference. We formulate the problem of density propagation through layers of a DNN and solve it using an Ensemble Density Propagation (EnDP) scheme. The EnDP approach allows us to propagate moments of the variational probability distribution across the layers of a Bayesian DNN, enabling the estimation of the mean and covariance of the predictive distribution at the output of the model. Our experiments using MNIST and CIFAR-10 datasets show a significant improvement in the robustness of the trained models to random noise and adversarial attacks.



## **10. Audio Attacks and Defenses against AED Systems -- A Practical Study**

cs.SD

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2106.07428v4)

**Authors**: Rodrigo dos Santos, Shirin Nilizadeh

**Abstracts**: In this paper, we evaluate deep learning-enabled AED systems against evasion attacks based on adversarial examples. We test the robustness of multiple security critical AED tasks, implemented as CNNs classifiers, as well as existing third-party Nest devices, manufactured by Google, which run their own black-box deep learning models. Our adversarial examples use audio perturbations made of white and background noises. Such disturbances are easy to create, to perform and to reproduce, and can be accessible to a large number of potential attackers, even non-technically savvy ones.   We show that an adversary can focus on audio adversarial inputs to cause AED systems to misclassify, achieving high success rates, even when we use small levels of a given type of noisy disturbance. For instance, on the case of the gunshot sound class, we achieve nearly 100% success rate when employing as little as 0.05 white noise level. Similarly to what has been previously done by works focusing on adversarial examples from the image domain as well as on the speech recognition domain. We then, seek to improve classifiers' robustness through countermeasures. We employ adversarial training and audio denoising. We show that these countermeasures, when applied to audio input, can be successful, either in isolation or in combination, generating relevant increases of nearly fifty percent in the performance of the classifiers when these are under attack.



## **11. A black-box adversarial attack for poisoning clustering**

cs.LG

18 pages, Pattern Recognition 2022

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2009.05474v4)

**Authors**: Antonio Emanuele Cinà, Alessandro Torcinovich, Marcello Pelillo

**Abstracts**: Clustering algorithms play a fundamental role as tools in decision-making and sensible automation processes. Due to the widespread use of these applications, a robustness analysis of this family of algorithms against adversarial noise has become imperative. To the best of our knowledge, however, only a few works have currently addressed this problem. In an attempt to fill this gap, in this work, we propose a black-box adversarial attack for crafting adversarial samples to test the robustness of clustering algorithms. We formulate the problem as a constrained minimization program, general in its structure and customizable by the attacker according to her capability constraints. We do not assume any information about the internal structure of the victim clustering algorithm, and we allow the attacker to query it as a service only. In the absence of any derivative information, we perform the optimization with a custom approach inspired by the Abstract Genetic Algorithm (AGA). In the experimental part, we demonstrate the sensibility of different single and ensemble clustering algorithms against our crafted adversarial samples on different scenarios. Furthermore, we perform a comparison of our algorithm with a state-of-the-art approach showing that we are able to reach or even outperform its performance. Finally, to highlight the general nature of the generated noise, we show that our attacks are transferable even against supervised algorithms such as SVMs, random forests, and neural networks.



## **12. Universal Multi-Party Poisoning Attacks**

cs.LG

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/1809.03474v3)

**Authors**: Saeed Mahloujifar, Mohammad Mahmoody, Ameer Mohammed

**Abstracts**: In this work, we demonstrate universal multi-party poisoning attacks that adapt and apply to any multi-party learning process with arbitrary interaction pattern between the parties. More generally, we introduce and study $(k,p)$-poisoning attacks in which an adversary controls $k\in[m]$ of the parties, and for each corrupted party $P_i$, the adversary submits some poisoned data $\mathcal{T}'_i$ on behalf of $P_i$ that is still ``$(1-p)$-close'' to the correct data $\mathcal{T}_i$ (e.g., $1-p$ fraction of $\mathcal{T}'_i$ is still honestly generated). We prove that for any ``bad'' property $B$ of the final trained hypothesis $h$ (e.g., $h$ failing on a particular test example or having ``large'' risk) that has an arbitrarily small constant probability of happening without the attack, there always is a $(k,p)$-poisoning attack that increases the probability of $B$ from $\mu$ to by $\mu^{1-p \cdot k/m} = \mu + \Omega(p \cdot k/m)$. Our attack only uses clean labels, and it is online.   More generally, we prove that for any bounded function $f(x_1,\dots,x_n) \in [0,1]$ defined over an $n$-step random process $\mathbf{X} = (x_1,\dots,x_n)$, an adversary who can override each of the $n$ blocks with even dependent probability $p$ can increase the expected output by at least $\Omega(p \cdot \mathrm{Var}[f(\mathbf{x})])$.



## **13. Sparse Adversarial Video Attacks with Spatial Transformations**

cs.CV

The short version of this work will appear in the BMVC 2021  conference

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05468v1)

**Authors**: Ronghui Mu, Wenjie Ruan, Leandro Soriano Marcolino, Qiang Ni

**Abstracts**: In recent years, a significant amount of research efforts concentrated on adversarial attacks on images, while adversarial video attacks have seldom been explored. We propose an adversarial attack strategy on videos, called DeepSAVA. Our model includes both additive perturbation and spatial transformation by a unified optimisation framework, where the structural similarity index (SSIM) measure is adopted to measure the adversarial distance. We design an effective and novel optimisation scheme which alternatively utilizes Bayesian optimisation to identify the most influential frame in a video and Stochastic gradient descent (SGD) based optimisation to produce both additive and spatial-transformed perturbations. Doing so enables DeepSAVA to perform a very sparse attack on videos for maintaining human imperceptibility while still achieving state-of-the-art performance in terms of both attack success rate and adversarial transferability. Our intensive experiments on various types of deep neural networks and video datasets confirm the superiority of DeepSAVA.



## **14. Are Transformers More Robust Than CNNs?**

cs.CV

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05464v1)

**Authors**: Yutong Bai, Jieru Mei, Alan Yuille, Cihang Xie

**Abstracts**: Transformer emerges as a powerful tool for visual recognition. In addition to demonstrating competitive performance on a broad range of visual benchmarks, recent works also argue that Transformers are much more robust than Convolutions Neural Networks (CNNs). Nonetheless, surprisingly, we find these conclusions are drawn from unfair experimental settings, where Transformers and CNNs are compared at different scales and are applied with distinct training frameworks. In this paper, we aim to provide the first fair & in-depth comparisons between Transformers and CNNs, focusing on robustness evaluations.   With our unified training setup, we first challenge the previous belief that Transformers outshine CNNs when measuring adversarial robustness. More surprisingly, we find CNNs can easily be as robust as Transformers on defending against adversarial attacks, if they properly adopt Transformers' training recipes. While regarding generalization on out-of-distribution samples, we show pre-training on (external) large-scale datasets is not a fundamental request for enabling Transformers to achieve better performance than CNNs. Moreover, our ablations suggest such stronger generalization is largely benefited by the Transformer's self-attention-like architectures per se, rather than by other training setups. We hope this work can help the community better understand and benchmark the robustness of Transformers and CNNs. The code and models are publicly available at https://github.com/ytongbai/ViTs-vs-CNNs.



## **15. Statistical Perspectives on Reliability of Artificial Intelligence Systems**

cs.SE

40 pages

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2111.05391v1)

**Authors**: Yili Hong, Jiayi Lian, Li Xu, Jie Min, Yueyao Wang, Laura J. Freeman, Xinwei Deng

**Abstracts**: Artificial intelligence (AI) systems have become increasingly popular in many areas. Nevertheless, AI technologies are still in their developing stages, and many issues need to be addressed. Among those, the reliability of AI systems needs to be demonstrated so that the AI systems can be used with confidence by the general public. In this paper, we provide statistical perspectives on the reliability of AI systems. Different from other considerations, the reliability of AI systems focuses on the time dimension. That is, the system can perform its designed functionality for the intended period. We introduce a so-called SMART statistical framework for AI reliability research, which includes five components: Structure of the system, Metrics of reliability, Analysis of failure causes, Reliability assessment, and Test planning. We review traditional methods in reliability data analysis and software reliability, and discuss how those existing methods can be transformed for reliability modeling and assessment of AI systems. We also describe recent developments in modeling and analysis of AI reliability and outline statistical research challenges in this area, including out-of-distribution detection, the effect of the training set, adversarial attacks, model accuracy, and uncertainty quantification, and discuss how those topics can be related to AI reliability, with illustrative examples. Finally, we discuss data collection and test planning for AI reliability assessment and how to improve system designs for higher AI reliability. The paper closes with some concluding remarks.



## **16. TDGIA:Effective Injection Attacks on Graph Neural Networks**

cs.LG

KDD 2021 research track paper

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2106.06663v2)

**Authors**: Xu Zou, Qinkai Zheng, Yuxiao Dong, Xinyu Guan, Evgeny Kharlamov, Jialiang Lu, Jie Tang

**Abstracts**: Graph Neural Networks (GNNs) have achieved promising performance in various real-world applications. However, recent studies have shown that GNNs are vulnerable to adversarial attacks. In this paper, we study a recently-introduced realistic attack scenario on graphs -- graph injection attack (GIA). In the GIA scenario, the adversary is not able to modify the existing link structure and node attributes of the input graph, instead the attack is performed by injecting adversarial nodes into it. We present an analysis on the topological vulnerability of GNNs under GIA setting, based on which we propose the Topological Defective Graph Injection Attack (TDGIA) for effective injection attacks. TDGIA first introduces the topological defective edge selection strategy to choose the original nodes for connecting with the injected ones. It then designs the smooth feature optimization objective to generate the features for the injected nodes. Extensive experiments on large-scale datasets show that TDGIA can consistently and significantly outperform various attack baselines in attacking dozens of defense GNN models. Notably, the performance drop on target GNNs resultant from TDGIA is more than double the damage brought by the best attack solution among hundreds of submissions on KDD-CUP 2020.



## **17. A Unified Game-Theoretic Interpretation of Adversarial Robustness**

cs.LG

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2103.07364v2)

**Authors**: Jie Ren, Die Zhang, Yisen Wang, Lu Chen, Zhanpeng Zhou, Yiting Chen, Xu Cheng, Xin Wang, Meng Zhou, Jie Shi, Quanshi Zhang

**Abstracts**: This paper provides a unified view to explain different adversarial attacks and defense methods, i.e. the view of multi-order interactions between input variables of DNNs. Based on the multi-order interaction, we discover that adversarial attacks mainly affect high-order interactions to fool the DNN. Furthermore, we find that the robustness of adversarially trained DNNs comes from category-specific low-order interactions. Our findings provide a potential method to unify adversarial perturbations and robustness, which can explain the existing defense methods in a principle way. Besides, our findings also make a revision of previous inaccurate understanding of the shape bias of adversarially learned features.



## **18. Membership Inference Attacks Against Self-supervised Speech Models**

cs.CR

Submitted to ICASSP 2022. Source code available at  https://github.com/RayTzeng/s3m-membership-inference

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2111.05113v1)

**Authors**: Wei-Cheng Tseng, Wei-Tsung Kao, Hung-yi Lee

**Abstracts**: Recently, adapting the idea of self-supervised learning (SSL) on continuous speech has started gaining attention. SSL models pre-trained on a huge amount of unlabeled audio can generate general-purpose representations that benefit a wide variety of speech processing tasks. Despite their ubiquitous deployment, however, the potential privacy risks of these models have not been well investigated. In this paper, we present the first privacy analysis on several SSL speech models using Membership Inference Attacks (MIA) under black-box access. The experiment results show that these pre-trained models are vulnerable to MIA and prone to membership information leakage with high adversarial advantage scores in both utterance-level and speaker-level. Furthermore, we also conduct several ablation studies to understand the factors that contribute to the success of MIA.



## **19. A Statistical Difference Reduction Method for Escaping Backdoor Detection**

cs.LG

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2111.05077v1)

**Authors**: Pengfei Xia, Hongjing Niu, Ziqiang Li, Bin Li

**Abstracts**: Recent studies show that Deep Neural Networks (DNNs) are vulnerable to backdoor attacks. An infected model behaves normally on benign inputs, whereas its prediction will be forced to an attack-specific target on adversarial data. Several detection methods have been developed to distinguish inputs to defend against such attacks. The common hypothesis that these defenses rely on is that there are large statistical differences between the latent representations of clean and adversarial inputs extracted by the infected model. However, although it is important, comprehensive research on whether the hypothesis must be true is lacking. In this paper, we focus on it and study the following relevant questions: 1) What are the properties of the statistical differences? 2) How to effectively reduce them without harming the attack intensity? 3) What impact does this reduction have on difference-based defenses? Our work is carried out on the three questions. First, by introducing the Maximum Mean Discrepancy (MMD) as the metric, we identify that the statistical differences of multi-level representations are all large, not just the highest level. Then, we propose a Statistical Difference Reduction Method (SDRM) by adding a multi-level MMD constraint to the loss function during training a backdoor model to effectively reduce the differences. Last, three typical difference-based detection methods are examined. The F1 scores of these defenses drop from 90%-100% on the regularly trained backdoor models to 60%-70% on the models trained with SDRM on all two datasets, four model architectures, and four attack methods. The results indicate that the proposed method can be used to enhance existing attacks to escape backdoor detection algorithms.



## **20. Tightening the Approximation Error of Adversarial Risk with Auto Loss Function Search**

cs.LG

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2111.05063v1)

**Authors**: Pengfei Xia, Ziqiang Li, Bin Li

**Abstracts**: Numerous studies have demonstrated that deep neural networks are easily misled by adversarial examples. Effectively evaluating the adversarial robustness of a model is important for its deployment in practical applications. Currently, a common type of evaluation is to approximate the adversarial risk of a model as a robustness indicator by constructing malicious instances and executing attacks. Unfortunately, there is an error (gap) between the approximate value and the true value. Previous studies manually design attack methods to achieve a smaller error, which is inefficient and may miss a better solution. In this paper, we establish the tightening of the approximation error as an optimization problem and try to solve it with an algorithm. More specifically, we first analyze that replacing the non-convex and discontinuous 0-1 loss with a surrogate loss, a necessary compromise in calculating the approximation, is one of the main reasons for the error. Then we propose AutoLoss-AR, the first method for searching loss functions for tightening the approximation error of adversarial risk. Extensive experiments are conducted in multiple settings. The results demonstrate the effectiveness of the proposed method: the best-discovered loss functions outperform the handcrafted baseline by 0.9%-2.9% and 0.7%-2.0% on MNIST and CIFAR-10, respectively. Besides, we also verify that the searched losses can be transferred to other settings and explore why they are better than the baseline by visualizing the local loss landscape.



## **21. GraphAttacker: A General Multi-Task GraphAttack Framework**

cs.LG

17 pages,9 figeures

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2101.06855v2)

**Authors**: Jinyin Chen, Dunjie Zhang, Zhaoyan Ming, Kejie Huang, Wenrong Jiang, Chen Cui

**Abstracts**: Graph neural networks (GNNs) have been successfully exploited in graph analysis tasks in many real-world applications. The competition between attack and defense methods also enhances the robustness of GNNs. In this competition, the development of adversarial training methods put forward higher requirement for the diversity of attack examples. By contrast, most attack methods with specific attack strategies are difficult to satisfy such a requirement. To address this problem, we propose GraphAttacker, a novel generic graph attack framework that can flexibly adjust the structures and the attack strategies according to the graph analysis tasks. GraphAttacker generates adversarial examples through alternate training on three key components: the multi-strategy attack generator (MAG), the similarity discriminator (SD), and the attack discriminator (AD), based on the generative adversarial network (GAN). Furthermore, we introduce a novel similarity modification rate SMR to conduct a stealthier attack considering the change of node similarity distribution. Experiments on various benchmark datasets demonstrate that GraphAttacker can achieve state-of-the-art attack performance on graph analysis tasks of node classification, graph classification, and link prediction, no matter the adversarial training is conducted or not. Moreover, we also analyze the unique characteristics of each task and their specific response in the unified attack framework. The project code is available at https://github.com/honoluluuuu/GraphAttacker.



## **22. Reversible Attack based on Local Visual Adversarial Perturbation**

cs.CV

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2110.02700v2)

**Authors**: Li Chen, Shaowei Zhu, Zhaoxia Yin

**Abstracts**: Deep learning is getting more and more outstanding performance in many tasks such as autonomous driving and face recognition and also has been challenged by different kinds of attacks. Adding perturbations that are imperceptible to human vision in an image can mislead the neural network model to get wrong results with high confidence. Adversarial Examples are images that have been added with specific noise to mislead a deep neural network model However, adding noise to images destroys the original data, making the examples useless in digital forensics and other fields. To prevent illegal or unauthorized access of image data such as human faces and ensure no affection to legal use reversible adversarial attack technique is rise. The original image can be recovered from its reversible adversarial example. However, the existing reversible adversarial examples generation strategies are all designed for the traditional imperceptible adversarial perturbation. How to get reversibility for locally visible adversarial perturbation? In this paper, we propose a new method for generating reversible adversarial examples based on local visual adversarial perturbation. The information needed for image recovery is embedded into the area beyond the adversarial patch by reversible data hiding technique. To reduce image distortion and improve visual quality, lossless compression and B-R-G embedding principle are adopted. Experiments on ImageNet dataset show that our method can restore the original images error-free while ensuring the attack performance.



## **23. On Robustness of Neural Ordinary Differential Equations**

cs.LG

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/1910.05513v3)

**Authors**: Hanshu Yan, Jiawei Du, Vincent Y. F. Tan, Jiashi Feng

**Abstracts**: Neural ordinary differential equations (ODEs) have been attracting increasing attention in various research domains recently. There have been some works studying optimization issues and approximation capabilities of neural ODEs, but their robustness is still yet unclear. In this work, we fill this important gap by exploring robustness properties of neural ODEs both empirically and theoretically. We first present an empirical study on the robustness of the neural ODE-based networks (ODENets) by exposing them to inputs with various types of perturbations and subsequently investigating the changes of the corresponding outputs. In contrast to conventional convolutional neural networks (CNNs), we find that the ODENets are more robust against both random Gaussian perturbations and adversarial attack examples. We then provide an insightful understanding of this phenomenon by exploiting a certain desirable property of the flow of a continuous-time ODE, namely that integral curves are non-intersecting. Our work suggests that, due to their intrinsic robustness, it is promising to use neural ODEs as a basic block for building robust deep network models. To further enhance the robustness of vanilla neural ODEs, we propose the time-invariant steady neural ODE (TisODE), which regularizes the flow on perturbed data via the time-invariant property and the imposition of a steady-state constraint. We show that the TisODE method outperforms vanilla neural ODEs and also can work in conjunction with other state-of-the-art architectural methods to build more robust deep networks. \url{https://github.com/HanshuYAN/TisODE}



## **24. Bayesian Framework for Gradient Leakage**

cs.LG

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04706v1)

**Authors**: Mislav Balunović, Dimitar I. Dimitrov, Robin Staab, Martin Vechev

**Abstracts**: Federated learning is an established method for training machine learning models without sharing training data. However, recent work has shown that it cannot guarantee data privacy as shared gradients can still leak sensitive information. To formalize the problem of gradient leakage, we propose a theoretical framework that enables, for the first time, analysis of the Bayes optimal adversary phrased as an optimization problem. We demonstrate that existing leakage attacks can be seen as approximations of this optimal adversary with different assumptions on the probability distributions of the input data and gradients. Our experiments confirm the effectiveness of the Bayes optimal adversary when it has knowledge of the underlying distribution. Further, our experimental evaluation shows that several existing heuristic defenses are not effective against stronger attacks, especially early in the training process. Thus, our findings indicate that the construction of more effective defenses and their evaluation remains an open problem.



## **25. HAPSSA: Holistic Approach to PDF Malware Detection Using Signal and Statistical Analysis**

cs.CR

Submitted version - MILCOM 2021 IEEE Military Communications  Conference

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04703v1)

**Authors**: Tajuddin Manhar Mohammed, Lakshmanan Nataraj, Satish Chikkagoudar, Shivkumar Chandrasekaran, B. S. Manjunath

**Abstracts**: Malicious PDF documents present a serious threat to various security organizations that require modern threat intelligence platforms to effectively analyze and characterize the identity and behavior of PDF malware. State-of-the-art approaches use machine learning (ML) to learn features that characterize PDF malware. However, ML models are often susceptible to evasion attacks, in which an adversary obfuscates the malware code to avoid being detected by an Antivirus. In this paper, we derive a simple yet effective holistic approach to PDF malware detection that leverages signal and statistical analysis of malware binaries. This includes combining orthogonal feature space models from various static and dynamic malware detection methods to enable generalized robustness when faced with code obfuscations. Using a dataset of nearly 30,000 PDF files containing both malware and benign samples, we show that our holistic approach maintains a high detection rate (99.92%) of PDF malware and even detects new malicious files created by simple methods that remove the obfuscation conducted by malware authors to hide their malware, which are undetected by most antiviruses.



## **26. A Separation Result Between Data-oblivious and Data-aware Poisoning Attacks**

cs.LG

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2003.12020v2)

**Authors**: Samuel Deng, Sanjam Garg, Somesh Jha, Saeed Mahloujifar, Mohammad Mahmoody, Abhradeep Thakurta

**Abstracts**: Poisoning attacks have emerged as a significant security threat to machine learning algorithms. It has been demonstrated that adversaries who make small changes to the training set, such as adding specially crafted data points, can hurt the performance of the output model. Some of the stronger poisoning attacks require the full knowledge of the training data. This leaves open the possibility of achieving the same attack results using poisoning attacks that do not have the full knowledge of the clean training set.   In this work, we initiate a theoretical study of the problem above. Specifically, for the case of feature selection with LASSO, we show that full-information adversaries (that craft poisoning examples based on the rest of the training data) are provably stronger than the optimal attacker that is oblivious to the training set yet has access to the distribution of the data. Our separation result shows that the two setting of data-aware and data-oblivious are fundamentally different and we cannot hope to always achieve the same attack or defense results in these scenarios.



## **27. DeepSteal: Advanced Model Extractions Leveraging Efficient Weight Stealing in Memories**

cs.CR

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04625v1)

**Authors**: Adnan Siraj Rakin, Md Hafizul Islam Chowdhuryy, Fan Yao, Deliang Fan

**Abstracts**: Recent advancements of Deep Neural Networks (DNNs) have seen widespread deployment in multiple security-sensitive domains. The need of resource-intensive training and use of valuable domain-specific training data have made these models a top intellectual property (IP) for model owners. One of the major threats to the DNN privacy is model extraction attacks where adversaries attempt to steal sensitive information in DNN models. Recent studies show hardware-based side channel attacks can reveal internal knowledge about DNN models (e.g., model architectures) However, to date, existing attacks cannot extract detailed model parameters (e.g., weights/biases). In this work, for the first time, we propose an advanced model extraction attack framework DeepSteal that effectively steals DNN weights with the aid of memory side-channel attack. Our proposed DeepSteal comprises two key stages. Firstly, we develop a new weight bit information extraction method, called HammerLeak, through adopting the rowhammer based hardware fault technique as the information leakage vector. HammerLeak leverages several novel system-level techniques tailed for DNN applications to enable fast and efficient weight stealing. Secondly, we propose a novel substitute model training algorithm with Mean Clustering weight penalty, which leverages the partial leaked bit information effectively and generates a substitute prototype of the target victim model. We evaluate this substitute model extraction method on three popular image datasets (e.g., CIFAR-10/100/GTSRB) and four DNN architectures (e.g., ResNet-18/34/Wide-ResNet/VGG-11). The extracted substitute model has successfully achieved more than 90 % test accuracy on deep residual networks for the CIFAR-10 dataset. Moreover, our extracted substitute model could also generate effective adversarial input samples to fool the victim model.



## **28. Better Safe Than Sorry: Preventing Delusive Adversaries with Adversarial Training**

cs.LG

NeurIPS 2021

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2102.04716v3)

**Authors**: Lue Tao, Lei Feng, Jinfeng Yi, Sheng-Jun Huang, Songcan Chen

**Abstracts**: Delusive attacks aim to substantially deteriorate the test accuracy of the learning model by slightly perturbing the features of correctly labeled training examples. By formalizing this malicious attack as finding the worst-case training data within a specific $\infty$-Wasserstein ball, we show that minimizing adversarial risk on the perturbed data is equivalent to optimizing an upper bound of natural risk on the original data. This implies that adversarial training can serve as a principled defense against delusive attacks. Thus, the test accuracy decreased by delusive attacks can be largely recovered by adversarial training. To further understand the internal mechanism of the defense, we disclose that adversarial training can resist the delusive perturbations by preventing the learner from overly relying on non-robust features in a natural setting. Finally, we complement our theoretical findings with a set of experiments on popular benchmark datasets, which show that the defense withstands six different practical attacks. Both theoretical and empirical results vote for adversarial training when confronted with delusive adversaries.



## **29. Robust and Information-theoretically Safe Bias Classifier against Adversarial Attacks**

cs.LG

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04404v1)

**Authors**: Lijia Yu, Xiao-Shan Gao

**Abstracts**: In this paper, the bias classifier is introduced, that is, the bias part of a DNN with Relu as the activation function is used as a classifier. The work is motivated by the fact that the bias part is a piecewise constant function with zero gradient and hence cannot be directly attacked by gradient-based methods to generate adversaries such as FGSM. The existence of the bias classifier is proved an effective training method for the bias classifier is proposed. It is proved that by adding a proper random first-degree part to the bias classifier, an information-theoretically safe classifier against the original-model gradient-based attack is obtained in the sense that the attack generates a totally random direction for generating adversaries. This seems to be the first time that the concept of information-theoretically safe classifier is proposed. Several attack methods for the bias classifier are proposed and numerical experiments are used to show that the bias classifier is more robust than DNNs against these attacks in most cases.



## **30. Get a Model! Model Hijacking Attack Against Machine Learning Models**

cs.CR

To Appear in NDSS 2022

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04394v1)

**Authors**: Ahmed Salem, Michael Backes, Yang Zhang

**Abstracts**: Machine learning (ML) has established itself as a cornerstone for various critical applications ranging from autonomous driving to authentication systems. However, with this increasing adoption rate of machine learning models, multiple attacks have emerged. One class of such attacks is training time attack, whereby an adversary executes their attack before or during the machine learning model training. In this work, we propose a new training time attack against computer vision based machine learning models, namely model hijacking attack. The adversary aims to hijack a target model to execute a different task than its original one without the model owner noticing. Model hijacking can cause accountability and security risks since a hijacked model owner can be framed for having their model offering illegal or unethical services. Model hijacking attacks are launched in the same way as existing data poisoning attacks. However, one requirement of the model hijacking attack is to be stealthy, i.e., the data samples used to hijack the target model should look similar to the model's original training dataset. To this end, we propose two different model hijacking attacks, namely Chameleon and Adverse Chameleon, based on a novel encoder-decoder style ML model, namely the Camouflager. Our evaluation shows that both of our model hijacking attacks achieve a high attack success rate, with a negligible drop in model utility.



## **31. Geometrically Adaptive Dictionary Attack on Face Recognition**

cs.CV

Accepted at WACV 2022

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04371v1)

**Authors**: Junyoung Byun, Hyojun Go, Changick Kim

**Abstracts**: CNN-based face recognition models have brought remarkable performance improvement, but they are vulnerable to adversarial perturbations. Recent studies have shown that adversaries can fool the models even if they can only access the models' hard-label output. However, since many queries are needed to find imperceptible adversarial noise, reducing the number of queries is crucial for these attacks. In this paper, we point out two limitations of existing decision-based black-box attacks. We observe that they waste queries for background noise optimization, and they do not take advantage of adversarial perturbations generated for other images. We exploit 3D face alignment to overcome these limitations and propose a general strategy for query-efficient black-box attacks on face recognition named Geometrically Adaptive Dictionary Attack (GADA). Our core idea is to create an adversarial perturbation in the UV texture map and project it onto the face in the image. It greatly improves query efficiency by limiting the perturbation search space to the facial area and effectively recycling previous perturbations. We apply the GADA strategy to two existing attack methods and show overwhelming performance improvement in the experiments on the LFW and CPLFW datasets. Furthermore, we also present a novel attack strategy that can circumvent query similarity-based stateful detection that identifies the process of query-based black-box attacks.



## **32. Robustness of Graph Neural Networks at Scale**

cs.LG

39 pages, 22 figures, 17 tables NeurIPS 2021

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2110.14038v3)

**Authors**: Simon Geisler, Tobias Schmidt, Hakan Şirin, Daniel Zügner, Aleksandar Bojchevski, Stephan Günnemann

**Abstracts**: Graph Neural Networks (GNNs) are increasingly important given their popularity and the diversity of applications. Yet, existing studies of their vulnerability to adversarial attacks rely on relatively small graphs. We address this gap and study how to attack and defend GNNs at scale. We propose two sparsity-aware first-order optimization attacks that maintain an efficient representation despite optimizing over a number of parameters which is quadratic in the number of nodes. We show that common surrogate losses are not well-suited for global attacks on GNNs. Our alternatives can double the attack strength. Moreover, to improve GNNs' reliability we design a robust aggregation function, Soft Median, resulting in an effective defense at all scales. We evaluate our attacks and defense with standard GNNs on graphs more than 100 times larger compared to previous work. We even scale one order of magnitude further by extending our techniques to a scalable GNN.



## **33. Characterizing the adversarial vulnerability of speech self-supervised learning**

cs.SD

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04330v1)

**Authors**: Haibin Wu, Bo Zheng, Xu Li, Xixin Wu, Hung-yi Lee, Helen Meng

**Abstracts**: A leaderboard named Speech processing Universal PERformance Benchmark (SUPERB), which aims at benchmarking the performance of a shared self-supervised learning (SSL) speech model across various downstream speech tasks with minimal modification of architectures and small amount of data, has fueled the research for speech representation learning. The SUPERB demonstrates speech SSL upstream models improve the performance of various downstream tasks through just minimal adaptation. As the paradigm of the self-supervised learning upstream model followed by downstream tasks arouses more attention in the speech community, characterizing the adversarial robustness of such paradigm is of high priority. In this paper, we make the first attempt to investigate the adversarial vulnerability of such paradigm under the attacks from both zero-knowledge adversaries and limited-knowledge adversaries. The experimental results illustrate that the paradigm proposed by SUPERB is seriously vulnerable to limited-knowledge adversaries, and the attacks generated by zero-knowledge adversaries are with transferability. The XAB test verifies the imperceptibility of crafted adversarial attacks.



## **34. Graph Robustness Benchmark: Benchmarking the Adversarial Robustness of Graph Machine Learning**

cs.LG

21 pages, 12 figures, NeurIPS 2021 Datasets and Benchmarks Track

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04314v1)

**Authors**: Qinkai Zheng, Xu Zou, Yuxiao Dong, Yukuo Cen, Da Yin, Jiarong Xu, Yang Yang, Jie Tang

**Abstracts**: Adversarial attacks on graphs have posed a major threat to the robustness of graph machine learning (GML) models. Naturally, there is an ever-escalating arms race between attackers and defenders. However, the strategies behind both sides are often not fairly compared under the same and realistic conditions. To bridge this gap, we present the Graph Robustness Benchmark (GRB) with the goal of providing a scalable, unified, modular, and reproducible evaluation for the adversarial robustness of GML models. GRB standardizes the process of attacks and defenses by 1) developing scalable and diverse datasets, 2) modularizing the attack and defense implementations, and 3) unifying the evaluation protocol in refined scenarios. By leveraging the GRB pipeline, the end-users can focus on the development of robust GML models with automated data processing and experimental evaluations. To support open and reproducible research on graph adversarial learning, GRB also hosts public leaderboards across different scenarios. As a starting point, we conduct extensive experiments to benchmark baseline techniques. GRB is open-source and welcomes contributions from the community. Datasets, codes, leaderboards are available at https://cogdl.ai/grb/home.



## **35. DeepMoM: Robust Deep Learning With Median-of-Means**

stat.ML

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2105.14035v2)

**Authors**: Shih-Ting Huang, Johannes Lederer

**Abstracts**: Data used in deep learning is notoriously problematic. For example, data are usually combined from diverse sources, rarely cleaned and vetted thoroughly, and sometimes corrupted on purpose. Intentional corruption that targets the weak spots of algorithms has been studied extensively under the label of "adversarial attacks." In contrast, the arguably much more common case of corruption that reflects the limited quality of data has been studied much less. Such "random" corruptions are due to measurement errors, unreliable sources, convenience sampling, and so forth. These kinds of corruption are common in deep learning, because data are rarely collected according to strict protocols -- in strong contrast to the formalized data collection in some parts of classical statistics. This paper concerns such corruption. We introduce an approach motivated by very recent insights into median-of-means and Le Cam's principle, we show that the approach can be readily implemented, and we demonstrate that it performs very well in practice. In conclusion, we believe that our approach is a very promising alternative to standard parameter training based on least-squares and cross-entropy loss.



## **36. On the Effectiveness of Small Input Noise for Defending Against Query-based Black-Box Attacks**

cs.CR

Accepted at WACV 2022

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2101.04829v2)

**Authors**: Junyoung Byun, Hyojun Go, Changick Kim

**Abstracts**: While deep neural networks show unprecedented performance in various tasks, the vulnerability to adversarial examples hinders their deployment in safety-critical systems. Many studies have shown that attacks are also possible even in a black-box setting where an adversary cannot access the target model's internal information. Most black-box attacks are based on queries, each of which obtains the target model's output for an input, and many recent studies focus on reducing the number of required queries. In this paper, we pay attention to an implicit assumption of query-based black-box adversarial attacks that the target model's output exactly corresponds to the query input. If some randomness is introduced into the model, it can break the assumption, and thus, query-based attacks may have tremendous difficulty in both gradient estimation and local search, which are the core of their attack process. From this motivation, we observe even a small additive input noise can neutralize most query-based attacks and name this simple yet effective approach Small Noise Defense (SND). We analyze how SND can defend against query-based black-box attacks and demonstrate its effectiveness against eight state-of-the-art attacks with CIFAR-10 and ImageNet datasets. Even with strong defense ability, SND almost maintains the original classification accuracy and computational speed. SND is readily applicable to pre-trained models by adding only one line of code at the inference.



## **37. Defense Against Explanation Manipulation**

cs.LG

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04303v1)

**Authors**: Ruixiang Tang, Ninghao Liu, Fan Yang, Na Zou, Xia Hu

**Abstracts**: Explainable machine learning attracts increasing attention as it improves transparency of models, which is helpful for machine learning to be trusted in real applications. However, explanation methods have recently been demonstrated to be vulnerable to manipulation, where we can easily change a model's explanation while keeping its prediction constant. To tackle this problem, some efforts have been paid to use more stable explanation methods or to change model configurations. In this work, we tackle the problem from the training perspective, and propose a new training scheme called Adversarial Training on EXplanations (ATEX) to improve the internal explanation stability of a model regardless of the specific explanation method being applied. Instead of directly specifying explanation values over data instances, ATEX only puts requirement on model predictions which avoids involving second-order derivatives in optimization. As a further discussion, we also find that explanation stability is closely related to another property of the model, i.e., the risk of being exposed to adversarial attack. Through experiments, besides showing that ATEX improves model robustness against manipulation targeting explanation, it also brings additional benefits including smoothing explanations and improving the efficacy of adversarial training if applied to the model.



## **38. A Unified Game-Theoretic Interpretation of Adversarial Robustness**

cs.LG

the previous version is arXiv:2103.07364, but I mistakenly apply a  new ID for the paper

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.03536v2)

**Authors**: Jie Ren, Die Zhang, Yisen Wang, Lu Chen, Zhanpeng Zhou, Yiting Chen, Xu Cheng, Xin Wang, Meng Zhou, Jie Shi, Quanshi Zhang

**Abstracts**: This paper provides a unified view to explain different adversarial attacks and defense methods, \emph{i.e.} the view of multi-order interactions between input variables of DNNs. Based on the multi-order interaction, we discover that adversarial attacks mainly affect high-order interactions to fool the DNN. Furthermore, we find that the robustness of adversarially trained DNNs comes from category-specific low-order interactions. Our findings provide a potential method to unify adversarial perturbations and robustness, which can explain the existing defense methods in a principle way. Besides, our findings also make a revision of previous inaccurate understanding of the shape bias of adversarially learned features.



## **39. Generative Dynamic Patch Attack**

cs.CV

Published as a conference paper at BMVC 2021

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04266v1)

**Authors**: Xiang Li, Shihao Ji

**Abstracts**: Adversarial patch attack is a family of attack algorithms that perturb a part of image to fool a deep neural network model. Existing patch attacks mostly consider injecting adversarial patches at input-agnostic locations: either a predefined location or a random location. This attack setup may be sufficient for attack but has considerable limitations when using it for adversarial training. Thus, robust models trained with existing patch attacks cannot effectively defend other adversarial attacks. In this paper, we first propose an end-to-end patch attack algorithm, Generative Dynamic Patch Attack (GDPA), which generates both patch pattern and patch location adversarially for each input image. We show that GDPA is a generic attack framework that can produce dynamic/static and visible/invisible patches with a few configuration changes. Secondly, GDPA can be readily integrated for adversarial training to improve model robustness to various adversarial attacks. Extensive experiments on VGGFace, Traffic Sign and ImageNet show that GDPA achieves higher attack success rates than state-of-the-art patch attacks, while adversarially trained model with GDPA demonstrates superior robustness to adversarial patch attacks than competing methods. Our source code can be found at https://github.com/lxuniverse/gdpa.



## **40. Natural Adversarial Objects**

cs.CV

**SubmitDate**: 2021-11-07    [paper-pdf](http://arxiv.org/pdf/2111.04204v1)

**Authors**: Felix Lau, Nishant Subramani, Sasha Harrison, Aerin Kim, Elliot Branson, Rosanne Liu

**Abstracts**: Although state-of-the-art object detection methods have shown compelling performance, models often are not robust to adversarial attacks and out-of-distribution data. We introduce a new dataset, Natural Adversarial Objects (NAO), to evaluate the robustness of object detection models. NAO contains 7,934 images and 9,943 objects that are unmodified and representative of real-world scenarios, but cause state-of-the-art detection models to misclassify with high confidence. The mean average precision (mAP) of EfficientDet-D7 drops 74.5% when evaluated on NAO compared to the standard MSCOCO validation set.   Moreover, by comparing a variety of object detection architectures, we find that better performance on MSCOCO validation set does not necessarily translate to better performance on NAO, suggesting that robustness cannot be simply achieved by training a more accurate model.   We further investigate why examples in NAO are difficult to detect and classify. Experiments of shuffling image patches reveal that models are overly sensitive to local texture. Additionally, using integrated gradients and background replacement, we find that the detection model is reliant on pixel information within the bounding box, and insensitive to the background context when predicting class labels. NAO can be downloaded at https://drive.google.com/drive/folders/15P8sOWoJku6SSEiHLEts86ORfytGezi8.



## **41. Adversarial Attacks on Multi-task Visual Perception for Autonomous Driving**

cs.CV

Accepted for publication at Journal of Imaging Science and Technology

**SubmitDate**: 2021-11-07    [paper-pdf](http://arxiv.org/pdf/2107.07449v2)

**Authors**: Ibrahim Sobh, Ahmed Hamed, Varun Ravi Kumar, Senthil Yogamani

**Abstracts**: Deep neural networks (DNNs) have accomplished impressive success in various applications, including autonomous driving perception tasks, in recent years. On the other hand, current deep neural networks are easily fooled by adversarial attacks. This vulnerability raises significant concerns, particularly in safety-critical applications. As a result, research into attacking and defending DNNs has gained much coverage. In this work, detailed adversarial attacks are applied on a diverse multi-task visual perception deep network across distance estimation, semantic segmentation, motion detection, and object detection. The experiments consider both white and black box attacks for targeted and un-targeted cases, while attacking a task and inspecting the effect on all the others, in addition to inspecting the effect of applying a simple defense method. We conclude this paper by comparing and discussing the experimental results, proposing insights and future work. The visualizations of the attacks are available at https://youtu.be/6AixN90budY.



## **42. On the Convergence of Prior-Guided Zeroth-Order Optimization Algorithms**

stat.ML

NeurIPS 2021; code available at https://github.com/csy530216/pg-zoo

**SubmitDate**: 2021-11-07    [paper-pdf](http://arxiv.org/pdf/2107.10110v2)

**Authors**: Shuyu Cheng, Guoqiang Wu, Jun Zhu

**Abstracts**: Zeroth-order (ZO) optimization is widely used to handle challenging tasks, such as query-based black-box adversarial attacks and reinforcement learning. Various attempts have been made to integrate prior information into the gradient estimation procedure based on finite differences, with promising empirical results. However, their convergence properties are not well understood. This paper makes an attempt to fill up this gap by analyzing the convergence of prior-guided ZO algorithms under a greedy descent framework with various gradient estimators. We provide a convergence guarantee for the prior-guided random gradient-free (PRGF) algorithms. Moreover, to further accelerate over greedy descent methods, we present a new accelerated random search (ARS) algorithm that incorporates prior information, together with a convergence analysis. Finally, our theoretical results are confirmed by experiments on several numerical benchmarks as well as adversarial attacks.



## **43. On the Robustness of Domain Constraints**

cs.CR

Accepted to the 28th ACM Conference on Computer and Communications  Security. Seoul, South Korea

**SubmitDate**: 2021-11-07    [paper-pdf](http://arxiv.org/pdf/2105.08619v2)

**Authors**: Ryan Sheatsley, Blaine Hoak, Eric Pauley, Yohan Beugin, Michael J. Weisman, Patrick McDaniel

**Abstracts**: Machine learning is vulnerable to adversarial examples-inputs designed to cause models to perform poorly. However, it is unclear if adversarial examples represent realistic inputs in the modeled domains. Diverse domains such as networks and phishing have domain constraints-complex relationships between features that an adversary must satisfy for an attack to be realized (in addition to any adversary-specific goals). In this paper, we explore how domain constraints limit adversarial capabilities and how adversaries can adapt their strategies to create realistic (constraint-compliant) examples. In this, we develop techniques to learn domain constraints from data, and show how the learned constraints can be integrated into the adversarial crafting process. We evaluate the efficacy of our approach in network intrusion and phishing datasets and find: (1) up to 82% of adversarial examples produced by state-of-the-art crafting algorithms violate domain constraints, (2) domain constraints are robust to adversarial examples; enforcing constraints yields an increase in model accuracy by up to 34%. We observe not only that adversaries must alter inputs to satisfy domain constraints, but that these constraints make the generation of valid adversarial examples far more challenging.



## **44. Adversarial Robustness of Deep Code Comment Generation**

cs.SE

**SubmitDate**: 2021-11-07    [paper-pdf](http://arxiv.org/pdf/2108.00213v2)

**Authors**: Yu Zhou, Xiaoqing Zhang, Juanjuan Shen, Tingting Han, Taolue Chen, Harald Gall

**Abstracts**: Deep neural networks (DNNs) have shown remarkable performance in a variety of domains such as computer vision, speech recognition, or natural language processing. Recently they also have been applied to various software engineering tasks, typically involving processing source code. DNNs are well-known to be vulnerable to adversarial examples, i.e., fabricated inputs that could lead to various misbehaviors of the DNN model while being perceived as benign by humans. In this paper, we focus on the code comment generation task in software engineering and study the robustness issue of the DNNs when they are applied to this task. We propose ACCENT, an identifier substitution approach to craft adversarial code snippets, which are syntactically correct and semantically close to the original code snippet, but may mislead the DNNs to produce completely irrelevant code comments. In order to improve the robustness, ACCENT also incorporates a novel training method, which can be applied to existing code comment generation models. We conduct comprehensive experiments to evaluate our approach by attacking the mainstream encoder-decoder architectures on two large-scale publicly available datasets. The results show that ACCENT efficiently produces stable attacks with functionality-preserving adversarial examples, and the generated examples have better transferability compared with baselines. We also confirm, via experiments, the effectiveness in improving model robustness with our training method.



## **45. Drawing Robust Scratch Tickets: Subnetworks with Inborn Robustness Are Found within Randomly Initialized Networks**

cs.LG

Accepted at NeurIPS 2021

**SubmitDate**: 2021-11-06    [paper-pdf](http://arxiv.org/pdf/2110.14068v2)

**Authors**: Yonggan Fu, Qixuan Yu, Yang Zhang, Shang Wu, Xu Ouyang, David Cox, Yingyan Lin

**Abstracts**: Deep Neural Networks (DNNs) are known to be vulnerable to adversarial attacks, i.e., an imperceptible perturbation to the input can mislead DNNs trained on clean images into making erroneous predictions. To tackle this, adversarial training is currently the most effective defense method, by augmenting the training set with adversarial samples generated on the fly. Interestingly, we discover for the first time that there exist subnetworks with inborn robustness, matching or surpassing the robust accuracy of the adversarially trained networks with comparable model sizes, within randomly initialized networks without any model training, indicating that adversarial training on model weights is not indispensable towards adversarial robustness. We name such subnetworks Robust Scratch Tickets (RSTs), which are also by nature efficient. Distinct from the popular lottery ticket hypothesis, neither the original dense networks nor the identified RSTs need to be trained. To validate and understand this fascinating finding, we further conduct extensive experiments to study the existence and properties of RSTs under different models, datasets, sparsity patterns, and attacks, drawing insights regarding the relationship between DNNs' robustness and their initialization/overparameterization. Furthermore, we identify the poor adversarial transferability between RSTs of different sparsity ratios drawn from the same randomly initialized dense network, and propose a Random RST Switch (R2S) technique, which randomly switches between different RSTs, as a novel defense method built on top of RSTs. We believe our findings about RSTs have opened up a new perspective to study model robustness and extend the lottery ticket hypothesis.



## **46. TRS: Transferability Reduced Ensemble via Encouraging Gradient Diversity and Model Smoothness**

cs.LG

Proceedings of the 35th Conference on Neural Information Processing  Systems (NeurIPS 2021)

**SubmitDate**: 2021-11-06    [paper-pdf](http://arxiv.org/pdf/2104.00671v2)

**Authors**: Zhuolin Yang, Linyi Li, Xiaojun Xu, Shiliang Zuo, Qian Chen, Benjamin Rubinstein, Pan Zhou, Ce Zhang, Bo Li

**Abstracts**: Adversarial Transferability is an intriguing property - adversarial perturbation crafted against one model is also effective against another model, while these models are from different model families or training processes. To better protect ML systems against adversarial attacks, several questions are raised: what are the sufficient conditions for adversarial transferability and how to bound it? Is there a way to reduce the adversarial transferability in order to improve the robustness of an ensemble ML model? To answer these questions, in this work we first theoretically analyze and outline sufficient conditions for adversarial transferability between models; then propose a practical algorithm to reduce the transferability between base models within an ensemble to improve its robustness. Our theoretical analysis shows that only promoting the orthogonality between gradients of base models is not enough to ensure low transferability; in the meantime, the model smoothness is an important factor to control the transferability. We also provide the lower and upper bounds of adversarial transferability under certain conditions. Inspired by our theoretical analysis, we propose an effective Transferability Reduced Smooth(TRS) ensemble training strategy to train a robust ensemble with low transferability by enforcing both gradient orthogonality and model smoothness between base models. We conduct extensive experiments on TRS and compare with 6 state-of-the-art ensemble baselines against 8 whitebox attacks on different datasets, demonstrating that the proposed TRS outperforms all baselines significantly.



## **47. Reconstructing Training Data from Diverse ML Models by Ensemble Inversion**

cs.LG

9 pages, 8 figures, WACV 2022

**SubmitDate**: 2021-11-05    [paper-pdf](http://arxiv.org/pdf/2111.03702v1)

**Authors**: Qian Wang, Daniel Kurz

**Abstracts**: Model Inversion (MI), in which an adversary abuses access to a trained Machine Learning (ML) model attempting to infer sensitive information about its original training data, has attracted increasing research attention. During MI, the trained model under attack (MUA) is usually frozen and used to guide the training of a generator, such as a Generative Adversarial Network (GAN), to reconstruct the distribution of the original training data of that model. This might cause leakage of original training samples, and if successful, the privacy of dataset subjects will be at risk if the training data contains Personally Identifiable Information (PII). Therefore, an in-depth investigation of the potentials of MI techniques is crucial for the development of corresponding defense techniques. High-quality reconstruction of training data based on a single model is challenging. However, existing MI literature does not explore targeting multiple models jointly, which may provide additional information and diverse perspectives to the adversary.   We propose the ensemble inversion technique that estimates the distribution of original training data by training a generator constrained by an ensemble (or set) of trained models with shared subjects or entities. This technique leads to noticeable improvements of the quality of the generated samples with distinguishable features of the dataset entities compared to MI of a single ML model. We achieve high quality results without any dataset and show how utilizing an auxiliary dataset that's similar to the presumed training data improves the results. The impact of model diversity in the ensemble is thoroughly investigated and additional constraints are utilized to encourage sharp predictions and high activations for the reconstructed samples, leading to more accurate reconstruction of training images.



## **48. Visualizing the Emergence of Intermediate Visual Patterns in DNNs**

cs.CV

**SubmitDate**: 2021-11-05    [paper-pdf](http://arxiv.org/pdf/2111.03505v1)

**Authors**: Mingjie Li, Shaobo Wang, Quanshi Zhang

**Abstracts**: This paper proposes a method to visualize the discrimination power of intermediate-layer visual patterns encoded by a DNN. Specifically, we visualize (1) how the DNN gradually learns regional visual patterns in each intermediate layer during the training process, and (2) the effects of the DNN using non-discriminative patterns in low layers to construct disciminative patterns in middle/high layers through the forward propagation. Based on our visualization method, we can quantify knowledge points (i.e., the number of discriminative visual patterns) learned by the DNN to evaluate the representation capacity of the DNN. Furthermore, this method also provides new insights into signal-processing behaviors of existing deep-learning techniques, such as adversarial attacks and knowledge distillation.



## **49. Adversarial Attacks on Knowledge Graph Embeddings via Instance Attribution Methods**

cs.LG

2021 Conference on Empirical Methods in Natural Language Processing  (EMNLP 2021)

**SubmitDate**: 2021-11-04    [paper-pdf](http://arxiv.org/pdf/2111.03120v1)

**Authors**: Peru Bhardwaj, John Kelleher, Luca Costabello, Declan O'Sullivan

**Abstracts**: Despite the widespread use of Knowledge Graph Embeddings (KGE), little is known about the security vulnerabilities that might disrupt their intended behaviour. We study data poisoning attacks against KGE models for link prediction. These attacks craft adversarial additions or deletions at training time to cause model failure at test time. To select adversarial deletions, we propose to use the model-agnostic instance attribution methods from Interpretable Machine Learning, which identify the training instances that are most influential to a neural model's predictions on test instances. We use these influential triples as adversarial deletions. We further propose a heuristic method to replace one of the two entities in each influential triple to generate adversarial additions. Our experiments show that the proposed strategies outperform the state-of-art data poisoning attacks on KGE models and improve the MRR degradation due to the attacks by up to 62% over the baselines.



## **50. Scanflow: A multi-graph framework for Machine Learning workflow management, supervision, and debugging**

cs.LG

**SubmitDate**: 2021-11-04    [paper-pdf](http://arxiv.org/pdf/2111.03003v1)

**Authors**: Gusseppe Bravo-Rocca, Peini Liu, Jordi Guitart, Ajay Dholakia, David Ellison, Jeffrey Falkanger, Miroslav Hodak

**Abstracts**: Machine Learning (ML) is more than just training models, the whole workflow must be considered. Once deployed, a ML model needs to be watched and constantly supervised and debugged to guarantee its validity and robustness in unexpected situations. Debugging in ML aims to identify (and address) the model weaknesses in not trivial contexts. Several techniques have been proposed to identify different types of model weaknesses, such as bias in classification, model decay, adversarial attacks, etc., yet there is not a generic framework that allows them to work in a collaborative, modular, portable, iterative way and, more importantly, flexible enough to allow both human- and machine-driven techniques. In this paper, we propose a novel containerized directed graph framework to support and accelerate end-to-end ML workflow management, supervision, and debugging. The framework allows defining and deploying ML workflows in containers, tracking their metadata, checking their behavior in production, and improving the models by using both learned and human-provided knowledge. We demonstrate these capabilities by integrating in the framework two hybrid systems to detect data drift distribution which identify the samples that are far from the latent space of the original distribution, ask for human intervention, and whether retrain the model or wrap it with a filter to remove the noise of corrupted data at inference time. We test these systems on MNIST-C, CIFAR-10-C, and FashionMNIST-C datasets, obtaining promising accuracy results with the help of human involvement.



