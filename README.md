# Latest Adversarial Attack Papers
**update at 2022-03-02 06:31:52**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Artificial Intelligence for Cyber Security (AICS)**

cs.CR

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.14010v1)

**Authors**: James Holt, Edward Raff, Ahmad Ridley, Dennis Ross, Arunesh Sinha, Diane Staheli, William Streilen, Milind Tambe, Yevgeniy Vorobeychik, Allan Wollaber

**Abstracts**: The workshop will focus on the application of AI to problems in cyber security. Cyber systems generate large volumes of data, utilizing this effectively is beyond human capabilities. Additionally, adversaries continue to develop new attacks. Hence, AI methods are required to understand and protect the cyber domain. These challenges are widely studied in enterprise networks, but there are many gaps in research and practice as well as novel problems in other domains.   In general, AI techniques are still not widely adopted in the real world. Reasons include: (1) a lack of certification of AI for security, (2) a lack of formal study of the implications of practical constraints (e.g., power, memory, storage) for AI systems in the cyber domain, (3) known vulnerabilities such as evasion, poisoning attacks, (4) lack of meaningful explanations for security analysts, and (5) lack of analyst trust in AI solutions. There is a need for the research community to develop novel solutions for these practical issues.



## **2. Load-Altering Attacks Against Power Grids under COVID-19 Low-Inertia Conditions**

cs.CR

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2201.10505v2)

**Authors**: Subhash Lakshminarayana, Juan Ospina, Charalambos Konstantinou

**Abstracts**: The COVID-19 pandemic has impacted our society by forcing shutdowns and shifting the way people interacted worldwide. In relation to the impacts on the electric grid, it created a significant decrease in energy demands across the globe. Recent studies have shown that the low demand conditions caused by COVID-19 lockdowns combined with large renewable generation have resulted in extremely low-inertia grid conditions. In this work, we examine how an attacker could exploit these {scenarios} to cause unsafe grid operating conditions by executing load-altering attacks (LAAs) targeted at compromising hundreds of thousands of IoT-connected high-wattage loads in low-inertia power systems. Our study focuses on analyzing the impact of the COVID-19 mitigation measures on U.S. regional transmission operators (RTOs), formulating a plausible and realistic least-effort LAA targeted at transmission systems with low-inertia conditions, and evaluating the probability of these large-scale LAAs. Theoretical and simulation results are presented based on the WSCC 9-bus {and IEEE 118-bus} test systems. Results demonstrate how adversaries could provoke major frequency disturbances by targeting vulnerable load buses in low-inertia systems and offer insights into how the temporal fluctuations of renewable energy sources, considering generation scheduling, impact the grid's vulnerability to LAAs.



## **3. MaMaDroid2.0 -- The Holes of Control Flow Graphs**

cs.CR

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13922v1)

**Authors**: Harel Berger, Chen Hajaj, Enrico Mariconti, Amit Dvir

**Abstracts**: Android malware is a continuously expanding threat to billions of mobile users around the globe. Detection systems are updated constantly to address these threats. However, a backlash takes the form of evasion attacks, in which an adversary changes malicious samples such that those samples will be misclassified as benign. This paper fully inspects a well-known Android malware detection system, MaMaDroid, which analyzes the control flow graph of the application. Changes to the portion of benign samples in the train set and models are considered to see their effect on the classifier. The changes in the ratio between benign and malicious samples have a clear effect on each one of the models, resulting in a decrease of more than 40% in their detection rate. Moreover, adopted ML models are implemented as well, including 5-NN, Decision Tree, and Adaboost. Exploration of the six models reveals a typical behavior in different cases, of tree-based models and distance-based models. Moreover, three novel attacks that manipulate the CFG and their detection rates are described for each one of the targeted models. The attacks decrease the detection rate of most of the models to 0%, with regards to different ratios of benign to malicious apps. As a result, a new version of MaMaDroid is engineered. This model fuses the CFG of the app and static analysis of features of the app. This improved model is proved to be robust against evasion attacks targeting both CFG-based models and static analysis models, achieving a detection rate of more than 90% against each one of the attacks.



## **4. Formally verified asymptotic consensus in robust networks**

cs.PL

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13833v1)

**Authors**: Mohit Tekriwal, Avi Tachna-Fram, Jean-Baptiste Jeannin, Manos Kapritsos, Dimitra Panagou

**Abstracts**: Distributed architectures are used to improve performance and reliability of various systems. An important capability of a distributed architecture is the ability to reach consensus among all its nodes. To achieve this, several consensus algorithms have been proposed for various scenarii, and many of these algorithms come with proofs of correctness that are not mechanically checked. Unfortunately, those proofs are known to be intricate and prone to errors.   In this paper, we formalize and mechanically check a consensus algorithm widely used in the distributed controls community: the Weighted-Mean Subsequence Reduced (W-MSR) algorithm proposed by Le Blanc et al. This algorithm provides a way to achieve asymptotic consensus in a distributed controls scenario in the presence of adversarial agents (attackers) that may not update their states based on the nominal consensus protocol, and may share inaccurate information with their neighbors. Using the Coq proof assistant, we formalize the necessary and sufficient conditions required to achieve resilient asymptotic consensus under the assumed attacker model. We leverage the existing Coq formalizations of graph theory, finite sets and sequences of the mathcomp library for our development. To our knowledge, this is the first mechanical proof of an asymptotic consensus algorithm. During the formalization, we clarify several imprecisions in the paper proof, including an imprecision on quantifiers in the main theorem.



## **5. Robust Textual Embedding against Word-level Adversarial Attacks**

cs.CL

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13817v1)

**Authors**: Yichen Yang, Xiaosen Wang, Kun He

**Abstracts**: We attribute the vulnerability of natural language processing models to the fact that similar inputs are converted to dissimilar representations in the embedding space, leading to inconsistent outputs, and propose a novel robust training method, termed Fast Triplet Metric Learning (FTML). Specifically, we argue that the original sample should have similar representation with its adversarial counterparts and distinguish its representation from other samples for better robustness. To this end, we adopt the triplet metric learning into the standard training to pull the words closer to their positive samples (i.e., synonyms) and push away their negative samples (i.e., non-synonyms) in the embedding space. Extensive experiments demonstrate that FTML can significantly promote the model robustness against various advanced adversarial attacks while keeping competitive classification accuracy on original samples. Besides, our method is efficient as it only needs to adjust the embedding and introduces very little overhead on the standard training. Our work shows the great potential of improving the textual robustness through robust word embedding.



## **6. Towards Robust Stacked Capsule Autoencoder with Hybrid Adversarial Training**

cs.CV

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13755v1)

**Authors**: Jiazhu Dai, Siwei Xiong

**Abstracts**: Capsule networks (CapsNets) are new neural networks that classify images based on the spatial relationships of features. By analyzing the pose of features and their relative positions, it is more capable to recognize images after affine transformation. The stacked capsule autoencoder (SCAE) is a state-of-the-art CapsNet, and achieved unsupervised classification of CapsNets for the first time. However, the security vulnerabilities and the robustness of the SCAE has rarely been explored. In this paper, we propose an evasion attack against SCAE, where the attacker can generate adversarial perturbations based on reducing the contribution of the object capsules in SCAE related to the original category of the image. The adversarial perturbations are then applied to the original images, and the perturbed images will be misclassified. Furthermore, we propose a defense method called Hybrid Adversarial Training (HAT) against such evasion attacks. HAT makes use of adversarial training and adversarial distillation to achieve better robustness and stability. We evaluate the defense method and the experimental results show that the refined SCAE model can achieve 82.14% classification accuracy under evasion attack. The source code is available at https://github.com/FrostbiteXSW/SCAE_Defense.



## **7. On the Robustness of CountSketch to Adaptive Inputs**

cs.DS

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13736v1)

**Authors**: Edith Cohen, Xin Lyu, Jelani Nelson, Tamás Sarlós, Moshe Shechner, Uri Stemmer

**Abstracts**: CountSketch is a popular dimensionality reduction technique that maps vectors to a lower dimension using randomized linear measurements. The sketch supports recovering $\ell_2$-heavy hitters of a vector (entries with $v[i]^2 \geq \frac{1}{k}\|\boldsymbol{v}\|^2_2$). We study the robustness of the sketch in adaptive settings where input vectors may depend on the output from prior inputs. Adaptive settings arise in processes with feedback or with adversarial attacks. We show that the classic estimator is not robust, and can be attacked with a number of queries of the order of the sketch size. We propose a robust estimator (for a slightly modified sketch) that allows for quadratic number of queries in the sketch size, which is an improvement factor of $\sqrt{k}$ (for $k$ heavy hitters) over prior work.



## **8. An Empirical Study on the Intrinsic Privacy of SGD**

cs.LG

21 pages, 11 figures, 8 tables

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/1912.02919v4)

**Authors**: Stephanie L. Hyland, Shruti Tople

**Abstracts**: Introducing noise in the training of machine learning systems is a powerful way to protect individual privacy via differential privacy guarantees, but comes at a cost to utility. This work looks at whether the inherent randomness of stochastic gradient descent (SGD) could contribute to privacy, effectively reducing the amount of \emph{additional} noise required to achieve a given privacy guarantee. We conduct a large-scale empirical study to examine this question. Training a grid of over 120,000 models across four datasets (tabular and images) on convex and non-convex objectives, we demonstrate that the random seed has a larger impact on model weights than any individual training example. We test the distribution over weights induced by the seed, finding that the simple convex case can be modelled with a multivariate Gaussian posterior, while neural networks exhibit multi-modal and non-Gaussian weight distributions. By casting convex SGD as a Gaussian mechanism, we then estimate an `intrinsic' data-dependent $\epsilon_i(\mathcal{D})$, finding values as low as 6.3, dropping to 1.9 using empirical estimates. We use a membership inference attack to estimate $\epsilon$ for non-convex SGD and demonstrate that hiding the random seed from the adversary results in a statistically significant reduction in attack performance, corresponding to a reduction in the effective $\epsilon$. These results provide empirical evidence that SGD exhibits appreciable variability relative to its dataset sensitivity, and this `intrinsic noise' has the potential to be leveraged to improve the utility of privacy-preserving machine learning.



## **9. Enhance transferability of adversarial examples with model architecture**

cs.LG

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13625v1)

**Authors**: Mingyuan Fan, Wenzhong Guo, Shengxing Yu, Zuobin Ying, Ximeng Liu

**Abstracts**: Transferability of adversarial examples is of critical importance to launch black-box adversarial attacks, where attackers are only allowed to access the output of the target model. However, under such a challenging but practical setting, the crafted adversarial examples are always prone to overfitting to the proxy model employed, presenting poor transferability. In this paper, we suggest alleviating the overfitting issue from a novel perspective, i.e., designing a fitted model architecture. Specifically, delving the bottom of the cause of poor transferability, we arguably decompose and reconstruct the existing model architecture into an effective model architecture, namely multi-track model architecture (MMA). The adversarial examples crafted on the MMA can maximumly relieve the effect of model-specified features to it and toward the vulnerable directions adopted by diverse architectures. Extensive experimental evaluation demonstrates that the transferability of adversarial examples based on the MMA significantly surpass other state-of-the-art model architectures by up to 40% with comparable overhead.



## **10. GRAPHITE: Generating Automatic Physical Examples for Machine-Learning Attacks on Computer Vision Systems**

cs.CR

IEEE European Symposium on Security and Privacy 2022 (EuroS&P 2022)

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2002.07088v6)

**Authors**: Ryan Feng, Neal Mangaokar, Jiefeng Chen, Earlence Fernandes, Somesh Jha, Atul Prakash

**Abstracts**: This paper investigates an adversary's ease of attack in generating adversarial examples for real-world scenarios. We address three key requirements for practical attacks for the real-world: 1) automatically constraining the size and shape of the attack so it can be applied with stickers, 2) transform-robustness, i.e., robustness of a attack to environmental physical variations such as viewpoint and lighting changes, and 3) supporting attacks in not only white-box, but also black-box hard-label scenarios, so that the adversary can attack proprietary models. In this work, we propose GRAPHITE, an efficient and general framework for generating attacks that satisfy the above three key requirements. GRAPHITE takes advantage of transform-robustness, a metric based on expectation over transforms (EoT), to automatically generate small masks and optimize with gradient-free optimization. GRAPHITE is also flexible as it can easily trade-off transform-robustness, perturbation size, and query count in black-box settings. On a GTSRB model in a hard-label black-box setting, we are able to find attacks on all possible 1,806 victim-target class pairs with averages of 77.8% transform-robustness, perturbation size of 16.63% of the victim images, and 126K queries per pair. For digital-only attacks where achieving transform-robustness is not a requirement, GRAPHITE is able to find successful small-patch attacks with an average of only 566 queries for 92.2% of victim-target pairs. GRAPHITE is also able to find successful attacks using perturbations that modify small areas of the input image against PatchGuard, a recently proposed defense against patch-based attacks.



## **11. Markov Chain Monte Carlo-Based Machine Unlearning: Unlearning What Needs to be Forgotten**

cs.LG

Proceedings of the 2022 ACM Asia Conference on Computer and  Communications Security (ASIA CCS '22), May 30-June 3, 2022, Nagasaki, Japan

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13585v1)

**Authors**: Quoc Phong Nguyen, Ryutaro Oikawa, Dinil Mon Divakaran, Mun Choon Chan, Bryan Kian Hsiang Low

**Abstracts**: As the use of machine learning (ML) models is becoming increasingly popular in many real-world applications, there are practical challenges that need to be addressed for model maintenance. One such challenge is to 'undo' the effect of a specific subset of dataset used for training a model. This specific subset may contain malicious or adversarial data injected by an attacker, which affects the model performance. Another reason may be the need for a service provider to remove data pertaining to a specific user to respect the user's privacy. In both cases, the problem is to 'unlearn' a specific subset of the training data from a trained model without incurring the costly procedure of retraining the whole model from scratch. Towards this goal, this paper presents a Markov chain Monte Carlo-based machine unlearning (MCU) algorithm. MCU helps to effectively and efficiently unlearn a trained model from subsets of training dataset. Furthermore, we show that with MCU, we are able to explain the effect of a subset of a training dataset on the model prediction. Thus, MCU is useful for examining subsets of data to identify the adversarial data to be removed. Similarly, MCU can be used to erase the lineage of a user's personal data from trained ML models, thus upholding a user's "right to be forgotten". We empirically evaluate the performance of our proposed MCU algorithm on real-world phishing and diabetes datasets. Results show that MCU can achieve a desirable performance by efficiently removing the effect of a subset of training dataset and outperform an existing algorithm that utilizes the remaining dataset.



## **12. A Unified Wasserstein Distributional Robustness Framework for Adversarial Training**

cs.LG

**SubmitDate**: 2022-02-27    [paper-pdf](http://arxiv.org/pdf/2202.13437v1)

**Authors**: Tuan Anh Bui, Trung Le, Quan Tran, He Zhao, Dinh Phung

**Abstracts**: It is well-known that deep neural networks (DNNs) are susceptible to adversarial attacks, exposing a severe fragility of deep learning systems. As the result, adversarial training (AT) method, by incorporating adversarial examples during training, represents a natural and effective approach to strengthen the robustness of a DNN-based classifier. However, most AT-based methods, notably PGD-AT and TRADES, typically seek a pointwise adversary that generates the worst-case adversarial example by independently perturbing each data sample, as a way to "probe" the vulnerability of the classifier. Arguably, there are unexplored benefits in considering such adversarial effects from an entire distribution. To this end, this paper presents a unified framework that connects Wasserstein distributional robustness with current state-of-the-art AT methods. We introduce a new Wasserstein cost function and a new series of risk functions, with which we show that standard AT methods are special cases of their counterparts in our framework. This connection leads to an intuitive relaxation and generalization of existing AT methods and facilitates the development of a new family of distributional robustness AT-based algorithms. Extensive experiments show that our distributional robustness AT algorithms robustify further their standard AT counterparts in various settings.



## **13. Finding Optimal Tangent Points for Reducing Distortions of Hard-label Attacks**

cs.CV

Accepted at NeurIPS 2021. The missing square term in Eqn.(13), as  well as many other mistakes of the previous version, have been fixed in the  current version

**SubmitDate**: 2022-02-27    [paper-pdf](http://arxiv.org/pdf/2111.07492v5)

**Authors**: Chen Ma, Xiangyu Guo, Li Chen, Jun-Hai Yong, Yisen Wang

**Abstracts**: One major problem in black-box adversarial attacks is the high query complexity in the hard-label attack setting, where only the top-1 predicted label is available. In this paper, we propose a novel geometric-based approach called Tangent Attack (TA), which identifies an optimal tangent point of a virtual hemisphere located on the decision boundary to reduce the distortion of the attack. Assuming the decision boundary is locally flat, we theoretically prove that the minimum $\ell_2$ distortion can be obtained by reaching the decision boundary along the tangent line passing through such tangent point in each iteration. To improve the robustness of our method, we further propose a generalized method which replaces the hemisphere with a semi-ellipsoid to adapt to curved decision boundaries. Our approach is free of pre-training. Extensive experiments conducted on the ImageNet and CIFAR-10 datasets demonstrate that our approach can consume only a small number of queries to achieve the low-magnitude distortion. The implementation source code is released online at https://github.com/machanic/TangentAttack.



## **14. CC-Cert: A Probabilistic Approach to Certify General Robustness of Neural Networks**

cs.LG

In Proceedings of AAAI-22, the Thirty-Sixth AAAI Conference on  Artificial Intelligence

**SubmitDate**: 2022-02-27    [paper-pdf](http://arxiv.org/pdf/2109.10696v2)

**Authors**: Mikhail Pautov, Nurislam Tursynbek, Marina Munkhoeva, Nikita Muravev, Aleksandr Petiushko, Ivan Oseledets

**Abstracts**: In safety-critical machine learning applications, it is crucial to defend models against adversarial attacks -- small modifications of the input that change the predictions. Besides rigorously studied $\ell_p$-bounded additive perturbations, recently proposed semantic perturbations (e.g. rotation, translation) raise a serious concern on deploying ML systems in real-world. Therefore, it is important to provide provable guarantees for deep learning models against semantically meaningful input transformations. In this paper, we propose a new universal probabilistic certification approach based on Chernoff-Cramer bounds that can be used in general attack settings. We estimate the probability of a model to fail if the attack is sampled from a certain distribution. Our theoretical findings are supported by experimental results on different datasets.



## **15. Socialbots on Fire: Modeling Adversarial Behaviors of Socialbots via Multi-Agent Hierarchical Reinforcement Learning**

cs.SI

Accepted to The ACM Web Conference 2022

**SubmitDate**: 2022-02-26    [paper-pdf](http://arxiv.org/pdf/2110.10655v2)

**Authors**: Thai Le, Long Tran-Thanh, Dongwon Lee

**Abstracts**: Socialbots are software-driven user accounts on social platforms, acting autonomously (mimicking human behavior), with the aims to influence the opinions of other users or spread targeted misinformation for particular goals. As socialbots undermine the ecosystem of social platforms, they are often considered harmful. As such, there have been several computational efforts to auto-detect the socialbots. However, to our best knowledge, the adversarial nature of these socialbots has not yet been studied. This begs a question "can adversaries, controlling socialbots, exploit AI techniques to their advantage?" To this question, we successfully demonstrate that indeed it is possible for adversaries to exploit computational learning mechanism such as reinforcement learning (RL) to maximize the influence of socialbots while avoiding being detected. We first formulate the adversarial socialbot learning as a cooperative game between two functional hierarchical RL agents. While one agent curates a sequence of activities that can avoid the detection, the other agent aims to maximize network influence by selectively connecting with right users. Our proposed policy networks train with a vast amount of synthetic graphs and generalize better than baselines on unseen real-life graphs both in terms of maximizing network influence (up to +18%) and sustainable stealthiness (up to +40% undetectability) under a strong bot detector (with 90% detection accuracy). During inference, the complexity of our approach scales linearly, independent of a network's structure and the virality of news. This makes our approach a practical adversarial attack when deployed in a real-life setting.



## **16. Natural Attack for Pre-trained Models of Code**

cs.SE

To appear in the Technical Track of ICSE 2022

**SubmitDate**: 2022-02-26    [paper-pdf](http://arxiv.org/pdf/2201.08698v2)

**Authors**: Zhou Yang, Jieke Shi, Junda He, David Lo

**Abstracts**: Pre-trained models of code have achieved success in many important software engineering tasks. However, these powerful models are vulnerable to adversarial attacks that slightly perturb model inputs to make a victim model produce wrong outputs. Current works mainly attack models of code with examples that preserve operational program semantics but ignore a fundamental requirement for adversarial example generation: perturbations should be natural to human judges, which we refer to as naturalness requirement.   In this paper, we propose ALERT (nAturaLnEss AwaRe ATtack), a black-box attack that adversarially transforms inputs to make victim models produce wrong outputs. Different from prior works, this paper considers the natural semantic of generated examples at the same time as preserving the operational semantic of original inputs. Our user study demonstrates that human developers consistently consider that adversarial examples generated by ALERT are more natural than those generated by the state-of-the-art work by Zhang et al. that ignores the naturalness requirement. On attacking CodeBERT, our approach can achieve attack success rates of 53.62%, 27.79%, and 35.78% across three downstream tasks: vulnerability prediction, clone detection and code authorship attribution. On GraphCodeBERT, our approach can achieve average success rates of 76.95%, 7.96% and 61.47% on the three tasks. The above outperforms the baseline by 14.07% and 18.56% on the two pre-trained models on average. Finally, we investigated the value of the generated adversarial examples to harden victim models through an adversarial fine-tuning procedure and demonstrated the accuracy of CodeBERT and GraphCodeBERT against ALERT-generated adversarial examples increased by 87.59% and 92.32%, respectively.



## **17. Projective Ranking-based GNN Evasion Attacks**

cs.LG

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12993v1)

**Authors**: He Zhang, Xingliang Yuan, Chuan Zhou, Shirui Pan

**Abstracts**: Graph neural networks (GNNs) offer promising learning methods for graph-related tasks. However, GNNs are at risk of adversarial attacks. Two primary limitations of the current evasion attack methods are highlighted: (1) The current GradArgmax ignores the "long-term" benefit of the perturbation. It is faced with zero-gradient and invalid benefit estimates in certain situations. (2) In the reinforcement learning-based attack methods, the learned attack strategies might not be transferable when the attack budget changes. To this end, we first formulate the perturbation space and propose an evaluation framework and the projective ranking method. We aim to learn a powerful attack strategy then adapt it as little as possible to generate adversarial samples under dynamic budget settings. In our method, based on mutual information, we rank and assess the attack benefits of each perturbation for an effective attack strategy. By projecting the strategy, our method dramatically minimizes the cost of learning a new attack strategy when the attack budget changes. In the comparative assessment with GradArgmax and RL-S2V, the results show our method owns high attack performance and effective transferability. The visualization of our method also reveals various attack patterns in the generation of adversarial samples.



## **18. Attacks and Faults Injection in Self-Driving Agents on the Carla Simulator -- Experience Report**

cs.AI

submitted version; appeared at: International Conference on Computer  Safety, Reliability, and Security. Springer, Cham, 2021

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12991v1)

**Authors**: Niccolò Piazzesi, Massimo Hong, Andrea Ceccarelli

**Abstracts**: Machine Learning applications are acknowledged at the foundation of autonomous driving, because they are the enabling technology for most driving tasks. However, the inclusion of trained agents in automotive systems exposes the vehicle to novel attacks and faults, that can result in safety threats to the driv-ing tasks. In this paper we report our experimental campaign on the injection of adversarial attacks and software faults in a self-driving agent running in a driving simulator. We show that adversarial attacks and faults injected in the trained agent can lead to erroneous decisions and severely jeopardize safety. The paper shows a feasible and easily-reproducible approach based on open source simula-tor and tools, and the results clearly motivate the need of both protective measures and extensive testing campaigns.



## **19. Does Label Differential Privacy Prevent Label Inference Attacks?**

cs.LG

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12968v1)

**Authors**: Ruihan Wu, Jin Peng Zhou, Kilian Q. Weinberger, Chuan Guo

**Abstracts**: Label differential privacy (LDP) is a popular framework for training private ML models on datasets with public features and sensitive private labels. Despite its rigorous privacy guarantee, it has been observed that in practice LDP does not preclude label inference attacks (LIAs): Models trained with LDP can be evaluated on the public training features to recover, with high accuracy, the very private labels that it was designed to protect. In this work, we argue that this phenomenon is not paradoxical and that LDP merely limits the advantage of an LIA adversary compared to predicting training labels using the Bayes classifier. At LDP $\epsilon=0$ this advantage is zero, hence the optimal attack is to predict according to the Bayes classifier and is independent of the training labels. Finally, we empirically demonstrate that our result closely captures the behavior of simulated attacks on both synthetic and real world datasets.



## **20. Robust and Accurate Authorship Attribution via Program Normalization**

cs.LG

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2007.00772v3)

**Authors**: Yizhen Wang, Mohannad Alhanahnah, Ke Wang, Mihai Christodorescu, Somesh Jha

**Abstracts**: Source code attribution approaches have achieved remarkable accuracy thanks to the rapid advances in deep learning. However, recent studies shed light on their vulnerability to adversarial attacks. In particular, they can be easily deceived by adversaries who attempt to either create a forgery of another author or to mask the original author. To address these emerging issues, we formulate this security challenge into a general threat model, the $\textit{relational adversary}$, that allows an arbitrary number of the semantics-preserving transformations to be applied to an input in any problem space. Our theoretical investigation shows the conditions for robustness and the trade-off between robustness and accuracy in depth. Motivated by these insights, we present a novel learning framework, $\textit{normalize-and-predict}$ ($\textit{N&P}$), that in theory guarantees the robustness of any authorship-attribution approach. We conduct an extensive evaluation of $\textit{N&P}$ in defending two of the latest authorship-attribution approaches against state-of-the-art attack methods. Our evaluation demonstrates that $\textit{N&P}$ improves the accuracy on adversarial inputs by as much as 70% over the vanilla models. More importantly, $\textit{N&P}$ also increases robust accuracy to 45% higher than adversarial training while running over 40 times faster.



## **21. ARIA: Adversarially Robust Image Attribution for Content Provenance**

cs.CV

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12860v1)

**Authors**: Maksym Andriushchenko, Xiaoyang Rebecca Li, Geoffrey Oxholm, Thomas Gittings, Tu Bui, Nicolas Flammarion, John Collomosse

**Abstracts**: Image attribution -- matching an image back to a trusted source -- is an emerging tool in the fight against online misinformation. Deep visual fingerprinting models have recently been explored for this purpose. However, they are not robust to tiny input perturbations known as adversarial examples. First we illustrate how to generate valid adversarial images that can easily cause incorrect image attribution. Then we describe an approach to prevent imperceptible adversarial attacks on deep visual fingerprinting models, via robust contrastive learning. The proposed training procedure leverages training on $\ell_\infty$-bounded adversarial examples, it is conceptually simple and incurs only a small computational overhead. The resulting models are substantially more robust, are accurate even on unperturbed images, and perform well even over a database with millions of images. In particular, we achieve 91.6% standard and 85.1% adversarial recall under $\ell_\infty$-bounded perturbations on manipulated images compared to 80.1% and 0.0% from prior work. We also show that robustness generalizes to other types of imperceptible perturbations unseen during training. Finally, we show how to train an adversarially robust image comparator model for detecting editorial changes in matched images.



## **22. Short Paper: Device- and Locality-Specific Fingerprinting of Shared NISQ Quantum Computers**

cs.CR

5 pages, 8 figures, HASP 2021 author version

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12731v1)

**Authors**: Allen Mi, Shuwen Deng, Jakub Szefer

**Abstracts**: Fingerprinting of quantum computer devices is a new threat that poses a challenge to shared, cloud-based quantum computers. Fingerprinting can allow adversaries to map quantum computer infrastructures, uniquely identify cloud-based devices which otherwise have no public identifiers, and it can assist other adversarial attacks. This work shows idle tomography-based fingerprinting method based on crosstalk-induced errors in NISQ quantum computers. The device- and locality-specific fingerprinting results show prediction accuracy values of $99.1\%$ and $95.3\%$, respectively.



## **23. Detection as Regression: Certified Object Detection by Median Smoothing**

cs.CV

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2007.03730v4)

**Authors**: Ping-yeh Chiang, Michael J. Curry, Ahmed Abdelkader, Aounon Kumar, John Dickerson, Tom Goldstein

**Abstracts**: Despite the vulnerability of object detectors to adversarial attacks, very few defenses are known to date. While adversarial training can improve the empirical robustness of image classifiers, a direct extension to object detection is very expensive. This work is motivated by recent progress on certified classification by randomized smoothing. We start by presenting a reduction from object detection to a regression problem. Then, to enable certified regression, where standard mean smoothing fails, we propose median smoothing, which is of independent interest. We obtain the first model-agnostic, training-free, and certified defense for object detection against $\ell_2$-bounded attacks. The code for all experiments in the paper is available at http://github.com/Ping-C/CertifiedObjectDetection .



## **24. On the Effectiveness of Dataset Watermarking in Adversarial Settings**

cs.CR

7 pages, 2 figures. Will appear in the proceedings of CODASPY-IWSPA  2022

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12506v1)

**Authors**: Buse Gul Atli Tekgul, N. Asokan

**Abstracts**: In a data-driven world, datasets constitute a significant economic value. Dataset owners who spend time and money to collect and curate the data are incentivized to ensure that their datasets are not used in ways that they did not authorize. When such misuse occurs, dataset owners need technical mechanisms for demonstrating their ownership of the dataset in question. Dataset watermarking provides one approach for ownership demonstration which can, in turn, deter unauthorized use. In this paper, we investigate a recently proposed data provenance method, radioactive data, to assess if it can be used to demonstrate ownership of (image) datasets used to train machine learning (ML) models. The original paper reported that radioactive data is effective in white-box settings. We show that while this is true for large datasets with many classes, it is not as effective for datasets where the number of classes is low $(\leq 30)$ or the number of samples per class is low $(\leq 500)$. We also show that, counter-intuitively, the black-box verification technique is effective for all datasets used in this paper, even when white-box verification is not. Given this observation, we show that the confidence in white-box verification can be improved by using watermarked samples directly during the verification process. We also highlight the need to assess the robustness of radioactive data if it were to be used for ownership demonstration since it is an adversarial setting unlike provenance identification.   Compared to dataset watermarking, ML model watermarking has been explored more extensively in recent literature. However, most of the model watermarking techniques can be defeated via model extraction. We show that radioactive data can effectively survive model extraction attacks, which raises the possibility that it can be used for ML model ownership verification robust against model extraction.



## **25. Understanding Adversarial Robustness from Feature Maps of Convolutional Layers**

cs.CV

10pages

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12435v1)

**Authors**: Cong Xu, Min Yang

**Abstracts**: The adversarial robustness of a neural network mainly relies on two factors, one is the feature representation capacity of the network, and the other is its resistance ability to perturbations. In this paper, we study the anti-perturbation ability of the network from the feature maps of convolutional layers. Our theoretical analysis discovers that larger convolutional features before average pooling can contribute to better resistance to perturbations, but the conclusion is not true for max pooling. Based on the theoretical findings, we present two feasible ways to improve the robustness of existing neural networks. The proposed approaches are very simple and only require upsampling the inputs or modifying the stride configuration of convolution operators. We test our approaches on several benchmark neural network architectures, including AlexNet, VGG16, RestNet18 and PreActResNet18, and achieve non-trivial improvements on both natural accuracy and robustness under various attacks. Our study brings new insights into the design of robust neural networks. The code is available at \url{https://github.com/MTandHJ/rcm}.



## **26. AEVA: Black-box Backdoor Detection Using Adversarial Extreme Value Analysis**

cs.LG

**SubmitDate**: 2022-02-24    [paper-pdf](http://arxiv.org/pdf/2110.14880v4)

**Authors**: Junfeng Guo, Ang Li, Cong Liu

**Abstracts**: Deep neural networks (DNNs) are proved to be vulnerable against backdoor attacks. A backdoor is often embedded in the target DNNs through injecting a backdoor trigger into training examples, which can cause the target DNNs misclassify an input attached with the backdoor trigger. Existing backdoor detection methods often require the access to the original poisoned training data, the parameters of the target DNNs, or the predictive confidence for each given input, which are impractical in many real-world applications, e.g., on-device deployed DNNs. We address the black-box hard-label backdoor detection problem where the DNN is fully black-box and only its final output label is accessible. We approach this problem from the optimization perspective and show that the objective of backdoor detection is bounded by an adversarial objective. Further theoretical and empirical studies reveal that this adversarial objective leads to a solution with highly skewed distribution; a singularity is often observed in the adversarial map of a backdoor-infected example, which we call the adversarial singularity phenomenon. Based on this observation, we propose the adversarial extreme value analysis(AEVA) to detect backdoors in black-box neural networks. AEVA is based on an extreme value analysis of the adversarial map, computed from the monte-carlo gradient estimation. Evidenced by extensive experiments across multiple popular tasks and backdoor attacks, our approach is shown effective in detecting backdoor attacks under the black-box hard-label scenarios.



## **27. Bounding Membership Inference**

cs.LG

**SubmitDate**: 2022-02-24    [paper-pdf](http://arxiv.org/pdf/2202.12232v1)

**Authors**: Anvith Thudi, Ilia Shumailov, Franziska Boenisch, Nicolas Papernot

**Abstracts**: Differential Privacy (DP) is the de facto standard for reasoning about the privacy guarantees of a training algorithm. Despite the empirical observation that DP reduces the vulnerability of models to existing membership inference (MI) attacks, a theoretical underpinning as to why this is the case is largely missing in the literature. In practice, this means that models need to be trained with DP guarantees that greatly decrease their accuracy. In this paper, we provide a tighter bound on the accuracy of any MI adversary when a training algorithm provides $\epsilon$-DP. Our bound informs the design of a novel privacy amplification scheme, where an effective training set is sub-sampled from a larger set prior to the beginning of training, to greatly reduce the bound on MI accuracy. As a result, our scheme enables $\epsilon$-DP users to employ looser DP guarantees when training their model to limit the success of any MI adversary; this ensures that the model's accuracy is less impacted by the privacy guarantee. Finally, we discuss implications of our MI bound on the field of machine unlearning.



## **28. Dynamic Defense Against Byzantine Poisoning Attacks in Federated Learning**

cs.LG

10 pages

**SubmitDate**: 2022-02-24    [paper-pdf](http://arxiv.org/pdf/2007.15030v2)

**Authors**: Nuria Rodríguez-Barroso, Eugenio Martínez-Cámara, M. Victoria Luzón, Francisco Herrera

**Abstracts**: Federated learning, as a distributed learning that conducts the training on the local devices without accessing to the training data, is vulnerable to Byzatine poisoning adversarial attacks. We argue that the federated learning model has to avoid those kind of adversarial attacks through filtering out the adversarial clients by means of the federated aggregation operator. We propose a dynamic federated aggregation operator that dynamically discards those adversarial clients and allows to prevent the corruption of the global learning model. We assess it as a defense against adversarial attacks deploying a deep learning classification model in a federated learning setting on the Fed-EMNIST Digits, Fashion MNIST and CIFAR-10 image datasets. The results show that the dynamic selection of the clients to aggregate enhances the performance of the global learning model and discards the adversarial and poor (with low quality models) clients.



## **29. Towards Effective and Robust Neural Trojan Defenses via Input Filtering**

cs.CR

**SubmitDate**: 2022-02-24    [paper-pdf](http://arxiv.org/pdf/2202.12154v1)

**Authors**: Kien Do, Haripriya Harikumar, Hung Le, Dung Nguyen, Truyen Tran, Santu Rana, Dang Nguyen, Willy Susilo, Svetha Venkatesh

**Abstracts**: Trojan attacks on deep neural networks are both dangerous and surreptitious. Over the past few years, Trojan attacks have advanced from using only a simple trigger and targeting only one class to using many sophisticated triggers and targeting multiple classes. However, Trojan defenses have not caught up with this development. Most defense methods still make out-of-date assumptions about Trojan triggers and target classes, thus, can be easily circumvented by modern Trojan attacks. In this paper, we advocate general defenses that are effective and robust against various Trojan attacks and propose two novel "filtering" defenses with these characteristics called Variational Input Filtering (VIF) and Adversarial Input Filtering (AIF). VIF and AIF leverage variational inference and adversarial training respectively to purify all potential Trojan triggers in the input at run time without making any assumption about their numbers and forms. We further extend "filtering" to "filtering-then-contrasting" - a new defense mechanism that helps avoid the drop in classification accuracy on clean data caused by filtering. Extensive experimental results show that our proposed defenses significantly outperform 4 well-known defenses in mitigating 5 different Trojan attacks including the two state-of-the-art which defeat many strong defenses.



## **30. HODA: Hardness-Oriented Detection of Model Extraction Attacks**

cs.LG

15 pages, 12 figures, 7 tables, 2 Alg

**SubmitDate**: 2022-02-24    [paper-pdf](http://arxiv.org/pdf/2106.11424v2)

**Authors**: Amir Mahdi Sadeghzadeh, Amir Mohammad Sobhanian, Faezeh Dehghan, Rasool Jalili

**Abstracts**: Model Extraction attacks exploit the target model's prediction API to create a surrogate model in order to steal or reconnoiter the functionality of the target model in the black-box setting. Several recent studies have shown that a data-limited adversary who has no or limited access to the samples from the target model's training data distribution can use synthesis or semantically similar samples to conduct model extraction attacks. In this paper, we define the hardness degree of a sample using the concept of learning difficulty. The hardness degree of a sample depends on the epoch number that the predicted label of that sample converges. We investigate the hardness degree of samples and demonstrate that the hardness degree histogram of a data-limited adversary's sample sequences is distinguishable from the hardness degree histogram of benign users' samples sequences. We propose Hardness-Oriented Detection Approach (HODA) to detect the sample sequences of model extraction attacks. The results demonstrate that HODA can detect the sample sequences of model extraction attacks with a high success rate by only monitoring 100 samples of them, and it outperforms all previous model extraction detection methods.



## **31. Feature Importance-aware Transferable Adversarial Attacks**

cs.CV

Accepted to ICCV 2021

**SubmitDate**: 2022-02-24    [paper-pdf](http://arxiv.org/pdf/2107.14185v3)

**Authors**: Zhibo Wang, Hengchang Guo, Zhifei Zhang, Wenxin Liu, Zhan Qin, Kui Ren

**Abstracts**: Transferability of adversarial examples is of central importance for attacking an unknown model, which facilitates adversarial attacks in more practical scenarios, e.g., black-box attacks. Existing transferable attacks tend to craft adversarial examples by indiscriminately distorting features to degrade prediction accuracy in a source model without aware of intrinsic features of objects in the images. We argue that such brute-force degradation would introduce model-specific local optimum into adversarial examples, thus limiting the transferability. By contrast, we propose the Feature Importance-aware Attack (FIA), which disrupts important object-aware features that dominate model decisions consistently. More specifically, we obtain feature importance by introducing the aggregate gradient, which averages the gradients with respect to feature maps of the source model, computed on a batch of random transforms of the original clean image. The gradients will be highly correlated to objects of interest, and such correlation presents invariance across different models. Besides, the random transforms will preserve intrinsic features of objects and suppress model-specific information. Finally, the feature importance guides to search for adversarial examples towards disrupting critical features, achieving stronger transferability. Extensive experimental evaluation demonstrates the effectiveness and superior performance of the proposed FIA, i.e., improving the success rate by 9.5% against normally trained models and 12.8% against defense models as compared to the state-of-the-art transferable attacks. Code is available at: https://github.com/hcguoO0/FIA



## **32. Robust Probabilistic Time Series Forecasting**

cs.LG

AISTATS 2022 camera ready version

**SubmitDate**: 2022-02-24    [paper-pdf](http://arxiv.org/pdf/2202.11910v1)

**Authors**: TaeHo Yoon, Youngsuk Park, Ernest K. Ryu, Yuyang Wang

**Abstracts**: Probabilistic time series forecasting has played critical role in decision-making processes due to its capability to quantify uncertainties. Deep forecasting models, however, could be prone to input perturbations, and the notion of such perturbations, together with that of robustness, has not even been completely established in the regime of probabilistic forecasting. In this work, we propose a framework for robust probabilistic time series forecasting. First, we generalize the concept of adversarial input perturbations, based on which we formulate the concept of robustness in terms of bounded Wasserstein deviation. Then we extend the randomized smoothing technique to attain robust probabilistic forecasters with theoretical robustness certificates against certain classes of adversarial perturbations. Lastly, extensive experiments demonstrate that our methods are empirically effective in enhancing the forecast quality under additive adversarial attacks and forecast consistency under supplement of noisy observations.



## **33. Improving Robustness of Convolutional Neural Networks Using Element-Wise Activation Scaling**

cs.CV

**SubmitDate**: 2022-02-24    [paper-pdf](http://arxiv.org/pdf/2202.11898v1)

**Authors**: Zhi-Yuan Zhang, Di Liu

**Abstracts**: Recent works reveal that re-calibrating the intermediate activation of adversarial examples can improve the adversarial robustness of a CNN model. The state of the arts [Baiet al., 2021] and [Yanet al., 2021] explores this feature at the channel level, i.e. the activation of a channel is uniformly scaled by a factor. In this paper, we investigate the intermediate activation manipulation at a more fine-grained level. Instead of uniformly scaling the activation, we individually adjust each element within an activation and thus propose Element-Wise Activation Scaling, dubbed EWAS, to improve CNNs' adversarial robustness. Experimental results on ResNet-18 and WideResNet with CIFAR10 and SVHN show that EWAS significantly improves the robustness accuracy. Especially for ResNet18 on CIFAR10, EWAS increases the adversarial accuracy by 37.65% to 82.35% against C&W attack. EWAS is simple yet very effective in terms of improving robustness. The codes are anonymously available at https://anonymous.4open.science/r/EWAS-DD64.



## **34. FastZIP: Faster and More Secure Zero-Interaction Pairing**

cs.CR

ACM MobiSys '21; Fixed ambiguity in flow diagram (Figure 2). Code and  data are available at: https://github.com/seemoo-lab/fastzip

**SubmitDate**: 2022-02-23    [paper-pdf](http://arxiv.org/pdf/2106.04907v3)

**Authors**: Mikhail Fomichev, Julia Hesse, Lars Almon, Timm Lippert, Jun Han, Matthias Hollick

**Abstracts**: With the advent of the Internet of Things (IoT), establishing a secure channel between smart devices becomes crucial. Recent research proposes zero-interaction pairing (ZIP), which enables pairing without user assistance by utilizing devices' physical context (e.g., ambient audio) to obtain a shared secret key. The state-of-the-art ZIP schemes suffer from three limitations: (1) prolonged pairing time (i.e., minutes or hours), (2) vulnerability to brute-force offline attacks on a shared key, and (3) susceptibility to attacks caused by predictable context (e.g., replay attack) because they rely on limited entropy of physical context to protect a shared key. We address these limitations, proposing FastZIP, a novel ZIP scheme that significantly reduces pairing time while preventing offline and predictable context attacks. In particular, we adapt a recently introduced Fuzzy Password-Authenticated Key Exchange (fPAKE) protocol and utilize sensor fusion, maximizing their advantages. We instantiate FastZIP for intra-car device pairing to demonstrate its feasibility and show how the design of FastZIP can be adapted to other ZIP use cases. We implement FastZIP and evaluate it by driving four cars for a total of 800 km. We achieve up to three times shorter pairing time compared to the state-of-the-art ZIP schemes while assuring robust security with adversarial error rates below 0.5%.



## **35. Distributed and Mobile Message Level Relaying/Replaying of GNSS Signals**

cs.CR

**SubmitDate**: 2022-02-23    [paper-pdf](http://arxiv.org/pdf/2202.11341v1)

**Authors**: M. Lenhart, M. Spanghero, P. Papadimitratos

**Abstracts**: With the introduction of Navigation Message Authentication (NMA), future Global Navigation Satellite Systems (GNSSs) prevent spoofing by simulation, i.e., the generation of forged satellite signals based on public information. However, authentication does not prevent record-and-replay attacks, commonly termed as meaconing. These attacks are less powerful in terms of adversarial control over the victim receiver location and time, but by acting at the signal level, they are not thwarted by NMA. This makes replaying/relaying attacks a significant threat for GNSS. While there are numerous investigations on meaconing, the majority does not rely on actual implementation and experimental evaluation in real-world settings. In this work, we contribute to the improvement of the experimental understanding of meaconing attacks. We design and implement a system capable of real-time, distributed, and mobile meaconing, built with off-the-shelf hardware. We extend from basic distributed attacks, with signals from different locations relayed over the Internet and replayed within range of the victim receiver(s): this has high bandwidth requirements and thus depends on the quality of service of the available network to work. To overcome this limitation, we propose to replay on message level, including the authentication part of the payload. The resultant reduced bandwidth enables the attacker to operate in mobile scenarios, as well as to replay signals from multiple GNSS constellations and/or bands simultaneously. Additionally, the attacker can delay individually selected satellite signals to potentially influence the victim position and time solution in a more fine-grained manner. Our versatile test-bench, enabling different types of replaying/relaying attacks, facilitates testing realistic scenarios towards new and improved replaying/relaying-focused countermeasures in GNSS receivers.



## **36. LPF-Defense: 3D Adversarial Defense based on Frequency Analysis**

cs.CV

15 pages, 7 figures

**SubmitDate**: 2022-02-23    [paper-pdf](http://arxiv.org/pdf/2202.11287v1)

**Authors**: Hanieh Naderi, Arian Etemadi, Kimia Noorbakhsh, Shohreh Kasaei

**Abstracts**: Although 3D point cloud classification has recently been widely deployed in different application scenarios, it is still very vulnerable to adversarial attacks. This increases the importance of robust training of 3D models in the face of adversarial attacks. Based on our analysis on the performance of existing adversarial attacks, more adversarial perturbations are found in the mid and high-frequency components of input data. Therefore, by suppressing the high-frequency content in the training phase, the models robustness against adversarial examples is improved. Experiments showed that the proposed defense method decreases the success rate of six attacks on PointNet, PointNet++ ,, and DGCNN models. In particular, improvements are achieved with an average increase of classification accuracy by 3.8 % on drop100 attack and 4.26 % on drop200 attack compared to the state-of-the-art methods. The method also improves models accuracy on the original dataset compared to other available methods.



## **37. Sound Adversarial Audio-Visual Navigation**

cs.SD

This work aims to do an adversarial sound intervention for robust  audio-visual navigation

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10910v1)

**Authors**: Yinfeng Yu, Wenbing Huang, Fuchun Sun, Changan Chen, Yikai Wang, Xiaohong Liu

**Abstracts**: Audio-visual navigation task requires an agent to find a sound source in a realistic, unmapped 3D environment by utilizing egocentric audio-visual observations. Existing audio-visual navigation works assume a clean environment that solely contains the target sound, which, however, would not be suitable in most real-world applications due to the unexpected sound noise or intentional interference. In this work, we design an acoustically complex environment in which, besides the target sound, there exists a sound attacker playing a zero-sum game with the agent. More specifically, the attacker can move and change the volume and category of the sound to make the agent suffer from finding the sounding object while the agent tries to dodge the attack and navigate to the goal under the intervention. Under certain constraints to the attacker, we can improve the robustness of the agent towards unexpected sound attacks in audio-visual navigation. For better convergence, we develop a joint training mechanism by employing the property of a centralized critic with decentralized actors. Experiments on two real-world 3D scan datasets, Replica, and Matterport3D, verify the effectiveness and the robustness of the agent trained under our designed environment when transferred to the clean environment or the one containing sound attackers with random policy. Project: \url{https://yyf17.github.io/SAAVN}.



## **38. DEMO: Relay/Replay Attacks on GNSS signals**

cs.CR

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10897v1)

**Authors**: M. Lenhart, M. Spanghero, P. Papadimitratos

**Abstracts**: Global Navigation Satellite Systems (GNSS) are ubiquitously relied upon for positioning and timing. Detection and prevention of attacks against GNSS have been researched over the last decades, but many of these attacks and countermeasures were evaluated based on simulation. This work contributes to the experimental investigation of GNSS vulnerabilities, implementing a relay/replay attack with off-the-shelf hardware. Operating at the signal level, this attack type is not hindered by cryptographically protected transmissions, such as Galileo's Open Signals Navigation Message Authentication (OS-NMA). The attack we investigate involves two colluding adversaries, relaying signals over large distances, to effectively spoof a GNSS receiver. We demonstrate the attack using off-the-shelf hardware, we investigate the requirements for such successful colluding attacks, and how they can be enhanced, e.g., allowing for finer adversarial control over the victim receiver.



## **39. Protecting GNSS-based Services using Time Offset Validation**

cs.CR

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10891v1)

**Authors**: K. Zhang, M. Spanghero, P. Papadimitratos

**Abstracts**: Global navigation satellite systems (GNSS) provide pervasive accurate positioning and timing services for a large gamut of applications, from Time based One-Time Passwords (TOPT), to power grid and cellular systems. However, there can be security concerns for the applications due to the vulnerability of GNSS. It is important to observe that GNSS receivers are components of platforms, in principle having rich connectivity to different network infrastructures. Of particular interest is the access to a variety of timing sources, as those can be used to validate GNSS-provided location and time. Therefore, we consider off-the-shelf platforms and how to detect if the GNSS receiver is attacked or not, by cross-checking the GNSS time and time from other available sources. First, we survey different technologies to analyze their availability, accuracy, and trustworthiness for time synchronization. Then, we propose a validation approach for absolute and relative time. Moreover, we design a framework and experimental setup for the evaluation of the results. Attacks can be detected based on WiFi supplied time when the adversary shifts the GNSS provided time, more than 23.942us; with Network Time Protocol (NTP) supplied time when the adversary-induced shift is more than 2.046ms. Consequently, the proposal significantly limits the capability of an adversary to manipulate the victim GNSS receiver.



## **40. Adversarial Defense by Latent Style Transformations**

cs.CV

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2006.09701v2)

**Authors**: Shuo Wang, Surya Nepal, Alsharif Abuadbba, Carsten Rudolph, Marthie Grobler

**Abstracts**: Machine learning models have demonstrated vulnerability to adversarial attacks, more specifically misclassification of adversarial examples.   In this paper, we investigate an attack-agnostic defense against adversarial attacks on high-resolution images by detecting suspicious inputs.   The intuition behind our approach is that the essential characteristics of a normal image are generally consistent with non-essential style transformations, e.g., slightly changing the facial expression of human portraits.   In contrast, adversarial examples are generally sensitive to such transformations.   In our approach to detect adversarial instances, we propose an in\underline{V}ertible \underline{A}utoencoder based on the \underline{S}tyleGAN2 generator via \underline{A}dversarial training (VASA) to inverse images to disentangled latent codes that reveal hierarchical styles.   We then build a set of edited copies with non-essential style transformations by performing latent shifting and reconstruction, based on the correspondences between latent codes and style transformations.   The classification-based consistency of these edited copies is used to distinguish adversarial instances.



## **41. Surrogate Representation Learning with Isometric Mapping for Gray-box Graph Adversarial Attacks**

cs.AI

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2110.10482v3)

**Authors**: Zihan Liu, Yun Luo, Zelin Zang, Stan Z. Li

**Abstracts**: Gray-box graph attacks aim at disrupting the performance of the victim model by using inconspicuous attacks with limited knowledge of the victim model. The parameters of the victim model and the labels of the test nodes are invisible to the attacker. To obtain the gradient on the node attributes or graph structure, the attacker constructs an imaginary surrogate model trained under supervision. However, there is a lack of discussion on the training of surrogate models and the robustness of provided gradient information. The general node classification model loses the topology of the nodes on the graph, which is, in fact, an exploitable prior for the attacker. This paper investigates the effect of representation learning of surrogate models on the transferability of gray-box graph adversarial attacks. To reserve the topology in the surrogate embedding, we propose Surrogate Representation Learning with Isometric Mapping (SRLIM). By using Isometric mapping method, our proposed SRLIM can constrain the topological structure of nodes from the input layer to the embedding space, that is, to maintain the similarity of nodes in the propagation process. Experiments prove the effectiveness of our approach through the improvement in the performance of the adversarial attacks generated by the gradient-based attacker in untargeted poisoning gray-box setups.



## **42. Universal adversarial perturbation for remote sensing images**

cs.CV

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10693v1)

**Authors**: Zhaoxia Yin, Qingyu Wang, Jin Tang, Bin Luo

**Abstracts**: Recently, with the application of deep learning in the remote sensing image (RSI) field, the classification accuracy of the RSI has been greatly improved compared with traditional technology. However, even state-of-the-art object recognition convolutional neural networks are fooled by the universal adversarial perturbation (UAP). To verify that UAP makes the RSI classification model error classification, this paper proposes a novel method combining an encoder-decoder network with an attention mechanism. Firstly, the former can learn the distribution of perturbations better, then the latter is used to find the main regions concerned by the RSI classification model. Finally, the generated regions are used to fine-tune the perturbations making the model misclassified with fewer perturbations. The experimental results show that the UAP can make the RSI misclassify, and the attack success rate (ASR) of our proposed method on the RSI data set is as high as 97.35%.



## **43. Seeing is Living? Rethinking the Security of Facial Liveness Verification in the Deepfake Era**

cs.CR

Accepted as a full paper at USENIX Security '22

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10673v1)

**Authors**: Changjiang Li, Li Wang, Shouling Ji, Xuhong Zhang, Zhaohan Xi, Shanqing Guo, Ting Wang

**Abstracts**: Facial Liveness Verification (FLV) is widely used for identity authentication in many security-sensitive domains and offered as Platform-as-a-Service (PaaS) by leading cloud vendors. Yet, with the rapid advances in synthetic media techniques (e.g., deepfake), the security of FLV is facing unprecedented challenges, about which little is known thus far.   To bridge this gap, in this paper, we conduct the first systematic study on the security of FLV in real-world settings. Specifically, we present LiveBugger, a new deepfake-powered attack framework that enables customizable, automated security evaluation of FLV. Leveraging LiveBugger, we perform a comprehensive empirical assessment of representative FLV platforms, leading to a set of interesting findings. For instance, most FLV APIs do not use anti-deepfake detection; even for those with such defenses, their effectiveness is concerning (e.g., it may detect high-quality synthesized videos but fail to detect low-quality ones). We then conduct an in-depth analysis of the factors impacting the attack performance of LiveBugger: a) the bias (e.g., gender or race) in FLV can be exploited to select victims; b) adversarial training makes deepfake more effective to bypass FLV; c) the input quality has a varying influence on different deepfake techniques to bypass FLV. Based on these findings, we propose a customized, two-stage approach that can boost the attack success rate by up to 70%. Further, we run proof-of-concept attacks on several representative applications of FLV (i.e., the clients of FLV APIs) to illustrate the practical implications: due to the vulnerability of the APIs, many downstream applications are vulnerable to deepfake. Finally, we discuss potential countermeasures to improve the security of FLV. Our findings have been confirmed by the corresponding vendors.



## **44. Fingerprinting Deep Neural Networks Globally via Universal Adversarial Perturbations**

cs.CR

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.08602v2)

**Authors**: Zirui Peng, Shaofeng Li, Guoxing Chen, Cheng Zhang, Haojin Zhu, Minhui Xue

**Abstracts**: In this paper, we propose a novel and practical mechanism which enables the service provider to verify whether a suspect model is stolen from the victim model via model extraction attacks. Our key insight is that the profile of a DNN model's decision boundary can be uniquely characterized by its \textit{Universal Adversarial Perturbations (UAPs)}. UAPs belong to a low-dimensional subspace and piracy models' subspaces are more consistent with victim model's subspace compared with non-piracy model. Based on this, we propose a UAP fingerprinting method for DNN models and train an encoder via \textit{contrastive learning} that takes fingerprint as inputs, outputs a similarity score. Extensive studies show that our framework can detect model IP breaches with confidence $> 99.99 \%$ within only $20$ fingerprints of the suspect model. It has good generalizability across different model architectures and is robust against post-modifications on stolen models.



## **45. Robust Stochastic Linear Contextual Bandits Under Adversarial Attacks**

stat.ML

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2106.02978v2)

**Authors**: Qin Ding, Cho-Jui Hsieh, James Sharpnack

**Abstracts**: Stochastic linear contextual bandit algorithms have substantial applications in practice, such as recommender systems, online advertising, clinical trials, etc. Recent works show that optimal bandit algorithms are vulnerable to adversarial attacks and can fail completely in the presence of attacks. Existing robust bandit algorithms only work for the non-contextual setting under the attack of rewards and cannot improve the robustness in the general and popular contextual bandit environment. In addition, none of the existing methods can defend against attacked context. In this work, we provide the first robust bandit algorithm for stochastic linear contextual bandit setting under a fully adaptive and omniscient attack with sub-linear regret. Our algorithm not only works under the attack of rewards, but also under attacked context. Moreover, it does not need any information about the attack budget or the particular form of the attack. We provide theoretical guarantees for our proposed algorithm and show by experiments that our proposed algorithm improves the robustness against various kinds of popular attacks.



## **46. Behaviour-Diverse Automatic Penetration Testing: A Curiosity-Driven Multi-Objective Deep Reinforcement Learning Approach**

cs.LG

6 pages,4 Figures

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10630v1)

**Authors**: Yizhou Yang, Xin Liu

**Abstracts**: Penetration Testing plays a critical role in evaluating the security of a target network by emulating real active adversaries. Deep Reinforcement Learning (RL) is seen as a promising solution to automating the process of penetration tests by reducing human effort and improving reliability. Existing RL solutions focus on finding a specific attack path to impact the target hosts. However, in reality, a diverse range of attack variations are needed to provide comprehensive assessments of the target network's security level. Hence, the attack agents must consider multiple objectives when penetrating the network. Nevertheless, this challenge is not adequately addressed in the existing literature. To this end, we formulate the automatic penetration testing in the Multi-Objective Reinforcement Learning (MORL) framework and propose a Chebyshev decomposition critic to find diverse adversary strategies that balance different objectives in the penetration test. Additionally, the number of available actions increases with the agent consistently probing the target network, making the training process intractable in many practical situations. Thus, we introduce a coverage-based masking mechanism that reduces attention on previously selected actions to help the agent adapt to future exploration. Experimental evaluation on a range of scenarios demonstrates the superiority of our proposed approach when compared to adapted algorithms in terms of multi-objective learning and performance efficiency.



## **47. On the Effectiveness of Adversarial Training against Backdoor Attacks**

cs.LG

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10627v1)

**Authors**: Yinghua Gao, Dongxian Wu, Jingfeng Zhang, Guanhao Gan, Shu-Tao Xia, Gang Niu, Masashi Sugiyama

**Abstracts**: DNNs' demand for massive data forces practitioners to collect data from the Internet without careful check due to the unacceptable cost, which brings potential risks of backdoor attacks. A backdoored model always predicts a target class in the presence of a predefined trigger pattern, which can be easily realized via poisoning a small amount of data. In general, adversarial training is believed to defend against backdoor attacks since it helps models to keep their prediction unchanged even if we perturb the input image (as long as within a feasible range). Unfortunately, few previous studies succeed in doing so. To explore whether adversarial training could defend against backdoor attacks or not, we conduct extensive experiments across different threat models and perturbation budgets, and find the threat model in adversarial training matters. For instance, adversarial training with spatial adversarial examples provides notable robustness against commonly-used patch-based backdoor attacks. We further propose a hybrid strategy which provides satisfactory robustness across different backdoor attacks.



## **48. Adversarial Attacks on Speech Recognition Systems for Mission-Critical Applications: A Survey**

cs.SD

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10594v1)

**Authors**: Ngoc Dung Huynh, Mohamed Reda Bouadjenek, Imran Razzak, Kevin Lee, Chetan Arora, Ali Hassani, Arkady Zaslavsky

**Abstracts**: A Machine-Critical Application is a system that is fundamentally necessary to the success of specific and sensitive operations such as search and recovery, rescue, military, and emergency management actions. Recent advances in Machine Learning, Natural Language Processing, voice recognition, and speech processing technologies have naturally allowed the development and deployment of speech-based conversational interfaces to interact with various machine-critical applications. While these conversational interfaces have allowed users to give voice commands to carry out strategic and critical activities, their robustness to adversarial attacks remains uncertain and unclear. Indeed, Adversarial Artificial Intelligence (AI) which refers to a set of techniques that attempt to fool machine learning models with deceptive data, is a growing threat in the AI and machine learning research community, in particular for machine-critical applications. The most common reason of adversarial attacks is to cause a malfunction in a machine learning model. An adversarial attack might entail presenting a model with inaccurate or fabricated samples as it's training data, or introducing maliciously designed data to deceive an already trained model. While focusing on speech recognition for machine-critical applications, in this paper, we first review existing speech recognition techniques, then, we investigate the effectiveness of adversarial attacks and defenses against these systems, before outlining research challenges, defense recommendations, and future work. This paper is expected to serve researchers and practitioners as a reference to help them in understanding the challenges, position themselves and, ultimately, help them to improve existing models of speech recognition for mission-critical applications. Keywords: Mission-Critical Applications, Adversarial AI, Speech Recognition Systems.



## **49. Privacy Leakage of Adversarial Training Models in Federated Learning Systems**

cs.LG

6 pages, 6 figures. Submitted to CVPR'22 workshop "The Art of  Robustness"

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2202.10546v1)

**Authors**: Jingyang Zhang, Yiran Chen, Hai Li

**Abstracts**: Adversarial Training (AT) is crucial for obtaining deep neural networks that are robust to adversarial attacks, yet recent works found that it could also make models more vulnerable to privacy attacks. In this work, we further reveal this unsettling property of AT by designing a novel privacy attack that is practically applicable to the privacy-sensitive Federated Learning (FL) systems. Using our method, the attacker can exploit AT models in the FL system to accurately reconstruct users' private training images even when the training batch size is large. Code is available at https://github.com/zjysteven/PrivayAttack_AT_FL.



## **50. Analysing Security and Privacy Threats in the Lockdown Periods of COVID-19 Pandemic: Twitter Dataset Case Study**

cs.CR

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2202.10543v1)

**Authors**: Bibhas Sharma, Ishan Karunanayake, Rahat Masood, Muhammad Ikram

**Abstracts**: The COVID-19 pandemic will be remembered as a uniquely disruptive period that altered the lives of billions of citizens globally, resulting in new-normal for the way people live and work. With the coronavirus pandemic, everyone had to adapt to the "work or study from home" operating model that has transformed our online lives and exponentially increased the use of cyberspace. Concurrently, there has been a huge spike in social media platforms such as Facebook and Twitter during the COVID-19 lockdown periods. These lockdown periods have resulted in a set of new cybercrimes, thereby allowing attackers to victimise users of social media platforms in times of fear, uncertainty, and doubt. The threats range from running phishing campaigns and malicious domains to extracting private information about victims for malicious purposes. This research paper performs a large-scale study to investigate the impact of lockdown periods during the COVID-19 pandemic on the security and privacy of social media users. We analyse 10.6 Million COVID-related tweets from 533 days of data crawling and investigate users' security and privacy behaviour in three different periods (i.e., before, during, and after lockdown). Our study shows that users unintentionally share more personal identifiable information when writing about the pandemic situation in their tweets. The privacy risk reaches 100% if a user posts three or more sensitive tweets about the pandemic. We investigate the number of suspicious domains shared in social media during different pandemic phases. Our analysis reveals an increase in suspicious domains during the lockdown compared to other lockdown phases. We observe that IT, Search Engines, and Businesses are the top three categories that contain suspicious domains. Our analysis reveals that adversaries' strategies to instigate malicious activities change with the country's pandemic situation.



