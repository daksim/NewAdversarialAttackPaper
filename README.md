# Latest Adversarial Attack Papers
**update at 2022-10-28 06:31:38**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. On the Versatile Uses of Partial Distance Correlation in Deep Learning**

cs.CV

This paper has been selected as best paper award for ECCV 2022!

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2207.09684v2) [paper-pdf](http://arxiv.org/pdf/2207.09684v2)

**Authors**: Xingjian Zhen, Zihang Meng, Rudrasis Chakraborty, Vikas Singh

**Abstract**: Comparing the functional behavior of neural network models, whether it is a single network over time or two (or more networks) during or post-training, is an essential step in understanding what they are learning (and what they are not), and for identifying strategies for regularization or efficiency improvements. Despite recent progress, e.g., comparing vision transformers to CNNs, systematic comparison of function, especially across different networks, remains difficult and is often carried out layer by layer. Approaches such as canonical correlation analysis (CCA) are applicable in principle, but have been sparingly used so far. In this paper, we revisit a (less widely known) from statistics, called distance correlation (and its partial variant), designed to evaluate correlation between feature spaces of different dimensions. We describe the steps necessary to carry out its deployment for large scale models -- this opens the door to a surprising array of applications ranging from conditioning one deep model w.r.t. another, learning disentangled representations as well as optimizing diverse models that would directly be more robust to adversarial attacks. Our experiments suggest a versatile regularizer (or constraint) with many advantages, which avoids some of the common difficulties one faces in such analyses. Code is at https://github.com/zhenxingjian/Partial_Distance_Correlation.



## **2. Identifying Threats, Cybercrime and Digital Forensic Opportunities in Smart City Infrastructure via Threat Modeling**

cs.CR

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.14692v1) [paper-pdf](http://arxiv.org/pdf/2210.14692v1)

**Authors**: Yee Ching Tok, Sudipta Chattopadhyay

**Abstract**: Technological advances have enabled multiple countries to consider implementing Smart City Infrastructure to provide in-depth insights into different data points and enhance the lives of citizens. Unfortunately, these new technological implementations also entice adversaries and cybercriminals to execute cyber-attacks and commit criminal acts on these modern infrastructures. Given the borderless nature of cyber attacks, varying levels of understanding of smart city infrastructure and ongoing investigation workloads, law enforcement agencies and investigators would be hard-pressed to respond to these kinds of cybercrime. Without an investigative capability by investigators, these smart infrastructures could become new targets favored by cybercriminals.   To address the challenges faced by investigators, we propose a common definition of smart city infrastructure. Based on the definition, we utilize the STRIDE threat modeling methodology and the Microsoft Threat Modeling Tool to identify threats present in the infrastructure and create a threat model which can be further customized or extended by interested parties. Next, we map offences, possible evidence sources and types of threats identified to help investigators understand what crimes could have been committed and what evidence would be required in their investigation work. Finally, noting that Smart City Infrastructure investigations would be a global multi-faceted challenge, we discuss technical and legal opportunities in digital forensics on Smart City Infrastructure.



## **3. Certified Robustness in Federated Learning**

cs.LG

Accepted at Workshop on Federated Learning: Recent Advances and New  Challenges, NeurIPS 2022

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2206.02535v2) [paper-pdf](http://arxiv.org/pdf/2206.02535v2)

**Authors**: Motasem Alfarra, Juan C. Pérez, Egor Shulgin, Peter Richtárik, Bernard Ghanem

**Abstract**: Federated learning has recently gained significant attention and popularity due to its effectiveness in training machine learning models on distributed data privately. However, as in the single-node supervised learning setup, models trained in federated learning suffer from vulnerability to imperceptible input transformations known as adversarial attacks, questioning their deployment in security-related applications. In this work, we study the interplay between federated training, personalization, and certified robustness. In particular, we deploy randomized smoothing, a widely-used and scalable certification method, to certify deep networks trained on a federated setup against input perturbations and transformations. We find that the simple federated averaging technique is effective in building not only more accurate, but also more certifiably-robust models, compared to training solely on local data. We further analyze personalization, a popular technique in federated training that increases the model's bias towards local data, on robustness. We show several advantages of personalization over both~(that is, only training on local data and federated training) in building more robust models with faster training. Finally, we explore the robustness of mixtures of global and local~(i.e. personalized) models, and find that the robustness of local models degrades as they diverge from the global model



## **4. Short Paper: Static and Microarchitectural ML-Based Approaches For Detecting Spectre Vulnerabilities and Attacks**

cs.CR

5 pages, 2 figures. Accepted to the Hardware and Architectural  Support for Security and Privacy (HASP'22), in conjunction with the 55th  IEEE/ACM International Symposium on Microarchitecture (MICRO'22)

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.14452v1) [paper-pdf](http://arxiv.org/pdf/2210.14452v1)

**Authors**: Chidera Biringa, Gaspard Baye, Gökhan Kul

**Abstract**: Spectre intrusions exploit speculative execution design vulnerabilities in modern processors. The attacks violate the principles of isolation in programs to gain unauthorized private user information. Current state-of-the-art detection techniques utilize micro-architectural features or vulnerable speculative code to detect these threats. However, these techniques are insufficient as Spectre attacks have proven to be more stealthy with recently discovered variants that bypass current mitigation mechanisms. Side-channels generate distinct patterns in processor cache, and sensitive information leakage is dependent on source code vulnerable to Spectre attacks, where an adversary uses these vulnerabilities, such as branch prediction, which causes a data breach. Previous studies predominantly approach the detection of Spectre attacks using the microarchitectural analysis, a reactive approach. Hence, in this paper, we present the first comprehensive evaluation of static and microarchitectural analysis-assisted machine learning approaches to detect Spectre vulnerable code snippets (preventive) and Spectre attacks (reactive). We evaluate the performance trade-offs in employing classifiers for detecting Spectre vulnerabilities and attacks.



## **5. Improving Adversarial Robustness via Joint Classification and Multiple Explicit Detection Classes**

cs.CV

21 pages, 6 figures

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.14410v1) [paper-pdf](http://arxiv.org/pdf/2210.14410v1)

**Authors**: Sina Baharlouei, Fatemeh Sheikholeslami, Meisam Razaviyayn, Zico Kolter

**Abstract**: This work concerns the development of deep networks that are certifiably robust to adversarial attacks. Joint robust classification-detection was recently introduced as a certified defense mechanism, where adversarial examples are either correctly classified or assigned to the "abstain" class. In this work, we show that such a provable framework can benefit by extension to networks with multiple explicit abstain classes, where the adversarial examples are adaptively assigned to those. We show that naively adding multiple abstain classes can lead to "model degeneracy", then we propose a regularization approach and a training method to counter this degeneracy by promoting full use of the multiple abstain classes. Our experiments demonstrate that the proposed approach consistently achieves favorable standard vs. robust verified accuracy tradeoffs, outperforming state-of-the-art algorithms for various choices of number of abstain classes.



## **6. Adaptive Test-Time Defense with the Manifold Hypothesis**

cs.LG

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.14404v1) [paper-pdf](http://arxiv.org/pdf/2210.14404v1)

**Authors**: Zhaoyuan Yang, Zhiwei Xu, Jing Zhang, Richard Hartley, Peter Tu

**Abstract**: In this work, we formulate a novel framework of adversarial robustness using the manifold hypothesis. Our framework provides sufficient conditions for defending against adversarial examples. We develop a test-time defense method with our formulation and variational inference. The developed approach combines manifold learning with the Bayesian framework to provide adversarial robustness without the need for adversarial training. We show that our proposed approach can provide adversarial robustness even if attackers are aware of existence of test-time defense. In additions, our approach can also serve as a test-time defense mechanism for variational autoencoders.



## **7. Robustness of Locally Differentially Private Graph Analysis Against Poisoning**

cs.CR

22 pages, 6 figures

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.14376v1) [paper-pdf](http://arxiv.org/pdf/2210.14376v1)

**Authors**: Jacob Imola, Amrita Roy Chowdhury, Kamalika Chaudhuri

**Abstract**: Locally differentially private (LDP) graph analysis allows private analysis on a graph that is distributed across multiple users. However, such computations are vulnerable to data poisoning attacks where an adversary can skew the results by submitting malformed data. In this paper, we formally study the impact of poisoning attacks for graph degree estimation protocols under LDP. We make two key technical contributions. First, we observe LDP makes a protocol more vulnerable to poisoning -- the impact of poisoning is worse when the adversary can directly poison their (noisy) responses, rather than their input data. Second, we observe that graph data is naturally redundant -- every edge is shared between two users. Leveraging this data redundancy, we design robust degree estimation protocols under LDP that can significantly reduce the impact of data poisoning and compute degree estimates with high accuracy. We evaluate our proposed robust degree estimation protocols under poisoning attacks on real-world datasets to demonstrate their efficacy in practice.



## **8. Accelerating Certified Robustness Training via Knowledge Transfer**

cs.LG

NeurIPS '22 Camera Ready version (with appendix)

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.14283v1) [paper-pdf](http://arxiv.org/pdf/2210.14283v1)

**Authors**: Pratik Vaishnavi, Kevin Eykholt, Amir Rahmati

**Abstract**: Training deep neural network classifiers that are certifiably robust against adversarial attacks is critical to ensuring the security and reliability of AI-controlled systems. Although numerous state-of-the-art certified training methods have been developed, they are computationally expensive and scale poorly with respect to both dataset and network complexity. Widespread usage of certified training is further hindered by the fact that periodic retraining is necessary to incorporate new data and network improvements. In this paper, we propose Certified Robustness Transfer (CRT), a general-purpose framework for reducing the computational overhead of any certifiably robust training method through knowledge transfer. Given a robust teacher, our framework uses a novel training loss to transfer the teacher's robustness to the student. We provide theoretical and empirical validation of CRT. Our experiments on CIFAR-10 show that CRT speeds up certified robustness training by $8 \times$ on average across three different architecture generations while achieving comparable robustness to state-of-the-art methods. We also show that CRT can scale to large-scale datasets like ImageNet.



## **9. Similarity between Units of Natural Language: The Transition from Coarse to Fine Estimation**

cs.CL

PhD thesis

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.14275v1) [paper-pdf](http://arxiv.org/pdf/2210.14275v1)

**Authors**: Wenchuan Mu

**Abstract**: Capturing the similarities between human language units is crucial for explaining how humans associate different objects, and therefore its computation has received extensive attention, research, and applications. With the ever-increasing amount of information around us, calculating similarity becomes increasingly complex, especially in many cases, such as legal or medical affairs, measuring similarity requires extra care and precision, as small acts within a language unit can have significant real-world effects. My research goal in this thesis is to develop regression models that account for similarities between language units in a more refined way.   Computation of similarity has come a long way, but approaches to debugging the measures are often based on continually fitting human judgment values. To this end, my goal is to develop an algorithm that precisely catches loopholes in a similarity calculation. Furthermore, most methods have vague definitions of the similarities they compute and are often difficult to interpret. The proposed framework addresses both shortcomings. It constantly improves the model through catching different loopholes. In addition, every refinement of the model provides a reasonable explanation. The regression model introduced in this thesis is called progressively refined similarity computation, which combines attack testing with adversarial training. The similarity regression model of this thesis achieves state-of-the-art performance in handling edge cases.



## **10. Leveraging the Verifier's Dilemma to Double Spend in Bitcoin**

cs.CR

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.14072v1) [paper-pdf](http://arxiv.org/pdf/2210.14072v1)

**Authors**: Tong Cao, Jérémie Decouchant, Jiangshan Yu

**Abstract**: We describe and analyze perishing mining, a novel block-withholding mining strategy that lures profit-driven miners away from doing useful work on the public chain by releasing block headers from a privately maintained chain. We then introduce the dual private chain (DPC) attack, where an adversary that aims at double spending increases its success rate by intermittently dedicating part of its hash power to perishing mining. We detail the DPC attack's Markov decision process, evaluate its double spending success rate using Monte Carlo simulations. We show that the DPC attack lowers Bitcoin's security bound in the presence of profit-driven miners that do not wait to validate the transactions of a block before mining on it.



## **11. A White-Box Adversarial Attack Against a Digital Twin**

cs.CR

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.14018v1) [paper-pdf](http://arxiv.org/pdf/2210.14018v1)

**Authors**: Wilson Patterson, Ivan Fernandez, Subash Neupane, Milan Parmar, Sudip Mittal, Shahram Rahimi

**Abstract**: Recent research has shown that Machine Learning/Deep Learning (ML/DL) models are particularly vulnerable to adversarial perturbations, which are small changes made to the input data in order to fool a machine learning classifier. The Digital Twin, which is typically described as consisting of a physical entity, a virtual counterpart, and the data connections in between, is increasingly being investigated as a means of improving the performance of physical entities by leveraging computational techniques, which are enabled by the virtual counterpart. This paper explores the susceptibility of Digital Twin (DT), a virtual model designed to accurately reflect a physical object using ML/DL classifiers that operate as Cyber Physical Systems (CPS), to adversarial attacks. As a proof of concept, we first formulate a DT of a vehicular system using a deep neural network architecture and then utilize it to launch an adversarial attack. We attack the DT model by perturbing the input to the trained model and show how easily the model can be broken with white-box attacks.



## **12. Causal Information Bottleneck Boosts Adversarial Robustness of Deep Neural Network**

cs.LG

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.14229v1) [paper-pdf](http://arxiv.org/pdf/2210.14229v1)

**Authors**: Huan Hua, Jun Yan, Xi Fang, Weiquan Huang, Huilin Yin, Wancheng Ge

**Abstract**: The information bottleneck (IB) method is a feasible defense solution against adversarial attacks in deep learning. However, this method suffers from the spurious correlation, which leads to the limitation of its further improvement of adversarial robustness. In this paper, we incorporate the causal inference into the IB framework to alleviate such a problem. Specifically, we divide the features obtained by the IB method into robust features (content information) and non-robust features (style information) via the instrumental variables to estimate the causal effects. With the utilization of such a framework, the influence of non-robust features could be mitigated to strengthen the adversarial robustness. We make an analysis of the effectiveness of our proposed method. The extensive experiments in MNIST, FashionMNIST, and CIFAR-10 show that our method exhibits the considerable robustness against multiple adversarial attacks. Our code would be released.



## **13. CalFAT: Calibrated Federated Adversarial Training with Label Skewness**

cs.LG

Accepted to the Conference on the Advances in Neural Information  Processing Systems (NeurIPS) 2022

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2205.14926v2) [paper-pdf](http://arxiv.org/pdf/2205.14926v2)

**Authors**: Chen Chen, Yuchen Liu, Xingjun Ma, Lingjuan Lyu

**Abstract**: Recent studies have shown that, like traditional machine learning, federated learning (FL) is also vulnerable to adversarial attacks. To improve the adversarial robustness of FL, federated adversarial training (FAT) methods have been proposed to apply adversarial training locally before global aggregation. Although these methods demonstrate promising results on independent identically distributed (IID) data, they suffer from training instability on non-IID data with label skewness, resulting in degraded natural accuracy. This tends to hinder the application of FAT in real-world applications where the label distribution across the clients is often skewed. In this paper, we study the problem of FAT under label skewness, and reveal one root cause of the training instability and natural accuracy degradation issues: skewed labels lead to non-identical class probabilities and heterogeneous local models. We then propose a Calibrated FAT (CalFAT) approach to tackle the instability issue by calibrating the logits adaptively to balance the classes. We show both theoretically and empirically that the optimization of CalFAT leads to homogeneous local models across the clients and better convergence points.



## **14. FocusedCleaner: Sanitizing Poisoned Graphs for Robust GNN-based Node Classification**

cs.LG

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.13815v1) [paper-pdf](http://arxiv.org/pdf/2210.13815v1)

**Authors**: Yulin Zhu, Liang Tong, Kai Zhou

**Abstract**: Recently, a lot of research attention has been devoted to exploring Web security, a most representative topic is the adversarial robustness of graph mining algorithms. Especially, a widely deployed adversarial attacks formulation is the graph manipulation attacks by modifying the relational data to mislead the Graph Neural Networks' (GNNs) predictions. Naturally, an intrinsic question one would ask is whether we can accurately identify the manipulations over graphs - we term this problem as poisoned graph sanitation. In this paper, we present FocusedCleaner, a poisoned graph sanitation framework consisting of two modules: bi-level structural learning and victim node detection. In particular, the structural learning module will reserve the attack process to steadily sanitize the graph while the detection module provides the "focus" - a narrowed and more accurate search region - to structural learning. These two modules will operate in iterations and reinforce each other to sanitize a poisoned graph step by step. Extensive experiments demonstrate that FocusedCleaner outperforms the state-of-the-art baselines both on poisoned graph sanitation and improving robustness.



## **15. Flexible Android Malware Detection Model based on Generative Adversarial Networks with Code Tensor**

cs.CR

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.14225v1) [paper-pdf](http://arxiv.org/pdf/2210.14225v1)

**Authors**: Zhao Yang, Fengyang Deng, Linxi Han

**Abstract**: The behavior of malware threats is gradually increasing, heightened the need for malware detection. However, existing malware detection methods only target at the existing malicious samples, the detection of fresh malicious code and variants of malicious code is limited. In this paper, we propose a novel scheme that detects malware and its variants efficiently. Based on the idea of the generative adversarial networks (GANs), we obtain the `true' sample distribution that satisfies the characteristics of the real malware, use them to deceive the discriminator, thus achieve the defense against malicious code attacks and improve malware detection. Firstly, a new Android malware APK to image texture feature extraction segmentation method is proposed, which is called segment self-growing texture segmentation algorithm. Secondly, tensor singular value decomposition (tSVD) based on the low-tubal rank transforms malicious features with different sizes into a fixed third-order tensor uniformly, which is entered into the neural network for training and learning. Finally, a flexible Android malware detection model based on GANs with code tensor (MTFD-GANs) is proposed. Experiments show that the proposed model can generally surpass the traditional malware detection model, with a maximum improvement efficiency of 41.6\%. At the same time, the newly generated samples of the GANs generator greatly enrich the sample diversity. And retraining malware detector can effectively improve the detection efficiency and robustness of traditional models.



## **16. Differential Evolution based Dual Adversarial Camouflage: Fooling Human Eyes and Object Detectors**

cs.CV

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.08870v2) [paper-pdf](http://arxiv.org/pdf/2210.08870v2)

**Authors**: Jialiang Sun, Tingsong Jiang, Wen Yao, Donghua Wang, Xiaoqian Chen

**Abstract**: Recent studies reveal that deep neural network (DNN) based object detectors are vulnerable to adversarial attacks in the form of adding the perturbation to the images, leading to the wrong output of object detectors. Most current existing works focus on generating perturbed images, also called adversarial examples, to fool object detectors. Though the generated adversarial examples themselves can remain a certain naturalness, most of them can still be easily observed by human eyes, which limits their further application in the real world. To alleviate this problem, we propose a differential evolution based dual adversarial camouflage (DE_DAC) method, composed of two stages to fool human eyes and object detectors simultaneously. Specifically, we try to obtain the camouflage texture, which can be rendered over the surface of the object. In the first stage, we optimize the global texture to minimize the discrepancy between the rendered object and the scene images, making human eyes difficult to distinguish. In the second stage, we design three loss functions to optimize the local texture, making object detectors ineffective. In addition, we introduce the differential evolution algorithm to search for the near-optimal areas of the object to attack, improving the adversarial performance under certain attack area limitations. Besides, we also study the performance of adaptive DE_DAC, which can be adapted to the environment. Experiments show that our proposed method could obtain a good trade-off between the fooling human eyes and object detectors under multiple specific scenes and objects.



## **17. Musings on the HashGraph Protocol: Its Security and Its Limitations**

cs.CR

30 pages, 16 figures

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.13682v1) [paper-pdf](http://arxiv.org/pdf/2210.13682v1)

**Authors**: Vinesh Sridhar, Erica Blum, Jonathan Katz

**Abstract**: The HashGraph Protocol is a Byzantine fault tolerant atomic broadcast protocol. Its novel use of locally stored metadata allows parties to recover a consistent ordering of their log just by examining their local data, removing the need for a voting protocol. Our paper's first contribution is to present a rewritten proof of security for the HashGraph Protocol that follows the consistency and liveness paradigm used in the atomic broadcast literature. In our second contribution, we show a novel adversarial strategy that stalls the protocol from committing data to the log for an expected exponential number of rounds. This proves tight the exponential upper bound conjectured in the original paper. We believe that our proof of security will make it easier to compare HashGraph with other atomic broadcast protocols and to incorporate its ideas into new constructions. We also believe that our attack might inspire more research into similar attacks for other DAG-based atomic broadcast protocols.



## **18. Analyzing Privacy Leakage in Machine Learning via Multiple Hypothesis Testing: A Lesson From Fano**

cs.LG

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.13662v1) [paper-pdf](http://arxiv.org/pdf/2210.13662v1)

**Authors**: Chuan Guo, Alexandre Sablayrolles, Maziar Sanjabi

**Abstract**: Differential privacy (DP) is by far the most widely accepted framework for mitigating privacy risks in machine learning. However, exactly how small the privacy parameter $\epsilon$ needs to be to protect against certain privacy risks in practice is still not well-understood. In this work, we study data reconstruction attacks for discrete data and analyze it under the framework of multiple hypothesis testing. We utilize different variants of the celebrated Fano's inequality to derive upper bounds on the inferential power of a data reconstruction adversary when the model is trained differentially privately. Importantly, we show that if the underlying private data takes values from a set of size $M$, then the target privacy parameter $\epsilon$ can be $O(\log M)$ before the adversary gains significant inferential power. Our analysis offers theoretical evidence for the empirical effectiveness of DP against data reconstruction attacks even at relatively large values of $\epsilon$.



## **19. SpacePhish: The Evasion-space of Adversarial Attacks against Phishing Website Detectors using Machine Learning**

cs.CR

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.13660v1) [paper-pdf](http://arxiv.org/pdf/2210.13660v1)

**Authors**: Giovanni Apruzzese, Mauro Conti, Ying Yuan

**Abstract**: Existing literature on adversarial Machine Learning (ML) focuses either on showing attacks that break every ML model, or defenses that withstand most attacks. Unfortunately, little consideration is given to the actual \textit{cost} of the attack or the defense. Moreover, adversarial samples are often crafted in the "feature-space", making the corresponding evaluations of questionable value. Simply put, the current situation does not allow to estimate the actual threat posed by adversarial attacks, leading to a lack of secure ML systems.   We aim to clarify such confusion in this paper. By considering the application of ML for Phishing Website Detection (PWD), we formalize the "evasion-space" in which an adversarial perturbation can be introduced to fool a ML-PWD -- demonstrating that even perturbations in the "feature-space" are useful. Then, we propose a realistic threat model describing evasion attacks against ML-PWD that are cheap to stage, and hence intrinsically more attractive for real phishers. Finally, we perform the first statistically validated assessment of state-of-the-art ML-PWD against 12 evasion attacks. Our evaluation shows (i) the true efficacy of evasion attempts that are more likely to occur; and (ii) the impact of perturbations crafted in different evasion-spaces. Our realistic evasion attempts induce a statistically significant degradation (3-10% at $p\!<$0.05), and their cheap cost makes them a subtle threat. Notably, however, some ML-PWD are immune to our most realistic attacks ($p$=0.22). Our contribution paves the way for a much needed re-assessment of adversarial attacks against ML systems for cybersecurity.



## **20. On the Robustness of Dataset Inference**

cs.LG

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.13631v1) [paper-pdf](http://arxiv.org/pdf/2210.13631v1)

**Authors**: Sebastian Szyller, Rui Zhang, Jian Liu, N. Asokan

**Abstract**: Machine learning (ML) models are costly to train as they can require a significant amount of data, computational resources and technical expertise. Thus, they constitute valuable intellectual property that needs protection from adversaries wanting to steal them. Ownership verification techniques allow the victims of model stealing attacks to demonstrate that a suspect model was in fact stolen from theirs. Although a number of ownership verification techniques based on watermarking or fingerprinting have been proposed, most of them fall short either in terms of security guarantees (well-equipped adversaries can evade verification) or computational cost. A fingerprinting technique introduced at ICLR '21, Dataset Inference (DI), has been shown to offer better robustness and efficiency than prior methods. The authors of DI provided a correctness proof for linear (suspect) models. However, in the same setting, we prove that DI suffers from high false positives (FPs) -- it can incorrectly identify an independent model trained with non-overlapping data from the same distribution as stolen. We further prove that DI also triggers FPs in realistic, non-linear suspect models. We then confirm empirically that DI leads to FPs, with high confidence. Second, we show that DI also suffers from false negatives (FNs) -- an adversary can fool DI by regularising a stolen model's decision boundaries using adversarial training, thereby leading to an FN. To this end, we demonstrate that DI fails to identify a model adversarially trained from a stolen dataset -- the setting where DI is the hardest to evade. Finally, we discuss the implications of our findings, the viability of fingerprinting-based ownership verification in general, and suggest directions for future work.



## **21. Deep VULMAN: A Deep Reinforcement Learning-Enabled Cyber Vulnerability Management Framework**

cs.AI

12 pages, 3 figures

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2208.02369v2) [paper-pdf](http://arxiv.org/pdf/2208.02369v2)

**Authors**: Soumyadeep Hore, Ankit Shah, Nathaniel D. Bastian

**Abstract**: Cyber vulnerability management is a critical function of a cybersecurity operations center (CSOC) that helps protect organizations against cyber-attacks on their computer and network systems. Adversaries hold an asymmetric advantage over the CSOC, as the number of deficiencies in these systems is increasing at a significantly higher rate compared to the expansion rate of the security teams to mitigate them in a resource-constrained environment. The current approaches are deterministic and one-time decision-making methods, which do not consider future uncertainties when prioritizing and selecting vulnerabilities for mitigation. These approaches are also constrained by the sub-optimal distribution of resources, providing no flexibility to adjust their response to fluctuations in vulnerability arrivals. We propose a novel framework, Deep VULMAN, consisting of a deep reinforcement learning agent and an integer programming method to fill this gap in the cyber vulnerability management process. Our sequential decision-making framework, first, determines the near-optimal amount of resources to be allocated for mitigation under uncertainty for a given system state and then determines the optimal set of prioritized vulnerability instances for mitigation. Our proposed framework outperforms the current methods in prioritizing the selection of important organization-specific vulnerabilities, on both simulated and real-world vulnerability data, observed over a one-year period.



## **22. Probabilistic Categorical Adversarial Attack & Adversarial Training**

cs.LG

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.09364v2) [paper-pdf](http://arxiv.org/pdf/2210.09364v2)

**Authors**: Pengfei He, Han Xu, Jie Ren, Yuxuan Wan, Zitao Liu, Jiliang Tang

**Abstract**: The existence of adversarial examples brings huge concern for people to apply Deep Neural Networks (DNNs) in safety-critical tasks. However, how to generate adversarial examples with categorical data is an important problem but lack of extensive exploration. Previously established methods leverage greedy search method, which can be very time-consuming to conduct successful attack. This also limits the development of adversarial training and potential defenses for categorical data. To tackle this problem, we propose Probabilistic Categorical Adversarial Attack (PCAA), which transfers the discrete optimization problem to a continuous problem that can be solved efficiently by Projected Gradient Descent. In our paper, we theoretically analyze its optimality and time complexity to demonstrate its significant advantage over current greedy based attacks. Moreover, based on our attack, we propose an efficient adversarial training framework. Through a comprehensive empirical study, we justify the effectiveness of our proposed attack and defense algorithms.



## **23. Driver Locations Harvesting Attack on pRide**

cs.CR

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.13263v1) [paper-pdf](http://arxiv.org/pdf/2210.13263v1)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstract**: Privacy preservation in Ride-Hailing Services (RHS) is intended to protect privacy of drivers and riders. pRide, published in IEEE Trans. Vehicular Technology 2021, is a prediction based privacy-preserving RHS protocol to match riders with an optimum driver. In the protocol, the Service Provider (SP) homomorphically computes Euclidean distances between encrypted locations of drivers and rider. Rider selects an optimum driver using decrypted distances augmented by a new-ride-emergence prediction. To improve the effectiveness of driver selection, the paper proposes an enhanced version where each driver gives encrypted distances to each corner of her grid. To thwart a rider from using these distances to launch an inference attack, the SP blinds these distances before sharing them with the rider. In this work, we propose a passive attack where an honest-but-curious adversary rider who makes a single ride request and receives the blinded distances from SP can recover the constants used to blind the distances. Using the unblinded distances, rider to driver distance and Google Nearest Road API, the adversary can obtain the precise locations of responding drivers. We conduct experiments with random on-road driver locations for four different cities. Our experiments show that we can determine the precise locations of at least 80% of the drivers participating in the enhanced pRide protocol.



## **24. SealClub: Computer-aided Paper Document Authentication**

cs.CR

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.07884v2) [paper-pdf](http://arxiv.org/pdf/2210.07884v2)

**Authors**: Martín Ochoa, Jorge Toro-Pozo, David Basin

**Abstract**: Digital authentication is a mature field, offering a range of solutions with rigorous mathematical guarantees. Nevertheless, paper documents, where cryptographic techniques are not directly applicable, are still widely utilized due to usability and legal reasons. We propose a novel approach to authenticating paper documents using smartphones by taking short videos of them. Our solution combines cryptographic and image comparison techniques to detect and highlight subtle semantic-changing attacks on rich documents, containing text and graphics, that could go unnoticed by humans. We rigorously analyze our approach, proving that it is secure against strong adversaries capable of compromising different system components. We also measure its accuracy empirically on a set of 128 videos of paper documents, half containing subtle forgeries. Our algorithm finds all forgeries accurately (no false alarms) after analyzing 5.13 frames on average (corresponding to 1.28 seconds of video). Highlighted regions are large enough to be visible to users, but small enough to precisely locate forgeries. Thus, our approach provides a promising way for users to authenticate paper documents using conventional smartphones under realistic conditions.



## **25. Sardino: Ultra-Fast Dynamic Ensemble for Secure Visual Sensing at Mobile Edge**

cs.CV

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2204.08189v3) [paper-pdf](http://arxiv.org/pdf/2204.08189v3)

**Authors**: Qun Song, Zhenyu Yan, Wenjie Luo, Rui Tan

**Abstract**: Adversarial example attack endangers the mobile edge systems such as vehicles and drones that adopt deep neural networks for visual sensing. This paper presents {\em Sardino}, an active and dynamic defense approach that renews the inference ensemble at run time to develop security against the adaptive adversary who tries to exfiltrate the ensemble and construct the corresponding effective adversarial examples. By applying consistency check and data fusion on the ensemble's predictions, Sardino can detect and thwart adversarial inputs. Compared with the training-based ensemble renewal, we use HyperNet to achieve {\em one million times} acceleration and per-frame ensemble renewal that presents the highest level of difficulty to the prerequisite exfiltration attacks. We design a run-time planner that maximizes the ensemble size in favor of security while maintaining the processing frame rate. Beyond adversarial examples, Sardino can also address the issue of out-of-distribution inputs effectively. This paper presents extensive evaluation of Sardino's performance in counteracting adversarial examples and applies it to build a real-time car-borne traffic sign recognition system. Live on-road tests show the built system's effectiveness in maintaining frame rate and detecting out-of-distribution inputs due to the false positives of a preceding YOLO-based traffic sign detector.



## **26. Ares: A System-Oriented Wargame Framework for Adversarial ML**

cs.LG

Presented at the DLS Workshop at S&P 2022

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.12952v1) [paper-pdf](http://arxiv.org/pdf/2210.12952v1)

**Authors**: Farhan Ahmed, Pratik Vaishnavi, Kevin Eykholt, Amir Rahmati

**Abstract**: Since the discovery of adversarial attacks against machine learning models nearly a decade ago, research on adversarial machine learning has rapidly evolved into an eternal war between defenders, who seek to increase the robustness of ML models against adversarial attacks, and adversaries, who seek to develop better attacks capable of weakening or defeating these defenses. This domain, however, has found little buy-in from ML practitioners, who are neither overtly concerned about these attacks affecting their systems in the real world nor are willing to trade off the accuracy of their models in pursuit of robustness against these attacks.   In this paper, we motivate the design and implementation of Ares, an evaluation framework for adversarial ML that allows researchers to explore attacks and defenses in a realistic wargame-like environment. Ares frames the conflict between the attacker and defender as two agents in a reinforcement learning environment with opposing objectives. This allows the introduction of system-level evaluation metrics such as time to failure and evaluation of complex strategies such as moving target defenses. We provide the results of our initial exploration involving a white-box attacker against an adversarially trained defender.



## **27. Backdoor Attacks in Federated Learning by Rare Embeddings and Gradient Ensembling**

cs.LG

Accepted to EMNLP 2022, 9 pages and Appendix

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2204.14017v2) [paper-pdf](http://arxiv.org/pdf/2204.14017v2)

**Authors**: KiYoon Yoo, Nojun Kwak

**Abstract**: Recent advances in federated learning have demonstrated its promising capability to learn on decentralized datasets. However, a considerable amount of work has raised concerns due to the potential risks of adversaries participating in the framework to poison the global model for an adversarial purpose. This paper investigates the feasibility of model poisoning for backdoor attacks through rare word embeddings of NLP models. In text classification, less than 1% of adversary clients suffices to manipulate the model output without any drop in the performance on clean sentences. For a less complex dataset, a mere 0.1% of adversary clients is enough to poison the global model effectively. We also propose a technique specialized in the federated learning scheme called Gradient Ensemble, which enhances the backdoor performance in all our experimental settings.



## **28. TextHacker: Learning based Hybrid Local Search Algorithm for Text Hard-label Adversarial Attack**

cs.CL

Accepted by EMNLP 2022 Findings, Code is available at  https://github.com/JHL-HUST/TextHacker

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2201.08193v2) [paper-pdf](http://arxiv.org/pdf/2201.08193v2)

**Authors**: Zhen Yu, Xiaosen Wang, Wanxiang Che, Kun He

**Abstract**: Existing textual adversarial attacks usually utilize the gradient or prediction confidence to generate adversarial examples, making it hard to be deployed in real-world applications. To this end, we consider a rarely investigated but more rigorous setting, namely hard-label attack, in which the attacker can only access the prediction label. In particular, we find we can learn the importance of different words via the change on prediction label caused by word substitutions on the adversarial examples. Based on this observation, we propose a novel adversarial attack, termed Text Hard-label attacker (TextHacker). TextHacker randomly perturbs lots of words to craft an adversarial example. Then, TextHacker adopts a hybrid local search algorithm with the estimation of word importance from the attack history to minimize the adversarial perturbation. Extensive evaluations for text classification and textual entailment show that TextHacker significantly outperforms existing hard-label attacks regarding the attack performance as well as adversary quality.



## **29. A Secure Design Pattern Approach Toward Tackling Lateral-Injection Attacks**

cs.CR

4 pages, 3 figures. Accepted to The 15th IEEE International  Conference on Security of Information and Networks (SIN)

**SubmitDate**: 2022-10-23    [abs](http://arxiv.org/abs/2210.12877v1) [paper-pdf](http://arxiv.org/pdf/2210.12877v1)

**Authors**: Chidera Biringa, Gökhan Kul

**Abstract**: Software weaknesses that create attack surfaces for adversarial exploits, such as lateral SQL injection (LSQLi) attacks, are usually introduced during the design phase of software development. Security design patterns are sometimes applied to tackle these weaknesses. However, due to the stealthy nature of lateral-based attacks, employing traditional security patterns to address these threats is insufficient. Hence, we present SEAL, a secure design that extrapolates architectural, design, and implementation abstraction levels to delegate security strategies toward tackling LSQLi attacks. We evaluated SEAL using case study software, where we assumed the role of an adversary and injected several attack vectors tasked with compromising the confidentiality and integrity of its database. Our evaluation of SEAL demonstrated its capacity to address LSQLi attacks.



## **30. TAPE: Assessing Few-shot Russian Language Understanding**

cs.CL

Accepted to EMNLP 2022 Findings

**SubmitDate**: 2022-10-23    [abs](http://arxiv.org/abs/2210.12813v1) [paper-pdf](http://arxiv.org/pdf/2210.12813v1)

**Authors**: Ekaterina Taktasheva, Tatiana Shavrina, Alena Fenogenova, Denis Shevelev, Nadezhda Katricheva, Maria Tikhonova, Albina Akhmetgareeva, Oleg Zinkevich, Anastasiia Bashmakova, Svetlana Iordanskaia, Alena Spiridonova, Valentina Kurenshchikova, Ekaterina Artemova, Vladislav Mikhailov

**Abstract**: Recent advances in zero-shot and few-shot learning have shown promise for a scope of research and practical purposes. However, this fast-growing area lacks standardized evaluation suites for non-English languages, hindering progress outside the Anglo-centric paradigm. To address this line of research, we propose TAPE (Text Attack and Perturbation Evaluation), a novel benchmark that includes six more complex NLU tasks for Russian, covering multi-hop reasoning, ethical concepts, logic and commonsense knowledge. The TAPE's design focuses on systematic zero-shot and few-shot NLU evaluation: (i) linguistic-oriented adversarial attacks and perturbations for analyzing robustness, and (ii) subpopulations for nuanced interpretation. The detailed analysis of testing the autoregressive baselines indicates that simple spelling-based perturbations affect the performance the most, while paraphrasing the input has a more negligible effect. At the same time, the results demonstrate a significant gap between the neural and human baselines for most tasks. We publicly release TAPE (tape-benchmark.com) to foster research on robust LMs that can generalize to new tasks when little to no supervision is available.



## **31. Adversarial Pretraining of Self-Supervised Deep Networks: Past, Present and Future**

cs.LG

**SubmitDate**: 2022-10-23    [abs](http://arxiv.org/abs/2210.13463v1) [paper-pdf](http://arxiv.org/pdf/2210.13463v1)

**Authors**: Guo-Jun Qi, Mubarak Shah

**Abstract**: In this paper, we review adversarial pretraining of self-supervised deep networks including both convolutional neural networks and vision transformers. Unlike the adversarial training with access to labeled examples, adversarial pretraining is complicated as it only has access to unlabeled examples. To incorporate adversaries into pretraining models on either input or feature level, we find that existing approaches are largely categorized into two groups: memory-free instance-wise attacks imposing worst-case perturbations on individual examples, and memory-based adversaries shared across examples over iterations. In particular, we review several representative adversarial pretraining models based on Contrastive Learning (CL) and Masked Image Modeling (MIM), respectively, two popular self-supervised pretraining methods in literature. We also review miscellaneous issues about computing overheads, input-/feature-level adversaries, as well as other adversarial pretraining approaches beyond the above two groups. Finally, we discuss emerging trends and future directions about the relations between adversarial and cooperative pretraining, unifying adversarial CL and MIM pretraining, and the trade-off between accuracy and robustness in adversarial pretraining.



## **32. GANI: Global Attacks on Graph Neural Networks via Imperceptible Node Injections**

cs.LG

**SubmitDate**: 2022-10-23    [abs](http://arxiv.org/abs/2210.12598v1) [paper-pdf](http://arxiv.org/pdf/2210.12598v1)

**Authors**: Junyuan Fang, Haixian Wen, Jiajing Wu, Qi Xuan, Zibin Zheng, Chi K. Tse

**Abstract**: Graph neural networks (GNNs) have found successful applications in various graph-related tasks. However, recent studies have shown that many GNNs are vulnerable to adversarial attacks. In a vast majority of existing studies, adversarial attacks on GNNs are launched via direct modification of the original graph such as adding/removing links, which may not be applicable in practice. In this paper, we focus on a realistic attack operation via injecting fake nodes. The proposed Global Attack strategy via Node Injection (GANI) is designed under the comprehensive consideration of an unnoticeable perturbation setting from both structure and feature domains. Specifically, to make the node injections as imperceptible and effective as possible, we propose a sampling operation to determine the degree of the newly injected nodes, and then generate features and select neighbors for these injected nodes based on the statistical information of features and evolutionary perturbations obtained from a genetic algorithm, respectively. In particular, the proposed feature generation mechanism is suitable for both binary and continuous node features. Extensive experimental results on benchmark datasets against both general and defended GNNs show strong attack performance of GANI. Moreover, the imperceptibility analyses also demonstrate that GANI achieves a relatively unnoticeable injection on benchmark datasets.



## **33. Efficient (Soft) Q-Learning for Text Generation with Limited Good Data**

cs.CL

Code available at  https://github.com/HanGuo97/soft-Q-learning-for-text-generation

**SubmitDate**: 2022-10-22    [abs](http://arxiv.org/abs/2106.07704v4) [paper-pdf](http://arxiv.org/pdf/2106.07704v4)

**Authors**: Han Guo, Bowen Tan, Zhengzhong Liu, Eric P. Xing, Zhiting Hu

**Abstract**: Maximum likelihood estimation (MLE) is the predominant algorithm for training text generation models. This paradigm relies on direct supervision examples, which is not applicable to many emerging applications, such as generating adversarial attacks or generating prompts to control language models. Reinforcement learning (RL) on the other hand offers a more flexible solution by allowing users to plug in arbitrary task metrics as reward. Yet previous RL algorithms for text generation, such as policy gradient (on-policy RL) and Q-learning (off-policy RL), are often notoriously inefficient or unstable to train due to the large sequence space and the sparse reward received only at the end of sequences. In this paper, we introduce a new RL formulation for text generation from the soft Q-learning (SQL) perspective. It enables us to draw from the latest RL advances, such as path consistency learning, to combine the best of on-/off-policy updates, and learn effectively from sparse reward. We apply the approach to a wide range of novel text generation tasks, including learning from noisy/negative examples, adversarial attacks, and prompt generation. Experiments show our approach consistently outperforms both task-specialized algorithms and the previous RL methods.



## **34. Hindering Adversarial Attacks with Implicit Neural Representations**

cs.LG

**SubmitDate**: 2022-10-22    [abs](http://arxiv.org/abs/2210.13982v1) [paper-pdf](http://arxiv.org/pdf/2210.13982v1)

**Authors**: Andrei A. Rusu, Dan A. Calian, Sven Gowal, Raia Hadsell

**Abstract**: We introduce the Lossy Implicit Network Activation Coding (LINAC) defence, an input transformation which successfully hinders several common adversarial attacks on CIFAR-$10$ classifiers for perturbations up to $\epsilon = 8/255$ in $L_\infty$ norm and $\epsilon = 0.5$ in $L_2$ norm. Implicit neural representations are used to approximately encode pixel colour intensities in $2\text{D}$ images such that classifiers trained on transformed data appear to have robustness to small perturbations without adversarial training or large drops in performance. The seed of the random number generator used to initialise and train the implicit neural representation turns out to be necessary information for stronger generic attacks, suggesting its role as a private key. We devise a Parametric Bypass Approximation (PBA) attack strategy for key-based defences, which successfully invalidates an existing method in this category. Interestingly, our LINAC defence also hinders some transfer and adaptive attacks, including our novel PBA strategy. Our results emphasise the importance of a broad range of customised attacks despite apparent robustness according to standard evaluations. LINAC source code and parameters of defended classifier evaluated throughout this submission are available: https://github.com/deepmind/linac



## **35. RORL: Robust Offline Reinforcement Learning via Conservative Smoothing**

cs.LG

Accepted by Advances in Neural Information Processing Systems  (NeurIPS) 2022

**SubmitDate**: 2022-10-22    [abs](http://arxiv.org/abs/2206.02829v3) [paper-pdf](http://arxiv.org/pdf/2206.02829v3)

**Authors**: Rui Yang, Chenjia Bai, Xiaoteng Ma, Zhaoran Wang, Chongjie Zhang, Lei Han

**Abstract**: Offline reinforcement learning (RL) provides a promising direction to exploit massive amount of offline data for complex decision-making tasks. Due to the distribution shift issue, current offline RL algorithms are generally designed to be conservative in value estimation and action selection. However, such conservatism can impair the robustness of learned policies when encountering observation deviation under realistic conditions, such as sensor errors and adversarial attacks. To trade off robustness and conservatism, we propose Robust Offline Reinforcement Learning (RORL) with a novel conservative smoothing technique. In RORL, we explicitly introduce regularization on the policy and the value function for states near the dataset, as well as additional conservative value estimation on these states. Theoretically, we show RORL enjoys a tighter suboptimality bound than recent theoretical results in linear MDPs. We demonstrate that RORL can achieve state-of-the-art performance on the general offline RL benchmark and is considerably robust to adversarial observation perturbations.



## **36. ADDMU: Detection of Far-Boundary Adversarial Examples with Data and Model Uncertainty Estimation**

cs.CL

18 pages, EMNLP 2022, main conference, long paper

**SubmitDate**: 2022-10-22    [abs](http://arxiv.org/abs/2210.12396v1) [paper-pdf](http://arxiv.org/pdf/2210.12396v1)

**Authors**: Fan Yin, Yao Li, Cho-Jui Hsieh, Kai-Wei Chang

**Abstract**: Adversarial Examples Detection (AED) is a crucial defense technique against adversarial attacks and has drawn increasing attention from the Natural Language Processing (NLP) community. Despite the surge of new AED methods, our studies show that existing methods heavily rely on a shortcut to achieve good performance. In other words, current search-based adversarial attacks in NLP stop once model predictions change, and thus most adversarial examples generated by those attacks are located near model decision boundaries. To surpass this shortcut and fairly evaluate AED methods, we propose to test AED methods with \textbf{F}ar \textbf{B}oundary (\textbf{FB}) adversarial examples. Existing methods show worse than random guess performance under this scenario. To overcome this limitation, we propose a new technique, \textbf{ADDMU}, \textbf{a}dversary \textbf{d}etection with \textbf{d}ata and \textbf{m}odel \textbf{u}ncertainty, which combines two types of uncertainty estimation for both regular and FB adversarial example detection. Our new method outperforms previous methods by 3.6 and 6.0 \emph{AUC} points under each scenario. Finally, our analysis shows that the two types of uncertainty provided by \textbf{ADDMU} can be leveraged to characterize adversarial examples and identify the ones that contribute most to model's robustness in adversarial training.



## **37. TCAB: A Large-Scale Text Classification Attack Benchmark**

cs.LG

32 pages, 7 figures, and 14 tables

**SubmitDate**: 2022-10-21    [abs](http://arxiv.org/abs/2210.12233v1) [paper-pdf](http://arxiv.org/pdf/2210.12233v1)

**Authors**: Kalyani Asthana, Zhouhang Xie, Wencong You, Adam Noack, Jonathan Brophy, Sameer Singh, Daniel Lowd

**Abstract**: We introduce the Text Classification Attack Benchmark (TCAB), a dataset for analyzing, understanding, detecting, and labeling adversarial attacks against text classifiers. TCAB includes 1.5 million attack instances, generated by twelve adversarial attacks targeting three classifiers trained on six source datasets for sentiment analysis and abuse detection in English. Unlike standard text classification, text attacks must be understood in the context of the target classifier that is being attacked, and thus features of the target classifier are important as well. TCAB includes all attack instances that are successful in flipping the predicted label; a subset of the attacks are also labeled by human annotators to determine how frequently the primary semantics are preserved. The process of generating attacks is automated, so that TCAB can easily be extended to incorporate new text attacks and better classifiers as they are developed. In addition to the primary tasks of detecting and labeling attacks, TCAB can also be used for attack localization, attack target labeling, and attack characterization. TCAB code and dataset are available at https://react-nlp.github.io/tcab/.



## **38. The Dark Side of AutoML: Towards Architectural Backdoor Search**

cs.CR

**SubmitDate**: 2022-10-21    [abs](http://arxiv.org/abs/2210.12179v1) [paper-pdf](http://arxiv.org/pdf/2210.12179v1)

**Authors**: Ren Pang, Changjiang Li, Zhaohan Xi, Shouling Ji, Ting Wang

**Abstract**: This paper asks the intriguing question: is it possible to exploit neural architecture search (NAS) as a new attack vector to launch previously improbable attacks? Specifically, we present EVAS, a new attack that leverages NAS to find neural architectures with inherent backdoors and exploits such vulnerability using input-aware triggers. Compared with existing attacks, EVAS demonstrates many interesting properties: (i) it does not require polluting training data or perturbing model parameters; (ii) it is agnostic to downstream fine-tuning or even re-training from scratch; (iii) it naturally evades defenses that rely on inspecting model parameters or training data. With extensive evaluation on benchmark datasets, we show that EVAS features high evasiveness, transferability, and robustness, thereby expanding the adversary's design spectrum. We further characterize the mechanisms underlying EVAS, which are possibly explainable by architecture-level ``shortcuts'' that recognize trigger patterns. This work raises concerns about the current practice of NAS and points to potential directions to develop effective countermeasures.



## **39. Evolution of Neural Tangent Kernels under Benign and Adversarial Training**

cs.LG

Accepted to the Conference on Advances in Neural Information  Processing Systems (NeurIPS) 2022

**SubmitDate**: 2022-10-21    [abs](http://arxiv.org/abs/2210.12030v1) [paper-pdf](http://arxiv.org/pdf/2210.12030v1)

**Authors**: Noel Loo, Ramin Hasani, Alexander Amini, Daniela Rus

**Abstract**: Two key challenges facing modern deep learning are mitigating deep networks' vulnerability to adversarial attacks and understanding deep learning's generalization capabilities. Towards the first issue, many defense strategies have been developed, with the most common being Adversarial Training (AT). Towards the second challenge, one of the dominant theories that has emerged is the Neural Tangent Kernel (NTK) -- a characterization of neural network behavior in the infinite-width limit. In this limit, the kernel is frozen, and the underlying feature map is fixed. In finite widths, however, there is evidence that feature learning happens at the earlier stages of the training (kernel learning) before a second phase where the kernel remains fixed (lazy training). While prior work has aimed at studying adversarial vulnerability through the lens of the frozen infinite-width NTK, there is no work that studies the adversarial robustness of the empirical/finite NTK during training. In this work, we perform an empirical study of the evolution of the empirical NTK under standard and adversarial training, aiming to disambiguate the effect of adversarial training on kernel learning and lazy training. We find under adversarial training, the empirical NTK rapidly converges to a different kernel (and feature map) than standard training. This new kernel provides adversarial robustness, even when non-robust training is performed on top of it. Furthermore, we find that adversarial training on top of a fixed kernel can yield a classifier with $76.1\%$ robust accuracy under PGD attacks with $\varepsilon = 4/255$ on CIFAR-10.



## **40. Fact-Saboteurs: A Taxonomy of Evidence Manipulation Attacks against Fact-Verification Systems**

cs.CR

**SubmitDate**: 2022-10-21    [abs](http://arxiv.org/abs/2209.03755v2) [paper-pdf](http://arxiv.org/pdf/2209.03755v2)

**Authors**: Sahar Abdelnabi, Mario Fritz

**Abstract**: Mis- and disinformation are now a substantial global threat to our security and safety. To cope with the scale of online misinformation, one viable solution is to automate the fact-checking of claims by retrieving and verifying against relevant evidence. While major recent advances have been achieved in pushing forward the automatic fact-verification, a comprehensive evaluation of the possible attack vectors against such systems is still lacking. Particularly, the automated fact-verification process might be vulnerable to the exact disinformation campaigns it is trying to combat. In this work, we assume an adversary that automatically tampers with the online evidence in order to disrupt the fact-checking model via camouflaging the relevant evidence, or planting a misleading one. We first propose an exploratory taxonomy that spans these two targets and the different threat model dimensions. Guided by this, we design and propose several potential attack methods. We show that it is possible to subtly modify claim-salient snippets in the evidence, in addition to generating diverse and claim-aligned evidence. As a result, we highly degrade the fact-checking performance under many different permutations of the taxonomy's dimensions. The attacks are also robust against post-hoc modifications of the claim. Our analysis further hints at potential limitations in models' inference when faced with contradicting evidence. We emphasize that these attacks can have harmful implications on the inspectable and human-in-the-loop usage scenarios of such models, and we conclude by discussing challenges and directions for future defenses.



## **41. Assaying Out-Of-Distribution Generalization in Transfer Learning**

cs.LG

**SubmitDate**: 2022-10-21    [abs](http://arxiv.org/abs/2207.09239v2) [paper-pdf](http://arxiv.org/pdf/2207.09239v2)

**Authors**: Florian Wenzel, Andrea Dittadi, Peter Vincent Gehler, Carl-Johann Simon-Gabriel, Max Horn, Dominik Zietlow, David Kernert, Chris Russell, Thomas Brox, Bernt Schiele, Bernhard Schölkopf, Francesco Locatello

**Abstract**: Since out-of-distribution generalization is a generally ill-posed problem, various proxy targets (e.g., calibration, adversarial robustness, algorithmic corruptions, invariance across shifts) were studied across different research programs resulting in different recommendations. While sharing the same aspirational goal, these approaches have never been tested under the same experimental conditions on real data. In this paper, we take a unified view of previous work, highlighting message discrepancies that we address empirically, and providing recommendations on how to measure the robustness of a model and how to improve it. To this end, we collect 172 publicly available dataset pairs for training and out-of-distribution evaluation of accuracy, calibration error, adversarial attacks, environment invariance, and synthetic corruptions. We fine-tune over 31k networks, from nine different architectures in the many- and few-shot setting. Our findings confirm that in- and out-of-distribution accuracies tend to increase jointly, but show that their relation is largely dataset-dependent, and in general more nuanced and more complex than posited by previous, smaller scale studies.



## **42. A Survey of Machine Unlearning**

cs.LG

discuss new and recent works as well as proof-reading

**SubmitDate**: 2022-10-21    [abs](http://arxiv.org/abs/2209.02299v5) [paper-pdf](http://arxiv.org/pdf/2209.02299v5)

**Authors**: Thanh Tam Nguyen, Thanh Trung Huynh, Phi Le Nguyen, Alan Wee-Chung Liew, Hongzhi Yin, Quoc Viet Hung Nguyen

**Abstract**: Today, computer systems hold large amounts of personal data. Yet while such an abundance of data allows breakthroughs in artificial intelligence, and especially machine learning (ML), its existence can be a threat to user privacy, and it can weaken the bonds of trust between humans and AI. Recent regulations now require that, on request, private information about a user must be removed from both computer systems and from ML models, i.e. ``the right to be forgotten''). While removing data from back-end databases should be straightforward, it is not sufficient in the AI context as ML models often `remember' the old data. Contemporary adversarial attacks on trained models have proven that we can learn whether an instance or an attribute belonged to the training data. This phenomenon calls for a new paradigm, namely machine unlearning, to make ML models forget about particular data. It turns out that recent works on machine unlearning have not been able to completely solve the problem due to the lack of common frameworks and resources. Therefore, this paper aspires to present a comprehensive examination of machine unlearning's concepts, scenarios, methods, and applications. Specifically, as a category collection of cutting-edge studies, the intention behind this article is to serve as a comprehensive resource for researchers and practitioners seeking an introduction to machine unlearning and its formulations, design criteria, removal requests, algorithms, and applications. In addition, we aim to highlight the key findings, current trends, and new research areas that have not yet featured the use of machine unlearning but could benefit greatly from it. We hope this survey serves as a valuable resource for ML researchers and those seeking to innovate privacy technologies. Our resources are publicly available at https://github.com/tamlhp/awesome-machine-unlearning.



## **43. Identifying Human Strategies for Generating Word-Level Adversarial Examples**

cs.CL

Findings of EMNLP 2022

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.11598v1) [paper-pdf](http://arxiv.org/pdf/2210.11598v1)

**Authors**: Maximilian Mozes, Bennett Kleinberg, Lewis D. Griffin

**Abstract**: Adversarial examples in NLP are receiving increasing research attention. One line of investigation is the generation of word-level adversarial examples against fine-tuned Transformer models that preserve naturalness and grammaticality. Previous work found that human- and machine-generated adversarial examples are comparable in their naturalness and grammatical correctness. Most notably, humans were able to generate adversarial examples much more effortlessly than automated attacks. In this paper, we provide a detailed analysis of exactly how humans create these adversarial examples. By exploring the behavioural patterns of human workers during the generation process, we identify statistically significant tendencies based on which words humans prefer to select for adversarial replacement (e.g., word frequencies, word saliencies, sentiment) as well as where and when words are replaced in an input sequence. With our findings, we seek to inspire efforts that harness human strategies for more robust NLP models.



## **44. Similarity of Neural Architectures Based on Input Gradient Transferability**

cs.LG

21pages, 10 figures, 1.5MB

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.11407v1) [paper-pdf](http://arxiv.org/pdf/2210.11407v1)

**Authors**: Jaehui Hwang, Dongyoon Han, Byeongho Heo, Song Park, Sanghyuk Chun, Jong-Seok Lee

**Abstract**: In this paper, we aim to design a quantitative similarity function between two neural architectures. Specifically, we define a model similarity using input gradient transferability. We generate adversarial samples of two networks and measure the average accuracy of the networks on adversarial samples of each other. If two networks are highly correlated, then the attack transferability will be high, resulting in high similarity. Using the similarity score, we investigate two topics: (1) Which network component contributes to the model diversity? (2) How does model diversity affect practical scenarios? We answer the first question by providing feature importance analysis and clustering analysis. The second question is validated by two different scenarios: model ensemble and knowledge distillation. Our findings show that model diversity takes a key role when interacting with different neural architectures. For example, we found that more diversity leads to better ensemble performance. We also observe that the relationship between teacher and student networks and distillation performance depends on the choice of the base architecture of the teacher and student networks. We expect our analysis tool helps a high-level understanding of differences between various neural architectures as well as practical guidance when using multiple architectures.



## **45. Surprises in adversarially-trained linear regression**

stat.ML

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2205.12695v2) [paper-pdf](http://arxiv.org/pdf/2205.12695v2)

**Authors**: Antônio H. Ribeiro, Dave Zachariah, Thomas B. Schön

**Abstract**: State-of-the-art machine learning models can be vulnerable to very small input perturbations that are adversarially constructed. Adversarial training is an effective approach to defend against such examples. It is formulated as a min-max problem, searching for the best solution when the training data was corrupted by the worst-case attacks. For linear regression problems, adversarial training can be formulated as a convex problem. We use this reformulation to make two technical contributions: First, we formulate the training problem as an instance of robust regression to reveal its connection to parameter-shrinking methods, specifically that $\ell_\infty$-adversarial training produces sparse solutions. Secondly, we study adversarial training in the overparameterized regime, i.e. when there are more parameters than data. We prove that adversarial training with small disturbances gives the solution with the minimum-norm that interpolates the training data. Ridge regression and lasso approximate such interpolating solutions as their regularization parameter vanishes. By contrast, for adversarial training, the transition into the interpolation regime is abrupt and for non-zero values of disturbance. This result is proved and illustrated with numerical examples.



## **46. Attacking Motion Estimation with Adversarial Snow**

cs.CV

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.11242v1) [paper-pdf](http://arxiv.org/pdf/2210.11242v1)

**Authors**: Jenny Schmalfuss, Lukas Mehl, Andrés Bruhn

**Abstract**: Current adversarial attacks for motion estimation (optical flow) optimize small per-pixel perturbations, which are unlikely to appear in the real world. In contrast, we exploit a real-world weather phenomenon for a novel attack with adversarially optimized snow. At the core of our attack is a differentiable renderer that consistently integrates photorealistic snowflakes with realistic motion into the 3D scene. Through optimization we obtain adversarial snow that significantly impacts the optical flow while being indistinguishable from ordinary snow. Surprisingly, the impact of our novel attack is largest on methods that previously showed a high robustness to small L_p perturbations.



## **47. UKP-SQuARE v2: Explainability and Adversarial Attacks for Trustworthy QA**

cs.CL

Accepted at AACL 2022 as Demo Paper

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2208.09316v3) [paper-pdf](http://arxiv.org/pdf/2208.09316v3)

**Authors**: Rachneet Sachdeva, Haritz Puerto, Tim Baumgärtner, Sewin Tariverdian, Hao Zhang, Kexin Wang, Hossain Shaikh Saadi, Leonardo F. R. Ribeiro, Iryna Gurevych

**Abstract**: Question Answering (QA) systems are increasingly deployed in applications where they support real-world decisions. However, state-of-the-art models rely on deep neural networks, which are difficult to interpret by humans. Inherently interpretable models or post hoc explainability methods can help users to comprehend how a model arrives at its prediction and, if successful, increase their trust in the system. Furthermore, researchers can leverage these insights to develop new methods that are more accurate and less biased. In this paper, we introduce SQuARE v2, the new version of SQuARE, to provide an explainability infrastructure for comparing models based on methods such as saliency maps and graph-based explanations. While saliency maps are useful to inspect the importance of each input token for the model's prediction, graph-based explanations from external Knowledge Graphs enable the users to verify the reasoning behind the model prediction. In addition, we provide multiple adversarial attacks to compare the robustness of QA models. With these explainability methods and adversarial attacks, we aim to ease the research on trustworthy QA models. SQuARE is available on https://square.ukp-lab.de.



## **48. Analyzing the Robustness of Decentralized Horizontal and Vertical Federated Learning Architectures in a Non-IID Scenario**

cs.LG

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.11061v1) [paper-pdf](http://arxiv.org/pdf/2210.11061v1)

**Authors**: Pedro Miguel Sánchez Sánchez, Alberto Huertas Celdrán, Enrique Tomás Martínez Beltrán, Daniel Demeter, Gérôme Bovet, Gregorio Martínez Pérez, Burkhard Stiller

**Abstract**: Federated learning (FL) allows participants to collaboratively train machine and deep learning models while protecting data privacy. However, the FL paradigm still presents drawbacks affecting its trustworthiness since malicious participants could launch adversarial attacks against the training process. Related work has studied the robustness of horizontal FL scenarios under different attacks. However, there is a lack of work evaluating the robustness of decentralized vertical FL and comparing it with horizontal FL architectures affected by adversarial attacks. Thus, this work proposes three decentralized FL architectures, one for horizontal and two for vertical scenarios, namely HoriChain, VertiChain, and VertiComb. These architectures present different neural networks and training protocols suitable for horizontal and vertical scenarios. Then, a decentralized, privacy-preserving, and federated use case with non-IID data to classify handwritten digits is deployed to evaluate the performance of the three architectures. Finally, a set of experiments computes and compares the robustness of the proposed architectures when they are affected by different data poisoning based on image watermarks and gradient poisoning adversarial attacks. The experiments show that even though particular configurations of both attacks can destroy the classification performance of the architectures, HoriChain is the most robust one.



## **49. Defending Person Detection Against Adversarial Patch Attack by using Universal Defensive Frame**

cs.CV

Accepted at IEEE Transactions on Image Processing (TIP), 2022

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2204.13004v2) [paper-pdf](http://arxiv.org/pdf/2204.13004v2)

**Authors**: Youngjoon Yu, Hong Joo Lee, Hakmin Lee, Yong Man Ro

**Abstract**: Person detection has attracted great attention in the computer vision area and is an imperative element in human-centric computer vision. Although the predictive performances of person detection networks have been improved dramatically, they are vulnerable to adversarial patch attacks. Changing the pixels in a restricted region can easily fool the person detection network in safety-critical applications such as autonomous driving and security systems. Despite the necessity of countering adversarial patch attacks, very few efforts have been dedicated to defending person detection against adversarial patch attack. In this paper, we propose a novel defense strategy that defends against an adversarial patch attack by optimizing a defensive frame for person detection. The defensive frame alleviates the effect of the adversarial patch while maintaining person detection performance with clean person. The proposed defensive frame in the person detection is generated with a competitive learning algorithm which makes an iterative competition between detection threatening module and detection shielding module in person detection. Comprehensive experimental results demonstrate that the proposed method effectively defends person detection against adversarial patch attacks.



## **50. Chaos Theory and Adversarial Robustness**

cs.LG

13 pages, 6 figures

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.13235v1) [paper-pdf](http://arxiv.org/pdf/2210.13235v1)

**Authors**: Jonathan S. Kent

**Abstract**: Neural Networks, being susceptible to adversarial attacks, should face a strict level of scrutiny before being deployed in critical or adversarial applications. This paper uses ideas from Chaos Theory to explain, analyze, and quantify the degree to which Neural Networks are susceptible to or robust against adversarial attacks. Our results show that susceptibility to attack grows significantly with the depth of the model, which has significant safety implications for the design of Neural Networks for production environments. We also demonstrate how to quickly and easily approximate the certified robustness radii for extremely large models, which until now has been computationally infeasible to calculate directly, as well as show a clear relationship between our new susceptibility metric and post-attack accuracy.



