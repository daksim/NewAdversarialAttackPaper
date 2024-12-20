# Latest Adversarial Attack Papers
**update at 2024-12-20 16:22:25**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. AutoTrust: Benchmarking Trustworthiness in Large Vision Language Models for Autonomous Driving**

cs.CV

55 pages, 14 figures

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.15206v1) [paper-pdf](http://arxiv.org/pdf/2412.15206v1)

**Authors**: Shuo Xing, Hongyuan Hua, Xiangbo Gao, Shenzhe Zhu, Renjie Li, Kexin Tian, Xiaopeng Li, Heng Huang, Tianbao Yang, Zhangyang Wang, Yang Zhou, Huaxiu Yao, Zhengzhong Tu

**Abstract**: Recent advancements in large vision language models (VLMs) tailored for autonomous driving (AD) have shown strong scene understanding and reasoning capabilities, making them undeniable candidates for end-to-end driving systems. However, limited work exists on studying the trustworthiness of DriveVLMs -- a critical factor that directly impacts public transportation safety. In this paper, we introduce AutoTrust, a comprehensive trustworthiness benchmark for large vision-language models in autonomous driving (DriveVLMs), considering diverse perspectives -- including trustfulness, safety, robustness, privacy, and fairness. We constructed the largest visual question-answering dataset for investigating trustworthiness issues in driving scenarios, comprising over 10k unique scenes and 18k queries. We evaluated six publicly available VLMs, spanning from generalist to specialist, from open-source to commercial models. Our exhaustive evaluations have unveiled previously undiscovered vulnerabilities of DriveVLMs to trustworthiness threats. Specifically, we found that the general VLMs like LLaVA-v1.6 and GPT-4o-mini surprisingly outperform specialized models fine-tuned for driving in terms of overall trustworthiness. DriveVLMs like DriveLM-Agent are particularly vulnerable to disclosing sensitive information. Additionally, both generalist and specialist VLMs remain susceptible to adversarial attacks and struggle to ensure unbiased decision-making across diverse environments and populations. Our findings call for immediate and decisive action to address the trustworthiness of DriveVLMs -- an issue of critical importance to public safety and the welfare of all citizens relying on autonomous transportation systems. Our benchmark is publicly available at \url{https://github.com/taco-group/AutoTrust}, and the leaderboard is released at \url{https://taco-group.github.io/AutoTrust/}.



## **2. Do Parameters Reveal More than Loss for Membership Inference?**

cs.LG

Accepted to Transactions on Machine Learning Research (TMLR)

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2406.11544v4) [paper-pdf](http://arxiv.org/pdf/2406.11544v4)

**Authors**: Anshuman Suri, Xiao Zhang, David Evans

**Abstract**: Membership inference attacks are used as a key tool for disclosure auditing. They aim to infer whether an individual record was used to train a model. While such evaluations are useful to demonstrate risk, they are computationally expensive and often make strong assumptions about potential adversaries' access to models and training environments, and thus do not provide tight bounds on leakage from potential attacks. We show how prior claims around black-box access being sufficient for optimal membership inference do not hold for stochastic gradient descent, and that optimal membership inference indeed requires white-box access. Our theoretical results lead to a new white-box inference attack, IHA (Inverse Hessian Attack), that explicitly uses model parameters by taking advantage of computing inverse-Hessian vector products. Our results show that both auditors and adversaries may be able to benefit from access to model parameters, and we advocate for further research into white-box methods for membership inference.



## **3. Accuracy Limits as a Barrier to Biometric System Security**

cs.CR

14 pages, 4 figures

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.13099v2) [paper-pdf](http://arxiv.org/pdf/2412.13099v2)

**Authors**: Axel Durbet, Paul-Marie Grollemund, Pascal Lafourcade, Kevin Thiry-Atighehchi

**Abstract**: Biometric systems are widely used for identity verification and identification, including authentication (i.e., one-to-one matching to verify a claimed identity) and identification (i.e., one-to-many matching to find a subject in a database). The matching process relies on measuring similarities or dissimilarities between a fresh biometric template and enrolled templates. The False Match Rate FMR is a key metric for assessing the accuracy and reliability of such systems. This paper analyzes biometric systems based on their FMR, with two main contributions. First, we explore untargeted attacks, where an adversary aims to impersonate any user within a database. We determine the number of trials required for an attacker to successfully impersonate a user and derive the critical population size (i.e., the maximum number of users in the database) required to maintain a given level of security. Furthermore, we compute the critical FMR value needed to ensure resistance against untargeted attacks as the database size increases. Second, we revisit the biometric birthday problem to evaluate the approximate and exact probabilities that two users in a database collide (i.e., can impersonate each other). Based on this analysis, we derive both the approximate critical population size and the critical FMR value needed to bound the likelihood of such collisions occurring with a given probability. These thresholds offer insights for designing systems that mitigate the risk of impersonation and collisions, particularly in large-scale biometric databases. Our findings indicate that current biometric systems fail to deliver sufficient accuracy to achieve an adequate security level against untargeted attacks, even in small-scale databases. Moreover, state-of-the-art systems face significant challenges in addressing the biometric birthday problem, especially as database sizes grow.



## **4. SLIFER: Investigating Performance and Robustness of Malware Detection Pipelines**

cs.CR

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2405.14478v3) [paper-pdf](http://arxiv.org/pdf/2405.14478v3)

**Authors**: Andrea Ponte, Dmitrijs Trizna, Luca Demetrio, Battista Biggio, Ivan Tesfai Ogbu, Fabio Roli

**Abstract**: As a result of decades of research, Windows malware detection is approached through a plethora of techniques. However, there is an ongoing mismatch between academia -- which pursues an optimal performances in terms of detection rate and low false alarms -- and the requirements of real-world scenarios. In particular, academia focuses on combining static and dynamic analysis within a single or ensemble of models, falling into several pitfalls like (i) firing dynamic analysis without considering the computational burden it requires; (ii) discarding impossible-to-analyze samples; and (iii) analyzing robustness against adversarial attacks without considering that malware detectors are complemented with more non-machine-learning components. Thus, in this paper we bridge these gaps, by investigating the properties of malware detectors built with multiple and different types of analysis. To do so, we develop SLIFER, a Windows malware detection pipeline sequentially leveraging both static and dynamic analysis, interrupting computations as soon as one module triggers an alarm, requiring dynamic analysis only when needed. Contrary to the state of the art, we investigate how to deal with samples that impede analyzes, showing how much they impact performances, concluding that it is better to flag them as legitimate to not drastically increase false alarms. Lastly, we perform a robustness evaluation of SLIFER. Counter-intuitively, the injection of new content is either blocked more by signatures than dynamic analysis, due to byte artifacts created by the attack, or it is able to avoid detection from signatures, as they rely on constraints on file size disrupted by attacks. As far as we know, we are the first to investigate the properties of sequential malware detectors, shedding light on their behavior in real production environment.



## **5. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2410.23091v5) [paper-pdf](http://arxiv.org/pdf/2410.23091v5)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark). The code is available at https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff



## **6. Grimm: A Plug-and-Play Perturbation Rectifier for Graph Neural Networks Defending against Poisoning Attacks**

cs.LG

19 pages, 13 figures

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.08555v2) [paper-pdf](http://arxiv.org/pdf/2412.08555v2)

**Authors**: Ao Liu, Wenshan Li, Beibei Li, Wengang Ma, Tao Li, Pan Zhou

**Abstract**: Recent studies have revealed the vulnerability of graph neural networks (GNNs) to adversarial poisoning attacks on node classification tasks. Current defensive methods require substituting the original GNNs with defense models, regardless of the original's type. This approach, while targeting adversarial robustness, compromises the enhancements developed in prior research to boost GNNs' practical performance. Here we introduce Grimm, the first plug-and-play defense model. With just a minimal interface requirement for extracting features from any layer of the protected GNNs, Grimm is thus enabled to seamlessly rectify perturbations. Specifically, we utilize the feature trajectories (FTs) generated by GNNs, as they evolve through epochs, to reflect the training status of the networks. We then theoretically prove that the FTs of victim nodes will inevitably exhibit discriminable anomalies. Consequently, inspired by the natural parallelism between the biological nervous and immune systems, we construct Grimm, a comprehensive artificial immune system for GNNs. Grimm not only detects abnormal FTs and rectifies adversarial edges during training but also operates efficiently in parallel, thereby mirroring the concurrent functionalities of its biological counterparts. We experimentally confirm that Grimm offers four empirically validated advantages: 1) Harmlessness, as it does not actively interfere with GNN training; 2) Parallelism, ensuring monitoring, detection, and rectification functions operate independently of the GNN training process; 3) Generalizability, demonstrating compatibility with mainstream GNNs such as GCN, GAT, and GraphSAGE; and 4) Transferability, as the detectors for abnormal FTs can be efficiently transferred across different systems for one-step rectification.



## **7. DG-Mamba: Robust and Efficient Dynamic Graph Structure Learning with Selective State Space Models**

cs.LG

Accepted by the Main Technical Track of the 39th Annual AAAI  Conference on Artificial Intelligence (AAAI-2025)

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.08160v4) [paper-pdf](http://arxiv.org/pdf/2412.08160v4)

**Authors**: Haonan Yuan, Qingyun Sun, Zhaonan Wang, Xingcheng Fu, Cheng Ji, Yongjian Wang, Bo Jin, Jianxin Li

**Abstract**: Dynamic graphs exhibit intertwined spatio-temporal evolutionary patterns, widely existing in the real world. Nevertheless, the structure incompleteness, noise, and redundancy result in poor robustness for Dynamic Graph Neural Networks (DGNNs). Dynamic Graph Structure Learning (DGSL) offers a promising way to optimize graph structures. However, aside from encountering unacceptable quadratic complexity, it overly relies on heuristic priors, making it hard to discover underlying predictive patterns. How to efficiently refine the dynamic structures, capture intrinsic dependencies, and learn robust representations, remains under-explored. In this work, we propose the novel DG-Mamba, a robust and efficient Dynamic Graph structure learning framework with the Selective State Space Models (Mamba). To accelerate the spatio-temporal structure learning, we propose a kernelized dynamic message-passing operator that reduces the quadratic time complexity to linear. To capture global intrinsic dynamics, we establish the dynamic graph as a self-contained system with State Space Model. By discretizing the system states with the cross-snapshot graph adjacency, we enable the long-distance dependencies capturing with the selective snapshot scan. To endow learned dynamic structures more expressive with informativeness, we propose the self-supervised Principle of Relevant Information for DGSL to regularize the most relevant yet least redundant information, enhancing global robustness. Extensive experiments demonstrate the superiority of the robustness and efficiency of our DG-Mamba compared with the state-of-the-art baselines against adversarial attacks.



## **8. How Does the Smoothness Approximation Method Facilitate Generalization for Federated Adversarial Learning?**

cs.LG

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.08282v2) [paper-pdf](http://arxiv.org/pdf/2412.08282v2)

**Authors**: Wenjun Ding, Ying An, Lixing Chen, Shichao Kan, Fan Wu, Zhe Qu

**Abstract**: Federated Adversarial Learning (FAL) is a robust framework for resisting adversarial attacks on federated learning. Although some FAL studies have developed efficient algorithms, they primarily focus on convergence performance and overlook generalization. Generalization is crucial for evaluating algorithm performance on unseen data. However, generalization analysis is more challenging due to non-smooth adversarial loss functions. A common approach to addressing this issue is to leverage smoothness approximation. In this paper, we develop algorithm stability measures to evaluate the generalization performance of two popular FAL algorithms: \textit{Vanilla FAL (VFAL)} and {\it Slack FAL (SFAL)}, using three different smooth approximation methods: 1) \textit{Surrogate Smoothness Approximation (SSA)}, (2) \textit{Randomized Smoothness Approximation (RSA)}, and (3) \textit{Over-Parameterized Smoothness Approximation (OPSA)}. Based on our in-depth analysis, we answer the question of how to properly set the smoothness approximation method to mitigate generalization error in FAL. Moreover, we identify RSA as the most effective method for reducing generalization error. In highly data-heterogeneous scenarios, we also recommend employing SFAL to mitigate the deterioration of generalization performance caused by heterogeneity. Based on our theoretical results, we provide insights to help develop more efficient FAL algorithms, such as designing new metrics and dynamic aggregation rules to mitigate heterogeneity.



## **9. Unleashing the Unseen: Harnessing Benign Datasets for Jailbreaking Large Language Models**

cs.CR

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2410.00451v3) [paper-pdf](http://arxiv.org/pdf/2410.00451v3)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Jun Sun

**Abstract**: Despite significant ongoing efforts in safety alignment, large language models (LLMs) such as GPT-4 and LLaMA 3 remain vulnerable to jailbreak attacks that can induce harmful behaviors, including through the use of adversarial suffixes. Building on prior research, we hypothesize that these adversarial suffixes are not mere bugs but may represent features that can dominate the LLM's behavior. To evaluate this hypothesis, we conduct several experiments. First, we demonstrate that benign features can be effectively made to function as adversarial suffixes, i.e., we develop a feature extraction method to extract sample-agnostic features from benign dataset in the form of suffixes and show that these suffixes may effectively compromise safety alignment. Second, we show that adversarial suffixes generated from jailbreak attacks may contain meaningful features, i.e., appending the same suffix to different prompts results in responses exhibiting specific characteristics. Third, we show that such benign-yet-safety-compromising features can be easily introduced through fine-tuning using only benign datasets. As a result, we are able to completely eliminate GPT's safety alignment in a blackbox setting through finetuning with only benign data. Our code and data is available at \url{https://github.com/suffix-maybe-feature/adver-suffix-maybe-features}.



## **10. Doubly-Universal Adversarial Perturbations: Deceiving Vision-Language Models Across Both Images and Text with a Single Perturbation**

cs.CV

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.08108v2) [paper-pdf](http://arxiv.org/pdf/2412.08108v2)

**Authors**: Hee-Seon Kim, Minbeom Kim, Changick Kim

**Abstract**: Large Vision-Language Models (VLMs) have demonstrated remarkable performance across multimodal tasks by integrating vision encoders with large language models (LLMs). However, these models remain vulnerable to adversarial attacks. Among such attacks, Universal Adversarial Perturbations (UAPs) are especially powerful, as a single optimized perturbation can mislead the model across various input images. In this work, we introduce a novel UAP specifically designed for VLMs: the Doubly-Universal Adversarial Perturbation (Doubly-UAP), capable of universally deceiving VLMs across both image and text inputs. To successfully disrupt the vision encoder's fundamental process, we analyze the core components of the attention mechanism. After identifying value vectors in the middle-to-late layers as the most vulnerable, we optimize Doubly-UAP in a label-free manner with a frozen model. Despite being developed as a black-box to the LLM, Doubly-UAP achieves high attack success rates on VLMs, consistently outperforming baseline methods across vision-language tasks. Extensive ablation studies and analyses further demonstrate the robustness of Doubly-UAP and provide insights into how it influences internal attention mechanisms.



## **11. Towards Provable Security in Industrial Control Systems Via Dynamic Protocol Attestation**

cs.CR

This paper was accepted into the ICSS'24 workshop

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.14467v1) [paper-pdf](http://arxiv.org/pdf/2412.14467v1)

**Authors**: Arthur Amorim, Trevor Kann, Max Taylor, Lance Joneckis

**Abstract**: Industrial control systems (ICSs) increasingly rely on digital technologies vulnerable to cyber attacks. Cyber attackers can infiltrate ICSs and execute malicious actions. Individually, each action seems innocuous. But taken together, they cause the system to enter an unsafe state. These attacks have resulted in dramatic consequences such as physical damage, economic loss, and environmental catastrophes. This paper introduces a methodology that restricts actions using protocols. These protocols only allow safe actions to execute. Protocols are written in a domain specific language we have embedded in an interactive theorem prover (ITP). The ITP enables formal, machine-checked proofs to ensure protocols maintain safety properties. We use dynamic attestation to ensure ICSs conform to their protocol even if an adversary compromises a component. Since protocol conformance prevents unsafe actions, the previously mentioned cyber attacks become impossible. We demonstrate the effectiveness of our methodology using an example from the Fischertechnik Industry 4.0 platform. We measure dynamic attestation's impact on latency and throughput. Our approach is a starting point for studying how to combine formal methods and protocol design to thwart attacks intended to cripple ICSs.



## **12. Adversarial Hubness in Multi-Modal Retrieval**

cs.CR

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.14113v1) [paper-pdf](http://arxiv.org/pdf/2412.14113v1)

**Authors**: Tingwei Zhang, Fnu Suya, Rishi Jha, Collin Zhang, Vitaly Shmatikov

**Abstract**: Hubness is a phenomenon in high-dimensional vector spaces where a single point from the natural distribution is unusually close to many other points. This is a well-known problem in information retrieval that causes some items to accidentally (and incorrectly) appear relevant to many queries. In this paper, we investigate how attackers can exploit hubness to turn any image or audio input in a multi-modal retrieval system into an adversarial hub. Adversarial hubs can be used to inject universal adversarial content (e.g., spam) that will be retrieved in response to thousands of different queries, as well as for targeted attacks on queries related to specific, attacker-chosen concepts. We present a method for creating adversarial hubs and evaluate the resulting hubs on benchmark multi-modal retrieval datasets and an image-to-image retrieval system based on a tutorial from Pinecone, a popular vector database. For example, in text-caption-to-image retrieval, a single adversarial hub is retrieved as the top-1 most relevant image for more than 21,000 out of 25,000 test queries (by contrast, the most common natural hub is the top-1 response to only 102 queries). We also investigate whether techniques for mitigating natural hubness are an effective defense against adversarial hubs, and show that they are not effective against hubs that target queries related to specific concepts.



## **13. Certification of Speaker Recognition Models to Additive Perturbations**

cs.SD

13 pages, 10 figures; AAAI-2025 accepted paper

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2404.18791v2) [paper-pdf](http://arxiv.org/pdf/2404.18791v2)

**Authors**: Dmitrii Korzh, Elvir Karimov, Mikhail Pautov, Oleg Y. Rogov, Ivan Oseledets

**Abstract**: Speaker recognition technology is applied to various tasks, from personal virtual assistants to secure access systems. However, the robustness of these systems against adversarial attacks, particularly to additive perturbations, remains a significant challenge. In this paper, we pioneer applying robustness certification techniques to speaker recognition, initially developed for the image domain. Our work covers this gap by transferring and improving randomized smoothing certification techniques against norm-bounded additive perturbations for classification and few-shot learning tasks to speaker recognition. We demonstrate the effectiveness of these methods on VoxCeleb 1 and 2 datasets for several models. We expect this work to improve the robustness of voice biometrics and accelerate the research of certification methods in the audio domain.



## **14. Adversarial Robustness of Link Sign Prediction in Signed Graphs**

cs.LG

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2401.10590v2) [paper-pdf](http://arxiv.org/pdf/2401.10590v2)

**Authors**: Jialong Zhou, Xing Ai, Yuni Lai, Tomasz Michalak, Gaolei Li, Jianhua Li, Kai Zhou

**Abstract**: Signed graphs serve as fundamental data structures for representing positive and negative relationships in social networks, with signed graph neural networks (SGNNs) emerging as the primary tool for their analysis. Our investigation reveals that balance theory, while essential for modeling signed relationships in SGNNs, inadvertently introduces exploitable vulnerabilities to black-box attacks. To demonstrate this vulnerability, we propose balance-attack, a novel adversarial strategy specifically designed to compromise graph balance degree, and develop an efficient heuristic algorithm to solve the associated NP-hard optimization problem. While existing approaches attempt to restore attacked graphs through balance learning techniques, they face a critical challenge we term "Irreversibility of Balance-related Information," where restored edges fail to align with original attack targets. To address this limitation, we introduce Balance Augmented-Signed Graph Contrastive Learning (BA-SGCL), an innovative framework that combines contrastive learning with balance augmentation techniques to achieve robust graph representations. By maintaining high balance degree in the latent space, BA-SGCL effectively circumvents the irreversibility challenge and enhances model resilience. Extensive experiments across multiple SGNN architectures and real-world datasets demonstrate both the effectiveness of our proposed balance-attack and the superior robustness of BA-SGCL, advancing the security and reliability of signed graph analysis in social networks. Datasets and codes of the proposed framework are at the github repository https://anonymous.4open.science/r/BA-SGCL-submit-DF41/.



## **15. A Review of the Duality of Adversarial Learning in Network Intrusion: Attacks and Countermeasures**

cs.CR

23 pages, 2 figures, 5 tables

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13880v1) [paper-pdf](http://arxiv.org/pdf/2412.13880v1)

**Authors**: Shalini Saini, Anitha Chennamaneni, Babatunde Sawyerr

**Abstract**: Deep learning solutions are instrumental in cybersecurity, harnessing their ability to analyze vast datasets, identify complex patterns, and detect anomalies. However, malevolent actors can exploit these capabilities to orchestrate sophisticated attacks, posing significant challenges to defenders and traditional security measures. Adversarial attacks, particularly those targeting vulnerabilities in deep learning models, present a nuanced and substantial threat to cybersecurity. Our study delves into adversarial learning threats such as Data Poisoning, Test Time Evasion, and Reverse Engineering, specifically impacting Network Intrusion Detection Systems. Our research explores the intricacies and countermeasures of attacks to deepen understanding of network security challenges amidst adversarial threats. In our study, we present insights into the dynamic realm of adversarial learning and its implications for network intrusion. The intersection of adversarial attacks and defenses within network traffic data, coupled with advances in machine learning and deep learning techniques, represents a relatively underexplored domain. Our research lays the groundwork for strengthening defense mechanisms to address the potential breaches in network security and privacy posed by adversarial attacks. Through our in-depth analysis, we identify domain-specific research gaps, such as the scarcity of real-life attack data and the evaluation of AI-based solutions for network traffic. Our focus on these challenges aims to stimulate future research efforts toward the development of resilient network defense strategies.



## **16. Cultivating Archipelago of Forests: Evolving Robust Decision Trees through Island Coevolution**

cs.LG

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13762v1) [paper-pdf](http://arxiv.org/pdf/2412.13762v1)

**Authors**: Adam Żychowski, Andrew Perrault, Jacek Mańdziuk

**Abstract**: Decision trees are widely used in machine learning due to their simplicity and interpretability, but they often lack robustness to adversarial attacks and data perturbations. The paper proposes a novel island-based coevolutionary algorithm (ICoEvoRDF) for constructing robust decision tree ensembles. The algorithm operates on multiple islands, each containing populations of decision trees and adversarial perturbations. The populations on each island evolve independently, with periodic migration of top-performing decision trees between islands. This approach fosters diversity and enhances the exploration of the solution space, leading to more robust and accurate decision tree ensembles. ICoEvoRDF utilizes a popular game theory concept of mixed Nash equilibrium for ensemble weighting, which further leads to improvement in results. ICoEvoRDF is evaluated on 20 benchmark datasets, demonstrating its superior performance compared to state-of-the-art methods in optimizing both adversarial accuracy and minimax regret. The flexibility of ICoEvoRDF allows for the integration of decision trees from various existing methods, providing a unified framework for combining diverse solutions. Our approach offers a promising direction for developing robust and interpretable machine learning models



## **17. A2RNet: Adversarial Attack Resilient Network for Robust Infrared and Visible Image Fusion**

cs.CV

9 pages, 8 figures, The 39th Annual AAAI Conference on Artificial  Intelligence

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.09954v2) [paper-pdf](http://arxiv.org/pdf/2412.09954v2)

**Authors**: Jiawei Li, Hongwei Yu, Jiansheng Chen, Xinlong Ding, Jinlong Wang, Jinyuan Liu, Bochao Zou, Huimin Ma

**Abstract**: Infrared and visible image fusion (IVIF) is a crucial technique for enhancing visual performance by integrating unique information from different modalities into one fused image. Exiting methods pay more attention to conducting fusion with undisturbed data, while overlooking the impact of deliberate interference on the effectiveness of fusion results. To investigate the robustness of fusion models, in this paper, we propose a novel adversarial attack resilient network, called $\textrm{A}^{\textrm{2}}$RNet. Specifically, we develop an adversarial paradigm with an anti-attack loss function to implement adversarial attacks and training. It is constructed based on the intrinsic nature of IVIF and provide a robust foundation for future research advancements. We adopt a Unet as the pipeline with a transformer-based defensive refinement module (DRM) under this paradigm, which guarantees fused image quality in a robust coarse-to-fine manner. Compared to previous works, our method mitigates the adverse effects of adversarial perturbations, consistently maintaining high-fidelity fusion results. Furthermore, the performance of downstream tasks can also be well maintained under adversarial attacks. Code is available at https://github.com/lok-18/A2RNet.



## **18. Physics-Based Adversarial Attack on Near-Infrared Human Detector for Nighttime Surveillance Camera Systems**

cs.CV

Appeared in ACM MM 2023

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13709v1) [paper-pdf](http://arxiv.org/pdf/2412.13709v1)

**Authors**: Muyao Niu, Zhuoxiao Li, Yifan Zhan, Huy H. Nguyen, Isao Echizen, Yinqiang Zheng

**Abstract**: Many surveillance cameras switch between daytime and nighttime modes based on illuminance levels. During the day, the camera records ordinary RGB images through an enabled IR-cut filter. At night, the filter is disabled to capture near-infrared (NIR) light emitted from NIR LEDs typically mounted around the lens. While RGB-based AI algorithm vulnerabilities have been widely reported, the vulnerabilities of NIR-based AI have rarely been investigated. In this paper, we identify fundamental vulnerabilities in NIR-based image understanding caused by color and texture loss due to the intrinsic characteristics of clothes' reflectance and cameras' spectral sensitivity in the NIR range. We further show that the nearly co-located configuration of illuminants and cameras in existing surveillance systems facilitates concealing and fully passive attacks in the physical world. Specifically, we demonstrate how retro-reflective and insulation plastic tapes can manipulate the intensity distribution of NIR images. We showcase an attack on the YOLO-based human detector using binary patterns designed in the digital space (via black-box query and searching) and then physically realized using tapes pasted onto clothes. Our attack highlights significant reliability concerns for nighttime surveillance systems, which are intended to enhance security. Codes Available: https://github.com/MyNiuuu/AdvNIR



## **19. Mitigating Adversarial Attacks in LLMs through Defensive Suffix Generation**

cs.CV

9 pages, 2 figures

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13705v1) [paper-pdf](http://arxiv.org/pdf/2412.13705v1)

**Authors**: Minkyoung Kim, Yunha Kim, Hyeram Seo, Heejung Choi, Jiye Han, Gaeun Kee, Soyoung Ko, HyoJe Jung, Byeolhee Kim, Young-Hak Kim, Sanghyun Park, Tae Joon Jun

**Abstract**: Large language models (LLMs) have exhibited outstanding performance in natural language processing tasks. However, these models remain susceptible to adversarial attacks in which slight input perturbations can lead to harmful or misleading outputs. A gradient-based defensive suffix generation algorithm is designed to bolster the robustness of LLMs. By appending carefully optimized defensive suffixes to input prompts, the algorithm mitigates adversarial influences while preserving the models' utility. To enhance adversarial understanding, a novel total loss function ($L_{\text{total}}$) combining defensive loss ($L_{\text{def}}$) and adversarial loss ($L_{\text{adv}}$) generates defensive suffixes more effectively. Experimental evaluations conducted on open-source LLMs such as Gemma-7B, mistral-7B, Llama2-7B, and Llama2-13B show that the proposed method reduces attack success rates (ASR) by an average of 11\% compared to models without defensive suffixes. Additionally, the perplexity score of Gemma-7B decreased from 6.57 to 3.93 when applying the defensive suffix generated by openELM-270M. Furthermore, TruthfulQA evaluations demonstrate consistent improvements with Truthfulness scores increasing by up to 10\% across tested configurations. This approach significantly enhances the security of LLMs in critical applications without requiring extensive retraining.



## **20. Enhancing Adversarial Transferability with Adversarial Weight Tuning**

cs.CR

Accepted by AAAI 2025

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2408.09469v3) [paper-pdf](http://arxiv.org/pdf/2408.09469v3)

**Authors**: Jiahao Chen, Zhou Feng, Rui Zeng, Yuwen Pu, Chunyi Zhou, Yi Jiang, Yuyou Gan, Jinbao Li, Shouling Ji

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial examples (AEs) that mislead the model while appearing benign to human observers. A critical concern is the transferability of AEs, which enables black-box attacks without direct access to the target model. However, many previous attacks have failed to explain the intrinsic mechanism of adversarial transferability. In this paper, we rethink the property of transferable AEs and reformalize the formulation of transferability. Building on insights from this mechanism, we analyze the generalization of AEs across models with different architectures and prove that we can find a local perturbation to mitigate the gap between surrogate and target models. We further establish the inner connections between model smoothness and flat local maxima, both of which contribute to the transferability of AEs. Further, we propose a new adversarial attack algorithm, \textbf{A}dversarial \textbf{W}eight \textbf{T}uning (AWT), which adaptively adjusts the parameters of the surrogate model using generated AEs to optimize the flat local maxima and model smoothness simultaneously, without the need for extra data. AWT is a data-free tuning method that combines gradient-based and model-based attack methods to enhance the transferability of AEs. Extensive experiments on a variety of models with different architectures on ImageNet demonstrate that AWT yields superior performance over other attacks, with an average increase of nearly 5\% and 10\% attack success rates on CNN-based and Transformer-based models, respectively, compared to state-of-the-art attacks.



## **21. Understanding Key Point Cloud Features for Development Three-dimensional Adversarial Attacks**

cs.CV

10 pages, 6 figures

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2210.14164v4) [paper-pdf](http://arxiv.org/pdf/2210.14164v4)

**Authors**: Hanieh Naderi, Chinthaka Dinesh, Ivan V. Bajic, Shohreh Kasaei

**Abstract**: Adversarial attacks pose serious challenges for deep neural network (DNN)-based analysis of various input signals. In the case of three-dimensional point clouds, methods have been developed to identify points that play a key role in network decision, and these become crucial in generating existing adversarial attacks. For example, a saliency map approach is a popular method for identifying adversarial drop points, whose removal would significantly impact the network decision. This paper seeks to enhance the understanding of three-dimensional adversarial attacks by exploring which point cloud features are most important for predicting adversarial points. Specifically, Fourteen key point cloud features such as edge intensity and distance from the centroid are defined, and multiple linear regression is employed to assess their predictive power for adversarial points. Based on critical feature selection insights, a new attack method has been developed to evaluate whether the selected features can generate an attack successfully. Unlike traditional attack methods that rely on model-specific vulnerabilities, this approach focuses on the intrinsic characteristics of the point clouds themselves. It is demonstrated that these features can predict adversarial points across four different DNN architectures, Point Network (PointNet), PointNet++, Dynamic Graph Convolutional Neural Networks (DGCNN), and Point Convolutional Network (PointConv) outperforming random guessing and achieving results comparable to saliency map-based attacks. This study has important engineering applications, such as enhancing the security and robustness of three-dimensional point cloud-based systems in fields like robotics and autonomous driving.



## **22. Novel AI Camera Camouflage: Face Cloaking Without Full Disguise**

cs.CV

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13507v1) [paper-pdf](http://arxiv.org/pdf/2412.13507v1)

**Authors**: David Noever, Forrest McKee

**Abstract**: This study demonstrates a novel approach to facial camouflage that combines targeted cosmetic perturbations and alpha transparency layer manipulation to evade modern facial recognition systems. Unlike previous methods -- such as CV dazzle, adversarial patches, and theatrical disguises -- this work achieves effective obfuscation through subtle modifications to key-point regions, particularly the brow, nose bridge, and jawline. Empirical testing with Haar cascade classifiers and commercial systems like BetaFaceAPI and Microsoft Bing Visual Search reveals that vertical perturbations near dense facial key points significantly disrupt detection without relying on overt disguises. Additionally, leveraging alpha transparency attacks in PNG images creates a dual-layer effect: faces remain visible to human observers but disappear in machine-readable RGB layers, rendering them unidentifiable during reverse image searches. The results highlight the potential for creating scalable, low-visibility facial obfuscation strategies that balance effectiveness and subtlety, opening pathways for defeating surveillance while maintaining plausible anonymity.



## **23. Safeguarding Virtual Healthcare: A Novel Attacker-Centric Model for Data Security and Privacy**

cs.CR

6 pages, 3 figures, 3 tables

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13440v1) [paper-pdf](http://arxiv.org/pdf/2412.13440v1)

**Authors**: Suvineetha Herath, Haywood Gelman, John Hastings, Yong Wang

**Abstract**: The rapid growth of remote healthcare delivery has introduced significant security and privacy risks to protected health information (PHI). Analysis of a comprehensive healthcare security breach dataset covering 2009-2023 reveals their significant prevalence and impact. This study investigates the root causes of such security incidents and introduces the Attacker-Centric Approach (ACA), a novel threat model tailored to protect PHI. ACA addresses limitations in existing threat models and regulatory frameworks by adopting a holistic attacker-focused perspective, examining threats from the viewpoint of cyber adversaries, their motivations, tactics, and potential attack vectors. Leveraging established risk management frameworks, ACA provides a multi-layered approach to threat identification, risk assessment, and proactive mitigation strategies. A comprehensive threat library classifies physical, third-party, external, and internal threats. ACA's iterative nature and feedback mechanisms enable continuous adaptation to emerging threats, ensuring sustained effectiveness. ACA allows healthcare providers to proactively identify and mitigate vulnerabilities, fostering trust and supporting the secure adoption of virtual care technologies.



## **24. Safeguarding System Prompts for LLMs**

cs.CR

20 pages, 7 figures, 6 tables

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13426v1) [paper-pdf](http://arxiv.org/pdf/2412.13426v1)

**Authors**: Zhifeng Jiang, Zhihua Jin, Guoliang He

**Abstract**: Large language models (LLMs) are increasingly utilized in applications where system prompts, which guide model outputs, play a crucial role. These prompts often contain business logic and sensitive information, making their protection essential. However, adversarial and even regular user queries can exploit LLM vulnerabilities to expose these hidden prompts. To address this issue, we present PromptKeeper, a novel defense mechanism for system prompt privacy. By reliably detecting worst-case leakage and regenerating outputs without the system prompt when necessary, PromptKeeper ensures robust protection against prompt extraction attacks via either adversarial or regular queries, while preserving conversational capability and runtime efficiency during benign user interactions.



## **25. Targeted View-Invariant Adversarial Perturbations for 3D Object Recognition**

cs.CV

Accepted to AAAI-25 Workshop on Artificial Intelligence for Cyber  Security (AICS): http://aics.site/AICS2025/index.html

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.13376v1) [paper-pdf](http://arxiv.org/pdf/2412.13376v1)

**Authors**: Christian Green, Mehmet Ergezer, Abdurrahman Zeybey

**Abstract**: Adversarial attacks pose significant challenges in 3D object recognition, especially in scenarios involving multi-view analysis where objects can be observed from varying angles. This paper introduces View-Invariant Adversarial Perturbations (VIAP), a novel method for crafting robust adversarial examples that remain effective across multiple viewpoints. Unlike traditional methods, VIAP enables targeted attacks capable of manipulating recognition systems to classify objects as specific, pre-determined labels, all while using a single universal perturbation. Leveraging a dataset of 1,210 images across 121 diverse rendered 3D objects, we demonstrate the effectiveness of VIAP in both targeted and untargeted settings. Our untargeted perturbations successfully generate a singular adversarial noise robust to 3D transformations, while targeted attacks achieve exceptional results, with top-1 accuracies exceeding 95% across various epsilon values. These findings highlight VIAPs potential for real-world applications, such as testing the robustness of 3D recognition systems. The proposed method sets a new benchmark for view-invariant adversarial robustness, advancing the field of adversarial machine learning for 3D object recognition.



## **26. Class-RAG: Real-Time Content Moderation with Retrieval Augmented Generation**

cs.AI

11 pages, submit to ACL

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2410.14881v2) [paper-pdf](http://arxiv.org/pdf/2410.14881v2)

**Authors**: Jianfa Chen, Emily Shen, Trupti Bavalatti, Xiaowen Lin, Yongkai Wang, Shuming Hu, Harihar Subramanyam, Ksheeraj Sai Vepuri, Ming Jiang, Ji Qi, Li Chen, Nan Jiang, Ankit Jain

**Abstract**: Robust content moderation classifiers are essential for the safety of Generative AI systems. In this task, differences between safe and unsafe inputs are often extremely subtle, making it difficult for classifiers (and indeed, even humans) to properly distinguish violating vs. benign samples without context or explanation. Scaling risk discovery and mitigation through continuous model fine-tuning is also slow, challenging and costly, preventing developers from being able to respond quickly and effectively to emergent harms. We propose a Classification approach employing Retrieval-Augmented Generation (Class-RAG). Class-RAG extends the capability of its base LLM through access to a retrieval library which can be dynamically updated to enable semantic hotfixing for immediate, flexible risk mitigation. Compared to model fine-tuning, Class-RAG demonstrates flexibility and transparency in decision-making, outperforms on classification and is more robust against adversarial attack, as evidenced by empirical studies. Our findings also suggest that Class-RAG performance scales with retrieval library size, indicating that increasing the library size is a viable and low-cost approach to improve content moderation.



## **27. Concept-ROT: Poisoning Concepts in Large Language Models with Model Editing**

cs.LG

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.13341v1) [paper-pdf](http://arxiv.org/pdf/2412.13341v1)

**Authors**: Keltin Grimes, Marco Christiani, David Shriver, Marissa Connor

**Abstract**: Model editing methods modify specific behaviors of Large Language Models by altering a small, targeted set of network weights and require very little data and compute. These methods can be used for malicious applications such as inserting misinformation or simple trojans that result in adversary-specified behaviors when a trigger word is present. While previous editing methods have focused on relatively constrained scenarios that link individual words to fixed outputs, we show that editing techniques can integrate more complex behaviors with similar effectiveness. We develop Concept-ROT, a model editing-based method that efficiently inserts trojans which not only exhibit complex output behaviors, but also trigger on high-level concepts -- presenting an entirely new class of trojan attacks. Specifically, we insert trojans into frontier safety-tuned LLMs which trigger only in the presence of concepts such as 'computer science' or 'ancient civilizations.' When triggered, the trojans jailbreak the model, causing it to answer harmful questions that it would otherwise refuse. Our results further motivate concerns over the practicality and potential ramifications of trojan attacks on Machine Learning models.



## **28. LLM Whisperer: An Inconspicuous Attack to Bias LLM Responses**

cs.CR

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2406.04755v3) [paper-pdf](http://arxiv.org/pdf/2406.04755v3)

**Authors**: Weiran Lin, Anna Gerchanovsky, Omer Akgul, Lujo Bauer, Matt Fredrikson, Zifan Wang

**Abstract**: Writing effective prompts for large language models (LLM) can be unintuitive and burdensome. In response, services that optimize or suggest prompts have emerged. While such services can reduce user effort, they also introduce a risk: the prompt provider can subtly manipulate prompts to produce heavily biased LLM responses. In this work, we show that subtle synonym replacements in prompts can increase the likelihood (by a difference up to 78%) that LLMs mention a target concept (e.g., a brand, political party, nation). We substantiate our observations through a user study, showing our adversarially perturbed prompts 1) are indistinguishable from unaltered prompts by humans, 2) push LLMs to recommend target concepts more often, and 3) make users more likely to notice target concepts, all without arousing suspicion. The practicality of this attack has the potential to undermine user autonomy. Among other measures, we recommend implementing warnings against using prompts from untrusted parties.



## **29. A New Adversarial Perspective for LiDAR-based 3D Object Detection**

cs.CV

11 pages, 7 figures, AAAI2025

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.13017v1) [paper-pdf](http://arxiv.org/pdf/2412.13017v1)

**Authors**: Shijun Zheng, Weiquan Liu, Yu Guo, Yu Zang, Siqi Shen, Cheng Wang

**Abstract**: Autonomous vehicles (AVs) rely on LiDAR sensors for environmental perception and decision-making in driving scenarios. However, ensuring the safety and reliability of AVs in complex environments remains a pressing challenge. To address this issue, we introduce a real-world dataset (ROLiD) comprising LiDAR-scanned point clouds of two random objects: water mist and smoke. In this paper, we introduce a novel adversarial perspective by proposing an attack framework that utilizes water mist and smoke to simulate environmental interference. Specifically, we propose a point cloud sequence generation method using a motion and content decomposition generative adversarial network named PCS-GAN to simulate the distribution of random objects. Furthermore, leveraging the simulated LiDAR scanning characteristics implemented with Range Image, we examine the effects of introducing random object perturbations at various positions on the target vehicle. Extensive experiments demonstrate that adversarial perturbations based on random objects effectively deceive vehicle detection and reduce the recognition rate of 3D object detection models.



## **30. AnyAttack: Targeted Adversarial Attacks on Vision-Language Models toward Any Images**

cs.LG

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2410.05346v2) [paper-pdf](http://arxiv.org/pdf/2410.05346v2)

**Authors**: Jiaming Zhang, Junhong Ye, Xingjun Ma, Yige Li, Yunfan Yang, Jitao Sang, Dit-Yan Yeung

**Abstract**: Due to their multimodal capabilities, Vision-Language Models (VLMs) have found numerous impactful applications in real-world scenarios. However, recent studies have revealed that VLMs are vulnerable to image-based adversarial attacks, particularly targeted adversarial images that manipulate the model to generate harmful content specified by the adversary. Current attack methods rely on predefined target labels to create targeted adversarial attacks, which limits their scalability and applicability for large-scale robustness evaluations. In this paper, we propose AnyAttack, a self-supervised framework that generates targeted adversarial images for VLMs without label supervision, allowing any image to serve as a target for the attack. Our framework employs the pre-training and fine-tuning paradigm, with the adversarial noise generator pre-trained on the large-scale LAION-400M dataset. This large-scale pre-training endows our method with powerful transferability across a wide range of VLMs. Extensive experiments on five mainstream open-source VLMs (CLIP, BLIP, BLIP2, InstructBLIP, and MiniGPT-4) across three multimodal tasks (image-text retrieval, multimodal classification, and image captioning) demonstrate the effectiveness of our attack. Additionally, we successfully transfer AnyAttack to multiple commercial VLMs, including Google Gemini, Claude Sonnet, Microsoft Copilot and OpenAI GPT. These results reveal an unprecedented risk to VLMs, highlighting the need for effective countermeasures.



## **31. Adaptive Epsilon Adversarial Training for Robust Gravitational Wave Parameter Estimation Using Normalizing Flows**

cs.LG

Due to new experimental results to add to the paper, this version no  longer accurately reflects the current state of our research. Therefore, we  are withdrawing the paper while further experiments are conducted. We will  submit a new version in the future. We apologize for any inconvenience this  may cause

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.07559v2) [paper-pdf](http://arxiv.org/pdf/2412.07559v2)

**Authors**: Yiqian Yang, Xihua Zhu, Fan Zhang

**Abstract**: Adversarial training with Normalizing Flow (NF) models is an emerging research area aimed at improving model robustness through adversarial samples. In this study, we focus on applying adversarial training to NF models for gravitational wave parameter estimation. We propose an adaptive epsilon method for Fast Gradient Sign Method (FGSM) adversarial training, which dynamically adjusts perturbation strengths based on gradient magnitudes using logarithmic scaling. Our hybrid architecture, combining ResNet and Inverse Autoregressive Flow, reduces the Negative Log Likelihood (NLL) loss by 47\% under FGSM attacks compared to the baseline model, while maintaining an NLL of 4.2 on clean data (only 5\% higher than the baseline). For perturbation strengths between 0.01 and 0.1, our model achieves an average NLL of 5.8, outperforming both fixed-epsilon (NLL: 6.7) and progressive-epsilon (NLL: 7.2) methods. Under stronger Projected Gradient Descent attacks with perturbation strength of 0.05, our model maintains an NLL of 6.4, demonstrating superior robustness while avoiding catastrophic overfitting.



## **32. PROSAC: Provably Safe Certification for Machine Learning Models under Adversarial Attacks**

cs.LG

Accepted to AAAI2025

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2402.02629v2) [paper-pdf](http://arxiv.org/pdf/2402.02629v2)

**Authors**: Chen Feng, Ziquan Liu, Zhuo Zhi, Ilija Bogunovic, Carsten Gerner-Beuerle, Miguel Rodrigues

**Abstract**: It is widely known that state-of-the-art machine learning models, including vision and language models, can be seriously compromised by adversarial perturbations. It is therefore increasingly relevant to develop capabilities to certify their performance in the presence of the most effective adversarial attacks. Our paper offers a new approach to certify the performance of machine learning models in the presence of adversarial attacks with population level risk guarantees. In particular, we introduce the notion of $(\alpha,\zeta)$-safe machine learning model. We propose a hypothesis testing procedure, based on the availability of a calibration set, to derive statistical guarantees providing that the probability of declaring that the adversarial (population) risk of a machine learning model is less than $\alpha$ (i.e. the model is safe), while the model is in fact unsafe (i.e. the model adversarial population risk is higher than $\alpha$), is less than $\zeta$. We also propose Bayesian optimization algorithms to determine efficiently whether a machine learning model is $(\alpha,\zeta)$-safe in the presence of an adversarial attack, along with statistical guarantees. We apply our framework to a range of machine learning models - including various sizes of vision Transformer (ViT) and ResNet models - impaired by a variety of adversarial attacks, such as PGDAttack, MomentumAttack, GenAttack and BanditAttack, to illustrate the operation of our approach. Importantly, we show that ViT's are generally more robust to adversarial attacks than ResNets, and large models are generally more robust than smaller models. Our approach goes beyond existing empirical adversarial risk-based certification guarantees. It formulates rigorous (and provable) performance guarantees that can be used to satisfy regulatory requirements mandating the use of state-of-the-art technical tools.



## **33. Deep Learning for Resilient Adversarial Decision Fusion in Byzantine Networks**

cs.LG

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12739v1) [paper-pdf](http://arxiv.org/pdf/2412.12739v1)

**Authors**: Kassem Kallas

**Abstract**: This paper introduces a deep learning-based framework for resilient decision fusion in adversarial multi-sensor networks, providing a unified mathematical setup that encompasses diverse scenarios, including varying Byzantine node proportions, synchronized and unsynchronized attacks, unbalanced priors, adaptive strategies, and Markovian states. Unlike traditional methods, which depend on explicit parameter tuning and are limited by scenario-specific assumptions, the proposed approach employs a deep neural network trained on a globally constructed dataset to generalize across all cases without requiring adaptation. Extensive simulations validate the method's robustness, achieving superior accuracy, minimal error probability, and scalability compared to state-of-the-art techniques, while ensuring computational efficiency for real-time applications. This unified framework demonstrates the potential of deep learning to revolutionize decision fusion by addressing the challenges posed by Byzantine nodes in dynamic adversarial environments.



## **34. On the Impact of Hard Adversarial Instances on Overfitting in Adversarial Training**

cs.LG

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2112.07324v2) [paper-pdf](http://arxiv.org/pdf/2112.07324v2)

**Authors**: Chen Liu, Zhichao Huang, Mathieu Salzmann, Tong Zhang, Sabine Süsstrunk

**Abstract**: Adversarial training is a popular method to robustify models against adversarial attacks. However, it exhibits much more severe overfitting than training on clean inputs. In this work, we investigate this phenomenon from the perspective of training instances, i.e., training input-target pairs. Based on a quantitative metric measuring the relative difficulty of an instance in the training set, we analyze the model's behavior on training instances of different difficulty levels. This lets us demonstrate that the decay in generalization performance of adversarial training is a result of fitting hard adversarial instances. We theoretically verify our observations for both linear and general nonlinear models, proving that models trained on hard instances have worse generalization performance than ones trained on easy instances, and that this generalization gap increases with the size of the adversarial budget. Finally, we investigate solutions to mitigate adversarial overfitting in several scenarios, including fast adversarial training and fine-tuning a pretrained model with additional data. Our results demonstrate that using training data adaptively improves the model's robustness.



## **35. Building Gradient Bridges: Label Leakage from Restricted Gradient Sharing in Federated Learning**

cs.LG

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12640v1) [paper-pdf](http://arxiv.org/pdf/2412.12640v1)

**Authors**: Rui Zhang, Ka-Ho Chow, Ping Li

**Abstract**: The growing concern over data privacy, the benefits of utilizing data from diverse sources for model training, and the proliferation of networked devices with enhanced computational capabilities have all contributed to the rise of federated learning (FL). The clients in FL collaborate to train a global model by uploading gradients computed on their private datasets without collecting raw data. However, a new attack surface has emerged from gradient sharing, where adversaries can restore the label distribution of a victim's private data by analyzing the obtained gradients. To mitigate this privacy leakage, existing lightweight defenses restrict the sharing of gradients, such as encrypting the final-layer gradients or locally updating the parameters within. In this paper, we introduce a novel attack called Gradient Bridge (GDBR) that recovers the label distribution of training data from the limited gradient information shared in FL. GDBR explores the relationship between the layer-wise gradients, tracks the flow of gradients, and analytically derives the batch training labels. Extensive experiments show that GDBR can accurately recover more than 80% of labels in various FL settings. GDBR highlights the inadequacy of restricted gradient sharing-based defenses and calls for the design of effective defense schemes in FL.



## **36. Improving the Transferability of 3D Point Cloud Attack via Spectral-aware Admix and Optimization Designs**

cs.CV

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12626v1) [paper-pdf](http://arxiv.org/pdf/2412.12626v1)

**Authors**: Shiyu Hu, Daizong Liu, Wei Hu

**Abstract**: Deep learning models for point clouds have shown to be vulnerable to adversarial attacks, which have received increasing attention in various safety-critical applications such as autonomous driving, robotics, and surveillance. Existing 3D attackers generally design various attack strategies in the white-box setting, requiring the prior knowledge of 3D model details. However, real-world 3D applications are in the black-box setting, where we can only acquire the outputs of the target classifier. Although few recent works try to explore the black-box attack, they still achieve limited attack success rates (ASR). To alleviate this issue, this paper focuses on attacking the 3D models in a transfer-based black-box setting, where we first carefully design adversarial examples in a white-box surrogate model and then transfer them to attack other black-box victim models. Specifically, we propose a novel Spectral-aware Admix with Augmented Optimization method (SAAO) to improve the adversarial transferability. In particular, since traditional Admix strategy are deployed in the 2D domain that adds pixel-wise images for perturbing, we can not directly follow it to merge point clouds in coordinate domain as it will destroy the geometric shapes. Therefore, we design spectral-aware fusion that performs Graph Fourier Transform (GFT) to get spectral features of the point clouds and add them in the spectral domain. Afterward, we run a few steps with spectral-aware weighted Admix to select better optimization paths as well as to adjust corresponding learning weights. At last, we run more steps to generate adversarial spectral feature along the optimization path and perform Inverse-GFT on the adversarial spectral feature to obtain the adversarial example in the data domain. Experiments show that our SAAO achieves better transferability compared to existing 3D attack methods.



## **37. Jailbreaking? One Step Is Enough!**

cs.CL

17 pages

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12621v1) [paper-pdf](http://arxiv.org/pdf/2412.12621v1)

**Authors**: Weixiong Zheng, Peijian Zeng, Yiwei Li, Hongyan Wu, Nankai Lin, Junhao Chen, Aimin Yang, Yongmei Zhou

**Abstract**: Large language models (LLMs) excel in various tasks but remain vulnerable to jailbreak attacks, where adversaries manipulate prompts to generate harmful outputs. Examining jailbreak prompts helps uncover the shortcomings of LLMs. However, current jailbreak methods and the target model's defenses are engaged in an independent and adversarial process, resulting in the need for frequent attack iterations and redesigning attacks for different models. To address these gaps, we propose a Reverse Embedded Defense Attack (REDA) mechanism that disguises the attack intention as the "defense". intention against harmful content. Specifically, REDA starts from the target response, guiding the model to embed harmful content within its defensive measures, thereby relegating harmful content to a secondary role and making the model believe it is performing a defensive task. The attacking model considers that it is guiding the target model to deal with harmful content, while the target model thinks it is performing a defensive task, creating an illusion of cooperation between the two. Additionally, to enhance the model's confidence and guidance in "defensive" intentions, we adopt in-context learning (ICL) with a small number of attack examples and construct a corresponding dataset of attack examples. Extensive evaluations demonstrate that the REDA method enables cross-model attacks without the need to redesign attack strategies for different models, enables successful jailbreak in one iteration, and outperforms existing methods on both open-source and closed-source models.



## **38. WaterPark: A Robustness Assessment of Language Model Watermarking**

cs.CR

22 pages

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2411.13425v2) [paper-pdf](http://arxiv.org/pdf/2411.13425v2)

**Authors**: Jiacheng Liang, Zian Wang, Lauren Hong, Shouling Ji, Ting Wang

**Abstract**: Various watermarking methods (``watermarkers'') have been proposed to identify LLM-generated texts; yet, due to the lack of unified evaluation platforms, many critical questions remain under-explored: i) What are the strengths/limitations of various watermarkers, especially their attack robustness? ii) How do various design choices impact their robustness? iii) How to optimally operate watermarkers in adversarial environments? To fill this gap, we systematize existing LLM watermarkers and watermark removal attacks, mapping out their design spaces. We then develop WaterPark, a unified platform that integrates 10 state-of-the-art watermarkers and 12 representative attacks. More importantly, by leveraging WaterPark, we conduct a comprehensive assessment of existing watermarkers, unveiling the impact of various design choices on their attack robustness. We further explore the best practices to operate watermarkers in adversarial environments. We believe our study sheds light on current LLM watermarking techniques while WaterPark serves as a valuable testbed to facilitate future research.



## **39. Attack On Prompt: Backdoor Attack in Prompt-Based Continual Learning**

cs.LG

Accepted to AAAI 2025

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2406.19753v2) [paper-pdf](http://arxiv.org/pdf/2406.19753v2)

**Authors**: Trang Nguyen, Anh Tran, Nhat Ho

**Abstract**: Prompt-based approaches offer a cutting-edge solution to data privacy issues in continual learning, particularly in scenarios involving multiple data suppliers where long-term storage of private user data is prohibited. Despite delivering state-of-the-art performance, its impressive remembering capability can become a double-edged sword, raising security concerns as it might inadvertently retain poisoned knowledge injected during learning from private user data. Following this insight, in this paper, we expose continual learning to a potential threat: backdoor attack, which drives the model to follow a desired adversarial target whenever a specific trigger is present while still performing normally on clean samples. We highlight three critical challenges in executing backdoor attacks on incremental learners and propose corresponding solutions: (1) \emph{Transferability}: We employ a surrogate dataset and manipulate prompt selection to transfer backdoor knowledge to data from other suppliers; (2) \emph{Resiliency}: We simulate static and dynamic states of the victim to ensure the backdoor trigger remains robust during intense incremental learning processes; and (3) \emph{Authenticity}: We apply binary cross-entropy loss as an anti-cheating factor to prevent the backdoor trigger from devolving into adversarial noise. Extensive experiments across various benchmark datasets and continual learners validate our continual backdoor framework, achieving up to $100\%$ attack success rate, with further ablation studies confirming our contributions' effectiveness.



## **40. Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?**

cs.LG

accepted by KDD2025

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2408.08685v2) [paper-pdf](http://arxiv.org/pdf/2408.08685v2)

**Authors**: Zhongjian Zhang, Xiao Wang, Huichi Zhou, Yue Yu, Mengmei Zhang, Cheng Yang, Chuan Shi

**Abstract**: Graph neural networks (GNNs) are vulnerable to adversarial attacks, especially for topology perturbations, and many methods that improve the robustness of GNNs have received considerable attention. Recently, we have witnessed the significant success of large language models (LLMs), leading many to explore the great potential of LLMs on GNNs. However, they mainly focus on improving the performance of GNNs by utilizing LLMs to enhance the node features. Therefore, we ask: Will the robustness of GNNs also be enhanced with the powerful understanding and inference capabilities of LLMs? By presenting the empirical results, we find that despite that LLMs can improve the robustness of GNNs, there is still an average decrease of 23.1% in accuracy, implying that the GNNs remain extremely vulnerable against topology attacks. Therefore, another question is how to extend the capabilities of LLMs on graph adversarial robustness. In this paper, we propose an LLM-based robust graph structure inference framework, LLM4RGNN, which distills the inference capabilities of GPT-4 into a local LLM for identifying malicious edges and an LM-based edge predictor for finding missing important edges, so as to recover a robust graph structure. Extensive experiments demonstrate that LLM4RGNN consistently improves the robustness across various GNNs. Even in some cases where the perturbation ratio increases to 40%, the accuracy of GNNs is still better than that on the clean graph. The source code can be found in https://github.com/zhongjian-zhang/LLM4RGNN.



## **41. Human-in-the-Loop Generation of Adversarial Texts: A Case Study on Tibetan Script**

cs.CL

Review Version; Submitted to NAACL 2025 Demo Track

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12478v1) [paper-pdf](http://arxiv.org/pdf/2412.12478v1)

**Authors**: Xi Cao, Yuan Sun, Jiajun Li, Quzong Gesang, Nuo Qun, Tashi Nyima

**Abstract**: DNN-based language models perform excellently on various tasks, but even SOTA LLMs are susceptible to textual adversarial attacks. Adversarial texts play crucial roles in multiple subfields of NLP. However, current research has the following issues. (1) Most textual adversarial attack methods target rich-resourced languages. How do we generate adversarial texts for less-studied languages? (2) Most textual adversarial attack methods are prone to generating invalid or ambiguous adversarial texts. How do we construct high-quality adversarial robustness benchmarks? (3) New language models may be immune to part of previously generated adversarial texts. How do we update adversarial robustness benchmarks? To address the above issues, we introduce HITL-GAT, a system based on a general approach to human-in-the-loop generation of adversarial texts. HITL-GAT contains four stages in one pipeline: victim model construction, adversarial example generation, high-quality benchmark construction, and adversarial robustness evaluation. Additionally, we utilize HITL-GAT to make a case study on Tibetan script which can be a reference for the adversarial research of other less-studied languages.



## **42. Architectural Patterns for Designing Quantum Artificial Intelligence Systems**

cs.SE

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2411.10487v3) [paper-pdf](http://arxiv.org/pdf/2411.10487v3)

**Authors**: Mykhailo Klymenko, Thong Hoang, Xiwei Xu, Zhenchang Xing, Muhammad Usman, Qinghua Lu, Liming Zhu

**Abstract**: Utilising quantum computing technology to enhance artificial intelligence systems is expected to improve training and inference times, increase robustness against noise and adversarial attacks, and reduce the number of parameters without compromising accuracy. However, moving beyond proof-of-concept or simulations to develop practical applications of these systems while ensuring high software quality faces significant challenges due to the limitations of quantum hardware and the underdeveloped knowledge base in software engineering for such systems. In this work, we have conducted a systematic mapping study to identify the challenges and solutions associated with the software architecture of quantum-enhanced artificial intelligence systems. The results of the systematic mapping study reveal several architectural patterns that describe how quantum components can be integrated into inference engines, as well as middleware patterns that facilitate communication between classical and quantum components. Each pattern realises a trade-off between various software quality attributes, such as efficiency, scalability, trainability, simplicity, portability, and deployability. The outcomes of this work have been compiled into a catalogue of architectural patterns.



## **43. Adversarially robust generalization theory via Jacobian regularization for deep neural networks**

stat.ML

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12449v1) [paper-pdf](http://arxiv.org/pdf/2412.12449v1)

**Authors**: Dongya Wu, Xin Li

**Abstract**: Powerful deep neural networks are vulnerable to adversarial attacks. To obtain adversarially robust models, researchers have separately developed adversarial training and Jacobian regularization techniques. There are abundant theoretical and empirical studies for adversarial training, but theoretical foundations for Jacobian regularization are still lacking. In this study, we show that Jacobian regularization is closely related to adversarial training in that $\ell_{2}$ or $\ell_{1}$ Jacobian regularized loss serves as an approximate upper bound on the adversarially robust loss under $\ell_{2}$ or $\ell_{\infty}$ adversarial attack respectively. Further, we establish the robust generalization gap for Jacobian regularized risk minimizer via bounding the Rademacher complexity of both the standard loss function class and Jacobian regularization function class. Our theoretical results indicate that the norms of Jacobian are related to both standard and robust generalization. We also perform experiments on MNIST data classification to demonstrate that Jacobian regularized risk minimization indeed serves as a surrogate for adversarially robust risk minimization, and that reducing the norms of Jacobian can improve both standard and robust generalization. This study promotes both theoretical and empirical understandings to adversarially robust generalization via Jacobian regularization.



## **44. Quantum Adversarial Machine Learning and Defense Strategies: Challenges and Opportunities**

quant-ph

24 pages, 9 figures, 12 tables

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.12373v1) [paper-pdf](http://arxiv.org/pdf/2412.12373v1)

**Authors**: Eric Yocam, Anthony Rizi, Mahesh Kamepalli, Varghese Vaidyan, Yong Wang, Gurcan Comert

**Abstract**: As quantum computing continues to advance, the development of quantum-secure neural networks is crucial to prevent adversarial attacks. This paper proposes three quantum-secure design principles: (1) using post-quantum cryptography, (2) employing quantum-resistant neural network architectures, and (3) ensuring transparent and accountable development and deployment. These principles are supported by various quantum strategies, including quantum data anonymization, quantum-resistant neural networks, and quantum encryption. The paper also identifies open issues in quantum security, privacy, and trust, and recommends exploring adaptive adversarial attacks and auto adversarial attacks as future directions. The proposed design principles and recommendations provide guidance for developing quantum-secure neural networks, ensuring the integrity and reliability of machine learning models in the quantum era.



## **45. Multi-Robot Target Tracking with Sensing and Communication Danger Zones**

cs.RO

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2404.07880v3) [paper-pdf](http://arxiv.org/pdf/2404.07880v3)

**Authors**: Jiazhen Liu, Peihan Li, Yuwei Wu, Gaurav S. Sukhatme, Vijay Kumar, Lifeng Zhou

**Abstract**: Multi-robot target tracking finds extensive applications in different scenarios, such as environmental surveillance and wildfire management, which require the robustness of the practical deployment of multi-robot systems in uncertain and dangerous environments. Traditional approaches often focus on the performance of tracking accuracy with no modeling and assumption of the environments, neglecting potential environmental hazards which result in system failures in real-world deployments. To address this challenge, we investigate multi-robot target tracking in the adversarial environment considering sensing and communication attacks with uncertainty. We design specific strategies to avoid different danger zones and proposed a multi-agent tracking framework under the perilous environment. We approximate the probabilistic constraints and formulate practical optimization strategies to address computational challenges efficiently. We evaluate the performance of our proposed methods in simulations to demonstrate the ability of robots to adjust their risk-aware behaviors under different levels of environmental uncertainty and risk confidence. The proposed method is further validated via real-world robot experiments where a team of drones successfully track dynamic ground robots while being risk-aware of the sensing and/or communication danger zones.



## **46. Adversarial Attacks on Large Language Models in Medicine**

cs.AI

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2406.12259v3) [paper-pdf](http://arxiv.org/pdf/2406.12259v3)

**Authors**: Yifan Yang, Qiao Jin, Furong Huang, Zhiyong Lu

**Abstract**: The integration of Large Language Models (LLMs) into healthcare applications offers promising advancements in medical diagnostics, treatment recommendations, and patient care. However, the susceptibility of LLMs to adversarial attacks poses a significant threat, potentially leading to harmful outcomes in delicate medical contexts. This study investigates the vulnerability of LLMs to two types of adversarial attacks in three medical tasks. Utilizing real-world patient data, we demonstrate that both open-source and proprietary LLMs are susceptible to manipulation across multiple tasks. This research further reveals that domain-specific tasks demand more adversarial data in model fine-tuning than general domain tasks for effective attack execution, especially for more capable models. We discover that while integrating adversarial data does not markedly degrade overall model performance on medical benchmarks, it does lead to noticeable shifts in fine-tuned model weights, suggesting a potential pathway for detecting and countering model attacks. This research highlights the urgent need for robust security measures and the development of defensive mechanisms to safeguard LLMs in medical applications, to ensure their safe and effective deployment in healthcare settings.



## **47. Robust Synthetic Data-Driven Detection of Living-Off-the-Land Reverse Shells**

cs.CR

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2402.18329v2) [paper-pdf](http://arxiv.org/pdf/2402.18329v2)

**Authors**: Dmitrijs Trizna, Luca Demetrio, Battista Biggio, Fabio Roli

**Abstract**: Living-off-the-land (LOTL) techniques pose a significant challenge to security operations, exploiting legitimate tools to execute malicious commands that evade traditional detection methods. To address this, we present a robust augmentation framework for cyber defense systems as Security Information and Event Management (SIEM) solutions, enabling the detection of LOTL attacks such as reverse shells through machine learning. Leveraging real-world threat intelligence and adversarial training, our framework synthesizes diverse malicious datasets while preserving the variability of legitimate activity, ensuring high accuracy and low false-positive rates. We validate our approach through extensive experiments on enterprise-scale datasets, achieving a 90\% improvement in detection rates over non-augmented baselines at an industry-grade False Positive Rate (FPR) of $10^{-5}$. We define black-box data-driven attacks that successfully evade unprotected models, and develop defenses to mitigate them, producing adversarially robust variants of ML models. Ethical considerations are central to this work; we discuss safeguards for synthetic data generation and the responsible release of pre-trained models across four best performing architectures, including both adversarially and regularly trained variants: https://huggingface.co/dtrizna/quasarnix. Furthermore, we provide a malicious LOTL dataset containing over 1 million augmented attack variants to enable reproducible research and community collaboration: https://huggingface.co/datasets/dtrizna/QuasarNix. This work offers a reproducible, scalable, and production-ready defense against evolving LOTL threats.



## **48. Sonar-based Deep Learning in Underwater Robotics: Overview, Robustness and Challenges**

cs.RO

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.11840v1) [paper-pdf](http://arxiv.org/pdf/2412.11840v1)

**Authors**: Martin Aubard, Ana Madureira, Luís Teixeira, José Pinto

**Abstract**: With the growing interest in underwater exploration and monitoring, Autonomous Underwater Vehicles (AUVs) have become essential. The recent interest in onboard Deep Learning (DL) has advanced real-time environmental interaction capabilities relying on efficient and accurate vision-based DL models. However, the predominant use of sonar in underwater environments, characterized by limited training data and inherent noise, poses challenges to model robustness. This autonomy improvement raises safety concerns for deploying such models during underwater operations, potentially leading to hazardous situations. This paper aims to provide the first comprehensive overview of sonar-based DL under the scope of robustness. It studies sonar-based DL perception task models, such as classification, object detection, segmentation, and SLAM. Furthermore, the paper systematizes sonar-based state-of-the-art datasets, simulators, and robustness methods such as neural network verification, out-of-distribution, and adversarial attacks. This paper highlights the lack of robustness in sonar-based DL research and suggests future research pathways, notably establishing a baseline sonar-based dataset and bridging the simulation-to-reality gap.



## **49. Transferable Adversarial Face Attack with Text Controlled Attribute**

cs.CV

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.11735v1) [paper-pdf](http://arxiv.org/pdf/2412.11735v1)

**Authors**: Wenyun Li, Zheng Zhang, Xiangyuan Lan, Dongmei Jiang

**Abstract**: Traditional adversarial attacks typically produce adversarial examples under norm-constrained conditions, whereas unrestricted adversarial examples are free-form with semantically meaningful perturbations. Current unrestricted adversarial impersonation attacks exhibit limited control over adversarial face attributes and often suffer from low transferability. In this paper, we propose a novel Text Controlled Attribute Attack (TCA$^2$) to generate photorealistic adversarial impersonation faces guided by natural language. Specifically, the category-level personal softmax vector is employed to precisely guide the impersonation attacks. Additionally, we propose both data and model augmentation strategies to achieve transferable attacks on unknown target models. Finally, a generative model, \textit{i.e}, Style-GAN, is utilized to synthesize impersonated faces with desired attributes. Extensive experiments on two high-resolution face recognition datasets validate that our TCA$^2$ method can generate natural text-guided adversarial impersonation faces with high transferability. We also evaluate our method on real-world face recognition systems, \textit{i.e}, Face++ and Aliyun, further demonstrating the practical potential of our approach.



## **50. Against All Odds: Overcoming Typology, Script, and Language Confusion in Multilingual Embedding Inversion Attacks**

cs.CL

11 pages, 4 figures, 7 tables

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2408.11749v2) [paper-pdf](http://arxiv.org/pdf/2408.11749v2)

**Authors**: Yiyi Chen, Russa Biswas, Heather Lent, Johannes Bjerva

**Abstract**: Large Language Models (LLMs) are susceptible to malicious influence by cyber attackers through intrusions such as adversarial, backdoor, and embedding inversion attacks. In response, the burgeoning field of LLM Security aims to study and defend against such threats. Thus far, the majority of works in this area have focused on monolingual English models, however, emerging research suggests that multilingual LLMs may be more vulnerable to various attacks than their monolingual counterparts. While previous work has investigated embedding inversion over a small subset of European languages, it is challenging to extrapolate these findings to languages from different linguistic families and with differing scripts. To this end, we explore the security of multilingual LLMs in the context of embedding inversion attacks and investigate cross-lingual and cross-script inversion across 20 languages, spanning over 8 language families and 12 scripts. Our findings indicate that languages written in Arabic script and Cyrillic script are particularly vulnerable to embedding inversion, as are languages within the Indo-Aryan language family. We further observe that inversion models tend to suffer from language confusion, sometimes greatly reducing the efficacy of an attack. Accordingly, we systematically explore this bottleneck for inversion models, uncovering predictable patterns which could be leveraged by attackers. Ultimately, this study aims to further the field's understanding of the outstanding security vulnerabilities facing multilingual LLMs and raise awareness for the languages most at risk of negative impact from these attacks.



