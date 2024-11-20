# Latest Adversarial Attack Papers
**update at 2024-11-20 11:30:44**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Attribute Inference Attacks for Federated Regression Tasks**

cs.LG

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12697v1) [paper-pdf](http://arxiv.org/pdf/2411.12697v1)

**Authors**: Francesco Diana, Othmane Marfoq, Chuan Xu, Giovanni Neglia, Frédéric Giroire, Eoin Thomas

**Abstract**: Federated Learning (FL) enables multiple clients, such as mobile phones and IoT devices, to collaboratively train a global machine learning model while keeping their data localized. However, recent studies have revealed that the training phase of FL is vulnerable to reconstruction attacks, such as attribute inference attacks (AIA), where adversaries exploit exchanged messages and auxiliary public information to uncover sensitive attributes of targeted clients. While these attacks have been extensively studied in the context of classification tasks, their impact on regression tasks remains largely unexplored. In this paper, we address this gap by proposing novel model-based AIAs specifically designed for regression tasks in FL environments. Our approach considers scenarios where adversaries can either eavesdrop on exchanged messages or directly interfere with the training process. We benchmark our proposed attacks against state-of-the-art methods using real-world datasets. The results demonstrate a significant increase in reconstruction accuracy, particularly in heterogeneous client datasets, a common scenario in FL. The efficacy of our model-based AIAs makes them better candidates for empirically quantifying privacy leakage for federated regression tasks.



## **2. Stochastic BIQA: Median Randomized Smoothing for Certified Blind Image Quality Assessment**

eess.IV

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12575v1) [paper-pdf](http://arxiv.org/pdf/2411.12575v1)

**Authors**: Ekaterina Shumitskaya, Mikhail Pautov, Dmitriy Vatolin, Anastasia Antsiferova

**Abstract**: Most modern No-Reference Image-Quality Assessment (NR-IQA) metrics are based on neural networks vulnerable to adversarial attacks. Attacks on such metrics lead to incorrect image/video quality predictions, which poses significant risks, especially in public benchmarks. Developers of image processing algorithms may unfairly increase the score of a target IQA metric without improving the actual quality of the adversarial image. Although some empirical defenses for IQA metrics were proposed, they do not provide theoretical guarantees and may be vulnerable to adaptive attacks. This work focuses on developing a provably robust no-reference IQA metric. Our method is based on Median Smoothing (MS) combined with an additional convolution denoiser with ranking loss to improve the SROCC and PLCC scores of the defended IQA metric. Compared with two prior methods on three datasets, our method exhibited superior SROCC and PLCC scores while maintaining comparable certified guarantees.



## **3. Variational Bayesian Bow tie Neural Networks with Shrinkage**

stat.ML

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.11132v2) [paper-pdf](http://arxiv.org/pdf/2411.11132v2)

**Authors**: Alisa Sheinkman, Sara Wade

**Abstract**: Despite the dominant role of deep models in machine learning, limitations persist, including overconfident predictions, susceptibility to adversarial attacks, and underestimation of variability in predictions. The Bayesian paradigm provides a natural framework to overcome such issues and has become the gold standard for uncertainty estimation with deep models, also providing improved accuracy and a framework for tuning critical hyperparameters. However, exact Bayesian inference is challenging, typically involving variational algorithms that impose strong independence and distributional assumptions. Moreover, existing methods are sensitive to the architectural choice of the network. We address these issues by constructing a relaxed version of the standard feed-forward rectified neural network, and employing Polya-Gamma data augmentation tricks to render a conditionally linear and Gaussian model. Additionally, we use sparsity-promoting priors on the weights of the neural network for data-driven architectural design. To approximate the posterior, we derive a variational inference algorithm that avoids distributional assumptions and independence across layers and is a faster alternative to the usual Markov Chain Monte Carlo schemes.



## **4. NMT-Obfuscator Attack: Ignore a sentence in translation with only one word**

cs.CL

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12473v1) [paper-pdf](http://arxiv.org/pdf/2411.12473v1)

**Authors**: Sahar Sadrizadeh, César Descalzo, Ljiljana Dolamic, Pascal Frossard

**Abstract**: Neural Machine Translation systems are used in diverse applications due to their impressive performance. However, recent studies have shown that these systems are vulnerable to carefully crafted small perturbations to their inputs, known as adversarial attacks. In this paper, we propose a new type of adversarial attack against NMT models. In this attack, we find a word to be added between two sentences such that the second sentence is ignored and not translated by the NMT model. The word added between the two sentences is such that the whole adversarial text is natural in the source language. This type of attack can be harmful in practical scenarios since the attacker can hide malicious information in the automatic translation made by the target NMT model. Our experiments show that different NMT models and translation tasks are vulnerable to this type of attack. Our attack can successfully force the NMT models to ignore the second part of the input in the translation for more than 50% of all cases while being able to maintain low perplexity for the whole input.



## **5. Efficient Verifiable Differential Privacy with Input Authenticity in the Local and Shuffle Model**

cs.CR

21 pages, 13 figures, 2 tables; accepted for publication in the  Proceedings on the 25th Privacy Enhancing Technologies Symposium (PoPETs)  2025

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2406.18940v2) [paper-pdf](http://arxiv.org/pdf/2406.18940v2)

**Authors**: Tariq Bontekoe, Hassan Jameel Asghar, Fatih Turkmen

**Abstract**: Local differential privacy (LDP) enables the efficient release of aggregate statistics without having to trust the central server (aggregator), as in the central model of differential privacy, and simultaneously protects a client's sensitive data. The shuffle model with LDP provides an additional layer of privacy, by disconnecting the link between clients and the aggregator. However, LDP has been shown to be vulnerable to malicious clients who can perform both input and output manipulation attacks, i.e., before and after applying the LDP mechanism, to skew the aggregator's results. In this work, we show how to prevent malicious clients from compromising LDP schemes. Our only realistic assumption is that the initial raw input is authenticated; the rest of the processing pipeline, e.g., formatting the input and applying the LDP mechanism, may be under adversarial control. We give several real-world examples where this assumption is justified. Our proposed schemes for verifiable LDP (VLDP), prevent both input and output manipulation attacks against generic LDP mechanisms, requiring only one-time interaction between client and server, unlike existing alternatives [37, 43]. Most importantly, we are the first to provide an efficient scheme for VLDP in the shuffle model. We describe, and prove security of, two schemes for VLDP in the local model, and one in the shuffle model. We show that all schemes are highly practical, with client run times of less than 2 seconds, and server run times of 5-7 milliseconds per client.



## **6. DeTrigger: A Gradient-Centric Approach to Backdoor Attack Mitigation in Federated Learning**

cs.LG

14 pages

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12220v1) [paper-pdf](http://arxiv.org/pdf/2411.12220v1)

**Authors**: Kichang Lee, Yujin Shin, Jonghyuk Yun, Jun Han, JeongGil Ko

**Abstract**: Federated Learning (FL) enables collaborative model training across distributed devices while preserving local data privacy, making it ideal for mobile and embedded systems. However, the decentralized nature of FL also opens vulnerabilities to model poisoning attacks, particularly backdoor attacks, where adversaries implant trigger patterns to manipulate model predictions. In this paper, we propose DeTrigger, a scalable and efficient backdoor-robust federated learning framework that leverages insights from adversarial attack methodologies. By employing gradient analysis with temperature scaling, DeTrigger detects and isolates backdoor triggers, allowing for precise model weight pruning of backdoor activations without sacrificing benign model knowledge. Extensive evaluations across four widely used datasets demonstrate that DeTrigger achieves up to 251x faster detection than traditional methods and mitigates backdoor attacks by up to 98.9%, with minimal impact on global model accuracy. Our findings establish DeTrigger as a robust and scalable solution to protect federated learning environments against sophisticated backdoor threats.



## **7. Architectural Patterns for Designing Quantum Artificial Intelligence Systems**

cs.SE

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.10487v2) [paper-pdf](http://arxiv.org/pdf/2411.10487v2)

**Authors**: Mykhailo Klymenko, Thong Hoang, Xiwei Xu, Zhenchang Xing, Muhammad Usman, Qinghua Lu, Liming Zhu

**Abstract**: Utilising quantum computing technology to enhance artificial intelligence systems is expected to improve training and inference times, increase robustness against noise and adversarial attacks, and reduce the number of parameters without compromising accuracy. However, moving beyond proof-of-concept or simulations to develop practical applications of these systems while ensuring high software quality faces significant challenges due to the limitations of quantum hardware and the underdeveloped knowledge base in software engineering for such systems. In this work, we have conducted a systematic mapping study to identify the challenges and solutions associated with the software architecture of quantum-enhanced artificial intelligence systems. Our review uncovered several architectural patterns that describe how quantum components can be integrated into inference engines, as well as middleware patterns that facilitate communication between classical and quantum components. These insights have been compiled into a catalog of architectural patterns. Each pattern realises a trade-off between efficiency, scalability, trainability, simplicity, portability and deployability, and other software quality attributes.



## **8. Adversarial Multi-Agent Reinforcement Learning for Proactive False Data Injection Detection**

eess.SY

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12130v1) [paper-pdf](http://arxiv.org/pdf/2411.12130v1)

**Authors**: Kejun Chen, Truc Nguyen, Malik Hassanaly

**Abstract**: Smart inverters are instrumental in the integration of renewable and distributed energy resources (DERs) into the electric grid. Such inverters rely on communication layers for continuous control and monitoring, potentially exposing them to cyber-physical attacks such as false data injection attacks (FDIAs). We propose to construct a defense strategy against a priori unknown FDIAs with a multi-agent reinforcement learning (MARL) framework. The first agent is an adversary that simulates and discovers various FDIA strategies, while the second agent is a defender in charge of detecting and localizing FDIAs. This approach enables the defender to be trained against new FDIAs continuously generated by the adversary. The numerical results demonstrate that the proposed MARL defender outperforms a supervised offline defender. Additionally, we show that the detection skills of an MARL defender can be combined with that of an offline defender through a transfer learning approach.



## **9. Theoretical Corrections and the Leveraging of Reinforcement Learning to Enhance Triangle Attack**

cs.LG

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.12071v1) [paper-pdf](http://arxiv.org/pdf/2411.12071v1)

**Authors**: Nicole Meng, Caleb Manicke, David Chen, Yingjie Lao, Caiwen Ding, Pengyu Hong, Kaleel Mahmood

**Abstract**: Adversarial examples represent a serious issue for the application of machine learning models in many sensitive domains. For generating adversarial examples, decision based black-box attacks are one of the most practical techniques as they only require query access to the model. One of the most recently proposed state-of-the-art decision based black-box attacks is Triangle Attack (TA). In this paper, we offer a high-level description of TA and explain potential theoretical limitations. We then propose a new decision based black-box attack, Triangle Attack with Reinforcement Learning (TARL). Our new attack addresses the limits of TA by leveraging reinforcement learning. This creates an attack that can achieve similar, if not better, attack accuracy than TA with half as many queries on state-of-the-art classifiers and defenses across ImageNet and CIFAR-10.



## **10. Exploring adversarial robustness of JPEG AI: methodology, comparison and new methods**

eess.IV

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11795v1) [paper-pdf](http://arxiv.org/pdf/2411.11795v1)

**Authors**: Egor Kovalev, Georgii Bychkov, Khaled Abud, Aleksandr Gushchin, Anna Chistyakova, Sergey Lavrushkin, Dmitriy Vatolin, Anastasia Antsiferova

**Abstract**: Adversarial robustness of neural networks is an increasingly important area of research, combining studies on computer vision models, large language models (LLMs), and others. With the release of JPEG AI - the first standard for end-to-end neural image compression (NIC) methods - the question of its robustness has become critically significant. JPEG AI is among the first international, real-world applications of neural-network-based models to be embedded in consumer devices. However, research on NIC robustness has been limited to open-source codecs and a narrow range of attacks. This paper proposes a new methodology for measuring NIC robustness to adversarial attacks. We present the first large-scale evaluation of JPEG AI's robustness, comparing it with other NIC models. Our evaluation results and code are publicly available online (link is hidden for a blind review).



## **11. Robust Subgraph Learning by Monitoring Early Training Representations**

cs.LG

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2403.09901v2) [paper-pdf](http://arxiv.org/pdf/2403.09901v2)

**Authors**: Sepideh Neshatfar, Salimeh Yasaei Sekeh

**Abstract**: Graph neural networks (GNNs) have attracted significant attention for their outstanding performance in graph learning and node classification tasks. However, their vulnerability to adversarial attacks, particularly through susceptible nodes, poses a challenge in decision-making. The need for robust graph summarization is evident in adversarial challenges resulting from the propagation of attacks throughout the entire graph. In this paper, we address both performance and adversarial robustness in graph input by introducing the novel technique SHERD (Subgraph Learning Hale through Early Training Representation Distances). SHERD leverages information from layers of a partially trained graph convolutional network (GCN) to detect susceptible nodes during adversarial attacks using standard distance metrics. The method identifies "vulnerable (bad)" nodes and removes such nodes to form a robust subgraph while maintaining node classification performance. Through our experiments, we demonstrate the increased performance of SHERD in enhancing robustness by comparing the network's performance on original and subgraph inputs against various baselines alongside existing adversarial attacks. Our experiments across multiple datasets, including citation datasets such as Cora, Citeseer, and Pubmed, as well as microanatomical tissue structures of cell graphs in the placenta, highlight that SHERD not only achieves substantial improvement in robust performance but also outperforms several baselines in terms of node classification accuracy and computational complexity.



## **12. Eidos: Efficient, Imperceptible Adversarial 3D Point Clouds**

cs.CV

Preprint

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2405.14210v2) [paper-pdf](http://arxiv.org/pdf/2405.14210v2)

**Authors**: Hanwei Zhang, Luo Cheng, Qisong He, Wei Huang, Renjue Li, Ronan Sicre, Xiaowei Huang, Holger Hermanns, Lijun Zhang

**Abstract**: Classification of 3D point clouds is a challenging machine learning (ML) task with important real-world applications in a spectrum from autonomous driving and robot-assisted surgery to earth observation from low orbit. As with other ML tasks, classification models are notoriously brittle in the presence of adversarial attacks. These are rooted in imperceptible changes to inputs with the effect that a seemingly well-trained model ends up misclassifying the input. This paper adds to the understanding of adversarial attacks by presenting Eidos, a framework providing Efficient Imperceptible aDversarial attacks on 3D pOint cloudS. Eidos supports a diverse set of imperceptibility metrics. It employs an iterative, two-step procedure to identify optimal adversarial examples, thereby enabling a runtime-imperceptibility trade-off. We provide empirical evidence relative to several popular 3D point cloud classification models and several established 3D attack methods, showing Eidos' superiority with respect to efficiency as well as imperceptibility.



## **13. Bitcoin Under Volatile Block Rewards: How Mempool Statistics Can Influence Bitcoin Mining**

cs.CR

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11702v1) [paper-pdf](http://arxiv.org/pdf/2411.11702v1)

**Authors**: Roozbeh Sarenche, Alireza Aghabagherloo, Svetla Nikova, Bart Preneel

**Abstract**: As Bitcoin experiences more halving events, the protocol reward converges to zero, making transaction fees the primary source of miner rewards. This shift in Bitcoin's incentivization mechanism, which introduces volatility into block rewards, could lead to the emergence of new security threats or intensify existing ones. Previous security analyses of Bitcoin have either considered a fixed block reward model or a highly simplified volatile model, overlooking the complexities of Bitcoin's mempool behavior.   In this paper, we present a reinforcement learning-based tool designed to analyze mining strategies under a more realistic volatile model. Our tool uses the Asynchronous Advantage Actor-Critic (A3C) algorithm to derive near-optimal mining strategies while interacting with an environment that models the complexity of the Bitcoin mempool. This tool enables the analysis of adversarial mining strategies, such as selfish mining and undercutting, both before and after difficulty adjustments, providing insights into the effects of mining attacks in both the short and long term.   Our analysis reveals that Bitcoin users' trend of offering higher fees to speed up the inclusion of their transactions in the chain can incentivize payoff-maximizing miners to deviate from the honest strategy. In the fixed reward model, a disincentive for the selfish mining attack is the initial loss period of at least two weeks, during which the attack is not profitable. However, our analysis shows that once the protocol reward diminishes to zero in the future, or even currently on days when transaction fees are comparable to the protocol reward, mining pools might be incentivized to abandon honest mining to gain an immediate profit.



## **14. TrojanRobot: Backdoor Attacks Against Robotic Manipulation in the Physical World**

cs.RO

Initial version with preliminary results. We welcome any feedback or  suggestions

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11683v1) [paper-pdf](http://arxiv.org/pdf/2411.11683v1)

**Authors**: Xianlong Wang, Hewen Pan, Hangtao Zhang, Minghui Li, Shengshan Hu, Ziqi Zhou, Lulu Xue, Peijin Guo, Yichen Wang, Wei Wan, Aishan Liu, Leo Yu Zhang

**Abstract**: Robotic manipulation refers to the autonomous handling and interaction of robots with objects using advanced techniques in robotics and artificial intelligence. The advent of powerful tools such as large language models (LLMs) and large vision-language models (LVLMs) has significantly enhanced the capabilities of these robots in environmental perception and decision-making. However, the introduction of these intelligent agents has led to security threats such as jailbreak attacks and adversarial attacks.   In this research, we take a further step by proposing a backdoor attack specifically targeting robotic manipulation and, for the first time, implementing backdoor attack in the physical world. By embedding a backdoor visual language model into the visual perception module within the robotic system, we successfully mislead the robotic arm's operation in the physical world, given the presence of common items as triggers. Experimental evaluations in the physical world demonstrate the effectiveness of the proposed backdoor attack.



## **15. Few-shot Model Extraction Attacks against Sequential Recommender Systems**

cs.LG

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11677v1) [paper-pdf](http://arxiv.org/pdf/2411.11677v1)

**Authors**: Hui Zhang, Fu Liu

**Abstract**: Among adversarial attacks against sequential recommender systems, model extraction attacks represent a method to attack sequential recommendation models without prior knowledge. Existing research has primarily concentrated on the adversary's execution of black-box attacks through data-free model extraction. However, a significant gap remains in the literature concerning the development of surrogate models by adversaries with access to few-shot raw data (10\% even less). That is, the challenge of how to construct a surrogate model with high functional similarity within the context of few-shot data scenarios remains an issue that requires resolution.This study addresses this gap by introducing a novel few-shot model extraction framework against sequential recommenders, which is designed to construct a superior surrogate model with the utilization of few-shot data. The proposed few-shot model extraction framework is comprised of two components: an autoregressive augmentation generation strategy and a bidirectional repair loss-facilitated model distillation procedure. Specifically, to generate synthetic data that closely approximate the distribution of raw data, autoregressive augmentation generation strategy integrates a probabilistic interaction sampler to extract inherent dependencies and a synthesis determinant signal module to characterize user behavioral patterns. Subsequently, bidirectional repair loss, which target the discrepancies between the recommendation lists, is designed as auxiliary loss to rectify erroneous predictions from surrogate models, transferring knowledge from the victim model to the surrogate model effectively. Experiments on three datasets show that the proposed few-shot model extraction framework yields superior surrogate models.



## **16. Formal Verification of Deep Neural Networks for Object Detection**

cs.CV

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2407.01295v5) [paper-pdf](http://arxiv.org/pdf/2407.01295v5)

**Authors**: Yizhak Y. Elboher, Avraham Raviv, Yael Leibovich Weiss, Omer Cohen, Roy Assa, Guy Katz, Hillel Kugler

**Abstract**: Deep neural networks (DNNs) are widely used in real-world applications, yet they remain vulnerable to errors and adversarial attacks. Formal verification offers a systematic approach to identify and mitigate these vulnerabilities, enhancing model robustness and reliability. While most existing verification methods focus on image classification models, this work extends formal verification to the more complex domain of emph{object detection} models. We propose a formulation for verifying the robustness of such models and demonstrate how state-of-the-art verification tools, originally developed for classification, can be adapted for this purpose. Our experiments, conducted on various datasets and networks, highlight the ability of formal verification to uncover vulnerabilities in object detection models, underscoring the need to extend verification efforts to this domain. This work lays the foundation for further research into formal verification across a broader range of computer vision applications.



## **17. The Dark Side of Trust: Authority Citation-Driven Jailbreak Attacks on Large Language Models**

cs.LG

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11407v1) [paper-pdf](http://arxiv.org/pdf/2411.11407v1)

**Authors**: Xikang Yang, Xuehai Tang, Jizhong Han, Songlin Hu

**Abstract**: The widespread deployment of large language models (LLMs) across various domains has showcased their immense potential while exposing significant safety vulnerabilities. A major concern is ensuring that LLM-generated content aligns with human values. Existing jailbreak techniques reveal how this alignment can be compromised through specific prompts or adversarial suffixes. In this study, we introduce a new threat: LLMs' bias toward authority. While this inherent bias can improve the quality of outputs generated by LLMs, it also introduces a potential vulnerability, increasing the risk of producing harmful content. Notably, the biases in LLMs is the varying levels of trust given to different types of authoritative information in harmful queries. For example, malware development often favors trust GitHub. To better reveal the risks with LLM, we propose DarkCite, an adaptive authority citation matcher and generator designed for a black-box setting. DarkCite matches optimal citation types to specific risk types and generates authoritative citations relevant to harmful instructions, enabling more effective jailbreak attacks on aligned LLMs.Our experiments show that DarkCite achieves a higher attack success rate (e.g., LLama-2 at 76% versus 68%) than previous methods. To counter this risk, we propose an authenticity and harm verification defense strategy, raising the average defense pass rate (DPR) from 11% to 74%. More importantly, the ability to link citations to the content they encompass has become a foundational function in LLMs, amplifying the influence of LLMs' bias toward authority.



## **18. Hacking Back the AI-Hacker: Prompt Injection as a Defense Against LLM-driven Cyberattacks**

cs.CR

v0.2 (evaluated on more agents)

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2410.20911v2) [paper-pdf](http://arxiv.org/pdf/2410.20911v2)

**Authors**: Dario Pasquini, Evgenios M. Kornaropoulos, Giuseppe Ateniese

**Abstract**: Large language models (LLMs) are increasingly being harnessed to automate cyberattacks, making sophisticated exploits more accessible and scalable. In response, we propose a new defense strategy tailored to counter LLM-driven cyberattacks. We introduce Mantis, a defensive framework that exploits LLMs' susceptibility to adversarial inputs to undermine malicious operations. Upon detecting an automated cyberattack, Mantis plants carefully crafted inputs into system responses, leading the attacker's LLM to disrupt their own operations (passive defense) or even compromise the attacker's machine (active defense). By deploying purposefully vulnerable decoy services to attract the attacker and using dynamic prompt injections for the attacker's LLM, Mantis can autonomously hack back the attacker. In our experiments, Mantis consistently achieved over 95% effectiveness against automated LLM-driven attacks. To foster further research and collaboration, Mantis is available as an open-source tool: https://github.com/pasquini-dario/project_mantis



## **19. Adapting to Cyber Threats: A Phishing Evolution Network (PEN) Framework for Phishing Generation and Analyzing Evolution Patterns using Large Language Models**

cs.CR

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11389v1) [paper-pdf](http://arxiv.org/pdf/2411.11389v1)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Shuo Wang, Hongsheng Hu, Alsharif Abuadbba, Carsten Rudolph

**Abstract**: Phishing remains a pervasive cyber threat, as attackers craft deceptive emails to lure victims into revealing sensitive information. While Artificial Intelligence (AI), particularly deep learning, has become a key component in defending against phishing attacks, these approaches face critical limitations. The scarcity of publicly available, diverse, and updated data, largely due to privacy concerns, constrains their effectiveness. As phishing tactics evolve rapidly, models trained on limited, outdated data struggle to detect new, sophisticated deception strategies, leaving systems vulnerable to an ever-growing array of attacks. Addressing this gap is essential to strengthening defenses in an increasingly hostile cyber landscape. To address this gap, we propose the Phishing Evolution Network (PEN), a framework leveraging large language models (LLMs) and adversarial training mechanisms to continuously generate high quality and realistic diverse phishing samples, and analyze features of LLM-provided phishing to understand evolving phishing patterns. We evaluate the quality and diversity of phishing samples generated by PEN and find that it produces over 80% realistic phishing samples, effectively expanding phishing datasets across seven dominant types. These PEN-generated samples enhance the performance of current phishing detectors, leading to a 40% improvement in detection accuracy. Additionally, the use of PEN significantly boosts model robustness, reducing detectors' sensitivity to perturbations by up to 60%, thereby decreasing attack success rates under adversarial conditions. When we analyze the phishing patterns that are used in LLM-generated phishing, the cognitive complexity and the tone of time limitation are detected with statistically significant differences compared with existing phishing.



## **20. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2410.23091v4) [paper-pdf](http://arxiv.org/pdf/2410.23091v4)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark). The code is available at \href{https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff}{https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff}



## **21. Countering Backdoor Attacks in Image Recognition: A Survey and Evaluation of Mitigation Strategies**

cs.CR

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2411.11200v1) [paper-pdf](http://arxiv.org/pdf/2411.11200v1)

**Authors**: Kealan Dunnett, Reza Arablouei, Dimity Miller, Volkan Dedeoglu, Raja Jurdak

**Abstract**: The widespread adoption of deep learning across various industries has introduced substantial challenges, particularly in terms of model explainability and security. The inherent complexity of deep learning models, while contributing to their effectiveness, also renders them susceptible to adversarial attacks. Among these, backdoor attacks are especially concerning, as they involve surreptitiously embedding specific triggers within training data, causing the model to exhibit aberrant behavior when presented with input containing the triggers. Such attacks often exploit vulnerabilities in outsourced processes, compromising model integrity without affecting performance on clean (trigger-free) input data. In this paper, we present a comprehensive review of existing mitigation strategies designed to counter backdoor attacks in image recognition. We provide an in-depth analysis of the theoretical foundations, practical efficacy, and limitations of these approaches. In addition, we conduct an extensive benchmarking of sixteen state-of-the-art approaches against eight distinct backdoor attacks, utilizing three datasets, four model architectures, and three poisoning ratios. Our results, derived from 122,236 individual experiments, indicate that while many approaches provide some level of protection, their performance can vary considerably. Furthermore, when compared to two seminal approaches, most newer approaches do not demonstrate substantial improvements in overall performance or consistency across diverse settings. Drawing from these findings, we propose potential directions for developing more effective and generalizable defensive mechanisms in the future.



## **22. Optimal Denial-of-Service Attacks Against Partially-Observable Real-Time Monitoring Systems**

cs.IT

arXiv admin note: text overlap with arXiv:2403.04489

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2409.16794v2) [paper-pdf](http://arxiv.org/pdf/2409.16794v2)

**Authors**: Saad Kriouile, Mohamad Assaad, Amira Alloum, Touraj Soleymani

**Abstract**: In this paper, we investigate the impact of denial-of-service attacks on the status updating of a cyber-physical system with one or more sensors connected to a remote monitor via unreliable channels. We approach the problem from the perspective of an adversary that can strategically jam a subset of the channels. The sources are modeled as Markov chains, and the performance of status updating is measured based on the age of incorrect information at the monitor. Our objective is to derive jamming policies that strike a balance between the degradation of the system's performance and the conservation of the adversary's energy. For a single-source scenario, we formulate the problem as a partially-observable Markov decision process, and rigorously prove that the optimal jamming policy is of a threshold form. We then extend the problem to a multi-source scenario. We formulate this problem as a restless multi-armed bandit, and provide a jamming policy based on the Whittle's index. Our numerical results highlight the performance of our policies compared to baseline policies.



## **23. CLMIA: Membership Inference Attacks via Unsupervised Contrastive Learning**

cs.LG

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2411.11144v1) [paper-pdf](http://arxiv.org/pdf/2411.11144v1)

**Authors**: Depeng Chen, Xiao Liu, Jie Cui, Hong Zhong

**Abstract**: Since machine learning model is often trained on a limited data set, the model is trained multiple times on the same data sample, which causes the model to memorize most of the training set data. Membership Inference Attacks (MIAs) exploit this feature to determine whether a data sample is used for training a machine learning model. However, in realistic scenarios, it is difficult for the adversary to obtain enough qualified samples that mark accurate identity information, especially since most samples are non-members in real world applications. To address this limitation, in this paper, we propose a new attack method called CLMIA, which uses unsupervised contrastive learning to train an attack model without using extra membership status information. Meanwhile, in CLMIA, we require only a small amount of data with known membership status to fine-tune the attack model. Experimental results demonstrate that CLMIA performs better than existing attack methods for different datasets and model structures, especially with data with less marked identity information. In addition, we experimentally find that the attack performs differently for different proportions of labeled identity information for member and non-member data. More analysis proves that our attack method performs better with less labeled identity information, which applies to more realistic scenarios.



## **24. JailbreakLens: Interpreting Jailbreak Mechanism in the Lens of Representation and Circuit**

cs.CR

18 pages, 10 figures

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2411.11114v1) [paper-pdf](http://arxiv.org/pdf/2411.11114v1)

**Authors**: Zeqing He, Zhibo Wang, Zhixuan Chu, Huiyu Xu, Rui Zheng, Kui Ren, Chun Chen

**Abstract**: Despite the outstanding performance of Large language models (LLMs) in diverse tasks, they are vulnerable to jailbreak attacks, wherein adversarial prompts are crafted to bypass their security mechanisms and elicit unexpected responses.Although jailbreak attacks are prevalent, the understanding of their underlying mechanisms remains limited. Recent studies have explain typical jailbreaking behavior (e.g., the degree to which the model refuses to respond) of LLMs by analyzing the representation shifts in their latent space caused by jailbreak prompts or identifying key neurons that contribute to the success of these attacks. However, these studies neither explore diverse jailbreak patterns nor provide a fine-grained explanation from the failure of circuit to the changes of representational, leaving significant gaps in uncovering the jailbreak mechanism. In this paper, we propose JailbreakLens, an interpretation framework that analyzes jailbreak mechanisms from both representation (which reveals how jailbreaks alter the model's harmfulness perception) and circuit perspectives (which uncovers the causes of these deceptions by identifying key circuits contributing to the vulnerability), tracking their evolution throughout the entire response generation process. We then conduct an in-depth evaluation of jailbreak behavior on four mainstream LLMs under seven jailbreak strategies. Our evaluation finds that jailbreak prompts amplify components that reinforce affirmative responses while suppressing those that produce refusal. Although this manipulation shifts model representations toward safe clusters to deceive the LLM, leading it to provide detailed responses instead of refusals, it still produce abnormal activation which can be caught in the circuit analysis.



## **25. Exploring the Adversarial Frontier: Quantifying Robustness via Adversarial Hypervolume**

cs.CR

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2403.05100v2) [paper-pdf](http://arxiv.org/pdf/2403.05100v2)

**Authors**: Ping Guo, Cheng Gong, Xi Lin, Zhiyuan Yang, Qingfu Zhang

**Abstract**: The escalating threat of adversarial attacks on deep learning models, particularly in security-critical fields, has underscored the need for robust deep learning systems. Conventional robustness evaluations have relied on adversarial accuracy, which measures a model's performance under a specific perturbation intensity. However, this singular metric does not fully encapsulate the overall resilience of a model against varying degrees of perturbation. To address this gap, we propose a new metric termed adversarial hypervolume, assessing the robustness of deep learning models comprehensively over a range of perturbation intensities from a multi-objective optimization standpoint. This metric allows for an in-depth comparison of defense mechanisms and recognizes the trivial improvements in robustness afforded by less potent defensive strategies. Additionally, we adopt a novel training algorithm that enhances adversarial robustness uniformly across various perturbation intensities, in contrast to methods narrowly focused on optimizing adversarial accuracy. Our extensive empirical studies validate the effectiveness of the adversarial hypervolume metric, demonstrating its ability to reveal subtle differences in robustness that adversarial accuracy overlooks. This research contributes a new measure of robustness and establishes a standard for assessing and benchmarking the resilience of current and future defensive models against adversarial threats.



## **26. Game-Theoretic Neyman-Pearson Detection to Combat Strategic Evasion**

cs.CR

**SubmitDate**: 2024-11-16    [abs](http://arxiv.org/abs/2206.05276v3) [paper-pdf](http://arxiv.org/pdf/2206.05276v3)

**Authors**: Yinan Hu, Quanyan Zhu

**Abstract**: The security in networked systems depends greatly on recognizing and identifying adversarial behaviors. Traditional detection methods focus on specific categories of attacks and have become inadequate for increasingly stealthy and deceptive attacks that are designed to bypass detection strategically. This work aims to develop a holistic theory to countermeasure such evasive attacks. We focus on extending a fundamental class of statistical-based detection methods based on Neyman-Pearson's (NP) hypothesis testing formulation. We propose game-theoretic frameworks to capture the conflicting relationship between a strategic evasive attacker and an evasion-aware NP detector. By analyzing both the equilibrium behaviors of the attacker and the NP detector, we characterize their performance using Equilibrium Receiver-Operational-Characteristic (EROC) curves. We show that the evasion-aware NP detectors outperform the passive ones in the way that the former can act strategically against the attacker's behavior and adaptively modify their decision rules based on the received messages. In addition, we extend our framework to a sequential setting where the user sends out identically distributed messages. We corroborate the analytical results with a case study of anomaly detection.



## **27. A Survey of Graph Unlearning**

cs.LG

22 page review paper on graph unlearning

**SubmitDate**: 2024-11-16    [abs](http://arxiv.org/abs/2310.02164v3) [paper-pdf](http://arxiv.org/pdf/2310.02164v3)

**Authors**: Anwar Said, Yuying Zhao, Tyler Derr, Mudassir Shabbir, Waseem Abbas, Xenofon Koutsoukos

**Abstract**: Graph unlearning emerges as a crucial advancement in the pursuit of responsible AI, providing the means to remove sensitive data traces from trained models, thereby upholding the right to be forgotten. It is evident that graph machine learning exhibits sensitivity to data privacy and adversarial attacks, necessitating the application of graph unlearning techniques to address these concerns effectively. In this comprehensive survey paper, we present the first systematic review of graph unlearning approaches, encompassing a diverse array of methodologies and offering a detailed taxonomy and up-to-date literature overview to facilitate the understanding of researchers new to this field. To ensure clarity, we provide lucid explanations of the fundamental concepts and evaluation measures used in graph unlearning, catering to a broader audience with varying levels of expertise. Delving into potential applications, we explore the versatility of graph unlearning across various domains, including but not limited to social networks, adversarial settings, recommender systems, and resource-constrained environments like the Internet of Things, illustrating its potential impact in safeguarding data privacy and enhancing AI systems' robustness. Finally, we shed light on promising research directions, encouraging further progress and innovation within the domain of graph unlearning. By laying a solid foundation and fostering continued progress, this survey seeks to inspire researchers to further advance the field of graph unlearning, thereby instilling confidence in the ethical growth of AI systems and reinforcing the responsible application of machine learning techniques in various domains.



## **28. Verifiably Robust Conformal Prediction**

cs.LO

Accepted at NeurIPS 2024

**SubmitDate**: 2024-11-16    [abs](http://arxiv.org/abs/2405.18942v3) [paper-pdf](http://arxiv.org/pdf/2405.18942v3)

**Authors**: Linus Jeary, Tom Kuipers, Mehran Hosseini, Nicola Paoletti

**Abstract**: Conformal Prediction (CP) is a popular uncertainty quantification method that provides distribution-free, statistically valid prediction sets, assuming that training and test data are exchangeable. In such a case, CP's prediction sets are guaranteed to cover the (unknown) true test output with a user-specified probability. Nevertheless, this guarantee is violated when the data is subjected to adversarial attacks, which often result in a significant loss of coverage. Recently, several approaches have been put forward to recover CP guarantees in this setting. These approaches leverage variations of randomised smoothing to produce conservative sets which account for the effect of the adversarial perturbations. They are, however, limited in that they only support $\ell^2$-bounded perturbations and classification tasks. This paper introduces VRCP (Verifiably Robust Conformal Prediction), a new framework that leverages recent neural network verification methods to recover coverage guarantees under adversarial attacks. Our VRCP method is the first to support perturbations bounded by arbitrary norms including $\ell^1$, $\ell^2$, and $\ell^\infty$, as well as regression tasks. We evaluate and compare our approach on image classification tasks (CIFAR10, CIFAR100, and TinyImageNet) and regression tasks for deep reinforcement learning environments. In every case, VRCP achieves above nominal coverage and yields significantly more efficient and informative prediction regions than the SotA.



## **29. Towards Physically-Realizable Adversarial Attacks in Embodied Vision Navigation**

cs.CV

8 pages, 6 figures, submitted to the 2025 IEEE International  Conference on Robotics & Automation (ICRA)

**SubmitDate**: 2024-11-16    [abs](http://arxiv.org/abs/2409.10071v3) [paper-pdf](http://arxiv.org/pdf/2409.10071v3)

**Authors**: Meng Chen, Jiawei Tu, Chao Qi, Yonghao Dang, Feng Zhou, Wei Wei, Jianqin Yin

**Abstract**: The deployment of embodied navigation agents in safety-critical environments raises concerns about their vulnerability to adversarial attacks on deep neural networks. However, current attack methods often lack practicality due to challenges in transitioning from the digital to the physical world, while existing physical attacks for object detection fail to achieve both multi-view effectiveness and naturalness. To address this, we propose a practical attack method for embodied navigation by attaching adversarial patches with learnable textures and opacity to objects. Specifically, to ensure effectiveness across varying viewpoints, we employ a multi-view optimization strategy based on object-aware sampling, which uses feedback from the navigation model to optimize the patch's texture. To make the patch inconspicuous to human observers, we introduce a two-stage opacity optimization mechanism, where opacity is refined after texture optimization. Experimental results show our adversarial patches reduce navigation success rates by about 40%, outperforming previous methods in practicality, effectiveness, and naturalness. Code is available at: [https://github.com/chen37058/Physical-Attacks-in-Embodied-Navigation].



## **30. Sim-CLIP: Unsupervised Siamese Adversarial Fine-Tuning for Robust and Semantically-Rich Vision-Language Models**

cs.CV

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2407.14971v2) [paper-pdf](http://arxiv.org/pdf/2407.14971v2)

**Authors**: Md Zarif Hossain, Ahmed Imteaj

**Abstract**: Vision-language models (VLMs) have achieved significant strides in recent times specially in multimodal tasks, yet they remain susceptible to adversarial attacks on their vision components. To address this, we propose Sim-CLIP, an unsupervised adversarial fine-tuning method that enhances the robustness of the widely-used CLIP vision encoder against such attacks while maintaining semantic richness and specificity. By employing a Siamese architecture with cosine similarity loss, Sim-CLIP learns semantically meaningful and attack-resilient visual representations without requiring large batch sizes or momentum encoders. Our results demonstrate that VLMs enhanced with Sim-CLIP's fine-tuned CLIP encoder exhibit significantly enhanced robustness against adversarial attacks, while preserving semantic meaning of the perturbed images. Notably, Sim-CLIP does not require additional training or fine-tuning of the VLM itself; replacing the original vision encoder with our fine-tuned Sim-CLIP suffices to provide robustness. This work underscores the significance of reinforcing foundational models like CLIP to safeguard the reliability of downstream VLM applications, paving the way for more secure and effective multimodal systems.



## **31. Comparing Robustness Against Adversarial Attacks in Code Generation: LLM-Generated vs. Human-Written**

cs.SE

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10565v1) [paper-pdf](http://arxiv.org/pdf/2411.10565v1)

**Authors**: Md Abdul Awal, Mrigank Rochan, Chanchal K. Roy

**Abstract**: Thanks to the widespread adoption of Large Language Models (LLMs) in software engineering research, the long-standing dream of automated code generation has become a reality on a large scale. Nowadays, LLMs such as GitHub Copilot and ChatGPT are extensively used in code generation for enterprise and open-source software development and maintenance. Despite their unprecedented successes in code generation, research indicates that codes generated by LLMs exhibit vulnerabilities and security issues. Several studies have been conducted to evaluate code generated by LLMs, considering various aspects such as security, vulnerability, code smells, and robustness. While some studies have compared the performance of LLMs with that of humans in various software engineering tasks, there's a notable gap in research: no studies have directly compared human-written and LLM-generated code for their robustness analysis. To fill this void, this paper introduces an empirical study to evaluate the adversarial robustness of Pre-trained Models of Code (PTMCs) fine-tuned on code written by humans and generated by LLMs against adversarial attacks for software clone detection. These attacks could potentially undermine software security and reliability. We consider two datasets, two state-of-the-art PTMCs, two robustness evaluation criteria, and three metrics to use in our experiments. Regarding effectiveness criteria, PTMCs fine-tuned on human-written code always demonstrate more robustness than those fine-tuned on LLMs-generated code. On the other hand, in terms of adversarial code quality, in 75% experimental combinations, PTMCs fine-tuned on the human-written code exhibit more robustness than the PTMCs fine-tuned on the LLMs-generated code.



## **32. An undetectable watermark for generative image models**

cs.CR

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2410.07369v2) [paper-pdf](http://arxiv.org/pdf/2410.07369v2)

**Authors**: Sam Gunn, Xuandong Zhao, Dawn Song

**Abstract**: We present the first undetectable watermarking scheme for generative image models. Undetectability ensures that no efficient adversary can distinguish between watermarked and un-watermarked images, even after making many adaptive queries. In particular, an undetectable watermark does not degrade image quality under any efficiently computable metric. Our scheme works by selecting the initial latents of a diffusion model using a pseudorandom error-correcting code (Christ and Gunn, 2024), a strategy which guarantees undetectability and robustness. We experimentally demonstrate that our watermarks are quality-preserving and robust using Stable Diffusion 2.1. Our experiments verify that, in contrast to every prior scheme we tested, our watermark does not degrade image quality. Our experiments also demonstrate robustness: existing watermark removal attacks fail to remove our watermark from images without significantly degrading the quality of the images. Finally, we find that we can robustly encode 512 bits in our watermark, and up to 2500 bits when the images are not subjected to watermark removal attacks. Our code is available at https://github.com/XuandongZhao/PRC-Watermark.



## **33. Llama Guard 3 Vision: Safeguarding Human-AI Image Understanding Conversations**

cs.CV

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10414v1) [paper-pdf](http://arxiv.org/pdf/2411.10414v1)

**Authors**: Jianfeng Chi, Ujjwal Karn, Hongyuan Zhan, Eric Smith, Javier Rando, Yiming Zhang, Kate Plawiak, Zacharie Delpierre Coudert, Kartikeya Upasani, Mahesh Pasupuleti

**Abstract**: We introduce Llama Guard 3 Vision, a multimodal LLM-based safeguard for human-AI conversations that involves image understanding: it can be used to safeguard content for both multimodal LLM inputs (prompt classification) and outputs (response classification). Unlike the previous text-only Llama Guard versions (Inan et al., 2023; Llama Team, 2024b,a), it is specifically designed to support image reasoning use cases and is optimized to detect harmful multimodal (text and image) prompts and text responses to these prompts. Llama Guard 3 Vision is fine-tuned on Llama 3.2-Vision and demonstrates strong performance on the internal benchmarks using the MLCommons taxonomy. We also test its robustness against adversarial attacks. We believe that Llama Guard 3 Vision serves as a good starting point to build more capable and robust content moderation tools for human-AI conversation with multimodal capabilities.



## **34. Continual Adversarial Reinforcement Learning (CARL) of False Data Injection detection: forgetting and explainability**

cs.LG

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10367v1) [paper-pdf](http://arxiv.org/pdf/2411.10367v1)

**Authors**: Pooja Aslami, Kejun Chen, Timothy M. Hansen, Malik Hassanaly

**Abstract**: False data injection attacks (FDIAs) on smart inverters are a growing concern linked to increased renewable energy production. While data-based FDIA detection methods are also actively developed, we show that they remain vulnerable to impactful and stealthy adversarial examples that can be crafted using Reinforcement Learning (RL). We propose to include such adversarial examples in data-based detection training procedure via a continual adversarial RL (CARL) approach. This way, one can pinpoint the deficiencies of data-based detection, thereby offering explainability during their incremental improvement. We show that a continual learning implementation is subject to catastrophic forgetting, and additionally show that forgetting can be addressed by employing a joint training strategy on all generated FDIA scenarios.



## **35. Safe Text-to-Image Generation: Simply Sanitize the Prompt Embedding**

cs.CR

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10329v1) [paper-pdf](http://arxiv.org/pdf/2411.10329v1)

**Authors**: Huming Qiu, Guanxu Chen, Mi Zhang, Min Yang

**Abstract**: In recent years, text-to-image (T2I) generation models have made significant progress in generating high-quality images that align with text descriptions. However, these models also face the risk of unsafe generation, potentially producing harmful content that violates usage policies, such as explicit material. Existing safe generation methods typically focus on suppressing inappropriate content by erasing undesired concepts from visual representations, while neglecting to sanitize the textual representation. Although these methods help mitigate the risk of misuse to certain extent, their robustness remains insufficient when dealing with adversarial attacks.   Given that semantic consistency between input text and output image is a fundamental requirement for T2I models, we identify that textual representations (i.e., prompt embeddings) are likely the primary source of unsafe generation. To this end, we propose a vision-agnostic safe generation framework, Embedding Sanitizer (ES), which focuses on erasing inappropriate concepts from prompt embeddings and uses the sanitized embeddings to guide the model for safe generation. ES is applied to the output of the text encoder as a plug-and-play module, enabling seamless integration with different T2I models as well as other safeguards. In addition, ES's unique scoring mechanism assigns a score to each token in the prompt to indicate its potential harmfulness, and dynamically adjusts the sanitization intensity to balance defensive performance and generation quality. Through extensive evaluation on five prompt benchmarks, our approach achieves state-of-the-art robustness by sanitizing the source (prompt embedding) of unsafe generation compared to nine baseline methods. It significantly outperforms existing safeguards in terms of interpretability and controllability while maintaining generation quality.



## **36. MDHP-Net: Detecting Injection Attacks on In-vehicle Network using Multi-Dimensional Hawkes Process and Temporal Model**

cs.CR

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10258v1) [paper-pdf](http://arxiv.org/pdf/2411.10258v1)

**Authors**: Qi Liu, Yanchen Liu, Ruifeng Li, Chenhong Cao, Yufeng Li, Xingyu Li, Peng Wang, Runhan Feng

**Abstract**: The integration of intelligent and connected technologies in modern vehicles, while offering enhanced functionalities through Electronic Control Unit and interfaces like OBD-II and telematics, also exposes the vehicle's in-vehicle network (IVN) to potential cyberattacks. In this paper, we consider a specific type of cyberattack known as the injection attack. As demonstrated by empirical data from real-world cybersecurity adversarial competitions(available at https://mimic2024.xctf.org.cn/race/qwmimic2024 ), these injection attacks have excitation effect over time, gradually manipulating network traffic and disrupting the vehicle's normal functioning, ultimately compromising both its stability and safety. To profile the abnormal behavior of attackers, we propose a novel injection attack detector to extract long-term features of attack behavior. Specifically, we first provide a theoretical analysis of modeling the time-excitation effects of the attack using Multi-Dimensional Hawkes Process (MDHP). A gradient descent solver specifically tailored for MDHP, MDHP-GDS, is developed to accurately estimate optimal MDHP parameters. We then propose an injection attack detector, MDHP-Net, which integrates optimal MDHP parameters with MDHP-LSTM blocks to enhance temporal feature extraction. By introducing MDHP parameters, MDHP-Net captures complex temporal features that standard Long Short-Term Memory (LSTM) cannot, enriching temporal dependencies within our customized structure. Extensive evaluations demonstrate the effectiveness of our proposed detection approach.



## **37. Fault Injection and Safe-Error Attack for Extraction of Embedded Neural Network Models**

cs.CR

Accepted at SECAI Workshop, ESORICS 2023 (v2. Fix notations)

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2308.16703v2) [paper-pdf](http://arxiv.org/pdf/2308.16703v2)

**Authors**: Kevin Hector, Pierre-Alain Moellic, Mathieu Dumont, Jean-Max Dutertre

**Abstract**: Model extraction emerges as a critical security threat with attack vectors exploiting both algorithmic and implementation-based approaches. The main goal of an attacker is to steal as much information as possible about a protected victim model, so that he can mimic it with a substitute model, even with a limited access to similar training data. Recently, physical attacks such as fault injection have shown worrying efficiency against the integrity and confidentiality of embedded models. We focus on embedded deep neural network models on 32-bit microcontrollers, a widespread family of hardware platforms in IoT, and the use of a standard fault injection strategy - Safe Error Attack (SEA) - to perform a model extraction attack with an adversary having a limited access to training data. Since the attack strongly depends on the input queries, we propose a black-box approach to craft a successful attack set. For a classical convolutional neural network, we successfully recover at least 90% of the most significant bits with about 1500 crafted inputs. These information enable to efficiently train a substitute model, with only 8% of the training dataset, that reaches high fidelity and near identical accuracy level than the victim model.



## **38. A Hard-Label Cryptanalytic Extraction of Non-Fully Connected Deep Neural Networks using Side-Channel Attacks**

cs.CR

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10174v1) [paper-pdf](http://arxiv.org/pdf/2411.10174v1)

**Authors**: Benoit Coqueret, Mathieu Carbone, Olivier Sentieys, Gabriel Zaid

**Abstract**: During the past decade, Deep Neural Networks (DNNs) proved their value on a large variety of subjects. However despite their high value and public accessibility, the protection of the intellectual property of DNNs is still an issue and an emerging research field. Recent works have successfully extracted fully-connected DNNs using cryptanalytic methods in hard-label settings, proving that it was possible to copy a DNN with high fidelity, i.e., high similitude in the output predictions. However, the current cryptanalytic attacks cannot target complex, i.e., not fully connected, DNNs and are limited to special cases of neurons present in deep networks.   In this work, we introduce a new end-to-end attack framework designed for model extraction of embedded DNNs with high fidelity. We describe a new black-box side-channel attack which splits the DNN in several linear parts for which we can perform cryptanalytic extraction and retrieve the weights in hard-label settings. With this method, we are able to adapt cryptanalytic extraction, for the first time, to non-fully connected DNNs, while maintaining a high fidelity. We validate our contributions by targeting several architectures implemented on a microcontroller unit, including a Multi-Layer Perceptron (MLP) of 1.7 million parameters and a shortened MobileNetv1. Our framework successfully extracts all of these DNNs with high fidelity (88.4% for the MobileNetv1 and 93.2% for the MLP). Furthermore, we use the stolen model to generate adversarial examples and achieve close to white-box performance on the victim's model (95.8% and 96.7% transfer rate).



## **39. Adversarial Robustness of VAEs across Intersectional Subgroups**

cs.LG

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2407.03864v2) [paper-pdf](http://arxiv.org/pdf/2407.03864v2)

**Authors**: Chethan Krishnamurthy Ramanaik, Arjun Roy, Eirini Ntoutsi

**Abstract**: Despite advancements in Autoencoders (AEs) for tasks like dimensionality reduction, representation learning and data generation, they remain vulnerable to adversarial attacks. Variational Autoencoders (VAEs), with their probabilistic approach to disentangling latent spaces, show stronger resistance to such perturbations compared to deterministic AEs; however, their resilience against adversarial inputs is still a concern. This study evaluates the robustness of VAEs against non-targeted adversarial attacks by optimizing minimal sample-specific perturbations to cause maximal damage across diverse demographic subgroups (combinations of age and gender). We investigate two questions: whether there are robustness disparities among subgroups, and what factors contribute to these disparities, such as data scarcity and representation entanglement. Our findings reveal that robustness disparities exist but are not always correlated with the size of the subgroup. By using downstream gender and age classifiers and examining latent embeddings, we highlight the vulnerability of subgroups like older women, who are prone to misclassification due to adversarial perturbations pushing their representations toward those of other subgroups.



## **40. Edge-Only Universal Adversarial Attacks in Distributed Learning**

cs.CR

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10500v1) [paper-pdf](http://arxiv.org/pdf/2411.10500v1)

**Authors**: Giulio Rossolini, Tommaso Baldi, Alessandro Biondi, Giorgio Buttazzo

**Abstract**: Distributed learning frameworks, which partition neural network models across multiple computing nodes, enhance efficiency in collaborative edge-cloud systems but may also introduce new vulnerabilities. In this work, we explore the feasibility of generating universal adversarial attacks when an attacker has access to the edge part of the model only, which consists in the first network layers. Unlike traditional universal adversarial perturbations (UAPs) that require full model knowledge, our approach shows that adversaries can induce effective mispredictions in the unknown cloud part by leveraging key features on the edge side. Specifically, we train lightweight classifiers from intermediate features available at the edge, i.e., before the split point, and use them in a novel targeted optimization to craft effective UAPs. Our results on ImageNet demonstrate strong attack transferability to the unknown cloud part. Additionally, we analyze the capability of an attacker to achieve targeted adversarial effect with edge-only knowledge, revealing intriguing behaviors. By introducing the first adversarial attacks with edge-only knowledge in split inference, this work underscores the importance of addressing partial model access in adversarial robustness, encouraging further research in this area.



## **41. Prompt-Guided Environmentally Consistent Adversarial Patch**

cs.CV

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10498v1) [paper-pdf](http://arxiv.org/pdf/2411.10498v1)

**Authors**: Chaoqun Li, Huanqian Yan, Lifeng Zhou, Tairan Chen, Zhuodong Liu, Hang Su

**Abstract**: Adversarial attacks in the physical world pose a significant threat to the security of vision-based systems, such as facial recognition and autonomous driving. Existing adversarial patch methods primarily focus on improving attack performance, but they often produce patches that are easily detectable by humans and struggle to achieve environmental consistency, i.e., blending patches into the environment. This paper introduces a novel approach for generating adversarial patches, which addresses both the visual naturalness and environmental consistency of the patches. We propose Prompt-Guided Environmentally Consistent Adversarial Patch (PG-ECAP), a method that aligns the patch with the environment to ensure seamless integration into the environment. The approach leverages diffusion models to generate patches that are both environmental consistency and effective in evading detection. To further enhance the naturalness and consistency, we introduce two alignment losses: Prompt Alignment Loss and Latent Space Alignment Loss, ensuring that the generated patch maintains its adversarial properties while fitting naturally within its environment. Extensive experiments in both digital and physical domains demonstrate that PG-ECAP outperforms existing methods in attack success rate and environmental consistency.



## **42. Self-Defense: Optimal QIF Solutions and Application to Website Fingerprinting**

cs.CR

38th IEEE Computer Security Foundations Symposium, IEEE, Jun 2025,  Santa Cruz, United States

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10059v1) [paper-pdf](http://arxiv.org/pdf/2411.10059v1)

**Authors**: Andreas Athanasiou, Konstantinos Chatzikokolakis, Catuscia Palamidessi

**Abstract**: Quantitative Information Flow (QIF) provides a robust information-theoretical framework for designing secure systems with minimal information leakage. While previous research has addressed the design of such systems under hard constraints (e.g. application limitations) and soft constraints (e.g. utility), scenarios often arise where the core system's behavior is considered fixed. In such cases, the challenge is to design a new component for the existing system that minimizes leakage without altering the original system. In this work we address this problem by proposing optimal solutions for constructing a new row, in a known and unmodifiable information-theoretic channel, aiming at minimizing the leakage. We first model two types of adversaries: an exact-guessing adversary, aiming to guess the secret in one try, and a s-distinguishing one, which tries to distinguish the secret s from all the other secrets.Then, we discuss design strategies for both fixed and unknown priors by offering, for each adversary, an optimal solution under linear constraints, using Linear Programming.We apply our approach to the problem of website fingerprinting defense, considering a scenario where a site administrator can modify their own site but not others. We experimentally evaluate our proposed solutions against other natural approaches. First, we sample real-world news websites and then, for both adversaries, we demonstrate that the proposed solutions are effective in achieving the least leakage. Finally, we simulate an actual attack by training an ML classifier for the s-distinguishing adversary and show that our approach decreases the accuracy of the attacker.



## **43. EveGuard: Defeating Vibration-based Side-Channel Eavesdropping with Audio Adversarial Perturbations**

cs.CR

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10034v1) [paper-pdf](http://arxiv.org/pdf/2411.10034v1)

**Authors**: Jung-Woo Chang, Ke Sun, David Xia, Xinyu Zhang, Farinaz Koushanfar

**Abstract**: Vibrometry-based side channels pose a significant privacy risk, exploiting sensors like mmWave radars, light sensors, and accelerometers to detect vibrations from sound sources or proximate objects, enabling speech eavesdropping. Despite various proposed defenses, these involve costly hardware solutions with inherent physical limitations. This paper presents EveGuard, a software-driven defense framework that creates adversarial audio, protecting voice privacy from side channels without compromising human perception. We leverage the distinct sensing capabilities of side channels and traditional microphones where side channels capture vibrations and microphones record changes in air pressure, resulting in different frequency responses. EveGuard first proposes a perturbation generator model (PGM) that effectively suppresses sensor-based eavesdropping while maintaining high audio quality. Second, to enable end-to-end training of PGM, we introduce a new domain translation task called Eve-GAN for inferring an eavesdropped signal from a given audio. We further apply few-shot learning to mitigate the data collection overhead for Eve-GAN training. Our extensive experiments show that EveGuard achieves a protection rate of more than 97 percent from audio classifiers and significantly hinders eavesdropped audio reconstruction. We further validate the performance of EveGuard across three adaptive attack mechanisms. We have conducted a user study to verify the perceptual quality of our perturbed audio.



## **44. Toward Robust and Accurate Adversarial Camouflage Generation against Vehicle Detectors**

cs.CV

14 pages. arXiv admin note: substantial text overlap with  arXiv:2402.15853

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10029v1) [paper-pdf](http://arxiv.org/pdf/2411.10029v1)

**Authors**: Jiawei Zhou, Linye Lyu, Daojing He, Yu Li

**Abstract**: Adversarial camouflage is a widely used physical attack against vehicle detectors for its superiority in multi-view attack performance. One promising approach involves using differentiable neural renderers to facilitate adversarial camouflage optimization through gradient back-propagation. However, existing methods often struggle to capture environmental characteristics during the rendering process or produce adversarial textures that can precisely map to the target vehicle. Moreover, these approaches neglect diverse weather conditions, reducing the efficacy of generated camouflage across varying weather scenarios. To tackle these challenges, we propose a robust and accurate camouflage generation method, namely RAUCA. The core of RAUCA is a novel neural rendering component, End-to-End Neural Renderer Plus (E2E-NRP), which can accurately optimize and project vehicle textures and render images with environmental characteristics such as lighting and weather. In addition, we integrate a multi-weather dataset for camouflage generation, leveraging the E2E-NRP to enhance the attack robustness. Experimental results on six popular object detectors show that RAUCA-final outperforms existing methods in both simulation and real-world settings.



## **45. Provably Unlearnable Data Examples**

cs.LG

Accepted to Network and Distributed System Security (NDSS) Symposium  2025, San Diego, CA, USA. Source code is available at  https://github.com/NeuralSec/certified-data-learnability

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2405.03316v2) [paper-pdf](http://arxiv.org/pdf/2405.03316v2)

**Authors**: Derui Wang, Minhui Xue, Bo Li, Seyit Camtepe, Liming Zhu

**Abstract**: The exploitation of publicly accessible data has led to escalating concerns regarding data privacy and intellectual property (IP) breaches in the age of artificial intelligence. To safeguard both data privacy and IP-related domain knowledge, efforts have been undertaken to render shared data unlearnable for unauthorized models in the wild. Existing methods apply empirically optimized perturbations to the data in the hope of disrupting the correlation between the inputs and the corresponding labels such that the data samples are converted into Unlearnable Examples (UEs). Nevertheless, the absence of mechanisms to verify the robustness of UEs against uncertainty in unauthorized models and their training procedures engenders several under-explored challenges. First, it is hard to quantify the unlearnability of UEs against unauthorized adversaries from different runs of training, leaving the soundness of the defense in obscurity. Particularly, as a prevailing evaluation metric, empirical test accuracy faces generalization errors and may not plausibly represent the quality of UEs. This also leaves room for attackers, as there is no rigid guarantee of the maximal test accuracy achievable by attackers. Furthermore, we find that a simple recovery attack can restore the clean-task performance of the classifiers trained on UEs by slightly perturbing the learned weights. To mitigate the aforementioned problems, in this paper, we propose a mechanism for certifying the so-called $(q, \eta)$-Learnability of an unlearnable dataset via parametric smoothing. A lower certified $(q, \eta)$-Learnability indicates a more robust and effective protection over the dataset. Concretely, we 1) improve the tightness of certified $(q, \eta)$-Learnability and 2) design Provably Unlearnable Examples (PUEs) which have reduced $(q, \eta)$-Learnability.



## **46. Confidence-aware Denoised Fine-tuning of Off-the-shelf Models for Certified Robustness**

cs.CV

26 pages; TMLR 2024; Code is available at  https://github.com/suhyeok24/FT-CADIS

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.08933v2) [paper-pdf](http://arxiv.org/pdf/2411.08933v2)

**Authors**: Suhyeok Jang, Seojin Kim, Jinwoo Shin, Jongheon Jeong

**Abstract**: The remarkable advances in deep learning have led to the emergence of many off-the-shelf classifiers, e.g., large pre-trained models. However, since they are typically trained on clean data, they remain vulnerable to adversarial attacks. Despite this vulnerability, their superior performance and transferability make off-the-shelf classifiers still valuable in practice, demanding further work to provide adversarial robustness for them in a post-hoc manner. A recently proposed method, denoised smoothing, leverages a denoiser model in front of the classifier to obtain provable robustness without additional training. However, the denoiser often creates hallucination, i.e., images that have lost the semantics of their originally assigned class, leading to a drop in robustness. Furthermore, its noise-and-denoise procedure introduces a significant distribution shift from the original distribution, causing the denoised smoothing framework to achieve sub-optimal robustness. In this paper, we introduce Fine-Tuning with Confidence-Aware Denoised Image Selection (FT-CADIS), a novel fine-tuning scheme to enhance the certified robustness of off-the-shelf classifiers. FT-CADIS is inspired by the observation that the confidence of off-the-shelf classifiers can effectively identify hallucinated images during denoised smoothing. Based on this, we develop a confidence-aware training objective to handle such hallucinated images and improve the stability of fine-tuning from denoised images. In this way, the classifier can be fine-tuned using only images that are beneficial for adversarial robustness. We also find that such a fine-tuning can be done by updating a small fraction of parameters of the classifier. Extensive experiments demonstrate that FT-CADIS has established the state-of-the-art certified robustness among denoised smoothing methods across all $\ell_2$-adversary radius in various benchmarks.



## **47. IDEATOR: Jailbreaking Large Vision-Language Models Using Themselves**

cs.CV

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.00827v2) [paper-pdf](http://arxiv.org/pdf/2411.00827v2)

**Authors**: Ruofan Wang, Bo Wang, Xiaosen Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: As large Vision-Language Models (VLMs) grow in prominence, ensuring their safe deployment has become critical. Recent studies have explored VLM robustness against jailbreak attacks--techniques that exploit model vulnerabilities to elicit harmful outputs. However, the limited availability of diverse multi-modal data has led current approaches to rely heavily on adversarial or manually crafted images derived from harmful text datasets, which may lack effectiveness and diversity across different contexts. In this paper, we propose a novel jailbreak method named IDEATOR, which autonomously generates malicious image-text pairs for black-box jailbreak attacks. IDEATOR is based on the insight that VLMs themselves could serve as powerful red team models for generating multimodal jailbreak prompts. Specifically, IDEATOR uses a VLM to create targeted jailbreak texts and pairs them with jailbreak images generated by a state-of-the-art diffusion model. Our extensive experiments demonstrate IDEATOR's high effectiveness and transferability. Notably, it achieves a 94% success rate in jailbreaking MiniGPT-4 with an average of only 5.34 queries, and high success rates of 82%, 88%, and 75% when transferred to LLaVA, InstructBLIP, and Meta's Chameleon, respectively. IDEATOR uncovers specific vulnerabilities in VLMs under black-box conditions, underscoring the need for improved safety mechanisms.



## **48. Adversarial Attacks Using Differentiable Rendering: A Survey**

cs.LG

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2411.09749v1) [paper-pdf](http://arxiv.org/pdf/2411.09749v1)

**Authors**: Matthew Hull, Chao Zhang, Zsolt Kira, Duen Horng Chau

**Abstract**: Differentiable rendering methods have emerged as a promising means for generating photo-realistic and physically plausible adversarial attacks by manipulating 3D objects and scenes that can deceive deep neural networks (DNNs). Recently, differentiable rendering capabilities have evolved significantly into a diverse landscape of libraries, such as Mitsuba, PyTorch3D, and methods like Neural Radiance Fields and 3D Gaussian Splatting for solving inverse rendering problems that share conceptually similar properties commonly used to attack DNNs, such as back-propagation and optimization. However, the adversarial machine learning research community has not yet fully explored or understood such capabilities for generating attacks. Some key reasons are that researchers often have different attack goals, such as misclassification or misdetection, and use different tasks to accomplish these goals by manipulating different representation in a scene, such as the mesh or texture of an object. This survey adopts a task-oriented unifying framework that systematically summarizes common tasks, such as manipulating textures, altering illumination, and modifying 3D meshes to exploit vulnerabilities in DNNs. Our framework enables easy comparison of existing works, reveals research gaps and spotlights exciting future research directions in this rapidly evolving field. Through focusing on how these tasks enable attacks on various DNNs such as image classification, facial recognition, object detection, optical flow and depth estimation, our survey helps researchers and practitioners better understand the vulnerabilities of computer vision systems against photorealistic adversarial attacks that could threaten real-world applications.



## **49. DiffPAD: Denoising Diffusion-based Adversarial Patch Decontamination**

cs.CV

Accepted to 2025 IEEE/CVF Winter Conference on Applications of  Computer Vision (WACV)

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2410.24006v2) [paper-pdf](http://arxiv.org/pdf/2410.24006v2)

**Authors**: Jia Fu, Xiao Zhang, Sepideh Pashami, Fatemeh Rahimian, Anders Holst

**Abstract**: In the ever-evolving adversarial machine learning landscape, developing effective defenses against patch attacks has become a critical challenge, necessitating reliable solutions to safeguard real-world AI systems. Although diffusion models have shown remarkable capacity in image synthesis and have been recently utilized to counter $\ell_p$-norm bounded attacks, their potential in mitigating localized patch attacks remains largely underexplored. In this work, we propose DiffPAD, a novel framework that harnesses the power of diffusion models for adversarial patch decontamination. DiffPAD first performs super-resolution restoration on downsampled input images, then adopts binarization, dynamic thresholding scheme and sliding window for effective localization of adversarial patches. Such a design is inspired by the theoretically derived correlation between patch size and diffusion restoration error that is generalized across diverse patch attack scenarios. Finally, DiffPAD applies inpainting techniques to the original input images with the estimated patch region being masked. By integrating closed-form solutions for super-resolution restoration and image inpainting into the conditional reverse sampling process of a pre-trained diffusion model, DiffPAD obviates the need for text guidance or fine-tuning. Through comprehensive experiments, we demonstrate that DiffPAD not only achieves state-of-the-art adversarial robustness against patch attacks but also excels in recovering naturalistic images without patch remnants. The source code is available at https://github.com/JasonFu1998/DiffPAD.



## **50. Are nuclear masks all you need for improved out-of-domain generalisation? A closer look at cancer classification in histopathology**

eess.IV

Poster at NeurIPS 2024

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2411.09373v1) [paper-pdf](http://arxiv.org/pdf/2411.09373v1)

**Authors**: Dhananjay Tomar, Alexander Binder, Andreas Kleppe

**Abstract**: Domain generalisation in computational histopathology is challenging because the images are substantially affected by differences among hospitals due to factors like fixation and staining of tissue and imaging equipment. We hypothesise that focusing on nuclei can improve the out-of-domain (OOD) generalisation in cancer detection. We propose a simple approach to improve OOD generalisation for cancer detection by focusing on nuclear morphology and organisation, as these are domain-invariant features critical in cancer detection. Our approach integrates original images with nuclear segmentation masks during training, encouraging the model to prioritise nuclei and their spatial arrangement. Going beyond mere data augmentation, we introduce a regularisation technique that aligns the representations of masks and original images. We show, using multiple datasets, that our method improves OOD generalisation and also leads to increased robustness to image corruptions and adversarial attacks. The source code is available at https://github.com/undercutspiky/SFL/



