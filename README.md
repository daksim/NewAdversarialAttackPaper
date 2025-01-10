# Latest Adversarial Attack Papers
**update at 2025-01-10 09:46:28**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Correlated Privacy Mechanisms for Differentially Private Distributed Mean Estimation**

cs.IT

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2407.03289v2) [paper-pdf](http://arxiv.org/pdf/2407.03289v2)

**Authors**: Sajani Vithana, Viveck R. Cadambe, Flavio P. Calmon, Haewon Jeong

**Abstract**: Differentially private distributed mean estimation (DP-DME) is a fundamental building block in privacy-preserving federated learning, where a central server estimates the mean of $d$-dimensional vectors held by $n$ users while ensuring $(\epsilon,\delta)$-DP. Local differential privacy (LDP) and distributed DP with secure aggregation (SA) are the most common notions of DP used in DP-DME settings with an untrusted server. LDP provides strong resilience to dropouts, colluding users, and adversarial attacks, but suffers from poor utility. In contrast, SA-based DP-DME achieves an $O(n)$ utility gain over LDP in DME, but requires increased communication and computation overheads and complex multi-round protocols to handle dropouts and attacks. In this work, we present a generalized framework for DP-DME, that captures LDP and SA-based mechanisms as extreme cases. Our framework provides a foundation for developing and analyzing a variety of DP-DME protocols that leverage correlated privacy mechanisms across users. To this end, we propose CorDP-DME, a novel DP-DME mechanism based on the correlated Gaussian mechanism, that spans the gap between DME with LDP and distributed DP. We prove that CorDP-DME offers a favorable balance between utility and resilience to dropout and collusion. We provide an information-theoretic analysis of CorDP-DME, and derive theoretical guarantees for utility under any given privacy parameters and dropout/colluding user thresholds. Our results demonstrate that (anti) correlated Gaussian DP mechanisms can significantly improve utility in mean estimation tasks compared to LDP -- even in adversarial settings -- while maintaining better resilience to dropouts and attacks compared to distributed DP.



## **2. Resilient Peer-to-peer Learning based on Adaptive Aggregation**

cs.LG

11 pages

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.04610v1) [paper-pdf](http://arxiv.org/pdf/2501.04610v1)

**Authors**: Chandreyee Bhowmick, Xenofon Koutsoukos

**Abstract**: Collaborative learning in peer-to-peer networks offers the benefits of distributed learning while mitigating the risks associated with single points of failure inherent in centralized servers. However, adversarial workers pose potential threats by attempting to inject malicious information into the network. Thus, ensuring the resilience of peer-to-peer learning emerges as a pivotal research objective. The challenge is exacerbated in the presence of non-convex loss functions and non-iid data distributions. This paper introduces a resilient aggregation technique tailored for such scenarios, aimed at fostering similarity among peers' learning processes. The aggregation weights are determined through an optimization procedure, and use the loss function computed using the neighbor's models and individual private data, thereby addressing concerns regarding data privacy in distributed machine learning. Theoretical analysis demonstrates convergence of parameters with non-convex loss functions and non-iid data distributions. Empirical evaluations across three distinct machine learning tasks support the claims. The empirical findings, which encompass a range of diverse attack models, also demonstrate improved accuracy when compared to existing methodologies.



## **3. Tougher Text, Smarter Models: Raising the Bar for Adversarial Defence Benchmarks**

cs.CL

Will be presented as an oral in-person presentation at the conference  of COLING 2025

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.02654v2) [paper-pdf](http://arxiv.org/pdf/2501.02654v2)

**Authors**: Yang Wang, Chenghua Lin

**Abstract**: Recent advancements in natural language processing have highlighted the vulnerability of deep learning models to adversarial attacks. While various defence mechanisms have been proposed, there is a lack of comprehensive benchmarks that evaluate these defences across diverse datasets, models, and tasks. In this work, we address this gap by presenting an extensive benchmark for textual adversarial defence that significantly expands upon previous work. Our benchmark incorporates a wide range of datasets, evaluates state-of-the-art defence mechanisms, and extends the assessment to include critical tasks such as single-sentence classification, similarity and paraphrase identification, natural language inference, and commonsense reasoning. This work not only serves as a valuable resource for researchers and practitioners in the field of adversarial robustness but also identifies key areas for future research in textual adversarial defence. By establishing a new standard for benchmarking in this domain, we aim to accelerate progress towards more robust and reliable natural language processing systems.



## **4. Towards Fair Class-wise Robustness: Class Optimal Distribution Adversarial Training**

cs.LG

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.04527v1) [paper-pdf](http://arxiv.org/pdf/2501.04527v1)

**Authors**: Hongxin Zhi, Hongtao Yu, Shaome Li, Xiuming Zhao, Yiteng Wu

**Abstract**: Adversarial training has proven to be a highly effective method for improving the robustness of deep neural networks against adversarial attacks. Nonetheless, it has been observed to exhibit a limitation in terms of robust fairness, characterized by a significant disparity in robustness across different classes. Recent efforts to mitigate this problem have turned to class-wise reweighted methods. However, these methods suffer from a lack of rigorous theoretical analysis and are limited in their exploration of the weight space, as they mainly rely on existing heuristic algorithms or intuition to compute weights. In addition, these methods fail to guarantee the consistency of the optimization direction due to the decoupled optimization of weights and the model parameters. They potentially lead to suboptimal weight assignments and consequently, a suboptimal model. To address these problems, this paper proposes a novel min-max training framework, Class Optimal Distribution Adversarial Training (CODAT), which employs distributionally robust optimization to fully explore the class-wise weight space, thus enabling the identification of the optimal weight with theoretical guarantees. Furthermore, we derive a closed-form optimal solution to the internal maximization and then get a deterministic equivalent objective function, which provides a theoretical basis for the joint optimization of weights and model parameters. Meanwhile, we propose a fairness elasticity coefficient for the evaluation of the algorithm with regard to both robustness and robust fairness. Experimental results on various datasets show that the proposed method can effectively improve the robust fairness of the model and outperform the state-of-the-art approaches.



## **5. Multichannel Steganography: A Provably Secure Hybrid Steganographic Model for Secure Communication**

cs.CR

18 pages, 8 figures, 3 algorithms, This version is a preprint  uploaded to arXiv

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.04511v1) [paper-pdf](http://arxiv.org/pdf/2501.04511v1)

**Authors**: Obinna Omego, Michal Bosy

**Abstract**: This study introduces a novel steganographic model that synthesizes Steganography by Cover Modification (CMO) and Steganography by Cover Synthesis (CSY), enhancing both security and undetectability by generating cover messages or parameters while retaining the original cover's form, thus minimizing detection risks and overcoming the limitations of single-method techniques. Building upon this model, a refined Steganographic Communication Protocol is proposed, enhancing resilience against sophisticated threats such as Multichannel Replay Attacks and Multichannel Man-in-the-Middle Attacks, fortifying the protocol against potential tampering and improving upon prior works. To evaluate the security of the proposed protocol, a novel adversarial model is developed simulating a probabilistic polynomial time (PPT) adversary capable of intercepting communications across multiple channels. This model assesses the adversary's ability to compromise the protocol, providing a comprehensive security analysis. Finally, this study explores the practicality and adaptability of the model to both constrained environments like SMS banking and resource-rich settings such as blockchain transactions, demonstrating their potential to enhance financial services and security. These contributions present a robust and adaptable framework for secure steganographic communication, offering practical solutions for secure communications across diverse environments.



## **6. Rethinking Byzantine Robustness in Federated Recommendation from Sparse Aggregation Perspective**

cs.CR

accepted by AAAI 2025

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.03301v2) [paper-pdf](http://arxiv.org/pdf/2501.03301v2)

**Authors**: Zhongjian Zhang, Mengmei Zhang, Xiao Wang, Lingjuan Lyu, Bo Yan, Junping Du, Chuan Shi

**Abstract**: To preserve user privacy in recommender systems, federated recommendation (FR) based on federated learning (FL) emerges, keeping the personal data on the local client and updating a model collaboratively. Unlike FL, FR has a unique sparse aggregation mechanism, where the embedding of each item is updated by only partial clients, instead of full clients in a dense aggregation of general FL. Recently, as an essential principle of FL, model security has received increasing attention, especially for Byzantine attacks, where malicious clients can send arbitrary updates. The problem of exploring the Byzantine robustness of FR is particularly critical since in the domains applying FR, e.g., e-commerce, malicious clients can be injected easily by registering new accounts. However, existing Byzantine works neglect the unique sparse aggregation of FR, making them unsuitable for our problem. Thus, we make the first effort to investigate Byzantine attacks on FR from the perspective of sparse aggregation, which is non-trivial: it is not clear how to define Byzantine robustness under sparse aggregations and design Byzantine attacks under limited knowledge/capability. In this paper, we reformulate the Byzantine robustness under sparse aggregation by defining the aggregation for a single item as the smallest execution unit. Then we propose a family of effective attack strategies, named Spattack, which exploit the vulnerability in sparse aggregation and are categorized along the adversary's knowledge and capability. Extensive experimental results demonstrate that Spattack can effectively prevent convergence and even break down defenses under a few malicious clients, raising alarms for securing FR systems.



## **7. Rethinking Adversarial Attacks in Reinforcement Learning from Policy Distribution Perspective**

cs.LG

10 pages, 2 figures, 2 tables

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.03562v2) [paper-pdf](http://arxiv.org/pdf/2501.03562v2)

**Authors**: Tianyang Duan, Zongyuan Zhang, Zheng Lin, Yue Gao, Ling Xiong, Yong Cui, Hongbin Liang, Xianhao Chen, Heming Cui, Dong Huang

**Abstract**: Deep Reinforcement Learning (DRL) suffers from uncertainties and inaccuracies in the observation signal in realworld applications. Adversarial attack is an effective method for evaluating the robustness of DRL agents. However, existing attack methods targeting individual sampled actions have limited impacts on the overall policy distribution, particularly in continuous action spaces. To address these limitations, we propose the Distribution-Aware Projected Gradient Descent attack (DAPGD). DAPGD uses distribution similarity as the gradient perturbation input to attack the policy network, which leverages the entire policy distribution rather than relying on individual samples. We utilize the Bhattacharyya distance in DAPGD to measure policy similarity, enabling sensitive detection of subtle but critical differences between probability distributions. Our experiment results demonstrate that DAPGD achieves SOTA results compared to the baselines in three robot navigation tasks, achieving an average 22.03% higher reward drop compared to the best baseline.



## **8. Location Privacy Threats and Protections in 6G Vehicular Networks: A Comprehensive Review**

cs.CR

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2305.04503v2) [paper-pdf](http://arxiv.org/pdf/2305.04503v2)

**Authors**: Baihe Ma, Xu Wang, Xiaojie Lin, Yanna Jiang, Caijun Sun, Zhe Wang, Guangsheng Yu, Suirui Zhu, Ying He, Wei Ni, Ren Ping Liu

**Abstract**: Location privacy is critical in vehicular networks, where drivers' trajectories and personal information can be exposed, allowing adversaries to launch data and physical attacks that threaten drivers' safety and personal security. This survey reviews comprehensively different localization techniques, including widely used ones like sensing infrastructure-based, optical vision-based, and cellular radio-based localization, and identifies inadequately addressed location privacy concerns. We classify Location Privacy Preserving Mechanisms (LPPMs) into user-side, server-side, and user-server-interface-based, and evaluate their effectiveness. Our analysis shows that the user-server-interface-based LPPMs have received insufficient attention in the literature, despite their paramount importance in vehicular networks. Further, we examine methods for balancing data utility and privacy protection for existing LPPMs in vehicular networks and highlight emerging challenges from future upper-layer location privacy attacks, wireless technologies, and network convergences. By providing insights into the relationship between localization techniques and location privacy, and evaluating the effectiveness of different LPPMs, this survey can help inform the development of future LPPMs in vehicular networks.



## **9. Proof-of-Learning with Incentive Security**

cs.CR

20 pages, 4 figures

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2404.09005v7) [paper-pdf](http://arxiv.org/pdf/2404.09005v7)

**Authors**: Zishuo Zhao, Zhixuan Fang, Xuechao Wang, Xi Chen, Hongxu Su, Haibo Xiao, Yuan Zhou

**Abstract**: Most concurrent blockchain systems rely heavily on the Proof-of-Work (PoW) or Proof-of-Stake (PoS) mechanisms for decentralized consensus and security assurance. However, the substantial energy expenditure stemming from computationally intensive yet meaningless tasks has raised considerable concerns surrounding traditional PoW approaches, The PoS mechanism, while free of energy consumption, is subject to security and economic issues. Addressing these issues, the paradigm of Proof-of-Useful-Work (PoUW) seeks to employ challenges of practical significance as PoW, thereby imbuing energy consumption with tangible value. While previous efforts in Proof of Learning (PoL) explored the utilization of deep learning model training SGD tasks as PoUW challenges, recent research has revealed its vulnerabilities to adversarial attacks and the theoretical hardness in crafting a byzantine-secure PoL mechanism. In this paper, we introduce the concept of incentive-security that incentivizes rational provers to behave honestly for their best interest, bypassing the existing hardness to design a PoL mechanism with computational efficiency, a provable incentive-security guarantee and controllable difficulty. Particularly, our work is secure against two attacks, and also improves the computational overhead from $\Theta(1)$ to $O(\frac{\log E}{E})$. Furthermore, while most recent research assumes trusted problem providers and verifiers, our design also guarantees frontend incentive-security even when problem providers are untrusted, and verifier incentive-security that bypasses the Verifier's Dilemma. By incorporating ML training into blockchain consensus mechanisms with provable guarantees, our research not only proposes an eco-friendly solution to blockchain systems, but also provides a proposal for a completely decentralized computing power market in the new AI age.



## **10. Light-weight Fine-tuning Method for Defending Adversarial Noise in Pre-trained Medical Vision-Language Models**

cs.CV

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2407.02716v2) [paper-pdf](http://arxiv.org/pdf/2407.02716v2)

**Authors**: Xu Han, Linghao Jin, Xuezhe Ma, Xiaofeng Liu

**Abstract**: Fine-tuning pre-trained Vision-Language Models (VLMs) has shown remarkable capabilities in medical image and textual depiction synergy. Nevertheless, many pre-training datasets are restricted by patient privacy concerns, potentially containing noise that can adversely affect downstream performance. Moreover, the growing reliance on multi-modal generation exacerbates this issue because of its susceptibility to adversarial attacks. To investigate how VLMs trained on adversarial noisy data perform on downstream medical tasks, we first craft noisy upstream datasets using multi-modal adversarial attacks. Through our comprehensive analysis, we unveil that moderate noise enhances model robustness and transferability, but increasing noise levels negatively impact downstream task performance. To mitigate this issue, we propose rectify adversarial noise (RAN) framework, a recipe designed to effectively defend adversarial attacks and rectify the influence of upstream noise during fine-tuning.



## **11. Synthetic Data Privacy Metrics**

cs.LG

14 pages, 2 figures

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.03941v1) [paper-pdf](http://arxiv.org/pdf/2501.03941v1)

**Authors**: Amy Steier, Lipika Ramaswamy, Andre Manoel, Alexa Haushalter

**Abstract**: Recent advancements in generative AI have made it possible to create synthetic datasets that can be as accurate as real-world data for training AI models, powering statistical insights, and fostering collaboration with sensitive datasets while offering strong privacy guarantees. Effectively measuring the empirical privacy of synthetic data is an important step in the process. However, while there is a multitude of new privacy metrics being published every day, there currently is no standardization. In this paper, we review the pros and cons of popular metrics that include simulations of adversarial attacks. We also review current best practices for amending generative models to enhance the privacy of the data they create (e.g. differential privacy).



## **12. Not all tokens are created equal: Perplexity Attention Weighted Networks for AI generated text detection**

cs.CL

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.03940v1) [paper-pdf](http://arxiv.org/pdf/2501.03940v1)

**Authors**: Pablo Miralles-González, Javier Huertas-Tato, Alejandro Martín, David Camacho

**Abstract**: The rapid advancement in large language models (LLMs) has significantly enhanced their ability to generate coherent and contextually relevant text, raising concerns about the misuse of AI-generated content and making it critical to detect it. However, the task remains challenging, particularly in unseen domains or with unfamiliar LLMs. Leveraging LLM next-token distribution outputs offers a theoretically appealing approach for detection, as they encapsulate insights from the models' extensive pre-training on diverse corpora. Despite its promise, zero-shot methods that attempt to operationalize these outputs have met with limited success. We hypothesize that one of the problems is that they use the mean to aggregate next-token distribution metrics across tokens, when some tokens are naturally easier or harder to predict and should be weighted differently. Based on this idea, we propose the Perplexity Attention Weighted Network (PAWN), which uses the last hidden states of the LLM and positions to weight the sum of a series of features based on metrics from the next-token distribution across the sequence length. Although not zero-shot, our method allows us to cache the last hidden states and next-token distribution metrics on disk, greatly reducing the training resource requirements. PAWN shows competitive and even better performance in-distribution than the strongest baselines (fine-tuned LMs) with a fraction of their trainable parameters. Our model also generalizes better to unseen domains and source models, with smaller variability in the decision boundary across distribution shifts. It is also more robust to adversarial attacks, and if the backbone has multilingual capabilities, it presents decent generalization to languages not seen during supervised training, with LLaMA3-1B reaching a mean macro-averaged F1 score of 81.46% in cross-validation with nine languages.



## **13. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2410.23091v6) [paper-pdf](http://arxiv.org/pdf/2410.23091v6)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark). The code is available at https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff.



## **14. A Volumetric Approach to Privacy of Dynamical Systems**

eess.SY

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.02893v2) [paper-pdf](http://arxiv.org/pdf/2501.02893v2)

**Authors**: Chuanghong Weng, Ehsan Nekouei

**Abstract**: Information-theoretic metrics, such as mutual information, have been widely used to evaluate privacy leakage in dynamic systems. However, these approaches are typically limited to stochastic systems and face computational challenges. In this paper, we introduce a novel volumetric framework for analyzing privacy in systems affected by unknown but bounded noise. Our model considers a dynamic system comprising public and private states, where an observation set of the public state is released. An adversary utilizes the observed public state to infer an uncertainty set of the private state, referred to as the inference attack. We define the evolution dynamics of these inference attacks and quantify the privacy level of the private state using the volume of its uncertainty sets. For linear scalar systems, we derive an explicit formulation of the uncertainty set. For multi-dimensional linear systems, we develop an approximate computation method leveraging interval analysis. We investigate the properties of the proposed volumetric privacy measure and demonstrate that it is bounded by the information gain derived from the observation set. Furthermore, we propose an optimization approach to designing privacy filter using randomization and linear programming based on the proposed privacy measure. The effectiveness of the optimal privacy filter design is evaluated through a production-inventory case study, illustrating its robustness against the inference attack.



## **15. Echomix: a Strong Anonymity System with Messaging**

cs.CR

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.02933v2) [paper-pdf](http://arxiv.org/pdf/2501.02933v2)

**Authors**: Ewa J Infeld, David Stainton, Leif Ryge, Threebit Hacker

**Abstract**: Echomix is a practical mix network framework and a suite of associated protocols providing strong metadata privacy against realistic modern adversaries. It is distinguished from other anonymity systems by a resistance to traffic analysis by global adversaries, compromised contacts and network infrastructure, quantum decryption algorithms, and statistical and confirmation attacks typical for multi-client messaging setting. It is implemented as Katzenpost, a robust software project, and used in multiple deployed systems, and features relatively low latency and bandwidth overhead.   The contributions of this paper are: (1) Improvements on leading mix network designs, supported by rigorous analysis. These include solutions to crucial vulnerabilities to traffic analysis, malicious servers and active attacks. (2) A cryptographic group messaging protocol with strong metadata protection guarantees and reliability. (3) Hybrid post-quantum nested packet encryption.



## **16. Graph Neural Backdoor: Fundamentals, Methodologies, Applications, and Future Directions**

cs.LG

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2406.10573v2) [paper-pdf](http://arxiv.org/pdf/2406.10573v2)

**Authors**: Xiao Yang, Gaolei Li, Jianhua Li

**Abstract**: Graph Neural Networks (GNNs) have significantly advanced various downstream graph-relevant tasks, encompassing recommender systems, molecular structure prediction, social media analysis, etc. Despite the boosts of GNN, recent research has empirically demonstrated its potential vulnerability to backdoor attacks, wherein adversaries employ triggers to poison input samples, inducing GNN to adversary-premeditated malicious outputs. This is typically due to the controlled training process, or the deployment of untrusted models, such as delegating model training to third-party service, leveraging external training sets, and employing pre-trained models from online sources. Although there's an ongoing increase in research on GNN backdoors, comprehensive investigation into this field is lacking. To bridge this gap, we propose the first survey dedicated to GNN backdoors. We begin by outlining the fundamental definition of GNN, followed by the detailed summarization and categorization of current GNN backdoor attacks and defenses based on their technical characteristics and application scenarios. Subsequently, the analysis of the applicability and use cases of GNN backdoors is undertaken. Finally, the exploration of potential research directions of GNN backdoors is presented. This survey aims to explore the principles of graph backdoors, provide insights to defenders, and promote future security research.



## **17. Unraveling Responsiveness of Chained BFT Consensus with Network Delay**

cs.DC

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.03695v1) [paper-pdf](http://arxiv.org/pdf/2501.03695v1)

**Authors**: Yining Tang, Qihang Luo, Runchao Han, Jianyu Niu, Chen Feng, Yinqian Zhang

**Abstract**: With the advancement of blockchain technology, chained Byzantine Fault Tolerant (BFT) protocols have been increasingly adopted in practical systems, making their performance a crucial aspect of the study. In this paper, we introduce a unified framework utilizing Markov Decision Processes (MDP) to model and assess the performance of three prominent chained BFT protocols. Our framework effectively captures complex adversarial behaviors, focusing on two key performance metrics: chain growth and commitment rate. We implement the optimal attack strategies obtained from MDP analysis on an existing evaluation platform for chained BFT protocols and conduct extensive experiments under various settings to validate our theoretical results. Through rigorous theoretical analysis and thorough practical experiments, we provide an in-depth evaluation of chained BFT protocols under diverse attack scenarios, uncovering optimal attack strategies. Contrary to conventional belief, our findings reveal that while responsiveness can enhance performance, it is not universally beneficial across all scenarios. This work not only deepens our understanding of chained BFT protocols, but also offers valuable insights and analytical tools that can inform the design of more robust and efficient protocols.



## **18. Transferable Adversarial Examples with Bayes Approach**

cs.LG

Accepted in AsiaCCS'25

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2208.06538v2) [paper-pdf](http://arxiv.org/pdf/2208.06538v2)

**Authors**: Mingyuan Fan, Cen Chen, Wenmeng Zhou, Yinggui Wang

**Abstract**: The vulnerability of deep neural networks (DNNs) to black-box adversarial attacks is one of the most heated topics in trustworthy AI. In such attacks, the attackers operate without any insider knowledge of the model, making the cross-model transferability of adversarial examples critical. Despite the potential for adversarial examples to be effective across various models, it has been observed that adversarial examples that are specifically crafted for a specific model often exhibit poor transferability. In this paper, we explore the transferability of adversarial examples via the lens of Bayesian approach. Specifically, we leverage Bayesian approach to probe the transferability and then study what constitutes a transferability-promoting prior. Following this, we design two concrete transferability-promoting priors, along with an adaptive dynamic weighting strategy for instances sampled from these priors. Employing these techniques, we present BayAtk. Extensive experiments illustrate the significant effectiveness of BayAtk in crafting more transferable adversarial examples against both undefended and defended black-box models compared to existing state-of-the-art attacks.



## **19. PhishAgent: A Robust Multimodal Agent for Phishing Webpage Detection**

cs.CR

Accepted at AAAI 2025

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2408.10738v2) [paper-pdf](http://arxiv.org/pdf/2408.10738v2)

**Authors**: Tri Cao, Chengyu Huang, Yuexin Li, Huilin Wang, Amy He, Nay Oo, Bryan Hooi

**Abstract**: Phishing attacks are a major threat to online security, exploiting user vulnerabilities to steal sensitive information. Various methods have been developed to counteract phishing, each with varying levels of accuracy, but they also face notable limitations. In this study, we introduce PhishAgent, a multimodal agent that combines a wide range of tools, integrating both online and offline knowledge bases with Multimodal Large Language Models (MLLMs). This combination leads to broader brand coverage, which enhances brand recognition and recall. Furthermore, we propose a multimodal information retrieval framework designed to extract the relevant top k items from offline knowledge bases, using available information from a webpage, including logos and HTML. Our empirical results, based on three real-world datasets, demonstrate that the proposed framework significantly enhances detection accuracy and reduces both false positives and false negatives, while maintaining model efficiency. Additionally, PhishAgent shows strong resilience against various types of adversarial attacks.



## **20. ChatBug: A Common Vulnerability of Aligned LLMs Induced by Chat Templates**

cs.CR

This paper is accepted to AAAI 2025

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2406.12935v2) [paper-pdf](http://arxiv.org/pdf/2406.12935v2)

**Authors**: Fengqing Jiang, Zhangchen Xu, Luyao Niu, Bill Yuchen Lin, Radha Poovendran

**Abstract**: Large language models (LLMs) are expected to follow instructions from users and engage in conversations. Techniques to enhance LLMs' instruction-following capabilities typically fine-tune them using data structured according to a predefined chat template. Although chat templates are shown to be effective in optimizing LLM performance, their impact on safety alignment of LLMs has been less understood, which is crucial for deploying LLMs safely at scale.   In this paper, we investigate how chat templates affect safety alignment of LLMs. We identify a common vulnerability, named ChatBug, that is introduced by chat templates. Our key insight to identify ChatBug is that the chat templates provide a rigid format that need to be followed by LLMs, but not by users. Hence, a malicious user may not necessarily follow the chat template when prompting LLMs. Instead, malicious users could leverage their knowledge of the chat template and accordingly craft their prompts to bypass safety alignments of LLMs. We develop two attacks to exploit the ChatBug vulnerability. We demonstrate that a malicious user can exploit the ChatBug vulnerability of eight state-of-the-art (SOTA) LLMs and effectively elicit unintended responses from these models. Moreover, we show that ChatBug can be exploited by existing jailbreak attacks to enhance their attack success rates. We investigate potential countermeasures to ChatBug. Our results show that while adversarial training effectively mitigates the ChatBug vulnerability, the victim model incurs significant performance degradation. These results highlight the trade-off between safety alignment and helpfulness. Developing new methods for instruction tuning to balance this trade-off is an open and critical direction for future research



## **21. Countering Backdoor Attacks in Image Recognition: A Survey and Evaluation of Mitigation Strategies**

cs.CR

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2411.11200v2) [paper-pdf](http://arxiv.org/pdf/2411.11200v2)

**Authors**: Kealan Dunnett, Reza Arablouei, Dimity Miller, Volkan Dedeoglu, Raja Jurdak

**Abstract**: The widespread adoption of deep learning across various industries has introduced substantial challenges, particularly in terms of model explainability and security. The inherent complexity of deep learning models, while contributing to their effectiveness, also renders them susceptible to adversarial attacks. Among these, backdoor attacks are especially concerning, as they involve surreptitiously embedding specific triggers within training data, causing the model to exhibit aberrant behavior when presented with input containing the triggers. Such attacks often exploit vulnerabilities in outsourced processes, compromising model integrity without affecting performance on clean (trigger-free) input data. In this paper, we present a comprehensive review of existing mitigation strategies designed to counter backdoor attacks in image recognition. We provide an in-depth analysis of the theoretical foundations, practical efficacy, and limitations of these approaches. In addition, we conduct an extensive benchmarking of sixteen state-of-the-art approaches against eight distinct backdoor attacks, utilizing three datasets, four model architectures, and three poisoning ratios. Our results, derived from 122,236 individual experiments, indicate that while many approaches provide some level of protection, their performance can vary considerably. Furthermore, when compared to two seminal approaches, most newer approaches do not demonstrate substantial improvements in overall performance or consistency across diverse settings. Drawing from these findings, we propose potential directions for developing more effective and generalizable defensive mechanisms in the future.



## **22. Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting**

cs.LG

11 pages, 5 figures

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2412.08099v2) [paper-pdf](http://arxiv.org/pdf/2412.08099v2)

**Authors**: Fuqiang Liu, Sicong Jiang, Luis Miranda-Moreno, Seongjin Choi, Lijun Sun

**Abstract**: Large Language Models (LLMs) have recently demonstrated significant potential in the field of time series forecasting, offering impressive capabilities in handling complex temporal data. However, their robustness and reliability in real-world applications remain under-explored, particularly concerning their susceptibility to adversarial attacks. In this paper, we introduce a targeted adversarial attack framework for LLM-based time series forecasting. By employing both gradient-free and black-box optimization methods, we generate minimal yet highly effective perturbations that significantly degrade the forecasting accuracy across multiple datasets and LLM architectures. Our experiments, which include models like TimeGPT and LLM-Time with GPT-3.5, GPT-4, LLaMa, and Mistral, show that adversarial attacks lead to much more severe performance degradation than random noise, and demonstrate the broad effectiveness of our attacks across different LLMs. The results underscore the critical vulnerabilities of LLMs in time series forecasting, highlighting the need for robust defense mechanisms to ensure their reliable deployment in practical applications.



## **23. When Should Selfish Miners Double-Spend?**

cs.CR

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.03227v1) [paper-pdf](http://arxiv.org/pdf/2501.03227v1)

**Authors**: Mustafa Doger, Sennur Ulukus

**Abstract**: Although, both double-spending and selfish-mining attacks have been extensively studied since the ``Bitcoin'' whitepaper of Nakamoto and the ``majority is not enough'' paper of Eyal and Sirer, there has been no rigorous stochastic analysis of an attack that combines the two, except for the complicated MDP models. In this paper, we first combine stubborn and selfish mining attacks, i.e., construct a strategy where the attacker acts stubborn until its private branch reaches a certain length and then switches to act selfish. We provide the optimal stubbornness for each parameter regime. Next, we provide the maximum stubbornness that is still more profitable than honest mining and argue a connection between the level of stubbornness and the $k$-confirmation rule. We show that, at each attack cycle, if the level of stubbornness is higher than $k$, there is a risk of double-spending which comes at no-cost to the adversary. The result can be seen as a guide for picking $k$ in the $k$-confirmation rule in a blockchain design. At each cycle, for a given stubbornness level, we rigorously formulate how great the risk of double-spending is. We provide the minimum double-spend value needed for an attack to be profitable in the regimes where the scheme is less profitable than honest mining. We further modify the attack in the stubborn regime in order to conceal the attack and increase the double-spending probability. Finally, we evaluate the results and provide the optimal and the maximum stubbornness levels for each parameter regime as well as the revenue. As a case study, with Bitcoin's $k=6$ block confirmation rule, we evaluate the revenue and double-spending risk of the attacks for each pool parameter.



## **24. The Robustness of Spiking Neural Networks in Federated Learning with Compression Against Non-omniscient Byzantine Attacks**

cs.CR

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.03306v1) [paper-pdf](http://arxiv.org/pdf/2501.03306v1)

**Authors**: Manh V. Nguyen, Liang Zhao, Bobin Deng, Shaoen Wu

**Abstract**: Spiking Neural Networks (SNNs), which offer exceptional energy efficiency for inference, and Federated Learning (FL), which offers privacy-preserving distributed training, is a rising area of interest that highly beneficial towards Internet of Things (IoT) devices. Despite this, research that tackles Byzantine attacks and bandwidth limitation in FL-SNNs, both poses significant threats on model convergence and training times, still remains largely unexplored. Going beyond proposing a solution for both of these problems, in this work we highlight the dual benefits of FL-SNNs, against non-omniscient Byzantine adversaries (ones that restrict attackers access to local clients datasets), and greater communication efficiency, over FL-ANNs. Specifically, we discovered that a simple integration of Top-\k{appa} sparsification into the FL apparatus can help leverage the advantages of the SNN models in both greatly reducing bandwidth usage and significantly boosting the robustness of FL training against non-omniscient Byzantine adversaries. Most notably, we saw a massive improvement of roughly 40% accuracy gain in FL-SNNs training under the lethal MinMax attack



## **25. Leader Rotation Is Not Enough: Scrutinizing Leadership Democracy of Chained BFT Consensus**

cs.CR

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.02970v1) [paper-pdf](http://arxiv.org/pdf/2501.02970v1)

**Authors**: Yining Tang, Runchao Han, Jianyu Niu, Chen Feng, Yinqian Zhang

**Abstract**: With the growing popularity of blockchains, modern chained BFT protocols combining chaining and leader rotation to obtain better efficiency and leadership democracy have received increasing interest. Although the efficiency provisions of chained BFT protocols have been thoroughly analyzed, the leadership democracy has received little attention in prior work. In this paper, we scrutinize the leadership democracy of four representative chained BFT protocols, especially under attack. To this end, we propose a unified framework with two evaluation metrics, i.e., chain quality and censorship resilience, and quantitatively analyze chosen protocols through the Markov Decision Process (MDP). With this framework, we further examine the impact of two key components, i.e., voting pattern and leader rotation on leadership democracy. Our results indicate that leader rotation is not enough to provide the leadership democracy guarantee; an adversary could utilize the design, e.g., voting pattern, to deteriorate the leadership democracy significantly. Based on the analysis results, we propose customized countermeasures for three evaluated protocols to improve their leadership democracy with only slight protocol overhead and no change of consensus rules. We also discuss future directions toward building more democratic chained BFT protocols.



## **26. Seeing the Whole in the Parts in Self-Supervised Representation Learning**

cs.LG

20 pages

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.02860v1) [paper-pdf](http://arxiv.org/pdf/2501.02860v1)

**Authors**: Arthur Aubret, Céline Teulière, Jochen Triesch

**Abstract**: Recent successes in self-supervised learning (SSL) model spatial co-occurrences of visual features either by masking portions of an image or by aggressively cropping it. Here, we propose a new way to model spatial co-occurrences by aligning local representations (before pooling) with a global image representation. We present CO-SSL, a family of instance discrimination methods and show that it outperforms previous methods on several datasets, including ImageNet-1K where it achieves 71.5% of Top-1 accuracy with 100 pre-training epochs. CO-SSL is also more robust to noise corruption, internal corruption, small adversarial attacks, and large training crop sizes. Our analysis further indicates that CO-SSL learns highly redundant local representations, which offers an explanation for its robustness. Overall, our work suggests that aligning local and global representations may be a powerful principle of unsupervised category learning.



## **27. MBTSAD: Mitigating Backdoors in Language Models Based on Token Splitting and Attention Distillation**

cs.CR

Accepted by ICTAI 2024

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.02754v1) [paper-pdf](http://arxiv.org/pdf/2501.02754v1)

**Authors**: Yidong Ding, Jiafei Niu, Ping Yi

**Abstract**: In recent years, attention-based models have excelled across various domains but remain vulnerable to backdoor attacks, often from downloading or fine-tuning on poisoned datasets. Many current methods to mitigate backdoors in NLP models rely on the pre-trained (unfine-tuned) weights, but these methods fail in scenarios where the pre-trained weights are not available. In this work, we propose MBTSAD, which can mitigate backdoors in the language model by utilizing only a small subset of clean data and does not require pre-trained weights. Specifically, MBTSAD retrains the backdoored model on a dataset generated by token splitting. Then MBTSAD leverages attention distillation, the retrained model is the teacher model, and the original backdoored model is the student model. Experimental results demonstrate that MBTSAD achieves comparable backdoor mitigation performance as the methods based on pre-trained weights while maintaining the performance on clean data. MBTSAD does not rely on pre-trained weights, enhancing its utility in scenarios where pre-trained weights are inaccessible. In addition, we simplify the min-max problem of adversarial training and visualize text representations to discover that the token splitting method in MBTSAD's first step generates Out-of-Distribution (OOD) data, leading the model to learn more generalized features and eliminate backdoor patterns.



## **28. Persistence of Backdoor-based Watermarks for Neural Networks: A Comprehensive Evaluation**

cs.LG

Preprint. Under Review

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.02704v1) [paper-pdf](http://arxiv.org/pdf/2501.02704v1)

**Authors**: Anh Tu Ngo, Chuan Song Heng, Nandish Chattopadhyay, Anupam Chattopadhyay

**Abstract**: Deep Neural Networks (DNNs) have gained considerable traction in recent years due to the unparalleled results they gathered. However, the cost behind training such sophisticated models is resource intensive, resulting in many to consider DNNs to be intellectual property (IP) to model owners. In this era of cloud computing, high-performance DNNs are often deployed all over the internet so that people can access them publicly. As such, DNN watermarking schemes, especially backdoor-based watermarks, have been actively developed in recent years to preserve proprietary rights. Nonetheless, there lies much uncertainty on the robustness of existing backdoor watermark schemes, towards both adversarial attacks and unintended means such as fine-tuning neural network models. One reason for this is that no complete guarantee of robustness can be assured in the context of backdoor-based watermark. In this paper, we extensively evaluate the persistence of recent backdoor-based watermarks within neural networks in the scenario of fine-tuning, we propose/develop a novel data-driven idea to restore watermark after fine-tuning without exposing the trigger set. Our empirical results show that by solely introducing training data after fine-tuning, the watermark can be restored if model parameters do not shift dramatically during fine-tuning. Depending on the types of trigger samples used, trigger accuracy can be reinstated to up to 100%. Our study further explores how the restoration process works using loss landscape visualization, as well as the idea of introducing training data in fine-tuning stage to alleviate watermark vanishing.



## **29. Layer-Level Self-Exposure and Patch: Affirmative Token Mitigation for Jailbreak Attack Defense**

cs.CR

**SubmitDate**: 2025-01-05    [abs](http://arxiv.org/abs/2501.02629v1) [paper-pdf](http://arxiv.org/pdf/2501.02629v1)

**Authors**: Yang Ouyang, Hengrui Gu, Shuhang Lin, Wenyue Hua, Jie Peng, Bhavya Kailkhura, Tianlong Chen, Kaixiong Zhou

**Abstract**: As large language models (LLMs) are increasingly deployed in diverse applications, including chatbot assistants and code generation, aligning their behavior with safety and ethical standards has become paramount. However, jailbreak attacks, which exploit vulnerabilities to elicit unintended or harmful outputs, threaten LLMs' safety significantly. In this paper, we introduce Layer-AdvPatcher, a novel methodology designed to defend against jailbreak attacks by utilizing an unlearning strategy to patch specific layers within LLMs through self-augmented datasets. Our insight is that certain layer(s), tend to produce affirmative tokens when faced with harmful prompts. By identifying these layers and adversarially exposing them to generate more harmful data, one can understand their inherent and diverse vulnerabilities to attacks. With these exposures, we then "unlearn" these issues, reducing the impact of affirmative tokens and hence minimizing jailbreak risks while keeping the model's responses to safe queries intact. We conduct extensive experiments on two models, four benchmark datasets, and multiple state-of-the-art jailbreak benchmarks to demonstrate the efficacy of our approach. Results indicate that our framework reduces the harmfulness and attack success rate of jailbreak attacks without compromising utility for benign queries compared to recent defense methods.



## **30. Unveiling the Threat of Fraud Gangs to Graph Neural Networks: Multi-Target Graph Injection Attacks Against GNN-Based Fraud Detectors**

cs.LG

19 pages, 5 figures, 12 tables, The 39th AAAI Conference on  Artificial Intelligence (AAAI 2025)

**SubmitDate**: 2025-01-05    [abs](http://arxiv.org/abs/2412.18370v2) [paper-pdf](http://arxiv.org/pdf/2412.18370v2)

**Authors**: Jinhyeok Choi, Heehyeon Kim, Joyce Jiyoung Whang

**Abstract**: Graph neural networks (GNNs) have emerged as an effective tool for fraud detection, identifying fraudulent users, and uncovering malicious behaviors. However, attacks against GNN-based fraud detectors and their risks have rarely been studied, thereby leaving potential threats unaddressed. Recent findings suggest that frauds are increasingly organized as gangs or groups. In this work, we design attack scenarios where fraud gangs aim to make their fraud nodes misclassified as benign by camouflaging their illicit activities in collusion. Based on these scenarios, we study adversarial attacks against GNN-based fraud detectors by simulating attacks of fraud gangs in three real-world fraud cases: spam reviews, fake news, and medical insurance frauds. We define these attacks as multi-target graph injection attacks and propose MonTi, a transformer-based Multi-target one-Time graph injection attack model. MonTi simultaneously generates attributes and edges of all attack nodes with a transformer encoder, capturing interdependencies between attributes and edges more effectively than most existing graph injection attack methods that generate these elements sequentially. Additionally, MonTi adaptively allocates the degree budget for each attack node to explore diverse injection structures involving target, candidate, and attack nodes, unlike existing methods that fix the degree budget across all attack nodes. Experiments show that MonTi outperforms the state-of-the-art graph injection attack methods on five real-world graphs.



## **31. GCP: Guarded Collaborative Perception with Spatial-Temporal Aware Malicious Agent Detection**

cs.CV

15 pages

**SubmitDate**: 2025-01-05    [abs](http://arxiv.org/abs/2501.02450v1) [paper-pdf](http://arxiv.org/pdf/2501.02450v1)

**Authors**: Yihang Tao, Senkang Hu, Yue Hu, Haonan An, Hangcheng Cao, Yuguang Fang

**Abstract**: Collaborative perception significantly enhances autonomous driving safety by extending each vehicle's perception range through message sharing among connected and autonomous vehicles. Unfortunately, it is also vulnerable to adversarial message attacks from malicious agents, resulting in severe performance degradation. While existing defenses employ hypothesis-and-verification frameworks to detect malicious agents based on single-shot outliers, they overlook temporal message correlations, which can be circumvented by subtle yet harmful perturbations in model input and output spaces. This paper reveals a novel blind area confusion (BAC) attack that compromises existing single-shot outlier-based detection methods. As a countermeasure, we propose GCP, a Guarded Collaborative Perception framework based on spatial-temporal aware malicious agent detection, which maintains single-shot spatial consistency through a confidence-scaled spatial concordance loss, while simultaneously examining temporal anomalies by reconstructing historical bird's eye view motion flows in low-confidence regions. We also employ a joint spatial-temporal Benjamini-Hochberg test to synthesize dual-domain anomaly results for reliable malicious agent detection. Extensive experiments demonstrate GCP's superior performance under diverse attack scenarios, achieving up to 34.69% improvements in AP@0.5 compared to the state-of-the-art CP defense strategies under BAC attacks, while maintaining consistent 5-8% improvements under other typical attacks. Code will be released at https://github.com/CP-Security/GCP.git.



## **32. A Practical Examination of AI-Generated Text Detectors for Large Language Models**

cs.CL

8 pages. Submitted to NAACL

**SubmitDate**: 2025-01-04    [abs](http://arxiv.org/abs/2412.05139v2) [paper-pdf](http://arxiv.org/pdf/2412.05139v2)

**Authors**: Brian Tufts, Xuandong Zhao, Lei Li

**Abstract**: The proliferation of large language models has raised growing concerns about their misuse, particularly in cases where AI-generated text is falsely attributed to human authors. Machine-generated content detectors claim to effectively identify such text under various conditions and from any language model. This paper critically evaluates these claims by assessing several popular detectors (RADAR, Wild, T5Sentinel, Fast-DetectGPT, GPTID, LogRank, Binoculars) on a range of domains, datasets, and models that these detectors have not previously encountered. We employ various prompting strategies to simulate adversarial attacks, demonstrating that even moderate efforts can significantly evade detection. We emphasize the importance of the true positive rate at a specific false positive rate (TPR@FPR) metric and demonstrate that these detectors perform poorly in certain settings, with TPR@.01 as low as 0%. Our findings suggest that both trained and zero-shot detectors struggle to maintain high sensitivity while achieving a reasonable true positive rate.



## **33. GNSS/GPS Spoofing and Jamming Identification Using Machine Learning and Deep Learning**

cs.CR

**SubmitDate**: 2025-01-04    [abs](http://arxiv.org/abs/2501.02352v1) [paper-pdf](http://arxiv.org/pdf/2501.02352v1)

**Authors**: Ali Ghanbarzade, Hossein Soleimani

**Abstract**: The increasing reliance on Global Navigation Satellite Systems (GNSS), particularly the Global Positioning System (GPS), underscores the urgent need to safeguard these technologies against malicious threats such as spoofing and jamming. As the backbone for positioning, navigation, and timing (PNT) across various applications including transportation, telecommunications, and emergency services GNSS is vulnerable to deliberate interference that poses significant risks. Spoofing attacks, which involve transmitting counterfeit GNSS signals to mislead receivers into calculating incorrect positions, can result in serious consequences, from navigational errors in civilian aviation to security breaches in military operations. Furthermore, the lack of inherent security measures within GNSS systems makes them attractive targets for adversaries. While GNSS/GPS jamming and spoofing systems consist of numerous components, the ability to distinguish authentic signals from malicious ones is essential for maintaining system integrity. Recent advancements in machine learning and deep learning provide promising avenues for enhancing detection and mitigation strategies against these threats. This paper addresses both spoofing and jamming by tackling real-world challenges through machine learning, deep learning, and computer vision techniques. Through extensive experiments on two real-world datasets related to spoofing and jamming detection using advanced algorithms, we achieved state of the art results. In the GNSS/GPS jamming detection task, we attained approximately 99% accuracy, improving performance by around 5% compared to previous studies. Additionally, we addressed a challenging tasks related to spoofing detection, yielding results that underscore the potential of machine learning and deep learning in this domain.



## **34. Distillation-Enhanced Physical Adversarial Attacks**

cs.CV

7 pages, 5 figures

**SubmitDate**: 2025-01-04    [abs](http://arxiv.org/abs/2501.02232v1) [paper-pdf](http://arxiv.org/pdf/2501.02232v1)

**Authors**: Wei Liu, Yonglin Wu, Chaoqun Li, Zhuodong Liu, Huanqian Yan

**Abstract**: The study of physical adversarial patches is crucial for identifying vulnerabilities in AI-based recognition systems and developing more robust deep learning models. While recent research has focused on improving patch stealthiness for greater practical applicability, achieving an effective balance between stealth and attack performance remains a significant challenge. To address this issue, we propose a novel physical adversarial attack method that leverages knowledge distillation. Specifically, we first define a stealthy color space tailored to the target environment to ensure smooth blending. Then, we optimize an adversarial patch in an unconstrained color space, which serves as the 'teacher' patch. Finally, we use an adversarial knowledge distillation module to transfer the teacher patch's knowledge to the 'student' patch, guiding the optimization of the stealthy patch. Experimental results show that our approach improves attack performance by 20%, while maintaining stealth, highlighting its practical value.



## **35. 2-in-1 Accelerator: Enabling Random Precision Switch for Winning Both Adversarial Robustness and Efficiency**

cs.LG

Accepted at MICRO 2021

**SubmitDate**: 2025-01-04    [abs](http://arxiv.org/abs/2109.05223v3) [paper-pdf](http://arxiv.org/pdf/2109.05223v3)

**Authors**: Yonggan Fu, Yang Zhao, Qixuan Yu, Chaojian Li, Yingyan Celine Lin

**Abstract**: The recent breakthroughs of deep neural networks (DNNs) and the advent of billions of Internet of Things (IoT) devices have excited an explosive demand for intelligent IoT devices equipped with domain-specific DNN accelerators. However, the deployment of DNN accelerator enabled intelligent functionality into real-world IoT devices still remains particularly challenging. First, powerful DNNs often come at prohibitive complexities, whereas IoT devices often suffer from stringent resource constraints. Second, while DNNs are vulnerable to adversarial attacks especially on IoT devices exposed to complex real-world environments, many IoT applications require strict security. Existing DNN accelerators mostly tackle only one of the two aforementioned challenges (i.e., efficiency or adversarial robustness) while neglecting or even sacrificing the other. To this end, we propose a 2-in-1 Accelerator, an integrated algorithm-accelerator co-design framework aiming at winning both the adversarial robustness and efficiency of DNN accelerators. Specifically, we first propose a Random Precision Switch (RPS) algorithm that can effectively defend DNNs against adversarial attacks by enabling random DNN quantization as an in-situ model switch. Furthermore, we propose a new precision-scalable accelerator featuring (1) a new precision-scalable MAC unit architecture which spatially tiles the temporal MAC units to boost both the achievable efficiency and flexibility and (2) a systematically optimized dataflow that is searched by our generic accelerator optimizer. Extensive experiments and ablation studies validate that our 2-in-1 Accelerator can not only aggressively boost both the adversarial robustness and efficiency of DNN accelerators under various attacks, but also naturally support instantaneous robustness-efficiency trade-offs adapting to varied resources without the necessity of DNN retraining.



## **36. Drawing Robust Scratch Tickets: Subnetworks with Inborn Robustness Are Found within Randomly Initialized Networks**

cs.LG

Accepted at NeurIPS 2021

**SubmitDate**: 2025-01-04    [abs](http://arxiv.org/abs/2110.14068v4) [paper-pdf](http://arxiv.org/pdf/2110.14068v4)

**Authors**: Yonggan Fu, Qixuan Yu, Yang Zhang, Shang Wu, Xu Ouyang, David Cox, Yingyan Celine Lin

**Abstract**: Deep Neural Networks (DNNs) are known to be vulnerable to adversarial attacks, i.e., an imperceptible perturbation to the input can mislead DNNs trained on clean images into making erroneous predictions. To tackle this, adversarial training is currently the most effective defense method, by augmenting the training set with adversarial samples generated on the fly. Interestingly, we discover for the first time that there exist subnetworks with inborn robustness, matching or surpassing the robust accuracy of the adversarially trained networks with comparable model sizes, within randomly initialized networks without any model training, indicating that adversarial training on model weights is not indispensable towards adversarial robustness. We name such subnetworks Robust Scratch Tickets (RSTs), which are also by nature efficient. Distinct from the popular lottery ticket hypothesis, neither the original dense networks nor the identified RSTs need to be trained. To validate and understand this fascinating finding, we further conduct extensive experiments to study the existence and properties of RSTs under different models, datasets, sparsity patterns, and attacks, drawing insights regarding the relationship between DNNs' robustness and their initialization/overparameterization. Furthermore, we identify the poor adversarial transferability between RSTs of different sparsity ratios drawn from the same randomly initialized dense network, and propose a Random RST Switch (R2S) technique, which randomly switches between different RSTs, as a novel defense method built on top of RSTs. We believe our findings about RSTs have opened up a new perspective to study model robustness and extend the lottery ticket hypothesis.



## **37. Patch-Fool: Are Vision Transformers Always Robust Against Adversarial Perturbations?**

cs.CV

Accepted at ICLR 2022

**SubmitDate**: 2025-01-04    [abs](http://arxiv.org/abs/2203.08392v3) [paper-pdf](http://arxiv.org/pdf/2203.08392v3)

**Authors**: Yonggan Fu, Shunyao Zhang, Shang Wu, Cheng Wan, Yingyan Celine Lin

**Abstract**: Vision transformers (ViTs) have recently set off a new wave in neural architecture design thanks to their record-breaking performance in various vision tasks. In parallel, to fulfill the goal of deploying ViTs into real-world vision applications, their robustness against potential malicious attacks has gained increasing attention. In particular, recent works show that ViTs are more robust against adversarial attacks as compared with convolutional neural networks (CNNs), and conjecture that this is because ViTs focus more on capturing global interactions among different input/feature patches, leading to their improved robustness to local perturbations imposed by adversarial attacks. In this work, we ask an intriguing question: "Under what kinds of perturbations do ViTs become more vulnerable learners compared to CNNs?" Driven by this question, we first conduct a comprehensive experiment regarding the robustness of both ViTs and CNNs under various existing adversarial attacks to understand the underlying reason favoring their robustness. Based on the drawn insights, we then propose a dedicated attack framework, dubbed Patch-Fool, that fools the self-attention mechanism by attacking its basic component (i.e., a single patch) with a series of attention-aware optimization techniques. Interestingly, our Patch-Fool framework shows for the first time that ViTs are not necessarily more robust than CNNs against adversarial perturbations. In particular, we find that ViTs are more vulnerable learners compared with CNNs against our Patch-Fool attack which is consistent across extensive experiments, and the observations from Sparse/Mild Patch-Fool, two variants of Patch-Fool, indicate an intriguing insight that the perturbation density and strength on each patch seem to be the key factors that influence the robustness ranking between ViTs and CNNs.



## **38. NeRFool: Uncovering the Vulnerability of Generalizable Neural Radiance Fields against Adversarial Perturbations**

cs.CV

Accepted by ICML 2023

**SubmitDate**: 2025-01-04    [abs](http://arxiv.org/abs/2306.06359v2) [paper-pdf](http://arxiv.org/pdf/2306.06359v2)

**Authors**: Yonggan Fu, Ye Yuan, Souvik Kundu, Shang Wu, Shunyao Zhang, Yingyan Celine Lin

**Abstract**: Generalizable Neural Radiance Fields (GNeRF) are one of the most promising real-world solutions for novel view synthesis, thanks to their cross-scene generalization capability and thus the possibility of instant rendering on new scenes. While adversarial robustness is essential for real-world applications, little study has been devoted to understanding its implication on GNeRF. We hypothesize that because GNeRF is implemented by conditioning on the source views from new scenes, which are often acquired from the Internet or third-party providers, there are potential new security concerns regarding its real-world applications. Meanwhile, existing understanding and solutions for neural networks' adversarial robustness may not be applicable to GNeRF, due to its 3D nature and uniquely diverse operations. To this end, we present NeRFool, which to the best of our knowledge is the first work that sets out to understand the adversarial robustness of GNeRF. Specifically, NeRFool unveils the vulnerability patterns and important insights regarding GNeRF's adversarial robustness. Built upon the above insights gained from NeRFool, we further develop NeRFool+, which integrates two techniques capable of effectively attacking GNeRF across a wide range of target views, and provide guidelines for defending against our proposed attacks. We believe that our NeRFool/NeRFool+ lays the initial foundation for future innovations in developing robust real-world GNeRF solutions. Our codes are available at: https://github.com/GATECH-EIC/NeRFool.



## **39. Exploring Secure Machine Learning Through Payload Injection and FGSM Attacks on ResNet-50**

cs.CR

**SubmitDate**: 2025-01-04    [abs](http://arxiv.org/abs/2501.02147v1) [paper-pdf](http://arxiv.org/pdf/2501.02147v1)

**Authors**: Umesh Yadav, Suman Niraula, Gaurav Kumar Gupta, Bicky Yadav

**Abstract**: This paper investigates the resilience of a ResNet-50 image classification model under two prominent security threats: Fast Gradient Sign Method (FGSM) adversarial attacks and malicious payload injection. Initially, the model attains a 53.33% accuracy on clean images. When subjected to FGSM perturbations, its overall accuracy remains unchanged; however, the model's confidence in incorrect predictions notably increases. Concurrently, a payload injection scheme is successfully executed in 93.33% of the tested samples, revealing how stealthy attacks can manipulate model predictions without degrading visual quality. These findings underscore the vulnerability of even high-performing neural networks and highlight the urgency of developing more robust defense mechanisms for security-critical applications.



## **40. AVTrustBench: Assessing and Enhancing Reliability and Robustness in Audio-Visual LLMs**

cs.CV

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.02135v1) [paper-pdf](http://arxiv.org/pdf/2501.02135v1)

**Authors**: Sanjoy Chowdhury, Sayan Nag, Subhrajyoti Dasgupta, Yaoting Wang, Mohamed Elhoseiny, Ruohan Gao, Dinesh Manocha

**Abstract**: With the rapid advancement of Multi-modal Large Language Models (MLLMs), several diagnostic benchmarks have recently been developed to assess these models' multi-modal reasoning proficiency. However, these benchmarks are restricted to assessing primarily the visual aspect and do not examine the holistic audio-visual (AV) understanding. Moreover, currently, there are no benchmarks that investigate the capabilities of AVLLMs to calibrate their responses when presented with perturbed inputs. To this end, we introduce Audio-Visual Trustworthiness assessment Benchmark (AVTrustBench), comprising 600K samples spanning over 9 meticulously crafted tasks, evaluating the capabilities of AVLLMs across three distinct dimensions: Adversarial attack, Compositional reasoning, and Modality-specific dependency. Using our benchmark we extensively evaluate 13 state-of-the-art AVLLMs. The findings reveal that the majority of existing models fall significantly short of achieving human-like comprehension, offering valuable insights for future research directions. To alleviate the limitations in the existing approaches, we further propose a robust, model-agnostic calibrated audio-visual preference optimization based training strategy CAVPref, obtaining a gain up to 30.19% across all 9 tasks. We will publicly release our code and benchmark to facilitate future research in this direction.



## **41. Towards Robust and Accurate Stability Estimation of Local Surrogate Models in Text-based Explainable AI**

cs.LG

12 pages, 1 figure, 4 tables. arXiv admin note: substantial text  overlap with arXiv:2406.15839. substantial text overlap with arXiv:2501.01516

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.02042v1) [paper-pdf](http://arxiv.org/pdf/2501.02042v1)

**Authors**: Christopher Burger, Charles Walter, Thai Le, Lingwei Chen

**Abstract**: Recent work has investigated the concept of adversarial attacks on explainable AI (XAI) in the NLP domain with a focus on examining the vulnerability of local surrogate methods such as Lime to adversarial perturbations or small changes on the input of a machine learning (ML) model. In such attacks, the generated explanation is manipulated while the meaning and structure of the original input remain similar under the ML model. Such attacks are especially alarming when XAI is used as a basis for decision making (e.g., prescribing drugs based on AI medical predictors) or for legal action (e.g., legal dispute involving AI software). Although weaknesses across many XAI methods have been shown to exist, the reasons behind why remain little explored. Central to this XAI manipulation is the similarity measure used to calculate how one explanation differs from another. A poor choice of similarity measure can lead to erroneous conclusions about the stability or adversarial robustness of an XAI method. Therefore, this work investigates a variety of similarity measures designed for text-based ranked lists referenced in related work to determine their comparative suitability for use. We find that many measures are overly sensitive, resulting in erroneous estimates of stability. We then propose a weighting scheme for text-based data that incorporates the synonymity between the features within an explanation, providing more accurate estimates of the actual weakness of XAI methods to adversarial examples.



## **42. Detecting and Mitigating Adversarial Attacks on Deep Learning-Based MRI Reconstruction Without Any Retraining**

cs.CV

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.01908v1) [paper-pdf](http://arxiv.org/pdf/2501.01908v1)

**Authors**: Mahdi Saberi, Chi Zhang, Mehmet Akcakaya

**Abstract**: Deep learning (DL) methods, especially those based on physics-driven DL, have become the state-of-the-art for reconstructing sub-sampled magnetic resonance imaging (MRI) data. However, studies have shown that these methods are susceptible to small adversarial input perturbations, or attacks, resulting in major distortions in the output images. Various strategies have been proposed to reduce the effects of these attacks, but they require retraining and may lower reconstruction quality for non-perturbed/clean inputs. In this work, we propose a novel approach for detecting and mitigating adversarial attacks on MRI reconstruction models without any retraining. Our detection strategy is based on the idea of cyclic measurement consistency. The output of the model is mapped to another set of MRI measurements for a different sub-sampling pattern, and this synthesized data is reconstructed with the same model. Intuitively, without an attack, the second reconstruction is expected to be consistent with the first, while with an attack, disruptions are present. Subsequently, this idea is extended to devise a novel objective function, which is minimized within a small ball around the attack input for mitigation. Experimental results show that our method substantially reduces the impact of adversarial perturbations across different datasets, attack types/strengths and PD-DL networks, and qualitatively and quantitatively outperforms conventional mitigation methods that involve retraining.



## **43. Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions**

cs.CL

Our code is publicly available at  https://github.com/UKPLab/POATE-attack

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.01872v2) [paper-pdf](http://arxiv.org/pdf/2501.01872v2)

**Authors**: Rachneet Sachdeva, Rima Hazra, Iryna Gurevych

**Abstract**: Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. In this work, we introduce POATE (Polar Opposite query generation, Adversarial Template construction, and Elaboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. To counter this, we propose Intent-Aware CoT and Reverse Thinking CoT, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. These methods enhance reasoning robustness and strengthen the model's defense against adversarial exploits.



## **44. PB-UAP: Hybrid Universal Adversarial Attack For Image Segmentation**

cs.CV

Accepted by ICASSP 2025

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2412.16651v2) [paper-pdf](http://arxiv.org/pdf/2412.16651v2)

**Authors**: Yufei Song, Ziqi Zhou, Minghui Li, Xianlong Wang, Hangtao Zhang, Menghao Deng, Wei Wan, Shengshan Hu, Leo Yu Zhang

**Abstract**: With the rapid advancement of deep learning, the model robustness has become a significant research hotspot, \ie, adversarial attacks on deep neural networks. Existing works primarily focus on image classification tasks, aiming to alter the model's predicted labels. Due to the output complexity and deeper network architectures, research on adversarial examples for segmentation models is still limited, particularly for universal adversarial perturbations. In this paper, we propose a novel universal adversarial attack method designed for segmentation models, which includes dual feature separation and low-frequency scattering modules. The two modules guide the training of adversarial examples in the pixel and frequency space, respectively. Experiments demonstrate that our method achieves high attack success rates surpassing the state-of-the-art methods, and exhibits strong transferability across different models.



## **45. Rerouting LLM Routers**

cs.CR

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.01818v1) [paper-pdf](http://arxiv.org/pdf/2501.01818v1)

**Authors**: Avital Shafran, Roei Schuster, Thomas Ristenpart, Vitaly Shmatikov

**Abstract**: LLM routers aim to balance quality and cost of generation by classifying queries and routing them to a cheaper or more expensive LLM depending on their complexity. Routers represent one type of what we call LLM control planes: systems that orchestrate use of one or more LLMs. In this paper, we investigate routers' adversarial robustness.   We first define LLM control plane integrity, i.e., robustness of LLM orchestration to adversarial inputs, as a distinct problem in AI safety. Next, we demonstrate that an adversary can generate query-independent token sequences we call ``confounder gadgets'' that, when added to any query, cause LLM routers to send the query to a strong LLM.   Our quantitative evaluation shows that this attack is successful both in white-box and black-box settings against a variety of open-source and commercial routers, and that confounding queries do not affect the quality of LLM responses. Finally, we demonstrate that gadgets can be effective while maintaining low perplexity, thus perplexity-based filtering is not an effective defense. We finish by investigating alternative defenses.



## **46. How Toxic Can You Get? Search-based Toxicity Testing for Large Language Models**

cs.SE

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.01741v1) [paper-pdf](http://arxiv.org/pdf/2501.01741v1)

**Authors**: Simone Corbo, Luca Bancale, Valeria De Gennaro, Livia Lestingi, Vincenzo Scotti, Matteo Camilli

**Abstract**: Language is a deep-rooted means of perpetration of stereotypes and discrimination. Large Language Models (LLMs), now a pervasive technology in our everyday lives, can cause extensive harm when prone to generating toxic responses. The standard way to address this issue is to align the LLM, which, however, dampens the issue without constituting a definitive solution. Therefore, testing LLM even after alignment efforts remains crucial for detecting any residual deviations with respect to ethical standards. We present EvoTox, an automated testing framework for LLMs' inclination to toxicity, providing a way to quantitatively assess how much LLMs can be pushed towards toxic responses even in the presence of alignment. The framework adopts an iterative evolution strategy that exploits the interplay between two LLMs, the System Under Test (SUT) and the Prompt Generator steering SUT responses toward higher toxicity. The toxicity level is assessed by an automated oracle based on an existing toxicity classifier. We conduct a quantitative and qualitative empirical evaluation using four state-of-the-art LLMs as evaluation subjects having increasing complexity (7-13 billion parameters). Our quantitative evaluation assesses the cost-effectiveness of four alternative versions of EvoTox against existing baseline methods, based on random search, curated datasets of toxic prompts, and adversarial attacks. Our qualitative assessment engages human evaluators to rate the fluency of the generated prompts and the perceived toxicity of the responses collected during the testing sessions. Results indicate that the effectiveness, in terms of detected toxicity level, is significantly higher than the selected baseline methods (effect size up to 1.0 against random search and up to 0.99 against adversarial attacks). Furthermore, EvoTox yields a limited cost overhead (from 22% to 35% on average).



## **47. Adaptive Meta-learning-based Adversarial Training for Robust Automatic Modulation Classification**

cs.LG

Submitted to IEEE International Conference on Communications (ICC)  2025

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.01620v1) [paper-pdf](http://arxiv.org/pdf/2501.01620v1)

**Authors**: Amirmohammad Bamdad, Ali Owfi, Fatemeh Afghah

**Abstract**: DL-based automatic modulation classification (AMC) models are highly susceptible to adversarial attacks, where even minimal input perturbations can cause severe misclassifications. While adversarially training an AMC model based on an adversarial attack significantly increases its robustness against that attack, the AMC model will still be defenseless against other adversarial attacks. The theoretically infinite possibilities for adversarial perturbations mean that an AMC model will inevitably encounter new unseen adversarial attacks if it is ever to be deployed to a real-world communication system. Moreover, the computational limitations and challenges of obtaining new data in real-time will not allow a full training process for the AMC model to adapt to the new attack when it is online. To this end, we propose a meta-learning-based adversarial training framework for AMC models that substantially enhances robustness against unseen adversarial attacks and enables fast adaptation to these attacks using just a few new training samples, if any are available. Our results demonstrate that this training framework provides superior robustness and accuracy with much less online training time than conventional adversarial training of AMC models, making it highly efficient for real-world deployment.



## **48. BLAST: A Stealthy Backdoor Leverage Attack against Cooperative Multi-Agent Deep Reinforcement Learning based Systems**

cs.AI

12. arXiv admin note: substantial text overlap with arXiv:2409.07775

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.01593v1) [paper-pdf](http://arxiv.org/pdf/2501.01593v1)

**Authors**: Yinbo Yu, Saihao Yan, Xueyu Yin, Jing Fang, Jiajia Liu

**Abstract**: Recent studies have shown that cooperative multi-agent deep reinforcement learning (c-MADRL) is under the threat of backdoor attacks. Once a backdoor trigger is observed, it will perform malicious actions leading to failures or malicious goals. However, existing backdoor attacks suffer from several issues, e.g., instant trigger patterns lack stealthiness, the backdoor is trained or activated by an additional network, or all agents are backdoored. To this end, in this paper, we propose a novel backdoor leverage attack against c-MADRL, BLAST, which attacks the entire multi-agent team by embedding the backdoor only in a single agent. Firstly, we introduce adversary spatiotemporal behavior patterns as the backdoor trigger rather than manual-injected fixed visual patterns or instant status and control the period to perform malicious actions. This method can guarantee the stealthiness and practicality of BLAST. Secondly, we hack the original reward function of the backdoor agent via unilateral guidance to inject BLAST, so as to achieve the \textit{leverage attack effect} that can pry open the entire multi-agent system via a single backdoor agent. We evaluate our BLAST against 3 classic c-MADRL algorithms (VDN, QMIX, and MAPPO) in 2 popular c-MADRL environments (SMAC and Pursuit), and 2 existing defense mechanisms. The experimental results demonstrate that BLAST can achieve a high attack success rate while maintaining a low clean performance variance rate.



## **49. Familiarity-Based Open-Set Recognition Under Adversarial Attacks**

cs.CV

Published in: Proceedings of the 6th Northern Lights Deep Learning  Conference (NLDL), PMLR 265, 2025

**SubmitDate**: 2025-01-02    [abs](http://arxiv.org/abs/2311.05006v2) [paper-pdf](http://arxiv.org/pdf/2311.05006v2)

**Authors**: Philip Enevoldsen, Christian Gundersen, Nico Lang, Serge Belongie, Christian Igel

**Abstract**: Open-set recognition (OSR), the identification of novel categories, can be a critical component when deploying classification models in real-world applications. Recent work has shown that familiarity-based scoring rules such as the Maximum Softmax Probability (MSP) or the Maximum Logit Score (MLS) are strong baselines when the closed-set accuracy is high. However, one of the potential weaknesses of familiarity-based OSR are adversarial attacks. Here, we study gradient-based adversarial attacks on familiarity scores for both types of attacks, False Familiarity and False Novelty attacks, and evaluate their effectiveness in informed and uninformed settings on TinyImageNet. Furthermore, we explore how novel and familiar samples react to adversarial attacks and formulate the adversarial reaction score as an alternative OSR scoring rule, which shows a high correlation with the MLS familiarity score.



## **50. Safeguarding Large Language Models in Real-time with Tunable Safety-Performance Trade-offs**

cs.CL

**SubmitDate**: 2025-01-02    [abs](http://arxiv.org/abs/2501.02018v1) [paper-pdf](http://arxiv.org/pdf/2501.02018v1)

**Authors**: Joao Fonseca, Andrew Bell, Julia Stoyanovich

**Abstract**: Large Language Models (LLMs) have been shown to be susceptible to jailbreak attacks, or adversarial attacks used to illicit high risk behavior from a model. Jailbreaks have been exploited by cybercriminals and blackhat actors to cause significant harm, highlighting the critical need to safeguard widely-deployed models. Safeguarding approaches, which include fine-tuning models or having LLMs "self-reflect", may lengthen the inference time of a model, incur a computational penalty, reduce the semantic fluency of an output, and restrict ``normal'' model behavior. Importantly, these Safety-Performance Trade-offs (SPTs) remain an understudied area. In this work, we introduce a novel safeguard, called SafeNudge, that combines Controlled Text Generation with "nudging", or using text interventions to change the behavior of a model. SafeNudge triggers during text-generation while a jailbreak attack is being executed, and can reduce successful jailbreak attempts by 30% by guiding the LLM towards a safe responses. It adds minimal latency to inference and has a negligible impact on the semantic fluency of outputs. Further, we allow for tunable SPTs. SafeNudge is open-source and available through https://pypi.org/, and is compatible with models loaded with the Hugging Face "transformers" library.



