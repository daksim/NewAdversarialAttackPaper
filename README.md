# Latest Adversarial Attack Papers
**update at 2025-02-10 11:16:32**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. ADAPT to Robustify Prompt Tuning Vision Transformers**

cs.LG

Published in Transactions on Machine Learning Research (2025)

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2403.13196v2) [paper-pdf](http://arxiv.org/pdf/2403.13196v2)

**Authors**: Masih Eskandar, Tooba Imtiaz, Zifeng Wang, Jennifer Dy

**Abstract**: The performance of deep models, including Vision Transformers, is known to be vulnerable to adversarial attacks. Many existing defenses against these attacks, such as adversarial training, rely on full-model fine-tuning to induce robustness in the models. These defenses require storing a copy of the entire model, that can have billions of parameters, for each task. At the same time, parameter-efficient prompt tuning is used to adapt large transformer-based models to downstream tasks without the need to save large copies. In this paper, we examine parameter-efficient prompt tuning of Vision Transformers for downstream tasks under the lens of robustness. We show that previous adversarial defense methods, when applied to the prompt tuning paradigm, suffer from gradient obfuscation and are vulnerable to adaptive attacks. We introduce ADAPT, a novel framework for performing adaptive adversarial training in the prompt tuning paradigm. Our method achieves competitive robust accuracy of ~40% w.r.t. SOTA robustness methods using full-model fine-tuning, by tuning only ~1% of the number of parameters.



## **2. Do Unlearning Methods Remove Information from Language Model Weights?**

cs.LG

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2410.08827v3) [paper-pdf](http://arxiv.org/pdf/2410.08827v3)

**Authors**: Aghyad Deeb, Fabien Roger

**Abstract**: Large Language Models' knowledge of how to perform cyber-security attacks, create bioweapons, and manipulate humans poses risks of misuse. Previous work has proposed methods to unlearn this knowledge. Historically, it has been unclear whether unlearning techniques are removing information from the model weights or just making it harder to access. To disentangle these two objectives, we propose an adversarial evaluation method to test for the removal of information from model weights: we give an attacker access to some facts that were supposed to be removed, and using those, the attacker tries to recover other facts from the same distribution that cannot be guessed from the accessible facts. We show that using fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy when applied to current unlearning methods for information learned during pretraining, revealing the limitations of these methods in removing information from the model weights. Our results also suggest that unlearning evaluations that measure unlearning robustness on information learned during an additional fine-tuning phase may overestimate robustness compared to evaluations that attempt to unlearn information learned during pretraining.



## **3. Federated Learning for Anomaly Detection in Energy Consumption Data: Assessing the Vulnerability to Adversarial Attacks**

cs.LG

12th IEEE Conference on Technologies for Sustainability

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.05041v1) [paper-pdf](http://arxiv.org/pdf/2502.05041v1)

**Authors**: Yohannis Kifle Telila, Damitha Senevirathne, Dumindu Tissera, Apurva Narayan, Miriam A. M. Capretz, Katarina Grolinger

**Abstract**: Anomaly detection is crucial in the energy sector to identify irregular patterns indicating equipment failures, energy theft, or other issues. Machine learning techniques for anomaly detection have achieved great success, but are typically centralized, involving sharing local data with a central server which raises privacy and security concerns. Federated Learning (FL) has been gaining popularity as it enables distributed learning without sharing local data. However, FL depends on neural networks, which are vulnerable to adversarial attacks that manipulate data, leading models to make erroneous predictions. While adversarial attacks have been explored in the image domain, they remain largely unexplored in time series problems, especially in the energy domain. Moreover, the effect of adversarial attacks in the FL setting is also mostly unknown. This paper assesses the vulnerability of FL-based anomaly detection in energy data to adversarial attacks. Specifically, two state-of-the-art models, Long Short Term Memory (LSTM) and Transformers, are used to detect anomalies in an FL setting, and two white-box attack methods, Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD), are employed to perturb the data. The results show that FL is more sensitive to PGD attacks than to FGSM attacks, attributed to PGD's iterative nature, resulting in an accuracy drop of over 10% even with naive, weaker attacks. Moreover, FL is more affected by these attacks than centralized learning, highlighting the need for defense mechanisms in FL.



## **4. Robust Graph Learning Against Adversarial Evasion Attacks via Prior-Free Diffusion-Based Structure Purification**

cs.LG

Accepted for poster at WWW 2025

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.05000v1) [paper-pdf](http://arxiv.org/pdf/2502.05000v1)

**Authors**: Jiayi Luo, Qingyun Sun, Haonan Yuan, Xingcheng Fu, Jianxin Li

**Abstract**: Adversarial evasion attacks pose significant threats to graph learning, with lines of studies that have improved the robustness of Graph Neural Networks (GNNs). However, existing works rely on priors about clean graphs or attacking strategies, which are often heuristic and inconsistent. To achieve robust graph learning over different types of evasion attacks and diverse datasets, we investigate this problem from a prior-free structure purification perspective. Specifically, we propose a novel Diffusion-based Structure Purification framework named DiffSP, which creatively incorporates the graph diffusion model to learn intrinsic distributions of clean graphs and purify the perturbed structures by removing adversaries under the direction of the captured predictive patterns without relying on priors. DiffSP is divided into the forward diffusion process and the reverse denoising process, during which structure purification is achieved. To avoid valuable information loss during the forward process, we propose an LID-driven nonisotropic diffusion mechanism to selectively inject noise anisotropically. To promote semantic alignment between the clean graph and the purified graph generated during the reverse process, we reduce the generation uncertainty by the proposed graph transfer entropy guided denoising mechanism. Extensive experiments demonstrate the superior robustness of DiffSP against evasion attacks.



## **5. Securing 5G Bootstrapping: A Two-Layer IBS Authentication Protocol**

cs.CR

13 pages, 4 figures, 3 tables. This work has been submitted to the  IEEE for possible publication

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.04915v1) [paper-pdf](http://arxiv.org/pdf/2502.04915v1)

**Authors**: Yilu Dong, Rouzbeh Behnia, Attila A. Yavuz, Syed Rafiul Hussain

**Abstract**: The lack of authentication during the initial bootstrapping phase between cellular devices and base stations allows attackers to deploy fake base stations and send malicious messages to the devices. These attacks have been a long-existing problem in cellular networks, enabling adversaries to launch denial-of-service (DoS), information leakage, and location-tracking attacks. While some defense mechanisms are introduced in 5G, (e.g., encrypting user identifiers to mitigate IMSI catchers), the initial communication between devices and base stations remains unauthenticated, leaving a critical security gap. To address this, we propose E2IBS, a novel and efficient two-layer identity-based signature scheme designed for seamless integration with existing cellular protocols. We implement E2IBS on an open-source 5G stack and conduct a comprehensive performance evaluation against alternative solutions. Compared to the state-of-the-art Schnorr-HIBS, E2IBS reduces attack surfaces, enables fine-grained lawful interception, and achieves 2x speed in verification, making it a practical solution for securing 5G base station authentication.



## **6. From Allies to Adversaries: Manipulating LLM Tool-Calling through Adversarial Injection**

cs.CR

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2412.10198v2) [paper-pdf](http://arxiv.org/pdf/2412.10198v2)

**Authors**: Haowei Wang, Rupeng Zhang, Junjie Wang, Mingyang Li, Yuekai Huang, Dandan Wang, Qing Wang

**Abstract**: Tool-calling has changed Large Language Model (LLM) applications by integrating external tools, significantly enhancing their functionality across diverse tasks. However, this integration also introduces new security vulnerabilities, particularly in the tool scheduling mechanisms of LLM, which have not been extensively studied. To fill this gap, we present ToolCommander, a novel framework designed to exploit vulnerabilities in LLM tool-calling systems through adversarial tool injection. Our framework employs a well-designed two-stage attack strategy. Firstly, it injects malicious tools to collect user queries, then dynamically updates the injected tools based on the stolen information to enhance subsequent attacks. These stages enable ToolCommander to execute privacy theft, launch denial-of-service attacks, and even manipulate business competition by triggering unscheduled tool-calling. Notably, the ASR reaches 91.67% for privacy theft and hits 100% for denial-of-service and unscheduled tool calling in certain cases. Our work demonstrates that these vulnerabilities can lead to severe consequences beyond simple misuse of tool-calling systems, underscoring the urgent need for robust defensive strategies to secure LLM Tool-calling systems.



## **7. DMPA: Model Poisoning Attacks on Decentralized Federated Learning for Model Differences**

cs.LG

8 pages, 3 figures

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.04771v1) [paper-pdf](http://arxiv.org/pdf/2502.04771v1)

**Authors**: Chao Feng, Yunlong Li, Yuanzhe Gao, Alberto Huertas Celdrán, Jan von der Assen, Gérôme Bovet, Burkhard Stiller

**Abstract**: Federated learning (FL) has garnered significant attention as a prominent privacy-preserving Machine Learning (ML) paradigm. Decentralized FL (DFL) eschews traditional FL's centralized server architecture, enhancing the system's robustness and scalability. However, these advantages of DFL also create new vulnerabilities for malicious participants to execute adversarial attacks, especially model poisoning attacks. In model poisoning attacks, malicious participants aim to diminish the performance of benign models by creating and disseminating the compromised model. Existing research on model poisoning attacks has predominantly concentrated on undermining global models within the Centralized FL (CFL) paradigm, while there needs to be more research in DFL. To fill the research gap, this paper proposes an innovative model poisoning attack called DMPA. This attack calculates the differential characteristics of multiple malicious client models and obtains the most effective poisoning strategy, thereby orchestrating a collusive attack by multiple participants. The effectiveness of this attack is validated across multiple datasets, with results indicating that the DMPA approach consistently surpasses existing state-of-the-art FL model poisoning attack strategies.



## **8. Real-Time Privacy Risk Measurement with Privacy Tokens for Gradient Leakage**

cs.LG

There is something wrong with the order of Figures 8-11. And I need  to add an experiment with differential privacy quantization mutual  information value

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.02913v3) [paper-pdf](http://arxiv.org/pdf/2502.02913v3)

**Authors**: Jiayang Meng, Tao Huang, Hong Chen, Xin Shi, Qingyu Huang, Chen Hou

**Abstract**: The widespread deployment of deep learning models in privacy-sensitive domains has amplified concerns regarding privacy risks, particularly those stemming from gradient leakage during training. Current privacy assessments primarily rely on post-training attack simulations. However, these methods are inherently reactive, unable to encompass all potential attack scenarios, and often based on idealized adversarial assumptions. These limitations underscore the need for proactive approaches to privacy risk assessment during the training process. To address this gap, we propose the concept of privacy tokens, which are derived directly from private gradients during training. Privacy tokens encapsulate gradient features and, when combined with data features, offer valuable insights into the extent of private information leakage from training data, enabling real-time measurement of privacy risks without relying on adversarial attack simulations. Additionally, we employ Mutual Information (MI) as a robust metric to quantify the relationship between training data and gradients, providing precise and continuous assessments of privacy leakage throughout the training process. Extensive experiments validate our framework, demonstrating the effectiveness of privacy tokens and MI in identifying and quantifying privacy risks. This proactive approach marks a significant advancement in privacy monitoring, promoting the safer deployment of deep learning models in sensitive applications.



## **9. Mechanistic Understandings of Representation Vulnerabilities and Engineering Robust Vision Transformers**

cs.CV

10 pages, 5 figures

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.04679v1) [paper-pdf](http://arxiv.org/pdf/2502.04679v1)

**Authors**: Chashi Mahiul Islam, Samuel Jacob Chacko, Mao Nishino, Xiuwen Liu

**Abstract**: While transformer-based models dominate NLP and vision applications, their underlying mechanisms to map the input space to the label space semantically are not well understood. In this paper, we study the sources of known representation vulnerabilities of vision transformers (ViT), where perceptually identical images can have very different representations and semantically unrelated images can have the same representation. Our analysis indicates that imperceptible changes to the input can result in significant representation changes, particularly in later layers, suggesting potential instabilities in the performance of ViTs. Our comprehensive study reveals that adversarial effects, while subtle in early layers, propagate and amplify through the network, becoming most pronounced in middle to late layers. This insight motivates the development of NeuroShield-ViT, a novel defense mechanism that strategically neutralizes vulnerable neurons in earlier layers to prevent the cascade of adversarial effects. We demonstrate NeuroShield-ViT's effectiveness across various attacks, particularly excelling against strong iterative attacks, and showcase its remarkable zero-shot generalization capabilities. Without fine-tuning, our method achieves a competitive accuracy of 77.8% on adversarial examples, surpassing conventional robustness methods. Our results shed new light on how adversarial effects propagate through ViT layers, while providing a promising approach to enhance the robustness of vision transformers against adversarial attacks. Additionally, they provide a promising approach to enhance the robustness of vision transformers against adversarial attacks.



## **10. Confidence Elicitation: A New Attack Vector for Large Language Models**

cs.LG

Published in ICLR 2025. The code is publicly available at  https://github.com/Aniloid2/Confidence_Elicitation_Attacks

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.04643v1) [paper-pdf](http://arxiv.org/pdf/2502.04643v1)

**Authors**: Brian Formento, Chuan Sheng Foo, See-Kiong Ng

**Abstract**: A fundamental issue in deep learning has been adversarial robustness. As these systems have scaled, such issues have persisted. Currently, large language models (LLMs) with billions of parameters suffer from adversarial attacks just like their earlier, smaller counterparts. However, the threat models have changed. Previously, having gray-box access, where input embeddings or output logits/probabilities were visible to the user, might have been reasonable. However, with the introduction of closed-source models, no information about the model is available apart from the generated output. This means that current black-box attacks can only utilize the final prediction to detect if an attack is successful. In this work, we investigate and demonstrate the potential of attack guidance, akin to using output probabilities, while having only black-box access in a classification setting. This is achieved through the ability to elicit confidence from the model. We empirically show that the elicited confidence is calibrated and not hallucinated for current LLMs. By minimizing the elicited confidence, we can therefore increase the likelihood of misclassification. Our new proposed paradigm demonstrates promising state-of-the-art results on three datasets across two models (LLaMA-3-8B-Instruct and Mistral-7B-Instruct-V0.3) when comparing our technique to existing hard-label black-box attack methods that introduce word-level substitutions.



## **11. Regularized Robustly Reliable Learners and Instance Targeted Attacks**

cs.LG

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2410.10572v2) [paper-pdf](http://arxiv.org/pdf/2410.10572v2)

**Authors**: Avrim Blum, Donya Saless

**Abstract**: Instance-targeted data poisoning attacks, where an adversary corrupts a training set to induce errors on specific test points, have raised significant concerns. Balcan et al (2022) proposed an approach to addressing this challenge by defining a notion of robustly-reliable learners that provide per-instance guarantees of correctness under well-defined assumptions, even in the presence of data poisoning attacks. They then give a generic optimal (but computationally inefficient) robustly reliable learner as well as a computationally efficient algorithm for the case of linear separators over log-concave distributions.   In this work, we address two challenges left open by Balcan et al (2022). The first is that the definition of robustly-reliable learners in Balcan et al (2022) becomes vacuous for highly-flexible hypothesis classes: if there are two classifiers h_0, h_1 \in H both with zero error on the training set such that h_0(x) \neq h_1(x), then a robustly-reliable learner must abstain on x. We address this problem by defining a modified notion of regularized robustly-reliable learners that allows for nontrivial statements in this case. The second is that the generic algorithm of Balcan et al (2022) requires re-running an ERM oracle (essentially, retraining the classifier) on each test point x, which is generally impractical even if ERM can be implemented efficiently. To tackle this problem, we show that at least in certain interesting cases we can design algorithms that can produce their outputs in time sublinear in training time, by using techniques from dynamic algorithm design.



## **12. Adapting to Evolving Adversaries with Regularized Continual Robust Training**

cs.LG

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.04248v1) [paper-pdf](http://arxiv.org/pdf/2502.04248v1)

**Authors**: Sihui Dai, Christian Cianfarani, Arjun Bhagoji, Vikash Sehwag, Prateek Mittal

**Abstract**: Robust training methods typically defend against specific attack types, such as Lp attacks with fixed budgets, and rarely account for the fact that defenders may encounter new attacks over time. A natural solution is to adapt the defended model to new adversaries as they arise via fine-tuning, a method which we call continual robust training (CRT). However, when implemented naively, fine-tuning on new attacks degrades robustness on previous attacks. This raises the question: how can we improve the initial training and fine-tuning of the model to simultaneously achieve robustness against previous and new attacks? We present theoretical results which show that the gap in a model's robustness against different attacks is bounded by how far each attack perturbs a sample in the model's logit space, suggesting that regularizing with respect to this logit space distance can help maintain robustness against previous attacks. Extensive experiments on 3 datasets (CIFAR-10, CIFAR-100, and ImageNette) and over 100 attack combinations demonstrate that the proposed regularization improves robust accuracy with little overhead in training time. Our findings and open-source code lay the groundwork for the deployment of models robust to evolving attacks.



## **13. Provably Robust Explainable Graph Neural Networks against Graph Perturbation Attacks**

cs.CR

Accepted by ICLR 2025

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.04224v1) [paper-pdf](http://arxiv.org/pdf/2502.04224v1)

**Authors**: Jiate Li, Meng Pang, Yun Dong, Jinyuan Jia, Binghui Wang

**Abstract**: Explaining Graph Neural Network (XGNN) has gained growing attention to facilitate the trust of using GNNs, which is the mainstream method to learn graph data. Despite their growing attention, Existing XGNNs focus on improving the explanation performance, and its robustness under attacks is largely unexplored. We noticed that an adversary can slightly perturb the graph structure such that the explanation result of XGNNs is largely changed. Such vulnerability of XGNNs could cause serious issues particularly in safety/security-critical applications. In this paper, we take the first step to study the robustness of XGNN against graph perturbation attacks, and propose XGNNCert, the first provably robust XGNN. Particularly, our XGNNCert can provably ensure the explanation result for a graph under the worst-case graph perturbation attack is close to that without the attack, while not affecting the GNN prediction, when the number of perturbed edges is bounded. Evaluation results on multiple graph datasets and GNN explainers show the effectiveness of XGNNCert.



## **14. "Short-length" Adversarial Training Helps LLMs Defend "Long-length" Jailbreak Attacks: Theoretical and Empirical Evidence**

cs.LG

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.04204v1) [paper-pdf](http://arxiv.org/pdf/2502.04204v1)

**Authors**: Shaopeng Fu, Liang Ding, Di Wang

**Abstract**: Jailbreak attacks against large language models (LLMs) aim to induce harmful behaviors in LLMs through carefully crafted adversarial prompts. To mitigate attacks, one way is to perform adversarial training (AT)-based alignment, i.e., training LLMs on some of the most adversarial prompts to help them learn how to behave safely under attacks. During AT, the length of adversarial prompts plays a critical role in the robustness of aligned LLMs. This paper focuses on adversarial suffix jailbreak attacks and unveils that to defend against a jailbreak attack with an adversarial suffix of length $\Theta(M)$, it is enough to align LLMs on prompts with adversarial suffixes of length $\Theta(\sqrt{M})$. Theoretically, we analyze the adversarial in-context learning of linear transformers on linear regression tasks and prove a robust generalization bound for trained transformers. The bound depends on the term $\Theta(\sqrt{M_{\text{test}}}/M_{\text{train}})$, where $M_{\text{train}}$ and $M_{\text{test}}$ are the number of adversarially perturbed in-context samples during training and testing. Empirically, we conduct AT on popular open-source LLMs and evaluate their robustness against jailbreak attacks of different adversarial suffix lengths. Results confirm a positive correlation between the attack success rate and the ratio of the square root of the adversarial suffix during jailbreaking to the length during AT. Our findings show that it is practical to defend "long-length" jailbreak attacks via efficient "short-length" AT. The code is available at https://github.com/fshp971/adv-icl.



## **15. Assessing and Prioritizing Ransomware Risk Based on Historical Victim Data**

cs.CR

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.04421v1) [paper-pdf](http://arxiv.org/pdf/2502.04421v1)

**Authors**: Spencer Massengale, Philip Huff

**Abstract**: We present an approach to identifying which ransomware adversaries are most likely to target specific entities, thereby assisting these entities in formulating better protection strategies. Ransomware poses a formidable cybersecurity threat characterized by profit-driven motives, a complex underlying economy supporting criminal syndicates, and the overt nature of its attacks. This type of malware has consistently ranked among the most prevalent, with a rapid escalation in activity observed. Recent estimates indicate that approximately two-thirds of organizations experienced ransomware attacks in 2023 \cite{Sophos2023Ransomware}. A central tactic in ransomware campaigns is publicizing attacks to coerce victims into paying ransoms. Our study utilizes public disclosures from ransomware victims to predict the likelihood of an entity being targeted by a specific ransomware variant. We employ a Large Language Model (LLM) architecture that uses a unique chain-of-thought, multi-shot prompt methodology to define adversary SKRAM (Skills, Knowledge, Resources, Authorities, and Motivation) profiles from ransomware bulletins, threat reports, and news items. This analysis is enriched with publicly available victim data and is further enhanced by a heuristic for generating synthetic data that reflects victim profiles. Our work culminates in the development of a machine learning model that assists organizations in prioritizing ransomware threats and formulating defenses based on the tactics, techniques, and procedures (TTP) of the most likely attackers.



## **16. G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks**

cs.MA

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2410.11782v3) [paper-pdf](http://arxiv.org/pdf/2410.11782v3)

**Authors**: Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Tianlong Chen, Dawei Cheng

**Abstract**: Recent advancements in large language model (LLM)-based agents have demonstrated that collective intelligence can significantly surpass the capabilities of individual agents, primarily due to well-crafted inter-agent communication topologies. Despite the diverse and high-performing designs available, practitioners often face confusion when selecting the most effective pipeline for their specific task: \textit{Which topology is the best choice for my task, avoiding unnecessary communication token overhead while ensuring high-quality solution?} In response to this dilemma, we introduce G-Designer, an adaptive, efficient, and robust solution for multi-agent deployment, which dynamically designs task-aware, customized communication topologies. Specifically, G-Designer models the multi-agent system as a multi-agent network, leveraging a variational graph auto-encoder to encode both the nodes (agents) and a task-specific virtual node, and decodes a task-adaptive and high-performing communication topology. Extensive experiments on six benchmarks showcase that G-Designer is: \textbf{(1) high-performing}, achieving superior results on MMLU with accuracy at $84.50\%$ and on HumanEval with pass@1 at $89.90\%$; \textbf{(2) task-adaptive}, architecting communication protocols tailored to task difficulty, reducing token consumption by up to $95.33\%$ on HumanEval; and \textbf{(3) adversarially robust}, defending against agent adversarial attacks with merely $0.3\%$ accuracy drop.



## **17. Adversarial Attacks for Drift Detection**

cs.LG

Accepted at ESANN 2025

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2411.16591v2) [paper-pdf](http://arxiv.org/pdf/2411.16591v2)

**Authors**: Fabian Hinder, Valerie Vaquet, Barbara Hammer

**Abstract**: Concept drift refers to the change of data distributions over time. While drift poses a challenge for learning models, requiring their continual adaption, it is also relevant in system monitoring to detect malfunctions, system failures, and unexpected behavior. In the latter case, the robust and reliable detection of drifts is imperative. This work studies the shortcomings of commonly used drift detection schemes. We show how to construct data streams that are drifting without being detected. We refer to those as drift adversarials. In particular, we compute all possible adversairals for common detection schemes and underpin our theoretical findings with empirical evaluations.



## **18. The Gradient Puppeteer: Adversarial Domination in Gradient Leakage Attacks through Model Poisoning**

cs.CR

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.04106v1) [paper-pdf](http://arxiv.org/pdf/2502.04106v1)

**Authors**: Kunlan Xiang, Haomiao Yang, Meng Hao, Haoxin Wang, Shaofeng Li, Zikang Ding, Tianwei Zhang

**Abstract**: In Federated Learning (FL), clients share gradients with a central server while keeping their data local. However, malicious servers could deliberately manipulate the models to reconstruct clients' data from shared gradients, posing significant privacy risks. Although such active gradient leakage attacks (AGLAs) have been widely studied, they suffer from several limitations including incomplete attack coverage and poor stealthiness. In this paper, we address these limitations with two core contributions. First, we introduce a new theoretical analysis approach, which uniformly models AGLAs as backdoor poisoning. This analysis approach reveals that the core principle of AGLAs is to bias the gradient space to prioritize the reconstruction of a small subset of samples while sacrificing the majority, which theoretically explains the above limitations of existing AGLAs. Second, we propose Enhanced Gradient Global Vulnerability (EGGV), the first AGLA that achieves complete attack coverage while evading client-side detection. In particular, EGGV employs a gradient projector and a jointly optimized discriminator to assess gradient vulnerability, steering the gradient space toward the point most prone to data leakage. Extensive experiments show that EGGV achieves complete attack coverage and surpasses SOTA with at least a 43% increase in reconstruction quality (PSNR) and a 45% improvement in stealthiness (D-SNR).



## **19. Consumer INS Coupled with Carrier Phase Measurements for GNSS Spoofing Detection**

cs.CR

Presented at ION ITM/PTTI 2025

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.03870v1) [paper-pdf](http://arxiv.org/pdf/2502.03870v1)

**Authors**: Tore Johansson, Marco Spanghero, Panos Papadimitratos

**Abstract**: Global Navigation Satellite Systems enable precise localization and timing even for highly mobile devices, but legacy implementations provide only limited support for the new generation of security-enhanced signals. Inertial Measurement Units have proved successful in augmenting the accuracy and robustness of the GNSS-provided navigation solution, but effective navigation based on inertial techniques in denied contexts requires high-end sensors. However, commercially available mobile devices usually embed a much lower-grade inertial system. To counteract an attacker transmitting all the adversarial signals from a single antenna, we exploit carrier phase-based observations coupled with a low-end inertial sensor to identify spoofing and meaconing. By short-time integration with an inertial platform, which tracks the displacement of the GNSS antenna, the high-frequency movement at the receiver is correlated with the variation in the carrier phase. In this way, we identify legitimate transmitters, based on their geometrical diversity with respect to the antenna system movement. We introduce a platform designed to effectively compare different tiers of commercial INS platforms with a GNSS receiver. By characterizing different inertial sensors, we show that simple MEMS INS perform as well as high-end industrial-grade sensors. Sensors traditionally considered unsuited for navigation purposes offer great performance at the short integration times used to evaluate the carrier phase information consistency against the high-frequency movement. Results from laboratory evaluation and through field tests at Jammertest 2024 show that the detector is up to 90% accurate in correctly identifying spoofing (or the lack of it), without any modification to the receiver structure, and with mass-production grade INS typical for mobile phones.



## **20. Time-based GNSS attack detection**

cs.CR

IEEE Transactions on Aerospace and Electronic Systems (Early Access)

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.03868v1) [paper-pdf](http://arxiv.org/pdf/2502.03868v1)

**Authors**: Marco Spanghero, Panos Papadimitratos

**Abstract**: To safeguard Civilian Global Navigation Satellite Systems (GNSS) external information available to the platform encompassing the GNSS receiver can be used to detect attacks. Cross-checking the GNSS-provided time against alternative multiple trusted time sources can lead to attack detection aiming at controlling the GNSS receiver time. Leveraging external, network-connected secure time providers and onboard clock references, we achieve detection even under fine-grained time attacks. We provide an extensive evaluation of our multi-layered defense against adversaries mounting attacks against the GNSS receiver along with controlling the network link. We implement adversaries spanning from simplistic spoofers to advanced ones synchronized with the GNSS constellation. We demonstrate attack detection is possible in all tested cases (sharp discontinuity, smooth take-over, and coordinated network manipulation) without changes to the structure of the GNSS receiver. Leveraging the diversity of the reference time sources, detection of take-over time push as low as 150us is possible. Smooth take-overs forcing variations as low as 30ns are also detected based on on-board precision oscillators. The method (and thus the evaluation) is largely agnostic to the satellite constellation and the attacker type, making time-based data validation of GNSS information compatible with existing receivers and readily deployable.



## **21. On Robust Reinforcement Learning with Lipschitz-Bounded Policy Networks**

cs.LG

Accepted to the Symposium on Systems Theory in Data and Optimization  (SysDO 2024)

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2405.11432v3) [paper-pdf](http://arxiv.org/pdf/2405.11432v3)

**Authors**: Nicholas H. Barbara, Ruigang Wang, Ian R. Manchester

**Abstract**: This paper presents a study of robust policy networks in deep reinforcement learning. We investigate the benefits of policy parameterizations that naturally satisfy constraints on their Lipschitz bound, analyzing their empirical performance and robustness on two representative problems: pendulum swing-up and Atari Pong. We illustrate that policy networks with smaller Lipschitz bounds are more robust to disturbances, random noise, and targeted adversarial attacks than unconstrained policies composed of vanilla multi-layer perceptrons or convolutional neural networks. However, the structure of the Lipschitz layer is important. We find that the widely-used method of spectral normalization is too conservative and severely impacts clean performance, whereas more expressive Lipschitz layers such as the recently-proposed Sandwich layer can achieve improved robustness without sacrificing clean performance.



## **22. On Effects of Steering Latent Representation for Large Language Model Unlearning**

cs.CL

Accepted at AAAI-25 Main Technical Track

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2408.06223v3) [paper-pdf](http://arxiv.org/pdf/2408.06223v3)

**Authors**: Dang Huu-Tien, Trung-Tin Pham, Hoang Thanh-Tung, Naoya Inoue

**Abstract**: Representation Misdirection for Unlearning (RMU), which steers model representation in the intermediate layer to a target random representation, is an effective method for large language model (LLM) unlearning. Despite its high performance, the underlying cause and explanation remain underexplored. In this paper, we theoretically demonstrate that steering forget representations in the intermediate layer reduces token confidence, causing LLMs to generate wrong or nonsense responses. We investigate how the coefficient influences the alignment of forget-sample representations with the random direction and hint at the optimal coefficient values for effective unlearning across different network layers. We show that RMU unlearned models are robust against adversarial jailbreak attacks. Furthermore, our empirical analysis shows that RMU is less effective when applied to the middle and later layers in LLMs. To resolve this drawback, we propose Adaptive RMU--a simple yet effective alternative method that makes unlearning effective with most layers. Extensive experiments demonstrate that Adaptive RMU significantly improves the unlearning performance compared to prior art while incurring no additional computational cost.



## **23. How vulnerable is my policy? Adversarial attacks on modern behavior cloning policies**

cs.LG

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.03698v1) [paper-pdf](http://arxiv.org/pdf/2502.03698v1)

**Authors**: Basavasagar Patil, Akansha Kalra, Guanhong Tao, Daniel S. Brown

**Abstract**: Learning from Demonstration (LfD) algorithms have shown promising results in robotic manipulation tasks, but their vulnerability to adversarial attacks remains underexplored. This paper presents a comprehensive study of adversarial attacks on both classic and recently proposed algorithms, including Behavior Cloning (BC), LSTM-GMM, Implicit Behavior Cloning (IBC), Diffusion Policy (DP), and VQ-Behavior Transformer (VQ-BET). We study the vulnerability of these methods to untargeted, targeted and universal adversarial perturbations. While explicit policies, such as BC, LSTM-GMM and VQ-BET can be attacked in the same manner as standard computer vision models, we find that attacks for implicit and denoising policy models are nuanced and require developing novel attack methods. Our experiments on several simulated robotic manipulation tasks reveal that most of the current methods are highly vulnerable to adversarial perturbations. We also show that these attacks are transferable across algorithms, architectures, and tasks, raising concerning security vulnerabilities with potentially a white-box threat model. In addition, we test the efficacy of a randomized smoothing, a widely used adversarial defense technique, and highlight its limitation in defending against attacks on complex and multi-modal action distribution common in complex control tasks. In summary, our findings highlight the vulnerabilities of modern BC algorithms, paving way for future work in addressing such limitations.



## **24. DocMIA: Document-Level Membership Inference Attacks against DocVQA Models**

cs.LG

ICLR 2025

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.03692v1) [paper-pdf](http://arxiv.org/pdf/2502.03692v1)

**Authors**: Khanh Nguyen, Raouf Kerkouche, Mario Fritz, Dimosthenis Karatzas

**Abstract**: Document Visual Question Answering (DocVQA) has introduced a new paradigm for end-to-end document understanding, and quickly became one of the standard benchmarks for multimodal LLMs. Automating document processing workflows, driven by DocVQA models, presents significant potential for many business sectors. However, documents tend to contain highly sensitive information, raising concerns about privacy risks associated with training such DocVQA models. One significant privacy vulnerability, exploited by the membership inference attack, is the possibility for an adversary to determine if a particular record was part of the model's training data. In this paper, we introduce two novel membership inference attacks tailored specifically to DocVQA models. These attacks are designed for two different adversarial scenarios: a white-box setting, where the attacker has full access to the model architecture and parameters, and a black-box setting, where only the model's outputs are available. Notably, our attacks assume the adversary lacks access to auxiliary datasets, which is more realistic in practice but also more challenging. Our unsupervised methods outperform existing state-of-the-art membership inference attacks across a variety of DocVQA models and datasets, demonstrating their effectiveness and highlighting the privacy risks in this domain.



## **25. Towards Fair Medical AI: Adversarial Debiasing of 3D CT Foundation Embeddings**

cs.CV

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.04386v1) [paper-pdf](http://arxiv.org/pdf/2502.04386v1)

**Authors**: Guangyao Zheng, Michael A. Jacobs, Vladimir Braverman, Vishwa S. Parekh

**Abstract**: Self-supervised learning has revolutionized medical imaging by enabling efficient and generalizable feature extraction from large-scale unlabeled datasets. Recently, self-supervised foundation models have been extended to three-dimensional (3D) computed tomography (CT) data, generating compact, information-rich embeddings with 1408 features that achieve state-of-the-art performance on downstream tasks such as intracranial hemorrhage detection and lung cancer risk forecasting. However, these embeddings have been shown to encode demographic information, such as age, sex, and race, which poses a significant risk to the fairness of clinical applications.   In this work, we propose a Variation Autoencoder (VAE) based adversarial debiasing framework to transform these embeddings into a new latent space where demographic information is no longer encoded, while maintaining the performance of critical downstream tasks. We validated our approach on the NLST lung cancer screening dataset, demonstrating that the debiased embeddings effectively eliminate multiple encoded demographic information and improve fairness without compromising predictive accuracy for lung cancer risk at 1-year and 2-year intervals. Additionally, our approach ensures the embeddings are robust against adversarial bias attacks. These results highlight the potential of adversarial debiasing techniques to ensure fairness and equity in clinical applications of self-supervised 3D CT embeddings, paving the way for their broader adoption in unbiased medical decision-making.



## **26. OverThink: Slowdown Attacks on Reasoning LLMs**

cs.LG

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.02542v2) [paper-pdf](http://arxiv.org/pdf/2502.02542v2)

**Authors**: Abhinav Kumar, Jaechul Roh, Ali Naseh, Marzena Karpinska, Mohit Iyyer, Amir Houmansadr, Eugene Bagdasarian

**Abstract**: We increase overhead for applications that rely on reasoning LLMs-we force models to spend an amplified number of reasoning tokens, i.e., "overthink", to respond to the user query while providing contextually correct answers. The adversary performs an OVERTHINK attack by injecting decoy reasoning problems into the public content that is used by the reasoning LLM (e.g., for RAG applications) during inference time. Due to the nature of our decoy problems (e.g., a Markov Decision Process), modified texts do not violate safety guardrails. We evaluated our attack across closed-(OpenAI o1, o1-mini, o3-mini) and open-(DeepSeek R1) weights reasoning models on the FreshQA and SQuAD datasets. Our results show up to 18x slowdown on FreshQA dataset and 46x slowdown on SQuAD dataset. The attack also shows high transferability across models. To protect applications, we discuss and implement defenses leveraging LLM-based and system design approaches. Finally, we discuss societal, financial, and energy impacts of OVERTHINK attack which could amplify the costs for third-party applications operating reasoning models.



## **27. Dual-Flow: Transferable Multi-Target, Instance-Agnostic Attacks via In-the-wild Cascading Flow Optimization**

cs.CV

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.02096v2) [paper-pdf](http://arxiv.org/pdf/2502.02096v2)

**Authors**: Yixiao Chen, Shikun Sun, Jianshu Li, Ruoyu Li, Zhe Li, Junliang Xing

**Abstract**: Adversarial attacks are widely used to evaluate model robustness, and in black-box scenarios, the transferability of these attacks becomes crucial. Existing generator-based attacks have excellent generalization and transferability due to their instance-agnostic nature. However, when training generators for multi-target tasks, the success rate of transfer attacks is relatively low due to the limitations of the model's capacity. To address these challenges, we propose a novel Dual-Flow framework for multi-target instance-agnostic adversarial attacks, utilizing Cascading Distribution Shift Training to develop an adversarial velocity function. Extensive experiments demonstrate that Dual-Flow significantly improves transferability over previous multi-target generative attacks. For example, it increases the success rate from Inception-v3 to ResNet-152 by 34.58%. Furthermore, our attack method shows substantially stronger robustness against defense mechanisms, such as adversarially trained models.



## **28. Understanding and Enhancing the Transferability of Jailbreaking Attacks**

cs.LG

Accepted by ICLR 2025

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.03052v1) [paper-pdf](http://arxiv.org/pdf/2502.03052v1)

**Authors**: Runqi Lin, Bo Han, Fengwang Li, Tongling Liu

**Abstract**: Jailbreaking attacks can effectively manipulate open-source large language models (LLMs) to produce harmful responses. However, these attacks exhibit limited transferability, failing to disrupt proprietary LLMs consistently. To reliably identify vulnerabilities in proprietary LLMs, this work investigates the transferability of jailbreaking attacks by analysing their impact on the model's intent perception. By incorporating adversarial sequences, these attacks can redirect the source LLM's focus away from malicious-intent tokens in the original input, thereby obstructing the model's intent recognition and eliciting harmful responses. Nevertheless, these adversarial sequences fail to mislead the target LLM's intent perception, allowing the target LLM to refocus on malicious-intent tokens and abstain from responding. Our analysis further reveals the inherent distributional dependency within the generated adversarial sequences, whose effectiveness stems from overfitting the source LLM's parameters, resulting in limited transferability to target LLMs. To this end, we propose the Perceived-importance Flatten (PiF) method, which uniformly disperses the model's focus across neutral-intent tokens in the original input, thus obscuring malicious-intent tokens without relying on overfitted adversarial sequences. Extensive experiments demonstrate that PiF provides an effective and efficient red-teaming evaluation for proprietary LLMs.



## **29. SelfDefend: LLMs Can Defend Themselves against Jailbreaking in a Practical Manner**

cs.CR

Accepted by USENIX Security Symposium 2025. Please cite the  conference version of this paper, i.e., "Xunguang Wang, Daoyuan Wu, Zhenlan  Ji, Zongjie Li, Pingchuan Ma, Shuai Wang, Yingjiu Li, Yang Liu, Ning Liu, and  Juergen Rahmel. SelfDefend: LLMs Can Defend Themselves against Jailbreaking  in a Practical Manner. In Proc. USENIX Security, 2025."

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2406.05498v3) [paper-pdf](http://arxiv.org/pdf/2406.05498v3)

**Authors**: Xunguang Wang, Daoyuan Wu, Zhenlan Ji, Zongjie Li, Pingchuan Ma, Shuai Wang, Yingjiu Li, Yang Liu, Ning Liu, Juergen Rahmel

**Abstract**: Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs) and has evolved into multiple categories: human-based, optimization-based, generation-based, and the recent indirect and multilingual jailbreaks. However, delivering a practical jailbreak defense is challenging because it needs to not only handle all the above jailbreak attacks but also incur negligible delays to user prompts, as well as be compatible with both open-source and closed-source LLMs. Inspired by how the traditional security concept of shadow stacks defends against memory overflow attacks, this paper introduces a generic LLM jailbreak defense framework called SelfDefend, which establishes a shadow LLM as a defense instance (in detection state) to concurrently protect the target LLM instance (in normal answering state) in the normal stack and collaborate with it for checkpoint-based access control. The effectiveness of SelfDefend builds upon our observation that existing LLMs can identify harmful prompts or intentions in user queries, which we empirically validate using mainstream GPT-3.5/4 models against major jailbreak attacks. To further improve the defense's robustness and minimize costs, we employ a data distillation approach to tune dedicated open-source defense models. When deployed to protect GPT-3.5/4, Claude, Llama-2-7b/13b, and Mistral, these models outperform seven state-of-the-art defenses and match the performance of GPT-4-based SelfDefend, with significantly lower extra delays. Further experiments show that the tuned models are robust to adaptive jailbreaks and prompt injections.



## **30. Large Language Model Adversarial Landscape Through the Lens of Attack Objectives**

cs.CR

15 pages

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.02960v1) [paper-pdf](http://arxiv.org/pdf/2502.02960v1)

**Authors**: Nan Wang, Kane Walter, Yansong Gao, Alsharif Abuadbba

**Abstract**: Large Language Models (LLMs) represent a transformative leap in artificial intelligence, enabling the comprehension, generation, and nuanced interaction with human language on an unparalleled scale. However, LLMs are increasingly vulnerable to a range of adversarial attacks that threaten their privacy, reliability, security, and trustworthiness. These attacks can distort outputs, inject biases, leak sensitive information, or disrupt the normal functioning of LLMs, posing significant challenges across various applications.   In this paper, we provide a novel comprehensive analysis of the adversarial landscape of LLMs, framed through the lens of attack objectives. By concentrating on the core goals of adversarial actors, we offer a fresh perspective that examines threats from the angles of privacy, integrity, availability, and misuse, moving beyond conventional taxonomies that focus solely on attack techniques. This objective-driven adversarial landscape not only highlights the strategic intent behind different adversarial approaches but also sheds light on the evolving nature of these threats and the effectiveness of current defenses. Our analysis aims to guide researchers and practitioners in better understanding, anticipating, and mitigating these attacks, ultimately contributing to the development of more resilient and robust LLM systems.



## **31. Wolfpack Adversarial Attack for Robust Multi-Agent Reinforcement Learning**

cs.LG

8 pages main, 21 pages appendix with reference. Submitted to ICML  2025

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.02844v1) [paper-pdf](http://arxiv.org/pdf/2502.02844v1)

**Authors**: Sunwoo Lee, Jaebak Hwang, Yonghyeon Jo, Seungyul Han

**Abstract**: Traditional robust methods in multi-agent reinforcement learning (MARL) often struggle against coordinated adversarial attacks in cooperative scenarios. To address this limitation, we propose the Wolfpack Adversarial Attack framework, inspired by wolf hunting strategies, which targets an initial agent and its assisting agents to disrupt cooperation. Additionally, we introduce the Wolfpack-Adversarial Learning for MARL (WALL) framework, which trains robust MARL policies to defend against the proposed Wolfpack attack by fostering system-wide collaboration. Experimental results underscore the devastating impact of the Wolfpack attack and the significant robustness improvements achieved by WALL.



## **32. MARAGE: Transferable Multi-Model Adversarial Attack for Retrieval-Augmented Generation Data Extraction**

cs.CL

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.04360v1) [paper-pdf](http://arxiv.org/pdf/2502.04360v1)

**Authors**: Xiao Hu, Eric Liu, Weizhou Wang, Xiangyu Guo, David Lie

**Abstract**: Retrieval-Augmented Generation (RAG) offers a solution to mitigate hallucinations in Large Language Models (LLMs) by grounding their outputs to knowledge retrieved from external sources. The use of private resources and data in constructing these external data stores can expose them to risks of extraction attacks, in which attackers attempt to steal data from these private databases. Existing RAG extraction attacks often rely on manually crafted prompts, which limit their effectiveness. In this paper, we introduce a framework called MARAGE for optimizing an adversarial string that, when appended to user queries submitted to a target RAG system, causes outputs containing the retrieved RAG data verbatim. MARAGE leverages a continuous optimization scheme that integrates gradients from multiple models with different architectures simultaneously to enhance the transferability of the optimized string to unseen models. Additionally, we propose a strategy that emphasizes the initial tokens in the target RAG data, further improving the attack's generalizability. Evaluations show that MARAGE consistently outperforms both manual and optimization-based baselines across multiple LLMs and RAG datasets, while maintaining robust transferability to previously unseen models. Moreover, we conduct probing tasks to shed light on the reasons why MARAGE is more effective compared to the baselines and to analyze the impact of our approach on the model's internal state.



## **33. Semantic Entanglement-Based Ransomware Detection via Probabilistic Latent Encryption Mapping**

cs.CR

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02730v1) [paper-pdf](http://arxiv.org/pdf/2502.02730v1)

**Authors**: Mohammad Eisa, Quentin Yardley, Rafael Witherspoon, Harriet Pendlebury, Clement Rutherford

**Abstract**: Encryption-based attacks have introduced significant challenges for detection mechanisms that rely on predefined signatures, heuristic indicators, or static rule-based classifications. Probabilistic Latent Encryption Mapping presents an alternative detection framework that models ransomware-induced encryption behaviors through statistical representations of entropy deviations and probabilistic dependencies in execution traces. Unlike conventional approaches that depend on explicit bytecode analysis or predefined cryptographic function call monitoring, probabilistic inference techniques classify encryption anomalies based on their underlying statistical characteristics, ensuring greater adaptability to polymorphic attack strategies. Evaluations demonstrate that entropy-driven classification reduces false positive rates while maintaining high detection accuracy across diverse ransomware families and encryption methodologies. Experimental results further highlight the framework's ability to differentiate between benign encryption workflows and adversarial cryptographic manipulations, ensuring that classification performance remains effective across cloud-based and localized execution environments. Benchmark comparisons illustrate that probabilistic modeling exhibits advantages over heuristic and machine learning-based detection approaches, particularly in handling previously unseen encryption techniques and adversarial obfuscation strategies. Computational efficiency analysis confirms that detection latency remains within operational feasibility constraints, reinforcing the viability of probabilistic encryption classification for real-time security infrastructures. The ability to systematically infer encryption-induced deviations without requiring static attack signatures strengthens detection robustness against adversarial evasion techniques.



## **34. Enforcing Demographic Coherence: A Harms Aware Framework for Reasoning about Private Data Release**

cs.CR

42 pages

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02709v1) [paper-pdf](http://arxiv.org/pdf/2502.02709v1)

**Authors**: Mark Bun, Marco Carmosino, Palak Jain, Gabriel Kaptchuk, Satchit Sivakumar

**Abstract**: The technical literature about data privacy largely consists of two complementary approaches: formal definitions of conditions sufficient for privacy preservation and attacks that demonstrate privacy breaches. Differential privacy is an accepted standard in the former sphere. However, differential privacy's powerful adversarial model and worst-case guarantees may make it too stringent in some situations, especially when achieving it comes at a significant cost to data utility. Meanwhile, privacy attacks aim to expose real and worrying privacy risks associated with existing data release processes but often face criticism for being unrealistic. Moreover, the literature on attacks generally does not identify what properties are necessary to defend against them.   We address the gap between these approaches by introducing demographic coherence, a condition inspired by privacy attacks that we argue is necessary for data privacy. This condition captures privacy violations arising from inferences about individuals that are incoherent with respect to the demographic patterns in the data. Our framework focuses on confidence rated predictors, which can in turn be distilled from almost any data-informed process. Thus, we capture privacy threats that exist even when no attack is explicitly being carried out. Our framework not only provides a condition with respect to which data release algorithms can be analysed but suggests natural experimental evaluation methodologies that could be used to build practical intuition and make tangible assessment of risks. Finally, we argue that demographic coherence is weaker than differential privacy: we prove that every differentially private data release is also demographically coherent, and that there are demographically coherent algorithms which are not differentially private.



## **35. DiffBreak: Breaking Diffusion-Based Purification with Adaptive Attacks**

cs.CR

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2411.16598v2) [paper-pdf](http://arxiv.org/pdf/2411.16598v2)

**Authors**: Andre Kassis, Urs Hengartner, Yaoliang Yu

**Abstract**: Diffusion-based purification (DBP) has emerged as a cornerstone defense against adversarial examples (AEs), widely regarded as robust due to its use of diffusion models (DMs) that project AEs onto the natural data distribution. However, contrary to prior assumptions, we theoretically prove that adaptive gradient-based attacks nullify this foundational claim, effectively targeting the DM rather than the classifier and causing purified outputs to align with adversarial distributions. This surprising discovery prompts a reassessment of DBP's robustness, revealing it stems from critical flaws in backpropagation techniques used so far for attacking DBP. To address these gaps, we introduce DiffBreak, a novel and reliable gradient library for DBP, which exposes how adaptive attacks drastically degrade its robustness. In stricter majority-vote settings, where classifier decisions aggregate predictions over multiple purified inputs, DBP retains partial robustness to traditional norm-bounded AEs due to its stochasticity disrupting adversarial alignment. However, we propose a novel adaptation of a recent optimization method against deepfake watermarking, crafting systemic adversarial perturbations that defeat DBP even under these conditions, ultimately challenging its viability as a defense without improvements.



## **36. Dissecting Adversarial Robustness of Multimodal LM Agents**

cs.LG

ICLR 2025. Also oral at NeurIPS 2024 Open-World Agents Workshop

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2406.12814v3) [paper-pdf](http://arxiv.org/pdf/2406.12814v3)

**Authors**: Chen Henry Wu, Rishi Shah, Jing Yu Koh, Ruslan Salakhutdinov, Daniel Fried, Aditi Raghunathan

**Abstract**: As language models (LMs) are used to build autonomous agents in real environments, ensuring their adversarial robustness becomes a critical challenge. Unlike chatbots, agents are compound systems with multiple components taking actions, which existing LMs safety evaluations do not adequately address. To bridge this gap, we manually create 200 targeted adversarial tasks and evaluation scripts in a realistic threat model on top of VisualWebArena, a real environment for web agents. To systematically examine the robustness of agents, we propose the Agent Robustness Evaluation (ARE) framework. ARE views the agent as a graph showing the flow of intermediate outputs between components and decomposes robustness as the flow of adversarial information on the graph. We find that we can successfully break latest agents that use black-box frontier LMs, including those that perform reflection and tree search. With imperceptible perturbations to a single image (less than 5% of total web page pixels), an attacker can hijack these agents to execute targeted adversarial goals with success rates up to 67%. We also use ARE to rigorously evaluate how the robustness changes as new components are added. We find that inference-time compute that typically improves benign performance can open up new vulnerabilities and harm robustness. An attacker can compromise the evaluator used by the reflexion agent and the value function of the tree search agent, which increases the attack success relatively by 15% and 20%. Our data and code for attacks, defenses, and evaluation are at https://github.com/ChenWu98/agent-attack



## **37. Certifying LLM Safety against Adversarial Prompting**

cs.CL

Accepted at COLM 2024: https://openreview.net/forum?id=9Ik05cycLq

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2309.02705v4) [paper-pdf](http://arxiv.org/pdf/2309.02705v4)

**Authors**: Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Aaron Jiaxun Li, Soheil Feizi, Himabindu Lakkaraju

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that add malicious tokens to an input prompt to bypass the safety guardrails of an LLM and cause it to produce harmful content. In this work, we introduce erase-and-check, the first framework for defending against adversarial prompts with certifiable safety guarantees. Given a prompt, our procedure erases tokens individually and inspects the resulting subsequences using a safety filter. Our safety certificate guarantees that harmful prompts are not mislabeled as safe due to an adversarial attack up to a certain size. We implement the safety filter in two ways, using Llama 2 and DistilBERT, and compare the performance of erase-and-check for the two cases. We defend against three attack modes: i) adversarial suffix, where an adversarial sequence is appended at the end of a harmful prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Our experimental results demonstrate that this procedure can obtain strong certified safety guarantees on harmful prompts while maintaining good empirical performance on safe prompts. Additionally, we propose three efficient empirical defenses: i) RandEC, a randomized subsampling version of erase-and-check; ii) GreedyEC, which greedily erases tokens that maximize the softmax score of the harmful class; and iii) GradEC, which uses gradient information to optimize tokens to erase. We demonstrate their effectiveness against adversarial prompts generated by the Greedy Coordinate Gradient (GCG) attack algorithm. The code for our experiments is available at https://github.com/aounon/certified-llm-safety.



## **38. Uncertainty Quantification for Collaborative Object Detection Under Adversarial Attacks**

cs.CV

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02537v1) [paper-pdf](http://arxiv.org/pdf/2502.02537v1)

**Authors**: Huiqun Huang, Cong Chen, Jean-Philippe Monteuuis, Jonathan Petit, Fei Miao

**Abstract**: Collaborative Object Detection (COD) and collaborative perception can integrate data or features from various entities, and improve object detection accuracy compared with individual perception. However, adversarial attacks pose a potential threat to the deep learning COD models, and introduce high output uncertainty. With unknown attack models, it becomes even more challenging to improve COD resiliency and quantify the output uncertainty for highly dynamic perception scenes such as autonomous vehicles. In this study, we propose the Trusted Uncertainty Quantification in Collaborative Perception framework (TUQCP). TUQCP leverages both adversarial training and uncertainty quantification techniques to enhance the adversarial robustness of existing COD models. More specifically, TUQCP first adds perturbations to the shared information of randomly selected agents during object detection collaboration by adversarial training. TUQCP then alleviates the impacts of adversarial attacks by providing output uncertainty estimation through learning-based module and uncertainty calibration through conformal prediction. Our framework works for early and intermediate collaboration COD models and single-agent object detection models. We evaluate TUQCP on V2X-Sim, a comprehensive collaborative perception dataset for autonomous driving, and demonstrate a 80.41% improvement in object detection accuracy compared to the baselines under the same adversarial attacks. TUQCP demonstrates the importance of uncertainty quantification to COD under adversarial attacks.



## **39. The TIP of the Iceberg: Revealing a Hidden Class of Task-in-Prompt Adversarial Attacks on LLMs**

cs.CR

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2501.18626v3) [paper-pdf](http://arxiv.org/pdf/2501.18626v3)

**Authors**: Sergey Berezin, Reza Farahbakhsh, Noel Crespi

**Abstract**: We present a novel class of jailbreak adversarial attacks on LLMs, termed Task-in-Prompt (TIP) attacks. Our approach embeds sequence-to-sequence tasks (e.g., cipher decoding, riddles, code execution) into the model's prompt to indirectly generate prohibited inputs. To systematically assess the effectiveness of these attacks, we introduce the PHRYGE benchmark. We demonstrate that our techniques successfully circumvent safeguards in six state-of-the-art language models, including GPT-4o and LLaMA 3.2. Our findings highlight critical weaknesses in current LLM safety alignments and underscore the urgent need for more sophisticated defence strategies.   Warning: this paper contains examples of unethical inquiries used solely for research purposes.



## **40. Medical Multimodal Model Stealing Attacks via Adversarial Domain Alignment**

cs.CR

Accepted at AAAI 2025

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02438v1) [paper-pdf](http://arxiv.org/pdf/2502.02438v1)

**Authors**: Yaling Shen, Zhixiong Zhuang, Kun Yuan, Maria-Irina Nicolae, Nassir Navab, Nicolas Padoy, Mario Fritz

**Abstract**: Medical multimodal large language models (MLLMs) are becoming an instrumental part of healthcare systems, assisting medical personnel with decision making and results analysis. Models for radiology report generation are able to interpret medical imagery, thus reducing the workload of radiologists. As medical data is scarce and protected by privacy regulations, medical MLLMs represent valuable intellectual property. However, these assets are potentially vulnerable to model stealing, where attackers aim to replicate their functionality via black-box access. So far, model stealing for the medical domain has focused on classification; however, existing attacks are not effective against MLLMs. In this paper, we introduce Adversarial Domain Alignment (ADA-STEAL), the first stealing attack against medical MLLMs. ADA-STEAL relies on natural images, which are public and widely available, as opposed to their medical counterparts. We show that data augmentation with adversarial noise is sufficient to overcome the data distribution gap between natural images and the domain-specific distribution of the victim MLLM. Experiments on the IU X-RAY and MIMIC-CXR radiology datasets demonstrate that Adversarial Domain Alignment enables attackers to steal the medical MLLM without any access to medical data.



## **41. Rule-ATT&CK Mapper (RAM): Mapping SIEM Rules to TTPs Using LLMs**

cs.CR

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02337v1) [paper-pdf](http://arxiv.org/pdf/2502.02337v1)

**Authors**: Prasanna N. Wudali, Moshe Kravchik, Ehud Malul, Parth A. Gandhi, Yuval Elovici, Asaf Shabtai

**Abstract**: The growing frequency of cyberattacks has heightened the demand for accurate and efficient threat detection systems. SIEM platforms are important for analyzing log data and detecting adversarial activities through rule-based queries, also known as SIEM rules. The efficiency of the threat analysis process relies heavily on mapping these SIEM rules to the relevant attack techniques in the MITRE ATT&CK framework. Inaccurate annotation of SIEM rules can result in the misinterpretation of attacks, increasing the likelihood that threats will be overlooked. Existing solutions for annotating SIEM rules with MITRE ATT&CK technique labels have notable limitations: manual annotation of SIEM rules is both time-consuming and prone to errors, and ML-based approaches mainly focus on annotating unstructured free text sources rather than structured data like SIEM rules. Structured data often contains limited information, further complicating the annotation process and making it a challenging task. To address these challenges, we propose Rule-ATT&CK Mapper (RAM), a novel framework that leverages LLMs to automate the mapping of structured SIEM rules to MITRE ATT&CK techniques. RAM's multi-stage pipeline, which was inspired by the prompt chaining technique, enhances mapping accuracy without requiring LLM pre-training or fine-tuning. Using the Splunk Security Content dataset, we evaluate RAM's performance using several LLMs, including GPT-4-Turbo, Qwen, IBM Granite, and Mistral. Our evaluation highlights GPT-4-Turbo's superior performance, which derives from its enriched knowledge base, and an ablation study emphasizes the importance of external contextual knowledge in overcoming the limitations of LLMs' implicit knowledge for domain-specific tasks. These findings demonstrate RAM's potential in automating cybersecurity workflows and provide valuable insights for future advancements in this field.



## **42. FRAUD-RLA: A new reinforcement learning adversarial attack against credit card fraud detection**

cs.LG

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02290v1) [paper-pdf](http://arxiv.org/pdf/2502.02290v1)

**Authors**: Daniele Lunghi, Yannick Molinghen, Alkis Simitsis, Tom Lenaerts, Gianluca Bontempi

**Abstract**: Adversarial attacks pose a significant threat to data-driven systems, and researchers have spent considerable resources studying them. Despite its economic relevance, this trend largely overlooked the issue of credit card fraud detection. To address this gap, we propose a new threat model that demonstrates the limitations of existing attacks and highlights the necessity to investigate new approaches. We then design a new adversarial attack for credit card fraud detection, employing reinforcement learning to bypass classifiers. This attack, called FRAUD-RLA, is designed to maximize the attacker's reward by optimizing the exploration-exploitation tradeoff and working with significantly less required knowledge than competitors. Our experiments, conducted on three different heterogeneous datasets and against two fraud detection systems, indicate that FRAUD-RLA is effective, even considering the severe limitations imposed by our threat model.



## **43. Model Supply Chain Poisoning: Backdooring Pre-trained Models via Embedding Indistinguishability**

cs.CR

ACM Web Conference 2025 (Oral)

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2401.15883v3) [paper-pdf](http://arxiv.org/pdf/2401.15883v3)

**Authors**: Hao Wang, Shangwei Guo, Jialing He, Hangcheng Liu, Tianwei Zhang, Tao Xiang

**Abstract**: Pre-trained models (PTMs) are widely adopted across various downstream tasks in the machine learning supply chain. Adopting untrustworthy PTMs introduces significant security risks, where adversaries can poison the model supply chain by embedding hidden malicious behaviors (backdoors) into PTMs. However, existing backdoor attacks to PTMs can only achieve partially task-agnostic and the embedded backdoors are easily erased during the fine-tuning process. This makes it challenging for the backdoors to persist and propagate through the supply chain. In this paper, we propose a novel and severer backdoor attack, TransTroj, which enables the backdoors embedded in PTMs to efficiently transfer in the model supply chain. In particular, we first formalize this attack as an indistinguishability problem between poisoned and clean samples in the embedding space. We decompose embedding indistinguishability into pre- and post-indistinguishability, representing the similarity of the poisoned and reference embeddings before and after the attack. Then, we propose a two-stage optimization that separately optimizes triggers and victim PTMs to achieve embedding indistinguishability. We evaluate TransTroj on four PTMs and six downstream tasks. Experimental results show that our method significantly outperforms SOTA task-agnostic backdoor attacks -- achieving nearly 100% attack success rate on most downstream tasks -- and demonstrates robustness under various system settings. Our findings underscore the urgent need to secure the model supply chain against such transferable backdoor attacks. The code is available at https://github.com/haowang-cqu/TransTroj .



## **44. Multi-Domain Graph Foundation Models: Robust Knowledge Transfer via Topology Alignment**

cs.SI

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02017v1) [paper-pdf](http://arxiv.org/pdf/2502.02017v1)

**Authors**: Shuo Wang, Bokui Wang, Zhixiang Shen, Boyan Deng, Zhao Kang

**Abstract**: Recent advances in CV and NLP have inspired researchers to develop general-purpose graph foundation models through pre-training across diverse domains. However, a fundamental challenge arises from the substantial differences in graph topologies across domains. Additionally, real-world graphs are often sparse and prone to noisy connections and adversarial attacks. To address these issues, we propose the Multi-Domain Graph Foundation Model (MDGFM), a unified framework that aligns and leverages cross-domain topological information to facilitate robust knowledge transfer. MDGFM bridges different domains by adaptively balancing features and topology while refining original graphs to eliminate noise and align topological structures. To further enhance knowledge transfer, we introduce an efficient prompt-tuning approach. By aligning topologies, MDGFM not only improves multi-domain pre-training but also enables robust knowledge transfer to unseen domains. Theoretical analyses provide guarantees of MDGFM's effectiveness and domain generalization capabilities. Extensive experiments on both homophilic and heterophilic graph datasets validate the robustness and efficacy of our method.



## **45. Evaluating the Robustness of the "Ensemble Everything Everywhere" Defense**

cs.LG

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2411.14834v2) [paper-pdf](http://arxiv.org/pdf/2411.14834v2)

**Authors**: Jie Zhang, Christian Schlarmann, Kristina Nikolić, Nicholas Carlini, Francesco Croce, Matthias Hein, Florian Tramèr

**Abstract**: Ensemble everything everywhere is a defense to adversarial examples that was recently proposed to make image classifiers robust. This defense works by ensembling a model's intermediate representations at multiple noisy image resolutions, producing a single robust classification. This defense was shown to be effective against multiple state-of-the-art attacks. Perhaps even more convincingly, it was shown that the model's gradients are perceptually aligned: attacks against the model produce noise that perceptually resembles the targeted class.   In this short note, we show that this defense is not robust to adversarial attack. We first show that the defense's randomness and ensembling method cause severe gradient masking. We then use standard adaptive attack techniques to reduce the defense's robust accuracy from 48% to 14% on CIFAR-100 and from 62% to 11% on CIFAR-10, under the $\ell_\infty$-norm threat model with $\varepsilon=8/255$.



## **46. Adversarial Reasoning at Jailbreaking Time**

cs.LG

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01633v1) [paper-pdf](http://arxiv.org/pdf/2502.01633v1)

**Authors**: Mahdi Sabbaghi, Paul Kassianik, George Pappas, Yaron Singer, Amin Karbasi, Hamed Hassani

**Abstract**: As large language models (LLMs) are becoming more capable and widespread, the study of their failure cases is becoming increasingly important. Recent advances in standardizing, measuring, and scaling test-time compute suggest new methodologies for optimizing models to achieve high performance on hard tasks. In this paper, we apply these advances to the task of model jailbreaking: eliciting harmful responses from aligned LLMs. We develop an adversarial reasoning approach to automatic jailbreaking via test-time computation that achieves SOTA attack success rates (ASR) against many aligned LLMs, even the ones that aim to trade inference-time compute for adversarial robustness. Our approach introduces a new paradigm in understanding LLM vulnerabilities, laying the foundation for the development of more robust and trustworthy AI systems.



## **47. Robust-LLaVA: On the Effectiveness of Large-Scale Robust Image Encoders for Multi-modal Large Language Models**

cs.CV

Under Review

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01576v1) [paper-pdf](http://arxiv.org/pdf/2502.01576v1)

**Authors**: Hashmat Shadab Malik, Fahad Shamshad, Muzammal Naseer, Karthik Nandakumar, Fahad Khan, Salman Khan

**Abstract**: Multi-modal Large Language Models (MLLMs) excel in vision-language tasks but remain vulnerable to visual adversarial perturbations that can induce hallucinations, manipulate responses, or bypass safety mechanisms. Existing methods seek to mitigate these risks by applying constrained adversarial fine-tuning to CLIP vision encoders on ImageNet-scale data, ensuring their generalization ability is preserved. However, this limited adversarial training restricts robustness and broader generalization. In this work, we explore an alternative approach of leveraging existing vision classification models that have been adversarially pre-trained on large-scale data. Our analysis reveals two principal contributions: (1) the extensive scale and diversity of adversarial pre-training enables these models to demonstrate superior robustness against diverse adversarial threats, ranging from imperceptible perturbations to advanced jailbreaking attempts, without requiring additional adversarial training, and (2) end-to-end MLLM integration with these robust models facilitates enhanced adaptation of language components to robust visual features, outperforming existing plug-and-play methodologies on complex reasoning tasks. Through systematic evaluation across visual question-answering, image captioning, and jail-break attacks, we demonstrate that MLLMs trained with these robust models achieve superior adversarial robustness while maintaining favorable clean performance. Our framework achieves 2x and 1.5x average robustness gains in captioning and VQA tasks, respectively, and delivers over 10% improvement against jailbreak attacks. Code and pretrained models will be available at https://github.com/HashmatShadab/Robust-LLaVA.



## **48. Quantum Quandaries: Unraveling Encoding Vulnerabilities in Quantum Neural Networks**

quant-ph

7 Pages

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01486v1) [paper-pdf](http://arxiv.org/pdf/2502.01486v1)

**Authors**: Suryansh Upadhyay, Swaroop Ghosh

**Abstract**: Quantum computing (QC) has the potential to revolutionize fields like machine learning, security, and healthcare. Quantum machine learning (QML) has emerged as a promising area, enhancing learning algorithms using quantum computers. However, QML models are lucrative targets due to their high training costs and extensive training times. The scarcity of quantum resources and long wait times further exacerbate the challenge. Additionally, QML providers may rely on third party quantum clouds for hosting models, exposing them and their training data to potential threats. As QML as a Service (QMLaaS) becomes more prevalent, reliance on third party quantum clouds poses a significant security risk. This work demonstrates that adversaries in quantum cloud environments can exploit white box access to QML models to infer the users encoding scheme by analyzing circuit transpilation artifacts. The extracted data can be reused for training clone models or sold for profit. We validate the proposed attack through simulations, achieving high accuracy in distinguishing between encoding schemes. We report that 95% of the time, the encoding can be predicted correctly. To mitigate this threat, we propose a transient obfuscation layer that masks encoding fingerprints using randomized rotations and entanglement, reducing adversarial detection to near random chance 42% , with a depth overhead of 8.5% for a 5 layer QNN design.



## **49. DeTrigger: A Gradient-Centric Approach to Backdoor Attack Mitigation in Federated Learning**

cs.LG

21 pages

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2411.12220v2) [paper-pdf](http://arxiv.org/pdf/2411.12220v2)

**Authors**: Kichang Lee, Yujin Shin, Jonghyuk Yun, Songkuk Kim, Jun Han, JeongGil Ko

**Abstract**: Federated Learning (FL) enables collaborative model training across distributed devices while preserving local data privacy, making it ideal for mobile and embedded systems. However, the decentralized nature of FL also opens vulnerabilities to model poisoning attacks, particularly backdoor attacks, where adversaries implant trigger patterns to manipulate model predictions. In this paper, we propose DeTrigger, a scalable and efficient backdoor-robust federated learning framework that leverages insights from adversarial attack methodologies. By employing gradient analysis with temperature scaling, DeTrigger detects and isolates backdoor triggers, allowing for precise model weight pruning of backdoor activations without sacrificing benign model knowledge. Extensive evaluations across four widely used datasets demonstrate that DeTrigger achieves up to 251x faster detection than traditional methods and mitigates backdoor attacks by up to 98.9%, with minimal impact on global model accuracy. Our findings establish DeTrigger as a robust and scalable solution to protect federated learning environments against sophisticated backdoor threats.



## **50. Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models**

cs.CL

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01386v1) [paper-pdf](http://arxiv.org/pdf/2502.01386v1)

**Authors**: Yuyang Gong, Zhuo Chen, Miaokun Chen, Fengchang Yu, Wei Lu, Xiaofeng Wang, Xiaozhong Liu, Jiawei Liu

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become essential for tasks such as question answering and content generation. However, their increasing impact on public opinion and information dissemination has made them a critical focus for security research due to inherent vulnerabilities. Previous studies have predominantly addressed attacks targeting factual or single-query manipulations. In this paper, we address a more practical scenario: topic-oriented adversarial opinion manipulation attacks on RAG models, where LLMs are required to reason and synthesize multiple perspectives, rendering them particularly susceptible to systematic knowledge poisoning. Specifically, we propose Topic-FlipRAG, a two-stage manipulation attack pipeline that strategically crafts adversarial perturbations to influence opinions across related queries. This approach combines traditional adversarial ranking attack techniques and leverages the extensive internal relevant knowledge and reasoning capabilities of LLMs to execute semantic-level perturbations. Experiments show that the proposed attacks effectively shift the opinion of the model's outputs on specific topics, significantly impacting user information perception. Current mitigation methods cannot effectively defend against such attacks, highlighting the necessity for enhanced safeguards for RAG systems, and offering crucial insights for LLM security research.



