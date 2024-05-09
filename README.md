# Latest Adversarial Attack Papers
**update at 2024-05-09 23:52:50**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Air Gap: Protecting Privacy-Conscious Conversational Agents**

cs.CR

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05175v1) [paper-pdf](http://arxiv.org/pdf/2405.05175v1)

**Authors**: Eugene Bagdasaryan, Ren Yi, Sahra Ghalebikesabi, Peter Kairouz, Marco Gruteser, Sewoong Oh, Borja Balle, Daniel Ramage

**Abstract**: The growing use of large language model (LLM)-based conversational agents to manage sensitive user data raises significant privacy concerns. While these agents excel at understanding and acting on context, this capability can be exploited by malicious actors. We introduce a novel threat model where adversarial third-party apps manipulate the context of interaction to trick LLM-based agents into revealing private information not relevant to the task at hand.   Grounded in the framework of contextual integrity, we introduce AirGapAgent, a privacy-conscious agent designed to prevent unintended data leakage by restricting the agent's access to only the data necessary for a specific task. Extensive experiments using Gemini, GPT, and Mistral models as agents validate our approach's effectiveness in mitigating this form of context hijacking while maintaining core agent functionality. For example, we show that a single-query context hijacking attack on a Gemini Ultra agent reduces its ability to protect user data from 94% to 45%, while an AirGapAgent achieves 97% protection, rendering the same attack ineffective.



## **2. Filtering and smoothing estimation algorithms from uncertain nonlinear observations with time-correlated additive noise and random deception attacks**

eess.SP

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05157v1) [paper-pdf](http://arxiv.org/pdf/2405.05157v1)

**Authors**: R. Caballero-Águila, J. Hu, J. Linares-Pérez

**Abstract**: This paper discusses the problem of estimating a stochastic signal from nonlinear uncertain observations with time-correlated additive noise described by a first-order Markov process. Random deception attacks are assumed to be launched by an adversary, and both this phenomenon and the uncertainty in the observations are modelled by two sets of Bernoulli random variables. Under the assumption that the evolution model generating the signal to be estimated is unknown and only the mean and covariance functions of the processes involved in the observation equation are available, recursive algorithms based on linear approximations of the real observations are proposed for the least-squares filtering and fixed-point smoothing problems. Finally, the feasibility and effectiveness of the developed estimation algorithms are verified by a numerical simulation example, where the impact of uncertain observation and deception attack probabilities on estimation accuracy is evaluated.



## **3. Towards Efficient Training and Evaluation of Robust Models against $l_0$ Bounded Adversarial Perturbations**

cs.LG

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05075v1) [paper-pdf](http://arxiv.org/pdf/2405.05075v1)

**Authors**: Xuyang Zhong, Yixiao Huang, Chen Liu

**Abstract**: This work studies sparse adversarial perturbations bounded by $l_0$ norm. We propose a white-box PGD-like attack method named sparse-PGD to effectively and efficiently generate such perturbations. Furthermore, we combine sparse-PGD with a black-box attack to comprehensively and more reliably evaluate the models' robustness against $l_0$ bounded adversarial perturbations. Moreover, the efficiency of sparse-PGD enables us to conduct adversarial training to build robust models against sparse perturbations. Extensive experiments demonstrate that our proposed attack algorithm exhibits strong performance in different scenarios. More importantly, compared with other robust models, our adversarially trained model demonstrates state-of-the-art robustness against various sparse attacks. Codes are available at https://github.com/CityU-MLO/sPGD.



## **4. Adversarial Threats to Automatic Modulation Open Set Recognition in Wireless Networks**

cs.CR

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05022v1) [paper-pdf](http://arxiv.org/pdf/2405.05022v1)

**Authors**: Yandie Yang, Sicheng Zhang, Kuixian Li, Qiao Tian, Yun Lin

**Abstract**: Automatic Modulation Open Set Recognition (AMOSR) is a crucial technological approach for cognitive radio communications, wireless spectrum management, and interference monitoring within wireless networks. Numerous studies have shown that AMR is highly susceptible to minimal perturbations carefully designed by malicious attackers, leading to misclassification of signals. However, the adversarial security issue of AMOSR has not yet been explored. This paper adopts the perspective of attackers and proposes an Open Set Adversarial Attack (OSAttack), aiming at investigating the adversarial vulnerabilities of various AMOSR methods. Initially, an adversarial threat model for AMOSR scenarios is established. Subsequently, by analyzing the decision criteria of both discriminative and generative open set recognition, OSFGSM and OSPGD are proposed to reduce the performance of AMOSR. Finally, the influence of OSAttack on AMOSR is evaluated utilizing a range of qualitative and quantitative indicators. The results indicate that despite the increased resistance of AMOSR models to conventional interference signals, they remain vulnerable to attacks by adversarial examples.



## **5. Deep Reinforcement Learning with Spiking Q-learning**

cs.NE

15 pages, 7 figures

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2201.09754v3) [paper-pdf](http://arxiv.org/pdf/2201.09754v3)

**Authors**: Ding Chen, Peixi Peng, Tiejun Huang, Yonghong Tian

**Abstract**: With the help of special neuromorphic hardware, spiking neural networks (SNNs) are expected to realize artificial intelligence (AI) with less energy consumption. It provides a promising energy-efficient way for realistic control tasks by combining SNNs with deep reinforcement learning (RL). There are only a few existing SNN-based RL methods at present. Most of them either lack generalization ability or employ Artificial Neural Networks (ANNs) to estimate value function in training. The former needs to tune numerous hyper-parameters for each scenario, and the latter limits the application of different types of RL algorithm and ignores the large energy consumption in training. To develop a robust spike-based RL method, we draw inspiration from non-spiking interneurons found in insects and propose the deep spiking Q-network (DSQN), using the membrane voltage of non-spiking neurons as the representation of Q-value, which can directly learn robust policies from high-dimensional sensory inputs using end-to-end RL. Experiments conducted on 17 Atari games demonstrate the DSQN is effective and even outperforms the ANN-based deep Q-network (DQN) in most games. Moreover, the experiments show superior learning stability and robustness to adversarial attacks of DSQN.



## **6. Learning-Based Difficulty Calibration for Enhanced Membership Inference Attacks**

cs.CR

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2401.04929v2) [paper-pdf](http://arxiv.org/pdf/2401.04929v2)

**Authors**: Haonan Shi, Tu Ouyang, An Wang

**Abstract**: Machine learning models, in particular deep neural networks, are currently an integral part of various applications, from healthcare to finance. However, using sensitive data to train these models raises concerns about privacy and security. One method that has emerged to verify if the trained models are privacy-preserving is Membership Inference Attacks (MIA), which allows adversaries to determine whether a specific data point was part of a model's training dataset. While a series of MIAs have been proposed in the literature, only a few can achieve high True Positive Rates (TPR) in the low False Positive Rate (FPR) region (0.01%~1%). This is a crucial factor to consider for an MIA to be practically useful in real-world settings. In this paper, we present a novel approach to MIA that is aimed at significantly improving TPR at low FPRs. Our method, named learning-based difficulty calibration for MIA(LDC-MIA), characterizes data records by their hardness levels using a neural network classifier to determine membership. The experiment results show that LDC-MIA can improve TPR at low FPR by up to 4x compared to the other difficulty calibration based MIAs. It also has the highest Area Under ROC curve (AUC) across all datasets. Our method's cost is comparable with most of the existing MIAs, but is orders of magnitude more efficient than one of the state-of-the-art methods, LiRA, while achieving similar performance.



## **7. BiasKG: Adversarial Knowledge Graphs to Induce Bias in Large Language Models**

cs.CL

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.04756v1) [paper-pdf](http://arxiv.org/pdf/2405.04756v1)

**Authors**: Chu Fei Luo, Ahmad Ghawanmeh, Xiaodan Zhu, Faiza Khan Khattak

**Abstract**: Modern large language models (LLMs) have a significant amount of world knowledge, which enables strong performance in commonsense reasoning and knowledge-intensive tasks when harnessed properly. The language model can also learn social biases, which has a significant potential for societal harm. There have been many mitigation strategies proposed for LLM safety, but it is unclear how effective they are for eliminating social biases. In this work, we propose a new methodology for attacking language models with knowledge graph augmented generation. We refactor natural language stereotypes into a knowledge graph, and use adversarial attacking strategies to induce biased responses from several open- and closed-source language models. We find our method increases bias in all models, even those trained with safety guardrails. This demonstrates the need for further research in AI safety, and further work in this new adversarial space.



## **8. Demonstration of an Adversarial Attack Against a Multimodal Vision Language Model for Pathology Imaging**

eess.IV

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2401.02565v3) [paper-pdf](http://arxiv.org/pdf/2401.02565v3)

**Authors**: Poojitha Thota, Jai Prakash Veerla, Partha Sai Guttikonda, Mohammad S. Nasr, Shirin Nilizadeh, Jacob M. Luber

**Abstract**: In the context of medical artificial intelligence, this study explores the vulnerabilities of the Pathology Language-Image Pretraining (PLIP) model, a Vision Language Foundation model, under targeted attacks. Leveraging the Kather Colon dataset with 7,180 H&E images across nine tissue types, our investigation employs Projected Gradient Descent (PGD) adversarial perturbation attacks to induce misclassifications intentionally. The outcomes reveal a 100% success rate in manipulating PLIP's predictions, underscoring its susceptibility to adversarial perturbations. The qualitative analysis of adversarial examples delves into the interpretability challenges, shedding light on nuanced changes in predictions induced by adversarial manipulations. These findings contribute crucial insights into the interpretability, domain adaptation, and trustworthiness of Vision Language Models in medical imaging. The study emphasizes the pressing need for robust defenses to ensure the reliability of AI models. The source codes for this experiment can be found at https://github.com/jaiprakash1824/VLM_Adv_Attack.



## **9. Fully Automated Selfish Mining Analysis in Efficient Proof Systems Blockchains**

cs.CR

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04420v1) [paper-pdf](http://arxiv.org/pdf/2405.04420v1)

**Authors**: Krishnendu Chatterjee, Amirali Ebrahimzadeh, Mehrdad Karrabi, Krzysztof Pietrzak, Michelle Yeo, Đorđe Žikelić

**Abstract**: We study selfish mining attacks in longest-chain blockchains like Bitcoin, but where the proof of work is replaced with efficient proof systems -- like proofs of stake or proofs of space -- and consider the problem of computing an optimal selfish mining attack which maximizes expected relative revenue of the adversary, thus minimizing the chain quality. To this end, we propose a novel selfish mining attack that aims to maximize this objective and formally model the attack as a Markov decision process (MDP). We then present a formal analysis procedure which computes an $\epsilon$-tight lower bound on the optimal expected relative revenue in the MDP and a strategy that achieves this $\epsilon$-tight lower bound, where $\epsilon>0$ may be any specified precision. Our analysis is fully automated and provides formal guarantees on the correctness. We evaluate our selfish mining attack and observe that it achieves superior expected relative revenue compared to two considered baselines.   In concurrent work [Sarenche FC'24] does an automated analysis on selfish mining in predictable longest-chain blockchains based on efficient proof systems. Predictable means the randomness for the challenges is fixed for many blocks (as used e.g., in Ouroboros), while we consider unpredictable (Bitcoin-like) chains where the challenge is derived from the previous block.



## **10. NeuroIDBench: An Open-Source Benchmark Framework for the Standardization of Methodology in Brainwave-based Authentication Research**

cs.CR

21 pages, 5 Figures, 3 tables, Submitted to the Journal of  Information Security and Applications

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2402.08656v4) [paper-pdf](http://arxiv.org/pdf/2402.08656v4)

**Authors**: Avinash Kumar Chaurasia, Matin Fallahi, Thorsten Strufe, Philipp Terhörst, Patricia Arias Cabarcos

**Abstract**: Biometric systems based on brain activity have been proposed as an alternative to passwords or to complement current authentication techniques. By leveraging the unique brainwave patterns of individuals, these systems offer the possibility of creating authentication solutions that are resistant to theft, hands-free, accessible, and potentially even revocable. However, despite the growing stream of research in this area, faster advance is hindered by reproducibility problems. Issues such as the lack of standard reporting schemes for performance results and system configuration, or the absence of common evaluation benchmarks, make comparability and proper assessment of different biometric solutions challenging. Further, barriers are erected to future work when, as so often, source code is not published open access. To bridge this gap, we introduce NeuroIDBench, a flexible open source tool to benchmark brainwave-based authentication models. It incorporates nine diverse datasets, implements a comprehensive set of pre-processing parameters and machine learning algorithms, enables testing under two common adversary models (known vs unknown attacker), and allows researchers to generate full performance reports and visualizations. We use NeuroIDBench to investigate the shallow classifiers and deep learning-based approaches proposed in the literature, and to test robustness across multiple sessions. We observe a 37.6% reduction in Equal Error Rate (EER) for unknown attacker scenarios (typically not tested in the literature), and we highlight the importance of session variability to brainwave authentication. All in all, our results demonstrate the viability and relevance of NeuroIDBench in streamlining fair comparisons of algorithms, thereby furthering the advancement of brainwave-based authentication through robust methodological practices.



## **11. Revisiting character-level adversarial attacks**

cs.LG

Accepted in ICML 2024

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04346v1) [paper-pdf](http://arxiv.org/pdf/2405.04346v1)

**Authors**: Elias Abad Rocamora, Yongtao Wu, Fanghui Liu, Grigorios G. Chrysos, Volkan Cevher

**Abstract**: Adversarial attacks in Natural Language Processing apply perturbations in the character or token levels. Token-level attacks, gaining prominence for their use of gradient-based methods, are susceptible to altering sentence semantics, leading to invalid adversarial examples. While character-level attacks easily maintain semantics, they have received less attention as they cannot easily adopt popular gradient-based methods, and are thought to be easy to defend. Challenging these beliefs, we introduce Charmer, an efficient query-based adversarial attack capable of achieving high attack success rate (ASR) while generating highly similar adversarial examples. Our method successfully targets both small (BERT) and large (Llama 2) models. Specifically, on BERT with SST-2, Charmer improves the ASR in 4.84% points and the USE similarity in 8% points with respect to the previous art. Our implementation is available in https://github.com/LIONS-EPFL/Charmer.



## **12. Who Wrote This? The Key to Zero-Shot LLM-Generated Text Detection Is GECScore**

cs.CL

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04286v1) [paper-pdf](http://arxiv.org/pdf/2405.04286v1)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xuebo Liu, Lidia S. Chao, Min Zhang

**Abstract**: The efficacy of an large language model (LLM) generated text detector depends substantially on the availability of sizable training data. White-box zero-shot detectors, which require no such data, are nonetheless limited by the accessibility of the source model of the LLM-generated text. In this paper, we propose an simple but effective black-box zero-shot detection approach, predicated on the observation that human-written texts typically contain more grammatical errors than LLM-generated texts. This approach entails computing the Grammar Error Correction Score (GECScore) for the given text to distinguish between human-written and LLM-generated text. Extensive experimental results show that our method outperforms current state-of-the-art (SOTA) zero-shot and supervised methods, achieving an average AUROC of 98.7% and showing strong robustness against paraphrase and adversarial perturbation attacks.



## **13. A Stealthy Wrongdoer: Feature-Oriented Reconstruction Attack against Split Learning**

cs.CR

Accepted to CVPR 2024

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04115v1) [paper-pdf](http://arxiv.org/pdf/2405.04115v1)

**Authors**: Xiaoyang Xu, Mengda Yang, Wenzhe Yi, Ziang Li, Juan Wang, Hongxin Hu, Yong Zhuang, Yaxin Liu

**Abstract**: Split Learning (SL) is a distributed learning framework renowned for its privacy-preserving features and minimal computational requirements. Previous research consistently highlights the potential privacy breaches in SL systems by server adversaries reconstructing training data. However, these studies often rely on strong assumptions or compromise system utility to enhance attack performance. This paper introduces a new semi-honest Data Reconstruction Attack on SL, named Feature-Oriented Reconstruction Attack (FORA). In contrast to prior works, FORA relies on limited prior knowledge, specifically that the server utilizes auxiliary samples from the public without knowing any client's private information. This allows FORA to conduct the attack stealthily and achieve robust performance. The key vulnerability exploited by FORA is the revelation of the model representation preference in the smashed data output by victim client. FORA constructs a substitute client through feature-level transfer learning, aiming to closely mimic the victim client's representation preference. Leveraging this substitute client, the server trains the attack model to effectively reconstruct private data. Extensive experiments showcase FORA's superior performance compared to state-of-the-art methods. Furthermore, the paper systematically evaluates the proposed method's applicability across diverse settings and advanced defense strategies.



## **14. Explainability-Informed Targeted Malware Misclassification**

cs.CR

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04010v1) [paper-pdf](http://arxiv.org/pdf/2405.04010v1)

**Authors**: Quincy Card, Kshitiz Aryal, Maanak Gupta

**Abstract**: In recent years, there has been a surge in malware attacks across critical infrastructures, requiring further research and development of appropriate response and remediation strategies in malware detection and classification. Several works have used machine learning models for malware classification into categories, and deep neural networks have shown promising results. However, these models have shown its vulnerabilities against intentionally crafted adversarial attacks, which yields misclassification of a malicious file. Our paper explores such adversarial vulnerabilities of neural network based malware classification system in the dynamic and online analysis environments. To evaluate our approach, we trained Feed Forward Neural Networks (FFNN) to classify malware categories based on features obtained from dynamic and online analysis environments. We use the state-of-the-art method, SHapley Additive exPlanations (SHAP), for the feature attribution for malware classification, to inform the adversarial attackers about the features with significant importance on classification decision. Using the explainability-informed features, we perform targeted misclassification adversarial white-box evasion attacks using the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks against the trained classifier. Our results demonstrated high evasion rate for some instances of attacks, showing a clear vulnerability of a malware classifier for such attacks. We offer recommendations for a balanced approach and a benchmark for much-needed future research into evasion attacks against malware classifiers, and develop more robust and trustworthy solutions.



## **15. Navigating Quantum Security Risks in Networked Environments: A Comprehensive Study of Quantum-Safe Network Protocols**

cs.CR

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2404.08232v2) [paper-pdf](http://arxiv.org/pdf/2404.08232v2)

**Authors**: Yaser Baseri, Vikas Chouhan, Abdelhakim Hafid

**Abstract**: The emergence of quantum computing poses a formidable security challenge to network protocols traditionally safeguarded by classical cryptographic algorithms. This paper provides an exhaustive analysis of vulnerabilities introduced by quantum computing in a diverse array of widely utilized security protocols across the layers of the TCP/IP model, including TLS, IPsec, SSH, PGP, and more. Our investigation focuses on precisely identifying vulnerabilities susceptible to exploitation by quantum adversaries at various migration stages for each protocol while also assessing the associated risks and consequences for secure communication. We delve deep into the impact of quantum computing on each protocol, emphasizing potential threats posed by quantum attacks and scrutinizing the effectiveness of post-quantum cryptographic solutions. Through carefully evaluating vulnerabilities and risks that network protocols face in the post-quantum era, this study provides invaluable insights to guide the development of appropriate countermeasures. Our findings contribute to a broader comprehension of quantum computing's influence on network security and offer practical guidance for protocol designers, implementers, and policymakers in addressing the challenges stemming from the advancement of quantum computing. This comprehensive study is a crucial step toward fortifying the security of networked environments in the quantum age.



## **16. Enhancing O-RAN Security: Evasion Attacks and Robust Defenses for Graph Reinforcement Learning-based Connection Management**

cs.CR

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2405.03891v1) [paper-pdf](http://arxiv.org/pdf/2405.03891v1)

**Authors**: Ravikumar Balakrishnan, Marius Arvinte, Nageen Himayat, Hosein Nikopour, Hassnaa Moustafa

**Abstract**: Adversarial machine learning, focused on studying various attacks and defenses on machine learning (ML) models, is rapidly gaining importance as ML is increasingly being adopted for optimizing wireless systems such as Open Radio Access Networks (O-RAN). A comprehensive modeling of the security threats and the demonstration of adversarial attacks and defenses on practical AI based O-RAN systems is still in its nascent stages. We begin by conducting threat modeling to pinpoint attack surfaces in O-RAN using an ML-based Connection management application (xApp) as an example. The xApp uses a Graph Neural Network trained using Deep Reinforcement Learning and achieves on average 54% improvement in the coverage rate measured as the 5th percentile user data rates. We then formulate and demonstrate evasion attacks that degrade the coverage rates by as much as 50% through injecting bounded noise at different threat surfaces including the open wireless medium itself. Crucially, we also compare and contrast the effectiveness of such attacks on the ML-based xApp and a non-ML based heuristic. We finally develop and demonstrate robust training-based defenses against the challenging physical/jamming-based attacks and show a 15% improvement in the coverage rates when compared to employing no defense over a range of noise budgets



## **17. On Adversarial Examples for Text Classification by Perturbing Latent Representations**

cs.LG

7 pages

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2405.03789v1) [paper-pdf](http://arxiv.org/pdf/2405.03789v1)

**Authors**: Korn Sooksatra, Bikram Khanal, Pablo Rivas

**Abstract**: Recently, with the advancement of deep learning, several applications in text classification have advanced significantly. However, this improvement comes with a cost because deep learning is vulnerable to adversarial examples. This weakness indicates that deep learning is not very robust. Fortunately, the input of a text classifier is discrete. Hence, it can prevent the classifier from state-of-the-art attacks. Nonetheless, previous works have generated black-box attacks that successfully manipulate the discrete values of the input to find adversarial examples. Therefore, instead of changing the discrete values, we transform the input into its embedding vector containing real values to perform the state-of-the-art white-box attacks. Then, we convert the perturbed embedding vector back into a text and name it an adversarial example. In summary, we create a framework that measures the robustness of a text classifier by using the gradients of the classifier.



## **18. RandOhm: Mitigating Impedance Side-channel Attacks using Randomized Circuit Configurations**

cs.CR

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2401.08925v2) [paper-pdf](http://arxiv.org/pdf/2401.08925v2)

**Authors**: Saleh Khalaj Monfared, Domenic Forte, Shahin Tajik

**Abstract**: Physical side-channel attacks can compromise the security of integrated circuits. Most physical side-channel attacks (e.g., power or electromagnetic) exploit the dynamic behavior of a chip, typically manifesting as changes in current consumption or voltage fluctuations where algorithmic countermeasures, such as masking, can effectively mitigate them. However, as demonstrated recently, these mitigation techniques are not entirely effective against backscattered side-channel attacks such as impedance analysis. In the case of an impedance attack, an adversary exploits the data-dependent impedance variations of the chip power delivery network (PDN) to extract secret information. In this work, we introduce RandOhm, which exploits a moving target defense (MTD) strategy based on the partial reconfiguration (PR) feature of mainstream FPGAs and programmable SoCs to defend against impedance side-channel attacks. We demonstrate that the information leakage through the PDN impedance could be significantly reduced via runtime reconfiguration of the secret-sensitive parts of the circuitry. Hence, by constantly randomizing the placement and routing of the circuit, one can decorrelate the data-dependent computation from the impedance value. Moreover, in contrast to existing PR-based countermeasures, RandOhm deploys open-source bitstream manipulation tools on programmable SoCs to speed up the randomization and provide real-time protection. To validate our claims, we apply RandOhm to AES ciphers realized on 28-nm FPGAs. We analyze the resiliency of our approach by performing non-profiled and profiled impedance analysis attacks and investigate the overhead of our mitigation in terms of delay and performance.



## **19. Understanding the Vulnerability of Skeleton-based Human Activity Recognition via Black-box Attack**

cs.CV

Accepted in Pattern Recognition. arXiv admin note: substantial text  overlap with arXiv:2103.05266

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2211.11312v2) [paper-pdf](http://arxiv.org/pdf/2211.11312v2)

**Authors**: Yunfeng Diao, He Wang, Tianjia Shao, Yong-Liang Yang, Kun Zhou, David Hogg, Meng Wang

**Abstract**: Human Activity Recognition (HAR) has been employed in a wide range of applications, e.g. self-driving cars, where safety and lives are at stake. Recently, the robustness of skeleton-based HAR methods have been questioned due to their vulnerability to adversarial attacks. However, the proposed attacks require the full-knowledge of the attacked classifier, which is overly restrictive. In this paper, we show such threats indeed exist, even when the attacker only has access to the input/output of the model. To this end, we propose the very first black-box adversarial attack approach in skeleton-based HAR called BASAR. BASAR explores the interplay between the classification boundary and the natural motion manifold. To our best knowledge, this is the first time data manifold is introduced in adversarial attacks on time series. Via BASAR, we find on-manifold adversarial samples are extremely deceitful and rather common in skeletal motions, in contrast to the common belief that adversarial samples only exist off-manifold. Through exhaustive evaluation, we show that BASAR can deliver successful attacks across classifiers, datasets, and attack modes. By attack, BASAR helps identify the potential causes of the model vulnerability and provides insights on possible improvements. Finally, to mitigate the newly identified threat, we propose a new adversarial training approach by leveraging the sophisticated distributions of on/off-manifold adversarial samples, called mixed manifold-based adversarial training (MMAT). MMAT can successfully help defend against adversarial attacks without compromising classification accuracy.



## **20. Provably Unlearnable Examples**

cs.LG

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2405.03316v1) [paper-pdf](http://arxiv.org/pdf/2405.03316v1)

**Authors**: Derui Wang, Minhui Xue, Bo Li, Seyit Camtepe, Liming Zhu

**Abstract**: The exploitation of publicly accessible data has led to escalating concerns regarding data privacy and intellectual property (IP) breaches in the age of artificial intelligence. As a strategy to safeguard both data privacy and IP-related domain knowledge, efforts have been undertaken to render shared data unlearnable for unauthorized models in the wild. Existing methods apply empirically optimized perturbations to the data in the hope of disrupting the correlation between the inputs and the corresponding labels such that the data samples are converted into Unlearnable Examples (UEs). Nevertheless, the absence of mechanisms that can verify how robust the UEs are against unknown unauthorized models and train-time techniques engenders several problems. First, the empirically optimized perturbations may suffer from the problem of cross-model generalization, which echoes the fact that the unauthorized models are usually unknown to the defender. Second, UEs can be mitigated by train-time techniques such as data augmentation and adversarial training. Furthermore, we find that a simple recovery attack can restore the clean-task performance of the classifiers trained on UEs by slightly perturbing the learned weights. To mitigate the aforementioned problems, in this paper, we propose a mechanism for certifying the so-called $(q, \eta)$-Learnability of an unlearnable dataset via parametric smoothing. A lower certified $(q, \eta)$-Learnability indicates a more robust protection over the dataset. Finally, we try to 1) improve the tightness of certified $(q, \eta)$-Learnability and 2) design Provably Unlearnable Examples (PUEs) which have reduced $(q, \eta)$-Learnability. According to experimental results, PUEs demonstrate both decreased certified $(q, \eta)$-Learnability and enhanced empirical robustness compared to existing UEs.



## **21. Illusory Attacks: Information-Theoretic Detectability Matters in Adversarial Attacks**

cs.AI

ICLR 2024 Spotlight (top 5%)

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2207.10170v5) [paper-pdf](http://arxiv.org/pdf/2207.10170v5)

**Authors**: Tim Franzmeyer, Stephen McAleer, João F. Henriques, Jakob N. Foerster, Philip H. S. Torr, Adel Bibi, Christian Schroeder de Witt

**Abstract**: Autonomous agents deployed in the real world need to be robust against adversarial attacks on sensory inputs. Robustifying agent policies requires anticipating the strongest attacks possible. We demonstrate that existing observation-space attacks on reinforcement learning agents have a common weakness: while effective, their lack of information-theoretic detectability constraints makes them detectable using automated means or human inspection. Detectability is undesirable to adversaries as it may trigger security escalations. We introduce {\epsilon}-illusory, a novel form of adversarial attack on sequential decision-makers that is both effective and of {\epsilon}-bounded statistical detectability. We propose a novel dual ascent algorithm to learn such attacks end-to-end. Compared to existing attacks, we empirically find {\epsilon}-illusory to be significantly harder to detect with automated methods, and a small study with human participants (IRB approval under reference R84123/RE001) suggests they are similarly harder to detect for humans. Our findings suggest the need for better anomaly detectors, as well as effective hardware- and system-level defenses. The project website can be found at https://tinyurl.com/illusory-attacks.



## **22. Purify Unlearnable Examples via Rate-Constrained Variational Autoencoders**

cs.CR

Accepted by ICML 2024

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2405.01460v2) [paper-pdf](http://arxiv.org/pdf/2405.01460v2)

**Authors**: Yi Yu, Yufei Wang, Song Xia, Wenhan Yang, Shijian Lu, Yap-Peng Tan, Alex C. Kot

**Abstract**: Unlearnable examples (UEs) seek to maximize testing error by making subtle modifications to training examples that are correctly labeled. Defenses against these poisoning attacks can be categorized based on whether specific interventions are adopted during training. The first approach is training-time defense, such as adversarial training, which can mitigate poisoning effects but is computationally intensive. The other approach is pre-training purification, e.g., image short squeezing, which consists of several simple compressions but often encounters challenges in dealing with various UEs. Our work provides a novel disentanglement mechanism to build an efficient pre-training purification method. Firstly, we uncover rate-constrained variational autoencoders (VAEs), demonstrating a clear tendency to suppress the perturbations in UEs. We subsequently conduct a theoretical analysis for this phenomenon. Building upon these insights, we introduce a disentangle variational autoencoder (D-VAE), capable of disentangling the perturbations with learnable class-wise embeddings. Based on this network, a two-stage purification approach is naturally developed. The first stage focuses on roughly eliminating perturbations, while the second stage produces refined, poison-free results, ensuring effectiveness and robustness across various scenarios. Extensive experiments demonstrate the remarkable performance of our method across CIFAR-10, CIFAR-100, and a 100-class ImageNet-subset. Code is available at https://github.com/yuyi-sd/D-VAE.



## **23. Are aligned neural networks adversarially aligned?**

cs.CL

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2306.15447v2) [paper-pdf](http://arxiv.org/pdf/2306.15447v2)

**Authors**: Nicholas Carlini, Milad Nasr, Christopher A. Choquette-Choo, Matthew Jagielski, Irena Gao, Anas Awadalla, Pang Wei Koh, Daphne Ippolito, Katherine Lee, Florian Tramer, Ludwig Schmidt

**Abstract**: Large language models are now tuned to align with the goals of their creators, namely to be "helpful and harmless." These models should respond helpfully to user questions, but refuse to answer requests that could cause harm. However, adversarial users can construct inputs which circumvent attempts at alignment. In this work, we study adversarial alignment, and ask to what extent these models remain aligned when interacting with an adversarial user who constructs worst-case inputs (adversarial examples). These inputs are designed to cause the model to emit harmful content that would otherwise be prohibited. We show that existing NLP-based optimization attacks are insufficiently powerful to reliably attack aligned text models: even when current NLP-based attacks fail, we can find adversarial inputs with brute force. As a result, the failure of current attacks should not be seen as proof that aligned text models remain aligned under adversarial inputs.   However the recent trend in large-scale ML models is multimodal models that allow users to provide images that influence the text that is generated. We show these models can be easily attacked, i.e., induced to perform arbitrary un-aligned behavior through adversarial perturbation of the input image. We conjecture that improved NLP attacks may demonstrate this same level of adversarial control over text-only models.



## **24. Exploring Frequencies via Feature Mixing and Meta-Learning for Improving Adversarial Transferability**

cs.CV

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2405.03193v1) [paper-pdf](http://arxiv.org/pdf/2405.03193v1)

**Authors**: Juanjuan Weng, Zhiming Luo, Shaozi Li

**Abstract**: Recent studies have shown that Deep Neural Networks (DNNs) are susceptible to adversarial attacks, with frequency-domain analysis underscoring the significance of high-frequency components in influencing model predictions. Conversely, targeting low-frequency components has been effective in enhancing attack transferability on black-box models. In this study, we introduce a frequency decomposition-based feature mixing method to exploit these frequency characteristics in both clean and adversarial samples. Our findings suggest that incorporating features of clean samples into adversarial features extracted from adversarial examples is more effective in attacking normally-trained models, while combining clean features with the adversarial features extracted from low-frequency parts decomposed from the adversarial samples yields better results in attacking defense models. However, a conflict issue arises when these two mixing approaches are employed simultaneously. To tackle the issue, we propose a cross-frequency meta-optimization approach comprising the meta-train step, meta-test step, and final update. In the meta-train step, we leverage the low-frequency components of adversarial samples to boost the transferability of attacks against defense models. Meanwhile, in the meta-test step, we utilize adversarial samples to stabilize gradients, thereby enhancing the attack's transferability against normally trained models. For the final update, we update the adversarial sample based on the gradients obtained from both meta-train and meta-test steps. Our proposed method is evaluated through extensive experiments on the ImageNet-Compatible dataset, affirming its effectiveness in improving the transferability of attacks on both normally-trained CNNs and defense models.   The source code is available at https://github.com/WJJLL/MetaSSA.



## **25. To Each (Textual Sequence) Its Own: Improving Memorized-Data Unlearning in Large Language Models**

cs.LG

Published as a conference paper at ICML 2024

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2405.03097v1) [paper-pdf](http://arxiv.org/pdf/2405.03097v1)

**Authors**: George-Octavian Barbulescu, Peter Triantafillou

**Abstract**: LLMs have been found to memorize training textual sequences and regurgitate verbatim said sequences during text generation time. This fact is known to be the cause of privacy and related (e.g., copyright) problems. Unlearning in LLMs then takes the form of devising new algorithms that will properly deal with these side-effects of memorized data, while not hurting the model's utility. We offer a fresh perspective towards this goal, namely, that each textual sequence to be forgotten should be treated differently when being unlearned based on its degree of memorization within the LLM. We contribute a new metric for measuring unlearning quality, an adversarial attack showing that SOTA algorithms lacking this perspective fail for privacy, and two new unlearning methods based on Gradient Ascent and Task Arithmetic, respectively. A comprehensive performance evaluation across an extensive suite of NLP tasks then mapped the solution space, identifying the best solutions under different scales in model capacities and forget set sizes and quantified the gains of the new approaches.



## **26. A Characterization of Semi-Supervised Adversarially-Robust PAC Learnability**

cs.LG

NeurIPS 2022 camera-ready

**SubmitDate**: 2024-05-05    [abs](http://arxiv.org/abs/2202.05420v3) [paper-pdf](http://arxiv.org/pdf/2202.05420v3)

**Authors**: Idan Attias, Steve Hanneke, Yishay Mansour

**Abstract**: We study the problem of learning an adversarially robust predictor to test time attacks in the semi-supervised PAC model. We address the question of how many labeled and unlabeled examples are required to ensure learning. We show that having enough unlabeled data (the size of a labeled sample that a fully-supervised method would require), the labeled sample complexity can be arbitrarily smaller compared to previous works, and is sharply characterized by a different complexity measure. We prove nearly matching upper and lower bounds on this sample complexity. This shows that there is a significant benefit in semi-supervised robust learning even in the worst-case distribution-free model, and establishes a gap between the supervised and semi-supervised label complexities which is known not to hold in standard non-robust PAC learning.



## **27. Adversarially Robust PAC Learnability of Real-Valued Functions**

cs.LG

accepted to ICML2023

**SubmitDate**: 2024-05-05    [abs](http://arxiv.org/abs/2206.12977v3) [paper-pdf](http://arxiv.org/pdf/2206.12977v3)

**Authors**: Idan Attias, Steve Hanneke

**Abstract**: We study robustness to test-time adversarial attacks in the regression setting with $\ell_p$ losses and arbitrary perturbation sets. We address the question of which function classes are PAC learnable in this setting. We show that classes of finite fat-shattering dimension are learnable in both realizable and agnostic settings. Moreover, for convex function classes, they are even properly learnable. In contrast, some non-convex function classes provably require improper learning algorithms. Our main technique is based on a construction of an adversarially robust sample compression scheme of a size determined by the fat-shattering dimension. Along the way, we introduce a novel agnostic sample compression scheme for real-valued functions, which may be of independent interest.



## **28. Defense against Joint Poison and Evasion Attacks: A Case Study of DERMS**

cs.CR

**SubmitDate**: 2024-05-05    [abs](http://arxiv.org/abs/2405.02989v1) [paper-pdf](http://arxiv.org/pdf/2405.02989v1)

**Authors**: Zain ul Abdeen, Padmaksha Roy, Ahmad Al-Tawaha, Rouxi Jia, Laura Freeman, Peter Beling, Chen-Ching Liu, Alberto Sangiovanni-Vincentelli, Ming Jin

**Abstract**: There is an upward trend of deploying distributed energy resource management systems (DERMS) to control modern power grids. However, DERMS controller communication lines are vulnerable to cyberattacks that could potentially impact operational reliability. While a data-driven intrusion detection system (IDS) can potentially thwart attacks during deployment, also known as the evasion attack, the training of the detection algorithm may be corrupted by adversarial data injected into the database, also known as the poisoning attack. In this paper, we propose the first framework of IDS that is robust against joint poisoning and evasion attacks. We formulate the defense mechanism as a bilevel optimization, where the inner and outer levels deal with attacks that occur during training time and testing time, respectively. We verify the robustness of our method on the IEEE-13 bus feeder model against a diverse set of poisoning and evasion attack scenarios. The results indicate that our proposed method outperforms the baseline technique in terms of accuracy, precision, and recall for intrusion detection.



## **29. You Only Need Half: Boosting Data Augmentation by Using Partial Content**

cs.CV

Technical report,16 pages

**SubmitDate**: 2024-05-05    [abs](http://arxiv.org/abs/2405.02830v1) [paper-pdf](http://arxiv.org/pdf/2405.02830v1)

**Authors**: Juntao Hu, Yuan Wu

**Abstract**: We propose a novel data augmentation method termed You Only Need hAlf (YONA), which simplifies the augmentation process. YONA bisects an image, substitutes one half with noise, and applies data augmentation techniques to the remaining half. This method reduces the redundant information in the original image, encourages neural networks to recognize objects from incomplete views, and significantly enhances neural networks' robustness. YONA is distinguished by its properties of parameter-free, straightforward application, enhancing various existing data augmentation strategies, and thereby bolstering neural networks' robustness without additional computational cost. To demonstrate YONA's efficacy, extensive experiments were carried out. These experiments confirm YONA's compatibility with diverse data augmentation methods and neural network architectures, yielding substantial improvements in CIFAR classification tasks, sometimes outperforming conventional image-level data augmentation methods. Furthermore, YONA markedly increases the resilience of neural networks to adversarial attacks. Additional experiments exploring YONA's variants conclusively show that masking half of an image optimizes performance. The code is available at https://github.com/HansMoe/YONA.



## **30. Trojans in Large Language Models of Code: A Critical Review through a Trigger-Based Taxonomy**

cs.SE

arXiv admin note: substantial text overlap with arXiv:2305.03803

**SubmitDate**: 2024-05-05    [abs](http://arxiv.org/abs/2405.02828v1) [paper-pdf](http://arxiv.org/pdf/2405.02828v1)

**Authors**: Aftab Hussain, Md Rafiqul Islam Rabin, Toufique Ahmed, Bowen Xu, Premkumar Devanbu, Mohammad Amin Alipour

**Abstract**: Large language models (LLMs) have provided a lot of exciting new capabilities in software development. However, the opaque nature of these models makes them difficult to reason about and inspect. Their opacity gives rise to potential security risks, as adversaries can train and deploy compromised models to disrupt the software development process in the victims' organization.   This work presents an overview of the current state-of-the-art trojan attacks on large language models of code, with a focus on triggers -- the main design point of trojans -- with the aid of a novel unifying trigger taxonomy framework. We also aim to provide a uniform definition of the fundamental concepts in the area of trojans in Code LLMs. Finally, we draw implications of findings on how code models learn on trigger design.



## **31. Assessing Adversarial Robustness of Large Language Models: An Empirical Study**

cs.CL

16 pages, 9 figures, 10 tables

**SubmitDate**: 2024-05-04    [abs](http://arxiv.org/abs/2405.02764v1) [paper-pdf](http://arxiv.org/pdf/2405.02764v1)

**Authors**: Zeyu Yang, Zhao Meng, Xiaochen Zheng, Roger Wattenhofer

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing, but their robustness against adversarial attacks remains a critical concern. We presents a novel white-box style attack approach that exposes vulnerabilities in leading open-source LLMs, including Llama, OPT, and T5. We assess the impact of model size, structure, and fine-tuning strategies on their resistance to adversarial perturbations. Our comprehensive evaluation across five diverse text classification tasks establishes a new benchmark for LLM robustness. The findings of this study have far-reaching implications for the reliable deployment of LLMs in real-world applications and contribute to the advancement of trustworthy AI systems.



## **32. Updating Windows Malware Detectors: Balancing Robustness and Regression against Adversarial EXEmples**

cs.CR

11 pages, 3 figures, 7 tables

**SubmitDate**: 2024-05-04    [abs](http://arxiv.org/abs/2405.02646v1) [paper-pdf](http://arxiv.org/pdf/2405.02646v1)

**Authors**: Matous Kozak, Luca Demetrio, Dmitrijs Trizna, Fabio Roli

**Abstract**: Adversarial EXEmples are carefully-perturbed programs tailored to evade machine learning Windows malware detectors, with an on-going effort in developing robust models able to address detection effectiveness. However, even if robust models can prevent the majority of EXEmples, to maintain predictive power over time, models are fine-tuned to newer threats, leading either to partial updates or time-consuming retraining from scratch. Thus, even if the robustness against attacks is higher, the new models might suffer a regression in performance by misclassifying threats that were previously correctly detected. For these reasons, we study the trade-off between accuracy and regression when updating Windows malware detectors, by proposing EXE-scanner, a plugin that can be chained to existing detectors to promptly stop EXEmples without causing regression. We empirically show that previously-proposed hardening techniques suffer a regression of accuracy when updating non-robust models. On the contrary, we show that EXE-scanner exhibits comparable performance to robust models without regression of accuracy, and we show how to properly chain it after the base classifier to obtain the best performance without the need of costly retraining. To foster reproducibility, we openly release source code, along with the dataset of adversarial EXEmples based on state-of-the-art perturbation algorithms.



## **33. A Group Key Establishment Scheme**

cs.CR

**SubmitDate**: 2024-05-04    [abs](http://arxiv.org/abs/2109.15037v2) [paper-pdf](http://arxiv.org/pdf/2109.15037v2)

**Authors**: Sueda Guzey, Gunes Karabulut Kurt, Enver Ozdemir

**Abstract**: Group authentication is a method of confirmation that a set of users belong to a group and of distributing a common key among them. Unlike the standard authentication schemes where one central authority authenticates users one by one, group authentication can handle the authentication process at once for all members of the group. The recently presented group authentication algorithms mainly exploit Lagrange's polynomial interpolation along with elliptic curve groups over finite fields. As a fresh approach, this work suggests use of linear spaces for group authentication and key establishment for a group of any size. The approach with linear spaces introduces a reduced computation and communication load to establish a common shared key among the group members. The advantages of using vector spaces make the proposed method applicable to energy and resource constrained devices. In addition to providing lightweight authentication and key agreement, this proposal allows any user in a group to make a non-member to be a member, which is expected to be useful for autonomous systems in the future. The scheme is designed in a way that the sponsors of such members can easily be recognized by anyone in the group. Unlike the other group authentication schemes based on Lagrange's polynomial interpolation, the proposed scheme doesn't provide a tool for adversaries to compromise the whole group secrets by using only a few members' shares as well as it allows to recognize a non-member easily, which prevents service interruption attacks.



## **34. Leveraging the Human Ventral Visual Stream to Improve Neural Network Robustness**

cs.CV

**SubmitDate**: 2024-05-04    [abs](http://arxiv.org/abs/2405.02564v1) [paper-pdf](http://arxiv.org/pdf/2405.02564v1)

**Authors**: Zhenan Shao, Linjian Ma, Bo Li, Diane M. Beck

**Abstract**: Human object recognition exhibits remarkable resilience in cluttered and dynamic visual environments. In contrast, despite their unparalleled performance across numerous visual tasks, Deep Neural Networks (DNNs) remain far less robust than humans, showing, for example, a surprising susceptibility to adversarial attacks involving image perturbations that are (almost) imperceptible to humans. Human object recognition likely owes its robustness, in part, to the increasingly resilient representations that emerge along the hierarchy of the ventral visual cortex. Here we show that DNNs, when guided by neural representations from a hierarchical sequence of regions in the human ventral visual stream, display increasing robustness to adversarial attacks. These neural-guided models also exhibit a gradual shift towards more human-like decision-making patterns and develop hierarchically smoother decision surfaces. Importantly, the resulting representational spaces differ in important ways from those produced by conventional smoothing methods, suggesting that such neural-guidance may provide previously unexplored robustness solutions. Our findings support the gradual emergence of human robustness along the ventral visual hierarchy and suggest that the key to DNN robustness may lie in increasing emulation of the human brain.



## **35. Machine Learning Robustness: A Primer**

cs.LG

**SubmitDate**: 2024-05-04    [abs](http://arxiv.org/abs/2404.00897v3) [paper-pdf](http://arxiv.org/pdf/2404.00897v3)

**Authors**: Houssem Ben Braiek, Foutse Khomh

**Abstract**: This chapter explores the foundational concept of robustness in Machine Learning (ML) and its integral role in establishing trustworthiness in Artificial Intelligence (AI) systems. The discussion begins with a detailed definition of robustness, portraying it as the ability of ML models to maintain stable performance across varied and unexpected environmental conditions. ML robustness is dissected through several lenses: its complementarity with generalizability; its status as a requirement for trustworthy AI; its adversarial vs non-adversarial aspects; its quantitative metrics; and its indicators such as reproducibility and explainability. The chapter delves into the factors that impede robustness, such as data bias, model complexity, and the pitfalls of underspecified ML pipelines. It surveys key techniques for robustness assessment from a broad perspective, including adversarial attacks, encompassing both digital and physical realms. It covers non-adversarial data shifts and nuances of Deep Learning (DL) software testing methodologies. The discussion progresses to explore amelioration strategies for bolstering robustness, starting with data-centric approaches like debiasing and augmentation. Further examination includes a variety of model-centric methods such as transfer learning, adversarial training, and randomized smoothing. Lastly, post-training methods are discussed, including ensemble techniques, pruning, and model repairs, emerging as cost-effective strategies to make models more resilient against the unpredictable. This chapter underscores the ongoing challenges and limitations in estimating and achieving ML robustness by existing approaches. It offers insights and directions for future research on this crucial concept, as a prerequisite for trustworthy AI systems.



## **36. GReAT: A Graph Regularized Adversarial Training Method**

cs.LG

25 pages including references. 7 figures and 6 tables

**SubmitDate**: 2024-05-03    [abs](http://arxiv.org/abs/2310.05336v2) [paper-pdf](http://arxiv.org/pdf/2310.05336v2)

**Authors**: Samet Bayram, Kenneth Barner

**Abstract**: This paper presents GReAT (Graph Regularized Adversarial Training), a novel regularization method designed to enhance the robust classification performance of deep learning models. Adversarial examples, characterized by subtle perturbations that can mislead models, pose a significant challenge in machine learning. Although adversarial training is effective in defending against such attacks, it often overlooks the underlying data structure. In response, GReAT integrates graph based regularization into the adversarial training process, leveraging the data's inherent structure to enhance model robustness. By incorporating graph information during training, GReAT defends against adversarial attacks and improves generalization to unseen data. Extensive evaluations on benchmark datasets demonstrate that GReAT outperforms state of the art methods in robustness, achieving notable improvements in classification accuracy. Specifically, compared to the second best methods, GReAT achieves a performance increase of approximately 4.87% for CIFAR10 against FGSM attack and 10.57% for SVHN against FGSM attack. Additionally, for CIFAR10, GReAT demonstrates a performance increase of approximately 11.05% against PGD attack, and for SVHN, a 5.54% increase against PGD attack. This paper provides detailed insights into the proposed methodology, including numerical results and comparisons with existing approaches, highlighting the significant impact of GReAT in advancing the performance of deep learning models.



## **37. Improving Interpretation Faithfulness for Vision Transformers**

cs.CV

Accepted by ICML 2024

**SubmitDate**: 2024-05-03    [abs](http://arxiv.org/abs/2311.17983v2) [paper-pdf](http://arxiv.org/pdf/2311.17983v2)

**Authors**: Lijie Hu, Yixin Liu, Ninghao Liu, Mengdi Huai, Lichao Sun, Di Wang

**Abstract**: Vision Transformers (ViTs) have achieved state-of-the-art performance for various vision tasks. One reason behind the success lies in their ability to provide plausible innate explanations for the behavior of neural architectures. However, ViTs suffer from issues with explanation faithfulness, as their focal points are fragile to adversarial attacks and can be easily changed with even slight perturbations on the input image. In this paper, we propose a rigorous approach to mitigate these issues by introducing Faithful ViTs (FViTs). Briefly speaking, an FViT should have the following two properties: (1) The top-$k$ indices of its self-attention vector should remain mostly unchanged under input perturbation, indicating stable explanations; (2) The prediction distribution should be robust to perturbations. To achieve this, we propose a new method called Denoised Diffusion Smoothing (DDS), which adopts randomized smoothing and diffusion-based denoising. We theoretically prove that processing ViTs directly with DDS can turn them into FViTs. We also show that Gaussian noise is nearly optimal for both $\ell_2$ and $\ell_\infty$-norm cases. Finally, we demonstrate the effectiveness of our approach through comprehensive experiments and evaluations. Results show that FViTs are more robust against adversarial attacks while maintaining the explainability of attention, indicating higher faithfulness.



## **38. Adversarial Botometer: Adversarial Analysis for Social Bot Detection**

cs.SI

**SubmitDate**: 2024-05-03    [abs](http://arxiv.org/abs/2405.02016v1) [paper-pdf](http://arxiv.org/pdf/2405.02016v1)

**Authors**: Shaghayegh Najari, Davood Rafiee, Mostafa Salehi, Reza Farahbakhsh

**Abstract**: Social bots play a significant role in many online social networks (OSN) as they imitate human behavior. This fact raises difficult questions about their capabilities and potential risks. Given the recent advances in Generative AI (GenAI), social bots are capable of producing highly realistic and complex content that mimics human creativity. As the malicious social bots emerge to deceive people with their unrealistic content, identifying them and distinguishing the content they produce has become an actual challenge for numerous social platforms. Several approaches to this problem have already been proposed in the literature, but the proposed solutions have not been widely evaluated. To address this issue, we evaluate the behavior of a text-based bot detector in a competitive environment where some scenarios are proposed: \textit{First}, the tug-of-war between a bot and a bot detector is examined. It is interesting to analyze which party is more likely to prevail and which circumstances influence these expectations. In this regard, we model the problem as a synthetic adversarial game in which a conversational bot and a bot detector are engaged in strategic online interactions. \textit{Second}, the bot detection model is evaluated under attack examples generated by a social bot; to this end, we poison the dataset with attack examples and evaluate the model performance under this condition. \textit{Finally}, to investigate the impact of the dataset, a cross-domain analysis is performed. Through our comprehensive evaluation of different categories of social bots using two benchmark datasets, we were able to demonstrate some achivement that could be utilized in future works.



## **39. From Attack to Defense: Insights into Deep Learning Security Measures in Black-Box Settings**

cs.CR

**SubmitDate**: 2024-05-03    [abs](http://arxiv.org/abs/2405.01963v1) [paper-pdf](http://arxiv.org/pdf/2405.01963v1)

**Authors**: Firuz Juraev, Mohammed Abuhamad, Eric Chan-Tin, George K. Thiruvathukal, Tamer Abuhmed

**Abstract**: Deep Learning (DL) is rapidly maturing to the point that it can be used in safety- and security-crucial applications. However, adversarial samples, which are undetectable to the human eye, pose a serious threat that can cause the model to misbehave and compromise the performance of such applications. Addressing the robustness of DL models has become crucial to understanding and defending against adversarial attacks. In this study, we perform comprehensive experiments to examine the effect of adversarial attacks and defenses on various model architectures across well-known datasets. Our research focuses on black-box attacks such as SimBA, HopSkipJump, MGAAttack, and boundary attacks, as well as preprocessor-based defensive mechanisms, including bits squeezing, median smoothing, and JPEG filter. Experimenting with various models, our results demonstrate that the level of noise needed for the attack increases as the number of layers increases. Moreover, the attack success rate decreases as the number of layers increases. This indicates that model complexity and robustness have a significant relationship. Investigating the diversity and robustness relationship, our experiments with diverse models show that having a large number of parameters does not imply higher robustness. Our experiments extend to show the effects of the training dataset on model robustness. Using various datasets such as ImageNet-1000, CIFAR-100, and CIFAR-10 are used to evaluate the black-box attacks. Considering the multiple dimensions of our analysis, e.g., model complexity and training dataset, we examined the behavior of black-box attacks when models apply defenses. Our results show that applying defense strategies can significantly reduce attack effectiveness. This research provides in-depth analysis and insight into the robustness of DL models against various attacks, and defenses.



## **40. Impact of Architectural Modifications on Deep Learning Adversarial Robustness**

cs.CV

**SubmitDate**: 2024-05-03    [abs](http://arxiv.org/abs/2405.01934v1) [paper-pdf](http://arxiv.org/pdf/2405.01934v1)

**Authors**: Firuz Juraev, Mohammed Abuhamad, Simon S. Woo, George K Thiruvathukal, Tamer Abuhmed

**Abstract**: Rapid advancements of deep learning are accelerating adoption in a wide variety of applications, including safety-critical applications such as self-driving vehicles, drones, robots, and surveillance systems. These advancements include applying variations of sophisticated techniques that improve the performance of models. However, such models are not immune to adversarial manipulations, which can cause the system to misbehave and remain unnoticed by experts. The frequency of modifications to existing deep learning models necessitates thorough analysis to determine the impact on models' robustness. In this work, we present an experimental evaluation of the effects of model modifications on deep learning model robustness using adversarial attacks. Our methodology involves examining the robustness of variations of models against various adversarial attacks. By conducting our experiments, we aim to shed light on the critical issue of maintaining the reliability and safety of deep learning models in safety- and security-critical applications. Our results indicate the pressing demand for an in-depth assessment of the effects of model changes on the robustness of models.



## **41. Stability of Explainable Recommendation**

cs.IR

**SubmitDate**: 2024-05-03    [abs](http://arxiv.org/abs/2405.01849v1) [paper-pdf](http://arxiv.org/pdf/2405.01849v1)

**Authors**: Sairamvinay Vijayaraghavan, Prasant Mohapatra

**Abstract**: Explainable Recommendation has been gaining attention over the last few years in industry and academia. Explanations provided along with recommendations in a recommender system framework have many uses: particularly reasoning why a suggestion is provided and how well an item aligns with a user's personalized preferences. Hence, explanations can play a huge role in influencing users to purchase products. However, the reliability of the explanations under varying scenarios has not been strictly verified from an empirical perspective. Unreliable explanations can bear strong consequences such as attackers leveraging explanations for manipulating and tempting users to purchase target items that the attackers would want to promote. In this paper, we study the vulnerability of existent feature-oriented explainable recommenders, particularly analyzing their performance under different levels of external noises added into model parameters. We conducted experiments by analyzing three important state-of-the-art (SOTA) explainable recommenders when trained on two widely used e-commerce based recommendation datasets of different scales. We observe that all the explainable models are vulnerable to increased noise levels. Experimental results verify our hypothesis that the ability to explain recommendations does decrease along with increasing noise levels and particularly adversarial noise does contribute to a much stronger decrease. Our study presents an empirical verification on the topic of robust explanations in recommender systems which can be extended to different types of explainable recommenders in RS.



## **42. A Novel Approach to Guard from Adversarial Attacks using Stable Diffusion**

cs.LG

**SubmitDate**: 2024-05-03    [abs](http://arxiv.org/abs/2405.01838v1) [paper-pdf](http://arxiv.org/pdf/2405.01838v1)

**Authors**: Trinath Sai Subhash Reddy Pittala, Uma Maheswara Rao Meleti, Geethakrishna Puligundla

**Abstract**: Recent developments in adversarial machine learning have highlighted the importance of building robust AI systems to protect against increasingly sophisticated attacks. While frameworks like AI Guardian are designed to defend against these threats, they often rely on assumptions that can limit their effectiveness. For example, they may assume attacks only come from one direction or include adversarial images in their training data. Our proposal suggests a different approach to the AI Guardian framework. Instead of including adversarial examples in the training process, we propose training the AI system without them. This aims to create a system that is inherently resilient to a wider range of attacks. Our method focuses on a dynamic defense strategy using stable diffusion that learns continuously and models threats comprehensively. We believe this approach can lead to a more generalized and robust defense against adversarial attacks.   In this paper, we outline our proposed approach, including the theoretical basis, experimental design, and expected impact on improving AI security against adversarial threats.



## **43. Explainability Guided Adversarial Evasion Attacks on Malware Detectors**

cs.CR

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01728v1) [paper-pdf](http://arxiv.org/pdf/2405.01728v1)

**Authors**: Kshitiz Aryal, Maanak Gupta, Mahmoud Abdelsalam, Moustafa Saleh

**Abstract**: As the focus on security of Artificial Intelligence (AI) is becoming paramount, research on crafting and inserting optimal adversarial perturbations has become increasingly critical. In the malware domain, this adversarial sample generation relies heavily on the accuracy and placement of crafted perturbation with the goal of evading a trained classifier. This work focuses on applying explainability techniques to enhance the adversarial evasion attack on a machine-learning-based Windows PE malware detector. The explainable tool identifies the regions of PE malware files that have the most significant impact on the decision-making process of a given malware detector, and therefore, the same regions can be leveraged to inject the adversarial perturbation for maximum efficiency. Profiling all the PE malware file regions based on their impact on the malware detector's decision enables the derivation of an efficient strategy for identifying the optimal location for perturbation injection. The strategy should incorporate the region's significance in influencing the malware detector's decision and the sensitivity of the PE malware file's integrity towards modifying that region. To assess the utility of explainable AI in crafting an adversarial sample of Windows PE malware, we utilize the DeepExplainer module of SHAP for determining the contribution of each region of PE malware to its detection by a CNN-based malware detector, MalConv. Furthermore, we analyzed the significance of SHAP values at a more granular level by subdividing each section of Windows PE into small subsections. We then performed an adversarial evasion attack on the subsections based on the corresponding SHAP values of the byte sequences.



## **44. ATTAXONOMY: Unpacking Differential Privacy Guarantees Against Practical Adversaries**

cs.CR

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01716v1) [paper-pdf](http://arxiv.org/pdf/2405.01716v1)

**Authors**: Rachel Cummings, Shlomi Hod, Jayshree Sarathy, Marika Swanberg

**Abstract**: Differential Privacy (DP) is a mathematical framework that is increasingly deployed to mitigate privacy risks associated with machine learning and statistical analyses. Despite the growing adoption of DP, its technical privacy parameters do not lend themselves to an intelligible description of the real-world privacy risks associated with that deployment: the guarantee that most naturally follows from the DP definition is protection against membership inference by an adversary who knows all but one data record and has unlimited auxiliary knowledge. In many settings, this adversary is far too strong to inform how to set real-world privacy parameters.   One approach for contextualizing privacy parameters is via defining and measuring the success of technical attacks, but doing so requires a systematic categorization of the relevant attack space. In this work, we offer a detailed taxonomy of attacks, showing the various dimensions of attacks and highlighting that many real-world settings have been understudied. Our taxonomy provides a roadmap for analyzing real-world deployments and developing theoretical bounds for more informative privacy attacks. We operationalize our taxonomy by using it to analyze a real-world case study, the Israeli Ministry of Health's recent release of a birth dataset using DP, showing how the taxonomy enables fine-grained threat modeling and provides insight towards making informed privacy parameter choices. Finally, we leverage the taxonomy towards defining a more realistic attack than previously considered in the literature, namely a distributional reconstruction attack: we generalize Balle et al.'s notion of reconstruction robustness to a less-informed adversary with distributional uncertainty, and extend the worst-case guarantees of DP to this average-case setting.



## **45. David and Goliath: An Empirical Evaluation of Attacks and Defenses for QNNs at the Deep Edge**

cs.LG

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2404.05688v2) [paper-pdf](http://arxiv.org/pdf/2404.05688v2)

**Authors**: Miguel Costa, Sandro Pinto

**Abstract**: ML is shifting from the cloud to the edge. Edge computing reduces the surface exposing private data and enables reliable throughput guarantees in real-time applications. Of the panoply of devices deployed at the edge, resource-constrained MCUs, e.g., Arm Cortex-M, are more prevalent, orders of magnitude cheaper, and less power-hungry than application processors or GPUs. Thus, enabling intelligence at the deep edge is the zeitgeist, with researchers focusing on unveiling novel approaches to deploy ANNs on these constrained devices. Quantization is a well-established technique that has proved effective in enabling the deployment of neural networks on MCUs; however, it is still an open question to understand the robustness of QNNs in the face of adversarial examples.   To fill this gap, we empirically evaluate the effectiveness of attacks and defenses from (full-precision) ANNs on (constrained) QNNs. Our evaluation includes three QNNs targeting TinyML applications, ten attacks, and six defenses. With this study, we draw a set of interesting findings. First, quantization increases the point distance to the decision boundary and leads the gradient estimated by some attacks to explode or vanish. Second, quantization can act as a noise attenuator or amplifier, depending on the noise magnitude, and causes gradient misalignment. Regarding adversarial defenses, we conclude that input pre-processing defenses show impressive results on small perturbations; however, they fall short as the perturbation increases. At the same time, train-based defenses increase the average point distance to the decision boundary, which holds after quantization. However, we argue that train-based defenses still need to smooth the quantization-shift and gradient misalignment phenomenons to counteract adversarial example transferability to QNNs. All artifacts are open-sourced to enable independent validation of results.



## **46. Adversarial Attacks on Reinforcement Learning Agents for Command and Control**

cs.CR

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01693v1) [paper-pdf](http://arxiv.org/pdf/2405.01693v1)

**Authors**: Ahaan Dabholkar, James Z. Hare, Mark Mittrick, John Richardson, Nicholas Waytowich, Priya Narayanan, Saurabh Bagchi

**Abstract**: Given the recent impact of Deep Reinforcement Learning in training agents to win complex games like StarCraft and DoTA(Defense Of The Ancients) - there has been a surge in research for exploiting learning based techniques for professional wargaming, battlefield simulation and modeling. Real time strategy games and simulators have become a valuable resource for operational planning and military research. However, recent work has shown that such learning based approaches are highly susceptible to adversarial perturbations. In this paper, we investigate the robustness of an agent trained for a Command and Control task in an environment that is controlled by an active adversary. The C2 agent is trained on custom StarCraft II maps using the state of the art RL algorithms - A3C and PPO. We empirically show that an agent trained using these algorithms is highly susceptible to noise injected by the adversary and investigate the effects these perturbations have on the performance of the trained agent. Our work highlights the urgent need to develop more robust training algorithms especially for critical arenas like the battlefield.



## **47. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

cs.CR

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2312.03853v3) [paper-pdf](http://arxiv.org/pdf/2312.03853v3)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: Recently, we have witnessed a rise in the use of Large Language Models (LLMs), especially in applications like chatbot assistants. Safety mechanisms and specialized training procedures are implemented to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Bard (and, to some extent, Bing chat) by making them impersonate complex personas with personality characteristics that are not aligned with a truthful assistant. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversations then followed a role-play style to elicit prohibited responses. By making use of personas, we show that such responses are actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial personas, one can overcome safety mechanisms set out by ChatGPT and Bard. We also introduce several ways of activating such adversarial personas, which show that both chatbots are vulnerable to this kind of attack. With the same principle, we introduce two defenses that push the model to interpret trustworthy personalities and make it more robust against such attacks.



## **48. Generative AI in Cybersecurity**

cs.CR

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01674v1) [paper-pdf](http://arxiv.org/pdf/2405.01674v1)

**Authors**: Shivani Metta, Isaac Chang, Jack Parker, Michael P. Roman, Arturo F. Ehuan

**Abstract**: The dawn of Generative Artificial Intelligence (GAI), characterized by advanced models such as Generative Pre-trained Transformers (GPT) and other Large Language Models (LLMs), has been pivotal in reshaping the field of data analysis, pattern recognition, and decision-making processes. This surge in GAI technology has ushered in not only innovative opportunities for data processing and automation but has also introduced significant cybersecurity challenges.   As GAI rapidly progresses, it outstrips the current pace of cybersecurity protocols and regulatory frameworks, leading to a paradox wherein the same innovations meant to safeguard digital infrastructures also enhance the arsenal available to cyber criminals. These adversaries, adept at swiftly integrating and exploiting emerging technologies, may utilize GAI to develop malware that is both more covert and adaptable, thus complicating traditional cybersecurity efforts.   The acceleration of GAI presents an ambiguous frontier for cybersecurity experts, offering potent tools for threat detection and response, while concurrently providing cyber attackers with the means to engineer more intricate and potent malware. Through the joint efforts of Duke Pratt School of Engineering, Coalfire, and Safebreach, this research undertakes a meticulous analysis of how malicious agents are exploiting GAI to augment their attack strategies, emphasizing a critical issue for the integrity of future cybersecurity initiatives. The study highlights the critical need for organizations to proactively identify and develop more complex defensive strategies to counter the sophisticated employment of GAI in malware creation.



## **49. Position Paper: Beyond Robustness Against Single Attack Types**

cs.LG

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01349v1) [paper-pdf](http://arxiv.org/pdf/2405.01349v1)

**Authors**: Sihui Dai, Chong Xiang, Tong Wu, Prateek Mittal

**Abstract**: Current research on defending against adversarial examples focuses primarily on achieving robustness against a single attack type such as $\ell_2$ or $\ell_{\infty}$-bounded attacks. However, the space of possible perturbations is much larger and currently cannot be modeled by a single attack type. The discrepancy between the focus of current defenses and the space of attacks of interest calls to question the practicality of existing defenses and the reliability of their evaluation. In this position paper, we argue that the research community should look beyond single attack robustness, and we draw attention to three potential directions involving robustness against multiple attacks: simultaneous multiattack robustness, unforeseen attack robustness, and a newly defined problem setting which we call continual adaptive robustness. We provide a unified framework which rigorously defines these problem settings, synthesize existing research in these fields, and outline open directions. We hope that our position paper inspires more research in simultaneous multiattack, unforeseen attack, and continual adaptive robustness.



## **50. LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked**

cs.CL

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2308.07308v4) [paper-pdf](http://arxiv.org/pdf/2308.07308v4)

**Authors**: Mansi Phute, Alec Helbling, Matthew Hull, ShengYun Peng, Sebastian Szyller, Cory Cornelius, Duen Horng Chau

**Abstract**: Large language models (LLMs) are popular for high-quality text generation but can produce harmful content, even when aligned with human values through reinforcement learning. Adversarial prompts can bypass their safety measures. We propose LLM Self Defense, a simple approach to defend against these attacks by having an LLM screen the induced responses. Our method does not require any fine-tuning, input preprocessing, or iterative output generation. Instead, we incorporate the generated content into a pre-defined prompt and employ another instance of an LLM to analyze the text and predict whether it is harmful. We test LLM Self Defense on GPT 3.5 and Llama 2, two of the current most prominent LLMs against various types of attacks, such as forcefully inducing affirmative responses to prompts and prompt engineering attacks. Notably, LLM Self Defense succeeds in reducing the attack success rate to virtually 0 using both GPT 3.5 and Llama 2. The code is publicly available at https://github.com/poloclub/llm-self-defense



