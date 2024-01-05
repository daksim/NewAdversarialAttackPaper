# Latest Adversarial Attack Papers
**update at 2024-01-05 10:01:00**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Mining Temporal Attack Patterns from Cyberthreat Intelligence Reports**

cs.CR

A modified version of this pre-print is submitted to IEEE  Transactions on Software Engineering, and is under review

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.01883v1) [paper-pdf](http://arxiv.org/pdf/2401.01883v1)

**Authors**: Md Rayhanur Rahman, Brandon Wroblewski, Quinn Matthews, Brantley Morgan, Tim Menzies, Laurie Williams

**Abstract**: Defending from cyberattacks requires practitioners to operate on high-level adversary behavior. Cyberthreat intelligence (CTI) reports on past cyberattack incidents describe the chain of malicious actions with respect to time. To avoid repeating cyberattack incidents, practitioners must proactively identify and defend against recurring chain of actions - which we refer to as temporal attack patterns. Automatically mining the patterns among actions provides structured and actionable information on the adversary behavior of past cyberattacks. The goal of this paper is to aid security practitioners in prioritizing and proactive defense against cyberattacks by mining temporal attack patterns from cyberthreat intelligence reports. To this end, we propose ChronoCTI, an automated pipeline for mining temporal attack patterns from cyberthreat intelligence (CTI) reports of past cyberattacks. To construct ChronoCTI, we build the ground truth dataset of temporal attack patterns and apply state-of-the-art large language models, natural language processing, and machine learning techniques. We apply ChronoCTI on a set of 713 CTI reports, where we identify 124 temporal attack patterns - which we categorize into nine pattern categories. We identify that the most prevalent pattern category is to trick victim users into executing malicious code to initiate the attack, followed by bypassing the anti-malware system in the victim network. Based on the observed patterns, we advocate organizations to train users about cybersecurity best practices, introduce immutable operating systems with limited functionalities, and enforce multi-user authentications. Moreover, we advocate practitioners to leverage the automated mining capability of ChronoCTI and design countermeasures against the recurring attack patterns.



## **2. Attackers reveal their arsenal: An investigation of adversarial techniques in CTI reports**

cs.CR

This version is submitted to ACM Transactions on Privacy and  Security. This version is under review

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.01865v1) [paper-pdf](http://arxiv.org/pdf/2401.01865v1)

**Authors**: Md Rayhanur Rahman, Setu Kumar Basak, Rezvan Mahdavi Hezaveh, Laurie Williams

**Abstract**: Context: Cybersecurity vendors often publish cyber threat intelligence (CTI) reports, referring to the written artifacts on technical and forensic analysis of the techniques used by the malware in APT attacks. Objective: The goal of this research is to inform cybersecurity practitioners about how adversaries form cyberattacks through an analysis of adversarial techniques documented in cyberthreat intelligence reports. Dataset: We use 594 adversarial techniques cataloged in MITRE ATT\&CK. We systematically construct a set of 667 CTI reports that MITRE ATT\&CK used as citations in the descriptions of the cataloged adversarial techniques. Methodology: We analyze the frequency and trend of adversarial techniques, followed by a qualitative analysis of the implementation of techniques. Next, we perform association rule mining to identify pairs of techniques recurring in APT attacks. We then perform qualitative analysis to identify the underlying relations among the techniques in the recurring pairs. Findings: The set of 667 CTI reports documents 10,370 techniques in total, and we identify 19 prevalent techniques accounting for 37.3\% of documented techniques. We also identify 425 statistically significant recurring pairs and seven types of relations among the techniques in these pairs. The top three among the seven relationships suggest that techniques used by the malware inter-relate with one another in terms of (a) abusing or affecting the same system assets, (b) executing in sequences, and (c) overlapping in their implementations. Overall, the study quantifies how adversaries leverage techniques through malware in APT attacks based on publicly reported documents. We advocate organizations prioritize their defense against the identified prevalent techniques and actively hunt for potential malicious intrusion based on the identified pairs of techniques.



## **3. Towards Robust Semantic Segmentation against Patch-based Attack via Attention Refinement**

cs.CV

30 pages, 3 figures, 12 tables

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.01750v1) [paper-pdf](http://arxiv.org/pdf/2401.01750v1)

**Authors**: Zheng Yuan, Jie Zhang, Yude Wang, Shiguang Shan, Xilin Chen

**Abstract**: The attention mechanism has been proven effective on various visual tasks in recent years. In the semantic segmentation task, the attention mechanism is applied in various methods, including the case of both Convolution Neural Networks (CNN) and Vision Transformer (ViT) as backbones. However, we observe that the attention mechanism is vulnerable to patch-based adversarial attacks. Through the analysis of the effective receptive field, we attribute it to the fact that the wide receptive field brought by global attention may lead to the spread of the adversarial patch. To address this issue, in this paper, we propose a Robust Attention Mechanism (RAM) to improve the robustness of the semantic segmentation model, which can notably relieve the vulnerability against patch-based attacks. Compared to the vallina attention mechanism, RAM introduces two novel modules called Max Attention Suppression and Random Attention Dropout, both of which aim to refine the attention matrix and limit the influence of a single adversarial patch on the semantic segmentation results of other positions. Extensive experiments demonstrate the effectiveness of our RAM to improve the robustness of semantic segmentation models against various patch-based attack methods under different attack settings.



## **4. An Initial Investigation of Neural Replay Simulator for Over-the-Air Adversarial Perturbations to Automatic Speaker Verification**

cs.SD

Accepted in ICASSP 2024

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2310.05354v4) [paper-pdf](http://arxiv.org/pdf/2310.05354v4)

**Authors**: Jiaqi Li, Li Wang, Liumeng Xue, Lei Wang, Zhizheng Wu

**Abstract**: Deep Learning has advanced Automatic Speaker Verification (ASV) in the past few years. Although it is known that deep learning-based ASV systems are vulnerable to adversarial examples in digital access, there are few studies on adversarial attacks in the context of physical access, where a replay process (i.e., over the air) is involved. An over-the-air attack involves a loudspeaker, a microphone, and a replaying environment that impacts the movement of the sound wave. Our initial experiment confirms that the replay process impacts the effectiveness of the over-the-air attack performance. This study performs an initial investigation towards utilizing a neural replay simulator to improve over-the-air adversarial attack robustness. This is achieved by using a neural waveform synthesizer to simulate the replay process when estimating the adversarial perturbations. Experiments conducted on the ASVspoof2019 dataset confirm that the neural replay simulator can considerably increase the success rates of over-the-air adversarial attacks. This raises the concern for adversarial attacks on speaker verification in physical access applications.



## **5. Will 6G be Semantic Communications? Opportunities and Challenges from Task Oriented and Secure Communications to Integrated Sensing**

cs.NI

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.01531v1) [paper-pdf](http://arxiv.org/pdf/2401.01531v1)

**Authors**: Yalin E. Sagduyu, Tugba Erpek, Aylin Yener, Sennur Ulukus

**Abstract**: This paper explores opportunities and challenges of task (goal)-oriented and semantic communications for next-generation (NextG) communication networks through the integration of multi-task learning. This approach employs deep neural networks representing a dedicated encoder at the transmitter and multiple task-specific decoders at the receiver, collectively trained to handle diverse tasks including semantic information preservation, source input reconstruction, and integrated sensing and communications. To extend the applicability from point-to-point links to multi-receiver settings, we envision the deployment of decoders at various receivers, where decentralized learning addresses the challenges of communication load and privacy concerns, leveraging federated learning techniques that distribute model updates across decentralized nodes. However, the efficacy of this approach is contingent on the robustness of the employed deep learning models. We scrutinize potential vulnerabilities stemming from adversarial attacks during both training and testing phases. These attacks aim to manipulate both the inputs at the encoder at the transmitter and the signals received over the air on the receiver side, highlighting the importance of fortifying semantic communications against potential multi-domain exploits. Overall, the joint and robust design of task-oriented communications, semantic communications, and integrated sensing and communications in a multi-task learning framework emerges as the key enabler for context-aware, resource-efficient, and secure communications ultimately needed in NextG network systems.



## **6. JMA: a General Algorithm to Craft Nearly Optimal Targeted Adversarial Example**

cs.LG

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.01199v1) [paper-pdf](http://arxiv.org/pdf/2401.01199v1)

**Authors**: Benedetta Tondi, Wei Guo, Mauro Barni

**Abstract**: Most of the approaches proposed so far to craft targeted adversarial examples against Deep Learning classifiers are highly suboptimal and typically rely on increasing the likelihood of the target class, thus implicitly focusing on one-hot encoding settings. In this paper, we propose a more general, theoretically sound, targeted attack that resorts to the minimization of a Jacobian-induced MAhalanobis distance (JMA) term, taking into account the effort (in the input space) required to move the latent space representation of the input sample in a given direction. The minimization is solved by exploiting the Wolfe duality theorem, reducing the problem to the solution of a Non-Negative Least Square (NNLS) problem. The proposed algorithm provides an optimal solution to a linearized version of the adversarial example problem originally introduced by Szegedy et al. \cite{szegedy2013intriguing}. The experiments we carried out confirm the generality of the proposed attack which is proven to be effective under a wide variety of output encoding schemes. Noticeably, the JMA attack is also effective in a multi-label classification scenario, being capable to induce a targeted modification of up to half the labels in a complex multilabel classification scenario with 20 labels, a capability that is out of reach of all the attacks proposed so far. As a further advantage, the JMA attack usually requires very few iterations, thus resulting more efficient than existing methods.



## **7. Dual Teacher Knowledge Distillation with Domain Alignment for Face Anti-spoofing**

cs.CV

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.01102v1) [paper-pdf](http://arxiv.org/pdf/2401.01102v1)

**Authors**: Zhe Kong, Wentian Zhang, Tao Wang, Kaihao Zhang, Yuexiang Li, Xiaoying Tang, Wenhan Luo

**Abstract**: Face recognition systems have raised concerns due to their vulnerability to different presentation attacks, and system security has become an increasingly critical concern. Although many face anti-spoofing (FAS) methods perform well in intra-dataset scenarios, their generalization remains a challenge. To address this issue, some methods adopt domain adversarial training (DAT) to extract domain-invariant features. However, the competition between the encoder and the domain discriminator can cause the network to be difficult to train and converge. In this paper, we propose a domain adversarial attack (DAA) method to mitigate the training instability problem by adding perturbations to the input images, which makes them indistinguishable across domains and enables domain alignment. Moreover, since models trained on limited data and types of attacks cannot generalize well to unknown attacks, we propose a dual perceptual and generative knowledge distillation framework for face anti-spoofing that utilizes pre-trained face-related models containing rich face priors. Specifically, we adopt two different face-related models as teachers to transfer knowledge to the target student model. The pre-trained teacher models are not from the task of face anti-spoofing but from perceptual and generative tasks, respectively, which implicitly augment the data. By combining both DAA and dual-teacher knowledge distillation, we develop a dual teacher knowledge distillation with domain alignment framework (DTDA) for face anti-spoofing. The advantage of our proposed method has been verified through extensive ablation studies and comparison with state-of-the-art methods on public datasets across multiple protocols.



## **8. Imperio: Language-Guided Backdoor Attacks for Arbitrary Model Control**

cs.CR

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.01085v1) [paper-pdf](http://arxiv.org/pdf/2401.01085v1)

**Authors**: Ka-Ho Chow, Wenqi Wei, Lei Yu

**Abstract**: Revolutionized by the transformer architecture, natural language processing (NLP) has received unprecedented attention. While advancements in NLP models have led to extensive research into their backdoor vulnerabilities, the potential for these advancements to introduce new backdoor threats remains unexplored. This paper proposes Imperio, which harnesses the language understanding capabilities of NLP models to enrich backdoor attacks. Imperio provides a new model control experience. It empowers the adversary to control the victim model with arbitrary output through language-guided instructions. This is achieved using a language model to fuel a conditional trigger generator, with optimizations designed to extend its language understanding capabilities to backdoor instruction interpretation and execution. Our experiments across three datasets, five attacks, and nine defenses confirm Imperio's effectiveness. It can produce contextually adaptive triggers from text descriptions and control the victim model with desired outputs, even in scenarios not encountered during training. The attack maintains a high success rate across complex datasets without compromising the accuracy of clean inputs and also exhibits resilience against representative defenses. The source code is available at \url{https://khchow.com/Imperio}.



## **9. Safety and Performance, Why Not Both? Bi-Objective Optimized Model Compression against Heterogeneous Attacks Toward AI Software Deployment**

cs.AI

Accepted by IEEE Transactions on Software Engineering (TSE).  Camera-ready Version. arXiv admin note: substantial text overlap with  arXiv:2208.05969

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.00996v1) [paper-pdf](http://arxiv.org/pdf/2401.00996v1)

**Authors**: Jie Zhu, Leye Wang, Xiao Han, Anmin Liu, Tao Xie

**Abstract**: The size of deep learning models in artificial intelligence (AI) software is increasing rapidly, hindering the large-scale deployment on resource-restricted devices (e.g., smartphones). To mitigate this issue, AI software compression plays a crucial role, which aims to compress model size while keeping high performance. However, the intrinsic defects in a big model may be inherited by the compressed one. Such defects may be easily leveraged by adversaries, since a compressed model is usually deployed in a large number of devices without adequate protection. In this article, we aim to address the safe model compression problem from the perspective of safety-performance co-optimization. Specifically, inspired by the test-driven development (TDD) paradigm in software engineering, we propose a test-driven sparse training framework called SafeCompress. By simulating the attack mechanism as safety testing, SafeCompress can automatically compress a big model to a small one following the dynamic sparse training paradigm. Then, considering two kinds of representative and heterogeneous attack mechanisms, i.e., black-box membership inference attack and white-box membership inference attack, we develop two concrete instances called BMIA-SafeCompress and WMIA-SafeCompress. Further, we implement another instance called MMIA-SafeCompress by extending SafeCompress to defend against the occasion when adversaries conduct black-box and white-box membership inference attacks simultaneously. We conduct extensive experiments on five datasets for both computer vision and natural language processing tasks. The results show the effectiveness and generalizability of our framework. We also discuss how to adapt SafeCompress to other attacks besides membership inference attack, demonstrating the flexibility of SafeCompress.



## **10. Adversarial Item Promotion on Visually-Aware Recommender Systems by Guided Diffusion**

cs.IR

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2312.15826v3) [paper-pdf](http://arxiv.org/pdf/2312.15826v3)

**Authors**: Lijian Chen, Wei Yuan, Tong Chen, Guanhua Ye, Quoc Viet Hung Nguyen, Hongzhi Yin

**Abstract**: Visually-aware recommender systems have found widespread application in domains where visual elements significantly contribute to the inference of users' potential preferences. While the incorporation of visual information holds the promise of enhancing recommendation accuracy and alleviating the cold-start problem, it is essential to point out that the inclusion of item images may introduce substantial security challenges. Some existing works have shown that the item provider can manipulate item exposure rates to its advantage by constructing adversarial images. However, these works cannot reveal the real vulnerability of visually-aware recommender systems because (1) The generated adversarial images are markedly distorted, rendering them easily detectable by human observers; (2) The effectiveness of the attacks is inconsistent and even ineffective in some scenarios. To shed light on the real vulnerabilities of visually-aware recommender systems when confronted with adversarial images, this paper introduces a novel attack method, IPDGI (Item Promotion by Diffusion Generated Image). Specifically, IPDGI employs a guided diffusion model to generate adversarial samples designed to deceive visually-aware recommender systems. Taking advantage of accurately modeling benign images' distribution by diffusion models, the generated adversarial images have high fidelity with original images, ensuring the stealth of our IPDGI. To demonstrate the effectiveness of our proposed methods, we conduct extensive experiments on two commonly used e-commerce recommendation datasets (Amazon Beauty and Amazon Baby) with several typical visually-aware recommender systems. The experimental results show that our attack method has a significant improvement in both the performance of promoting the long-tailed (i.e., unpopular) items and the quality of generated adversarial images.



## **11. Passive Inference Attacks on Split Learning via Adversarial Regularization**

cs.CR

17 pages, 20 figures

**SubmitDate**: 2024-01-01    [abs](http://arxiv.org/abs/2310.10483v3) [paper-pdf](http://arxiv.org/pdf/2310.10483v3)

**Authors**: Xiaochen Zhu, Xinjian Luo, Yuncheng Wu, Yangfan Jiang, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Split Learning (SL) has emerged as a practical and efficient alternative to traditional federated learning. While previous attempts to attack SL have often relied on overly strong assumptions or targeted easily exploitable models, we seek to develop more practical attacks. We introduce SDAR, a novel attack framework against SL with an honest-but-curious server. SDAR leverages auxiliary data and adversarial regularization to learn a decodable simulator of the client's private model, which can effectively infer the client's private features under the vanilla SL, and both features and labels under the U-shaped SL. We perform extensive experiments in both configurations to validate the effectiveness of our proposed attacks. Notably, in challenging but practical scenarios where existing passive attacks struggle to reconstruct the client's private data effectively, SDAR consistently achieves attack performance comparable to active attacks. On CIFAR-10, at the deep split level of 7, SDAR achieves private feature reconstruction with less than 0.025 mean squared error in both the vanilla and the U-shaped SL, and attains a label inference accuracy of over 98% in the U-shaped setting, while existing attacks fail to produce non-trivial results.



## **12. Channel Reciprocity Attacks Using Intelligent Surfaces with Non-Diagonal Phase Shifts**

eess.SP

**SubmitDate**: 2024-01-01    [abs](http://arxiv.org/abs/2309.11665v2) [paper-pdf](http://arxiv.org/pdf/2309.11665v2)

**Authors**: Haoyu Wang, Zhu Han, A. Lee Swindlehurst

**Abstract**: While reconfigurable intelligent surface (RIS) technology has been shown to provide numerous benefits to wireless systems, in the hands of an adversary such technology can also be used to disrupt communication links. This paper describes and analyzes an RIS-based attack on multi-antenna wireless systems that operate in time-division duplex mode under the assumption of channel reciprocity. In particular, we show how an RIS with a non-diagonal (ND) phase shift matrix (referred to here as an ND-RIS) can be deployed to maliciously break the channel reciprocity and hence degrade the downlink network performance. Such an attack is entirely passive and difficult to detect and counteract. We provide a theoretical analysis of the degradation in the sum ergodic rate that results when an arbitrary malicious ND-RIS is deployed and design an approach based on the genetic algorithm for optimizing the ND structure under partial knowledge of the available channel state information. Our simulation results validate the analysis and demonstrate that an ND-RIS channel reciprocity attack can dramatically reduce the downlink throughput.



## **13. Is It Possible to Backdoor Face Forgery Detection with Natural Triggers?**

cs.CV

**SubmitDate**: 2023-12-31    [abs](http://arxiv.org/abs/2401.00414v1) [paper-pdf](http://arxiv.org/pdf/2401.00414v1)

**Authors**: Xiaoxuan Han, Songlin Yang, Wei Wang, Ziwen He, Jing Dong

**Abstract**: Deep neural networks have significantly improved the performance of face forgery detection models in discriminating Artificial Intelligent Generated Content (AIGC). However, their security is significantly threatened by the injection of triggers during model training (i.e., backdoor attacks). Although existing backdoor defenses and manual data selection can mitigate those using human-eye-sensitive triggers, such as patches or adversarial noises, the more challenging natural backdoor triggers remain insufficiently researched. To further investigate natural triggers, we propose a novel analysis-by-synthesis backdoor attack against face forgery detection models, which embeds natural triggers in the latent space. We thoroughly study such backdoor vulnerability from two perspectives: (1) Model Discrimination (Optimization-Based Trigger): we adopt a substitute detection model and find the trigger by minimizing the cross-entropy loss; (2) Data Distribution (Custom Trigger): we manipulate the uncommon facial attributes in the long-tailed distribution to generate poisoned samples without the supervision from detection models. Furthermore, to completely evaluate the detection models towards the latest AIGC, we utilize both state-of-the-art StyleGAN and Stable Diffusion for trigger generation. Finally, these backdoor triggers introduce specific semantic features to the generated poisoned samples (e.g., skin textures and smile), which are more natural and robust. Extensive experiments show that our method is superior from three levels: (1) Attack Success Rate: ours achieves a high attack success rate (over 99%) and incurs a small model accuracy drop (below 0.2%) with a low poisoning rate (less than 3%); (2) Backdoor Defense: ours shows better robust performance when faced with existing backdoor defense methods; (3) Human Inspection: ours is less human-eye-sensitive from a comprehensive user study.



## **14. Dictionary Attack on IMU-based Gait Authentication**

cs.CR

12 pages, 9 figures, accepted at AISec23 colocated with ACM CCS,  November 30, 2023, Copenhagen, Denmark

**SubmitDate**: 2023-12-31    [abs](http://arxiv.org/abs/2309.11766v2) [paper-pdf](http://arxiv.org/pdf/2309.11766v2)

**Authors**: Rajesh Kumar, Can Isik, Chilukuri K. Mohan

**Abstract**: We present a novel adversarial model for authentication systems that use gait patterns recorded by the inertial measurement unit (IMU) built into smartphones. The attack idea is inspired by and named after the concept of a dictionary attack on knowledge (PIN or password) based authentication systems. In particular, this work investigates whether it is possible to build a dictionary of IMUGait patterns and use it to launch an attack or find an imitator who can actively reproduce IMUGait patterns that match the target's IMUGait pattern. Nine physically and demographically diverse individuals walked at various levels of four predefined controllable and adaptable gait factors (speed, step length, step width, and thigh-lift), producing 178 unique IMUGait patterns. Each pattern attacked a wide variety of user authentication models. The deeper analysis of error rates (before and after the attack) challenges the belief that authentication systems based on IMUGait patterns are the most difficult to spoof; further research is needed on adversarial models and associated countermeasures.



## **15. Forbidden Facts: An Investigation of Competing Objectives in Llama-2**

cs.LG

Accepted to the ATTRIB and SoLaR workshops at NeurIPS 2023; (v3:  clarified experimental details)

**SubmitDate**: 2023-12-31    [abs](http://arxiv.org/abs/2312.08793v3) [paper-pdf](http://arxiv.org/pdf/2312.08793v3)

**Authors**: Tony T. Wang, Miles Wang, Kaivalya Hariharan, Nir Shavit

**Abstract**: LLMs often face competing pressures (for example helpfulness vs. harmlessness). To understand how models resolve such conflicts, we study Llama-2-chat models on the forbidden fact task. Specifically, we instruct Llama-2 to truthfully complete a factual recall statement while forbidding it from saying the correct answer. This often makes the model give incorrect answers. We decompose Llama-2 into 1000+ components, and rank each one with respect to how useful it is for forbidding the correct answer. We find that in aggregate, around 35 components are enough to reliably implement the full suppression behavior. However, these components are fairly heterogeneous and many operate using faulty heuristics. We discover that one of these heuristics can be exploited via a manually designed adversarial attack which we call The California Attack. Our results highlight some roadblocks standing in the way of being able to successfully interpret advanced ML systems. Project website available at https://forbiddenfacts.github.io .



## **16. Explainability-Driven Leaf Disease Classification using Adversarial Training and Knowledge Distillation**

cs.CV

10 pages, 8 figures, Accepted by ICAART 2024

**SubmitDate**: 2023-12-30    [abs](http://arxiv.org/abs/2401.00334v1) [paper-pdf](http://arxiv.org/pdf/2401.00334v1)

**Authors**: Sebastian-Vasile Echim, Iulian-Marius Tăiatu, Dumitru-Clementin Cercel, Florin Pop

**Abstract**: This work focuses on plant leaf disease classification and explores three crucial aspects: adversarial training, model explainability, and model compression. The models' robustness against adversarial attacks is enhanced through adversarial training, ensuring accurate classification even in the presence of threats. Leveraging explainability techniques, we gain insights into the model's decision-making process, improving trust and transparency. Additionally, we explore model compression techniques to optimize computational efficiency while maintaining classification performance. Through our experiments, we determine that on a benchmark dataset, the robustness can be the price of the classification accuracy with performance reductions of 3%-20% for regular tests and gains of 50%-70% for adversarial attack tests. We also demonstrate that a student model can be 15-25 times more computationally efficient for a slight performance reduction, distilling the knowledge of more complex models.



## **17. Unraveling the Connections between Privacy and Certified Robustness in Federated Learning Against Poisoning Attacks**

cs.CR

ACM CCS 2023

**SubmitDate**: 2023-12-30    [abs](http://arxiv.org/abs/2209.04030v3) [paper-pdf](http://arxiv.org/pdf/2209.04030v3)

**Authors**: Chulin Xie, Yunhui Long, Pin-Yu Chen, Qinbin Li, Arash Nourian, Sanmi Koyejo, Bo Li

**Abstract**: Federated learning (FL) provides an efficient paradigm to jointly train a global model leveraging data from distributed users. As local training data comes from different users who may not be trustworthy, several studies have shown that FL is vulnerable to poisoning attacks. Meanwhile, to protect the privacy of local users, FL is usually trained in a differentially private way (DPFL). Thus, in this paper, we ask: What are the underlying connections between differential privacy and certified robustness in FL against poisoning attacks? Can we leverage the innate privacy property of DPFL to provide certified robustness for FL? Can we further improve the privacy of FL to improve such robustness certification? We first investigate both user-level and instance-level privacy of FL and provide formal privacy analysis to achieve improved instance-level privacy. We then provide two robustness certification criteria: certified prediction and certified attack inefficacy for DPFL on both user and instance levels. Theoretically, we provide the certified robustness of DPFL based on both criteria given a bounded number of adversarial users or instances. Empirically, we conduct extensive experiments to verify our theories under a range of poisoning attacks on different datasets. We find that increasing the level of privacy protection in DPFL results in stronger certified attack inefficacy; however, it does not necessarily lead to a stronger certified prediction. Thus, achieving the optimal certified prediction requires a proper balance between privacy and utility loss.



## **18. Fourier Analysis on Robustness of Graph Convolutional Neural Networks for Skeleton-based Action Recognition**

cs.CV

18 pages, 13 figures

**SubmitDate**: 2023-12-30    [abs](http://arxiv.org/abs/2305.17939v2) [paper-pdf](http://arxiv.org/pdf/2305.17939v2)

**Authors**: Nariki Tanaka, Hiroshi Kera, Kazuhiko Kawamoto

**Abstract**: Using Fourier analysis, we explore the robustness and vulnerability of graph convolutional neural networks (GCNs) for skeleton-based action recognition. We adopt a joint Fourier transform (JFT), a combination of the graph Fourier transform (GFT) and the discrete Fourier transform (DFT), to examine the robustness of adversarially-trained GCNs against adversarial attacks and common corruptions. Experimental results with the NTU RGB+D dataset reveal that adversarial training does not introduce a robustness trade-off between adversarial attacks and low-frequency perturbations, which typically occurs during image classification based on convolutional neural networks. This finding indicates that adversarial training is a practical approach to enhancing robustness against adversarial attacks and common corruptions in skeleton-based action recognition. Furthermore, we find that the Fourier approach cannot explain vulnerability against skeletal part occlusion corruption, which highlights its limitations. These findings extend our understanding of the robustness of GCNs, potentially guiding the development of more robust learning methods for skeleton-based action recognition.



## **19. ReMAV: Reward Modeling of Autonomous Vehicles for Finding Likely Failure Events**

cs.AI

**SubmitDate**: 2023-12-30    [abs](http://arxiv.org/abs/2308.14550v2) [paper-pdf](http://arxiv.org/pdf/2308.14550v2)

**Authors**: Aizaz Sharif, Dusica Marijan

**Abstract**: Autonomous vehicles are advanced driving systems that are well known to be vulnerable to various adversarial attacks, compromising vehicle safety and posing a risk to other road users. Rather than actively training complex adversaries by interacting with the environment, there is a need to first intelligently find and reduce the search space to only those states where autonomous vehicles are found to be less confident. In this paper, we propose a black-box testing framework ReMAV that uses offline trajectories first to analyze the existing behavior of autonomous vehicles and determine appropriate thresholds to find the probability of failure events. To this end, we introduce a three-step methodology which i) uses offline state action pairs of any autonomous vehicle under test, ii) builds an abstract behavior representation using our designed reward modeling technique to analyze states with uncertain driving decisions, and iii) uses a disturbance model for minimal perturbation attacks where the driving decisions are less confident. Our reward modeling technique helps in creating a behavior representation that allows us to highlight regions of likely uncertain behavior even when the standard autonomous vehicle performs well. We perform our experiments in a high-fidelity urban driving environment using three different driving scenarios containing single- and multi-agent interactions. Our experiment shows an increase in 35, 23, 48, and 50% in the occurrences of vehicle collision, road object collision, pedestrian collision, and offroad steering events, respectively by the autonomous vehicle under test, demonstrating a significant increase in failure events. We compare ReMAV with two baselines and show that ReMAV demonstrates significantly better effectiveness in generating failure events compared to the baselines in all evaluation metrics.



## **20. CamPro: Camera-based Anti-Facial Recognition**

cs.CV

Accepted by NDSS Symposium 2024

**SubmitDate**: 2023-12-30    [abs](http://arxiv.org/abs/2401.00151v1) [paper-pdf](http://arxiv.org/pdf/2401.00151v1)

**Authors**: Wenjun Zhu, Yuan Sun, Jiani Liu, Yushi Cheng, Xiaoyu Ji, Wenyuan Xu

**Abstract**: The proliferation of images captured from millions of cameras and the advancement of facial recognition (FR) technology have made the abuse of FR a severe privacy threat. Existing works typically rely on obfuscation, synthesis, or adversarial examples to modify faces in images to achieve anti-facial recognition (AFR). However, the unmodified images captured by camera modules that contain sensitive personally identifiable information (PII) could still be leaked. In this paper, we propose a novel approach, CamPro, to capture inborn AFR images. CamPro enables well-packed commodity camera modules to produce images that contain little PII and yet still contain enough information to support other non-sensitive vision applications, such as person detection. Specifically, CamPro tunes the configuration setup inside the camera image signal processor (ISP), i.e., color correction matrix and gamma correction, to achieve AFR, and designs an image enhancer to keep the image quality for possible human viewers. We implemented and validated CamPro on a proof-of-concept camera, and our experiments demonstrate its effectiveness on ten state-of-the-art black-box FR models. The results show that CamPro images can significantly reduce face identification accuracy to 0.3\% while having little impact on the targeted non-sensitive vision application. Furthermore, we find that CamPro is resilient to adaptive attackers who have re-trained their FR models using images generated by CamPro, even with full knowledge of privacy-preserving ISP parameters.



## **21. TPatch: A Triggered Physical Adversarial Patch**

cs.CR

Appeared in 32nd USENIX Security Symposium (USENIX Security 23)

**SubmitDate**: 2023-12-30    [abs](http://arxiv.org/abs/2401.00148v1) [paper-pdf](http://arxiv.org/pdf/2401.00148v1)

**Authors**: Wenjun Zhu, Xiaoyu Ji, Yushi Cheng, Shibo Zhang, Wenyuan Xu

**Abstract**: Autonomous vehicles increasingly utilize the vision-based perception module to acquire information about driving environments and detect obstacles. Correct detection and classification are important to ensure safe driving decisions. Existing works have demonstrated the feasibility of fooling the perception models such as object detectors and image classifiers with printed adversarial patches. However, most of them are indiscriminately offensive to every passing autonomous vehicle. In this paper, we propose TPatch, a physical adversarial patch triggered by acoustic signals. Unlike other adversarial patches, TPatch remains benign under normal circumstances but can be triggered to launch a hiding, creating or altering attack by a designed distortion introduced by signal injection attacks towards cameras. To avoid the suspicion of human drivers and make the attack practical and robust in the real world, we propose a content-based camouflage method and an attack robustness enhancement method to strengthen it. Evaluations with three object detectors, YOLO V3/V5 and Faster R-CNN, and eight image classifiers demonstrate the effectiveness of TPatch in both the simulation and the real world. We also discuss possible defenses at the sensor, algorithm, and system levels.



## **22. Comparing the robustness of modern no-reference image- and video-quality metrics to adversarial attacks**

cs.CV

**SubmitDate**: 2023-12-29    [abs](http://arxiv.org/abs/2310.06958v3) [paper-pdf](http://arxiv.org/pdf/2310.06958v3)

**Authors**: Anastasia Antsiferova, Khaled Abud, Aleksandr Gushchin, Ekaterina Shumitskaya, Sergey Lavrushkin, Dmitriy Vatolin

**Abstract**: Nowadays neural-network-based image- and video-quality metrics show better performance compared to traditional methods. However, they also became more vulnerable to adversarial attacks that increase metrics' scores without improving visual quality. The existing benchmarks of quality metrics compare their performance in terms of correlation with subjective quality and calculation time. However, the adversarial robustness of image-quality metrics is also an area worth researching. In this paper, we analyse modern metrics' robustness to different adversarial attacks. We adopted adversarial attacks from computer vision tasks and compared attacks' efficiency against 15 no-reference image/video-quality metrics. Some metrics showed high resistance to adversarial attacks which makes their usage in benchmarks safer than vulnerable metrics. The benchmark accepts new metrics submissions for researchers who want to make their metrics more robust to attacks or to find such metrics for their needs. Try our benchmark using pip install robustness-benchmark.



## **23. MVPatch: More Vivid Patch for Adversarial Camouflaged Attacks on Object Detectors in the Physical World**

cs.CR

14 pages, 8 figures, submitted to IEEE Transactions on Information  Forensics & Security

**SubmitDate**: 2023-12-29    [abs](http://arxiv.org/abs/2312.17431v1) [paper-pdf](http://arxiv.org/pdf/2312.17431v1)

**Authors**: Zheng Zhou, Hongbo Zhao, Ju Liu, Qiaosheng Zhang, Guangbiao Wang, Chunlei Wang, Wenquan Feng

**Abstract**: Recent research has shown that adversarial patches can manipulate outputs from object detection models. However, the conspicuous patterns on these patches may draw more attention and raise suspicions among humans. Moreover, existing works have primarily focused on the attack performance of individual models and have neglected the generation of adversarial patches for ensemble attacks on multiple object detection models. To tackle these concerns, we propose a novel approach referred to as the More Vivid Patch (MVPatch), which aims to improve the transferability and stealthiness of adversarial patches while considering the limitations observed in prior paradigms, such as easy identification and poor transferability. Our approach incorporates an attack algorithm that decreases object confidence scores of multiple object detectors by using the ensemble attack loss function, thereby enhancing the transferability of adversarial patches. Additionally, we propose a lightweight visual similarity measurement algorithm realized by the Compared Specified Image Similarity (CSS) loss function, which allows for the generation of natural and stealthy adversarial patches without the reliance on additional generative models. Extensive experiments demonstrate that the proposed MVPatch algorithm achieves superior attack transferability compared to similar algorithms in both digital and physical domains, while also exhibiting a more natural appearance. These findings emphasize the remarkable stealthiness and transferability of the proposed MVPatch attack algorithm.



## **24. Can you See me? On the Visibility of NOPs against Android Malware Detectors**

cs.CR

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.17356v1) [paper-pdf](http://arxiv.org/pdf/2312.17356v1)

**Authors**: Diego Soi, Davide Maiorca, Giorgio Giacinto, Harel Berger

**Abstract**: Android malware still represents the most significant threat to mobile systems. While Machine Learning systems are increasingly used to identify these threats, past studies have revealed that attackers can bypass these detection mechanisms by making subtle changes to Android applications, such as adding specific API calls. These modifications are often referred to as No OPerations (NOP), which ideally should not alter the semantics of the program. However, many NOPs can be spotted and eliminated by refining the app analysis process. This paper proposes a visibility metric that assesses the difficulty in spotting NOPs and similar non-operational codes. We tested our metric on a state-of-the-art, opcode-based deep learning system for Android malware detection. We implemented attacks on the feature and problem spaces and calculated their visibility according to our metric. The attained results show an intriguing trade-off between evasion efficacy and detectability: our metric can be valuable to ensure the real effectiveness of an adversarial attack, also serving as a useful aid to develop better defenses.



## **25. Timeliness: A New Design Metric and a New Attack Surface**

cs.IT

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.17220v1) [paper-pdf](http://arxiv.org/pdf/2312.17220v1)

**Authors**: Priyanka Kaswan, Sennur Ulukus

**Abstract**: As the landscape of time-sensitive applications gains prominence in 5G/6G communications, timeliness of information updates at network nodes has become crucial, which is popularly quantified in the literature by the age of information metric. However, as we devise policies to improve age of information of our systems, we inadvertently introduce a new vulnerability for adversaries to exploit. In this article, we comprehensively discuss the diverse threats that age-based systems are vulnerable to. We begin with discussion on densely interconnected networks that employ gossiping between nodes to expedite dissemination of dynamic information in the network, and show how the age-based nature of gossiping renders these networks uniquely susceptible to threats such as timestomping attacks, jamming attacks, and the propagation of misinformation. Later, we survey adversarial works within simpler network settings, specifically in one-hop and two-hop configurations, and delve into adversarial robustness concerning challenges posed by jamming, timestomping, and issues related to privacy leakage. We conclude this article with future directions that aim to address challenges posed by more intelligent adversaries and robustness of networks to them.



## **26. Explainability-Based Adversarial Attack on Graphs Through Edge Perturbation**

cs.CR

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.17301v1) [paper-pdf](http://arxiv.org/pdf/2312.17301v1)

**Authors**: Dibaloke Chanda, Saba Heidari Gheshlaghi, Nasim Yahya Soltani

**Abstract**: Despite the success of graph neural networks (GNNs) in various domains, they exhibit susceptibility to adversarial attacks. Understanding these vulnerabilities is crucial for developing robust and secure applications. In this paper, we investigate the impact of test time adversarial attacks through edge perturbations which involve both edge insertions and deletions. A novel explainability-based method is proposed to identify important nodes in the graph and perform edge perturbation between these nodes. The proposed method is tested for node classification with three different architectures and datasets. The results suggest that introducing edges between nodes of different classes has higher impact as compared to removing edges among nodes within the same class.



## **27. On the Robustness of Decision-Focused Learning**

cs.LG

17 pages, 45 figures, submitted to AAAI artificial intelligence for  operations research workshop

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2311.16487v3) [paper-pdf](http://arxiv.org/pdf/2311.16487v3)

**Authors**: Yehya Farhat

**Abstract**: Decision-Focused Learning (DFL) is an emerging learning paradigm that tackles the task of training a machine learning (ML) model to predict missing parameters of an incomplete optimization problem, where the missing parameters are predicted. DFL trains an ML model in an end-to-end system, by integrating the prediction and optimization tasks, providing better alignment of the training and testing objectives. DFL has shown a lot of promise and holds the capacity to revolutionize decision-making in many real-world applications. However, very little is known about the performance of these models under adversarial attacks. We adopt ten unique DFL methods and benchmark their performance under two distinctly focused attacks adapted towards the Predict-then-Optimize problem setting. Our study proposes the hypothesis that the robustness of a model is highly correlated with its ability to find predictions that lead to optimal decisions without deviating from the ground-truth label. Furthermore, we provide insight into how to target the models that violate this condition and show how these models respond differently depending on the achieved optimality at the end of their training cycles.



## **28. BlackboxBench: A Comprehensive Benchmark of Black-box Adversarial Attacks**

cs.CR

37 pages, 29 figures

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.16979v1) [paper-pdf](http://arxiv.org/pdf/2312.16979v1)

**Authors**: Meixi Zheng, Xuanchen Yan, Zihao Zhu, Hongrui Chen, Baoyuan Wu

**Abstract**: Adversarial examples are well-known tools to evaluate the vulnerability of deep neural networks (DNNs). Although lots of adversarial attack algorithms have been developed, it is still challenging in the practical scenario that the model's parameters and architectures are inaccessible to the attacker/evaluator, i.e., black-box adversarial attacks. Due to the practical importance, there has been rapid progress from recent algorithms, reflected by the quick increase in attack success rate and the quick decrease in query numbers to the target model. However, there is a lack of thorough evaluations and comparisons among these algorithms, causing difficulties of tracking the real progress, analyzing advantages and disadvantages of different technical routes, as well as designing future development roadmap of this field. Thus, in this work, we aim at building a comprehensive benchmark of black-box adversarial attacks, called BlackboxBench. It mainly provides: 1) a unified, extensible and modular-based codebase, implementing 25 query-based attack algorithms and 30 transfer-based attack algorithms; 2) comprehensive evaluations: we evaluate the implemented algorithms against several mainstreaming model architectures on 2 widely used datasets (CIFAR-10 and a subset of ImageNet), leading to 14,106 evaluations in total; 3) thorough analysis and new insights, as well analytical tools. The website and source codes of BlackboxBench are available at https://blackboxbench.github.io/ and https://github.com/SCLBD/BlackboxBench/, respectively.



## **29. Attack Tree Analysis for Adversarial Evasion Attacks**

cs.CR

10 pages

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.16957v1) [paper-pdf](http://arxiv.org/pdf/2312.16957v1)

**Authors**: Yuki Yamaguchi, Toshiaki Aoki

**Abstract**: Recently, the evolution of deep learning has promoted the application of machine learning (ML) to various systems. However, there are ML systems, such as autonomous vehicles, that cause critical damage when they misclassify. Conversely, there are ML-specific attacks called adversarial attacks based on the characteristics of ML systems. For example, one type of adversarial attack is an evasion attack, which uses minute perturbations called "adversarial examples" to intentionally misclassify classifiers. Therefore, it is necessary to analyze the risk of ML-specific attacks in introducing ML base systems. In this study, we propose a quantitative evaluation method for analyzing the risk of evasion attacks using attack trees. The proposed method consists of the extension of the conventional attack tree to analyze evasion attacks and the systematic construction method of the extension. In the extension of the conventional attack tree, we introduce ML and conventional attack nodes to represent various characteristics of evasion attacks. In the systematic construction process, we propose a procedure to construct the attack tree. The procedure consists of three steps: (1) organizing information about attack methods in the literature to a matrix, (2) identifying evasion attack scenarios from methods in the matrix, and (3) constructing the attack tree from the identified scenarios using a pattern. Finally, we conducted experiments on three ML image recognition systems to demonstrate the versatility and effectiveness of our proposed method.



## **30. DOEPatch: Dynamically Optimized Ensemble Model for Adversarial Patches Generation**

cs.CV

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.16907v1) [paper-pdf](http://arxiv.org/pdf/2312.16907v1)

**Authors**: Wenyi Tan, Yang Li, Chenxing Zhao, Zhunga Liu, Quan Pan

**Abstract**: Object detection is a fundamental task in various applications ranging from autonomous driving to intelligent security systems. However, recognition of a person can be hindered when their clothing is decorated with carefully designed graffiti patterns, leading to the failure of object detection. To achieve greater attack potential against unknown black-box models, adversarial patches capable of affecting the outputs of multiple-object detection models are required. While ensemble models have proven effective, current research in the field of object detection typically focuses on the simple fusion of the outputs of all models, with limited attention being given to developing general adversarial patches that can function effectively in the physical world. In this paper, we introduce the concept of energy and treat the adversarial patches generation process as an optimization of the adversarial patches to minimize the total energy of the ``person'' category. Additionally, by adopting adversarial training, we construct a dynamically optimized ensemble model. During training, the weight parameters of the attacked target models are adjusted to find the balance point at which the generated adversarial patches can effectively attack all target models. We carried out six sets of comparative experiments and tested our algorithm on five mainstream object detection models. The adversarial patches generated by our algorithm can reduce the recognition accuracy of YOLOv2 and YOLOv3 to 13.19\% and 29.20\%, respectively. In addition, we conducted experiments to test the effectiveness of T-shirts covered with our adversarial patches in the physical world and could achieve that people are not recognized by the object detection model. Finally, leveraging the Grad-CAM tool, we explored the attack mechanism of adversarial patches from an energetic perspective.



## **31. Adversarial Attacks on Image Classification Models: Analysis and Defense**

cs.CV

This is the accepted version of the paper presented at the 10th  International Conference on Business Analytics and Intelligence (ICBAI'24).  The conference was organized by the Indian Institute of Science, Bangalore,  India, from December 18 - 20, 2023. The paper is 10 pages long and it  contains 14 tables and 11 figures

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.16880v1) [paper-pdf](http://arxiv.org/pdf/2312.16880v1)

**Authors**: Jaydip Sen, Abhiraj Sen, Ananda Chatterjee

**Abstract**: The notion of adversarial attacks on image classification models based on convolutional neural networks (CNN) is introduced in this work. To classify images, deep learning models called CNNs are frequently used. However, when the networks are subject to adversarial attacks, extremely potent and previously trained CNN models that perform quite effectively on image datasets for image classification tasks may perform poorly. In this work, one well-known adversarial attack known as the fast gradient sign method (FGSM) is explored and its adverse effects on the performances of image classification models are examined. The FGSM attack is simulated on three pre-trained image classifier CNN architectures, ResNet-101, AlexNet, and RegNetY 400MF using randomly chosen images from the ImageNet dataset. The classification accuracies of the models are computed in the absence and presence of the attack to demonstrate the detrimental effect of the attack on the performances of the classifiers. Finally, a mechanism is proposed to defend against the FGSM attack based on a modified defensive distillation-based approach. Extensive results are presented for the validation of the proposed scheme.



## **32. Adv-Diffusion: Imperceptible Adversarial Face Identity Attack via Latent Diffusion Model**

cs.CV

Accepted by AAAI 2024

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.11285v2) [paper-pdf](http://arxiv.org/pdf/2312.11285v2)

**Authors**: Decheng Liu, Xijun Wang, Chunlei Peng, Nannan Wang, Ruiming Hu, Xinbo Gao

**Abstract**: Adversarial attacks involve adding perturbations to the source image to cause misclassification by the target model, which demonstrates the potential of attacking face recognition models. Existing adversarial face image generation methods still can't achieve satisfactory performance because of low transferability and high detectability. In this paper, we propose a unified framework Adv-Diffusion that can generate imperceptible adversarial identity perturbations in the latent space but not the raw pixel space, which utilizes strong inpainting capabilities of the latent diffusion model to generate realistic adversarial images. Specifically, we propose the identity-sensitive conditioned diffusion generative model to generate semantic perturbations in the surroundings. The designed adaptive strength-based adversarial perturbation algorithm can ensure both attack transferability and stealthiness. Extensive qualitative and quantitative experiments on the public FFHQ and CelebA-HQ datasets prove the proposed method achieves superior performance compared with the state-of-the-art methods without an extra generative model training process. The source code is available at https://github.com/kopper-xdu/Adv-Diffusion.



## **33. Temporal Knowledge Distillation for Time-Sensitive Financial Services Applications**

cs.LG

arXiv admin note: text overlap with arXiv:2101.01689

**SubmitDate**: 2023-12-28    [abs](http://arxiv.org/abs/2312.16799v1) [paper-pdf](http://arxiv.org/pdf/2312.16799v1)

**Authors**: Hongda Shen, Eren Kurshan

**Abstract**: Detecting anomalies has become an increasingly critical function in the financial service industry. Anomaly detection is frequently used in key compliance and risk functions such as financial crime detection fraud and cybersecurity. The dynamic nature of the underlying data patterns especially in adversarial environments like fraud detection poses serious challenges to the machine learning models. Keeping up with the rapid changes by retraining the models with the latest data patterns introduces pressures in balancing the historical and current patterns while managing the training data size. Furthermore the model retraining times raise problems in time-sensitive and high-volume deployment systems where the retraining period directly impacts the models ability to respond to ongoing attacks in a timely manner. In this study we propose a temporal knowledge distillation-based label augmentation approach (TKD) which utilizes the learning from older models to rapidly boost the latest model and effectively reduces the model retraining times to achieve improved agility. Experimental results show that the proposed approach provides advantages in retraining times while improving the model performance.



## **34. Multi-Task Models Adversarial Attacks**

cs.LG

19 pages, 6 figures

**SubmitDate**: 2023-12-27    [abs](http://arxiv.org/abs/2305.12066v3) [paper-pdf](http://arxiv.org/pdf/2305.12066v3)

**Authors**: Lijun Zhang, Xiao Liu, Kaleel Mahmood, Caiwen Ding, Hui Guan

**Abstract**: Multi-Task Learning (MTL) involves developing a singular model, known as a multi-task model, to concurrently perform multiple tasks. While the security of single-task models has been thoroughly studied, multi-task models pose several critical security questions, such as 1) their vulnerability to single-task adversarial attacks, 2) the possibility of designing attacks that target multiple tasks, and 3) the impact of task sharing and adversarial training on their resilience to such attacks. This paper addresses these queries through detailed analysis and rigorous experimentation. First, we explore the adaptation of single-task white-box attacks to multi-task models and identify their limitations. We then introduce a novel attack framework, the Gradient Balancing Multi-Task Attack (GB-MTA), which treats attacking a multi-task model as an optimization problem. This problem, based on averaged relative loss change across tasks, is approximated as an integer linear programming problem. Extensive evaluations on MTL benchmarks, NYUv2 and Tiny-Taxonomy, demonstrate GB-MTA's effectiveness against both standard and adversarially trained multi-task models. The results also highlight a trade-off between task accuracy improvement via parameter sharing and increased model vulnerability due to enhanced attack transferability.



## **35. Adversarial Attacks on LoRa Device Identification and Rogue Signal Detection with Deep Learning**

cs.CR

**SubmitDate**: 2023-12-27    [abs](http://arxiv.org/abs/2312.16715v1) [paper-pdf](http://arxiv.org/pdf/2312.16715v1)

**Authors**: Yalin E. Sagduyu, Tugba Erpek

**Abstract**: Low-Power Wide-Area Network (LPWAN) technologies, such as LoRa, have gained significant attention for their ability to enable long-range, low-power communication for Internet of Things (IoT) applications. However, the security of LoRa networks remains a major concern, particularly in scenarios where device identification and classification of legitimate and spoofed signals are crucial. This paper studies a deep learning framework to address these challenges, considering LoRa device identification and legitimate vs. rogue LoRa device classification tasks. A deep neural network (DNN), either a convolutional neural network (CNN) or feedforward neural network (FNN), is trained for each task by utilizing real experimental I/Q data for LoRa signals, while rogue signals are generated by using kernel density estimation (KDE) of received signals by rogue devices. Fast Gradient Sign Method (FGSM)-based adversarial attacks are considered for LoRa signal classification tasks using deep learning models. The impact of these attacks is assessed on the performance of two tasks, namely device identification and legitimate vs. rogue device classification, by utilizing separate or common perturbations against these signal classification tasks. Results presented in this paper quantify the level of transferability of adversarial attacks on different LoRa signal classification tasks as a major vulnerability and highlight the need to make IoT applications robust to adversarial attacks.



## **36. Frauds Bargain Attack: Generating Adversarial Text Samples via Word Manipulation Process**

cs.CL

21 pages, 9 tables, 3 figures

**SubmitDate**: 2023-12-27    [abs](http://arxiv.org/abs/2303.01234v2) [paper-pdf](http://arxiv.org/pdf/2303.01234v2)

**Authors**: Mingze Ni, Zhensu Sun, Wei Liu

**Abstract**: Recent research has revealed that natural language processing (NLP) models are vulnerable to adversarial examples. However, the current techniques for generating such examples rely on deterministic heuristic rules, which fail to produce optimal adversarial examples. In response, this study proposes a new method called the Fraud's Bargain Attack (FBA), which uses a randomization mechanism to expand the search space and produce high-quality adversarial examples with a higher probability of success. FBA uses the Metropolis-Hasting sampler, a type of Markov Chain Monte Carlo sampler, to improve the selection of adversarial examples from all candidates generated by a customized stochastic process called the Word Manipulation Process (WMP). The WMP method modifies individual words in a contextually-aware manner through insertion, removal, or substitution. Through extensive experiments, this study demonstrates that FBA outperforms other methods in terms of attack success rate, imperceptibility and sentence quality.



## **37. Evaluating the security of CRYSTALS-Dilithium in the quantum random oracle model**

cs.CR

21 pages

**SubmitDate**: 2023-12-27    [abs](http://arxiv.org/abs/2312.16619v1) [paper-pdf](http://arxiv.org/pdf/2312.16619v1)

**Authors**: Kelsey A. Jackson, Carl A. Miller, Daochen Wang

**Abstract**: In the wake of recent progress on quantum computing hardware, the National Institute of Standards and Technology (NIST) is standardizing cryptographic protocols that are resistant to attacks by quantum adversaries. The primary digital signature scheme that NIST has chosen is CRYSTALS-Dilithium. The hardness of this scheme is based on the hardness of three computational problems: Module Learning with Errors (MLWE), Module Short Integer Solution (MSIS), and SelfTargetMSIS. MLWE and MSIS have been well-studied and are widely believed to be secure. However, SelfTargetMSIS is novel and, though classically as hard as MSIS, its quantum hardness is unclear. In this paper, we provide the first proof of the hardness of SelfTargetMSIS via a reduction from MLWE in the Quantum Random Oracle Model (QROM). Our proof uses recently developed techniques in quantum reprogramming and rewinding. A central part of our approach is a proof that a certain hash function, derived from the MSIS problem, is collapsing. From this approach, we deduce a new security proof for Dilithium under appropriate parameter settings. Compared to the only other rigorous security proof for a variant of Dilithium, Dilithium-QROM, our proof has the advantage of being applicable under the condition q = 1 mod 2n, where q denotes the modulus and n the dimension of the underlying algebraic ring. This condition is part of the original Dilithium proposal and is crucial for the efficient implementation of the scheme. We provide new secure parameter sets for Dilithium under the condition q = 1 mod 2n, finding that our public key sizes and signature sizes are about 2.5 to 2.8 times larger than those of Dilithium-QROM for the same security levels.



## **38. Natural Adversarial Patch Generation Method Based on Latent Diffusion Model**

cs.CV

**SubmitDate**: 2023-12-27    [abs](http://arxiv.org/abs/2312.16401v1) [paper-pdf](http://arxiv.org/pdf/2312.16401v1)

**Authors**: Xianyi Chen, Fazhan Liu, Dong Jiang, Kai Yan

**Abstract**: Recently, some research show that deep neural networks are vulnerable to the adversarial attacks, the well-trainned samples or patches could be used to trick the neural network detector or human visual perception. However, these adversarial patches, with their conspicuous and unusual patterns, lack camouflage and can easily raise suspicion in the real world. To solve this problem, this paper proposed a novel adversarial patch method called the Latent Diffusion Patch (LDP), in which, a pretrained encoder is first designed to compress the natural images into a feature space with key characteristics. Then trains the diffusion model using the above feature space. Finally, explore the latent space of the pretrained diffusion model using the image denoising technology. It polishes the patches and images through the powerful natural abilities of diffusion models, making them more acceptable to the human visual system. Experimental results, both digital and physical worlds, show that LDPs achieve a visual subjectivity score of 87.3%, while still maintaining effective attack capabilities.



## **39. SlowTrack: Increasing the Latency of Camera-based Perception in Autonomous Driving Using Adversarial Examples**

cs.CV

Accepted by AAAI 2024

**SubmitDate**: 2023-12-26    [abs](http://arxiv.org/abs/2312.09520v2) [paper-pdf](http://arxiv.org/pdf/2312.09520v2)

**Authors**: Chen Ma, Ningfei Wang, Qi Alfred Chen, Chao Shen

**Abstract**: In Autonomous Driving (AD), real-time perception is a critical component responsible for detecting surrounding objects to ensure safe driving. While researchers have extensively explored the integrity of AD perception due to its safety and security implications, the aspect of availability (real-time performance) or latency has received limited attention. Existing works on latency-based attack have focused mainly on object detection, i.e., a component in camera-based AD perception, overlooking the entire camera-based AD perception, which hinders them to achieve effective system-level effects, such as vehicle crashes. In this paper, we propose SlowTrack, a novel framework for generating adversarial attacks to increase the execution time of camera-based AD perception. We propose a novel two-stage attack strategy along with the three new loss function designs. Our evaluation is conducted on four popular camera-based AD perception pipelines, and the results demonstrate that SlowTrack significantly outperforms existing latency-based attacks while maintaining comparable imperceptibility levels. Furthermore, we perform the evaluation on Baidu Apollo, an industry-grade full-stack AD system, and LGSVL, a production-grade AD simulator, with two scenarios to compare the system-level effects of SlowTrack and existing attacks. Our evaluation results show that the system-level effects can be significantly improved, i.e., the vehicle crash rate of SlowTrack is around 95% on average while existing works only have around 30%.



## **40. Model Stealing Attack against Recommender System**

cs.CR

**SubmitDate**: 2023-12-26    [abs](http://arxiv.org/abs/2312.11571v2) [paper-pdf](http://arxiv.org/pdf/2312.11571v2)

**Authors**: Zhihao Zhu, Rui Fan, Chenwang Wu, Yi Yang, Defu Lian, Enhong Chen

**Abstract**: Recent studies have demonstrated the vulnerability of recommender systems to data privacy attacks. However, research on the threat to model privacy in recommender systems, such as model stealing attacks, is still in its infancy. Some adversarial attacks have achieved model stealing attacks against recommender systems, to some extent, by collecting abundant training data of the target model (target data) or making a mass of queries. In this paper, we constrain the volume of available target data and queries and utilize auxiliary data, which shares the item set with the target data, to promote model stealing attacks. Although the target model treats target and auxiliary data differently, their similar behavior patterns allow them to be fused using an attention mechanism to assist attacks. Besides, we design stealing functions to effectively extract the recommendation list obtained by querying the target model. Experimental results show that the proposed methods are applicable to most recommender systems and various scenarios and exhibit excellent attack performance on multiple datasets.



## **41. MENLI: Robust Evaluation Metrics from Natural Language Inference**

cs.CL

TACL 2023 Camera-ready version; updated after proofreading by the  journal

**SubmitDate**: 2023-12-26    [abs](http://arxiv.org/abs/2208.07316v5) [paper-pdf](http://arxiv.org/pdf/2208.07316v5)

**Authors**: Yanran Chen, Steffen Eger

**Abstract**: Recently proposed BERT-based evaluation metrics for text generation perform well on standard benchmarks but are vulnerable to adversarial attacks, e.g., relating to information correctness. We argue that this stems (in part) from the fact that they are models of semantic similarity. In contrast, we develop evaluation metrics based on Natural Language Inference (NLI), which we deem a more appropriate modeling. We design a preference-based adversarial attack framework and show that our NLI based metrics are much more robust to the attacks than the recent BERT-based metrics. On standard benchmarks, our NLI based metrics outperform existing summarization metrics, but perform below SOTA MT metrics. However, when combining existing metrics with our NLI metrics, we obtain both higher adversarial robustness (15%-30%) and higher quality metrics as measured on standard benchmarks (+5% to 30%).



## **42. Punctuation Matters! Stealthy Backdoor Attack for Language Models**

cs.CL

NLPCC 2023

**SubmitDate**: 2023-12-26    [abs](http://arxiv.org/abs/2312.15867v1) [paper-pdf](http://arxiv.org/pdf/2312.15867v1)

**Authors**: Xuan Sheng, Zhicheng Li, Zhaoyang Han, Xiangmao Chang, Piji Li

**Abstract**: Recent studies have pointed out that natural language processing (NLP) models are vulnerable to backdoor attacks. A backdoored model produces normal outputs on the clean samples while performing improperly on the texts with triggers that the adversary injects. However, previous studies on textual backdoor attack pay little attention to stealthiness. Moreover, some attack methods even cause grammatical issues or change the semantic meaning of the original texts. Therefore, they can easily be detected by humans or defense systems. In this paper, we propose a novel stealthy backdoor attack method against textual models, which is called \textbf{PuncAttack}. It leverages combinations of punctuation marks as the trigger and chooses proper locations strategically to replace them. Through extensive experiments, we demonstrate that the proposed method can effectively compromise multiple models in various tasks. Meanwhile, we conduct automatic evaluation and human inspection, which indicate the proposed method possesses good performance of stealthiness without bringing grammatical issues and altering the meaning of sentences.



## **43. Attention Deficit is Ordered! Fooling Deformable Vision Transformers with Collaborative Adversarial Patches**

cs.CV

12 pages, 14 figures

**SubmitDate**: 2023-12-25    [abs](http://arxiv.org/abs/2311.12914v2) [paper-pdf](http://arxiv.org/pdf/2311.12914v2)

**Authors**: Quazi Mishkatul Alam, Bilel Tarchoun, Ihsen Alouani, Nael Abu-Ghazaleh

**Abstract**: The latest generation of transformer-based vision models has proven to be superior to Convolutional Neural Network (CNN)-based models across several vision tasks, largely attributed to their remarkable prowess in relation modeling. Deformable vision transformers significantly reduce the quadratic complexity of attention modeling by using sparse attention structures, enabling them to incorporate features across different scales and be used in large-scale applications, such as multi-view vision systems. Recent work has demonstrated adversarial attacks against conventional vision transformers; we show that these attacks do not transfer to deformable transformers due to their sparse attention structure. Specifically, attention in deformable transformers is modeled using pointers to the most relevant other tokens. In this work, we contribute for the first time adversarial attacks that manipulate the attention of deformable transformers, redirecting it to focus on irrelevant parts of the image. We also develop new collaborative attacks where a source patch manipulates attention to point to a target patch, which contains the adversarial noise to fool the model. In our experiments, we observe that altering less than 1% of the patched area in the input field results in a complete drop to 0% AP in single-view object detection using MS COCO and a 0% MODA in multi-view object detection using Wildtrack.



## **44. Adversarial Prompt Tuning for Vision-Language Models**

cs.CV

**SubmitDate**: 2023-12-25    [abs](http://arxiv.org/abs/2311.11261v2) [paper-pdf](http://arxiv.org/pdf/2311.11261v2)

**Authors**: Jiaming Zhang, Xingjun Ma, Xin Wang, Lingyu Qiu, Jiaqi Wang, Yu-Gang Jiang, Jitao Sang

**Abstract**: With the rapid advancement of multimodal learning, pre-trained Vision-Language Models (VLMs) such as CLIP have demonstrated remarkable capacities in bridging the gap between visual and language modalities. However, these models remain vulnerable to adversarial attacks, particularly in the image modality, presenting considerable security risks. This paper introduces Adversarial Prompt Tuning (AdvPT), a novel technique to enhance the adversarial robustness of image encoders in VLMs. AdvPT innovatively leverages learnable text prompts and aligns them with adversarial image embeddings, to address the vulnerabilities inherent in VLMs without the need for extensive parameter training or modification of the model architecture. We demonstrate that AdvPT improves resistance against white-box and black-box adversarial attacks and exhibits a synergistic effect when combined with existing image-processing-based defense techniques, further boosting defensive capabilities. Comprehensive experimental analyses provide insights into adversarial prompt tuning, a novel paradigm devoted to improving resistance to adversarial images through textual input modifications, paving the way for future robust multimodal learning research. These findings open up new possibilities for enhancing the security of VLMs. Our code is available at https://github.com/jiamingzhang94/Adversarial-Prompt-Tuning.



## **45. Vulnerability of Machine Learning Approaches Applied in IoT-based Smart Grid: A Review**

cs.CR

**SubmitDate**: 2023-12-25    [abs](http://arxiv.org/abs/2308.15736v3) [paper-pdf](http://arxiv.org/pdf/2308.15736v3)

**Authors**: Zhenyong Zhang, Mengxiang Liu, Mingyang Sun, Ruilong Deng, Peng Cheng, Dusit Niyato, Mo-Yuen Chow, Jiming Chen

**Abstract**: Machine learning (ML) sees an increasing prevalence of being used in the internet-of-things (IoT)-based smart grid. However, the trustworthiness of ML is a severe issue that must be addressed to accommodate the trend of ML-based smart grid applications (MLsgAPPs). The adversarial distortion injected into the power signal will greatly affect the system's normal control and operation. Therefore, it is imperative to conduct vulnerability assessment for MLsgAPPs applied in the context of safety-critical power systems. In this paper, we provide a comprehensive review of the recent progress in designing attack and defense methods for MLsgAPPs. Unlike the traditional survey about ML security, this is the first review work about the security of MLsgAPPs that focuses on the characteristics of power systems. We first highlight the specifics for constructing the adversarial attacks on MLsgAPPs. Then, the vulnerability of MLsgAPP is analyzed from both the aspects of the power system and ML model. Afterward, a comprehensive survey is conducted to review and compare existing studies about the adversarial attacks on MLsgAPPs in scenarios of generation, transmission, distribution, and consumption, and the countermeasures are reviewed according to the attacks that they defend against. Finally, the future research directions are discussed on the attacker's and defender's side, respectively. We also analyze the potential vulnerability of large language model-based (e.g., ChatGPT) power system applications. Overall, we encourage more researchers to contribute to investigating the adversarial issues of MLsgAPPs.



## **46. Privacy-Preserving Neural Graph Databases**

cs.DB

**SubmitDate**: 2023-12-25    [abs](http://arxiv.org/abs/2312.15591v1) [paper-pdf](http://arxiv.org/pdf/2312.15591v1)

**Authors**: Qi Hu, Haoran Li, Jiaxin Bai, Yangqiu Song

**Abstract**: In the era of big data and rapidly evolving information systems, efficient and accurate data retrieval has become increasingly crucial. Neural graph databases (NGDBs) have emerged as a powerful paradigm that combines the strengths of graph databases (graph DBs) and neural networks to enable efficient storage, retrieval, and analysis of graph-structured data. The usage of neural embedding storage and complex neural logical query answering provides NGDBs with generalization ability. When the graph is incomplete, by extracting latent patterns and representations, neural graph databases can fill gaps in the graph structure, revealing hidden relationships and enabling accurate query answering. Nevertheless, this capability comes with inherent trade-offs, as it introduces additional privacy risks to the database. Malicious attackers can infer more sensitive information in the database using well-designed combinatorial queries, such as by comparing the answer sets of where Turing Award winners born before 1950 and after 1940 lived, the living places of Turing Award winner Hinton are probably exposed, although the living places may have been deleted in the training due to the privacy concerns. In this work, inspired by the privacy protection in graph embeddings, we propose a privacy-preserving neural graph database (P-NGDB) to alleviate the risks of privacy leakage in NGDBs. We introduce adversarial training techniques in the training stage to force the NGDBs to generate indistinguishable answers when queried with private information, enhancing the difficulty of inferring sensitive information through combinations of multiple innocuous queries. Extensive experiment results on three datasets show that P-NGDB can effectively protect private information in the graph database while delivering high-quality public answers responses to queries.



## **47. Adversarial Data Poisoning for Fake News Detection: How to Make a Model Misclassify a Target News without Modifying It**

cs.LG

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2312.15228v2) [paper-pdf](http://arxiv.org/pdf/2312.15228v2)

**Authors**: Federico Siciliano, Luca Maiano, Lorenzo Papa, Federica Baccini, Irene Amerini, Fabrizio Silvestri

**Abstract**: Fake news detection models are critical to countering disinformation but can be manipulated through adversarial attacks. In this position paper, we analyze how an attacker can compromise the performance of an online learning detector on specific news content without being able to manipulate the original target news. In some contexts, such as social networks, where the attacker cannot exert complete control over all the information, this scenario can indeed be quite plausible. Therefore, we show how an attacker could potentially introduce poisoning data into the training data to manipulate the behavior of an online learning method. Our initial findings reveal varying susceptibility of logistic regression models based on complexity and attack type.



## **48. Towards Transferable Adversarial Attacks with Centralized Perturbation**

cs.CV

10 pages, 9 figures, accepted by AAAI 2024

**SubmitDate**: 2023-12-23    [abs](http://arxiv.org/abs/2312.06199v2) [paper-pdf](http://arxiv.org/pdf/2312.06199v2)

**Authors**: Shangbo Wu, Yu-an Tan, Yajie Wang, Ruinan Ma, Wencong Ma, Yuanzhang Li

**Abstract**: Adversarial transferability enables black-box attacks on unknown victim deep neural networks (DNNs), rendering attacks viable in real-world scenarios. Current transferable attacks create adversarial perturbation over the entire image, resulting in excessive noise that overfit the source model. Concentrating perturbation to dominant image regions that are model-agnostic is crucial to improving adversarial efficacy. However, limiting perturbation to local regions in the spatial domain proves inadequate in augmenting transferability. To this end, we propose a transferable adversarial attack with fine-grained perturbation optimization in the frequency domain, creating centralized perturbation. We devise a systematic pipeline to dynamically constrain perturbation optimization to dominant frequency coefficients. The constraint is optimized in parallel at each iteration, ensuring the directional alignment of perturbation optimization with model prediction. Our approach allows us to centralize perturbation towards sample-specific important frequency features, which are shared by DNNs, effectively mitigating source model overfitting. Experiments demonstrate that by dynamically centralizing perturbation on dominating frequency coefficients, crafted adversarial examples exhibit stronger transferability, and allowing them to bypass various defenses.



## **49. SODA: Protecting Proprietary Information in On-Device Machine Learning Models**

cs.LG

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2312.15036v1) [paper-pdf](http://arxiv.org/pdf/2312.15036v1)

**Authors**: Akanksha Atrey, Ritwik Sinha, Saayan Mitra, Prashant Shenoy

**Abstract**: The growth of low-end hardware has led to a proliferation of machine learning-based services in edge applications. These applications gather contextual information about users and provide some services, such as personalized offers, through a machine learning (ML) model. A growing practice has been to deploy such ML models on the user's device to reduce latency, maintain user privacy, and minimize continuous reliance on a centralized source. However, deploying ML models on the user's edge device can leak proprietary information about the service provider. In this work, we investigate on-device ML models that are used to provide mobile services and demonstrate how simple attacks can leak proprietary information of the service provider. We show that different adversaries can easily exploit such models to maximize their profit and accomplish content theft. Motivated by the need to thwart such attacks, we present an end-to-end framework, SODA, for deploying and serving on edge devices while defending against adversarial usage. Our results demonstrate that SODA can detect adversarial usage with 89% accuracy in less than 50 queries with minimal impact on service performance, latency, and storage.



## **50. Differentiable JPEG: The Devil is in the Details**

cs.CV

Accepted at WACV 2024. Project page:  https://christophreich1996.github.io/differentiable_jpeg/ WACV paper:  https://openaccess.thecvf.com/content/WACV2024/html/Reich_Differentiable_JPEG_The_Devil_Is_in_the_Details_WACV_2024_paper.html

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2309.06978v4) [paper-pdf](http://arxiv.org/pdf/2309.06978v4)

**Authors**: Christoph Reich, Biplob Debnath, Deep Patel, Srimat Chakradhar

**Abstract**: JPEG remains one of the most widespread lossy image coding methods. However, the non-differentiable nature of JPEG restricts the application in deep learning pipelines. Several differentiable approximations of JPEG have recently been proposed to address this issue. This paper conducts a comprehensive review of existing diff. JPEG approaches and identifies critical details that have been missed by previous methods. To this end, we propose a novel diff. JPEG approach, overcoming previous limitations. Our approach is differentiable w.r.t. the input image, the JPEG quality, the quantization tables, and the color conversion parameters. We evaluate the forward and backward performance of our diff. JPEG approach against existing methods. Additionally, extensive ablations are performed to evaluate crucial design choices. Our proposed diff. JPEG resembles the (non-diff.) reference implementation best, significantly surpassing the recent-best diff. approach by $3.47$dB (PSNR) on average. For strong compression rates, we can even improve PSNR by $9.51$dB. Strong adversarial attack results are yielded by our diff. JPEG, demonstrating the effective gradient approximation. Our code is available at https://github.com/necla-ml/Diff-JPEG.



