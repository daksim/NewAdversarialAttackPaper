# Latest Adversarial Attack Papers
**update at 2023-02-28 19:31:06**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Understanding Adversarial Attacks on Observations in Deep Reinforcement Learning**

cs.LG

**SubmitDate**: 2023-02-27    [abs](http://arxiv.org/abs/2106.15860v3) [paper-pdf](http://arxiv.org/pdf/2106.15860v3)

**Authors**: You Qiaoben, Chengyang Ying, Xinning Zhou, Hang Su, Jun Zhu, Bo Zhang

**Abstract**: Deep reinforcement learning models are vulnerable to adversarial attacks that can decrease a victim's cumulative expected reward by manipulating the victim's observations. Despite the efficiency of previous optimization-based methods for generating adversarial noise in supervised learning, such methods might not be able to achieve the lowest cumulative reward since they do not explore the environmental dynamics in general. In this paper, we provide a framework to better understand the existing methods by reformulating the problem of adversarial attacks on reinforcement learning in the function space. Our reformulation generates an optimal adversary in the function space of the targeted attacks, repelling them via a generic two-stage framework. In the first stage, we train a deceptive policy by hacking the environment, and discover a set of trajectories routing to the lowest reward or the worst-case performance. Next, the adversary misleads the victim to imitate the deceptive policy by perturbing the observations. Compared to existing approaches, we theoretically show that our adversary is stronger under an appropriate noise level. Extensive experiments demonstrate our method's superiority in terms of efficiency and effectiveness, achieving the state-of-the-art performance in both Atari and MuJoCo environments.



## **2. Implicit Poisoning Attacks in Two-Agent Reinforcement Learning: Adversarial Policies for Training-Time Attacks**

cs.LG

**SubmitDate**: 2023-02-27    [abs](http://arxiv.org/abs/2302.13851v1) [paper-pdf](http://arxiv.org/pdf/2302.13851v1)

**Authors**: Mohammad Mohammadi, Jonathan Nöther, Debmalya Mandal, Adish Singla, Goran Radanovic

**Abstract**: In targeted poisoning attacks, an attacker manipulates an agent-environment interaction to force the agent into adopting a policy of interest, called target policy. Prior work has primarily focused on attacks that modify standard MDP primitives, such as rewards or transitions. In this paper, we study targeted poisoning attacks in a two-agent setting where an attacker implicitly poisons the effective environment of one of the agents by modifying the policy of its peer. We develop an optimization framework for designing optimal attacks, where the cost of the attack measures how much the solution deviates from the assumed default policy of the peer agent. We further study the computational properties of this optimization framework. Focusing on a tabular setting, we show that in contrast to poisoning attacks based on MDP primitives (transitions and (unbounded) rewards), which are always feasible, it is NP-hard to determine the feasibility of implicit poisoning attacks. We provide characterization results that establish sufficient conditions for the feasibility of the attack problem, as well as an upper and a lower bound on the optimal cost of the attack. We propose two algorithmic approaches for finding an optimal adversarial policy: a model-based approach with tabular policies and a model-free approach with parametric/neural policies. We showcase the efficacy of the proposed algorithms through experiments.



## **3. Locality-Sensitive Hashing Does Not Guarantee Privacy! Attacks on Google's FLoC and the MinHash Hierarchy System**

cs.CR

14 pages, 9 figures submitted to PETS 2023

**SubmitDate**: 2023-02-27    [abs](http://arxiv.org/abs/2302.13635v1) [paper-pdf](http://arxiv.org/pdf/2302.13635v1)

**Authors**: Florian Turati, Carlos Cotrini, Karel Kubicek, David Basin

**Abstract**: Recently proposed systems aim at achieving privacy using locality-sensitive hashing. We show how these approaches fail by presenting attacks against two such systems: Google's FLoC proposal for privacy-preserving targeted advertising and the MinHash Hierarchy, a system for processing mobile users' traffic behavior in a privacy-preserving way. Our attacks refute the pre-image resistance, anonymity, and privacy guarantees claimed for these systems.   In the case of FLoC, we show how to deanonymize users using Sybil attacks and to reconstruct 10% or more of the browsing history for 30% of its users using Generative Adversarial Networks. We achieve this only analyzing the hashes used by FLoC. For MinHash, we precisely identify the movement of a subset of individuals and, on average, we can limit users' movement to just 10% of the possible geographic area, again using just the hashes. In addition, we refute their differential privacy claims.



## **4. Online Black-Box Confidence Estimation of Deep Neural Networks**

cs.CV

**SubmitDate**: 2023-02-27    [abs](http://arxiv.org/abs/2302.13578v1) [paper-pdf](http://arxiv.org/pdf/2302.13578v1)

**Authors**: Fabian Woitschek, Georg Schneider

**Abstract**: Autonomous driving (AD) and advanced driver assistance systems (ADAS) increasingly utilize deep neural networks (DNNs) for improved perception or planning. Nevertheless, DNNs are quite brittle when the data distribution during inference deviates from the data distribution during training. This represents a challenge when deploying in partly unknown environments like in the case of ADAS. At the same time, the standard confidence of DNNs remains high even if the classification reliability decreases. This is problematic since following motion control algorithms consider the apparently confident prediction as reliable even though it might be considerably wrong. To reduce this problem real-time capable confidence estimation is required that better aligns with the actual reliability of the DNN classification. Additionally, the need exists for black-box confidence estimation to enable the homogeneous inclusion of externally developed components to an entire system. In this work we explore this use case and introduce the neighborhood confidence (NHC) which estimates the confidence of an arbitrary DNN for classification. The metric can be used for black-box systems since only the top-1 class output is required and does not need access to the gradients, the training dataset or a hold-out validation dataset. Evaluation on different data distributions, including small in-domain distribution shifts, out-of-domain data or adversarial attacks, shows that the NHC performs better or on par with a comparable method for online white-box confidence estimation in low data regimes which is required for real-time capable AD/ADAS.



## **5. Physical Adversarial Attacks on Deep Neural Networks for Traffic Sign Recognition: A Feasibility Study**

cs.CV

**SubmitDate**: 2023-02-27    [abs](http://arxiv.org/abs/2302.13570v1) [paper-pdf](http://arxiv.org/pdf/2302.13570v1)

**Authors**: Fabian Woitschek, Georg Schneider

**Abstract**: Deep Neural Networks (DNNs) are increasingly applied in the real world in safety critical applications like advanced driver assistance systems. An example for such use case is represented by traffic sign recognition systems. At the same time, it is known that current DNNs can be fooled by adversarial attacks, which raises safety concerns if those attacks can be applied under realistic conditions. In this work we apply different black-box attack methods to generate perturbations that are applied in the physical environment and can be used to fool systems under different environmental conditions. To the best of our knowledge we are the first to combine a general framework for physical attacks with different black-box attack methods and study the impact of the different methods on the success rate of the attack under the same setting. We show that reliable physical adversarial attacks can be performed with different methods and that it is also possible to reduce the perceptibility of the resulting perturbations. The findings highlight the need for viable defenses of a DNN even in the black-box case, but at the same time form the basis for securing a DNN with methods like adversarial training which utilizes adversarial attacks to augment the original training data.



## **6. Aegis: Mitigating Targeted Bit-flip Attacks against Deep Neural Networks**

cs.CR

**SubmitDate**: 2023-02-27    [abs](http://arxiv.org/abs/2302.13520v1) [paper-pdf](http://arxiv.org/pdf/2302.13520v1)

**Authors**: Jialai Wang, Ziyuan Zhang, Meiqi Wang, Han Qiu, Tianwei Zhang, Qi Li, Zongpeng Li, Tao Wei, Chao Zhang

**Abstract**: Bit-flip attacks (BFAs) have attracted substantial attention recently, in which an adversary could tamper with a small number of model parameter bits to break the integrity of DNNs. To mitigate such threats, a batch of defense methods are proposed, focusing on the untargeted scenarios. Unfortunately, they either require extra trustworthy applications or make models more vulnerable to targeted BFAs. Countermeasures against targeted BFAs, stealthier and more purposeful by nature, are far from well established.   In this work, we propose Aegis, a novel defense method to mitigate targeted BFAs. The core observation is that existing targeted attacks focus on flipping critical bits in certain important layers. Thus, we design a dynamic-exit mechanism to attach extra internal classifiers (ICs) to hidden layers. This mechanism enables input samples to early-exit from different layers, which effectively upsets the adversary's attack plans. Moreover, the dynamic-exit mechanism randomly selects ICs for predictions during each inference to significantly increase the attack cost for the adaptive attacks where all defense mechanisms are transparent to the adversary. We further propose a robustness training strategy to adapt ICs to the attack scenarios by simulating BFAs during the IC training phase, to increase model robustness. Extensive evaluations over four well-known datasets and two popular DNN structures reveal that Aegis could effectively mitigate different state-of-the-art targeted attacks, reducing attack success rate by 5-10$\times$, significantly outperforming existing defense methods.



## **7. CBA: Contextual Background Attack against Optical Aerial Detection in the Physical World**

cs.CV

**SubmitDate**: 2023-02-27    [abs](http://arxiv.org/abs/2302.13519v1) [paper-pdf](http://arxiv.org/pdf/2302.13519v1)

**Authors**: Jiawei Lian, Xiaofei Wang, Yuru Su, Mingyang Ma, Shaohui Mei

**Abstract**: Patch-based physical attacks have increasingly aroused concerns.   However, most existing methods focus on obscuring targets captured on the ground, and some of these methods are simply extended to deceive aerial detectors.   They smear the targeted objects in the physical world with the elaborated adversarial patches, which can only slightly sway the aerial detectors' prediction and with weak attack transferability.   To address the above issues, we propose to perform Contextual Background Attack (CBA), a novel physical attack framework against aerial detection, which can achieve strong attack efficacy and transferability in the physical world even without smudging the interested objects at all.   Specifically, the targets of interest, i.e. the aircraft in aerial images, are adopted to mask adversarial patches.   The pixels outside the mask area are optimized to make the generated adversarial patches closely cover the critical contextual background area for detection, which contributes to gifting adversarial patches with more robust and transferable attack potency in the real world.   To further strengthen the attack performance, the adversarial patches are forced to be outside targets during training, by which the detected objects of interest, both on and outside patches, benefit the accumulation of attack efficacy.   Consequently, the sophisticatedly designed patches are gifted with solid fooling efficacy against objects both on and outside the adversarial patches simultaneously.   Extensive proportionally scaled experiments are performed in physical scenarios, demonstrating the superiority and potential of the proposed framework for physical attacks.   We expect that the proposed physical attack method will serve as a benchmark for assessing the adversarial robustness of diverse aerial detectors and defense methods.



## **8. PolyScope: Multi-Policy Access Control Analysis to Triage Android Scoped Storage**

cs.CR

14 pages, 5 figures, submitted to IEEE TDSC. arXiv admin note:  substantial text overlap with arXiv:2008.03593

**SubmitDate**: 2023-02-27    [abs](http://arxiv.org/abs/2302.13506v1) [paper-pdf](http://arxiv.org/pdf/2302.13506v1)

**Authors**: Yu-Tsung Lee, Haining Chen, William Enck, Hayawardh Vijayakumar, Ninghui Li, Zhiyun Qian, Giuseppe Petracca

**Abstract**: Android's filesystem access control is a crucial aspect of its system integrity. It utilizes a combination of mandatory access controls, such as SELinux, and discretionary access controls, like Unix permissions, along with specialized access controls such as Android permissions to safeguard OEM and Android services from third-party applications. However, when OEMs introduce differentiating features, they often create vulnerabilities due to their inability to properly reconfigure this complex policy combination. To address this, we introduce the POLYSCOPE tool, which triages Android filesystem access control policies to identify attack operations - authorized operations that may be exploited by adversaries to elevate their privileges. POLYSCOPE has three significant advantages over prior analyses: it allows for the independent extension and analysis of individual policy models, understands the flexibility untrusted parties have in modifying access control policies, and can identify attack operations that system configurations permit. We demonstrate the effectiveness of POLYSCOPE by examining the impact of Scoped Storage on Android, revealing that it reduces the number of attack operations possible on external storage resources by over 50%. However, because OEMs only partially adopt Scoped Storage, we also uncover two previously unknown vulnerabilities, demonstrating how POLYSCOPE can assess an ideal scenario where all apps comply with Scoped Storage, which can reduce the number of untrusted parties accessing attack operations by over 65% on OEM systems. POLYSCOPE thus helps Android OEMs evaluate complex access control policies to pinpoint the attack operations that require further examination.



## **9. Contextual adversarial attack against aerial detection in the physical world**

cs.CV

**SubmitDate**: 2023-02-27    [abs](http://arxiv.org/abs/2302.13487v1) [paper-pdf](http://arxiv.org/pdf/2302.13487v1)

**Authors**: Jiawei Lian, Xiaofei Wang, Yuru Su, Mingyang Ma, Shaohui Mei

**Abstract**: Deep Neural Networks (DNNs) have been extensively utilized in aerial detection. However, DNNs' sensitivity and vulnerability to maliciously elaborated adversarial examples have progressively garnered attention. Recently, physical attacks have gradually become a hot issue due to they are more practical in the real world, which poses great threats to some security-critical applications. In this paper, we take the first attempt to perform physical attacks in contextual form against aerial detection in the physical world. We propose an innovative contextual attack method against aerial detection in real scenarios, which achieves powerful attack performance and transfers well between various aerial object detectors without smearing or blocking the interested objects to hide. Based on the findings that the targets' contextual information plays an important role in aerial detection by observing the detectors' attention maps, we propose to make full use of the contextual area of the interested targets to elaborate contextual perturbations for the uncovered attacks in real scenarios. Extensive proportionally scaled experiments are conducted to evaluate the effectiveness of the proposed contextual attack method, which demonstrates the proposed method's superiority in both attack efficacy and physical practicality.



## **10. Randomness in ML Defenses Helps Persistent Attackers and Hinders Evaluators**

cs.LG

**SubmitDate**: 2023-02-27    [abs](http://arxiv.org/abs/2302.13464v1) [paper-pdf](http://arxiv.org/pdf/2302.13464v1)

**Authors**: Keane Lucas, Matthew Jagielski, Florian Tramèr, Lujo Bauer, Nicholas Carlini

**Abstract**: It is becoming increasingly imperative to design robust ML defenses. However, recent work has found that many defenses that initially resist state-of-the-art attacks can be broken by an adaptive adversary. In this work we take steps to simplify the design of defenses and argue that white-box defenses should eschew randomness when possible. We begin by illustrating a new issue with the deployment of randomized defenses that reduces their security compared to their deterministic counterparts. We then provide evidence that making defenses deterministic simplifies robustness evaluation, without reducing the effectiveness of a truly robust defense. Finally, we introduce a new defense evaluation framework that leverages a defense's deterministic nature to better evaluate its adversarial robustness.



## **11. Investigating the Security of EV Charging Mobile Applications As an Attack Surface**

cs.CR

**SubmitDate**: 2023-02-26    [abs](http://arxiv.org/abs/2211.10603v2) [paper-pdf](http://arxiv.org/pdf/2211.10603v2)

**Authors**: K. Sarieddine, M. A. Sayed, S. Torabi, R. Atallah, C. Assi

**Abstract**: In this paper, we study the security posture of the EV charging ecosystem against a new type of remote that exploits vulnerabilities in the EV charging mobile applications as an attack surface. We leverage a combination of static and dynamic analysis techniques to analyze the security of widely used EV charging mobile applications. Our analysis was performed on 31 of the most widely used mobile applications including their interactions with various components such as cloud management systems. The attack, scenarios that exploit these vulnerabilities were verified on a real-time co-simulation test bed. Our discoveries indicate the lack of user/vehicle verification and improper authorization for critical functions, which allow adversaries to remotely hijack charging sessions and launch attacks against the connected critical infrastructure. The attacks were demonstrated using the EVCS mobile applications showing the feasibility and the applicability of our attacks. Indeed, we discuss specific remote attack scenarios and their impact on EV users. More importantly, our analysis results demonstrate the feasibility of leveraging existing vulnerabilities across various EV charging mobile applications to perform wide-scale coordinated remote charging/discharging attacks against the connected critical infrastructure (e.g., power grid), with significant economical and operational implications. Finally, we propose countermeasures to secure the infrastructure and impede adversaries from performing reconnaissance and launching remote attacks using compromised accounts.



## **12. Adversarial Path Planning for Optimal Camera Positioning**

cs.CG

**SubmitDate**: 2023-02-26    [abs](http://arxiv.org/abs/2302.07051v2) [paper-pdf](http://arxiv.org/pdf/2302.07051v2)

**Authors**: Gaia Carenini, Alexandre Duplessis

**Abstract**: The use of visual sensors is flourishing, driven among others by the several applications in detection and prevention of crimes or dangerous events. While the problem of optimal camera placement for total coverage has been solved for a decade or so, that of the arrangement of cameras maximizing the recognition of objects "in-transit" is still open. The objective of this paper is to attack this problem by providing an adversarial method of proven optimality based on the resolution of Hamilton-Jacobi equations. The problem is attacked by first assuming the perspective of an adversary, i.e. computing explicitly the path minimizing the probability of detection and the quality of reconstruction. Building on this result, we introduce an optimality measure for camera configurations and perform a simulated annealing algorithm to find the optimal camera placement.



## **13. Empowering Graph Representation Learning with Test-Time Graph Transformation**

cs.LG

ICLR 2023

**SubmitDate**: 2023-02-26    [abs](http://arxiv.org/abs/2210.03561v2) [paper-pdf](http://arxiv.org/pdf/2210.03561v2)

**Authors**: Wei Jin, Tong Zhao, Jiayuan Ding, Yozen Liu, Jiliang Tang, Neil Shah

**Abstract**: As powerful tools for representation learning on graphs, graph neural networks (GNNs) have facilitated various applications from drug discovery to recommender systems. Nevertheless, the effectiveness of GNNs is immensely challenged by issues related to data quality, such as distribution shift, abnormal features and adversarial attacks. Recent efforts have been made on tackling these issues from a modeling perspective which requires additional cost of changing model architectures or re-training model parameters. In this work, we provide a data-centric view to tackle these issues and propose a graph transformation framework named GTrans which adapts and refines graph data at test time to achieve better performance. We provide theoretical analysis on the design of the framework and discuss why adapting graph data works better than adapting the model. Extensive experiments have demonstrated the effectiveness of GTrans on three distinct scenarios for eight benchmark datasets where suboptimal data is presented. Remarkably, GTrans performs the best in most cases with improvements up to 2.8%, 8.2% and 3.8% over the best baselines on three experimental settings. Code is released at https://github.com/ChandlerBang/GTrans.



## **14. Deep Learning-based Multi-Organ CT Segmentation with Adversarial Data Augmentation**

eess.IV

Accepted at SPIE Medical Imaging 2023

**SubmitDate**: 2023-02-25    [abs](http://arxiv.org/abs/2302.13172v1) [paper-pdf](http://arxiv.org/pdf/2302.13172v1)

**Authors**: Shaoyan Pan, Shao-Yuan Lo, Min Huang, Chaoqiong Ma, Jacob Wynne, Tonghe Wang, Tian Liu, Xiaofeng Yang

**Abstract**: In this work, we propose an adversarial attack-based data augmentation method to improve the deep-learning-based segmentation algorithm for the delineation of Organs-At-Risk (OAR) in abdominal Computed Tomography (CT) to facilitate radiation therapy. We introduce Adversarial Feature Attack for Medical Image (AFA-MI) augmentation, which forces the segmentation network to learn out-of-distribution statistics and improve generalization and robustness to noises. AFA-MI augmentation consists of three steps: 1) generate adversarial noises by Fast Gradient Sign Method (FGSM) on the intermediate features of the segmentation network's encoder; 2) inject the generated adversarial noises into the network, intentionally compromising performance; 3) optimize the network with both clean and adversarial features. Experiments are conducted segmenting the heart, left and right kidney, liver, left and right lung, spinal cord, and stomach. We first evaluate the AFA-MI augmentation using nnUnet and TT-Vnet on the test data from a public abdominal dataset and an institutional dataset. In addition, we validate how AFA-MI affects the networks' robustness to the noisy data by evaluating the networks with added Gaussian noises of varying magnitudes to the institutional dataset. Network performance is quantitatively evaluated using Dice Similarity Coefficient (DSC) for volume-based accuracy. Also, Hausdorff Distance (HD) is applied for surface-based accuracy. On the public dataset, nnUnet with AFA-MI achieves DSC = 0.85 and HD = 6.16 millimeters (mm); and TT-Vnet achieves DSC = 0.86 and HD = 5.62 mm. AFA-MI augmentation further improves all contour accuracies up to 0.217 DSC score when tested on images with Gaussian noises. AFA-MI augmentation is therefore demonstrated to improve segmentation performance and robustness in CT multi-organ segmentation.



## **15. Chaotic Variational Auto encoder-based Adversarial Machine Learning**

cs.LG

24 pages, 6 figures and 5 tables

**SubmitDate**: 2023-02-25    [abs](http://arxiv.org/abs/2302.12959v1) [paper-pdf](http://arxiv.org/pdf/2302.12959v1)

**Authors**: Pavan Venkata Sainadh Reddy, Yelleti Vivek, Gopi Pranay, Vadlamani Ravi

**Abstract**: Machine Learning (ML) has become the new contrivance in almost every field. This makes them a target of fraudsters by various adversary attacks, thereby hindering the performance of ML models. Evasion and Data-Poison-based attacks are well acclaimed, especially in finance, healthcare, etc. This motivated us to propose a novel computationally less expensive attack mechanism based on the adversarial sample generation by Variational Auto Encoder (VAE). It is well known that Wavelet Neural Network (WNN) is considered computationally efficient in solving image and audio processing, speech recognition, and time-series forecasting. This paper proposed VAE-Deep-Wavelet Neural Network (VAE-Deep-WNN), where Encoder and Decoder employ WNN networks. Further, we proposed chaotic variants of both VAE with Multi-layer perceptron (MLP) and Deep-WNN and named them C-VAE-MLP and C-VAE-Deep-WNN, respectively. Here, we employed a Logistic map to generate random noise in the latent space. In this paper, we performed VAE-based adversary sample generation and applied it to various problems related to finance and cybersecurity domain-related problems such as loan default, credit card fraud, and churn modelling, etc., We performed both Evasion and Data-Poison attacks on Logistic Regression (LR) and Decision Tree (DT) models. The results indicated that VAE-Deep-WNN outperformed the rest in the majority of the datasets and models. However, its chaotic variant C-VAE-Deep-WNN performed almost similarly to VAE-Deep-WNN in the majority of the datasets.



## **16. Edge-Based Detection and Localization of Adversarial Oscillatory Load Attacks Orchestrated By Compromised EV Charging Stations**

cs.CR

**SubmitDate**: 2023-02-24    [abs](http://arxiv.org/abs/2302.12890v1) [paper-pdf](http://arxiv.org/pdf/2302.12890v1)

**Authors**: Khaled Sarieddine, Mohammad Ali Sayed, Sadegh Torabi, Ribal Atallah, Chadi Assi

**Abstract**: In this paper, we investigate an edge-based approach for the detection and localization of coordinated oscillatory load attacks initiated by exploited EV charging stations against the power grid. We rely on the behavioral characteristics of the power grid in the presence of interconnected EVCS while combining cyber and physical layer features to implement deep learning algorithms for the effective detection of oscillatory load attacks at the EVCS. We evaluate the proposed detection approach by building a real-time test bed to synthesize benign and malicious data, which was generated by analyzing real-life EV charging data collected during recent years. The results demonstrate the effectiveness of the implemented approach with the Convolutional Long-Short Term Memory model producing optimal classification accuracy (99.4\%). Moreover, our analysis results shed light on the impact of such detection mechanisms towards building resiliency into different levels of the EV charging ecosystem while allowing power grid operators to localize attacks and take further mitigation measures. Specifically, we managed to decentralize the detection mechanism of oscillatory load attacks and create an effective alternative for operator-centric mechanisms to mitigate multi-operator and MitM oscillatory load attacks against the power grid. Finally, we leverage the created test bed to evaluate a distributed mitigation technique, which can be deployed on public/private charging stations to average out the impact of oscillatory load attacks while allowing the power system to recover smoothly within 1 second with minimal overhead.



## **17. Take Me Home: Reversing Distribution Shifts using Reinforcement Learning**

cs.LG

preprint (under submission)

**SubmitDate**: 2023-02-24    [abs](http://arxiv.org/abs/2302.10341v2) [paper-pdf](http://arxiv.org/pdf/2302.10341v2)

**Authors**: Vivian Lin, Kuk Jin Jang, Souradeep Dutta, Michele Caprio, Oleg Sokolsky, Insup Lee

**Abstract**: Deep neural networks have repeatedly been shown to be non-robust to the uncertainties of the real world. Even subtle adversarial attacks and naturally occurring distribution shifts wreak havoc on systems relying on deep neural networks. In response to this, current state-of-the-art techniques use data-augmentation to enrich the training distribution of the model and consequently improve robustness to natural distribution shifts. We propose an alternative approach that allows the system to recover from distribution shifts online. Specifically, our method applies a sequence of semantic-preserving transformations to bring the shifted data closer in distribution to the training set, as measured by the Wasserstein distance. We formulate the problem of sequence selection as an MDP, which we solve using reinforcement learning. To aid in our estimates of Wasserstein distance, we employ dimensionality reduction through orthonormal projection. We provide both theoretical and empirical evidence that orthonormal projection preserves characteristics of the data at the distributional level. Finally, we apply our distribution shift recovery approach to the ImageNet-C benchmark for distribution shifts, targeting shifts due to additive noise and image histogram modifications. We demonstrate an improvement in average accuracy up to 14.21% across a variety of state-of-the-art ImageNet classifiers.



## **18. Adversarial Robustness for Tabular Data through Cost and Utility Awareness**

cs.LG

The first two authors contributed equally. To appear in the  proceedings of NDSS 2023

**SubmitDate**: 2023-02-24    [abs](http://arxiv.org/abs/2208.13058v2) [paper-pdf](http://arxiv.org/pdf/2208.13058v2)

**Authors**: Klim Kireev, Bogdan Kulynych, Carmela Troncoso

**Abstract**: Many safety-critical applications of machine learning, such as fraud or abuse detection, use data in tabular domains. Adversarial examples can be particularly damaging for these applications. Yet, existing works on adversarial robustness primarily focus on machine-learning models in image and text domains. We argue that, due to the differences between tabular data and images or text, existing threat models are not suitable for tabular domains. These models do not capture that the costs of an attack could be more significant than imperceptibility, or that the adversary could assign different values to the utility obtained from deploying different adversarial examples. We demonstrate that, due to these differences, the attack and defense methods used for images and text cannot be directly applied to tabular settings. We address these issues by proposing new cost and utility-aware threat models that are tailored to the adversarial capabilities and constraints of attackers targeting tabular domains. We introduce a framework that enables us to design attack and defense mechanisms that result in models protected against cost and utility-aware adversaries, for example, adversaries constrained by a certain financial budget. We show that our approach is effective on three datasets corresponding to applications for which adversarial examples can have economic and social implications.



## **19. Defending Against Backdoor Attacks by Layer-wise Feature Analysis**

cs.CR

This paper is accepted by PAKDD 2023

**SubmitDate**: 2023-02-24    [abs](http://arxiv.org/abs/2302.12758v1) [paper-pdf](http://arxiv.org/pdf/2302.12758v1)

**Authors**: Najeeb Moharram Jebreel, Josep Domingo-Ferrer, Yiming Li

**Abstract**: Training deep neural networks (DNNs) usually requires massive training data and computational resources. Users who cannot afford this may prefer to outsource training to a third party or resort to publicly available pre-trained models. Unfortunately, doing so facilitates a new training-time attack (i.e., backdoor attack) against DNNs. This attack aims to induce misclassification of input samples containing adversary-specified trigger patterns. In this paper, we first conduct a layer-wise feature analysis of poisoned and benign samples from the target class. We find out that the feature difference between benign and poisoned samples tends to be maximum at a critical layer, which is not always the one typically used in existing defenses, namely the layer before fully-connected layers. We also demonstrate how to locate this critical layer based on the behaviors of benign samples. We then propose a simple yet effective method to filter poisoned samples by analyzing the feature differences between suspicious and benign samples at the critical layer. We conduct extensive experiments on two benchmark datasets, which confirm the effectiveness of our defense.



## **20. Harnessing the Speed and Accuracy of Machine Learning to Advance Cybersecurity**

cs.CR

**SubmitDate**: 2023-02-24    [abs](http://arxiv.org/abs/2302.12415v1) [paper-pdf](http://arxiv.org/pdf/2302.12415v1)

**Authors**: Khatoon Mohammed

**Abstract**: As cyber attacks continue to increase in frequency and sophistication, detecting malware has become a critical task for maintaining the security of computer systems. Traditional signature-based methods of malware detection have limitations in detecting complex and evolving threats. In recent years, machine learning (ML) has emerged as a promising solution to detect malware effectively. ML algorithms are capable of analyzing large datasets and identifying patterns that are difficult for humans to identify. This paper presents a comprehensive review of the state-of-the-art ML techniques used in malware detection, including supervised and unsupervised learning, deep learning, and reinforcement learning. We also examine the challenges and limitations of ML-based malware detection, such as the potential for adversarial attacks and the need for large amounts of labeled data. Furthermore, we discuss future directions in ML-based malware detection, including the integration of multiple ML algorithms and the use of explainable AI techniques to enhance the interpret ability of ML-based detection systems. Our research highlights the potential of ML-based techniques to improve the speed and accuracy of malware detection, and contribute to enhancing cybersecurity



## **21. Principled Data-Driven Decision Support for Cyber-Forensic Investigations**

cs.CR

**SubmitDate**: 2023-02-24    [abs](http://arxiv.org/abs/2211.13345v2) [paper-pdf](http://arxiv.org/pdf/2211.13345v2)

**Authors**: Soodeh Atefi, Sakshyam Panda, Manos Panaousis, Aron Laszka

**Abstract**: In the wake of a cybersecurity incident, it is crucial to promptly discover how the threat actors breached security in order to assess the impact of the incident and to develop and deploy countermeasures that can protect against further attacks. To this end, defenders can launch a cyber-forensic investigation, which discovers the techniques that the threat actors used in the incident. A fundamental challenge in such an investigation is prioritizing the investigation of particular techniques since the investigation of each technique requires time and effort, but forensic analysts cannot know which ones were actually used before investigating them. To ensure prompt discovery, it is imperative to provide decision support that can help forensic analysts with this prioritization. A recent study demonstrated that data-driven decision support, based on a dataset of prior incidents, can provide state-of-the-art prioritization. However, this data-driven approach, called DISCLOSE, is based on a heuristic that utilizes only a subset of the available information and does not approximate optimal decisions. To improve upon this heuristic, we introduce a principled approach for data-driven decision support for cyber-forensic investigations. We formulate the decision-support problem using a Markov decision process, whose states represent the states of a forensic investigation. To solve the decision problem, we propose a Monte Carlo tree search based method, which relies on a k-NN regression over prior incidents to estimate state-transition probabilities. We evaluate our proposed approach on multiple versions of the MITRE ATT&CK dataset, which is a knowledge base of adversarial techniques and tactics based on real-world cyber incidents, and demonstrate that our approach outperforms DISCLOSE in terms of techniques discovered per effort spent.



## **22. HyperAttack: Multi-Gradient-Guided White-box Adversarial Structure Attack of Hypergraph Neural Networks**

cs.LG

10+2pages,9figures

**SubmitDate**: 2023-02-24    [abs](http://arxiv.org/abs/2302.12407v1) [paper-pdf](http://arxiv.org/pdf/2302.12407v1)

**Authors**: Chao Hu, Ruishi Yu, Binqi Zeng, Yu Zhan, Ying Fu, Quan Zhang, Rongkai Liu, Heyuan Shi

**Abstract**: Hypergraph neural networks (HGNN) have shown superior performance in various deep learning tasks, leveraging the high-order representation ability to formulate complex correlations among data by connecting two or more nodes through hyperedge modeling. Despite the well-studied adversarial attacks on Graph Neural Networks (GNN), there is few study on adversarial attacks against HGNN, which leads to a threat to the safety of HGNN applications. In this paper, we introduce HyperAttack, the first white-box adversarial attack framework against hypergraph neural networks. HyperAttack conducts a white-box structure attack by perturbing hyperedge link status towards the target node with the guidance of both gradients and integrated gradients. We evaluate HyperAttack on the widely-used Cora and PubMed datasets and three hypergraph neural networks with typical hypergraph modeling techniques. Compared to state-of-the-art white-box structural attack methods for GNN, HyperAttack achieves a 10-20X improvement in time efficiency while also increasing attack success rates by 1.3%-3.7%. The results show that HyperAttack can achieve efficient adversarial attacks that balance effectiveness and time costs.



## **23. On the Hardness of Robustness Transfer: A Perspective from Rademacher Complexity over Symmetric Difference Hypothesis Space**

cs.LG

**SubmitDate**: 2023-02-23    [abs](http://arxiv.org/abs/2302.12351v1) [paper-pdf](http://arxiv.org/pdf/2302.12351v1)

**Authors**: Yuyang Deng, Nidham Gazagnadou, Junyuan Hong, Mehrdad Mahdavi, Lingjuan Lyu

**Abstract**: Recent studies demonstrated that the adversarially robust learning under $\ell_\infty$ attack is harder to generalize to different domains than standard domain adaptation. How to transfer robustness across different domains has been a key question in domain adaptation field. To investigate the fundamental difficulty behind adversarially robust domain adaptation (or robustness transfer), we propose to analyze a key complexity measure that controls the cross-domain generalization: the adversarial Rademacher complexity over {\em symmetric difference hypothesis space} $\mathcal{H} \Delta \mathcal{H}$. For linear models, we show that adversarial version of this complexity is always greater than the non-adversarial one, which reveals the intrinsic hardness of adversarially robust domain adaptation. We also establish upper bounds on this complexity measure. Then we extend them to the ReLU neural network class by upper bounding the adversarial Rademacher complexity in the binary classification setting. Finally, even though the robust domain adaptation is provably harder, we do find positive relation between robust learning and standard domain adaptation. We explain \emph{how adversarial training helps domain adaptation in terms of standard risk}. We believe our results initiate the study of the generalization theory of adversarially robust domain adaptation, and could shed lights on distributed adversarially robust learning from heterogeneous sources, e.g., federated learning scenario.



## **24. Characterizing Internal Evasion Attacks in Federated Learning**

cs.LG

16 pages, 8 figures (14 images if counting sub-figures separately),  Camera ready version for AISTATS 2023, longer version of paper submitted to  CrossFL 2022 poster workshop, code available at  (https://github.com/tj-kim/pFedDef_v1)

**SubmitDate**: 2023-02-23    [abs](http://arxiv.org/abs/2209.08412v2) [paper-pdf](http://arxiv.org/pdf/2209.08412v2)

**Authors**: Taejin Kim, Shubhranshu Singh, Nikhil Madaan, Carlee Joe-Wong

**Abstract**: Federated learning allows for clients in a distributed system to jointly train a machine learning model. However, clients' models are vulnerable to attacks during the training and testing phases. In this paper, we address the issue of adversarial clients performing "internal evasion attacks": crafting evasion attacks at test time to deceive other clients. For example, adversaries may aim to deceive spam filters and recommendation systems trained with federated learning for monetary gain. The adversarial clients have extensive information about the victim model in a federated learning setting, as weight information is shared amongst clients. We are the first to characterize the transferability of such internal evasion attacks for different learning methods and analyze the trade-off between model accuracy and robustness depending on the degree of similarities in client data. We show that adversarial training defenses in the federated learning setting only display limited improvements against internal attacks. However, combining adversarial training with personalized federated learning frameworks increases relative internal attack robustness by 60% compared to federated adversarial training and performs well under limited system resources.



## **25. Boosting Adversarial Transferability using Dynamic Cues**

cs.CV

International Conference on Learning Representations (ICLR'23),  Code:https://bit.ly/3Xd9gRQ

**SubmitDate**: 2023-02-23    [abs](http://arxiv.org/abs/2302.12252v1) [paper-pdf](http://arxiv.org/pdf/2302.12252v1)

**Authors**: Muzammal Naseer, Ahmad Mahmood, Salman Khan, Fahad Khan

**Abstract**: The transferability of adversarial perturbations between image models has been extensively studied. In this case, an attack is generated from a known surrogate \eg, the ImageNet trained model, and transferred to change the decision of an unknown (black-box) model trained on an image dataset. However, attacks generated from image models do not capture the dynamic nature of a moving object or a changing scene due to a lack of temporal cues within image models. This leads to reduced transferability of adversarial attacks from representation-enriched \emph{image} models such as Supervised Vision Transformers (ViTs), Self-supervised ViTs (\eg, DINO), and Vision-language models (\eg, CLIP) to black-box \emph{video} models. In this work, we induce dynamic cues within the image models without sacrificing their original performance on images. To this end, we optimize \emph{temporal prompts} through frozen image models to capture motion dynamics. Our temporal prompts are the result of a learnable transformation that allows optimizing for temporal gradients during an adversarial attack to fool the motion dynamics. Specifically, we introduce spatial (image) and temporal (video) cues within the same source model through task-specific prompts. Attacking such prompts maximizes the adversarial transferability from image-to-video and image-to-image models using the attacks designed for image models. Our attack results indicate that the attacker does not need specialized architectures, \eg, divided space-time attention, 3D convolutions, or multi-view convolution networks for different data modalities. Image models are effective surrogates to optimize an adversarial attack to fool black-box models in a changing environment over time. Code is available at https://bit.ly/3Xd9gRQ



## **26. More than you've asked for: A Comprehensive Analysis of Novel Prompt Injection Threats to Application-Integrated Large Language Models**

cs.CR

**SubmitDate**: 2023-02-23    [abs](http://arxiv.org/abs/2302.12173v1) [paper-pdf](http://arxiv.org/pdf/2302.12173v1)

**Authors**: Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, Mario Fritz

**Abstract**: We are currently witnessing dramatic advances in the capabilities of Large Language Models (LLMs). They are already being adopted in practice and integrated into many systems, including integrated development environments (IDEs) and search engines. The functionalities of current LLMs can be modulated via natural language prompts, while their exact internal functionality remains implicit and unassessable. This property, which makes them adaptable to even unseen tasks, might also make them susceptible to targeted adversarial prompting. Recently, several ways to misalign LLMs using Prompt Injection (PI) attacks have been introduced. In such attacks, an adversary can prompt the LLM to produce malicious content or override the original instructions and the employed filtering schemes. Recent work showed that these attacks are hard to mitigate, as state-of-the-art LLMs are instruction-following. So far, these attacks assumed that the adversary is directly prompting the LLM.   In this work, we show that augmenting LLMs with retrieval and API calling capabilities (so-called Application-Integrated LLMs) induces a whole new set of attack vectors. These LLMs might process poisoned content retrieved from the Web that contains malicious prompts pre-injected and selected by adversaries. We demonstrate that an attacker can indirectly perform such PI attacks. Based on this key insight, we systematically analyze the resulting threat landscape of Application-Integrated LLMs and discuss a variety of new attack vectors. To demonstrate the practical viability of our attacks, we implemented specific demonstrations of the proposed attacks within synthetic applications. In summary, our work calls for an urgent evaluation of current mitigation techniques and an investigation of whether new techniques are needed to defend LLMs against these threats.



## **27. A Plot is Worth a Thousand Words: Model Information Stealing Attacks via Scientific Plots**

cs.CR

To appear in the 32nd USENIX Security Symposium, August 2023,  Anaheim, CA, USA

**SubmitDate**: 2023-02-23    [abs](http://arxiv.org/abs/2302.11982v1) [paper-pdf](http://arxiv.org/pdf/2302.11982v1)

**Authors**: Boyang Zhang, Xinlei He, Yun Shen, Tianhao Wang, Yang Zhang

**Abstract**: Building advanced machine learning (ML) models requires expert knowledge and many trials to discover the best architecture and hyperparameter settings. Previous work demonstrates that model information can be leveraged to assist other attacks, such as membership inference, generating adversarial examples. Therefore, such information, e.g., hyperparameters, should be kept confidential. It is well known that an adversary can leverage a target ML model's output to steal the model's information. In this paper, we discover a new side channel for model information stealing attacks, i.e., models' scientific plots which are extensively used to demonstrate model performance and are easily accessible. Our attack is simple and straightforward. We leverage the shadow model training techniques to generate training data for the attack model which is essentially an image classifier. Extensive evaluation on three benchmark datasets shows that our proposed attack can effectively infer the architecture/hyperparameters of image classifiers based on convolutional neural network (CNN) given the scientific plot generated from it. We also reveal that the attack's success is mainly caused by the shape of the scientific plots, and further demonstrate that the attacks are robust in various scenarios. Given the simplicity and effectiveness of the attack method, our study indicates scientific plots indeed constitute a valid side channel for model information stealing attacks. To mitigate the attacks, we propose several defense mechanisms that can reduce the original attacks' accuracy while maintaining the plot utility. However, such defenses can still be bypassed by adaptive attacks.



## **28. CalFAT: Calibrated Federated Adversarial Training with Label Skewness**

cs.LG

Accepted to the Conference on the Advances in Neural Information  Processing Systems (NeurIPS) 2022

**SubmitDate**: 2023-02-23    [abs](http://arxiv.org/abs/2205.14926v3) [paper-pdf](http://arxiv.org/pdf/2205.14926v3)

**Authors**: Chen Chen, Yuchen Liu, Xingjun Ma, Lingjuan Lyu

**Abstract**: Recent studies have shown that, like traditional machine learning, federated learning (FL) is also vulnerable to adversarial attacks. To improve the adversarial robustness of FL, federated adversarial training (FAT) methods have been proposed to apply adversarial training locally before global aggregation. Although these methods demonstrate promising results on independent identically distributed (IID) data, they suffer from training instability on non-IID data with label skewness, resulting in degraded natural accuracy. This tends to hinder the application of FAT in real-world applications where the label distribution across the clients is often skewed. In this paper, we study the problem of FAT under label skewness, and reveal one root cause of the training instability and natural accuracy degradation issues: skewed labels lead to non-identical class probabilities and heterogeneous local models. We then propose a Calibrated FAT (CalFAT) approach to tackle the instability issue by calibrating the logits adaptively to balance the classes. We show both theoretically and empirically that the optimization of CalFAT leads to homogeneous local models across the clients and better convergence points.



## **29. Adversarial Contrastive Distillation with Adaptive Denoising**

cs.CV

accepted for ICASSP 2023

**SubmitDate**: 2023-02-23    [abs](http://arxiv.org/abs/2302.08764v2) [paper-pdf](http://arxiv.org/pdf/2302.08764v2)

**Authors**: Yuzheng Wang, Zhaoyu Chen, Dingkang Yang, Yang Liu, Siao Liu, Wenqiang Zhang, Lizhe Qi

**Abstract**: Adversarial Robustness Distillation (ARD) is a novel method to boost the robustness of small models. Unlike general adversarial training, its robust knowledge transfer can be less easily restricted by the model capacity. However, the teacher model that provides the robustness of knowledge does not always make correct predictions, interfering with the student's robust performances. Besides, in the previous ARD methods, the robustness comes entirely from one-to-one imitation, ignoring the relationship between examples. To this end, we propose a novel structured ARD method called Contrastive Relationship DeNoise Distillation (CRDND). We design an adaptive compensation module to model the instability of the teacher. Moreover, we utilize the contrastive relationship to explore implicit robustness knowledge among multiple examples. Experimental results on multiple attack benchmarks show CRDND can transfer robust knowledge efficiently and achieves state-of-the-art performances.



## **30. Mitigating Adversarial Attacks in Deepfake Detection: An Exploration of Perturbation and AI Techniques**

cs.LG

**SubmitDate**: 2023-02-22    [abs](http://arxiv.org/abs/2302.11704v1) [paper-pdf](http://arxiv.org/pdf/2302.11704v1)

**Authors**: Saminder Dhesi, Laura Fontes, Pedro Machado, Isibor Kennedy Ihianle, Farhad Fassihi Tash, David Ada Adama

**Abstract**: Deep learning is a crucial aspect of machine learning, but it also makes these techniques vulnerable to adversarial examples, which can be seen in a variety of applications. These examples can even be targeted at humans, leading to the creation of false media, such as deepfakes, which are often used to shape public opinion and damage the reputation of public figures. This article will explore the concept of adversarial examples, which are comprised of perturbations added to clean images or videos, and their ability to deceive DL algorithms. The proposed approach achieved a precision value of accuracy of 76.2% on the DFDC dataset.



## **31. Decorrelative Network Architecture for Robust Electrocardiogram Classification**

cs.LG

16 pages, 6 figures

**SubmitDate**: 2023-02-22    [abs](http://arxiv.org/abs/2207.09031v3) [paper-pdf](http://arxiv.org/pdf/2207.09031v3)

**Authors**: Christopher Wiedeman, Ge Wang

**Abstract**: Artificial intelligence has made great progress in medical data analysis, but the lack of robustness and trustworthiness has kept these methods from being widely deployed. As it is not possible to train networks that are accurate in all situations, models must recognize situations where they cannot operate confidently. Bayesian deep learning methods sample the model parameter space to estimate uncertainty, but these parameters are often subject to the same vulnerabilities, which can be exploited by adversarial attacks. We propose a novel ensemble approach based on feature decorrelation and Fourier partitioning for teaching networks diverse complementary features, reducing the chance of perturbation-based fooling. We test our approach on electrocardiogram classification, demonstrating superior accuracy confidence measurement, on a variety of adversarial attacks. For example, on our ensemble trained with both decorrelation and Fourier partitioning scored a 50.18% inference accuracy and 48.01% uncertainty accuracy (area under the curve) on {\epsilon} = 50 projected gradient descent attacks, while a conventionally trained ensemble scored 21.1% and 30.31% on these metrics respectively. Our approach does not require expensive optimization with adversarial samples and can be scaled to large problems. These methods can easily be applied to other tasks for more robust and trustworthy models.



## **32. Disrupting Adversarial Transferability in Deep Neural Networks**

cs.LG

20 pages, 13 figures

**SubmitDate**: 2023-02-22    [abs](http://arxiv.org/abs/2108.12492v3) [paper-pdf](http://arxiv.org/pdf/2108.12492v3)

**Authors**: Christopher Wiedeman, Ge Wang

**Abstract**: Adversarial attack transferability is well-recognized in deep learning. Prior work has partially explained transferability by recognizing common adversarial subspaces and correlations between decision boundaries, but little is known beyond this. We propose that transferability between seemingly different models is due to a high linear correlation between the feature sets that different networks extract. In other words, two models trained on the same task that are distant in the parameter space likely extract features in the same fashion, just with trivial affine transformations between the latent spaces. Furthermore, we show how applying a feature correlation loss, which decorrelates the extracted features in a latent space, can reduce the transferability of adversarial attacks between models, suggesting that the models complete tasks in semantically different ways. Finally, we propose a Dual Neck Autoencoder (DNA), which leverages this feature correlation loss to create two meaningfully different encodings of input information with reduced transferability.



## **33. Public Key Encryption with Secure Key Leasing**

quant-ph

67 pages, 4 figures

**SubmitDate**: 2023-02-22    [abs](http://arxiv.org/abs/2302.11663v1) [paper-pdf](http://arxiv.org/pdf/2302.11663v1)

**Authors**: Shweta Agrawal, Fuyuki Kitagawa, Ryo Nishimaki, Shota Yamada, Takashi Yamakawa

**Abstract**: We introduce the notion of public key encryption with secure key leasing (PKE-SKL). Our notion supports the leasing of decryption keys so that a leased key achieves the decryption functionality but comes with the guarantee that if the quantum decryption key returned by a user passes a validity test, then the user has lost the ability to decrypt. Our notion is similar in spirit to the notion of secure software leasing (SSL) introduced by Ananth and La Placa (Eurocrypt 2021) but captures significantly more general adversarial strategies. In more detail, our adversary is not restricted to use an honest evaluation algorithm to run pirated software. Our results can be summarized as follows:   1. Definitions: We introduce the definition of PKE with secure key leasing and formalize security notions.   2. Constructing PKE with Secure Key Leasing: We provide a construction of PKE-SKL by leveraging a PKE scheme that satisfies a new security notion that we call consistent or inconsistent security against key leasing attacks (CoIC-KLA security). We then construct a CoIC-KLA secure PKE scheme using 1-key Ciphertext-Policy Functional Encryption (CPFE) that in turn can be based on any IND-CPA secure PKE scheme.   3. Identity Based Encryption, Attribute Based Encryption and Functional Encryption with Secure Key Leasing: We provide definitions of secure key leasing in the context of advanced encryption schemes such as identity based encryption (IBE), attribute-based encryption (ABE) and functional encryption (FE). Then we provide constructions by combining the above PKE-SKL with standard IBE, ABE and FE schemes.



## **34. Designing a Visual Cryptography Curriculum for K-12 Education**

cs.CR

**SubmitDate**: 2023-02-22    [abs](http://arxiv.org/abs/2302.11655v1) [paper-pdf](http://arxiv.org/pdf/2302.11655v1)

**Authors**: Pranathi Rayavaram, Sreekriti Sista, Ashwin Jagadeesha, Justin Marwad, Nathan Percival, Sashank Narain, Claire Seungeun Lee

**Abstract**: We have designed and developed a simple, visual, and narrative K-12 cybersecurity curriculum leveraging the Scratch programming platform to demonstrate and teach fundamental cybersecurity concepts such as confidentiality, integrity protection, and authentication. The visual curriculum simulates a real-world scenario of a user and a bank performing a bank transaction and an adversary attempting to attack the transaction.We have designed six visual scenarios, the curriculum first introduces students to three visual scenarios demonstrating attacks that exist when systems do not integrate concepts such as confidentiality, integrity protection, and authentication. Then, it introduces them to three visual scenarios that build on the attacks to demonstrate and teach how these fundamental concepts can be used to defend against them. We conducted an evaluation of our curriculum through a study with 18 middle and high school students. To evaluate the student's comprehension of these concepts we distributed a technical survey, where overall average of students answering these questions related to the demonstrated concepts is 9.28 out of 10. Furthermore, the survey results revealed that 66.7% found the system extremely easy and the remaining 27.8% found it easy to use and understand.



## **35. Feature Partition Aggregation: A Fast Certified Defense Against a Union of Sparse Adversarial Attacks**

cs.LG

**SubmitDate**: 2023-02-22    [abs](http://arxiv.org/abs/2302.11628v1) [paper-pdf](http://arxiv.org/pdf/2302.11628v1)

**Authors**: Zayd Hammoudeh, Daniel Lowd

**Abstract**: Deep networks are susceptible to numerous types of adversarial attacks. Certified defenses provide guarantees on a model's robustness, but most of these defenses are restricted to a single attack type. In contrast, this paper proposes feature partition aggregation (FPA) - a certified defense against a union of attack types, namely evasion, backdoor, and poisoning attacks. We specifically consider an $\ell_0$ or sparse attacker that arbitrarily controls an unknown subset of the training and test features - even across all instances. FPA generates robustness guarantees via an ensemble whose submodels are trained on disjoint feature sets. Following existing certified sparse defenses, we generalize FPA's guarantees to top-$k$ predictions. FPA significantly outperforms state-of-the-art sparse defenses providing larger and stronger robustness guarantees, while simultaneously being up to 5,000${\times}$ faster.



## **36. PAD: Towards Principled Adversarial Malware Detection Against Evasion Attacks**

cs.CR

20 pages; In submission

**SubmitDate**: 2023-02-22    [abs](http://arxiv.org/abs/2302.11328v1) [paper-pdf](http://arxiv.org/pdf/2302.11328v1)

**Authors**: Deqiang Li, Shicheng Cui, Yun Li, Jia Xu, Fu Xiao, Shouhuai Xu

**Abstract**: Machine Learning (ML) techniques facilitate automating malicious software (malware for short) detection, but suffer from evasion attacks. Many researchers counter such attacks in heuristic manners short of both theoretical guarantees and defense effectiveness. We hence propose a new adversarial training framework, termed Principled Adversarial Malware Detection (PAD), which encourages convergence guarantees for robust optimization methods. PAD lays on a learnable convex measurement that quantifies distribution-wise discrete perturbations and protects the malware detector from adversaries, by which for smooth detectors, adversarial training can be performed heuristically with theoretical treatments. To promote defense effectiveness, we propose a new mixture of attacks to instantiate PAD for enhancing the deep neural network-based measurement and malware detector. Experimental results on two Android malware datasets demonstrate: (i) the proposed method significantly outperforms the state-of-the-art defenses; (ii) it can harden the ML-based malware detection against 27 evasion attacks with detection accuracies greater than 83.45%, while suffering an accuracy decrease smaller than 2.16% in the absence of attacks; (iii) it matches or outperforms many anti-malware scanners in VirusTotal service against realistic adversarial malware.



## **37. A Hitting Time Analysis for Stochastic Time-Varying Functions with Applications to Adversarial Attacks on Computation of Markov Decision Processes**

math.OC

**SubmitDate**: 2023-02-22    [abs](http://arxiv.org/abs/2302.11190v1) [paper-pdf](http://arxiv.org/pdf/2302.11190v1)

**Authors**: Ali Yekkehkhany, Han Feng, Donghao Ying, Javad Lavaei

**Abstract**: Stochastic time-varying optimization is an integral part of learning in which the shape of the function changes over time in a non-deterministic manner. This paper considers multiple models of stochastic time variation and analyzes the corresponding notion of hitting time for each model, i.e., the period after which optimizing the stochastic time-varying function reveals informative statistics on the optimization of the target function. The studied models of time variation are motivated by adversarial attacks on the computation of value iteration in Markov decision processes. In this application, the hitting time quantifies the extent that the computation is robust to adversarial disturbance. We develop upper bounds on the hitting time by analyzing the contraction-expansion transformation appeared in the time-variation models. We prove that the hitting time of the value function in the value iteration with a probabilistic contraction-expansion transformation is logarithmic in terms of the inverse of a desired precision. In addition, the hitting time is analyzed for optimization of unknown continuous or discrete time-varying functions whose noisy evaluations are revealed over time. The upper bound for a continuous function is super-quadratic (but sub-cubic) in terms of the inverse of a desired precision and the upper bound for a discrete function is logarithmic in terms of the cardinality of the function domain. Improved bounds for convex functions are obtained and we show that such functions are learned faster than non-convex functions. Finally, we study a time-varying linear model with additive noise, where hitting time is bounded with the notion of shape dominance.



## **38. MultiRobustBench: Benchmarking Robustness Against Multiple Attacks**

cs.LG

**SubmitDate**: 2023-02-21    [abs](http://arxiv.org/abs/2302.10980v1) [paper-pdf](http://arxiv.org/pdf/2302.10980v1)

**Authors**: Sihui Dai, Saeed Mahloujifar, Chong Xiang, Vikash Sehwag, Pin-Yu Chen, Prateek Mittal

**Abstract**: The bulk of existing research in defending against adversarial examples focuses on defending against a single (typically bounded Lp-norm) attack, but for a practical setting, machine learning (ML) models should be robust to a wide variety of attacks. In this paper, we present the first unified framework for considering multiple attacks against ML models. Our framework is able to model different levels of learner's knowledge about the test-time adversary, allowing us to model robustness against unforeseen attacks and robustness against unions of attacks. Using our framework, we present the first leaderboard, MultiRobustBench, for benchmarking multiattack evaluation which captures performance across attack types and attack strengths. We evaluate the performance of 16 defended models for robustness against a set of 9 different attack types, including Lp-based threat models, spatial transformations, and color changes, at 20 different attack strengths (180 attacks total). Additionally, we analyze the state of current defenses against multiple attacks. Our analysis shows that while existing defenses have made progress in terms of average robustness across the set of attacks used, robustness against the worst-case attack is still a big open problem as all existing models perform worse than random guessing.



## **39. Attacking Fake News Detectors via Manipulating News Social Engagement**

cs.SI

In Proceedings of the ACM Web Conference 2023 (WWW'23)

**SubmitDate**: 2023-02-21    [abs](http://arxiv.org/abs/2302.07363v2) [paper-pdf](http://arxiv.org/pdf/2302.07363v2)

**Authors**: Haoran Wang, Yingtong Dou, Canyu Chen, Lichao Sun, Philip S. Yu, Kai Shu

**Abstract**: Social media is one of the main sources for news consumption, especially among the younger generation. With the increasing popularity of news consumption on various social media platforms, there has been a surge of misinformation which includes false information or unfounded claims. As various text- and social context-based fake news detectors are proposed to detect misinformation on social media, recent works start to focus on the vulnerabilities of fake news detectors. In this paper, we present the first adversarial attack framework against Graph Neural Network (GNN)-based fake news detectors to probe their robustness. Specifically, we leverage a multi-agent reinforcement learning (MARL) framework to simulate the adversarial behavior of fraudsters on social media. Research has shown that in real-world settings, fraudsters coordinate with each other to share different news in order to evade the detection of fake news detectors. Therefore, we modeled our MARL framework as a Markov Game with bot, cyborg, and crowd worker agents, which have their own distinctive cost, budget, and influence. We then use deep Q-learning to search for the optimal policy that maximizes the rewards. Extensive experimental results on two real-world fake news propagation datasets demonstrate that our proposed framework can effectively sabotage the GNN-based fake news detector performance. We hope this paper can provide insights for future research on fake news detection.



## **40. MalProtect: Stateful Defense Against Adversarial Query Attacks in ML-based Malware Detection**

cs.LG

**SubmitDate**: 2023-02-21    [abs](http://arxiv.org/abs/2302.10739v1) [paper-pdf](http://arxiv.org/pdf/2302.10739v1)

**Authors**: Aqib Rashid, Jose Such

**Abstract**: ML models are known to be vulnerable to adversarial query attacks. In these attacks, queries are iteratively perturbed towards a particular class without any knowledge of the target model besides its output. The prevalence of remotely-hosted ML classification models and Machine-Learning-as-a-Service platforms means that query attacks pose a real threat to the security of these systems. To deal with this, stateful defenses have been proposed to detect query attacks and prevent the generation of adversarial examples by monitoring and analyzing the sequence of queries received by the system. Several stateful defenses have been proposed in recent years. However, these defenses rely solely on similarity or out-of-distribution detection methods that may be effective in other domains. In the malware detection domain, the methods to generate adversarial examples are inherently different, and therefore we find that such detection mechanisms are significantly less effective. Hence, in this paper, we present MalProtect, which is a stateful defense against query attacks in the malware detection domain. MalProtect uses several threat indicators to detect attacks. Our results show that it reduces the evasion rate of adversarial query attacks by 80+\% in Android and Windows malware, across a range of attacker scenarios. In the first evaluation of its kind, we show that MalProtect outperforms prior stateful defenses, especially under the peak adversarial threat.



## **41. Characterizing the Optimal 0-1 Loss for Multi-class Classification with a Test-time Attacker**

cs.LG

**SubmitDate**: 2023-02-21    [abs](http://arxiv.org/abs/2302.10722v1) [paper-pdf](http://arxiv.org/pdf/2302.10722v1)

**Authors**: Sihui Dai, Wenxin Ding, Arjun Nitin Bhagoji, Daniel Cullina, Ben Y. Zhao, Haitao Zheng, Prateek Mittal

**Abstract**: Finding classifiers robust to adversarial examples is critical for their safe deployment. Determining the robustness of the best possible classifier under a given threat model for a given data distribution and comparing it to that achieved by state-of-the-art training methods is thus an important diagnostic tool. In this paper, we find achievable information-theoretic lower bounds on loss in the presence of a test-time attacker for multi-class classifiers on any discrete dataset. We provide a general framework for finding the optimal 0-1 loss that revolves around the construction of a conflict hypergraph from the data and adversarial constraints. We further define other variants of the attacker-classifier game that determine the range of the optimal loss more efficiently than the full-fledged hypergraph construction. Our evaluation shows, for the first time, an analysis of the gap to optimal robustness for classifiers in the multi-class setting on benchmark datasets.



## **42. Interpretable Spectrum Transformation Attacks to Speaker Recognition**

cs.SD

**SubmitDate**: 2023-02-21    [abs](http://arxiv.org/abs/2302.10686v1) [paper-pdf](http://arxiv.org/pdf/2302.10686v1)

**Authors**: Jiadi Yao, Hong Luo, Xiao-Lei Zhang

**Abstract**: The success of adversarial attacks to speaker recognition is mainly in white-box scenarios. When applying the adversarial voices that are generated by attacking white-box surrogate models to black-box victim models, i.e. \textit{transfer-based} black-box attacks, the transferability of the adversarial voices is not only far from satisfactory, but also lacks interpretable basis. To address these issues, in this paper, we propose a general framework, named spectral transformation attack based on modified discrete cosine transform (STA-MDCT), to improve the transferability of the adversarial voices to a black-box victim model. Specifically, we first apply MDCT to the input voice. Then, we slightly modify the energy of different frequency bands for capturing the salient regions of the adversarial noise in the time-frequency domain that are critical to a successful attack. Unlike existing approaches that operate voices in the time domain, the proposed framework operates voices in the time-frequency domain, which improves the interpretability, transferability, and imperceptibility of the attack. Moreover, it can be implemented with any gradient-based attackers. To utilize the advantage of model ensembling, we not only implement STA-MDCT with a single white-box surrogate model, but also with an ensemble of surrogate models. Finally, we visualize the saliency maps of adversarial voices by the class activation maps (CAM), which offers an interpretable basis to transfer-based attacks in speaker recognition for the first time. Extensive comparison results with five representative attackers show that the CAM visualization clearly explains the effectiveness of STA-MDCT, and the weaknesses of the comparison methods; the proposed method outperforms the comparison methods by a large margin.



## **43. Adversarial Deep Reinforcement Learning for Improving the Robustness of Multi-agent Autonomous Driving Policies**

cs.AI

**SubmitDate**: 2023-02-21    [abs](http://arxiv.org/abs/2112.11937v3) [paper-pdf](http://arxiv.org/pdf/2112.11937v3)

**Authors**: Aizaz Sharif, Dusica Marijan

**Abstract**: Autonomous cars are well known for being vulnerable to adversarial attacks that can compromise the safety of the car and pose danger to other road users. To effectively defend against adversaries, it is required to not only test autonomous cars for finding driving errors but to improve the robustness of the cars to these errors. To this end, in this paper, we propose a two-step methodology for autonomous cars that consists of (i) finding failure states in autonomous cars by training the adversarial driving agent, and (ii) improving the robustness of autonomous cars by retraining them with effective adversarial inputs. Our methodology supports testing autonomous cars in a multi-agent environment, where we train and compare adversarial car policy on two custom reward functions to test the driving control decision of autonomous cars. We run experiments in a vision-based high-fidelity urban driving simulated environment. Our results show that adversarial testing can be used for finding erroneous autonomous driving behavior, followed by adversarial training for improving the robustness of deep reinforcement learning-based autonomous driving policies. We demonstrate that the autonomous cars retrained using the effective adversarial inputs noticeably increase the performance of their driving policies in terms of reduced collision and offroad steering errors.



## **44. CatchBackdoor: Backdoor Testing by Critical Trojan Neural Path Identification via Differential Fuzzing**

cs.CR

There are some problems in the experiment so we need to withdraw this  paper. We will upload the new version after revision

**SubmitDate**: 2023-02-21    [abs](http://arxiv.org/abs/2112.13064v2) [paper-pdf](http://arxiv.org/pdf/2112.13064v2)

**Authors**: Haibo Jin, Ruoxi Chen, Jinyin Chen, Yao Cheng, Chong Fu, Ting Wang, Yue Yu, Zhaoyan Ming

**Abstract**: The success of deep neural networks (DNNs) in real-world applications has benefited from abundant pre-trained models. However, the backdoored pre-trained models can pose a significant trojan threat to the deployment of downstream DNNs. Existing DNN testing methods are mainly designed to find incorrect corner case behaviors in adversarial settings but fail to discover the backdoors crafted by strong trojan attacks. Observing the trojan network behaviors shows that they are not just reflected by a single compromised neuron as proposed by previous work but attributed to the critical neural paths in the activation intensity and frequency of multiple neurons. This work formulates the DNN backdoor testing and proposes the CatchBackdoor framework. Via differential fuzzing of critical neurons from a small number of benign examples, we identify the trojan paths and particularly the critical ones, and generate backdoor testing examples by simulating the critical neurons in the identified paths. Extensive experiments demonstrate the superiority of CatchBackdoor, with higher detection performance than existing methods. CatchBackdoor works better on detecting backdoors by stealthy blending and adaptive attacks, which existing methods fail to detect. Moreover, our experiments show that CatchBackdoor may reveal the potential backdoors of models in Model Zoo.



## **45. A Survey of Trustworthy Federated Learning with Perspectives on Security, Robustness, and Privacy**

cs.LG

**SubmitDate**: 2023-02-21    [abs](http://arxiv.org/abs/2302.10637v1) [paper-pdf](http://arxiv.org/pdf/2302.10637v1)

**Authors**: Yifei Zhang, Dun Zeng, Jinglong Luo, Zenglin Xu, Irwin King

**Abstract**: Trustworthy artificial intelligence (AI) technology has revolutionized daily life and greatly benefited human society. Among various AI technologies, Federated Learning (FL) stands out as a promising solution for diverse real-world scenarios, ranging from risk evaluation systems in finance to cutting-edge technologies like drug discovery in life sciences. However, challenges around data isolation and privacy threaten the trustworthiness of FL systems. Adversarial attacks against data privacy, learning algorithm stability, and system confidentiality are particularly concerning in the context of distributed training in federated learning. Therefore, it is crucial to develop FL in a trustworthy manner, with a focus on security, robustness, and privacy. In this survey, we propose a comprehensive roadmap for developing trustworthy FL systems and summarize existing efforts from three key aspects: security, robustness, and privacy. We outline the threats that pose vulnerabilities to trustworthy federated learning across different stages of development, including data processing, model training, and deployment. To guide the selection of the most appropriate defense methods, we discuss specific technical solutions for realizing each aspect of Trustworthy FL (TFL). Our approach differs from previous work that primarily discusses TFL from a legal perspective or presents FL from a high-level, non-technical viewpoint.



## **46. Generalization Bounds for Adversarial Contrastive Learning**

cs.LG

**SubmitDate**: 2023-02-21    [abs](http://arxiv.org/abs/2302.10633v1) [paper-pdf](http://arxiv.org/pdf/2302.10633v1)

**Authors**: Xin Zou, Weiwei Liu

**Abstract**: Deep networks are well-known to be fragile to adversarial attacks, and adversarial training is one of the most popular methods used to train a robust model. To take advantage of unlabeled data, recent works have applied adversarial training to contrastive learning (Adversarial Contrastive Learning; ACL for short) and obtain promising robust performance. However, the theory of ACL is not well understood. To fill this gap, we leverage the Rademacher complexity to analyze the generalization performance of ACL, with a particular focus on linear models and multi-layer neural networks under $\ell_p$ attack ($p \ge 1$). Our theory shows that the average adversarial risk of the downstream tasks can be upper bounded by the adversarial unsupervised risk of the upstream task. The experimental results validate our theory.



## **47. Certified Defences Against Adversarial Patch Attacks on Semantic Segmentation**

cs.CV

accepted at ICLR 2023

**SubmitDate**: 2023-02-21    [abs](http://arxiv.org/abs/2209.05980v2) [paper-pdf](http://arxiv.org/pdf/2209.05980v2)

**Authors**: Maksym Yatsura, Kaspar Sakmann, N. Grace Hua, Matthias Hein, Jan Hendrik Metzen

**Abstract**: Adversarial patch attacks are an emerging security threat for real world deep learning applications. We present Demasked Smoothing, the first approach (up to our knowledge) to certify the robustness of semantic segmentation models against this threat model. Previous work on certifiably defending against patch attacks has mostly focused on image classification task and often required changes in the model architecture and additional training which is undesirable and computationally expensive. In Demasked Smoothing, any segmentation model can be applied without particular training, fine-tuning, or restriction of the architecture. Using different masking strategies, Demasked Smoothing can be applied both for certified detection and certified recovery. In extensive experiments we show that Demasked Smoothing can on average certify 64% of the pixel predictions for a 1% patch in the detection task and 48% against a 0.5% patch for the recovery task on the ADE20K dataset.



## **48. Internal Wasserstein Distance for Adversarial Attack and Defense**

cs.LG

**SubmitDate**: 2023-02-21    [abs](http://arxiv.org/abs/2103.07598v4) [paper-pdf](http://arxiv.org/pdf/2103.07598v4)

**Authors**: Qicheng Wang, Shuhai Zhang, Jiezhang Cao, Jincheng Li, Mingkui Tan, Yang Xiang

**Abstract**: Deep neural networks (DNNs) are known to be vulnerable to adversarial attacks that would trigger misclassification of DNNs but may be imperceptible to human perception. Adversarial defense has been an important way to improve the robustness of DNNs. Existing attack methods often construct adversarial examples relying on some metrics like the $\ell_p$ distance to perturb samples. However, these metrics can be insufficient to conduct adversarial attacks due to their limited perturbations. In this paper, we propose a new internal Wasserstein distance (IWD) to capture the semantic similarity of two samples, and thus it helps to obtain larger perturbations than currently used metrics such as the $\ell_p$ distance. We then apply the internal Wasserstein distance to perform adversarial attack and defense. In particular, we develop a novel attack method relying on IWD to calculate the similarities between an image and its adversarial examples. In this way, we can generate diverse and semantically similar adversarial examples that are more difficult to defend by existing defense methods. Moreover, we devise a new defense method relying on IWD to learn robust models against unseen adversarial examples. We provide both thorough theoretical and empirical evidence to support our methods.



## **49. Model-based feature selection for neural networks: A mixed-integer programming approach**

math.OC

15 pages, 3 figures, 5 tables

**SubmitDate**: 2023-02-20    [abs](http://arxiv.org/abs/2302.10344v1) [paper-pdf](http://arxiv.org/pdf/2302.10344v1)

**Authors**: Shudian Zhao, Calvin Tsay, Jan Kronqvist

**Abstract**: In this work, we develop a novel input feature selection framework for ReLU-based deep neural networks (DNNs), which builds upon a mixed-integer optimization approach. While the method is generally applicable to various classification tasks, we focus on finding input features for image classification for clarity of presentation. The idea is to use a trained DNN, or an ensemble of trained DNNs, to identify the salient input features. The input feature selection is formulated as a sequence of mixed-integer linear programming (MILP) problems that find sets of sparse inputs that maximize the classification confidence of each category. These ''inverse'' problems are regularized by the number of inputs selected for each category and by distribution constraints. Numerical results on the well-known MNIST and FashionMNIST datasets show that the proposed input feature selection allows us to drastically reduce the size of the input to $\sim$15\% while maintaining a good classification accuracy. This allows us to design DNNs with significantly fewer connections, reducing computational effort and producing DNNs that are more robust towards adversarial attacks.



## **50. Robust Fair Clustering: A Novel Fairness Attack and Defense Framework**

cs.LG

Accepted to the 11th International Conference on Learning  Representations (ICLR 2023)

**SubmitDate**: 2023-02-20    [abs](http://arxiv.org/abs/2210.01953v3) [paper-pdf](http://arxiv.org/pdf/2210.01953v3)

**Authors**: Anshuman Chhabra, Peizhao Li, Prasant Mohapatra, Hongfu Liu

**Abstract**: Clustering algorithms are widely used in many societal resource allocation applications, such as loan approvals and candidate recruitment, among others, and hence, biased or unfair model outputs can adversely impact individuals that rely on these applications. To this end, many fair clustering approaches have been recently proposed to counteract this issue. Due to the potential for significant harm, it is essential to ensure that fair clustering algorithms provide consistently fair outputs even under adversarial influence. However, fair clustering algorithms have not been studied from an adversarial attack perspective. In contrast to previous research, we seek to bridge this gap and conduct a robustness analysis against fair clustering by proposing a novel black-box fairness attack. Through comprehensive experiments, we find that state-of-the-art models are highly susceptible to our attack as it can reduce their fairness performance significantly. Finally, we propose Consensus Fair Clustering (CFC), the first robust fair clustering approach that transforms consensus clustering into a fair graph partitioning problem, and iteratively learns to generate fair cluster outputs. Experimentally, we observe that CFC is highly robust to the proposed attack and is thus a truly robust fair clustering alternative.



