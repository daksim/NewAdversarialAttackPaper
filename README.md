# Latest Adversarial Attack Papers
**update at 2022-04-10 06:31:30**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Reinforcement Learning for Linear Quadratic Control is Vulnerable Under Cost Manipulation**

eess.SY

This paper is yet to be peer-reviewed; Typos are corrected in ver 2

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2203.05774v2)

**Authors**: Yunhan Huang, Quanyan Zhu

**Abstracts**: In this work, we study the deception of a Linear-Quadratic-Gaussian (LQG) agent by manipulating the cost signals. We show that a small falsification of the cost parameters will only lead to a bounded change in the optimal policy. The bound is linear on the amount of falsification the attacker can apply to the cost parameters. We propose an attack model where the attacker aims to mislead the agent into learning a `nefarious' policy by intentionally falsifying the cost parameters. We formulate the attack's problem as a convex optimization problem and develop necessary and sufficient conditions to check the achievability of the attacker's goal.   We showcase the adversarial manipulation on two types of LQG learners: the batch RL learner and the other is the adaptive dynamic programming (ADP) learner. Our results demonstrate that with only 2.296% of falsification on the cost data, the attacker misleads the batch RL into learning the 'nefarious' policy that leads the vehicle to a dangerous position. The attacker can also gradually trick the ADP learner into learning the same `nefarious' policy by consistently feeding the learner a falsified cost signal that stays close to the actual cost signal. The paper aims to raise people's awareness of the security threats faced by RL-enabled control systems.



## **2. IRShield: A Countermeasure Against Adversarial Physical-Layer Wireless Sensing**

cs.CR

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2112.01967v2)

**Authors**: Paul Staat, Simon Mulzer, Stefan Roth, Veelasha Moonsamy, Markus Heinrichs, Rainer Kronberger, Aydin Sezgin, Christof Paar

**Abstracts**: Wireless radio channels are known to contain information about the surrounding propagation environment, which can be extracted using established wireless sensing methods. Thus, today's ubiquitous wireless devices are attractive targets for passive eavesdroppers to launch reconnaissance attacks. In particular, by overhearing standard communication signals, eavesdroppers obtain estimations of wireless channels which can give away sensitive information about indoor environments. For instance, by applying simple statistical methods, adversaries can infer human motion from wireless channel observations, allowing to remotely monitor premises of victims. In this work, building on the advent of intelligent reflecting surfaces (IRSs), we propose IRShield as a novel countermeasure against adversarial wireless sensing. IRShield is designed as a plug-and-play privacy-preserving extension to existing wireless networks. At the core of IRShield, we design an IRS configuration algorithm to obfuscate wireless channels. We validate the effectiveness with extensive experimental evaluations. In a state-of-the-art human motion detection attack using off-the-shelf Wi-Fi devices, IRShield lowered detection rates to 5% or less.



## **3. Adversarial Machine Learning Attacks Against Video Anomaly Detection Systems**

cs.CV

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2204.03141v1)

**Authors**: Furkan Mumcu, Keval Doshi, Yasin Yilmaz

**Abstracts**: Anomaly detection in videos is an important computer vision problem with various applications including automated video surveillance. Although adversarial attacks on image understanding models have been heavily investigated, there is not much work on adversarial machine learning targeting video understanding models and no previous work which focuses on video anomaly detection. To this end, we investigate an adversarial machine learning attack against video anomaly detection systems, that can be implemented via an easy-to-perform cyber-attack. Since surveillance cameras are usually connected to the server running the anomaly detection model through a wireless network, they are prone to cyber-attacks targeting the wireless connection. We demonstrate how Wi-Fi deauthentication attack, a notoriously easy-to-perform and effective denial-of-service (DoS) attack, can be utilized to generate adversarial data for video anomaly detection systems. Specifically, we apply several effects caused by the Wi-Fi deauthentication attack on video quality (e.g., slow down, freeze, fast forward, low resolution) to the popular benchmark datasets for video anomaly detection. Our experiments with several state-of-the-art anomaly detection models show that the attackers can significantly undermine the reliability of video anomaly detection systems by causing frequent false alarms and hiding physical anomalies from the surveillance system.



## **4. Control barrier function based attack-recovery with provable guarantees**

cs.SY

8 pages, 6 figures

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.03077v1)

**Authors**: Kunal Garg, Ricardo G. Sanfelice, Alvaro A. Cardenas

**Abstracts**: This paper studies provable security guarantees for cyber-physical systems (CPS) under actuator attacks. In particular, we consider CPS safety and propose a new attack-detection mechanism based on a zeroing control barrier function (ZCBF) condition. In addition we design an adaptive recovery mechanism based on how close the system is from violating safety. We show that the attack-detection mechanism is sound, i.e., there are no false negatives for adversarial attacks. Finally, we use a Quadratic Programming (QP) approach for online recovery (and nominal) control synthesis. We demonstrate the effectiveness of the proposed method in a simulation case study involving a quadrotor with an attack on its motors.



## **5. Sampling-based Fast Gradient Rescaling Method for Highly Transferable Adversarial Attacks**

cs.CV

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.02887v1)

**Authors**: Xu Han, Anmin Liu, Yifeng Xiong, Yanbo Fan, Kun He

**Abstracts**: Deep neural networks have shown to be very vulnerable to adversarial examples crafted by adding human-imperceptible perturbations to benign inputs. After achieving impressive attack success rates in the white-box setting, more focus is shifted to black-box attacks. In either case, the common gradient-based approaches generally use the $sign$ function to generate perturbations at the end of the process. However, only a few works pay attention to the limitation of the $sign$ function. Deviation between the original gradient and the generated noises may lead to inaccurate gradient update estimation and suboptimal solutions for adversarial transferability, which is crucial for black-box attacks. To address this issue, we propose a Sampling-based Fast Gradient Rescaling Method (S-FGRM) to improve the transferability of the crafted adversarial examples. Specifically, we use data rescaling to substitute the inefficient $sign$ function in gradient-based attacks without extra computational cost. We also propose a Depth First Sampling method to eliminate the fluctuation of rescaling and stabilize the gradient update. Our method can be used in any gradient-based optimizations and is extensible to be integrated with various input transformation or ensemble methods for further improving the adversarial transferability. Extensive experiments on the standard ImageNet dataset show that our S-FGRM could significantly boost the transferability of gradient-based attacks and outperform the state-of-the-art baselines.



## **6. Distilling Robust and Non-Robust Features in Adversarial Examples by Information Bottleneck**

cs.LG

NeurIPS 2021

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.02735v1)

**Authors**: Junho Kim, Byung-Kwan Lee, Yong Man Ro

**Abstracts**: Adversarial examples, generated by carefully crafted perturbation, have attracted considerable attention in research fields. Recent works have argued that the existence of the robust and non-robust features is a primary cause of the adversarial examples, and investigated their internal interactions in the feature space. In this paper, we propose a way of explicitly distilling feature representation into the robust and non-robust features, using Information Bottleneck. Specifically, we inject noise variation to each feature unit and evaluate the information flow in the feature representation to dichotomize feature units either robust or non-robust, based on the noise variation magnitude. Through comprehensive experiments, we demonstrate that the distilled features are highly correlated with adversarial prediction, and they have human-perceptible semantic information by themselves. Furthermore, we present an attack mechanism intensifying the gradient of non-robust features that is directly related to the model prediction, and validate its effectiveness of breaking model robustness.



## **7. Rolling Colors: Adversarial Laser Exploits against Traffic Light Recognition**

cs.CV

To be published in USENIX Security 2022

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.02675v1)

**Authors**: Chen Yan, Zhijian Xu, Zhanyuan Yin, Xiaoyu Ji, Wenyuan Xu

**Abstracts**: Traffic light recognition is essential for fully autonomous driving in urban areas. In this paper, we investigate the feasibility of fooling traffic light recognition mechanisms by shedding laser interference on the camera. By exploiting the rolling shutter of CMOS sensors, we manage to inject a color stripe overlapped on the traffic light in the image, which can cause a red light to be recognized as a green light or vice versa. To increase the success rate, we design an optimization method to search for effective laser parameters based on empirical models of laser interference. Our evaluation in emulated and real-world setups on 2 state-of-the-art recognition systems and 5 cameras reports a maximum success rate of 30% and 86.25% for Red-to-Green and Green-to-Red attacks. We observe that the attack is effective in continuous frames from more than 40 meters away against a moving vehicle, which may cause end-to-end impacts on self-driving such as running a red light or emergency stop. To mitigate the threat, we propose redesigning the rolling shutter mechanism.



## **8. Adversarial Analysis of the Differentially-Private Federated Learning in Cyber-Physical Critical Infrastructures**

cs.CR

11 pages, 5 figures, 4 tables. This work has been submitted to IEEE  for possible publication. Copyright may be transferred without notice, after  which this version may no longer be accessible

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.02654v1)

**Authors**: Md Tamjid Hossain, Shahriar Badsha, Hung, La, Haoting Shen, Shafkat Islam, Ibrahim Khalil, Xun Yi

**Abstracts**: Differential privacy (DP) is considered to be an effective privacy-preservation method to secure the promising distributed machine learning (ML) paradigm-federated learning (FL) from privacy attacks (e.g., membership inference attack). Nevertheless, while the DP mechanism greatly alleviates privacy concerns, recent studies have shown that it can be exploited to conduct security attacks (e.g., false data injection attacks). To address such attacks on FL-based applications in critical infrastructures, in this paper, we perform the first systematic study on the DP-exploited poisoning attacks from an adversarial point of view. We demonstrate that the DP method, despite providing a level of privacy guarantee, can effectively open a new poisoning attack vector for the adversary. Our theoretical analysis and empirical evaluation of a smart grid dataset show the FL performance degradation (sub-optimal model generation) scenario due to the differential noise-exploited selective model poisoning attacks. As a countermeasure, we propose a reinforcement learning-based differential privacy level selection (rDP) process. The rDP process utilizes the differential privacy parameters (privacy loss, information leakage probability, etc.) and the losses to intelligently generate an optimal privacy level for the nodes. The evaluation shows the accumulated reward and errors of the proposed technique converge to an optimal privacy policy.



## **9. Who Is the Strongest Enemy? Towards Optimal and Efficient Evasion Attacks in Deep RL**

cs.LG

In the 10th International Conference on Learning Representations  (ICLR 2022)

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2106.05087v4)

**Authors**: Yanchao Sun, Ruijie Zheng, Yongyuan Liang, Furong Huang

**Abstracts**: Evaluating the worst-case performance of a reinforcement learning (RL) agent under the strongest/optimal adversarial perturbations on state observations (within some constraints) is crucial for understanding the robustness of RL agents. However, finding the optimal adversary is challenging, in terms of both whether we can find the optimal attack and how efficiently we can find it. Existing works on adversarial RL either use heuristics-based methods that may not find the strongest adversary, or directly train an RL-based adversary by treating the agent as a part of the environment, which can find the optimal adversary but may become intractable in a large state space. This paper introduces a novel attacking method to find the optimal attacks through collaboration between a designed function named "actor" and an RL-based learner named "director". The actor crafts state perturbations for a given policy perturbation direction, and the director learns to propose the best policy perturbation directions. Our proposed algorithm, PA-AD, is theoretically optimal and significantly more efficient than prior RL-based works in environments with large state spaces. Empirical results show that our proposed PA-AD universally outperforms state-of-the-art attacking methods in various Atari and MuJoCo environments. By applying PA-AD to adversarial training, we achieve state-of-the-art empirical robustness in multiple tasks under strong adversaries.



## **10. Patch-Fool: Are Vision Transformers Always Robust Against Adversarial Perturbations?**

cs.CV

Accepted at ICLR 2022

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2203.08392v2)

**Authors**: Yonggan Fu, Shunyao Zhang, Shang Wu, Cheng Wan, Yingyan Lin

**Abstracts**: Vision transformers (ViTs) have recently set off a new wave in neural architecture design thanks to their record-breaking performance in various vision tasks. In parallel, to fulfill the goal of deploying ViTs into real-world vision applications, their robustness against potential malicious attacks has gained increasing attention. In particular, recent works show that ViTs are more robust against adversarial attacks as compared with convolutional neural networks (CNNs), and conjecture that this is because ViTs focus more on capturing global interactions among different input/feature patches, leading to their improved robustness to local perturbations imposed by adversarial attacks. In this work, we ask an intriguing question: "Under what kinds of perturbations do ViTs become more vulnerable learners compared to CNNs?" Driven by this question, we first conduct a comprehensive experiment regarding the robustness of both ViTs and CNNs under various existing adversarial attacks to understand the underlying reason favoring their robustness. Based on the drawn insights, we then propose a dedicated attack framework, dubbed Patch-Fool, that fools the self-attention mechanism by attacking its basic component (i.e., a single patch) with a series of attention-aware optimization techniques. Interestingly, our Patch-Fool framework shows for the first time that ViTs are not necessarily more robust than CNNs against adversarial perturbations. In particular, we find that ViTs are more vulnerable learners compared with CNNs against our Patch-Fool attack which is consistent across extensive experiments, and the observations from Sparse/Mild Patch-Fool, two variants of Patch-Fool, indicate an intriguing insight that the perturbation density and strength on each patch seem to be the key factors that influence the robustness ranking between ViTs and CNNs.



## **11. Exploring Robust Architectures for Deep Artificial Neural Networks**

cs.LG

27 pages, 16 figures

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2106.15850v2)

**Authors**: Asim Waqas, Ghulam Rasool, Hamza Farooq, Nidhal C. Bouaynaya

**Abstracts**: The architectures of deep artificial neural networks (DANNs) are routinely studied to improve their predictive performance. However, the relationship between the architecture of a DANN and its robustness to noise and adversarial attacks is less explored. We investigate how the robustness of DANNs relates to their underlying graph architectures or structures. This study: (1) starts by exploring the design space of architectures of DANNs using graph-theoretic robustness measures; (2) transforms the graphs to DANN architectures to train/validate/test on various image classification tasks; (3) explores the relationship between the robustness of trained DANNs against noise and adversarial attacks and the robustness of their underlying architectures estimated via graph-theoretic measures. We show that the topological entropy and Olivier-Ricci curvature of the underlying graphs can quantify the robustness performance of DANNs. The said relationship is stronger for complex tasks and large DANNs. Our work will allow autoML and neural architecture search community to explore design spaces of robust and accurate DANNs.



## **12. User-Level Differential Privacy against Attribute Inference Attack of Speech Emotion Recognition in Federated Learning**

cs.CR

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2204.02500v1)

**Authors**: Tiantian Feng, Raghuveer Peri, Shrikanth Narayanan

**Abstracts**: Many existing privacy-enhanced speech emotion recognition (SER) frameworks focus on perturbing the original speech data through adversarial training within a centralized machine learning setup. However, this privacy protection scheme can fail since the adversary can still access the perturbed data. In recent years, distributed learning algorithms, especially federated learning (FL), have gained popularity to protect privacy in machine learning applications. While FL provides good intuition to safeguard privacy by keeping the data on local devices, prior work has shown that privacy attacks, such as attribute inference attacks, are achievable for SER systems trained using FL. In this work, we propose to evaluate the user-level differential privacy (UDP) in mitigating the privacy leaks of the SER system in FL. UDP provides theoretical privacy guarantees with privacy parameters $\epsilon$ and $\delta$. Our results show that the UDP can effectively decrease attribute information leakage while keeping the utility of the SER system with the adversary accessing one model update. However, the efficacy of the UDP suffers when the FL system leaks more model updates to the adversary. We make the code publicly available to reproduce the results in https://github.com/usc-sail/fed-ser-leakage.



## **13. Training-Free Robust Multimodal Learning via Sample-Wise Jacobian Regularization**

cs.CV

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2204.02485v1)

**Authors**: Zhengqi Gao, Sucheng Ren, Zihui Xue, Siting Li, Hang Zhao

**Abstracts**: Multimodal fusion emerges as an appealing technique to improve model performances on many tasks. Nevertheless, the robustness of such fusion methods is rarely involved in the present literature. In this paper, we propose a training-free robust late-fusion method by exploiting conditional independence assumption and Jacobian regularization. Our key is to minimize the Frobenius norm of a Jacobian matrix, where the resulting optimization problem is relaxed to a tractable Sylvester equation. Furthermore, we provide a theoretical error bound of our method and some insights about the function of the extra modality. Several numerical experiments on AV-MNIST, RAVDESS, and VGGsound demonstrate the efficacy of our method under both adversarial attacks and random corruptions.



## **14. Hear No Evil: Towards Adversarial Robustness of Automatic Speech Recognition via Multi-Task Learning**

eess.AS

Submitted to Insterspeech 2022

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2204.02381v1)

**Authors**: Nilaksh Das, Duen Horng Chau

**Abstracts**: As automatic speech recognition (ASR) systems are now being widely deployed in the wild, the increasing threat of adversarial attacks raises serious questions about the security and reliability of using such systems. On the other hand, multi-task learning (MTL) has shown success in training models that can resist adversarial attacks in the computer vision domain. In this work, we investigate the impact of performing such multi-task learning on the adversarial robustness of ASR models in the speech domain. We conduct extensive MTL experimentation by combining semantically diverse tasks such as accent classification and ASR, and evaluate a wide range of adversarial settings. Our thorough analysis reveals that performing MTL with semantically diverse tasks consistently makes it harder for an adversarial attack to succeed. We also discuss in detail the serious pitfalls and their related remedies that have a significant impact on the robustness of MTL models. Our proposed MTL approach shows considerable absolute improvements in adversarially targeted WER ranging from 17.25 up to 59.90 compared to single-task learning baselines (attention decoder and CTC respectively). Ours is the first in-depth study that uncovers adversarial robustness gains from multi-task learning for ASR.



## **15. A Survey of Adversarial Learning on Graphs**

cs.LG

Preprint; 16 pages, 2 figures

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2003.05730v3)

**Authors**: Liang Chen, Jintang Li, Jiaying Peng, Tao Xie, Zengxu Cao, Kun Xu, Xiangnan He, Zibin Zheng, Bingzhe Wu

**Abstracts**: Deep learning models on graphs have achieved remarkable performance in various graph analysis tasks, e.g., node classification, link prediction, and graph clustering. However, they expose uncertainty and unreliability against the well-designed inputs, i.e., adversarial examples. Accordingly, a line of studies has emerged for both attack and defense addressed in different graph analysis tasks, leading to the arms race in graph adversarial learning. Despite the booming works, there still lacks a unified problem definition and a comprehensive review. To bridge this gap, we investigate and summarize the existing works on graph adversarial learning tasks systemically. Specifically, we survey and unify the existing works w.r.t. attack and defense in graph analysis tasks, and give appropriate definitions and taxonomies at the same time. Besides, we emphasize the importance of related evaluation metrics, investigate and summarize them comprehensively. Hopefully, our works can provide a comprehensive overview and offer insights for the relevant researchers. Latest advances in graph adversarial learning are summarized in our GitHub repository https://github.com/EdisonLeeeee/Graph-Adversarial-Learning.



## **16. Training strategy for a lightweight countermeasure model for automatic speaker verification**

cs.SD

ASVspoof2021

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2203.17031v2)

**Authors**: Yen-Lun Liao, Xuanjun Chen, Chung-Che Wang, Jyh-Shing Roger Jang

**Abstracts**: The countermeasure (CM) model is developed to protect Automatic Speaker Verification (ASV) systems from spoof attacks and prevent resulting personal information leakage. Based on practicality and security considerations, the CM model is usually deployed on edge devices, which have more limited computing resources and storage space than cloud-based systems. This work proposes training strategies for a lightweight CM model for ASV, using generalized end-to-end (GE2E) pre-training and adversarial fine-tuning to improve performance, and applying knowledge distillation (KD) to reduce the size of the CM model. In the evaluation phase of the ASVspoof 2021 Logical Access task, the lightweight ResNetSE model reaches min t-DCF 0.2695 and EER 3.54%. Compared to the teacher model, the lightweight student model only uses 22.5% of parameters and 21.1% of multiply and accumulate operands of the teacher model.



## **17. Understanding and Improving Graph Injection Attack by Promoting Unnoticeability**

cs.LG

ICLR2022, 42 pages, 22 figures

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2202.08057v2)

**Authors**: Yongqiang Chen, Han Yang, Yonggang Zhang, Kaili Ma, Tongliang Liu, Bo Han, James Cheng

**Abstracts**: Recently Graph Injection Attack (GIA) emerges as a practical attack scenario on Graph Neural Networks (GNNs), where the adversary can merely inject few malicious nodes instead of modifying existing nodes or edges, i.e., Graph Modification Attack (GMA). Although GIA has achieved promising results, little is known about why it is successful and whether there is any pitfall behind the success. To understand the power of GIA, we compare it with GMA and find that GIA can be provably more harmful than GMA due to its relatively high flexibility. However, the high flexibility will also lead to great damage to the homophily distribution of the original graph, i.e., similarity among neighbors. Consequently, the threats of GIA can be easily alleviated or even prevented by homophily-based defenses designed to recover the original homophily. To mitigate the issue, we introduce a novel constraint -- homophily unnoticeability that enforces GIA to preserve the homophily, and propose Harmonious Adversarial Objective (HAO) to instantiate it. Extensive experiments verify that GIA with HAO can break homophily-based defenses and outperform previous GIA attacks by a significant margin. We believe our methods can serve for a more reliable evaluation of the robustness of GNNs.



## **18. Adversarial Detection without Model Information**

cs.CV

This paper has 14 pages of content and 2 pages of references

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2202.04271v2)

**Authors**: Abhishek Moitra, Youngeun Kim, Priyadarshini Panda

**Abstracts**: Prior state-of-the-art adversarial detection works are classifier model dependent, i.e., they require classifier model outputs and parameters for training the detector or during adversarial detection. This makes their detection approach classifier model specific. Furthermore, classifier model outputs and parameters might not always be accessible. To this end, we propose a classifier model independent adversarial detection method using a simple energy function to distinguish between adversarial and natural inputs. We train a standalone detector independent of the classifier model, with a layer-wise energy separation (LES) training to increase the separation between natural and adversarial energies. With this, we perform energy distribution-based adversarial detection. Our method achieves comparable performance with state-of-the-art detection works (ROC-AUC > 0.9) across a wide range of gradient, score and gaussian noise attacks on CIFAR10, CIFAR100 and TinyImagenet datasets. Furthermore, compared to prior works, our detection approach is light-weight, requires less amount of training data (40% of the actual dataset) and is transferable across different datasets. For reproducibility, we provide layer-wise energy separation training code at https://github.com/Intelligent-Computing-Lab-Yale/Energy-Separation-Training



## **19. GAIL-PT: A Generic Intelligent Penetration Testing Framework with Generative Adversarial Imitation Learning**

cs.CR

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2204.01975v1)

**Authors**: Jinyin Chen, Shulong Hu, Haibin Zheng, Changyou Xing, Guomin Zhang

**Abstracts**: Penetration testing (PT) is an efficient network testing and vulnerability mining tool by simulating a hacker's attack for valuable information applied in some areas. Compared with manual PT, intelligent PT has become a dominating mainstream due to less time-consuming and lower labor costs. Unfortunately, RL-based PT is still challenged in real exploitation scenarios because the agent's action space is usually high-dimensional discrete, thus leading to algorithm convergence difficulty. Besides, most PT methods still rely on the decisions of security experts. Addressing the challenges, for the first time, we introduce expert knowledge to guide the agent to make better decisions in RL-based PT and propose a Generative Adversarial Imitation Learning-based generic intelligent Penetration testing framework, denoted as GAIL-PT, to solve the problems of higher labor costs due to the involvement of security experts and high-dimensional discrete action space. Specifically, first, we manually collect the state-action pairs to construct an expert knowledge base when the pre-trained RL / DRL model executes successful penetration testings. Second, we input the expert knowledge and the state-action pairs generated online by the different RL / DRL models into the discriminator of GAIL for training. At last, we apply the output reward of the discriminator to guide the agent to perform the action with a higher penetration success rate to improve PT's performance. Extensive experiments conducted on the real target host and simulated network scenarios show that GAIL-PT achieves the SOTA penetration performance against DeepExploit in exploiting actual target Metasploitable2 and Q-learning in optimizing penetration path, not only in small-scale with or without honey-pot network environments but also in the large-scale virtual network environment.



## **20. Recent improvements of ASR models in the face of adversarial attacks**

cs.CR

Submitted to Interspeech 2022

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2203.16536v2)

**Authors**: Raphael Olivier, Bhiksha Raj

**Abstracts**: Like many other tasks involving neural networks, Speech Recognition models are vulnerable to adversarial attacks. However recent research has pointed out differences between attacks and defenses on ASR models compared to image models. Improving the robustness of ASR models requires a paradigm shift from evaluating attacks on one or a few models to a systemic approach in evaluation. We lay the ground for such research by evaluating on various architectures a representative set of adversarial attacks: targeted and untargeted, optimization and speech processing-based, white-box, black-box and targeted attacks. Our results show that the relative strengths of different attack algorithms vary considerably when changing the model architecture, and that the results of some attacks are not to be blindly trusted. They also indicate that training choices such as self-supervised pretraining can significantly impact robustness by enabling transferable perturbations. We release our source code as a package that should help future research in evaluating their attacks and defenses.



## **21. Experimental quantum adversarial learning with programmable superconducting qubits**

quant-ph

26 pages, 17 figures, 8 algorithms

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2204.01738v1)

**Authors**: Wenhui Ren, Weikang Li, Shibo Xu, Ke Wang, Wenjie Jiang, Feitong Jin, Xuhao Zhu, Jiachen Chen, Zixuan Song, Pengfei Zhang, Hang Dong, Xu Zhang, Jinfeng Deng, Yu Gao, Chuanyu Zhang, Yaozu Wu, Bing Zhang, Qiujiang Guo, Hekang Li, Zhen Wang, Jacob Biamonte, Chao Song, Dong-Ling Deng, H. Wang

**Abstracts**: Quantum computing promises to enhance machine learning and artificial intelligence. Different quantum algorithms have been proposed to improve a wide spectrum of machine learning tasks. Yet, recent theoretical works show that, similar to traditional classifiers based on deep classical neural networks, quantum classifiers would suffer from the vulnerability problem: adding tiny carefully-crafted perturbations to the legitimate original data samples would facilitate incorrect predictions at a notably high confidence level. This will pose serious problems for future quantum machine learning applications in safety and security-critical scenarios. Here, we report the first experimental demonstration of quantum adversarial learning with programmable superconducting qubits. We train quantum classifiers, which are built upon variational quantum circuits consisting of ten transmon qubits featuring average lifetimes of 150 $\mu$s, and average fidelities of simultaneous single- and two-qubit gates above 99.94% and 99.4% respectively, with both real-life images (e.g., medical magnetic resonance imaging scans) and quantum data. We demonstrate that these well-trained classifiers (with testing accuracy up to 99%) can be practically deceived by small adversarial perturbations, whereas an adversarial training process would significantly enhance their robustness to such perturbations. Our results reveal experimentally a crucial vulnerability aspect of quantum learning systems under adversarial scenarios and demonstrate an effective defense strategy against adversarial attacks, which provide a valuable guide for quantum artificial intelligence applications with both near-term and future quantum devices.



## **22. DAD: Data-free Adversarial Defense at Test Time**

cs.LG

WACV 2022. Project page: https://sites.google.com/view/dad-wacv22

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2204.01568v1)

**Authors**: Gaurav Kumar Nayak, Ruchit Rawal, Anirban Chakraborty

**Abstracts**: Deep models are highly susceptible to adversarial attacks. Such attacks are carefully crafted imperceptible noises that can fool the network and can cause severe consequences when deployed. To encounter them, the model requires training data for adversarial training or explicit regularization-based techniques. However, privacy has become an important concern, restricting access to only trained models but not the training data (e.g. biometric data). Also, data curation is expensive and companies may have proprietary rights over it. To handle such situations, we propose a completely novel problem of 'test-time adversarial defense in absence of training data and even their statistics'. We solve it in two stages: a) detection and b) correction of adversarial samples. Our adversarial sample detection framework is initially trained on arbitrary data and is subsequently adapted to the unlabelled test data through unsupervised domain adaptation. We further correct the predictions on detected adversarial samples by transforming them in Fourier domain and obtaining their low frequency component at our proposed suitable radius for model prediction. We demonstrate the efficacy of our proposed technique via extensive experiments against several adversarial attacks and for different model architectures and datasets. For a non-robust Resnet-18 model pre-trained on CIFAR-10, our detection method correctly identifies 91.42% adversaries. Also, we significantly improve the adversarial accuracy from 0% to 37.37% with a minimal drop of 0.02% in clean accuracy on state-of-the-art 'Auto Attack' without having to retrain the model.



## **23. RobustSense: Defending Adversarial Attack for Secure Device-Free Human Activity Recognition**

cs.CR

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2204.01560v1)

**Authors**: Jianfei Yang, Han Zou, Lihua Xie

**Abstracts**: Deep neural networks have empowered accurate device-free human activity recognition, which has wide applications. Deep models can extract robust features from various sensors and generalize well even in challenging situations such as data-insufficient cases. However, these systems could be vulnerable to input perturbations, i.e. adversarial attacks. We empirically demonstrate that both black-box Gaussian attacks and modern adversarial white-box attacks can render their accuracies to plummet. In this paper, we firstly point out that such phenomenon can bring severe safety hazards to device-free sensing systems, and then propose a novel learning framework, RobustSense, to defend common attacks. RobustSense aims to achieve consistent predictions regardless of whether there exists an attack on its input or not, alleviating the negative effect of distribution perturbation caused by adversarial attacks. Extensive experiments demonstrate that our proposed method can significantly enhance the model robustness of existing deep models, overcoming possible attacks. The results validate that our method works well on wireless human activity recognition and person identification systems. To the best of our knowledge, this is the first work to investigate adversarial attacks and further develop a novel defense framework for wireless human activity recognition in mobile computing research.



## **24. PRADA: Practical Black-Box Adversarial Attacks against Neural Ranking Models**

cs.IR

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2204.01321v1)

**Authors**: Chen Wu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstracts**: Neural ranking models (NRMs) have shown remarkable success in recent years, especially with pre-trained language models. However, deep neural models are notorious for their vulnerability to adversarial examples. Adversarial attacks may become a new type of web spamming technique given our increased reliance on neural information retrieval models. Therefore, it is important to study potential adversarial attacks to identify vulnerabilities of NRMs before they are deployed.   In this paper, we introduce the Adversarial Document Ranking Attack (ADRA) task against NRMs, which aims to promote a target document in rankings by adding adversarial perturbations to its text. We focus on the decision-based black-box attack setting, where the attackers have no access to the model parameters and gradients, but can only acquire the rank positions of the partial retrieved list by querying the target model. This attack setting is realistic in real-world search engines. We propose a novel Pseudo Relevance-based ADversarial ranking Attack method (PRADA) that learns a surrogate model based on Pseudo Relevance Feedback (PRF) to generate gradients for finding the adversarial perturbations.   Experiments on two web search benchmark datasets show that PRADA can outperform existing attack strategies and successfully fool the NRM with small indiscernible perturbations of text.



## **25. Captcha Attack: Turning Captchas Against Humanity**

cs.CR

Currently under submission

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2201.04014v3)

**Authors**: Mauro Conti, Luca Pajola, Pier Paolo Tricomi

**Abstracts**: Nowadays, people generate and share massive content on online platforms (e.g., social networks, blogs). In 2021, the 1.9 billion daily active Facebook users posted around 150 thousand photos every minute. Content moderators constantly monitor these online platforms to prevent the spreading of inappropriate content (e.g., hate speech, nudity images). Based on deep learning (DL) advances, Automatic Content Moderators (ACM) help human moderators handle high data volume. Despite their advantages, attackers can exploit weaknesses of DL components (e.g., preprocessing, model) to affect their performance. Therefore, an attacker can leverage such techniques to spread inappropriate content by evading ACM.   In this work, we propose CAPtcha Attack (CAPA), an adversarial technique that allows users to spread inappropriate text online by evading ACM controls. CAPA, by generating custom textual CAPTCHAs, exploits ACM's careless design implementations and internal procedures vulnerabilities. We test our attack on real-world ACM, and the results confirm the ferocity of our simple yet effective attack, reaching up to a 100% evasion success in most cases. At the same time, we demonstrate the difficulties in designing CAPA mitigations, opening new challenges in CAPTCHAs research area.



## **26. Detecting In-vehicle Intrusion via Semi-supervised Learning-based Convolutional Adversarial Autoencoders**

cs.CR

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2204.01193v1)

**Authors**: Thien-Nu Hoang, Daehee Kim

**Abstracts**: With the development of autonomous vehicle technology, the controller area network (CAN) bus has become the de facto standard for an in-vehicle communication system because of its simplicity and efficiency. However, without any encryption and authentication mechanisms, the in-vehicle network using the CAN protocol is susceptible to a wide range of attacks. Many studies, which are mostly based on machine learning, have proposed installing an intrusion detection system (IDS) for anomaly detection in the CAN bus system. Although machine learning methods have many advantages for IDS, previous models usually require a large amount of labeled data, which results in high time and labor costs. To handle this problem, we propose a novel semi-supervised learning-based convolutional adversarial autoencoder model in this paper. The proposed model combines two popular deep learning models: autoencoder and generative adversarial networks. First, the model is trained with unlabeled data to learn the manifolds of normal and attack patterns. Then, only a small number of labeled samples are used in supervised training. The proposed model can detect various kinds of message injection attacks, such as DoS, fuzzy, and spoofing, as well as unknown attacks. The experimental results show that the proposed model achieves the highest F1 score of 0.99 and a low error rate of 0.1\% with limited labeled data compared to other supervised methods. In addition, we show that the model can meet the real-time requirement by analyzing the model complexity in terms of the number of trainable parameters and inference time. This study successfully reduced the number of model parameters by five times and the inference time by eight times, compared to a state-of-the-art model.



## **27. DST: Dynamic Substitute Training for Data-free Black-box Attack**

cs.CV

Accepted by CVPR2022

**SubmitDate**: 2022-04-03    [paper-pdf](http://arxiv.org/pdf/2204.00972v1)

**Authors**: Wenxuan Wang, Xuelin Qian, Yanwei Fu, Xiangyang Xue

**Abstracts**: With the wide applications of deep neural network models in various computer vision tasks, more and more works study the model vulnerability to adversarial examples. For data-free black box attack scenario, existing methods are inspired by the knowledge distillation, and thus usually train a substitute model to learn knowledge from the target model using generated data as input. However, the substitute model always has a static network structure, which limits the attack ability for various target models and tasks. In this paper, we propose a novel dynamic substitute training attack method to encourage substitute model to learn better and faster from the target model. Specifically, a dynamic substitute structure learning strategy is proposed to adaptively generate optimal substitute model structure via a dynamic gate according to different target models and tasks. Moreover, we introduce a task-driven graph-based structure information learning constrain to improve the quality of generated training data, and facilitate the substitute model learning structural relationships from the target model multiple outputs. Extensive experiments have been conducted to verify the efficacy of the proposed attack method, which can achieve better performance compared with the state-of-the-art competitors on several datasets.



## **28. Adversarial Neon Beam: Robust Physical-World Adversarial Attack to DNNs**

cs.CV

**SubmitDate**: 2022-04-02    [paper-pdf](http://arxiv.org/pdf/2204.00853v1)

**Authors**: Chengyin Hu, Kalibinuer Tiliwalidi

**Abstracts**: In the physical world, light affects the performance of deep neural networks. Nowadays, many products based on deep neural network have been put into daily life. There are few researches on the effect of light on the performance of deep neural network models. However, the adversarial perturbations generated by light may have extremely dangerous effects on these systems. In this work, we propose an attack method called adversarial neon beam (AdvNB), which can execute the physical attack by obtaining the physical parameters of adversarial neon beams with very few queries. Experiments show that our algorithm can achieve advanced attack effect in both digital test and physical test. In the digital environment, 99.3% attack success rate was achieved, and in the physical environment, 100% attack success rate was achieved. Compared with the most advanced physical attack methods, our method can achieve better physical perturbation concealment. In addition, by analyzing the experimental data, we reveal some new phenomena brought about by the adversarial neon beam attack.



## **29. Precise Statistical Analysis of Classification Accuracies for Adversarial Training**

stat.ML

80 pages; to appear in the Annals of Statistics

**SubmitDate**: 2022-04-02    [paper-pdf](http://arxiv.org/pdf/2010.11213v2)

**Authors**: Adel Javanmard, Mahdi Soltanolkotabi

**Abstracts**: Despite the wide empirical success of modern machine learning algorithms and models in a multitude of applications, they are known to be highly susceptible to seemingly small indiscernible perturbations to the input data known as \emph{adversarial attacks}. A variety of recent adversarial training procedures have been proposed to remedy this issue. Despite the success of such procedures at increasing accuracy on adversarially perturbed inputs or \emph{robust accuracy}, these techniques often reduce accuracy on natural unperturbed inputs or \emph{standard accuracy}. Complicating matters further, the effect and trend of adversarial training procedures on standard and robust accuracy is rather counter intuitive and radically dependent on a variety of factors including the perceived form of the perturbation during training, size/quality of data, model overparameterization, etc. In this paper we focus on binary classification problems where the data is generated according to the mixture of two Gaussians with general anisotropic covariance matrices and derive a precise characterization of the standard and robust accuracy for a class of minimax adversarially trained models. We consider a general norm-based adversarial model, where the adversary can add perturbations of bounded $\ell_p$ norm to each input data, for an arbitrary $p\ge 1$. Our comprehensive analysis allows us to theoretically explain several intriguing empirical phenomena and provide a precise understanding of the role of different problem parameters on standard and robust accuracies.



## **30. SkeleVision: Towards Adversarial Resiliency of Person Tracking with Multi-Task Learning**

cs.CV

**SubmitDate**: 2022-04-02    [paper-pdf](http://arxiv.org/pdf/2204.00734v1)

**Authors**: Nilaksh Das, Sheng-Yun Peng, Duen Horng Chau

**Abstracts**: Person tracking using computer vision techniques has wide ranging applications such as autonomous driving, home security and sports analytics. However, the growing threat of adversarial attacks raises serious concerns regarding the security and reliability of such techniques. In this work, we study the impact of multi-task learning (MTL) on the adversarial robustness of the widely used SiamRPN tracker, in the context of person tracking. Specifically, we investigate the effect of jointly learning with semantically analogous tasks of person tracking and human keypoint detection. We conduct extensive experiments with more powerful adversarial attacks that can be physically realizable, demonstrating the practical value of our approach. Our empirical study with simulated as well as real-world datasets reveals that training with MTL consistently makes it harder to attack the SiamRPN tracker, compared to typically training only on the single task of person tracking.



## **31. FrequencyLowCut Pooling -- Plug & Play against Catastrophic Overfitting**

cs.CV

**SubmitDate**: 2022-04-01    [paper-pdf](http://arxiv.org/pdf/2204.00491v1)

**Authors**: Julia Grabinski, Steffen Jung, Janis Keuper, Margret Keuper

**Abstracts**: Over the last years, Convolutional Neural Networks (CNNs) have been the dominating neural architecture in a wide range of computer vision tasks. From an image and signal processing point of view, this success might be a bit surprising as the inherent spatial pyramid design of most CNNs is apparently violating basic signal processing laws, i.e. Sampling Theorem in their down-sampling operations. However, since poor sampling appeared not to affect model accuracy, this issue has been broadly neglected until model robustness started to receive more attention. Recent work [17] in the context of adversarial attacks and distribution shifts, showed after all, that there is a strong correlation between the vulnerability of CNNs and aliasing artifacts induced by poor down-sampling operations. This paper builds on these findings and introduces an aliasing free down-sampling operation which can easily be plugged into any CNN architecture: FrequencyLowCut pooling. Our experiments show, that in combination with simple and fast FGSM adversarial training, our hyper-parameter free operator significantly improves model robustness and avoids catastrophic overfitting.



## **32. Sensor Data Validation and Driving Safety in Autonomous Driving Systems**

cs.CV

PhD Thesis, City University of Hong Kong

**SubmitDate**: 2022-04-01    [paper-pdf](http://arxiv.org/pdf/2203.16130v2)

**Authors**: Jindi Zhang

**Abstracts**: Autonomous driving technology has drawn a lot of attention due to its fast development and extremely high commercial values. The recent technological leap of autonomous driving can be primarily attributed to the progress in the environment perception. Good environment perception provides accurate high-level environment information which is essential for autonomous vehicles to make safe and precise driving decisions and strategies. Moreover, such progress in accurate environment perception would not be possible without deep learning models and advanced onboard sensors, such as optical sensors (LiDARs and cameras), radars, GPS. However, the advanced sensors and deep learning models are prone to recently invented attack methods. For example, LiDARs and cameras can be compromised by optical attacks, and deep learning models can be attacked by adversarial examples. The attacks on advanced sensors and deep learning models can largely impact the accuracy of the environment perception, posing great threats to the safety and security of autonomous vehicles. In this thesis, we study the detection methods against the attacks on onboard sensors and the linkage between attacked deep learning models and driving safety for autonomous vehicles. To detect the attacks, redundant data sources can be exploited, since information distortions caused by attacks in victim sensor data result in inconsistency with the information from other redundant sources. To study the linkage between attacked deep learning models and driving safety...



## **33. Multi-Expert Adversarial Attack Detection in Person Re-identification Using Context Inconsistency**

cs.CV

Accepted at IEEE ICCV 2021

**SubmitDate**: 2022-04-01    [paper-pdf](http://arxiv.org/pdf/2108.09891v2)

**Authors**: Xueping Wang, Shasha Li, Min Liu, Yaonan Wang, Amit K. Roy-Chowdhury

**Abstracts**: The success of deep neural networks (DNNs) has promoted the widespread applications of person re-identification (ReID). However, ReID systems inherit the vulnerability of DNNs to malicious attacks of visually inconspicuous adversarial perturbations. Detection of adversarial attacks is, therefore, a fundamental requirement for robust ReID systems. In this work, we propose a Multi-Expert Adversarial Attack Detection (MEAAD) approach to achieve this goal by checking context inconsistency, which is suitable for any DNN-based ReID systems. Specifically, three kinds of context inconsistencies caused by adversarial attacks are employed to learn a detector for distinguishing the perturbed examples, i.e., a) the embedding distances between a perturbed query person image and its top-K retrievals are generally larger than those between a benign query image and its top-K retrievals, b) the embedding distances among the top-K retrievals of a perturbed query image are larger than those of a benign query image, c) the top-K retrievals of a benign query image obtained with multiple expert ReID models tend to be consistent, which is not preserved when attacks are present. Extensive experiments on the Market1501 and DukeMTMC-ReID datasets show that, as the first adversarial attack detection approach for ReID, MEAAD effectively detects various adversarial attacks and achieves high ROC-AUC (over 97.5%).



## **34. Effect of Balancing Data Using Synthetic Data on the Performance of Machine Learning Classifiers for Intrusion Detection in Computer Networks**

cs.LG

**SubmitDate**: 2022-04-01    [paper-pdf](http://arxiv.org/pdf/2204.00144v1)

**Authors**: Ayesha S. Dina, A. B. Siddique, D. Manivannan

**Abstracts**: Attacks on computer networks have increased significantly in recent days, due in part to the availability of sophisticated tools for launching such attacks as well as thriving underground cyber-crime economy to support it. Over the past several years, researchers in academia and industry used machine learning (ML) techniques to design and implement Intrusion Detection Systems (IDSes) for computer networks. Many of these researchers used datasets collected by various organizations to train ML models for predicting intrusions. In many of the datasets used in such systems, data are imbalanced (i.e., not all classes have equal amount of samples). With unbalanced data, the predictive models developed using ML algorithms may produce unsatisfactory classifiers which would affect accuracy in predicting intrusions. Traditionally, researchers used over-sampling and under-sampling for balancing data in datasets to overcome this problem. In this work, in addition to over-sampling, we also use a synthetic data generation method, called Conditional Generative Adversarial Network (CTGAN), to balance data and study their effect on various ML classifiers. To the best of our knowledge, no one else has used CTGAN to generate synthetic samples to balance intrusion detection datasets. Based on extensive experiments using a widely used dataset NSL-KDD, we found that training ML models on dataset balanced with synthetic samples generated by CTGAN increased prediction accuracy by up to $8\%$, compared to training the same ML models over unbalanced data. Our experiments also show that the accuracy of some ML models trained over data balanced with random over-sampling decline compared to the same ML models trained over unbalanced data.



## **35. Reverse Engineering of Imperceptible Adversarial Image Perturbations**

cs.CV

**SubmitDate**: 2022-04-01    [paper-pdf](http://arxiv.org/pdf/2203.14145v2)

**Authors**: Yifan Gong, Yuguang Yao, Yize Li, Yimeng Zhang, Xiaoming Liu, Xue Lin, Sijia Liu

**Abstracts**: It has been well recognized that neural network based image classifiers are easily fooled by images with tiny perturbations crafted by an adversary. There has been a vast volume of research to generate and defend such adversarial attacks. However, the following problem is left unexplored: How to reverse-engineer adversarial perturbations from an adversarial image? This leads to a new adversarial learning paradigm--Reverse Engineering of Deceptions (RED). If successful, RED allows us to estimate adversarial perturbations and recover the original images. However, carefully crafted, tiny adversarial perturbations are difficult to recover by optimizing a unilateral RED objective. For example, the pure image denoising method may overfit to minimizing the reconstruction error but hardly preserve the classification properties of the true adversarial perturbations. To tackle this challenge, we formalize the RED problem and identify a set of principles crucial to the RED approach design. Particularly, we find that prediction alignment and proper data augmentation (in terms of spatial transformations) are two criteria to achieve a generalizable RED approach. By integrating these RED principles with image denoising, we propose a new Class-Discriminative Denoising based RED framework, termed CDD-RED. Extensive experiments demonstrate the effectiveness of CDD-RED under different evaluation metrics (ranging from the pixel-level, prediction-level to the attribution-level alignment) and a variety of attack generation methods (e.g., FGSM, PGD, CW, AutoAttack, and adaptive attacks).



## **36. Scalable Whitebox Attacks on Tree-based Models**

stat.ML

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2204.00103v1)

**Authors**: Giuseppe Castiglione, Gavin Ding, Masoud Hashemi, Christopher Srinivasa, Ga Wu

**Abstracts**: Adversarial robustness is one of the essential safety criteria for guaranteeing the reliability of machine learning models. While various adversarial robustness testing approaches were introduced in the last decade, we note that most of them are incompatible with non-differentiable models such as tree ensembles. Since tree ensembles are widely used in industry, this reveals a crucial gap between adversarial robustness research and practical applications. This paper proposes a novel whitebox adversarial robustness testing approach for tree ensemble models. Concretely, the proposed approach smooths the tree ensembles through temperature controlled sigmoid functions, which enables gradient descent-based adversarial attacks. By leveraging sampling and the log-derivative trick, the proposed approach can scale up to testing tasks that were previously unmanageable. We compare the approach against both random perturbations and blackbox approaches on multiple public datasets (and corresponding models). Our results show that the proposed method can 1) successfully reveal the adversarial vulnerability of tree ensemble models without causing computational pressure for testing and 2) flexibly balance the search performance and time complexity to meet various testing criteria.



## **37. Parallel Proof-of-Work with Concrete Bounds**

cs.CR

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2204.00034v1)

**Authors**: Patrik Keller, Rainer Böhme

**Abstracts**: Authorization is challenging in distributed systems that cannot rely on the identification of nodes. Proof-of-work offers an alternative gate-keeping mechanism, but its probabilistic nature is incompatible with conventional security definitions. Recent related work establishes concrete bounds for the failure probability of Bitcoin's sequential proof-of-work mechanism. We propose a family of state replication protocols using parallel proof-of-work. Our bottom-up design from an agreement sub-protocol allows us to give concrete bounds for the failure probability in adversarial synchronous networks. After the typical interval of 10 minutes, parallel proof-of-work offers two orders of magnitude more security than sequential proof-of-work. This means that state updates can be sufficiently secure to support commits after one block (i.e., after 10 minutes), removing the risk of double-spending in many applications. We offer guidance on the optimal choice of parameters for a wide range of network and attacker assumptions. Simulations show that the proposed construction is robust against violations of design assumptions.



## **38. Truth Serum: Poisoning Machine Learning Models to Reveal Their Secrets**

cs.CR

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2204.00032v1)

**Authors**: Florian Tramèr, Reza Shokri, Ayrton San Joaquin, Hoang Le, Matthew Jagielski, Sanghyun Hong, Nicholas Carlini

**Abstracts**: We introduce a new class of attacks on machine learning models. We show that an adversary who can poison a training dataset can cause models trained on this dataset to leak significant private details of training points belonging to other parties. Our active inference attacks connect two independent lines of work targeting the integrity and privacy of machine learning training data.   Our attacks are effective across membership inference, attribute inference, and data extraction. For example, our targeted attacks can poison <0.1% of the training dataset to boost the performance of inference attacks by 1 to 2 orders of magnitude. Further, an adversary who controls a significant fraction of the training data (e.g., 50%) can launch untargeted attacks that enable 8x more precise inference on all other users' otherwise-private data points.   Our results cast doubts on the relevance of cryptographic privacy guarantees in multiparty computation protocols for machine learning, if parties can arbitrarily select their share of training data.



## **39. Improving Adversarial Transferability via Neuron Attribution-Based Attacks**

cs.LG

CVPR 2022

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2204.00008v1)

**Authors**: Jianping Zhang, Weibin Wu, Jen-tse Huang, Yizhan Huang, Wenxuan Wang, Yuxin Su, Michael R. Lyu

**Abstracts**: Deep neural networks (DNNs) are known to be vulnerable to adversarial examples. It is thus imperative to devise effective attack algorithms to identify the deficiencies of DNNs beforehand in security-sensitive applications. To efficiently tackle the black-box setting where the target model's particulars are unknown, feature-level transfer-based attacks propose to contaminate the intermediate feature outputs of local models, and then directly employ the crafted adversarial samples to attack the target model. Due to the transferability of features, feature-level attacks have shown promise in synthesizing more transferable adversarial samples. However, existing feature-level attacks generally employ inaccurate neuron importance estimations, which deteriorates their transferability. To overcome such pitfalls, in this paper, we propose the Neuron Attribution-based Attack (NAA), which conducts feature-level attacks with more accurate neuron importance estimations. Specifically, we first completely attribute a model's output to each neuron in a middle layer. We then derive an approximation scheme of neuron attribution to tremendously reduce the computation overhead. Finally, we weight neurons based on their attribution results and launch feature-level attacks. Extensive experiments confirm the superiority of our approach to the state-of-the-art benchmarks.



## **40. Towards Robust Rain Removal Against Adversarial Attacks: A Comprehensive Benchmark Analysis and Beyond**

cs.CV

10 pages, 6 figures, to appear in CVPR 2022

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2203.16931v1)

**Authors**: Yi Yu, Wenhan Yang, Yap-Peng Tan, Alex C. Kot

**Abstracts**: Rain removal aims to remove rain streaks from images/videos and reduce the disruptive effects caused by rain. It not only enhances image/video visibility but also allows many computer vision algorithms to function properly. This paper makes the first attempt to conduct a comprehensive study on the robustness of deep learning-based rain removal methods against adversarial attacks. Our study shows that, when the image/video is highly degraded, rain removal methods are more vulnerable to the adversarial attacks as small distortions/perturbations become less noticeable or detectable. In this paper, we first present a comprehensive empirical evaluation of various methods at different levels of attacks and with various losses/targets to generate the perturbations from the perspective of human perception and machine analysis tasks. A systematic evaluation of key modules in existing methods is performed in terms of their robustness against adversarial attacks. From the insights of our analysis, we construct a more robust deraining method by integrating these effective modules. Finally, we examine various types of adversarial attacks that are specific to deraining problems and their effects on both human and machine vision tasks, including 1) rain region attacks, adding perturbations only in the rain regions to make the perturbations in the attacked rain images less visible; 2) object-sensitive attacks, adding perturbations only in regions near the given objects. Code is available at https://github.com/yuyi-sd/Robust_Rain_Removal.



## **41. Assessing the risk of re-identification arising from an attack on anonymised data**

cs.LG

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2203.16921v1)

**Authors**: Anna Antoniou, Giacomo Dossena, Julia MacMillan, Steven Hamblin, David Clifton, Paula Petrone

**Abstracts**: Objective: The use of routinely-acquired medical data for research purposes requires the protection of patient confidentiality via data anonymisation. The objective of this work is to calculate the risk of re-identification arising from a malicious attack to an anonymised dataset, as described below. Methods: We first present an analytical means of estimating the probability of re-identification of a single patient in a k-anonymised dataset of Electronic Health Record (EHR) data. Second, we generalize this solution to obtain the probability of multiple patients being re-identified. We provide synthetic validation via Monte Carlo simulations to illustrate the accuracy of the estimates obtained. Results: The proposed analytical framework for risk estimation provides re-identification probabilities that are in agreement with those provided by simulation in a number of scenarios. Our work is limited by conservative assumptions which inflate the re-identification probability. Discussion: Our estimates show that the re-identification probability increases with the proportion of the dataset maliciously obtained and that it has an inverse relationship with the equivalence class size. Our recursive approach extends the applicability domain to the general case of a multi-patient re-identification attack in an arbitrary k-anonymisation scheme. Conclusion: We prescribe a systematic way to parametrize the k-anonymisation process based on a pre-determined re-identification probability. We observed that the benefits of a reduced re-identification risk that come with increasing k-size may not be worth the reduction in data granularity when one is considering benchmarking the re-identification probability on the size of the portion of the dataset maliciously obtained by the adversary.



## **42. Attack Impact Evaluation by Exact Convexification through State Space Augmentation**

eess.SY

8 pages

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2203.16803v1)

**Authors**: Hampei Sasahara, Takashi Tanaka, Henrik Sandberg

**Abstracts**: We address the attack impact evaluation problem for control system security. We formulate the problem as a Markov decision process with a temporally joint chance constraint that forces the adversary to avoid being detected throughout the considered time period. Owing to the joint constraint, the optimal control policy depends not only on the current state but also on the entire history, which leads to the explosion of the search space and makes the problem generally intractable. It is shown that whether an alarm has been triggered or not, in addition to the current state is sufficient for specifying the optimal decision at each time step. Augmentation of the information to the state space induces an equivalent convex optimization problem, which is tractable using standard solvers.



## **43. The Block-based Mobile PDE Systems Are Not Secure -- Experimental Attacks**

cs.CR

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2203.16349v2)

**Authors**: Niusen Chen, Bo Chen, Weisong Shi

**Abstracts**: Nowadays, mobile devices have been used broadly to store and process sensitive data. To ensure confidentiality of the sensitive data, Full Disk Encryption (FDE) is often integrated in mainstream mobile operating systems like Android and iOS. FDE however cannot defend against coercive attacks in which the adversary can force the device owner to disclose the decryption key. To combat the coercive attacks, Plausibly Deniable Encryption (PDE) is leveraged to plausibly deny the very existence of sensitive data. However, most of the existing PDE systems for mobile devices are deployed at the block layer and suffer from deniability compromises.   Having observed that none of existing works in the literature have experimentally demonstrated the aforementioned compromises, our work bridges this gap by experimentally confirming the deniability compromises of the block-layer mobile PDE systems. We have built a mobile device testbed, which consists of a host computing device and a flash storage device. Additionally, we have deployed both the hidden volume PDE and the steganographic file system at the block layer of the testbed and performed disk forensics to assess potential compromises on the raw NAND flash. Our experimental results confirm it is indeed possible for the adversary to compromise the block-layer PDE systems by accessing the raw NAND flash in practice. We also discuss potential issues when performing such attacks in real world.



## **44. Well-classified Examples are Underestimated in Classification with Deep Neural Networks**

cs.LG

Accepted by AAAI 2022; 17 pages, 11 figures, 13 tables

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2110.06537v5)

**Authors**: Guangxiang Zhao, Wenkai Yang, Xuancheng Ren, Lei Li, Yunfang Wu, Xu Sun

**Abstracts**: The conventional wisdom behind learning deep classification models is to focus on bad-classified examples and ignore well-classified examples that are far from the decision boundary. For instance, when training with cross-entropy loss, examples with higher likelihoods (i.e., well-classified examples) contribute smaller gradients in back-propagation. However, we theoretically show that this common practice hinders representation learning, energy optimization, and margin growth. To counteract this deficiency, we propose to reward well-classified examples with additive bonuses to revive their contribution to the learning process. This counterexample theoretically addresses these three issues. We empirically support this claim by directly verifying the theoretical results or significant performance improvement with our counterexample on diverse tasks, including image classification, graph classification, and machine translation. Furthermore, this paper shows that we can deal with complex scenarios, such as imbalanced classification, OOD detection, and applications under adversarial attacks because our idea can solve these three issues. Code is available at: https://github.com/lancopku/well-classified-examples-are-underestimated.



## **45. Example-based Explanations with Adversarial Attacks for Respiratory Sound Analysis**

cs.SD

Submitted to INTERSPEECH 2022

**SubmitDate**: 2022-03-30    [paper-pdf](http://arxiv.org/pdf/2203.16141v1)

**Authors**: Yi Chang, Zhao Ren, Thanh Tam Nguyen, Wolfgang Nejdl, Björn W. Schuller

**Abstracts**: Respiratory sound classification is an important tool for remote screening of respiratory-related diseases such as pneumonia, asthma, and COVID-19. To facilitate the interpretability of classification results, especially ones based on deep learning, many explanation methods have been proposed using prototypes. However, existing explanation techniques often assume that the data is non-biased and the prediction results can be explained by a set of prototypical examples. In this work, we develop a unified example-based explanation method for selecting both representative data (prototypes) and outliers (criticisms). In particular, we propose a novel application of adversarial attacks to generate an explanation spectrum of data instances via an iterative fast gradient sign method. Such unified explanation can avoid over-generalisation and bias by allowing human experts to assess the model mistakes case by case. We performed a wide range of quantitative and qualitative evaluations to show that our approach generates effective and understandable explanation and is robust with many deep learning models



## **46. Fooling the primate brain with minimal, targeted image manipulation**

q-bio.NC

**SubmitDate**: 2022-03-30    [paper-pdf](http://arxiv.org/pdf/2011.05623v3)

**Authors**: Li Yuan, Will Xiao, Giorgia Dellaferrera, Gabriel Kreiman, Francis E. H. Tay, Jiashi Feng, Margaret S. Livingstone

**Abstracts**: Artificial neural networks (ANNs) are considered the current best models of biological vision. ANNs are the best predictors of neural activity in the ventral stream; moreover, recent work has demonstrated that ANN models fitted to neuronal activity can guide the synthesis of images that drive pre-specified response patterns in small neuronal populations. Despite the success in predicting and steering firing activity, these results have not been connected with perceptual or behavioral changes. Here we propose an array of methods for creating minimal, targeted image perturbations that lead to changes in both neuronal activity and perception as reflected in behavior. We generated 'deceptive images' of human faces, monkey faces, and noise patterns so that they are perceived as a different, pre-specified target category, and measured both monkey neuronal responses and human behavior to these images. We found several effective methods for changing primate visual categorization that required much smaller image change compared to untargeted noise. Our work shares the same goal with adversarial attack, namely the manipulation of images with minimal, targeted noise that leads ANN models to misclassify the images. Our results represent a valuable step in quantifying and characterizing the differences in perturbation robustness of biological and artificial vision.



## **47. StyleFool: Fooling Video Classification Systems via Style Transfer**

cs.CV

18 pages, 7 figures

**SubmitDate**: 2022-03-30    [paper-pdf](http://arxiv.org/pdf/2203.16000v1)

**Authors**: Yuxin Cao, Xi Xiao, Ruoxi Sun, Derui Wang, Minhui Xue, Sheng Wen

**Abstracts**: Video classification systems are vulnerable to adversarial attacks, which can create severe security problems in video verification. Current black-box attacks need a large number of queries to succeed, resulting in high computational overhead in the process of attack. On the other hand, attacks with restricted perturbations are ineffective against defenses such as denoising or adversarial training. In this paper, we focus on unrestricted perturbations and propose StyleFool, a black-box video adversarial attack via style transfer to fool the video classification system. StyleFool first utilizes color theme proximity to select the best style image, which helps avoid unnatural details in the stylized videos. Meanwhile, the target class confidence is additionally considered in targeted attack to influence the output distribution of the classifier by moving the stylized video closer to or even across the decision boundary. A gradient-free method is then employed to further optimize the adversarial perturbation. We carry out extensive experiments to evaluate StyleFool on two standard datasets, UCF-101 and HMDB-51. The experimental results suggest that StyleFool outperforms the state-of-the-art adversarial attacks in terms of both number of queries and robustness against existing defenses. We identify that 50% of the stylized videos in untargeted attack do not need any query since they can already fool the video classification model. Furthermore, we evaluate the indistinguishability through a user study to show that the adversarial samples of StyleFool look imperceptible to human eyes, despite unrestricted perturbations.



## **48. NICGSlowDown: Evaluating the Efficiency Robustness of Neural Image Caption Generation Models**

cs.CV

This paper is accepted at CVPR2022

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2203.15859v1)

**Authors**: Simin Chen, Zihe Song, Mirazul Haque, Cong Liu, Wei Yang

**Abstracts**: Neural image caption generation (NICG) models have received massive attention from the research community due to their excellent performance in visual understanding. Existing work focuses on improving NICG model accuracy while efficiency is less explored. However, many real-world applications require real-time feedback, which highly relies on the efficiency of NICG models. Recent research observed that the efficiency of NICG models could vary for different inputs. This observation brings in a new attack surface of NICG models, i.e., An adversary might be able to slightly change inputs to cause the NICG models to consume more computational resources. To further understand such efficiency-oriented threats, we propose a new attack approach, NICGSlowDown, to evaluate the efficiency robustness of NICG models. Our experimental results show that NICGSlowDown can generate images with human-unnoticeable perturbations that will increase the NICG model latency up to 483.86%. We hope this research could raise the community's concern about the efficiency robustness of NICG models.



## **49. Characterizing the adversarial vulnerability of speech self-supervised learning**

cs.SD

Accepted by ICASSP 2022

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2111.04330v2)

**Authors**: Haibin Wu, Bo Zheng, Xu Li, Xixin Wu, Hung-yi Lee, Helen Meng

**Abstracts**: A leaderboard named Speech processing Universal PERformance Benchmark (SUPERB), which aims at benchmarking the performance of a shared self-supervised learning (SSL) speech model across various downstream speech tasks with minimal modification of architectures and small amount of data, has fueled the research for speech representation learning. The SUPERB demonstrates speech SSL upstream models improve the performance of various downstream tasks through just minimal adaptation. As the paradigm of the self-supervised learning upstream model followed by downstream tasks arouses more attention in the speech community, characterizing the adversarial robustness of such paradigm is of high priority. In this paper, we make the first attempt to investigate the adversarial vulnerability of such paradigm under the attacks from both zero-knowledge adversaries and limited-knowledge adversaries. The experimental results illustrate that the paradigm proposed by SUPERB is seriously vulnerable to limited-knowledge adversaries, and the attacks generated by zero-knowledge adversaries are with transferability. The XAB test verifies the imperceptibility of crafted adversarial attacks.



## **50. Adaptative Perturbation Patterns: Realistic Adversarial Learning for Robust Intrusion Detection**

cs.CR

18 pages, 6 tables, 10 figures, Future Internet journal

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2203.04234v2)

**Authors**: João Vitorino, Nuno Oliveira, Isabel Praça

**Abstracts**: Adversarial attacks pose a major threat to machine learning and to the systems that rely on it. In the cybersecurity domain, adversarial cyber-attack examples capable of evading detection are especially concerning. Nonetheless, an example generated for a domain with tabular data must be realistic within that domain. This work establishes the fundamental constraint levels required to achieve realism and introduces the Adaptative Perturbation Pattern Method (A2PM) to fulfill these constraints in a gray-box setting. A2PM relies on pattern sequences that are independently adapted to the characteristics of each class to create valid and coherent data perturbations. The proposed method was evaluated in a cybersecurity case study with two scenarios: Enterprise and Internet of Things (IoT) networks. Multilayer Perceptron (MLP) and Random Forest (RF) classifiers were created with regular and adversarial training, using the CIC-IDS2017 and IoT-23 datasets. In each scenario, targeted and untargeted attacks were performed against the classifiers, and the generated examples were compared with the original network traffic flows to assess their realism. The obtained results demonstrate that A2PM provides a scalable generation of realistic adversarial examples, which can be advantageous for both adversarial training and attacks.



