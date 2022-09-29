# Latest Adversarial Attack Papers
**update at 2022-09-30 06:31:28**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. A Survey on Physical Adversarial Attack in Computer Vision**

cs.CV

**SubmitDate**: 2022-09-28    [paper-pdf](http://arxiv.org/pdf/2209.14262v1)

**Authors**: Donghua Wang, Wen Yao, Tingsong Jiang, Guijiang Tang, Xiaoqian Chen

**Abstracts**: In the past decade, deep learning has dramatically changed the traditional hand-craft feature manner with strong feature learning capability, resulting in tremendous improvement of conventional tasks. However, deep neural networks have recently been demonstrated vulnerable to adversarial examples, a kind of malicious samples crafted by small elaborately designed noise, which mislead the DNNs to make the wrong decisions while remaining imperceptible to humans. Adversarial examples can be divided into digital adversarial attacks and physical adversarial attacks. The digital adversarial attack is mostly performed in lab environments, focusing on improving the performance of adversarial attack algorithms. In contrast, the physical adversarial attack focus on attacking the physical world deployed DNN systems, which is a more challenging task due to the complex physical environment (i.e., brightness, occlusion, and so on). Although the discrepancy between digital adversarial and physical adversarial examples is small, the physical adversarial examples have a specific design to overcome the effect of the complex physical environment. In this paper, we review the development of physical adversarial attacks in DNN-based computer vision tasks, including image recognition tasks, object detection tasks, and semantic segmentation. For the sake of completeness of the algorithm evolution, we will briefly introduce the works that do not involve the physical adversarial attack. We first present a categorization scheme to summarize the current physical adversarial attacks. Then discuss the advantages and disadvantages of the existing physical adversarial attacks and focus on the technique used to maintain the adversarial when applied into physical environment. Finally, we point out the issues of the current physical adversarial attacks to be solved and provide promising research directions.



## **2. A Closer Look at Evaluating the Bit-Flip Attack Against Deep Neural Networks**

cs.CR

Extended version from IEEE IOLTS'2022 short paper

**SubmitDate**: 2022-09-28    [paper-pdf](http://arxiv.org/pdf/2209.14243v1)

**Authors**: Kevin Hector, Mathieu Dumont, Pierre-Alain Moellic, Jean-Max Dutertre

**Abstracts**: Deep neural network models are massively deployed on a wide variety of hardware platforms. This results in the appearance of new attack vectors that significantly extend the standard attack surface, extensively studied by the adversarial machine learning community. One of the first attack that aims at drastically dropping the performance of a model, by targeting its parameters (weights) stored in memory, is the Bit-Flip Attack (BFA). In this work, we point out several evaluation challenges related to the BFA. First of all, the lack of an adversary's budget in the standard threat model is problematic, especially when dealing with physical attacks. Moreover, since the BFA presents critical variability, we discuss the influence of some training parameters and the importance of the model architecture. This work is the first to present the impact of the BFA against fully-connected architectures that present different behaviors compared to convolutional neural networks. These results highlight the importance of defining robust and sound evaluation methodologies to properly evaluate the dangers of parameter-based attacks as well as measure the real level of robustness offered by a defense.



## **3. Exploring the Relationship between Architecture and Adversarially Robust Generalization**

cs.LG

**SubmitDate**: 2022-09-28    [paper-pdf](http://arxiv.org/pdf/2209.14105v1)

**Authors**: Shiyu Tang, Siyuan Liang, Ruihao Gong, Aishan Liu, Xianglong Liu, Dacheng Tao

**Abstracts**: Adversarial training has been demonstrated to be one of the most effective remedies for defending adversarial examples, yet it often suffers from the huge robustness generalization gap on unseen testing adversaries, deemed as the \emph{adversarially robust generalization problem}. Despite the preliminary understandings devoted on adversarially robust generalization, little is known from the architectural perspective. Thus, this paper tries to bridge the gap by systematically examining the most representative architectures (e.g., Vision Transformers and CNNs). In particular, we first comprehensively evaluated \emph{20} adversarially trained architectures on ImageNette and CIFAR-10 datasets towards several adversaries (multiple $\ell_p$-norm adversarial attacks), and found that Vision Transformers (e.g., PVT, CoAtNet) often yield better adversarially robust generalization. To further understand what architectural ingredients favor adversarially robust generalization, we delve into several key building blocks and revealed the fact via the lens of Rademacher complexity that the higher weight sparsity contributes significantly towards the better adversarially robust generalization of Vision Transformers, which can be often achieved by attention layers. Our extensive studies discovered the close relationship between architectural design and adversarially robust generalization, and instantiated several important insights. We hope our findings could help to better understand the mechanism towards designing robust deep learning architectures.



## **4. Machine Beats Machine: Machine Learning Models to Defend Against Adversarial Attacks**

cs.LG

**SubmitDate**: 2022-09-28    [paper-pdf](http://arxiv.org/pdf/2209.13963v1)

**Authors**: Jože M. Rožanec, Dimitrios Papamartzivanos, Entso Veliou, Theodora Anastasiou, Jelle Keizer, Blaž Fortuna, Dunja Mladenić

**Abstracts**: We propose using a two-layered deployment of machine learning models to prevent adversarial attacks. The first layer determines whether the data was tampered, while the second layer solves a domain-specific problem. We explore three sets of features and three dataset variations to train machine learning models. Our results show clustering algorithms achieved promising results. In particular, we consider the best results were obtained by applying the DBSCAN algorithm to the structured structural similarity index measure computed between the images and a white reference image.



## **5. Adaptive Image Transformations for Transfer-based Adversarial Attack**

cs.CV

34 pages, 7 figures, 11 tables. Accepted by ECCV2022

**SubmitDate**: 2022-09-28    [paper-pdf](http://arxiv.org/pdf/2111.13844v4)

**Authors**: Zheng Yuan, Jie Zhang, Shiguang Shan

**Abstracts**: Adversarial attacks provide a good way to study the robustness of deep learning models. One category of methods in transfer-based black-box attack utilizes several image transformation operations to improve the transferability of adversarial examples, which is effective, but fails to take the specific characteristic of the input image into consideration. In this work, we propose a novel architecture, called Adaptive Image Transformation Learner (AITL), which incorporates different image transformation operations into a unified framework to further improve the transferability of adversarial examples. Unlike the fixed combinational transformations used in existing works, our elaborately designed transformation learner adaptively selects the most effective combination of image transformations specific to the input image. Extensive experiments on ImageNet demonstrate that our method significantly improves the attack success rates on both normally trained models and defense models under various settings.



## **6. Understanding Real-world Threats to Deep Learning Models in Android Apps**

cs.CR

18 pages, 9 figures, accepted by CCS'22

**SubmitDate**: 2022-09-28    [paper-pdf](http://arxiv.org/pdf/2209.09577v2)

**Authors**: Zizhuang Deng, Kai Chen, Guozhu Meng, Xiaodong Zhang, Ke Xu, Yao Cheng

**Abstracts**: Famous for its superior performance, deep learning (DL) has been popularly used within many applications, which also at the same time attracts various threats to the models. One primary threat is from adversarial attacks. Researchers have intensively studied this threat for several years and proposed dozens of approaches to create adversarial examples (AEs). But most of the approaches are only evaluated on limited models and datasets (e.g., MNIST, CIFAR-10). Thus, the effectiveness of attacking real-world DL models is not quite clear. In this paper, we perform the first systematic study of adversarial attacks on real-world DNN models and provide a real-world model dataset named RWM. Particularly, we design a suite of approaches to adapt current AE generation algorithms to the diverse real-world DL models, including automatically extracting DL models from Android apps, capturing the inputs and outputs of the DL models in apps, generating AEs and validating them by observing the apps' execution. For black-box DL models, we design a semantic-based approach to build suitable datasets and use them for training substitute models when performing transfer-based attacks. After analyzing 245 DL models collected from 62,583 real-world apps, we have a unique opportunity to understand the gap between real-world DL models and contemporary AE generation algorithms. To our surprise, the current AE generation algorithms can only directly attack 6.53% of the models. Benefiting from our approach, the success rate upgrades to 47.35%.



## **7. On the Limitations of Stochastic Pre-processing Defenses**

cs.LG

Accepted by Proceedings of the 36th Conference on Neural Information  Processing Systems

**SubmitDate**: 2022-09-28    [paper-pdf](http://arxiv.org/pdf/2206.09491v2)

**Authors**: Yue Gao, Ilia Shumailov, Kassem Fawaz, Nicolas Papernot

**Abstracts**: Defending against adversarial examples remains an open problem. A common belief is that randomness at inference increases the cost of finding adversarial inputs. An example of such a defense is to apply a random transformation to inputs prior to feeding them to the model. In this paper, we empirically and theoretically investigate such stochastic pre-processing defenses and demonstrate that they are flawed. First, we show that most stochastic defenses are weaker than previously thought; they lack sufficient randomness to withstand even standard attacks like projected gradient descent. This casts doubt on a long-held assumption that stochastic defenses invalidate attacks designed to evade deterministic defenses and force attackers to integrate the Expectation over Transformation (EOT) concept. Second, we show that stochastic defenses confront a trade-off between adversarial robustness and model invariance; they become less effective as the defended model acquires more invariance to their randomization. Future work will need to decouple these two effects. We also discuss implications and guidance for future research.



## **8. Attacking Compressed Vision Transformers**

cs.LG

**SubmitDate**: 2022-09-28    [paper-pdf](http://arxiv.org/pdf/2209.13785v1)

**Authors**: Swapnil Parekh, Devansh Shah, Pratyush Shukla

**Abstracts**: Vision Transformers are increasingly embedded in industrial systems due to their superior performance, but their memory and power requirements make deploying them to edge devices a challenging task. Hence, model compression techniques are now widely used to deploy models on edge devices as they decrease the resource requirements and make model inference very fast and efficient. But their reliability and robustness from a security perspective is another major issue in safety-critical applications. Adversarial attacks are like optical illusions for ML algorithms and they can severely impact the accuracy and reliability of models. In this work we investigate the transferability of adversarial samples across the SOTA Vision Transformer models across 3 SOTA compressed versions and infer the effects different compression techniques have on adversarial attacks.



## **9. Suppress with a Patch: Revisiting Universal Adversarial Patch Attacks against Object Detection**

cs.CV

Accepted for publication at ICECCME 2022

**SubmitDate**: 2022-09-27    [paper-pdf](http://arxiv.org/pdf/2209.13353v1)

**Authors**: Svetlana Pavlitskaya, Jonas Hendl, Sebastian Kleim, Leopold Müller, Fabian Wylczoch, J. Marius Zöllner

**Abstracts**: Adversarial patch-based attacks aim to fool a neural network with an intentionally generated noise, which is concentrated in a particular region of an input image. In this work, we perform an in-depth analysis of different patch generation parameters, including initialization, patch size, and especially positioning a patch in an image during training. We focus on the object vanishing attack and run experiments with YOLOv3 as a model under attack in a white-box setting and use images from the COCO dataset. Our experiments have shown, that inserting a patch inside a window of increasing size during training leads to a significant increase in attack strength compared to a fixed position. The best results were obtained when a patch was positioned randomly during training, while patch position additionally varied within a batch.



## **10. Mitigating Attacks on Artificial Intelligence-based Spectrum Sensing for Cellular Network Signals**

cs.NI

IEEE GLOBECOM 2022 Publication

**SubmitDate**: 2022-09-27    [paper-pdf](http://arxiv.org/pdf/2209.13007v1)

**Authors**: Ferhat Ozgur Catak, Murat Kuzlu, Salih Sarp, Evren Catak, Umit Cali

**Abstracts**: Cellular networks (LTE, 5G, and beyond) are dramatically growing with high demand from consumers and more promising than the other wireless networks with advanced telecommunication technologies. The main goal of these networks is to connect billions of devices, systems, and users with high-speed data transmission, high cell capacity, and low latency, as well as to support a wide range of new applications, such as virtual reality, metaverse, telehealth, online education, autonomous and flying vehicles, advanced manufacturing, and many more. To achieve these goals, spectrum sensing has been paid more attention, along with new approaches using artificial intelligence (AI) methods for spectrum management in cellular networks. This paper provides a vulnerability analysis of spectrum sensing approaches using AI-based semantic segmentation models for identifying cellular network signals under adversarial attacks with and without defensive distillation methods. The results showed that mitigation methods can significantly reduce the vulnerabilities of AI-based spectrum sensing models against adversarial attacks.



## **11. Reliable Representations Make A Stronger Defender: Unsupervised Structure Refinement for Robust GNN**

cs.LG

Accepted in KDD2022

**SubmitDate**: 2022-09-27    [paper-pdf](http://arxiv.org/pdf/2207.00012v3)

**Authors**: Kuan Li, Yang Liu, Xiang Ao, Jianfeng Chi, Jinghua Feng, Hao Yang, Qing He

**Abstracts**: Benefiting from the message passing mechanism, Graph Neural Networks (GNNs) have been successful on flourish tasks over graph data. However, recent studies have shown that attackers can catastrophically degrade the performance of GNNs by maliciously modifying the graph structure. A straightforward solution to remedy this issue is to model the edge weights by learning a metric function between pairwise representations of two end nodes, which attempts to assign low weights to adversarial edges. The existing methods use either raw features or representations learned by supervised GNNs to model the edge weights. However, both strategies are faced with some immediate problems: raw features cannot represent various properties of nodes (e.g., structure information), and representations learned by supervised GNN may suffer from the poor performance of the classifier on the poisoned graph. We need representations that carry both feature information and as mush correct structure information as possible and are insensitive to structural perturbations. To this end, we propose an unsupervised pipeline, named STABLE, to optimize the graph structure. Finally, we input the well-refined graph into a downstream classifier. For this part, we design an advanced GCN that significantly enhances the robustness of vanilla GCN without increasing the time complexity. Extensive experiments on four real-world graph benchmarks demonstrate that STABLE outperforms the state-of-the-art methods and successfully defends against various attacks.



## **12. FG-UAP: Feature-Gathering Universal Adversarial Perturbation**

cs.CV

27 pages, 4 figures

**SubmitDate**: 2022-09-27    [paper-pdf](http://arxiv.org/pdf/2209.13113v1)

**Authors**: Zhixing Ye, Xinwen Cheng, Xiaolin Huang

**Abstracts**: Deep Neural Networks (DNNs) are susceptible to elaborately designed perturbations, whether such perturbations are dependent or independent of images. The latter one, called Universal Adversarial Perturbation (UAP), is very attractive for model robustness analysis, since its independence of input reveals the intrinsic characteristics of the model. Relatively, another interesting observation is Neural Collapse (NC), which means the feature variability may collapse during the terminal phase of training. Motivated by this, we propose to generate UAP by attacking the layer where NC phenomenon happens. Because of NC, the proposed attack could gather all the natural images' features to its surrounding, which is hence called Feature-Gathering UAP (FG-UAP).   We evaluate the effectiveness our proposed algorithm on abundant experiments, including untargeted and targeted universal attacks, attacks under limited dataset, and transfer-based black-box attacks among different architectures including Vision Transformers, which are believed to be more robust. Furthermore, we investigate FG-UAP in the view of NC by analyzing the labels and extracted features of adversarial examples, finding that collapse phenomenon becomes stronger after the model is corrupted. The code will be released when the paper is accepted.



## **13. Cascading Failures in Power Grids**

eess.SY

**SubmitDate**: 2022-09-26    [paper-pdf](http://arxiv.org/pdf/2209.08116v2)

**Authors**: Rounak Meyur

**Abstracts**: This paper studies the consequences of a human-initiated targeted attack on the national electric power system. We consider two kinds of attacks: ($i$) an attack by an adversary that uses a tactical weapon and destroys a large part of the grid, by physically targeting a large geographic region; ($ii$) a targeted attack by an adversary that takes out a small number of critical components in the network simultaneously. Our analysis uses ($i$) a realistic representation of the underlying power grid, including the topology, the control and protection components, ($ii$) a realistic representation of the targeted attack scenario, and ($iii$) a dynamic stability analysis, that goes beyond traditional work comprising structural and linear flow analysis. Such realistic analysis is expensive, but critical since it can capture cascading failures that result from transient instabilities introduced due to the attack. Our model acknowledges the presence of hidden failures in the protection systems resulting in relay misoperations. We analyze the extent of cascading outages for different levels of hidden failures. Our results show that: ($i$) the power grid is vulnerable to both these attacks, ($ii$) the tactical attack has significant social, economic and health damage but need not result in a regional cascade; on the contrary the targeted attack can cause significant cascade and lead to power outage over a large region. Our work shows the necessity to harden the power grid not just to cyber-attacks but also to physical attacks. Furthermore, we show that realistic representations and analysis can lead to fundamentally new insights that simplified models are unlikely to capture. Finally, the methods and results help us identify critical elements in the grid; the system can then be hardened in a more precise manner to reduce the vulnerabilities.



## **14. ASK: Adversarial Soft k-Nearest Neighbor Attack and Defense**

cs.LG

**SubmitDate**: 2022-09-26    [paper-pdf](http://arxiv.org/pdf/2106.14300v4)

**Authors**: Ren Wang, Tianqi Chen, Philip Yao, Sijia Liu, Indika Rajapakse, Alfred Hero

**Abstracts**: K-Nearest Neighbor (kNN)-based deep learning methods have been applied to many applications due to their simplicity and geometric interpretability. However, the robustness of kNN-based classification models has not been thoroughly explored and kNN attack strategies are underdeveloped. In this paper, we propose an Adversarial Soft kNN (ASK) loss to both design more effective kNN attack strategies and to develop better defenses against them. Our ASK loss approach has two advantages. First, ASK loss can better approximate the kNN's probability of classification error than objectives proposed in previous works. Second, the ASK loss is interpretable: it preserves the mutual information between the perturbed input and the in-class-reference data. We use the ASK loss to generate a novel attack method called the ASK-Attack (ASK-Atk), which shows superior attack efficiency and accuracy degradation relative to previous kNN attacks. Based on the ASK-Atk, we then derive an ASK-\underline{Def}ense (ASK-Def) method that optimizes the worst-case training loss induced by ASK-Atk. Experiments on CIFAR-10 (ImageNet) show that (i) ASK-Atk achieves $\geq 13\%$ ($\geq 13\%$) improvement in attack success rate over previous kNN attacks, and (ii) ASK-Def outperforms the conventional adversarial training method by $\geq 6.9\%$ ($\geq 3.5\%$) in terms of robustness improvement.



## **15. Formally verified asymptotic consensus in robust networks**

cs.PL

**SubmitDate**: 2022-09-26    [paper-pdf](http://arxiv.org/pdf/2202.13833v2)

**Authors**: Mohit Tekriwal, Avi Tachna-Fram, Jean-Baptiste Jeannin, Manos Kapritsos, Dimitra Panagou

**Abstracts**: Distributed architectures are used to improve performance and reliability of various systems. An important capability of a distributed architecture is the ability to reach consensus among all its nodes. To achieve this, several consensus algorithms have been proposed for various scenarii, and many of these algorithms come with proofs of correctness that are not mechanically checked. Unfortunately, those proofs are known to be intricate and prone to errors.   In this paper, we formalize and mechanically check a consensus algorithm widely used in the distributed controls community: the Weighted-Mean Subsequence Reduced (W-MSR) algorithm proposed by Le Blanc et al. This algorithm provides a way to achieve asymptotic consensus in a distributed controls scenario in the presence of adversarial agents (attackers) that may not update their states based on the nominal consensus protocol, and may share inaccurate information with their neighbors. Using the Coq proof assistant, we formalize the necessary and sufficient conditions required to achieve resilient asymptotic consensus under the assumed attacker model. We leverage the existing Coq formalizations of graph theory, finite sets and sequences of the mathcomp library for our development. To our knowledge, this is the first mechanical proof of an asymptotic consensus algorithm. During the formalization, we clarify several imprecisions in the paper proof, including an imprecision on quantifiers in the main theorem.



## **16. RORL: Robust Offline Reinforcement Learning via Conservative Smoothing**

cs.LG

Accepted by Advances in Neural Information Processing Systems  (NeurIPS) 2022

**SubmitDate**: 2022-09-26    [paper-pdf](http://arxiv.org/pdf/2206.02829v2)

**Authors**: Rui Yang, Chenjia Bai, Xiaoteng Ma, Zhaoran Wang, Chongjie Zhang, Lei Han

**Abstracts**: Offline reinforcement learning (RL) provides a promising direction to exploit the massive amount of offline data for complex decision-making tasks. Due to the distribution shift issue, current offline RL algorithms are generally designed to be conservative in value estimation and action selection. However, such conservatism can impair the robustness of learned policies when encountering observation deviation under realistic conditions, such as sensor errors and adversarial attacks. To trade off robustness and conservatism, we propose Robust Offline Reinforcement Learning (RORL) with a novel conservative smoothing technique. In RORL, we explicitly introduce regularization on the policy and the value function for states near the dataset, as well as additional conservative value estimation on these OOD states. Theoretically, we show RORL enjoys a tighter suboptimality bound than recent theoretical results in linear MDPs. We demonstrate that RORL can achieve state-of-the-art performance on the general offline RL benchmark and is considerably robust to adversarial observation perturbations.



## **17. Black-Box Dissector: Towards Erasing-based Hard-Label Model Stealing Attack**

cs.CV

**SubmitDate**: 2022-09-26    [paper-pdf](http://arxiv.org/pdf/2105.00623v3)

**Authors**: Yixu Wang, Jie Li, Hong Liu, Yan Wang, Yongjian Wu, Feiyue Huang, Rongrong Ji

**Abstracts**: Previous studies have verified that the functionality of black-box models can be stolen with full probability outputs. However, under the more practical hard-label setting, we observe that existing methods suffer from catastrophic performance degradation. We argue this is due to the lack of rich information in the probability prediction and the overfitting caused by hard labels. To this end, we propose a novel hard-label model stealing method termed \emph{black-box dissector}, which consists of two erasing-based modules. One is a CAM-driven erasing strategy that is designed to increase the information capacity hidden in hard labels from the victim model. The other is a random-erasing-based self-knowledge distillation module that utilizes soft labels from the substitute model to mitigate overfitting. Extensive experiments on four widely-used datasets consistently demonstrate that our method outperforms state-of-the-art methods, with an improvement of at most $8.27\%$. We also validate the effectiveness and practical potential of our method on real-world APIs and defense methods. Furthermore, our method promotes other downstream tasks, \emph{i.e.}, transfer adversarial attacks.



## **18. Activation Learning by Local Competitions**

cs.NE

31pages

**SubmitDate**: 2022-09-26    [paper-pdf](http://arxiv.org/pdf/2209.13400v1)

**Authors**: Hongchao Zhou

**Abstracts**: The backpropagation that drives the success of deep learning is most likely different from the learning mechanism of the brain. In this paper, we develop a biology-inspired learning rule that discovers features by local competitions among neurons, following the idea of Hebb's famous proposal. It is demonstrated that the unsupervised features learned by this local learning rule can serve as a pre-training model to improve the performance of some supervised learning tasks. More importantly, this local learning rule enables us to build a new learning paradigm very different from the backpropagation, named activation learning, where the output activation of the neural network roughly measures how probable the input patterns are. The activation learning is capable of learning plentiful local features from few shots of input patterns, and demonstrates significantly better performances than the backpropagation algorithm when the number of training samples is relatively small. This learning paradigm unifies unsupervised learning, supervised learning and generative models, and is also more secure against adversarial attack, paving a road to some possibilities of creating general-task neural networks.



## **19. Exploiting Trust for Resilient Hypothesis Testing with Malicious Robots**

cs.RO

12 pages, 4 figures, extended version of conference submission

**SubmitDate**: 2022-09-25    [paper-pdf](http://arxiv.org/pdf/2209.12285v1)

**Authors**: Matthew Cavorsi, Orhan Eren Akgün, Michal Yemini, Andrea Goldsmith, Stephanie Gil

**Abstracts**: We develop a resilient binary hypothesis testing framework for decision making in adversarial multi-robot crowdsensing tasks. This framework exploits stochastic trust observations between robots to arrive at tractable, resilient decision making at a centralized Fusion Center (FC) even when i) there exist malicious robots in the network and their number may be larger than the number of legitimate robots, and ii) the FC uses one-shot noisy measurements from all robots. We derive two algorithms to achieve this. The first is the Two Stage Approach (2SA) that estimates the legitimacy of robots based on received trust observations, and provably minimizes the probability of detection error in the worst-case malicious attack. Here, the proportion of malicious robots is known but arbitrary. For the case of an unknown proportion of malicious robots, we develop the Adversarial Generalized Likelihood Ratio Test (A-GLRT) that uses both the reported robot measurements and trust observations to estimate the trustworthiness of robots, their reporting strategy, and the correct hypothesis simultaneously. We exploit special problem structure to show that this approach remains computationally tractable despite several unknown problem parameters. We deploy both algorithms in a hardware experiment where a group of robots conducts crowdsensing of traffic conditions on a mock-up road network similar in spirit to Google Maps, subject to a Sybil attack. We extract the trust observations for each robot from actual communication signals which provide statistical information on the uniqueness of the sender. We show that even when the malicious robots are in the majority, the FC can reduce the probability of detection error to 30.5% and 29% for the 2SA and the A-GLRT respectively.



## **20. Residue-Based Natural Language Adversarial Attack Detection**

cs.CL

**SubmitDate**: 2022-09-25    [paper-pdf](http://arxiv.org/pdf/2204.10192v2)

**Authors**: Vyas Raina, Mark Gales

**Abstracts**: Deep learning based systems are susceptible to adversarial attacks, where a small, imperceptible change at the input alters the model prediction. However, to date the majority of the approaches to detect these attacks have been designed for image processing systems. Many popular image adversarial detection approaches are able to identify adversarial examples from embedding feature spaces, whilst in the NLP domain existing state of the art detection approaches solely focus on input text features, without consideration of model embedding spaces. This work examines what differences result when porting these image designed strategies to Natural Language Processing (NLP) tasks - these detectors are found to not port over well. This is expected as NLP systems have a very different form of input: discrete and sequential in nature, rather than the continuous and fixed size inputs for images. As an equivalent model-focused NLP detection approach, this work proposes a simple sentence-embedding "residue" based detector to identify adversarial examples. On many tasks, it out-performs ported image domain detectors and recent state of the art NLP specific detectors.



## **21. SPRITZ-1.5C: Employing Deep Ensemble Learning for Improving the Security of Computer Networks against Adversarial Attacks**

cs.CR

**SubmitDate**: 2022-09-25    [paper-pdf](http://arxiv.org/pdf/2209.12195v1)

**Authors**: Ehsan Nowroozi, Mohammadreza Mohammadi, Erkay Savas, Mauro Conti, Yassine Mekdad

**Abstracts**: In the past few years, Convolutional Neural Networks (CNN) have demonstrated promising performance in various real-world cybersecurity applications, such as network and multimedia security. However, the underlying fragility of CNN structures poses major security problems, making them inappropriate for use in security-oriented applications including such computer networks. Protecting these architectures from adversarial attacks necessitates using security-wise architectures that are challenging to attack.   In this study, we present a novel architecture based on an ensemble classifier that combines the enhanced security of 1-Class classification (known as 1C) with the high performance of conventional 2-Class classification (known as 2C) in the absence of attacks.Our architecture is referred to as the 1.5-Class (SPRITZ-1.5C) classifier and constructed using a final dense classifier, one 2C classifier (i.e., CNNs), and two parallel 1C classifiers (i.e., auto-encoders). In our experiments, we evaluated the robustness of our proposed architecture by considering eight possible adversarial attacks in various scenarios. We performed these attacks on the 2C and SPRITZ-1.5C architectures separately. The experimental results of our study showed that the Attack Success Rate (ASR) of the I-FGSM attack against a 2C classifier trained with the N-BaIoT dataset is 0.9900. In contrast, the ASR is 0.0000 for the SPRITZ-1.5C classifier.



## **22. Robust Reinforcement Learning as a Stackelberg Game via Adaptively-Regularized Adversarial Training**

cs.LG

**SubmitDate**: 2022-09-24    [paper-pdf](http://arxiv.org/pdf/2202.09514v2)

**Authors**: Peide Huang, Mengdi Xu, Fei Fang, Ding Zhao

**Abstracts**: Robust Reinforcement Learning (RL) focuses on improving performances under model errors or adversarial attacks, which facilitates the real-life deployment of RL agents. Robust Adversarial Reinforcement Learning (RARL) is one of the most popular frameworks for robust RL. However, most of the existing literature models RARL as a zero-sum simultaneous game with Nash equilibrium as the solution concept, which could overlook the sequential nature of RL deployments, produce overly conservative agents, and induce training instability. In this paper, we introduce a novel hierarchical formulation of robust RL - a general-sum Stackelberg game model called RRL-Stack - to formalize the sequential nature and provide extra flexibility for robust training. We develop the Stackelberg Policy Gradient algorithm to solve RRL-Stack, leveraging the Stackelberg learning dynamics by considering the adversary's response. Our method generates challenging yet solvable adversarial environments which benefit RL agents' robust learning. Our algorithm demonstrates better training stability and robustness against different testing conditions in the single-agent robotics control and multi-agent highway merging tasks.



## **23. RSD-GAN: Regularized Sobolev Defense GAN Against Speech-to-Text Adversarial Attacks**

cs.SD

Paper ACCEPTED FOR PUBLICATION IEEE Signal Processing Letters Journal

**SubmitDate**: 2022-09-24    [paper-pdf](http://arxiv.org/pdf/2207.06858v2)

**Authors**: Mohammad Esmaeilpour, Nourhene Chaalia, Patrick Cardinal

**Abstracts**: This paper introduces a new synthesis-based defense algorithm for counteracting with a varieties of adversarial attacks developed for challenging the performance of the cutting-edge speech-to-text transcription systems. Our algorithm implements a Sobolev-based GAN and proposes a novel regularizer for effectively controlling over the functionality of the entire generative model, particularly the discriminator network during training. Our achieved results upon carrying out numerous experiments on the victim DeepSpeech, Kaldi, and Lingvo speech transcription systems corroborate the remarkable performance of our defense approach against a comprehensive range of targeted and non-targeted adversarial attacks.



## **24. Approximate better, Attack stronger: Adversarial Example Generation via Asymptotically Gaussian Mixture Distribution**

cs.LG

**SubmitDate**: 2022-09-24    [paper-pdf](http://arxiv.org/pdf/2209.11964v1)

**Authors**: Zhengwei Fang, Rui Wang, Tao Huang, Liping Jing

**Abstracts**: Strong adversarial examples are the keys to evaluating and enhancing the robustness of deep neural networks. The popular adversarial attack algorithms maximize the non-concave loss function using the gradient ascent. However, the performance of each attack is usually sensitive to, for instance, minor image transformations due to insufficient information (only one input example, few white-box source models and unknown defense strategies). Hence, the crafted adversarial examples are prone to overfit the source model, which limits their transferability to unidentified architectures. In this paper, we propose Multiple Asymptotically Normal Distribution Attacks (MultiANDA), a novel method that explicitly characterizes adversarial perturbations from a learned distribution. Specifically, we approximate the posterior distribution over the perturbations by taking advantage of the asymptotic normality property of stochastic gradient ascent (SGA), then apply the ensemble strategy on this procedure to estimate a Gaussian mixture model for a better exploration of the potential optimization space. Drawing perturbations from the learned distribution allow us to generate any number of adversarial examples for each input. The approximated posterior essentially describes the stationary distribution of SGA iterations, which captures the geometric information around the local optimum. Thus, the samples drawn from the distribution reliably maintain the transferability. Our proposed method outperforms nine state-of-the-art black-box attacks on deep learning models with or without defenses through extensive experiments on seven normally trained and seven defence models.



## **25. Faith: An Efficient Framework for Transformer Verification on GPUs**

cs.LG

Published in ATC'22

**SubmitDate**: 2022-09-23    [paper-pdf](http://arxiv.org/pdf/2209.12708v1)

**Authors**: Boyuan Feng, Tianqi Tang, Yuke Wang, Zhaodong Chen, Zheng Wang, Shu Yang, Yuan Xie, Yufei Ding

**Abstracts**: Transformer verification draws increasing attention in machine learning research and industry. It formally verifies the robustness of transformers against adversarial attacks such as exchanging words in a sentence with synonyms. However, the performance of transformer verification is still not satisfactory due to bound-centric computation which is significantly different from standard neural networks. In this paper, we propose Faith, an efficient framework for transformer verification on GPUs. We first propose a semantic-aware computation graph transformation to identify semantic information such as bound computation in transformer verification. We exploit such semantic information to enable efficient kernel fusion at the computation graph level. Second, we propose a verification-specialized kernel crafter to efficiently map transformer verification to modern GPUs. This crafter exploits a set of GPU hardware supports to accelerate verification specialized operations which are usually memory-intensive. Third, we propose an expert-guided autotuning to incorporate expert knowledge on GPU backends to facilitate large search space exploration. Extensive evaluations show that Faith achieves $2.1\times$ to $3.4\times$ ($2.6\times$ on average) speedup over state-of-the-art frameworks.



## **26. Real-time Adversarial Perturbations against Deep Reinforcement Learning Policies: Attacks and Defenses**

cs.LG

Will appear in the proceedings of ESORICS 2022; 13 pages, 6 figures,  6 tables

**SubmitDate**: 2022-09-23    [paper-pdf](http://arxiv.org/pdf/2106.08746v4)

**Authors**: Buse G. A. Tekgul, Shelly Wang, Samuel Marchal, N. Asokan

**Abstracts**: Deep reinforcement learning (DRL) is vulnerable to adversarial perturbations. Adversaries can mislead the policies of DRL agents by perturbing the state of the environment observed by the agents. Existing attacks are feasible in principle, but face challenges in practice, either by being too slow to fool DRL policies in real time or by modifying past observations stored in the agent's memory. We show that Universal Adversarial Perturbations (UAP), independent of the individual inputs to which they are applied, can fool DRL policies effectively and in real time. We introduce three attack variants leveraging UAP. Via an extensive evaluation using three Atari 2600 games, we show that our attacks are effective, as they fully degrade the performance of three different DRL agents (up to 100%, even when the $l_\infty$ bound on the perturbation is as small as 0.01). It is faster than the frame rate (60 Hz) of image capture and considerably faster than prior attacks ($\approx 1.8$ms). Our attack technique is also efficient, incurring an online computational cost of $\approx 0.027$ms. Using two tasks involving robotic movement, we confirm that our results generalize to complex DRL tasks. Furthermore, we demonstrate that the effectiveness of known defenses diminishes against universal perturbations. We introduce an effective technique that detects all known adversarial perturbations against DRL policies, including all universal perturbations presented in this paper.



## **27. MixTailor: Mixed Gradient Aggregation for Robust Learning Against Tailored Attacks**

cs.LG

To appear at the Transactions on Machine Learning Research (TMLR)

**SubmitDate**: 2022-09-23    [paper-pdf](http://arxiv.org/pdf/2207.07941v2)

**Authors**: Ali Ramezani-Kebrya, Iman Tabrizian, Fartash Faghri, Petar Popovski

**Abstracts**: Implementations of SGD on distributed systems create new vulnerabilities, which can be identified and misused by one or more adversarial agents. Recently, it has been shown that well-known Byzantine-resilient gradient aggregation schemes are indeed vulnerable to informed attackers that can tailor the attacks (Fang et al., 2020; Xie et al., 2020b). We introduce MixTailor, a scheme based on randomization of the aggregation strategies that makes it impossible for the attacker to be fully informed. Deterministic schemes can be integrated into MixTailor on the fly without introducing any additional hyperparameters. Randomization decreases the capability of a powerful adversary to tailor its attacks, while the resulting randomized aggregation scheme is still competitive in terms of performance. For both iid and non-iid settings, we establish almost sure convergence guarantees that are both stronger and more general than those available in the literature. Our empirical studies across various datasets, attacks, and settings, validate our hypothesis and show that MixTailor successfully defends when well-known Byzantine-tolerant schemes fail.



## **28. Reducing Exploitability with Population Based Training**

cs.LG

Presented at New Frontiers in Adversarial Machine Learning Workshop,  ICML 2022

**SubmitDate**: 2022-09-23    [paper-pdf](http://arxiv.org/pdf/2208.05083v2)

**Authors**: Pavel Czempin, Adam Gleave

**Abstracts**: Self-play reinforcement learning has achieved state-of-the-art, and often superhuman, performance in a variety of zero-sum games. Yet prior work has found that policies that are highly capable against regular opponents can fail catastrophically against adversarial policies: an opponent trained explicitly against the victim. Prior defenses using adversarial training were able to make the victim robust to a specific adversary, but the victim remained vulnerable to new ones. We conjecture this limitation was due to insufficient diversity of adversaries seen during training. We propose a defense using population based training to pit the victim against a diverse set of opponents. We evaluate this defense's robustness against new adversaries in two low-dimensional environments. Our defense increases robustness against adversaries, as measured by number of attacker training timesteps to exploit the victim. Furthermore, we show that robustness is correlated with the size of the opponent population.



## **29. Privacy Attacks Against Biometric Models with Fewer Samples: Incorporating the Output of Multiple Models**

cs.CV

This is a major revision of a paper titled "Inverting Biometric  Models with Fewer Samples: Incorporating the Output of Multiple Models" by  the same authors that appears at IJCB 2022

**SubmitDate**: 2022-09-22    [paper-pdf](http://arxiv.org/pdf/2209.11020v1)

**Authors**: Sohaib Ahmad, Benjamin Fuller, Kaleel Mahmood

**Abstracts**: Authentication systems are vulnerable to model inversion attacks where an adversary is able to approximate the inverse of a target machine learning model. Biometric models are a prime candidate for this type of attack. This is because inverting a biometric model allows the attacker to produce a realistic biometric input to spoof biometric authentication systems.   One of the main constraints in conducting a successful model inversion attack is the amount of training data required. In this work, we focus on iris and facial biometric systems and propose a new technique that drastically reduces the amount of training data necessary. By leveraging the output of multiple models, we are able to conduct model inversion attacks with 1/10th the training set size of Ahmad and Fuller (IJCB 2020) for iris data and 1/1000th the training set size of Mai et al. (Pattern Analysis and Machine Intelligence 2019) for facial data. We denote our new attack technique as structured random with alignment loss. Our attacks are black-box, requiring no knowledge of the weights of the target neural network, only the dimension, and values of the output vector.   To show the versatility of the alignment loss, we apply our attack framework to the task of membership inference (Shokri et al., IEEE S&P 2017) on biometric data. For the iris, membership inference attack against classification networks improves from 52% to 62% accuracy.



## **30. In Differential Privacy, There is Truth: On Vote Leakage in Ensemble Private Learning**

cs.LG

To appear at NeurIPS 2022

**SubmitDate**: 2022-09-22    [paper-pdf](http://arxiv.org/pdf/2209.10732v1)

**Authors**: Jiaqi Wang, Roei Schuster, Ilia Shumailov, David Lie, Nicolas Papernot

**Abstracts**: When learning from sensitive data, care must be taken to ensure that training algorithms address privacy concerns. The canonical Private Aggregation of Teacher Ensembles, or PATE, computes output labels by aggregating the predictions of a (possibly distributed) collection of teacher models via a voting mechanism. The mechanism adds noise to attain a differential privacy guarantee with respect to the teachers' training data. In this work, we observe that this use of noise, which makes PATE predictions stochastic, enables new forms of leakage of sensitive information. For a given input, our adversary exploits this stochasticity to extract high-fidelity histograms of the votes submitted by the underlying teachers. From these histograms, the adversary can learn sensitive attributes of the input such as race, gender, or age. Although this attack does not directly violate the differential privacy guarantee, it clearly violates privacy norms and expectations, and would not be possible at all without the noise inserted to obtain differential privacy. In fact, counter-intuitively, the attack becomes easier as we add more noise to provide stronger differential privacy. We hope this encourages future work to consider privacy holistically rather than treat differential privacy as a panacea.



## **31. Fair Robust Active Learning by Joint Inconsistency**

cs.LG

11 pages, 3 figures

**SubmitDate**: 2022-09-22    [paper-pdf](http://arxiv.org/pdf/2209.10729v1)

**Authors**: Tsung-Han Wu, Shang-Tse Chen, Winston H. Hsu

**Abstracts**: Fair Active Learning (FAL) utilized active learning techniques to achieve high model performance with limited data and to reach fairness between sensitive groups (e.g., genders). However, the impact of the adversarial attack, which is vital for various safety-critical machine learning applications, is not yet addressed in FAL. Observing this, we introduce a novel task, Fair Robust Active Learning (FRAL), integrating conventional FAL and adversarial robustness. FRAL requires ML models to leverage active learning techniques to jointly achieve equalized performance on benign data and equalized robustness against adversarial attacks between groups. In this new task, previous FAL methods generally face the problem of unbearable computational burden and ineffectiveness. Therefore, we develop a simple yet effective FRAL strategy by Joint INconsistency (JIN). To efficiently find samples that can boost the performance and robustness of disadvantaged groups for labeling, our method exploits the prediction inconsistency between benign and adversarial samples as well as between standard and robust models. Extensive experiments under diverse datasets and sensitive groups demonstrate that our method not only achieves fairer performance on benign samples but also obtains fairer robustness under white-box PGD attacks compared with existing active learning and FAL baselines. We are optimistic that FRAL would pave a new path for developing safe and robust ML research and applications such as facial attribute recognition in biometrics systems.



## **32. Formulating Robustness Against Unforeseen Attacks**

cs.LG

NeurIPS 2022

**SubmitDate**: 2022-09-21    [paper-pdf](http://arxiv.org/pdf/2204.13779v2)

**Authors**: Sihui Dai, Saeed Mahloujifar, Prateek Mittal

**Abstracts**: Existing defenses against adversarial examples such as adversarial training typically assume that the adversary will conform to a specific or known threat model, such as $\ell_p$ perturbations within a fixed budget. In this paper, we focus on the scenario where there is a mismatch in the threat model assumed by the defense during training, and the actual capabilities of the adversary at test time. We ask the question: if the learner trains against a specific "source" threat model, when can we expect robustness to generalize to a stronger unknown "target" threat model during test-time? Our key contribution is to formally define the problem of learning and generalization with an unforeseen adversary, which helps us reason about the increase in adversarial risk from the conventional perspective of a known adversary. Applying our framework, we derive a generalization bound which relates the generalization gap between source and target threat models to variation of the feature extractor, which measures the expected maximum difference between extracted features across a given threat model. Based on our generalization bound, we propose adversarial training with variation regularization (AT-VR) which reduces variation of the feature extractor across the source threat model during training. We empirically demonstrate that AT-VR can lead to improved generalization to unforeseen attacks during test-time compared to standard adversarial training. Additionally, we combine variation regularization with perceptual adversarial training [Laidlaw et al. 2021] to achieve state-of-the-art robustness on unforeseen attacks. Our code is publicly available at https://github.com/inspire-group/variation-regularization.



## **33. Adversarial Formal Semantics of Attack Trees and Related Problems**

cs.GT

In Proceedings GandALF 2022, arXiv:2209.09333

**SubmitDate**: 2022-09-21    [paper-pdf](http://arxiv.org/pdf/2209.10322v1)

**Authors**: Thomas Brihaye, Sophie Pinchinat, Alexandre Terefenko

**Abstracts**: Security is a subject of increasing attention in our actual society in order to protect critical resources from information disclosure, theft or damage. The informal model of attack trees introduced by Schneier, and widespread in the industry, is advocated in the 2008 NATO report to govern the evaluation of the threat in risk analysis. Attack-defense trees have since been the subject of many theoretical works addressing different formal approaches.   In 2017, M. Audinot et al. introduced a path semantics over a transition system for attack trees. Inspired by the later, we propose a two-player interpretation of the attack-tree formalism. To do so, we replace transition systems by concurrent game arenas and our associated semantics consist of strategies. We then show that the emptiness problem, known to be NP-complete for the path semantics, is now PSPACE-complete. Additionally, we show that the membership problem is coNP-complete for our two-player interpretation while it collapses to P in the path semantics.



## **34. Can You Still See Me?: Reconstructing Robot Operations Over End-to-End Encrypted Channels**

cs.CR

13 pages, 7 figures, 9 tables, Poster presented at wisec'22

**SubmitDate**: 2022-09-21    [paper-pdf](http://arxiv.org/pdf/2205.08426v2)

**Authors**: Ryan Shah, Chuadhry Mujeeb Ahmed, Shishir Nagaraja

**Abstracts**: Connected robots play a key role in Industry 4.0, providing automation and higher efficiency for many industrial workflows. Unfortunately, these robots can leak sensitive information regarding these operational workflows to remote adversaries. While there exists mandates for the use of end-to-end encryption for data transmission in such settings, it is entirely possible for passive adversaries to fingerprint and reconstruct entire workflows being carried out -- establishing an understanding of how facilities operate. In this paper, we investigate whether a remote attacker can accurately fingerprint robot movements and ultimately reconstruct operational workflows. Using a neural network approach to traffic analysis, we find that one can predict TLS-encrypted movements with around ~60% accuracy, increasing to near-perfect accuracy under realistic network conditions. Further, we also find that attackers can reconstruct warehousing workflows with similar success. Ultimately, simply adopting best cybersecurity practices is clearly not enough to stop even weak (passive) adversaries.



## **35. Fingerprinting Robot Movements via Acoustic Side Channel**

cs.CR

11 pages, 4 figures, 7 tables

**SubmitDate**: 2022-09-21    [paper-pdf](http://arxiv.org/pdf/2209.10240v1)

**Authors**: Ryan Shah, Mujeeb Ahmed, Shishir Nagaraja

**Abstracts**: In this paper, we present an acoustic side channel attack which makes use of smartphone microphones recording a robot in operation to exploit acoustic properties of the sound to fingerprint a robot's movements. In this work we consider the possibility of an insider adversary who is within physical proximity of a robotic system (such as a technician or robot operator), equipped with only their smartphone microphone. Through the acoustic side-channel, we demonstrate that it is indeed possible to fingerprint not only individual robot movements within 3D space, but also patterns of movements which could lead to inferring the purpose of the movements (i.e. surgical procedures which a surgical robot is undertaking) and hence, resulting in potential privacy violations. Upon evaluation, we find that individual robot movements can be fingerprinted with around 75% accuracy, decreasing slightly with more fine-grained movement meta-data such as distance and speed. Furthermore, workflows could be reconstructed with around 62% accuracy as a whole, with more complex movements such as pick-and-place or packing reconstructed with near perfect accuracy. As well as this, in some environments such as surgical settings, audio may be recorded and transmitted over VoIP, such as for education/teaching purposes or in remote telemedicine. The question here is, can the same attack be successful even when VoIP communication is employed, and how does packet loss impact the captured audio and the success of the attack? Using the same characteristics of acoustic sound for plain audio captured by the smartphone, the attack was 90% accurate in fingerprinting VoIP samples on average, 15% higher than the baseline without the VoIP codec employed. This opens up new research questions regarding anonymous communications to protect robotic systems from acoustic side channel attacks via VoIP communication networks.



## **36. Reconstructing Robot Operations via Radio-Frequency Side-Channel**

cs.CR

10 pages, 7 figures, 4 tables

**SubmitDate**: 2022-09-21    [paper-pdf](http://arxiv.org/pdf/2209.10179v1)

**Authors**: Ryan Shah, Mujeeb Ahmed, Shishir Nagaraja

**Abstracts**: Connected teleoperated robotic systems play a key role in ensuring operational workflows are carried out with high levels of accuracy and low margins of error. In recent years, a variety of attacks have been proposed that actively target the robot itself from the cyber domain. However, little attention has been paid to the capabilities of a passive attacker. In this work, we investigate whether an insider adversary can accurately fingerprint robot movements and operational warehousing workflows via the radio frequency side channel in a stealthy manner. Using an SVM for classification, we found that an adversary can fingerprint individual robot movements with at least 96% accuracy, increasing to near perfect accuracy when reconstructing entire warehousing workflows.



## **37. Audit and Improve Robustness of Private Neural Networks on Encrypted Data**

cs.LG

10 pages, 10 figures

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2209.09996v1)

**Authors**: Jiaqi Xue, Lei Xu, Lin Chen, Weidong Shi, Kaidi Xu, Qian Lou

**Abstracts**: Performing neural network inference on encrypted data without decryption is one popular method to enable privacy-preserving neural networks (PNet) as a service. Compared with regular neural networks deployed for machine-learning-as-a-service, PNet requires additional encoding, e.g., quantized-precision numbers, and polynomial activation. Encrypted input also introduces novel challenges such as adversarial robustness and security. To the best of our knowledge, we are the first to study questions including (i) Whether PNet is more robust against adversarial inputs than regular neural networks? (ii) How to design a robust PNet given the encrypted input without decryption? We propose PNet-Attack to generate black-box adversarial examples that can successfully attack PNet in both target and untarget manners. The attack results show that PNet robustness against adversarial inputs needs to be improved. This is not a trivial task because the PNet model owner does not have access to the plaintext of the input values, which prevents the application of existing detection and defense methods such as input tuning, model normalization, and adversarial training. To tackle this challenge, we propose a new fast and accurate noise insertion method, called RPNet, to design Robust and Private Neural Networks. Our comprehensive experiments show that PNet-Attack reduces at least $2.5\times$ queries than prior works. We theoretically analyze our RPNet methods and demonstrate that RPNet can decrease $\sim 91.88\%$ attack success rate.



## **38. SoK: Decentralized Finance (DeFi) Attacks**

cs.CR

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2208.13035v2)

**Authors**: Liyi Zhou, Xihan Xiong, Jens Ernstberger, Stefanos Chaliasos, Zhipeng Wang, Ye Wang, Kaihua Qin, Roger Wattenhofer, Dawn Song, Arthur Gervais

**Abstracts**: Within just four years, the blockchain-based Decentralized Finance (DeFi) ecosystem has accumulated a peak total value locked (TVL) of more than 253 billion USD. This surge in DeFi's popularity has, unfortunately, been accompanied by many impactful incidents. According to our data, users, liquidity providers, speculators, and protocol operators suffered a total loss of at least 3.24 USD from Apr 30, 2018 to Apr 30, 2022. Given the blockchain's transparency and increasing incident frequency, two questions arise: How can we systematically measure, evaluate, and compare DeFi incidents? How can we learn from past attacks to strengthen DeFi security?   In this paper, we introduce a common reference frame to systematically evaluate and compare DeFi incidents, including both attacks and accidents. We investigate 77 academic papers, 30 audit reports, and 181 real-world incidents. Our open data reveals several gaps between academia and the practitioners' community. For example, few academic papers address "price oracle attacks" and "permissonless interactions", while our data suggests that they are the two most frequent incident types (15% and 10.5% correspondingly). We also investigate potential defenses, and find that: (i) 103 (56%) of the attacks are not executed atomically, granting a rescue time frame for defenders; (ii) SoTA bytecode similarity analysis can at least detect 31 vulnerable/23 adversarial contracts; and (iii) 33 (15.3%) of the adversaries leak potentially identifiable information by interacting with centralized exchanges.



## **39. Leveraging Local Patch Differences in Multi-Object Scenes for Generative Adversarial Attacks**

cs.CV

Accepted at WACV 2023 (Round 1)

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2209.09883v1)

**Authors**: Abhishek Aich, Shasha Li, Chengyu Song, M. Salman Asif, Srikanth V. Krishnamurthy, Amit K. Roy-Chowdhury

**Abstracts**: State-of-the-art generative model-based attacks against image classifiers overwhelmingly focus on single-object (i.e., single dominant object) images. Different from such settings, we tackle a more practical problem of generating adversarial perturbations using multi-object (i.e., multiple dominant objects) images as they are representative of most real-world scenes. Our goal is to design an attack strategy that can learn from such natural scenes by leveraging the local patch differences that occur inherently in such images (e.g. difference between the local patch on the object `person' and the object `bike' in a traffic scene). Our key idea is: to misclassify an adversarial multi-object image, each local patch in the image should confuse the victim classifier. Based on this, we propose a novel generative attack (called Local Patch Difference or LPD-Attack) where a novel contrastive loss function uses the aforesaid local differences in feature space of multi-object scenes to optimize the perturbation generator. Through various experiments across diverse victim convolutional neural networks, we show that our approach outperforms baseline generative attacks with highly transferable perturbations when evaluated under different white-box and black-box settings.



## **40. Sparse Vicious Attacks on Graph Neural Networks**

cs.LG

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2209.09688v1)

**Authors**: Giovanni Trappolini, Valentino Maiorca, Silvio Severino, Emanuele Rodolà, Fabrizio Silvestri, Gabriele Tolomei

**Abstracts**: Graph Neural Networks (GNNs) have proven to be successful in several predictive modeling tasks for graph-structured data.   Amongst those tasks, link prediction is one of the fundamental problems for many real-world applications, such as recommender systems.   However, GNNs are not immune to adversarial attacks, i.e., carefully crafted malicious examples that are designed to fool the predictive model.   In this work, we focus on a specific, white-box attack to GNN-based link prediction models, where a malicious node aims to appear in the list of recommended nodes for a given target victim.   To achieve this goal, the attacker node may also count on the cooperation of other existing peers that it directly controls, namely on the ability to inject a number of ``vicious'' nodes in the network.   Specifically, all these malicious nodes can add new edges or remove existing ones, thereby perturbing the original graph.   Thus, we propose SAVAGE, a novel framework and a method to mount this type of link prediction attacks.   SAVAGE formulates the adversary's goal as an optimization task, striking the balance between the effectiveness of the attack and the sparsity of malicious resources required.   Extensive experiments conducted on real-world and synthetic datasets demonstrate that adversarial attacks implemented through SAVAGE indeed achieve high attack success rate yet using a small amount of vicious nodes.   Finally, despite those attacks require full knowledge of the target model, we show that they are successfully transferable to other black-box methods for link prediction.



## **41. I-GWAS: Privacy-Preserving Interdependent Genome-Wide Association Studies**

q-bio.GN

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2208.08361v2)

**Authors**: Túlio Pascoal, Jérémie Decouchant, Antoine Boutet, Marcus Völp

**Abstracts**: Genome-wide Association Studies (GWASes) identify genomic variations that are statistically associated with a trait, such as a disease, in a group of individuals. Unfortunately, careless sharing of GWAS statistics might give rise to privacy attacks. Several works attempted to reconcile secure processing with privacy-preserving releases of GWASes. However, we highlight that these approaches remain vulnerable if GWASes utilize overlapping sets of individuals and genomic variations. In such conditions, we show that even when relying on state-of-the-art techniques for protecting releases, an adversary could reconstruct the genomic variations of up to 28.6% of participants, and that the released statistics of up to 92.3% of the genomic variations would enable membership inference attacks. We introduce I-GWAS, a novel framework that securely computes and releases the results of multiple possibly interdependent GWASes. I-GWAS continuously releases privacy-preserving and noise-free GWAS results as new genomes become available.



## **42. FrequencyLowCut Pooling -- Plug & Play against Catastrophic Overfitting**

cs.CV

accepted at ECCV 2022

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2204.00491v2)

**Authors**: Julia Grabinski, Steffen Jung, Janis Keuper, Margret Keuper

**Abstracts**: Over the last years, Convolutional Neural Networks (CNNs) have been the dominating neural architecture in a wide range of computer vision tasks. From an image and signal processing point of view, this success might be a bit surprising as the inherent spatial pyramid design of most CNNs is apparently violating basic signal processing laws, i.e. Sampling Theorem in their down-sampling operations. However, since poor sampling appeared not to affect model accuracy, this issue has been broadly neglected until model robustness started to receive more attention. Recent work [17] in the context of adversarial attacks and distribution shifts, showed after all, that there is a strong correlation between the vulnerability of CNNs and aliasing artifacts induced by poor down-sampling operations. This paper builds on these findings and introduces an aliasing free down-sampling operation which can easily be plugged into any CNN architecture: FrequencyLowCut pooling. Our experiments show, that in combination with simple and fast FGSM adversarial training, our hyper-parameter free operator significantly improves model robustness and avoids catastrophic overfitting.



## **43. GAMA: Generative Adversarial Multi-Object Scene Attacks**

cs.CV

Accepted at NeurIPS 2022; First two authors contributed equally;  Includes Supplementary Material

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2209.09502v1)

**Authors**: Abhishek Aich, Calvin Khang-Ta, Akash Gupta, Chengyu Song, Srikanth V. Krishnamurthy, M. Salman Asif, Amit K. Roy-Chowdhury

**Abstracts**: The majority of methods for crafting adversarial attacks have focused on scenes with a single dominant object (e.g., images from ImageNet). On the other hand, natural scenes include multiple dominant objects that are semantically related. Thus, it is crucial to explore designing attack strategies that look beyond learning on single-object scenes or attack single-object victim classifiers. Due to their inherent property of strong transferability of perturbations to unknown models, this paper presents the first approach of using generative models for adversarial attacks on multi-object scenes. In order to represent the relationships between different objects in the input scene, we leverage upon the open-sourced pre-trained vision-language model CLIP (Contrastive Language-Image Pre-training), with the motivation to exploit the encoded semantics in the language space along with the visual space. We call this attack approach Generative Adversarial Multi-object scene Attacks (GAMA). GAMA demonstrates the utility of the CLIP model as an attacker's tool to train formidable perturbation generators for multi-object scenes. Using the joint image-text features to train the generator, we show that GAMA can craft potent transferable perturbations in order to fool victim classifiers in various attack settings. For example, GAMA triggers ~16% more misclassification than state-of-the-art generative approaches in black-box settings where both the classifier architecture and data distribution of the attacker are different from the victim. Our code will be made publicly available soon.



## **44. Learn2Weight: Parameter Adaptation against Similar-domain Adversarial Attacks**

cs.LG

Accepted in COLING 2022

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2205.07315v2)

**Authors**: Siddhartha Datta

**Abstracts**: Recent work in black-box adversarial attacks for NLP systems has attracted much attention. Prior black-box attacks assume that attackers can observe output labels from target models based on selected inputs. In this work, inspired by adversarial transferability, we propose a new type of black-box NLP adversarial attack that an attacker can choose a similar domain and transfer the adversarial examples to the target domain and cause poor performance in target model. Based on domain adaptation theory, we then propose a defensive strategy, called Learn2Weight, which trains to predict the weight adjustments for a target model in order to defend against an attack of similar-domain adversarial examples. Using Amazon multi-domain sentiment classification datasets, we empirically show that Learn2Weight is effective against the attack compared to standard black-box defense methods such as adversarial training and defensive distillation. This work contributes to the growing literature on machine learning safety.



## **45. Security and Privacy of Wireless Beacon Systems**

cs.CR

13 pages, 3 figures

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2107.05868v2)

**Authors**: Aldar C-F. Chan, Raymond M. H. Chung

**Abstracts**: Bluetooth Low Energy (BLE) beacons have been increasingly used in smart city applications, such as location-based and proximity-based services, to enable Internet of Things to interact with people in vicinity or enhance context-awareness. Their widespread deployment in human-centric applications makes them an attractive target to adversaries for social or economic reasons. In fact, beacons are reportedly exposed to various security issues and privacy concerns. A characterization of attacks against beacon systems is given to help understand adversary motives, required adversarial capabilities, potential impact and possible defence mechanisms for different threats, with a view to facilitating security evaluation and protection formulation for beacon systems.



## **46. Parallel Proof-of-Work with Concrete Bounds**

cs.CR

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2204.00034v2)

**Authors**: Patrik Keller, Rainer Böhme

**Abstracts**: Authorization is challenging in distributed systems that cannot rely on the identification of nodes. Proof-of-work offers an alternative gate-keeping mechanism, but its probabilistic nature is incompatible with conventional security definitions. Recent related work establishes concrete bounds for the failure probability of Bitcoin's sequential proof-of-work mechanism. We propose a family of state replication protocols using parallel proof-of-work. Our bottom-up design from an agreement sub-protocol allows us to give concrete bounds for the failure probability in adversarial synchronous networks. After the typical interval of 10 minutes, parallel proof-of-work offers two orders of magnitude more security than sequential proof-of-work. This means that state updates can be sufficiently secure to support commits after one block (i.e., after 10 minutes), removing the risk of double-spending in many applications. We offer guidance on the optimal choice of parameters for a wide range of network and attacker assumptions. Simulations show that the proposed construction is robust against violations of design assumptions.



## **47. A Transferable and Automatic Tuning of Deep Reinforcement Learning for Cost Effective Phishing Detection**

cs.CR

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2209.09033v1)

**Authors**: Orel Lavie, Asaf Shabtai, Gilad Katz

**Abstracts**: Many challenging real-world problems require the deployment of ensembles multiple complementary learning models to reach acceptable performance levels. While effective, applying the entire ensemble to every sample is costly and often unnecessary. Deep Reinforcement Learning (DRL) offers a cost-effective alternative, where detectors are dynamically chosen based on the output of their predecessors, with their usefulness weighted against their computational cost. Despite their potential, DRL-based solutions are not widely used in this capacity, partly due to the difficulties in configuring the reward function for each new task, the unpredictable reactions of the DRL agent to changes in the data, and the inability to use common performance metrics (e.g., TPR/FPR) to guide the algorithm's performance. In this study we propose methods for fine-tuning and calibrating DRL-based policies so that they can meet multiple performance goals. Moreover, we present a method for transferring effective security policies from one dataset to another. Finally, we demonstrate that our approach is highly robust against adversarial attacks.



## **48. Encrypted Semantic Communication Using Adversarial Training for Privacy Preserving**

cs.IT

submitted to IEEE Wireless Communications Letters

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2209.09008v1)

**Authors**: Xinlai Luo, Zhiyong Chen, Meixia Tao, Feng Yang

**Abstracts**: Semantic communication is implemented based on shared background knowledge, but the sharing mechanism risks privacy leakage. In this letter, we propose an encrypted semantic communication system (ESCS) for privacy preserving, which combines universality and confidentiality. The universality is reflected in that all network modules of the proposed ESCS are trained based on a shared database, which is suitable for large-scale deployment in practical scenarios. Meanwhile, the confidentiality is achieved by symmetric encryption. Based on the adversarial training, we design an adversarial encryption training scheme to guarantee the accuracy of semantic communication in both encrypted and unencrypted modes. Experiment results show that the proposed ESCS with the adversarial encryption training scheme can perform well regardless of whether the semantic information is encrypted. It is difficult for the attacker to reconstruct the original semantic information from the eavesdropped message.



## **49. Catoptric Light can be Dangerous: Effective Physical-World Attack by Natural Phenomenon**

cs.CV

arXiv admin note: substantial text overlap with arXiv:2209.09652,  arXiv:2209.02430

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2209.11739v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstracts**: Deep neural networks (DNNs) have achieved great success in many tasks. Therefore, it is crucial to evaluate the robustness of advanced DNNs. The traditional methods use stickers as physical perturbations to fool the classifiers, which is difficult to achieve stealthiness and there exists printing loss. Some new types of physical attacks use light beam to perform attacks (e.g., laser, projector), whose optical patterns are artificial rather than natural. In this work, we study a new type of physical attack, called adversarial catoptric light (AdvCL), in which adversarial perturbations are generated by common natural phenomena, catoptric light, to achieve stealthy and naturalistic adversarial attacks against advanced DNNs in physical environments. Carefully designed experiments demonstrate the effectiveness of the proposed method in simulated and real-world environments. The attack success rate is 94.90% in a subset of ImageNet and 83.50% in the real-world environment. We also discuss some of AdvCL's transferability and defense strategy against this attack.



## **50. Adversarial Color Projection: A Projector-Based Physical Attack to DNNs**

cs.CR

arXiv admin note: substantial text overlap with arXiv:2209.02430

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2209.09652v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstracts**: Recent advances have shown that deep neural networks (DNNs) are susceptible to adversarial perturbations. Therefore, it is necessary to evaluate the robustness of advanced DNNs using adversarial attacks. However, traditional physical attacks that use stickers as perturbations are more vulnerable than recent light-based physical attacks. In this work, we propose a projector-based physical attack called adversarial color projection (AdvCP), which performs an adversarial attack by manipulating the physical parameters of the projected light. Experiments show the effectiveness of our method in both digital and physical environments. The experimental results demonstrate that the proposed method has excellent attack transferability, which endows AdvCP with effective blackbox attack. We prospect AdvCP threats to future vision-based systems and applications and propose some ideas for light-based physical attacks.



