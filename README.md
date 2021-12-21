# Latest Adversarial Attack Papers
**update at 2021-12-21 17:05:41**

[中文版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. An Evasion Attack against Stacked Capsule Autoencoder**

cs.LG

**SubmitDate**: 2021-12-20    [paper-pdf](http://arxiv.org/pdf/2010.07230v5)

**Authors**: Jiazhu Dai, Siwei Xiong

**Abstracts**: Capsule network is a type of neural network that uses the spatial relationship between features to classify images. By capturing the poses and relative positions between features, its ability to recognize affine transformation is improved, and it surpasses traditional convolutional neural networks (CNNs) when handling translation, rotation and scaling. The Stacked Capsule Autoencoder (SCAE) is the state-of-the-art capsule network. The SCAE encodes an image as capsules, each of which contains poses of features and their correlations. The encoded contents are then input into the downstream classifier to predict the categories of the images. Existing research mainly focuses on the security of capsule networks with dynamic routing or EM routing, and little attention has been given to the security and robustness of the SCAE. In this paper, we propose an evasion attack against the SCAE. After a perturbation is generated based on the output of the object capsules in the model, it is added to an image to reduce the contribution of the object capsules related to the original category of the image so that the perturbed image will be misclassified. We evaluate the attack using an image classification experiment, and the experimental results indicate that the attack can achieve high success rates and stealthiness. It confirms that the SCAE has a security vulnerability whereby it is possible to craft adversarial samples without changing the original structure of the image to fool the classifiers. We hope that our work will make the community aware of the threat of this attack and raise the attention given to the SCAE's security.



## **2. Adversarial Attacks on Spiking Convolutional Networks for Event-based Vision**

cs.CV

16 pages, preprint, submitted to ICLR 2022

**SubmitDate**: 2021-12-20    [paper-pdf](http://arxiv.org/pdf/2110.02929v2)

**Authors**: Julian Büchel, Gregor Lenz, Yalun Hu, Sadique Sheik, Martino Sorbaro

**Abstracts**: Event-based sensing using dynamic vision sensors is gaining traction in low-power vision applications. Spiking neural networks work well with the sparse nature of event-based data and suit deployment on low-power neuromorphic hardware. Being a nascent field, the sensitivity of spiking neural networks to potentially malicious adversarial attacks has received very little attention so far. In this work, we show how white-box adversarial attack algorithms can be adapted to the discrete and sparse nature of event-based visual data, and to the continuous-time setting of spiking neural networks. We test our methods on the N-MNIST and IBM Gestures neuromorphic vision datasets and show adversarial perturbations achieve a high success rate, by injecting a relatively small number of appropriately placed events. We also verify, for the first time, the effectiveness of these perturbations directly on neuromorphic hardware. Finally, we discuss the properties of the resulting perturbations and possible future directions.



## **3. Certified Federated Adversarial Training**

cs.LG

First presented at the 1st NeurIPS Workshop on New Frontiers in  Federated Learning (NFFL 2021)

**SubmitDate**: 2021-12-20    [paper-pdf](http://arxiv.org/pdf/2112.10525v1)

**Authors**: Giulio Zizzo, Ambrish Rawat, Mathieu Sinn, Sergio Maffeis, Chris Hankin

**Abstracts**: In federated learning (FL), robust aggregation schemes have been developed to protect against malicious clients. Many robust aggregation schemes rely on certain numbers of benign clients being present in a quorum of workers. This can be hard to guarantee when clients can join at will, or join based on factors such as idle system status, and connected to power and WiFi. We tackle the scenario of securing FL systems conducting adversarial training when a quorum of workers could be completely malicious. We model an attacker who poisons the model to insert a weakness into the adversarial training such that the model displays apparent adversarial robustness, while the attacker can exploit the inserted weakness to bypass the adversarial training and force the model to misclassify adversarial examples. We use abstract interpretation techniques to detect such stealthy attacks and block the corrupted model updates. We show that this defence can preserve adversarial robustness even against an adaptive attacker.



## **4. Unifying Model Explainability and Robustness for Joint Text Classification and Rationale Extraction**

cs.CL

AAAI 2022

**SubmitDate**: 2021-12-20    [paper-pdf](http://arxiv.org/pdf/2112.10424v1)

**Authors**: Dongfang Li, Baotian Hu, Qingcai Chen, Tujie Xu, Jingcong Tao, Yunan Zhang

**Abstracts**: Recent works have shown explainability and robustness are two crucial ingredients of trustworthy and reliable text classification. However, previous works usually address one of two aspects: i) how to extract accurate rationales for explainability while being beneficial to prediction; ii) how to make the predictive model robust to different types of adversarial attacks. Intuitively, a model that produces helpful explanations should be more robust against adversarial attacks, because we cannot trust the model that outputs explanations but changes its prediction under small perturbations. To this end, we propose a joint classification and rationale extraction model named AT-BMC. It includes two key mechanisms: mixed Adversarial Training (AT) is designed to use various perturbations in discrete and embedding space to improve the model's robustness, and Boundary Match Constraint (BMC) helps to locate rationales more precisely with the guidance of boundary information. Performances on benchmark datasets demonstrate that the proposed AT-BMC outperforms baselines on both classification and rationale extraction by a large margin. Robustness analysis shows that the proposed AT-BMC decreases the attack success rate effectively by up to 69%. The empirical results indicate that there are connections between robust models and better explanations.



## **5. Knowledge Cross-Distillation for Membership Privacy**

cs.CR

Under Review

**SubmitDate**: 2021-12-20    [paper-pdf](http://arxiv.org/pdf/2111.01363v2)

**Authors**: Rishav Chourasia, Batnyam Enkhtaivan, Kunihiro Ito, Junki Mori, Isamu Teranishi, Hikaru Tsuchida

**Abstracts**: A membership inference attack (MIA) poses privacy risks on the training data of a machine learning model. With an MIA, an attacker guesses if the target data are a member of the training dataset. The state-of-the-art defense against MIAs, distillation for membership privacy (DMP), requires not only private data to protect but a large amount of unlabeled public data. However, in certain privacy-sensitive domains, such as medical and financial, the availability of public data is not obvious. Moreover, a trivial method to generate the public data by using generative adversarial networks significantly decreases the model accuracy, as reported by the authors of DMP. To overcome this problem, we propose a novel defense against MIAs using knowledge distillation without requiring public data. Our experiments show that the privacy protection and accuracy of our defense are comparable with those of DMP for the benchmark tabular datasets used in MIA researches, Purchase100 and Texas100, and our defense has much better privacy-utility trade-off than those of the existing defenses without using public data for image dataset CIFAR10.



## **6. Toward Evaluating Re-identification Risks in the Local Privacy Model**

cs.CR

Accepted at Transactions on Data Privacy

**SubmitDate**: 2021-12-19    [paper-pdf](http://arxiv.org/pdf/2010.08238v5)

**Authors**: Takao Murakami, Kenta Takahashi

**Abstracts**: LDP (Local Differential Privacy) has recently attracted much attention as a metric of data privacy that prevents the inference of personal data from obfuscated data in the local model. However, there are scenarios in which the adversary wants to perform re-identification attacks to link the obfuscated data to users in this model. LDP can cause excessive obfuscation and destroy the utility in these scenarios because it is not designed to directly prevent re-identification. In this paper, we propose a measure of re-identification risks, which we call PIE (Personal Information Entropy). The PIE is designed so that it directly prevents re-identification attacks in the local model. It lower-bounds the lowest possible re-identification error probability (i.e., Bayes error probability) of the adversary. We analyze the relation between LDP and the PIE, and analyze the PIE and utility in distribution estimation for two obfuscation mechanisms providing LDP. Through experiments, we show that when we consider re-identification as a privacy risk, LDP can cause excessive obfuscation and destroy the utility. Then we show that the PIE can be used to guarantee low re-identification risks for the local obfuscation mechanisms while keeping high utility.



## **7. Attacking Point Cloud Segmentation with Color-only Perturbation**

cs.CV

**SubmitDate**: 2021-12-18    [paper-pdf](http://arxiv.org/pdf/2112.05871v2)

**Authors**: Jiacen Xu, Zhe Zhou, Boyuan Feng, Yufei Ding, Zhou Li

**Abstracts**: Recent research efforts on 3D point-cloud semantic segmentation have achieved outstanding performance by adopting deep CNN (convolutional neural networks) and GCN (graph convolutional networks). However, the robustness of these complex models has not been systematically analyzed. Given that semantic segmentation has been applied in many safety-critical applications (e.g., autonomous driving, geological sensing), it is important to fill this knowledge gap, in particular, how these models are affected under adversarial samples. While adversarial attacks against point cloud have been studied, we found all of them were targeting single-object recognition, and the perturbation is done on the point coordinates. We argue that the coordinate-based perturbation is unlikely to realize under the physical-world constraints. Hence, we propose a new color-only perturbation method named COLPER, and tailor it to semantic segmentation. By evaluating COLPER on an indoor dataset (S3DIS) and an outdoor dataset (Semantic3D) against three point cloud segmentation models (PointNet++, DeepGCNs, and RandLA-Net), we found color-only perturbation is sufficient to significantly drop the segmentation accuracy and aIoU, under both targeted and non-targeted attack settings.



## **8. Adversarial Attack for Uncertainty Estimation: Identifying Critical Regions in Neural Networks**

cs.LG

15 pages, 6 figures, Neural Process Lett (2021)

**SubmitDate**: 2021-12-18    [paper-pdf](http://arxiv.org/pdf/2107.07618v2)

**Authors**: Ismail Alarab, Simant Prakoonwit

**Abstracts**: We propose a novel method to capture data points near decision boundary in neural network that are often referred to a specific type of uncertainty. In our approach, we sought to perform uncertainty estimation based on the idea of adversarial attack method. In this paper, uncertainty estimates are derived from the input perturbations, unlike previous studies that provide perturbations on the model's parameters as in Bayesian approach. We are able to produce uncertainty with couple of perturbations on the inputs. Interestingly, we apply the proposed method to datasets derived from blockchain. We compare the performance of model uncertainty with the most recent uncertainty methods. We show that the proposed method has revealed a significant outperformance over other methods and provided less risk to capture model uncertainty in machine learning.



## **9. Dynamic Defender-Attacker Blotto Game**

eess.SY

**SubmitDate**: 2021-12-18    [paper-pdf](http://arxiv.org/pdf/2112.09890v1)

**Authors**: Daigo Shishika, Yue Guan, Michael Dorothy, Vijay Kumar

**Abstracts**: This work studies a dynamic, adversarial resource allocation problem in environments modeled as graphs. A blue team of defender robots are deployed in the environment to protect the nodes from a red team of attacker robots. We formulate the engagement as a discrete-time dynamic game, where the robots can move at most one hop in each time step. The game terminates with the attacker's win if any location has more attacker robots than defender robots at any time. The goal is to identify dynamic resource allocation strategies, as well as the conditions that determines the winner: graph structure, available resources, and initial conditions. We analyze the problem using reachable sets and show how the outdegree of the underlying graph directly influences the difficulty of the defending task. Furthermore, we provide algorithms that identify sufficiency of attacker's victory.



## **10. Formalizing Generalization and Robustness of Neural Networks to Weight Perturbations**

cs.LG

This version has been accepted for poster presentation at NeurIPS  2021

**SubmitDate**: 2021-12-17    [paper-pdf](http://arxiv.org/pdf/2103.02200v2)

**Authors**: Yu-Lin Tsai, Chia-Yi Hsu, Chia-Mu Yu, Pin-Yu Chen

**Abstracts**: Studying the sensitivity of weight perturbation in neural networks and its impacts on model performance, including generalization and robustness, is an active research topic due to its implications on a wide range of machine learning tasks such as model compression, generalization gap assessment, and adversarial attacks. In this paper, we provide the first integral study and analysis for feed-forward neural networks in terms of the robustness in pairwise class margin and its generalization behavior under weight perturbation. We further design a new theory-driven loss function for training generalizable and robust neural networks against weight perturbations. Empirical experiments are conducted to validate our theoretical analysis. Our results offer fundamental insights for characterizing the generalization and robustness of neural networks against weight perturbations.



## **11. Reasoning Chain Based Adversarial Attack for Multi-hop Question Answering**

cs.CL

10 pages including reference, 4 figures

**SubmitDate**: 2021-12-17    [paper-pdf](http://arxiv.org/pdf/2112.09658v1)

**Authors**: Jiayu Ding, Siyuan Wang, Qin Chen, Zhongyu Wei

**Abstracts**: Recent years have witnessed impressive advances in challenging multi-hop QA tasks. However, these QA models may fail when faced with some disturbance in the input text and their interpretability for conducting multi-hop reasoning remains uncertain. Previous adversarial attack works usually edit the whole question sentence, which has limited effect on testing the entity-based multi-hop inference ability. In this paper, we propose a multi-hop reasoning chain based adversarial attack method. We formulate the multi-hop reasoning chains starting from the query entity to the answer entity in the constructed graph, which allows us to align the question to each reasoning hop and thus attack any hop. We categorize the questions into different reasoning types and adversarially modify part of the question corresponding to the selected reasoning hop to generate the distracting sentence. We test our adversarial scheme on three QA models on HotpotQA dataset. The results demonstrate significant performance reduction on both answer and supporting facts prediction, verifying the effectiveness of our reasoning chain based attack method for multi-hop reasoning models and the vulnerability of them. Our adversarial re-training further improves the performance and robustness of these models.



## **12. Who Is the Strongest Enemy? Towards Optimal and Efficient Evasion Attacks in Deep RL**

cs.LG

**SubmitDate**: 2021-12-17    [paper-pdf](http://arxiv.org/pdf/2106.05087v2)

**Authors**: Yanchao Sun, Ruijie Zheng, Yongyuan Liang, Furong Huang

**Abstracts**: Evaluating the worst-case performance of a reinforcement learning (RL) agent under the strongest/optimal adversarial perturbations on state observations (within some constraints) is crucial for understanding the robustness of RL agents. However, finding the optimal adversary is challenging, in terms of both whether we can find the optimal attack and how efficiently we can find it. Existing works on adversarial RL either use heuristics-based methods that may not find the strongest adversary, or directly train an RL-based adversary by treating the agent as a part of the environment, which can find the optimal adversary but may become intractable in a large state space. This paper introduces a novel attacking method to find the optimal attacks through collaboration between a designed function named ''actor'' and an RL-based learner named "director". The actor crafts state perturbations for a given policy perturbation direction, and the director learns to propose the best policy perturbation directions. Our proposed algorithm, PA-AD, is theoretically optimal and significantly more efficient than prior RL-based works in environments with large state spaces. Empirical results show that our proposed PA-AD universally outperforms state-of-the-art attacking methods in various Atari and MuJoCo environments. By applying PA-AD to adversarial training, we achieve state-of-the-art empirical robustness in multiple tasks under strong adversaries.



## **13. Dynamics-aware Adversarial Attack of 3D Sparse Convolution Network**

cs.CV

**SubmitDate**: 2021-12-17    [paper-pdf](http://arxiv.org/pdf/2112.09428v1)

**Authors**: An Tao, Yueqi Duan, He Wang, Ziyi Wu, Pengliang Ji, Haowen Sun, Jie Zhou, Jiwen Lu

**Abstracts**: In this paper, we investigate the dynamics-aware adversarial attack problem in deep neural networks. Most existing adversarial attack algorithms are designed under a basic assumption -- the network architecture is fixed throughout the attack process. However, this assumption does not hold for many recently proposed networks, e.g. 3D sparse convolution network, which contains input-dependent execution to improve computational efficiency. It results in a serious issue of lagged gradient, making the learned attack at the current step ineffective due to the architecture changes afterward. To address this issue, we propose a Leaded Gradient Method (LGM) and show the significant effects of the lagged gradient. More specifically, we re-formulate the gradients to be aware of the potential dynamic changes of network architectures, so that the learned attack better "leads" the next step than the dynamics-unaware methods when network architecture changes dynamically. Extensive experiments on various datasets show that our LGM achieves impressive performance on semantic segmentation and classification. Compared with the dynamic-unaware methods, LGM achieves about 20% lower mIoU averagely on the ScanNet and S3DIS datasets. LGM also outperforms the recent point cloud attacks.



## **14. APTSHIELD: A Stable, Efficient and Real-time APT Detection System for Linux Hosts**

cs.CR

**SubmitDate**: 2021-12-17    [paper-pdf](http://arxiv.org/pdf/2112.09008v2)

**Authors**: Tiantian Zhu, Jinkai Yu, Tieming Chen, Jiayu Wang, Jie Ying, Ye Tian, Mingqi Lv, Yan Chen, Yuan Fan, Ting Wang

**Abstracts**: Advanced Persistent Threat (APT) attack usually refers to the form of long-term, covert and sustained attack on specific targets, with an adversary using advanced attack techniques to destroy the key facilities of an organization. APT attacks have caused serious security threats and massive financial loss worldwide. Academics and industry thereby have proposed a series of solutions to detect APT attacks, such as dynamic/static code analysis, traffic detection, sandbox technology, endpoint detection and response (EDR), etc. However, existing defenses are failed to accurately and effectively defend against the current APT attacks that exhibit strong persistent, stealthy, diverse and dynamic characteristics due to the weak data source integrity, large data processing overhead and poor real-time performance in the process of real-world scenarios.   To overcome these difficulties, in this paper we propose APTSHIELD, a stable, efficient and real-time APT detection system for Linux hosts. In the aspect of data collection, audit is selected to stably collect kernel data of the operating system so as to carry out a complete portrait of the attack based on comprehensive analysis and comparison of existing logging tools; In the aspect of data processing, redundant semantics skipping and non-viable node pruning are adopted to reduce the amount of data, so as to reduce the overhead of the detection system; In the aspect of attack detection, an APT attack detection framework based on ATT\&CK model is designed to carry out real-time attack response and alarm through the transfer and aggregation of labels. Experimental results on both laboratory and Darpa Engagement show that our system can effectively detect web vulnerability attacks, file-less attacks and remote access trojan attacks, and has a low false positive rate, which adds far more value than the existing frontier work.



## **15. Deep Bayesian Learning for Car Hacking Detection**

cs.CR

**SubmitDate**: 2021-12-17    [paper-pdf](http://arxiv.org/pdf/2112.09333v1)

**Authors**: Laha Ale, Scott A. King, Ning Zhang

**Abstracts**: With the rise of self-drive cars and connected vehicles, cars are equipped with various devices to assistant the drivers or support self-drive systems. Undoubtedly, cars have become more intelligent as we can deploy more and more devices and software on the cars. Accordingly, the security of assistant and self-drive systems in the cars becomes a life-threatening issue as smart cars can be invaded by malicious attacks that cause traffic accidents. Currently, canonical machine learning and deep learning methods are extensively employed in car hacking detection. However, machine learning and deep learning methods can easily be overconfident and defeated by carefully designed adversarial examples. Moreover, those methods cannot provide explanations for security engineers for further analysis. In this work, we investigated Deep Bayesian Learning models to detect and analyze car hacking behaviors. The Bayesian learning methods can capture the uncertainty of the data and avoid overconfident issues. Moreover, the Bayesian models can provide more information to support the prediction results that can help security engineers further identify the attacks. We have compared our model with deep learning models and the results show the advantages of our proposed model. The code of this work is publicly available



## **16. Generation of Wheel Lockup Attacks on Nonlinear Dynamics of Vehicle Traction**

eess.SY

Submitted to American Control Conference 2022 (ACC 2022), 6 pages

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2112.09229v1)

**Authors**: Alireza Mohammadi, Hafiz Malik, Masoud Abbaszadeh

**Abstracts**: There is ample evidence in the automotive cybersecurity literature that the car brake ECUs can be maliciously reprogrammed. Motivated by such threat, this paper investigates the capabilities of an adversary who can directly control the frictional brake actuators and would like to induce wheel lockup conditions leading to catastrophic road injuries. This paper demonstrates that the adversary despite having a limited knowledge of the tire-road interaction characteristics has the capability of driving the states of the vehicle traction dynamics to a vicinity of the lockup manifold in a finite time by means of a properly designed attack policy for the frictional brakes. This attack policy relies on employing a predefined-time controller and a nonlinear disturbance observer acting on the wheel slip error dynamics. Simulations under various road conditions demonstrate the effectiveness of the proposed attack policy.



## **17. All You Need is RAW: Defending Against Adversarial Attacks with Camera Image Pipelines**

cs.CV

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2112.09219v1)

**Authors**: Yuxuan Zhang, Bo Dong, Felix Heide

**Abstracts**: Existing neural networks for computer vision tasks are vulnerable to adversarial attacks: adding imperceptible perturbations to the input images can fool these methods to make a false prediction on an image that was correctly predicted without the perturbation. Various defense methods have proposed image-to-image mapping methods, either including these perturbations in the training process or removing them in a preprocessing denoising step. In doing so, existing methods often ignore that the natural RGB images in today's datasets are not captured but, in fact, recovered from RAW color filter array captures that are subject to various degradations in the capture. In this work, we exploit this RAW data distribution as an empirical prior for adversarial defense. Specifically, we proposed a model-agnostic adversarial defensive method, which maps the input RGB images to Bayer RAW space and back to output RGB using a learned camera image signal processing (ISP) pipeline to eliminate potential adversarial patterns. The proposed method acts as an off-the-shelf preprocessing module and, unlike model-specific adversarial training methods, does not require adversarial images to train. As a result, the method generalizes to unseen tasks without additional retraining. Experiments on large-scale datasets (e.g., ImageNet, COCO) for different vision tasks (e.g., classification, semantic segmentation, object detection) validate that the method significantly outperforms existing methods across task domains.



## **18. Direction-Aggregated Attack for Transferable Adversarial Examples**

cs.LG

ACM JETC JOURNAL Accepted

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2104.09172v2)

**Authors**: Tianjin Huang, Vlado Menkovski, Yulong Pei, YuHao Wang, Mykola Pechenizkiy

**Abstracts**: Deep neural networks are vulnerable to adversarial examples that are crafted by imposing imperceptible changes to the inputs. However, these adversarial examples are most successful in white-box settings where the model and its parameters are available. Finding adversarial examples that are transferable to other models or developed in a black-box setting is significantly more difficult. In this paper, we propose the Direction-Aggregated adversarial attacks that deliver transferable adversarial examples. Our method utilizes aggregated direction during the attack process for avoiding the generated adversarial examples overfitting to the white-box model. Extensive experiments on ImageNet show that our proposed method improves the transferability of adversarial examples significantly and outperforms state-of-the-art attacks, especially against adversarial robust models. The best averaged attack success rates of our proposed method reaches 94.6\% against three adversarial trained models and 94.8\% against five defense methods. It also reveals that current defense approaches do not prevent transferable adversarial attacks.



## **19. TAFIM: Targeted Adversarial Attacks against Facial Image Manipulations**

cs.CV

Paper Video: https://youtu.be/btHCrVMKbzw Project Page:  https://shivangi-aneja.github.io/projects/tafim/

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2112.09151v1)

**Authors**: Shivangi Aneja, Lev Markhasin, Matthias Niessner

**Abstracts**: Face image manipulation methods, despite having many beneficial applications in computer graphics, can also raise concerns by affecting an individual's privacy or spreading disinformation. In this work, we propose a proactive defense to prevent face manipulation from happening in the first place. To this end, we introduce a novel data-driven approach that produces image-specific perturbations which are embedded in the original images. The key idea is that these protected images prevent face manipulation by causing the manipulation model to produce a predefined manipulation target (uniformly colored output image in our case) instead of the actual manipulation. Compared to traditional adversarial attacks that optimize noise patterns for each image individually, our generalized model only needs a single forward pass, thus running orders of magnitude faster and allowing for easy integration in image processing stacks, even on resource-constrained devices like smartphones. In addition, we propose to leverage a differentiable compression approximation, hence making generated perturbations robust to common image compression. We further show that a generated perturbation can simultaneously prevent against multiple manipulation methods.



## **20. Combating Adversaries with Anti-Adversaries**

cs.LG

Accepted to AAAI Conference on Artificial Intelligence (AAAI'22)

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2103.14347v2)

**Authors**: Motasem Alfarra, Juan C. Pérez, Ali Thabet, Adel Bibi, Philip H. S. Torr, Bernard Ghanem

**Abstracts**: Deep neural networks are vulnerable to small input perturbations known as adversarial attacks. Inspired by the fact that these adversaries are constructed by iteratively minimizing the confidence of a network for the true class label, we propose the anti-adversary layer, aimed at countering this effect. In particular, our layer generates an input perturbation in the opposite direction of the adversarial one and feeds the classifier a perturbed version of the input. Our approach is training-free and theoretically supported. We verify the effectiveness of our approach by combining our layer with both nominally and robustly trained models and conduct large-scale experiments from black-box to adaptive attacks on CIFAR10, CIFAR100, and ImageNet. Our layer significantly enhances model robustness while coming at no cost on clean accuracy.



## **21. Anti-Tamper Radio: System-Level Tamper Detection for Computing Systems**

cs.CR

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2112.09014v1)

**Authors**: Paul Staat, Johannes Tobisch, Christian Zenger, Christof Paar

**Abstracts**: A whole range of attacks becomes possible when adversaries gain physical access to computing systems that process or contain sensitive data. Examples include side-channel analysis, bus probing, device cloning, or implanting hardware Trojans. Defending against these kinds of attacks is considered a challenging endeavor, requiring anti-tamper solutions to monitor the physical environment of the system. Current solutions range from simple switches, which detect if a case is opened, to meshes of conducting material that provide more fine-grained detection of integrity violations. However, these solutions suffer from an intricate trade-off between physical security on the one side and reliability, cost, and difficulty to manufacture on the other. In this work, we demonstrate that radio wave propagation in an enclosed system of complex geometry is sensitive against adversarial physical manipulation. We present an anti-tamper radio (ATR) solution as a method for tamper detection, which combines high detection sensitivity and reliability with ease-of-use. ATR constantly monitors the wireless signal propagation behavior within the boundaries of a metal case. Tamper attempts such as insertion of foreign objects, will alter the observed radio signal response, subsequently raising an alarm. The ATR principle is applicable in many computing systems that require physical security such as servers, ATMs, and smart meters. As a case study, we use 19" servers and thoroughly investigate capabilities and limits of the ATR. Using a custom-built automated probing station, we simulate probing attacks by inserting needles with high precision into protected environments. Our experimental results show that our ATR implementation can detect 16 mm insertions of needles of diameter as low as 0.1 mm under ideal conditions. In the more realistic environment of a running 19" server, we demonstrate reliable [...]



## **22. A Heterogeneous Graph Learning Model for Cyber-Attack Detection**

cs.CR

12pages,7figures,40 references

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2112.08986v1)

**Authors**: Mingqi Lv, Chengyu Dong, Tieming Chen, Tiantian Zhu, Qijie Song, Yuan Fan

**Abstracts**: A cyber-attack is a malicious attempt by experienced hackers to breach the target information system. Usually, the cyber-attacks are characterized as hybrid TTPs (Tactics, Techniques, and Procedures) and long-term adversarial behaviors, making the traditional intrusion detection methods ineffective. Most existing cyber-attack detection systems are implemented based on manually designed rules by referring to domain knowledge (e.g., threat models, threat intelligences). However, this process is lack of intelligence and generalization ability. Aiming at this limitation, this paper proposes an intelligent cyber-attack detection method based on provenance data. To effective and efficient detect cyber-attacks from a huge number of system events in the provenance data, we firstly model the provenance data by a heterogeneous graph to capture the rich context information of each system entities (e.g., process, file, socket, etc.), and learns a semantic vector representation for each system entity. Then, we perform online cyber-attack detection by sampling a small and compact local graph from the heterogeneous graph, and classifying the key system entities as malicious or benign. We conducted a series of experiments on two provenance datasets with real cyber-attacks. The experiment results show that the proposed method outperforms other learning based detection models, and has competitive performance against state-of-the-art rule based cyber-attack detection systems.



## **23. Finding Optimal Tangent Points for Reducing Distortions of Hard-label Attacks**

cs.CV

Accepted at NeurIPS 2021, including the appendix. In the previous  versions (v1 and v2), the experimental results of Table 10 are incorrect and  have been corrected in the current version

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2111.07492v3)

**Authors**: Chen Ma, Xiangyu Guo, Li Chen, Jun-Hai Yong, Yisen Wang

**Abstracts**: One major problem in black-box adversarial attacks is the high query complexity in the hard-label attack setting, where only the top-1 predicted label is available. In this paper, we propose a novel geometric-based approach called Tangent Attack (TA), which identifies an optimal tangent point of a virtual hemisphere located on the decision boundary to reduce the distortion of the attack. Assuming the decision boundary is locally flat, we theoretically prove that the minimum $\ell_2$ distortion can be obtained by reaching the decision boundary along the tangent line passing through such tangent point in each iteration. To improve the robustness of our method, we further propose a generalized method which replaces the hemisphere with a semi-ellipsoid to adapt to curved decision boundaries. Our approach is free of hyperparameters and pre-training. Extensive experiments conducted on the ImageNet and CIFAR-10 datasets demonstrate that our approach can consume only a small number of queries to achieve the low-magnitude distortion. The implementation source code is released online at https://github.com/machanic/TangentAttack.



## **24. Addressing Adversarial Machine Learning Attacks in Smart Healthcare Perspectives**

cs.DC

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2112.08862v1)

**Authors**: Arawinkumaar Selvakkumar, Shantanu Pal, Zahra Jadidi

**Abstracts**: Smart healthcare systems are gaining popularity with the rapid development of intelligent sensors, the Internet of Things (IoT) applications and services, and wireless communications. However, at the same time, several vulnerabilities and adversarial attacks make it challenging for a safe and secure smart healthcare system from a security point of view. Machine learning has been used widely to develop suitable models to predict and mitigate attacks. Still, the attacks could trick the machine learning models and misclassify outputs generated by the model. As a result, it leads to incorrect decisions, for example, false disease detection and wrong treatment plans for patients. In this paper, we address the type of adversarial attacks and their impact on smart healthcare systems. We propose a model to examine how adversarial attacks impact machine learning classifiers. To test the model, we use a medical image dataset. Our model can classify medical images with high accuracy. We then attacked the model with a Fast Gradient Sign Method attack (FGSM) to cause the model to predict the images and misclassify them inaccurately. Using transfer learning, we train a VGG-19 model with the medical dataset and later implement the FGSM to the Convolutional Neural Network (CNN) to examine the significant impact it causes on the performance and accuracy of the machine learning model. Our results demonstrate that the adversarial attack misclassifies the images, causing the model's accuracy rate to drop from 88% to 11%.



## **25. Towards Robust Neural Image Compression: Adversarial Attack and Model Finetuning**

cs.CV

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2112.08691v1)

**Authors**: Tong Chen, Zhan Ma

**Abstracts**: Deep neural network based image compression has been extensively studied. Model robustness is largely overlooked, though it is crucial to service enabling. We perform the adversarial attack by injecting a small amount of noise perturbation to original source images, and then encode these adversarial examples using prevailing learnt image compression models. Experiments report severe distortion in the reconstruction of adversarial examples, revealing the general vulnerability of existing methods, regardless of the settings used in underlying compression model (e.g., network architecture, loss function, quality scale) and optimization strategy used for injecting perturbation (e.g., noise threshold, signal distance measurement). Later, we apply the iterative adversarial finetuning to refine pretrained models. In each iteration, random source images and adversarial examples are mixed to update underlying model. Results show the effectiveness of the proposed finetuning strategy by substantially improving the compression model robustness. Overall, our methodology is simple, effective, and generalizable, making it attractive for developing robust learnt image compression solution. All materials have been made publicly accessible at https://njuvision.github.io/RobustNIC for reproducible research.



## **26. Model Stealing Attacks Against Inductive Graph Neural Networks**

cs.CR

To Appear in the 43rd IEEE Symposium on Security and Privacy, May  22-26, 2022

**SubmitDate**: 2021-12-15    [paper-pdf](http://arxiv.org/pdf/2112.08331v1)

**Authors**: Yun Shen, Xinlei He, Yufei Han, Yang Zhang

**Abstracts**: Many real-world data come in the form of graphs. Graph neural networks (GNNs), a new family of machine learning (ML) models, have been proposed to fully leverage graph data to build powerful applications. In particular, the inductive GNNs, which can generalize to unseen data, become mainstream in this direction. Machine learning models have shown great potential in various tasks and have been deployed in many real-world scenarios. To train a good model, a large amount of data as well as computational resources are needed, leading to valuable intellectual property. Previous research has shown that ML models are prone to model stealing attacks, which aim to steal the functionality of the target models. However, most of them focus on the models trained with images and texts. On the other hand, little attention has been paid to models trained with graph data, i.e., GNNs. In this paper, we fill the gap by proposing the first model stealing attacks against inductive GNNs. We systematically define the threat model and propose six attacks based on the adversary's background knowledge and the responses of the target models. Our evaluation on six benchmark datasets shows that the proposed model stealing attacks against GNNs achieve promising performance.



## **27. Meta Adversarial Perturbations**

cs.LG

Published in AAAI 2022 Workshop

**SubmitDate**: 2021-12-15    [paper-pdf](http://arxiv.org/pdf/2111.10291v2)

**Authors**: Chia-Hung Yuan, Pin-Yu Chen, Chia-Mu Yu

**Abstracts**: A plethora of attack methods have been proposed to generate adversarial examples, among which the iterative methods have been demonstrated the ability to find a strong attack. However, the computation of an adversarial perturbation for a new data point requires solving a time-consuming optimization problem from scratch. To generate a stronger attack, it normally requires updating a data point with more iterations. In this paper, we show the existence of a meta adversarial perturbation (MAP), a better initialization that causes natural images to be misclassified with high probability after being updated through only a one-step gradient ascent update, and propose an algorithm for computing such perturbations. We conduct extensive experiments, and the empirical results demonstrate that state-of-the-art deep neural networks are vulnerable to meta perturbations. We further show that these perturbations are not only image-agnostic, but also model-agnostic, as a single perturbation generalizes well across unseen data points and different neural network architectures.



## **28. Temporal Shuffling for Defending Deep Action Recognition Models against Adversarial Attacks**

cs.CV

**SubmitDate**: 2021-12-15    [paper-pdf](http://arxiv.org/pdf/2112.07921v1)

**Authors**: Jaehui Hwang, Huan Zhang, Jun-Ho Choi, Cho-Jui Hsieh, Jong-Seok Lee

**Abstracts**: Recently, video-based action recognition methods using convolutional neural networks (CNNs) achieve remarkable recognition performance. However, there is still lack of understanding about the generalization mechanism of action recognition models. In this paper, we suggest that action recognition models rely on the motion information less than expected, and thus they are robust to randomization of frame orders. Based on this observation, we develop a novel defense method using temporal shuffling of input videos against adversarial attacks for action recognition models. Another observation enabling our defense method is that adversarial perturbations on videos are sensitive to temporal destruction. To the best of our knowledge, this is the first attempt to design a defense method specific to video-based action recognition models.



## **29. Adversarial Examples for Extreme Multilabel Text Classification**

cs.LG

**SubmitDate**: 2021-12-14    [paper-pdf](http://arxiv.org/pdf/2112.07512v1)

**Authors**: Mohammadreza Qaraei, Rohit Babbar

**Abstracts**: Extreme Multilabel Text Classification (XMTC) is a text classification problem in which, (i) the output space is extremely large, (ii) each data point may have multiple positive labels, and (iii) the data follows a strongly imbalanced distribution. With applications in recommendation systems and automatic tagging of web-scale documents, the research on XMTC has been focused on improving prediction accuracy and dealing with imbalanced data. However, the robustness of deep learning based XMTC models against adversarial examples has been largely underexplored.   In this paper, we investigate the behaviour of XMTC models under adversarial attacks. To this end, first, we define adversarial attacks in multilabel text classification problems. We categorize attacking multilabel text classifiers as (a) positive-targeted, where the target positive label should fall out of top-k predicted labels, and (b) negative-targeted, where the target negative label should be among the top-k predicted labels. Then, by experiments on APLC-XLNet and AttentionXML, we show that XMTC models are highly vulnerable to positive-targeted attacks but more robust to negative-targeted ones. Furthermore, our experiments show that the success rate of positive-targeted adversarial attacks has an imbalanced distribution. More precisely, tail classes are highly vulnerable to adversarial attacks for which an attacker can generate adversarial samples with high similarity to the actual data-points. To overcome this problem, we explore the effect of rebalanced loss functions in XMTC where not only do they increase accuracy on tail classes, but they also improve the robustness of these classes against adversarial attacks. The code for our experiments is available at https://github.com/xmc-aalto/adv-xmtc



## **30. Multi-Leader Congestion Games with an Adversary**

cs.GT

**SubmitDate**: 2021-12-14    [paper-pdf](http://arxiv.org/pdf/2112.07435v1)

**Authors**: Tobias Harks, Mona Henle, Max Klimm, Jannik Matuschke, Anja Schedel

**Abstracts**: We study a multi-leader single-follower congestion game where multiple users (leaders) choose one resource out of a set of resources and, after observing the realized loads, an adversary (single-follower) attacks the resources with maximum loads, causing additional costs for the leaders. For the resulting strategic game among the leaders, we show that pure Nash equilibria may fail to exist and therefore, we consider approximate equilibria instead. As our first main result, we show that the existence of a $K$-approximate equilibrium can always be guaranteed, where $K \approx 1.1974$ is the unique solution of a cubic polynomial equation. To this end, we give a polynomial time combinatorial algorithm which computes a $K$-approximate equilibrium. The factor $K$ is tight, meaning that there is an instance that does not admit an $\alpha$-approximate equilibrium for any $\alpha<K$. Thus $\alpha=K$ is the smallest possible value of $\alpha$ such that the existence of an $\alpha$-approximate equilibrium can be guaranteed for any instance of the considered game. Secondly, we focus on approximate equilibria of a given fixed instance. We show how to compute efficiently a best approximate equilibrium, that is, with smallest possible $\alpha$ among all $\alpha$-approximate equilibria of the given instance.



## **31. Robustifying automatic speech recognition by extracting slowly varying features**

eess.AS

**SubmitDate**: 2021-12-14    [paper-pdf](http://arxiv.org/pdf/2112.07400v1)

**Authors**: Matias Pizarro, Dorothea Kolossa, Asja Fischer

**Abstracts**: In the past few years, it has been shown that deep learning systems are highly vulnerable under attacks with adversarial examples. Neural-network-based automatic speech recognition (ASR) systems are no exception. Targeted and untargeted attacks can modify an audio input signal in such a way that humans still recognise the same words, while ASR systems are steered to predict a different transcription. In this paper, we propose a defense mechanism against targeted adversarial attacks consisting in removing fast-changing features from the audio signals, either by applying slow feature analysis, a low-pass filter, or both, before feeding the input to the ASR system. We perform an empirical analysis of hybrid ASR models trained on data pre-processed in such a way. While the resulting models perform quite well on benign data, they are significantly more robust against targeted adversarial attacks: Our final, proposed model shows a performance on clean data similar to the baseline model, while being more than four times more robust.



## **32. On the Impact of Hard Adversarial Instances on Overfitting in Adversarial Training**

cs.LG

**SubmitDate**: 2021-12-14    [paper-pdf](http://arxiv.org/pdf/2112.07324v1)

**Authors**: Chen Liu, Zhichao Huang, Mathieu Salzmann, Tong Zhang, Sabine Süsstrunk

**Abstracts**: Adversarial training is a popular method to robustify models against adversarial attacks. However, it exhibits much more severe overfitting than training on clean inputs. In this work, we investigate this phenomenon from the perspective of training instances, i.e., training input-target pairs. Based on a quantitative metric measuring instances' difficulty, we analyze the model's behavior on training instances of different difficulty levels. This lets us show that the decay in generalization performance of adversarial training is a result of the model's attempt to fit hard adversarial instances. We theoretically verify our observations for both linear and general nonlinear models, proving that models trained on hard instances have worse generalization performance than ones trained on easy instances. Furthermore, we prove that the difference in the generalization gap between models trained by instances of different difficulty levels increases with the size of the adversarial budget. Finally, we conduct case studies on methods mitigating adversarial overfitting in several scenarios. Our analysis shows that methods successfully mitigating adversarial overfitting all avoid fitting hard adversarial instances, while ones fitting hard adversarial instances do not achieve true robustness.



## **33. Improving Calibration through the Relationship with Adversarial Robustness**

cs.LG

Published at NeurIPS-2021

**SubmitDate**: 2021-12-14    [paper-pdf](http://arxiv.org/pdf/2006.16375v2)

**Authors**: Yao Qin, Xuezhi Wang, Alex Beutel, Ed H. Chi

**Abstracts**: Neural networks lack adversarial robustness, i.e., they are vulnerable to adversarial examples that through small perturbations to inputs cause incorrect predictions. Further, trust is undermined when models give miscalibrated predictions, i.e., the predicted probability is not a good indicator of how much we should trust our model. In this paper, we study the connection between adversarial robustness and calibration and find that the inputs for which the model is sensitive to small perturbations (are easily attacked) are more likely to have poorly calibrated predictions. Based on this insight, we examine if calibration can be improved by addressing those adversarially unrobust inputs. To this end, we propose Adversarial Robustness based Adaptive Label Smoothing (AR-AdaLS) that integrates the correlations of adversarial robustness and calibration into training by adaptively softening labels for an example based on how easily it can be attacked by an adversary. We find that our method, taking the adversarial robustness of the in-distribution data into consideration, leads to better calibration over the model even under distributional shifts. In addition, AR-AdaLS can also be applied to an ensemble model to further improve model calibration.



## **34. Defending Against Multiple and Unforeseen Adversarial Videos**

cs.LG

Accepted in IEEE Transactions on Image Processing (TIP)

**SubmitDate**: 2021-12-14    [paper-pdf](http://arxiv.org/pdf/2009.05244v3)

**Authors**: Shao-Yuan Lo, Vishal M. Patel

**Abstracts**: Adversarial robustness of deep neural networks has been actively investigated. However, most existing defense approaches are limited to a specific type of adversarial perturbations. Specifically, they often fail to offer resistance to multiple attack types simultaneously, i.e., they lack multi-perturbation robustness. Furthermore, compared to image recognition problems, the adversarial robustness of video recognition models is relatively unexplored. While several studies have proposed how to generate adversarial videos, only a handful of approaches about defense strategies have been published in the literature. In this paper, we propose one of the first defense strategies against multiple types of adversarial videos for video recognition. The proposed method, referred to as MultiBN, performs adversarial training on multiple adversarial video types using multiple independent batch normalization (BN) layers with a learning-based BN selection module. With a multiple BN structure, each BN brach is responsible for learning the distribution of a single perturbation type and thus provides more precise distribution estimations. This mechanism benefits dealing with multiple perturbation types. The BN selection module detects the attack type of an input video and sends it to the corresponding BN branch, making MultiBN fully automatic and allowing end-to-end training. Compared to present adversarial training approaches, the proposed MultiBN exhibits stronger multi-perturbation robustness against different and even unforeseen adversarial video types, ranging from Lp-bounded attacks and physically realizable attacks. This holds true on different datasets and target models. Moreover, we conduct an extensive analysis to study the properties of the multiple BN structure.



## **35. MuxLink: Circumventing Learning-Resilient MUX-Locking Using Graph Neural Network-based Link Prediction**

cs.CR

Will be published in Proc. Design, Automation and Test in Europe  (DATE) 2022

**SubmitDate**: 2021-12-14    [paper-pdf](http://arxiv.org/pdf/2112.07178v1)

**Authors**: Lilas Alrahis, Satwik Patnaik, Muhammad Shafique, Ozgur Sinanoglu

**Abstracts**: Logic locking has received considerable interest as a prominent technique for protecting the design intellectual property from untrusted entities, especially the foundry. Recently, machine learning (ML)-based attacks have questioned the security guarantees of logic locking, and have demonstrated considerable success in deciphering the secret key without relying on an oracle, hence, proving to be very useful for an adversary in the fab. Such ML-based attacks have triggered the development of learning-resilient locking techniques. The most advanced state-of-the-art deceptive MUX-based locking (D-MUX) and the symmetric MUX-based locking techniques have recently demonstrated resilience against existing ML-based attacks. Both defense techniques obfuscate the design by inserting key-controlled MUX logic, ensuring that all the secret inputs to the MUXes are equiprobable.   In this work, we show that these techniques primarily introduce local and limited changes to the circuit without altering the global structure of the design. By leveraging this observation, we propose a novel graph neural network (GNN)-based link prediction attack, MuxLink, that successfully breaks both the D-MUX and symmetric MUX-locking techniques, relying only on the underlying structure of the locked design, i.e., in an oracle-less setting. Our trained GNN model learns the structure of the given circuit and the composition of gates around the non-obfuscated wires, thereby generating meaningful link embeddings that help decipher the secret inputs to the MUXes. The proposed MuxLink achieves key prediction accuracy and precision up to 100% on D-MUX and symmetric MUX-locked ISCAS-85 and ITC-99 benchmarks, fully unlocking the designs. We open-source MuxLink [1].



## **36. CMUA-Watermark: A Cross-Model Universal Adversarial Watermark for Combating Deepfakes**

cs.CV

9 pages, 7 figures, Thirty-Sixth AAAI Conference on Artificial  Intelligence, AAAI22

**SubmitDate**: 2021-12-14    [paper-pdf](http://arxiv.org/pdf/2105.10872v2)

**Authors**: Hao Huang, Yongtao Wang, Zhaoyu Chen, Yuze Zhang, Yuheng Li, Zhi Tang, Wei Chu, Jingdong Chen, Weisi Lin, Kai-Kuang Ma

**Abstracts**: Malicious applications of deepfakes (i.e., technologies generating target facial attributes or entire faces from facial images) have posed a huge threat to individuals' reputation and security. To mitigate these threats, recent studies have proposed adversarial watermarks to combat deepfake models, leading them to generate distorted outputs. Despite achieving impressive results, these adversarial watermarks have low image-level and model-level transferability, meaning that they can protect only one facial image from one specific deepfake model. To address these issues, we propose a novel solution that can generate a Cross-Model Universal Adversarial Watermark (CMUA-Watermark), protecting a large number of facial images from multiple deepfake models. Specifically, we begin by proposing a cross-model universal attack pipeline that attacks multiple deepfake models iteratively. Then, we design a two-level perturbation fusion strategy to alleviate the conflict between the adversarial watermarks generated by different facial images and models. Moreover, we address the key problem in cross-model optimization with a heuristic approach to automatically find the suitable attack step sizes for different models, further weakening the model-level conflict. Finally, we introduce a more reasonable and comprehensive evaluation method to fully test the proposed method and compare it with existing ones. Extensive experimental results demonstrate that the proposed CMUA-Watermark can effectively distort the fake facial images generated by multiple deepfake models while achieving a better performance than existing methods.



## **37. Real-Time Neural Voice Camouflage**

cs.SD

14 pages

**SubmitDate**: 2021-12-14    [paper-pdf](http://arxiv.org/pdf/2112.07076v1)

**Authors**: Mia Chiquier, Chengzhi Mao, Carl Vondrick

**Abstracts**: Automatic speech recognition systems have created exciting possibilities for applications, however they also enable opportunities for systematic eavesdropping. We propose a method to camouflage a person's voice over-the-air from these systems without inconveniencing the conversation between people in the room. Standard adversarial attacks are not effective in real-time streaming situations because the characteristics of the signal will have changed by the time the attack is executed. We introduce predictive attacks, which achieve real-time performance by forecasting the attack that will be the most effective in the future. Under real-time constraints, our method jams the established speech recognition system DeepSpeech 4.17x more than baselines as measured through word error rate, and 7.27x more as measured through character error rate. We furthermore demonstrate our approach is practically effective in realistic environments over physical distances.



## **38. On the Privacy Risks of Deploying Recurrent Neural Networks in Machine Learning**

cs.CR

Under Double-Blind Review

**SubmitDate**: 2021-12-13    [paper-pdf](http://arxiv.org/pdf/2110.03054v2)

**Authors**: Yunhao Yang, Parham Gohari, Ufuk Topcu

**Abstracts**: We study the privacy implications of deploying recurrent neural networks (RNNs) in machine learning models. We focus on a class of privacy threats, called membership inference attacks (MIAs), which aim to infer whether or not specific data records have been used to train a model. Considering three machine learning applications, namely, machine translation, deep reinforcement learning, and image classification, we provide empirical evidence that RNNs are more vulnerable to MIAs than the alternative feed-forward architectures. We then study differential privacy methods to protect the privacy of the training dataset of RNNs. These methods are known to provide rigorous privacy guarantees irrespective of the adversary's model. We develop an alternative differential privacy mechanism to the so-called DP-FedAvg algorithm, which instead of obfuscating gradients during training, obfuscates the model's output. Unlike the existing work, the mechanism allows for post-training adjustment of the privacy parameters without having to retrain the model. We provide numerical results suggesting that the mechanism provides a strong shield against MIAs while trading off marginal utility.



## **39. Signal Injection Attacks against CCD Image Sensors**

cs.CR

**SubmitDate**: 2021-12-13    [paper-pdf](http://arxiv.org/pdf/2108.08881v2)

**Authors**: Sebastian Köhler, Richard Baker, Ivan Martinovic

**Abstracts**: Since cameras have become a crucial part in many safety-critical systems and applications, such as autonomous vehicles and surveillance, a large body of academic and non-academic work has shown attacks against their main component - the image sensor. However, these attacks are limited to coarse-grained and often suspicious injections because light is used as an attack vector. Furthermore, due to the nature of optical attacks, they require the line-of-sight between the adversary and the target camera.   In this paper, we present a novel post-transducer signal injection attack against CCD image sensors, as they are used in professional, scientific, and even military settings. We show how electromagnetic emanation can be used to manipulate the image information captured by a CCD image sensor with the granularity down to the brightness of individual pixels. We study the feasibility of our attack and then demonstrate its effects in the scenario of automatic barcode scanning. Our results indicate that the injected distortion can disrupt automated vision-based intelligent systems.



## **40. Better Safe Than Sorry: Preventing Delusive Adversaries with Adversarial Training**

cs.LG

NeurIPS 2021

**SubmitDate**: 2021-12-13    [paper-pdf](http://arxiv.org/pdf/2102.04716v4)

**Authors**: Lue Tao, Lei Feng, Jinfeng Yi, Sheng-Jun Huang, Songcan Chen

**Abstracts**: Delusive attacks aim to substantially deteriorate the test accuracy of the learning model by slightly perturbing the features of correctly labeled training examples. By formalizing this malicious attack as finding the worst-case training data within a specific $\infty$-Wasserstein ball, we show that minimizing adversarial risk on the perturbed data is equivalent to optimizing an upper bound of natural risk on the original data. This implies that adversarial training can serve as a principled defense against delusive attacks. Thus, the test accuracy decreased by delusive attacks can be largely recovered by adversarial training. To further understand the internal mechanism of the defense, we disclose that adversarial training can resist the delusive perturbations by preventing the learner from overly relying on non-robust features in a natural setting. Finally, we complement our theoretical findings with a set of experiments on popular benchmark datasets, which show that the defense withstands six different practical attacks. Both theoretical and empirical results vote for adversarial training when confronted with delusive adversaries.



## **41. A Separation Result Between Data-oblivious and Data-aware Poisoning Attacks**

cs.LG

**SubmitDate**: 2021-12-13    [paper-pdf](http://arxiv.org/pdf/2003.12020v3)

**Authors**: Samuel Deng, Sanjam Garg, Somesh Jha, Saeed Mahloujifar, Mohammad Mahmoody, Abhradeep Thakurta

**Abstracts**: Poisoning attacks have emerged as a significant security threat to machine learning algorithms. It has been demonstrated that adversaries who make small changes to the training set, such as adding specially crafted data points, can hurt the performance of the output model. Some of the stronger poisoning attacks require the full knowledge of the training data. This leaves open the possibility of achieving the same attack results using poisoning attacks that do not have the full knowledge of the clean training set.   In this work, we initiate a theoretical study of the problem above. Specifically, for the case of feature selection with LASSO, we show that full-information adversaries (that craft poisoning examples based on the rest of the training data) are provably stronger than the optimal attacker that is oblivious to the training set yet has access to the distribution of the data. Our separation result shows that the two setting of data-aware and data-oblivious are fundamentally different and we cannot hope to always achieve the same attack or defense results in these scenarios.



## **42. Learning Classical Readout Quantum PUFs based on single-qubit gates**

quant-ph

11 pages, 9 figures

**SubmitDate**: 2021-12-13    [paper-pdf](http://arxiv.org/pdf/2112.06661v1)

**Authors**: Anna Pappa, Niklas Pirnay, Jean-Pierre Seifert

**Abstracts**: Physical Unclonable Functions (PUFs) have been proposed as a way to identify and authenticate electronic devices. Recently, several ideas have been presented that aim to achieve the same for quantum devices. Some of these constructions apply single-qubit gates in order to provide a secure fingerprint of the quantum device. In this work, we formalize the class of Classical Readout Quantum PUFs (CR-QPUFs) using the statistical query (SQ) model and explicitly show insufficient security for CR-QPUFs based on single qubit rotation gates, when the adversary has SQ access to the CR-QPUF. We demonstrate how a malicious party can learn the CR-QPUF characteristics and forge the signature of a quantum device through a modelling attack using a simple regression of low-degree polynomials. The proposed modelling attack was successfully implemented in a real-world scenario on real IBM Q quantum machines. We thoroughly discuss the prospects and problems of CR-QPUFs where quantum device imperfections are used as a secure fingerprint.



## **43. Triangle Attack: A Query-efficient Decision-based Adversarial Attack**

cs.CV

10 pages

**SubmitDate**: 2021-12-13    [paper-pdf](http://arxiv.org/pdf/2112.06569v1)

**Authors**: Xiaosen Wang, Zeliang Zhang, Kangheng Tong, Dihong Gong, Kun He, Zhifeng Li, Wei Liu

**Abstracts**: Decision-based attack poses a severe threat to real-world applications since it regards the target model as a black box and only accesses the hard prediction label. Great efforts have been made recently to decrease the number of queries; however, existing decision-based attacks still require thousands of queries in order to generate good quality adversarial examples. In this work, we find that a benign sample, the current and the next adversarial examples could naturally construct a triangle in a subspace for any iterative attacks. Based on the law of sines, we propose a novel Triangle Attack (TA) to optimize the perturbation by utilizing the geometric information that the longer side is always opposite the larger angle in any triangle. However, directly applying such information on the input image is ineffective because it cannot thoroughly explore the neighborhood of the input sample in the high dimensional space. To address this issue, TA optimizes the perturbation in the low frequency space for effective dimensionality reduction owing to the generality of such geometric property. Extensive evaluations on the ImageNet dataset demonstrate that TA achieves a much higher attack success rate within 1,000 queries and needs a much less number of queries to achieve the same attack success rate under various perturbation budgets than existing decision-based attacks. With such high efficiency, we further demonstrate the applicability of TA on real-world API, i.e., Tencent Cloud API.



## **44. Back to the Drawing Board: A Critical Evaluation of Poisoning Attacks on Production Federated Learning**

cs.LG

To appear in the IEEE Symposium on Security & Privacy (Oakland), 2022

**SubmitDate**: 2021-12-13    [paper-pdf](http://arxiv.org/pdf/2108.10241v2)

**Authors**: Virat Shejwalkar, Amir Houmansadr, Peter Kairouz, Daniel Ramage

**Abstracts**: While recent works have indicated that federated learning (FL) may be vulnerable to poisoning attacks by compromised clients, their real impact on production FL systems is not fully understood. In this work, we aim to develop a comprehensive systemization for poisoning attacks on FL by enumerating all possible threat models, variations of poisoning, and adversary capabilities. We specifically put our focus on untargeted poisoning attacks, as we argue that they are significantly relevant to production FL deployments.   We present a critical analysis of untargeted poisoning attacks under practical, production FL environments by carefully characterizing the set of realistic threat models and adversarial capabilities. Our findings are rather surprising: contrary to the established belief, we show that FL is highly robust in practice even when using simple, low-cost defenses. We go even further and propose novel, state-of-the-art data and model poisoning attacks, and show via an extensive set of experiments across three benchmark datasets how (in)effective poisoning attacks are in the presence of simple defense mechanisms. We aim to correct previous misconceptions and offer concrete guidelines to conduct more accurate (and more realistic) research on this topic.



## **45. Detecting Audio Adversarial Examples with Logit Noising**

cs.CR

10 pages, 12 figures, In Proceedings of the 37th Annual Computer  Security Applications Conference (ACSAC) 2021

**SubmitDate**: 2021-12-13    [paper-pdf](http://arxiv.org/pdf/2112.06443v1)

**Authors**: Namgyu Park, Sangwoo Ji, Jong Kim

**Abstracts**: Automatic speech recognition (ASR) systems are vulnerable to audio adversarial examples that attempt to deceive ASR systems by adding perturbations to benign speech signals. Although an adversarial example and the original benign wave are indistinguishable to humans, the former is transcribed as a malicious target sentence by ASR systems. Several methods have been proposed to generate audio adversarial examples and feed them directly into the ASR system (over-line). Furthermore, many researchers have demonstrated the feasibility of robust physical audio adversarial examples(over-air). To defend against the attacks, several studies have been proposed. However, deploying them in a real-world situation is difficult because of accuracy drop or time overhead. In this paper, we propose a novel method to detect audio adversarial examples by adding noise to the logits before feeding them into the decoder of the ASR. We show that carefully selected noise can significantly impact the transcription results of the audio adversarial examples, whereas it has minimal impact on the transcription results of benign audio waves. Based on this characteristic, we detect audio adversarial examples by comparing the transcription altered by logit noising with its original transcription. The proposed method can be easily applied to ASR systems without any structural changes or additional training. The experimental results show that the proposed method is robust to over-line audio adversarial examples as well as over-air audio adversarial examples compared with state-of-the-art detection methods.



## **46. BACKDOORL: Backdoor Attack against Competitive Reinforcement Learning**

cs.CR

**SubmitDate**: 2021-12-12    [paper-pdf](http://arxiv.org/pdf/2105.00579v3)

**Authors**: Lun Wang, Zaynah Javed, Xian Wu, Wenbo Guo, Xinyu Xing, Dawn Song

**Abstracts**: Recent research has confirmed the feasibility of backdoor attacks in deep reinforcement learning (RL) systems. However, the existing attacks require the ability to arbitrarily modify an agent's observation, constraining the application scope to simple RL systems such as Atari games. In this paper, we migrate backdoor attacks to more complex RL systems involving multiple agents and explore the possibility of triggering the backdoor without directly manipulating the agent's observation. As a proof of concept, we demonstrate that an adversary agent can trigger the backdoor of the victim agent with its own action in two-player competitive RL systems. We prototype and evaluate BACKDOORL in four competitive environments. The results show that when the backdoor is activated, the winning rate of the victim drops by 17% to 37% compared to when not activated.



## **47. Interpolated Joint Space Adversarial Training for Robust and Generalizable Defenses**

cs.CV

Under submission

**SubmitDate**: 2021-12-12    [paper-pdf](http://arxiv.org/pdf/2112.06323v1)

**Authors**: Chun Pong Lau, Jiang Liu, Hossein Souri, Wei-An Lin, Soheil Feizi, Rama Chellappa

**Abstracts**: Adversarial training (AT) is considered to be one of the most reliable defenses against adversarial attacks. However, models trained with AT sacrifice standard accuracy and do not generalize well to novel attacks. Recent works show generalization improvement with adversarial samples under novel threat models such as on-manifold threat model or neural perceptual threat model. However, the former requires exact manifold information while the latter requires algorithm relaxation. Motivated by these considerations, we exploit the underlying manifold information with Normalizing Flow, ensuring that exact manifold assumption holds. Moreover, we propose a novel threat model called Joint Space Threat Model (JSTM), which can serve as a special case of the neural perceptual threat model that does not require additional relaxation to craft the corresponding adversarial attacks. Under JSTM, we develop novel adversarial attacks and defenses. The mixup strategy improves the standard accuracy of neural networks but sacrifices robustness when combined with AT. To tackle this issue, we propose the Robust Mixup strategy in which we maximize the adversity of the interpolated images and gain robustness and prevent overfitting. Our experiments show that Interpolated Joint Space Adversarial Training (IJSAT) achieves good performance in standard accuracy, robustness, and generalization in CIFAR-10/100, OM-ImageNet, and CIFAR-10-C datasets. IJSAT is also flexible and can be used as a data augmentation method to improve standard accuracy and combine with many existing AT approaches to improve robustness.



## **48. A Game-Theoretical Self-Adaptation Framework for Securing Software-Intensive Systems**

cs.SE

**SubmitDate**: 2021-12-12    [paper-pdf](http://arxiv.org/pdf/2112.07588v1)

**Authors**: Mingyue Zhang, Nianyu Li, Sridhar Adepu, Eunsuk Kang, Zhi Jin

**Abstracts**: The increasing prevalence of security attacks on software-intensive systems calls for new, effective methods for detecting and responding to these attacks. As one promising approach, game theory provides analytical tools for modeling the interaction between the system and the adversarial environment and designing reliable defense. In this paper, we propose an approach for securing software-intensive systems using a rigorous game-theoretical framework. First, a self-adaptation framework is deployed on a component-based software intensive system, which periodically monitors the system for anomalous behaviors. A learning-based method is proposed to detect possible on-going attacks on the system components and predict potential threats to components. Then, an algorithm is designed to automatically build a \emph{Bayesian game} based on the system architecture (of which some components might have been compromised) once an attack is detected, in which the system components are modeled as independent players in the game. Finally, an optimal defensive policy is computed by solving the Bayesian game to achieve the best system utility, which amounts to minimizing the impact of the attack. We conduct two sets of experiments on two general benchmark tasks for security domain. Moreover, we systematically present a case study on a real-world water treatment testbed, i.e. the Secure Water Treatment System. Experiment results show the applicability and the effectiveness of our approach.



## **49. FCA: Learning a 3D Full-coverage Vehicle Camouflage for Multi-view Physical Adversarial Attack**

cs.CV

9 pages, 5 figures

**SubmitDate**: 2021-12-12    [paper-pdf](http://arxiv.org/pdf/2109.07193v3)

**Authors**: Donghua Wang, Tingsong Jiang, Jialiang Sun, Weien Zhou, Xiaoya Zhang, Zhiqiang Gong, Wen Yao, Xiaoqian Chen

**Abstracts**: Physical adversarial attacks in object detection have attracted increasing attention. However, most previous works focus on hiding the objects from the detector by generating an individual adversarial patch, which only covers the planar part of the vehicle's surface and fails to attack the detector in physical scenarios for multi-view, long-distance and partially occluded objects. To bridge the gap between digital attacks and physical attacks, we exploit the full 3D vehicle surface to propose a robust Full-coverage Camouflage Attack (FCA) to fool detectors. Specifically, we first try rendering the nonplanar camouflage texture over the full vehicle surface. To mimic the real-world environment conditions, we then introduce a transformation function to transfer the rendered camouflaged vehicle into a photo realistic scenario. Finally, we design an efficient loss function to optimize the camouflage texture. Experiments show that the full-coverage camouflage attack can not only outperform state-of-the-art methods under various test cases but also generalize to different environments, vehicles, and object detectors. The code of FCA will be available at: https://idrl-lab.github.io/Full-coverage-camouflage-adversarial-attack/.



## **50. A Note on the Post-Quantum Security of (Ring) Signatures**

quant-ph

**SubmitDate**: 2021-12-11    [paper-pdf](http://arxiv.org/pdf/2112.06078v1)

**Authors**: Rohit Chatterjee, Kai-Min Chung, Xiao Liang, Giulio Malavolta

**Abstracts**: This work revisits the security of classical signatures and ring signatures in a quantum world. For (ordinary) signatures, we focus on the arguably preferable security notion of blind-unforgeability recently proposed by Alagic et al. (Eurocrypt'20). We present two short signature schemes achieving this notion: one is in the quantum random oracle model, assuming quantum hardness of SIS; and the other is in the plain model, assuming quantum hardness of LWE with super-polynomial modulus. Prior to this work, the only known blind-unforgeable schemes are Lamport's one-time signature and the Winternitz one-time signature, and both of them are in the quantum random oracle model.   For ring signatures, the recent work by Chatterjee et al. (Crypto'21) proposes a definition trying to capture adversaries with quantum access to the signer. However, it is unclear if their definition, when restricted to the classical world, is as strong as the standard security notion for ring signatures. They also present a construction that only partially achieves (even) this seeming weak definition, in the sense that the adversary can only conduct superposition attacks over the messages, but not the rings. We propose a new definition that does not suffer from the above issue. Our definition is an analog to the blind-unforgeability in the ring signature setting. Moreover, assuming the quantum hardness of LWE, we construct a compiler converting any blind-unforgeable (ordinary) signatures to a ring signature satisfying our definition.



