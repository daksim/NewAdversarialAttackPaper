# Latest Adversarial Attack Papers
**update at 2022-07-06 06:31:28**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Wild Networks: Exposure of 5G Network Infrastructures to Adversarial Examples**

cs.CR

**SubmitDate**: 2022-07-04    [paper-pdf](http://arxiv.org/pdf/2207.01531v1)

**Authors**: Giovanni Apruzzese, Rodion Vladimirov, Aliya Tastemirova, Pavel Laskov

**Abstracts**: Fifth Generation (5G) networks must support billions of heterogeneous devices while guaranteeing optimal Quality of Service (QoS). Such requirements are impossible to meet with human effort alone, and Machine Learning (ML) represents a core asset in 5G. ML, however, is known to be vulnerable to adversarial examples; moreover, as our paper will show, the 5G context is exposed to a yet another type of adversarial ML attacks that cannot be formalized with existing threat models. Proactive assessment of such risks is also challenging due to the lack of ML-powered 5G equipment available for adversarial ML research.   To tackle these problems, we propose a novel adversarial ML threat model that is particularly suited to 5G scenarios, and is agnostic to the precise function solved by ML. In contrast to existing ML threat models, our attacks do not require any compromise of the target 5G system while still being viable due to the QoS guarantees and the open nature of 5G networks. Furthermore, we propose an original framework for realistic ML security assessments based on public data. We proactively evaluate our threat model on 6 applications of ML envisioned in 5G. Our attacks affect both the training and the inference stages, can degrade the performance of state-of-the-art ML systems, and have a lower entry barrier than previous attacks.



## **2. Adversarial Ensemble Training by Jointly Learning Label Dependencies and Member Models**

cs.LG

**SubmitDate**: 2022-07-04    [paper-pdf](http://arxiv.org/pdf/2206.14477v2)

**Authors**: Lele Wang, Bin Liu

**Abstracts**: Training an ensemble of different sub-models has empirically proven to be an effective strategy to improve deep neural networks' adversarial robustness. Current ensemble training methods for image recognition usually encode the image labels by one-hot vectors, which neglect dependency relationships between the labels. Here we propose a novel adversarial ensemble training approach to jointly learn the label dependencies and the member models. Our approach adaptively exploits the learned label dependencies to promote the diversity of the member models. We test our approach on widely used datasets MNIST, FasionMNIST, and CIFAR-10. Results show that our approach is more robust against black-box attacks compared with the state-of-the-art methods. Our code is available at https://github.com/ZJLAB-AMMI/LSD.



## **3. Hessian-Free Second-Order Adversarial Examples for Adversarial Learning**

cs.LG

**SubmitDate**: 2022-07-04    [paper-pdf](http://arxiv.org/pdf/2207.01396v1)

**Authors**: Yaguan Qian, Yuqi Wang, Bin Wang, Zhaoquan Gu, Yuhan Guo, Wassim Swaileh

**Abstracts**: Recent studies show deep neural networks (DNNs) are extremely vulnerable to the elaborately designed adversarial examples. Adversarial learning with those adversarial examples has been proved as one of the most effective methods to defend against such an attack. At present, most existing adversarial examples generation methods are based on first-order gradients, which can hardly further improve models' robustness, especially when facing second-order adversarial attacks. Compared with first-order gradients, second-order gradients provide a more accurate approximation of the loss landscape with respect to natural examples. Inspired by this, our work crafts second-order adversarial examples and uses them to train DNNs. Nevertheless, second-order optimization involves time-consuming calculation for Hessian-inverse. We propose an approximation method through transforming the problem into an optimization in the Krylov subspace, which remarkably reduce the computational complexity to speed up the training procedure. Extensive experiments conducted on the MINIST and CIFAR-10 datasets show that our adversarial learning with second-order adversarial examples outperforms other fisrt-order methods, which can improve the model robustness against a wide range of attacks.



## **4. Training strategy for a lightweight countermeasure model for automatic speaker verification**

cs.SD

Need to finish experiment

**SubmitDate**: 2022-07-04    [paper-pdf](http://arxiv.org/pdf/2203.17031v4)

**Authors**: Yen-Lun Liao, Xuanjun Chen, Chung-Che Wang, Jyh-Shing Roger Jang

**Abstracts**: The countermeasure (CM) model is developed to protect Automatic Speaker Verification (ASV) systems from spoof attacks and prevent resulting personal information leakage. Based on practicality and security considerations, the CM model is usually deployed on edge devices, which have more limited computing resources and storage space than cloud-based systems. This work proposes training strategies for a lightweight CM model for ASV, using generalized end-to-end (GE2E) pre-training and adversarial fine-tuning to improve performance, and applying knowledge distillation (KD) to reduce the size of the CM model. In the evaluation phase of the ASVspoof 2021 Logical Access task, the lightweight ResNetSE model reaches min t-DCF 0.2695 and EER 3.54%. Compared to the teacher model, the lightweight student model only uses 22.5% of parameters and 21.1% of multiply and accumulate operands of the teacher model.



## **5. BadHash: Invisible Backdoor Attacks against Deep Hashing with Clean Label**

cs.CV

conference

**SubmitDate**: 2022-07-04    [paper-pdf](http://arxiv.org/pdf/2207.00278v2)

**Authors**: Shengshan Hu, Ziqi Zhou, Yechao Zhang, Leo Yu Zhang, Yifeng Zheng, Yuanyuan HE, Hai Jin

**Abstracts**: Due to its powerful feature learning capability and high efficiency, deep hashing has achieved great success in large-scale image retrieval. Meanwhile, extensive works have demonstrated that deep neural networks (DNNs) are susceptible to adversarial examples, and exploring adversarial attack against deep hashing has attracted many research efforts. Nevertheless, backdoor attack, another famous threat to DNNs, has not been studied for deep hashing yet. Although various backdoor attacks have been proposed in the field of image classification, existing approaches failed to realize a truly imperceptive backdoor attack that enjoys invisible triggers and clean label setting simultaneously, and they also cannot meet the intrinsic demand of image retrieval backdoor.   In this paper, we propose BadHash, the first generative-based imperceptible backdoor attack against deep hashing, which can effectively generate invisible and input-specific poisoned images with clean label. Specifically, we first propose a new conditional generative adversarial network (cGAN) pipeline to effectively generate poisoned samples. For any given benign image, it seeks to generate a natural-looking poisoned counterpart with a unique invisible trigger. In order to improve the attack effectiveness, we introduce a label-based contrastive learning network LabCLN to exploit the semantic characteristics of different labels, which are subsequently used for confusing and misleading the target model to learn the embedded trigger. We finally explore the mechanism of backdoor attacks on image retrieval in the hash space. Extensive experiments on multiple benchmark datasets verify that BadHash can generate imperceptible poisoned samples with strong attack ability and transferability over state-of-the-art deep hashing schemes.



## **6. Removing Batch Normalization Boosts Adversarial Training**

cs.LG

ICML 2022

**SubmitDate**: 2022-07-04    [paper-pdf](http://arxiv.org/pdf/2207.01156v1)

**Authors**: Haotao Wang, Aston Zhang, Shuai Zheng, Xingjian Shi, Mu Li, Zhangyang Wang

**Abstracts**: Adversarial training (AT) defends deep neural networks against adversarial attacks. One challenge that limits its practical application is the performance degradation on clean samples. A major bottleneck identified by previous works is the widely used batch normalization (BN), which struggles to model the different statistics of clean and adversarial training samples in AT. Although the dominant approach is to extend BN to capture this mixture of distribution, we propose to completely eliminate this bottleneck by removing all BN layers in AT. Our normalizer-free robust training (NoFrost) method extends recent advances in normalizer-free networks to AT for its unexplored advantage on handling the mixture distribution challenge. We show that NoFrost achieves adversarial robustness with only a minor sacrifice on clean sample accuracy. On ImageNet with ResNet50, NoFrost achieves $74.06\%$ clean accuracy, which drops merely $2.00\%$ from standard training. In contrast, BN-based AT obtains $59.28\%$ clean accuracy, suffering a significant $16.78\%$ drop from standard training. In addition, NoFrost achieves a $23.56\%$ adversarial robustness against PGD attack, which improves the $13.57\%$ robustness in BN-based AT. We observe better model smoothness and larger decision margins from NoFrost, which make the models less sensitive to input perturbations and thus more robust. Moreover, when incorporating more data augmentations into NoFrost, it achieves comprehensive robustness against multiple distribution shifts. Code and pre-trained models are public at https://github.com/amazon-research/normalizer-free-robust-training.



## **7. Transferable Graph Backdoor Attack**

cs.CR

Accepted by the 25th International Symposium on Research in Attacks,  Intrusions, and Defenses

**SubmitDate**: 2022-07-04    [paper-pdf](http://arxiv.org/pdf/2207.00425v2)

**Authors**: Shuiqiao Yang, Bao Gia Doan, Paul Montague, Olivier De Vel, Tamas Abraham, Seyit Camtepe, Damith C. Ranasinghe, Salil S. Kanhere

**Abstracts**: Graph Neural Networks (GNNs) have achieved tremendous success in many graph mining tasks benefitting from the message passing strategy that fuses the local structure and node features for better graph representation learning. Despite the success of GNNs, and similar to other types of deep neural networks, GNNs are found to be vulnerable to unnoticeable perturbations on both graph structure and node features. Many adversarial attacks have been proposed to disclose the fragility of GNNs under different perturbation strategies to create adversarial examples. However, vulnerability of GNNs to successful backdoor attacks was only shown recently. In this paper, we disclose the TRAP attack, a Transferable GRAPh backdoor attack. The core attack principle is to poison the training dataset with perturbation-based triggers that can lead to an effective and transferable backdoor attack. The perturbation trigger for a graph is generated by performing the perturbation actions on the graph structure via a gradient based score matrix from a surrogate model. Compared with prior works, TRAP attack is different in several ways: i) it exploits a surrogate Graph Convolutional Network (GCN) model to generate perturbation triggers for a blackbox based backdoor attack; ii) it generates sample-specific perturbation triggers which do not have a fixed pattern; and iii) the attack transfers, for the first time in the context of GNNs, to different GNN models when trained with the forged poisoned training dataset. Through extensive evaluations on four real-world datasets, we demonstrate the effectiveness of the TRAP attack to build transferable backdoors in four different popular GNNs using four real-world datasets



## **8. RAF: Recursive Adversarial Attacks on Face Recognition Using Extremely Limited Queries**

cs.CV

**SubmitDate**: 2022-07-04    [paper-pdf](http://arxiv.org/pdf/2207.01149v1)

**Authors**: Keshav Kasichainula, Hadi Mansourifar, Weidong Shi

**Abstracts**: Recent successful adversarial attacks on face recognition show that, despite the remarkable progress of face recognition models, they are still far behind the human intelligence for perception and recognition. It reveals the vulnerability of deep convolutional neural networks (CNNs) as state-of-the-art building block for face recognition models against adversarial examples, which can cause certain consequences for secure systems. Gradient-based adversarial attacks are widely studied before and proved to be successful against face recognition models. However, finding the optimized perturbation per each face needs to submitting the significant number of queries to the target model. In this paper, we propose recursive adversarial attack on face recognition using automatic face warping which needs extremely limited number of queries to fool the target model. Instead of a random face warping procedure, the warping functions are applied on specific detected regions of face like eyebrows, nose, lips, etc. We evaluate the robustness of proposed method in the decision-based black-box attack setting, where the attackers have no access to the model parameters and gradients, but hard-label predictions and confidence scores are provided by the target model.



## **9. Generating gender-ambiguous voices for privacy-preserving speech recognition**

cs.SD

5 pages, 4 figures, submitted to INTERSPEECH

**SubmitDate**: 2022-07-03    [paper-pdf](http://arxiv.org/pdf/2207.01052v1)

**Authors**: Dimitrios Stoidis, Andrea Cavallaro

**Abstracts**: Our voice encodes a uniquely identifiable pattern which can be used to infer private attributes, such as gender or identity, that an individual might wish not to reveal when using a speech recognition service. To prevent attribute inference attacks alongside speech recognition tasks, we present a generative adversarial network, GenGAN, that synthesises voices that conceal the gender or identity of a speaker. The proposed network includes a generator with a U-Net architecture that learns to fool a discriminator. We condition the generator only on gender information and use an adversarial loss between signal distortion and privacy preservation. We show that GenGAN improves the trade-off between privacy and utility compared to privacy-preserving representation learning methods that consider gender information as a sensitive attribute to protect.



## **10. Adversarial Attack and Defense of YOLO Detectors in Autonomous Driving Scenarios**

cs.CV

Accepted by 2022 IEEE Intelligent Vehicles Symposium (IV 2022)

**SubmitDate**: 2022-07-03    [paper-pdf](http://arxiv.org/pdf/2202.04781v2)

**Authors**: Jung Im Choi, Qing Tian

**Abstracts**: Visual detection is a key task in autonomous driving, and it serves as a crucial foundation for self-driving planning and control. Deep neural networks have achieved promising results in various visual tasks, but they are known to be vulnerable to adversarial attacks. A comprehensive understanding of deep visual detectors' vulnerability is required before people can improve their robustness. However, only a few adversarial attack/defense works have focused on object detection, and most of them employed only classification and/or localization losses, ignoring the objectness aspect. In this paper, we identify a serious objectness-related adversarial vulnerability in YOLO detectors and present an effective attack strategy targeting the objectness aspect of visual detection in autonomous vehicles. Furthermore, to address such vulnerability, we propose a new objectness-aware adversarial training approach for visual detection. Experiments show that the proposed attack targeting the objectness aspect is 45.17% and 43.50% more effective than those generated from classification and/or localization losses on the KITTI and COCO traffic datasets, respectively. Also, the proposed adversarial defense approach can improve the detectors' robustness against objectness-oriented attacks by up to 21% and 12% mAP on KITTI and COCO traffic, respectively.



## **11. AGIC: Approximate Gradient Inversion Attack on Federated Learning**

cs.LG

This paper is accepted at the 41st International Symposium on  Reliable Distributed Systems (SRDS 2022)

**SubmitDate**: 2022-07-03    [paper-pdf](http://arxiv.org/pdf/2204.13784v2)

**Authors**: Jin Xu, Chi Hong, Jiyue Huang, Lydia Y. Chen, Jérémie Decouchant

**Abstracts**: Federated learning is a private-by-design distributed learning paradigm where clients train local models on their own data before a central server aggregates their local updates to compute a global model. Depending on the aggregation method used, the local updates are either the gradients or the weights of local learning models. Recent reconstruction attacks apply a gradient inversion optimization on the gradient update of a single minibatch to reconstruct the private data used by clients during training. As the state-of-the-art reconstruction attacks solely focus on single update, realistic adversarial scenarios are overlooked, such as observation across multiple updates and updates trained from multiple mini-batches. A few studies consider a more challenging adversarial scenario where only model updates based on multiple mini-batches are observable, and resort to computationally expensive simulation to untangle the underlying samples for each local step. In this paper, we propose AGIC, a novel Approximate Gradient Inversion Attack that efficiently and effectively reconstructs images from both model or gradient updates, and across multiple epochs. In a nutshell, AGIC (i) approximates gradient updates of used training samples from model updates to avoid costly simulation procedures, (ii) leverages gradient/model updates collected from multiple epochs, and (iii) assigns increasing weights to layers with respect to the neural network structure for reconstruction quality. We extensively evaluate AGIC on three datasets, CIFAR-10, CIFAR-100 and ImageNet. Our results show that AGIC increases the peak signal-to-noise ratio (PSNR) by up to 50% compared to two representative state-of-the-art gradient inversion attacks. Furthermore, AGIC is faster than the state-of-the-art simulation based attack, e.g., it is 5x faster when attacking FedAvg with 8 local steps in between model updates.



## **12. Tricking the Hashing Trick: A Tight Lower Bound on the Robustness of CountSketch to Adaptive Inputs**

cs.DS

**SubmitDate**: 2022-07-03    [paper-pdf](http://arxiv.org/pdf/2207.00956v1)

**Authors**: Edith Cohen, Jelani Nelson, Tamás Sarlós, Uri Stemmer

**Abstracts**: CountSketch and Feature Hashing (the "hashing trick") are popular randomized dimensionality reduction methods that support recovery of $\ell_2$-heavy hitters (keys $i$ where $v_i^2 > \epsilon \|\boldsymbol{v}\|_2^2$) and approximate inner products. When the inputs are {\em not adaptive} (do not depend on prior outputs), classic estimators applied to a sketch of size $O(\ell/\epsilon)$ are accurate for a number of queries that is exponential in $\ell$. When inputs are adaptive, however, an adversarial input can be constructed after $O(\ell)$ queries with the classic estimator and the best known robust estimator only supports $\tilde{O}(\ell^2)$ queries. In this work we show that this quadratic dependence is in a sense inherent: We design an attack that after $O(\ell^2)$ queries produces an adversarial input vector whose sketch is highly biased. Our attack uses "natural" non-adaptive inputs (only the final adversarial input is chosen adaptively) and universally applies with any correct estimator, including one that is unknown to the attacker. In that, we expose inherent vulnerability of this fundamental method.



## **13. When Are Linear Stochastic Bandits Attackable?**

cs.LG

27 pages, 3 figures, ICML 2022

**SubmitDate**: 2022-07-03    [paper-pdf](http://arxiv.org/pdf/2110.09008v2)

**Authors**: Huazheng Wang, Haifeng Xu, Hongning Wang

**Abstracts**: We study adversarial attacks on linear stochastic bandits: by manipulating the rewards, an adversary aims to control the behaviour of the bandit algorithm. Perhaps surprisingly, we first show that some attack goals can never be achieved. This is in sharp contrast to context-free stochastic bandits, and is intrinsically due to the correlation among arms in linear stochastic bandits. Motivated by this finding, this paper studies the attackability of a $k$-armed linear bandit environment. We first provide a complete necessity and sufficiency characterization of attackability based on the geometry of the arms' context vectors. We then propose a two-stage attack method against LinUCB and Robust Phase Elimination. The method first asserts whether the given environment is attackable; and if yes, it poisons the rewards to force the algorithm to pull a target arm linear times using only a sublinear cost. Numerical experiments further validate the effectiveness and cost-efficiency of the proposed attack method.



## **14. Certifiably Robust Policy Learning against Adversarial Communication in Multi-agent Systems**

cs.LG

**SubmitDate**: 2022-07-02    [paper-pdf](http://arxiv.org/pdf/2206.10158v2)

**Authors**: Yanchao Sun, Ruijie Zheng, Parisa Hassanzadeh, Yongyuan Liang, Soheil Feizi, Sumitra Ganesh, Furong Huang

**Abstracts**: Communication is important in many multi-agent reinforcement learning (MARL) problems for agents to share information and make good decisions. However, when deploying trained communicative agents in a real-world application where noise and potential attackers exist, the safety of communication-based policies becomes a severe issue that is underexplored. Specifically, if communication messages are manipulated by malicious attackers, agents relying on untrustworthy communication may take unsafe actions that lead to catastrophic consequences. Therefore, it is crucial to ensure that agents will not be misled by corrupted communication, while still benefiting from benign communication. In this work, we consider an environment with $N$ agents, where the attacker may arbitrarily change the communication from any $C<\frac{N-1}{2}$ agents to a victim agent. For this strong threat model, we propose a certifiable defense by constructing a message-ensemble policy that aggregates multiple randomly ablated message sets. Theoretical analysis shows that this message-ensemble policy can utilize benign communication while being certifiably robust to adversarial communication, regardless of the attacking algorithm. Experiments in multiple environments verify that our defense significantly improves the robustness of trained policies against various types of attacks.



## **15. Backdoor Attack is A Devil in Federated GAN-based Medical Image Synthesis**

cs.CV

13 pages, 4 figures

**SubmitDate**: 2022-07-02    [paper-pdf](http://arxiv.org/pdf/2207.00762v1)

**Authors**: Ruinan Jin, Xiaoxiao Li

**Abstracts**: Deep Learning-based image synthesis techniques have been applied in healthcare research for generating medical images to support open research. Training generative adversarial neural networks (GAN) usually requires large amounts of training data. Federated learning (FL) provides a way of training a central model using distributed data from different medical institutions while keeping raw data locally. However, FL is vulnerable to backdoor attack, an adversarial by poisoning training data, given the central server cannot access the original data directly. Most backdoor attack strategies focus on classification models and centralized domains. In this study, we propose a way of attacking federated GAN (FedGAN) by treating the discriminator with a commonly used data poisoning strategy in backdoor attack classification models. We demonstrate that adding a small trigger with size less than 0.5 percent of the original image size can corrupt the FL-GAN model. Based on the proposed attack, we provide two effective defense strategies: global malicious detection and local training regularization. We show that combining the two defense strategies yields a robust medical image generation.



## **16. Efficient Adversarial Training With Data Pruning**

cs.LG

**SubmitDate**: 2022-07-01    [paper-pdf](http://arxiv.org/pdf/2207.00694v1)

**Authors**: Maximilian Kaufmann, Yiren Zhao, Ilia Shumailov, Robert Mullins, Nicolas Papernot

**Abstracts**: Neural networks are susceptible to adversarial examples-small input perturbations that cause models to fail. Adversarial training is one of the solutions that stops adversarial examples; models are exposed to attacks during training and learn to be resilient to them. Yet, such a procedure is currently expensive-it takes a long time to produce and train models with adversarial samples, and, what is worse, it occasionally fails. In this paper we demonstrate data pruning-a method for increasing adversarial training efficiency through data sub-sampling.We empirically show that data pruning leads to improvements in convergence and reliability of adversarial training, albeit with different levels of utility degradation. For example, we observe that using random sub-sampling of CIFAR10 to drop 40% of data, we lose 8% adversarial accuracy against the strongest attackers, while by using only 20% of data we lose 14% adversarial accuracy and reduce runtime by a factor of 3. Interestingly, we discover that in some settings data pruning brings benefits from both worlds-it both improves adversarial accuracy and training time.



## **17. Watermarking Graph Neural Networks based on Backdoor Attacks**

cs.LG

13 pages, 9 figures

**SubmitDate**: 2022-07-01    [paper-pdf](http://arxiv.org/pdf/2110.11024v3)

**Authors**: Jing Xu, Stjepan Picek

**Abstracts**: Graph Neural Networks (GNNs) have achieved promising performance in various real-world applications. Building a powerful GNN model is not a trivial task, as it requires a large amount of training data, powerful computing resources, and human expertise in fine-tuning the model. What is more, with the development of adversarial attacks, e.g., model stealing attacks, GNNs raise challenges to model authentication. To avoid copyright infringement on GNNs, it is necessary to verify the ownership of the GNN models.   In this paper, we present a watermarking framework for GNNs for both graph and node classification tasks. We 1) design two strategies to generate watermarked data for the graph classification task and one for the node classification task, 2) embed the watermark into the host model through training to obtain the watermarked GNN model, and 3) verify the ownership of the suspicious model in a black-box setting. The experiments show that our framework can verify the ownership of GNN models with a very high probability (around $95\%$) for both tasks. Finally, we experimentally show that our watermarking approach is robust against two model modifications and an input reformation defense against backdoor attacks.



## **18. More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks**

cs.CR

15 pages, 15 figures

**SubmitDate**: 2022-07-01    [paper-pdf](http://arxiv.org/pdf/2202.03195v3)

**Authors**: Jing Xu, Rui Wang, Stefanos Koffas, Kaitai Liang, Stjepan Picek

**Abstracts**: Graph Neural Networks (GNNs) are a class of deep learning-based methods for processing graph domain information. GNNs have recently become a widely used graph analysis method due to their superior ability to learn representations for complex graph data. However, due to privacy concerns and regulation restrictions, centralized GNNs can be difficult to apply to data-sensitive scenarios. Federated learning (FL) is an emerging technology developed for privacy-preserving settings when several parties need to train a shared global model collaboratively. Although several research works have applied FL to train GNNs (Federated GNNs), there is no research on their robustness to backdoor attacks.   This paper bridges this gap by conducting two types of backdoor attacks in Federated GNNs: centralized backdoor attacks (CBA) and distributed backdoor attacks (DBA). Our experiments show that the DBA attack success rate is higher than CBA in almost all evaluated cases. For CBA, the attack success rate of all local triggers is similar to the global trigger even if the training set of the adversarial party is embedded with the global trigger. To further explore the properties of two backdoor attacks in Federated GNNs, we evaluate the attack performance for a different number of clients, trigger sizes, poisoning intensities, and trigger densities. Moreover, we explore the robustness of DBA and CBA against two state-of-the-art defenses. We find that both attacks are robust against the investigated defenses, necessitating the need to consider backdoor attacks in Federated GNNs as a novel threat that requires custom defenses.



## **19. Effect of Homomorphic Encryption on the Performance of Training Federated Learning Generative Adversarial Networks**

cs.CR

**SubmitDate**: 2022-07-01    [paper-pdf](http://arxiv.org/pdf/2207.00263v1)

**Authors**: Ignjat Pejic, Rui Wang, Kaitai Liang

**Abstracts**: A Generative Adversarial Network (GAN) is a deep-learning generative model in the field of Machine Learning (ML) that involves training two Neural Networks (NN) using a sizable data set. In certain fields, such as medicine, the training data may be hospital patient records that are stored across different hospitals. The classic centralized approach would involve sending the data to a centralized server where the model would be trained. However, that would involve breaching the privacy and confidentiality of the patients and their data, which would be unacceptable. Therefore, Federated Learning (FL), an ML technique that trains ML models in a distributed setting without data ever leaving the host device, would be a better alternative to the centralized option. In this ML technique, only parameters and certain metadata would be communicated. In spite of that, there still exist attacks that can infer user data using the parameters and metadata. A fully privacy-preserving solution involves homomorphically encrypting (HE) the data communicated. This paper will focus on the performance loss of training an FL-GAN with three different types of Homomorphic Encryption: Partial Homomorphic Encryption (PHE), Somewhat Homomorphic Encryption (SHE), and Fully Homomorphic Encryption (FHE). We will also test the performance loss of Multi-Party Computations (MPC), as it has homomorphic properties. The performances will be compared to the performance of training an FL-GAN without encryption as well. Our experiments show that the more complex the encryption method is, the longer it takes, with the extra time taken for HE is quite significant in comparison to the base case of FL.



## **20. Threat Assessment in Machine Learning based Systems**

cs.CR

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2207.00091v1)

**Authors**: Lionel Nganyewou Tidjon, Foutse Khomh

**Abstracts**: Machine learning is a field of artificial intelligence (AI) that is becoming essential for several critical systems, making it a good target for threat actors. Threat actors exploit different Tactics, Techniques, and Procedures (TTPs) against the confidentiality, integrity, and availability of Machine Learning (ML) systems. During the ML cycle, they exploit adversarial TTPs to poison data and fool ML-based systems. In recent years, multiple security practices have been proposed for traditional systems but they are not enough to cope with the nature of ML-based systems. In this paper, we conduct an empirical study of threats reported against ML-based systems with the aim to understand and characterize the nature of ML threats and identify common mitigation strategies. The study is based on 89 real-world ML attack scenarios from the MITRE's ATLAS database, the AI Incident Database, and the literature; 854 ML repositories from the GitHub search and the Python Packaging Advisory database, selected based on their reputation. Attacks from the AI Incident Database and the literature are used to identify vulnerabilities and new types of threats that were not documented in ATLAS. Results show that convolutional neural networks were one of the most targeted models among the attack scenarios. ML repositories with the largest vulnerability prominence include TensorFlow, OpenCV, and Notebook. In this paper, we also report the most frequent vulnerabilities in the studied ML repositories, the most targeted ML phases and models, the most used TTPs in ML phases and attack scenarios. This information is particularly important for red/blue teams to better conduct attacks/defenses, for practitioners to prevent threats during ML development, and for researchers to develop efficient defense mechanisms.



## **21. MEAD: A Multi-Armed Approach for Evaluation of Adversarial Examples Detectors**

cs.CV

This paper has been accepted to appear in the Proceedings of the 2022  European Conference on Machine Learning and Data Mining (ECML-PKDD), 19th to  the 23rd of September, Grenoble, France

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2206.15415v1)

**Authors**: Federica Granese, Marine Picot, Marco Romanelli, Francisco Messina, Pablo Piantanida

**Abstracts**: Detection of adversarial examples has been a hot topic in the last years due to its importance for safely deploying machine learning algorithms in critical applications. However, the detection methods are generally validated by assuming a single implicitly known attack strategy, which does not necessarily account for real-life threats. Indeed, this can lead to an overoptimistic assessment of the detectors' performance and may induce some bias in the comparison between competing detection schemes. We propose a novel multi-armed framework, called MEAD, for evaluating detectors based on several attack strategies to overcome this limitation. Among them, we make use of three new objectives to generate attacks. The proposed performance metric is based on the worst-case scenario: detection is successful if and only if all different attacks are correctly recognized. Empirically, we show the effectiveness of our approach. Moreover, the poor performance obtained for state-of-the-art detectors opens a new exciting line of research.



## **22. The Topological BERT: Transforming Attention into Topology for Natural Language Processing**

cs.CL

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2206.15195v1)

**Authors**: Ilan Perez, Raphael Reinauer

**Abstracts**: In recent years, the introduction of the Transformer models sparked a revolution in natural language processing (NLP). BERT was one of the first text encoders using only the attention mechanism without any recurrent parts to achieve state-of-the-art results on many NLP tasks.   This paper introduces a text classifier using topological data analysis. We use BERT's attention maps transformed into attention graphs as the only input to that classifier. The model can solve tasks such as distinguishing spam from ham messages, recognizing whether a sentence is grammatically correct, or evaluating a movie review as negative or positive. It performs comparably to the BERT baseline and outperforms it on some tasks.   Additionally, we propose a new method to reduce the number of BERT's attention heads considered by the topological classifier, which allows us to prune the number of heads from 144 down to as few as ten with no reduction in performance. Our work also shows that the topological model displays higher robustness against adversarial attacks than the original BERT model, which is maintained during the pruning process. To the best of our knowledge, this work is the first to confront topological-based models with adversarial attacks in the context of NLP.



## **23. Reliable Representations Make A Stronger Defender: Unsupervised Structure Refinement for Robust GNN**

cs.LG

Accepted in KDD2022

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2207.00012v1)

**Authors**: Kuan Li, Yang Liu, Xiang Ao, Jianfeng Chi, Jinghua Feng, Hao Yang, Qing He

**Abstracts**: Benefiting from the message passing mechanism, Graph Neural Networks (GNNs) have been successful on flourish tasks over graph data. However, recent studies have shown that attackers can catastrophically degrade the performance of GNNs by maliciously modifying the graph structure. A straightforward solution to remedy this issue is to model the edge weights by learning a metric function between pairwise representations of two end nodes, which attempts to assign low weights to adversarial edges. The existing methods use either raw features or representations learned by supervised GNNs to model the edge weights. However, both strategies are faced with some immediate problems: raw features cannot represent various properties of nodes (e.g., structure information), and representations learned by supervised GNN may suffer from the poor performance of the classifier on the poisoned graph. We need representations that carry both feature information and as mush correct structure information as possible and are insensitive to structural perturbations. To this end, we propose an unsupervised pipeline, named STABLE, to optimize the graph structure. Finally, we input the well-refined graph into a downstream classifier. For this part, we design an advanced GCN that significantly enhances the robustness of vanilla GCN without increasing the time complexity. Extensive experiments on four real-world graph benchmarks demonstrate that STABLE outperforms the state-of-the-art methods and successfully defends against various attacks.



## **24. FIDO2 With Two Displays$\unicode{x2013}$Or How to Protect Security-Critical Web Transactions Against Malware Attacks**

cs.CR

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2206.13358v2)

**Authors**: Timon Hackenjos, Benedikt Wagner, Julian Herr, Jochen Rill, Marek Wehmer, Niklas Goerke, Ingmar Baumgart

**Abstracts**: With the rise of attacks on online accounts in the past years, more and more services offer two-factor authentication for their users. Having factors out of two of the three categories something you know, something you have and something you are should ensure that an attacker cannot compromise two of them at once. Thus, an adversary should not be able to maliciously interact with one's account. However, this is only true if one considers a weak adversary. In particular, since most current solutions only authenticate a session and not individual transactions, they are noneffective if one's device is infected with malware. For online banking, the banking industry has long since identified the need for authenticating transactions. However, specifications of such authentication schemes are not public and implementation details vary wildly from bank to bank with most still being unable to protect against malware. In this work, we present a generic approach to tackle the problem of malicious account takeovers, even in the presence of malware. To this end, we define a new paradigm to improve two-factor authentication that involves the concepts of one-out-of-two security and transaction authentication. Web authentication schemes following this paradigm can protect security-critical transactions against manipulation, even if one of the factors is completely compromised. Analyzing existing authentication schemes, we find that they do not realize one-out-of-two security. We give a blueprint of how to design secure web authentication schemes in general. Based on this blueprint we propose FIDO2 With Two Displays (FIDO2D), a new web authentication scheme based on the FIDO2 standard and prove its security using Tamarin. We hope that our work inspires a new wave of more secure web authentication schemes, which protect security-critical transactions even against attacks with malware.



## **25. An Intermediate-level Attack Framework on The Basis of Linear Regression**

cs.CV

Accepted by TPAMI; Code is available at  https://github.com/qizhangli/ila-plus-plus-lr

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2203.10723v2)

**Authors**: Yiwen Guo, Qizhang Li, Wangmeng Zuo, Hao Chen

**Abstracts**: This paper substantially extends our work published at ECCV, in which an intermediate-level attack was proposed to improve the transferability of some baseline adversarial examples. Specifically, we advocate a framework in which a direct linear mapping from the intermediate-level discrepancies (between adversarial features and benign features) to prediction loss of the adversarial example is established. By delving deep into the core components of such a framework, we show that 1) a variety of linear regression models can all be considered in order to establish the mapping, 2) the magnitude of the finally obtained intermediate-level adversarial discrepancy is correlated with the transferability, 3) further boost of the performance can be achieved by performing multiple runs of the baseline attack with random initialization. In addition, by leveraging these findings, we achieve new state-of-the-arts on transfer-based $\ell_\infty$ and $\ell_2$ attacks. Our code is publicly available at https://github.com/qizhangli/ila-plus-plus-lr.



## **26. On the Challenges of Detecting Side-Channel Attacks in SGX**

cs.CR

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2011.14599v2)

**Authors**: Jianyu Jiang, Claudio Soriente, Ghassan Karame

**Abstracts**: Existing tools to detect side-channel attacks on Intel SGX are grounded on the observation that attacks affect the performance of the victim application. As such, all detection tools monitor the potential victim and raise an alarm if the witnessed performance (in terms of runtime, enclave interruptions, cache misses, etc.) is out of the ordinary.   In this paper, we show that monitoring the performance of enclaves to detect side-channel attacks may not be effective. Our core intuition is that all monitoring tools are geared towards an adversary that interferes with the victim's execution in order to extract the most number of secret bits (e.g., the entire secret) in one or few runs. They cannot, however, detect an adversary that leaks smaller portions of the secret - as small as a single bit - at each execution of the victim. In particular, by minimizing the information leaked at each run, the impact of any side-channel attack on the application's performance is significantly lowered - ensuring that the detection tool does not detect an attack. By repeating the attack multiple times, each time on a different part of the secret, the adversary can recover the whole secret and remain undetected. Based on this intuition, we adapt known attacks leveraging page-tables and L3 cache to bypass existing detection mechanisms. We show experimentally how an attacker can successfully exfiltrate the secret key used in an enclave running various cryptographic routines of libgcrypt. Beyond cryptographic libraries, we also show how to compromise the predictions of enclaves running decision-tree routines of OpenCV. Our evaluation results suggest that performance-based detection tools do not deter side-channel attacks on SGX enclaves and that effective detection mechanisms are yet to be designed.



## **27. Depth-2 Neural Networks Under a Data-Poisoning Attack**

cs.LG

32 page, 7 figures

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2005.01699v3)

**Authors**: Sayar Karmakar, Anirbit Mukherjee, Theodore Papamarkou

**Abstracts**: In this work, we study the possibility of defending against data-poisoning attacks while training a shallow neural network in a regression setup. We focus on doing supervised learning for a class of depth-2 finite-width neural networks, which includes single-filter convolutional networks. In this class of networks, we attempt to learn the network weights in the presence of a malicious oracle doing stochastic, bounded and additive adversarial distortions on the true output during training. For the non-gradient stochastic algorithm that we construct, we prove worst-case near-optimal trade-offs among the magnitude of the adversarial attack, the weight approximation accuracy, and the confidence achieved by the proposed algorithm. As our algorithm uses mini-batching, we analyze how the mini-batch size affects convergence. We also show how to utilize the scaling of the outer layer weights to counter output-poisoning attacks depending on the probability of attack. Lastly, we give experimental evidence demonstrating how our algorithm outperforms stochastic gradient descent under different input data distributions, including instances of heavy-tailed distributions.



## **28. IBP Regularization for Verified Adversarial Robustness via Branch-and-Bound**

cs.LG

ICML 2022 Workshop on Formal Verification of Machine Learning

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2206.14772v1)

**Authors**: Alessandro De Palma, Rudy Bunel, Krishnamurthy Dvijotham, M. Pawan Kumar, Robert Stanforth

**Abstracts**: Recent works have tried to increase the verifiability of adversarially trained networks by running the attacks over domains larger than the original perturbations and adding various regularization terms to the objective. However, these algorithms either underperform or require complex and expensive stage-wise training procedures, hindering their practical applicability. We present IBP-R, a novel verified training algorithm that is both simple and effective. IBP-R induces network verifiability by coupling adversarial attacks on enlarged domains with a regularization term, based on inexpensive interval bound propagation, that minimizes the gap between the non-convex verification problem and its approximations. By leveraging recent branch-and-bound frameworks, we show that IBP-R obtains state-of-the-art verified robustness-accuracy trade-offs for small perturbations on CIFAR-10 while training significantly faster than relevant previous work. Additionally, we present UPB, a novel branching strategy that, relying on a simple heuristic based on $\beta$-CROWN, reduces the cost of state-of-the-art branching algorithms while yielding splits of comparable quality.



## **29. longhorns at DADC 2022: How many linguists does it take to fool a Question Answering model? A systematic approach to adversarial attacks**

cs.CL

Accepted at DADC2022

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2206.14729v1)

**Authors**: Venelin Kovatchev, Trina Chatterjee, Venkata S Govindarajan, Jifan Chen, Eunsol Choi, Gabriella Chronis, Anubrata Das, Katrin Erk, Matthew Lease, Junyi Jessy Li, Yating Wu, Kyle Mahowald

**Abstracts**: Developing methods to adversarially challenge NLP systems is a promising avenue for improving both model performance and interpretability. Here, we describe the approach of the team "longhorns" on Task 1 of the The First Workshop on Dynamic Adversarial Data Collection (DADC), which asked teams to manually fool a model on an Extractive Question Answering task. Our team finished first, with a model error rate of 62%. We advocate for a systematic, linguistically informed approach to formulating adversarial questions, and we describe the results of our pilot experiments, as well as our official submission.



## **30. Private Graph Extraction via Feature Explanations**

cs.LG

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2206.14724v1)

**Authors**: Iyiola E. Olatunji, Mandeep Rathee, Thorben Funke, Megha Khosla

**Abstracts**: Privacy and interpretability are two of the important ingredients for achieving trustworthy machine learning. We study the interplay of these two aspects in graph machine learning through graph reconstruction attacks. The goal of the adversary here is to reconstruct the graph structure of the training data given access to model explanations. Based on the different kinds of auxiliary information available to the adversary, we propose several graph reconstruction attacks. We show that additional knowledge of post-hoc feature explanations substantially increases the success rate of these attacks. Further, we investigate in detail the differences between attack performance with respect to three different classes of explanation methods for graph neural networks: gradient-based, perturbation-based, and surrogate model-based methods. While gradient-based explanations reveal the most in terms of the graph structure, we find that these explanations do not always score high in utility. For the other two classes of explanations, privacy leakage increases with an increase in explanation utility. Finally, we propose a defense based on a randomized response mechanism for releasing the explanations which substantially reduces the attack success rate. Our anonymized code is available.



## **31. Enhancing Security of Memristor Computing System Through Secure Weight Mapping**

cs.ET

6 pages, 4 figures, accepted by IEEE ISVLSI 2022

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2206.14498v1)

**Authors**: Minhui Zou, Junlong Zhou, Xiaotong Cui, Wei Wang, Shahar Kvatinsky

**Abstracts**: Emerging memristor computing systems have demonstrated great promise in improving the energy efficiency of neural network (NN) algorithms. The NN weights stored in memristor crossbars, however, may face potential theft attacks due to the nonvolatility of the memristor devices. In this paper, we propose to protect the NN weights by mapping selected columns of them in the form of 1's complements and leaving the other columns in their original form, preventing the adversary from knowing the exact representation of each weight. The results show that compared with prior work, our method achieves effectiveness comparable to the best of them and reduces the hardware overhead by more than 18X.



## **32. Guided Diffusion Model for Adversarial Purification**

cs.CV

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2205.14969v3)

**Authors**: Jinyi Wang, Zhaoyang Lyu, Dahua Lin, Bo Dai, Hongfei Fu

**Abstracts**: With wider application of deep neural networks (DNNs) in various algorithms and frameworks, security threats have become one of the concerns. Adversarial attacks disturb DNN-based image classifiers, in which attackers can intentionally add imperceptible adversarial perturbations on input images to fool the classifiers. In this paper, we propose a novel purification approach, referred to as guided diffusion model for purification (GDMP), to help protect classifiers from adversarial attacks. The core of our approach is to embed purification into the diffusion denoising process of a Denoised Diffusion Probabilistic Model (DDPM), so that its diffusion process could submerge the adversarial perturbations with gradually added Gaussian noises, and both of these noises can be simultaneously removed following a guided denoising process. On our comprehensive experiments across various datasets, the proposed GDMP is shown to reduce the perturbations raised by adversarial attacks to a shallow range, thereby significantly improving the correctness of classification. GDMP improves the robust accuracy by 5%, obtaining 90.1% under PGD attack on the CIFAR10 dataset. Moreover, GDMP achieves 70.94% robustness on the challenging ImageNet dataset.



## **33. A Deep Learning Approach to Create DNS Amplification Attacks**

cs.CR

12 pages, 6 figures, Conference: to 2022 4th International Conference  on Management Science and Industrial Engineering (MSIE) (MSIE 2022), DOI:  https://doi.org/10.1145/3535782.3535838, accepted to conference above, not  yet published

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2206.14346v1)

**Authors**: Jared Mathews, Prosenjit Chatterjee, Shankar Banik, Cory Nance

**Abstracts**: In recent years, deep learning has shown itself to be an incredibly valuable tool in cybersecurity as it helps network intrusion detection systems to classify attacks and detect new ones. Adversarial learning is the process of utilizing machine learning to generate a perturbed set of inputs to then feed to the neural network to misclassify it. Much of the current work in the field of adversarial learning has been conducted in image processing and natural language processing with a wide variety of algorithms. Two algorithms of interest are the Elastic-Net Attack on Deep Neural Networks and TextAttack. In our experiment the EAD and TextAttack algorithms are applied to a Domain Name System amplification classifier. The algorithms are used to generate malicious Distributed Denial of Service adversarial examples to then feed as inputs to the network intrusion detection systems neural network to classify as valid traffic. We show in this work that both image processing and natural language processing adversarial learning algorithms can be applied against a network intrusion detection neural network.



## **34. Linear Model Against Malicious Adversaries with Local Differential Privacy**

cs.CR

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2202.02448v2)

**Authors**: Guanhong Miao, A. Adam Ding, Samuel S. Wu

**Abstracts**: Scientific collaborations benefit from collaborative learning of distributed sources, but remain difficult to achieve when data are sensitive. In recent years, privacy preserving techniques have been widely studied to analyze distributed data across different agencies while protecting sensitive information. Most existing privacy preserving techniques are designed to resist semi-honest adversaries and require intense computation to perform data analysis. Secure collaborative learning is significantly difficult with the presence of malicious adversaries who may deviates from the secure protocol. Another challenge is to maintain high computation efficiency with privacy protection. In this paper, matrix encryption is applied to encrypt data such that the secure schemes are against malicious adversaries, including chosen plaintext attack, known plaintext attack, and collusion attack. The encryption scheme also achieves local differential privacy. Moreover, cross validation is studied to prevent overfitting without additional communication cost. Empirical experiments on real-world datasets demonstrate that the proposed schemes are computationally efficient compared to existing techniques against malicious adversary and semi-honest model.



## **35. An Empirical Study of Challenges in Converting Deep Learning Models**

cs.LG

Accepted for publication in ICSME 2022

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2206.14322v1)

**Authors**: Moses Openja, Amin Nikanjam, Ahmed Haj Yahmed, Foutse Khomh, Zhen Ming, Jiang

**Abstracts**: There is an increase in deploying Deep Learning (DL)-based software systems in real-world applications. Usually DL models are developed and trained using DL frameworks that have their own internal mechanisms/formats to represent and train DL models, and usually those formats cannot be recognized by other frameworks. Moreover, trained models are usually deployed in environments different from where they were developed. To solve the interoperability issue and make DL models compatible with different frameworks/environments, some exchange formats are introduced for DL models, like ONNX and CoreML. However, ONNX and CoreML were never empirically evaluated by the community to reveal their prediction accuracy, performance, and robustness after conversion. Poor accuracy or non-robust behavior of converted models may lead to poor quality of deployed DL-based software systems. We conduct, in this paper, the first empirical study to assess ONNX and CoreML for converting trained DL models. In our systematic approach, two popular DL frameworks, Keras and PyTorch, are used to train five widely used DL models on three popular datasets. The trained models are then converted to ONNX and CoreML and transferred to two runtime environments designated for such formats, to be evaluated. We investigate the prediction accuracy before and after conversion. Our results unveil that the prediction accuracy of converted models are at the same level of originals. The performance (time cost and memory consumption) of converted models are studied as well. The size of models are reduced after conversion, which can result in optimized DL-based software deployment. Converted models are generally assessed as robust at the same level of originals. However, obtained results show that CoreML models are more vulnerable to adversarial attacks compared to ONNX.



## **36. Collecting high-quality adversarial data for machine reading comprehension tasks with humans and models in the loop**

cs.CL

8 pages, 3 figures, for more information about the shared task please  go to https://dadcworkshop.github.io/

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2206.14272v1)

**Authors**: Damian Y. Romero Diaz, Magdalena Anioł, John Culnan

**Abstracts**: We present our experience as annotators in the creation of high-quality, adversarial machine-reading-comprehension data for extractive QA for Task 1 of the First Workshop on Dynamic Adversarial Data Collection (DADC). DADC is an emergent data collection paradigm with both models and humans in the loop. We set up a quasi-experimental annotation design and perform quantitative analyses across groups with different numbers of annotators focusing on successful adversarial attacks, cost analysis, and annotator confidence correlation. We further perform a qualitative analysis of our perceived difficulty of the task given the different topics of the passages in our dataset and conclude with recommendations and suggestions that might be of value to people working on future DADC tasks and related annotation interfaces.



## **37. How to Steer Your Adversary: Targeted and Efficient Model Stealing Defenses with Gradient Redirection**

cs.LG

ICML 2022

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2206.14157v1)

**Authors**: Mantas Mazeika, Bo Li, David Forsyth

**Abstracts**: Model stealing attacks present a dilemma for public machine learning APIs. To protect financial investments, companies may be forced to withhold important information about their models that could facilitate theft, including uncertainty estimates and prediction explanations. This compromise is harmful not only to users but also to external transparency. Model stealing defenses seek to resolve this dilemma by making models harder to steal while preserving utility for benign users. However, existing defenses have poor performance in practice, either requiring enormous computational overheads or severe utility trade-offs. To meet these challenges, we present a new approach to model stealing defenses called gradient redirection. At the core of our approach is a provably optimal, efficient algorithm for steering an adversary's training updates in a targeted manner. Combined with improvements to surrogate networks and a novel coordinated defense strategy, our gradient redirection defense, called GRAD${}^2$, achieves small utility trade-offs and low computational overhead, outperforming the best prior defenses. Moreover, we demonstrate how gradient redirection enables reprogramming the adversary with arbitrary behavior, which we hope will foster work on new avenues of defense.



## **38. Debiasing Learning for Membership Inference Attacks Against Recommender Systems**

cs.IR

Accepted by KDD 2022

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2206.12401v2)

**Authors**: Zihan Wang, Na Huang, Fei Sun, Pengjie Ren, Zhumin Chen, Hengliang Luo, Maarten de Rijke, Zhaochun Ren

**Abstracts**: Learned recommender systems may inadvertently leak information about their training data, leading to privacy violations. We investigate privacy threats faced by recommender systems through the lens of membership inference. In such attacks, an adversary aims to infer whether a user's data is used to train the target recommender. To achieve this, previous work has used a shadow recommender to derive training data for the attack model, and then predicts the membership by calculating difference vectors between users' historical interactions and recommended items. State-of-the-art methods face two challenging problems: (1) training data for the attack model is biased due to the gap between shadow and target recommenders, and (2) hidden states in recommenders are not observational, resulting in inaccurate estimations of difference vectors. To address the above limitations, we propose a Debiasing Learning for Membership Inference Attacks against recommender systems (DL-MIA) framework that has four main components: (1) a difference vector generator, (2) a disentangled encoder, (3) a weight estimator, and (4) an attack model. To mitigate the gap between recommenders, a variational auto-encoder (VAE) based disentangled encoder is devised to identify recommender invariant and specific features. To reduce the estimation bias, we design a weight estimator, assigning a truth-level score for each difference vector to indicate estimation accuracy. We evaluate DL-MIA against both general recommenders and sequential recommenders on three real-world datasets. Experimental results show that DL-MIA effectively alleviates training and estimation biases simultaneously, and achieves state-of-the-art attack performance.



## **39. On the amplification of security and privacy risks by post-hoc explanations in machine learning models**

cs.LG

9 pages, appendix: 2 pages

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2206.14004v1)

**Authors**: Pengrui Quan, Supriyo Chakraborty, Jeya Vikranth Jeyakumar, Mani Srivastava

**Abstracts**: A variety of explanation methods have been proposed in recent years to help users gain insights into the results returned by neural networks, which are otherwise complex and opaque black-boxes. However, explanations give rise to potential side-channels that can be leveraged by an adversary for mounting attacks on the system. In particular, post-hoc explanation methods that highlight input dimensions according to their importance or relevance to the result also leak information that weakens security and privacy. In this work, we perform the first systematic characterization of the privacy and security risks arising from various popular explanation techniques. First, we propose novel explanation-guided black-box evasion attacks that lead to 10 times reduction in query count for the same success rate. We show that the adversarial advantage from explanations can be quantified as a reduction in the total variance of the estimated gradient. Second, we revisit the membership information leaked by common explanations. Contrary to observations in prior studies, via our modified attacks we show significant leakage of membership information (above 100% improvement over prior results), even in a much stricter black-box setting. Finally, we study explanation-guided model extraction attacks and demonstrate adversarial gains through a large reduction in query count.



## **40. Increasing Confidence in Adversarial Robustness Evaluations**

cs.LG

Oral at CVPR 2022 Workshop (Art of Robustness). Project website  https://zimmerrol.github.io/active-tests/

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2206.13991v1)

**Authors**: Roland S. Zimmermann, Wieland Brendel, Florian Tramer, Nicholas Carlini

**Abstracts**: Hundreds of defenses have been proposed to make deep neural networks robust against minimal (adversarial) input perturbations. However, only a handful of these defenses held up their claims because correctly evaluating robustness is extremely challenging: Weak attacks often fail to find adversarial examples even if they unknowingly exist, thereby making a vulnerable network look robust. In this paper, we propose a test to identify weak attacks, and thus weak defense evaluations. Our test slightly modifies a neural network to guarantee the existence of an adversarial example for every sample. Consequentially, any correct attack must succeed in breaking this modified network. For eleven out of thirteen previously-published defenses, the original evaluation of the defense fails our test, while stronger attacks that break these defenses pass it. We hope that attack unit tests - such as ours - will be a major component in future robustness evaluations and increase confidence in an empirical field that is currently riddled with skepticism.



## **41. Ownership Verification of DNN Architectures via Hardware Cache Side Channels**

cs.CR

The paper has been accepted by IEEE Transactions on Circuits and  Systems for Video Technology

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2102.03523v4)

**Authors**: Xiaoxuan Lou, Shangwei Guo, Jiwei Li, Tianwei Zhang

**Abstracts**: Deep Neural Networks (DNN) are gaining higher commercial values in computer vision applications, e.g., image classification, video analytics, etc. This calls for urgent demands of the intellectual property (IP) protection of DNN models. In this paper, we present a novel watermarking scheme to achieve the ownership verification of DNN architectures. Existing works all embedded watermarks into the model parameters while treating the architecture as public property. These solutions were proven to be vulnerable by an adversary to detect or remove the watermarks. In contrast, we claim the model architectures as an important IP for model owners, and propose to implant watermarks into the architectures. We design new algorithms based on Neural Architecture Search (NAS) to generate watermarked architectures, which are unique enough to represent the ownership, while maintaining high model usability. Such watermarks can be extracted via side-channel-based model extraction techniques with high fidelity. We conduct comprehensive experiments on watermarked CNN models for image classification tasks and the experimental results show our scheme has negligible impact on the model performance, and exhibits strong robustness against various model transformations and adaptive attacks.



## **42. Deep Image Destruction: Vulnerability of Deep Image-to-Image Models against Adversarial Attacks**

cs.CV

ICPR2022

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2104.15022v2)

**Authors**: Jun-Ho Choi, Huan Zhang, Jun-Hyuk Kim, Cho-Jui Hsieh, Jong-Seok Lee

**Abstracts**: Recently, the vulnerability of deep image classification models to adversarial attacks has been investigated. However, such an issue has not been thoroughly studied for image-to-image tasks that take an input image and generate an output image (e.g., colorization, denoising, deblurring, etc.) This paper presents comprehensive investigations into the vulnerability of deep image-to-image models to adversarial attacks. For five popular image-to-image tasks, 16 deep models are analyzed from various standpoints such as output quality degradation due to attacks, transferability of adversarial examples across different tasks, and characteristics of perturbations. We show that unlike image classification tasks, the performance degradation on image-to-image tasks largely differs depending on various factors, e.g., attack methods and task objectives. In addition, we analyze the effectiveness of conventional defense methods used for classification models in improving the robustness of the image-to-image models.



## **43. Improving Privacy and Security in Unmanned Aerial Vehicles Network using Blockchain**

cs.CR

18 Pages; 14 Figures; 2 Tables

**SubmitDate**: 2022-06-27    [paper-pdf](http://arxiv.org/pdf/2201.06100v2)

**Authors**: Hardik Sachdeva, Shivam Gupta, Anushka Misra, Khushbu Chauhan, Mayank Dave

**Abstracts**: Unmanned Aerial Vehicles (UAVs), also known as drones, have exploded in every segment present in todays business industry. They have scope in reinventing old businesses, and they are even developing new opportunities for various brands and franchisors. UAVs are used in the supply chain, maintaining surveillance and serving as mobile hotspots. Although UAVs have potential applications, they bring several societal concerns and challenges that need addressing in public safety, privacy, and cyber security. UAVs are prone to various cyber-attacks and vulnerabilities; they can also be hacked and misused by malicious entities resulting in cyber-crime. The adversaries can exploit these vulnerabilities, leading to data loss, property, and destruction of life. One can partially detect the attacks like false information dissemination, jamming, gray hole, blackhole, and GPS spoofing by monitoring the UAV behavior, but it may not resolve privacy issues. This paper presents secure communication between UAVs using blockchain technology. Our approach involves building smart contracts and making a secure and reliable UAV adhoc network. This network will be resilient to various network attacks and is secure against malicious intrusions.



## **44. Adversarially Robust Learning of Real-Valued Functions**

cs.LG

**SubmitDate**: 2022-06-26    [paper-pdf](http://arxiv.org/pdf/2206.12977v1)

**Authors**: Idan Attias, Steve Hanneke

**Abstracts**: We study robustness to test-time adversarial attacks in the regression setting with $\ell_p$ losses and arbitrary perturbation sets. We address the question of which function classes are PAC learnable in this setting. We show that classes of finite fat-shattering dimension are learnable. Moreover, for convex function classes, they are even properly learnable. In contrast, some non-convex function classes provably require improper learning algorithms. We also discuss extensions to agnostic learning. Our main technique is based on a construction of an adversarially robust sample compression scheme of a size determined by the fat-shattering dimension.



## **45. Cascading Failures in Smart Grids under Random, Targeted and Adaptive Attacks**

cs.SI

Accepted for publication as a book chapter. arXiv admin note:  substantial text overlap with arXiv:1402.6809

**SubmitDate**: 2022-06-25    [paper-pdf](http://arxiv.org/pdf/2206.12735v1)

**Authors**: Sushmita Ruj, Arindam Pal

**Abstracts**: We study cascading failures in smart grids, where an attacker selectively compromises the nodes with probabilities proportional to their degrees, betweenness, or clustering coefficient. This implies that nodes with high degrees, betweenness, or clustering coefficients are attacked with higher probability. We mathematically and experimentally analyze the sizes of the giant components of the networks under different types of targeted attacks, and compare the results with the corresponding sizes under random attacks. We show that networks disintegrate faster for targeted attacks compared to random attacks. A targeted attack on a small fraction of high degree nodes disintegrates one or both of the networks, whereas both the networks contain giant components for random attack on the same fraction of nodes. An important observation is that an attacker has an advantage if it compromises nodes based on their betweenness, rather than based on degree or clustering coefficient.   We next study adaptive attacks, where an attacker compromises nodes in rounds. Here, some nodes are compromised in each round based on their degree, betweenness or clustering coefficients, instead of compromising all nodes together. In this case, the degree, betweenness, or clustering coefficient is calculated before the start of each round, instead of at the beginning. We show experimentally that an adversary has an advantage in this adaptive approach, compared to compromising the same number of nodes all at once.



## **46. Empirical Evaluation of Physical Adversarial Patch Attacks Against Overhead Object Detection Models**

cs.CV

**SubmitDate**: 2022-06-25    [paper-pdf](http://arxiv.org/pdf/2206.12725v1)

**Authors**: Gavin S. Hartnett, Li Ang Zhang, Caolionn O'Connell, Andrew J. Lohn, Jair Aguirre

**Abstracts**: Adversarial patches are images designed to fool otherwise well-performing neural network-based computer vision models. Although these attacks were initially conceived of and studied digitally, in that the raw pixel values of the image were perturbed, recent work has demonstrated that these attacks can successfully transfer to the physical world. This can be accomplished by printing out the patch and adding it into scenes of newly captured images or video footage. In this work we further test the efficacy of adversarial patch attacks in the physical world under more challenging conditions. We consider object detection models trained on overhead imagery acquired through aerial or satellite cameras, and we test physical adversarial patches inserted into scenes of a desert environment. Our main finding is that it is far more difficult to successfully implement the adversarial patch attacks under these conditions than in the previously considered conditions. This has important implications for AI safety as the real-world threat posed by adversarial examples may be overstated.



## **47. Defending Multimodal Fusion Models against Single-Source Adversaries**

cs.CV

CVPR 2021

**SubmitDate**: 2022-06-25    [paper-pdf](http://arxiv.org/pdf/2206.12714v1)

**Authors**: Karren Yang, Wan-Yi Lin, Manash Barman, Filipe Condessa, Zico Kolter

**Abstracts**: Beyond achieving high performance across many vision tasks, multimodal models are expected to be robust to single-source faults due to the availability of redundant information between modalities. In this paper, we investigate the robustness of multimodal neural networks against worst-case (i.e., adversarial) perturbations on a single modality. We first show that standard multimodal fusion models are vulnerable to single-source adversaries: an attack on any single modality can overcome the correct information from multiple unperturbed modalities and cause the model to fail. This surprising vulnerability holds across diverse multimodal tasks and necessitates a solution. Motivated by this finding, we propose an adversarially robust fusion strategy that trains the model to compare information coming from all the input sources, detect inconsistencies in the perturbed modality compared to the other modalities, and only allow information from the unperturbed modalities to pass through. Our approach significantly improves on state-of-the-art methods in single-source robustness, achieving gains of 7.8-25.2% on action recognition, 19.7-48.2% on object detection, and 1.6-6.7% on sentiment analysis, without degrading performance on unperturbed (i.e., clean) data.



## **48. Defense against adversarial attacks on deep convolutional neural networks through nonlocal denoising**

cs.CV

**SubmitDate**: 2022-06-25    [paper-pdf](http://arxiv.org/pdf/2206.12685v1)

**Authors**: Sandhya Aneja, Nagender Aneja, Pg Emeroylariffion Abas, Abdul Ghani Naim

**Abstracts**: Despite substantial advances in network architecture performance, the susceptibility of adversarial attacks makes deep learning challenging to implement in safety-critical applications. This paper proposes a data-centric approach to addressing this problem. A nonlocal denoising method with different luminance values has been used to generate adversarial examples from the Modified National Institute of Standards and Technology database (MNIST) and Canadian Institute for Advanced Research (CIFAR-10) data sets. Under perturbation, the method provided absolute accuracy improvements of up to 9.3% in the MNIST data set and 13% in the CIFAR-10 data set. Training using transformed images with higher luminance values increases the robustness of the classifier. We have shown that transfer learning is disadvantageous for adversarial machine learning. The results indicate that simple adversarial examples can improve resilience and make deep learning easier to apply in various applications.



## **49. RSTAM: An Effective Black-Box Impersonation Attack on Face Recognition using a Mobile and Compact Printer**

cs.CV

**SubmitDate**: 2022-06-25    [paper-pdf](http://arxiv.org/pdf/2206.12590v1)

**Authors**: Xiaoliang Liu, Furao Shen, Jian Zhao, Changhai Nie

**Abstracts**: Face recognition has achieved considerable progress in recent years thanks to the development of deep neural networks, but it has recently been discovered that deep neural networks are vulnerable to adversarial examples. This means that face recognition models or systems based on deep neural networks are also susceptible to adversarial examples. However, the existing methods of attacking face recognition models or systems with adversarial examples can effectively complete white-box attacks but not black-box impersonation attacks, physical attacks, or convenient attacks, particularly on commercial face recognition systems. In this paper, we propose a new method to attack face recognition models or systems called RSTAM, which enables an effective black-box impersonation attack using an adversarial mask printed by a mobile and compact printer. First, RSTAM enhances the transferability of the adversarial masks through our proposed random similarity transformation strategy. Furthermore, we propose a random meta-optimization strategy for ensembling several pre-trained face models to generate more general adversarial masks. Finally, we conduct experiments on the CelebA-HQ, LFW, Makeup Transfer (MT), and CASIA-FaceV5 datasets. The performance of the attacks is also evaluated on state-of-the-art commercial face recognition systems: Face++, Baidu, Aliyun, Tencent, and Microsoft. Extensive experiments show that RSTAM can effectively perform black-box impersonation attacks on face recognition models or systems.



## **50. Defending Backdoor Attacks on Vision Transformer via Patch Processing**

cs.CV

**SubmitDate**: 2022-06-24    [paper-pdf](http://arxiv.org/pdf/2206.12381v1)

**Authors**: Khoa D. Doan, Yingjie Lao, Peng Yang, Ping Li

**Abstracts**: Vision Transformers (ViTs) have a radically different architecture with significantly less inductive bias than Convolutional Neural Networks. Along with the improvement in performance, security and robustness of ViTs are also of great importance to study. In contrast to many recent works that exploit the robustness of ViTs against adversarial examples, this paper investigates a representative causative attack, i.e., backdoor. We first examine the vulnerability of ViTs against various backdoor attacks and find that ViTs are also quite vulnerable to existing attacks. However, we observe that the clean-data accuracy and backdoor attack success rate of ViTs respond distinctively to patch transformations before the positional encoding. Then, based on this finding, we propose an effective method for ViTs to defend both patch-based and blending-based trigger backdoor attacks via patch processing. The performances are evaluated on several benchmark datasets, including CIFAR10, GTSRB, and TinyImageNet, which show the proposed novel defense is very successful in mitigating backdoor attacks for ViTs. To the best of our knowledge, this paper presents the first defensive strategy that utilizes a unique characteristic of ViTs against backdoor attacks.



