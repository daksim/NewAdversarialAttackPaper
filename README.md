# Latest Adversarial Attack Papers
**update at 2022-11-10 19:44:40**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Accountable and Explainable Methods for Complex Reasoning over Text**

cs.LG

PhD Thesis

**SubmitDate**: 2022-11-09    [abs](http://arxiv.org/abs/2211.04946v1) [paper-pdf](http://arxiv.org/pdf/2211.04946v1)

**Authors**: Pepa Atanasova

**Abstract**: A major concern of Machine Learning (ML) models is their opacity. They are deployed in an increasing number of applications where they often operate as black boxes that do not provide explanations for their predictions. Among others, the potential harms associated with the lack of understanding of the models' rationales include privacy violations, adversarial manipulations, and unfair discrimination. As a result, the accountability and transparency of ML models have been posed as critical desiderata by works in policy and law, philosophy, and computer science.   In computer science, the decision-making process of ML models has been studied by developing accountability and transparency methods. Accountability methods, such as adversarial attacks and diagnostic datasets, expose vulnerabilities of ML models that could lead to malicious manipulations or systematic faults in their predictions. Transparency methods explain the rationales behind models' predictions gaining the trust of relevant stakeholders and potentially uncovering mistakes and unfairness in models' decisions. To this end, transparency methods have to meet accountability requirements as well, e.g., being robust and faithful to the underlying rationales of a model.   This thesis presents my research that expands our collective knowledge in the areas of accountability and transparency of ML models developed for complex reasoning tasks over text.



## **2. Lipschitz Continuous Algorithms for Graph Problems**

cs.DS

**SubmitDate**: 2022-11-09    [abs](http://arxiv.org/abs/2211.04674v1) [paper-pdf](http://arxiv.org/pdf/2211.04674v1)

**Authors**: Soh Kumabe, Yuichi Yoshida

**Abstract**: It has been widely observed in the machine learning community that a small perturbation to the input can cause a large change in the prediction of a trained model, and such phenomena have been intensively studied in the machine learning community under the name of adversarial attacks. Because graph algorithms also are widely used for decision making and knowledge discovery, it is important to design graph algorithms that are robust against adversarial attacks. In this study, we consider the Lipschitz continuity of algorithms as a robustness measure and initiate a systematic study of the Lipschitz continuity of algorithms for (weighted) graph problems.   Depending on how we embed the output solution to a metric space, we can think of several Lipschitzness notions. We mainly consider the one that is invariant under scaling of weights, and we provide Lipschitz continuous algorithms and lower bounds for the minimum spanning tree problem, the shortest path problem, and the maximum weight matching problem. In particular, our shortest path algorithm is obtained by first designing an algorithm for unweighted graphs that are robust against edge contractions and then applying it to the unweighted graph constructed from the original weighted graph.   Then, we consider another Lipschitzness notion induced by a natural mapping that maps the output solution to its characteristic vector. It turns out that no Lipschitz continuous algorithm exists for this Lipschitz notion, and we instead design algorithms with bounded pointwise Lipschitz constants for the minimum spanning tree problem and the maximum weight bipartite matching problem. Our algorithm for the latter problem is based on an LP relaxation with entropy regularization.



## **3. FedDef: Defense Against Gradient Leakage in Federated Learning-based Network Intrusion Detection Systems**

cs.CR

14 pages, 9 figures, submitted to TIFS

**SubmitDate**: 2022-11-09    [abs](http://arxiv.org/abs/2210.04052v2) [paper-pdf](http://arxiv.org/pdf/2210.04052v2)

**Authors**: Jiahui Chen, Yi Zhao, Qi Li, Xuewei Feng, Ke Xu

**Abstract**: Deep learning (DL) methods have been widely applied to anomaly-based network intrusion detection system (NIDS) to detect malicious traffic. To expand the usage scenarios of DL-based methods, the federated learning (FL) framework allows multiple users to train a global model on the basis of respecting individual data privacy. However, it has not yet been systematically evaluated how robust FL-based NIDSs are against existing privacy attacks under existing defenses. To address this issue, we propose two privacy evaluation metrics designed for FL-based NIDSs, including (1) privacy score that evaluates the similarity between the original and recovered traffic features using reconstruction attacks, and (2) evasion rate against NIDSs using Generative Adversarial Network-based adversarial attack with the reconstructed benign traffic. We conduct experiments to show that existing defenses provide little protection that the corresponding adversarial traffic can even evade the SOTA NIDS Kitsune. To defend against such attacks and build a more robust FL-based NIDS, we further propose FedDef, a novel optimization-based input perturbation defense strategy with theoretical guarantee. It achieves both high utility by minimizing the gradient distance and strong privacy protection by maximizing the input distance. We experimentally evaluate four existing defenses on four datasets and show that our defense outperforms all the baselines in terms of privacy protection with up to 7 times higher privacy score, while maintaining model accuracy loss within 3% under optimal parameter combination.



## **4. CgAT: Center-Guided Adversarial Training for Deep Hashing-Based Retrieval**

cs.CV

has no contributions

**SubmitDate**: 2022-11-09    [abs](http://arxiv.org/abs/2204.10779v3) [paper-pdf](http://arxiv.org/pdf/2204.10779v3)

**Authors**: Xunguang Wang, Yinqun Lin, Xiaomeng Li

**Abstract**: Deep hashing has been extensively utilized in massive image retrieval because of its efficiency and effectiveness. However, deep hashing models are vulnerable to adversarial examples, making it essential to develop adversarial defense methods for image retrieval. Existing solutions achieved limited defense performance because of using weak adversarial samples for training and lacking discriminative optimization objectives to learn robust features. In this paper, we present a min-max based Center-guided Adversarial Training, namely CgAT, to improve the robustness of deep hashing networks through worst adversarial examples. Specifically, we first formulate the center code as a semantically-discriminative representative of the input image content, which preserves the semantic similarity with positive samples and dissimilarity with negative examples. We prove that a mathematical formula can calculate the center code immediately. After obtaining the center codes in each optimization iteration of the deep hashing network, they are adopted to guide the adversarial training process. On the one hand, CgAT generates the worst adversarial examples as augmented data by maximizing the Hamming distance between the hash codes of the adversarial examples and the center codes. On the other hand, CgAT learns to mitigate the effects of adversarial samples by minimizing the Hamming distance to the center codes. Extensive experiments on the benchmark datasets demonstrate the effectiveness of our adversarial training algorithm in defending against adversarial attacks for deep hashing-based retrieval. Compared with the current state-of-the-art defense method, we significantly improve the defense performance by an average of 18.61%, 12.35%, and 11.56% on FLICKR-25K, NUS-WIDE, and MS-COCO, respectively.



## **5. On the Versatile Uses of Partial Distance Correlation in Deep Learning**

cs.CV

This paper has been selected as best paper award for ECCV 2022!

**SubmitDate**: 2022-11-08    [abs](http://arxiv.org/abs/2207.09684v3) [paper-pdf](http://arxiv.org/pdf/2207.09684v3)

**Authors**: Xingjian Zhen, Zihang Meng, Rudrasis Chakraborty, Vikas Singh

**Abstract**: Comparing the functional behavior of neural network models, whether it is a single network over time or two (or more networks) during or post-training, is an essential step in understanding what they are learning (and what they are not), and for identifying strategies for regularization or efficiency improvements. Despite recent progress, e.g., comparing vision transformers to CNNs, systematic comparison of function, especially across different networks, remains difficult and is often carried out layer by layer. Approaches such as canonical correlation analysis (CCA) are applicable in principle, but have been sparingly used so far. In this paper, we revisit a (less widely known) from statistics, called distance correlation (and its partial variant), designed to evaluate correlation between feature spaces of different dimensions. We describe the steps necessary to carry out its deployment for large scale models -- this opens the door to a surprising array of applications ranging from conditioning one deep model w.r.t. another, learning disentangled representations as well as optimizing diverse models that would directly be more robust to adversarial attacks. Our experiments suggest a versatile regularizer (or constraint) with many advantages, which avoids some of the common difficulties one faces in such analyses. Code is at https://github.com/zhenxingjian/Partial_Distance_Correlation.



## **6. NaturalAdversaries: Can Naturalistic Adversaries Be as Effective as Artificial Adversaries?**

cs.CL

Findings of EMNLP 2022

**SubmitDate**: 2022-11-08    [abs](http://arxiv.org/abs/2211.04364v1) [paper-pdf](http://arxiv.org/pdf/2211.04364v1)

**Authors**: Saadia Gabriel, Hamid Palangi, Yejin Choi

**Abstract**: While a substantial body of prior work has explored adversarial example generation for natural language understanding tasks, these examples are often unrealistic and diverge from the real-world data distributions. In this work, we introduce a two-stage adversarial example generation framework (NaturalAdversaries), for designing adversaries that are effective at fooling a given classifier and demonstrate natural-looking failure cases that could plausibly occur during in-the-wild deployment of the models.   At the first stage a token attribution method is used to summarize a given classifier's behaviour as a function of the key tokens in the input. In the second stage a generative model is conditioned on the key tokens from the first stage. NaturalAdversaries is adaptable to both black-box and white-box adversarial attacks based on the level of access to the model parameters. Our results indicate these adversaries generalize across domains, and offer insights for future research on improving robustness of neural text classification models.



## **7. Preserving Semantics in Textual Adversarial Attacks**

cs.CL

8 pages, 4 figures

**SubmitDate**: 2022-11-08    [abs](http://arxiv.org/abs/2211.04205v1) [paper-pdf](http://arxiv.org/pdf/2211.04205v1)

**Authors**: David Herel, Hugo Cisneros, Tomas Mikolov

**Abstract**: Adversarial attacks in NLP challenge the way we look at language models. The goal of this kind of adversarial attack is to modify the input text to fool a classifier while maintaining the original meaning of the text. Although most existing adversarial attacks claim to fulfill the constraint of semantics preservation, careful scrutiny shows otherwise. We show that the problem lies in the text encoders used to determine the similarity of adversarial examples, specifically in the way they are trained. Unsupervised training methods make these encoders more susceptible to problems with antonym recognition. To overcome this, we introduce a simple, fully supervised sentence embedding technique called Semantics-Preserving-Encoder (SPE). The results show that our solution minimizes the variation in the meaning of the adversarial examples generated. It also significantly improves the overall quality of adversarial examples, as confirmed by human evaluators. Furthermore, it can be used as a component in any existing attack to speed up its execution while maintaining similar attack success.



## **8. A Hypergraph-Based Machine Learning Ensemble Network Intrusion Detection System**

cs.CR

12 pages, 10 figures

**SubmitDate**: 2022-11-08    [abs](http://arxiv.org/abs/2211.03933v1) [paper-pdf](http://arxiv.org/pdf/2211.03933v1)

**Authors**: Zong-Zhi Lin, Thomas D. Pike, Mark M. Bailey, Nathaniel D. Bastian

**Abstract**: Network intrusion detection systems (NIDS) to detect malicious attacks continues to meet challenges. NIDS are vulnerable to auto-generated port scan infiltration attempts and NIDS are often developed offline, resulting in a time lag to prevent the spread of infiltration to other parts of a network. To address these challenges, we use hypergraphs to capture evolving patterns of port scan attacks via the set of internet protocol addresses and destination ports, thereby deriving a set of hypergraph-based metrics to train a robust and resilient ensemble machine learning (ML) NIDS that effectively monitors and detects port scanning activities and adversarial intrusions while evolving intelligently in real-time. Through the combination of (1) intrusion examples, (2) NIDS update rules, (3) attack threshold choices to trigger NIDS retraining requests, and (4) production environment with no prior knowledge of the nature of network traffic 40 scenarios were auto-generated to evaluate the ML ensemble NIDS comprising three tree-based models. Results show that under the model settings of an Update-ALL-NIDS rule (namely, retrain and update all the three models upon the same NIDS retraining request) the proposed ML ensemble NIDS produced the best results with nearly 100% detection performance throughout the simulation, exhibiting robustness in the complex dynamics of the simulated cyber-security scenario.



## **9. Are AlphaZero-like Agents Robust to Adversarial Perturbations?**

cs.AI

Accepted by Neurips 2022

**SubmitDate**: 2022-11-07    [abs](http://arxiv.org/abs/2211.03769v1) [paper-pdf](http://arxiv.org/pdf/2211.03769v1)

**Authors**: Li-Cheng Lan, Huan Zhang, Ti-Rong Wu, Meng-Yu Tsai, I-Chen Wu, Cho-Jui Hsieh

**Abstract**: The success of AlphaZero (AZ) has demonstrated that neural-network-based Go AIs can surpass human performance by a large margin. Given that the state space of Go is extremely large and a human player can play the game from any legal state, we ask whether adversarial states exist for Go AIs that may lead them to play surprisingly wrong actions. In this paper, we first extend the concept of adversarial examples to the game of Go: we generate perturbed states that are ``semantically'' equivalent to the original state by adding meaningless moves to the game, and an adversarial state is a perturbed state leading to an undoubtedly inferior action that is obvious even for Go beginners. However, searching the adversarial state is challenging due to the large, discrete, and non-differentiable search space. To tackle this challenge, we develop the first adversarial attack on Go AIs that can efficiently search for adversarial states by strategically reducing the search space. This method can also be extended to other board games such as NoGo. Experimentally, we show that the actions taken by both Policy-Value neural network (PV-NN) and Monte Carlo tree search (MCTS) can be misled by adding one or two meaningless stones; for example, on 58\% of the AlphaGo Zero self-play games, our method can make the widely used KataGo agent with 50 simulations of MCTS plays a losing action by adding two meaningless stones. We additionally evaluated the adversarial examples found by our algorithm with amateur human Go players and 90\% of examples indeed lead the Go agent to play an obviously inferior action. Our code is available at \url{https://PaperCode.cc/GoAttack}.



## **10. Neural Architectural Backdoors**

cs.CR

**SubmitDate**: 2022-11-07    [abs](http://arxiv.org/abs/2210.12179v2) [paper-pdf](http://arxiv.org/pdf/2210.12179v2)

**Authors**: Ren Pang, Changjiang Li, Zhaohan Xi, Shouling Ji, Ting Wang

**Abstract**: This paper asks the intriguing question: is it possible to exploit neural architecture search (NAS) as a new attack vector to launch previously improbable attacks? Specifically, we present EVAS, a new attack that leverages NAS to find neural architectures with inherent backdoors and exploits such vulnerability using input-aware triggers. Compared with existing attacks, EVAS demonstrates many interesting properties: (i) it does not require polluting training data or perturbing model parameters; (ii) it is agnostic to downstream fine-tuning or even re-training from scratch; (iii) it naturally evades defenses that rely on inspecting model parameters or training data. With extensive evaluation on benchmark datasets, we show that EVAS features high evasiveness, transferability, and robustness, thereby expanding the adversary's design spectrum. We further characterize the mechanisms underlying EVAS, which are possibly explainable by architecture-level ``shortcuts'' that recognize trigger patterns. This work raises concerns about the current practice of NAS and points to potential directions to develop effective countermeasures.



## **11. Deviations in Representations Induced by Adversarial Attacks**

cs.LG

**SubmitDate**: 2022-11-07    [abs](http://arxiv.org/abs/2211.03714v1) [paper-pdf](http://arxiv.org/pdf/2211.03714v1)

**Authors**: Daniel Steinberg, Paul Munro

**Abstract**: Deep learning has been a popular topic and has achieved success in many areas. It has drawn the attention of researchers and machine learning practitioners alike, with developed models deployed to a variety of settings. Along with its achievements, research has shown that deep learning models are vulnerable to adversarial attacks. This finding brought about a new direction in research, whereby algorithms were developed to attack and defend vulnerable networks. Our interest is in understanding how these attacks effect change on the intermediate representations of deep learning models. We present a method for measuring and analyzing the deviations in representations induced by adversarial attacks, progressively across a selected set of layers. Experiments are conducted using an assortment of attack algorithms, on the CIFAR-10 dataset, with plots created to visualize the impact of adversarial attacks across different layers in a network.



## **12. Black-Box Attack against GAN-Generated Image Detector with Contrastive Perturbation**

cs.CV

**SubmitDate**: 2022-11-07    [abs](http://arxiv.org/abs/2211.03509v1) [paper-pdf](http://arxiv.org/pdf/2211.03509v1)

**Authors**: Zijie Lou, Gang Cao, Man Lin

**Abstract**: Visually realistic GAN-generated facial images raise obvious concerns on potential misuse. Many effective forensic algorithms have been developed to detect such synthetic images in recent years. It is significant to assess the vulnerability of such forensic detectors against adversarial attacks. In this paper, we propose a new black-box attack method against GAN-generated image detectors. A novel contrastive learning strategy is adopted to train the encoder-decoder network based anti-forensic model under a contrastive loss function. GAN images and their simulated real counterparts are constructed as positive and negative samples, respectively. Leveraging on the trained attack model, imperceptible contrastive perturbation could be applied to input synthetic images for removing GAN fingerprint to some extent. As such, existing GAN-generated image detectors are expected to be deceived. Extensive experimental results verify that the proposed attack effectively reduces the accuracy of three state-of-the-art detectors on six popular GANs. High visual quality of the attacked images is also achieved. The source code will be available at https://github.com/ZXMMD/BAttGAND.



## **13. On the Anonymity of Peer-To-Peer Network Anonymity Schemes Used by Cryptocurrencies**

cs.CR

**SubmitDate**: 2022-11-07    [abs](http://arxiv.org/abs/2201.11860v4) [paper-pdf](http://arxiv.org/pdf/2201.11860v4)

**Authors**: Piyush Kumar Sharma, Devashish Gosain, Claudia Diaz

**Abstract**: Cryptocurrency systems can be subject to deanonimization attacks by exploiting the network-level communication on their peer-to-peer network. Adversaries who control a set of colluding node(s) within the peer-to-peer network can observe transactions being exchanged and infer the parties involved. Thus, various network anonymity schemes have been proposed to mitigate this problem, with some solutions providing theoretical anonymity guarantees.   In this work, we model such peer-to-peer network anonymity solutions and evaluate their anonymity guarantees. To do so, we propose a novel framework that uses Bayesian inference to obtain the probability distributions linking transactions to their possible originators. We characterize transaction anonymity with those distributions, using entropy as metric of adversarial uncertainty on the originator's identity. In particular, we model Dandelion, Dandelion++ and Lightning Network. We study different configurations and demonstrate that none of them offers acceptable anonymity to their users. For instance, our analysis reveals that in the widely deployed Lightning Network, with 1% strategically chosen colluding nodes the adversary can uniquely determine the originator for about 50% of the total transactions in the network. In Dandelion, an adversary that controls 15% of the nodes has on average uncertainty among only 8 possible originators. Moreover, we observe that due to the way Dandelion and Dandelion++ are designed, increasing the network size does not correspond to an increase in the anonymity set of potential originators. Alarmingly, our longitudinal analysis of Lightning Network reveals rather an inverse trend -- with the growth of the network the overall anonymity decreases.



## **14. NIP: Neuron-level Inverse Perturbation Against Adversarial Attacks**

cs.CV

There are some problems in the figure so we need to withdraw this  paper. We will upload the new version after revision

**SubmitDate**: 2022-11-07    [abs](http://arxiv.org/abs/2112.13060v2) [paper-pdf](http://arxiv.org/pdf/2112.13060v2)

**Authors**: Ruoxi Chen, Haibo Jin, Jinyin Chen, Haibin Zheng, Yue Yu, Shouling Ji

**Abstract**: Although deep learning models have achieved unprecedented success, their vulnerabilities towards adversarial attacks have attracted increasing attention, especially when deployed in security-critical domains. To address the challenge, numerous defense strategies, including reactive and proactive ones, have been proposed for robustness improvement. From the perspective of image feature space, some of them cannot reach satisfying results due to the shift of features. Besides, features learned by models are not directly related to classification results. Different from them, We consider defense method essentially from model inside and investigated the neuron behaviors before and after attacks. We observed that attacks mislead the model by dramatically changing the neurons that contribute most and least to the correct label. Motivated by it, we introduce the concept of neuron influence and further divide neurons into front, middle and tail part. Based on it, we propose neuron-level inverse perturbation(NIP), the first neuron-level reactive defense method against adversarial attacks. By strengthening front neurons and weakening those in the tail part, NIP can eliminate nearly all adversarial perturbations while still maintaining high benign accuracy. Besides, it can cope with different sizes of perturbations via adaptivity, especially larger ones. Comprehensive experiments conducted on three datasets and six models show that NIP outperforms the state-of-the-art baselines against eleven adversarial attacks. We further provide interpretable proofs via neuron activation and visualization for better understanding.



## **15. Adversarial Reconfigurable Intelligent Surface Against Physical Layer Key Generation**

eess.SP

**SubmitDate**: 2022-11-07    [abs](http://arxiv.org/abs/2206.10955v2) [paper-pdf](http://arxiv.org/pdf/2206.10955v2)

**Authors**: Zhuangkun Wei, Bin Li, Weisi Guo

**Abstract**: The development of reconfigurable intelligent surfaces (RIS) has recently advanced the research of physical layer security (PLS). Beneficial impacts of RIS include but are not limited to offering a new degree-of-freedom (DoF) for key-less PLS optimization, and increasing channel randomness for physical layer secret key generation (PL-SKG). However, there is a lack of research studying how adversarial RIS can be used to attack and obtain legitimate secret keys generated by PL-SKG. In this work, we show an Eve-controlled adversarial RIS (Eve-RIS), by inserting into the legitimate channel a random and reciprocal channel, can partially reconstruct the secret keys from the legitimate PL-SKG process. To operationalize this concept, we design Eve-RIS schemes against two PL-SKG techniques used: (i) the CSI-based PL-SKG, and (ii) the two-way cross multiplication based PL-SKG. The channel probing at Eve-RIS is realized by compressed sensing designs with a small number of radio-frequency (RF) chains. Then, the optimal RIS phase is obtained by maximizing the Eve-RIS inserted deceiving channel. Our analysis and results show that even with a passive RIS, our proposed Eve-RIS can achieve a high key match rate with legitimate users, and is resistant to most of the current defensive approaches. This means the novel Eve-RIS provides a new eavesdropping threat on PL-SKG, which can spur new research areas to counter adversarial RIS attacks.



## **16. Contrastive Weighted Learning for Near-Infrared Gaze Estimation**

cs.CV

**SubmitDate**: 2022-11-06    [abs](http://arxiv.org/abs/2211.03073v1) [paper-pdf](http://arxiv.org/pdf/2211.03073v1)

**Authors**: Adam Lee

**Abstract**: Appearance-based gaze estimation has been very successful with the use of deep learning. Many following works improved domain generalization for gaze estimation. However, even though there has been much progress in domain generalization for gaze estimation, most of the recent work have been focused on cross-dataset performance -- accounting for different distributions in illuminations, head pose, and lighting. Although improving gaze estimation in different distributions of RGB images is important, near-infrared image based gaze estimation is also critical for gaze estimation in dark settings. Also there are inherent limitations relying solely on supervised learning for regression tasks. This paper contributes to solving these problems and proposes GazeCWL, a novel framework for gaze estimation with near-infrared images using contrastive learning. This leverages adversarial attack techniques for data augmentation and a novel contrastive loss function specifically for regression tasks that effectively clusters the features of different samples in the latent space. Our model outperforms previous domain generalization models in infrared image based gaze estimation and outperforms the baseline by 45.6\% while improving the state-of-the-art by 8.6\%, we demonstrate the efficacy of our method.



## **17. Privacy-Preserving Models for Legal Natural Language Processing**

cs.CL

Camera ready, to appear at the Natural Legal Language Processing  Workshop 2022 co-located with EMNLP

**SubmitDate**: 2022-11-05    [abs](http://arxiv.org/abs/2211.02956v1) [paper-pdf](http://arxiv.org/pdf/2211.02956v1)

**Authors**: Ying Yin, Ivan Habernal

**Abstract**: Pre-training large transformer models with in-domain data improves domain adaptation and helps gain performance on the domain-specific downstream tasks. However, sharing models pre-trained on potentially sensitive data is prone to adversarial privacy attacks. In this paper, we asked to which extent we can guarantee privacy of pre-training data and, at the same time, achieve better downstream performance on legal tasks without the need of additional labeled data. We extensively experiment with scalable self-supervised learning of transformer models under the formal paradigm of differential privacy and show that under specific training configurations we can improve downstream performance without sacrifying privacy protection for the in-domain data. Our main contribution is utilizing differential privacy for large-scale pre-training of transformer language models in the legal NLP domain, which, to the best of our knowledge, has not been addressed before.



## **18. Adversarial Attacks on Transformers-Based Malware Detectors**

cs.CR

Accepted to the 2022 NeurIPS ML Safety Workshop. Code available at  https://github.com/yashjakhotiya/Adversarial-Attacks-On-Transformers

**SubmitDate**: 2022-11-05    [abs](http://arxiv.org/abs/2210.00008v2) [paper-pdf](http://arxiv.org/pdf/2210.00008v2)

**Authors**: Yash Jakhotiya, Heramb Patil, Jugal Rawlani, Dr. Sunil B. Mane

**Abstract**: Signature-based malware detectors have proven to be insufficient as even a small change in malignant executable code can bypass these signature-based detectors. Many machine learning-based models have been proposed to efficiently detect a wide variety of malware. Many of these models are found to be susceptible to adversarial attacks - attacks that work by generating intentionally designed inputs that can force these models to misclassify. Our work aims to explore vulnerabilities in the current state of the art malware detectors to adversarial attacks. We train a Transformers-based malware detector, carry out adversarial attacks resulting in a misclassification rate of 23.9% and propose defenses that reduce this misclassification rate to half. An implementation of our work can be found at https://github.com/yashjakhotiya/Adversarial-Attacks-On-Transformers.



## **19. Physical Adversarial Attack meets Computer Vision: A Decade Survey**

cs.CV

32 pages. Under Review

**SubmitDate**: 2022-11-05    [abs](http://arxiv.org/abs/2209.15179v2) [paper-pdf](http://arxiv.org/pdf/2209.15179v2)

**Authors**: Hui Wei, Hao Tang, Xuemei Jia, Hanxun Yu, Zhubo Li, Zhixiang Wang, Shin'ichi Satoh, Zheng Wang

**Abstract**: Although Deep Neural Networks (DNNs) have achieved impressive results in computer vision, their exposed vulnerability to adversarial attacks remains a serious concern. A series of works has shown that by adding elaborate perturbations to images, DNNs could have catastrophic degradation in performance metrics. And this phenomenon does not only exist in the digital space but also in the physical space. Therefore, estimating the security of these DNNs-based systems is critical for safely deploying them in the real world, especially for security-critical applications, e.g., autonomous cars, video surveillance, and medical diagnosis. In this paper, we focus on physical adversarial attacks and provide a comprehensive survey of over 150 existing papers. We first clarify the concept of the physical adversarial attack and analyze its characteristics. Then, we define the adversarial medium, essential to perform attacks in the physical world. Next, we present the physical adversarial attack methods in task order: classification, detection, and re-identification, and introduce their performance in solving the trilemma: effectiveness, stealthiness, and robustness. In the end, we discuss the current challenges and potential future directions.



## **20. Is RobustBench/AutoAttack a suitable Benchmark for Adversarial Robustness?**

cs.CV

AAAI-22 AdvML Workshop ShortPaper

**SubmitDate**: 2022-11-05    [abs](http://arxiv.org/abs/2112.01601v2) [paper-pdf](http://arxiv.org/pdf/2112.01601v2)

**Authors**: Peter Lorenz, Dominik Strassel, Margret Keuper, Janis Keuper

**Abstract**: Recently, RobustBench (Croce et al. 2020) has become a widely recognized benchmark for the adversarial robustness of image classification networks. In its most commonly reported sub-task, RobustBench evaluates and ranks the adversarial robustness of trained neural networks on CIFAR10 under AutoAttack (Croce and Hein 2020b) with l-inf perturbations limited to eps = 8/255. With leading scores of the currently best performing models of around 60% of the baseline, it is fair to characterize this benchmark to be quite challenging. Despite its general acceptance in recent literature, we aim to foster discussion about the suitability of RobustBench as a key indicator for robustness which could be generalized to practical applications. Our line of argumentation against this is two-fold and supported by excessive experiments presented in this paper: We argue that I) the alternation of data by AutoAttack with l-inf, eps = 8/255 is unrealistically strong, resulting in close to perfect detection rates of adversarial samples even by simple detection algorithms and human observers. We also show that other attack methods are much harder to detect while achieving similar success rates. II) That results on low-resolution data sets like CIFAR10 do not generalize well to higher resolution images as gradient-based attacks appear to become even more detectable with increasing resolutions.



## **21. Stateful Detection of Adversarial Reprogramming**

cs.GT

**SubmitDate**: 2022-11-05    [abs](http://arxiv.org/abs/2211.02885v1) [paper-pdf](http://arxiv.org/pdf/2211.02885v1)

**Authors**: Yang Zheng, Xiaoyi Feng, Zhaoqiang Xia, Xiaoyue Jiang, Maura Pintor, Ambra Demontis, Battista Biggio, Fabio Roli

**Abstract**: Adversarial reprogramming allows stealing computational resources by repurposing machine learning models to perform a different task chosen by the attacker. For example, a model trained to recognize images of animals can be reprogrammed to recognize medical images by embedding an adversarial program in the images provided as inputs. This attack can be perpetrated even if the target model is a black box, supposed that the machine-learning model is provided as a service and the attacker can query the model and collect its outputs. So far, no defense has been demonstrated effective in this scenario. We show for the first time that this attack is detectable using stateful defenses, which store the queries made to the classifier and detect the abnormal cases in which they are similar. Once a malicious query is detected, the account of the user who made it can be blocked. Thus, the attacker must create many accounts to perpetrate the attack. To decrease this number, the attacker could create the adversarial program against a surrogate classifier and then fine-tune it by making few queries to the target model. In this scenario, the effectiveness of the stateful defense is reduced, but we show that it is still effective.



## **22. Textual Manifold-based Defense Against Natural Language Adversarial Examples**

cs.CL

**SubmitDate**: 2022-11-05    [abs](http://arxiv.org/abs/2211.02878v1) [paper-pdf](http://arxiv.org/pdf/2211.02878v1)

**Authors**: Dang Minh Nguyen, Luu Anh Tuan

**Abstract**: Recent studies on adversarial images have shown that they tend to leave the underlying low-dimensional data manifold, making them significantly more challenging for current models to make correct predictions. This so-called off-manifold conjecture has inspired a novel line of defenses against adversarial attacks on images. In this study, we find a similar phenomenon occurs in the contextualized embedding space induced by pretrained language models, in which adversarial texts tend to have their embeddings diverge from the manifold of natural ones. Based on this finding, we propose Textual Manifold-based Defense (TMD), a defense mechanism that projects text embeddings onto an approximated embedding manifold before classification. It reduces the complexity of potential adversarial examples, which ultimately enhances the robustness of the protected model. Through extensive experiments, our method consistently and significantly outperforms previous defenses under various attack settings without trading off clean accuracy. To the best of our knowledge, this is the first NLP defense that leverages the manifold structure against adversarial attacks. Our code is available at \url{https://github.com/dangne/tmd}.



## **23. A Resource Allocation Scheme for Energy Demand Management in 6G-enabled Smart Grid**

cs.NI

2023 North American Innovative Smart Grid Technologies Conference

**SubmitDate**: 2022-11-05    [abs](http://arxiv.org/abs/2207.00154v2) [paper-pdf](http://arxiv.org/pdf/2207.00154v2)

**Authors**: Shafkat Islam, Ioannis Zografopoulos, Md Tamjid Hossain, Shahriar Badsha, Charalambos Konstantinou

**Abstract**: Smart grid (SG) systems enhance grid resilience and efficient operation, leveraging the bidirectional flow of energy and information between generation facilities and prosumers. For energy demand management (EDM), the SG network requires computing a large amount of data generated by massive Internet-of-things sensors and advanced metering infrastructure (AMI) with minimal latency. This paper proposes a deep reinforcement learning (DRL)-based resource allocation scheme in a 6G-enabled SG edge network to offload resource-consuming EDM computation to edge servers. Automatic resource provisioning is achieved by harnessing the computational capabilities of smart meters in the dynamic edge network. To enforce DRL-assisted policies in dense 6G networks, the state information from multiple edge servers is required. However, adversaries can "poison" such information through false state injection (FSI) attacks, exhausting SG edge computing resources. Toward addressing this issue, we investigate the impact of such FSI attacks with respect to abusive utilization of edge resources, and develop a lightweight FSI detection mechanism based on supervised classifiers. Simulation results demonstrate the efficacy of DRL in dynamic resource allocation, the impact of the FSI attacks, and the effectiveness of the detection technique.



## **24. On Trace of PGD-Like Adversarial Attacks**

cs.CV

**SubmitDate**: 2022-11-05    [abs](http://arxiv.org/abs/2205.09586v2) [paper-pdf](http://arxiv.org/pdf/2205.09586v2)

**Authors**: Mo Zhou, Vishal M. Patel

**Abstract**: Adversarial attacks pose safety and security concerns to deep learning applications, but their characteristics are under-explored. Yet largely imperceptible, a strong trace could have been left by PGD-like attacks in an adversarial example. Recall that PGD-like attacks trigger the ``local linearity'' of a network, which implies different extents of linearity for benign or adversarial examples. Inspired by this, we construct an Adversarial Response Characteristics (ARC) feature to reflect the model's gradient consistency around the input to indicate the extent of linearity. Under certain conditions, it qualitatively shows a gradually varying pattern from benign example to adversarial example, as the latter leads to Sequel Attack Effect (SAE). To quantitatively evaluate the effectiveness of ARC, we conduct experiments on CIFAR-10 and ImageNet for attack detection and attack type recognition in a challenging setting. The results suggest that SAE is an effective and unique trace of PGD-like attacks reflected through the ARC feature. The ARC feature is intuitive, light-weighted, non-intrusive, and data-undemanding.



## **25. Fairness-aware Regression Robust to Adversarial Attacks**

cs.CR

**SubmitDate**: 2022-11-04    [abs](http://arxiv.org/abs/2211.04449v1) [paper-pdf](http://arxiv.org/pdf/2211.04449v1)

**Authors**: Yulu Jin, Lifeng Lai

**Abstract**: In this paper, we take a first step towards answering the question of how to design fair machine learning algorithms that are robust to adversarial attacks. Using a minimax framework, we aim to design an adversarially robust fair regression model that achieves optimal performance in the presence of an attacker who is able to add a carefully designed adversarial data point to the dataset or perform a rank-one attack on the dataset. By solving the proposed nonsmooth nonconvex-nonconcave minimax problem, the optimal adversary as well as the robust fairness-aware regression model are obtained. For both synthetic data and real-world datasets, numerical results illustrate that the proposed adversarially robust fair models have better performance on poisoned datasets than other fair machine learning models in both prediction accuracy and group-based fairness measure.



## **26. Improving Adversarial Robustness to Sensitivity and Invariance Attacks with Deep Metric Learning**

cs.LG

v1

**SubmitDate**: 2022-11-04    [abs](http://arxiv.org/abs/2211.02468v1) [paper-pdf](http://arxiv.org/pdf/2211.02468v1)

**Authors**: Anaelia Ovalle, Evan Czyzycki, Cho-Jui Hsieh

**Abstract**: Intentionally crafted adversarial samples have effectively exploited weaknesses in deep neural networks. A standard method in adversarial robustness assumes a framework to defend against samples crafted by minimally perturbing a sample such that its corresponding model output changes. These sensitivity attacks exploit the model's sensitivity toward task-irrelevant features. Another form of adversarial sample can be crafted via invariance attacks, which exploit the model underestimating the importance of relevant features. Previous literature has indicated a tradeoff in defending against both attack types within a strictly L_p bounded defense. To promote robustness toward both types of attacks beyond Euclidean distance metrics, we use metric learning to frame adversarial regularization as an optimal transport problem. Our preliminary results indicate that regularizing over invariant perturbations in our framework improves both invariant and sensitivity defense.



## **27. Rickrolling the Artist: Injecting Invisible Backdoors into Text-Guided Image Generation Models**

cs.LG

25 pages, 16 figures, 5 tables

**SubmitDate**: 2022-11-04    [abs](http://arxiv.org/abs/2211.02408v1) [paper-pdf](http://arxiv.org/pdf/2211.02408v1)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Kristian Kersting

**Abstract**: While text-to-image synthesis currently enjoys great popularity among researchers and the general public, the security of these models has been neglected so far. Many text-guided image generation models rely on pre-trained text encoders from external sources, and their users trust that the retrieved models will behave as promised. Unfortunately, this might not be the case. We introduce backdoor attacks against text-guided generative models and demonstrate that their text encoders pose a major tampering risk. Our attacks only slightly alter an encoder so that no suspicious model behavior is apparent for image generations with clean prompts. By then inserting a single non-Latin character into the prompt, the adversary can trigger the model to either generate images with pre-defined attributes or images following a hidden, potentially malicious description. We empirically demonstrate the high effectiveness of our attacks on Stable Diffusion and highlight that the injection process of a single backdoor takes less than two minutes. Besides phrasing our approach solely as an attack, it can also force an encoder to forget phrases related to certain concepts, such as nudity or violence, and help to make image generation safer.



## **28. Logits are predictive of network type**

cs.CV

**SubmitDate**: 2022-11-04    [abs](http://arxiv.org/abs/2211.02272v1) [paper-pdf](http://arxiv.org/pdf/2211.02272v1)

**Authors**: Ali Borji

**Abstract**: We show that it is possible to predict which deep network has generated a given logit vector with accuracy well above chance. We utilize a number of networks on a dataset, initialized with random weights or pretrained weights, as well as fine-tuned networks. A classifier is then trained on the logit vectors of the trained set of this dataset to map the logit vector to the network index that has generated it. The classifier is then evaluated on the test set of the dataset. Results are better with randomly initialized networks, but also generalize to pretrained networks as well as fine-tuned ones. Classification accuracy is higher using unnormalized logits than normalized ones. We find that there is little transfer when applying a classifier to the same networks but with different sets of weights. In addition to help better understand deep networks and the way they encode uncertainty, we anticipate our finding to be useful in some applications (e.g. tailoring an adversarial attack for a certain type of network). Code is available at https://github.com/aliborji/logits.



## **29. Quantum Man-in-the-middle Attacks: a Game-theoretic Approach with Applications to Radars**

eess.SP

**SubmitDate**: 2022-11-04    [abs](http://arxiv.org/abs/2211.02228v1) [paper-pdf](http://arxiv.org/pdf/2211.02228v1)

**Authors**: YInan Hu, Quanyan Zhu

**Abstract**: The detection and discrimination of quantum states serve a crucial role in quantum signal processing, a discipline that studies methods and techniques to process signals that obey the quantum mechanics frameworks. However, just like classical detection, evasive behaviors also exist in quantum detection. In this paper, we formulate an adversarial quantum detection scenario where the detector is passive and does not know the quantum states have been distorted by an attacker. We compare the performance of a passive detector with the one of a non-adversarial detector to demonstrate how evasive behaviors can undermine the performance of quantum detection. We use a case study of target detection with quantum radars to corroborate our analytical results.



## **30. Adversarial Defense via Neural Oscillation inspired Gradient Masking**

cs.LG

**SubmitDate**: 2022-11-04    [abs](http://arxiv.org/abs/2211.02223v1) [paper-pdf](http://arxiv.org/pdf/2211.02223v1)

**Authors**: Chunming Jiang, Yilei Zhang

**Abstract**: Spiking neural networks (SNNs) attract great attention due to their low power consumption, low latency, and biological plausibility. As they are widely deployed in neuromorphic devices for low-power brain-inspired computing, security issues become increasingly important. However, compared to deep neural networks (DNNs), SNNs currently lack specifically designed defense methods against adversarial attacks. Inspired by neural membrane potential oscillation, we propose a novel neural model that incorporates the bio-inspired oscillation mechanism to enhance the security of SNNs. Our experiments show that SNNs with neural oscillation neurons have better resistance to adversarial attacks than ordinary SNNs with LIF neurons on kinds of architectures and datasets. Furthermore, we propose a defense method that changes model's gradients by replacing the form of oscillation, which hides the original training gradients and confuses the attacker into using gradients of 'fake' neurons to generate invalid adversarial samples. Our experiments suggest that the proposed defense method can effectively resist both single-step and iterative attacks with comparable defense effectiveness and much less computational costs than adversarial training methods on DNNs. To the best of our knowledge, this is the first work that establishes adversarial defense through masking surrogate gradients on SNNs.



## **31. Fault-Tolerant Federated Reinforcement Learning with Theoretical Guarantee**

cs.LG

Published at NeurIPS 2021. Extended version with proofs and  additional experimental details and results. New version changes: reduced  file size of figures; added a diagram illustrating the problem setting; added  link to code on GitHub; modified proof for Theorem 6 (highlighted in red)

**SubmitDate**: 2022-11-03    [abs](http://arxiv.org/abs/2110.14074v2) [paper-pdf](http://arxiv.org/pdf/2110.14074v2)

**Authors**: Flint Xiaofeng Fan, Yining Ma, Zhongxiang Dai, Wei Jing, Cheston Tan, Bryan Kian Hsiang Low

**Abstract**: The growing literature of Federated Learning (FL) has recently inspired Federated Reinforcement Learning (FRL) to encourage multiple agents to federatively build a better decision-making policy without sharing raw trajectories. Despite its promising applications, existing works on FRL fail to I) provide theoretical analysis on its convergence, and II) account for random system failures and adversarial attacks. Towards this end, we propose the first FRL framework the convergence of which is guaranteed and tolerant to less than half of the participating agents being random system failures or adversarial attackers. We prove that the sample efficiency of the proposed framework is guaranteed to improve with the number of agents and is able to account for such potential failures or attacks. All theoretical results are empirically verified on various RL benchmark tasks.



## **32. Clean-label Backdoor Attack against Deep Hashing based Retrieval**

cs.CV

**SubmitDate**: 2022-11-03    [abs](http://arxiv.org/abs/2109.08868v2) [paper-pdf](http://arxiv.org/pdf/2109.08868v2)

**Authors**: Kuofeng Gao, Jiawang Bai, Bin Chen, Dongxian Wu, Shu-Tao Xia

**Abstract**: Deep hashing has become a popular method in large-scale image retrieval due to its computational and storage efficiency. However, recent works raise the security concerns of deep hashing. Although existing works focus on the vulnerability of deep hashing in terms of adversarial perturbations, we identify a more pressing threat, backdoor attack, when the attacker has access to the training data. A backdoored deep hashing model behaves normally on original query images, while returning the images with the target label when the trigger presents, which makes the attack hard to be detected. In this paper, we uncover this security concern by utilizing clean-label data poisoning. To the best of our knowledge, this is the first attempt at the backdoor attack against deep hashing models. To craft the poisoned images, we first generate the targeted adversarial patch as the backdoor trigger. Furthermore, we propose the confusing perturbations to disturb the hashing code learning, such that the hashing model can learn more about the trigger. The confusing perturbations are imperceptible and generated by dispersing the images with the target label in the Hamming space. We have conducted extensive experiments to verify the efficacy of our backdoor attack under various settings. For instance, it can achieve 63% targeted mean average precision on ImageNet under 48 bits code length with only 40 poisoned images.



## **33. Physically Adversarial Attacks and Defenses in Computer Vision: A Survey**

cs.CV

**SubmitDate**: 2022-11-03    [abs](http://arxiv.org/abs/2211.01671v1) [paper-pdf](http://arxiv.org/pdf/2211.01671v1)

**Authors**: Xingxing Wei, Bangzheng Pu, Jiefan Lu, Baoyuan Wu

**Abstract**: Although Deep Neural Networks (DNNs) have been widely applied in various real-world scenarios, they are vulnerable to adversarial examples. The current adversarial attacks in computer vision can be divided into digital attacks and physical attacks according to their different attack forms. Compared with digital attacks, which generate perturbations in the digital pixels, physical attacks are more practical in the real world. Owing to the serious security problem caused by physically adversarial examples, many works have been proposed to evaluate the physically adversarial robustness of DNNs in the past years. In this paper, we summarize a survey versus the current physically adversarial attacks and physically adversarial defenses in computer vision. To establish a taxonomy, we organize the current physical attacks from attack tasks, attack forms, and attack methods, respectively. Thus, readers can have a systematic knowledge about this topic from different aspects. For the physical defenses, we establish the taxonomy from pre-processing, in-processing, and post-processing for the DNN models to achieve a full coverage of the adversarial defenses. Based on the above survey, we finally discuss the challenges of this research field and further outlook the future direction.



## **34. Enhancing Transferability of Adversarial Examples with Spatial Momentum**

cs.CV

Accepted as Oral by 5-th Chinese Conference on Pattern Recognition  and Computer Vision, PRCV 2022

**SubmitDate**: 2022-11-03    [abs](http://arxiv.org/abs/2203.13479v2) [paper-pdf](http://arxiv.org/pdf/2203.13479v2)

**Authors**: Guoqiu Wang, Huanqian Yan, Xingxing Wei

**Abstract**: Many adversarial attack methods achieve satisfactory attack success rates under the white-box setting, but they usually show poor transferability when attacking other DNN models. Momentum-based attack is one effective method to improve transferability. It integrates the momentum term into the iterative process, which can stabilize the update directions by adding the gradients' temporal correlation for each pixel. We argue that only this temporal momentum is not enough, the gradients from the spatial domain within an image, i.e. gradients from the context pixels centered on the target pixel are also important to the stabilization. For that, we propose a novel method named Spatial Momentum Iterative FGSM attack (SMI-FGSM), which introduces the mechanism of momentum accumulation from temporal domain to spatial domain by considering the context information from different regions within the image. SMI-FGSM is then integrated with temporal momentum to simultaneously stabilize the gradients' update direction from both the temporal and spatial domains. Extensive experiments show that our method indeed further enhances adversarial transferability. It achieves the best transferability success rate for multiple mainstream undefended and defended models, which outperforms the state-of-the-art attack methods by a large margin of 10\% on average.



## **35. Leveraging Domain Features for Detecting Adversarial Attacks Against Deep Speech Recognition in Noise**

eess.AS

**SubmitDate**: 2022-11-03    [abs](http://arxiv.org/abs/2211.01621v1) [paper-pdf](http://arxiv.org/pdf/2211.01621v1)

**Authors**: Christian Heider Nielsen, Zheng-Hua Tan

**Abstract**: In recent years, significant progress has been made in deep model-based automatic speech recognition (ASR), leading to its widespread deployment in the real world. At the same time, adversarial attacks against deep ASR systems are highly successful. Various methods have been proposed to defend ASR systems from these attacks. However, existing classification based methods focus on the design of deep learning models while lacking exploration of domain specific features. This work leverages filter bank-based features to better capture the characteristics of attacks for improved detection. Furthermore, the paper analyses the potentials of using speech and non-speech parts separately in detecting adversarial attacks. In the end, considering adverse environments where ASR systems may be deployed, we study the impact of acoustic noise of various types and signal-to-noise ratios. Extensive experiments show that the inverse filter bank features generally perform better in both clean and noisy environments, the detection is effective using either speech or non-speech part, and the acoustic noise can largely degrade the detection performance.



## **36. Robust Few-shot Learning Without Using any Adversarial Samples**

cs.CV

TNNLS Submission (Under Review)

**SubmitDate**: 2022-11-03    [abs](http://arxiv.org/abs/2211.01598v1) [paper-pdf](http://arxiv.org/pdf/2211.01598v1)

**Authors**: Gaurav Kumar Nayak, Ruchit Rawal, Inder Khatri, Anirban Chakraborty

**Abstract**: The high cost of acquiring and annotating samples has made the `few-shot' learning problem of prime importance. Existing works mainly focus on improving performance on clean data and overlook robustness concerns on the data perturbed with adversarial noise. Recently, a few efforts have been made to combine the few-shot problem with the robustness objective using sophisticated Meta-Learning techniques. These methods rely on the generation of adversarial samples in every episode of training, which further adds a computational burden. To avoid such time-consuming and complicated procedures, we propose a simple but effective alternative that does not require any adversarial samples. Inspired by the cognitive decision-making process in humans, we enforce high-level feature matching between the base class data and their corresponding low-frequency samples in the pretraining stage via self distillation. The model is then fine-tuned on the samples of novel classes where we additionally improve the discriminability of low-frequency query set features via cosine similarity. On a 1-shot setting of the CIFAR-FS dataset, our method yields a massive improvement of $60.55\%$ & $62.05\%$ in adversarial accuracy on the PGD and state-of-the-art Auto Attack, respectively, with a minor drop in clean accuracy compared to the baseline. Moreover, our method only takes $1.69\times$ of the standard training time while being $\approx$ $5\times$ faster than state-of-the-art adversarial meta-learning methods. The code is available at https://github.com/vcl-iisc/robust-few-shot-learning.



## **37. Data-free Defense of Black Box Models Against Adversarial Attacks**

cs.LG

TIFS Submission (Under Review)

**SubmitDate**: 2022-11-03    [abs](http://arxiv.org/abs/2211.01579v1) [paper-pdf](http://arxiv.org/pdf/2211.01579v1)

**Authors**: Gaurav Kumar Nayak, Inder Khatri, Shubham Randive, Ruchit Rawal, Anirban Chakraborty

**Abstract**: Several companies often safeguard their trained deep models (i.e. details of architecture, learnt weights, training details etc.) from third-party users by exposing them only as black boxes through APIs. Moreover, they may not even provide access to the training data due to proprietary reasons or sensitivity concerns. We make the first attempt to provide adversarial robustness to the black box models in a data-free set up. We construct synthetic data via generative model and train surrogate network using model stealing techniques. To minimize adversarial contamination on perturbed samples, we propose `wavelet noise remover' (WNR) that performs discrete wavelet decomposition on input images and carefully select only a few important coefficients determined by our `wavelet coefficient selection module' (WCSM). To recover the high-frequency content of the image after noise removal via WNR, we further train a `regenerator' network with an objective to retrieve the coefficients such that the reconstructed image yields similar to original predictions on the surrogate model. At test time, WNR combined with trained regenerator network is prepended to the black box network, resulting in a high boost in adversarial accuracy. Our method improves the adversarial accuracy on CIFAR-10 by 38.98% and 32.01% on state-of-the-art Auto Attack compared to baseline, even when the attacker uses surrogate architecture (Alexnet-half and Alexnet) similar to the black box architecture (Alexnet) with same model stealing strategy as defender. The code is available at https://github.com/vcl-iisc/data-free-black-box-defense



## **38. The Impostor Among US(B): Off-Path Injection Attacks on USB Communications**

cs.CR

To appear in USENIX Security 2023

**SubmitDate**: 2022-11-03    [abs](http://arxiv.org/abs/2211.01109v2) [paper-pdf](http://arxiv.org/pdf/2211.01109v2)

**Authors**: Robert Dumitru, Daniel Genkin, Andrew Wabnitz, Yuval Yarom

**Abstract**: USB is the most prevalent peripheral interface in modern computer systems and its inherent insecurities make it an appealing attack vector. A well-known limitation of USB is that traffic is not encrypted. This allows on-path adversaries to trivially perform man-in-the-middle attacks. Off-path attacks that compromise the confidentiality of communications have also been shown to be possible. However, so far no off-path attacks that breach USB communications integrity have been demonstrated.   In this work we show that the integrity of USB communications is not guaranteed even against off-path attackers.Specifically, we design and build malicious devices that, even when placed outside of the path between a victim device and the host, can inject data to that path. Using our developed injectors we can falsify the provenance of data input as interpreted by a host computer system. By injecting on behalf of trusted victim devices we can circumvent any software-based authorisation policy defences that computer systems employ against common USB attacks. We demonstrate two concrete attacks. The first injects keystrokes allowing an attacker to execute commands. The second demonstrates file-contents replacement including during system install from a USB disk. We test the attacks on 29 USB 2.0 and USB 3.x hubs and find 14 of them to be vulnerable.



## **39. On the Adversarial Robustness of Vision Transformers**

cs.CV

Published in Transactions on Machine Learning Research (TMLR). Codes  available at  https://github.com/RulinShao/on-the-adversarial-robustness-of-visual-transformer

**SubmitDate**: 2022-11-02    [abs](http://arxiv.org/abs/2103.15670v3) [paper-pdf](http://arxiv.org/pdf/2103.15670v3)

**Authors**: Rulin Shao, Zhouxing Shi, Jinfeng Yi, Pin-Yu Chen, Cho-Jui Hsieh

**Abstract**: Following the success in advancing natural language processing and understanding, transformers are expected to bring revolutionary changes to computer vision. This work provides a comprehensive study on the robustness of vision transformers (ViTs) against adversarial perturbations. Tested on various white-box and transfer attack settings, we find that ViTs possess better adversarial robustness when compared with MLP-Mixer and convolutional neural networks (CNNs) including ConvNeXt, and this observation also holds for certified robustness. Through frequency analysis and feature visualization, we summarize the following main observations contributing to the improved robustness of ViTs: 1) Features learned by ViTs contain less high-frequency patterns that have spurious correlation, which helps explain why ViTs are less sensitive to high-frequency perturbations than CNNs and MLP-Mixer, and there is a high correlation between how much the model learns high-frequency features and its robustness against different frequency-based perturbations. 2) Introducing convolutional or tokens-to-token blocks for learning high-frequency features in ViTs can improve classification accuracy but at the cost of adversarial robustness. 3) Modern CNN designs that borrow techniques from ViTs including activation function, layer norm, larger kernel size to imitate the global attention, and patchify the images as inputs, etc., could help bridge the performance gap between ViTs and CNNs not only in terms of performance, but also certified and empirical adversarial robustness. Moreover, we show adversarial training is also applicable to ViT for training robust models, and sharpness-aware minimization can also help improve robustness, while pre-training with clean images on larger datasets does not significantly improve adversarial robustness.



## **40. Distributed Black-box Attack against Image Classification Cloud Services**

cs.LG

10 pages, 11 figures

**SubmitDate**: 2022-11-02    [abs](http://arxiv.org/abs/2210.16371v2) [paper-pdf](http://arxiv.org/pdf/2210.16371v2)

**Authors**: Han Wu, Sareh Rowlands, Johan Wahlstrom

**Abstract**: Black-box adversarial attacks can fool image classifiers into misclassifying images without requiring access to model structure and weights. Recently proposed black-box attacks can achieve a success rate of more than 95% after less than 1,000 queries. The question then arises of whether black-box attacks have become a real threat against IoT devices that rely on cloud APIs to achieve image classification. To shed some light on this, note that prior research has primarily focused on increasing the success rate and reducing the number of required queries. However, another crucial factor for black-box attacks against cloud APIs is the time required to perform the attack. This paper applies black-box attacks directly to cloud APIs rather than to local models, thereby avoiding multiple mistakes made in prior research. Further, we exploit load balancing to enable distributed black-box attacks that can reduce the attack time by a factor of about five for both local search and gradient estimation methods.



## **41. Isometric Representations in Neural Networks Improve Robustness**

cs.LG

14 pages, 4 figures

**SubmitDate**: 2022-11-02    [abs](http://arxiv.org/abs/2211.01236v1) [paper-pdf](http://arxiv.org/pdf/2211.01236v1)

**Authors**: Kosio Beshkov, Jonas Verhellen, Mikkel Elle Lepperød

**Abstract**: Artificial and biological agents cannon learn given completely random and unstructured data. The structure of data is encoded in the metric relationships between data points. In the context of neural networks, neuronal activity within a layer forms a representation reflecting the transformation that the layer implements on its inputs. In order to utilize the structure in the data in a truthful manner, such representations should reflect the input distances and thus be continuous and isometric. Supporting this statement, recent findings in neuroscience propose that generalization and robustness are tied to neural representations being continuously differentiable. In machine learning, most algorithms lack robustness and are generally thought to rely on aspects of the data that differ from those that humans use, as is commonly seen in adversarial attacks. During cross-entropy classification, the metric and structural properties of network representations are usually broken both between and within classes. This side effect from training can lead to instabilities under perturbations near locations where such structure is not preserved. One of the standard solutions to obtain robustness is to add ad hoc regularization terms, but to our knowledge, forcing representations to preserve the metric structure of the input data as a stabilising mechanism has not yet been studied. In this work, we train neural networks to perform classification while simultaneously maintaining within-class metric structure, leading to isometric within-class representations. Such network representations turn out to be beneficial for accurate and robust inference. By stacking layers with this property we create a network architecture that facilitates hierarchical manipulation of internal neural representations. Finally, we verify that isometric regularization improves the robustness to adversarial attacks on MNIST.



## **42. BATT: Backdoor Attack with Transformation-based Triggers**

cs.CR

5 pages

**SubmitDate**: 2022-11-02    [abs](http://arxiv.org/abs/2211.01806v1) [paper-pdf](http://arxiv.org/pdf/2211.01806v1)

**Authors**: Tong Xu, Yiming Li, Yong Jiang, Shu-Tao Xia

**Abstract**: Deep neural networks (DNNs) are vulnerable to backdoor attacks. The backdoor adversaries intend to maliciously control the predictions of attacked DNNs by injecting hidden backdoors that can be activated by adversary-specified trigger patterns during the training process. One recent research revealed that most of the existing attacks failed in the real physical world since the trigger contained in the digitized test samples may be different from that of the one used for training. Accordingly, users can adopt spatial transformations as the image pre-processing to deactivate hidden backdoors. In this paper, we explore the previous findings from another side. We exploit classical spatial transformations (i.e. rotation and translation) with the specific parameter as trigger patterns to design a simple yet effective poisoning-based backdoor attack. For example, only images rotated to a particular angle can activate the embedded backdoor of attacked DNNs. Extensive experiments are conducted, verifying the effectiveness of our attack under both digital and physical settings and its resistance to existing backdoor defenses.



## **43. Driver Locations Harvesting Attack on pRide**

cs.CR

**SubmitDate**: 2022-11-02    [abs](http://arxiv.org/abs/2210.13263v2) [paper-pdf](http://arxiv.org/pdf/2210.13263v2)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstract**: Privacy preservation in Ride-Hailing Services (RHS) is intended to protect privacy of drivers and riders. pRide, published in IEEE Trans. Vehicular Technology 2021, is a prediction based privacy-preserving RHS protocol to match riders with an optimum driver. In the protocol, the Service Provider (SP) homomorphically computes Euclidean distances between encrypted locations of drivers and rider. Rider selects an optimum driver using decrypted distances augmented by a new-ride-emergence prediction. To improve the effectiveness of driver selection, the paper proposes an enhanced version where each driver gives encrypted distances to each corner of her grid. To thwart a rider from using these distances to launch an inference attack, the SP blinds these distances before sharing them with the rider. In this work, we propose a passive attack where an honest-but-curious adversary rider who makes a single ride request and receives the blinded distances from SP can recover the constants used to blind the distances. Using the unblinded distances, rider to driver distance and Google Nearest Road API, the adversary can obtain the precise locations of responding drivers. We conduct experiments with random on-road driver locations for four different cities. Our experiments show that we can determine the precise locations of at least 80% of the drivers participating in the enhanced pRide protocol.



## **44. Defending with Errors: Approximate Computing for Robustness of Deep Neural Networks**

cs.CR

arXiv admin note: substantial text overlap with arXiv:2006.07700

**SubmitDate**: 2022-11-02    [abs](http://arxiv.org/abs/2211.01182v1) [paper-pdf](http://arxiv.org/pdf/2211.01182v1)

**Authors**: Amira Guesmi, Ihsen Alouani, Khaled N. Khasawneh, Mouna Baklouti, Tarek Frikha, Mohamed Abid, Nael Abu-Ghazaleh

**Abstract**: Machine-learning architectures, such as Convolutional Neural Networks (CNNs) are vulnerable to adversarial attacks: inputs crafted carefully to force the system output to a wrong label. Since machine-learning is being deployed in safety-critical and security-sensitive domains, such attacks may have catastrophic security and safety consequences. In this paper, we propose for the first time to use hardware-supported approximate computing to improve the robustness of machine-learning classifiers. We show that successful adversarial attacks against the exact classifier have poor transferability to the approximate implementation. Surprisingly, the robustness advantages also apply to white-box attacks where the attacker has unrestricted access to the approximate classifier implementation: in this case, we show that substantially higher levels of adversarial noise are needed to produce adversarial examples. Furthermore, our approximate computing model maintains the same level in terms of classification accuracy, does not require retraining, and reduces resource utilization and energy consumption of the CNN. We conducted extensive experiments on a set of strong adversarial attacks; We empirically show that the proposed implementation increases the robustness of a LeNet-5, Alexnet and VGG-11 CNNs considerably with up to 50% by-product saving in energy consumption due to the simpler nature of the approximate logic.



## **45. a-RNA: Adversarial Radio Noise Attack to Fool Radar-based Environment Perception Systems**

cs.CR

**SubmitDate**: 2022-11-02    [abs](http://arxiv.org/abs/2211.01112v1) [paper-pdf](http://arxiv.org/pdf/2211.01112v1)

**Authors**: Amira Guesmi, Ihsen Alouani

**Abstract**: Due to their robustness to degraded capturing conditions, radars are widely used for environment perception, which is a critical task in applications like autonomous vehicles. More specifically, Ultra-Wide Band (UWB) radars are particularly efficient for short range settings as they carry rich information on the environment. Recent UWB-based systems rely on Machine Learning (ML) to exploit the rich signature of these sensors. However, ML classifiers are susceptible to adversarial examples, which are created from raw data to fool the classifier such that it assigns the input to the wrong class. These attacks represent a serious threat to systems integrity, especially for safety-critical applications. In this work, we present a new adversarial attack on UWB radars in which an adversary injects adversarial radio noise in the wireless channel to cause an obstacle recognition failure. First, based on signals collected in real-life environment, we show that conventional attacks fail to generate robust noise under realistic conditions. We propose a-RNA, i.e., Adversarial Radio Noise Attack to overcome these issues. Specifically, a-RNA generates an adversarial noise that is efficient without synchronization between the input signal and the noise. Moreover, a-RNA generated noise is, by-design, robust against pre-processing countermeasures such as filtering-based defenses. Moreover, in addition to the undetectability objective by limiting the noise magnitude budget, a-RNA is also efficient in the presence of sophisticated defenses in the spectral domain by introducing a frequency budget. We believe this work should alert about potentially critical implementations of adversarial attacks on radar systems that should be taken seriously.



## **46. Improving transferability of 3D adversarial attacks with scale and shear transformations**

cs.CV

10 pages

**SubmitDate**: 2022-11-02    [abs](http://arxiv.org/abs/2211.01093v1) [paper-pdf](http://arxiv.org/pdf/2211.01093v1)

**Authors**: Jinali Zhang, Yinpeng Dong, Jun Zhu, Jihong Zhu, Minchi Kuang, Xiaming Yuan

**Abstract**: Previous work has shown that 3D point cloud classifiers can be vulnerable to adversarial examples. However, most of the existing methods are aimed at white-box attacks, where the parameters and other information of the classifiers are known in the attack, which is unrealistic for real-world applications. In order to improve the attack performance of the black-box classifiers, the research community generally uses the transfer-based black-box attack. However, the transferability of current 3D attacks is still relatively low. To this end, this paper proposes Scale and Shear (SS) Attack to generate 3D adversarial examples with strong transferability. Specifically, we randomly scale or shear the input point cloud, so that the attack will not overfit the white-box model, thereby improving the transferability of the attack. Extensive experiments show that the SS attack proposed in this paper can be seamlessly combined with the existing state-of-the-art (SOTA) 3D point cloud attack methods to form more powerful attack methods, and the SS attack improves the transferability over 3.6 times compare to the baseline. Moreover, while substantially outperforming the baseline methods, the SS attack achieves SOTA transferability under various defenses. Our code will be available online at https://github.com/cuge1995/SS-attack



## **47. Projective Ranking-based GNN Evasion Attacks**

cs.LG

Accepted by IEEE Transactions on Knowledge and Data Engineering

**SubmitDate**: 2022-11-02    [abs](http://arxiv.org/abs/2202.12993v2) [paper-pdf](http://arxiv.org/pdf/2202.12993v2)

**Authors**: He Zhang, Xingliang Yuan, Chuan Zhou, Shirui Pan

**Abstract**: Graph neural networks (GNNs) offer promising learning methods for graph-related tasks. However, GNNs are at risk of adversarial attacks. Two primary limitations of the current evasion attack methods are highlighted: (1) The current GradArgmax ignores the "long-term" benefit of the perturbation. It is faced with zero-gradient and invalid benefit estimates in certain situations. (2) In the reinforcement learning-based attack methods, the learned attack strategies might not be transferable when the attack budget changes. To this end, we first formulate the perturbation space and propose an evaluation framework and the projective ranking method. We aim to learn a powerful attack strategy then adapt it as little as possible to generate adversarial samples under dynamic budget settings. In our method, based on mutual information, we rank and assess the attack benefits of each perturbation for an effective attack strategy. By projecting the strategy, our method dramatically minimizes the cost of learning a new attack strategy when the attack budget changes. In the comparative assessment with GradArgmax and RL-S2V, the results show our method owns high attack performance and effective transferability. The visualization of our method also reveals various attack patterns in the generation of adversarial samples.



## **48. Certified Robustness of Quantum Classifiers against Adversarial Examples through Quantum Noise**

quant-ph

Submitted to IEEE ICASSP 2023

**SubmitDate**: 2022-11-02    [abs](http://arxiv.org/abs/2211.00887v1) [paper-pdf](http://arxiv.org/pdf/2211.00887v1)

**Authors**: Jhih-Cing Huang, Yu-Lin Tsai, Chao-Han Huck Yang, Cheng-Fang Su, Chia-Mu Yu, Pin-Yu Chen, Sy-Yen Kuo

**Abstract**: Recently, quantum classifiers have been known to be vulnerable to adversarial attacks, where quantum classifiers are fooled by imperceptible noises to have misclassification. In this paper, we propose one first theoretical study that utilizing the added quantum random rotation noise can improve the robustness of quantum classifiers against adversarial attacks. We connect the definition of differential privacy and demonstrate the quantum classifier trained with the natural presence of additive noise is differentially private. Lastly, we derive a certified robustness bound to enable quantum classifiers to defend against adversarial examples supported by experimental results.



## **49. LMD: A Learnable Mask Network to Detect Adversarial Examples for Speaker Verification**

eess.AS

13 pages, 9 figures

**SubmitDate**: 2022-11-02    [abs](http://arxiv.org/abs/2211.00825v1) [paper-pdf](http://arxiv.org/pdf/2211.00825v1)

**Authors**: Xing Chen, Jie Wang, Xiao-Lei Zhang, Wei-Qiang Zhang, Kunde Yang

**Abstract**: Although the security of automatic speaker verification (ASV) is seriously threatened by recently emerged adversarial attacks, there have been some countermeasures to alleviate the threat. However, many defense approaches not only require the prior knowledge of the attackers but also possess weak interpretability. To address this issue, in this paper, we propose an attacker-independent and interpretable method, named learnable mask detector (LMD), to separate adversarial examples from the genuine ones. It utilizes score variation as an indicator to detect adversarial examples, where the score variation is the absolute discrepancy between the ASV scores of an original audio recording and its transformed audio synthesized from its masked complex spectrogram. A core component of the score variation detector is to generate the masked spectrogram by a neural network. The neural network needs only genuine examples for training, which makes it an attacker-independent approach. Its interpretability lies that the neural network is trained to minimize the score variation of the targeted ASV, and maximize the number of the masked spectrogram bins of the genuine training examples. Its foundation is based on the observation that, masking out the vast majority of the spectrogram bins with little speaker information will inevitably introduce a large score variation to the adversarial example, and a small score variation to the genuine example. Experimental results with 12 attackers and two representative ASV systems show that our proposed method outperforms five state-of-the-art baselines. The extensive experimental results can also be a benchmark for the detection-based ASV defenses.



## **50. On the detection of synthetic images generated by diffusion models**

cs.CV

**SubmitDate**: 2022-11-01    [abs](http://arxiv.org/abs/2211.00680v1) [paper-pdf](http://arxiv.org/pdf/2211.00680v1)

**Authors**: Riccardo Corvi, Davide Cozzolino, Giada Zingarini, Giovanni Poggi, Koki Nagano, Luisa Verdoliva

**Abstract**: Over the past decade, there has been tremendous progress in creating synthetic media, mainly thanks to the development of powerful methods based on generative adversarial networks (GAN). Very recently, methods based on diffusion models (DM) have been gaining the spotlight. In addition to providing an impressive level of photorealism, they enable the creation of text-based visual content, opening up new and exciting opportunities in many different application fields, from arts to video games. On the other hand, this property is an additional asset in the hands of malicious users, who can generate and distribute fake media perfectly adapted to their attacks, posing new challenges to the media forensic community. With this work, we seek to understand how difficult it is to distinguish synthetic images generated by diffusion models from pristine ones and whether current state-of-the-art detectors are suitable for the task. To this end, first we expose the forensics traces left by diffusion models, then study how current detectors, developed for GAN-generated images, perform on these new synthetic images, especially in challenging social-networks scenarios involving image compression and resizing. Datasets and code are available at github.com/grip-unina/DMimageDetection.



