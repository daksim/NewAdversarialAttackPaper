# Latest Adversarial Attack Papers
**update at 2022-04-06 06:31:29**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. DAD: Data-free Adversarial Defense at Test Time**

cs.LG

WACV 2022. Project page: https://sites.google.com/view/dad-wacv22

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2204.01568v1)

**Authors**: Gaurav Kumar Nayak, Ruchit Rawal, Anirban Chakraborty

**Abstracts**: Deep models are highly susceptible to adversarial attacks. Such attacks are carefully crafted imperceptible noises that can fool the network and can cause severe consequences when deployed. To encounter them, the model requires training data for adversarial training or explicit regularization-based techniques. However, privacy has become an important concern, restricting access to only trained models but not the training data (e.g. biometric data). Also, data curation is expensive and companies may have proprietary rights over it. To handle such situations, we propose a completely novel problem of 'test-time adversarial defense in absence of training data and even their statistics'. We solve it in two stages: a) detection and b) correction of adversarial samples. Our adversarial sample detection framework is initially trained on arbitrary data and is subsequently adapted to the unlabelled test data through unsupervised domain adaptation. We further correct the predictions on detected adversarial samples by transforming them in Fourier domain and obtaining their low frequency component at our proposed suitable radius for model prediction. We demonstrate the efficacy of our proposed technique via extensive experiments against several adversarial attacks and for different model architectures and datasets. For a non-robust Resnet-18 model pre-trained on CIFAR-10, our detection method correctly identifies 91.42% adversaries. Also, we significantly improve the adversarial accuracy from 0% to 37.37% with a minimal drop of 0.02% in clean accuracy on state-of-the-art 'Auto Attack' without having to retrain the model.



## **2. RobustSense: Defending Adversarial Attack for Secure Device-Free Human Activity Recognition**

cs.CR

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2204.01560v1)

**Authors**: Jianfei Yang, Han Zou, Lihua Xie

**Abstracts**: Deep neural networks have empowered accurate device-free human activity recognition, which has wide applications. Deep models can extract robust features from various sensors and generalize well even in challenging situations such as data-insufficient cases. However, these systems could be vulnerable to input perturbations, i.e. adversarial attacks. We empirically demonstrate that both black-box Gaussian attacks and modern adversarial white-box attacks can render their accuracies to plummet. In this paper, we firstly point out that such phenomenon can bring severe safety hazards to device-free sensing systems, and then propose a novel learning framework, RobustSense, to defend common attacks. RobustSense aims to achieve consistent predictions regardless of whether there exists an attack on its input or not, alleviating the negative effect of distribution perturbation caused by adversarial attacks. Extensive experiments demonstrate that our proposed method can significantly enhance the model robustness of existing deep models, overcoming possible attacks. The results validate that our method works well on wireless human activity recognition and person identification systems. To the best of our knowledge, this is the first work to investigate adversarial attacks and further develop a novel defense framework for wireless human activity recognition in mobile computing research.



## **3. PRADA: Practical Black-Box Adversarial Attacks against Neural Ranking Models**

cs.IR

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2204.01321v1)

**Authors**: Chen Wu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstracts**: Neural ranking models (NRMs) have shown remarkable success in recent years, especially with pre-trained language models. However, deep neural models are notorious for their vulnerability to adversarial examples. Adversarial attacks may become a new type of web spamming technique given our increased reliance on neural information retrieval models. Therefore, it is important to study potential adversarial attacks to identify vulnerabilities of NRMs before they are deployed.   In this paper, we introduce the Adversarial Document Ranking Attack (ADRA) task against NRMs, which aims to promote a target document in rankings by adding adversarial perturbations to its text. We focus on the decision-based black-box attack setting, where the attackers have no access to the model parameters and gradients, but can only acquire the rank positions of the partial retrieved list by querying the target model. This attack setting is realistic in real-world search engines. We propose a novel Pseudo Relevance-based ADversarial ranking Attack method (PRADA) that learns a surrogate model based on Pseudo Relevance Feedback (PRF) to generate gradients for finding the adversarial perturbations.   Experiments on two web search benchmark datasets show that PRADA can outperform existing attack strategies and successfully fool the NRM with small indiscernible perturbations of text.



## **4. Captcha Attack: Turning Captchas Against Humanity**

cs.CR

Currently under submission

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2201.04014v3)

**Authors**: Mauro Conti, Luca Pajola, Pier Paolo Tricomi

**Abstracts**: Nowadays, people generate and share massive content on online platforms (e.g., social networks, blogs). In 2021, the 1.9 billion daily active Facebook users posted around 150 thousand photos every minute. Content moderators constantly monitor these online platforms to prevent the spreading of inappropriate content (e.g., hate speech, nudity images). Based on deep learning (DL) advances, Automatic Content Moderators (ACM) help human moderators handle high data volume. Despite their advantages, attackers can exploit weaknesses of DL components (e.g., preprocessing, model) to affect their performance. Therefore, an attacker can leverage such techniques to spread inappropriate content by evading ACM.   In this work, we propose CAPtcha Attack (CAPA), an adversarial technique that allows users to spread inappropriate text online by evading ACM controls. CAPA, by generating custom textual CAPTCHAs, exploits ACM's careless design implementations and internal procedures vulnerabilities. We test our attack on real-world ACM, and the results confirm the ferocity of our simple yet effective attack, reaching up to a 100% evasion success in most cases. At the same time, we demonstrate the difficulties in designing CAPA mitigations, opening new challenges in CAPTCHAs research area.



## **5. Detecting In-vehicle Intrusion via Semi-supervised Learning-based Convolutional Adversarial Autoencoders**

cs.CR

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2204.01193v1)

**Authors**: Thien-Nu Hoang, Daehee Kim

**Abstracts**: With the development of autonomous vehicle technology, the controller area network (CAN) bus has become the de facto standard for an in-vehicle communication system because of its simplicity and efficiency. However, without any encryption and authentication mechanisms, the in-vehicle network using the CAN protocol is susceptible to a wide range of attacks. Many studies, which are mostly based on machine learning, have proposed installing an intrusion detection system (IDS) for anomaly detection in the CAN bus system. Although machine learning methods have many advantages for IDS, previous models usually require a large amount of labeled data, which results in high time and labor costs. To handle this problem, we propose a novel semi-supervised learning-based convolutional adversarial autoencoder model in this paper. The proposed model combines two popular deep learning models: autoencoder and generative adversarial networks. First, the model is trained with unlabeled data to learn the manifolds of normal and attack patterns. Then, only a small number of labeled samples are used in supervised training. The proposed model can detect various kinds of message injection attacks, such as DoS, fuzzy, and spoofing, as well as unknown attacks. The experimental results show that the proposed model achieves the highest F1 score of 0.99 and a low error rate of 0.1\% with limited labeled data compared to other supervised methods. In addition, we show that the model can meet the real-time requirement by analyzing the model complexity in terms of the number of trainable parameters and inference time. This study successfully reduced the number of model parameters by five times and the inference time by eight times, compared to a state-of-the-art model.



## **6. DST: Dynamic Substitute Training for Data-free Black-box Attack**

cs.CV

Accepted by CVPR2022

**SubmitDate**: 2022-04-03    [paper-pdf](http://arxiv.org/pdf/2204.00972v1)

**Authors**: Wenxuan Wang, Xuelin Qian, Yanwei Fu, Xiangyang Xue

**Abstracts**: With the wide applications of deep neural network models in various computer vision tasks, more and more works study the model vulnerability to adversarial examples. For data-free black box attack scenario, existing methods are inspired by the knowledge distillation, and thus usually train a substitute model to learn knowledge from the target model using generated data as input. However, the substitute model always has a static network structure, which limits the attack ability for various target models and tasks. In this paper, we propose a novel dynamic substitute training attack method to encourage substitute model to learn better and faster from the target model. Specifically, a dynamic substitute structure learning strategy is proposed to adaptively generate optimal substitute model structure via a dynamic gate according to different target models and tasks. Moreover, we introduce a task-driven graph-based structure information learning constrain to improve the quality of generated training data, and facilitate the substitute model learning structural relationships from the target model multiple outputs. Extensive experiments have been conducted to verify the efficacy of the proposed attack method, which can achieve better performance compared with the state-of-the-art competitors on several datasets.



## **7. Adversarial Neon Beam: Robust Physical-World Adversarial Attack to DNNs**

cs.CV

**SubmitDate**: 2022-04-02    [paper-pdf](http://arxiv.org/pdf/2204.00853v1)

**Authors**: Chengyin Hu, Kalibinuer Tiliwalidi

**Abstracts**: In the physical world, light affects the performance of deep neural networks. Nowadays, many products based on deep neural network have been put into daily life. There are few researches on the effect of light on the performance of deep neural network models. However, the adversarial perturbations generated by light may have extremely dangerous effects on these systems. In this work, we propose an attack method called adversarial neon beam (AdvNB), which can execute the physical attack by obtaining the physical parameters of adversarial neon beams with very few queries. Experiments show that our algorithm can achieve advanced attack effect in both digital test and physical test. In the digital environment, 99.3% attack success rate was achieved, and in the physical environment, 100% attack success rate was achieved. Compared with the most advanced physical attack methods, our method can achieve better physical perturbation concealment. In addition, by analyzing the experimental data, we reveal some new phenomena brought about by the adversarial neon beam attack.



## **8. Precise Statistical Analysis of Classification Accuracies for Adversarial Training**

stat.ML

80 pages; to appear in the Annals of Statistics

**SubmitDate**: 2022-04-02    [paper-pdf](http://arxiv.org/pdf/2010.11213v2)

**Authors**: Adel Javanmard, Mahdi Soltanolkotabi

**Abstracts**: Despite the wide empirical success of modern machine learning algorithms and models in a multitude of applications, they are known to be highly susceptible to seemingly small indiscernible perturbations to the input data known as \emph{adversarial attacks}. A variety of recent adversarial training procedures have been proposed to remedy this issue. Despite the success of such procedures at increasing accuracy on adversarially perturbed inputs or \emph{robust accuracy}, these techniques often reduce accuracy on natural unperturbed inputs or \emph{standard accuracy}. Complicating matters further, the effect and trend of adversarial training procedures on standard and robust accuracy is rather counter intuitive and radically dependent on a variety of factors including the perceived form of the perturbation during training, size/quality of data, model overparameterization, etc. In this paper we focus on binary classification problems where the data is generated according to the mixture of two Gaussians with general anisotropic covariance matrices and derive a precise characterization of the standard and robust accuracy for a class of minimax adversarially trained models. We consider a general norm-based adversarial model, where the adversary can add perturbations of bounded $\ell_p$ norm to each input data, for an arbitrary $p\ge 1$. Our comprehensive analysis allows us to theoretically explain several intriguing empirical phenomena and provide a precise understanding of the role of different problem parameters on standard and robust accuracies.



## **9. SkeleVision: Towards Adversarial Resiliency of Person Tracking with Multi-Task Learning**

cs.CV

**SubmitDate**: 2022-04-02    [paper-pdf](http://arxiv.org/pdf/2204.00734v1)

**Authors**: Nilaksh Das, Sheng-Yun Peng, Duen Horng Chau

**Abstracts**: Person tracking using computer vision techniques has wide ranging applications such as autonomous driving, home security and sports analytics. However, the growing threat of adversarial attacks raises serious concerns regarding the security and reliability of such techniques. In this work, we study the impact of multi-task learning (MTL) on the adversarial robustness of the widely used SiamRPN tracker, in the context of person tracking. Specifically, we investigate the effect of jointly learning with semantically analogous tasks of person tracking and human keypoint detection. We conduct extensive experiments with more powerful adversarial attacks that can be physically realizable, demonstrating the practical value of our approach. Our empirical study with simulated as well as real-world datasets reveals that training with MTL consistently makes it harder to attack the SiamRPN tracker, compared to typically training only on the single task of person tracking.



## **10. FrequencyLowCut Pooling -- Plug & Play against Catastrophic Overfitting**

cs.CV

**SubmitDate**: 2022-04-01    [paper-pdf](http://arxiv.org/pdf/2204.00491v1)

**Authors**: Julia Grabinski, Steffen Jung, Janis Keuper, Margret Keuper

**Abstracts**: Over the last years, Convolutional Neural Networks (CNNs) have been the dominating neural architecture in a wide range of computer vision tasks. From an image and signal processing point of view, this success might be a bit surprising as the inherent spatial pyramid design of most CNNs is apparently violating basic signal processing laws, i.e. Sampling Theorem in their down-sampling operations. However, since poor sampling appeared not to affect model accuracy, this issue has been broadly neglected until model robustness started to receive more attention. Recent work [17] in the context of adversarial attacks and distribution shifts, showed after all, that there is a strong correlation between the vulnerability of CNNs and aliasing artifacts induced by poor down-sampling operations. This paper builds on these findings and introduces an aliasing free down-sampling operation which can easily be plugged into any CNN architecture: FrequencyLowCut pooling. Our experiments show, that in combination with simple and fast FGSM adversarial training, our hyper-parameter free operator significantly improves model robustness and avoids catastrophic overfitting.



## **11. Sensor Data Validation and Driving Safety in Autonomous Driving Systems**

cs.CV

PhD Thesis, City University of Hong Kong

**SubmitDate**: 2022-04-01    [paper-pdf](http://arxiv.org/pdf/2203.16130v2)

**Authors**: Jindi Zhang

**Abstracts**: Autonomous driving technology has drawn a lot of attention due to its fast development and extremely high commercial values. The recent technological leap of autonomous driving can be primarily attributed to the progress in the environment perception. Good environment perception provides accurate high-level environment information which is essential for autonomous vehicles to make safe and precise driving decisions and strategies. Moreover, such progress in accurate environment perception would not be possible without deep learning models and advanced onboard sensors, such as optical sensors (LiDARs and cameras), radars, GPS. However, the advanced sensors and deep learning models are prone to recently invented attack methods. For example, LiDARs and cameras can be compromised by optical attacks, and deep learning models can be attacked by adversarial examples. The attacks on advanced sensors and deep learning models can largely impact the accuracy of the environment perception, posing great threats to the safety and security of autonomous vehicles. In this thesis, we study the detection methods against the attacks on onboard sensors and the linkage between attacked deep learning models and driving safety for autonomous vehicles. To detect the attacks, redundant data sources can be exploited, since information distortions caused by attacks in victim sensor data result in inconsistency with the information from other redundant sources. To study the linkage between attacked deep learning models and driving safety...



## **12. Multi-Expert Adversarial Attack Detection in Person Re-identification Using Context Inconsistency**

cs.CV

Accepted at IEEE ICCV 2021

**SubmitDate**: 2022-04-01    [paper-pdf](http://arxiv.org/pdf/2108.09891v2)

**Authors**: Xueping Wang, Shasha Li, Min Liu, Yaonan Wang, Amit K. Roy-Chowdhury

**Abstracts**: The success of deep neural networks (DNNs) has promoted the widespread applications of person re-identification (ReID). However, ReID systems inherit the vulnerability of DNNs to malicious attacks of visually inconspicuous adversarial perturbations. Detection of adversarial attacks is, therefore, a fundamental requirement for robust ReID systems. In this work, we propose a Multi-Expert Adversarial Attack Detection (MEAAD) approach to achieve this goal by checking context inconsistency, which is suitable for any DNN-based ReID systems. Specifically, three kinds of context inconsistencies caused by adversarial attacks are employed to learn a detector for distinguishing the perturbed examples, i.e., a) the embedding distances between a perturbed query person image and its top-K retrievals are generally larger than those between a benign query image and its top-K retrievals, b) the embedding distances among the top-K retrievals of a perturbed query image are larger than those of a benign query image, c) the top-K retrievals of a benign query image obtained with multiple expert ReID models tend to be consistent, which is not preserved when attacks are present. Extensive experiments on the Market1501 and DukeMTMC-ReID datasets show that, as the first adversarial attack detection approach for ReID, MEAAD effectively detects various adversarial attacks and achieves high ROC-AUC (over 97.5%).



## **13. Effect of Balancing Data Using Synthetic Data on the Performance of Machine Learning Classifiers for Intrusion Detection in Computer Networks**

cs.LG

**SubmitDate**: 2022-04-01    [paper-pdf](http://arxiv.org/pdf/2204.00144v1)

**Authors**: Ayesha S. Dina, A. B. Siddique, D. Manivannan

**Abstracts**: Attacks on computer networks have increased significantly in recent days, due in part to the availability of sophisticated tools for launching such attacks as well as thriving underground cyber-crime economy to support it. Over the past several years, researchers in academia and industry used machine learning (ML) techniques to design and implement Intrusion Detection Systems (IDSes) for computer networks. Many of these researchers used datasets collected by various organizations to train ML models for predicting intrusions. In many of the datasets used in such systems, data are imbalanced (i.e., not all classes have equal amount of samples). With unbalanced data, the predictive models developed using ML algorithms may produce unsatisfactory classifiers which would affect accuracy in predicting intrusions. Traditionally, researchers used over-sampling and under-sampling for balancing data in datasets to overcome this problem. In this work, in addition to over-sampling, we also use a synthetic data generation method, called Conditional Generative Adversarial Network (CTGAN), to balance data and study their effect on various ML classifiers. To the best of our knowledge, no one else has used CTGAN to generate synthetic samples to balance intrusion detection datasets. Based on extensive experiments using a widely used dataset NSL-KDD, we found that training ML models on dataset balanced with synthetic samples generated by CTGAN increased prediction accuracy by up to $8\%$, compared to training the same ML models over unbalanced data. Our experiments also show that the accuracy of some ML models trained over data balanced with random over-sampling decline compared to the same ML models trained over unbalanced data.



## **14. Reverse Engineering of Imperceptible Adversarial Image Perturbations**

cs.CV

**SubmitDate**: 2022-04-01    [paper-pdf](http://arxiv.org/pdf/2203.14145v2)

**Authors**: Yifan Gong, Yuguang Yao, Yize Li, Yimeng Zhang, Xiaoming Liu, Xue Lin, Sijia Liu

**Abstracts**: It has been well recognized that neural network based image classifiers are easily fooled by images with tiny perturbations crafted by an adversary. There has been a vast volume of research to generate and defend such adversarial attacks. However, the following problem is left unexplored: How to reverse-engineer adversarial perturbations from an adversarial image? This leads to a new adversarial learning paradigm--Reverse Engineering of Deceptions (RED). If successful, RED allows us to estimate adversarial perturbations and recover the original images. However, carefully crafted, tiny adversarial perturbations are difficult to recover by optimizing a unilateral RED objective. For example, the pure image denoising method may overfit to minimizing the reconstruction error but hardly preserve the classification properties of the true adversarial perturbations. To tackle this challenge, we formalize the RED problem and identify a set of principles crucial to the RED approach design. Particularly, we find that prediction alignment and proper data augmentation (in terms of spatial transformations) are two criteria to achieve a generalizable RED approach. By integrating these RED principles with image denoising, we propose a new Class-Discriminative Denoising based RED framework, termed CDD-RED. Extensive experiments demonstrate the effectiveness of CDD-RED under different evaluation metrics (ranging from the pixel-level, prediction-level to the attribution-level alignment) and a variety of attack generation methods (e.g., FGSM, PGD, CW, AutoAttack, and adaptive attacks).



## **15. Scalable Whitebox Attacks on Tree-based Models**

stat.ML

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2204.00103v1)

**Authors**: Giuseppe Castiglione, Gavin Ding, Masoud Hashemi, Christopher Srinivasa, Ga Wu

**Abstracts**: Adversarial robustness is one of the essential safety criteria for guaranteeing the reliability of machine learning models. While various adversarial robustness testing approaches were introduced in the last decade, we note that most of them are incompatible with non-differentiable models such as tree ensembles. Since tree ensembles are widely used in industry, this reveals a crucial gap between adversarial robustness research and practical applications. This paper proposes a novel whitebox adversarial robustness testing approach for tree ensemble models. Concretely, the proposed approach smooths the tree ensembles through temperature controlled sigmoid functions, which enables gradient descent-based adversarial attacks. By leveraging sampling and the log-derivative trick, the proposed approach can scale up to testing tasks that were previously unmanageable. We compare the approach against both random perturbations and blackbox approaches on multiple public datasets (and corresponding models). Our results show that the proposed method can 1) successfully reveal the adversarial vulnerability of tree ensemble models without causing computational pressure for testing and 2) flexibly balance the search performance and time complexity to meet various testing criteria.



## **16. Parallel Proof-of-Work with Concrete Bounds**

cs.CR

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2204.00034v1)

**Authors**: Patrik Keller, Rainer Böhme

**Abstracts**: Authorization is challenging in distributed systems that cannot rely on the identification of nodes. Proof-of-work offers an alternative gate-keeping mechanism, but its probabilistic nature is incompatible with conventional security definitions. Recent related work establishes concrete bounds for the failure probability of Bitcoin's sequential proof-of-work mechanism. We propose a family of state replication protocols using parallel proof-of-work. Our bottom-up design from an agreement sub-protocol allows us to give concrete bounds for the failure probability in adversarial synchronous networks. After the typical interval of 10 minutes, parallel proof-of-work offers two orders of magnitude more security than sequential proof-of-work. This means that state updates can be sufficiently secure to support commits after one block (i.e., after 10 minutes), removing the risk of double-spending in many applications. We offer guidance on the optimal choice of parameters for a wide range of network and attacker assumptions. Simulations show that the proposed construction is robust against violations of design assumptions.



## **17. Truth Serum: Poisoning Machine Learning Models to Reveal Their Secrets**

cs.CR

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2204.00032v1)

**Authors**: Florian Tramèr, Reza Shokri, Ayrton San Joaquin, Hoang Le, Matthew Jagielski, Sanghyun Hong, Nicholas Carlini

**Abstracts**: We introduce a new class of attacks on machine learning models. We show that an adversary who can poison a training dataset can cause models trained on this dataset to leak significant private details of training points belonging to other parties. Our active inference attacks connect two independent lines of work targeting the integrity and privacy of machine learning training data.   Our attacks are effective across membership inference, attribute inference, and data extraction. For example, our targeted attacks can poison <0.1% of the training dataset to boost the performance of inference attacks by 1 to 2 orders of magnitude. Further, an adversary who controls a significant fraction of the training data (e.g., 50%) can launch untargeted attacks that enable 8x more precise inference on all other users' otherwise-private data points.   Our results cast doubts on the relevance of cryptographic privacy guarantees in multiparty computation protocols for machine learning, if parties can arbitrarily select their share of training data.



## **18. Training strategy for a lightweight countermeasure model for automatic speaker verification**

cs.SD

ASVspoof2021

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2203.17031v1)

**Authors**: Yen-Lun Liao, Xuanjun Chen, Chung-Che Wang, Jyh-Shing Roger Jang

**Abstracts**: The countermeasure (CM) model is developed to protect Automatic Speaker Verification (ASV) systems from spoof attacks and prevent resulting personal information leakage. Based on practicality and security considerations, the CM model is usually deployed on edge devices, which have more limited computing resources and storage space than cloud- based systems. This work proposes training strategies for a lightweight CM model for ASV, using generalized end- to-end (GE2E) pre-training and adversarial fine-tuning to improve performance, and applying knowledge distillation (KD) to reduce the size of the CM model. In the evalua- tion phase of the ASVspoof 2021 Logical Access task, the lightweight ResNetSE model reaches min t-DCF 0.2695 and EER 3.54%. Compared to the teacher model, the lightweight student model only uses 22.5% of parameters and 21.1% of multiply and accumulate operands of the teacher model.



## **19. Improving Adversarial Transferability via Neuron Attribution-Based Attacks**

cs.LG

CVPR 2022

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2204.00008v1)

**Authors**: Jianping Zhang, Weibin Wu, Jen-tse Huang, Yizhan Huang, Wenxuan Wang, Yuxin Su, Michael R. Lyu

**Abstracts**: Deep neural networks (DNNs) are known to be vulnerable to adversarial examples. It is thus imperative to devise effective attack algorithms to identify the deficiencies of DNNs beforehand in security-sensitive applications. To efficiently tackle the black-box setting where the target model's particulars are unknown, feature-level transfer-based attacks propose to contaminate the intermediate feature outputs of local models, and then directly employ the crafted adversarial samples to attack the target model. Due to the transferability of features, feature-level attacks have shown promise in synthesizing more transferable adversarial samples. However, existing feature-level attacks generally employ inaccurate neuron importance estimations, which deteriorates their transferability. To overcome such pitfalls, in this paper, we propose the Neuron Attribution-based Attack (NAA), which conducts feature-level attacks with more accurate neuron importance estimations. Specifically, we first completely attribute a model's output to each neuron in a middle layer. We then derive an approximation scheme of neuron attribution to tremendously reduce the computation overhead. Finally, we weight neurons based on their attribution results and launch feature-level attacks. Extensive experiments confirm the superiority of our approach to the state-of-the-art benchmarks.



## **20. Towards Robust Rain Removal Against Adversarial Attacks: A Comprehensive Benchmark Analysis and Beyond**

cs.CV

10 pages, 6 figures, to appear in CVPR 2022

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2203.16931v1)

**Authors**: Yi Yu, Wenhan Yang, Yap-Peng Tan, Alex C. Kot

**Abstracts**: Rain removal aims to remove rain streaks from images/videos and reduce the disruptive effects caused by rain. It not only enhances image/video visibility but also allows many computer vision algorithms to function properly. This paper makes the first attempt to conduct a comprehensive study on the robustness of deep learning-based rain removal methods against adversarial attacks. Our study shows that, when the image/video is highly degraded, rain removal methods are more vulnerable to the adversarial attacks as small distortions/perturbations become less noticeable or detectable. In this paper, we first present a comprehensive empirical evaluation of various methods at different levels of attacks and with various losses/targets to generate the perturbations from the perspective of human perception and machine analysis tasks. A systematic evaluation of key modules in existing methods is performed in terms of their robustness against adversarial attacks. From the insights of our analysis, we construct a more robust deraining method by integrating these effective modules. Finally, we examine various types of adversarial attacks that are specific to deraining problems and their effects on both human and machine vision tasks, including 1) rain region attacks, adding perturbations only in the rain regions to make the perturbations in the attacked rain images less visible; 2) object-sensitive attacks, adding perturbations only in regions near the given objects. Code is available at https://github.com/yuyi-sd/Robust_Rain_Removal.



## **21. Assessing the risk of re-identification arising from an attack on anonymised data**

cs.LG

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2203.16921v1)

**Authors**: Anna Antoniou, Giacomo Dossena, Julia MacMillan, Steven Hamblin, David Clifton, Paula Petrone

**Abstracts**: Objective: The use of routinely-acquired medical data for research purposes requires the protection of patient confidentiality via data anonymisation. The objective of this work is to calculate the risk of re-identification arising from a malicious attack to an anonymised dataset, as described below. Methods: We first present an analytical means of estimating the probability of re-identification of a single patient in a k-anonymised dataset of Electronic Health Record (EHR) data. Second, we generalize this solution to obtain the probability of multiple patients being re-identified. We provide synthetic validation via Monte Carlo simulations to illustrate the accuracy of the estimates obtained. Results: The proposed analytical framework for risk estimation provides re-identification probabilities that are in agreement with those provided by simulation in a number of scenarios. Our work is limited by conservative assumptions which inflate the re-identification probability. Discussion: Our estimates show that the re-identification probability increases with the proportion of the dataset maliciously obtained and that it has an inverse relationship with the equivalence class size. Our recursive approach extends the applicability domain to the general case of a multi-patient re-identification attack in an arbitrary k-anonymisation scheme. Conclusion: We prescribe a systematic way to parametrize the k-anonymisation process based on a pre-determined re-identification probability. We observed that the benefits of a reduced re-identification risk that come with increasing k-size may not be worth the reduction in data granularity when one is considering benchmarking the re-identification probability on the size of the portion of the dataset maliciously obtained by the adversary.



## **22. Attack Impact Evaluation by Exact Convexification through State Space Augmentation**

eess.SY

8 pages

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2203.16803v1)

**Authors**: Hampei Sasahara, Takashi Tanaka, Henrik Sandberg

**Abstracts**: We address the attack impact evaluation problem for control system security. We formulate the problem as a Markov decision process with a temporally joint chance constraint that forces the adversary to avoid being detected throughout the considered time period. Owing to the joint constraint, the optimal control policy depends not only on the current state but also on the entire history, which leads to the explosion of the search space and makes the problem generally intractable. It is shown that whether an alarm has been triggered or not, in addition to the current state is sufficient for specifying the optimal decision at each time step. Augmentation of the information to the state space induces an equivalent convex optimization problem, which is tractable using standard solvers.



## **23. The Block-based Mobile PDE Systems Are Not Secure -- Experimental Attacks**

cs.CR

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2203.16349v2)

**Authors**: Niusen Chen, Bo Chen, Weisong Shi

**Abstracts**: Nowadays, mobile devices have been used broadly to store and process sensitive data. To ensure confidentiality of the sensitive data, Full Disk Encryption (FDE) is often integrated in mainstream mobile operating systems like Android and iOS. FDE however cannot defend against coercive attacks in which the adversary can force the device owner to disclose the decryption key. To combat the coercive attacks, Plausibly Deniable Encryption (PDE) is leveraged to plausibly deny the very existence of sensitive data. However, most of the existing PDE systems for mobile devices are deployed at the block layer and suffer from deniability compromises.   Having observed that none of existing works in the literature have experimentally demonstrated the aforementioned compromises, our work bridges this gap by experimentally confirming the deniability compromises of the block-layer mobile PDE systems. We have built a mobile device testbed, which consists of a host computing device and a flash storage device. Additionally, we have deployed both the hidden volume PDE and the steganographic file system at the block layer of the testbed and performed disk forensics to assess potential compromises on the raw NAND flash. Our experimental results confirm it is indeed possible for the adversary to compromise the block-layer PDE systems by accessing the raw NAND flash in practice. We also discuss potential issues when performing such attacks in real world.



## **24. Well-classified Examples are Underestimated in Classification with Deep Neural Networks**

cs.LG

Accepted by AAAI 2022; 17 pages, 11 figures, 13 tables

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2110.06537v5)

**Authors**: Guangxiang Zhao, Wenkai Yang, Xuancheng Ren, Lei Li, Yunfang Wu, Xu Sun

**Abstracts**: The conventional wisdom behind learning deep classification models is to focus on bad-classified examples and ignore well-classified examples that are far from the decision boundary. For instance, when training with cross-entropy loss, examples with higher likelihoods (i.e., well-classified examples) contribute smaller gradients in back-propagation. However, we theoretically show that this common practice hinders representation learning, energy optimization, and margin growth. To counteract this deficiency, we propose to reward well-classified examples with additive bonuses to revive their contribution to the learning process. This counterexample theoretically addresses these three issues. We empirically support this claim by directly verifying the theoretical results or significant performance improvement with our counterexample on diverse tasks, including image classification, graph classification, and machine translation. Furthermore, this paper shows that we can deal with complex scenarios, such as imbalanced classification, OOD detection, and applications under adversarial attacks because our idea can solve these three issues. Code is available at: https://github.com/lancopku/well-classified-examples-are-underestimated.



## **25. Example-based Explanations with Adversarial Attacks for Respiratory Sound Analysis**

cs.SD

Submitted to INTERSPEECH 2022

**SubmitDate**: 2022-03-30    [paper-pdf](http://arxiv.org/pdf/2203.16141v1)

**Authors**: Yi Chang, Zhao Ren, Thanh Tam Nguyen, Wolfgang Nejdl, Björn W. Schuller

**Abstracts**: Respiratory sound classification is an important tool for remote screening of respiratory-related diseases such as pneumonia, asthma, and COVID-19. To facilitate the interpretability of classification results, especially ones based on deep learning, many explanation methods have been proposed using prototypes. However, existing explanation techniques often assume that the data is non-biased and the prediction results can be explained by a set of prototypical examples. In this work, we develop a unified example-based explanation method for selecting both representative data (prototypes) and outliers (criticisms). In particular, we propose a novel application of adversarial attacks to generate an explanation spectrum of data instances via an iterative fast gradient sign method. Such unified explanation can avoid over-generalisation and bias by allowing human experts to assess the model mistakes case by case. We performed a wide range of quantitative and qualitative evaluations to show that our approach generates effective and understandable explanation and is robust with many deep learning models



## **26. Fooling the primate brain with minimal, targeted image manipulation**

q-bio.NC

**SubmitDate**: 2022-03-30    [paper-pdf](http://arxiv.org/pdf/2011.05623v3)

**Authors**: Li Yuan, Will Xiao, Giorgia Dellaferrera, Gabriel Kreiman, Francis E. H. Tay, Jiashi Feng, Margaret S. Livingstone

**Abstracts**: Artificial neural networks (ANNs) are considered the current best models of biological vision. ANNs are the best predictors of neural activity in the ventral stream; moreover, recent work has demonstrated that ANN models fitted to neuronal activity can guide the synthesis of images that drive pre-specified response patterns in small neuronal populations. Despite the success in predicting and steering firing activity, these results have not been connected with perceptual or behavioral changes. Here we propose an array of methods for creating minimal, targeted image perturbations that lead to changes in both neuronal activity and perception as reflected in behavior. We generated 'deceptive images' of human faces, monkey faces, and noise patterns so that they are perceived as a different, pre-specified target category, and measured both monkey neuronal responses and human behavior to these images. We found several effective methods for changing primate visual categorization that required much smaller image change compared to untargeted noise. Our work shares the same goal with adversarial attack, namely the manipulation of images with minimal, targeted noise that leads ANN models to misclassify the images. Our results represent a valuable step in quantifying and characterizing the differences in perturbation robustness of biological and artificial vision.



## **27. StyleFool: Fooling Video Classification Systems via Style Transfer**

cs.CV

18 pages, 7 figures

**SubmitDate**: 2022-03-30    [paper-pdf](http://arxiv.org/pdf/2203.16000v1)

**Authors**: Yuxin Cao, Xi Xiao, Ruoxi Sun, Derui Wang, Minhui Xue, Sheng Wen

**Abstracts**: Video classification systems are vulnerable to adversarial attacks, which can create severe security problems in video verification. Current black-box attacks need a large number of queries to succeed, resulting in high computational overhead in the process of attack. On the other hand, attacks with restricted perturbations are ineffective against defenses such as denoising or adversarial training. In this paper, we focus on unrestricted perturbations and propose StyleFool, a black-box video adversarial attack via style transfer to fool the video classification system. StyleFool first utilizes color theme proximity to select the best style image, which helps avoid unnatural details in the stylized videos. Meanwhile, the target class confidence is additionally considered in targeted attack to influence the output distribution of the classifier by moving the stylized video closer to or even across the decision boundary. A gradient-free method is then employed to further optimize the adversarial perturbation. We carry out extensive experiments to evaluate StyleFool on two standard datasets, UCF-101 and HMDB-51. The experimental results suggest that StyleFool outperforms the state-of-the-art adversarial attacks in terms of both number of queries and robustness against existing defenses. We identify that 50% of the stylized videos in untargeted attack do not need any query since they can already fool the video classification model. Furthermore, we evaluate the indistinguishability through a user study to show that the adversarial samples of StyleFool look imperceptible to human eyes, despite unrestricted perturbations.



## **28. Recent improvements of ASR models in the face of adversarial attacks**

cs.CR

Submitted to Interspeech 2022

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2203.16536v1)

**Authors**: Raphael Olivier, Bhiksha Raj

**Abstracts**: Like many other tasks involving neural networks, Speech Recognition models are vulnerable to adversarial attacks. However recent research has pointed out differences between attacks and defenses on ASR models compared to image models. Improving the robustness of ASR models requires a paradigm shift from evaluating attacks on one or a few models to a systemic approach in evaluation. We lay the ground for such research by evaluating on various architectures a representative set of adversarial attacks: targeted and untargeted, optimization and speech processing-based, white-box, black-box and targeted attacks. Our results show that the relative strengths of different attack algorithms vary considerably when changing the model architecture, and that the results of some attacks are not to be blindly trusted. They also indicate that training choices such as self-supervised pretraining can significantly impact robustness by enabling transferable perturbations. We release our source code as a package that should help future research in evaluating their attacks and defenses.



## **29. NICGSlowDown: Evaluating the Efficiency Robustness of Neural Image Caption Generation Models**

cs.CV

This paper is accepted at CVPR2022

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2203.15859v1)

**Authors**: Simin Chen, Zihe Song, Mirazul Haque, Cong Liu, Wei Yang

**Abstracts**: Neural image caption generation (NICG) models have received massive attention from the research community due to their excellent performance in visual understanding. Existing work focuses on improving NICG model accuracy while efficiency is less explored. However, many real-world applications require real-time feedback, which highly relies on the efficiency of NICG models. Recent research observed that the efficiency of NICG models could vary for different inputs. This observation brings in a new attack surface of NICG models, i.e., An adversary might be able to slightly change inputs to cause the NICG models to consume more computational resources. To further understand such efficiency-oriented threats, we propose a new attack approach, NICGSlowDown, to evaluate the efficiency robustness of NICG models. Our experimental results show that NICGSlowDown can generate images with human-unnoticeable perturbations that will increase the NICG model latency up to 483.86%. We hope this research could raise the community's concern about the efficiency robustness of NICG models.



## **30. Characterizing the adversarial vulnerability of speech self-supervised learning**

cs.SD

Accepted by ICASSP 2022

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2111.04330v2)

**Authors**: Haibin Wu, Bo Zheng, Xu Li, Xixin Wu, Hung-yi Lee, Helen Meng

**Abstracts**: A leaderboard named Speech processing Universal PERformance Benchmark (SUPERB), which aims at benchmarking the performance of a shared self-supervised learning (SSL) speech model across various downstream speech tasks with minimal modification of architectures and small amount of data, has fueled the research for speech representation learning. The SUPERB demonstrates speech SSL upstream models improve the performance of various downstream tasks through just minimal adaptation. As the paradigm of the self-supervised learning upstream model followed by downstream tasks arouses more attention in the speech community, characterizing the adversarial robustness of such paradigm is of high priority. In this paper, we make the first attempt to investigate the adversarial vulnerability of such paradigm under the attacks from both zero-knowledge adversaries and limited-knowledge adversaries. The experimental results illustrate that the paradigm proposed by SUPERB is seriously vulnerable to limited-knowledge adversaries, and the attacks generated by zero-knowledge adversaries are with transferability. The XAB test verifies the imperceptibility of crafted adversarial attacks.



## **31. Adaptative Perturbation Patterns: Realistic Adversarial Learning for Robust Intrusion Detection**

cs.CR

18 pages, 6 tables, 10 figures, Future Internet journal

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2203.04234v2)

**Authors**: João Vitorino, Nuno Oliveira, Isabel Praça

**Abstracts**: Adversarial attacks pose a major threat to machine learning and to the systems that rely on it. In the cybersecurity domain, adversarial cyber-attack examples capable of evading detection are especially concerning. Nonetheless, an example generated for a domain with tabular data must be realistic within that domain. This work establishes the fundamental constraint levels required to achieve realism and introduces the Adaptative Perturbation Pattern Method (A2PM) to fulfill these constraints in a gray-box setting. A2PM relies on pattern sequences that are independently adapted to the characteristics of each class to create valid and coherent data perturbations. The proposed method was evaluated in a cybersecurity case study with two scenarios: Enterprise and Internet of Things (IoT) networks. Multilayer Perceptron (MLP) and Random Forest (RF) classifiers were created with regular and adversarial training, using the CIC-IDS2017 and IoT-23 datasets. In each scenario, targeted and untargeted attacks were performed against the classifiers, and the generated examples were compared with the original network traffic flows to assess their realism. The obtained results demonstrate that A2PM provides a scalable generation of realistic adversarial examples, which can be advantageous for both adversarial training and attacks.



## **32. Exploring Frequency Adversarial Attacks for Face Forgery Detection**

cs.CV

Accepted by CVPR2022

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2203.15674v1)

**Authors**: Shuai Jia, Chao Ma, Taiping Yao, Bangjie Yin, Shouhong Ding, Xiaokang Yang

**Abstracts**: Various facial manipulation techniques have drawn serious public concerns in morality, security, and privacy. Although existing face forgery classifiers achieve promising performance on detecting fake images, these methods are vulnerable to adversarial examples with injected imperceptible perturbations on the pixels. Meanwhile, many face forgery detectors always utilize the frequency diversity between real and fake faces as a crucial clue. In this paper, instead of injecting adversarial perturbations into the spatial domain, we propose a frequency adversarial attack method against face forgery detectors. Concretely, we apply discrete cosine transform (DCT) on the input images and introduce a fusion module to capture the salient region of adversary in the frequency domain. Compared with existing adversarial attacks (e.g. FGSM, PGD) in the spatial domain, our method is more imperceptible to human observers and does not degrade the visual quality of the original images. Moreover, inspired by the idea of meta-learning, we also propose a hybrid adversarial attack that performs attacks in both the spatial and frequency domains. Extensive experiments indicate that the proposed method fools not only the spatial-based detectors but also the state-of-the-art frequency-based detectors effectively. In addition, the proposed frequency attack enhances the transferability across face forgery detectors as black-box attacks.



## **33. Adaptive Image Transformations for Transfer-based Adversarial Attack**

cs.CV

33 pages, 7 figures, 10 tables

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2111.13844v2)

**Authors**: Zheng Yuan, Jie Zhang, Shiguang Shan

**Abstracts**: Adversarial attacks provide a good way to study the robustness of deep learning models. One category of methods in transfer-based black-box attack utilizes several image transformation operations to improve the transferability of adversarial examples, which is effective, but fails to take the specific characteristic of the input image into consideration. In this work, we propose a novel architecture, called Adaptive Image Transformation Learner (AITL), which incorporates different image transformation operations into a unified framework to further improve the transferability of adversarial examples. Unlike the fixed combinational transformations used in existing works, our elaborately designed transformation learner adaptively selects the most effective combination of image transformations specific to the input image. Extensive experiments on ImageNet demonstrate that our method significantly improves the attack success rates on both normally trained models and defense models under various settings.



## **34. Treatment Learning Transformer for Noisy Image Classification**

cs.CV

Preprint. The first version was finished in May 2018

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2203.15529v1)

**Authors**: Chao-Han Huck Yang, I-Te Danny Hung, Yi-Chieh Liu, Pin-Yu Chen

**Abstracts**: Current top-notch deep learning (DL) based vision models are primarily based on exploring and exploiting the inherent correlations between training data samples and their associated labels. However, a known practical challenge is their degraded performance against "noisy" data, induced by different circumstances such as spurious correlations, irrelevant contexts, domain shift, and adversarial attacks. In this work, we incorporate this binary information of "existence of noise" as treatment into image classification tasks to improve prediction accuracy by jointly estimating their treatment effects. Motivated from causal variational inference, we propose a transformer-based architecture, Treatment Learning Transformer (TLT), that uses a latent generative model to estimate robust feature representations from current observational input for noise image classification. Depending on the estimated noise level (modeled as a binary treatment factor), TLT assigns the corresponding inference network trained by the designed causal loss for prediction. We also create new noisy image datasets incorporating a wide range of noise factors (e.g., object masking, style transfer, and adversarial perturbation) for performance benchmarking. The superior performance of TLT in noisy image classification is further validated by several refutation evaluation metrics. As a by-product, TLT also improves visual salience methods for perceiving noisy images.



## **35. Spotting adversarial samples for speaker verification by neural vocoders**

cs.SD

Accepted by ICASSP 2022

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2107.00309v3)

**Authors**: Haibin Wu, Po-chun Hsu, Ji Gao, Shanshan Zhang, Shen Huang, Jian Kang, Zhiyong Wu, Helen Meng, Hung-yi Lee

**Abstracts**: Automatic speaker verification (ASV), one of the most important technology for biometric identification, has been widely adopted in security-critical applications. However, ASV is seriously vulnerable to recently emerged adversarial attacks, yet effective countermeasures against them are limited. In this paper, we adopt neural vocoders to spot adversarial samples for ASV. We use the neural vocoder to re-synthesize audio and find that the difference between the ASV scores for the original and re-synthesized audio is a good indicator for discrimination between genuine and adversarial samples. This effort is, to the best of our knowledge, among the first to pursue such a technical direction for detecting time-domain adversarial samples for ASV, and hence there is a lack of established baselines for comparison. Consequently, we implement the Griffin-Lim algorithm as the detection baseline. The proposed approach achieves effective detection performance that outperforms the baselines in all the settings. We also show that the neural vocoder adopted in the detection framework is dataset-independent. Our codes will be made open-source for future works to do fair comparison.



## **36. Mel Frequency Spectral Domain Defenses against Adversarial Attacks on Speech Recognition Systems**

eess.AS

This paper is 5 pages long and was submitted to Interspeech 2022

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2203.15283v1)

**Authors**: Nicholas Mehlman, Anirudh Sreeram, Raghuveer Peri, Shrikanth Narayanan

**Abstracts**: A variety of recent works have looked into defenses for deep neural networks against adversarial attacks particularly within the image processing domain. Speech processing applications such as automatic speech recognition (ASR) are increasingly relying on deep learning models, and so are also prone to adversarial attacks. However, many of the defenses explored for ASR simply adapt the image-domain defenses, which may not provide optimal robustness. This paper explores speech specific defenses using the mel spectral domain, and introduces a novel defense method called 'mel domain noise flooding' (MDNF). MDNF applies additive noise to the mel spectrogram of a speech utterance prior to re-synthesising the audio signal. We test the defenses against strong white-box adversarial attacks such as projected gradient descent (PGD) and Carlini-Wagner (CW) attacks, and show better robustness compared to a randomized smoothing baseline across strong threat models.



## **37. Robust Structured Declarative Classifiers for 3D Point Clouds: Defending Adversarial Attacks with Implicit Gradients**

cs.CV

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2203.15245v1)

**Authors**: Kaidong Li, Ziming Zhang, Cuncong Zhong, Guanghui Wang

**Abstracts**: Deep neural networks for 3D point cloud classification, such as PointNet, have been demonstrated to be vulnerable to adversarial attacks. Current adversarial defenders often learn to denoise the (attacked) point clouds by reconstruction, and then feed them to the classifiers as input. In contrast to the literature, we propose a family of robust structured declarative classifiers for point cloud classification, where the internal constrained optimization mechanism can effectively defend adversarial attacks through implicit gradients. Such classifiers can be formulated using a bilevel optimization framework. We further propose an effective and efficient instantiation of our approach, namely, Lattice Point Classifier (LPC), based on structured sparse coding in the permutohedral lattice and 2D convolutional neural networks (CNNs) that is end-to-end trainable. We demonstrate state-of-the-art robust point cloud classification performance on ModelNet40 and ScanNet under seven different attackers. For instance, we achieve 89.51% and 83.16% test accuracy on each dataset under the recent JGBA attacker that outperforms DUP-Net and IF-Defense with PointNet by ~70%. Demo code is available at https://zhang-vislab.github.io.



## **38. Zero-Query Transfer Attacks on Context-Aware Object Detectors**

cs.CV

CVPR 2022 Accepted

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2203.15230v1)

**Authors**: Zikui Cai, Shantanu Rane, Alejandro E. Brito, Chengyu Song, Srikanth V. Krishnamurthy, Amit K. Roy-Chowdhury, M. Salman Asif

**Abstracts**: Adversarial attacks perturb images such that a deep neural network produces incorrect classification results. A promising approach to defend against adversarial attacks on natural multi-object scenes is to impose a context-consistency check, wherein, if the detected objects are not consistent with an appropriately defined context, then an attack is suspected. Stronger attacks are needed to fool such context-aware detectors. We present the first approach for generating context-consistent adversarial attacks that can evade the context-consistency check of black-box object detectors operating on complex, natural scenes. Unlike many black-box attacks that perform repeated attempts and open themselves to detection, we assume a "zero-query" setting, where the attacker has no knowledge of the classification decisions of the victim system. First, we derive multiple attack plans that assign incorrect labels to victim objects in a context-consistent manner. Then we design and use a novel data structure that we call the perturbation success probability matrix, which enables us to filter the attack plans and choose the one most likely to succeed. This final attack plan is implemented using a perturbation-bounded adversarial attack algorithm. We compare our zero-query attack against a few-query scheme that repeatedly checks if the victim system is fooled. We also compare against state-of-the-art context-agnostic attacks. Against a context-aware defense, the fooling rate of our zero-query approach is significantly higher than context-agnostic approaches and higher than that achievable with up to three rounds of the few-query scheme.



## **39. Synthesizing Attack-Aware Control and Active Sensing Strategies under Reactive Sensor Attacks**

math.OC

6 pages, 1 figure, 1 table, 1 algorithm

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2204.01584v1)

**Authors**: Sumukha Udupa, Abhishek N. Kulkarni, Shuo Han, Nandi O. Leslie, Charles A. Kamhoua, Jie Fu

**Abstracts**: We consider the probabilistic planning problem for a defender (P1) who can jointly query the sensors and take control actions to reach a set of goal states while being aware of possible sensor attacks by an adversary (P2) who has perfect observations. To synthesize a provably correct, attack-aware control and active sensing strategy for P1, we construct a stochastic game on graph where the augmented state includes the actual game state (known by the attacker), the belief of the defender about the game state (constructed by the attacker given the attacker's information about the defender's information). We presented an algorithm to solve a belief-based, randomized strategy for P1 to ensure satisfying P1's reachability objective with probability one, under the worst case sensor attacks carried out by an informed P2. The correctness of the algorithm is proven and illustrated with an example.



## **40. A Robust Phased Elimination Algorithm for Corruption-Tolerant Gaussian Process Bandits**

stat.ML

Added references

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2202.01850v2)

**Authors**: Ilija Bogunovic, Zihan Li, Andreas Krause, Jonathan Scarlett

**Abstracts**: We consider the sequential optimization of an unknown, continuous, and expensive to evaluate reward function, from noisy and adversarially corrupted observed rewards. When the corruption attacks are subject to a suitable budget $C$ and the function lives in a Reproducing Kernel Hilbert Space (RKHS), the problem can be posed as corrupted Gaussian process (GP) bandit optimization. We propose a novel robust elimination-type algorithm that runs in epochs, combines exploration with infrequent switching to select a small subset of actions, and plays each action for multiple time instants. Our algorithm, Robust GP Phased Elimination (RGP-PE), successfully balances robustness to corruptions with exploration and exploitation such that its performance degrades minimally in the presence (or absence) of adversarial corruptions. When $T$ is the number of samples and $\gamma_T$ is the maximal information gain, the corruption-dependent term in our regret bound is $O(C \gamma_T^{3/2})$, which is significantly tighter than the existing $O(C \sqrt{T \gamma_T})$ for several commonly-considered kernels. We perform the first empirical study of robustness in the corrupted GP bandit setting, and show that our algorithm is robust against a variety of adversarial attacks.



## **41. Neurosymbolic hybrid approach to driver collision warning**

cs.CV

SPIE Defense and Commercial Sensing 2022

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2203.15076v1)

**Authors**: Kyongsik Yun, Thomas Lu, Alexander Huyen, Patrick Hammer, Pei Wang

**Abstracts**: There are two main algorithmic approaches to autonomous driving systems: (1) An end-to-end system in which a single deep neural network learns to map sensory input directly into appropriate warning and driving responses. (2) A mediated hybrid recognition system in which a system is created by combining independent modules that detect each semantic feature. While some researchers believe that deep learning can solve any problem, others believe that a more engineered and symbolic approach is needed to cope with complex environments with less data. Deep learning alone has achieved state-of-the-art results in many areas, from complex gameplay to predicting protein structures. In particular, in image classification and recognition, deep learning models have achieved accuracies as high as humans. But sometimes it can be very difficult to debug if the deep learning model doesn't work. Deep learning models can be vulnerable and are very sensitive to changes in data distribution. Generalization can be problematic. It's usually hard to prove why it works or doesn't. Deep learning models can also be vulnerable to adversarial attacks. Here, we combine deep learning-based object recognition and tracking with an adaptive neurosymbolic network agent, called the Non-Axiomatic Reasoning System (NARS), that can adapt to its environment by building concepts based on perceptual sequences. We achieved an improved intersection-over-union (IOU) object recognition performance of 0.65 in the adaptive retraining model compared to IOU 0.31 in the COCO data pre-trained model. We improved the object detection limits using RADAR sensors in a simulated environment, and demonstrated the weaving car detection capability by combining deep learning-based object detection and tracking with a neurosymbolic model.



## **42. Poisoning and Backdooring Contrastive Learning**

cs.LG

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2106.09667v2)

**Authors**: Nicholas Carlini, Andreas Terzis

**Abstracts**: Multimodal contrastive learning methods like CLIP train on noisy and uncurated training datasets. This is cheaper than labeling datasets manually, and even improves out-of-distribution robustness. We show that this practice makes backdoor and poisoning attacks a significant threat. By poisoning just 0.01% of a dataset (e.g., just 300 images of the 3 million-example Conceptual Captions dataset), we can cause the model to misclassify test images by overlaying a small patch. Targeted poisoning attacks, whereby the model misclassifies a particular test input with an adversarially-desired label, are even easier requiring control of 0.0001% of the dataset (e.g., just three out of the 3 million images). Our attacks call into question whether training on noisy and uncurated Internet scrapes is desirable.



## **43. Boosting Black-Box Adversarial Attacks with Meta Learning**

cs.LG

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2203.14607v1)

**Authors**: Junjie Fu, Jian Sun, Gang Wang

**Abstracts**: Deep neural networks (DNNs) have achieved remarkable success in diverse fields. However, it has been demonstrated that DNNs are very vulnerable to adversarial examples even in black-box settings. A large number of black-box attack methods have been proposed to in the literature. However, those methods usually suffer from low success rates and large query counts, which cannot fully satisfy practical purposes. In this paper, we propose a hybrid attack method which trains meta adversarial perturbations (MAPs) on surrogate models and performs black-box attacks by estimating gradients of the models. Our method uses the meta adversarial perturbation as an initialization and subsequently trains any black-box attack method for several epochs. Furthermore, the MAPs enjoy favorable transferability and universality, in the sense that they can be employed to boost performance of other black-box adversarial attack methods. Extensive experiments demonstrate that our method can not only improve the attack success rates, but also reduces the number of queries compared to other methods.



## **44. Protecting Facial Privacy: Generating Adversarial Identity Masks via Style-robust Makeup Transfer**

cs.CV

Accepted by CVPR2022. Code is available at  https://github.com/CGCL-codes/AMT-GAN

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2203.03121v2)

**Authors**: Shengshan Hu, Xiaogeng Liu, Yechao Zhang, Minghui Li, Leo Yu Zhang, Hai Jin, Libing Wu

**Abstracts**: While deep face recognition (FR) systems have shown amazing performance in identification and verification, they also arouse privacy concerns for their excessive surveillance on users, especially for public face images widely spread on social networks. Recently, some studies adopt adversarial examples to protect photos from being identified by unauthorized face recognition systems. However, existing methods of generating adversarial face images suffer from many limitations, such as awkward visual, white-box setting, weak transferability, making them difficult to be applied to protect face privacy in reality. In this paper, we propose adversarial makeup transfer GAN (AMT-GAN), a novel face protection method aiming at constructing adversarial face images that preserve stronger black-box transferability and better visual quality simultaneously. AMT-GAN leverages generative adversarial networks (GAN) to synthesize adversarial face images with makeup transferred from reference images. In particular, we introduce a new regularization module along with a joint training strategy to reconcile the conflicts between the adversarial noises and the cycle consistence loss in makeup transfer, achieving a desirable balance between the attack strength and visual changes. Extensive experiments verify that compared with state of the arts, AMT-GAN can not only preserve a comfortable visual quality, but also achieve a higher attack success rate over commercial FR APIs, including Face++, Aliyun, and Microsoft.



## **45. Practical Evaluation of Adversarial Robustness via Adaptive Auto Attack**

cs.CV

Accepted by CVPR 2022

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2203.05154v3)

**Authors**: Ye Liu, Yaya Cheng, Lianli Gao, Xianglong Liu, Qilong Zhang, Jingkuan Song

**Abstracts**: Defense models against adversarial attacks have grown significantly, but the lack of practical evaluation methods has hindered progress. Evaluation can be defined as looking for defense models' lower bound of robustness given a budget number of iterations and a test dataset. A practical evaluation method should be convenient (i.e., parameter-free), efficient (i.e., fewer iterations) and reliable (i.e., approaching the lower bound of robustness). Towards this target, we propose a parameter-free Adaptive Auto Attack (A$^3$) evaluation method which addresses the efficiency and reliability in a test-time-training fashion. Specifically, by observing that adversarial examples to a specific defense model follow some regularities in their starting points, we design an Adaptive Direction Initialization strategy to speed up the evaluation. Furthermore, to approach the lower bound of robustness under the budget number of iterations, we propose an online statistics-based discarding strategy that automatically identifies and abandons hard-to-attack images. Extensive experiments demonstrate the effectiveness of our A$^3$. Particularly, we apply A$^3$ to nearly 50 widely-used defense models. By consuming much fewer iterations than existing methods, i.e., $1/10$ on average (10$\times$ speed up), we achieve lower robust accuracy in all cases. Notably, we won $\textbf{first place}$ out of 1681 teams in CVPR 2021 White-box Adversarial Attacks on Defense Models competitions with this method. Code is available at: $\href{https://github.com/liuye6666/adaptive_auto_attack}{https://github.com/liuye6666/adaptive\_auto\_attack}$



## **46. Essential Features: Content-Adaptive Pixel Discretization to Improve Model Robustness to Adaptive Adversarial Attacks**

cs.CV

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2012.01699v3)

**Authors**: Ryan Feng, Wu-chi Feng, Atul Prakash

**Abstracts**: Preprocessing defenses such as pixel discretization are appealing to remove adversarial attacks due to their simplicity. However, they have been shown to be ineffective except on simple datasets such as MNIST. We hypothesize that existing discretization approaches failed because using a fixed codebook for the entire dataset limits their ability to balance image representation and codeword separability. We propose a per-image adaptive preprocessing defense called Essential Features, which first applies adaptive blurring to push perturbed pixel values back to their original value and then discretizes the image to an image-adaptive codebook to reduce the color space. Essential Features thus constrains the attack space by forcing the adversary to perturb large regions both locally and color-wise for its effects to survive the preprocessing. Against adaptive attacks, we find that our approach increases the $L_2$ and $L_\infty$ robustness on higher resolution datasets.



## **47. Adversarial Representation Sharing: A Quantitative and Secure Collaborative Learning Framework**

cs.CR

**SubmitDate**: 2022-03-27    [paper-pdf](http://arxiv.org/pdf/2203.14299v1)

**Authors**: Jikun Chen, Feng Qiang, Na Ruan

**Abstracts**: The performance of deep learning models highly depends on the amount of training data. It is common practice for today's data holders to merge their datasets and train models collaboratively, which yet poses a threat to data privacy. Different from existing methods such as secure multi-party computation (MPC) and federated learning (FL), we find representation learning has unique advantages in collaborative learning due to the lower communication overhead and task-independency. However, data representations face the threat of model inversion attacks. In this article, we formally define the collaborative learning scenario, and quantify data utility and privacy. Then we present ARS, a collaborative learning framework wherein users share representations of data to train models, and add imperceptible adversarial noise to data representations against reconstruction or attribute extraction attacks. By evaluating ARS in different contexts, we demonstrate that our mechanism is effective against model inversion attacks, and achieves a balance between privacy and utility. The ARS framework has wide applicability. First, ARS is valid for various data types, not limited to images. Second, data representations shared by users can be utilized in different tasks. Third, the framework can be easily extended to the vertical data partitioning scenario.



## **48. Rebuild and Ensemble: Exploring Defense Against Text Adversaries**

cs.CL

work in progress

**SubmitDate**: 2022-03-27    [paper-pdf](http://arxiv.org/pdf/2203.14207v1)

**Authors**: Linyang Li, Demin Song, Jiehang Zeng, Ruotian Ma, Xipeng Qiu

**Abstracts**: Adversarial attacks can mislead strong neural models; as such, in NLP tasks, substitution-based attacks are difficult to defend. Current defense methods usually assume that the substitution candidates are accessible, which cannot be widely applied against adversarial attacks unless knowing the mechanism of the attacks. In this paper, we propose a \textbf{Rebuild and Ensemble} Framework to defend against adversarial attacks in texts without knowing the candidates. We propose a rebuild mechanism to train a robust model and ensemble the rebuilt texts during inference to achieve good adversarial defense results. Experiments show that our method can improve accuracy under the current strong attack methods.



## **49. HINT: Hierarchical Neuron Concept Explainer**

cs.CV

Accepted by CVPR 2022

**SubmitDate**: 2022-03-27    [paper-pdf](http://arxiv.org/pdf/2203.14196v1)

**Authors**: Andong Wang, Wei-Ning Lee, Xiaojuan Qi

**Abstracts**: To interpret deep networks, one main approach is to associate neurons with human-understandable concepts. However, existing methods often ignore the inherent relationships of different concepts (e.g., dog and cat both belong to animals), and thus lose the chance to explain neurons responsible for higher-level concepts (e.g., animal). In this paper, we study hierarchical concepts inspired by the hierarchical cognition process of human beings. To this end, we propose HIerarchical Neuron concepT explainer (HINT) to effectively build bidirectional associations between neurons and hierarchical concepts in a low-cost and scalable manner. HINT enables us to systematically and quantitatively study whether and how the implicit hierarchical relationships of concepts are embedded into neurons, such as identifying collaborative neurons responsible to one concept and multimodal neurons for different concepts, at different semantic levels from concrete concepts (e.g., dog) to more abstract ones (e.g., animal). Finally, we verify the faithfulness of the associations using Weakly Supervised Object Localization, and demonstrate its applicability in various tasks such as discovering saliency regions and explaining adversarial attacks. Code is available on https://github.com/AntonotnaWang/HINT.



## **50. How to Robustify Black-Box ML Models? A Zeroth-Order Optimization Perspective**

cs.LG

Accepted as ICLR'22 Spotlight Paper

**SubmitDate**: 2022-03-27    [paper-pdf](http://arxiv.org/pdf/2203.14195v1)

**Authors**: Yimeng Zhang, Yuguang Yao, Jinghan Jia, Jinfeng Yi, Mingyi Hong, Shiyu Chang, Sijia Liu

**Abstracts**: The lack of adversarial robustness has been recognized as an important issue for state-of-the-art machine learning (ML) models, e.g., deep neural networks (DNNs). Thereby, robustifying ML models against adversarial attacks is now a major focus of research. However, nearly all existing defense methods, particularly for robust training, made the white-box assumption that the defender has the access to the details of an ML model (or its surrogate alternatives if available), e.g., its architectures and parameters. Beyond existing works, in this paper we aim to address the problem of black-box defense: How to robustify a black-box model using just input queries and output feedback? Such a problem arises in practical scenarios, where the owner of the predictive model is reluctant to share model information in order to preserve privacy. To this end, we propose a general notion of defensive operation that can be applied to black-box models, and design it through the lens of denoised smoothing (DS), a first-order (FO) certified defense technique. To allow the design of merely using model queries, we further integrate DS with the zeroth-order (gradient-free) optimization. However, a direct implementation of zeroth-order (ZO) optimization suffers a high variance of gradient estimates, and thus leads to ineffective defense. To tackle this problem, we next propose to prepend an autoencoder (AE) to a given (black-box) model so that DS can be trained using variance-reduced ZO optimization. We term the eventual defense as ZO-AE-DS. In practice, we empirically show that ZO-AE- DS can achieve improved accuracy, certified robustness, and query complexity over existing baselines. And the effectiveness of our approach is justified under both image classification and image reconstruction tasks. Codes are available at https://github.com/damon-demon/Black-Box-Defense.



