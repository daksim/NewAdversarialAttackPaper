# Latest Adversarial Attack Papers
**update at 2022-05-19 06:31:31**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. F3B: A Low-Latency Commit-and-Reveal Architecture to Mitigate Blockchain Front-Running**

cs.CR

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08529v1)

**Authors**: Haoqian Zhang, Louis-Henri Merino, Vero Estrada-Galinanes, Bryan Ford

**Abstracts**: Front-running attacks, which benefit from advanced knowledge of pending transactions, have proliferated in the cryptocurrency space since the emergence of decentralized finance. Front-running causes devastating losses to honest participants$\unicode{x2013}$estimated at \$280M each month$\unicode{x2013}$and endangers the fairness of the ecosystem. We present Flash Freezing Flash Boys (F3B), a blockchain architecture to address front-running attacks by relying on a commit-and-reveal scheme where the contents of transactions are encrypted and later revealed by a decentralized secret-management committee once the underlying consensus layer has committed the transaction. F3B mitigates front-running attacks because an adversary can no longer read the content of a transaction before commitment, thus preventing the adversary from benefiting from advance knowledge of pending transactions. We design F3B to be agnostic to the underlying consensus algorithm and compatible with legacy smart contracts by addressing front-running at the blockchain architecture level. Unlike existing commit-and-reveal approaches, F3B only requires writing data onto the underlying blockchain once, establishing a significant overhead reduction. An exploration of F3B shows that with a secret-management committee consisting of 8 and 128 members, F3B presents between 0.1 and 1.8 seconds of transaction-processing latency, respectively.



## **2. On the Privacy of Decentralized Machine Learning**

cs.CR

17 pages

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08443v1)

**Authors**: Dario Pasquini, Mathilde Raynal, Carmela Troncoso

**Abstracts**: In this work, we carry out the first, in-depth, privacy analysis of Decentralized Learning -- a collaborative machine learning framework aimed at circumventing the main limitations of federated learning. We identify the decentralized learning properties that affect users' privacy and we introduce a suite of novel attacks for both passive and active decentralized adversaries. We demonstrate that, contrary to what is claimed by decentralized learning proposers, decentralized learning does not offer any security advantages over more practical approaches such as federated learning. Rather, it tends to degrade users' privacy by increasing the attack surface and enabling any user in the system to perform powerful privacy attacks such as gradient inversion, and even gain full control over honest users' local model. We also reveal that, given the state of the art in protections, privacy-preserving configurations of decentralized learning require abandoning any possible advantage over the federated setup, completely defeating the objective of the decentralized approach.



## **3. Can You Still See Me?: Reconstructing Robot Operations Over End-to-End Encrypted Channels**

cs.CR

13 pages, 7 figures, poster presented at wisec'22

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08426v1)

**Authors**: Ryan Shah, Chuadhry Mujeeb Ahmed, Shishir Nagaraja

**Abstracts**: Connected robots play a key role in Industry 4.0, providing automation and higher efficiency for many industrial workflows. Unfortunately, these robots can leak sensitive information regarding these operational workflows to remote adversaries. While there exists mandates for the use of end-to-end encryption for data transmission in such settings, it is entirely possible for passive adversaries to fingerprint and reconstruct entire workflows being carried out -- establishing an understanding of how facilities operate. In this paper, we investigate whether a remote attacker can accurately fingerprint robot movements and ultimately reconstruct operational workflows. Using a neural network approach to traffic analysis, we find that one can predict TLS-encrypted movements with around \textasciitilde60\% accuracy, increasing to near-perfect accuracy under realistic network conditions. Further, we also find that attackers can reconstruct warehousing workflows with similar success. Ultimately, simply adopting best cybersecurity practices is clearly not enough to stop even weak (passive) adversaries.



## **4. Bankrupting DoS Attackers Despite Uncertainty**

cs.CR

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08287v1)

**Authors**: Trisha Chakraborty, Abir Islam, Valerie King, Daniel Rayborn, Jared Saia, Maxwell Young

**Abstracts**: On-demand provisioning in the cloud allows for services to remain available despite massive denial-of-service (DoS) attacks. Unfortunately, on-demand provisioning is expensive and must be weighed against the costs incurred by an adversary. This leads to a recent threat known as economic denial-of-sustainability (EDoS), where the cost for defending a service is higher than that of attacking.   A natural approach for combating EDoS is to impose costs via resource burning (RB). Here, a client must verifiably consume resources -- for example, by solving a computational challenge -- before service is rendered. However, prior approaches with security guarantees do not account for the cost on-demand provisioning.   Another valuable defensive tool is to use a classifier in order to discern good jobs from a legitimate client, versus bad jobs from the adversary. However, while useful, uncertainty arises from classification error, which still allows bad jobs to consume server resources. Thus, classification is not a solution by itself.   Here, we propose an EDoS defense, RootDef, that leverages both RB and classification, while accounting for both the costs of resource burning and on-demand provisioning. Specifically, against an adversary that expends $B$ resources to attack, the total cost for defending is $\tilde{O}( \sqrt{B\,g} + B^{2/3} + g)$, where $g$ is the number of good jobs and $\tilde{O}$ refers to hidden logarithmic factors in the total number of jobs $n$. Notably, for large $B$ relative to $g$, the adversary has higher cost, implying that the algorithm has an economic advantage. Finally, we prove a lower bound showing that RootDef has total costs that are asymptotically tight up to logarithmic factors in $n$.



## **5. How Not to Handle Keys: Timing Attacks on FIDO Authenticator Privacy**

cs.CR

to be published in the 22nd Privacy Enhancing Technologies Symposium  (PETS 2022)

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08071v1)

**Authors**: Michal Kepkowski, Lucjan Hanzlik, Ian Wood, Mohamed Ali Kaafar

**Abstracts**: This paper presents a timing attack on the FIDO2 (Fast IDentity Online) authentication protocol that allows attackers to link user accounts stored in vulnerable authenticators, a serious privacy concern. FIDO2 is a new standard specified by the FIDO industry alliance for secure token online authentication. It complements the W3C WebAuthn specification by providing means to use a USB token or other authenticator as a second factor during the authentication process. From a cryptographic perspective, the protocol is a simple challenge-response where the elliptic curve digital signature algorithm is used to sign challenges. To protect the privacy of the user the token uses unique key pairs per service. To accommodate for small memory, tokens use various techniques that make use of a special parameter called a key handle sent by the service to the token. We identify and analyse a vulnerability in the way the processing of key handles is implemented that allows attackers to remotely link user accounts on multiple services. We show that for vulnerable authenticators there is a difference between the time it takes to process a key handle for a different service but correct authenticator, and for a different authenticator but correct service. This difference can be used to perform a timing attack allowing an adversary to link user's accounts across services. We present several real world examples of adversaries that are in a position to execute our attack and can benefit from linking accounts. We found that two of the eight hardware authenticators we tested were vulnerable despite FIDO level 1 certification. This vulnerability cannot be easily mitigated on authenticators because, for security reasons, they usually do not allow firmware updates. In addition, we show that due to the way existing browsers implement the WebAuthn standard, the attack can be executed remotely.



## **6. User-Level Differential Privacy against Attribute Inference Attack of Speech Emotion Recognition in Federated Learning**

cs.CR

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2204.02500v2)

**Authors**: Tiantian Feng, Raghuveer Peri, Shrikanth Narayanan

**Abstracts**: Many existing privacy-enhanced speech emotion recognition (SER) frameworks focus on perturbing the original speech data through adversarial training within a centralized machine learning setup. However, this privacy protection scheme can fail since the adversary can still access the perturbed data. In recent years, distributed learning algorithms, especially federated learning (FL), have gained popularity to protect privacy in machine learning applications. While FL provides good intuition to safeguard privacy by keeping the data on local devices, prior work has shown that privacy attacks, such as attribute inference attacks, are achievable for SER systems trained using FL. In this work, we propose to evaluate the user-level differential privacy (UDP) in mitigating the privacy leaks of the SER system in FL. UDP provides theoretical privacy guarantees with privacy parameters $\epsilon$ and $\delta$. Our results show that the UDP can effectively decrease attribute information leakage while keeping the utility of the SER system with the adversary accessing one model update. However, the efficacy of the UDP suffers when the FL system leaks more model updates to the adversary. We make the code publicly available to reproduce the results in https://github.com/usc-sail/fed-ser-leakage.



## **7. RoVISQ: Reduction of Video Service Quality via Adversarial Attacks on Deep Learning-based Video Compression**

cs.CV

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2203.10183v2)

**Authors**: Jung-Woo Chang, Mojan Javaheripi, Seira Hidano, Farinaz Koushanfar

**Abstracts**: Video compression plays a crucial role in video streaming and classification systems by maximizing the end-user quality of experience (QoE) at a given bandwidth budget. In this paper, we conduct the first systematic study for adversarial attacks on deep learning-based video compression and downstream classification systems. Our attack framework, dubbed RoVISQ, manipulates the Rate-Distortion (R-D) relationship of a video compression model to achieve one or both of the following goals: (1) increasing the network bandwidth, (2) degrading the video quality for end-users. We further devise new objectives for targeted and untargeted attacks to a downstream video classification service. Finally, we design an input-invariant perturbation that universally disrupts video compression and classification systems in real time. Unlike previously proposed attacks on video classification, our adversarial perturbations are the first to withstand compression. We empirically show the resilience of RoVISQ attacks against various defenses, i.e., adversarial training, video denoising, and JPEG compression. Our extensive experimental results on various video datasets show RoVISQ attacks deteriorate peak signal-to-noise ratio by up to 5.6dB and the bit-rate by up to 2.4 times while achieving over 90% attack success rate on a downstream classifier.



## **8. Classification Auto-Encoder based Detector against Diverse Data Poisoning Attacks**

cs.LG

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2108.04206v2)

**Authors**: Fereshteh Razmi, Li Xiong

**Abstracts**: Poisoning attacks are a category of adversarial machine learning threats in which an adversary attempts to subvert the outcome of the machine learning systems by injecting crafted data into training data set, thus increasing the machine learning model's test error. The adversary can tamper with the data feature space, data labels, or both, each leading to a different attack strategy with different strengths. Various detection approaches have recently emerged, each focusing on one attack strategy. The Achilles heel of many of these detection approaches is their dependence on having access to a clean, untampered data set. In this paper, we propose CAE, a Classification Auto-Encoder based detector against diverse poisoned data. CAE can detect all forms of poisoning attacks using a combination of reconstruction and classification errors without having any prior knowledge of the attack strategy. We show that an enhanced version of CAE (called CAE+) does not have to employ a clean data set to train the defense model. Our experimental results on three real datasets MNIST, Fashion-MNIST and CIFAR demonstrate that our proposed method can maintain its functionality under up to 30% contaminated data and help the defended SVM classifier to regain its best accuracy.



## **9. Transferability of Adversarial Attacks on Synthetic Speech Detection**

cs.SD

5 pages, submit to Interspeech2022

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07711v1)

**Authors**: Jiacheng Deng, Shunyi Chen, Li Dong, Diqun Yan, Rangding Wang

**Abstracts**: Synthetic speech detection is one of the most important research problems in audio security. Meanwhile, deep neural networks are vulnerable to adversarial attacks. Therefore, we establish a comprehensive benchmark to evaluate the transferability of adversarial attacks on the synthetic speech detection task. Specifically, we attempt to investigate: 1) The transferability of adversarial attacks between different features. 2) The influence of varying extraction hyperparameters of features on the transferability of adversarial attacks. 3) The effect of clipping or self-padding operation on the transferability of adversarial attacks. By performing these analyses, we summarise the weaknesses of synthetic speech detectors and the transferability behaviours of adversarial attacks, which provide insights for future research. More details can be found at https://gitee.com/djc_QRICK/Attack-Transferability-On-Synthetic-Detection.



## **10. SemAttack: Natural Textual Attacks via Different Semantic Spaces**

cs.CL

Published at Findings of NAACL 2022

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.01287v2)

**Authors**: Boxin Wang, Chejian Xu, Xiangyu Liu, Yu Cheng, Bo Li

**Abstracts**: Recent studies show that pre-trained language models (LMs) are vulnerable to textual adversarial attacks. However, existing attack methods either suffer from low attack success rates or fail to search efficiently in the exponentially large perturbation space. We propose an efficient and effective framework SemAttack to generate natural adversarial text by constructing different semantic perturbation functions. In particular, SemAttack optimizes the generated perturbations constrained on generic semantic spaces, including typo space, knowledge space (e.g., WordNet), contextualized semantic space (e.g., the embedding space of BERT clusterings), or the combination of these spaces. Thus, the generated adversarial texts are more semantically close to the original inputs. Extensive experiments reveal that state-of-the-art (SOTA) large-scale LMs (e.g., DeBERTa-v2) and defense strategies (e.g., FreeLB) are still vulnerable to SemAttack. We further demonstrate that SemAttack is general and able to generate natural adversarial texts for different languages (e.g., English and Chinese) with high attack success rates. Human evaluations also confirm that our generated adversarial texts are natural and barely affect human performance. Our code is publicly available at https://github.com/AI-secure/SemAttack.



## **11. SGBA: A Stealthy Scapegoat Backdoor Attack against Deep Neural Networks**

cs.CR

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2104.01026v3)

**Authors**: Ying He, Zhili Shen, Chang Xia, Jingyu Hua, Wei Tong, Sheng Zhong

**Abstracts**: Outsourced deep neural networks have been demonstrated to suffer from patch-based trojan attacks, in which an adversary poisons the training sets to inject a backdoor in the obtained model so that regular inputs can be still labeled correctly while those carrying a specific trigger are falsely given a target label. Due to the severity of such attacks, many backdoor detection and containment systems have recently, been proposed for deep neural networks. One major category among them are various model inspection schemes, which hope to detect backdoors before deploying models from non-trusted third-parties. In this paper, we show that such state-of-the-art schemes can be defeated by a so-called Scapegoat Backdoor Attack, which introduces a benign scapegoat trigger in data poisoning to prevent the defender from reversing the real abnormal trigger. In addition, it confines the values of network parameters within the same variances of those from clean model during training, which further significantly enhances the difficulty of the defender to learn the differences between legal and illegal models through machine-learning approaches. Our experiments on 3 popular datasets show that it can escape detection by all five state-of-the-art model inspection schemes. Moreover, this attack brings almost no side-effects on the attack effectiveness and guarantees the universal feature of the trigger compared with original patch-based trojan attacks.



## **12. Unreasonable Effectiveness of Last Hidden Layer Activations for Adversarial Robustness**

cs.LG

IEEE COMPSAC 2022 publication full version

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2202.07342v2)

**Authors**: Omer Faruk Tuna, Ferhat Ozgur Catak, M. Taner Eskil

**Abstracts**: In standard Deep Neural Network (DNN) based classifiers, the general convention is to omit the activation function in the last (output) layer and directly apply the softmax function on the logits to get the probability scores of each class. In this type of architectures, the loss value of the classifier against any output class is directly proportional to the difference between the final probability score and the label value of the associated class. Standard White-box adversarial evasion attacks, whether targeted or untargeted, mainly try to exploit the gradient of the model loss function to craft adversarial samples and fool the model. In this study, we show both mathematically and experimentally that using some widely known activation functions in the output layer of the model with high temperature values has the effect of zeroing out the gradients for both targeted and untargeted attack cases, preventing attackers from exploiting the model's loss function to craft adversarial samples. We've experimentally verified the efficacy of our approach on MNIST (Digit), CIFAR10 datasets. Detailed experiments confirmed that our approach substantially improves robustness against gradient-based targeted and untargeted attack threats. And, we showed that the increased non-linearity at the output layer has some additional benefits against some other attack methods like Deepfool attack.



## **13. Attacking and Defending Deep Reinforcement Learning Policies**

cs.LG

nine pages

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07626v1)

**Authors**: Chao Wang

**Abstracts**: Recent studies have shown that deep reinforcement learning (DRL) policies are vulnerable to adversarial attacks, which raise concerns about applications of DRL to safety-critical systems. In this work, we adopt a principled way and study the robustness of DRL policies to adversarial attacks from the perspective of robust optimization. Within the framework of robust optimization, optimal adversarial attacks are given by minimizing the expected return of the policy, and correspondingly a good defense mechanism should be realized by improving the worst-case performance of the policy. Considering that attackers generally have no access to the training environment, we propose a greedy attack algorithm, which tries to minimize the expected return of the policy without interacting with the environment, and a defense algorithm, which performs adversarial training in a max-min form. Experiments on Atari game environments show that our attack algorithm is more effective and leads to worse return of the policy than existing attack algorithms, and our defense algorithm yields policies more robust than existing defense methods to a range of adversarial attacks (including our proposed attack algorithm).



## **14. More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks**

cs.CR

17 pages, 13 figures

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2202.03195v2)

**Authors**: Jing Xu, Rui Wang, Stefanos Koffas, Kaitai Liang, Stjepan Picek

**Abstracts**: Graph Neural Networks (GNNs) are a class of deep learning-based methods for processing graph domain information. GNNs have recently become a widely used graph analysis method due to their superior ability to learn representations for complex graph data. However, due to privacy concerns and regulation restrictions, centralized GNNs can be difficult to apply to data-sensitive scenarios. Federated learning (FL) is an emerging technology developed for privacy-preserving settings when several parties need to train a shared global model collaboratively. Although several research works have applied FL to train GNNs (Federated GNNs), there is no research on their robustness to backdoor attacks.   This paper bridges this gap by conducting two types of backdoor attacks in Federated GNNs: centralized backdoor attacks (CBA) and distributed backdoor attacks (DBA). CBA is conducted by embedding the same global trigger during training for every malicious party, while DBA is conducted by decomposing a global trigger into separate local triggers and embedding them into the training datasets of different malicious parties, respectively. Our experiments show that the DBA attack success rate is higher than CBA in almost all evaluated cases. For CBA, the attack success rate of all local triggers is similar to the global trigger even if the training set of the adversarial party is embedded with the global trigger. To further explore the properties of two backdoor attacks in Federated GNNs, we evaluate the attack performance for different number of clients, trigger sizes, poisoning intensities, and trigger densities. Moreover, we explore the robustness of DBA and CBA against two state-of-the-art defenses. We find that both attacks are robust against the investigated defenses, necessitating the need to consider backdoor attacks in Federated GNNs as a novel threat that requires custom defenses.



## **15. Learning Classical Readout Quantum PUFs based on single-qubit gates**

quant-ph

12 pages, 9 figures

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2112.06661v2)

**Authors**: Niklas Pirnay, Anna Pappa, Jean-Pierre Seifert

**Abstracts**: Physical Unclonable Functions (PUFs) have been proposed as a way to identify and authenticate electronic devices. Recently, several ideas have been presented that aim to achieve the same for quantum devices. Some of these constructions apply single-qubit gates in order to provide a secure fingerprint of the quantum device. In this work, we formalize the class of Classical Readout Quantum PUFs (CR-QPUFs) using the statistical query (SQ) model and explicitly show insufficient security for CR-QPUFs based on single qubit rotation gates, when the adversary has SQ access to the CR-QPUF. We demonstrate how a malicious party can learn the CR-QPUF characteristics and forge the signature of a quantum device through a modelling attack using a simple regression of low-degree polynomials. The proposed modelling attack was successfully implemented in a real-world scenario on real IBM Q quantum machines. We thoroughly discuss the prospects and problems of CR-QPUFs where quantum device imperfections are used as a secure fingerprint.



## **16. Manifold Characteristics That Predict Downstream Task Performance**

cs.LG

Currently under review

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07477v1)

**Authors**: Ruan van der Merwe, Gregory Newman, Etienne Barnard

**Abstracts**: Pretraining methods are typically compared by evaluating the accuracy of linear classifiers, transfer learning performance, or visually inspecting the representation manifold's (RM) lower-dimensional projections. We show that the differences between methods can be understood more clearly by investigating the RM directly, which allows for a more detailed comparison. To this end, we propose a framework and new metric to measure and compare different RMs. We also investigate and report on the RM characteristics for various pretraining methods. These characteristics are measured by applying sequentially larger local alterations to the input data, using white noise injections and Projected Gradient Descent (PGD) adversarial attacks, and then tracking each datapoint. We calculate the total distance moved for each datapoint and the relative change in distance between successive alterations. We show that self-supervised methods learn an RM where alterations lead to large but constant size changes, indicating a smoother RM than fully supervised methods. We then combine these measurements into one metric, the Representation Manifold Quality Metric (RMQM), where larger values indicate larger and less variable step sizes, and show that RMQM correlates positively with performance on downstream tasks.



## **17. Robust Representation via Dynamic Feature Aggregation**

cs.CV

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07466v1)

**Authors**: Haozhe Liu, Haoqin Ji, Yuexiang Li, Nanjun He, Haoqian Wu, Feng Liu, Linlin Shen, Yefeng Zheng

**Abstracts**: Deep convolutional neural network (CNN) based models are vulnerable to the adversarial attacks. One of the possible reasons is that the embedding space of CNN based model is sparse, resulting in a large space for the generation of adversarial samples. In this study, we propose a method, denoted as Dynamic Feature Aggregation, to compress the embedding space with a novel regularization. Particularly, the convex combination between two samples are regarded as the pivot for aggregation. In the embedding space, the selected samples are guided to be similar to the representation of the pivot. On the other side, to mitigate the trivial solution of such regularization, the last fully-connected layer of the model is replaced by an orthogonal classifier, in which the embedding codes for different classes are processed orthogonally and separately. With the regularization and orthogonal classifier, a more compact embedding space can be obtained, which accordingly improves the model robustness against adversarial attacks. An averaging accuracy of 56.91% is achieved by our method on CIFAR-10 against various attack methods, which significantly surpasses a solid baseline (Mixup) by a margin of 37.31%. More surprisingly, empirical results show that, the proposed method can also achieve the state-of-the-art performance for out-of-distribution (OOD) detection, due to the learned compact feature space. An F1 score of 0.937 is achieved by the proposed method, when adopting CIFAR-10 as in-distribution (ID) dataset and LSUN as OOD dataset. Code is available at https://github.com/HaozheLiu-ST/DynamicFeatureAggregation.



## **18. Diffusion Models for Adversarial Purification**

cs.LG

ICML 2022

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07460v1)

**Authors**: Weili Nie, Brandon Guo, Yujia Huang, Chaowei Xiao, Arash Vahdat, Anima Anandkumar

**Abstracts**: Adversarial purification refers to a class of defense methods that remove adversarial perturbations using a generative model. These methods do not make assumptions on the form of attack and the classification model, and thus can defend pre-existing classifiers against unseen threats. However, their performance currently falls behind adversarial training methods. In this work, we propose DiffPure that uses diffusion models for adversarial purification: Given an adversarial example, we first diffuse it with a small amount of noise following a forward diffusion process, and then recover the clean image through a reverse generative process. To evaluate our method against strong adaptive attacks in an efficient and scalable way, we propose to use the adjoint method to compute full gradients of the reverse generative process. Extensive experiments on three image datasets including CIFAR-10, ImageNet and CelebA-HQ with three classifier architectures including ResNet, WideResNet and ViT demonstrate that our method achieves the state-of-the-art results, outperforming current adversarial training and adversarial purification methods, often by a large margin. Project page: https://diffpure.github.io.



## **19. Trustworthy Graph Neural Networks: Aspects, Methods and Trends**

cs.LG

36 pages, 7 tables, 4 figures

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07424v1)

**Authors**: He Zhang, Bang Wu, Xingliang Yuan, Shirui Pan, Hanghang Tong, Jian Pei

**Abstracts**: Graph neural networks (GNNs) have emerged as a series of competent graph learning methods for diverse real-world scenarios, ranging from daily applications like recommendation systems and question answering to cutting-edge technologies such as drug discovery in life sciences and n-body simulation in astrophysics. However, task performance is not the only requirement for GNNs. Performance-oriented GNNs have exhibited potential adverse effects like vulnerability to adversarial attacks, unexplainable discrimination against disadvantaged groups, or excessive resource consumption in edge computing environments. To avoid these unintentional harms, it is necessary to build competent GNNs characterised by trustworthiness. To this end, we propose a comprehensive roadmap to build trustworthy GNNs from the view of the various computing technologies involved. In this survey, we introduce basic concepts and comprehensively summarise existing efforts for trustworthy GNNs from six aspects, including robustness, explainability, privacy, fairness, accountability, and environmental well-being. Additionally, we highlight the intricate cross-aspect relations between the above six aspects of trustworthy GNNs. Finally, we present a thorough overview of trending directions for facilitating the research and industrialisation of trustworthy GNNs.



## **20. Parameter Adaptation for Joint Distribution Shifts**

cs.LG

**SubmitDate**: 2022-05-15    [paper-pdf](http://arxiv.org/pdf/2205.07315v1)

**Authors**: Siddhartha Datta

**Abstracts**: While different methods exist to tackle distinct types of distribution shift, such as label shift (in the form of adversarial attacks) or domain shift, tackling the joint shift setting is still an open problem. Through the study of a joint distribution shift manifesting both adversarial and domain-specific perturbations, we not only show that a joint shift worsens model performance compared to their individual shifts, but that the use of a similar domain worsens performance than a dissimilar domain. To curb the performance drop, we study the use of perturbation sets motivated by input and parameter space bounds, and adopt a meta learning strategy (hypernetworks) to model parameters w.r.t. test-time inputs to recover performance.



## **21. CE-based white-box adversarial attacks will not work using super-fitting**

cs.LG

**SubmitDate**: 2022-05-15    [paper-pdf](http://arxiv.org/pdf/2205.02741v2)

**Authors**: Youhuan Yang, Lei Sun, Leyu Dai, Song Guo, Xiuqing Mao, Xiaoqin Wang, Bayi Xu

**Abstracts**: Deep neural networks are widely used in various fields because of their powerful performance. However, recent studies have shown that deep learning models are vulnerable to adversarial attacks, i.e., adding a slight perturbation to the input will make the model obtain wrong results. This is especially dangerous for some systems with high-security requirements, so this paper proposes a new defense method by using the model super-fitting state to improve the model's adversarial robustness (i.e., the accuracy under adversarial attacks). This paper mathematically proves the effectiveness of super-fitting and enables the model to reach this state quickly by minimizing unrelated category scores (MUCS). Theoretically, super-fitting can resist any existing (even future) CE-based white-box adversarial attacks. In addition, this paper uses a variety of powerful attack algorithms to evaluate the adversarial robustness of super-fitting, and the proposed method is compared with nearly 50 defense models from recent conferences. The experimental results show that the super-fitting method in this paper can make the trained model obtain the highest adversarial robustness.



## **22. Measuring Vulnerabilities of Malware Detectors with Explainability-Guided Evasion Attacks**

cs.CR

**SubmitDate**: 2022-05-15    [paper-pdf](http://arxiv.org/pdf/2111.10085v3)

**Authors**: Ruoxi Sun, Wei Wang, Tian Dong, Shaofeng Li, Minhui Xue, Gareth Tyson, Haojin Zhu, Mingyu Guo, Surya Nepal

**Abstracts**: Numerous open-source and commercial malware detectors are available. However, their efficacy is threatened by new adversarial attacks, whereby malware attempts to evade detection, e.g., by performing feature-space manipulation. In this work, we propose an explainability-guided and model-agnostic framework for measuring the efficacy of malware detectors when confronted with adversarial attacks. The framework introduces the concept of Accrued Malicious Magnitude (AMM) to identify which malware features should be manipulated to maximize the likelihood of evading detection. We then use this framework to test several state-of-the-art malware detectors' ability to detect manipulated malware. We find that (i) commercial antivirus engines are vulnerable to AMM-guided manipulated samples; (ii) the ability of a manipulated malware generated using one detector to evade detection by another detector (i.e., transferability) depends on the overlap of features with large AMM values between the different detectors; and (iii) AMM values effectively measure the importance of features and explain the ability to evade detection. Our findings shed light on the weaknesses of current malware detectors, as well as how they can be improved.



## **23. Unsupervised Abnormal Traffic Detection through Topological Flow Analysis**

cs.LG

**SubmitDate**: 2022-05-14    [paper-pdf](http://arxiv.org/pdf/2205.07109v1)

**Authors**: Paul Irofti, Andrei Pătraşcu, Andrei Iulian Hîji

**Abstracts**: Cyberthreats are a permanent concern in our modern technological world. In the recent years, sophisticated traffic analysis techniques and anomaly detection (AD) algorithms have been employed to face the more and more subversive adversarial attacks. A malicious intrusion, defined as an invasive action intending to illegally exploit private resources, manifests through unusual data traffic and/or abnormal connectivity pattern. Despite the plethora of statistical or signature-based detectors currently provided in the literature, the topological connectivity component of a malicious flow is less exploited. Furthermore, a great proportion of the existing statistical intrusion detectors are based on supervised learning, that relies on labeled data. By viewing network flows as weighted directed interactions between a pair of nodes, in this paper we present a simple method that facilitate the use of connectivity graph features in unsupervised anomaly detection algorithms. We test our methodology on real network traffic datasets and observe several improvements over standard AD.



## **24. Learning Coated Adversarial Camouflages for Object Detectors**

cs.CV

**SubmitDate**: 2022-05-14    [paper-pdf](http://arxiv.org/pdf/2109.00124v3)

**Authors**: Yexin Duan, Jialin Chen, Xingyu Zhou, Junhua Zou, Zhengyun He, Jin Zhang, Wu Zhang, Zhisong Pan

**Abstracts**: An adversary can fool deep neural network object detectors by generating adversarial noises. Most of the existing works focus on learning local visible noises in an adversarial "patch" fashion. However, the 2D patch attached to a 3D object tends to suffer from an inevitable reduction in attack performance as the viewpoint changes. To remedy this issue, this work proposes the Coated Adversarial Camouflage (CAC) to attack the detectors in arbitrary viewpoints. Unlike the patch trained in the 2D space, our camouflage generated by a conceptually different training framework consists of 3D rendering and dense proposals attack. Specifically, we make the camouflage perform 3D spatial transformations according to the pose changes of the object. Based on the multi-view rendering results, the top-n proposals of the region proposal network are fixed, and all the classifications in the fixed dense proposals are attacked simultaneously to output errors. In addition, we build a virtual 3D scene to fairly and reproducibly evaluate different attacks. Extensive experiments demonstrate the superiority of CAC over the existing attacks, and it shows impressive performance both in the virtual scene and the real world. This poses a potential threat to the security-critical computer vision systems.



## **25. Rethinking Classifier and Adversarial Attack**

cs.LG

**SubmitDate**: 2022-05-14    [paper-pdf](http://arxiv.org/pdf/2205.02743v2)

**Authors**: Youhuan Yang, Lei Sun, Leyu Dai, Song Guo, Xiuqing Mao, Xiaoqin Wang, Bayi Xu

**Abstracts**: Various defense models have been proposed to resist adversarial attack algorithms, but existing adversarial robustness evaluation methods always overestimate the adversarial robustness of these models (i.e., not approaching the lower bound of robustness). To solve this problem, this paper uses the proposed decouple space method to divide the classifier into two parts: non-linear and linear. Then, this paper defines the representation vector of the original example (and its space, i.e., the representation space) and uses the iterative optimization of Absolute Classification Boundaries Initialization (ACBI) to obtain a better attack starting point. Particularly, this paper applies ACBI to nearly 50 widely-used defense models (including 8 architectures). Experimental results show that ACBI achieves lower robust accuracy in all cases.



## **26. Evaluating Membership Inference Through Adversarial Robustness**

cs.CR

Accepted by The Computer Journal. Pre-print version

**SubmitDate**: 2022-05-14    [paper-pdf](http://arxiv.org/pdf/2205.06986v1)

**Authors**: Zhaoxi Zhang, Leo Yu Zhang, Xufei Zheng, Bilal Hussain Abbasi, Shengshan Hu

**Abstracts**: The usage of deep learning is being escalated in many applications. Due to its outstanding performance, it is being used in a variety of security and privacy-sensitive areas in addition to conventional applications. One of the key aspects of deep learning efficacy is to have abundant data. This trait leads to the usage of data which can be highly sensitive and private, which in turn causes wariness with regard to deep learning in the general public. Membership inference attacks are considered lethal as they can be used to figure out whether a piece of data belongs to the training dataset or not. This can be problematic with regards to leakage of training data information and its characteristics. To highlight the significance of these types of attacks, we propose an enhanced methodology for membership inference attacks based on adversarial robustness, by adjusting the directions of adversarial perturbations through label smoothing under a white-box setting. We evaluate our proposed method on three datasets: Fashion-MNIST, CIFAR-10, and CIFAR-100. Our experimental results reveal that the performance of our method surpasses that of the existing adversarial robustness-based method when attacking normally trained models. Additionally, through comparing our technique with the state-of-the-art metric-based membership inference methods, our proposed method also shows better performance when attacking adversarially trained models. The code for reproducing the results of this work is available at \url{https://github.com/plll4zzx/Evaluating-Membership-Inference-Through-Adversarial-Robustness}.



## **27. Universal Post-Training Backdoor Detection**

cs.LG

**SubmitDate**: 2022-05-13    [paper-pdf](http://arxiv.org/pdf/2205.06900v1)

**Authors**: Hang Wang, Zhen Xiang, David J. Miller, George Kesidis

**Abstracts**: A Backdoor attack (BA) is an important type of adversarial attack against deep neural network classifiers, wherein test samples from one or more source classes will be (mis)classified to the attacker's target class when a backdoor pattern (BP) is embedded. In this paper, we focus on the post-training backdoor defense scenario commonly considered in the literature, where the defender aims to detect whether a trained classifier was backdoor attacked, without any access to the training set. To the best of our knowledge, existing post-training backdoor defenses are all designed for BAs with presumed BP types, where each BP type has a specific embedding function. They may fail when the actual BP type used by the attacker (unknown to the defender) is different from the BP type assumed by the defender. In contrast, we propose a universal post-training defense that detects BAs with arbitrary types of BPs, without making any assumptions about the BP type. Our detector leverages the influence of the BA, independently of the BP type, on the landscape of the classifier's outputs prior to the softmax layer. For each class, a maximum margin statistic is estimated using a set of random vectors; detection inference is then performed by applying an unsupervised anomaly detector to these statistics. Thus, our detector is also an advance relative to most existing post-training methods by not needing any legitimate clean samples, and can efficiently detect BAs with arbitrary numbers of source classes. These advantages of our detector over several state-of-the-art methods are demonstrated on four datasets, for three different types of BPs, and for a variety of attack configurations. Finally, we propose a novel, general approach for BA mitigation once a detection is made.



## **28. secml: A Python Library for Secure and Explainable Machine Learning**

cs.LG

Accepted for publication to SoftwareX. Published version can be found  at: https://doi.org/10.1016/j.softx.2022.101095

**SubmitDate**: 2022-05-13    [paper-pdf](http://arxiv.org/pdf/1912.10013v2)

**Authors**: Maura Pintor, Luca Demetrio, Angelo Sotgiu, Marco Melis, Ambra Demontis, Battista Biggio

**Abstracts**: We present \texttt{secml}, an open-source Python library for secure and explainable machine learning. It implements the most popular attacks against machine learning, including test-time evasion attacks to generate adversarial examples against deep neural networks and training-time poisoning attacks against support vector machines and many other algorithms. These attacks enable evaluating the security of learning algorithms and the corresponding defenses under both white-box and black-box threat models. To this end, \texttt{secml} provides built-in functions to compute security evaluation curves, showing how quickly classification performance decreases against increasing adversarial perturbations of the input data. \texttt{secml} also includes explainability methods to help understand why adversarial attacks succeed against a given model, by visualizing the most influential features and training prototypes contributing to each decision. It is distributed under the Apache License 2.0 and hosted at \url{https://github.com/pralab/secml}.



## **29. Privacy Preserving Release of Mobile Sensor Data**

cs.CR

12 pages, 10 figures, 1 table

**SubmitDate**: 2022-05-13    [paper-pdf](http://arxiv.org/pdf/2205.06641v1)

**Authors**: Rahat Masood, Wing Yan Cheng, Dinusha Vatsalan, Deepak Mishra, Hassan Jameel Asghar, Mohamed Ali Kaafar

**Abstracts**: Sensors embedded in mobile smart devices can monitor users' activity with high accuracy to provide a variety of services to end-users ranging from precise geolocation, health monitoring, and handwritten word recognition. However, this involves the risk of accessing and potentially disclosing sensitive information of individuals to the apps that may lead to privacy breaches. In this paper, we aim to minimize privacy leakages that may lead to user identification on mobile devices through user tracking and distinguishability while preserving the functionality of apps and services. We propose a privacy-preserving mechanism that effectively handles the sensor data fluctuations (e.g., inconsistent sensor readings while walking, sitting, and running at different times) by formulating the data as time-series modeling and forecasting. The proposed mechanism also uses the notion of correlated noise-series against noise filtering attacks from an adversary, which aims to filter out the noise from the perturbed data to re-identify the original data. Unlike existing solutions, our mechanism keeps running in isolation without the interaction of a user or a service provider. We perform rigorous experiments on benchmark datasets and show that our proposed mechanism limits user tracking and distinguishability threats to a significant extent compared to the original data while maintaining a reasonable level of utility of functionalities. In general, we show that our obfuscation mechanism reduces the user trackability threat by 60\% across all the datasets while maintaining the utility loss below 0.5 Mean Absolute Error (MAE). We also observe that our mechanism is more effective in large datasets. For example, with the Swipes dataset, the distinguishability risk is reduced by 60\% on average while the utility loss is below 0.5 MAE.



## **30. Authentication Attacks on Projection-based Cancelable Biometric Schemes (long version)**

cs.CR

arXiv admin note: text overlap with arXiv:1910.01389 by other authors

**SubmitDate**: 2022-05-13    [paper-pdf](http://arxiv.org/pdf/2110.15163v5)

**Authors**: Axel Durbet, Pascal Lafourcade, Denis Migdal, Kevin Thiry-Atighehchi, Paul-Marie Grollemund

**Abstracts**: Cancelable biometric schemes aim at generating secure biometric templates by combining user specific tokens, such as password, stored secret or salt, along with biometric data. This type of transformation is constructed as a composition of a biometric transformation with a feature extraction algorithm. The security requirements of cancelable biometric schemes concern the irreversibility, unlinkability and revocability of templates, without losing in accuracy of comparison. While several schemes were recently attacked regarding these requirements, full reversibility of such a composition in order to produce colliding biometric characteristics, and specifically presentation attacks, were never demonstrated to the best of our knowledge. In this paper, we formalize these attacks for a traditional cancelable scheme with the help of integer linear programming (ILP) and quadratically constrained quadratic programming (QCQP). Solving these optimization problems allows an adversary to slightly alter its fingerprint image in order to impersonate any individual. Moreover, in an even more severe scenario, it is possible to simultaneously impersonate several individuals.



## **31. Uncertify: Attacks Against Neural Network Certification**

cs.LG

**SubmitDate**: 2022-05-13    [paper-pdf](http://arxiv.org/pdf/2108.11299v3)

**Authors**: Tobias Lorenz, Marta Kwiatkowska, Mario Fritz

**Abstracts**: A key concept towards reliable, robust, and safe AI systems is the idea to implement fallback strategies when predictions of the AI cannot be trusted. Certifiers for neural networks have made great progress towards provable robustness guarantees against evasion attacks using adversarial examples. These methods guarantee for some predictions that a certain class of manipulations or attacks could not have changed the outcome. For the remaining predictions without guarantees, the method abstains from making a prediction and a fallback strategy needs to be invoked, which is typically more costly, less accurate, or even involves a human operator. While this is a key concept towards safe and secure AI, we show for the first time that this strategy comes with its own security risks, as such fallback strategies can be deliberately triggered by an adversary. In particular, we conduct the first systematic analysis of training-time attacks against certifiers in practical application pipelines, identifying new threat vectors that can be exploited to degrade the overall system. Using these insights, we design two backdoor attacks against network certifiers, which can drastically reduce certified robustness. For example, adding 1% poisoned data during training is sufficient to reduce certified robustness by up to 95 percentage points, effectively rendering the certifier useless. We analyze how such novel attacks can compromise the overall system's integrity or availability. Our extensive experiments across multiple datasets, model architectures, and certifiers demonstrate the wide applicability of these attacks. A first investigation into potential defenses shows that current approaches are insufficient to mitigate the issue, highlighting the need for new, more specific solutions.



## **32. Millimeter-Wave Automotive Radar Spoofing**

cs.CR

**SubmitDate**: 2022-05-13    [paper-pdf](http://arxiv.org/pdf/2205.06567v1)

**Authors**: Mihai Ordean, Flavio D. Garcia

**Abstracts**: Millimeter-wave radar systems are one of the core components of the safety-critical Advanced Driver Assistant System (ADAS) of a modern vehicle. Due to their ability to operate efficiently despite bad weather conditions and poor visibility, they are often the only reliable sensor a car has to detect and evaluate potential dangers in the surrounding environment. In this paper, we propose several attacks against automotive radars for the purposes of assessing their reliability in real-world scenarios. Using COTS hardware, we are able to successfully interfere with automotive-grade FMCW radars operating in the commonly used 77GHz frequency band, deployed in real-world, truly wireless environments. Our strongest type of interference is able to trick the victim into detecting virtual (moving) objects. We also extend this attack with a novel method that leverages noise to remove real-world objects, thus complementing the aforementioned object spoofing attack. We evaluate the viability of our attacks in two ways. First, we establish a baseline by implementing and evaluating an unrealistically powerful adversary which requires synchronization to the victim in a limited setup that uses wire-based chirp synchronization. Later, we implement, for the first time, a truly wireless attack that evaluates a weaker but realistic adversary which is non-synchronized and does not require any adjustment feedback from the victim. Finally, we provide theoretical fundamentals for our findings, and discuss the efficiency of potential countermeasures against the proposed attacks. We plan to release our software as open-source.



## **33. l-Leaks: Membership Inference Attacks with Logits**

cs.LG

10pages,6figures

**SubmitDate**: 2022-05-13    [paper-pdf](http://arxiv.org/pdf/2205.06469v1)

**Authors**: Shuhao Li, Yajie Wang, Yuanzhang Li, Yu-an Tan

**Abstracts**: Machine Learning (ML) has made unprecedented progress in the past several decades. However, due to the memorability of the training data, ML is susceptible to various attacks, especially Membership Inference Attacks (MIAs), the objective of which is to infer the model's training data. So far, most of the membership inference attacks against ML classifiers leverage the shadow model with the same structure as the target model. However, empirical results show that these attacks can be easily mitigated if the shadow model is not clear about the network structure of the target model.   In this paper, We present attacks based on black-box access to the target model. We name our attack \textbf{l-Leaks}. The l-Leaks follows the intuition that if an established shadow model is similar enough to the target model, then the adversary can leverage the shadow model's information to predict a target sample's membership.The logits of the trained target model contain valuable sample knowledge. We build the shadow model by learning the logits of the target model and making the shadow model more similar to the target model. Then shadow model will have sufficient confidence in the member samples of the target model. We also discuss the effect of the shadow model's different network structures to attack results. Experiments over different networks and datasets demonstrate that both of our attacks achieve strong performance.



## **34. Bitcoin's Latency--Security Analysis Made Simple**

cs.CR

**SubmitDate**: 2022-05-13    [paper-pdf](http://arxiv.org/pdf/2203.06357v2)

**Authors**: Dongning Guo, Ling Ren

**Abstracts**: Simple closed-form upper and lower bounds are developed for the security of the Nakamoto consensus as a function of the confirmation depth, the honest and adversarial block mining rates, and an upper bound on the block propagation delay. The bounds are exponential in the confirmation depth and apply regardless of the adversary's attack strategy. The gap between the upper and lower bounds is small for Bitcoin's parameters. For example, assuming an average block interval of 10 minutes, a network delay bound of ten seconds, and 10% adversarial mining power, the widely used 6-block confirmation rule yields a safety violation between 0.11% and 0.35% probability.



## **35. How to Combine Membership-Inference Attacks on Multiple Updated Models**

cs.LG

31 pages, 9 figures

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2205.06369v1)

**Authors**: Matthew Jagielski, Stanley Wu, Alina Oprea, Jonathan Ullman, Roxana Geambasu

**Abstracts**: A large body of research has shown that machine learning models are vulnerable to membership inference (MI) attacks that violate the privacy of the participants in the training data. Most MI research focuses on the case of a single standalone model, while production machine-learning platforms often update models over time, on data that often shifts in distribution, giving the attacker more information. This paper proposes new attacks that take advantage of one or more model updates to improve MI. A key part of our approach is to leverage rich information from standalone MI attacks mounted separately against the original and updated models, and to combine this information in specific ways to improve attack effectiveness. We propose a set of combination functions and tuning methods for each, and present both analytical and quantitative justification for various options. Our results on four public datasets show that our attacks are effective at using update information to give the adversary a significant advantage over attacks on standalone models, but also compared to a prior MI attack that takes advantage of model updates in a related machine-unlearning setting. We perform the first measurements of the impact of distribution shift on MI attacks with model updates, and show that a more drastic distribution shift results in significantly higher MI risk than a gradual shift. Our code is available at https://www.github.com/stanleykywu/model-updates.



## **36. Anomaly Detection of Adversarial Examples using Class-conditional Generative Adversarial Networks**

cs.LG

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2105.10101v2)

**Authors**: Hang Wang, David J. Miller, George Kesidis

**Abstracts**: Deep Neural Networks (DNNs) have been shown vulnerable to Test-Time Evasion attacks (TTEs, or adversarial examples), which, by making small changes to the input, alter the DNN's decision. We propose an unsupervised attack detector on DNN classifiers based on class-conditional Generative Adversarial Networks (GANs). We model the distribution of clean data conditioned on the predicted class label by an Auxiliary Classifier GAN (AC-GAN). Given a test sample and its predicted class, three detection statistics are calculated based on the AC-GAN Generator and Discriminator. Experiments on image classification datasets under various TTE attacks show that our method outperforms previous detection methods. We also investigate the effectiveness of anomaly detection using different DNN layers (input features or internal-layer features) and demonstrate, as one might expect, that anomalies are harder to detect using features closer to the DNN's output layer.



## **37. Sample Complexity Bounds for Robustly Learning Decision Lists against Evasion Attacks**

cs.LG

To appear in the proceedings of International Joint Conference on  Artificial Intelligence (2022)

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2205.06127v1)

**Authors**: Pascale Gourdeau, Varun Kanade, Marta Kwiatkowska, James Worrell

**Abstracts**: A fundamental problem in adversarial machine learning is to quantify how much training data is needed in the presence of evasion attacks. In this paper we address this issue within the framework of PAC learning, focusing on the class of decision lists. Given that distributional assumptions are essential in the adversarial setting, we work with probability distributions on the input data that satisfy a Lipschitz condition: nearby points have similar probability. Our key results illustrate that the adversary's budget (that is, the number of bits it can perturb on each input) is a fundamental quantity in determining the sample complexity of robust learning. Our first main result is a sample-complexity lower bound: the class of monotone conjunctions (essentially the simplest non-trivial hypothesis class on the Boolean hypercube) and any superclass has sample complexity at least exponential in the adversary's budget. Our second main result is a corresponding upper bound: for every fixed $k$ the class of $k$-decision lists has polynomial sample complexity against a $\log(n)$-bounded adversary. This sheds further light on the question of whether an efficient PAC learning algorithm can always be used as an efficient $\log(n)$-robust learning algorithm under the uniform distribution.



## **38. From IP to transport and beyond: cross-layer attacks against applications**

cs.CR

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2205.06085v1)

**Authors**: Tianxiang Dai, Philipp Jeitner, Haya Shulman, Michael Waidner

**Abstracts**: We perform the first analysis of methodologies for launching DNS cache poisoning: manipulation at the IP layer, hijack of the inter-domain routing and probing open ports via side channels. We evaluate these methodologies against DNS resolvers in the Internet and compare them with respect to effectiveness, applicability and stealth. Our study shows that DNS cache poisoning is a practical and pervasive threat.   We then demonstrate cross-layer attacks that leverage DNS cache poisoning for attacking popular systems, ranging from security mechanisms, such as RPKI, to applications, such as VoIP. In addition to more traditional adversarial goals, most notably impersonation and Denial of Service, we show for the first time that DNS cache poisoning can even enable adversaries to bypass cryptographic defences: we demonstrate how DNS cache poisoning can facilitate BGP prefix hijacking of networks protected with RPKI even when all the other networks apply route origin validation to filter invalid BGP announcements. Our study shows that DNS plays a much more central role in the Internet security than previously assumed.   We recommend mitigations for securing the applications and for preventing cache poisoning.



## **39. Segmentation-Consistent Probabilistic Lesion Counting**

eess.IV

Accepted at Medical Imaging with Deep Learning (MIDL) 2022

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2204.05276v2)

**Authors**: Julien Schroeter, Chelsea Myers-Colet, Douglas L Arnold, Tal Arbel

**Abstracts**: Lesion counts are important indicators of disease severity, patient prognosis, and treatment efficacy, yet counting as a task in medical imaging is often overlooked in favor of segmentation. This work introduces a novel continuously differentiable function that maps lesion segmentation predictions to lesion count probability distributions in a consistent manner. The proposed end-to-end approach--which consists of voxel clustering, lesion-level voxel probability aggregation, and Poisson-binomial counting--is non-parametric and thus offers a robust and consistent way to augment lesion segmentation models with post hoc counting capabilities. Experiments on Gadolinium-enhancing lesion counting demonstrate that our method outputs accurate and well-calibrated count distributions that capture meaningful uncertainty information. They also reveal that our model is suitable for multi-task learning of lesion segmentation, is efficient in low data regimes, and is robust to adversarial attacks.



## **40. Stalloris: RPKI Downgrade Attack**

cs.CR

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2205.06064v1)

**Authors**: Tomas Hlavacek, Philipp Jeitner, Donika Mirdita, Haya Shulman, Michael Waidner

**Abstracts**: We demonstrate the first downgrade attacks against RPKI. The key design property in RPKI that allows our attacks is the tradeoff between connectivity and security: when networks cannot retrieve RPKI information from publication points, they make routing decisions in BGP without validating RPKI. We exploit this tradeoff to develop attacks that prevent the retrieval of the RPKI objects from the public repositories, thereby disabling RPKI validation and exposing the RPKI-protected networks to prefix hijack attacks.   We demonstrate experimentally that at least 47% of the public repositories are vulnerable against a specific version of our attacks, a rate-limiting off-path downgrade attack. We also show that all the current RPKI relying party implementations are vulnerable to attacks by a malicious publication point. This translates to 20.4% of the IPv4 address space.   We provide recommendations for preventing our downgrade attacks. However, resolving the fundamental problem is not straightforward: if the relying parties prefer security over connectivity and insist on RPKI validation when ROAs cannot be retrieved, the victim AS may become disconnected from many more networks than just the one that the adversary wishes to hijack. Our work shows that the publication points are a critical infrastructure for Internet connectivity and security. Our main recommendation is therefore that the publication points should be hosted on robust platforms guaranteeing a high degree of connectivity.



## **41. Infrared Invisible Clothing:Hiding from Infrared Detectors at Multiple Angles in Real World**

cs.CV

Accepted by CVPR 2022, ORAL

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2205.05909v1)

**Authors**: Xiaopei Zhu, Zhanhao Hu, Siyuan Huang, Jianmin Li, Xiaolin Hu

**Abstracts**: Thermal infrared imaging is widely used in body temperature measurement, security monitoring, and so on, but its safety research attracted attention only in recent years. We proposed the infrared adversarial clothing, which could fool infrared pedestrian detectors at different angles. We simulated the process from cloth to clothing in the digital world and then designed the adversarial "QR code" pattern. The core of our method is to design a basic pattern that can be expanded periodically, and make the pattern after random cropping and deformation still have an adversarial effect, then we can process the flat cloth with an adversarial pattern into any 3D clothes. The results showed that the optimized "QR code" pattern lowered the Average Precision (AP) of YOLOv3 by 87.7%, while the random "QR code" pattern and blank pattern lowered the AP of YOLOv3 by 57.9% and 30.1%, respectively, in the digital world. We then manufactured an adversarial shirt with a new material: aerogel. Physical-world experiments showed that the adversarial "QR code" pattern clothing lowered the AP of YOLOv3 by 64.6%, while the random "QR code" pattern clothing and fully heat-insulated clothing lowered the AP of YOLOv3 by 28.3% and 22.8%, respectively. We used the model ensemble technique to improve the attack transferability to unseen models.



## **42. Using Frequency Attention to Make Adversarial Patch Powerful Against Person Detector**

cs.CV

10pages, 4 figures

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2205.04638v2)

**Authors**: Xiaochun Lei, Chang Lu, Zetao Jiang, Zhaoting Gong, Xiang Cai, Linjun Lu

**Abstracts**: Deep neural networks (DNNs) are vulnerable to adversarial attacks. In particular, object detectors may be attacked by applying a particular adversarial patch to the image. However, because the patch shrinks during preprocessing, most existing approaches that employ adversarial patches to attack object detectors would diminish the attack success rate on small and medium targets. This paper proposes a Frequency Module(FRAN), a frequency-domain attention module for guiding patch generation. This is the first study to introduce frequency domain attention to optimize the attack capabilities of adversarial patches. Our method increases the attack success rates of small and medium targets by 4.18% and 3.89%, respectively, over the state-of-the-art attack method for fooling the human detector while assaulting YOLOv3 without reducing the attack success rate of big targets.



## **43. The Hijackers Guide To The Galaxy: Off-Path Taking Over Internet Resources**

cs.CR

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2205.05473v1)

**Authors**: Tianxiang Dai, Philipp Jeitner, Haya Shulman, Michael Waidner

**Abstracts**: Internet resources form the basic fabric of the digital society. They provide the fundamental platform for digital services and assets, e.g., for critical infrastructures, financial services, government. Whoever controls that fabric effectively controls the digital society.   In this work we demonstrate that the current practices of Internet resources management, of IP addresses, domains, certificates and virtual platforms are insecure. Over long periods of time adversaries can maintain control over Internet resources which they do not own and perform stealthy manipulations, leading to devastating attacks. We show that network adversaries can take over and manipulate at least 68% of the assigned IPv4 address space as well as 31% of the top Alexa domains. We demonstrate such attacks by hijacking the accounts associated with the digital resources.   For hijacking the accounts we launch off-path DNS cache poisoning attacks, to redirect the password recovery link to the adversarial hosts. We then demonstrate that the adversaries can manipulate the resources associated with these accounts. We find all the tested providers vulnerable to our attacks.   We recommend mitigations for blocking the attacks that we present in this work. Nevertheless, the countermeasures cannot solve the fundamental problem - the management of the Internet resources should be revised to ensure that applying transactions cannot be done so easily and stealthily as is currently possible.



## **44. Sardino: Ultra-Fast Dynamic Ensemble for Secure Visual Sensing at Mobile Edge**

cs.CV

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2204.08189v2)

**Authors**: Qun Song, Zhenyu Yan, Wenjie Luo, Rui Tan

**Abstracts**: Adversarial example attack endangers the mobile edge systems such as vehicles and drones that adopt deep neural networks for visual sensing. This paper presents {\em Sardino}, an active and dynamic defense approach that renews the inference ensemble at run time to develop security against the adaptive adversary who tries to exfiltrate the ensemble and construct the corresponding effective adversarial examples. By applying consistency check and data fusion on the ensemble's predictions, Sardino can detect and thwart adversarial inputs. Compared with the training-based ensemble renewal, we use HyperNet to achieve {\em one million times} acceleration and per-frame ensemble renewal that presents the highest level of difficulty to the prerequisite exfiltration attacks. We design a run-time planner that maximizes the ensemble size in favor of security while maintaining the processing frame rate. Beyond adversarial examples, Sardino can also address the issue of out-of-distribution inputs effectively. This paper presents extensive evaluation of Sardino's performance in counteracting adversarial examples and applies it to build a real-time car-borne traffic sign recognition system. Live on-road tests show the built system's effectiveness in maintaining frame rate and detecting out-of-distribution inputs due to the false positives of a preceding YOLO-based traffic sign detector.



## **45. Developing Imperceptible Adversarial Patches to Camouflage Military Assets From Computer Vision Enabled Technologies**

cs.CV

8 pages, 4 figures, 4 tables, submitted to WCCI 2022

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2202.08892v2)

**Authors**: Chris Wise, Jo Plested

**Abstracts**: Convolutional neural networks (CNNs) have demonstrated rapid progress and a high level of success in object detection. However, recent evidence has highlighted their vulnerability to adversarial attacks. These attacks are calculated image perturbations or adversarial patches that result in object misclassification or detection suppression. Traditional camouflage methods are impractical when applied to disguise aircraft and other large mobile assets from autonomous detection in intelligence, surveillance and reconnaissance technologies and fifth generation missiles. In this paper we present a unique method that produces imperceptible patches capable of camouflaging large military assets from computer vision-enabled technologies. We developed these patches by maximising object detection loss whilst limiting the patch's colour perceptibility. This work also aims to further the understanding of adversarial examples and their effects on object detection algorithms.



## **46. A Word is Worth A Thousand Dollars: Adversarial Attack on Tweets Fools Stock Prediction**

cs.CR

NAACL short paper, github: https://github.com/yonxie/AdvFinTweet

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2205.01094v2)

**Authors**: Yong Xie, Dakuo Wang, Pin-Yu Chen, Jinjun Xiong, Sijia Liu, Sanmi Koyejo

**Abstracts**: More and more investors and machine learning models rely on social media (e.g., Twitter and Reddit) to gather real-time information and sentiment to predict stock price movements. Although text-based models are known to be vulnerable to adversarial attacks, whether stock prediction models have similar vulnerability is underexplored. In this paper, we experiment with a variety of adversarial attack configurations to fool three stock prediction victim models. We address the task of adversarial generation by solving combinatorial optimization problems with semantics and budget constraints. Our results show that the proposed attack method can achieve consistent success rates and cause significant monetary loss in trading simulation by simply concatenating a perturbed but semantically similar tweet.



## **47. Privacy Enhancement for Cloud-Based Few-Shot Learning**

cs.LG

14 pages, 13 figures, 3 tables. Preprint. Accepted in IEEE WCCI 2022  International Joint Conference on Neural Networks (IJCNN)

**SubmitDate**: 2022-05-10    [paper-pdf](http://arxiv.org/pdf/2205.07864v1)

**Authors**: Archit Parnami, Muhammad Usama, Liyue Fan, Minwoo Lee

**Abstracts**: Requiring less data for accurate models, few-shot learning has shown robustness and generality in many application domains. However, deploying few-shot models in untrusted environments may inflict privacy concerns, e.g., attacks or adversaries that may breach the privacy of user-supplied data. This paper studies the privacy enhancement for the few-shot learning in an untrusted environment, e.g., the cloud, by establishing a novel privacy-preserved embedding space that preserves the privacy of data and maintains the accuracy of the model. We examine the impact of various image privacy methods such as blurring, pixelization, Gaussian noise, and differentially private pixelization (DP-Pix) on few-shot image classification and propose a method that learns privacy-preserved representation through the joint loss. The empirical results show how privacy-performance trade-off can be negotiated for privacy-enhanced few-shot learning.



## **48. SYNFI: Pre-Silicon Fault Analysis of an Open-Source Secure Element**

cs.CR

**SubmitDate**: 2022-05-10    [paper-pdf](http://arxiv.org/pdf/2205.04775v1)

**Authors**: Pascal Nasahl, Miguel Osorio, Pirmin Vogel, Michael Schaffner, Timothy Trippel, Dominic Rizzo, Stefan Mangard

**Abstracts**: Fault attacks are active, physical attacks that an adversary can leverage to alter the control-flow of embedded devices to gain access to sensitive information or bypass protection mechanisms. Due to the severity of these attacks, manufacturers deploy hardware-based fault defenses into security-critical systems, such as secure elements. The development of these countermeasures is a challenging task due to the complex interplay of circuit components and because contemporary design automation tools tend to optimize inserted structures away, thereby defeating their purpose. Hence, it is critical that such countermeasures are rigorously verified post-synthesis. As classical functional verification techniques fall short of assessing the effectiveness of countermeasures, developers have to resort to methods capable of injecting faults in a simulation testbench or into a physical chip. However, developing test sequences to inject faults in simulation is an error-prone task and performing fault attacks on a chip requires specialized equipment and is incredibly time-consuming. To that end, this paper introduces SYNFI, a formal pre-silicon fault verification framework that operates on synthesized netlists. SYNFI can be used to analyze the general effect of faults on the input-output relationship in a circuit and its fault countermeasures, and thus enables hardware designers to assess and verify the effectiveness of embedded countermeasures in a systematic and semi-automatic way. To demonstrate that SYNFI is capable of handling unmodified, industry-grade netlists synthesized with commercial and open tools, we analyze OpenTitan, the first open-source secure element. In our analysis, we identified critical security weaknesses in the unprotected AES block, developed targeted countermeasures, reassessed their security, and contributed these countermeasures back to the OpenTitan repository.



## **49. Semi-Targeted Model Poisoning Attack on Federated Learning via Backward Error Analysis**

cs.LG

Published in IJCNN 2022

**SubmitDate**: 2022-05-10    [paper-pdf](http://arxiv.org/pdf/2203.11633v2)

**Authors**: Yuwei Sun, Hideya Ochiai, Jun Sakuma

**Abstracts**: Model poisoning attacks on federated learning (FL) intrude in the entire system via compromising an edge model, resulting in malfunctioning of machine learning models. Such compromised models are tampered with to perform adversary-desired behaviors. In particular, we considered a semi-targeted situation where the source class is predetermined however the target class is not. The goal is to cause the global classifier to misclassify data of the source class. Though approaches such as label flipping have been adopted to inject poisoned parameters into FL, it has been shown that their performances are usually class-sensitive varying with different target classes applied. Typically, an attack can become less effective when shifting to a different target class. To overcome this challenge, we propose the Attacking Distance-aware Attack (ADA) to enhance a poisoning attack by finding the optimized target class in the feature space. Moreover, we studied a more challenging situation where an adversary had limited prior knowledge about a client's data. To tackle this problem, ADA deduces pair-wise distances between different classes in the latent feature space from shared model parameters based on the backward error analysis. We performed extensive empirical evaluations on ADA by varying the factor of attacking frequency in three different image classification tasks. As a result, ADA succeeded in increasing the attack performance by 1.8 times in the most challenging case with an attacking frequency of 0.01.



## **50. Fingerprinting of DNN with Black-box Design and Verification**

cs.CR

**SubmitDate**: 2022-05-10    [paper-pdf](http://arxiv.org/pdf/2203.10902v3)

**Authors**: Shuo Wang, Sharif Abuadbba, Sidharth Agarwal, Kristen Moore, Ruoxi Sun, Minhui Xue, Surya Nepal, Seyit Camtepe, Salil Kanhere

**Abstracts**: Cloud-enabled Machine Learning as a Service (MLaaS) has shown enormous promise to transform how deep learning models are developed and deployed. Nonetheless, there is a potential risk associated with the use of such services since a malicious party can modify them to achieve an adverse result. Therefore, it is imperative for model owners, service providers, and end-users to verify whether the deployed model has not been tampered with or not. Such verification requires public verifiability (i.e., fingerprinting patterns are available to all parties, including adversaries) and black-box access to the deployed model via APIs. Existing watermarking and fingerprinting approaches, however, require white-box knowledge (such as gradient) to design the fingerprinting and only support private verifiability, i.e., verification by an honest party.   In this paper, we describe a practical watermarking technique that enables black-box knowledge in fingerprint design and black-box queries during verification. The service ensures the integrity of cloud-based services through public verification (i.e. fingerprinting patterns are available to all parties, including adversaries). If an adversary manipulates a model, this will result in a shift in the decision boundary. Thus, the underlying principle of double-black watermarking is that a model's decision boundary could serve as an inherent fingerprint for watermarking. Our approach captures the decision boundary by generating a limited number of encysted sample fingerprints, which are a set of naturally transformed and augmented inputs enclosed around the model's decision boundary in order to capture the inherent fingerprints of the model. We evaluated our watermarking approach against a variety of model integrity attacks and model compression attacks.



