# Latest Adversarial Attack Papers
**update at 2022-11-19 15:42:13**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Adaptive Test-Time Defense with the Manifold Hypothesis**

cs.LG

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2210.14404v3) [paper-pdf](http://arxiv.org/pdf/2210.14404v3)

**Authors**: Zhaoyuan Yang, Zhiwei Xu, Jing Zhang, Richard Hartley, Peter Tu

**Abstract**: In this work, we formulate a novel framework of adversarial robustness using the manifold hypothesis. Our framework provides sufficient conditions for defending against adversarial examples. We develop a test-time defense method with variational inference and our formulation. The developed approach combines manifold learning with variational inference to provide adversarial robustness without the need for adversarial training. We show that our approach can provide adversarial robustness even if attackers are aware of the existence of test-time defense. In addition, our approach can also serve as a test-time defense mechanism for variational autoencoders.



## **2. UPTON: Unattributable Authorship Text via Data Poisoning**

cs.CY

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2211.09717v1) [paper-pdf](http://arxiv.org/pdf/2211.09717v1)

**Authors**: Ziyao Wang, Thai Le, Dongwon Lee

**Abstract**: In online medium such as opinion column in Bloomberg, The Guardian and Western Journal, aspiring writers post their writings for various reasons with their names often proudly open. However, it may occur that such a writer wants to write in other venues anonymously or under a pseudonym (e.g., activist, whistle-blower). However, if an attacker has already built an accurate authorship attribution (AA) model based off of the writings from such platforms, attributing an anonymous writing to the known authorship is possible. Therefore, in this work, we ask a question "can one make the writings and texts, T, in the open spaces such as opinion sharing platforms unattributable so that AA models trained from T cannot attribute authorship well?" Toward this question, we present a novel solution, UPTON, that exploits textual data poisoning method to disturb the training process of AA models. UPTON uses data poisoning to destroy the authorship feature only in training samples by perturbing them, and try to make released textual data unlearnable on deep neuron networks. It is different from previous obfuscation works, that use adversarial attack to modify the test samples and mislead an AA model, and also the backdoor works, which use trigger words both in test and training samples and only change the model output when trigger words occur. Using four authorship datasets (e.g., IMDb10, IMDb64, Enron and WJO), then, we present empirical validation where: (1)UPTON is able to downgrade the test accuracy to about 30% with carefully designed target-selection methods. (2)UPTON poisoning is able to preserve most of the original semantics. The BERTSCORE between the clean and UPTON poisoned texts are higher than 0.95. The number is very closed to 1.00, which means no sematic change. (3)UPTON is also robust towards spelling correction systems.



## **3. An efficient combination of quantum error correction and authentication**

quant-ph

30 pages, 10 figures

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2211.09686v1) [paper-pdf](http://arxiv.org/pdf/2211.09686v1)

**Authors**: Yfke Dulek, Garazi Muguruza, Florian Speelman

**Abstract**: When sending quantum information over a channel, we want to ensure that the message remains intact. Quantum error correction and quantum authentication both aim to protect (quantum) information, but approach this task from two very different directions: error-correcting codes protect against probabilistic channel noise and are meant to be very robust against small errors, while authentication codes prevent adversarial attacks and are designed to be very sensitive against any error, including small ones.   In practice, when sending an authenticated state over a noisy channel, one would have to wrap it in an error-correcting code to counterbalance the sensitivity of the underlying authentication scheme. We study the question of whether this can be done more efficiently by combining the two functionalities in a single code. To illustrate the potential of such a combination, we design the threshold code, a modification of the trap authentication code which preserves that code's authentication properties, but which is naturally robust against depolarizing channel noise. We show that the threshold code needs polylogarithmically fewer qubits to achieve the same level of security and robustness, compared to the naive composition of the trap code with any concatenated CSS code. We believe our analysis opens the door to combining more general error-correction and authentication codes, which could improve the practicality of the resulting scheme.



## **4. Towards Good Practices in Evaluating Transfer Adversarial Attacks**

cs.CR

Our code and a list of categorized attacks are publicly available at  https://github.com/ZhengyuZhao/TransferAttackEval

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2211.09565v1) [paper-pdf](http://arxiv.org/pdf/2211.09565v1)

**Authors**: Zhengyu Zhao, Hanwei Zhang, Renjue Li, Ronan Sicre, Laurent Amsaleg, Michael Backes

**Abstract**: Transfer adversarial attacks raise critical security concerns in real-world, black-box scenarios. However, the actual progress of attack methods is difficult to assess due to two main limitations in existing evaluations. First, existing evaluations are unsystematic and sometimes unfair since new methods are often directly added to old ones without complete comparisons to similar methods. Second, existing evaluations mainly focus on transferability but overlook another key attack property: stealthiness. In this work, we design good practices to address these limitations. We first introduce a new attack categorization, which enables our systematic analyses of similar attacks in each specific category. Our analyses lead to new findings that complement or even challenge existing knowledge. Furthermore, we comprehensively evaluate 23 representative attacks against 9 defenses on ImageNet. We pay particular attention to stealthiness, by adopting diverse imperceptibility metrics and looking into new, finer-grained characteristics. Our evaluation reveals new important insights: 1) Transferability is highly contextual, and some white-box defenses may give a false sense of security since they are actually vulnerable to (black-box) transfer attacks; 2) All transfer attacks are less stealthy, and their stealthiness can vary dramatically under the same $L_{\infty}$ bound.



## **5. Ignore Previous Prompt: Attack Techniques For Language Models**

cs.CL

ML Safety Workshop NeurIPS 2022

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2211.09527v1) [paper-pdf](http://arxiv.org/pdf/2211.09527v1)

**Authors**: Fábio Perez, Ian Ribeiro

**Abstract**: Transformer-based large language models (LLMs) provide a powerful foundation for natural language tasks in large-scale customer-facing applications. However, studies that explore their vulnerabilities emerging from malicious user interaction are scarce. By proposing PromptInject, a prosaic alignment framework for mask-based iterative adversarial prompt composition, we examine how GPT-3, the most widely deployed language model in production, can be easily misaligned by simple handcrafted inputs. In particular, we investigate two types of attacks -- goal hijacking and prompt leaking -- and demonstrate that even low-aptitude, but sufficiently ill-intentioned agents, can easily exploit GPT-3's stochastic nature, creating long-tail risks. The code for PromptInject is available at https://github.com/agencyenterprise/PromptInject.



## **6. Look Closer to Your Enemy: Learning to Attack via Teacher-student Mimicking**

cs.CV

13 pages, 8 figures, NDSS

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2207.13381v3) [paper-pdf](http://arxiv.org/pdf/2207.13381v3)

**Authors**: Mingjie Wang, Zhiqing Tang, Sirui Li, Dingwen Xiao

**Abstract**: This paper aims to generate realistic attack samples of person re-identification, ReID, by reading the enemy's mind (VM). In this paper, we propose a novel inconspicuous and controllable ReID attack baseline, LCYE, to generate adversarial query images. Concretely, LCYE first distills VM's knowledge via teacher-student memory mimicking in the proxy task. Then this knowledge prior acts as an explicit cipher conveying what is essential and realistic, believed by VM, for accurate adversarial misleading. Besides, benefiting from the multiple opposing task framework of LCYE, we further investigate the interpretability and generalization of ReID models from the view of the adversarial attack, including cross-domain adaption, cross-model consensus, and online learning process. Extensive experiments on four ReID benchmarks show that our method outperforms other state-of-the-art attackers with a large margin in white-box, black-box, and target attacks. Our code is now available at https://gitfront.io/r/user-3704489/mKXusqDT4ffr/LCYE/.



## **7. Phantom Sponges: Exploiting Non-Maximum Suppression to Attack Deep Object Detectors**

cs.CV

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2205.13618v3) [paper-pdf](http://arxiv.org/pdf/2205.13618v3)

**Authors**: Avishag Shapira, Alon Zolfi, Luca Demetrio, Battista Biggio, Asaf Shabtai

**Abstract**: Adversarial attacks against deep learning-based object detectors have been studied extensively in the past few years. Most of the attacks proposed have targeted the model's integrity (i.e., caused the model to make incorrect predictions), while adversarial attacks targeting the model's availability, a critical aspect in safety-critical domains such as autonomous driving, have not yet been explored by the machine learning research community. In this paper, we propose a novel attack that negatively affects the decision latency of an end-to-end object detection pipeline. We craft a universal adversarial perturbation (UAP) that targets a widely used technique integrated in many object detector pipelines -- non-maximum suppression (NMS). Our experiments demonstrate the proposed UAP's ability to increase the processing time of individual frames by adding "phantom" objects that overload the NMS algorithm while preserving the detection of the original objects which allows the attack to go undetected for a longer period of time.



## **8. Reasons for the Superiority of Stochastic Estimators over Deterministic Ones: Robustness, Consistency and Perceptual Quality**

eess.IV

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2211.08944v2) [paper-pdf](http://arxiv.org/pdf/2211.08944v2)

**Authors**: Guy Ohayon, Theo Adrai, Michael Elad, Tomer Michaeli

**Abstract**: Stochastic restoration algorithms allow to explore the space of solutions that correspond to the degraded input. In this paper we reveal additional fundamental advantages of stochastic methods over deterministic ones, which further motivate their use. First, we prove that any restoration algorithm that attains perfect perceptual quality and whose outputs are consistent with the input must be a posterior sampler, and is thus required to be stochastic. Second, we illustrate that while deterministic restoration algorithms may attain high perceptual quality, this can be achieved only by filling up the space of all possible source images using an extremely sensitive mapping, which makes them highly vulnerable to adversarial attacks. Indeed, we show that enforcing deterministic models to be robust to such attacks profoundly hinders their perceptual quality, while robustifying stochastic models hardly influences their perceptual quality, and improves their output variability. These findings provide a motivation to foster progress in stochastic restoration methods, paving the way to better recovery algorithms.



## **9. Generalizable Deepfake Detection with Phase-Based Motion Analysis**

cs.CV

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2211.09363v1) [paper-pdf](http://arxiv.org/pdf/2211.09363v1)

**Authors**: Ekta Prashnani, Michael Goebel, B. S. Manjunath

**Abstract**: We propose PhaseForensics, a DeepFake (DF) video detection method that leverages a phase-based motion representation of facial temporal dynamics. Existing methods relying on temporal inconsistencies for DF detection present many advantages over the typical frame-based methods. However, they still show limited cross-dataset generalization and robustness to common distortions. These shortcomings are partially due to error-prone motion estimation and landmark tracking, or the susceptibility of the pixel intensity-based features to spatial distortions and the cross-dataset domain shifts. Our key insight to overcome these issues is to leverage the temporal phase variations in the band-pass components of the Complex Steerable Pyramid on face sub-regions. This not only enables a robust estimate of the temporal dynamics in these regions, but is also less prone to cross-dataset variations. Furthermore, the band-pass filters used to compute the local per-frame phase form an effective defense against the perturbations commonly seen in gradient-based adversarial attacks. Overall, with PhaseForensics, we show improved distortion and adversarial robustness, and state-of-the-art cross-dataset generalization, with 91.2% video-level AUC on the challenging CelebDFv2 (a recent state-of-the-art compares at 86.9%).



## **10. Targeted Attention for Generalized- and Zero-Shot Learning**

cs.CV

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2211.09322v1) [paper-pdf](http://arxiv.org/pdf/2211.09322v1)

**Authors**: Abhijit Suprem

**Abstract**: The Zero-Shot Learning (ZSL) task attempts to learn concepts without any labeled data. Unlike traditional classification/detection tasks, the evaluation environment is provided unseen classes never encountered during training. As such, it remains both challenging, and promising on a variety of fronts, including unsupervised concept learning, domain adaptation, and dataset drift detection. Recently, there have been a variety of approaches towards solving ZSL, including improved metric learning methods, transfer learning, combinations of semantic and image domains using, e.g. word vectors, and generative models to model the latent space of known classes to classify unseen classes. We find many approaches require intensive training augmentation with attributes or features that may be commonly unavailable (attribute-based learning) or susceptible to adversarial attacks (generative learning). We propose combining approaches from the related person re-identification task for ZSL, with key modifications to ensure sufficiently improved performance in the ZSL setting without the need for feature or training dataset augmentation. We are able to achieve state-of-the-art performance on the CUB200 and Cars196 datasets in the ZSL setting compared to recent works, with NMI (normalized mutual inference) of 63.27 and top-1 of 61.04 for CUB200, and NMI 66.03 with top-1 82.75% in Cars196. We also show state-of-the-art results in the Generalized Zero-Shot Learning (GZSL) setting, with Harmonic Mean R-1 of 66.14% on the CUB200 dataset.



## **11. Fair Robust Active Learning by Joint Inconsistency**

cs.LG

11 pages, 2 figures, 8 tables

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2209.10729v2) [paper-pdf](http://arxiv.org/pdf/2209.10729v2)

**Authors**: Tsung-Han Wu, Hung-Ting Su, Shang-Tse Chen, Winston H. Hsu

**Abstract**: Fairness and robustness play vital roles in trustworthy machine learning. Observing safety-critical needs in various annotation-expensive vision applications, we introduce a novel learning framework, Fair Robust Active Learning (FRAL), generalizing conventional active learning to fair and adversarial robust scenarios. This framework allows us to achieve standard and robust minimax fairness with limited acquired labels. In FRAL, we then observe existing fairness-aware data selection strategies suffer from either ineffectiveness under severe data imbalance or inefficiency due to huge computations of adversarial training. To address these two problems, we develop a novel Joint INconsistency (JIN) method exploiting prediction inconsistencies between benign and adversarial inputs as well as between standard and robust models. These two inconsistencies can be used to identify potential fairness gains and data imbalance mitigations. Thus, by performing label acquisition with our inconsistency-based ranking metrics, we can alleviate the class imbalance issue and enhance minimax fairness with limited computation. Extensive experiments on diverse datasets and sensitive groups demonstrate that our method obtains the best results in standard and robust fairness under white-box PGD attacks compared with existing active data selection baselines.



## **12. Differentially Private Optimizers Can Learn Adversarially Robust Models**

cs.LG

**SubmitDate**: 2022-11-16    [abs](http://arxiv.org/abs/2211.08942v1) [paper-pdf](http://arxiv.org/pdf/2211.08942v1)

**Authors**: Yuan Zhang, Zhiqi Bu

**Abstract**: Machine learning models have shone in a variety of domains and attracted increasing attention from both the security and the privacy communities. One important yet worrying question is: will training models under the differential privacy (DP) constraint unfavorably impact on the adversarial robustness? While previous works have postulated that privacy comes at the cost of worse robustness, we give the first theoretical analysis to show that DP models can indeed be robust and accurate, even sometimes more robust than their naturally-trained non-private counterparts. We observe three key factors that influence the privacy-robustness-accuracy tradeoff: (1) hyperparameters for DP optimizers are critical; (2) pre-training on public data significantly mitigates the accuracy and robustness drop; (3) choice of DP optimizers makes a difference. With these factors set properly, we achieve 90\% natural accuracy, 72\% robust accuracy ($+9\%$ than the non-private model) under $l_2(0.5)$ attack, and 69\% robust accuracy ($+16\%$ than the non-private model) with pre-trained SimCLRv2 model under $l_\infty(4/255)$ attack on CIFAR10 with $\epsilon=2$. In fact, we show both theoretically and empirically that DP models are Pareto optimal on the accuracy-robustness tradeoff. Empirically, the robustness of DP models is consistently observed on MNIST, Fashion MNIST and CelebA datasets, with ResNet and Vision Transformer. We believe our encouraging results are a significant step towards training models that are private as well as robust.



## **13. Attacking Object Detector Using A Universal Targeted Label-Switch Patch**

cs.LG

**SubmitDate**: 2022-11-16    [abs](http://arxiv.org/abs/2211.08859v1) [paper-pdf](http://arxiv.org/pdf/2211.08859v1)

**Authors**: Avishag Shapira, Ron Bitton, Dan Avraham, Alon Zolfi, Yuval Elovici, Asaf Shabtai

**Abstract**: Adversarial attacks against deep learning-based object detectors (ODs) have been studied extensively in the past few years. These attacks cause the model to make incorrect predictions by placing a patch containing an adversarial pattern on the target object or anywhere within the frame. However, none of prior research proposed a misclassification attack on ODs, in which the patch is applied on the target object. In this study, we propose a novel, universal, targeted, label-switch attack against the state-of-the-art object detector, YOLO. In our attack, we use (i) a tailored projection function to enable the placement of the adversarial patch on multiple target objects in the image (e.g., cars), each of which may be located a different distance away from the camera or have a different view angle relative to the camera, and (ii) a unique loss function capable of changing the label of the attacked objects. The proposed universal patch, which is trained in the digital domain, is transferable to the physical domain. We performed an extensive evaluation using different types of object detectors, different video streams captured by different cameras, and various target classes, and evaluated different configurations of the adversarial patch in the physical domain.



## **14. T-SEA: Transfer-based Self-Ensemble Attack on Object Detection**

cs.CV

10 pages, 5 figures

**SubmitDate**: 2022-11-16    [abs](http://arxiv.org/abs/2211.09773v1) [paper-pdf](http://arxiv.org/pdf/2211.09773v1)

**Authors**: Hao Huang, Ziyan Chen, Huanran Chen, Yongtao Wang, Kevin Zhang

**Abstract**: Compared to query-based black-box attacks, transfer-based black-box attacks do not require any information of the attacked models, which ensures their secrecy. However, most existing transfer-based approaches rely on ensembling multiple models to boost the attack transferability, which is time- and resource-intensive, not to mention the difficulty of obtaining diverse models on the same task. To address this limitation, in this work, we focus on the single-model transfer-based black-box attack on object detection, utilizing only one model to achieve a high-transferability adversarial attack on multiple black-box detectors. Specifically, we first make observations on the patch optimization process of the existing method and propose an enhanced attack framework by slightly adjusting its training strategies. Then, we analogize patch optimization with regular model optimization, proposing a series of self-ensemble approaches on the input data, the attacked model, and the adversarial patch to efficiently make use of the limited information and prevent the patch from overfitting. The experimental results show that the proposed framework can be applied with multiple classical base attack methods (e.g., PGD and MIM) to greatly improve the black-box transferability of the well-optimized patch on multiple mainstream detectors, meanwhile boosting white-box performance. Our code is available at https://github.com/VDIGPKU/T-SEA.



## **15. Adversarial Camouflage for Node Injection Attack on Graphs**

cs.LG

**SubmitDate**: 2022-11-16    [abs](http://arxiv.org/abs/2208.01819v2) [paper-pdf](http://arxiv.org/pdf/2208.01819v2)

**Authors**: Shuchang Tao, Qi Cao, Huawei Shen, Yunfan Wu, Liang Hou, Xueqi Cheng

**Abstract**: Node injection attacks against Graph Neural Networks (GNNs) have received emerging attention as a practical attack scenario, where the attacker injects malicious nodes instead of modifying node features or edges to degrade the performance of GNNs. Despite the initial success of node injection attacks, we find that the injected nodes by existing methods are easy to be distinguished from the original normal nodes by defense methods and limiting their attack performance in practice. To solve the above issues, we devote to camouflage node injection attack, i.e., camouflaging injected malicious nodes (structure/attributes) as the normal ones that appear legitimate/imperceptible to defense methods. The non-Euclidean nature of graph data and the lack of human prior brings great challenges to the formalization, implementation, and evaluation of camouflage on graphs. In this paper, we first propose and formulate the camouflage of injected nodes from both the fidelity and diversity of the ego networks centered around injected nodes. Then, we design an adversarial CAmouflage framework for Node injection Attack, namely CANA, to improve the camouflage while ensuring the attack performance. Several novel indicators for graph camouflage are further designed for a comprehensive evaluation. Experimental results demonstrate that when equipping existing node injection attack methods with our proposed CANA framework, the attack performance against defense methods as well as node camouflage is significantly improved.



## **16. Semi-supervised Conditional GAN for Simultaneous Generation and Detection of Phishing URLs: A Game theoretic Perspective**

cs.CR

5 Pages, 4 figures, 2 tables

**SubmitDate**: 2022-11-16    [abs](http://arxiv.org/abs/2108.01852v3) [paper-pdf](http://arxiv.org/pdf/2108.01852v3)

**Authors**: Sharif Amit Kamran, Shamik Sengupta, Alireza Tavakkoli

**Abstract**: Spear Phishing is a type of cyber-attack where the attacker sends hyperlinks through email on well-researched targets. The objective is to obtain sensitive information by imitating oneself as a trustworthy website. In recent times, deep learning has become the standard for defending against such attacks. However, these architectures were designed with only defense in mind. Moreover, the attacker's perspective and motivation are absent while creating such models. To address this, we need a game-theoretic approach to understand the perspective of the attacker (Hacker) and the defender (Phishing URL detector). We propose a Conditional Generative Adversarial Network with novel training strategy for real-time phishing URL detection. Additionally, we train our architecture in a semi-supervised manner to distinguish between adversarial and real examples, along with detecting malicious and benign URLs. We also design two games between the attacker and defender in training and deployment settings by utilizing the game-theoretic perspective. Our experiments confirm that the proposed architecture surpasses recent state-of-the-art architectures for phishing URLs detection.



## **17. Improving Interpretability via Regularization of Neural Activation Sensitivity**

cs.LG

**SubmitDate**: 2022-11-16    [abs](http://arxiv.org/abs/2211.08686v1) [paper-pdf](http://arxiv.org/pdf/2211.08686v1)

**Authors**: Ofir Moshe, Gil Fidel, Ron Bitton, Asaf Shabtai

**Abstract**: State-of-the-art deep neural networks (DNNs) are highly effective at tackling many real-world tasks. However, their wide adoption in mission-critical contexts is hampered by two major weaknesses - their susceptibility to adversarial attacks and their opaqueness. The former raises concerns about the security and generalization of DNNs in real-world conditions, whereas the latter impedes users' trust in their output. In this research, we (1) examine the effect of adversarial robustness on interpretability and (2) present a novel approach for improving the interpretability of DNNs that is based on regularization of neural activation sensitivity. We evaluate the interpretability of models trained using our method to that of standard models and models trained using state-of-the-art adversarial robustness techniques. Our results show that adversarially robust models are superior to standard models and that models trained using our proposed method are even better than adversarially robust models in terms of interpretability.



## **18. Nano-Resolution Visual Identifiers Enable Secure Monitoring in Next-Generation Cyber-Physical Systems**

cs.CR

**SubmitDate**: 2022-11-16    [abs](http://arxiv.org/abs/2211.08678v1) [paper-pdf](http://arxiv.org/pdf/2211.08678v1)

**Authors**: Hao Wang, Xiwen Chen, Abolfazl Razi, Michael Kozicki, Rahul Amin, Mark Manfredo

**Abstract**: Today's supply chains heavily rely on cyber-physical systems such as intelligent transportation, online shopping, and E-commerce. It is advantageous to track goods in real-time by web-based registration and authentication of products after any substantial change or relocation. Despite recent advantages in technology-based tracking systems, most supply chains still rely on plainly printed tags such as barcodes and Quick Response (QR) codes for tracking purposes. Although affordable and efficient, these tags convey no security against counterfeit and cloning attacks, raising privacy concerns. It is a critical matter since a few security breaches in merchandise databases in recent years has caused crucial social and economic impacts such as identity loss, social panic, and loss of trust in the community. This paper considers an end-to-end system using dendrites as nano-resolution visual identifiers to secure supply chains. Dendrites are formed by generating fractal metallic patterns on transparent substrates through an electrochemical process, which can be used as secure identifiers due to their natural randomness, high entropy, and unclonable features. The proposed framework compromises the back-end program for identification and authentication, a web-based application for mobile devices, and a cloud database. We review architectural design, dendrite operational phases (personalization, registration, inspection), a lightweight identification method based on 2D graph-matching, and a deep 3D image authentication method based on Digital Holography (DH). A two-step search is proposed to make the system scalable by limiting the search space to samples with high similarity scores in a lower-dimensional space. We conclude by presenting our solution to make dendrites secure against adversarial attacks.



## **19. Person Text-Image Matching via Text-Featur Interpretability Embedding and External Attack Node Implantation**

cs.CV

**SubmitDate**: 2022-11-16    [abs](http://arxiv.org/abs/2211.08657v1) [paper-pdf](http://arxiv.org/pdf/2211.08657v1)

**Authors**: Fan Li, Hang Zhou, Huafeng Li, Yafei Zhang, Zhengtao Yu

**Abstract**: Person text-image matching, also known as textbased person search, aims to retrieve images of specific pedestrians using text descriptions. Although person text-image matching has made great research progress, existing methods still face two challenges. First, the lack of interpretability of text features makes it challenging to effectively align them with their corresponding image features. Second, the same pedestrian image often corresponds to multiple different text descriptions, and a single text description can correspond to multiple different images of the same identity. The diversity of text descriptions and images makes it difficult for a network to extract robust features that match the two modalities. To address these problems, we propose a person text-image matching method by embedding text-feature interpretability and an external attack node. Specifically, we improve the interpretability of text features by providing them with consistent semantic information with image features to achieve the alignment of text and describe image region features.To address the challenges posed by the diversity of text and the corresponding person images, we treat the variation caused by diversity to features as caused by perturbation information and propose a novel adversarial attack and defense method to solve it. In the model design, graph convolution is used as the basic framework for feature representation and the adversarial attacks caused by text and image diversity on feature extraction is simulated by implanting an additional attack node in the graph convolution layer to improve the robustness of the model against text and image diversity. Extensive experiments demonstrate the effectiveness and superiority of text-pedestrian image matching over existing methods. The source code of the method is published at



## **20. Membership Inference Attacks Against Temporally Correlated Data in Deep Reinforcement Learning**

cs.LG

**SubmitDate**: 2022-11-16    [abs](http://arxiv.org/abs/2109.03975v3) [paper-pdf](http://arxiv.org/pdf/2109.03975v3)

**Authors**: Maziar Gomrokchi, Susan Amin, Hossein Aboutalebi, Alexander Wong, Doina Precup

**Abstract**: While significant research advances have been made in the field of deep reinforcement learning, there have been no concrete adversarial attack strategies in literature tailored for studying the vulnerability of deep reinforcement learning algorithms to membership inference attacks. In such attacking systems, the adversary targets the set of collected input data on which the deep reinforcement learning algorithm has been trained. To address this gap, we propose an adversarial attack framework designed for testing the vulnerability of a state-of-the-art deep reinforcement learning algorithm to a membership inference attack. In particular, we design a series of experiments to investigate the impact of temporal correlation, which naturally exists in reinforcement learning training data, on the probability of information leakage. Moreover, we compare the performance of \emph{collective} and \emph{individual} membership attacks against the deep reinforcement learning algorithm. Experimental results show that the proposed adversarial attack framework is surprisingly effective at inferring data with an accuracy exceeding $84\%$ in individual and $97\%$ in collective modes in three different continuous control Mujoco tasks, which raises serious privacy concerns in this regard. Finally, we show that the learning state of the reinforcement learning algorithm influences the level of privacy breaches significantly.



## **21. Universal Distributional Decision-based Black-box Adversarial Attack with Reinforcement Learning**

cs.LG

10 pages, 2 figures, conference

**SubmitDate**: 2022-11-15    [abs](http://arxiv.org/abs/2211.08384v1) [paper-pdf](http://arxiv.org/pdf/2211.08384v1)

**Authors**: Yiran Huang, Yexu Zhou, Michael Hefenbrock, Till Riedel, Likun Fang, Michael Beigl

**Abstract**: The vulnerability of the high-performance machine learning models implies a security risk in applications with real-world consequences. Research on adversarial attacks is beneficial in guiding the development of machine learning models on the one hand and finding targeted defenses on the other. However, most of the adversarial attacks today leverage the gradient or logit information from the models to generate adversarial perturbation. Works in the more realistic domain: decision-based attacks, which generate adversarial perturbation solely based on observing the output label of the targeted model, are still relatively rare and mostly use gradient-estimation strategies. In this work, we propose a pixel-wise decision-based attack algorithm that finds a distribution of adversarial perturbation through a reinforcement learning algorithm. We call this method Decision-based Black-box Attack with Reinforcement learning (DBAR). Experiments show that the proposed approach outperforms state-of-the-art decision-based attacks with a higher attack success rate and greater transferability.



## **22. Resisting Graph Adversarial Attack via Cooperative Homophilous Augmentation**

cs.LG

The paper has been accepted for presentation at ECML PKDD 2022

**SubmitDate**: 2022-11-15    [abs](http://arxiv.org/abs/2211.08068v1) [paper-pdf](http://arxiv.org/pdf/2211.08068v1)

**Authors**: Zhihao Zhu, Chenwang Wu, Min Zhou, Hao Liao, Defu Lian, Enhong Chen

**Abstract**: Recent studies show that Graph Neural Networks(GNNs) are vulnerable and easily fooled by small perturbations, which has raised considerable concerns for adapting GNNs in various safety-critical applications. In this work, we focus on the emerging but critical attack, namely, Graph Injection Attack(GIA), in which the adversary poisons the graph by injecting fake nodes instead of modifying existing structures or node attributes. Inspired by findings that the adversarial attacks are related to the increased heterophily on perturbed graphs (the adversary tends to connect dissimilar nodes), we propose a general defense framework CHAGNN against GIA through cooperative homophilous augmentation of graph data and model. Specifically, the model generates pseudo-labels for unlabeled nodes in each round of training to reduce heterophilous edges of nodes with distinct labels. The cleaner graph is fed back to the model, producing more informative pseudo-labels. In such an iterative manner, model robustness is then promisingly enhanced. We present the theoretical analysis of the effect of homophilous augmentation and provide the guarantee of the proposal's validity. Experimental results empirically demonstrate the effectiveness of CHAGNN in comparison with recent state-of-the-art defense methods on diverse real-world datasets.



## **23. MORA: Improving Ensemble Robustness Evaluation with Model-Reweighing Attack**

cs.LG

To appear in NeurIPS 2022. Project repository:  https://github.com/lafeat/mora

**SubmitDate**: 2022-11-15    [abs](http://arxiv.org/abs/2211.08008v1) [paper-pdf](http://arxiv.org/pdf/2211.08008v1)

**Authors**: Yunrui Yu, Xitong Gao, Cheng-Zhong Xu

**Abstract**: Adversarial attacks can deceive neural networks by adding tiny perturbations to their input data. Ensemble defenses, which are trained to minimize attack transferability among sub-models, offer a promising research direction to improve robustness against such attacks while maintaining a high accuracy on natural inputs. We discover, however, that recent state-of-the-art (SOTA) adversarial attack strategies cannot reliably evaluate ensemble defenses, sizeably overestimating their robustness. This paper identifies the two factors that contribute to this behavior. First, these defenses form ensembles that are notably difficult for existing gradient-based method to attack, due to gradient obfuscation. Second, ensemble defenses diversify sub-model gradients, presenting a challenge to defeat all sub-models simultaneously, simply summing their contributions may counteract the overall attack objective; yet, we observe that ensemble may still be fooled despite most sub-models being correct. We therefore introduce MORA, a model-reweighing attack to steer adversarial example synthesis by reweighing the importance of sub-model gradients. MORA finds that recent ensemble defenses all exhibit varying degrees of overestimated robustness. Comparing it against recent SOTA white-box attacks, it can converge orders of magnitude faster while achieving higher attack success rates across all ensemble models examined with three different ensemble modes (i.e., ensembling by either softmax, voting or logits). In particular, most ensemble defenses exhibit near or exactly 0% robustness against MORA with $\ell^\infty$ perturbation within 0.02 on CIFAR-10, and 0.01 on CIFAR-100. We make MORA open source with reproducible results and pre-trained models; and provide a leaderboard of ensemble defenses under various attack strategies.



## **24. Security Closure of IC Layouts Against Hardware Trojans**

cs.CR

To appear in ISPD'23

**SubmitDate**: 2022-11-15    [abs](http://arxiv.org/abs/2211.07997v1) [paper-pdf](http://arxiv.org/pdf/2211.07997v1)

**Authors**: Fangzhou Wang, Qijing Wang, Bangqi Fu, Shui Jiang, Xiaopeng Zhang, Lilas Alrahis, Ozgur Sinanoglu, Johann Knechtel, Tsung-Yi Ho, Evangeline F. Y. Young

**Abstract**: Due to cost benefits, supply chains of integrated circuits (ICs) are largely outsourced nowadays. However, passing ICs through various third-party providers gives rise to many threats, like piracy of IC intellectual property or insertion of hardware Trojans, i.e., malicious circuit modifications.   In this work, we proactively and systematically harden the physical layouts of ICs against post-design insertion of Trojans. Toward that end, we propose a multiplexer-based logic-locking scheme that is (i) devised for layout-level Trojan prevention, (ii) resilient against state-of-the-art, oracle-less machine learning attacks, and (iii) fully integrated into a tailored, yet generic, commercial-grade design flow. Our work provides in-depth security and layout analysis on a challenging benchmark suite. We show that ours can render layouts resilient, with reasonable overheads, against Trojan insertion in general and also against second-order attacks (i.e., adversaries seeking to bypass the locking defense in an oracle-less setting).   We release our layout artifacts for independent verification [29] and we will release our methodology's source code.



## **25. Towards Robust Numerical Question Answering: Diagnosing Numerical Capabilities of NLP Systems**

cs.CL

Accepted by EMNLP'2022

**SubmitDate**: 2022-11-14    [abs](http://arxiv.org/abs/2211.07455v1) [paper-pdf](http://arxiv.org/pdf/2211.07455v1)

**Authors**: Jialiang Xu, Mengyu Zhou, Xinyi He, Shi Han, Dongmei Zhang

**Abstract**: Numerical Question Answering is the task of answering questions that require numerical capabilities. Previous works introduce general adversarial attacks to Numerical Question Answering, while not systematically exploring numerical capabilities specific to the topic. In this paper, we propose to conduct numerical capability diagnosis on a series of Numerical Question Answering systems and datasets. A series of numerical capabilities are highlighted, and corresponding dataset perturbations are designed. Empirical results indicate that existing systems are severely challenged by these perturbations. E.g., Graph2Tree experienced a 53.83% absolute accuracy drop against the ``Extra'' perturbation on ASDiv-a, and BART experienced 13.80% accuracy drop against the ``Language'' perturbation on the numerical subset of DROP. As a counteracting approach, we also investigate the effectiveness of applying perturbations as data augmentation to relieve systems' lack of robust numerical capabilities. With experiment analysis and empirical studies, it is demonstrated that Numerical Question Answering with robust numerical capabilities is still to a large extent an open question. We discuss future directions of Numerical Question Answering and summarize guidelines on future dataset collection and system design.



## **26. Privacy and Security in Network Controlled Systems via Dynamic Masking**

eess.SY

**SubmitDate**: 2022-11-14    [abs](http://arxiv.org/abs/2211.07328v1) [paper-pdf](http://arxiv.org/pdf/2211.07328v1)

**Authors**: Mohamed Abdalmoaty, Sribalaji C. Anand, André M. H. Teixeira

**Abstract**: In this paper, we propose a new architecture to enhance the privacy and security of networked control systems against malicious adversaries. We consider an adversary which first learns the system dynamics (privacy) using system identification techniques, and then performs a data injection attack (security). In particular, we consider an adversary conducting zero-dynamics attacks (ZDA) which maximizes the performance cost of the system whilst staying undetected. However, using the proposed architecture, we show that it is possible to (i) introduce significant bias in the system estimates of the adversary: thus providing privacy of the system parameters, and (ii) efficiently detect attacks when the adversary performs a ZDA using the identified system: thus providing security. Through numerical simulations, we illustrate the efficacy of the proposed architecture.



## **27. Jacobian Norm with Selective Input Gradient Regularization for Improved and Interpretable Adversarial Defense**

cs.LG

Under review

**SubmitDate**: 2022-11-14    [abs](http://arxiv.org/abs/2207.13036v4) [paper-pdf](http://arxiv.org/pdf/2207.13036v4)

**Authors**: Deyin Liu, Lin Wu, Haifeng Zhao, Farid Boussaid, Mohammed Bennamoun, Xianghua Xie

**Abstract**: Deep neural networks (DNNs) are known to be vulnerable to adversarial examples that are crafted with imperceptible perturbations, i.e., a small change in an input image can induce a mis-classification, and thus threatens the reliability of deep learning based deployment systems. Adversarial training (AT) is often adopted to improve robustness through training a mixture of corrupted and clean data. However, most of AT based methods are ineffective in dealing with transferred adversarial examples which are generated to fool a wide spectrum of defense models, and thus cannot satisfy the generalization requirement raised in real-world scenarios. Moreover, adversarially training a defense model in general cannot produce interpretable predictions towards the inputs with perturbations, whilst a highly interpretable robust model is required by different domain experts to understand the behaviour of a DNN. In this work, we propose a novel approach based on Jacobian norm and Selective Input Gradient Regularization (J-SIGR), which suggests the linearized robustness through Jacobian normalization and also regularizes the perturbation-based saliency maps to imitate the model's interpretable predictions. As such, we achieve both the improved defense and high interpretability of DNNs. Finally, we evaluate our method across different architectures against powerful adversarial attacks. Experiments demonstrate that the proposed J-SIGR confers improved robustness against transferred adversarial attacks, and we also show that the predictions from the neural network are easy to interpret.



## **28. Securing Access to Untrusted Services From TEEs with GateKeeper**

cs.CR

**SubmitDate**: 2022-11-14    [abs](http://arxiv.org/abs/2211.07185v1) [paper-pdf](http://arxiv.org/pdf/2211.07185v1)

**Authors**: Meni Orenbach, Bar Raveh, Alon Berkenstadt, Yan Michalevsky, Shachar Itzhaky, Mark Silberstein

**Abstract**: Applications running in Trusted Execution Environments (TEEs) commonly use untrusted external services such as host File System. Adversaries may maliciously alter the normal service behavior to trigger subtle application bugs that would have never occurred under correct service operation, causing data leaks and integrity violations. Unfortunately, existing manual protections are incomplete and ad-hoc, whereas formally-verified ones require special expertise.   We introduce GateKeeper, a framework to develop mitigations and vulnerability checkers for such attacks by leveraging lightweight formal models of untrusted services. With the attack seen as a violation of a services' functional correctness, GateKeeper takes a novel approach to develop a comprehensive model of a service without requiring formal methods expertise. We harness available testing suites routinely used in service development to tighten the model to known correct service implementation. GateKeeper uses the resulting model to automatically generate (1) a correct-by-construction runtime service validator in C that is linked with a trusted application and guards each service invocation to conform to the model; and (2) a targeted model-driven vulnerability checker for analyzing black-box applications.   We evaluate GateKeeper on Intel SGX enclaves. We develop comprehensive models of a POSIX file system and OS synchronization primitives while using thousands of existing test suites to tighten their models to the actual Linux implementations. We generate the validator and integrate it with Graphene-SGX, and successfully protect unmodified Memcached and SQLite with negligible overheads. The generated vulnerability checker detects novel vulnerabilities in the Graphene-SGX protection layer and production applications.



## **29. Robust Deep Semi-Supervised Learning: A Brief Introduction**

cs.LG

We will rewrite this paper

**SubmitDate**: 2022-11-14    [abs](http://arxiv.org/abs/2202.05975v2) [paper-pdf](http://arxiv.org/pdf/2202.05975v2)

**Authors**: Lan-Zhe Guo, Zhi Zhou, Yu-Feng Li

**Abstract**: Semi-supervised learning (SSL) is the branch of machine learning that aims to improve learning performance by leveraging unlabeled data when labels are insufficient. Recently, SSL with deep models has proven to be successful on standard benchmark tasks. However, they are still vulnerable to various robustness threats in real-world applications as these benchmarks provide perfect unlabeled data, while in realistic scenarios, unlabeled data could be corrupted. Many researchers have pointed out that after exploiting corrupted unlabeled data, SSL suffers severe performance degradation problems. Thus, there is an urgent need to develop SSL algorithms that could work robustly with corrupted unlabeled data. To fully understand robust SSL, we conduct a survey study. We first clarify a formal definition of robust SSL from the perspective of machine learning. Then, we classify the robustness threats into three categories: i) distribution corruption, i.e., unlabeled data distribution is mismatched with labeled data; ii) feature corruption, i.e., the features of unlabeled examples are adversarially attacked; and iii) label corruption, i.e., the label distribution of unlabeled data is imbalanced. Under this unified taxonomy, we provide a thorough review and discussion of recent works that focus on these issues. Finally, we propose possible promising directions within robust SSL to provide insights for future research.



## **30. Optimization for Robustness Evaluation beyond $\ell_p$ Metrics**

cs.LG

5 pages, 1 figure, 3 tables, accepted by the 14th International OPT  Workshop on Optimization for Machine Learning, and submitted to the 2023 IEEE  International Conference on Acoustics, Speech, and Signal Processing (ICASSP  2023)

**SubmitDate**: 2022-11-14    [abs](http://arxiv.org/abs/2210.00621v2) [paper-pdf](http://arxiv.org/pdf/2210.00621v2)

**Authors**: Hengyue Liang, Buyun Liang, Ying Cui, Tim Mitchell, Ju Sun

**Abstract**: Empirical evaluation of deep learning models against adversarial attacks entails solving nontrivial constrained optimization problems. Popular algorithms for solving these constrained problems rely on projected gradient descent (PGD) and require careful tuning of multiple hyperparameters. Moreover, PGD can only handle $\ell_1$, $\ell_2$, and $\ell_\infty$ attack models due to the use of analytical projectors. In this paper, we introduce a novel algorithmic framework that blends a general-purpose constrained-optimization solver PyGRANSO, With Constraint-Folding (PWCF), to add reliability and generality to robustness evaluation. PWCF 1) finds good-quality solutions without the need of delicate hyperparameter tuning, and 2) can handle general attack models, e.g., general $\ell_p$ ($p \geq 0$) and perceptual attacks, which are inaccessible to PGD-based algorithms.



## **31. Watermarking Graph Neural Networks based on Backdoor Attacks**

cs.LG

18 pages, 9 figures

**SubmitDate**: 2022-11-13    [abs](http://arxiv.org/abs/2110.11024v5) [paper-pdf](http://arxiv.org/pdf/2110.11024v5)

**Authors**: Jing Xu, Stefanos Koffas, Oguzhan Ersoy, Stjepan Picek

**Abstract**: Graph Neural Networks (GNNs) have achieved promising performance in various real-world applications. Building a powerful GNN model is not a trivial task, as it requires a large amount of training data, powerful computing resources, and human expertise in fine-tuning the model. Moreover, with the development of adversarial attacks, e.g., model stealing attacks, GNNs raise challenges to model authentication. To avoid copyright infringement on GNNs, verifying the ownership of the GNN models is necessary.   This paper presents a watermarking framework for GNNs for both graph and node classification tasks. We 1) design two strategies to generate watermarked data for the graph classification task and one for the node classification task, 2) embed the watermark into the host model through training to obtain the watermarked GNN model, and 3) verify the ownership of the suspicious model in a black-box setting. The experiments show that our framework can verify the ownership of GNN models with a very high probability (up to $99\%$) for both tasks. Finally, we experimentally show that our watermarking approach is robust against a state-of-the-art model extraction technique and four state-of-the-art defenses against backdoor attacks.



## **32. Physical-World Optical Adversarial Attacks on 3D Face Recognition**

cs.CV

Submitted to CVPR 2023

**SubmitDate**: 2022-11-13    [abs](http://arxiv.org/abs/2205.13412v3) [paper-pdf](http://arxiv.org/pdf/2205.13412v3)

**Authors**: Yanjie Li, Yiquan Li, Xuelong Dai, Songtao Guo, Bin Xiao

**Abstract**: 2D face recognition has been proven insecure for physical adversarial attacks. However, few studies have investigated the possibility of attacking real-world 3D face recognition systems. 3D-printed attacks recently proposed cannot generate adversarial points in the air. In this paper, we attack 3D face recognition systems through elaborate optical noises. We took structured light 3D scanners as our attack target. End-to-end attack algorithms are designed to generate adversarial illumination for 3D faces through the inherent or an additional projector to produce adversarial points at arbitrary positions. Nevertheless, face reflectance is a complex procedure because the skin is translucent. To involve this projection-and-capture procedure in optimization loops, we model it by Lambertian rendering model and use SfSNet to estimate the albedo. Moreover, to improve the resistance to distance and angle changes while maintaining the perturbation unnoticeable, a 3D transform invariant loss and two kinds of sensitivity maps are introduced. Experiments are conducted in both simulated and physical worlds. We successfully attacked point-cloud-based and depth-image-based 3D face recognition algorithms while needing fewer perturbations than previous state-of-the-art physical-world 3D adversarial attacks.



## **33. Adversarial Attacks and Defenses in Physiological Computing: A Systematic Review**

cs.LG

National Science Open, 2022

**SubmitDate**: 2022-11-13    [abs](http://arxiv.org/abs/2102.02729v4) [paper-pdf](http://arxiv.org/pdf/2102.02729v4)

**Authors**: Dongrui Wu, Jiaxin Xu, Weili Fang, Yi Zhang, Liuqing Yang, Xiaodong Xu, Hanbin Luo, Xiang Yu

**Abstract**: Physiological computing uses human physiological data as system inputs in real time. It includes, or significantly overlaps with, brain-computer interfaces, affective computing, adaptive automation, health informatics, and physiological signal based biometrics. Physiological computing increases the communication bandwidth from the user to the computer, but is also subject to various types of adversarial attacks, in which the attacker deliberately manipulates the training and/or test examples to hijack the machine learning algorithm output, leading to possible user confusion, frustration, injury, or even death. However, the vulnerability of physiological computing systems has not been paid enough attention to, and there does not exist a comprehensive review on adversarial attacks to them. This paper fills this gap, by providing a systematic review on the main research areas of physiological computing, different types of adversarial attacks and their applications to physiological computing, and the corresponding defense strategies. We hope this review will attract more research interests on the vulnerability of physiological computing systems, and more importantly, defense strategies to make them more secure.



## **34. TrojViT: Trojan Insertion in Vision Transformers**

cs.LG

10 pages, 4 figures, 10 tables

**SubmitDate**: 2022-11-13    [abs](http://arxiv.org/abs/2208.13049v2) [paper-pdf](http://arxiv.org/pdf/2208.13049v2)

**Authors**: Mengxin Zheng, Qian Lou, Lei Jiang

**Abstract**: Vision Transformers (ViTs) have demonstrated the state-of-the-art performance in various vision-related tasks. The success of ViTs motivates adversaries to perform backdoor attacks on ViTs. Although the vulnerability of traditional CNNs to backdoor attacks is well-known, backdoor attacks on ViTs are seldom-studied. Compared to CNNs capturing pixel-wise local features by convolutions, ViTs extract global context information through patches and attentions. Na\"ively transplanting CNN-specific backdoor attacks to ViTs yields only a low clean data accuracy and a low attack success rate. In this paper, we propose a stealth and practical ViT-specific backdoor attack $TrojViT$. Rather than an area-wise trigger used by CNN-specific backdoor attacks, TrojViT generates a patch-wise trigger designed to build a Trojan composed of some vulnerable bits on the parameters of a ViT stored in DRAM memory through patch salience ranking and attention-target loss. TrojViT further uses minimum-tuned parameter update to reduce the bit number of the Trojan. Once the attacker inserts the Trojan into the ViT model by flipping the vulnerable bits, the ViT model still produces normal inference accuracy with benign inputs. But when the attacker embeds a trigger into an input, the ViT model is forced to classify the input to a predefined target class. We show that flipping only few vulnerable bits identified by TrojViT on a ViT model using the well-known RowHammer can transform the model into a backdoored one. We perform extensive experiments of multiple datasets on various ViT models. TrojViT can classify $99.64\%$ of test images to a target class by flipping $345$ bits on a ViT for ImageNet.



## **35. SoftHebb: Bayesian Inference in Unsupervised Hebbian Soft Winner-Take-All Networks**

cs.LG

**SubmitDate**: 2022-11-12    [abs](http://arxiv.org/abs/2107.05747v4) [paper-pdf](http://arxiv.org/pdf/2107.05747v4)

**Authors**: Timoleon Moraitis, Dmitry Toichkin, Adrien Journé, Yansong Chua, Qinghai Guo

**Abstract**: Hebbian plasticity in winner-take-all (WTA) networks is highly attractive for neuromorphic on-chip learning, owing to its efficient, local, unsupervised, and on-line nature. Moreover, its biological plausibility may help overcome important limitations of artificial algorithms, such as their susceptibility to adversarial attacks, and their high demands for training-example quantity and repetition. However, Hebbian WTA learning has found little use in machine learning (ML), likely because it has been missing an optimization theory compatible with deep learning (DL). Here we show rigorously that WTA networks constructed by standard DL elements, combined with a Hebbian-like plasticity that we derive, maintain a Bayesian generative model of the data. Importantly, without any supervision, our algorithm, SoftHebb, minimizes cross-entropy, i.e. a common loss function in supervised DL. We show this theoretically and in practice. The key is a "soft" WTA where there is no absolute "hard" winner neuron. Strikingly, in shallow-network comparisons with backpropagation (BP), SoftHebb shows advantages beyond its Hebbian efficiency. Namely, it converges in fewer iterations, and is significantly more robust to noise and adversarial attacks. Notably, attacks that maximally confuse SoftHebb are also confusing to the human eye, potentially linking human perceptual robustness, with Hebbian WTA circuits of cortex. Finally, SoftHebb can generate synthetic objects as interpolations of real object classes. All in all, Hebbian efficiency, theoretical underpinning, cross-entropy-minimization, and surprising empirical advantages, suggest that SoftHebb may inspire highly neuromorphic and radically different, but practical and advantageous learning algorithms and hardware accelerators.



## **36. Practical No-box Adversarial Attacks with Training-free Hybrid Image Transformation**

cs.CV

**SubmitDate**: 2022-11-12    [abs](http://arxiv.org/abs/2203.04607v2) [paper-pdf](http://arxiv.org/pdf/2203.04607v2)

**Authors**: Qilong Zhang, Chaoning Zhang, Chaoqun Li, Jingkuan Song, Lianli Gao

**Abstract**: In recent years, the adversarial vulnerability of deep neural networks (DNNs) has raised increasing attention. Among all the threat models, no-box attacks are the most practical but extremely challenging since they neither rely on any knowledge of the target model or similar substitute model, nor access the dataset for training a new substitute model. Although a recent method has attempted such an attack in a loose sense, its performance is not good enough and computational overhead of training is expensive. In this paper, we move a step forward and show the existence of a \textbf{training-free} adversarial perturbation under the no-box threat model, which can be successfully used to attack different DNNs in real-time. Motivated by our observation that high-frequency component (HFC) domains in low-level features and plays a crucial role in classification, we attack an image mainly by manipulating its frequency components. Specifically, the perturbation is manipulated by suppression of the original HFC and adding of noisy HFC. We empirically and experimentally analyze the requirements of effective noisy HFC and show that it should be regionally homogeneous, repeating and dense. Extensive experiments on the ImageNet dataset demonstrate the effectiveness of our proposed no-box method. It attacks ten well-known models with a success rate of \textbf{98.13\%} on average, which outperforms state-of-the-art no-box attacks by \textbf{29.39\%}. Furthermore, our method is even competitive to mainstream transfer-based black-box attacks.



## **37. Generating Textual Adversaries with Minimal Perturbation**

cs.CL

To appear in EMNLP Findings 2022. The code is available at  https://github.com/xingyizhao/TAMPERS

**SubmitDate**: 2022-11-12    [abs](http://arxiv.org/abs/2211.06571v1) [paper-pdf](http://arxiv.org/pdf/2211.06571v1)

**Authors**: Xingyi Zhao, Lu Zhang, Depeng Xu, Shuhan Yuan

**Abstract**: Many word-level adversarial attack approaches for textual data have been proposed in recent studies. However, due to the massive search space consisting of combinations of candidate words, the existing approaches face the problem of preserving the semantics of texts when crafting adversarial counterparts. In this paper, we develop a novel attack strategy to find adversarial texts with high similarity to the original texts while introducing minimal perturbation. The rationale is that we expect the adversarial texts with small perturbation can better preserve the semantic meaning of original texts. Experiments show that, compared with state-of-the-art attack approaches, our approach achieves higher success rates and lower perturbation rates in four benchmark datasets.



## **38. An investigation of security controls and MITRE ATT\&CK techniques**

cs.CR

**SubmitDate**: 2022-11-11    [abs](http://arxiv.org/abs/2211.06500v1) [paper-pdf](http://arxiv.org/pdf/2211.06500v1)

**Authors**: Md Rayhanur Rahman, Laurie Williams

**Abstract**: Attackers utilize a plethora of adversarial techniques in cyberattacks to compromise the confidentiality, integrity, and availability of the target organizations and systems. Information security standards such as NIST, ISO/IEC specify hundreds of security controls that organizations can enforce to protect and defend the information systems from adversarial techniques. However, implementing all the available controls at the same time can be infeasible and security controls need to be investigated in terms of their mitigation ability over adversarial techniques used in cyberattacks as well. The goal of this research is to aid organizations in making informed choices on security controls to defend against cyberthreats through an investigation of adversarial techniques used in current cyberattacks. In this study, we investigated the extent of mitigation of 298 NIST SP800-53 controls over 188 adversarial techniques used in 669 cybercrime groups and malware cataloged in the MITRE ATT\&CK framework based upon an existing mapping between the controls and techniques. We identify that, based on the mapping, only 101 out of 298 control are capable of mitigating adversarial techniques. However, we also identify that 53 adversarial techniques cannot be mitigated by any existing controls, and these techniques primarily aid adversaries in bypassing system defense and discovering targeted system information. We identify a set of 20 critical controls that can mitigate 134 adversarial techniques, and on average, can mitigate 72\% of all techniques used by 98\% of the cataloged adversaries in MITRE ATT\&CK. We urge organizations, that do not have any controls enforced in place, to implement the top controls identified in the study.



## **39. Blockchain Technology to Secure Bluetooth**

cs.CR

7 pages, 6 figures

**SubmitDate**: 2022-11-11    [abs](http://arxiv.org/abs/2211.06451v1) [paper-pdf](http://arxiv.org/pdf/2211.06451v1)

**Authors**: Athanasios Kalogiratos, Ioanna Kantzavelou

**Abstract**: Bluetooth is a communication technology used to wirelessly exchange data between devices. In the last few years there have been found a great number of security vulnerabilities, and adversaries are taking advantage of them causing harm and significant loss. Numerous system security updates have been approved and installed in order to sort out security holes and bugs, and prevent attacks that could expose personal or other valuable information. But those updates are not sufficient and appropriate and new bugs keep showing up. In Bluetooth technology, pairing is identified as the step where most bugs are found and most attacks target this particular process part of Bluetooth. A new technology that has been proved bulletproof when it comes to security and the exchange of sensitive information is Blockchain. Blockchain technology is promising to be incorporated well in a network of smart devices, and secure an Internet of Things (IoT), where Bluetooth technology is being extensively used. This work presents a vulnerability discovered in Bluetooth pairing process, and proposes a Blockchain solution approach to secure pairing and mitigate this vulnerability. The paper first introduces the Bluetooth technology and delves into how Blockchain technology can be a solution to certain security problems. Then a solution approach shows how Blockchain can be integrated and implemented to ensure the required level of security. Certain attack incidents on Bluetooth vulnerable points are examined and discussion and conclusions give the extension of the security related problems.



## **40. Test-time adversarial detection and robustness for localizing humans using ultra wide band channel impulse responses**

cs.LG

5 pages, 4 figures, ICASSP Conference

**SubmitDate**: 2022-11-10    [abs](http://arxiv.org/abs/2211.05854v1) [paper-pdf](http://arxiv.org/pdf/2211.05854v1)

**Authors**: Abhiram Kolli, Muhammad Jehanzeb Mirza, Horst Possegger, Horst Bischof

**Abstract**: Keyless entry systems in cars are adopting neural networks for localizing its operators. Using test-time adversarial defences equip such systems with the ability to defend against adversarial attacks without prior training on adversarial samples. We propose a test-time adversarial example detector which detects the input adversarial example through quantifying the localized intermediate responses of a pre-trained neural network and confidence scores of an auxiliary softmax layer. Furthermore, in order to make the network robust, we extenuate the non-relevant features by non-iterative input sample clipping. Using our approach, mean performance over 15 levels of adversarial perturbations is increased by 55.33% for the fast gradient sign method (FGSM) and 6.3% for both the basic iterative method (BIM) and the projected gradient method (PGD).



## **41. A Practical Introduction to Side-Channel Extraction of Deep Neural Network Parameters**

cs.CR

Accepted at Smart Card Research and Advanced Application Conference  (CARDIS 2022)

**SubmitDate**: 2022-11-10    [abs](http://arxiv.org/abs/2211.05590v1) [paper-pdf](http://arxiv.org/pdf/2211.05590v1)

**Authors**: Raphael Joud, Pierre-Alain Moellic, Simon Pontie, Jean-Baptiste Rigaud

**Abstract**: Model extraction is a major threat for embedded deep neural network models that leverages an extended attack surface. Indeed, by physically accessing a device, an adversary may exploit side-channel leakages to extract critical information of a model (i.e., its architecture or internal parameters). Different adversarial objectives are possible including a fidelity-based scenario where the architecture and parameters are precisely extracted (model cloning). We focus this work on software implementation of deep neural networks embedded in a high-end 32-bit microcontroller (Cortex-M7) and expose several challenges related to fidelity-based parameters extraction through side-channel analysis, from the basic multiplication operation to the feed-forward connection through the layers. To precisely extract the value of parameters represented in the single-precision floating point IEEE-754 standard, we propose an iterative process that is evaluated with both simulations and traces from a Cortex-M7 target. To our knowledge, this work is the first to target such an high-end 32-bit platform. Importantly, we raise and discuss the remaining challenges for the complete extraction of a deep neural network model, more particularly the critical case of biases.



## **42. Impact of Adversarial Training on Robustness and Generalizability of Language Models**

cs.CL

**SubmitDate**: 2022-11-10    [abs](http://arxiv.org/abs/2211.05523v1) [paper-pdf](http://arxiv.org/pdf/2211.05523v1)

**Authors**: Enes Altinisik, Hassan Sajjad, Husrev Taha Sencar, Safa Messaoud, Sanjay Chawla

**Abstract**: Adversarial training is widely acknowledged as the most effective defense against adversarial attacks. However, it is also well established that achieving both robustness and generalization in adversarially trained models involves a trade-off. The goal of this work is to provide an in depth comparison of different approaches for adversarial training in language models. Specifically, we study the effect of pre-training data augmentation as well as training time input perturbations vs. embedding space perturbations on the robustness and generalization of BERT-like language models. Our findings suggest that better robustness can be achieved by pre-training data augmentation or by training with input space perturbation. However, training with embedding space perturbation significantly improves generalization. A linguistic correlation analysis of neurons of the learned models reveal that the improved generalization is due to `more specialized' neurons. To the best of our knowledge, this is the first work to carry out a deep qualitative analysis of different methods of generating adversarial examples in adversarial training of language models.



## **43. On the Privacy Risks of Algorithmic Recourse**

cs.LG

**SubmitDate**: 2022-11-10    [abs](http://arxiv.org/abs/2211.05427v1) [paper-pdf](http://arxiv.org/pdf/2211.05427v1)

**Authors**: Martin Pawelczyk, Himabindu Lakkaraju, Seth Neel

**Abstract**: As predictive models are increasingly being employed to make consequential decisions, there is a growing emphasis on developing techniques that can provide algorithmic recourse to affected individuals. While such recourses can be immensely beneficial to affected individuals, potential adversaries could also exploit these recourses to compromise privacy. In this work, we make the first attempt at investigating if and how an adversary can leverage recourses to infer private information about the underlying model's training data. To this end, we propose a series of novel membership inference attacks which leverage algorithmic recourse. More specifically, we extend the prior literature on membership inference attacks to the recourse setting by leveraging the distances between data instances and their corresponding counterfactuals output by state-of-the-art recourse methods. Extensive experimentation with real world and synthetic datasets demonstrates significant privacy leakage through recourses. Our work establishes unintended privacy leakage as an important risk in the widespread adoption of recourse methods.



## **44. Stay Home Safe with Starving Federated Data**

cs.LG

11 pages, 12 figures, 7 tables, accepted as a conference paper at  IEEE UV 2022, Boston, USA

**SubmitDate**: 2022-11-10    [abs](http://arxiv.org/abs/2211.05410v1) [paper-pdf](http://arxiv.org/pdf/2211.05410v1)

**Authors**: Jaechul Roh, Yajun Fang

**Abstract**: Over the past few years, the field of adversarial attack received numerous attention from various researchers with the help of successful attack success rate against well-known deep neural networks that were acknowledged to achieve high classification ability in various tasks. However, majority of the experiments were completed under a single model, which we believe it may not be an ideal case in a real-life situation. In this paper, we introduce a novel federated adversarial training method for smart home face recognition, named FLATS, where we observed some interesting findings that may not be easily noticed in a traditional adversarial attack to federated learning experiments. By applying different variations to the hyperparameters, we have spotted that our method can make the global model to be robust given a starving federated environment. Our code can be found on https://github.com/jcroh0508/FLATS.



## **45. Adversarial Training for High-Stakes Reliability**

cs.LG

30 pages, 7 figures, NeurIPS camera-ready

**SubmitDate**: 2022-11-10    [abs](http://arxiv.org/abs/2205.01663v5) [paper-pdf](http://arxiv.org/pdf/2205.01663v5)

**Authors**: Daniel M. Ziegler, Seraphina Nix, Lawrence Chan, Tim Bauman, Peter Schmidt-Nielsen, Tao Lin, Adam Scherlis, Noa Nabeshima, Ben Weinstein-Raun, Daniel de Haas, Buck Shlegeris, Nate Thomas

**Abstract**: In the future, powerful AI systems may be deployed in high-stakes settings, where a single failure could be catastrophic. One technique for improving AI safety in high-stakes settings is adversarial training, which uses an adversary to generate examples to train on in order to achieve better worst-case performance.   In this work, we used a safe language generation task (``avoid injuries'') as a testbed for achieving high reliability through adversarial training. We created a series of adversarial training techniques -- including a tool that assists human adversaries -- to find and eliminate failures in a classifier that filters text completions suggested by a generator. In our task, we determined that we can set very conservative classifier thresholds without significantly impacting the quality of the filtered outputs. We found that adversarial training increased robustness to the adversarial attacks that we trained on -- doubling the time for our contractors to find adversarial examples both with our tool (from 13 to 26 minutes) and without (from 20 to 44 minutes) -- without affecting in-distribution performance.   We hope to see further work in the high-stakes reliability setting, including more powerful tools for enhancing human adversaries and better ways to measure high levels of reliability, until we can confidently rule out the possibility of catastrophic deployment-time failures of powerful models.



## **46. Using Deception in Markov Game to Understand Adversarial Behaviors through a Capture-The-Flag Environment**

cs.GT

Accepted at GameSec 2022

**SubmitDate**: 2022-11-09    [abs](http://arxiv.org/abs/2210.15011v2) [paper-pdf](http://arxiv.org/pdf/2210.15011v2)

**Authors**: Siddhant Bhambri, Purv Chauhan, Frederico Araujo, Adam Doupé, Subbarao Kambhampati

**Abstract**: Identifying the actual adversarial threat against a system vulnerability has been a long-standing challenge for cybersecurity research. To determine an optimal strategy for the defender, game-theoretic based decision models have been widely used to simulate the real-world attacker-defender scenarios while taking the defender's constraints into consideration. In this work, we focus on understanding human attacker behaviors in order to optimize the defender's strategy. To achieve this goal, we model attacker-defender engagements as Markov Games and search for their Bayesian Stackelberg Equilibrium. We validate our modeling approach and report our empirical findings using a Capture-The-Flag (CTF) setup, and we conduct user studies on adversaries with varying skill-levels. Our studies show that application-level deceptions are an optimal mitigation strategy against targeted attacks -- outperforming classic cyber-defensive maneuvers, such as patching or blocking network requests. We use this result to further hypothesize over the attacker's behaviors when trapped in an embedded honeypot environment and present a detailed analysis of the same.



## **47. Are All Edges Necessary? A Unified Framework for Graph Purification**

cs.SI

**SubmitDate**: 2022-11-09    [abs](http://arxiv.org/abs/2211.05184v1) [paper-pdf](http://arxiv.org/pdf/2211.05184v1)

**Authors**: Zishan Gu, Jintang Li, Liang Chen

**Abstract**: Graph Neural Networks (GNNs) as deep learning models working on graph-structure data have achieved advanced performance in many works. However, it has been proved repeatedly that, not all edges in a graph are necessary for the training of machine learning models. In other words, some of the connections between nodes may bring redundant or even misleading information to downstream tasks. In this paper, we try to provide a method to drop edges in order to purify the graph data from a new perspective. Specifically, it is a framework to purify graphs with the least loss of information, under which the core problems are how to better evaluate the edges and how to delete the relatively redundant edges with the least loss of information. To address the above two problems, we propose several measurements for the evaluation and different judges and filters for the edge deletion. We also introduce a residual-iteration strategy and a surrogate model for measurements requiring unknown information. The experimental results show that our proposed measurements for KL divergence with constraints to maintain the connectivity of the graph and delete edges in an iterative way can find out the most edges while keeping the performance of GNNs. What's more, further experiments show that this method also achieves the best defense performance against adversarial attacks.



## **48. Accountable and Explainable Methods for Complex Reasoning over Text**

cs.LG

PhD Thesis

**SubmitDate**: 2022-11-09    [abs](http://arxiv.org/abs/2211.04946v1) [paper-pdf](http://arxiv.org/pdf/2211.04946v1)

**Authors**: Pepa Atanasova

**Abstract**: A major concern of Machine Learning (ML) models is their opacity. They are deployed in an increasing number of applications where they often operate as black boxes that do not provide explanations for their predictions. Among others, the potential harms associated with the lack of understanding of the models' rationales include privacy violations, adversarial manipulations, and unfair discrimination. As a result, the accountability and transparency of ML models have been posed as critical desiderata by works in policy and law, philosophy, and computer science.   In computer science, the decision-making process of ML models has been studied by developing accountability and transparency methods. Accountability methods, such as adversarial attacks and diagnostic datasets, expose vulnerabilities of ML models that could lead to malicious manipulations or systematic faults in their predictions. Transparency methods explain the rationales behind models' predictions gaining the trust of relevant stakeholders and potentially uncovering mistakes and unfairness in models' decisions. To this end, transparency methods have to meet accountability requirements as well, e.g., being robust and faithful to the underlying rationales of a model.   This thesis presents my research that expands our collective knowledge in the areas of accountability and transparency of ML models developed for complex reasoning tasks over text.



## **49. Lipschitz Continuous Algorithms for Graph Problems**

cs.DS

**SubmitDate**: 2022-11-09    [abs](http://arxiv.org/abs/2211.04674v1) [paper-pdf](http://arxiv.org/pdf/2211.04674v1)

**Authors**: Soh Kumabe, Yuichi Yoshida

**Abstract**: It has been widely observed in the machine learning community that a small perturbation to the input can cause a large change in the prediction of a trained model, and such phenomena have been intensively studied in the machine learning community under the name of adversarial attacks. Because graph algorithms also are widely used for decision making and knowledge discovery, it is important to design graph algorithms that are robust against adversarial attacks. In this study, we consider the Lipschitz continuity of algorithms as a robustness measure and initiate a systematic study of the Lipschitz continuity of algorithms for (weighted) graph problems.   Depending on how we embed the output solution to a metric space, we can think of several Lipschitzness notions. We mainly consider the one that is invariant under scaling of weights, and we provide Lipschitz continuous algorithms and lower bounds for the minimum spanning tree problem, the shortest path problem, and the maximum weight matching problem. In particular, our shortest path algorithm is obtained by first designing an algorithm for unweighted graphs that are robust against edge contractions and then applying it to the unweighted graph constructed from the original weighted graph.   Then, we consider another Lipschitzness notion induced by a natural mapping that maps the output solution to its characteristic vector. It turns out that no Lipschitz continuous algorithm exists for this Lipschitz notion, and we instead design algorithms with bounded pointwise Lipschitz constants for the minimum spanning tree problem and the maximum weight bipartite matching problem. Our algorithm for the latter problem is based on an LP relaxation with entropy regularization.



## **50. FedDef: Defense Against Gradient Leakage in Federated Learning-based Network Intrusion Detection Systems**

cs.CR

14 pages, 9 figures, submitted to TIFS

**SubmitDate**: 2022-11-09    [abs](http://arxiv.org/abs/2210.04052v2) [paper-pdf](http://arxiv.org/pdf/2210.04052v2)

**Authors**: Jiahui Chen, Yi Zhao, Qi Li, Xuewei Feng, Ke Xu

**Abstract**: Deep learning (DL) methods have been widely applied to anomaly-based network intrusion detection system (NIDS) to detect malicious traffic. To expand the usage scenarios of DL-based methods, the federated learning (FL) framework allows multiple users to train a global model on the basis of respecting individual data privacy. However, it has not yet been systematically evaluated how robust FL-based NIDSs are against existing privacy attacks under existing defenses. To address this issue, we propose two privacy evaluation metrics designed for FL-based NIDSs, including (1) privacy score that evaluates the similarity between the original and recovered traffic features using reconstruction attacks, and (2) evasion rate against NIDSs using Generative Adversarial Network-based adversarial attack with the reconstructed benign traffic. We conduct experiments to show that existing defenses provide little protection that the corresponding adversarial traffic can even evade the SOTA NIDS Kitsune. To defend against such attacks and build a more robust FL-based NIDS, we further propose FedDef, a novel optimization-based input perturbation defense strategy with theoretical guarantee. It achieves both high utility by minimizing the gradient distance and strong privacy protection by maximizing the input distance. We experimentally evaluate four existing defenses on four datasets and show that our defense outperforms all the baselines in terms of privacy protection with up to 7 times higher privacy score, while maintaining model accuracy loss within 3% under optimal parameter combination.



