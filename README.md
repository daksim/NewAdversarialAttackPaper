# Latest Adversarial Attack Papers
**update at 2023-04-13 10:58:34**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Multi-Glimpse Network: A Robust and Efficient Classification Architecture based on Recurrent Downsampled Attention**

cs.CV

Accepted at BMVC 2021

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2111.02018v2) [paper-pdf](http://arxiv.org/pdf/2111.02018v2)

**Authors**: Sia Huat Tan, Runpei Dong, Kaisheng Ma

**Abstract**: Most feedforward convolutional neural networks spend roughly the same efforts for each pixel. Yet human visual recognition is an interaction between eye movements and spatial attention, which we will have several glimpses of an object in different regions. Inspired by this observation, we propose an end-to-end trainable Multi-Glimpse Network (MGNet) which aims to tackle the challenges of high computation and the lack of robustness based on recurrent downsampled attention mechanism. Specifically, MGNet sequentially selects task-relevant regions of an image to focus on and then adaptively combines all collected information for the final prediction. MGNet expresses strong resistance against adversarial attacks and common corruptions with less computation. Also, MGNet is inherently more interpretable as it explicitly informs us where it focuses during each iteration. Our experiments on ImageNet100 demonstrate the potential of recurrent downsampled attention mechanisms to improve a single feedforward manner. For example, MGNet improves 4.76% accuracy on average in common corruptions with only 36.9% computational cost. Moreover, while the baseline incurs an accuracy drop to 7.6%, MGNet manages to maintain 44.2% accuracy in the same PGD attack strength with ResNet-50 backbone. Our code is available at https://github.com/siahuat0727/MGNet.



## **2. Identification of Systematic Errors of Image Classifiers on Rare Subgroups**

cs.CV

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2303.05072v2) [paper-pdf](http://arxiv.org/pdf/2303.05072v2)

**Authors**: Jan Hendrik Metzen, Robin Hutmacher, N. Grace Hua, Valentyn Boreiko, Dan Zhang

**Abstract**: Despite excellent average-case performance of many image classifiers, their performance can substantially deteriorate on semantically coherent subgroups of the data that were under-represented in the training data. These systematic errors can impact both fairness for demographic minority groups as well as robustness and safety under domain shift. A major challenge is to identify such subgroups with subpar performance when the subgroups are not annotated and their occurrence is very rare. We leverage recent advances in text-to-image models and search in the space of textual descriptions of subgroups ("prompts") for subgroups where the target model has low performance on the prompt-conditioned synthesized data. To tackle the exponentially growing number of subgroups, we employ combinatorial testing. We denote this procedure as PromptAttack as it can be interpreted as an adversarial attack in a prompt space. We study subgroup coverage and identifiability with PromptAttack in a controlled setting and find that it identifies systematic errors with high accuracy. Thereupon, we apply PromptAttack to ImageNet classifiers and identify novel systematic errors on rare subgroups.



## **3. Optimal Detector Placement in Networked Control Systems under Cyber-attacks with Applications to Power Networks**

eess.SY

7 pages, 4 figures, accepted to IFAC 2023

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2304.05710v1) [paper-pdf](http://arxiv.org/pdf/2304.05710v1)

**Authors**: Anh Tung Nguyen, Sribalaji C. Anand, André M. H. Teixeira, Alexander Medvedev

**Abstract**: This paper proposes a game-theoretic method to address the problem of optimal detector placement in a networked control system under cyber-attacks. The networked control system is composed of interconnected agents where each agent is regulated by its local controller over unprotected communication, which leaves the system vulnerable to malicious cyber-attacks. To guarantee a given local performance, the defender optimally selects a single agent on which to place a detector at its local controller with the purpose of detecting cyber-attacks. On the other hand, an adversary optimally chooses a single agent on which to conduct a cyber-attack on its input with the aim of maximally worsening the local performance while remaining stealthy to the defender. First, we present a necessary and sufficient condition to ensure that the maximal attack impact on the local performance is bounded, which restricts the possible actions of the defender to a subset of available agents. Then, by considering the maximal attack impact on the local performance as a game payoff, we cast the problem of finding optimal actions of the defender and the adversary as a zero-sum game. Finally, with the possible action sets of the defender and the adversary, an algorithm is devoted to determining the Nash equilibria of the zero-sum game that yield the optimal detector placement. The proposed method is illustrated on an IEEE benchmark for power systems.



## **4. SoK: Certified Robustness for Deep Neural Networks**

cs.LG

To appear at 2023 IEEE Symposium on Security and Privacy (SP)  (Version 8); include recent progress till Apr 2023 in Version 9; 14 pages for  the main text; benchmark & tool website:  http://sokcertifiedrobustness.github.io/

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2009.04131v9) [paper-pdf](http://arxiv.org/pdf/2009.04131v9)

**Authors**: Linyi Li, Tao Xie, Bo Li

**Abstract**: Great advances in deep neural networks (DNNs) have led to state-of-the-art performance on a wide range of tasks. However, recent studies have shown that DNNs are vulnerable to adversarial attacks, which have brought great concerns when deploying these models to safety-critical applications such as autonomous driving. Different defense approaches have been proposed against adversarial attacks, including: a) empirical defenses, which can usually be adaptively attacked again without providing robustness certification; and b) certifiably robust approaches, which consist of robustness verification providing the lower bound of robust accuracy against any attacks under certain conditions and corresponding robust training approaches. In this paper, we systematize certifiably robust approaches and related practical and theoretical implications and findings. We also provide the first comprehensive benchmark on existing robustness verification and training approaches on different datasets. In particular, we 1) provide a taxonomy for the robustness verification and training approaches, as well as summarize the methodologies for representative algorithms, 2) reveal the characteristics, strengths, limitations, and fundamental connections among these approaches, 3) discuss current research progresses, theoretical barriers, main challenges, and future directions for certifiably robust approaches for DNNs, and 4) provide an open-sourced unified platform to evaluate 20+ representative certifiably robust approaches.



## **5. Generative Adversarial Networks-Driven Cyber Threat Intelligence Detection Framework for Securing Internet of Things**

cs.CR

The paper is accepted and will be published in the IEEE DCOSS-IoT  2023 Conference Proceedings

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2304.05644v1) [paper-pdf](http://arxiv.org/pdf/2304.05644v1)

**Authors**: Mohamed Amine Ferrag, Djallel Hamouda, Merouane Debbah, Leandros Maglaras, Abderrahmane Lakas

**Abstract**: While the benefits of 6G-enabled Internet of Things (IoT) are numerous, providing high-speed, low-latency communication that brings new opportunities for innovation and forms the foundation for continued growth in the IoT industry, it is also important to consider the security challenges and risks associated with the technology. In this paper, we propose a two-stage intrusion detection framework for securing IoTs, which is based on two detectors. In the first stage, we propose an adversarial training approach using generative adversarial networks (GAN) to help the first detector train on robust features by supplying it with adversarial examples as validation sets. Consequently, the classifier would perform very well against adversarial attacks. Then, we propose a deep learning (DL) model for the second detector to identify intrusions. We evaluated the proposed approach's efficiency in terms of detection accuracy and robustness against adversarial attacks. Experiment results with a new cyber security dataset demonstrate the effectiveness of the proposed methodology in detecting both intrusions and persistent adversarial examples with a weighted avg of 96%, 95%, 95%, and 95% for precision, recall, f1-score, and accuracy, respectively.



## **6. Overload: Latency Attacks on Object Detection for Edge Devices**

cs.CV

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2304.05370v2) [paper-pdf](http://arxiv.org/pdf/2304.05370v2)

**Authors**: Erh-Chung Chen, Pin-Yu Chen, I-Hsin Chung, Che-rung Lee

**Abstract**: Nowadays, the deployment of deep learning based applications on edge devices is an essential task owing to the increasing demands on intelligent services. However, the limited computing resources on edge nodes make the models vulnerable to attacks, such that the predictions made by models are unreliable. In this paper, we investigate latency attacks on deep learning applications. Unlike common adversarial attacks for misclassification, the goal of latency attacks is to increase the inference time, which may stop applications from responding to the requests within a reasonable time. This kind of attack is ubiquitous for various applications, and we use object detection to demonstrate how such kind of attacks work. We also design a framework named Overload to generate latency attacks at scale. Our method is based on a newly formulated optimization problem and a novel technique, called spatial attention, to increase the inference time of object detection. We have conducted experiments using YOLOv5 models on Nvidia NX. The experimental results show that with latency attacks, the inference time of a single image can be increased ten times longer in reference to the normal setting. Moreover, comparing to existing methods, our attacking method is simpler and more effective.



## **7. On the Adversarial Inversion of Deep Biometric Representations**

cs.CV

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2304.05561v1) [paper-pdf](http://arxiv.org/pdf/2304.05561v1)

**Authors**: Gioacchino Tangari, Shreesh Keskar, Hassan Jameel Asghar, Dali Kaafar

**Abstract**: Biometric authentication service providers often claim that it is not possible to reverse-engineer a user's raw biometric sample, such as a fingerprint or a face image, from its mathematical (feature-space) representation. In this paper, we investigate this claim on the specific example of deep neural network (DNN) embeddings. Inversion of DNN embeddings has been investigated for explaining deep image representations or synthesizing normalized images. Existing studies leverage full access to all layers of the original model, as well as all possible information on the original dataset. For the biometric authentication use case, we need to investigate this under adversarial settings where an attacker has access to a feature-space representation but no direct access to the exact original dataset nor the original learned model. Instead, we assume varying degree of attacker's background knowledge about the distribution of the dataset as well as the original learned model (architecture and training process). In these cases, we show that the attacker can exploit off-the-shelf DNN models and public datasets, to mimic the behaviour of the original learned model to varying degrees of success, based only on the obtained representation and attacker's prior knowledge. We propose a two-pronged attack that first infers the original DNN by exploiting the model footprint on the embedding, and then reconstructs the raw data by using the inferred model. We show the practicality of the attack on popular DNNs trained for two prominent biometric modalities, face and fingerprint recognition. The attack can effectively infer the original recognition model (mean accuracy 83\% for faces, 86\% for fingerprints), and can craft effective biometric reconstructions that are successfully authenticated with 1-vs-1 authentication accuracy of up to 92\% for some models.



## **8. Unfooling Perturbation-Based Post Hoc Explainers**

cs.AI

Accepted to AAAI-23. See the companion blog post at  https://medium.com/@craymichael/noncompliance-in-algorithmic-audits-and-defending-auditors-5b9fbdab2615.  9 pages (not including references and supplemental)

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2205.14772v3) [paper-pdf](http://arxiv.org/pdf/2205.14772v3)

**Authors**: Zachariah Carmichael, Walter J Scheirer

**Abstract**: Monumental advancements in artificial intelligence (AI) have lured the interest of doctors, lenders, judges, and other professionals. While these high-stakes decision-makers are optimistic about the technology, those familiar with AI systems are wary about the lack of transparency of its decision-making processes. Perturbation-based post hoc explainers offer a model agnostic means of interpreting these systems while only requiring query-level access. However, recent work demonstrates that these explainers can be fooled adversarially. This discovery has adverse implications for auditors, regulators, and other sentinels. With this in mind, several natural questions arise - how can we audit these black box systems? And how can we ascertain that the auditee is complying with the audit in good faith? In this work, we rigorously formalize this problem and devise a defense against adversarial attacks on perturbation-based explainers. We propose algorithms for the detection (CAD-Detect) and defense (CAD-Defend) of these attacks, which are aided by our novel conditional anomaly detection approach, KNN-CAD. We demonstrate that our approach successfully detects whether a black box system adversarially conceals its decision-making process and mitigates the adversarial attack on real-world data for the prevalent explainers, LIME and SHAP.



## **9. Existence and Minimax Theorems for Adversarial Surrogate Risks in Binary Classification**

cs.LG

37 pages. version 2: corrects several errors and employs a  significantly different proof technique. version 3: modifies the arXiv author  list but has no other changes

**SubmitDate**: 2023-04-11    [abs](http://arxiv.org/abs/2206.09098v3) [paper-pdf](http://arxiv.org/pdf/2206.09098v3)

**Authors**: Natalie S. Frank, Jonathan Niles-Weed

**Abstract**: Adversarial training is one of the most popular methods for training methods robust to adversarial attacks, however, it is not well-understood from a theoretical perspective. We prove and existence, regularity, and minimax theorems for adversarial surrogate risks. Our results explain some empirical observations on adversarial robustness from prior work and suggest new directions in algorithm development. Furthermore, our results extend previously known existence and minimax theorems for the adversarial classification risk to surrogate risks.



## **10. Fiat-Shamir for Proofs Lacks a Proof Even in the Presence of Shared Entanglement**

quant-ph

63 pages, 2 figures

**SubmitDate**: 2023-04-11    [abs](http://arxiv.org/abs/2204.02265v3) [paper-pdf](http://arxiv.org/pdf/2204.02265v3)

**Authors**: Frédéric Dupuis, Philippe Lamontagne, Louis Salvail

**Abstract**: We explore the cryptographic power of arbitrary shared physical resources. The most general such resource is access to a fresh entangled quantum state at the outset of each protocol execution. We call this the Common Reference Quantum State (CRQS) model, in analogy to the well-known Common Reference String (CRS). The CRQS model is a natural generalization of the CRS model but appears to be more powerful: in the two-party setting, a CRQS can sometimes exhibit properties associated with a Random Oracle queried once by measuring a maximally entangled state in one of many mutually unbiased bases. We formalize this notion as a Weak One-Time Random Oracle (WOTRO), where we only ask of the $m$--bit output to have some randomness when conditioned on the $n$--bit input.   We show that when $n-m\in\omega(\lg n)$, any protocol for WOTRO in the CRQS model can be attacked by an (inefficient) adversary. Moreover, our adversary is efficiently simulatable, which rules out the possibility of proving the computational security of a scheme by a black-box reduction to a cryptographic game assumption. On the other hand, we introduce a non-game quantum assumption for hash functions that implies WOTRO in the CRQ\$ model (where the CRQS consists only of EPR pairs). We first build a statistically secure WOTRO protocol where $m=n$, then hash the output.   The impossibility of WOTRO has the following consequences. First, we show the black-box impossibility of a quantum Fiat-Shamir transform, extending the impossibility result of Bitansky et al. (TCC '13) to the CRQS model. Second, we show a black-box impossibility result for a strenghtened version of quantum lightning (Zhandry, Eurocrypt '19) where quantum bolts have an additional parameter that cannot be changed without generating new bolts.



## **11. MENLI: Robust Evaluation Metrics from Natural Language Inference**

cs.CL

TACL 2023 Camera-ready; github link fixed+Fig.3 legend fixed

**SubmitDate**: 2023-04-11    [abs](http://arxiv.org/abs/2208.07316v4) [paper-pdf](http://arxiv.org/pdf/2208.07316v4)

**Authors**: Yanran Chen, Steffen Eger

**Abstract**: Recently proposed BERT-based evaluation metrics for text generation perform well on standard benchmarks but are vulnerable to adversarial attacks, e.g., relating to information correctness. We argue that this stems (in part) from the fact that they are models of semantic similarity. In contrast, we develop evaluation metrics based on Natural Language Inference (NLI), which we deem a more appropriate modeling. We design a preference-based adversarial attack framework and show that our NLI based metrics are much more robust to the attacks than the recent BERT-based metrics. On standard benchmarks, our NLI based metrics outperform existing summarization metrics, but perform below SOTA MT metrics. However, when combining existing metrics with our NLI metrics, we obtain both higher adversarial robustness (15%-30%) and higher quality metrics as measured on standard benchmarks (+5% to 30%).



## **12. Visually Adversarial Attacks and Defenses in the Physical World: A Survey**

cs.CV

**SubmitDate**: 2023-04-11    [abs](http://arxiv.org/abs/2211.01671v3) [paper-pdf](http://arxiv.org/pdf/2211.01671v3)

**Authors**: Xingxing Wei, Bangzheng Pu, Jiefan Lu, Baoyuan Wu

**Abstract**: Although Deep Neural Networks (DNNs) have been widely applied in various real-world scenarios, they are vulnerable to adversarial examples. The current adversarial attacks in computer vision can be divided into digital attacks and physical attacks according to their different attack forms. Compared with digital attacks, which generate perturbations in the digital pixels, physical attacks are more practical in the real world. Owing to the serious security problem caused by physically adversarial examples, many works have been proposed to evaluate the physically adversarial robustness of DNNs in the past years. In this paper, we summarize a survey versus the current physically adversarial attacks and physically adversarial defenses in computer vision. To establish a taxonomy, we organize the current physical attacks from attack tasks, attack forms, and attack methods, respectively. Thus, readers can have a systematic knowledge of this topic from different aspects. For the physical defenses, we establish the taxonomy from pre-processing, in-processing, and post-processing for the DNN models to achieve full coverage of the adversarial defenses. Based on the above survey, we finally discuss the challenges of this research field and further outlook on the future direction.



## **13. A Game-theoretic Framework for Federated Learning**

cs.LG

**SubmitDate**: 2023-04-11    [abs](http://arxiv.org/abs/2304.05836v1) [paper-pdf](http://arxiv.org/pdf/2304.05836v1)

**Authors**: Xiaojin Zhang, Lixin Fan, Siwei Wang, Wenjie Li, Kai Chen, Qiang Yang

**Abstract**: In federated learning, benign participants aim to optimize a global model collaboratively. However, the risk of \textit{privacy leakage} cannot be ignored in the presence of \textit{semi-honest} adversaries. Existing research has focused either on designing protection mechanisms or on inventing attacking mechanisms. While the battle between defenders and attackers seems never-ending, we are concerned with one critical question: is it possible to prevent potential attacks in advance? To address this, we propose the first game-theoretic framework that considers both FL defenders and attackers in terms of their respective payoffs, which include computational costs, FL model utilities, and privacy leakage risks. We name this game the Federated Learning Security Game (FLSG), in which neither defenders nor attackers are aware of all participants' payoffs.   To handle the \textit{incomplete information} inherent in this situation, we propose associating the FLSG with an \textit{oracle} that has two primary responsibilities. First, the oracle provides lower and upper bounds of the payoffs for the players. Second, the oracle acts as a correlation device, privately providing suggested actions to each player. With this novel framework, we analyze the optimal strategies of defenders and attackers. Furthermore, we derive and demonstrate conditions under which the attacker, as a rational decision-maker, should always follow the oracle's suggestion \textit{not to attack}.



## **14. RecUP-FL: Reconciling Utility and Privacy in Federated Learning via User-configurable Privacy Defense**

cs.LG

**SubmitDate**: 2023-04-11    [abs](http://arxiv.org/abs/2304.05135v1) [paper-pdf](http://arxiv.org/pdf/2304.05135v1)

**Authors**: Yue Cui, Syed Irfan Ali Meerza, Zhuohang Li, Luyang Liu, Jiaxin Zhang, Jian Liu

**Abstract**: Federated learning (FL) provides a variety of privacy advantages by allowing clients to collaboratively train a model without sharing their private data. However, recent studies have shown that private information can still be leaked through shared gradients. To further minimize the risk of privacy leakage, existing defenses usually require clients to locally modify their gradients (e.g., differential privacy) prior to sharing with the server. While these approaches are effective in certain cases, they regard the entire data as a single entity to protect, which usually comes at a large cost in model utility. In this paper, we seek to reconcile utility and privacy in FL by proposing a user-configurable privacy defense, RecUP-FL, that can better focus on the user-specified sensitive attributes while obtaining significant improvements in utility over traditional defenses. Moreover, we observe that existing inference attacks often rely on a machine learning model to extract the private information (e.g., attributes). We thus formulate such a privacy defense as an adversarial learning problem, where RecUP-FL generates slight perturbations that can be added to the gradients before sharing to fool adversary models. To improve the transferability to un-queryable black-box adversary models, inspired by the idea of meta-learning, RecUP-FL forms a model zoo containing a set of substitute models and iteratively alternates between simulations of the white-box and the black-box adversarial attack scenarios to generate perturbations. Extensive experiments on four datasets under various adversarial settings (both attribute inference attack and data reconstruction attack) show that RecUP-FL can meet user-specified privacy constraints over the sensitive attributes while significantly improving the model utility compared with state-of-the-art privacy defenses.



## **15. EGC: Image Generation and Classification via a Single Energy-Based Model**

cs.CV

**SubmitDate**: 2023-04-11    [abs](http://arxiv.org/abs/2304.02012v2) [paper-pdf](http://arxiv.org/pdf/2304.02012v2)

**Authors**: Qiushan Guo, Chuofan Ma, Yi Jiang, Zehuan Yuan, Yizhou Yu, Ping Luo

**Abstract**: Learning image classification and image generation using the same set of network parameters is a challenging problem. Recent advanced approaches perform well in one task often exhibit poor performance in the other. This work introduces an energy-based classifier and generator, namely EGC, which can achieve superior performance in both tasks using a single neural network. Unlike a conventional classifier that outputs a label given an image (i.e., a conditional distribution $p(y|\mathbf{x})$), the forward pass in EGC is a classifier that outputs a joint distribution $p(\mathbf{x},y)$, enabling an image generator in its backward pass by marginalizing out the label $y$. This is done by estimating the energy and classification probability given a noisy image in the forward pass, while denoising it using the score function estimated in the backward pass. EGC achieves competitive generation results compared with state-of-the-art approaches on ImageNet-1k, CelebA-HQ and LSUN Church, while achieving superior classification accuracy and robustness against adversarial attacks on CIFAR-10. This work represents the first successful attempt to simultaneously excel in both tasks using a single set of network parameters. We believe that EGC bridges the gap between discriminative and generative learning.



## **16. Non-Asymptotic Lower Bounds For Training Data Reconstruction**

cs.LG

Corrected minor typos

**SubmitDate**: 2023-04-11    [abs](http://arxiv.org/abs/2303.16372v3) [paper-pdf](http://arxiv.org/pdf/2303.16372v3)

**Authors**: Prateeti Mukherjee, Satya Lokam

**Abstract**: We investigate semantic guarantees of private learning algorithms for their resilience to training Data Reconstruction Attacks (DRAs) by informed adversaries. To this end, we derive non-asymptotic minimax lower bounds on the adversary's reconstruction error against learners that satisfy differential privacy (DP) and metric differential privacy (mDP). Furthermore, we demonstrate that our lower bound analysis for the latter also covers the high dimensional regime, wherein, the input data dimensionality may be larger than the adversary's query budget. Motivated by the theoretical improvements conferred by metric DP, we extend the privacy analysis of popular deep learning algorithms such as DP-SGD and Projected Noisy SGD to cover the broader notion of metric differential privacy.



## **17. Benchmarking the Physical-world Adversarial Robustness of Vehicle Detection**

cs.CV

CVPR 2023 workshop

**SubmitDate**: 2023-04-11    [abs](http://arxiv.org/abs/2304.05098v1) [paper-pdf](http://arxiv.org/pdf/2304.05098v1)

**Authors**: Tianyuan Zhang, Yisong Xiao, Xiaoya Zhang, Hao Li, Lu Wang

**Abstract**: Adversarial attacks in the physical world can harm the robustness of detection models. Evaluating the robustness of detection models in the physical world can be challenging due to the time-consuming and labor-intensive nature of many experiments. Thus, virtual simulation experiments can provide a solution to this challenge. However, there is no unified detection benchmark based on virtual simulation environment. To address this challenge, we proposed an instant-level data generation pipeline based on the CARLA simulator. Using this pipeline, we generated the DCI dataset and conducted extensive experiments on three detection models and three physical adversarial attacks. The dataset covers 7 continuous and 1 discrete scenes, with over 40 angles, 20 distances, and 20,000 positions. The results indicate that Yolo v6 had strongest resistance, with only a 6.59% average AP drop, and ASA was the most effective attack algorithm with a 14.51% average AP reduction, twice that of other algorithms. Static scenes had higher recognition AP, and results under different weather conditions were similar. Adversarial attack algorithm improvement may be approaching its 'limitation'.



## **18. Simultaneous Adversarial Attacks On Multiple Face Recognition System Components**

cs.CV

**SubmitDate**: 2023-04-11    [abs](http://arxiv.org/abs/2304.05048v1) [paper-pdf](http://arxiv.org/pdf/2304.05048v1)

**Authors**: Inderjeet Singh, Kazuya Kakizaki, Toshinori Araki

**Abstract**: In this work, we investigate the potential threat of adversarial examples to the security of face recognition systems. Although previous research has explored the adversarial risk to individual components of FRSs, our study presents an initial exploration of an adversary simultaneously fooling multiple components: the face detector and feature extractor in an FRS pipeline. We propose three multi-objective attacks on FRSs and demonstrate their effectiveness through a preliminary experimental analysis on a target system. Our attacks achieved up to 100% Attack Success Rates against both the face detector and feature extractor and were able to manipulate the face detection probability by up to 50% depending on the adversarial objective. This research identifies and examines novel attack vectors against FRSs and suggests possible ways to augment the robustness by leveraging the attack vector's knowledge during training of an FRS's components.



## **19. How many dimensions are required to find an adversarial example?**

cs.LG

Comments welcome! V2: minor edits for clarity

**SubmitDate**: 2023-04-11    [abs](http://arxiv.org/abs/2303.14173v2) [paper-pdf](http://arxiv.org/pdf/2303.14173v2)

**Authors**: Charles Godfrey, Henry Kvinge, Elise Bishoff, Myles Mckay, Davis Brown, Tim Doster, Eleanor Byler

**Abstract**: Past work exploring adversarial vulnerability have focused on situations where an adversary can perturb all dimensions of model input. On the other hand, a range of recent works consider the case where either (i) an adversary can perturb a limited number of input parameters or (ii) a subset of modalities in a multimodal problem. In both of these cases, adversarial examples are effectively constrained to a subspace $V$ in the ambient input space $\mathcal{X}$. Motivated by this, in this work we investigate how adversarial vulnerability depends on $\dim(V)$. In particular, we show that the adversarial success of standard PGD attacks with $\ell^p$ norm constraints behaves like a monotonically increasing function of $\epsilon (\frac{\dim(V)}{\dim \mathcal{X}})^{\frac{1}{q}}$ where $\epsilon$ is the perturbation budget and $\frac{1}{p} + \frac{1}{q} =1$, provided $p > 1$ (the case $p=1$ presents additional subtleties which we analyze in some detail). This functional form can be easily derived from a simple toy linear model, and as such our results land further credence to arguments that adversarial examples are endemic to locally linear models on high dimensional spaces.



## **20. Gradient-based Uncertainty Attribution for Explainable Bayesian Deep Learning**

cs.LG

Accepted to CVPR 2023

**SubmitDate**: 2023-04-10    [abs](http://arxiv.org/abs/2304.04824v1) [paper-pdf](http://arxiv.org/pdf/2304.04824v1)

**Authors**: Hanjing Wang, Dhiraj Joshi, Shiqiang Wang, Qiang Ji

**Abstract**: Predictions made by deep learning models are prone to data perturbations, adversarial attacks, and out-of-distribution inputs. To build a trusted AI system, it is therefore critical to accurately quantify the prediction uncertainties. While current efforts focus on improving uncertainty quantification accuracy and efficiency, there is a need to identify uncertainty sources and take actions to mitigate their effects on predictions. Therefore, we propose to develop explainable and actionable Bayesian deep learning methods to not only perform accurate uncertainty quantification but also explain the uncertainties, identify their sources, and propose strategies to mitigate the uncertainty impacts. Specifically, we introduce a gradient-based uncertainty attribution method to identify the most problematic regions of the input that contribute to the prediction uncertainty. Compared to existing methods, the proposed UA-Backprop has competitive accuracy, relaxed assumptions, and high efficiency. Moreover, we propose an uncertainty mitigation strategy that leverages the attribution results as attention to further improve the model performance. Both qualitative and quantitative evaluations are conducted to demonstrate the effectiveness of our proposed methods.



## **21. Language-Driven Anchors for Zero-Shot Adversarial Robustness**

cs.CV

11 pages

**SubmitDate**: 2023-04-10    [abs](http://arxiv.org/abs/2301.13096v2) [paper-pdf](http://arxiv.org/pdf/2301.13096v2)

**Authors**: Xiao Li, Wei Zhang, Yining Liu, Zhanhao Hu, Bo Zhang, Xiaolin Hu

**Abstract**: Deep neural networks are known to be susceptible to adversarial attacks. In this work, we focus on improving adversarial robustness in the challenging zero-shot image classification setting. To address this issue, we propose LAAT, a novel Language-driven, Anchor-based Adversarial Training strategy. LAAT utilizes a text encoder to generate fixed anchors (normalized feature embeddings) for each category and then uses these anchors for adversarial training. By leveraging the semantic consistency of the text encoders, LAAT can enhance the adversarial robustness of the image model on novel categories without additional examples. We identify the large cosine similarity problem of recent text encoders and design several effective techniques to address it. The experimental results demonstrate that LAAT significantly improves zero-shot adversarial performance, outperforming previous state-of-the-art adversarially robust one-shot methods. Moreover, our method produces substantial zero-shot adversarial robustness when models are trained on large datasets such as ImageNet-1K and applied to several downstream datasets.



## **22. Reinforcement Learning-Based Black-Box Model Inversion Attacks**

cs.LG

CVPR 2023, Accepted

**SubmitDate**: 2023-04-10    [abs](http://arxiv.org/abs/2304.04625v1) [paper-pdf](http://arxiv.org/pdf/2304.04625v1)

**Authors**: Gyojin Han, Jaehyun Choi, Haeil Lee, Junmo Kim

**Abstract**: Model inversion attacks are a type of privacy attack that reconstructs private data used to train a machine learning model, solely by accessing the model. Recently, white-box model inversion attacks leveraging Generative Adversarial Networks (GANs) to distill knowledge from public datasets have been receiving great attention because of their excellent attack performance. On the other hand, current black-box model inversion attacks that utilize GANs suffer from issues such as being unable to guarantee the completion of the attack process within a predetermined number of query accesses or achieve the same level of performance as white-box attacks. To overcome these limitations, we propose a reinforcement learning-based black-box model inversion attack. We formulate the latent space search as a Markov Decision Process (MDP) problem and solve it with reinforcement learning. Our method utilizes the confidence scores of the generated images to provide rewards to an agent. Finally, the private data can be reconstructed using the latent vectors found by the agent trained in the MDP. The experiment results on various datasets and models demonstrate that our attack successfully recovers the private information of the target model by achieving state-of-the-art attack performance. We emphasize the importance of studies on privacy-preserving machine learning by proposing a more advanced black-box model inversion attack.



## **23. Defense-Prefix for Preventing Typographic Attacks on CLIP**

cs.CV

Under review

**SubmitDate**: 2023-04-10    [abs](http://arxiv.org/abs/2304.04512v1) [paper-pdf](http://arxiv.org/pdf/2304.04512v1)

**Authors**: Hiroki Azuma, Yusuke Matsui

**Abstract**: Vision-language pre-training models (VLPs) have exhibited revolutionary improvements in various vision-language tasks. In VLP, some adversarial attacks fool a model into false or absurd classifications. Previous studies addressed these attacks by fine-tuning the model or changing its architecture. However, these methods risk losing the original model's performance and are difficult to apply to downstream tasks. In particular, their applicability to other tasks has not been considered. In this study, we addressed the reduction of the impact of typographic attacks on CLIP without changing the model parameters. To achieve this, we expand the idea of ``prefix learning'' and introduce our simple yet effective method: Defense-Prefix (DP), which inserts the DP token before a class name to make words ``robust'' against typographic attacks. Our method can be easily applied to downstream tasks, such as object detection, because the proposed method is independent of the model parameters. Our method significantly improves the accuracy of classification tasks for typographic attack datasets, while maintaining the zero-shot capabilities of the model. In addition, we leverage our proposed method for object detection, demonstrating its high applicability and effectiveness. The codes and datasets will be publicly available.



## **24. Robust Neural Architecture Search**

cs.LG

**SubmitDate**: 2023-04-10    [abs](http://arxiv.org/abs/2304.02845v2) [paper-pdf](http://arxiv.org/pdf/2304.02845v2)

**Authors**: Xunyu Zhu, Jian Li, Yong Liu, Weiping Wang

**Abstract**: Neural Architectures Search (NAS) becomes more and more popular over these years. However, NAS-generated models tends to suffer greater vulnerability to various malicious attacks. Lots of robust NAS methods leverage adversarial training to enhance the robustness of NAS-generated models, however, they neglected the nature accuracy of NAS-generated models. In our paper, we propose a novel NAS method, Robust Neural Architecture Search (RNAS). To design a regularization term to balance accuracy and robustness, RNAS generates architectures with both high accuracy and good robustness. To reduce search cost, we further propose to use noise examples instead adversarial examples as input to search architectures. Extensive experiments show that RNAS achieves state-of-the-art (SOTA) performance on both image classification and adversarial attacks, which illustrates the proposed RNAS achieves a good tradeoff between robustness and accuracy.



## **25. Generating Adversarial Attacks in the Latent Space**

cs.LG

**SubmitDate**: 2023-04-10    [abs](http://arxiv.org/abs/2304.04386v1) [paper-pdf](http://arxiv.org/pdf/2304.04386v1)

**Authors**: Nitish Shukla, Sudipta Banerjee

**Abstract**: Adversarial attacks in the input (pixel) space typically incorporate noise margins such as $L_1$ or $L_{\infty}$-norm to produce imperceptibly perturbed data that confound deep learning networks. Such noise margins confine the magnitude of permissible noise. In this work, we propose injecting adversarial perturbations in the latent (feature) space using a generative adversarial network, removing the need for margin-based priors. Experiments on MNIST, CIFAR10, Fashion-MNIST, CIFAR100 and Stanford Dogs datasets support the effectiveness of the proposed method in generating adversarial attacks in the latent space while ensuring a high degree of visual realism with respect to pixel-based adversarial attack methods.



## **26. Certifiable Black-Box Attack: Ensuring Provably Successful Attack for Adversarial Examples**

cs.LG

**SubmitDate**: 2023-04-10    [abs](http://arxiv.org/abs/2304.04343v1) [paper-pdf](http://arxiv.org/pdf/2304.04343v1)

**Authors**: Hanbin Hong, Yuan Hong

**Abstract**: Black-box adversarial attacks have shown strong potential to subvert machine learning models. Existing black-box adversarial attacks craft the adversarial examples by iteratively querying the target model and/or leveraging the transferability of a local surrogate model. Whether such attack can succeed remains unknown to the adversary when empirically designing the attack. In this paper, to our best knowledge, we take the first step to study a new paradigm of adversarial attacks -- certifiable black-box attack that can guarantee the attack success rate of the crafted adversarial examples. Specifically, we revise the randomized smoothing to establish novel theories for ensuring the attack success rate of the adversarial examples. To craft the adversarial examples with the certifiable attack success rate (CASR) guarantee, we design several novel techniques, including a randomized query method to query the target model, an initialization method with smoothed self-supervised perturbation to derive certifiable adversarial examples, and a geometric shifting method to reduce the perturbation size of the certifiable adversarial examples for better imperceptibility. We have comprehensively evaluated the performance of the certifiable black-box attack on CIFAR10 and ImageNet datasets against different levels of defenses. Both theoretical and experimental results have validated the effectiveness of the proposed certifiable attack.



## **27. Unsupervised Multi-Criteria Adversarial Detection in Deep Image Retrieval**

cs.CV

**SubmitDate**: 2023-04-09    [abs](http://arxiv.org/abs/2304.04228v1) [paper-pdf](http://arxiv.org/pdf/2304.04228v1)

**Authors**: Yanru Xiao, Cong Wang, Xing Gao

**Abstract**: The vulnerability in the algorithm supply chain of deep learning has imposed new challenges to image retrieval systems in the downstream. Among a variety of techniques, deep hashing is gaining popularity. As it inherits the algorithmic backend from deep learning, a handful of attacks are recently proposed to disrupt normal image retrieval. Unfortunately, the defense strategies in softmax classification are not readily available to be applied in the image retrieval domain. In this paper, we propose an efficient and unsupervised scheme to identify unique adversarial behaviors in the hamming space. In particular, we design three criteria from the perspectives of hamming distance, quantization loss and denoising to defend against both untargeted and targeted attacks, which collectively limit the adversarial space. The extensive experiments on four datasets demonstrate 2-23% improvements of detection rates with minimum computational overhead for real-time image queries.



## **28. Adversarially Robust Neural Architecture Search for Graph Neural Networks**

cs.LG

Accepted as a conference paper at CVPR 2023

**SubmitDate**: 2023-04-09    [abs](http://arxiv.org/abs/2304.04168v1) [paper-pdf](http://arxiv.org/pdf/2304.04168v1)

**Authors**: Beini Xie, Heng Chang, Ziwei Zhang, Xin Wang, Daixin Wang, Zhiqiang Zhang, Rex Ying, Wenwu Zhu

**Abstract**: Graph Neural Networks (GNNs) obtain tremendous success in modeling relational data. Still, they are prone to adversarial attacks, which are massive threats to applying GNNs to risk-sensitive domains. Existing defensive methods neither guarantee performance facing new data/tasks or adversarial attacks nor provide insights to understand GNN robustness from an architectural perspective. Neural Architecture Search (NAS) has the potential to solve this problem by automating GNN architecture designs. Nevertheless, current graph NAS approaches lack robust design and are vulnerable to adversarial attacks. To tackle these challenges, we propose a novel Robust Neural Architecture search framework for GNNs (G-RNA). Specifically, we design a robust search space for the message-passing mechanism by adding graph structure mask operations into the search space, which comprises various defensive operation candidates and allows us to search for defensive GNNs. Furthermore, we define a robustness metric to guide the search procedure, which helps to filter robust architectures. In this way, G-RNA helps understand GNN robustness from an architectural perspective and effectively searches for optimal adversarial robust GNNs. Extensive experimental results on benchmark datasets show that G-RNA significantly outperforms manually designed robust GNNs and vanilla graph NAS baselines by 12.1% to 23.4% under adversarial attacks.



## **29. Exploring the Connection between Robust and Generative Models**

cs.LG

technical report, 6 pages, 6 figures

**SubmitDate**: 2023-04-08    [abs](http://arxiv.org/abs/2304.04033v1) [paper-pdf](http://arxiv.org/pdf/2304.04033v1)

**Authors**: Senad Beadini, Iacopo Masi

**Abstract**: We offer a study that connects robust discriminative classifiers trained with adversarial training (AT) with generative modeling in the form of Energy-based Models (EBM). We do so by decomposing the loss of a discriminative classifier and showing that the discriminative model is also aware of the input data density. Though a common assumption is that adversarial points leave the manifold of the input data, our study finds out that, surprisingly, untargeted adversarial points in the input space are very likely under the generative model hidden inside the discriminative classifier -- have low energy in the EBM. We present two evidence: untargeted attacks are even more likely than the natural data and their likelihood increases as the attack strength increases. This allows us to easily detect them and craft a novel attack called High-Energy PGD that fools the classifier yet has energy similar to the data set.



## **30. On anti-stochastic properties of unlabeled graphs**

cs.DM

**SubmitDate**: 2023-04-08    [abs](http://arxiv.org/abs/2112.04395v4) [paper-pdf](http://arxiv.org/pdf/2112.04395v4)

**Authors**: Sergei Kiselev, Andrey Kupavskii, Oleg Verbitsky, Maksim Zhukovskii

**Abstract**: We study vulnerability of a uniformly distributed random graph to an attack by an adversary who aims for a global change of the distribution while being able to make only a local change in the graph. We call a graph property $A$ anti-stochastic if the probability that a random graph $G$ satisfies $A$ is small but, with high probability, there is a small perturbation transforming $G$ into a graph satisfying $A$. While for labeled graphs such properties are easy to obtain from binary covering codes, the existence of anti-stochastic properties for unlabeled graphs is not so evident. If an admissible perturbation is either the addition or the deletion of one edge, we exhibit an anti-stochastic property that is satisfied by a random unlabeled graph of order $n$ with probability $(2+o(1))/n^2$, which is as small as possible. We also express another anti-stochastic property in terms of the degree sequence of a graph. This property has probability $(2+o(1))/(n\ln n)$, which is optimal up to factor of 2.



## **31. RobCaps: Evaluating the Robustness of Capsule Networks against Affine Transformations and Adversarial Attacks**

cs.LG

To appear at the 2023 International Joint Conference on Neural  Networks (IJCNN), Queensland, Australia, June 2023

**SubmitDate**: 2023-04-08    [abs](http://arxiv.org/abs/2304.03973v1) [paper-pdf](http://arxiv.org/pdf/2304.03973v1)

**Authors**: Alberto Marchisio, Antonio De Marco, Alessio Colucci, Maurizio Martina, Muhammad Shafique

**Abstract**: Capsule Networks (CapsNets) are able to hierarchically preserve the pose relationships between multiple objects for image classification tasks. Other than achieving high accuracy, another relevant factor in deploying CapsNets in safety-critical applications is the robustness against input transformations and malicious adversarial attacks.   In this paper, we systematically analyze and evaluate different factors affecting the robustness of CapsNets, compared to traditional Convolutional Neural Networks (CNNs). Towards a comprehensive comparison, we test two CapsNet models and two CNN models on the MNIST, GTSRB, and CIFAR10 datasets, as well as on the affine-transformed versions of such datasets. With a thorough analysis, we show which properties of these architectures better contribute to increasing the robustness and their limitations. Overall, CapsNets achieve better robustness against adversarial examples and affine transformations, compared to a traditional CNN with a similar number of parameters. Similar conclusions have been derived for deeper versions of CapsNets and CNNs. Moreover, our results unleash a key finding that the dynamic routing does not contribute much to improving the CapsNets' robustness. Indeed, the main generalization contribution is due to the hierarchical feature learning through capsules.



## **32. TSFool: Crafting Highly-imperceptible Adversarial Time Series through Multi-objective Black-box Attack to Fool RNN Classifiers**

cs.LG

9 pages, 7 figures

**SubmitDate**: 2023-04-08    [abs](http://arxiv.org/abs/2209.06388v2) [paper-pdf](http://arxiv.org/pdf/2209.06388v2)

**Authors**: Yanyun Wang, Dehui Du, Yuanhao Liu

**Abstract**: Neural network (NN) classifiers are vulnerable to adversarial attacks. Although the existing gradient-based attacks achieve state-of-the-art performance in feed-forward NNs and image recognition tasks, they do not perform as well on time series classification with recurrent neural network (RNN) models. This is because the cyclical structure of RNN prevents direct model differentiation and the visual sensitivity of time series data to perturbations challenges the traditional local optimization objective of the adversarial attack. In this paper, a black-box method called TSFool is proposed to efficiently craft highly-imperceptible adversarial time series for RNN classifiers. We propose a novel global optimization objective named Camouflage Coefficient to consider the imperceptibility of adversarial samples from the perspective of class distribution, and accordingly refine the adversarial attack as a multi-objective optimization problem to enhance the perturbation quality. To get rid of the dependence on gradient information, we also propose a new idea that introduces a representation model for RNN to capture deeply embedded vulnerable samples having otherness between their features and latent manifold, based on which the optimization solution can be heuristically approximated. Experiments on 10 UCR datasets are conducted to confirm that TSFool averagely outperforms existing methods with a 46.3% higher attack success rate, 87.4% smaller perturbation and 25.6% better Camouflage Coefficient at a similar time cost.



## **33. Benchmarking the Robustness of Quantized Models**

cs.LG

Workshop at IEEE Conference on Computer Vision and Pattern  Recognition 2023

**SubmitDate**: 2023-04-08    [abs](http://arxiv.org/abs/2304.03968v1) [paper-pdf](http://arxiv.org/pdf/2304.03968v1)

**Authors**: Yisong Xiao, Tianyuan Zhang, Shunchang Liu, Haotong Qin

**Abstract**: Quantization has emerged as an essential technique for deploying deep neural networks (DNNs) on devices with limited resources. However, quantized models exhibit vulnerabilities when exposed to various noises in real-world applications. Despite the importance of evaluating the impact of quantization on robustness, existing research on this topic is limited and often disregards established principles of robustness evaluation, resulting in incomplete and inconclusive findings. To address this gap, we thoroughly evaluated the robustness of quantized models against various noises (adversarial attacks, natural corruptions, and systematic noises) on ImageNet. Extensive experiments demonstrate that lower-bit quantization is more resilient to adversarial attacks but is more susceptible to natural corruptions and systematic noises. Notably, our investigation reveals that impulse noise (in natural corruptions) and the nearest neighbor interpolation (in systematic noises) have the most significant impact on quantized models. Our research contributes to advancing the robust quantization of models and their deployment in real-world scenarios.



## **34. Robust Deep Learning Models Against Semantic-Preserving Adversarial Attack**

cs.LG

Paper accepted by the 2023 International Joint Conference on Neural  Networks (IJCNN 2023)

**SubmitDate**: 2023-04-08    [abs](http://arxiv.org/abs/2304.03955v1) [paper-pdf](http://arxiv.org/pdf/2304.03955v1)

**Authors**: Dashan Gao, Yunce Zhao, Yinghua Yao, Zeqi Zhang, Bifei Mao, Xin Yao

**Abstract**: Deep learning models can be fooled by small $l_p$-norm adversarial perturbations and natural perturbations in terms of attributes. Although the robustness against each perturbation has been explored, it remains a challenge to address the robustness against joint perturbations effectively. In this paper, we study the robustness of deep learning models against joint perturbations by proposing a novel attack mechanism named Semantic-Preserving Adversarial (SPA) attack, which can then be used to enhance adversarial training. Specifically, we introduce an attribute manipulator to generate natural and human-comprehensible perturbations and a noise generator to generate diverse adversarial noises. Based on such combined noises, we optimize both the attribute value and the diversity variable to generate jointly-perturbed samples. For robust training, we adversarially train the deep learning model against the generated joint perturbations. Empirical results on four benchmarks show that the SPA attack causes a larger performance decline with small $l_{\infty}$ norm-ball constraints compared to existing approaches. Furthermore, our SPA-enhanced training outperforms existing defense methods against such joint perturbations.



## **35. Discrete Point-wise Attack Is Not Enough: Generalized Manifold Adversarial Attack for Face Recognition**

cs.CV

Accepted by CVPR2023

**SubmitDate**: 2023-04-08    [abs](http://arxiv.org/abs/2301.06083v2) [paper-pdf](http://arxiv.org/pdf/2301.06083v2)

**Authors**: Qian Li, Yuxiao Hu, Ye Liu, Dongxiao Zhang, Xin Jin, Yuntian Chen

**Abstract**: Classical adversarial attacks for Face Recognition (FR) models typically generate discrete examples for target identity with a single state image. However, such paradigm of point-wise attack exhibits poor generalization against numerous unknown states of identity and can be easily defended. In this paper, by rethinking the inherent relationship between the face of target identity and its variants, we introduce a new pipeline of Generalized Manifold Adversarial Attack (GMAA) to achieve a better attack performance by expanding the attack range. Specifically, this expansion lies on two aspects - GMAA not only expands the target to be attacked from one to many to encourage a good generalization ability for the generated adversarial examples, but it also expands the latter from discrete points to manifold by leveraging the domain knowledge that face expression change can be continuous, which enhances the attack effect as a data augmentation mechanism did. Moreover, we further design a dual supervision with local and global constraints as a minor contribution to improve the visual quality of the generated adversarial examples. We demonstrate the effectiveness of our method based on extensive experiments, and reveal that GMAA promises a semantic continuous adversarial space with a higher generalization ability and visual quality



## **36. SoK: Decentralized Finance (DeFi) Attacks**

cs.CR

**SubmitDate**: 2023-04-07    [abs](http://arxiv.org/abs/2208.13035v3) [paper-pdf](http://arxiv.org/pdf/2208.13035v3)

**Authors**: Liyi Zhou, Xihan Xiong, Jens Ernstberger, Stefanos Chaliasos, Zhipeng Wang, Ye Wang, Kaihua Qin, Roger Wattenhofer, Dawn Song, Arthur Gervais

**Abstract**: Within just four years, the blockchain-based Decentralized Finance (DeFi) ecosystem has accumulated a peak total value locked (TVL) of more than 253 billion USD. This surge in DeFi's popularity has, unfortunately, been accompanied by many impactful incidents. According to our data, users, liquidity providers, speculators, and protocol operators suffered a total loss of at least 3.24 billion USD from Apr 30, 2018 to Apr 30, 2022. Given the blockchain's transparency and increasing incident frequency, two questions arise: How can we systematically measure, evaluate, and compare DeFi incidents? How can we learn from past attacks to strengthen DeFi security?   In this paper, we introduce a common reference frame to systematically evaluate and compare DeFi incidents, including both attacks and accidents. We investigate 77 academic papers, 30 audit reports, and 181 real-world incidents. Our data reveals several gaps between academia and the practitioners' community. For example, few academic papers address "price oracle attacks" and "permissonless interactions", while our data suggests that they are the two most frequent incident types (15% and 10.5% correspondingly). We also investigate potential defenses, and find that: (i) 103 (56%) of the attacks are not executed atomically, granting a rescue time frame for defenders; (ii) SoTA bytecode similarity analysis can at least detect 31 vulnerable/23 adversarial contracts; and (iii) 33 (15.3%) of the adversaries leak potentially identifiable information by interacting with centralized exchanges.



## **37. AMS-DRL: Learning Multi-Pursuit Evasion for Safe Targeted Navigation of Drones**

cs.RO

**SubmitDate**: 2023-04-07    [abs](http://arxiv.org/abs/2304.03443v1) [paper-pdf](http://arxiv.org/pdf/2304.03443v1)

**Authors**: Jiaping Xiao, Mir Feroskhan

**Abstract**: Safe navigation of drones in the presence of adversarial physical attacks from multiple pursuers is a challenging task. This paper proposes a novel approach, asynchronous multi-stage deep reinforcement learning (AMS-DRL), to train an adversarial neural network that can learn from the actions of multiple pursuers and adapt quickly to their behavior, enabling the drone to avoid attacks and reach its target. Our approach guarantees convergence by ensuring Nash Equilibrium among agents from the game-theory analysis. We evaluate our method in extensive simulations and show that it outperforms baselines with higher navigation success rates. We also analyze how parameters such as the relative maximum speed affect navigation performance. Furthermore, we have conducted physical experiments and validated the effectiveness of the trained policies in real-time flights. A success rate heatmap is introduced to elucidate how spatial geometry influences navigation outcomes. Project website: https://github.com/NTU-UAVG/AMS-DRL-for-Pursuit-Evasion.



## **38. LP-BFGS attack: An adversarial attack based on the Hessian with limited pixels**

cs.CR

15 pages, 7 figures

**SubmitDate**: 2023-04-07    [abs](http://arxiv.org/abs/2210.15446v2) [paper-pdf](http://arxiv.org/pdf/2210.15446v2)

**Authors**: Jiebao Zhang, Wenhua Qian, Rencan Nie, Jinde Cao, Dan Xu

**Abstract**: Deep neural networks are vulnerable to adversarial attacks. Most $L_{0}$-norm based white-box attacks craft perturbations by the gradient of models to the input. Since the computation cost and memory limitation of calculating the Hessian matrix, the application of Hessian or approximate Hessian in white-box attacks is gradually shelved. In this work, we note that the sparsity requirement on perturbations naturally lends itself to the usage of Hessian information. We study the attack performance and computation cost of the attack method based on the Hessian with a limited number of perturbation pixels. Specifically, we propose the Limited Pixel BFGS (LP-BFGS) attack method by incorporating the perturbation pixel selection strategy and the BFGS algorithm. Pixels with top-k attribution scores calculated by the Integrated Gradient method are regarded as optimization variables of the LP-BFGS attack. Experimental results across different networks and datasets demonstrate that our approach has comparable attack ability with reasonable computation in different numbers of perturbation pixels compared with existing solutions.



## **39. EZClone: Improving DNN Model Extraction Attack via Shape Distillation from GPU Execution Profiles**

cs.LG

11 pages, 6 tables, 4 figures

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2304.03388v1) [paper-pdf](http://arxiv.org/pdf/2304.03388v1)

**Authors**: Jonah O'Brien Weiss, Tiago Alves, Sandip Kundu

**Abstract**: Deep Neural Networks (DNNs) have become ubiquitous due to their performance on prediction and classification problems. However, they face a variety of threats as their usage spreads. Model extraction attacks, which steal DNNs, endanger intellectual property, data privacy, and security. Previous research has shown that system-level side-channels can be used to leak the architecture of a victim DNN, exacerbating these risks. We propose two DNN architecture extraction techniques catering to various threat models. The first technique uses a malicious, dynamically linked version of PyTorch to expose a victim DNN architecture through the PyTorch profiler. The second, called EZClone, exploits aggregate (rather than time-series) GPU profiles as a side-channel to predict DNN architecture, employing a simple approach and assuming little adversary capability as compared to previous work. We investigate the effectiveness of EZClone when minimizing the complexity of the attack, when applied to pruned models, and when applied across GPUs. We find that EZClone correctly predicts DNN architectures for the entire set of PyTorch vision architectures with 100% accuracy. No other work has shown this degree of architecture prediction accuracy with the same adversarial constraints or using aggregate side-channel information. Prior work has shown that, once a DNN has been successfully cloned, further attacks such as model evasion or model inversion can be accelerated significantly.



## **40. Reliable Learning for Test-time Attacks and Distribution Shift**

cs.LG

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2304.03370v1) [paper-pdf](http://arxiv.org/pdf/2304.03370v1)

**Authors**: Maria-Florina Balcan, Steve Hanneke, Rattana Pukdee, Dravyansh Sharma

**Abstract**: Machine learning algorithms are often used in environments which are not captured accurately even by the most carefully obtained training data, either due to the possibility of `adversarial' test-time attacks, or on account of `natural' distribution shift. For test-time attacks, we introduce and analyze a novel robust reliability guarantee, which requires a learner to output predictions along with a reliability radius $\eta$, with the meaning that its prediction is guaranteed to be correct as long as the adversary has not perturbed the test point farther than a distance $\eta$. We provide learners that are optimal in the sense that they always output the best possible reliability radius on any test point, and we characterize the reliable region, i.e. the set of points where a given reliability radius is attainable. We additionally analyze reliable learners under distribution shift, where the test points may come from an arbitrary distribution Q different from the training distribution P. For both cases, we bound the probability mass of the reliable region for several interesting examples, for linear separators under nearly log-concave and s-concave distributions, as well as for smooth boundary classifiers under smooth probability distributions.



## **41. Improving Visual Question Answering Models through Robustness Analysis and In-Context Learning with a Chain of Basic Questions**

cs.CV

28 pages

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2304.03147v1) [paper-pdf](http://arxiv.org/pdf/2304.03147v1)

**Authors**: Jia-Hong Huang, Modar Alfadly, Bernard Ghanem, Marcel Worring

**Abstract**: Deep neural networks have been critical in the task of Visual Question Answering (VQA), with research traditionally focused on improving model accuracy. Recently, however, there has been a trend towards evaluating the robustness of these models against adversarial attacks. This involves assessing the accuracy of VQA models under increasing levels of noise in the input, which can target either the image or the proposed query question, dubbed the main question. However, there is currently a lack of proper analysis of this aspect of VQA. This work proposes a new method that utilizes semantically related questions, referred to as basic questions, acting as noise to evaluate the robustness of VQA models. It is hypothesized that as the similarity of a basic question to the main question decreases, the level of noise increases. To generate a reasonable noise level for a given main question, a pool of basic questions is ranked based on their similarity to the main question, and this ranking problem is cast as a LASSO optimization problem. Additionally, this work proposes a novel robustness measure, R_score, and two basic question datasets to standardize the analysis of VQA model robustness. The experimental results demonstrate that the proposed evaluation method effectively analyzes the robustness of VQA models. Moreover, the experiments show that in-context learning with a chain of basic questions can enhance model accuracy.



## **42. Public Key Encryption with Secure Key Leasing**

quant-ph

68 pages, 4 figures. added related works and a comparison with a  concurrent work (2023-04-07)

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2302.11663v2) [paper-pdf](http://arxiv.org/pdf/2302.11663v2)

**Authors**: Shweta Agrawal, Fuyuki Kitagawa, Ryo Nishimaki, Shota Yamada, Takashi Yamakawa

**Abstract**: We introduce the notion of public key encryption with secure key leasing (PKE-SKL). Our notion supports the leasing of decryption keys so that a leased key achieves the decryption functionality but comes with the guarantee that if the quantum decryption key returned by a user passes a validity test, then the user has lost the ability to decrypt. Our notion is similar in spirit to the notion of secure software leasing (SSL) introduced by Ananth and La Placa (Eurocrypt 2021) but captures significantly more general adversarial strategies. In more detail, our adversary is not restricted to use an honest evaluation algorithm to run pirated software. Our results can be summarized as follows:   1. Definitions: We introduce the definition of PKE with secure key leasing and formalize security notions.   2. Constructing PKE with Secure Key Leasing: We provide a construction of PKE-SKL by leveraging a PKE scheme that satisfies a new security notion that we call consistent or inconsistent security against key leasing attacks (CoIC-KLA security). We then construct a CoIC-KLA secure PKE scheme using 1-key Ciphertext-Policy Functional Encryption (CPFE) that in turn can be based on any IND-CPA secure PKE scheme.   3. Identity Based Encryption, Attribute Based Encryption and Functional Encryption with Secure Key Leasing: We provide definitions of secure key leasing in the context of advanced encryption schemes such as identity based encryption (IBE), attribute-based encryption (ABE) and functional encryption (FE). Then we provide constructions by combining the above PKE-SKL with standard IBE, ABE and FE schemes.



## **43. StratDef: Strategic Defense Against Adversarial Attacks in ML-based Malware Detection**

cs.LG

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2202.07568v5) [paper-pdf](http://arxiv.org/pdf/2202.07568v5)

**Authors**: Aqib Rashid, Jose Such

**Abstract**: Over the years, most research towards defenses against adversarial attacks on machine learning models has been in the image recognition domain. The malware detection domain has received less attention despite its importance. Moreover, most work exploring these defenses has focused on several methods but with no strategy when applying them. In this paper, we introduce StratDef, which is a strategic defense system based on a moving target defense approach. We overcome challenges related to the systematic construction, selection, and strategic use of models to maximize adversarial robustness. StratDef dynamically and strategically chooses the best models to increase the uncertainty for the attacker while minimizing critical aspects in the adversarial ML domain, like attack transferability. We provide the first comprehensive evaluation of defenses against adversarial attacks on machine learning for malware detection, where our threat model explores different levels of threat, attacker knowledge, capabilities, and attack intensities. We show that StratDef performs better than other defenses even when facing the peak adversarial threat. We also show that, of the existing defenses, only a few adversarially-trained models provide substantially better protection than just using vanilla models but are still outperformed by StratDef.



## **44. PAD: Towards Principled Adversarial Malware Detection Against Evasion Attacks**

cs.CR

Accepted by IEEE Transactions on Dependable and Secure Computing; To  appear

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2302.11328v2) [paper-pdf](http://arxiv.org/pdf/2302.11328v2)

**Authors**: Deqiang Li, Shicheng Cui, Yun Li, Jia Xu, Fu Xiao, Shouhuai Xu

**Abstract**: Machine Learning (ML) techniques can facilitate the automation of malicious software (malware for short) detection, but suffer from evasion attacks. Many studies counter such attacks in heuristic manners, lacking theoretical guarantees and defense effectiveness. In this paper, we propose a new adversarial training framework, termed Principled Adversarial Malware Detection (PAD), which offers convergence guarantees for robust optimization methods. PAD lays on a learnable convex measurement that quantifies distribution-wise discrete perturbations to protect malware detectors from adversaries, whereby for smooth detectors, adversarial training can be performed with theoretical treatments. To promote defense effectiveness, we propose a new mixture of attacks to instantiate PAD to enhance deep neural network-based measurements and malware detectors. Experimental results on two Android malware datasets demonstrate: (i) the proposed method significantly outperforms the state-of-the-art defenses; (ii) it can harden ML-based malware detection against 27 evasion attacks with detection accuracies greater than 83.45%, at the price of suffering an accuracy decrease smaller than 2.16% in the absence of attacks; (iii) it matches or outperforms many anti-malware scanners in VirusTotal against realistic adversarial malware.



## **45. Robust Upper Bounds for Adversarial Training**

cs.LG

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2112.09279v2) [paper-pdf](http://arxiv.org/pdf/2112.09279v2)

**Authors**: Dimitris Bertsimas, Xavier Boix, Kimberly Villalobos Carballo, Dick den Hertog

**Abstract**: Many state-of-the-art adversarial training methods for deep learning leverage upper bounds of the adversarial loss to provide security guarantees against adversarial attacks. Yet, these methods rely on convex relaxations to propagate lower and upper bounds for intermediate layers, which affect the tightness of the bound at the output layer. We introduce a new approach to adversarial training by minimizing an upper bound of the adversarial loss that is based on a holistic expansion of the network instead of separate bounds for each layer. This bound is facilitated by state-of-the-art tools from Robust Optimization; it has closed-form and can be effectively trained using backpropagation. We derive two new methods with the proposed approach. The first method (Approximated Robust Upper Bound or aRUB) uses the first order approximation of the network as well as basic tools from Linear Robust Optimization to obtain an empirical upper bound of the adversarial loss that can be easily implemented. The second method (Robust Upper Bound or RUB), computes a provable upper bound of the adversarial loss. Across a variety of tabular and vision data sets we demonstrate the effectiveness of our approach -- RUB is substantially more robust than state-of-the-art methods for larger perturbations, while aRUB matches the performance of state-of-the-art methods for small perturbations.



## **46. Improving Fast Adversarial Training with Prior-Guided Knowledge**

cs.LG

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2304.00202v2) [paper-pdf](http://arxiv.org/pdf/2304.00202v2)

**Authors**: Xiaojun Jia, Yong Zhang, Xingxing Wei, Baoyuan Wu, Ke Ma, Jue Wang, Xiaochun Cao

**Abstract**: Fast adversarial training (FAT) is an efficient method to improve robustness. However, the original FAT suffers from catastrophic overfitting, which dramatically and suddenly reduces robustness after a few training epochs. Although various FAT variants have been proposed to prevent overfitting, they require high training costs. In this paper, we investigate the relationship between adversarial example quality and catastrophic overfitting by comparing the training processes of standard adversarial training and FAT. We find that catastrophic overfitting occurs when the attack success rate of adversarial examples becomes worse. Based on this observation, we propose a positive prior-guided adversarial initialization to prevent overfitting by improving adversarial example quality without extra training costs. This initialization is generated by using high-quality adversarial perturbations from the historical training process. We provide theoretical analysis for the proposed initialization and propose a prior-guided regularization method that boosts the smoothness of the loss function. Additionally, we design a prior-guided ensemble FAT method that averages the different model weights of historical models using different decay rates. Our proposed method, called FGSM-PGK, assembles the prior-guided knowledge, i.e., the prior-guided initialization and model weights, acquired during the historical training process. Evaluations of four datasets demonstrate the superiority of the proposed method.



## **47. UNICORN: A Unified Backdoor Trigger Inversion Framework**

cs.LG

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2304.02786v1) [paper-pdf](http://arxiv.org/pdf/2304.02786v1)

**Authors**: Zhenting Wang, Kai Mei, Juan Zhai, Shiqing Ma

**Abstract**: The backdoor attack, where the adversary uses inputs stamped with triggers (e.g., a patch) to activate pre-planted malicious behaviors, is a severe threat to Deep Neural Network (DNN) models. Trigger inversion is an effective way of identifying backdoor models and understanding embedded adversarial behaviors. A challenge of trigger inversion is that there are many ways of constructing the trigger. Existing methods cannot generalize to various types of triggers by making certain assumptions or attack-specific constraints. The fundamental reason is that existing work does not consider the trigger's design space in their formulation of the inversion problem. This work formally defines and analyzes the triggers injected in different spaces and the inversion problem. Then, it proposes a unified framework to invert backdoor triggers based on the formalization of triggers and the identified inner behaviors of backdoor models from our analysis. Our prototype UNICORN is general and effective in inverting backdoor triggers in DNNs. The code can be found at https://github.com/RU-System-Software-and-Security/UNICORN.



## **48. Planning for Attacker Entrapment in Adversarial Settings**

cs.AI

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2303.00822v2) [paper-pdf](http://arxiv.org/pdf/2303.00822v2)

**Authors**: Brittany Cates, Anagha Kulkarni, Sarath Sreedharan

**Abstract**: In this paper, we propose a planning framework to generate a defense strategy against an attacker who is working in an environment where a defender can operate without the attacker's knowledge. The objective of the defender is to covertly guide the attacker to a trap state from which the attacker cannot achieve their goal. Further, the defender is constrained to achieve its goal within K number of steps, where K is calculated as a pessimistic lower bound within which the attacker is unlikely to suspect a threat in the environment. Such a defense strategy is highly useful in real world systems like honeypots or honeynets, where an unsuspecting attacker interacts with a simulated production system while assuming it is the actual production system. Typically, the interaction between an attacker and a defender is captured using game theoretic frameworks. Our problem formulation allows us to capture it as a much simpler infinite horizon discounted MDP, in which the optimal policy for the MDP gives the defender's strategy against the actions of the attacker. Through empirical evaluation, we show the merits of our problem formulation.



## **49. Domain Generalization with Adversarial Intensity Attack for Medical Image Segmentation**

eess.IV

Code is available upon publication

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2304.02720v1) [paper-pdf](http://arxiv.org/pdf/2304.02720v1)

**Authors**: Zheyuan Zhang, Bin Wang, Lanhong Yao, Ugur Demir, Debesh Jha, Ismail Baris Turkbey, Boqing Gong, Ulas Bagci

**Abstract**: Most statistical learning algorithms rely on an over-simplified assumption, that is, the train and test data are independent and identically distributed. In real-world scenarios, however, it is common for models to encounter data from new and different domains to which they were not exposed to during training. This is often the case in medical imaging applications due to differences in acquisition devices, imaging protocols, and patient characteristics. To address this problem, domain generalization (DG) is a promising direction as it enables models to handle data from previously unseen domains by learning domain-invariant features robust to variations across different domains. To this end, we introduce a novel DG method called Adversarial Intensity Attack (AdverIN), which leverages adversarial training to generate training data with an infinite number of styles and increase data diversity while preserving essential content information. We conduct extensive evaluation experiments on various multi-domain segmentation datasets, including 2D retinal fundus optic disc/cup and 3D prostate MRI. Our results demonstrate that AdverIN significantly improves the generalization ability of the segmentation models, achieving significant improvement on these challenging datasets. Code is available upon publication.



## **50. A Certified Radius-Guided Attack Framework to Image Segmentation Models**

cs.CV

Accepted by EuroSP 2023

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2304.02693v1) [paper-pdf](http://arxiv.org/pdf/2304.02693v1)

**Authors**: Wenjie Qu, Youqi Li, Binghui Wang

**Abstract**: Image segmentation is an important problem in many safety-critical applications. Recent studies show that modern image segmentation models are vulnerable to adversarial perturbations, while existing attack methods mainly follow the idea of attacking image classification models. We argue that image segmentation and classification have inherent differences, and design an attack framework specially for image segmentation models. Our attack framework is inspired by certified radius, which was originally used by defenders to defend against adversarial perturbations to classification models. We are the first, from the attacker perspective, to leverage the properties of certified radius and propose a certified radius guided attack framework against image segmentation models. Specifically, we first adapt randomized smoothing, the state-of-the-art certification method for classification models, to derive the pixel's certified radius. We then focus more on disrupting pixels with relatively smaller certified radii and design a pixel-wise certified radius guided loss, when plugged into any existing white-box attack, yields our certified radius-guided white-box attack. Next, we propose the first black-box attack to image segmentation models via bandit. We design a novel gradient estimator, based on bandit feedback, which is query-efficient and provably unbiased and stable. We use this gradient estimator to design a projected bandit gradient descent (PBGD) attack, as well as a certified radius-guided PBGD (CR-PBGD) attack. We prove our PBGD and CR-PBGD attacks can achieve asymptotically optimal attack performance with an optimal rate. We evaluate our certified-radius guided white-box and black-box attacks on multiple modern image segmentation models and datasets. Our results validate the effectiveness of our certified radius-guided attack framework.



