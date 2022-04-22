# Latest Adversarial Attack Papers
**update at 2022-04-23 06:31:30**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Adversarial Contrastive Learning by Permuting Cluster Assignments**

cs.LG

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2204.10314v1)

**Authors**: Muntasir Wahed, Afrina Tabassum, Ismini Lourentzou

**Abstracts**: Contrastive learning has gained popularity as an effective self-supervised representation learning technique. Several research directions improve traditional contrastive approaches, e.g., prototypical contrastive methods better capture the semantic similarity among instances and reduce the computational burden by considering cluster prototypes or cluster assignments, while adversarial instance-wise contrastive methods improve robustness against a variety of attacks. To the best of our knowledge, no prior work jointly considers robustness, cluster-wise semantic similarity and computational efficiency. In this work, we propose SwARo, an adversarial contrastive framework that incorporates cluster assignment permutations to generate representative adversarial samples. We evaluate SwARo on multiple benchmark datasets and against various white-box and black-box attacks, obtaining consistent improvements over state-of-the-art baselines.



## **2. Robustness of Machine Learning Models Beyond Adversarial Attacks**

cs.LG

25 pages, 7 figures

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2204.10046v1)

**Authors**: Sebastian Scher, Andreas Trügler

**Abstracts**: Correctly quantifying the robustness of machine learning models is a central aspect in judging their suitability for specific tasks, and thus, ultimately, for generating trust in the models. We show that the widely used concept of adversarial robustness and closely related metrics based on counterfactuals are not necessarily valid metrics for determining the robustness of ML models against perturbations that occur "naturally", outside specific adversarial attack scenarios. Additionally, we argue that generic robustness metrics in principle are insufficient for determining real-world-robustness. Instead we propose a flexible approach that models possible perturbations in input data individually for each application. This is then combined with a probabilistic approach that computes the likelihood that a real-world perturbation will change a prediction, thus giving quantitative information of the robustness of the trained machine learning model. The method does not require access to the internals of the classifier and thus in principle works for any black-box model. It is, however, based on Monte-Carlo sampling and thus only suited for input spaces with small dimensions. We illustrate our approach on two dataset, as well as on analytically solvable cases. Finally, we discuss ideas on how real-world robustness could be computed or estimated in high-dimensional input spaces.



## **3. Is Neuron Coverage Needed to Make Person Detection More Robust?**

cs.CV

Accepted for publication at CVPR 2022 TCV workshop

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2204.10027v1)

**Authors**: Svetlana Pavlitskaya, Şiyar Yıkmış, J. Marius Zöllner

**Abstracts**: The growing use of deep neural networks (DNNs) in safety- and security-critical areas like autonomous driving raises the need for their systematic testing. Coverage-guided testing (CGT) is an approach that applies mutation or fuzzing according to a predefined coverage metric to find inputs that cause misbehavior. With the introduction of a neuron coverage metric, CGT has also recently been applied to DNNs. In this work, we apply CGT to the task of person detection in crowded scenes. The proposed pipeline uses YOLOv3 for person detection and includes finding DNN bugs via sampling and mutation, and subsequent DNN retraining on the updated training set. To be a bug, we require a mutated image to cause a significant performance drop compared to a clean input. In accordance with the CGT, we also consider an additional requirement of increased coverage in the bug definition. In order to explore several types of robustness, our approach includes natural image transformations, corruptions, and adversarial examples generated with the Daedalus attack. The proposed framework has uncovered several thousand cases of incorrect DNN behavior. The relative change in mAP performance of the retrained models reached on average between 26.21\% and 64.24\% for different robustness types. However, we have found no evidence that the investigated coverage metrics can be advantageously used to improve robustness.



## **4. Eliminating Backdoor Triggers for Deep Neural Networks Using Attention Relation Graph Distillation**

cs.LG

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2204.09975v1)

**Authors**: Jun Xia, Ting Wang, Jieping Ding, Xian Wei, Mingsong Chen

**Abstracts**: Due to the prosperity of Artificial Intelligence (AI) techniques, more and more backdoors are designed by adversaries to attack Deep Neural Networks (DNNs).Although the state-of-the-art method Neural Attention Distillation (NAD) can effectively erase backdoor triggers from DNNs, it still suffers from non-negligible Attack Success Rate (ASR) together with lowered classification ACCuracy (ACC), since NAD focuses on backdoor defense using attention features (i.e., attention maps) of the same order. In this paper, we introduce a novel backdoor defense framework named Attention Relation Graph Distillation (ARGD), which fully explores the correlation among attention features with different orders using our proposed Attention Relation Graphs (ARGs). Based on the alignment of ARGs between both teacher and student models during knowledge distillation, ARGD can eradicate more backdoor triggers than NAD. Comprehensive experimental results show that, against six latest backdoor attacks, ARGD outperforms NAD by up to 94.85% reduction in ASR, while ACC can be improved by up to 3.23%.



## **5. On the Certified Robustness for Ensemble Models and Beyond**

cs.LG

ICLR 2022. 51 pages, 10 pages for main text. Forum and code:  https://openreview.net/forum?id=tUa4REjGjTf

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2107.10873v2)

**Authors**: Zhuolin Yang, Linyi Li, Xiaojun Xu, Bhavya Kailkhura, Tao Xie, Bo Li

**Abstracts**: Recent studies show that deep neural networks (DNN) are vulnerable to adversarial examples, which aim to mislead DNNs by adding perturbations with small magnitude. To defend against such attacks, both empirical and theoretical defense approaches have been extensively studied for a single ML model. In this work, we aim to analyze and provide the certified robustness for ensemble ML models, together with the sufficient and necessary conditions of robustness for different ensemble protocols. Although ensemble models are shown more robust than a single model empirically; surprisingly, we find that in terms of the certified robustness the standard ensemble models only achieve marginal improvement compared to a single model. Thus, to explore the conditions that guarantee to provide certifiably robust ensemble ML models, we first prove that diversified gradient and large confidence margin are sufficient and necessary conditions for certifiably robust ensemble models under the model-smoothness assumption. We then provide the bounded model-smoothness analysis based on the proposed Ensemble-before-Smoothing strategy. We also prove that an ensemble model can always achieve higher certified robustness than a single base model under mild conditions. Inspired by the theoretical findings, we propose the lightweight Diversity Regularized Training (DRT) to train certifiably robust ensemble ML models. Extensive experiments show that our DRT enhanced ensembles can consistently achieve higher certified robustness than existing single and ensemble ML models, demonstrating the state-of-the-art certified L2-robustness on MNIST, CIFAR-10, and ImageNet datasets.



## **6. Fast AdvProp**

cs.CV

ICLR 2022 camera ready version

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2204.09838v1)

**Authors**: Jieru Mei, Yucheng Han, Yutong Bai, Yixiao Zhang, Yingwei Li, Xianhang Li, Alan Yuille, Cihang Xie

**Abstracts**: Adversarial Propagation (AdvProp) is an effective way to improve recognition models, leveraging adversarial examples. Nonetheless, AdvProp suffers from the extremely slow training speed, mainly because: a) extra forward and backward passes are required for generating adversarial examples; b) both original samples and their adversarial counterparts are used for training (i.e., 2$\times$ data). In this paper, we introduce Fast AdvProp, which aggressively revamps AdvProp's costly training components, rendering the method nearly as cheap as the vanilla training. Specifically, our modifications in Fast AdvProp are guided by the hypothesis that disentangled learning with adversarial examples is the key for performance improvements, while other training recipes (e.g., paired clean and adversarial training samples, multi-step adversarial attackers) could be largely simplified.   Our empirical results show that, compared to the vanilla training baseline, Fast AdvProp is able to further model performance on a spectrum of visual benchmarks, without incurring extra training cost. Additionally, our ablations find Fast AdvProp scales better if larger models are used, is compatible with existing data augmentation methods (i.e., Mixup and CutMix), and can be easily adapted to other recognition tasks like object detection. The code is available here: https://github.com/meijieru/fast_advprop.



## **7. GUARD: Graph Universal Adversarial Defense**

cs.LG

Code is publicly available at https://github.com/EdisonLeeeee/GUARD

**SubmitDate**: 2022-04-20    [paper-pdf](http://arxiv.org/pdf/2204.09803v1)

**Authors**: Jintang Li, Jie Liao, Ruofan Wu, Liang Chen, Changhua Meng, Zibin Zheng, Weiqiang Wang

**Abstracts**: Recently, graph convolutional networks (GCNs) have shown to be vulnerable to small adversarial perturbations, which becomes a severe threat and largely limits their applications in security-critical scenarios. To mitigate such a threat, considerable research efforts have been devoted to increasing the robustness of GCNs against adversarial attacks. However, current approaches for defense are typically designed for the whole graph and consider the global performance, posing challenges in protecting important local nodes from stronger adversarial targeted attacks. In this work, we present a simple yet effective method, named \textbf{\underline{G}}raph \textbf{\underline{U}}niversal \textbf{\underline{A}}dve\textbf{\underline{R}}sarial \textbf{\underline{D}}efense (GUARD). Unlike previous works, GUARD protects each individual node from attacks with a universal defensive patch, which is generated once and can be applied to any node (node-agnostic) in a graph. Extensive experiments on four benchmark datasets demonstrate that our method significantly improves robustness for several established GCNs against multiple adversarial attacks and outperforms existing adversarial defense methods by large margins. Our code is publicly available at https://github.com/EdisonLeeeee/GUARD.



## **8. Backdooring Explainable Machine Learning**

cs.CR

**SubmitDate**: 2022-04-20    [paper-pdf](http://arxiv.org/pdf/2204.09498v1)

**Authors**: Maximilian Noppel, Lukas Peter, Christian Wressnegger

**Abstracts**: Explainable machine learning holds great potential for analyzing and understanding learning-based systems. These methods can, however, be manipulated to present unfaithful explanations, giving rise to powerful and stealthy adversaries. In this paper, we demonstrate blinding attacks that can fully disguise an ongoing attack against the machine learning model. Similar to neural backdoors, we modify the model's prediction upon trigger presence but simultaneously also fool the provided explanation. This enables an adversary to hide the presence of the trigger or point the explanation to entirely different portions of the input, throwing a red herring. We analyze different manifestations of such attacks for different explanation types in the image domain, before we resume to conduct a red-herring attack against malware classification.



## **9. Adversarial Scratches: Deployable Attacks to CNN Classifiers**

cs.LG

This paper stems from 'Scratch that! An Evolution-based Adversarial  Attack against Neural Networks' for which an arXiv preprint is available at  arXiv:1912.02316. Further studies led to a complete overhaul of the work,  resulting in this paper. This work was submitted for review in Pattern  Recognition (Elsevier)

**SubmitDate**: 2022-04-20    [paper-pdf](http://arxiv.org/pdf/2204.09397v1)

**Authors**: Loris Giulivi, Malhar Jere, Loris Rossi, Farinaz Koushanfar, Gabriela Ciocarlie, Briland Hitaj, Giacomo Boracchi

**Abstracts**: A growing body of work has shown that deep neural networks are susceptible to adversarial examples. These take the form of small perturbations applied to the model's input which lead to incorrect predictions. Unfortunately, most literature focuses on visually imperceivable perturbations to be applied to digital images that often are, by design, impossible to be deployed to physical targets. We present Adversarial Scratches: a novel L0 black-box attack, which takes the form of scratches in images, and which possesses much greater deployability than other state-of-the-art attacks. Adversarial Scratches leverage B\'ezier Curves to reduce the dimension of the search space and possibly constrain the attack to a specific location. We test Adversarial Scratches in several scenarios, including a publicly available API and images of traffic signs. Results show that, often, our attack achieves higher fooling rate than other deployable state-of-the-art methods, while requiring significantly fewer queries and modifying very few pixels.



## **10. You Are What You Write: Preserving Privacy in the Era of Large Language Models**

cs.CL

**SubmitDate**: 2022-04-20    [paper-pdf](http://arxiv.org/pdf/2204.09391v1)

**Authors**: Richard Plant, Valerio Giuffrida, Dimitra Gkatzia

**Abstracts**: Large scale adoption of large language models has introduced a new era of convenient knowledge transfer for a slew of natural language processing tasks. However, these models also run the risk of undermining user trust by exposing unwanted information about the data subjects, which may be extracted by a malicious party, e.g. through adversarial attacks. We present an empirical investigation into the extent of the personal information encoded into pre-trained representations by a range of popular models, and we show a positive correlation between the complexity of a model, the amount of data used in pre-training, and data leakage. In this paper, we present the first wide coverage evaluation and comparison of some of the most popular privacy-preserving algorithms, on a large, multi-lingual dataset on sentiment analysis annotated with demographic information (location, age and gender). The results show since larger and more complex models are more prone to leaking private information, use of privacy-preserving methods is highly desirable. We also find that highly privacy-preserving technologies like differential privacy (DP) can have serious model utility effects, which can be ameliorated using hybrid or metric-DP techniques.



## **11. Identifying Near-Optimal Single-Shot Attacks on ICSs with Limited Process Knowledge**

cs.CR

This paper has been accepted at Applied Cryptography and Network  Security (ACNS) 2022

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2204.09106v1)

**Authors**: Herson Esquivel-Vargas, John Henry Castellanos, Marco Caselli, Nils Ole Tippenhauer, Andreas Peter

**Abstracts**: Industrial Control Systems (ICSs) rely on insecure protocols and devices to monitor and operate critical infrastructure. Prior work has demonstrated that powerful attackers with detailed system knowledge can manipulate exchanged sensor data to deteriorate performance of the process, even leading to full shutdowns of plants. Identifying those attacks requires iterating over all possible sensor values, and running detailed system simulation or analysis to identify optimal attacks. That setup allows adversaries to identify attacks that are most impactful when applied on the system for the first time, before the system operators become aware of the manipulations.   In this work, we investigate if constrained attackers without detailed system knowledge and simulators can identify comparable attacks. In particular, the attacker only requires abstract knowledge on general information flow in the plant, instead of precise algorithms, operating parameters, process models, or simulators. We propose an approach that allows single-shot attacks, i.e., near-optimal attacks that are reliably shutting down a system on the first try. The approach is applied and validated on two use cases, and demonstrated to achieve comparable results to prior work, which relied on detailed system information and simulations.



## **12. Indiscriminate Data Poisoning Attacks on Neural Networks**

cs.LG

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2204.09092v1)

**Authors**: Yiwei Lu, Gautam Kamath, Yaoliang Yu

**Abstracts**: Data poisoning attacks, in which a malicious adversary aims to influence a model by injecting "poisoned" data into the training process, have attracted significant recent attention. In this work, we take a closer look at existing poisoning attacks and connect them with old and new algorithms for solving sequential Stackelberg games. By choosing an appropriate loss function for the attacker and optimizing with algorithms that exploit second-order information, we design poisoning attacks that are effective on neural networks. We present efficient implementations that exploit modern auto-differentiation packages and allow simultaneous and coordinated generation of tens of thousands of poisoned points, in contrast to existing methods that generate poisoned points one by one. We further perform extensive experiments that empirically explore the effect of data poisoning attacks on deep neural networks.



## **13. A Brief Survey on Deep Learning Based Data Hiding**

cs.CR

v2: reorganize some sections and add several new papers published in  2021~2022

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2103.01607v2)

**Authors**: Chaoning Zhang, Chenguo Lin, Philipp Benz, Kejiang Chen, Weiming Zhang, In So Kweon

**Abstracts**: Data hiding is the art of concealing messages with limited perceptual changes. Recently, deep learning has enriched it from various perspectives with significant progress. In this work, we conduct a brief yet comprehensive review of existing literature for deep learning based data hiding (deep hiding) by first classifying it according to three essential properties (i.e., capacity, security and robustness), and outline three commonly used architectures. Based on this, we summarize specific strategies for different applications of data hiding, including basic hiding, steganography, watermarking and light field messaging. Finally, further insight into deep hiding is provided by incorporating the perspective of adversarial attack.



## **14. Jacobian Ensembles Improve Robustness Trade-offs to Adversarial Attacks**

cs.LG

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2204.08726v1)

**Authors**: Kenneth T. Co, David Martinez-Rego, Zhongyuan Hau, Emil C. Lupu

**Abstracts**: Deep neural networks have become an integral part of our software infrastructure and are being deployed in many widely-used and safety-critical applications. However, their integration into many systems also brings with it the vulnerability to test time attacks in the form of Universal Adversarial Perturbations (UAPs). UAPs are a class of perturbations that when applied to any input causes model misclassification. Although there is an ongoing effort to defend models against these adversarial attacks, it is often difficult to reconcile the trade-offs in model accuracy and robustness to adversarial attacks. Jacobian regularization has been shown to improve the robustness of models against UAPs, whilst model ensembles have been widely adopted to improve both predictive performance and model robustness. In this work, we propose a novel approach, Jacobian Ensembles-a combination of Jacobian regularization and model ensembles to significantly increase the robustness against UAPs whilst maintaining or improving model accuracy. Our results show that Jacobian Ensembles achieves previously unseen levels of accuracy and robustness, greatly improving over previous methods that tend to skew towards only either accuracy or robustness.



## **15. Topology and geometry of data manifold in deep learning**

cs.LG

12 pages, 15 figures

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2204.08624v1)

**Authors**: German Magai, Anton Ayzenberg

**Abstracts**: Despite significant advances in the field of deep learning in applications to various fields, explaining the inner processes of deep learning models remains an important and open question. The purpose of this article is to describe and substantiate the geometric and topological view of the learning process of neural networks. Our attention is focused on the internal representation of neural networks and on the dynamics of changes in the topology and geometry of the data manifold on different layers. We also propose a method for assessing the generalizing ability of neural networks based on topological descriptors. In this paper, we use the concepts of topological data analysis and intrinsic dimension, and we present a wide range of experiments on different datasets and different configurations of convolutional neural network architectures. In addition, we consider the issue of the geometry of adversarial attacks in the classification task and spoofing attacks on face recognition systems. Our work is a contribution to the development of an important area of explainable and interpretable AI through the example of computer vision.



## **16. Poisons that are learned faster are more effective**

cs.LG

8 pages, 4 figures. Accepted to CVPR 2022 Art of Robustness Workshop

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2204.08615v1)

**Authors**: Pedro Sandoval-Segura, Vasu Singla, Liam Fowl, Jonas Geiping, Micah Goldblum, David Jacobs, Tom Goldstein

**Abstracts**: Imperceptible poisoning attacks on entire datasets have recently been touted as methods for protecting data privacy. However, among a number of defenses preventing the practical use of these techniques, early-stopping stands out as a simple, yet effective defense. To gauge poisons' vulnerability to early-stopping, we benchmark error-minimizing, error-maximizing, and synthetic poisons in terms of peak test accuracy over 100 epochs and make a number of surprising observations. First, we find that poisons that reach a low training loss faster have lower peak test accuracy. Second, we find that a current state-of-the-art error-maximizing poison is 7 times less effective when poison training is stopped at epoch 8. Third, we find that stronger, more transferable adversarial attacks do not make stronger poisons. We advocate for evaluating poisons in terms of peak test accuracy.



## **17. Metamorphic Testing-based Adversarial Attack to Fool Deepfake Detectors**

cs.CV

paper submitted to 26TH International Conference on Pattern  Recognition (ICPR2022)

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2204.08612v1)

**Authors**: Nyee Thoang Lim, Meng Yi Kuan, Muxin Pu, Mei Kuan Lim, Chun Yong Chong

**Abstracts**: Deepfakes utilise Artificial Intelligence (AI) techniques to create synthetic media where the likeness of one person is replaced with another. There are growing concerns that deepfakes can be maliciously used to create misleading and harmful digital contents. As deepfakes become more common, there is a dire need for deepfake detection technology to help spot deepfake media. Present deepfake detection models are able to achieve outstanding accuracy (>90%). However, most of them are limited to within-dataset scenario, where the same dataset is used for training and testing. Most models do not generalise well enough in cross-dataset scenario, where models are tested on unseen datasets from another source. Furthermore, state-of-the-art deepfake detection models rely on neural network-based classification models that are known to be vulnerable to adversarial attacks. Motivated by the need for a robust deepfake detection model, this study adapts metamorphic testing (MT) principles to help identify potential factors that could influence the robustness of the examined model, while overcoming the test oracle problem in this domain. Metamorphic testing is specifically chosen as the testing technique as it fits our demand to address learning-based system testing with probabilistic outcomes from largely black-box components, based on potentially large input domains. We performed our evaluations on MesoInception-4 and TwoStreamNet models, which are the state-of-the-art deepfake detection models. This study identified makeup application as an adversarial attack that could fool deepfake detectors. Our experimental results demonstrate that both the MesoInception-4 and TwoStreamNet models degrade in their performance by up to 30\% when the input data is perturbed with makeup.



## **18. UNBUS: Uncertainty-aware Deep Botnet Detection System in Presence of Perturbed Samples**

cs.CR

8 pages, 5 figures, 5 Tables

**SubmitDate**: 2022-04-18    [paper-pdf](http://arxiv.org/pdf/2204.09502v1)

**Authors**: Rahim Taheri

**Abstracts**: A rising number of botnet families have been successfully detected using deep learning architectures. While the variety of attacks increases, these architectures should become more robust against attacks. They have been proven to be very sensitive to small but well constructed perturbations in the input. Botnet detection requires extremely low false-positive rates (FPR), which are not commonly attainable in contemporary deep learning. Attackers try to increase the FPRs by making poisoned samples. The majority of recent research has focused on the use of model loss functions to build adversarial examples and robust models. In this paper, two LSTM-based classification algorithms for botnet classification with an accuracy higher than 98\% are presented. Then, the adversarial attack is proposed, which reduces the accuracy to about30\%. Then, by examining the methods for computing the uncertainty, the defense method is proposed to increase the accuracy to about 70\%. By using the deep ensemble and stochastic weight averaging quantification methods it has been investigated the uncertainty of the accuracy in the proposed methods.



## **19. A Comprehensive Survey on Trustworthy Graph Neural Networks: Privacy, Robustness, Fairness, and Explainability**

cs.LG

**SubmitDate**: 2022-04-18    [paper-pdf](http://arxiv.org/pdf/2204.08570v1)

**Authors**: Enyan Dai, Tianxiang Zhao, Huaisheng Zhu, Junjie Xu, Zhimeng Guo, Hui Liu, Jiliang Tang, Suhang Wang

**Abstracts**: Graph Neural Networks (GNNs) have made rapid developments in the recent years. Due to their great ability in modeling graph-structured data, GNNs are vastly used in various applications, including high-stakes scenarios such as financial analysis, traffic predictions, and drug discovery. Despite their great potential in benefiting humans in the real world, recent study shows that GNNs can leak private information, are vulnerable to adversarial attacks, can inherit and magnify societal bias from training data and lack interpretability, which have risk of causing unintentional harm to the users and society. For example, existing works demonstrate that attackers can fool the GNNs to give the outcome they desire with unnoticeable perturbation on training graph. GNNs trained on social networks may embed the discrimination in their decision process, strengthening the undesirable societal bias. Consequently, trustworthy GNNs in various aspects are emerging to prevent the harm from GNN models and increase the users' trust in GNNs. In this paper, we give a comprehensive survey of GNNs in the computational aspects of privacy, robustness, fairness, and explainability. For each aspect, we give the taxonomy of the related methods and formulate the general frameworks for the multiple categories of trustworthy GNNs. We also discuss the future research directions of each aspect and connections between these aspects to help achieve trustworthiness.



## **20. Special Session: Towards an Agile Design Methodology for Efficient, Reliable, and Secure ML Systems**

cs.AR

Appears at 40th IEEE VLSI Test Symposium (VTS 2022), 14 pages

**SubmitDate**: 2022-04-18    [paper-pdf](http://arxiv.org/pdf/2204.09514v1)

**Authors**: Shail Dave, Alberto Marchisio, Muhammad Abdullah Hanif, Amira Guesmi, Aviral Shrivastava, Ihsen Alouani, Muhammad Shafique

**Abstracts**: The real-world use cases of Machine Learning (ML) have exploded over the past few years. However, the current computing infrastructure is insufficient to support all real-world applications and scenarios. Apart from high efficiency requirements, modern ML systems are expected to be highly reliable against hardware failures as well as secure against adversarial and IP stealing attacks. Privacy concerns are also becoming a first-order issue. This article summarizes the main challenges in agile development of efficient, reliable and secure ML systems, and then presents an outline of an agile design methodology to generate efficient, reliable and secure ML systems based on user-defined constraints and objectives.



## **21. Optimal Layered Defense For Site Protection**

cs.OH

**SubmitDate**: 2022-04-18    [paper-pdf](http://arxiv.org/pdf/2204.08961v1)

**Authors**: Tsvetan Asamov, Emre Yamangil, Endre Boros, Paul Kantor, Fred Roberts

**Abstracts**: We present a model for layered security with applications to the protection of sites such as stadiums or large gathering places. We formulate the problem as one of maximizing the capture of illegal contraband. The objective function is indefinite and only limited information can be gained when the problem is solved by standard convex optimization methods. In order to solve the model, we develop a dynamic programming approach, and study its convergence properties. Additionally, we formulate a version of the problem aimed at addressing intelligent adversaries who can adjust their direction of attack as they observe changes in the site security. Furthermore, we also develop a method for the solution of the latter model. Finally, we perform computational experiments to demonstrate the use of our methods.



## **22. Sardino: Ultra-Fast Dynamic Ensemble for Secure Visual Sensing at Mobile Edge**

cs.CV

**SubmitDate**: 2022-04-18    [paper-pdf](http://arxiv.org/pdf/2204.08189v1)

**Authors**: Qun Song, Zhenyu Yan, Wenjie Luo, Rui Tan

**Abstracts**: Adversarial example attack endangers the mobile edge systems such as vehicles and drones that adopt deep neural networks for visual sensing. This paper presents {\em Sardino}, an active and dynamic defense approach that renews the inference ensemble at run time to develop security against the adaptive adversary who tries to exfiltrate the ensemble and construct the corresponding effective adversarial examples. By applying consistency check and data fusion on the ensemble's predictions, Sardino can detect and thwart adversarial inputs. Compared with the training-based ensemble renewal, we use HyperNet to achieve {\em one million times} acceleration and per-frame ensemble renewal that presents the highest level of difficulty to the prerequisite exfiltration attacks. Moreover, the robustness of the renewed ensembles against adversarial examples is enhanced with adversarial learning for the HyperNet. We design a run-time planner that maximizes the ensemble size in favor of security while maintaining the processing frame rate. Beyond adversarial examples, Sardino can also address the issue of out-of-distribution inputs effectively. This paper presents extensive evaluation of Sardino's performance in counteracting adversarial examples and applies it to build a real-time car-borne traffic sign recognition system. Live on-road tests show the built system's effectiveness in maintaining frame rate and detecting out-of-distribution inputs due to the false positives of a preceding YOLO-based traffic sign detector.



## **23. Learning Compositional Representations for Effective Low-Shot Generalization**

cs.CV

**SubmitDate**: 2022-04-17    [paper-pdf](http://arxiv.org/pdf/2204.08090v1)

**Authors**: Samarth Mishra, Pengkai Zhu, Venkatesh Saligrama

**Abstracts**: We propose Recognition as Part Composition (RPC), an image encoding approach inspired by human cognition. It is based on the cognitive theory that humans recognize complex objects by components, and that they build a small compact vocabulary of concepts to represent each instance with. RPC encodes images by first decomposing them into salient parts, and then encoding each part as a mixture of a small number of prototypes, each representing a certain concept. We find that this type of learning inspired by human cognition can overcome hurdles faced by deep convolutional networks in low-shot generalization tasks, like zero-shot learning, few-shot learning and unsupervised domain adaptation. Furthermore, we find a classifier using an RPC image encoder is fairly robust to adversarial attacks, that deep neural networks are known to be prone to. Given that our image encoding principle is based on human cognition, one would expect the encodings to be interpretable by humans, which we find to be the case via crowd-sourcing experiments. Finally, we propose an application of these interpretable encodings in the form of generating synthetic attribute annotations for evaluating zero-shot learning methods on new datasets.



## **24. Residue-Based Natural Language Adversarial Attack Detection**

cs.CL

**SubmitDate**: 2022-04-17    [paper-pdf](http://arxiv.org/pdf/2204.10192v1)

**Authors**: Vyas Raina, Mark Gales

**Abstracts**: Deep learning based systems are susceptible to adversarial attacks, where a small, imperceptible change at the input alters the model prediction. However, to date the majority of the approaches to detect these attacks have been designed for image processing systems. Many popular image adversarial detection approaches are able to identify adversarial examples from embedding feature spaces, whilst in the NLP domain existing state of the art detection approaches solely focus on input text features, without consideration of model embedding spaces. This work examines what differences result when porting these image designed strategies to Natural Language Processing (NLP) tasks - these detectors are found to not port over well. This is expected as NLP systems have a very different form of input: discrete and sequential in nature, rather than the continuous and fixed size inputs for images. As an equivalent model-focused NLP detection approach, this work proposes a simple sentence-embedding "residue" based detector to identify adversarial examples. On many tasks, it out-performs ported image domain detectors and recent state of the art NLP specific detectors.



## **25. Towards Comprehensive Testing on the Robustness of Cooperative Multi-agent Reinforcement Learning**

cs.MA

**SubmitDate**: 2022-04-17    [paper-pdf](http://arxiv.org/pdf/2204.07932v1)

**Authors**: Jun Guo, Yonghong Chen, Yihang Hao, Zixin Yin, Yin Yu, Simin Li

**Abstracts**: While deep neural networks (DNNs) have strengthened the performance of cooperative multi-agent reinforcement learning (c-MARL), the agent policy can be easily perturbed by adversarial examples. Considering the safety critical applications of c-MARL, such as traffic management, power management and unmanned aerial vehicle control, it is crucial to test the robustness of c-MARL algorithm before it was deployed in reality. Existing adversarial attacks for MARL could be used for testing, but is limited to one robustness aspects (e.g., reward, state, action), while c-MARL model could be attacked from any aspect. To overcome the challenge, we propose MARLSafe, the first robustness testing framework for c-MARL algorithms. First, motivated by Markov Decision Process (MDP), MARLSafe consider the robustness of c-MARL algorithms comprehensively from three aspects, namely state robustness, action robustness and reward robustness. Any c-MARL algorithm must simultaneously satisfy these robustness aspects to be considered secure. Second, due to the scarceness of c-MARL attack, we propose c-MARL attacks as robustness testing algorithms from multiple aspects. Experiments on \textit{SMAC} environment reveals that many state-of-the-art c-MARL algorithms are of low robustness in all aspect, pointing out the urgent need to test and enhance robustness of c-MARL algorithms.



## **26. SETTI: A Self-supervised Adversarial Malware Detection Architecture in an IoT Environment**

cs.CR

20 pages, 6 figures, 2 Tables, Submitted to ACM Transactions on  Multimedia Computing, Communications, and Applications

**SubmitDate**: 2022-04-16    [paper-pdf](http://arxiv.org/pdf/2204.07772v1)

**Authors**: Marjan Golmaryami, Rahim Taheri, Zahra Pooranian, Mohammad Shojafar, Pei Xiao

**Abstracts**: In recent years, malware detection has become an active research topic in the area of Internet of Things (IoT) security. The principle is to exploit knowledge from large quantities of continuously generated malware. Existing algorithms practice available malware features for IoT devices and lack real-time prediction behaviors. More research is thus required on malware detection to cope with real-time misclassification of the input IoT data. Motivated by this, in this paper we propose an adversarial self-supervised architecture for detecting malware in IoT networks, SETTI, considering samples of IoT network traffic that may not be labeled. In the SETTI architecture, we design three self-supervised attack techniques, namely Self-MDS, GSelf-MDS and ASelf-MDS. The Self-MDS method considers the IoT input data and the adversarial sample generation in real-time. The GSelf-MDS builds a generative adversarial network model to generate adversarial samples in the self-supervised structure. Finally, ASelf-MDS utilizes three well-known perturbation sample techniques to develop adversarial malware and inject it over the self-supervised architecture. Also, we apply a defence method to mitigate these attacks, namely adversarial self-supervised training to protect the malware detection architecture against injecting the malicious samples. To validate the attack and defence algorithms, we conduct experiments on two recent IoT datasets: IoT23 and NBIoT. Comparison of the results shows that in the IoT23 dataset, the Self-MDS method has the most damaging consequences from the attacker's point of view by reducing the accuracy rate from 98% to 74%. In the NBIoT dataset, the ASelf-MDS method is the most devastating algorithm that can plunge the accuracy rate from 98% to 77%.



## **27. Homomorphic Encryption and Federated Learning based Privacy-Preserving CNN Training: COVID-19 Detection Use-Case**

cs.CR

European Interdisciplinary Cybersecurity Conference (EICC) 2022  publication

**SubmitDate**: 2022-04-16    [paper-pdf](http://arxiv.org/pdf/2204.07752v1)

**Authors**: Febrianti Wibawa, Ferhat Ozgur Catak, Salih Sarp, Murat Kuzlu, Umit Cali

**Abstracts**: Medical data is often highly sensitive in terms of data privacy and security concerns. Federated learning, one type of machine learning techniques, has been started to use for the improvement of the privacy and security of medical data. In the federated learning, the training data is distributed across multiple machines, and the learning process is performed in a collaborative manner. There are several privacy attacks on deep learning (DL) models to get the sensitive information by attackers. Therefore, the DL model itself should be protected from the adversarial attack, especially for applications using medical data. One of the solutions for this problem is homomorphic encryption-based model protection from the adversary collaborator. This paper proposes a privacy-preserving federated learning algorithm for medical data using homomorphic encryption. The proposed algorithm uses a secure multi-party computation protocol to protect the deep learning model from the adversaries. In this study, the proposed algorithm using a real-world medical dataset is evaluated in terms of the model performance.



## **28. An Overview of Compressible and Learnable Image Transformation with Secret Key and Its Applications**

cs.CV

**SubmitDate**: 2022-04-16    [paper-pdf](http://arxiv.org/pdf/2201.11006v2)

**Authors**: Hitoshi Kiya, AprilPyone MaungMaung, Yuma Kinoshita, Shoko Imaizumi, Sayaka Shiota

**Abstracts**: This article presents an overview of image transformation with a secret key and its applications. Image transformation with a secret key enables us not only to protect visual information on plain images but also to embed unique features controlled with a key into images. In addition, numerous encryption methods can generate encrypted images that are compressible and learnable for machine learning. Various applications of such transformation have been developed by using these properties. In this paper, we focus on a class of image transformation referred to as learnable image encryption, which is applicable to privacy-preserving machine learning and adversarially robust defense. Detailed descriptions of both transformation algorithms and performances are provided. Moreover, we discuss robustness against various attacks.



## **29. Revisiting the Adversarial Robustness-Accuracy Tradeoff in Robot Learning**

cs.RO

**SubmitDate**: 2022-04-15    [paper-pdf](http://arxiv.org/pdf/2204.07373v1)

**Authors**: Mathias Lechner, Alexander Amini, Daniela Rus, Thomas A. Henzinger

**Abstracts**: Adversarial training (i.e., training on adversarially perturbed input data) is a well-studied method for making neural networks robust to potential adversarial attacks during inference. However, the improved robustness does not come for free but rather is accompanied by a decrease in overall model accuracy and performance. Recent work has shown that, in practical robot learning applications, the effects of adversarial training do not pose a fair trade-off but inflict a net loss when measured in holistic robot performance. This work revisits the robustness-accuracy trade-off in robot learning by systematically analyzing if recent advances in robust training methods and theory in conjunction with adversarial robot learning can make adversarial training suitable for real-world robot applications. We evaluate a wide variety of robot learning tasks ranging from autonomous driving in a high-fidelity environment amenable to sim-to-real deployment, to mobile robot gesture recognition. Our results demonstrate that, while these techniques make incremental improvements on the trade-off on a relative scale, the negative side-effects caused by adversarial training still outweigh the improvements by an order of magnitude. We conclude that more substantial advances in robust learning methods are necessary before they can benefit robot learning tasks in practice.



## **30. Can You Spot the Chameleon? Adversarially Camouflaging Images from Co-Salient Object Detection**

cs.CV

Accepted to CVPR 2022

**SubmitDate**: 2022-04-18    [paper-pdf](http://arxiv.org/pdf/2009.09258v5)

**Authors**: Ruijun Gao, Qing Guo, Felix Juefei-Xu, Hongkai Yu, Huazhu Fu, Wei Feng, Yang Liu, Song Wang

**Abstracts**: Co-salient object detection (CoSOD) has recently achieved significant progress and played a key role in retrieval-related tasks. However, it inevitably poses an entirely new safety and security issue, i.e., highly personal and sensitive content can potentially be extracting by powerful CoSOD methods. In this paper, we address this problem from the perspective of adversarial attacks and identify a novel task: adversarial co-saliency attack. Specially, given an image selected from a group of images containing some common and salient objects, we aim to generate an adversarial version that can mislead CoSOD methods to predict incorrect co-salient regions. Note that, compared with general white-box adversarial attacks for classification, this new task faces two additional challenges: (1) low success rate due to the diverse appearance of images in the group; (2) low transferability across CoSOD methods due to the considerable difference between CoSOD pipelines. To address these challenges, we propose the very first black-box joint adversarial exposure and noise attack (Jadena), where we jointly and locally tune the exposure and additive perturbations of the image according to a newly designed high-feature-level contrast-sensitive loss function. Our method, without any information on the state-of-the-art CoSOD methods, leads to significant performance degradation on various co-saliency detection datasets and makes the co-salient objects undetectable. This can have strong practical benefits in properly securing the large number of personal photos currently shared on the Internet. Moreover, our method is potential to be utilized as a metric for evaluating the robustness of CoSOD methods.



## **31. Robotic and Generative Adversarial Attacks in Offline Writer-independent Signature Verification**

cs.RO

**SubmitDate**: 2022-04-14    [paper-pdf](http://arxiv.org/pdf/2204.07246v1)

**Authors**: Jordan J. Bird

**Abstracts**: This study explores how robots and generative approaches can be used to mount successful false-acceptance adversarial attacks on signature verification systems. Initially, a convolutional neural network topology and data augmentation strategy are explored and tuned, producing an 87.12% accurate model for the verification of 2,640 human signatures. Two robots are then tasked with forging 50 signatures, where 25 are used for the verification attack, and the remaining 25 are used for tuning of the model to defend against them. Adversarial attacks on the system show that there exists an information security risk; the Line-us robotic arm can fool the system 24% of the time and the iDraw 2.0 robot 32% of the time. A conditional GAN finds similar success, with around 30% forged signatures misclassified as genuine. Following fine-tune transfer learning of robotic and generative data, adversarial attacks are reduced below the model threshold by both robots and the GAN. It is observed that tuning the model reduces the risk of attack by robots to 8% and 12%, and that conditional generative adversarial attacks can be reduced to 4% when 25 images are presented and 5% when 1000 images are presented.



## **32. ExPLoit: Extracting Private Labels in Split Learning**

cs.CR

**SubmitDate**: 2022-04-14    [paper-pdf](http://arxiv.org/pdf/2112.01299v2)

**Authors**: Sanjay Kariyappa, Moinuddin K Qureshi

**Abstracts**: Split learning is a popular technique used for vertical federated learning (VFL), where the goal is to jointly train a model on the private input and label data held by two parties. This technique uses a split-model, trained end-to-end, by exchanging the intermediate representations (IR) of the inputs and gradients of the IR between the two parties. We propose ExPLoit - a label-leakage attack that allows an adversarial input-owner to extract the private labels of the label-owner during split-learning. ExPLoit frames the attack as a supervised learning problem by using a novel loss function that combines gradient-matching and several regularization terms developed using key properties of the dataset and models. Our evaluations show that ExPLoit can uncover the private labels with near-perfect accuracy of up to 99.96%. Our findings underscore the need for better training techniques for VFL.



## **33. From Environmental Sound Representation to Robustness of 2D CNN Models Against Adversarial Attacks**

cs.SD

32 pages, Preprint Submitted to Journal of Applied Acoustics. arXiv  admin note: substantial text overlap with arXiv:2007.13703

**SubmitDate**: 2022-04-14    [paper-pdf](http://arxiv.org/pdf/2204.07018v1)

**Authors**: Mohammad Esmaeilpour, Patrick Cardinal, Alessandro Lameiras Koerich

**Abstracts**: This paper investigates the impact of different standard environmental sound representations (spectrograms) on the recognition performance and adversarial attack robustness of a victim residual convolutional neural network, namely ResNet-18. Our main motivation for focusing on such a front-end classifier rather than other complex architectures is balancing recognition accuracy and the total number of training parameters. Herein, we measure the impact of different settings required for generating more informative Mel-frequency cepstral coefficient (MFCC), short-time Fourier transform (STFT), and discrete wavelet transform (DWT) representations on our front-end model. This measurement involves comparing the classification performance over the adversarial robustness. We demonstrate an inverse relationship between recognition accuracy and model robustness against six benchmarking attack algorithms on the balance of average budgets allocated by the adversary and the attack cost. Moreover, our experimental results have shown that while the ResNet-18 model trained on DWT spectrograms achieves a high recognition accuracy, attacking this model is relatively more costly for the adversary than other 2D representations. We also report some results on different convolutional neural network architectures such as ResNet-34, ResNet-56, AlexNet, and GoogLeNet, SB-CNN, and LSTM-based.



## **34. Finding MNEMON: Reviving Memories of Node Embeddings**

cs.LG

To Appear in the 29th ACM Conference on Computer and Communications  Security (CCS), November 7-11, 2022

**SubmitDate**: 2022-04-14    [paper-pdf](http://arxiv.org/pdf/2204.06963v1)

**Authors**: Yun Shen, Yufei Han, Zhikun Zhang, Min Chen, Ting Yu, Michael Backes, Yang Zhang, Gianluca Stringhini

**Abstracts**: Previous security research efforts orbiting around graphs have been exclusively focusing on either (de-)anonymizing the graphs or understanding the security and privacy issues of graph neural networks. Little attention has been paid to understand the privacy risks of integrating the output from graph embedding models (e.g., node embeddings) with complex downstream machine learning pipelines. In this paper, we fill this gap and propose a novel model-agnostic graph recovery attack that exploits the implicit graph structural information preserved in the embeddings of graph nodes. We show that an adversary can recover edges with decent accuracy by only gaining access to the node embedding matrix of the original graph without interactions with the node embedding models. We demonstrate the effectiveness and applicability of our graph recovery attack through extensive experiments.



## **35. Arbitrarily Varying Wiretap Channels with Non-Causal Side Information at the Jammer**

cs.IT

**SubmitDate**: 2022-04-14    [paper-pdf](http://arxiv.org/pdf/2001.03035v4)

**Authors**: Carsten Rudolf Janda, Moritz Wiese, Eduard A. Jorswieck, Holger Boche

**Abstracts**: Secure communication in a potentially malicious environment becomes more and more important. The arbitrarily varying wiretap channel (AVWC) provides information theoretical bounds on how much information can be exchanged even in the presence of an active attacker. If the active attacker has non-causal side information, situations in which a legitimate communication system has been hacked, can be modeled. We investigate the AVWC with non-causal side information at the jammer for the case that there exists a best channel to the eavesdropper. Non-causal side information means that the transmitted codeword is known to an active adversary before it is transmitted. By considering the maximum error criterion, we allow also messages to be known at the jammer before the corresponding codeword is transmitted. A single letter formula for the common randomness secrecy capacity is derived. Additionally, we provide a single letter formula for the common randomness secrecy capacity, for the cases that the channel to the eavesdropper is strongly degraded, strongly noisier, or strongly less capable with respect to the main channel. Furthermore, we compare our results to the random code secrecy capacity for the cases of maximum error criterion but without non-causal side information at the jammer, maximum error criterion with non-causal side information of the messages at the jammer, and the case of average error criterion without non-causal side information at the jammer.



## **36. Improving Adversarial Transferability with Gradient Refining**

cs.CV

Accepted at CVPR 2021 Workshop on Adversarial Machine Learning in  Real-World Computer Vision Systems and Online Challenges. The extension  vision of this paper, please refer to arxiv:2203.13479

**SubmitDate**: 2022-04-14    [paper-pdf](http://arxiv.org/pdf/2105.04834v3)

**Authors**: Guoqiu Wang, Huanqian Yan, Ying Guo, Xingxing Wei

**Abstracts**: Deep neural networks are vulnerable to adversarial examples, which are crafted by adding human-imperceptible perturbations to original images. Most existing adversarial attack methods achieve nearly 100% attack success rates under the white-box setting, but only achieve relatively low attack success rates under the black-box setting. To improve the transferability of adversarial examples for the black-box setting, several methods have been proposed, e.g., input diversity, translation-invariant attack, and momentum-based attack. In this paper, we propose a method named Gradient Refining, which can further improve the adversarial transferability by correcting useless gradients introduced by input diversity through multiple transformations. Our method is generally applicable to many gradient-based attack methods combined with input diversity. Extensive experiments are conducted on the ImageNet dataset and our method can achieve an average transfer success rate of 82.07% for three different models under single-model setting, which outperforms the other state-of-the-art methods by a large margin of 6.0% averagely. And we have applied the proposed method to the competition CVPR 2021 Unrestricted Adversarial Attacks on ImageNet organized by Alibaba and won the second place in attack success rates among 1558 teams.



## **37. Real-time Adversarial Perturbations against Deep Reinforcement Learning Policies: Attacks and Defenses**

cs.LG

13 pages, 6 figures

**SubmitDate**: 2022-04-14    [paper-pdf](http://arxiv.org/pdf/2106.08746v3)

**Authors**: Buse G. A. Tekgul, Shelly Wang, Samuel Marchal, N. Asokan

**Abstracts**: Recent work has shown that deep reinforcement learning (DRL) policies are vulnerable to adversarial perturbations. Adversaries can mislead policies of DRL agents by perturbing the state of the environment observed by the agents. Existing attacks are feasible in principle but face challenges in practice, either by being too slow to fool DRL policies in real time or by modifying past observations stored in the agent's memory. We show that using the Universal Adversarial Perturbation (UAP) method to compute perturbations, independent of the individual inputs to which they are applied to, can fool DRL policies effectively and in real time. We describe three such attack variants. Via an extensive evaluation using three Atari 2600 games, we show that our attacks are effective, as they fully degrade the performance of three different DRL agents (up to 100%, even when the $l_\infty$ bound on the perturbation is as small as 0.01). It is faster compared to the response time (0.6ms on average) of different DRL policies, and considerably faster than prior attacks using adversarial perturbations (1.8ms on average). We also show that our attack technique is efficient, incurring an online computational cost of 0.027ms on average. Using two further tasks involving robotic movement, we confirm that our results generalize to more complex DRL tasks. Furthermore, we demonstrate that the effectiveness of known defenses diminishes against universal perturbations. We propose an effective technique that detects all known adversarial perturbations against DRL policies, including all the universal perturbations presented in this paper.



## **38. Overparameterized Linear Regression under Adversarial Attacks**

stat.ML

**SubmitDate**: 2022-04-13    [paper-pdf](http://arxiv.org/pdf/2204.06274v1)

**Authors**: Antônio H. Ribeiro, Thomas B. Schön

**Abstracts**: As machine learning models start to be used in critical applications, their vulnerabilities and brittleness become a pressing concern. Adversarial attacks are a popular framework for studying these vulnerabilities. In this work, we study the error of linear regression in the face of adversarial attacks. We provide bounds of the error in terms of the traditional risk and the parameter norm and show how these bounds can be leveraged and make it possible to use analysis from non-adversarial setups to study the adversarial risk. The usefulness of these results is illustrated by shedding light on whether or not overparameterized linear models can be adversarially robust. We show that adding features to linear models might be either a source of additional robustness or brittleness. We show that these differences appear due to scaling and how the $\ell_1$ and $\ell_2$ norms of random projections concentrate. We also show how the reformulation we propose allows for solving adversarial training as a convex optimization problem. This is then used as a tool to study how adversarial training and other regularization methods might affect the robustness of the estimated models.



## **39. Towards A Critical Evaluation of Robustness for Deep Learning Backdoor Countermeasures**

cs.CR

**SubmitDate**: 2022-04-13    [paper-pdf](http://arxiv.org/pdf/2204.06273v1)

**Authors**: Huming Qiu, Hua Ma, Zhi Zhang, Alsharif Abuadbba, Wei Kang, Anmin Fu, Yansong Gao

**Abstracts**: Since Deep Learning (DL) backdoor attacks have been revealed as one of the most insidious adversarial attacks, a number of countermeasures have been developed with certain assumptions defined in their respective threat models. However, the robustness of these countermeasures is inadvertently ignored, which can introduce severe consequences, e.g., a countermeasure can be misused and result in a false implication of backdoor detection.   For the first time, we critically examine the robustness of existing backdoor countermeasures with an initial focus on three influential model-inspection ones that are Neural Cleanse (S&P'19), ABS (CCS'19), and MNTD (S&P'21). Although the three countermeasures claim that they work well under their respective threat models, they have inherent unexplored non-robust cases depending on factors such as given tasks, model architectures, datasets, and defense hyper-parameter, which are \textit{not even rooted from delicate adaptive attacks}. We demonstrate how to trivially bypass them aligned with their respective threat models by simply varying aforementioned factors. Particularly, for each defense, formal proofs or empirical studies are used to reveal its two non-robust cases where it is not as robust as it claims or expects, especially the recent MNTD. This work highlights the necessity of thoroughly evaluating the robustness of backdoor countermeasures to avoid their misleading security implications in unknown non-robust cases.



## **40. Towards Practical Robustness Analysis for DNNs based on PAC-Model Learning**

cs.LG

**SubmitDate**: 2022-04-13    [paper-pdf](http://arxiv.org/pdf/2101.10102v2)

**Authors**: Renjue Li, Pengfei Yang, Cheng-Chao Huang, Youcheng Sun, Bai Xue, Lijun Zhang

**Abstracts**: To analyse local robustness properties of deep neural networks (DNNs), we present a practical framework from a model learning perspective. Based on black-box model learning with scenario optimisation, we abstract the local behaviour of a DNN via an affine model with the probably approximately correct (PAC) guarantee. From the learned model, we can infer the corresponding PAC-model robustness property. The innovation of our work is the integration of model learning into PAC robustness analysis: that is, we construct a PAC guarantee on the model level instead of sample distribution, which induces a more faithful and accurate robustness evaluation. This is in contrast to existing statistical methods without model learning. We implement our method in a prototypical tool named DeepPAC. As a black-box method, DeepPAC is scalable and efficient, especially when DNNs have complex structures or high-dimensional inputs. We extensively evaluate DeepPAC, with 4 baselines (using formal verification, statistical methods, testing and adversarial attack) and 20 DNN models across 3 datasets, including MNIST, CIFAR-10, and ImageNet. It is shown that DeepPAC outperforms the state-of-the-art statistical method PROVERO, and it achieves more practical robustness analysis than the formal verification tool ERAN. Also, its results are consistent with existing DNN testing work like DeepGini.



## **41. Stealing Malware Classifiers and AVs at Low False Positive Conditions**

cs.CR

12 pages, 8 figures, 6 tables. Under review

**SubmitDate**: 2022-04-13    [paper-pdf](http://arxiv.org/pdf/2204.06241v1)

**Authors**: Maria Rigaki, Sebastian Garcia

**Abstracts**: Model stealing attacks have been successfully used in many machine learning domains, but there is little understanding of how these attacks work in the malware detection domain. Malware detection and, in general, security domains have very strong requirements of low false positive rates (FPR). However, these requirements are not the primary focus of the existing model stealing literature. Stealing attacks create surrogate models that perform similarly to a target model using a limited amount of queries to the target. The first stage of this study is the evaluation of active learning model stealing attacks against publicly available stand-alone machine learning malware classifiers and antivirus products (AVs). We propose a new neural network architecture for surrogate models that outperforms the existing state of the art on low FPR conditions. The surrogates were evaluated on their agreement with the targeted models. Good surrogates of the stand-alone classifiers were created with up to 99% agreement with the target models, using less than 4% of the original training dataset size. Good AV surrogates were also possible to train, but with a lower agreement. The second stage used the best surrogates as well as the target models to generate adversarial malware using the MAB framework to test stand-alone models and AVs (offline and online). Results showed that surrogate models could generate adversarial samples that evade the targets but are less successful than the targets themselves. Using surrogates, however, is a necessity for attackers, given that attacks against AVs are extremely time-consuming and easily detected when the AVs are connected to the internet.



## **42. Liuer Mihou: A Practical Framework for Generating and Evaluating Grey-box Adversarial Attacks against NIDS**

cs.CR

16 pages, 8 figures, planning on submitting to ACM CCS 2022

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2204.06113v1)

**Authors**: Ke He, Dan Dongseong Kim, Jing Sun, Jeong Do Yoo, Young Hun Lee, Huy Kang Kim

**Abstracts**: Due to its high expressiveness and speed, Deep Learning (DL) has become an increasingly popular choice as the detection algorithm for Network-based Intrusion Detection Systems (NIDSes). Unfortunately, DL algorithms are vulnerable to adversarial examples that inject imperceptible modifications to the input and cause the DL algorithm to misclassify the input. Existing adversarial attacks in the NIDS domain often manipulate the traffic features directly, which hold no practical significance because traffic features cannot be replayed in a real network. It remains a research challenge to generate practical and evasive adversarial attacks.   This paper presents the Liuer Mihou attack that generates practical and replayable adversarial network packets that can bypass anomaly-based NIDS deployed in the Internet of Things (IoT) networks. The core idea behind Liuer Mihou is to exploit adversarial transferability and generate adversarial packets on a surrogate NIDS constrained by predefined mutation operations to ensure practicality. We objectively analyse the evasiveness of Liuer Mihou against four ML-based algorithms (LOF, OCSVM, RRCF, and SOM) and the state-of-the-art NIDS, Kitsune. From the results of our experiment, we gain valuable insights into necessary conditions on the adversarial transferability of anomaly detection algorithms. Going beyond a theoretical setting, we replay the adversarial attack in a real IoT testbed to examine the practicality of Liuer Mihou. Furthermore, we demonstrate that existing feature-level adversarial defence cannot defend against Liuer Mihou and constructively criticise the limitations of feature-level adversarial defences.



## **43. Optimal Membership Inference Bounds for Adaptive Composition of Sampled Gaussian Mechanisms**

cs.CR

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2204.06106v1)

**Authors**: Saeed Mahloujifar, Alexandre Sablayrolles, Graham Cormode, Somesh Jha

**Abstracts**: Given a trained model and a data sample, membership-inference (MI) attacks predict whether the sample was in the model's training set. A common countermeasure against MI attacks is to utilize differential privacy (DP) during model training to mask the presence of individual examples. While this use of DP is a principled approach to limit the efficacy of MI attacks, there is a gap between the bounds provided by DP and the empirical performance of MI attacks. In this paper, we derive bounds for the \textit{advantage} of an adversary mounting a MI attack, and demonstrate tightness for the widely-used Gaussian mechanism. We further show bounds on the \textit{confidence} of MI attacks. Our bounds are much stronger than those obtained by DP analysis. For example, analyzing a setting of DP-SGD with $\epsilon=4$ would obtain an upper bound on the advantage of $\approx0.36$ based on our analyses, while getting bound of $\approx 0.97$ using the analysis of previous work that convert $\epsilon$ to membership inference bounds.   Finally, using our analysis, we provide MI metrics for models trained on CIFAR10 dataset. To the best of our knowledge, our analysis provides the state-of-the-art membership inference bounds for the privacy.



## **44. Membership Inference Attacks From First Principles**

cs.CR

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2112.03570v2)

**Authors**: Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramer

**Abstracts**: A membership inference attack allows an adversary to query a trained machine learning model to predict whether or not a particular example was contained in the model's training dataset. These attacks are currently evaluated using average-case "accuracy" metrics that fail to characterize whether the attack can confidently identify any members of the training set. We argue that attacks should instead be evaluated by computing their true-positive rate at low (e.g., <0.1%) false-positive rates, and find most prior attacks perform poorly when evaluated in this way. To address this we develop a Likelihood Ratio Attack (LiRA) that carefully combines multiple ideas from the literature. Our attack is 10x more powerful at low false-positive rates, and also strictly dominates prior attacks on existing metrics.



## **45. Rate Coding or Direct Coding: Which One is Better for Accurate, Robust, and Energy-efficient Spiking Neural Networks?**

cs.NE

Accepted to ICASSP2022

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2202.03133v2)

**Authors**: Youngeun Kim, Hyoungseob Park, Abhishek Moitra, Abhiroop Bhattacharjee, Yeshwanth Venkatesha, Priyadarshini Panda

**Abstracts**: Recent Spiking Neural Networks (SNNs) works focus on an image classification task, therefore various coding techniques have been proposed to convert an image into temporal binary spikes. Among them, rate coding and direct coding are regarded as prospective candidates for building a practical SNN system as they show state-of-the-art performance on large-scale datasets. Despite their usage, there is little attention to comparing these two coding schemes in a fair manner. In this paper, we conduct a comprehensive analysis of the two codings from three perspectives: accuracy, adversarial robustness, and energy-efficiency. First, we compare the performance of two coding techniques with various architectures and datasets. Then, we measure the robustness of the coding techniques on two adversarial attack methods. Finally, we compare the energy-efficiency of two coding schemes on a digital hardware platform. Our results show that direct coding can achieve better accuracy especially for a small number of timesteps. In contrast, rate coding shows better robustness to adversarial attacks owing to the non-differentiable spike generation process. Rate coding also yields higher energy-efficiency than direct coding which requires multi-bit precision for the first layer. Our study explores the characteristics of two codings, which is an important design consideration for building SNNs. The code is made available at https://github.com/Intelligent-Computing-Lab-Yale/Rate-vs-Direct.



## **46. Masked Faces with Faced Masks**

cs.CV

8 pages

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2201.06427v2)

**Authors**: Jiayi Zhu, Qing Guo, Felix Juefei-Xu, Yihao Huang, Yang Liu, Geguang Pu

**Abstracts**: Modern face recognition systems (FRS) still fall short when the subjects are wearing facial masks, a common theme in the age of respiratory pandemics. An intuitive partial remedy is to add a mask detector to flag any masked faces so that the FRS can act accordingly for those low-confidence masked faces. In this work, we set out to investigate the potential vulnerability of such FRS equipped with a mask detector, on large-scale masked faces, which might trigger a serious risk, e.g., letting a suspect evade the FRS where both facial identity and mask are undetected. As existing face recognizers and mask detectors have high performance in their respective tasks, it is significantly challenging to simultaneously fool them and preserve the transferability of the attack. We formulate the new task as the generation of realistic & adversarial-faced mask and make three main contributions: First, we study the naive Delanunay-based masking method (DM) to simulate the process of wearing a faced mask that is cropped from a template image, which reveals the main challenges of this new task. Second, we further equip the DM with the adversarial noise attack and propose the adversarial noise Delaunay-based masking method (AdvNoise-DM) that can fool the face recognition and mask detection effectively but make the face less natural. Third, we propose the adversarial filtering Delaunay-based masking method denoted as MF2M by employing the adversarial filtering for AdvNoise-DM and obtain more natural faces. With the above efforts, the final version not only leads to significant performance deterioration of the state-of-the-art (SOTA) deep learning-based FRS, but also remains undetected by the SOTA facial mask detector, thus successfully fooling both systems at the same time.



## **47. Automated Attacker Synthesis for Distributed Protocols**

cs.CR

24 pages, 15 figures

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2004.01220v4)

**Authors**: Max von Hippel, Cole Vick, Stavros Tripakis, Cristina Nita-Rotaru

**Abstracts**: Distributed protocols should be robust to both benign malfunction (e.g. packet loss or delay) and attacks (e.g. message replay) from internal or external adversaries. In this paper we take a formal approach to the automated synthesis of attackers, i.e. adversarial processes that can cause the protocol to malfunction. Specifically, given a formal threat model capturing the distributed protocol model and network topology, as well as the placement, goals, and interface (inputs and outputs) of potential attackers, we automatically synthesize an attacker. We formalize four attacker synthesis problems - across attackers that always succeed versus those that sometimes fail, and attackers that attack forever versus those that do not - and we propose algorithmic solutions to two of them. We report on a prototype implementation called KORG and its application to TCP as a case-study. Our experiments show that KORG can automatically generate well-known attacks for TCP within seconds or minutes.



## **48. Catch Me If You Can: Blackbox Adversarial Attacks on Automatic Speech Recognition using Frequency Masking**

cs.SD

11 pages, 7 figures and 3 tables

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2112.01821v2)

**Authors**: Xiaoliang Wu, Ajitha Rajan

**Abstracts**: Automatic speech recognition (ASR) models are prevalent, particularly in applications for voice navigation and voice control of domestic appliances. The computational core of ASRs are deep neural networks (DNNs) that have been shown to be susceptible to adversarial perturbations; easily misused by attackers to generate malicious outputs. To help test the security and robustnesss of ASRS, we propose techniques that generate blackbox (agnostic to the DNN), untargeted adversarial attacks that are portable across ASRs. This is in contrast to existing work that focuses on whitebox targeted attacks that are time consuming and lack portability.   Our techniques generate adversarial attacks that have no human audible difference by manipulating the audio signal using a psychoacoustic model that maintains the audio perturbations below the thresholds of human perception. We evaluate portability and effectiveness of our techniques using three popular ASRs and two input audio datasets using the metrics - Word Error Rate (WER) of output transcription, Similarity to original audio, attack Success Rate on different ASRs and Detection score by a defense system. We found our adversarial attacks were portable across ASRs, not easily detected by a state-of-the-art defense system, and had significant difference in output transcriptions while sounding similar to original audio.



## **49. Staircase Sign Method for Boosting Adversarial Attacks**

cs.CV

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2104.09722v2)

**Authors**: Qilong Zhang, Xiaosu Zhu, Jingkuan Song, Lianli Gao, Heng Tao Shen

**Abstracts**: Crafting adversarial examples for the transfer-based attack is challenging and remains a research hot spot. Currently, such attack methods are based on the hypothesis that the substitute model and the victim model learn similar decision boundaries, and they conventionally apply Sign Method (SM) to manipulate the gradient as the resultant perturbation. Although SM is efficient, it only extracts the sign of gradient units but ignores their value difference, which inevitably leads to a deviation. Therefore, we propose a novel Staircase Sign Method (S$^2$M) to alleviate this issue, thus boosting attacks. Technically, our method heuristically divides the gradient sign into several segments according to the values of the gradient units, and then assigns each segment with a staircase weight for better crafting adversarial perturbation. As a result, our adversarial examples perform better in both white-box and black-box manner without being more visible. Since S$^2$M just manipulates the resultant gradient, our method can be generally integrated into the family of FGSM algorithms, and the computational overhead is negligible. Extensive experiments on the ImageNet dataset demonstrate the effectiveness of our proposed methods, which significantly improve the transferability (i.e., on average, \textbf{5.1\%} for normally trained models and \textbf{12.8\%} for adversarially trained defenses). Our code is available at \url{https://github.com/qilong-zhang/Staircase-sign-method}.



## **50. A survey in Adversarial Defences and Robustness in NLP**

cs.CL

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2203.06414v2)

**Authors**: Shreya Goyal, Sumanth Doddapaneni, Mitesh M. Khapra, Balaraman Ravindran

**Abstracts**: In recent years, it has been seen that deep neural networks are lacking robustness and are likely to break in case of adversarial perturbations in input data. Strong adversarial attacks are proposed by various authors for computer vision and Natural Language Processing (NLP). As a counter-effort, several defense mechanisms are also proposed to save these networks from failing. In contrast with image data, generating adversarial attacks and defending these models is not easy in NLP because of the discrete nature of the text data. However, numerous methods for adversarial defense are proposed of late, for different NLP tasks such as text classification, named entity recognition, natural language inferencing, etc. These methods are not just used for defending neural networks from adversarial attacks, but also used as a regularization mechanism during training, saving the model from overfitting. The proposed survey is an attempt to review different methods proposed for adversarial defenses in NLP in the recent past by proposing a novel taxonomy. This survey also highlights the fragility of the advanced deep neural networks in NLP and the challenges in defending them.



