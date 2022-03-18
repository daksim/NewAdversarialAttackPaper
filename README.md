# Latest Adversarial Attack Papers
**update at 2022-03-19 06:32:07**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Bayesian Framework for Gradient Leakage**

cs.LG

**SubmitDate**: 2022-03-17    [paper-pdf](http://arxiv.org/pdf/2111.04706v2)

**Authors**: Mislav Balunović, Dimitar I. Dimitrov, Robin Staab, Martin Vechev

**Abstracts**: Federated learning is an established method for training machine learning models without sharing training data. However, recent work has shown that it cannot guarantee data privacy as shared gradients can still leak sensitive information. To formalize the problem of gradient leakage, we propose a theoretical framework that enables, for the first time, analysis of the Bayes optimal adversary phrased as an optimization problem. We demonstrate that existing leakage attacks can be seen as approximations of this optimal adversary with different assumptions on the probability distributions of the input data and gradients. Our experiments confirm the effectiveness of the Bayes optimal adversary when it has knowledge of the underlying distribution. Further, our experimental evaluation shows that several existing heuristic defenses are not effective against stronger attacks, especially early in the training process. Thus, our findings indicate that the construction of more effective defenses and their evaluation remains an open problem.



## **2. SPAA: Stealthy Projector-based Adversarial Attacks on Deep Image Classifiers**

cs.CV

**SubmitDate**: 2022-03-17    [paper-pdf](http://arxiv.org/pdf/2012.05858v3)

**Authors**: Bingyao Huang, Haibin Ling

**Abstracts**: Light-based adversarial attacks use spatial augmented reality (SAR) techniques to fool image classifiers by altering the physical light condition with a controllable light source, e.g., a projector. Compared with physical attacks that place hand-crafted adversarial objects, projector-based ones obviate modifying the physical entities, and can be performed transiently and dynamically by altering the projection pattern. However, subtle light perturbations are insufficient to fool image classifiers, due to the complex environment and project-and-capture process. Thus, existing approaches focus on projecting clearly perceptible adversarial patterns, while the more interesting yet challenging goal, stealthy projector-based attack, remains open. In this paper, for the first time, we formulate this problem as an end-to-end differentiable process and propose a Stealthy Projector-based Adversarial Attack (SPAA) solution. In SPAA, we approximate the real Project-and-Capture process using a deep neural network named PCNet, then we include PCNet in the optimization of projector-based attacks such that the generated adversarial projection is physically plausible. Finally, to generate both robust and stealthy adversarial projections, we propose an algorithm that uses minimum perturbation and adversarial confidence thresholds to alternate between the adversarial loss and stealthiness loss optimization. Our experimental evaluations show that SPAA clearly outperforms other methods by achieving higher attack success rates and meanwhile being stealthier, for both targeted and untargeted attacks.



## **3. PiDAn: A Coherence Optimization Approach for Backdoor Attack Detection and Mitigation in Deep Neural Networks**

cs.LG

**SubmitDate**: 2022-03-17    [paper-pdf](http://arxiv.org/pdf/2203.09289v1)

**Authors**: Yue Wang, Wenqing Li, Esha Sarkar, Muhammad Shafique, Michail Maniatakos, Saif Eddin Jabari

**Abstracts**: Backdoor attacks impose a new threat in Deep Neural Networks (DNNs), where a backdoor is inserted into the neural network by poisoning the training dataset, misclassifying inputs that contain the adversary trigger. The major challenge for defending against these attacks is that only the attacker knows the secret trigger and the target class. The problem is further exacerbated by the recent introduction of "Hidden Triggers", where the triggers are carefully fused into the input, bypassing detection by human inspection and causing backdoor identification through anomaly detection to fail. To defend against such imperceptible attacks, in this work we systematically analyze how representations, i.e., the set of neuron activations for a given DNN when using the training data as inputs, are affected by backdoor attacks. We propose PiDAn, an algorithm based on coherence optimization purifying the poisoned data. Our analysis shows that representations of poisoned data and authentic data in the target class are still embedded in different linear subspaces, which implies that they show different coherence with some latent spaces. Based on this observation, the proposed PiDAn algorithm learns a sample-wise weight vector to maximize the projected coherence of weighted samples, where we demonstrate that the learned weight vector has a natural "grouping effect" and is distinguishable between authentic data and poisoned data. This enables the systematic detection and mitigation of backdoor attacks. Based on our theoretical analysis and experimental results, we demonstrate the effectiveness of PiDAn in defending against backdoor attacks that use different settings of poisoned samples on GTSRB and ILSVRC2012 datasets. Our PiDAn algorithm can detect more than 90% infected classes and identify 95% poisoned samples.



## **4. On the Properties of Adversarially-Trained CNNs**

cs.CV

**SubmitDate**: 2022-03-17    [paper-pdf](http://arxiv.org/pdf/2203.09243v1)

**Authors**: Mattia Carletti, Matteo Terzi, Gian Antonio Susto

**Abstracts**: Adversarial Training has proved to be an effective training paradigm to enforce robustness against adversarial examples in modern neural network architectures. Despite many efforts, explanations of the foundational principles underpinning the effectiveness of Adversarial Training are limited and far from being widely accepted by the Deep Learning community. In this paper, we describe surprising properties of adversarially-trained models, shedding light on mechanisms through which robustness against adversarial attacks is implemented. Moreover, we highlight limitations and failure modes affecting these models that were not discussed by prior works. We conduct extensive analyses on a wide range of architectures and datasets, performing a deep comparison between robust and natural models.



## **5. Improving the Transferability of Targeted Adversarial Examples through Object-Based Diverse Input**

cs.CV

Accepted at CVPR 2022

**SubmitDate**: 2022-03-17    [paper-pdf](http://arxiv.org/pdf/2203.09123v1)

**Authors**: Junyoung Byun, Seungju Cho, Myung-Joon Kwon, Hee-Seon Kim, Changick Kim

**Abstracts**: The transferability of adversarial examples allows the deception on black-box models, and transfer-based targeted attacks have attracted a lot of interest due to their practical applicability. To maximize the transfer success rate, adversarial examples should avoid overfitting to the source model, and image augmentation is one of the primary approaches for this. However, prior works utilize simple image transformations such as resizing, which limits input diversity. To tackle this limitation, we propose the object-based diverse input (ODI) method that draws an adversarial image on a 3D object and induces the rendered image to be classified as the target class. Our motivation comes from the humans' superior perception of an image printed on a 3D object. If the image is clear enough, humans can recognize the image content in a variety of viewing conditions. Likewise, if an adversarial example looks like the target class to the model, the model should also classify the rendered image of the 3D object as the target class. The ODI method effectively diversifies the input by leveraging an ensemble of multiple source objects and randomizing viewing conditions. In our experimental results on the ImageNet-Compatible dataset, this method boosts the average targeted attack success rate from 28.3% to 47.0% compared to the state-of-the-art methods. We also demonstrate the applicability of the ODI method to adversarial examples on the face verification task and its superior performance improvement. Our code is available at https://github.com/dreamflake/ODI.



## **6. Probabilistic Margins for Instance Reweighting in Adversarial Training**

cs.LG

17 pages, 4 figures

**SubmitDate**: 2022-03-17    [paper-pdf](http://arxiv.org/pdf/2106.07904v2)

**Authors**: Qizhou Wang, Feng Liu, Bo Han, Tongliang Liu, Chen Gong, Gang Niu, Mingyuan Zhou, Masashi Sugiyama

**Abstracts**: Reweighting adversarial data during training has been recently shown to improve adversarial robustness, where data closer to the current decision boundaries are regarded as more critical and given larger weights. However, existing methods measuring the closeness are not very reliable: they are discrete and can take only a few values, and they are path-dependent, i.e., they may change given the same start and end points with different attack paths. In this paper, we propose three types of probabilistic margin (PM), which are continuous and path-independent, for measuring the aforementioned closeness and reweighting adversarial data. Specifically, a PM is defined as the difference between two estimated class-posterior probabilities, e.g., such the probability of the true label minus the probability of the most confusing label given some natural data. Though different PMs capture different geometric properties, all three PMs share a negative correlation with the vulnerability of data: data with larger/smaller PMs are safer/riskier and should have smaller/larger weights. Experiments demonstrate that PMs are reliable measurements and PM-based reweighting methods outperform state-of-the-art methods.



## **7. BLOWN: A Blockchain Protocol for Single-Hop Wireless Networks under Adversarial SINR**

cs.CR

18 pages, 11 figures, journal paper

**SubmitDate**: 2022-03-17    [paper-pdf](http://arxiv.org/pdf/2103.08361v3)

**Authors**: Minghui Xu, Feng Zhao, Yifei Zou, Chunchi Liu, Xiuzhen Cheng, Falko Dressler

**Abstracts**: Known as a distributed ledger technology (DLT), blockchain has attracted much attention due to its properties such as decentralization, security, immutability and transparency, and its potential of servicing as an infrastructure for various applications. Blockchain can empower wireless networks with identity management, data integrity, access control, and high-level security. However, previous studies on blockchain-enabled wireless networks mostly focus on proposing architectures or building systems with popular blockchain protocols. Nevertheless, such existing protocols have obvious shortcomings when adopted in wireless networks where nodes may have limited physical resources, may fall short of well-established reliable channels, or may suffer from variable bandwidths impacted by environments or jamming attacks. In this paper, we propose a novel consensus protocol named Proof-of-Channel (PoC) leveraging the natural properties of wireless communications, and develop a permissioned BLOWN protocol (BLOckchain protocol for Wireless Networks) for single-hop wireless networks under an adversarial SINR model. We formalize BLOWN with the universal composition framework and prove its security properties, namely persistence and liveness, as well as its strengths in countering against adversarial jamming, double-spending, and Sybil attacks, which are also demonstrated by extensive simulation studies.



## **8. Robustness through Cognitive Dissociation Mitigation in Contrastive Adversarial Training**

cs.LG

**SubmitDate**: 2022-03-16    [paper-pdf](http://arxiv.org/pdf/2203.08959v1)

**Authors**: Adir Rahamim, Itay Naeh

**Abstracts**: In this paper, we introduce a novel neural network training framework that increases model's adversarial robustness to adversarial attacks while maintaining high clean accuracy by combining contrastive learning (CL) with adversarial training (AT). We propose to improve model robustness to adversarial attacks by learning feature representations that are consistent under both data augmentations and adversarial perturbations. We leverage contrastive learning to improve adversarial robustness by considering an adversarial example as another positive example, and aim to maximize the similarity between random augmentations of data samples and their adversarial example, while constantly updating the classification head in order to avoid a cognitive dissociation between the classification head and the embedding space. This dissociation is caused by the fact that CL updates the network up to the embedding space, while freezing the classification head which is used to generate new positive adversarial examples. We validate our method, Contrastive Learning with Adversarial Features(CLAF), on the CIFAR-10 dataset on which it outperforms both robust accuracy and clean accuracy over alternative supervised and self-supervised adversarial learning methods.



## **9. Provable Adversarial Robustness for Fractional Lp Threat Models**

cs.LG

AISTATS 2022 accepted paper

**SubmitDate**: 2022-03-16    [paper-pdf](http://arxiv.org/pdf/2203.08945v1)

**Authors**: Alexander Levine, Soheil Feizi

**Abstracts**: In recent years, researchers have extensively studied adversarial robustness in a variety of threat models, including L_0, L_1, L_2, and L_infinity-norm bounded adversarial attacks. However, attacks bounded by fractional L_p "norms" (quasi-norms defined by the L_p distance with 0<p<1) have yet to be thoroughly considered. We proactively propose a defense with several desirable properties: it provides provable (certified) robustness, scales to ImageNet, and yields deterministic (rather than high-probability) certified guarantees when applied to quantized data (e.g., images). Our technique for fractional L_p robustness constructs expressive, deep classifiers that are globally Lipschitz with respect to the L_p^p metric, for any 0<p<1. However, our method is even more general: we can construct classifiers which are globally Lipschitz with respect to any metric defined as the sum of concave functions of components. Our approach builds on a recent work, Levine and Feizi (2021), which provides a provable defense against L_1 attacks. However, we demonstrate that our proposed guarantees are highly non-vacuous, compared to the trivial solution of using (Levine and Feizi, 2021) directly and applying norm inequalities. Code is available at https://github.com/alevine0/fractionalLpRobustness.



## **10. Semantic-preserving Reinforcement Learning Attack Against Graph Neural Networks for Malware Detection**

cs.CR

**SubmitDate**: 2022-03-16    [paper-pdf](http://arxiv.org/pdf/2009.05602v3)

**Authors**: Lan Zhang, Peng Liu, Yoon-Ho Choi, Ping Chen

**Abstracts**: As an increasing number of deep-learning-based malware scanners have been proposed, the existing evasion techniques, including code obfuscation and polymorphic malware, are found to be less effective. In this work, we propose a reinforcement learning-based semantics-preserving (i.e.functionality-preserving) attack against black-box GNNs (GraphNeural Networks) for malware detection. The key factor of adversarial malware generation via semantic Nops insertion is to select the appropriate semanticNopsand their corresponding basic blocks. The proposed attack uses reinforcement learning to automatically make these "how to select" decisions. To evaluate the attack, we have trained two kinds of GNNs with five types(i.e., Backdoor, Trojan-Downloader, Trojan-Ransom, Adware, and Worm) of Windows malware samples and various benign Windows programs. The evaluation results have shown that the proposed attack can achieve a significantly higher evasion rate than three baseline attacks, namely the semantics-preserving random instruction insertion attack, the semantics-preserving accumulative instruction insertion attack, and the semantics-preserving gradient-based instruction insertion attack.



## **11. Attacking deep networks with surrogate-based adversarial black-box methods is easy**

cs.LG

ICLR 2022

**SubmitDate**: 2022-03-16    [paper-pdf](http://arxiv.org/pdf/2203.08725v1)

**Authors**: Nicholas A. Lord, Romain Mueller, Luca Bertinetto

**Abstracts**: A recent line of work on black-box adversarial attacks has revived the use of transfer from surrogate models by integrating it into query-based search. However, we find that existing approaches of this type underperform their potential, and can be overly complicated besides. Here, we provide a short and simple algorithm which achieves state-of-the-art results through a search which uses the surrogate network's class-score gradients, with no need for other priors or heuristics. The guiding assumption of the algorithm is that the studied networks are in a fundamental sense learning similar functions, and that a transfer attack from one to the other should thus be fairly "easy". This assumption is validated by the extremely low query counts and failure rates achieved: e.g. an untargeted attack on a VGG-16 ImageNet network using a ResNet-152 as the surrogate yields a median query count of 6 at a success rate of 99.9%. Code is available at https://github.com/fiveai/GFCS.



## **12. On the Security & Privacy in Federated Learning**

cs.CR

**SubmitDate**: 2022-03-16    [paper-pdf](http://arxiv.org/pdf/2112.05423v2)

**Authors**: Gorka Abad, Stjepan Picek, Víctor Julio Ramírez-Durán, Aitor Urbieta

**Abstracts**: Recent privacy awareness initiatives such as the EU General Data Protection Regulation subdued Machine Learning (ML) to privacy and security assessments. Federated Learning (FL) grants a privacy-driven, decentralized training scheme that improves ML models' security. The industry's fast-growing adaptation and security evaluations of FL technology exposed various vulnerabilities that threaten FL's confidentiality, integrity, or availability (CIA). This work assesses the CIA of FL by reviewing the state-of-the-art (SoTA) and creating a threat model that embraces the attack's surface, adversarial actors, capabilities, and goals. We propose the first unifying taxonomy for attacks and defenses and provide promising future research directions.



## **13. Towards Practical Certifiable Patch Defense with Vision Transformer**

cs.CV

Accepted by CVPR2022

**SubmitDate**: 2022-03-16    [paper-pdf](http://arxiv.org/pdf/2203.08519v1)

**Authors**: Zhaoyu Chen, Bo Li, Jianghe Xu, Shuang Wu, Shouhong Ding, Wenqiang Zhang

**Abstracts**: Patch attacks, one of the most threatening forms of physical attack in adversarial examples, can lead networks to induce misclassification by modifying pixels arbitrarily in a continuous region. Certifiable patch defense can guarantee robustness that the classifier is not affected by patch attacks. Existing certifiable patch defenses sacrifice the clean accuracy of classifiers and only obtain a low certified accuracy on toy datasets. Furthermore, the clean and certified accuracy of these methods is still significantly lower than the accuracy of normal classification networks, which limits their application in practice. To move towards a practical certifiable patch defense, we introduce Vision Transformer (ViT) into the framework of Derandomized Smoothing (DS). Specifically, we propose a progressive smoothed image modeling task to train Vision Transformer, which can capture the more discriminable local context of an image while preserving the global semantic information. For efficient inference and deployment in the real world, we innovatively reconstruct the global self-attention structure of the original ViT into isolated band unit self-attention. On ImageNet, under 2% area patch attacks our method achieves 41.70% certified accuracy, a nearly 1-fold increase over the previous best method (26.00%). Simultaneously, our method achieves 78.58% clean accuracy, which is quite close to the normal ResNet-101 accuracy. Extensive experiments show that our method obtains state-of-the-art clean and certified accuracy with inferring efficiently on CIFAR-10 and ImageNet.



## **14. SHIELD: Defending Textual Neural Networks against Multiple Black-Box Adversarial Attacks with Stochastic Multi-Expert Patcher**

cs.LG

Accepted to the 60th Annual Meeting of the Association for  Computational Linguistics (ACL'22)

**SubmitDate**: 2022-03-16    [paper-pdf](http://arxiv.org/pdf/2011.08908v2)

**Authors**: Thai Le, Noseong Park, Dongwon Lee

**Abstracts**: Even though several methods have proposed to defend textual neural network (NN) models against black-box adversarial attacks, they often defend against a specific text perturbation strategy and/or require re-training the models from scratch. This leads to a lack of generalization in practice and redundant computation. In particular, the state-of-the-art transformer models (e.g., BERT, RoBERTa) require great time and computation resources. By borrowing an idea from software engineering, in order to address these limitations, we propose a novel algorithm, SHIELD, which modifies and re-trains only the last layer of a textual NN, and thus it "patches" and "transforms" the NN into a stochastic weighted ensemble of multi-expert prediction heads. Considering that most of current black-box attacks rely on iterative search mechanisms to optimize their adversarial perturbations, SHIELD confuses the attackers by automatically utilizing different weighted ensembles of predictors depending on the input. In other words, SHIELD breaks a fundamental assumption of the attack, which is a victim NN model remains constant during an attack. By conducting comprehensive experiments, we demonstrate that all of CNN, RNN, BERT, and RoBERTa-based textual NNs, once patched by SHIELD, exhibit a relative enhancement of 15%--70% in accuracy on average against 14 different black-box attacks, outperforming 6 defensive baselines across 3 public datasets. All codes are to be released.



## **15. CROP: Certifying Robust Policies for Reinforcement Learning through Functional Smoothing**

cs.LG

Published as a conference paper at ICLR 2022

**SubmitDate**: 2022-03-16    [paper-pdf](http://arxiv.org/pdf/2106.09292v2)

**Authors**: Fan Wu, Linyi Li, Zijian Huang, Yevgeniy Vorobeychik, Ding Zhao, Bo Li

**Abstracts**: As reinforcement learning (RL) has achieved great success and been even adopted in safety-critical domains such as autonomous vehicles, a range of empirical studies have been conducted to improve its robustness against adversarial attacks. However, how to certify its robustness with theoretical guarantees still remains challenging. In this paper, we present the first unified framework CROP (Certifying Robust Policies for RL) to provide robustness certification on both action and reward levels. In particular, we propose two robustness certification criteria: robustness of per-state actions and lower bound of cumulative rewards. We then develop a local smoothing algorithm for policies derived from Q-functions to guarantee the robustness of actions taken along the trajectory; we also develop a global smoothing algorithm for certifying the lower bound of a finite-horizon cumulative reward, as well as a novel local smoothing algorithm to perform adaptive search in order to obtain tighter reward certification. Empirically, we apply CROP to evaluate several existing empirically robust RL algorithms, including adversarial training and different robust regularization, in four environments (two representative Atari games, Highway, and CartPole). Furthermore, by evaluating these algorithms against adversarial attacks, we demonstrate that our certification are often tight. All experiment results are available at website https://crop-leaderboard.github.io.



## **16. Patch-Fool: Are Vision Transformers Always Robust Against Adversarial Perturbations?**

cs.CV

Accepted at ICLR 2022

**SubmitDate**: 2022-03-16    [paper-pdf](http://arxiv.org/pdf/2203.08392v1)

**Authors**: Yonggan Fu, Shunyao Zhang, Shang Wu, Cheng Wan, Yingyan Lin

**Abstracts**: Vision transformers (ViTs) have recently set off a new wave in neural architecture design thanks to their record-breaking performance in various vision tasks. In parallel, to fulfill the goal of deploying ViTs into real-world vision applications, their robustness against potential malicious attacks has gained increasing attention. In particular, recent works show that ViTs are more robust against adversarial attacks as compared with convolutional neural networks (CNNs), and conjecture that this is because ViTs focus more on capturing global interactions among different input/feature patches, leading to their improved robustness to local perturbations imposed by adversarial attacks. In this work, we ask an intriguing question: "Under what kinds of perturbations do ViTs become more vulnerable learners compared to CNNs?" Driven by this question, we first conduct a comprehensive experiment regarding the robustness of both ViTs and CNNs under various existing adversarial attacks to understand the underlying reason favoring their robustness. Based on the drawn insights, we then propose a dedicated attack framework, dubbed Patch-Fool, that fools the self-attention mechanism by attacking its basic component (i.e., a single patch) with a series of attention-aware optimization techniques. Interestingly, our Patch-Fool framework shows for the first time that ViTs are not necessarily more robust than CNNs against adversarial perturbations. In particular, we find that ViTs are more vulnerable learners compared with CNNs against our Patch-Fool attack which is consistent across extensive experiments, and the observations from Sparse/Mild Patch-Fool, two variants of Patch-Fool, indicate an intriguing insight that the perturbation density and strength on each patch seem to be the key factors that influence the robustness ranking between ViTs and CNNs.



## **17. Synthesis of the Supremal Covert Attacker Against Unknown Supervisors by Using Observations**

eess.SY

**SubmitDate**: 2022-03-16    [paper-pdf](http://arxiv.org/pdf/2203.08360v1)

**Authors**: Ruochen Tai, Liyong Lin, Yuting Zhu, Rong Su

**Abstracts**: In this paper, we consider the problem of synthesizing the supremal covert damage-reachable attacker under the normality assumption, in the setup where the model of the supervisor is unknown to the adversary but the adversary has recorded a (prefix-closed) finite set of observations of the runs of the closed-loop system. The synthesized attacker needs to ensure both the damage-reachability and the covertness against all the supervisors which are consistent with the given set of observations. There is a gap between the de facto supremality, assuming the model of the supervisor is known, and the supremality that can be attained with a limited knowledge of the model of the supervisor, from the adversary's point of view. We consider the setup where the attacker can exercise sensor replacement/deletion attacks and actuator enablement/disablement attacks. The solution methodology proposed in this work is to reduce the synthesis of the supremal covert damage-reachable attacker, given the model of the plant and the finite set of observations, to the synthesis of the supremal safe supervisor for certain transformed plant, which shows the decidability of the observation-assisted covert attacker synthesis problem. The effectiveness of our approach is illustrated on a water tank example adapted from the literature.



## **18. Improving Question Answering Model Robustness with Synthetic Adversarial Data Generation**

cs.CL

EMNLP 2021

**SubmitDate**: 2022-03-15    [paper-pdf](http://arxiv.org/pdf/2104.08678v3)

**Authors**: Max Bartolo, Tristan Thrush, Robin Jia, Sebastian Riedel, Pontus Stenetorp, Douwe Kiela

**Abstracts**: Despite recent progress, state-of-the-art question answering models remain vulnerable to a variety of adversarial attacks. While dynamic adversarial data collection, in which a human annotator tries to write examples that fool a model-in-the-loop, can improve model robustness, this process is expensive which limits the scale of the collected data. In this work, we are the first to use synthetic adversarial data generation to make question answering models more robust to human adversaries. We develop a data generation pipeline that selects source passages, identifies candidate answers, generates questions, then finally filters or re-labels them to improve quality. Using this approach, we amplify a smaller human-written adversarial dataset to a much larger set of synthetic question-answer pairs. By incorporating our synthetic data, we improve the state-of-the-art on the AdversarialQA dataset by 3.7F1 and improve model generalisation on nine of the twelve MRQA datasets. We further conduct a novel human-in-the-loop evaluation to show that our models are considerably more robust to new human-written adversarial examples: crowdworkers can fool our model only 8.8% of the time on average, compared to 17.6% for a model trained without synthetic data.



## **19. Knowledge Enhanced Machine Learning Pipeline against Diverse Adversarial Attacks**

cs.LG

International Conference on Machine Learning 2021, 37 pages, 8  figures, 9 tables

**SubmitDate**: 2022-03-15    [paper-pdf](http://arxiv.org/pdf/2106.06235v2)

**Authors**: Nezihe Merve Gürel, Xiangyu Qi, Luka Rimanic, Ce Zhang, Bo Li

**Abstracts**: Despite the great successes achieved by deep neural networks (DNNs), recent studies show that they are vulnerable against adversarial examples, which aim to mislead DNNs by adding small adversarial perturbations. Several defenses have been proposed against such attacks, while many of them have been adaptively attacked. In this work, we aim to enhance the ML robustness from a different perspective by leveraging domain knowledge: We propose a Knowledge Enhanced Machine Learning Pipeline (KEMLP) to integrate domain knowledge (i.e., logic relationships among different predictions) into a probabilistic graphical model via first-order logic rules. In particular, we develop KEMLP by integrating a diverse set of weak auxiliary models based on their logical relationships to the main DNN model that performs the target task. Theoretically, we provide convergence results and prove that, under mild conditions, the prediction of KEMLP is more robust than that of the main DNN model. Empirically, we take road sign recognition as an example and leverage the relationships between road signs and their shapes and contents as domain knowledge. We show that compared with adversarial training and other baselines, KEMLP achieves higher robustness against physical attacks, $\mathcal{L}_p$ bounded attacks, unforeseen attacks, and natural corruptions under both whitebox and blackbox settings, while still maintaining high clean accuracy.



## **20. Frequency-driven Imperceptible Adversarial Attack on Semantic Similarity**

cs.CV

10 pages, 7 figure, CVPR 2022 conference

**SubmitDate**: 2022-03-15    [paper-pdf](http://arxiv.org/pdf/2203.05151v2)

**Authors**: Cheng Luo, Qinliang Lin, Weicheng Xie, Bizhu Wu, Jinheng Xie, Linlin Shen

**Abstracts**: Current adversarial attack research reveals the vulnerability of learning-based classifiers against carefully crafted perturbations. However, most existing attack methods have inherent limitations in cross-dataset generalization as they rely on a classification layer with a closed set of categories. Furthermore, the perturbations generated by these methods may appear in regions easily perceptible to the human visual system (HVS). To circumvent the former problem, we propose a novel algorithm that attacks semantic similarity on feature representations. In this way, we are able to fool classifiers without limiting attacks to a specific dataset. For imperceptibility, we introduce the low-frequency constraint to limit perturbations within high-frequency components, ensuring perceptual similarity between adversarial examples and originals. Extensive experiments on three datasets (CIFAR-10, CIFAR-100, and ImageNet-1K) and three public online platforms indicate that our attack can yield misleading and transferable adversarial examples across architectures and datasets. Additionally, visualization results and quantitative performance (in terms of four different metrics) show that the proposed algorithm generates more imperceptible perturbations than the state-of-the-art methods. Code is made available at.



## **21. Towards Adversarial Control Loops in Sensor Attacks: A Case Study to Control the Kinematics and Actuation of Embedded Systems**

cs.CR

**SubmitDate**: 2022-03-15    [paper-pdf](http://arxiv.org/pdf/2203.07670v1)

**Authors**: Yazhou Tu, Sara Rampazzi, Xiali Hei

**Abstracts**: Recent works investigated attacks on sensors by influencing analog sensor components with acoustic, light, and electromagnetic signals. Such attacks can have extensive security, reliability, and safety implications since many types of the targeted sensors are also widely used in critical process control, robotics, automation, and industrial control systems. While existing works advanced our understanding of the physical-level risks that are hidden from a digital-domain perspective, gaps exist in how the attack can be guided to achieve system-level control in real-time, continuous processes. This paper proposes an adversarial control loop-based approach for real-time attacks on control systems relying on sensors. We study how to utilize the system feedback extracted from physical-domain signals to guide the attacks. In the attack process, injection signals are adjusted in real time based on the extracted feedback to exert targeted influence on a victim control system that is continuously affected by the injected perturbations and applying changes to the physical environment. In our case study, we investigate how an external adversarial control system can be constructed over sensor-actuator systems and demonstrate the attacks with program-controlled processes to manipulate the victim system without accessing its internal statuses.



## **22. A Regularization Method to Improve Adversarial Robustness of Neural Networks for ECG Signal Classification**

cs.LG

This paper has been published by Computers in Biology and Medicine

**SubmitDate**: 2022-03-15    [paper-pdf](http://arxiv.org/pdf/2110.09759v2)

**Authors**: Linhai Ma, Liang Liang

**Abstracts**: Electrocardiogram (ECG) is the most widely used diagnostic tool to monitor the condition of the human heart. By using deep neural networks (DNNs), interpretation of ECG signals can be fully automated for the identification of potential abnormalities in a patient's heart in a fraction of a second. Studies have shown that given a sufficiently large amount of training data, DNN accuracy for ECG classification could reach human-expert cardiologist level. However, despite of the excellent performance in classification accuracy, DNNs are highly vulnerable to adversarial noises that are subtle changes in the input of a DNN and may lead to a wrong class-label prediction. It is challenging and essential to improve robustness of DNNs against adversarial noises, which are a threat to life-critical applications. In this work, we proposed a regularization method to improve DNN robustness from the perspective of noise-to-signal ratio (NSR) for the application of ECG signal classification. We evaluated our method on PhysioNet MIT-BIH dataset and CPSC2018 ECG dataset, and the results show that our method can substantially enhance DNN robustness against adversarial noises generated from adversarial attacks, with a minimal change in accuracy on clean data.



## **23. Semantically Distributed Robust Optimization for Vision-and-Language Inference**

cs.CV

Findings of ACL 2022; code available at  https://github.com/ASU-APG/VLI_SDRO

**SubmitDate**: 2022-03-14    [paper-pdf](http://arxiv.org/pdf/2110.07165v2)

**Authors**: Tejas Gokhale, Abhishek Chaudhary, Pratyay Banerjee, Chitta Baral, Yezhou Yang

**Abstracts**: Analysis of vision-and-language models has revealed their brittleness under linguistic phenomena such as paraphrasing, negation, textual entailment, and word substitutions with synonyms or antonyms. While data augmentation techniques have been designed to mitigate against these failure modes, methods that can integrate this knowledge into the training pipeline remain under-explored. In this paper, we present \textbf{SDRO}, a model-agnostic method that utilizes a set linguistic transformations in a distributed robust optimization setting, along with an ensembling technique to leverage these transformations during inference. Experiments on benchmark datasets with images (NLVR$^2$) and video (VIOLIN) demonstrate performance improvements as well as robustness to adversarial attacks. Experiments on binary VQA explore the generalizability of this method to other V\&L tasks.



## **24. RES-HD: Resilient Intelligent Fault Diagnosis Against Adversarial Attacks Using Hyper-Dimensional Computing**

cs.CR

**SubmitDate**: 2022-03-14    [paper-pdf](http://arxiv.org/pdf/2203.08148v1)

**Authors**: Onat Gungor, Tajana Rosing, Baris Aksanli

**Abstracts**: Industrial Internet of Things (I-IoT) enables fully automated production systems by continuously monitoring devices and analyzing collected data. Machine learning methods are commonly utilized for data analytics in such systems. Cyber-attacks are a grave threat to I-IoT as they can manipulate legitimate inputs, corrupting ML predictions and causing disruptions in the production systems. Hyper-dimensional computing (HDC) is a brain-inspired machine learning method that has been shown to be sufficiently accurate while being extremely robust, fast, and energy-efficient. In this work, we use HDC for intelligent fault diagnosis against different adversarial attacks. Our black-box adversarial attacks first train a substitute model and create perturbed test instances using this trained model. These examples are then transferred to the target models. The change in the classification accuracy is measured as the difference before and after the attacks. This change measures the resiliency of a learning method. Our experiments show that HDC leads to a more resilient and lightweight learning solution than the state-of-the-art deep learning methods. HDC has up to 67.5% higher resiliency compared to the state-of-the-art methods while being up to 25.1% faster to train.



## **25. Defending From Physically-Realizable Adversarial Attacks Through Internal Over-Activation Analysis**

cs.CV

**SubmitDate**: 2022-03-14    [paper-pdf](http://arxiv.org/pdf/2203.07341v1)

**Authors**: Giulio Rossolini, Federico Nesti, Fabio Brau, Alessandro Biondi, Giorgio Buttazzo

**Abstracts**: This work presents Z-Mask, a robust and effective strategy to improve the adversarial robustness of convolutional networks against physically-realizable adversarial attacks. The presented defense relies on specific Z-score analysis performed on the internal network features to detect and mask the pixels corresponding to adversarial objects in the input image. To this end, spatially contiguous activations are examined in shallow and deep layers to suggest potential adversarial regions. Such proposals are then aggregated through a multi-thresholding mechanism. The effectiveness of Z-Mask is evaluated with an extensive set of experiments carried out on models for both semantic segmentation and object detection. The evaluation is performed with both digital patches added to the input images and printed patches positioned in the real world. The obtained results confirm that Z-Mask outperforms the state-of-the-art methods in terms of both detection accuracy and overall performance of the networks under attack. Additional experiments showed that Z-Mask is also robust against possible defense-aware attacks.



## **26. MACER: Attack-free and Scalable Robust Training via Maximizing Certified Radius**

cs.LG

Published in ICLR 2020. 20 Pages

**SubmitDate**: 2022-03-14    [paper-pdf](http://arxiv.org/pdf/2001.02378v4)

**Authors**: Runtian Zhai, Chen Dan, Di He, Huan Zhang, Boqing Gong, Pradeep Ravikumar, Cho-Jui Hsieh, Liwei Wang

**Abstracts**: Adversarial training is one of the most popular ways to learn robust models but is usually attack-dependent and time costly. In this paper, we propose the MACER algorithm, which learns robust models without using adversarial training but performs better than all existing provable l2-defenses. Recent work shows that randomized smoothing can be used to provide a certified l2 radius to smoothed classifiers, and our algorithm trains provably robust smoothed classifiers via MAximizing the CErtified Radius (MACER). The attack-free characteristic makes MACER faster to train and easier to optimize. In our experiments, we show that our method can be applied to modern deep neural networks on a wide range of datasets, including Cifar-10, ImageNet, MNIST, and SVHN. For all tasks, MACER spends less training time than state-of-the-art adversarial training algorithms, and the learned models achieve larger average certified radius.



## **27. On the benefits of knowledge distillation for adversarial robustness**

cs.LG

**SubmitDate**: 2022-03-14    [paper-pdf](http://arxiv.org/pdf/2203.07159v1)

**Authors**: Javier Maroto, Guillermo Ortiz-Jiménez, Pascal Frossard

**Abstracts**: Knowledge distillation is normally used to compress a big network, or teacher, onto a smaller one, the student, by training it to match its outputs. Recently, some works have shown that robustness against adversarial attacks can also be distilled effectively to achieve good rates of robustness on mobile-friendly models. In this work, however, we take a different point of view, and show that knowledge distillation can be used directly to boost the performance of state-of-the-art models in adversarial robustness. In this sense, we present a thorough analysis and provide general guidelines to distill knowledge from a robust teacher and boost the clean and adversarial performance of a student model even further. To that end, we present Adversarial Knowledge Distillation (AKD), a new framework to improve a model's robust performance, consisting on adversarially training a student on a mixture of the original labels and the teacher outputs. Through carefully controlled ablation studies, we show that using early-stopping, model ensembles and weak adversarial training are key techniques to maximize performance of the student, and show that these insights generalize across different robust distillation techniques. Finally, we provide insights on the effect of robust knowledge distillation on the dynamics of the student network, and show that AKD mostly improves the calibration of the network and modify its training dynamics on samples that the model finds difficult to learn, or even memorize.



## **28. Detection of Electromagnetic Signal Injection Attacks on Actuator Systems**

cs.CR

**SubmitDate**: 2022-03-14    [paper-pdf](http://arxiv.org/pdf/2203.07102v1)

**Authors**: Youqian Zhang, Kasper Rasmussen

**Abstracts**: An actuator is a device that converts electricity into another form of energy, typically physical movement. They are absolutely essential for any system that needs to impact or modify the physical world, and are used in millions of systems of all sizes, all over the world, from cars and spacecraft to factory control systems and critical infrastructure. An actuator is a "dumb device" that is entirely controlled by the surrounding electronics, e.g., a microcontroller, and thus cannot authenticate its control signals or do any other form of processing. The problem we look at in this paper is how the wires that connect an actuator to its control electronics can act like antennas, picking up electromagnetic signals from the environment. This makes it possible for a remote attacker to wirelessly inject signals (energy) into these wires to bypass the controller and directly control the actuator.   To detect such attacks, we propose a novel detection method that allows the microcontroller to monitor the control signal and detect attacks as a deviation from the intended value. We have managed to do this without requiring the microcontroller to sample the signal at a high rate or run any signal processing. That makes our defense mechanism practical and easy to integrate into existing systems. Our method is general and applies to any type of actuator (provided a few basic assumptions are met), and can deal with adversaries with arbitrarily high transmission power. We implement our detection method on two different practical systems to show its generality, effectiveness, and robustness.



## **29. Beyond ImageNet Attack: Towards Crafting Adversarial Examples for Black-box Domains**

cs.CV

Accepted by ICLR 2022

**SubmitDate**: 2022-03-14    [paper-pdf](http://arxiv.org/pdf/2201.11528v4)

**Authors**: Qilong Zhang, Xiaodan Li, Yuefeng Chen, Jingkuan Song, Lianli Gao, Yuan He, Hui Xue

**Abstracts**: Adversarial examples have posed a severe threat to deep neural networks due to their transferable nature. Currently, various works have paid great efforts to enhance the cross-model transferability, which mostly assume the substitute model is trained in the same domain as the target model. However, in reality, the relevant information of the deployed model is unlikely to leak. Hence, it is vital to build a more practical black-box threat model to overcome this limitation and evaluate the vulnerability of deployed models. In this paper, with only the knowledge of the ImageNet domain, we propose a Beyond ImageNet Attack (BIA) to investigate the transferability towards black-box domains (unknown classification tasks). Specifically, we leverage a generative model to learn the adversarial function for disrupting low-level features of input images. Based on this framework, we further propose two variants to narrow the gap between the source and target domains from the data and model perspectives, respectively. Extensive experiments on coarse-grained and fine-grained domains demonstrate the effectiveness of our proposed methods. Notably, our methods outperform state-of-the-art approaches by up to 7.71\% (towards coarse-grained domains) and 25.91\% (towards fine-grained domains) on average. Our code is available at \url{https://github.com/qilong-zhang/Beyond-ImageNet-Attack}.



## **30. Data Poisoning Won't Save You From Facial Recognition**

cs.LG

ICLR 2022

**SubmitDate**: 2022-03-14    [paper-pdf](http://arxiv.org/pdf/2106.14851v2)

**Authors**: Evani Radiya-Dixit, Sanghyun Hong, Nicholas Carlini, Florian Tramèr

**Abstracts**: Data poisoning has been proposed as a compelling defense against facial recognition models trained on Web-scraped pictures. Users can perturb images they post online, so that models will misclassify future (unperturbed) pictures. We demonstrate that this strategy provides a false sense of security, as it ignores an inherent asymmetry between the parties: users' pictures are perturbed once and for all before being published (at which point they are scraped) and must thereafter fool all future models -- including models trained adaptively against the users' past attacks, or models that use technologies discovered after the attack. We evaluate two systems for poisoning attacks against large-scale facial recognition, Fawkes (500'000+ downloads) and LowKey. We demonstrate how an "oblivious" model trainer can simply wait for future developments in computer vision to nullify the protection of pictures collected in the past. We further show that an adversary with black-box access to the attack can (i) train a robust model that resists the perturbations of collected pictures and (ii) detect poisoned pictures uploaded online. We caution that facial recognition poisoning will not admit an "arms race" between attackers and defenders. Once perturbed pictures are scraped, the attack cannot be changed so any future successful defense irrevocably undermines users' privacy.



## **31. Efficient universal shuffle attack for visual object tracking**

cs.CV

accepted for ICASSP 2022

**SubmitDate**: 2022-03-14    [paper-pdf](http://arxiv.org/pdf/2203.06898v1)

**Authors**: Siao Liu, Zhaoyu Chen, Wei Li, Jiwei Zhu, Jiafeng Wang, Wenqiang Zhang, Zhongxue Gan

**Abstracts**: Recently, adversarial attacks have been applied in visual object tracking to deceive deep trackers by injecting imperceptible perturbations into video frames. However, previous work only generates the video-specific perturbations, which restricts its application scenarios. In addition, existing attacks are difficult to implement in reality due to the real-time of tracking and the re-initialization mechanism. To address these issues, we propose an offline universal adversarial attack called Efficient Universal Shuffle Attack. It takes only one perturbation to cause the tracker malfunction on all videos. To improve the computational efficiency and attack performance, we propose a greedy gradient strategy and a triple loss to efficiently capture and attack model-specific feature representations through the gradients. Experimental results show that EUSA can significantly reduce the performance of state-of-the-art trackers on OTB2015 and VOT2018.



## **32. Defending Against Adversarial Attack in ECG Classification with Adversarial Distillation Training**

eess.SP

**SubmitDate**: 2022-03-14    [paper-pdf](http://arxiv.org/pdf/2203.09487v1)

**Authors**: Jiahao Shao, Shijia Geng, Zhaoji Fu, Weilun Xu, Tong Liu, Shenda Hong

**Abstracts**: In clinics, doctors rely on electrocardiograms (ECGs) to assess severe cardiac disorders. Owing to the development of technology and the increase in health awareness, ECG signals are currently obtained by using medical and commercial devices. Deep neural networks (DNNs) can be used to analyze these signals because of their high accuracy rate. However, researchers have found that adversarial attacks can significantly reduce the accuracy of DNNs. Studies have been conducted to defend ECG-based DNNs against traditional adversarial attacks, such as projected gradient descent (PGD), and smooth adversarial perturbation (SAP) which targets ECG classification; however, to the best of our knowledge, no study has completely explored the defense against adversarial attacks targeting ECG classification. Thus, we did different experiments to explore the effects of defense methods against white-box adversarial attack and black-box adversarial attack targeting ECG classification, and we found that some common defense methods performed well against these attacks. Besides, we proposed a new defense method called Adversarial Distillation Training (ADT) which comes from defensive distillation and can effectively improve the generalization performance of DNNs. The results show that our method performed more effectively against adversarial attacks targeting on ECG classification than the other baseline methods, namely, adversarial training, defensive distillation, Jacob regularization, and noise-to-signal ratio regularization. Furthermore, we found that our method performed better against PGD attacks with low noise levels, which means that our method has stronger robustness.



## **33. Generating Practical Adversarial Network Traffic Flows Using NIDSGAN**

cs.CR

**SubmitDate**: 2022-03-13    [paper-pdf](http://arxiv.org/pdf/2203.06694v1)

**Authors**: Bolor-Erdene Zolbayar, Ryan Sheatsley, Patrick McDaniel, Michael J. Weisman, Sencun Zhu, Shitong Zhu, Srikanth Krishnamurthy

**Abstracts**: Network intrusion detection systems (NIDS) are an essential defense for computer networks and the hosts within them. Machine learning (ML) nowadays predominantly serves as the basis for NIDS decision making, where models are tuned to reduce false alarms, increase detection rates, and detect known and unknown attacks. At the same time, ML models have been found to be vulnerable to adversarial examples that undermine the downstream task. In this work, we ask the practical question of whether real-world ML-based NIDS can be circumvented by crafted adversarial flows, and if so, how can they be created. We develop the generative adversarial network (GAN)-based attack algorithm NIDSGAN and evaluate its effectiveness against realistic ML-based NIDS. Two main challenges arise for generating adversarial network traffic flows: (1) the network features must obey the constraints of the domain (i.e., represent realistic network behavior), and (2) the adversary must learn the decision behavior of the target NIDS without knowing its model internals (e.g., architecture and meta-parameters) and training data. Despite these challenges, the NIDSGAN algorithm generates highly realistic adversarial traffic flows that evade ML-based NIDS. We evaluate our attack algorithm against two state-of-the-art DNN-based NIDS in whitebox, blackbox, and restricted-blackbox threat models and achieve success rates which are on average 99%, 85%, and 70%, respectively. We also show that our attack algorithm can evade NIDS based on classical ML models including logistic regression, SVM, decision trees and KNNs, with a success rate of 70% on average. Our results demonstrate that deploying ML-based NIDS without careful defensive strategies against adversarial flows may (and arguably likely will) lead to future compromises.



## **34. LAS-AT: Adversarial Training with Learnable Attack Strategy**

cs.CV

**SubmitDate**: 2022-03-13    [paper-pdf](http://arxiv.org/pdf/2203.06616v1)

**Authors**: Xiaojun Jia, Yong Zhang, Baoyuan Wu, Ke Ma, Jue Wang, Xiaochun Cao

**Abstracts**: Adversarial training (AT) is always formulated as a minimax problem, of which the performance depends on the inner optimization that involves the generation of adversarial examples (AEs). Most previous methods adopt Projected Gradient Decent (PGD) with manually specifying attack parameters for AE generation. A combination of the attack parameters can be referred to as an attack strategy. Several works have revealed that using a fixed attack strategy to generate AEs during the whole training phase limits the model robustness and propose to exploit different attack strategies at different training stages to improve robustness. But those multi-stage hand-crafted attack strategies need much domain expertise, and the robustness improvement is limited. In this paper, we propose a novel framework for adversarial training by introducing the concept of "learnable attack strategy", dubbed LAS-AT, which learns to automatically produce attack strategies to improve the model robustness. Our framework is composed of a target network that uses AEs for training to improve robustness and a strategy network that produces attack strategies to control the AE generation. Experimental evaluations on three benchmark databases demonstrate the superiority of the proposed method. The code is released at https://github.com/jiaxiaojunQAQ/LAS-AT.



## **35. One Parameter Defense -- Defending against Data Inference Attacks via Differential Privacy**

cs.CR

**SubmitDate**: 2022-03-13    [paper-pdf](http://arxiv.org/pdf/2203.06580v1)

**Authors**: Dayong Ye, Sheng Shen, Tianqing Zhu, Bo Liu, Wanlei Zhou

**Abstracts**: Machine learning models are vulnerable to data inference attacks, such as membership inference and model inversion attacks. In these types of breaches, an adversary attempts to infer a data record's membership in a dataset or even reconstruct this data record using a confidence score vector predicted by the target model. However, most existing defense methods only protect against membership inference attacks. Methods that can combat both types of attacks require a new model to be trained, which may not be time-efficient. In this paper, we propose a differentially private defense method that handles both types of attacks in a time-efficient manner by tuning only one parameter, the privacy budget. The central idea is to modify and normalize the confidence score vectors with a differential privacy mechanism which preserves privacy and obscures membership and reconstructed data. Moreover, this method can guarantee the order of scores in the vector to avoid any loss in classification accuracy. The experimental results show the method to be an effective and timely defense against both membership inference and model inversion attacks with no reduction in accuracy.



## **36. Model Inversion Attack against Transfer Learning: Inverting a Model without Accessing It**

cs.CR

**SubmitDate**: 2022-03-13    [paper-pdf](http://arxiv.org/pdf/2203.06570v1)

**Authors**: Dayong Ye, Huiqiang Chen, Shuai Zhou, Tianqing Zhu, Wanlei Zhou, Shouling Ji

**Abstracts**: Transfer learning is an important approach that produces pre-trained teacher models which can be used to quickly build specialized student models. However, recent research on transfer learning has found that it is vulnerable to various attacks, e.g., misclassification and backdoor attacks. However, it is still not clear whether transfer learning is vulnerable to model inversion attacks. Launching a model inversion attack against transfer learning scheme is challenging. Not only does the student model hide its structural parameters, but it is also inaccessible to the adversary. Hence, when targeting a student model, both the white-box and black-box versions of existing model inversion attacks fail. White-box attacks fail as they need the target model's parameters. Black-box attacks fail as they depend on making repeated queries of the target model. However, they may not mean that transfer learning models are impervious to model inversion attacks. Hence, with this paper, we initiate research into model inversion attacks against transfer learning schemes with two novel attack methods. Both are black-box attacks, suiting different situations, that do not rely on queries to the target student model. In the first method, the adversary has the data samples that share the same distribution as the training set of the teacher model. In the second method, the adversary does not have any such samples. Experiments show that highly recognizable data records can be recovered with both of these methods. This means that even if a model is an inaccessible black-box, it can still be inverted.



## **37. Query-Efficient Black-box Adversarial Attacks Guided by a Transfer-based Prior**

cs.LG

Accepted by IEEE Transactions on Pattern Recognition and Machine  Intelligence (TPAMI). The official version is at  https://ieeexplore.ieee.org/document/9609659

**SubmitDate**: 2022-03-13    [paper-pdf](http://arxiv.org/pdf/2203.06560v1)

**Authors**: Yinpeng Dong, Shuyu Cheng, Tianyu Pang, Hang Su, Jun Zhu

**Abstracts**: Adversarial attacks have been extensively studied in recent years since they can identify the vulnerability of deep learning models before deployed. In this paper, we consider the black-box adversarial setting, where the adversary needs to craft adversarial examples without access to the gradients of a target model. Previous methods attempted to approximate the true gradient either by using the transfer gradient of a surrogate white-box model or based on the feedback of model queries. However, the existing methods inevitably suffer from low attack success rates or poor query efficiency since it is difficult to estimate the gradient in a high-dimensional input space with limited information. To address these problems and improve black-box attacks, we propose two prior-guided random gradient-free (PRGF) algorithms based on biased sampling and gradient averaging, respectively. Our methods can take the advantage of a transfer-based prior given by the gradient of a surrogate model and the query information simultaneously. Through theoretical analyses, the transfer-based prior is appropriately integrated with model queries by an optimal coefficient in each method. Extensive experiments demonstrate that, in comparison with the alternative state-of-the-arts, both of our methods require much fewer queries to attack black-box models with higher success rates.



## **38. Label-only Model Inversion Attack: The Attack that Requires the Least Information**

cs.CR

**SubmitDate**: 2022-03-13    [paper-pdf](http://arxiv.org/pdf/2203.06555v1)

**Authors**: Dayong Ye, Tianqing Zhu, Shuai Zhou, Bo Liu, Wanlei Zhou

**Abstracts**: In a model inversion attack, an adversary attempts to reconstruct the data records, used to train a target model, using only the model's output. In launching a contemporary model inversion attack, the strategies discussed are generally based on either predicted confidence score vectors, i.e., black-box attacks, or the parameters of a target model, i.e., white-box attacks. However, in the real world, model owners usually only give out the predicted labels; the confidence score vectors and model parameters are hidden as a defense mechanism to prevent such attacks. Unfortunately, we have found a model inversion method that can reconstruct the input data records based only on the output labels. We believe this is the attack that requires the least information to succeed and, therefore, has the best applicability. The key idea is to exploit the error rate of the target model to compute the median distance from a set of data records to the decision boundary of the target model. The distance, then, is used to generate confidence score vectors which are adopted to train an attack model to reconstruct the data records. The experimental results show that highly recognizable data records can be reconstructed with far less information than existing methods.



## **39. Mal2GCN: A Robust Malware Detection Approach Using Deep Graph Convolutional Networks With Non-Negative Weights**

cs.CR

13 pages, 12 figures, 5 tables

**SubmitDate**: 2022-03-12    [paper-pdf](http://arxiv.org/pdf/2108.12473v2)

**Authors**: Omid Kargarnovin, Amir Mahdi Sadeghzadeh, Rasool Jalili

**Abstracts**: With the growing pace of using Deep Learning (DL) to solve various problems, securing these models against adversaries has become one of the main concerns of researchers. Recent studies have shown that DL-based malware detectors are vulnerable to adversarial examples. An adversary can create carefully crafted adversarial examples to evade DL-based malware detectors. In this paper, we propose Mal2GCN, a robust malware detection model that uses Function Call Graph (FCG) representation of executable files combined with Graph Convolution Network (GCN) to detect Windows malware. Since FCG representation of executable files is more robust than raw byte sequence representation, numerous proposed adversarial example generating methods are ineffective in evading Mal2GCN. Moreover, we use the non-negative training method to transform Mal2GCN to a monotonically non-decreasing function; thereby, it becomes theoretically robust against appending attacks. We then present a black-box source code-based adversarial malware generation approach that can be used to evaluate the robustness of malware detection models against real-world adversaries. The proposed approach injects adversarial codes into the various locations of malware source codes to evade malware detection models. The experiments demonstrate that Mal2GCN with non-negative weights has high accuracy in detecting Windows malware, and it is also robust against adversarial attacks that add benign features to the Malware source code.



## **40. A Survey in Adversarial Defences and Robustness in NLP**

cs.CL

**SubmitDate**: 2022-03-12    [paper-pdf](http://arxiv.org/pdf/2203.06414v1)

**Authors**: Shreya Goyal, Sumanth Doddapaneni, Mitesh M. Khapra, Balaraman Ravindran

**Abstracts**: In recent years, it has been seen that deep neural networks are lacking robustness and are likely to break in case of adversarial perturbations in input data. Strong adversarial attacks are proposed by various authors for computer vision and Natural Language Processing (NLP). As a counter-effort, several defense mechanisms are also proposed to save these networks from failing. In contrast with image data, generating adversarial attacks and defending these models is not easy in NLP because of the discrete nature of the text data. However, numerous methods for adversarial defense are proposed of late, for different NLP tasks such as text classification, named entity recognition, natural language inferencing, etc. These methods are not just used for defending neural networks from adversarial attacks, but also used as a regularization mechanism during training, saving the model from overfitting. The proposed survey is an attempt to review different methods proposed for adversarial defenses in NLP in the recent past by proposing a novel taxonomy. This survey also highlights the fragility of the advanced deep neural networks in NLP and the challenges in defending them.



## **41. Detecting CAN Masquerade Attacks with Signal Clustering Similarity**

cs.CR

8 pages, 5 figures, 3 tables

**SubmitDate**: 2022-03-11    [paper-pdf](http://arxiv.org/pdf/2201.02665v2)

**Authors**: Pablo Moriano, Robert A. Bridges, Michael D. Iannacone

**Abstracts**: Vehicular Controller Area Networks (CANs) are susceptible to cyber attacks of different levels of sophistication. Fabrication attacks are the easiest to administer -- an adversary simply sends (extra) frames on a CAN -- but also the easiest to detect because they disrupt frame frequency. To overcome time-based detection methods, adversaries must administer masquerade attacks by sending frames in lieu of (and therefore at the expected time of) benign frames but with malicious payloads. Research efforts have proven that CAN attacks, and masquerade attacks in particular, can affect vehicle functionality. Examples include causing unintended acceleration, deactivation of vehicle's brakes, as well as steering the vehicle. We hypothesize that masquerade attacks modify the nuanced correlations of CAN signal time series and how they cluster together. Therefore, changes in cluster assignments should indicate anomalous behavior. We confirm this hypothesis by leveraging our previously developed capability for reverse engineering CAN signals (i.e., CAN-D [Controller Area Network Decoder]) and focus on advancing the state of the art for detecting masquerade attacks by analyzing time series extracted from raw CAN frames. Specifically, we demonstrate that masquerade attacks can be detected by computing time series clustering similarity using hierarchical clustering on the vehicle's CAN signals (time series) and comparing the clustering similarity across CAN captures with and without attacks. We test our approach in a previously collected CAN dataset with masquerade attacks (i.e., the ROAD dataset) and develop a forensic tool as a proof of concept to demonstrate the potential of the proposed approach for detecting CAN masquerade attacks.



## **42. On Adversarial Robustness of Trajectory Prediction for Autonomous Vehicles**

cs.CV

13 pages, 13 figures, accepted by CVPR 2022

**SubmitDate**: 2022-03-11    [paper-pdf](http://arxiv.org/pdf/2201.05057v2)

**Authors**: Qingzhao Zhang, Shengtuo Hu, Jiachen Sun, Qi Alfred Chen, Z. Morley Mao

**Abstracts**: Trajectory prediction is a critical component for autonomous vehicles (AVs) to perform safe planning and navigation. However, few studies have analyzed the adversarial robustness of trajectory prediction or investigated whether the worst-case prediction can still lead to safe planning. To bridge this gap, we study the adversarial robustness of trajectory prediction models by proposing a new adversarial attack that perturbs normal vehicle trajectories to maximize the prediction error. Our experiments on three models and three datasets show that the adversarial prediction increases the prediction error by more than 150%. Our case studies show that if an adversary drives a vehicle close to the target AV following the adversarial trajectory, the AV may make an inaccurate prediction and even make unsafe driving decisions. We also explore possible mitigation techniques via data augmentation and trajectory smoothing. The implementation is open source at https://github.com/zqzqz/AdvTrajectoryPrediction.



## **43. Sparse Black-box Video Attack with Reinforcement Learning**

cs.CV

Accepted at IJCV 2022

**SubmitDate**: 2022-03-11    [paper-pdf](http://arxiv.org/pdf/2001.03754v3)

**Authors**: Xingxing Wei, Huanqian Yan, Bo Li

**Abstracts**: Adversarial attacks on video recognition models have been explored recently. However, most existing works treat each video frame equally and ignore their temporal interactions. To overcome this drawback, a few methods try to select some key frames and then perform attacks based on them. Unfortunately, their selection strategy is independent of the attacking step, therefore the resulting performance is limited. Instead, we argue the frame selection phase is closely relevant with the attacking phase. The key frames should be adjusted according to the attacking results. For that, we formulate the black-box video attacks into a Reinforcement Learning (RL) framework. Specifically, the environment in RL is set as the recognition model, and the agent in RL plays the role of frame selecting. By continuously querying the recognition models and receiving the attacking feedback, the agent gradually adjusts its frame selection strategy and adversarial perturbations become smaller and smaller. We conduct a series of experiments with two mainstream video recognition models: C3D and LRCN on the public UCF-101 and HMDB-51 datasets. The results demonstrate that the proposed method can significantly reduce the adversarial perturbations with efficient query times.



## **44. Block-Sparse Adversarial Attack to Fool Transformer-Based Text Classifiers**

cs.CL

ICASSP 2022, Code available at:  https://github.com/sssadrizadeh/transformer-text-classifier-attack

**SubmitDate**: 2022-03-11    [paper-pdf](http://arxiv.org/pdf/2203.05948v1)

**Authors**: Sahar Sadrizadeh, Ljiljana Dolamic, Pascal Frossard

**Abstracts**: Recently, it has been shown that, in spite of the significant performance of deep neural networks in different fields, those are vulnerable to adversarial examples. In this paper, we propose a gradient-based adversarial attack against transformer-based text classifiers. The adversarial perturbation in our method is imposed to be block-sparse so that the resultant adversarial example differs from the original sentence in only a few words. Due to the discrete nature of textual data, we perform gradient projection to find the minimizer of our proposed optimization problem. Experimental results demonstrate that, while our adversarial attack maintains the semantics of the sentence, it can reduce the accuracy of GPT-2 to less than 5% on different datasets (AG News, MNLI, and Yelp Reviews). Furthermore, the block-sparsity constraint of the proposed optimization problem results in small perturbations in the adversarial example.



## **45. Learning from Attacks: Attacking Variational Autoencoder for Improving Image Classification**

cs.LG

**SubmitDate**: 2022-03-11    [paper-pdf](http://arxiv.org/pdf/2203.07027v1)

**Authors**: Jianzhang Zheng, Fan Yang, Hao Shen, Xuan Tang, Mingsong Chen, Liang Song, Xian Wei

**Abstracts**: Adversarial attacks are often considered as threats to the robustness of Deep Neural Networks (DNNs). Various defending techniques have been developed to mitigate the potential negative impact of adversarial attacks against task predictions. This work analyzes adversarial attacks from a different perspective. Namely, adversarial examples contain implicit information that is useful to the predictions i.e., image classification, and treat the adversarial attacks against DNNs for data self-expression as extracted abstract representations that are capable of facilitating specific learning tasks. We propose an algorithmic framework that leverages the advantages of the DNNs for data self-expression and task-specific predictions, to improve image classification. The framework jointly learns a DNN for attacking Variational Autoencoder (VAE) networks and a DNN for classification, coined as Attacking VAE for Improve Classification (AVIC). The experiment results show that AVIC can achieve higher accuracy on standard datasets compared to the training with clean examples and the traditional adversarial training.



## **46. Reinforcement Learning for Linear Quadratic Control is Vulnerable Under Cost Manipulation**

eess.SY

This paper is yet to be peer-reviewed

**SubmitDate**: 2022-03-11    [paper-pdf](http://arxiv.org/pdf/2203.05774v1)

**Authors**: Yunhan Huang, Quanyan Zhu

**Abstracts**: In this work, we study the deception of a Linear-Quadratic-Gaussian (LQG) agent by manipulating the cost signals. We show that a small falsification on the cost parameters will only lead to a bounded change in the optimal policy and the bound is linear on the amount of falsification the attacker can apply on the cost parameters. We propose an attack model where the goal of the attacker is to mislead the agent into learning a `nefarious' policy with intended falsification on the cost parameters. We formulate the attack's problem as an optimization problem, which is proved to be convex, and developed necessary and sufficient conditions to check the achievability of the attacker's goal.   We showcase the adversarial manipulation on two types of LQG learners: the batch RL learner and the other is the adaptive dynamic programming (ADP) learner. Our results demonstrate that with only 2.296% of falsification on the cost data, the attacker misleads the batch RL into learning the 'nefarious' policy that leads the vehicle to a dangerous position. The attacker can also gradually trick the ADP learner into learning the same `nefarious' policy by consistently feeding the learner a falsified cost signal that stays close to the true cost signal. The aim of the paper is to raise people's awareness of the security threats faced by RL-enabled control systems.



## **47. Single Loop Gaussian Homotopy Method for Non-convex Optimization**

math.OC

45 pages

**SubmitDate**: 2022-03-11    [paper-pdf](http://arxiv.org/pdf/2203.05717v1)

**Authors**: Hidenori Iwakiri, Yuhang Wang, Shinji Ito, Akiko Takeda

**Abstracts**: The Gaussian homotopy (GH) method is a popular approach to finding better local minima for non-convex optimization problems by gradually changing the problem to be solved from a simple one to the original target one. Existing GH-based methods consisting of a double loop structure incur high computational costs, which may limit their potential for practical application. We propose a novel single loop framework for GH methods (SLGH) for both deterministic and stochastic settings. For those applications in which the convolution calculation required to build a GH function is difficult, we present zeroth-order SLGH algorithms with gradient-free oracles. The convergence rate of (zeroth-order) SLGH depends on the decreasing speed of a smoothing hyperparameter, and when the hyperparameter is chosen appropriately, it becomes consistent with the convergence rate of (zeroth-order) gradient descent. In experiments that included artificial highly non-convex examples and black-box adversarial attacks, we have demonstrated that our algorithms converge much faster than an existing double loop GH method while outperforming gradient descent-based methods in terms of finding a better solution.



## **48. Formalizing and Estimating Distribution Inference Risks**

cs.LG

Update: New version with more theoretical results and a deeper  exploration of results. We noted some discrepancies in our experiments on the  CelebA dataset and re-ran all of our experiments for this dataset, updating  Table 1 and Figures 2c, 3b, 4, 7a, and 8a in the process. These did not  substantially impact our results, and our conclusions and observations in  trends remain unchanged

**SubmitDate**: 2022-03-10    [paper-pdf](http://arxiv.org/pdf/2109.06024v5)

**Authors**: Anshuman Suri, David Evans

**Abstracts**: Distribution inference, sometimes called property inference, infers statistical properties about a training set from access to a model trained on that data. Distribution inference attacks can pose serious risks when models are trained on private data, but are difficult to distinguish from the intrinsic purpose of statistical machine learning -- namely, to produce models that capture statistical properties about a distribution. Motivated by Yeom et al.'s membership inference framework, we propose a formal definition of distribution inference attacks that is general enough to describe a broad class of attacks distinguishing between possible training distributions. We show how our definition captures previous ratio-based property inference attacks as well as new kinds of attack including revealing the average node degree or clustering coefficient of a training graph. To understand distribution inference risks, we introduce a metric that quantifies observed leakage by relating it to the leakage that would occur if samples from the training distribution were provided directly to the adversary. We report on a series of experiments across a range of different distributions using both novel black-box attacks and improved versions of the state-of-the-art white-box attacks. Our results show that inexpensive attacks are often as effective as expensive meta-classifier attacks, and that there are surprising asymmetries in the effectiveness of attacks.



## **49. TraSw: Tracklet-Switch Adversarial Attacks against Multi-Object Tracking**

cs.CV

**SubmitDate**: 2022-03-10    [paper-pdf](http://arxiv.org/pdf/2111.08954v2)

**Authors**: Delv Lin, Qi Chen, Chengyu Zhou, Kun He

**Abstracts**: Multi-Object Tracking (MOT) has achieved aggressive progress and derives many excellent deep learning models. However, the robustness of the trackers is rarely studied, and it is challenging to attack the MOT system since its mature association algorithms are designed to be robust against errors during the tracking. In this work, we analyze the vulnerability of popular pedestrian MOT trackers and propose a novel adversarial attack method called Tracklet-Switch (TraSw) against the complete tracking pipeline of MOT. TraSw can fool the advanced deep trackers (i.e., FairMOT and ByteTrack) to fail to track the targets in the subsequent frames by attacking very few frames. Experiments on the MOT-Challenge datasets (i.e., 2DMOT15, MOT17, and MOT20) show that TraSw can achieve an extraordinarily high success rate of over 95% by attacking only four frames on average. To our knowledge, this is the first work on the adversarial attack against pedestrian MOT trackers. The code is available at https://github.com/DerryHub/FairMOT-attack .



## **50. SoK: On the Semantic AI Security in Autonomous Driving**

cs.CR

Project website: https://sites.google.com/view/cav-sec/pass

**SubmitDate**: 2022-03-10    [paper-pdf](http://arxiv.org/pdf/2203.05314v1)

**Authors**: Junjie Shen, Ningfei Wang, Ziwen Wan, Yunpeng Luo, Takami Sato, Zhisheng Hu, Xinyang Zhang, Shengjian Guo, Zhenyu Zhong, Kang Li, Ziming Zhao, Chunming Qiao, Qi Alfred Chen

**Abstracts**: Autonomous Driving (AD) systems rely on AI components to make safety and correct driving decisions. Unfortunately, today's AI algorithms are known to be generally vulnerable to adversarial attacks. However, for such AI component-level vulnerabilities to be semantically impactful at the system level, it needs to address non-trivial semantic gaps both (1) from the system-level attack input spaces to those at AI component level, and (2) from AI component-level attack impacts to those at the system level. In this paper, we define such research space as semantic AI security as opposed to generic AI security. Over the past 5 years, increasingly more research works are performed to tackle such semantic AI security challenges in AD context, which has started to show an exponential growth trend.   In this paper, we perform the first systematization of knowledge of such growing semantic AD AI security research space. In total, we collect and analyze 53 such papers, and systematically taxonomize them based on research aspects critical for the security field. We summarize 6 most substantial scientific gaps observed based on quantitative comparisons both vertically among existing AD AI security works and horizontally with security works from closely-related domains. With these, we are able to provide insights and potential future directions not only at the design level, but also at the research goal, methodology, and community levels. To address the most critical scientific methodology-level gap, we take the initiative to develop an open-source, uniform, and extensible system-driven evaluation platform, named PASS, for the semantic AD AI security research community. We also use our implemented platform prototype to showcase the capabilities and benefits of such a platform using representative semantic AD AI attacks.



