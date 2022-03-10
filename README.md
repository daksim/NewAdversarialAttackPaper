# Latest Adversarial Attack Papers
**update at 2022-03-11 06:31:49**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Reverse Engineering $\ell_p$ attacks: A block-sparse optimization approach with recovery guarantees**

cs.LG

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2203.04886v1)

**Authors**: Darshan Thaker, Paris Giampouras, René Vidal

**Abstracts**: Deep neural network-based classifiers have been shown to be vulnerable to imperceptible perturbations to their input, such as $\ell_p$-bounded norm adversarial attacks. This has motivated the development of many defense methods, which are then broken by new attacks, and so on. This paper focuses on a different but related problem of reverse engineering adversarial attacks. Specifically, given an attacked signal, we study conditions under which one can determine the type of attack ($\ell_1$, $\ell_2$ or $\ell_\infty$) and recover the clean signal. We pose this problem as a block-sparse recovery problem, where both the signal and the attack are assumed to lie in a union of subspaces that includes one subspace per class and one subspace per attack type. We derive geometric conditions on the subspaces under which any attacked signal can be decomposed as the sum of a clean signal plus an attack. In addition, by determining the subspaces that contain the signal and the attack, we can also classify the signal and determine the attack type. Experiments on digit and face classification demonstrate the effectiveness of the proposed approach.



## **2. Defending Black-box Skeleton-based Human Activity Classifiers**

cs.CV

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2203.04713v1)

**Authors**: He Wang, Yunfeng Diao, Zichang Tan, Guodong Guo

**Abstracts**: Deep learning has been regarded as the `go to' solution for many tasks today, but its intrinsic vulnerability to malicious attacks has become a major concern. The vulnerability is affected by a variety of factors including models, tasks, data, and attackers. Consequently, methods such as Adversarial Training and Randomized Smoothing have been proposed to tackle the problem in a wide range of applications. In this paper, we investigate skeleton-based Human Activity Recognition, which is an important type of time-series data but under-explored in defense against attacks. Our method is featured by (1) a new Bayesian Energy-based formulation of robust discriminative classifiers, (2) a new parameterization of the adversarial sample manifold of actions, and (3) a new post-train Bayesian treatment on both the adversarial samples and the classifier. We name our framework Bayesian Energy-based Adversarial Training or BEAT. BEAT is straightforward but elegant, which turns vulnerable black-box classifiers into robust ones without sacrificing accuracy. It demonstrates surprising and universal effectiveness across a wide range of action classifiers and datasets, under various attacks.



## **3. Robust Federated Learning Against Adversarial Attacks for Speech Emotion Recognition**

cs.SD

11 pages, 6 figures, 3 tables

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2203.04696v1)

**Authors**: Yi Chang, Sofiane Laridi, Zhao Ren, Gregory Palmer, Björn W. Schuller, Marco Fisichella

**Abstracts**: Due to the development of machine learning and speech processing, speech emotion recognition has been a popular research topic in recent years. However, the speech data cannot be protected when it is uploaded and processed on servers in the internet-of-things applications of speech emotion recognition. Furthermore, deep neural networks have proven to be vulnerable to human-indistinguishable adversarial perturbations. The adversarial attacks generated from the perturbations may result in deep neural networks wrongly predicting the emotional states. We propose a novel federated adversarial learning framework for protecting both data and deep neural networks. The proposed framework consists of i) federated learning for data privacy, and ii) adversarial training at the training stage and randomisation at the testing stage for model robustness. The experiments show that our proposed framework can effectively protect the speech data locally and improve the model robustness against a series of adversarial attacks.



## **4. Shadows can be Dangerous: Stealthy and Effective Physical-world Adversarial Attack by Natural Phenomenon**

cs.CV

This paper has been accepted by CVPR2022. Code:  https://github.com/hncszyq/ShadowAttack

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2203.03818v2)

**Authors**: Yiqi Zhong, Xianming Liu, Deming Zhai, Junjun Jiang, Xiangyang Ji

**Abstracts**: Estimating the risk level of adversarial examples is essential for safely deploying machine learning models in the real world. One popular approach for physical-world attacks is to adopt the "sticker-pasting" strategy, which however suffers from some limitations, including difficulties in access to the target or printing by valid colors. A new type of non-invasive attacks emerged recently, which attempt to cast perturbation onto the target by optics based tools, such as laser beam and projector. However, the added optical patterns are artificial but not natural. Thus, they are still conspicuous and attention-grabbed, and can be easily noticed by humans. In this paper, we study a new type of optical adversarial examples, in which the perturbations are generated by a very common natural phenomenon, shadow, to achieve naturalistic and stealthy physical-world adversarial attack under the black-box setting. We extensively evaluate the effectiveness of this new attack on both simulated and real-world environments. Experimental results on traffic sign recognition demonstrate that our algorithm can generate adversarial examples effectively, reaching 98.23% and 90.47% success rates on LISA and GTSRB test sets respectively, while continuously misleading a moving camera over 95% of the time in real-world scenarios. We also offer discussions about the limitations and the defense mechanism of this attack.



## **5. Evaluation and Generation of Physical Adversarial Patch**

cs.CV

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2203.04623v1)

**Authors**: Xiao Yang, Yinpeng Dong, Tianyu Pang, Zihao Xiao, Hang Su, Jun Zhu

**Abstracts**: Recent studies have revealed the vulnerability of face recognition models against physical adversarial patches, which raises security concerns about the deployed face recognition systems. However, it is still challenging to ensure the reproducibility for most attack algorithms under complex physical conditions, which leads to the lack of a systematic evaluation of the existing methods. It is therefore imperative to develop a framework that can enable a comprehensive evaluation of the vulnerability of face recognition in the physical world. To this end, we propose to simulate the complex transformations of faces in the physical world via 3D-face modeling, which serves as a digital counterpart of physical faces. The generic framework allows us to control different face variations and physical conditions to conduct reproducible evaluations comprehensively. With this digital simulator, we further propose a Face3DAdv method considering the 3D face transformations and realistic physical variations. Extensive experiments validate that Face3DAdv can significantly improve the effectiveness of diverse physically realizable adversarial patches in both simulated and physical environments, against various white-box and black-box face recognition models.



## **6. Practical No-box Adversarial Attacks with Training-free Hybrid Image Transformation**

cs.CV

This is the revision (the previous version rated 8,8,5,4 in ICLR2022,  where 8 denotes "accept, good paper"), which has been further polished and  added many new experiments

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2203.04607v1)

**Authors**: Qilong Zhang, Chaoning Zhang, Chaoqun Li, Jingkuan Song, Lianli Gao, Heng Tao Shen

**Abstracts**: In recent years, the adversarial vulnerability of deep neural networks (DNNs) has raised increasing attention. Among all the threat models, no-box attacks are the most practical but extremely challenging since they neither rely on any knowledge of the target model or similar substitute model, nor access the dataset for training a new substitute model. Although a recent method has attempted such an attack in a loose sense, its performance is not good enough and computational overhead of training is expensive. In this paper, we move a step forward and show the existence of a \textbf{training-free} adversarial perturbation under the no-box threat model, which can be successfully used to attack different DNNs in real-time. Motivated by our observation that high-frequency component (HFC) domains in low-level features and plays a crucial role in classification, we attack an image mainly by manipulating its frequency components. Specifically, the perturbation is manipulated by suppression of the original HFC and adding of noisy HFC. We empirically and experimentally analyze the requirements of effective noisy HFC and show that it should be regionally homogeneous, repeating and dense. Extensive experiments on the ImageNet dataset demonstrate the effectiveness of our proposed no-box method. It attacks ten well-known models with a success rate of \textbf{98.13\%} on average, which outperforms state-of-the-art no-box attacks by \textbf{29.39\%}. Furthermore, our method is even competitive to mainstream transfer-based black-box attacks.



## **7. The Dangerous Combo: Fileless Malware and Cryptojacking**

cs.CR

9 Pages - Accepted to be published in SoutheastCon 2022 IEEE Region 3  Technical, Professional, and Student Conference. Mobile, Alabama, USA. Mar  31st to Apr 03rd 2022. https://ieeesoutheastcon.org/

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2203.03175v2)

**Authors**: Said Varlioglu, Nelly Elsayed, Zag ElSayed, Murat Ozer

**Abstracts**: Fileless malware and cryptojacking attacks have appeared independently as the new alarming threats in 2017. After 2020, fileless attacks have been devastating for victim organizations with low-observable characteristics. Also, the amount of unauthorized cryptocurrency mining has increased after 2019. Adversaries have started to merge these two different cyberattacks to gain more invisibility and profit under "Fileless Cryptojacking." This paper aims to provide a literature review in academic papers and industry reports for this new threat. Additionally, we present a new threat hunting-oriented DFIR approach with the best practices derived from field experience as well as the literature. Last, this paper reviews the fundamentals of the fileless threat that can also help ransomware researchers examine similar patterns.



## **8. Targeted Attack on Deep RL-based Autonomous Driving with Learned Visual Patterns**

cs.LG

7 pages, 4 figures; Accepted at ICRA 2022

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2109.07723v2)

**Authors**: Prasanth Buddareddygari, Travis Zhang, Yezhou Yang, Yi Ren

**Abstracts**: Recent studies demonstrated the vulnerability of control policies learned through deep reinforcement learning against adversarial attacks, raising concerns about the application of such models to risk-sensitive tasks such as autonomous driving. Threat models for these demonstrations are limited to (1) targeted attacks through real-time manipulation of the agent's observation, and (2) untargeted attacks through manipulation of the physical environment. The former assumes full access to the agent's states/observations at all times, while the latter has no control over attack outcomes. This paper investigates the feasibility of targeted attacks through visually learned patterns placed on physical objects in the environment, a threat model that combines the practicality and effectiveness of the existing ones. Through analysis, we demonstrate that a pre-trained policy can be hijacked within a time window, e.g., performing an unintended self-parking, when an adversarial object is present. To enable the attack, we adopt an assumption that the dynamics of both the environment and the agent can be learned by the attacker. Lastly, we empirically show the effectiveness of the proposed attack on different driving scenarios, perform a location robustness test, and study the tradeoff between the attack strength and its effectiveness. Code is available at https://github.com/ASU-APG/Targeted-Physical-Adversarial-Attacks-on-AD



## **9. Machine Learning in NextG Networks via Generative Adversarial Networks**

cs.LG

47 pages, 7 figures, 12 tables

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2203.04453v1)

**Authors**: Ender Ayanoglu, Kemal Davaslioglu, Yalin E. Sagduyu

**Abstracts**: Generative Adversarial Networks (GANs) are Machine Learning (ML) algorithms that have the ability to address competitive resource allocation problems together with detection and mitigation of anomalous behavior. In this paper, we investigate their use in next-generation (NextG) communications within the context of cognitive networks to address i) spectrum sharing, ii) detecting anomalies, and iii) mitigating security attacks. GANs have the following advantages. First, they can learn and synthesize field data, which can be costly, time consuming, and nonrepeatable. Second, they enable pre-training classifiers by using semi-supervised data. Third, they facilitate increased resolution. Fourth, they enable the recovery of corrupted bits in the spectrum. The paper provides the basics of GANs, a comparative discussion on different kinds of GANs, performance measures for GANs in computer vision and image processing as well as wireless applications, a number of datasets for wireless applications, performance measures for general classifiers, a survey of the literature on GANs for i)-iii) above, and future research directions. As a use case of GAN for NextG communications, we show that a GAN can be effectively applied for anomaly detection in signal classification (e.g., user authentication) outperforming another state-of-the-art ML technique such as an autoencoder.



## **10. DeepSE-WF: Unified Security Estimation for Website Fingerprinting Defenses**

cs.CR

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2203.04428v1)

**Authors**: Alexander Veicht, Cedric Renggli, Diogo Barradas

**Abstracts**: Website fingerprinting (WF) attacks, usually conducted with the help of a machine learning-based classifier, enable a network eavesdropper to pinpoint which web page a user is accessing through the inspection of traffic patterns. These attacks have been shown to succeed even when users browse the Internet through encrypted tunnels, e.g., through Tor or VPNs. To assess the security of new defenses against WF attacks, recent works have proposed feature-dependent theoretical frameworks that estimate the Bayes error of an adversary's features set or the mutual information leaked by manually-crafted features. Unfortunately, as state-of-the-art WF attacks increasingly rely on deep learning and latent feature spaces, security estimations based on simpler (and less informative) manually-crafted features can no longer be trusted to assess the potential success of a WF adversary in defeating such defenses. In this work, we propose DeepSE-WF, a novel WF security estimation framework that leverages specialized kNN-based estimators to produce Bayes error and mutual information estimates from learned latent feature spaces, thus bridging the gap between current WF attacks and security estimation methods. Our evaluation reveals that DeepSE-WF produces tighter security estimates than previous frameworks, reducing the required computational resources to output security estimations by one order of magnitude.



## **11. Disrupting Adversarial Transferability in Deep Neural Networks**

cs.LG

20 pages, 13 figures

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2108.12492v2)

**Authors**: Christopher Wiedeman, Ge Wang

**Abstracts**: Adversarial attack transferability is well-recognized in deep learning. Prior work has partially explained transferability by recognizing common adversarial subspaces and correlations between decision boundaries, but little is known beyond this. We propose that transferability between seemingly different models is due to a high linear correlation between the feature sets that different networks extract. In other words, two models trained on the same task that are distant in the parameter space likely extract features in the same fashion, just with trivial affine transformations between the latent spaces. Furthermore, we show how applying a feature correlation loss, which decorrelates the extracted features in a latent space, can reduce the transferability of adversarial attacks between models, suggesting that the models complete tasks in semantically different ways. Finally, we propose a Dual Neck Autoencoder (DNA), which leverages this feature correlation loss to create two meaningfully different encodings of input information with reduced transferability.



## **12. RAPTEE: Leveraging trusted execution environments for Byzantine-tolerant peer sampling services**

cs.DC

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2203.04258v1)

**Authors**: Matthieu Pigaglio, Joachim Bruneau-Queyreix, David Bromberg, Davide Frey, Etienne Rivière, Laurent Réveillère

**Abstracts**: Peer sampling is a first-class abstraction used in distributed systems for overlay management and information dissemination. The goal of peer sampling is to continuously build and refresh a partial and local view of the full membership of a dynamic, large-scale distributed system. Malicious nodes under the control of an adversary may aim at being over-represented in the views of correct nodes, increasing their impact on the proper operation of protocols built over peer sampling. State-of-the-art Byzantine resilient peer sampling protocols reduce this bias as long as Byzantines are not overly present. This paper studies the benefits brought to the resilience of peer sampling services when considering that a small portion of trusted nodes can run code whose authenticity and integrity can be assessed within a trusted execution environment, and specifically Intel's software guard extensions technology (SGX). We present RAPTEE, a protocol that builds and leverages trusted gossip-based communications to hamper an adversary's ability to increase its system-wide representation in the views of all nodes. We apply RAPTEE to BRAHMS, the most resilient peer sampling protocol to date. Experiments with 10,000 nodes show that with only 1% of SGX-capable devices, RAPTEE can reduce the proportion of Byzantine IDs in the view of honest nodes by up to 17% when the system contains 10% of Byzantine nodes. In addition, the security guarantees of RAPTEE hold even in the presence of a powerful attacker attempting to identify trusted nodes and injecting view-poisoned trusted nodes.



## **13. Towards Effective and Robust Neural Trojan Defenses via Input Filtering**

cs.CR

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2202.12154v3)

**Authors**: Kien Do, Haripriya Harikumar, Hung Le, Dung Nguyen, Truyen Tran, Santu Rana, Dang Nguyen, Willy Susilo, Svetha Venkatesh

**Abstracts**: Trojan attacks on deep neural networks are both dangerous and surreptitious. Over the past few years, Trojan attacks have advanced from using only a single input-agnostic trigger and targeting only one class to using multiple, input-specific triggers and targeting multiple classes. However, Trojan defenses have not caught up with this development. Most defense methods still make out-of-date assumptions about Trojan triggers and target classes, thus, can be easily circumvented by modern Trojan attacks. To deal with this problem, we propose two novel "filtering" defenses called Variational Input Filtering (VIF) and Adversarial Input Filtering (AIF) which leverage lossy data compression and adversarial learning respectively to effectively purify all potential Trojan triggers in the input at run time without making assumptions about the number of triggers/target classes or the input dependence property of triggers. In addition, we introduce a new defense mechanism called "Filtering-then-Contrasting" (FtC) which helps avoid the drop in classification accuracy on clean data caused by "filtering", and combine it with VIF/AIF to derive new defenses of this kind. Extensive experimental results and ablation studies show that our proposed defenses significantly outperform well-known baseline defenses in mitigating five advanced Trojan attacks including two recent state-of-the-art while being quite robust to small amounts of training data and large-norm triggers.



## **14. Adaptative Perturbation Patterns: Realistic Adversarial Learning for Robust NIDS**

cs.CR

16 pages, 6 tables, 8 figures, Future Internet journal

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2203.04234v1)

**Authors**: João Vitorino, Nuno Oliveira, Isabel Praça

**Abstracts**: Adversarial attacks pose a major threat to machine learning and to the systems that rely on it. Nonetheless, adversarial examples cannot be freely generated for domains with tabular data, such as cybersecurity. This work establishes the fundamental constraint levels required to achieve realism and introduces the Adaptative Perturbation Pattern Method (A2PM) to fulfill these constraints in a gray-box setting. A2PM relies on pattern sequences that are independently adapted to the characteristics of each class to create valid and coherent data perturbations. The developed method was evaluated in a cybersecurity case study with two scenarios: Enterprise and Internet of Things (IoT) networks. Multilayer Perceptron (MLP) and Random Forest (RF) classifiers were created with regular and adversarial training, using the CIC-IDS2017 and IoT-23 datasets. In each scenario, targeted and untargeted attacks were performed against the classifiers, and the generated examples were compared with the original network traffic flows to assess their realism. The obtained results demonstrate that A2PM provides a time efficient generation of realistic adversarial examples, which can be advantageous for both adversarial training and attacks.



## **15. Robustly-reliable learners under poisoning attacks**

cs.LG

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2203.04160v1)

**Authors**: Maria-Florina Balcan, Avrim Blum, Steve Hanneke, Dravyansh Sharma

**Abstracts**: Data poisoning attacks, in which an adversary corrupts a training set with the goal of inducing specific desired mistakes, have raised substantial concern: even just the possibility of such an attack can make a user no longer trust the results of a learning system. In this work, we show how to achieve strong robustness guarantees in the face of such attacks across multiple axes.   We provide robustly-reliable predictions, in which the predicted label is guaranteed to be correct so long as the adversary has not exceeded a given corruption budget, even in the presence of instance targeted attacks, where the adversary knows the test example in advance and aims to cause a specific failure on that example. Our guarantees are substantially stronger than those in prior approaches, which were only able to provide certificates that the prediction of the learning algorithm does not change, as opposed to certifying that the prediction is correct, as we are able to achieve in our work. Remarkably, we provide a complete characterization of learnability in this setting, in particular, nearly-tight matching upper and lower bounds on the region that can be certified, as well as efficient algorithms for computing this region given an ERM oracle. Moreover, for the case of linear separators over logconcave distributions, we provide efficient truly polynomial time algorithms (i.e., non-oracle algorithms) for such robustly-reliable predictions.   We also extend these results to the active setting where the algorithm adaptively asks for labels of specific informative examples, and the difficulty is that the adversary might even be adaptive to this interaction, as well as to the agnostic learning setting where there is no perfect classifier even over the uncorrupted data.



## **16. Adversarial Texture for Fooling Person Detectors in the Physical World**

cs.CV

Accepted by CVPR 2022

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2203.03373v2)

**Authors**: Zhanhao Hu, Siyuan Huang, Xiaopei Zhu, Xiaolin Hu, Fuchun Sun, Bo Zhang

**Abstracts**: Nowadays, cameras equipped with AI systems can capture and analyze images to detect people automatically. However, the AI system can make mistakes when receiving deliberately designed patterns in the real world, i.e., physical adversarial examples. Prior works have shown that it is possible to print adversarial patches on clothes to evade DNN-based person detectors. However, these adversarial examples could have catastrophic drops in the attack success rate when the viewing angle (i.e., the camera's angle towards the object) changes. To perform a multi-angle attack, we propose Adversarial Texture (AdvTexture). AdvTexture can cover clothes with arbitrary shapes so that people wearing such clothes can hide from person detectors from different viewing angles. We propose a generative method, named Toroidal-Cropping-based Expandable Generative Attack (TC-EGA), to craft AdvTexture with repetitive structures. We printed several pieces of cloth with AdvTexure and then made T-shirts, skirts, and dresses in the physical world. Experiments showed that these clothes could fool person detectors in the physical world.



## **17. Shape-invariant 3D Adversarial Point Clouds**

cs.CV

Accepted at CVPR 2022

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2203.04041v1)

**Authors**: Qidong Huang, Xiaoyi Dong, Dongdong Chen, Hang Zhou, Weiming Zhang, Nenghai Yu

**Abstracts**: Adversary and invisibility are two fundamental but conflict characters of adversarial perturbations. Previous adversarial attacks on 3D point cloud recognition have often been criticized for their noticeable point outliers, since they just involve an "implicit constrain" like global distance loss in the time-consuming optimization to limit the generated noise. While point cloud is a highly structured data format, it is hard to metric and constrain its perturbation with a simple loss properly. In this paper, we propose a novel Point-Cloud Sensitivity Map to boost both the efficiency and imperceptibility of point perturbations. This map reveals the vulnerability of point cloud recognition models when encountering shape-invariant adversarial noises. These noises are designed along the shape surface with an "explicit constrain" instead of extra distance loss. Specifically, we first apply a reversible coordinate transformation on each point of the point cloud input, to reduce one degree of point freedom and limit its movement on the tangent plane. Then we calculate the best attacking direction with the gradients of the transformed point cloud obtained on the white-box model. Finally we assign each point with a non-negative score to construct the sensitivity map, which benefits both white-box adversarial invisibility and black-box query-efficiency extended in our work. Extensive evaluations prove that our method can achieve the superior performance on various point cloud recognition models, with its satisfying adversarial imperceptibility and strong resistance to different point cloud defense settings. Our code is available at: https://github.com/shikiw/SI-Adv.



## **18. ART-Point: Improving Rotation Robustness of Point Cloud Classifiers via Adversarial Rotation**

cs.CV

CVPR 2022

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2203.03888v1)

**Authors**: Robin Wang, Yibo Yang, Dacheng Tao

**Abstracts**: Point cloud classifiers with rotation robustness have been widely discussed in the 3D deep learning community. Most proposed methods either use rotation invariant descriptors as inputs or try to design rotation equivariant networks. However, robust models generated by these methods have limited performance under clean aligned datasets due to modifications on the original classifiers or input space. In this study, for the first time, we show that the rotation robustness of point cloud classifiers can also be acquired via adversarial training with better performance on both rotated and clean datasets. Specifically, our proposed framework named ART-Point regards the rotation of the point cloud as an attack and improves rotation robustness by training the classifier on inputs with Adversarial RoTations. We contribute an axis-wise rotation attack that uses back-propagated gradients of the pre-trained model to effectively find the adversarial rotations. To avoid model over-fitting on adversarial inputs, we construct rotation pools that leverage the transferability of adversarial rotations among samples to increase the diversity of training data. Moreover, we propose a fast one-step optimization to efficiently reach the final robust model. Experiments show that our proposed rotation attack achieves a high success rate and ART-Point can be used on most existing classifiers to improve the rotation robustness while showing better performance on clean datasets than state-of-the-art methods.



## **19. Submodularity-based False Data Injection Attack Scheme in Multi-agent Dynamical Systems**

math.DS

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2201.06017v2)

**Authors**: Xiaoyu Luo, Chengcheng Zhao, Chongrong Fang, Jianping He

**Abstracts**: Consensus in multi-agent dynamical systems is prone to be sabotaged by the adversary, which has attracted much attention due to its key role in broad applications. In this paper, we study a new false data injection (FDI) attack design problem, where the adversary with limited capability aims to select a subset of agents and manipulate their local multi-dimensional states to maximize the consensus convergence error. We first formulate the FDI attack design problem as a combinatorial optimization problem and prove it is NP-hard. Then, based on the submodularity optimization theory, we show the convergence error is a submodular function of the set of the compromised agents, which satisfies the property of diminishing marginal returns. In other words, the benefit of adding an extra agent to the compromised set decreases as that set becomes larger. With this property, we exploit the greedy scheme to find the optimal compromised agent set that can produce the maximum convergence error when adding one extra agent to that set each time. Thus, the FDI attack set selection algorithms are developed to obtain the near-optimal subset of the compromised agents. Furthermore, we derive the analytical suboptimality bounds and the worst-case running time under the proposed algorithms. Extensive simulation results are conducted to show the effectiveness of the proposed algorithm.



## **20. Adversarial Attacks in Cooperative AI**

cs.LG

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2111.14833v3)

**Authors**: Ted Fujimoto, Arthur Paul Pedersen

**Abstracts**: Single-agent reinforcement learning algorithms in a multi-agent environment are inadequate for fostering cooperation. If intelligent agents are to interact and work together to solve complex problems, methods that counter non-cooperative behavior are needed to facilitate the training of multiple agents. This is the goal of cooperative AI. Recent research in adversarial machine learning, however, shows that models (e.g., image classifiers) can be easily deceived into making inferior decisions. Meanwhile, an important line of research in cooperative AI has focused on introducing algorithmic improvements that accelerate learning of optimally cooperative behavior. We argue that prominent methods of cooperative AI are exposed to weaknesses analogous to those studied in prior machine learning research. More specifically, we show that three algorithms inspired by human-like social intelligence are, in principle, vulnerable to attacks that exploit weaknesses introduced by cooperative AI's algorithmic improvements and report experimental findings that illustrate how these vulnerabilities can be exploited in practice.



## **21. Taxonomy of Machine Learning Safety: A Survey and Primer**

cs.LG

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2106.04823v2)

**Authors**: Sina Mohseni, Haotao Wang, Zhiding Yu, Chaowei Xiao, Zhangyang Wang, Jay Yadawa

**Abstracts**: The open-world deployment of Machine Learning (ML) algorithms in safety-critical applications such as autonomous vehicles needs to address a variety of ML vulnerabilities such as interpretability, verifiability, and performance limitations. Research explores different approaches to improve ML dependability by proposing new models and training techniques to reduce generalization error, achieve domain adaptation, and detect outlier examples and adversarial attacks. However, there is a missing connection between ongoing ML research and well-established safety principles. In this paper, we present a structured and comprehensive review of ML techniques to improve the dependability of ML algorithms in uncontrolled open-world settings. From this review, we propose the Taxonomy of ML Safety that maps state-of-the-art ML techniques to key engineering safety strategies. Our taxonomy of ML safety presents a safety-oriented categorization of ML techniques to provide guidance for improving dependability of the ML design and development. The proposed taxonomy can serve as a safety checklist to aid designers in improving coverage and diversity of safety strategies employed in any given ML system.



## **22. Defending Graph Convolutional Networks against Dynamic Graph Perturbations via Bayesian Self-supervision**

cs.LG

The paper is accepted by AAAI 2022

**SubmitDate**: 2022-03-07    [paper-pdf](http://arxiv.org/pdf/2203.03762v1)

**Authors**: Jun Zhuang, Mohammad Al Hasan

**Abstracts**: In recent years, plentiful evidence illustrates that Graph Convolutional Networks (GCNs) achieve extraordinary accomplishments on the node classification task. However, GCNs may be vulnerable to adversarial attacks on label-scarce dynamic graphs. Many existing works aim to strengthen the robustness of GCNs; for instance, adversarial training is used to shield GCNs against malicious perturbations. However, these works fail on dynamic graphs for which label scarcity is a pressing issue. To overcome label scarcity, self-training attempts to iteratively assign pseudo-labels to highly confident unlabeled nodes but such attempts may suffer serious degradation under dynamic graph perturbations. In this paper, we generalize noisy supervision as a kind of self-supervised learning method and then propose a novel Bayesian self-supervision model, namely GraphSS, to address the issue. Extensive experiments demonstrate that GraphSS can not only affirmatively alert the perturbations on dynamic graphs but also effectively recover the prediction of a node classifier when the graph is under such perturbations. These two advantages prove to be generalized over three classic GCNs across five public graph datasets.



## **23. Art-Attack: Black-Box Adversarial Attack via Evolutionary Art**

cs.CR

**SubmitDate**: 2022-03-07    [paper-pdf](http://arxiv.org/pdf/2203.04405v1)

**Authors**: Phoenix Williams, Ke Li

**Abstracts**: Deep neural networks (DNNs) have achieved state-of-the-art performance in many tasks but have shown extreme vulnerabilities to attacks generated by adversarial examples. Many works go with a white-box attack that assumes total access to the targeted model including its architecture and gradients. A more realistic assumption is the black-box scenario where an attacker only has access to the targeted model by querying some input and observing its predicted class probabilities. Different from most prevalent black-box attacks that make use of substitute models or gradient estimation, this paper proposes a gradient-free attack by using a concept of evolutionary art to generate adversarial examples that iteratively evolves a set of overlapping transparent shapes. To evaluate the effectiveness of our proposed method, we attack three state-of-the-art image classification models trained on the CIFAR-10 dataset in a targeted manner. We conduct a parameter study outlining the impact the number and type of shapes have on the proposed attack's performance. In comparison to state-of-the-art black-box attacks, our attack is more effective at generating adversarial examples and achieves a higher attack success rate on all three baseline models.



## **24. Uncertify: Attacks Against Neural Network Certification**

cs.LG

**SubmitDate**: 2022-03-07    [paper-pdf](http://arxiv.org/pdf/2108.11299v2)

**Authors**: Tobias Lorenz, Marta Kwiatkowska, Mario Fritz

**Abstracts**: Certifiers for neural networks have made great progress towards provable robustness guarantees against evasion attacks using adversarial examples. However, introducing certifiers into deep learning systems also opens up new attack vectors, which need to be considered before deployment. In this work, we conduct the first systematic analysis of training-time attacks against certifiers in practical application pipelines, identifying new threat vectors that can be exploited to degrade the overall system. Using these insights, we design two backdoor attacks against network certifiers, which can drastically reduce certified robustness. For example, adding 1% poisoned data points during training is sufficient to reduce certified robustness by up to 95 percentage points, effectively rendering the certifier useless. We analyze how such novel attacks can compromise the overall system's integrity or availability. Our extensive experiments across multiple datasets, model architectures, and certifiers demonstrate the wide applicability of these attacks. A first investigation into potential defenses shows that current approaches are insufficient to mitigate the issue, highlighting the need for new, more specific solutions.



## **25. Searching for Robust Neural Architectures via Comprehensive and Reliable Evaluation**

cs.LG

**SubmitDate**: 2022-03-07    [paper-pdf](http://arxiv.org/pdf/2203.03128v1)

**Authors**: Jialiang Sun, Tingsong Jiang, Chao Li, Weien Zhou, Xiaoya Zhang, Wen Yao, Xiaoqian Chen

**Abstracts**: Neural architecture search (NAS) could help search for robust network architectures, where defining robustness evaluation metrics is the important procedure. However, current robustness evaluations in NAS are not sufficiently comprehensive and reliable. In particular, the common practice only considers adversarial noise and quantified metrics such as the Jacobian matrix, whereas, some studies indicated that the models are also vulnerable to other types of noises such as natural noise. In addition, existing methods taking adversarial noise as the evaluation just use the robust accuracy of the FGSM or PGD, but these adversarial attacks could not provide the adequately reliable evaluation, leading to the vulnerability of the models under stronger attacks. To alleviate the above problems, we propose a novel framework, called Auto Adversarial Attack and Defense (AAAD), where we employ neural architecture search methods, and four types of robustness evaluations are considered, including adversarial noise, natural noise, system noise and quantified metrics, thereby assisting in finding more robust architectures. Also, among the adversarial noise, we use the composite adversarial attack obtained by random search as the new metric to evaluate the robustness of the model architectures. The empirical results on the CIFAR10 dataset show that the searched efficient attack could help find more robust architectures.



## **26. Protecting Facial Privacy: Generating Adversarial Identity Masks via Style-robust Makeup Transfer**

cs.CV

Accepted by CVPR2022, NOT the camera-ready version

**SubmitDate**: 2022-03-07    [paper-pdf](http://arxiv.org/pdf/2203.03121v1)

**Authors**: Shengshan Hu, Xiaogeng Liu, Yechao Zhang, Minghui Li, Leo Yu Zhang, Hai Jin, Libing Wu

**Abstracts**: While deep face recognition (FR) systems have shown amazing performance in identification and verification, they also arouse privacy concerns for their excessive surveillance on users, especially for public face images widely spread on social networks. Recently, some studies adopt adversarial examples to protect photos from being identified by unauthorized face recognition systems. However, existing methods of generating adversarial face images suffer from many limitations, such as awkward visual, white-box setting, weak transferability, making them difficult to be applied to protect face privacy in reality. In this paper, we propose adversarial makeup transfer GAN (AMT-GAN), a novel face protection method aiming at constructing adversarial face images that preserve stronger black-box transferability and better visual quality simultaneously. AMT-GAN leverages generative adversarial networks (GAN) to synthesize adversarial face images with makeup transferred from reference images. In particular, we introduce a new regularization module along with a joint training strategy to reconcile the conflicts between the adversarial noises and the cycle consistence loss in makeup transfer, achieving a desirable balance between the attack strength and visual changes. Extensive experiments verify that compared with state of the arts, AMT-GAN can not only preserve a comfortable visual quality, but also achieve a higher attack success rate over commercial FR APIs, including Face++, Aliyun, and Microsoft.



## **27. Can You Hear It? Backdoor Attacks via Ultrasonic Triggers**

cs.CR

**SubmitDate**: 2022-03-06    [paper-pdf](http://arxiv.org/pdf/2107.14569v3)

**Authors**: Stefanos Koffas, Jing Xu, Mauro Conti, Stjepan Picek

**Abstracts**: This work explores backdoor attacks for automatic speech recognition systems where we inject inaudible triggers. By doing so, we make the backdoor attack challenging to detect for legitimate users, and thus, potentially more dangerous. We conduct experiments on two versions of a speech dataset and three neural networks and explore the performance of our attack concerning the duration, position, and type of the trigger. Our results indicate that less than 1% of poisoned data is sufficient to deploy a backdoor attack and reach a 100% attack success rate. We observed that short, non-continuous triggers result in highly successful attacks. However, since our trigger is inaudible, it can be as long as possible without raising any suspicions making the attack more effective. Finally, we conducted our attack in actual hardware and saw that an adversary could manipulate inference in an Android application by playing the inaudible trigger over the air.



## **28. An Improved Genetic Algorithm and Its Application in Neural Network Adversarial Attack**

cs.NE

18 pages, 9 figures, 9 tables and 23 References

**SubmitDate**: 2022-03-06    [paper-pdf](http://arxiv.org/pdf/2110.01818v5)

**Authors**: Dingming Yang, Zeyu Yu, Hongqiang Yuan, Yanrong Cui

**Abstracts**: The choice of crossover and mutation strategies plays a crucial role in the searchability, convergence efficiency and precision of genetic algorithms. In this paper, a novel improved genetic algorithm is proposed by improving the crossover and mutation operation of the simple genetic algorithm, and it is verified by 15 test functions. The qualitative results show that, compared with three other mainstream swarm intelligence optimization algorithms, the algorithm can not only improve the global search ability, convergence efficiency and precision, but also increase the success rate of convergence to the optimal value under the same experimental conditions. The quantitative results show that the algorithm performs superiorly in 13 of the 15 tested functions. The Wilcoxon rank-sum test was used for statistical evaluation, showing the significant advantage of the algorithm at $95\%$ confidence intervals. Finally, the algorithm is applied to neural network adversarial attacks. The applied results show that the method does not need the structure and parameter information inside the neural network model, and it can obtain the adversarial samples with high confidence in a brief time just by the classification and confidence information output from the neural network.



## **29. Finding Dynamics Preserving Adversarial Winning Tickets**

cs.LG

Accepted by AISTATS2022

**SubmitDate**: 2022-03-06    [paper-pdf](http://arxiv.org/pdf/2202.06488v3)

**Authors**: Xupeng Shi, Pengfei Zheng, A. Adam Ding, Yuan Gao, Weizhong Zhang

**Abstracts**: Modern deep neural networks (DNNs) are vulnerable to adversarial attacks and adversarial training has been shown to be a promising method for improving the adversarial robustness of DNNs. Pruning methods have been considered in adversarial context to reduce model capacity and improve adversarial robustness simultaneously in training. Existing adversarial pruning methods generally mimic the classical pruning methods for natural training, which follow the three-stage 'training-pruning-fine-tuning' pipelines. We observe that such pruning methods do not necessarily preserve the dynamics of dense networks, making it potentially hard to be fine-tuned to compensate the accuracy degradation in pruning. Based on recent works of \textit{Neural Tangent Kernel} (NTK), we systematically study the dynamics of adversarial training and prove the existence of trainable sparse sub-network at initialization which can be trained to be adversarial robust from scratch. This theoretically verifies the \textit{lottery ticket hypothesis} in adversarial context and we refer such sub-network structure as \textit{Adversarial Winning Ticket} (AWT). We also show empirical evidences that AWT preserves the dynamics of adversarial training and achieve equal performance as dense adversarial training.



## **30. aaeCAPTCHA: The Design and Implementation of Audio Adversarial CAPTCHA**

cs.CR

Accepted at 7th IEEE European Symposium on Security and Privacy  (EuroS&P 2022)

**SubmitDate**: 2022-03-05    [paper-pdf](http://arxiv.org/pdf/2203.02735v1)

**Authors**: Md Imran Hossen, Xiali Hei

**Abstracts**: CAPTCHAs are designed to prevent malicious bot programs from abusing websites. Most online service providers deploy audio CAPTCHAs as an alternative to text and image CAPTCHAs for visually impaired users. However, prior research investigating the security of audio CAPTCHAs found them highly vulnerable to automated attacks using Automatic Speech Recognition (ASR) systems. To improve the robustness of audio CAPTCHAs against automated abuses, we present the design and implementation of an audio adversarial CAPTCHA (aaeCAPTCHA) system in this paper. The aaeCAPTCHA system exploits audio adversarial examples as CAPTCHAs to prevent the ASR systems from automatically solving them. Furthermore, we conducted a rigorous security evaluation of our new audio CAPTCHA design against five state-of-the-art DNN-based ASR systems and three commercial Speech-to-Text (STT) services. Our experimental evaluations demonstrate that aaeCAPTCHA is highly secure against these speech recognition technologies, even when the attacker has complete knowledge of the current attacks against audio adversarial examples. We also conducted a usability evaluation of the proof-of-concept implementation of the aaeCAPTCHA scheme. Our results show that it achieves high robustness at a moderate usability cost compared to normal audio CAPTCHAs. Finally, our extensive analysis highlights that aaeCAPTCHA can significantly enhance the security and robustness of traditional audio CAPTCHA systems while maintaining similar usability.



## **31. Generating Out of Distribution Adversarial Attack using Latent Space Poisoning**

cs.CV

IEEE SPL 2021

**SubmitDate**: 2022-03-05    [paper-pdf](http://arxiv.org/pdf/2012.05027v2)

**Authors**: Ujjwal Upadhyay, Prerana Mukherjee

**Abstracts**: Traditional adversarial attacks rely upon the perturbations generated by gradients from the network which are generally safeguarded by gradient guided search to provide an adversarial counterpart to the network. In this paper, we propose a novel mechanism of generating adversarial examples where the actual image is not corrupted rather its latent space representation is utilized to tamper with the inherent structure of the image while maintaining the perceptual quality intact and to act as legitimate data samples. As opposed to gradient-based attacks, the latent space poisoning exploits the inclination of classifiers to model the independent and identical distribution of the training dataset and tricks it by producing out of distribution samples. We train a disentangled variational autoencoder (beta-VAE) to model the data in latent space and then we add noise perturbations using a class-conditioned distribution function to the latent space under the constraint that it is misclassified to the target label. Our empirical results on MNIST, SVHN, and CelebA dataset validate that the generated adversarial examples can easily fool robust l_0, l_2, l_inf norm classifiers designed using provably robust defense mechanisms.



## **32. Adversarial samples for deep monocular 6D object pose estimation**

cs.CV

15 pages

**SubmitDate**: 2022-03-05    [paper-pdf](http://arxiv.org/pdf/2203.00302v2)

**Authors**: Jinlai Zhang, Weiming Li, Shuang Liang, Hao Wang, Jihong Zhu

**Abstracts**: Estimating 6D object pose from an RGB image is important for many real-world applications such as autonomous driving and robotic grasping. Recent deep learning models have achieved significant progress on this task but their robustness received little research attention. In this work, for the first time, we study adversarial samples that can fool deep learning models with imperceptible perturbations to input image. In particular, we propose a Unified 6D pose estimation Attack, namely U6DA, which can successfully attack several state-of-the-art (SOTA) deep learning models for 6D pose estimation. The key idea of our U6DA is to fool the models to predict wrong results for object instance localization and shape that are essential for correct 6D pose estimation. Specifically, we explore a transfer-based black-box attack to 6D pose estimation. We design the U6DA loss to guide the generation of adversarial examples, the loss aims to shift the segmentation attention map away from its original position. We show that the generated adversarial samples are not only effective for direct 6D pose estimation models, but also are able to attack two-stage models regardless of their robust RANSAC modules. Extensive experiments were conducted to demonstrate the effectiveness, transferability, and anti-defense capability of our U6DA on large-scale public benchmarks. We also introduce a new U6DA-Linemod dataset for robustness study of the 6D pose estimation task. Our codes and dataset will be available at \url{https://github.com/cuge1995/U6DA}.



## **33. Training privacy-preserving video analytics pipelines by suppressing features that reveal information about private attributes**

cs.CV

**SubmitDate**: 2022-03-05    [paper-pdf](http://arxiv.org/pdf/2203.02635v1)

**Authors**: Chau Yi Li, Andrea Cavallaro

**Abstracts**: Deep neural networks are increasingly deployed for scene analytics, including to evaluate the attention and reaction of people exposed to out-of-home advertisements. However, the features extracted by a deep neural network that was trained to predict a specific, consensual attribute (e.g. emotion) may also encode and thus reveal information about private, protected attributes (e.g. age or gender). In this work, we focus on such leakage of private information at inference time. We consider an adversary with access to the features extracted by the layers of a deployed neural network and use these features to predict private attributes. To prevent the success of such an attack, we modify the training of the network using a confusion loss that encourages the extraction of features that make it difficult for the adversary to accurately predict private attributes. We validate this training approach on image-based tasks using a publicly available dataset. Results show that, compared to the original network, the proposed PrivateNet can reduce the leakage of private information of a state-of-the-art emotion recognition classifier by 2.88% for gender and by 13.06% for age group, with a minimal effect on task accuracy.



## **34. Optimal Clock Synchronization with Signatures**

cs.DC

**SubmitDate**: 2022-03-04    [paper-pdf](http://arxiv.org/pdf/2203.02553v1)

**Authors**: Christoph Lenzen, Julian Loss

**Abstracts**: Cryptographic signatures can be used to increase the resilience of distributed systems against adversarial attacks, by increasing the number of faulty parties that can be tolerated. While this is well-studied for consensus, it has been underexplored in the context of fault-tolerant clock synchronization, even in fully connected systems. Here, the honest parties of an $n$-node system are required to compute output clocks of small skew (i.e., maximum phase offset) despite local clock rates varying between $1$ and $\vartheta>1$, end-to-end communication delays varying between $d-u$ and $d$, and the interference from malicious parties. So far, it is only known that clock pulses of skew $d$ can be generated with (trivially optimal) resilience of $\lceil n/2\rceil-1$ (PODC `19), improving over the tight bound of $\lceil n/3\rceil-1$ holding without signatures for \emph{any} skew bound (STOC `84, PODC `85). Since typically $d\gg u$ and $\vartheta-1\ll 1$, this is far from the lower bound of $u+(\vartheta-1)d$ that applies even in the fault-free case (IPL `01).   We prove matching upper and lower bounds of $\Theta(u+(\vartheta-1)d)$ on the skew for the resilience range from $\lceil n/3\rceil$ to $\lceil n/2\rceil-1$. The algorithm showing the upper bound is, under the assumption that the adversary cannot forge signatures, deterministic. The lower bound holds even if clocks are initially perfectly synchronized, message delays between honest nodes are known, $\vartheta$ is arbitrarily close to one, and the synchronization algorithm is randomized. This has crucial implications for network designers that seek to leverage signatures for providing more robust time. In contrast to the setting without signatures, they must ensure that an attacker cannot easily bypass the lower bound on the delay on links with a faulty endpoint.



## **35. Medical Aegis: Robust adversarial protectors for medical images**

cs.CV

**SubmitDate**: 2022-03-04    [paper-pdf](http://arxiv.org/pdf/2111.10969v4)

**Authors**: Qingsong Yao, Zecheng He, S. Kevin Zhou

**Abstracts**: Deep neural network based medical image systems are vulnerable to adversarial examples. Many defense mechanisms have been proposed in the literature, however, the existing defenses assume a passive attacker who knows little about the defense system and does not change the attack strategy according to the defense. Recent works have shown that a strong adaptive attack, where an attacker is assumed to have full knowledge about the defense system, can easily bypass the existing defenses. In this paper, we propose a novel adversarial example defense system called Medical Aegis. To the best of our knowledge, Medical Aegis is the first defense in the literature that successfully addresses the strong adaptive adversarial example attacks to medical images. Medical Aegis boasts two-tier protectors: The first tier of Cushion weakens the adversarial manipulation capability of an attack by removing its high-frequency components, yet posing a minimal effect on classification performance of the original image; the second tier of Shield learns a set of per-class DNN models to predict the logits of the protected model. Deviation from the Shield's prediction indicates adversarial examples. Shield is inspired by the observations in our stress tests that there exist robust trails in the shallow layers of a DNN model, which the adaptive attacks can hardly destruct. Experimental results show that the proposed defense accurately detects adaptive attacks, with negligible overhead for model inference.



## **36. Adversarial Patterns: Building Robust Android Malware Classifiers**

cs.CR

**SubmitDate**: 2022-03-04    [paper-pdf](http://arxiv.org/pdf/2203.02121v1)

**Authors**: Dipkamal Bhusal, Nidhi Rastogi

**Abstracts**: Deep learning-based classifiers have substantially improved recognition of malware samples. However, these classifiers can be vulnerable to adversarial input perturbations. Any vulnerability in malware classifiers poses significant threats to the platforms they defend. Therefore, to create stronger defense models against malware, we must understand the patterns in input perturbations caused by an adversary. This survey paper presents a comprehensive study on adversarial machine learning for android malware classifiers. We first present an extensive background in building a machine learning classifier for android malware, covering both image-based and text-based feature extraction approaches. Then, we examine the pattern and advancements in the state-of-the-art research in evasion attacks and defenses. Finally, we present guidelines for designing robust malware classifiers and enlist research directions for the future.



## **37. Label Leakage and Protection from Forward Embedding in Vertical Federated Learning**

cs.LG

**SubmitDate**: 2022-03-04    [paper-pdf](http://arxiv.org/pdf/2203.01451v2)

**Authors**: Jiankai Sun, Xin Yang, Yuanshun Yao, Chong Wang

**Abstracts**: Vertical federated learning (vFL) has gained much attention and been deployed to solve machine learning problems with data privacy concerns in recent years. However, some recent work demonstrated that vFL is vulnerable to privacy leakage even though only the forward intermediate embedding (rather than raw features) and backpropagated gradients (rather than raw labels) are communicated between the involved participants. As the raw labels often contain highly sensitive information, some recent work has been proposed to prevent the label leakage from the backpropagated gradients effectively in vFL. However, these work only identified and defended the threat of label leakage from the backpropagated gradients. None of these work has paid attention to the problem of label leakage from the intermediate embedding. In this paper, we propose a practical label inference method which can steal private labels effectively from the shared intermediate embedding even though some existing protection methods such as label differential privacy and gradients perturbation are applied. The effectiveness of the label attack is inseparable from the correlation between the intermediate embedding and corresponding private labels. To mitigate the issue of label leakage from the forward embedding, we add an additional optimization goal at the label party to limit the label stealing ability of the adversary by minimizing the distance correlation between the intermediate embedding and corresponding private labels. We conducted massive experiments to demonstrate the effectiveness of our proposed protection methods.



## **38. Differentially Private Label Protection in Split Learning**

cs.LG

**SubmitDate**: 2022-03-04    [paper-pdf](http://arxiv.org/pdf/2203.02073v1)

**Authors**: Xin Yang, Jiankai Sun, Yuanshun Yao, Junyuan Xie, Chong Wang

**Abstracts**: Split learning is a distributed training framework that allows multiple parties to jointly train a machine learning model over vertically partitioned data (partitioned by attributes). The idea is that only intermediate computation results, rather than private features and labels, are shared between parties so that raw training data remains private. Nevertheless, recent works showed that the plaintext implementation of split learning suffers from severe privacy risks that a semi-honest adversary can easily reconstruct labels. In this work, we propose \textsf{TPSL} (Transcript Private Split Learning), a generic gradient perturbation based split learning framework that provides provable differential privacy guarantee. Differential privacy is enforced on not only the model weights, but also the communicated messages in the distributed computation setting. Our experiments on large-scale real-world datasets demonstrate the robustness and effectiveness of \textsf{TPSL} against label leakage attacks. We also find that \textsf{TPSL} have a better utility-privacy trade-off than baselines.



## **39. Can Authoritative Governments Abuse the Right to Access?**

cs.CR

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2203.02068v1)

**Authors**: Cédric Lauradoux

**Abstracts**: The right to access is a great tool provided by the GDPR to empower data subjects with their data. However, it needs to be implemented properly otherwise it could turn subject access requests against the subjects privacy. Indeed, recent works have shown that it is possible to abuse the right to access using impersonation attacks. We propose to extend those impersonation attacks by considering that the adversary has an access to governmental resources. In this case, the adversary can forge official documents or exploit copy of them. Our attack affects more people than one may expect. To defeat the attacks from this kind of adversary, several solutions are available like multi-factors or proof of aliveness. Our attacks highlight the need for strong procedures to authenticate subject access requests.



## **40. Autonomous and Resilient Control for Optimal LEO Satellite Constellation Coverage Against Space Threats**

eess.SY

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2203.02050v1)

**Authors**: Yuhan Zhao, Quanyan Zhu

**Abstracts**: LEO satellite constellation coverage has served as the base platform for various space applications. However, the rapidly evolving security environment such as orbit debris and adversarial space threats are greatly endangering the security of satellite constellation and integrity of the satellite constellation coverage. As on-orbit repairs are challenging, a distributed and autonomous protection mechanism is necessary to ensure the adaptation and self-healing of the satellite constellation coverage from different attacks. To this end, we establish an integrative and distributed framework to enable resilient satellite constellation coverage planning and control in a single orbit. Each satellite can make decisions individually to recover from adversarial and non-adversarial attacks and keep providing coverage service. We first provide models and methodologies to measure the coverage performance. Then, we formulate the joint resilient coverage planning-control problem as a two-stage problem. A coverage game is proposed to find the equilibrium constellation deployment for resilient coverage planning and an agent-based algorithm is developed to compute the equilibrium. The multi-waypoint Model Predictive Control (MPC) methodology is adopted to achieve autonomous self-healing control. Finally, we use a typical LEO satellite constellation as a case study to corroborate the results.



## **41. Why adversarial training can hurt robust accuracy**

cs.LG

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2203.02006v1)

**Authors**: Jacob Clarysse, Julia Hörmann, Fanny Yang

**Abstracts**: Machine learning classifiers with high test accuracy often perform poorly under adversarial attacks. It is commonly believed that adversarial training alleviates this issue. In this paper, we demonstrate that, surprisingly, the opposite may be true -- Even though adversarial training helps when enough data is available, it may hurt robust generalization in the small sample size regime. We first prove this phenomenon for a high-dimensional linear classification setting with noiseless observations. Our proof provides explanatory insights that may also transfer to feature learning models. Further, we observe in experiments on standard image datasets that the same behavior occurs for perceptible attacks that effectively reduce class information such as mask attacks and object corruptions.



## **42. Dynamic Backdoor Attacks Against Machine Learning Models**

cs.CR

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2003.03675v2)

**Authors**: Ahmed Salem, Rui Wen, Michael Backes, Shiqing Ma, Yang Zhang

**Abstracts**: Machine learning (ML) has made tremendous progress during the past decade and is being adopted in various critical real-world applications. However, recent research has shown that ML models are vulnerable to multiple security and privacy attacks. In particular, backdoor attacks against ML models have recently raised a lot of awareness. A successful backdoor attack can cause severe consequences, such as allowing an adversary to bypass critical authentication systems.   Current backdooring techniques rely on adding static triggers (with fixed patterns and locations) on ML model inputs which are prone to detection by the current backdoor detection mechanisms. In this paper, we propose the first class of dynamic backdooring techniques against deep neural networks (DNN), namely Random Backdoor, Backdoor Generating Network (BaN), and conditional Backdoor Generating Network (c-BaN). Triggers generated by our techniques can have random patterns and locations, which reduce the efficacy of the current backdoor detection mechanisms. In particular, BaN and c-BaN based on a novel generative network are the first two schemes that algorithmically generate triggers. Moreover, c-BaN is the first conditional backdooring technique that given a target label, it can generate a target-specific trigger. Both BaN and c-BaN are essentially a general framework which renders the adversary the flexibility for further customizing backdoor attacks.   We extensively evaluate our techniques on three benchmark datasets: MNIST, CelebA, and CIFAR-10. Our techniques achieve almost perfect attack performance on backdoored data with a negligible utility loss. We further show that our techniques can bypass current state-of-the-art defense mechanisms against backdoor attacks, including ABS, Februus, MNTD, Neural Cleanse, and STRIP.



## **43. Assessing the Robustness of Visual Question Answering Models**

cs.CV

24 pages, 13 figures, International Journal of Computer Vision (IJCV)  [under review]. arXiv admin note: substantial text overlap with  arXiv:1711.06232, arXiv:1709.04625

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/1912.01452v2)

**Authors**: Jia-Hong Huang, Modar Alfadly, Bernard Ghanem, Marcel Worring

**Abstracts**: Deep neural networks have been playing an essential role in the task of Visual Question Answering (VQA). Until recently, their accuracy has been the main focus of research. Now there is a trend toward assessing the robustness of these models against adversarial attacks by evaluating the accuracy of these models under increasing levels of noisiness in the inputs of VQA models. In VQA, the attack can target the image and/or the proposed query question, dubbed main question, and yet there is a lack of proper analysis of this aspect of VQA. In this work, we propose a new method that uses semantically related questions, dubbed basic questions, acting as noise to evaluate the robustness of VQA models. We hypothesize that as the similarity of a basic question to the main question decreases, the level of noise increases. To generate a reasonable noise level for a given main question, we rank a pool of basic questions based on their similarity with this main question. We cast this ranking problem as a LASSO optimization problem. We also propose a novel robustness measure Rscore and two large-scale basic question datasets in order to standardize robustness analysis of VQA models. The experimental results demonstrate that the proposed evaluation method is able to effectively analyze the robustness of VQA models. To foster the VQA research, we will publish our proposed datasets.



## **44. Detection of Word Adversarial Examples in Text Classification: Benchmark and Baseline via Robust Density Estimation**

cs.CL

Findings of ACL 2022

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2203.01677v1)

**Authors**: KiYoon Yoo, Jangho Kim, Jiho Jang, Nojun Kwak

**Abstracts**: Word-level adversarial attacks have shown success in NLP models, drastically decreasing the performance of transformer-based models in recent years. As a countermeasure, adversarial defense has been explored, but relatively few efforts have been made to detect adversarial examples. However, detecting adversarial examples may be crucial for automated tasks (e.g. review sentiment analysis) that wish to amass information about a certain population and additionally be a step towards a robust defense system. To this end, we release a dataset for four popular attack methods on four datasets and four models to encourage further research in this field. Along with it, we propose a competitive baseline based on density estimation that has the highest AUC on 29 out of 30 dataset-attack-model combinations. Source code is available in https://github.com/anoymous92874838/text-adv-detection.



## **45. On Improving Adversarial Transferability of Vision Transformers**

cs.CV

ICLR'22 (Spotlight), the first two authors contributed equally. Code:  https://t.ly/hBbW

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2106.04169v3)

**Authors**: Muzammal Naseer, Kanchana Ranasinghe, Salman Khan, Fahad Shahbaz Khan, Fatih Porikli

**Abstracts**: Vision transformers (ViTs) process input images as sequences of patches via self-attention; a radically different architecture than convolutional neural networks (CNNs). This makes it interesting to study the adversarial feature space of ViT models and their transferability. In particular, we observe that adversarial patterns found via conventional adversarial attacks show very \emph{low} black-box transferability even for large ViT models. We show that this phenomenon is only due to the sub-optimal attack procedures that do not leverage the true representation potential of ViTs. A deep ViT is composed of multiple blocks, with a consistent architecture comprising of self-attention and feed-forward layers, where each block is capable of independently producing a class token. Formulating an attack using only the last class token (conventional approach) does not directly leverage the discriminative information stored in the earlier tokens, leading to poor adversarial transferability of ViTs. Using the compositional nature of ViT models, we enhance transferability of existing attacks by introducing two novel strategies specific to the architecture of ViT models. (i) Self-Ensemble: We propose a method to find multiple discriminative pathways by dissecting a single ViT model into an ensemble of networks. This allows explicitly utilizing class-specific information at each ViT block. (ii) Token Refinement: We then propose to refine the tokens to further enhance the discriminative capacity at each block of ViT. Our token refinement systematically combines the class tokens with structural information preserved within the patch tokens.



## **46. On Robustness of Neural Ordinary Differential Equations**

cs.LG

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/1910.05513v4)

**Authors**: Hanshu Yan, Jiawei Du, Vincent Y. F. Tan, Jiashi Feng

**Abstracts**: Neural ordinary differential equations (ODEs) have been attracting increasing attention in various research domains recently. There have been some works studying optimization issues and approximation capabilities of neural ODEs, but their robustness is still yet unclear. In this work, we fill this important gap by exploring robustness properties of neural ODEs both empirically and theoretically. We first present an empirical study on the robustness of the neural ODE-based networks (ODENets) by exposing them to inputs with various types of perturbations and subsequently investigating the changes of the corresponding outputs. In contrast to conventional convolutional neural networks (CNNs), we find that the ODENets are more robust against both random Gaussian perturbations and adversarial attack examples. We then provide an insightful understanding of this phenomenon by exploiting a certain desirable property of the flow of a continuous-time ODE, namely that integral curves are non-intersecting. Our work suggests that, due to their intrinsic robustness, it is promising to use neural ODEs as a basic block for building robust deep network models. To further enhance the robustness of vanilla neural ODEs, we propose the time-invariant steady neural ODE (TisODE), which regularizes the flow on perturbed data via the time-invariant property and the imposition of a steady-state constraint. We show that the TisODE method outperforms vanilla neural ODEs and also can work in conjunction with other state-of-the-art architectural methods to build more robust deep networks.



## **47. Authentication Attacks on Projection-based Cancelable Biometric Schemes**

cs.CR

arXiv admin note: text overlap with arXiv:1910.01389 by other authors

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2110.15163v2)

**Authors**: Axel Durbet, Pascal Lafourcade, Denis Migdal, Kevin Thiry-Atighehchi, Paul-Marie Grollemund

**Abstracts**: Cancelable biometric schemes aim at generating secure biometric templates by combining user specific tokens, such as password, stored secret or salt, along with biometric data. This type of transformation is constructed as a composition of a biometric transformation with a feature extraction algorithm. The security requirements of cancelable biometric schemes concern the irreversibility, unlinkability and revocability of templates, without losing in accuracy of comparison. While several schemes were recently attacked regarding these requirements, full reversibility of such a composition in order to produce colliding biometric characteristics, and specifically presentation attacks, were never demonstrated to the best of our knowledge. In this paper, we formalize these attacks for a traditional cancelable scheme with the help of integer linear programming (ILP) and quadratically constrained quadratic programming (QCQP). Solving these optimization problems allows an adversary to slightly alter its fingerprint image in order to impersonate any individual. Moreover, in an even more severe scenario, it is possible to simultaneously impersonate several individuals.



## **48. Ad2Attack: Adaptive Adversarial Attack on Real-Time UAV Tracking**

cs.CV

7 pages, 7 figures, accepted by ICRA 2022

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2203.01516v1)

**Authors**: Changhong Fu, Sihang Li, Xinnan Yuan, Junjie Ye, Ziang Cao, Fangqiang Ding

**Abstracts**: Visual tracking is adopted to extensive unmanned aerial vehicle (UAV)-related applications, which leads to a highly demanding requirement on the robustness of UAV trackers. However, adding imperceptible perturbations can easily fool the tracker and cause tracking failures. This risk is often overlooked and rarely researched at present. Therefore, to help increase awareness of the potential risk and the robustness of UAV tracking, this work proposes a novel adaptive adversarial attack approach, i.e., Ad$^2$Attack, against UAV object tracking. Specifically, adversarial examples are generated online during the resampling of the search patch image, which leads trackers to lose the target in the following frames. Ad$^2$Attack is composed of a direct downsampling module and a super-resolution upsampling module with adaptive stages. A novel optimization function is proposed for balancing the imperceptibility and efficiency of the attack. Comprehensive experiments on several well-known benchmarks and real-world conditions show the effectiveness of our attack method, which dramatically reduces the performance of the most advanced Siamese trackers.



## **49. Two Attacks On Proof-of-Stake GHOST/Ethereum**

cs.CR

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2203.01315v1)

**Authors**: Joachim Neu, Ertem Nusret Tas, David Tse

**Abstracts**: We present two attacks targeting the Proof-of-Stake (PoS) Ethereum consensus protocol. The first attack suggests a fundamental conceptual incompatibility between PoS and the Greedy Heaviest-Observed Sub-Tree (GHOST) fork choice paradigm employed by PoS Ethereum. In a nutshell, PoS allows an adversary with a vanishing amount of stake to produce an unlimited number of equivocating blocks. While most equivocating blocks will be orphaned, such orphaned `uncle blocks' still influence fork choice under the GHOST paradigm, bestowing upon the adversary devastating control over the canonical chain. While the Latest Message Driven (LMD) aspect of current PoS Ethereum prevents a straightforward application of this attack, our second attack shows how LMD specifically can be exploited to obtain a new variant of the balancing attack that overcomes a recent protocol addition that was intended to mitigate balancing-type attacks. Thus, in its current form, PoS Ethereum without and with LMD is vulnerable to our first and second attack, respectively.



## **50. Detecting Adversarial Perturbations in Multi-Task Perception**

cs.CV

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2203.01177v1)

**Authors**: Marvin Klingner, Varun Ravi Kumar, Senthil Yogamani, Andreas Bär, Tim Fingscheidt

**Abstracts**: While deep neural networks (DNNs) achieve impressive performance on environment perception tasks, their sensitivity to adversarial perturbations limits their use in practical applications. In this paper, we (i) propose a novel adversarial perturbation detection scheme based on multi-task perception of complex vision tasks (i.e., depth estimation and semantic segmentation). Specifically, adversarial perturbations are detected by inconsistencies between extracted edges of the input image, the depth output, and the segmentation output. To further improve this technique, we (ii) develop a novel edge consistency loss between all three modalities, thereby improving their initial consistency which in turn supports our detection scheme. We verify our detection scheme's effectiveness by employing various known attacks and image noises. In addition, we (iii) develop a multi-task adversarial attack, aiming at fooling both tasks as well as our detection scheme. Experimental evaluation on the Cityscapes and KITTI datasets shows that under an assumption of a 5% false positive rate up to 100% of images are correctly detected as adversarially perturbed, depending on the strength of the perturbation. Code will be available on github. A short video at https://youtu.be/KKa6gOyWmH4 provides qualitative results.



