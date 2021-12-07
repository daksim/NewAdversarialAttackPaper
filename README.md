# Latest Adversarial Attack Papers
**update at 2021-12-07 23:56:41**

[中文版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Context-Aware Transfer Attacks for Object Detection**

cs.CV

accepted to AAAI 2022

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2112.03223v1)

**Authors**: Zikui Cai, Xinxin Xie, Shasha Li, Mingjun Yin, Chengyu Song, Srikanth V. Krishnamurthy, Amit K. Roy-Chowdhury, M. Salman Asif

**Abstracts**: Blackbox transfer attacks for image classifiers have been extensively studied in recent years. In contrast, little progress has been made on transfer attacks for object detectors. Object detectors take a holistic view of the image and the detection of one object (or lack thereof) often depends on other objects in the scene. This makes such detectors inherently context-aware and adversarial attacks in this space are more challenging than those targeting image classifiers. In this paper, we present a new approach to generate context-aware attacks for object detectors. We show that by using co-occurrence of objects and their relative locations and sizes as context information, we can successfully generate targeted mis-categorization attacks that achieve higher transfer success rates on blackbox object detectors than the state-of-the-art. We test our approach on a variety of object detectors with images from PASCAL VOC and MS COCO datasets and demonstrate up to $20$ percentage points improvement in performance compared to the other state-of-the-art methods.



## **2. Improving the Adversarial Robustness for Speaker Verification by Self-Supervised Learning**

cs.SD

Accepted by TASLP

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2106.00273v3)

**Authors**: Haibin Wu, Xu Li, Andy T. Liu, Zhiyong Wu, Helen Meng, Hung-yi Lee

**Abstracts**: Previous works have shown that automatic speaker verification (ASV) is seriously vulnerable to malicious spoofing attacks, such as replay, synthetic speech, and recently emerged adversarial attacks. Great efforts have been dedicated to defending ASV against replay and synthetic speech; however, only a few approaches have been explored to deal with adversarial attacks. All the existing approaches to tackle adversarial attacks for ASV require the knowledge for adversarial samples generation, but it is impractical for defenders to know the exact attack algorithms that are applied by the in-the-wild attackers. This work is among the first to perform adversarial defense for ASV without knowing the specific attack algorithms. Inspired by self-supervised learning models (SSLMs) that possess the merits of alleviating the superficial noise in the inputs and reconstructing clean samples from the interrupted ones, this work regards adversarial perturbations as one kind of noise and conducts adversarial defense for ASV by SSLMs. Specifically, we propose to perform adversarial defense from two perspectives: 1) adversarial perturbation purification and 2) adversarial perturbation detection. Experimental results show that our detection module effectively shields the ASV by detecting adversarial samples with an accuracy of around 80%. Moreover, since there is no common metric for evaluating the adversarial defense performance for ASV, this work also formalizes evaluation metrics for adversarial defense considering both purification and detection based approaches into account. We sincerely encourage future works to benchmark their approaches based on the proposed evaluation framework.



## **3. Adversarial Example Detection for DNN Models: A Review and Experimental Comparison**

cs.CV

To be published on Artificial Intelligence Review journal (after  minor revision)

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2105.00203v3)

**Authors**: Ahmed Aldahdooh, Wassim Hamidouche, Sid Ahmed Fezza, Olivier Deforges

**Abstracts**: Deep learning (DL) has shown great success in many human-related tasks, which has led to its adoption in many computer vision based applications, such as security surveillance systems, autonomous vehicles and healthcare. Such safety-critical applications have to draw their path to success deployment once they have the capability to overcome safety-critical challenges. Among these challenges are the defense against or/and the detection of the adversarial examples (AEs). Adversaries can carefully craft small, often imperceptible, noise called perturbations to be added to the clean image to generate the AE. The aim of AE is to fool the DL model which makes it a potential risk for DL applications. Many test-time evasion attacks and countermeasures,i.e., defense or detection methods, are proposed in the literature. Moreover, few reviews and surveys were published and theoretically showed the taxonomy of the threats and the countermeasure methods with little focus in AE detection methods. In this paper, we focus on image classification task and attempt to provide a survey for detection methods of test-time evasion attacks on neural network classifiers. A detailed discussion for such methods is provided with experimental results for eight state-of-the-art detectors under different scenarios on four datasets. We also provide potential challenges and future perspectives for this research direction.



## **4. Robust Person Re-identification with Multi-Modal Joint Defence**

cs.CV

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2111.09571v2)

**Authors**: Yunpeng Gong, Lifei Chen

**Abstracts**: The Person Re-identification (ReID) system based on metric learning has been proved to inherit the vulnerability of deep neural networks (DNNs), which are easy to be fooled by adversarail metric attacks. Existing work mainly relies on adversarial training for metric defense, and more methods have not been fully studied. By exploring the impact of attacks on the underlying features, we propose targeted methods for metric attacks and defence methods. In terms of metric attack, we use the local color deviation to construct the intra-class variation of the input to attack color features. In terms of metric defenses, we propose a joint defense method which includes two parts of proactive defense and passive defense. Proactive defense helps to enhance the robustness of the model to color variations and the learning of structure relations across multiple modalities by constructing different inputs from multimodal images, and passive defense exploits the invariance of structural features in a changing pixel space by circuitous scaling to preserve structural features while eliminating some of the adversarial noise. Extensive experiments demonstrate that the proposed joint defense compared with the existing adversarial metric defense methods which not only against multiple attacks at the same time but also has not significantly reduced the generalization capacity of the model. The code is available at https://github.com/finger-monkey/multi-modal_joint_defence.



## **5. ML Attack Models: Adversarial Attacks and Data Poisoning Attacks**

cs.LG

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2112.02797v1)

**Authors**: Jing Lin, Long Dang, Mohamed Rahouti, Kaiqi Xiong

**Abstracts**: Many state-of-the-art ML models have outperformed humans in various tasks such as image classification. With such outstanding performance, ML models are widely used today. However, the existence of adversarial attacks and data poisoning attacks really questions the robustness of ML models. For instance, Engstrom et al. demonstrated that state-of-the-art image classifiers could be easily fooled by a small rotation on an arbitrary image. As ML systems are being increasingly integrated into safety and security-sensitive applications, adversarial attacks and data poisoning attacks pose a considerable threat. This chapter focuses on the two broad and important areas of ML security: adversarial attacks and data poisoning attacks.



## **6. An Improved Genetic Algorithm and Its Application in Neural Network Adversarial Attack**

cs.NE

14 pages, 7 figures, 4 tables and 20 References

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2110.01818v4)

**Authors**: Dingming Yang, Zeyu Yu, Hongqiang Yuan, Yanrong Cui

**Abstracts**: The choice of crossover and mutation strategies plays a crucial role in the search ability, convergence efficiency and precision of genetic algorithms. In this paper, a novel improved genetic algorithm is proposed by improving the crossover and mutation operation of the simple genetic algorithm, and it is verified by four test functions. Simulation results show that, comparing with three other mainstream swarm intelligence optimization algorithms, the algorithm can not only improve the global search ability, convergence efficiency and precision, but also increase the success rate of convergence to the optimal value under the same experimental conditions. Finally, the algorithm is applied to neural networks adversarial attacks. The applied results show that the method does not need the structure and parameter information inside the neural network model, and it can obtain the adversarial samples with high confidence in a brief time just by the classification and confidence information output from the neural network.



## **7. Staring Down the Digital Fulda Gap Path Dependency as a Cyber Defense Vulnerability**

cs.CY

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2112.02773v1)

**Authors**: Jan Kallberg

**Abstracts**: Academia, homeland security, defense, and media have accepted the perception that critical infrastructure in a future cyber war cyber conflict is the main gateway for a massive cyber assault on the U.S. The question is not if the assumption is correct or not, the question is instead of how did we arrive at that assumption. The cyber paradigm considers critical infrastructure the primary attack vector for future cyber conflicts. The national vulnerability embedded in critical infrastructure is given a position in the cyber discourse as close to an unquestionable truth as a natural law.   The American reaction to Sept. 11, and any attack on U.S. soil, hint to an adversary that attacking critical infrastructure to create hardship for the population could work contrary to the intended softening of the will to resist foreign influence. It is more likely that attacks that affect the general population instead strengthen the will to resist and fight, similar to the British reaction to the German bombing campaign Blitzen in 1940. We cannot rule out attacks that affect the general population, but there are not enough adversarial offensive capabilities to attack all 16 critical infrastructure sectors and gain strategic momentum. An adversary has limited cyberattack capabilities and needs to prioritize cyber targets that are aligned with the overall strategy. Logically, an adversary will focus their OCO on operations that has national security implications and support their military operations by denying, degrading, and confusing the U.S. information environment and U.S. cyber assets.



## **8. Label-Only Membership Inference Attacks**

cs.CR

16 pages, 11 figures, 2 tables Revision 2: 19 pages, 12 figures, 3  tables. Improved text and additional experiments. Final ICML paper

**SubmitDate**: 2021-12-05    [paper-pdf](http://arxiv.org/pdf/2007.14321v3)

**Authors**: Christopher A. Choquette-Choo, Florian Tramer, Nicholas Carlini, Nicolas Papernot

**Abstracts**: Membership inference attacks are one of the simplest forms of privacy leakage for machine learning models: given a data point and model, determine whether the point was used to train the model. Existing membership inference attacks exploit models' abnormal confidence when queried on their training data. These attacks do not apply if the adversary only gets access to models' predicted labels, without a confidence measure. In this paper, we introduce label-only membership inference attacks. Instead of relying on confidence scores, our attacks evaluate the robustness of a model's predicted labels under perturbations to obtain a fine-grained membership signal. These perturbations include common data augmentations or adversarial examples. We empirically show that our label-only membership inference attacks perform on par with prior attacks that required access to model confidences. We further demonstrate that label-only attacks break multiple defenses against membership inference attacks that (implicitly or explicitly) rely on a phenomenon we call confidence masking. These defenses modify a model's confidence scores in order to thwart attacks, but leave the model's predicted labels unchanged. Our label-only attacks demonstrate that confidence-masking is not a viable defense strategy against membership inference. Finally, we investigate worst-case label-only attacks, that infer membership for a small number of outlier data points. We show that label-only attacks also match confidence-based attacks in this setting. We find that training models with differential privacy and (strong) L2 regularization are the only known defense strategies that successfully prevents all attacks. This remains true even when the differential privacy budget is too high to offer meaningful provable guarantees.



## **9. Learning Swarm Interaction Dynamics from Density Evolution**

eess.SY

**SubmitDate**: 2021-12-05    [paper-pdf](http://arxiv.org/pdf/2112.02675v1)

**Authors**: Christos Mavridis, Amoolya Tirumalai, John Baras

**Abstracts**: We consider the problem of understanding the coordinated movements of biological or artificial swarms. In this regard, we propose a learning scheme to estimate the coordination laws of the interacting agents from observations of the swarm's density over time. We describe the dynamics of the swarm based on pairwise interactions according to a Cucker-Smale flocking model, and express the swarm's density evolution as the solution to a system of mean-field hydrodynamic equations. We propose a new family of parametric functions to model the pairwise interactions, which allows for the mean-field macroscopic system of integro-differential equations to be efficiently solved as an augmented system of PDEs. Finally, we incorporate the augmented system in an iterative optimization scheme to learn the dynamics of the interacting agents from observations of the swarm's density evolution over time. The results of this work can offer an alternative approach to study how animal flocks coordinate, create new control schemes for large networked systems, and serve as a central part of defense mechanisms against adversarial drone attacks.



## **10. Stochastic Local Winner-Takes-All Networks Enable Profound Adversarial Robustness**

cs.LG

Bayesian Deep Learning Workshop, NeurIPS 2021

**SubmitDate**: 2021-12-05    [paper-pdf](http://arxiv.org/pdf/2112.02671v1)

**Authors**: Konstantinos P. Panousis, Sotirios Chatzis, Sergios Theodoridis

**Abstracts**: This work explores the potency of stochastic competition-based activations, namely Stochastic Local Winner-Takes-All (LWTA), against powerful (gradient-based) white-box and black-box adversarial attacks; we especially focus on Adversarial Training settings. In our work, we replace the conventional ReLU-based nonlinearities with blocks comprising locally and stochastically competing linear units. The output of each network layer now yields a sparse output, depending on the outcome of winner sampling in each block. We rely on the Variational Bayesian framework for training and inference; we incorporate conventional PGD-based adversarial training arguments to increase the overall adversarial robustness. As we experimentally show, the arising networks yield state-of-the-art robustness against powerful adversarial attacks while retaining very high classification rate in the benign case.



## **11. Formalizing and Estimating Distribution Inference Risks**

cs.LG

Shorter version of work available at arXiv:2106.03699 Update: New  version with more theoretical results and a deeper exploration of results

**SubmitDate**: 2021-12-05    [paper-pdf](http://arxiv.org/pdf/2109.06024v4)

**Authors**: Anshuman Suri, David Evans

**Abstracts**: Distribution inference, sometimes called property inference, infers statistical properties about a training set from access to a model trained on that data. Distribution inference attacks can pose serious risks when models are trained on private data, but are difficult to distinguish from the intrinsic purpose of statistical machine learning -- namely, to produce models that capture statistical properties about a distribution. Motivated by Yeom et al.'s membership inference framework, we propose a formal definition of distribution inference attacks that is general enough to describe a broad class of attacks distinguishing between possible training distributions. We show how our definition captures previous ratio-based property inference attacks as well as new kinds of attack including revealing the average node degree or clustering coefficient of a training graph. To understand distribution inference risks, we introduce a metric that quantifies observed leakage by relating it to the leakage that would occur if samples from the training distribution were provided directly to the adversary. We report on a series of experiments across a range of different distributions using both novel black-box attacks and improved versions of the state-of-the-art white-box attacks. Our results show that inexpensive attacks are often as effective as expensive meta-classifier attacks, and that there are surprising asymmetries in the effectiveness of attacks.



## **12. Adv-4-Adv: Thwarting Changing Adversarial Perturbations via Adversarial Domain Adaptation**

cs.CV

9 pages

**SubmitDate**: 2021-12-04    [paper-pdf](http://arxiv.org/pdf/2112.00428v2)

**Authors**: Tianyue Zheng, Zhe Chen, Shuya Ding, Chao Cai, Jun Luo

**Abstracts**: Whereas adversarial training can be useful against specific adversarial perturbations, they have also proven ineffective in generalizing towards attacks deviating from those used for training. However, we observe that this ineffectiveness is intrinsically connected to domain adaptability, another crucial issue in deep learning for which adversarial domain adaptation appears to be a promising solution. Consequently, we proposed Adv-4-Adv as a novel adversarial training method that aims to retain robustness against unseen adversarial perturbations. Essentially, Adv-4-Adv treats attacks incurring different perturbations as distinct domains, and by leveraging the power of adversarial domain adaptation, it aims to remove the domain/attack-specific features. This forces a trained model to learn a robust domain-invariant representation, which in turn enhances its generalization ability. Extensive evaluations on Fashion-MNIST, SVHN, CIFAR-10, and CIFAR-100 demonstrate that a model trained by Adv-4-Adv based on samples crafted by simple attacks (e.g., FGSM) can be generalized to more advanced attacks (e.g., PGD), and the performance exceeds state-of-the-art proposals on these datasets.



## **13. SoK: Certified Robustness for Deep Neural Networks**

cs.LG

14 pages for the main text

**SubmitDate**: 2021-12-04    [paper-pdf](http://arxiv.org/pdf/2009.04131v3)

**Authors**: Linyi Li, Xiangyu Qi, Tao Xie, Bo Li

**Abstracts**: Great advances in deep neural networks (DNNs) have led to state-of-the-art performance on a wide range of tasks. However, recent studies have shown that DNNs are vulnerable to adversarial attacks, which have brought great concerns when deploying these models to safety-critical applications such as autonomous driving. Different defense approaches have been proposed against adversarial attacks, including: a) empirical defenses, which usually can be adaptively attacked again without providing robustness certification; and b) certifiably robust approaches which consist of robustness verification providing the lower bound of robust accuracy against any attacks under certain conditions and corresponding robust training approaches. In this paper, we systematize the certifiably robust approaches and related practical and theoretical implications and findings. We also provide the first comprehensive benchmark on existing robustness verification and training approaches on different datasets. In particular, we 1) provide a taxonomy for the robustness verification and training approaches, as well as summarize the methodologies for representative algorithms, 2) reveal the characteristics, strengths, limitations, and fundamental connections among these approaches, 3) discuss current research progresses, theoretical barriers, main challenges, and future directions for certifiably robust approaches for DNNs, and 4) provide an open-sourced unified platform to evaluate over 20 representative certifiably robust approaches for a wide range of DNNs.



## **14. Statically Detecting Adversarial Malware through Randomised Chaining**

cs.CR

**SubmitDate**: 2021-12-04    [paper-pdf](http://arxiv.org/pdf/2111.14037v2)

**Authors**: Matthew Crawford, Wei Wang, Ruoxi Sun, Minhui Xue

**Abstracts**: With the rapid growth of malware attacks, more antivirus developers consider deploying machine learning technologies into their productions. Researchers and developers published various machine learning-based detectors with high precision on malware detection in recent years. Although numerous machine learning-based malware detectors are available, they face various machine learning-targeted attacks, including evasion and adversarial attacks. This project explores how and why adversarial examples evade malware detectors, then proposes a randomised chaining method to defend against adversarial malware statically. This research is crucial for working towards combating the pertinent malware cybercrime.



## **15. Generalized Likelihood Ratio Test for Adversarially Robust Hypothesis Testing**

stat.ML

Submitted to the IEEE Transactions on Signal Processing

**SubmitDate**: 2021-12-04    [paper-pdf](http://arxiv.org/pdf/2112.02209v1)

**Authors**: Bhagyashree Puranik, Upamanyu Madhow, Ramtin Pedarsani

**Abstracts**: Machine learning models are known to be susceptible to adversarial attacks which can cause misclassification by introducing small but well designed perturbations. In this paper, we consider a classical hypothesis testing problem in order to develop fundamental insight into defending against such adversarial perturbations. We interpret an adversarial perturbation as a nuisance parameter, and propose a defense based on applying the generalized likelihood ratio test (GLRT) to the resulting composite hypothesis testing problem, jointly estimating the class of interest and the adversarial perturbation. While the GLRT approach is applicable to general multi-class hypothesis testing, we first evaluate it for binary hypothesis testing in white Gaussian noise under $\ell_{\infty}$ norm-bounded adversarial perturbations, for which a known minimax defense optimizing for the worst-case attack provides a benchmark. We derive the worst-case attack for the GLRT defense, and show that its asymptotic performance (as the dimension of the data increases) approaches that of the minimax defense. For non-asymptotic regimes, we show via simulations that the GLRT defense is competitive with the minimax approach under the worst-case attack, while yielding a better robustness-accuracy tradeoff under weaker attacks. We also illustrate the GLRT approach for a multi-class hypothesis testing problem, for which a minimax strategy is not known, evaluating its performance under both noise-agnostic and noise-aware adversarial settings, by providing a method to find optimal noise-aware attacks, and heuristics to find noise-agnostic attacks that are close to optimal in the high SNR regime.



## **16. IRShield: A Countermeasure Against Adversarial Physical-Layer Wireless Sensing**

cs.CR

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2112.01967v1)

**Authors**: Paul Staat, Simon Mulzer, Stefan Roth, Veelasha Moonsamy, Aydin Sezgin, Christof Paar

**Abstracts**: Wireless radio channels are known to contain information about the surrounding propagation environment, which can be extracted using established wireless sensing methods. Thus, today's ubiquitous wireless devices are attractive targets for passive eavesdroppers to launch reconnaissance attacks. In particular, by overhearing standard communication signals, eavesdroppers obtain estimations of wireless channels which can give away sensitive information about indoor environments. For instance, by applying simple statistical methods, adversaries can infer human motion from wireless channel observations, allowing to remotely monitor premises of victims. In this work, building on the advent of intelligent reflecting surfaces (IRSs), we propose IRShield as a novel countermeasure against adversarial wireless sensing. IRShield is designed as a plug-and-play privacy-preserving extension to existing wireless networks. At the core of IRShield, we design an IRS configuration algorithm to obfuscate wireless channels. We validate the effectiveness with extensive experimental evaluations. In a state-of-the-art human motion detection attack using off-the-shelf Wi-Fi devices, IRShield lowered detection rates to 5% or less.



## **17. Mind the box: $l_1$-APGD for sparse adversarial attacks on image classifiers**

cs.LG

In ICML 2021

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2103.01208v2)

**Authors**: Francesco Croce, Matthias Hein

**Abstracts**: We show that when taking into account also the image domain $[0,1]^d$, established $l_1$-projected gradient descent (PGD) attacks are suboptimal as they do not consider that the effective threat model is the intersection of the $l_1$-ball and $[0,1]^d$. We study the expected sparsity of the steepest descent step for this effective threat model and show that the exact projection onto this set is computationally feasible and yields better performance. Moreover, we propose an adaptive form of PGD which is highly effective even with a small budget of iterations. Our resulting $l_1$-APGD is a strong white-box attack showing that prior works overestimated their $l_1$-robustness. Using $l_1$-APGD for adversarial training we get a robust classifier with SOTA $l_1$-robustness. Finally, we combine $l_1$-APGD and an adaptation of the Square Attack to $l_1$ into $l_1$-AutoAttack, an ensemble of attacks which reliably assesses adversarial robustness for the threat model of $l_1$-ball intersected with $[0,1]^d$.



## **18. Graph Neural Networks Inspired by Classical Iterative Algorithms**

cs.LG

accepted as long oral for ICML 2021

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2103.06064v4)

**Authors**: Yongyi Yang, Tang Liu, Yangkun Wang, Jinjing Zhou, Quan Gan, Zhewei Wei, Zheng Zhang, Zengfeng Huang, David Wipf

**Abstracts**: Despite the recent success of graph neural networks (GNN), common architectures often exhibit significant limitations, including sensitivity to oversmoothing, long-range dependencies, and spurious edges, e.g., as can occur as a result of graph heterophily or adversarial attacks. To at least partially address these issues within a simple transparent framework, we consider a new family of GNN layers designed to mimic and integrate the update rules of two classical iterative algorithms, namely, proximal gradient descent and iterative reweighted least squares (IRLS). The former defines an extensible base GNN architecture that is immune to oversmoothing while nonetheless capturing long-range dependencies by allowing arbitrary propagation steps. In contrast, the latter produces a novel attention mechanism that is explicitly anchored to an underlying end-to-end energy function, contributing stability with respect to edge uncertainty. When combined we obtain an extremely simple yet robust model that we evaluate across disparate scenarios including standardized benchmarks, adversarially-perturbated graphs, graphs with heterophily, and graphs involving long-range dependencies. In doing so, we compare against SOTA GNN approaches that have been explicitly designed for the respective task, achieving competitive or superior node classification accuracy. Our code is available at https://github.com/FFTYYY/TWIRLS.



## **19. Blackbox Untargeted Adversarial Testing of Automatic Speech Recognition Systems**

cs.SD

10 pages, 6 figures and 7 tables

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2112.01821v1)

**Authors**: Xiaoliang Wu, Ajitha Rajan

**Abstracts**: Automatic speech recognition (ASR) systems are prevalent, particularly in applications for voice navigation and voice control of domestic appliances. The computational core of ASRs are deep neural networks (DNNs) that have been shown to be susceptible to adversarial perturbations; easily misused by attackers to generate malicious outputs. To help test the correctness of ASRS, we propose techniques that automatically generate blackbox (agnostic to the DNN), untargeted adversarial attacks that are portable across ASRs. Much of the existing work on adversarial ASR testing focuses on targeted attacks, i.e generating audio samples given an output text. Targeted techniques are not portable, customised to the structure of DNNs (whitebox) within a specific ASR. In contrast, our method attacks the signal processing stage of the ASR pipeline that is shared across most ASRs. Additionally, we ensure the generated adversarial audio samples have no human audible difference by manipulating the acoustic signal using a psychoacoustic model that maintains the signal below the thresholds of human perception. We evaluate portability and effectiveness of our techniques using three popular ASRs and three input audio datasets using the metrics - WER of output text, Similarity to original audio and attack Success Rate on different ASRs. We found our testing techniques were portable across ASRs, with the adversarial audio samples producing high Success Rates, WERs and Similarities to the original audio.



## **20. Attack-Centric Approach for Evaluating Transferability of Adversarial Samples in Machine Learning Models**

cs.LG

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2112.01777v1)

**Authors**: Tochukwu Idika, Ismail Akturk

**Abstracts**: Transferability of adversarial samples became a serious concern due to their impact on the reliability of machine learning system deployments, as they find their way into many critical applications. Knowing factors that influence transferability of adversarial samples can assist experts to make informed decisions on how to build robust and reliable machine learning systems. The goal of this study is to provide insights on the mechanisms behind the transferability of adversarial samples through an attack-centric approach. This attack-centric perspective interprets how adversarial samples would transfer by assessing the impact of machine learning attacks (that generated them) on a given input dataset. To achieve this goal, we generated adversarial samples using attacker models and transferred these samples to victim models. We analyzed the behavior of adversarial samples on victim models and outlined four factors that can influence the transferability of adversarial samples. Although these factors are not necessarily exhaustive, they provide useful insights to researchers and practitioners of machine learning systems.



## **21. Single-Shot Black-Box Adversarial Attacks Against Malware Detectors: A Causal Language Model Approach**

cs.CR

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2112.01724v1)

**Authors**: James Lee Hu, Mohammadreza Ebrahimi, Hsinchun Chen

**Abstracts**: Deep Learning (DL)-based malware detectors are increasingly adopted for early detection of malicious behavior in cybersecurity. However, their sensitivity to adversarial malware variants has raised immense security concerns. Generating such adversarial variants by the defender is crucial to improving the resistance of DL-based malware detectors against them. This necessity has given rise to an emerging stream of machine learning research, Adversarial Malware example Generation (AMG), which aims to generate evasive adversarial malware variants that preserve the malicious functionality of a given malware. Within AMG research, black-box method has gained more attention than white-box methods. However, most black-box AMG methods require numerous interactions with the malware detectors to generate adversarial malware examples. Given that most malware detectors enforce a query limit, this could result in generating non-realistic adversarial examples that are likely to be detected in practice due to lack of stealth. In this study, we show that a novel DL-based causal language model enables single-shot evasion (i.e., with only one query to malware detector) by treating the content of the malware executable as a byte sequence and training a Generative Pre-Trained Transformer (GPT). Our proposed method, MalGPT, significantly outperformed the leading benchmark methods on a real-world malware dataset obtained from VirusTotal, achieving over 24.51\% evasion rate. MalGPT enables cybersecurity researchers to develop advanced defense capabilities by emulating large-scale realistic AMG.



## **22. Adversarial Attacks against a Satellite-borne Multispectral Cloud Detector**

cs.CV

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2112.01723v1)

**Authors**: Andrew Du, Yee Wei Law, Michele Sasdelli, Bo Chen, Ken Clarke, Michael Brown, Tat-Jun Chin

**Abstracts**: Data collected by Earth-observing (EO) satellites are often afflicted by cloud cover. Detecting the presence of clouds -- which is increasingly done using deep learning -- is crucial preprocessing in EO applications. In fact, advanced EO satellites perform deep learning-based cloud detection on board the satellites and downlink only clear-sky data to save precious bandwidth. In this paper, we highlight the vulnerability of deep learning-based cloud detection towards adversarial attacks. By optimising an adversarial pattern and superimposing it into a cloudless scene, we bias the neural network into detecting clouds in the scene. Since the input spectra of cloud detectors include the non-visible bands, we generated our attacks in the multispectral domain. This opens up the potential of multi-objective attacks, specifically, adversarial biasing in the cloud-sensitive bands and visual camouflage in the visible bands. We also investigated mitigation strategies against the adversarial attacks. We hope our work further builds awareness of the potential of adversarial attacks in the EO community.



## **23. Is RobustBench/AutoAttack a suitable Benchmark for Adversarial Robustness?**

cs.CV

AAAI-22 AdvML Workshop ShortPaper

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2112.01601v1)

**Authors**: Peter Lorenz, Dominik Strassel, Margret Keuper, Janis Keuper

**Abstracts**: Recently, RobustBench (Croce et al. 2020) has become a widely recognized benchmark for the adversarial robustness of image classification networks. In its most commonly reported sub-task, RobustBench evaluates and ranks the adversarial robustness of trained neural networks on CIFAR10 under AutoAttack (Croce and Hein 2020b) with l-inf perturbations limited to eps = 8/255. With leading scores of the currently best performing models of around 60% of the baseline, it is fair to characterize this benchmark to be quite challenging. Despite its general acceptance in recent literature, we aim to foster discussion about the suitability of RobustBench as a key indicator for robustness which could be generalized to practical applications. Our line of argumentation against this is two-fold and supported by excessive experiments presented in this paper: We argue that I) the alternation of data by AutoAttack with l-inf, eps = 8/255 is unrealistically strong, resulting in close to perfect detection rates of adversarial samples even by simple detection algorithms and human observers. We also show that other attack methods are much harder to detect while achieving similar success rates. II) That results on low-resolution data sets like CIFAR10 do not generalize well to higher resolution images as gradient-based attacks appear to become even more detectable with increasing resolutions.



## **24. Is Approximation Universally Defensive Against Adversarial Attacks in Deep Neural Networks?**

cs.LG

Accepted for publication in DATE 2022

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2112.01555v1)

**Authors**: Ayesha Siddique, Khaza Anuarul Hoque

**Abstracts**: Approximate computing is known for its effectiveness in improvising the energy efficiency of deep neural network (DNN) accelerators at the cost of slight accuracy loss. Very recently, the inexact nature of approximate components, such as approximate multipliers have also been reported successful in defending adversarial attacks on DNNs models. Since the approximation errors traverse through the DNN layers as masked or unmasked, this raises a key research question-can approximate computing always offer a defense against adversarial attacks in DNNs, i.e., are they universally defensive? Towards this, we present an extensive adversarial robustness analysis of different approximate DNN accelerators (AxDNNs) using the state-of-the-art approximate multipliers. In particular, we evaluate the impact of ten adversarial attacks on different AxDNNs using the MNIST and CIFAR-10 datasets. Our results demonstrate that adversarial attacks on AxDNNs can cause 53% accuracy loss whereas the same attack may lead to almost no accuracy loss (as low as 0.06%) in the accurate DNN. Thus, approximate computing cannot be referred to as a universal defense strategy against adversarial attacks.



## **25. FedRAD: Federated Robust Adaptive Distillation**

cs.LG

Accepted for 1st NeurIPS Workshop on New Frontiers in Federated  Learning (NFFL 2021), Virtual Meeting

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2112.01405v1)

**Authors**: Stefán Páll Sturluson, Samuel Trew, Luis Muñoz-González, Matei Grama, Jonathan Passerat-Palmbach, Daniel Rueckert, Amir Alansary

**Abstracts**: The robustness of federated learning (FL) is vital for the distributed training of an accurate global model that is shared among large number of clients. The collaborative learning framework by typically aggregating model updates is vulnerable to model poisoning attacks from adversarial clients. Since the shared information between the global server and participants are only limited to model parameters, it is challenging to detect bad model updates. Moreover, real-world datasets are usually heterogeneous and not independent and identically distributed (Non-IID) among participants, which makes the design of such robust FL pipeline more difficult. In this work, we propose a novel robust aggregation method, Federated Robust Adaptive Distillation (FedRAD), to detect adversaries and robustly aggregate local models based on properties of the median statistic, and then performing an adapted version of ensemble Knowledge Distillation. We run extensive experiments to evaluate the proposed method against recently published works. The results show that FedRAD outperforms all other aggregators in the presence of adversaries, as well as in heterogeneous data distributions.



## **26. A Unified Framework for Adversarial Attack and Defense in Constrained Feature Space**

cs.AI

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2112.01156v1)

**Authors**: Thibault Simonetto, Salijona Dyrmishi, Salah Ghamizi, Maxime Cordy, Yves Le Traon

**Abstracts**: The generation of feasible adversarial examples is necessary for properly assessing models that work on constrained feature space. However, it remains a challenging task to enforce constraints into attacks that were designed for computer vision. We propose a unified framework to generate feasible adversarial examples that satisfy given domain constraints. Our framework supports the use cases reported in the literature and can handle both linear and non-linear constraints. We instantiate our framework into two algorithms: a gradient-based attack that introduces constraints in the loss function to maximize, and a multi-objective search algorithm that aims for misclassification, perturbation minimization, and constraint satisfaction. We show that our approach is effective on two datasets from different domains, with a success rate of up to 100%, where state-of-the-art attacks fail to generate a single feasible example. In addition to adversarial retraining, we propose to introduce engineered non-convex constraints to improve model adversarial robustness. We demonstrate that this new defense is as effective as adversarial retraining. Our framework forms the starting point for research on constrained adversarial attacks and provides relevant baselines and datasets that future research can exploit.



## **27. Adversarial Robustness of Deep Reinforcement Learning based Dynamic Recommender Systems**

cs.LG

arXiv admin note: text overlap with arXiv:2006.07934

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2112.00973v1)

**Authors**: Siyu Wang, Yuanjiang Cao, Xiaocong Chen, Lina Yao, Xianzhi Wang, Quan Z. Sheng

**Abstracts**: Adversarial attacks, e.g., adversarial perturbations of the input and adversarial samples, pose significant challenges to machine learning and deep learning techniques, including interactive recommendation systems. The latent embedding space of those techniques makes adversarial attacks difficult to detect at an early stage. Recent advance in causality shows that counterfactual can also be considered one of ways to generate the adversarial samples drawn from different distribution as the training samples. We propose to explore adversarial examples and attack agnostic detection on reinforcement learning-based interactive recommendation systems. We first craft different types of adversarial examples by adding perturbations to the input and intervening on the casual factors. Then, we augment recommendation systems by detecting potential attacks with a deep learning-based classifier based on the crafted data. Finally, we study the attack strength and frequency of adversarial examples and evaluate our model on standard datasets with multiple crafting methods. Our extensive experiments show that most adversarial attacks are effective, and both attack strength and attack frequency impact the attack performance. The strategically-timed attack achieves comparative attack performance with only 1/3 to 1/2 attack frequency. Besides, our black-box detector trained with one crafting method has the generalization ability over several other crafting methods.



## **28. Learning Task-aware Robust Deep Learning Systems**

cs.LG

9 Pages

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2010.05125v2)

**Authors**: Keji Han, Yun Li, Xianzhong Long, Yao Ge

**Abstracts**: Many works demonstrate that deep learning system is vulnerable to adversarial attack. A deep learning system consists of two parts: the deep learning task and the deep model. Nowadays, most existing works investigate the impact of the deep model on robustness of deep learning systems, ignoring the impact of the learning task. In this paper, we adopt the binary and interval label encoding strategy to redefine the classification task and design corresponding loss to improve robustness of the deep learning system. Our method can be viewed as improving the robustness of deep learning systems from both the learning task and deep model. Experimental results demonstrate that our learning task-aware method is much more robust than traditional classification while retaining the accuracy.



## **29. They See Me Rollin': Inherent Vulnerability of the Rolling Shutter in CMOS Image Sensors**

cs.CV

15 pages, 15 figures

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2101.10011v2)

**Authors**: Sebastian Köhler, Giulio Lovisotto, Simon Birnbach, Richard Baker, Ivan Martinovic

**Abstracts**: In this paper, we describe how the electronic rolling shutter in CMOS image sensors can be exploited using a bright, modulated light source (e.g., an inexpensive, off-the-shelf laser), to inject fine-grained image disruptions. We demonstrate the attack on seven different CMOS cameras, ranging from cheap IoT to semi-professional surveillance cameras, to highlight the wide applicability of the rolling shutter attack. We model the fundamental factors affecting a rolling shutter attack in an uncontrolled setting. We then perform an exhaustive evaluation of the attack's effect on the task of object detection, investigating the effect of attack parameters. We validate our model against empirical data collected on two separate cameras, showing that by simply using information from the camera's datasheet the adversary can accurately predict the injected distortion size and optimize their attack accordingly. We find that an adversary can hide up to 75% of objects perceived by state-of-the-art detectors by selecting appropriate attack parameters. We also investigate the stealthiness of the attack in comparison to a na\"{i}ve camera blinding attack, showing that common image distortion metrics can not detect the attack presence. Therefore, we present a new, accurate and lightweight enhancement to the backbone network of an object detector to recognize rolling shutter attacks. Overall, our results indicate that rolling shutter attacks can substantially reduce the performance and reliability of vision-based intelligent systems.



## **30. Certified Adversarial Defenses Meet Out-of-Distribution Corruptions: Benchmarking Robustness and Simple Baselines**

cs.LG

21 pages, 15 figures, and 9 tables

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2112.00659v1)

**Authors**: Jiachen Sun, Akshay Mehra, Bhavya Kailkhura, Pin-Yu Chen, Dan Hendrycks, Jihun Hamm, Z. Morley Mao

**Abstracts**: Certified robustness guarantee gauges a model's robustness to test-time attacks and can assess the model's readiness for deployment in the real world. In this work, we critically examine how the adversarial robustness guarantees from randomized smoothing-based certification methods change when state-of-the-art certifiably robust models encounter out-of-distribution (OOD) data. Our analysis demonstrates a previously unknown vulnerability of these models to low-frequency OOD data such as weather-related corruptions, rendering these models unfit for deployment in the wild. To alleviate this issue, we propose a novel data augmentation scheme, FourierMix, that produces augmentations to improve the spectral coverage of the training data. Furthermore, we propose a new regularizer that encourages consistent predictions on noise perturbations of the augmented data to improve the quality of the smoothed models. We find that FourierMix augmentations help eliminate the spectral bias of certifiably robust models enabling them to achieve significantly better robustness guarantees on a range of OOD benchmarks. Our evaluation also uncovers the inability of current OOD benchmarks at highlighting the spectral biases of the models. To this end, we propose a comprehensive benchmarking suite that contains corruptions from different regions in the spectral domain. Evaluation of models trained with popular augmentation methods on the proposed suite highlights their spectral biases and establishes the superiority of FourierMix trained models at achieving better-certified robustness guarantees under OOD shifts over the entire frequency spectrum.



## **31. Well-classified Examples are Underestimated in Classification with Deep Neural Networks**

cs.LG

Accepted by AAAI 2022; 16 pages, 11 figures, 13 tables

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2110.06537v3)

**Authors**: Guangxiang Zhao, Wenkai Yang, Xuancheng Ren, Lei Li, Xu Sun

**Abstracts**: The conventional wisdom behind learning deep classification models is to focus on bad-classified examples and ignore well-classified examples that are far from the decision boundary. For instance, when training with cross-entropy loss, examples with higher likelihoods (i.e., well-classified examples) contribute smaller gradients in back-propagation. However, we theoretically show that this common practice hinders representation learning, energy optimization, and the growth of margin. To counteract this deficiency, we propose to reward well-classified examples with additive bonuses to revive their contribution to learning. This counterexample theoretically addresses these three issues. We empirically support this claim by directly verify the theoretical results or through the significant performance improvement with our counterexample on diverse tasks, including image classification, graph classification, and machine translation. Furthermore, this paper shows that because our idea can solve these three issues, we can deal with complex scenarios, such as imbalanced classification, OOD detection, and applications under adversarial attacks. Code is available at: https://github.com/lancopku/well-classified-examples-are-underestimated.



## **32. Understanding Adversarial Attacks on Observations in Deep Reinforcement Learning**

cs.LG

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2106.15860v2)

**Authors**: You Qiaoben, Chengyang Ying, Xinning Zhou, Hang Su, Jun Zhu, Bo Zhang

**Abstracts**: Deep reinforcement learning models are vulnerable to adversarial attacks that can decrease a victim's cumulative expected reward by manipulating the victim's observations. Despite the efficiency of previous optimization-based methods for generating adversarial noise in supervised learning, such methods might not be able to achieve the lowest cumulative reward since they do not explore the environmental dynamics in general. In this paper, we provide a framework to better understand the existing methods by reformulating the problem of adversarial attacks on reinforcement learning in the function space. Our reformulation generates an optimal adversary in the function space of the targeted attacks, repelling them via a generic two-stage framework. In the first stage, we train a deceptive policy by hacking the environment, and discover a set of trajectories routing to the lowest reward or the worst-case performance. Next, the adversary misleads the victim to imitate the deceptive policy by perturbing the observations. Compared to existing approaches, we theoretically show that our adversary is stronger under an appropriate noise level. Extensive experiments demonstrate our method's superiority in terms of efficiency and effectiveness, achieving the state-of-the-art performance in both Atari and MuJoCo environments.



## **33. $\ell_\infty$-Robustness and Beyond: Unleashing Efficient Adversarial Training**

cs.LG

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2112.00378v1)

**Authors**: Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstracts**: Neural networks are vulnerable to adversarial attacks: adding well-crafted, imperceptible perturbations to their input can modify their output. Adversarial training is one of the most effective approaches in training robust models against such attacks. However, it is much slower than vanilla training of neural networks since it needs to construct adversarial examples for the entire training data at every iteration, which has hampered its effectiveness. Recently, Fast Adversarial Training was proposed that can obtain robust models efficiently. However, the reasons behind its success are not fully understood, and more importantly, it can only train robust models for $\ell_\infty$-bounded attacks as it uses FGSM during training. In this paper, by leveraging the theory of coreset selection we show how selecting a small subset of training data provides a more principled approach towards reducing the time complexity of robust training. Unlike existing methods, our approach can be adapted to a wide variety of training objectives, including TRADES, $\ell_p$-PGD, and Perceptual Adversarial Training. Our experimental results indicate that our approach speeds up adversarial training by 2-3 times, while experiencing a small reduction in the clean and robust accuracy.



## **34. Designing a Location Trace Anonymization Contest**

cs.CR

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2107.10407v2)

**Authors**: Takao Murakami, Hiromi Arai, Koki Hamada, Takuma Hatano, Makoto Iguchi, Hiroaki Kikuchi, Atsushi Kuromasa, Hiroshi Nakagawa, Yuichi Nakamura, Kenshiro Nishiyama, Ryo Nojima, Hidenobu Oguri, Chiemi Watanabe, Akira Yamada, Takayasu Yamaguchi, Yuji Yamaoka

**Abstracts**: For a better understanding of anonymization methods for location traces, we have designed and held a location trace anonymization contest. Our contest deals with a long trace (400 events per user) and fine-grained locations (1024 regions). In our contest, each team anonymizes her original traces, and then the other teams perform privacy attacks against the anonymized traces in a partial-knowledge attacker model where the adversary does not know the original traces. To realize such a contest, we propose a location synthesizer that has diversity and utility; the synthesizer generates different synthetic traces for each team while preserving various statistical features of real traces. We also show that re-identification alone is insufficient as a privacy risk and that trace inference should be added as an additional risk. Specifically, we show an example of anonymization that is perfectly secure against re-identification and is not secure against trace inference. Based on this, our contest evaluates both the re-identification risk and trace inference risk and analyzes their relationship. Through our contest, we show several findings in a situation where both defense and attack compete together. In particular, we show that an anonymization method secure against trace inference is also secure against re-identification under the presence of appropriate pseudonymization.



## **35. Push Stricter to Decide Better: A Class-Conditional Feature Adaptive Framework for Improving Adversarial Robustness**

cs.CV

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2112.00323v1)

**Authors**: Jia-Li Yin, Lehui Xie, Wanqing Zhu, Ximeng Liu, Bo-Hao Chen

**Abstracts**: In response to the threat of adversarial examples, adversarial training provides an attractive option for enhancing the model robustness by training models on online-augmented adversarial examples. However, most of the existing adversarial training methods focus on improving the robust accuracy by strengthening the adversarial examples but neglecting the increasing shift between natural data and adversarial examples, leading to a dramatic decrease in natural accuracy. To maintain the trade-off between natural and robust accuracy, we alleviate the shift from the perspective of feature adaption and propose a Feature Adaptive Adversarial Training (FAAT) optimizing the class-conditional feature adaption across natural data and adversarial examples. Specifically, we propose to incorporate a class-conditional discriminator to encourage the features become (1) class-discriminative and (2) invariant to the change of adversarial attacks. The novel FAAT framework enables the trade-off between natural and robust accuracy by generating features with similar distribution across natural and adversarial data, and achieve higher overall robustness benefited from the class-discriminative feature characteristics. Experiments on various datasets demonstrate that FAAT produces more discriminative features and performs favorably against state-of-the-art methods. Codes are available at https://github.com/VisionFlow/FAAT.



## **36. Adversarial Attacks Against Deep Generative Models on Data: A Survey**

cs.CR

To be published in IEEE Transactions on Knowledge and Data  Engineering

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2112.00247v1)

**Authors**: Hui Sun, Tianqing Zhu, Zhiqiu Zhang, Dawei Jin. Ping Xiong, Wanlei Zhou

**Abstracts**: Deep generative models have gained much attention given their ability to generate data for applications as varied as healthcare to financial technology to surveillance, and many more - the most popular models being generative adversarial networks and variational auto-encoders. Yet, as with all machine learning models, ever is the concern over security breaches and privacy leaks and deep generative models are no exception. These models have advanced so rapidly in recent years that work on their security is still in its infancy. In an attempt to audit the current and future threats against these models, and to provide a roadmap for defense preparations in the short term, we prepared this comprehensive and specialized survey on the security and privacy preservation of GANs and VAEs. Our focus is on the inner connection between attacks and model architectures and, more specifically, on five components of deep generative models: the training data, the latent code, the generators/decoders of GANs/ VAEs, the discriminators/encoders of GANs/ VAEs, and the generated data. For each model, component and attack, we review the current research progress and identify the key challenges. The paper concludes with a discussion of possible future attacks and research directions in the field.



## **37. Model Extraction Attacks on Graph Neural Networks: Taxonomy and Realization**

cs.LG

This paper has been published in the 17th ACM ASIA Conference on  Computer and Communications Security (ACM ASIACCS 2022)

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2010.12751v2)

**Authors**: Bang Wu, Xiangwen Yang, Shirui Pan, Xingliang Yuan

**Abstracts**: Machine learning models are shown to face a severe threat from Model Extraction Attacks, where a well-trained private model owned by a service provider can be stolen by an attacker pretending as a client. Unfortunately, prior works focus on the models trained over the Euclidean space, e.g., images and texts, while how to extract a GNN model that contains a graph structure and node features is yet to be explored. In this paper, for the first time, we comprehensively investigate and develop model extraction attacks against GNN models. We first systematically formalise the threat modelling in the context of GNN model extraction and classify the adversarial threats into seven categories by considering different background knowledge of the attacker, e.g., attributes and/or neighbour connections of the nodes obtained by the attacker. Then we present detailed methods which utilise the accessible knowledge in each threat to implement the attacks. By evaluating over three real-world datasets, our attacks are shown to extract duplicated models effectively, i.e., 84% - 89% of the inputs in the target domain have the same output predictions as the victim model.



## **38. Robust Multiple-Path Orienteering Problem: Securing Against Adversarial Attacks**

cs.RO

submitted to TRO

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2003.13896v3)

**Authors**: Guangyao Shi, Lifeng Zhou, Pratap Tokekar

**Abstracts**: The multiple-path orienteering problem asks for paths for a team of robots that maximize the total reward collected while satisfying budget constraints on the path length. This problem models many multi-robot routing tasks such as exploring unknown environments and information gathering for environmental monitoring. In this paper, we focus on how to make the robot team robust to failures when operating in adversarial environments. We introduce the Robust Multiple-path Orienteering Problem (RMOP) where we seek worst-case guarantees against an adversary that is capable of attacking at most $\alpha$ robots. We consider two versions of this problem: RMOP offline and RMOP online. In the offline version, there is no communication or replanning when robots execute their plans and our main contribution is a general approximation scheme with a bounded approximation guarantee that depends on $\alpha$ and the approximation factor for single robot orienteering. In particular, we show that the algorithm yields a (i) constant-factor approximation when the cost function is modular; (ii) $\log$ factor approximation when the cost function is submodular; and (iii) constant-factor approximation when the cost function is submodular but the robots are allowed to exceed their path budgets by a bounded amount. In the online version, RMOP is modeled as a two-player sequential game and solved adaptively in a receding horizon fashion based on Monte Carlo Tree Search (MCTS). In addition to theoretical analysis, we perform simulation studies for ocean monitoring and tunnel information-gathering applications to demonstrate the efficacy of our approach.



## **39. Trustworthy Medical Segmentation with Uncertainty Estimation**

eess.IV

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.05978v2)

**Authors**: Giuseppina Carannante, Dimah Dera, Nidhal C. Bouaynaya, Ghulam Rasool, Hassan M. Fathallah-Shaykh

**Abstracts**: Deep Learning (DL) holds great promise in reshaping the healthcare systems given its precision, efficiency, and objectivity. However, the brittleness of DL models to noisy and out-of-distribution inputs is ailing their deployment in the clinic. Most systems produce point estimates without further information about model uncertainty or confidence. This paper introduces a new Bayesian deep learning framework for uncertainty quantification in segmentation neural networks, specifically encoder-decoder architectures. The proposed framework uses the first-order Taylor series approximation to propagate and learn the first two moments (mean and covariance) of the distribution of the model parameters given the training data by maximizing the evidence lower bound. The output consists of two maps: the segmented image and the uncertainty map of the segmentation. The uncertainty in the segmentation decisions is captured by the covariance matrix of the predictive distribution. We evaluate the proposed framework on medical image segmentation data from Magnetic Resonances Imaging and Computed Tomography scans. Our experiments on multiple benchmark datasets demonstrate that the proposed framework is more robust to noise and adversarial attacks as compared to state-of-the-art segmentation models. Moreover, the uncertainty map of the proposed framework associates low confidence (or equivalently high uncertainty) to patches in the test input images that are corrupted with noise, artifacts or adversarial attacks. Thus, the model can self-assess its segmentation decisions when it makes an erroneous prediction or misses part of the segmentation structures, e.g., tumor, by presenting higher values in the uncertainty map.



## **40. Defending Against Adversarial Denial-of-Service Data Poisoning Attacks**

cs.CR

Published at ACSAC DYNAMICS 2020

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2104.06744v3)

**Authors**: Nicolas M. Müller, Simon Roschmann, Konstantin Böttinger

**Abstracts**: Data poisoning is one of the most relevant security threats against machine learning and data-driven technologies. Since many applications rely on untrusted training data, an attacker can easily craft malicious samples and inject them into the training dataset to degrade the performance of machine learning models. As recent work has shown, such Denial-of-Service (DoS) data poisoning attacks are highly effective. To mitigate this threat, we propose a new approach of detecting DoS poisoned instances. In comparison to related work, we deviate from clustering and anomaly detection based approaches, which often suffer from the curse of dimensionality and arbitrary anomaly threshold selection. Rather, our defence is based on extracting information from the training data in such a generalized manner that we can identify poisoned samples based on the information present in the unpoisoned portion of the data. We evaluate our defence against two DoS poisoning attacks and seven datasets, and find that it reliably identifies poisoned instances. In comparison to related work, our defence improves false positive / false negative rates by at least 50%, often more.



## **41. FROB: Few-shot ROBust Model for Classification and Out-of-Distribution Detection**

cs.LG

Paper, 22 pages, Figures, Tables

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.15487v1)

**Authors**: Nikolaos Dionelis

**Abstracts**: Nowadays, classification and Out-of-Distribution (OoD) detection in the few-shot setting remain challenging aims due to rarity and the limited samples in the few-shot setting, and because of adversarial attacks. Accomplishing these aims is important for critical systems in safety, security, and defence. In parallel, OoD detection is challenging since deep neural network classifiers set high confidence to OoD samples away from the training data. To address such limitations, we propose the Few-shot ROBust (FROB) model for classification and few-shot OoD detection. We devise FROB for improved robustness and reliable confidence prediction for few-shot OoD detection. We generate the support boundary of the normal class distribution and combine it with few-shot Outlier Exposure (OE). We propose a self-supervised learning few-shot confidence boundary methodology based on generative and discriminative models. The contribution of FROB is the combination of the generated boundary in a self-supervised learning manner and the imposition of low confidence at this learned boundary. FROB implicitly generates strong adversarial samples on the boundary and forces samples from OoD, including our boundary, to be less confident by the classifier. FROB achieves generalization to unseen OoD with applicability to unknown, in the wild, test sets that do not correlate to the training datasets. To improve robustness, FROB redesigns OE to work even for zero-shots. By including our boundary, FROB reduces the threshold linked to the model's few-shot robustness; it maintains the OoD performance approximately independent of the number of few-shots. The few-shot robustness analysis evaluation of FROB on different sets and on One-Class Classification (OCC) data shows that FROB achieves competitive performance and outperforms benchmarks in terms of robustness to the outlier few-shot sample population and variability.



## **42. A Face Recognition System's Worst Morph Nightmare, Theoretically**

cs.CV

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.15416v1)

**Authors**: Una M. Kelly, Raymond Veldhuis, Luuk Spreeuwers

**Abstracts**: It has been shown that Face Recognition Systems (FRSs) are vulnerable to morphing attacks, but most research focusses on landmark-based morphs. A second method for generating morphs uses Generative Adversarial Networks, which results in convincingly real facial images that can be almost as challenging for FRSs as landmark-based attacks. We propose a method to create a third, different type of morph, that has the advantage of being easier to train. We introduce the theoretical concept of \textit{worst-case morphs}, which are those morphs that are most challenging for a fixed FRS. For a set of images and corresponding embeddings in an FRS's latent space, we generate images that approximate these worst-case morphs using a mapping from embedding space back to image space. While the resulting images are not yet as challenging as other morphs, they can provide valuable information in future research on Morphing Attack Detection (MAD) methods and on weaknesses of FRSs. Methods for MAD need to be validated on more varied morph databases. Our proposed method contributes to achieving such variation.



## **43. Medical Aegis: Robust adversarial protectors for medical images**

cs.CV

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.10969v2)

**Authors**: Qingsong Yao, Zecheng He, S. Kevin Zhou

**Abstracts**: Deep neural network based medical image systems are vulnerable to adversarial examples. Many defense mechanisms have been proposed in the literature, however, the existing defenses assume a passive attacker who knows little about the defense system and does not change the attack strategy according to the defense. Recent works have shown that a strong adaptive attack, where an attacker is assumed to have full knowledge about the defense system, can easily bypass the existing defenses. In this paper, we propose a novel adversarial example defense system called Medical Aegis. To the best of our knowledge, Medical Aegis is the first defense in the literature that successfully addresses the strong adaptive adversarial example attacks to medical images. Medical Aegis boasts two-tier protectors: The first tier of Cushion weakens the adversarial manipulation capability of an attack by removing its high-frequency components, yet posing a minimal effect on classification performance of the original image; the second tier of Shield learns a set of per-class DNN models to predict the logits of the protected model. Deviation from the Shield's prediction indicates adversarial examples. Shield is inspired by the observations in our stress tests that there exist robust trails in the shallow layers of a DNN model, which the adaptive attacks can hardly destruct. Experimental results show that the proposed defense accurately detects adaptive attacks, with negligible overhead for model inference.



## **44. COREATTACK: Breaking Up the Core Structure of Graphs**

cs.SI

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.15276v1)

**Authors**: Bo Zhou, Yuqian Lv, Jinhuan Wang, Jian Zhang, Qi Xuan

**Abstracts**: The concept of k-core in complex networks plays a key role in many applications, e.g., understanding the global structure, or identifying central/critical nodes, of a network. A malicious attacker with jamming ability can exploit the vulnerability of the k-core structure to attack the network and invalidate the network analysis methods, e.g., reducing the k-shell values of nodes can deceive graph algorithms, leading to the wrong decisions. In this paper, we investigate the robustness of the k-core structure under adversarial attacks by deleting edges, for the first time. Firstly, we give the general definition of targeted k-core attack, map it to the set cover problem which is NP-hard, and further introduce a series of evaluation metrics to measure the performance of attack methods. Then, we propose $Q$ index theoretically as the probability that the terminal node of an edge does not belong to the innermost core, which is further used to guide the design of our heuristic attack methods, namely COREATTACK and GreedyCOREATTACK. The experiments on a variety of real-world networks demonstrate that our methods behave much better than a series of baselines, in terms of much smaller Edge Change Rate (ECR) and False Attack Rate (FAR), achieving state-of-the-art attack performance. More impressively, for certain real-world networks, only deleting one edge from the k-core may lead to the collapse of the innermost core, even if this core contains dozens of nodes. Such a phenomenon indicates that the k-core structure could be extremely vulnerable under adversarial attacks, and its robustness thus should be carefully addressed to ensure the security of many graph algorithms.



## **45. Black-box Adversarial Attacks on Commercial Speech Platforms with Minimal Information**

cs.CR

A version of this paper appears in the proceedings of the 28th ACM  Conference on Computer and Communications Security (CCS 2021). The notes in  Tables 1 and 4 have been updated

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2110.09714v2)

**Authors**: Baolin Zheng, Peipei Jiang, Qian Wang, Qi Li, Chao Shen, Cong Wang, Yunjie Ge, Qingyang Teng, Shenyi Zhang

**Abstracts**: Adversarial attacks against commercial black-box speech platforms, including cloud speech APIs and voice control devices, have received little attention until recent years. The current "black-box" attacks all heavily rely on the knowledge of prediction/confidence scores to craft effective adversarial examples, which can be intuitively defended by service providers without returning these messages. In this paper, we propose two novel adversarial attacks in more practical and rigorous scenarios. For commercial cloud speech APIs, we propose Occam, a decision-only black-box adversarial attack, where only final decisions are available to the adversary. In Occam, we formulate the decision-only AE generation as a discontinuous large-scale global optimization problem, and solve it by adaptively decomposing this complicated problem into a set of sub-problems and cooperatively optimizing each one. Our Occam is a one-size-fits-all approach, which achieves 100% success rates of attacks with an average SNR of 14.23dB, on a wide range of popular speech and speaker recognition APIs, including Google, Alibaba, Microsoft, Tencent, iFlytek, and Jingdong, outperforming the state-of-the-art black-box attacks. For commercial voice control devices, we propose NI-Occam, the first non-interactive physical adversarial attack, where the adversary does not need to query the oracle and has no access to its internal information and training data. We combine adversarial attacks with model inversion attacks, and thus generate the physically-effective audio AEs with high transferability without any interaction with target devices. Our experimental results show that NI-Occam can successfully fool Apple Siri, Microsoft Cortana, Google Assistant, iFlytek and Amazon Echo with an average SRoA of 52% and SNR of 9.65dB, shedding light on non-interactive physical attacks against voice control devices.



## **46. Mitigating Adversarial Attacks by Distributing Different Copies to Different Users**

cs.CR

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.15160v1)

**Authors**: Jiyi Zhang, Wesley Joon-Wie Tann, Ee-Chien Chang

**Abstracts**: Machine learning models are vulnerable to adversarial attacks. In this paper, we consider the scenario where a model is to be distributed to multiple users, among which a malicious user attempts to attack another user. The malicious user probes its copy of the model to search for adversarial samples and then presents the found samples to the victim's model in order to replicate the attack. We point out that by distributing different copies of the model to different users, we can mitigate the attack such that adversarial samples found on one copy would not work on another copy. We first observed that training a model with different randomness indeed mitigates such replication to certain degree. However, there is no guarantee and retraining is computationally expensive. Next, we propose a flexible parameter rewriting method that directly modifies the model's parameters. This method does not require additional training and is able to induce different sets of adversarial samples in different copies in a more controllable manner. Experimentation studies show that our approach can significantly mitigate the attacks while retaining high classification accuracy. From this study, we believe that there are many further directions worth exploring.



## **47. Adversarial Robustness of Deep Code Comment Generation**

cs.SE

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2108.00213v3)

**Authors**: Yu Zhou, Xiaoqing Zhang, Juanjuan Shen, Tingting Han, Taolue Chen, Harald Gall

**Abstracts**: Deep neural networks (DNNs) have shown remarkable performance in a variety of domains such as computer vision, speech recognition, or natural language processing. Recently they also have been applied to various software engineering tasks, typically involving processing source code. DNNs are well-known to be vulnerable to adversarial examples, i.e., fabricated inputs that could lead to various misbehaviors of the DNN model while being perceived as benign by humans. In this paper, we focus on the code comment generation task in software engineering and study the robustness issue of the DNNs when they are applied to this task. We propose ACCENT, an identifier substitution approach to craft adversarial code snippets, which are syntactically correct and semantically close to the original code snippet, but may mislead the DNNs to produce completely irrelevant code comments. In order to improve the robustness, ACCENT also incorporates a novel training method, which can be applied to existing code comment generation models. We conduct comprehensive experiments to evaluate our approach by attacking the mainstream encoder-decoder architectures on two large-scale publicly available datasets. The results show that ACCENT efficiently produces stable attacks with functionality-preserving adversarial examples, and the generated examples have better transferability compared with baselines. We also confirm, via experiments, the effectiveness in improving model robustness with our training method.



## **48. Living-Off-The-Land Command Detection Using Active Learning**

cs.CR

14 pages, published in RAID 2021

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.15039v1)

**Authors**: Talha Ongun, Jack W. Stokes, Jonathan Bar Or, Ke Tian, Farid Tajaddodianfar, Joshua Neil, Christian Seifert, Alina Oprea, John C. Platt

**Abstracts**: In recent years, enterprises have been targeted by advanced adversaries who leverage creative ways to infiltrate their systems and move laterally to gain access to critical data. One increasingly common evasive method is to hide the malicious activity behind a benign program by using tools that are already installed on user computers. These programs are usually part of the operating system distribution or another user-installed binary, therefore this type of attack is called "Living-Off-The-Land". Detecting these attacks is challenging, as adversaries may not create malicious files on the victim computers and anti-virus scans fail to detect them. We propose the design of an Active Learning framework called LOLAL for detecting Living-Off-the-Land attacks that iteratively selects a set of uncertain and anomalous samples for labeling by a human analyst. LOLAL is specifically designed to work well when a limited number of labeled samples are available for training machine learning models to detect attacks. We investigate methods to represent command-line text using word-embedding techniques, and design ensemble boosting classifiers to distinguish malicious and benign samples based on the embedding representation. We leverage a large, anonymized dataset collected by an endpoint security product and demonstrate that our ensemble classifiers achieve an average F1 score of 0.96 at classifying different attack classes. We show that our active learning method consistently improves the classifier performance, as more training data is labeled, and converges in less than 30 iterations when starting with a small number of labeled instances.



## **49. Understanding and Increasing Efficiency of Frank-Wolfe Adversarial Training**

cs.LG

Accepted to ICML 2021 Adversarial Machine Learning Workshop. Under  review

**SubmitDate**: 2021-11-29    [paper-pdf](http://arxiv.org/pdf/2012.12368v4)

**Authors**: Theodoros Tsiligkaridis, Jay Roberts

**Abstracts**: Deep neural networks are easily fooled by small perturbations known as adversarial attacks. Adversarial Training (AT) is a technique that approximately solves a robust optimization problem to minimize the worst-case loss and is widely regarded as the most effective defense against such attacks. Due to the high computation time for generating strong adversarial examples for AT, single-step approaches have been proposed to reduce training time. However, these methods suffer from catastrophic overfitting where adversarial accuracy drops during training. Although improvements have been proposed, they increase training time and robustness is far from that of multi-step AT. We develop a theoretical framework for adversarial training with FW optimization (FW-AT) that reveals a geometric connection between the loss landscape and the $\ell_2$ distortion of $\ell_\infty$ FW attacks. We analytically show that high distortion of FW attacks is equivalent to small gradient variation along the attack path. It is then experimentally demonstrated on various deep neural network architectures that $\ell_\infty$ attacks against robust models achieve near maximal $\ell_2$ distortion, while standard networks have lower distortion. Furthermore, it is experimentally shown that catastrophic overfitting is strongly correlated with low distortion of FW attacks. To demonstrate the utility of our theoretical framework we develop FW-AT-Adapt, a novel adversarial training algorithm which uses a simple distortion measure to adapt the number of attack steps to increase efficiency without compromising robustness. FW-AT-Adapt provides training times on par with single-step fast AT methods and improves closing the gap between fast AT methods and multi-step PGD-AT with minimal loss in adversarial accuracy in white-box and black-box settings.



## **50. MedRDF: A Robust and Retrain-Less Diagnostic Framework for Medical Pretrained Models Against Adversarial Attack**

cs.CV

TMI under review

**SubmitDate**: 2021-11-29    [paper-pdf](http://arxiv.org/pdf/2111.14564v1)

**Authors**: Mengting Xu, Tao Zhang, Daoqiang Zhang

**Abstracts**: Deep neural networks are discovered to be non-robust when attacked by imperceptible adversarial examples, which is dangerous for it applied into medical diagnostic system that requires high reliability. However, the defense methods that have good effect in natural images may not be suitable for medical diagnostic tasks. The preprocessing methods (e.g., random resizing, compression) may lead to the loss of the small lesions feature in the medical image. Retraining the network on the augmented data set is also not practical for medical models that have already been deployed online. Accordingly, it is necessary to design an easy-to-deploy and effective defense framework for medical diagnostic tasks. In this paper, we propose a Robust and Retrain-Less Diagnostic Framework for Medical pretrained models against adversarial attack (i.e., MedRDF). It acts on the inference time of the pertained medical model. Specifically, for each test image, MedRDF firstly creates a large number of noisy copies of it, and obtains the output labels of these copies from the pretrained medical diagnostic model. Then, based on the labels of these copies, MedRDF outputs the final robust diagnostic result by majority voting. In addition to the diagnostic result, MedRDF produces the Robust Metric (RM) as the confidence of the result. Therefore, it is convenient and reliable to utilize MedRDF to convert pre-trained non-robust diagnostic models into robust ones. The experimental results on COVID-19 and DermaMNIST datasets verify the effectiveness of our MedRDF in improving the robustness of medical diagnostic models.



