# Latest Adversarial Attack Papers
**update at 2022-12-26 13:09:42**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Generate synthetic samples from tabular data**

cs.LG

**SubmitDate**: 2022-12-23    [abs](http://arxiv.org/abs/2209.06113v2) [paper-pdf](http://arxiv.org/pdf/2209.06113v2)

**Authors**: David Banh, Alan Huang

**Abstract**: Generating new samples from data sets can mitigate extra expensive operations, increased invasive procedures, and mitigate privacy issues. These novel samples that are statistically robust can be used as a temporary and intermediate replacement when privacy is a concern. This method can enable better data sharing practices without problems relating to identification issues or biases that are flaws for an adversarial attack.



## **2. Learned Systems Security**

cs.CR

**SubmitDate**: 2022-12-22    [abs](http://arxiv.org/abs/2212.10318v2) [paper-pdf](http://arxiv.org/pdf/2212.10318v2)

**Authors**: Roei Schuster, Jin Peng Zhou, Thorsten Eisenhofer, Paul Grubbs, Nicolas Papernot

**Abstract**: A learned system uses machine learning (ML) internally to improve performance. We can expect such systems to be vulnerable to some adversarial-ML attacks. Often, the learned component is shared between mutually-distrusting users or processes, much like microarchitectural resources such as caches, potentially giving rise to highly-realistic attacker models. However, compared to attacks on other ML-based systems, attackers face a level of indirection as they cannot interact directly with the learned model. Additionally, the difference between the attack surface of learned and non-learned versions of the same system is often subtle. These factors obfuscate the de-facto risks that the incorporation of ML carries. We analyze the root causes of potentially-increased attack surface in learned systems and develop a framework for identifying vulnerabilities that stem from the use of ML. We apply our framework to a broad set of learned systems under active development. To empirically validate the many vulnerabilities surfaced by our framework, we choose 3 of them and implement and evaluate exploits against prominent learned-system instances. We show that the use of ML caused leakage of past queries in a database, enabled a poisoning attack that causes exponential memory blowup in an index structure and crashes it in seconds, and enabled index users to snoop on each others' key distributions by timing queries over their own keys. We find that adversarial ML is a universal threat against learned systems, point to open research gaps in our understanding of learned-systems security, and conclude by discussing mitigations, while noting that data leakage is inherent in systems whose learned component is shared between multiple parties.



## **3. GAN-based Domain Inference Attack**

cs.LG

accepted by AAAI23

**SubmitDate**: 2022-12-22    [abs](http://arxiv.org/abs/2212.11810v1) [paper-pdf](http://arxiv.org/pdf/2212.11810v1)

**Authors**: Yuechun Gu, Keke Chen

**Abstract**: Model-based attacks can infer training data information from deep neural network models. These attacks heavily depend on the attacker's knowledge of the application domain, e.g., using it to determine the auxiliary data for model-inversion attacks. However, attackers may not know what the model is used for in practice. We propose a generative adversarial network (GAN) based method to explore likely or similar domains of a target model -- the model domain inference (MDI) attack. For a given target (classification) model, we assume that the attacker knows nothing but the input and output formats and can use the model to derive the prediction for any input in the desired form. Our basic idea is to use the target model to affect a GAN training process for a candidate domain's dataset that is easy to obtain. We find that the target model may distract the training procedure less if the domain is more similar to the target domain. We then measure the distraction level with the distance between GAN-generated datasets, which can be used to rank candidate domains for the target model. Our experiments show that the auxiliary dataset from an MDI top-ranked domain can effectively boost the result of model-inversion attacks.



## **4. Adversarial Machine Learning and Defense Game for NextG Signal Classification with Deep Learning**

cs.NI

**SubmitDate**: 2022-12-22    [abs](http://arxiv.org/abs/2212.11778v1) [paper-pdf](http://arxiv.org/pdf/2212.11778v1)

**Authors**: Yalin E. Sagduyu

**Abstract**: This paper presents a game-theoretic framework to study the interactions of attack and defense for deep learning-based NextG signal classification. NextG systems such as the one envisioned for a massive number of IoT devices can employ deep neural networks (DNNs) for various tasks such as user equipment identification, physical layer authentication, and detection of incumbent users (such as in the Citizens Broadband Radio Service (CBRS) band). By training another DNN as the surrogate model, an adversary can launch an inference (exploratory) attack to learn the behavior of the victim model, predict successful operation modes (e.g., channel access), and jam them. A defense mechanism can increase the adversary's uncertainty by introducing controlled errors in the victim model's decisions (i.e., poisoning the adversary's training data). This defense is effective against an attack but reduces the performance when there is no attack. The interactions between the defender and the adversary are formulated as a non-cooperative game, where the defender selects the probability of defending or the defense level itself (i.e., the ratio of falsified decisions) and the adversary selects the probability of attacking. The defender's objective is to maximize its reward (e.g., throughput or transmission success ratio), whereas the adversary's objective is to minimize this reward and its attack cost. The Nash equilibrium strategies are determined as operation modes such that no player can unilaterally improve its utility given the other's strategy is fixed. A fictitious play is formulated for each player to play the game repeatedly in response to the empirical frequency of the opponent's actions. The performance in Nash equilibrium is compared to the fixed attack and defense cases, and the resilience of NextG signal classification against attacks is quantified.



## **5. Aliasing is a Driver of Adversarial Attacks**

cs.CV

14 pages, 9 figures, 4 tables

**SubmitDate**: 2022-12-22    [abs](http://arxiv.org/abs/2212.11760v1) [paper-pdf](http://arxiv.org/pdf/2212.11760v1)

**Authors**: Adrián Rodríguez-Muñoz, Antonio Torralba

**Abstract**: Aliasing is a highly important concept in signal processing, as careful consideration of resolution changes is essential in ensuring transmission and processing quality of audio, image, and video. Despite this, up until recently aliasing has received very little consideration in Deep Learning, with all common architectures carelessly sub-sampling without considering aliasing effects. In this work, we investigate the hypothesis that the existence of adversarial perturbations is due in part to aliasing in neural networks. Our ultimate goal is to increase robustness against adversarial attacks using explainable, non-trained, structural changes only, derived from aliasing first principles. Our contributions are the following. First, we establish a sufficient condition for no aliasing for general image transformations. Next, we study sources of aliasing in common neural network layers, and derive simple modifications from first principles to eliminate or reduce it. Lastly, our experimental results show a solid link between anti-aliasing and adversarial attacks. Simply reducing aliasing already results in more robust classifiers, and combining anti-aliasing with robust training out-performs solo robust training on $L_2$ attacks with none or minimal losses in performance on $L_{\infty}$ attacks.



## **6. A Theoretical Study of The Effects of Adversarial Attacks on Sparse Regression**

cs.LG

first version

**SubmitDate**: 2022-12-22    [abs](http://arxiv.org/abs/2212.11209v2) [paper-pdf](http://arxiv.org/pdf/2212.11209v2)

**Authors**: Deepak Maurya, Jean Honorio

**Abstract**: This paper analyzes $\ell_1$ regularized linear regression under the challenging scenario of having only adversarially corrupted data for training. We use the primal-dual witness paradigm to provide provable performance guarantees for the support of the estimated regression parameter vector to match the actual parameter. Our theoretical analysis shows the counter-intuitive result that an adversary can influence sample complexity by corrupting the irrelevant features, i.e., those corresponding to zero coefficients of the regression parameter vector, which, consequently, do not affect the dependent variable. As any adversarially robust algorithm has its limitations, our theoretical analysis identifies the regimes under which the learning algorithm and adversary can dominate over each other. It helps us to analyze these fundamental limits and address critical scientific questions of which parameters (like mutual incoherence, the maximum and minimum eigenvalue of the covariance matrix, and the budget of adversarial perturbation) play a role in the high or low probability of success of the LASSO algorithm. Also, the derived sample complexity is logarithmic with respect to the size of the regression parameter vector, and our theoretical claims are validated by empirical analysis on synthetic and real-world datasets.



## **7. Suppress with a Patch: Revisiting Universal Adversarial Patch Attacks against Object Detection**

cs.CV

Accepted for publication at ICECCME 2022

**SubmitDate**: 2022-12-22    [abs](http://arxiv.org/abs/2209.13353v2) [paper-pdf](http://arxiv.org/pdf/2209.13353v2)

**Authors**: Svetlana Pavlitskaya, Jonas Hendl, Sebastian Kleim, Leopold Müller, Fabian Wylczoch, J. Marius Zöllner

**Abstract**: Adversarial patch-based attacks aim to fool a neural network with an intentionally generated noise, which is concentrated in a particular region of an input image. In this work, we perform an in-depth analysis of different patch generation parameters, including initialization, patch size, and especially positioning a patch in an image during training. We focus on the object vanishing attack and run experiments with YOLOv3 as a model under attack in a white-box setting and use images from the COCO dataset. Our experiments have shown, that inserting a patch inside a window of increasing size during training leads to a significant increase in attack strength compared to a fixed position. The best results were obtained when a patch was positioned randomly during training, while patch position additionally varied within a batch.



## **8. Vulnerabilities of Deep Learning-Driven Semantic Communications to Backdoor (Trojan) Attacks**

cs.CR

**SubmitDate**: 2022-12-21    [abs](http://arxiv.org/abs/2212.11205v1) [paper-pdf](http://arxiv.org/pdf/2212.11205v1)

**Authors**: Yalin E. Sagduyu, Tugba Erpek, Sennur Ulukus, Aylin Yener

**Abstract**: This paper highlights vulnerabilities of deep learning-driven semantic communications to backdoor (Trojan) attacks. Semantic communications aims to convey a desired meaning while transferring information from a transmitter to its receiver. An encoder-decoder pair that is represented by two deep neural networks (DNNs) as part of an autoencoder is trained to reconstruct signals such as images at the receiver by transmitting latent features of small size over a limited number of channel uses. In the meantime, another DNN of a semantic task classifier at the receiver is jointly trained with the autoencoder to check the meaning conveyed to the receiver. The complex decision space of the DNNs makes semantic communications susceptible to adversarial manipulations. In a backdoor (Trojan) attack, the adversary adds triggers to a small portion of training samples and changes the label to a target label. When the transfer of images is considered, the triggers can be added to the images or equivalently to the corresponding transmitted or received signals. In test time, the adversary activates these triggers by providing poisoned samples as input to the encoder (or decoder) of semantic communications. The backdoor attack can effectively change the semantic information transferred for the poisoned input samples to a target meaning. As the performance of semantic communications improves with the signal-to-noise ratio and the number of channel uses, the success of the backdoor attack increases as well. Also, increasing the Trojan ratio in training data makes the attack more successful. In the meantime, the effect of this attack on the unpoisoned input samples remains limited. Overall, this paper shows that the backdoor attack poses a serious threat to semantic communications and presents novel design guidelines to preserve the meaning of transferred information in the presence of backdoor attacks.



## **9. Revisiting Residual Networks for Adversarial Robustness: An Architectural Perspective**

cs.CV

**SubmitDate**: 2022-12-21    [abs](http://arxiv.org/abs/2212.11005v1) [paper-pdf](http://arxiv.org/pdf/2212.11005v1)

**Authors**: Shihua Huang, Zhichao Lu, Kalyanmoy Deb, Vishnu Naresh Boddeti

**Abstract**: Efforts to improve the adversarial robustness of convolutional neural networks have primarily focused on developing more effective adversarial training methods. In contrast, little attention was devoted to analyzing the role of architectural elements (such as topology, depth, and width) on adversarial robustness. This paper seeks to bridge this gap and present a holistic study on the impact of architectural design on adversarial robustness. We focus on residual networks and consider architecture design at the block level, i.e., topology, kernel size, activation, and normalization, as well as at the network scaling level, i.e., depth and width of each block in the network. In both cases, we first derive insights through systematic ablative experiments. Then we design a robust residual block, dubbed RobustResBlock, and a compound scaling rule, dubbed RobustScaling, to distribute depth and width at the desired FLOP count. Finally, we combine RobustResBlock and RobustScaling and present a portfolio of adversarially robust residual networks, RobustResNets, spanning a broad spectrum of model capacities. Experimental validation across multiple datasets and adversarial attacks demonstrate that RobustResNets consistently outperform both the standard WRNs and other existing robust architectures, achieving state-of-the-art AutoAttack robust accuracy of 61.1% without additional data and 63.7% with 500K external data while being $2\times$ more compact in terms of parameters. Code is available at \url{ https://github.com/zhichao-lu/robust-residual-network}



## **10. SoK: Let The Privacy Games Begin! A Unified Treatment of Data Inference Privacy in Machine Learning**

cs.LG

**SubmitDate**: 2022-12-21    [abs](http://arxiv.org/abs/2212.10986v1) [paper-pdf](http://arxiv.org/pdf/2212.10986v1)

**Authors**: Ahmed Salem, Giovanni Cherubin, David Evans, Boris Köpf, Andrew Paverd, Anshuman Suri, Shruti Tople, Santiago Zanella-Béguelin

**Abstract**: Deploying machine learning models in production may allow adversaries to infer sensitive information about training data. There is a vast literature analyzing different types of inference risks, ranging from membership inference to reconstruction attacks. Inspired by the success of games (i.e., probabilistic experiments) to study security properties in cryptography, some authors describe privacy inference risks in machine learning using a similar game-based style. However, adversary capabilities and goals are often stated in subtly different ways from one presentation to the other, which makes it hard to relate and compose results. In this paper, we present a game-based framework to systematize the body of knowledge on privacy inference risks in machine learning.



## **11. Hidden Poison: Machine Unlearning Enables Camouflaged Poisoning Attacks**

cs.LG

**SubmitDate**: 2022-12-21    [abs](http://arxiv.org/abs/2212.10717v1) [paper-pdf](http://arxiv.org/pdf/2212.10717v1)

**Authors**: Jimmy Z. Di, Jack Douglas, Jayadev Acharya, Gautam Kamath, Ayush Sekhari

**Abstract**: We introduce camouflaged data poisoning attacks, a new attack vector that arises in the context of machine unlearning and other settings when model retraining may be induced. An adversary first adds a few carefully crafted points to the training dataset such that the impact on the model's predictions is minimal. The adversary subsequently triggers a request to remove a subset of the introduced points at which point the attack is unleashed and the model's predictions are negatively affected. In particular, we consider clean-label targeted attacks (in which the goal is to cause the model to misclassify a specific test point) on datasets including CIFAR-10, Imagenette, and Imagewoof. This attack is realized by constructing camouflage datapoints that mask the effect of a poisoned dataset.



## **12. The Quantum Chernoff Divergence in Advantage Distillation for QKD and DIQKD**

quant-ph

Minor typo fixes

**SubmitDate**: 2022-12-21    [abs](http://arxiv.org/abs/2212.06975v2) [paper-pdf](http://arxiv.org/pdf/2212.06975v2)

**Authors**: Mikka Stasiuk, Norbert Lütkenhaus, Ernest Y. -Z. Tan

**Abstract**: Device-independent quantum key distribution (DIQKD) aims to mitigate adversarial exploitation of imperfections in quantum devices, by providing an approach for secret key distillation with modest security assumptions. Advantage distillation, a two-way communication procedure in error correction, has proven effective in raising noise tolerances in both device-dependent and device-independent QKD. Previously, device-independent security proofs against IID collective attacks were developed for an advantage distillation protocol known as the repetition-code protocol, based on security conditions involving the fidelity between some states in the protocol. However, there exists a gap between the sufficient and necessary security conditions, which hinders the calculation of tight noise-tolerance bounds based on the fidelity. We close this gap by presenting an alternative proof structure that replaces the fidelity with the quantum Chernoff divergence, a distinguishability measure that arises in symmetric hypothesis testing. Working in the IID collective attacks model, we derive matching sufficient and necessary conditions for the repetition-code protocol to be secure (up to a natural conjecture regarding the latter case) in terms of the quantum Chernoff divergence, hence indicating that this serves as the relevant quantity of interest for this protocol. Furthermore, using this security condition we obtain some improvements over previous results on the noise tolerance thresholds for DIQKD. Our results provide insight into a fundamental question in quantum information theory regarding the circumstances under which DIQKD is possible.



## **13. Is Semantic Communications Secure? A Tale of Multi-Domain Adversarial Attacks**

cs.CR

**SubmitDate**: 2022-12-20    [abs](http://arxiv.org/abs/2212.10438v1) [paper-pdf](http://arxiv.org/pdf/2212.10438v1)

**Authors**: Yalin E. Sagduyu, Tugba Erpek, Sennur Ulukus, Aylin Yener

**Abstract**: Semantic communications seeks to transfer information from a source while conveying a desired meaning to its destination. We model the transmitter-receiver functionalities as an autoencoder followed by a task classifier that evaluates the meaning of the information conveyed to the receiver. The autoencoder consists of an encoder at the transmitter to jointly model source coding, channel coding, and modulation, and a decoder at the receiver to jointly model demodulation, channel decoding and source decoding. By augmenting the reconstruction loss with a semantic loss, the two deep neural networks (DNNs) of this encoder-decoder pair are interactively trained with the DNN of the semantic task classifier. This approach effectively captures the latent feature space and reliably transfers compressed feature vectors with a small number of channel uses while keeping the semantic loss low. We identify the multi-domain security vulnerabilities of using the DNNs for semantic communications. Based on adversarial machine learning, we introduce test-time (targeted and non-targeted) adversarial attacks on the DNNs by manipulating their inputs at different stages of semantic communications. As a computer vision attack, small perturbations are injected to the images at the input of the transmitter's encoder. As a wireless attack, small perturbations signals are transmitted to interfere with the input of the receiver's decoder. By launching these stealth attacks individually or more effectively in a combined form as a multi-domain attack, we show that it is possible to change the semantics of the transferred information even when the reconstruction loss remains low. These multi-domain adversarial attacks pose as a serious threat to the semantics of information transfer (with larger impact than conventional jamming) and raise the need of defense methods for the safe adoption of semantic communications.



## **14. In and Out-of-Domain Text Adversarial Robustness via Label Smoothing**

cs.CL

Preprint. Under Submission

**SubmitDate**: 2022-12-20    [abs](http://arxiv.org/abs/2212.10258v1) [paper-pdf](http://arxiv.org/pdf/2212.10258v1)

**Authors**: Yahan Yang, Soham Dan, Dan Roth, Insup Lee

**Abstract**: Recently it has been shown that state-of-the-art NLP models are vulnerable to adversarial attacks, where the predictions of a model can be drastically altered by slight modifications to the input (such as synonym substitutions). While several defense techniques have been proposed, and adapted, to the discrete nature of text adversarial attacks, the benefits of general-purpose regularization methods such as label smoothing for language models, have not been studied. In this paper, we study the adversarial robustness provided by various label smoothing strategies in foundational models for diverse NLP tasks in both in-domain and out-of-domain settings. Our experiments show that label smoothing significantly improves adversarial robustness in pre-trained models like BERT, against various popular attacks. We also analyze the relationship between prediction confidence and robustness, showing that label smoothing reduces over-confident errors on adversarial examples.



## **15. A Comprehensive Study and Comparison of the Robustness of 3D Object Detectors Against Adversarial Attacks**

cs.CV

**SubmitDate**: 2022-12-20    [abs](http://arxiv.org/abs/2212.10230v1) [paper-pdf](http://arxiv.org/pdf/2212.10230v1)

**Authors**: Yifan Zhang, Junhui Hou, Yixuan Yuan

**Abstract**: Deep learning-based 3D object detectors have made significant progress in recent years and have been deployed in a wide range of applications. It is crucial to understand the robustness of detectors against adversarial attacks when employing detectors in security-critical applications. In this paper, we make the first attempt to conduct a thorough evaluation and analysis of the robustness of 3D detectors under adversarial attacks. Specifically, we first extend three kinds of adversarial attacks to the 3D object detection task to benchmark the robustness of state-of-the-art 3D object detectors against attacks on KITTI and Waymo datasets, subsequently followed by the analysis of the relationship between robustness and properties of detectors. Then, we explore the transferability of cross-model, cross-task, and cross-data attacks. We finally conduct comprehensive experiments of defense for 3D detectors, demonstrating that simple transformations like flipping are of little help in improving robustness when the strategy of transformation imposed on input point cloud data is exposed to attackers. Our findings will facilitate investigations in understanding and defending the adversarial attacks against 3D object detectors to advance this field.



## **16. Multi-head Uncertainty Inference for Adversarial Attack Detection**

cs.LG

**SubmitDate**: 2022-12-20    [abs](http://arxiv.org/abs/2212.10006v1) [paper-pdf](http://arxiv.org/pdf/2212.10006v1)

**Authors**: Yuqi Yang, Songyun Yang, Jiyang Xie. Zhongwei Si, Kai Guo, Ke Zhang, Kongming Liang

**Abstract**: Deep neural networks (DNNs) are sensitive and susceptible to tiny perturbation by adversarial attacks which causes erroneous predictions. Various methods, including adversarial defense and uncertainty inference (UI), have been developed in recent years to overcome the adversarial attacks. In this paper, we propose a multi-head uncertainty inference (MH-UI) framework for detecting adversarial attack examples. We adopt a multi-head architecture with multiple prediction heads (i.e., classifiers) to obtain predictions from different depths in the DNNs and introduce shallow information for the UI. Using independent heads at different depths, the normalized predictions are assumed to follow the same Dirichlet distribution, and we estimate distribution parameter of it by moment matching. Cognitive uncertainty brought by the adversarial attacks will be reflected and amplified on the distribution. Experimental results show that the proposed MH-UI framework can outperform all the referred UI methods in the adversarial attack detection task with different settings.



## **17. Defending Against Poisoning Attacks in Open-Domain Question Answering**

cs.CL

**SubmitDate**: 2022-12-20    [abs](http://arxiv.org/abs/2212.10002v1) [paper-pdf](http://arxiv.org/pdf/2212.10002v1)

**Authors**: Orion Weller, Aleem Khan, Nathaniel Weir, Dawn Lawrie, Benjamin Van Durme

**Abstract**: Recent work in open-domain question answering (ODQA) has shown that adversarial poisoning of the input contexts can cause large drops in accuracy for production systems. However, little to no work has proposed methods to defend against these attacks. To do so, we introduce a new method that uses query augmentation to search for a diverse set of retrieved passages that could answer the original question. We integrate these new passages into the model through the design of a novel confidence method, comparing the predicted answer to its appearance in the retrieved contexts (what we call Confidence from Answer Redundancy, e.g. CAR). Together these methods allow for a simple but effective way to defend against poisoning attacks and provide gains of 5-20% exact match across varying levels of data poisoning.



## **18. Deduplicating Training Data Mitigates Privacy Risks in Language Models**

cs.CR

ICML 2022 Camera Ready Version

**SubmitDate**: 2022-12-20    [abs](http://arxiv.org/abs/2202.06539v3) [paper-pdf](http://arxiv.org/pdf/2202.06539v3)

**Authors**: Nikhil Kandpal, Eric Wallace, Colin Raffel

**Abstract**: Past work has shown that large language models are susceptible to privacy attacks, where adversaries generate sequences from a trained model and detect which sequences are memorized from the training set. In this work, we show that the success of these attacks is largely due to duplication in commonly used web-scraped training sets. We first show that the rate at which language models regenerate training sequences is superlinearly related to a sequence's count in the training set. For instance, a sequence that is present 10 times in the training data is on average generated ~1000 times more often than a sequence that is present only once. We next show that existing methods for detecting memorized sequences have near-chance accuracy on non-duplicated training sequences. Finally, we find that after applying methods to deduplicate training data, language models are considerably more secure against these types of privacy attacks. Taken together, our results motivate an increased focus on deduplication in privacy-sensitive applications and a reevaluation of the practicality of existing privacy attacks.



## **19. Towards Robustness of Text-to-SQL Models Against Natural and Realistic Adversarial Table Perturbation**

cs.CL

Accepted by ACL 2022 (Oral)

**SubmitDate**: 2022-12-20    [abs](http://arxiv.org/abs/2212.09994v1) [paper-pdf](http://arxiv.org/pdf/2212.09994v1)

**Authors**: Xinyu Pi, Bing Wang, Yan Gao, Jiaqi Guo, Zhoujun Li, Jian-Guang Lou

**Abstract**: The robustness of Text-to-SQL parsers against adversarial perturbations plays a crucial role in delivering highly reliable applications. Previous studies along this line primarily focused on perturbations in the natural language question side, neglecting the variability of tables. Motivated by this, we propose the Adversarial Table Perturbation (ATP) as a new attacking paradigm to measure the robustness of Text-to-SQL models. Following this proposition, we curate ADVETA, the first robustness evaluation benchmark featuring natural and realistic ATPs. All tested state-of-the-art models experience dramatic performance drops on ADVETA, revealing models' vulnerability in real-world practices. To defend against ATP, we build a systematic adversarial training example generation framework tailored for better contextualization of tabular data. Experiments show that our approach not only brings the best robustness improvement against table-side perturbations but also substantially empowers models against NL-side perturbations. We release our benchmark and code at: https://github.com/microsoft/ContextualSP.



## **20. Task-Oriented Communications for NextG: End-to-End Deep Learning and AI Security Aspects**

cs.NI

**SubmitDate**: 2022-12-19    [abs](http://arxiv.org/abs/2212.09668v1) [paper-pdf](http://arxiv.org/pdf/2212.09668v1)

**Authors**: Yalin E. Sagduyu, Sennur Ulukus, Aylin Yener

**Abstract**: Communications systems to date are primarily designed with the goal of reliable (error-free) transfer of digital sequences (bits). Next generation (NextG) communication systems are beginning to explore shifting this design paradigm of reliably decoding bits to reliably executing a given task. Task-oriented communications system design is likely to find impactful applications, for example, considering the relative importance of messages. In this paper, a wireless signal classification is considered as the task to be performed in the NextG Radio Access Network (RAN) for signal intelligence and spectrum awareness applications such as user equipment (UE) identification and authentication, and incumbent signal detection for spectrum co-existence. For that purpose, edge devices collect wireless signals and communicate with the NextG base station (gNodeB) that needs to know the signal class. Edge devices may not have sufficient processing power and may not be trusted to perform the signal classification task, whereas the transfer of the captured signals from the edge devices to the gNodeB may not be efficient or even feasible subject to stringent delay, rate, and energy restrictions. We present a task-oriented communications approach, where all the transmitter, receiver and classifier functionalities are jointly trained as two deep neural networks (DNNs), one for the edge device and another for the gNodeB. We show that this approach achieves better accuracy with smaller DNNs compared to the baselines that treat communications and signal classification as two separate tasks. Finally, we discuss how adversarial machine learning poses a major security threat for the use of DNNs for task-oriented communications. We demonstrate the major performance loss under backdoor (Trojan) attacks and adversarial (evasion) attacks that target the training and test processes of task-oriented communications.



## **21. Adversarial Attacks against Windows PE Malware Detection: A Survey of the State-of-the-Art**

cs.CR

**SubmitDate**: 2022-12-19    [abs](http://arxiv.org/abs/2112.12310v3) [paper-pdf](http://arxiv.org/pdf/2112.12310v3)

**Authors**: Xiang Ling, Lingfei Wu, Jiangyu Zhang, Zhenqing Qu, Wei Deng, Xiang Chen, Yaguan Qian, Chunming Wu, Shouling Ji, Tianyue Luo, Jingzheng Wu, Yanjun Wu

**Abstract**: Malware has been one of the most damaging threats to computers that span across multiple operating systems and various file formats. To defend against ever-increasing and ever-evolving malware, tremendous efforts have been made to propose a variety of malware detection that attempt to effectively and efficiently detect malware so as to mitigate possible damages as early as possible. Recent studies have shown that, on the one hand, existing ML and DL techniques enable superior solutions in detecting newly emerging and previously unseen malware. However, on the other hand, ML and DL models are inherently vulnerable to adversarial attacks in the form of adversarial examples. In this paper, we focus on malware with the file format of portable executable (PE) in the family of Windows operating systems, namely Windows PE malware, as a representative case to study the adversarial attack methods in such adversarial settings. To be specific, we start by first outlining the general learning framework of Windows PE malware detection based on ML/DL and subsequently highlighting three unique challenges of performing adversarial attacks in the context of Windows PE malware. Then, we conduct a comprehensive and systematic review to categorize the state-of-the-art adversarial attacks against PE malware detection, as well as corresponding defenses to increase the robustness of Windows PE malware detection. Finally, we conclude the paper by first presenting other related attacks against Windows PE malware detection beyond the adversarial attacks and then shedding light on future research directions and opportunities. In addition, a curated resource list of adversarial attacks and defenses for Windows PE malware detection is also available at https://github.com/ryderling/adversarial-attacks-and-defenses-for-windows-pe-malware-detection.



## **22. Adversarial Sticker: A Stealthy Attack Method in the Physical World**

cs.CV

accepted by TPAMI 2022

**SubmitDate**: 2022-12-19    [abs](http://arxiv.org/abs/2104.06728v2) [paper-pdf](http://arxiv.org/pdf/2104.06728v2)

**Authors**: Xingxing Wei, Ying Guo, Jie Yu

**Abstract**: To assess the vulnerability of deep learning in the physical world, recent works introduce adversarial patches and apply them on different tasks. In this paper, we propose another kind of adversarial patch: the Meaningful Adversarial Sticker, a physically feasible and stealthy attack method by using real stickers existing in our life. Unlike the previous adversarial patches by designing perturbations, our method manipulates the sticker's pasting position and rotation angle on the objects to perform physical attacks. Because the position and rotation angle are less affected by the printing loss and color distortion, adversarial stickers can keep good attacking performance in the physical world. Besides, to make adversarial stickers more practical in real scenes, we conduct attacks in the black-box setting with the limited information rather than the white-box setting with all the details of threat models. To effectively solve for the sticker's parameters, we design the Region based Heuristic Differential Evolution Algorithm, which utilizes the new-found regional aggregation of effective solutions and the adaptive adjustment strategy of the evaluation criteria. Our method is comprehensively verified in the face recognition and then extended to the image retrieval and traffic sign recognition. Extensive experiments show the proposed method is effective and efficient in complex physical conditions and has a good generalization for different tasks.



## **23. AI Security for Geoscience and Remote Sensing: Challenges and Future Trends**

cs.CV

**SubmitDate**: 2022-12-19    [abs](http://arxiv.org/abs/2212.09360v1) [paper-pdf](http://arxiv.org/pdf/2212.09360v1)

**Authors**: Yonghao Xu, Tao Bai, Weikang Yu, Shizhen Chang, Peter M. Atkinson, Pedram Ghamisi

**Abstract**: Recent advances in artificial intelligence (AI) have significantly intensified research in the geoscience and remote sensing (RS) field. AI algorithms, especially deep learning-based ones, have been developed and applied widely to RS data analysis. The successful application of AI covers almost all aspects of Earth observation (EO) missions, from low-level vision tasks like super-resolution, denoising, and inpainting, to high-level vision tasks like scene classification, object detection, and semantic segmentation. While AI techniques enable researchers to observe and understand the Earth more accurately, the vulnerability and uncertainty of AI models deserve further attention, considering that many geoscience and RS tasks are highly safety-critical. This paper reviews the current development of AI security in the geoscience and RS field, covering the following five important aspects: adversarial attack, backdoor attack, federated learning, uncertainty, and explainability. Moreover, the potential opportunities and trends are discussed to provide insights for future research. To the best of the authors' knowledge, this paper is the first attempt to provide a systematic review of AI security-related research in the geoscience and RS community. Available code and datasets are also listed in the paper to move this vibrant field of research forward.



## **24. Review of security techniques for memristor computing systems**

cs.CR

15 pages, 5 figures

**SubmitDate**: 2022-12-19    [abs](http://arxiv.org/abs/2212.09347v1) [paper-pdf](http://arxiv.org/pdf/2212.09347v1)

**Authors**: Minhui Zou, Nan Du, Shahar Kvatinsky

**Abstract**: Neural network (NN) algorithms have become the dominant tool in visual object recognition, natural language processing, and robotics. To enhance the computational efficiency of these algorithms, in comparison to the traditional von Neuman computing architectures, researchers have been focusing on memristor computing systems. A major drawback when using memristor computing systems today is that, in the artificial intelligence (AI) era, well-trained NN models are intellectual property and, when loaded in the memristor computing systems, face theft threats, especially when running in edge devices. An adversary may steal the well-trained NN models through advanced attacks such as learning attacks and side-channel analysis. In this paper, we review different security techniques for protecting memristor computing systems. Two threat models are described based on their assumptions regarding the adversary's capabilities: a black-box (BB) model and a white-box (WB) model. We categorize the existing security techniques into five classes in the context of these threat models: thwarting learning attacks (BB), thwarting side-channel attacks (BB), NN model encryption (WB), NN weight transformation (WB), and fingerprint embedding (WB). We also present a cross-comparison of the limitations of the security techniques. This paper could serve as an aid when designing secure memristor computing systems.



## **25. TextGrad: Advancing Robustness Evaluation in NLP by Gradient-Driven Optimization**

cs.CL

18 pages, 2 figures

**SubmitDate**: 2022-12-19    [abs](http://arxiv.org/abs/2212.09254v1) [paper-pdf](http://arxiv.org/pdf/2212.09254v1)

**Authors**: Bairu Hou, Jinghan Jia, Yihua Zhang, Guanhua Zhang, Yang Zhang, Sijia Liu, Shiyu Chang

**Abstract**: Robustness evaluation against adversarial examples has become increasingly important to unveil the trustworthiness of the prevailing deep models in natural language processing (NLP). However, in contrast to the computer vision domain where the first-order projected gradient descent (PGD) is used as the benchmark approach to generate adversarial examples for robustness evaluation, there lacks a principled first-order gradient-based robustness evaluation framework in NLP. The emerging optimization challenges lie in 1) the discrete nature of textual inputs together with the strong coupling between the perturbation location and the actual content, and 2) the additional constraint that the perturbed text should be fluent and achieve a low perplexity under a language model. These challenges make the development of PGD-like NLP attacks difficult. To bridge the gap, we propose TextGrad, a new attack generator using gradient-driven optimization, supporting high-accuracy and high-quality assessment of adversarial robustness in NLP. Specifically, we address the aforementioned challenges in a unified optimization framework. And we develop an effective convex relaxation method to co-optimize the continuously-relaxed site selection and perturbation variables and leverage an effective sampling method to establish an accurate mapping from the continuous optimization variables to the discrete textual perturbations. Moreover, as a first-order attack generation method, TextGrad can be baked into adversarial training to further improve the robustness of NLP models. Extensive experiments are provided to demonstrate the effectiveness of TextGrad not only in attack generation for robustness evaluation but also in adversarial defense.



## **26. Minimizing Maximum Model Discrepancy for Transferable Black-box Targeted Attacks**

cs.CV

**SubmitDate**: 2022-12-18    [abs](http://arxiv.org/abs/2212.09035v1) [paper-pdf](http://arxiv.org/pdf/2212.09035v1)

**Authors**: Anqi Zhao, Tong Chu, Yahao Liu, Wen Li, Jingjing Li, Lixin Duan

**Abstract**: In this work, we study the black-box targeted attack problem from the model discrepancy perspective. On the theoretical side, we present a generalization error bound for black-box targeted attacks, which gives a rigorous theoretical analysis for guaranteeing the success of the attack. We reveal that the attack error on a target model mainly depends on empirical attack error on the substitute model and the maximum model discrepancy among substitute models. On the algorithmic side, we derive a new algorithm for black-box targeted attacks based on our theoretical analysis, in which we additionally minimize the maximum model discrepancy(M3D) of the substitute models when training the generator to generate adversarial examples. In this way, our model is capable of crafting highly transferable adversarial examples that are robust to the model variation, thus improving the success rate for attacking the black-box model. We conduct extensive experiments on the ImageNet dataset with different classification models, and our proposed approach outperforms existing state-of-the-art methods by a significant margin. Our codes will be released.



## **27. A Review of Speech-centric Trustworthy Machine Learning: Privacy, Safety, and Fairness**

cs.SD

**SubmitDate**: 2022-12-18    [abs](http://arxiv.org/abs/2212.09006v1) [paper-pdf](http://arxiv.org/pdf/2212.09006v1)

**Authors**: Tiantian Feng, Rajat Hebbar, Nicholas Mehlman, Xuan Shi, Aditya Kommineni, and Shrikanth Narayanan

**Abstract**: Speech-centric machine learning systems have revolutionized many leading domains ranging from transportation and healthcare to education and defense, profoundly changing how people live, work, and interact with each other. However, recent studies have demonstrated that many speech-centric ML systems may need to be considered more trustworthy for broader deployment. Specifically, concerns over privacy breaches, discriminating performance, and vulnerability to adversarial attacks have all been discovered in ML research fields. In order to address the above challenges and risks, a significant number of efforts have been made to ensure these ML systems are trustworthy, especially private, safe, and fair. In this paper, we conduct the first comprehensive survey on speech-centric trustworthy ML topics related to privacy, safety, and fairness. In addition to serving as a summary report for the research community, we point out several promising future research directions to inspire the researchers who wish to explore further in this area.



## **28. Bounding Membership Inference**

cs.LG

**SubmitDate**: 2022-12-18    [abs](http://arxiv.org/abs/2202.12232v4) [paper-pdf](http://arxiv.org/pdf/2202.12232v4)

**Authors**: Anvith Thudi, Ilia Shumailov, Franziska Boenisch, Nicolas Papernot

**Abstract**: Differential Privacy (DP) is the de facto standard for reasoning about the privacy guarantees of a training algorithm. Despite the empirical observation that DP reduces the vulnerability of models to existing membership inference (MI) attacks, a theoretical underpinning as to why this is the case is largely missing in the literature. In practice, this means that models need to be trained with DP guarantees that greatly decrease their accuracy.   In this paper, we provide a tighter bound on the positive accuracy (i.e., attack precision) of any MI adversary when a training algorithm provides $(\varepsilon, \delta)$-DP. Our bound informs the design of a novel privacy amplification scheme: an effective training set is sub-sampled from a larger set prior to the beginning of training. We find this greatly reduces the bound on MI positive accuracy. As a result, our scheme allows the use of looser DP guarantees to limit the success of any MI adversary; this ensures that the model's accuracy is less impacted by the privacy guarantee. While this clearly benefits entities working with far more data than they need to train on, it can also improve the accuracy-privacy trade-off on benchmarks studied in the academic literature. Consequently, we also find that subsampling decreases the effectiveness of a state-of-the-art MI attack (LiRA) much more effectively than training with stronger DP guarantees on MNIST and CIFAR10. We conclude by discussing implications of our MI bound on the field of machine unlearning.



## **29. Scalable Adversarial Attack Algorithms on Influence Maximization**

cs.SI

11 pages, 2 figures

**SubmitDate**: 2022-12-17    [abs](http://arxiv.org/abs/2209.00892v2) [paper-pdf](http://arxiv.org/pdf/2209.00892v2)

**Authors**: Lichao Sun, Xiaobin Rui, Wei Chen

**Abstract**: In this paper, we study the adversarial attacks on influence maximization under dynamic influence propagation models in social networks. In particular, given a known seed set S, the problem is to minimize the influence spread from S by deleting a limited number of nodes and edges. This problem reflects many application scenarios, such as blocking virus (e.g. COVID-19) propagation in social networks by quarantine and vaccination, blocking rumor spread by freezing fake accounts, or attacking competitor's influence by incentivizing some users to ignore the information from the competitor. In this paper, under the linear threshold model, we adapt the reverse influence sampling approach and provide efficient algorithms of sampling valid reverse reachable paths to solve the problem. We present three different design choices on reverse sampling, which all guarantee $1/2 - \varepsilon$ approximation (for any small $\varepsilon >0$) and an efficient running time.



## **30. Expeditious Saliency-guided Mix-up through Random Gradient Thresholding**

cs.CV

Accepted Long paper at 2nd Practical-DL Workshop at AAAI 2023. V2 fix  typo

**SubmitDate**: 2022-12-17    [abs](http://arxiv.org/abs/2212.04875v2) [paper-pdf](http://arxiv.org/pdf/2212.04875v2)

**Authors**: Minh-Long Luu, Zeyi Huang, Eric P. Xing, Yong Jae Lee, Haohan Wang

**Abstract**: Mix-up training approaches have proven to be effective in improving the generalization ability of Deep Neural Networks. Over the years, the research community expands mix-up methods into two directions, with extensive efforts to improve saliency-guided procedures but minimal focus on the arbitrary path, leaving the randomization domain unexplored. In this paper, inspired by the superior qualities of each direction over one another, we introduce a novel method that lies at the junction of the two routes. By combining the best elements of randomness and saliency utilization, our method balances speed, simplicity, and accuracy. We name our method R-Mix following the concept of "Random Mix-up". We demonstrate its effectiveness in generalization, weakly supervised object localization, calibration, and robustness to adversarial attacks. Finally, in order to address the question of whether there exists a better decision protocol, we train a Reinforcement Learning agent that decides the mix-up policies based on the classifier's performance, reducing dependency on human-designed objectives and hyperparameter tuning. Extensive experiments further show that the agent is capable of performing at the cutting-edge level, laying the foundation for a fully automatic mix-up. Our code is released at [https://github.com/minhlong94/Random-Mixup].



## **31. More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks**

cs.CR

15 pages, 14 figures

**SubmitDate**: 2022-12-16    [abs](http://arxiv.org/abs/2202.03195v4) [paper-pdf](http://arxiv.org/pdf/2202.03195v4)

**Authors**: Jing Xu, Rui Wang, Stefanos Koffas, Kaitai Liang, Stjepan Picek

**Abstract**: Graph Neural Networks (GNNs) are a class of deep learning-based methods for processing graph domain information. GNNs have recently become a widely used graph analysis method due to their superior ability to learn representations for complex graph data. However, due to privacy concerns and regulation restrictions, centralized GNNs can be difficult to apply to data-sensitive scenarios. Federated learning (FL) is an emerging technology developed for privacy-preserving settings when several parties need to train a shared global model collaboratively. Although several research works have applied FL to train GNNs (Federated GNNs), there is no research on their robustness to backdoor attacks.   This paper bridges this gap by conducting two types of backdoor attacks in Federated GNNs: centralized backdoor attacks (CBA) and distributed backdoor attacks (DBA). Our experiments show that the DBA attack success rate is higher than CBA in almost all evaluated cases. For CBA, the attack success rate of all local triggers is similar to the global trigger even if the training set of the adversarial party is embedded with the global trigger. To further explore the properties of two backdoor attacks in Federated GNNs, we evaluate the attack performance for a different number of clients, trigger sizes, poisoning intensities, and trigger densities. Moreover, we explore the robustness of DBA and CBA against two defenses. We find that both attacks are robust against the investigated defenses, necessitating the need to consider backdoor attacks in Federated GNNs as a novel threat that requires custom defenses.



## **32. LDL: A Defense for Label-Based Membership Inference Attacks**

cs.LG

to appear in ACM ASIA Conference on Computer and Communications  Security (ACM ASIACCS 2023)

**SubmitDate**: 2022-12-16    [abs](http://arxiv.org/abs/2212.01688v2) [paper-pdf](http://arxiv.org/pdf/2212.01688v2)

**Authors**: Arezoo Rajabi, Dinuka Sahabandu, Luyao Niu, Bhaskar Ramasubramanian, Radha Poovendran

**Abstract**: The data used to train deep neural network (DNN) models in applications such as healthcare and finance typically contain sensitive information. A DNN model may suffer from overfitting. Overfitted models have been shown to be susceptible to query-based attacks such as membership inference attacks (MIAs). MIAs aim to determine whether a sample belongs to the dataset used to train a classifier (members) or not (nonmembers). Recently, a new class of label based MIAs (LAB MIAs) was proposed, where an adversary was only required to have knowledge of predicted labels of samples. Developing a defense against an adversary carrying out a LAB MIA on DNN models that cannot be retrained remains an open problem. We present LDL, a light weight defense against LAB MIAs. LDL works by constructing a high-dimensional sphere around queried samples such that the model decision is unchanged for (noisy) variants of the sample within the sphere. This sphere of label-invariance creates ambiguity and prevents a querying adversary from correctly determining whether a sample is a member or a nonmember. We analytically characterize the success rate of an adversary carrying out a LAB MIA when LDL is deployed, and show that the formulation is consistent with experimental observations. We evaluate LDL on seven datasets -- CIFAR-10, CIFAR-100, GTSRB, Face, Purchase, Location, and Texas -- with varying sizes of training data. All of these datasets have been used by SOTA LAB MIAs. Our experiments demonstrate that LDL reduces the success rate of an adversary carrying out a LAB MIA in each case. We empirically compare LDL with defenses against LAB MIAs that require retraining of DNN models, and show that LDL performs favorably despite not needing to retrain the DNNs.



## **33. Adversarial Inter-Group Link Injection Degrades the Fairness of Graph Neural Networks**

cs.LG

A shorter version of this work has been accepted by IEEE ICDM 2022

**SubmitDate**: 2022-12-16    [abs](http://arxiv.org/abs/2209.05957v2) [paper-pdf](http://arxiv.org/pdf/2209.05957v2)

**Authors**: Hussain Hussain, Meng Cao, Sandipan Sikdar, Denis Helic, Elisabeth Lex, Markus Strohmaier, Roman Kern

**Abstract**: We present evidence for the existence and effectiveness of adversarial attacks on graph neural networks (GNNs) that aim to degrade fairness. These attacks can disadvantage a particular subgroup of nodes in GNN-based node classification, where nodes of the underlying network have sensitive attributes, such as race or gender. We conduct qualitative and experimental analyses explaining how adversarial link injection impairs the fairness of GNN predictions. For example, an attacker can compromise the fairness of GNN-based node classification by injecting adversarial links between nodes belonging to opposite subgroups and opposite class labels. Our experiments on empirical datasets demonstrate that adversarial fairness attacks can significantly degrade the fairness of GNN predictions (attacks are effective) with a low perturbation rate (attacks are efficient) and without a significant drop in accuracy (attacks are deceptive). This work demonstrates the vulnerability of GNN models to adversarial fairness attacks. We hope our findings raise awareness about this issue in our community and lay a foundation for the future development of GNN models that are more robust to such attacks.



## **34. Conditional Generative Adversarial Network for keystroke presentation attack**

cs.CR

**SubmitDate**: 2022-12-16    [abs](http://arxiv.org/abs/2212.08445v1) [paper-pdf](http://arxiv.org/pdf/2212.08445v1)

**Authors**: Idoia Eizaguirre-Peral, Lander Segurola-Gil, Francesco Zola

**Abstract**: Cybersecurity is a crucial step in data protection to ensure user security and personal data privacy. In this sense, many companies have started to control and restrict access to their data using authentication systems. However, these traditional authentication methods, are not enough for ensuring data protection, and for this reason, behavioral biometrics have gained importance. Despite their promising results and the wide range of applications, biometric systems have shown to be vulnerable to malicious attacks, such as Presentation Attacks. For this reason, in this work, we propose to study a new approach aiming to deploy a presentation attack towards a keystroke authentication system. Our idea is to use Conditional Generative Adversarial Networks (cGAN) for generating synthetic keystroke data that can be used for impersonating an authorized user. These synthetic data are generated following two different real use cases, one in which the order of the typed words is known (ordered dynamic) and the other in which this order is unknown (no-ordered dynamic). Finally, both keystroke dynamics (ordered and no-ordered) are validated using an external keystroke authentication system. Results indicate that the cGAN can effectively generate keystroke dynamics patterns that can be used for deceiving keystroke authentication systems.



## **35. CgAT: Center-Guided Adversarial Training for Deep Hashing-Based Retrieval**

cs.CV

**SubmitDate**: 2022-12-16    [abs](http://arxiv.org/abs/2204.10779v4) [paper-pdf](http://arxiv.org/pdf/2204.10779v4)

**Authors**: Xunguang Wang, Yinqun Lin, Xiaomeng Li

**Abstract**: Deep hashing has been extensively utilized in massive image retrieval because of its efficiency and effectiveness. However, deep hashing models are vulnerable to adversarial examples, making it essential to develop adversarial defense methods for image retrieval. Existing solutions achieved limited defense performance because of using weak adversarial samples for training and lacking discriminative optimization objectives to learn robust features. In this paper, we present a min-max based Center-guided Adversarial Training, namely CgAT, to improve the robustness of deep hashing networks through worst adversarial examples. Specifically, we first formulate the center code as a semantically-discriminative representative of the input image content, which preserves the semantic similarity with positive samples and dissimilarity with negative examples. We prove that a mathematical formula can calculate the center code immediately. After obtaining the center codes in each optimization iteration of the deep hashing network, they are adopted to guide the adversarial training process. On the one hand, CgAT generates the worst adversarial examples as augmented data by maximizing the Hamming distance between the hash codes of the adversarial examples and the center codes. On the other hand, CgAT learns to mitigate the effects of adversarial samples by minimizing the Hamming distance to the center codes. Extensive experiments on the benchmark datasets demonstrate the effectiveness of our adversarial training algorithm in defending against adversarial attacks for deep hashing-based retrieval. Compared with the current state-of-the-art defense method, we significantly improve the defense performance by an average of 18.61%, 12.35%, and 11.56% on FLICKR-25K, NUS-WIDE, and MS-COCO, respectively.



## **36. Jujutsu: A Two-stage Defense against Adversarial Patch Attacks on Deep Neural Networks**

cs.CR

To appear in AsiaCCS'23

**SubmitDate**: 2022-12-16    [abs](http://arxiv.org/abs/2108.05075v4) [paper-pdf](http://arxiv.org/pdf/2108.05075v4)

**Authors**: Zitao Chen, Pritam Dash, Karthik Pattabiraman

**Abstract**: Adversarial patch attacks create adversarial examples by injecting arbitrary distortions within a bounded region of the input to fool deep neural networks (DNNs). These attacks are robust (i.e., physically-realizable) and universally malicious, and hence represent a severe security threat to real-world DNN-based systems.   We propose Jujutsu, a two-stage technique to detect and mitigate robust and universal adversarial patch attacks. We first observe that adversarial patches are crafted as localized features that yield large influence on the prediction output, and continue to dominate the prediction on any input. Jujutsu leverages this observation for accurate attack detection with low false positives. Patch attacks corrupt only a localized region of the input, while the majority of the input remains unperturbed. Therefore, Jujutsu leverages generative adversarial networks (GAN) to perform localized attack recovery by synthesizing the semantic contents of the input that are corrupted by the attacks, and reconstructs a ``clean'' input for correct prediction.   We evaluate Jujutsu on four diverse datasets spanning 8 different DNN models, and find that it achieves superior performance and significantly outperforms four existing defenses. We further evaluate Jujutsu against physical-world attacks, as well as adaptive attacks.



## **37. Adversarial Example Defense via Perturbation Grading Strategy**

cs.CV

**SubmitDate**: 2022-12-16    [abs](http://arxiv.org/abs/2212.08341v1) [paper-pdf](http://arxiv.org/pdf/2212.08341v1)

**Authors**: Shaowei Zhu, Wanli Lyu, Bin Li, Zhaoxia Yin, Bin Luo

**Abstract**: Deep Neural Networks have been widely used in many fields. However, studies have shown that DNNs are easily attacked by adversarial examples, which have tiny perturbations and greatly mislead the correct judgment of DNNs. Furthermore, even if malicious attackers cannot obtain all the underlying model parameters, they can use adversarial examples to attack various DNN-based task systems. Researchers have proposed various defense methods to protect DNNs, such as reducing the aggressiveness of adversarial examples by preprocessing or improving the robustness of the model by adding modules. However, some defense methods are only effective for small-scale examples or small perturbations but have limited defense effects for adversarial examples with large perturbations. This paper assigns different defense strategies to adversarial perturbations of different strengths by grading the perturbations on the input examples. Experimental results show that the proposed method effectively improves defense performance. In addition, the proposed method does not modify any task model, which can be used as a preprocessing module, which significantly reduces the deployment cost in practical applications.



## **38. Adversarial Attack on Attackers: Post-Process to Mitigate Black-Box Score-Based Query Attacks**

cs.LG

accepted by NeurIPS 2022

**SubmitDate**: 2022-12-16    [abs](http://arxiv.org/abs/2205.12134v3) [paper-pdf](http://arxiv.org/pdf/2205.12134v3)

**Authors**: Sizhe Chen, Zhehao Huang, Qinghua Tao, Yingwen Wu, Cihang Xie, Xiaolin Huang

**Abstract**: The score-based query attacks (SQAs) pose practical threats to deep neural networks by crafting adversarial perturbations within dozens of queries, only using the model's output scores. Nonetheless, we note that if the loss trend of the outputs is slightly perturbed, SQAs could be easily misled and thereby become much less effective. Following this idea, we propose a novel defense, namely Adversarial Attack on Attackers (AAA), to confound SQAs towards incorrect attack directions by slightly modifying the output logits. In this way, (1) SQAs are prevented regardless of the model's worst-case robustness; (2) the original model predictions are hardly changed, i.e., no degradation on clean accuracy; (3) the calibration of confidence scores can be improved simultaneously. Extensive experiments are provided to verify the above advantages. For example, by setting $\ell_\infty=8/255$ on CIFAR-10, our proposed AAA helps WideResNet-28 secure 80.59% accuracy under Square attack (2500 queries), while the best prior defense (i.e., adversarial training) only attains 67.44%. Since AAA attacks SQA's general greedy strategy, such advantages of AAA over 8 defenses can be consistently observed on 8 CIFAR-10/ImageNet models under 6 SQAs, using different attack targets, bounds, norms, losses, and strategies. Moreover, AAA calibrates better without hurting the accuracy. Our code is available at https://github.com/Sizhe-Chen/AAA.



## **39. A Survey on Biometrics Authentication**

cs.CR

6 pages, 9 figures, 9 references

**SubmitDate**: 2022-12-16    [abs](http://arxiv.org/abs/2212.08224v1) [paper-pdf](http://arxiv.org/pdf/2212.08224v1)

**Authors**: Fangshi Zhou, Tianming Zhao

**Abstract**: Nowadays, traditional authentication methods are vulnerable to face attacks that are often based on inherent security issues. Professional attackers leverage adversarial offenses on the security holes. Biometrics has intrinsic advantages to overcome the traditional authentication methods on security, success rates, efficiency, and accessibility. Biometrics has wide prospects to implement various applications in fields. Whether in authentication security or clinical medicine, biometrics is one of the mainstream studies. In this paper, we surveyed and reviewed some related studies of biometrics, which are outstanding and significant in driving the development and popularization of biometrics. Although they still have some inherent disadvantages to restrict popularization, these obstacles could not conceal the promising future of biometrics. Multi-factors continuous biometrics authentication has become the mainstream trend of development. We reflect the findings as well as the challenges of the studies in the survey paper.



## **40. Quantifying the Preferential Direction of the Model Gradient in Adversarial Training With Projected Gradient Descent**

stat.ML

**SubmitDate**: 2022-12-15    [abs](http://arxiv.org/abs/2009.04709v4) [paper-pdf](http://arxiv.org/pdf/2009.04709v4)

**Authors**: Ricardo Bigolin Lanfredi, Joyce D. Schroeder, Tolga Tasdizen

**Abstract**: Adversarial training, especially projected gradient descent (PGD), has proven to be a successful approach for improving robustness against adversarial attacks. After adversarial training, gradients of models with respect to their inputs have a preferential direction. However, the direction of alignment is not mathematically well established, making it difficult to evaluate quantitatively. We propose a novel definition of this direction as the direction of the vector pointing toward the closest point of the support of the closest inaccurate class in decision space. To evaluate the alignment with this direction after adversarial training, we apply a metric that uses generative adversarial networks to produce the smallest residual needed to change the class present in the image. We show that PGD-trained models have a higher alignment than the baseline according to our definition, that our metric presents higher alignment values than a competing metric formulation, and that enforcing this alignment increases the robustness of models.



## **41. On Evaluating Adversarial Robustness of Chest X-ray Classification: Pitfalls and Best Practices**

eess.IV

**SubmitDate**: 2022-12-15    [abs](http://arxiv.org/abs/2212.08130v1) [paper-pdf](http://arxiv.org/pdf/2212.08130v1)

**Authors**: Salah Ghamizi, Maxime Cordy, Michail Papadakis, Yves Le Traon

**Abstract**: Vulnerability to adversarial attacks is a well-known weakness of Deep Neural Networks. While most of the studies focus on natural images with standardized benchmarks like ImageNet and CIFAR, little research has considered real world applications, in particular in the medical domain. Our research shows that, contrary to previous claims, robustness of chest x-ray classification is much harder to evaluate and leads to very different assessments based on the dataset, the architecture and robustness metric. We argue that previous studies did not take into account the peculiarity of medical diagnosis, like the co-occurrence of diseases, the disagreement of labellers (domain experts), the threat model of the attacks and the risk implications for each successful attack.   In this paper, we discuss the methodological foundations, review the pitfalls and best practices, and suggest new methodological considerations for evaluating the robustness of chest xray classification models. Our evaluation on 3 datasets, 7 models, and 18 diseases is the largest evaluation of robustness of chest x-ray classification models.



## **42. Alternating Objectives Generates Stronger PGD-Based Adversarial Attacks**

cs.LG

**SubmitDate**: 2022-12-15    [abs](http://arxiv.org/abs/2212.07992v1) [paper-pdf](http://arxiv.org/pdf/2212.07992v1)

**Authors**: Nikolaos Antoniou, Efthymios Georgiou, Alexandros Potamianos

**Abstract**: Designing powerful adversarial attacks is of paramount importance for the evaluation of $\ell_p$-bounded adversarial defenses. Projected Gradient Descent (PGD) is one of the most effective and conceptually simple algorithms to generate such adversaries. The search space of PGD is dictated by the steepest ascent directions of an objective. Despite the plethora of objective function choices, there is no universally superior option and robustness overestimation may arise from ill-suited objective selection. Driven by this observation, we postulate that the combination of different objectives through a simple loss alternating scheme renders PGD more robust towards design choices. We experimentally verify this assertion on a synthetic-data example and by evaluating our proposed method across 25 different $\ell_{\infty}$-robust models and 3 datasets. The performance improvement is consistent, when compared to the single loss counterparts. In the CIFAR-10 dataset, our strongest adversarial attack outperforms all of the white-box components of AutoAttack (AA) ensemble, as well as the most powerful attacks existing on the literature, achieving state-of-the-art results in the computational budget of our study ($T=100$, no restarts).



## **43. Holistic risk assessment of inference attacks in machine learning**

cs.CR

**SubmitDate**: 2022-12-15    [abs](http://arxiv.org/abs/2212.10628v1) [paper-pdf](http://arxiv.org/pdf/2212.10628v1)

**Authors**: Yang Yang

**Abstract**: As machine learning expanding application, there are more and more unignorable privacy and safety issues. Especially inference attacks against Machine Learning models allow adversaries to infer sensitive information about the target model, such as training data, model parameters, etc. Inference attacks can lead to serious consequences, including violating individuals privacy, compromising the intellectual property of the owner of the machine learning model. As far as concerned, researchers have studied and analyzed in depth several types of inference attacks, albeit in isolation, but there is still a lack of a holistic rick assessment of inference attacks against machine learning models, such as their application in different scenarios, the common factors affecting the performance of these attacks and the relationship among the attacks. As a result, this paper performs a holistic risk assessment of different inference attacks against Machine Learning models. This paper focuses on three kinds of representative attacks: membership inference attack, attribute inference attack and model stealing attack. And a threat model taxonomy is established. A total of 12 target models using three model architectures, including AlexNet, ResNet18 and Simple CNN, are trained on four datasets, namely CelebA, UTKFace, STL10 and FMNIST.



## **44. A Feedback-optimization-based Model-free Attack Scheme in Networked Control Systems**

math.OC

**SubmitDate**: 2022-12-15    [abs](http://arxiv.org/abs/2212.07633v1) [paper-pdf](http://arxiv.org/pdf/2212.07633v1)

**Authors**: Xiaoyu Luo, Chongrong Fang, Jianping He, Chengcheng Zhao, Dario Paccagnan

**Abstract**: The data-driven attack strategies recently have received much attention when the system model is unknown to the adversary. Depending on the unknown linear system model, the existing data-driven attack strategies have already designed lots of efficient attack schemes. In this paper, we study a completely model-free attack scheme regardless of whether the unknown system model is linear or nonlinear. The objective of the adversary with limited capability is to compromise state variables such that the output value follows a malicious trajectory. Specifically, we first construct a zeroth-order feedback optimization framework and uninterruptedly use probing signals for real-time measurements. Then, we iteratively update the attack signals along the composite direction of the gradient estimates of the objective function evaluations and the projected gradients. These objective function evaluations can be obtained only by real-time measurements. Furthermore, we characterize the optimality of the proposed model-free attack via the optimality gap, which is affected by the dimensions of the attack signal, the iterations of solutions, and the convergence rate of the system. Finally, we extend the proposed attack scheme to the system with internal inherent noise and analyze the effects of noise on the optimality gap. Extensive simulations are conducted to show the effectiveness of the proposed attack scheme.



## **45. Dissecting Distribution Inference**

cs.LG

Accepted at SaTML 2023

**SubmitDate**: 2022-12-15    [abs](http://arxiv.org/abs/2212.07591v1) [paper-pdf](http://arxiv.org/pdf/2212.07591v1)

**Authors**: Anshuman Suri, Yifu Lu, Yanjin Chen, David Evans

**Abstract**: A distribution inference attack aims to infer statistical properties of data used to train machine learning models. These attacks are sometimes surprisingly potent, but the factors that impact distribution inference risk are not well understood and demonstrated attacks often rely on strong and unrealistic assumptions such as full knowledge of training environments even in supposedly black-box threat scenarios. To improve understanding of distribution inference risks, we develop a new black-box attack that even outperforms the best known white-box attack in most settings. Using this new attack, we evaluate distribution inference risk while relaxing a variety of assumptions about the adversary's knowledge under black-box access, like known model architectures and label-only access. Finally, we evaluate the effectiveness of previously proposed defenses and introduce new defenses. We find that although noise-based defenses appear to be ineffective, a simple re-sampling defense can be highly effective. Code is available at https://github.com/iamgroot42/dissecting_distribution_inference



## **46. SAIF: Sparse Adversarial and Interpretable Attack Framework**

cs.CV

**SubmitDate**: 2022-12-14    [abs](http://arxiv.org/abs/2212.07495v1) [paper-pdf](http://arxiv.org/pdf/2212.07495v1)

**Authors**: Tooba Imtiaz, Morgan Kohler, Jared Miller, Zifeng Wang, Mario Sznaier, Octavia Camps, Jennifer Dy

**Abstract**: Adversarial attacks hamper the decision-making ability of neural networks by perturbing the input signal. The addition of calculated small distortion to images, for instance, can deceive a well-trained image classification network. In this work, we propose a novel attack technique called Sparse Adversarial and Interpretable Attack Framework (SAIF). Specifically, we design imperceptible attacks that contain low-magnitude perturbations at a small number of pixels and leverage these sparse attacks to reveal the vulnerability of classifiers. We use the Frank-Wolfe (conditional gradient) algorithm to simultaneously optimize the attack perturbations for bounded magnitude and sparsity with $O(1/\sqrt{T})$ convergence. Empirical results show that SAIF computes highly imperceptible and interpretable adversarial examples, and outperforms state-of-the-art sparse attack methods on the ImageNet dataset.



## **47. XRand: Differentially Private Defense against Explanation-Guided Attacks**

cs.LG

To be published at AAAI 2023

**SubmitDate**: 2022-12-14    [abs](http://arxiv.org/abs/2212.04454v3) [paper-pdf](http://arxiv.org/pdf/2212.04454v3)

**Authors**: Truc Nguyen, Phung Lai, NhatHai Phan, My T. Thai

**Abstract**: Recent development in the field of explainable artificial intelligence (XAI) has helped improve trust in Machine-Learning-as-a-Service (MLaaS) systems, in which an explanation is provided together with the model prediction in response to each query. However, XAI also opens a door for adversaries to gain insights into the black-box models in MLaaS, thereby making the models more vulnerable to several attacks. For example, feature-based explanations (e.g., SHAP) could expose the top important features that a black-box model focuses on. Such disclosure has been exploited to craft effective backdoor triggers against malware classifiers. To address this trade-off, we introduce a new concept of achieving local differential privacy (LDP) in the explanations, and from that we establish a defense, called XRand, against such attacks. We show that our mechanism restricts the information that the adversary can learn about the top important features, while maintaining the faithfulness of the explanations.



## **48. The Devil is in the GAN: Backdoor Attacks and Defenses in Deep Generative Models**

cs.CR

17 pages, 11 figures, 3 tables

**SubmitDate**: 2022-12-14    [abs](http://arxiv.org/abs/2108.01644v2) [paper-pdf](http://arxiv.org/pdf/2108.01644v2)

**Authors**: Ambrish Rawat, Killian Levacher, Mathieu Sinn

**Abstract**: Deep Generative Models (DGMs) are a popular class of deep learning models which find widespread use because of their ability to synthesize data from complex, high-dimensional manifolds. However, even with their increasing industrial adoption, they haven't been subject to rigorous security and privacy analysis. In this work we examine one such aspect, namely backdoor attacks on DGMs which can significantly limit the applicability of pre-trained models within a model supply chain and at the very least cause massive reputation damage for companies outsourcing DGMs form third parties.   While similar attacks scenarios have been studied in the context of classical prediction models, their manifestation in DGMs hasn't received the same attention. To this end we propose novel training-time attacks which result in corrupted DGMs that synthesize regular data under normal operations and designated target outputs for inputs sampled from a trigger distribution. These attacks are based on an adversarial loss function that combines the dual objectives of attack stealth and fidelity. We systematically analyze these attacks, and show their effectiveness for a variety of approaches like Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs), as well as different data domains including images and audio. Our experiments show that - even for large-scale industry-grade DGMs (like StyleGAN) - our attacks can be mounted with only modest computational effort. We also motivate suitable defenses based on static/dynamic model and output inspections, demonstrate their usefulness, and prescribe a practical and comprehensive defense strategy that paves the way for safe usage of DGMs.



## **49. Object-fabrication Targeted Attack for Object Detection**

cs.CV

**SubmitDate**: 2022-12-14    [abs](http://arxiv.org/abs/2212.06431v2) [paper-pdf](http://arxiv.org/pdf/2212.06431v2)

**Authors**: Xuchong Zhang, Changfeng Sun, Haoliang Han, Hang Wang, Hongbin Sun, Nanning Zheng

**Abstract**: Recent researches show that the deep learning based object detection is vulnerable to adversarial examples. Generally, the adversarial attack for object detection contains targeted attack and untargeted attack. According to our detailed investigations, the research on the former is relatively fewer than the latter and all the existing methods for the targeted attack follow the same mode, i.e., the object-mislabeling mode that misleads detectors to mislabel the detected object as a specific wrong label. However, this mode has limited attack success rate, universal and generalization performances. In this paper, we propose a new object-fabrication targeted attack mode which can mislead detectors to `fabricate' extra false objects with specific target labels. Furthermore, we design a dual attention based targeted feature space attack method to implement the proposed targeted attack mode. The attack performances of the proposed mode and method are evaluated on MS COCO and BDD100K datasets using FasterRCNN and YOLOv5. Evaluation results demonstrate that, the proposed object-fabrication targeted attack mode and the corresponding targeted feature space attack method show significant improvements in terms of image-specific attack, universal performance and generalization capability, compared with the previous targeted attack for object detection. Code will be made available.



## **50. ARCADE: Adversarially Regularized Convolutional Autoencoder for Network Anomaly Detection**

cs.LG

**SubmitDate**: 2022-12-14    [abs](http://arxiv.org/abs/2205.01432v3) [paper-pdf](http://arxiv.org/pdf/2205.01432v3)

**Authors**: Willian T. Lunardi, Martin Andreoni Lopez, Jean-Pierre Giacalone

**Abstract**: As the number of heterogenous IP-connected devices and traffic volume increase, so does the potential for security breaches. The undetected exploitation of these breaches can bring severe cybersecurity and privacy risks. Anomaly-based \acp{IDS} play an essential role in network security. In this paper, we present a practical unsupervised anomaly-based deep learning detection system called ARCADE (Adversarially Regularized Convolutional Autoencoder for unsupervised network anomaly DEtection). With a convolutional \ac{AE}, ARCADE automatically builds a profile of the normal traffic using a subset of raw bytes of a few initial packets of network flows so that potential network anomalies and intrusions can be efficiently detected before they cause more damage to the network. ARCADE is trained exclusively on normal traffic. An adversarial training strategy is proposed to regularize and decrease the \ac{AE}'s capabilities to reconstruct network flows that are out-of-the-normal distribution, thereby improving its anomaly detection capabilities. The proposed approach is more effective than state-of-the-art deep learning approaches for network anomaly detection. Even when examining only two initial packets of a network flow, ARCADE can effectively detect malware infection and network attacks. ARCADE presents 20 times fewer parameters than baselines, achieving significantly faster detection speed and reaction time.



