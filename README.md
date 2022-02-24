# Latest Adversarial Attack Papers
**update at 2022-02-24 09:40:01**

[中文版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Sound Adversarial Audio-Visual Navigation**

cs.SD

This work aims to do an adversarial sound intervention for robust  audio-visual navigation

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10910v1)

**Authors**: Yinfeng Yu, Wenbing Huang, Fuchun Sun, Changan Chen, Yikai Wang, Xiaohong Liu

**Abstracts**: Audio-visual navigation task requires an agent to find a sound source in a realistic, unmapped 3D environment by utilizing egocentric audio-visual observations. Existing audio-visual navigation works assume a clean environment that solely contains the target sound, which, however, would not be suitable in most real-world applications due to the unexpected sound noise or intentional interference. In this work, we design an acoustically complex environment in which, besides the target sound, there exists a sound attacker playing a zero-sum game with the agent. More specifically, the attacker can move and change the volume and category of the sound to make the agent suffer from finding the sounding object while the agent tries to dodge the attack and navigate to the goal under the intervention. Under certain constraints to the attacker, we can improve the robustness of the agent towards unexpected sound attacks in audio-visual navigation. For better convergence, we develop a joint training mechanism by employing the property of a centralized critic with decentralized actors. Experiments on two real-world 3D scan datasets, Replica, and Matterport3D, verify the effectiveness and the robustness of the agent trained under our designed environment when transferred to the clean environment or the one containing sound attackers with random policy. Project: \url{https://yyf17.github.io/SAAVN}.



## **2. DEMO: Relay/Replay Attacks on GNSS signals**

cs.CR

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10897v1)

**Authors**: M. Lenhart, M. Spanghero, P. Papadimitratos

**Abstracts**: Global Navigation Satellite Systems (GNSS) are ubiquitously relied upon for positioning and timing. Detection and prevention of attacks against GNSS have been researched over the last decades, but many of these attacks and countermeasures were evaluated based on simulation. This work contributes to the experimental investigation of GNSS vulnerabilities, implementing a relay/replay attack with off-the-shelf hardware. Operating at the signal level, this attack type is not hindered by cryptographically protected transmissions, such as Galileo's Open Signals Navigation Message Authentication (OS-NMA). The attack we investigate involves two colluding adversaries, relaying signals over large distances, to effectively spoof a GNSS receiver. We demonstrate the attack using off-the-shelf hardware, we investigate the requirements for such successful colluding attacks, and how they can be enhanced, e.g., allowing for finer adversarial control over the victim receiver.



## **3. Protecting GNSS-based Services using Time Offset Validation**

cs.CR

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10891v1)

**Authors**: K. Zhang, M. Spanghero, P. Papadimitratos

**Abstracts**: Global navigation satellite systems (GNSS) provide pervasive accurate positioning and timing services for a large gamut of applications, from Time based One-Time Passwords (TOPT), to power grid and cellular systems. However, there can be security concerns for the applications due to the vulnerability of GNSS. It is important to observe that GNSS receivers are components of platforms, in principle having rich connectivity to different network infrastructures. Of particular interest is the access to a variety of timing sources, as those can be used to validate GNSS-provided location and time. Therefore, we consider off-the-shelf platforms and how to detect if the GNSS receiver is attacked or not, by cross-checking the GNSS time and time from other available sources. First, we survey different technologies to analyze their availability, accuracy, and trustworthiness for time synchronization. Then, we propose a validation approach for absolute and relative time. Moreover, we design a framework and experimental setup for the evaluation of the results. Attacks can be detected based on WiFi supplied time when the adversary shifts the GNSS provided time, more than 23.942us; with Network Time Protocol (NTP) supplied time when the adversary-induced shift is more than 2.046ms. Consequently, the proposal significantly limits the capability of an adversary to manipulate the victim GNSS receiver.



## **4. FastZIP: Faster and More Secure Zero-Interaction Pairing**

cs.CR

ACM MobiSys '21; Fixed ambiguity in flow diagram (Figure 2). Code and  data are available at: https://github.com/seemoo-lab/fastzip

**SubmitDate**: 2022-02-23    [paper-pdf](http://arxiv.org/pdf/2106.04907v3)

**Authors**: Mikhail Fomichev, Julia Hesse, Lars Almon, Timm Lippert, Jun Han, Matthias Hollick

**Abstracts**: With the advent of the Internet of Things (IoT), establishing a secure channel between smart devices becomes crucial. Recent research proposes zero-interaction pairing (ZIP), which enables pairing without user assistance by utilizing devices' physical context (e.g., ambient audio) to obtain a shared secret key. The state-of-the-art ZIP schemes suffer from three limitations: (1) prolonged pairing time (i.e., minutes or hours), (2) vulnerability to brute-force offline attacks on a shared key, and (3) susceptibility to attacks caused by predictable context (e.g., replay attack) because they rely on limited entropy of physical context to protect a shared key. We address these limitations, proposing FastZIP, a novel ZIP scheme that significantly reduces pairing time while preventing offline and predictable context attacks. In particular, we adapt a recently introduced Fuzzy Password-Authenticated Key Exchange (fPAKE) protocol and utilize sensor fusion, maximizing their advantages. We instantiate FastZIP for intra-car device pairing to demonstrate its feasibility and show how the design of FastZIP can be adapted to other ZIP use cases. We implement FastZIP and evaluate it by driving four cars for a total of 800 km. We achieve up to three times shorter pairing time compared to the state-of-the-art ZIP schemes while assuring robust security with adversarial error rates below 0.5%.



## **5. Adversarial Defense by Latent Style Transformations**

cs.CV

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2006.09701v2)

**Authors**: Shuo Wang, Surya Nepal, Alsharif Abuadbba, Carsten Rudolph, Marthie Grobler

**Abstracts**: Machine learning models have demonstrated vulnerability to adversarial attacks, more specifically misclassification of adversarial examples.   In this paper, we investigate an attack-agnostic defense against adversarial attacks on high-resolution images by detecting suspicious inputs.   The intuition behind our approach is that the essential characteristics of a normal image are generally consistent with non-essential style transformations, e.g., slightly changing the facial expression of human portraits.   In contrast, adversarial examples are generally sensitive to such transformations.   In our approach to detect adversarial instances, we propose an in\underline{V}ertible \underline{A}utoencoder based on the \underline{S}tyleGAN2 generator via \underline{A}dversarial training (VASA) to inverse images to disentangled latent codes that reveal hierarchical styles.   We then build a set of edited copies with non-essential style transformations by performing latent shifting and reconstruction, based on the correspondences between latent codes and style transformations.   The classification-based consistency of these edited copies is used to distinguish adversarial instances.



## **6. Surrogate Representation Learning with Isometric Mapping for Gray-box Graph Adversarial Attacks**

cs.AI

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2110.10482v3)

**Authors**: Zihan Liu, Yun Luo, Zelin Zang, Stan Z. Li

**Abstracts**: Gray-box graph attacks aim at disrupting the performance of the victim model by using inconspicuous attacks with limited knowledge of the victim model. The parameters of the victim model and the labels of the test nodes are invisible to the attacker. To obtain the gradient on the node attributes or graph structure, the attacker constructs an imaginary surrogate model trained under supervision. However, there is a lack of discussion on the training of surrogate models and the robustness of provided gradient information. The general node classification model loses the topology of the nodes on the graph, which is, in fact, an exploitable prior for the attacker. This paper investigates the effect of representation learning of surrogate models on the transferability of gray-box graph adversarial attacks. To reserve the topology in the surrogate embedding, we propose Surrogate Representation Learning with Isometric Mapping (SRLIM). By using Isometric mapping method, our proposed SRLIM can constrain the topological structure of nodes from the input layer to the embedding space, that is, to maintain the similarity of nodes in the propagation process. Experiments prove the effectiveness of our approach through the improvement in the performance of the adversarial attacks generated by the gradient-based attacker in untargeted poisoning gray-box setups.



## **7. Universal adversarial perturbation for remote sensing images**

cs.CV

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10693v1)

**Authors**: Zhaoxia Yin, Qingyu Wang, Jin Tang, Bin Luo

**Abstracts**: Recently, with the application of deep learning in the remote sensing image (RSI) field, the classification accuracy of the RSI has been greatly improved compared with traditional technology. However, even state-of-the-art object recognition convolutional neural networks are fooled by the universal adversarial perturbation (UAP). To verify that UAP makes the RSI classification model error classification, this paper proposes a novel method combining an encoder-decoder network with an attention mechanism. Firstly, the former can learn the distribution of perturbations better, then the latter is used to find the main regions concerned by the RSI classification model. Finally, the generated regions are used to fine-tune the perturbations making the model misclassified with fewer perturbations. The experimental results show that the UAP can make the RSI misclassify, and the attack success rate (ASR) of our proposed method on the RSI data set is as high as 97.35%.



## **8. Seeing is Living? Rethinking the Security of Facial Liveness Verification in the Deepfake Era**

cs.CR

Accepted as a full paper at USENIX Security '22

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10673v1)

**Authors**: Changjiang Li, Li Wang, Shouling Ji, Xuhong Zhang, Zhaohan Xi, Shanqing Guo, Ting Wang

**Abstracts**: Facial Liveness Verification (FLV) is widely used for identity authentication in many security-sensitive domains and offered as Platform-as-a-Service (PaaS) by leading cloud vendors. Yet, with the rapid advances in synthetic media techniques (e.g., deepfake), the security of FLV is facing unprecedented challenges, about which little is known thus far.   To bridge this gap, in this paper, we conduct the first systematic study on the security of FLV in real-world settings. Specifically, we present LiveBugger, a new deepfake-powered attack framework that enables customizable, automated security evaluation of FLV. Leveraging LiveBugger, we perform a comprehensive empirical assessment of representative FLV platforms, leading to a set of interesting findings. For instance, most FLV APIs do not use anti-deepfake detection; even for those with such defenses, their effectiveness is concerning (e.g., it may detect high-quality synthesized videos but fail to detect low-quality ones). We then conduct an in-depth analysis of the factors impacting the attack performance of LiveBugger: a) the bias (e.g., gender or race) in FLV can be exploited to select victims; b) adversarial training makes deepfake more effective to bypass FLV; c) the input quality has a varying influence on different deepfake techniques to bypass FLV. Based on these findings, we propose a customized, two-stage approach that can boost the attack success rate by up to 70%. Further, we run proof-of-concept attacks on several representative applications of FLV (i.e., the clients of FLV APIs) to illustrate the practical implications: due to the vulnerability of the APIs, many downstream applications are vulnerable to deepfake. Finally, we discuss potential countermeasures to improve the security of FLV. Our findings have been confirmed by the corresponding vendors.



## **9. Fingerprinting Deep Neural Networks Globally via Universal Adversarial Perturbations**

cs.CR

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.08602v2)

**Authors**: Zirui Peng, Shaofeng Li, Guoxing Chen, Cheng Zhang, Haojin Zhu, Minhui Xue

**Abstracts**: In this paper, we propose a novel and practical mechanism which enables the service provider to verify whether a suspect model is stolen from the victim model via model extraction attacks. Our key insight is that the profile of a DNN model's decision boundary can be uniquely characterized by its \textit{Universal Adversarial Perturbations (UAPs)}. UAPs belong to a low-dimensional subspace and piracy models' subspaces are more consistent with victim model's subspace compared with non-piracy model. Based on this, we propose a UAP fingerprinting method for DNN models and train an encoder via \textit{contrastive learning} that takes fingerprint as inputs, outputs a similarity score. Extensive studies show that our framework can detect model IP breaches with confidence $> 99.99 \%$ within only $20$ fingerprints of the suspect model. It has good generalizability across different model architectures and is robust against post-modifications on stolen models.



## **10. Robust Stochastic Linear Contextual Bandits Under Adversarial Attacks**

stat.ML

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2106.02978v2)

**Authors**: Qin Ding, Cho-Jui Hsieh, James Sharpnack

**Abstracts**: Stochastic linear contextual bandit algorithms have substantial applications in practice, such as recommender systems, online advertising, clinical trials, etc. Recent works show that optimal bandit algorithms are vulnerable to adversarial attacks and can fail completely in the presence of attacks. Existing robust bandit algorithms only work for the non-contextual setting under the attack of rewards and cannot improve the robustness in the general and popular contextual bandit environment. In addition, none of the existing methods can defend against attacked context. In this work, we provide the first robust bandit algorithm for stochastic linear contextual bandit setting under a fully adaptive and omniscient attack with sub-linear regret. Our algorithm not only works under the attack of rewards, but also under attacked context. Moreover, it does not need any information about the attack budget or the particular form of the attack. We provide theoretical guarantees for our proposed algorithm and show by experiments that our proposed algorithm improves the robustness against various kinds of popular attacks.



## **11. Behaviour-Diverse Automatic Penetration Testing: A Curiosity-Driven Multi-Objective Deep Reinforcement Learning Approach**

cs.LG

6 pages,4 Figures

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10630v1)

**Authors**: Yizhou Yang, Xin Liu

**Abstracts**: Penetration Testing plays a critical role in evaluating the security of a target network by emulating real active adversaries. Deep Reinforcement Learning (RL) is seen as a promising solution to automating the process of penetration tests by reducing human effort and improving reliability. Existing RL solutions focus on finding a specific attack path to impact the target hosts. However, in reality, a diverse range of attack variations are needed to provide comprehensive assessments of the target network's security level. Hence, the attack agents must consider multiple objectives when penetrating the network. Nevertheless, this challenge is not adequately addressed in the existing literature. To this end, we formulate the automatic penetration testing in the Multi-Objective Reinforcement Learning (MORL) framework and propose a Chebyshev decomposition critic to find diverse adversary strategies that balance different objectives in the penetration test. Additionally, the number of available actions increases with the agent consistently probing the target network, making the training process intractable in many practical situations. Thus, we introduce a coverage-based masking mechanism that reduces attention on previously selected actions to help the agent adapt to future exploration. Experimental evaluation on a range of scenarios demonstrates the superiority of our proposed approach when compared to adapted algorithms in terms of multi-objective learning and performance efficiency.



## **12. On the Effectiveness of Adversarial Training against Backdoor Attacks**

cs.LG

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10627v1)

**Authors**: Yinghua Gao, Dongxian Wu, Jingfeng Zhang, Guanhao Gan, Shu-Tao Xia, Gang Niu, Masashi Sugiyama

**Abstracts**: DNNs' demand for massive data forces practitioners to collect data from the Internet without careful check due to the unacceptable cost, which brings potential risks of backdoor attacks. A backdoored model always predicts a target class in the presence of a predefined trigger pattern, which can be easily realized via poisoning a small amount of data. In general, adversarial training is believed to defend against backdoor attacks since it helps models to keep their prediction unchanged even if we perturb the input image (as long as within a feasible range). Unfortunately, few previous studies succeed in doing so. To explore whether adversarial training could defend against backdoor attacks or not, we conduct extensive experiments across different threat models and perturbation budgets, and find the threat model in adversarial training matters. For instance, adversarial training with spatial adversarial examples provides notable robustness against commonly-used patch-based backdoor attacks. We further propose a hybrid strategy which provides satisfactory robustness across different backdoor attacks.



## **13. Adversarial Attacks on Speech Recognition Systems for Mission-Critical Applications: A Survey**

cs.SD

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10594v1)

**Authors**: Ngoc Dung Huynh, Mohamed Reda Bouadjenek, Imran Razzak, Kevin Lee, Chetan Arora, Ali Hassani, Arkady Zaslavsky

**Abstracts**: A Machine-Critical Application is a system that is fundamentally necessary to the success of specific and sensitive operations such as search and recovery, rescue, military, and emergency management actions. Recent advances in Machine Learning, Natural Language Processing, voice recognition, and speech processing technologies have naturally allowed the development and deployment of speech-based conversational interfaces to interact with various machine-critical applications. While these conversational interfaces have allowed users to give voice commands to carry out strategic and critical activities, their robustness to adversarial attacks remains uncertain and unclear. Indeed, Adversarial Artificial Intelligence (AI) which refers to a set of techniques that attempt to fool machine learning models with deceptive data, is a growing threat in the AI and machine learning research community, in particular for machine-critical applications. The most common reason of adversarial attacks is to cause a malfunction in a machine learning model. An adversarial attack might entail presenting a model with inaccurate or fabricated samples as it's training data, or introducing maliciously designed data to deceive an already trained model. While focusing on speech recognition for machine-critical applications, in this paper, we first review existing speech recognition techniques, then, we investigate the effectiveness of adversarial attacks and defenses against these systems, before outlining research challenges, defense recommendations, and future work. This paper is expected to serve researchers and practitioners as a reference to help them in understanding the challenges, position themselves and, ultimately, help them to improve existing models of speech recognition for mission-critical applications. Keywords: Mission-Critical Applications, Adversarial AI, Speech Recognition Systems.



## **14. Privacy Leakage of Adversarial Training Models in Federated Learning Systems**

cs.LG

6 pages, 6 figures. Submitted to CVPR'22 workshop "The Art of  Robustness"

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2202.10546v1)

**Authors**: Jingyang Zhang, Yiran Chen, Hai Li

**Abstracts**: Adversarial Training (AT) is crucial for obtaining deep neural networks that are robust to adversarial attacks, yet recent works found that it could also make models more vulnerable to privacy attacks. In this work, we further reveal this unsettling property of AT by designing a novel privacy attack that is practically applicable to the privacy-sensitive Federated Learning (FL) systems. Using our method, the attacker can exploit AT models in the FL system to accurately reconstruct users' private training images even when the training batch size is large. Code is available at https://github.com/zjysteven/PrivayAttack_AT_FL.



## **15. Analysing Security and Privacy Threats in the Lockdown Periods of COVID-19 Pandemic: Twitter Dataset Case Study**

cs.CR

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2202.10543v1)

**Authors**: Bibhas Sharma, Ishan Karunanayake, Rahat Masood, Muhammad Ikram

**Abstracts**: The COVID-19 pandemic will be remembered as a uniquely disruptive period that altered the lives of billions of citizens globally, resulting in new-normal for the way people live and work. With the coronavirus pandemic, everyone had to adapt to the "work or study from home" operating model that has transformed our online lives and exponentially increased the use of cyberspace. Concurrently, there has been a huge spike in social media platforms such as Facebook and Twitter during the COVID-19 lockdown periods. These lockdown periods have resulted in a set of new cybercrimes, thereby allowing attackers to victimise users of social media platforms in times of fear, uncertainty, and doubt. The threats range from running phishing campaigns and malicious domains to extracting private information about victims for malicious purposes. This research paper performs a large-scale study to investigate the impact of lockdown periods during the COVID-19 pandemic on the security and privacy of social media users. We analyse 10.6 Million COVID-related tweets from 533 days of data crawling and investigate users' security and privacy behaviour in three different periods (i.e., before, during, and after lockdown). Our study shows that users unintentionally share more personal identifiable information when writing about the pandemic situation in their tweets. The privacy risk reaches 100% if a user posts three or more sensitive tweets about the pandemic. We investigate the number of suspicious domains shared in social media during different pandemic phases. Our analysis reveals an increase in suspicious domains during the lockdown compared to other lockdown phases. We observe that IT, Search Engines, and Businesses are the top three categories that contain suspicious domains. Our analysis reveals that adversaries' strategies to instigate malicious activities change with the country's pandemic situation.



## **16. RAILS: A Robust Adversarial Immune-inspired Learning System**

cs.NE

arXiv admin note: text overlap with arXiv:2012.10485

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2107.02840v2)

**Authors**: Ren Wang, Tianqi Chen, Stephen Lindsly, Cooper Stansbury, Alnawaz Rehemtulla, Indika Rajapakse, Alfred Hero

**Abstracts**: Adversarial attacks against deep neural networks (DNNs) are continuously evolving, requiring increasingly powerful defense strategies. We develop a novel adversarial defense framework inspired by the adaptive immune system: the Robust Adversarial Immune-inspired Learning System (RAILS). Initializing a population of exemplars that is balanced across classes, RAILS starts from a uniform label distribution that encourages diversity and uses an evolutionary optimization process to adaptively adjust the predictive label distribution in a manner that emulates the way the natural immune system recognizes novel pathogens. RAILS' evolutionary optimization process explicitly captures the tradeoff between robustness (diversity) and accuracy (specificity) of the network, and represents a new immune-inspired perspective on adversarial learning. The benefits of RAILS are empirically demonstrated under eight types of adversarial attacks on a DNN adversarial image classifier for several benchmark datasets, including: MNIST; SVHN; CIFAR-10; and CIFAR-10. We find that PGD is the most damaging attack strategy and that for this attack RAILS is significantly more robust than other methods, achieving improvements in adversarial robustness by $\geq 5.62\%, 12.5\%$, $10.32\%$, and $8.39\%$, on these respective datasets, without appreciable loss of classification accuracy. Codes for the results in this paper are available at https://github.com/wangren09/RAILS.



## **17. Adversarial Examples in Constrained Domains**

cs.CR

Accepted to IOS Press Journal of Computer Security

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2011.01183v2)

**Authors**: Ryan Sheatsley, Nicolas Papernot, Michael Weisman, Gunjan Verma, Patrick McDaniel

**Abstracts**: Machine learning algorithms have been shown to be vulnerable to adversarial manipulation through systematic modification of inputs (e.g., adversarial examples) in domains such as image recognition. Under the default threat model, the adversary exploits the unconstrained nature of images; each feature (pixel) is fully under control of the adversary. However, it is not clear how these attacks translate to constrained domains that limit which and how features can be modified by the adversary (e.g., network intrusion detection). In this paper, we explore whether constrained domains are less vulnerable than unconstrained domains to adversarial example generation algorithms. We create an algorithm for generating adversarial sketches: targeted universal perturbation vectors which encode feature saliency within the envelope of domain constraints. To assess how these algorithms perform, we evaluate them in constrained (e.g., network intrusion detection) and unconstrained (e.g., image recognition) domains. The results demonstrate that our approaches generate misclassification rates in constrained domains that were comparable to those of unconstrained domains (greater than 95%). Our investigation shows that the narrow attack surface exposed by constrained domains is still sufficiently large to craft successful adversarial examples; and thus, constraints do not appear to make a domain robust. Indeed, with as little as five randomly selected features, one can still generate adversarial examples.



## **18. A Tutorial on Adversarial Learning Attacks and Countermeasures**

cs.CR

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2202.10377v1)

**Authors**: Cato Pauling, Michael Gimson, Muhammed Qaid, Ahmad Kida, Basel Halak

**Abstracts**: Machine learning algorithms are used to construct a mathematical model for a system based on training data. Such a model is capable of making highly accurate predictions without being explicitly programmed to do so. These techniques have a great many applications in all areas of the modern digital economy and artificial intelligence. More importantly, these methods are essential for a rapidly increasing number of safety-critical applications such as autonomous vehicles and intelligent defense systems. However, emerging adversarial learning attacks pose a serious security threat that greatly undermines further such systems. The latter are classified into four types, evasion (manipulating data to avoid detection), poisoning (injection malicious training samples to disrupt retraining), model stealing (extraction), and inference (leveraging over-generalization on training data). Understanding this type of attacks is a crucial first step for the development of effective countermeasures. The paper provides a detailed tutorial on the principles of adversarial machining learning, explains the different attack scenarios, and gives an in-depth insight into the state-of-art defense mechanisms against this rising threat .



## **19. Cyber-Physical Defense in the Quantum Era**

cs.CR

14 pages, 7 figures, 1 table, 4 boxes

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2202.10354v1)

**Authors**: Michel Barbeau, Joaquin Garcia-Alfaro

**Abstracts**: Networked-Control Systems (NCSs), a type of cyber-physical systems, consist of tightly integrated computing, communication and control technologies. While being very flexible environments, they are vulnerable to computing and networking attacks. Recent NCSs hacking incidents had major impact. They call for more research on cyber-physical security. Fears about the use of quantum computing to break current cryptosystems make matters worse. While the quantum threat motivated the creation of new disciplines to handle the issue, such as post-quantum cryptography, other fields have overlooked the existence of quantum-enabled adversaries. This is the case of cyber-physical defense research, a distinct but complementary discipline to cyber-physical protection. Cyber-physical defense refers to the capability to detect and react in response to cyber-physical attacks. Concretely, it involves the integration of mechanisms to identify adverse events and prepare response plans, during and after incidents occur. In this paper, we make the assumption that the eventually available quantum computer will provide an advantage to adversaries against defenders, unless they also adopt this technology. We envision the necessity for a paradigm shift, where an increase of adversarial resources because of quantum supremacy does not translate into higher likelihood of disruptions. Consistently with current system design practices in other areas, such as the use of artificial intelligence for the reinforcement of attack detection tools, we outline a vision for next generation cyber-physical defense layers leveraging ideas from quantum computing and machine learning. Through an example, we show that defenders of NCSs can learn and improve their strategies to anticipate and recover from attacks.



## **20. Measurement-Device-Independent Quantum Secure Direct Communication with User Authentication**

quant-ph

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2202.10316v1)

**Authors**: Nayana Das, Goutam Paul

**Abstracts**: Quantum secure direct communication (QSDC) and deterministic secure quantum communication (DSQC) are two important branches of quantum cryptography, where one can transmit a secret message securely without encrypting it by a prior key. In the practical scenario, an adversary can apply detector-side-channel attacks to get some non-negligible amount of information about the secret message. Measurement-device-independent (MDI) quantum protocols can remove this kind of detector-side-channel attack, by introducing an untrusted third party (UTP), who performs all the measurements during the protocol with imperfect measurement devices. In this paper, we put forward the first MDI-QSDC protocol with user identity authentication, where both the sender and the receiver first check the authenticity of the other party and then exchange the secret message. Then we extend this to an MDI quantum dialogue (QD) protocol, where both the parties can send their respective secret messages after verifying the identity of the other party. Along with this, we also report the first MDI-DSQC protocol with user identity authentication. Theoretical analyses prove the security of our proposed protocols against common attacks.



## **21. HoneyModels: Machine Learning Honeypots**

cs.CR

Published in: MILCOM 2021 - 2021 IEEE Military Communications  Conference (MILCOM)

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2202.10309v1)

**Authors**: Ahmed Abdou, Ryan Sheatsley, Yohan Beugin, Tyler Shipp, Patrick McDaniel

**Abstracts**: Machine Learning is becoming a pivotal aspect of many systems today, offering newfound performance on classification and prediction tasks, but this rapid integration also comes with new unforeseen vulnerabilities. To harden these systems the ever-growing field of Adversarial Machine Learning has proposed new attack and defense mechanisms. However, a great asymmetry exists as these defensive methods can only provide security to certain models and lack scalability, computational efficiency, and practicality due to overly restrictive constraints. Moreover, newly introduced attacks can easily bypass defensive strategies by making subtle alterations. In this paper, we study an alternate approach inspired by honeypots to detect adversaries. Our approach yields learned models with an embedded watermark. When an adversary initiates an interaction with our model, attacks are encouraged to add this predetermined watermark stimulating detection of adversarial examples. We show that HoneyModels can reveal 69.5% of adversaries attempting to attack a Neural Network while preserving the original functionality of the model. HoneyModels offer an alternate direction to secure Machine Learning that slightly affects the accuracy while encouraging the creation of watermarked adversarial samples detectable by the HoneyModel but indistinguishable from others for the adversary.



## **22. Hardware Obfuscation of Digital FIR Filters**

cs.CR

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2202.10022v1)

**Authors**: Levent Aksoy, Alexander Hepp, Johanna Baehr, Samuel Pagliarini

**Abstracts**: A finite impulse response (FIR) filter is a ubiquitous block in digital signal processing applications. Its characteristics are determined by its coefficients, which are the intellectual property (IP) for its designer. However, in a hardware efficient realization, its coefficients become vulnerable to reverse engineering. This paper presents a filter design technique that can protect this IP, taking into account hardware complexity and ensuring that the filter behaves as specified only when a secret key is provided. To do so, coefficients are hidden among decoys, which are selected beyond possible values of coefficients using three alternative methods. As an attack scenario, an adversary at an untrusted foundry is considered. A reverse engineering technique is developed to find the chosen decoy selection method and explore the potential leakage of coefficients through decoys. An oracle-less attack is also used to find the secret key. Experimental results show that the proposed technique can lead to filter designs with competitive hardware complexity and higher resiliency to attacks with respect to previously proposed methods.



## **23. Learning to Attack with Fewer Pixels: A Probabilistic Post-hoc Framework for Refining Arbitrary Dense Adversarial Attacks**

cs.CV

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2010.06131v2)

**Authors**: He Zhao, Thanh Nguyen, Trung Le, Paul Montague, Olivier De Vel, Tamas Abraham, Dinh Phung

**Abstracts**: Deep neural network image classifiers are reported to be susceptible to adversarial evasion attacks, which use carefully crafted images created to mislead a classifier. Many adversarial attacks belong to the category of dense attacks, which generate adversarial examples by perturbing all the pixels of a natural image. To generate sparse perturbations, sparse attacks have been recently developed, which are usually independent attacks derived by modifying a dense attack's algorithm with sparsity regularisations, resulting in reduced attack efficiency. In this paper, we aim to tackle this task from a different perspective. We select the most effective perturbations from the ones generated from a dense attack, based on the fact we find that a considerable amount of the perturbations on an image generated by dense attacks may contribute little to attacking a classifier. Accordingly, we propose a probabilistic post-hoc framework that refines given dense attacks by significantly reducing the number of perturbed pixels but keeping their attack power, trained with mutual information maximisation. Given an arbitrary dense attack, the proposed model enjoys appealing compatibility for making its adversarial images more realistic and less detectable with fewer perturbations. Moreover, our framework performs adversarial attacks much faster than existing sparse attacks.



## **24. Transferring Adversarial Robustness Through Robust Representation Matching**

cs.LG

To appear at USENIX'22

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2202.09994v1)

**Authors**: Pratik Vaishnavi, Kevin Eykholt, Amir Rahmati

**Abstracts**: With the widespread use of machine learning, concerns over its security and reliability have become prevalent. As such, many have developed defenses to harden neural networks against adversarial examples, imperceptibly perturbed inputs that are reliably misclassified. Adversarial training in which adversarial examples are generated and used during training is one of the few known defenses able to reliably withstand such attacks against neural networks. However, adversarial training imposes a significant training overhead and scales poorly with model complexity and input dimension. In this paper, we propose Robust Representation Matching (RRM), a low-cost method to transfer the robustness of an adversarially trained model to a new model being trained for the same task irrespective of architectural differences. Inspired by student-teacher learning, our method introduces a novel training loss that encourages the student to learn the teacher's robust representations. Compared to prior works, RRM is superior with respect to both model performance and adversarial training time. On CIFAR-10, RRM trains a robust model $\sim 1.8\times$ faster than the state-of-the-art. Furthermore, RRM remains effective on higher-dimensional datasets. On Restricted-ImageNet, RRM trains a ResNet50 model $\sim 18\times$ faster than standard adversarial training.



## **25. Overparametrization improves robustness against adversarial attacks: A replication study**

cs.LG

**SubmitDate**: 2022-02-20    [paper-pdf](http://arxiv.org/pdf/2202.09735v1)

**Authors**: Ali Borji

**Abstracts**: Overparametrization has become a de facto standard in machine learning. Despite numerous efforts, our understanding of how and where overparametrization helps model accuracy and robustness is still limited. To this end, here we conduct an empirical investigation to systemically study and replicate previous findings in this area, in particular the study by Madry et al. Together with this study, our findings support the "universal law of robustness" recently proposed by Bubeck et al. We argue that while critical for robust perception, overparametrization may not be enough to achieve full robustness and smarter architectures e.g. the ones implemented by the human visual cortex) seem inevitable.



## **26. Runtime-Assured, Real-Time Neural Control of Microgrids**

eess.SY

**SubmitDate**: 2022-02-20    [paper-pdf](http://arxiv.org/pdf/2202.09710v1)

**Authors**: Amol Damare, Shouvik Roy, Scott A. Smolka, Scott D. Stoller

**Abstracts**: We present SimpleMG, a new, provably correct design methodology for runtime assurance of microgrids (MGs) with neural controllers. Our approach is centered around the Neural Simplex Architecture, which in turn is based on Sha et al.'s Simplex Control Architecture. Reinforcement Learning is used to synthesize high-performance neural controllers for MGs. Barrier Certificates are used to establish SimpleMG's runtime-assurance guarantees. We present a novel method to derive the condition for switching from the unverified neural controller to the verified-safe baseline controller, and we prove that the method is correct. We conduct an extensive experimental evaluation of SimpleMG using RTDS, a high-fidelity, real-time simulation environment for power systems, on a realistic model of a microgrid comprising three distributed energy resources (battery, photovoltaic, and diesel generator). Our experiments confirm that SimpleMG can be used to develop high-performance neural controllers for complex microgrids while assuring runtime safety, even in the presence of adversarial input attacks on the neural controller. Our experiments also demonstrate the benefits of online retraining of the neural controller while the baseline controller is in control



## **27. Detection of Stealthy Adversaries for Networked Unmanned Aerial Vehicles**

eess.SY

**SubmitDate**: 2022-02-19    [paper-pdf](http://arxiv.org/pdf/2202.09661v1)

**Authors**: Mohammad Bahrami, Hamidreza Jafarnejadsani

**Abstracts**: A network of unmanned aerial vehicles (UAVs) provides distributed coverage, reconfigurability, and maneuverability in performing complex cooperative tasks. However, it relies on wireless communications that can be susceptible to cyber adversaries and intrusions, disrupting the entire network's operation. This paper develops model-based centralized and decentralized observer techniques for detecting a class of stealthy intrusions, namely zero-dynamics and covert attacks, on networked UAVs in formation control settings. The centralized observer that runs in a control center leverages switching in the UAVs' communication topology for attack detection, and the decentralized observers, implemented onboard each UAV in the network, use the model of networked UAVs and locally available measurements. Experimental results are provided to show the effectiveness of the proposed detection schemes in different case studies.



## **28. Stochastic sparse adversarial attacks**

cs.LG

Final version published at the ICTAI 2021 conference with a best  student paper award. Codes are available through the link:  https://github.com/hhajri/stochastic-sparse-adv-attacks

**SubmitDate**: 2022-02-19    [paper-pdf](http://arxiv.org/pdf/2011.12423v4)

**Authors**: Manon Césaire, Lucas Schott, Hatem Hajri, Sylvain Lamprier, Patrick Gallinari

**Abstracts**: This paper introduces stochastic sparse adversarial attacks (SSAA), standing as simple, fast and purely noise-based targeted and untargeted attacks of neural network classifiers (NNC). SSAA offer new examples of sparse (or $L_0$) attacks for which only few methods have been proposed previously. These attacks are devised by exploiting a small-time expansion idea widely used for Markov processes. Experiments on small and large datasets (CIFAR-10 and ImageNet) illustrate several advantages of SSAA in comparison with the-state-of-the-art methods. For instance, in the untargeted case, our method called Voting Folded Gaussian Attack (VFGA) scales efficiently to ImageNet and achieves a significantly lower $L_0$ score than SparseFool (up to $\frac{2}{5}$) while being faster. Moreover, VFGA achieves better $L_0$ scores on ImageNet than Sparse-RS when both attacks are fully successful on a large number of samples.



## **29. Internal Wasserstein Distance for Adversarial Attack and Defense**

cs.LG

**SubmitDate**: 2022-02-19    [paper-pdf](http://arxiv.org/pdf/2103.07598v2)

**Authors**: Mingkui Tan, Shuhai Zhang, Jiezhang Cao, Jincheng Li, Yanwu Xu

**Abstracts**: Deep neural networks (DNNs) are known to be vulnerable to adversarial attacks that would trigger misclassification of DNNs but may be imperceptible to human perception. Adversarial defense has been important ways to improve the robustness of DNNs. Existing attack methods often construct adversarial examples relying on some metrics like the $\ell_p$ distance to perturb samples. However, these metrics can be insufficient to conduct adversarial attacks due to their limited perturbations. In this paper, we propose a new internal Wasserstein distance (IWD) to capture the semantic similarity of two samples, and thus it helps to obtain larger perturbations than currently used metrics such as the $\ell_p$ distance We then apply the internal Wasserstein distance to perform adversarial attack and defense. In particular, we develop a novel attack method relying on IWD to calculate the similarities between an image and its adversarial examples. In this way, we can generate diverse and semantically similar adversarial examples that are more difficult to defend by existing defense methods. Moreover, we devise a new defense method relying on IWD to learn robust models against unseen adversarial examples. We provide both thorough theoretical and empirical evidence to support our methods.



## **30. Robust Reinforcement Learning as a Stackelberg Game via Adaptively-Regularized Adversarial Training**

cs.LG

**SubmitDate**: 2022-02-19    [paper-pdf](http://arxiv.org/pdf/2202.09514v1)

**Authors**: Peide Huang, Mengdi Xu, Fei Fang, Ding Zhao

**Abstracts**: Robust Reinforcement Learning (RL) focuses on improving performances under model errors or adversarial attacks, which facilitates the real-life deployment of RL agents. Robust Adversarial Reinforcement Learning (RARL) is one of the most popular frameworks for robust RL. However, most of the existing literature models RARL as a zero-sum simultaneous game with Nash equilibrium as the solution concept, which could overlook the sequential nature of RL deployments, produce overly conservative agents, and induce training instability. In this paper, we introduce a novel hierarchical formulation of robust RL - a general-sum Stackelberg game model called RRL-Stack - to formalize the sequential nature and provide extra flexibility for robust training. We develop the Stackelberg Policy Gradient algorithm to solve RRL-Stack, leveraging the Stackelberg learning dynamics by considering the adversary's response. Our method generates challenging yet solvable adversarial environments which benefit RL agents' robust learning. Our algorithm demonstrates better training stability and robustness against different testing conditions in the single-agent robotics control and multi-agent highway merging tasks.



## **31. Attacks, Defenses, And Tools: A Framework To Facilitate Robust AI/ML Systems**

cs.CR

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2202.09465v1)

**Authors**: Mohamad Fazelnia, Igor Khokhlov, Mehdi Mirakhorli

**Abstracts**: Software systems are increasingly relying on Artificial Intelligence (AI) and Machine Learning (ML) components. The emerging popularity of AI techniques in various application domains attracts malicious actors and adversaries. Therefore, the developers of AI-enabled software systems need to take into account various novel cyber-attacks and vulnerabilities that these systems may be susceptible to. This paper presents a framework to characterize attacks and weaknesses associated with AI-enabled systems and provide mitigation techniques and defense strategies. This framework aims to support software designers in taking proactive measures in developing AI-enabled software, understanding the attack surface of such systems, and developing products that are resilient to various emerging attacks associated with ML. The developed framework covers a broad spectrum of attacks, mitigation techniques, and defensive and offensive tools. In this paper, we demonstrate the framework architecture and its major components, describe their attributes, and discuss the long-term goals of this research.



## **32. Black-box Node Injection Attack for Graph Neural Networks**

cs.LG

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2202.09389v1)

**Authors**: Mingxuan Ju, Yujie Fan, Yanfang Ye, Liang Zhao

**Abstracts**: Graph Neural Networks (GNNs) have drawn significant attentions over the years and been broadly applied to vital fields that require high security standard such as product recommendation and traffic forecasting. Under such scenarios, exploiting GNN's vulnerabilities and further downgrade its classification performance become highly incentive for adversaries. Previous attackers mainly focus on structural perturbations of existing graphs. Although they deliver promising results, the actual implementation needs capability of manipulating the graph connectivity, which is impractical in some circumstances. In this work, we study the possibility of injecting nodes to evade the victim GNN model, and unlike previous related works with white-box setting, we significantly restrict the amount of accessible knowledge and explore the black-box setting. Specifically, we model the node injection attack as a Markov decision process and propose GA2C, a graph reinforcement learning framework in the fashion of advantage actor critic, to generate realistic features for injected nodes and seamlessly merge them into the original graph following the same topology characteristics. Through our extensive experiments on multiple acknowledged benchmark datasets, we demonstrate the superior performance of our proposed GA2C over existing state-of-the-art methods. The data and source code are publicly accessible at: https://github.com/jumxglhf/GA2C.



## **33. Synthetic Disinformation Attacks on Automated Fact Verification Systems**

cs.CL

AAAI 2022

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2202.09381v1)

**Authors**: Yibing Du, Antoine Bosselut, Christopher D. Manning

**Abstracts**: Automated fact-checking is a needed technology to curtail the spread of online misinformation. One current framework for such solutions proposes to verify claims by retrieving supporting or refuting evidence from related textual sources. However, the realistic use cases for fact-checkers will require verifying claims against evidence sources that could be affected by the same misinformation. Furthermore, the development of modern NLP tools that can produce coherent, fabricated content would allow malicious actors to systematically generate adversarial disinformation for fact-checkers.   In this work, we explore the sensitivity of automated fact-checkers to synthetic adversarial evidence in two simulated settings: AdversarialAddition, where we fabricate documents and add them to the evidence repository available to the fact-checking system, and AdversarialModification, where existing evidence source documents in the repository are automatically altered. Our study across multiple models on three benchmarks demonstrates that these systems suffer significant performance drops against these attacks. Finally, we discuss the growing threat of modern NLG systems as generators of disinformation in the context of the challenges they pose to automated fact-checkers.



## **34. Exploring Adversarially Robust Training for Unsupervised Domain Adaptation**

cs.CV

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2202.09300v1)

**Authors**: Shao-Yuan Lo, Vishal M. Patel

**Abstracts**: Unsupervised Domain Adaptation (UDA) methods aim to transfer knowledge from a labeled source domain to an unlabeled target domain. UDA has been extensively studied in the computer vision literature. Deep networks have been shown to be vulnerable to adversarial attacks. However, very little focus is devoted to improving the adversarial robustness of deep UDA models, causing serious concerns about model reliability. Adversarial Training (AT) has been considered to be the most successful adversarial defense approach. Nevertheless, conventional AT requires ground-truth labels to generate adversarial examples and train models, which limits its effectiveness in the unlabeled target domain. In this paper, we aim to explore AT to robustify UDA models: How to enhance the unlabeled data robustness via AT while learning domain-invariant features for UDA? To answer this, we provide a systematic study into multiple AT variants that potentially apply to UDA. Moreover, we propose a novel Adversarially Robust Training method for UDA accordingly, referred to as ARTUDA. Extensive experiments on multiple attacks and benchmarks show that ARTUDA consistently improves the adversarial robustness of UDA models.



## **35. Resurrecting Trust in Facial Recognition: Mitigating Backdoor Attacks in Face Recognition to Prevent Potential Privacy Breaches**

cs.CV

15 pages

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2202.10320v1)

**Authors**: Reena Zelenkova, Jack Swallow, M. A. P. Chamikara, Dongxi Liu, Mohan Baruwal Chhetri, Seyit Camtepe, Marthie Grobler, Mahathir Almashor

**Abstracts**: Biometric data, such as face images, are often associated with sensitive information (e.g medical, financial, personal government records). Hence, a data breach in a system storing such information can have devastating consequences. Deep learning is widely utilized for face recognition (FR); however, such models are vulnerable to backdoor attacks executed by malicious parties. Backdoor attacks cause a model to misclassify a particular class as a target class during recognition. This vulnerability can allow adversaries to gain access to highly sensitive data protected by biometric authentication measures or allow the malicious party to masquerade as an individual with higher system permissions. Such breaches pose a serious privacy threat. Previous methods integrate noise addition mechanisms into face recognition models to mitigate this issue and improve the robustness of classification against backdoor attacks. However, this can drastically affect model accuracy. We propose a novel and generalizable approach (named BA-BAM: Biometric Authentication - Backdoor Attack Mitigation), that aims to prevent backdoor attacks on face authentication deep learning models through transfer learning and selective image perturbation. The empirical evidence shows that BA-BAM is highly robust and incurs a maximal accuracy drop of 2.4%, while reducing the attack success rate to a maximum of 20%. Comparisons with existing approaches show that BA-BAM provides a more practical backdoor mitigation approach for face recognition.



## **36. Critical Checkpoints for Evaluating Defence Models Against Adversarial Attack and Robustness**

cs.CR

16 pages, 8 figures

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2202.09039v1)

**Authors**: Kanak Tekwani, Manojkumar Parmar

**Abstracts**: From past couple of years there is a cycle of researchers proposing a defence model for adversaries in machine learning which is arguably defensible to most of the existing attacks in restricted condition (they evaluate on some bounded inputs or datasets). And then shortly another set of researcher finding the vulnerabilities in that defence model and breaking it by proposing a stronger attack model. Some common flaws are been noticed in the past defence models that were broken in very short time. Defence models being broken so easily is a point of concern as decision of many crucial activities are taken with the help of machine learning models. So there is an utter need of some defence checkpoints that any researcher should keep in mind while evaluating the soundness of technique and declaring it to be decent defence technique. In this paper, we have suggested few checkpoints that should be taken into consideration while building and evaluating the soundness of defence models. All these points are recommended after observing why some past defence models failed and how some model remained adamant and proved their soundness against some of the very strong attacks.



## **37. Debiasing Backdoor Attack: A Benign Application of Backdoor Attack in Eliminating Data Bias**

cs.CR

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2202.10582v1)

**Authors**: Shangxi Wu, Qiuyang He, Yi Zhang, Jitao Sang

**Abstracts**: Backdoor attack is a new AI security risk that has emerged in recent years. Drawing on the previous research of adversarial attack, we argue that the backdoor attack has the potential to tap into the model learning process and improve model performance. Based on Clean Accuracy Drop (CAD) in backdoor attack, we found that CAD came out of the effect of pseudo-deletion of data. We provided a preliminary explanation of this phenomenon from the perspective of model classification boundaries and observed that this pseudo-deletion had advantages over direct deletion in the data debiasing problem. Based on the above findings, we proposed Debiasing Backdoor Attack (DBA). It achieves SOTA in the debiasing task and has a broader application scenario than undersampling.



## **38. Explaining Adversarial Vulnerability with a Data Sparsity Hypothesis**

cs.AI

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2103.00778v3)

**Authors**: Mahsa Paknezhad, Cuong Phuc Ngo, Amadeus Aristo Winarto, Alistair Cheong, Chuen Yang Beh, Jiayang Wu, Hwee Kuan Lee

**Abstracts**: Despite many proposed algorithms to provide robustness to deep learning (DL) models, DL models remain susceptible to adversarial attacks. We hypothesize that the adversarial vulnerability of DL models stems from two factors. The first factor is data sparsity which is that in the high dimensional input data space, there exist large regions outside the support of the data distribution. The second factor is the existence of many redundant parameters in the DL models. Owing to these factors, different models are able to come up with different decision boundaries with comparably high prediction accuracy. The appearance of the decision boundaries in the space outside the support of the data distribution does not affect the prediction accuracy of the model. However, it makes an important difference in the adversarial robustness of the model. We hypothesize that the ideal decision boundary is as far as possible from the support of the data distribution. In this paper, we develop a training framework to observe if DL models are able to learn such a decision boundary spanning the space around the class distributions further from the data points themselves. Semi-supervised learning was deployed during training by leveraging unlabeled data generated in the space outside the support of the data distribution. We measured adversarial robustness of the models trained using this training framework against well-known adversarial attacks and by using robustness metrics. We found that models trained using our framework, as well as other regularization methods and adversarial training support our hypothesis of data sparsity and that models trained with these methods learn to have decision boundaries more similar to the aforementioned ideal decision boundary. The code for our training framework is available at https://github.com/MahsaPaknezhad/AdversariallyRobustTraining.



## **39. Amicable examples for informed source separation**

cs.SD

Accepted to ICASSP 2022

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2110.05059v2)

**Authors**: Naoya Takahashi, Yuki Mitsufuji

**Abstracts**: This paper deals with the problem of informed source separation (ISS), where the sources are accessible during the so-called \textit{encoding} stage. Previous works computed side-information during the encoding stage and source separation models were designed to utilize the side-information to improve the separation performance. In contrast, in this work, we improve the performance of a pretrained separation model that does not use any side-information. To this end, we propose to adopt an adversarial attack for the opposite purpose, i.e., rather than computing the perturbation to degrade the separation, we compute an imperceptible perturbation called amicable noise to improve the separation. Experimental results show that the proposed approach selectively improves the performance of the targeted separation model by 2.23 dB on average and is robust to signal compression. Moreover, we propose multi-model multi-purpose learning that control the effect of the perturbation on different models individually.



## **40. Morphence: Moving Target Defense Against Adversarial Examples**

cs.LG

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2108.13952v4)

**Authors**: Abderrahmen Amich, Birhanu Eshete

**Abstracts**: Robustness to adversarial examples of machine learning models remains an open topic of research. Attacks often succeed by repeatedly probing a fixed target model with adversarial examples purposely crafted to fool it. In this paper, we introduce Morphence, an approach that shifts the defense landscape by making a model a moving target against adversarial examples. By regularly moving the decision function of a model, Morphence makes it significantly challenging for repeated or correlated attacks to succeed. Morphence deploys a pool of models generated from a base model in a manner that introduces sufficient randomness when it responds to prediction queries. To ensure repeated or correlated attacks fail, the deployed pool of models automatically expires after a query budget is reached and the model pool is seamlessly replaced by a new model pool generated in advance. We evaluate Morphence on two benchmark image classification datasets (MNIST and CIFAR10) against five reference attacks (2 white-box and 3 black-box). In all cases, Morphence consistently outperforms the thus-far effective defense, adversarial training, even in the face of strong white-box attacks, while preserving accuracy on clean data.



## **41. What Doesn't Kill You Makes You Robust(er): How to Adversarially Train against Data Poisoning**

cs.LG

25 pages, 15 figures

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2102.13624v2)

**Authors**: Jonas Geiping, Liam Fowl, Gowthami Somepalli, Micah Goldblum, Michael Moeller, Tom Goldstein

**Abstracts**: Data poisoning is a threat model in which a malicious actor tampers with training data to manipulate outcomes at inference time. A variety of defenses against this threat model have been proposed, but each suffers from at least one of the following flaws: they are easily overcome by adaptive attacks, they severely reduce testing performance, or they cannot generalize to diverse data poisoning threat models. Adversarial training, and its variants, are currently considered the only empirically strong defense against (inference-time) adversarial attacks. In this work, we extend the adversarial training framework to defend against (training-time) data poisoning, including targeted and backdoor attacks. Our method desensitizes networks to the effects of such attacks by creating poisons during training and injecting them into training batches. We show that this defense withstands adaptive attacks, generalizes to diverse threat models, and incurs a better performance trade-off than previous defenses such as DP-SGD or (evasion) adversarial training.



## **42. Developing Imperceptible Adversarial Patches to Camouflage Military Assets From Computer Vision Enabled Technologies**

cs.CV

8 pages, 4 figures, 4 tables, submitted to WCCI 2022

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2202.08892v1)

**Authors**: Christopher Wise, Jo Plested

**Abstracts**: Convolutional neural networks (CNNs) have demonstrated rapid progress and a high level of success in object detection. However, recent evidence has highlighted their vulnerability to adversarial attacks. These attacks are calculated image perturbations or adversarial patches that result in object misclassification or detection suppression. Traditional camouflage methods are impractical when applied to disguise aircraft and other large mobile assets from autonomous detection in intelligence, surveillance and reconnaissance technologies and fifth generation missiles. In this paper we present a unique method that produces imperceptible patches capable of camouflaging large military assets from computer vision-enabled technologies. We developed these patches by maximising object detection loss whilst limiting the patch's colour perceptibility. This work also aims to further the understanding of adversarial examples and their effects on object detection algorithms.



## **43. Alexa versus Alexa: Controlling Smart Speakers by Self-Issuing Voice Commands**

cs.CR

15 pages, 5 figures, published in Proceedings of the 2022 ACM Asia  Conference on Computer and Communications Security (ASIA CCS '22)

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2202.08619v1)

**Authors**: Sergio Esposito, Daniele Sgandurra, Giampaolo Bella

**Abstracts**: We present Alexa versus Alexa (AvA), a novel attack that leverages audio files containing voice commands and audio reproduction methods in an offensive fashion, to gain control of Amazon Echo devices for a prolonged amount of time. AvA leverages the fact that Alexa running on an Echo device correctly interprets voice commands originated from audio files even when they are played by the device itself -- i.e., it leverages a command self-issue vulnerability. Hence, AvA removes the necessity of having a rogue speaker in proximity of the victim's Echo, a constraint that many attacks share. With AvA, an attacker can self-issue any permissible command to Echo, controlling it on behalf of the legitimate user. We have verified that, via AvA, attackers can control smart appliances within the household, buy unwanted items, tamper linked calendars and eavesdrop on the user. We also discovered two additional Echo vulnerabilities, which we call Full Volume and Break Tag Chain. The Full Volume increases the self-issue command recognition rate, by doubling it on average, hence allowing attackers to perform additional self-issue commands. Break Tag Chain increases the time a skill can run without user interaction, from eight seconds to more than one hour, hence enabling attackers to setup realistic social engineering scenarios. By exploiting these vulnerabilities, the adversary can self-issue commands that are correctly executed 99% of the times and can keep control of the device for a prolonged amount of time. We reported these vulnerabilities to Amazon via their vulnerability research program, who rated them with a Medium severity score. Finally, to assess limitations of AvA on a larger scale, we provide the results of a survey performed on a study group of 18 users, and we show that most of the limitations against AvA are hardly used in practice.



## **44. Improving Robustness of Deep Reinforcement Learning Agents: Environment Attack based on the Critic Network**

cs.LG

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2104.03154v2)

**Authors**: Lucas Schott, Hatem Hajri, Sylvain Lamprier

**Abstracts**: To improve policy robustness of deep reinforcement learning agents, a line of recent works focus on producing disturbances of the environment. Existing approaches of the literature to generate meaningful disturbances of the environment are adversarial reinforcement learning methods. These methods set the problem as a two-player game between the protagonist agent, which learns to perform a task in an environment, and the adversary agent, which learns to disturb the protagonist via modifications of the considered environment. Both protagonist and adversary are trained with deep reinforcement learning algorithms. Alternatively, we propose in this paper to build on gradient-based adversarial attacks, usually used for classification tasks for instance, that we apply on the critic network of the protagonist to identify efficient disturbances of the environment. Rather than learning an attacker policy, which usually reveals as very complex and unstable, we leverage the knowledge of the critic network of the protagonist, to dynamically complexify the task at each step of the learning process. We show that our method, while being faster and lighter, leads to significantly better improvements in policy robustness than existing methods of the literature.



## **45. GasHis-Transformer: A Multi-scale Visual Transformer Approach for Gastric Histopathology Image Classification**

cs.CV

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2104.14528v6)

**Authors**: Haoyuan Chen, Chen Li, Ge Wang, Xiaoyan Li, Md Rahaman, Hongzan Sun, Weiming Hu, Yixin Li, Wanli Liu, Changhao Sun, Shiliang Ai, Marcin Grzegorzek

**Abstracts**: Existing deep learning methods for diagnosis of gastric cancer commonly use convolutional neural network. Recently, the Visual Transformer has attracted great attention because of its performance and efficiency, but its applications are mostly in the field of computer vision. In this paper, a multi-scale visual transformer model, referred to as GasHis-Transformer, is proposed for Gastric Histopathological Image Classification (GHIC), which enables the automatic classification of microscopic gastric images into abnormal and normal cases. The GasHis-Transformer model consists of two key modules: A global information module and a local information module to extract histopathological features effectively. In our experiments, a public hematoxylin and eosin (H&E) stained gastric histopathological dataset with 280 abnormal and normal images are divided into training, validation and test sets by a ratio of 1 : 1 : 2. The GasHis-Transformer model is applied to estimate precision, recall, F1-score and accuracy on the test set of gastric histopathological dataset as 98.0%, 100.0%, 96.0% and 98.0%, respectively. Furthermore, a critical study is conducted to evaluate the robustness of GasHis-Transformer, where ten different noises including four adversarial attack and six conventional image noises are added. In addition, a clinically meaningful study is executed to test the gastrointestinal cancer identification performance of GasHis-Transformer with 620 abnormal images and achieves 96.8% accuracy. Finally, a comparative study is performed to test the generalizability with both H&E and immunohistochemical stained images on a lymphoma image dataset and a breast cancer dataset, producing comparable F1-scores (85.6% and 82.8%) and accuracies (83.9% and 89.4%), respectively. In conclusion, GasHisTransformer demonstrates high classification performance and shows its significant potential in the GHIC task.



## **46. Measuring the Transferability of $\ell_\infty$ Attacks by the $\ell_2$ Norm**

cs.LG

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2102.10343v3)

**Authors**: Sizhe Chen, Qinghua Tao, Zhixing Ye, Xiaolin Huang

**Abstracts**: Deep neural networks could be fooled by adversarial examples with trivial differences to original samples. To keep the difference imperceptible in human eyes, researchers bound the adversarial perturbations by the $\ell_\infty$ norm, which is now commonly served as the standard to align the strength of different attacks for a fair comparison. However, we propose that using the $\ell_\infty$ norm alone is not sufficient in measuring the attack strength, because even with a fixed $\ell_\infty$ distance, the $\ell_2$ distance also greatly affects the attack transferability between models. Through the discovery, we reach more in-depth understandings towards the attack mechanism, i.e., several existing methods attack black-box models better partly because they craft perturbations with 70\% to 130\% larger $\ell_2$ distances. Since larger perturbations naturally lead to better transferability, we thereby advocate that the strength of attacks should be simultaneously measured by both the $\ell_\infty$ and $\ell_2$ norm. Our proposal is firmly supported by extensive experiments on ImageNet dataset from 7 attacks, 4 white-box models, and 9 black-box models.



## **47. Towards Evaluating the Robustness of Neural Networks Learned by Transduction**

cs.LG

Paper published at ICLR 2022. arXiv admin note: text overlap with  arXiv:2106.08387

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2110.14735v2)

**Authors**: Jiefeng Chen, Xi Wu, Yang Guo, Yingyu Liang, Somesh Jha

**Abstracts**: There has been emerging interest in using transductive learning for adversarial robustness (Goldwasser et al., NeurIPS 2020; Wu et al., ICML 2020; Wang et al., ArXiv 2021). Compared to traditional defenses, these defense mechanisms "dynamically learn" the model based on test-time input; and theoretically, attacking these defenses reduces to solving a bilevel optimization problem, which poses difficulty in crafting adaptive attacks. In this paper, we examine these defense mechanisms from a principled threat analysis perspective. We formulate and analyze threat models for transductive-learning based defenses, and point out important subtleties. We propose the principle of attacking model space for solving bilevel attack objectives, and present Greedy Model Space Attack (GMSA), an attack framework that can serve as a new baseline for evaluating transductive-learning based defenses. Through systematic evaluation, we show that GMSA, even with weak instantiations, can break previous transductive-learning based defenses, which were resilient to previous attacks, such as AutoAttack. On the positive side, we report a somewhat surprising empirical result of "transductive adversarial training": Adversarially retraining the model using fresh randomness at the test time gives a significant increase in robustness against attacks we consider.



## **48. Generalizable Information Theoretic Causal Representation**

cs.LG

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2202.08388v1)

**Authors**: Mengyue Yang, Xinyu Cai, Furui Liu, Xu Chen, Zhitang Chen, Jianye Hao, Jun Wang

**Abstracts**: It is evidence that representation learning can improve model's performance over multiple downstream tasks in many real-world scenarios, such as image classification and recommender systems. Existing learning approaches rely on establishing the correlation (or its proxy) between features and the downstream task (labels), which typically results in a representation containing cause, effect and spurious correlated variables of the label. Its generalizability may deteriorate because of the unstability of the non-causal parts. In this paper, we propose to learn causal representation from observational data by regularizing the learning procedure with mutual information measures according to our hypothetical causal graph. The optimization involves a counterfactual loss, based on which we deduce a theoretical guarantee that the causality-inspired learning is with reduced sample complexity and better generalization ability. Extensive experiments show that the models trained on causal representations learned by our approach is robust under adversarial attacks and distribution shift.



## **49. Characterizing Attacks on Deep Reinforcement Learning**

cs.LG

AAMAS 2022, 13 pages, 6 figures

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/1907.09470v3)

**Authors**: Xinlei Pan, Chaowei Xiao, Warren He, Shuang Yang, Jian Peng, Mingjie Sun, Jinfeng Yi, Zijiang Yang, Mingyan Liu, Bo Li, Dawn Song

**Abstracts**: Recent studies show that Deep Reinforcement Learning (DRL) models are vulnerable to adversarial attacks, which attack DRL models by adding small perturbations to the observations. However, some attacks assume full availability of the victim model, and some require a huge amount of computation, making them less feasible for real world applications. In this work, we make further explorations of the vulnerabilities of DRL by studying other aspects of attacks on DRL using realistic and efficient attacks. First, we adapt and propose efficient black-box attacks when we do not have access to DRL model parameters. Second, to address the high computational demands of existing attacks, we introduce efficient online sequential attacks that exploit temporal consistency across consecutive steps. Third, we explore the possibility of an attacker perturbing other aspects in the DRL setting, such as the environment dynamics. Finally, to account for imperfections in how an attacker would inject perturbations in the physical world, we devise a method for generating a robust physical perturbations to be printed. The attack is evaluated on a real-world robot under various conditions. We conduct extensive experiments both in simulation such as Atari games, robotics and autonomous driving, and on real-world robotics, to compare the effectiveness of the proposed attacks with baseline approaches. To the best of our knowledge, we are the first to apply adversarial attacks on DRL systems to physical robots.



## **50. Real-Time Neural Voice Camouflage**

cs.SD

14 pages

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2112.07076v2)

**Authors**: Mia Chiquier, Chengzhi Mao, Carl Vondrick

**Abstracts**: Automatic speech recognition systems have created exciting possibilities for applications, however they also enable opportunities for systematic eavesdropping. We propose a method to camouflage a person's voice over-the-air from these systems without inconveniencing the conversation between people in the room. Standard adversarial attacks are not effective in real-time streaming situations because the characteristics of the signal will have changed by the time the attack is executed. We introduce predictive attacks, which achieve real-time performance by forecasting the attack that will be the most effective in the future. Under real-time constraints, our method jams the established speech recognition system DeepSpeech 3.9x more than baselines as measured through word error rate, and 6.6x more as measured through character error rate. We furthermore demonstrate our approach is practically effective in realistic environments over physical distances.



