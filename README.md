# Latest Adversarial Attack Papers
**update at 2023-03-23 17:09:02**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Evaluating the Role of Target Arguments in Rumour Stance Classification**

cs.CL

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12665v1) [paper-pdf](http://arxiv.org/pdf/2303.12665v1)

**Authors**: Yue Li, Carolina Scarton

**Abstract**: Considering a conversation thread, stance classification aims to identify the opinion (e.g. agree or disagree) of replies towards a given target. The target of the stance is expected to be an essential component in this task, being one of the main factors that make it different from sentiment analysis. However, a recent study shows that a target-oblivious model outperforms target-aware models, suggesting that targets are not useful when predicting stance. This paper re-examines this phenomenon for rumour stance classification (RSC) on social media, where a target is a rumour story implied by the source tweet in the conversation. We propose adversarial attacks in the test data, aiming to assess the models robustness and evaluate the role of the data in the models performance. Results show that state-of-the-art models, including approaches that use the entire conversation thread, overly relying on superficial signals. Our hypothesis is that the naturally high occurrence of target-independent direct replies in RSC (e.g. "this is fake" or just "fake") results in the impressive performance of target-oblivious models, highlighting the risk of target instances being treated as noise during training.



## **2. Reliable and Efficient Evaluation of Adversarial Robustness for Deep Hashing-Based Retrieval**

cs.CV

arXiv admin note: text overlap with arXiv:2204.10779

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12658v1) [paper-pdf](http://arxiv.org/pdf/2303.12658v1)

**Authors**: Xunguang Wang, Jiawang Bai, Xinyue Xu, Xiaomeng Li

**Abstract**: Deep hashing has been extensively applied to massive image retrieval due to its efficiency and effectiveness. Recently, several adversarial attacks have been presented to reveal the vulnerability of deep hashing models against adversarial examples. However, existing attack methods suffer from degraded performance or inefficiency because they underutilize the semantic relations between original samples or spend a lot of time learning these relations with a deep neural network. In this paper, we propose a novel Pharos-guided Attack, dubbed PgA, to evaluate the adversarial robustness of deep hashing networks reliably and efficiently. Specifically, we design pharos code to represent the semantics of the benign image, which preserves the similarity to semantically relevant samples and dissimilarity to irrelevant ones. It is proven that we can quickly calculate the pharos code via a simple math formula. Accordingly, PgA can directly conduct a reliable and efficient attack on deep hashing-based retrieval by maximizing the similarity between the hash code of the adversarial example and the pharos code. Extensive experiments on the benchmark datasets verify that the proposed algorithm outperforms the prior state-of-the-arts in both attack strength and speed.



## **3. RoBIC: A benchmark suite for assessing classifiers robustness**

cs.CV

4 pages, accepted to ICIP 2021

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2102.05368v2) [paper-pdf](http://arxiv.org/pdf/2102.05368v2)

**Authors**: Thibault Maho, Benoît Bonnet, Teddy Furon, Erwan Le Merrer

**Abstract**: Many defenses have emerged with the development of adversarial attacks. Models must be objectively evaluated accordingly. This paper systematically tackles this concern by proposing a new parameter-free benchmark we coin RoBIC. RoBIC fairly evaluates the robustness of image classifiers using a new half-distortion measure. It gauges the robustness of the network against white and black box attacks, independently of its accuracy. RoBIC is faster than the other available benchmarks. We present the significant differences in the robustness of 16 recent models as assessed by RoBIC.



## **4. CgAT: Center-Guided Adversarial Training for Deep Hashing-Based Retrieval**

cs.CV

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2204.10779v5) [paper-pdf](http://arxiv.org/pdf/2204.10779v5)

**Authors**: Xunguang Wang, Yiqun Lin, Xiaomeng Li

**Abstract**: Deep hashing has been extensively utilized in massive image retrieval because of its efficiency and effectiveness. However, deep hashing models are vulnerable to adversarial examples, making it essential to develop adversarial defense methods for image retrieval. Existing solutions achieved limited defense performance because of using weak adversarial samples for training and lacking discriminative optimization objectives to learn robust features. In this paper, we present a min-max based Center-guided Adversarial Training, namely CgAT, to improve the robustness of deep hashing networks through worst adversarial examples. Specifically, we first formulate the center code as a semantically-discriminative representative of the input image content, which preserves the semantic similarity with positive samples and dissimilarity with negative examples. We prove that a mathematical formula can calculate the center code immediately. After obtaining the center codes in each optimization iteration of the deep hashing network, they are adopted to guide the adversarial training process. On the one hand, CgAT generates the worst adversarial examples as augmented data by maximizing the Hamming distance between the hash codes of the adversarial examples and the center codes. On the other hand, CgAT learns to mitigate the effects of adversarial samples by minimizing the Hamming distance to the center codes. Extensive experiments on the benchmark datasets demonstrate the effectiveness of our adversarial training algorithm in defending against adversarial attacks for deep hashing-based retrieval. Compared with the current state-of-the-art defense method, we significantly improve the defense performance by an average of 18.61\%, 12.35\%, and 11.56\% on FLICKR-25K, NUS-WIDE, and MS-COCO, respectively. The code is available at https://github.com/xunguangwang/CgAT.



## **5. Membership Inference Attacks against Diffusion Models**

cs.CR

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2302.03262v2) [paper-pdf](http://arxiv.org/pdf/2302.03262v2)

**Authors**: Tomoya Matsumoto, Takayuki Miura, Naoto Yanai

**Abstract**: Diffusion models have attracted attention in recent years as innovative generative models. In this paper, we investigate whether a diffusion model is resistant to a membership inference attack, which evaluates the privacy leakage of a machine learning model. We primarily discuss the diffusion model from the standpoints of comparison with a generative adversarial network (GAN) as conventional models and hyperparameters unique to the diffusion model, i.e., time steps, sampling steps, and sampling variances. We conduct extensive experiments with DDIM as a diffusion model and DCGAN as a GAN on the CelebA and CIFAR-10 datasets in both white-box and black-box settings and then confirm if the diffusion model is comparably resistant to a membership inference attack as GAN. Next, we demonstrate that the impact of time steps is significant and intermediate steps in a noise schedule are the most vulnerable to the attack. We also found two key insights through further analysis. First, we identify that DDIM is vulnerable to the attack for small sample sizes instead of achieving a lower FID. Second, sampling steps in hyperparameters are important for resistance to the attack, whereas the impact of sampling variances is quite limited.



## **6. Do Backdoors Assist Membership Inference Attacks?**

cs.CR

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12589v1) [paper-pdf](http://arxiv.org/pdf/2303.12589v1)

**Authors**: Yumeki Goto, Nami Ashizawa, Toshiki Shibahara, Naoto Yanai

**Abstract**: When an adversary provides poison samples to a machine learning model, privacy leakage, such as membership inference attacks that infer whether a sample was included in the training of the model, becomes effective by moving the sample to an outlier. However, the attacks can be detected because inference accuracy deteriorates due to poison samples. In this paper, we discuss a \textit{backdoor-assisted membership inference attack}, a novel membership inference attack based on backdoors that return the adversary's expected output for a triggered sample. We found three crucial insights through experiments with an academic benchmark dataset. We first demonstrate that the backdoor-assisted membership inference attack is unsuccessful. Second, when we analyzed loss distributions to understand the reason for the unsuccessful results, we found that backdoors cannot separate loss distributions of training and non-training samples. In other words, backdoors cannot affect the distribution of clean samples. Third, we also show that poison and triggered samples activate neurons of different distributions. Specifically, backdoors make any clean sample an inlier, contrary to poisoning samples. As a result, we confirm that backdoors cannot assist membership inference.



## **7. Autonomous Intelligent Cyber-defense Agent (AICA) Reference Architecture. Release 2.0**

cs.CR

This is a major revision and extension of the earlier release of AICA  Reference Architecture

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/1803.10664v3) [paper-pdf](http://arxiv.org/pdf/1803.10664v3)

**Authors**: Alexander Kott, Paul Théron, Martin Drašar, Edlira Dushku, Benoît LeBlanc, Paul Losiewicz, Alessandro Guarino, Luigi Mancini, Agostino Panico, Mauno Pihelgas, Krzysztof Rzadca, Fabio De Gaspari

**Abstract**: This report - a major revision of its previous release - describes a reference architecture for intelligent software agents performing active, largely autonomous cyber-defense actions on military networks of computing and communicating devices. The report is produced by the North Atlantic Treaty Organization (NATO) Research Task Group (RTG) IST-152 "Intelligent Autonomous Agents for Cyber Defense and Resilience". In a conflict with a technically sophisticated adversary, NATO military tactical networks will operate in a heavily contested battlefield. Enemy software cyber agents - malware - will infiltrate friendly networks and attack friendly command, control, communications, computers, intelligence, surveillance, and reconnaissance and computerized weapon systems. To fight them, NATO needs artificial cyber hunters - intelligent, autonomous, mobile agents specialized in active cyber defense. With this in mind, in 2016, NATO initiated RTG IST-152. Its objective has been to help accelerate the development and transition to practice of such software agents by producing a reference architecture and technical roadmap. This report presents the concept and architecture of an Autonomous Intelligent Cyber-defense Agent (AICA). We describe the rationale of the AICA concept, explain the methodology and purpose that drive the definition of the AICA Reference Architecture, and review some of the main features and challenges of AICAs.



## **8. Sibling-Attack: Rethinking Transferable Adversarial Attacks against Face Recognition**

cs.CV

8 pages, 5 fivures, accepted by CVPR 2023 as a poster paper

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12512v1) [paper-pdf](http://arxiv.org/pdf/2303.12512v1)

**Authors**: Zexin Li, Bangjie Yin, Taiping Yao, Juefeng Guo, Shouhong Ding, Simin Chen, Cong Liu

**Abstract**: A hard challenge in developing practical face recognition (FR) attacks is due to the black-box nature of the target FR model, i.e., inaccessible gradient and parameter information to attackers. While recent research took an important step towards attacking black-box FR models through leveraging transferability, their performance is still limited, especially against online commercial FR systems that can be pessimistic (e.g., a less than 50% ASR--attack success rate on average). Motivated by this, we present Sibling-Attack, a new FR attack technique for the first time explores a novel multi-task perspective (i.e., leveraging extra information from multi-correlated tasks to boost attacking transferability). Intuitively, Sibling-Attack selects a set of tasks correlated with FR and picks the Attribute Recognition (AR) task as the task used in Sibling-Attack based on theoretical and quantitative analysis. Sibling-Attack then develops an optimization framework that fuses adversarial gradient information through (1) constraining the cross-task features to be under the same space, (2) a joint-task meta optimization framework that enhances the gradient compatibility among tasks, and (3) a cross-task gradient stabilization method which mitigates the oscillation effect during attacking. Extensive experiments demonstrate that Sibling-Attack outperforms state-of-the-art FR attack techniques by a non-trivial margin, boosting ASR by 12.61% and 55.77% on average on state-of-the-art pre-trained FR models and two well-known, widely used commercial FR systems.



## **9. SCRAMBLE-CFI: Mitigating Fault-Induced Control-Flow Attacks on OpenTitan**

cs.CR

Accepted at GLSVLS'23

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.03711v2) [paper-pdf](http://arxiv.org/pdf/2303.03711v2)

**Authors**: Pascal Nasahl, Stefan Mangard

**Abstract**: Secure elements physically exposed to adversaries are frequently targeted by fault attacks. These attacks can be utilized to hijack the control-flow of software allowing the attacker to bypass security measures, extract sensitive data, or gain full code execution. In this paper, we systematically analyze the threat vector of fault-induced control-flow manipulations on the open-source OpenTitan secure element. Our thorough analysis reveals that current countermeasures of this chip either induce large area overheads or still cannot prevent the attacker from exploiting the identified threats. In this context, we introduce SCRAMBLE-CFI, an encryption-based control-flow integrity scheme utilizing existing hardware features of OpenTitan. SCRAMBLE-CFI confines, with minimal hardware overhead, the impact of fault-induced control-flow attacks by encrypting each function with a different encryption tweak at load-time. At runtime, code only can be successfully decrypted when the correct decryption tweak is active. We open-source our hardware changes and release our LLVM toolchain automatically protecting programs. Our analysis shows that SCRAMBLE-CFI complementarily enhances security guarantees of OpenTitan with a negligible hardware overhead of less than 3.97 % and a runtime overhead of 7.02 % for the Embench-IoT benchmarks.



## **10. Revisiting DeepFool: generalization and improvement**

cs.LG

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12481v1) [paper-pdf](http://arxiv.org/pdf/2303.12481v1)

**Authors**: Alireza Abdollahpourrostam, Mahed Abroshan, Seyed-Mohsen Moosavi-Dezfooli

**Abstract**: Deep neural networks have been known to be vulnerable to adversarial examples, which are inputs that are modified slightly to fool the network into making incorrect predictions. This has led to a significant amount of research on evaluating the robustness of these networks against such perturbations. One particularly important robustness metric is the robustness to minimal l2 adversarial perturbations. However, existing methods for evaluating this robustness metric are either computationally expensive or not very accurate. In this paper, we introduce a new family of adversarial attacks that strike a balance between effectiveness and computational efficiency. Our proposed attacks are generalizations of the well-known DeepFool (DF) attack, while they remain simple to understand and implement. We demonstrate that our attacks outperform existing methods in terms of both effectiveness and computational efficiency. Our proposed attacks are also suitable for evaluating the robustness of large models and can be used to perform adversarial training (AT) to achieve state-of-the-art robustness to minimal l2 adversarial perturbations.



## **11. Distribution-restrained Softmax Loss for the Model Robustness**

cs.LG

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12363v1) [paper-pdf](http://arxiv.org/pdf/2303.12363v1)

**Authors**: Hao Wang, Chen Li, Jinzhe Jiang, Xin Zhang, Yaqian Zhao, Weifeng Gong

**Abstract**: Recently, the robustness of deep learning models has received widespread attention, and various methods for improving model robustness have been proposed, including adversarial training, model architecture modification, design of loss functions, certified defenses, and so on. However, the principle of the robustness to attacks is still not fully understood, also the related research is still not sufficient. Here, we have identified a significant factor that affects the robustness of models: the distribution characteristics of softmax values for non-real label samples. We found that the results after an attack are highly correlated with the distribution characteristics, and thus we proposed a loss function to suppress the distribution diversity of softmax. A large number of experiments have shown that our method can improve robustness without significant time consumption.



## **12. Wasserstein Adversarial Examples on Univariant Time Series Data**

cs.LG

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12357v1) [paper-pdf](http://arxiv.org/pdf/2303.12357v1)

**Authors**: Wenjie Wang, Li Xiong, Jian Lou

**Abstract**: Adversarial examples are crafted by adding indistinguishable perturbations to normal examples in order to fool a well-trained deep learning model to misclassify. In the context of computer vision, this notion of indistinguishability is typically bounded by $L_{\infty}$ or other norms. However, these norms are not appropriate for measuring indistinguishiability for time series data. In this work, we propose adversarial examples in the Wasserstein space for time series data for the first time and utilize Wasserstein distance to bound the perturbation between normal examples and adversarial examples. We introduce Wasserstein projected gradient descent (WPGD), an adversarial attack method for perturbing univariant time series data. We leverage the closed-form solution of Wasserstein distance in the 1D space to calculate the projection step of WPGD efficiently with the gradient descent method. We further propose a two-step projection so that the search of adversarial examples in the Wasserstein space is guided and constrained by Euclidean norms to yield more effective and imperceptible perturbations. We empirically evaluate the proposed attack on several time series datasets in the healthcare domain. Extensive results demonstrate that the Wasserstein attack is powerful and can successfully attack most of the target classifiers with a high attack success rate. To better study the nature of Wasserstein adversarial example, we evaluate a strong defense mechanism named Wasserstein smoothing for potential certified robustness defense. Although the defense can achieve some accuracy gain, it still has limitations in many cases and leaves space for developing a stronger certified robustness method to Wasserstein adversarial examples on univariant time series data.



## **13. Bankrupting Sybil Despite Churn**

cs.CR

41 pages, 6 figures. arXiv admin note: text overlap with  arXiv:2006.02893, arXiv:1911.06462

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2010.06834v4) [paper-pdf](http://arxiv.org/pdf/2010.06834v4)

**Authors**: Diksha Gupta, Jared Saia, Maxwell Young

**Abstract**: A Sybil attack occurs when an adversary controls multiple identifiers (IDs) in a system. Limiting the number of Sybil (bad) IDs to a minority is critical to the use of well-established tools for tolerating malicious behavior, such as Byzantine agreement and secure multiparty computation.   A popular technique for enforcing a Sybil minority is resource burning: the verifiable consumption of a network resource, such as computational power, bandwidth, or memory. Unfortunately, typical defenses based on resource burning require non-Sybil (good) IDs to consume at least as many resources as the adversary. Additionally, they have a high resource burning cost, even when the system membership is relatively stable.   Here, we present a new Sybil defense, ERGO, that guarantees (1) there is always a minority of bad IDs; and (2) when the system is under significant attack, the good IDs consume asymptotically less resources than the bad. In particular, for churn rate that can vary exponentially, the resource burning rate for good IDs under ERGO is O(\sqrt{TJ} + J), where T is the resource burning rate of the adversary, and J is the join rate of good IDs. We show this resource burning rate is asymptotically optimal for a large class of algorithms.   We empirically evaluate ERGO alongside prior Sybil defenses. Additionally, we show that ERGO can be combined with machine learning techniques for classifying Sybil IDs, while preserving its theoretical guarantees. Based on our experiments comparing ERGO with two previous Sybil defenses, ERGO improves on the amount of resource burning relative to the adversary by up to 2 orders of magnitude without machine learning, and up to 3 orders of magnitude using machine learning.



## **14. X-CANIDS: Signal-Aware Explainable Intrusion Detection System for Controller Area Network-Based In-Vehicle Network**

cs.CR

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12278v1) [paper-pdf](http://arxiv.org/pdf/2303.12278v1)

**Authors**: Seonghoon Jeong, Sangho Lee, Hwejae Lee, Huy Kang Kim

**Abstract**: Controller Area Network (CAN) is an essential networking protocol that connects multiple electronic control units (ECUs) in a vehicle. However, CAN-based in-vehicle networks (IVNs) face security risks owing to the CAN mechanisms. An adversary can sabotage a vehicle by leveraging the security risks if they can access the CAN bus. Thus, recent actions and cybersecurity regulations (e.g., UNR 155) require carmakers to implement intrusion detection systems (IDSs) in their vehicles. An IDS should detect cyberattacks and provide a forensic capability to analyze attacks. Although many IDSs have been proposed, considerations regarding their feasibility and explainability remain lacking. This study proposes X-CANIDS, which is a novel IDS for CAN-based IVNs. X-CANIDS dissects the payloads in CAN messages into human-understandable signals using a CAN database. The signals improve the intrusion detection performance compared with the use of bit representations of raw payloads. These signals also enable an understanding of which signal or ECU is under attack. X-CANIDS can detect zero-day attacks because it does not require any labeled dataset in the training phase. We confirmed the feasibility of the proposed method through a benchmark test on an automotive-grade embedded device with a GPU. The results of this work will be valuable to carmakers and researchers considering the installation of in-vehicle IDSs for their vehicles.



## **15. State-of-the-art optical-based physical adversarial attacks for deep learning computer vision systems**

cs.CV

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12249v1) [paper-pdf](http://arxiv.org/pdf/2303.12249v1)

**Authors**: Junbin Fang, You Jiang, Canjian Jiang, Zoe L. Jiang, Siu-Ming Yiu, Chuanyi Liu

**Abstract**: Adversarial attacks can mislead deep learning models to make false predictions by implanting small perturbations to the original input that are imperceptible to the human eye, which poses a huge security threat to the computer vision systems based on deep learning. Physical adversarial attacks, which is more realistic, as the perturbation is introduced to the input before it is being captured and converted to a binary image inside the vision system, when compared to digital adversarial attacks. In this paper, we focus on physical adversarial attacks and further classify them into invasive and non-invasive. Optical-based physical adversarial attack techniques (e.g. using light irradiation) belong to the non-invasive category. As the perturbations can be easily ignored by humans as the perturbations are very similar to the effects generated by a natural environment in the real world. They are highly invisibility and executable and can pose a significant or even lethal threats to real systems. This paper focuses on optical-based physical adversarial attack techniques for computer vision systems, with emphasis on the introduction and discussion of optical-based physical adversarial attack techniques.



## **16. Task-Oriented Communications for NextG: End-to-End Deep Learning and AI Security Aspects**

cs.NI

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2212.09668v2) [paper-pdf](http://arxiv.org/pdf/2212.09668v2)

**Authors**: Yalin E. Sagduyu, Sennur Ulukus, Aylin Yener

**Abstract**: Communications systems to date are primarily designed with the goal of reliable transfer of digital sequences (bits). Next generation (NextG) communication systems are beginning to explore shifting this design paradigm to reliably executing a given task such as in task-oriented communications. In this paper, wireless signal classification is considered as the task for the NextG Radio Access Network (RAN), where edge devices collect wireless signals for spectrum awareness and communicate with the NextG base station (gNodeB) that needs to identify the signal label. Edge devices may not have sufficient processing power and may not be trusted to perform the signal classification task, whereas the transfer of signals to the gNodeB may not be feasible due to stringent delay, rate, and energy restrictions. Task-oriented communications is considered by jointly training the transmitter, receiver and classifier functionalities as an encoder-decoder pair for the edge device and the gNodeB. This approach improves the accuracy compared to the separated case of signal transfer followed by classification. Adversarial machine learning poses a major security threat to the use of deep learning for task-oriented communications. A major performance loss is shown when backdoor (Trojan) and adversarial (evasion) attacks target the training and test processes of task-oriented communications.



## **17. Towards Compositional Adversarial Robustness: Generalizing Adversarial Training to Composite Semantic Perturbations**

cs.CV

CVPR 2023. The research demo is at https://hsiung.cc/CARBEN/

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2202.04235v3) [paper-pdf](http://arxiv.org/pdf/2202.04235v3)

**Authors**: Lei Hsiung, Yun-Yun Tsai, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Model robustness against adversarial examples of single perturbation type such as the $\ell_{p}$-norm has been widely studied, yet its generalization to more realistic scenarios involving multiple semantic perturbations and their composition remains largely unexplored. In this paper, we first propose a novel method for generating composite adversarial examples. Our method can find the optimal attack composition by utilizing component-wise projected gradient descent and automatic attack-order scheduling. We then propose generalized adversarial training (GAT) to extend model robustness from $\ell_{p}$-ball to composite semantic perturbations, such as the combination of Hue, Saturation, Brightness, Contrast, and Rotation. Results obtained using ImageNet and CIFAR-10 datasets indicate that GAT can be robust not only to all the tested types of a single attack, but also to any combination of such attacks. GAT also outperforms baseline $\ell_{\infty}$-norm bounded adversarial training approaches by a significant margin.



## **18. Efficient Decision-based Black-box Patch Attacks on Video Recognition**

cs.CV

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11917v1) [paper-pdf](http://arxiv.org/pdf/2303.11917v1)

**Authors**: Kaixun Jiang, Zhaoyu Chen, Tony Huang, Jiafeng Wang, Dingkang Yang, Bo Li, Yan Wang, Wenqiang Zhang

**Abstract**: Although Deep Neural Networks (DNNs) have demonstrated excellent performance, they are vulnerable to adversarial patches that introduce perceptible and localized perturbations to the input. Generating adversarial patches on images has received much attention, while adversarial patches on videos have not been well investigated. Further, decision-based attacks, where attackers only access the predicted hard labels by querying threat models, have not been well explored on video models either, even if they are practical in real-world video recognition scenes. The absence of such studies leads to a huge gap in the robustness assessment for video models. To bridge this gap, this work first explores decision-based patch attacks on video models. We analyze that the huge parameter space brought by videos and the minimal information returned by decision-based models both greatly increase the attack difficulty and query burden. To achieve a query-efficient attack, we propose a spatial-temporal differential evolution (STDE) framework. First, STDE introduces target videos as patch textures and only adds patches on keyframes that are adaptively selected by temporal difference. Second, STDE takes minimizing the patch area as the optimization objective and adopts spatialtemporal mutation and crossover to search for the global optimum without falling into the local optimum. Experiments show STDE has demonstrated state-of-the-art performance in terms of threat, efficiency and imperceptibility. Hence, STDE has the potential to be a powerful tool for evaluating the robustness of video recognition models.



## **19. The Threat of Adversarial Attacks on Machine Learning in Network Security -- A Survey**

cs.CR

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/1911.02621v3) [paper-pdf](http://arxiv.org/pdf/1911.02621v3)

**Authors**: Olakunle Ibitoye, Rana Abou-Khamis, Mohamed el Shehaby, Ashraf Matrawy, M. Omair Shafiq

**Abstract**: Machine learning models have made many decision support systems to be faster, more accurate, and more efficient. However, applications of machine learning in network security face a more disproportionate threat of active adversarial attacks compared to other domains. This is because machine learning applications in network security such as malware detection, intrusion detection, and spam filtering are by themselves adversarial in nature. In what could be considered an arm's race between attackers and defenders, adversaries constantly probe machine learning systems with inputs that are explicitly designed to bypass the system and induce a wrong prediction. In this survey, we first provide a taxonomy of machine learning techniques, tasks, and depth. We then introduce a classification of machine learning in network security applications. Next, we examine various adversarial attacks against machine learning in network security and introduce two classification approaches for adversarial attacks in network security. First, we classify adversarial attacks in network security based on a taxonomy of network security applications. Secondly, we categorize adversarial attacks in network security into a problem space vs feature space dimensional classification model. We then analyze the various defenses against adversarial attacks on machine learning-based network security applications. We conclude by introducing an adversarial risk grid map and evaluating several existing adversarial attacks against machine learning in network security using the risk grid map. We also identify where each attack classification resides within the adversarial risk grid map.



## **20. OTJR: Optimal Transport Meets Optimal Jacobian Regularization for Adversarial Robustness**

cs.CV

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11793v1) [paper-pdf](http://arxiv.org/pdf/2303.11793v1)

**Authors**: Binh M. Le, Shahroz Tariq, Simon S. Woo

**Abstract**: Deep neural networks are widely recognized as being vulnerable to adversarial perturbation. To overcome this challenge, developing a robust classifier is crucial. So far, two well-known defenses have been adopted to improve the learning of robust classifiers, namely adversarial training (AT) and Jacobian regularization. However, each approach behaves differently against adversarial perturbations. First, our work carefully analyzes and characterizes these two schools of approaches, both theoretically and empirically, to demonstrate how each approach impacts the robust learning of a classifier. Next, we propose our novel Optimal Transport with Jacobian regularization method, dubbed OTJR, jointly incorporating the input-output Jacobian regularization into the AT by leveraging the optimal transport theory. In particular, we employ the Sliced Wasserstein (SW) distance that can efficiently push the adversarial samples' representations closer to those of clean samples, regardless of the number of classes within the dataset. The SW distance provides the adversarial samples' movement directions, which are much more informative and powerful for the Jacobian regularization. Our extensive experiments demonstrate the effectiveness of our proposed method, which jointly incorporates Jacobian regularization into AT. Furthermore, we demonstrate that our proposed method consistently enhances the model's robustness with CIFAR-100 dataset under various adversarial attack settings, achieving up to 28.49% under AutoAttack.



## **21. Generative AI for Cyber Threat-Hunting in 6G-enabled IoT Networks**

cs.CR

The paper is accepted and will be published in the IEEE/ACM CCGrid  2023 Conference Proceedings

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11751v1) [paper-pdf](http://arxiv.org/pdf/2303.11751v1)

**Authors**: Mohamed Amine Ferrag, Merouane Debbah, Muna Al-Hawawreh

**Abstract**: The next generation of cellular technology, 6G, is being developed to enable a wide range of new applications and services for the Internet of Things (IoT). One of 6G's main advantages for IoT applications is its ability to support much higher data rates and bandwidth as well as to support ultra-low latency. However, with this increased connectivity will come to an increased risk of cyber threats, as attackers will be able to exploit the large network of connected devices. Generative Artificial Intelligence (AI) can be used to detect and prevent cyber attacks by continuously learning and adapting to new threats and vulnerabilities. In this paper, we discuss the use of generative AI for cyber threat-hunting (CTH) in 6G-enabled IoT networks. Then, we propose a new generative adversarial network (GAN) and Transformer-based model for CTH in 6G-enabled IoT Networks. The experimental analysis results with a new cyber security dataset demonstrate that the Transformer-based security model for CTH can detect IoT attacks with a high overall accuracy of 95%. We examine the challenges and opportunities and conclude by highlighting the potential of generative AI in enhancing the security of 6G-enabled IoT networks and call for further research to be conducted in this area.



## **22. Poisoning Attacks in Federated Edge Learning for Digital Twin 6G-enabled IoTs: An Anticipatory Study**

cs.CR

The paper is accepted and will be published in the IEEE ICC 2023  Conference Proceedings

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11745v1) [paper-pdf](http://arxiv.org/pdf/2303.11745v1)

**Authors**: Mohamed Amine Ferrag, Burak Kantarci, Lucas C. Cordeiro, Merouane Debbah, Kim-Kwang Raymond Choo

**Abstract**: Federated edge learning can be essential in supporting privacy-preserving, artificial intelligence (AI)-enabled activities in digital twin 6G-enabled Internet of Things (IoT) environments. However, we need to also consider the potential of attacks targeting the underlying AI systems (e.g., adversaries seek to corrupt data on the IoT devices during local updates or corrupt the model updates); hence, in this article, we propose an anticipatory study for poisoning attacks in federated edge learning for digital twin 6G-enabled IoT environments. Specifically, we study the influence of adversaries on the training and development of federated learning models in digital twin 6G-enabled IoT environments. We demonstrate that attackers can carry out poisoning attacks in two different learning settings, namely: centralized learning and federated learning, and successful attacks can severely reduce the model's accuracy. We comprehensively evaluate the attacks on a new cyber security dataset designed for IoT applications with three deep neural networks under the non-independent and identically distributed (Non-IID) data and the independent and identically distributed (IID) data. The poisoning attacks, on an attack classification problem, can lead to a decrease in accuracy from 94.93% to 85.98% with IID data and from 94.18% to 30.04% with Non-IID.



## **23. Manipulating Transfer Learning for Property Inference**

cs.LG

Accepted to CVPR 2023

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11643v1) [paper-pdf](http://arxiv.org/pdf/2303.11643v1)

**Authors**: Yulong Tian, Fnu Suya, Anshuman Suri, Fengyuan Xu, David Evans

**Abstract**: Transfer learning is a popular method for tuning pretrained (upstream) models for different downstream tasks using limited data and computational resources. We study how an adversary with control over an upstream model used in transfer learning can conduct property inference attacks on a victim's tuned downstream model. For example, to infer the presence of images of a specific individual in the downstream training set. We demonstrate attacks in which an adversary can manipulate the upstream model to conduct highly effective and specific property inference attacks (AUC score $> 0.9$), without incurring significant performance loss on the main task. The main idea of the manipulation is to make the upstream model generate activations (intermediate features) with different distributions for samples with and without a target property, thus enabling the adversary to distinguish easily between downstream models trained with and without training examples that have the target property. Our code is available at https://github.com/yulongt23/Transfer-Inference.



## **24. An Observer-based Switching Algorithm for Safety under Sensor Denial-of-Service Attacks**

eess.SY

Accepted at the 2023 American Control Conference (ACC)

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11640v1) [paper-pdf](http://arxiv.org/pdf/2303.11640v1)

**Authors**: Santiago Jimenez Leudo, Kunal Garg, Ricardo G. Sanfelice, Alvaro A. Cardenas

**Abstract**: The design of safe-critical control algorithms for systems under Denial-of-Service (DoS) attacks on the system output is studied in this work. We aim to address scenarios where attack-mitigation approaches are not feasible, and the system needs to maintain safety under adversarial attacks. We propose an attack-recovery strategy by designing a switching observer and characterizing bounds in the error of a state estimation scheme by specifying tolerable limits on the time length of attacks. Then, we propose a switching control algorithm that renders forward invariant a set for the observer. Thus, by satisfying the error bounds of the state estimation, we guarantee that the safe set is rendered conditionally invariant with respect to a set of initial conditions. A numerical example illustrates the efficacy of the approach.



## **25. Enhancing the Self-Universality for Transferable Targeted Attacks**

cs.CV

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2209.03716v2) [paper-pdf](http://arxiv.org/pdf/2209.03716v2)

**Authors**: Zhipeng Wei, Jingjing Chen, Zuxuan Wu, Yu-Gang Jiang

**Abstract**: In this paper, we propose a novel transfer-based targeted attack method that optimizes the adversarial perturbations without any extra training efforts for auxiliary networks on training data. Our new attack method is proposed based on the observation that highly universal adversarial perturbations tend to be more transferable for targeted attacks. Therefore, we propose to make the perturbation to be agnostic to different local regions within one image, which we called as self-universality. Instead of optimizing the perturbations on different images, optimizing on different regions to achieve self-universality can get rid of using extra data. Specifically, we introduce a feature similarity loss that encourages the learned perturbations to be universal by maximizing the feature similarity between adversarial perturbed global images and randomly cropped local regions. With the feature similarity loss, our method makes the features from adversarial perturbations to be more dominant than that of benign images, hence improving targeted transferability. We name the proposed attack method as Self-Universality (SU) attack. Extensive experiments demonstrate that SU can achieve high success rates for transfer-based targeted attacks. On ImageNet-compatible dataset, SU yields an improvement of 12\% compared with existing state-of-the-art methods. Code is available at https://github.com/zhipeng-wei/Self-Universality.



## **26. Semi-supervised Semantics-guided Adversarial Training for Trajectory Prediction**

cs.LG

11 pages, adversarial training for trajectory prediction

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2205.14230v2) [paper-pdf](http://arxiv.org/pdf/2205.14230v2)

**Authors**: Ruochen Jiao, Xiangguo Liu, Takami Sato, Qi Alfred Chen, Qi Zhu

**Abstract**: Predicting the trajectories of surrounding objects is a critical task for self-driving vehicles and many other autonomous systems. Recent works demonstrate that adversarial attacks on trajectory prediction, where small crafted perturbations are introduced to history trajectories, may significantly mislead the prediction of future trajectories and induce unsafe planning. However, few works have addressed enhancing the robustness of this important safety-critical task.In this paper, we present a novel adversarial training method for trajectory prediction. Compared with typical adversarial training on image tasks, our work is challenged by more random input with rich context and a lack of class labels. To address these challenges, we propose a method based on a semi-supervised adversarial autoencoder, which models disentangled semantic features with domain knowledge and provides additional latent labels for the adversarial training. Extensive experiments with different types of attacks demonstrate that our Semisupervised Semantics-guided Adversarial Training (SSAT) method can effectively mitigate the impact of adversarial attacks by up to 73% and outperform other popular defense methods. In addition, experiments show that our method can significantly improve the system's robust generalization to unseen patterns of attacks. We believe that such semantics-guided architecture and advancement on robust generalization is an important step for developing robust prediction models and enabling safe decision-making.



## **27. STDLens: Model Hijacking-resilient Federated Learning for Object Detection**

cs.CR

CVPR 2023. Source Code: https://github.com/git-disl/STDLens

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11511v1) [paper-pdf](http://arxiv.org/pdf/2303.11511v1)

**Authors**: Ka-Ho Chow, Ling Liu, Wenqi Wei, Fatih Ilhan, Yanzhao Wu

**Abstract**: Federated Learning (FL) has been gaining popularity as a collaborative learning framework to train deep learning-based object detection models over a distributed population of clients. Despite its advantages, FL is vulnerable to model hijacking. The attacker can control how the object detection system should misbehave by implanting Trojaned gradients using only a small number of compromised clients in the collaborative learning process. This paper introduces STDLens, a principled approach to safeguarding FL against such attacks. We first investigate existing mitigation mechanisms and analyze their failures caused by the inherent errors in spatial clustering analysis on gradients. Based on the insights, we introduce a three-tier forensic framework to identify and expel Trojaned gradients and reclaim the performance over the course of FL. We consider three types of adaptive attacks and demonstrate the robustness of STDLens against advanced adversaries. Extensive experiments show that STDLens can protect FL against different model hijacking attacks and outperform existing methods in identifying and removing Trojaned gradients with significantly higher precision and much lower false-positive rates.



## **28. Deep Composite Face Image Attacks: Generation, Vulnerability and Detection**

cs.CV

The submitted paper is accepted in IEEE Access 2023

**SubmitDate**: 2023-03-20    [abs](http://arxiv.org/abs/2211.11039v3) [paper-pdf](http://arxiv.org/pdf/2211.11039v3)

**Authors**: Jag Mohan Singh, Raghavendra Ramachandra

**Abstract**: Face manipulation attacks have drawn the attention of biometric researchers because of their vulnerability to Face Recognition Systems (FRS). This paper proposes a novel scheme to generate Composite Face Image Attacks (CFIA) based on facial attributes using Generative Adversarial Networks (GANs). Given the face images corresponding to two unique data subjects, the proposed CFIA method will independently generate the segmented facial attributes, then blend them using transparent masks to generate the CFIA samples. We generate $526$ unique CFIA combinations of facial attributes for each pair of contributory data subjects. Extensive experiments are carried out on our newly generated CFIA dataset consisting of 1000 unique identities with 2000 bona fide samples and 526000 CFIA samples, thus resulting in an overall 528000 face image samples. {{We present a sequence of experiments to benchmark the attack potential of CFIA samples using four different automatic FRS}}. We introduced a new metric named Generalized Morphing Attack Potential (G-MAP) to benchmark the vulnerability of generated attacks on FRS effectively. Additional experiments are performed on the representative subset of the CFIA dataset to benchmark both perceptual quality and human observer response. Finally, the CFIA detection performance is benchmarked using three different single image based face Morphing Attack Detection (MAD) algorithms. The source code of the proposed method together with CFIA dataset will be made publicly available: \url{https://github.com/jagmohaniiit/LatentCompositionCode}



## **29. GNN-Ensemble: Towards Random Decision Graph Neural Networks**

cs.LG

**SubmitDate**: 2023-03-20    [abs](http://arxiv.org/abs/2303.11376v1) [paper-pdf](http://arxiv.org/pdf/2303.11376v1)

**Authors**: Wenqi Wei, Mu Qiao, Divyesh Jadav

**Abstract**: Graph Neural Networks (GNNs) have enjoyed wide spread applications in graph-structured data. However, existing graph based applications commonly lack annotated data. GNNs are required to learn latent patterns from a limited amount of training data to perform inferences on a vast amount of test data. The increased complexity of GNNs, as well as a single point of model parameter initialization, usually lead to overfitting and sub-optimal performance. In addition, it is known that GNNs are vulnerable to adversarial attacks. In this paper, we push one step forward on the ensemble learning of GNNs with improved accuracy, generalization, and adversarial robustness. Following the principles of stochastic modeling, we propose a new method called GNN-Ensemble to construct an ensemble of random decision graph neural networks whose capacity can be arbitrarily expanded for improvement in performance. The essence of the method is to build multiple GNNs in randomly selected substructures in the topological space and subfeatures in the feature space, and then combine them for final decision making. These GNNs in different substructure and subfeature spaces generalize their classification in complementary ways. Consequently, their combined classification performance can be improved and overfitting on the training data can be effectively reduced. In the meantime, we show that GNN-Ensemble can significantly improve the adversarial robustness against attacks on GNNs.



## **30. Adversarial Attacks against Binary Similarity Systems**

cs.CR

**SubmitDate**: 2023-03-20    [abs](http://arxiv.org/abs/2303.11143v1) [paper-pdf](http://arxiv.org/pdf/2303.11143v1)

**Authors**: Gianluca Capozzi, Daniele Cono D'Elia, Giuseppe Antonio Di Luna, Leonardo Querzoni

**Abstract**: In recent years, binary analysis gained traction as a fundamental approach to inspect software and guarantee its security. Due to the exponential increase of devices running software, much research is now moving towards new autonomous solutions based on deep learning models, as they have been showing state-of-the-art performances in solving binary analysis problems. One of the hot topics in this context is binary similarity, which consists in determining if two functions in assembly code are compiled from the same source code. However, it is unclear how deep learning models for binary similarity behave in an adversarial context. In this paper, we study the resilience of binary similarity models against adversarial examples, showing that they are susceptible to both targeted and untargeted attacks (w.r.t. similarity goals) performed by black-box and white-box attackers. In more detail, we extensively test three current state-of-the-art solutions for binary similarity against two black-box greedy attacks, including a new technique that we call Spatial Greedy, and one white-box attack in which we repurpose a gradient-guided strategy used in attacks to image classifiers.



## **31. Who Is the Strongest Enemy? Towards Optimal and Efficient Evasion Attacks in Deep RL**

cs.LG

In the 10th International Conference on Learning Representations  (ICLR 2022)

**SubmitDate**: 2023-03-20    [abs](http://arxiv.org/abs/2106.05087v5) [paper-pdf](http://arxiv.org/pdf/2106.05087v5)

**Authors**: Yanchao Sun, Ruijie Zheng, Yongyuan Liang, Furong Huang

**Abstract**: Evaluating the worst-case performance of a reinforcement learning (RL) agent under the strongest/optimal adversarial perturbations on state observations (within some constraints) is crucial for understanding the robustness of RL agents. However, finding the optimal adversary is challenging, in terms of both whether we can find the optimal attack and how efficiently we can find it. Existing works on adversarial RL either use heuristics-based methods that may not find the strongest adversary, or directly train an RL-based adversary by treating the agent as a part of the environment, which can find the optimal adversary but may become intractable in a large state space. This paper introduces a novel attacking method to find the optimal attacks through collaboration between a designed function named "actor" and an RL-based learner named "director". The actor crafts state perturbations for a given policy perturbation direction, and the director learns to propose the best policy perturbation directions. Our proposed algorithm, PA-AD, is theoretically optimal and significantly more efficient than prior RL-based works in environments with large state spaces. Empirical results show that our proposed PA-AD universally outperforms state-of-the-art attacking methods in various Atari and MuJoCo environments. By applying PA-AD to adversarial training, we achieve state-of-the-art empirical robustness in multiple tasks under strong adversaries. The codebase is released at https://github.com/umd-huang-lab/paad_adv_rl.



## **32. Translate your gibberish: black-box adversarial attack on machine translation systems**

cs.CL

**SubmitDate**: 2023-03-20    [abs](http://arxiv.org/abs/2303.10974v1) [paper-pdf](http://arxiv.org/pdf/2303.10974v1)

**Authors**: Andrei Chertkov, Olga Tsymboi, Mikhail Pautov, Ivan Oseledets

**Abstract**: Neural networks are deployed widely in natural language processing tasks on the industrial scale, and perhaps the most often they are used as compounds of automatic machine translation systems. In this work, we present a simple approach to fool state-of-the-art machine translation tools in the task of translation from Russian to English and vice versa. Using a novel black-box gradient-free tensor-based optimizer, we show that many online translation tools, such as Google, DeepL, and Yandex, may both produce wrong or offensive translations for nonsensical adversarial input queries and refuse to translate seemingly benign input phrases. This vulnerability may interfere with understanding a new language and simply worsen the user's experience while using machine translation systems, and, hence, additional improvements of these tools are required to establish better translation.



## **33. Revisiting Realistic Test-Time Training: Sequential Inference and Adaptation by Anchored Clustering Regularized Self-Training**

cs.LG

Test-time training, Self-training. arXiv admin note: substantial text  overlap with arXiv:2206.02721

**SubmitDate**: 2023-03-20    [abs](http://arxiv.org/abs/2303.10856v1) [paper-pdf](http://arxiv.org/pdf/2303.10856v1)

**Authors**: Yongyi Su, Xun Xu, Tianrui Li, Kui Jia

**Abstract**: Deploying models on target domain data subject to distribution shift requires adaptation. Test-time training (TTT) emerges as a solution to this adaptation under a realistic scenario where access to full source domain data is not available, and instant inference on the target domain is required. Despite many efforts into TTT, there is a confusion over the experimental settings, thus leading to unfair comparisons. In this work, we first revisit TTT assumptions and categorize TTT protocols by two key factors. Among the multiple protocols, we adopt a realistic sequential test-time training (sTTT) protocol, under which we develop a test-time anchored clustering (TTAC) approach to enable stronger test-time feature learning. TTAC discovers clusters in both source and target domains and matches the target clusters to the source ones to improve adaptation. When source domain information is strictly absent (i.e. source-free) we further develop an efficient method to infer source domain distributions for anchored clustering. Finally, self-training~(ST) has demonstrated great success in learning from unlabeled data and we empirically figure out that applying ST alone to TTT is prone to confirmation bias. Therefore, a more effective TTT approach is introduced by regularizing self-training with anchored clustering, and the improved model is referred to as TTAC++. We demonstrate that, under all TTT protocols, TTAC++ consistently outperforms the state-of-the-art methods on five TTT datasets, including corrupted target domain, selected hard samples, synthetic-to-real adaptation and adversarially attacked target domain. We hope this work will provide a fair benchmarking of TTT methods, and future research should be compared within respective protocols.



## **34. Defending Adversarial Attacks on Deep Learning Based Power Allocation in Massive MIMO Using Denoising Autoencoders**

eess.SP

This work has been published in the IEEE Transactions on Cognitive  Communications and Networking

**SubmitDate**: 2023-03-20    [abs](http://arxiv.org/abs/2211.15365v3) [paper-pdf](http://arxiv.org/pdf/2211.15365v3)

**Authors**: Rajeev Sahay, Minjun Zhang, David J. Love, Christopher G. Brinton

**Abstract**: Recent work has advocated for the use of deep learning to perform power allocation in the downlink of massive MIMO (maMIMO) networks. Yet, such deep learning models are vulnerable to adversarial attacks. In the context of maMIMO power allocation, adversarial attacks refer to the injection of subtle perturbations into the deep learning model's input, during inference (i.e., the adversarial perturbation is injected into inputs during deployment after the model has been trained) that are specifically crafted to force the trained regression model to output an infeasible power allocation solution. In this work, we develop an autoencoder-based mitigation technique, which allows deep learning-based power allocation models to operate in the presence of adversaries without requiring retraining. Specifically, we develop a denoising autoencoder (DAE), which learns a mapping between potentially perturbed data and its corresponding unperturbed input. We test our defense across multiple attacks and in multiple threat models and demonstrate its ability to (i) mitigate the effects of adversarial attacks on power allocation networks using two common precoding schemes, (ii) outperform previously proposed benchmarks for mitigating regression-based adversarial attacks on maMIMO networks, (iii) retain accurate performance in the absence of an attack, and (iv) operate with low computational overhead.



## **35. k-SALSA: k-anonymous synthetic averaging of retinal images via local style alignment**

cs.CV

European Conference on Computer Vision (ECCV), 2022

**SubmitDate**: 2023-03-20    [abs](http://arxiv.org/abs/2303.10824v1) [paper-pdf](http://arxiv.org/pdf/2303.10824v1)

**Authors**: Minkyu Jeon, Hyeonjin Park, Hyunwoo J. Kim, Michael Morley, Hyunghoon Cho

**Abstract**: The application of modern machine learning to retinal image analyses offers valuable insights into a broad range of human health conditions beyond ophthalmic diseases. Additionally, data sharing is key to fully realizing the potential of machine learning models by providing a rich and diverse collection of training data. However, the personally-identifying nature of retinal images, encompassing the unique vascular structure of each individual, often prevents this data from being shared openly. While prior works have explored image de-identification strategies based on synthetic averaging of images in other domains (e.g. facial images), existing techniques face difficulty in preserving both privacy and clinical utility in retinal images, as we demonstrate in our work. We therefore introduce k-SALSA, a generative adversarial network (GAN)-based framework for synthesizing retinal fundus images that summarize a given private dataset while satisfying the privacy notion of k-anonymity. k-SALSA brings together state-of-the-art techniques for training and inverting GANs to achieve practical performance on retinal images. Furthermore, k-SALSA leverages a new technique, called local style alignment, to generate a synthetic average that maximizes the retention of fine-grain visual patterns in the source images, thus improving the clinical utility of the generated images. On two benchmark datasets of diabetic retinopathy (EyePACS and APTOS), we demonstrate our improvement upon existing methods with respect to image fidelity, classification performance, and mitigation of membership inference attacks. Our work represents a step toward broader sharing of retinal images for scientific collaboration. Code is available at https://github.com/hcholab/k-salsa.



## **36. CBA: Contextual Background Attack against Optical Aerial Detection in the Physical World**

cs.CV

**SubmitDate**: 2023-03-20    [abs](http://arxiv.org/abs/2302.13519v2) [paper-pdf](http://arxiv.org/pdf/2302.13519v2)

**Authors**: Jiawei Lian, Xiaofei Wang, Yuru Su, Mingyang Ma, Shaohui Mei

**Abstract**: Patch-based physical attacks have increasingly aroused concerns.   However, most existing methods focus on obscuring targets captured on the ground, and some of these methods are simply extended to deceive aerial detectors.   They smear the targeted objects in the physical world with the elaborated adversarial patches, which can only slightly sway the aerial detectors' prediction and with weak attack transferability.   To address the above issues, we propose to perform Contextual Background Attack (CBA), a novel physical attack framework against aerial detection, which can achieve strong attack efficacy and transferability in the physical world even without smudging the interested objects at all.   Specifically, the targets of interest, i.e. the aircraft in aerial images, are adopted to mask adversarial patches.   The pixels outside the mask area are optimized to make the generated adversarial patches closely cover the critical contextual background area for detection, which contributes to gifting adversarial patches with more robust and transferable attack potency in the real world.   To further strengthen the attack performance, the adversarial patches are forced to be outside targets during training, by which the detected objects of interest, both on and outside patches, benefit the accumulation of attack efficacy.   Consequently, the sophisticatedly designed patches are gifted with solid fooling efficacy against objects both on and outside the adversarial patches simultaneously.   Extensive proportionally scaled experiments are performed in physical scenarios, demonstrating the superiority and potential of the proposed framework for physical attacks.   We expect that the proposed physical attack method will serve as a benchmark for assessing the adversarial robustness of diverse aerial detectors and defense methods.



## **37. Randomized Adversarial Training via Taylor Expansion**

cs.LG

CVPR 2023

**SubmitDate**: 2023-03-19    [abs](http://arxiv.org/abs/2303.10653v1) [paper-pdf](http://arxiv.org/pdf/2303.10653v1)

**Authors**: Gaojie Jin, Xinping Yi, Dengyu Wu, Ronghui Mu, Xiaowei Huang

**Abstract**: In recent years, there has been an explosion of research into developing more robust deep neural networks against adversarial examples. Adversarial training appears as one of the most successful methods. To deal with both the robustness against adversarial examples and the accuracy over clean examples, many works develop enhanced adversarial training methods to achieve various trade-offs between them. Leveraging over the studies that smoothed update on weights during training may help find flat minima and improve generalization, we suggest reconciling the robustness-accuracy trade-off from another perspective, i.e., by adding random noise into deterministic weights. The randomized weights enable our design of a novel adversarial training method via Taylor expansion of a small Gaussian noise, and we show that the new adversarial training method can flatten loss landscape and find flat minima. With PGD, CW, and Auto Attacks, an extensive set of experiments demonstrate that our method enhances the state-of-the-art adversarial training methods, boosting both robustness and clean accuracy. The code is available at https://github.com/Alexkael/Randomized-Adversarial-Training.



## **38. AdaptGuard: Defending Against Universal Attacks for Model Adaptation**

cs.CR

15 pages, 4 figures

**SubmitDate**: 2023-03-19    [abs](http://arxiv.org/abs/2303.10594v1) [paper-pdf](http://arxiv.org/pdf/2303.10594v1)

**Authors**: Lijun Sheng, Jian Liang, Ran He, Zilei Wang, Tieniu Tan

**Abstract**: Model adaptation aims at solving the domain transfer problem under the constraint of only accessing the pretrained source models. With the increasing considerations of data privacy and transmission efficiency, this paradigm has been gaining recent popularity. This paper studies the vulnerability to universal attacks transferred from the source domain during model adaptation algorithms due to the existence of the malicious providers. We explore both universal adversarial perturbations and backdoor attacks as loopholes on the source side and discover that they still survive in the target models after adaptation. To address this issue, we propose a model preprocessing framework, named AdaptGuard, to improve the security of model adaptation algorithms. AdaptGuard avoids direct use of the risky source parameters through knowledge distillation and utilizes the pseudo adversarial samples under adjusted radius to enhance the robustness. AdaptGuard is a plug-and-play module that requires neither robust pretrained models nor any changes for the following model adaptation algorithms. Extensive results on three commonly used datasets and two popular adaptation methods validate that AdaptGuard can effectively defend against universal attacks and maintain clean accuracy in the target domain simultaneously. We hope this research will shed light on the safety and robustness of transfer learning.



## **39. NoisyHate: Benchmarking Content Moderation Machine Learning Models with Human-Written Perturbations Online**

cs.LG

**SubmitDate**: 2023-03-18    [abs](http://arxiv.org/abs/2303.10430v1) [paper-pdf](http://arxiv.org/pdf/2303.10430v1)

**Authors**: Yiran Ye, Thai Le, Dongwon Lee

**Abstract**: Online texts with toxic content are a threat in social media that might cause cyber harassment. Although many platforms applied measures, such as machine learning-based hate-speech detection systems, to diminish their effect, those toxic content publishers can still evade the system by modifying the spelling of toxic words. Those modified words are also known as human-written text perturbations. Many research works developed certain techniques to generate adversarial samples to help the machine learning models obtain the ability to recognize those perturbations. However, there is still a gap between those machine-generated perturbations and human-written perturbations. In this paper, we introduce a benchmark test set containing human-written perturbations online for toxic speech detection models. We also recruited a group of workers to evaluate the quality of this test set and dropped low-quality samples. Meanwhile, to check if our perturbation can be normalized to its clean version, we applied spell corrector algorithms on this dataset. Finally, we test this data on state-of-the-art language models, such as BERT and RoBERTa, and black box APIs, such as perspective API, to demonstrate the adversarial attack with real human-written perturbations is still effective.



## **40. FedRight: An Effective Model Copyright Protection for Federated Learning**

cs.CR

**SubmitDate**: 2023-03-18    [abs](http://arxiv.org/abs/2303.10399v1) [paper-pdf](http://arxiv.org/pdf/2303.10399v1)

**Authors**: Jinyin Chen, Mingjun Li, Mingjun Li, Haibin Zheng

**Abstract**: Federated learning (FL), an effective distributed machine learning framework, implements model training and meanwhile protects local data privacy. It has been applied to a broad variety of practice areas due to its great performance and appreciable profits. Who owns the model, and how to protect the copyright has become a real problem. Intuitively, the existing property rights protection methods in centralized scenarios (e.g., watermark embedding and model fingerprints) are possible solutions for FL. But they are still challenged by the distributed nature of FL in aspects of the no data sharing, parameter aggregation, and federated training settings. For the first time, we formalize the problem of copyright protection for FL, and propose FedRight to protect model copyright based on model fingerprints, i.e., extracting model features by generating adversarial examples as model fingerprints. FedRight outperforms previous works in four key aspects: (i) Validity: it extracts model features to generate transferable fingerprints to train a detector to verify the copyright of the model. (ii) Fidelity: it is with imperceptible impact on the federated training, thus promising good main task performance. (iii) Robustness: it is empirically robust against malicious attacks on copyright protection, i.e., fine-tuning, model pruning, and adaptive attacks. (iv) Black-box: it is valid in the black-box forensic scenario where only application programming interface calls to the model are available. Extensive evaluations across 3 datasets and 9 model structures demonstrate FedRight's superior fidelity, validity, and robustness.



## **41. Practical Cross-System Shilling Attacks with Limited Access to Data**

cs.IR

AAAI 2023

**SubmitDate**: 2023-03-18    [abs](http://arxiv.org/abs/2302.07145v2) [paper-pdf](http://arxiv.org/pdf/2302.07145v2)

**Authors**: Meifang Zeng, Ke Li, Bingchuan Jiang, Liujuan Cao, Hui Li

**Abstract**: In shilling attacks, an adversarial party injects a few fake user profiles into a Recommender System (RS) so that the target item can be promoted or demoted. Although much effort has been devoted to developing shilling attack methods, we find that existing approaches are still far from practical. In this paper, we analyze the properties a practical shilling attack method should have and propose a new concept of Cross-system Attack. With the idea of Cross-system Attack, we design a Practical Cross-system Shilling Attack (PC-Attack) framework that requires little information about the victim RS model and the target RS data for conducting attacks. PC-Attack is trained to capture graph topology knowledge from public RS data in a self-supervised manner. Then, it is fine-tuned on a small portion of target data that is easy to access to construct fake profiles. Extensive experiments have demonstrated the superiority of PC-Attack over state-of-the-art baselines. Our implementation of PC-Attack is available at https://github.com/KDEGroup/PC-Attack.



## **42. Detection of Uncertainty in Exceedance of Threshold (DUET): An Adversarial Patch Localizer**

cs.CV

This paper has won the Best Paper Award in IEEE/ACM International  Conference on Big Data Computing, Applications and Technologies (BDCAT) 2022

**SubmitDate**: 2023-03-18    [abs](http://arxiv.org/abs/2303.10291v1) [paper-pdf](http://arxiv.org/pdf/2303.10291v1)

**Authors**: Terence Jie Chua, Wenhan Yu, Jun Zhao

**Abstract**: Development of defenses against physical world attacks such as adversarial patches is gaining traction within the research community. We contribute to the field of adversarial patch detection by introducing an uncertainty-based adversarial patch localizer which localizes adversarial patch on an image, permitting post-processing patch-avoidance or patch-reconstruction. We quantify our prediction uncertainties with the development of \textit{\textbf{D}etection of \textbf{U}ncertainties in the \textbf{E}xceedance of \textbf{T}hreshold} (DUET) algorithm. This algorithm provides a framework to ascertain confidence in the adversarial patch localization, which is essential for safety-sensitive applications such as self-driving cars and medical imaging. We conducted experiments on localizing adversarial patches and found our proposed DUET model outperforms baseline models. We then conduct further analyses on our choice of model priors and the adoption of Bayesian Neural Networks in different layers within our model architecture. We found that isometric gaussian priors in Bayesian Neural Networks are suitable for patch localization tasks and the presence of Bayesian layers in the earlier neural network blocks facilitates top-end localization performance, while Bayesian layers added in the later neural network blocks contribute to better model generalization. We then propose two different well-performing models to tackle different use cases.



## **43. Robust Adversarial Attacks Detection based on Explainable Deep Reinforcement Learning For UAV Guidance and Planning**

cs.LG

13 pages, 18 figures

**SubmitDate**: 2023-03-17    [abs](http://arxiv.org/abs/2206.02670v3) [paper-pdf](http://arxiv.org/pdf/2206.02670v3)

**Authors**: Thomas Hickling, Nabil Aouf, Phillippa Spencer

**Abstract**: The dangers of adversarial attacks on Uncrewed Aerial Vehicle (UAV) agents operating in public are increasing. Adopting AI-based techniques and, more specifically, Deep Learning (DL) approaches to control and guide these UAVs can be beneficial in terms of performance but can add concerns regarding the safety of those techniques and their vulnerability against adversarial attacks. Confusion in the agent's decision-making process caused by these attacks can seriously affect the safety of the UAV. This paper proposes an innovative approach based on the explainability of DL methods to build an efficient detector that will protect these DL schemes and the UAVs adopting them from attacks. The agent adopts a Deep Reinforcement Learning (DRL) scheme for guidance and planning. The agent is trained with a Deep Deterministic Policy Gradient (DDPG) with Prioritised Experience Replay (PER) DRL scheme that utilises Artificial Potential Field (APF) to improve training times and obstacle avoidance performance. A simulated environment for UAV explainable DRL-based planning and guidance, including obstacles and adversarial attacks, is built. The adversarial attacks are generated by the Basic Iterative Method (BIM) algorithm and reduced obstacle course completion rates from 97\% to 35\%. Two adversarial attack detectors are proposed to counter this reduction. The first one is a Convolutional Neural Network Adversarial Detector (CNN-AD), which achieves accuracy in the detection of 80\%. The second detector utilises a Long Short Term Memory (LSTM) network. It achieves an accuracy of 91\% with faster computing times compared to the CNN-AD, allowing for real-time adversarial detection.



## **44. Robust Mode Connectivity-Oriented Adversarial Defense: Enhancing Neural Network Robustness Against Diversified $\ell_p$ Attacks**

cs.AI

**SubmitDate**: 2023-03-17    [abs](http://arxiv.org/abs/2303.10225v1) [paper-pdf](http://arxiv.org/pdf/2303.10225v1)

**Authors**: Ren Wang, Yuxuan Li, Sijia Liu

**Abstract**: Adversarial robustness is a key concept in measuring the ability of neural networks to defend against adversarial attacks during the inference phase. Recent studies have shown that despite the success of improving adversarial robustness against a single type of attack using robust training techniques, models are still vulnerable to diversified $\ell_p$ attacks. To achieve diversified $\ell_p$ robustness, we propose a novel robust mode connectivity (RMC)-oriented adversarial defense that contains two population-based learning phases. The first phase, RMC, is able to search the model parameter space between two pre-trained models and find a path containing points with high robustness against diversified $\ell_p$ attacks. In light of the effectiveness of RMC, we develop a second phase, RMC-based optimization, with RMC serving as the basic unit for further enhancement of neural network diversified $\ell_p$ robustness. To increase computational efficiency, we incorporate learning with a self-robust mode connectivity (SRMC) module that enables the fast proliferation of the population used for endpoints of RMC. Furthermore, we draw parallels between SRMC and the human immune system. Experimental results on various datasets and model architectures demonstrate that the proposed defense methods can achieve high diversified $\ell_p$ robustness against $\ell_\infty$, $\ell_2$, $\ell_1$, and hybrid attacks. Codes are available at \url{https://github.com/wangren09/MCGR}.



## **45. Can AI-Generated Text be Reliably Detected?**

cs.CL

**SubmitDate**: 2023-03-17    [abs](http://arxiv.org/abs/2303.11156v1) [paper-pdf](http://arxiv.org/pdf/2303.11156v1)

**Authors**: Vinu Sankar Sadasivan, Aounon Kumar, Sriram Balasubramanian, Wenxiao Wang, Soheil Feizi

**Abstract**: The rapid progress of Large Language Models (LLMs) has made them capable of performing astonishingly well on various tasks including document completion and question answering. The unregulated use of these models, however, can potentially lead to malicious consequences such as plagiarism, generating fake news, spamming, etc. Therefore, reliable detection of AI-generated text can be critical to ensure the responsible use of LLMs. Recent works attempt to tackle this problem either using certain model signatures present in the generated text outputs or by applying watermarking techniques that imprint specific patterns onto them. In this paper, both empirically and theoretically, we show that these detectors are not reliable in practical scenarios. Empirically, we show that paraphrasing attacks, where a light paraphraser is applied on top of the generative text model, can break a whole range of detectors, including the ones using the watermarking schemes as well as neural network-based detectors and zero-shot classifiers. We then provide a theoretical impossibility result indicating that for a sufficiently good language model, even the best-possible detector can only perform marginally better than a random classifier. Finally, we show that even LLMs protected by watermarking schemes can be vulnerable against spoofing attacks where adversarial humans can infer hidden watermarking signatures and add them to their generated text to be detected as text generated by the LLMs, potentially causing reputational damages to their developers. We believe these results can open an honest conversation in the community regarding the ethical and reliable use of AI-generated text.



## **46. Fuzziness-tuned: Improving the Transferability of Adversarial Examples**

cs.LG

**SubmitDate**: 2023-03-17    [abs](http://arxiv.org/abs/2303.10078v1) [paper-pdf](http://arxiv.org/pdf/2303.10078v1)

**Authors**: Xiangyuan Yang, Jie Lin, Hanlin Zhang, Xinyu Yang, Peng Zhao

**Abstract**: With the development of adversarial attacks, adversairal examples have been widely used to enhance the robustness of the training models on deep neural networks. Although considerable efforts of adversarial attacks on improving the transferability of adversarial examples have been developed, the attack success rate of the transfer-based attacks on the surrogate model is much higher than that on victim model under the low attack strength (e.g., the attack strength $\epsilon=8/255$). In this paper, we first systematically investigated this issue and found that the enormous difference of attack success rates between the surrogate model and victim model is caused by the existence of a special area (known as fuzzy domain in our paper), in which the adversarial examples in the area are classified wrongly by the surrogate model while correctly by the victim model. Then, to eliminate such enormous difference of attack success rates for improving the transferability of generated adversarial examples, a fuzziness-tuned method consisting of confidence scaling mechanism and temperature scaling mechanism is proposed to ensure the generated adversarial examples can effectively skip out of the fuzzy domain. The confidence scaling mechanism and the temperature scaling mechanism can collaboratively tune the fuzziness of the generated adversarial examples through adjusting the gradient descent weight of fuzziness and stabilizing the update direction, respectively. Specifically, the proposed fuzziness-tuned method can be effectively integrated with existing adversarial attacks to further improve the transferability of adverarial examples without changing the time complexity. Extensive experiments demonstrated that fuzziness-tuned method can effectively enhance the transferability of adversarial examples in the latest transfer-based attacks.



## **47. Adversarial Counterfactual Visual Explanations**

cs.CV

CVPR 2023 camera-ready; Main manuscript + supplementary material

**SubmitDate**: 2023-03-17    [abs](http://arxiv.org/abs/2303.09962v1) [paper-pdf](http://arxiv.org/pdf/2303.09962v1)

**Authors**: Guillaume Jeanneret, Loïc Simon, Frédéric Jurie

**Abstract**: Counterfactual explanations and adversarial attacks have a related goal: flipping output labels with minimal perturbations regardless of their characteristics. Yet, adversarial attacks cannot be used directly in a counterfactual explanation perspective, as such perturbations are perceived as noise and not as actionable and understandable image modifications. Building on the robust learning literature, this paper proposes an elegant method to turn adversarial attacks into semantically meaningful perturbations, without modifying the classifiers to explain. The proposed approach hypothesizes that Denoising Diffusion Probabilistic Models are excellent regularizers for avoiding high-frequency and out-of-distribution perturbations when generating adversarial attacks. The paper's key idea is to build attacks through a diffusion model to polish them. This allows studying the target model regardless of its robustification level. Extensive experimentation shows the advantages of our counterfactual explanation approach over current State-of-the-Art in multiple testbeds.



## **48. Learning to Unlearn: Instance-wise Unlearning for Pre-trained Classifiers**

cs.LG

Preprint

**SubmitDate**: 2023-03-17    [abs](http://arxiv.org/abs/2301.11578v2) [paper-pdf](http://arxiv.org/pdf/2301.11578v2)

**Authors**: Sungmin Cha, Sungjun Cho, Dasol Hwang, Honglak Lee, Taesup Moon, Moontae Lee

**Abstract**: Since the recent advent of regulations for data protection (e.g., the General Data Protection Regulation), there has been increasing demand in deleting information learned from sensitive data in pre-trained models without retraining from scratch. The inherent vulnerability of neural networks towards adversarial attacks and unfairness also calls for a robust method to remove or correct information in an instance-wise fashion, while retaining the predictive performance across remaining data. To this end, we define instance-wise unlearning, of which the goal is to delete information on a set of instances from a pre-trained model, by either misclassifying each instance away from its original prediction or relabeling the instance to a different label. We also propose two methods that reduce forgetting on the remaining data: 1) utilizing adversarial examples to overcome forgetting at the representation-level and 2) leveraging weight importance metrics to pinpoint network parameters guilty of propagating unwanted information. Both methods only require the pre-trained model and data instances to forget, allowing painless application to real-life settings where the entire training set is unavailable. Through extensive experimentation on various image classification benchmarks, we show that our approach effectively preserves knowledge of remaining data while unlearning given instances in both single-task and continual unlearning scenarios.



## **49. It Is All About Data: A Survey on the Effects of Data on Adversarial Robustness**

cs.LG

41 pages, 25 figures, under review

**SubmitDate**: 2023-03-17    [abs](http://arxiv.org/abs/2303.09767v1) [paper-pdf](http://arxiv.org/pdf/2303.09767v1)

**Authors**: Peiyu Xiong, Michael Tegegn, Jaskeerat Singh Sarin, Shubhraneel Pal, Julia Rubin

**Abstract**: Adversarial examples are inputs to machine learning models that an attacker has intentionally designed to confuse the model into making a mistake. Such examples pose a serious threat to the applicability of machine-learning-based systems, especially in life- and safety-critical domains. To address this problem, the area of adversarial robustness investigates mechanisms behind adversarial attacks and defenses against these attacks. This survey reviews literature that focuses on the effects of data used by a model on the model's adversarial robustness. It systematically identifies and summarizes the state-of-the-art research in this area and further discusses gaps of knowledge and promising future research directions.



## **50. Exorcising ''Wraith'': Protecting LiDAR-based Object Detector in Automated Driving System from Appearing Attacks**

cs.CR

Accepted by USENIX Sercurity 2023

**SubmitDate**: 2023-03-17    [abs](http://arxiv.org/abs/2303.09731v1) [paper-pdf](http://arxiv.org/pdf/2303.09731v1)

**Authors**: Qifan Xiao, Xudong Pan, Yifan Lu, Mi Zhang, Jiarun Dai, Min Yang

**Abstract**: Automated driving systems rely on 3D object detectors to recognize possible obstacles from LiDAR point clouds. However, recent works show the adversary can forge non-existent cars in the prediction results with a few fake points (i.e., appearing attack). By removing statistical outliers, existing defenses are however designed for specific attacks or biased by predefined heuristic rules. Towards more comprehensive mitigation, we first systematically inspect the mechanism of recent appearing attacks: Their common weaknesses are observed in crafting fake obstacles which (i) have obvious differences in the local parts compared with real obstacles and (ii) violate the physical relation between depth and point density. In this paper, we propose a novel plug-and-play defensive module which works by side of a trained LiDAR-based object detector to eliminate forged obstacles where a major proportion of local parts have low objectness, i.e., to what degree it belongs to a real object. At the core of our module is a local objectness predictor, which explicitly incorporates the depth information to model the relation between depth and point density, and predicts each local part of an obstacle with an objectness score. Extensive experiments show, our proposed defense eliminates at least 70% cars forged by three known appearing attacks in most cases, while, for the best previous defense, less than 30% forged cars are eliminated. Meanwhile, under the same circumstance, our defense incurs less overhead for AP/precision on cars compared with existing defenses. Furthermore, We validate the effectiveness of our proposed defense on simulation-based closed-loop control driving tests in the open-source system of Baidu's Apollo.



