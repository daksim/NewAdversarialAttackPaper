# Latest Adversarial Attack Papers
**update at 2023-01-28 10:40:23**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Certified Interpretability Robustness for Class Activation Mapping**

cs.LG

13 pages, 5 figures. Accepted to Machine Learning for Autonomous  Driving Workshop at NeurIPS 2020

**SubmitDate**: 2023-01-26    [abs](http://arxiv.org/abs/2301.11324v1) [paper-pdf](http://arxiv.org/pdf/2301.11324v1)

**Authors**: Alex Gu, Tsui-Wei Weng, Pin-Yu Chen, Sijia Liu, Luca Daniel

**Abstract**: Interpreting machine learning models is challenging but crucial for ensuring the safety of deep networks in autonomous driving systems. Due to the prevalence of deep learning based perception models in autonomous vehicles, accurately interpreting their predictions is crucial. While a variety of such methods have been proposed, most are shown to lack robustness. Yet, little has been done to provide certificates for interpretability robustness. Taking a step in this direction, we present CORGI, short for Certifiably prOvable Robustness Guarantees for Interpretability mapping. CORGI is an algorithm that takes in an input image and gives a certifiable lower bound for the robustness of the top k pixels of its CAM interpretability map. We show the effectiveness of CORGI via a case study on traffic sign data, certifying lower bounds on the minimum adversarial perturbation not far from (4-5x) state-of-the-art attack methods.



## **2. Hybrid Protection of Digital FIR Filters**

cs.CR

**SubmitDate**: 2023-01-26    [abs](http://arxiv.org/abs/2301.11115v1) [paper-pdf](http://arxiv.org/pdf/2301.11115v1)

**Authors**: Levent Aksoy, Quang-Linh Nguyen, Felipe Almeida, Jaan Raik, Marie-Lise Flottes, Sophie Dupuis, Samuel Pagliarini

**Abstract**: A digital Finite Impulse Response (FIR) filter is a ubiquitous block in digital signal processing applications and its behavior is determined by its coefficients. To protect filter coefficients from an adversary, efficient obfuscation techniques have been proposed, either by hiding them behind decoys or replacing them by key bits. In this article, we initially introduce a query attack that can discover the secret key of such obfuscated FIR filters, which could not be broken by existing prominent attacks. Then, we propose a first of its kind hybrid technique, including both hardware obfuscation and logic locking using a point function for the protection of parallel direct and transposed forms of digital FIR filters. Experimental results show that the hybrid protection technique can lead to FIR filters with higher security while maintaining the hardware complexity competitive or superior to those locked by prominent logic locking methods. It is also shown that the protected multiplier blocks and FIR filters are resilient to existing attacks. The results on different forms and realizations of FIR filters show that the parallel direct form FIR filter has a promising potential for a secure design.



## **3. Improving the Transferability of Adversarial Attacks on Face Recognition with Beneficial Perturbation Feature Augmentation**

cs.CV

**SubmitDate**: 2023-01-26    [abs](http://arxiv.org/abs/2210.16117v2) [paper-pdf](http://arxiv.org/pdf/2210.16117v2)

**Authors**: Fengfan Zhou, Hefei Ling, Yuxuan Shi, Jiazhong Chen, Zongyi Li, Ping Li

**Abstract**: Face recognition (FR) models can be easily fooled by adversarial examples, which are crafted by adding imperceptible perturbations on benign face images. To improve the transferability of adversarial face examples, we propose a novel attack method called Beneficial Perturbation Feature Augmentation Attack (BPFA), which reduces the overfitting of adversarial examples to surrogate FR models by constantly generating new models that have the similar effect of hard samples to craft the adversarial examples. Specifically, in the backpropagation, BPFA records the gradients on pre-selected features and uses the gradient on the input image to craft the adversarial example. In the next forward propagation, BPFA leverages the recorded gradients to add perturbations (i.e., beneficial perturbations) that can be pitted against the adversarial example on their corresponding features. The optimization process of the adversarial example and the optimization process of the beneficial perturbations added on the features correspond to a minimax two-player game. Extensive experiments demonstrate that BPFA can significantly boost the transferability of adversarial attacks on FR.



## **4. Revisiting the Adversarial Robustness-Accuracy Tradeoff in Robot Learning**

cs.RO

**SubmitDate**: 2023-01-25    [abs](http://arxiv.org/abs/2204.07373v2) [paper-pdf](http://arxiv.org/pdf/2204.07373v2)

**Authors**: Mathias Lechner, Alexander Amini, Daniela Rus, Thomas A. Henzinger

**Abstract**: Adversarial training (i.e., training on adversarially perturbed input data) is a well-studied method for making neural networks robust to potential adversarial attacks during inference. However, the improved robustness does not come for free but rather is accompanied by a decrease in overall model accuracy and performance. Recent work has shown that, in practical robot learning applications, the effects of adversarial training do not pose a fair trade-off but inflict a net loss when measured in holistic robot performance. This work revisits the robustness-accuracy trade-off in robot learning by systematically analyzing if recent advances in robust training methods and theory in conjunction with adversarial robot learning, are capable of making adversarial training suitable for real-world robot applications. We evaluate three different robot learning tasks ranging from autonomous driving in a high-fidelity environment amenable to sim-to-real deployment to mobile robot navigation and gesture recognition. Our results demonstrate that, while these techniques make incremental improvements on the trade-off on a relative scale, the negative impact on the nominal accuracy caused by adversarial training still outweighs the improved robustness by an order of magnitude. We conclude that although progress is happening, further advances in robust learning methods are necessary before they can benefit robot learning tasks in practice.



## **5. RobustPdM: Designing Robust Predictive Maintenance against Adversarial Attacks**

cs.CR

**SubmitDate**: 2023-01-25    [abs](http://arxiv.org/abs/2301.10822v1) [paper-pdf](http://arxiv.org/pdf/2301.10822v1)

**Authors**: Ayesha Siddique, Ripan Kumar Kundu, Gautam Raj Mode, Khaza Anuarul Hoque

**Abstract**: The state-of-the-art predictive maintenance (PdM) techniques have shown great success in reducing maintenance costs and downtime of complicated machines while increasing overall productivity through extensive utilization of Internet-of-Things (IoT) and Deep Learning (DL). Unfortunately, IoT sensors and DL algorithms are both prone to cyber-attacks. For instance, DL algorithms are known for their susceptibility to adversarial examples. Such adversarial attacks are vastly under-explored in the PdM domain. This is because the adversarial attacks in the computer vision domain for classification tasks cannot be directly applied to the PdM domain for multivariate time series (MTS) regression tasks. In this work, we propose an end-to-end methodology to design adversarially robust PdM systems by extensively analyzing the effect of different types of adversarial attacks and proposing a novel adversarial defense technique for DL-enabled PdM models. First, we propose novel MTS Projected Gradient Descent (PGD) and MTS PGD with random restarts (PGD_r) attacks. Then, we evaluate the impact of MTS PGD and PGD_r along with MTS Fast Gradient Sign Method (FGSM) and MTS Basic Iterative Method (BIM) on Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), Convolutional Neural Network (CNN), and Bi-directional LSTM based PdM system. Our results using NASA's turbofan engine dataset show that adversarial attacks can cause a severe defect (up to 11X) in the RUL prediction, outperforming the effectiveness of the state-of-the-art PdM attacks by 3X. Furthermore, we present a novel approximate adversarial training method to defend against adversarial attacks. We observe that approximate adversarial training can significantly improve the robustness of PdM models (up to 54X) and outperforms the state-of-the-art PdM defense methods by offering 3X more robustness.



## **6. Characterizing the Influence of Graph Elements**

cs.LG

**SubmitDate**: 2023-01-25    [abs](http://arxiv.org/abs/2210.07441v2) [paper-pdf](http://arxiv.org/pdf/2210.07441v2)

**Authors**: Zizhang Chen, Peizhao Li, Hongfu Liu, Pengyu Hong

**Abstract**: Influence function, a method from robust statistics, measures the changes of model parameters or some functions about model parameters concerning the removal or modification of training instances. It is an efficient and useful post-hoc method for studying the interpretability of machine learning models without the need for expensive model re-training. Recently, graph convolution networks (GCNs), which operate on graph data, have attracted a great deal of attention. However, there is no preceding research on the influence functions of GCNs to shed light on the effects of removing training nodes/edges from an input graph. Since the nodes/edges in a graph are interdependent in GCNs, it is challenging to derive influence functions for GCNs. To fill this gap, we started with the simple graph convolution (SGC) model that operates on an attributed graph and formulated an influence function to approximate the changes in model parameters when a node or an edge is removed from an attributed graph. Moreover, we theoretically analyzed the error bound of the estimated influence of removing an edge. We experimentally validated the accuracy and effectiveness of our influence estimation function. In addition, we showed that the influence function of an SGC model could be used to estimate the impact of removing training nodes/edges on the test performance of the SGC without re-training the model. Finally, we demonstrated how to use influence functions to guide the adversarial attacks on GCNs effectively.



## **7. Extending Adversarial Attacks to Produce Adversarial Class Probability Distributions**

cs.LG

Final version as accepted in JMLR. Attribution requirements are  provided at http://jmlr.org/papers/v24/21-0326.html

**SubmitDate**: 2023-01-25    [abs](http://arxiv.org/abs/2004.06383v3) [paper-pdf](http://arxiv.org/pdf/2004.06383v3)

**Authors**: Jon Vadillo, Roberto Santana, Jose A. Lozano

**Abstract**: Despite the remarkable performance and generalization levels of deep learning models in a wide range of artificial intelligence tasks, it has been demonstrated that these models can be easily fooled by the addition of imperceptible yet malicious perturbations to natural inputs. These altered inputs are known in the literature as adversarial examples. In this paper, we propose a novel probabilistic framework to generalize and extend adversarial attacks in order to produce a desired probability distribution for the classes when we apply the attack method to a large number of inputs. This novel attack paradigm provides the adversary with greater control over the target model, thereby exposing, in a wide range of scenarios, threats against deep learning models that cannot be conducted by the conventional paradigms. We introduce four different strategies to efficiently generate such attacks, and illustrate our approach by extending multiple adversarial attack algorithms. We also experimentally validate our approach for the spoken command classification task and the Tweet emotion classification task, two exemplary machine learning problems in the audio and text domain, respectively. Our results demonstrate that we can closely approximate any probability distribution for the classes while maintaining a high fooling rate and even prevent the attacks from being detected by label-shift detection methods.



## **8. On the Adversarial Robustness of Camera-based 3D Object Detection**

cs.CV

**SubmitDate**: 2023-01-25    [abs](http://arxiv.org/abs/2301.10766v1) [paper-pdf](http://arxiv.org/pdf/2301.10766v1)

**Authors**: Shaoyuan Xie, Zichao Li, Zeyu Wang, Cihang Xie

**Abstract**: In recent years, camera-based 3D object detection has gained widespread attention for its ability to achieve high performance with low computational cost. However, the robustness of these methods to adversarial attacks has not been thoroughly examined. In this study, we conduct the first comprehensive investigation of the robustness of leading camera-based 3D object detection methods under various adversarial conditions. Our experiments reveal five interesting findings: (a) the use of accurate depth estimation effectively improves robustness; (b) depth-estimation-free approaches do not show superior robustness; (c) bird's-eye-view-based representations exhibit greater robustness against localization attacks; (d) incorporating multi-frame benign inputs can effectively mitigate adversarial attacks; and (e) addressing long-tail problems can enhance robustness. We hope our work can provide guidance for the design of future camera-based object detection modules with improved adversarial robustness.



## **9. A Study on FGSM Adversarial Training for Neural Retrieval**

cs.IR

Accepted at ECIR 2023

**SubmitDate**: 2023-01-25    [abs](http://arxiv.org/abs/2301.10576v1) [paper-pdf](http://arxiv.org/pdf/2301.10576v1)

**Authors**: Simon Lupart, Stéphane Clinchant

**Abstract**: Neural retrieval models have acquired significant effectiveness gains over the last few years compared to term-based methods. Nevertheless, those models may be brittle when faced to typos, distribution shifts or vulnerable to malicious attacks. For instance, several recent papers demonstrated that such variations severely impacted models performances, and then tried to train more resilient models. Usual approaches include synonyms replacements or typos injections -- as data-augmentation -- and the use of more robust tokenizers (characterBERT, BPE-dropout). To further complement the literature, we investigate in this paper adversarial training as another possible solution to this robustness issue. Our comparison includes the two main families of BERT-based neural retrievers, i.e. dense and sparse, with and without distillation techniques. We then demonstrate that one of the most simple adversarial training techniques -- the Fast Gradient Sign Method (FGSM) -- can improve first stage rankers robustness and effectiveness. In particular, FGSM increases models performances on both in-domain and out-of-domain distributions, and also on queries with typos, for multiple neural retrievers.



## **10. A Data-Centric Approach for Improving Adversarial Training Through the Lens of Out-of-Distribution Detection**

cs.LG

Accepted to CSICC 2023

**SubmitDate**: 2023-01-25    [abs](http://arxiv.org/abs/2301.10454v1) [paper-pdf](http://arxiv.org/pdf/2301.10454v1)

**Authors**: Mohammad Azizmalayeri, Arman Zarei, Alireza Isavand, Mohammad Taghi Manzuri, Mohammad Hossein Rohban

**Abstract**: Current machine learning models achieve super-human performance in many real-world applications. Still, they are susceptible against imperceptible adversarial perturbations. The most effective solution for this problem is adversarial training that trains the model with adversarially perturbed samples instead of original ones. Various methods have been developed over recent years to improve adversarial training such as data augmentation or modifying training attacks. In this work, we examine the same problem from a new data-centric perspective. For this purpose, we first demonstrate that the existing model-based methods can be equivalent to applying smaller perturbation or optimization weights to the hard training examples. By using this finding, we propose detecting and removing these hard samples directly from the training procedure rather than applying complicated algorithms to mitigate their effects. For detection, we use maximum softmax probability as an effective method in out-of-distribution detection since we can consider the hard samples as the out-of-distribution samples for the whole data distribution. Our results on SVHN and CIFAR-10 datasets show the effectiveness of this method in improving the adversarial training without adding too much computational cost.



## **11. BDMMT: Backdoor Sample Detection for Language Models through Model Mutation Testing**

cs.CL

**SubmitDate**: 2023-01-25    [abs](http://arxiv.org/abs/2301.10412v1) [paper-pdf](http://arxiv.org/pdf/2301.10412v1)

**Authors**: Jiali Wei, Ming Fan, Wenjing Jiao, Wuxia Jin, Ting Liu

**Abstract**: Deep neural networks (DNNs) and natural language processing (NLP) systems have developed rapidly and have been widely used in various real-world fields. However, they have been shown to be vulnerable to backdoor attacks. Specifically, the adversary injects a backdoor into the model during the training phase, so that input samples with backdoor triggers are classified as the target class. Some attacks have achieved high attack success rates on the pre-trained language models (LMs), but there have yet to be effective defense methods. In this work, we propose a defense method based on deep model mutation testing. Our main justification is that backdoor samples are much more robust than clean samples if we impose random mutations on the LMs and that backdoors are generalizable. We first confirm the effectiveness of model mutation testing in detecting backdoor samples and select the most appropriate mutation operators. We then systematically defend against three extensively studied backdoor attack levels (i.e., char-level, word-level, and sentence-level) by detecting backdoor samples. We also make the first attempt to defend against the latest style-level backdoor attacks. We evaluate our approach on three benchmark datasets (i.e., IMDB, Yelp, and AG news) and three style transfer datasets (i.e., SST-2, Hate-speech, and AG news). The extensive experimental results demonstrate that our approach can detect backdoor samples more efficiently and accurately than the three state-of-the-art defense approaches.



## **12. Dynamics-aware Adversarial Attack of Adaptive Neural Networks**

cs.CV

**SubmitDate**: 2023-01-25    [abs](http://arxiv.org/abs/2210.08159v2) [paper-pdf](http://arxiv.org/pdf/2210.08159v2)

**Authors**: An Tao, Yueqi Duan, Yingqi Wang, Jiwen Lu, Jie Zhou

**Abstract**: In this paper, we investigate the dynamics-aware adversarial attack problem of adaptive neural networks. Most existing adversarial attack algorithms are designed under a basic assumption -- the network architecture is fixed throughout the attack process. However, this assumption does not hold for many recently proposed adaptive neural networks, which adaptively deactivate unnecessary execution units based on inputs to improve computational efficiency. It results in a serious issue of lagged gradient, making the learned attack at the current step ineffective due to the architecture change afterward. To address this issue, we propose a Leaded Gradient Method (LGM) and show the significant effects of the lagged gradient. More specifically, we reformulate the gradients to be aware of the potential dynamic changes of network architectures, so that the learned attack better "leads" the next step than the dynamics-unaware methods when network architecture changes dynamically. Extensive experiments on representative types of adaptive neural networks for both 2D images and 3D point clouds show that our LGM achieves impressive adversarial attack performance compared with the dynamic-unaware attack methods.



## **13. Blockchain-aided Secure Semantic Communication for AI-Generated Content in Metaverse**

cs.CR

10 pages, 8 figures, journal

**SubmitDate**: 2023-01-25    [abs](http://arxiv.org/abs/2301.11289v1) [paper-pdf](http://arxiv.org/pdf/2301.11289v1)

**Authors**: Yijing Lin, Hongyang Du, Dusit Niyato, Jiangtian Nie, Jiayi Zhang, Yanyu Cheng, Zhaohui Yang

**Abstract**: The construction of virtual transportation networks requires massive data to be transmitted from edge devices to Virtual Service Providers (VSP) to facilitate circulations between the physical and virtual domains in Metaverse. Leveraging semantic communication for reducing information redundancy, VSPs can receive semantic data from edge devices to provide varied services through advanced techniques, e.g., AI-Generated Content (AIGC), for users to explore digital worlds. But the use of semantic communication raises a security issue because attackers could send malicious semantic data with similar semantic information but different desired content to break Metaverse services and cause wrong output of AIGC. Therefore, in this paper, we first propose a blockchain-aided semantic communication framework for AIGC services in virtual transportation networks to facilitate interactions of the physical and virtual domains among VSPs and edge devices. We illustrate a training-based targeted semantic attack scheme to generate adversarial semantic data by various loss functions. We also design a semantic defense scheme that uses the blockchain and zero-knowledge proofs to tell the difference between the semantic similarities of adversarial and authentic semantic data and to check the authenticity of semantic data transformations. Simulation results show that the proposed defense method can reduce the semantic similarity of the adversarial semantic data and the authentic ones by up to 30% compared with the attack scheme.



## **14. To Trust or Not To Trust Prediction Scores for Membership Inference Attacks**

cs.LG

15 pages, 8 figures, 10 tables

**SubmitDate**: 2023-01-24    [abs](http://arxiv.org/abs/2111.09076v3) [paper-pdf](http://arxiv.org/pdf/2111.09076v3)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Kristian Kersting

**Abstract**: Membership inference attacks (MIAs) aim to determine whether a specific sample was used to train a predictive model. Knowing this may indeed lead to a privacy breach. Most MIAs, however, make use of the model's prediction scores - the probability of each output given some input - following the intuition that the trained model tends to behave differently on its training data. We argue that this is a fallacy for many modern deep network architectures. Consequently, MIAs will miserably fail since overconfidence leads to high false-positive rates not only on known domains but also on out-of-distribution data and implicitly acts as a defense against MIAs. Specifically, using generative adversarial networks, we are able to produce a potentially infinite number of samples falsely classified as part of the training data. In other words, the threat of MIAs is overestimated, and less information is leaked than previously assumed. Moreover, there is actually a trade-off between the overconfidence of models and their susceptibility to MIAs: the more classifiers know when they do not know, making low confidence predictions, the more they reveal the training data.



## **15. Robustness through Data Augmentation Loss Consistency**

cs.LG

40 pages

**SubmitDate**: 2023-01-24    [abs](http://arxiv.org/abs/2110.11205v3) [paper-pdf](http://arxiv.org/pdf/2110.11205v3)

**Authors**: Tianjian Huang, Shaunak Halbe, Chinnadhurai Sankar, Pooyan Amini, Satwik Kottur, Alborz Geramifard, Meisam Razaviyayn, Ahmad Beirami

**Abstract**: While deep learning through empirical risk minimization (ERM) has succeeded at achieving human-level performance at a variety of complex tasks, ERM is not robust to distribution shifts or adversarial attacks. Synthetic data augmentation followed by empirical risk minimization (DA-ERM) is a simple and widely used solution to improve robustness in ERM. In addition, consistency regularization can be applied to further improve the robustness of the model by forcing the representation of the original sample and the augmented one to be similar. However, existing consistency regularization methods are not applicable to covariant data augmentation, where the label in the augmented sample is dependent on the augmentation function. For example, dialog state covaries with named entity when we augment data with a new named entity. In this paper, we propose data augmented loss invariant regularization (DAIR), a simple form of consistency regularization that is applied directly at the loss level rather than intermediate features, making it widely applicable to both invariant and covariant data augmentation regardless of network architecture, problem setup, and task. We apply DAIR to real-world learning problems involving covariant data augmentation: robust neural task-oriented dialog state tracking and robust visual question answering. We also apply DAIR to tasks involving invariant data augmentation: robust regression, robust classification against adversarial attacks, and robust ImageNet classification under distribution shift. Our experiments show that DAIR consistently outperforms ERM and DA-ERM with little marginal computational cost and sets new state-of-the-art results in several benchmarks involving covariant data augmentation. Our code of all experiments is available at: https://github.com/optimization-for-data-driven-science/DAIR.git



## **16. RAIN: RegulArization on Input and Network for Black-Box Domain Adaptation**

cs.CV

**SubmitDate**: 2023-01-24    [abs](http://arxiv.org/abs/2208.10531v2) [paper-pdf](http://arxiv.org/pdf/2208.10531v2)

**Authors**: Qucheng Peng, Zhengming Ding, Lingjuan Lyu, Lichao Sun, Chen Chen

**Abstract**: Source-Free domain adaptation transits the source-trained model towards target domain without exposing the source data, trying to dispel these concerns about data privacy and security. However, this paradigm is still at risk of data leakage due to adversarial attacks on the source model. Hence, the Black-Box setting only allows to use the outputs of source model, but still suffers from overfitting on the source domain more severely due to source model's unseen weights. In this paper, we propose a novel approach named RAIN (RegulArization on Input and Network) for Black-Box domain adaptation from both input-level and network-level regularization. For the input-level, we design a new data augmentation technique as Phase MixUp, which highlights task-relevant objects in the interpolations, thus enhancing input-level regularization and class consistency for target models. For network-level, we develop a Subnetwork Distillation mechanism to transfer knowledge from the target subnetwork to the full target network via knowledge distillation, which thus alleviates overfitting on the source domain by learning diverse target representations. Extensive experiments show that our method achieves state-of-the-art performance on several cross-domain benchmarks under both single- and multi-source black-box domain adaptation.



## **17. Robust Fair Clustering: A Novel Fairness Attack and Defense Framework**

cs.LG

Accepted to the 11th International Conference on Learning  Representations (ICLR 2023)

**SubmitDate**: 2023-01-24    [abs](http://arxiv.org/abs/2210.01953v2) [paper-pdf](http://arxiv.org/pdf/2210.01953v2)

**Authors**: Anshuman Chhabra, Peizhao Li, Prasant Mohapatra, Hongfu Liu

**Abstract**: Clustering algorithms are widely used in many societal resource allocation applications, such as loan approvals and candidate recruitment, among others, and hence, biased or unfair model outputs can adversely impact individuals that rely on these applications. To this end, many fair clustering approaches have been recently proposed to counteract this issue. Due to the potential for significant harm, it is essential to ensure that fair clustering algorithms provide consistently fair outputs even under adversarial influence. However, fair clustering algorithms have not been studied from an adversarial attack perspective. In contrast to previous research, we seek to bridge this gap and conduct a robustness analysis against fair clustering by proposing a novel black-box fairness attack. Through comprehensive experiments, we find that state-of-the-art models are highly susceptible to our attack as it can reduce their fairness performance significantly. Finally, we propose Consensus Fair Clustering (CFC), the first robust fair clustering approach that transforms consensus clustering into a fair graph partitioning problem, and iteratively learns to generate fair cluster outputs. Experimentally, we observe that CFC is highly robust to the proposed attack and is thus a truly robust fair clustering alternative.



## **18. DODEM: DOuble DEfense Mechanism Against Adversarial Attacks Towards Secure Industrial Internet of Things Analytics**

cs.CR

**SubmitDate**: 2023-01-23    [abs](http://arxiv.org/abs/2301.09740v1) [paper-pdf](http://arxiv.org/pdf/2301.09740v1)

**Authors**: Onat Gungor, Tajana Rosing, Baris Aksanli

**Abstract**: Industrial Internet of Things (I-IoT) is a collaboration of devices, sensors, and networking equipment to monitor and collect data from industrial operations. Machine learning (ML) methods use this data to make high-level decisions with minimal human intervention. Data-driven predictive maintenance (PDM) is a crucial ML-based I-IoT application to find an optimal maintenance schedule for industrial assets. The performance of these ML methods can seriously be threatened by adversarial attacks where an adversary crafts perturbed data and sends it to the ML model to deteriorate its prediction performance. The models should be able to stay robust against these attacks where robustness is measured by how much perturbation in input data affects model performance. Hence, there is a need for effective defense mechanisms that can protect these models against adversarial attacks. In this work, we propose a double defense mechanism to detect and mitigate adversarial attacks in I-IoT environments. We first detect if there is an adversarial attack on a given sample using novelty detection algorithms. Then, based on the outcome of our algorithm, marking an instance as attack or normal, we select adversarial retraining or standard training to provide a secondary defense layer. If there is an attack, adversarial retraining provides a more robust model, while we apply standard training for regular samples. Since we may not know if an attack will take place, our adaptive mechanism allows us to consider irregular changes in data. The results show that our double defense strategy is highly efficient where we can improve model robustness by up to 64.6% and 52% compared to standard and adversarial retraining, respectively.



## **19. ESWORD: Implementation of Wireless Jamming Attacks in a Real-World Emulated Network**

cs.NI

6 pages, 7 figures, 1 table. IEEE Wireless Communications and  Networking Conference (WCNC), Glasgow, Scotland, March 2023

**SubmitDate**: 2023-01-23    [abs](http://arxiv.org/abs/2301.09615v1) [paper-pdf](http://arxiv.org/pdf/2301.09615v1)

**Authors**: Clifton Paul Robinson, Leonardo Bonati, Tara Van Nieuwstadt, Teddy Reiss, Pedram Johari, Michele Polese, Hieu Nguyen, Curtis Watson, Tommaso Melodia

**Abstract**: Wireless jamming attacks have plagued wireless communication systems and will continue to do so going forward with technological advances. These attacks fall under the category of Electronic Warfare (EW), a continuously growing area in both attack and defense of the electromagnetic spectrum, with one subcategory being electronic attacks. Jamming attacks fall under this specific subcategory of EW as they comprise adversarial signals that attempt to disrupt, deny, degrade, destroy, or deceive legitimate signals in the electromagnetic spectrum. While jamming is not going away, recent research advances have started to get the upper hand against these attacks by leveraging new methods and techniques, such as machine learning. However, testing such jamming solutions on a wide and realistic scale is a daunting task due to strict regulations on spectrum emissions. In this paper, we introduce eSWORD, the first large-scale framework that allows users to safely conduct real-time and controlled jamming experiments with hardware-in-the-loop. This is done by integrating eSWORD into the Colosseum wireless network emulator that enables large-scale experiments with up to 50 software-defined radio nodes. We compare the performance of eSWORD with that of real-world jamming systems by using an over-the-air wireless testbed (ensuring safe measures were taken when conducting experiments). Our experimental results demonstrate that eSWORD follows similar patterns in throughput, signal-to-noise ratio, and link status to real-world jamming experiments, testifying to the high accuracy of the emulated eSWORD setup.



## **20. BayBFed: Bayesian Backdoor Defense for Federated Learning**

cs.LG

**SubmitDate**: 2023-01-23    [abs](http://arxiv.org/abs/2301.09508v1) [paper-pdf](http://arxiv.org/pdf/2301.09508v1)

**Authors**: Kavita Kumari, Phillip Rieger, Hossein Fereidooni, Murtuza Jadliwala, Ahmad-Reza Sadeghi

**Abstract**: Federated learning (FL) allows participants to jointly train a machine learning model without sharing their private data with others. However, FL is vulnerable to poisoning attacks such as backdoor attacks. Consequently, a variety of defenses have recently been proposed, which have primarily utilized intermediary states of the global model (i.e., logits) or distance of the local models (i.e., L2-norm) from the global model to detect malicious backdoors. However, as these approaches directly operate on client updates, their effectiveness depends on factors such as clients' data distribution or the adversary's attack strategies. In this paper, we introduce a novel and more generic backdoor defense framework, called BayBFed, which proposes to utilize probability distributions over client updates to detect malicious updates in FL: it computes a probabilistic measure over the clients' updates to keep track of any adjustments made in the updates, and uses a novel detection algorithm that can leverage this probabilistic measure to efficiently detect and filter out malicious updates. Thus, it overcomes the shortcomings of previous approaches that arise due to the direct usage of client updates; as our probabilistic measure will include all aspects of the local client training strategies. BayBFed utilizes two Bayesian Non-Parametric extensions: (i) a Hierarchical Beta-Bernoulli process to draw a probabilistic measure given the clients' updates, and (ii) an adaptation of the Chinese Restaurant Process (CRP), referred by us as CRP-Jensen, which leverages this probabilistic measure to detect and filter out malicious updates. We extensively evaluate our defense approach on five benchmark datasets: CIFAR10, Reddit, IoT intrusion detection, MNIST, and FMNIST, and show that it can effectively detect and eliminate malicious updates in FL without deteriorating the benign performance of the global model.



## **21. Practical Adversarial Attacks Against AI-Driven Power Allocation in a Distributed MIMO Network**

eess.SP

6 pages, 10 figures, accepted for presentation in International  Conference on Communications (ICC) 2023 in Communication and Information  System Security Symposium

**SubmitDate**: 2023-01-23    [abs](http://arxiv.org/abs/2301.09305v1) [paper-pdf](http://arxiv.org/pdf/2301.09305v1)

**Authors**: Ömer Faruk Tuna, Fehmi Emre Kadan, Leyli Karaçay

**Abstract**: In distributed multiple-input multiple-output (D-MIMO) networks, power control is crucial to optimize the spectral efficiencies of users and max-min fairness (MMF) power control is a commonly used strategy as it satisfies uniform quality-of-service to all users. The optimal solution of MMF power control requires high complexity operations and hence deep neural network based artificial intelligence (AI) solutions are proposed to decrease the complexity. Although quite accurate models can be achieved by using AI, these models have some intrinsic vulnerabilities against adversarial attacks where carefully crafted perturbations are applied to the input of the AI model. In this work, we show that threats against the target AI model which might be originated from malicious users or radio units can substantially decrease the network performance by applying a successful adversarial sample, even in the most constrained circumstances. We also demonstrate that the risk associated with these kinds of adversarial attacks is higher than the conventional attack threats. Detailed simulations reveal the effectiveness of adversarial attacks and the necessity of smart defense techniques.



## **22. ContraBERT: Enhancing Code Pre-trained Models via Contrastive Learning**

cs.SE

**SubmitDate**: 2023-01-22    [abs](http://arxiv.org/abs/2301.09072v1) [paper-pdf](http://arxiv.org/pdf/2301.09072v1)

**Authors**: Shangqing Liu, Bozhi Wu, Xiaofei Xie, Guozhu Meng, Yang Liu

**Abstract**: Large-scale pre-trained models such as CodeBERT, GraphCodeBERT have earned widespread attention from both academia and industry. Attributed to the superior ability in code representation, they have been further applied in multiple downstream tasks such as clone detection, code search and code translation. However, it is also observed that these state-of-the-art pre-trained models are susceptible to adversarial attacks. The performance of these pre-trained models drops significantly with simple perturbations such as renaming variable names. This weakness may be inherited by their downstream models and thereby amplified at an unprecedented scale. To this end, we propose an approach namely ContraBERT that aims to improve the robustness of pre-trained models via contrastive learning. Specifically, we design nine kinds of simple and complex data augmentation operators on the programming language (PL) and natural language (NL) data to construct different variants. Furthermore, we continue to train the existing pre-trained models by masked language modeling (MLM) and contrastive pre-training task on the original samples with their augmented variants to enhance the robustness of the model. The extensive experiments demonstrate that ContraBERT can effectively improve the robustness of the existing pre-trained models. Further study also confirms that these robustness-enhanced models provide improvements as compared to original models over four popular downstream tasks.



## **23. Provable Unrestricted Adversarial Training without Compromise with Generalizability**

cs.LG

**SubmitDate**: 2023-01-22    [abs](http://arxiv.org/abs/2301.09069v1) [paper-pdf](http://arxiv.org/pdf/2301.09069v1)

**Authors**: Lilin Zhang, Ning Yang, Yanchao Sun, Philip S. Yu

**Abstract**: Adversarial training (AT) is widely considered as the most promising strategy to defend against adversarial attacks and has drawn increasing interest from researchers. However, the existing AT methods still suffer from two challenges. First, they are unable to handle unrestricted adversarial examples (UAEs), which are built from scratch, as opposed to restricted adversarial examples (RAEs), which are created by adding perturbations bound by an $l_p$ norm to observed examples. Second, the existing AT methods often achieve adversarial robustness at the expense of standard generalizability (i.e., the accuracy on natural examples) because they make a tradeoff between them. To overcome these challenges, we propose a unique viewpoint that understands UAEs as imperceptibly perturbed unobserved examples. Also, we find that the tradeoff results from the separation of the distributions of adversarial examples and natural examples. Based on these ideas, we propose a novel AT approach called Provable Unrestricted Adversarial Training (PUAT), which can provide a target classifier with comprehensive adversarial robustness against both UAE and RAE, and simultaneously improve its standard generalizability. Particularly, PUAT utilizes partially labeled data to achieve effective UAE generation by accurately capturing the natural data distribution through a novel augmented triple-GAN. At the same time, PUAT extends the traditional AT by introducing the supervised loss of the target classifier into the adversarial loss and achieves the alignment between the UAE distribution, the natural data distribution, and the distribution learned by the classifier, with the collaboration of the augmented triple-GAN. Finally, the solid theoretical analysis and extensive experiments conducted on widely-used benchmarks demonstrate the superiority of PUAT.



## **24. SUPER-Net: Trustworthy Medical Image Segmentation with Uncertainty Propagation in Encoder-Decoder Networks**

eess.IV

**SubmitDate**: 2023-01-21    [abs](http://arxiv.org/abs/2111.05978v3) [paper-pdf](http://arxiv.org/pdf/2111.05978v3)

**Authors**: Giuseppina Carannante, Dimah Dera, Nidhal C. Bouaynaya, Hassan M. Fathallah-Shaykh, Ghulam Rasool

**Abstract**: Deep Learning (DL) holds great promise in reshaping the healthcare industry owing to its precision, efficiency, and objectivity. However, the brittleness of DL models to noisy and out-of-distribution inputs is ailing their deployment in the clinic. Most models produce point estimates without further information about model uncertainty or confidence. This paper introduces a new Bayesian DL framework for uncertainty quantification in segmentation neural networks: SUPER-Net: trustworthy medical image Segmentation with Uncertainty Propagation in Encoder-decodeR Networks. SUPER-Net analytically propagates, using Taylor series approximations, the first two moments (mean and covariance) of the posterior distribution of the model parameters across the nonlinear layers. In particular, SUPER-Net simultaneously learns the mean and covariance without expensive post-hoc Monte Carlo sampling or model ensembling. The output consists of two simultaneous maps: the segmented image and its pixelwise uncertainty map, which corresponds to the covariance matrix of the predictive distribution. We conduct an extensive evaluation of SUPER-Net on medical image segmentation of Magnetic Resonances Imaging and Computed Tomography scans under various noisy and adversarial conditions. Our experiments on multiple benchmark datasets demonstrate that SUPER-Net is more robust to noise and adversarial attacks than state-of-the-art segmentation models. Moreover, the uncertainty map of the proposed SUPER-Net associates low confidence (or equivalently high uncertainty) to patches in the test input images that are corrupted with noise, artifacts, or adversarial attacks. Perhaps more importantly, the model exhibits the ability of self-assessment of its segmentation decisions, notably when making erroneous predictions due to noise or adversarial examples.



## **25. Dynamics-aware Adversarial Attack of 3D Sparse Convolution Network**

cs.CV

We have improved the quality of this work and updated a new version  to address the limitations of the proposed method

**SubmitDate**: 2023-01-21    [abs](http://arxiv.org/abs/2112.09428v2) [paper-pdf](http://arxiv.org/pdf/2112.09428v2)

**Authors**: An Tao, Yueqi Duan, He Wang, Ziyi Wu, Pengliang Ji, Haowen Sun, Jie Zhou, Jiwen Lu

**Abstract**: In this paper, we investigate the dynamics-aware adversarial attack problem in deep neural networks. Most existing adversarial attack algorithms are designed under a basic assumption -- the network architecture is fixed throughout the attack process. However, this assumption does not hold for many recently proposed networks, e.g. 3D sparse convolution network, which contains input-dependent execution to improve computational efficiency. It results in a serious issue of lagged gradient, making the learned attack at the current step ineffective due to the architecture changes afterward. To address this issue, we propose a Leaded Gradient Method (LGM) and show the significant effects of the lagged gradient. More specifically, we re-formulate the gradients to be aware of the potential dynamic changes of network architectures, so that the learned attack better "leads" the next step than the dynamics-unaware methods when network architecture changes dynamically. Extensive experiments on various datasets show that our LGM achieves impressive performance on semantic segmentation and classification. Compared with the dynamic-unaware methods, LGM achieves about 20% lower mIoU averagely on the ScanNet and S3DIS datasets. LGM also outperforms the recent point cloud attacks.



## **26. Passive Defense Against 3D Adversarial Point Clouds Through the Lens of 3D Steganalysis**

cs.MM

This paper is out-of-date

**SubmitDate**: 2023-01-21    [abs](http://arxiv.org/abs/2205.08738v2) [paper-pdf](http://arxiv.org/pdf/2205.08738v2)

**Authors**: Jiahao Zhu

**Abstract**: Nowadays, 3D data plays an indelible role in the computer vision field. However, extensive studies have proved that deep neural networks (DNNs) fed with 3D data, such as point clouds, are susceptible to adversarial examples, which aim to misguide DNNs and might bring immeasurable losses. Currently, 3D adversarial point clouds are chiefly generated in three fashions, i.e., point shifting, point adding, and point dropping. These point manipulations would modify geometrical properties and local correlations of benign point clouds more or less. Motivated by this basic fact, we propose to defend such adversarial examples with the aid of 3D steganalysis techniques. Specifically, we first introduce an adversarial attack and defense model adapted from the celebrated Prisoners' Problem in steganography to help us comprehend 3D adversarial attack and defense more generally. Then we rethink two significant but vague concepts in the field of adversarial example, namely, active defense and passive defense, from the perspective of steganalysis. Most importantly, we design a 3D adversarial point cloud detector through the lens of 3D steganalysis. Our detector is double-blind, that is to say, it does not rely on the exact knowledge of the adversarial attack means and victim models. To enable the detector to effectively detect malicious point clouds, we craft a 64-D discriminant feature set, including features related to first-order and second-order local descriptions of point clouds. To our knowledge, this work is the first to apply 3D steganalysis to 3D adversarial example defense. Extensive experimental results demonstrate that the proposed 3D adversarial point cloud detector can achieve good detection performance on multiple types of 3D adversarial point clouds.



## **27. How Potent are Evasion Attacks for Poisoning Federated Learning-Based Signal Classifiers?**

eess.SP

6 pages, Accepted to IEEE ICC 2023

**SubmitDate**: 2023-01-21    [abs](http://arxiv.org/abs/2301.08866v1) [paper-pdf](http://arxiv.org/pdf/2301.08866v1)

**Authors**: Su Wang, Rajeev Sahay, Christopher G. Brinton

**Abstract**: There has been recent interest in leveraging federated learning (FL) for radio signal classification tasks. In FL, model parameters are periodically communicated from participating devices, training on their own local datasets, to a central server which aggregates them into a global model. While FL has privacy/security advantages due to raw data not leaving the devices, it is still susceptible to several adversarial attacks. In this work, we reveal the susceptibility of FL-based signal classifiers to model poisoning attacks, which compromise the training process despite not observing data transmissions. In this capacity, we develop an attack framework in which compromised FL devices perturb their local datasets using adversarial evasion attacks. As a result, the training process of the global model significantly degrades on in-distribution signals (i.e., signals received over channels with identical distributions at each edge device). We compare our work to previously proposed FL attacks and reveal that as few as one adversarial device operating with a low-powered perturbation under our attack framework can induce the potent model poisoning attack to the global classifier. Moreover, we find that more devices partaking in adversarial poisoning will proportionally degrade the classification performance.



## **28. Robot Skill Learning Via Classical Robotics-Based Generated Datasets: Advantages, Disadvantages, and Future Improvement**

cs.RO

**SubmitDate**: 2023-01-20    [abs](http://arxiv.org/abs/2301.08794v1) [paper-pdf](http://arxiv.org/pdf/2301.08794v1)

**Authors**: Batu Kaan Oezen

**Abstract**: Why do we not profit from our long-existing classical robotics knowledge and look for some alternative way for data collection? The situation ignoring all existing methods might be such a waste. This article argues that a dataset created using a classical robotics algorithm is a crucial part of future development. This developed classic algorithm has a perfect domain adaptation and generalization property, and most importantly, collecting datasets based on them is quite easy. It is well known that current robot skill-learning approaches perform exceptionally badly in the unseen domain, and their performance against adversarial attacks is quite limited as long as they do not have a very exclusive big dataset. Our experiment is the initial steps of using a dataset created by classical robotics codes. Our experiment investigated possible trajectory collection based on classical robotics. It addressed some advantages and disadvantages and pointed out other future development ideas.



## **29. StratDef: Strategic Defense Against Adversarial Attacks in ML-based Malware Detection**

cs.LG

**SubmitDate**: 2023-01-20    [abs](http://arxiv.org/abs/2202.07568v4) [paper-pdf](http://arxiv.org/pdf/2202.07568v4)

**Authors**: Aqib Rashid, Jose Such

**Abstract**: Over the years, most research towards defenses against adversarial attacks on machine learning models has been in the image recognition domain. The malware detection domain has received less attention despite its importance. Moreover, most work exploring these defenses has focused on several methods but with no strategy when applying them. In this paper, we introduce StratDef, which is a strategic defense system based on a moving target defense approach. We overcome challenges related to the systematic construction, selection, and strategic use of models to maximize adversarial robustness. StratDef dynamically and strategically chooses the best models to increase the uncertainty for the attacker while minimizing critical aspects in the adversarial ML domain, like attack transferability. We provide the first comprehensive evaluation of defenses against adversarial attacks on machine learning for malware detection, where our threat model explores different levels of threat, attacker knowledge, capabilities, and attack intensities. We show that StratDef performs better than other defenses even when facing the peak adversarial threat. We also show that, of the existing defenses, only a few adversarially-trained models provide substantially better protection than just using vanilla models but are still outperformed by StratDef.



## **30. On the Relationship Between Information-Theoretic Privacy Metrics And Probabilistic Information Privacy**

cs.IT

**SubmitDate**: 2023-01-20    [abs](http://arxiv.org/abs/2301.08401v1) [paper-pdf](http://arxiv.org/pdf/2301.08401v1)

**Authors**: Chong Xiao Wang, Wee Peng Tay

**Abstract**: Information-theoretic (IT) measures based on $f$-divergences have recently gained interest as a measure of privacy leakage as they allow for trading off privacy against utility using only a single-value characterization. However, their operational interpretations in the privacy context are unclear. In this paper, we relate the notion of probabilistic information privacy (IP) to several IT privacy metrics based on $f$-divergences. We interpret probabilistic IP under both the detection and estimation frameworks and link it to differential privacy, thus allowing a precise operational interpretation of these IT privacy metrics. We show that the $\chi^2$-divergence privacy metric is stronger than those based on total variation distance and Kullback-Leibler divergence. Therefore, we further develop a data-driven empirical risk framework based on the $\chi^2$-divergence privacy metric and realized using deep neural networks. This framework is agnostic to the adversarial attack model. Empirical experiments demonstrate the efficacy of our approach.



## **31. BO-DBA: Query-Efficient Decision-Based Adversarial Attacks via Bayesian Optimization**

cs.LG

**SubmitDate**: 2023-01-19    [abs](http://arxiv.org/abs/2106.02732v2) [paper-pdf](http://arxiv.org/pdf/2106.02732v2)

**Authors**: Zhuosheng Zhang, Shucheng Yu

**Abstract**: Decision-based attacks (DBA), wherein attackers perturb inputs to spoof learning algorithms by observing solely the output labels, are a type of severe adversarial attacks against Deep Neural Networks (DNNs) requiring minimal knowledge of attackers. State-of-the-art DBA attacks relying on zeroth-order gradient estimation require an excessive number of queries. Recently, Bayesian optimization (BO) has shown promising in reducing the number of queries in score-based attacks (SBA), in which attackers need to observe real-valued probability scores as outputs. However, extending BO to the setting of DBA is nontrivial because in DBA only output labels instead of real-valued scores, as needed by BO, are available to attackers. In this paper, we close this gap by proposing an efficient DBA attack, namely BO-DBA. Different from existing approaches, BO-DBA generates adversarial examples by searching so-called \emph{directions of perturbations}. It then formulates the problem as a BO problem that minimizes the real-valued distortion of perturbations. With the optimized perturbation generation process, BO-DBA converges much faster than the state-of-the-art DBA techniques. Experimental results on pre-trained ImageNet classifiers show that BO-DBA converges within 200 queries while the state-of-the-art DBA techniques need over 15,000 queries to achieve the same level of perturbation distortion. BO-DBA also shows similar attack success rates even as compared to BO-based SBA attacks but with less distortion.



## **32. RNAS-CL: Robust Neural Architecture Search by Cross-Layer Knowledge Distillation**

cs.CV

17 pages, 12 figures

**SubmitDate**: 2023-01-19    [abs](http://arxiv.org/abs/2301.08092v1) [paper-pdf](http://arxiv.org/pdf/2301.08092v1)

**Authors**: Utkarsh Nath, Yancheng Wang, Yingzhen Yang

**Abstract**: Deep Neural Networks are vulnerable to adversarial attacks. Neural Architecture Search (NAS), one of the driving tools of deep neural networks, demonstrates superior performance in prediction accuracy in various machine learning applications. However, it is unclear how it performs against adversarial attacks. Given the presence of a robust teacher, it would be interesting to investigate if NAS would produce robust neural architecture by inheriting robustness from the teacher. In this paper, we propose Robust Neural Architecture Search by Cross-Layer Knowledge Distillation (RNAS-CL), a novel NAS algorithm that improves the robustness of NAS by learning from a robust teacher through cross-layer knowledge distillation. Unlike previous knowledge distillation methods that encourage close student/teacher output only in the last layer, RNAS-CL automatically searches for the best teacher layer to supervise each student layer. Experimental result evidences the effectiveness of RNAS-CL and shows that RNAS-CL produces small and robust neural architecture.



## **33. Evaluating the Robustness of Trigger Set-Based Watermarks Embedded in Deep Neural Networks**

cs.CR

15 pages, accepted at IEEE TDSC

**SubmitDate**: 2023-01-19    [abs](http://arxiv.org/abs/2106.10147v2) [paper-pdf](http://arxiv.org/pdf/2106.10147v2)

**Authors**: Suyoung Lee, Wonho Song, Suman Jana, Meeyoung Cha, Sooel Son

**Abstract**: Trigger set-based watermarking schemes have gained emerging attention as they provide a means to prove ownership for deep neural network model owners. In this paper, we argue that state-of-the-art trigger set-based watermarking algorithms do not achieve their designed goal of proving ownership. We posit that this impaired capability stems from two common experimental flaws that the existing research practice has committed when evaluating the robustness of watermarking algorithms: (1) incomplete adversarial evaluation and (2) overlooked adaptive attacks. We conduct a comprehensive adversarial evaluation of 11 representative watermarking schemes against six of the existing attacks and demonstrate that each of these watermarking schemes lacks robustness against at least two non-adaptive attacks. We also propose novel adaptive attacks that harness the adversary's knowledge of the underlying watermarking algorithm of a target model. We demonstrate that the proposed attacks effectively break all of the 11 watermarking schemes, consequently allowing adversaries to obscure the ownership of any watermarked model. We encourage follow-up studies to consider our guidelines when evaluating the robustness of their watermarking schemes via conducting comprehensive adversarial evaluation that includes our adaptive attacks to demonstrate a meaningful upper bound of watermark robustness.



## **34. Exposing Fine-Grained Adversarial Vulnerability of Face Anti-Spoofing Models**

cs.CV

**SubmitDate**: 2023-01-18    [abs](http://arxiv.org/abs/2205.14851v2) [paper-pdf](http://arxiv.org/pdf/2205.14851v2)

**Authors**: Songlin Yang, Wei Wang, Chenye Xu, Ziwen He, Bo Peng, Jing Dong

**Abstract**: Face anti-spoofing aims to discriminate the spoofing face images (e.g., printed photos) from live ones. However, adversarial examples greatly challenge its credibility, where adding some perturbation noise can easily change the predictions. Previous works conducted adversarial attack methods to evaluate the face anti-spoofing performance without any fine-grained analysis that which model architecture or auxiliary feature is vulnerable to the adversary. To handle this problem, we propose a novel framework to expose the fine-grained adversarial vulnerability of the face anti-spoofing models, which consists of a multitask module and a semantic feature augmentation (SFA) module. The multitask module can obtain different semantic features for further evaluation, but only attacking these semantic features fails to reflect the discrimination-related vulnerability. We then design the SFA module to introduce the data distribution prior for more discrimination-related gradient directions for generating adversarial examples. Comprehensive experiments show that SFA module increases the attack success rate by nearly 40$\%$ on average. We conduct this fine-grained adversarial analysis on different annotations, geometric maps, and backbone networks (e.g., Resnet network). These fine-grained adversarial examples can be used for selecting robust backbone networks and auxiliary features. They also can be used for adversarial training, which makes it practical to further improve the accuracy and robustness of the face anti-spoofing models.



## **35. Generative Adversarial Networks to infer velocity components in rotating turbulent flows**

physics.flu-dyn

**SubmitDate**: 2023-01-18    [abs](http://arxiv.org/abs/2301.07541v1) [paper-pdf](http://arxiv.org/pdf/2301.07541v1)

**Authors**: Tianyi Li, Michele Buzzicotti, Luca Biferale, Fabio Bonaccorso

**Abstract**: Inference problems for two-dimensional snapshots of rotating turbulent flows are studied. We perform a systematic quantitative benchmark of point-wise and statistical reconstruction capabilities of the linear Extended Proper Orthogonal Decomposition (EPOD) method, a non-linear Convolutional Neural Network (CNN) and a Generative Adversarial Network (GAN). We attack the important task of inferring one velocity component out of the measurement of a second one, and two cases are studied: (I) both components lay in the plane orthogonal to the rotation axis and (II) one of the two is parallel to the rotation axis. We show that EPOD method works well only for the former case where both components are strongly correlated, while CNN and GAN always outperform EPOD both concerning point-wise and statistical reconstructions. For case (II), when the input and output data are weakly correlated, all methods fail to reconstruct faithfully the point-wise information. In this case, only GAN is able to reconstruct the field in a statistical sense. The analysis is performed using both standard validation tools based on L2 spatial distance between the prediction and the ground truth and more sophisticated multi-scale analysis using wavelet decomposition. Statistical validation is based on standard Jensen-Shannon divergence between the probability density functions, spectral properties and multi-scale flatness.



## **36. Accurate Detection of Paroxysmal Atrial Fibrillation with Certified-GAN and Neural Architecture Search**

cs.LG

19 pages

**SubmitDate**: 2023-01-17    [abs](http://arxiv.org/abs/2301.10173v1) [paper-pdf](http://arxiv.org/pdf/2301.10173v1)

**Authors**: Mehdi Asadi, Fatemeh Poursalim, Mohammad Loni, Masoud Daneshtalab, Mikael Sjödin, Arash Gharehbaghi

**Abstract**: This paper presents a novel machine learning framework for detecting Paroxysmal Atrial Fibrillation (PxAF), a pathological characteristic of Electrocardiogram (ECG) that can lead to fatal conditions such as heart attack. To enhance the learning process, the framework involves a Generative Adversarial Network (GAN) along with a Neural Architecture Search (NAS) in the data preparation and classifier optimization phases. The GAN is innovatively invoked to overcome the class imbalance of the training data by producing the synthetic ECG for PxAF class in a certified manner. The effect of the certified GAN is statistically validated. Instead of using a general-purpose classifier, the NAS automatically designs a highly accurate convolutional neural network architecture customized for the PxAF classification task. Experimental results show that the accuracy of the proposed framework exhibits a high value of 99% which not only enhances state-of-the-art by up to 5.1%, but also improves the classification performance of the two widely-accepted baseline methods, ResNet-18, and Auto-Sklearn, by 2.2% and 6.1%.



## **37. Denoising Diffusion Probabilistic Models as a Defense against Adversarial Attacks**

cs.LG

**SubmitDate**: 2023-01-17    [abs](http://arxiv.org/abs/2301.06871v1) [paper-pdf](http://arxiv.org/pdf/2301.06871v1)

**Authors**: Lars Lien Ankile, Anna Midgley, Sebastian Weisshaar

**Abstract**: Neural Networks are infamously sensitive to small perturbations in their inputs, making them vulnerable to adversarial attacks. This project evaluates the performance of Denoising Diffusion Probabilistic Models (DDPM) as a purification technique to defend against adversarial attacks. This works by adding noise to an adversarial example before removing it through the reverse process of the diffusion model. We evaluate the approach on the PatchCamelyon data set for histopathologic scans of lymph node sections and find an improvement of the robust accuracy by up to 88\% of the original model's accuracy, constituting a considerable improvement over the vanilla model and our baselines. The project code is located at https://github.com/ankile/Adversarial-Diffusion.



## **38. Database Matching Under Noisy Synchronization Errors**

cs.IT

**SubmitDate**: 2023-01-17    [abs](http://arxiv.org/abs/2301.06796v1) [paper-pdf](http://arxiv.org/pdf/2301.06796v1)

**Authors**: Serhat Bakirtas, Elza Erkip

**Abstract**: The re-identification or de-anonymization of users from anonymized data through matching with publicly-available correlated user data has raised privacy concerns, leading to the complementary measure of obfuscation in addition to anonymization. Recent research provides a fundamental understanding of the conditions under which privacy attacks, in the form of database matching, are successful in the presence of obfuscation. Motivated by synchronization errors stemming from the sampling of time-indexed databases, this paper presents a unified framework considering both obfuscation and synchronization errors and investigates the matching of databases under noisy entry repetitions. By investigating different structures for the repetition pattern, replica detection and seeded deletion detection algorithms are devised and sufficient and necessary conditions for successful matching are derived. Finally, the impacts of some variations of the underlying assumptions, such as adversarial deletion model, seedless database matching and zero-rate regime, on the results are discussed. Overall, our results provide insights into the privacy-preserving publication of anonymized and obfuscated time-indexed data as well as the closely-related problem of the capacity of synchronization channels.



## **39. Adversarial AI in Insurance: Pervasiveness and Resilience**

cs.LG

**SubmitDate**: 2023-01-17    [abs](http://arxiv.org/abs/2301.07520v1) [paper-pdf](http://arxiv.org/pdf/2301.07520v1)

**Authors**: Elisa Luciano, Matteo Cattaneo, Ron Kenett

**Abstract**: The rapid and dynamic pace of Artificial Intelligence (AI) and Machine Learning (ML) is revolutionizing the insurance sector. AI offers significant, very much welcome advantages to insurance companies, and is fundamental to their customer-centricity strategy. It also poses challenges, in the project and implementation phase. Among those, we study Adversarial Attacks, which consist of the creation of modified input data to deceive an AI system and produce false outputs. We provide examples of attacks on insurance AI applications, categorize them, and argue on defence methods and precautionary systems, considering that they can involve few-shot and zero-shot multilabelling. A related topic, with growing interest, is the validation and verification of systems incorporating AI and ML components. These topics are discussed in various sections of this paper.



## **40. Imperceptible Adversarial Attack via Invertible Neural Networks**

cs.CV

**SubmitDate**: 2023-01-17    [abs](http://arxiv.org/abs/2211.15030v3) [paper-pdf](http://arxiv.org/pdf/2211.15030v3)

**Authors**: Zihan Chen, Ziyue Wang, Junjie Huang, Wentao Zhao, Xiao Liu, Dejian Guan

**Abstract**: Adding perturbations via utilizing auxiliary gradient information or discarding existing details of the benign images are two common approaches for generating adversarial examples. Though visual imperceptibility is the desired property of adversarial examples, conventional adversarial attacks still generate traceable adversarial perturbations. In this paper, we introduce a novel Adversarial Attack via Invertible Neural Networks (AdvINN) method to produce robust and imperceptible adversarial examples. Specifically, AdvINN fully takes advantage of the information preservation property of Invertible Neural Networks and thereby generates adversarial examples by simultaneously adding class-specific semantic information of the target class and dropping discriminant information of the original class. Extensive experiments on CIFAR-10, CIFAR-100, and ImageNet-1K demonstrate that the proposed AdvINN method can produce less imperceptible adversarial images than the state-of-the-art methods and AdvINN yields more robust adversarial examples with high confidence compared to other adversarial attacks.



## **41. Private Eye: On the Limits of Textual Screen Peeking via Eyeglass Reflections in Video Conferencing**

cs.CR

**SubmitDate**: 2023-01-16    [abs](http://arxiv.org/abs/2205.03971v3) [paper-pdf](http://arxiv.org/pdf/2205.03971v3)

**Authors**: Yan Long, Chen Yan, Shilin Xiao, Shivan Prasad, Wenyuan Xu, Kevin Fu

**Abstract**: Using mathematical modeling and human subjects experiments, this research explores the extent to which emerging webcams might leak recognizable textual and graphical information gleaming from eyeglass reflections captured by webcams. The primary goal of our work is to measure, compute, and predict the factors, limits, and thresholds of recognizability as webcam technology evolves in the future. Our work explores and characterizes the viable threat models based on optical attacks using multi-frame super resolution techniques on sequences of video frames. Our models and experimental results in a controlled lab setting show it is possible to reconstruct and recognize with over 75% accuracy on-screen texts that have heights as small as 10 mm with a 720p webcam. We further apply this threat model to web textual contents with varying attacker capabilities to find thresholds at which text becomes recognizable. Our user study with 20 participants suggests present-day 720p webcams are sufficient for adversaries to reconstruct textual content on big-font websites. Our models further show that the evolution towards 4K cameras will tip the threshold of text leakage to reconstruction of most header texts on popular websites. Besides textual targets, a case study on recognizing a closed-world dataset of Alexa top 100 websites with 720p webcams shows a maximum recognition accuracy of 94% with 10 participants even without using machine-learning models. Our research proposes near-term mitigations including a software prototype that users can use to blur the eyeglass areas of their video streams. For possible long-term defenses, we advocate an individual reflection testing procedure to assess threats under various settings, and justify the importance of following the principle of least privilege for privacy-sensitive scenarios.



## **42. Defending Backdoor Attacks on Vision Transformer via Patch Processing**

cs.CV

**SubmitDate**: 2023-01-16    [abs](http://arxiv.org/abs/2206.12381v2) [paper-pdf](http://arxiv.org/pdf/2206.12381v2)

**Authors**: Khoa D. Doan, Yingjie Lao, Peng Yang, Ping Li

**Abstract**: Vision Transformers (ViTs) have a radically different architecture with significantly less inductive bias than Convolutional Neural Networks. Along with the improvement in performance, security and robustness of ViTs are also of great importance to study. In contrast to many recent works that exploit the robustness of ViTs against adversarial examples, this paper investigates a representative causative attack, i.e., backdoor. We first examine the vulnerability of ViTs against various backdoor attacks and find that ViTs are also quite vulnerable to existing attacks. However, we observe that the clean-data accuracy and backdoor attack success rate of ViTs respond distinctively to patch transformations before the positional encoding. Then, based on this finding, we propose an effective method for ViTs to defend both patch-based and blending-based trigger backdoor attacks via patch processing. The performances are evaluated on several benchmark datasets, including CIFAR10, GTSRB, and TinyImageNet, which show the proposed novel defense is very successful in mitigating backdoor attacks for ViTs. To the best of our knowledge, this paper presents the first defensive strategy that utilizes a unique characteristic of ViTs against backdoor attacks.   The paper will appear in the Proceedings of the AAAI'23 Conference. This work was initially submitted in November 2021 to CVPR'22, then it was re-submitted to ECCV'22. The paper was made public in June 2022. The authors sincerely thank all the referees from the Program Committees of CVPR'22, ECCV'22, and AAAI'23.



## **43. Meta Generative Attack on Person Reidentification**

cs.CV

**SubmitDate**: 2023-01-16    [abs](http://arxiv.org/abs/2301.06286v1) [paper-pdf](http://arxiv.org/pdf/2301.06286v1)

**Authors**: A V Subramanyam

**Abstract**: Adversarial attacks have been recently investigated in person re-identification. These attacks perform well under cross dataset or cross model setting. However, the challenges present in cross-dataset cross-model scenario does not allow these models to achieve similar accuracy. To this end, we propose our method with the goal of achieving better transferability against different models and across datasets. We generate a mask to obtain better performance across models and use meta learning to boost the generalizability in the challenging cross-dataset cross-model setting. Experiments on Market-1501, DukeMTMC-reID and MSMT-17 demonstrate favorable results compared to other attacks.



## **44. A Search-Based Testing Approach for Deep Reinforcement Learning Agents**

cs.SE

**SubmitDate**: 2023-01-14    [abs](http://arxiv.org/abs/2206.07813v2) [paper-pdf](http://arxiv.org/pdf/2206.07813v2)

**Authors**: Amirhossein Zolfagharian, Manel Abdellatif, Lionel Briand, Mojtaba Bagherzadeh, Ramesh S

**Abstract**: Deep Reinforcement Learning (DRL) algorithms have been increasingly employed during the last decade to solve various decision-making problems such as autonomous driving and robotics. However, these algorithms have faced great challenges when deployed in safety-critical environments since they often exhibit erroneous behaviors that can lead to potentially critical errors. One way to assess the safety of DRL agents is to test them to detect possible faults leading to critical failures during their execution. This raises the question of how we can efficiently test DRL policies to ensure their correctness and adherence to safety requirements. Most existing works on testing DRL agents use adversarial attacks that perturb states or actions of the agent. However, such attacks often lead to unrealistic states of the environment. Their main goal is to test the robustness of DRL agents rather than testing the compliance of agents' policies with respect to requirements. Due to the huge state space of DRL environments, the high cost of test execution, and the black-box nature of DRL algorithms, the exhaustive testing of DRL agents is impossible. In this paper, we propose a Search-based Testing Approach of Reinforcement Learning Agents (STARLA) to test the policy of a DRL agent by effectively searching for failing executions of the agent within a limited testing budget. We use machine learning models and a dedicated genetic algorithm to narrow the search towards faulty episodes. We apply STARLA on Deep-Q-Learning agents which are widely used as benchmarks and show that it significantly outperforms Random Testing by detecting more faults related to the agent's policy. We also investigate how to extract rules that characterize faulty episodes of the DRL agent using our search results. Such rules can be used to understand the conditions under which the agent fails and thus assess its deployment risks.



## **45. SoK: Data Privacy in Virtual Reality**

cs.HC

**SubmitDate**: 2023-01-14    [abs](http://arxiv.org/abs/2301.05940v1) [paper-pdf](http://arxiv.org/pdf/2301.05940v1)

**Authors**: Gonzalo Munilla Garrido, Vivek Nair, Dawn Song

**Abstract**: The adoption of virtual reality (VR) technologies has rapidly gained momentum in recent years as companies around the world begin to position the so-called "metaverse" as the next major medium for accessing and interacting with the internet. While consumers have become accustomed to a degree of data harvesting on the web, the real-time nature of data sharing in the metaverse indicates that privacy concerns are likely to be even more prevalent in the new "Web 3.0." Research into VR privacy has demonstrated that a plethora of sensitive personal information is observable by various would-be adversaries from just a few minutes of telemetry data. On the other hand, we have yet to see VR parallels for many privacy-preserving tools aimed at mitigating threats on conventional platforms. This paper aims to systematize knowledge on the landscape of VR privacy threats and countermeasures by proposing a comprehensive taxonomy of data attributes, protections, and adversaries based on the study of 68 collected publications. We complement our qualitative discussion with a statistical analysis of the risk associated with various data sources inherent to VR in consideration of the known attacks and defenses. By focusing on highlighting the clear outstanding opportunities, we hope to motivate and guide further research into this increasingly important field.



## **46. Deepfake Detection using Biological Features: A Survey**

cs.CV

**SubmitDate**: 2023-01-14    [abs](http://arxiv.org/abs/2301.05819v1) [paper-pdf](http://arxiv.org/pdf/2301.05819v1)

**Authors**: Kundan Patil, Shrushti Kale, Jaivanti Dhokey, Abhishek Gulhane

**Abstract**: Deepfake is a deep learning-based technique that makes it easy to change or modify images and videos. In investigations and court, visual evidence is commonly employed, but these pieces of evidence may now be suspect due to technological advancements in deepfake. Deepfakes have been used to blackmail individuals, plan terrorist attacks, disseminate false information, defame individuals, and foment political turmoil. This study describes the history of deepfake, its development and detection, and the challenges based on physiological measurements such as eyebrow recognition, eye blinking detection, eye movement detection, ear and mouth detection, and heartbeat detection. The study also proposes a scope in this field and compares the different biological features and their classifiers. Deepfakes are created using the generative adversarial network (GANs) model, and were once easy to detect by humans due to visible artifacts. However, as technology has advanced, deepfakes have become highly indistinguishable from natural images, making it important to review detection methods.



## **47. $A^{3}D$: A Platform of Searching for Robust Neural Architectures and Efficient Adversarial Attacks**

cs.LG

**SubmitDate**: 2023-01-14    [abs](http://arxiv.org/abs/2203.03128v2) [paper-pdf](http://arxiv.org/pdf/2203.03128v2)

**Authors**: Jialiang Sun, Wen Yao, Tingsong Jiang, Chao Li, Xiaoqian Chen

**Abstract**: The robustness of deep neural networks (DNN) models has attracted increasing attention due to the urgent need for security in many applications. Numerous existing open-sourced tools or platforms are developed to evaluate the robustness of DNN models by ensembling the majority of adversarial attack or defense algorithms. Unfortunately, current platforms do not possess the ability to optimize the architectures of DNN models or the configuration of adversarial attacks to further enhance the robustness of models or the performance of adversarial attacks. To alleviate these problems, in this paper, we first propose a novel platform called auto adversarial attack and defense ($A^{3}D$), which can help search for robust neural network architectures and efficient adversarial attacks. In $A^{3}D$, we employ multiple neural architecture search methods, which consider different robustness evaluation metrics, including four types of noises: adversarial noise, natural noise, system noise, and quantified metrics, resulting in finding robust architectures. Besides, we propose a mathematical model for auto adversarial attack, and provide multiple optimization algorithms to search for efficient adversarial attacks. In addition, we combine auto adversarial attack and defense together to form a unified framework. Among auto adversarial defense, the searched efficient attack can be used as the new robustness evaluation to further enhance the robustness. In auto adversarial attack, the searched robust architectures can be utilized as the threat model to help find stronger adversarial attacks. Experiments on CIFAR10, CIFAR100, and ImageNet datasets demonstrate the feasibility and effectiveness of the proposed platform, which can also provide a benchmark and toolkit for researchers in the application of automated machine learning in evaluating and improving the DNN model robustnesses.



## **48. Threat Models over Space and Time: A Case Study of E2EE Messaging Applications**

cs.CR

**SubmitDate**: 2023-01-13    [abs](http://arxiv.org/abs/2301.05653v1) [paper-pdf](http://arxiv.org/pdf/2301.05653v1)

**Authors**: Partha Das Chowdhury, Maria Sameen, Jenny Blessing, Nicholas Boucher, Joseph Gardiner, Tom Burrows, Ross Anderson, Awais Rashid

**Abstract**: Threat modelling is foundational to secure systems engineering and should be done in consideration of the context within which systems operate. On the other hand, the continuous evolution of both the technical sophistication of threats and the system attack surface is an inescapable reality. In this work, we explore the extent to which real-world systems engineering reflects the changing threat context. To this end we examine the desktop clients of six widely used end-to-end-encrypted mobile messaging applications to understand the extent to which they adjusted their threat model over space (when enabling clients on new platforms, such as desktop clients) and time (as new threats emerged). We experimented with short-lived adversarial access against these desktop clients and analyzed the results with respect to two popular threat elicitation frameworks, STRIDE and LINDDUN. The results demonstrate that system designers need to both recognise the threats in the evolving context within which systems operate and, more importantly, to mitigate them by rescoping trust boundaries in a manner that those within the administrative boundary cannot violate security and privacy properties. Such a nuanced understanding of trust boundary scopes and their relationship with administrative boundaries allows for better administration of shared components, including securing them with safe defaults.



## **49. Resilient Model Predictive Control of Distributed Systems Under Attack Using Local Attack Identification**

cs.SY

Submitted for review to Springer Natural Computer Science on November  18th 2022

**SubmitDate**: 2023-01-13    [abs](http://arxiv.org/abs/2301.05547v1) [paper-pdf](http://arxiv.org/pdf/2301.05547v1)

**Authors**: Sarah Braun, Sebastian Albrecht, Sergio Lucia

**Abstract**: With the growing share of renewable energy sources, the uncertainty in power supply is increasing. In addition to the inherent fluctuations in the renewables, this is due to the threat of deliberate malicious attacks, which may become more revalent with a growing number of distributed generation units. Also in other safety-critical technology sectors, control systems are becoming more and more decentralized, causing the targets for attackers and thus the risk of attacks to increase. It is thus essential that distributed controllers are robust toward these uncertainties and able to react quickly to disturbances of any kind. To this end, we present novel methods for model-based identification of attacks and combine them with distributed model predictive control to obtain a resilient framework for adaptively robust control. The methodology is specially designed for distributed setups with limited local information due to privacy and security reasons. To demonstrate the efficiency of the method, we introduce a mathematical model for physically coupled microgrids under the uncertain influence of renewable generation and adversarial attacks, and perform numerical experiments, applying the proposed method for microgrid control.



## **50. PMFault: Faulting and Bricking Server CPUs through Management Interfaces**

cs.CR

For demo and source code, visit https://zt-chen.github.io/PMFault/

**SubmitDate**: 2023-01-13    [abs](http://arxiv.org/abs/2301.05538v1) [paper-pdf](http://arxiv.org/pdf/2301.05538v1)

**Authors**: Zitai Chen, David Oswald

**Abstract**: Apart from the actual CPU, modern server motherboards contain other auxiliary components, for example voltage regulators for power management. Those are connected to the CPU and the separate Baseboard Management Controller (BMC) via the I2C-based PMBus.   In this paper, using the case study of the widely used Supermicro X11SSL motherboard, we show how remotely exploitable software weaknesses in the BMC (or other processors with PMBus access) can be used to access the PMBus and then perform hardware-based fault injection attacks on the main CPU. The underlying weaknesses include insecure firmware encryption and signing mechanisms, a lack of authentication for the firmware upgrade process and the IPMI KCS control interface, as well as the motherboard design (with the PMBus connected to the BMC and SMBus by default).   First, we show that undervolting through the PMBus allows breaking the integrity guarantees of SGX enclaves, bypassing Intel's countermeasures against previous undervolting attacks like Plundervolt/V0ltPwn. Second, we experimentally show that overvolting outside the specified range has the potential of permanently damaging Intel Xeon CPUs, rendering the server inoperable. We assess the impact of our findings on other server motherboards made by Supermicro and ASRock.   Our attacks, dubbed PMFault, can be carried out by a privileged software adversary and do not require physical access to the server motherboard or knowledge of the BMC login credentials.   We responsibly disclosed the issues reported in this paper to Supermicro and discuss possible countermeasures at different levels. To the best of our knowledge, the 12th generation of Supermicro motherboards, which was designed before we reported PMFault to Supermicro, is not vulnerable.



