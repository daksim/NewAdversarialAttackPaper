# Latest Adversarial Attack Papers
**update at 2022-09-15 06:31:28**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Certified Defences Against Adversarial Patch Attacks on Semantic Segmentation**

cs.CV

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05980v1)

**Authors**: Maksym Yatsura, Kaspar Sakmann, N. Grace Hua, Matthias Hein, Jan Hendrik Metzen

**Abstracts**: Adversarial patch attacks are an emerging security threat for real world deep learning applications. We present Demasked Smoothing, the first approach (up to our knowledge) to certify the robustness of semantic segmentation models against this threat model. Previous work on certifiably defending against patch attacks has mostly focused on image classification task and often required changes in the model architecture and additional training which is undesirable and computationally expensive. In Demasked Smoothing, any segmentation model can be applied without particular training, fine-tuning, or restriction of the architecture. Using different masking strategies, Demasked Smoothing can be applied both for certified detection and certified recovery. In extensive experiments we show that Demasked Smoothing can on average certify 64% of the pixel predictions for a 1% patch in the detection task and 48% against a 0.5% patch for the recovery task on the ADE20K dataset.



## **2. Adversarial Inter-Group Link Injection Degrades the Fairness of Graph Neural Networks**

cs.LG

A shorter version of this work has been accepted by IEEE ICDM 2022

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05957v1)

**Authors**: Hussain Hussain, Meng Cao, Sandipan Sikdar, Denis Helic, Elisabeth Lex, Markus Strohmaier, Roman Kern

**Abstracts**: We present evidence for the existence and effectiveness of adversarial attacks on graph neural networks (GNNs) that aim to degrade fairness. These attacks can disadvantage a particular subgroup of nodes in GNN-based node classification, where nodes of the underlying network have sensitive attributes, such as race or gender. We conduct qualitative and experimental analyses explaining how adversarial link injection impairs the fairness of GNN predictions. For example, an attacker can compromise the fairness of GNN-based node classification by injecting adversarial links between nodes belonging to opposite subgroups and opposite class labels. Our experiments on empirical datasets demonstrate that adversarial fairness attacks can significantly degrade the fairness of GNN predictions (attacks are effective) with a low perturbation rate (attacks are efficient) and without a significant drop in accuracy (attacks are deceptive). This work demonstrates the vulnerability of GNN models to adversarial fairness attacks. We hope our findings raise awareness about this issue in our community and lay a foundation for the future development of GNN models that are more robust to such attacks.



## **3. An Evolutionary, Gradient-Free, Query-Efficient, Black-Box Algorithm for Generating Adversarial Instances in Deep Networks**

cs.CV

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2208.08297v2)

**Authors**: Raz Lapid, Zvika Haramaty, Moshe Sipper

**Abstracts**: Deep neural networks (DNNs) are sensitive to adversarial data in a variety of scenarios, including the black-box scenario, where the attacker is only allowed to query the trained model and receive an output. Existing black-box methods for creating adversarial instances are costly, often using gradient estimation or training a replacement network. This paper introduces \textbf{Qu}ery-Efficient \textbf{E}volutiona\textbf{ry} \textbf{Attack}, \textit{QuEry Attack}, an untargeted, score-based, black-box attack. QuEry Attack is based on a novel objective function that can be used in gradient-free optimization problems. The attack only requires access to the output logits of the classifier and is thus not affected by gradient masking. No additional information is needed, rendering our method more suitable to real-life situations. We test its performance with three different state-of-the-art models -- Inception-v3, ResNet-50, and VGG-16-BN -- against three benchmark datasets: MNIST, CIFAR10 and ImageNet. Furthermore, we evaluate QuEry Attack's performance on non-differential transformation defenses and state-of-the-art robust models. Our results demonstrate the superior performance of QuEry Attack, both in terms of accuracy score and query efficiency.



## **4. Bayesian Pseudo Labels: Expectation Maximization for Robust and Efficient Semi-Supervised Segmentation**

cs.CV

MICCAI 2022 (Early accept, Student Travel Award)

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2208.04435v3)

**Authors**: Mou-Cheng Xu, Yukun Zhou, Chen Jin, Marius de Groot, Daniel C. Alexander, Neil P. Oxtoby, Yipeng Hu, Joseph Jacob

**Abstracts**: This paper concerns pseudo labelling in segmentation. Our contribution is fourfold. Firstly, we present a new formulation of pseudo-labelling as an Expectation-Maximization (EM) algorithm for clear statistical interpretation. Secondly, we propose a semi-supervised medical image segmentation method purely based on the original pseudo labelling, namely SegPL. We demonstrate SegPL is a competitive approach against state-of-the-art consistency regularisation based methods on semi-supervised segmentation on a 2D multi-class MRI brain tumour segmentation task and a 3D binary CT lung vessel segmentation task. The simplicity of SegPL allows less computational cost comparing to prior methods. Thirdly, we demonstrate that the effectiveness of SegPL may originate from its robustness against out-of-distribution noises and adversarial attacks. Lastly, under the EM framework, we introduce a probabilistic generalisation of SegPL via variational inference, which learns a dynamic threshold for pseudo labelling during the training. We show that SegPL with variational inference can perform uncertainty estimation on par with the gold-standard method Deep Ensemble.



## **5. Adversarial Coreset Selection for Efficient Robust Training**

cs.LG

Extended version of the ECCV2022 paper: arXiv:2112.00378. arXiv admin  note: substantial text overlap with arXiv:2112.00378

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05785v1)

**Authors**: Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstracts**: Neural networks are vulnerable to adversarial attacks: adding well-crafted, imperceptible perturbations to their input can modify their output. Adversarial training is one of the most effective approaches to training robust models against such attacks. Unfortunately, this method is much slower than vanilla training of neural networks since it needs to construct adversarial examples for the entire training data at every iteration. By leveraging the theory of coreset selection, we show how selecting a small subset of training data provides a principled approach to reducing the time complexity of robust training. To this end, we first provide convergence guarantees for adversarial coreset selection. In particular, we show that the convergence bound is directly related to how well our coresets can approximate the gradient computed over the entire training data. Motivated by our theoretical analysis, we propose using this gradient approximation error as our adversarial coreset selection objective to reduce the training set size effectively. Once built, we run adversarial training over this subset of the training data. Unlike existing methods, our approach can be adapted to a wide variety of training objectives, including TRADES, $\ell_p$-PGD, and Perceptual Adversarial Training. We conduct extensive experiments to demonstrate that our approach speeds up adversarial training by 2-3 times while experiencing a slight degradation in the clean and robust accuracy.



## **6. Adaptive Perturbation Generation for Multiple Backdoors Detection**

cs.CV

7 pages, 5 figures

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05244v2)

**Authors**: Yuhang Wang, Huafeng Shi, Rui Min, Ruijia Wu, Siyuan Liang, Yichao Wu, Ding Liang, Aishan Liu

**Abstracts**: Extensive evidence has demonstrated that deep neural networks (DNNs) are vulnerable to backdoor attacks, which motivates the development of backdoor detection methods. Existing backdoor detection methods are typically tailored for backdoor attacks with individual specific types (e.g., patch-based or perturbation-based). However, adversaries are likely to generate multiple types of backdoor attacks in practice, which challenges the current detection strategies. Based on the fact that adversarial perturbations are highly correlated with trigger patterns, this paper proposes the Adaptive Perturbation Generation (APG) framework to detect multiple types of backdoor attacks by adaptively injecting adversarial perturbations. Since different trigger patterns turn out to show highly diverse behaviors under the same adversarial perturbations, we first design the global-to-local strategy to fit the multiple types of backdoor triggers via adjusting the region and budget of attacks. To further increase the efficiency of perturbation injection, we introduce a gradient-guided mask generation strategy to search for the optimal regions for adversarial attacks. Extensive experiments conducted on multiple datasets (CIFAR-10, GTSRB, Tiny-ImageNet) demonstrate that our method outperforms state-of-the-art baselines by large margins(+12%).



## **7. A Tale of HodgeRank and Spectral Method: Target Attack Against Rank Aggregation Is the Fixed Point of Adversarial Game**

cs.LG

33 pages,  https://github.com/alphaprime/Target_Attack_Rank_Aggregation

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05742v1)

**Authors**: Ke Ma, Qianqian Xu, Jinshan Zeng, Guorong Li, Xiaochun Cao, Qingming Huang

**Abstracts**: Rank aggregation with pairwise comparisons has shown promising results in elections, sports competitions, recommendations, and information retrieval. However, little attention has been paid to the security issue of such algorithms, in contrast to numerous research work on the computational and statistical characteristics. Driven by huge profits, the potential adversary has strong motivation and incentives to manipulate the ranking list. Meanwhile, the intrinsic vulnerability of the rank aggregation methods is not well studied in the literature. To fully understand the possible risks, we focus on the purposeful adversary who desires to designate the aggregated results by modifying the pairwise data in this paper. From the perspective of the dynamical system, the attack behavior with a target ranking list is a fixed point belonging to the composition of the adversary and the victim. To perform the targeted attack, we formulate the interaction between the adversary and the victim as a game-theoretic framework consisting of two continuous operators while Nash equilibrium is established. Then two procedures against HodgeRank and RankCentrality are constructed to produce the modification of the original data. Furthermore, we prove that the victims will produce the target ranking list once the adversary masters the complete information. It is noteworthy that the proposed methods allow the adversary only to hold incomplete information or imperfect feedback and perform the purposeful attack. The effectiveness of the suggested target attack strategies is demonstrated by a series of toy simulations and several real-world data experiments. These experimental results show that the proposed methods could achieve the attacker's goal in the sense that the leading candidate of the perturbed ranking list is the designated one by the adversary.



## **8. Sample Complexity of an Adversarial Attack on UCB-based Best-arm Identification Policy**

cs.LG

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05692v1)

**Authors**: Varsha Pendyala

**Abstracts**: In this work I study the problem of adversarial perturbations to rewards, in a Multi-armed bandit (MAB) setting. Specifically, I focus on an adversarial attack to a UCB type best-arm identification policy applied to a stochastic MAB. The UCB attack presented in [1] results in pulling a target arm K very often. I used the attack model of [1] to derive the sample complexity required for selecting target arm K as the best arm. I have proved that the stopping condition of UCB based best-arm identification algorithm given in [2], can be achieved by the target arm K in T rounds, where T depends only on the total number of arms and $\sigma$ parameter of $\sigma^2-$ sub-Gaussian random rewards of the arms.



## **9. Replay-based Recovery for Autonomous Robotic Vehicles from Sensor Deception Attacks**

cs.RO

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.04554v2)

**Authors**: Pritam Dash, Guanpeng Li, Mehdi Karimibiuki, Karthik Pattabiraman

**Abstracts**: Sensors are crucial for autonomous operation in robotic vehicles (RV). Physical attacks on sensors such as sensor tampering or spoofing can feed erroneous values to RVs through physical channels, which results in mission failures. In this paper, we present DeLorean, a comprehensive diagnosis and recovery framework for securing autonomous RVs from physical attacks. We consider a strong form of physical attack called sensor deception attacks (SDAs), in which the adversary targets multiple sensors of different types simultaneously (even including all sensors). Under SDAs, DeLorean inspects the attack induced errors, identifies the targeted sensors, and prevents the erroneous sensor inputs from being used in RV's feedback control loop. DeLorean replays historic state information in the feedback control loop and recovers the RV from attacks. Our evaluation on four real and two simulated RVs shows that DeLorean can recover RVs from different attacks, and ensure mission success in 94% of the cases (on average), without any crashes. DeLorean incurs low performance, memory and battery overheads.



## **10. Boosting Robustness Verification of Semantic Feature Neighborhoods**

cs.LG

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.05446v1)

**Authors**: Anan Kabaha, Dana Drachsler-Cohen

**Abstracts**: Deep neural networks have been shown to be vulnerable to adversarial attacks that perturb inputs based on semantic features. Existing robustness analyzers can reason about semantic feature neighborhoods to increase the networks' reliability. However, despite the significant progress in these techniques, they still struggle to scale to deep networks and large neighborhoods. In this work, we introduce VeeP, an active learning approach that splits the verification process into a series of smaller verification steps, each is submitted to an existing robustness analyzer. The key idea is to build on prior steps to predict the next optimal step. The optimal step is predicted by estimating the certification velocity and sensitivity via parametric regression. We evaluate VeeP on MNIST, Fashion-MNIST, CIFAR-10 and ImageNet and show that it can analyze neighborhoods of various features: brightness, contrast, hue, saturation, and lightness. We show that, on average, given a 90 minute timeout, VeeP verifies 96% of the maximally certifiable neighborhoods within 29 minutes, while existing splitting approaches verify, on average, 73% of the maximally certifiable neighborhoods within 58 minutes.



## **11. Bilevel Optimization with a Lower-level Contraction: Optimal Sample Complexity without Warm-Start**

stat.ML

35 pages, 2 figures. Code at  https://github.com/CSML-IIT-UCL/bioptexps

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2202.03397v2)

**Authors**: Riccardo Grazzi, Massimiliano Pontil, Saverio Salzo

**Abstracts**: We analyze a general class of bilevel problems, in which the upper-level problem consists in the minimization of a smooth objective function and the lower-level problem is to find the fixed point of a smooth contraction map. This type of problems include instances of meta-learning, equilibrium models, hyperparameter optimization and data poisoning adversarial attacks. Several recent works have proposed algorithms which warm-start the lower level problem, i.e. they use the previous lower-level approximate solution as a staring point for the lower-level solver. This warm-start procedure allows one to improve the sample complexity in both the stochastic and deterministic settings, achieving in some cases the order-wise optimal sample complexity. However, there are situations, e.g., meta learning and equilibrium models, in which the warm-start procedure is not well-suited or ineffective. In this work we show that without warm-start, it is still possible to achieve order-wise optimal or near-optimal sample complexity. In particular, we propose a simple method which uses stochastic fixed point iterations at the lower-level and projected inexact gradient descent at the upper-level, that reaches an $\epsilon$-stationary point using $O(\epsilon^{-2})$ and $\tilde{O}(\epsilon^{-1})$ samples for the stochastic and the deterministic setting, respectively. Finally, compared to methods using warm-start, our approach yields a simpler analysis that does not need to study the coupled interactions between the upper-level and lower-level iterates



## **12. Fiat-Shamir for Proofs Lacks a Proof Even in the Presence of Shared Entanglement**

quant-ph

62 pages, 2 figures

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2204.02265v2)

**Authors**: Frédéric Dupuis, Philippe Lamontagne, Louis Salvail

**Abstracts**: We explore the cryptographic power of arbitrary shared physical resources. The most general such resource is access to a fresh entangled quantum state at the outset of each protocol execution. We call this the Common Reference Quantum State (CRQS) model, in analogy to the well-known Common Reference String (CRS). The CRQS model is a natural generalization of the CRS model but appears to be more powerful: in the two-party setting, a CRQS can sometimes exhibit properties associated with a Random Oracle queried once by measuring a maximally entangled state in one of many mutually unbiased bases. We formalize this notion as a Weak One-Time Random Oracle (WOTRO), where we only ask of the $m$--bit output to have some randomness when conditioned on the $n$--bit input.   We show that when $n-m\in\omega(\lg n)$, any protocol for WOTRO in the CRQS model can be attacked by an (inefficient) adversary. Moreover, our adversary is efficiently simulatable, which rules out the possibility of proving the computational security of a scheme by a black-box reduction to a cryptographic game assumption. On the other hand, we introduce a non-game quantum assumption for hash functions that implies WOTRO in the CRQ\$ model (where the CRQS consists only of EPR pairs). We first build a statistically secure WOTRO protocol where $m=n$, then hash the output.   The impossibility of WOTRO has the following consequences. First, we show the black-box impossibility of a quantum Fiat-Shamir transform, extending the impossibility result of Bitansky et al. (TCC '13) to the CRQS model. Second, we show a black-box impossibility result for a strenghtened version of quantum lightning (Zhandry, Eurocrypt '19) where quantum bolts have an additional parameter that cannot be changed without generating new bolts.



## **13. A Survey of Machine Unlearning**

cs.LG

fixed overlaps

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.02299v4)

**Authors**: Thanh Tam Nguyen, Thanh Trung Huynh, Phi Le Nguyen, Alan Wee-Chung Liew, Hongzhi Yin, Quoc Viet Hung Nguyen

**Abstracts**: Computer systems hold a large amount of personal data over decades. On the one hand, such data abundance allows breakthroughs in artificial intelligence (AI), especially machine learning (ML) models. On the other hand, it can threaten the privacy of users and weaken the trust between humans and AI. Recent regulations require that private information about a user can be removed from computer systems in general and from ML models in particular upon request (e.g. the "right to be forgotten"). While removing data from back-end databases should be straightforward, it is not sufficient in the AI context as ML models often "remember" the old data. Existing adversarial attacks proved that we can learn private membership or attributes of the training data from the trained models. This phenomenon calls for a new paradigm, namely machine unlearning, to make ML models forget about particular data. It turns out that recent works on machine unlearning have not been able to solve the problem completely due to the lack of common frameworks and resources. In this survey paper, we seek to provide a thorough investigation of machine unlearning in its definitions, scenarios, mechanisms, and applications. Specifically, as a categorical collection of state-of-the-art research, we hope to provide a broad reference for those seeking a primer on machine unlearning and its various formulations, design requirements, removal requests, algorithms, and uses in a variety of ML applications. Furthermore, we hope to outline key findings and trends in the paradigm as well as highlight new areas of research that have yet to see the application of machine unlearning, but could nonetheless benefit immensely. We hope this survey provides a valuable reference for ML researchers as well as those seeking to innovate privacy technologies. Our resources are at https://github.com/tamlhp/awesome-machine-unlearning.



## **14. GRNN: Generative Regression Neural Network -- A Data Leakage Attack for Federated Learning**

cs.LG

The source code can be found at: https://github.com/Rand2AI/GRNN

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2105.00529v3)

**Authors**: Hanchi Ren, Jingjing Deng, Xianghua Xie

**Abstracts**: Data privacy has become an increasingly important issue in Machine Learning (ML), where many approaches have been developed to tackle this challenge, e.g. cryptography (Homomorphic Encryption (HE), Differential Privacy (DP), etc.) and collaborative training (Secure Multi-Party Computation (MPC), Distributed Learning and Federated Learning (FL)). These techniques have a particular focus on data encryption or secure local computation. They transfer the intermediate information to the third party to compute the final result. Gradient exchanging is commonly considered to be a secure way of training a robust model collaboratively in Deep Learning (DL). However, recent researches have demonstrated that sensitive information can be recovered from the shared gradient. Generative Adversarial Network (GAN), in particular, has shown to be effective in recovering such information. However, GAN based techniques require additional information, such as class labels which are generally unavailable for privacy-preserved learning. In this paper, we show that, in the FL system, image-based privacy data can be easily recovered in full from the shared gradient only via our proposed Generative Regression Neural Network (GRNN). We formulate the attack to be a regression problem and optimize two branches of the generative model by minimizing the distance between gradients. We evaluate our method on several image classification tasks. The results illustrate that our proposed GRNN outperforms state-of-the-art methods with better stability, stronger robustness, and higher accuracy. It also has no convergence requirement to the global FL model. Moreover, we demonstrate information leakage using face re-identification. Some defense strategies are also discussed in this work.



## **15. Semantic-Preserving Adversarial Code Comprehension**

cs.CL

Accepted by COLING 2022

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.05130v1)

**Authors**: Yiyang Li, Hongqiu Wu, Hai Zhao

**Abstracts**: Based on the tremendous success of pre-trained language models (PrLMs) for source code comprehension tasks, current literature studies either ways to further improve the performance (generalization) of PrLMs, or their robustness against adversarial attacks. However, they have to compromise on the trade-off between the two aspects and none of them consider improving both sides in an effective and practical way. To fill this gap, we propose Semantic-Preserving Adversarial Code Embeddings (SPACE) to find the worst-case semantic-preserving attacks while forcing the model to predict the correct labels under these worst cases. Experiments and analysis demonstrate that SPACE can stay robust against state-of-the-art attacks while boosting the performance of PrLMs for code.



## **16. Passive Triangulation Attack on ORide**

cs.CR

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2208.12216v2)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstracts**: Privacy preservation in Ride Hailing Services is intended to protect privacy of drivers and riders. ORide is one of the early RHS proposals published at USENIX Security Symposium 2017. In the ORide protocol, riders and drivers, operating in a zone, encrypt their locations using a Somewhat Homomorphic Encryption scheme (SHE) and forward them to the Service Provider (SP). SP homomorphically computes the squared Euclidean distance between riders and available drivers. Rider receives the encrypted distances and selects the optimal rider after decryption. In order to prevent a triangulation attack, SP randomly permutes the distances before sending them to the rider. In this work, we use propose a passive attack that uses triangulation to determine coordinates of all participating drivers whose permuted distances are available from the points of view of multiple honest-but-curious adversary riders. An attack on ORide was published at SAC 2021. The same paper proposes a countermeasure using noisy Euclidean distances to thwart their attack. We extend our attack to determine locations of drivers when given their permuted and noisy Euclidean distances from multiple points of reference, where the noise perturbation comes from a uniform distribution. We conduct experiments with different number of drivers and for different perturbation values. Our experiments show that we can determine locations of all drivers participating in the ORide protocol. For the perturbed distance version of the ORide protocol, our algorithm reveals locations of about 25% to 50% of participating drivers. Our algorithm runs in time polynomial in number of drivers.



## **17. CARE: Certifiably Robust Learning with Reasoning via Variational Inference**

cs.LG

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.05055v1)

**Authors**: Jiawei Zhang, Linyi Li, Ce Zhang, Bo Li

**Abstracts**: Despite great recent advances achieved by deep neural networks (DNNs), they are often vulnerable to adversarial attacks. Intensive research efforts have been made to improve the robustness of DNNs; however, most empirical defenses can be adaptively attacked again, and the theoretically certified robustness is limited, especially on large-scale datasets. One potential root cause of such vulnerabilities for DNNs is that although they have demonstrated powerful expressiveness, they lack the reasoning ability to make robust and reliable predictions. In this paper, we aim to integrate domain knowledge to enable robust learning with the reasoning paradigm. In particular, we propose a certifiably robust learning with reasoning pipeline (CARE), which consists of a learning component and a reasoning component. Concretely, we use a set of standard DNNs to serve as the learning component to make semantic predictions, and we leverage the probabilistic graphical models, such as Markov logic networks (MLN), to serve as the reasoning component to enable knowledge/logic reasoning. However, it is known that the exact inference of MLN (reasoning) is #P-complete, which limits the scalability of the pipeline. To this end, we propose to approximate the MLN inference via variational inference based on an efficient expectation maximization algorithm. In particular, we leverage graph convolutional networks (GCNs) to encode the posterior distribution during variational inference and update the parameters of GCNs (E-step) and the weights of knowledge rules in MLN (M-step) iteratively. We conduct extensive experiments on different datasets and show that CARE achieves significantly higher certified robustness compared with the state-of-the-art baselines. We additionally conducted different ablation studies to demonstrate the empirical robustness of CARE and the effectiveness of different knowledge integration.



## **18. GFCL: A GRU-based Federated Continual Learning Framework against Data Poisoning Attacks in IoV**

cs.LG

11 pages, 12 figures, 3 tables; This work has been submitted to the  IEEE Transactions on Vehicular Technology for possible publication. Copyright  may be transferred without notice, after which this version may no longer be  accessible

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2204.11010v2)

**Authors**: Anum Talpur, Mohan Gurusamy

**Abstracts**: Integration of machine learning (ML) in 5G-based Internet of Vehicles (IoV) networks has enabled intelligent transportation and smart traffic management. Nonetheless, the security against adversarial poisoning attacks is also increasingly becoming a challenging task. Specifically, Deep Reinforcement Learning (DRL) is one of the widely used ML designs in IoV applications. The standard ML security techniques are not effective in DRL where the algorithm learns to solve sequential decision-making through continuous interaction with the environment, and the environment is time-varying, dynamic, and mobile. In this paper, we propose a Gated Recurrent Unit (GRU)-based federated continual learning (GFCL) anomaly detection framework against Sybil-based data poisoning attacks in IoV. The objective is to present a lightweight and scalable framework that learns and detects the illegitimate behavior without having a-priori training dataset consisting of attack samples. We use GRU to predict a future data sequence to analyze and detect illegitimate behavior from vehicles in a federated learning-based distributed manner. We investigate the performance of our framework using real-world vehicle mobility traces. The results demonstrate the effectiveness of our proposed solution in terms of different performance metrics.



## **19. Generate novel and robust samples from data: accessible sharing without privacy concerns**

cs.LG

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.06113v1)

**Authors**: David Banh, Alan Huang

**Abstracts**: Generating new samples from data sets can mitigate extra expensive operations, increased invasive procedures, and mitigate privacy issues. These novel samples that are statistically robust can be used as a temporary and intermediate replacement when privacy is a concern. This method can enable better data sharing practices without problems relating to identification issues or biases that are flaws for an adversarial attack.



## **20. Resisting Deep Learning Models Against Adversarial Attack Transferability via Feature Randomization**

cs.CR

**SubmitDate**: 2022-09-11    [paper-pdf](http://arxiv.org/pdf/2209.04930v1)

**Authors**: Ehsan Nowroozi, Mohammadreza Mohammadi, Pargol Golmohammadi, Yassine Mekdad, Mauro Conti, Selcuk Uluagac

**Abstracts**: In the past decades, the rise of artificial intelligence has given us the capabilities to solve the most challenging problems in our day-to-day lives, such as cancer prediction and autonomous navigation. However, these applications might not be reliable if not secured against adversarial attacks. In addition, recent works demonstrated that some adversarial examples are transferable across different models. Therefore, it is crucial to avoid such transferability via robust models that resist adversarial manipulations. In this paper, we propose a feature randomization-based approach that resists eight adversarial attacks targeting deep learning models in the testing phase. Our novel approach consists of changing the training strategy in the target network classifier and selecting random feature samples. We consider the attacker with a Limited-Knowledge and Semi-Knowledge conditions to undertake the most prevalent types of adversarial attacks. We evaluate the robustness of our approach using the well-known UNSW-NB15 datasets that include realistic and synthetic attacks. Afterward, we demonstrate that our strategy outperforms the existing state-of-the-art approach, such as the Most Powerful Attack, which consists of fine-tuning the network model against specific adversarial attacks. Finally, our experimental results show that our methodology can secure the target network and resists adversarial attack transferability by over 60%.



## **21. Detecting Adversarial Perturbations in Multi-Task Perception**

cs.CV

Accepted at IROS 2022

**SubmitDate**: 2022-09-11    [paper-pdf](http://arxiv.org/pdf/2203.01177v2)

**Authors**: Marvin Klingner, Varun Ravi Kumar, Senthil Yogamani, Andreas Bär, Tim Fingscheidt

**Abstracts**: While deep neural networks (DNNs) achieve impressive performance on environment perception tasks, their sensitivity to adversarial perturbations limits their use in practical applications. In this paper, we (i) propose a novel adversarial perturbation detection scheme based on multi-task perception of complex vision tasks (i.e., depth estimation and semantic segmentation). Specifically, adversarial perturbations are detected by inconsistencies between extracted edges of the input image, the depth output, and the segmentation output. To further improve this technique, we (ii) develop a novel edge consistency loss between all three modalities, thereby improving their initial consistency which in turn supports our detection scheme. We verify our detection scheme's effectiveness by employing various known attacks and image noises. In addition, we (iii) develop a multi-task adversarial attack, aiming at fooling both tasks as well as our detection scheme. Experimental evaluation on the Cityscapes and KITTI datasets shows that under an assumption of a 5% false positive rate up to 100% of images are correctly detected as adversarially perturbed, depending on the strength of the perturbation. Code is available at https://github.com/ifnspaml/AdvAttackDet. A short video at https://youtu.be/KKa6gOyWmH4 provides qualitative results.



## **22. Robustness through Cognitive Dissociation Mitigation in Contrastive Adversarial Training**

cs.LG

Accepted to ICMLC 2022

**SubmitDate**: 2022-09-11    [paper-pdf](http://arxiv.org/pdf/2203.08959v3)

**Authors**: Adir Rahamim, Itay Naeh

**Abstracts**: In this paper, we introduce a novel neural network training framework that increases model's adversarial robustness to adversarial attacks while maintaining high clean accuracy by combining contrastive learning (CL) with adversarial training (AT). We propose to improve model robustness to adversarial attacks by learning feature representations that are consistent under both data augmentations and adversarial perturbations. We leverage contrastive learning to improve adversarial robustness by considering an adversarial example as another positive example, and aim to maximize the similarity between random augmentations of data samples and their adversarial example, while constantly updating the classification head in order to avoid a cognitive dissociation between the classification head and the embedding space. This dissociation is caused by the fact that CL updates the network up to the embedding space, while freezing the classification head which is used to generate new positive adversarial examples. We validate our method, Contrastive Learning with Adversarial Features(CLAF), on the CIFAR-10 dataset on which it outperforms both robust accuracy and clean accuracy over alternative supervised and self-supervised adversarial learning methods.



## **23. Scattering Model Guided Adversarial Examples for SAR Target Recognition: Attack and Defense**

cs.CV

**SubmitDate**: 2022-09-11    [paper-pdf](http://arxiv.org/pdf/2209.04779v1)

**Authors**: Bowen Peng, Bo Peng, Jie Zhou, Jianyue Xie, Li Liu

**Abstracts**: Deep Neural Networks (DNNs) based Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR) systems have shown to be highly vulnerable to adversarial perturbations that are deliberately designed yet almost imperceptible but can bias DNN inference when added to targeted objects. This leads to serious safety concerns when applying DNNs to high-stake SAR ATR applications. Therefore, enhancing the adversarial robustness of DNNs is essential for implementing DNNs to modern real-world SAR ATR systems. Toward building more robust DNN-based SAR ATR models, this article explores the domain knowledge of SAR imaging process and proposes a novel Scattering Model Guided Adversarial Attack (SMGAA) algorithm which can generate adversarial perturbations in the form of electromagnetic scattering response (called adversarial scatterers). The proposed SMGAA consists of two parts: 1) a parametric scattering model and corresponding imaging method and 2) a customized gradient-based optimization algorithm. First, we introduce the effective Attributed Scattering Center Model (ASCM) and a general imaging method to describe the scattering behavior of typical geometric structures in the SAR imaging process. By further devising several strategies to take the domain knowledge of SAR target images into account and relax the greedy search procedure, the proposed method does not need to be prudentially finetuned, but can efficiently to find the effective ASCM parameters to fool the SAR classifiers and facilitate the robust model training. Comprehensive evaluations on the MSTAR dataset show that the adversarial scatterers generated by SMGAA are more robust to perturbations and transformations in the SAR processing chain than the currently studied attacks, and are effective to construct a defensive model against the malicious scatterers.



## **24. Atomic cross-chain exchanges of shared assets**

cs.CR

**SubmitDate**: 2022-09-10    [paper-pdf](http://arxiv.org/pdf/2202.12855v3)

**Authors**: Krishnasuri Narayanam, Venkatraman Ramakrishna, Dhinakaran Vinayagamurthy, Sandeep Nishad

**Abstracts**: A core enabler for blockchain or DLT interoperability is the ability to atomically exchange assets held by mutually untrusting owners on different ledgers. This atomic swap problem has been well-studied, with the Hash Time Locked Contract (HTLC) emerging as a canonical solution. HTLC ensures atomicity of exchange, albeit with caveats for node failure and timeliness of claims. But a bigger limitation of HTLC is that it only applies to a model consisting of two adversarial parties having sole ownership of a single asset in each ledger. Realistic extensions of the model in which assets may be jointly owned by multiple parties, all of whose consents are required for exchanges, or where multiple assets must be exchanged for one, are susceptible to collusion attacks and hence cannot be handled by HTLC. In this paper, we generalize the model of asset exchanges across DLT networks and present a taxonomy of use cases, describe the threat model, and propose MPHTLC, an augmented HTLC protocol for atomic multi-owner-and-asset exchanges. We analyze the correctness, safety, and application scope of MPHTLC. As proof-of-concept, we show how MPHTLC primitives can be implemented in networks built on Hyperledger Fabric and Corda, and how MPHTLC can be implemented in the Hyperledger Labs Weaver framework by augmenting its existing HTLC protocol.



## **25. Phantom Sponges: Exploiting Non-Maximum Suppression to Attack Deep Object Detectors**

cs.CV

**SubmitDate**: 2022-09-10    [paper-pdf](http://arxiv.org/pdf/2205.13618v2)

**Authors**: Avishag Shapira, Alon Zolfi, Luca Demetrio, Battista Biggio, Asaf Shabtai

**Abstracts**: Adversarial attacks against deep learning-based object detectors have been studied extensively in the past few years. Most of the attacks proposed have targeted the model's integrity (i.e., caused the model to make incorrect predictions), while adversarial attacks targeting the model's availability, a critical aspect in safety-critical domains such as autonomous driving, have not yet been explored by the machine learning research community. In this paper, we propose a novel attack that negatively affects the decision latency of an end-to-end object detection pipeline. We craft a universal adversarial perturbation (UAP) that targets a widely used technique integrated in many object detector pipelines -- non-maximum suppression (NMS). Our experiments demonstrate the proposed UAP's ability to increase the processing time of individual frames by adding "phantom" objects that overload the NMS algorithm while preserving the detection of the original objects (which allows the attack to go undetected for a longer period of time).



## **26. Bankrupting DoS Attackers Despite Uncertainty**

cs.CR

**SubmitDate**: 2022-09-10    [paper-pdf](http://arxiv.org/pdf/2205.08287v2)

**Authors**: Trisha Chakraborty, Abir Islam, Valerie King, Daniel Rayborn, Jared Saia, Maxwell Young

**Abstracts**: On-demand provisioning in the cloud allows for services to remain available despite massive denial-of-service (DoS) attacks. Unfortunately, on-demand provisioning is expensive and must be weighed against the costs incurred by an adversary. This leads to a recent threat known as {\it economic denial-of-sustainability (EDoS)}, where the cost for defending a service is higher than that of attacking.   A natural tool for combating EDoS is to impose costs via resource burning (RB). Here, a client must verifiably consume resources -- for example, by solving a computational challenge -- before service is rendered. However, prior RB-based defenses with security guarantees do not account for the cost of on-demand provisioning.   Another common approach is the use of heuristics -- such as a client's reputation score or the geographical location -- to identify and discard spurious job requests. However, these heuristics may err and existing approaches do not provide security guarantees when this occurs.   Here, we propose an EDoS defense, LCharge, that uses resource burning while accounting for on-demand provisioning. LCharge leverages an estimate of the number of job requests from honest clients (i.e., good jobs) in any set $S$ of requests to within an $O(\alpha)$-factor, for any unknown $\alpha>0$, but retains a strong security guarantee despite the uncertainty of this estimate. Specifically, against an adversary that expends $B$ resources to attack, the total cost for defending is $O( \alpha^{5/2}\sqrt{B\,(g+1)} + \alpha^3(g+\alpha))$ where $g$ is the number of good jobs. Notably, for large $B$ relative to $g$ and $\alpha$, the adversary has higher cost, implying that the algorithm has an economic advantage. Finally, we prove a lower bound for our problem of $\Omega(\sqrt{\alpha B g})$, showing that the cost of LCharge is asymptotically tight for $\alpha=\Theta(1)$.



## **27. The Space of Adversarial Strategies**

cs.CR

Accepted to the 32nd USENIX Security Symposium

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2209.04521v1)

**Authors**: Ryan Sheatsley, Blaine Hoak, Eric Pauley, Patrick McDaniel

**Abstracts**: Adversarial examples, inputs designed to induce worst-case behavior in machine learning models, have been extensively studied over the past decade. Yet, our understanding of this phenomenon stems from a rather fragmented pool of knowledge; at present, there are a handful of attacks, each with disparate assumptions in threat models and incomparable definitions of optimality. In this paper, we propose a systematic approach to characterize worst-case (i.e., optimal) adversaries. We first introduce an extensible decomposition of attacks in adversarial machine learning by atomizing attack components into surfaces and travelers. With our decomposition, we enumerate over components to create 576 attacks (568 of which were previously unexplored). Next, we propose the Pareto Ensemble Attack (PEA): a theoretical attack that upper-bounds attack performance. With our new attacks, we measure performance relative to the PEA on: both robust and non-robust models, seven datasets, and three extended lp-based threat models incorporating compute costs, formalizing the Space of Adversarial Strategies. From our evaluation we find that attack performance to be highly contextual: the domain, model robustness, and threat model can have a profound influence on attack efficacy. Our investigation suggests that future studies measuring the security of machine learning should: (1) be contextualized to the domain & threat models, and (2) go beyond the handful of known attacks used today.



## **28. SoK: Certified Robustness for Deep Neural Networks**

cs.LG

To appear at 2023 IEEE Symposium on Security and Privacy (SP); 14  pages for the main text; benchmark & tool website:  http://sokcertifiedrobustness.github.io/

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2009.04131v8)

**Authors**: Linyi Li, Tao Xie, Bo Li

**Abstracts**: Great advances in deep neural networks (DNNs) have led to state-of-the-art performance on a wide range of tasks. However, recent studies have shown that DNNs are vulnerable to adversarial attacks, which have brought great concerns when deploying these models to safety-critical applications such as autonomous driving. Different defense approaches have been proposed against adversarial attacks, including: a) empirical defenses, which can usually be adaptively attacked again without providing robustness certification; and b) certifiably robust approaches, which consist of robustness verification providing the lower bound of robust accuracy against any attacks under certain conditions and corresponding robust training approaches. In this paper, we systematize certifiably robust approaches and related practical and theoretical implications and findings. We also provide the first comprehensive benchmark on existing robustness verification and training approaches on different datasets. In particular, we 1) provide a taxonomy for the robustness verification and training approaches, as well as summarize the methodologies for representative algorithms, 2) reveal the characteristics, strengths, limitations, and fundamental connections among these approaches, 3) discuss current research progresses, theoretical barriers, main challenges, and future directions for certifiably robust approaches for DNNs, and 4) provide an open-sourced unified platform to evaluate 20+ representative certifiably robust approaches.



## **29. Adversarial Examples in Constrained Domains**

cs.CR

Accepted to IOS Press Journal of Computer Security

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2011.01183v3)

**Authors**: Ryan Sheatsley, Nicolas Papernot, Michael Weisman, Gunjan Verma, Patrick McDaniel

**Abstracts**: Machine learning algorithms have been shown to be vulnerable to adversarial manipulation through systematic modification of inputs (e.g., adversarial examples) in domains such as image recognition. Under the default threat model, the adversary exploits the unconstrained nature of images; each feature (pixel) is fully under control of the adversary. However, it is not clear how these attacks translate to constrained domains that limit which and how features can be modified by the adversary (e.g., network intrusion detection). In this paper, we explore whether constrained domains are less vulnerable than unconstrained domains to adversarial example generation algorithms. We create an algorithm for generating adversarial sketches: targeted universal perturbation vectors which encode feature saliency within the envelope of domain constraints. To assess how these algorithms perform, we evaluate them in constrained (e.g., network intrusion detection) and unconstrained (e.g., image recognition) domains. The results demonstrate that our approaches generate misclassification rates in constrained domains that were comparable to those of unconstrained domains (greater than 95%). Our investigation shows that the narrow attack surface exposed by constrained domains is still sufficiently large to craft successful adversarial examples; and thus, constraints do not appear to make a domain robust. Indeed, with as little as five randomly selected features, one can still generate adversarial examples.



## **30. Robust-by-Design Classification via Unitary-Gradient Neural Networks**

cs.LG

Under review

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2209.04293v1)

**Authors**: Fabio Brau, Giulio Rossolini, Alessandro Biondi, Giorgio Buttazzo

**Abstracts**: The use of neural networks in safety-critical systems requires safe and robust models, due to the existence of adversarial attacks. Knowing the minimal adversarial perturbation of any input x, or, equivalently, knowing the distance of x from the classification boundary, allows evaluating the classification robustness, providing certifiable predictions. Unfortunately, state-of-the-art techniques for computing such a distance are computationally expensive and hence not suited for online applications. This work proposes a novel family of classifiers, namely Signed Distance Classifiers (SDCs), that, from a theoretical perspective, directly output the exact distance of x from the classification boundary, rather than a probability score (e.g., SoftMax). SDCs represent a family of robust-by-design classifiers. To practically address the theoretical requirements of a SDC, a novel network architecture named Unitary-Gradient Neural Network is presented. Experimental results show that the proposed architecture approximates a signed distance classifier, hence allowing an online certifiable classification of x at the cost of a single inference.



## **31. Improving Out-of-Distribution Detection via Epistemic Uncertainty Adversarial Training**

cs.LG

8 pages, 5 figures

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2209.03148v2)

**Authors**: Derek Everett, Andre T. Nguyen, Luke E. Richards, Edward Raff

**Abstracts**: The quantification of uncertainty is important for the adoption of machine learning, especially to reject out-of-distribution (OOD) data back to human experts for review. Yet progress has been slow, as a balance must be struck between computational efficiency and the quality of uncertainty estimates. For this reason many use deep ensembles of neural networks or Monte Carlo dropout for reasonable uncertainty estimates at relatively minimal compute and memory. Surprisingly, when we focus on the real-world applicable constraint of $\leq 1\%$ false positive rate (FPR), prior methods fail to reliably detect OOD samples as such. Notably, even Gaussian random noise fails to trigger these popular OOD techniques. We help to alleviate this problem by devising a simple adversarial training scheme that incorporates an attack of the epistemic uncertainty predicted by the dropout ensemble. We demonstrate this method improves OOD detection performance on standard data (i.e., not adversarially crafted), and improves the standardized partial AUC from near-random guessing performance to $\geq 0.75$.



## **32. Harnessing Perceptual Adversarial Patches for Crowd Counting**

cs.CV

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2109.07986v2)

**Authors**: Shunchang Liu, Jiakai Wang, Aishan Liu, Yingwei Li, Yijie Gao, Xianglong Liu, Dacheng Tao

**Abstracts**: Crowd counting, which has been widely adopted for estimating the number of people in safety-critical scenes, is shown to be vulnerable to adversarial examples in the physical world (e.g., adversarial patches). Though harmful, adversarial examples are also valuable for evaluating and better understanding model robustness. However, existing adversarial example generation methods for crowd counting lack strong transferability among different black-box models, which limits their practicability for real-world systems. Motivated by the fact that attacking transferability is positively correlated to the model-invariant characteristics, this paper proposes the Perceptual Adversarial Patch (PAP) generation framework to tailor the adversarial perturbations for crowd counting scenes using the model-shared perceptual features. Specifically, we handcraft an adaptive crowd density weighting approach to capture the invariant scale perception features across various models and utilize the density guided attention to capture the model-shared position perception. Both of them are demonstrated to improve the attacking transferability of our adversarial patches. Extensive experiments show that our PAP could achieve state-of-the-art attacking performance in both the digital and physical world, and outperform previous proposals by large margins (at most +685.7 MAE and +699.5 MSE). Besides, we empirically demonstrate that adversarial training with our PAP can benefit the performance of vanilla models in alleviating several practical challenges in crowd counting scenarios, including generalization across datasets (up to -376.0 MAE and -354.9 MSE) and robustness towards complex backgrounds (up to -10.3 MAE and -16.4 MSE).



## **33. Uncovering the Connection Between Differential Privacy and Certified Robustness of Federated Learning against Poisoning Attacks**

cs.CR

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2209.04030v1)

**Authors**: Chulin Xie, Yunhui Long, Pin-Yu Chen, Bo Li

**Abstracts**: Federated learning (FL) provides an efficient paradigm to jointly train a global model leveraging data from distributed users. As the local training data come from different users who may not be trustworthy, several studies have shown that FL is vulnerable to poisoning attacks. Meanwhile, to protect the privacy of local users, FL is always trained in a differentially private way (DPFL). Thus, in this paper, we ask: Can we leverage the innate privacy property of DPFL to provide certified robustness against poisoning attacks? Can we further improve the privacy of FL to improve such certification? We first investigate both user-level and instance-level privacy of FL and propose novel mechanisms to achieve improved instance-level privacy. We then provide two robustness certification criteria: certified prediction and certified attack cost for DPFL on both levels. Theoretically, we prove the certified robustness of DPFL under a bounded number of adversarial users or instances. Empirically, we conduct extensive experiments to verify our theories under a range of attacks on different datasets. We show that DPFL with a tighter privacy guarantee always provides stronger robustness certification in terms of certified attack cost, but the optimal certified prediction is achieved under a proper balance between privacy protection and utility loss.



## **34. Evaluating the Security of Aircraft Systems**

cs.CR

38 pages,

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2209.04028v1)

**Authors**: Edan Habler, Ron Bitton, Asaf Shabtai

**Abstracts**: The sophistication and complexity of cyber attacks and the variety of targeted platforms have been growing in recent years. Various adversaries are abusing an increasing range of platforms, e.g., enterprise platforms, mobile phones, PCs, transportation systems, and industrial control systems. In recent years, we have witnessed various cyber attacks on transportation systems, including attacks on ports, airports, and trains. It is only a matter of time before transportation systems become a more common target of cyber attackers. Due to the enormous potential damage inherent in attacking vehicles carrying many passengers and the lack of security measures applied in traditional airborne systems, the vulnerability of aircraft systems is one of the most concerning topics in the vehicle security domain. This paper provides a comprehensive review of aircraft systems and components and their various networks, emphasizing the cyber threats they are exposed to and the impact of a cyber attack on these components and networks and the essential capabilities of the aircraft. In addition, we present a comprehensive and in-depth taxonomy that standardizes the knowledge and understanding of cyber security in the avionics field from an adversary's perspective. The taxonomy divides techniques into relevant categories (tactics) reflecting the various phases of the adversarial attack lifecycle and maps existing attacks according to the MITRE ATT&CK methodology. Furthermore, we analyze the security risks among the various systems according to the potential threat actors and categorize the threats based on STRIDE threat model. Future work directions are presented as guidelines for industry and academia.



## **35. SafeNet: The Unreasonable Effectiveness of Ensembles in Private Collaborative Learning**

cs.CR

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2205.09986v2)

**Authors**: Harsh Chaudhari, Matthew Jagielski, Alina Oprea

**Abstracts**: Secure multiparty computation (MPC) has been proposed to allow multiple mutually distrustful data owners to jointly train machine learning (ML) models on their combined data. However, by design, MPC protocols faithfully compute the training functionality, which the adversarial ML community has shown to leak private information and can be tampered with in poisoning attacks. In this work, we argue that model ensembles, implemented in our framework called SafeNet, are a highly MPC-amenable way to avoid many adversarial ML attacks. The natural partitioning of data amongst owners in MPC training allows this approach to be highly scalable at training time, provide provable protection from poisoning attacks, and provably defense against a number of privacy attacks. We demonstrate SafeNet's efficiency, accuracy, and resilience to poisoning on several machine learning datasets and models trained in end-to-end and transfer learning scenarios. For instance, SafeNet reduces backdoor attack success significantly, while achieving $39\times$ faster training and $36 \times$ less communication than the four-party MPC framework of Dalskov et al. Our experiments show that ensembling retains these benefits even in many non-iid settings. The simplicity, cheap setup, and robustness properties of ensembling make it a strong first choice for training ML models privately in MPC.



## **36. Incorporating Locality of Images to Generate Targeted Transferable Adversarial Examples**

cs.CV

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2209.03716v1)

**Authors**: Zhipeng Wei, Jingjing Chen, Zuxuan Wu, Yu-Gang Jiang

**Abstracts**: Despite that leveraging the transferability of adversarial examples can attain a fairly high attack success rate for non-targeted attacks, it does not work well in targeted attacks since the gradient directions from a source image to a targeted class are usually different in different DNNs. To increase the transferability of target attacks, recent studies make efforts in aligning the feature of the generated adversarial example with the feature distributions of the targeted class learned from an auxiliary network or a generative adversarial network. However, these works assume that the training dataset is available and require a lot of time to train networks, which makes it hard to apply to real-world scenarios. In this paper, we revisit adversarial examples with targeted transferability from the perspective of universality and find that highly universal adversarial perturbations tend to be more transferable. Based on this observation, we propose the Locality of Images (LI) attack to improve targeted transferability. Specifically, instead of using the classification loss only, LI introduces a feature similarity loss between intermediate features from adversarial perturbed original images and randomly cropped images, which makes the features from adversarial perturbations to be more dominant than that of benign images, hence improving targeted transferability. Through incorporating locality of images into optimizing perturbations, the LI attack emphasizes that targeted perturbations should be universal to diverse input patterns, even local image patches. Extensive experiments demonstrate that LI can achieve high success rates for transfer-based targeted attacks. On attacking the ImageNet-compatible dataset, LI yields an improvement of 12\% compared with existing state-of-the-art methods.



## **37. Exploring Adversarial Attacks and Defenses in Vision Transformers trained with DINO**

cs.CV

ICML 2022 Workshop paper accepted at AdvML Frontiers

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2206.06761v4)

**Authors**: Javier Rando, Nasib Naimi, Thomas Baumann, Max Mathys

**Abstracts**: This work conducts the first analysis on the robustness against adversarial attacks on self-supervised Vision Transformers trained using DINO. First, we evaluate whether features learned through self-supervision are more robust to adversarial attacks than those emerging from supervised learning. Then, we present properties arising for attacks in the latent space. Finally, we evaluate whether three well-known defense strategies can increase adversarial robustness in downstream tasks by only fine-tuning the classification head to provide robustness even in view of limited compute resources. These defense strategies are: Adversarial Training, Ensemble Adversarial Training and Ensemble of Specialized Networks.



## **38. Feature Importance Guided Attack: A Model Agnostic Adversarial Attack**

cs.LG

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2106.14815v2)

**Authors**: Gilad Gressel, Niranjan Hegde, Archana Sreekumar, Rishikumar Radhakrishnan, Kalyani Harikumar, Anjali S., Michael Darling

**Abstracts**: Research in adversarial learning has primarily focused on homogeneous unstructured datasets, which often map into the problem space naturally. Inverting a feature space attack on heterogeneous datasets into the problem space is much more challenging, particularly the task of finding the perturbation to perform. This work presents a formal search strategy: the `Feature Importance Guided Attack' (FIGA), which finds perturbations in the feature space of heterogeneous tabular datasets to produce evasion attacks. We first demonstrate FIGA in the feature space and then in the problem space. FIGA assumes no prior knowledge of the defending model's learning algorithm and does not require any gradient information. FIGA assumes knowledge of the feature representation and the mean feature values of defending model's dataset. FIGA leverages feature importance rankings by perturbing the most important features of the input in the direction of the target class. While FIGA is conceptually similar to other work which uses feature selection processes (e.g., mimicry attacks), we formalize an attack algorithm with three tunable parameters and investigate the strength of FIGA on tabular datasets. We demonstrate the effectiveness of FIGA by evading phishing detection models trained on four different tabular phishing datasets and one financial dataset with an average success rate of 94%. We extend FIGA to the phishing problem space by limiting the possible perturbations to be valid and feasible in the phishing domain. We generate valid adversarial phishing sites that are visually identical to their unperturbed counterpart and use them to attack six tabular ML models achieving a 13.05% average success rate.



## **39. AdaptOver: Adaptive Overshadowing Attacks in Cellular Networks**

cs.CR

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2106.05039v3)

**Authors**: Simon Erni, Martin Kotuliak, Patrick Leu, Marc Roeschlin, Srdjan Capkun

**Abstracts**: In cellular networks, attacks on the communication link between a mobile device and the core network significantly impact privacy and availability. Up until now, fake base stations have been required to execute such attacks. Since they require a continuously high output power to attract victims, they are limited in range and can be easily detected both by operators and dedicated apps on users' smartphones.   This paper introduces AdaptOver - a MITM attack system designed for cellular networks, specifically for LTE and 5G-NSA. AdaptOver allows an adversary to decode, overshadow (replace) and inject arbitrary messages over the air in either direction between the network and the mobile device. Using overshadowing, AdaptOver can cause a persistent ($\geq$ 12h) DoS or a privacy leak by triggering a UE to transmit its persistent identifier (IMSI) in plain text. These attacks can be launched against all users within a cell or specifically target a victim based on its phone number.   We implement AdaptOver using a software-defined radio and a low-cost amplification setup. We demonstrate the effects and practicality of the attacks on a live operational LTE and 5G-NSA network with a wide range of smartphones. Our experiments show that AdaptOver can launch an attack on a victim more than 3.8km away from the attacker. Given its practicability and efficiency, AdaptOver shows that existing countermeasures that are focused on fake base stations are no longer sufficient, marking a paradigm shift for designing security mechanisms in cellular networks.



## **40. Combing for Credentials: Active Pattern Extraction from Smart Reply**

cs.CR

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2207.10802v2)

**Authors**: Bargav Jayaraman, Esha Ghosh, Melissa Chase, Sambuddha Roy, Huseyin Inan, Wei Dai, David Evans

**Abstracts**: With the wide availability of large pre-trained language models such as GPT-2 and BERT, the recent trend has been to fine-tune a pre-trained model to achieve state-of-the-art performance on a downstream task. One natural example is the "Smart Reply" application where a pre-trained model is tuned to provide suggested responses for a given query message. Since these models are often tuned using sensitive data such as emails or chat transcripts, it is important to understand and mitigate the risk that the model leaks its tuning data. We investigate potential information leakage vulnerabilities in a typical Smart Reply pipeline and introduce a new type of active extraction attack that exploits canonical patterns in text containing sensitive data. We show experimentally that it is possible for an adversary to extract sensitive user information present in the training data. We explore potential mitigation strategies and demonstrate empirically how differential privacy appears to be an effective defense mechanism to such pattern extraction attacks.



## **41. Inferring Sensitive Attributes from Model Explanations**

cs.CR

ACM CIKM 2022

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2208.09967v2)

**Authors**: Vasisht Duddu, Antoine Boutet

**Abstracts**: Model explanations provide transparency into a trained machine learning model's blackbox behavior to a model builder. They indicate the influence of different input attributes to its corresponding model prediction. The dependency of explanations on input raises privacy concerns for sensitive user data. However, current literature has limited discussion on privacy risks of model explanations.   We focus on the specific privacy risk of attribute inference attack wherein an adversary infers sensitive attributes of an input (e.g., race and sex) given its model explanations. We design the first attribute inference attack against model explanations in two threat models where model builder either (a) includes the sensitive attributes in training data and input or (b) censors the sensitive attributes by not including them in the training data and input.   We evaluate our proposed attack on four benchmark datasets and four state-of-the-art algorithms. We show that an adversary can successfully infer the value of sensitive attributes from explanations in both the threat models accurately. Moreover, the attack is successful even by exploiting only the explanations corresponding to sensitive attributes. These suggest that our attack is effective against explanations and poses a practical threat to data privacy.   On combining the model predictions (an attack surface exploited by prior attacks) with explanations, we note that the attack success does not improve. Additionally, the attack success on exploiting model explanations is better compared to exploiting only model predictions. These suggest that model explanations are a strong attack surface to exploit for an adversary.



## **42. Securing the Spike: On the Transferabilty and Security of Spiking Neural Networks to Adversarial Examples**

cs.NE

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2209.03358v1)

**Authors**: Nuo Xu, Kaleel Mahmood, Haowen Fang, Ethan Rathbun, Caiwen Ding, Wujie Wen

**Abstracts**: Spiking neural networks (SNNs) have attracted much attention for their high energy efficiency and for recent advances in their classification performance. However, unlike traditional deep learning approaches, the analysis and study of the robustness of SNNs to adversarial examples remains relatively underdeveloped. In this work we advance the field of adversarial machine learning through experimentation and analyses of three important SNN security attributes. First, we show that successful white-box adversarial attacks on SNNs are highly dependent on the underlying surrogate gradient technique. Second, we analyze the transferability of adversarial examples generated by SNNs and other state-of-the-art architectures like Vision Transformers and Big Transfer CNNs. We demonstrate that SNNs are not often deceived by adversarial examples generated by Vision Transformers and certain types of CNNs. Lastly, we develop a novel white-box attack that generates adversarial examples capable of fooling both SNN models and non-SNN models simultaneously. Our experiments and analyses are broad and rigorous covering two datasets (CIFAR-10 and CIFAR-100), five different white-box attacks and twelve different classifier models.



## **43. Minotaur: Multi-Resource Blockchain Consensus**

cs.CR

To appear in ACM CCS 2022

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2201.11780v2)

**Authors**: Matthias Fitzi, Xuechao Wang, Sreeram Kannan, Aggelos Kiayias, Nikos Leonardos, Pramod Viswanath, Gerui Wang

**Abstracts**: Resource-based consensus is the backbone of permissionless distributed ledger systems. The security of such protocols relies fundamentally on the level of resources actively engaged in the system. The variety of different resources (and related proof protocols, some times referred to as PoX in the literature) raises the fundamental question whether it is possible to utilize many of them in tandem and build multi-resource consensus protocols. The challenge in combining different resources is to achieve fungibility between them, in the sense that security would hold as long as the cumulative adversarial power across all resources is bounded.   In this work, we put forth Minotaur, a multi-resource blockchain consensus protocol that combines proof-of-work (PoW) and proof-of-stake (PoS), and we prove it optimally fungible. At the core of our design, Minotaur operates in epochs while continuously sampling the active computational power to provide a fair exchange between the two resources, work and stake. Further, we demonstrate the ability of Minotaur to handle a higher degree of work fluctuation as compared to the Bitcoin blockchain; we also generalize Minotaur to any number of resources.   We demonstrate the simplicity of Minotaur via implementing a full stack client in Rust (available open source). We use the client to test the robustness of Minotaur to variable mining power and combined work/stake attacks and demonstrate concrete empirical evidence towards the suitability of Minotaur to serve as the consensus layer of a real-world blockchain.



## **44. Distributed Adversarial Training to Robustify Deep Neural Networks at Scale**

cs.LG

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2206.06257v2)

**Authors**: Gaoyuan Zhang, Songtao Lu, Yihua Zhang, Xiangyi Chen, Pin-Yu Chen, Quanfu Fan, Lee Martie, Lior Horesh, Mingyi Hong, Sijia Liu

**Abstracts**: Current deep neural networks (DNNs) are vulnerable to adversarial attacks, where adversarial perturbations to the inputs can change or manipulate classification. To defend against such attacks, an effective and popular approach, known as adversarial training (AT), has been shown to mitigate the negative impact of adversarial attacks by virtue of a min-max robust training method. While effective, it remains unclear whether it can successfully be adapted to the distributed learning context. The power of distributed optimization over multiple machines enables us to scale up robust training over large models and datasets. Spurred by that, we propose distributed adversarial training (DAT), a large-batch adversarial training framework implemented over multiple machines. We show that DAT is general, which supports training over labeled and unlabeled data, multiple types of attack generation methods, and gradient compression operations favored for distributed optimization. Theoretically, we provide, under standard conditions in the optimization theory, the convergence rate of DAT to the first-order stationary points in general non-convex settings. Empirically, we demonstrate that DAT either matches or outperforms state-of-the-art robust accuracies and achieves a graceful training speedup (e.g., on ResNet-50 under ImageNet). Codes are available at https://github.com/dat-2022/dat.



## **45. Privacy Against Inference Attacks in Vertical Federated Learning**

cs.LG

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2207.11788v3)

**Authors**: Borzoo Rassouli, Morteza Varasteh, Deniz Gunduz

**Abstracts**: Vertical federated learning is considered, where an active party, having access to true class labels, wishes to build a classification model by utilizing more features from a passive party, which has no access to the labels, to improve the model accuracy. In the prediction phase, with logistic regression as the classification model, several inference attack techniques are proposed that the adversary, i.e., the active party, can employ to reconstruct the passive party's features, regarded as sensitive information. These attacks, which are mainly based on a classical notion of the center of a set, i.e., the Chebyshev center, are shown to be superior to those proposed in the literature. Moreover, several theoretical performance guarantees are provided for the aforementioned attacks. Subsequently, we consider the minimum amount of information that the adversary needs to fully reconstruct the passive party's features. In particular, it is shown that when the passive party holds one feature, and the adversary is only aware of the signs of the parameters involved, it can perfectly reconstruct that feature when the number of predictions is large enough. Next, as a defense mechanism, a privacy-preserving scheme is proposed that worsen the adversary's reconstruction attacks, while preserving the full benefits that VFL brings to the active party. Finally, experimental results demonstrate the effectiveness of the proposed attacks and the privacy-preserving scheme.



## **46. Fact-Saboteurs: A Taxonomy of Evidence Manipulation Attacks against Fact-Verification Systems**

cs.CR

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2209.03755v1)

**Authors**: Sahar Abdelnabi, Mario Fritz

**Abstracts**: Mis- and disinformation are now a substantial global threat to our security and safety. To cope with the scale of online misinformation, one viable solution is to automate the fact-checking of claims by retrieving and verifying against relevant evidence. While major recent advances have been achieved in pushing forward the automatic fact-verification, a comprehensive evaluation of the possible attack vectors against such systems is still lacking. Particularly, the automated fact-verification process might be vulnerable to the exact disinformation campaigns it is trying to combat. In this work, we assume an adversary that automatically tampers with the online evidence in order to disrupt the fact-checking model via camouflaging the relevant evidence, or planting a misleading one. We first propose an exploratory taxonomy that spans these two targets and the different threat model dimensions. Guided by this, we design and propose several potential attack methods. We show that it is possible to subtly modify claim-salient snippets in the evidence, in addition to generating diverse and claim-aligned evidence. As a result, we highly degrade the fact-checking performance under many different permutations of the taxonomy's dimensions. The attacks are also robust against post-hoc modifications of the claim. Our analysis further hints at potential limitations in models' inference when faced with contradicting evidence. We emphasize that these attacks can have harmful implications on the inspectable and human-in-the-loop usage scenarios of such models, and we conclude by discussing challenges and directions for future defenses.



## **47. State of Security Awareness in the AM Industry: 2020 Survey**

cs.CR

The material was presented at ASTM ICAM 2021 and a publication was  accepted for publication as a Selected Technical Papers (STP)

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2209.03073v1)

**Authors**: Mark Yampolskiy, Paul Bates, Mohsen Seifi, Nima Shamsaei

**Abstracts**: Security of Additive Manufacturing (AM) gets increased attention due to the growing proliferation and adoption of AM in a variety of applications and business models. However, there is a significant disconnect between AM community focused on manufacturing and AM Security community focused on securing this highly computerized manufacturing technology. To bridge this gap, we surveyed the America Makes AM community, asking in total eleven AM security-related questions aiming to discover the existing concerns, posture, and expectations. The first set of questions aimed to discover how many of these organizations use AM, outsource AM, or provide AM as a service. Then we asked about biggest security concerns as well as about assessment of who the potential adversaries might be and their motivation for attack. We then proceeded with questions on any experienced security incidents, if any security risk assessment was conducted, and if the participants' organizations were partnering with external experts to secure AM. Lastly, we asked whether security measures are implemented at all and, if yes, whether they fall under the general cyber-security category. Out of 69 participants affiliated with commercial industry, agencies, and academia, 53 have completed the entire survey. This paper presents the results of this survey, as well as provides our assessment of the AM Security posture. The answers are a mixture of what we could label as expected, "shocking but not surprising," and completely unexpected. Assuming that the provided answers are somewhat representative to the current state of the AM industry, we conclude that the industry is not ready to prevent or detect AM-specific attacks that have been demonstrated in the research literature.



## **48. On the Transferability of Adversarial Examples between Encrypted Models**

cs.CV

to be appear in ISPACS 2022

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2209.02997v1)

**Authors**: Miki Tanaka, Isao Echizen, Hitoshi Kiya

**Abstracts**: Deep neural networks (DNNs) are well known to be vulnerable to adversarial examples (AEs). In addition, AEs have adversarial transferability, namely, AEs generated for a source model fool other (target) models. In this paper, we investigate the transferability of models encrypted for adversarially robust defense for the first time. To objectively verify the property of transferability, the robustness of models is evaluated by using a benchmark attack method, called AutoAttack. In an image-classification experiment, the use of encrypted models is confirmed not only to be robust against AEs but to also reduce the influence of AEs in terms of the transferability of models.



## **49. Adversarial Mask: Real-World Universal Adversarial Attack on Face Recognition Model**

cs.CV

16 pages, 9 figures

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2111.10759v3)

**Authors**: Alon Zolfi, Shai Avidan, Yuval Elovici, Asaf Shabtai

**Abstracts**: Deep learning-based facial recognition (FR) models have demonstrated state-of-the-art performance in the past few years, even when wearing protective medical face masks became commonplace during the COVID-19 pandemic. Given the outstanding performance of these models, the machine learning research community has shown increasing interest in challenging their robustness. Initially, researchers presented adversarial attacks in the digital domain, and later the attacks were transferred to the physical domain. However, in many cases, attacks in the physical domain are conspicuous, and thus may raise suspicion in real-world environments (e.g., airports). In this paper, we propose Adversarial Mask, a physical universal adversarial perturbation (UAP) against state-of-the-art FR models that is applied on face masks in the form of a carefully crafted pattern. In our experiments, we examined the transferability of our adversarial mask to a wide range of FR model architectures and datasets. In addition, we validated our adversarial mask's effectiveness in real-world experiments (CCTV use case) by printing the adversarial pattern on a fabric face mask. In these experiments, the FR system was only able to identify 3.34% of the participants wearing the mask (compared to a minimum of 83.34% with other evaluated masks). A demo of our experiments can be found at: https://youtu.be/_TXkDO5z11w.



## **50. Facial De-morphing: Extracting Component Faces from a Single Morph**

cs.CV

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2209.02933v1)

**Authors**: Sudipta Banerjee, Prateek Jaiswal, Arun Ross

**Abstracts**: A face morph is created by strategically combining two or more face images corresponding to multiple identities. The intention is for the morphed image to match with multiple identities. Current morph attack detection strategies can detect morphs but cannot recover the images or identities used in creating them. The task of deducing the individual face images from a morphed face image is known as \textit{de-morphing}. Existing work in de-morphing assume the availability of a reference image pertaining to one identity in order to recover the image of the accomplice - i.e., the other identity. In this work, we propose a novel de-morphing method that can recover images of both identities simultaneously from a single morphed face image without needing a reference image or prior information about the morphing process. We propose a generative adversarial network that achieves single image-based de-morphing with a surprisingly high degree of visual realism and biometric similarity with the original face images. We demonstrate the performance of our method on landmark-based morphs and generative model-based morphs with promising results.



