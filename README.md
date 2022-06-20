# Latest Adversarial Attack Papers
**update at 2022-06-21 06:31:27**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Is Multi-Modal Necessarily Better? Robustness Evaluation of Multi-modal Fake News Detection**

cs.AI

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2206.08788v1)

**Authors**: Jinyin Chen, Chengyu Jia, Haibin Zheng, Ruoxi Chen, Chenbo Fu

**Abstracts**: The proliferation of fake news and its serious negative social influence push fake news detection methods to become necessary tools for web managers. Meanwhile, the multi-media nature of social media makes multi-modal fake news detection popular for its ability to capture more modal features than uni-modal detection methods. However, current literature on multi-modal detection is more likely to pursue the detection accuracy but ignore the robustness of the detector. To address this problem, we propose a comprehensive robustness evaluation of multi-modal fake news detectors. In this work, we simulate the attack methods of malicious users and developers, i.e., posting fake news and injecting backdoors. Specifically, we evaluate multi-modal detectors with five adversarial and two backdoor attack methods. Experiment results imply that: (1) The detection performance of the state-of-the-art detectors degrades significantly under adversarial attacks, even worse than general detectors; (2) Most multi-modal detectors are more vulnerable when subjected to attacks on visual modality than textual modality; (3) Popular events' images will cause significant degradation to the detectors when they are subjected to backdoor attacks; (4) The performance of these detectors under multi-modal attacks is worse than under uni-modal attacks; (5) Defensive methods will improve the robustness of the multi-modal detectors.



## **2. Detecting Adversarial Examples in Batches -- a geometrical approach**

cs.LG

Submitted to AdvML workshop at ICML2022

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2206.08738v1)

**Authors**: Danush Kumar Venkatesh, Peter Steinbach

**Abstracts**: Many deep learning methods have successfully solved complex tasks in computer vision and speech recognition applications. Nonetheless, the robustness of these models has been found to be vulnerable to perturbed inputs or adversarial examples, which are imperceptible to the human eye, but lead the model to erroneous output decisions. In this study, we adapt and introduce two geometric metrics, density and coverage, and evaluate their use in detecting adversarial samples in batches of unseen data. We empirically study these metrics using MNIST and two real-world biomedical datasets from MedMNIST, subjected to two different adversarial attacks. Our experiments show promising results for both metrics to detect adversarial examples. We believe that his work can lay the ground for further study on these metrics' use in deployed machine learning systems to monitor for possible attacks by adversarial examples or related pathologies such as dataset shift.



## **3. Minimum Noticeable Difference based Adversarial Privacy Preserving Image Generation**

cs.CV

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2206.08638v1)

**Authors**: Wen Sun, Jian Jin, Weisi Lin

**Abstracts**: Deep learning models are found to be vulnerable to adversarial examples, as wrong predictions can be caused by small perturbation in input for deep learning models. Most of the existing works of adversarial image generation try to achieve attacks for most models, while few of them make efforts on guaranteeing the perceptual quality of the adversarial examples. High quality adversarial examples matter for many applications, especially for the privacy preserving. In this work, we develop a framework based on the Minimum Noticeable Difference (MND) concept to generate adversarial privacy preserving images that have minimum perceptual difference from the clean ones but are able to attack deep learning models. To achieve this, an adversarial loss is firstly proposed to make the deep learning models attacked by the adversarial images successfully. Then, a perceptual quality-preserving loss is developed by taking the magnitude of perturbation and perturbation-caused structural and gradient changes into account, which aims to preserve high perceptual quality for adversarial image generation. To the best of our knowledge, this is the first work on exploring quality-preserving adversarial image generation based on the MND concept for privacy preserving. To evaluate its performance in terms of perceptual quality, the deep models on image classification and face recognition are tested with the proposed method and several anchor methods in this work. Extensive experimental results demonstrate that the proposed MND framework is capable of generating adversarial images with remarkably improved performance metrics (e.g., PSNR, SSIM, and MOS) than that generated with the anchor methods.



## **4. Query-Efficient and Scalable Black-Box Adversarial Attacks on Discrete Sequential Data via Bayesian Optimization**

cs.LG

ICML 2022; Codes at  https://github.com/snu-mllab/DiscreteBlockBayesAttack

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2206.08575v1)

**Authors**: Deokjae Lee, Seungyong Moon, Junhyeok Lee, Hyun Oh Song

**Abstracts**: We focus on the problem of adversarial attacks against models on discrete sequential data in the black-box setting where the attacker aims to craft adversarial examples with limited query access to the victim model. Existing black-box attacks, mostly based on greedy algorithms, find adversarial examples using pre-computed key positions to perturb, which severely limits the search space and might result in suboptimal solutions. To this end, we propose a query-efficient black-box attack using Bayesian optimization, which dynamically computes important positions using an automatic relevance determination (ARD) categorical kernel. We introduce block decomposition and history subsampling techniques to improve the scalability of Bayesian optimization when an input sequence becomes long. Moreover, we develop a post-optimization algorithm that finds adversarial examples with smaller perturbation size. Experiments on natural language and protein classification tasks demonstrate that our method consistently achieves higher attack success rate with significant reduction in query count and modification rate compared to the previous state-of-the-art methods.



## **5. Adversarial Attack and Defense for Non-Parametric Two-Sample Tests**

cs.LG

Accepted by ICML 2022

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2202.03077v2)

**Authors**: Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**Abstracts**: Non-parametric two-sample tests (TSTs) that judge whether two sets of samples are drawn from the same distribution, have been widely used in the analysis of critical data. People tend to employ TSTs as trusted basic tools and rarely have any doubt about their reliability. This paper systematically uncovers the failure mode of non-parametric TSTs through adversarial attacks and then proposes corresponding defense strategies. First, we theoretically show that an adversary can upper-bound the distributional shift which guarantees the attack's invisibility. Furthermore, we theoretically find that the adversary can also degrade the lower bound of a TST's test power, which enables us to iteratively minimize the test criterion in order to search for adversarial pairs. To enable TST-agnostic attacks, we propose an ensemble attack (EA) framework that jointly minimizes the different types of test criteria. Second, to robustify TSTs, we propose a max-min optimization that iteratively generates adversarial pairs to train the deep kernels. Extensive experiments on both simulated and real-world datasets validate the adversarial vulnerabilities of non-parametric TSTs and the effectiveness of our proposed defense. Source code is available at https://github.com/GodXuxilie/Robust-TST.git.



## **6. A Unified Evaluation of Textual Backdoor Learning: Frameworks and Benchmarks**

cs.LG

19 pages

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2206.08514v1)

**Authors**: Ganqu Cui, Lifan Yuan, Bingxiang He, Yangyi Chen, Zhiyuan Liu, Maosong Sun

**Abstracts**: Textual backdoor attacks are a kind of practical threat to NLP systems. By injecting a backdoor in the training phase, the adversary could control model predictions via predefined triggers. As various attack and defense models have been proposed, it is of great significance to perform rigorous evaluations. However, we highlight two issues in previous backdoor learning evaluations: (1) The differences between real-world scenarios (e.g. releasing poisoned datasets or models) are neglected, and we argue that each scenario has its own constraints and concerns, thus requires specific evaluation protocols; (2) The evaluation metrics only consider whether the attacks could flip the models' predictions on poisoned samples and retain performances on benign samples, but ignore that poisoned samples should also be stealthy and semantic-preserving. To address these issues, we categorize existing works into three practical scenarios in which attackers release datasets, pre-trained models, and fine-tuned models respectively, then discuss their unique evaluation methodologies. On metrics, to completely evaluate poisoned samples, we use grammar error increase and perplexity difference for stealthiness, along with text similarity for validity. After formalizing the frameworks, we develop an open-source toolkit OpenBackdoor to foster the implementations and evaluations of textual backdoor learning. With this toolkit, we perform extensive experiments to benchmark attack and defense models under the suggested paradigm. To facilitate the underexplored defenses against poisoned datasets, we further propose CUBE, a simple yet strong clustering-based defense baseline. We hope that our frameworks and benchmarks could serve as the cornerstones for future model development and evaluations.



## **7. I Know What You Trained Last Summer: A Survey on Stealing Machine Learning Models and Defences**

cs.LG

Under review at ACM Computing Surveys

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2206.08451v1)

**Authors**: Daryna Oliynyk, Rudolf Mayer, Andreas Rauber

**Abstracts**: Machine Learning-as-a-Service (MLaaS) has become a widespread paradigm, making even the most complex machine learning models available for clients via e.g. a pay-per-query principle. This allows users to avoid time-consuming processes of data collection, hyperparameter tuning, and model training. However, by giving their customers access to the (predictions of their) models, MLaaS providers endanger their intellectual property, such as sensitive training data, optimised hyperparameters, or learned model parameters. Adversaries can create a copy of the model with (almost) identical behavior using the the prediction labels only. While many variants of this attack have been described, only scattered defence strategies have been proposed, addressing isolated threats. This raises the necessity for a thorough systematisation of the field of model stealing, to arrive at a comprehensive understanding why these attacks are successful, and how they could be holistically defended against. We address this by categorising and comparing model stealing attacks, assessing their performance, and exploring corresponding defence techniques in different settings. We propose a taxonomy for attack and defence approaches, and provide guidelines on how to select the right attack or defence strategy based on the goal and available resources. Finally, we analyse which defences are rendered less effective by current attack strategies.



## **8. Boosting the Adversarial Transferability of Surrogate Model with Dark Knowledge**

cs.LG

26 pages, 5 figures

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2206.08316v1)

**Authors**: Dingcheng Yang, Zihao Xiao, Wenjian Yu

**Abstracts**: Deep neural networks (DNNs) for image classification are known to be vulnerable to adversarial examples. And, the adversarial examples have transferability, which means an adversarial example for a DNN model can fool another black-box model with a non-trivial probability. This gave birth of the transfer-based adversarial attack where the adversarial examples generated by a pretrained or known model (called surrogate model) are used to conduct black-box attack. There are some work on how to generate the adversarial examples from a given surrogate model to achieve better transferability. However, training a special surrogate model to generate adversarial examples with better transferability is relatively under-explored. In this paper, we propose a method of training a surrogate model with abundant dark knowledge to boost the adversarial transferability of the adversarial examples generated by the surrogate model. This trained surrogate model is named dark surrogate model (DSM), and the proposed method to train DSM consists of two key components: a teacher model extracting dark knowledge and providing soft labels, and the mixing augmentation skill which enhances the dark knowledge of training data. Extensive experiments have been conducted to show that the proposed method can substantially improve the adversarial transferability of surrogate model across different architectures of surrogate model and optimizers for generating adversarial examples. We also show that the proposed method can be applied to other scenarios of transfer-based attack that contain dark knowledge, like face verification.



## **9. Adversarial Patch Attacks and Defences in Vision-Based Tasks: A Survey**

cs.CV

A. Sharma and Y. Bian share equal contribution

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2206.08304v1)

**Authors**: Abhijith Sharma, Yijun Bian, Phil Munz, Apurva Narayan

**Abstracts**: Adversarial attacks in deep learning models, especially for safety-critical systems, are gaining more and more attention in recent years, due to the lack of trust in the security and robustness of AI models. Yet the more primitive adversarial attacks might be physically infeasible or require some resources that are hard to access like the training data, which motivated the emergence of patch attacks. In this survey, we provide a comprehensive overview to cover existing techniques of adversarial patch attacks, aiming to help interested researchers quickly catch up with the progress in this field. We also discuss existing techniques for developing detection and defences against adversarial patches, aiming to help the community better understand this field and its applications in the real world.



## **10. Adversarial Robustness of Graph-based Anomaly Detection**

cs.CR

arXiv admin note: substantial text overlap with arXiv:2106.09989

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2206.08260v1)

**Authors**: Yulin Zhu, Yuni Lai, Kaifa Zhao, Xiapu Luo, Mingquan Yuan, Jian Ren, Kai Zhou

**Abstracts**: Graph-based anomaly detection is becoming prevalent due to the powerful representation abilities of graphs as well as recent advances in graph mining techniques. These GAD tools, however, expose a new attacking surface, ironically due to their unique advantage of being able to exploit the relations among data. That is, attackers now can manipulate those relations (i.e., the structure of the graph) to allow target nodes to evade detection or degenerate the classification performance of the detection. In this paper, we exploit this vulnerability by designing the structural poisoning attacks to a FeXtra-based GAD system termed OddBall as well as the black box attacks against GCN-based GAD systems by attacking the imbalanced lienarized GCN ( LGCN ). Specifically, we formulate the attack against OddBall and LGCN as a one-level optimization problem by incorporating different regression techniques, where the key technical challenge is to efficiently solve the problem in a discrete domain. We propose a novel attack method termed BinarizedAttack based on gradient descent. Comparing to prior arts, BinarizedAttack can better use the gradient information, making it particularly suitable for solving discrete optimization problems, thus opening the door to studying a new type of attack against security analytic tools that rely on graph data.



## **11. Adversarial attacks on voter model dynamics in complex networks**

physics.soc-ph

7 pages, 5 figures

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2111.09561v2)

**Authors**: Katsumi Chiyomaru, Kazuhiro Takemoto

**Abstracts**: This study investigates adversarial attacks conducted to distort voter model dynamics in complex networks. Specifically, a simple adversarial attack method is proposed to hold the state of opinions of an individual closer to the target state in the voter model dynamics. This indicates that even when one opinion is the majority, the vote outcome can be inverted (i.e., the outcome can lean toward the other opinion) by adding extremely small (hard-to-detect) perturbations strategically generated in social networks. Adversarial attacks are relatively more effective in complex (large and dense) networks. These results indicate that opinion dynamics can be unknowingly distorted.



## **12. Adversarial Privacy Protection on Speech Enhancement**

cs.SD

5 pages, 6 figures

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2206.08170v1)

**Authors**: Mingyu Dong, Diqun Yan, Rangding Wang

**Abstracts**: Speech is easily leaked imperceptibly, such as being recorded by mobile phones in different situations. Private content in speech may be maliciously extracted through speech enhancement technology. Speech enhancement technology has developed rapidly along with deep neural networks (DNNs), but adversarial examples can cause DNNs to fail. In this work, we propose an adversarial method to degrade speech enhancement systems. Experimental results show that generated adversarial examples can erase most content information in original examples or replace it with target speech content through speech enhancement. The word error rate (WER) between an enhanced original example and enhanced adversarial example recognition result can reach 89.0%. WER of target attack between enhanced adversarial example and target example is low to 33.75% . Adversarial perturbation can bring the rate of change to the original example to more than 1.4430. This work can prevent the malicious extraction of speech.



## **13. Detecting Adversarial Examples Is (Nearly) As Hard As Classifying Them**

cs.LG

ICML 2022 (Long Talk)

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2107.11630v2)

**Authors**: Florian Tramèr

**Abstracts**: Making classifiers robust to adversarial examples is hard. Thus, many defenses tackle the seemingly easier task of detecting perturbed inputs. We show a barrier towards this goal. We prove a general hardness reduction between detection and classification of adversarial examples: given a robust detector for attacks at distance {\epsilon} (in some metric), we can build a similarly robust (but inefficient) classifier for attacks at distance {\epsilon}/2. Our reduction is computationally inefficient, and thus cannot be used to build practical classifiers. Instead, it is a useful sanity check to test whether empirical detection results imply something much stronger than the authors presumably anticipated. To illustrate, we revisit 13 detector defenses. For 11/13 cases, we show that the claimed detection results would imply an inefficient classifier with robustness far beyond the state-of-the-art.



## **14. Adversarial Attacks on Gaussian Process Bandits**

stat.ML

Accepted to ICML 2022

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2110.08449v3)

**Authors**: Eric Han, Jonathan Scarlett

**Abstracts**: Gaussian processes (GP) are a widely-adopted tool used to sequentially optimize black-box functions, where evaluations are costly and potentially noisy. Recent works on GP bandits have proposed to move beyond random noise and devise algorithms robust to adversarial attacks. This paper studies this problem from the attacker's perspective, proposing various adversarial attack methods with differing assumptions on the attacker's strength and prior information. Our goal is to understand adversarial attacks on GP bandits from theoretical and practical perspectives. We focus primarily on targeted attacks on the popular GP-UCB algorithm and a related elimination-based algorithm, based on adversarially perturbing the function $f$ to produce another function $\tilde{f}$ whose optima are in some target region $\mathcal{R}_{\rm target}$. Based on our theoretical analysis, we devise both white-box attacks (known $f$) and black-box attacks (unknown $f$), with the former including a Subtraction attack and Clipping attack, and the latter including an Aggressive subtraction attack. We demonstrate that adversarial attacks on GP bandits can succeed in forcing the algorithm towards $\mathcal{R}_{\rm target}$ even with a low attack budget, and we test our attacks' effectiveness on a diverse range of objective functions.



## **15. Analysis and Extensions of Adversarial Training for Video Classification**

cs.CV

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2206.07953v1)

**Authors**: Kaleab A. Kinfu, René Vidal

**Abstracts**: Adversarial training (AT) is a simple yet effective defense against adversarial attacks to image classification systems, which is based on augmenting the training set with attacks that maximize the loss. However, the effectiveness of AT as a defense for video classification has not been thoroughly studied. Our first contribution is to show that generating optimal attacks for video requires carefully tuning the attack parameters, especially the step size. Notably, we show that the optimal step size varies linearly with the attack budget. Our second contribution is to show that using a smaller (sub-optimal) attack budget at training time leads to a more robust performance at test time. Based on these findings, we propose three defenses against attacks with variable attack budgets. The first one, Adaptive AT, is a technique where the attack budget is drawn from a distribution that is adapted as training iterations proceed. The second, Curriculum AT, is a technique where the attack budget is increased as training iterations proceed. The third, Generative AT, further couples AT with a denoising generative adversarial network to boost robust performance. Experiments on the UCF101 dataset demonstrate that the proposed methods improve adversarial robustness against multiple attack types.



## **16. A Comprehensive Test Pattern Generation Approach Exploiting SAT Attack for Logic Locking**

cs.CR

10 pages, 8 figures

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2204.11307v2)

**Authors**: Yadi Zhong, Ujjwal Guin

**Abstracts**: The need for reducing manufacturing defect escape in today's safety-critical applications requires increased fault coverage. However, generating a test set using commercial automatic test pattern generation (ATPG) tools that lead to zero-defect escape is still an open problem. It is challenging to detect all stuck-at faults to reach 100% fault coverage. In parallel, the hardware security community has been actively involved in developing solutions for logic locking to prevent IP piracy. Locks (e.g., XOR gates) are inserted in different locations of the netlist so that an adversary cannot determine the secret key. Unfortunately, the Boolean satisfiability (SAT) based attack, introduced in [1], can break different logic locking schemes in minutes. In this paper, we propose a novel test pattern generation approach using the powerful SAT attack on logic locking. A stuck-at fault is modeled as a locked gate with a secret key. Our modeling of stuck-at faults preserves the property of fault activation and propagation. We show that the input pattern that determines the key is a test for the stuck-at fault. We propose two different approaches for test pattern generation. First, a single stuck-at fault is targeted, and a corresponding locked circuit with one key bit is created. This approach generates one test pattern per fault. Second, we consider a group of faults and convert the circuit to its locked version with multiple key bits. The inputs obtained from the SAT tool are the test set for detecting this group of faults. Our approach is able to find test patterns for hard-to-detect faults that were previously failed in commercial ATPG tools. The proposed test pattern generation approach can efficiently detect redundant faults present in a circuit. We demonstrate the effectiveness of the approach on ITC'99 benchmarks. The results show that we can achieve a perfect fault coverage reaching 100%.



## **17. Architectural Backdoors in Neural Networks**

cs.LG

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.07840v1)

**Authors**: Mikel Bober-Irizar, Ilia Shumailov, Yiren Zhao, Robert Mullins, Nicolas Papernot

**Abstracts**: Machine learning is vulnerable to adversarial manipulation. Previous literature has demonstrated that at the training stage attackers can manipulate data and data sampling procedures to control model behaviour. A common attack goal is to plant backdoors i.e. force the victim model to learn to recognise a trigger known only by the adversary. In this paper, we introduce a new class of backdoor attacks that hide inside model architectures i.e. in the inductive bias of the functions used to train. These backdoors are simple to implement, for instance by publishing open-source code for a backdoored model architecture that others will reuse unknowingly. We demonstrate that model architectural backdoors represent a real threat and, unlike other approaches, can survive a complete re-training from scratch. We formalise the main construction principles behind architectural backdoors, such as a link between the input and the output, and describe some possible protections against them. We evaluate our attacks on computer vision benchmarks of different scales and demonstrate the underlying vulnerability is pervasive in a variety of training settings.



## **18. Search-Based Testing Approach for Deep Reinforcement Learning Agents**

cs.SE

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.07813v1)

**Authors**: Amirhossein Zolfagharian, Manel Abdellatif, Lionel Briand, Mojtaba Bagherzadeh, Ramesh S

**Abstracts**: Deep Reinforcement Learning (DRL) algorithms have been increasingly employed during the last decade to solve various decision-making problems such as autonomous driving and robotics. However, these algorithms have faced great challenges when deployed in safety-critical environments since they often exhibit erroneous behaviors that can lead to potentially critical errors. One way to assess the safety of DRL agents is to test them to detect possible faults leading to critical failures during their execution. This raises the question of how we can efficiently test DRL policies to ensure their correctness and adherence to safety requirements. Most existing works on testing DRL agents use adversarial attacks that perturb states or actions of the agent. However, such attacks often lead to unrealistic states of the environment. Their main goal is to test the robustness of DRL agents rather than testing the compliance of agents' policies with respect to requirements. Due to the huge state space of DRL environments, the high cost of test execution, and the black-box nature of DRL algorithms, the exhaustive testing of DRL agents is impossible. In this paper, we propose a Search-based Testing Approach of Reinforcement Learning Agents (STARLA) to test the policy of a DRL agent by effectively searching for failing executions of the agent within a limited testing budget. We use machine learning models and a dedicated genetic algorithm to narrow the search towards faulty episodes. We apply STARLA on a Deep-Q-Learning agent which is widely used as a benchmark and show that it significantly outperforms Random Testing by detecting more faults related to the agent's policy. We also investigate how to extract rules that characterize faulty episodes of the DRL agent using our search results. Such rules can be used to understand the conditions under which the agent fails and thus assess its deployment risks.



## **19. Poison Forensics: Traceback of Data Poisoning Attacks in Neural Networks**

cs.CR

18 pages

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2110.06904v2)

**Authors**: Shawn Shan, Arjun Nitin Bhagoji, Haitao Zheng, Ben Y. Zhao

**Abstracts**: In adversarial machine learning, new defenses against attacks on deep learning systems are routinely broken soon after their release by more powerful attacks. In this context, forensic tools can offer a valuable complement to existing defenses, by tracing back a successful attack to its root cause, and offering a path forward for mitigation to prevent similar attacks in the future.   In this paper, we describe our efforts in developing a forensic traceback tool for poison attacks on deep neural networks. We propose a novel iterative clustering and pruning solution that trims "innocent" training samples, until all that remains is the set of poisoned data responsible for the attack. Our method clusters training samples based on their impact on model parameters, then uses an efficient data unlearning method to prune innocent clusters. We empirically demonstrate the efficacy of our system on three types of dirty-label (backdoor) poison attacks and three types of clean-label poison attacks, across domains of computer vision and malware classification. Our system achieves over 98.4% precision and 96.8% recall across all attacks. We also show that our system is robust against four anti-forensics measures specifically designed to attack it.



## **20. Learn to Adapt: Robust Drift Detection in Security Domain**

cs.CR

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.07581v1)

**Authors**: Aditya Kuppa, Nhien-An Le-Khac

**Abstracts**: Deploying robust machine learning models has to account for concept drifts arising due to the dynamically changing and non-stationary nature of data. Addressing drifts is particularly imperative in the security domain due to the ever-evolving threat landscape and lack of sufficiently labeled training data at the deployment time leading to performance degradation. Recently proposed concept drift detection methods in literature tackle this problem by identifying the changes in feature/data distributions and periodically retraining the models to learn new concepts. While these types of strategies should absolutely be conducted when possible, they are not robust towards attacker-induced drifts and suffer from a delay in detecting new attacks. We aim to address these shortcomings in this work. we propose a robust drift detector that not only identifies drifted samples but also discovers new classes as they arrive in an on-line fashion. We evaluate the proposed method with two security-relevant data sets -- network intrusion data set released in 2018 and APT Command and Control dataset combined with web categorization data. Our evaluation shows that our drifting detection method is not only highly accurate but also robust towards adversarial drifts and discovers new classes from drifted samples.



## **21. Creating a Secure Underlay for the Internet**

cs.NI

Usenix Security 2022

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.06879v2)

**Authors**: Henry Birge-Lee, Joel Wanner, Grace Cimaszewski, Jonghoon Kwon, Liang Wang, Francois Wirz, Prateek Mittal, Adrian Perrig, Yixin Sun

**Abstracts**: Adversaries can exploit inter-domain routing vulnerabilities to intercept communication and compromise the security of critical Internet applications. Meanwhile the deployment of secure routing solutions such as Border Gateway Protocol Security (BGPsec) and Scalability, Control and Isolation On Next-generation networks (SCION) are still limited. How can we leverage emerging secure routing backbones and extend their security properties to the broader Internet?   We design and deploy an architecture to bootstrap secure routing. Our key insight is to abstract the secure routing backbone as a virtual Autonomous System (AS), called Secure Backbone AS (SBAS). While SBAS appears as one AS to the Internet, it is a federated network where routes are exchanged between participants using a secure backbone. SBAS makes BGP announcements for its customers' IP prefixes at multiple locations (referred to as Points of Presence or PoPs) allowing traffic from non-participating hosts to be routed to a nearby SBAS PoP (where it is then routed over the secure backbone to the true prefix owner). In this manner, we are the first to integrate a federated secure non-BGP routing backbone with the BGP-speaking Internet.   We present a real-world deployment of our architecture that uses SCIONLab to emulate the secure backbone and the PEERING framework to make BGP announcements to the Internet. A combination of real-world attacks and Internet-scale simulations shows that SBAS substantially reduces the threat of routing attacks. Finally, we survey network operators to better understand optimal governance and incentive models.



## **22. TPC: Transformation-Specific Smoothing for Point Cloud Models**

cs.CV

Accepted as a conference paper at ICML 2022

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2201.12733v3)

**Authors**: Wenda Chu, Linyi Li, Bo Li

**Abstracts**: Point cloud models with neural network architectures have achieved great success and have been widely used in safety-critical applications, such as Lidar-based recognition systems in autonomous vehicles. However, such models are shown vulnerable to adversarial attacks which aim to apply stealthy semantic transformations such as rotation and tapering to mislead model predictions. In this paper, we propose a transformation-specific smoothing framework TPC, which provides tight and scalable robustness guarantees for point cloud models against semantic transformation attacks. We first categorize common 3D transformations into three categories: additive (e.g., shearing), composable (e.g., rotation), and indirectly composable (e.g., tapering), and we present generic robustness certification strategies for all categories respectively. We then specify unique certification protocols for a range of specific semantic transformations and their compositions. Extensive experiments on several common 3D transformations show that TPC significantly outperforms the state of the art. For example, our framework boosts the certified accuracy against twisting transformation along z-axis (within 20$^\circ$) from 20.3$\%$ to 83.8$\%$. Codes and models are available at https://github.com/Qianhewu/Point-Cloud-Smoothing.



## **23. Towards Understanding Adversarial Robustness of Optical Flow Networks**

cs.CV

CVPR 2022

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2103.16255v3)

**Authors**: Simon Schrodi, Tonmoy Saikia, Thomas Brox

**Abstracts**: Recent work demonstrated the lack of robustness of optical flow networks to physical patch-based adversarial attacks. The possibility to physically attack a basic component of automotive systems is a reason for serious concerns. In this paper, we analyze the cause of the problem and show that the lack of robustness is rooted in the classical aperture problem of optical flow estimation in combination with bad choices in the details of the network architecture. We show how these mistakes can be rectified in order to make optical flow networks robust to physical patch-based attacks. Additionally, we take a look at global white-box attacks in the scope of optical flow. We find that targeted white-box attacks can be crafted to bias flow estimation models towards any desired output, but this requires access to the input images and model weights. However, in the case of universal attacks, we find that optical flow networks are robust. Code is available at https://github.com/lmb-freiburg/understanding_flow_robustness.



## **24. Hardening DNNs against Transfer Attacks during Network Compression using Greedy Adversarial Pruning**

cs.LG

4 pages, 2 figures

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.07406v1)

**Authors**: Jonah O'Brien Weiss, Tiago Alves, Sandip Kundu

**Abstracts**: The prevalence and success of Deep Neural Network (DNN) applications in recent years have motivated research on DNN compression, such as pruning and quantization. These techniques accelerate model inference, reduce power consumption, and reduce the size and complexity of the hardware necessary to run DNNs, all with little to no loss in accuracy. However, since DNNs are vulnerable to adversarial inputs, it is important to consider the relationship between compression and adversarial robustness. In this work, we investigate the adversarial robustness of models produced by several irregular pruning schemes and by 8-bit quantization. Additionally, while conventional pruning removes the least important parameters in a DNN, we investigate the effect of an unconventional pruning method: removing the most important model parameters based on the gradient on adversarial inputs. We call this method Greedy Adversarial Pruning (GAP) and we find that this pruning method results in models that are resistant to transfer attacks from their uncompressed counterparts.



## **25. Morphence-2.0: Evasion-Resilient Moving Target Defense Powered by Out-of-Distribution Detection**

cs.CR

13 pages, 6 figures, 2 tables. arXiv admin note: substantial text  overlap with arXiv:2108.13952

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.07321v1)

**Authors**: Abderrahmen Amich, Ata Kaboudi, Birhanu Eshete

**Abstracts**: Evasion attacks against machine learning models often succeed via iterative probing of a fixed target model, whereby an attack that succeeds once will succeed repeatedly. One promising approach to counter this threat is making a model a moving target against adversarial inputs. To this end, we introduce Morphence-2.0, a scalable moving target defense (MTD) powered by out-of-distribution (OOD) detection to defend against adversarial examples. By regularly moving the decision function of a model, Morphence-2.0 makes it significantly challenging for repeated or correlated attacks to succeed. Morphence-2.0 deploys a pool of models generated from a base model in a manner that introduces sufficient randomness when it responds to prediction queries. Via OOD detection, Morphence-2.0 is equipped with a scheduling approach that assigns adversarial examples to robust decision functions and benign samples to an undefended accurate models. To ensure repeated or correlated attacks fail, the deployed pool of models automatically expires after a query budget is reached and the model pool is seamlessly replaced by a new model pool generated in advance. We evaluate Morphence-2.0 on two benchmark image classification datasets (MNIST and CIFAR10) against 4 reference attacks (3 white-box and 1 black-box). Morphence-2.0 consistently outperforms prior defenses while preserving accuracy on clean data and reducing attack transferability. We also show that, when powered by OOD detection, Morphence-2.0 is able to precisely make an input-based movement of the model's decision function that leads to higher prediction accuracy on both adversarial and benign queries.



## **26. Autoregressive Perturbations for Data Poisoning**

cs.LG

22 pages, 13 figures. Code available at  https://github.com/psandovalsegura/autoregressive-poisoning

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.03693v2)

**Authors**: Pedro Sandoval-Segura, Vasu Singla, Jonas Geiping, Micah Goldblum, Tom Goldstein, David W. Jacobs

**Abstracts**: The prevalence of data scraping from social media as a means to obtain datasets has led to growing concerns regarding unauthorized use of data. Data poisoning attacks have been proposed as a bulwark against scraping, as they make data "unlearnable" by adding small, imperceptible perturbations. Unfortunately, existing methods require knowledge of both the target architecture and the complete dataset so that a surrogate network can be trained, the parameters of which are used to generate the attack. In this work, we introduce autoregressive (AR) poisoning, a method that can generate poisoned data without access to the broader dataset. The proposed AR perturbations are generic, can be applied across different datasets, and can poison different architectures. Compared to existing unlearnable methods, our AR poisons are more resistant against common defenses such as adversarial training and strong data augmentations. Our analysis further provides insight into what makes an effective data poison.



## **27. Fast and Reliable Evaluation of Adversarial Robustness with Minimum-Margin Attack**

cs.LG

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.07314v1)

**Authors**: Ruize Gao, Jiongxiao Wang, Kaiwen Zhou, Feng Liu, Binghui Xie, Gang Niu, Bo Han, James Cheng

**Abstracts**: The AutoAttack (AA) has been the most reliable method to evaluate adversarial robustness when considerable computational resources are available. However, the high computational cost (e.g., 100 times more than that of the project gradient descent attack) makes AA infeasible for practitioners with limited computational resources, and also hinders applications of AA in the adversarial training (AT). In this paper, we propose a novel method, minimum-margin (MM) attack, to fast and reliably evaluate adversarial robustness. Compared with AA, our method achieves comparable performance but only costs 3% of the computational time in extensive experiments. The reliability of our method lies in that we evaluate the quality of adversarial examples using the margin between two targets that can precisely identify the most adversarial example. The computational efficiency of our method lies in an effective Sequential TArget Ranking Selection (STARS) method, ensuring that the cost of the MM attack is independent of the number of classes. The MM attack opens a new way for evaluating adversarial robustness and provides a feasible and reliable way to generate high-quality adversarial examples in AT.



## **28. Human Eyes Inspired Recurrent Neural Networks are More Robust Against Adversarial Noises**

cs.CV

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.07282v1)

**Authors**: Minkyu Choi, Yizhen Zhang, Kuan Han, Xiaokai Wang, Zhongming Liu

**Abstracts**: Compared to human vision, computer vision based on convolutional neural networks (CNN) are more vulnerable to adversarial noises. This difference is likely attributable to how the eyes sample visual input and how the brain processes retinal samples through its dorsal and ventral visual pathways, which are under-explored for computer vision. Inspired by the brain, we design recurrent neural networks, including an input sampler that mimics the human retina, a dorsal network that guides where to look next, and a ventral network that represents the retinal samples. Taking these modules together, the models learn to take multiple glances at an image, attend to a salient part at each glance, and accumulate the representation over time to recognize the image. We test such models for their robustness against a varying level of adversarial noises with a special focus on the effect of different input sampling strategies. Our findings suggest that retinal foveation and sampling renders a model more robust against adversarial noises, and the model may correct itself from an attack when it is given a longer time to take more glances at an image. In conclusion, robust visual recognition can benefit from the combined use of three brain-inspired mechanisms: retinal transformation, attention guided eye movement, and recurrent processing, as opposed to feedforward-only CNNs.



## **29. Can Linear Programs Have Adversarial Examples? A Causal Perspective**

cs.LG

Main paper: 9 pages, References: 2 page, Supplement: 2 pages. Main  paper: 2 figures, 3 tables, Supplement: 1 figure, 1 table

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2105.12697v5)

**Authors**: Matej Zečević, Devendra Singh Dhami, Kristian Kersting

**Abstracts**: The recent years have been marked by extended research on adversarial attacks, especially on deep neural networks. With this work we intend on posing and investigating the question of whether the phenomenon might be more general in nature, that is, adversarial-style attacks outside classification. Specifically, we investigate optimization problems starting with Linear Programs (LPs). We start off by demonstrating the shortcoming of a naive mapping between the formalism of adversarial examples and LPs, to then reveal how we can provide the missing piece -- intriguingly, through the Pearlian notion of Causality. Characteristically, we show the direct influence of the Structural Causal Model (SCM) onto the subsequent LP optimization, which ultimately exposes a notion of confounding in LPs (inherited by said SCM) that allows for adversarial-style attacks. We provide both the general proof formally alongside existential proofs of such intriguing LP-parameterizations based on SCM for three combinatorial problems, namely Linear Assignment, Shortest Path and a real world problem of energy systems.



## **30. Defending Observation Attacks in Deep Reinforcement Learning via Detection and Denoising**

cs.LG

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2206.07188v1)

**Authors**: Zikang Xiong, Joe Eappen, He Zhu, Suresh Jagannathan

**Abstracts**: Neural network policies trained using Deep Reinforcement Learning (DRL) are well-known to be susceptible to adversarial attacks. In this paper, we consider attacks manifesting as perturbations in the observation space managed by the external environment. These attacks have been shown to downgrade policy performance significantly. We focus our attention on well-trained deterministic and stochastic neural network policies in the context of continuous control benchmarks subject to four well-studied observation space adversarial attacks. To defend against these attacks, we propose a novel defense strategy using a detect-and-denoise schema. Unlike previous adversarial training approaches that sample data in adversarial scenarios, our solution does not require sampling data in an environment under attack, thereby greatly reducing risk during training. Detailed experimental results show that our technique is comparable with state-of-the-art adversarial training approaches.



## **31. Proximal Splitting Adversarial Attacks for Semantic Segmentation**

cs.LG

Code available at:  https://github.com/jeromerony/alma_prox_segmentation

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2206.07179v1)

**Authors**: Jérôme Rony, Jean-Christophe Pesquet, Ismail Ben Ayed

**Abstracts**: Classification has been the focal point of research on adversarial attacks, but only a few works investigate methods suited to denser prediction tasks, such as semantic segmentation. The methods proposed in these works do not accurately solve the adversarial segmentation problem and, therefore, are overoptimistic in terms of size of the perturbations required to fool models. Here, we propose a white-box attack for these models based on a proximal splitting to produce adversarial perturbations with much smaller $\ell_1$, $\ell_2$, or $\ell_\infty$ norms. Our attack can handle large numbers of constraints within a nonconvex minimization framework via an Augmented Lagrangian approach, coupled with adaptive constraint scaling and masking strategies. We demonstrate that our attack significantly outperforms previously proposed ones, as well as classification attacks that we adapted for segmentation, providing a first comprehensive benchmark for this dense task. Our results push current limits concerning robustness evaluations in segmentation tasks.



## **32. FENCE: Feasible Evasion Attacks on Neural Networks in Constrained Environments**

cs.CR

35 pages

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/1909.10480v4)

**Authors**: Alesia Chernikova, Alina Oprea

**Abstracts**: As advances in Deep Neural Networks (DNNs) demonstrate unprecedented levels of performance in many critical applications, their vulnerability to attacks is still an open question. We consider evasion attacks at testing time against Deep Learning in constrained environments, in which dependencies between features need to be satisfied. These situations may arise naturally in tabular data or may be the result of feature engineering in specific application domains, such as threat detection in cyber security. We propose a general iterative gradient-based framework called FENCE for crafting evasion attacks that take into consideration the specifics of constrained domains and application requirements. We apply it against Feed-Forward Neural Networks trained for two cyber security applications: network traffic botnet classification and malicious domain classification, to generate feasible adversarial examples. We extensively evaluate the success rate and performance of our attacks, compare their improvement over several baselines, and analyze factors that impact the attack success rate, including the optimization objective and the data imbalance. We show that with minimal effort (e.g., generating 12 additional network connections), an attacker can change the model's prediction from the Malicious class to Benign and evade the classifier. We show that models trained on datasets with higher imbalance are more vulnerable to our FENCE attacks. Finally, we demonstrate the potential of performing adversarial training in constrained domains to increase the model resilience against these evasion attacks.



## **33. A Layered Reference Model for Penetration Testing with Reinforcement Learning and Attack Graphs**

cs.CR

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2206.06934v1)

**Authors**: Tyler Cody

**Abstracts**: This paper considers key challenges to using reinforcement learning (RL) with attack graphs to automate penetration testing in real-world applications from a systems perspective. RL approaches to automated penetration testing are actively being developed, but there is no consensus view on the representation of computer networks with which RL should be interacting. Moreover, there are significant open challenges to how those representations can be grounded to the real networks where RL solution methods are applied. This paper elaborates on representation and grounding using topic challenges of interacting with real networks in real-time, emulating realistic adversary behavior, and handling unstable, evolving networks. These challenges are both practical and mathematical, and they directly concern the reliability and dependability of penetration testing systems. This paper proposes a layered reference model to help organize related research and engineering efforts. The presented layered reference model contrasts traditional models of attack graph workflows because it is not scoped to a sequential, feed-forward generation and analysis process, but to broader aspects of lifecycle and continuous deployment. Researchers and practitioners can use the presented layered reference model as a first-principles outline to help orient the systems engineering of their penetration testing systems.



## **34. When adversarial attacks become interpretable counterfactual explanations**

cs.AI

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2206.06854v1)

**Authors**: Mathieu Serrurier, Franck Mamalet, Thomas Fel, Louis Béthune, Thibaut Boissin

**Abstracts**: We argue that, when learning a 1-Lipschitz neural network with the dual loss of an optimal transportation problem, the gradient of the model is both the direction of the transportation plan and the direction to the closest adversarial attack. Traveling along the gradient to the decision boundary is no more an adversarial attack but becomes a counterfactual explanation, explicitly transporting from one class to the other. Through extensive experiments on XAI metrics, we find that the simple saliency map method, applied on such networks, becomes a reliable explanation, and outperforms the state-of-the-art explanation approaches on unconstrained models. The proposed networks were already known to be certifiably robust, and we prove that they are also explainable with a fast and simple method.



## **35. Exploring Adversarial Attacks and Defenses in Vision Transformers trained with DINO**

cs.CV

6 pages workshop paper accepted at AdvML Frontiers (ICML 2022)

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2206.06761v1)

**Authors**: Javier Rando, Nasib Naimi, Thomas Baumann, Max Mathys

**Abstracts**: This work conducts the first analysis on the robustness against adversarial attacks on self-supervised Vision Transformers trained using DINO. First, we evaluate whether features learned through self-supervision are more robust to adversarial attacks than those emerging from supervised learning. Then, we present properties arising for attacks in the latent space. Finally, we evaluate whether three well-known defense strategies can increase adversarial robustness in downstream tasks by only fine-tuning the classification head to provide robustness even in view of limited compute resources. These defense strategies are: Adversarial Training, Ensemble Adversarial Training and Ensemble of Specialized Networks.



## **36. Adversarial Vulnerability of Randomized Ensembles**

cs.LG

Published as a conference paper in ICML 2022

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2206.06737v1)

**Authors**: Hassan Dbouk, Naresh R. Shanbhag

**Abstracts**: Despite the tremendous success of deep neural networks across various tasks, their vulnerability to imperceptible adversarial perturbations has hindered their deployment in the real world. Recently, works on randomized ensembles have empirically demonstrated significant improvements in adversarial robustness over standard adversarially trained (AT) models with minimal computational overhead, making them a promising solution for safety-critical resource-constrained applications. However, this impressive performance raises the question: Are these robustness gains provided by randomized ensembles real? In this work we address this question both theoretically and empirically. We first establish theoretically that commonly employed robustness evaluation methods such as adaptive PGD provide a false sense of security in this setting. Subsequently, we propose a theoretically-sound and efficient adversarial attack algorithm (ARC) capable of compromising random ensembles even in cases where adaptive PGD fails to do so. We conduct comprehensive experiments across a variety of network architectures, training schemes, datasets, and norms to support our claims, and empirically establish that randomized ensembles are in fact more vulnerable to $\ell_p$-bounded adversarial perturbations than even standard AT models. Our code can be found at https://github.com/hsndbk4/ARC.



## **37. Downlink Power Allocation in Massive MIMO via Deep Learning: Adversarial Attacks and Training**

cs.LG

13 pages, 14 figures, published in IEEE Transactions on Cognitive  Communications and Networking

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2206.06592v1)

**Authors**: B. R. Manoj, Meysam Sadeghi, Erik G. Larsson

**Abstracts**: The successful emergence of deep learning (DL) in wireless system applications has raised concerns about new security-related challenges. One such security challenge is adversarial attacks. Although there has been much work demonstrating the susceptibility of DL-based classification tasks to adversarial attacks, regression-based problems in the context of a wireless system have not been studied so far from an attack perspective. The aim of this paper is twofold: (i) we consider a regression problem in a wireless setting and show that adversarial attacks can break the DL-based approach and (ii) we analyze the effectiveness of adversarial training as a defensive technique in adversarial settings and show that the robustness of DL-based wireless system against attacks improves significantly. Specifically, the wireless application considered in this paper is the DL-based power allocation in the downlink of a multicell massive multi-input-multi-output system, where the goal of the attack is to yield an infeasible solution by the DL model. We extend the gradient-based adversarial attacks: fast gradient sign method (FGSM), momentum iterative FGSM, and projected gradient descent method to analyze the susceptibility of the considered wireless application with and without adversarial training. We analyze the deep neural network (DNN) models performance against these attacks, where the adversarial perturbations are crafted using both the white-box and black-box attacks.



## **38. Person Re-identification Method Based on Color Attack and Joint Defence**

cs.CV

Accepted by CVPR2022 Workshops  (https://openaccess.thecvf.com/content/CVPR2022W/HCIS/html/Gong_Person_Re-Identification_Method_Based_on_Color_Attack_and_Joint_Defence_CVPRW_2022_paper.html)

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2111.09571v4)

**Authors**: Yunpeng Gong, Liqing Huang, Lifei Chen

**Abstracts**: The main challenges of ReID is the intra-class variations caused by color deviation under different camera conditions. Simultaneously, we find that most of the existing adversarial metric attacks are realized by interfering with the color characteristics of the sample. Based on this observation, we first propose a local transformation attack (LTA) based on color variation. It uses more obvious color variation to randomly disturb the color of the retrieved image, rather than adding random noise. Experiments show that the performance of the proposed LTA method is better than the advanced attack methods. Furthermore, considering that the contour feature is the main factor of the robustness of adversarial training, and the color feature will directly affect the success rate of attack. Therefore, we further propose joint adversarial defense (JAD) method, which include proactive defense and passive defense. Proactive defense fuse multi-modality images to enhance the contour feature and color feature, and considers local homomorphic transformation to solve the over-fitting problem. Passive defense exploits the invariance of contour feature during image scaling to mitigate the adversarial disturbance on contour feature. Finally, a series of experimental results show that the proposed joint adversarial defense method is more competitive than a state-of-the-art methods.



## **39. Towards Alternative Techniques for Improving Adversarial Robustness: Analysis of Adversarial Training at a Spectrum of Perturbations**

cs.LG

**SubmitDate**: 2022-06-13    [paper-pdf](http://arxiv.org/pdf/2206.06496v1)

**Authors**: Kaustubh Sridhar, Souradeep Dutta, Ramneet Kaur, James Weimer, Oleg Sokolsky, Insup Lee

**Abstracts**: Adversarial training (AT) and its variants have spearheaded progress in improving neural network robustness to adversarial perturbations and common corruptions in the last few years. Algorithm design of AT and its variants are focused on training models at a specified perturbation strength $\epsilon$ and only using the feedback from the performance of that $\epsilon$-robust model to improve the algorithm. In this work, we focus on models, trained on a spectrum of $\epsilon$ values. We analyze three perspectives: model performance, intermediate feature precision and convolution filter sensitivity. In each, we identify alternative improvements to AT that otherwise wouldn't have been apparent at a single $\epsilon$. Specifically, we find that for a PGD attack at some strength $\delta$, there is an AT model at some slightly larger strength $\epsilon$, but no greater, that generalizes best to it. Hence, we propose overdesigning for robustness where we suggest training models at an $\epsilon$ just above $\delta$. Second, we observe (across various $\epsilon$ values) that robustness is highly sensitive to the precision of intermediate features and particularly those after the first and second layer. Thus, we propose adding a simple quantization to defenses that improves accuracy on seen and unseen adaptive attacks. Third, we analyze convolution filters of each layer of models at increasing $\epsilon$ and notice that those of the first and second layer may be solely responsible for amplifying input perturbations. We present our findings and demonstrate our techniques through experiments with ResNet and WideResNet models on the CIFAR-10 and CIFAR-10-C datasets.



## **40. Distributed Adversarial Training to Robustify Deep Neural Networks at Scale**

cs.LG

**SubmitDate**: 2022-06-13    [paper-pdf](http://arxiv.org/pdf/2206.06257v1)

**Authors**: Gaoyuan Zhang, Songtao Lu, Yihua Zhang, Xiangyi Chen, Pin-Yu Chen, Quanfu Fan, Lee Martie, Lior Horesh, Mingyi Hong, Sijia Liu

**Abstracts**: Current deep neural networks (DNNs) are vulnerable to adversarial attacks, where adversarial perturbations to the inputs can change or manipulate classification. To defend against such attacks, an effective and popular approach, known as adversarial training (AT), has been shown to mitigate the negative impact of adversarial attacks by virtue of a min-max robust training method. While effective, it remains unclear whether it can successfully be adapted to the distributed learning context. The power of distributed optimization over multiple machines enables us to scale up robust training over large models and datasets. Spurred by that, we propose distributed adversarial training (DAT), a large-batch adversarial training framework implemented over multiple machines. We show that DAT is general, which supports training over labeled and unlabeled data, multiple types of attack generation methods, and gradient compression operations favored for distributed optimization. Theoretically, we provide, under standard conditions in the optimization theory, the convergence rate of DAT to the first-order stationary points in general non-convex settings. Empirically, we demonstrate that DAT either matches or outperforms state-of-the-art robust accuracies and achieves a graceful training speedup (e.g., on ResNet-50 under ImageNet). Codes are available at https://github.com/dat-2022/dat.



## **41. Adversarial Models Towards Data Availability and Integrity of Distributed State Estimation for Industrial IoT-Based Smart Grid**

cs.CR

11 pages (DC), Journal manuscript

**SubmitDate**: 2022-06-13    [paper-pdf](http://arxiv.org/pdf/2206.06027v1)

**Authors**: Haftu Tasew Reda, Abdun Mahmood, Adnan Anwar, Naveen Chilamkurti

**Abstracts**: Security issue of distributed state estimation (DSE) is an important prospect for the rapidly growing smart grid ecosystem. Any coordinated cyberattack targeting the distributed system of state estimators can cause unrestrained estimation errors and can lead to a myriad of security risks, including failure of power system operation. This article explores the security threats of a smart grid arising from the exploitation of DSE vulnerabilities. To this aim, novel adversarial strategies based on two-stage data availability and integrity attacks are proposed towards a distributed industrial Internet of Things-based smart grid. The former's attack goal is to prevent boundary data exchange among distributed control centers, while the latter's attack goal is to inject a falsified data to cause local and global system unobservability. The proposed framework is evaluated on IEEE standard 14-bus system and benchmarked against the state-of-the-art research. Experimental results show that the proposed two-stage cyberattack results in an estimated error of approximately 34.74% compared to an error of the order of 10^-3 under normal operating conditions.



## **42. Universal, transferable and targeted adversarial attacks**

cs.LG

**SubmitDate**: 2022-06-13    [paper-pdf](http://arxiv.org/pdf/1908.11332v4)

**Authors**: Junde Wu, Rao Fu

**Abstracts**: Deep Neural Networks have been found vulnerable re-cently. A kind of well-designed inputs, which called adver-sarial examples, can lead the networks to make incorrectpredictions. Depending on the different scenarios, goalsand capabilities, the difficulties of the attacks are different.For example, a targeted attack is more difficult than a non-targeted attack, a universal attack is more difficult than anon-universal attack, a transferable attack is more difficultthan a nontransferable one. The question is: Is there existan attack that can meet all these requirements? In this pa-per, we answer this question by producing a kind of attacksunder these conditions. We learn a universal mapping tomap the sources to the adversarial examples. These exam-ples can fool classification networks to classify all of theminto one targeted class, and also have strong transferability.Our code is released at: xxxxx.



## **43. Revisiting and Advancing Fast Adversarial Training Through The Lens of Bi-Level Optimization**

cs.LG

**SubmitDate**: 2022-06-13    [paper-pdf](http://arxiv.org/pdf/2112.12376v5)

**Authors**: Yihua Zhang, Guanhua Zhang, Prashant Khanduri, Mingyi Hong, Shiyu Chang, Sijia Liu

**Abstracts**: Adversarial training (AT) is a widely recognized defense mechanism to gain the robustness of deep neural networks against adversarial attacks. It is built on min-max optimization (MMO), where the minimizer (i.e., defender) seeks a robust model to minimize the worst-case training loss in the presence of adversarial examples crafted by the maximizer (i.e., attacker). However, the conventional MMO method makes AT hard to scale. Thus, Fast-AT (Wong et al., 2020) and other recent algorithms attempt to simplify MMO by replacing its maximization step with the single gradient sign-based attack generation step. Although easy to implement, Fast-AT lacks theoretical guarantees, and its empirical performance is unsatisfactory due to the issue of robust catastrophic overfitting when training with strong adversaries. In this paper, we advance Fast-AT from the fresh perspective of bi-level optimization (BLO). We first show that the commonly-used Fast-AT is equivalent to using a stochastic gradient algorithm to solve a linearized BLO problem involving a sign operation. However, the discrete nature of the sign operation makes it difficult to understand the algorithm performance. Inspired by BLO, we design and analyze a new set of robust training algorithms termed Fast Bi-level AT (Fast-BAT), which effectively defends sign-based projected gradient descent (PGD) attacks without using any gradient sign method or explicit robust regularization. In practice, we show our method yields substantial robustness improvements over baselines across multiple models and datasets. Codes are available at https://github.com/OPTML-Group/Fast-BAT.



## **44. Deploying Convolutional Networks on Untrusted Platforms Using 2D Holographic Reduced Representations**

cs.LG

To appear in the Proceedings of the 39 th International Conference on  Machine Learning, Baltimore, Maryland, USA, PMLR 162, 2022

**SubmitDate**: 2022-06-13    [paper-pdf](http://arxiv.org/pdf/2206.05893v1)

**Authors**: Mohammad Mahmudul Alam, Edward Raff, Tim Oates, James Holt

**Abstracts**: Due to the computational cost of running inference for a neural network, the need to deploy the inferential steps on a third party's compute environment or hardware is common. If the third party is not fully trusted, it is desirable to obfuscate the nature of the inputs and outputs, so that the third party can not easily determine what specific task is being performed. Provably secure protocols for leveraging an untrusted party exist but are too computational demanding to run in practice. We instead explore a different strategy of fast, heuristic security that we call Connectionist Symbolic Pseudo Secrets. By leveraging Holographic Reduced Representations (HRR), we create a neural network with a pseudo-encryption style defense that empirically shows robustness to attack, even under threat models that unrealistically favor the adversary.



## **45. InBiaseD: Inductive Bias Distillation to Improve Generalization and Robustness through Shape-awareness**

cs.CV

Accepted at 1st Conference on Lifelong Learning Agents (CoLLAs 2022)

**SubmitDate**: 2022-06-12    [paper-pdf](http://arxiv.org/pdf/2206.05846v1)

**Authors**: Shruthi Gowda, Bahram Zonooz, Elahe Arani

**Abstracts**: Humans rely less on spurious correlations and trivial cues, such as texture, compared to deep neural networks which lead to better generalization and robustness. It can be attributed to the prior knowledge or the high-level cognitive inductive bias present in the brain. Therefore, introducing meaningful inductive bias to neural networks can help learn more generic and high-level representations and alleviate some of the shortcomings. We propose InBiaseD to distill inductive bias and bring shape-awareness to the neural networks. Our method includes a bias alignment objective that enforces the networks to learn more generic representations that are less vulnerable to unintended cues in the data which results in improved generalization performance. InBiaseD is less susceptible to shortcut learning and also exhibits lower texture bias. The better representations also aid in improving robustness to adversarial attacks and we hence plugin InBiaseD seamlessly into the existing adversarial training schemes to show a better trade-off between generalization and robustness.



## **46. Consistent Attack: Universal Adversarial Perturbation on Embodied Vision Navigation**

cs.LG

**SubmitDate**: 2022-06-12    [paper-pdf](http://arxiv.org/pdf/2206.05751v1)

**Authors**: You Qiaoben, Chengyang Ying, Xinning Zhou, Hang Su, Jun Zhu, Bo Zhang

**Abstracts**: Embodied agents in vision navigation coupled with deep neural networks have attracted increasing attention. However, deep neural networks are vulnerable to malicious adversarial noises, which may potentially cause catastrophic failures in Embodied Vision Navigation. Among these adversarial noises, universal adversarial perturbations (UAP), i.e., the image-agnostic perturbation applied on each frame received by the agent, are more critical for Embodied Vision Navigation since they are computation-efficient and application-practical during the attack. However, existing UAP methods do not consider the system dynamics of Embodied Vision Navigation. For extending UAP in the sequential decision setting, we formulate the disturbed environment under the universal noise $\delta$, as a $\delta$-disturbed Markov Decision Process ($\delta$-MDP). Based on the formulation, we analyze the properties of $\delta$-MDP and propose two novel Consistent Attack methods for attacking Embodied agents, which first consider the dynamic of the MDP by estimating the disturbed Q function and the disturbed distribution. In spite of victim models, our Consistent Attack can cause a significant drop in the performance for the Goalpoint task in habitat. Extensive experimental results indicate that there exist potential risks for applying Embodied Vision Navigation methods to the real world.



## **47. Darknet Traffic Classification and Adversarial Attacks**

cs.LG

**SubmitDate**: 2022-06-12    [paper-pdf](http://arxiv.org/pdf/2206.06371v1)

**Authors**: Nhien Rust-Nguyen, Mark Stamp

**Abstracts**: The anonymous nature of darknets is commonly exploited for illegal activities. Previous research has employed machine learning and deep learning techniques to automate the detection of darknet traffic in an attempt to block these criminal activities. This research aims to improve darknet traffic detection by assessing Support Vector Machines (SVM), Random Forest (RF), Convolutional Neural Networks (CNN), and Auxiliary-Classifier Generative Adversarial Networks (AC-GAN) for classification of such traffic and the underlying application types. We find that our RF model outperforms the state-of-the-art machine learning techniques used in prior work with the CIC-Darknet2020 dataset. To evaluate the robustness of our RF classifier, we obfuscate select application type classes to simulate realistic adversarial attack scenarios. We demonstrate that our best-performing classifier can be defeated by such attacks, and we consider ways to deal with such adversarial attacks.



## **48. Security of Machine Learning-Based Anomaly Detection in Cyber Physical Systems**

cs.DC

**SubmitDate**: 2022-06-12    [paper-pdf](http://arxiv.org/pdf/2206.05678v1)

**Authors**: Zahra Jadidi, Shantanu Pal, Nithesh Nayak K, Arawinkumaar Selvakkumar, Chih-Chia Chang, Maedeh Beheshti, Alireza Jolfaei

**Abstracts**: In this study, we focus on the impact of adversarial attacks on deep learning-based anomaly detection in CPS networks and implement a mitigation approach against the attack by retraining models using adversarial samples. We use the Bot-IoT and Modbus IoT datasets to represent the two CPS networks. We train deep learning models and generate adversarial samples using these datasets. These datasets are captured from IoT and Industrial IoT (IIoT) networks. They both provide samples of normal and attack activities. The deep learning model trained with these datasets showed high accuracy in detecting attacks. An Artificial Neural Network (ANN) is adopted with one input layer, four intermediate layers, and one output layer. The output layer has two nodes representing the binary classification results. To generate adversarial samples for the experiment, we used a function called the `fast_gradient_method' from the Cleverhans library. The experimental result demonstrates the influence of FGSM adversarial samples on the accuracy of the predictions and proves the effectiveness of using the retrained model to defend against adversarial attacks.



## **49. An Efficient Method for Sample Adversarial Perturbations against Nonlinear Support Vector Machines**

cs.LG

**SubmitDate**: 2022-06-12    [paper-pdf](http://arxiv.org/pdf/2206.05664v1)

**Authors**: Wen Su, Qingna Li

**Abstracts**: Adversarial perturbations have drawn great attentions in various machine learning models. In this paper, we investigate the sample adversarial perturbations for nonlinear support vector machines (SVMs). Due to the implicit form of the nonlinear functions mapping data to the feature space, it is difficult to obtain the explicit form of the adversarial perturbations. By exploring the special property of nonlinear SVMs, we transform the optimization problem of attacking nonlinear SVMs into a nonlinear KKT system. Such a system can be solved by various numerical methods. Numerical results show that our method is efficient in computing adversarial perturbations.



## **50. Reward Poisoning Attacks on Offline Multi-Agent Reinforcement Learning**

cs.LG

**SubmitDate**: 2022-06-11    [paper-pdf](http://arxiv.org/pdf/2206.01888v2)

**Authors**: Young Wu, Jeremey McMahan, Xiaojin Zhu, Qiaomin Xie

**Abstracts**: We expose the danger of reward poisoning in offline multi-agent reinforcement learning (MARL), whereby an attacker can modify the reward vectors to different learners in an offline data set while incurring a poisoning cost. Based on the poisoned data set, all rational learners using some confidence-bound-based MARL algorithm will infer that a target policy - chosen by the attacker and not necessarily a solution concept originally - is the Markov perfect dominant strategy equilibrium for the underlying Markov Game, hence they will adopt this potentially damaging target policy in the future. We characterize the exact conditions under which the attacker can install a target policy. We further show how the attacker can formulate a linear program to minimize its poisoning cost. Our work shows the need for robust MARL against adversarial attacks.



