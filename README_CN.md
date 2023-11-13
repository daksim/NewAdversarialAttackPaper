# Latest Adversarial Attack Papers
**update at 2023-11-13 15:25:12**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Triad: Trusted Timestamps in Untrusted Environments**

Triad：不可信环境中的可信时间戳 cs.CR

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2311.06156v1) [paper-pdf](http://arxiv.org/pdf/2311.06156v1)

**Authors**: Gabriel P. Fernandez, Andrey Brito, Christof Fetzer

**Abstract**: We aim to provide trusted time measurement mechanisms to applications and cloud infrastructure deployed in environments that could harbor potential adversaries, including the hardware infrastructure provider. Despite Trusted Execution Environments (TEEs) providing multiple security functionalities, timestamps from the Operating System are not covered. Nevertheless, some services require time for validating permissions or ordering events. To address that need, we introduce Triad, a trusted timestamp dispatcher of time readings. The solution provides trusted timestamps enforced by mutually supportive enclave-based clock servers that create a continuous trusted timeline. We leverage enclave properties such as forced exits and CPU-based counters to mitigate attacks on the server's timestamp counters. Triad produces trusted, confidential, monotonically-increasing timestamps with bounded error and desirable, non-trivial properties. Our implementation relies on Intel SGX and SCONE, allowing transparent usage. We evaluate Triad's error and behavior in multiple dimensions.

摘要: 我们的目标是为部署在可能存在潜在对手(包括硬件基础设施提供商)的环境中的应用程序和云基础设施提供可信时间测量机制。尽管可信执行环境(TEE)提供了多种安全功能，但不包括来自操作系统的时间戳。然而，一些服务需要时间来验证权限或对事件进行排序。为了满足这一需求，我们引入了Triad，这是一个可信的时间戳读数分派器。该解决方案提供可信的时间戳，由相互支持的基于Enclave的时钟服务器执行，从而创建连续的可信时间线。我们利用强制退出和基于CPU的计数器等Enclave属性来减少对服务器时间戳计数器的攻击。Triad生成可信的、保密的、单调递增的时间戳，具有有界的误差和所需的非平凡属性。我们的实施依赖于Intel SGX和Scon，允许透明使用。我们从多个维度评估三合会的错误和行为。



## **2. Fight Fire with Fire: Combating Adversarial Patch Attacks using Pattern-randomized Defensive Patches**

以火还击：使用模式随机防御补丁对抗对抗性补丁攻击 cs.CV

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2311.06122v1) [paper-pdf](http://arxiv.org/pdf/2311.06122v1)

**Authors**: Jianan Feng, Jiachun Li, Changqing Miao, Jianjun Huang, Wei You, Wenchang Shi, Bin Liang

**Abstract**: Object detection has found extensive applications in various tasks, but it is also susceptible to adversarial patch attacks. Existing defense methods often necessitate modifications to the target model or result in unacceptable time overhead. In this paper, we adopt a counterattack approach, following the principle of "fight fire with fire," and propose a novel and general methodology for defending adversarial attacks. We utilize an active defense strategy by injecting two types of defensive patches, canary and woodpecker, into the input to proactively probe or weaken potential adversarial patches without altering the target model. Moreover, inspired by randomization techniques employed in software security, we employ randomized canary and woodpecker injection patterns to defend against defense-aware attacks. The effectiveness and practicality of the proposed method are demonstrated through comprehensive experiments. The results illustrate that canary and woodpecker achieve high performance, even when confronted with unknown attack methods, while incurring limited time overhead. Furthermore, our method also exhibits sufficient robustness against defense-aware attacks, as evidenced by adaptive attack experiments.

摘要: 目标检测在各种任务中得到了广泛的应用，但它也容易受到对抗性补丁的攻击。现有的防御方法通常需要对目标模型进行修改，或者导致不可接受的时间开销。本文采用反击的方法，遵循“以火还火”的原则，提出了一种新颖的、通用的防御对抗性攻击的方法。我们利用主动防御策略，在输入中注入两种类型的防御补丁，金丝雀和啄木鸟，在不改变目标模型的情况下主动探测或削弱潜在的敌方补丁。此外，受软件安全中使用的随机化技术的启发，我们使用随机化的金丝雀和啄木鸟注入模式来防御防御感知攻击。通过综合实验，验证了该方法的有效性和实用性。实验结果表明，金丝雀和啄木鸟在攻击方式未知的情况下也能获得较高的性能，同时具有有限的时间开销。此外，自适应攻击实验表明，该方法对防御感知攻击也表现出了足够的鲁棒性。



## **3. Blades: A Unified Benchmark Suite for Byzantine Attacks and Defenses in Federated Learning**

刀片：联邦学习中拜占庭攻击和防御的统一基准套件 cs.CR

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2206.05359v4) [paper-pdf](http://arxiv.org/pdf/2206.05359v4)

**Authors**: Shenghui Li, Edith Ngai, Fanghua Ye, Li Ju, Tianru Zhang, Thiemo Voigt

**Abstract**: Federated learning (FL) facilitates distributed training across different IoT and edge devices, safeguarding the privacy of their data. The inherent distributed structure of FL introduces vulnerabilities, especially from adversarial devices aiming to skew local updates to their advantage. Despite the plethora of research focusing on Byzantine-resilient FL, the academic community has yet to establish a comprehensive benchmark suite, pivotal for impartial assessment and comparison of different techniques. This paper presents Blades, a scalable, extensible, and easily configurable benchmark suite that supports researchers and developers in efficiently implementing and validating novel strategies against baseline algorithms in Byzantine-resilient FL. Blades contains built-in implementations of representative attack and defense strategies and offers a user-friendly interface that seamlessly integrates new ideas. Using Blades, we re-evaluate representative attacks and defenses on wide-ranging experimental configurations (approximately 1,500 trials in total). Through our extensive experiments, we gained new insights into FL robustness and highlighted previously overlooked limitations due to the absence of thorough evaluations and comparisons of baselines under various attack settings.

摘要: 联合学习(FL)可促进跨不同物联网和边缘设备的分布式培训，保护其数据隐私。FL固有的分布式结构引入了漏洞，特别是来自敌方设备的漏洞，旨在歪曲本地更新以利于他们。尽管有太多关于拜占庭式外语的研究，但学术界还没有建立一个全面的基准套件，这是公正评估和比较不同技术的关键。本文介绍了Blade，一个可伸缩、可扩展且易于配置的基准测试套件，它支持研究人员和开发人员有效地实施和验证针对拜占庭弹性FL中的基线算法的新策略。Blade包含典型攻击和防御策略的内置实现，并提供无缝集成新想法的用户友好界面。使用Blade，我们在广泛的实验配置(总共约1,500次试验)上重新评估了典型的攻击和防御。通过我们广泛的实验，我们获得了对外语稳健性的新见解，并强调了由于缺乏对各种攻击设置下的基线进行彻底的评估和比较而被忽视的局限性。



## **4. Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration**

基于自校正的大语言模型隶属度推理攻击 cs.CL

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2311.06062v1) [paper-pdf](http://arxiv.org/pdf/2311.06062v1)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: Membership Inference Attacks (MIA) aim to infer whether a target data record has been utilized for model training or not. Prior attempts have quantified the privacy risks of language models (LMs) via MIAs, but there is still no consensus on whether existing MIA algorithms can cause remarkable privacy leakage on practical Large Language Models (LLMs). Existing MIAs designed for LMs can be classified into two categories: reference-free and reference-based attacks. They are both based on the hypothesis that training records consistently strike a higher probability of being sampled. Nevertheless, this hypothesis heavily relies on the overfitting of target models, which will be mitigated by multiple regularization methods and the generalization of LLMs. The reference-based attack seems to achieve promising effectiveness in LLMs, which measures a more reliable membership signal by comparing the probability discrepancy between the target model and the reference model. However, the performance of reference-based attack is highly dependent on a reference dataset that closely resembles the training dataset, which is usually inaccessible in the practical scenario. Overall, existing MIAs are unable to effectively unveil privacy leakage over practical fine-tuned LLMs that are overfitting-free and private. We propose a Membership Inference Attack based on Self-calibrated Probabilistic Variation (SPV-MIA). Specifically, since memorization in LLMs is inevitable during the training process and occurs before overfitting, we introduce a more reliable membership signal, probabilistic variation, which is based on memorization rather than overfitting. Furthermore, we introduce a self-prompt approach, which constructs the dataset to fine-tune the reference model by prompting the target LLM itself. In this manner, the adversary can collect a dataset with a similar distribution from public APIs.

摘要: 成员关系推理攻击(MIA)的目的是推断目标数据记录是否已被用于模型训练。以往的研究已经通过MIA量化了语言模型的隐私风险，但对于现有的MIA算法是否会在实际的大型语言模型上造成显著的隐私泄漏，目前还没有达成共识。现有的针对LMS设计的MIA可以分为两类：无引用攻击和基于引用攻击。它们都是基于这样的假设，即培训记录始终具有更高的被抽样概率。然而，这一假设在很大程度上依赖于目标模型的过度拟合，而多种正则化方法和LLMS的推广将缓解这一问题。基于参考的攻击在LLMS中似乎取得了很好的效果，它通过比较目标模型和参考模型之间的概率差异来衡量更可靠的成员信号。然而，基于参考的攻击的性能高度依赖于与训练数据集非常相似的参考数据集，这在实际场景中通常是不可访问的。总体而言，现有的MIA无法有效地揭示实用的微调LLM的隐私泄露，这些LLM是免装修和私密的。提出了一种基于自校准概率变异的成员推理攻击(SPV-MIA)。具体地说，由于LLMS中的记忆在训练过程中是不可避免的，并且发生在过适应之前，因此我们引入了一种更可靠的隶属度信号-概率变异，它基于记忆而不是过适应。此外，我们引入了一种自我提示的方法，该方法构建数据集，通过提示目标LLM本身来微调参考模型。通过这种方式，攻击者可以从公共API收集具有类似分布的数据集。



## **5. Robust Adversarial Attacks Detection for Deep Learning based Relative Pose Estimation for Space Rendezvous**

基于深度学习的空间交会相对位姿估计的鲁棒对抗攻击检测 cs.CV

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2311.05992v1) [paper-pdf](http://arxiv.org/pdf/2311.05992v1)

**Authors**: Ziwei Wang, Nabil Aouf, Jose Pizarro, Christophe Honvault

**Abstract**: Research on developing deep learning techniques for autonomous spacecraft relative navigation challenges is continuously growing in recent years. Adopting those techniques offers enhanced performance. However, such approaches also introduce heightened apprehensions regarding the trustability and security of such deep learning methods through their susceptibility to adversarial attacks. In this work, we propose a novel approach for adversarial attack detection for deep neural network-based relative pose estimation schemes based on the explainability concept. We develop for an orbital rendezvous scenario an innovative relative pose estimation technique adopting our proposed Convolutional Neural Network (CNN), which takes an image from the chaser's onboard camera and outputs accurately the target's relative position and rotation. We perturb seamlessly the input images using adversarial attacks that are generated by the Fast Gradient Sign Method (FGSM). The adversarial attack detector is then built based on a Long Short Term Memory (LSTM) network which takes the explainability measure namely SHapley Value from the CNN-based pose estimator and flags the detection of adversarial attacks when acting. Simulation results show that the proposed adversarial attack detector achieves a detection accuracy of 99.21%. Both the deep relative pose estimator and adversarial attack detector are then tested on real data captured from our laboratory-designed setup. The experimental results from our laboratory-designed setup demonstrate that the proposed adversarial attack detector achieves an average detection accuracy of 96.29%.

摘要: 近年来，针对自主航天器相对导航挑战的深度学习技术的研究不断发展。采用这些技术可以提高性能。然而，这种方法也加剧了人们对这种深度学习方法的可信性和安全性的担忧，因为它们容易受到对手的攻击。在这项工作中，我们提出了一种基于可解释性概念的深度神经网络相对位姿估计算法的敌意攻击检测方法。对于轨道交会场景，我们开发了一种创新的相对位姿估计技术，该技术采用我们提出的卷积神经网络(CNN)，从追逐者的机载相机获取图像，并准确地输出目标的相对位置和旋转。我们使用由快速梯度符号方法(FGSM)产生的对抗性攻击来无缝地扰动输入图像。然后基于长短期记忆(LSTM)网络构建对抗性攻击检测器，该网络从基于CNN的姿态估计器中获取可解释性度量Shapley值，并在动作时标记检测到的对抗性攻击。仿真结果表明，所提出的对抗性攻击检测器的检测准确率达到99.21%。然后，对深度相对位姿估计器和敌方攻击检测器进行了测试，这些数据来自我们实验室设计的设置。实验结果表明，本文提出的对抗性攻击检测器的平均检测准确率为96.29%。



## **6. Resilient and constrained consensus against adversarial attacks: A distributed MPC framework**

抗敌意攻击的弹性受限共识：一种分布式MPC框架 eess.SY

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2311.05935v1) [paper-pdf](http://arxiv.org/pdf/2311.05935v1)

**Authors**: Henglai Wei, Kunwu Zhang, Hui Zhang, Yang Shi

**Abstract**: There has been a growing interest in realizing the resilient consensus of the multi-agent system (MAS) under cyber-attacks, which aims to achieve the consensus of normal agents (i.e., agents without attacks) in a network, depending on the neighboring information. The literature has developed mean-subsequence-reduced (MSR) algorithms for the MAS with F adversarial attacks and has shown that the consensus is achieved for the normal agents when the communication network is at least (2F+1)-robust. However, such a stringent requirement on the communication network needs to be relaxed to enable more practical applications. Our objective is, for the first time, to achieve less stringent conditions on the network, while ensuring the resilient consensus for the general linear MAS subject to control input constraints. In this work, we propose a distributed resilient consensus framework, consisting of a pre-designed consensus protocol and distributed model predictive control (DMPC) optimization, which can help significantly reduce the requirement on the network robustness and effectively handle the general linear constrained MAS under adversarial attacks. By employing a novel distributed adversarial attack detection mechanism based on the history information broadcast by neighbors and a convex set (i.e., resilience set), we can evaluate the reliability of communication links. Moreover, we show that the recursive feasibility of the associated DMPC optimization problem can be guaranteed. The proposed consensus protocol features the following properties: 1) by minimizing a group of control variables, the consensus performance is optimized; 2) the resilient consensus of the general linear constrained MAS subject to F-locally adversarial attacks is achieved when the communication network is (F+1)-robust. Finally, numerical simulation results are presented to verify the theoretical results.

摘要: 在网络攻击下实现多智能体系统(MAS)的弹性共识越来越受到人们的关注，MAS的目标是依靠周围的信息实现网络中正常智能体(即未受攻击的智能体)的一致性。已有文献提出了具有F个对抗性攻击的MAS的均值-子序列简化(MSR)算法，并表明当通信网络至少具有(2F+1)-健壮性时，对于正常的代理是一致的。然而，需要放宽对通信网络的如此严格的要求，以实现更实际的应用。我们的目标是，第一次达到对网络不那么严格的条件，同时确保一般线性MAS在控制输入约束下的弹性共识。在这项工作中，我们提出了一种分布式弹性共识框架，由预先设计的共识协议和分布式模型预测控制(DMPC)优化组成，能够显著降低对网络健壮性的要求，并有效地应对对抗性攻击下的一般线性约束MAS。通过采用一种新的基于邻居广播历史信息和凸集(即弹性集)的分布式对抗攻击检测机制，可以评估通信链路的可靠性。此外，我们还证明了相关的DMPC优化问题的递归可行性是可以保证的。所提出的一致性协议具有以下特点：1)通过最小化一组控制变量来优化一致性性能；2)当通信网络是(F+1)-鲁棒时，获得了一般线性约束MAS在F-局部对抗攻击下的弹性一致性。最后，给出了数值模拟结果，验证了理论结果。



## **7. On the (In)security of Peer-to-Peer Decentralized Machine Learning**

点对点分散机器学习的安全性研究 cs.CR

IEEE S&P'23 (Previous title: "On the Privacy of Decentralized Machine  Learning") + Fixed error in neighbors-discovery trick

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2205.08443v3) [paper-pdf](http://arxiv.org/pdf/2205.08443v3)

**Authors**: Dario Pasquini, Mathilde Raynal, Carmela Troncoso

**Abstract**: In this work, we carry out the first, in-depth, privacy analysis of Decentralized Learning -- a collaborative machine learning framework aimed at addressing the main limitations of federated learning. We introduce a suite of novel attacks for both passive and active decentralized adversaries. We demonstrate that, contrary to what is claimed by decentralized learning proposers, decentralized learning does not offer any security advantage over federated learning. Rather, it increases the attack surface enabling any user in the system to perform privacy attacks such as gradient inversion, and even gain full control over honest users' local model. We also show that, given the state of the art in protections, privacy-preserving configurations of decentralized learning require fully connected networks, losing any practical advantage over the federated setup and therefore completely defeating the objective of the decentralized approach.

摘要: 在这项工作中，我们进行了第一次，深入的，隐私分析的分散学习--一个合作的机器学习框架，旨在解决联合学习的主要限制。我们针对被动的和主动的分散攻击引入了一套新的攻击。我们证明，与去中心化学习提出者所声称的相反，去中心化学习并不比联邦学习提供任何安全优势。相反，它增加了攻击面，使系统中的任何用户都可以执行诸如梯度反转等隐私攻击，甚至获得对诚实用户的本地模型的完全控制。我们还表明，考虑到保护技术的最新水平，去中心化学习的隐私保护配置需要完全连接的网络，失去了与联邦设置相比的任何实际优势，因此完全违背了去中心化方法的目标。



## **8. Scale-MIA: A Scalable Model Inversion Attack against Secure Federated Learning via Latent Space Reconstruction**

Scale-MIA：一种基于潜在空间重构的安全联邦学习可扩展模型反转攻击 cs.LG

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2311.05808v1) [paper-pdf](http://arxiv.org/pdf/2311.05808v1)

**Authors**: Shanghao Shi, Ning Wang, Yang Xiao, Chaoyu Zhang, Yi Shi, Y. Thomas Hou, Wenjing Lou

**Abstract**: Federated learning is known for its capability to safeguard participants' data privacy. However, recently emerged model inversion attacks (MIAs) have shown that a malicious parameter server can reconstruct individual users' local data samples through model updates. The state-of-the-art attacks either rely on computation-intensive search-based optimization processes to recover each input batch, making scaling difficult, or they involve the malicious parameter server adding extra modules before the global model architecture, rendering the attacks too conspicuous and easily detectable.   To overcome these limitations, we propose Scale-MIA, a novel MIA capable of efficiently and accurately recovering training samples of clients from the aggregated updates, even when the system is under the protection of a robust secure aggregation protocol. Unlike existing approaches treating models as black boxes, Scale-MIA recognizes the importance of the intricate architecture and inner workings of machine learning models. It identifies the latent space as the critical layer for breaching privacy and decomposes the complex recovery task into an innovative two-step process to reduce computation complexity. The first step involves reconstructing the latent space representations (LSRs) from the aggregated model updates using a closed-form inversion mechanism, leveraging specially crafted adversarial linear layers. In the second step, the whole input batches are recovered from the LSRs by feeding them into a fine-tuned generative decoder.   We implemented Scale-MIA on multiple commonly used machine learning models and conducted comprehensive experiments across various settings. The results demonstrate that Scale-MIA achieves excellent recovery performance on different datasets, exhibiting high reconstruction rates, accuracy, and attack efficiency on a larger scale compared to state-of-the-art MIAs.

摘要: 联合学习以其保护参与者数据隐私的能力而闻名。然而，最近出现的模型反转攻击(MIA)表明，恶意参数服务器可以通过模型更新来重建个人用户的本地数据样本。最先进的攻击要么依赖于基于搜索的计算密集型优化过程来恢复每个输入批次，使扩展变得困难，要么涉及恶意参数服务器在全局模型体系结构之前添加额外的模块，使攻击过于显眼和容易检测。为了克服这些局限性，我们提出了Scale-MIA，一种新的MIA，即使在系统处于健壮的安全聚合协议的保护下，也能够从聚合的更新中高效而准确地恢复客户端的训练样本。与将模型视为黑盒的现有方法不同，Scale-MIA认识到机器学习模型复杂的体系结构和内部工作原理的重要性。它将潜在空间识别为侵犯隐私的关键层，并将复杂的恢复任务分解为一个创新的两步过程，以降低计算复杂度。第一步涉及使用闭合形式的反转机制从聚集的模型更新重构潜在空间表示(LSR)，利用特制的对抗性线性层。在第二步中，通过将输入批次馈送到微调的产生式解码器，从LSR中恢复整个输入批次。我们在多种常用的机器学习模型上实现了Scale-MIA，并在不同的环境下进行了全面的实验。结果表明，Scale-MIA在不同的数据集上表现出了良好的恢复性能，与现有的MIA相比，在更大的范围内表现出更高的重建率、准确性和攻击效率。



## **9. What Distributions are Robust to Indiscriminate Poisoning Attacks for Linear Learners?**

什么分布对线性学习者的不分青红皂白的中毒攻击是健壮的？ cs.LG

NeurIPS 2023 camera-ready version, 39 pages

**SubmitDate**: 2023-11-09    [abs](http://arxiv.org/abs/2307.01073v2) [paper-pdf](http://arxiv.org/pdf/2307.01073v2)

**Authors**: Fnu Suya, Xiao Zhang, Yuan Tian, David Evans

**Abstract**: We study indiscriminate poisoning for linear learners where an adversary injects a few crafted examples into the training data with the goal of forcing the induced model to incur higher test error. Inspired by the observation that linear learners on some datasets are able to resist the best known attacks even without any defenses, we further investigate whether datasets can be inherently robust to indiscriminate poisoning attacks for linear learners. For theoretical Gaussian distributions, we rigorously characterize the behavior of an optimal poisoning attack, defined as the poisoning strategy that attains the maximum risk of the induced model at a given poisoning budget. Our results prove that linear learners can indeed be robust to indiscriminate poisoning if the class-wise data distributions are well-separated with low variance and the size of the constraint set containing all permissible poisoning points is also small. These findings largely explain the drastic variation in empirical attack performance of the state-of-the-art poisoning attacks on linear learners across benchmark datasets, making an important initial step towards understanding the underlying reasons some learning tasks are vulnerable to data poisoning attacks.

摘要: 我们研究了线性学习器的无差别中毒，其中对手将一些精心制作的示例注入训练数据，目的是迫使诱导模型产生更高的测试错误。受一些数据集上的线性学习器即使没有任何防御也能够抵抗最知名的攻击的观察的启发，我们进一步研究了数据集是否可以对线性学习器的不分青红皂白的中毒攻击具有固有的鲁棒性。对于理论上的高斯分布，我们严格描述了最佳中毒攻击的行为，定义为在给定的中毒预算下达到诱导模型的最大风险的中毒策略。我们的研究结果证明，线性学习器可以确实是强大的不分青红皂白中毒，如果类明智的数据分布是分开的低方差和大小的约束集包含所有允许的中毒点也很小。这些发现在很大程度上解释了在基准数据集上对线性学习器的最先进的中毒攻击的经验攻击性能的巨大变化，这是理解一些学习任务容易受到数据中毒攻击的根本原因的重要的第一步。



## **10. Robust Retraining-free GAN Fingerprinting via Personalized Normalization**

基于个性化归一化的稳健的无需再训练的GaN指纹图谱 cs.CV

**SubmitDate**: 2023-11-09    [abs](http://arxiv.org/abs/2311.05478v1) [paper-pdf](http://arxiv.org/pdf/2311.05478v1)

**Authors**: Jianwei Fei, Zhihua Xia, Benedetta Tondi, Mauro Barni

**Abstract**: In recent years, there has been significant growth in the commercial applications of generative models, licensed and distributed by model developers to users, who in turn use them to offer services. In this scenario, there is a need to track and identify the responsible user in the presence of a violation of the license agreement or any kind of malicious usage. Although there are methods enabling Generative Adversarial Networks (GANs) to include invisible watermarks in the images they produce, generating a model with a different watermark, referred to as a fingerprint, for each user is time- and resource-consuming due to the need to retrain the model to include the desired fingerprint. In this paper, we propose a retraining-free GAN fingerprinting method that allows model developers to easily generate model copies with the same functionality but different fingerprints. The generator is modified by inserting additional Personalized Normalization (PN) layers whose parameters (scaling and bias) are generated by two dedicated shallow networks (ParamGen Nets) taking the fingerprint as input. A watermark decoder is trained simultaneously to extract the fingerprint from the generated images. The proposed method can embed different fingerprints inside the GAN by just changing the input of the ParamGen Nets and performing a feedforward pass, without finetuning or retraining. The performance of the proposed method in terms of robustness against both model-level and image-level attacks is also superior to the state-of-the-art.

摘要: 近年来，生成性模型的商业应用有了显著的增长，由模型开发者许可并分发给用户，用户反过来使用它们来提供服务。在这种情况下，需要在存在违反许可协议或任何类型的恶意使用的情况下跟踪和识别责任用户。尽管存在使生成性对抗网络(GAN)能够在其产生的图像中包括不可见水印的方法，但是为每个用户生成具有不同水印的模型(称为指纹)是耗时和耗费资源的，因为需要重新训练模型以包括所需的指纹。在本文中，我们提出了一种无需再训练的GaN指纹识别方法，该方法允许模型开发人员轻松地生成具有相同功能但不同指纹的模型副本。通过插入额外的个性化归一化(PN)层来修改生成器，这些层的参数(比例和偏差)由两个以指纹为输入的专用浅层网络(参数生成网络)生成。同时训练水印解码器从生成的图像中提取指纹。该方法只需改变参数生成网络的输入并执行前馈传递，即可在GaN中嵌入不同的指纹，而无需精调或重新训练。在抵抗模型级和图像级攻击方面，该方法的性能也优于现有的方法。



## **11. ABIGX: A Unified Framework for eXplainable Fault Detection and Classification**

ABIGX：一个可解释的故障检测和分类的统一框架 cs.LG

**SubmitDate**: 2023-11-09    [abs](http://arxiv.org/abs/2311.05316v1) [paper-pdf](http://arxiv.org/pdf/2311.05316v1)

**Authors**: Yue Zhuo, Jinchuan Qian, Zhihuan Song, Zhiqiang Ge

**Abstract**: For explainable fault detection and classification (FDC), this paper proposes a unified framework, ABIGX (Adversarial fault reconstruction-Based Integrated Gradient eXplanation). ABIGX is derived from the essentials of previous successful fault diagnosis methods, contribution plots (CP) and reconstruction-based contribution (RBC). It is the first explanation framework that provides variable contributions for the general FDC models. The core part of ABIGX is the adversarial fault reconstruction (AFR) method, which rethinks the FR from the perspective of adversarial attack and generalizes to fault classification models with a new fault index. For fault classification, we put forward a new problem of fault class smearing, which intrinsically hinders the correct explanation. We prove that ABIGX effectively mitigates this problem and outperforms the existing gradient-based explanation methods. For fault detection, we theoretically bridge ABIGX with conventional fault diagnosis methods by proving that CP and RBC are the linear specifications of ABIGX. The experiments evaluate the explanations of FDC by quantitative metrics and intuitive illustrations, the results of which show the general superiority of ABIGX to other advanced explanation methods.

摘要: 对于可解释故障检测与分类，本文提出了一个统一的框架ABIGX(基于对抗性故障重构的集成梯度解释)。ABIGX是从以往成功的故障诊断方法、贡献图(CP)和基于重构的贡献(RBC)的基础上发展而来的。它是第一个为一般FDC模型提供可变贡献的解释框架。ABIGX的核心部分是对抗性故障重构(AFR)方法，它从对抗性攻击的角度重新考虑了对抗性故障重构，并将其推广到具有新的故障指数的故障分类模型。对于故障分类，我们提出了一个新的故障类模糊问题，这本质上阻碍了正确的解释。我们证明了ABIGX有效地缓解了这一问题，并优于现有的基于梯度的解释方法。在故障检测方面，通过证明CP和RBC是ABIGX的线性规范，从理论上将ABIGX与传统的故障诊断方法联系起来。实验通过量化度量和直观的图示对FDC的解释进行了评估，结果显示了ABIGX相对于其他高级解释方法的总体优势。



## **12. Beyond Pretrained Features: Noisy Image Modeling Provides Adversarial Defense**

超越预先训练的特征：噪声图像建模提供对抗性防御 cs.CV

NeurIPS 2023

**SubmitDate**: 2023-11-09    [abs](http://arxiv.org/abs/2302.01056v3) [paper-pdf](http://arxiv.org/pdf/2302.01056v3)

**Authors**: Zunzhi You, Daochang Liu, Bohyung Han, Chang Xu

**Abstract**: Recent advancements in masked image modeling (MIM) have made it a prevailing framework for self-supervised visual representation learning. The MIM pretrained models, like most deep neural network methods, remain vulnerable to adversarial attacks, limiting their practical application, and this issue has received little research attention. In this paper, we investigate how this powerful self-supervised learning paradigm can provide adversarial robustness to downstream classifiers. During the exploration, we find that noisy image modeling (NIM), a simple variant of MIM that adopts denoising as the pre-text task, reconstructs noisy images surprisingly well despite severe corruption. Motivated by this observation, we propose an adversarial defense method, referred to as De^3, by exploiting the pretrained decoder for denoising. Through De^3, NIM is able to enhance adversarial robustness beyond providing pretrained features. Furthermore, we incorporate a simple modification, sampling the noise scale hyperparameter from random distributions, and enable the defense to achieve a better and tunable trade-off between accuracy and robustness. Experimental results demonstrate that, in terms of adversarial robustness, NIM is superior to MIM thanks to its effective denoising capability. Moreover, the defense provided by NIM achieves performance on par with adversarial training while offering the extra tunability advantage. Source code and models are available at https://github.com/youzunzhi/NIM-AdvDef.

摘要: 蒙版图像建模(MIM)的最新进展使其成为自监督视觉表征学习的主流框架。与大多数深度神经网络方法一样，MIM预训练模型仍然容易受到对手攻击，限制了其实际应用，而且这个问题几乎没有受到研究的关注。在本文中，我们研究了这种强大的自我监督学习范例如何为下游分类器提供对抗性稳健性。在探索过程中，我们发现噪声图像建模(NIM)是MIM的一种简单变体，它采用去噪作为文本前任务，尽管存在严重的破坏，但重建噪声图像的效果出人意料地好。基于这一观察结果，我们提出了一种对抗性防御方法，称为De^3，利用预先训练的解码器进行去噪。通过de^3，NIM能够增强对手的健壮性，而不仅仅是提供预先训练的功能。此外，我们结合了一个简单的修改，从随机分布中采样噪声尺度超参数，使防御能够在精度和稳健性之间实现更好和可调的折衷。实验结果表明，在对抗鲁棒性方面，NIM由于其有效的去噪能力而优于MIM。此外，NIM提供的防守在提供额外的可调性优势的同时，实现了与对手训练同等的性能。源代码和模型可在https://github.com/youzunzhi/NIM-AdvDef.上获得



## **13. Counter-Empirical Attacking based on Adversarial Reinforcement Learning for Time-Relevant Scoring System**

基于对抗性强化学习的时间相关评分系统的反经验攻击 cs.LG

submitted to TKDE on 08-Jun-2022, receive the 1st round decision  (major revision) on 20-Apr-2023, submitted to TKDE 2nd time on 30-May-2023,  receive the 2nd round decision (major revision) on 30-Sep-2023, submitted to  TKDE 3rd time on 15-Oct-2023, now under review for the 3rd round of reviewing

**SubmitDate**: 2023-11-09    [abs](http://arxiv.org/abs/2311.05144v1) [paper-pdf](http://arxiv.org/pdf/2311.05144v1)

**Authors**: Xiangguo Sun, Hong Cheng, Hang Dong, Bo Qiao, Si Qin, Qingwei Lin

**Abstract**: Scoring systems are commonly seen for platforms in the era of big data. From credit scoring systems in financial services to membership scores in E-commerce shopping platforms, platform managers use such systems to guide users towards the encouraged activity pattern, and manage resources more effectively and more efficiently thereby. To establish such scoring systems, several "empirical criteria" are firstly determined, followed by dedicated top-down design for each factor of the score, which usually requires enormous effort to adjust and tune the scoring function in the new application scenario. What's worse, many fresh projects usually have no ground-truth or any experience to evaluate a reasonable scoring system, making the designing even harder. To reduce the effort of manual adjustment of the scoring function in every new scoring system, we innovatively study the scoring system from the preset empirical criteria without any ground truth, and propose a novel framework to improve the system from scratch. In this paper, we propose a "counter-empirical attacking" mechanism that can generate "attacking" behavior traces and try to break the empirical rules of the scoring system. Then an adversarial "enhancer" is applied to evaluate the scoring system and find the improvement strategy. By training the adversarial learning problem, a proper scoring function can be learned to be robust to the attacking activity traces that are trying to violate the empirical criteria. Extensive experiments have been conducted on two scoring systems including a shared computing resource platform and a financial credit system. The experimental results have validated the effectiveness of our proposed framework.

摘要: 在大数据时代，评分系统在平台上很常见。从金融服务的信用评分系统到电子商务购物平台的会员评分，平台管理者使用这些系统来引导用户走向鼓励活动模式，从而更有效和更高效地管理资源。要建立这样的评分系统，首先要确定几个“经验标准”，然后针对评分的每个因素进行专门的自上而下的设计，这通常需要付出巨大的努力来调整和调整新的应用场景中的评分函数。更糟糕的是，许多新项目通常没有实际情况或任何经验来评估合理的评分系统，这使得设计变得更加困难。为了减少每个新评分系统中人工调整评分函数的工作量，我们创新性地从预设的经验标准出发，在没有任何基础事实的情况下对评分系统进行研究，并提出了一个新的框架来从头开始改进该系统。在本文中，我们提出了一种可以产生攻击行为痕迹的反经验攻击机制，试图打破评分系统的经验规则。然后应用一个对抗性的“增强器”对评分系统进行评估，并找到改进策略。通过训练对抗性学习问题，可以学习适当的得分函数，以对试图违反经验标准的攻击活动痕迹具有健壮性。在包括共享计算资源平台和金融信用系统在内的两个评分系统上进行了广泛的实验。实验结果验证了该框架的有效性。



## **14. Byzantine-Robust Distributed Online Learning: Taming Adversarial Participants in An Adversarial Environment**

拜占庭-稳健的分布式在线学习：在对抗性环境中驯服对抗性参与者 cs.LG

**SubmitDate**: 2023-11-09    [abs](http://arxiv.org/abs/2307.07980v2) [paper-pdf](http://arxiv.org/pdf/2307.07980v2)

**Authors**: Xingrong Dong, Zhaoxian Wu, Qing Ling, Zhi Tian

**Abstract**: This paper studies distributed online learning under Byzantine attacks. The performance of an online learning algorithm is often characterized by (adversarial) regret, which evaluates the quality of one-step-ahead decision-making when an environment provides adversarial losses, and a sublinear bound is preferred. But we prove that, even with a class of state-of-the-art robust aggregation rules, in an adversarial environment and in the presence of Byzantine participants, distributed online gradient descent can only achieve a linear adversarial regret bound, which is tight. This is the inevitable consequence of Byzantine attacks, even though we can control the constant of the linear adversarial regret to a reasonable level. Interestingly, when the environment is not fully adversarial so that the losses of the honest participants are i.i.d. (independent and identically distributed), we show that sublinear stochastic regret, in contrast to the aforementioned adversarial regret, is possible. We develop a Byzantine-robust distributed online momentum algorithm to attain such a sublinear stochastic regret bound. Extensive numerical experiments corroborate our theoretical analysis.

摘要: 本文研究了拜占庭攻击下的分布式在线学习。在线学习算法的性能通常以（对抗性）后悔为特征，当环境提供对抗性损失时，它评估一步决策的质量，并且次线性界限是首选。但是我们证明了，即使有一类最先进的鲁棒聚集规则，在对抗环境中，在拜占庭参与者的存在下，分布式在线梯度下降只能实现线性对抗遗憾界，这是紧的。这是拜占庭攻击的必然结果，即使我们可以控制线性对抗性后悔的常数到一个合理的水平。有趣的是，当环境不是完全对抗时，诚实参与者的损失是独立同分布的。（独立同分布），我们表明，次线性随机遗憾，与上述对抗性遗憾，是可能的。我们开发了一个拜占庭鲁棒分布式在线动量算法，以达到这样的次线性随机遗憾界。大量的数值实验证实了我们的理论分析。



## **15. Familiarity-Based Open-Set Recognition Under Adversarial Attacks**

对抗性攻击下基于熟悉度的开集识别 cs.CV

Published in: The 2nd Workshop and Challenges for Out-of-Distribution  Generalization in Computer Vision, ICCV 2023

**SubmitDate**: 2023-11-08    [abs](http://arxiv.org/abs/2311.05006v1) [paper-pdf](http://arxiv.org/pdf/2311.05006v1)

**Authors**: Philip Enevoldsen, Christian Gundersen, Nico Lang, Serge Belongie, Christian Igel

**Abstract**: Open-set recognition (OSR), the identification of novel categories, can be a critical component when deploying classification models in real-world applications. Recent work has shown that familiarity-based scoring rules such as the Maximum Softmax Probability (MSP) or the Maximum Logit Score (MLS) are strong baselines when the closed-set accuracy is high. However, one of the potential weaknesses of familiarity-based OSR are adversarial attacks. Here, we present gradient-based adversarial attacks on familiarity scores for both types of attacks, False Familiarity and False Novelty attacks, and evaluate their effectiveness in informed and uninformed settings on TinyImageNet.

摘要: 开放集识别(OSR)是对新类别的识别，在现实应用中部署分类模型时可能是一个关键组件。最近的工作表明，当封闭集精度较高时，基于熟悉度的评分规则，如最大软最大概率(MSP)或最大Logit分数(MLS)，是很强的基线。然而，基于熟悉度的OSR的一个潜在弱点是对抗性攻击。在这里，我们提出了基于梯度的敌意攻击，针对两种类型的攻击，虚假熟悉和虚假新奇攻击，并在TinyImageNet上评估了它们在知情和不知情环境下的有效性。



## **16. Be Careful When Evaluating Explanations Regarding Ground Truth**

评估有关基本事实的解释时要小心 cs.CV

**SubmitDate**: 2023-11-08    [abs](http://arxiv.org/abs/2311.04813v1) [paper-pdf](http://arxiv.org/pdf/2311.04813v1)

**Authors**: Hubert Baniecki, Maciej Chrabaszcz, Andreas Holzinger, Bastian Pfeifer, Anna Saranti, Przemyslaw Biecek

**Abstract**: Evaluating explanations of image classifiers regarding ground truth, e.g. segmentation masks defined by human perception, primarily evaluates the quality of the models under consideration rather than the explanation methods themselves. Driven by this observation, we propose a framework for $\textit{jointly}$ evaluating the robustness of safety-critical systems that $\textit{combine}$ a deep neural network with an explanation method. These are increasingly used in real-world applications like medical image analysis or robotics. We introduce a fine-tuning procedure to (mis)align model$\unicode{x2013}$explanation pipelines with ground truth and use it to quantify the potential discrepancy between worst and best-case scenarios of human alignment. Experiments across various model architectures and post-hoc local interpretation methods provide insights into the robustness of vision transformers and the overall vulnerability of such AI systems to potential adversarial attacks.

摘要: 评价图像分类器关于地面事实的解释，例如由人的感知定义的分割掩模，主要是评价所考虑的模型的质量，而不是解释方法本身。在此基础上，我们提出了一种将深层神经网络与解释方法相结合的安全关键系统健壮性评价框架。它们越来越多地被用于医学图像分析或机器人等现实世界的应用中。我们引入了一个微调过程来(错误地)将模型$\Unicode{x2013}$解释管道与地面事实对齐，并使用它来量化人类对齐的最坏情况和最好情况之间的潜在差异。通过各种模型架构和事后本地解释方法的实验，可以深入了解视觉转换器的稳健性，以及此类人工智能系统对潜在对手攻击的整体脆弱性。



## **17. VLAttack: Multimodal Adversarial Attacks on Vision-Language Tasks via Pre-trained Models**

VLAttack：通过预先训练的模型对视觉语言任务进行多模式对抗性攻击 cs.CR

Accepted by NeurIPS 2023

**SubmitDate**: 2023-11-08    [abs](http://arxiv.org/abs/2310.04655v2) [paper-pdf](http://arxiv.org/pdf/2310.04655v2)

**Authors**: Ziyi Yin, Muchao Ye, Tianrong Zhang, Tianyu Du, Jinguo Zhu, Han Liu, Jinghui Chen, Ting Wang, Fenglong Ma

**Abstract**: Vision-Language (VL) pre-trained models have shown their superiority on many multimodal tasks. However, the adversarial robustness of such models has not been fully explored. Existing approaches mainly focus on exploring the adversarial robustness under the white-box setting, which is unrealistic. In this paper, we aim to investigate a new yet practical task to craft image and text perturbations using pre-trained VL models to attack black-box fine-tuned models on different downstream tasks. Towards this end, we propose VLAttack to generate adversarial samples by fusing perturbations of images and texts from both single-modal and multimodal levels. At the single-modal level, we propose a new block-wise similarity attack (BSA) strategy to learn image perturbations for disrupting universal representations. Besides, we adopt an existing text attack strategy to generate text perturbations independent of the image-modal attack. At the multimodal level, we design a novel iterative cross-search attack (ICSA) method to update adversarial image-text pairs periodically, starting with the outputs from the single-modal level. We conduct extensive experiments to attack three widely-used VL pretrained models for six tasks on eight datasets. Experimental results show that the proposed VLAttack framework achieves the highest attack success rates on all tasks compared with state-of-the-art baselines, which reveals a significant blind spot in the deployment of pre-trained VL models. Codes will be released soon.

摘要: 视觉-语言(VL)预训练模型在许多多通道任务中显示了其优越性。然而，这类模型的对抗性健壮性还没有得到充分的研究。现有的研究方法主要集中在研究白盒环境下的对抗稳健性，这是不现实的。在本文中，我们的目标是研究一种新的但实用的任务，使用预先训练的VL模型来攻击不同下游任务的黑盒微调模型来制造图像和文本扰动。为此，我们提出了VLAttack，通过从单模和多模两个层次融合图像和文本的扰动来生成对抗性样本。在单模式层面，我们提出了一种新的分块相似性攻击(BSA)策略来学习图像扰动以破坏通用表示。此外，我们采用了一种现有的文本攻击策略来产生独立于图像模式攻击的文本扰动。在多模态级，我们设计了一种新的迭代交叉搜索攻击(ICSA)方法，从单模式级的输出开始，周期性地更新敌方图文对。我们进行了广泛的实验来攻击三个广泛使用的VL预训练模型，在八个数据集上执行六个任务。实验结果表明，VLAttack框架在所有任务上获得了最高的攻击成功率，这揭示了预先训练的VL模型的部署存在明显的盲点。代码很快就会发布。



## **18. Evading Watermark based Detection of AI-Generated Content**

基于规避水印的人工智能生成内容检测 cs.LG

To appear in ACM Conference on Computer and Communications Security  (CCS), 2023

**SubmitDate**: 2023-11-08    [abs](http://arxiv.org/abs/2305.03807v5) [paper-pdf](http://arxiv.org/pdf/2305.03807v5)

**Authors**: Zhengyuan Jiang, Jinghuai Zhang, Neil Zhenqiang Gong

**Abstract**: A generative AI model can generate extremely realistic-looking content, posing growing challenges to the authenticity of information. To address the challenges, watermark has been leveraged to detect AI-generated content. Specifically, a watermark is embedded into an AI-generated content before it is released. A content is detected as AI-generated if a similar watermark can be decoded from it. In this work, we perform a systematic study on the robustness of such watermark-based AI-generated content detection. We focus on AI-generated images. Our work shows that an attacker can post-process a watermarked image via adding a small, human-imperceptible perturbation to it, such that the post-processed image evades detection while maintaining its visual quality. We show the effectiveness of our attack both theoretically and empirically. Moreover, to evade detection, our adversarial post-processing method adds much smaller perturbations to AI-generated images and thus better maintain their visual quality than existing popular post-processing methods such as JPEG compression, Gaussian blur, and Brightness/Contrast. Our work shows the insufficiency of existing watermark-based detection of AI-generated content, highlighting the urgent needs of new methods. Our code is publicly available: https://github.com/zhengyuan-jiang/WEvade.

摘要: 生成性人工智能模型可以生成极其逼真的内容，对信息的真实性提出了越来越大的挑战。为了应对这些挑战，水印被用来检测人工智能生成的内容。具体地说，水印在发布之前被嵌入到人工智能生成的内容中。如果可以从内容中解码类似的水印，则该内容被检测为人工智能生成的内容。在这项工作中，我们对这种基于水印的人工智能内容检测的稳健性进行了系统的研究。我们专注于人工智能生成的图像。我们的工作表明，攻击者可以通过在水印图像上添加一个人类无法察觉的小扰动来对其进行后处理，从而在保持其视觉质量的同时逃避检测。我们从理论和经验上证明了我们的攻击的有效性。此外，为了逃避检测，我们的对抗性后处理方法向人工智能生成的图像添加了更小的扰动，从而比现有的流行的后处理方法，如JPEG压缩、高斯模糊和亮度/对比度，更好地保持了它们的视觉质量。我们的工作显示了现有基于水印的人工智能生成内容检测的不足，突出了对新方法的迫切需求。我们的代码是公开提供的：https://github.com/zhengyuan-jiang/WEvade.



## **19. Army of Thieves: Enhancing Black-Box Model Extraction via Ensemble based sample selection**

小偷大军：通过基于集成的样本选择增强黑盒模型提取 cs.LG

10 pages, 5 figures, paper accepted to WACV 2024

**SubmitDate**: 2023-11-08    [abs](http://arxiv.org/abs/2311.04588v1) [paper-pdf](http://arxiv.org/pdf/2311.04588v1)

**Authors**: Akshit Jindal, Vikram Goyal, Saket Anand, Chetan Arora

**Abstract**: Machine Learning (ML) models become vulnerable to Model Stealing Attacks (MSA) when they are deployed as a service. In such attacks, the deployed model is queried repeatedly to build a labelled dataset. This dataset allows the attacker to train a thief model that mimics the original model. To maximize query efficiency, the attacker has to select the most informative subset of data points from the pool of available data. Existing attack strategies utilize approaches like Active Learning and Semi-Supervised learning to minimize costs. However, in the black-box setting, these approaches may select sub-optimal samples as they train only one thief model. Depending on the thief model's capacity and the data it was pretrained on, the model might even select noisy samples that harm the learning process. In this work, we explore the usage of an ensemble of deep learning models as our thief model. We call our attack Army of Thieves(AOT) as we train multiple models with varying complexities to leverage the crowd's wisdom. Based on the ensemble's collective decision, uncertain samples are selected for querying, while the most confident samples are directly included in the training data. Our approach is the first one to utilize an ensemble of thief models to perform model extraction. We outperform the base approaches of existing state-of-the-art methods by at least 3% and achieve a 21% higher adversarial sample transferability than previous work for models trained on the CIFAR-10 dataset.

摘要: 机器学习(ML)模型在作为服务部署时容易受到模型窃取攻击(MSA)。在此类攻击中，部署的模型被重复查询以构建标记的数据集。此数据集允许攻击者训练模仿原始模型的窃贼模型。为了最大限度地提高查询效率，攻击者必须从可用数据池中选择信息最丰富的数据点子集。现有的攻击策略使用主动学习和半监督学习等方法来最小化成本。然而，在黑盒设置中，这些方法可能会选择次优样本，因为它们只训练一个窃贼模型。根据窃贼模型的容量和预先训练的数据，该模型甚至可能选择影响学习过程的噪声样本。在这项工作中，我们探索了一组深度学习模型作为我们的小偷模型的使用。我们称我们的攻击为盗贼大军(AOT)，因为我们训练了多种复杂程度不同的模型，以利用人群的智慧。基于集成的集体决策，选择不确定的样本进行查询，而最有信心的样本直接包含在训练数据中。我们的方法是第一个利用小偷模型集合来执行模型提取的方法。我们的性能比现有最先进方法的基本方法至少高出3%，并且比以前在CIFAR-10数据集上训练的模型的对抗性样本可转移性高21%。



## **20. Constrained Adaptive Attacks: Realistic Evaluation of Adversarial Examples and Robust Training of Deep Neural Networks for Tabular Data**

受限自适应攻击：对抗性实例的真实评估和表格数据的深度神经网络的稳健训练 cs.LG

**SubmitDate**: 2023-11-08    [abs](http://arxiv.org/abs/2311.04503v1) [paper-pdf](http://arxiv.org/pdf/2311.04503v1)

**Authors**: Thibault Simonetto, Salah Ghamizi, Antoine Desjardins, Maxime Cordy, Yves Le Traon

**Abstract**: State-of-the-art deep learning models for tabular data have recently achieved acceptable performance to be deployed in industrial settings. However, the robustness of these models remains scarcely explored. Contrary to computer vision, there is to date no realistic protocol to properly evaluate the adversarial robustness of deep tabular models due to intrinsic properties of tabular data such as categorical features, immutability, and feature relationship constraints. To fill this gap, we propose CAA, the first efficient evasion attack for constrained tabular deep learning models. CAA is an iterative parameter-free attack that combines gradient and search attacks to generate adversarial examples under constraints. We leverage CAA to build a benchmark of deep tabular models across three popular use cases: credit scoring, phishing and botnet attacks detection. Our benchmark supports ten threat models with increasing capabilities of the attacker, and reflects real-world attack scenarios for each use case. Overall, our results demonstrate how domain knowledge, adversarial training, and attack budgets impact the robustness assessment of deep tabular models and provide security practitioners with a set of recommendations to improve the robustness of deep tabular models against various evasion attack scenarios.

摘要: 最先进的表格数据深度学习模型最近已经达到了可接受的性能，可以部署在工业环境中。然而，这些模型的鲁棒性仍然很少探索。与计算机视觉相反，由于表格数据的固有属性，例如分类特征，不变性和特征关系约束，迄今为止还没有现实的协议来正确评估深度表格模型的对抗鲁棒性。为了填补这一空白，我们提出了CAA，这是第一个针对受限表格深度学习模型的有效规避攻击。CAA是一种迭代的无参数攻击，它结合了梯度和搜索攻击，以在约束条件下生成对抗性示例。我们利用CAA在三个流行的用例中构建深度表格模型的基准：信用评分，网络钓鱼和僵尸网络攻击检测。我们的基准测试支持10种攻击者能力不断增强的威胁模型，并反映了每个用例的真实攻击场景。总的来说，我们的研究结果展示了领域知识、对抗性训练和攻击预算如何影响深度表格模型的鲁棒性评估，并为安全从业人员提供了一组建议，以提高深度表格模型对各种规避攻击场景的鲁棒性。



## **21. SyncBleed: A Realistic Threat Model and Mitigation Strategy for Zero-Involvement Pairing and Authentication (ZIPA)**

SyncBleed：一种现实的零参与配对和认证(ZIPA)威胁模型及缓解策略 cs.CR

**SubmitDate**: 2023-11-08    [abs](http://arxiv.org/abs/2311.04433v1) [paper-pdf](http://arxiv.org/pdf/2311.04433v1)

**Authors**: Isaac Ahlgren, Jack West, Kyuin Lee, George K. Thiruvathukal, Neil Klingensmith

**Abstract**: Zero Involvement Pairing and Authentication (ZIPA) is a promising technique for auto-provisioning large networks of Internet-of-Things (IoT) devices. Presently, these networks use password-based authentication, which is difficult to scale to more than a handful of devices. To deal with this challenge, ZIPA enabled devices autonomously extract identical authentication or encryption keys from ambient environmental signals. However, during the key negotiation process, existing ZIPA systems leak information on a public wireless channel which can allow adversaries to learn the key. We demonstrate a passive attack called SyncBleed, which uses leaked information to reconstruct keys generated by ZIPA systems. To mitigate SyncBleed, we present TREVOR, an improved key generation technique that produces nearly identical bit sequences from environmental signals without leaking information. We demonstrate that TREVOR can generate keys from a variety of environmental signal types under 4 seconds, consistently achieving a 90-95% bit agreement rate across devices within various environmental sources.

摘要: 零参与配对和认证(ZIPA)是自动配置大型物联网(IoT)设备网络的一种很有前途的技术。目前，这些网络使用基于密码的身份验证，很难扩展到几台以上的设备。为了应对这一挑战，支持ZIPA的设备会自动从环境环境信号中提取相同的身份验证或加密密钥。然而，在密钥协商过程中，现有的ZIPA系统在公共无线信道上泄漏信息，从而允许攻击者获取密钥。我们演示了一种名为SyncBleed的被动攻击，该攻击利用泄漏的信息来重建ZIPA系统生成的密钥。为了缓解SyncBleed，我们提出了一种改进的密钥生成技术Trevor，它可以从环境信号中产生几乎相同的比特序列，而不会泄露信息。我们证明，Trevor可以在4秒内从各种环境信号类型生成密钥，在各种环境来源的设备上一致实现90%-95%的比特符合率。



## **22. Unveiling Safety Vulnerabilities of Large Language Models**

揭开大型语言模型的安全漏洞 cs.CL

To be published in GEM workshop. Conference on Empirical Methods in  Natural Language Processing (EMNLP). 2023

**SubmitDate**: 2023-11-07    [abs](http://arxiv.org/abs/2311.04124v1) [paper-pdf](http://arxiv.org/pdf/2311.04124v1)

**Authors**: George Kour, Marcel Zalmanovici, Naama Zwerdling, Esther Goldbraich, Ora Nova Fandina, Ateret Anaby-Tavor, Orna Raz, Eitan Farchi

**Abstract**: As large language models become more prevalent, their possible harmful or inappropriate responses are a cause for concern. This paper introduces a unique dataset containing adversarial examples in the form of questions, which we call AttaQ, designed to provoke such harmful or inappropriate responses. We assess the efficacy of our dataset by analyzing the vulnerabilities of various models when subjected to it. Additionally, we introduce a novel automatic approach for identifying and naming vulnerable semantic regions - input semantic areas for which the model is likely to produce harmful outputs. This is achieved through the application of specialized clustering techniques that consider both the semantic similarity of the input attacks and the harmfulness of the model's responses. Automatically identifying vulnerable semantic regions enhances the evaluation of model weaknesses, facilitating targeted improvements to its safety mechanisms and overall reliability.

摘要: 随着大型语言模型变得越来越普遍，它们可能带来的有害或不恰当的反应令人担忧。本文介绍了一种独特的数据集，它以问题的形式包含了对抗性的例子，我们称之为Attaq，旨在引起这种有害或不适当的反应。我们通过分析各种模型在受到其影响时的漏洞来评估我们的数据集的有效性。此外，我们引入了一种新的自动方法来识别和命名易受攻击的语义区--模型可能产生有害输出的输入语义区。这是通过应用专门的集群技术来实现的，该技术同时考虑了输入攻击的语义相似性和模型响应的危害性。自动识别易受攻击的语义区域增强了对模型弱点的评估，促进了对其安全机制和总体可靠性的有针对性的改进。



## **23. A Lightweight and Secure PUF-Based Authentication and Key-exchange Protocol for IoT Devices**

一种基于PUF的轻量级安全物联网设备认证和密钥交换协议 cs.CR

**SubmitDate**: 2023-11-07    [abs](http://arxiv.org/abs/2311.04078v1) [paper-pdf](http://arxiv.org/pdf/2311.04078v1)

**Authors**: Chandranshu Gupta, Gaurav Varshney

**Abstract**: The Internet of Things (IoT) has improved people's lives by seamlessly integrating into many facets of modern life and facilitating information sharing across platforms. Device Authentication and Key exchange are major challenges for the IoT. High computational resource requirements for cryptographic primitives and message transmission during Authentication make the existing methods like PKI and IBE not suitable for these resource constrained devices. PUF appears to offer a practical and economical security mechanism in place of typically sophisticated cryptosystems like PKI and IBE. PUF provides an unclonable and tamper sensitive unique signature based on the PUF chip by using manufacturing process variability. Therefore, in this study, we use lightweight bitwise XOR, hash function, and PUF to Authenticate IoT devices. Despite several studies employing the PUF to authenticate communication between IoT devices, to the authors' knowledge, existing solutions require intermediary gateway and internet capabilities by the IoT device to directly interact with a Server for Authentication and hence, are not scalable when the IoT device works on different technologies like BLE, Zigbee, etc. To address the aforementioned issue, we present a system in which the IoT device does not require a continuous active internet connection to communicate with the server in order to Authenticate itself. The results of a thorough security study are validated against adversarial attacks and PUF modeling attacks. For formal security validation, the AVISPA verification tool is also used. Performance study recommends this protocol's lightweight characteristics. The proposed protocol's acceptability and defenses against adversarial assaults are supported by a prototype developed with ESP32.

摘要: 物联网(IoT)无缝融入现代生活的方方面面，促进跨平台信息共享，改善了人们的生活。设备身份验证和密钥交换是物联网面临的主要挑战。认证过程中对加密原语和消息传输的高计算资源要求使得现有的PKI和IBE等方法不适合这些资源受限的设备。PUF似乎提供了一种实用而经济的安全机制，取代了PKI和IBE等典型的复杂密码系统。PUF通过使用制造工艺可变性，基于PUF芯片提供了不可克隆和篡改敏感的唯一签名。因此，在本研究中，我们使用轻量级按位异或、哈希函数和PUF对物联网设备进行身份验证。尽管有几项研究使用PUF来认证物联网设备之间的通信，但据作者所知，现有的解决方案需要物联网设备的中间网关和互联网能力来直接与服务器进行交互以进行认证，因此，当物联网设备在不同的技术(如BLE、Zigbee等)上工作时，该解决方案是不可扩展的。为了解决上述问题，我们提供了一种系统，其中物联网设备不需要连续的活动互联网连接来与服务器通信以认证其自身。深入的安全研究结果在对抗攻击和PUF建模攻击中得到了验证。对于正式的安全验证，还使用了Avispa验证工具。性能研究推荐该协议的轻量级特性。用ESP32开发的原型支持了所提出的协议的可接受性和对对手攻击的防御。



## **24. Structural Causal Models Reveal Confounder Bias in Linear Program Modelling**

结构因果模型揭示线性规划建模中的共同创始人偏差 cs.LG

Published at the 15th Asian Conference on Machine Learning (ACML  2023) Journal Track. Main paper: 19 pages, References: 2 pages, Supplement:  .5 page. Main paper: 3 figures, 3 tables, Supplement: 1 table

**SubmitDate**: 2023-11-07    [abs](http://arxiv.org/abs/2105.12697v6) [paper-pdf](http://arxiv.org/pdf/2105.12697v6)

**Authors**: Matej Zečević, Devendra Singh Dhami, Kristian Kersting

**Abstract**: The recent years have been marked by extended research on adversarial attacks, especially on deep neural networks. With this work we intend on posing and investigating the question of whether the phenomenon might be more general in nature, that is, adversarial-style attacks outside classical classification tasks. Specifically, we investigate optimization problems as they constitute a fundamental part of modern AI research. To this end, we consider the base class of optimizers namely Linear Programs (LPs). On our initial attempt of a na\"ive mapping between the formalism of adversarial examples and LPs, we quickly identify the key ingredients missing for making sense of a reasonable notion of adversarial examples for LPs. Intriguingly, the formalism of Pearl's notion to causality allows for the right description of adversarial like examples for LPs. Characteristically, we show the direct influence of the Structural Causal Model (SCM) onto the subsequent LP optimization, which ultimately exposes a notion of confounding in LPs (inherited by said SCM) that allows for adversarial-style attacks. We provide both the general proof formally alongside existential proofs of such intriguing LP-parameterizations based on SCM for three combinatorial problems, namely Linear Assignment, Shortest Path and a real world problem of energy systems.

摘要: 近年来，对抗性攻击，特别是深度神经网络的研究不断深入。通过这项工作，我们打算提出和调查这一现象是否在本质上更一般，即经典分类任务之外的对抗性攻击。具体地说，我们研究优化问题，因为它们构成了现代人工智能研究的基本部分。为此，我们考虑了优化器的基类，即线性规划(LP)。在我们最初尝试在对抗性例子的形式主义和有限合伙人之间建立一个朴素的映射时，我们很快就确定了理解有限合伙人对抗性例子的合理概念所缺少的关键因素。有趣的是，珀尔对因果关系概念的形式主义允许正确地描述有限责任公司的对抗性例子。在特征上，我们展示了结构因果模型(SCM)对随后的LP优化的直接影响，这最终暴露了LP中的混淆概念(由所述SCM继承)，允许对抗性风格的攻击。对于三个组合问题，即线性分配问题、最短路问题和一个真实世界的能量系统问题，我们给出了基于SCM的这类有趣的LP-参数化的形式证明和存在性证明。



## **25. FD-MIA: Efficient Attacks on Fairness-enhanced Models**

FD-MIA：对公平性增强模型的有效攻击 cs.LG

Under review

**SubmitDate**: 2023-11-07    [abs](http://arxiv.org/abs/2311.03865v1) [paper-pdf](http://arxiv.org/pdf/2311.03865v1)

**Authors**: Huan Tian, Guangsheng Zhang, Bo Liu, Tianqing Zhu, Ming Ding, Wanlei Zhou

**Abstract**: Previous studies have developed fairness methods for biased models that exhibit discriminatory behaviors towards specific subgroups. While these models have shown promise in achieving fair predictions, recent research has identified their potential vulnerability to score-based membership inference attacks (MIAs). In these attacks, adversaries can infer whether a particular data sample was used during training by analyzing the model's prediction scores. However, our investigations reveal that these score-based MIAs are ineffective when targeting fairness-enhanced models in binary classifications. The attack models trained to launch the MIAs degrade into simplistic threshold models, resulting in lower attack performance. Meanwhile, we observe that fairness methods often lead to prediction performance degradation for the majority subgroups of the training data. This raises the barrier to successful attacks and widens the prediction gaps between member and non-member data. Building upon these insights, we propose an efficient MIA method against fairness-enhanced models based on fairness discrepancy results (FD-MIA). It leverages the difference in the predictions from both the original and fairness-enhanced models and exploits the observed prediction gaps as attack clues. We also explore potential strategies for mitigating privacy leakages. Extensive experiments validate our findings and demonstrate the efficacy of the proposed method.

摘要: 以前的研究已经为对特定子组表现出歧视性行为的有偏见的模型开发了公平方法。虽然这些模型在实现公平预测方面表现出了希望，但最近的研究发现，它们在基于分数的成员关系推理攻击(MIA)中具有潜在的脆弱性。在这些攻击中，攻击者可以通过分析模型的预测分数来推断训练期间是否使用了特定的数据样本。然而，我们的研究表明，这些基于分数的MIA在针对二进制分类中的公平性增强模型时是无效的。被训练来发起MIA的攻击模型降级为简单的阈值模型，导致较低的攻击性能。同时，我们观察到公平性方法经常导致训练数据的大多数子组的预测性能下降。这提高了成功攻击的障碍，并扩大了成员和非成员数据之间的预测差距。在此基础上，我们提出了一种针对基于公平性差异结果的公平性增强模型的高效MIA方法(FD-MIA)。它利用了原始模型和公平性增强模型中预测的差异，并利用观察到的预测差距作为攻击线索。我们还探索了减轻隐私泄露的潜在策略。大量的实验验证了我们的发现，并证明了该方法的有效性。



## **26. Detecting Language Model Attacks with Perplexity**

基于困惑的语言模型攻击检测 cs.CL

**SubmitDate**: 2023-11-07    [abs](http://arxiv.org/abs/2308.14132v3) [paper-pdf](http://arxiv.org/pdf/2308.14132v3)

**Authors**: Gabriel Alon, Michael Kamfonas

**Abstract**: A novel hack involving Large Language Models (LLMs) has emerged, exploiting adversarial suffixes to deceive models into generating perilous responses. Such jailbreaks can trick LLMs into providing intricate instructions to a malicious user for creating explosives, orchestrating a bank heist, or facilitating the creation of offensive content. By evaluating the perplexity of queries with adversarial suffixes using an open-source LLM (GPT-2), we found that they have exceedingly high perplexity values. As we explored a broad range of regular (non-adversarial) prompt varieties, we concluded that false positives are a significant challenge for plain perplexity filtering. A Light-GBM trained on perplexity and token length resolved the false positives and correctly detected most adversarial attacks in the test set.

摘要: 出现了一种涉及大型语言模型(LLM)的新黑客攻击，利用敌意后缀欺骗模型生成危险的响应。此类越狱可以诱使LLMS向恶意用户提供复杂的指令，以制造爆炸物、策划银行抢劫或为创建攻击性内容提供便利。通过使用开放源代码的LLM(GPT-2)对带有敌意后缀的查询的困惑度进行评估，我们发现它们具有极高的困惑度值。随着我们探索了广泛的常规(非对抗性)提示类型，我们得出结论，假阳性对于普通困惑过滤来说是一个重大挑战。一种针对困惑和令牌长度的Light-GBM解决了假阳性问题，并正确地检测到了测试集中的大多数对抗性攻击。



## **27. Competence-Based Analysis of Language Models**

基于能力的语言模型分析 cs.CL

**SubmitDate**: 2023-11-07    [abs](http://arxiv.org/abs/2303.00333v3) [paper-pdf](http://arxiv.org/pdf/2303.00333v3)

**Authors**: Adam Davies, Jize Jiang, ChengXiang Zhai

**Abstract**: Despite the recent success of large, pretrained neural language models (LLMs) on a variety of prompting tasks, these models can be alarmingly brittle to small changes in inputs or application contexts. To better understand such behavior and motivate the design of more robust LLMs, we provide a causal formulation of linguistic competence in the context of LLMs and propose a general framework to study and measure LLM competence. Our framework, CALM (Competence-based Analysis of Language Models), establishes the first quantitative measure of LLM competence, which we study by damaging models' internal representations of various linguistic properties in the course of performing various tasks using causal probing and evaluating models' alignment under these interventions with a given causal model. We also develop a novel approach for performing causal probing interventions using gradient-based adversarial attacks, which can target a broader range of properties and representations than existing techniques. We carry out a case study of CALM using these interventions to analyze BERT and RoBERTa's competence across a variety of lexical inference tasks, showing that the CALM framework and competence metric can be valuable tools for explaining and predicting their behavior across these tasks.

摘要: 尽管大型的、预先训练的神经语言模型(LLM)最近在各种提示任务上取得了成功，但这些模型对于输入或应用环境的微小变化可能会非常脆弱。为了更好地理解这种行为，并激励设计更稳健的LLM，我们在LLMS的背景下提出了语言能力的因果表述，并提出了一个研究和测量LLM能力的一般框架。我们的基于能力的语言模型分析框架建立了第一个LLM能力的定量测量，我们通过破坏模型在执行各种任务的过程中对各种语言属性的内部表征进行研究，并评估模型在这些干预措施下与给定的因果模型的一致性。我们还开发了一种新的方法来使用基于梯度的对抗性攻击来执行因果探测干预，该方法可以针对比现有技术更广泛的属性和表示。我们使用这些干预措施对CAMLE进行了个案研究，分析了Bert和Roberta在各种词汇推理任务上的能力，结果表明，CAMLE框架和能力度量可以成为解释和预测他们在这些任务中的行为的有价值的工具。



## **28. Probabilistic Categorical Adversarial Attack & Adversarial Training**

概率分类对抗性攻击与对抗性训练 cs.LG

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2210.09364v3) [paper-pdf](http://arxiv.org/pdf/2210.09364v3)

**Authors**: Han Xu, Pengfei He, Jie Ren, Yuxuan Wan, Zitao Liu, Hui Liu, Jiliang Tang

**Abstract**: The existence of adversarial examples brings huge concern for people to apply Deep Neural Networks (DNNs) in safety-critical tasks. However, how to generate adversarial examples with categorical data is an important problem but lack of extensive exploration. Previously established methods leverage greedy search method, which can be very time-consuming to conduct successful attack. This also limits the development of adversarial training and potential defenses for categorical data. To tackle this problem, we propose Probabilistic Categorical Adversarial Attack (PCAA), which transfers the discrete optimization problem to a continuous problem that can be solved efficiently by Projected Gradient Descent. In our paper, we theoretically analyze its optimality and time complexity to demonstrate its significant advantage over current greedy based attacks. Moreover, based on our attack, we propose an efficient adversarial training framework. Through a comprehensive empirical study, we justify the effectiveness of our proposed attack and defense algorithms.

摘要: 对抗性实例的存在给深度神经网络在安全关键任务中的应用带来了极大的关注。然而，如何利用分类数据生成对抗性实例是一个重要的问题，但缺乏广泛的探索。以前建立的方法利用贪婪搜索方法，进行成功的攻击可能非常耗时。这也限制了对抗性训练的发展和对分类数据的潜在防御。为了解决这个问题，我们提出了概率分类对抗性攻击(PCAA)，它将离散的优化问题转化为一个连续的问题，可以用投影梯度下降法有效地解决。在本文中，我们从理论上分析了它的最优性和时间复杂性，以证明它相对于现有的基于贪婪的攻击具有显著的优势。此外，基于我们的攻击，我们提出了一个有效的对抗性训练框架。通过全面的实证研究，验证了本文提出的攻防算法的有效性。



## **29. On existence, uniqueness and scalability of adversarial robustness measures for AI classifiers**

人工智能分类器对抗健壮性度量的存在性、唯一性和可伸缩性 stat.ML

16 pages, 3 figures

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2310.14421v2) [paper-pdf](http://arxiv.org/pdf/2310.14421v2)

**Authors**: Illia Horenko

**Abstract**: Simply-verifiable mathematical conditions for existence, uniqueness and explicit analytical computation of minimal adversarial paths (MAP) and minimal adversarial distances (MAD) for (locally) uniquely-invertible classifiers, for generalized linear models (GLM), and for entropic AI (EAI) are formulated and proven. Practical computation of MAP and MAD, their comparison and interpretations for various classes of AI tools (for neuronal networks, boosted random forests, GLM and EAI) are demonstrated on the common synthetic benchmarks: on a double Swiss roll spiral and its extensions, as well as on the two biomedical data problems (for the health insurance claim predictions, and for the heart attack lethality classification). On biomedical applications it is demonstrated how MAP provides unique minimal patient-specific risk-mitigating interventions in the predefined subsets of accessible control variables.

摘要: 提出并证明了(局部)唯一可逆分类器、广义线性模型(GLM)和熵人工智能(EAI)的最小对抗路径(MAP)和最小对抗距离(MAD)的存在唯一性和显式解析计算的简单可验证的数学条件.MAP和MAD的实际计算，以及它们对各种人工智能工具(用于神经元网络、增强随机森林、GLM和EAI)的比较和解释，在常见的合成基准上进行了演示：在双瑞士辊螺旋及其扩展上，以及在两个生物医学数据问题上(用于健康保险索赔预测和心脏病发作死亡分类)。在生物医学应用方面，它展示了MAP如何在可访问的控制变量的预定义子集中提供独特的、最小限度的患者特定风险缓解干预。



## **30. Preserving Privacy in GANs Against Membership Inference Attack**

防止成员推理攻击的GANS中的隐私保护 cs.LG

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2311.03172v1) [paper-pdf](http://arxiv.org/pdf/2311.03172v1)

**Authors**: Mohammadhadi Shateri, Francisco Messina, Fabrice Labeau, Pablo Piantanida

**Abstract**: Generative Adversarial Networks (GANs) have been widely used for generating synthetic data for cases where there is a limited size real-world dataset or when data holders are unwilling to share their data samples. Recent works showed that GANs, due to overfitting and memorization, might leak information regarding their training data samples. This makes GANs vulnerable to Membership Inference Attacks (MIAs). Several defense strategies have been proposed in the literature to mitigate this privacy issue. Unfortunately, defense strategies based on differential privacy are proven to reduce extensively the quality of the synthetic data points. On the other hand, more recent frameworks such as PrivGAN and PAR-GAN are not suitable for small-size training datasets. In the present work, the overfitting in GANs is studied in terms of the discriminator, and a more general measure of overfitting based on the Bhattacharyya coefficient is defined. Then, inspired by Fano's inequality, our first defense mechanism against MIAs is proposed. This framework, which requires only a simple modification in the loss function of GANs, is referred to as the maximum entropy GAN or MEGAN and significantly improves the robustness of GANs to MIAs. As a second defense strategy, a more heuristic model based on minimizing the information leaked from generated samples about the training data points is presented. This approach is referred to as mutual information minimization GAN (MIMGAN) and uses a variational representation of the mutual information to minimize the information that a synthetic sample might leak about the whole training data set. Applying the proposed frameworks to some commonly used data sets against state-of-the-art MIAs reveals that the proposed methods can reduce the accuracy of the adversaries to the level of random guessing accuracy with a small reduction in the quality of the synthetic data samples.

摘要: 生成性对抗网络(GANS)已被广泛用于生成合成数据，用于在现实世界数据集大小有限的情况下或当数据持有者不愿共享他们的数据样本的情况下。最近的研究表明，由于过度适应和记忆，Gans可能会泄露关于其训练数据样本的信息。这使得GAN容易受到成员身份推断攻击(MIA)。文献中已经提出了几种防御策略来缓解这一隐私问题。不幸的是，事实证明，基于差异隐私的防御策略会大大降低合成数据点的质量。另一方面，PrivGAN和PAR-GAN等较新的框架不适合小规模的训练数据集。本文从判别子的角度研究了Gans中的过拟合问题，定义了一种基于Bhattacharyya系数的更一般的过拟合度量。然后，受Fano不等式的启发，提出了我们针对MIA的第一种防御机制。这种框架只需要对GANS的损失函数进行简单的修改，被称为最大熵GAN或Megan，显著提高了GANS对MIA的鲁棒性。作为第二种防御策略，提出了一种更启发式的模型，该模型基于最小化关于训练数据点的生成样本所泄漏的信息。这种方法被称为互信息最小化GAN(MIMGAN)，它使用互信息的变分表示来最小化合成样本可能泄漏的关于整个训练数据集的信息。将提出的框架应用于一些常用的针对最新MIA的数据集，结果表明，所提出的方法可以在略微降低合成数据样本的质量的情况下，将对手的准确率降低到随机猜测准确率的水平。



## **31. Quantum-Error-Mitigated Detectable Byzantine Agreement with Dynamical Decoupling for Distributed Quantum Computing**

分布式量子计算中抑制量子误差的动态解耦可检测拜占庭协议 quant-ph

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2311.03097v1) [paper-pdf](http://arxiv.org/pdf/2311.03097v1)

**Authors**: Matthew Prest, Kuan-Cheng Chen

**Abstract**: In the burgeoning domain of distributed quantum computing, achieving consensus amidst adversarial settings remains a pivotal challenge. We introduce an enhancement to the Quantum Byzantine Agreement (QBA) protocol, uniquely incorporating advanced error mitigation techniques: Twirled Readout Error Extinction (T-REx) and dynamical decoupling (DD). Central to this refined approach is the utilization of a Noisy Intermediate Scale Quantum (NISQ) source device for heightened performance. Extensive tests on both simulated and real-world quantum devices, notably IBM's quantum computer, provide compelling evidence of the effectiveness of our T-REx and DD adaptations in mitigating prevalent quantum channel errors.   Subsequent to the entanglement distribution, our protocol adopts a verification method reminiscent of Quantum Key Distribution (QKD) schemes. The Commander then issues orders encoded in specific quantum states, like Retreat or Attack. In situations where received orders diverge, lieutenants engage in structured games to reconcile discrepancies. Notably, the frequency of these games is contingent upon the Commander's strategies and the overall network size. Our empirical findings underscore the enhanced resilience and effectiveness of the protocol in diverse scenarios. Nonetheless, scalability emerges as a concern with the growth of the network size. To sum up, our research illuminates the considerable potential of fortified quantum consensus systems in the NISQ era, highlighting the imperative for sustained research in bolstering quantum ecosystems.

摘要: 在蓬勃发展的分布式量子计算领域，在敌对环境中达成共识仍然是一个关键挑战。我们引入了对量子拜占庭协议(QBA)的增强，独特地结合了先进的错误缓解技术：旋转读出错误消除(T-REX)和动态解耦(DD)。这一改进方法的核心是利用噪声中尺度量子(NISQ)源设备来提高性能。对模拟和真实量子设备的广泛测试，特别是IBM的量子计算机，提供了令人信服的证据，证明我们的T-Rex和DD适配在缓解普遍存在的量子通道错误方面的有效性。在纠缠分发之后，我们的协议采用了一种类似于量子密钥分发(QKD)方案的验证方法。然后，指挥官发布以特定量子状态编码的命令，如撤退或攻击。在收到的订单不同的情况下，副手们会进行结构化的游戏，以协调差异。值得注意的是，这些游戏的频率取决于指挥官的战略和整体网络规模。我们的经验发现强调了该协议在不同情况下的增强的弹性和有效性。尽管如此，随着网络规模的增长，可扩展性成为一个令人担忧的问题。总而言之，我们的研究阐明了NISQ时代强化的量子共识系统的巨大潜力，突显了在支持量子生态系统方面进行持续研究的必要性。



## **32. SoK: Memorisation in machine learning**

SOK：机器学习中的记忆 cs.LG

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2311.03075v1) [paper-pdf](http://arxiv.org/pdf/2311.03075v1)

**Authors**: Dmitrii Usynin, Moritz Knolle, Georgios Kaissis

**Abstract**: Quantifying the impact of individual data samples on machine learning models is an open research problem. This is particularly relevant when complex and high-dimensional relationships have to be learned from a limited sample of the data generating distribution, such as in deep learning. It was previously shown that, in these cases, models rely not only on extracting patterns which are helpful for generalisation, but also seem to be required to incorporate some of the training data more or less as is, in a process often termed memorisation. This raises the question: if some memorisation is a requirement for effective learning, what are its privacy implications? In this work we unify a broad range of previous definitions and perspectives on memorisation in ML, discuss their interplay with model generalisation and their implications of these phenomena on data privacy. Moreover, we systematise methods allowing practitioners to detect the occurrence of memorisation or quantify it and contextualise our findings in a broad range of ML learning settings. Finally, we discuss memorisation in the context of privacy attacks, differential privacy (DP) and adversarial actors.

摘要: 量化单个数据样本对机器学习模型的影响是一个开放的研究问题。当必须从数据生成分布的有限样本中学习复杂和高维关系时，这尤其相关，例如在深度学习中。先前已经表明，在这些情况下，模型不仅依赖于提取有助于泛化的模式，而且似乎还需要或多或少地将一些训练数据纳入一个通常被称为记忆的过程中。这就提出了一个问题：如果记忆是有效学习的必要条件，那么它对隐私的影响是什么？在这项工作中，我们统一了以前关于ML中记忆的广泛定义和观点，讨论了它们与模型泛化的相互作用，以及它们对数据隐私的影响。此外，我们系统化的方法允许从业者检测记忆的发生或量化它，并在广泛的ML学习环境中将我们的发现与背景联系起来。最后，我们讨论了在隐私攻击、差异隐私(DP)和敌对行为的背景下的记忆。



## **33. Can LLMs Follow Simple Rules?**

低收入国家能遵循简单的规则吗？ cs.AI

Project website: https://eecs.berkeley.edu/~normanmu/llm_rules

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2311.04235v1) [paper-pdf](http://arxiv.org/pdf/2311.04235v1)

**Authors**: Norman Mu, Sarah Chen, Zifan Wang, Sizhe Chen, David Karamardian, Lulwa Aljeraisy, Dan Hendrycks, David Wagner

**Abstract**: As Large Language Models (LLMs) are deployed with increasing real-world responsibilities, it is important to be able to specify and constrain the behavior of these systems in a reliable manner. Model developers may wish to set explicit rules for the model, such as "do not generate abusive content", but these may be circumvented by jailbreaking techniques. Evaluating how well LLMs follow developer-provided rules in the face of adversarial inputs typically requires manual review, which slows down monitoring and methods development. To address this issue, we propose Rule-following Language Evaluation Scenarios (RuLES), a programmatic framework for measuring rule-following ability in LLMs. RuLES consists of 15 simple text scenarios in which the model is instructed to obey a set of rules in natural language while interacting with the human user. Each scenario has a concise evaluation program to determine whether the model has broken any rules in a conversation. Through manual exploration of model behavior in our scenarios, we identify 6 categories of attack strategies and collect two suites of test cases: one consisting of unique conversations from manual testing and one that systematically implements strategies from the 6 categories. Across various popular proprietary and open models such as GPT-4 and Llama 2, we find that all models are susceptible to a wide variety of adversarial hand-crafted user inputs, though GPT-4 is the best-performing model. Additionally, we evaluate open models under gradient-based attacks and find significant vulnerabilities. We propose RuLES as a challenging new setting for research into exploring and defending against both manual and automatic attacks on LLMs.

摘要: 随着大型语言模型(LLM)的部署承担着越来越多的现实责任，能够以可靠的方式指定和约束这些系统的行为是很重要的。模型开发人员可能希望为模型设置明确的规则，例如“不要生成滥用内容”，但可以通过越狱技术绕过这些规则。评估LLM在面对敌对输入时遵循开发人员提供的规则的情况通常需要手动审查，这会减缓监测和方法开发的速度。为了解决这一问题，我们提出了规则遵循语言评估场景(Rules)，这是一个衡量LLMS中规则遵循能力的程序性框架。规则由15个简单的文本场景组成，在这些场景中，模型被指示在与人类用户交互时遵守一组自然语言规则。每个场景都有一个简明的评估程序，以确定模型是否在对话中违反了任何规则。通过手动探索场景中的模型行为，我们确定了6类攻击策略，并收集了两套测试用例：一套由手动测试中的独特对话组成，另一套系统地实现了这6类策略。纵观各种流行的专有和开放机型，如GPT-4和Llama 2，我们发现所有机型都容易受到各种对抗性手工用户输入的影响，尽管GPT-4是性能最好的机型。此外，我们在基于梯度的攻击下对开放模型进行了评估，发现了显著的漏洞。我们提出规则作为一个具有挑战性的新环境，用于研究探索和防御对LLMS的手动和自动攻击。



## **34. DP-DCAN: Differentially Private Deep Contrastive Autoencoder Network for Single-cell Clustering**

DP-DCAN：用于单小区分簇的差分专用深度对比自动编码器网络 cs.LG

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2311.03410v1) [paper-pdf](http://arxiv.org/pdf/2311.03410v1)

**Authors**: Huifa Li, Jie Fu, Zhili Chen, Xiaomin Yang, Haitao Liu, Xinpeng Ling

**Abstract**: Single-cell RNA sequencing (scRNA-seq) is important to transcriptomic analysis of gene expression. Recently, deep learning has facilitated the analysis of high-dimensional single-cell data. Unfortunately, deep learning models may leak sensitive information about users. As a result, Differential Privacy (DP) is increasingly used to protect privacy. However, existing DP methods usually perturb whole neural networks to achieve differential privacy, and hence result in great performance overheads. To address this challenge, in this paper, we take advantage of the uniqueness of the autoencoder that it outputs only the dimension-reduced vector in the middle of the network, and design a Differentially Private Deep Contrastive Autoencoder Network (DP-DCAN) by partial network perturbation for single-cell clustering. Since only partial network is added with noise, the performance improvement is obvious and twofold: one part of network is trained with less noise due to a bigger privacy budget, and the other part is trained without any noise. Experimental results of six datasets have verified that DP-DCAN is superior to the traditional DP scheme with whole network perturbation. Moreover, DP-DCAN demonstrates strong robustness to adversarial attacks. The code is available at https://github.com/LFD-byte/DP-DCAN.

摘要: 单细胞RNA测序(scRNA-seq)对于基因表达的转录分析具有重要意义。最近，深度学习为高维单细胞数据的分析提供了便利。不幸的是，深度学习模型可能会泄露用户的敏感信息。因此，差异隐私(DP)越来越多地被用来保护隐私。然而，现有的DP方法通常会对整个神经网络进行扰动以实现差分隐私，从而导致很大的性能开销。为了应对这一挑战，本文利用自动编码器只输出网络中间降维向量的独特性，设计了一种基于部分网络扰动的差分私有深度对比自动编码器网络(DP-DCAN)，用于单小区聚类。由于只有部分网络添加了噪声，因此性能改善是明显的，而且是双重的：一部分网络的训练由于较大的隐私预算而噪声较小，而另一部分网络训练时没有任何噪声。在6个数据集上的实验结果表明，DP-DCAN算法优于传统的全网扰动下的DP算法。此外，DP-DCAN对敌方攻击表现出很强的鲁棒性。代码可在https://github.com/LFD-byte/DP-DCAN.上获得



## **35. MAWSEO: Adversarial Wiki Search Poisoning for Illicit Online Promotion**

MAWSEO：恶意维基搜索投毒进行非法在线推广 cs.CR

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2304.11300v2) [paper-pdf](http://arxiv.org/pdf/2304.11300v2)

**Authors**: Zilong Lin, Zhengyi Li, Xiaojing Liao, XiaoFeng Wang, Xiaozhong Liu

**Abstract**: As a prominent instance of vandalism edits, Wiki search poisoning for illicit promotion is a cybercrime in which the adversary aims at editing Wiki articles to promote illicit businesses through Wiki search results of relevant queries. In this paper, we report a study that, for the first time, shows that such stealthy blackhat SEO on Wiki can be automated. Our technique, called MAWSEO, employs adversarial revisions to achieve real-world cybercriminal objectives, including rank boosting, vandalism detection evasion, topic relevancy, semantic consistency, user awareness (but not alarming) of promotional content, etc. Our evaluation and user study demonstrate that MAWSEO is capable of effectively and efficiently generating adversarial vandalism edits, which can bypass state-of-the-art built-in Wiki vandalism detectors, and also get promotional content through to Wiki users without triggering their alarms. In addition, we investigated potential defense, including coherence based detection and adversarial training of vandalism detection, against our attack in the Wiki ecosystem.

摘要: 作为破坏编辑的一个突出例子，非法推广的维基搜索中毒是一种网络犯罪，对手旨在编辑维基文章，通过相关查询的维基搜索结果来推广非法业务。在这篇文章中，我们报告了一项研究，首次表明维基上这种隐蔽的黑帽SEO可以自动化。我们的技术，称为MAWSEO，使用对抗性修订来实现现实世界的网络犯罪目标，包括排名提升、破坏检测规避、主题相关性、语义一致性、用户对促销内容的感知(但不令人震惊)等。我们的评估和用户研究表明，MAWSEO能够有效和高效地生成对抗性破坏编辑，这可以绕过最先进的内置维基破坏检测器，还可以在不触发维基用户警报的情况下将宣传内容传递给维基用户。此外，我们调查了针对我们在维基生态系统中的攻击的潜在防御，包括基于一致性的检测和恶意检测的对抗性培训。



## **36. CT-GAT: Cross-Task Generative Adversarial Attack based on Transferability**

CT-GAT：基于可转移性的跨任务生成性对抗攻击 cs.CL

Accepted to EMNLP 2023 main conference Corrected the header error in  Figure 3

**SubmitDate**: 2023-11-05    [abs](http://arxiv.org/abs/2310.14265v2) [paper-pdf](http://arxiv.org/pdf/2310.14265v2)

**Authors**: Minxuan Lv, Chengwei Dai, Kun Li, Wei Zhou, Songlin Hu

**Abstract**: Neural network models are vulnerable to adversarial examples, and adversarial transferability further increases the risk of adversarial attacks. Current methods based on transferability often rely on substitute models, which can be impractical and costly in real-world scenarios due to the unavailability of training data and the victim model's structural details. In this paper, we propose a novel approach that directly constructs adversarial examples by extracting transferable features across various tasks. Our key insight is that adversarial transferability can extend across different tasks. Specifically, we train a sequence-to-sequence generative model named CT-GAT using adversarial sample data collected from multiple tasks to acquire universal adversarial features and generate adversarial examples for different tasks. We conduct experiments on ten distinct datasets, and the results demonstrate that our method achieves superior attack performance with small cost.

摘要: 神经网络模型容易受到对抗性例子的影响，对抗性的可转移性进一步增加了对抗性攻击的风险。目前基于可转移性的方法往往依赖于替代模型，由于训练数据和受害者模型的结构细节的不可用，这在现实世界的场景中可能是不切实际和昂贵的。在本文中，我们提出了一种新的方法，该方法通过提取跨任务的可转移特征来直接构造对抗性实例。我们的关键洞察是，对抗性转移可以扩展到不同的任务。具体地说，我们使用从多个任务收集的对抗性样本数据来训练序列到序列生成模型CT-GAT，以获取通用的对抗性特征，并为不同的任务生成对抗性实例。我们在10个不同的数据集上进行了实验，结果表明，该方法以较小的代价获得了优越的攻击性能。



## **37. DeepZero: Scaling up Zeroth-Order Optimization for Deep Model Training**

DeepZero：深度模型训练的零阶放大优化 cs.LG

**SubmitDate**: 2023-11-05    [abs](http://arxiv.org/abs/2310.02025v2) [paper-pdf](http://arxiv.org/pdf/2310.02025v2)

**Authors**: Aochuan Chen, Yimeng Zhang, Jinghan Jia, James Diffenderfer, Jiancheng Liu, Konstantinos Parasyris, Yihua Zhang, Zheng Zhang, Bhavya Kailkhura, Sijia Liu

**Abstract**: Zeroth-order (ZO) optimization has become a popular technique for solving machine learning (ML) problems when first-order (FO) information is difficult or impossible to obtain. However, the scalability of ZO optimization remains an open problem: Its use has primarily been limited to relatively small-scale ML problems, such as sample-wise adversarial attack generation. To our best knowledge, no prior work has demonstrated the effectiveness of ZO optimization in training deep neural networks (DNNs) without a significant decrease in performance. To overcome this roadblock, we develop DeepZero, a principled ZO deep learning (DL) framework that can scale ZO optimization to DNN training from scratch through three primary innovations. First, we demonstrate the advantages of coordinate-wise gradient estimation (CGE) over randomized vector-wise gradient estimation in training accuracy and computational efficiency. Second, we propose a sparsity-induced ZO training protocol that extends the model pruning methodology using only finite differences to explore and exploit the sparse DL prior in CGE. Third, we develop the methods of feature reuse and forward parallelization to advance the practical implementations of ZO training. Our extensive experiments show that DeepZero achieves state-of-the-art (SOTA) accuracy on ResNet-20 trained on CIFAR-10, approaching FO training performance for the first time. Furthermore, we show the practical utility of DeepZero in applications of certified adversarial defense and DL-based partial differential equation error correction, achieving 10-20% improvement over SOTA. We believe our results will inspire future research on scalable ZO optimization and contribute to advancing DL with black box.

摘要: 当一阶（FO）信息很难或不可能获得时，零阶（ZO）优化已经成为解决机器学习（ML）问题的流行技术。然而，ZO优化的可扩展性仍然是一个悬而未决的问题：它的使用主要限于相对较小规模的ML问题，例如样本对抗攻击生成。据我们所知，没有先前的工作证明了ZO优化在训练深度神经网络（DNN）时的有效性，而不会显着降低性能。为了克服这个障碍，我们开发了DeepZero，这是一个原则性的ZO深度学习（DL）框架，可以通过三个主要创新从零开始将ZO优化扩展到DNN训练。首先，我们证明了坐标梯度估计（CGE）在训练精度和计算效率方面优于随机向量梯度估计。其次，我们提出了一个稀疏诱导的ZO训练协议，扩展了模型修剪方法，只使用有限的差异，探索和利用稀疏DL之前在CGE。第三，我们开发了特征重用和前向并行化的方法，以推进ZO训练的实际实现。我们广泛的实验表明，DeepZero在CIFAR-10上训练的ResNet-20上达到了最先进的（SOTA）精度，首次接近FO训练性能。此外，我们还展示了DeepZero在认证对抗防御和基于DL的偏微分方程纠错应用中的实际效用，比SOTA提高了10-20%。我们相信我们的研究结果将激发未来可扩展ZO优化的研究，并有助于推进DL与黑盒。



## **38. Exploring Transferability of Multimodal Adversarial Samples for Vision-Language Pre-training Models with Contrastive Learning**

基于对比学习的视觉语言预训练模型多通道对抗性样本可转换性研究 cs.MM

**SubmitDate**: 2023-11-05    [abs](http://arxiv.org/abs/2308.12636v2) [paper-pdf](http://arxiv.org/pdf/2308.12636v2)

**Authors**: Youze Wang, Wenbo Hu, Yinpeng Dong, Hanwang Zhang, Richang Hong

**Abstract**: Vision-language pre-training models (VLP) are vulnerable, especially to multimodal adversarial samples, which can be crafted by adding imperceptible perturbations on both original images and texts. However, under the black-box setting, there have been no works to explore the transferability of multimodal adversarial attacks against the VLP models. In this work, we take CLIP as the surrogate model and propose a gradient-based multimodal attack method to generate transferable adversarial samples against the VLP models. By applying the gradient to optimize the adversarial images and adversarial texts simultaneously, our method can better search for and attack the vulnerable images and text information pairs. To improve the transferability of the attack, we utilize contrastive learning including image-text contrastive learning and intra-modal contrastive learning to have a more generalized understanding of the underlying data distribution and mitigate the overfitting of the surrogate model so that the generated multimodal adversarial samples have a higher transferability for VLP models. Extensive experiments validate the effectiveness of the proposed method.

摘要: 视觉语言预训练模型(VLP)是脆弱的，特别是对多模式对抗性样本，这些样本可以通过在原始图像和文本上添加不可察觉的扰动来构建。然而，在黑箱环境下，还没有研究针对VLP模型的多通道对抗性攻击的可转移性。在本文中，我们以CLIP为代理模型，提出了一种基于梯度的多模式攻击方法来生成可传输的对抗VLP模型的样本。通过应用梯度同时优化敌意图像和敌意文本，我们的方法可以更好地搜索和攻击易受攻击的图文信息对。为了提高攻击的可转移性，我们利用对比学习，包括图文对比学习和通道内对比学习，以更全面地了解底层数据分布，并减少代理模型的过度拟合，使生成的多通道对抗性样本对VLP模型具有更高的可转移性。大量实验验证了该方法的有效性。



## **39. Unfolding Local Growth Rate Estimates for (Almost) Perfect Adversarial Detection**

关于(几乎)完美敌意检测的局部增长率估计 cs.CV

accepted at VISAPP23

**SubmitDate**: 2023-11-04    [abs](http://arxiv.org/abs/2212.06776v3) [paper-pdf](http://arxiv.org/pdf/2212.06776v3)

**Authors**: Peter Lorenz, Margret Keuper, Janis Keuper

**Abstract**: Convolutional neural networks (CNN) define the state-of-the-art solution on many perceptual tasks. However, current CNN approaches largely remain vulnerable against adversarial perturbations of the input that have been crafted specifically to fool the system while being quasi-imperceptible to the human eye. In recent years, various approaches have been proposed to defend CNNs against such attacks, for example by model hardening or by adding explicit defence mechanisms. Thereby, a small "detector" is included in the network and trained on the binary classification task of distinguishing genuine data from data containing adversarial perturbations. In this work, we propose a simple and light-weight detector, which leverages recent findings on the relation between networks' local intrinsic dimensionality (LID) and adversarial attacks. Based on a re-interpretation of the LID measure and several simple adaptations, we surpass the state-of-the-art on adversarial detection by a significant margin and reach almost perfect results in terms of F1-score for several networks and datasets. Sources available at: https://github.com/adverML/multiLID

摘要: 卷积神经网络(CNN)定义了许多感知任务的最先进的解决方案。然而，目前的CNN方法在很大程度上仍然容易受到输入的对抗性扰动，这些扰动是专门为愚弄系统而设计的，而人眼几乎察觉不到。近年来，已经提出了各种方法来保护CNN免受此类攻击，例如通过模型硬化或通过添加显式防御机制。因此，在网络中包括一个小的“检测器”，并在区分真实数据和包含对抗性扰动的数据的二进制分类任务上进行训练。在这项工作中，我们提出了一个简单而轻量级的检测器，它利用了最近关于网络的局部固有维度(LID)与对手攻击之间关系的研究结果。基于对LID度量的重新解释和几个简单的适应，我们在对手检测方面远远超过了最先进的水平，并在几个网络和数据集的F1得分方面取得了几乎完美的结果。资料来源：https://github.com/adverML/multiLID



## **40. MTS-DVGAN: Anomaly Detection in Cyber-Physical Systems using a Dual Variational Generative Adversarial Network**

MTS-DVGAN：基于对偶变分生成对抗网络的网络物理系统异常检测 cs.CR

27 pages, 14 figures, 8 tables. Accepted by Computers & Security

**SubmitDate**: 2023-11-04    [abs](http://arxiv.org/abs/2311.02378v1) [paper-pdf](http://arxiv.org/pdf/2311.02378v1)

**Authors**: Haili Sun, Yan Huang, Lansheng Han, Cai Fu, Hongle Liu, Xiang Long

**Abstract**: Deep generative models are promising in detecting novel cyber-physical attacks, mitigating the vulnerability of Cyber-physical systems (CPSs) without relying on labeled information. Nonetheless, these generative models face challenges in identifying attack behaviors that closely resemble normal data, or deviate from the normal data distribution but are in close proximity to the manifold of the normal cluster in latent space. To tackle this problem, this article proposes a novel unsupervised dual variational generative adversarial model named MST-DVGAN, to perform anomaly detection in multivariate time series data for CPS security. The central concept is to enhance the model's discriminative capability by widening the distinction between reconstructed abnormal samples and their normal counterparts. Specifically, we propose an augmented module by imposing contrastive constraints on the reconstruction process to obtain a more compact embedding. Then, by exploiting the distribution property and modeling the normal patterns of multivariate time series, a variational autoencoder is introduced to force the generative adversarial network (GAN) to generate diverse samples. Furthermore, two augmented loss functions are designed to extract essential characteristics in a self-supervised manner through mutual guidance between the augmented samples and original samples. Finally, a specific feature center loss is introduced for the generator network to enhance its stability. Empirical experiments are conducted on three public datasets, namely SWAT, WADI and NSL_KDD. Comparing with the state-of-the-art methods, the evaluation results show that the proposed MTS-DVGAN is more stable and can achieve consistent performance improvement.

摘要: 深度生成模型在检测新的网络物理攻击、减轻网络物理系统(CPSS)的脆弱性方面很有希望，而不依赖于标签信息。然而，这些生成性模型在识别与正态数据非常相似的攻击行为，或偏离正态数据分布但在潜在空间中非常接近正态簇流形的攻击行为方面面临挑战。针对这一问题，本文提出了一种新的无监督双变分生成对抗模型MST-DVGAN，用于在多变量时间序列数据中进行异常检测，以保证CPS的安全性。其核心概念是通过扩大重建的异常样本与其正常样本之间的区别来增强模型的区分能力。具体地说，通过对重建过程施加对比约束，我们提出了一种增广模块，以获得更紧凑的嵌入。然后，通过利用多变量时间序列的分布特性和正态模式建模，引入变分自动编码器来强制生成性对抗网络(GAN)生成不同的样本。此外，还设计了两个增广损失函数，通过增广样本和原始样本之间的相互指导，以自监督的方式提取本质特征。最后，为提高发电机网络的稳定性，引入了特定的特征中心损耗。在SWAT、WADI和NSL_KDD三个公共数据集上进行了实验。评估结果表明，与现有方法相比，本文提出的MTS-DVGAN算法具有更高的稳定性，并能实现持续的性能提升。



## **41. From Trojan Horses to Castle Walls: Unveiling Bilateral Backdoor Effects in Diffusion Models**

从特洛伊木马到城堡墙：在扩散模型中揭示双边后门效应 cs.LG

10 pages, 6 figures, 7 tables

**SubmitDate**: 2023-11-04    [abs](http://arxiv.org/abs/2311.02373v1) [paper-pdf](http://arxiv.org/pdf/2311.02373v1)

**Authors**: Zhuoshi Pan, Yuguang Yao, Gaowen Liu, Bingquan Shen, H. Vicky Zhao, Ramana Rao Kompella, Sijia Liu

**Abstract**: While state-of-the-art diffusion models (DMs) excel in image generation, concerns regarding their security persist. Earlier research highlighted DMs' vulnerability to backdoor attacks, but these studies placed stricter requirements than conventional methods like 'BadNets' in image classification. This is because the former necessitates modifications to the diffusion sampling and training procedures. Unlike the prior work, we investigate whether generating backdoor attacks in DMs can be as simple as BadNets, i.e., by only contaminating the training dataset without tampering the original diffusion process. In this more realistic backdoor setting, we uncover bilateral backdoor effects that not only serve an adversarial purpose (compromising the functionality of DMs) but also offer a defensive advantage (which can be leveraged for backdoor defense). Specifically, we find that a BadNets-like backdoor attack remains effective in DMs for producing incorrect images (misaligned with the intended text conditions), and thereby yielding incorrect predictions when DMs are used as classifiers. Meanwhile, backdoored DMs exhibit an increased ratio of backdoor triggers, a phenomenon we refer to as `trigger amplification', among the generated images. We show that this latter insight can be used to enhance the detection of backdoor-poisoned training data. Even under a low backdoor poisoning ratio, studying the backdoor effects of DMs is also valuable for designing anti-backdoor image classifiers. Last but not least, we establish a meaningful linkage between backdoor attacks and the phenomenon of data replications by exploring DMs' inherent data memorization tendencies. The codes of our work are available at https://github.com/OPTML-Group/BiBadDiff.

摘要: 虽然最先进的扩散模型(DM)在图像生成方面表现出色，但对其安全性的担忧依然存在。早期的研究强调了DM对后门攻击的脆弱性，但这些研究在图像分类方面对图像分类提出了比BadNets等传统方法更严格的要求。这是因为前者需要对扩散抽样和训练程序进行修改。与以前的工作不同，我们研究了在DM中生成后门攻击是否可以像BadNets一样简单，即只污染训练数据集而不篡改原始的扩散过程。在这个更现实的后门设置中，我们揭示了双边后门效应，这些后门效应不仅服务于敌对目的(损害DM的功能)，而且提供了防御优势(可用于后门防御)。具体地说，我们发现类似BadNets的后门攻击在DM中仍然有效，因为它会产生错误的图像(与预期的文本条件不一致)，从而在DM用作分类器时产生错误的预测。与此同时，走后门的DM在生成的图像中显示出后门触发器的比例增加，我们将这种现象称为“触发放大”。我们表明，后一种见解可以用于增强对后门中毒训练数据的检测。即使在低后门投毒率的情况下，研究DM的后门效应对于反后门图像分类器的设计也是有价值的。最后但并非最不重要的是，我们通过探索DM固有的数据记忆倾向，在后门攻击和数据复制现象之间建立了有意义的联系。我们工作的代码可以在https://github.com/OPTML-Group/BiBadDiff.上找到



## **42. Secure compilation of rich smart contracts on poor UTXO blockchains**

在贫穷的UTXO区块链上安全地编译丰富的智能合同 cs.CR

**SubmitDate**: 2023-11-04    [abs](http://arxiv.org/abs/2305.09545v2) [paper-pdf](http://arxiv.org/pdf/2305.09545v2)

**Authors**: Massimo Bartoletti, Riccardo Marchesin, Roberto Zunino

**Abstract**: Most blockchain platforms from Ethereum onwards render smart contracts as stateful reactive objects that update their state and transfer crypto-assets in response to transactions. A drawback of this design is that when users submit a transaction, they cannot predict in which state it will be executed. This exposes them to transaction-ordering attacks, a widespread class of attacks where adversaries with the power to construct blocks of transactions can extract value from smart contracts (the so-called MEV attacks). The UTXO model is an alternative blockchain design that thwarts these attacks by requiring new transactions to spend past ones: since transactions have unique identifiers, reordering attacks are ineffective. Currently, the blockchains following the UTXO model either provide contracts with limited expressiveness (Bitcoin), or require complex run-time environments (Cardano). We present ILLUM, an Intermediate-Level Language for the UTXO Model. ILLUM can express real-world smart contracts, e.g. those found in Decentralized Finance. We define a compiler from ILLUM to a bare-bone UTXO blockchain with loop-free scripts. Our compilation target only requires minimal extensions to Bitcoin Script: in particular, we exploit covenants, a mechanism for preserving scripts along chains of transactions. We prove the security of our compiler: namely, any attack targeting the compiled contract is also observable at the ILLUM level. Hence, the compiler does not introduce new vulnerabilities that were not already present in the source ILLUM contract. Finally, we discuss the suitability of ILLUM as a compilation target for higher-level contract languages.

摘要: 从Etherum开始，大多数区块链平台都将智能合约呈现为有状态的反应对象，这些对象更新其状态并传输加密资产以响应交易。这种设计的一个缺点是，当用户提交事务时，他们无法预测该事务将在哪种状态下执行。这使他们面临交易顺序攻击，这是一种广泛存在的攻击类别，在这种攻击中，有能力构建交易块的对手可以从智能合约中提取价值(所谓的MEV攻击)。UTXO模型是一种替代区块链设计，通过要求新交易花费过去的交易来挫败这些攻击：由于交易具有唯一标识符，重新排序攻击是无效的。目前，遵循UTXO模式的区块链要么提供表现力有限的合同(比特币)，要么需要复杂的运行时环境(Cardano)。我们介绍了Illum，一种用于UTXO模型的中级语言。Illum可以表示现实世界中的智能合约，例如在去中心化金融中找到的那些。我们定义了一个从Illum到具有无循环脚本的基本UTXO区块链的编译器。我们的编译目标只需要对比特币脚本进行最小程度的扩展：尤其是，我们利用了契诺，这是一种在交易链上保留脚本的机制。我们证明了我们的编译器的安全性：也就是说，任何针对已编译约定的攻击也可以在Illum级别上观察到。因此，编译器不会引入源Illum协定中尚未存在的新漏洞。最后，我们讨论了Illum作为高级契约语言的编译目标的适用性。



## **43. Generative Adversarial Networks to infer velocity components in rotating turbulent flows**

生成对抗网络在旋转湍流中的速度分量推断 physics.flu-dyn

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2301.07541v2) [paper-pdf](http://arxiv.org/pdf/2301.07541v2)

**Authors**: Tianyi Li, Michele Buzzicotti, Luca Biferale, Fabio Bonaccorso

**Abstract**: Inference problems for two-dimensional snapshots of rotating turbulent flows are studied. We perform a systematic quantitative benchmark of point-wise and statistical reconstruction capabilities of the linear Extended Proper Orthogonal Decomposition (EPOD) method, a non-linear Convolutional Neural Network (CNN) and a Generative Adversarial Network (GAN). We attack the important task of inferring one velocity component out of the measurement of a second one, and two cases are studied: (I) both components lay in the plane orthogonal to the rotation axis and (II) one of the two is parallel to the rotation axis. We show that EPOD method works well only for the former case where both components are strongly correlated, while CNN and GAN always outperform EPOD both concerning point-wise and statistical reconstructions. For case (II), when the input and output data are weakly correlated, all methods fail to reconstruct faithfully the point-wise information. In this case, only GAN is able to reconstruct the field in a statistical sense. The analysis is performed using both standard validation tools based on $L_2$ spatial distance between the prediction and the ground truth and more sophisticated multi-scale analysis using wavelet decomposition. Statistical validation is based on standard Jensen-Shannon divergence between the probability density functions, spectral properties and multi-scale flatness.

摘要: 研究了旋转湍流二维快照的推断问题。我们对线性扩展本征正交分解(EPOD)方法、非线性卷积神经网络(CNN)和生成性对抗网络(GAN)的逐点重建和统计重建能力进行了系统的定量基准测试。我们提出了从第二个速度分量的测量中推断出一个速度分量的重要任务，并研究了两种情况：(I)两个分量都位于与旋转轴垂直的平面上；(Ii)两个分量中的一个平行于旋转轴。我们发现，EPOD方法只适用于强相关的前一种情况，而CNN和GAN在逐点重建和统计重建方面总是优于EPOD。对于情况(II)，当输入和输出数据弱相关时，所有方法都不能忠实地重建逐点信息。在这种情况下，只有GaN能够在统计意义上重建场。分析使用了基于$L_2的标准验证工具和更复杂的基于小波分解的多尺度分析。统计验证基于概率密度函数、光谱特性和多尺度平坦度之间的标准Jensen-Shannon散度。



## **44. HINT: Healthy Influential-Noise based Training to Defend against Data Poisoning Attacks**

提示：健康影响-基于噪音的培训可防御数据中毒攻击 cs.LG

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2309.08549v2) [paper-pdf](http://arxiv.org/pdf/2309.08549v2)

**Authors**: Minh-Hao Van, Alycia N. Carey, Xintao Wu

**Abstract**: While numerous defense methods have been proposed to prohibit potential poisoning attacks from untrusted data sources, most research works only defend against specific attacks, which leaves many avenues for an adversary to exploit. In this work, we propose an efficient and robust training approach to defend against data poisoning attacks based on influence functions, named Healthy Influential-Noise based Training. Using influence functions, we craft healthy noise that helps to harden the classification model against poisoning attacks without significantly affecting the generalization ability on test data. In addition, our method can perform effectively when only a subset of the training data is modified, instead of the current method of adding noise to all examples that has been used in several previous works. We conduct comprehensive evaluations over two image datasets with state-of-the-art poisoning attacks under different realistic attack scenarios. Our empirical results show that HINT can efficiently protect deep learning models against the effect of both untargeted and targeted poisoning attacks.

摘要: 虽然已经提出了许多防御方法来阻止来自不受信任的数据源的潜在中毒攻击，但大多数研究工作只防御特定的攻击，这给对手留下了许多可以利用的途径。在这项工作中，我们提出了一种基于影响函数的高效、健壮的数据中毒攻击训练方法，即基于健康影响噪声的训练方法。利用影响函数构造健康噪声，在不显著影响测试数据泛化能力的情况下，有助于加强分类模型对中毒攻击的抵抗能力。此外，我们的方法可以在只修改训练数据的子集的情况下有效地执行，而不是在以前的几个工作中使用的向所有样本添加噪声的方法。在不同的真实攻击场景下，我们对两个具有最新技术的中毒攻击的图像数据集进行了综合评估。我们的实验结果表明，提示可以有效地保护深度学习模型免受非定向和定向中毒攻击的影响。



## **45. Adaptive Data Analysis in a Balanced Adversarial Model**

均衡对抗性模型中的自适应数据分析 cs.LG

Accepted to NeurIPS 2023 (Spotlight)

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2305.15452v2) [paper-pdf](http://arxiv.org/pdf/2305.15452v2)

**Authors**: Kobbi Nissim, Uri Stemmer, Eliad Tsfadia

**Abstract**: In adaptive data analysis, a mechanism gets $n$ i.i.d. samples from an unknown distribution $D$, and is required to provide accurate estimations to a sequence of adaptively chosen statistical queries with respect to $D$. Hardt and Ullman (FOCS 2014) and Steinke and Ullman (COLT 2015) showed that in general, it is computationally hard to answer more than $\Theta(n^2)$ adaptive queries, assuming the existence of one-way functions.   However, these negative results strongly rely on an adversarial model that significantly advantages the adversarial analyst over the mechanism, as the analyst, who chooses the adaptive queries, also chooses the underlying distribution $D$. This imbalance raises questions with respect to the applicability of the obtained hardness results -- an analyst who has complete knowledge of the underlying distribution $D$ would have little need, if at all, to issue statistical queries to a mechanism which only holds a finite number of samples from $D$.   We consider more restricted adversaries, called \emph{balanced}, where each such adversary consists of two separated algorithms: The \emph{sampler} who is the entity that chooses the distribution and provides the samples to the mechanism, and the \emph{analyst} who chooses the adaptive queries, but has no prior knowledge of the underlying distribution (and hence has no a priori advantage with respect to the mechanism). We improve the quality of previous lower bounds by revisiting them using an efficient \emph{balanced} adversary, under standard public-key cryptography assumptions. We show that these stronger hardness assumptions are unavoidable in the sense that any computationally bounded \emph{balanced} adversary that has the structure of all known attacks, implies the existence of public-key cryptography.

摘要: 在适应性数据分析中，一个机制得到$n$I.I.D.来自未知分布$D$的样本，并且需要对关于$D$的适应性选择的统计查询序列提供准确的估计。Hardt和Ullman(FOCS 2014)和Steinke和Ullman(COLT 2015)表明，假设存在单向函数，通常很难回答超过$\theta(n^2)$自适应查询。然而，这些负面结果强烈依赖于对抗性模型，该模型显著地使对抗性分析师相对于该机制具有优势，因为选择自适应查询的分析师也选择基础分布$D$。这种不平衡对所获得的硬度结果的适用性提出了问题--完全了解基本分布$D$的分析员将几乎不需要向一个仅保存来自$D$的有限数量样本的机制发出统计查询。我们考虑了更受限制的对手，称为\emph{平衡}，其中每个这样的对手由两个独立的算法组成：\emph{Sampler}是选择分布并向机制提供样本的实体，以及\emph{Analyst}选择自适应查询，但对底层分布没有先验知识(因此在机制方面没有先验优势)。在标准的公钥密码学假设下，我们通过使用一个有效的、平衡的对手来重新访问以前的下界，从而提高了它们的质量。我们证明了这些更强的难度假设是不可避免的，因为任何具有所有已知攻击结构的计算有界的对手都意味着公钥密码学的存在。



## **46. The Alignment Problem in Context**

上下文中的对齐问题 cs.LG

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2311.02147v1) [paper-pdf](http://arxiv.org/pdf/2311.02147v1)

**Authors**: Raphaël Millière

**Abstract**: A core challenge in the development of increasingly capable AI systems is to make them safe and reliable by ensuring their behaviour is consistent with human values. This challenge, known as the alignment problem, does not merely apply to hypothetical future AI systems that may pose catastrophic risks; it already applies to current systems, such as large language models, whose potential for harm is rapidly increasing. In this paper, I assess whether we are on track to solve the alignment problem for large language models, and what that means for the safety of future AI systems. I argue that existing strategies for alignment are insufficient, because large language models remain vulnerable to adversarial attacks that can reliably elicit unsafe behaviour. I offer an explanation of this lingering vulnerability on which it is not simply a contingent limitation of current language models, but has deep technical ties to a crucial aspect of what makes these models useful and versatile in the first place -- namely, their remarkable aptitude to learn "in context" directly from user instructions. It follows that the alignment problem is not only unsolved for current AI systems, but may be intrinsically difficult to solve without severely undermining their capabilities. Furthermore, this assessment raises concerns about the prospect of ensuring the safety of future and more capable AI systems.

摘要: 开发能力日益强大的人工智能系统的核心挑战是通过确保其行为符合人类价值观，使其安全可靠。这一挑战被称为对齐问题，不仅适用于可能带来灾难性风险的假设性未来人工智能系统;它已经适用于当前系统，例如大型语言模型，其潜在危害正在迅速增加。在本文中，我评估了我们是否正在解决大型语言模型的对齐问题，以及这对未来AI系统的安全性意味着什么。我认为，现有的对齐策略是不够的，因为大型语言模型仍然容易受到对抗性攻击，可以可靠地引发不安全的行为。我对这个挥之不去的漏洞进行了解释，它不仅仅是当前语言模型的偶然限制，而是与使这些模型首先变得有用和通用的关键方面有着深刻的技术联系-即它们直接从用户指令中“在上下文中”学习的非凡能力。因此，对齐问题不仅是当前人工智能系统未解决的问题，而且在不严重破坏其能力的情况下，可能本质上很难解决。此外，这一评估引发了人们对确保未来更强大的人工智能系统安全性的担忧。



## **47. Bucks for Buckets (B4B): Active Defenses Against Stealing Encoders**

以桶换钱(B4B)：主动防御窃取编码器 cs.LG

Accepted at NeurIPS2023

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2310.08571v2) [paper-pdf](http://arxiv.org/pdf/2310.08571v2)

**Authors**: Jan Dubiński, Stanisław Pawlak, Franziska Boenisch, Tomasz Trzciński, Adam Dziedzic

**Abstract**: Machine Learning as a Service (MLaaS) APIs provide ready-to-use and high-utility encoders that generate vector representations for given inputs. Since these encoders are very costly to train, they become lucrative targets for model stealing attacks during which an adversary leverages query access to the API to replicate the encoder locally at a fraction of the original training costs. We propose Bucks for Buckets (B4B), the first active defense that prevents stealing while the attack is happening without degrading representation quality for legitimate API users. Our defense relies on the observation that the representations returned to adversaries who try to steal the encoder's functionality cover a significantly larger fraction of the embedding space than representations of legitimate users who utilize the encoder to solve a particular downstream task.vB4B leverages this to adaptively adjust the utility of the returned representations according to a user's coverage of the embedding space. To prevent adaptive adversaries from eluding our defense by simply creating multiple user accounts (sybils), B4B also individually transforms each user's representations. This prevents the adversary from directly aggregating representations over multiple accounts to create their stolen encoder copy. Our active defense opens a new path towards securely sharing and democratizing encoders over public APIs.

摘要: 机器学习即服务(MLaaS)API提供了现成的、高实用的编码器，可以为给定的输入生成向量表示。由于这些编码器的培训成本非常高，他们成为模型窃取攻击的有利可图的目标，在攻击期间，对手利用对API的查询访问来以原始培训成本的一小部分在本地复制编码器。我们提出了Bucks for Buckets(B4B)，这是第一种主动防御，可以在攻击发生时防止窃取，而不会降低合法API用户的表示质量。我们的辩护依赖于这样的观察，即返回给试图窃取编码器功能的对手的表示覆盖的嵌入空间比利用编码器解决特定下游任务的合法用户的表示大得多。vB4B利用这一点来根据用户对嵌入空间的覆盖自适应地调整返回的表示的效用。为了防止适应性对手通过简单地创建多个用户帐户(Sybils)来逃避我们的防御，B4B还单独转换每个用户的表示。这可以防止对手直接在多个帐户上聚合表示，以创建他们被盗的编码器副本。我们的积极防御为通过公共API安全共享和民主化编码器开辟了一条新的道路。



## **48. Efficient Black-Box Adversarial Attacks on Neural Text Detectors**

基于神经文本检测器的高效黑盒对抗攻击 cs.CL

Accepted at ICNLSP 2023

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2311.01873v1) [paper-pdf](http://arxiv.org/pdf/2311.01873v1)

**Authors**: Vitalii Fishchuk, Daniel Braun

**Abstract**: Neural text detectors are models trained to detect whether a given text was generated by a language model or written by a human. In this paper, we investigate three simple and resource-efficient strategies (parameter tweaking, prompt engineering, and character-level mutations) to alter texts generated by GPT-3.5 that are unsuspicious or unnoticeable for humans but cause misclassification by neural text detectors. The results show that especially parameter tweaking and character-level mutations are effective strategies.

摘要: 神经文本检测器是经过训练的模型，用于检测给定的文本是由语言模型生成的还是由人类编写的。在本文中，我们研究了三种简单且资源高效的策略(参数调整、即时工程和字符级突变)来更改由GPT-3.5生成的文本，这些文本对人类来说是不可疑的或不可察觉的，但会导致神经文本检测器的错误分类。结果表明，特别是参数调整和字符级突变是有效的策略。



## **49. Adversarial Attacks against Binary Similarity Systems**

二进制相似系统的对抗性攻击 cs.CR

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2303.11143v2) [paper-pdf](http://arxiv.org/pdf/2303.11143v2)

**Authors**: Gianluca Capozzi, Daniele Cono D'Elia, Giuseppe Antonio Di Luna, Leonardo Querzoni

**Abstract**: In recent years, binary analysis gained traction as a fundamental approach to inspect software and guarantee its security. Due to the exponential increase of devices running software, much research is now moving towards new autonomous solutions based on deep learning models, as they have been showing state-of-the-art performances in solving binary analysis problems. One of the hot topics in this context is binary similarity, which consists in determining if two functions in assembly code are compiled from the same source code. However, it is unclear how deep learning models for binary similarity behave in an adversarial context. In this paper, we study the resilience of binary similarity models against adversarial examples, showing that they are susceptible to both targeted and untargeted attacks (w.r.t. similarity goals) performed by black-box and white-box attackers. In more detail, we extensively test three current state-of-the-art solutions for binary similarity against two black-box greedy attacks, including a new technique that we call Spatial Greedy, and one white-box attack in which we repurpose a gradient-guided strategy used in attacks to image classifiers.

摘要: 近年来，二进制分析作为检查软件和保证其安全性的基本方法得到了越来越多的重视。由于运行软件的设备呈指数级增长，许多研究现在正在转向基于深度学习模型的新的自主解决方案，因为它们在解决二进制分析问题方面表现出了最先进的性能。这方面的一个热门话题是二进制相似性，即确定汇编代码中的两个函数是否从相同的源代码编译而来。然而，目前还不清楚二元相似性的深度学习模型在对抗性环境中的表现如何。在本文中，我们研究了二进制相似模型对敌意例子的弹性，表明它们对目标攻击和非目标攻击都敏感(w.r.t.相似性目标)由黑盒和白盒攻击者执行。更详细地，我们针对两种黑盒贪婪攻击广泛地测试了三种当前最先进的二进制相似性解决方案，其中包括一种称为空间贪婪的新技术，以及一种白盒攻击，其中我们将攻击中使用的梯度引导策略重新用于图像分类器。



## **50. Adversarial Attacks on Cooperative Multi-agent Bandits**

合作多智能体盗贼的对抗性攻击 cs.LG

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2311.01698v1) [paper-pdf](http://arxiv.org/pdf/2311.01698v1)

**Authors**: Jinhang Zuo, Zhiyao Zhang, Xuchuang Wang, Cheng Chen, Shuai Li, John C. S. Lui, Mohammad Hajiesmaili, Adam Wierman

**Abstract**: Cooperative multi-agent multi-armed bandits (CMA2B) consider the collaborative efforts of multiple agents in a shared multi-armed bandit game. We study latent vulnerabilities exposed by this collaboration and consider adversarial attacks on a few agents with the goal of influencing the decisions of the rest. More specifically, we study adversarial attacks on CMA2B in both homogeneous settings, where agents operate with the same arm set, and heterogeneous settings, where agents have distinct arm sets. In the homogeneous setting, we propose attack strategies that, by targeting just one agent, convince all agents to select a particular target arm $T-o(T)$ times while incurring $o(T)$ attack costs in $T$ rounds. In the heterogeneous setting, we prove that a target arm attack requires linear attack costs and propose attack strategies that can force a maximum number of agents to suffer linear regrets while incurring sublinear costs and only manipulating the observations of a few target agents. Numerical experiments validate the effectiveness of our proposed attack strategies.

摘要: 合作多智能体多武装土匪(CMA2B)考虑多个智能体在共享的多臂土匪博弈中的协作努力。我们研究了这种合作暴露的潜在漏洞，并考虑了对几个代理的对抗性攻击，目的是影响其余代理的决策。更具体地说，我们研究了在同构设置和异质设置下对CMA2B的对抗性攻击，在同构设置中，代理使用相同的ARM集操作，而在异质设置中，代理具有不同的ARM集。在同类环境下，我们提出了攻击策略，通过只针对一个代理，说服所有代理选择特定的目标臂$T-o(T)$次，同时在$T$轮中产生$o(T)$攻击成本。在异质环境下，我们证明了一次目标ARM攻击需要线性攻击代价，并提出了一种攻击策略，该策略可以迫使最大数量的代理遭受线性后悔，同时产生次线性代价，并且只操纵少数目标代理的观测。数值实验验证了本文提出的攻击策略的有效性。



