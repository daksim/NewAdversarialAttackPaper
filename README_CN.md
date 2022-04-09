# Latest Adversarial Attack Papers
**update at 2022-04-10 06:31:30**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Reinforcement Learning for Linear Quadratic Control is Vulnerable Under Cost Manipulation**

线性二次控制的强化学习在成本操纵下的脆弱性 eess.SY

This paper is yet to be peer-reviewed; Typos are corrected in ver 2

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2203.05774v2)

**Authors**: Yunhan Huang, Quanyan Zhu

**Abstracts**: In this work, we study the deception of a Linear-Quadratic-Gaussian (LQG) agent by manipulating the cost signals. We show that a small falsification of the cost parameters will only lead to a bounded change in the optimal policy. The bound is linear on the amount of falsification the attacker can apply to the cost parameters. We propose an attack model where the attacker aims to mislead the agent into learning a `nefarious' policy by intentionally falsifying the cost parameters. We formulate the attack's problem as a convex optimization problem and develop necessary and sufficient conditions to check the achievability of the attacker's goal.   We showcase the adversarial manipulation on two types of LQG learners: the batch RL learner and the other is the adaptive dynamic programming (ADP) learner. Our results demonstrate that with only 2.296% of falsification on the cost data, the attacker misleads the batch RL into learning the 'nefarious' policy that leads the vehicle to a dangerous position. The attacker can also gradually trick the ADP learner into learning the same `nefarious' policy by consistently feeding the learner a falsified cost signal that stays close to the actual cost signal. The paper aims to raise people's awareness of the security threats faced by RL-enabled control systems.

摘要: 在这项工作中，我们通过操纵费用信号来研究线性二次高斯(LQG)代理的欺骗。我们表明，对成本参数的微小篡改只会导致最优策略的有限变化。该界限与攻击者可以应用于成本参数的伪造量呈线性关系。我们提出了一个攻击模型，其中攻击者旨在通过故意伪造成本参数来误导代理学习“邪恶的”策略。我们将攻击问题描述为一个凸优化问题，并给出了检验攻击者目标可达性的充要条件。我们展示了在两种类型的LQG学习器上的对抗操作：批处理RL学习器和自适应动态规划(ADP)学习器。我们的结果表明，在只有2.296%的成本数据被篡改的情况下，攻击者误导批次RL学习将车辆引向危险位置的“邪恶”策略。攻击者还可以通过始终如一地向学习者提供接近实际成本信号的伪造成本信号，逐渐诱骗ADP学习者学习相同的“邪恶”策略。本文旨在提高人们对启用RL的控制系统所面临的安全威胁的认识。



## **2. IRShield: A Countermeasure Against Adversarial Physical-Layer Wireless Sensing**

IRShield：对抗敌意物理层无线侦听的对策 cs.CR

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2112.01967v2)

**Authors**: Paul Staat, Simon Mulzer, Stefan Roth, Veelasha Moonsamy, Markus Heinrichs, Rainer Kronberger, Aydin Sezgin, Christof Paar

**Abstracts**: Wireless radio channels are known to contain information about the surrounding propagation environment, which can be extracted using established wireless sensing methods. Thus, today's ubiquitous wireless devices are attractive targets for passive eavesdroppers to launch reconnaissance attacks. In particular, by overhearing standard communication signals, eavesdroppers obtain estimations of wireless channels which can give away sensitive information about indoor environments. For instance, by applying simple statistical methods, adversaries can infer human motion from wireless channel observations, allowing to remotely monitor premises of victims. In this work, building on the advent of intelligent reflecting surfaces (IRSs), we propose IRShield as a novel countermeasure against adversarial wireless sensing. IRShield is designed as a plug-and-play privacy-preserving extension to existing wireless networks. At the core of IRShield, we design an IRS configuration algorithm to obfuscate wireless channels. We validate the effectiveness with extensive experimental evaluations. In a state-of-the-art human motion detection attack using off-the-shelf Wi-Fi devices, IRShield lowered detection rates to 5% or less.

摘要: 众所周知，无线无线电信道包含有关周围传播环境的信息，可以使用已建立的无线侦听方法提取这些信息。因此，如今无处不在的无线设备是被动窃听者发动侦察攻击的诱人目标。特别是，通过窃听标准通信信号，窃听者获得对无线信道的估计，这可能会泄露有关室内环境的敏感信息。例如，通过应用简单的统计方法，攻击者可以从无线信道观测中推断人体运动，从而允许远程监控受害者的办公场所。在这项工作中，基于智能反射面(IRS)的出现，我们提出了IRShield作为一种新的对抗敌意无线传感的对策。IRShield被设计为现有无线网络的即插即用隐私保护扩展。在IRShield的核心部分，我们设计了一种IRS配置算法来对无线信道进行混淆。我们通过大量的实验评估验证了该方法的有效性。在一次使用现成Wi-Fi设备的最先进的人体运动检测攻击中，IRShield将检测率降至5%或更低。



## **3. Adversarial Machine Learning Attacks Against Video Anomaly Detection Systems**

针对视频异常检测系统的对抗性机器学习攻击 cs.CV

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2204.03141v1)

**Authors**: Furkan Mumcu, Keval Doshi, Yasin Yilmaz

**Abstracts**: Anomaly detection in videos is an important computer vision problem with various applications including automated video surveillance. Although adversarial attacks on image understanding models have been heavily investigated, there is not much work on adversarial machine learning targeting video understanding models and no previous work which focuses on video anomaly detection. To this end, we investigate an adversarial machine learning attack against video anomaly detection systems, that can be implemented via an easy-to-perform cyber-attack. Since surveillance cameras are usually connected to the server running the anomaly detection model through a wireless network, they are prone to cyber-attacks targeting the wireless connection. We demonstrate how Wi-Fi deauthentication attack, a notoriously easy-to-perform and effective denial-of-service (DoS) attack, can be utilized to generate adversarial data for video anomaly detection systems. Specifically, we apply several effects caused by the Wi-Fi deauthentication attack on video quality (e.g., slow down, freeze, fast forward, low resolution) to the popular benchmark datasets for video anomaly detection. Our experiments with several state-of-the-art anomaly detection models show that the attackers can significantly undermine the reliability of video anomaly detection systems by causing frequent false alarms and hiding physical anomalies from the surveillance system.

摘要: 视频中的异常检测是一个重要的计算机视觉问题，在包括自动视频监控在内的各种应用中都有应用。虽然针对图像理解模型的对抗性攻击已经得到了大量的研究，但针对视频理解模型的对抗性机器学习的研究还很少，也没有专门针对视频异常检测的工作。为此，我们研究了一种针对视频异常检测系统的对抗性机器学习攻击，该攻击可以通过易于执行的网络攻击来实现。由于监控摄像头通常通过无线网络连接到运行异常检测模型的服务器，因此它们容易受到针对无线连接的网络攻击。我们演示了如何利用Wi-Fi解除身份验证攻击，这是一种众所周知的易于执行和有效的拒绝服务(DoS)攻击，可以为视频异常检测系统生成敌意数据。具体地说，我们将Wi-Fi解除身份验证攻击对视频质量造成的几种影响(例如，减速、冻结、快进、低分辨率)应用于流行的基准数据集，用于视频异常检测。我们用几种最先进的异常检测模型进行的实验表明，攻击者可以通过频繁的误报警和对监控系统隐藏物理异常来显著破坏视频异常检测系统的可靠性。



## **4. Control barrier function based attack-recovery with provable guarantees**

具有可证明保证的基于控制屏障函数的攻击恢复 cs.SY

8 pages, 6 figures

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.03077v1)

**Authors**: Kunal Garg, Ricardo G. Sanfelice, Alvaro A. Cardenas

**Abstracts**: This paper studies provable security guarantees for cyber-physical systems (CPS) under actuator attacks. In particular, we consider CPS safety and propose a new attack-detection mechanism based on a zeroing control barrier function (ZCBF) condition. In addition we design an adaptive recovery mechanism based on how close the system is from violating safety. We show that the attack-detection mechanism is sound, i.e., there are no false negatives for adversarial attacks. Finally, we use a Quadratic Programming (QP) approach for online recovery (and nominal) control synthesis. We demonstrate the effectiveness of the proposed method in a simulation case study involving a quadrotor with an attack on its motors.

摘要: 研究了网络物理系统在执行器攻击下的可证明安全保证问题。特别是，我们考虑了CPS的安全性，提出了一种新的基于归零控制屏障函数(ZCBF)条件的攻击检测机制。此外，我们还设计了一种基于系统与违反安全的距离的自适应恢复机制。我们证明了攻击检测机制是健全的，即对于对抗性攻击没有漏报。最后，我们使用二次规划(QP)方法进行在线恢复(和标称)控制综合。在一个四旋翼发动机受到攻击的仿真案例研究中，我们证明了所提方法的有效性。



## **5. Sampling-based Fast Gradient Rescaling Method for Highly Transferable Adversarial Attacks**

一种基于采样的高可转移对抗性攻击快速梯度重缩放方法 cs.CV

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.02887v1)

**Authors**: Xu Han, Anmin Liu, Yifeng Xiong, Yanbo Fan, Kun He

**Abstracts**: Deep neural networks have shown to be very vulnerable to adversarial examples crafted by adding human-imperceptible perturbations to benign inputs. After achieving impressive attack success rates in the white-box setting, more focus is shifted to black-box attacks. In either case, the common gradient-based approaches generally use the $sign$ function to generate perturbations at the end of the process. However, only a few works pay attention to the limitation of the $sign$ function. Deviation between the original gradient and the generated noises may lead to inaccurate gradient update estimation and suboptimal solutions for adversarial transferability, which is crucial for black-box attacks. To address this issue, we propose a Sampling-based Fast Gradient Rescaling Method (S-FGRM) to improve the transferability of the crafted adversarial examples. Specifically, we use data rescaling to substitute the inefficient $sign$ function in gradient-based attacks without extra computational cost. We also propose a Depth First Sampling method to eliminate the fluctuation of rescaling and stabilize the gradient update. Our method can be used in any gradient-based optimizations and is extensible to be integrated with various input transformation or ensemble methods for further improving the adversarial transferability. Extensive experiments on the standard ImageNet dataset show that our S-FGRM could significantly boost the transferability of gradient-based attacks and outperform the state-of-the-art baselines.

摘要: 深度神经网络已被证明非常容易受到敌意例子的攻击，这些例子是通过在良性输入中添加人类无法察觉的扰动来构建的。在白盒攻击中取得了令人印象深刻的攻击成功率之后，更多的注意力转移到了黑盒攻击上。在这两种情况下，常见的基于梯度的方法通常使用$SIGN$函数在过程结束时生成扰动。然而，只有少数著作注意到$SIGN$函数的局限性。原始梯度与产生的噪声之间的偏差可能会导致不准确的梯度更新估计和对抗性转移的次优解，这是黑盒攻击的关键。针对这一问题，我们提出了一种基于采样的快速梯度重缩放方法(S-FGRM)来提高恶意例子的可转移性。具体地说，在基于梯度的攻击中，我们使用数据重缩放来代替低效的$sign$函数，而不需要额外的计算代价。我们还提出了深度优先采样的方法，消除了重缩放的波动，稳定了梯度更新。我们的方法可以用于任何基于梯度的优化，并且可以扩展到与各种输入变换或集成方法相集成，以进一步提高对抗性转移。在标准ImageNet数据集上的大量实验表明，我们的S-FGRM可以显著提高基于梯度的攻击的可转移性，并优于最新的基线。



## **6. Distilling Robust and Non-Robust Features in Adversarial Examples by Information Bottleneck**

利用信息瓶颈从对抗性实例中提取稳健和非稳健特征 cs.LG

NeurIPS 2021

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.02735v1)

**Authors**: Junho Kim, Byung-Kwan Lee, Yong Man Ro

**Abstracts**: Adversarial examples, generated by carefully crafted perturbation, have attracted considerable attention in research fields. Recent works have argued that the existence of the robust and non-robust features is a primary cause of the adversarial examples, and investigated their internal interactions in the feature space. In this paper, we propose a way of explicitly distilling feature representation into the robust and non-robust features, using Information Bottleneck. Specifically, we inject noise variation to each feature unit and evaluate the information flow in the feature representation to dichotomize feature units either robust or non-robust, based on the noise variation magnitude. Through comprehensive experiments, we demonstrate that the distilled features are highly correlated with adversarial prediction, and they have human-perceptible semantic information by themselves. Furthermore, we present an attack mechanism intensifying the gradient of non-robust features that is directly related to the model prediction, and validate its effectiveness of breaking model robustness.

摘要: 由精心设计的扰动产生的对抗性例子在研究领域引起了相当大的关注。最近的工作认为，稳健特征和非稳健特征的存在是造成对抗性例子的主要原因，并研究了它们在特征空间中的内在交互作用。在本文中，我们提出了一种利用信息瓶颈将特征表示显式提取为稳健和非稳健特征的方法。具体地说，我们将噪声变化注入到每个特征单元中，并评估特征表示中的信息流，以基于噪声变化的大小来区分稳健或非稳健的特征单元。通过综合实验，我们证明所提取的特征与对抗性预测高度相关，并且它们本身就具有人类可感知的语义信息。此外，提出了一种增强与模型预测直接相关的非稳健特征梯度的攻击机制，并验证了其打破模型稳健性的有效性。



## **7. Rolling Colors: Adversarial Laser Exploits against Traffic Light Recognition**

滚动颜色：对抗红绿灯识别的激光攻击 cs.CV

To be published in USENIX Security 2022

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.02675v1)

**Authors**: Chen Yan, Zhijian Xu, Zhanyuan Yin, Xiaoyu Ji, Wenyuan Xu

**Abstracts**: Traffic light recognition is essential for fully autonomous driving in urban areas. In this paper, we investigate the feasibility of fooling traffic light recognition mechanisms by shedding laser interference on the camera. By exploiting the rolling shutter of CMOS sensors, we manage to inject a color stripe overlapped on the traffic light in the image, which can cause a red light to be recognized as a green light or vice versa. To increase the success rate, we design an optimization method to search for effective laser parameters based on empirical models of laser interference. Our evaluation in emulated and real-world setups on 2 state-of-the-art recognition systems and 5 cameras reports a maximum success rate of 30% and 86.25% for Red-to-Green and Green-to-Red attacks. We observe that the attack is effective in continuous frames from more than 40 meters away against a moving vehicle, which may cause end-to-end impacts on self-driving such as running a red light or emergency stop. To mitigate the threat, we propose redesigning the rolling shutter mechanism.

摘要: 红绿灯识别是城市地区实现全自动驾驶的关键。在本文中，我们研究了通过在摄像机上散布激光干涉来欺骗交通灯识别机制的可行性。通过利用CMOS传感器的滚动快门，我们成功地在图像中的交通灯上注入了重叠的彩色条纹，这可以使红灯被识别为绿灯，反之亦然。为了提高成功率，我们设计了一种基于激光干涉经验模型的优化方法来搜索有效的激光参数。我们在2个最先进的识别系统和5个摄像头上的模拟和真实设置中的评估报告，红色到绿色和绿色到红色攻击的最大成功率分别为30%和86.25%。我们观察到，攻击在40米以外的连续帧中对移动的车辆有效，这可能会对自动驾驶造成端到端的影响，如闯红灯或紧急停车。为了减轻威胁，我们建议重新设计滚动快门机构。



## **8. Adversarial Analysis of the Differentially-Private Federated Learning in Cyber-Physical Critical Infrastructures**

网络物理关键基础设施中非私有联邦学习的对抗性分析 cs.CR

11 pages, 5 figures, 4 tables. This work has been submitted to IEEE  for possible publication. Copyright may be transferred without notice, after  which this version may no longer be accessible

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.02654v1)

**Authors**: Md Tamjid Hossain, Shahriar Badsha, Hung, La, Haoting Shen, Shafkat Islam, Ibrahim Khalil, Xun Yi

**Abstracts**: Differential privacy (DP) is considered to be an effective privacy-preservation method to secure the promising distributed machine learning (ML) paradigm-federated learning (FL) from privacy attacks (e.g., membership inference attack). Nevertheless, while the DP mechanism greatly alleviates privacy concerns, recent studies have shown that it can be exploited to conduct security attacks (e.g., false data injection attacks). To address such attacks on FL-based applications in critical infrastructures, in this paper, we perform the first systematic study on the DP-exploited poisoning attacks from an adversarial point of view. We demonstrate that the DP method, despite providing a level of privacy guarantee, can effectively open a new poisoning attack vector for the adversary. Our theoretical analysis and empirical evaluation of a smart grid dataset show the FL performance degradation (sub-optimal model generation) scenario due to the differential noise-exploited selective model poisoning attacks. As a countermeasure, we propose a reinforcement learning-based differential privacy level selection (rDP) process. The rDP process utilizes the differential privacy parameters (privacy loss, information leakage probability, etc.) and the losses to intelligently generate an optimal privacy level for the nodes. The evaluation shows the accumulated reward and errors of the proposed technique converge to an optimal privacy policy.

摘要: 差分隐私(DP)被认为是一种有效的隐私保护方法，可以保护分布式机器学习(ML)范型联合学习(FL)免受隐私攻击(如成员推理攻击)。然而，虽然DP机制极大地缓解了对隐私的担忧，但最近的研究表明，它可以被利用来进行安全攻击(例如，虚假数据注入攻击)。为了解决这类针对关键基础设施中基于FL的应用程序的攻击，本文首次从对抗的角度对DP利用的中毒攻击进行了系统的研究。我们证明，虽然DP方法提供了一定程度的隐私保障，但可以有效地为攻击者打开一个新的中毒攻击载体。我们的理论分析和对智能电网数据集的经验评估表明，差值噪声利用的选择性模型中毒攻击导致FL性能下降(次优模型生成)。作为对策，我们提出了一种基于强化学习的差异隐私级别选择(RDP)过程。RDP过程使用不同的隐私参数(隐私丢失、信息泄露概率等)。以及智能地为节点生成最佳隐私级别的损失。评估结果表明，该技术的累积奖赏和误差收敛于最优隐私策略。



## **9. Who Is the Strongest Enemy? Towards Optimal and Efficient Evasion Attacks in Deep RL**

谁是最强大的敌人？面向最优高效逃避攻击的Deep RL cs.LG

In the 10th International Conference on Learning Representations  (ICLR 2022)

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2106.05087v4)

**Authors**: Yanchao Sun, Ruijie Zheng, Yongyuan Liang, Furong Huang

**Abstracts**: Evaluating the worst-case performance of a reinforcement learning (RL) agent under the strongest/optimal adversarial perturbations on state observations (within some constraints) is crucial for understanding the robustness of RL agents. However, finding the optimal adversary is challenging, in terms of both whether we can find the optimal attack and how efficiently we can find it. Existing works on adversarial RL either use heuristics-based methods that may not find the strongest adversary, or directly train an RL-based adversary by treating the agent as a part of the environment, which can find the optimal adversary but may become intractable in a large state space. This paper introduces a novel attacking method to find the optimal attacks through collaboration between a designed function named "actor" and an RL-based learner named "director". The actor crafts state perturbations for a given policy perturbation direction, and the director learns to propose the best policy perturbation directions. Our proposed algorithm, PA-AD, is theoretically optimal and significantly more efficient than prior RL-based works in environments with large state spaces. Empirical results show that our proposed PA-AD universally outperforms state-of-the-art attacking methods in various Atari and MuJoCo environments. By applying PA-AD to adversarial training, we achieve state-of-the-art empirical robustness in multiple tasks under strong adversaries.

摘要: 评估强化学习(RL)代理在状态观测(在某些约束范围内)的最强/最优对抗扰动下的最坏情况下的性能对于理解RL代理的稳健性至关重要。然而，就我们是否能找到最佳攻击以及找到最佳攻击的效率而言，找到最佳对手是具有挑战性的。现有的对抗性RL研究要么使用基于启发式的方法，可能找不到最强的对手，要么将智能体视为环境的一部分，直接训练基于RL的对手，这可以找到最优的对手，但在大的状态空间中可能变得难以处理。本文提出了一种新的攻击方法，通过设计一个名为“参与者”的函数和一个名为“导演”的基于RL的学习器之间的协作来寻找最优攻击。参与者为给定的政策扰动方向制作状态扰动，导演学习提出最佳政策扰动方向。我们提出的算法PA-AD在理论上是最优的，并且在具有大状态空间的环境中比以前的基于RL的工作更有效。实验结果表明，在不同的Atari和MuJoCo环境下，我们提出的PA-AD攻击方法的性能普遍优于最新的攻击方法。通过将PA-AD应用于对抗性训练，我们在强对手下的多任务中获得了最先进的经验稳健性。



## **10. Patch-Fool: Are Vision Transformers Always Robust Against Adversarial Perturbations?**

补丁-愚人：视觉变形金刚在对抗敌方干扰时总是健壮吗？ cs.CV

Accepted at ICLR 2022

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2203.08392v2)

**Authors**: Yonggan Fu, Shunyao Zhang, Shang Wu, Cheng Wan, Yingyan Lin

**Abstracts**: Vision transformers (ViTs) have recently set off a new wave in neural architecture design thanks to their record-breaking performance in various vision tasks. In parallel, to fulfill the goal of deploying ViTs into real-world vision applications, their robustness against potential malicious attacks has gained increasing attention. In particular, recent works show that ViTs are more robust against adversarial attacks as compared with convolutional neural networks (CNNs), and conjecture that this is because ViTs focus more on capturing global interactions among different input/feature patches, leading to their improved robustness to local perturbations imposed by adversarial attacks. In this work, we ask an intriguing question: "Under what kinds of perturbations do ViTs become more vulnerable learners compared to CNNs?" Driven by this question, we first conduct a comprehensive experiment regarding the robustness of both ViTs and CNNs under various existing adversarial attacks to understand the underlying reason favoring their robustness. Based on the drawn insights, we then propose a dedicated attack framework, dubbed Patch-Fool, that fools the self-attention mechanism by attacking its basic component (i.e., a single patch) with a series of attention-aware optimization techniques. Interestingly, our Patch-Fool framework shows for the first time that ViTs are not necessarily more robust than CNNs against adversarial perturbations. In particular, we find that ViTs are more vulnerable learners compared with CNNs against our Patch-Fool attack which is consistent across extensive experiments, and the observations from Sparse/Mild Patch-Fool, two variants of Patch-Fool, indicate an intriguing insight that the perturbation density and strength on each patch seem to be the key factors that influence the robustness ranking between ViTs and CNNs.

摘要: 视觉转换器(VITS)最近掀起了神经结构设计的新浪潮，这要归功于它们在各种视觉任务中的创纪录表现。与此同时，为了实现将VITS部署到现实世界视觉应用中的目标，它们对潜在恶意攻击的健壮性得到了越来越多的关注。特别是，最近的研究表明，与卷积神经网络(CNN)相比，VITS对对抗攻击具有更强的鲁棒性，推测这是因为VITS更注重捕捉不同输入/特征块之间的全局交互，从而提高了它们对敌对攻击造成的局部扰动的鲁棒性。在这项工作中，我们提出了一个耐人寻味的问题：“在什么样的扰动下，VITS比CNN更容易成为学习者？”在这个问题的驱动下，我们首先对VITS和CNN在各种现有的对抗性攻击下的健壮性进行了全面的实验，以了解有利于其健壮性的潜在原因。在此基础上，我们提出了一个专门的攻击框架，称为Patch-Fool，它通过使用一系列注意力感知优化技术来攻击自我注意机制的基本组成部分(即单个补丁)来愚弄自我注意机制。有趣的是，我们的Patch-Fool框架首次表明，VITS在对抗对手扰动时并不一定比CNN更健壮。特别是，我们发现VITS比CNN更容易学习，这在广泛的实验中是一致的，并且来自Patch-Fool的两个变种稀疏/温和Patch-Fool的观察表明，每个补丁上的扰动密度和强度似乎是影响VITS和CNN之间健壮性排名的关键因素。



## **11. Exploring Robust Architectures for Deep Artificial Neural Networks**

探索深度人工神经网络的健壮体系结构 cs.LG

27 pages, 16 figures

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2106.15850v2)

**Authors**: Asim Waqas, Ghulam Rasool, Hamza Farooq, Nidhal C. Bouaynaya

**Abstracts**: The architectures of deep artificial neural networks (DANNs) are routinely studied to improve their predictive performance. However, the relationship between the architecture of a DANN and its robustness to noise and adversarial attacks is less explored. We investigate how the robustness of DANNs relates to their underlying graph architectures or structures. This study: (1) starts by exploring the design space of architectures of DANNs using graph-theoretic robustness measures; (2) transforms the graphs to DANN architectures to train/validate/test on various image classification tasks; (3) explores the relationship between the robustness of trained DANNs against noise and adversarial attacks and the robustness of their underlying architectures estimated via graph-theoretic measures. We show that the topological entropy and Olivier-Ricci curvature of the underlying graphs can quantify the robustness performance of DANNs. The said relationship is stronger for complex tasks and large DANNs. Our work will allow autoML and neural architecture search community to explore design spaces of robust and accurate DANNs.

摘要: 人们经常研究深度人工神经网络(DEN)的结构，以提高其预测性能。然而，DANN的体系结构与其对噪声和敌意攻击的稳健性之间的关系却鲜有人研究。我们研究了DNA的健壮性如何与其底层的图体系结构或结构相关。本研究：(1)使用图论稳健性度量方法探索DANN体系结构的设计空间；(2)将图转换为DANN体系结构，以对各种图像分类任务进行训练/验证/测试；(3)探索训练的DANN体系结构对噪声和敌对攻击的健壮性与其底层体系结构的健壮性之间的关系。我们证明了基础图的拓扑熵和Olivier-Ricci曲率可以量化DANS的稳健性。对于复杂的任务和大的丹尼，上述关系更加牢固。我们的工作将允许AutoML和神经架构搜索社区探索健壮和准确的DAN的设计空间。



## **12. User-Level Differential Privacy against Attribute Inference Attack of Speech Emotion Recognition in Federated Learning**

联合学习中抵抗语音情感识别属性推理攻击的用户级差分隐私 cs.CR

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2204.02500v1)

**Authors**: Tiantian Feng, Raghuveer Peri, Shrikanth Narayanan

**Abstracts**: Many existing privacy-enhanced speech emotion recognition (SER) frameworks focus on perturbing the original speech data through adversarial training within a centralized machine learning setup. However, this privacy protection scheme can fail since the adversary can still access the perturbed data. In recent years, distributed learning algorithms, especially federated learning (FL), have gained popularity to protect privacy in machine learning applications. While FL provides good intuition to safeguard privacy by keeping the data on local devices, prior work has shown that privacy attacks, such as attribute inference attacks, are achievable for SER systems trained using FL. In this work, we propose to evaluate the user-level differential privacy (UDP) in mitigating the privacy leaks of the SER system in FL. UDP provides theoretical privacy guarantees with privacy parameters $\epsilon$ and $\delta$. Our results show that the UDP can effectively decrease attribute information leakage while keeping the utility of the SER system with the adversary accessing one model update. However, the efficacy of the UDP suffers when the FL system leaks more model updates to the adversary. We make the code publicly available to reproduce the results in https://github.com/usc-sail/fed-ser-leakage.

摘要: 许多现有的隐私增强型语音情感识别(SER)框架专注于通过集中式机器学习设置中的对抗性训练来扰乱原始语音数据。然而，这种隐私保护方案可能会失败，因为攻击者仍然可以访问受干扰的数据。近年来，分布式学习算法，特别是联邦学习(FL)算法在机器学习应用中保护隐私得到了广泛的应用。虽然FL通过将数据保存在本地设备上来提供良好的直觉来保护隐私，但先前的工作表明，使用FL训练的SER系统可以实现隐私攻击，例如属性推理攻击。在这项工作中，我们建议评估用户级差异隐私(UDP)在缓解FL中SER系统的隐私泄漏方面的作用。UDP通过隐私参数$\epsilon$和$\Delta$提供理论上的隐私保证。实验结果表明，UDP协议在保持SER系统可用性的同时，有效地减少了属性信息泄露，且攻击者只需访问一次模型更新。然而，当FL系统向对手泄露更多的模型更新时，UDP的效率会受到影响。我们将代码公开，以便在https://github.com/usc-sail/fed-ser-leakage.中重现结果



## **13. Training-Free Robust Multimodal Learning via Sample-Wise Jacobian Regularization**

基于样本明智雅可比正则化的免训练鲁棒多模学习 cs.CV

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2204.02485v1)

**Authors**: Zhengqi Gao, Sucheng Ren, Zihui Xue, Siting Li, Hang Zhao

**Abstracts**: Multimodal fusion emerges as an appealing technique to improve model performances on many tasks. Nevertheless, the robustness of such fusion methods is rarely involved in the present literature. In this paper, we propose a training-free robust late-fusion method by exploiting conditional independence assumption and Jacobian regularization. Our key is to minimize the Frobenius norm of a Jacobian matrix, where the resulting optimization problem is relaxed to a tractable Sylvester equation. Furthermore, we provide a theoretical error bound of our method and some insights about the function of the extra modality. Several numerical experiments on AV-MNIST, RAVDESS, and VGGsound demonstrate the efficacy of our method under both adversarial attacks and random corruptions.

摘要: 多通道融合是提高模型在许多任务上性能的一种很有吸引力的技术。然而，这种融合方法的稳健性在目前的文献中很少涉及。本文利用条件独立性假设和雅可比正则化，提出了一种无需训练的鲁棒晚融合方法。我们的关键是最小化雅可比矩阵的Frobenius范数，由此产生的优化问题被松弛到一个容易处理的Sylvester方程。此外，我们还给出了该方法的理论误差界，并对额外通道的作用提出了一些见解。在AV-MNIST、RAVDESS和VGGound上的几个数值实验证明了我们的方法在对抗攻击和随机破坏下的有效性。



## **14. Hear No Evil: Towards Adversarial Robustness of Automatic Speech Recognition via Multi-Task Learning**

听而不闻：通过多任务学习实现自动语音识别的对抗健壮性 eess.AS

Submitted to Insterspeech 2022

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2204.02381v1)

**Authors**: Nilaksh Das, Duen Horng Chau

**Abstracts**: As automatic speech recognition (ASR) systems are now being widely deployed in the wild, the increasing threat of adversarial attacks raises serious questions about the security and reliability of using such systems. On the other hand, multi-task learning (MTL) has shown success in training models that can resist adversarial attacks in the computer vision domain. In this work, we investigate the impact of performing such multi-task learning on the adversarial robustness of ASR models in the speech domain. We conduct extensive MTL experimentation by combining semantically diverse tasks such as accent classification and ASR, and evaluate a wide range of adversarial settings. Our thorough analysis reveals that performing MTL with semantically diverse tasks consistently makes it harder for an adversarial attack to succeed. We also discuss in detail the serious pitfalls and their related remedies that have a significant impact on the robustness of MTL models. Our proposed MTL approach shows considerable absolute improvements in adversarially targeted WER ranging from 17.25 up to 59.90 compared to single-task learning baselines (attention decoder and CTC respectively). Ours is the first in-depth study that uncovers adversarial robustness gains from multi-task learning for ASR.

摘要: 随着自动语音识别(ASR)系统的广泛应用，日益增长的对抗性攻击威胁对使用这类系统的安全性和可靠性提出了严重的问题。另一方面，多任务学习(MTL)在训练模型抵抗计算机视觉领域中的敌意攻击方面取得了成功。在这项工作中，我们研究了执行这种多任务学习对ASR模型在语音域的对抗健壮性的影响。我们通过结合重音分类和ASR等语义多样化的任务来进行广泛的MTL实验，并评估了广泛的对抗性环境。我们的全面分析表明，以语义多样化的任务执行MTL始终会使敌方攻击更难成功。我们还详细讨论了对MTL模型的稳健性有重大影响的严重陷阱及其相关补救措施。与单任务学习基线(注意解码器和CTC)相比，我们提出的MTL方法在相反的目标WER上有相当大的绝对改善，从17.25%到59.90%。我们的研究是第一次深入研究ASR从多任务学习中获得的对手健壮性收益。



## **15. A Survey of Adversarial Learning on Graphs**

图上的对抗性学习研究综述 cs.LG

Preprint; 16 pages, 2 figures

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2003.05730v3)

**Authors**: Liang Chen, Jintang Li, Jiaying Peng, Tao Xie, Zengxu Cao, Kun Xu, Xiangnan He, Zibin Zheng, Bingzhe Wu

**Abstracts**: Deep learning models on graphs have achieved remarkable performance in various graph analysis tasks, e.g., node classification, link prediction, and graph clustering. However, they expose uncertainty and unreliability against the well-designed inputs, i.e., adversarial examples. Accordingly, a line of studies has emerged for both attack and defense addressed in different graph analysis tasks, leading to the arms race in graph adversarial learning. Despite the booming works, there still lacks a unified problem definition and a comprehensive review. To bridge this gap, we investigate and summarize the existing works on graph adversarial learning tasks systemically. Specifically, we survey and unify the existing works w.r.t. attack and defense in graph analysis tasks, and give appropriate definitions and taxonomies at the same time. Besides, we emphasize the importance of related evaluation metrics, investigate and summarize them comprehensively. Hopefully, our works can provide a comprehensive overview and offer insights for the relevant researchers. Latest advances in graph adversarial learning are summarized in our GitHub repository https://github.com/EdisonLeeeee/Graph-Adversarial-Learning.

摘要: 图的深度学习模型在各种图分析任务中取得了显著的性能，如节点分类、链接预测、图聚类等。然而，它们暴露了相对于精心设计的投入的不确定性和不可靠性，即对抗性例子。相应地，在不同的图分析任务中出现了一系列针对攻击和防御的研究，导致了图对抗学习中的军备竞赛。尽管工作开展得如火如荼，但仍缺乏统一的问题定义和全面审查。为了弥补这一差距，我们系统地调查和总结了已有的关于图对抗性学习任务的工作。具体地说，我们对现有的作品进行了调查和统一。图分析任务中的攻击和防御，同时给出相应的定义和分类。此外，我们还强调了相关评价指标的重要性，并对其进行了全面的调查和总结。希望我们的工作能够提供一个全面的概述，并为相关研究人员提供见解。在我们的GitHub知识库https://github.com/EdisonLeeeee/Graph-Adversarial-Learning.中总结了图形对抗性学习的最新进展



## **16. Training strategy for a lightweight countermeasure model for automatic speaker verification**

一种轻量级说话人自动确认对策模型的训练策略 cs.SD

ASVspoof2021

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2203.17031v2)

**Authors**: Yen-Lun Liao, Xuanjun Chen, Chung-Che Wang, Jyh-Shing Roger Jang

**Abstracts**: The countermeasure (CM) model is developed to protect Automatic Speaker Verification (ASV) systems from spoof attacks and prevent resulting personal information leakage. Based on practicality and security considerations, the CM model is usually deployed on edge devices, which have more limited computing resources and storage space than cloud-based systems. This work proposes training strategies for a lightweight CM model for ASV, using generalized end-to-end (GE2E) pre-training and adversarial fine-tuning to improve performance, and applying knowledge distillation (KD) to reduce the size of the CM model. In the evaluation phase of the ASVspoof 2021 Logical Access task, the lightweight ResNetSE model reaches min t-DCF 0.2695 and EER 3.54%. Compared to the teacher model, the lightweight student model only uses 22.5% of parameters and 21.1% of multiply and accumulate operands of the teacher model.

摘要: 对策(CM)模型是为了保护自动说话人验证(ASV)系统免受欺骗攻击并防止由此导致的个人信息泄露而开发的。基于实用性和安全性考虑，CM模型通常部署在边缘设备上，与基于云的系统相比，边缘设备的计算资源和存储空间更有限。提出了一种面向ASV的轻量级CM模型的训练策略，使用通用端到端(GE2E)预训练和对抗性微调来提高性能，并应用知识蒸馏(KD)来减小CM模型的规模。在ASVspoof2021逻辑访问任务的评估阶段，轻量级ResNetSE模型达到了最小t-DCF值0.2695和EER3.54%.与教师模型相比，轻量级学生模型仅使用了教师模型22.5%的参数和21.1%的乘加操作数。



## **17. Understanding and Improving Graph Injection Attack by Promoting Unnoticeability**

通过提高不可察觉来理解和改进图注入攻击 cs.LG

ICLR2022, 42 pages, 22 figures

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2202.08057v2)

**Authors**: Yongqiang Chen, Han Yang, Yonggang Zhang, Kaili Ma, Tongliang Liu, Bo Han, James Cheng

**Abstracts**: Recently Graph Injection Attack (GIA) emerges as a practical attack scenario on Graph Neural Networks (GNNs), where the adversary can merely inject few malicious nodes instead of modifying existing nodes or edges, i.e., Graph Modification Attack (GMA). Although GIA has achieved promising results, little is known about why it is successful and whether there is any pitfall behind the success. To understand the power of GIA, we compare it with GMA and find that GIA can be provably more harmful than GMA due to its relatively high flexibility. However, the high flexibility will also lead to great damage to the homophily distribution of the original graph, i.e., similarity among neighbors. Consequently, the threats of GIA can be easily alleviated or even prevented by homophily-based defenses designed to recover the original homophily. To mitigate the issue, we introduce a novel constraint -- homophily unnoticeability that enforces GIA to preserve the homophily, and propose Harmonious Adversarial Objective (HAO) to instantiate it. Extensive experiments verify that GIA with HAO can break homophily-based defenses and outperform previous GIA attacks by a significant margin. We believe our methods can serve for a more reliable evaluation of the robustness of GNNs.

摘要: 图注入攻击(GIA)是近年来在图神经网络(GNN)上出现的一种实用攻击方案，即攻击者只能注入少量恶意节点，而不需要修改已有的节点或边，即图修改攻击(GMA)。尽管GIA取得了令人振奋的成果，但人们对它为什么成功以及成功背后是否存在陷阱知之甚少。为了理解GIA的力量，我们将其与GMA进行比较，发现由于其相对较高的灵活性，GIA显然比GMA更具危害性。然而，较高的灵活性也会对原图的同源分布造成很大的破坏，即邻域间的相似性。因此，GIA的威胁可以很容易地减轻，甚至可以通过基于同质性的防御措施来恢复原始的同质性。为了缓解这一问题，我们引入了一种新的约束--同形不可察觉，强制GIA保持同形，并提出了和谐对抗目标(HAO)来实例化它。广泛的实验证明，带有HAO的GIA可以打破基于同源的防御，并显著超过之前的GIA攻击。我们相信，我们的方法可以更可靠地评估GNN的稳健性。



## **18. Adversarial Detection without Model Information**

无模型信息的对抗性检测 cs.CV

This paper has 14 pages of content and 2 pages of references

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2202.04271v2)

**Authors**: Abhishek Moitra, Youngeun Kim, Priyadarshini Panda

**Abstracts**: Prior state-of-the-art adversarial detection works are classifier model dependent, i.e., they require classifier model outputs and parameters for training the detector or during adversarial detection. This makes their detection approach classifier model specific. Furthermore, classifier model outputs and parameters might not always be accessible. To this end, we propose a classifier model independent adversarial detection method using a simple energy function to distinguish between adversarial and natural inputs. We train a standalone detector independent of the classifier model, with a layer-wise energy separation (LES) training to increase the separation between natural and adversarial energies. With this, we perform energy distribution-based adversarial detection. Our method achieves comparable performance with state-of-the-art detection works (ROC-AUC > 0.9) across a wide range of gradient, score and gaussian noise attacks on CIFAR10, CIFAR100 and TinyImagenet datasets. Furthermore, compared to prior works, our detection approach is light-weight, requires less amount of training data (40% of the actual dataset) and is transferable across different datasets. For reproducibility, we provide layer-wise energy separation training code at https://github.com/Intelligent-Computing-Lab-Yale/Energy-Separation-Training

摘要: 以往的对抗性检测工作依赖于分类器模型，即它们需要分类器模型的输出和参数来训练检测器或在对抗性检测过程中。这使得它们的检测方法具有分类器模型的特殊性。此外，分类器模型输出和参数可能并不总是可访问的。为此，我们提出了一种独立于分类器模型的敌意检测方法，该方法使用一个简单的能量函数来区分敌意输入和自然输入。我们训练一个独立于分类器模型的独立检测器，通过分层能量分离(LES)训练来增加自然能量和敌对能量之间的分离。在此基础上，我们进行了基于能量分布的敌意检测。我们的方法在CIFAR10、CIFAR100和TinyImagenet数据集上的各种梯度、得分和高斯噪声攻击下获得了与最先进的检测工作(ROC-AUC>0.9)相当的性能。此外，与以前的工作相比，我们的检测方法是轻量级的，需要更少的训练数据量(实际数据集的40%)，并且可以在不同的数据集之间传输。为了重现性，我们在https://github.com/Intelligent-Computing-Lab-Yale/Energy-Separation-Training上提供了分层能量分离培训代码



## **19. GAIL-PT: A Generic Intelligent Penetration Testing Framework with Generative Adversarial Imitation Learning**

GAIL-PT：一种具有生成性对抗性模仿学习的通用智能渗透测试框架 cs.CR

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2204.01975v1)

**Authors**: Jinyin Chen, Shulong Hu, Haibin Zheng, Changyou Xing, Guomin Zhang

**Abstracts**: Penetration testing (PT) is an efficient network testing and vulnerability mining tool by simulating a hacker's attack for valuable information applied in some areas. Compared with manual PT, intelligent PT has become a dominating mainstream due to less time-consuming and lower labor costs. Unfortunately, RL-based PT is still challenged in real exploitation scenarios because the agent's action space is usually high-dimensional discrete, thus leading to algorithm convergence difficulty. Besides, most PT methods still rely on the decisions of security experts. Addressing the challenges, for the first time, we introduce expert knowledge to guide the agent to make better decisions in RL-based PT and propose a Generative Adversarial Imitation Learning-based generic intelligent Penetration testing framework, denoted as GAIL-PT, to solve the problems of higher labor costs due to the involvement of security experts and high-dimensional discrete action space. Specifically, first, we manually collect the state-action pairs to construct an expert knowledge base when the pre-trained RL / DRL model executes successful penetration testings. Second, we input the expert knowledge and the state-action pairs generated online by the different RL / DRL models into the discriminator of GAIL for training. At last, we apply the output reward of the discriminator to guide the agent to perform the action with a higher penetration success rate to improve PT's performance. Extensive experiments conducted on the real target host and simulated network scenarios show that GAIL-PT achieves the SOTA penetration performance against DeepExploit in exploiting actual target Metasploitable2 and Q-learning in optimizing penetration path, not only in small-scale with or without honey-pot network environments but also in the large-scale virtual network environment.

摘要: 渗透测试(PT)是一种有效的网络测试和漏洞挖掘工具，通过模拟黑客对某些领域应用的有价值信息的攻击而实现。与人工PT相比，智能PT由于耗时更少、人力成本更低而成为主流。遗憾的是，基于RL的PT在实际开发场景中仍然面临挑战，因为智能体的动作空间通常是高维离散的，从而导致算法收敛困难。此外，大多数PT方法仍然依赖于安全专家的决策。针对这些挑战，我们首次在基于RL的PT中引入专家知识来指导智能体做出更好的决策，并提出了一种基于生成性对抗模仿学习的通用智能渗透测试框架GAIL-PT，以解决由于安全专家的参与和高维离散动作空间而导致的人工成本较高的问题。具体地说，首先，当预先训练的RL/DRL模型执行成功的渗透测试时，我们手动收集状态-动作对来构建专家知识库。其次，将不同RL/DRL模型在线生成的专家知识和状态-动作对输入到GAIL的鉴别器中进行训练。最后，我们利用鉴别器的输出奖励来指导智能体执行具有较高渗透成功率的动作，以提高PT的性能。在真实目标主机和模拟网络场景上进行的大量实验表明，无论是在有或没有蜜罐网络环境中，Gail-PT在利用实际目标元可分性2和Q-学习优化穿透路径方面都达到了DeepDevelopit的SOTA穿透性能。



## **20. Recent improvements of ASR models in the face of adversarial attacks**

面对对抗性攻击的ASR模型的最新改进 cs.CR

Submitted to Interspeech 2022

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2203.16536v2)

**Authors**: Raphael Olivier, Bhiksha Raj

**Abstracts**: Like many other tasks involving neural networks, Speech Recognition models are vulnerable to adversarial attacks. However recent research has pointed out differences between attacks and defenses on ASR models compared to image models. Improving the robustness of ASR models requires a paradigm shift from evaluating attacks on one or a few models to a systemic approach in evaluation. We lay the ground for such research by evaluating on various architectures a representative set of adversarial attacks: targeted and untargeted, optimization and speech processing-based, white-box, black-box and targeted attacks. Our results show that the relative strengths of different attack algorithms vary considerably when changing the model architecture, and that the results of some attacks are not to be blindly trusted. They also indicate that training choices such as self-supervised pretraining can significantly impact robustness by enabling transferable perturbations. We release our source code as a package that should help future research in evaluating their attacks and defenses.

摘要: 像涉及神经网络的许多其他任务一样，语音识别模型容易受到敌意攻击。然而，最近的研究指出，与图像模型相比，ASR模型在攻击和防御方面存在差异。要提高ASR模型的稳健性，需要从评估针对一个或几个模型的攻击转变为评估的系统性方法。我们通过在不同的体系结构上评估一组具有代表性的对抗性攻击：目标攻击和非目标攻击、基于优化和语音处理的攻击、白盒攻击、黑盒攻击和目标攻击，为这类研究奠定了基础。结果表明，随着模型结构的改变，不同攻击算法的相对强度有很大差异，某些攻击的结果不能盲目信任。它们还表明，自我监督预训练等训练选择可以通过实现可转移的扰动来显著影响稳健性。我们将我们的源代码作为一个包发布，这应该有助于未来的研究评估他们的攻击和防御。



## **21. Experimental quantum adversarial learning with programmable superconducting qubits**

基于可编程超导量子比特的实验量子对抗学习 quant-ph

26 pages, 17 figures, 8 algorithms

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2204.01738v1)

**Authors**: Wenhui Ren, Weikang Li, Shibo Xu, Ke Wang, Wenjie Jiang, Feitong Jin, Xuhao Zhu, Jiachen Chen, Zixuan Song, Pengfei Zhang, Hang Dong, Xu Zhang, Jinfeng Deng, Yu Gao, Chuanyu Zhang, Yaozu Wu, Bing Zhang, Qiujiang Guo, Hekang Li, Zhen Wang, Jacob Biamonte, Chao Song, Dong-Ling Deng, H. Wang

**Abstracts**: Quantum computing promises to enhance machine learning and artificial intelligence. Different quantum algorithms have been proposed to improve a wide spectrum of machine learning tasks. Yet, recent theoretical works show that, similar to traditional classifiers based on deep classical neural networks, quantum classifiers would suffer from the vulnerability problem: adding tiny carefully-crafted perturbations to the legitimate original data samples would facilitate incorrect predictions at a notably high confidence level. This will pose serious problems for future quantum machine learning applications in safety and security-critical scenarios. Here, we report the first experimental demonstration of quantum adversarial learning with programmable superconducting qubits. We train quantum classifiers, which are built upon variational quantum circuits consisting of ten transmon qubits featuring average lifetimes of 150 $\mu$s, and average fidelities of simultaneous single- and two-qubit gates above 99.94% and 99.4% respectively, with both real-life images (e.g., medical magnetic resonance imaging scans) and quantum data. We demonstrate that these well-trained classifiers (with testing accuracy up to 99%) can be practically deceived by small adversarial perturbations, whereas an adversarial training process would significantly enhance their robustness to such perturbations. Our results reveal experimentally a crucial vulnerability aspect of quantum learning systems under adversarial scenarios and demonstrate an effective defense strategy against adversarial attacks, which provide a valuable guide for quantum artificial intelligence applications with both near-term and future quantum devices.

摘要: 量子计算有望增强机器学习和人工智能。已经提出了不同的量子算法来改进广泛的机器学习任务。然而，最近的理论研究表明，与基于深度经典神经网络的传统分类器类似，量子分类器将受到脆弱性问题的困扰：在合法的原始数据样本中添加精心设计的微小扰动，将有助于在相当高的置信度水平下进行错误预测。这将给未来量子机器学习在安全和安保关键场景中的应用带来严重问题。在这里，我们报告了第一个利用可编程超导量子比特进行量子对抗学习的实验演示。我们训练量子分类器，它建立在由10个传态量子比特组成的变分量子电路上，平均寿命为150$\MU$s，同时具有99.94%和99.4%以上的同时单量子比特门和双量子比特门的平均保真度，使用真实图像(例如医学磁共振成像扫描)和量子数据。我们证明了这些训练有素的分类器(测试准确率高达99%)实际上可以被微小的对抗性扰动所欺骗，而对抗性训练过程将显著增强它们对此类扰动的稳健性。我们的结果在实验上揭示了量子学习系统在对抗场景下的一个关键弱点，并展示了一种有效的防御策略，这为量子人工智能在近期和未来的量子设备应用提供了有价值的指导。



## **22. DAD: Data-free Adversarial Defense at Test Time**

DAD：测试时的无数据对抗性防御 cs.LG

WACV 2022. Project page: https://sites.google.com/view/dad-wacv22

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2204.01568v1)

**Authors**: Gaurav Kumar Nayak, Ruchit Rawal, Anirban Chakraborty

**Abstracts**: Deep models are highly susceptible to adversarial attacks. Such attacks are carefully crafted imperceptible noises that can fool the network and can cause severe consequences when deployed. To encounter them, the model requires training data for adversarial training or explicit regularization-based techniques. However, privacy has become an important concern, restricting access to only trained models but not the training data (e.g. biometric data). Also, data curation is expensive and companies may have proprietary rights over it. To handle such situations, we propose a completely novel problem of 'test-time adversarial defense in absence of training data and even their statistics'. We solve it in two stages: a) detection and b) correction of adversarial samples. Our adversarial sample detection framework is initially trained on arbitrary data and is subsequently adapted to the unlabelled test data through unsupervised domain adaptation. We further correct the predictions on detected adversarial samples by transforming them in Fourier domain and obtaining their low frequency component at our proposed suitable radius for model prediction. We demonstrate the efficacy of our proposed technique via extensive experiments against several adversarial attacks and for different model architectures and datasets. For a non-robust Resnet-18 model pre-trained on CIFAR-10, our detection method correctly identifies 91.42% adversaries. Also, we significantly improve the adversarial accuracy from 0% to 37.37% with a minimal drop of 0.02% in clean accuracy on state-of-the-art 'Auto Attack' without having to retrain the model.

摘要: 深度模型非常容易受到对抗性攻击。这类攻击是精心设计的难以察觉的噪音，可以愚弄网络，在部署时可能会造成严重后果。为了遇到它们，该模型需要用于对抗性训练的训练数据或明确的基于正则化的技术。然而，隐私已经成为一个重要的问题，只限制对训练模型的访问，而不限制对训练数据(例如生物特征数据)的访问。此外，数据管理成本高昂，公司可能对其拥有专有权。为了处理这种情况，我们提出了一个全新的问题，即在没有训练数据甚至其统计数据的情况下进行测试时间对抗性防御。我们分两个阶段来解决这个问题：a)对手样本的检测和b)对手样本的校正。我们的对抗性样本检测框架首先在任意数据上进行训练，然后通过无监督的领域自适应来适应未标记的测试数据。通过对检测到的敌意样本进行傅立叶变换，并在我们提出的适合模型预测的半径处获得它们的低频分量，进一步修正了预测。我们通过针对几种对抗性攻击以及针对不同模型架构和数据集的广泛实验，证明了我们所提出的技术的有效性。对于在CIFAR-10上预先训练的非健壮RESNET-18模型，我们的检测方法正确识别了91.42%的对手。此外，在不需要重新训练模型的情况下，我们显著地将对手准确率从0%提高到37.37%，而对最先进的自动攻击的干净准确率最小下降了0.02%。



## **23. RobustSense: Defending Adversarial Attack for Secure Device-Free Human Activity Recognition**

RobustSense：防御恶意攻击，实现安全的无设备人类活动识别 cs.CR

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2204.01560v1)

**Authors**: Jianfei Yang, Han Zou, Lihua Xie

**Abstracts**: Deep neural networks have empowered accurate device-free human activity recognition, which has wide applications. Deep models can extract robust features from various sensors and generalize well even in challenging situations such as data-insufficient cases. However, these systems could be vulnerable to input perturbations, i.e. adversarial attacks. We empirically demonstrate that both black-box Gaussian attacks and modern adversarial white-box attacks can render their accuracies to plummet. In this paper, we firstly point out that such phenomenon can bring severe safety hazards to device-free sensing systems, and then propose a novel learning framework, RobustSense, to defend common attacks. RobustSense aims to achieve consistent predictions regardless of whether there exists an attack on its input or not, alleviating the negative effect of distribution perturbation caused by adversarial attacks. Extensive experiments demonstrate that our proposed method can significantly enhance the model robustness of existing deep models, overcoming possible attacks. The results validate that our method works well on wireless human activity recognition and person identification systems. To the best of our knowledge, this is the first work to investigate adversarial attacks and further develop a novel defense framework for wireless human activity recognition in mobile computing research.

摘要: 深度神经网络能够实现准确的无设备人体活动识别，具有广泛的应用前景。深度模型可以从各种传感器中提取稳健的特征，即使在数据不足等具有挑战性的情况下也能很好地推广。然而，这些系统可能容易受到输入扰动，即对抗性攻击。我们的经验证明，无论是黑盒高斯攻击还是现代对抗性白盒攻击，它们的准确率都会直线下降。在本文中，我们首先指出这种现象会给无设备感知系统带来严重的安全隐患，然后提出一种新的学习框架RobustSense来防御常见的攻击。RobustSense的目标是在输入是否存在攻击的情况下实现一致的预测，缓解因对抗性攻击而导致的分布扰动的负面影响。大量实验表明，该方法能够显著增强已有深度模型的模型稳健性，克服可能的攻击。实验结果表明，该方法在无线人体活动识别和身份识别系统中具有较好的效果。据我们所知，这是第一次在移动计算研究中研究对抗性攻击，并进一步开发出一种新的无线人类活动识别防御框架。



## **24. PRADA: Practical Black-Box Adversarial Attacks against Neural Ranking Models**

Prada：针对神经排序模型的实用黑箱对抗性攻击 cs.IR

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2204.01321v1)

**Authors**: Chen Wu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstracts**: Neural ranking models (NRMs) have shown remarkable success in recent years, especially with pre-trained language models. However, deep neural models are notorious for their vulnerability to adversarial examples. Adversarial attacks may become a new type of web spamming technique given our increased reliance on neural information retrieval models. Therefore, it is important to study potential adversarial attacks to identify vulnerabilities of NRMs before they are deployed.   In this paper, we introduce the Adversarial Document Ranking Attack (ADRA) task against NRMs, which aims to promote a target document in rankings by adding adversarial perturbations to its text. We focus on the decision-based black-box attack setting, where the attackers have no access to the model parameters and gradients, but can only acquire the rank positions of the partial retrieved list by querying the target model. This attack setting is realistic in real-world search engines. We propose a novel Pseudo Relevance-based ADversarial ranking Attack method (PRADA) that learns a surrogate model based on Pseudo Relevance Feedback (PRF) to generate gradients for finding the adversarial perturbations.   Experiments on two web search benchmark datasets show that PRADA can outperform existing attack strategies and successfully fool the NRM with small indiscernible perturbations of text.

摘要: 近年来，神经网络排序模型(NRM)取得了显著的成功，尤其是使用了预先训练好的语言模型。然而，深层神经模型因其易受敌意例子的攻击而臭名昭著。鉴于我们对神经信息检索模型的日益依赖，对抗性攻击可能成为一种新型的Web垃圾邮件技术。因此，在部署NRM之前，研究潜在的敌意攻击以识别NRM的漏洞是很重要的。在本文中，我们引入了针对NRMS的对抗性文档排名攻击(ADRA)任务，其目的是通过在文本中添加对抗性扰动来提升目标文档的排名。重点研究了基于决策的黑盒攻击环境，其中攻击者无法获取模型参数和梯度，只能通过查询目标模型获得部分检索列表的排名位置。这种攻击设置在现实世界的搜索引擎中是现实的。提出了一种新的基于伪相关性的对抗性排序攻击方法(PRADA)，该方法通过学习基于伪相关反馈(PRF)的代理模型来生成用于发现对抗性扰动的梯度。在两个网络搜索基准数据集上的实验表明，Prada可以超越现有的攻击策略，并成功地利用文本的微小不可分辨扰动来欺骗NRM。



## **25. Captcha Attack: Turning Captchas Against Humanity**

Captcha攻击：使Captchas反人类 cs.CR

Currently under submission

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2201.04014v3)

**Authors**: Mauro Conti, Luca Pajola, Pier Paolo Tricomi

**Abstracts**: Nowadays, people generate and share massive content on online platforms (e.g., social networks, blogs). In 2021, the 1.9 billion daily active Facebook users posted around 150 thousand photos every minute. Content moderators constantly monitor these online platforms to prevent the spreading of inappropriate content (e.g., hate speech, nudity images). Based on deep learning (DL) advances, Automatic Content Moderators (ACM) help human moderators handle high data volume. Despite their advantages, attackers can exploit weaknesses of DL components (e.g., preprocessing, model) to affect their performance. Therefore, an attacker can leverage such techniques to spread inappropriate content by evading ACM.   In this work, we propose CAPtcha Attack (CAPA), an adversarial technique that allows users to spread inappropriate text online by evading ACM controls. CAPA, by generating custom textual CAPTCHAs, exploits ACM's careless design implementations and internal procedures vulnerabilities. We test our attack on real-world ACM, and the results confirm the ferocity of our simple yet effective attack, reaching up to a 100% evasion success in most cases. At the same time, we demonstrate the difficulties in designing CAPA mitigations, opening new challenges in CAPTCHAs research area.

摘要: 如今，人们在在线平台(如社交网络、博客)上生成和分享大量内容。2021年，Facebook的19亿日活跃用户每分钟发布约15万张照片。内容版主不断监控这些在线平台，以防止传播不恰当的内容(例如，仇恨言论、裸体图片)。基于深度学习(DL)的进步，自动内容审核者(ACM)帮助人工审核者处理大量数据。尽管具有优势，攻击者仍可以利用DL组件的弱点(例如，预处理、模型)来影响其性能。因此，攻击者可以利用这些技术通过规避ACM来传播不适当的内容。在这项工作中，我们提出了验证码攻击(CAPA)，这是一种敌意技术，允许用户通过逃避ACM控制来在线传播不适当的文本。通过生成自定义文本验证码，CAPA可以利用ACM粗心的设计实现和内部过程漏洞。我们在真实的ACM上测试了我们的攻击，结果证实了我们简单而有效的攻击的凶猛，在大多数情况下达到了100%的规避成功。同时，我们展示了设计CAPA缓解措施的困难，为CAPTCHAS研究领域开辟了新的挑战。



## **26. Detecting In-vehicle Intrusion via Semi-supervised Learning-based Convolutional Adversarial Autoencoders**

基于半监督学习的卷积对抗性自动编码器车载入侵检测 cs.CR

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2204.01193v1)

**Authors**: Thien-Nu Hoang, Daehee Kim

**Abstracts**: With the development of autonomous vehicle technology, the controller area network (CAN) bus has become the de facto standard for an in-vehicle communication system because of its simplicity and efficiency. However, without any encryption and authentication mechanisms, the in-vehicle network using the CAN protocol is susceptible to a wide range of attacks. Many studies, which are mostly based on machine learning, have proposed installing an intrusion detection system (IDS) for anomaly detection in the CAN bus system. Although machine learning methods have many advantages for IDS, previous models usually require a large amount of labeled data, which results in high time and labor costs. To handle this problem, we propose a novel semi-supervised learning-based convolutional adversarial autoencoder model in this paper. The proposed model combines two popular deep learning models: autoencoder and generative adversarial networks. First, the model is trained with unlabeled data to learn the manifolds of normal and attack patterns. Then, only a small number of labeled samples are used in supervised training. The proposed model can detect various kinds of message injection attacks, such as DoS, fuzzy, and spoofing, as well as unknown attacks. The experimental results show that the proposed model achieves the highest F1 score of 0.99 and a low error rate of 0.1\% with limited labeled data compared to other supervised methods. In addition, we show that the model can meet the real-time requirement by analyzing the model complexity in terms of the number of trainable parameters and inference time. This study successfully reduced the number of model parameters by five times and the inference time by eight times, compared to a state-of-the-art model.

摘要: 随着自动驾驶汽车技术的发展，控制器局域网(CAN)总线以其简单高效的特点已成为车载通信系统的事实标准。然而，在没有任何加密和认证机制的情况下，使用CAN协议的车载网络容易受到广泛的攻击。许多研究大多基于机器学习，提出在CAN总线系统中安装入侵检测系统(入侵检测系统)进行异常检测。虽然机器学习方法在入侵检测中有很多优点，但是以前的模型通常需要大量的标记数据，这导致了很高的时间和人力成本。针对这一问题，本文提出了一种新的基于半监督学习的卷积对抗性自动编码器模型。该模型结合了两种流行的深度学习模型：自动编码器和产生式对抗网络。首先，用未标记的数据训练模型，学习正常模式和攻击模式的流形。然后，只使用少量的标记样本进行有监督的训练。该模型可以检测各种类型的消息注入攻击，如DoS、模糊攻击、欺骗攻击以及未知攻击。实验结果表明，与其他监督方法相比，该模型在标签数据有限的情况下获得了最高的F1值0.99和较低的错误率0.1。此外，从可训练参数个数和推理时间两个方面分析了模型的复杂性，结果表明该模型能够满足实时性的要求。与最先进的模型相比，本研究成功地将模型参数的数量减少了5倍，推理时间减少了8倍。



## **27. DST: Dynamic Substitute Training for Data-free Black-box Attack**

DST：无数据黑盒攻击的动态替补训练 cs.CV

Accepted by CVPR2022

**SubmitDate**: 2022-04-03    [paper-pdf](http://arxiv.org/pdf/2204.00972v1)

**Authors**: Wenxuan Wang, Xuelin Qian, Yanwei Fu, Xiangyang Xue

**Abstracts**: With the wide applications of deep neural network models in various computer vision tasks, more and more works study the model vulnerability to adversarial examples. For data-free black box attack scenario, existing methods are inspired by the knowledge distillation, and thus usually train a substitute model to learn knowledge from the target model using generated data as input. However, the substitute model always has a static network structure, which limits the attack ability for various target models and tasks. In this paper, we propose a novel dynamic substitute training attack method to encourage substitute model to learn better and faster from the target model. Specifically, a dynamic substitute structure learning strategy is proposed to adaptively generate optimal substitute model structure via a dynamic gate according to different target models and tasks. Moreover, we introduce a task-driven graph-based structure information learning constrain to improve the quality of generated training data, and facilitate the substitute model learning structural relationships from the target model multiple outputs. Extensive experiments have been conducted to verify the efficacy of the proposed attack method, which can achieve better performance compared with the state-of-the-art competitors on several datasets.

摘要: 随着深度神经网络模型在各种计算机视觉任务中的广泛应用，越来越多的工作研究了模型对对抗性例子的脆弱性。对于无数据黑盒攻击场景，现有的方法都是受到知识提炼的启发，因此通常训练一个替代模型，以生成的数据作为输入从目标模型学习知识。然而，替代模型总是具有静态的网络结构，这限制了对各种目标模型和任务的攻击能力。本文提出了一种新的动态替补训练攻击方法，以鼓励替补模型更好、更快地向目标模型学习。具体地，提出了一种动态替换结构学习策略，根据不同的目标模型和任务，通过动态门自适应地生成最优替换模型结构。此外，我们还引入了一种任务驱动的基于图的结构信息学习约束，以提高生成的训练数据的质量，并便于替换模型从目标模型的多个输出中学习结构关系。大量的实验验证了所提出的攻击方法的有效性，该方法在多个数据集上取得了比最先进的竞争对手更好的性能。



## **28. Adversarial Neon Beam: Robust Physical-World Adversarial Attack to DNNs**

对抗性霓虹灯：对DNN的强大物理世界对抗性攻击 cs.CV

**SubmitDate**: 2022-04-02    [paper-pdf](http://arxiv.org/pdf/2204.00853v1)

**Authors**: Chengyin Hu, Kalibinuer Tiliwalidi

**Abstracts**: In the physical world, light affects the performance of deep neural networks. Nowadays, many products based on deep neural network have been put into daily life. There are few researches on the effect of light on the performance of deep neural network models. However, the adversarial perturbations generated by light may have extremely dangerous effects on these systems. In this work, we propose an attack method called adversarial neon beam (AdvNB), which can execute the physical attack by obtaining the physical parameters of adversarial neon beams with very few queries. Experiments show that our algorithm can achieve advanced attack effect in both digital test and physical test. In the digital environment, 99.3% attack success rate was achieved, and in the physical environment, 100% attack success rate was achieved. Compared with the most advanced physical attack methods, our method can achieve better physical perturbation concealment. In addition, by analyzing the experimental data, we reveal some new phenomena brought about by the adversarial neon beam attack.

摘要: 在物理世界中，光线会影响深度神经网络的性能。如今，许多基于深度神经网络的产品已经进入日常生活。关于光照对深度神经网络模型性能影响的研究很少。然而，光产生的对抗性扰动可能会对这些系统产生极其危险的影响。在这项工作中，我们提出了一种称为对抗性霓虹束的攻击方法(AdvNB)，该方法只需很少的查询就可以获得对抗性霓虹束的物理参数来执行物理攻击。实验表明，该算法在数字测试和物理测试中均能达到较好的攻击效果。在数字环境下，攻击成功率达到99.3%，在物理环境下，攻击成功率达到100%。与最先进的物理攻击方法相比，我们的方法可以实现更好的物理扰动隐藏。此外，通过对实验数据的分析，揭示了对抗性霓虹束攻击带来的一些新现象。



## **29. Precise Statistical Analysis of Classification Accuracies for Adversarial Training**

对抗性训练中分类精度的精确统计分析 stat.ML

80 pages; to appear in the Annals of Statistics

**SubmitDate**: 2022-04-02    [paper-pdf](http://arxiv.org/pdf/2010.11213v2)

**Authors**: Adel Javanmard, Mahdi Soltanolkotabi

**Abstracts**: Despite the wide empirical success of modern machine learning algorithms and models in a multitude of applications, they are known to be highly susceptible to seemingly small indiscernible perturbations to the input data known as \emph{adversarial attacks}. A variety of recent adversarial training procedures have been proposed to remedy this issue. Despite the success of such procedures at increasing accuracy on adversarially perturbed inputs or \emph{robust accuracy}, these techniques often reduce accuracy on natural unperturbed inputs or \emph{standard accuracy}. Complicating matters further, the effect and trend of adversarial training procedures on standard and robust accuracy is rather counter intuitive and radically dependent on a variety of factors including the perceived form of the perturbation during training, size/quality of data, model overparameterization, etc. In this paper we focus on binary classification problems where the data is generated according to the mixture of two Gaussians with general anisotropic covariance matrices and derive a precise characterization of the standard and robust accuracy for a class of minimax adversarially trained models. We consider a general norm-based adversarial model, where the adversary can add perturbations of bounded $\ell_p$ norm to each input data, for an arbitrary $p\ge 1$. Our comprehensive analysis allows us to theoretically explain several intriguing empirical phenomena and provide a precise understanding of the role of different problem parameters on standard and robust accuracies.

摘要: 尽管现代机器学习算法和模型在许多应用中取得了广泛的经验上的成功，但众所周知，它们对输入数据的看起来很小、难以辨别的扰动非常敏感，称为\emph(对抗性攻击)。最近提出了各种对抗性训练程序来解决这个问题。尽管这些程序成功地提高了反向扰动输入的准确性或\emph{稳健精度}，但这些技术往往会降低自然扰动输入的准确性或\emph{标准精度}。更复杂的是，对抗性训练过程对标准和稳健精度的影响和趋势是相当违反直觉的，并且从根本上依赖于各种因素，包括训练过程中所感知的扰动形式、数据的大小/质量、模型的过度参数化等。在本文中，我们关注根据两个高斯和一般各向异性协方差矩阵的混合来生成数据的二进制分类问题，并推导了一类极小极大对抗性训练模型的标准和稳健精度的精确表征。我们考虑了一个一般的基于范数的对抗性模型，其中对手可以对每个输入数据添加有界的$\p$范数的扰动，对于任意的$p\ge 1$。我们的全面分析使我们能够从理论上解释几个有趣的经验现象，并提供对不同问题参数对标准和稳健精度的作用的精确理解。



## **30. SkeleVision: Towards Adversarial Resiliency of Person Tracking with Multi-Task Learning**

SkeleVision：基于多任务学习的人跟踪的对抗性 cs.CV

**SubmitDate**: 2022-04-02    [paper-pdf](http://arxiv.org/pdf/2204.00734v1)

**Authors**: Nilaksh Das, Sheng-Yun Peng, Duen Horng Chau

**Abstracts**: Person tracking using computer vision techniques has wide ranging applications such as autonomous driving, home security and sports analytics. However, the growing threat of adversarial attacks raises serious concerns regarding the security and reliability of such techniques. In this work, we study the impact of multi-task learning (MTL) on the adversarial robustness of the widely used SiamRPN tracker, in the context of person tracking. Specifically, we investigate the effect of jointly learning with semantically analogous tasks of person tracking and human keypoint detection. We conduct extensive experiments with more powerful adversarial attacks that can be physically realizable, demonstrating the practical value of our approach. Our empirical study with simulated as well as real-world datasets reveals that training with MTL consistently makes it harder to attack the SiamRPN tracker, compared to typically training only on the single task of person tracking.

摘要: 使用计算机视觉技术的人物跟踪具有广泛的应用，如自动驾驶、家庭安全和体育分析。然而，对抗性攻击的威胁越来越大，这引起了人们对这种技术的安全性和可靠性的严重关切。在这项工作中，我们研究了多任务学习(MTL)对广泛使用的SiamRPN跟踪器的对抗健壮性的影响，在个人跟踪的背景下。具体地说，我们考察了联合学习与语义相似的人物跟踪和人体关键点检测任务的效果。我们进行了更强大的对抗性攻击的广泛实验，这些攻击可以在物理上实现，证明了我们方法的实用价值。我们用模拟和真实世界的数据集进行的经验研究表明，与通常只进行单一人跟踪任务的训练相比，使用MTL进行训练始终使攻击SiamRPN追踪器变得更加困难。



## **31. FrequencyLowCut Pooling -- Plug & Play against Catastrophic Overfitting**

FredencyLowCut池--针对灾难性过拟合的即插即用 cs.CV

**SubmitDate**: 2022-04-01    [paper-pdf](http://arxiv.org/pdf/2204.00491v1)

**Authors**: Julia Grabinski, Steffen Jung, Janis Keuper, Margret Keuper

**Abstracts**: Over the last years, Convolutional Neural Networks (CNNs) have been the dominating neural architecture in a wide range of computer vision tasks. From an image and signal processing point of view, this success might be a bit surprising as the inherent spatial pyramid design of most CNNs is apparently violating basic signal processing laws, i.e. Sampling Theorem in their down-sampling operations. However, since poor sampling appeared not to affect model accuracy, this issue has been broadly neglected until model robustness started to receive more attention. Recent work [17] in the context of adversarial attacks and distribution shifts, showed after all, that there is a strong correlation between the vulnerability of CNNs and aliasing artifacts induced by poor down-sampling operations. This paper builds on these findings and introduces an aliasing free down-sampling operation which can easily be plugged into any CNN architecture: FrequencyLowCut pooling. Our experiments show, that in combination with simple and fast FGSM adversarial training, our hyper-parameter free operator significantly improves model robustness and avoids catastrophic overfitting.

摘要: 在过去的几年里，卷积神经网络(CNN)已经成为在广泛的计算机视觉任务中占主导地位的神经结构。从图像和信号处理的角度来看，这一成功可能有点令人惊讶，因为大多数CNN固有的空间金字塔设计显然违反了基本的信号处理定律，即下采样操作中的采样定理。然而，由于较差的采样似乎不会影响模型的精度，所以这个问题一直被广泛忽视，直到模型的稳健性开始受到更多的关注。最近的工作[17]在对抗性攻击和分布转移的背景下，毕竟表明在CNN的脆弱性和糟糕的下采样操作引起的混叠伪像之间存在很强的相关性。本文以这些发现为基础，介绍了一种无混叠的下采样操作，该操作可以很容易地插入到任何CNN架构中：FrequencyLowCut池。我们的实验表明，结合简单快速的FGSM对抗性训练，我们的超参数自由算子显著地提高了模型的稳健性，并避免了灾难性的过拟合。



## **32. Sensor Data Validation and Driving Safety in Autonomous Driving Systems**

自动驾驶系统中的传感器数据验证与驾驶安全 cs.CV

PhD Thesis, City University of Hong Kong

**SubmitDate**: 2022-04-01    [paper-pdf](http://arxiv.org/pdf/2203.16130v2)

**Authors**: Jindi Zhang

**Abstracts**: Autonomous driving technology has drawn a lot of attention due to its fast development and extremely high commercial values. The recent technological leap of autonomous driving can be primarily attributed to the progress in the environment perception. Good environment perception provides accurate high-level environment information which is essential for autonomous vehicles to make safe and precise driving decisions and strategies. Moreover, such progress in accurate environment perception would not be possible without deep learning models and advanced onboard sensors, such as optical sensors (LiDARs and cameras), radars, GPS. However, the advanced sensors and deep learning models are prone to recently invented attack methods. For example, LiDARs and cameras can be compromised by optical attacks, and deep learning models can be attacked by adversarial examples. The attacks on advanced sensors and deep learning models can largely impact the accuracy of the environment perception, posing great threats to the safety and security of autonomous vehicles. In this thesis, we study the detection methods against the attacks on onboard sensors and the linkage between attacked deep learning models and driving safety for autonomous vehicles. To detect the attacks, redundant data sources can be exploited, since information distortions caused by attacks in victim sensor data result in inconsistency with the information from other redundant sources. To study the linkage between attacked deep learning models and driving safety...

摘要: 自动驾驶技术因其快速发展和极高的商业价值而备受关注。最近自动驾驶的技术飞跃主要归功于环境感知的进步。良好的环境感知提供了准确的高层环境信息，这对自动驾驶车辆做出安全、准确的驾驶决策和策略至关重要。此外，如果没有深度学习模型和先进的车载传感器，如光学传感器(激光雷达和照相机)、雷达、全球定位系统，准确的环境感知方面的进展是不可能的。然而，先进的传感器和深度学习模型很容易受到最近发明的攻击方法的影响。例如，激光雷达和摄像头可能会受到光学攻击，深度学习模型可能会受到对抗性例子的攻击。对先进传感器和深度学习模型的攻击会在很大程度上影响环境感知的准确性，对自动驾驶车辆的安全构成极大威胁。在本文中，我们研究了针对车载传感器攻击的检测方法，以及被攻击的深度学习模型与自主车辆驾驶安全之间的联系。为了检测攻击，可以利用冗余数据源，因为攻击导致受害者传感器数据中的信息失真导致与来自其他冗余源的信息不一致。为了研究被攻击的深度学习模型和驾驶安全之间的联系...



## **33. Multi-Expert Adversarial Attack Detection in Person Re-identification Using Context Inconsistency**

基于上下文不一致的人重识别中的多专家对抗攻击检测 cs.CV

Accepted at IEEE ICCV 2021

**SubmitDate**: 2022-04-01    [paper-pdf](http://arxiv.org/pdf/2108.09891v2)

**Authors**: Xueping Wang, Shasha Li, Min Liu, Yaonan Wang, Amit K. Roy-Chowdhury

**Abstracts**: The success of deep neural networks (DNNs) has promoted the widespread applications of person re-identification (ReID). However, ReID systems inherit the vulnerability of DNNs to malicious attacks of visually inconspicuous adversarial perturbations. Detection of adversarial attacks is, therefore, a fundamental requirement for robust ReID systems. In this work, we propose a Multi-Expert Adversarial Attack Detection (MEAAD) approach to achieve this goal by checking context inconsistency, which is suitable for any DNN-based ReID systems. Specifically, three kinds of context inconsistencies caused by adversarial attacks are employed to learn a detector for distinguishing the perturbed examples, i.e., a) the embedding distances between a perturbed query person image and its top-K retrievals are generally larger than those between a benign query image and its top-K retrievals, b) the embedding distances among the top-K retrievals of a perturbed query image are larger than those of a benign query image, c) the top-K retrievals of a benign query image obtained with multiple expert ReID models tend to be consistent, which is not preserved when attacks are present. Extensive experiments on the Market1501 and DukeMTMC-ReID datasets show that, as the first adversarial attack detection approach for ReID, MEAAD effectively detects various adversarial attacks and achieves high ROC-AUC (over 97.5%).

摘要: 深度神经网络(DNN)的成功促进了人的再识别(ReID)的广泛应用。然而，REID系统继承了DNN对视觉上不明显的对抗性扰动的恶意攻击的脆弱性。因此，对敌意攻击的检测是健壮的Reid系统的基本要求。在这项工作中，我们提出了一种多专家对抗攻击检测(MEAAD)方法，通过检查上下文不一致性来实现这一目标，该方法适用于任何基于DNN的REID系统。具体地说，利用对抗性攻击引起的三种上下文不一致来学习用于区分扰动示例的检测器，即a)扰动查询人图像与其top-K检索之间的嵌入距离通常大于良性查询图像与其top-K检索之间的嵌入距离，b)扰动查询图像的top-K检索之间的嵌入距离大于良性查询图像的嵌入距离，c)用多个专家Reid模型获得的良性查询图像的top-K检索趋于一致，当存在攻击时不被保存。在Market1501和DukeMTMC-Reid数据集上的大量实验表明，MEAAD作为REID的第一种对抗性攻击检测方法，有效地检测了各种对抗性攻击，并获得了高ROC-AUC(97.5%以上)。



## **34. Effect of Balancing Data Using Synthetic Data on the Performance of Machine Learning Classifiers for Intrusion Detection in Computer Networks**

计算机网络入侵检测中数据均衡对机器学习分类器性能的影响 cs.LG

**SubmitDate**: 2022-04-01    [paper-pdf](http://arxiv.org/pdf/2204.00144v1)

**Authors**: Ayesha S. Dina, A. B. Siddique, D. Manivannan

**Abstracts**: Attacks on computer networks have increased significantly in recent days, due in part to the availability of sophisticated tools for launching such attacks as well as thriving underground cyber-crime economy to support it. Over the past several years, researchers in academia and industry used machine learning (ML) techniques to design and implement Intrusion Detection Systems (IDSes) for computer networks. Many of these researchers used datasets collected by various organizations to train ML models for predicting intrusions. In many of the datasets used in such systems, data are imbalanced (i.e., not all classes have equal amount of samples). With unbalanced data, the predictive models developed using ML algorithms may produce unsatisfactory classifiers which would affect accuracy in predicting intrusions. Traditionally, researchers used over-sampling and under-sampling for balancing data in datasets to overcome this problem. In this work, in addition to over-sampling, we also use a synthetic data generation method, called Conditional Generative Adversarial Network (CTGAN), to balance data and study their effect on various ML classifiers. To the best of our knowledge, no one else has used CTGAN to generate synthetic samples to balance intrusion detection datasets. Based on extensive experiments using a widely used dataset NSL-KDD, we found that training ML models on dataset balanced with synthetic samples generated by CTGAN increased prediction accuracy by up to $8\%$, compared to training the same ML models over unbalanced data. Our experiments also show that the accuracy of some ML models trained over data balanced with random over-sampling decline compared to the same ML models trained over unbalanced data.

摘要: 最近几天，针对计算机网络的攻击显著增加，部分原因是可以使用复杂的工具来发动此类攻击，以及蓬勃发展的地下网络犯罪经济为其提供支持。在过去的几年里，学术界和工业界的研究人员使用机器学习(ML)技术来设计和实现计算机网络的入侵检测系统(IDSS)。这些研究人员中的许多人使用不同组织收集的数据集来训练ML模型以预测入侵。在这种系统中使用的许多数据集中，数据是不平衡的(即，不是所有类别都具有相同数量的样本量)。在数据不平衡的情况下，使用ML算法开发的预测模型可能会产生不令人满意的分类器，这将影响预测入侵的准确性。传统上，研究人员使用过采样和欠采样来平衡数据集中的数据，以克服这一问题。在这项工作中，除了过采样，我们还使用了一种称为条件生成对抗网络(CTGAN)的合成数据生成方法来平衡数据并研究它们对各种ML分类器的影响。据我们所知，还没有人使用CTGAN来生成合成样本来平衡入侵检测数据集。基于广泛使用的数据集NSL-KDD的大量实验，我们发现，与在不平衡数据上训练相同的ML模型相比，在CTGAN生成的合成样本平衡的数据集上训练ML模型可以将预测精度提高高达8美元。我们的实验还表明，与在非平衡数据上训练的相同ML模型相比，在均衡数据上训练的某些ML模型的准确率有所下降。



## **35. Reverse Engineering of Imperceptible Adversarial Image Perturbations**

不可感知的对抗性图像扰动的逆向工程 cs.CV

**SubmitDate**: 2022-04-01    [paper-pdf](http://arxiv.org/pdf/2203.14145v2)

**Authors**: Yifan Gong, Yuguang Yao, Yize Li, Yimeng Zhang, Xiaoming Liu, Xue Lin, Sijia Liu

**Abstracts**: It has been well recognized that neural network based image classifiers are easily fooled by images with tiny perturbations crafted by an adversary. There has been a vast volume of research to generate and defend such adversarial attacks. However, the following problem is left unexplored: How to reverse-engineer adversarial perturbations from an adversarial image? This leads to a new adversarial learning paradigm--Reverse Engineering of Deceptions (RED). If successful, RED allows us to estimate adversarial perturbations and recover the original images. However, carefully crafted, tiny adversarial perturbations are difficult to recover by optimizing a unilateral RED objective. For example, the pure image denoising method may overfit to minimizing the reconstruction error but hardly preserve the classification properties of the true adversarial perturbations. To tackle this challenge, we formalize the RED problem and identify a set of principles crucial to the RED approach design. Particularly, we find that prediction alignment and proper data augmentation (in terms of spatial transformations) are two criteria to achieve a generalizable RED approach. By integrating these RED principles with image denoising, we propose a new Class-Discriminative Denoising based RED framework, termed CDD-RED. Extensive experiments demonstrate the effectiveness of CDD-RED under different evaluation metrics (ranging from the pixel-level, prediction-level to the attribution-level alignment) and a variety of attack generation methods (e.g., FGSM, PGD, CW, AutoAttack, and adaptive attacks).

摘要: 众所周知，基于神经网络的图像分类器很容易被对手制作的带有微小扰动的图像所愚弄。已经有大量的研究来产生和防御这种对抗性攻击。然而，以下问题仍未得到探索：如何从对抗性图像中逆向设计对抗性扰动？这导致了一种新的对抗性学习范式--欺骗的逆向工程(RED)。如果成功，RED允许我们估计敌方干扰并恢复原始图像。然而，精心设计的微小对抗性干扰很难通过优化单边红色目标来恢复。例如，纯图像去噪方法可能过于适合最小化重建误差，但很难保持真实对抗性扰动的分类性质。为了应对这一挑战，我们将RED问题形式化，并确定一组对RED方法设计至关重要的原则。特别是，我们发现预测对齐和适当的数据增强(在空间变换方面)是实现可推广的RED方法的两个标准。通过将这些RED原理与图像去噪相结合，我们提出了一种新的基于类别区分的RED去噪框架，称为CDD-RED。大量的实验证明了CDD-RED在不同的评估指标(从像素级、预测级到属性级对齐)和各种攻击生成方法(如FGSM、PGD、CW、AutoAttack和自适应攻击)下的有效性。



## **36. Scalable Whitebox Attacks on Tree-based Models**

基于树模型的可伸缩白盒攻击 stat.ML

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2204.00103v1)

**Authors**: Giuseppe Castiglione, Gavin Ding, Masoud Hashemi, Christopher Srinivasa, Ga Wu

**Abstracts**: Adversarial robustness is one of the essential safety criteria for guaranteeing the reliability of machine learning models. While various adversarial robustness testing approaches were introduced in the last decade, we note that most of them are incompatible with non-differentiable models such as tree ensembles. Since tree ensembles are widely used in industry, this reveals a crucial gap between adversarial robustness research and practical applications. This paper proposes a novel whitebox adversarial robustness testing approach for tree ensemble models. Concretely, the proposed approach smooths the tree ensembles through temperature controlled sigmoid functions, which enables gradient descent-based adversarial attacks. By leveraging sampling and the log-derivative trick, the proposed approach can scale up to testing tasks that were previously unmanageable. We compare the approach against both random perturbations and blackbox approaches on multiple public datasets (and corresponding models). Our results show that the proposed method can 1) successfully reveal the adversarial vulnerability of tree ensemble models without causing computational pressure for testing and 2) flexibly balance the search performance and time complexity to meet various testing criteria.

摘要: 对抗稳健性是保证机器学习模型可靠性的基本安全准则之一。虽然在过去的十年中引入了各种对抗性健壮性测试方法，但我们注意到其中大多数方法与不可微模型(如树集成)不兼容。由于树形集成在工业中的广泛应用，这揭示了对抗性稳健性研究与实际应用之间的关键差距。提出了一种新的树集成模型白盒对抗健壮性测试方法。具体地说，该方法通过温度控制的Sigmoid函数来平滑树集合，从而实现基于梯度下降的对抗性攻击。通过利用采样和对数导数技巧，建议的方法可以扩展到测试以前无法管理的任务。我们在多个公共数据集(以及相应的模型)上将该方法与随机扰动和黑盒方法进行了比较。结果表明，该方法能够在不增加测试计算压力的情况下，1)成功地揭示树集成模型的对抗性漏洞，2)灵活地平衡搜索性能和时间复杂度，以满足不同的测试标准。



## **37. Parallel Proof-of-Work with Concrete Bounds**

具有具体界限的并行工作证明 cs.CR

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2204.00034v1)

**Authors**: Patrik Keller, Rainer Böhme

**Abstracts**: Authorization is challenging in distributed systems that cannot rely on the identification of nodes. Proof-of-work offers an alternative gate-keeping mechanism, but its probabilistic nature is incompatible with conventional security definitions. Recent related work establishes concrete bounds for the failure probability of Bitcoin's sequential proof-of-work mechanism. We propose a family of state replication protocols using parallel proof-of-work. Our bottom-up design from an agreement sub-protocol allows us to give concrete bounds for the failure probability in adversarial synchronous networks. After the typical interval of 10 minutes, parallel proof-of-work offers two orders of magnitude more security than sequential proof-of-work. This means that state updates can be sufficiently secure to support commits after one block (i.e., after 10 minutes), removing the risk of double-spending in many applications. We offer guidance on the optimal choice of parameters for a wide range of network and attacker assumptions. Simulations show that the proposed construction is robust against violations of design assumptions.

摘要: 在不能依赖节点标识的分布式系统中，授权是具有挑战性的。工作证明提供了一种替代的把关机制，但其概率性质与传统的安全定义不兼容。最近的相关工作为比特币的序贯验证机制的失效概率建立了具体的界限。我们提出了一类使用并行工作证明的状态复制协议。我们从协议子协议开始的自下而上的设计允许我们给出对抗性同步网络中故障概率的具体界。在典型的10分钟间隔之后，并行工作证明提供的安全性比顺序工作证明高两个数量级。这意味着状态更新可以足够安全，以支持在一个数据块(即10分钟之后)后提交，从而消除了许多应用程序中重复支出的风险。我们为各种网络和攻击者假设提供参数最佳选择的指导。仿真结果表明，所提出的结构对违反设计假设具有较强的鲁棒性。



## **38. Truth Serum: Poisoning Machine Learning Models to Reveal Their Secrets**

真相血清：毒化机器学习模型以揭示其秘密 cs.CR

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2204.00032v1)

**Authors**: Florian Tramèr, Reza Shokri, Ayrton San Joaquin, Hoang Le, Matthew Jagielski, Sanghyun Hong, Nicholas Carlini

**Abstracts**: We introduce a new class of attacks on machine learning models. We show that an adversary who can poison a training dataset can cause models trained on this dataset to leak significant private details of training points belonging to other parties. Our active inference attacks connect two independent lines of work targeting the integrity and privacy of machine learning training data.   Our attacks are effective across membership inference, attribute inference, and data extraction. For example, our targeted attacks can poison <0.1% of the training dataset to boost the performance of inference attacks by 1 to 2 orders of magnitude. Further, an adversary who controls a significant fraction of the training data (e.g., 50%) can launch untargeted attacks that enable 8x more precise inference on all other users' otherwise-private data points.   Our results cast doubts on the relevance of cryptographic privacy guarantees in multiparty computation protocols for machine learning, if parties can arbitrarily select their share of training data.

摘要: 我们在机器学习模型上引入了一类新的攻击。我们表明，可以毒化训练数据集的对手可以导致在该数据集上训练的模型泄露属于其他方的训练点的重要私人细节。我们的主动推理攻击将两个独立的工作线连接在一起，目标是机器学习训练数据的完整性和隐私。我们的攻击在成员关系推理、属性推理和数据提取方面都是有效的。例如，我们的有针对性的攻击可以毒化<0.1%的训练数据集，将推理攻击的性能提高1到2个数量级。此外，控制很大一部分训练数据(例如50%)的对手可以发起无目标攻击，从而能够对所有其他用户的其他私有数据点进行8倍的精确推断。我们的结果对用于机器学习的多方计算协议中的密码隐私保证的相关性提出了怀疑，如果各方可以任意选择他们在训练数据中的份额。



## **39. Improving Adversarial Transferability via Neuron Attribution-Based Attacks**

通过基于神经元属性的攻击提高对手的可转换性 cs.LG

CVPR 2022

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2204.00008v1)

**Authors**: Jianping Zhang, Weibin Wu, Jen-tse Huang, Yizhan Huang, Wenxuan Wang, Yuxin Su, Michael R. Lyu

**Abstracts**: Deep neural networks (DNNs) are known to be vulnerable to adversarial examples. It is thus imperative to devise effective attack algorithms to identify the deficiencies of DNNs beforehand in security-sensitive applications. To efficiently tackle the black-box setting where the target model's particulars are unknown, feature-level transfer-based attacks propose to contaminate the intermediate feature outputs of local models, and then directly employ the crafted adversarial samples to attack the target model. Due to the transferability of features, feature-level attacks have shown promise in synthesizing more transferable adversarial samples. However, existing feature-level attacks generally employ inaccurate neuron importance estimations, which deteriorates their transferability. To overcome such pitfalls, in this paper, we propose the Neuron Attribution-based Attack (NAA), which conducts feature-level attacks with more accurate neuron importance estimations. Specifically, we first completely attribute a model's output to each neuron in a middle layer. We then derive an approximation scheme of neuron attribution to tremendously reduce the computation overhead. Finally, we weight neurons based on their attribution results and launch feature-level attacks. Extensive experiments confirm the superiority of our approach to the state-of-the-art benchmarks.

摘要: 深度神经网络(DNN)很容易受到敌意例子的攻击。因此，设计有效的攻击算法来预先识别DNN在安全敏感应用中的缺陷是当务之急。为了有效地处理目标模型细节未知的黑箱环境，基于特征级转移的攻击提出了污染局部模型的中间特征输出，然后直接利用精心制作的敌意样本来攻击目标模型。由于特征的可转移性，特征级攻击在合成更多可转移的对手样本方面显示出了希望。然而，现有的特征级攻击一般采用不准确的神经元重要性估计，这降低了它们的可转移性。为了克服这些缺陷，本文提出了基于神经元属性的攻击(NAA)，它通过更准确的神经元重要性估计来进行特征级别的攻击。具体地说，我们首先将模型的输出完全归因于中间层的每个神经元。然后，我们推导了神经元属性的近似方案，极大地减少了计算开销。最后，我们根据神经元的属性结果对其进行加权，并发起特征级别的攻击。广泛的实验证实了我们的方法对最先进的基准的优越性。



## **40. Towards Robust Rain Removal Against Adversarial Attacks: A Comprehensive Benchmark Analysis and Beyond**

对抗对抗性攻击的稳健除雨：综合基准分析及进一步研究 cs.CV

10 pages, 6 figures, to appear in CVPR 2022

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2203.16931v1)

**Authors**: Yi Yu, Wenhan Yang, Yap-Peng Tan, Alex C. Kot

**Abstracts**: Rain removal aims to remove rain streaks from images/videos and reduce the disruptive effects caused by rain. It not only enhances image/video visibility but also allows many computer vision algorithms to function properly. This paper makes the first attempt to conduct a comprehensive study on the robustness of deep learning-based rain removal methods against adversarial attacks. Our study shows that, when the image/video is highly degraded, rain removal methods are more vulnerable to the adversarial attacks as small distortions/perturbations become less noticeable or detectable. In this paper, we first present a comprehensive empirical evaluation of various methods at different levels of attacks and with various losses/targets to generate the perturbations from the perspective of human perception and machine analysis tasks. A systematic evaluation of key modules in existing methods is performed in terms of their robustness against adversarial attacks. From the insights of our analysis, we construct a more robust deraining method by integrating these effective modules. Finally, we examine various types of adversarial attacks that are specific to deraining problems and their effects on both human and machine vision tasks, including 1) rain region attacks, adding perturbations only in the rain regions to make the perturbations in the attacked rain images less visible; 2) object-sensitive attacks, adding perturbations only in regions near the given objects. Code is available at https://github.com/yuyi-sd/Robust_Rain_Removal.

摘要: 除雨的目的是去除图像/视频中的雨纹，减少降雨造成的干扰。它不仅增强了图像/视频的可见性，还允许许多计算机视觉算法正常工作。本文首次尝试对基于深度学习的降雨方法对敌方攻击的稳健性进行了全面的研究。我们的研究表明，当图像/视频高度退化时，随着微小的失真/扰动变得不那么明显或可检测到，雨滴去除方法更容易受到敌意攻击。在本文中，我们首先从人的感知和机器分析任务的角度，对不同攻击级别和不同损失/目标的各种方法进行了全面的经验评估，以产生扰动。对现有方法中的关键模块进行了系统的评估，评估了它们对对手攻击的健壮性。根据我们的分析，我们通过整合这些有效的模块来构造一个更健壮的去噪方法。最后，我们研究了针对去重问题的各种类型的对抗性攻击及其对人类和机器视觉任务的影响，包括1)雨区域攻击，仅在雨区域添加扰动，以使被攻击的雨图像中的扰动不那么明显；2)对象敏感攻击，仅在给定对象附近的区域添加扰动。代码可在https://github.com/yuyi-sd/Robust_Rain_Removal.上找到



## **41. Assessing the risk of re-identification arising from an attack on anonymised data**

评估匿名数据遭受攻击后重新识别的风险 cs.LG

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2203.16921v1)

**Authors**: Anna Antoniou, Giacomo Dossena, Julia MacMillan, Steven Hamblin, David Clifton, Paula Petrone

**Abstracts**: Objective: The use of routinely-acquired medical data for research purposes requires the protection of patient confidentiality via data anonymisation. The objective of this work is to calculate the risk of re-identification arising from a malicious attack to an anonymised dataset, as described below. Methods: We first present an analytical means of estimating the probability of re-identification of a single patient in a k-anonymised dataset of Electronic Health Record (EHR) data. Second, we generalize this solution to obtain the probability of multiple patients being re-identified. We provide synthetic validation via Monte Carlo simulations to illustrate the accuracy of the estimates obtained. Results: The proposed analytical framework for risk estimation provides re-identification probabilities that are in agreement with those provided by simulation in a number of scenarios. Our work is limited by conservative assumptions which inflate the re-identification probability. Discussion: Our estimates show that the re-identification probability increases with the proportion of the dataset maliciously obtained and that it has an inverse relationship with the equivalence class size. Our recursive approach extends the applicability domain to the general case of a multi-patient re-identification attack in an arbitrary k-anonymisation scheme. Conclusion: We prescribe a systematic way to parametrize the k-anonymisation process based on a pre-determined re-identification probability. We observed that the benefits of a reduced re-identification risk that come with increasing k-size may not be worth the reduction in data granularity when one is considering benchmarking the re-identification probability on the size of the portion of the dataset maliciously obtained by the adversary.

摘要: 目的：将常规获取的医疗数据用于研究目的需要通过数据匿名化保护患者的机密性。这项工作的目标是计算恶意攻击引起的对匿名数据集的重新识别风险，如下所述。方法：我们首先提出了一种分析方法来估计电子健康记录(EHR)数据的k匿名数据集中单个患者重新识别的可能性。其次，我们推广这一解决方案，以获得多个患者被重新识别的概率。我们通过蒙特卡罗模拟提供了综合验证，以说明所获得的估计的准确性。结果：建议的风险评估分析框架提供的重新识别概率与模拟在许多情况下提供的概率一致。我们的工作受到保守假设的限制，这些假设夸大了重新识别的概率。讨论：我们的估计表明，重新识别的概率随着恶意获得的数据集的比例而增加，并且与等价类的大小成反比。我们的递归方法将适用范围扩展到任意k-匿名化方案中的多患者重新识别攻击的一般情况。结论：我们规定了一种系统的方法，基于预先确定的重新识别概率来对k-匿名化过程进行参数化。我们观察到，当考虑根据对手恶意获得的部分数据集的大小来对重新识别概率进行基准测试时，随着k大小的增加而来的重新识别风险降低的好处可能不值得减少数据粒度。



## **42. Attack Impact Evaluation by Exact Convexification through State Space Augmentation**

基于状态空间增强的精确凸化攻击效果评估 eess.SY

8 pages

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2203.16803v1)

**Authors**: Hampei Sasahara, Takashi Tanaka, Henrik Sandberg

**Abstracts**: We address the attack impact evaluation problem for control system security. We formulate the problem as a Markov decision process with a temporally joint chance constraint that forces the adversary to avoid being detected throughout the considered time period. Owing to the joint constraint, the optimal control policy depends not only on the current state but also on the entire history, which leads to the explosion of the search space and makes the problem generally intractable. It is shown that whether an alarm has been triggered or not, in addition to the current state is sufficient for specifying the optimal decision at each time step. Augmentation of the information to the state space induces an equivalent convex optimization problem, which is tractable using standard solvers.

摘要: 我们解决了控制系统安全的攻击影响评估问题。我们将问题描述为一个具有时间联合机会约束的马尔可夫决策过程，迫使对手在所考虑的时间段内避免被发现。由于联合约束，最优控制策略不仅依赖于当前状态，还依赖于整个历史，这导致搜索空间的爆炸性，使问题普遍难以解决。结果表明，除了当前状态外，警报是否已被触发足以确定每个时间步的最优决策。将信息扩充到状态空间将导致等价的凸优化问题，使用标准求解器可以很容易地处理该问题。



## **43. The Block-based Mobile PDE Systems Are Not Secure -- Experimental Attacks**

基于分组的移动PDE系统不安全--实验性攻击 cs.CR

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2203.16349v2)

**Authors**: Niusen Chen, Bo Chen, Weisong Shi

**Abstracts**: Nowadays, mobile devices have been used broadly to store and process sensitive data. To ensure confidentiality of the sensitive data, Full Disk Encryption (FDE) is often integrated in mainstream mobile operating systems like Android and iOS. FDE however cannot defend against coercive attacks in which the adversary can force the device owner to disclose the decryption key. To combat the coercive attacks, Plausibly Deniable Encryption (PDE) is leveraged to plausibly deny the very existence of sensitive data. However, most of the existing PDE systems for mobile devices are deployed at the block layer and suffer from deniability compromises.   Having observed that none of existing works in the literature have experimentally demonstrated the aforementioned compromises, our work bridges this gap by experimentally confirming the deniability compromises of the block-layer mobile PDE systems. We have built a mobile device testbed, which consists of a host computing device and a flash storage device. Additionally, we have deployed both the hidden volume PDE and the steganographic file system at the block layer of the testbed and performed disk forensics to assess potential compromises on the raw NAND flash. Our experimental results confirm it is indeed possible for the adversary to compromise the block-layer PDE systems by accessing the raw NAND flash in practice. We also discuss potential issues when performing such attacks in real world.

摘要: 如今，移动设备已被广泛用于存储和处理敏感数据。为了确保敏感数据的机密性，Android和iOS等主流移动操作系统经常集成全盘加密(FDE)。然而，FDE无法抵御强制攻击，在这种攻击中，对手可以迫使设备所有者披露解密密钥。为了对抗强制攻击，可信可否认加密(PDE)被用来可信地否认敏感数据的存在。然而，大多数现有的移动设备PDE系统都部署在块层，并受到不可否认性妥协的影响。在观察到现有的文献中没有一项工作在实验上证明了上述妥协之后，我们的工作通过实验确认了块层移动PDE系统的否认妥协来弥合了这一差距。我们搭建了一个移动设备试验台，它由主机计算设备和闪存设备组成。此外，我们在测试床的块层部署了隐藏卷PDE和隐写文件系统，并执行了磁盘取证以评估对原始NAND闪存的潜在危害。我们的实验结果证实了攻击者在实践中确实有可能通过访问原始的NAND闪存来危害块层PDE系统。我们还讨论了在现实世界中执行此类攻击时的潜在问题。



## **44. Well-classified Examples are Underestimated in Classification with Deep Neural Networks**

深度神经网络在分类中低估分类好的样本 cs.LG

Accepted by AAAI 2022; 17 pages, 11 figures, 13 tables

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2110.06537v5)

**Authors**: Guangxiang Zhao, Wenkai Yang, Xuancheng Ren, Lei Li, Yunfang Wu, Xu Sun

**Abstracts**: The conventional wisdom behind learning deep classification models is to focus on bad-classified examples and ignore well-classified examples that are far from the decision boundary. For instance, when training with cross-entropy loss, examples with higher likelihoods (i.e., well-classified examples) contribute smaller gradients in back-propagation. However, we theoretically show that this common practice hinders representation learning, energy optimization, and margin growth. To counteract this deficiency, we propose to reward well-classified examples with additive bonuses to revive their contribution to the learning process. This counterexample theoretically addresses these three issues. We empirically support this claim by directly verifying the theoretical results or significant performance improvement with our counterexample on diverse tasks, including image classification, graph classification, and machine translation. Furthermore, this paper shows that we can deal with complex scenarios, such as imbalanced classification, OOD detection, and applications under adversarial attacks because our idea can solve these three issues. Code is available at: https://github.com/lancopku/well-classified-examples-are-underestimated.

摘要: 学习深度分类模型背后的传统智慧是专注于分类不好的例子，而忽略远离决策边界的分类良好的例子。例如，当使用交叉熵损失进行训练时，具有较高似然的示例(即，分类良好的示例)在反向传播中贡献较小的梯度。然而，我们从理论上表明，这种常见的做法阻碍了表示学习、能量优化和利润率增长。为了弥补这一不足，我们建议用额外的奖金奖励分类良好的例子，以恢复他们对学习过程的贡献。这个反例从理论上解决了这三个问题。我们通过直接验证理论结果或通过我们的反例在包括图像分类、图形分类和机器翻译在内的不同任务上的显著性能改进来经验地支持这一论断。此外，本文还表明，我们的思想可以解决这三个问题，因此我们可以处理复杂的场景，如不平衡分类、面向对象的检测和对手攻击下的应用。代码可从以下网址获得：https://github.com/lancopku/well-classified-examples-are-underestimated.



## **45. Example-based Explanations with Adversarial Attacks for Respiratory Sound Analysis**

呼吸音分析中对抗性攻击的实例解释 cs.SD

Submitted to INTERSPEECH 2022

**SubmitDate**: 2022-03-30    [paper-pdf](http://arxiv.org/pdf/2203.16141v1)

**Authors**: Yi Chang, Zhao Ren, Thanh Tam Nguyen, Wolfgang Nejdl, Björn W. Schuller

**Abstracts**: Respiratory sound classification is an important tool for remote screening of respiratory-related diseases such as pneumonia, asthma, and COVID-19. To facilitate the interpretability of classification results, especially ones based on deep learning, many explanation methods have been proposed using prototypes. However, existing explanation techniques often assume that the data is non-biased and the prediction results can be explained by a set of prototypical examples. In this work, we develop a unified example-based explanation method for selecting both representative data (prototypes) and outliers (criticisms). In particular, we propose a novel application of adversarial attacks to generate an explanation spectrum of data instances via an iterative fast gradient sign method. Such unified explanation can avoid over-generalisation and bias by allowing human experts to assess the model mistakes case by case. We performed a wide range of quantitative and qualitative evaluations to show that our approach generates effective and understandable explanation and is robust with many deep learning models

摘要: 呼吸音分类是远程筛查肺炎、哮喘和新冠肺炎等呼吸系统相关疾病的重要工具。为了便于分类结果的可解释性，特别是基于深度学习的分类结果，已经提出了许多使用原型的解释方法。然而，现有的解释技术往往假设数据是无偏的，预测结果可以通过一组典型例子来解释。在这项工作中，我们开发了一个统一的基于实例的解释方法，用于选择代表性数据(原型)和离群值(批评)。特别是，我们提出了一种新的对抗性攻击的应用，通过迭代快速梯度符号方法来生成数据实例的解释谱。这种统一的解释允许人类专家逐一评估模型错误，从而避免过度概括和偏见。我们进行了广泛的定量和定性评估，表明我们的方法产生了有效和可理解的解释，并对许多深度学习模型具有健壮性



## **46. Fooling the primate brain with minimal, targeted image manipulation**

通过最小的、有针对性的图像处理来愚弄灵长类动物的大脑 q-bio.NC

**SubmitDate**: 2022-03-30    [paper-pdf](http://arxiv.org/pdf/2011.05623v3)

**Authors**: Li Yuan, Will Xiao, Giorgia Dellaferrera, Gabriel Kreiman, Francis E. H. Tay, Jiashi Feng, Margaret S. Livingstone

**Abstracts**: Artificial neural networks (ANNs) are considered the current best models of biological vision. ANNs are the best predictors of neural activity in the ventral stream; moreover, recent work has demonstrated that ANN models fitted to neuronal activity can guide the synthesis of images that drive pre-specified response patterns in small neuronal populations. Despite the success in predicting and steering firing activity, these results have not been connected with perceptual or behavioral changes. Here we propose an array of methods for creating minimal, targeted image perturbations that lead to changes in both neuronal activity and perception as reflected in behavior. We generated 'deceptive images' of human faces, monkey faces, and noise patterns so that they are perceived as a different, pre-specified target category, and measured both monkey neuronal responses and human behavior to these images. We found several effective methods for changing primate visual categorization that required much smaller image change compared to untargeted noise. Our work shares the same goal with adversarial attack, namely the manipulation of images with minimal, targeted noise that leads ANN models to misclassify the images. Our results represent a valuable step in quantifying and characterizing the differences in perturbation robustness of biological and artificial vision.

摘要: 人工神经网络(ANN)被认为是目前最好的生物视觉模型。神经网络是腹侧神经流中神经活动的最佳预测因子；此外，最近的工作表明，适合于神经元活动的神经网络模型可以指导图像的合成，这些图像驱动了小神经元群体中预先指定的反应模式。尽管在预测和指导射击活动方面取得了成功，但这些结果并没有与感知或行为变化联系在一起。在这里，我们提出了一系列方法来创建最小的、有针对性的图像扰动，这些扰动导致神经活动和感知的变化，反映在行为上。我们生成了人脸、猴子脸和噪音模式的“欺骗性图像”，以便它们被视为不同的、预先指定的目标类别，并测量了猴子对这些图像的神经元反应和人类行为。我们发现了几种有效的方法来改变灵长类动物的视觉分类，与非目标噪声相比，这些方法需要的图像改变要小得多。我们的工作与对抗性攻击有着相同的目标，即以最小的目标噪声操纵图像，从而导致ANN模型对图像进行错误分类。我们的结果在量化和表征生物视觉和人工视觉在扰动稳健性方面的差异方面迈出了有价值的一步。



## **47. StyleFool: Fooling Video Classification Systems via Style Transfer**

StyleFool：通过样式转换愚弄视频分类系统 cs.CV

18 pages, 7 figures

**SubmitDate**: 2022-03-30    [paper-pdf](http://arxiv.org/pdf/2203.16000v1)

**Authors**: Yuxin Cao, Xi Xiao, Ruoxi Sun, Derui Wang, Minhui Xue, Sheng Wen

**Abstracts**: Video classification systems are vulnerable to adversarial attacks, which can create severe security problems in video verification. Current black-box attacks need a large number of queries to succeed, resulting in high computational overhead in the process of attack. On the other hand, attacks with restricted perturbations are ineffective against defenses such as denoising or adversarial training. In this paper, we focus on unrestricted perturbations and propose StyleFool, a black-box video adversarial attack via style transfer to fool the video classification system. StyleFool first utilizes color theme proximity to select the best style image, which helps avoid unnatural details in the stylized videos. Meanwhile, the target class confidence is additionally considered in targeted attack to influence the output distribution of the classifier by moving the stylized video closer to or even across the decision boundary. A gradient-free method is then employed to further optimize the adversarial perturbation. We carry out extensive experiments to evaluate StyleFool on two standard datasets, UCF-101 and HMDB-51. The experimental results suggest that StyleFool outperforms the state-of-the-art adversarial attacks in terms of both number of queries and robustness against existing defenses. We identify that 50% of the stylized videos in untargeted attack do not need any query since they can already fool the video classification model. Furthermore, we evaluate the indistinguishability through a user study to show that the adversarial samples of StyleFool look imperceptible to human eyes, despite unrestricted perturbations.

摘要: 视频分类系统容易受到敌意攻击，这会给视频验证带来严重的安全问题。当前的黑盒攻击需要大量的查询才能成功，导致攻击过程中的计算开销很高。另一方面，受限扰动的攻击对诸如去噪或对抗性训练等防御措施无效。本文针对无限制扰动，提出了StyleFool，一种通过风格转移来欺骗视频分类系统的黑盒视频对抗性攻击。StyleFool首先利用颜色主题贴近度来选择最佳风格的图像，这有助于避免风格化视频中不自然的细节。同时，在目标攻击中还考虑了目标类置信度，通过将风格化视频移动到更接近甚至跨越决策边界的位置来影响分类器的输出分布。然后采用无梯度方法进一步优化对抗性扰动。我们在两个标准数据集UCF-101和HMDB-51上进行了大量的实验来评估StyleFool。实验结果表明，StyleFool在查询次数和对现有防御的健壮性方面都优于最先进的对抗性攻击。我们发现，在非定向攻击中，50%的风格化视频不需要任何查询，因为它们已经可以欺骗视频分类模型。此外，我们通过用户研究对StyleFool的不可区分性进行了评估，以表明StyleFool的敌意样本在人眼看来是不可察觉的，尽管存在无限的扰动。



## **48. NICGSlowDown: Evaluating the Efficiency Robustness of Neural Image Caption Generation Models**

NICGSlowDown：评估神经图像字幕生成模型的效率和稳健性 cs.CV

This paper is accepted at CVPR2022

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2203.15859v1)

**Authors**: Simin Chen, Zihe Song, Mirazul Haque, Cong Liu, Wei Yang

**Abstracts**: Neural image caption generation (NICG) models have received massive attention from the research community due to their excellent performance in visual understanding. Existing work focuses on improving NICG model accuracy while efficiency is less explored. However, many real-world applications require real-time feedback, which highly relies on the efficiency of NICG models. Recent research observed that the efficiency of NICG models could vary for different inputs. This observation brings in a new attack surface of NICG models, i.e., An adversary might be able to slightly change inputs to cause the NICG models to consume more computational resources. To further understand such efficiency-oriented threats, we propose a new attack approach, NICGSlowDown, to evaluate the efficiency robustness of NICG models. Our experimental results show that NICGSlowDown can generate images with human-unnoticeable perturbations that will increase the NICG model latency up to 483.86%. We hope this research could raise the community's concern about the efficiency robustness of NICG models.

摘要: 神经图像字幕生成(NICG)模型因其在视觉理解方面的优异性能而受到研究界的广泛关注。现有的工作主要集中在提高NICG模型的精度上，而对效率的研究较少。然而，许多现实世界的应用需要实时反馈，这高度依赖于NICG模型的效率。最近的研究发现，对于不同的投入，NICG模型的效率可能会有所不同。这一观察结果为NICG模型带来了一个新的攻击面，即对手可能能够稍微改变输入以导致NICG模型消耗更多的计算资源。为了进一步理解这种面向效率的威胁，我们提出了一种新的攻击方法NICGSlowDown来评估NICG模型的效率健壮性。我们的实验结果表明，NICGSlowDown可以生成具有人类不可察觉的扰动的图像，这将使NICG模型的延迟增加483.86%。我们希望这项研究能引起社会各界对NICG模型的效率和稳健性的关注。



## **49. Characterizing the adversarial vulnerability of speech self-supervised learning**

表征语音自监督学习的对抗性脆弱性 cs.SD

Accepted by ICASSP 2022

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2111.04330v2)

**Authors**: Haibin Wu, Bo Zheng, Xu Li, Xixin Wu, Hung-yi Lee, Helen Meng

**Abstracts**: A leaderboard named Speech processing Universal PERformance Benchmark (SUPERB), which aims at benchmarking the performance of a shared self-supervised learning (SSL) speech model across various downstream speech tasks with minimal modification of architectures and small amount of data, has fueled the research for speech representation learning. The SUPERB demonstrates speech SSL upstream models improve the performance of various downstream tasks through just minimal adaptation. As the paradigm of the self-supervised learning upstream model followed by downstream tasks arouses more attention in the speech community, characterizing the adversarial robustness of such paradigm is of high priority. In this paper, we make the first attempt to investigate the adversarial vulnerability of such paradigm under the attacks from both zero-knowledge adversaries and limited-knowledge adversaries. The experimental results illustrate that the paradigm proposed by SUPERB is seriously vulnerable to limited-knowledge adversaries, and the attacks generated by zero-knowledge adversaries are with transferability. The XAB test verifies the imperceptibility of crafted adversarial attacks.

摘要: 一个名为语音处理通用性能基准(SUBB)的排行榜推动了语音表示学习的研究，该基准测试旨在以最小的体系结构和少量数据对共享的自我监督学习(SSLE)语音模型在各种下游语音任务中的性能进行基准测试。这一出色的演示表明，语音SSL上行模型仅通过最小的适配就可以提高各种下游任务的性能。随着上游自主学习模型和下游任务的学习范式在语音界引起了更多的关注，表征这种范式的对抗性稳健性是当务之急。在本文中，我们首次尝试研究了这种范式在零知识和有限知识两种攻击下的攻击脆弱性。实验结果表明，Superb提出的范式对有限知识攻击者具有很强的脆弱性，而零知识攻击者产生的攻击具有可移植性。Xab测试验证了精心设计的敌意攻击的隐蔽性。



## **50. Adaptative Perturbation Patterns: Realistic Adversarial Learning for Robust Intrusion Detection**

自适应扰动模式：用于稳健入侵检测的现实对抗性学习 cs.CR

18 pages, 6 tables, 10 figures, Future Internet journal

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2203.04234v2)

**Authors**: João Vitorino, Nuno Oliveira, Isabel Praça

**Abstracts**: Adversarial attacks pose a major threat to machine learning and to the systems that rely on it. In the cybersecurity domain, adversarial cyber-attack examples capable of evading detection are especially concerning. Nonetheless, an example generated for a domain with tabular data must be realistic within that domain. This work establishes the fundamental constraint levels required to achieve realism and introduces the Adaptative Perturbation Pattern Method (A2PM) to fulfill these constraints in a gray-box setting. A2PM relies on pattern sequences that are independently adapted to the characteristics of each class to create valid and coherent data perturbations. The proposed method was evaluated in a cybersecurity case study with two scenarios: Enterprise and Internet of Things (IoT) networks. Multilayer Perceptron (MLP) and Random Forest (RF) classifiers were created with regular and adversarial training, using the CIC-IDS2017 and IoT-23 datasets. In each scenario, targeted and untargeted attacks were performed against the classifiers, and the generated examples were compared with the original network traffic flows to assess their realism. The obtained results demonstrate that A2PM provides a scalable generation of realistic adversarial examples, which can be advantageous for both adversarial training and attacks.

摘要: 对抗性攻击对机器学习和依赖机器学习的系统构成了重大威胁。在网络安全领域，能够逃避检测的敌意网络攻击实例尤其令人担忧。尽管如此，为包含表格数据的域生成的示例在该域中必须是真实的。这项工作建立了实现真实感所需的基本约束水平，并引入了自适应扰动模式方法(A2PM)来满足灰箱设置中的这些约束。A2PM依赖于独立适应每一类的特征的模式序列来创建有效且一致的数据扰动。该方法在一个网络安全案例研究中进行了评估，其中包含两个场景：企业和物联网(IoT)网络。使用CIC-IDS2017和IoT-23数据集，通过定期和对抗性训练创建了多层感知器(MLP)和随机森林(RF)分类器。在每个场景中，对分类器执行目标攻击和非目标攻击，并将生成的示例与原始网络流量进行比较，以评估其真实性。所获得的结果表明，A2PM提供了可扩展的真实对抗性实例生成，这对于对抗性训练和攻击都是有利的。



