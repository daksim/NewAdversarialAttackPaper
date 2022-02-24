# Latest Adversarial Attack Papers
**update at 2022-02-24 09:40:01**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Sound Adversarial Audio-Visual Navigation**

声音对抗性视听导航 cs.SD

This work aims to do an adversarial sound intervention for robust  audio-visual navigation

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10910v1)

**Authors**: Yinfeng Yu, Wenbing Huang, Fuchun Sun, Changan Chen, Yikai Wang, Xiaohong Liu

**Abstracts**: Audio-visual navigation task requires an agent to find a sound source in a realistic, unmapped 3D environment by utilizing egocentric audio-visual observations. Existing audio-visual navigation works assume a clean environment that solely contains the target sound, which, however, would not be suitable in most real-world applications due to the unexpected sound noise or intentional interference. In this work, we design an acoustically complex environment in which, besides the target sound, there exists a sound attacker playing a zero-sum game with the agent. More specifically, the attacker can move and change the volume and category of the sound to make the agent suffer from finding the sounding object while the agent tries to dodge the attack and navigate to the goal under the intervention. Under certain constraints to the attacker, we can improve the robustness of the agent towards unexpected sound attacks in audio-visual navigation. For better convergence, we develop a joint training mechanism by employing the property of a centralized critic with decentralized actors. Experiments on two real-world 3D scan datasets, Replica, and Matterport3D, verify the effectiveness and the robustness of the agent trained under our designed environment when transferred to the clean environment or the one containing sound attackers with random policy. Project: \url{https://yyf17.github.io/SAAVN}.

摘要: 视听导航任务要求代理通过利用以自我为中心的视听观察，在真实的、未映射的3D环境中找到声源。现有的视听导航作品假设一个干净的环境，只包含目标声音，然而，由于意外的声音噪声或故意的干扰，这在大多数现实世界的应用中是不合适的。在本工作中，我们设计了一个复杂的声学环境，在这个环境中，除了目标声音外，还有一个声音攻击者与Agent进行零和游戏。更具体地说，攻击者可以移动和改变声音的音量和类别，使代理在试图躲避攻击并导航到干预下的目标时，难以找到探测对象。在对攻击者有一定约束的情况下，可以提高Agent对视听导航中意外声音攻击的鲁棒性。为了更好地收敛，我们利用集中批评家和分散参与者的性质开发了一种联合训练机制。在两个真实的三维扫描数据集Replica和Matterport3D上进行了实验，验证了在我们设计的环境下训练的代理在迁移到干净的环境和包含随机策略的声音攻击者的环境下的有效性和健壮性。项目：\url{https://yyf17.github.io/SAAVN}.



## **2. DEMO: Relay/Replay Attacks on GNSS signals**

演示：对GNSS信号的中继/重放攻击 cs.CR

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10897v1)

**Authors**: M. Lenhart, M. Spanghero, P. Papadimitratos

**Abstracts**: Global Navigation Satellite Systems (GNSS) are ubiquitously relied upon for positioning and timing. Detection and prevention of attacks against GNSS have been researched over the last decades, but many of these attacks and countermeasures were evaluated based on simulation. This work contributes to the experimental investigation of GNSS vulnerabilities, implementing a relay/replay attack with off-the-shelf hardware. Operating at the signal level, this attack type is not hindered by cryptographically protected transmissions, such as Galileo's Open Signals Navigation Message Authentication (OS-NMA). The attack we investigate involves two colluding adversaries, relaying signals over large distances, to effectively spoof a GNSS receiver. We demonstrate the attack using off-the-shelf hardware, we investigate the requirements for such successful colluding attacks, and how they can be enhanced, e.g., allowing for finer adversarial control over the victim receiver.

摘要: 全球导航卫星系统(GNSS)的定位和授时无处不在地依赖于全球导航卫星系统(GNSS)。近几十年来，对GNSS攻击的检测和预防一直在研究之中，但许多攻击和对策都是基于仿真进行评估的。这项工作有助于对GNSS漏洞的实验研究，使用现成的硬件实现中继/重放攻击。在信号级运行，这种攻击类型不会受到密码保护传输的阻碍，例如伽利略的开放信号导航消息验证(OS-NMA)。我们调查的攻击涉及两个串通的对手，他们远距离中继信号，以有效地欺骗GNSS接收器。我们使用现成的硬件演示了攻击，我们调查了这种成功的合谋攻击的要求，以及如何增强这些要求，例如，允许对受害者接收器进行更精细的敌意控制。



## **3. Protecting GNSS-based Services using Time Offset Validation**

使用时间偏移验证保护基于GNSS的服务 cs.CR

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10891v1)

**Authors**: K. Zhang, M. Spanghero, P. Papadimitratos

**Abstracts**: Global navigation satellite systems (GNSS) provide pervasive accurate positioning and timing services for a large gamut of applications, from Time based One-Time Passwords (TOPT), to power grid and cellular systems. However, there can be security concerns for the applications due to the vulnerability of GNSS. It is important to observe that GNSS receivers are components of platforms, in principle having rich connectivity to different network infrastructures. Of particular interest is the access to a variety of timing sources, as those can be used to validate GNSS-provided location and time. Therefore, we consider off-the-shelf platforms and how to detect if the GNSS receiver is attacked or not, by cross-checking the GNSS time and time from other available sources. First, we survey different technologies to analyze their availability, accuracy, and trustworthiness for time synchronization. Then, we propose a validation approach for absolute and relative time. Moreover, we design a framework and experimental setup for the evaluation of the results. Attacks can be detected based on WiFi supplied time when the adversary shifts the GNSS provided time, more than 23.942us; with Network Time Protocol (NTP) supplied time when the adversary-induced shift is more than 2.046ms. Consequently, the proposal significantly limits the capability of an adversary to manipulate the victim GNSS receiver.

摘要: 全球导航卫星系统(GNSS)为从基于时间的一次性密码(TOPT)到电网和蜂窝系统的大量应用提供无处不在的精确定位和授时服务。然而，由于GNSS的脆弱性，应用程序可能存在安全问题。重要的是要注意到，全球导航卫星系统接收器是平台的组成部分，原则上与不同的网络基础设施有丰富的连接。特别令人感兴趣的是对各种计时源的访问，因为这些时间源可用于验证全球导航卫星系统提供的位置和时间。因此，我们考虑了现成的平台，以及如何通过交叉检查GNSS时间和其他可用来源的时间来检测GNSS接收机是否受到攻击。首先，我们综述了不同的时间同步技术，以分析它们在时间同步方面的可用性、准确性和可信性。然后，我们提出了一种绝对时间和相对时间的验证方法。此外，我们还设计了评价结果的框架和实验装置。当对手移动GNSS提供的时间大于23.942us时，可以基于WiFi提供的时间检测攻击；当对手引起的移动超过2.046ms时，基于网络时间协议(NTP)提供的时间可以检测到攻击。因此，该提案极大地限制了对手操纵受害者GNSS接收器的能力。



## **4. FastZIP: Faster and More Secure Zero-Interaction Pairing**

FastZip：更快、更安全的零交互配对 cs.CR

ACM MobiSys '21; Fixed ambiguity in flow diagram (Figure 2). Code and  data are available at: https://github.com/seemoo-lab/fastzip

**SubmitDate**: 2022-02-23    [paper-pdf](http://arxiv.org/pdf/2106.04907v3)

**Authors**: Mikhail Fomichev, Julia Hesse, Lars Almon, Timm Lippert, Jun Han, Matthias Hollick

**Abstracts**: With the advent of the Internet of Things (IoT), establishing a secure channel between smart devices becomes crucial. Recent research proposes zero-interaction pairing (ZIP), which enables pairing without user assistance by utilizing devices' physical context (e.g., ambient audio) to obtain a shared secret key. The state-of-the-art ZIP schemes suffer from three limitations: (1) prolonged pairing time (i.e., minutes or hours), (2) vulnerability to brute-force offline attacks on a shared key, and (3) susceptibility to attacks caused by predictable context (e.g., replay attack) because they rely on limited entropy of physical context to protect a shared key. We address these limitations, proposing FastZIP, a novel ZIP scheme that significantly reduces pairing time while preventing offline and predictable context attacks. In particular, we adapt a recently introduced Fuzzy Password-Authenticated Key Exchange (fPAKE) protocol and utilize sensor fusion, maximizing their advantages. We instantiate FastZIP for intra-car device pairing to demonstrate its feasibility and show how the design of FastZIP can be adapted to other ZIP use cases. We implement FastZIP and evaluate it by driving four cars for a total of 800 km. We achieve up to three times shorter pairing time compared to the state-of-the-art ZIP schemes while assuring robust security with adversarial error rates below 0.5%.

摘要: 随着物联网(IoT)的到来，在智能设备之间建立安全通道变得至关重要。最近的研究提出了零交互配对(ZIP)，它通过利用设备的物理上下文(例如，环境音频)来获得共享密钥，从而在没有用户帮助的情况下实现配对。现有的ZIP方案有三个局限性：(1)配对时间延长(即几分钟或几小时)，(2)容易受到对共享密钥的暴力离线攻击，以及(3)容易受到由可预测上下文引起的攻击(例如，重放攻击)，因为它们依赖于有限的物理上下文熵来保护共享密钥。我们解决了这些限制，提出了FastZip，这是一种新颖的ZIP方案，可以显著减少配对时间，同时防止离线和可预测的上下文攻击。特别是，我们采用了最近推出的模糊口令认证密钥交换(FPAKE)协议，并利用传感器融合，最大限度地发挥了它们的优势。我们将FastZip实例化用于车内设备配对，以演示其可行性，并展示FastZip的设计如何适用于其他ZIP用例。我们实施了FastZip，并通过驾驶4辆车总共行驶800公里对其进行评估。与最先进的ZIP方案相比，我们实现了高达3倍的配对时间，同时确保了健壮的安全性，对手错误率低于0.5%。



## **5. Adversarial Defense by Latent Style Transformations**

潜在风格转换的对抗性防御 cs.CV

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2006.09701v2)

**Authors**: Shuo Wang, Surya Nepal, Alsharif Abuadbba, Carsten Rudolph, Marthie Grobler

**Abstracts**: Machine learning models have demonstrated vulnerability to adversarial attacks, more specifically misclassification of adversarial examples.   In this paper, we investigate an attack-agnostic defense against adversarial attacks on high-resolution images by detecting suspicious inputs.   The intuition behind our approach is that the essential characteristics of a normal image are generally consistent with non-essential style transformations, e.g., slightly changing the facial expression of human portraits.   In contrast, adversarial examples are generally sensitive to such transformations.   In our approach to detect adversarial instances, we propose an in\underline{V}ertible \underline{A}utoencoder based on the \underline{S}tyleGAN2 generator via \underline{A}dversarial training (VASA) to inverse images to disentangled latent codes that reveal hierarchical styles.   We then build a set of edited copies with non-essential style transformations by performing latent shifting and reconstruction, based on the correspondences between latent codes and style transformations.   The classification-based consistency of these edited copies is used to distinguish adversarial instances.

摘要: 机器学习模型已经显示出对敌意攻击的脆弱性，更具体地说，是对对抗性例子的错误分类。在这篇文章中，我们研究了一种通过检测可疑输入来抵抗高分辨率图像上的敌意攻击的攻击不可知性防御方法。我们的方法背后的直觉是，正常图像的基本特征通常与非必要的样式转换一致，例如，稍微改变人物肖像的面部表情。相反，对抗性的例子通常对这样的转换很敏感。在检测敌意实例的方法中，我们提出了一种基于下划线{S}tyleGAN2生成器的下划线{V}易错下划线{A}utoender，它通过下划线{A}变异训练(VASA)将图像逆变成显示分层样式的解缠潜代码。然后根据潜在代码和样式转换之间的对应关系，通过潜移位和重构，构建一组具有非本质样式转换的编辑副本。这些编辑副本的基于分类的一致性被用来区分对抗性实例。



## **6. Surrogate Representation Learning with Isometric Mapping for Gray-box Graph Adversarial Attacks**

基于等距映射的灰盒图对抗攻击代理表示学习 cs.AI

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2110.10482v3)

**Authors**: Zihan Liu, Yun Luo, Zelin Zang, Stan Z. Li

**Abstracts**: Gray-box graph attacks aim at disrupting the performance of the victim model by using inconspicuous attacks with limited knowledge of the victim model. The parameters of the victim model and the labels of the test nodes are invisible to the attacker. To obtain the gradient on the node attributes or graph structure, the attacker constructs an imaginary surrogate model trained under supervision. However, there is a lack of discussion on the training of surrogate models and the robustness of provided gradient information. The general node classification model loses the topology of the nodes on the graph, which is, in fact, an exploitable prior for the attacker. This paper investigates the effect of representation learning of surrogate models on the transferability of gray-box graph adversarial attacks. To reserve the topology in the surrogate embedding, we propose Surrogate Representation Learning with Isometric Mapping (SRLIM). By using Isometric mapping method, our proposed SRLIM can constrain the topological structure of nodes from the input layer to the embedding space, that is, to maintain the similarity of nodes in the propagation process. Experiments prove the effectiveness of our approach through the improvement in the performance of the adversarial attacks generated by the gradient-based attacker in untargeted poisoning gray-box setups.

摘要: 灰盒图攻击的目的是在有限的受害者模型知识下，利用不明显的攻击破坏受害者模型的性能。受害者模型的参数和测试节点的标签对攻击者是不可见的。为了获得节点属性或图结构上的梯度，攻击者构建了一个在监督下训练的虚拟代理模型。然而，对于代理模型的训练和提供的梯度信息的稳健性，目前还缺乏讨论。一般节点分类模型会丢失图上节点的拓扑，这实际上是攻击者可以利用的先验信息。研究了代理模型的表示学习对灰盒图对抗攻击可转移性的影响。为了保留代理嵌入中的拓扑结构，我们提出了基于等距映射的代理表示学习算法(SRLIM)。通过使用等距映射方法，我们提出的SRLIM可以将节点的拓扑结构从输入层约束到嵌入空间，即在传播过程中保持节点的相似性。实验证明，在无目标中毒灰盒设置下，基于梯度的攻击者生成的对抗性攻击的性能得到了提高，证明了该方法的有效性。



## **7. Universal adversarial perturbation for remote sensing images**

遥感图像的普遍对抗性摄动 cs.CV

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10693v1)

**Authors**: Zhaoxia Yin, Qingyu Wang, Jin Tang, Bin Luo

**Abstracts**: Recently, with the application of deep learning in the remote sensing image (RSI) field, the classification accuracy of the RSI has been greatly improved compared with traditional technology. However, even state-of-the-art object recognition convolutional neural networks are fooled by the universal adversarial perturbation (UAP). To verify that UAP makes the RSI classification model error classification, this paper proposes a novel method combining an encoder-decoder network with an attention mechanism. Firstly, the former can learn the distribution of perturbations better, then the latter is used to find the main regions concerned by the RSI classification model. Finally, the generated regions are used to fine-tune the perturbations making the model misclassified with fewer perturbations. The experimental results show that the UAP can make the RSI misclassify, and the attack success rate (ASR) of our proposed method on the RSI data set is as high as 97.35%.

摘要: 近年来，随着深度学习技术在遥感图像领域的应用，遥感图像的分类精度与传统技术相比有了很大的提高。然而，即使是最先进的目标识别卷积神经网络也会被普遍的对抗性摄动(UAP)所欺骗。为了验证UAP对RSI分类模型进行错误分类，提出了一种编解码器网络与注意力机制相结合的新方法。前者能更好地学习扰动的分布，后者用于寻找RSI分类模型关注的主要区域。最后，生成的区域被用来微调扰动，使得模型在较少扰动的情况下被误分类。实验结果表明，UAP可以使RSI发生误分类，本文提出的方法在RSI数据集上的攻击成功率高达97.35%。



## **8. Seeing is Living? Rethinking the Security of Facial Liveness Verification in the Deepfake Era**

看就是活？深伪时代下人脸活体验证安全性的再思考 cs.CR

Accepted as a full paper at USENIX Security '22

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10673v1)

**Authors**: Changjiang Li, Li Wang, Shouling Ji, Xuhong Zhang, Zhaohan Xi, Shanqing Guo, Ting Wang

**Abstracts**: Facial Liveness Verification (FLV) is widely used for identity authentication in many security-sensitive domains and offered as Platform-as-a-Service (PaaS) by leading cloud vendors. Yet, with the rapid advances in synthetic media techniques (e.g., deepfake), the security of FLV is facing unprecedented challenges, about which little is known thus far.   To bridge this gap, in this paper, we conduct the first systematic study on the security of FLV in real-world settings. Specifically, we present LiveBugger, a new deepfake-powered attack framework that enables customizable, automated security evaluation of FLV. Leveraging LiveBugger, we perform a comprehensive empirical assessment of representative FLV platforms, leading to a set of interesting findings. For instance, most FLV APIs do not use anti-deepfake detection; even for those with such defenses, their effectiveness is concerning (e.g., it may detect high-quality synthesized videos but fail to detect low-quality ones). We then conduct an in-depth analysis of the factors impacting the attack performance of LiveBugger: a) the bias (e.g., gender or race) in FLV can be exploited to select victims; b) adversarial training makes deepfake more effective to bypass FLV; c) the input quality has a varying influence on different deepfake techniques to bypass FLV. Based on these findings, we propose a customized, two-stage approach that can boost the attack success rate by up to 70%. Further, we run proof-of-concept attacks on several representative applications of FLV (i.e., the clients of FLV APIs) to illustrate the practical implications: due to the vulnerability of the APIs, many downstream applications are vulnerable to deepfake. Finally, we discuss potential countermeasures to improve the security of FLV. Our findings have been confirmed by the corresponding vendors.

摘要: 面部活体验证(FLV)广泛用于许多安全敏感领域的身份验证，并由领先的云供应商以平台即服务(PaaS)的形式提供。然而，随着合成媒体技术(如深度假冒)的快速发展，FLV的安全正面临着前所未有的挑战，目前对此知之甚少。为了弥补这一差距，本文首次对FLV在现实环境下的安全性进行了系统的研究。具体地说，我们介绍了LiveBugger，这是一个新的深度假冒支持的攻击框架，可以对FLV进行可定制的、自动化的安全评估。利用LiveBugger，我们对有代表性的FLV平台进行了全面的实证评估，得出了一系列有趣的发现。例如，大多数FLV API不使用防深伪检测，即使是有这种防御的API，其有效性也是令人担忧的(例如，它可能会检测到高质量的合成视频，但无法检测到低质量的合成视频)。然后，我们深入分析了影响LiveBugger攻击性能的因素：a)FLV中的偏见(如性别或种族)可以被用来选择受害者；b)对抗性训练使深伪更有效地绕过FLV；c)输入质量对不同的绕过FLV的深伪技术有不同的影响。基于这些发现，我们提出了一种定制的两阶段方法，可以将攻击成功率提高高达70%。此外，我们对FLV的几个有代表性的应用程序(即FLV API的客户端)进行了概念验证攻击，以说明其实际意义：由于API的脆弱性，许多下游应用程序都容易受到深度假冒的攻击。最后，我们讨论了提高FLV安全性的潜在对策。我们的发现已经得到了相应供应商的证实。



## **9. Fingerprinting Deep Neural Networks Globally via Universal Adversarial Perturbations**

基于普遍对抗性扰动的深度神经网络全局指纹识别 cs.CR

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.08602v2)

**Authors**: Zirui Peng, Shaofeng Li, Guoxing Chen, Cheng Zhang, Haojin Zhu, Minhui Xue

**Abstracts**: In this paper, we propose a novel and practical mechanism which enables the service provider to verify whether a suspect model is stolen from the victim model via model extraction attacks. Our key insight is that the profile of a DNN model's decision boundary can be uniquely characterized by its \textit{Universal Adversarial Perturbations (UAPs)}. UAPs belong to a low-dimensional subspace and piracy models' subspaces are more consistent with victim model's subspace compared with non-piracy model. Based on this, we propose a UAP fingerprinting method for DNN models and train an encoder via \textit{contrastive learning} that takes fingerprint as inputs, outputs a similarity score. Extensive studies show that our framework can detect model IP breaches with confidence $> 99.99 \%$ within only $20$ fingerprints of the suspect model. It has good generalizability across different model architectures and is robust against post-modifications on stolen models.

摘要: 本文提出了一种新颖而实用的机制，使服务提供商能够通过模型提取攻击来验证受害者模型中的可疑模型是否被窃取。我们的主要见解是DNN模型的决策边界的轮廓可以由它的\textit(通用对抗性扰动(UAP))来唯一地刻画。UAP属于低维子空间，与非盗版模型相比，盗版模型的子空间与受害者模型的子空间更加一致。在此基础上，提出了一种DNN模型的UAP指纹识别方法，并以指纹为输入，通过对比学习训练编码器，输出相似度得分。大量的研究表明，我们的框架可以在可疑模型的$2 0$指纹范围内以>99.99$的置信度检测到模型IP泄露。它具有良好的跨不同模型体系结构的通用性，并且对窃取模型的后期修改具有健壮性。



## **10. Robust Stochastic Linear Contextual Bandits Under Adversarial Attacks**

对抗性攻击下的鲁棒随机线性上下文带 stat.ML

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2106.02978v2)

**Authors**: Qin Ding, Cho-Jui Hsieh, James Sharpnack

**Abstracts**: Stochastic linear contextual bandit algorithms have substantial applications in practice, such as recommender systems, online advertising, clinical trials, etc. Recent works show that optimal bandit algorithms are vulnerable to adversarial attacks and can fail completely in the presence of attacks. Existing robust bandit algorithms only work for the non-contextual setting under the attack of rewards and cannot improve the robustness in the general and popular contextual bandit environment. In addition, none of the existing methods can defend against attacked context. In this work, we provide the first robust bandit algorithm for stochastic linear contextual bandit setting under a fully adaptive and omniscient attack with sub-linear regret. Our algorithm not only works under the attack of rewards, but also under attacked context. Moreover, it does not need any information about the attack budget or the particular form of the attack. We provide theoretical guarantees for our proposed algorithm and show by experiments that our proposed algorithm improves the robustness against various kinds of popular attacks.

摘要: 随机线性上下文盗贼算法在推荐系统、在线广告、临床试验等领域有着广泛的应用。最近的研究表明，最优盗贼算法很容易受到敌意攻击，并且在存在攻击的情况下可能完全失效。现有的鲁棒盗版算法只适用于奖励攻击下的非上下文环境，不能提高在一般流行的上下文盗版环境下的鲁棒性。此外，现有的方法都不能抵御上下文攻击。在这项工作中，我们针对随机线性上下文盗贼设置，在具有子线性遗憾的完全自适应和全知攻击下，提供了第一个鲁棒盗贼算法。我们的算法不仅可以在奖励攻击下工作，而且可以在受攻击的环境下工作。此外，它不需要关于攻击预算或特定攻击形式的任何信息。我们为我们提出的算法提供了理论上的保证，实验表明，我们提出的算法提高了对各种流行攻击的鲁棒性。



## **11. Behaviour-Diverse Automatic Penetration Testing: A Curiosity-Driven Multi-Objective Deep Reinforcement Learning Approach**

行为多样化的自动渗透测试：一种好奇心驱动的多目标深度强化学习方法 cs.LG

6 pages,4 Figures

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10630v1)

**Authors**: Yizhou Yang, Xin Liu

**Abstracts**: Penetration Testing plays a critical role in evaluating the security of a target network by emulating real active adversaries. Deep Reinforcement Learning (RL) is seen as a promising solution to automating the process of penetration tests by reducing human effort and improving reliability. Existing RL solutions focus on finding a specific attack path to impact the target hosts. However, in reality, a diverse range of attack variations are needed to provide comprehensive assessments of the target network's security level. Hence, the attack agents must consider multiple objectives when penetrating the network. Nevertheless, this challenge is not adequately addressed in the existing literature. To this end, we formulate the automatic penetration testing in the Multi-Objective Reinforcement Learning (MORL) framework and propose a Chebyshev decomposition critic to find diverse adversary strategies that balance different objectives in the penetration test. Additionally, the number of available actions increases with the agent consistently probing the target network, making the training process intractable in many practical situations. Thus, we introduce a coverage-based masking mechanism that reduces attention on previously selected actions to help the agent adapt to future exploration. Experimental evaluation on a range of scenarios demonstrates the superiority of our proposed approach when compared to adapted algorithms in terms of multi-objective learning and performance efficiency.

摘要: 渗透测试通过模拟真实的主动对手，在评估目标网络的安全性方面起着至关重要的作用。深度强化学习(RL)被认为是一种很有前途的解决方案，可以通过减少人工工作量和提高可靠性来实现渗透测试过程的自动化。现有的RL解决方案侧重于寻找特定的攻击路径来影响目标主机。然而，在现实中，需要一系列不同的攻击变体来提供对目标网络安全级别的全面评估。因此，攻击代理在渗透网络时必须考虑多个目标。然而，在现有的文献中，这一挑战没有得到充分的解决。为此，我们制定了多目标强化学习(MORL)框架中的自动渗透测试，并提出了切比雪夫分解批评者来寻找在渗透测试中平衡不同目标的不同对手策略。此外，随着代理持续探测目标网络，可用操作的数量增加，使得培训过程在许多实际情况下变得棘手。因此，我们引入了一种基于覆盖的掩蔽机制，该机制减少了对先前选择的操作的关注，以帮助代理适应未来的探索。在一系列场景上的实验评估表明，与自适应算法相比，我们提出的方法在多目标学习和性能效率方面具有优越性。



## **12. On the Effectiveness of Adversarial Training against Backdoor Attacks**

论对抗性训练对抗后门攻击的有效性 cs.LG

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10627v1)

**Authors**: Yinghua Gao, Dongxian Wu, Jingfeng Zhang, Guanhao Gan, Shu-Tao Xia, Gang Niu, Masashi Sugiyama

**Abstracts**: DNNs' demand for massive data forces practitioners to collect data from the Internet without careful check due to the unacceptable cost, which brings potential risks of backdoor attacks. A backdoored model always predicts a target class in the presence of a predefined trigger pattern, which can be easily realized via poisoning a small amount of data. In general, adversarial training is believed to defend against backdoor attacks since it helps models to keep their prediction unchanged even if we perturb the input image (as long as within a feasible range). Unfortunately, few previous studies succeed in doing so. To explore whether adversarial training could defend against backdoor attacks or not, we conduct extensive experiments across different threat models and perturbation budgets, and find the threat model in adversarial training matters. For instance, adversarial training with spatial adversarial examples provides notable robustness against commonly-used patch-based backdoor attacks. We further propose a hybrid strategy which provides satisfactory robustness across different backdoor attacks.

摘要: DNNS对海量数据的需求迫使从业者在没有仔细检查的情况下从互联网上收集数据，这是因为不可接受的成本，这带来了后门攻击的潜在风险。回溯模型总是在预定义的触发器模式存在的情况下预测目标类，这可以通过毒化少量数据轻松实现。一般说来，对抗性训练被认为可以抵御后门攻击，因为它有助于模型保持预测不变，即使我们扰乱了输入图像(只要在可行的范围内)。不幸的是，之前的研究很少成功做到这一点。为了探索对抗性训练是否能够抵御后门攻击，我们在不同的威胁模型和扰动预算上进行了广泛的实验，找到了对抗性训练事项中的威胁模型。例如，带有空间对抗性示例的对抗性训练针对常用的基于补丁的后门攻击提供了显著的健壮性。我们进一步提出了一种混合策略，该策略能够在不同的后门攻击中提供令人满意的健壮性。



## **13. Adversarial Attacks on Speech Recognition Systems for Mission-Critical Applications: A Survey**

针对任务关键型应用的语音识别系统的敌意攻击：综述 cs.SD

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10594v1)

**Authors**: Ngoc Dung Huynh, Mohamed Reda Bouadjenek, Imran Razzak, Kevin Lee, Chetan Arora, Ali Hassani, Arkady Zaslavsky

**Abstracts**: A Machine-Critical Application is a system that is fundamentally necessary to the success of specific and sensitive operations such as search and recovery, rescue, military, and emergency management actions. Recent advances in Machine Learning, Natural Language Processing, voice recognition, and speech processing technologies have naturally allowed the development and deployment of speech-based conversational interfaces to interact with various machine-critical applications. While these conversational interfaces have allowed users to give voice commands to carry out strategic and critical activities, their robustness to adversarial attacks remains uncertain and unclear. Indeed, Adversarial Artificial Intelligence (AI) which refers to a set of techniques that attempt to fool machine learning models with deceptive data, is a growing threat in the AI and machine learning research community, in particular for machine-critical applications. The most common reason of adversarial attacks is to cause a malfunction in a machine learning model. An adversarial attack might entail presenting a model with inaccurate or fabricated samples as it's training data, or introducing maliciously designed data to deceive an already trained model. While focusing on speech recognition for machine-critical applications, in this paper, we first review existing speech recognition techniques, then, we investigate the effectiveness of adversarial attacks and defenses against these systems, before outlining research challenges, defense recommendations, and future work. This paper is expected to serve researchers and practitioners as a reference to help them in understanding the challenges, position themselves and, ultimately, help them to improve existing models of speech recognition for mission-critical applications. Keywords: Mission-Critical Applications, Adversarial AI, Speech Recognition Systems.

摘要: 机器关键型应用程序是成功完成特定和敏感操作(如搜索和恢复、救援、军事和紧急管理操作)所必需的系统。机器学习、自然语言处理、语音识别和语音处理技术的最新进展自然允许开发和部署基于语音的会话界面，以与各种机器关键应用程序交互。虽然这些对话界面允许用户发出语音命令来执行战略和关键活动，但它们对对手攻击的健壮性仍然不确定和不清楚。事实上，对抗性人工智能(AI)是指一套试图用欺骗性数据愚弄机器学习模型的技术，在AI和机器学习研究界是一个越来越大的威胁，特别是对于机器关键应用。敌意攻击最常见的原因是导致机器学习模型出现故障。敌意攻击可能需要向模型提供不准确或伪造的样本作为其训练数据，或者引入恶意设计的数据来欺骗已经训练过的模型。在重点研究机器关键应用中的语音识别的同时，本文首先回顾了现有的语音识别技术，然后研究了针对这些系统的对抗性攻击和防御的有效性，然后概述了研究挑战、防御建议和未来的工作。本文期望为研究人员和实践者提供参考，帮助他们了解挑战，定位自己，并最终帮助他们改进现有的关键任务应用的语音识别模型。关键词：任务关键型应用、对抗性人工智能、语音识别系统。



## **14. Privacy Leakage of Adversarial Training Models in Federated Learning Systems**

联合学习系统中对抗性训练模型的隐私泄露 cs.LG

6 pages, 6 figures. Submitted to CVPR'22 workshop "The Art of  Robustness"

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2202.10546v1)

**Authors**: Jingyang Zhang, Yiran Chen, Hai Li

**Abstracts**: Adversarial Training (AT) is crucial for obtaining deep neural networks that are robust to adversarial attacks, yet recent works found that it could also make models more vulnerable to privacy attacks. In this work, we further reveal this unsettling property of AT by designing a novel privacy attack that is practically applicable to the privacy-sensitive Federated Learning (FL) systems. Using our method, the attacker can exploit AT models in the FL system to accurately reconstruct users' private training images even when the training batch size is large. Code is available at https://github.com/zjysteven/PrivayAttack_AT_FL.

摘要: 对抗性训练(AT)对于获得对敌方攻击健壮的深层神经网络至关重要，然而最近的研究发现，它也会使模型更容易受到隐私攻击。在这项工作中，我们通过设计一种实际适用于隐私敏感的联邦学习(FL)系统的新型隐私攻击，进一步揭示了AT的这一令人不安的特性。使用我们的方法，攻击者可以利用FL系统中的AT模型来准确地重建用户的私人训练图像，即使在训练批量很大的情况下也是如此。代码可在https://github.com/zjysteven/PrivayAttack_AT_FL.上获得



## **15. Analysing Security and Privacy Threats in the Lockdown Periods of COVID-19 Pandemic: Twitter Dataset Case Study**

分析冠状病毒大流行封锁期的安全和隐私威胁：Twitter数据集案例研究 cs.CR

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2202.10543v1)

**Authors**: Bibhas Sharma, Ishan Karunanayake, Rahat Masood, Muhammad Ikram

**Abstracts**: The COVID-19 pandemic will be remembered as a uniquely disruptive period that altered the lives of billions of citizens globally, resulting in new-normal for the way people live and work. With the coronavirus pandemic, everyone had to adapt to the "work or study from home" operating model that has transformed our online lives and exponentially increased the use of cyberspace. Concurrently, there has been a huge spike in social media platforms such as Facebook and Twitter during the COVID-19 lockdown periods. These lockdown periods have resulted in a set of new cybercrimes, thereby allowing attackers to victimise users of social media platforms in times of fear, uncertainty, and doubt. The threats range from running phishing campaigns and malicious domains to extracting private information about victims for malicious purposes. This research paper performs a large-scale study to investigate the impact of lockdown periods during the COVID-19 pandemic on the security and privacy of social media users. We analyse 10.6 Million COVID-related tweets from 533 days of data crawling and investigate users' security and privacy behaviour in three different periods (i.e., before, during, and after lockdown). Our study shows that users unintentionally share more personal identifiable information when writing about the pandemic situation in their tweets. The privacy risk reaches 100% if a user posts three or more sensitive tweets about the pandemic. We investigate the number of suspicious domains shared in social media during different pandemic phases. Our analysis reveals an increase in suspicious domains during the lockdown compared to other lockdown phases. We observe that IT, Search Engines, and Businesses are the top three categories that contain suspicious domains. Our analysis reveals that adversaries' strategies to instigate malicious activities change with the country's pandemic situation.

摘要: 冠状病毒大流行将被铭记为一个独特的破坏性时期，它改变了全球数十亿公民的生活，导致人们生活和工作方式的新常态。随着冠状病毒的流行，每个人都不得不适应“在家工作或学习”的运营模式，这种模式改变了我们的在线生活，并成倍增加了对网络空间的使用。与此同时，在冠状病毒禁售期，Facebook和Twitter等社交媒体平台出现了巨大的激增。这些禁闭期导致了一系列新的网络犯罪，从而允许攻击者在恐惧、不确定和怀疑的时候伤害社交媒体平台的用户。威胁范围从运行钓鱼活动和恶意域到出于恶意目的提取受害者的私人信息。本研究对冠状病毒大流行期间的封锁期对社交媒体用户安全和隐私的影响进行了大规模研究。我们从533天的数据爬行中分析了1060万条与COVID相关的推文，并调查了用户在三个不同时期(即锁定前、锁定期间和锁定后)的安全和隐私行为。我们的研究表明，用户在推文中写下大流行情况时，无意中分享了更多的个人可识别信息。如果用户发布三条或三条以上关于疫情的敏感推文，隐私风险将达到100%。我们调查了不同流行阶段在社交媒体上共享的可疑域名的数量。我们的分析显示，与其他锁定阶段相比，锁定期间可疑域名的数量有所增加。我们发现IT、搜索引擎和企业是包含可疑域名的前三大类别。我们的分析显示，敌方煽动恶意活动的策略会随着该国大流行形势的变化而变化。



## **16. RAILS: A Robust Adversarial Immune-inspired Learning System**

Rails：一种健壮的对抗性免疫启发学习系统 cs.NE

arXiv admin note: text overlap with arXiv:2012.10485

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2107.02840v2)

**Authors**: Ren Wang, Tianqi Chen, Stephen Lindsly, Cooper Stansbury, Alnawaz Rehemtulla, Indika Rajapakse, Alfred Hero

**Abstracts**: Adversarial attacks against deep neural networks (DNNs) are continuously evolving, requiring increasingly powerful defense strategies. We develop a novel adversarial defense framework inspired by the adaptive immune system: the Robust Adversarial Immune-inspired Learning System (RAILS). Initializing a population of exemplars that is balanced across classes, RAILS starts from a uniform label distribution that encourages diversity and uses an evolutionary optimization process to adaptively adjust the predictive label distribution in a manner that emulates the way the natural immune system recognizes novel pathogens. RAILS' evolutionary optimization process explicitly captures the tradeoff between robustness (diversity) and accuracy (specificity) of the network, and represents a new immune-inspired perspective on adversarial learning. The benefits of RAILS are empirically demonstrated under eight types of adversarial attacks on a DNN adversarial image classifier for several benchmark datasets, including: MNIST; SVHN; CIFAR-10; and CIFAR-10. We find that PGD is the most damaging attack strategy and that for this attack RAILS is significantly more robust than other methods, achieving improvements in adversarial robustness by $\geq 5.62\%, 12.5\%$, $10.32\%$, and $8.39\%$, on these respective datasets, without appreciable loss of classification accuracy. Codes for the results in this paper are available at https://github.com/wangren09/RAILS.

摘要: 针对深度神经网络(DNNs)的敌意攻击正在不断演变，需要越来越强大的防御策略。在自适应免疫系统的启发下，我们提出了一种新的对抗性防御框架：鲁棒对抗性免疫启发学习系统(Rails)。Rails初始化跨类别平衡的样本群体，从鼓励多样性的统一标签分布开始，并使用进化优化过程以模拟自然免疫系统识别新病原体的方式自适应地调整预测性标签分布。Rails的进化优化过程明确地捕捉到了网络的稳健性(多样性)和准确性(特异性)之间的权衡，并代表了对抗性学习的一种受免疫启发的新视角。在对DNN对抗性图像分类器的八种类型的对抗性攻击下，针对几个基准数据集，包括：MNIST、SVHN、CIFAR-10和CIFAR-10，经验证明了Rails的好处。我们发现PGD是最具破坏性的攻击策略，对于这种攻击，Rails的鲁棒性明显高于其他方法，在这些数据集上分别获得了5.62美元、12.5美元、10.32美元和8.39美元的对手健壮性改善，而分类精度没有明显的损失。结果表明，PGD是最具破坏性的攻击策略，对于这种攻击，Rails的健壮性明显高于其他方法，在分类精度没有明显损失的情况下，分别提高了5.62美元、12.5美元、10.32美元和8.39美元。有关本文结果的代码，请访问https://github.com/wangren09/RAILS.



## **17. Adversarial Examples in Constrained Domains**

受限领域中的对抗性例子 cs.CR

Accepted to IOS Press Journal of Computer Security

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2011.01183v2)

**Authors**: Ryan Sheatsley, Nicolas Papernot, Michael Weisman, Gunjan Verma, Patrick McDaniel

**Abstracts**: Machine learning algorithms have been shown to be vulnerable to adversarial manipulation through systematic modification of inputs (e.g., adversarial examples) in domains such as image recognition. Under the default threat model, the adversary exploits the unconstrained nature of images; each feature (pixel) is fully under control of the adversary. However, it is not clear how these attacks translate to constrained domains that limit which and how features can be modified by the adversary (e.g., network intrusion detection). In this paper, we explore whether constrained domains are less vulnerable than unconstrained domains to adversarial example generation algorithms. We create an algorithm for generating adversarial sketches: targeted universal perturbation vectors which encode feature saliency within the envelope of domain constraints. To assess how these algorithms perform, we evaluate them in constrained (e.g., network intrusion detection) and unconstrained (e.g., image recognition) domains. The results demonstrate that our approaches generate misclassification rates in constrained domains that were comparable to those of unconstrained domains (greater than 95%). Our investigation shows that the narrow attack surface exposed by constrained domains is still sufficiently large to craft successful adversarial examples; and thus, constraints do not appear to make a domain robust. Indeed, with as little as five randomly selected features, one can still generate adversarial examples.

摘要: 已经证明机器学习算法通过对诸如图像识别等领域中的输入(例如，对抗性示例)进行系统修改而容易受到对抗性操纵。在默认威胁模型下，敌方利用图像的不受约束的性质；每个功能(像素)都完全在敌方的控制之下。然而，目前尚不清楚这些攻击如何转化为限制哪些特征以及如何被攻击者修改的约束域(例如，网络入侵检测)。在这篇文章中，我们探讨了约束域是否比非约束域更不容易受到敌意示例生成算法的影响。我们创建了一种生成对抗性草图的算法：目标通用扰动向量，它在域约束的包络内编码特征显著性。为了评估这些算法的性能，我们在受限(例如，网络入侵检测)和非受限(例如，图像识别)域中对它们进行评估。结果表明，我们的方法在受限领域产生的错误分类率与非约束领域相当(大于95%)。我们的调查表明，受约束域暴露的狭窄攻击面仍然足够大，足以伪造成功的敌意示例；因此，约束似乎不会使域变得健壮。事实上，只需随机选择5个特征，就仍然可以生成对抗性的例子。



## **18. A Tutorial on Adversarial Learning Attacks and Countermeasures**

对抗性学习攻击与对策教程 cs.CR

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2202.10377v1)

**Authors**: Cato Pauling, Michael Gimson, Muhammed Qaid, Ahmad Kida, Basel Halak

**Abstracts**: Machine learning algorithms are used to construct a mathematical model for a system based on training data. Such a model is capable of making highly accurate predictions without being explicitly programmed to do so. These techniques have a great many applications in all areas of the modern digital economy and artificial intelligence. More importantly, these methods are essential for a rapidly increasing number of safety-critical applications such as autonomous vehicles and intelligent defense systems. However, emerging adversarial learning attacks pose a serious security threat that greatly undermines further such systems. The latter are classified into four types, evasion (manipulating data to avoid detection), poisoning (injection malicious training samples to disrupt retraining), model stealing (extraction), and inference (leveraging over-generalization on training data). Understanding this type of attacks is a crucial first step for the development of effective countermeasures. The paper provides a detailed tutorial on the principles of adversarial machining learning, explains the different attack scenarios, and gives an in-depth insight into the state-of-art defense mechanisms against this rising threat .

摘要: 机器学习算法用于基于训练数据构建系统的数学模型。这样的模型能够做出高度精确的预测，而不需要明确地编程来这样做。这些技术在现代数字经济和人工智能的各个领域都有大量的应用。更重要的是，这些方法对于迅速增加的安全关键型应用(如自动驾驶汽车和智能防御系统)至关重要。然而，新出现的对抗性学习攻击构成了严重的安全威胁，极大地破坏了这样的系统。后者分为四种类型：逃避(操纵数据以避免检测)、中毒(注入恶意训练样本以中断再训练)、模型窃取(提取)和推理(利用训练数据的过度泛化)。了解这类攻击是制定有效对策的关键第一步。本文详细介绍了对抗性机器学习的原理，解释了不同的攻击场景，并深入了解了针对这一不断上升的威胁的最新防御机制。



## **19. Cyber-Physical Defense in the Quantum Era**

量子时代的网络物理防御 cs.CR

14 pages, 7 figures, 1 table, 4 boxes

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2202.10354v1)

**Authors**: Michel Barbeau, Joaquin Garcia-Alfaro

**Abstracts**: Networked-Control Systems (NCSs), a type of cyber-physical systems, consist of tightly integrated computing, communication and control technologies. While being very flexible environments, they are vulnerable to computing and networking attacks. Recent NCSs hacking incidents had major impact. They call for more research on cyber-physical security. Fears about the use of quantum computing to break current cryptosystems make matters worse. While the quantum threat motivated the creation of new disciplines to handle the issue, such as post-quantum cryptography, other fields have overlooked the existence of quantum-enabled adversaries. This is the case of cyber-physical defense research, a distinct but complementary discipline to cyber-physical protection. Cyber-physical defense refers to the capability to detect and react in response to cyber-physical attacks. Concretely, it involves the integration of mechanisms to identify adverse events and prepare response plans, during and after incidents occur. In this paper, we make the assumption that the eventually available quantum computer will provide an advantage to adversaries against defenders, unless they also adopt this technology. We envision the necessity for a paradigm shift, where an increase of adversarial resources because of quantum supremacy does not translate into higher likelihood of disruptions. Consistently with current system design practices in other areas, such as the use of artificial intelligence for the reinforcement of attack detection tools, we outline a vision for next generation cyber-physical defense layers leveraging ideas from quantum computing and machine learning. Through an example, we show that defenders of NCSs can learn and improve their strategies to anticipate and recover from attacks.

摘要: 网络控制系统(NCSs)是一种集计算、通信和控制技术于一体的网络物理系统。虽然它们是非常灵活的环境，但很容易受到计算和网络攻击。最近NCS的黑客事件产生了重大影响。他们呼吁对网络物理安全进行更多研究。对使用量子计算来破解现有密码系统的担忧使情况变得更糟。虽然量子威胁促使创建新的学科来处理这个问题，如后量子密码学，但其他领域忽略了量子对手的存在。这就是网络物理防御研究的情况，这是一门与网络物理保护截然不同但相辅相成的学科。网络物理防御是指检测并响应网络物理攻击的能力。具体地说，它涉及整合各种机制，以便在事件发生期间和之后识别不良事件并准备应对计划。在这篇文章中，我们假设最终可用的量子计算机将为对手对抗防御者提供优势，除非他们也采用这种技术。我们预见了范式转变的必要性，在这种情况下，由于量子优势而增加的对抗性资源并不会转化为更高的破坏可能性。与其他领域目前的系统设计实践一致，例如使用人工智能来加强攻击检测工具，我们利用量子计算和机器学习的想法勾勒出下一代网络物理防御层的愿景。通过一个实例，我们表明NCS的防御者可以学习和改进他们的策略，以预测攻击并从攻击中恢复。



## **20. Measurement-Device-Independent Quantum Secure Direct Communication with User Authentication**

具有用户认证的独立于测量设备的量子安全直接通信 quant-ph

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2202.10316v1)

**Authors**: Nayana Das, Goutam Paul

**Abstracts**: Quantum secure direct communication (QSDC) and deterministic secure quantum communication (DSQC) are two important branches of quantum cryptography, where one can transmit a secret message securely without encrypting it by a prior key. In the practical scenario, an adversary can apply detector-side-channel attacks to get some non-negligible amount of information about the secret message. Measurement-device-independent (MDI) quantum protocols can remove this kind of detector-side-channel attack, by introducing an untrusted third party (UTP), who performs all the measurements during the protocol with imperfect measurement devices. In this paper, we put forward the first MDI-QSDC protocol with user identity authentication, where both the sender and the receiver first check the authenticity of the other party and then exchange the secret message. Then we extend this to an MDI quantum dialogue (QD) protocol, where both the parties can send their respective secret messages after verifying the identity of the other party. Along with this, we also report the first MDI-DSQC protocol with user identity authentication. Theoretical analyses prove the security of our proposed protocols against common attacks.

摘要: 量子安全直接通信(QSDC)和确定性安全量子通信(DSQC)是量子密码学的两个重要分支。在实际场景中，攻击者可以应用检测器端信道攻击来获取有关秘密消息的一些不可忽略的信息。测量设备无关(MDI)量子协议可以通过引入一个不可信的第三方(UTP)来消除这种探测器侧信道攻击，该第三方使用不完善的测量设备执行协议中的所有测量。本文提出了第一个具有用户身份认证的MDI-QSDC协议，其中发送方和接收方都先检查对方的真实性，然后交换秘密消息。然后我们将其扩展到MDI量子对话(QD)协议，在该协议中，双方可以在验证对方的身份后发送各自的秘密消息。同时，我们还报道了第一个支持用户身份认证的MDI-DSQC协议。理论分析证明了我们提出的协议具有抗常见攻击的安全性。



## **21. HoneyModels: Machine Learning Honeypots**

HoneyModels：机器学习的蜜罐 cs.CR

Published in: MILCOM 2021 - 2021 IEEE Military Communications  Conference (MILCOM)

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2202.10309v1)

**Authors**: Ahmed Abdou, Ryan Sheatsley, Yohan Beugin, Tyler Shipp, Patrick McDaniel

**Abstracts**: Machine Learning is becoming a pivotal aspect of many systems today, offering newfound performance on classification and prediction tasks, but this rapid integration also comes with new unforeseen vulnerabilities. To harden these systems the ever-growing field of Adversarial Machine Learning has proposed new attack and defense mechanisms. However, a great asymmetry exists as these defensive methods can only provide security to certain models and lack scalability, computational efficiency, and practicality due to overly restrictive constraints. Moreover, newly introduced attacks can easily bypass defensive strategies by making subtle alterations. In this paper, we study an alternate approach inspired by honeypots to detect adversaries. Our approach yields learned models with an embedded watermark. When an adversary initiates an interaction with our model, attacks are encouraged to add this predetermined watermark stimulating detection of adversarial examples. We show that HoneyModels can reveal 69.5% of adversaries attempting to attack a Neural Network while preserving the original functionality of the model. HoneyModels offer an alternate direction to secure Machine Learning that slightly affects the accuracy while encouraging the creation of watermarked adversarial samples detectable by the HoneyModel but indistinguishable from others for the adversary.

摘要: 机器学习正在成为当今许多系统的一个关键方面，它在分类和预测任务上提供了新的性能，但这种快速集成也伴随着新的不可预见的漏洞。为了强化这些系统，不断发展的对抗性机器学习领域提出了新的攻防机制。然而，由于这些防御方法只能为某些模型提供安全性，并且由于过于严格的约束而缺乏可扩展性、计算效率和实用性，因此存在很大的不对称性。此外，新引入的攻击可以通过微妙的更改轻松绕过防御策略。在本文中，我们研究了一种受蜜罐启发的另一种检测对手的方法。我们的方法产生带有嵌入水印的学习模型。当敌方发起与我们的模型的交互时，鼓励攻击添加该预定水印来刺激对敌方示例的检测。结果表明，HoneyModels在保持模型原有功能的同时，可以发现69.5%的攻击者试图攻击神经网络。HoneyModel为确保机器学习的安全提供了另一种方向，这对准确性略有影响，同时鼓励创建HoneyModel可以检测到的带水印的敌意样本，但对于敌手来说无法与其他样本区分开来。



## **22. Hardware Obfuscation of Digital FIR Filters**

数字FIR滤波器的硬件混淆 cs.CR

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2202.10022v1)

**Authors**: Levent Aksoy, Alexander Hepp, Johanna Baehr, Samuel Pagliarini

**Abstracts**: A finite impulse response (FIR) filter is a ubiquitous block in digital signal processing applications. Its characteristics are determined by its coefficients, which are the intellectual property (IP) for its designer. However, in a hardware efficient realization, its coefficients become vulnerable to reverse engineering. This paper presents a filter design technique that can protect this IP, taking into account hardware complexity and ensuring that the filter behaves as specified only when a secret key is provided. To do so, coefficients are hidden among decoys, which are selected beyond possible values of coefficients using three alternative methods. As an attack scenario, an adversary at an untrusted foundry is considered. A reverse engineering technique is developed to find the chosen decoy selection method and explore the potential leakage of coefficients through decoys. An oracle-less attack is also used to find the secret key. Experimental results show that the proposed technique can lead to filter designs with competitive hardware complexity and higher resiliency to attacks with respect to previously proposed methods.

摘要: 有限脉冲响应(FIR)过滤是数字信号处理应用中普遍存在的一种挡路。它的特性是由它的系数决定的，这些系数是它的设计者的知识产权(IP)。然而，在硬件高效实现中，其系数容易受到逆向工程的影响。本文提出了一种过滤的设计技术，它可以保护这个IP，考虑到硬件的复杂性，并确保只有在提供密钥的情况下，过滤才能按照规定的方式运行。为此，系数隐藏在诱饵中，使用三种替代方法选择超出系数可能值的诱饵。作为攻击场景，考虑不可信铸造厂的对手。开发了一种逆向工程技术来寻找所选择的诱饵选择方法，并通过诱饵探测系数的潜在泄漏。也可以使用无预言机攻击来查找密钥。实验结果表明，与以往的过滤设计方法相比，该方法可以设计出硬件复杂度更高、抗攻击能力更强的好胜设计。



## **23. Learning to Attack with Fewer Pixels: A Probabilistic Post-hoc Framework for Refining Arbitrary Dense Adversarial Attacks**

学习用更少的像素进行攻击：一种精化任意密集对手攻击的概率后自组织框架 cs.CV

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2010.06131v2)

**Authors**: He Zhao, Thanh Nguyen, Trung Le, Paul Montague, Olivier De Vel, Tamas Abraham, Dinh Phung

**Abstracts**: Deep neural network image classifiers are reported to be susceptible to adversarial evasion attacks, which use carefully crafted images created to mislead a classifier. Many adversarial attacks belong to the category of dense attacks, which generate adversarial examples by perturbing all the pixels of a natural image. To generate sparse perturbations, sparse attacks have been recently developed, which are usually independent attacks derived by modifying a dense attack's algorithm with sparsity regularisations, resulting in reduced attack efficiency. In this paper, we aim to tackle this task from a different perspective. We select the most effective perturbations from the ones generated from a dense attack, based on the fact we find that a considerable amount of the perturbations on an image generated by dense attacks may contribute little to attacking a classifier. Accordingly, we propose a probabilistic post-hoc framework that refines given dense attacks by significantly reducing the number of perturbed pixels but keeping their attack power, trained with mutual information maximisation. Given an arbitrary dense attack, the proposed model enjoys appealing compatibility for making its adversarial images more realistic and less detectable with fewer perturbations. Moreover, our framework performs adversarial attacks much faster than existing sparse attacks.

摘要: 据报道，深度神经网络图像分类器容易受到敌意规避攻击，这些攻击使用精心制作的图像来误导分类器。许多对抗性攻击属于密集攻击的范畴，通过扰乱自然图像的所有像素来生成对抗性示例。为了产生稀疏扰动，最近发展了稀疏攻击，这些攻击通常是通过用稀疏正则化修改稠密攻击算法而得到的独立攻击，从而降低了攻击效率。在本文中，我们旨在从不同的角度来看待撞击这一任务。我们从密集攻击产生的扰动中选择最有效的扰动，因为我们发现密集攻击对图像产生的相当大的扰动对攻击分类器的贡献很小。因此，我们提出了一种概率后自组织框架，它通过显著减少扰动像素的数量，但保持它们的攻击能力，并用互信息最大化来训练，从而优化给定的密集攻击。在给定任意密集攻击的情况下，所提出的模型具有良好的兼容性，使其对抗性图像更逼真，且在较少扰动的情况下不易被检测到。此外，我们的框架执行对抗性攻击的速度比现有的稀疏攻击要快得多。



## **24. Transferring Adversarial Robustness Through Robust Representation Matching**

通过鲁棒表示匹配传递对抗鲁棒性 cs.LG

To appear at USENIX'22

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2202.09994v1)

**Authors**: Pratik Vaishnavi, Kevin Eykholt, Amir Rahmati

**Abstracts**: With the widespread use of machine learning, concerns over its security and reliability have become prevalent. As such, many have developed defenses to harden neural networks against adversarial examples, imperceptibly perturbed inputs that are reliably misclassified. Adversarial training in which adversarial examples are generated and used during training is one of the few known defenses able to reliably withstand such attacks against neural networks. However, adversarial training imposes a significant training overhead and scales poorly with model complexity and input dimension. In this paper, we propose Robust Representation Matching (RRM), a low-cost method to transfer the robustness of an adversarially trained model to a new model being trained for the same task irrespective of architectural differences. Inspired by student-teacher learning, our method introduces a novel training loss that encourages the student to learn the teacher's robust representations. Compared to prior works, RRM is superior with respect to both model performance and adversarial training time. On CIFAR-10, RRM trains a robust model $\sim 1.8\times$ faster than the state-of-the-art. Furthermore, RRM remains effective on higher-dimensional datasets. On Restricted-ImageNet, RRM trains a ResNet50 model $\sim 18\times$ faster than standard adversarial training.

摘要: 随着机器学习的广泛应用，人们普遍关注机器学习的安全性和可靠性。因此，许多人已经开发出防御措施，以加强神经网络对敌意例子的抵挡，这些例子是潜移默化的，输入被可靠地错误分类。对抗性训练，即在训练期间生成并使用对抗性例子，是为数不多的能够可靠地抵御针对神经网络的此类攻击的已知防御措施之一。然而，对抗性训练带来了巨大的训练开销，并且与模型复杂度和输入维度的比例关系不佳。在本文中，我们提出了鲁棒表示匹配(RRM)，这是一种低成本的方法，可以将敌对训练模型的鲁棒性转移到为同一任务训练的新模型，而不考虑体系结构的差异。受师生学习的启发，我们的方法引入了一种新颖的训练损失，鼓励学生学习教师的健壮表征。与前人的工作相比，RRM在模型性能和对抗性训练时间方面都具有优势。在CIFAR-10上，RRM训练的健壮模型$\sim\比最先进的模型快1.8倍。此外，RRM在高维数据集上仍然有效。在受限的ImageNet上，RRM训练的ResNet50型号$\sim比标准对手训练快18倍。



## **25. Overparametrization improves robustness against adversarial attacks: A replication study**

过度参数化提高对抗对手攻击的稳健性：一项重复研究 cs.LG

**SubmitDate**: 2022-02-20    [paper-pdf](http://arxiv.org/pdf/2202.09735v1)

**Authors**: Ali Borji

**Abstracts**: Overparametrization has become a de facto standard in machine learning. Despite numerous efforts, our understanding of how and where overparametrization helps model accuracy and robustness is still limited. To this end, here we conduct an empirical investigation to systemically study and replicate previous findings in this area, in particular the study by Madry et al. Together with this study, our findings support the "universal law of robustness" recently proposed by Bubeck et al. We argue that while critical for robust perception, overparametrization may not be enough to achieve full robustness and smarter architectures e.g. the ones implemented by the human visual cortex) seem inevitable.

摘要: 过度参数化已经成为机器学习中事实上的标准。尽管做了很多努力，我们对过度参数化如何以及在哪里有助于模型的准确性和健壮性的理解仍然有限。为此，我们在这里进行了实证调查，以系统地研究和复制这一领域的前人研究成果，特别是Madry等人的研究。结合这项研究，我们的发现支持了Bubeck等人最近提出的“稳健性普遍定律”。我们认为，虽然过度参数化对于鲁棒感知至关重要，但过度参数化可能不足以实现完全的鲁棒性和更智能的架构(例如，由人眼视皮层实现的架构)似乎是不可避免的。



## **26. Runtime-Assured, Real-Time Neural Control of Microgrids**

保证运行时间的微电网实时神经控制 eess.SY

**SubmitDate**: 2022-02-20    [paper-pdf](http://arxiv.org/pdf/2202.09710v1)

**Authors**: Amol Damare, Shouvik Roy, Scott A. Smolka, Scott D. Stoller

**Abstracts**: We present SimpleMG, a new, provably correct design methodology for runtime assurance of microgrids (MGs) with neural controllers. Our approach is centered around the Neural Simplex Architecture, which in turn is based on Sha et al.'s Simplex Control Architecture. Reinforcement Learning is used to synthesize high-performance neural controllers for MGs. Barrier Certificates are used to establish SimpleMG's runtime-assurance guarantees. We present a novel method to derive the condition for switching from the unverified neural controller to the verified-safe baseline controller, and we prove that the method is correct. We conduct an extensive experimental evaluation of SimpleMG using RTDS, a high-fidelity, real-time simulation environment for power systems, on a realistic model of a microgrid comprising three distributed energy resources (battery, photovoltaic, and diesel generator). Our experiments confirm that SimpleMG can be used to develop high-performance neural controllers for complex microgrids while assuring runtime safety, even in the presence of adversarial input attacks on the neural controller. Our experiments also demonstrate the benefits of online retraining of the neural controller while the baseline controller is in control

摘要: 我们提出了SimpleMG，这是一种新的、可以证明是正确的设计方法，用于带神经控制器的微电网(MG)的运行时保证。我们的方法是以神经单纯形体系结构为中心的，而神经单纯形体系结构又基于沙等人的单纯形控制体系结构。强化学习被用来综合高性能的磁控系统的神经控制器。屏障证书用于建立SimpleMG的运行时保证。我们提出了一种新的方法来推导从未经验证的神经控制器切换到验证安全的基线控制器的条件，并证明了该方法的正确性。在一个由三种分布式能源(电池、光伏和柴油发电机)组成的真实微电网模型上，我们使用一个高保真、实时的电力系统仿真环境RTDS对SimpleMG进行了广泛的实验评估。我们的实验证实，SimpleMG可以用来开发复杂微网格的高性能神经控制器，同时保证运行时的安全性，即使在神经控制器受到敌意输入攻击的情况下也是如此。我们的实验也证明了在基线控制器处于控制状态时在线重新训练神经控制器的好处。



## **27. Detection of Stealthy Adversaries for Networked Unmanned Aerial Vehicles**

网络化无人机隐身对手的检测 eess.SY

**SubmitDate**: 2022-02-19    [paper-pdf](http://arxiv.org/pdf/2202.09661v1)

**Authors**: Mohammad Bahrami, Hamidreza Jafarnejadsani

**Abstracts**: A network of unmanned aerial vehicles (UAVs) provides distributed coverage, reconfigurability, and maneuverability in performing complex cooperative tasks. However, it relies on wireless communications that can be susceptible to cyber adversaries and intrusions, disrupting the entire network's operation. This paper develops model-based centralized and decentralized observer techniques for detecting a class of stealthy intrusions, namely zero-dynamics and covert attacks, on networked UAVs in formation control settings. The centralized observer that runs in a control center leverages switching in the UAVs' communication topology for attack detection, and the decentralized observers, implemented onboard each UAV in the network, use the model of networked UAVs and locally available measurements. Experimental results are provided to show the effectiveness of the proposed detection schemes in different case studies.

摘要: 无人驾驶飞行器(UAV)网络在执行复杂的协作任务时提供了分布式覆盖、可重构性和机动性。然而，它依赖于无线通信，这可能会受到网络对手和入侵的影响，扰乱整个网络的运行。提出了一种基于模型的集中式和分散式观测器技术，用于检测编队控制环境下网络化无人机的一类隐身入侵，即零动态攻击和隐蔽攻击。在控制中心运行的集中式观察器利用无人机通信拓扑中的切换来进行攻击检测，而在网络中的每架无人机上实现的分散式观察器使用联网的无人机模型和本地可用的测量。实验结果表明，所提出的检测方案在不同的案例研究中是有效的。



## **28. Stochastic sparse adversarial attacks**

随机稀疏对抗性攻击 cs.LG

Final version published at the ICTAI 2021 conference with a best  student paper award. Codes are available through the link:  https://github.com/hhajri/stochastic-sparse-adv-attacks

**SubmitDate**: 2022-02-19    [paper-pdf](http://arxiv.org/pdf/2011.12423v4)

**Authors**: Manon Césaire, Lucas Schott, Hatem Hajri, Sylvain Lamprier, Patrick Gallinari

**Abstracts**: This paper introduces stochastic sparse adversarial attacks (SSAA), standing as simple, fast and purely noise-based targeted and untargeted attacks of neural network classifiers (NNC). SSAA offer new examples of sparse (or $L_0$) attacks for which only few methods have been proposed previously. These attacks are devised by exploiting a small-time expansion idea widely used for Markov processes. Experiments on small and large datasets (CIFAR-10 and ImageNet) illustrate several advantages of SSAA in comparison with the-state-of-the-art methods. For instance, in the untargeted case, our method called Voting Folded Gaussian Attack (VFGA) scales efficiently to ImageNet and achieves a significantly lower $L_0$ score than SparseFool (up to $\frac{2}{5}$) while being faster. Moreover, VFGA achieves better $L_0$ scores on ImageNet than Sparse-RS when both attacks are fully successful on a large number of samples.

摘要: 介绍了随机稀疏对抗攻击(SSAA)，即简单、快速、纯基于噪声的神经网络分类器(NNC)目标攻击和非目标攻击(NNC)。SSAA为稀疏(或$L_0$)攻击提供了新的例子，以前只有很少的方法被提出。这些攻击是通过利用马尔可夫过程中广泛使用的小时间扩展思想来设计的。在小型和大型数据集(CIFAR-10和ImageNet)上的实验表明，与最先进的方法相比，SSAA具有一些优势。例如，在无目标的情况下，我们的方法称为投票折叠高斯攻击(VFGA)，可以有效地扩展到ImageNet，并且获得比SparseFool(最高可达$\frac{2}{5}$)低得多的$L_0$分数，同时速度更快。此外，当两种攻击在大量样本上都完全成功时，VFGA在ImageNet上获得了比稀疏RS更好的$L_0$得分。



## **29. Internal Wasserstein Distance for Adversarial Attack and Defense**

对抗性攻防的瓦瑟斯坦内部距离 cs.LG

**SubmitDate**: 2022-02-19    [paper-pdf](http://arxiv.org/pdf/2103.07598v2)

**Authors**: Mingkui Tan, Shuhai Zhang, Jiezhang Cao, Jincheng Li, Yanwu Xu

**Abstracts**: Deep neural networks (DNNs) are known to be vulnerable to adversarial attacks that would trigger misclassification of DNNs but may be imperceptible to human perception. Adversarial defense has been important ways to improve the robustness of DNNs. Existing attack methods often construct adversarial examples relying on some metrics like the $\ell_p$ distance to perturb samples. However, these metrics can be insufficient to conduct adversarial attacks due to their limited perturbations. In this paper, we propose a new internal Wasserstein distance (IWD) to capture the semantic similarity of two samples, and thus it helps to obtain larger perturbations than currently used metrics such as the $\ell_p$ distance We then apply the internal Wasserstein distance to perform adversarial attack and defense. In particular, we develop a novel attack method relying on IWD to calculate the similarities between an image and its adversarial examples. In this way, we can generate diverse and semantically similar adversarial examples that are more difficult to defend by existing defense methods. Moreover, we devise a new defense method relying on IWD to learn robust models against unseen adversarial examples. We provide both thorough theoretical and empirical evidence to support our methods.

摘要: 深度神经网络(DNNs)很容易受到敌意攻击，这些攻击可能会导致DNN的错误分类，但可能无法被人类感知到。对抗性防御已经成为提高DNNs健壮性的重要途径。现有的攻击方法通常依赖于$\ell_p$距离等度量来构建敌意示例来扰动样本。然而，由于其有限的扰动，这些度量可能不足以进行对抗性攻击。本文提出了一种新的内部Wasserstein距离(IWD)来刻画两个样本之间的语义相似性，从而有助于获得比目前使用的$\\ell_p$距离等度量更大的扰动。然后利用内部Wasserstein距离进行对抗性攻击和防御。特别地，我们开发了一种新的攻击方法，该方法依赖于IWD来计算图像与其对手示例之间的相似度。这样，我们就可以生成不同的、语义相似的对抗性例子，而这些例子是现有防御方法更难防御的。此外，我们设计了一种新的防御方法，依靠IWD学习鲁棒模型来抵御看不见的对手例子。我们提供了充分的理论和经验证据来支持我们的方法。



## **30. Robust Reinforcement Learning as a Stackelberg Game via Adaptively-Regularized Adversarial Training**

基于自适应正则化对抗性训练的Stackelberg博弈鲁棒强化学习 cs.LG

**SubmitDate**: 2022-02-19    [paper-pdf](http://arxiv.org/pdf/2202.09514v1)

**Authors**: Peide Huang, Mengdi Xu, Fei Fang, Ding Zhao

**Abstracts**: Robust Reinforcement Learning (RL) focuses on improving performances under model errors or adversarial attacks, which facilitates the real-life deployment of RL agents. Robust Adversarial Reinforcement Learning (RARL) is one of the most popular frameworks for robust RL. However, most of the existing literature models RARL as a zero-sum simultaneous game with Nash equilibrium as the solution concept, which could overlook the sequential nature of RL deployments, produce overly conservative agents, and induce training instability. In this paper, we introduce a novel hierarchical formulation of robust RL - a general-sum Stackelberg game model called RRL-Stack - to formalize the sequential nature and provide extra flexibility for robust training. We develop the Stackelberg Policy Gradient algorithm to solve RRL-Stack, leveraging the Stackelberg learning dynamics by considering the adversary's response. Our method generates challenging yet solvable adversarial environments which benefit RL agents' robust learning. Our algorithm demonstrates better training stability and robustness against different testing conditions in the single-agent robotics control and multi-agent highway merging tasks.

摘要: 鲁棒强化学习(RL)侧重于提高在模型错误或敌意攻击下的性能，有利于RL Agent的实际部署。鲁棒对抗强化学习(RARL)是目前最流行的鲁棒对抗强化学习框架之一。然而，现有的文献大多将RARL建模为以纳什均衡为解概念的零和同时博弈，这可能会忽略RL部署的序贯性质，产生过于保守的代理，并导致训练不稳定。在本文中，我们引入了一种新的鲁棒RL的分层表示-称为RRL-Stack的一般和Stackelberg博弈模型-以形式化顺序性质，并为鲁棒训练提供额外的灵活性。我们开发了Stackelberg策略梯度算法来求解RRL-Stack，通过考虑对手的响应来利用Stackelberg学习动态。我们的方法产生了具有挑战性但可解决的对抗环境，这有利于RL Agent的鲁棒学习。在单智能体机器人控制和多智能体公路合并任务中，我们的算法对不同的测试条件表现出较好的训练稳定性和鲁棒性。



## **31. Attacks, Defenses, And Tools: A Framework To Facilitate Robust AI/ML Systems**

攻击、防御和工具：促进健壮AI/ML系统的框架 cs.CR

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2202.09465v1)

**Authors**: Mohamad Fazelnia, Igor Khokhlov, Mehdi Mirakhorli

**Abstracts**: Software systems are increasingly relying on Artificial Intelligence (AI) and Machine Learning (ML) components. The emerging popularity of AI techniques in various application domains attracts malicious actors and adversaries. Therefore, the developers of AI-enabled software systems need to take into account various novel cyber-attacks and vulnerabilities that these systems may be susceptible to. This paper presents a framework to characterize attacks and weaknesses associated with AI-enabled systems and provide mitigation techniques and defense strategies. This framework aims to support software designers in taking proactive measures in developing AI-enabled software, understanding the attack surface of such systems, and developing products that are resilient to various emerging attacks associated with ML. The developed framework covers a broad spectrum of attacks, mitigation techniques, and defensive and offensive tools. In this paper, we demonstrate the framework architecture and its major components, describe their attributes, and discuss the long-term goals of this research.

摘要: 软件系统越来越依赖人工智能(AI)和机器学习(ML)组件。人工智能技术在各个应用领域的新兴普及吸引了恶意行为者和对手。因此，人工智能软件系统的开发人员需要考虑到这些系统可能容易受到的各种新型网络攻击和漏洞。本文提出了一个框架来描述与人工智能系统相关的攻击和弱点，并提供缓解技术和防御策略。该框架旨在支持软件设计人员在开发支持人工智能的软件时采取主动措施，了解此类系统的攻击面，并开发对与ML相关的各种新兴攻击具有弹性的产品。开发的框架涵盖了广泛的攻击、缓解技术以及防御和进攻工具。在本文中，我们展示了框架体系结构及其主要组件，描述了它们的属性，并讨论了本研究的长期目标。



## **32. Black-box Node Injection Attack for Graph Neural Networks**

图神经网络的黑盒节点注入攻击 cs.LG

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2202.09389v1)

**Authors**: Mingxuan Ju, Yujie Fan, Yanfang Ye, Liang Zhao

**Abstracts**: Graph Neural Networks (GNNs) have drawn significant attentions over the years and been broadly applied to vital fields that require high security standard such as product recommendation and traffic forecasting. Under such scenarios, exploiting GNN's vulnerabilities and further downgrade its classification performance become highly incentive for adversaries. Previous attackers mainly focus on structural perturbations of existing graphs. Although they deliver promising results, the actual implementation needs capability of manipulating the graph connectivity, which is impractical in some circumstances. In this work, we study the possibility of injecting nodes to evade the victim GNN model, and unlike previous related works with white-box setting, we significantly restrict the amount of accessible knowledge and explore the black-box setting. Specifically, we model the node injection attack as a Markov decision process and propose GA2C, a graph reinforcement learning framework in the fashion of advantage actor critic, to generate realistic features for injected nodes and seamlessly merge them into the original graph following the same topology characteristics. Through our extensive experiments on multiple acknowledged benchmark datasets, we demonstrate the superior performance of our proposed GA2C over existing state-of-the-art methods. The data and source code are publicly accessible at: https://github.com/jumxglhf/GA2C.

摘要: 多年来，图神经网络(GNNs)引起了人们的广泛关注，并被广泛应用于产品推荐、流量预测等对安全性要求较高的重要领域。在这种情况下，利用GNN的漏洞并进一步降低其分类性能成为对手的极大诱因。以往的攻击者主要集中在现有图的结构扰动上。虽然它们提供了有希望的结果，但实际实现需要能够操作图形连接，这在某些情况下是不切实际的。在这项工作中，我们研究了注入节点来逃避受害者GNN模型的可能性，与以往白盒设置的相关工作不同，我们显著限制了可访问的知识量，并探索了黑盒设置。具体地说，我们将节点注入攻击建模为马尔可夫决策过程，并提出了一种优势角色批判式的图强化学习框架GA2C，用于生成注入节点的真实特征，并按照相同的拓扑特征将其无缝合并到原始图中。通过我们在多个公认的基准数据集上的广泛实验，我们证明了我们提出的GA2C比现有最先进的方法具有更好的性能。数据和源代码可在以下网址公开访问：https://github.com/jumxglhf/GA2C.



## **33. Synthetic Disinformation Attacks on Automated Fact Verification Systems**

对自动事实验证系统的合成虚假信息攻击 cs.CL

AAAI 2022

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2202.09381v1)

**Authors**: Yibing Du, Antoine Bosselut, Christopher D. Manning

**Abstracts**: Automated fact-checking is a needed technology to curtail the spread of online misinformation. One current framework for such solutions proposes to verify claims by retrieving supporting or refuting evidence from related textual sources. However, the realistic use cases for fact-checkers will require verifying claims against evidence sources that could be affected by the same misinformation. Furthermore, the development of modern NLP tools that can produce coherent, fabricated content would allow malicious actors to systematically generate adversarial disinformation for fact-checkers.   In this work, we explore the sensitivity of automated fact-checkers to synthetic adversarial evidence in two simulated settings: AdversarialAddition, where we fabricate documents and add them to the evidence repository available to the fact-checking system, and AdversarialModification, where existing evidence source documents in the repository are automatically altered. Our study across multiple models on three benchmarks demonstrates that these systems suffer significant performance drops against these attacks. Finally, we discuss the growing threat of modern NLG systems as generators of disinformation in the context of the challenges they pose to automated fact-checkers.

摘要: 自动事实核查是遏制在线错误信息传播所必需的技术。目前这类解决方案的一个框架建议通过从相关文本来源检索、支持或驳斥证据来核实主张。然而，事实核查人员的现实用例将需要对照可能受到相同错误信息影响的证据来源来验证声明。此外，能够产生连贯的、捏造的内容的现代NLP工具的开发将允许恶意行为者系统地为事实核查人员生成对抗性的虚假信息。在这项工作中，我们在两个模拟设置中探索了自动事实检查器对合成敌对证据的敏感性：AdversarialAddition，我们伪造文档并将它们添加到事实检查系统可用的证据储存库；AdversarialMoentation，其中储存库中的现有证据源文档被自动更改。我们在三个基准测试的多个模型上的研究表明，这些系统在抵御这些攻击时性能显著下降。最后，我们讨论了现代NLG系统作为虚假信息生成器的日益增长的威胁，在它们对自动事实核查人员构成挑战的背景下。



## **34. Exploring Adversarially Robust Training for Unsupervised Domain Adaptation**

无监督领域自适应的对抗性鲁棒训练探索 cs.CV

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2202.09300v1)

**Authors**: Shao-Yuan Lo, Vishal M. Patel

**Abstracts**: Unsupervised Domain Adaptation (UDA) methods aim to transfer knowledge from a labeled source domain to an unlabeled target domain. UDA has been extensively studied in the computer vision literature. Deep networks have been shown to be vulnerable to adversarial attacks. However, very little focus is devoted to improving the adversarial robustness of deep UDA models, causing serious concerns about model reliability. Adversarial Training (AT) has been considered to be the most successful adversarial defense approach. Nevertheless, conventional AT requires ground-truth labels to generate adversarial examples and train models, which limits its effectiveness in the unlabeled target domain. In this paper, we aim to explore AT to robustify UDA models: How to enhance the unlabeled data robustness via AT while learning domain-invariant features for UDA? To answer this, we provide a systematic study into multiple AT variants that potentially apply to UDA. Moreover, we propose a novel Adversarially Robust Training method for UDA accordingly, referred to as ARTUDA. Extensive experiments on multiple attacks and benchmarks show that ARTUDA consistently improves the adversarial robustness of UDA models.

摘要: 无监督域自适应(UDA)方法旨在将知识从有标签的源域转移到无标签的目标域。UDA在计算机视觉文献中得到了广泛的研究。深层网络已被证明容易受到敌意攻击。然而，很少有人致力于提高深度UDA模型的对抗健壮性，这引起了人们对模型可靠性的严重关注。对抗性训练(AT)被认为是最成功的对抗性防御方法。然而，传统的自动测试需要地面事实标签来生成对抗性示例和训练模型，这限制了其在未标记的目标领域的有效性。在本文中，我们的目标是探索AT对UDA模型的鲁棒性：如何在学习UDA的域不变性特征的同时，通过AT增强未标记数据的健壮性？为了回答这个问题，我们对可能适用于UDA的多个AT变体进行了系统研究。此外，我们还针对UDA提出了一种新颖的对抗性鲁棒训练方法，称为ARTUDA。在多个攻击和基准测试上的大量实验表明，ARTUDA一致地提高了UDA模型的对抗健壮性。



## **35. Resurrecting Trust in Facial Recognition: Mitigating Backdoor Attacks in Face Recognition to Prevent Potential Privacy Breaches**

恢复面部识别中的信任：减轻面部识别中的后门攻击，以防止潜在的隐私泄露 cs.CV

15 pages

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2202.10320v1)

**Authors**: Reena Zelenkova, Jack Swallow, M. A. P. Chamikara, Dongxi Liu, Mohan Baruwal Chhetri, Seyit Camtepe, Marthie Grobler, Mahathir Almashor

**Abstracts**: Biometric data, such as face images, are often associated with sensitive information (e.g medical, financial, personal government records). Hence, a data breach in a system storing such information can have devastating consequences. Deep learning is widely utilized for face recognition (FR); however, such models are vulnerable to backdoor attacks executed by malicious parties. Backdoor attacks cause a model to misclassify a particular class as a target class during recognition. This vulnerability can allow adversaries to gain access to highly sensitive data protected by biometric authentication measures or allow the malicious party to masquerade as an individual with higher system permissions. Such breaches pose a serious privacy threat. Previous methods integrate noise addition mechanisms into face recognition models to mitigate this issue and improve the robustness of classification against backdoor attacks. However, this can drastically affect model accuracy. We propose a novel and generalizable approach (named BA-BAM: Biometric Authentication - Backdoor Attack Mitigation), that aims to prevent backdoor attacks on face authentication deep learning models through transfer learning and selective image perturbation. The empirical evidence shows that BA-BAM is highly robust and incurs a maximal accuracy drop of 2.4%, while reducing the attack success rate to a maximum of 20%. Comparisons with existing approaches show that BA-BAM provides a more practical backdoor mitigation approach for face recognition.

摘要: 生物特征数据(如面部图像)通常与敏感信息(如医疗、金融、个人政府记录)相关联。因此，存储此类信息的系统中的数据泄露可能会造成毁灭性的后果。深度学习被广泛用于人脸识别(FR)；然而，此类模型容易受到恶意方执行的后门攻击。后门攻击会导致模型在识别过程中将特定类错误分类为目标类。此漏洞可让攻击者访问受生物特征验证措施保护的高度敏感数据，或允许恶意方伪装成具有更高系统权限的个人。这类侵犯隐私的行为构成了严重的隐私威胁。以往的方法将噪声添加机制集成到人脸识别模型中，以缓解这一问题，并提高分类对后门攻击的鲁棒性。但是，这可能会极大地影响模型精度。我们提出了一种新颖的、可推广的方法(BA-BAM：Biometry Authentication-Backdoor Attack Mitigation)，旨在通过迁移学习和选择性图像扰动来防止对人脸认证深度学习模型的后门攻击。实验结果表明，BA-BAM算法具有很强的鲁棒性，最大准确率下降2.4%，而攻击成功率最高可达20%。与现有方法的比较表明，BA-BAM为人脸识别提供了一种更实用的后门缓解方法。



## **36. Critical Checkpoints for Evaluating Defence Models Against Adversarial Attack and Robustness**

评估防御模型对抗攻击和健壮性的关键检查点 cs.CR

16 pages, 8 figures

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2202.09039v1)

**Authors**: Kanak Tekwani, Manojkumar Parmar

**Abstracts**: From past couple of years there is a cycle of researchers proposing a defence model for adversaries in machine learning which is arguably defensible to most of the existing attacks in restricted condition (they evaluate on some bounded inputs or datasets). And then shortly another set of researcher finding the vulnerabilities in that defence model and breaking it by proposing a stronger attack model. Some common flaws are been noticed in the past defence models that were broken in very short time. Defence models being broken so easily is a point of concern as decision of many crucial activities are taken with the help of machine learning models. So there is an utter need of some defence checkpoints that any researcher should keep in mind while evaluating the soundness of technique and declaring it to be decent defence technique. In this paper, we have suggested few checkpoints that should be taken into consideration while building and evaluating the soundness of defence models. All these points are recommended after observing why some past defence models failed and how some model remained adamant and proved their soundness against some of the very strong attacks.

摘要: 在过去的几年里，研究人员在机器学习中提出了一个针对对手的防御模型，该模型可以在有限条件下防御大多数现有的攻击(他们在一些有界的输入或数据集上进行评估)。不久，另一组研究人员发现了该防御模型中的漏洞，并提出了一种更强大的攻击模型来打破它。过去的防御模式在很短的时间内就被打破了，人们注意到了一些常见的缺陷。防御模型如此容易被打破是一个令人担忧的问题，因为许多关键活动的决策都是在机器学习模型的帮助下做出的。因此，非常需要一些防御检查点，任何研究者在评估技术的可靠性并宣布它是一种像样的防御技术时，都应该牢记这一点。在本文中，我们提出了在建立和评估防御模型的可靠性时应考虑的几个检查点。所有这些观点都是在观察了过去的一些防御模型失败的原因，以及一些模型是如何保持顽固的，并证明了它们在一些非常强大的攻击下是健全的之后推荐的。



## **37. Debiasing Backdoor Attack: A Benign Application of Backdoor Attack in Eliminating Data Bias**

去偏向后门攻击：后门攻击在消除数据偏差中的良性应用 cs.CR

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2202.10582v1)

**Authors**: Shangxi Wu, Qiuyang He, Yi Zhang, Jitao Sang

**Abstracts**: Backdoor attack is a new AI security risk that has emerged in recent years. Drawing on the previous research of adversarial attack, we argue that the backdoor attack has the potential to tap into the model learning process and improve model performance. Based on Clean Accuracy Drop (CAD) in backdoor attack, we found that CAD came out of the effect of pseudo-deletion of data. We provided a preliminary explanation of this phenomenon from the perspective of model classification boundaries and observed that this pseudo-deletion had advantages over direct deletion in the data debiasing problem. Based on the above findings, we proposed Debiasing Backdoor Attack (DBA). It achieves SOTA in the debiasing task and has a broader application scenario than undersampling.

摘要: 后门攻击是近年来出现的一种新的AI安全风险。在借鉴前人对抗性攻击研究的基础上，我们认为后门攻击具有挖掘模型学习过程、提高模型性能的潜力。基于后门攻击中的清洁精度下降(CAD)，我们发现CAD摆脱了数据假删除的影响。我们从模型分类边界的角度对这一现象进行了初步的解释，并观察到这种伪删除在数据去偏问题上比直接删除具有优势。基于上述发现，我们提出了去偏向后门攻击(DBA)。它在去偏任务中实现了SOTA，比欠采样具有更广泛的应用场景。



## **38. Explaining Adversarial Vulnerability with a Data Sparsity Hypothesis**

用数据稀疏性假说解释对抗性脆弱性 cs.AI

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2103.00778v3)

**Authors**: Mahsa Paknezhad, Cuong Phuc Ngo, Amadeus Aristo Winarto, Alistair Cheong, Chuen Yang Beh, Jiayang Wu, Hwee Kuan Lee

**Abstracts**: Despite many proposed algorithms to provide robustness to deep learning (DL) models, DL models remain susceptible to adversarial attacks. We hypothesize that the adversarial vulnerability of DL models stems from two factors. The first factor is data sparsity which is that in the high dimensional input data space, there exist large regions outside the support of the data distribution. The second factor is the existence of many redundant parameters in the DL models. Owing to these factors, different models are able to come up with different decision boundaries with comparably high prediction accuracy. The appearance of the decision boundaries in the space outside the support of the data distribution does not affect the prediction accuracy of the model. However, it makes an important difference in the adversarial robustness of the model. We hypothesize that the ideal decision boundary is as far as possible from the support of the data distribution. In this paper, we develop a training framework to observe if DL models are able to learn such a decision boundary spanning the space around the class distributions further from the data points themselves. Semi-supervised learning was deployed during training by leveraging unlabeled data generated in the space outside the support of the data distribution. We measured adversarial robustness of the models trained using this training framework against well-known adversarial attacks and by using robustness metrics. We found that models trained using our framework, as well as other regularization methods and adversarial training support our hypothesis of data sparsity and that models trained with these methods learn to have decision boundaries more similar to the aforementioned ideal decision boundary. The code for our training framework is available at https://github.com/MahsaPaknezhad/AdversariallyRobustTraining.

摘要: 尽管提出了许多算法来提供深度学习(DL)模型的鲁棒性，但是DL模型仍然容易受到敌意攻击。我们假设DL模型的对抗脆弱性源于两个因素。第一个因素是数据稀疏性，即在高维输入数据空间中，存在数据分布支持之外的大区域。第二个因素是DL模型中存在许多冗余参数。由于这些因素的影响，不同的模型能够给出不同的决策边界，具有较高的预测精度。在数据分布支持度之外的空间出现决策边界并不影响模型的预测精度。然而，它在模型的对抗性鲁棒性方面有很大的不同。我们假设理想的决策边界尽可能远离数据分布的支持。在本文中，我们开发了一个训练框架来观察DL模型是否能够从数据点本身进一步学习跨越类分布周围空间的决策边界。通过利用在数据分布支持之外的空间中生成的未标记数据，在训练期间部署半监督学习。我们通过使用健壮性度量来衡量使用该训练框架训练的模型对众所周知的敌意攻击的敌意稳健性。我们发现，使用我们的框架训练的模型，以及其他正则化方法和对抗性训练，都支持我们的数据稀疏性假设，并且用这些方法训练的模型学习的决策边界更类似于前面提到的理想决策边界。我们培训框架的代码可以在https://github.com/MahsaPaknezhad/AdversariallyRobustTraining.上找到



## **39. Amicable examples for informed source separation**

知情信源分离的友好示例 cs.SD

Accepted to ICASSP 2022

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2110.05059v2)

**Authors**: Naoya Takahashi, Yuki Mitsufuji

**Abstracts**: This paper deals with the problem of informed source separation (ISS), where the sources are accessible during the so-called \textit{encoding} stage. Previous works computed side-information during the encoding stage and source separation models were designed to utilize the side-information to improve the separation performance. In contrast, in this work, we improve the performance of a pretrained separation model that does not use any side-information. To this end, we propose to adopt an adversarial attack for the opposite purpose, i.e., rather than computing the perturbation to degrade the separation, we compute an imperceptible perturbation called amicable noise to improve the separation. Experimental results show that the proposed approach selectively improves the performance of the targeted separation model by 2.23 dB on average and is robust to signal compression. Moreover, we propose multi-model multi-purpose learning that control the effect of the perturbation on different models individually.

摘要: 本文研究信息源分离(ISS)问题，即信息源在所谓的\textit{编码}阶段是可访问的。以前的工作是在编码阶段计算边信息，并设计了源分离模型来利用边信息来提高分离性能。相反，在这项工作中，我们改进了不使用任何边信息的预训练分离模型的性能。为此，我们建议采取相反目的的对抗性攻击，即，我们不计算扰动来降低分离度，而是计算一种称为友好噪声的不可察觉的扰动来改善分离度。实验结果表明，该方法选择性地将目标分离模型的性能平均提高了2.23dB，并且对信号压缩具有较强的鲁棒性。此外，我们还提出了多模型多目标学习，分别控制扰动对不同模型的影响。



## **40. Morphence: Moving Target Defense Against Adversarial Examples**

Morphence：针对敌方的移动目标防御示例 cs.LG

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2108.13952v4)

**Authors**: Abderrahmen Amich, Birhanu Eshete

**Abstracts**: Robustness to adversarial examples of machine learning models remains an open topic of research. Attacks often succeed by repeatedly probing a fixed target model with adversarial examples purposely crafted to fool it. In this paper, we introduce Morphence, an approach that shifts the defense landscape by making a model a moving target against adversarial examples. By regularly moving the decision function of a model, Morphence makes it significantly challenging for repeated or correlated attacks to succeed. Morphence deploys a pool of models generated from a base model in a manner that introduces sufficient randomness when it responds to prediction queries. To ensure repeated or correlated attacks fail, the deployed pool of models automatically expires after a query budget is reached and the model pool is seamlessly replaced by a new model pool generated in advance. We evaluate Morphence on two benchmark image classification datasets (MNIST and CIFAR10) against five reference attacks (2 white-box and 3 black-box). In all cases, Morphence consistently outperforms the thus-far effective defense, adversarial training, even in the face of strong white-box attacks, while preserving accuracy on clean data.

摘要: 对机器学习模型的对抗性示例的鲁棒性仍然是一个开放的研究课题。攻击通常通过反复探测固定的目标模型而得逞，其中带有故意设计的敌意示例来愚弄它。在本文中，我们介绍了Morphence，一种通过使模型成为移动目标来对抗对手示例来改变防御格局的方法。通过定期移动模型的决策函数，Morphence使重复或相关攻击的成功变得极具挑战性。Morphence以在响应预测查询时引入足够的随机性的方式部署从基础模型生成的模型池。为确保重复或相关攻击失败，部署的模型池在达到查询预算后自动过期，并由预先生成的新模型池无缝替换。我们在两个基准图像分类数据集(MNIST和CIFAR10)上测试了Morphence在5个参考攻击(2个白盒和3个黑盒)下的性能。在所有情况下，Morphence的表现都始终如一地优于迄今有效的防御、对抗性训练，即使面对强大的白盒攻击，也能保持干净数据的准确性。



## **41. What Doesn't Kill You Makes You Robust(er): How to Adversarially Train against Data Poisoning**

什么不会杀死你，让你变得健壮(呃)：如何对抗数据中毒 cs.LG

25 pages, 15 figures

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2102.13624v2)

**Authors**: Jonas Geiping, Liam Fowl, Gowthami Somepalli, Micah Goldblum, Michael Moeller, Tom Goldstein

**Abstracts**: Data poisoning is a threat model in which a malicious actor tampers with training data to manipulate outcomes at inference time. A variety of defenses against this threat model have been proposed, but each suffers from at least one of the following flaws: they are easily overcome by adaptive attacks, they severely reduce testing performance, or they cannot generalize to diverse data poisoning threat models. Adversarial training, and its variants, are currently considered the only empirically strong defense against (inference-time) adversarial attacks. In this work, we extend the adversarial training framework to defend against (training-time) data poisoning, including targeted and backdoor attacks. Our method desensitizes networks to the effects of such attacks by creating poisons during training and injecting them into training batches. We show that this defense withstands adaptive attacks, generalizes to diverse threat models, and incurs a better performance trade-off than previous defenses such as DP-SGD or (evasion) adversarial training.

摘要: 数据中毒是一种威胁模型，在该模型中，恶意行为者篡改训练数据以在推断时操纵结果。针对此威胁模型提出了多种防御方案，但每种方案都至少存在以下缺陷之一：它们很容易被自适应攻击克服，严重降低了测试性能，或者不能推广到不同的数据中毒威胁模型。对抗性训练及其变体，目前被认为是对抗(推理时间)对抗性攻击的唯一经验性强防御。在这项工作中，我们扩展了对抗性训练框架来防御(训练时)数据中毒，包括有针对性的攻击和后门攻击。我们的方法通过在训练期间制造毒药并将它们注入训练批次来使网络对此类攻击的影响变得不敏感。我们表明，这种防御可以抵抗适应性攻击，适用于不同的威胁模型，并且比以前的防御(如DP-SGD或(回避)对抗性训练)具有更好的性能权衡。



## **42. Developing Imperceptible Adversarial Patches to Camouflage Military Assets From Computer Vision Enabled Technologies**

利用计算机视觉技术开发隐形对抗性补丁来伪装军事资产 cs.CV

8 pages, 4 figures, 4 tables, submitted to WCCI 2022

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2202.08892v1)

**Authors**: Christopher Wise, Jo Plested

**Abstracts**: Convolutional neural networks (CNNs) have demonstrated rapid progress and a high level of success in object detection. However, recent evidence has highlighted their vulnerability to adversarial attacks. These attacks are calculated image perturbations or adversarial patches that result in object misclassification or detection suppression. Traditional camouflage methods are impractical when applied to disguise aircraft and other large mobile assets from autonomous detection in intelligence, surveillance and reconnaissance technologies and fifth generation missiles. In this paper we present a unique method that produces imperceptible patches capable of camouflaging large military assets from computer vision-enabled technologies. We developed these patches by maximising object detection loss whilst limiting the patch's colour perceptibility. This work also aims to further the understanding of adversarial examples and their effects on object detection algorithms.

摘要: 卷积神经网络(CNNs)在目标检测方面取得了快速的进展和很高的成功率。然而，最近的证据突显了它们在对抗性攻击中的脆弱性。这些攻击是经过计算的图像扰动或导致对象误分类或检测抑制的对抗性补丁。传统的伪装方法用于伪装飞机和其他大型机动资产，使其免受情报、监视和侦察技术以及第五代导弹的自主探测，是不切实际的。在这篇文章中，我们提出了一种独特的方法，它可以从计算机视觉技术中产生能够伪装大型军事资产的隐形补丁。我们开发了这些补丁，通过最大化目标检测损失，同时限制补丁的颜色敏感度。这项工作还旨在进一步理解对抗性例子及其对目标检测算法的影响。



## **43. Alexa versus Alexa: Controlling Smart Speakers by Self-Issuing Voice Commands**

Alexa与Alexa：通过自行发出语音命令控制智能扬声器 cs.CR

15 pages, 5 figures, published in Proceedings of the 2022 ACM Asia  Conference on Computer and Communications Security (ASIA CCS '22)

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2202.08619v1)

**Authors**: Sergio Esposito, Daniele Sgandurra, Giampaolo Bella

**Abstracts**: We present Alexa versus Alexa (AvA), a novel attack that leverages audio files containing voice commands and audio reproduction methods in an offensive fashion, to gain control of Amazon Echo devices for a prolonged amount of time. AvA leverages the fact that Alexa running on an Echo device correctly interprets voice commands originated from audio files even when they are played by the device itself -- i.e., it leverages a command self-issue vulnerability. Hence, AvA removes the necessity of having a rogue speaker in proximity of the victim's Echo, a constraint that many attacks share. With AvA, an attacker can self-issue any permissible command to Echo, controlling it on behalf of the legitimate user. We have verified that, via AvA, attackers can control smart appliances within the household, buy unwanted items, tamper linked calendars and eavesdrop on the user. We also discovered two additional Echo vulnerabilities, which we call Full Volume and Break Tag Chain. The Full Volume increases the self-issue command recognition rate, by doubling it on average, hence allowing attackers to perform additional self-issue commands. Break Tag Chain increases the time a skill can run without user interaction, from eight seconds to more than one hour, hence enabling attackers to setup realistic social engineering scenarios. By exploiting these vulnerabilities, the adversary can self-issue commands that are correctly executed 99% of the times and can keep control of the device for a prolonged amount of time. We reported these vulnerabilities to Amazon via their vulnerability research program, who rated them with a Medium severity score. Finally, to assess limitations of AvA on a larger scale, we provide the results of a survey performed on a study group of 18 users, and we show that most of the limitations against AvA are hardly used in practice.

摘要: 我们提出了Alexa vs.Alexa(AVA)，这是一种新颖的攻击，它以攻击性的方式利用包含语音命令和音频复制方法的音频文件来获得对Amazon Echo设备的长时间控制。AVA利用在Echo设备上运行的Alexa能够正确解释源自音频文件的语音命令这一事实，即使音频文件是由设备本身播放的，即它利用了命令自发布漏洞。因此，AVA消除了在受害者的回声附近设置流氓扬声器的必要性，这是许多攻击都有的限制。使用AVA，攻击者可以自行向Echo发出任何允许的命令，并代表合法用户控制它。我们已经证实，通过AVA，攻击者可以控制家庭内的智能家电，购买不需要的物品，篡改链接的日历，并窃听用户。我们还发现了另外两个Echo漏洞，我们称之为Full Volume和Break Tag Chain。全音量通过将其平均翻倍来提高自发布命令识别率，从而允许攻击者执行额外的自发布命令。中断标签链增加了技能在没有用户交互的情况下可以运行的时间，从8秒增加到1小时以上，从而使攻击者能够设置现实的社会工程场景。通过利用这些漏洞，攻击者可以自行发出99%的正确执行次数的命令，并可以在较长时间内保持对设备的控制。我们通过他们的漏洞研究计划向亚马逊报告了这些漏洞，亚马逊对它们进行了中等严重程度的评级。最后，为了在更大的范围内评估AVA的局限性，我们提供了对18名用户进行的一项调查的结果，我们发现大多数针对AVA的限制在实践中几乎没有使用。



## **44. Improving Robustness of Deep Reinforcement Learning Agents: Environment Attack based on the Critic Network**

提高深度强化学习代理的健壮性：基于批判网络的环境攻击 cs.LG

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2104.03154v2)

**Authors**: Lucas Schott, Hatem Hajri, Sylvain Lamprier

**Abstracts**: To improve policy robustness of deep reinforcement learning agents, a line of recent works focus on producing disturbances of the environment. Existing approaches of the literature to generate meaningful disturbances of the environment are adversarial reinforcement learning methods. These methods set the problem as a two-player game between the protagonist agent, which learns to perform a task in an environment, and the adversary agent, which learns to disturb the protagonist via modifications of the considered environment. Both protagonist and adversary are trained with deep reinforcement learning algorithms. Alternatively, we propose in this paper to build on gradient-based adversarial attacks, usually used for classification tasks for instance, that we apply on the critic network of the protagonist to identify efficient disturbances of the environment. Rather than learning an attacker policy, which usually reveals as very complex and unstable, we leverage the knowledge of the critic network of the protagonist, to dynamically complexify the task at each step of the learning process. We show that our method, while being faster and lighter, leads to significantly better improvements in policy robustness than existing methods of the literature.

摘要: 为了提高深度强化学习代理的策略鲁棒性，最近的一系列工作集中在产生环境扰动上。现有文献中产生有意义的环境扰动的方法是对抗性强化学习方法。这些方法将问题设置为主角Agent和对手Agent之间的两人博弈，前者学习在环境中执行任务，后者学习通过修改所考虑的环境来干扰主角。利用深度强化学习算法对主角和对手进行训练。或者，我们在本文中建议建立基于梯度的对抗性攻击，通常用于分类任务，例如，我们应用于主人公的批评网络来识别环境的有效干扰。我们不学习通常表现为非常复杂和不稳定的攻击者策略，而是利用主人公批评网络的知识，在学习过程的每一步动态地使任务复杂化。我们表明，虽然我们的方法更快、更轻，但与现有的文献方法相比，我们的方法在政策稳健性方面的改善要明显更好。



## **45. GasHis-Transformer: A Multi-scale Visual Transformer Approach for Gastric Histopathology Image Classification**

GasHis-Transformer：一种用于胃组织病理图像分类的多尺度视觉变换方法 cs.CV

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2104.14528v6)

**Authors**: Haoyuan Chen, Chen Li, Ge Wang, Xiaoyan Li, Md Rahaman, Hongzan Sun, Weiming Hu, Yixin Li, Wanli Liu, Changhao Sun, Shiliang Ai, Marcin Grzegorzek

**Abstracts**: Existing deep learning methods for diagnosis of gastric cancer commonly use convolutional neural network. Recently, the Visual Transformer has attracted great attention because of its performance and efficiency, but its applications are mostly in the field of computer vision. In this paper, a multi-scale visual transformer model, referred to as GasHis-Transformer, is proposed for Gastric Histopathological Image Classification (GHIC), which enables the automatic classification of microscopic gastric images into abnormal and normal cases. The GasHis-Transformer model consists of two key modules: A global information module and a local information module to extract histopathological features effectively. In our experiments, a public hematoxylin and eosin (H&E) stained gastric histopathological dataset with 280 abnormal and normal images are divided into training, validation and test sets by a ratio of 1 : 1 : 2. The GasHis-Transformer model is applied to estimate precision, recall, F1-score and accuracy on the test set of gastric histopathological dataset as 98.0%, 100.0%, 96.0% and 98.0%, respectively. Furthermore, a critical study is conducted to evaluate the robustness of GasHis-Transformer, where ten different noises including four adversarial attack and six conventional image noises are added. In addition, a clinically meaningful study is executed to test the gastrointestinal cancer identification performance of GasHis-Transformer with 620 abnormal images and achieves 96.8% accuracy. Finally, a comparative study is performed to test the generalizability with both H&E and immunohistochemical stained images on a lymphoma image dataset and a breast cancer dataset, producing comparable F1-scores (85.6% and 82.8%) and accuracies (83.9% and 89.4%), respectively. In conclusion, GasHisTransformer demonstrates high classification performance and shows its significant potential in the GHIC task.

摘要: 现有的胃癌诊断深度学习方法普遍采用卷积神经网络。近年来，视觉变压器因其高性能和高效率而备受关注，但其应用大多集中在计算机视觉领域。本文提出了一种用于胃组织病理图像分类(GHIC)的多尺度视觉转换器模型(简称GasHis-Transformer)，该模型能够自动将胃显微图像分类为异常和正常病例。GasHis-Transformer模型由两个关键模块组成：全局信息模块和局部信息模块，有效地提取组织病理学特征。在我们的实验中，一个公共的苏木精伊红(H&E)染色的胃组织病理学数据集以1：1：2的比例分为训练集、验证集和测试集，训练集、验证集和测试集的比例为1：1：2。应用GasHis-Transformer模型估计胃组织病理学数据集的准确率、召回率、F1得分和准确率分别为98.0%、100.0%、96.0%和98.0%。此外，还对GasHis-Transformer的稳健性进行了关键研究，添加了10种不同的噪声，包括4种对抗性攻击和6种常规图像噪声。另外，利用620幅异常图像对GasHis-Transformer的胃肠道肿瘤识别性能进行了有临床意义的测试，准确率达到96.8%。最后，在淋巴瘤图像数据集和乳腺癌数据集上对H&E和免疫组织化学染色图像的泛化能力进行了比较研究，得到了可比的F1得分(85.6%和82.8%)和准确率(83.9%和89.4%)。总之，GasHisTransformer表现出很高的分类性能，并在GHIC任务中显示出巨大的潜力。



## **46. Measuring the Transferability of $\ell_\infty$ Attacks by the $\ell_2$ Norm**

用$\ELL_2$范数度量$\ELL_\INFTY$攻击的可转移性 cs.LG

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2102.10343v3)

**Authors**: Sizhe Chen, Qinghua Tao, Zhixing Ye, Xiaolin Huang

**Abstracts**: Deep neural networks could be fooled by adversarial examples with trivial differences to original samples. To keep the difference imperceptible in human eyes, researchers bound the adversarial perturbations by the $\ell_\infty$ norm, which is now commonly served as the standard to align the strength of different attacks for a fair comparison. However, we propose that using the $\ell_\infty$ norm alone is not sufficient in measuring the attack strength, because even with a fixed $\ell_\infty$ distance, the $\ell_2$ distance also greatly affects the attack transferability between models. Through the discovery, we reach more in-depth understandings towards the attack mechanism, i.e., several existing methods attack black-box models better partly because they craft perturbations with 70\% to 130\% larger $\ell_2$ distances. Since larger perturbations naturally lead to better transferability, we thereby advocate that the strength of attacks should be simultaneously measured by both the $\ell_\infty$ and $\ell_2$ norm. Our proposal is firmly supported by extensive experiments on ImageNet dataset from 7 attacks, 4 white-box models, and 9 black-box models.

摘要: 深层神经网络可能会被与原始样本有微小差异的对抗性例子所欺骗。为了保持这种差异在人眼中不可察觉，研究人员用$\ell_\infty$范数来约束对抗性扰动，这现在通常被用作对不同攻击强度进行公平比较的标准。然而，我们认为单独使用$\ell_\infty$范数来度量攻击强度是不够的，因为即使在固定的$\ell_\infty$距离的情况下，$\ell_2$距离也会极大地影响攻击在模型之间的可转移性。通过这一发现，我们对攻击机制有了更深入的理解，即现有的几种方法对黑盒模型的攻击效果较好，部分原因是它们设计的扰动具有较大的$ell_2$70~130\{##**$}${##**$}}。由于更大的扰动自然会导致更好的可转移性，因此我们主张攻击的强度应该同时用$\ell_inty$和$\ell_2$范数来度量。我们的建议得到了来自7个攻击、4个白盒模型和9个黑盒模型的ImageNet数据集的广泛实验的坚定支持。



## **47. Towards Evaluating the Robustness of Neural Networks Learned by Transduction**

基于转导学习的神经网络鲁棒性评价方法研究 cs.LG

Paper published at ICLR 2022. arXiv admin note: text overlap with  arXiv:2106.08387

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2110.14735v2)

**Authors**: Jiefeng Chen, Xi Wu, Yang Guo, Yingyu Liang, Somesh Jha

**Abstracts**: There has been emerging interest in using transductive learning for adversarial robustness (Goldwasser et al., NeurIPS 2020; Wu et al., ICML 2020; Wang et al., ArXiv 2021). Compared to traditional defenses, these defense mechanisms "dynamically learn" the model based on test-time input; and theoretically, attacking these defenses reduces to solving a bilevel optimization problem, which poses difficulty in crafting adaptive attacks. In this paper, we examine these defense mechanisms from a principled threat analysis perspective. We formulate and analyze threat models for transductive-learning based defenses, and point out important subtleties. We propose the principle of attacking model space for solving bilevel attack objectives, and present Greedy Model Space Attack (GMSA), an attack framework that can serve as a new baseline for evaluating transductive-learning based defenses. Through systematic evaluation, we show that GMSA, even with weak instantiations, can break previous transductive-learning based defenses, which were resilient to previous attacks, such as AutoAttack. On the positive side, we report a somewhat surprising empirical result of "transductive adversarial training": Adversarially retraining the model using fresh randomness at the test time gives a significant increase in robustness against attacks we consider.

摘要: 人们对使用转导学习来实现对抗鲁棒性产生了新的兴趣(Goldwasser等人，NeurIPS 2020；Wu等人，ICML 2020；Wang等人，Arxiv 2021)。与传统防御相比，这些防御机制“动态学习”基于测试时间输入的模型；从理论上讲，攻击这些防御归结为求解一个双层优化问题，这给自适应攻击的设计带来了困难。在本文中，我们从原则性威胁分析的角度来检查这些防御机制。建立并分析了基于传导式学习防御的威胁模型，指出了重要的细微之处。我们提出了攻击模型空间求解双层攻击目标的原理，并提出了贪婪模型空间攻击(GMSA)这一攻击框架，可作为评估基于转导学习的防御的新基线。通过系统的评估，我们证明了GMSA即使在弱实例化的情况下，也可以打破以往基于传导式学习的防御机制，这些防御机制对AutoAttack等先前的攻击是有弹性的。从积极的一面来看，我们报告了一个有点令人惊讶的“转导对抗训练”的经验结果：在测试时使用新的随机性对模型进行对抗性重新训练，可以显著提高对我们所考虑的攻击的鲁棒性。



## **48. Generalizable Information Theoretic Causal Representation**

广义信息论因果表示 cs.LG

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2202.08388v1)

**Authors**: Mengyue Yang, Xinyu Cai, Furui Liu, Xu Chen, Zhitang Chen, Jianye Hao, Jun Wang

**Abstracts**: It is evidence that representation learning can improve model's performance over multiple downstream tasks in many real-world scenarios, such as image classification and recommender systems. Existing learning approaches rely on establishing the correlation (or its proxy) between features and the downstream task (labels), which typically results in a representation containing cause, effect and spurious correlated variables of the label. Its generalizability may deteriorate because of the unstability of the non-causal parts. In this paper, we propose to learn causal representation from observational data by regularizing the learning procedure with mutual information measures according to our hypothetical causal graph. The optimization involves a counterfactual loss, based on which we deduce a theoretical guarantee that the causality-inspired learning is with reduced sample complexity and better generalization ability. Extensive experiments show that the models trained on causal representations learned by our approach is robust under adversarial attacks and distribution shift.

摘要: 事实证明，在图像分类和推荐系统等实际场景中，表征学习可以提高模型在多个下游任务上的性能。现有的学习方法依赖于在特征和下游任务(标签)之间建立相关性(或其代理)，这通常导致包含标签的原因、结果和虚假相关变量的表示。由于非因果部分的不稳定性，其泛化能力可能会恶化。在本文中，我们建议根据假设的因果图，通过互信息度量来规范学习过程，从而从观测数据中学习因果表示。这种优化涉及到反事实的损失，在此基础上，我们推导出因果启发学习具有更低的样本复杂度和更好的泛化能力的理论保证。大量实验表明，该方法训练的因果表示模型在对抗攻击和分布偏移情况下具有较好的鲁棒性。



## **49. Characterizing Attacks on Deep Reinforcement Learning**

深度强化学习攻击的特征描述 cs.LG

AAMAS 2022, 13 pages, 6 figures

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/1907.09470v3)

**Authors**: Xinlei Pan, Chaowei Xiao, Warren He, Shuang Yang, Jian Peng, Mingjie Sun, Jinfeng Yi, Zijiang Yang, Mingyan Liu, Bo Li, Dawn Song

**Abstracts**: Recent studies show that Deep Reinforcement Learning (DRL) models are vulnerable to adversarial attacks, which attack DRL models by adding small perturbations to the observations. However, some attacks assume full availability of the victim model, and some require a huge amount of computation, making them less feasible for real world applications. In this work, we make further explorations of the vulnerabilities of DRL by studying other aspects of attacks on DRL using realistic and efficient attacks. First, we adapt and propose efficient black-box attacks when we do not have access to DRL model parameters. Second, to address the high computational demands of existing attacks, we introduce efficient online sequential attacks that exploit temporal consistency across consecutive steps. Third, we explore the possibility of an attacker perturbing other aspects in the DRL setting, such as the environment dynamics. Finally, to account for imperfections in how an attacker would inject perturbations in the physical world, we devise a method for generating a robust physical perturbations to be printed. The attack is evaluated on a real-world robot under various conditions. We conduct extensive experiments both in simulation such as Atari games, robotics and autonomous driving, and on real-world robotics, to compare the effectiveness of the proposed attacks with baseline approaches. To the best of our knowledge, we are the first to apply adversarial attacks on DRL systems to physical robots.

摘要: 最近的研究表明，深度强化学习(DRL)模型容易受到敌意攻击，这种攻击是通过在观测值中添加小扰动来攻击DRL模型的。然而，一些攻击假设受害者模型完全可用，而另一些攻击需要大量的计算，这使得它们在现实世界的应用程序中不太可行。在这项工作中，我们通过研究对DRL的其他方面的攻击，使用真实而有效的攻击，进一步探讨了DRL的漏洞。首先，当我们无法获得DRL模型参数时，我们采用并提出了有效的黑盒攻击。其次，为了解决现有攻击对计算的高要求，我们引入了高效的在线顺序攻击，该攻击利用了连续步骤之间的时间一致性。第三，我们探讨攻击者干扰DRL设置中其他方面的可能性，例如环境动态。最后，为了说明攻击者如何在物理世界中注入扰动的不完善之处，我们设计了一种生成要打印的健壮物理扰动的方法。在不同条件下对真实机器人进行了攻击评估。我们在Atari游戏、机器人和自动驾驶等模拟游戏中，以及在真实机器人上进行了广泛的实验，以比较所提出的攻击和基线方法的有效性。据我们所知，我们是第一个将针对DRL系统的对抗性攻击应用于物理机器人的公司。



## **50. Real-Time Neural Voice Camouflage**

实时神经语音伪装 cs.SD

14 pages

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2112.07076v2)

**Authors**: Mia Chiquier, Chengzhi Mao, Carl Vondrick

**Abstracts**: Automatic speech recognition systems have created exciting possibilities for applications, however they also enable opportunities for systematic eavesdropping. We propose a method to camouflage a person's voice over-the-air from these systems without inconveniencing the conversation between people in the room. Standard adversarial attacks are not effective in real-time streaming situations because the characteristics of the signal will have changed by the time the attack is executed. We introduce predictive attacks, which achieve real-time performance by forecasting the attack that will be the most effective in the future. Under real-time constraints, our method jams the established speech recognition system DeepSpeech 3.9x more than baselines as measured through word error rate, and 6.6x more as measured through character error rate. We furthermore demonstrate our approach is practically effective in realistic environments over physical distances.

摘要: 自动语音识别系统为应用创造了令人兴奋的可能性，然而它们也为系统窃听提供了机会。我们提出了一种方法，从这些系统中伪装出人的空中语音，而不会给房间里的人之间的对话带来不便。标准对抗性攻击在实时流情况下无效，因为在执行攻击时信号的特性将发生变化。我们引入预测性攻击，通过预测未来最有效的攻击来实现实时性能。在实时约束条件下，我们的方法对已建立的语音识别系统DeepSpeech的拥塞程度是基线的3.9倍，通过字符错误率的衡量是基线的6.6倍。我们进一步证明了我们的方法在物理距离上的现实环境中是实际有效的。



