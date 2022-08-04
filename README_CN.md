# Latest Adversarial Attack Papers
**update at 2022-08-05 06:31:28**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Abusing Commodity DRAMs in IoT Devices to Remotely Spy on Temperature**

在物联网设备中滥用商品DRAM远程监视温度 cs.CR

Submitted to IEEE TIFS and currently under review

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.02125v1)

**Authors**: Florian Frank, Wenjie Xiong, Nikolaos Athanasios Anagnostopoulos, André Schaller, Tolga Arul, Farinaz Koushanfar, Stefan Katzenbeisser, Ulrich Ruhrmair, Jakub Szefer

**Abstracts**: The ubiquity and pervasiveness of modern Internet of Things (IoT) devices opens up vast possibilities for novel applications, but simultaneously also allows spying on, and collecting data from, unsuspecting users to a previously unseen extent. This paper details a new attack form in this vein, in which the decay properties of widespread, off-the-shelf DRAM modules are exploited to accurately sense the temperature in the vicinity of the DRAM-carrying device. Among others, this enables adversaries to remotely and purely digitally spy on personal behavior in users' private homes, or to collect security-critical data in server farms, cloud storage centers, or commercial production lines. We demonstrate that our attack can be performed by merely compromising the software of an IoT device and does not require hardware modifications or physical access at attack time. It can achieve temperature resolutions of up to 0.5{\deg}C over a range of 0{\deg}C to 70{\deg}C in practice. Perhaps most interestingly, it even works in devices that do not have a dedicated temperature sensor on board. To complete our work, we discuss practical attack scenarios as well as possible countermeasures against our temperature espionage attacks.

摘要: 现代物联网(IoT)设备的无处不在和无处不在，为新的应用打开了巨大的可能性，但同时也允许对毫无戒心的用户进行间谍活动，并从他们那里收集数据，达到前所未有的程度。本文详细介绍了一种新的攻击形式，利用广泛存在的现成DRAM模块的衰减特性来准确检测DRAM携带设备附近的温度。其中，这使攻击者能够远程、纯数字地监视用户私人住宅中的个人行为，或者收集服务器群、云存储中心或商业生产线中的安全关键数据。我们证明，我们的攻击可以仅通过危害物联网设备的软件来执行，并且在攻击时不需要修改硬件或进行物理访问。实际应用表明，在0~70℃的温度范围内，温度分辨率最高可达0.5℃。也许最有趣的是，它甚至可以在没有专用温度传感器的设备上工作。为了完成我们的工作，我们讨论了实际的攻击方案以及针对我们的温度间谍攻击的可能对策。



## **2. Local Differential Privacy for Federated Learning**

联合学习中的局部差分隐私 cs.CR

17 pages

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2202.06053v2)

**Authors**: M. A. P. Chamikara, Dongxi Liu, Seyit Camtepe, Surya Nepal, Marthie Grobler, Peter Bertok, Ibrahim Khalil

**Abstracts**: Advanced adversarial attacks such as membership inference and model memorization can make federated learning (FL) vulnerable and potentially leak sensitive private data. Local differentially private (LDP) approaches are gaining more popularity due to stronger privacy notions and native support for data distribution compared to other differentially private (DP) solutions. However, DP approaches assume that the FL server (that aggregates the models) is honest (run the FL protocol honestly) or semi-honest (run the FL protocol honestly while also trying to learn as much information as possible). These assumptions make such approaches unrealistic and unreliable for real-world settings. Besides, in real-world industrial environments (e.g., healthcare), the distributed entities (e.g., hospitals) are already composed of locally running machine learning models (this setting is also referred to as the cross-silo setting). Existing approaches do not provide a scalable mechanism for privacy-preserving FL to be utilized under such settings, potentially with untrusted parties. This paper proposes a new local differentially private FL (named LDPFL) protocol for industrial settings. LDPFL can run in industrial settings with untrusted entities while enforcing stronger privacy guarantees than existing approaches. LDPFL shows high FL model performance (up to 98%) under small privacy budgets (e.g., epsilon = 0.5) in comparison to existing methods.

摘要: 高级对抗性攻击，如成员推理和模型记忆，会使联邦学习(FL)容易受到攻击，并可能泄露敏感的私人数据。与其他差异私有(DP)解决方案相比，本地差异私有(LDP)方法由于更强的隐私概念和对数据分发的本地支持而越来越受欢迎。然而，DP方法假设FL服务器(聚集模型)是诚实的(诚实地运行FL协议)或半诚实的(诚实地运行FL协议，同时还试图了解尽可能多的信息)。这些假设使得这种方法对于现实世界的设置来说是不现实和不可靠的。此外，在真实世界的工业环境(例如，医疗保健)中，分布式实体(例如，医院)已经由本地运行的机器学习模型组成(该设置也被称为跨竖井设置)。现有方法没有提供用于保护隐私的FL的可扩展机制以在这样的设置下使用，可能与不可信方一起使用。提出了一种适用于工业环境的局部差分私有FL协议(简称LDPFL)。LDPFL可以在具有不可信实体的工业环境中运行，同时执行比现有方法更强大的隐私保障。与现有方法相比，LDPFL在较小的隐私预算(例如，epsilon=0.5)下表现出高的FL模型性能(高达98%)。



## **3. SAC-AP: Soft Actor Critic based Deep Reinforcement Learning for Alert Prioritization**

SAC-AP：基于软参与者批评者的深度强化学习告警优先级 cs.CR

8 pages, 8 figures, IEEE WORLD CONGRESS ON COMPUTATIONAL INTELLIGENCE  2022

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2207.13666v3)

**Authors**: Lalitha Chavali, Tanay Gupta, Paresh Saxena

**Abstracts**: Intrusion detection systems (IDS) generate a large number of false alerts which makes it difficult to inspect true positives. Hence, alert prioritization plays a crucial role in deciding which alerts to investigate from an enormous number of alerts that are generated by IDS. Recently, deep reinforcement learning (DRL) based deep deterministic policy gradient (DDPG) off-policy method has shown to achieve better results for alert prioritization as compared to other state-of-the-art methods. However, DDPG is prone to the problem of overfitting. Additionally, it also has a poor exploration capability and hence it is not suitable for problems with a stochastic environment. To address these limitations, we present a soft actor-critic based DRL algorithm for alert prioritization (SAC-AP), an off-policy method, based on the maximum entropy reinforcement learning framework that aims to maximize the expected reward while also maximizing the entropy. Further, the interaction between an adversary and a defender is modeled as a zero-sum game and a double oracle framework is utilized to obtain the approximate mixed strategy Nash equilibrium (MSNE). SAC-AP finds robust alert investigation policies and computes pure strategy best response against opponent's mixed strategy. We present the overall design of SAC-AP and evaluate its performance as compared to other state-of-the art alert prioritization methods. We consider defender's loss, i.e., the defender's inability to investigate the alerts that are triggered due to attacks, as the performance metric. Our results show that SAC-AP achieves up to 30% decrease in defender's loss as compared to the DDPG based alert prioritization method and hence provides better protection against intrusions. Moreover, the benefits are even higher when SAC-AP is compared to other traditional alert prioritization methods including Uniform, GAIN, RIO and Suricata.

摘要: 入侵检测系统(入侵检测系统)产生大量的错误警报，使得对真实阳性的检测变得困难。因此，警报优先级在决定从由入侵检测系统生成的大量警报中调查哪些警报时起着至关重要的作用。近年来，与其他方法相比，基于深度强化学习(DRL)的深度确定性策略梯度(DDPG)非策略方法能够获得更好的告警优先级排序结果。然而，DDPG容易出现过度匹配的问题。此外，它的探测能力也很差，因此不适合于具有随机环境的问题。针对这些局限性，我们提出了一种基于软参与者-批评者的DRL警报优先排序算法(SAC-AP)，这是一种基于最大熵强化学习框架的非策略方法，旨在最大化期望回报的同时最大化熵。在此基础上，将对手和防御者之间的相互作用建模为零和博弈，并利用双预言框架得到近似的混合策略纳什均衡。SAC-AP发现稳健的警戒调查策略，并针对对手的混合策略计算纯策略的最佳响应。我们介绍了SAC-AP的总体设计，并与其他最先进的警报优先排序方法进行了比较，评估了其性能。我们将防御者的损失，即防御者无法调查由于攻击而触发的警报作为性能指标。结果表明，与基于DDPG的告警优先级排序方法相比，SAC-AP可以减少高达30%的防御者损失，从而提供更好的防御入侵保护。此外，当SAC-AP与其他传统的警报优先排序方法(包括Uniform、Gain、Rio和Suricata)相比时，好处甚至更高。



## **4. Spectrum Focused Frequency Adversarial Attacks for Automatic Modulation Classification**

用于自动调制分类的频谱聚焦频率对抗攻击 cs.CR

6 pages, 9 figures

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.01919v1)

**Authors**: Sicheng Zhang, Jiarun Yu, Zhida Bao, Shiwen Mao, Yun Lin

**Abstracts**: Artificial intelligence (AI) technology has provided a potential solution for automatic modulation recognition (AMC). Unfortunately, AI-based AMC models are vulnerable to adversarial examples, which seriously threatens the efficient, secure and trusted application of AI in AMC. This issue has attracted the attention of researchers. Various studies on adversarial attacks and defenses evolve in a spiral. However, the existing adversarial attack methods are all designed in the time domain. They introduce more high-frequency components in the frequency domain, due to abrupt updates in the time domain. For this issue, from the perspective of frequency domain, we propose a spectrum focused frequency adversarial attacks (SFFAA) for AMC model, and further draw on the idea of meta-learning, propose a Meta-SFFAA algorithm to improve the transferability in the black-box attacks. Extensive experiments, qualitative and quantitative metrics demonstrate that the proposed algorithm can concentrate the adversarial energy on the spectrum where the signal is located, significantly improve the adversarial attack performance while maintaining the concealment in the frequency domain.

摘要: 人工智能(AI)技术为自动调制识别(AMC)提供了一种潜在的解决方案。不幸的是，基于人工智能的AMC模型容易受到敌意例子的攻击，这严重威胁了人工智能在AMC中的高效、安全和可信的应用。这个问题已经引起了研究人员的关注。关于对抗性攻击和防御的各种研究呈螺旋式发展。然而，现有的对抗性攻击方法都是在时间域设计的。由于时间域中的突然更新，它们在频域中引入了更多的高频分量。针对这一问题，从频域的角度出发，提出了一种针对AMC模型的频谱聚焦频率对抗攻击算法(SFFAA)，并进一步借鉴元学习的思想，提出了一种Meta-SFFAA算法来提高黑盒攻击的可转移性。大量的实验、定性和定量指标表明，该算法可以将对抗能量集中在信号所在的频谱上，在保持频域隐蔽性的同时，显著提高了对抗攻击的性能。



## **5. Mass Exit Attacks on the Lightning Network**

闪电网络上的大规模出口攻击 cs.CR

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.01908v1)

**Authors**: Anastasios Sidiropoulos, Cosimo Sguanci

**Abstracts**: The Lightning Network (LN) has enjoyed rapid growth over recent years, and has become the most popular scaling solution for the Bitcoin blockchain. The security of the LN hinges on the ability of the nodes to close a channel by settling their balances, which requires confirming a transaction on the Bitcoin blockchain within a pre-agreed time period. This inherent timing restriction that the LN must satisfy, make it susceptible to attacks that seek to increase the congestion on the Bitcoin blockchain, thus preventing correct protocol execution. We study the susceptibility of the LN to \emph{mass exit} attacks, in the presence of a small coalition of adversarial nodes. This is a scenario where an adversary forces a large set of honest protocol participants to interact with the blockchain. We focus on two types of attacks: (i) The first is a \emph{zombie} attack, where a set of $k$ nodes become unresponsive with the goal to lock the funds of many channels for a period of time longer than what the LN protocol dictates. (ii) The second is a \emph{mass double-spend} attack, where a set of $k$ nodes attempt to steal funds by submitting many closing transactions that settle channels using expired protocol states; this causes many honest nodes to have to quickly respond by submitting invalidating transactions. We show via simulations that, under historically-plausible congestion conditions, with mild statistical assumptions on channel balances, both of the attacks can be performed by a very small coalition. To perform our simulations, we formulate the problem of finding a worst-case coalition of $k$ adversarial nodes as a graph cut problem. Our experimental findings are supported by a theoretical justification based on the scale-free topology of the LN.

摘要: 闪电网络(Lightning Network，LN)近年来增长迅速，已成为比特币区块链最受欢迎的扩展解决方案。LN的安全性取决于节点通过结算余额关闭通道的能力，这需要在预先商定的时间段内确认比特币区块链上的交易。LN必须满足的这一固有时间限制使其容易受到攻击，这些攻击试图增加比特币区块链上的拥塞，从而阻止正确的协议执行。我们研究了在存在一个小的敌方节点联盟的情况下，LN对EMPH{MASS EXIT}攻击的敏感性。这是一种对手迫使大量诚实的协议参与者与区块链交互的场景。我们主要关注两种类型的攻击：(I)第一种是僵尸攻击，其中一组$k$节点变得无响应，目标是锁定多个频道的资金长于LN协议规定的时间段。(Ii)第二种攻击是大规模双重花费攻击，其中一组$k$节点试图通过提交许多关闭的事务来窃取资金，这些事务使用过期的协议状态来结算通道；这导致许多诚实的节点不得不通过提交无效事务来快速响应。我们通过模拟表明，在历史上看似合理的拥塞条件下，在对信道平衡的温和统计假设下，这两种攻击都可以由非常小的联盟来执行。为了执行我们的模拟，我们将寻找$k$个敌对节点的最坏情况联盟的问题描述为一个图割问题。我们的实验结果得到了基于LN的无标度拓扑的理论证明。



## **6. On the Evaluation of User Privacy in Deep Neural Networks using Timing Side Channel**

基于时序侧通道的深度神经网络用户隐私评估研究 cs.CR

15 pages, 20 figures

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.01113v2)

**Authors**: Shubhi Shukla, Manaar Alam, Sarani Bhattacharya, Debdeep Mukhopadhyay, Pabitra Mitra

**Abstracts**: Recent Deep Learning (DL) advancements in solving complex real-world tasks have led to its widespread adoption in practical applications. However, this opportunity comes with significant underlying risks, as many of these models rely on privacy-sensitive data for training in a variety of applications, making them an overly-exposed threat surface for privacy violations. Furthermore, the widespread use of cloud-based Machine-Learning-as-a-Service (MLaaS) for its robust infrastructure support has broadened the threat surface to include a variety of remote side-channel attacks. In this paper, we first identify and report a novel data-dependent timing side-channel leakage (termed Class Leakage) in DL implementations originating from non-constant time branching operation in a widely used DL framework PyTorch. We further demonstrate a practical inference-time attack where an adversary with user privilege and hard-label black-box access to an MLaaS can exploit Class Leakage to compromise the privacy of MLaaS users. DL models are vulnerable to Membership Inference Attack (MIA), where an adversary's objective is to deduce whether any particular data has been used while training the model. In this paper, as a separate case study, we demonstrate that a DL model secured with differential privacy (a popular countermeasure against MIA) is still vulnerable to MIA against an adversary exploiting Class Leakage. We develop an easy-to-implement countermeasure by making a constant-time branching operation that alleviates the Class Leakage and also aids in mitigating MIA. We have chosen two standard benchmarking image classification datasets, CIFAR-10 and CIFAR-100 to train five state-of-the-art pre-trained DL models, over two different computing environments having Intel Xeon and Intel i7 processors to validate our approach.

摘要: 最近深度学习(DL)在解决复杂现实世界任务方面的进步导致了它在实际应用中的广泛采用。然而，这种机会伴随着巨大的潜在风险，因为这些模型中的许多依赖于隐私敏感数据来进行各种应用程序的培训，使它们成为侵犯隐私的过度暴露的威胁表面。此外，基于云的机器学习即服务(MLaaS)因其强大的基础设施支持而广泛使用，扩大了威胁面，包括各种远程侧通道攻击。在这篇文章中，我们首先识别和报告了一种新的数据相关的定时侧通道泄漏(称为类泄漏)，该泄漏是由广泛使用的动态链接库框架中的非常数时间分支操作引起的。我们进一步展示了一个实用的推理时间攻击，其中具有用户权限和硬标签黑盒访问MLaaS的攻击者可以利用类泄漏来危害MLaaS用户的隐私。DL模型容易受到成员推理攻击(MIA)，对手的目标是推断在训练模型时是否使用了特定的数据。在本文中，作为一个单独的案例研究，我们证明了在差异隐私保护下的DL模型(一种流行的针对MIA的对策)仍然容易受到MIA对利用类泄漏的攻击者的攻击。我们开发了一种易于实现的对策，通过进行恒定时间分支操作来缓解类泄漏，并帮助缓解MIA。我们选择了两个标准的基准图像分类数据集，CIFAR-10和CIFAR-100来训练五个最先进的预训练的DL模型，在两种不同的计算环境中使用Intel Xeon和Intel i7处理器来验证我们的方法。



## **7. Robust Graph Neural Networks using Weighted Graph Laplacian**

基于加权图拉普拉斯的稳健图神经网络 cs.LG

Accepted at IEEE International Conference on Signal Processing and  Communications (SPCOM), 2022

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.01853v1)

**Authors**: Bharat Runwal, Vivek, Sandeep Kumar

**Abstracts**: Graph neural network (GNN) is achieving remarkable performances in a variety of application domains. However, GNN is vulnerable to noise and adversarial attacks in input data. Making GNN robust against noises and adversarial attacks is an important problem. The existing defense methods for GNNs are computationally demanding and are not scalable. In this paper, we propose a generic framework for robustifying GNN known as Weighted Laplacian GNN (RWL-GNN). The method combines Weighted Graph Laplacian learning with the GNN implementation. The proposed method benefits from the positive semi-definiteness property of Laplacian matrix, feature smoothness, and latent features via formulating a unified optimization framework, which ensures the adversarial/noisy edges are discarded and connections in the graph are appropriately weighted. For demonstration, the experiments are conducted with Graph convolutional neural network(GCNN) architecture, however, the proposed framework is easily amenable to any existing GNN architecture. The simulation results with benchmark dataset establish the efficacy of the proposed method, both in accuracy and computational efficiency. Code can be accessed at https://github.com/Bharat-Runwal/RWL-GNN.

摘要: 图形神经网络(GNN)在各种应用领域都取得了令人瞩目的成绩。然而，GNN很容易受到输入数据中的噪声和对抗性攻击。如何使GNN对噪声和敌意攻击具有健壮性是一个重要的问题。现有的GNN防御方法计算量大且不可扩展。在本文中，我们提出了一种称为加权拉普拉斯GNN(RWL-GNN)的通用GNN框架。该方法将加权图拉普拉斯学习与GNN实现相结合。该方法充分利用了拉普拉斯矩阵的正半定性、特征的光滑性和潜在特征，建立了统一的优化框架，确保了对敌边/噪声边的丢弃和图中连接的适当加权。为了进行演示，实验使用了图卷积神经网络(GCNN)结构，然而，所提出的框架可以很容易地服从于任何现有的GNN结构。利用基准数据集的仿真结果验证了该方法在精度和计算效率上的有效性。代码可在https://github.com/Bharat-Runwal/RWL-GNN.上访问



## **8. Multiclass ASMA vs Targeted PGD Attack in Image Segmentation**

图像分割中多类ASMA与靶向PGD攻击的比较 cs.CV

10 pages, 6 figures

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.01844v1)

**Authors**: Johnson Vo, Jiabao Xie, Sahil Patel

**Abstracts**: Deep learning networks have demonstrated high performance in a large variety of applications, such as image classification, speech recognition, and natural language processing. However, there exists a major vulnerability exploited by the use of adversarial attacks. An adversarial attack imputes images by altering the input image very slightly, making it nearly undetectable to the naked eye, but results in a very different classification by the network. This paper explores the projected gradient descent (PGD) attack and the Adaptive Mask Segmentation Attack (ASMA) on the image segmentation DeepLabV3 model using two types of architectures: MobileNetV3 and ResNet50, It was found that PGD was very consistent in changing the segmentation to be its target while the generalization of ASMA to a multiclass target was not as effective. The existence of such attack however puts all of image classification deep learning networks in danger of exploitation.

摘要: 深度学习网络在图像分类、语音识别、自然语言处理等多种应用中表现出了很高的性能。然而，存在一个通过使用对抗性攻击来利用的重大漏洞。敌意攻击通过非常轻微地更改输入图像来计算图像，使其几乎无法被肉眼检测到，但会导致网络进行非常不同的分类。利用两种结构：MobileNetV3和ResNet50，对DeepLabV3图像分割模型进行了投影梯度下降(PGD)攻击和自适应掩码分割攻击(ASMA)的研究，发现PGD在改变分割为其目标方面具有很好的一致性，而ASMA对多类目标的泛化效果不佳。然而，这种攻击的存在使所有的图像分类深度学习网络都处于被利用的危险之中。



## **9. Adversarial Camouflage for Node Injection Attack on Graphs**

图上节点注入攻击的对抗性伪装 cs.LG

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.01819v1)

**Authors**: Shuchang Tao, Qi Cao, Huawei Shen, Yunfan Wu, Liang Hou, Xueqi Cheng

**Abstracts**: Node injection attacks against Graph Neural Networks (GNNs) have received emerging attention as a practical attack scenario, where the attacker injects malicious nodes instead of modifying node features or edges to degrade the performance of GNNs. Despite the initial success of node injection attacks, we find that the injected nodes by existing methods are easy to be distinguished from the original normal nodes by defense methods and limiting their attack performance in practice. To solve the above issues, we devote to camouflage node injection attack, i.e., camouflaging injected malicious nodes (structure/attributes) as the normal ones that appear legitimate/imperceptible to defense methods. The non-Euclidean nature of graph data and the lack of human prior brings great challenges to the formalization, implementation, and evaluation of camouflage on graphs. In this paper, we first propose and formulate the camouflage of injected nodes from both the fidelity and diversity of the ego networks centered around injected nodes. Then, we design an adversarial CAmouflage framework for Node injection Attack, namely CANA, to improve the camouflage while ensuring the attack performance. Several novel indicators for graph camouflage are further designed for a comprehensive evaluation. Experimental results demonstrate that when equipping existing node injection attack methods with our proposed CANA framework, the attack performance against defense methods as well as node camouflage is significantly improved.

摘要: 针对图神经网络的节点注入攻击作为一种实用的攻击场景受到了越来越多的关注，即攻击者注入恶意节点而不是修改节点特征或边来降低图神经网络的性能。尽管节点注入攻击取得了初步的成功，但我们发现，现有方法注入的节点很容易通过防御方法与原来的正常节点区分开来，限制了它们在实践中的攻击性能。为了解决上述问题，我们致力于伪装节点注入攻击，即伪装注入的恶意节点(结构/属性)作为正常的合法/不可察觉的防御方法。图数据的非欧几里得性质和人类先验知识的缺乏给图上伪装的形式化、实现和评估带来了巨大的挑战。本文首先从以注入节点为中心的EGO网络的保真度和多样性两个方面提出并构造了注入节点的伪装。然后，设计了一种节点注入攻击的对抗性伪装框架CANA，在保证攻击性能的同时提高伪装性能。进一步设计了几种新的图形伪装指标，进行了综合评价。实验结果表明，在现有的节点注入攻击方法中加入CANA框架后，对防御方法的攻击性能以及对节点伪装的攻击性能都得到了显著提高。



## **10. Success of Uncertainty-Aware Deep Models Depends on Data Manifold Geometry**

不确定性感知深度模型的成功依赖于数据流形几何 cs.LG

9 pages

**SubmitDate**: 2022-08-02    [paper-pdf](http://arxiv.org/pdf/2208.01705v1)

**Authors**: Mark Penrod, Harrison Termotto, Varshini Reddy, Jiayu Yao, Finale Doshi-Velez, Weiwei Pan

**Abstracts**: For responsible decision making in safety-critical settings, machine learning models must effectively detect and process edge-case data. Although existing works show that predictive uncertainty is useful for these tasks, it is not evident from literature which uncertainty-aware models are best suited for a given dataset. Thus, we compare six uncertainty-aware deep learning models on a set of edge-case tasks: robustness to adversarial attacks as well as out-of-distribution and adversarial detection. We find that the geometry of the data sub-manifold is an important factor in determining the success of various models. Our finding suggests an interesting direction in the study of uncertainty-aware deep learning models.

摘要: 为了在安全关键环境中做出负责任的决策，机器学习模型必须有效地检测和处理边缘案例数据。虽然现有的工作表明，预测不确定性对这些任务是有用的，但从文献中并不明显地看到，哪些不确定性感知模型最适合给定的数据集。因此，我们在一组边缘情况任务上比较了六种不确定性感知的深度学习模型：对对手攻击的健壮性以及分布外和对抗性检测。我们发现，数据子流形的几何形状是决定各种模型成功与否的重要因素。我们的发现为不确定性感知深度学习模型的研究提供了一个有趣的方向。



## **11. CAPD: A Context-Aware, Policy-Driven Framework for Secure and Resilient IoBT Operations**

CAPD：环境感知、策略驱动的IoBT安全弹性运营框架 cs.CR

**SubmitDate**: 2022-08-02    [paper-pdf](http://arxiv.org/pdf/2208.01703v1)

**Authors**: Sai Sree Laya Chukkapalli, Anupam Joshi, Tim Finin, Robert F. Erbacher

**Abstracts**: The Internet of Battlefield Things (IoBT) will advance the operational effectiveness of infantry units. However, this requires autonomous assets such as sensors, drones, combat equipment, and uncrewed vehicles to collaborate, securely share information, and be resilient to adversary attacks in contested multi-domain operations. CAPD addresses this problem by providing a context-aware, policy-driven framework supporting data and knowledge exchange among autonomous entities in a battlespace. We propose an IoBT ontology that facilitates controlled information sharing to enable semantic interoperability between systems. Its key contributions include providing a knowledge graph with a shared semantic schema, integration with background knowledge, efficient mechanisms for enforcing data consistency and drawing inferences, and supporting attribute-based access control. The sensors in the IoBT provide data that create populated knowledge graphs based on the ontology. This paper describes using CAPD to detect and mitigate adversary actions. CAPD enables situational awareness using reasoning over the sensed data and SPARQL queries. For example, adversaries can cause sensor failure or hijacking and disrupt the tactical networks to degrade video surveillance. In such instances, CAPD uses an ontology-based reasoner to see how alternative approaches can still support the mission. Depending on bandwidth availability, the reasoner initiates the creation of a reduced frame rate grayscale video by active transcoding or transmits only still images. This ability to reason over the mission sensed environment and attack context permits the autonomous IoBT system to exhibit resilience in contested conditions.

摘要: 战场物联网(IoBT)将提高步兵部队的作战效能。然而，这需要传感器、无人机、作战设备和无人驾驶车辆等自主资产进行协作，安全地共享信息，并在有争议的多领域行动中对对手攻击具有弹性。CAPD通过提供支持战场空间中自治实体之间的数据和知识交换的上下文感知、策略驱动的框架来解决这一问题。我们提出了一种IoBT本体，它促进了受控信息共享，从而实现了系统之间的语义互操作。它的主要贡献包括提供具有共享语义模式的知识图，与背景知识的集成，执行数据一致性和推理的有效机制，以及支持基于属性的访问控制。IoBT中的传感器提供基于本体创建填充的知识图的数据。本文描述了使用CAPD来检测和缓解恶意行为。CAPD通过对感测数据和SPARQL查询进行推理来实现态势感知。例如，敌手可能会导致传感器故障或劫持，并扰乱战术网络以降低视频监控。在这种情况下，CAPD使用基于本体的推理机来查看替代方法如何仍然能够支持任务。根据带宽可用性，推理器通过主动代码转换来启动降低帧速率的灰度视频的创建，或者仅传输静止图像。这种对任务感知环境和攻击上下文进行推理的能力使自主IoBT系统在竞争条件下表现出弹性。



## **12. Adversarial Detection Avoidance Attacks: Evaluating the robustness of perceptual hashing-based client-side scanning**

对抗性检测避免攻击：评估基于感知散列的客户端扫描的健壮性 cs.CR

This is a revised version of the paper published at USENIX Security  2022. We now use a semi-automated procedure to remove duplicates from the  ImageNet dataset

**SubmitDate**: 2022-08-02    [paper-pdf](http://arxiv.org/pdf/2106.09820v3)

**Authors**: Shubham Jain, Ana-Maria Cretu, Yves-Alexandre de Montjoye

**Abstracts**: End-to-end encryption (E2EE) by messaging platforms enable people to securely and privately communicate with one another. Its widespread adoption however raised concerns that illegal content might now be shared undetected. Following the global pushback against key escrow systems, client-side scanning based on perceptual hashing has been recently proposed by tech companies, governments and researchers to detect illegal content in E2EE communications. We here propose the first framework to evaluate the robustness of perceptual hashing-based client-side scanning to detection avoidance attacks and show current systems to not be robust. More specifically, we propose three adversarial attacks--a general black-box attack and two white-box attacks for discrete cosine transform-based algorithms--against perceptual hashing algorithms. In a large-scale evaluation, we show perceptual hashing-based client-side scanning mechanisms to be highly vulnerable to detection avoidance attacks in a black-box setting, with more than 99.9% of images successfully attacked while preserving the content of the image. We furthermore show our attack to generate diverse perturbations, strongly suggesting that straightforward mitigation strategies would be ineffective. Finally, we show that the larger thresholds necessary to make the attack harder would probably require more than one billion images to be flagged and decrypted daily, raising strong privacy concerns. Taken together, our results shed serious doubts on the robustness of perceptual hashing-based client-side scanning mechanisms currently proposed by governments, organizations, and researchers around the world.

摘要: 消息传递平台的端到端加密(E2EE)使人们能够安全且私密地相互通信。然而，它的广泛采用引发了人们的担忧，即非法内容现在可能被分享而不被发现。继全球对密钥托管系统的抵制之后，科技公司、政府和研究人员最近提出了基于感知散列的客户端扫描，以检测E2EE通信中的非法内容。我们在这里提出了第一个框架来评估基于感知散列的客户端扫描对检测规避攻击的健壮性，并表明当前的系统是不健壮的。更具体地说，我们针对感知散列算法提出了三种对抗性攻击--一种通用的黑盒攻击和两种基于离散余弦变换的算法的白盒攻击。在大规模的评估中，我们发现基于感知散列的客户端扫描机制在黑盒环境下非常容易受到检测回避攻击，99.9%以上的图像在保护图像内容的同时被攻击成功。此外，我们还展示了我们的攻击会产生不同的扰动，强烈表明直接的缓解策略将是无效的。最后，我们指出，提高攻击难度所需的更大门槛可能需要每天标记和解密超过10亿张图像，这引发了强烈的隐私问题。综上所述，我们的结果对目前世界各地的政府、组织和研究人员提出的基于感知散列的客户端扫描机制的健壮性提出了严重的质疑。



## **13. Quantum Lock: A Provable Quantum Communication Advantage**

量子锁：一种可证明的量子通信优势 quant-ph

Replacement of paper "Hybrid PUF: A Novel Way to Enhance the Security  of Classical PUFs" (arXiv:2110.09469)

**SubmitDate**: 2022-08-02    [paper-pdf](http://arxiv.org/pdf/2110.09469v3)

**Authors**: Kaushik Chakraborty, Mina Doosti, Yao Ma, Chirag Wadhwa, Myrto Arapinis, Elham Kashefi

**Abstracts**: Physical unclonable functions(PUFs) provide a unique fingerprint to a physical entity by exploiting the inherent physical randomness. Gao et al. discussed the vulnerability of most current-day PUFs to sophisticated machine learning-based attacks. We address this problem by integrating classical PUFs and existing quantum communication technology. Specifically, this paper proposes a generic design of provably secure PUFs, called hybrid locked PUFs(HLPUFs), providing a practical solution for securing classical PUFs. An HLPUF uses a classical PUF(CPUF), and encodes the output into non-orthogonal quantum states to hide the outcomes of the underlying CPUF from any adversary. Here we introduce a quantum lock to protect the HLPUFs from any general adversaries. The indistinguishability property of the non-orthogonal quantum states, together with the quantum lockdown technique prevents the adversary from accessing the outcome of the CPUFs. Moreover, we show that by exploiting non-classical properties of quantum states, the HLPUF allows the server to reuse the challenge-response pairs for further client authentication. This result provides an efficient solution for running PUF-based client authentication for an extended period while maintaining a small-sized challenge-response pairs database on the server side. Later, we support our theoretical contributions by instantiating the HLPUFs design using accessible real-world CPUFs. We use the optimal classical machine-learning attacks to forge both the CPUFs and HLPUFs, and we certify the security gap in our numerical simulation for construction which is ready for implementation.

摘要: 物理不可克隆函数(PUF)通过利用固有的物理随机性为物理实体提供唯一指纹。高等人。讨论了当前大多数PUF对复杂的基于机器学习的攻击的脆弱性。我们通过将经典的PUF和现有的量子通信技术相结合来解决这个问题。具体地说，本文提出了一种可证明安全的PUF的通用设计，称为混合锁定PUF(HLPUF)，为保护经典PUF提供了一种实用的解决方案。HLPUF使用经典的PUF(CPUF)，并将输出编码为非正交的量子态，以向任何对手隐藏底层CPUF的结果。在这里，我们引入量子锁来保护HLPUF免受任何一般对手的攻击。非正交量子态的不可分辨特性，加上量子锁定技术，阻止了攻击者访问CPUF的结果。此外，我们证明了通过利用量子态的非经典属性，HLPUF允许服务器重用挑战-响应对来进行进一步的客户端认证。这一结果为长期运行基于PUF的客户端身份验证提供了一个有效的解决方案，同时在服务器端维护一个小型的挑战-响应对数据库。随后，我们通过使用可访问的真实CPUF来实例化HLPUF设计来支持我们的理论贡献。我们使用最优经典机器学习攻击来伪造CPUF和HLPUF，并证明了我们的构造数值模拟中的安全漏洞。



## **14. SCFI: State Machine Control-Flow Hardening Against Fault Attacks**

SCFI：针对故障攻击的状态机控制流强化 cs.CR

**SubmitDate**: 2022-08-02    [paper-pdf](http://arxiv.org/pdf/2208.01356v1)

**Authors**: Pascal Nasahl, Martin Unterguggenberger, Rishub Nagpal, Robert Schilling, David Schrammel, Stefan Mangard

**Abstracts**: Fault injection (FI) is a powerful attack methodology allowing an adversary to entirely break the security of a target device. As finite-state machines (FSMs) are fundamental hardware building blocks responsible for controlling systems, inducing faults into these controllers enables an adversary to hijack the execution of the integrated circuit. A common defense strategy mitigating these attacks is to manually instantiate FSMs multiple times and detect faults using a majority voting logic. However, as each additional FSM instance only provides security against one additional induced fault, this approach scales poorly in a multi-fault attack scenario.   In this paper, we present SCFI: a strong, probabilistic FSM protection mechanism ensuring that control-flow deviations from the intended control-flow are detected even in the presence of multiple faults. At its core, SCFI consists of a hardened next-state function absorbing the execution history as well as the FSM's control signals to derive the next state. When either the absorbed inputs, the state registers, or the function itself are affected by faults, SCFI triggers an error with no detection latency. We integrate SCFI into a synthesis tool capable of automatically hardening arbitrary unprotected FSMs without user interaction and open-source the tool. Our evaluation shows that SCFI provides strong protection guarantees with a better area-time product than FSMs protected using classical redundancy-based approaches. Finally, we formally verify the resilience of the protected state machines using a pre-silicon fault analysis tool.

摘要: 故障注入(FI)是一种强大的攻击方法，允许对手完全破坏目标设备的安全。由于有限状态机(FSM)是负责控制系统的基本硬件构建块，因此在这些控制器中引入故障使对手能够劫持集成电路的执行。缓解这些攻击的常见防御策略是多次手动实例化FSM并使用多数投票逻辑检测故障。然而，由于每个额外的FSM实例仅针对一个额外的诱发故障提供安全性，因此该方法在多故障攻击场景中伸缩性不佳。在本文中，我们提出了SCFI：一种强大的、概率的有限状态机保护机制，确保即使在存在多个故障的情况下也能检测到控制流与预期控制流的偏差。在其核心，SCFI由一个强化的下一状态函数组成，该函数吸收执行历史以及FSM的控制信号以得出下一状态。当吸收的输入、状态寄存器或功能本身受到故障影响时，SCFI会在没有检测延迟的情况下触发错误。我们将SCFI集成到一个合成工具中，该工具能够自动硬化任意不受保护的FSM，而无需用户交互并开放该工具的源代码。我们的评估表明，SCFI提供了强大的保护保证，比使用经典的基于冗余的方法保护的FSM具有更好的区域-时间乘积。最后，我们使用预硅故障分析工具正式验证了受保护状态机的弹性。



## **15. Understanding Adversarial Robustness of Vision Transformers via Cauchy Problem**

通过柯西问题理解视觉变形器的对抗稳健性 cs.CV

Accepted by ECML-PKDD 2022

**SubmitDate**: 2022-08-01    [paper-pdf](http://arxiv.org/pdf/2208.00906v1)

**Authors**: Zheng Wang, Wenjie Ruan

**Abstracts**: Recent research on the robustness of deep learning has shown that Vision Transformers (ViTs) surpass the Convolutional Neural Networks (CNNs) under some perturbations, e.g., natural corruption, adversarial attacks, etc. Some papers argue that the superior robustness of ViT comes from the segmentation of its input images; others say that the Multi-head Self-Attention (MSA) is the key to preserving the robustness. In this paper, we aim to introduce a principled and unified theoretical framework to investigate such an argument on ViT's robustness. We first theoretically prove that, unlike Transformers in Natural Language Processing, ViTs are Lipschitz continuous. Then we theoretically analyze the adversarial robustness of ViTs from the perspective of the Cauchy Problem, via which we can quantify how the robustness propagates through layers. We demonstrate that the first and last layers are the critical factors to affect the robustness of ViTs. Furthermore, based on our theory, we empirically show that unlike the claims from existing research, MSA only contributes to the adversarial robustness of ViTs under weak adversarial attacks, e.g., FGSM, and surprisingly, MSA actually comprises the model's adversarial robustness under stronger attacks, e.g., PGD attacks.

摘要: 最近关于深度学习稳健性的研究表明，视觉转换器(VITS)在某些扰动下优于卷积神经网络(CNNS)，如自然腐败、敌意攻击等。一些文献认为VIT优越的稳健性来自于其输入图像的分割；另一些文献则认为多头自我注意(MSA)是保持稳健性的关键。在本文中，我们旨在引入一个原则性和统一的理论框架来研究这种关于VIT稳健性的争论。我们首先从理论上证明，与自然语言处理中的变形金刚不同，VITS是Lipschitz连续的。然后，我们从柯西问题的角度对VITS的对抗健壮性进行了理论分析，通过它我们可以量化健壮性是如何通过层传播的。我们证明了第一层和最后一层是影响VITS稳健性的关键因素。此外，基于我们的理论，我们的经验表明，与现有研究的结论不同，MSA仅有助于VITS在弱对抗攻击(如FGSM)下的对抗健壮性，并且令人惊讶的是，MSA实际上包含了该模型在更强攻击(如PGD攻击)下的对抗健壮性。



## **16. Attacking Adversarial Defences by Smoothing the Loss Landscape**

通过平滑损失图景来攻击对抗性防御 cs.LG

**SubmitDate**: 2022-08-01    [paper-pdf](http://arxiv.org/pdf/2208.00862v1)

**Authors**: Panagiotis Eustratiadis, Henry Gouk, Da Li, Timothy Hospedales

**Abstracts**: This paper investigates a family of methods for defending against adversarial attacks that owe part of their success to creating a noisy, discontinuous, or otherwise rugged loss landscape that adversaries find difficult to navigate. A common, but not universal, way to achieve this effect is via the use of stochastic neural networks. We show that this is a form of gradient obfuscation, and propose a general extension to gradient-based adversaries based on the Weierstrass transform, which smooths the surface of the loss function and provides more reliable gradient estimates. We further show that the same principle can strengthen gradient-free adversaries. We demonstrate the efficacy of our loss-smoothing method against both stochastic and non-stochastic adversarial defences that exhibit robustness due to this type of obfuscation. Furthermore, we provide analysis of how it interacts with Expectation over Transformation; a popular gradient-sampling method currently used to attack stochastic defences.

摘要: 本文研究了一系列防御对手攻击的方法，这些攻击的成功部分归因于创建了一个嘈杂的、不连续的或以其他方式崎岖的损失场景，对手发现很难导航。实现这一效果的一种常见但并不普遍的方法是通过使用随机神经网络。我们证明了这是一种梯度混淆的形式，并提出了一种基于魏尔斯特拉斯变换的对基于梯度的攻击的一般扩展，它平滑了损失函数的表面，并提供了更可靠的梯度估计。我们进一步证明，同样的原理可以加强无梯度的对手。我们证明了我们的损失平滑方法对随机和非随机对抗防御的有效性，这些防御由于这种类型的混淆而表现出稳健性。此外，我们还分析了它如何与变换上的期望相互作用，变换上的期望是目前用于攻击随机防御的一种流行的梯度抽样方法。



## **17. On the Detection of Adaptive Adversarial Attacks in Speaker Verification Systems**

说话人确认系统中自适应敌意攻击的检测 cs.CR

**SubmitDate**: 2022-08-01    [paper-pdf](http://arxiv.org/pdf/2202.05725v2)

**Authors**: Zesheng Chen

**Abstracts**: Speaker verification systems have been widely used in smart phones and Internet of things devices to identify legitimate users. In recent work, it has been shown that adversarial attacks, such as FAKEBOB, can work effectively against speaker verification systems. The goal of this paper is to design a detector that can distinguish an original audio from an audio contaminated by adversarial attacks. Specifically, our designed detector, called MEH-FEST, calculates the minimum energy in high frequencies from the short-time Fourier transform of an audio and uses it as a detection metric. Through both analysis and experiments, we show that our proposed detector is easy to implement, fast to process an input audio, and effective in determining whether an audio is corrupted by FAKEBOB attacks. The experimental results indicate that the detector is extremely effective: with near zero false positive and false negative rates for detecting FAKEBOB attacks in Gaussian mixture model (GMM) and i-vector speaker verification systems. Moreover, adaptive adversarial attacks against our proposed detector and their countermeasures are discussed and studied, showing the game between attackers and defenders.

摘要: 说话人验证系统已被广泛应用于智能手机和物联网设备中，以识别合法用户。最近的工作表明，FAKEBOB等对抗性攻击可以有效地对抗说话人验证系统。本文的目标是设计一种能够区分原始音频和被敌意攻击污染的音频的检测器。具体地说，我们设计的检测器MEH-FEST从音频的短时傅里叶变换计算高频最小能量，并将其用作检测度量。通过分析和实验表明，我们提出的检测器实现简单，处理输入音频的速度快，并能有效地判断音频是否被FAKEBOB攻击破坏。实验结果表明，该检测器对混合高斯模型(GMM)和I-向量说话人确认系统中的FAKEBOB攻击具有极高的检测效率：几乎为零的误检率和漏检率。此外，还讨论和研究了针对我们提出的检测器的自适应对抗性攻击及其对策，展示了攻击者和防御者之间的博弈。



## **18. The Geometry of Adversarial Training in Binary Classification**

二元分类中对抗性训练的几何问题 cs.LG

**SubmitDate**: 2022-08-01    [paper-pdf](http://arxiv.org/pdf/2111.13613v2)

**Authors**: Leon Bungert, Nicolás García Trillos, Ryan Murray

**Abstracts**: We establish an equivalence between a family of adversarial training problems for non-parametric binary classification and a family of regularized risk minimization problems where the regularizer is a nonlocal perimeter functional. The resulting regularized risk minimization problems admit exact convex relaxations of the type $L^1+$ (nonlocal) $\operatorname{TV}$, a form frequently studied in image analysis and graph-based learning. A rich geometric structure is revealed by this reformulation which in turn allows us to establish a series of properties of optimal solutions of the original problem, including the existence of minimal and maximal solutions (interpreted in a suitable sense), and the existence of regular solutions (also interpreted in a suitable sense). In addition, we highlight how the connection between adversarial training and perimeter minimization problems provides a novel, directly interpretable, statistical motivation for a family of regularized risk minimization problems involving perimeter/total variation. The majority of our theoretical results are independent of the distance used to define adversarial attacks.

摘要: 我们建立了一类非参数二分类的对抗性训练问题和一类正则化风险最小化问题之间的等价关系，其中正则化子是非局部周长泛函。由此产生的正则化风险最小化问题允许类型为$L^1+$(非局部)$操作符{TV}$的精确凸松弛，这是图像分析和基于图的学习中经常研究的一种形式。它揭示了原问题最优解的一系列性质，包括最小解和最大解的存在性(在适当的意义上解释)和正则解的存在(在适当的意义上解释)。此外，我们强调了对抗性训练和周长最小化问题之间的联系如何为一类涉及周长/总变异的正则化风险最小化问题提供了一种新颖的、直接可解释的统计动机。我们的大多数理论结果与用于定义对抗性攻击的距离无关。



## **19. Is current research on adversarial robustness addressing the right problem?**

目前关于对手稳健性的研究解决了正确的问题吗？ cs.CV

**SubmitDate**: 2022-07-31    [paper-pdf](http://arxiv.org/pdf/2208.00539v1)

**Authors**: Ali Borji

**Abstracts**: Short answer: Yes, Long answer: No! Indeed, research on adversarial robustness has led to invaluable insights helping us understand and explore different aspects of the problem. Many attacks and defenses have been proposed over the last couple of years. The problem, however, remains largely unsolved and poorly understood. Here, I argue that the current formulation of the problem serves short term goals, and needs to be revised for us to achieve bigger gains. Specifically, the bound on perturbation has created a somewhat contrived setting and needs to be relaxed. This has misled us to focus on model classes that are not expressive enough to begin with. Instead, inspired by human vision and the fact that we rely more on robust features such as shape, vertices, and foreground objects than non-robust features such as texture, efforts should be steered towards looking for significantly different classes of models. Maybe instead of narrowing down on imperceptible adversarial perturbations, we should attack a more general problem which is finding architectures that are simultaneously robust to perceptible perturbations, geometric transformations (e.g. rotation, scaling), image distortions (lighting, blur), and more (e.g. occlusion, shadow). Only then we may be able to solve the problem of adversarial vulnerability.

摘要: 简短的答案是：是的，长期的答案是：不！事实上，对对手健壮性的研究已经带来了宝贵的见解，帮助我们理解和探索问题的不同方面。在过去的几年里，已经提出了许多攻击和防御措施。然而，这个问题在很大程度上仍然没有得到解决，人们对此知之甚少。在这里，我认为，目前对问题的表述是为短期目标服务的，需要进行修改，以便我们实现更大的收益。具体地说，微扰的界限创造了一种有点做作的设置，需要放松。这误导了我们将注意力集中在一开始就不够有表现力的模型类上。取而代之的是，受人类视觉的启发，以及我们更依赖于形状、顶点和前景对象等稳健特征而不是纹理等非稳健特征的事实，应该努力寻找显著不同类别的模型。也许我们不应该缩小到不可感知的对抗性扰动，而应该解决一个更一般的问题，即寻找同时对可感知扰动、几何变换(例如旋转、缩放)、图像失真(照明、模糊)以及更多(例如遮挡、阴影)具有健壮性的体系结构。只有到那时，我们才可能解决对手脆弱性的问题。



## **20. DNNShield: Dynamic Randomized Model Sparsification, A Defense Against Adversarial Machine Learning**

DNNShield：动态随机化模型稀疏化，对抗对抗性机器学习 cs.CR

**SubmitDate**: 2022-07-31    [paper-pdf](http://arxiv.org/pdf/2208.00498v1)

**Authors**: Mohammad Hossein Samavatian, Saikat Majumdar, Kristin Barber, Radu Teodorescu

**Abstracts**: DNNs are known to be vulnerable to so-called adversarial attacks that manipulate inputs to cause incorrect results that can be beneficial to an attacker or damaging to the victim. Recent works have proposed approximate computation as a defense mechanism against machine learning attacks. We show that these approaches, while successful for a range of inputs, are insufficient to address stronger, high-confidence adversarial attacks. To address this, we propose DNNSHIELD, a hardware-accelerated defense that adapts the strength of the response to the confidence of the adversarial input. Our approach relies on dynamic and random sparsification of the DNN model to achieve inference approximation efficiently and with fine-grain control over the approximation error. DNNSHIELD uses the output distribution characteristics of sparsified inference compared to a dense reference to detect adversarial inputs. We show an adversarial detection rate of 86% when applied to VGG16 and 88% when applied to ResNet50, which exceeds the detection rate of the state of the art approaches, with a much lower overhead. We demonstrate a software/hardware-accelerated FPGA prototype, which reduces the performance impact of DNNSHIELD relative to software-only CPU and GPU implementations.

摘要: 众所周知，DNN容易受到所谓的对抗性攻击，即操纵输入导致不正确的结果，从而对攻击者有利或对受害者造成损害。最近的工作提出了近似计算作为一种防御机器学习攻击的机制。我们表明，这些方法虽然对一系列投入是成功的，但不足以应对更强大、高信心的对抗性攻击。为了解决这个问题，我们提出了DNNSHIELD，这是一种硬件加速防御，它根据对手输入的置信度来调整响应的强度。我们的方法依赖于DNN模型的动态和随机稀疏化，以实现高效的推理逼近和对逼近误差的细粒度控制。DNNSHIELD利用稀疏推理相对于密集引用的输出分布特性来检测敌意输入。当应用于VGG16时，敌意检测率为86%，当应用于ResNet50时，敌意检测率为88%，这超过了现有方法的检测率，并且开销要低得多。我们展示了一个软件/硬件加速的FPGA原型，它相对于纯软件的CPU和GPU实现降低了DNNSHIELD的性能影响。



## **21. Adversarial Robustness Verification and Attack Synthesis in Stochastic Systems**

随机系统中的对抗健壮性验证与攻击综合 cs.CR

To Appear, 35th IEEE Computer Security Foundations Symposium (2022)

**SubmitDate**: 2022-07-31    [paper-pdf](http://arxiv.org/pdf/2110.02125v2)

**Authors**: Lisa Oakley, Alina Oprea, Stavros Tripakis

**Abstracts**: Probabilistic model checking is a useful technique for specifying and verifying properties of stochastic systems including randomized protocols and reinforcement learning models. Existing methods rely on the assumed structure and probabilities of certain system transitions. These assumptions may be incorrect, and may even be violated by an adversary who gains control of system components.   In this paper, we develop a formal framework for adversarial robustness in systems modeled as discrete time Markov chains (DTMCs). We base our framework on existing methods for verifying probabilistic temporal logic properties and extend it to include deterministic, memoryless policies acting in Markov decision processes (MDPs). Our framework includes a flexible approach for specifying structure-preserving and non structure-preserving adversarial models. We outline a class of threat models under which adversaries can perturb system transitions, constrained by an $\varepsilon$ ball around the original transition probabilities.   We define three main DTMC adversarial robustness problems: adversarial robustness verification, maximal $\delta$ synthesis, and worst case attack synthesis. We present two optimization-based solutions to these three problems, leveraging traditional and parametric probabilistic model checking techniques. We then evaluate our solutions on two stochastic protocols and a collection of Grid World case studies, which model an agent acting in an environment described as an MDP. We find that the parametric solution results in fast computation for small parameter spaces. In the case of less restrictive (stronger) adversaries, the number of parameters increases, and directly computing property satisfaction probabilities is more scalable. We demonstrate the usefulness of our definitions and solutions by comparing system outcomes over various properties, threat models, and case studies.

摘要: 概率模型检验是描述和验证随机系统性质的一种有用技术，包括随机化协议和强化学习模型。现有的方法依赖于某些系统转变的假设结构和概率。这些假设可能是不正确的，甚至可能被控制系统组件的对手违反。在这篇文章中，我们发展了一个形式化的框架，在离散时间马尔可夫链(DTMC)建模的系统中的对手稳健性。我们的框架基于现有的概率时态逻辑属性验证方法，并将其扩展到马尔可夫决策过程(MDP)中的确定性、无记忆策略。我们的框架包括一种灵活的方法来指定结构保持和非结构保持的对抗性模型。我们概述了一类威胁模型，在该模型下，攻击者可以干扰系统的转移，并受围绕原始转移概率的$\varepsilon$球的约束。我们定义了三个主要的DTMC攻击健壮性问题：攻击健壮性验证、最大$\Delta$合成和最坏情况攻击合成。对于这三个问题，我们提出了两种基于优化的解决方案，利用传统的和参数概率模型检测技术。然后，我们在两个随机协议和一系列网格世界案例研究上评估我们的解决方案，这些案例研究对在被描述为MDP的环境中行为的代理进行建模。我们发现，对于小的参数空间，参数解导致了快速的计算。在限制较少(较强)的对手的情况下，参数的数量增加，并且直接计算属性满意概率更具可扩展性。我们通过比较各种属性、威胁模型和案例研究的系统结果来证明我们的定义和解决方案的有效性。



## **22. Robust Real-World Image Super-Resolution against Adversarial Attacks**

抵抗敌意攻击的稳健的真实世界图像超分辨率 cs.CV

ACM-MM 2021, Code:  https://github.com/lhaof/Robust-SR-against-Adversarial-Attacks

**SubmitDate**: 2022-07-31    [paper-pdf](http://arxiv.org/pdf/2208.00428v1)

**Authors**: Jiutao Yue, Haofeng Li, Pengxu Wei, Guanbin Li, Liang Lin

**Abstracts**: Recently deep neural networks (DNNs) have achieved significant success in real-world image super-resolution (SR). However, adversarial image samples with quasi-imperceptible noises could threaten deep learning SR models. In this paper, we propose a robust deep learning framework for real-world SR that randomly erases potential adversarial noises in the frequency domain of input images or features. The rationale is that on the SR task clean images or features have a different pattern from the attacked ones in the frequency domain. Observing that existing adversarial attacks usually add high-frequency noises to input images, we introduce a novel random frequency mask module that blocks out high-frequency components possibly containing the harmful perturbations in a stochastic manner. Since the frequency masking may not only destroys the adversarial perturbations but also affects the sharp details in a clean image, we further develop an adversarial sample classifier based on the frequency domain of images to determine if applying the proposed mask module. Based on the above ideas, we devise a novel real-world image SR framework that combines the proposed frequency mask modules and the proposed adversarial classifier with an existing super-resolution backbone network. Experiments show that our proposed method is more insensitive to adversarial attacks and presents more stable SR results than existing models and defenses.

摘要: 最近，深度神经网络(DNN)在真实世界图像超分辨率(SR)方面取得了显著的成功。然而，含有准不可感知噪声的对抗性图像样本可能会威胁到深度学习随机共振模型。在本文中，我们提出了一种稳健的深度学习框架，该框架可以在输入图像或特征的频域中随机消除潜在的对抗性噪声。其基本原理是，在SR任务中，干净的图像或特征在频域具有与受攻击的图像或特征不同的模式。针对现有的敌意攻击通常会在输入图像中加入高频噪声的问题，提出了一种新的随机频率掩码模块，以随机的方式屏蔽掉可能包含有害扰动的高频分量。由于频率掩蔽不仅会破坏图像中的对抗性扰动，而且会影响清晰图像中的清晰细节，因此我们进一步提出了一种基于图像频域的对抗性样本分类器，以确定是否应用所提出的掩码模块。基于上述思想，我们设计了一种新的真实图像SR框架，将所提出的频率掩码模块和所提出的对抗性分类器与现有的超分辨率骨干网络相结合。实验表明，与已有的模型和防御方法相比，我们提出的方法对敌意攻击更不敏感，并且提供了更稳定的SR结果。



## **23. Electromagnetic Signal Injection Attacks on Differential Signaling**

对差分信令的电磁信号注入攻击 cs.CR

14 pages, 15 figures

**SubmitDate**: 2022-07-31    [paper-pdf](http://arxiv.org/pdf/2208.00343v1)

**Authors**: Youqian Zhang, Kasper Rasmussen

**Abstracts**: Differential signaling is a method of data transmission that uses two complementary electrical signals to encode information. This allows a receiver to reject any noise by looking at the difference between the two signals, assuming the noise affects both signals in the same way. Many protocols such as USB, Ethernet, and HDMI use differential signaling to achieve a robust communication channel in a noisy environment. This generally works well and has led many to believe that it is infeasible to remotely inject attacking signals into such a differential pair. In this paper we challenge this assumption and show that an adversary can in fact inject malicious signals from a distance, purely using common-mode injection, i.e., injecting into both wires at the same time. We show how this allows an attacker to inject bits or even arbitrary messages into a communication line. Such an attack is a significant threat to many applications, from home security and privacy to automotive systems, critical infrastructure, or implantable medical devices; in which incorrect data or unauthorized control could cause significant damage, or even fatal accidents.   We show in detail the principles of how an electromagnetic signal can bypass the noise rejection of differential signaling, and eventually result in incorrect bits in the receiver. We show how an attacker can exploit this to achieve a successful injection of an arbitrary bit, and we analyze the success rate of injecting longer arbitrary messages. We demonstrate the attack on a real system and show that the success rate can reach as high as $90\%$. Finally, we present a case study where we wirelessly inject a message into a Controller Area Network (CAN) bus, which is a differential signaling bus protocol used in many critical applications, including the automotive and aviation sector.

摘要: 差分信令是一种数据传输方法，它使用两个互补的电信号来编码信息。这允许接收器通过查看两个信号之间的差异来抑制任何噪声，假设噪声以相同的方式影响两个信号。USB、以太网和HDMI等许多协议使用差分信令在噪声环境中实现可靠的通信通道。这通常效果良好，并导致许多人认为，将攻击信号远程注入到这样的差分对中是不可行的。在这篇文章中，我们挑战了这一假设，并证明了敌手实际上可以从远处注入恶意信号，纯粹使用共模注入，即同时注入两条线路。我们展示了这如何允许攻击者向通信线路注入比特甚至任意消息。这样的攻击对许多应用程序都是一个重大威胁，从家庭安全和隐私到汽车系统、关键基础设施或植入式医疗设备；在这些应用程序中，错误的数据或未经授权的控制可能会导致重大破坏，甚至致命的事故。我们详细展示了电磁信号如何绕过差分信号的噪声抑制，并最终导致接收器中的错误比特的原理。我们展示了攻击者如何利用这一点来成功注入任意位，并分析了注入更长的任意消息的成功率。我们在一个真实的系统上演示了攻击，并表明攻击成功率可以高达90美元。最后，我们给出了一个案例研究，在该案例中，我们将消息无线注入控制器区域网络(CAN)总线，这是一种在许多关键应用中使用的差分信号总线协议，包括汽车和航空领域。



## **24. Backdoor Attack is a Devil in Federated GAN-based Medical Image Synthesis**

后门攻击是联邦GAN医学图像合成中的一大难题 cs.CV

13 pages, 4 figures, Accepted by MICCAI 2022 SASHIMI Workshop

**SubmitDate**: 2022-07-30    [paper-pdf](http://arxiv.org/pdf/2207.00762v2)

**Authors**: Ruinan Jin, Xiaoxiao Li

**Abstracts**: Deep Learning-based image synthesis techniques have been applied in healthcare research for generating medical images to support open research. Training generative adversarial neural networks (GAN) usually requires large amounts of training data. Federated learning (FL) provides a way of training a central model using distributed data from different medical institutions while keeping raw data locally. However, FL is vulnerable to backdoor attack, an adversarial by poisoning training data, given the central server cannot access the original data directly. Most backdoor attack strategies focus on classification models and centralized domains. In this study, we propose a way of attacking federated GAN (FedGAN) by treating the discriminator with a commonly used data poisoning strategy in backdoor attack classification models. We demonstrate that adding a small trigger with size less than 0.5 percent of the original image size can corrupt the FL-GAN model. Based on the proposed attack, we provide two effective defense strategies: global malicious detection and local training regularization. We show that combining the two defense strategies yields a robust medical image generation.

摘要: 基于深度学习的图像合成技术已被应用于医疗保健研究，以生成支持开放研究的医学图像。生成对抗神经网络(GAN)的训练通常需要大量的训练数据。联合学习(FL)提供了一种使用来自不同医疗机构的分布式数据训练中央模型的方法，同时保持本地的原始数据。然而，由于中央服务器不能直接访问原始数据，FL很容易受到后门攻击，这是通过毒化训练数据而产生的敌意。大多数后门攻击策略侧重于分类模型和集中域。在这项研究中，我们提出了一种利用后门攻击分类模型中常用的数据中毒策略来处理鉴别器来攻击联邦GAN(FedGAN)的方法。我们证明，添加一个尺寸小于原始图像尺寸0.5%的小触发器可以破坏FL-GaN模型。基于提出的攻击，我们提出了两种有效的防御策略：全局恶意检测和局部训练正则化。我们表明，结合这两种防御策略可以产生稳健的医学图像生成。



## **25. Towards Privacy-Preserving, Real-Time and Lossless Feature Matching**

走向隐私保护、实时无损的特征匹配 cs.CV

**SubmitDate**: 2022-07-30    [paper-pdf](http://arxiv.org/pdf/2208.00214v1)

**Authors**: Qiang Meng, Feng Zhou

**Abstracts**: Most visual retrieval applications store feature vectors for downstream matching tasks. These vectors, from where user information can be spied out, will cause privacy leakage if not carefully protected. To mitigate privacy risks, current works primarily utilize non-invertible transformations or fully cryptographic algorithms. However, transformation-based methods usually fail to achieve satisfying matching performances while cryptosystems suffer from heavy computational overheads. In addition, secure levels of current methods should be improved to confront potential adversary attacks. To address these issues, this paper proposes a plug-in module called SecureVector that protects features by random permutations, 4L-DEC converting and existing homomorphic encryption techniques. For the first time, SecureVector achieves real-time and lossless feature matching among sanitized features, along with much higher security levels than current state-of-the-arts. Extensive experiments on face recognition, person re-identification, image retrieval, and privacy analyses demonstrate the effectiveness of our method. Given limited public projects in this field, codes of our method and implemented baselines are made open-source in https://github.com/IrvingMeng/SecureVector.

摘要: 大多数视觉检索应用存储用于下游匹配任务的特征向量。这些媒介可以窥探用户信息，如果不小心保护，将导致隐私泄露。为了减轻隐私风险，目前的工作主要使用不可逆变换或完全加密算法。然而，基于变换的方法往往不能达到令人满意的匹配性能，而密码系统的计算开销很大。此外，应改进现有方法的安全级别，以应对潜在的对手攻击。为了解决这些问题，本文提出了一种称为安全向量的插件模块，它通过随机排列、4L-DEC转换和现有的同态加密技术来保护特征。SecureVector首次实现了经过消毒的功能之间的实时和无损功能匹配，以及比当前最先进的安全级别高得多的安全级别。在人脸识别、身份识别、图像检索和隐私分析等方面的大量实验证明了该方法的有效性。考虑到这一领域的公共项目有限，我们的方法代码和实现的基线在https://github.com/IrvingMeng/SecureVector.中是开源的



## **26. Towards Bridging the gap between Empirical and Certified Robustness against Adversarial Examples**

弥合经验性和认证的对抗实例稳健性之间的差距 cs.LG

An abridged version of this work has been presented at ICLR 2021  Workshop on Security and Safety in Machine Learning Systems:  https://aisecure-workshop.github.io/aml-iclr2021/papers/2.pdf

**SubmitDate**: 2022-07-30    [paper-pdf](http://arxiv.org/pdf/2102.05096v3)

**Authors**: Jay Nandy, Sudipan Saha, Wynne Hsu, Mong Li Lee, Xiao Xiang Zhu

**Abstracts**: The current state-of-the-art defense methods against adversarial examples typically focus on improving either empirical or certified robustness. Among them, adversarially trained (AT) models produce empirical state-of-the-art defense against adversarial examples without providing any robustness guarantees for large classifiers or higher-dimensional inputs. In contrast, existing randomized smoothing based models achieve state-of-the-art certified robustness while significantly degrading the empirical robustness against adversarial examples. In this paper, we propose a novel method, called \emph{Certification through Adaptation}, that transforms an AT model into a randomized smoothing classifier during inference to provide certified robustness for $\ell_2$ norm without affecting their empirical robustness against adversarial attacks. We also propose \emph{Auto-Noise} technique that efficiently approximates the appropriate noise levels to flexibly certify the test examples using randomized smoothing technique. Our proposed \emph{Certification through Adaptation} with \emph{Auto-Noise} technique achieves an \textit{average certified radius (ACR) scores} up to $1.102$ and $1.148$ respectively for CIFAR-10 and ImageNet datasets using AT models without affecting their empirical robustness or benign accuracy. Therefore, our paper is a step towards bridging the gap between the empirical and certified robustness against adversarial examples by achieving both using the same classifier.

摘要: 当前针对敌意例子的最先进的防御方法通常侧重于提高经验性或经验证的稳健性。其中，对抗性训练(AT)模型针对对抗性实例提供了经验最新的防御，而没有为大型分类器或高维输入提供任何健壮性保证。相比之下，现有的基于随机平滑的模型实现了最先进的经过验证的稳健性，同时显著降低了对对抗性例子的经验稳健性。在本文中，我们提出了一种新的方法，称为自适应认证，该方法在推理过程中将AT模型转换为随机平滑分类器，从而在不影响其对对手攻击的经验健壮性的情况下，提供对$EELL_2$范数的认证稳健性。我们还提出了有效逼近适当噪声水平的自动噪声技术，以灵活地使用随机化平滑技术来证明测试用例。在不影响其经验稳健性和良好精度的情况下，我们提出的采用自适应技术的自适应认证技术在使用AT模型的CIFAR-10和ImageNet数据集上分别获得了高达1.102美元和1.148美元的平均认证半径分数。因此，我们的论文通过使用相同的分类器实现了经验和验证的对抗实例之间的鲁棒性之间的差距，从而迈出了一步。



## **27. Robust Trajectory Prediction against Adversarial Attacks**

对抗敌方攻击的稳健弹道预测 cs.LG

**SubmitDate**: 2022-07-29    [paper-pdf](http://arxiv.org/pdf/2208.00094v1)

**Authors**: Yulong Cao, Danfei Xu, Xinshuo Weng, Zhuoqing Mao, Anima Anandkumar, Chaowei Xiao, Marco Pavone

**Abstracts**: Trajectory prediction using deep neural networks (DNNs) is an essential component of autonomous driving (AD) systems. However, these methods are vulnerable to adversarial attacks, leading to serious consequences such as collisions. In this work, we identify two key ingredients to defend trajectory prediction models against adversarial attacks including (1) designing effective adversarial training methods and (2) adding domain-specific data augmentation to mitigate the performance degradation on clean data. We demonstrate that our method is able to improve the performance by 46% on adversarial data and at the cost of only 3% performance degradation on clean data, compared to the model trained with clean data. Additionally, compared to existing robust methods, our method can improve performance by 21% on adversarial examples and 9% on clean data. Our robust model is evaluated with a planner to study its downstream impacts. We demonstrate that our model can significantly reduce the severe accident rates (e.g., collisions and off-road driving).

摘要: 基于深度神经网络的轨迹预测是自动驾驶系统的重要组成部分。然而，这些方法容易受到对抗性攻击，导致碰撞等严重后果。在这项工作中，我们确定了两个关键因素来防御弹道预测模型的对抗性攻击，包括(1)设计有效的对抗性训练方法和(2)添加特定领域的数据增强来缓解在干净数据上的性能下降。我们证明，与使用干净数据训练的模型相比，我们的方法能够在对抗性数据上提高46%的性能，而在干净数据上的代价只有3%的性能下降。此外，与现有的稳健方法相比，我们的方法在对抗性实例上的性能提高了21%，在干净数据上的性能提高了9%。我们的稳健模型由规划者评估，以研究其下游影响。我们证明，我们的模型可以显著降低严重事故率(例如，碰撞和越野驾驶)。



## **28. Sampling Attacks on Meta Reinforcement Learning: A Minimax Formulation and Complexity Analysis**

元强化学习中的抽样攻击：一种极小极大公式及其复杂性分析 cs.LG

**SubmitDate**: 2022-07-29    [paper-pdf](http://arxiv.org/pdf/2208.00081v1)

**Authors**: Tao Li, Haozhe Lei, Quanyan Zhu

**Abstracts**: Meta reinforcement learning (meta RL), as a combination of meta-learning ideas and reinforcement learning (RL), enables the agent to adapt to different tasks using a few samples. However, this sampling-based adaptation also makes meta RL vulnerable to adversarial attacks. By manipulating the reward feedback from sampling processes in meta RL, an attacker can mislead the agent into building wrong knowledge from training experience, which deteriorates the agent's performance when dealing with different tasks after adaptation. This paper provides a game-theoretical underpinning for understanding this type of security risk. In particular, we formally define the sampling attack model as a Stackelberg game between the attacker and the agent, which yields a minimax formulation. It leads to two online attack schemes: Intermittent Attack and Persistent Attack, which enable the attacker to learn an optimal sampling attack, defined by an $\epsilon$-first-order stationary point, within $\mathcal{O}(\epsilon^{-2})$ iterations. These attack schemes freeride the learning progress concurrently without extra interactions with the environment. By corroborating the convergence results with numerical experiments, we observe that a minor effort of the attacker can significantly deteriorate the learning performance, and the minimax approach can also help robustify the meta RL algorithms.

摘要: 元强化学习作为元学习思想和强化学习的结合，使智能体能够利用少量的样本来适应不同的任务。然而，这种基于采样的自适应也使得Meta RL容易受到对手攻击。通过操纵Meta RL中采样过程的奖励反馈，攻击者可以误导代理从训练经验中建立错误的知识，从而降低代理在适应后处理不同任务的性能。本文为理解这种类型的安全风险提供了博弈论基础。特别地，我们将抽样攻击模型正式定义为攻击者和代理之间的Stackelberg博弈，从而产生极小极大公式。它导致了两种在线攻击方案：间歇攻击和持续攻击，使攻击者能够在$\mathcal{O}(\epsilon^{-2})$迭代内学习由$\epsilon$-一阶固定点定义的最优抽样攻击。这些攻击方案同时加快了学习过程，而无需与环境进行额外的交互。通过数值实验证实了收敛结果，我们观察到攻击者的微小努力会显著降低学习性能，并且极小极大方法也有助于增强Meta RL算法的健壮性。



## **29. Can We Mitigate Backdoor Attack Using Adversarial Detection Methods?**

我们可以使用对抗性检测方法来减少后门攻击吗？ cs.LG

Accepted by IEEE TDSC

**SubmitDate**: 2022-07-28    [paper-pdf](http://arxiv.org/pdf/2006.14871v2)

**Authors**: Kaidi Jin, Tianwei Zhang, Chao Shen, Yufei Chen, Ming Fan, Chenhao Lin, Ting Liu

**Abstracts**: Deep Neural Networks are well known to be vulnerable to adversarial attacks and backdoor attacks, where minor modifications on the input are able to mislead the models to give wrong results. Although defenses against adversarial attacks have been widely studied, investigation on mitigating backdoor attacks is still at an early stage. It is unknown whether there are any connections and common characteristics between the defenses against these two attacks. We conduct comprehensive studies on the connections between adversarial examples and backdoor examples of Deep Neural Networks to seek to answer the question: can we detect backdoor using adversarial detection methods. Our insights are based on the observation that both adversarial examples and backdoor examples have anomalies during the inference process, highly distinguishable from benign samples. As a result, we revise four existing adversarial defense methods for detecting backdoor examples. Extensive evaluations indicate that these approaches provide reliable protection against backdoor attacks, with a higher accuracy than detecting adversarial examples. These solutions also reveal the relations of adversarial examples, backdoor examples and normal samples in model sensitivity, activation space and feature space. This is able to enhance our understanding about the inherent features of these two attacks and the defense opportunities.

摘要: 众所周知，深度神经网络容易受到对抗性攻击和后门攻击，在这些攻击中，对输入的微小修改能够误导模型给出错误的结果。尽管针对敌意攻击的防御已经被广泛研究，但关于减轻后门攻击的调查仍处于早期阶段。目前尚不清楚针对这两种攻击的防御之间是否有任何联系和共同特征。我们对深度神经网络的对抗性实例和后门实例之间的联系进行了全面的研究，试图回答这样一个问题：我们是否可以使用对抗性检测方法来检测后门。我们的洞察是基于这样的观察，即对抗性例子和后门例子在推理过程中都有异常，与良性样本具有高度的区分性。因此，我们对现有的四种检测后门实例的对抗性防御方法进行了修改。广泛的评估表明，这些方法提供了可靠的后门攻击保护，比检测敌意示例具有更高的准确性。这些解还揭示了对抗性样本、后门样本和正常样本在模型敏感度、激活空间和特征空间中的关系。这能够增进我们对这两起袭击的内在特征和防御机会的了解。



## **30. Look Closer to Your Enemy: Learning to Attack via Teacher-student Mimicking**

走近你的敌人：通过师生模仿学习攻击 cs.CV

13 pages, 8 figures, NDSS

**SubmitDate**: 2022-07-28    [paper-pdf](http://arxiv.org/pdf/2207.13381v2)

**Authors**: Mingejie Wang, Zhiqing Tang, Sirui Li, Dingwen Xiao

**Abstracts**: This paper aims to generate realistic attack samples of person re-identification, ReID, by reading the enemy's mind (VM). In this paper, we propose a novel inconspicuous and controllable ReID attack baseline, LCYE, to generate adversarial query images. Concretely, LCYE first distills VM's knowledge via teacher-student memory mimicking in the proxy task. Then this knowledge prior acts as an explicit cipher conveying what is essential and realistic, believed by VM, for accurate adversarial misleading. Besides, benefiting from the multiple opposing task framework of LCYE, we further investigate the interpretability and generalization of ReID models from the view of the adversarial attack, including cross-domain adaption, cross-model consensus, and online learning process. Extensive experiments on four ReID benchmarks show that our method outperforms other state-of-the-art attackers with a large margin in white-box, black-box, and target attacks. Our code is now available at https://gitfront.io/r/user-3704489/mKXusqDT4ffr/LCYE/.

摘要: 本文旨在通过读取敌人的心理(Vm)来生成真实的人重新识别的攻击样本，Reid。本文提出了一种新的隐蔽可控的Reid攻击基线--LCYE，用于生成敌意查询图像。具体来说，LCYE首先通过模仿代理任务中的师生记忆来提取VM的知识。然后，这种先验知识就像一个明确的密码，传达了被VM认为是必要和现实的东西，以实现准确的对抗性误导。此外，得益于LCYE的多重对立任务框架，我们从对抗性攻击的角度进一步考察了Reid模型的可解释性和泛化，包括跨域适应、跨模型共识和在线学习过程。在四个Reid基准测试上的大量实验表明，我们的方法在白盒、黑盒和目标攻击中的性能远远优于其他最先进的攻击者。我们的代码现已在https://gitfront.io/r/user-3704489/mKXusqDT4ffr/LCYE/.上提供



## **31. Privacy-Preserving Federated Recurrent Neural Networks**

隐私保护的联邦递归神经网络 cs.CR

**SubmitDate**: 2022-07-28    [paper-pdf](http://arxiv.org/pdf/2207.13947v1)

**Authors**: Sinem Sav, Abdulrahman Diaa, Apostolos Pyrgelis, Jean-Philippe Bossuat, Jean-Pierre Hubaux

**Abstracts**: We present RHODE, a novel system that enables privacy-preserving training of and prediction on Recurrent Neural Networks (RNNs) in a federated learning setting by relying on multiparty homomorphic encryption (MHE). RHODE preserves the confidentiality of the training data, the model, and the prediction data; and it mitigates the federated learning attacks that target the gradients under a passive-adversary threat model. We propose a novel packing scheme, multi-dimensional packing, for a better utilization of Single Instruction, Multiple Data (SIMD) operations under encryption. With multi-dimensional packing, RHODE enables the efficient processing, in parallel, of a batch of samples. To avoid the exploding gradients problem, we also provide several clip-by-value approximations for enabling gradient clipping under encryption. We experimentally show that the model performance with RHODE remains similar to non-secure solutions both for homogeneous and heterogeneous data distribution among the data holders. Our experimental evaluation shows that RHODE scales linearly with the number of data holders and the number of timesteps, sub-linearly and sub-quadratically with the number of features and the number of hidden units of RNNs, respectively. To the best of our knowledge, RHODE is the first system that provides the building blocks for the training of RNNs and its variants, under encryption in a federated learning setting.

摘要: 我们提出了一种新的系统Rhode，它依靠多方同态加密(MHE)在联邦学习环境中实现对递归神经网络(RNN)的隐私保护训练和预测。Rhode保留了训练数据、模型和预测数据的机密性；它缓解了被动对手威胁模型下针对梯度的联合学习攻击。为了更好地利用加密环境下的单指令、多数据(SIMD)运算，提出了一种新的打包方案--多维打包。通过多维包装，Rhode能够并行高效地处理一批样品。为了避免爆炸的梯度问题，我们还提供了几种逐值近似的方法来实现加密下的梯度裁剪。我们的实验表明，对于数据持有者之间的同质和异质数据分布，Rhode模型的性能与非安全解决方案相似。我们的实验评估表明，Rhode与数据持有者数量和时间步数成线性关系，分别与RNN的特征数和隐含单元数成亚线性和次二次关系。据我们所知，Rhode是第一个在联合学习环境中加密的、为RNN及其变体的训练提供构建块的系统。



## **32. Label-Only Membership Inference Attack against Node-Level Graph Neural Networks**

针对节点级图神经网络的仅标签隶属度推理攻击 cs.CR

**SubmitDate**: 2022-07-27    [paper-pdf](http://arxiv.org/pdf/2207.13766v1)

**Authors**: Mauro Conti, Jiaxin Li, Stjepan Picek, Jing Xu

**Abstracts**: Graph Neural Networks (GNNs), inspired by Convolutional Neural Networks (CNNs), aggregate the message of nodes' neighbors and structure information to acquire expressive representations of nodes for node classification, graph classification, and link prediction. Previous studies have indicated that GNNs are vulnerable to Membership Inference Attacks (MIAs), which infer whether a node is in the training data of GNNs and leak the node's private information, like the patient's disease history. The implementation of previous MIAs takes advantage of the models' probability output, which is infeasible if GNNs only provide the prediction label (label-only) for the input.   In this paper, we propose a label-only MIA against GNNs for node classification with the help of GNNs' flexible prediction mechanism, e.g., obtaining the prediction label of one node even when neighbors' information is unavailable. Our attacking method achieves around 60\% accuracy, precision, and Area Under the Curve (AUC) for most datasets and GNN models, some of which are competitive or even better than state-of-the-art probability-based MIAs implemented under our environment and settings. Additionally, we analyze the influence of the sampling method, model selection approach, and overfitting level on the attack performance of our label-only MIA. Both of those factors have an impact on the attack performance. Then, we consider scenarios where assumptions about the adversary's additional dataset (shadow dataset) and extra information about the target model are relaxed. Even in those scenarios, our label-only MIA achieves a better attack performance in most cases. Finally, we explore the effectiveness of possible defenses, including Dropout, Regularization, Normalization, and Jumping knowledge. None of those four defenses prevent our attack completely.

摘要: 图神经网络(GNN)受卷积神经网络(CNN)的启发，将节点的邻居信息和结构信息聚合在一起，得到节点的表达形式，用于节点分类、图分类和链接预测。以往的研究表明，GNN容易受到成员关系推断攻击(MIA)，MIA可以推断节点是否在GNN的训练数据中，并泄露节点的私有信息，如患者的病史。以前的MIA的实现利用了模型的概率输出，如果GNN只为输入提供预测标签(仅标签)，这是不可行的。本文利用GNN灵活的预测机制，提出了一种针对GNN的只有标签的MIA用于节点分类，例如，即使在邻居信息不可用的情况下也能获得一个节点的预测标签。对于大多数数据集和GNN模型，我们的攻击方法达到了大约60%的准确率、精确度和曲线下面积(AUC)，其中一些可以与在我们的环境和设置下实现的最先进的基于概率的MIA相媲美，甚至更好。此外，我们还分析了采样方法、模型选择方法和过拟合程度对仅标签MIA攻击性能的影响。这两个因素都会对攻击性能产生影响。然后，我们考虑放松对对手的额外数据集(阴影数据集)和关于目标模型的额外信息的假设。即使在这些情况下，我们的仅标签MIA在大多数情况下也可以实现更好的攻击性能。最后，我们探讨了可能的防御措施的有效性，包括丢弃、正则化、正规化和跳跃知识。这四种防御手段都不能完全阻止我们的进攻。



## **33. Membership Inference Attacks via Adversarial Examples**

基于对抗性例子的成员关系推理攻击 cs.LG

**SubmitDate**: 2022-07-27    [paper-pdf](http://arxiv.org/pdf/2207.13572v1)

**Authors**: Hamid Jalalzai, Elie Kadoche, Rémi Leluc, Vincent Plassier

**Abstracts**: The raise of machine learning and deep learning led to significant improvement in several domains. This change is supported by both the dramatic rise in computation power and the collection of large datasets. Such massive datasets often include personal data which can represent a threat to privacy. Membership inference attacks are a novel direction of research which aims at recovering training data used by a learning algorithm. In this paper, we develop a mean to measure the leakage of training data leveraging a quantity appearing as a proxy of the total variation of a trained model near its training samples. We extend our work by providing a novel defense mechanism. Our contributions are supported by empirical evidence through convincing numerical experiments.

摘要: 机器学习和深度学习的兴起导致了几个领域的显著改善。计算能力的戏剧性增长和大型数据集的收集都支持这种变化。如此庞大的数据集通常包括可能对隐私构成威胁的个人数据。隶属度推理攻击是一个新的研究方向，其目的是恢复学习算法所使用的训练数据。在本文中，我们开发了一种方法来衡量训练数据的泄漏，该方法利用一个量来衡量训练模型在其训练样本附近的总变异。我们通过提供一种新颖的防御机制来扩展我们的工作。通过令人信服的数值实验，我们的贡献得到了经验证据的支持。



## **34. Robust Textual Embedding against Word-level Adversarial Attacks**

抵抗词级敌意攻击的稳健文本嵌入 cs.CL

Accepted by UAI 2022, code is available at  https://github.com/JHL-HUST/FTML

**SubmitDate**: 2022-07-27    [paper-pdf](http://arxiv.org/pdf/2202.13817v2)

**Authors**: Yichen Yang, Xiaosen Wang, Kun He

**Abstracts**: We attribute the vulnerability of natural language processing models to the fact that similar inputs are converted to dissimilar representations in the embedding space, leading to inconsistent outputs, and we propose a novel robust training method, termed Fast Triplet Metric Learning (FTML). Specifically, we argue that the original sample should have similar representation with its adversarial counterparts and distinguish its representation from other samples for better robustness. To this end, we adopt the triplet metric learning into the standard training to pull words closer to their positive samples (i.e., synonyms) and push away their negative samples (i.e., non-synonyms) in the embedding space. Extensive experiments demonstrate that FTML can significantly promote the model robustness against various advanced adversarial attacks while keeping competitive classification accuracy on original samples. Besides, our method is efficient as it only needs to adjust the embedding and introduces very little overhead on the standard training. Our work shows great potential of improving the textual robustness through robust word embedding.

摘要: 我们将自然语言处理模型的脆弱性归因于相似的输入在嵌入空间中被转换为不相似的表示，从而导致输出不一致，并提出了一种新的稳健训练方法，称为快速三重度量学习(Fast Triplet Metric Learning，FTML)。具体地说，我们认为原始样本应该具有与对手样本相似的表示，并将其表示与其他样本区分开来，以获得更好的稳健性。为此，我们将三元组度量学习引入标准训练中，将单词拉近其正样本(即同义词)，并在嵌入空间中推开其负样本(即非同义词)。大量实验表明，FTML能够显著提高模型对各种高级对抗性攻击的稳健性，同时保持对原始样本的竞争性分类精度。此外，我们的方法是有效的，因为它只需要调整嵌入，并且对标准训练的开销很小。我们的工作显示了通过稳健的词嵌入来提高文本稳健性的巨大潜力。



## **35. Improved and Interpretable Defense to Transferred Adversarial Examples by Jacobian Norm with Selective Input Gradient Regularization**

基于选择输入梯度正则化的雅可比范数对转移对抗性实例的改进和可解释防御 cs.LG

Under review

**SubmitDate**: 2022-07-27    [paper-pdf](http://arxiv.org/pdf/2207.13036v2)

**Authors**: Deyin Liu, Lin Wu, Farid Boussaid, Mohammed Bennamoun

**Abstracts**: Deep neural networks (DNNs) are known to be vulnerable to adversarial examples that are crafted with imperceptible perturbations, i.e., a small change in an input image can induce a mis-classification, and thus threatens the reliability of deep learning based deployment systems. Adversarial training (AT) is often adopted to improve the robustness of DNNs through training a mixture of corrupted and clean data. However, most of AT based methods are ineffective in dealing with \textit{transferred adversarial examples} which are generated to fool a wide spectrum of defense models, and thus cannot satisfy the generalization requirement raised in real-world scenarios. Moreover, adversarially training a defense model in general cannot produce interpretable predictions towards the inputs with perturbations, whilst a highly interpretable robust model is required by different domain experts to understand the behaviour of a DNN. In this work, we propose an approach based on Jacobian norm and Selective Input Gradient Regularization (J-SIGR), which suggests the linearized robustness through Jacobian normalization and also regularizes the perturbation-based saliency maps to imitate the model's interpretable predictions. As such, we achieve both the improved defense and high interpretability of DNNs. Finally, we evaluate our method across different architectures against powerful adversarial attacks. Experiments demonstrate that the proposed J-SIGR confers improved robustness against transferred adversarial attacks, and we also show that the predictions from the neural network are easy to interpret.

摘要: 众所周知，深度神经网络(DNN)容易受到带有不可察觉扰动的敌意示例的影响，即输入图像的微小变化就会导致误分类，从而威胁到基于深度学习的部署系统的可靠性。为了提高DNN的鲁棒性，经常采用对抗性训练(AT)，方法是训练一组受破坏的和一组干净的数据。然而，大多数基于AT的方法都不能有效地处理为愚弄各种防御模型而生成的文本传递的对抗性实例，从而不能满足现实场景中提出的泛化要求。此外，对抗性地训练防御模型一般不能产生对带有扰动的输入的可解释预测，而不同领域的专家需要高度可解释的稳健模型来理解DNN的行为。在这项工作中，我们提出了一种基于雅可比范数和选择性输入梯度正则化(J-SIGR)的方法，该方法通过雅可比归一化来证明线性化的稳健性，并对基于扰动的显著图进行正则化来模拟模型的可解释预测。因此，我们实现了DNN的改进的防御性和高度的可解释性。最后，我们在不同的体系结构上对我们的方法进行了评估，以对抗强大的对手攻击。实验表明，所提出的J-SIGR算法对转移攻击具有较好的稳健性，并且神经网络的预测结果易于解释。



## **36. Point Cloud Attacks in Graph Spectral Domain: When 3D Geometry Meets Graph Signal Processing**

图谱域中的点云攻击：当3D几何遇到图信号处理时 cs.CV

arXiv admin note: substantial text overlap with arXiv:2202.07261

**SubmitDate**: 2022-07-27    [paper-pdf](http://arxiv.org/pdf/2207.13326v1)

**Authors**: Daizong Liu, Wei Hu, Xin Li

**Abstracts**: With the increasing attention in various 3D safety-critical applications, point cloud learning models have been shown to be vulnerable to adversarial attacks. Although existing 3D attack methods achieve high success rates, they delve into the data space with point-wise perturbation, which may neglect the geometric characteristics. Instead, we propose point cloud attacks from a new perspective -- the graph spectral domain attack, aiming to perturb graph transform coefficients in the spectral domain that corresponds to varying certain geometric structure. Specifically, leveraging on graph signal processing, we first adaptively transform the coordinates of points onto the spectral domain via graph Fourier transform (GFT) for compact representation. Then, we analyze the influence of different spectral bands on the geometric structure, based on which we propose to perturb the GFT coefficients via a learnable graph spectral filter. Considering the low-frequency components mainly contribute to the rough shape of the 3D object, we further introduce a low-frequency constraint to limit perturbations within imperceptible high-frequency components. Finally, the adversarial point cloud is generated by transforming the perturbed spectral representation back to the data domain via the inverse GFT. Experimental results demonstrate the effectiveness of the proposed attack in terms of both the imperceptibility and attack success rates.

摘要: 随着各种3D安全关键应用的日益关注，点云学习模型已被证明容易受到敌意攻击。现有的3D攻击方法虽然成功率很高，但都是以逐点扰动的方式深入数据空间，可能忽略了数据的几何特征。相反，我们从一个新的角度提出了点云攻击--图谱域攻击，目的是扰动对应于改变某些几何结构的谱域中的图变换系数。具体地说，利用图形信号处理，我们首先通过图形傅里叶变换(GFT)将点的坐标自适应地变换到谱域上以进行紧凑表示。然后，我们分析了不同谱带对几何结构的影响，并在此基础上提出了通过可学习的图谱滤波器来扰动GFT系数。考虑到低频分量主要影响三维物体的粗略形状，我们进一步引入了低频约束来限制不可察觉的高频分量内的扰动。最后，通过逆GFT将扰动后的谱表示变换回数据域，生成对抗性点云。实验结果表明，该攻击在不可感知性和攻击成功率方面都是有效的。



## **37. Perception-Aware Attack: Creating Adversarial Music via Reverse-Engineering Human Perception**

感知攻击：通过逆向工程人类感知创造对抗性音乐 cs.SD

ACM CCS 2022

**SubmitDate**: 2022-07-26    [paper-pdf](http://arxiv.org/pdf/2207.13192v1)

**Authors**: Rui Duan, Zhe Qu, Shangqing Zhao, Leah Ding, Yao Liu, Zhuo Lu

**Abstracts**: Recently, adversarial machine learning attacks have posed serious security threats against practical audio signal classification systems, including speech recognition, speaker recognition, and music copyright detection. Previous studies have mainly focused on ensuring the effectiveness of attacking an audio signal classifier via creating a small noise-like perturbation on the original signal. It is still unclear if an attacker is able to create audio signal perturbations that can be well perceived by human beings in addition to its attack effectiveness. This is particularly important for music signals as they are carefully crafted with human-enjoyable audio characteristics.   In this work, we formulate the adversarial attack against music signals as a new perception-aware attack framework, which integrates human study into adversarial attack design. Specifically, we conduct a human study to quantify the human perception with respect to a change of a music signal. We invite human participants to rate their perceived deviation based on pairs of original and perturbed music signals, and reverse-engineer the human perception process by regression analysis to predict the human-perceived deviation given a perturbed signal. The perception-aware attack is then formulated as an optimization problem that finds an optimal perturbation signal to minimize the prediction of perceived deviation from the regressed human perception model. We use the perception-aware framework to design a realistic adversarial music attack against YouTube's copyright detector. Experiments show that the perception-aware attack produces adversarial music with significantly better perceptual quality than prior work.

摘要: 近年来，对抗性机器学习攻击对语音识别、说话人识别、音乐版权检测等实用音频信号分类系统构成了严重的安全威胁。以前的研究主要集中在通过在原始信号上产生类似噪声的小扰动来确保攻击音频信号分类器的有效性。目前还不清楚攻击者是否能够制造出人类能够很好地感知的音频信号扰动，以及它的攻击效率。这对于音乐信号尤其重要，因为它们是精心制作的，具有人类享受的音频特征。在这项工作中，我们将对音乐信号的对抗性攻击描述为一种新的感知感知攻击框架，将人类学习融入到对抗性攻击设计中。具体地说，我们进行了一项人类研究，以量化人类对音乐信号变化的感知。我们邀请人类参与者根据原始和扰动音乐信号对他们的感知偏差进行评级，并通过回归分析反向工程人类感知过程，以预测给定扰动信号的人感知偏差。然后，感知攻击被描述为一个优化问题，该优化问题找到最优扰动信号，以最小化对回归的人类感知模型的感知偏差的预测。我们使用感知感知框架设计了一个针对YouTube版权检测器的现实对抗性音乐攻击。实验表明，感知攻击产生的对抗性音乐的感知质量明显好于以往的工作。



## **38. FlashSyn: Flash Loan Attack Synthesis via Counter Example Driven Approximation**

FlashSyn：基于反例驱动近似的闪贷攻击合成 cs.PL

29 pages, 8 figures, technical report

**SubmitDate**: 2022-07-26    [paper-pdf](http://arxiv.org/pdf/2206.10708v2)

**Authors**: Zhiyang Chen, Sidi Mohamed Beillahi, Fan Long

**Abstracts**: In decentralized finance (DeFi) ecosystem, lenders can offer flash loans to borrowers, i.e., loans that are only valid within a blockchain transaction and must be repaid with some fees by the end of that transaction. Unlike normal loans, flash loans allow borrowers to borrow a large amount of assets without upfront collaterals deposits. Malicious adversaries can use flash loans to gather large amount of assets to launch costly exploitations targeting DeFi protocols. In this paper, we introduce a new framework for automated synthesis of adversarial contracts that exploit DeFi protocols using flash loans. To bypass the complexity of a DeFi protocol, we propose a new technique to approximate the DeFi protocol functional behaviors using numerical methods (polynomial linear regression and nearest-neighbor interpolation). We then construct an optimization query using the approximated functions of the DeFi protocol to find an adversarial attack constituted of a sequence of functions invocations with optimal parameters that gives the maximum profit. To improve the accuracy of the approximation, we propose a new counterexamples-driven approximation refinement technique. We implement our framework in a tool called FlashSyn. We evaluate FlashSyn on 12 DeFi protocols that were victims to flash loan attacks and DeFi protocols from Damn Vulnerable DeFi challenges. FlashSyn automatically synthesizes an adversarial attack for each one of them.

摘要: 在去中心化金融(Defi)生态系统中，贷款人可以向借款人提供闪存贷款，即仅在区块链交易中有效且必须在该交易结束前支付一定费用的贷款。与正常贷款不同，闪付贷款允许借款人借入大量资产，而无需预付抵押金。恶意攻击者可以使用闪存贷款来收集大量资产，以发起针对Defi协议的代价高昂的攻击。在这篇文章中，我们介绍了一个新的框架，用于自动合成利用闪存贷款的Defi协议的对抗性合同。为了绕过DEFI协议的复杂性，我们提出了一种利用数值方法(多项式线性回归和最近邻内插)来逼近DEFI协议功能行为的新技术。然后，我们使用DEFI协议的近似函数构造一个优化查询，以找到由一系列具有最优参数的函数调用组成的对抗性攻击，从而给出最大利润。为了提高逼近的精度，我们提出了一种新的反例驱动的逼近求精技术。我们在一个名为FlashSyn的工具中实现我们的框架。我们对FlashSyn的12个Defi协议进行了评估，这些协议是闪电贷款攻击的受害者，并且是来自Damn Vulnerable Defi Challenges的Defi协议。FlashSyn会自动为它们中的每一个合成一次对抗性攻击。



## **39. Exploring the Unprecedented Privacy Risks of the Metaverse**

探索Metverse前所未有的隐私风险 cs.CR

**SubmitDate**: 2022-07-26    [paper-pdf](http://arxiv.org/pdf/2207.13176v1)

**Authors**: Vivek Nair, Gonzalo Munilla Garrido, Dawn Song

**Abstracts**: Thirty study participants playtested an innocent-looking "escape room" game in virtual reality (VR). Behind the scenes, an adversarial program had accurately inferred over 25 personal data attributes, from anthropometrics like height and wingspan to demographics like age and gender, within just a few minutes of gameplay. As notoriously data-hungry companies become increasingly involved in VR development, this experimental scenario may soon represent a typical VR user experience. While virtual telepresence applications (and the so-called "metaverse") have recently received increased attention and investment from major tech firms, these environments remain relatively under-studied from a security and privacy standpoint. In this work, we illustrate how VR attackers can covertly ascertain dozens of personal data attributes from seemingly-anonymous users of popular metaverse applications like VRChat. These attackers can be as simple as other VR users without special privilege, and the potential scale and scope of this data collection far exceed what is feasible within traditional mobile and web applications. We aim to shed light on the unique privacy risks of the metaverse, and provide the first holistic framework for understanding intrusive data harvesting attacks in these emerging VR ecosystems.

摘要: 30名研究参与者在虚拟现实(VR)中玩了一个看起来很无辜的“逃生室”游戏。在幕后，一个对抗性的程序在玩游戏的短短几分钟内就准确地推断出了25个人的数据属性，从身高和翼展等人体测量数据到年龄和性别等人口统计数据。随着以渴望数据著称的公司越来越多地参与到VR开发中来，这种实验场景可能很快就会代表一种典型的VR用户体验。虽然虚拟远程呈现应用(以及所谓的虚拟现实)最近得到了主要科技公司越来越多的关注和投资，但从安全和隐私的角度来看，这些环境的研究仍然相对较少。在这项工作中，我们展示了VR攻击者如何从VRChat等流行虚拟世界应用程序的看似匿名的用户那里秘密确定数十个个人数据属性。这些攻击者可以像其他没有特殊权限的VR用户一样简单，而且这种数据收集的潜在规模和范围远远超出了传统移动和网络应用程序中的可行范围。我们的目标是阐明虚拟世界独特的隐私风险，并提供第一个整体框架，以了解这些新兴的虚拟现实生态系统中的侵入性数据收集攻击。



## **40. LGV: Boosting Adversarial Example Transferability from Large Geometric Vicinity**

LGV：增强来自大几何范围的对抗性范例的可转移性 cs.LG

Accepted at ECCV 2022

**SubmitDate**: 2022-07-26    [paper-pdf](http://arxiv.org/pdf/2207.13129v1)

**Authors**: Martin Gubri, Maxime Cordy, Mike Papadakis, Yves Le Traon, Koushik Sen

**Abstracts**: We propose transferability from Large Geometric Vicinity (LGV), a new technique to increase the transferability of black-box adversarial attacks. LGV starts from a pretrained surrogate model and collects multiple weight sets from a few additional training epochs with a constant and high learning rate. LGV exploits two geometric properties that we relate to transferability. First, models that belong to a wider weight optimum are better surrogates. Second, we identify a subspace able to generate an effective surrogate ensemble among this wider optimum. Through extensive experiments, we show that LGV alone outperforms all (combinations of) four established test-time transformations by 1.8 to 59.9 percentage points. Our findings shed new light on the importance of the geometry of the weight space to explain the transferability of adversarial examples.

摘要: 为了提高黑盒对抗攻击的可转移性，我们提出了大几何邻域可转移性(LGV)的新技术。LGV从一个预先训练的代理模型开始，从几个额外的训练时期收集多个权值集，具有恒定和高的学习率。LGV利用了我们与可转移性相关的两个几何属性。首先，属于更广泛的重量最优的模型是更好的替代品。其次，我们在这个更广泛的最优解中识别出一个能够产生有效代理集成的子空间。通过广泛的实验，我们表明LGV本身就比所有四个已建立的测试时间转换(组合)高出1.8到59.9个百分点。我们的发现揭示了权重空间的几何对于解释对抗性例子的可转移性的重要性。



## **41. Making Corgis Important for Honeycomb Classification: Adversarial Attacks on Concept-based Explainability Tools**

让柯基对蜂巢分类变得重要：对基于概念的可解释性工具的对抗性攻击 cs.LG

AdvML Frontiers 2022 @ ICML 2022 workshop

**SubmitDate**: 2022-07-26    [paper-pdf](http://arxiv.org/pdf/2110.07120v2)

**Authors**: Davis Brown, Henry Kvinge

**Abstracts**: Methods for model explainability have become increasingly critical for testing the fairness and soundness of deep learning. Concept-based interpretability techniques, which use a small set of human-interpretable concept exemplars in order to measure the influence of a concept on a model's internal representation of input, are an important thread in this line of research. In this work we show that these explainability methods can suffer the same vulnerability to adversarial attacks as the models they are meant to analyze. We demonstrate this phenomenon on two well-known concept-based interpretability methods: TCAV and faceted feature visualization. We show that by carefully perturbing the examples of the concept that is being investigated, we can radically change the output of the interpretability method. The attacks that we propose can either induce positive interpretations (polka dots are an important concept for a model when classifying zebras) or negative interpretations (stripes are not an important factor in identifying images of a zebra). Our work highlights the fact that in safety-critical applications, there is need for security around not only the machine learning pipeline but also the model interpretation process.

摘要: 对于测试深度学习的公平性和稳健性，模型可解释性的方法变得越来越重要。基于概念的可解释性技术是这一研究领域的一条重要线索，它使用一小部分人类可解释的概念样本来衡量概念对模型输入的内部表示的影响。在这项工作中，我们表明这些可解释性方法可以遭受与它们要分析的模型相同的对抗性攻击漏洞。我们在两种著名的基于概念的可解释性方法上演示了这一现象：TCAV和刻面特征可视化。我们证明，通过仔细地扰动正在研究的概念的例子，我们可以从根本上改变可解释性方法的输出。我们提出的攻击既可以引起积极的解释(斑马分类时，圆点是模型的重要概念)，也可以引起消极的解释(斑马的条纹不是识别斑马图像的重要因素)。我们的工作突出了这样一个事实，即在安全关键型应用程序中，不仅需要围绕机器学习管道，而且需要围绕模型解释过程进行安全保护。



## **42. TnT Attacks! Universal Naturalistic Adversarial Patches Against Deep Neural Network Systems**

TNT攻击！针对深度神经网络系统的普遍自然主义对抗性补丁 cs.CV

Accepted for publication in the IEEE Transactions on Information  Forensics & Security (TIFS)

**SubmitDate**: 2022-07-26    [paper-pdf](http://arxiv.org/pdf/2111.09999v2)

**Authors**: Bao Gia Doan, Minhui Xue, Shiqing Ma, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstracts**: Deep neural networks are vulnerable to attacks from adversarial inputs and, more recently, Trojans to misguide or hijack the model's decision. We expose the existence of an intriguing class of spatially bounded, physically realizable, adversarial examples -- Universal NaTuralistic adversarial paTches -- we call TnTs, by exploring the superset of the spatially bounded adversarial example space and the natural input space within generative adversarial networks. Now, an adversary can arm themselves with a patch that is naturalistic, less malicious-looking, physically realizable, highly effective achieving high attack success rates, and universal. A TnT is universal because any input image captured with a TnT in the scene will: i) misguide a network (untargeted attack); or ii) force the network to make a malicious decision (targeted attack). Interestingly, now, an adversarial patch attacker has the potential to exert a greater level of control -- the ability to choose a location-independent, natural-looking patch as a trigger in contrast to being constrained to noisy perturbations -- an ability is thus far shown to be only possible with Trojan attack methods needing to interfere with the model building processes to embed a backdoor at the risk discovery; but, still realize a patch deployable in the physical world. Through extensive experiments on the large-scale visual classification task, ImageNet with evaluations across its entire validation set of 50,000 images, we demonstrate the realistic threat from TnTs and the robustness of the attack. We show a generalization of the attack to create patches achieving higher attack success rates than existing state-of-the-art methods. Our results show the generalizability of the attack to different visual classification tasks (CIFAR-10, GTSRB, PubFig) and multiple state-of-the-art deep neural networks such as WideResnet50, Inception-V3 and VGG-16.

摘要: 深度神经网络很容易受到敌意输入的攻击，最近还受到特洛伊木马的攻击，以误导或劫持模型的决策。我们通过探索空间受限的对抗性实例空间和生成性对抗性网络中的自然输入空间的超集，揭示了一类有趣的空间有界的、物理上可实现的对抗性例子的存在--通用的自然主义对抗性斑块--我们称之为TNTs。现在，对手可以用一个自然主义的、看起来不那么恶毒的、物理上可实现的、高效的、实现高攻击成功率和通用性的补丁来武装自己。TNT是通用的，因为在场景中使用TNT捕获的任何输入图像将：i)误导网络(非定向攻击)；或ii)迫使网络做出恶意决策(定向攻击)。有趣的是，现在，敌意补丁攻击者有可能施加更高级别的控制--选择与位置无关的、看起来自然的补丁作为触发器的能力，而不是受限于嘈杂的干扰--到目前为止，这种能力被证明只有在需要干扰模型构建过程以在风险发现时嵌入后门的特洛伊木马攻击方法中才是可能的；但是，仍然实现了可在物理世界中部署的补丁。通过对大规模视觉分类任务ImageNet的大量实验，以及对其50,000张图像的整个验证集的评估，我们展示了TNT的现实威胁和攻击的健壮性。我们展示了创建补丁的攻击的泛化，实现了比现有最先进的方法更高的攻击成功率。实验结果表明，该攻击对不同的视觉分类任务(CIFAR-10、GTSRB、PubFig)和多种最新的深度神经网络如WideResnet50、Inception-V3和VGG-16具有较强的泛化能力。



## **43. Verification-Aided Deep Ensemble Selection**

辅助验证的深度集成选择 cs.LG

To appear in FMCAD 2022

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2202.03898v2)

**Authors**: Guy Amir, Tom Zelazny, Guy Katz, Michael Schapira

**Abstracts**: Deep neural networks (DNNs) have become the technology of choice for realizing a variety of complex tasks. However, as highlighted by many recent studies, even an imperceptible perturbation to a correctly classified input can lead to misclassification by a DNN. This renders DNNs vulnerable to strategic input manipulations by attackers, and also oversensitive to environmental noise.   To mitigate this phenomenon, practitioners apply joint classification by an *ensemble* of DNNs. By aggregating the classification outputs of different individual DNNs for the same input, ensemble-based classification reduces the risk of misclassifications due to the specific realization of the stochastic training process of any single DNN. However, the effectiveness of a DNN ensemble is highly dependent on its members *not simultaneously erring* on many different inputs.   In this case study, we harness recent advances in DNN verification to devise a methodology for identifying ensemble compositions that are less prone to simultaneous errors, even when the input is adversarially perturbed -- resulting in more robustly-accurate ensemble-based classification.   Our proposed framework uses a DNN verifier as a backend, and includes heuristics that help reduce the high complexity of directly verifying ensembles. More broadly, our work puts forth a novel universal objective for formal verification that can potentially improve the robustness of real-world, deep-learning-based systems across a variety of application domains.

摘要: 深度神经网络(DNN)已经成为实现各种复杂任务的首选技术。然而，正如最近的许多研究所强调的那样，即使是对正确分类的输入进行了不可察觉的扰动，也可能导致DNN的错误分类。这使得DNN容易受到攻击者的战略性输入操纵，并且对环境噪声过于敏感。为了缓解这一现象，从业者根据DNN的“集合”进行联合分类。通过聚合不同个体DNN对同一输入的分类输出，基于集成的分类降低了由于任意单个DNN的随机训练过程的具体实现而导致的误分类风险。然而，DNN合奏的有效性高度依赖于其成员，而不是同时在许多不同的输入上出错。在这个案例研究中，我们利用DNN验证方面的最新进展来设计一种方法，用于识别不太容易同时出错的集成成分，即使在输入受到相反的扰动时也是如此--从而产生更健壮的基于集成的分类。我们提出的框架使用DNN验证器作为后端，并包括有助于降低直接验证集成的高复杂性的启发式算法。更广泛地说，我们的工作为形式验证提出了一个新的通用目标，可以潜在地提高现实世界中基于深度学习的系统在各种应用领域的健壮性。



## **44. $p$-DkNN: Out-of-Distribution Detection Through Statistical Testing of Deep Representations**

$p$-DkNN：基于深度表示统计测试的失配检测 cs.LG

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2207.12545v1)

**Authors**: Adam Dziedzic, Stephan Rabanser, Mohammad Yaghini, Armin Ale, Murat A. Erdogdu, Nicolas Papernot

**Abstracts**: The lack of well-calibrated confidence estimates makes neural networks inadequate in safety-critical domains such as autonomous driving or healthcare. In these settings, having the ability to abstain from making a prediction on out-of-distribution (OOD) data can be as important as correctly classifying in-distribution data. We introduce $p$-DkNN, a novel inference procedure that takes a trained deep neural network and analyzes the similarity structures of its intermediate hidden representations to compute $p$-values associated with the end-to-end model prediction. The intuition is that statistical tests performed on latent representations can serve not only as a classifier, but also offer a statistically well-founded estimation of uncertainty. $p$-DkNN is scalable and leverages the composition of representations learned by hidden layers, which makes deep representation learning successful. Our theoretical analysis builds on Neyman-Pearson classification and connects it to recent advances in selective classification (reject option). We demonstrate advantageous trade-offs between abstaining from predicting on OOD inputs and maintaining high accuracy on in-distribution inputs. We find that $p$-DkNN forces adaptive attackers crafting adversarial examples, a form of worst-case OOD inputs, to introduce semantically meaningful changes to the inputs.

摘要: 缺乏经过良好校准的置信度估计，使得神经网络在自动驾驶或医疗保健等安全关键领域不够充分。在这些设置中，能够避免对分布外(OOD)数据进行预测与正确地对分布内数据进行分类一样重要。我们介绍了一种新的推理过程$p$-DkNN，它利用训练好的深度神经网络并分析其中间隐含表示的相似结构来计算与端到端模型预测相关的$p$值。人们的直觉是，对潜在表征进行的统计测试不仅可以作为分类器，还可以提供对不确定性的统计上有充分依据的估计。$p$-DkNN是可伸缩的，并利用隐藏层学习的表示的组合，这使得深度表示学习成功。我们的理论分析建立在Neyman-Pearson分类的基础上，并将其与选择性分类(拒绝选项)的最新进展联系起来。我们展示了在避免预测OOD输入和保持分布内输入的高准确性之间的有利权衡。我们发现，$p$-DkNN迫使自适应攻击者精心制作敌意示例，这是最坏情况下OOD输入的一种形式，以对输入进行语义上有意义的更改。



## **45. TAFIM: Targeted Adversarial Attacks against Facial Image Manipulations**

TAFIM：针对面部图像处理的有针对性的对抗性攻击 cs.CV

(ECCV 2022 Paper) Video: https://youtu.be/11VMOJI7tKg Project Page:  https://shivangi-aneja.github.io/projects/tafim/

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2112.09151v2)

**Authors**: Shivangi Aneja, Lev Markhasin, Matthias Niessner

**Abstracts**: Face manipulation methods can be misused to affect an individual's privacy or to spread disinformation. To this end, we introduce a novel data-driven approach that produces image-specific perturbations which are embedded in the original images. The key idea is that these protected images prevent face manipulation by causing the manipulation model to produce a predefined manipulation target (uniformly colored output image in our case) instead of the actual manipulation. In addition, we propose to leverage differentiable compression approximation, hence making generated perturbations robust to common image compression. In order to prevent against multiple manipulation methods simultaneously, we further propose a novel attention-based fusion of manipulation-specific perturbations. Compared to traditional adversarial attacks that optimize noise patterns for each image individually, our generalized model only needs a single forward pass, thus running orders of magnitude faster and allowing for easy integration in image processing stacks, even on resource-constrained devices like smartphones.

摘要: 面部处理方法可能被滥用来影响个人隐私或传播虚假信息。为此，我们引入了一种新的数据驱动方法，该方法产生嵌入在原始图像中的特定于图像的扰动。其关键思想是，这些受保护的图像通过使操纵模型产生预定义的操纵目标(在我们的例子中为均匀着色的输出图像)而不是实际的操纵来防止面部操纵。此外，我们建议利用可微压缩近似，从而使所产生的扰动对普通图像压缩具有健壮性。为了防止多种操作方法同时出现，我们进一步提出了一种新的基于注意力的操作特定扰动融合方法。与分别优化每个图像的噪声模式的传统对抗性攻击相比，我们的通用模型只需要一次前向传递，因此运行速度快几个数量级，并允许轻松集成到图像处理堆栈中，即使在智能手机等资源受限的设备上也是如此。



## **46. SegPGD: An Effective and Efficient Adversarial Attack for Evaluating and Boosting Segmentation Robustness**

SegPGD：一种评估和提高分割健壮性的高效对抗性攻击 cs.CV

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2207.12391v1)

**Authors**: Jindong Gu, Hengshuang Zhao, Volker Tresp, Philip Torr

**Abstracts**: Deep neural network-based image classifications are vulnerable to adversarial perturbations. The image classifications can be easily fooled by adding artificial small and imperceptible perturbations to input images. As one of the most effective defense strategies, adversarial training was proposed to address the vulnerability of classification models, where the adversarial examples are created and injected into training data during training. The attack and defense of classification models have been intensively studied in past years. Semantic segmentation, as an extension of classifications, has also received great attention recently. Recent work shows a large number of attack iterations are required to create effective adversarial examples to fool segmentation models. The observation makes both robustness evaluation and adversarial training on segmentation models challenging. In this work, we propose an effective and efficient segmentation attack method, dubbed SegPGD. Besides, we provide a convergence analysis to show the proposed SegPGD can create more effective adversarial examples than PGD under the same number of attack iterations. Furthermore, we propose to apply our SegPGD as the underlying attack method for segmentation adversarial training. Since SegPGD can create more effective adversarial examples, the adversarial training with our SegPGD can boost the robustness of segmentation models. Our proposals are also verified with experiments on popular Segmentation model architectures and standard segmentation datasets.

摘要: 基于深度神经网络的图像分类容易受到对抗性扰动的影响。通过在输入图像中添加人为的微小和不可察觉的扰动，可以很容易地欺骗图像分类。对抗性训练作为最有效的防御策略之一，被提出用来解决分类模型的脆弱性，即在训练过程中创建对抗性实例并注入训练数据。分类模型的攻防问题在过去的几年里得到了广泛的研究。语义切分作为分类的延伸，近年来也受到了极大的关注。最近的工作表明，需要大量的攻击迭代来创建有效的对抗性示例来愚弄分段模型。这种观察结果使得分割模型的健壮性评估和对抗性训练都具有挑战性。在这项工作中，我们提出了一种有效且高效的分段攻击方法，称为SegPGD。此外，我们还进行了收敛分析，结果表明，在相同的攻击迭代次数下，所提出的SegPGD算法能够生成比PGD算法更有效的攻击实例。此外，我们建议将我们的SegPGD作为分割对手训练的底层攻击方法。由于SegPGD可以生成更有效的对抗性实例，因此使用我们的SegPGD进行对抗性训练可以提高分割模型的稳健性。在流行的分割模型体系结构和标准分割数据集上的实验也验证了我们的建议。



## **47. Adversarial Attack across Datasets**

跨数据集的对抗性攻击 cs.CV

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2110.07718v2)

**Authors**: Yunxiao Qin, Yuanhao Xiong, Jinfeng Yi, Lihong Cao, Cho-Jui Hsieh

**Abstracts**: Existing transfer attack methods commonly assume that the attacker knows the training set (e.g., the label set, the input size) of the black-box victim models, which is usually unrealistic because in some cases the attacker cannot know this information. In this paper, we define a Generalized Transferable Attack (GTA) problem where the attacker doesn't know this information and is acquired to attack any randomly encountered images that may come from unknown datasets. To solve the GTA problem, we propose a novel Image Classification Eraser (ICE) that trains a particular attacker to erase classification information of any images from arbitrary datasets. Experiments on several datasets demonstrate that ICE greatly outperforms existing transfer attacks on GTA, and show that ICE uses similar texture-like noises to perturb different images from different datasets. Moreover, fast fourier transformation analysis indicates that the main components in each ICE noise are three sine waves for the R, G, and B image channels. Inspired by this interesting finding, we then design a novel Sine Attack (SA) method to optimize the three sine waves. Experiments show that SA performs comparably to ICE, indicating that the three sine waves are effective and enough to break DNNs under the GTA setting.

摘要: 现有的传输攻击方法通常假设攻击者知道黑盒受害者模型的训练集(例如，标签集、输入大小)，这通常是不现实的，因为在某些情况下攻击者无法知道该信息。在本文中，我们定义了一个广义可转移攻击(GTA)问题，其中攻击者不知道这些信息，并且被获取来攻击任何可能来自未知数据集的随机遇到的图像。为了解决GTA问题，我们提出了一种新的图像分类橡皮擦(ICE)，它训练特定的攻击者从任意数据集中擦除任何图像的分类信息。在几个数据集上的实验表明，ICE的性能大大优于现有的GTA传输攻击，并表明ICE使用类似纹理的噪声来扰动来自不同数据集的不同图像。此外，快速傅立叶变换分析表明，每个ICE噪声的主要分量是R、G和B图像通道的三个正弦波。受这一有趣发现的启发，我们设计了一种新的正弦攻击(SA)方法来优化三个正弦波。实验表明，SA的性能与ICE相当，说明在GTA设置下，这三个正弦波都是有效的，足以破解DNN。



## **48. Improving Adversarial Robustness via Mutual Information Estimation**

利用互信息估计提高对手的稳健性 cs.LG

This version has modified Eq.2 and its proof in the published version

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2207.12203v1)

**Authors**: Dawei Zhou, Nannan Wang, Xinbo Gao, Bo Han, Xiaoyu Wang, Yibing Zhan, Tongliang Liu

**Abstracts**: Deep neural networks (DNNs) are found to be vulnerable to adversarial noise. They are typically misled by adversarial samples to make wrong predictions. To alleviate this negative effect, in this paper, we investigate the dependence between outputs of the target model and input adversarial samples from the perspective of information theory, and propose an adversarial defense method. Specifically, we first measure the dependence by estimating the mutual information (MI) between outputs and the natural patterns of inputs (called natural MI) and MI between outputs and the adversarial patterns of inputs (called adversarial MI), respectively. We find that adversarial samples usually have larger adversarial MI and smaller natural MI compared with those w.r.t. natural samples. Motivated by this observation, we propose to enhance the adversarial robustness by maximizing the natural MI and minimizing the adversarial MI during the training process. In this way, the target model is expected to pay more attention to the natural pattern that contains objective semantics. Empirical evaluations demonstrate that our method could effectively improve the adversarial accuracy against multiple attacks.

摘要: 深度神经网络(DNN)被发现容易受到对抗性噪声的影响。他们通常会被对抗性样本误导，做出错误的预测。为了缓解这种负面影响，本文从信息论的角度研究了目标模型的输出与输入敌方样本之间的依赖关系，并提出了一种对抗性防御方法。具体地说，我们首先通过估计输出与输入的自然模式之间的互信息(称为自然MI)和输出与输入的对抗性模式之间的互信息(称为对抗性MI)来度量依赖。我们发现，与W.r.t.相比，对抗性样本通常具有较大的对抗性MI和较小的自然MI。天然样品。基于这一观察结果，我们提出在训练过程中通过最大化自然MI和最小化对手MI来增强对手的稳健性。这样，目标模型将更多地关注包含客观语义的自然模式。实验结果表明，该方法能够有效地提高对抗多重攻击的准确率。



## **49. Versatile Weight Attack via Flipping Limited Bits**

通过翻转有限比特进行多功能重量攻击 cs.CR

Extension of our ICLR 2021 work: arXiv:2102.10496

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2207.12405v1)

**Authors**: Jiawang Bai, Baoyuan Wu, Zhifeng Li, Shu-tao Xia

**Abstracts**: To explore the vulnerability of deep neural networks (DNNs), many attack paradigms have been well studied, such as the poisoning-based backdoor attack in the training stage and the adversarial attack in the inference stage. In this paper, we study a novel attack paradigm, which modifies model parameters in the deployment stage. Considering the effectiveness and stealthiness goals, we provide a general formulation to perform the bit-flip based weight attack, where the effectiveness term could be customized depending on the attacker's purpose. Furthermore, we present two cases of the general formulation with different malicious purposes, i.e., single sample attack (SSA) and triggered samples attack (TSA). To this end, we formulate this problem as a mixed integer programming (MIP) to jointly determine the state of the binary bits (0 or 1) in the memory and learn the sample modification. Utilizing the latest technique in integer programming, we equivalently reformulate this MIP problem as a continuous optimization problem, which can be effectively and efficiently solved using the alternating direction method of multipliers (ADMM) method. Consequently, the flipped critical bits can be easily determined through optimization, rather than using a heuristic strategy. Extensive experiments demonstrate the superiority of SSA and TSA in attacking DNNs.

摘要: 为了探索深层神经网络的脆弱性，人们研究了许多攻击范例，如训练阶段的基于中毒的后门攻击和推理阶段的对抗性攻击。本文研究了一种在部署阶段对模型参数进行修改的新型攻击范式。考虑到攻击的有效性和隐蔽性目标，我们给出了基于比特翻转的权重攻击的一般公式，其中有效项可以根据攻击者的目的进行定制。此外，我们还给出了两种具有不同恶意目的的通用公式，即单样本攻击(SSA)和触发样本攻击(TSA)。为此，我们将该问题描述为混合整数规划(MIP)，以共同确定存储器中二进制位(0或1)的状态并学习样本修改。利用整数规划的最新技术，我们将MIP问题等价地转化为一个连续优化问题，并利用乘子交替方向法(ADMM)对其进行了有效求解。因此，可以通过优化而不是使用启发式策略来容易地确定翻转的关键比特。大量的实验证明了SSA和TSA在攻击DNN方面的优势。



## **50. Privacy Against Inference Attacks in Vertical Federated Learning**

垂直联合学习中抵抗推理攻击的隐私保护 cs.LG

**SubmitDate**: 2022-07-24    [paper-pdf](http://arxiv.org/pdf/2207.11788v1)

**Authors**: Borzoo Rassouli, Morteza Varasteh, Deniz Gunduz

**Abstracts**: Vertical federated learning is considered, where an active party, having access to true class labels, wishes to build a classification model by utilizing more features from a passive party, which has no access to the labels, to improve the model accuracy. In the prediction phase, with logistic regression as the classification model, several inference attack techniques are proposed that the adversary, i.e., the active party, can employ to reconstruct the passive party's features, regarded as sensitive information. These attacks, which are mainly based on a classical notion of the center of a set, i.e., the Chebyshev center, are shown to be superior to those proposed in the literature. Moreover, several theoretical performance guarantees are provided for the aforementioned attacks. Subsequently, we consider the minimum amount of information that the adversary needs to fully reconstruct the passive party's features. In particular, it is shown that when the passive party holds one feature, and the adversary is only aware of the signs of the parameters involved, it can perfectly reconstruct that feature when the number of predictions is large enough. Next, as a defense mechanism, two privacy-preserving schemes are proposed that worsen the adversary's reconstruction attacks, while preserving the full benefits that VFL brings to the active party. Finally, experimental results demonstrate the effectiveness of the proposed attacks and the privacy-preserving schemes.

摘要: 考虑垂直联合学习，其中可以访问真实类别标签的主动方希望通过利用来自被动方的更多特征来构建分类模型，而被动方不能访问标签，以提高模型的精度。在预测阶段，以Logistic回归为分类模型，提出了几种推理攻击技术，对手即主动方可以用来重构被动方的特征，并将其视为敏感信息。这些攻击主要基于经典的集合中心概念，即切比雪夫中心，被证明优于文献中提出的攻击。此外，还为上述攻击提供了几个理论上的性能保证。随后，我们考虑了对手完全重建被动方特征所需的最小信息量。特别地，当被动方持有一个特征，并且对手只知道所涉及的参数的符号时，当预测次数足够大时，它可以完美地重构该特征。接下来，作为一种防御机制，提出了两种隐私保护方案，这两种方案在保留VFL给主动方带来的全部利益的同时，恶化了对手的重构攻击。最后，实验结果证明了所提出的攻击和隐私保护方案的有效性。



