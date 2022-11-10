# Latest Adversarial Attack Papers
**update at 2022-11-10 19:44:40**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Accountable and Explainable Methods for Complex Reasoning over Text**

基于文本的复杂推理的可靠和可解释方法 cs.LG

PhD Thesis

**SubmitDate**: 2022-11-09    [abs](http://arxiv.org/abs/2211.04946v1) [paper-pdf](http://arxiv.org/pdf/2211.04946v1)

**Authors**: Pepa Atanasova

**Abstract**: A major concern of Machine Learning (ML) models is their opacity. They are deployed in an increasing number of applications where they often operate as black boxes that do not provide explanations for their predictions. Among others, the potential harms associated with the lack of understanding of the models' rationales include privacy violations, adversarial manipulations, and unfair discrimination. As a result, the accountability and transparency of ML models have been posed as critical desiderata by works in policy and law, philosophy, and computer science.   In computer science, the decision-making process of ML models has been studied by developing accountability and transparency methods. Accountability methods, such as adversarial attacks and diagnostic datasets, expose vulnerabilities of ML models that could lead to malicious manipulations or systematic faults in their predictions. Transparency methods explain the rationales behind models' predictions gaining the trust of relevant stakeholders and potentially uncovering mistakes and unfairness in models' decisions. To this end, transparency methods have to meet accountability requirements as well, e.g., being robust and faithful to the underlying rationales of a model.   This thesis presents my research that expands our collective knowledge in the areas of accountability and transparency of ML models developed for complex reasoning tasks over text.

摘要: 机器学习(ML)模型的一个主要问题是其不透明性。它们被部署在越来越多的应用程序中，在这些应用程序中，它们通常作为黑匣子运行，不能为他们的预测提供解释。其中，与缺乏对模型原理的理解相关的潜在危害包括侵犯隐私、敌意操纵和不公平歧视。因此，ML模型的问责制和透明度已被政策和法律、哲学和计算机科学领域的著作视为迫切需要。在计算机科学中，人们通过发展问责和透明方法来研究ML模型的决策过程。诸如对抗性攻击和诊断数据集等问责方法暴露了ML模型的漏洞，这些漏洞可能导致恶意操纵或预测中的系统性错误。透明度方法解释了模型预测背后的原理，赢得了相关利益相关者的信任，并潜在地揭示了模型决策中的错误和不公平。为此目的，透明度方法还必须满足问责制要求，例如，稳健并忠实于模式的基本理由。这篇论文介绍了我的研究，旨在扩大我们在ML模型的责任和透明度领域的集体知识，这些模型是为复杂的文本推理任务开发的。



## **2. Lipschitz Continuous Algorithms for Graph Problems**

图问题的Lipschitz连续算法 cs.DS

**SubmitDate**: 2022-11-09    [abs](http://arxiv.org/abs/2211.04674v1) [paper-pdf](http://arxiv.org/pdf/2211.04674v1)

**Authors**: Soh Kumabe, Yuichi Yoshida

**Abstract**: It has been widely observed in the machine learning community that a small perturbation to the input can cause a large change in the prediction of a trained model, and such phenomena have been intensively studied in the machine learning community under the name of adversarial attacks. Because graph algorithms also are widely used for decision making and knowledge discovery, it is important to design graph algorithms that are robust against adversarial attacks. In this study, we consider the Lipschitz continuity of algorithms as a robustness measure and initiate a systematic study of the Lipschitz continuity of algorithms for (weighted) graph problems.   Depending on how we embed the output solution to a metric space, we can think of several Lipschitzness notions. We mainly consider the one that is invariant under scaling of weights, and we provide Lipschitz continuous algorithms and lower bounds for the minimum spanning tree problem, the shortest path problem, and the maximum weight matching problem. In particular, our shortest path algorithm is obtained by first designing an algorithm for unweighted graphs that are robust against edge contractions and then applying it to the unweighted graph constructed from the original weighted graph.   Then, we consider another Lipschitzness notion induced by a natural mapping that maps the output solution to its characteristic vector. It turns out that no Lipschitz continuous algorithm exists for this Lipschitz notion, and we instead design algorithms with bounded pointwise Lipschitz constants for the minimum spanning tree problem and the maximum weight bipartite matching problem. Our algorithm for the latter problem is based on an LP relaxation with entropy regularization.

摘要: 机器学习界已经广泛观察到，输入的微小扰动会导致训练模型的预测发生很大变化，这种现象已经在机器学习界以对抗攻击的名义进行了深入的研究。由于图算法也被广泛用于决策和知识发现，因此设计对对手攻击具有健壮性的图算法是很重要的。在这项研究中，我们将算法的Lipschitz连续性作为一个稳健性度量，并开始系统地研究(加权)图问题的算法的Lipschitz连续性。根据我们将输出解嵌入到度量空间的方式，我们可以想到几个Lipschitzness概念。我们主要考虑在权值缩放下不变的问题，给出了最小生成树问题、最短路问题和最大权匹配问题的Lipschitz连续算法和下界。特别是，我们的最短路径算法是通过设计一个对边收缩具有健壮性的未加权图的算法来获得的，然后将其应用于由原始加权图构造的未加权图。然后，我们考虑由自然映射诱导的另一个Lipschitzness概念，该映射将输出解映射到其特征向量。结果表明，对于这种Lipschitz概念，不存在Lipschitz连续算法，而是针对最小生成树问题和最大权二部匹配问题设计了逐点Lipschitz常数有界的算法。对于后一个问题，我们的算法是基于带熵正则化的LP松弛算法。



## **3. FedDef: Defense Against Gradient Leakage in Federated Learning-based Network Intrusion Detection Systems**

FedDef：基于联邦学习的网络入侵检测系统的梯度泄漏防御 cs.CR

14 pages, 9 figures, submitted to TIFS

**SubmitDate**: 2022-11-09    [abs](http://arxiv.org/abs/2210.04052v2) [paper-pdf](http://arxiv.org/pdf/2210.04052v2)

**Authors**: Jiahui Chen, Yi Zhao, Qi Li, Xuewei Feng, Ke Xu

**Abstract**: Deep learning (DL) methods have been widely applied to anomaly-based network intrusion detection system (NIDS) to detect malicious traffic. To expand the usage scenarios of DL-based methods, the federated learning (FL) framework allows multiple users to train a global model on the basis of respecting individual data privacy. However, it has not yet been systematically evaluated how robust FL-based NIDSs are against existing privacy attacks under existing defenses. To address this issue, we propose two privacy evaluation metrics designed for FL-based NIDSs, including (1) privacy score that evaluates the similarity between the original and recovered traffic features using reconstruction attacks, and (2) evasion rate against NIDSs using Generative Adversarial Network-based adversarial attack with the reconstructed benign traffic. We conduct experiments to show that existing defenses provide little protection that the corresponding adversarial traffic can even evade the SOTA NIDS Kitsune. To defend against such attacks and build a more robust FL-based NIDS, we further propose FedDef, a novel optimization-based input perturbation defense strategy with theoretical guarantee. It achieves both high utility by minimizing the gradient distance and strong privacy protection by maximizing the input distance. We experimentally evaluate four existing defenses on four datasets and show that our defense outperforms all the baselines in terms of privacy protection with up to 7 times higher privacy score, while maintaining model accuracy loss within 3% under optimal parameter combination.

摘要: 深度学习方法已被广泛应用于基于异常的网络入侵检测系统中，以检测恶意流量。为了扩展基于DL的方法的使用场景，联邦学习(FL)框架允许多个用户在尊重个人数据隐私的基础上训练全局模型。然而，还没有系统地评估基于FL的NIDS在现有防御系统下对现有隐私攻击的健壮性。针对这一问题，我们提出了两个针对FL网络入侵检测系统的隐私评估指标，包括：(1)隐私评分，通过重构攻击评估原始流量特征和恢复流量特征之间的相似性；(2)利用重构的良性流量对基于网络的生成性对抗性攻击的逃避率。我们进行的实验表明，现有的防御措施提供的保护很少，相应的敌意流量甚至可以避开Sota NIDS Kitsune。为了防御此类攻击，构建一个更健壮的基于FL的网络入侵检测系统，我们进一步提出了一种新的基于优化的输入扰动防御策略FedDef，并提供了理论上的保证。它既通过最小化梯度距离实现了高效用，又通过最大化输入距离实现了强大的隐私保护。我们在四个数据集上对四种现有的防御措施进行了实验评估，结果表明，我们的防御措施在隐私保护方面的表现优于所有基线，隐私得分高达7倍，同时在最优参数组合下将模型精度损失保持在3%以内。



## **4. CgAT: Center-Guided Adversarial Training for Deep Hashing-Based Retrieval**

CgAT：中心引导的基于深度散列的对抗性训练 cs.CV

has no contributions

**SubmitDate**: 2022-11-09    [abs](http://arxiv.org/abs/2204.10779v3) [paper-pdf](http://arxiv.org/pdf/2204.10779v3)

**Authors**: Xunguang Wang, Yinqun Lin, Xiaomeng Li

**Abstract**: Deep hashing has been extensively utilized in massive image retrieval because of its efficiency and effectiveness. However, deep hashing models are vulnerable to adversarial examples, making it essential to develop adversarial defense methods for image retrieval. Existing solutions achieved limited defense performance because of using weak adversarial samples for training and lacking discriminative optimization objectives to learn robust features. In this paper, we present a min-max based Center-guided Adversarial Training, namely CgAT, to improve the robustness of deep hashing networks through worst adversarial examples. Specifically, we first formulate the center code as a semantically-discriminative representative of the input image content, which preserves the semantic similarity with positive samples and dissimilarity with negative examples. We prove that a mathematical formula can calculate the center code immediately. After obtaining the center codes in each optimization iteration of the deep hashing network, they are adopted to guide the adversarial training process. On the one hand, CgAT generates the worst adversarial examples as augmented data by maximizing the Hamming distance between the hash codes of the adversarial examples and the center codes. On the other hand, CgAT learns to mitigate the effects of adversarial samples by minimizing the Hamming distance to the center codes. Extensive experiments on the benchmark datasets demonstrate the effectiveness of our adversarial training algorithm in defending against adversarial attacks for deep hashing-based retrieval. Compared with the current state-of-the-art defense method, we significantly improve the defense performance by an average of 18.61%, 12.35%, and 11.56% on FLICKR-25K, NUS-WIDE, and MS-COCO, respectively.

摘要: 深度哈希法以其高效、高效的特点在海量图像检索中得到了广泛应用。然而，深度哈希模型很容易受到敌意例子的攻击，因此有必要开发针对图像检索的对抗性防御方法。现有的解决方案由于使用弱对抗性样本进行训练，并且缺乏区分优化目标来学习稳健特征，使得防御性能受到限制。本文提出了一种基于最小-最大值的中心引导敌意训练算法，即CgAT，通过最坏的敌意例子来提高深度哈希网络的健壮性。具体地说，我们首先将中心代码定义为输入图像内容的语义区分代表，它保留了与正例的语义相似性和与反例的不相似性。我们证明了一个数学公式可以立即计算中心代码。在获得深度散列网络每次优化迭代的中心代码后，将其用于指导对抗性训练过程。一方面，CgAT通过最大化对抗性示例的哈希码与中心码之间的汉明距离来生成最差的对抗性示例作为扩充数据。另一方面，CgAT通过最小化到中心码的汉明距离来学习减轻对抗性样本的影响。在基准数据集上的大量实验表明，我们的对抗性训练算法在防御基于深度散列的检索的对抗性攻击方面是有效的。与当前最先进的防御方法相比，我们在Flickr-25K、NUS-Wide和MS-COCO上的防御性能分别平均提高了18.61%、12.35%和11.56%。



## **5. On the Versatile Uses of Partial Distance Correlation in Deep Learning**

[TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None cs.CV

This paper has been selected as best paper award for ECCV 2022!

**SubmitDate**: 2022-11-08    [abs](http://arxiv.org/abs/2207.09684v3) [paper-pdf](http://arxiv.org/pdf/2207.09684v3)

**Authors**: Xingjian Zhen, Zihang Meng, Rudrasis Chakraborty, Vikas Singh

**Abstract**: Comparing the functional behavior of neural network models, whether it is a single network over time or two (or more networks) during or post-training, is an essential step in understanding what they are learning (and what they are not), and for identifying strategies for regularization or efficiency improvements. Despite recent progress, e.g., comparing vision transformers to CNNs, systematic comparison of function, especially across different networks, remains difficult and is often carried out layer by layer. Approaches such as canonical correlation analysis (CCA) are applicable in principle, but have been sparingly used so far. In this paper, we revisit a (less widely known) from statistics, called distance correlation (and its partial variant), designed to evaluate correlation between feature spaces of different dimensions. We describe the steps necessary to carry out its deployment for large scale models -- this opens the door to a surprising array of applications ranging from conditioning one deep model w.r.t. another, learning disentangled representations as well as optimizing diverse models that would directly be more robust to adversarial attacks. Our experiments suggest a versatile regularizer (or constraint) with many advantages, which avoids some of the common difficulties one faces in such analyses. Code is at https://github.com/zhenxingjian/Partial_Distance_Correlation.

摘要: 比较神经网络模型的功能行为，无论是随着时间的推移是单个网络还是在训练期间或训练后的两个(或更多)网络，对于了解它们正在学习什么(以及它们不是什么)以及确定正规化或效率改进的策略是至关重要的一步。尽管最近取得了进展，例如，将视觉转换器与CNN进行了比较，但系统地比较功能，特别是跨不同网络的功能，仍然很困难，而且往往是逐层进行的。典型相关分析(CCA)等方法在原则上是适用的，但到目前为止一直很少使用。在本文中，我们回顾了统计学中的一个(不太广为人知的)方法，称为距离相关(及其部分变量)，旨在评估不同维度的特征空间之间的相关性。我们描述了在大规模模型中实施其部署所需的步骤--这为一系列令人惊讶的应用打开了大门，从调节一个深度模型到W.r.t。另一种是，学习分离的表示以及优化多样化的模型，这些模型将直接对对手攻击更健壮。我们的实验提出了一种具有许多优点的通用正则化(或约束)方法，它避免了人们在此类分析中面临的一些常见困难。代码在https://github.com/zhenxingjian/Partial_Distance_Correlation.上



## **6. NaturalAdversaries: Can Naturalistic Adversaries Be as Effective as Artificial Adversaries?**

自然广告：自然主义的对手能像人为的对手一样有效吗？ cs.CL

Findings of EMNLP 2022

**SubmitDate**: 2022-11-08    [abs](http://arxiv.org/abs/2211.04364v1) [paper-pdf](http://arxiv.org/pdf/2211.04364v1)

**Authors**: Saadia Gabriel, Hamid Palangi, Yejin Choi

**Abstract**: While a substantial body of prior work has explored adversarial example generation for natural language understanding tasks, these examples are often unrealistic and diverge from the real-world data distributions. In this work, we introduce a two-stage adversarial example generation framework (NaturalAdversaries), for designing adversaries that are effective at fooling a given classifier and demonstrate natural-looking failure cases that could plausibly occur during in-the-wild deployment of the models.   At the first stage a token attribution method is used to summarize a given classifier's behaviour as a function of the key tokens in the input. In the second stage a generative model is conditioned on the key tokens from the first stage. NaturalAdversaries is adaptable to both black-box and white-box adversarial attacks based on the level of access to the model parameters. Our results indicate these adversaries generalize across domains, and offer insights for future research on improving robustness of neural text classification models.

摘要: 虽然大量的先前工作已经探索了自然语言理解任务的对抗性示例生成，但这些示例往往是不现实的，并且与真实世界的数据分布背道而驰。在这项工作中，我们介绍了一个两阶段的对抗性实例生成框架(NaturalAdversary)，用于设计有效地愚弄给定分类器的对手，并展示在模型的野外部署过程中可能发生的看起来自然的失败案例。在第一阶段，使用标记归属方法将给定分类器的行为总结为输入中关键标记的函数。在第二阶段中，生成模型以来自第一阶段的密钥令牌为条件。基于对模型参数的访问级别，NaturalAdversary能够适应黑盒和白盒对抗性攻击。我们的结果表明这些对手具有跨域的泛化能力，并为未来提高神经文本分类模型的稳健性的研究提供了见解。



## **7. Preserving Semantics in Textual Adversarial Attacks**

文本对抗性攻击中的语义保护 cs.CL

8 pages, 4 figures

**SubmitDate**: 2022-11-08    [abs](http://arxiv.org/abs/2211.04205v1) [paper-pdf](http://arxiv.org/pdf/2211.04205v1)

**Authors**: David Herel, Hugo Cisneros, Tomas Mikolov

**Abstract**: Adversarial attacks in NLP challenge the way we look at language models. The goal of this kind of adversarial attack is to modify the input text to fool a classifier while maintaining the original meaning of the text. Although most existing adversarial attacks claim to fulfill the constraint of semantics preservation, careful scrutiny shows otherwise. We show that the problem lies in the text encoders used to determine the similarity of adversarial examples, specifically in the way they are trained. Unsupervised training methods make these encoders more susceptible to problems with antonym recognition. To overcome this, we introduce a simple, fully supervised sentence embedding technique called Semantics-Preserving-Encoder (SPE). The results show that our solution minimizes the variation in the meaning of the adversarial examples generated. It also significantly improves the overall quality of adversarial examples, as confirmed by human evaluators. Furthermore, it can be used as a component in any existing attack to speed up its execution while maintaining similar attack success.

摘要: NLP中的对抗性攻击挑战了我们看待语言模型的方式。这种对抗性攻击的目标是修改输入文本以愚弄分类器，同时保持文本的原始含义。尽管大多数现有的对抗性攻击声称满足语义保存的约束，但仔细观察发现并非如此。我们表明，问题在于用于确定对抗性例子相似性的文本编码器，特别是在它们被训练的方式上。无监督的训练方法使这些编码者更容易出现反义词识别问题。为了克服这一问题，我们引入了一种简单的、完全有监督的句子嵌入技术，称为语义保留编码器(SPE)。结果表明，我们的解决方案最大限度地减少了生成的对抗性例子的含义变化。它还显著提高了对抗性例子的整体质量，这一点得到了人类评估员的证实。此外，它可以用作任何现有攻击的组件，以加快其执行速度，同时保持类似的攻击成功。



## **8. A Hypergraph-Based Machine Learning Ensemble Network Intrusion Detection System**

一种基于超图的机器学习集成网络入侵检测系统 cs.CR

12 pages, 10 figures

**SubmitDate**: 2022-11-08    [abs](http://arxiv.org/abs/2211.03933v1) [paper-pdf](http://arxiv.org/pdf/2211.03933v1)

**Authors**: Zong-Zhi Lin, Thomas D. Pike, Mark M. Bailey, Nathaniel D. Bastian

**Abstract**: Network intrusion detection systems (NIDS) to detect malicious attacks continues to meet challenges. NIDS are vulnerable to auto-generated port scan infiltration attempts and NIDS are often developed offline, resulting in a time lag to prevent the spread of infiltration to other parts of a network. To address these challenges, we use hypergraphs to capture evolving patterns of port scan attacks via the set of internet protocol addresses and destination ports, thereby deriving a set of hypergraph-based metrics to train a robust and resilient ensemble machine learning (ML) NIDS that effectively monitors and detects port scanning activities and adversarial intrusions while evolving intelligently in real-time. Through the combination of (1) intrusion examples, (2) NIDS update rules, (3) attack threshold choices to trigger NIDS retraining requests, and (4) production environment with no prior knowledge of the nature of network traffic 40 scenarios were auto-generated to evaluate the ML ensemble NIDS comprising three tree-based models. Results show that under the model settings of an Update-ALL-NIDS rule (namely, retrain and update all the three models upon the same NIDS retraining request) the proposed ML ensemble NIDS produced the best results with nearly 100% detection performance throughout the simulation, exhibiting robustness in the complex dynamics of the simulated cyber-security scenario.

摘要: 网络入侵检测系统(NID)检测恶意攻击的能力不断受到挑战。NID容易受到自动生成的端口扫描渗透尝试的攻击，并且NID通常是离线开发的，导致防止渗透扩散到网络其他部分的时间滞后。为了应对这些挑战，我们使用超图来捕获端口扫描攻击的演变模式，通过一组互联网协议地址和目的端口，从而推导出一组基于超图的度量来训练稳健和有弹性的集成机器学习(ML)网络入侵检测系统，在实时智能演化的同时有效地监控和检测端口扫描活动和恶意入侵。通过组合(1)入侵实例、(2)网络入侵检测系统更新规则、(3)用于触发网络入侵检测系统再训练请求的攻击阈值选择、(4)在不知道网络流量性质的情况下的生产环境，自动生成40个场景来评估由三个基于树的模型组成的最大似然融合网络入侵检测系统。结果表明，在更新所有网络入侵检测系统规则的模型设置下(即根据相同的网络入侵检测系统再训练请求对所有三个模型进行重新训练和更新)，所提出的ML集成网络入侵检测系统在整个仿真过程中获得了接近100%的检测性能，在模拟的网络安全场景的复杂动态中表现出了健壮性。



## **9. Are AlphaZero-like Agents Robust to Adversarial Perturbations?**

类AlphaZero代理对对手的扰动稳健吗？ cs.AI

Accepted by Neurips 2022

**SubmitDate**: 2022-11-07    [abs](http://arxiv.org/abs/2211.03769v1) [paper-pdf](http://arxiv.org/pdf/2211.03769v1)

**Authors**: Li-Cheng Lan, Huan Zhang, Ti-Rong Wu, Meng-Yu Tsai, I-Chen Wu, Cho-Jui Hsieh

**Abstract**: The success of AlphaZero (AZ) has demonstrated that neural-network-based Go AIs can surpass human performance by a large margin. Given that the state space of Go is extremely large and a human player can play the game from any legal state, we ask whether adversarial states exist for Go AIs that may lead them to play surprisingly wrong actions. In this paper, we first extend the concept of adversarial examples to the game of Go: we generate perturbed states that are ``semantically'' equivalent to the original state by adding meaningless moves to the game, and an adversarial state is a perturbed state leading to an undoubtedly inferior action that is obvious even for Go beginners. However, searching the adversarial state is challenging due to the large, discrete, and non-differentiable search space. To tackle this challenge, we develop the first adversarial attack on Go AIs that can efficiently search for adversarial states by strategically reducing the search space. This method can also be extended to other board games such as NoGo. Experimentally, we show that the actions taken by both Policy-Value neural network (PV-NN) and Monte Carlo tree search (MCTS) can be misled by adding one or two meaningless stones; for example, on 58\% of the AlphaGo Zero self-play games, our method can make the widely used KataGo agent with 50 simulations of MCTS plays a losing action by adding two meaningless stones. We additionally evaluated the adversarial examples found by our algorithm with amateur human Go players and 90\% of examples indeed lead the Go agent to play an obviously inferior action. Our code is available at \url{https://PaperCode.cc/GoAttack}.

摘要: AlphaZero(AZ)的成功证明，基于神经网络的围棋可以大大超过人类的表现。鉴于围棋的状态空间非常大，而且人类棋手可以在任何合法的状态下玩这个游戏，我们问围棋人工智能是否存在可能导致他们下令人惊讶的错误动作的对抗性状态。在本文中，我们首先将对抗性例子的概念推广到围棋游戏中：我们通过在棋局中添加无意义的走法来产生在语义上等同于原始状态的扰动状态，而对抗性状态是导致毫无疑问的劣势的扰动状态，即使对围棋初学者来说也是显而易见的。然而，由于搜索空间大、离散和不可微，搜索敌对状态是具有挑战性的。为了应对这一挑战，我们在围棋人工智能上开发了第一个对抗性攻击，它可以通过战略性地缩小搜索空间来有效地搜索对抗性状态。这种方法也可以推广到其他棋类游戏，如NoGo。实验表明，在策略值神经网络(PV-NN)和蒙特卡罗树搜索(MCTS)算法中，添加一两个无意义的棋子都会误导它们的行为；例如，在58个AlphaGo Zero自玩游戏中，我们的方法可以通过添加两个无意义的棋子来使具有50个MCTS模拟的广泛使用的KataGo智能体发挥失败的作用。此外，我们还用业余人类围棋棋手对算法发现的对抗性实例进行了评估，90%的实例确实导致围棋代理下了明显的劣势。我们的代码可在\url{https://PaperCode.cc/GoAttack}.



## **10. Neural Architectural Backdoors**

神经架构后门 cs.CR

**SubmitDate**: 2022-11-07    [abs](http://arxiv.org/abs/2210.12179v2) [paper-pdf](http://arxiv.org/pdf/2210.12179v2)

**Authors**: Ren Pang, Changjiang Li, Zhaohan Xi, Shouling Ji, Ting Wang

**Abstract**: This paper asks the intriguing question: is it possible to exploit neural architecture search (NAS) as a new attack vector to launch previously improbable attacks? Specifically, we present EVAS, a new attack that leverages NAS to find neural architectures with inherent backdoors and exploits such vulnerability using input-aware triggers. Compared with existing attacks, EVAS demonstrates many interesting properties: (i) it does not require polluting training data or perturbing model parameters; (ii) it is agnostic to downstream fine-tuning or even re-training from scratch; (iii) it naturally evades defenses that rely on inspecting model parameters or training data. With extensive evaluation on benchmark datasets, we show that EVAS features high evasiveness, transferability, and robustness, thereby expanding the adversary's design spectrum. We further characterize the mechanisms underlying EVAS, which are possibly explainable by architecture-level ``shortcuts'' that recognize trigger patterns. This work raises concerns about the current practice of NAS and points to potential directions to develop effective countermeasures.

摘要: [TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None



## **11. Deviations in Representations Induced by Adversarial Attacks**

对抗性攻击引起的陈述偏差 cs.LG

**SubmitDate**: 2022-11-07    [abs](http://arxiv.org/abs/2211.03714v1) [paper-pdf](http://arxiv.org/pdf/2211.03714v1)

**Authors**: Daniel Steinberg, Paul Munro

**Abstract**: Deep learning has been a popular topic and has achieved success in many areas. It has drawn the attention of researchers and machine learning practitioners alike, with developed models deployed to a variety of settings. Along with its achievements, research has shown that deep learning models are vulnerable to adversarial attacks. This finding brought about a new direction in research, whereby algorithms were developed to attack and defend vulnerable networks. Our interest is in understanding how these attacks effect change on the intermediate representations of deep learning models. We present a method for measuring and analyzing the deviations in representations induced by adversarial attacks, progressively across a selected set of layers. Experiments are conducted using an assortment of attack algorithms, on the CIFAR-10 dataset, with plots created to visualize the impact of adversarial attacks across different layers in a network.

摘要: 深度学习一直是一个热门话题，并在许多领域取得了成功。它已经引起了研究人员和机器学习从业者的注意，开发的模型部署在各种环境中。除了取得的成就，研究还表明，深度学习模型很容易受到对手的攻击。这一发现带来了研究的新方向，据此开发了攻击和防御易受攻击的网络的算法。我们的兴趣是了解这些攻击如何影响深度学习模型的中间表示。我们提出了一种方法来测量和分析由对抗性攻击引起的表示中的偏差，逐步跨越选定的一组层。在CIFAR-10数据集上，使用各种攻击算法进行了实验，并创建了一些曲线图，以可视化网络中不同层的对抗性攻击的影响。



## **12. Black-Box Attack against GAN-Generated Image Detector with Contrastive Perturbation**

对比度微扰下对GaN生成图像探测器的黑盒攻击 cs.CV

**SubmitDate**: 2022-11-07    [abs](http://arxiv.org/abs/2211.03509v1) [paper-pdf](http://arxiv.org/pdf/2211.03509v1)

**Authors**: Zijie Lou, Gang Cao, Man Lin

**Abstract**: Visually realistic GAN-generated facial images raise obvious concerns on potential misuse. Many effective forensic algorithms have been developed to detect such synthetic images in recent years. It is significant to assess the vulnerability of such forensic detectors against adversarial attacks. In this paper, we propose a new black-box attack method against GAN-generated image detectors. A novel contrastive learning strategy is adopted to train the encoder-decoder network based anti-forensic model under a contrastive loss function. GAN images and their simulated real counterparts are constructed as positive and negative samples, respectively. Leveraging on the trained attack model, imperceptible contrastive perturbation could be applied to input synthetic images for removing GAN fingerprint to some extent. As such, existing GAN-generated image detectors are expected to be deceived. Extensive experimental results verify that the proposed attack effectively reduces the accuracy of three state-of-the-art detectors on six popular GANs. High visual quality of the attacked images is also achieved. The source code will be available at https://github.com/ZXMMD/BAttGAND.

摘要: 视觉逼真的GaN生成的面部图像引起了人们对潜在滥用的明显担忧。近年来，已经开发了许多有效的法医算法来检测此类合成图像。重要的是要评估这种法医探测器在对抗攻击时的脆弱性。本文提出了一种新的针对GaN图像探测器的黑盒攻击方法。采用一种新的对比学习策略，在对比损失函数下训练基于编解码器网络的反取证模型。GaN图像和模拟的真实图像分别被构造为正样本和负样本。利用训练好的攻击模型，对输入的合成图像进行不可察觉的对比度扰动，在一定程度上去除GaN指纹。因此，现有的GaN产生的图像探测器预计会被欺骗。大量的实验结果证明，所提出的攻击有效地降低了三个最先进的检测器对六个流行的GAN的准确性。还实现了受攻击图像的高视觉质量。源代码将在https://github.com/ZXMMD/BAttGAND.上提供



## **13. On the Anonymity of Peer-To-Peer Network Anonymity Schemes Used by Cryptocurrencies**

加密货币使用的对等网络匿名方案的匿名性研究 cs.CR

**SubmitDate**: 2022-11-07    [abs](http://arxiv.org/abs/2201.11860v4) [paper-pdf](http://arxiv.org/pdf/2201.11860v4)

**Authors**: Piyush Kumar Sharma, Devashish Gosain, Claudia Diaz

**Abstract**: Cryptocurrency systems can be subject to deanonimization attacks by exploiting the network-level communication on their peer-to-peer network. Adversaries who control a set of colluding node(s) within the peer-to-peer network can observe transactions being exchanged and infer the parties involved. Thus, various network anonymity schemes have been proposed to mitigate this problem, with some solutions providing theoretical anonymity guarantees.   In this work, we model such peer-to-peer network anonymity solutions and evaluate their anonymity guarantees. To do so, we propose a novel framework that uses Bayesian inference to obtain the probability distributions linking transactions to their possible originators. We characterize transaction anonymity with those distributions, using entropy as metric of adversarial uncertainty on the originator's identity. In particular, we model Dandelion, Dandelion++ and Lightning Network. We study different configurations and demonstrate that none of them offers acceptable anonymity to their users. For instance, our analysis reveals that in the widely deployed Lightning Network, with 1% strategically chosen colluding nodes the adversary can uniquely determine the originator for about 50% of the total transactions in the network. In Dandelion, an adversary that controls 15% of the nodes has on average uncertainty among only 8 possible originators. Moreover, we observe that due to the way Dandelion and Dandelion++ are designed, increasing the network size does not correspond to an increase in the anonymity set of potential originators. Alarmingly, our longitudinal analysis of Lightning Network reveals rather an inverse trend -- with the growth of the network the overall anonymity decreases.

摘要: 通过利用其对等网络上的网络级通信，加密货币系统可能会受到反匿名化攻击。在对等网络中控制一组串通节点的敌手可以观察正在交换的交易并推断所涉及的各方。因此，各种网络匿名方案被提出来缓解这一问题，一些解决方案提供了理论上的匿名性保证。在这项工作中，我们对这种对等网络匿名解决方案进行建模，并评估它们的匿名性保证。为此，我们提出了一个新的框架，它使用贝叶斯推理来获得将事务链接到可能的发起者的概率分布。我们使用这些分布来表征交易匿名性，使用熵作为对发起者身份的敌意不确定性的度量。特别是，我们对蒲公英、蒲公英++和闪电网络进行了建模。我们研究了不同的配置，并证明它们都不能为用户提供可接受的匿名性。例如，我们的分析表明，在广泛部署的闪电网络中，通过1%的策略选择合谋节点，对手可以唯一地确定网络中约50%的总交易的发起者。在蒲公英中，一个控制了15%节点的对手平均只有8个可能的发起者中存在不确定性。此外，我们观察到，由于蒲公英和蒲公英++的设计方式，增加网络规模并不对应于潜在发起者匿名性集合的增加。令人担忧的是，我们对Lightning Network的纵向分析揭示了一个相反的趋势--随着网络的增长，总体匿名性下降。



## **14. NIP: Neuron-level Inverse Perturbation Against Adversarial Attacks**

NIP：对抗对抗性攻击的神经元级逆摄动 cs.CV

There are some problems in the figure so we need to withdraw this  paper. We will upload the new version after revision

**SubmitDate**: 2022-11-07    [abs](http://arxiv.org/abs/2112.13060v2) [paper-pdf](http://arxiv.org/pdf/2112.13060v2)

**Authors**: Ruoxi Chen, Haibo Jin, Jinyin Chen, Haibin Zheng, Yue Yu, Shouling Ji

**Abstract**: Although deep learning models have achieved unprecedented success, their vulnerabilities towards adversarial attacks have attracted increasing attention, especially when deployed in security-critical domains. To address the challenge, numerous defense strategies, including reactive and proactive ones, have been proposed for robustness improvement. From the perspective of image feature space, some of them cannot reach satisfying results due to the shift of features. Besides, features learned by models are not directly related to classification results. Different from them, We consider defense method essentially from model inside and investigated the neuron behaviors before and after attacks. We observed that attacks mislead the model by dramatically changing the neurons that contribute most and least to the correct label. Motivated by it, we introduce the concept of neuron influence and further divide neurons into front, middle and tail part. Based on it, we propose neuron-level inverse perturbation(NIP), the first neuron-level reactive defense method against adversarial attacks. By strengthening front neurons and weakening those in the tail part, NIP can eliminate nearly all adversarial perturbations while still maintaining high benign accuracy. Besides, it can cope with different sizes of perturbations via adaptivity, especially larger ones. Comprehensive experiments conducted on three datasets and six models show that NIP outperforms the state-of-the-art baselines against eleven adversarial attacks. We further provide interpretable proofs via neuron activation and visualization for better understanding.

摘要: [TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:1129)'))) requestId:None



## **15. Adversarial Reconfigurable Intelligent Surface Against Physical Layer Key Generation**

对抗物理层密钥生成的对抗性可重构智能表面 eess.SP

**SubmitDate**: 2022-11-07    [abs](http://arxiv.org/abs/2206.10955v2) [paper-pdf](http://arxiv.org/pdf/2206.10955v2)

**Authors**: Zhuangkun Wei, Bin Li, Weisi Guo

**Abstract**: The development of reconfigurable intelligent surfaces (RIS) has recently advanced the research of physical layer security (PLS). Beneficial impacts of RIS include but are not limited to offering a new degree-of-freedom (DoF) for key-less PLS optimization, and increasing channel randomness for physical layer secret key generation (PL-SKG). However, there is a lack of research studying how adversarial RIS can be used to attack and obtain legitimate secret keys generated by PL-SKG. In this work, we show an Eve-controlled adversarial RIS (Eve-RIS), by inserting into the legitimate channel a random and reciprocal channel, can partially reconstruct the secret keys from the legitimate PL-SKG process. To operationalize this concept, we design Eve-RIS schemes against two PL-SKG techniques used: (i) the CSI-based PL-SKG, and (ii) the two-way cross multiplication based PL-SKG. The channel probing at Eve-RIS is realized by compressed sensing designs with a small number of radio-frequency (RF) chains. Then, the optimal RIS phase is obtained by maximizing the Eve-RIS inserted deceiving channel. Our analysis and results show that even with a passive RIS, our proposed Eve-RIS can achieve a high key match rate with legitimate users, and is resistant to most of the current defensive approaches. This means the novel Eve-RIS provides a new eavesdropping threat on PL-SKG, which can spur new research areas to counter adversarial RIS attacks.

摘要: [TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None



## **16. Contrastive Weighted Learning for Near-Infrared Gaze Estimation**

[TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:1129)'))) requestId:None cs.CV

**SubmitDate**: 2022-11-06    [abs](http://arxiv.org/abs/2211.03073v1) [paper-pdf](http://arxiv.org/pdf/2211.03073v1)

**Authors**: Adam Lee

**Abstract**: Appearance-based gaze estimation has been very successful with the use of deep learning. Many following works improved domain generalization for gaze estimation. However, even though there has been much progress in domain generalization for gaze estimation, most of the recent work have been focused on cross-dataset performance -- accounting for different distributions in illuminations, head pose, and lighting. Although improving gaze estimation in different distributions of RGB images is important, near-infrared image based gaze estimation is also critical for gaze estimation in dark settings. Also there are inherent limitations relying solely on supervised learning for regression tasks. This paper contributes to solving these problems and proposes GazeCWL, a novel framework for gaze estimation with near-infrared images using contrastive learning. This leverages adversarial attack techniques for data augmentation and a novel contrastive loss function specifically for regression tasks that effectively clusters the features of different samples in the latent space. Our model outperforms previous domain generalization models in infrared image based gaze estimation and outperforms the baseline by 45.6\% while improving the state-of-the-art by 8.6\%, we demonstrate the efficacy of our method.

摘要: 随着深度学习的使用，基于外表的凝视估计已经非常成功。随后的许多工作改进了视线估计的领域泛化。然而，尽管在凝视估计的领域泛化方面已经有了很大的进展，但最近的大部分工作都集中在跨数据集性能上--考虑了光照、头部姿势和光照的不同分布。虽然在不同分布的RGB图像中改进凝视估计是重要的，但基于近红外图像的凝视估计对于黑暗环境下的凝视估计也是至关重要的。此外，回归任务仅依靠有监督的学习也存在固有的局限性。为了解决这些问题，本文提出了一种新的基于对比学习的近红外图像凝视估计框架GazeCWL。该算法利用对抗性攻击技术进行数据扩充，并针对回归任务提出了一种新的对比损失函数，该函数能有效地将不同样本的特征在潜在空间中聚类。我们的模型在基于红外图像的视线估计中优于以往的域泛化模型，并且比基线提高了45.6\%，同时提高了8.6%，验证了我们的方法的有效性。



## **17. Privacy-Preserving Models for Legal Natural Language Processing**

合法自然语言处理中的隐私保护模型 cs.CL

Camera ready, to appear at the Natural Legal Language Processing  Workshop 2022 co-located with EMNLP

**SubmitDate**: 2022-11-05    [abs](http://arxiv.org/abs/2211.02956v1) [paper-pdf](http://arxiv.org/pdf/2211.02956v1)

**Authors**: Ying Yin, Ivan Habernal

**Abstract**: Pre-training large transformer models with in-domain data improves domain adaptation and helps gain performance on the domain-specific downstream tasks. However, sharing models pre-trained on potentially sensitive data is prone to adversarial privacy attacks. In this paper, we asked to which extent we can guarantee privacy of pre-training data and, at the same time, achieve better downstream performance on legal tasks without the need of additional labeled data. We extensively experiment with scalable self-supervised learning of transformer models under the formal paradigm of differential privacy and show that under specific training configurations we can improve downstream performance without sacrifying privacy protection for the in-domain data. Our main contribution is utilizing differential privacy for large-scale pre-training of transformer language models in the legal NLP domain, which, to the best of our knowledge, has not been addressed before.

摘要: 用域内数据预先训练大型变压器模型可以改进域适应，并有助于在特定于域的下游任务中获得性能。然而，针对潜在敏感数据预先训练的共享模型容易受到敌意隐私攻击。在本文中，我们提出了在何种程度上可以保证训练前数据的私密性，同时在不需要额外的标签数据的情况下，在合法任务上获得更好的下游性能。我们在区分隐私的形式化范式下对变压器模型进行了广泛的实验，结果表明，在特定的训练配置下，我们可以在不牺牲域内数据隐私保护的情况下提高下行性能。我们的主要贡献是利用差异隐私在法律NLP领域对变压器语言模型进行大规模的预培训，据我们所知，这以前从未被解决过。



## **18. Adversarial Attacks on Transformers-Based Malware Detectors**

对基于Transformers的恶意软件检测器的敌意攻击 cs.CR

Accepted to the 2022 NeurIPS ML Safety Workshop. Code available at  https://github.com/yashjakhotiya/Adversarial-Attacks-On-Transformers

**SubmitDate**: 2022-11-05    [abs](http://arxiv.org/abs/2210.00008v2) [paper-pdf](http://arxiv.org/pdf/2210.00008v2)

**Authors**: Yash Jakhotiya, Heramb Patil, Jugal Rawlani, Dr. Sunil B. Mane

**Abstract**: Signature-based malware detectors have proven to be insufficient as even a small change in malignant executable code can bypass these signature-based detectors. Many machine learning-based models have been proposed to efficiently detect a wide variety of malware. Many of these models are found to be susceptible to adversarial attacks - attacks that work by generating intentionally designed inputs that can force these models to misclassify. Our work aims to explore vulnerabilities in the current state of the art malware detectors to adversarial attacks. We train a Transformers-based malware detector, carry out adversarial attacks resulting in a misclassification rate of 23.9% and propose defenses that reduce this misclassification rate to half. An implementation of our work can be found at https://github.com/yashjakhotiya/Adversarial-Attacks-On-Transformers.

摘要: 事实证明，基于签名的恶意软件检测器是不够的，因为即使对恶意可执行代码进行很小的更改也可以绕过这些基于签名的检测器。已经提出了许多基于机器学习的模型来有效地检测各种恶意软件。其中许多模型被发现容易受到对抗性攻击-通过生成故意设计的输入来工作的攻击，可以迫使这些模型错误分类。我们的工作旨在探索当前最先进的恶意软件检测器中的漏洞，以进行对抗性攻击。我们训练了一个基于Transformers的恶意软件检测器，执行了导致23.9%错误分类率的对抗性攻击，并提出了将错误分类率降低到一半的防御措施。我们工作的实现可以在https://github.com/yashjakhotiya/Adversarial-Attacks-On-Transformers.上找到



## **19. Physical Adversarial Attack meets Computer Vision: A Decade Survey**

身体对抗攻击与计算机视觉相遇：十年综述 cs.CV

32 pages. Under Review

**SubmitDate**: 2022-11-05    [abs](http://arxiv.org/abs/2209.15179v2) [paper-pdf](http://arxiv.org/pdf/2209.15179v2)

**Authors**: Hui Wei, Hao Tang, Xuemei Jia, Hanxun Yu, Zhubo Li, Zhixiang Wang, Shin'ichi Satoh, Zheng Wang

**Abstract**: Although Deep Neural Networks (DNNs) have achieved impressive results in computer vision, their exposed vulnerability to adversarial attacks remains a serious concern. A series of works has shown that by adding elaborate perturbations to images, DNNs could have catastrophic degradation in performance metrics. And this phenomenon does not only exist in the digital space but also in the physical space. Therefore, estimating the security of these DNNs-based systems is critical for safely deploying them in the real world, especially for security-critical applications, e.g., autonomous cars, video surveillance, and medical diagnosis. In this paper, we focus on physical adversarial attacks and provide a comprehensive survey of over 150 existing papers. We first clarify the concept of the physical adversarial attack and analyze its characteristics. Then, we define the adversarial medium, essential to perform attacks in the physical world. Next, we present the physical adversarial attack methods in task order: classification, detection, and re-identification, and introduce their performance in solving the trilemma: effectiveness, stealthiness, and robustness. In the end, we discuss the current challenges and potential future directions.

摘要: 尽管深度神经网络(DNN)在计算机视觉方面取得了令人印象深刻的成果，但它们暴露出的易受对手攻击的脆弱性仍然是一个严重的问题。一系列工作表明，通过向图像添加精心设计的扰动，DNN可能会在性能指标上造成灾难性的降级。而这种现象不仅存在于数字空间，也存在于物理空间。因此，评估这些基于DNNS的系统的安全性对于在现实世界中安全地部署它们至关重要，特别是对于自动驾驶汽车、视频监控和医疗诊断等安全关键型应用。在这篇论文中，我们聚焦于物理对抗攻击，并提供了超过150篇现有论文的全面调查。首先厘清了身体对抗攻击的概念，分析了身体对抗攻击的特点。然后，我们定义了对抗性媒介，这是在物理世界中执行攻击所必需的。接下来，我们按任务顺序介绍了物理对抗攻击方法：分类、检测和重新识别，并介绍了它们在解决有效性、隐蔽性和健壮性这三个两难问题上的表现。最后，我们讨论了当前的挑战和潜在的未来方向。



## **20. Is RobustBench/AutoAttack a suitable Benchmark for Adversarial Robustness?**

RobustBtch/AutoAttack是衡量对手健壮性的合适基准吗？ cs.CV

AAAI-22 AdvML Workshop ShortPaper

**SubmitDate**: 2022-11-05    [abs](http://arxiv.org/abs/2112.01601v2) [paper-pdf](http://arxiv.org/pdf/2112.01601v2)

**Authors**: Peter Lorenz, Dominik Strassel, Margret Keuper, Janis Keuper

**Abstract**: Recently, RobustBench (Croce et al. 2020) has become a widely recognized benchmark for the adversarial robustness of image classification networks. In its most commonly reported sub-task, RobustBench evaluates and ranks the adversarial robustness of trained neural networks on CIFAR10 under AutoAttack (Croce and Hein 2020b) with l-inf perturbations limited to eps = 8/255. With leading scores of the currently best performing models of around 60% of the baseline, it is fair to characterize this benchmark to be quite challenging. Despite its general acceptance in recent literature, we aim to foster discussion about the suitability of RobustBench as a key indicator for robustness which could be generalized to practical applications. Our line of argumentation against this is two-fold and supported by excessive experiments presented in this paper: We argue that I) the alternation of data by AutoAttack with l-inf, eps = 8/255 is unrealistically strong, resulting in close to perfect detection rates of adversarial samples even by simple detection algorithms and human observers. We also show that other attack methods are much harder to detect while achieving similar success rates. II) That results on low-resolution data sets like CIFAR10 do not generalize well to higher resolution images as gradient-based attacks appear to become even more detectable with increasing resolutions.

摘要: 最近，RobustBch(Croce et al.2020)已成为图像分类网络对抗性稳健性的公认基准。在其最常见的子任务中，RobustBch评估和排名了AutoAttack(Croce和Hein 2020b)下训练的神经网络在CIFAR10上的对抗稳健性，其中l-inf扰动限制在EPS=8/255。由于目前表现最好的模型的领先分数约为基线的60%，因此可以公平地将该基准描述为相当具有挑战性。尽管它在最近的文献中被广泛接受，但我们的目标是促进关于RobustBitch作为健壮性的关键指标的适宜性的讨论，该指标可以推广到实际应用中。我们对此的论证是双重的，并得到了本文提供的大量实验的支持：我们认为：i)AutoAttack与l-inf，EPS=8/255的数据交互是不切实际的，导致即使使用简单的检测算法和人类观察者也能获得接近完美的敌意样本检测率。我们还表明，在获得类似成功率的情况下，其他攻击方法要难得多。Ii)在像CIFAR10这样的低分辨率数据集上的结果不能很好地推广到更高分辨率的图像，因为基于梯度的攻击似乎随着分辨率的增加而变得更容易检测到。



## **21. Stateful Detection of Adversarial Reprogramming**

[TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None cs.GT

**SubmitDate**: 2022-11-05    [abs](http://arxiv.org/abs/2211.02885v1) [paper-pdf](http://arxiv.org/pdf/2211.02885v1)

**Authors**: Yang Zheng, Xiaoyi Feng, Zhaoqiang Xia, Xiaoyue Jiang, Maura Pintor, Ambra Demontis, Battista Biggio, Fabio Roli

**Abstract**: Adversarial reprogramming allows stealing computational resources by repurposing machine learning models to perform a different task chosen by the attacker. For example, a model trained to recognize images of animals can be reprogrammed to recognize medical images by embedding an adversarial program in the images provided as inputs. This attack can be perpetrated even if the target model is a black box, supposed that the machine-learning model is provided as a service and the attacker can query the model and collect its outputs. So far, no defense has been demonstrated effective in this scenario. We show for the first time that this attack is detectable using stateful defenses, which store the queries made to the classifier and detect the abnormal cases in which they are similar. Once a malicious query is detected, the account of the user who made it can be blocked. Thus, the attacker must create many accounts to perpetrate the attack. To decrease this number, the attacker could create the adversarial program against a surrogate classifier and then fine-tune it by making few queries to the target model. In this scenario, the effectiveness of the stateful defense is reduced, but we show that it is still effective.

摘要: 对抗性重新编程允许通过重新调整机器学习模型的用途来执行攻击者选择的不同任务来窃取计算资源。例如，被训练为识别动物图像的模型可以通过在作为输入提供的图像中嵌入对抗性程序来重新编程以识别医学图像。即使目标模型是黑盒，假设机器学习模型作为服务提供，并且攻击者可以查询该模型并收集其输出，也可以进行这种攻击。到目前为止，还没有任何防御措施在这种情况下被证明是有效的。我们首次展示了使用状态防御可以检测到这种攻击，状态防御存储对分类器的查询，并检测它们相似的异常情况。一旦检测到恶意查询，就可以阻止进行该查询的用户的帐户。因此，攻击者必须创建多个帐户才能实施攻击。为了减少这个数字，攻击者可以创建针对代理分类器的对抗性程序，然后通过对目标模型进行少量查询来对其进行微调。在这种情况下，状态防御的有效性会降低，但我们证明它仍然有效。



## **22. Textual Manifold-based Defense Against Natural Language Adversarial Examples**

[TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None cs.CL

**SubmitDate**: 2022-11-05    [abs](http://arxiv.org/abs/2211.02878v1) [paper-pdf](http://arxiv.org/pdf/2211.02878v1)

**Authors**: Dang Minh Nguyen, Luu Anh Tuan

**Abstract**: Recent studies on adversarial images have shown that they tend to leave the underlying low-dimensional data manifold, making them significantly more challenging for current models to make correct predictions. This so-called off-manifold conjecture has inspired a novel line of defenses against adversarial attacks on images. In this study, we find a similar phenomenon occurs in the contextualized embedding space induced by pretrained language models, in which adversarial texts tend to have their embeddings diverge from the manifold of natural ones. Based on this finding, we propose Textual Manifold-based Defense (TMD), a defense mechanism that projects text embeddings onto an approximated embedding manifold before classification. It reduces the complexity of potential adversarial examples, which ultimately enhances the robustness of the protected model. Through extensive experiments, our method consistently and significantly outperforms previous defenses under various attack settings without trading off clean accuracy. To the best of our knowledge, this is the first NLP defense that leverages the manifold structure against adversarial attacks. Our code is available at \url{https://github.com/dangne/tmd}.

摘要: 最近对对抗性图像的研究表明，它们往往会离开底层的低维数据流形，这使得它们对当前模型做出正确预测的挑战要大得多。这种所谓的非流形猜想启发了一种针对图像的敌意攻击的新防线。在这项研究中，我们发现在由预训练语言模型产生的上下文嵌入空间中也出现了类似的现象，在这种情况下，对抗性文本的嵌入往往偏离了自然文本的流形。基于这一发现，我们提出了一种基于文本流形的防御机制(TMD)，该机制在分类前将文本嵌入到近似嵌入流形上。它降低了潜在敌意例子的复杂性，最终增强了受保护模型的健壮性。通过大量的实验，我们的方法在不牺牲干净的准确性的情况下，在各种攻击设置下一致且显著地优于以前的防御方法。据我们所知，这是第一个利用多种结构对抗对手攻击的NLP防御。我们的代码可在\url{https://github.com/dangne/tmd}.



## **23. A Resource Allocation Scheme for Energy Demand Management in 6G-enabled Smart Grid**

一种面向6G智能电网能源需求管理的资源分配方案 cs.NI

2023 North American Innovative Smart Grid Technologies Conference

**SubmitDate**: 2022-11-05    [abs](http://arxiv.org/abs/2207.00154v2) [paper-pdf](http://arxiv.org/pdf/2207.00154v2)

**Authors**: Shafkat Islam, Ioannis Zografopoulos, Md Tamjid Hossain, Shahriar Badsha, Charalambos Konstantinou

**Abstract**: Smart grid (SG) systems enhance grid resilience and efficient operation, leveraging the bidirectional flow of energy and information between generation facilities and prosumers. For energy demand management (EDM), the SG network requires computing a large amount of data generated by massive Internet-of-things sensors and advanced metering infrastructure (AMI) with minimal latency. This paper proposes a deep reinforcement learning (DRL)-based resource allocation scheme in a 6G-enabled SG edge network to offload resource-consuming EDM computation to edge servers. Automatic resource provisioning is achieved by harnessing the computational capabilities of smart meters in the dynamic edge network. To enforce DRL-assisted policies in dense 6G networks, the state information from multiple edge servers is required. However, adversaries can "poison" such information through false state injection (FSI) attacks, exhausting SG edge computing resources. Toward addressing this issue, we investigate the impact of such FSI attacks with respect to abusive utilization of edge resources, and develop a lightweight FSI detection mechanism based on supervised classifiers. Simulation results demonstrate the efficacy of DRL in dynamic resource allocation, the impact of the FSI attacks, and the effectiveness of the detection technique.

摘要: [TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None



## **24. On Trace of PGD-Like Adversarial Attacks**

[TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None cs.CV

**SubmitDate**: 2022-11-05    [abs](http://arxiv.org/abs/2205.09586v2) [paper-pdf](http://arxiv.org/pdf/2205.09586v2)

**Authors**: Mo Zhou, Vishal M. Patel

**Abstract**: Adversarial attacks pose safety and security concerns to deep learning applications, but their characteristics are under-explored. Yet largely imperceptible, a strong trace could have been left by PGD-like attacks in an adversarial example. Recall that PGD-like attacks trigger the ``local linearity'' of a network, which implies different extents of linearity for benign or adversarial examples. Inspired by this, we construct an Adversarial Response Characteristics (ARC) feature to reflect the model's gradient consistency around the input to indicate the extent of linearity. Under certain conditions, it qualitatively shows a gradually varying pattern from benign example to adversarial example, as the latter leads to Sequel Attack Effect (SAE). To quantitatively evaluate the effectiveness of ARC, we conduct experiments on CIFAR-10 and ImageNet for attack detection and attack type recognition in a challenging setting. The results suggest that SAE is an effective and unique trace of PGD-like attacks reflected through the ARC feature. The ARC feature is intuitive, light-weighted, non-intrusive, and data-undemanding.

摘要: [TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None



## **25. Fairness-aware Regression Robust to Adversarial Attacks**

公平性回归对对手攻击的健壮性 cs.CR

**SubmitDate**: 2022-11-04    [abs](http://arxiv.org/abs/2211.04449v1) [paper-pdf](http://arxiv.org/pdf/2211.04449v1)

**Authors**: Yulu Jin, Lifeng Lai

**Abstract**: In this paper, we take a first step towards answering the question of how to design fair machine learning algorithms that are robust to adversarial attacks. Using a minimax framework, we aim to design an adversarially robust fair regression model that achieves optimal performance in the presence of an attacker who is able to add a carefully designed adversarial data point to the dataset or perform a rank-one attack on the dataset. By solving the proposed nonsmooth nonconvex-nonconcave minimax problem, the optimal adversary as well as the robust fairness-aware regression model are obtained. For both synthetic data and real-world datasets, numerical results illustrate that the proposed adversarially robust fair models have better performance on poisoned datasets than other fair machine learning models in both prediction accuracy and group-based fairness measure.

摘要: 在本文中，我们向回答如何设计对对手攻击具有健壮性的公平的机器学习算法的问题迈出了第一步。使用极小极大框架，我们的目标是设计一个对抗健壮的公平回归模型，在攻击者能够将精心设计的对抗性数据点添加到数据集或对数据集执行等级1攻击的情况下，该模型实现最佳性能。通过求解所提出的非光滑非凸非凹极大极小问题，得到了最优对手以及稳健的公平感知回归模型。数值结果表明，对于人工数据和真实数据集，本文提出的对抗性稳健公平模型在预测精度和基于分组的公平性度量方面都优于其他公平机器学习模型。



## **26. Improving Adversarial Robustness to Sensitivity and Invariance Attacks with Deep Metric Learning**

深度度量学习提高对手对敏感度和不变性攻击的鲁棒性 cs.LG

v1

**SubmitDate**: 2022-11-04    [abs](http://arxiv.org/abs/2211.02468v1) [paper-pdf](http://arxiv.org/pdf/2211.02468v1)

**Authors**: Anaelia Ovalle, Evan Czyzycki, Cho-Jui Hsieh

**Abstract**: Intentionally crafted adversarial samples have effectively exploited weaknesses in deep neural networks. A standard method in adversarial robustness assumes a framework to defend against samples crafted by minimally perturbing a sample such that its corresponding model output changes. These sensitivity attacks exploit the model's sensitivity toward task-irrelevant features. Another form of adversarial sample can be crafted via invariance attacks, which exploit the model underestimating the importance of relevant features. Previous literature has indicated a tradeoff in defending against both attack types within a strictly L_p bounded defense. To promote robustness toward both types of attacks beyond Euclidean distance metrics, we use metric learning to frame adversarial regularization as an optimal transport problem. Our preliminary results indicate that regularizing over invariant perturbations in our framework improves both invariant and sensitivity defense.

摘要: 故意制作的敌意样本有效地利用了深层神经网络的弱点。对抗性稳健性的标准方法假设了一个框架，通过对样本进行最小程度的扰动来防御样本，从而使其相应的模型输出发生变化。这些敏感性攻击利用了模型对任务无关特征的敏感性。另一种形式的敌意样本可以通过不变性攻击来制作，该攻击利用了该模型低估了相关特征的重要性。以前的文献已经指出，在严格的Lp有界防御下，防御这两种攻击类型是一种权衡。为了提高对这两种攻击的稳健性，超越欧几里德距离度量，我们使用度量学习将对抗性正则化框架作为一个最优传输问题。我们的初步结果表明，在我们的框架中对不变扰动进行正则化可以改善不变项防御和敏感性防御。



## **27. Rickrolling the Artist: Injecting Invisible Backdoors into Text-Guided Image Generation Models**

点击艺术家：将看不见的后门注入文本引导的图像生成模型 cs.LG

25 pages, 16 figures, 5 tables

**SubmitDate**: 2022-11-04    [abs](http://arxiv.org/abs/2211.02408v1) [paper-pdf](http://arxiv.org/pdf/2211.02408v1)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Kristian Kersting

**Abstract**: While text-to-image synthesis currently enjoys great popularity among researchers and the general public, the security of these models has been neglected so far. Many text-guided image generation models rely on pre-trained text encoders from external sources, and their users trust that the retrieved models will behave as promised. Unfortunately, this might not be the case. We introduce backdoor attacks against text-guided generative models and demonstrate that their text encoders pose a major tampering risk. Our attacks only slightly alter an encoder so that no suspicious model behavior is apparent for image generations with clean prompts. By then inserting a single non-Latin character into the prompt, the adversary can trigger the model to either generate images with pre-defined attributes or images following a hidden, potentially malicious description. We empirically demonstrate the high effectiveness of our attacks on Stable Diffusion and highlight that the injection process of a single backdoor takes less than two minutes. Besides phrasing our approach solely as an attack, it can also force an encoder to forget phrases related to certain concepts, such as nudity or violence, and help to make image generation safer.

摘要: 虽然文本到图像的合成目前在研究人员和普通大众中很受欢迎，但到目前为止，这些模型的安全性一直被忽视。许多文本制导的图像生成模型依赖于来自外部来源的预先训练的文本编码器，它们的用户相信检索到的模型将如承诺的那样运行。不幸的是，情况可能并非如此。我们引入了针对文本引导的生成模型的后门攻击，并证明了它们的文本编码器构成了主要的篡改风险。我们的攻击只略微改变了编码器，因此对于具有干净提示的图像生成来说，没有明显的可疑模型行为。然后，通过在提示中插入单个非拉丁字符，攻击者可以触发模型生成具有预定义属性的图像，或在隐藏的潜在恶意描述之后生成图像。我们从经验上证明了我们对稳定扩散攻击的高效性，并强调了单个后门的注入过程只需不到两分钟。除了将我们的方法仅作为一种攻击来表述外，它还可以迫使编码者忘记与某些概念相关的短语，如裸体或暴力，并有助于使图像生成更安全。



## **28. Logits are predictive of network type**

[TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None cs.CV

**SubmitDate**: 2022-11-04    [abs](http://arxiv.org/abs/2211.02272v1) [paper-pdf](http://arxiv.org/pdf/2211.02272v1)

**Authors**: Ali Borji

**Abstract**: We show that it is possible to predict which deep network has generated a given logit vector with accuracy well above chance. We utilize a number of networks on a dataset, initialized with random weights or pretrained weights, as well as fine-tuned networks. A classifier is then trained on the logit vectors of the trained set of this dataset to map the logit vector to the network index that has generated it. The classifier is then evaluated on the test set of the dataset. Results are better with randomly initialized networks, but also generalize to pretrained networks as well as fine-tuned ones. Classification accuracy is higher using unnormalized logits than normalized ones. We find that there is little transfer when applying a classifier to the same networks but with different sets of weights. In addition to help better understand deep networks and the way they encode uncertainty, we anticipate our finding to be useful in some applications (e.g. tailoring an adversarial attack for a certain type of network). Code is available at https://github.com/aliborji/logits.

摘要: [TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None



## **29. Quantum Man-in-the-middle Attacks: a Game-theoretic Approach with Applications to Radars**

[TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None eess.SP

**SubmitDate**: 2022-11-04    [abs](http://arxiv.org/abs/2211.02228v1) [paper-pdf](http://arxiv.org/pdf/2211.02228v1)

**Authors**: YInan Hu, Quanyan Zhu

**Abstract**: The detection and discrimination of quantum states serve a crucial role in quantum signal processing, a discipline that studies methods and techniques to process signals that obey the quantum mechanics frameworks. However, just like classical detection, evasive behaviors also exist in quantum detection. In this paper, we formulate an adversarial quantum detection scenario where the detector is passive and does not know the quantum states have been distorted by an attacker. We compare the performance of a passive detector with the one of a non-adversarial detector to demonstrate how evasive behaviors can undermine the performance of quantum detection. We use a case study of target detection with quantum radars to corroborate our analytical results.

摘要: [TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None



## **30. Adversarial Defense via Neural Oscillation inspired Gradient Masking**

[TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None cs.LG

**SubmitDate**: 2022-11-04    [abs](http://arxiv.org/abs/2211.02223v1) [paper-pdf](http://arxiv.org/pdf/2211.02223v1)

**Authors**: Chunming Jiang, Yilei Zhang

**Abstract**: Spiking neural networks (SNNs) attract great attention due to their low power consumption, low latency, and biological plausibility. As they are widely deployed in neuromorphic devices for low-power brain-inspired computing, security issues become increasingly important. However, compared to deep neural networks (DNNs), SNNs currently lack specifically designed defense methods against adversarial attacks. Inspired by neural membrane potential oscillation, we propose a novel neural model that incorporates the bio-inspired oscillation mechanism to enhance the security of SNNs. Our experiments show that SNNs with neural oscillation neurons have better resistance to adversarial attacks than ordinary SNNs with LIF neurons on kinds of architectures and datasets. Furthermore, we propose a defense method that changes model's gradients by replacing the form of oscillation, which hides the original training gradients and confuses the attacker into using gradients of 'fake' neurons to generate invalid adversarial samples. Our experiments suggest that the proposed defense method can effectively resist both single-step and iterative attacks with comparable defense effectiveness and much less computational costs than adversarial training methods on DNNs. To the best of our knowledge, this is the first work that establishes adversarial defense through masking surrogate gradients on SNNs.

摘要: [TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None



## **31. Fault-Tolerant Federated Reinforcement Learning with Theoretical Guarantee**

[TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None cs.LG

Published at NeurIPS 2021. Extended version with proofs and  additional experimental details and results. New version changes: reduced  file size of figures; added a diagram illustrating the problem setting; added  link to code on GitHub; modified proof for Theorem 6 (highlighted in red)

**SubmitDate**: 2022-11-03    [abs](http://arxiv.org/abs/2110.14074v2) [paper-pdf](http://arxiv.org/pdf/2110.14074v2)

**Authors**: Flint Xiaofeng Fan, Yining Ma, Zhongxiang Dai, Wei Jing, Cheston Tan, Bryan Kian Hsiang Low

**Abstract**: The growing literature of Federated Learning (FL) has recently inspired Federated Reinforcement Learning (FRL) to encourage multiple agents to federatively build a better decision-making policy without sharing raw trajectories. Despite its promising applications, existing works on FRL fail to I) provide theoretical analysis on its convergence, and II) account for random system failures and adversarial attacks. Towards this end, we propose the first FRL framework the convergence of which is guaranteed and tolerant to less than half of the participating agents being random system failures or adversarial attackers. We prove that the sample efficiency of the proposed framework is guaranteed to improve with the number of agents and is able to account for such potential failures or attacks. All theoretical results are empirically verified on various RL benchmark tasks.

摘要: [TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None



## **32. Clean-label Backdoor Attack against Deep Hashing based Retrieval**

[TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None cs.CV

**SubmitDate**: 2022-11-03    [abs](http://arxiv.org/abs/2109.08868v2) [paper-pdf](http://arxiv.org/pdf/2109.08868v2)

**Authors**: Kuofeng Gao, Jiawang Bai, Bin Chen, Dongxian Wu, Shu-Tao Xia

**Abstract**: Deep hashing has become a popular method in large-scale image retrieval due to its computational and storage efficiency. However, recent works raise the security concerns of deep hashing. Although existing works focus on the vulnerability of deep hashing in terms of adversarial perturbations, we identify a more pressing threat, backdoor attack, when the attacker has access to the training data. A backdoored deep hashing model behaves normally on original query images, while returning the images with the target label when the trigger presents, which makes the attack hard to be detected. In this paper, we uncover this security concern by utilizing clean-label data poisoning. To the best of our knowledge, this is the first attempt at the backdoor attack against deep hashing models. To craft the poisoned images, we first generate the targeted adversarial patch as the backdoor trigger. Furthermore, we propose the confusing perturbations to disturb the hashing code learning, such that the hashing model can learn more about the trigger. The confusing perturbations are imperceptible and generated by dispersing the images with the target label in the Hamming space. We have conducted extensive experiments to verify the efficacy of our backdoor attack under various settings. For instance, it can achieve 63% targeted mean average precision on ImageNet under 48 bits code length with only 40 poisoned images.

摘要: [TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None



## **33. Physically Adversarial Attacks and Defenses in Computer Vision: A Survey**

[TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None cs.CV

**SubmitDate**: 2022-11-03    [abs](http://arxiv.org/abs/2211.01671v1) [paper-pdf](http://arxiv.org/pdf/2211.01671v1)

**Authors**: Xingxing Wei, Bangzheng Pu, Jiefan Lu, Baoyuan Wu

**Abstract**: Although Deep Neural Networks (DNNs) have been widely applied in various real-world scenarios, they are vulnerable to adversarial examples. The current adversarial attacks in computer vision can be divided into digital attacks and physical attacks according to their different attack forms. Compared with digital attacks, which generate perturbations in the digital pixels, physical attacks are more practical in the real world. Owing to the serious security problem caused by physically adversarial examples, many works have been proposed to evaluate the physically adversarial robustness of DNNs in the past years. In this paper, we summarize a survey versus the current physically adversarial attacks and physically adversarial defenses in computer vision. To establish a taxonomy, we organize the current physical attacks from attack tasks, attack forms, and attack methods, respectively. Thus, readers can have a systematic knowledge about this topic from different aspects. For the physical defenses, we establish the taxonomy from pre-processing, in-processing, and post-processing for the DNN models to achieve a full coverage of the adversarial defenses. Based on the above survey, we finally discuss the challenges of this research field and further outlook the future direction.

摘要: 尽管深度神经网络(DNN)已被广泛应用于各种现实场景中，但它们很容易受到对手例子的影响。根据攻击形式的不同，目前计算机视觉中的对抗性攻击可分为数字攻击和物理攻击。与在数字像素中产生扰动的数字攻击相比，物理攻击在现实世界中更实用。由于物理对抗实例带来了严重的安全问题，在过去的几年里，人们已经提出了许多工作来评估DNN的物理对抗健壮性。本文对当前计算机视觉中的身体对抗攻击和身体对抗防御进行了综述。为了建立分类，我们分别从攻击任务、攻击形式和攻击方法三个方面对当前的物理攻击进行了组织。因此，读者可以从不同的方面对这一主题有一个系统的了解。对于物理防御，我们从DNN模型的前处理、内处理和后处理三个方面建立了分类，以实现对抗性防御的全覆盖。在上述研究的基础上，我们最后讨论了该研究领域面临的挑战，并进一步展望了未来的研究方向。



## **34. Enhancing Transferability of Adversarial Examples with Spatial Momentum**

利用空间动量增强对抗性例句的可转移性 cs.CV

Accepted as Oral by 5-th Chinese Conference on Pattern Recognition  and Computer Vision, PRCV 2022

**SubmitDate**: 2022-11-03    [abs](http://arxiv.org/abs/2203.13479v2) [paper-pdf](http://arxiv.org/pdf/2203.13479v2)

**Authors**: Guoqiu Wang, Huanqian Yan, Xingxing Wei

**Abstract**: Many adversarial attack methods achieve satisfactory attack success rates under the white-box setting, but they usually show poor transferability when attacking other DNN models. Momentum-based attack is one effective method to improve transferability. It integrates the momentum term into the iterative process, which can stabilize the update directions by adding the gradients' temporal correlation for each pixel. We argue that only this temporal momentum is not enough, the gradients from the spatial domain within an image, i.e. gradients from the context pixels centered on the target pixel are also important to the stabilization. For that, we propose a novel method named Spatial Momentum Iterative FGSM attack (SMI-FGSM), which introduces the mechanism of momentum accumulation from temporal domain to spatial domain by considering the context information from different regions within the image. SMI-FGSM is then integrated with temporal momentum to simultaneously stabilize the gradients' update direction from both the temporal and spatial domains. Extensive experiments show that our method indeed further enhances adversarial transferability. It achieves the best transferability success rate for multiple mainstream undefended and defended models, which outperforms the state-of-the-art attack methods by a large margin of 10\% on average.

摘要: [TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None



## **35. Leveraging Domain Features for Detecting Adversarial Attacks Against Deep Speech Recognition in Noise**

[TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None eess.AS

**SubmitDate**: 2022-11-03    [abs](http://arxiv.org/abs/2211.01621v1) [paper-pdf](http://arxiv.org/pdf/2211.01621v1)

**Authors**: Christian Heider Nielsen, Zheng-Hua Tan

**Abstract**: In recent years, significant progress has been made in deep model-based automatic speech recognition (ASR), leading to its widespread deployment in the real world. At the same time, adversarial attacks against deep ASR systems are highly successful. Various methods have been proposed to defend ASR systems from these attacks. However, existing classification based methods focus on the design of deep learning models while lacking exploration of domain specific features. This work leverages filter bank-based features to better capture the characteristics of attacks for improved detection. Furthermore, the paper analyses the potentials of using speech and non-speech parts separately in detecting adversarial attacks. In the end, considering adverse environments where ASR systems may be deployed, we study the impact of acoustic noise of various types and signal-to-noise ratios. Extensive experiments show that the inverse filter bank features generally perform better in both clean and noisy environments, the detection is effective using either speech or non-speech part, and the acoustic noise can largely degrade the detection performance.

摘要: 近年来，基于深度模型的自动语音识别(ASR)技术取得了长足的进步，在现实世界中得到了广泛的应用。与此同时，针对深度ASR系统的对抗性攻击非常成功。已经提出了各种方法来防御ASR系统免受这些攻击。然而，现有的基于分类的方法侧重于深度学习模型的设计，而缺乏对特定领域特征的探索。这项工作利用基于滤波器组的功能来更好地捕获攻击的特征，以改进检测。此外，本文还分析了分别使用语音部分和非语音部分检测对抗性攻击的潜力。最后，考虑到ASR系统可能部署的不利环境，我们研究了各种类型的声噪声和信噪比对系统的影响。大量实验表明，逆滤波器组特征在清洁和噪声环境下都表现出较好的性能，无论是使用语音部分还是非语音部分，检测都是有效的，噪声会大大降低检测性能。



## **36. Robust Few-shot Learning Without Using any Adversarial Samples**

在不使用任何对抗性样本的情况下进行健壮的少机会学习 cs.CV

TNNLS Submission (Under Review)

**SubmitDate**: 2022-11-03    [abs](http://arxiv.org/abs/2211.01598v1) [paper-pdf](http://arxiv.org/pdf/2211.01598v1)

**Authors**: Gaurav Kumar Nayak, Ruchit Rawal, Inder Khatri, Anirban Chakraborty

**Abstract**: The high cost of acquiring and annotating samples has made the `few-shot' learning problem of prime importance. Existing works mainly focus on improving performance on clean data and overlook robustness concerns on the data perturbed with adversarial noise. Recently, a few efforts have been made to combine the few-shot problem with the robustness objective using sophisticated Meta-Learning techniques. These methods rely on the generation of adversarial samples in every episode of training, which further adds a computational burden. To avoid such time-consuming and complicated procedures, we propose a simple but effective alternative that does not require any adversarial samples. Inspired by the cognitive decision-making process in humans, we enforce high-level feature matching between the base class data and their corresponding low-frequency samples in the pretraining stage via self distillation. The model is then fine-tuned on the samples of novel classes where we additionally improve the discriminability of low-frequency query set features via cosine similarity. On a 1-shot setting of the CIFAR-FS dataset, our method yields a massive improvement of $60.55\%$ & $62.05\%$ in adversarial accuracy on the PGD and state-of-the-art Auto Attack, respectively, with a minor drop in clean accuracy compared to the baseline. Moreover, our method only takes $1.69\times$ of the standard training time while being $\approx$ $5\times$ faster than state-of-the-art adversarial meta-learning methods. The code is available at https://github.com/vcl-iisc/robust-few-shot-learning.

摘要: [TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None



## **37. Data-free Defense of Black Box Models Against Adversarial Attacks**

黑箱模型抵抗对手攻击的无数据防御 cs.LG

TIFS Submission (Under Review)

**SubmitDate**: 2022-11-03    [abs](http://arxiv.org/abs/2211.01579v1) [paper-pdf](http://arxiv.org/pdf/2211.01579v1)

**Authors**: Gaurav Kumar Nayak, Inder Khatri, Shubham Randive, Ruchit Rawal, Anirban Chakraborty

**Abstract**: Several companies often safeguard their trained deep models (i.e. details of architecture, learnt weights, training details etc.) from third-party users by exposing them only as black boxes through APIs. Moreover, they may not even provide access to the training data due to proprietary reasons or sensitivity concerns. We make the first attempt to provide adversarial robustness to the black box models in a data-free set up. We construct synthetic data via generative model and train surrogate network using model stealing techniques. To minimize adversarial contamination on perturbed samples, we propose `wavelet noise remover' (WNR) that performs discrete wavelet decomposition on input images and carefully select only a few important coefficients determined by our `wavelet coefficient selection module' (WCSM). To recover the high-frequency content of the image after noise removal via WNR, we further train a `regenerator' network with an objective to retrieve the coefficients such that the reconstructed image yields similar to original predictions on the surrogate model. At test time, WNR combined with trained regenerator network is prepended to the black box network, resulting in a high boost in adversarial accuracy. Our method improves the adversarial accuracy on CIFAR-10 by 38.98% and 32.01% on state-of-the-art Auto Attack compared to baseline, even when the attacker uses surrogate architecture (Alexnet-half and Alexnet) similar to the black box architecture (Alexnet) with same model stealing strategy as defender. The code is available at https://github.com/vcl-iisc/data-free-black-box-defense

摘要: 几家公司经常保护他们训练有素的深度模型(即建筑细节、学习的重量、训练细节等)。通过API仅将第三方用户暴露为黑盒。此外，由于专有原因或敏感性问题，它们甚至可能无法提供对培训数据的访问。我们首次尝试在无数据的设置中为黑盒模型提供对抗性健壮性。我们通过产生式模型构造合成数据，并使用模型窃取技术训练代理网络。为了最大限度地减少扰动样本的有害污染，我们提出了小波去噪器(WNR)，它对输入图像进行离散小波分解，并仔细地选择由我们的小波系数选择模块(WCSM)确定的几个重要系数。为了恢复图像经过WNR去噪后的高频内容，我们进一步训练了一个“再生器”网络，目的是恢复系数，使重建的图像产生与原始预测相似的代理模型。在测试时，将WNR与训练好的再生器网络相结合，加入到黑盒网络中，大大提高了对抗的准确率。与基准相比，我们的方法在CIFAR-10上的攻击准确率分别提高了38.98%和32.01%，即使攻击者使用类似于黑盒体系结构(Alexnet)的代理体系结构(Alexnet-Half和Alexnet)，并且与防御者使用相同的模型窃取策略。代码可在https://github.com/vcl-iisc/data-free-black-box-defense上获得



## **38. The Impostor Among US(B): Off-Path Injection Attacks on USB Communications**

[TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None cs.CR

To appear in USENIX Security 2023

**SubmitDate**: 2022-11-03    [abs](http://arxiv.org/abs/2211.01109v2) [paper-pdf](http://arxiv.org/pdf/2211.01109v2)

**Authors**: Robert Dumitru, Daniel Genkin, Andrew Wabnitz, Yuval Yarom

**Abstract**: USB is the most prevalent peripheral interface in modern computer systems and its inherent insecurities make it an appealing attack vector. A well-known limitation of USB is that traffic is not encrypted. This allows on-path adversaries to trivially perform man-in-the-middle attacks. Off-path attacks that compromise the confidentiality of communications have also been shown to be possible. However, so far no off-path attacks that breach USB communications integrity have been demonstrated.   In this work we show that the integrity of USB communications is not guaranteed even against off-path attackers.Specifically, we design and build malicious devices that, even when placed outside of the path between a victim device and the host, can inject data to that path. Using our developed injectors we can falsify the provenance of data input as interpreted by a host computer system. By injecting on behalf of trusted victim devices we can circumvent any software-based authorisation policy defences that computer systems employ against common USB attacks. We demonstrate two concrete attacks. The first injects keystrokes allowing an attacker to execute commands. The second demonstrates file-contents replacement including during system install from a USB disk. We test the attacks on 29 USB 2.0 and USB 3.x hubs and find 14 of them to be vulnerable.

摘要: USB是现代计算机系统中最流行的外设接口，其固有的不安全性使其成为一种吸引人的攻击媒介。USB的一个众所周知的限制是流量不加密。这使得路径上的对手可以很容易地执行中间人攻击。危害通信机密性的非路径攻击也已被证明是可能的。然而，到目前为止，还没有证明有破坏USB通信完整性的非路径攻击。在这项工作中，我们展示了即使是针对非路径攻击者，USB通信的完整性也不能得到保证。具体地说，我们设计和构建了恶意设备，即使放在受攻击设备和主机之间的路径之外，也可以向该路径注入数据。使用我们开发的注射器，我们可以伪造主机系统解释的数据输入的来源。通过代表受信任的受害者设备注入，我们可以绕过计算机系统针对常见USB攻击所采用的任何基于软件的授权策略防御。我们演示了两个具体的攻击。第一种是插入击键，允许攻击者执行命令。第二个示例演示了文件内容替换，包括在系统安装过程中从U盘进行替换。我们在29个USB 2.0和USB 3.x集线器上测试了这些攻击，发现其中14个集线器存在漏洞。



## **39. On the Adversarial Robustness of Vision Transformers**

关于视觉变形金刚的对抗稳健性 cs.CV

Published in Transactions on Machine Learning Research (TMLR). Codes  available at  https://github.com/RulinShao/on-the-adversarial-robustness-of-visual-transformer

**SubmitDate**: 2022-11-02    [abs](http://arxiv.org/abs/2103.15670v3) [paper-pdf](http://arxiv.org/pdf/2103.15670v3)

**Authors**: Rulin Shao, Zhouxing Shi, Jinfeng Yi, Pin-Yu Chen, Cho-Jui Hsieh

**Abstract**: Following the success in advancing natural language processing and understanding, transformers are expected to bring revolutionary changes to computer vision. This work provides a comprehensive study on the robustness of vision transformers (ViTs) against adversarial perturbations. Tested on various white-box and transfer attack settings, we find that ViTs possess better adversarial robustness when compared with MLP-Mixer and convolutional neural networks (CNNs) including ConvNeXt, and this observation also holds for certified robustness. Through frequency analysis and feature visualization, we summarize the following main observations contributing to the improved robustness of ViTs: 1) Features learned by ViTs contain less high-frequency patterns that have spurious correlation, which helps explain why ViTs are less sensitive to high-frequency perturbations than CNNs and MLP-Mixer, and there is a high correlation between how much the model learns high-frequency features and its robustness against different frequency-based perturbations. 2) Introducing convolutional or tokens-to-token blocks for learning high-frequency features in ViTs can improve classification accuracy but at the cost of adversarial robustness. 3) Modern CNN designs that borrow techniques from ViTs including activation function, layer norm, larger kernel size to imitate the global attention, and patchify the images as inputs, etc., could help bridge the performance gap between ViTs and CNNs not only in terms of performance, but also certified and empirical adversarial robustness. Moreover, we show adversarial training is also applicable to ViT for training robust models, and sharpness-aware minimization can also help improve robustness, while pre-training with clean images on larger datasets does not significantly improve adversarial robustness.

摘要: 随着自然语言处理和理解的成功，转换器有望给计算机视觉带来革命性的变化。这项工作提供了一个全面的研究视觉转换器(VITS)对对抗扰动的鲁棒性。在不同的白盒和传输攻击设置上进行测试，我们发现VITS与MLP-Mixer和包括ConvNeXt在内的卷积神经网络(CNN)相比具有更好的对抗健壮性，并且这一观察结果也证明了健壮性。通过频谱分析和特征可视化，我们总结了以下有助于提高VITS稳健性的主要观察结果：1)VITS学习的特征包含的具有虚假相关性的高频模式较少，这有助于解释为什么VITS对高频扰动的敏感度低于CNN和MLP-Mixer，并且模型学习高频特征的程度与其对不同频率扰动的稳健性之间存在高度相关性。2)在VITS中引入卷积或令牌到令牌块来学习高频特征可以提高分类精度，但代价是对手的健壮性。3)现代CNN设计借鉴了VITS的激活函数、层范数、更大的核大小来模拟全局注意、拼接图像作为输入等技术，不仅在性能上弥补了VITS和CNN之间的性能差距，而且在验证的和经验的对抗健壮性方面也有助于弥补VITS和CNN之间的差距。此外，我们还证明了对抗性训练也适用于VIT，用于训练稳健模型，并且清晰度感知最小化也有助于提高稳健性，而在较大的数据集上使用干净的图像进行预训练并不能显著提高对抗性鲁棒性。



## **40. Distributed Black-box Attack against Image Classification Cloud Services**

[TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None cs.LG

10 pages, 11 figures

**SubmitDate**: 2022-11-02    [abs](http://arxiv.org/abs/2210.16371v2) [paper-pdf](http://arxiv.org/pdf/2210.16371v2)

**Authors**: Han Wu, Sareh Rowlands, Johan Wahlstrom

**Abstract**: Black-box adversarial attacks can fool image classifiers into misclassifying images without requiring access to model structure and weights. Recently proposed black-box attacks can achieve a success rate of more than 95% after less than 1,000 queries. The question then arises of whether black-box attacks have become a real threat against IoT devices that rely on cloud APIs to achieve image classification. To shed some light on this, note that prior research has primarily focused on increasing the success rate and reducing the number of required queries. However, another crucial factor for black-box attacks against cloud APIs is the time required to perform the attack. This paper applies black-box attacks directly to cloud APIs rather than to local models, thereby avoiding multiple mistakes made in prior research. Further, we exploit load balancing to enable distributed black-box attacks that can reduce the attack time by a factor of about five for both local search and gradient estimation methods.

摘要: 黑盒对抗性攻击可以欺骗图像分类器对图像进行错误分类，而不需要访问模型结构和权重。最近提出的黑盒攻击在不到1000个查询的情况下可以达到95%以上的成功率。随之而来的问题是，黑盒攻击是否已经成为对依赖云API实现图像分类的物联网设备的真正威胁。为了阐明这一点，请注意，以前的研究主要集中在提高成功率和减少所需的查询数量上。然而，针对云API的黑盒攻击的另一个关键因素是执行攻击所需的时间。本文将黑盒攻击直接应用于云API，而不是本地模型，从而避免了以往研究中的多个错误。此外，我们利用负载平衡来实现分布式黑盒攻击，对于局部搜索和梯度估计方法，可以将攻击时间减少约5倍。



## **41. Isometric Representations in Neural Networks Improve Robustness**

神经网络中的等距表示提高了稳健性 cs.LG

14 pages, 4 figures

**SubmitDate**: 2022-11-02    [abs](http://arxiv.org/abs/2211.01236v1) [paper-pdf](http://arxiv.org/pdf/2211.01236v1)

**Authors**: Kosio Beshkov, Jonas Verhellen, Mikkel Elle Lepperød

**Abstract**: Artificial and biological agents cannon learn given completely random and unstructured data. The structure of data is encoded in the metric relationships between data points. In the context of neural networks, neuronal activity within a layer forms a representation reflecting the transformation that the layer implements on its inputs. In order to utilize the structure in the data in a truthful manner, such representations should reflect the input distances and thus be continuous and isometric. Supporting this statement, recent findings in neuroscience propose that generalization and robustness are tied to neural representations being continuously differentiable. In machine learning, most algorithms lack robustness and are generally thought to rely on aspects of the data that differ from those that humans use, as is commonly seen in adversarial attacks. During cross-entropy classification, the metric and structural properties of network representations are usually broken both between and within classes. This side effect from training can lead to instabilities under perturbations near locations where such structure is not preserved. One of the standard solutions to obtain robustness is to add ad hoc regularization terms, but to our knowledge, forcing representations to preserve the metric structure of the input data as a stabilising mechanism has not yet been studied. In this work, we train neural networks to perform classification while simultaneously maintaining within-class metric structure, leading to isometric within-class representations. Such network representations turn out to be beneficial for accurate and robust inference. By stacking layers with this property we create a network architecture that facilitates hierarchical manipulation of internal neural representations. Finally, we verify that isometric regularization improves the robustness to adversarial attacks on MNIST.

摘要: 给出完全随机和非结构化的数据，人工和生物制剂无法学习。数据的结构编码在数据点之间的度量关系中。在神经网络的背景下，一层内的神经元活动形成一种表示，反映该层对其输入实施的转换。为了真实地利用数据中的结构，这种表示应该反映输入距离，从而是连续的和等距的。支持这一说法的是，神经科学的最新发现表明，泛化和健壮性与神经表征的持续可微分性有关。在机器学习中，大多数算法缺乏健壮性，通常被认为依赖于与人类使用的数据不同的方面，就像在对抗性攻击中常见的那样。在交叉熵分类过程中，网络表示的度量和结构属性通常在类之间和类内都会被破坏。训练的这种副作用可能导致在这种结构未被保留的位置附近的扰动下的不稳定性。获得稳健性的标准解决方案之一是添加特别正则化项，但据我们所知，强制表示保留输入数据的度量结构作为稳定机制还没有研究过。在这项工作中，我们训练神经网络在执行分类的同时保持类内度量结构，从而导致类内等距表示。事实证明，这种网络表示有利于准确和健壮的推理。通过堆叠具有这一属性的层，我们创建了一个网络体系结构，它促进了内部神经表示的分层操作。最后，我们验证了等距正则化提高了对MNIST攻击的稳健性。



## **42. BATT: Backdoor Attack with Transformation-based Triggers**

BATT：使用基于转换的触发器进行后门攻击 cs.CR

5 pages

**SubmitDate**: 2022-11-02    [abs](http://arxiv.org/abs/2211.01806v1) [paper-pdf](http://arxiv.org/pdf/2211.01806v1)

**Authors**: Tong Xu, Yiming Li, Yong Jiang, Shu-Tao Xia

**Abstract**: Deep neural networks (DNNs) are vulnerable to backdoor attacks. The backdoor adversaries intend to maliciously control the predictions of attacked DNNs by injecting hidden backdoors that can be activated by adversary-specified trigger patterns during the training process. One recent research revealed that most of the existing attacks failed in the real physical world since the trigger contained in the digitized test samples may be different from that of the one used for training. Accordingly, users can adopt spatial transformations as the image pre-processing to deactivate hidden backdoors. In this paper, we explore the previous findings from another side. We exploit classical spatial transformations (i.e. rotation and translation) with the specific parameter as trigger patterns to design a simple yet effective poisoning-based backdoor attack. For example, only images rotated to a particular angle can activate the embedded backdoor of attacked DNNs. Extensive experiments are conducted, verifying the effectiveness of our attack under both digital and physical settings and its resistance to existing backdoor defenses.

摘要: 深度神经网络(DNN)很容易受到后门攻击。后门攻击者打算通过注入隐藏的后门来恶意控制受到攻击的DNN的预测，这些后门可以在训练过程中被对手指定的触发模式激活。最近的一项研究表明，大多数现有的攻击在现实世界中都失败了，因为数字化测试样本中包含的触发器可能不同于用于训练的触发器。相应地，用户可以采用空间变换作为图像的预处理来去激活隐藏的后门。在本文中，我们从另一个角度对前人的研究结果进行了探讨。我们利用经典的空间变换(即旋转和平移)，以特定参数作为触发模式，设计了一种简单而有效的基于中毒的后门攻击。例如，只有旋转到特定角度的图像才能激活被攻击的DNN的嵌入后门。进行了广泛的实验，验证了我们的攻击在数字和物理环境下的有效性，以及它对现有后门防御的抵抗力。



## **43. Driver Locations Harvesting Attack on pRide**

司机位置收割对Pride的攻击 cs.CR

**SubmitDate**: 2022-11-02    [abs](http://arxiv.org/abs/2210.13263v2) [paper-pdf](http://arxiv.org/pdf/2210.13263v2)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstract**: Privacy preservation in Ride-Hailing Services (RHS) is intended to protect privacy of drivers and riders. pRide, published in IEEE Trans. Vehicular Technology 2021, is a prediction based privacy-preserving RHS protocol to match riders with an optimum driver. In the protocol, the Service Provider (SP) homomorphically computes Euclidean distances between encrypted locations of drivers and rider. Rider selects an optimum driver using decrypted distances augmented by a new-ride-emergence prediction. To improve the effectiveness of driver selection, the paper proposes an enhanced version where each driver gives encrypted distances to each corner of her grid. To thwart a rider from using these distances to launch an inference attack, the SP blinds these distances before sharing them with the rider. In this work, we propose a passive attack where an honest-but-curious adversary rider who makes a single ride request and receives the blinded distances from SP can recover the constants used to blind the distances. Using the unblinded distances, rider to driver distance and Google Nearest Road API, the adversary can obtain the precise locations of responding drivers. We conduct experiments with random on-road driver locations for four different cities. Our experiments show that we can determine the precise locations of at least 80% of the drivers participating in the enhanced pRide protocol.

摘要: 网约车服务(RHS)中的隐私保护旨在保护司机和乘客的隐私。Pride，发表在IEEE Trans上。Vehicular Technology 2021是一种基于预测的隐私保护RHS协议，用于将乘客与最佳司机进行匹配。在该协议中，服务提供商(SP)同态地计算司机和乘客的加密位置之间的欧几里德距离。骑手使用解密的距离选择最优的司机，并增加了一个新的乘车出现预测。为了提高驾驶员选择的有效性，本文提出了一种增强版本，每个驾驶员给出了到其网格每个角落的加密距离。为了阻止骑手使用这些距离来发动推理攻击，SP在与骑手共享这些距离之前会先隐藏这些距离。在这项工作中，我们提出了一种被动攻击，在这种攻击中，诚实但好奇的敌方骑手发出一个骑行请求，并从SP接收到盲距离，就可以恢复用于盲距离的常量。使用非盲目距离、骑手到司机的距离和谷歌最近道路API，对手可以获得回应司机的准确位置。我们对四个不同城市的随机道路司机位置进行了实验。我们的实验表明，我们可以确定至少80%参与增强PROID协议的司机的准确位置。



## **44. Defending with Errors: Approximate Computing for Robustness of Deep Neural Networks**

错误防御：深度神经网络稳健性的近似计算 cs.CR

arXiv admin note: substantial text overlap with arXiv:2006.07700

**SubmitDate**: 2022-11-02    [abs](http://arxiv.org/abs/2211.01182v1) [paper-pdf](http://arxiv.org/pdf/2211.01182v1)

**Authors**: Amira Guesmi, Ihsen Alouani, Khaled N. Khasawneh, Mouna Baklouti, Tarek Frikha, Mohamed Abid, Nael Abu-Ghazaleh

**Abstract**: Machine-learning architectures, such as Convolutional Neural Networks (CNNs) are vulnerable to adversarial attacks: inputs crafted carefully to force the system output to a wrong label. Since machine-learning is being deployed in safety-critical and security-sensitive domains, such attacks may have catastrophic security and safety consequences. In this paper, we propose for the first time to use hardware-supported approximate computing to improve the robustness of machine-learning classifiers. We show that successful adversarial attacks against the exact classifier have poor transferability to the approximate implementation. Surprisingly, the robustness advantages also apply to white-box attacks where the attacker has unrestricted access to the approximate classifier implementation: in this case, we show that substantially higher levels of adversarial noise are needed to produce adversarial examples. Furthermore, our approximate computing model maintains the same level in terms of classification accuracy, does not require retraining, and reduces resource utilization and energy consumption of the CNN. We conducted extensive experiments on a set of strong adversarial attacks; We empirically show that the proposed implementation increases the robustness of a LeNet-5, Alexnet and VGG-11 CNNs considerably with up to 50% by-product saving in energy consumption due to the simpler nature of the approximate logic.

摘要: [TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None



## **45. a-RNA: Adversarial Radio Noise Attack to Fool Radar-based Environment Perception Systems**

[TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None cs.CR

**SubmitDate**: 2022-11-02    [abs](http://arxiv.org/abs/2211.01112v1) [paper-pdf](http://arxiv.org/pdf/2211.01112v1)

**Authors**: Amira Guesmi, Ihsen Alouani

**Abstract**: Due to their robustness to degraded capturing conditions, radars are widely used for environment perception, which is a critical task in applications like autonomous vehicles. More specifically, Ultra-Wide Band (UWB) radars are particularly efficient for short range settings as they carry rich information on the environment. Recent UWB-based systems rely on Machine Learning (ML) to exploit the rich signature of these sensors. However, ML classifiers are susceptible to adversarial examples, which are created from raw data to fool the classifier such that it assigns the input to the wrong class. These attacks represent a serious threat to systems integrity, especially for safety-critical applications. In this work, we present a new adversarial attack on UWB radars in which an adversary injects adversarial radio noise in the wireless channel to cause an obstacle recognition failure. First, based on signals collected in real-life environment, we show that conventional attacks fail to generate robust noise under realistic conditions. We propose a-RNA, i.e., Adversarial Radio Noise Attack to overcome these issues. Specifically, a-RNA generates an adversarial noise that is efficient without synchronization between the input signal and the noise. Moreover, a-RNA generated noise is, by-design, robust against pre-processing countermeasures such as filtering-based defenses. Moreover, in addition to the undetectability objective by limiting the noise magnitude budget, a-RNA is also efficient in the presence of sophisticated defenses in the spectral domain by introducing a frequency budget. We believe this work should alert about potentially critical implementations of adversarial attacks on radar systems that should be taken seriously.

摘要: [TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None



## **46. Improving transferability of 3D adversarial attacks with scale and shear transformations**

利用尺度和剪切变换提高3D对抗性攻击的可转移性 cs.CV

10 pages

**SubmitDate**: 2022-11-02    [abs](http://arxiv.org/abs/2211.01093v1) [paper-pdf](http://arxiv.org/pdf/2211.01093v1)

**Authors**: Jinali Zhang, Yinpeng Dong, Jun Zhu, Jihong Zhu, Minchi Kuang, Xiaming Yuan

**Abstract**: Previous work has shown that 3D point cloud classifiers can be vulnerable to adversarial examples. However, most of the existing methods are aimed at white-box attacks, where the parameters and other information of the classifiers are known in the attack, which is unrealistic for real-world applications. In order to improve the attack performance of the black-box classifiers, the research community generally uses the transfer-based black-box attack. However, the transferability of current 3D attacks is still relatively low. To this end, this paper proposes Scale and Shear (SS) Attack to generate 3D adversarial examples with strong transferability. Specifically, we randomly scale or shear the input point cloud, so that the attack will not overfit the white-box model, thereby improving the transferability of the attack. Extensive experiments show that the SS attack proposed in this paper can be seamlessly combined with the existing state-of-the-art (SOTA) 3D point cloud attack methods to form more powerful attack methods, and the SS attack improves the transferability over 3.6 times compare to the baseline. Moreover, while substantially outperforming the baseline methods, the SS attack achieves SOTA transferability under various defenses. Our code will be available online at https://github.com/cuge1995/SS-attack

摘要: 以前的工作已经表明，三维点云分类器可能容易受到敌意示例的攻击。然而，现有的大多数方法都是针对白盒攻击的，在白盒攻击中，分类器的参数和其他信息是已知的，这对于现实世界的应用是不现实的。为了提高黑盒分类器的攻击性能，研究界普遍采用基于转移的黑盒攻击。然而，目前3D攻击的可转移性仍然相对较低。为此，本文提出了一种基于尺度和切变的3D对抗性实例生成方法，具有很强的可转移性。具体地说，我们对输入点云进行随机缩放或剪切，使攻击不会过度符合白盒模型，从而提高了攻击的可转移性。大量实验表明，SS攻击可以与现有的SOTA三维点云攻击方法无缝结合，形成更强大的攻击方法，与基线相比，SS攻击的可转移性提高了3.6倍以上。此外，在大大超过基线方法的同时，SS攻击在各种防御下实现了SOTA的可转移性。我们的代码将在https://github.com/cuge1995/SS-attack上在线提供



## **47. Projective Ranking-based GNN Evasion Attacks**

[TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None cs.LG

Accepted by IEEE Transactions on Knowledge and Data Engineering

**SubmitDate**: 2022-11-02    [abs](http://arxiv.org/abs/2202.12993v2) [paper-pdf](http://arxiv.org/pdf/2202.12993v2)

**Authors**: He Zhang, Xingliang Yuan, Chuan Zhou, Shirui Pan

**Abstract**: Graph neural networks (GNNs) offer promising learning methods for graph-related tasks. However, GNNs are at risk of adversarial attacks. Two primary limitations of the current evasion attack methods are highlighted: (1) The current GradArgmax ignores the "long-term" benefit of the perturbation. It is faced with zero-gradient and invalid benefit estimates in certain situations. (2) In the reinforcement learning-based attack methods, the learned attack strategies might not be transferable when the attack budget changes. To this end, we first formulate the perturbation space and propose an evaluation framework and the projective ranking method. We aim to learn a powerful attack strategy then adapt it as little as possible to generate adversarial samples under dynamic budget settings. In our method, based on mutual information, we rank and assess the attack benefits of each perturbation for an effective attack strategy. By projecting the strategy, our method dramatically minimizes the cost of learning a new attack strategy when the attack budget changes. In the comparative assessment with GradArgmax and RL-S2V, the results show our method owns high attack performance and effective transferability. The visualization of our method also reveals various attack patterns in the generation of adversarial samples.

摘要: 图神经网络(GNN)为与图相关的任务提供了很有前途的学习方法。然而，GNN面临着遭到对抗性攻击的风险。指出了当前规避攻击方法的两个主要局限性：(1)当前的GradArgmax忽略了扰动的“长期”益处。在某些情况下，它面临着零梯度和无效的收益估计。(2)在基于强化学习的攻击方法中，当攻击预算发生变化时，学习到的攻击策略可能无法迁移。为此，我们首先定义了扰动空间，并提出了评价框架和投影排序方法。我们的目标是学习一种强大的攻击策略，然后尽可能少地对其进行调整，以在动态预算设置下生成对抗性样本。在我们的方法中，基于互信息，我们对有效的攻击策略的每个扰动的攻击收益进行排序和评估。通过预测策略，当攻击预算发生变化时，我们的方法显著地将学习新攻击策略的成本降至最低。在与GradArgmax和RL-S2V的对比评估中，结果表明该方法具有较高的攻击性能和有效的可转移性。我们方法的可视化还揭示了在生成对抗性样本时的各种攻击模式。



## **48. Certified Robustness of Quantum Classifiers against Adversarial Examples through Quantum Noise**

[TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None quant-ph

Submitted to IEEE ICASSP 2023

**SubmitDate**: 2022-11-02    [abs](http://arxiv.org/abs/2211.00887v1) [paper-pdf](http://arxiv.org/pdf/2211.00887v1)

**Authors**: Jhih-Cing Huang, Yu-Lin Tsai, Chao-Han Huck Yang, Cheng-Fang Su, Chia-Mu Yu, Pin-Yu Chen, Sy-Yen Kuo

**Abstract**: Recently, quantum classifiers have been known to be vulnerable to adversarial attacks, where quantum classifiers are fooled by imperceptible noises to have misclassification. In this paper, we propose one first theoretical study that utilizing the added quantum random rotation noise can improve the robustness of quantum classifiers against adversarial attacks. We connect the definition of differential privacy and demonstrate the quantum classifier trained with the natural presence of additive noise is differentially private. Lastly, we derive a certified robustness bound to enable quantum classifiers to defend against adversarial examples supported by experimental results.

摘要: 最近，人们已经知道量子分类器容易受到敌意攻击，即量子分类器被难以察觉的噪声愚弄而产生错误分类。在本文中，我们首次提出了利用附加的量子随机旋转噪声来提高量子分类器对敌意攻击的稳健性的理论研究。我们联系了差分隐私的定义，并证明了在自然存在加性噪声的情况下训练的量子分类器是差分隐私的。最后，我们得到了一个证明的健壮性界限，使量子分类器能够防御实验结果支持的敌意例子。



## **49. LMD: A Learnable Mask Network to Detect Adversarial Examples for Speaker Verification**

LMD：一种用于说话人确认的可学习掩码网络 eess.AS

13 pages, 9 figures

**SubmitDate**: 2022-11-02    [abs](http://arxiv.org/abs/2211.00825v1) [paper-pdf](http://arxiv.org/pdf/2211.00825v1)

**Authors**: Xing Chen, Jie Wang, Xiao-Lei Zhang, Wei-Qiang Zhang, Kunde Yang

**Abstract**: Although the security of automatic speaker verification (ASV) is seriously threatened by recently emerged adversarial attacks, there have been some countermeasures to alleviate the threat. However, many defense approaches not only require the prior knowledge of the attackers but also possess weak interpretability. To address this issue, in this paper, we propose an attacker-independent and interpretable method, named learnable mask detector (LMD), to separate adversarial examples from the genuine ones. It utilizes score variation as an indicator to detect adversarial examples, where the score variation is the absolute discrepancy between the ASV scores of an original audio recording and its transformed audio synthesized from its masked complex spectrogram. A core component of the score variation detector is to generate the masked spectrogram by a neural network. The neural network needs only genuine examples for training, which makes it an attacker-independent approach. Its interpretability lies that the neural network is trained to minimize the score variation of the targeted ASV, and maximize the number of the masked spectrogram bins of the genuine training examples. Its foundation is based on the observation that, masking out the vast majority of the spectrogram bins with little speaker information will inevitably introduce a large score variation to the adversarial example, and a small score variation to the genuine example. Experimental results with 12 attackers and two representative ASV systems show that our proposed method outperforms five state-of-the-art baselines. The extensive experimental results can also be a benchmark for the detection-based ASV defenses.

摘要: 尽管自动说话人确认(ASV)的安全性受到最近出现的敌意攻击的严重威胁，但已经有一些对策来缓解这种威胁。然而，许多防御方法不仅需要攻击者的先验知识，而且具有较弱的可解释性。针对这一问题，本文提出了一种独立于攻击者且可解释的方法，称为可学习掩码检测器(LMD)，用于区分敌意实例和真实实例。它利用分数变化作为检测敌意例子的指标，其中分数变化是原始音频记录的ASV分数与从其掩蔽的复谱图合成的变换音频之间的绝对差异。分数变化检测器的核心部件是通过神经网络生成被屏蔽的谱图。神经网络只需要真实的样本进行训练，这使它成为一种独立于攻击者的方法。它的可解释性在于神经网络被训练来最小化目标ASV的分数差异，并最大化真实训练样本的掩蔽谱图库的数量。它的基础是观察到，在几乎没有说话人信息的情况下掩蔽绝大多数的语谱库将不可避免地给对抗性例子带来大的分数变化，而对真实的例子带来小的分数变化。对12个攻击者和两个有代表性的ASV系统的实验结果表明，我们提出的方法的性能超过了五个最先进的基线。广泛的实验结果也可以作为基于检测的ASV防御的基准。



## **50. On the detection of synthetic images generated by diffusion models**

关于扩散模型生成的合成图像的检测 cs.CV

**SubmitDate**: 2022-11-01    [abs](http://arxiv.org/abs/2211.00680v1) [paper-pdf](http://arxiv.org/pdf/2211.00680v1)

**Authors**: Riccardo Corvi, Davide Cozzolino, Giada Zingarini, Giovanni Poggi, Koki Nagano, Luisa Verdoliva

**Abstract**: Over the past decade, there has been tremendous progress in creating synthetic media, mainly thanks to the development of powerful methods based on generative adversarial networks (GAN). Very recently, methods based on diffusion models (DM) have been gaining the spotlight. In addition to providing an impressive level of photorealism, they enable the creation of text-based visual content, opening up new and exciting opportunities in many different application fields, from arts to video games. On the other hand, this property is an additional asset in the hands of malicious users, who can generate and distribute fake media perfectly adapted to their attacks, posing new challenges to the media forensic community. With this work, we seek to understand how difficult it is to distinguish synthetic images generated by diffusion models from pristine ones and whether current state-of-the-art detectors are suitable for the task. To this end, first we expose the forensics traces left by diffusion models, then study how current detectors, developed for GAN-generated images, perform on these new synthetic images, especially in challenging social-networks scenarios involving image compression and resizing. Datasets and code are available at github.com/grip-unina/DMimageDetection.

摘要: 在过去的十年里，在创造合成媒体方面取得了巨大的进步，这主要归功于基于生成性对抗网络(GAN)的强大方法的发展。最近，基于扩散模型(DM)的方法得到了广泛的关注。除了提供令人印象深刻的照片真实感，它们还使基于文本的视觉内容的创建成为可能，在从艺术到视频游戏的许多不同应用领域开辟了新的令人兴奋的机会。另一方面，这一财产是恶意用户手中的额外资产，他们可以生成和传播完全适应他们的攻击的虚假媒体，给媒体取证社区提出了新的挑战。通过这项工作，我们试图了解区分由扩散模型生成的合成图像和原始图像有多难，以及当前最先进的探测器是否适合这项任务。为此，我们首先揭示扩散模型留下的取证痕迹，然后研究为GaN生成的图像开发的当前检测器如何在这些新的合成图像上执行，特别是在涉及图像压缩和大小调整的具有挑战性的社交网络场景中。数据集和代码可在githorb.com/执掌-unina/DMImageDetect上找到。



