# Latest Adversarial Attack Papers
**update at 2024-12-20 16:22:25**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. AutoTrust: Benchmarking Trustworthiness in Large Vision Language Models for Autonomous Driving**

AutoTrust：自动驾驶大视觉语言模型的可信度基准 cs.CV

55 pages, 14 figures

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.15206v1) [paper-pdf](http://arxiv.org/pdf/2412.15206v1)

**Authors**: Shuo Xing, Hongyuan Hua, Xiangbo Gao, Shenzhe Zhu, Renjie Li, Kexin Tian, Xiaopeng Li, Heng Huang, Tianbao Yang, Zhangyang Wang, Yang Zhou, Huaxiu Yao, Zhengzhong Tu

**Abstract**: Recent advancements in large vision language models (VLMs) tailored for autonomous driving (AD) have shown strong scene understanding and reasoning capabilities, making them undeniable candidates for end-to-end driving systems. However, limited work exists on studying the trustworthiness of DriveVLMs -- a critical factor that directly impacts public transportation safety. In this paper, we introduce AutoTrust, a comprehensive trustworthiness benchmark for large vision-language models in autonomous driving (DriveVLMs), considering diverse perspectives -- including trustfulness, safety, robustness, privacy, and fairness. We constructed the largest visual question-answering dataset for investigating trustworthiness issues in driving scenarios, comprising over 10k unique scenes and 18k queries. We evaluated six publicly available VLMs, spanning from generalist to specialist, from open-source to commercial models. Our exhaustive evaluations have unveiled previously undiscovered vulnerabilities of DriveVLMs to trustworthiness threats. Specifically, we found that the general VLMs like LLaVA-v1.6 and GPT-4o-mini surprisingly outperform specialized models fine-tuned for driving in terms of overall trustworthiness. DriveVLMs like DriveLM-Agent are particularly vulnerable to disclosing sensitive information. Additionally, both generalist and specialist VLMs remain susceptible to adversarial attacks and struggle to ensure unbiased decision-making across diverse environments and populations. Our findings call for immediate and decisive action to address the trustworthiness of DriveVLMs -- an issue of critical importance to public safety and the welfare of all citizens relying on autonomous transportation systems. Our benchmark is publicly available at \url{https://github.com/taco-group/AutoTrust}, and the leaderboard is released at \url{https://taco-group.github.io/AutoTrust/}.

摘要: 为自动驾驶(AD)量身定做的大型视觉语言模型(VLM)最近的进步显示了强大的场景理解和推理能力，使它们成为端到端驾驶系统的不可否认的候选者。然而，目前对DriveVLMS可信度的研究工作有限--这是直接影响公共交通安全的关键因素。在本文中，我们介绍了AutoTrust，这是一个针对自动驾驶中大型视觉语言模型(DriveVLMS)的综合可信度基准，考虑了不同的角度--包括可信性、安全性、健壮性、隐私和公平性。我们构建了最大的可视化问答数据集，用于调查驾驶场景中的可信度问题，包括超过10k个独特的场景和18k个查询。我们评估了六个公开可用的VLM，从通才到专家，从开源到商业模型。我们的详尽评估揭示了DriveVLM对可信度威胁之前未发现的漏洞。具体地说，我们发现像LLaVA-v1.6和GPT-40-mini这样的普通VLM在总体可信度方面出人意料地超过了专门为驾驶而调整的车型。像DriveLM-Agent这样的DriveVLM特别容易泄露敏感信息。此外，通才和专业的VLM仍然容易受到对抗性攻击，并努力确保在不同的环境和人群中做出公正的决策。我们的调查结果要求立即采取果断行动，解决DriveVLMS的可信性问题--这是一个对公共安全和依赖自动交通系统的所有公民的福利至关重要的问题。我们的基准在\url{https://github.com/taco-group/AutoTrust}，上公开可用，排行榜在\url{https://taco-group.github.io/AutoTrust/}.上发布



## **2. Do Parameters Reveal More than Loss for Membership Inference?**

参数揭示的不仅仅是会员推断的损失吗？ cs.LG

Accepted to Transactions on Machine Learning Research (TMLR)

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2406.11544v4) [paper-pdf](http://arxiv.org/pdf/2406.11544v4)

**Authors**: Anshuman Suri, Xiao Zhang, David Evans

**Abstract**: Membership inference attacks are used as a key tool for disclosure auditing. They aim to infer whether an individual record was used to train a model. While such evaluations are useful to demonstrate risk, they are computationally expensive and often make strong assumptions about potential adversaries' access to models and training environments, and thus do not provide tight bounds on leakage from potential attacks. We show how prior claims around black-box access being sufficient for optimal membership inference do not hold for stochastic gradient descent, and that optimal membership inference indeed requires white-box access. Our theoretical results lead to a new white-box inference attack, IHA (Inverse Hessian Attack), that explicitly uses model parameters by taking advantage of computing inverse-Hessian vector products. Our results show that both auditors and adversaries may be able to benefit from access to model parameters, and we advocate for further research into white-box methods for membership inference.

摘要: 成员关系推断攻击被用作信息披露审计的关键工具。他们的目的是推断个人记录是否被用来训练模型。虽然这样的评估有助于显示风险，但它们的计算成本很高，而且通常会对潜在对手访问模型和训练环境做出强有力的假设，因此不会对潜在攻击的泄漏提供严格的限制。我们证明了关于黑盒访问的关于最优成员关系推理的先前声明如何不适用于随机梯度下降，而最优成员关系推理确实需要白盒访问。我们的理论结果导致了一种新的白盒推理攻击，IHA(逆向Hessian攻击)，它通过计算逆向Hessian向量积来显式地使用模型参数。我们的结果表明，审计师和对手都可能从访问模型参数中受益，我们主张进一步研究成员关系推理的白盒方法。



## **3. Accuracy Limits as a Barrier to Biometric System Security**

准确性限制是生物识别系统安全性的障碍 cs.CR

14 pages, 4 figures

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.13099v2) [paper-pdf](http://arxiv.org/pdf/2412.13099v2)

**Authors**: Axel Durbet, Paul-Marie Grollemund, Pascal Lafourcade, Kevin Thiry-Atighehchi

**Abstract**: Biometric systems are widely used for identity verification and identification, including authentication (i.e., one-to-one matching to verify a claimed identity) and identification (i.e., one-to-many matching to find a subject in a database). The matching process relies on measuring similarities or dissimilarities between a fresh biometric template and enrolled templates. The False Match Rate FMR is a key metric for assessing the accuracy and reliability of such systems. This paper analyzes biometric systems based on their FMR, with two main contributions. First, we explore untargeted attacks, where an adversary aims to impersonate any user within a database. We determine the number of trials required for an attacker to successfully impersonate a user and derive the critical population size (i.e., the maximum number of users in the database) required to maintain a given level of security. Furthermore, we compute the critical FMR value needed to ensure resistance against untargeted attacks as the database size increases. Second, we revisit the biometric birthday problem to evaluate the approximate and exact probabilities that two users in a database collide (i.e., can impersonate each other). Based on this analysis, we derive both the approximate critical population size and the critical FMR value needed to bound the likelihood of such collisions occurring with a given probability. These thresholds offer insights for designing systems that mitigate the risk of impersonation and collisions, particularly in large-scale biometric databases. Our findings indicate that current biometric systems fail to deliver sufficient accuracy to achieve an adequate security level against untargeted attacks, even in small-scale databases. Moreover, state-of-the-art systems face significant challenges in addressing the biometric birthday problem, especially as database sizes grow.

摘要: 生物测定系统广泛用于身份验证和身份识别，包括身份验证(即，一对一匹配以验证所声称的身份)和身份(即，一对多匹配以在数据库中找到对象)。匹配过程依赖于测量新的生物特征模板和注册模板之间的相似性或差异性。误匹配率FMR是评估这类系统的准确性和可靠性的关键指标。本文分析了基于FMR的生物特征识别系统，主要有两个贡献。首先，我们探索非目标攻击，其中对手的目标是模拟数据库中的任何用户。我们确定攻击者成功模拟用户所需的试验次数，并得出维持给定安全级别所需的临界总体大小(即，数据库中的最大用户数)。此外，随着数据库大小的增加，我们还计算了确保抵抗非目标攻击所需的关键FMR值。其次，我们回顾了生物统计生日问题，以评估数据库中两个用户发生冲突(即可以相互冒充)的近似和准确概率。基于这一分析，我们推导出了以给定概率限制这种碰撞发生的可能性所需的近似临界种群大小和临界FMR值。这些阈值为设计降低模仿和冲突风险的系统提供了见解，特别是在大规模生物识别数据库中。我们的发现表明，即使在小规模的数据库中，当前的生物识别系统也无法提供足够的准确性来实现足够的安全级别来抵御非目标攻击。此外，最先进的系统在解决生物识别生日问题方面面临着重大挑战，特别是随着数据库大小的增长。



## **4. SLIFER: Investigating Performance and Robustness of Malware Detection Pipelines**

SIFER：调查恶意软件检测管道的性能和稳健性 cs.CR

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2405.14478v3) [paper-pdf](http://arxiv.org/pdf/2405.14478v3)

**Authors**: Andrea Ponte, Dmitrijs Trizna, Luca Demetrio, Battista Biggio, Ivan Tesfai Ogbu, Fabio Roli

**Abstract**: As a result of decades of research, Windows malware detection is approached through a plethora of techniques. However, there is an ongoing mismatch between academia -- which pursues an optimal performances in terms of detection rate and low false alarms -- and the requirements of real-world scenarios. In particular, academia focuses on combining static and dynamic analysis within a single or ensemble of models, falling into several pitfalls like (i) firing dynamic analysis without considering the computational burden it requires; (ii) discarding impossible-to-analyze samples; and (iii) analyzing robustness against adversarial attacks without considering that malware detectors are complemented with more non-machine-learning components. Thus, in this paper we bridge these gaps, by investigating the properties of malware detectors built with multiple and different types of analysis. To do so, we develop SLIFER, a Windows malware detection pipeline sequentially leveraging both static and dynamic analysis, interrupting computations as soon as one module triggers an alarm, requiring dynamic analysis only when needed. Contrary to the state of the art, we investigate how to deal with samples that impede analyzes, showing how much they impact performances, concluding that it is better to flag them as legitimate to not drastically increase false alarms. Lastly, we perform a robustness evaluation of SLIFER. Counter-intuitively, the injection of new content is either blocked more by signatures than dynamic analysis, due to byte artifacts created by the attack, or it is able to avoid detection from signatures, as they rely on constraints on file size disrupted by attacks. As far as we know, we are the first to investigate the properties of sequential malware detectors, shedding light on their behavior in real production environment.

摘要: 作为数十年研究的结果，Windows恶意软件检测是通过大量技术实现的。然而，学术界--追求在检测率和低虚警方面的最佳表现--与现实世界场景的要求之间存在着持续的不匹配。特别是，学术界专注于在单个或集成模型中结合静态和动态分析，陷入了几个陷阱，如(I)触发动态分析而不考虑其所需的计算负担；(Ii)丢弃无法分析的样本；以及(Iii)分析针对对手攻击的稳健性，而不考虑恶意软件检测器与更多非机器学习组件的补充。因此，在本文中，我们通过调查使用多种不同类型的分析构建的恶意软件检测器的属性来弥合这些差距。为此，我们开发了Slifer，这是一条Windows恶意软件检测管道，顺序地利用静态和动态分析，在一个模块触发警报时立即中断计算，仅在需要时才需要动态分析。与现有技术相反，我们调查了如何处理阻碍分析的样本，显示了它们对性能的影响程度，得出的结论是，最好将它们标记为合法，而不是大幅增加错误警报。最后，我们对Slifer算法进行了健壮性评估。与直觉相反的是，由于攻击产生的字节伪像，新内容的注入更多地被签名阻止，而不是动态分析，或者它能够避免从签名检测，因为它们依赖于被攻击破坏的文件大小限制。据我们所知，我们是第一个研究顺序恶意软件检测器的性质的人，揭示了它们在实际生产环境中的行为。



## **5. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

卡西姆·分歧：通过对抗性防御的扩散模型来启发性解纠缠 cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2410.23091v5) [paper-pdf](http://arxiv.org/pdf/2410.23091v5)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark). The code is available at https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff

摘要: 尽管不断努力保护神经分类器免受对手攻击，但它们仍然很脆弱，特别是面对看不见的攻击。相比之下，人类很难被微妙的操纵所欺骗，因为我们只根据基本因素做出判断。受到这一观察的启发，我们试图用基本的标签原因因素来建模标签生成，并结合标签非原因因素来辅助数据生成。对于一个对抗性的例子，我们的目标是将扰动区分为非致因因素，并仅基于标签致因因素进行预测。具体地说，我们提出了一个偶然扩散模型(CausalDiff)，该模型使扩散模型适用于条件数据生成，并通过向一个新的偶然信息瓶颈目标学习来区分这两种类型的偶然因素。经验上，CausalDiff在各种隐形攻击上的表现明显优于最先进的防御方法，在CIFAR-10上获得了86.39%(+4.01%)的平均健壮性，在CIFAR-100上获得了56.25%(+3.13%)的健壮性，在GTSRB(德国交通标志识别基准)上实现了82.62%(+4.93%)的平均健壮性。代码可在https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff上获得



## **6. Grimm: A Plug-and-Play Perturbation Rectifier for Graph Neural Networks Defending against Poisoning Attacks**

Grimm：用于图神经网络防御中毒攻击的即插即用微扰矫正器 cs.LG

19 pages, 13 figures

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.08555v2) [paper-pdf](http://arxiv.org/pdf/2412.08555v2)

**Authors**: Ao Liu, Wenshan Li, Beibei Li, Wengang Ma, Tao Li, Pan Zhou

**Abstract**: Recent studies have revealed the vulnerability of graph neural networks (GNNs) to adversarial poisoning attacks on node classification tasks. Current defensive methods require substituting the original GNNs with defense models, regardless of the original's type. This approach, while targeting adversarial robustness, compromises the enhancements developed in prior research to boost GNNs' practical performance. Here we introduce Grimm, the first plug-and-play defense model. With just a minimal interface requirement for extracting features from any layer of the protected GNNs, Grimm is thus enabled to seamlessly rectify perturbations. Specifically, we utilize the feature trajectories (FTs) generated by GNNs, as they evolve through epochs, to reflect the training status of the networks. We then theoretically prove that the FTs of victim nodes will inevitably exhibit discriminable anomalies. Consequently, inspired by the natural parallelism between the biological nervous and immune systems, we construct Grimm, a comprehensive artificial immune system for GNNs. Grimm not only detects abnormal FTs and rectifies adversarial edges during training but also operates efficiently in parallel, thereby mirroring the concurrent functionalities of its biological counterparts. We experimentally confirm that Grimm offers four empirically validated advantages: 1) Harmlessness, as it does not actively interfere with GNN training; 2) Parallelism, ensuring monitoring, detection, and rectification functions operate independently of the GNN training process; 3) Generalizability, demonstrating compatibility with mainstream GNNs such as GCN, GAT, and GraphSAGE; and 4) Transferability, as the detectors for abnormal FTs can be efficiently transferred across different systems for one-step rectification.

摘要: 最近的研究揭示了图神经网络(GNN)对节点分类任务的敌意中毒攻击的脆弱性。目前的防御方法需要用防御模型取代原始GNN，而不考虑原始GNN的类型。虽然这种方法的目标是对抗的稳健性，但它损害了先前研究中为提高GNN的实际性能而开发的增强。在这里，我们介绍格林，第一个即插即用的防御模型。从受保护的GNN的任何一层中提取特征只需要最小的接口要求，因此GRIMM就能够无缝地纠正扰动。具体地说，我们利用GNN生成的特征轨迹(FTs)来反映网络的训练状态。然后，我们从理论上证明了受害节点的FT将不可避免地表现出可区分的异常。因此，受生物神经系统和免疫系统的自然并行性的启发，我们构建了一个用于GNN的全面的人工免疫系统GRIMM。GRIMM不仅在训练过程中检测异常FT并纠正敌对边缘，而且还可以高效地并行操作，从而反映出其生物同行的并发功能。我们通过实验证实GRIMM提供了四个经验证的优势：1)无害，因为它不会主动干扰GNN训练；2)并行性，确保监测、检测和纠正功能独立于GNN训练过程运行；3)通用性，表现出与GCN、GAT和GraphSAGE等主流GNN的兼容性；以及4)可转移性，因为异常FT的检测器可以在不同的系统中高效传输，以便一步纠正。



## **7. DG-Mamba: Robust and Efficient Dynamic Graph Structure Learning with Selective State Space Models**

DG-Mamba：使用选择性状态空间模型稳健高效的动态图结构学习 cs.LG

Accepted by the Main Technical Track of the 39th Annual AAAI  Conference on Artificial Intelligence (AAAI-2025)

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.08160v4) [paper-pdf](http://arxiv.org/pdf/2412.08160v4)

**Authors**: Haonan Yuan, Qingyun Sun, Zhaonan Wang, Xingcheng Fu, Cheng Ji, Yongjian Wang, Bo Jin, Jianxin Li

**Abstract**: Dynamic graphs exhibit intertwined spatio-temporal evolutionary patterns, widely existing in the real world. Nevertheless, the structure incompleteness, noise, and redundancy result in poor robustness for Dynamic Graph Neural Networks (DGNNs). Dynamic Graph Structure Learning (DGSL) offers a promising way to optimize graph structures. However, aside from encountering unacceptable quadratic complexity, it overly relies on heuristic priors, making it hard to discover underlying predictive patterns. How to efficiently refine the dynamic structures, capture intrinsic dependencies, and learn robust representations, remains under-explored. In this work, we propose the novel DG-Mamba, a robust and efficient Dynamic Graph structure learning framework with the Selective State Space Models (Mamba). To accelerate the spatio-temporal structure learning, we propose a kernelized dynamic message-passing operator that reduces the quadratic time complexity to linear. To capture global intrinsic dynamics, we establish the dynamic graph as a self-contained system with State Space Model. By discretizing the system states with the cross-snapshot graph adjacency, we enable the long-distance dependencies capturing with the selective snapshot scan. To endow learned dynamic structures more expressive with informativeness, we propose the self-supervised Principle of Relevant Information for DGSL to regularize the most relevant yet least redundant information, enhancing global robustness. Extensive experiments demonstrate the superiority of the robustness and efficiency of our DG-Mamba compared with the state-of-the-art baselines against adversarial attacks.

摘要: 动态图形表现出交织在一起的时空演化模式，广泛存在于现实世界中。然而，动态图神经网络的结构不完备性、噪声和冗余性导致其健壮性较差。动态图结构学习(DGSL)为优化图结构提供了一种很有前途的方法。然而，除了遇到不可接受的二次型复杂性外，它还过度依赖启发式先验，使得发现潜在的预测模式变得困难。如何有效地提炼动态结构，捕获内在依赖关系，并学习健壮的表示，仍未得到探索。在这项工作中，我们提出了一种新颖的DG-MAMBA，这是一种基于选择状态空间模型(MAMBA)的健壮而高效的动态图结构学习框架。为了加速时空结构的学习，我们提出了一种核化的动态消息传递算子，将二次时间复杂度降为线性。为了捕捉全局内在动力学，我们利用状态空间模型将动态图建立为一个自包含系统。通过使用交叉快照图邻接关系对系统状态进行离散化，实现了选择性快照扫描的远程依赖捕获。为了使学习到的动态结构具有更强的信息性，我们提出了DGSL的相关信息自监督原则，将相关程度最高但冗余最少的信息正则化，增强了全局鲁棒性。大量的实验证明了DG-MAMBA算法的健壮性和高效性，与目前最先进的对抗攻击基线算法相比具有更好的性能。



## **8. How Does the Smoothness Approximation Method Facilitate Generalization for Federated Adversarial Learning?**

光滑度逼近方法如何促进联邦对抗学习的推广？ cs.LG

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.08282v2) [paper-pdf](http://arxiv.org/pdf/2412.08282v2)

**Authors**: Wenjun Ding, Ying An, Lixing Chen, Shichao Kan, Fan Wu, Zhe Qu

**Abstract**: Federated Adversarial Learning (FAL) is a robust framework for resisting adversarial attacks on federated learning. Although some FAL studies have developed efficient algorithms, they primarily focus on convergence performance and overlook generalization. Generalization is crucial for evaluating algorithm performance on unseen data. However, generalization analysis is more challenging due to non-smooth adversarial loss functions. A common approach to addressing this issue is to leverage smoothness approximation. In this paper, we develop algorithm stability measures to evaluate the generalization performance of two popular FAL algorithms: \textit{Vanilla FAL (VFAL)} and {\it Slack FAL (SFAL)}, using three different smooth approximation methods: 1) \textit{Surrogate Smoothness Approximation (SSA)}, (2) \textit{Randomized Smoothness Approximation (RSA)}, and (3) \textit{Over-Parameterized Smoothness Approximation (OPSA)}. Based on our in-depth analysis, we answer the question of how to properly set the smoothness approximation method to mitigate generalization error in FAL. Moreover, we identify RSA as the most effective method for reducing generalization error. In highly data-heterogeneous scenarios, we also recommend employing SFAL to mitigate the deterioration of generalization performance caused by heterogeneity. Based on our theoretical results, we provide insights to help develop more efficient FAL algorithms, such as designing new metrics and dynamic aggregation rules to mitigate heterogeneity.

摘要: 联合对抗学习(FAL)是一种用于抵抗对联合学习的敌意攻击的健壮框架。虽然一些FAL研究已经开发出了有效的算法，但它们主要集中在收敛性能上，而忽略了泛化。泛化是在未知数据上评估算法性能的关键。然而，由于对抗性损失函数的非光滑性，泛化分析具有更大的挑战性。解决此问题的一种常见方法是利用平滑近似。本文利用3种不同的光滑逼近方法：1)替代平滑逼近(SSA)}、(2)随机平滑逼近(RSA)}和(3)过参数平滑逼近(OPSA)}，对两种常见的FAL算法在深入分析的基础上，我们回答了如何合理地设置光滑度逼近方法来减小FAL中的泛化误差的问题。此外，我们认为RSA是减少泛化误差的最有效方法。在数据高度异构性的场景中，我们还建议使用SFAL来缓解异构性导致的泛化性能下降。基于我们的理论结果，我们提供了一些见解来帮助开发更高效的FAL算法，例如设计新的度量和动态聚集规则来缓解异构性。



## **9. Unleashing the Unseen: Harnessing Benign Datasets for Jailbreaking Large Language Models**

释放隐形：利用良性数据集破解大型语言模型 cs.CR

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2410.00451v3) [paper-pdf](http://arxiv.org/pdf/2410.00451v3)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Jun Sun

**Abstract**: Despite significant ongoing efforts in safety alignment, large language models (LLMs) such as GPT-4 and LLaMA 3 remain vulnerable to jailbreak attacks that can induce harmful behaviors, including through the use of adversarial suffixes. Building on prior research, we hypothesize that these adversarial suffixes are not mere bugs but may represent features that can dominate the LLM's behavior. To evaluate this hypothesis, we conduct several experiments. First, we demonstrate that benign features can be effectively made to function as adversarial suffixes, i.e., we develop a feature extraction method to extract sample-agnostic features from benign dataset in the form of suffixes and show that these suffixes may effectively compromise safety alignment. Second, we show that adversarial suffixes generated from jailbreak attacks may contain meaningful features, i.e., appending the same suffix to different prompts results in responses exhibiting specific characteristics. Third, we show that such benign-yet-safety-compromising features can be easily introduced through fine-tuning using only benign datasets. As a result, we are able to completely eliminate GPT's safety alignment in a blackbox setting through finetuning with only benign data. Our code and data is available at \url{https://github.com/suffix-maybe-feature/adver-suffix-maybe-features}.

摘要: 尽管在安全匹配方面正在进行重大努力，但GPT-4和大羊驼3等大型语言模型仍然容易受到越狱攻击，这些攻击可能会导致有害行为，包括通过使用对抗性后缀。在先前研究的基础上，我们假设这些对抗性后缀不仅仅是错误，而且可能代表可以主导LLM行为的特征。为了评估这一假设，我们进行了几个实验。首先，我们证明了良性特征可以有效地用作对抗性后缀，即，我们开发了一种特征提取方法来从良性数据集中提取与样本无关的后缀形式的特征，并表明这些后缀可以有效地危害安全对齐。其次，我们证明了越狱攻击产生的对抗性后缀可能包含有意义的特征，即在不同的提示后添加相同的后缀会导致响应表现出特定的特征。第三，我们表明，仅使用良性数据集，通过微调就可以很容易地引入这种良性但危及安全的特征。因此，我们能够通过仅使用良性数据进行微调，在黑盒设置中完全消除GPT的安全对齐。我们的代码和数据可在\url{https://github.com/suffix-maybe-feature/adver-suffix-maybe-features}.上获得



## **10. Doubly-Universal Adversarial Perturbations: Deceiving Vision-Language Models Across Both Images and Text with a Single Perturbation**

双重普遍对抗性扰动：通过单一扰动欺骗图像和文本的视觉语言模型 cs.CV

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.08108v2) [paper-pdf](http://arxiv.org/pdf/2412.08108v2)

**Authors**: Hee-Seon Kim, Minbeom Kim, Changick Kim

**Abstract**: Large Vision-Language Models (VLMs) have demonstrated remarkable performance across multimodal tasks by integrating vision encoders with large language models (LLMs). However, these models remain vulnerable to adversarial attacks. Among such attacks, Universal Adversarial Perturbations (UAPs) are especially powerful, as a single optimized perturbation can mislead the model across various input images. In this work, we introduce a novel UAP specifically designed for VLMs: the Doubly-Universal Adversarial Perturbation (Doubly-UAP), capable of universally deceiving VLMs across both image and text inputs. To successfully disrupt the vision encoder's fundamental process, we analyze the core components of the attention mechanism. After identifying value vectors in the middle-to-late layers as the most vulnerable, we optimize Doubly-UAP in a label-free manner with a frozen model. Despite being developed as a black-box to the LLM, Doubly-UAP achieves high attack success rates on VLMs, consistently outperforming baseline methods across vision-language tasks. Extensive ablation studies and analyses further demonstrate the robustness of Doubly-UAP and provide insights into how it influences internal attention mechanisms.

摘要: 大视觉语言模型(VLM)通过将视觉编码器与大语言模型(LLM)相结合，在多通道任务中表现出了显著的性能。然而，这些模型仍然容易受到对手的攻击。在这些攻击中，通用对抗性扰动(UAP)尤其强大，因为单个优化的扰动可以在不同的输入图像上误导模型。在这项工作中，我们介绍了一种新的专门针对VLMS设计的UAP：双重通用对抗性摄动(Double-Universal Aversarial微扰，Double-UAP)，能够在图像和文本输入之间普遍欺骗VLMS。为了成功地扰乱视觉编码器的基本过程，我们分析了注意机制的核心组件。在确定中后期价值向量最易受攻击后，我们使用冻结模型以无标签的方式对Double-UAP进行优化。尽管被开发为LLM的黑匣子，Double-UAP在VLM上实现了高攻击成功率，在视觉语言任务中始终优于基线方法。广泛的消融研究和分析进一步证明了Double-UAP的健壮性，并提供了对其如何影响内部注意机制的见解。



## **11. Towards Provable Security in Industrial Control Systems Via Dynamic Protocol Attestation**

通过动态协议认证实现工业控制系统的可证明安全性 cs.CR

This paper was accepted into the ICSS'24 workshop

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.14467v1) [paper-pdf](http://arxiv.org/pdf/2412.14467v1)

**Authors**: Arthur Amorim, Trevor Kann, Max Taylor, Lance Joneckis

**Abstract**: Industrial control systems (ICSs) increasingly rely on digital technologies vulnerable to cyber attacks. Cyber attackers can infiltrate ICSs and execute malicious actions. Individually, each action seems innocuous. But taken together, they cause the system to enter an unsafe state. These attacks have resulted in dramatic consequences such as physical damage, economic loss, and environmental catastrophes. This paper introduces a methodology that restricts actions using protocols. These protocols only allow safe actions to execute. Protocols are written in a domain specific language we have embedded in an interactive theorem prover (ITP). The ITP enables formal, machine-checked proofs to ensure protocols maintain safety properties. We use dynamic attestation to ensure ICSs conform to their protocol even if an adversary compromises a component. Since protocol conformance prevents unsafe actions, the previously mentioned cyber attacks become impossible. We demonstrate the effectiveness of our methodology using an example from the Fischertechnik Industry 4.0 platform. We measure dynamic attestation's impact on latency and throughput. Our approach is a starting point for studying how to combine formal methods and protocol design to thwart attacks intended to cripple ICSs.

摘要: 工业控制系统(ICSS)越来越依赖易受网络攻击的数字技术。网络攻击者可以渗透到ICSS中并执行恶意操作。单独来看，每一项行动似乎都是无害的。但综合起来，它们会导致系统进入不安全状态。这些袭击造成了严重的后果，如物质损失、经济损失和环境灾难。本文介绍了一种使用协议限制操作的方法。这些协议只允许执行安全操作。协议是用我们嵌入到交互式定理证明器(ITP)中的特定于领域的语言编写的。ITP允许正式的机器检查的证明，以确保协议维护安全属性。我们使用动态证明来确保ICSS符合他们的协议，即使对手破坏了组件。由于协议一致性可以防止不安全的行为，因此前面提到的网络攻击变得不可能。我们以Fischer Technik Industry 4.0平台为例，验证了该方法的有效性。我们测量了动态证明对延迟和吞吐量的影响。我们的方法是研究如何结合形式化方法和协议设计来挫败旨在削弱ICSS的攻击的一个起点。



## **12. Adversarial Hubness in Multi-Modal Retrieval**

多模式检索中的对抗性积极性 cs.CR

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.14113v1) [paper-pdf](http://arxiv.org/pdf/2412.14113v1)

**Authors**: Tingwei Zhang, Fnu Suya, Rishi Jha, Collin Zhang, Vitaly Shmatikov

**Abstract**: Hubness is a phenomenon in high-dimensional vector spaces where a single point from the natural distribution is unusually close to many other points. This is a well-known problem in information retrieval that causes some items to accidentally (and incorrectly) appear relevant to many queries. In this paper, we investigate how attackers can exploit hubness to turn any image or audio input in a multi-modal retrieval system into an adversarial hub. Adversarial hubs can be used to inject universal adversarial content (e.g., spam) that will be retrieved in response to thousands of different queries, as well as for targeted attacks on queries related to specific, attacker-chosen concepts. We present a method for creating adversarial hubs and evaluate the resulting hubs on benchmark multi-modal retrieval datasets and an image-to-image retrieval system based on a tutorial from Pinecone, a popular vector database. For example, in text-caption-to-image retrieval, a single adversarial hub is retrieved as the top-1 most relevant image for more than 21,000 out of 25,000 test queries (by contrast, the most common natural hub is the top-1 response to only 102 queries). We also investigate whether techniques for mitigating natural hubness are an effective defense against adversarial hubs, and show that they are not effective against hubs that target queries related to specific concepts.

摘要: Hubness是高维向量空间中的一种现象，在高维向量空间中，自然分布中的单个点异常接近许多其他点。这是信息检索中的一个众所周知的问题，它会导致某些项意外地(并且不正确地)显示为与许多查询相关。在本文中，我们研究攻击者如何利用Hubness将多模式检索系统中的任何图像或音频输入转换为敌对中心。对抗性集线器可用于注入将响应于数千个不同查询而检索的通用对抗性内容(例如，垃圾邮件)，以及用于针对与特定的攻击者选择的概念相关的查询的定向攻击。我们提出了一种创建敌意中心的方法，并在基准多模式检索数据集和基于流行的矢量数据库Pinecone的教程的图像到图像检索系统上对生成的中心进行了评估。例如，在文本字幕到图像的检索中，对于25,000个测试查询中的超过21,000个，单个敌意中心被检索为最相关的前1个图像(相比之下，最常见的自然中心是仅对102个查询的前1个响应)。我们还调查了缓解自然中心的技术是否是对抗中心的有效防御，并表明它们对针对与特定概念相关的查询的中心无效。



## **13. Certification of Speaker Recognition Models to Additive Perturbations**

说话人识别模型对加性扰动的认证 cs.SD

13 pages, 10 figures; AAAI-2025 accepted paper

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2404.18791v2) [paper-pdf](http://arxiv.org/pdf/2404.18791v2)

**Authors**: Dmitrii Korzh, Elvir Karimov, Mikhail Pautov, Oleg Y. Rogov, Ivan Oseledets

**Abstract**: Speaker recognition technology is applied to various tasks, from personal virtual assistants to secure access systems. However, the robustness of these systems against adversarial attacks, particularly to additive perturbations, remains a significant challenge. In this paper, we pioneer applying robustness certification techniques to speaker recognition, initially developed for the image domain. Our work covers this gap by transferring and improving randomized smoothing certification techniques against norm-bounded additive perturbations for classification and few-shot learning tasks to speaker recognition. We demonstrate the effectiveness of these methods on VoxCeleb 1 and 2 datasets for several models. We expect this work to improve the robustness of voice biometrics and accelerate the research of certification methods in the audio domain.

摘要: 说话人识别技术应用于各种任务，从个人虚拟助理到安全访问系统。然而，这些系统对对抗攻击（特别是对添加性扰动）的鲁棒性仍然是一个重大挑战。在本文中，我们率先将鲁棒性认证技术应用于说话人识别，该技术最初是针对图像领域开发的。我们的工作通过转移和改进随机平滑认证技术来弥补这一差距，以对抗分类和少数镜头学习任务的规范界添加性扰动。我们在VoxCeleb 1和2数据集上证明了这些方法对于多个模型的有效性。我们希望这项工作能够提高语音生物识别技术的稳健性，并加速音频领域认证方法的研究。



## **14. Adversarial Robustness of Link Sign Prediction in Signed Graphs**

带符号图中链接符号预测的对抗鲁棒性 cs.LG

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2401.10590v2) [paper-pdf](http://arxiv.org/pdf/2401.10590v2)

**Authors**: Jialong Zhou, Xing Ai, Yuni Lai, Tomasz Michalak, Gaolei Li, Jianhua Li, Kai Zhou

**Abstract**: Signed graphs serve as fundamental data structures for representing positive and negative relationships in social networks, with signed graph neural networks (SGNNs) emerging as the primary tool for their analysis. Our investigation reveals that balance theory, while essential for modeling signed relationships in SGNNs, inadvertently introduces exploitable vulnerabilities to black-box attacks. To demonstrate this vulnerability, we propose balance-attack, a novel adversarial strategy specifically designed to compromise graph balance degree, and develop an efficient heuristic algorithm to solve the associated NP-hard optimization problem. While existing approaches attempt to restore attacked graphs through balance learning techniques, they face a critical challenge we term "Irreversibility of Balance-related Information," where restored edges fail to align with original attack targets. To address this limitation, we introduce Balance Augmented-Signed Graph Contrastive Learning (BA-SGCL), an innovative framework that combines contrastive learning with balance augmentation techniques to achieve robust graph representations. By maintaining high balance degree in the latent space, BA-SGCL effectively circumvents the irreversibility challenge and enhances model resilience. Extensive experiments across multiple SGNN architectures and real-world datasets demonstrate both the effectiveness of our proposed balance-attack and the superior robustness of BA-SGCL, advancing the security and reliability of signed graph analysis in social networks. Datasets and codes of the proposed framework are at the github repository https://anonymous.4open.science/r/BA-SGCL-submit-DF41/.

摘要: 带符号图是表示社会网络中正负关系的基本数据结构，带符号图神经网络(SGNN)是分析这些关系的主要工具。我们的研究表明，平衡理论虽然对SGNN中的签名关系建模是必不可少的，但它无意中为黑盒攻击引入了可利用的漏洞。为了证明这一漏洞，我们提出了一种新的对抗性策略--平衡攻击，它是专门为折衷图平衡度而设计的，并开发了一个有效的启发式算法来解决相关的NP-Hard优化问题。虽然现有的方法试图通过平衡学习技术来恢复被攻击的图形，但它们面临着一个关键的挑战，我们将其称为“与平衡相关的信息的不可逆性”，其中恢复的边无法与原始攻击目标对齐。为了解决这一局限性，我们引入了平衡增强符号图对比学习(BA-SGCL)，这是一个结合对比学习和平衡增强技术的创新框架，以实现健壮的图表示。BA-SGCL通过在潜在空间保持较高的平衡度，有效地规避了模型的不可逆性挑战，增强了模型的弹性。在多个SGNN体系结构和真实数据集上的大量实验证明了我们提出的平衡攻击的有效性和BA-SGCL的卓越健壮性，从而提高了社交网络中签名图分析的安全性和可靠性。拟议框架的数据集和代码位于GitHub存储库https://anonymous.4open.science/r/BA-SGCL-submit-DF41/.



## **15. A Review of the Duality of Adversarial Learning in Network Intrusion: Attacks and Countermeasures**

网络入侵中对抗学习的二重性：攻击与对策 cs.CR

23 pages, 2 figures, 5 tables

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13880v1) [paper-pdf](http://arxiv.org/pdf/2412.13880v1)

**Authors**: Shalini Saini, Anitha Chennamaneni, Babatunde Sawyerr

**Abstract**: Deep learning solutions are instrumental in cybersecurity, harnessing their ability to analyze vast datasets, identify complex patterns, and detect anomalies. However, malevolent actors can exploit these capabilities to orchestrate sophisticated attacks, posing significant challenges to defenders and traditional security measures. Adversarial attacks, particularly those targeting vulnerabilities in deep learning models, present a nuanced and substantial threat to cybersecurity. Our study delves into adversarial learning threats such as Data Poisoning, Test Time Evasion, and Reverse Engineering, specifically impacting Network Intrusion Detection Systems. Our research explores the intricacies and countermeasures of attacks to deepen understanding of network security challenges amidst adversarial threats. In our study, we present insights into the dynamic realm of adversarial learning and its implications for network intrusion. The intersection of adversarial attacks and defenses within network traffic data, coupled with advances in machine learning and deep learning techniques, represents a relatively underexplored domain. Our research lays the groundwork for strengthening defense mechanisms to address the potential breaches in network security and privacy posed by adversarial attacks. Through our in-depth analysis, we identify domain-specific research gaps, such as the scarcity of real-life attack data and the evaluation of AI-based solutions for network traffic. Our focus on these challenges aims to stimulate future research efforts toward the development of resilient network defense strategies.

摘要: 深度学习解决方案在网络安全方面非常重要，可以利用它们分析海量数据集、识别复杂模式和检测异常的能力。然而，恶意行为者可以利用这些能力来策划复杂的攻击，给防御者和传统安全措施带来重大挑战。对抗性攻击，特别是针对深度学习模型中的漏洞的攻击，对网络安全构成了微妙而实质性的威胁。我们的研究深入到对抗性学习威胁，如数据中毒、测试时间逃避和逆向工程，特别是影响网络入侵检测系统。我们的研究探索了攻击的复杂性和对策，以加深对对手威胁中的网络安全挑战的理解。在我们的研究中，我们提出了对对抗学习的动态领域及其对网络入侵的影响的见解。网络流量数据中的对抗性攻击和防御的交集，再加上机器学习和深度学习技术的进步，代表着一个相对未被探索的领域。我们的研究为加强防御机制以应对对抗性攻击在网络安全和隐私方面的潜在破坏奠定了基础。通过我们的深入分析，我们找出了特定领域的研究差距，例如真实攻击数据的稀缺和基于人工智能的网络流量解决方案的评估。我们对这些挑战的关注旨在刺激未来开发弹性网络防御战略的研究努力。



## **16. Cultivating Archipelago of Forests: Evolving Robust Decision Trees through Island Coevolution**

培育森林群岛：通过岛屿共同进化进化稳健决策树 cs.LG

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13762v1) [paper-pdf](http://arxiv.org/pdf/2412.13762v1)

**Authors**: Adam Żychowski, Andrew Perrault, Jacek Mańdziuk

**Abstract**: Decision trees are widely used in machine learning due to their simplicity and interpretability, but they often lack robustness to adversarial attacks and data perturbations. The paper proposes a novel island-based coevolutionary algorithm (ICoEvoRDF) for constructing robust decision tree ensembles. The algorithm operates on multiple islands, each containing populations of decision trees and adversarial perturbations. The populations on each island evolve independently, with periodic migration of top-performing decision trees between islands. This approach fosters diversity and enhances the exploration of the solution space, leading to more robust and accurate decision tree ensembles. ICoEvoRDF utilizes a popular game theory concept of mixed Nash equilibrium for ensemble weighting, which further leads to improvement in results. ICoEvoRDF is evaluated on 20 benchmark datasets, demonstrating its superior performance compared to state-of-the-art methods in optimizing both adversarial accuracy and minimax regret. The flexibility of ICoEvoRDF allows for the integration of decision trees from various existing methods, providing a unified framework for combining diverse solutions. Our approach offers a promising direction for developing robust and interpretable machine learning models

摘要: 决策树因其简单性和可解释性在机器学习中得到了广泛的应用，但它们对敌意攻击和数据扰动往往缺乏健壮性。提出了一种基于孤岛的协同进化算法(ICoEvoRDF)，用于构造稳健的决策树集成。该算法在多个孤岛上运行，每个孤岛包含决策树种群和对抗性扰动。每个岛屿上的种群独立进化，表现最好的决策树在岛屿之间定期迁移。这种方法促进了多样性并增强了对解空间的探索，导致了更健壮和更准确的决策树集成。ICoEvoRDF使用了一个流行的博弈论概念-混合纳什均衡来进行集成加权，这进一步导致了结果的改进。ICoEvoRDF在20个基准数据集上进行了评估，显示出与最先进的方法相比，它在优化对手准确性和最小最大遗憾方面的优越性能。ICoEvoRDF的灵活性允许集成来自各种现有方法的决策树，为组合不同的解决方案提供统一的框架。我们的方法为开发健壮和可解释的机器学习模型提供了一个很有前途的方向



## **17. A2RNet: Adversarial Attack Resilient Network for Robust Infrared and Visible Image Fusion**

A2 RNet：用于鲁棒的红外和可见光图像融合的对抗攻击弹性网络 cs.CV

9 pages, 8 figures, The 39th Annual AAAI Conference on Artificial  Intelligence

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.09954v2) [paper-pdf](http://arxiv.org/pdf/2412.09954v2)

**Authors**: Jiawei Li, Hongwei Yu, Jiansheng Chen, Xinlong Ding, Jinlong Wang, Jinyuan Liu, Bochao Zou, Huimin Ma

**Abstract**: Infrared and visible image fusion (IVIF) is a crucial technique for enhancing visual performance by integrating unique information from different modalities into one fused image. Exiting methods pay more attention to conducting fusion with undisturbed data, while overlooking the impact of deliberate interference on the effectiveness of fusion results. To investigate the robustness of fusion models, in this paper, we propose a novel adversarial attack resilient network, called $\textrm{A}^{\textrm{2}}$RNet. Specifically, we develop an adversarial paradigm with an anti-attack loss function to implement adversarial attacks and training. It is constructed based on the intrinsic nature of IVIF and provide a robust foundation for future research advancements. We adopt a Unet as the pipeline with a transformer-based defensive refinement module (DRM) under this paradigm, which guarantees fused image quality in a robust coarse-to-fine manner. Compared to previous works, our method mitigates the adverse effects of adversarial perturbations, consistently maintaining high-fidelity fusion results. Furthermore, the performance of downstream tasks can also be well maintained under adversarial attacks. Code is available at https://github.com/lok-18/A2RNet.

摘要: 红外与可见光图像融合(IVIF)是通过将来自不同模式的独特信息融合到一幅融合图像中来提高视觉性能的关键技术。现有的方法更注重对未受干扰的数据进行融合，而忽略了有意干扰对融合结果有效性的影响。为了研究融合模型的稳健性，本文提出了一种新的对抗攻击弹性网络，称为$\tExtm{A}^{\tExtm{2}}$rnet。具体地说，我们开发了一个具有抗攻击损失函数的对抗性范例来实施对抗性攻击和训练。它是基于IVIF的内在本质而构建的，并为未来的研究进展提供了坚实的基础。在该模型下，我们采用了基于变换的防御性细化模块(DRM)作为流水线，保证了从粗到精的融合图像质量。与以前的工作相比，我们的方法减轻了对抗性扰动的不利影响，一致地保持了高保真的融合结果。此外，在对抗性攻击下，下游任务的性能也能得到很好的维持。代码可在https://github.com/lok-18/A2RNet.上找到



## **18. Physics-Based Adversarial Attack on Near-Infrared Human Detector for Nighttime Surveillance Camera Systems**

针对夜间监控摄像机系统近红外人体探测器的基于物理的对抗攻击 cs.CV

Appeared in ACM MM 2023

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13709v1) [paper-pdf](http://arxiv.org/pdf/2412.13709v1)

**Authors**: Muyao Niu, Zhuoxiao Li, Yifan Zhan, Huy H. Nguyen, Isao Echizen, Yinqiang Zheng

**Abstract**: Many surveillance cameras switch between daytime and nighttime modes based on illuminance levels. During the day, the camera records ordinary RGB images through an enabled IR-cut filter. At night, the filter is disabled to capture near-infrared (NIR) light emitted from NIR LEDs typically mounted around the lens. While RGB-based AI algorithm vulnerabilities have been widely reported, the vulnerabilities of NIR-based AI have rarely been investigated. In this paper, we identify fundamental vulnerabilities in NIR-based image understanding caused by color and texture loss due to the intrinsic characteristics of clothes' reflectance and cameras' spectral sensitivity in the NIR range. We further show that the nearly co-located configuration of illuminants and cameras in existing surveillance systems facilitates concealing and fully passive attacks in the physical world. Specifically, we demonstrate how retro-reflective and insulation plastic tapes can manipulate the intensity distribution of NIR images. We showcase an attack on the YOLO-based human detector using binary patterns designed in the digital space (via black-box query and searching) and then physically realized using tapes pasted onto clothes. Our attack highlights significant reliability concerns for nighttime surveillance systems, which are intended to enhance security. Codes Available: https://github.com/MyNiuuu/AdvNIR

摘要: 许多监控摄像头根据照度级别在白天和夜间模式之间切换。白天，相机通过启用的IR-Cut滤镜记录普通RGB图像。在夜间，滤光片被禁用以捕获通常安装在镜头周围的近红外LED发出的近红外(NIR)光。虽然基于RGB的人工智能算法漏洞已经被广泛报道，但基于近红外的人工智能漏洞很少被调查。在本文中，我们找出了基于近红外图像理解的基本缺陷，这些缺陷是由于衣服的反射率和相机在近红外范围内的光谱敏感度的固有特性造成的颜色和纹理的损失。我们进一步表明，在现有的监控系统中，光源和摄像机几乎位于同一位置的配置有助于在物理世界中进行隐蔽和完全被动的攻击。具体地说，我们演示了反向反射和绝缘塑料胶带如何操纵近红外图像的强度分布。我们展示了对基于YOLO的人体探测器的攻击，使用在数字空间设计的二进制模式(通过黑盒查询和搜索)，然后使用粘贴在衣服上的磁带物理实现。我们的攻击突显了人们对夜间监控系统可靠性的重大担忧，这些系统旨在增强安全性。可用代码：https://github.com/MyNiuuu/AdvNIR



## **19. Mitigating Adversarial Attacks in LLMs through Defensive Suffix Generation**

通过防御性后缀生成缓解LLM中的对抗攻击 cs.CV

9 pages, 2 figures

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13705v1) [paper-pdf](http://arxiv.org/pdf/2412.13705v1)

**Authors**: Minkyoung Kim, Yunha Kim, Hyeram Seo, Heejung Choi, Jiye Han, Gaeun Kee, Soyoung Ko, HyoJe Jung, Byeolhee Kim, Young-Hak Kim, Sanghyun Park, Tae Joon Jun

**Abstract**: Large language models (LLMs) have exhibited outstanding performance in natural language processing tasks. However, these models remain susceptible to adversarial attacks in which slight input perturbations can lead to harmful or misleading outputs. A gradient-based defensive suffix generation algorithm is designed to bolster the robustness of LLMs. By appending carefully optimized defensive suffixes to input prompts, the algorithm mitigates adversarial influences while preserving the models' utility. To enhance adversarial understanding, a novel total loss function ($L_{\text{total}}$) combining defensive loss ($L_{\text{def}}$) and adversarial loss ($L_{\text{adv}}$) generates defensive suffixes more effectively. Experimental evaluations conducted on open-source LLMs such as Gemma-7B, mistral-7B, Llama2-7B, and Llama2-13B show that the proposed method reduces attack success rates (ASR) by an average of 11\% compared to models without defensive suffixes. Additionally, the perplexity score of Gemma-7B decreased from 6.57 to 3.93 when applying the defensive suffix generated by openELM-270M. Furthermore, TruthfulQA evaluations demonstrate consistent improvements with Truthfulness scores increasing by up to 10\% across tested configurations. This approach significantly enhances the security of LLMs in critical applications without requiring extensive retraining.

摘要: 大型语言模型(LLM)在自然语言处理任务中表现出优异的性能。然而，这些模型仍然容易受到对抗性攻击，在这种攻击中，轻微的输入扰动可能会导致有害或误导性的输出。设计了一种基于梯度的防御性后缀生成算法，增强了LLMS的健壮性。通过在输入提示中添加经过精心优化的防御性后缀，该算法在保持模型实用性的同时减轻了对抗性影响。为了增强对对手的理解，一种新的总损失函数($L_{\Text{TOTAL}}$)结合了防御损失($L_{\Text{def}}$)和对抗性损失($L_{\Text{adv}}$)，更有效地生成防御后缀。在Gema-7B、Mistral-7B、Llama2-7B和Llama2-13B等开源LLMS上进行的实验评估表明，与没有防御后缀的模型相比，该方法的攻击成功率(ASR)平均降低了11%。此外，使用由OpenELM-270M生成的防御后缀后，GEMA-7B的困惑分数从6.57降至3.93。此外，TruthfulQA评估显示出持续的改进，在测试的配置中，真实性分数提高了高达10%。这种方法显著增强了关键应用中的低成本管理系统的安全性，而无需进行广泛的再培训。



## **20. Enhancing Adversarial Transferability with Adversarial Weight Tuning**

通过对抗权重调整增强对抗可移植性 cs.CR

Accepted by AAAI 2025

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2408.09469v3) [paper-pdf](http://arxiv.org/pdf/2408.09469v3)

**Authors**: Jiahao Chen, Zhou Feng, Rui Zeng, Yuwen Pu, Chunyi Zhou, Yi Jiang, Yuyou Gan, Jinbao Li, Shouling Ji

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial examples (AEs) that mislead the model while appearing benign to human observers. A critical concern is the transferability of AEs, which enables black-box attacks without direct access to the target model. However, many previous attacks have failed to explain the intrinsic mechanism of adversarial transferability. In this paper, we rethink the property of transferable AEs and reformalize the formulation of transferability. Building on insights from this mechanism, we analyze the generalization of AEs across models with different architectures and prove that we can find a local perturbation to mitigate the gap between surrogate and target models. We further establish the inner connections between model smoothness and flat local maxima, both of which contribute to the transferability of AEs. Further, we propose a new adversarial attack algorithm, \textbf{A}dversarial \textbf{W}eight \textbf{T}uning (AWT), which adaptively adjusts the parameters of the surrogate model using generated AEs to optimize the flat local maxima and model smoothness simultaneously, without the need for extra data. AWT is a data-free tuning method that combines gradient-based and model-based attack methods to enhance the transferability of AEs. Extensive experiments on a variety of models with different architectures on ImageNet demonstrate that AWT yields superior performance over other attacks, with an average increase of nearly 5\% and 10\% attack success rates on CNN-based and Transformer-based models, respectively, compared to state-of-the-art attacks.

摘要: 深度神经网络(DNN)很容易受到敌意例子(AE)的攻击，这些例子误导了模型，同时对人类观察者来说是良性的。一个关键的问题是AEs的可转移性，这使得黑盒攻击能够在不直接访问目标模型的情况下进行。然而，以往的许多攻击都未能解释对抗性转移的内在机制。在本文中，我们重新思考了可转让实体的性质，并对可转让的提法进行了改造。在此机制的基础上，我们分析了不同体系结构模型之间的AEs泛化，并证明了我们可以找到局部扰动来缓解代理模型和目标模型之间的差距。我们进一步建立了模型光滑性与平坦局部极大值之间的内在联系，这两者都有助于AEs的可转移性。在此基础上，提出了一种新的对抗性攻击算法AWT是一种无数据调整方法，它结合了基于梯度和基于模型的攻击方法来增强AE的可转移性。在ImageNet上对不同体系结构的各种模型进行的大量实验表明，AWT的攻击性能优于其他攻击，基于CNN和基于Transformer的模型的攻击成功率比最先进的攻击分别提高了近5%和10%。



## **21. Understanding Key Point Cloud Features for Development Three-dimensional Adversarial Attacks**

了解关键点云功能以开发三维对抗攻击 cs.CV

10 pages, 6 figures

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2210.14164v4) [paper-pdf](http://arxiv.org/pdf/2210.14164v4)

**Authors**: Hanieh Naderi, Chinthaka Dinesh, Ivan V. Bajic, Shohreh Kasaei

**Abstract**: Adversarial attacks pose serious challenges for deep neural network (DNN)-based analysis of various input signals. In the case of three-dimensional point clouds, methods have been developed to identify points that play a key role in network decision, and these become crucial in generating existing adversarial attacks. For example, a saliency map approach is a popular method for identifying adversarial drop points, whose removal would significantly impact the network decision. This paper seeks to enhance the understanding of three-dimensional adversarial attacks by exploring which point cloud features are most important for predicting adversarial points. Specifically, Fourteen key point cloud features such as edge intensity and distance from the centroid are defined, and multiple linear regression is employed to assess their predictive power for adversarial points. Based on critical feature selection insights, a new attack method has been developed to evaluate whether the selected features can generate an attack successfully. Unlike traditional attack methods that rely on model-specific vulnerabilities, this approach focuses on the intrinsic characteristics of the point clouds themselves. It is demonstrated that these features can predict adversarial points across four different DNN architectures, Point Network (PointNet), PointNet++, Dynamic Graph Convolutional Neural Networks (DGCNN), and Point Convolutional Network (PointConv) outperforming random guessing and achieving results comparable to saliency map-based attacks. This study has important engineering applications, such as enhancing the security and robustness of three-dimensional point cloud-based systems in fields like robotics and autonomous driving.

摘要: 对抗性攻击对基于深度神经网络(DNN)的各种输入信号分析提出了严峻的挑战。在三维点云的情况下，已经开发出方法来识别在网络决策中起关键作用的点，并且这些点在产生现有的对抗性攻击时变得至关重要。例如，显著图方法是一种流行的识别对抗性丢弃点的方法，其移除将显著影响网络决策。本文试图通过探索哪些点云特征对预测对抗性点最重要，来提高对三维对抗性攻击的理解。具体地，定义了14个关键点云特征，如边缘强度和到质心的距离，并使用多元线性回归来评估它们对敌对点的预测能力。基于关键特征选择的洞察力，提出了一种新的攻击方法，用于评估所选择的特征是否能够成功地产生攻击。与依赖于模型特定漏洞的传统攻击方法不同，该方法侧重于点云本身的内在特征。实验表明，这些特征可以在点网络(PointNet)、点网络++、动态图卷积神经网络(DGCNN)和点卷积网络(PointConv)四种不同的DNN体系结构上预测敌对点，其性能优于随机猜测，并获得与基于显著图的攻击相当的结果。该研究具有重要的工程应用价值，例如在机器人和自动驾驶等领域中增强基于三维点云的系统的安全性和健壮性。



## **22. Novel AI Camera Camouflage: Face Cloaking Without Full Disguise**

新颖的人工智能相机伪装：没有完全伪装的面部伪装 cs.CV

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13507v1) [paper-pdf](http://arxiv.org/pdf/2412.13507v1)

**Authors**: David Noever, Forrest McKee

**Abstract**: This study demonstrates a novel approach to facial camouflage that combines targeted cosmetic perturbations and alpha transparency layer manipulation to evade modern facial recognition systems. Unlike previous methods -- such as CV dazzle, adversarial patches, and theatrical disguises -- this work achieves effective obfuscation through subtle modifications to key-point regions, particularly the brow, nose bridge, and jawline. Empirical testing with Haar cascade classifiers and commercial systems like BetaFaceAPI and Microsoft Bing Visual Search reveals that vertical perturbations near dense facial key points significantly disrupt detection without relying on overt disguises. Additionally, leveraging alpha transparency attacks in PNG images creates a dual-layer effect: faces remain visible to human observers but disappear in machine-readable RGB layers, rendering them unidentifiable during reverse image searches. The results highlight the potential for creating scalable, low-visibility facial obfuscation strategies that balance effectiveness and subtlety, opening pathways for defeating surveillance while maintaining plausible anonymity.

摘要: 这项研究展示了一种新的面部伪装方法，该方法结合了有针对性的化妆品扰动和阿尔法透明层操作来逃避现代面部识别系统。与以前的方法不同--如令人眼花缭乱的简历、对抗性的补丁和戏剧性的伪装--这项工作通过对关键点区域进行微妙的修改，特别是眉毛、鼻梁和下巴轮廓，实现了有效的混淆。使用Haar级联分类器以及BetaFaceAPI和Microsoft Bing Visual Search等商业系统进行的经验测试表明，密集面部关键点附近的垂直扰动显著干扰检测，而不需要公开的伪装。此外，在PNG图像中利用Alpha透明度攻击会产生双层效果：人脸对人类观察者仍然可见，但在机器可读的RGB层中消失，从而使它们在反向图像搜索期间无法识别。这些结果突显了创建可扩展、低能见度的面部混淆策略的潜力，该策略平衡了有效性和微妙程度，为击败监视开辟了道路，同时保持了看似合理的匿名性。



## **23. Safeguarding Virtual Healthcare: A Novel Attacker-Centric Model for Data Security and Privacy**

保护虚拟医疗保健：一种以攻击者为中心的数据安全和隐私的新型模型 cs.CR

6 pages, 3 figures, 3 tables

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13440v1) [paper-pdf](http://arxiv.org/pdf/2412.13440v1)

**Authors**: Suvineetha Herath, Haywood Gelman, John Hastings, Yong Wang

**Abstract**: The rapid growth of remote healthcare delivery has introduced significant security and privacy risks to protected health information (PHI). Analysis of a comprehensive healthcare security breach dataset covering 2009-2023 reveals their significant prevalence and impact. This study investigates the root causes of such security incidents and introduces the Attacker-Centric Approach (ACA), a novel threat model tailored to protect PHI. ACA addresses limitations in existing threat models and regulatory frameworks by adopting a holistic attacker-focused perspective, examining threats from the viewpoint of cyber adversaries, their motivations, tactics, and potential attack vectors. Leveraging established risk management frameworks, ACA provides a multi-layered approach to threat identification, risk assessment, and proactive mitigation strategies. A comprehensive threat library classifies physical, third-party, external, and internal threats. ACA's iterative nature and feedback mechanisms enable continuous adaptation to emerging threats, ensuring sustained effectiveness. ACA allows healthcare providers to proactively identify and mitigate vulnerabilities, fostering trust and supporting the secure adoption of virtual care technologies.

摘要: 远程医疗服务的快速增长给受保护的健康信息(PHI)带来了重大的安全和隐私风险。对涵盖2009-2023年的全面医疗安全漏洞数据集的分析显示，这些漏洞的流行程度和影响很大。这项研究调查了此类安全事件的根本原因，并引入了以攻击者为中心的方法(ACA)，这是一种为保护PHI而量身定做的新型威胁模型。ACA通过采用以攻击者为中心的整体视角，从网络对手、他们的动机、战术和潜在攻击载体的角度检查威胁，解决了现有威胁模型和监管框架中的局限性。利用已建立的风险管理框架，ACA为威胁识别、风险评估和主动缓解战略提供了多层次的方法。全面的威胁库对物理威胁、第三方威胁、外部威胁和内部威胁进行分类。ACA的迭代性质和反馈机制使其能够不断适应新出现的威胁，确保持续有效。ACA允许医疗保健提供者主动识别和缓解漏洞，从而增强信任并支持安全采用虚拟护理技术。



## **24. Safeguarding System Prompts for LLMs**

LLM的保护系统预算 cs.CR

20 pages, 7 figures, 6 tables

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13426v1) [paper-pdf](http://arxiv.org/pdf/2412.13426v1)

**Authors**: Zhifeng Jiang, Zhihua Jin, Guoliang He

**Abstract**: Large language models (LLMs) are increasingly utilized in applications where system prompts, which guide model outputs, play a crucial role. These prompts often contain business logic and sensitive information, making their protection essential. However, adversarial and even regular user queries can exploit LLM vulnerabilities to expose these hidden prompts. To address this issue, we present PromptKeeper, a novel defense mechanism for system prompt privacy. By reliably detecting worst-case leakage and regenerating outputs without the system prompt when necessary, PromptKeeper ensures robust protection against prompt extraction attacks via either adversarial or regular queries, while preserving conversational capability and runtime efficiency during benign user interactions.

摘要: 大型语言模型（LLM）越来越多地用于指导模型输出的系统提示发挥着至关重要作用的应用程序。这些提示通常包含业务逻辑和敏感信息，因此对其的保护至关重要。然而，对抗性甚至常规用户查询都可能利用LLM漏洞来暴露这些隐藏的提示。为了解决这个问题，我们提出了Inbox Keeper，这是一种针对系统提示隐私的新型防御机制。通过可靠地检测最坏情况的泄漏并在必要时无需系统提示即可重新生成输出，Inbox Keeper确保了针对通过对抗性或常规查询进行的即时提取攻击的强大保护，同时在良性用户交互期间保留对话能力和运行时效率。



## **25. Targeted View-Invariant Adversarial Perturbations for 3D Object Recognition**

3D对象识别的目标视图不变对抗扰动 cs.CV

Accepted to AAAI-25 Workshop on Artificial Intelligence for Cyber  Security (AICS): http://aics.site/AICS2025/index.html

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.13376v1) [paper-pdf](http://arxiv.org/pdf/2412.13376v1)

**Authors**: Christian Green, Mehmet Ergezer, Abdurrahman Zeybey

**Abstract**: Adversarial attacks pose significant challenges in 3D object recognition, especially in scenarios involving multi-view analysis where objects can be observed from varying angles. This paper introduces View-Invariant Adversarial Perturbations (VIAP), a novel method for crafting robust adversarial examples that remain effective across multiple viewpoints. Unlike traditional methods, VIAP enables targeted attacks capable of manipulating recognition systems to classify objects as specific, pre-determined labels, all while using a single universal perturbation. Leveraging a dataset of 1,210 images across 121 diverse rendered 3D objects, we demonstrate the effectiveness of VIAP in both targeted and untargeted settings. Our untargeted perturbations successfully generate a singular adversarial noise robust to 3D transformations, while targeted attacks achieve exceptional results, with top-1 accuracies exceeding 95% across various epsilon values. These findings highlight VIAPs potential for real-world applications, such as testing the robustness of 3D recognition systems. The proposed method sets a new benchmark for view-invariant adversarial robustness, advancing the field of adversarial machine learning for 3D object recognition.

摘要: 对抗性攻击对3D对象识别提出了巨大的挑战，特别是在涉及多视角分析的场景中，其中可以从不同的角度观察对象。介绍了视点不变的对抗性扰动(VIAP)，这是一种新的方法，用于制作健壮的对抗性实例，并且在多个视点上保持有效。与传统方法不同，VIAP使能够操纵识别系统的定向攻击能够将对象分类为特定的、预先确定的标签，同时使用单一的通用扰动。利用121个不同渲染的3D对象上的1210个图像的数据集，我们演示了VIAP在目标和非目标环境中的有效性。我们的非目标扰动成功地产生了对3D变换稳健的单一对抗性噪声，而目标攻击获得了特殊的结果，在不同的epsilon值上TOP-1的准确率超过95%。这些发现突出了VIAP在现实世界中应用的潜力，例如测试3D识别系统的健壮性。该方法为视点不变的对抗性稳健性建立了一个新的基准，推动了对抗性机器学习在三维物体识别领域的发展。



## **26. Class-RAG: Real-Time Content Moderation with Retrieval Augmented Generation**

Class-RAG：具有检索增强生成的实时内容审核 cs.AI

11 pages, submit to ACL

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2410.14881v2) [paper-pdf](http://arxiv.org/pdf/2410.14881v2)

**Authors**: Jianfa Chen, Emily Shen, Trupti Bavalatti, Xiaowen Lin, Yongkai Wang, Shuming Hu, Harihar Subramanyam, Ksheeraj Sai Vepuri, Ming Jiang, Ji Qi, Li Chen, Nan Jiang, Ankit Jain

**Abstract**: Robust content moderation classifiers are essential for the safety of Generative AI systems. In this task, differences between safe and unsafe inputs are often extremely subtle, making it difficult for classifiers (and indeed, even humans) to properly distinguish violating vs. benign samples without context or explanation. Scaling risk discovery and mitigation through continuous model fine-tuning is also slow, challenging and costly, preventing developers from being able to respond quickly and effectively to emergent harms. We propose a Classification approach employing Retrieval-Augmented Generation (Class-RAG). Class-RAG extends the capability of its base LLM through access to a retrieval library which can be dynamically updated to enable semantic hotfixing for immediate, flexible risk mitigation. Compared to model fine-tuning, Class-RAG demonstrates flexibility and transparency in decision-making, outperforms on classification and is more robust against adversarial attack, as evidenced by empirical studies. Our findings also suggest that Class-RAG performance scales with retrieval library size, indicating that increasing the library size is a viable and low-cost approach to improve content moderation.

摘要: 健壮的内容审核分类器对于生成式人工智能系统的安全至关重要。在这项任务中，安全和不安全输入之间的差异通常非常细微，使得分类器(甚至人类)很难在没有上下文或解释的情况下正确区分违规和良性样本。通过持续的模型微调扩展风险发现和缓解也是缓慢、具有挑战性和成本高昂的，使开发人员无法快速有效地响应紧急危害。我们提出了一种基于检索扩展生成的分类方法(Class-RAG)。Class-RAG通过访问检索库来扩展其基本LLM的能力，检索库可以动态更新，以实现语义热修复，从而立即、灵活地降低风险。实验结果表明，与模型微调相比，Class-RAG在决策上表现出灵活性和透明性，在分类上表现出更好的性能，并且对敌意攻击具有更强的鲁棒性。我们的发现还表明，Class-RAG的性能随检索库的大小而变化，这表明增加库的大小是改善内容审核的一种可行且低成本的方法。



## **27. Concept-ROT: Poisoning Concepts in Large Language Models with Model Editing**

Concept-ROT：通过模型编辑毒害大型语言模型中的概念 cs.LG

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.13341v1) [paper-pdf](http://arxiv.org/pdf/2412.13341v1)

**Authors**: Keltin Grimes, Marco Christiani, David Shriver, Marissa Connor

**Abstract**: Model editing methods modify specific behaviors of Large Language Models by altering a small, targeted set of network weights and require very little data and compute. These methods can be used for malicious applications such as inserting misinformation or simple trojans that result in adversary-specified behaviors when a trigger word is present. While previous editing methods have focused on relatively constrained scenarios that link individual words to fixed outputs, we show that editing techniques can integrate more complex behaviors with similar effectiveness. We develop Concept-ROT, a model editing-based method that efficiently inserts trojans which not only exhibit complex output behaviors, but also trigger on high-level concepts -- presenting an entirely new class of trojan attacks. Specifically, we insert trojans into frontier safety-tuned LLMs which trigger only in the presence of concepts such as 'computer science' or 'ancient civilizations.' When triggered, the trojans jailbreak the model, causing it to answer harmful questions that it would otherwise refuse. Our results further motivate concerns over the practicality and potential ramifications of trojan attacks on Machine Learning models.

摘要: 模型编辑方法通过改变一组小的、有针对性的网络权重来修改大型语言模型的特定行为，并且需要非常少的数据和计算。这些方法可用于恶意应用程序，如插入错误信息或简单的特洛伊木马程序，当存在触发词时，这些木马程序会导致对手指定的行为。虽然以前的编辑方法专注于将单个单词链接到固定输出的相对受限的场景，但我们证明了编辑技术可以集成更复杂的行为，具有类似的有效性。我们开发了Concept-ROT，这是一种基于模型编辑的方法，它有效地插入特洛伊木马，这些特洛伊木马不仅表现出复杂的输出行为，而且还会触发高级概念--呈现出一种全新的特洛伊木马攻击类别。具体地说，我们将特洛伊木马程序插入到前沿安全调整的LLM中，这些LLM只有在存在诸如“计算机科学”或“古代文明”的概念时才会触发。一旦触发，特洛伊木马程序就会越狱，让它回答原本会拒绝的有害问题。我们的结果进一步引发了人们对木马攻击对机器学习模型的实用性和潜在后果的担忧。



## **28. LLM Whisperer: An Inconspicuous Attack to Bias LLM Responses**

LLM Whisperer：对LLM偏见回应的不起眼攻击 cs.CR

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2406.04755v3) [paper-pdf](http://arxiv.org/pdf/2406.04755v3)

**Authors**: Weiran Lin, Anna Gerchanovsky, Omer Akgul, Lujo Bauer, Matt Fredrikson, Zifan Wang

**Abstract**: Writing effective prompts for large language models (LLM) can be unintuitive and burdensome. In response, services that optimize or suggest prompts have emerged. While such services can reduce user effort, they also introduce a risk: the prompt provider can subtly manipulate prompts to produce heavily biased LLM responses. In this work, we show that subtle synonym replacements in prompts can increase the likelihood (by a difference up to 78%) that LLMs mention a target concept (e.g., a brand, political party, nation). We substantiate our observations through a user study, showing our adversarially perturbed prompts 1) are indistinguishable from unaltered prompts by humans, 2) push LLMs to recommend target concepts more often, and 3) make users more likely to notice target concepts, all without arousing suspicion. The practicality of this attack has the potential to undermine user autonomy. Among other measures, we recommend implementing warnings against using prompts from untrusted parties.

摘要: 为大型语言模型(LLM)编写有效的提示可能是不直观和繁琐的。作为回应，优化或建议提示的服务应运而生。虽然这类服务可以减少用户的工作，但它们也带来了风险：提示提供商可能会巧妙地操纵提示，以产生严重偏见的LLM响应。在这项工作中，我们表明，提示中微妙的同义词替换可以增加LLMS提到目标概念(例如，品牌、政党、国家)的可能性(差异高达78%)。我们通过一项用户研究证实了我们的观察结果，表明我们受到敌意干扰的提示1)与人类未更改的提示无法区分，2)推动LLMS更频繁地推荐目标概念，3)使用户更有可能注意到目标概念，所有这些都不会引起怀疑。这种攻击的实用性有可能破坏用户的自主性。在其他措施中，我们建议实施警告，以防止使用来自不受信任方的提示。



## **29. A New Adversarial Perspective for LiDAR-based 3D Object Detection**

基于LiDART的3D对象检测的新对抗视角 cs.CV

11 pages, 7 figures, AAAI2025

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.13017v1) [paper-pdf](http://arxiv.org/pdf/2412.13017v1)

**Authors**: Shijun Zheng, Weiquan Liu, Yu Guo, Yu Zang, Siqi Shen, Cheng Wang

**Abstract**: Autonomous vehicles (AVs) rely on LiDAR sensors for environmental perception and decision-making in driving scenarios. However, ensuring the safety and reliability of AVs in complex environments remains a pressing challenge. To address this issue, we introduce a real-world dataset (ROLiD) comprising LiDAR-scanned point clouds of two random objects: water mist and smoke. In this paper, we introduce a novel adversarial perspective by proposing an attack framework that utilizes water mist and smoke to simulate environmental interference. Specifically, we propose a point cloud sequence generation method using a motion and content decomposition generative adversarial network named PCS-GAN to simulate the distribution of random objects. Furthermore, leveraging the simulated LiDAR scanning characteristics implemented with Range Image, we examine the effects of introducing random object perturbations at various positions on the target vehicle. Extensive experiments demonstrate that adversarial perturbations based on random objects effectively deceive vehicle detection and reduce the recognition rate of 3D object detection models.

摘要: 自动驾驶汽车(AVs)依靠激光雷达传感器在驾驶场景中进行环境感知和决策。然而，确保无人机在复杂环境中的安全性和可靠性仍然是一个紧迫的挑战。为了解决这个问题，我们引入了一个真实世界的数据集(ROLiD)，它包含两个随机对象的激光雷达扫描的点云：细水雾和烟雾。在本文中，我们引入了一种新的对抗视角，提出了一种利用细水雾和烟雾来模拟环境干扰的攻击框架。具体地说，我们提出了一种利用运动和内容分解生成对抗网络PCS-GAN来模拟随机对象分布的点云序列生成方法。此外，利用利用Range Image实现的模拟LiDAR扫描特性，我们检查了在目标车辆的不同位置引入随机目标扰动的效果。大量实验表明，基于随机目标的对抗性扰动有效地欺骗了车辆检测，降低了三维目标检测模型的识别率。



## **30. AnyAttack: Targeted Adversarial Attacks on Vision-Language Models toward Any Images**

AnyAttack：针对任何图像的视觉语言模型的有针对性的对抗攻击 cs.LG

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2410.05346v2) [paper-pdf](http://arxiv.org/pdf/2410.05346v2)

**Authors**: Jiaming Zhang, Junhong Ye, Xingjun Ma, Yige Li, Yunfan Yang, Jitao Sang, Dit-Yan Yeung

**Abstract**: Due to their multimodal capabilities, Vision-Language Models (VLMs) have found numerous impactful applications in real-world scenarios. However, recent studies have revealed that VLMs are vulnerable to image-based adversarial attacks, particularly targeted adversarial images that manipulate the model to generate harmful content specified by the adversary. Current attack methods rely on predefined target labels to create targeted adversarial attacks, which limits their scalability and applicability for large-scale robustness evaluations. In this paper, we propose AnyAttack, a self-supervised framework that generates targeted adversarial images for VLMs without label supervision, allowing any image to serve as a target for the attack. Our framework employs the pre-training and fine-tuning paradigm, with the adversarial noise generator pre-trained on the large-scale LAION-400M dataset. This large-scale pre-training endows our method with powerful transferability across a wide range of VLMs. Extensive experiments on five mainstream open-source VLMs (CLIP, BLIP, BLIP2, InstructBLIP, and MiniGPT-4) across three multimodal tasks (image-text retrieval, multimodal classification, and image captioning) demonstrate the effectiveness of our attack. Additionally, we successfully transfer AnyAttack to multiple commercial VLMs, including Google Gemini, Claude Sonnet, Microsoft Copilot and OpenAI GPT. These results reveal an unprecedented risk to VLMs, highlighting the need for effective countermeasures.

摘要: 由于其多通道能力，视觉语言模型(VLM)在现实世界场景中发现了许多有影响力的应用。然而，最近的研究表明，VLM很容易受到基于图像的敌意攻击，特别是针对操纵模型以生成对手指定的有害内容的对抗性图像。当前的攻击方法依赖于预定义的目标标签来创建有针对性的对抗性攻击，这限制了它们在大规模健壮性评估中的可扩展性和适用性。在本文中，我们提出了AnyAttack，这是一个自监督框架，可以在没有标签监督的情况下为VLMS生成有针对性的敌意图像，允许任何图像作为攻击的目标。我们的框架采用了预训练和微调的范式，对抗噪声发生器在大规模LAION-400M数据集上进行了预训练。这种大规模的预培训使我们的方法在广泛的VLM中具有强大的可移植性。在三个多模式任务(图像-文本检索、多模式分类和图像字幕)上对五个主流开源VLMS(CLIP、BLIP、BLIP2、InstructBLIP和MiniGPT-4)进行了广泛的实验，证明了该攻击的有效性。此外，我们还成功地将AnyAttack移植到多个商业VLM上，包括Google Gemini、Claude Sonnet、Microsoft Copilot和OpenAI GPT。这些结果揭示了极小武器系统面临的前所未有的风险，突显了采取有效对策的必要性。



## **31. Adaptive Epsilon Adversarial Training for Robust Gravitational Wave Parameter Estimation Using Normalizing Flows**

使用正规化流进行鲁棒引力波参数估计的自适应Episodes对抗训练 cs.LG

Due to new experimental results to add to the paper, this version no  longer accurately reflects the current state of our research. Therefore, we  are withdrawing the paper while further experiments are conducted. We will  submit a new version in the future. We apologize for any inconvenience this  may cause

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.07559v2) [paper-pdf](http://arxiv.org/pdf/2412.07559v2)

**Authors**: Yiqian Yang, Xihua Zhu, Fan Zhang

**Abstract**: Adversarial training with Normalizing Flow (NF) models is an emerging research area aimed at improving model robustness through adversarial samples. In this study, we focus on applying adversarial training to NF models for gravitational wave parameter estimation. We propose an adaptive epsilon method for Fast Gradient Sign Method (FGSM) adversarial training, which dynamically adjusts perturbation strengths based on gradient magnitudes using logarithmic scaling. Our hybrid architecture, combining ResNet and Inverse Autoregressive Flow, reduces the Negative Log Likelihood (NLL) loss by 47\% under FGSM attacks compared to the baseline model, while maintaining an NLL of 4.2 on clean data (only 5\% higher than the baseline). For perturbation strengths between 0.01 and 0.1, our model achieves an average NLL of 5.8, outperforming both fixed-epsilon (NLL: 6.7) and progressive-epsilon (NLL: 7.2) methods. Under stronger Projected Gradient Descent attacks with perturbation strength of 0.05, our model maintains an NLL of 6.4, demonstrating superior robustness while avoiding catastrophic overfitting.

摘要: 利用归一化流量模型进行对抗性训练是一个新兴的研究领域，其目的是通过对抗性样本提高模型的稳健性。在这项研究中，我们将对抗性训练应用到引力波参数估计的神经网络模型中。提出了一种用于快速梯度符号法(FGSM)对抗训练的自适应epsilon方法，该方法利用对数尺度根据梯度大小动态调整扰动强度。我们的混合架构结合了ResNet和反向自回归流，与基线模型相比，在FGSM攻击下，负对数似然(NLL)损失降低了47%，而对于干净的数据，NLL保持在4.2(仅比基线高5%)。当微扰强度在0.01到0.1之间时，我们的模型的平均NLL为5.8，优于固定epsilon(NLL：6.7)和渐进epsilon(NLL：7.2)方法。在扰动强度为0.05的较强投影梯度下降攻击下，我们的模型保持了6.4的NLL，在避免灾难性过拟合的同时表现出了优越的稳健性。



## **32. PROSAC: Provably Safe Certification for Machine Learning Models under Adversarial Attacks**

PROSAC：对抗性攻击下的机器学习模型可证明安全的认证 cs.LG

Accepted to AAAI2025

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2402.02629v2) [paper-pdf](http://arxiv.org/pdf/2402.02629v2)

**Authors**: Chen Feng, Ziquan Liu, Zhuo Zhi, Ilija Bogunovic, Carsten Gerner-Beuerle, Miguel Rodrigues

**Abstract**: It is widely known that state-of-the-art machine learning models, including vision and language models, can be seriously compromised by adversarial perturbations. It is therefore increasingly relevant to develop capabilities to certify their performance in the presence of the most effective adversarial attacks. Our paper offers a new approach to certify the performance of machine learning models in the presence of adversarial attacks with population level risk guarantees. In particular, we introduce the notion of $(\alpha,\zeta)$-safe machine learning model. We propose a hypothesis testing procedure, based on the availability of a calibration set, to derive statistical guarantees providing that the probability of declaring that the adversarial (population) risk of a machine learning model is less than $\alpha$ (i.e. the model is safe), while the model is in fact unsafe (i.e. the model adversarial population risk is higher than $\alpha$), is less than $\zeta$. We also propose Bayesian optimization algorithms to determine efficiently whether a machine learning model is $(\alpha,\zeta)$-safe in the presence of an adversarial attack, along with statistical guarantees. We apply our framework to a range of machine learning models - including various sizes of vision Transformer (ViT) and ResNet models - impaired by a variety of adversarial attacks, such as PGDAttack, MomentumAttack, GenAttack and BanditAttack, to illustrate the operation of our approach. Importantly, we show that ViT's are generally more robust to adversarial attacks than ResNets, and large models are generally more robust than smaller models. Our approach goes beyond existing empirical adversarial risk-based certification guarantees. It formulates rigorous (and provable) performance guarantees that can be used to satisfy regulatory requirements mandating the use of state-of-the-art technical tools.

摘要: 众所周知，最先进的机器学习模型，包括视觉和语言模型，可能会受到对抗性扰动的严重影响。因此，越来越有必要发展能力，以证明它们在最有效的对抗性攻击下的表现。本文提供了一种新的方法来证明机器学习模型在种群水平风险保证的对抗性攻击下的性能。特别地，我们引入了$(\α，\Zeta)$-安全机器学习模型的概念。我们提出了一种假设检验程序，基于校准集的可用性来获得统计保证，假设宣布一个机器学习模型的对抗(总体)风险小于$\α$(即该模型是安全的)，而该模型实际上是不安全的(即该模型的对抗总体风险高于$\α$)的概率小于$\Zeta$。我们还提出了贝叶斯优化算法来有效地确定机器学习模型在存在对抗性攻击的情况下是否$(\α，\Zeta)$安全，并提供统计保证。我们将我们的框架应用于一系列机器学习模型，包括各种大小的视觉转换器(VIT)和ResNet模型，这些模型被各种敌意攻击所破坏，如PGDAttack、MomentumAttack、GenAttack和BanditAttack，以说明我们方法的操作。重要的是，我们发现VIT通常比ResNet对对手攻击更健壮，大模型通常比小模型更健壮。我们的方法超越了现有的经验对抗性、基于风险的认证保证。它制定了严格的(和可证明的)性能保证，可用于满足要求使用最先进技术工具的监管要求。



## **33. Deep Learning for Resilient Adversarial Decision Fusion in Byzantine Networks**

深度学习用于拜占庭网络中弹性对抗决策融合 cs.LG

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12739v1) [paper-pdf](http://arxiv.org/pdf/2412.12739v1)

**Authors**: Kassem Kallas

**Abstract**: This paper introduces a deep learning-based framework for resilient decision fusion in adversarial multi-sensor networks, providing a unified mathematical setup that encompasses diverse scenarios, including varying Byzantine node proportions, synchronized and unsynchronized attacks, unbalanced priors, adaptive strategies, and Markovian states. Unlike traditional methods, which depend on explicit parameter tuning and are limited by scenario-specific assumptions, the proposed approach employs a deep neural network trained on a globally constructed dataset to generalize across all cases without requiring adaptation. Extensive simulations validate the method's robustness, achieving superior accuracy, minimal error probability, and scalability compared to state-of-the-art techniques, while ensuring computational efficiency for real-time applications. This unified framework demonstrates the potential of deep learning to revolutionize decision fusion by addressing the challenges posed by Byzantine nodes in dynamic adversarial environments.

摘要: 提出了一种基于深度学习的对抗性多传感器网络弹性决策融合框架，提供了一个统一的数学模型，涵盖了拜占庭节点比例变化、同步和非同步攻击、不平衡先验、自适应策略和马尔可夫状态等多种场景。与依赖于显式参数调整的传统方法不同，该方法使用在全局构造的数据集上训练的深度神经网络来对所有情况进行泛化，而不需要自适应。大量的仿真验证了该方法的稳健性，与最先进的技术相比，实现了更高的精确度、最小的错误概率和可扩展性，同时确保了实时应用的计算效率。这一统一的框架展示了深度学习通过解决动态对抗环境中拜占庭节点带来的挑战来彻底改变决策融合的潜力。



## **34. On the Impact of Hard Adversarial Instances on Overfitting in Adversarial Training**

论硬对抗预设对对抗训练中过度配合的影响 cs.LG

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2112.07324v2) [paper-pdf](http://arxiv.org/pdf/2112.07324v2)

**Authors**: Chen Liu, Zhichao Huang, Mathieu Salzmann, Tong Zhang, Sabine Süsstrunk

**Abstract**: Adversarial training is a popular method to robustify models against adversarial attacks. However, it exhibits much more severe overfitting than training on clean inputs. In this work, we investigate this phenomenon from the perspective of training instances, i.e., training input-target pairs. Based on a quantitative metric measuring the relative difficulty of an instance in the training set, we analyze the model's behavior on training instances of different difficulty levels. This lets us demonstrate that the decay in generalization performance of adversarial training is a result of fitting hard adversarial instances. We theoretically verify our observations for both linear and general nonlinear models, proving that models trained on hard instances have worse generalization performance than ones trained on easy instances, and that this generalization gap increases with the size of the adversarial budget. Finally, we investigate solutions to mitigate adversarial overfitting in several scenarios, including fast adversarial training and fine-tuning a pretrained model with additional data. Our results demonstrate that using training data adaptively improves the model's robustness.

摘要: 对抗性训练是一种流行的方法，用来增强模型抵御对抗性攻击的能力。然而，它表现出比清洁投入培训严重得多的过度适应。在这项工作中，我们从训练实例的角度来研究这一现象，即训练输入-目标对。基于一个量化度量实例在训练集中的相对难度，我们分析了该模型在不同难度级别的训练实例上的行为。这使我们能够证明，对抗性训练的泛化性能的下降是拟合困难的对抗性实例的结果。我们从理论上验证了我们对线性和一般非线性模型的观察结果，证明了在硬实例上训练的模型比在简单实例上训练的模型具有更差的泛化性能，并且这种泛化差距随着对抗预算的大小而增大。最后，我们研究了在几种情况下缓解对抗过度匹配的解决方案，包括快速的对抗训练和用额外的数据微调预先训练的模型。结果表明，训练数据的自适应使用提高了模型的稳健性。



## **35. Building Gradient Bridges: Label Leakage from Restricted Gradient Sharing in Federated Learning**

构建梯度桥：联邦学习中受限制的梯度共享造成的标签泄漏 cs.LG

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12640v1) [paper-pdf](http://arxiv.org/pdf/2412.12640v1)

**Authors**: Rui Zhang, Ka-Ho Chow, Ping Li

**Abstract**: The growing concern over data privacy, the benefits of utilizing data from diverse sources for model training, and the proliferation of networked devices with enhanced computational capabilities have all contributed to the rise of federated learning (FL). The clients in FL collaborate to train a global model by uploading gradients computed on their private datasets without collecting raw data. However, a new attack surface has emerged from gradient sharing, where adversaries can restore the label distribution of a victim's private data by analyzing the obtained gradients. To mitigate this privacy leakage, existing lightweight defenses restrict the sharing of gradients, such as encrypting the final-layer gradients or locally updating the parameters within. In this paper, we introduce a novel attack called Gradient Bridge (GDBR) that recovers the label distribution of training data from the limited gradient information shared in FL. GDBR explores the relationship between the layer-wise gradients, tracks the flow of gradients, and analytically derives the batch training labels. Extensive experiments show that GDBR can accurately recover more than 80% of labels in various FL settings. GDBR highlights the inadequacy of restricted gradient sharing-based defenses and calls for the design of effective defense schemes in FL.

摘要: 对数据隐私的日益关注，利用来自不同来源的数据进行模型训练的好处，以及具有增强计算能力的联网设备的激增，所有这些都促进了联合学习(FL)的兴起。FL中的客户通过上传在其私有数据集上计算的梯度来协作训练全局模型，而不收集原始数据。然而，从梯度共享中出现了一个新的攻击面，攻击者可以通过分析获得的梯度来恢复受害者私人数据的标签分布。为了缓解这种隐私泄露，现有的轻量级防御限制了渐变的共享，例如加密最终层渐变或本地更新其中的参数。本文提出了一种新的攻击方法--梯度桥(GDBR)，它从FL共享的有限的梯度信息中恢复训练数据的标签分布。GDBR探索逐层梯度之间的关系，跟踪梯度的流动，并解析地得出批量训练标签。大量实验表明，在各种FL环境下，GDBR可以准确地恢复80%以上的标签。GDBR强调了基于受限梯度共享的防御的不足，并呼吁在FL中设计有效的防御方案。



## **36. Improving the Transferability of 3D Point Cloud Attack via Spectral-aware Admix and Optimization Designs**

通过光谱感知的混合和优化设计提高3D点云攻击的可转移性 cs.CV

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12626v1) [paper-pdf](http://arxiv.org/pdf/2412.12626v1)

**Authors**: Shiyu Hu, Daizong Liu, Wei Hu

**Abstract**: Deep learning models for point clouds have shown to be vulnerable to adversarial attacks, which have received increasing attention in various safety-critical applications such as autonomous driving, robotics, and surveillance. Existing 3D attackers generally design various attack strategies in the white-box setting, requiring the prior knowledge of 3D model details. However, real-world 3D applications are in the black-box setting, where we can only acquire the outputs of the target classifier. Although few recent works try to explore the black-box attack, they still achieve limited attack success rates (ASR). To alleviate this issue, this paper focuses on attacking the 3D models in a transfer-based black-box setting, where we first carefully design adversarial examples in a white-box surrogate model and then transfer them to attack other black-box victim models. Specifically, we propose a novel Spectral-aware Admix with Augmented Optimization method (SAAO) to improve the adversarial transferability. In particular, since traditional Admix strategy are deployed in the 2D domain that adds pixel-wise images for perturbing, we can not directly follow it to merge point clouds in coordinate domain as it will destroy the geometric shapes. Therefore, we design spectral-aware fusion that performs Graph Fourier Transform (GFT) to get spectral features of the point clouds and add them in the spectral domain. Afterward, we run a few steps with spectral-aware weighted Admix to select better optimization paths as well as to adjust corresponding learning weights. At last, we run more steps to generate adversarial spectral feature along the optimization path and perform Inverse-GFT on the adversarial spectral feature to obtain the adversarial example in the data domain. Experiments show that our SAAO achieves better transferability compared to existing 3D attack methods.

摘要: 点云的深度学习模型容易受到敌意攻击，在自动驾驶、机器人和监控等各种安全关键应用中受到越来越多的关注。现有的3D攻击者一般在白盒环境下设计各种攻击策略，需要对3D模型细节的先验知识。然而，现实世界中的3D应用程序处于黑盒设置中，我们只能获取目标分类器的输出。虽然最近很少有文献对黑盒攻击进行研究，但它们的攻击成功率(ASR)仍然有限。为了缓解这一问题，本文重点攻击基于转移的黑箱环境中的3D模型，首先在白箱代理模型中精心设计对抗性实例，然后将它们转移到攻击其他黑箱受害者模型。具体地说，我们提出了一种新的频谱感知广告混合增强优化方法(SAAO)来提高对抗可转移性。特别是，由于传统的AdMix策略是在二维域中添加像素级的图像进行扰动，因此不能直接跟随它在坐标域中合并点云，因为这会破坏几何形状。因此，我们设计了光谱感知融合，通过图形傅里叶变换(GFT)来获取点云的光谱特征，并将其添加到谱域中。然后，我们使用谱感知加权AdMix运行几个步骤来选择更好的优化路径以及调整相应的学习权重。最后，我们沿着优化路径运行更多的步骤来生成对抗性谱特征，并对对抗性谱特征进行逆GFT，得到数据域中的对抗性实例。实验表明，与现有的3D攻击方法相比，我们的SAAO实现了更好的可移植性。



## **37. Jailbreaking? One Step Is Enough!**

越狱？一步就够了！ cs.CL

17 pages

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12621v1) [paper-pdf](http://arxiv.org/pdf/2412.12621v1)

**Authors**: Weixiong Zheng, Peijian Zeng, Yiwei Li, Hongyan Wu, Nankai Lin, Junhao Chen, Aimin Yang, Yongmei Zhou

**Abstract**: Large language models (LLMs) excel in various tasks but remain vulnerable to jailbreak attacks, where adversaries manipulate prompts to generate harmful outputs. Examining jailbreak prompts helps uncover the shortcomings of LLMs. However, current jailbreak methods and the target model's defenses are engaged in an independent and adversarial process, resulting in the need for frequent attack iterations and redesigning attacks for different models. To address these gaps, we propose a Reverse Embedded Defense Attack (REDA) mechanism that disguises the attack intention as the "defense". intention against harmful content. Specifically, REDA starts from the target response, guiding the model to embed harmful content within its defensive measures, thereby relegating harmful content to a secondary role and making the model believe it is performing a defensive task. The attacking model considers that it is guiding the target model to deal with harmful content, while the target model thinks it is performing a defensive task, creating an illusion of cooperation between the two. Additionally, to enhance the model's confidence and guidance in "defensive" intentions, we adopt in-context learning (ICL) with a small number of attack examples and construct a corresponding dataset of attack examples. Extensive evaluations demonstrate that the REDA method enables cross-model attacks without the need to redesign attack strategies for different models, enables successful jailbreak in one iteration, and outperforms existing methods on both open-source and closed-source models.

摘要: 大型语言模型(LLM)在各种任务中表现出色，但仍然容易受到越狱攻击的攻击，在越狱攻击中，对手操纵提示以生成有害的输出。检查越狱提示有助于发现LLMS的缺点。然而，当前的越狱方法和目标模型的防御都是独立的和对抗性的过程，导致需要频繁的攻击迭代和针对不同模型的重新设计攻击。针对这些漏洞，我们提出了一种反向嵌入防御攻击(REDA)机制，将攻击意图伪装成“防御”。针对有害内容的意图。具体地说，Reda从目标响应开始，引导模型在其防御措施中嵌入有害内容，从而将有害内容降级为次要角色，并使模型相信它正在执行防御任务。攻击模型认为它是在引导目标模型处理有害内容，而目标模型则认为它是在执行防御任务，制造了两者合作的错觉。此外，为了增强模型对“防御”意图的可信度和指导性，我们采用了上下文中学习(ICL)的方法，并结合少量攻击实例构建了相应的攻击实例数据集。广泛的评估表明，REDA方法支持跨模型攻击，不需要针对不同的模型重新设计攻击策略，一次迭代即可成功越狱，并且在开源和闭源模型上的性能都优于现有方法。



## **38. WaterPark: A Robustness Assessment of Language Model Watermarking**

WaterPark：语言模型水印的稳健性评估 cs.CR

22 pages

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2411.13425v2) [paper-pdf](http://arxiv.org/pdf/2411.13425v2)

**Authors**: Jiacheng Liang, Zian Wang, Lauren Hong, Shouling Ji, Ting Wang

**Abstract**: Various watermarking methods (``watermarkers'') have been proposed to identify LLM-generated texts; yet, due to the lack of unified evaluation platforms, many critical questions remain under-explored: i) What are the strengths/limitations of various watermarkers, especially their attack robustness? ii) How do various design choices impact their robustness? iii) How to optimally operate watermarkers in adversarial environments? To fill this gap, we systematize existing LLM watermarkers and watermark removal attacks, mapping out their design spaces. We then develop WaterPark, a unified platform that integrates 10 state-of-the-art watermarkers and 12 representative attacks. More importantly, by leveraging WaterPark, we conduct a comprehensive assessment of existing watermarkers, unveiling the impact of various design choices on their attack robustness. We further explore the best practices to operate watermarkers in adversarial environments. We believe our study sheds light on current LLM watermarking techniques while WaterPark serves as a valuable testbed to facilitate future research.

摘要: 已经提出了各种水印方法来识别LLM生成的文本；然而，由于缺乏统一的评估平台，许多关键问题仍然没有得到充分的研究：i)各种水印的优点/局限性是什么，特别是它们的攻击稳健性？Ii)各种设计选择对其健壮性有何影响？三)如何在对抗性环境中以最佳方式使用水印？为了填补这一空白，我们对现有的LLM水印和水印移除攻击进行了系统化，规划了它们的设计空间。然后我们开发了Water Park，这是一个统一的平台，集成了10个最先进的水印和12个具有代表性的攻击。更重要的是，通过利用水上公园，我们对现有的水印进行了全面的评估，揭示了各种设计选择对其攻击健壮性的影响。我们进一步探索在对抗性环境中操作水印的最佳实践。我们相信我们的研究对当前的LLM数字水印技术有一定的启发作用，同时也为以后的研究提供了一个有价值的实验平台。



## **39. Attack On Prompt: Backdoor Attack in Prompt-Based Continual Learning**

攻击提示：基于预算的持续学习中的后门攻击 cs.LG

Accepted to AAAI 2025

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2406.19753v2) [paper-pdf](http://arxiv.org/pdf/2406.19753v2)

**Authors**: Trang Nguyen, Anh Tran, Nhat Ho

**Abstract**: Prompt-based approaches offer a cutting-edge solution to data privacy issues in continual learning, particularly in scenarios involving multiple data suppliers where long-term storage of private user data is prohibited. Despite delivering state-of-the-art performance, its impressive remembering capability can become a double-edged sword, raising security concerns as it might inadvertently retain poisoned knowledge injected during learning from private user data. Following this insight, in this paper, we expose continual learning to a potential threat: backdoor attack, which drives the model to follow a desired adversarial target whenever a specific trigger is present while still performing normally on clean samples. We highlight three critical challenges in executing backdoor attacks on incremental learners and propose corresponding solutions: (1) \emph{Transferability}: We employ a surrogate dataset and manipulate prompt selection to transfer backdoor knowledge to data from other suppliers; (2) \emph{Resiliency}: We simulate static and dynamic states of the victim to ensure the backdoor trigger remains robust during intense incremental learning processes; and (3) \emph{Authenticity}: We apply binary cross-entropy loss as an anti-cheating factor to prevent the backdoor trigger from devolving into adversarial noise. Extensive experiments across various benchmark datasets and continual learners validate our continual backdoor framework, achieving up to $100\%$ attack success rate, with further ablation studies confirming our contributions' effectiveness.

摘要: 基于提示的方法为持续学习中的数据隐私问题提供了一种尖端解决方案，特别是在涉及多个数据供应商的场景中，禁止长期存储私人用户数据。尽管提供了最先进的性能，但其令人印象深刻的记忆能力可能会成为一把双刃剑，这引发了安全问题，因为它可能会无意中保留在从私人用户数据学习过程中注入的有毒知识。根据这一见解，在本文中，我们将持续学习暴露于一个潜在的威胁：后门攻击，它驱动模型在出现特定触发时跟踪期望的对手目标，同时仍然在干净的样本上正常运行。我们强调了对增量学习者执行后门攻击的三个关键挑战并提出了相应的解决方案：(1)\emph{可传递性}：我们使用代理数据集并操纵提示选择来将后门知识传输到其他供应商的数据；(2)\emph{弹性}：我们模拟受害者的静态和动态，以确保后门触发在激烈的增量学习过程中保持健壮；以及(3)\emph{真实性}：我们应用二进制交叉熵损失作为反作弊因子，以防止后门触发演变为对抗性噪声。在各种基准数据集和不断学习的人中进行的广泛实验验证了我们的持续后门框架，实现了高达100美元的攻击成功率，进一步的消融研究证实了我们的贡献的有效性。



## **40. Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?**

大型语言模型能否提高图神经网络的对抗鲁棒性？ cs.LG

accepted by KDD2025

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2408.08685v2) [paper-pdf](http://arxiv.org/pdf/2408.08685v2)

**Authors**: Zhongjian Zhang, Xiao Wang, Huichi Zhou, Yue Yu, Mengmei Zhang, Cheng Yang, Chuan Shi

**Abstract**: Graph neural networks (GNNs) are vulnerable to adversarial attacks, especially for topology perturbations, and many methods that improve the robustness of GNNs have received considerable attention. Recently, we have witnessed the significant success of large language models (LLMs), leading many to explore the great potential of LLMs on GNNs. However, they mainly focus on improving the performance of GNNs by utilizing LLMs to enhance the node features. Therefore, we ask: Will the robustness of GNNs also be enhanced with the powerful understanding and inference capabilities of LLMs? By presenting the empirical results, we find that despite that LLMs can improve the robustness of GNNs, there is still an average decrease of 23.1% in accuracy, implying that the GNNs remain extremely vulnerable against topology attacks. Therefore, another question is how to extend the capabilities of LLMs on graph adversarial robustness. In this paper, we propose an LLM-based robust graph structure inference framework, LLM4RGNN, which distills the inference capabilities of GPT-4 into a local LLM for identifying malicious edges and an LM-based edge predictor for finding missing important edges, so as to recover a robust graph structure. Extensive experiments demonstrate that LLM4RGNN consistently improves the robustness across various GNNs. Even in some cases where the perturbation ratio increases to 40%, the accuracy of GNNs is still better than that on the clean graph. The source code can be found in https://github.com/zhongjian-zhang/LLM4RGNN.

摘要: 图神经网络(GNN)容易受到敌意攻击，尤其是对拓扑扰动的攻击，许多提高GNN健壮性的方法受到了广泛的关注。最近，我们目睹了大型语言模型(LLM)的巨大成功，这导致许多人探索LLM在GNN上的巨大潜力。然而，它们主要集中在通过利用LLMS来增强节点特征来提高GNN的性能。因此，我们问：GNN的健壮性是否也会随着LLMS强大的理解和推理能力而得到增强？通过给出实验结果，我们发现，尽管LLMS可以提高GNN的健壮性，但其准确率仍然平均下降23.1%，这意味着GNN仍然非常容易受到拓扑攻击。因此，另一个问题是如何扩展LLMS在图对抗健壮性方面的能力。本文提出了一种基于LLM的稳健图结构推理框架LLM4RGNN，该框架将GPT-4的推理能力抽象为用于识别恶意边的局部LLM和用于发现丢失重要边的基于LLM的边预测器，以恢复稳健的图结构。大量的实验表明，LLM4RGNN在不同的GNN上一致地提高了健壮性。即使在某些扰动比增加到40%的情况下，GNN的精度仍然好于干净图形上的精度。源代码可以在https://github.com/zhongjian-zhang/LLM4RGNN.中找到



## **41. Human-in-the-Loop Generation of Adversarial Texts: A Case Study on Tibetan Script**

对抗性文本的人在循环生成：以藏传文字为例 cs.CL

Review Version; Submitted to NAACL 2025 Demo Track

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12478v1) [paper-pdf](http://arxiv.org/pdf/2412.12478v1)

**Authors**: Xi Cao, Yuan Sun, Jiajun Li, Quzong Gesang, Nuo Qun, Tashi Nyima

**Abstract**: DNN-based language models perform excellently on various tasks, but even SOTA LLMs are susceptible to textual adversarial attacks. Adversarial texts play crucial roles in multiple subfields of NLP. However, current research has the following issues. (1) Most textual adversarial attack methods target rich-resourced languages. How do we generate adversarial texts for less-studied languages? (2) Most textual adversarial attack methods are prone to generating invalid or ambiguous adversarial texts. How do we construct high-quality adversarial robustness benchmarks? (3) New language models may be immune to part of previously generated adversarial texts. How do we update adversarial robustness benchmarks? To address the above issues, we introduce HITL-GAT, a system based on a general approach to human-in-the-loop generation of adversarial texts. HITL-GAT contains four stages in one pipeline: victim model construction, adversarial example generation, high-quality benchmark construction, and adversarial robustness evaluation. Additionally, we utilize HITL-GAT to make a case study on Tibetan script which can be a reference for the adversarial research of other less-studied languages.

摘要: 基于DNN的语言模型在各种任务中表现出色，但即使是Sota LLM也容易受到文本攻击。对抗性语篇在自然语言处理的多个子领域发挥着至关重要的作用。然而，目前的研究存在以下问题。(1)大多数文本对抗性攻击方法针对的是资源丰富的语言。如何为较少研究的语言生成对抗性文本？(2)大多数文本对抗性攻击方法容易产生无效或歧义的对抗性文本。我们如何构建高质量的对抗性健壮性基准？(3)新的语言模型可能对先前生成的部分对抗性文本免疫。我们如何更新对手健壮性基准？为了解决上述问题，我们引入了HITL-GAT，这是一个基于人在环中生成对抗性文本的通用方法的系统。HITL-GAT在一条流水线上包括四个阶段：受害者模型构建、对手实例生成、高质量基准构建和对手健壮性评估。此外，我们还利用HITL-GAT对藏文进行了实例研究，对其他研究较少的语言的对抗性研究具有一定的借鉴意义。



## **42. Architectural Patterns for Designing Quantum Artificial Intelligence Systems**

设计量子人工智能系统的架构模式 cs.SE

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2411.10487v3) [paper-pdf](http://arxiv.org/pdf/2411.10487v3)

**Authors**: Mykhailo Klymenko, Thong Hoang, Xiwei Xu, Zhenchang Xing, Muhammad Usman, Qinghua Lu, Liming Zhu

**Abstract**: Utilising quantum computing technology to enhance artificial intelligence systems is expected to improve training and inference times, increase robustness against noise and adversarial attacks, and reduce the number of parameters without compromising accuracy. However, moving beyond proof-of-concept or simulations to develop practical applications of these systems while ensuring high software quality faces significant challenges due to the limitations of quantum hardware and the underdeveloped knowledge base in software engineering for such systems. In this work, we have conducted a systematic mapping study to identify the challenges and solutions associated with the software architecture of quantum-enhanced artificial intelligence systems. The results of the systematic mapping study reveal several architectural patterns that describe how quantum components can be integrated into inference engines, as well as middleware patterns that facilitate communication between classical and quantum components. Each pattern realises a trade-off between various software quality attributes, such as efficiency, scalability, trainability, simplicity, portability, and deployability. The outcomes of this work have been compiled into a catalogue of architectural patterns.

摘要: 利用量子计算技术来增强人工智能系统，预计将改善训练和推理时间，提高对噪音和对手攻击的稳健性，并在不影响准确性的情况下减少参数数量。然而，由于量子硬件的限制和此类系统的软件工程知识库的不发达，超越概念验证或模拟来开发这些系统的实际应用，同时确保高软件质量面临着重大挑战。在这项工作中，我们进行了系统的映射研究，以确定与量子增强型人工智能系统的软件体系结构相关的挑战和解决方案。系统映射研究的结果揭示了几种描述量子组件如何集成到推理引擎中的架构模式，以及促进经典组件和量子组件之间通信的中间件模式。每个模式都实现了各种软件质量属性之间的权衡，例如效率、可伸缩性、可训练性、简单性、可移植性和可部署性。这项工作的成果已被汇编成建筑模式目录。



## **43. Adversarially robust generalization theory via Jacobian regularization for deep neural networks**

深度神经网络通过Jacobian正规化的对抗鲁棒概括理论 stat.ML

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12449v1) [paper-pdf](http://arxiv.org/pdf/2412.12449v1)

**Authors**: Dongya Wu, Xin Li

**Abstract**: Powerful deep neural networks are vulnerable to adversarial attacks. To obtain adversarially robust models, researchers have separately developed adversarial training and Jacobian regularization techniques. There are abundant theoretical and empirical studies for adversarial training, but theoretical foundations for Jacobian regularization are still lacking. In this study, we show that Jacobian regularization is closely related to adversarial training in that $\ell_{2}$ or $\ell_{1}$ Jacobian regularized loss serves as an approximate upper bound on the adversarially robust loss under $\ell_{2}$ or $\ell_{\infty}$ adversarial attack respectively. Further, we establish the robust generalization gap for Jacobian regularized risk minimizer via bounding the Rademacher complexity of both the standard loss function class and Jacobian regularization function class. Our theoretical results indicate that the norms of Jacobian are related to both standard and robust generalization. We also perform experiments on MNIST data classification to demonstrate that Jacobian regularized risk minimization indeed serves as a surrogate for adversarially robust risk minimization, and that reducing the norms of Jacobian can improve both standard and robust generalization. This study promotes both theoretical and empirical understandings to adversarially robust generalization via Jacobian regularization.

摘要: 强大的深度神经网络很容易受到敌意攻击。为了获得对抗性稳健的模型，研究人员分别开发了对抗性训练和雅可比正则化技术。对抗性训练已经有了丰富的理论和实证研究，但雅可比正则化的理论基础还很缺乏。本文证明了雅可比正则化与对抗训练密切相关，即雅可比正则化损失分别作为对抗性攻击下对抗性稳健损失的近似上界。进一步，我们通过对标准损失函数类和雅可比正则化函数类的Rademacher复杂性的界，建立了雅可比正则化风险最小化的鲁棒推广间隙。我们的理论结果表明，雅可比的范数既与标准推广有关，也与稳健推广有关。我们还在MNIST数据分类上进行了实验，证明了雅可比正则化风险最小化确实可以作为对抗性稳健风险最小化的替代，并且降低雅可比范数可以提高标准泛化和稳健泛化。本研究通过雅可比正则化，促进了对逆稳健泛化的理论和经验的理解。



## **44. Quantum Adversarial Machine Learning and Defense Strategies: Challenges and Opportunities**

量子对抗机器学习和防御策略：挑战和机遇 quant-ph

24 pages, 9 figures, 12 tables

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.12373v1) [paper-pdf](http://arxiv.org/pdf/2412.12373v1)

**Authors**: Eric Yocam, Anthony Rizi, Mahesh Kamepalli, Varghese Vaidyan, Yong Wang, Gurcan Comert

**Abstract**: As quantum computing continues to advance, the development of quantum-secure neural networks is crucial to prevent adversarial attacks. This paper proposes three quantum-secure design principles: (1) using post-quantum cryptography, (2) employing quantum-resistant neural network architectures, and (3) ensuring transparent and accountable development and deployment. These principles are supported by various quantum strategies, including quantum data anonymization, quantum-resistant neural networks, and quantum encryption. The paper also identifies open issues in quantum security, privacy, and trust, and recommends exploring adaptive adversarial attacks and auto adversarial attacks as future directions. The proposed design principles and recommendations provide guidance for developing quantum-secure neural networks, ensuring the integrity and reliability of machine learning models in the quantum era.

摘要: 随着量子计算的不断发展，量子安全神经网络的发展对于防止对抗攻击至关重要。本文提出了三个量子安全设计原则：（1）使用后量子密码学，（2）采用抗量子神经网络架构，（3）确保透明和负责任的开发和部署。这些原则得到各种量子策略的支持，包括量子数据匿名化、量子抵抗神经网络和量子加密。该论文还指出了量子安全、隐私和信任方面的未决问题，并建议探索自适应对抗攻击和自动对抗攻击作为未来的方向。提出的设计原则和建议为开发量子安全神经网络提供了指导，确保量子时代机器学习模型的完整性和可靠性。



## **45. Multi-Robot Target Tracking with Sensing and Communication Danger Zones**

具有传感和通信危险区的多机器人目标跟踪 cs.RO

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2404.07880v3) [paper-pdf](http://arxiv.org/pdf/2404.07880v3)

**Authors**: Jiazhen Liu, Peihan Li, Yuwei Wu, Gaurav S. Sukhatme, Vijay Kumar, Lifeng Zhou

**Abstract**: Multi-robot target tracking finds extensive applications in different scenarios, such as environmental surveillance and wildfire management, which require the robustness of the practical deployment of multi-robot systems in uncertain and dangerous environments. Traditional approaches often focus on the performance of tracking accuracy with no modeling and assumption of the environments, neglecting potential environmental hazards which result in system failures in real-world deployments. To address this challenge, we investigate multi-robot target tracking in the adversarial environment considering sensing and communication attacks with uncertainty. We design specific strategies to avoid different danger zones and proposed a multi-agent tracking framework under the perilous environment. We approximate the probabilistic constraints and formulate practical optimization strategies to address computational challenges efficiently. We evaluate the performance of our proposed methods in simulations to demonstrate the ability of robots to adjust their risk-aware behaviors under different levels of environmental uncertainty and risk confidence. The proposed method is further validated via real-world robot experiments where a team of drones successfully track dynamic ground robots while being risk-aware of the sensing and/or communication danger zones.

摘要: 多机器人目标跟踪在环境监测、野火管理等不同场景中有着广泛的应用，这就要求多机器人系统在不确定和危险环境中的实际部署具有很强的鲁棒性。传统的方法往往只关注跟踪精度的性能，没有对环境进行建模和假设，而忽略了实际部署中可能导致系统故障的环境危害。为了应对这一挑战，我们研究了在具有不确定性的感知和通信攻击的对抗性环境中的多机器人目标跟踪。设计了避开不同危险区域的具体策略，提出了危险环境下的多智能体跟踪框架。我们对概率约束进行近似，并制定实用的优化策略来有效地应对计算挑战。我们在仿真中评估了我们提出的方法的性能，以展示机器人在不同的环境不确定性和风险置信度下调整其风险意识行为的能力。通过真实世界的机器人实验进一步验证了所提出的方法，其中一组无人机成功地跟踪了动态的地面机器人，同时意识到了传感和/或通信危险区域的风险。



## **46. Adversarial Attacks on Large Language Models in Medicine**

医学中对大型语言模型的对抗攻击 cs.AI

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2406.12259v3) [paper-pdf](http://arxiv.org/pdf/2406.12259v3)

**Authors**: Yifan Yang, Qiao Jin, Furong Huang, Zhiyong Lu

**Abstract**: The integration of Large Language Models (LLMs) into healthcare applications offers promising advancements in medical diagnostics, treatment recommendations, and patient care. However, the susceptibility of LLMs to adversarial attacks poses a significant threat, potentially leading to harmful outcomes in delicate medical contexts. This study investigates the vulnerability of LLMs to two types of adversarial attacks in three medical tasks. Utilizing real-world patient data, we demonstrate that both open-source and proprietary LLMs are susceptible to manipulation across multiple tasks. This research further reveals that domain-specific tasks demand more adversarial data in model fine-tuning than general domain tasks for effective attack execution, especially for more capable models. We discover that while integrating adversarial data does not markedly degrade overall model performance on medical benchmarks, it does lead to noticeable shifts in fine-tuned model weights, suggesting a potential pathway for detecting and countering model attacks. This research highlights the urgent need for robust security measures and the development of defensive mechanisms to safeguard LLMs in medical applications, to ensure their safe and effective deployment in healthcare settings.

摘要: 将大型语言模型(LLM)集成到医疗保健应用程序中，在医疗诊断、治疗建议和患者护理方面提供了有希望的进步。然而，LLMS对对抗性攻击的敏感性构成了一个重大威胁，可能会在微妙的医疗环境中导致有害后果。本研究调查了LLMS在三个医疗任务中对两种类型的对抗性攻击的脆弱性。利用真实世界的患者数据，我们证明了开源和专有LLM都容易受到跨多个任务的操纵。这项研究进一步表明，特定领域的任务在模型微调中需要比一般领域任务更多的对抗性数据才能有效地执行攻击，特别是对于能力更强的模型。我们发现，虽然整合对抗性数据并不会显著降低医学基准上的整体模型性能，但它确实会导致微调模型权重的显著变化，这表明了一条检测和对抗模型攻击的潜在路径。这项研究强调了迫切需要强有力的安全措施和开发防御机制来保护医疗应用中的低成本管理，以确保其在医疗保健环境中的安全和有效部署。



## **47. Robust Synthetic Data-Driven Detection of Living-Off-the-Land Reverse Shells**

陆地生活反向壳的稳健综合数据驱动检测 cs.CR

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2402.18329v2) [paper-pdf](http://arxiv.org/pdf/2402.18329v2)

**Authors**: Dmitrijs Trizna, Luca Demetrio, Battista Biggio, Fabio Roli

**Abstract**: Living-off-the-land (LOTL) techniques pose a significant challenge to security operations, exploiting legitimate tools to execute malicious commands that evade traditional detection methods. To address this, we present a robust augmentation framework for cyber defense systems as Security Information and Event Management (SIEM) solutions, enabling the detection of LOTL attacks such as reverse shells through machine learning. Leveraging real-world threat intelligence and adversarial training, our framework synthesizes diverse malicious datasets while preserving the variability of legitimate activity, ensuring high accuracy and low false-positive rates. We validate our approach through extensive experiments on enterprise-scale datasets, achieving a 90\% improvement in detection rates over non-augmented baselines at an industry-grade False Positive Rate (FPR) of $10^{-5}$. We define black-box data-driven attacks that successfully evade unprotected models, and develop defenses to mitigate them, producing adversarially robust variants of ML models. Ethical considerations are central to this work; we discuss safeguards for synthetic data generation and the responsible release of pre-trained models across four best performing architectures, including both adversarially and regularly trained variants: https://huggingface.co/dtrizna/quasarnix. Furthermore, we provide a malicious LOTL dataset containing over 1 million augmented attack variants to enable reproducible research and community collaboration: https://huggingface.co/datasets/dtrizna/QuasarNix. This work offers a reproducible, scalable, and production-ready defense against evolving LOTL threats.

摘要: 谋生(LOTL)技术对安全操作构成了重大挑战，它们利用合法工具执行恶意命令，从而规避了传统的检测方法。为了解决这个问题，我们提出了一个强大的网络防御系统增强框架作为安全信息和事件管理(SIEM)解决方案，使能够通过机器学习检测LOTL攻击，如反向外壳。利用真实世界的威胁情报和对抗训练，我们的框架综合了不同的恶意数据集，同时保留了合法活动的可变性，确保了高准确性和低假阳性率。我们通过在企业级数据集上的广泛实验验证了我们的方法，在行业级假阳性率(FPR)为10^{-5}$的情况下，检测率比非增强基线提高了90%。我们定义了成功逃避不受保护的模型的黑盒数据驱动攻击，并开发防御措施来缓解它们，产生对手健壮的ML模型变体。伦理方面的考虑是这项工作的核心；我们讨论了合成数据生成的保障措施，以及在四个性能最好的体系结构中负责任地发布预先训练的模型，包括对抗性和常规训练的变体：https://huggingface.co/dtrizna/quasarnix.此外，我们还提供了一个包含100多万个扩展攻击变体的恶意LOTL数据集，以支持可重现的研究和社区协作：https://huggingface.co/datasets/dtrizna/QuasarNix.这项工作提供了针对不断演变的LOTL威胁的可复制、可扩展和生产就绪的防御。



## **48. Sonar-based Deep Learning in Underwater Robotics: Overview, Robustness and Challenges**

水下机器人中基于声纳的深度学习：概述、稳健性和挑战 cs.RO

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.11840v1) [paper-pdf](http://arxiv.org/pdf/2412.11840v1)

**Authors**: Martin Aubard, Ana Madureira, Luís Teixeira, José Pinto

**Abstract**: With the growing interest in underwater exploration and monitoring, Autonomous Underwater Vehicles (AUVs) have become essential. The recent interest in onboard Deep Learning (DL) has advanced real-time environmental interaction capabilities relying on efficient and accurate vision-based DL models. However, the predominant use of sonar in underwater environments, characterized by limited training data and inherent noise, poses challenges to model robustness. This autonomy improvement raises safety concerns for deploying such models during underwater operations, potentially leading to hazardous situations. This paper aims to provide the first comprehensive overview of sonar-based DL under the scope of robustness. It studies sonar-based DL perception task models, such as classification, object detection, segmentation, and SLAM. Furthermore, the paper systematizes sonar-based state-of-the-art datasets, simulators, and robustness methods such as neural network verification, out-of-distribution, and adversarial attacks. This paper highlights the lack of robustness in sonar-based DL research and suggests future research pathways, notably establishing a baseline sonar-based dataset and bridging the simulation-to-reality gap.

摘要: 随着人们对水下探测和监测的兴趣与日俱增，自主水下机器人(AUV)已经成为必不可少的工具。最近对车载深度学习(DL)的兴趣依赖于高效和准确的基于视觉的DL模型，具有先进的实时环境交互能力。然而，声纳在水下环境中的主要应用具有训练数据有限和固有噪声的特点，这给模型的稳健性带来了挑战。这种自主性的改进增加了在水下作业期间部署此类模型的安全问题，可能会导致危险情况。本文的目的是在稳健性的范围内，首次对基于声纳的数字水声通信进行全面的综述。研究了基于声纳的目标识别任务模型，如分类、目标检测、分割、SLAM等。此外，本文还系统化了基于声纳的最先进的数据集、模拟器和稳健性方法，如神经网络验证、分布外分布和敌方攻击。本文强调了基于声纳的数字图书馆研究缺乏稳健性，并提出了未来的研究路径，特别是建立基于声纳的基线数据集和弥合模拟与现实之间的差距。



## **49. Transferable Adversarial Face Attack with Text Controlled Attribute**

具有文本控制属性的可转移对抗面部攻击 cs.CV

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.11735v1) [paper-pdf](http://arxiv.org/pdf/2412.11735v1)

**Authors**: Wenyun Li, Zheng Zhang, Xiangyuan Lan, Dongmei Jiang

**Abstract**: Traditional adversarial attacks typically produce adversarial examples under norm-constrained conditions, whereas unrestricted adversarial examples are free-form with semantically meaningful perturbations. Current unrestricted adversarial impersonation attacks exhibit limited control over adversarial face attributes and often suffer from low transferability. In this paper, we propose a novel Text Controlled Attribute Attack (TCA$^2$) to generate photorealistic adversarial impersonation faces guided by natural language. Specifically, the category-level personal softmax vector is employed to precisely guide the impersonation attacks. Additionally, we propose both data and model augmentation strategies to achieve transferable attacks on unknown target models. Finally, a generative model, \textit{i.e}, Style-GAN, is utilized to synthesize impersonated faces with desired attributes. Extensive experiments on two high-resolution face recognition datasets validate that our TCA$^2$ method can generate natural text-guided adversarial impersonation faces with high transferability. We also evaluate our method on real-world face recognition systems, \textit{i.e}, Face++ and Aliyun, further demonstrating the practical potential of our approach.

摘要: 传统的对抗性攻击通常在范数受限的条件下产生对抗性示例，而不受限制的对抗性示例是自由形式的，具有语义意义的扰动。当前不受限制的对抗性模仿攻击对对抗性面孔属性的控制有限，并且往往存在可转移性低的问题。本文提出了一种新的文本控制属性攻击(TCA$^2$)，用于生成自然语言引导下的照片真实感对抗性模仿人脸。具体地说，采用类别级的个人Softmax向量来精确地指导模仿攻击。此外，我们还提出了数据和模型扩充策略来实现对未知目标模型的可转移攻击。最后，利用一个生成模型在两个高分辨率人脸识别数据集上的大量实验验证了我们的TCA$^2$方法能够生成具有很高可转移性的自然文本引导的对抗性模拟人脸。我们还在真实的人脸识别系统



## **50. Against All Odds: Overcoming Typology, Script, and Language Confusion in Multilingual Embedding Inversion Attacks**

克服一切困难：克服多语言嵌入倒置攻击中的类型学、脚本和语言混乱 cs.CL

11 pages, 4 figures, 7 tables

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2408.11749v2) [paper-pdf](http://arxiv.org/pdf/2408.11749v2)

**Authors**: Yiyi Chen, Russa Biswas, Heather Lent, Johannes Bjerva

**Abstract**: Large Language Models (LLMs) are susceptible to malicious influence by cyber attackers through intrusions such as adversarial, backdoor, and embedding inversion attacks. In response, the burgeoning field of LLM Security aims to study and defend against such threats. Thus far, the majority of works in this area have focused on monolingual English models, however, emerging research suggests that multilingual LLMs may be more vulnerable to various attacks than their monolingual counterparts. While previous work has investigated embedding inversion over a small subset of European languages, it is challenging to extrapolate these findings to languages from different linguistic families and with differing scripts. To this end, we explore the security of multilingual LLMs in the context of embedding inversion attacks and investigate cross-lingual and cross-script inversion across 20 languages, spanning over 8 language families and 12 scripts. Our findings indicate that languages written in Arabic script and Cyrillic script are particularly vulnerable to embedding inversion, as are languages within the Indo-Aryan language family. We further observe that inversion models tend to suffer from language confusion, sometimes greatly reducing the efficacy of an attack. Accordingly, we systematically explore this bottleneck for inversion models, uncovering predictable patterns which could be leveraged by attackers. Ultimately, this study aims to further the field's understanding of the outstanding security vulnerabilities facing multilingual LLMs and raise awareness for the languages most at risk of negative impact from these attacks.

摘要: 大型语言模型(LLM)容易受到网络攻击者通过对抗性、后门和嵌入反转攻击等入侵的恶意影响。作为回应，LLM Security这个新兴领域的目标是研究和防御此类威胁。到目前为止，这一领域的研究大多集中在单语英语模型上，然而，新的研究表明，多语种的LLM可能比单语的LLM更容易受到各种攻击。虽然以前的工作已经研究了在一小部分欧洲语言上嵌入倒置，但将这些发现外推到来自不同语系和不同脚本的语言是具有挑战性的。为此，我们在嵌入倒置攻击的情况下探索了多语言LLMS的安全性，并研究了跨语言和跨脚本的跨语言和跨脚本倒置，涉及8个语系和12个脚本。我们的发现表明，用阿拉伯文字和西里尔文字书写的语言特别容易嵌入倒置，印度-雅利安语系的语言也是如此。我们进一步观察到，倒置模型往往受到语言混乱的影响，有时会极大地降低攻击的有效性。因此，我们系统地探索了倒置模型的这一瓶颈，揭示了可被攻击者利用的可预测模式。最终，这项研究旨在加深外地对多语种土地管理系统面临的突出安全漏洞的了解，并提高对最有可能受到这些攻击的负面影响的语言的认识。



