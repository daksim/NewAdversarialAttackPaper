# Latest Adversarial Attack Papers
**update at 2025-01-10 09:46:28**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Correlated Privacy Mechanisms for Differentially Private Distributed Mean Estimation**

用于差异私有分布均值估计的相关隐私机制 cs.IT

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2407.03289v2) [paper-pdf](http://arxiv.org/pdf/2407.03289v2)

**Authors**: Sajani Vithana, Viveck R. Cadambe, Flavio P. Calmon, Haewon Jeong

**Abstract**: Differentially private distributed mean estimation (DP-DME) is a fundamental building block in privacy-preserving federated learning, where a central server estimates the mean of $d$-dimensional vectors held by $n$ users while ensuring $(\epsilon,\delta)$-DP. Local differential privacy (LDP) and distributed DP with secure aggregation (SA) are the most common notions of DP used in DP-DME settings with an untrusted server. LDP provides strong resilience to dropouts, colluding users, and adversarial attacks, but suffers from poor utility. In contrast, SA-based DP-DME achieves an $O(n)$ utility gain over LDP in DME, but requires increased communication and computation overheads and complex multi-round protocols to handle dropouts and attacks. In this work, we present a generalized framework for DP-DME, that captures LDP and SA-based mechanisms as extreme cases. Our framework provides a foundation for developing and analyzing a variety of DP-DME protocols that leverage correlated privacy mechanisms across users. To this end, we propose CorDP-DME, a novel DP-DME mechanism based on the correlated Gaussian mechanism, that spans the gap between DME with LDP and distributed DP. We prove that CorDP-DME offers a favorable balance between utility and resilience to dropout and collusion. We provide an information-theoretic analysis of CorDP-DME, and derive theoretical guarantees for utility under any given privacy parameters and dropout/colluding user thresholds. Our results demonstrate that (anti) correlated Gaussian DP mechanisms can significantly improve utility in mean estimation tasks compared to LDP -- even in adversarial settings -- while maintaining better resilience to dropouts and attacks compared to distributed DP.

摘要: 差分私有分布平均估计(DP-DME)是保护隐私的联合学习的基本构件，其中中央服务器估计$n$用户所持有的$d$维向量的平均值，同时确保$(？，？)$-DP。本地差异隐私(LDP)和具有安全聚合的分布式DP(SA)是DP-DME设置中使用不受信任服务器的最常见概念。LDP对辍学、串通用户和敌意攻击具有很强的弹性，但实用性较差。相比之下，基于SA的DP-DME在DME中比LDP获得$O(N)$效用收益，但需要更多的通信和计算开销以及复杂的多轮协议来处理丢弃和攻击。在这项工作中，我们提出了一个通用的DP-DME框架，该框架将基于LDP和SA的机制作为极端情况来捕获。我们的框架为开发和分析各种DP-DME协议提供了基础，这些协议利用了跨用户的相关隐私机制。为此，我们提出了一种新的基于相关高斯机制的DP-DME机制CorDP-DME，它跨越了DME与LDP和分布式DP之间的差距。我们证明了CorDP-DME在实用性和对丢弃和共谋的恢复能力之间提供了良好的平衡。我们对CorDP-DME进行了信息论分析，并推导出在任何给定的隐私参数和丢弃/合谋用户阈值下的效用的理论保证。我们的结果表明，与LDP相比，(反)相关的高斯DP机制可以显著提高均值估计任务的实用性--即使在对抗环境中--同时与分布式DP相比，保持更好的对丢弃和攻击的弹性。



## **2. Resilient Peer-to-peer Learning based on Adaptive Aggregation**

基于自适应聚合的弹性点对点学习 cs.LG

11 pages

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.04610v1) [paper-pdf](http://arxiv.org/pdf/2501.04610v1)

**Authors**: Chandreyee Bhowmick, Xenofon Koutsoukos

**Abstract**: Collaborative learning in peer-to-peer networks offers the benefits of distributed learning while mitigating the risks associated with single points of failure inherent in centralized servers. However, adversarial workers pose potential threats by attempting to inject malicious information into the network. Thus, ensuring the resilience of peer-to-peer learning emerges as a pivotal research objective. The challenge is exacerbated in the presence of non-convex loss functions and non-iid data distributions. This paper introduces a resilient aggregation technique tailored for such scenarios, aimed at fostering similarity among peers' learning processes. The aggregation weights are determined through an optimization procedure, and use the loss function computed using the neighbor's models and individual private data, thereby addressing concerns regarding data privacy in distributed machine learning. Theoretical analysis demonstrates convergence of parameters with non-convex loss functions and non-iid data distributions. Empirical evaluations across three distinct machine learning tasks support the claims. The empirical findings, which encompass a range of diverse attack models, also demonstrate improved accuracy when compared to existing methodologies.

摘要: 对等网络中的协作学习提供了分布式学习的好处，同时降低了与集中式服务器固有的单点故障相关的风险。然而，敌意工作者试图将恶意信息注入网络，从而构成潜在威胁。因此，确保对等学习的弹性成为一个关键的研究目标。在存在非凸损失函数和非IID数据分布的情况下，这一挑战更加严重。本文介绍了一种为此类场景量身定做的弹性聚合技术，旨在促进节点学习过程之间的相似性。聚集权重通过优化过程确定，并使用使用邻居的模型和个人隐私数据计算的损失函数，从而解决了分布式机器学习中对数据隐私的担忧。理论分析证明了参数在非凸损失函数和非IID数据分布下的收敛。对三个不同的机器学习任务进行的经验评估支持了这一说法。这些经验发现涵盖了一系列不同的攻击模型，与现有方法相比，也证明了更高的准确性。



## **3. Tougher Text, Smarter Models: Raising the Bar for Adversarial Defence Benchmarks**

更强硬的文本，更智能的模型：提高对抗性防御基准的门槛 cs.CL

Will be presented as an oral in-person presentation at the conference  of COLING 2025

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.02654v2) [paper-pdf](http://arxiv.org/pdf/2501.02654v2)

**Authors**: Yang Wang, Chenghua Lin

**Abstract**: Recent advancements in natural language processing have highlighted the vulnerability of deep learning models to adversarial attacks. While various defence mechanisms have been proposed, there is a lack of comprehensive benchmarks that evaluate these defences across diverse datasets, models, and tasks. In this work, we address this gap by presenting an extensive benchmark for textual adversarial defence that significantly expands upon previous work. Our benchmark incorporates a wide range of datasets, evaluates state-of-the-art defence mechanisms, and extends the assessment to include critical tasks such as single-sentence classification, similarity and paraphrase identification, natural language inference, and commonsense reasoning. This work not only serves as a valuable resource for researchers and practitioners in the field of adversarial robustness but also identifies key areas for future research in textual adversarial defence. By establishing a new standard for benchmarking in this domain, we aim to accelerate progress towards more robust and reliable natural language processing systems.

摘要: 自然语言处理的最新进展突显了深度学习模型在对抗性攻击中的脆弱性。虽然已经提出了各种防御机制，但缺乏全面的基准来评估不同数据集、模型和任务中的这些防御。在这项工作中，我们通过提供一个广泛的文本对抗防御基准来解决这一差距，该基准大大扩展了以前的工作。我们的基准纳入了广泛的数据集，评估了最先进的防御机制，并将评估扩展到包括关键任务，如单句分类、相似性和释义识别、自然语言推理和常识推理。这项工作不仅为对抗稳健性领域的研究人员和实践者提供了宝贵的资源，而且还确定了未来文本对抗防御研究的关键领域。通过在这一领域建立一个新的基准标准，我们的目标是加快朝着更健壮和可靠的自然语言处理系统的进展。



## **4. Towards Fair Class-wise Robustness: Class Optimal Distribution Adversarial Training**

迈向公平的班级稳健性：班级最优分布对抗训练 cs.LG

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.04527v1) [paper-pdf](http://arxiv.org/pdf/2501.04527v1)

**Authors**: Hongxin Zhi, Hongtao Yu, Shaome Li, Xiuming Zhao, Yiteng Wu

**Abstract**: Adversarial training has proven to be a highly effective method for improving the robustness of deep neural networks against adversarial attacks. Nonetheless, it has been observed to exhibit a limitation in terms of robust fairness, characterized by a significant disparity in robustness across different classes. Recent efforts to mitigate this problem have turned to class-wise reweighted methods. However, these methods suffer from a lack of rigorous theoretical analysis and are limited in their exploration of the weight space, as they mainly rely on existing heuristic algorithms or intuition to compute weights. In addition, these methods fail to guarantee the consistency of the optimization direction due to the decoupled optimization of weights and the model parameters. They potentially lead to suboptimal weight assignments and consequently, a suboptimal model. To address these problems, this paper proposes a novel min-max training framework, Class Optimal Distribution Adversarial Training (CODAT), which employs distributionally robust optimization to fully explore the class-wise weight space, thus enabling the identification of the optimal weight with theoretical guarantees. Furthermore, we derive a closed-form optimal solution to the internal maximization and then get a deterministic equivalent objective function, which provides a theoretical basis for the joint optimization of weights and model parameters. Meanwhile, we propose a fairness elasticity coefficient for the evaluation of the algorithm with regard to both robustness and robust fairness. Experimental results on various datasets show that the proposed method can effectively improve the robust fairness of the model and outperform the state-of-the-art approaches.

摘要: 对抗性训练已被证明是提高深层神经网络抵抗对抗性攻击的鲁棒性的一种非常有效的方法。尽管如此，人们观察到它在稳健公平方面表现出局限性，其特征是不同类别之间的稳健程度存在显著差异。最近缓解这一问题的努力已转向按类别重新加权的方法。然而，这些方法缺乏严谨的理论分析，并且主要依靠现有的启发式算法或直觉来计算权重，从而限制了对权重空间的探索。此外，由于权重和模型参数的解耦优化，这些方法不能保证优化方向的一致性。它们可能导致次优的权重分配，从而导致次优的模型。针对这些问题，本文提出了一种新的最小-最大训练框架--类最优分布对抗训练(CODAT)，该框架采用分布稳健优化来充分探索类的权值空间，从而在理论上保证了最优权值的识别。进而推导出内部极大化的闭合最优解，进而得到确定性的等价目标函数，为权重和模型参数的联合优化提供了理论依据。同时，我们还提出了一个公平弹性系数来评价算法的健壮性和健壮性。在不同数据集上的实验结果表明，该方法能有效地提高模型的鲁棒性公平性，并优于现有的方法。



## **5. Multichannel Steganography: A Provably Secure Hybrid Steganographic Model for Secure Communication**

多通道隐写术：一种可证明安全的混合隐写术模型，用于安全通信 cs.CR

18 pages, 8 figures, 3 algorithms, This version is a preprint  uploaded to arXiv

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.04511v1) [paper-pdf](http://arxiv.org/pdf/2501.04511v1)

**Authors**: Obinna Omego, Michal Bosy

**Abstract**: This study introduces a novel steganographic model that synthesizes Steganography by Cover Modification (CMO) and Steganography by Cover Synthesis (CSY), enhancing both security and undetectability by generating cover messages or parameters while retaining the original cover's form, thus minimizing detection risks and overcoming the limitations of single-method techniques. Building upon this model, a refined Steganographic Communication Protocol is proposed, enhancing resilience against sophisticated threats such as Multichannel Replay Attacks and Multichannel Man-in-the-Middle Attacks, fortifying the protocol against potential tampering and improving upon prior works. To evaluate the security of the proposed protocol, a novel adversarial model is developed simulating a probabilistic polynomial time (PPT) adversary capable of intercepting communications across multiple channels. This model assesses the adversary's ability to compromise the protocol, providing a comprehensive security analysis. Finally, this study explores the practicality and adaptability of the model to both constrained environments like SMS banking and resource-rich settings such as blockchain transactions, demonstrating their potential to enhance financial services and security. These contributions present a robust and adaptable framework for secure steganographic communication, offering practical solutions for secure communications across diverse environments.

摘要: 提出了一种新的隐写模型，该模型综合了基于覆盖修改的隐写(CMO)和基于覆盖合成的隐写(CSY)，通过生成覆盖消息或参数来增强安全性和不可检测性，同时保持了原始覆盖的形式，从而最大限度地降低了检测风险，克服了单一方法技术的局限性。在此模型的基础上，提出了一种改进的隐写通信协议，增强了对多通道重放攻击和多通道中间人攻击等复杂威胁的抵御能力，增强了协议对潜在篡改的抵抗力，并在已有工作的基础上进行了改进。为了评估该协议的安全性，建立了一个新的敌手模型，该模型模拟了概率多项式时间(PPT)敌手在多个信道上截获通信的能力。该模型评估对手破坏协议的能力，提供全面的安全分析。最后，本研究探讨了该模型对短信银行等受限环境和区块链交易等资源丰富环境的实用性和适应性，展示了其增强金融服务和安全性的潜力。这些贡献为安全隐写通信提供了一个强大和适应性强的框架，为跨不同环境的安全通信提供了实用的解决方案。



## **6. Rethinking Byzantine Robustness in Federated Recommendation from Sparse Aggregation Perspective**

从稀疏聚集角度重新思考联邦推荐中的拜占庭鲁棒性 cs.CR

accepted by AAAI 2025

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.03301v2) [paper-pdf](http://arxiv.org/pdf/2501.03301v2)

**Authors**: Zhongjian Zhang, Mengmei Zhang, Xiao Wang, Lingjuan Lyu, Bo Yan, Junping Du, Chuan Shi

**Abstract**: To preserve user privacy in recommender systems, federated recommendation (FR) based on federated learning (FL) emerges, keeping the personal data on the local client and updating a model collaboratively. Unlike FL, FR has a unique sparse aggregation mechanism, where the embedding of each item is updated by only partial clients, instead of full clients in a dense aggregation of general FL. Recently, as an essential principle of FL, model security has received increasing attention, especially for Byzantine attacks, where malicious clients can send arbitrary updates. The problem of exploring the Byzantine robustness of FR is particularly critical since in the domains applying FR, e.g., e-commerce, malicious clients can be injected easily by registering new accounts. However, existing Byzantine works neglect the unique sparse aggregation of FR, making them unsuitable for our problem. Thus, we make the first effort to investigate Byzantine attacks on FR from the perspective of sparse aggregation, which is non-trivial: it is not clear how to define Byzantine robustness under sparse aggregations and design Byzantine attacks under limited knowledge/capability. In this paper, we reformulate the Byzantine robustness under sparse aggregation by defining the aggregation for a single item as the smallest execution unit. Then we propose a family of effective attack strategies, named Spattack, which exploit the vulnerability in sparse aggregation and are categorized along the adversary's knowledge and capability. Extensive experimental results demonstrate that Spattack can effectively prevent convergence and even break down defenses under a few malicious clients, raising alarms for securing FR systems.

摘要: 为了在推荐系统中保护用户隐私，基于联合学习的联合推荐(FR)应运而生，它将个人数据保存在本地客户端，并协作更新模型。与FL不同，FR具有独特的稀疏聚合机制，其中每个项的嵌入只由部分客户端更新，而不是由普通FL的密集聚合中的完整客户端更新。最近，作为FL的一项基本原则，模型安全性受到了越来越多的关注，特别是对于拜占庭攻击，恶意客户端可以发送任意更新。探索FR的拜占庭健壮性的问题尤其关键，因为在应用FR的域中，例如电子商务，可以通过注册新帐户轻松地注入恶意客户端。然而，现有的拜占庭著作忽略了FR独特的稀疏聚集，这使得它们不适合我们的问题。因此，我们首次尝试从稀疏聚集的角度来研究拜占庭攻击FR，这并不是一件平凡的事情：如何定义稀疏聚集下的拜占庭健壮性，以及在有限的知识/能力下设计拜占庭攻击，目前还不清楚。本文通过将单个项的聚集定义为最小执行单元，重新定义了稀疏聚集下的拜占庭健壮性。然后，我们提出了一系列有效的攻击策略Spattack，该策略利用了稀疏聚集的脆弱性，并根据对手的知识和能力进行分类。广泛的实验结果表明，Spattack能够有效地防止收敛，甚至在少数恶意客户端下破坏防御，为FR系统的安全发出警报。



## **7. Rethinking Adversarial Attacks in Reinforcement Learning from Policy Distribution Perspective**

从策略分布角度重新思考强化学习中的对抗性攻击 cs.LG

10 pages, 2 figures, 2 tables

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2501.03562v2) [paper-pdf](http://arxiv.org/pdf/2501.03562v2)

**Authors**: Tianyang Duan, Zongyuan Zhang, Zheng Lin, Yue Gao, Ling Xiong, Yong Cui, Hongbin Liang, Xianhao Chen, Heming Cui, Dong Huang

**Abstract**: Deep Reinforcement Learning (DRL) suffers from uncertainties and inaccuracies in the observation signal in realworld applications. Adversarial attack is an effective method for evaluating the robustness of DRL agents. However, existing attack methods targeting individual sampled actions have limited impacts on the overall policy distribution, particularly in continuous action spaces. To address these limitations, we propose the Distribution-Aware Projected Gradient Descent attack (DAPGD). DAPGD uses distribution similarity as the gradient perturbation input to attack the policy network, which leverages the entire policy distribution rather than relying on individual samples. We utilize the Bhattacharyya distance in DAPGD to measure policy similarity, enabling sensitive detection of subtle but critical differences between probability distributions. Our experiment results demonstrate that DAPGD achieves SOTA results compared to the baselines in three robot navigation tasks, achieving an average 22.03% higher reward drop compared to the best baseline.

摘要: 深度强化学习(DRL)在实际应用中存在观测信号的不确定性和不准确性。对抗性攻击是评价DRL代理健壮性的有效方法。然而，现有的针对单个采样动作的攻击方法对整体策略分布的影响有限，特别是在连续动作空间中。为了解决这些局限性，我们提出了分布感知投影梯度下降攻击(DAPGD)。DAPGD使用分布相似度作为梯度扰动输入来攻击策略网络，该策略网络利用整个策略分布而不是依赖于单个样本。我们利用DAPGD中的Bhattacharyya距离来度量策略相似性，从而能够敏感地检测到概率分布之间的细微但关键的差异。实验结果表明，在三个机器人导航任务中，DAPGD获得了与基线相比的SOTA结果，获得了比最佳基线平均高22.03%的奖赏下降。



## **8. Location Privacy Threats and Protections in 6G Vehicular Networks: A Comprehensive Review**

6G车载网络中的位置隐私威胁和保护：全面回顾 cs.CR

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2305.04503v2) [paper-pdf](http://arxiv.org/pdf/2305.04503v2)

**Authors**: Baihe Ma, Xu Wang, Xiaojie Lin, Yanna Jiang, Caijun Sun, Zhe Wang, Guangsheng Yu, Suirui Zhu, Ying He, Wei Ni, Ren Ping Liu

**Abstract**: Location privacy is critical in vehicular networks, where drivers' trajectories and personal information can be exposed, allowing adversaries to launch data and physical attacks that threaten drivers' safety and personal security. This survey reviews comprehensively different localization techniques, including widely used ones like sensing infrastructure-based, optical vision-based, and cellular radio-based localization, and identifies inadequately addressed location privacy concerns. We classify Location Privacy Preserving Mechanisms (LPPMs) into user-side, server-side, and user-server-interface-based, and evaluate their effectiveness. Our analysis shows that the user-server-interface-based LPPMs have received insufficient attention in the literature, despite their paramount importance in vehicular networks. Further, we examine methods for balancing data utility and privacy protection for existing LPPMs in vehicular networks and highlight emerging challenges from future upper-layer location privacy attacks, wireless technologies, and network convergences. By providing insights into the relationship between localization techniques and location privacy, and evaluating the effectiveness of different LPPMs, this survey can help inform the development of future LPPMs in vehicular networks.

摘要: 位置隐私在车载网络中至关重要，在车载网络中，司机的轨迹和个人信息可能会被暴露，从而允许对手发动威胁司机安全和人身安全的数据和物理攻击。这项调查全面回顾了不同的定位技术，包括广泛使用的基于传感基础设施的定位技术、基于光学视觉的定位技术和基于蜂窝无线电的定位技术，并确定了没有充分解决位置隐私问题的问题。我们将位置隐私保护机制(LPPM)分为基于用户端、基于服务器端和基于用户-服务器接口，并对其有效性进行了评估。我们的分析表明，基于用户-服务器接口的LPPM在文献中没有得到足够的关注，尽管它们在车载网络中非常重要。此外，我们还研究了在车载网络中平衡现有LPPM的数据效用和隐私保护的方法，并强调了未来上层位置隐私攻击、无线技术和网络融合带来的新挑战。通过深入了解定位技术和位置隐私之间的关系，以及评估不同LPPM的有效性，这项调查有助于为未来LPPM在车载网络中的发展提供信息。



## **9. Proof-of-Learning with Incentive Security**

具有激励保障的学习证明 cs.CR

20 pages, 4 figures

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2404.09005v7) [paper-pdf](http://arxiv.org/pdf/2404.09005v7)

**Authors**: Zishuo Zhao, Zhixuan Fang, Xuechao Wang, Xi Chen, Hongxu Su, Haibo Xiao, Yuan Zhou

**Abstract**: Most concurrent blockchain systems rely heavily on the Proof-of-Work (PoW) or Proof-of-Stake (PoS) mechanisms for decentralized consensus and security assurance. However, the substantial energy expenditure stemming from computationally intensive yet meaningless tasks has raised considerable concerns surrounding traditional PoW approaches, The PoS mechanism, while free of energy consumption, is subject to security and economic issues. Addressing these issues, the paradigm of Proof-of-Useful-Work (PoUW) seeks to employ challenges of practical significance as PoW, thereby imbuing energy consumption with tangible value. While previous efforts in Proof of Learning (PoL) explored the utilization of deep learning model training SGD tasks as PoUW challenges, recent research has revealed its vulnerabilities to adversarial attacks and the theoretical hardness in crafting a byzantine-secure PoL mechanism. In this paper, we introduce the concept of incentive-security that incentivizes rational provers to behave honestly for their best interest, bypassing the existing hardness to design a PoL mechanism with computational efficiency, a provable incentive-security guarantee and controllable difficulty. Particularly, our work is secure against two attacks, and also improves the computational overhead from $\Theta(1)$ to $O(\frac{\log E}{E})$. Furthermore, while most recent research assumes trusted problem providers and verifiers, our design also guarantees frontend incentive-security even when problem providers are untrusted, and verifier incentive-security that bypasses the Verifier's Dilemma. By incorporating ML training into blockchain consensus mechanisms with provable guarantees, our research not only proposes an eco-friendly solution to blockchain systems, but also provides a proposal for a completely decentralized computing power market in the new AI age.

摘要: 大多数并发区块链系统严重依赖工作证明(POW)或风险证明(POS)机制来实现去中心化共识和安全保证。然而，计算密集但无意义的任务所产生的大量能源支出引起了人们对传统POW方法的相当大的担忧，POS机制虽然没有能源消耗，但受到安全和经济问题的影响。针对这些问题，有用工作证明(POUW)范式试图将具有实际意义的挑战作为POW来使用，从而使能源消耗具有有形价值。虽然先前在学习证明(Pol)方面的努力探索了利用深度学习模型训练SGD任务作为POW挑战，但最近的研究揭示了它对对手攻击的脆弱性以及在设计拜占庭安全的POL机制方面的理论难度。本文引入激励安全的概念，激励理性的证明者为了他们的最大利益而诚实地行事，绕过现有的困难，设计了一个具有计算效率、可证明的激励安全保证和可控难度的POL机制。特别地，我们的工作对两次攻击是安全的，并且还将计算开销从$\theta(1)$提高到$O(\frac{\log E}{E})$。此外，虽然最近的研究假设可信的问题提供者和验证者，但我们的设计也保证了前端激励-安全性，即使问题提供者是不可信的，并且验证者激励-安全绕过了验证者的困境。通过将ML培训融入到具有可证明保证的区块链共识机制中，我们的研究不仅为区块链系统提出了生态友好的解决方案，而且为新AI时代完全去中心化的计算能力市场提供了建议。



## **10. Light-weight Fine-tuning Method for Defending Adversarial Noise in Pre-trained Medical Vision-Language Models**

在预训练的医学视觉语言模型中防御对抗性噪音的轻量级微调方法 cs.CV

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2407.02716v2) [paper-pdf](http://arxiv.org/pdf/2407.02716v2)

**Authors**: Xu Han, Linghao Jin, Xuezhe Ma, Xiaofeng Liu

**Abstract**: Fine-tuning pre-trained Vision-Language Models (VLMs) has shown remarkable capabilities in medical image and textual depiction synergy. Nevertheless, many pre-training datasets are restricted by patient privacy concerns, potentially containing noise that can adversely affect downstream performance. Moreover, the growing reliance on multi-modal generation exacerbates this issue because of its susceptibility to adversarial attacks. To investigate how VLMs trained on adversarial noisy data perform on downstream medical tasks, we first craft noisy upstream datasets using multi-modal adversarial attacks. Through our comprehensive analysis, we unveil that moderate noise enhances model robustness and transferability, but increasing noise levels negatively impact downstream task performance. To mitigate this issue, we propose rectify adversarial noise (RAN) framework, a recipe designed to effectively defend adversarial attacks and rectify the influence of upstream noise during fine-tuning.

摘要: 微调预训练的视觉语言模型（VLM）在医学图像和文本描述协同方面表现出了非凡的能力。然而，许多预训练数据集受到患者隐私问题的限制，可能包含可能对下游性能产生不利影响的噪音。此外，对多模式发电的日益依赖加剧了这个问题，因为它容易受到对抗攻击。为了研究在对抗性有噪数据上训练的VLM如何执行下游医疗任务，我们首先使用多模式对抗攻击来制作有噪的上游数据集。通过我们的全面分析，我们发现适度的噪音增强了模型的稳健性和可移植性，但增加噪音水平会对下游任务性能产生负面影响。为了缓解这个问题，我们提出了纠正对抗性噪音（RAN）框架，该框架旨在有效防御对抗性攻击并纠正微调期间上游噪音的影响。



## **11. Synthetic Data Privacy Metrics**

合成数据隐私收件箱 cs.LG

14 pages, 2 figures

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.03941v1) [paper-pdf](http://arxiv.org/pdf/2501.03941v1)

**Authors**: Amy Steier, Lipika Ramaswamy, Andre Manoel, Alexa Haushalter

**Abstract**: Recent advancements in generative AI have made it possible to create synthetic datasets that can be as accurate as real-world data for training AI models, powering statistical insights, and fostering collaboration with sensitive datasets while offering strong privacy guarantees. Effectively measuring the empirical privacy of synthetic data is an important step in the process. However, while there is a multitude of new privacy metrics being published every day, there currently is no standardization. In this paper, we review the pros and cons of popular metrics that include simulations of adversarial attacks. We also review current best practices for amending generative models to enhance the privacy of the data they create (e.g. differential privacy).

摘要: 生成性人工智能的最新进展使创建与现实世界数据一样准确的合成数据集成为可能，用于训练人工智能模型、支持统计洞察并促进与敏感数据集的协作，同时提供强大的隐私保证。有效测量合成数据的经验隐私是该过程中的重要一步。然而，虽然每天都会发布大量新的隐私指标，但目前还没有标准化。在本文中，我们回顾了包括对抗攻击模拟在内的流行指标的利弊。我们还审查了修改生成模型以增强其创建数据的隐私性（例如差异隐私）的当前最佳实践。



## **12. Not all tokens are created equal: Perplexity Attention Weighted Networks for AI generated text detection**

并非所有令牌都是平等的：用于人工智能生成文本检测的困惑注意力加权网络 cs.CL

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.03940v1) [paper-pdf](http://arxiv.org/pdf/2501.03940v1)

**Authors**: Pablo Miralles-González, Javier Huertas-Tato, Alejandro Martín, David Camacho

**Abstract**: The rapid advancement in large language models (LLMs) has significantly enhanced their ability to generate coherent and contextually relevant text, raising concerns about the misuse of AI-generated content and making it critical to detect it. However, the task remains challenging, particularly in unseen domains or with unfamiliar LLMs. Leveraging LLM next-token distribution outputs offers a theoretically appealing approach for detection, as they encapsulate insights from the models' extensive pre-training on diverse corpora. Despite its promise, zero-shot methods that attempt to operationalize these outputs have met with limited success. We hypothesize that one of the problems is that they use the mean to aggregate next-token distribution metrics across tokens, when some tokens are naturally easier or harder to predict and should be weighted differently. Based on this idea, we propose the Perplexity Attention Weighted Network (PAWN), which uses the last hidden states of the LLM and positions to weight the sum of a series of features based on metrics from the next-token distribution across the sequence length. Although not zero-shot, our method allows us to cache the last hidden states and next-token distribution metrics on disk, greatly reducing the training resource requirements. PAWN shows competitive and even better performance in-distribution than the strongest baselines (fine-tuned LMs) with a fraction of their trainable parameters. Our model also generalizes better to unseen domains and source models, with smaller variability in the decision boundary across distribution shifts. It is also more robust to adversarial attacks, and if the backbone has multilingual capabilities, it presents decent generalization to languages not seen during supervised training, with LLaMA3-1B reaching a mean macro-averaged F1 score of 81.46% in cross-validation with nine languages.

摘要: 大型语言模型(LLM)的快速发展极大地增强了它们生成连贯和上下文相关文本的能力，这引起了人们对滥用人工智能生成的内容的担忧，并使检测它变得至关重要。然而，这项任务仍然具有挑战性，特别是在看不见的领域或具有不熟悉的LLM的领域。利用LLM下一个令牌分发输出提供了一种理论上有吸引力的检测方法，因为它们概括了模型对不同语料库的广泛预培训的见解。尽管有希望，但试图将这些产出付诸实施的零射击方法却取得了有限的成功。我们假设其中一个问题是，当一些令牌自然更容易或更难预测，并且应该以不同的权重进行加权时，它们使用平均值来聚合跨令牌的下一令牌分发度量。基于这一思想，我们提出了困惑注意力加权网络(PAWN)，它利用LLM的最后一个隐藏状态和位置来加权一系列特征的和，基于下一个令牌分布在整个序列长度上的度量。虽然不是零命中率，但我们的方法允许我们在磁盘上缓存最后的隐藏状态和下一个令牌分布度量，大大减少了训练资源需求。与最强的基线(微调LMS)相比，PAWN显示出具有竞争力的分布性能，甚至比它们的可训练参数的一小部分更好。我们的模型也更好地推广到看不见的域和源模型，跨分布转变的决策边界的可变性较小。LLaMA3-1B在与9种语言的交叉验证中达到了81.46%的平均宏观平均F1分数。



## **13. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

卡西姆·分歧：通过对抗性防御的扩散模型来启发性解纠缠 cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2410.23091v6) [paper-pdf](http://arxiv.org/pdf/2410.23091v6)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark). The code is available at https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff.

摘要: 尽管不断努力保护神经分类器免受对手攻击，但它们仍然很脆弱，特别是面对看不见的攻击。相比之下，人类很难被微妙的操纵所欺骗，因为我们只根据基本因素做出判断。受到这一观察的启发，我们试图用基本的标签原因因素来建模标签生成，并结合标签非原因因素来辅助数据生成。对于一个对抗性的例子，我们的目标是将扰动区分为非致因因素，并仅基于标签致因因素进行预测。具体地说，我们提出了一个偶然扩散模型(CausalDiff)，该模型使扩散模型适用于条件数据生成，并通过向一个新的偶然信息瓶颈目标学习来区分这两种类型的偶然因素。经验上，CausalDiff在各种隐形攻击上的表现明显优于最先进的防御方法，在CIFAR-10上获得了86.39%(+4.01%)的平均健壮性，在CIFAR-100上获得了56.25%(+3.13%)的健壮性，在GTSRB(德国交通标志识别基准)上实现了82.62%(+4.93%)的平均健壮性。代码可在https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff.上获得



## **14. A Volumetric Approach to Privacy of Dynamical Systems**

动态系统隐私的体积方法 eess.SY

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.02893v2) [paper-pdf](http://arxiv.org/pdf/2501.02893v2)

**Authors**: Chuanghong Weng, Ehsan Nekouei

**Abstract**: Information-theoretic metrics, such as mutual information, have been widely used to evaluate privacy leakage in dynamic systems. However, these approaches are typically limited to stochastic systems and face computational challenges. In this paper, we introduce a novel volumetric framework for analyzing privacy in systems affected by unknown but bounded noise. Our model considers a dynamic system comprising public and private states, where an observation set of the public state is released. An adversary utilizes the observed public state to infer an uncertainty set of the private state, referred to as the inference attack. We define the evolution dynamics of these inference attacks and quantify the privacy level of the private state using the volume of its uncertainty sets. For linear scalar systems, we derive an explicit formulation of the uncertainty set. For multi-dimensional linear systems, we develop an approximate computation method leveraging interval analysis. We investigate the properties of the proposed volumetric privacy measure and demonstrate that it is bounded by the information gain derived from the observation set. Furthermore, we propose an optimization approach to designing privacy filter using randomization and linear programming based on the proposed privacy measure. The effectiveness of the optimal privacy filter design is evaluated through a production-inventory case study, illustrating its robustness against the inference attack.

摘要: 信息论度量，如互信息，已被广泛用于评估动态系统中的隐私泄漏。然而，这些方法通常局限于随机系统，并面临计算挑战。在本文中，我们介绍了一种新的体积框架，用于分析受未知但有界噪声影响的系统的隐私。我们的模型考虑了一个由公共状态和私有状态组成的动态系统，其中发布了公共状态的观测集。敌手利用观察到的公共状态来推断私有状态的不确定性集合，称为推理攻击。我们定义了这些推理攻击的演化动态，并利用其不确定性集的体积来量化私有状态的隐私级别。对于线性标量系统，我们导出了不确定集的显式表达式。对于多维线性系统，我们提出了一种利用区间分析的近似计算方法。我们研究了所提出的体积隐私度量的性质，并证明了它受来自观测集的信息增益的限制。在此基础上，提出了一种基于随机化和线性规划的隐私过滤器优化设计方法。通过一个生产-库存案例的研究，评估了最优隐私过滤器设计的有效性，说明了其对推理攻击的稳健性。



## **15. Echomix: a Strong Anonymity System with Messaging**

Echomix：一个强大的消息传递匿名系统 cs.CR

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.02933v2) [paper-pdf](http://arxiv.org/pdf/2501.02933v2)

**Authors**: Ewa J Infeld, David Stainton, Leif Ryge, Threebit Hacker

**Abstract**: Echomix is a practical mix network framework and a suite of associated protocols providing strong metadata privacy against realistic modern adversaries. It is distinguished from other anonymity systems by a resistance to traffic analysis by global adversaries, compromised contacts and network infrastructure, quantum decryption algorithms, and statistical and confirmation attacks typical for multi-client messaging setting. It is implemented as Katzenpost, a robust software project, and used in multiple deployed systems, and features relatively low latency and bandwidth overhead.   The contributions of this paper are: (1) Improvements on leading mix network designs, supported by rigorous analysis. These include solutions to crucial vulnerabilities to traffic analysis, malicious servers and active attacks. (2) A cryptographic group messaging protocol with strong metadata protection guarantees and reliability. (3) Hybrid post-quantum nested packet encryption.

摘要: Echomix是一个实用的混合网络框架和一套相关协议，可针对现实的现代对手提供强大的元数据隐私。它与其他匿名系统的区别在于，它能够抵抗全球对手的流量分析、受损害的联系人和网络基础设施、量子解密算法以及多客户端消息传递设置中典型的统计和确认攻击。它作为KatzenPost实施，这是一个强大的软件项目，用于多个部署的系统，并且具有相对较低的延迟和带宽负担。   本文的贡献是：（1）在严格分析的支持下，对领先的混合网络设计进行了改进。其中包括针对流量分析、恶意服务器和主动攻击的关键漏洞的解决方案。(2)具有强大元数据保护保证和可靠性的加密群组消息协议。(3)混合后量子嵌套数据包加密。



## **16. Graph Neural Backdoor: Fundamentals, Methodologies, Applications, and Future Directions**

图形神经后门：基础知识、方法论、应用和未来方向 cs.LG

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2406.10573v2) [paper-pdf](http://arxiv.org/pdf/2406.10573v2)

**Authors**: Xiao Yang, Gaolei Li, Jianhua Li

**Abstract**: Graph Neural Networks (GNNs) have significantly advanced various downstream graph-relevant tasks, encompassing recommender systems, molecular structure prediction, social media analysis, etc. Despite the boosts of GNN, recent research has empirically demonstrated its potential vulnerability to backdoor attacks, wherein adversaries employ triggers to poison input samples, inducing GNN to adversary-premeditated malicious outputs. This is typically due to the controlled training process, or the deployment of untrusted models, such as delegating model training to third-party service, leveraging external training sets, and employing pre-trained models from online sources. Although there's an ongoing increase in research on GNN backdoors, comprehensive investigation into this field is lacking. To bridge this gap, we propose the first survey dedicated to GNN backdoors. We begin by outlining the fundamental definition of GNN, followed by the detailed summarization and categorization of current GNN backdoor attacks and defenses based on their technical characteristics and application scenarios. Subsequently, the analysis of the applicability and use cases of GNN backdoors is undertaken. Finally, the exploration of potential research directions of GNN backdoors is presented. This survey aims to explore the principles of graph backdoors, provide insights to defenders, and promote future security research.

摘要: 图神经网络(GNN)已经显著推进了各种下游与图相关的任务，包括推荐系统、分子结构预测、社交媒体分析等。尽管GNN得到了提升，但最近的研究经验表明，它对后门攻击具有潜在的脆弱性，即攻击者使用触发器来毒化输入样本，诱导GNN进行攻击者预谋的恶意输出。这通常是由于受控的培训过程或不受信任的模型的部署，例如将模型培训委托给第三方服务、利用外部培训集以及使用来自在线来源的预先培训的模型。尽管对GNN后门的研究在不断增加，但对这一领域的全面调查还很缺乏。为了弥补这一差距，我们建议对GNN后门进行第一次调查。我们首先概述了GNN的基本定义，然后根据其技术特征和应用场景对当前GNN后门攻击和防御进行了详细的总结和分类。随后，对GNN后门的适用性和使用案例进行了分析。最后，对GNN后门的潜在研究方向进行了展望。这项调查旨在探索图形后门的原理，为防御者提供见解，并促进未来的安全研究。



## **17. Unraveling Responsiveness of Chained BFT Consensus with Network Delay**

解开具有网络延迟的连锁BFT共识的响应性 cs.DC

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.03695v1) [paper-pdf](http://arxiv.org/pdf/2501.03695v1)

**Authors**: Yining Tang, Qihang Luo, Runchao Han, Jianyu Niu, Chen Feng, Yinqian Zhang

**Abstract**: With the advancement of blockchain technology, chained Byzantine Fault Tolerant (BFT) protocols have been increasingly adopted in practical systems, making their performance a crucial aspect of the study. In this paper, we introduce a unified framework utilizing Markov Decision Processes (MDP) to model and assess the performance of three prominent chained BFT protocols. Our framework effectively captures complex adversarial behaviors, focusing on two key performance metrics: chain growth and commitment rate. We implement the optimal attack strategies obtained from MDP analysis on an existing evaluation platform for chained BFT protocols and conduct extensive experiments under various settings to validate our theoretical results. Through rigorous theoretical analysis and thorough practical experiments, we provide an in-depth evaluation of chained BFT protocols under diverse attack scenarios, uncovering optimal attack strategies. Contrary to conventional belief, our findings reveal that while responsiveness can enhance performance, it is not universally beneficial across all scenarios. This work not only deepens our understanding of chained BFT protocols, but also offers valuable insights and analytical tools that can inform the design of more robust and efficient protocols.

摘要: 随着区块链技术的进步，链式拜占庭容错(BFT)协议越来越多地被应用到实际系统中，其性能成为研究的一个关键方面。本文介绍了一个利用马尔可夫决策过程(MDP)对三种主要的链式BFT协议进行建模和性能评估的统一框架。我们的框架有效地捕获了复杂的对抗性行为，重点关注两个关键的性能指标：链增长和承诺率。我们在已有的链式BFT协议评估平台上实现了从MDP分析得到的最优攻击策略，并在不同的环境下进行了大量的实验来验证我们的理论结果。通过严格的理论分析和深入的实践实验，我们对链式BFT协议在不同攻击场景下的性能进行了深入的评估，发现了最优的攻击策略。与传统的看法相反，我们的研究结果表明，尽管响应能力可以提高绩效，但并不是在所有情况下都是有益的。这项工作不仅加深了我们对链式BFT协议的理解，而且提供了有价值的见解和分析工具，可以为设计更健壮和更高效的协议提供信息。



## **18. Transferable Adversarial Examples with Bayes Approach**

使用Bayes方法的可转移对抗示例 cs.LG

Accepted in AsiaCCS'25

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2208.06538v2) [paper-pdf](http://arxiv.org/pdf/2208.06538v2)

**Authors**: Mingyuan Fan, Cen Chen, Wenmeng Zhou, Yinggui Wang

**Abstract**: The vulnerability of deep neural networks (DNNs) to black-box adversarial attacks is one of the most heated topics in trustworthy AI. In such attacks, the attackers operate without any insider knowledge of the model, making the cross-model transferability of adversarial examples critical. Despite the potential for adversarial examples to be effective across various models, it has been observed that adversarial examples that are specifically crafted for a specific model often exhibit poor transferability. In this paper, we explore the transferability of adversarial examples via the lens of Bayesian approach. Specifically, we leverage Bayesian approach to probe the transferability and then study what constitutes a transferability-promoting prior. Following this, we design two concrete transferability-promoting priors, along with an adaptive dynamic weighting strategy for instances sampled from these priors. Employing these techniques, we present BayAtk. Extensive experiments illustrate the significant effectiveness of BayAtk in crafting more transferable adversarial examples against both undefended and defended black-box models compared to existing state-of-the-art attacks.

摘要: 深度神经网络(DNN)对黑盒攻击的脆弱性是可信人工智能领域最热门的研究课题之一。在这种攻击中，攻击者在没有任何模型内部知识的情况下操作，这使得对抗性例子的跨模型可转移性至关重要。尽管对抗性例子有可能在各种模型中有效，但已经观察到，专门为特定模型制作的对抗性例子往往表现出较差的可转移性。在这篇文章中，我们通过贝叶斯方法的镜头来探讨对抗性例子的可转移性。具体地说，我们利用贝叶斯方法来探索可转让性，然后研究什么构成可转移性促进优先。在此基础上，我们设计了两个具体的可转移性提升先验，并针对从这些先验中抽取的实例设计了一种自适应的动态加权策略。利用这些技术，我们介绍了BayAtk。广泛的实验表明，与现有的最先进的攻击相比，BayAtk在针对无防御和有防御的黑盒模型创建更可转移的对手示例方面具有显著的有效性。



## **19. PhishAgent: A Robust Multimodal Agent for Phishing Webpage Detection**

PhishAgent：一种用于网络钓鱼网页检测的鲁棒多模式代理 cs.CR

Accepted at AAAI 2025

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2408.10738v2) [paper-pdf](http://arxiv.org/pdf/2408.10738v2)

**Authors**: Tri Cao, Chengyu Huang, Yuexin Li, Huilin Wang, Amy He, Nay Oo, Bryan Hooi

**Abstract**: Phishing attacks are a major threat to online security, exploiting user vulnerabilities to steal sensitive information. Various methods have been developed to counteract phishing, each with varying levels of accuracy, but they also face notable limitations. In this study, we introduce PhishAgent, a multimodal agent that combines a wide range of tools, integrating both online and offline knowledge bases with Multimodal Large Language Models (MLLMs). This combination leads to broader brand coverage, which enhances brand recognition and recall. Furthermore, we propose a multimodal information retrieval framework designed to extract the relevant top k items from offline knowledge bases, using available information from a webpage, including logos and HTML. Our empirical results, based on three real-world datasets, demonstrate that the proposed framework significantly enhances detection accuracy and reduces both false positives and false negatives, while maintaining model efficiency. Additionally, PhishAgent shows strong resilience against various types of adversarial attacks.

摘要: 网络钓鱼攻击是在线安全的主要威胁，利用用户漏洞窃取敏感信息。已经开发了各种方法来对抗网络钓鱼，每一种方法的精确度都不同，但它们也面临着显著的局限性。在本研究中，我们介绍了PhishAgent，一个结合了广泛工具的多通道代理，将线上和线下知识库与多通道大语言模型(MLLMS)相结合。这一组合导致了更广泛的品牌覆盖，从而提高了品牌认知度和召回率。此外，我们还提出了一个多通道信息检索框架，该框架利用网页中的可用信息，包括标识和超文本标记语言，从离线知识库中提取相关的前k个条目。基于三个真实数据集的实验结果表明，该框架在保持模型效率的同时，显著提高了检测准确率，减少了误报和漏报。此外，PhishAgent对各种类型的对抗性攻击表现出很强的韧性。



## **20. ChatBug: A Common Vulnerability of Aligned LLMs Induced by Chat Templates**

ChatBug：聊天模板引发的对齐LLM的常见漏洞 cs.CR

This paper is accepted to AAAI 2025

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2406.12935v2) [paper-pdf](http://arxiv.org/pdf/2406.12935v2)

**Authors**: Fengqing Jiang, Zhangchen Xu, Luyao Niu, Bill Yuchen Lin, Radha Poovendran

**Abstract**: Large language models (LLMs) are expected to follow instructions from users and engage in conversations. Techniques to enhance LLMs' instruction-following capabilities typically fine-tune them using data structured according to a predefined chat template. Although chat templates are shown to be effective in optimizing LLM performance, their impact on safety alignment of LLMs has been less understood, which is crucial for deploying LLMs safely at scale.   In this paper, we investigate how chat templates affect safety alignment of LLMs. We identify a common vulnerability, named ChatBug, that is introduced by chat templates. Our key insight to identify ChatBug is that the chat templates provide a rigid format that need to be followed by LLMs, but not by users. Hence, a malicious user may not necessarily follow the chat template when prompting LLMs. Instead, malicious users could leverage their knowledge of the chat template and accordingly craft their prompts to bypass safety alignments of LLMs. We develop two attacks to exploit the ChatBug vulnerability. We demonstrate that a malicious user can exploit the ChatBug vulnerability of eight state-of-the-art (SOTA) LLMs and effectively elicit unintended responses from these models. Moreover, we show that ChatBug can be exploited by existing jailbreak attacks to enhance their attack success rates. We investigate potential countermeasures to ChatBug. Our results show that while adversarial training effectively mitigates the ChatBug vulnerability, the victim model incurs significant performance degradation. These results highlight the trade-off between safety alignment and helpfulness. Developing new methods for instruction tuning to balance this trade-off is an open and critical direction for future research

摘要: 大型语言模型(LLM)应该遵循用户的指示并参与对话。增强LLMS的指令遵循能力的技术通常使用根据预定义的聊天模板构造的数据对其进行微调。尽管聊天模板被证明在优化LLM性能方面是有效的，但人们对它们对LLM安全调整的影响知之甚少，这对于安全地大规模部署LLMS至关重要。在本文中，我们研究了聊天模板如何影响LLMS的安全对齐。我们发现了聊天模板引入的一个名为ChatBug的常见漏洞。我们识别ChatBug的关键洞察力是，聊天模板提供了一种严格的格式，需要LLMS遵循，而不是用户。因此，恶意用户在提示LLMS时可能不一定遵循聊天模板。相反，恶意用户可以利用他们对聊天模板的了解，并相应地精心编制他们的提示，以绕过LLMS的安全对齐。我们开发了两个攻击来利用ChatBug漏洞。我们演示了恶意用户可以利用8个最先进的(SOTA)LLM的ChatBug漏洞，并有效地从这些模型中引发意外响应。此外，我们发现ChatBug可以被现有的越狱攻击所利用，以提高他们的攻击成功率。我们调查了针对ChatBug的潜在对策。我们的结果表明，虽然对抗性训练有效地缓解了ChatBug漏洞，但受害者模型导致了显著的性能下降。这些结果突显了安全性调整和帮助之间的权衡。开发新的教学调整方法来平衡这种权衡是未来研究的一个开放和关键的方向



## **21. Countering Backdoor Attacks in Image Recognition: A Survey and Evaluation of Mitigation Strategies**

对抗图像识别中的后门攻击：缓解策略的调查和评估 cs.CR

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2411.11200v2) [paper-pdf](http://arxiv.org/pdf/2411.11200v2)

**Authors**: Kealan Dunnett, Reza Arablouei, Dimity Miller, Volkan Dedeoglu, Raja Jurdak

**Abstract**: The widespread adoption of deep learning across various industries has introduced substantial challenges, particularly in terms of model explainability and security. The inherent complexity of deep learning models, while contributing to their effectiveness, also renders them susceptible to adversarial attacks. Among these, backdoor attacks are especially concerning, as they involve surreptitiously embedding specific triggers within training data, causing the model to exhibit aberrant behavior when presented with input containing the triggers. Such attacks often exploit vulnerabilities in outsourced processes, compromising model integrity without affecting performance on clean (trigger-free) input data. In this paper, we present a comprehensive review of existing mitigation strategies designed to counter backdoor attacks in image recognition. We provide an in-depth analysis of the theoretical foundations, practical efficacy, and limitations of these approaches. In addition, we conduct an extensive benchmarking of sixteen state-of-the-art approaches against eight distinct backdoor attacks, utilizing three datasets, four model architectures, and three poisoning ratios. Our results, derived from 122,236 individual experiments, indicate that while many approaches provide some level of protection, their performance can vary considerably. Furthermore, when compared to two seminal approaches, most newer approaches do not demonstrate substantial improvements in overall performance or consistency across diverse settings. Drawing from these findings, we propose potential directions for developing more effective and generalizable defensive mechanisms in the future.

摘要: 深度学习在各个行业的广泛采用带来了巨大的挑战，特别是在模型的可解释性和安全性方面。深度学习模型固有的复杂性，虽然有助于它们的有效性，但也使它们容易受到对手的攻击。其中，后门攻击尤其令人担忧，因为它们涉及在训练数据中秘密嵌入特定触发器，导致在输入包含触发器的输入时导致模型表现出异常行为。此类攻击通常利用外包流程中的漏洞，在不影响干净(无触发器)输入数据的性能的情况下损害模型完整性。在这篇文章中，我们提出了一个全面的审查现有的缓解策略，旨在对抗后门攻击的图像识别。我们对这些方法的理论基础、实践有效性和局限性进行了深入分析。此外，我们利用三个数据集、四个模型体系结构和三个投毒率，对针对八种不同后门攻击的16种最先进方法进行了广泛的基准测试。我们的结果来自122,236个单独的实验，表明虽然许多方法提供了一定程度的保护，但它们的性能可能会有很大的差异。此外，与两种开创性的方法相比，大多数较新的方法在总体性能或跨不同环境的一致性方面没有显示出实质性的改进。根据这些发现，我们提出了未来发展更有效和更具普遍性的防御机制的潜在方向。



## **22. Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting**

时间序列预测大型语言模型中的对抗漏洞 cs.LG

11 pages, 5 figures

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2412.08099v2) [paper-pdf](http://arxiv.org/pdf/2412.08099v2)

**Authors**: Fuqiang Liu, Sicong Jiang, Luis Miranda-Moreno, Seongjin Choi, Lijun Sun

**Abstract**: Large Language Models (LLMs) have recently demonstrated significant potential in the field of time series forecasting, offering impressive capabilities in handling complex temporal data. However, their robustness and reliability in real-world applications remain under-explored, particularly concerning their susceptibility to adversarial attacks. In this paper, we introduce a targeted adversarial attack framework for LLM-based time series forecasting. By employing both gradient-free and black-box optimization methods, we generate minimal yet highly effective perturbations that significantly degrade the forecasting accuracy across multiple datasets and LLM architectures. Our experiments, which include models like TimeGPT and LLM-Time with GPT-3.5, GPT-4, LLaMa, and Mistral, show that adversarial attacks lead to much more severe performance degradation than random noise, and demonstrate the broad effectiveness of our attacks across different LLMs. The results underscore the critical vulnerabilities of LLMs in time series forecasting, highlighting the need for robust defense mechanisms to ensure their reliable deployment in practical applications.

摘要: 大型语言模型最近在时间序列预测领域显示出巨大的潜力，在处理复杂的时间数据方面提供了令人印象深刻的能力。然而，它们在实际应用中的健壮性和可靠性仍然没有得到充分的研究，特别是关于它们对对手攻击的敏感性。本文提出了一种基于LLM的时间序列预测的对抗性攻击框架。通过使用无梯度和黑盒优化方法，我们产生了最小但高效的扰动，这些扰动显著降低了跨多个数据集和LLM体系结构的预测精度。我们的实验，包括使用GPT-3.5、GPT-4、LLAMA和Mistral的TimeGPT和LLM-Time模型，表明对抗性攻击导致的性能降级比随机噪声严重得多，并证明了我们的攻击在不同LLM上的广泛有效性。这些结果强调了低层管理在时间序列预测中的关键弱点，强调了需要强大的防御机制来确保其在实际应用中的可靠部署。



## **23. When Should Selfish Miners Double-Spend?**

自私的矿工何时应该加倍花钱？ cs.CR

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.03227v1) [paper-pdf](http://arxiv.org/pdf/2501.03227v1)

**Authors**: Mustafa Doger, Sennur Ulukus

**Abstract**: Although, both double-spending and selfish-mining attacks have been extensively studied since the ``Bitcoin'' whitepaper of Nakamoto and the ``majority is not enough'' paper of Eyal and Sirer, there has been no rigorous stochastic analysis of an attack that combines the two, except for the complicated MDP models. In this paper, we first combine stubborn and selfish mining attacks, i.e., construct a strategy where the attacker acts stubborn until its private branch reaches a certain length and then switches to act selfish. We provide the optimal stubbornness for each parameter regime. Next, we provide the maximum stubbornness that is still more profitable than honest mining and argue a connection between the level of stubbornness and the $k$-confirmation rule. We show that, at each attack cycle, if the level of stubbornness is higher than $k$, there is a risk of double-spending which comes at no-cost to the adversary. The result can be seen as a guide for picking $k$ in the $k$-confirmation rule in a blockchain design. At each cycle, for a given stubbornness level, we rigorously formulate how great the risk of double-spending is. We provide the minimum double-spend value needed for an attack to be profitable in the regimes where the scheme is less profitable than honest mining. We further modify the attack in the stubborn regime in order to conceal the attack and increase the double-spending probability. Finally, we evaluate the results and provide the optimal and the maximum stubbornness levels for each parameter regime as well as the revenue. As a case study, with Bitcoin's $k=6$ block confirmation rule, we evaluate the revenue and double-spending risk of the attacks for each pool parameter.

摘要: 尽管自从Nakamoto的《比特币》白皮书和EYAL和Sirer的《多数是不够的》白皮书以来，重复支出攻击和自私挖掘攻击都得到了广泛的研究，但除了复杂的MDP模型外，还没有对结合这两者的攻击进行严格的随机分析。在本文中，我们首先将顽固挖掘攻击和自私挖掘攻击相结合，即构造了一种策略，即攻击者在其私有分支达到一定长度时顽固行事，然后切换为自私行为。我们给出了每个参数区域的最优顽固性。接下来，我们提供了比诚实挖掘更有利可图的最大顽固，并论证了顽固程度与$k$确认规则之间的联系。我们证明，在每个攻击周期，如果顽固程度高于$k$，则存在重复支出的风险，这对对手来说是免费的。这一结果可以被视为区块链设计中在$k$-确认规则中挑选$k$的指南。在每个周期，对于给定的固执水平，我们都会严格地阐述重复支出的风险有多大。我们提供了攻击在利润低于诚实开采的政权中盈利所需的最低双重支出价值。我们进一步修改了顽固政权中的攻击，以隐藏攻击，增加重复支出的概率。最后，我们对结果进行了评估，并给出了每个参数机制的最优和最大顽固性水平以及收益。作为一个案例，在比特币的$k=6$块确认规则下，我们对每个池参数的攻击的收入和重复支出风险进行了评估。



## **24. The Robustness of Spiking Neural Networks in Federated Learning with Compression Against Non-omniscient Byzantine Attacks**

尖峰神经网络在压缩联邦学习中针对非无所不知的拜占庭攻击的鲁棒性 cs.CR

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.03306v1) [paper-pdf](http://arxiv.org/pdf/2501.03306v1)

**Authors**: Manh V. Nguyen, Liang Zhao, Bobin Deng, Shaoen Wu

**Abstract**: Spiking Neural Networks (SNNs), which offer exceptional energy efficiency for inference, and Federated Learning (FL), which offers privacy-preserving distributed training, is a rising area of interest that highly beneficial towards Internet of Things (IoT) devices. Despite this, research that tackles Byzantine attacks and bandwidth limitation in FL-SNNs, both poses significant threats on model convergence and training times, still remains largely unexplored. Going beyond proposing a solution for both of these problems, in this work we highlight the dual benefits of FL-SNNs, against non-omniscient Byzantine adversaries (ones that restrict attackers access to local clients datasets), and greater communication efficiency, over FL-ANNs. Specifically, we discovered that a simple integration of Top-\k{appa} sparsification into the FL apparatus can help leverage the advantages of the SNN models in both greatly reducing bandwidth usage and significantly boosting the robustness of FL training against non-omniscient Byzantine adversaries. Most notably, we saw a massive improvement of roughly 40% accuracy gain in FL-SNNs training under the lethal MinMax attack

摘要: 尖峰神经网络(SNN)为推理提供了出色的能量效率，联邦学习(FL)提供隐私保护的分布式训练，是物联网(IoT)设备高度有益的新兴兴趣领域。尽管如此，解决FL-SNN中的拜占庭攻击和带宽限制的研究仍然在很大程度上仍未被探索，这两个问题都对模型收敛和训练时间构成了重大威胁。除了为这两个问题提出解决方案外，在这项工作中，我们还强调了FL-SNN在对抗非无所不知的拜占庭对手(限制攻击者访问本地客户端数据集)和比FL-ANN更高的通信效率方面的双重好处。具体地说，我们发现，将Top-k{appa}稀疏算法简单地集成到FL设备中，可以帮助利用SNN模型的优势，既可以极大地减少带宽占用，又可以显著提高FL训练对非全知拜占庭对手的健壮性。最值得注意的是，我们看到在致命的MinMax攻击下，FL-SNN训练的准确率提高了大约40%



## **25. Leader Rotation Is Not Enough: Scrutinizing Leadership Democracy of Chained BFT Consensus**

领导人轮换还不够：审视BFT共识的领导民主 cs.CR

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.02970v1) [paper-pdf](http://arxiv.org/pdf/2501.02970v1)

**Authors**: Yining Tang, Runchao Han, Jianyu Niu, Chen Feng, Yinqian Zhang

**Abstract**: With the growing popularity of blockchains, modern chained BFT protocols combining chaining and leader rotation to obtain better efficiency and leadership democracy have received increasing interest. Although the efficiency provisions of chained BFT protocols have been thoroughly analyzed, the leadership democracy has received little attention in prior work. In this paper, we scrutinize the leadership democracy of four representative chained BFT protocols, especially under attack. To this end, we propose a unified framework with two evaluation metrics, i.e., chain quality and censorship resilience, and quantitatively analyze chosen protocols through the Markov Decision Process (MDP). With this framework, we further examine the impact of two key components, i.e., voting pattern and leader rotation on leadership democracy. Our results indicate that leader rotation is not enough to provide the leadership democracy guarantee; an adversary could utilize the design, e.g., voting pattern, to deteriorate the leadership democracy significantly. Based on the analysis results, we propose customized countermeasures for three evaluated protocols to improve their leadership democracy with only slight protocol overhead and no change of consensus rules. We also discuss future directions toward building more democratic chained BFT protocols.

摘要: 随着区块链的日益普及，现代链式BFT协议结合了链接和领导者轮换以获得更好的效率和领导民主，受到了越来越多的关注。虽然链式BFT协议的效率条款已经被彻底分析，但在以前的工作中，领导层民主几乎没有受到关注。在这篇文章中，我们仔细检查了四个有代表性的链式BFT协议的领导民主，特别是在攻击下。为此，我们提出了一个具有两个评价指标的统一框架，即链质量和审查韧性，并通过马尔可夫决策过程(MDP)对所选协议进行了定量分析。在这个框架下，我们进一步考察了投票模式和领导轮换这两个关键因素对领导民主的影响。研究结果表明，领导轮换不足以为领导民主提供保障；对手可以利用投票模式等设计来显著恶化领导民主。基于分析结果，我们对三个被评估的协议提出了定制的对策，以提高它们的领导民主，而协议开销很小，共识规则不变。我们还讨论了构建更民主的链式BFT协议的未来方向。



## **26. Seeing the Whole in the Parts in Self-Supervised Representation Learning**

自我监督的表象学习中的整体 cs.LG

20 pages

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.02860v1) [paper-pdf](http://arxiv.org/pdf/2501.02860v1)

**Authors**: Arthur Aubret, Céline Teulière, Jochen Triesch

**Abstract**: Recent successes in self-supervised learning (SSL) model spatial co-occurrences of visual features either by masking portions of an image or by aggressively cropping it. Here, we propose a new way to model spatial co-occurrences by aligning local representations (before pooling) with a global image representation. We present CO-SSL, a family of instance discrimination methods and show that it outperforms previous methods on several datasets, including ImageNet-1K where it achieves 71.5% of Top-1 accuracy with 100 pre-training epochs. CO-SSL is also more robust to noise corruption, internal corruption, small adversarial attacks, and large training crop sizes. Our analysis further indicates that CO-SSL learns highly redundant local representations, which offers an explanation for its robustness. Overall, our work suggests that aligning local and global representations may be a powerful principle of unsupervised category learning.

摘要: 自我监督学习（SSL）最近取得的成功通过掩蔽图像的部分或积极裁剪图像来建模视觉特征的空间共现。在这里，我们提出了一种通过将局部表示（池化之前）与全局图像表示对齐来建模空间共现的新方法。我们介绍了CO-SSL，这是一系列实例区分方法，并表明它在多个数据集上优于以前的方法，包括ImageNet-1 K，它在100个预训练时期内实现了Top-1的71.5%准确率。CO-SSL对噪音腐败、内部腐败、小型对抗攻击和大型训练作物规模也更稳健。我们的分析进一步表明，CO-SSL学习高度冗余的本地表示，这为其稳健性提供了解释。总体而言，我们的工作表明，将局部和全局表示对齐可能是无监督类别学习的一个强大原则。



## **27. MBTSAD: Mitigating Backdoors in Language Models Based on Token Splitting and Attention Distillation**

MBTSAD：缓解基于令牌分裂和注意力蒸馏的语言模型中的后门 cs.CR

Accepted by ICTAI 2024

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.02754v1) [paper-pdf](http://arxiv.org/pdf/2501.02754v1)

**Authors**: Yidong Ding, Jiafei Niu, Ping Yi

**Abstract**: In recent years, attention-based models have excelled across various domains but remain vulnerable to backdoor attacks, often from downloading or fine-tuning on poisoned datasets. Many current methods to mitigate backdoors in NLP models rely on the pre-trained (unfine-tuned) weights, but these methods fail in scenarios where the pre-trained weights are not available. In this work, we propose MBTSAD, which can mitigate backdoors in the language model by utilizing only a small subset of clean data and does not require pre-trained weights. Specifically, MBTSAD retrains the backdoored model on a dataset generated by token splitting. Then MBTSAD leverages attention distillation, the retrained model is the teacher model, and the original backdoored model is the student model. Experimental results demonstrate that MBTSAD achieves comparable backdoor mitigation performance as the methods based on pre-trained weights while maintaining the performance on clean data. MBTSAD does not rely on pre-trained weights, enhancing its utility in scenarios where pre-trained weights are inaccessible. In addition, we simplify the min-max problem of adversarial training and visualize text representations to discover that the token splitting method in MBTSAD's first step generates Out-of-Distribution (OOD) data, leading the model to learn more generalized features and eliminate backdoor patterns.

摘要: 近年来，基于注意力的模型在各个领域都表现出色，但仍然容易受到后门攻击，通常是通过下载或对有毒数据集进行微调。当前许多用于缓解NLP模型中后门的方法依赖于预先训练(未微调)的权重，但这些方法在预先训练的权重不可用的情况下失败。在这项工作中，我们提出了MBTSAD，它可以通过只利用一小部分干净的数据来减少语言模型中的后门，并且不需要预先训练的权重。具体地说，MBTSAD在令牌拆分生成的数据集上重新训练回溯模型。然后MBTSAD利用注意力蒸馏，再训练的模型是教师模型，原始的回溯模型是学生模型。实验结果表明，MBTSAD在保持在干净数据上的性能的同时，获得了与基于预训练权重的方法相当的后门抑制性能。MBTSAD不依赖于预先训练的权重，在无法访问预先训练的权重的情况下增强了它的实用性。此外，我们简化了对抗性训练的最小-最大问题，并将文本表示可视化，发现MBTSAD第一步中的令牌拆分方法产生了超出分布(OOD)的数据，从而使模型学习到更普遍的特征并消除了后门模式。



## **28. Persistence of Backdoor-based Watermarks for Neural Networks: A Comprehensive Evaluation**

基于后门的神经网络水印的持久性：综合评估 cs.LG

Preprint. Under Review

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.02704v1) [paper-pdf](http://arxiv.org/pdf/2501.02704v1)

**Authors**: Anh Tu Ngo, Chuan Song Heng, Nandish Chattopadhyay, Anupam Chattopadhyay

**Abstract**: Deep Neural Networks (DNNs) have gained considerable traction in recent years due to the unparalleled results they gathered. However, the cost behind training such sophisticated models is resource intensive, resulting in many to consider DNNs to be intellectual property (IP) to model owners. In this era of cloud computing, high-performance DNNs are often deployed all over the internet so that people can access them publicly. As such, DNN watermarking schemes, especially backdoor-based watermarks, have been actively developed in recent years to preserve proprietary rights. Nonetheless, there lies much uncertainty on the robustness of existing backdoor watermark schemes, towards both adversarial attacks and unintended means such as fine-tuning neural network models. One reason for this is that no complete guarantee of robustness can be assured in the context of backdoor-based watermark. In this paper, we extensively evaluate the persistence of recent backdoor-based watermarks within neural networks in the scenario of fine-tuning, we propose/develop a novel data-driven idea to restore watermark after fine-tuning without exposing the trigger set. Our empirical results show that by solely introducing training data after fine-tuning, the watermark can be restored if model parameters do not shift dramatically during fine-tuning. Depending on the types of trigger samples used, trigger accuracy can be reinstated to up to 100%. Our study further explores how the restoration process works using loss landscape visualization, as well as the idea of introducing training data in fine-tuning stage to alleviate watermark vanishing.

摘要: 近年来，由于深度神经网络(DNN)所收集的无与伦比的结果，它们获得了相当大的牵引力。然而，培训这种复杂模型背后的成本是资源密集型的，导致许多人将DNN视为模型所有者的知识产权(IP)。在这个云计算时代，高性能的DNN经常被部署在互联网上，以便人们可以公开访问它们。因此，DNN水印方案，特别是基于后门的水印方案，近年来得到了积极的发展，以保护所有权。尽管如此，现有的后门水印方案对敌意攻击和诸如微调神经网络模型等意外手段的稳健性存在很大的不确定性。这样做的一个原因是，在基于后门的水印的上下文中，不能完全保证稳健性。在本文中，我们广泛评估了最近在神经网络中基于后门的水印在微调情况下的持久性，我们提出/发展了一种新的数据驱动的思想，在不暴露触发集的情况下，在微调后恢复水印。实验结果表明，在模型参数在微调过程中不发生剧烈变化的情况下，只需引入微调后的训练数据，就可以恢复水印。根据使用的触发样本的类型，触发精度可以恢复到100%。我们的研究进一步探索了如何使用丢失景观可视化的恢复过程，以及在微调阶段引入训练数据来缓解水印消失的想法。



## **29. Layer-Level Self-Exposure and Patch: Affirmative Token Mitigation for Jailbreak Attack Defense**

分层自我暴露和补丁：越狱攻击防御的肯定代币缓解 cs.CR

**SubmitDate**: 2025-01-05    [abs](http://arxiv.org/abs/2501.02629v1) [paper-pdf](http://arxiv.org/pdf/2501.02629v1)

**Authors**: Yang Ouyang, Hengrui Gu, Shuhang Lin, Wenyue Hua, Jie Peng, Bhavya Kailkhura, Tianlong Chen, Kaixiong Zhou

**Abstract**: As large language models (LLMs) are increasingly deployed in diverse applications, including chatbot assistants and code generation, aligning their behavior with safety and ethical standards has become paramount. However, jailbreak attacks, which exploit vulnerabilities to elicit unintended or harmful outputs, threaten LLMs' safety significantly. In this paper, we introduce Layer-AdvPatcher, a novel methodology designed to defend against jailbreak attacks by utilizing an unlearning strategy to patch specific layers within LLMs through self-augmented datasets. Our insight is that certain layer(s), tend to produce affirmative tokens when faced with harmful prompts. By identifying these layers and adversarially exposing them to generate more harmful data, one can understand their inherent and diverse vulnerabilities to attacks. With these exposures, we then "unlearn" these issues, reducing the impact of affirmative tokens and hence minimizing jailbreak risks while keeping the model's responses to safe queries intact. We conduct extensive experiments on two models, four benchmark datasets, and multiple state-of-the-art jailbreak benchmarks to demonstrate the efficacy of our approach. Results indicate that our framework reduces the harmfulness and attack success rate of jailbreak attacks without compromising utility for benign queries compared to recent defense methods.

摘要: 随着大型语言模型(LLM)越来越多地部署在各种应用中，包括聊天机器人助手和代码生成，使它们的行为符合安全和道德标准变得至关重要。然而，越狱攻击利用漏洞来引发意外或有害的输出，严重威胁到LLMS的安全。在本文中，我们介绍了Layer-AdvPatcher，这是一种新的方法，旨在通过一种遗忘策略来通过自增强数据集修补LLMS中的特定层来防御越狱攻击。我们的洞察是，某些层面(S)，在面对有害的提示时，往往会产生肯定的表征。通过识别这些层并恶意暴露它们以生成更多有害数据，人们可以了解它们固有的和不同的攻击漏洞。有了这些暴露，我们就可以“忘掉”这些问题，减少肯定令牌的影响，从而最大限度地减少越狱风险，同时保持模型对安全查询的响应完好无损。我们在两个模型、四个基准数据集和多个最先进的越狱基准上进行了广泛的实验，以展示我们方法的有效性。结果表明，与现有的防御方法相比，该框架降低了越狱攻击的危害性和攻击成功率，而不影响良性查询的有效性。



## **30. Unveiling the Threat of Fraud Gangs to Graph Neural Networks: Multi-Target Graph Injection Attacks Against GNN-Based Fraud Detectors**

揭露欺诈团伙对图神经网络的威胁：针对基于GNN的欺诈检测器的多目标图注入攻击 cs.LG

19 pages, 5 figures, 12 tables, The 39th AAAI Conference on  Artificial Intelligence (AAAI 2025)

**SubmitDate**: 2025-01-05    [abs](http://arxiv.org/abs/2412.18370v2) [paper-pdf](http://arxiv.org/pdf/2412.18370v2)

**Authors**: Jinhyeok Choi, Heehyeon Kim, Joyce Jiyoung Whang

**Abstract**: Graph neural networks (GNNs) have emerged as an effective tool for fraud detection, identifying fraudulent users, and uncovering malicious behaviors. However, attacks against GNN-based fraud detectors and their risks have rarely been studied, thereby leaving potential threats unaddressed. Recent findings suggest that frauds are increasingly organized as gangs or groups. In this work, we design attack scenarios where fraud gangs aim to make their fraud nodes misclassified as benign by camouflaging their illicit activities in collusion. Based on these scenarios, we study adversarial attacks against GNN-based fraud detectors by simulating attacks of fraud gangs in three real-world fraud cases: spam reviews, fake news, and medical insurance frauds. We define these attacks as multi-target graph injection attacks and propose MonTi, a transformer-based Multi-target one-Time graph injection attack model. MonTi simultaneously generates attributes and edges of all attack nodes with a transformer encoder, capturing interdependencies between attributes and edges more effectively than most existing graph injection attack methods that generate these elements sequentially. Additionally, MonTi adaptively allocates the degree budget for each attack node to explore diverse injection structures involving target, candidate, and attack nodes, unlike existing methods that fix the degree budget across all attack nodes. Experiments show that MonTi outperforms the state-of-the-art graph injection attack methods on five real-world graphs.

摘要: 图神经网络(GNN)已经成为检测欺诈、识别欺诈用户和揭露恶意行为的有效工具。然而，对基于GNN的欺诈探测器的攻击及其风险很少被研究，从而使潜在的威胁得不到解决。最近的发现表明，诈骗越来越多地被组织成帮派或团体。在这项工作中，我们设计了攻击场景，其中欺诈团伙的目标是通过伪装他们在串通中的非法活动来使他们的欺诈节点错误地被归类为良性的。基于这些场景，我们通过模拟三个真实世界的欺诈案例：垃圾邮件评论、假新闻和医疗保险欺诈，研究了针对基于GNN的欺诈检测器的对抗性攻击。我们将这些攻击定义为多目标图注入攻击，并提出了一种基于变压器的多目标一次性图注入攻击模型MONTI。Monti使用转换器编码器同时生成所有攻击节点的属性和边，比大多数现有的按顺序生成这些元素的图注入攻击方法更有效地捕获属性和边之间的相互依赖关系。此外，与现有方法固定所有攻击节点的度预算不同，Monti自适应地为每个攻击节点分配度预算，以探索涉及目标、候选和攻击节点的不同注入结构。实验表明，Monti在五个真实图上的性能优于目前最先进的图注入攻击方法。



## **31. GCP: Guarded Collaborative Perception with Spatial-Temporal Aware Malicious Agent Detection**

GCP：具有时空感知恶意代理检测的保护协作感知 cs.CV

15 pages

**SubmitDate**: 2025-01-05    [abs](http://arxiv.org/abs/2501.02450v1) [paper-pdf](http://arxiv.org/pdf/2501.02450v1)

**Authors**: Yihang Tao, Senkang Hu, Yue Hu, Haonan An, Hangcheng Cao, Yuguang Fang

**Abstract**: Collaborative perception significantly enhances autonomous driving safety by extending each vehicle's perception range through message sharing among connected and autonomous vehicles. Unfortunately, it is also vulnerable to adversarial message attacks from malicious agents, resulting in severe performance degradation. While existing defenses employ hypothesis-and-verification frameworks to detect malicious agents based on single-shot outliers, they overlook temporal message correlations, which can be circumvented by subtle yet harmful perturbations in model input and output spaces. This paper reveals a novel blind area confusion (BAC) attack that compromises existing single-shot outlier-based detection methods. As a countermeasure, we propose GCP, a Guarded Collaborative Perception framework based on spatial-temporal aware malicious agent detection, which maintains single-shot spatial consistency through a confidence-scaled spatial concordance loss, while simultaneously examining temporal anomalies by reconstructing historical bird's eye view motion flows in low-confidence regions. We also employ a joint spatial-temporal Benjamini-Hochberg test to synthesize dual-domain anomaly results for reliable malicious agent detection. Extensive experiments demonstrate GCP's superior performance under diverse attack scenarios, achieving up to 34.69% improvements in AP@0.5 compared to the state-of-the-art CP defense strategies under BAC attacks, while maintaining consistent 5-8% improvements under other typical attacks. Code will be released at https://github.com/CP-Security/GCP.git.

摘要: 协同感知通过互联和自动驾驶车辆之间的信息共享扩大了每辆车的感知范围，从而显著提高了自动驾驶的安全性。不幸的是，它也容易受到恶意代理的敌意消息攻击，导致性能严重下降。虽然现有的防御措施使用假设和验证框架来检测基于单发离群值的恶意代理，但它们忽略了时间消息相关性，这可以通过模型输入和输出空间中微妙但有害的扰动来规避。提出了一种新的盲区混淆(BAC)攻击，该攻击破坏了现有的基于单击点的孤立点检测方法。作为对策，我们提出了一种基于时空感知恶意代理检测的警戒式协同感知框架GCP，它通过置信度尺度的空间一致性损失来保持单次拍摄的空间一致性，同时通过在低置信度区域重建历史鸟瞰运动流来检测时间异常。我们还使用时空联合Benjamini-Hochberg测试来合成双域异常结果，以实现可靠的恶意代理检测。大量的实验表明，GCP在不同的攻击场景下具有卓越的性能，在BAC攻击下，与最先进的CP防御策略相比，AP@0.5的性能提高高达34.69%，而在其他典型攻击下，GCP的性能保持一致的5%-8%。代码将在https://github.com/CP-Security/GCP.git.上发布



## **32. A Practical Examination of AI-Generated Text Detectors for Large Language Models**

大型语言模型的人工智能生成文本检测器的实践检验 cs.CL

8 pages. Submitted to NAACL

**SubmitDate**: 2025-01-04    [abs](http://arxiv.org/abs/2412.05139v2) [paper-pdf](http://arxiv.org/pdf/2412.05139v2)

**Authors**: Brian Tufts, Xuandong Zhao, Lei Li

**Abstract**: The proliferation of large language models has raised growing concerns about their misuse, particularly in cases where AI-generated text is falsely attributed to human authors. Machine-generated content detectors claim to effectively identify such text under various conditions and from any language model. This paper critically evaluates these claims by assessing several popular detectors (RADAR, Wild, T5Sentinel, Fast-DetectGPT, GPTID, LogRank, Binoculars) on a range of domains, datasets, and models that these detectors have not previously encountered. We employ various prompting strategies to simulate adversarial attacks, demonstrating that even moderate efforts can significantly evade detection. We emphasize the importance of the true positive rate at a specific false positive rate (TPR@FPR) metric and demonstrate that these detectors perform poorly in certain settings, with TPR@.01 as low as 0%. Our findings suggest that both trained and zero-shot detectors struggle to maintain high sensitivity while achieving a reasonable true positive rate.

摘要: 大型语言模型的激增引发了人们对它们滥用的日益担忧，特别是在人工智能生成的文本被错误地归因于人类作者的情况下。机器生成的内容检测器声称可以在各种条件下从任何语言模型有效地识别此类文本。本文通过评估几种流行的探测器(雷达、Wild、T5Sentinel、Fast-DetectGPT、GPTID、logrank、双筒望远镜)，对这些声称进行了批判性的评估，这些探测器以前从未遇到过。我们使用各种提示策略来模拟对抗性攻击，表明即使是适度的攻击也可以显著地躲避检测。我们强调了在特定的假阳性率(TPR@fPR)度量下的真阳性率的重要性，并证明了这些检测器在某些设置下表现很差，TPR@.01低至0%。我们的发现表明，训练有素的探测器和零射探测器都很难在保持高灵敏度的同时获得合理的真阳性率。



## **33. GNSS/GPS Spoofing and Jamming Identification Using Machine Learning and Deep Learning**

使用机器学习和深度学习进行GNSS/GPS欺骗和干扰识别 cs.CR

**SubmitDate**: 2025-01-04    [abs](http://arxiv.org/abs/2501.02352v1) [paper-pdf](http://arxiv.org/pdf/2501.02352v1)

**Authors**: Ali Ghanbarzade, Hossein Soleimani

**Abstract**: The increasing reliance on Global Navigation Satellite Systems (GNSS), particularly the Global Positioning System (GPS), underscores the urgent need to safeguard these technologies against malicious threats such as spoofing and jamming. As the backbone for positioning, navigation, and timing (PNT) across various applications including transportation, telecommunications, and emergency services GNSS is vulnerable to deliberate interference that poses significant risks. Spoofing attacks, which involve transmitting counterfeit GNSS signals to mislead receivers into calculating incorrect positions, can result in serious consequences, from navigational errors in civilian aviation to security breaches in military operations. Furthermore, the lack of inherent security measures within GNSS systems makes them attractive targets for adversaries. While GNSS/GPS jamming and spoofing systems consist of numerous components, the ability to distinguish authentic signals from malicious ones is essential for maintaining system integrity. Recent advancements in machine learning and deep learning provide promising avenues for enhancing detection and mitigation strategies against these threats. This paper addresses both spoofing and jamming by tackling real-world challenges through machine learning, deep learning, and computer vision techniques. Through extensive experiments on two real-world datasets related to spoofing and jamming detection using advanced algorithms, we achieved state of the art results. In the GNSS/GPS jamming detection task, we attained approximately 99% accuracy, improving performance by around 5% compared to previous studies. Additionally, we addressed a challenging tasks related to spoofing detection, yielding results that underscore the potential of machine learning and deep learning in this domain.

摘要: 对全球导航卫星系统(GNSS)，特别是全球定位系统(GPS)的日益依赖，突显出迫切需要保护这些技术免受欺骗和干扰等恶意威胁。作为定位、导航和授时(PNT)的骨干，GNSS跨越各种应用，包括交通、电信和紧急服务，容易受到人为干扰，从而带来重大风险。欺骗性攻击包括发送伪造的GNSS信号以误导接收器计算错误的位置，可能会导致严重的后果，从民用航空的导航错误到军事行动的安全漏洞。此外，全球导航卫星系统系统缺乏固有的安全措施，因此对对手很有吸引力。虽然GNSS/GPS干扰和欺骗系统由许多组件组成，但区分真实信号和恶意信号的能力对于维护系统完整性至关重要。机器学习和深度学习的最新进展为加强对这些威胁的检测和缓解战略提供了有希望的途径。本文通过机器学习、深度学习和计算机视觉技术解决现实世界中的挑战，从而解决欺骗和干扰问题。通过使用高级算法在两个与欺骗和干扰检测相关的真实数据集上进行广泛的实验，我们获得了最先进的结果。在GNSS/GPS干扰检测任务中，我们达到了大约99%的准确率，与以前的研究相比，性能提高了约5%。此外，我们解决了与欺骗检测相关的具有挑战性的任务，产生的结果突显了机器学习和深度学习在该领域的潜力。



## **34. Distillation-Enhanced Physical Adversarial Attacks**

蒸馏增强的物理对抗攻击 cs.CV

7 pages, 5 figures

**SubmitDate**: 2025-01-04    [abs](http://arxiv.org/abs/2501.02232v1) [paper-pdf](http://arxiv.org/pdf/2501.02232v1)

**Authors**: Wei Liu, Yonglin Wu, Chaoqun Li, Zhuodong Liu, Huanqian Yan

**Abstract**: The study of physical adversarial patches is crucial for identifying vulnerabilities in AI-based recognition systems and developing more robust deep learning models. While recent research has focused on improving patch stealthiness for greater practical applicability, achieving an effective balance between stealth and attack performance remains a significant challenge. To address this issue, we propose a novel physical adversarial attack method that leverages knowledge distillation. Specifically, we first define a stealthy color space tailored to the target environment to ensure smooth blending. Then, we optimize an adversarial patch in an unconstrained color space, which serves as the 'teacher' patch. Finally, we use an adversarial knowledge distillation module to transfer the teacher patch's knowledge to the 'student' patch, guiding the optimization of the stealthy patch. Experimental results show that our approach improves attack performance by 20%, while maintaining stealth, highlighting its practical value.

摘要: 物理对抗性补丁的研究对于识别基于人工智能的识别系统中的漏洞和开发更健壮的深度学习模型至关重要。虽然最近的研究集中在提高补丁的隐蔽性以获得更大的实用适用性，但在隐形和攻击性能之间实现有效平衡仍然是一个巨大的挑战。针对这一问题，我们提出了一种新的利用知识提取的物理对抗性攻击方法。具体地说，我们首先定义了一种适合目标环境的隐身颜色空间，以确保平滑混合。然后，我们在不受约束的颜色空间中优化一块对抗性的补丁，作为“教师”补丁。最后，利用对抗性知识提炼模块，将教师补丁的知识转移到学生补丁，指导隐身补丁的优化。实验结果表明，该方法在保持隐蔽性的情况下，攻击性能提高了20%，具有一定的实用价值。



## **35. 2-in-1 Accelerator: Enabling Random Precision Switch for Winning Both Adversarial Robustness and Efficiency**

二合一加速器：实现随机精确切换，赢得对抗性稳健性和效率 cs.LG

Accepted at MICRO 2021

**SubmitDate**: 2025-01-04    [abs](http://arxiv.org/abs/2109.05223v3) [paper-pdf](http://arxiv.org/pdf/2109.05223v3)

**Authors**: Yonggan Fu, Yang Zhao, Qixuan Yu, Chaojian Li, Yingyan Celine Lin

**Abstract**: The recent breakthroughs of deep neural networks (DNNs) and the advent of billions of Internet of Things (IoT) devices have excited an explosive demand for intelligent IoT devices equipped with domain-specific DNN accelerators. However, the deployment of DNN accelerator enabled intelligent functionality into real-world IoT devices still remains particularly challenging. First, powerful DNNs often come at prohibitive complexities, whereas IoT devices often suffer from stringent resource constraints. Second, while DNNs are vulnerable to adversarial attacks especially on IoT devices exposed to complex real-world environments, many IoT applications require strict security. Existing DNN accelerators mostly tackle only one of the two aforementioned challenges (i.e., efficiency or adversarial robustness) while neglecting or even sacrificing the other. To this end, we propose a 2-in-1 Accelerator, an integrated algorithm-accelerator co-design framework aiming at winning both the adversarial robustness and efficiency of DNN accelerators. Specifically, we first propose a Random Precision Switch (RPS) algorithm that can effectively defend DNNs against adversarial attacks by enabling random DNN quantization as an in-situ model switch. Furthermore, we propose a new precision-scalable accelerator featuring (1) a new precision-scalable MAC unit architecture which spatially tiles the temporal MAC units to boost both the achievable efficiency and flexibility and (2) a systematically optimized dataflow that is searched by our generic accelerator optimizer. Extensive experiments and ablation studies validate that our 2-in-1 Accelerator can not only aggressively boost both the adversarial robustness and efficiency of DNN accelerators under various attacks, but also naturally support instantaneous robustness-efficiency trade-offs adapting to varied resources without the necessity of DNN retraining.

摘要: 最近深度神经网络(DNN)的突破和数十亿物联网(IoT)设备的出现，刺激了对配备领域特定DNN加速器的智能物联网设备的爆炸性需求。然而，将DNN加速器支持的智能功能部署到现实世界的物联网设备中仍然具有特别大的挑战性。首先，强大的DNN通常具有令人望而却步的复杂性，而物联网设备往往受到严格的资源限制。其次，虽然DNN很容易受到敌意攻击，尤其是在暴露在复杂现实环境中的物联网设备上，但许多物联网应用程序需要严格的安全性。现有的DNN加速器大多只解决上述两个挑战中的一个(即效率或对手的健壮性)，而忽略甚至牺牲另一个挑战。为此，我们提出了一种2-in-1加速器，这是一种集成的算法-加速器协同设计框架，旨在同时获得DNN加速器的对抗健壮性和效率。具体地说，我们首先提出了一种随机精确切换(RPS)算法，通过启用随机DNN量化作为原位模型切换，可以有效地防御DNN的敌意攻击。此外，我们提出了一种新的精度可扩展的加速器，它具有(1)新的精度可扩展的MAC单元结构，该结构在空间上平铺时间MAC单元以提高可实现的效率和灵活性；(2)由我们的通用加速器优化器搜索的系统优化的数据流。广泛的实验和烧蚀研究证明，我们的2合1加速器不仅可以积极提高DNN加速器在各种攻击下的对抗健壮性和效率，而且自然地支持适应不同资源的瞬时健壮性和效率之间的权衡，而不需要对DNN进行再培训。



## **36. Drawing Robust Scratch Tickets: Subnetworks with Inborn Robustness Are Found within Randomly Initialized Networks**

绘制稳健的刮刮票：在随机初始化网络中发现具有天生稳健性的子网络 cs.LG

Accepted at NeurIPS 2021

**SubmitDate**: 2025-01-04    [abs](http://arxiv.org/abs/2110.14068v4) [paper-pdf](http://arxiv.org/pdf/2110.14068v4)

**Authors**: Yonggan Fu, Qixuan Yu, Yang Zhang, Shang Wu, Xu Ouyang, David Cox, Yingyan Celine Lin

**Abstract**: Deep Neural Networks (DNNs) are known to be vulnerable to adversarial attacks, i.e., an imperceptible perturbation to the input can mislead DNNs trained on clean images into making erroneous predictions. To tackle this, adversarial training is currently the most effective defense method, by augmenting the training set with adversarial samples generated on the fly. Interestingly, we discover for the first time that there exist subnetworks with inborn robustness, matching or surpassing the robust accuracy of the adversarially trained networks with comparable model sizes, within randomly initialized networks without any model training, indicating that adversarial training on model weights is not indispensable towards adversarial robustness. We name such subnetworks Robust Scratch Tickets (RSTs), which are also by nature efficient. Distinct from the popular lottery ticket hypothesis, neither the original dense networks nor the identified RSTs need to be trained. To validate and understand this fascinating finding, we further conduct extensive experiments to study the existence and properties of RSTs under different models, datasets, sparsity patterns, and attacks, drawing insights regarding the relationship between DNNs' robustness and their initialization/overparameterization. Furthermore, we identify the poor adversarial transferability between RSTs of different sparsity ratios drawn from the same randomly initialized dense network, and propose a Random RST Switch (R2S) technique, which randomly switches between different RSTs, as a novel defense method built on top of RSTs. We believe our findings about RSTs have opened up a new perspective to study model robustness and extend the lottery ticket hypothesis.

摘要: 深度神经网络(DNN)容易受到敌意攻击，即输入的不可察觉的扰动会误导训练在干净图像上的DNN做出错误的预测。为了解决这个问题，对抗性训练是目前最有效的防御方法，通过使用在运行中生成的对抗性样本来扩大训练集。有趣的是，我们首次发现，在没有任何模型训练的随机初始化网络中，存在具有天生健壮性的子网络，与具有类似模型大小的对抗性训练网络的鲁棒精度相当或超过，这表明对抗性模型权重的训练对于对抗性健壮性来说并不是必不可少的。我们将这样的子网络命名为健壮的暂存票(RST)，它本质上也是有效的。与流行的彩票假设不同，原始的密集网络和识别的RST都不需要训练。为了验证和理解这一有趣的发现，我们进一步进行了大量的实验，研究了不同模型、数据集、稀疏模式和攻击下RST的存在和性质，得出了DNN的健壮性与其初始化/过参数化之间的关系。此外，我们还发现了来自同一随机初始化密集网络的不同稀疏度的RST之间的对抗性较差，并提出了一种随机RST切换(R2S)技术，作为建立在RST之上的一种新的防御方法。我们相信，我们关于RST的发现为研究模型的稳健性和扩展彩票假说开辟了一个新的视角。



## **37. Patch-Fool: Are Vision Transformers Always Robust Against Adversarial Perturbations?**

补丁傻瓜：视觉变形金刚在对抗干扰方面总是稳健吗？ cs.CV

Accepted at ICLR 2022

**SubmitDate**: 2025-01-04    [abs](http://arxiv.org/abs/2203.08392v3) [paper-pdf](http://arxiv.org/pdf/2203.08392v3)

**Authors**: Yonggan Fu, Shunyao Zhang, Shang Wu, Cheng Wan, Yingyan Celine Lin

**Abstract**: Vision transformers (ViTs) have recently set off a new wave in neural architecture design thanks to their record-breaking performance in various vision tasks. In parallel, to fulfill the goal of deploying ViTs into real-world vision applications, their robustness against potential malicious attacks has gained increasing attention. In particular, recent works show that ViTs are more robust against adversarial attacks as compared with convolutional neural networks (CNNs), and conjecture that this is because ViTs focus more on capturing global interactions among different input/feature patches, leading to their improved robustness to local perturbations imposed by adversarial attacks. In this work, we ask an intriguing question: "Under what kinds of perturbations do ViTs become more vulnerable learners compared to CNNs?" Driven by this question, we first conduct a comprehensive experiment regarding the robustness of both ViTs and CNNs under various existing adversarial attacks to understand the underlying reason favoring their robustness. Based on the drawn insights, we then propose a dedicated attack framework, dubbed Patch-Fool, that fools the self-attention mechanism by attacking its basic component (i.e., a single patch) with a series of attention-aware optimization techniques. Interestingly, our Patch-Fool framework shows for the first time that ViTs are not necessarily more robust than CNNs against adversarial perturbations. In particular, we find that ViTs are more vulnerable learners compared with CNNs against our Patch-Fool attack which is consistent across extensive experiments, and the observations from Sparse/Mild Patch-Fool, two variants of Patch-Fool, indicate an intriguing insight that the perturbation density and strength on each patch seem to be the key factors that influence the robustness ranking between ViTs and CNNs.

摘要: 视觉转换器(VITS)最近掀起了神经结构设计的新浪潮，这要归功于它们在各种视觉任务中的创纪录表现。与此同时，为了实现将VITS部署到现实世界视觉应用中的目标，它们对潜在恶意攻击的健壮性得到了越来越多的关注。特别是，最近的研究表明，与卷积神经网络(CNN)相比，VITS对对抗攻击具有更强的鲁棒性，推测这是因为VITS更注重捕捉不同输入/特征块之间的全局交互，从而提高了它们对敌对攻击造成的局部扰动的鲁棒性。在这项工作中，我们提出了一个耐人寻味的问题：“在什么样的扰动下，VITS比CNN更容易成为学习者？”在这个问题的驱动下，我们首先对VITS和CNN在各种现有的对抗性攻击下的健壮性进行了全面的实验，以了解有利于其健壮性的潜在原因。在此基础上，我们提出了一个专门的攻击框架，称为Patch-Fool，它通过使用一系列注意力感知优化技术来攻击自我注意机制的基本组成部分(即单个补丁)来愚弄自我注意机制。有趣的是，我们的Patch-Fool框架首次表明，VITS在对抗对手扰动时并不一定比CNN更健壮。特别是，我们发现VITS比CNN更容易学习，这在广泛的实验中是一致的，并且来自Patch-Fool的两个变种稀疏/温和Patch-Fool的观察表明，每个补丁上的扰动密度和强度似乎是影响VITS和CNN之间健壮性排名的关键因素。



## **38. NeRFool: Uncovering the Vulnerability of Generalizable Neural Radiance Fields against Adversarial Perturbations**

NeRFool：揭示可推广神经辐射场对对抗性扰动的脆弱性 cs.CV

Accepted by ICML 2023

**SubmitDate**: 2025-01-04    [abs](http://arxiv.org/abs/2306.06359v2) [paper-pdf](http://arxiv.org/pdf/2306.06359v2)

**Authors**: Yonggan Fu, Ye Yuan, Souvik Kundu, Shang Wu, Shunyao Zhang, Yingyan Celine Lin

**Abstract**: Generalizable Neural Radiance Fields (GNeRF) are one of the most promising real-world solutions for novel view synthesis, thanks to their cross-scene generalization capability and thus the possibility of instant rendering on new scenes. While adversarial robustness is essential for real-world applications, little study has been devoted to understanding its implication on GNeRF. We hypothesize that because GNeRF is implemented by conditioning on the source views from new scenes, which are often acquired from the Internet or third-party providers, there are potential new security concerns regarding its real-world applications. Meanwhile, existing understanding and solutions for neural networks' adversarial robustness may not be applicable to GNeRF, due to its 3D nature and uniquely diverse operations. To this end, we present NeRFool, which to the best of our knowledge is the first work that sets out to understand the adversarial robustness of GNeRF. Specifically, NeRFool unveils the vulnerability patterns and important insights regarding GNeRF's adversarial robustness. Built upon the above insights gained from NeRFool, we further develop NeRFool+, which integrates two techniques capable of effectively attacking GNeRF across a wide range of target views, and provide guidelines for defending against our proposed attacks. We believe that our NeRFool/NeRFool+ lays the initial foundation for future innovations in developing robust real-world GNeRF solutions. Our codes are available at: https://github.com/GATECH-EIC/NeRFool.

摘要: 可概括神经辐射场(GNeRF)是现实世界中最有前途的新型视点合成解决方案之一，这要归功于它们的跨场景泛化能力，从而可以在新场景上进行即时渲染。虽然对抗的稳健性对于现实世界的应用是必不可少的，但很少有研究致力于了解其对GNeRF的影响。我们假设，由于GNeRF是通过对来自新场景的源视图进行条件处理来实现的，这些场景通常是从互联网或第三方提供商获得的，因此其现实世界的应用程序存在潜在的新的安全问题。同时，由于GNeRF的3D性质和独特的多样性操作，现有对神经网络对抗性稳健性的理解和解决方案可能不适用于GNeRF。为此，我们提出了NeRFool，据我们所知，这是第一个开始了解GNeRF的对手健壮性的工作。具体地说，NeRFool揭示了关于GNeRF的对手健壮性的漏洞模式和重要见解。基于以上从NeRFool获得的见解，我们进一步开发了NeRFool+，它集成了两种能够在广泛的目标视图中有效攻击GNeRF的技术，并为防御我们提出的攻击提供了指导方针。我们相信，我们的NeRFool/NeRFool+为未来在开发强大的现实世界GNeRF解决方案方面的创新奠定了初步基础。我们的代码请访问：https://github.com/GATECH-EIC/NeRFool.



## **39. Exploring Secure Machine Learning Through Payload Injection and FGSM Attacks on ResNet-50**

通过ResNet-50上的有效负载注入和FGSM攻击探索安全机器学习 cs.CR

**SubmitDate**: 2025-01-04    [abs](http://arxiv.org/abs/2501.02147v1) [paper-pdf](http://arxiv.org/pdf/2501.02147v1)

**Authors**: Umesh Yadav, Suman Niraula, Gaurav Kumar Gupta, Bicky Yadav

**Abstract**: This paper investigates the resilience of a ResNet-50 image classification model under two prominent security threats: Fast Gradient Sign Method (FGSM) adversarial attacks and malicious payload injection. Initially, the model attains a 53.33% accuracy on clean images. When subjected to FGSM perturbations, its overall accuracy remains unchanged; however, the model's confidence in incorrect predictions notably increases. Concurrently, a payload injection scheme is successfully executed in 93.33% of the tested samples, revealing how stealthy attacks can manipulate model predictions without degrading visual quality. These findings underscore the vulnerability of even high-performing neural networks and highlight the urgency of developing more robust defense mechanisms for security-critical applications.

摘要: 本文研究了ResNet-50图像分类模型在两种主要安全威胁下的弹性：快速梯度符号法（FGSM）对抗性攻击和恶意有效负载注入。最初，该模型在干净图像上获得了53.33%的准确率。当受到FGSM扰动时，其总体准确性保持不变;然而，模型对错误预测的信心显着增加。同时，有效载荷注入方案在93.33%的测试样本中成功执行，揭示了隐形攻击如何在不降低视觉质量的情况下操纵模型预测。这些发现凸显了即使是高性能神经网络的脆弱性，并凸显了为安全关键应用程序开发更强大的防御机制的紧迫性。



## **40. AVTrustBench: Assessing and Enhancing Reliability and Robustness in Audio-Visual LLMs**

AVTrustBench：评估和增强视听LLM的可靠性和稳健性 cs.CV

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.02135v1) [paper-pdf](http://arxiv.org/pdf/2501.02135v1)

**Authors**: Sanjoy Chowdhury, Sayan Nag, Subhrajyoti Dasgupta, Yaoting Wang, Mohamed Elhoseiny, Ruohan Gao, Dinesh Manocha

**Abstract**: With the rapid advancement of Multi-modal Large Language Models (MLLMs), several diagnostic benchmarks have recently been developed to assess these models' multi-modal reasoning proficiency. However, these benchmarks are restricted to assessing primarily the visual aspect and do not examine the holistic audio-visual (AV) understanding. Moreover, currently, there are no benchmarks that investigate the capabilities of AVLLMs to calibrate their responses when presented with perturbed inputs. To this end, we introduce Audio-Visual Trustworthiness assessment Benchmark (AVTrustBench), comprising 600K samples spanning over 9 meticulously crafted tasks, evaluating the capabilities of AVLLMs across three distinct dimensions: Adversarial attack, Compositional reasoning, and Modality-specific dependency. Using our benchmark we extensively evaluate 13 state-of-the-art AVLLMs. The findings reveal that the majority of existing models fall significantly short of achieving human-like comprehension, offering valuable insights for future research directions. To alleviate the limitations in the existing approaches, we further propose a robust, model-agnostic calibrated audio-visual preference optimization based training strategy CAVPref, obtaining a gain up to 30.19% across all 9 tasks. We will publicly release our code and benchmark to facilitate future research in this direction.

摘要: 随着多通道大型语言模型(MLLMS)的迅速发展，最近出现了几个用于评估这些模型的多通道推理能力的诊断基准。然而，这些基准仅限于主要评估视觉方面，而不检查整体视听(AV)理解。此外，目前还没有基准来调查AVLLMS在收到扰动输入时校准其响应的能力。为此，我们引入了视听可信度评估基准(AVTrustB边)，该基准包括60万个样本，跨越9个精心制作的任务，从三个不同的维度评估AVLLMS的能力：对抗攻击、成分推理和特定通道依赖。使用我们的基准，我们广泛评估了13个最先进的AVLLM。研究结果表明，现有的大多数模型都明显不能实现类似人类的理解，为未来的研究方向提供了有价值的见解。为了缓解现有方法的局限性，我们进一步提出了一种稳健的、与模型无关的、基于校准视听偏好优化的训练策略CAVPref，在所有9个任务中都获得了高达30.19%的收益。我们将公开发布我们的代码和基准，以促进未来在这一方向的研究。



## **41. Towards Robust and Accurate Stability Estimation of Local Surrogate Models in Text-based Explainable AI**

基于文本的可解释人工智能中局部代理模型的稳健而准确的稳定性估计 cs.LG

12 pages, 1 figure, 4 tables. arXiv admin note: substantial text  overlap with arXiv:2406.15839. substantial text overlap with arXiv:2501.01516

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.02042v1) [paper-pdf](http://arxiv.org/pdf/2501.02042v1)

**Authors**: Christopher Burger, Charles Walter, Thai Le, Lingwei Chen

**Abstract**: Recent work has investigated the concept of adversarial attacks on explainable AI (XAI) in the NLP domain with a focus on examining the vulnerability of local surrogate methods such as Lime to adversarial perturbations or small changes on the input of a machine learning (ML) model. In such attacks, the generated explanation is manipulated while the meaning and structure of the original input remain similar under the ML model. Such attacks are especially alarming when XAI is used as a basis for decision making (e.g., prescribing drugs based on AI medical predictors) or for legal action (e.g., legal dispute involving AI software). Although weaknesses across many XAI methods have been shown to exist, the reasons behind why remain little explored. Central to this XAI manipulation is the similarity measure used to calculate how one explanation differs from another. A poor choice of similarity measure can lead to erroneous conclusions about the stability or adversarial robustness of an XAI method. Therefore, this work investigates a variety of similarity measures designed for text-based ranked lists referenced in related work to determine their comparative suitability for use. We find that many measures are overly sensitive, resulting in erroneous estimates of stability. We then propose a weighting scheme for text-based data that incorporates the synonymity between the features within an explanation, providing more accurate estimates of the actual weakness of XAI methods to adversarial examples.

摘要: 最近的工作研究了NLP领域中对可解释人工智能(XAI)的对抗性攻击的概念，重点研究了本地代理方法(如Lime)对对抗性扰动或机器学习(ML)模型输入的微小变化的脆弱性。在这种攻击中，生成的解释被操纵，而原始输入的含义和结构在ML模型下保持相似。当XAI被用作决策基础(例如，基于人工智能医学预测开出药物)或法律行动(例如，涉及人工智能软件的法律纠纷)时，此类攻击尤其令人担忧。尽管许多XAI方法的弱点已经被证明存在，但为什么背后的原因仍然很少被探索。这种XAI操作的核心是用来计算一种解释与另一种解释的不同之处的相似性度量。如果相似性度量选择不当，可能会导致关于XAI方法的稳定性或对抗稳健性的错误结论。因此，这项工作研究了为相关工作中引用的基于文本的排序列表设计的各种相似性度量，以确定它们的相对适用性。我们发现，许多指标过于敏感，导致对稳定性的错误估计。然后，我们提出了一种基于文本的数据加权方案，该方案将特征之间的同义性合并到解释中，为对抗性例子提供了对XAI方法实际弱点的更准确的估计。



## **42. Detecting and Mitigating Adversarial Attacks on Deep Learning-Based MRI Reconstruction Without Any Retraining**

无需任何重新训练即可检测和缓解基于深度学习的MRI重建的对抗攻击 cs.CV

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.01908v1) [paper-pdf](http://arxiv.org/pdf/2501.01908v1)

**Authors**: Mahdi Saberi, Chi Zhang, Mehmet Akcakaya

**Abstract**: Deep learning (DL) methods, especially those based on physics-driven DL, have become the state-of-the-art for reconstructing sub-sampled magnetic resonance imaging (MRI) data. However, studies have shown that these methods are susceptible to small adversarial input perturbations, or attacks, resulting in major distortions in the output images. Various strategies have been proposed to reduce the effects of these attacks, but they require retraining and may lower reconstruction quality for non-perturbed/clean inputs. In this work, we propose a novel approach for detecting and mitigating adversarial attacks on MRI reconstruction models without any retraining. Our detection strategy is based on the idea of cyclic measurement consistency. The output of the model is mapped to another set of MRI measurements for a different sub-sampling pattern, and this synthesized data is reconstructed with the same model. Intuitively, without an attack, the second reconstruction is expected to be consistent with the first, while with an attack, disruptions are present. Subsequently, this idea is extended to devise a novel objective function, which is minimized within a small ball around the attack input for mitigation. Experimental results show that our method substantially reduces the impact of adversarial perturbations across different datasets, attack types/strengths and PD-DL networks, and qualitatively and quantitatively outperforms conventional mitigation methods that involve retraining.

摘要: 深度学习方法，特别是基于物理驱动的深度学习方法，已经成为重建亚采样磁共振成像(MRI)数据的最新方法。然而，研究表明，这些方法容易受到小的对抗性输入扰动或攻击，导致输出图像的严重失真。已经提出了各种策略来减少这些攻击的影响，但它们需要重新培训，并且可能会降低非扰动/干净输入的重建质量。在这项工作中，我们提出了一种新的方法来检测和缓解对MRI重建模型的敌意攻击，而不需要任何重新训练。我们的检测策略是基于循环测量一致性的思想。该模型的输出被映射到用于不同子采样模式的另一组MRI测量，并且利用相同的模型重建该合成数据。直观地说，在没有攻击的情况下，第二次重建预计与第一次一致，而在攻击时，会出现中断。随后，将这一思想扩展到设计了一个新的目标函数，该目标函数在攻击输入周围的小球内最小化以用于缓解。实验结果表明，我们的方法大大降低了不同数据集、攻击类型/强度和PD-DL网络之间的对抗性扰动的影响，并且在定性和定量上都优于传统的涉及再训练的缓解方法。



## **43. Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions**

将逻辑与自身对立：通过对比问题探索模型辩护 cs.CL

Our code is publicly available at  https://github.com/UKPLab/POATE-attack

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.01872v2) [paper-pdf](http://arxiv.org/pdf/2501.01872v2)

**Authors**: Rachneet Sachdeva, Rima Hazra, Iryna Gurevych

**Abstract**: Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. In this work, we introduce POATE (Polar Opposite query generation, Adversarial Template construction, and Elaboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. To counter this, we propose Intent-Aware CoT and Reverse Thinking CoT, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. These methods enhance reasoning robustness and strengthen the model's defense against adversarial exploits.

摘要: 大型语言模型尽管与人类价值观和伦理原则广泛一致，但仍然容易受到复杂的越狱攻击，这些攻击利用了它们的推理能力。现有的安全措施经常检测到公开的恶意意图，但无法解决细微的、推理驱动的漏洞。在这项工作中，我们介绍了POATE(极地相反查询生成，对抗性模板构建和精化)，这是一种新的越狱技术，利用对比推理来引发不道德的反应。波特在语义上设计了相反的意图，并将它们与对抗性模板整合在一起，以惊人的微妙程度引导模型指向有害的输出。我们对六个不同参数大小的不同语言模型家族进行了广泛的评估，以证明攻击的健壮性，与现有方法相比，攻击成功率显著提高(~44%)。针对这一问题，我们提出了意图感知COT和逆向思维COT，它们分解查询以检测恶意意图，并反向推理以评估和拒绝有害响应。这些方法增强了推理的健壮性，增强了模型对对手攻击的防御能力。



## **44. PB-UAP: Hybrid Universal Adversarial Attack For Image Segmentation**

PB-UAP：图像分割的混合通用对抗攻击 cs.CV

Accepted by ICASSP 2025

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2412.16651v2) [paper-pdf](http://arxiv.org/pdf/2412.16651v2)

**Authors**: Yufei Song, Ziqi Zhou, Minghui Li, Xianlong Wang, Hangtao Zhang, Menghao Deng, Wei Wan, Shengshan Hu, Leo Yu Zhang

**Abstract**: With the rapid advancement of deep learning, the model robustness has become a significant research hotspot, \ie, adversarial attacks on deep neural networks. Existing works primarily focus on image classification tasks, aiming to alter the model's predicted labels. Due to the output complexity and deeper network architectures, research on adversarial examples for segmentation models is still limited, particularly for universal adversarial perturbations. In this paper, we propose a novel universal adversarial attack method designed for segmentation models, which includes dual feature separation and low-frequency scattering modules. The two modules guide the training of adversarial examples in the pixel and frequency space, respectively. Experiments demonstrate that our method achieves high attack success rates surpassing the state-of-the-art methods, and exhibits strong transferability across different models.

摘要: 随着深度学习的快速发展，模型鲁棒性已成为一个重要的研究热点，即对深度神经网络的对抗性攻击。现有的工作主要集中在图像分类任务上，旨在改变模型的预测标签。由于输出复杂性和更深层次的网络架构，对分段模型对抗性示例的研究仍然有限，特别是对于普遍对抗性扰动。本文提出了一种针对分割模型设计的新型通用对抗攻击方法，其中包括双重特征分离和低频散射模块。这两个模块分别指导像素和频率空间中对抗性示例的训练。实验表明，我们的方法比最先进的方法具有更高的攻击成功率，并且在不同模型之间表现出很强的可移植性。



## **45. Rerouting LLM Routers**

重新规划LLM路由器 cs.CR

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.01818v1) [paper-pdf](http://arxiv.org/pdf/2501.01818v1)

**Authors**: Avital Shafran, Roei Schuster, Thomas Ristenpart, Vitaly Shmatikov

**Abstract**: LLM routers aim to balance quality and cost of generation by classifying queries and routing them to a cheaper or more expensive LLM depending on their complexity. Routers represent one type of what we call LLM control planes: systems that orchestrate use of one or more LLMs. In this paper, we investigate routers' adversarial robustness.   We first define LLM control plane integrity, i.e., robustness of LLM orchestration to adversarial inputs, as a distinct problem in AI safety. Next, we demonstrate that an adversary can generate query-independent token sequences we call ``confounder gadgets'' that, when added to any query, cause LLM routers to send the query to a strong LLM.   Our quantitative evaluation shows that this attack is successful both in white-box and black-box settings against a variety of open-source and commercial routers, and that confounding queries do not affect the quality of LLM responses. Finally, we demonstrate that gadgets can be effective while maintaining low perplexity, thus perplexity-based filtering is not an effective defense. We finish by investigating alternative defenses.

摘要: LLM路由器旨在通过对查询进行分类并根据其复杂性将它们路由到更便宜或更昂贵的LLM来平衡生成质量和成本。路由器代表一种我们称为LLM控制平面的类型：协调使用一个或多个LLM的系统。本文研究了路由器的对抗健壮性。我们首先定义了LLM控制平面的完整性，即LLM编排对敌意输入的健壮性，这是人工智能安全中的一个明显问题。接下来，我们将演示敌手可以生成与查询无关的令牌序列，我们称之为‘’置乱小工具‘’，当添加到任何查询中时，会导致LLM路由器将查询发送到强大的LLM。我们的定量评估表明，该攻击在白盒和黑盒环境下都是成功的，对各种开源和商业路由器的攻击都是成功的，混淆查询不会影响LLM响应的质量。最后，我们证明了小工具可以在保持低困惑的同时有效，因此基于困惑的过滤不是有效的防御。最后，我们将研究替代防御措施。



## **46. How Toxic Can You Get? Search-based Toxicity Testing for Large Language Models**

你能得到多大的毒性？基于搜索的大型语言模型毒性测试 cs.SE

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.01741v1) [paper-pdf](http://arxiv.org/pdf/2501.01741v1)

**Authors**: Simone Corbo, Luca Bancale, Valeria De Gennaro, Livia Lestingi, Vincenzo Scotti, Matteo Camilli

**Abstract**: Language is a deep-rooted means of perpetration of stereotypes and discrimination. Large Language Models (LLMs), now a pervasive technology in our everyday lives, can cause extensive harm when prone to generating toxic responses. The standard way to address this issue is to align the LLM, which, however, dampens the issue without constituting a definitive solution. Therefore, testing LLM even after alignment efforts remains crucial for detecting any residual deviations with respect to ethical standards. We present EvoTox, an automated testing framework for LLMs' inclination to toxicity, providing a way to quantitatively assess how much LLMs can be pushed towards toxic responses even in the presence of alignment. The framework adopts an iterative evolution strategy that exploits the interplay between two LLMs, the System Under Test (SUT) and the Prompt Generator steering SUT responses toward higher toxicity. The toxicity level is assessed by an automated oracle based on an existing toxicity classifier. We conduct a quantitative and qualitative empirical evaluation using four state-of-the-art LLMs as evaluation subjects having increasing complexity (7-13 billion parameters). Our quantitative evaluation assesses the cost-effectiveness of four alternative versions of EvoTox against existing baseline methods, based on random search, curated datasets of toxic prompts, and adversarial attacks. Our qualitative assessment engages human evaluators to rate the fluency of the generated prompts and the perceived toxicity of the responses collected during the testing sessions. Results indicate that the effectiveness, in terms of detected toxicity level, is significantly higher than the selected baseline methods (effect size up to 1.0 against random search and up to 0.99 against adversarial attacks). Furthermore, EvoTox yields a limited cost overhead (from 22% to 35% on average).

摘要: 语言是制造陈规定型观念和歧视的根深蒂固的手段。大语言模型(LLM)现在是我们日常生活中的一项普遍技术，当它容易产生有毒反应时，可能会造成广泛的危害。解决这一问题的标准方法是调整LLM，然而，这会抑制问题，而不会构成最终的解决方案。因此，即使在调整工作之后进行LLM测试，对于检测与道德标准有关的任何残余偏差仍然至关重要。我们提出了EvoTox，这是一个用于LLMS毒性倾向的自动化测试框架，提供了一种方法来定量评估即使在存在对齐的情况下，LLMS也可以被推向毒性反应的程度。该框架采用了一种迭代进化策略，利用了两个LLM之间的相互作用，被测系统(SUT)和即时生成器将SUT的响应转向更高的毒性。毒性水平由基于现有毒性分类器的自动先知进行评估。我们使用四个最先进的LLM作为评估对象，进行了定量和定性的实证评估，这些评估对象的复杂性越来越高(参数为70-130亿个)。我们的定量评估基于随机搜索、有毒提示的精选数据集和对抗性攻击，相对于现有的基线方法评估了EvoTox的四个替代版本的成本效益。我们的定性评估聘请人类评估员对生成的提示的流畅性和在测试期间收集的回答的感知毒性进行评级。结果表明，就检测到的毒性水平而言，该方法的有效性显著高于所选的基线方法(对随机搜索的效果大小高达1.0，对对抗性攻击的效果高达0.99)。此外，EvoTox产生的成本管理费用有限(平均从22%到35%)。



## **47. Adaptive Meta-learning-based Adversarial Training for Robust Automatic Modulation Classification**

基于自适应元学习的对抗训练用于鲁棒自动调制分类 cs.LG

Submitted to IEEE International Conference on Communications (ICC)  2025

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.01620v1) [paper-pdf](http://arxiv.org/pdf/2501.01620v1)

**Authors**: Amirmohammad Bamdad, Ali Owfi, Fatemeh Afghah

**Abstract**: DL-based automatic modulation classification (AMC) models are highly susceptible to adversarial attacks, where even minimal input perturbations can cause severe misclassifications. While adversarially training an AMC model based on an adversarial attack significantly increases its robustness against that attack, the AMC model will still be defenseless against other adversarial attacks. The theoretically infinite possibilities for adversarial perturbations mean that an AMC model will inevitably encounter new unseen adversarial attacks if it is ever to be deployed to a real-world communication system. Moreover, the computational limitations and challenges of obtaining new data in real-time will not allow a full training process for the AMC model to adapt to the new attack when it is online. To this end, we propose a meta-learning-based adversarial training framework for AMC models that substantially enhances robustness against unseen adversarial attacks and enables fast adaptation to these attacks using just a few new training samples, if any are available. Our results demonstrate that this training framework provides superior robustness and accuracy with much less online training time than conventional adversarial training of AMC models, making it highly efficient for real-world deployment.

摘要: 基于动态链接库的自动调制分类(AMC)模型容易受到对抗性攻击，即使是最小的输入扰动也会导致严重的误分类。虽然对基于对抗性攻击的AMC模型进行对抗性训练可显著提高其对该攻击的健壮性，但AMC模型仍将对其他对抗性攻击毫无防御能力。理论上对抗性扰动的无限可能性意味着，如果AMC模型被部署到现实世界的通信系统中，它将不可避免地遇到新的看不见的对抗性攻击。此外，实时获取新数据的计算限制和挑战将不允许AMC模型在在线时进行完整的训练过程来适应新的攻击。为此，我们提出了一种基于元学习的AMC模型对抗性训练框架，该框架大大增强了对未知对抗性攻击的稳健性，并能够使用少量新的训练样本(如果有的话)快速适应这些攻击。结果表明，与传统的AMC模型对抗性训练相比，该训练框架具有更好的健壮性和准确性，在线训练时间更少，对实际部署具有很高的效率。



## **48. BLAST: A Stealthy Backdoor Leverage Attack against Cooperative Multi-Agent Deep Reinforcement Learning based Systems**

AMPS：针对基于协作多智能体深度强化学习的系统的隐形后门杠杆攻击 cs.AI

12. arXiv admin note: substantial text overlap with arXiv:2409.07775

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.01593v1) [paper-pdf](http://arxiv.org/pdf/2501.01593v1)

**Authors**: Yinbo Yu, Saihao Yan, Xueyu Yin, Jing Fang, Jiajia Liu

**Abstract**: Recent studies have shown that cooperative multi-agent deep reinforcement learning (c-MADRL) is under the threat of backdoor attacks. Once a backdoor trigger is observed, it will perform malicious actions leading to failures or malicious goals. However, existing backdoor attacks suffer from several issues, e.g., instant trigger patterns lack stealthiness, the backdoor is trained or activated by an additional network, or all agents are backdoored. To this end, in this paper, we propose a novel backdoor leverage attack against c-MADRL, BLAST, which attacks the entire multi-agent team by embedding the backdoor only in a single agent. Firstly, we introduce adversary spatiotemporal behavior patterns as the backdoor trigger rather than manual-injected fixed visual patterns or instant status and control the period to perform malicious actions. This method can guarantee the stealthiness and practicality of BLAST. Secondly, we hack the original reward function of the backdoor agent via unilateral guidance to inject BLAST, so as to achieve the \textit{leverage attack effect} that can pry open the entire multi-agent system via a single backdoor agent. We evaluate our BLAST against 3 classic c-MADRL algorithms (VDN, QMIX, and MAPPO) in 2 popular c-MADRL environments (SMAC and Pursuit), and 2 existing defense mechanisms. The experimental results demonstrate that BLAST can achieve a high attack success rate while maintaining a low clean performance variance rate.

摘要: 最近的研究表明，协作多智能体深度强化学习(c-MADRL)受到后门攻击的威胁。一旦观察到后门触发器，它将执行导致失败或恶意目标的恶意操作。然而，现有的后门攻击存在几个问题，例如，即时触发模式缺乏隐蔽性，后门被额外的网络训练或激活，或者所有代理都被后门。为此，本文提出了一种新的针对c-MADRL的后门杠杆攻击，BLAST，它通过在单个代理中嵌入后门来攻击整个多代理团队。首先，引入敌方时空行为模式作为后门触发，而不是人工注入固定的视觉模式或即时状态，并控制恶意行为的执行周期。该方法可以保证爆破的隐蔽性和实用性。其次，通过单边引导注入BLAST来破解后门代理原有的奖励功能，从而达到通过单个后门代理撬开整个多代理系统的杠杆攻击效果。我们在两个流行的c-MADRL环境(SMAC和PROCESS)和两个现有的防御机制中，针对三种经典的c-MADRL算法(VDN、QMIX和MAPPO)对我们的BLAST进行了评估。实验结果表明，BLAST可以在保持较低的干净性能变异率的同时获得较高的攻击成功率。



## **49. Familiarity-Based Open-Set Recognition Under Adversarial Attacks**

对抗性攻击下基于家族关系的开放集识别 cs.CV

Published in: Proceedings of the 6th Northern Lights Deep Learning  Conference (NLDL), PMLR 265, 2025

**SubmitDate**: 2025-01-02    [abs](http://arxiv.org/abs/2311.05006v2) [paper-pdf](http://arxiv.org/pdf/2311.05006v2)

**Authors**: Philip Enevoldsen, Christian Gundersen, Nico Lang, Serge Belongie, Christian Igel

**Abstract**: Open-set recognition (OSR), the identification of novel categories, can be a critical component when deploying classification models in real-world applications. Recent work has shown that familiarity-based scoring rules such as the Maximum Softmax Probability (MSP) or the Maximum Logit Score (MLS) are strong baselines when the closed-set accuracy is high. However, one of the potential weaknesses of familiarity-based OSR are adversarial attacks. Here, we study gradient-based adversarial attacks on familiarity scores for both types of attacks, False Familiarity and False Novelty attacks, and evaluate their effectiveness in informed and uninformed settings on TinyImageNet. Furthermore, we explore how novel and familiar samples react to adversarial attacks and formulate the adversarial reaction score as an alternative OSR scoring rule, which shows a high correlation with the MLS familiarity score.

摘要: 开放集识别（OSR）是新型类别的识别，在现实应用程序中部署分类模型时可能是一个关键组件。最近的工作表明，当闭集准确性较高时，基于熟悉度的评分规则，例如最大Softmax概率（MSP）或最大Logit得分（MLS）是强大的基线。然而，基于熟悉度的OSR的潜在弱点之一是对抗性攻击。在这里，我们研究了基于梯度的对抗攻击对两种攻击（假熟悉性和假新奇性攻击）的熟悉度分数的熟悉度分数，并在TinyImageNet上评估它们在知情和不知情环境中的有效性。此外，我们探索了新颖且熟悉的样本如何对对抗性攻击做出反应，并将对抗性反应评分制定为替代OSR评分规则，该规则与MLS熟悉度评分具有高度相关性。



## **50. Safeguarding Large Language Models in Real-time with Tunable Safety-Performance Trade-offs**

通过可调的安全性能权衡实时保护大型语言模型 cs.CL

**SubmitDate**: 2025-01-02    [abs](http://arxiv.org/abs/2501.02018v1) [paper-pdf](http://arxiv.org/pdf/2501.02018v1)

**Authors**: Joao Fonseca, Andrew Bell, Julia Stoyanovich

**Abstract**: Large Language Models (LLMs) have been shown to be susceptible to jailbreak attacks, or adversarial attacks used to illicit high risk behavior from a model. Jailbreaks have been exploited by cybercriminals and blackhat actors to cause significant harm, highlighting the critical need to safeguard widely-deployed models. Safeguarding approaches, which include fine-tuning models or having LLMs "self-reflect", may lengthen the inference time of a model, incur a computational penalty, reduce the semantic fluency of an output, and restrict ``normal'' model behavior. Importantly, these Safety-Performance Trade-offs (SPTs) remain an understudied area. In this work, we introduce a novel safeguard, called SafeNudge, that combines Controlled Text Generation with "nudging", or using text interventions to change the behavior of a model. SafeNudge triggers during text-generation while a jailbreak attack is being executed, and can reduce successful jailbreak attempts by 30% by guiding the LLM towards a safe responses. It adds minimal latency to inference and has a negligible impact on the semantic fluency of outputs. Further, we allow for tunable SPTs. SafeNudge is open-source and available through https://pypi.org/, and is compatible with models loaded with the Hugging Face "transformers" library.

摘要: 大型语言模型(LLM)已被证明容易受到越狱攻击，即用于从模型中非法进行高风险行为的对抗性攻击。越狱已被网络犯罪分子和黑帽行为者利用，造成重大危害，突显出保护广泛部署的模型的迫切需要。保护方法，包括微调模型或让LLM“自我反思”，可能会延长模型的推理时间，招致计算惩罚，降低输出的语义流畅性，并限制“正常”的模型行为。重要的是，这些安全-性能权衡(SPTS)仍然是一个研究较少的领域。在这项工作中，我们引入了一种新的安全措施，称为安全轻推，它将受控文本生成与“轻推”相结合，即使用文本干预来改变模型的行为。安全轻推在执行越狱攻击时在文本生成过程中触发，通过引导LLM进行安全响应，可以将成功的越狱尝试减少30%。它增加了最小的推理延迟，并且对输出的语义流畅性的影响可以忽略不计。此外，我们还考虑了可调SPT。SafeNdge是开源的，可以通过https://pypi.org/，获得，并且与装载了拥抱脸“变形金刚”库的模型兼容。



