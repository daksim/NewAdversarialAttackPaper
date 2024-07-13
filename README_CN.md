# Latest Adversarial Attack Papers
**update at 2024-07-13 10:45:01**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. How to beat a Bayesian adversary**

如何击败Bayesian对手 cs.LG

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.08678v1) [paper-pdf](http://arxiv.org/pdf/2407.08678v1)

**Authors**: Zihan Ding, Kexin Jin, Jonas Latz, Chenguang Liu

**Abstract**: Deep neural networks and other modern machine learning models are often susceptible to adversarial attacks. Indeed, an adversary may often be able to change a model's prediction through a small, directed perturbation of the model's input - an issue in safety-critical applications. Adversarially robust machine learning is usually based on a minmax optimisation problem that minimises the machine learning loss under maximisation-based adversarial attacks.   In this work, we study adversaries that determine their attack using a Bayesian statistical approach rather than maximisation. The resulting Bayesian adversarial robustness problem is a relaxation of the usual minmax problem. To solve this problem, we propose Abram - a continuous-time particle system that shall approximate the gradient flow corresponding to the underlying learning problem. We show that Abram approximates a McKean-Vlasov process and justify the use of Abram by giving assumptions under which the McKean-Vlasov process finds the minimiser of the Bayesian adversarial robustness problem. We discuss two ways to discretise Abram and show its suitability in benchmark adversarial deep learning experiments.

摘要: 深度神经网络和其他现代机器学习模型往往容易受到敌意攻击。事实上，对手往往能够通过对模型的输入进行小的、直接的扰动来改变模型的预测--这在安全关键的应用程序中是一个问题。对抗性稳健机器学习通常基于最小化最大优化问题，该问题在基于最大化的对抗性攻击下最小化机器学习损失。在这项工作中，我们研究了使用贝叶斯统计方法而不是最大化来确定攻击的对手。由此产生的贝叶斯对抗健壮性问题是通常的极大极小问题的松弛。为了解决这个问题，我们提出了Abram-一个连续时间粒子系统，它应该近似于对应于底层学习问题的梯度流。我们证明了Abram逼近McKean-Vlasov过程，并通过给出McKean-Vlasov过程找到贝叶斯对抗健壮性问题的最小值的假设来证明Abram的使用。我们讨论了两种离散化Abram的方法，并在基准对抗性深度学习实验中展示了它的适用性。



## **2. Large-Scale Dataset Pruning in Adversarial Training through Data Importance Extrapolation**

通过数据重要性外推进行对抗训练中的大规模数据集修剪 cs.LG

8 pages, 5 figures, 3 tables, to be published in ICML: DMLR workshop

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2406.13283v2) [paper-pdf](http://arxiv.org/pdf/2406.13283v2)

**Authors**: Björn Nieth, Thomas Altstidl, Leo Schwinn, Björn Eskofier

**Abstract**: Their vulnerability to small, imperceptible attacks limits the adoption of deep learning models to real-world systems. Adversarial training has proven to be one of the most promising strategies against these attacks, at the expense of a substantial increase in training time. With the ongoing trend of integrating large-scale synthetic data this is only expected to increase even further. Thus, the need for data-centric approaches that reduce the number of training samples while maintaining accuracy and robustness arises. While data pruning and active learning are prominent research topics in deep learning, they are as of now largely unexplored in the adversarial training literature. We address this gap and propose a new data pruning strategy based on extrapolating data importance scores from a small set of data to a larger set. In an empirical evaluation, we demonstrate that extrapolation-based pruning can efficiently reduce dataset size while maintaining robustness.

摘要: 它们对小型、不可感知的攻击的脆弱性限制了深度学习模型在现实世界系统中的采用。事实证明，对抗训练是对抗这些攻击的最有希望的策略之一，但代价是训练时间的大幅增加。随着集成大规模合成数据的持续趋势，预计这一数字只会进一步增加。因此，需要以数据为中心的方法来减少训练样本数量，同时保持准确性和稳健性。虽然数据修剪和主动学习是深度学习中的重要研究主题，但迄今为止，对抗性训练文献中基本上尚未对其进行探讨。我们解决了这一差距，并提出了一种新的数据修剪策略，该策略基于将数据重要性分数从小数据集外推到大数据集。在经验评估中，我们证明基于外推的修剪可以有效地减少数据集大小，同时保持稳健性。



## **3. DART: A Solution for Decentralized Federated Learning Model Robustness Analysis**

DART：分散联邦学习模型鲁棒性分析的解决方案 cs.DC

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.08652v1) [paper-pdf](http://arxiv.org/pdf/2407.08652v1)

**Authors**: Chao Feng, Alberto Huertas Celdrán, Jan von der Assen, Enrique Tomás Martínez Beltrán, Gérôme Bovet, Burkhard Stiller

**Abstract**: Federated Learning (FL) has emerged as a promising approach to address privacy concerns inherent in Machine Learning (ML) practices. However, conventional FL methods, particularly those following the Centralized FL (CFL) paradigm, utilize a central server for global aggregation, which exhibits limitations such as bottleneck and single point of failure. To address these issues, the Decentralized FL (DFL) paradigm has been proposed, which removes the client-server boundary and enables all participants to engage in model training and aggregation tasks. Nevertheless, as CFL, DFL remains vulnerable to adversarial attacks, notably poisoning attacks that undermine model performance. While existing research on model robustness has predominantly focused on CFL, there is a noteworthy gap in understanding the model robustness of the DFL paradigm. In this paper, a thorough review of poisoning attacks targeting the model robustness in DFL systems, as well as their corresponding countermeasures, are presented. Additionally, a solution called DART is proposed to evaluate the robustness of DFL models, which is implemented and integrated into a DFL platform. Through extensive experiments, this paper compares the behavior of CFL and DFL under diverse poisoning attacks, pinpointing key factors affecting attack spread and effectiveness within the DFL. It also evaluates the performance of different defense mechanisms and investigates whether defense mechanisms designed for CFL are compatible with DFL. The empirical results provide insights into research challenges and suggest ways to improve the robustness of DFL models for future research.

摘要: 联合学习(FL)已经成为解决机器学习(ML)实践中固有的隐私问题的一种有前途的方法。然而，传统的FL方法，特别是那些遵循集中式FL(CFL)范例的方法，使用中央服务器进行全局聚合，这表现出诸如瓶颈和单点故障等限制。为了解决这些问题，提出了分散式FL(DFL)范例，它消除了客户端-服务器的边界，使所有参与者都能够参与模型训练和聚合任务。然而，作为CFL，DFL仍然容易受到对手的攻击，特别是破坏模型性能的中毒攻击。虽然现有的关于模型稳健性的研究主要集中在CFL上，但在理解DFL范式的模型稳健性方面存在着明显的差距。本文对DFL系统中以模型稳健性为目标的中毒攻击及其相应的对策进行了深入的综述。此外，还提出了一种称为DART的解决方案来评估DFL模型的健壮性，并将其实现并集成到DFL平台中。通过大量的实验，比较了CFL和DFL在不同的中毒攻击下的行为，找出了影响攻击在DFL内传播和有效性的关键因素。评估了不同防御机制的性能，并研究了为CFL设计的防御机制是否与DFL兼容。实证结果提供了对研究挑战的洞察，并为未来的研究提出了改进DFL模型的稳健性的方法。



## **4. RAIFLE: Reconstruction Attacks on Interaction-based Federated Learning with Adversarial Data Manipulation**

RAIFLE：对具有对抗性数据操纵的基于交互的联邦学习的重建攻击 cs.CR

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2310.19163v2) [paper-pdf](http://arxiv.org/pdf/2310.19163v2)

**Authors**: Dzung Pham, Shreyas Kulkarni, Amir Houmansadr

**Abstract**: Federated learning has emerged as a promising privacy-preserving solution for machine learning domains that rely on user interactions, particularly recommender systems and online learning to rank. While there has been substantial research on the privacy of traditional federated learning, little attention has been paid to the privacy properties of these interaction-based settings. In this work, we show that users face an elevated risk of having their private interactions reconstructed by the central server when the server can control the training features of the items that users interact with. We introduce RAIFLE, a novel optimization-based attack framework where the server actively manipulates the features of the items presented to users to increase the success rate of reconstruction. Our experiments with federated recommendation and online learning-to-rank scenarios demonstrate that RAIFLE is significantly more powerful than existing reconstruction attacks like gradient inversion, achieving high performance consistently in most settings. We discuss the pros and cons of several possible countermeasures to defend against RAIFLE in the context of interaction-based federated learning. Our code is open-sourced at https://github.com/dzungvpham/raifle.

摘要: 联合学习已经成为一种很有前途的隐私保护解决方案，适用于依赖用户交互的机器学习领域，特别是推荐系统和在线学习来排名。虽然已经有大量关于传统联合学习隐私的研究，但很少有人关注这些基于交互的设置的隐私属性。在这项工作中，我们表明，当中央服务器可以控制用户交互的项目的训练特征时，用户面临由中央服务器重建他们的私人交互的风险增加。我们引入了RAIFLE，一个新的基于优化的攻击框架，服务器主动操纵呈现给用户的项目的特征，以提高重建的成功率。我们在联邦推荐和在线学习排名场景下的实验表明，RAIFLE比现有的重建攻击(如梯度反转)要强大得多，在大多数情况下都能获得一致的高性能。我们讨论了在基于交互的联邦学习背景下防御RAIFLE的几种可能对策的利弊。我们的代码在https://github.com/dzungvpham/raifle.上是开源的



## **5. NeuroIDBench: An Open-Source Benchmark Framework for the Standardization of Methodology in Brainwave-based Authentication Research**

NeuroIDBench：基于脑电波的认证研究方法标准化的开源基准框架 cs.CR

21 pages, 5 Figures, 3 tables, Submitted to the Journal of  Information Security and Applications

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2402.08656v5) [paper-pdf](http://arxiv.org/pdf/2402.08656v5)

**Authors**: Avinash Kumar Chaurasia, Matin Fallahi, Thorsten Strufe, Philipp Terhörst, Patricia Arias Cabarcos

**Abstract**: Biometric systems based on brain activity have been proposed as an alternative to passwords or to complement current authentication techniques. By leveraging the unique brainwave patterns of individuals, these systems offer the possibility of creating authentication solutions that are resistant to theft, hands-free, accessible, and potentially even revocable. However, despite the growing stream of research in this area, faster advance is hindered by reproducibility problems. Issues such as the lack of standard reporting schemes for performance results and system configuration, or the absence of common evaluation benchmarks, make comparability and proper assessment of different biometric solutions challenging. Further, barriers are erected to future work when, as so often, source code is not published open access. To bridge this gap, we introduce NeuroIDBench, a flexible open source tool to benchmark brainwave-based authentication models. It incorporates nine diverse datasets, implements a comprehensive set of pre-processing parameters and machine learning algorithms, enables testing under two common adversary models (known vs unknown attacker), and allows researchers to generate full performance reports and visualizations. We use NeuroIDBench to investigate the shallow classifiers and deep learning-based approaches proposed in the literature, and to test robustness across multiple sessions. We observe a 37.6% reduction in Equal Error Rate (EER) for unknown attacker scenarios (typically not tested in the literature), and we highlight the importance of session variability to brainwave authentication. All in all, our results demonstrate the viability and relevance of NeuroIDBench in streamlining fair comparisons of algorithms, thereby furthering the advancement of brainwave-based authentication through robust methodological practices.

摘要: 基于大脑活动的生物识别系统已经被提出作为密码的替代方案，或者是对当前身份验证技术的补充。通过利用个人独特的脑电波模式，这些系统提供了创建防盗、免提、可访问甚至可能可撤销的身份验证解决方案的可能性。然而，尽管这一领域的研究越来越多，但重复性问题阻碍了更快的进展。缺乏性能结果和系统配置的标准报告方案，或缺乏通用的评估基准等问题，使不同生物识别解决方案的可比性和适当评估具有挑战性。此外，当源代码不公开、开放获取时，就会为未来的工作设置障碍。为了弥补这一差距，我们引入了NeuroIDBch，这是一个灵活的开源工具，用于对基于脑电波的身份验证模型进行基准测试。它整合了九个不同的数据集，实现了一套全面的预处理参数和机器学习算法，可以在两个常见的对手模型(已知和未知攻击者)下进行测试，并允许研究人员生成完整的性能报告和可视化。我们使用NeuroIDB边来研究文献中提出的浅层分类器和基于深度学习的方法，并测试多个会话的健壮性。我们观察到，对于未知攻击者场景(通常未在文献中进行测试)，等错误率(EER)降低了37.6%，并强调了会话可变性对脑电波身份验证的重要性。总而言之，我们的结果证明了NeuroIDBtch在简化公平的算法比较方面的可行性和相关性，从而通过稳健的方法学实践进一步推进基于脑电波的身份验证。



## **6. Boosting Adversarial Transferability for Skeleton-based Action Recognition via Exploring the Model Posterior Space**

通过探索模型后验空间增强基于线粒体的动作识别的对抗可移植性 cs.CV

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.08572v1) [paper-pdf](http://arxiv.org/pdf/2407.08572v1)

**Authors**: Yunfeng Diao, Baiqi Wu, Ruixuan Zhang, Xun Yang, Meng Wang, He Wang

**Abstract**: Skeletal motion plays a pivotal role in human activity recognition (HAR). Recently, attack methods have been proposed to identify the universal vulnerability of skeleton-based HAR(S-HAR). However, the research of adversarial transferability on S-HAR is largely missing. More importantly, existing attacks all struggle in transfer across unknown S-HAR models. We observed that the key reason is that the loss landscape of the action recognizers is rugged and sharp. Given the established correlation in prior studies~\cite{qin2022boosting,wu2020towards} between loss landscape and adversarial transferability, we assume and empirically validate that smoothing the loss landscape could potentially improve adversarial transferability on S-HAR. This is achieved by proposing a new post-train Dual Bayesian strategy, which can effectively explore the model posterior space for a collection of surrogates without the need for re-training. Furthermore, to craft adversarial examples along the motion manifold, we incorporate the attack gradient with information of the motion dynamics in a Bayesian manner. Evaluated on benchmark datasets, e.g. HDM05 and NTU 60, the average transfer success rate can reach as high as 35.9\% and 45.5\% respectively. In comparison, current state-of-the-art skeletal attacks achieve only 3.6\% and 9.8\%. The high adversarial transferability remains consistent across various surrogate, victim, and even defense models. Through a comprehensive analysis of the results, we provide insights on what surrogates are more likely to exhibit transferability, to shed light on future research.

摘要: 骨骼运动在人类活动识别(HAR)中起着至关重要的作用。近年来，为了识别基于骨架的HAR(S-HAR)的普遍脆弱性，人们提出了攻击方法。然而，关于S-哈尔对抗性转会的研究还很少。更重要的是，现有的进攻都在挣扎着跨越未知的S-哈尔模型进行转移。我们观察到，关键原因是动作识别器的损失情况是崎岖和尖锐的。鉴于先前的研究已经建立了损失情景与对手可转移性之间的相关性，我们假设并实证平滑损失情景可以潜在地提高S-HAR上的对手可转移性。这是通过提出一种新的训练后双重贝叶斯策略来实现的，该策略可以有效地探索代理集合的模型后验空间，而不需要重新训练。此外，为了制作沿运动流形的对抗性示例，我们以贝叶斯方式将攻击梯度与运动动力学信息相结合。在HDM05和NTU 60等基准数据集上进行评估，平均传输成功率分别高达35.9%和45.5%。相比之下，目前最先进的骨架攻击只实现了3.6%和9.8%。高度的对抗性可转移性在各种代理、受害者甚至防御模型中保持一致。通过对结果的综合分析，我们对哪些替代品更有可能表现出可转移性提供了见解，为未来的研究提供了启示。



## **7. BriDe Arbitrager: Enhancing Arbitrage in Ethereum 2.0 via Bribery-enabled Delayed Block Production**

BriDe Arbitrager：通过贿赂支持的延迟区块生产增强以太坊2.0中的套利 cs.NI

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.08537v1) [paper-pdf](http://arxiv.org/pdf/2407.08537v1)

**Authors**: Hulin Yang, Mingzhe Li, Jin Zhang, Alia Asheralieva, Qingsong Wei, Siow Mong Rick Goh

**Abstract**: The advent of Ethereum 2.0 has introduced significant changes, particularly the shift to Proof-of-Stake consensus. This change presents new opportunities and challenges for arbitrage. Amidst these changes, we introduce BriDe Arbitrager, a novel tool designed for Ethereum 2.0 that leverages Bribery-driven attacks to Delay block production and increase arbitrage gains. The main idea is to allow malicious proposers to delay block production by bribing validators/proposers, thereby gaining more time to identify arbitrage opportunities. Through analysing the bribery process, we design an adaptive bribery strategy. Additionally, we propose a Delayed Transaction Ordering Algorithm to leverage the delayed time to amplify arbitrage profits for malicious proposers. To ensure fairness and automate the bribery process, we design and implement a bribery smart contract and a bribery client. As a result, BriDe Arbitrager enables adversaries controlling a limited (< 1/4) fraction of the voting powers to delay block production via bribery and arbitrage more profit. Extensive experimental results based on Ethereum historical transactions demonstrate that BriDe Arbitrager yields an average of 8.66 ETH (16,442.23 USD) daily profits. Furthermore, our approach does not trigger any slashing mechanisms and remains effective even under Proposer Builder Separation and other potential mechanisms will be adopted by Ethereum.

摘要: Etherum 2.0的问世带来了重大变化，特别是转向利害关系证明共识。这种变化给套利带来了新的机遇和挑战。在这些变化中，我们引入了新娘套利，这是为以太2.0设计的一个新工具，它利用贿赂驱动的攻击来延迟大宗生产并增加套利收益。其主要思想是允许恶意提议者通过贿赂验证者/提议者来延迟区块生产，从而获得更多时间来识别套利机会。通过对贿赂过程的分析，设计了一种自适应的贿赂策略。此外，我们还提出了延迟交易排序算法，以利用延迟时间为恶意提出者放大套利利润。为了确保贿赂过程的公平性和自动化，我们设计并实现了一个贿赂智能合同和一个贿赂客户端。因此，新娘套利者使控制有限(<1/4)投票权的对手能够通过贿赂推迟阻止生产，并套利更多利润。基于以太历史交易的广泛实验结果表明，新娘套利者平均每天获得8.66 ETH(16,442.23美元)的利润。此外，我们的方法不会触发任何削减机制，即使在Proposer Builder Separation和Etherum将采用其他潜在机制的情况下也仍然有效。



## **8. Rethinking the Threat and Accessibility of Adversarial Attacks against Face Recognition Systems**

重新思考针对人脸识别系统的对抗性攻击的威胁和可及性 cs.CV

19 pages, 12 figures

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.08514v1) [paper-pdf](http://arxiv.org/pdf/2407.08514v1)

**Authors**: Yuxin Cao, Yumeng Zhu, Derui Wang, Sheng Wen, Minhui Xue, Jin Lu, Hao Ge

**Abstract**: Face recognition pipelines have been widely deployed in various mission-critical systems in trust, equitable and responsible AI applications. However, the emergence of adversarial attacks has threatened the security of the entire recognition pipeline. Despite the sheer number of attack methods proposed for crafting adversarial examples in both digital and physical forms, it is never an easy task to assess the real threat level of different attacks and obtain useful insight into the key risks confronted by face recognition systems. Traditional attacks view imperceptibility as the most important measurement to keep perturbations stealthy, while we suspect that industry professionals may possess a different opinion. In this paper, we delve into measuring the threat brought about by adversarial attacks from the perspectives of the industry and the applications of face recognition. In contrast to widely studied sophisticated attacks in the field, we propose an effective yet easy-to-launch physical adversarial attack, named AdvColor, against black-box face recognition pipelines in the physical world. AdvColor fools models in the recognition pipeline via directly supplying printed photos of human faces to the system under adversarial illuminations. Experimental results show that physical AdvColor examples can achieve a fooling rate of more than 96% against the anti-spoofing model and an overall attack success rate of 88% against the face recognition pipeline. We also conduct a survey on the threats of prevailing adversarial attacks, including AdvColor, to understand the gap between the machine-measured and human-assessed threat levels of different forms of adversarial attacks. The survey results surprisingly indicate that, compared to deliberately launched imperceptible attacks, perceptible but accessible attacks pose more lethal threats to real-world commercial systems of face recognition.

摘要: 人脸识别管道已广泛部署在各种任务关键系统中，用于信任、公平和负责任的人工智能应用。然而，对抗性攻击的出现威胁到了整个识别管道的安全。尽管人们提出了大量的攻击方法来制作数字和物理形式的敌意例子，但评估不同攻击的真实威胁级别并对人脸识别系统面临的关键风险进行有用的洞察从来都不是一项容易的任务。传统攻击将不可感知性视为保持扰动隐蔽性的最重要衡量标准，而我们怀疑行业专业人士可能持有不同的观点。在本文中，我们从行业和人脸识别应用的角度，深入研究了对抗性攻击带来的威胁的度量。与该领域广泛研究的复杂攻击不同，我们提出了一种针对物理世界中的黑盒人脸识别管道的有效且易于启动的物理对手攻击，称为AdvColor。AdvColor通过在对抗性照明下直接向系统提供打印的人脸照片来愚弄识别管道中的模型。实验结果表明，物理AdvColor实例对反欺骗模型的欺骗率达到96%以上，对人脸识别管道的整体攻击成功率达到88%。我们还对包括AdvColor在内的主流对抗性攻击的威胁进行了调查，以了解不同形式的对抗性攻击的机器测量和人类评估的威胁水平之间的差距。调查结果令人惊讶地表明，与故意发起的潜伏攻击相比，可感知但可访问的攻击对现实世界中的人脸识别商业系统构成了更致命的威胁。



## **9. Resilience of Entropy Model in Distributed Neural Networks**

分布式神经网络中的熵模型的弹性 cs.LG

accepted at ECCV 2024

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2403.00942v2) [paper-pdf](http://arxiv.org/pdf/2403.00942v2)

**Authors**: Milin Zhang, Mohammad Abdi, Shahriar Rifat, Francesco Restuccia

**Abstract**: Distributed deep neural networks (DNNs) have emerged as a key technique to reduce communication overhead without sacrificing performance in edge computing systems. Recently, entropy coding has been introduced to further reduce the communication overhead. The key idea is to train the distributed DNN jointly with an entropy model, which is used as side information during inference time to adaptively encode latent representations into bit streams with variable length. To the best of our knowledge, the resilience of entropy models is yet to be investigated. As such, in this paper we formulate and investigate the resilience of entropy models to intentional interference (e.g., adversarial attacks) and unintentional interference (e.g., weather changes and motion blur). Through an extensive experimental campaign with 3 different DNN architectures, 2 entropy models and 4 rate-distortion trade-off factors, we demonstrate that the entropy attacks can increase the communication overhead by up to 95%. By separating compression features in frequency and spatial domain, we propose a new defense mechanism that can reduce the transmission overhead of the attacked input by about 9% compared to unperturbed data, with only about 2% accuracy loss. Importantly, the proposed defense mechanism is a standalone approach which can be applied in conjunction with approaches such as adversarial training to further improve robustness. Code will be shared for reproducibility.

摘要: 分布式深度神经网络(DNN)已成为边缘计算系统中在不牺牲性能的前提下减少通信开销的关键技术。最近，引入了熵编码来进一步降低通信开销。该算法的核心思想是将分布的DNN与一个熵模型联合训练，作为推理时的辅助信息，自适应地将潜在的表示编码成可变长度的比特流。就我们所知，熵模型的弹性还有待研究。因此，在本文中，我们建立并研究了熵模型对有意干扰(例如，对抗性攻击)和无意干扰(例如，天气变化和运动模糊)的弹性。通过使用3种不同的DNN结构、2种熵模型和4种率失真权衡因子的广泛实验活动，我们证明了熵攻击可以使通信开销增加高达95%。通过在频域和空间域分离压缩特征，我们提出了一种新的防御机制，与未受干扰的数据相比，该机制可以使被攻击输入的传输开销减少约9%，而精确度损失仅约2%。重要的是，建议的防御机制是一种独立的方法，可以与对抗性训练等方法结合使用，以进一步提高健壮性。代码将被共享，以实现重现性。



## **10. Shedding More Light on Robust Classifiers under the lens of Energy-based Models**

在基于能量的模型的视角下更多地关注稳健分类器 cs.CV

Accepted at European Conference on Computer Vision (ECCV) 2024

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.06315v2) [paper-pdf](http://arxiv.org/pdf/2407.06315v2)

**Authors**: Mujtaba Hussain Mirza, Maria Rosaria Briglia, Senad Beadini, Iacopo Masi

**Abstract**: By reinterpreting a robust discriminative classifier as Energy-based Model (EBM), we offer a new take on the dynamics of adversarial training (AT). Our analysis of the energy landscape during AT reveals that untargeted attacks generate adversarial images much more in-distribution (lower energy) than the original data from the point of view of the model. Conversely, we observe the opposite for targeted attacks. On the ground of our thorough analysis, we present new theoretical and practical results that show how interpreting AT energy dynamics unlocks a better understanding: (1) AT dynamic is governed by three phases and robust overfitting occurs in the third phase with a drastic divergence between natural and adversarial energies (2) by rewriting the loss of TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization (TRADES) in terms of energies, we show that TRADES implicitly alleviates overfitting by means of aligning the natural energy with the adversarial one (3) we empirically show that all recent state-of-the-art robust classifiers are smoothing the energy landscape and we reconcile a variety of studies about understanding AT and weighting the loss function under the umbrella of EBMs. Motivated by rigorous evidence, we propose Weighted Energy Adversarial Training (WEAT), a novel sample weighting scheme that yields robust accuracy matching the state-of-the-art on multiple benchmarks such as CIFAR-10 and SVHN and going beyond in CIFAR-100 and Tiny-ImageNet. We further show that robust classifiers vary in the intensity and quality of their generative capabilities, and offer a simple method to push this capability, reaching a remarkable Inception Score (IS) and FID using a robust classifier without training for generative modeling. The code to reproduce our results is available at http://github.com/OmnAI-Lab/Robust-Classifiers-under-the-lens-of-EBM/ .

摘要: 通过将稳健的判别分类器重新解释为基于能量的模型(EBM)，我们提供了一种新的方法来研究对手训练(AT)的动态。我们对AT过程中的能量格局的分析表明，从模型的角度来看，非目标攻击产生的敌意图像比原始数据更不均匀(能量更低)。相反，我们在有针对性的攻击中观察到相反的情况。在我们深入分析的基础上，我们提出了新的理论和实践结果，表明解释AT能量动力学如何揭示更好的理解：(1)AT动态由三个阶段控制，鲁棒过拟合发生在第三阶段，自然能量和对抗能量之间存在巨大差异(2)通过代理损失最小化(交易)在能量方面改写了权衡激发的对抗性防御的损失，我们表明，交易通过将自然能量与对手能量对齐的方式隐含地缓解了过度匹配。(3)我们的经验表明，所有最近最先进的稳健分类器都在平滑能量格局，我们协调了关于理解AT和在EBM保护伞下加权损失函数的各种研究。在严格证据的激励下，我们提出了加权能量对抗训练(Weat)，这是一种新的样本加权方案，其精度与CIFAR-10和SVHN等多个基准测试的最新水平相当，并超过CIFAR-100和Tiny-ImageNet。我们进一步证明了健壮分类器在其生成能力的强度和质量上存在差异，并提供了一种简单的方法来推动这一能力，使用健壮分类器而不需要为生成性建模进行训练就可以达到显著的初始得分(IS)和FID。复制我们结果的代码可以在http://github.com/OmnAI-Lab/Robust-Classifiers-under-the-lens-of-EBM/上找到。



## **11. A Human-in-the-Middle Attack against Object Detection Systems**

针对对象检测系统的中间人攻击 cs.RO

Accepted by IEEE Transactions on Artificial Intelligence, 2024

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2208.07174v4) [paper-pdf](http://arxiv.org/pdf/2208.07174v4)

**Authors**: Han Wu, Sareh Rowlands, Johan Wahlstrom

**Abstract**: Object detection systems using deep learning models have become increasingly popular in robotics thanks to the rising power of CPUs and GPUs in embedded systems. However, these models are susceptible to adversarial attacks. While some attacks are limited by strict assumptions on access to the detection system, we propose a novel hardware attack inspired by Man-in-the-Middle attacks in cryptography. This attack generates a Universal Adversarial Perturbations (UAP) and injects the perturbation between the USB camera and the detection system via a hardware attack. Besides, prior research is misled by an evaluation metric that measures the model accuracy rather than the attack performance. In combination with our proposed evaluation metrics, we significantly increased the strength of adversarial perturbations. These findings raise serious concerns for applications of deep learning models in safety-critical systems, such as autonomous driving.

摘要: 由于嵌入式系统中中央处理器和图形处理器的性能不断提高，使用深度学习模型的对象检测系统在机器人领域变得越来越受欢迎。然而，这些模型很容易受到对抗攻击。虽然有些攻击受到对检测系统访问权限的严格假设的限制，但我们提出了一种受密码学中中间人攻击启发的新型硬件攻击。该攻击会产生通用对抗性扰动（UAP），并通过硬件攻击在USB摄像头和检测系统之间注入扰动。此外，之前的研究被衡量模型准确性而不是攻击性能的评估指标所误导。结合我们提出的评估指标，我们显着增加了对抗性扰动的强度。这些发现引发了深度学习模型在自动驾驶等安全关键系统中的应用的严重担忧。



## **12. Venomancer: Towards Imperceptible and Target-on-Demand Backdoor Attacks in Federated Learning**

毒液杀手：联邦学习中的不可感知和按需定向后门攻击 cs.CV

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.03144v2) [paper-pdf](http://arxiv.org/pdf/2407.03144v2)

**Authors**: Son Nguyen, Thinh Nguyen, Khoa D Doan, Kok-Seng Wong

**Abstract**: Federated Learning (FL) is a distributed machine learning approach that maintains data privacy by training on decentralized data sources. Similar to centralized machine learning, FL is also susceptible to backdoor attacks, where an attacker can compromise some clients by injecting a backdoor trigger into local models of those clients, leading to the global model's behavior being manipulated as desired by the attacker. Most backdoor attacks in FL assume a predefined target class and require control over a large number of clients or knowledge of benign clients' information. Furthermore, they are not imperceptible and are easily detected by human inspection due to clear artifacts left on the poison data. To overcome these challenges, we propose Venomancer, an effective backdoor attack that is imperceptible and allows target-on-demand. Specifically, imperceptibility is achieved by using a visual loss function to make the poison data visually indistinguishable from the original data. Target-on-demand property allows the attacker to choose arbitrary target classes via conditional adversarial training. Additionally, experiments showed that the method is robust against state-of-the-art defenses such as Norm Clipping, Weak DP, Krum, Multi-Krum, RLR, FedRAD, Deepsight, and RFLBAT. The source code is available at https://github.com/nguyenhongson1902/Venomancer.

摘要: 联合学习(FL)是一种分布式机器学习方法，通过对分散的数据源进行训练来维护数据隐私。与集中式机器学习类似，FL也容易受到后门攻击，攻击者可以通过向某些客户端的本地模型注入后门触发器来危害这些客户端，从而导致全局模型的行为被攻击者想要的操纵。FL中的大多数后门攻击假设一个预定义的目标类，并需要控制大量客户端或了解良性客户端的信息。此外，由于毒物数据上留下了明显的伪影，它们并不是不可察觉的，并且很容易被人类检查发现。为了克服这些挑战，我们提出了毒液杀手，这是一种有效的后门攻击，可以潜移默化，并允许按需锁定目标。具体地说，不可感知性是通过使用视觉损失函数来实现的，以使有毒数据在视觉上与原始数据不可区分。按需目标属性允许攻击者通过有条件的对抗性训练选择任意目标类。此外，实验表明，该方法对Norm裁剪、弱DP、Krum、多Krum、RLR、FedRAD、Deepsight和RFLBAT等最先进的防御方法具有较强的鲁棒性。源代码可在https://github.com/nguyenhongson1902/Venomancer.上找到



## **13. A Comprehensive Survey on the Security of Smart Grid: Challenges, Mitigations, and Future Research Opportunities**

智能电网安全性全面调查：挑战、缓解措施和未来研究机会 cs.CR

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07966v1) [paper-pdf](http://arxiv.org/pdf/2407.07966v1)

**Authors**: Arastoo Zibaeirad, Farnoosh Koleini, Shengping Bi, Tao Hou, Tao Wang

**Abstract**: In this study, we conduct a comprehensive review of smart grid security, exploring system architectures, attack methodologies, defense strategies, and future research opportunities. We provide an in-depth analysis of various attack vectors, focusing on new attack surfaces introduced by advanced components in smart grids. The review particularly includes an extensive analysis of coordinated attacks that incorporate multiple attack strategies and exploit vulnerabilities across various smart grid components to increase their adverse impact, demonstrating the complexity and potential severity of these threats. Following this, we examine innovative detection and mitigation strategies, including game theory, graph theory, blockchain, and machine learning, discussing their advancements in counteracting evolving threats and associated research challenges. In particular, our review covers a thorough examination of widely used machine learning-based mitigation strategies, analyzing their applications and research challenges spanning across supervised, unsupervised, semi-supervised, ensemble, and reinforcement learning. Further, we outline future research directions and explore new techniques and concerns. We first discuss the research opportunities for existing and emerging strategies, and then explore the potential role of new techniques, such as large language models (LLMs), and the emerging threat of adversarial machine learning in the future of smart grid security.

摘要: 在这项研究中，我们对智能电网安全进行了全面的回顾，探索了系统架构、攻击方法、防御策略和未来的研究机会。我们深入分析了各种攻击载体，重点分析了智能电网中先进组件引入的新攻击面。审查特别包括对协调攻击的广泛分析，这些攻击整合了多种攻击策略，并利用各种智能电网组件的漏洞来增加其不利影响，从而展示了这些威胁的复杂性和潜在严重性。随后，我们研究了创新的检测和缓解策略，包括博弈论、图论、区块链和机器学习，讨论了它们在应对不断演变的威胁和相关研究挑战方面的进展。特别是，我们的综述涵盖了广泛使用的基于机器学习的缓解策略的彻底检查，分析了它们在监督、非监督、半监督、集成和强化学习中的应用和研究挑战。此外，我们概述了未来的研究方向，并探索了新的技术和关注的问题。我们首先讨论了现有和新兴策略的研究机会，然后探讨了新技术的潜在作用，如大型语言模型(LLMS)，以及未来智能电网安全中对抗性机器学习的新威胁。



## **14. Adversarial Robustness Limits via Scaling-Law and Human-Alignment Studies**

通过比例定律和人际关系研究的对抗稳健性限制 cs.LG

ICML 2024

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2404.09349v2) [paper-pdf](http://arxiv.org/pdf/2404.09349v2)

**Authors**: Brian R. Bartoldson, James Diffenderfer, Konstantinos Parasyris, Bhavya Kailkhura

**Abstract**: This paper revisits the simple, long-studied, yet still unsolved problem of making image classifiers robust to imperceptible perturbations. Taking CIFAR10 as an example, SOTA clean accuracy is about $100$%, but SOTA robustness to $\ell_{\infty}$-norm bounded perturbations barely exceeds $70$%. To understand this gap, we analyze how model size, dataset size, and synthetic data quality affect robustness by developing the first scaling laws for adversarial training. Our scaling laws reveal inefficiencies in prior art and provide actionable feedback to advance the field. For instance, we discovered that SOTA methods diverge notably from compute-optimal setups, using excess compute for their level of robustness. Leveraging a compute-efficient setup, we surpass the prior SOTA with $20$% ($70$%) fewer training (inference) FLOPs. We trained various compute-efficient models, with our best achieving $74$% AutoAttack accuracy ($+3$% gain). However, our scaling laws also predict robustness slowly grows then plateaus at $90$%: dwarfing our new SOTA by scaling is impractical, and perfect robustness is impossible. To better understand this predicted limit, we carry out a small-scale human evaluation on the AutoAttack data that fools our top-performing model. Concerningly, we estimate that human performance also plateaus near $90$%, which we show to be attributable to $\ell_{\infty}$-constrained attacks' generation of invalid images not consistent with their original labels. Having characterized limiting roadblocks, we outline promising paths for future research.

摘要: 本文回顾了一个简单、研究已久但仍未解决的问题，即使图像分类器对不可察觉的扰动具有健壮性。以CIFAR10为例，SOTA的清洁精度约为$100$%，但对$\ell_{inty}$-范数有界摄动的鲁棒性仅略高于$70$%。为了理解这一差距，我们分析了模型大小、数据集大小和合成数据质量如何通过开发用于对抗性训练的第一个缩放规则来影响稳健性。我们的比例法则揭示了现有技术中的低效，并提供了可操作的反馈来推动该领域的发展。例如，我们发现SOTA方法与计算最优设置明显不同，使用过量计算作为其健壮性级别。利用高效计算的设置，我们比以前的SOTA少了20美元%(70美元%)的培训(推理)失败。我们训练了各种计算效率高的模型，最大限度地达到了$74$%的AutoAttack精度($+3$%的收益)。然而，我们的定标法则也预测稳健性在90美元时缓慢增长然后停滞不前：通过定标来使我们的新SOTA相形见绌是不切实际的，而且完美的稳健性是不可能的。为了更好地理解这一预测极限，我们对AutoAttack数据进行了小规模的人工评估，该评估愚弄了我们的最佳模型。令人担忧的是，我们估计人类的性能也停滞不前近90$%，我们表明这归因于$受限攻击生成的无效图像与其原始标签不一致。在描述了限制障碍的特征之后，我们概述了未来研究的有希望的道路。



## **15. Boosting Transferability in Vision-Language Attacks via Diversification along the Intersection Region of Adversarial Trajectory**

通过沿着对抗轨迹交叉区域的多样化来提高视觉语言攻击的可移植性 cs.CV

ECCV2024. Code is available at  https://github.com/SensenGao/VLPTransferAttack

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2403.12445v2) [paper-pdf](http://arxiv.org/pdf/2403.12445v2)

**Authors**: Sensen Gao, Xiaojun Jia, Xuhong Ren, Ivor Tsang, Qing Guo

**Abstract**: Vision-language pre-training (VLP) models exhibit remarkable capabilities in comprehending both images and text, yet they remain susceptible to multimodal adversarial examples (AEs).Strengthening attacks and uncovering vulnerabilities, especially common issues in VLP models (e.g., high transferable AEs), can advance reliable and practical VLP models. A recent work (i.e., Set-level guidance attack) indicates that augmenting image-text pairs to increase AE diversity along the optimization path enhances the transferability of adversarial examples significantly. However, this approach predominantly emphasizes diversity around the online adversarial examples (i.e., AEs in the optimization period), leading to the risk of overfitting the victim model and affecting the transferability.In this study, we posit that the diversity of adversarial examples towards the clean input and online AEs are both pivotal for enhancing transferability across VLP models. Consequently, we propose using diversification along the intersection region of adversarial trajectory to expand the diversity of AEs.To fully leverage the interaction between modalities, we introduce text-guided adversarial example selection during optimization. Furthermore, to further mitigate the potential overfitting, we direct the adversarial text deviating from the last intersection region along the optimization path, rather than adversarial images as in existing methods.Extensive experiments affirm the effectiveness of our method in improving transferability across various VLP models and downstream vision-and-language tasks.

摘要: 视觉语言预训练(VLP)模型在理解图像和文本方面表现出卓越的能力，但它们仍然容易受到多模式对抗性例子(AE)的影响，加强攻击和发现漏洞，特别是VLP模型中的常见问题(如高可转移性AEs)，可以促进可靠和实用的VLP模型。最近的一项工作(即集合级制导攻击)表明，增加图文对以增加优化路径上的声发射多样性显著地提高了对抗性例子的可转移性。然而，这种方法主要强调在线对抗性实例的多样性(即处于优化期的AEs)，导致受害者模型过度拟合的风险，并影响模型的可转移性。为了充分利用模式间的交互作用，我们在优化过程中引入了文本引导的对抗性范例选择。此外，为了进一步缓解潜在的过度匹配，我们沿着优化路径引导偏离最后一个交集区域的对抗性文本，而不是现有方法中的对抗性图像。大量实验证实了该方法在提高跨各种VLP模型和下游视觉语言任务的可转移性方面的有效性。



## **16. Targeted Augmented Data for Audio Deepfake Detection**

用于音频Deepfake检测的定向增强数据 cs.SD

Accepted in EUSIPCO 2024

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07598v1) [paper-pdf](http://arxiv.org/pdf/2407.07598v1)

**Authors**: Marcella Astrid, Enjie Ghorbel, Djamila Aouada

**Abstract**: The availability of highly convincing audio deepfake generators highlights the need for designing robust audio deepfake detectors. Existing works often rely solely on real and fake data available in the training set, which may lead to overfitting, thereby reducing the robustness to unseen manipulations. To enhance the generalization capabilities of audio deepfake detectors, we propose a novel augmentation method for generating audio pseudo-fakes targeting the decision boundary of the model. Inspired by adversarial attacks, we perturb original real data to synthesize pseudo-fakes with ambiguous prediction probabilities. Comprehensive experiments on two well-known architectures demonstrate that the proposed augmentation contributes to improving the generalization capabilities of these architectures.

摘要: 高度令人信服的音频深度伪造生成器的可用性凸显了设计稳健的音频深度伪造检测器的必要性。现有的作品通常仅依赖于训练集中可用的真实和虚假数据，这可能会导致过度匹配，从而降低对不可见操纵的鲁棒性。为了增强音频深度伪造检测器的概括能力，我们提出了一种新颖的增强方法，用于生成针对模型决策边界的音频伪伪造。受对抗攻击的启发，我们扰乱原始真实数据以合成预测概率模糊的伪假货。对两种知名架构的综合实验表明，所提出的增强有助于提高这些架构的概括能力。



## **17. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

DistriBlock：通过利用输出分布的特征来识别对抗性音频样本 cs.SD

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2305.17000v5) [paper-pdf](http://arxiv.org/pdf/2305.17000v5)

**Authors**: Matías P. Pizarro B., Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic curve for distinguishing target adversarial examples against clean and noisy data of 99% and 97%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.

摘要: 敌意攻击可以误导自动语音识别(ASR)系统预测任意目标文本，从而构成明显的安全威胁。为了防止此类攻击，我们提出了DistriBlock，这是一种适用于任何ASR系统的有效检测策略，它预测每个时间步输出令牌上的概率分布。我们测量了该分布的一组特征：输出概率的中位数、最大值和最小值，分布的熵，以及关于后续时间步分布的Kullback-Leibler和Jensen-Shannon散度。然后，通过利用对良性数据和恶意数据观察到的特征，我们应用二进制分类器，包括简单的基于阈值的分类、这种分类器的集成和神经网络。通过对不同的ASR系统和语言数据集的广泛分析，我们证明了该方法的最高性能，在干净和有噪声的数据下，接收器操作特征曲线下的平均面积分别为99%和97%。为了评估我们方法的健壮性，我们证明了可以绕过DistriBlock的自适应攻击示例的噪声要大得多，这使得它们更容易通过过滤来检测，并为保持系统的健壮性创造了另一种途径。



## **18. Evaluating the Adversarial Robustness of Retrieval-Based In-Context Learning for Large Language Models**

评估大型语言模型基于检索的上下文学习的对抗鲁棒性 cs.CL

COLM 2024, 29 pages, 6 figures

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2405.15984v2) [paper-pdf](http://arxiv.org/pdf/2405.15984v2)

**Authors**: Simon Chi Lok Yu, Jie He, Pasquale Minervini, Jeff Z. Pan

**Abstract**: With the emergence of large language models, such as LLaMA and OpenAI GPT-3, In-Context Learning (ICL) gained significant attention due to its effectiveness and efficiency. However, ICL is very sensitive to the choice, order, and verbaliser used to encode the demonstrations in the prompt. Retrieval-Augmented ICL methods try to address this problem by leveraging retrievers to extract semantically related examples as demonstrations. While this approach yields more accurate results, its robustness against various types of adversarial attacks, including perturbations on test samples, demonstrations, and retrieved data, remains under-explored. Our study reveals that retrieval-augmented models can enhance robustness against test sample attacks, outperforming vanilla ICL with a 4.87% reduction in Attack Success Rate (ASR); however, they exhibit overconfidence in the demonstrations, leading to a 2% increase in ASR for demonstration attacks. Adversarial training can help improve the robustness of ICL methods to adversarial attacks; however, such a training scheme can be too costly in the context of LLMs. As an alternative, we introduce an effective training-free adversarial defence method, DARD, which enriches the example pool with those attacked samples. We show that DARD yields improvements in performance and robustness, achieving a 15% reduction in ASR over the baselines. Code and data are released to encourage further research: https://github.com/simonucl/adv-retreival-icl

摘要: 随着大型语言模型的出现，如Llama和OpenAI GPT-3，情景中学习(ICL)因其有效性和高效性而受到广泛关注。但是，ICL对用于对提示符中的演示进行编码的选择、顺序和形容词非常敏感。检索增强的ICL方法试图通过利用检索器来提取语义相关的示例作为演示来解决这个问题。虽然这种方法可以产生更准确的结果，但它对各种类型的对抗性攻击的稳健性，包括对测试样本、演示和检索数据的扰动，仍然没有得到充分的研究。我们的研究表明，检索增强模型可以增强对测试样本攻击的健壮性，性能优于普通ICL，攻击成功率(ASR)降低4.87%；然而，它们在演示中表现出过度自信，导致演示攻击的ASR提高了2%。对抗性训练可以帮助提高ICL方法对对抗性攻击的稳健性；然而，在LLMS的背景下，这样的训练方案可能代价太高。作为另一种选择，我们引入了一种有效的无需训练的对抗防御方法DARD，它用被攻击的样本丰富了样本库。我们表明，DARD在性能和健壮性方面都有改进，ASR比基准降低了15%。发布代码和数据是为了鼓励进一步的研究：https://github.com/simonucl/adv-retreival-icl



## **19. Invisible Optical Adversarial Stripes on Traffic Sign against Autonomous Vehicles**

针对自动驾驶车辆的交通标志上的隐形光学对抗条纹 cs.CR

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07510v1) [paper-pdf](http://arxiv.org/pdf/2407.07510v1)

**Authors**: Dongfang Guo, Yuting Wu, Yimin Dai, Pengfei Zhou, Xin Lou, Rui Tan

**Abstract**: Camera-based computer vision is essential to autonomous vehicle's perception. This paper presents an attack that uses light-emitting diodes and exploits the camera's rolling shutter effect to create adversarial stripes in the captured images to mislead traffic sign recognition. The attack is stealthy because the stripes on the traffic sign are invisible to human. For the attack to be threatening, the recognition results need to be stable over consecutive image frames. To achieve this, we design and implement GhostStripe, an attack system that controls the timing of the modulated light emission to adapt to camera operations and victim vehicle movements. Evaluated on real testbeds, GhostStripe can stably spoof the traffic sign recognition results for up to 94\% of frames to a wrong class when the victim vehicle passes the road section. In reality, such attack effect may fool victim vehicles into life-threatening incidents. We discuss the countermeasures at the levels of camera sensor, perception model, and autonomous driving system.

摘要: 基于摄像头的计算机视觉对于自动驾驶汽车的感知是必不可少的。本文提出了一种利用发光二极管和利用摄像机的滚动快门效应在捕获的图像中产生对抗性条纹来误导交通标志识别的攻击方法。这次袭击是隐形的，因为交通标志上的条纹是人类看不见的。为了使攻击具有威胁性，识别结果需要在连续的图像帧上保持稳定。为了实现这一点，我们设计并实现了Ghost Strike，这是一个攻击系统，它控制调制光发射的时间，以适应相机操作和受害者车辆的移动。在真实的测试平台上进行了测试，当受害者车辆通过路段时，Ghost Strike可以稳定地将高达94%的帧的交通标志识别结果伪造到错误的类别。在现实中，这种攻击效果可能会欺骗受害者车辆发生危及生命的事件。分别从摄像机传感器、感知模型、自动驾驶系统三个层面探讨了对策。



## **20. Formal Verification of Object Detection**

对象检测的形式化验证 cs.CV

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.01295v3) [paper-pdf](http://arxiv.org/pdf/2407.01295v3)

**Authors**: Avraham Raviv, Yizhak Y. Elboher, Michelle Aluf-Medina, Yael Leibovich Weiss, Omer Cohen, Roy Assa, Guy Katz, Hillel Kugler

**Abstract**: Deep Neural Networks (DNNs) are ubiquitous in real-world applications, yet they remain vulnerable to errors and adversarial attacks. This work tackles the challenge of applying formal verification to ensure the safety of computer vision models, extending verification beyond image classification to object detection. We propose a general formulation for certifying the robustness of object detection models using formal verification and outline implementation strategies compatible with state-of-the-art verification tools. Our approach enables the application of these tools, originally designed for verifying classification models, to object detection. We define various attacks for object detection, illustrating the diverse ways adversarial inputs can compromise neural network outputs. Our experiments, conducted on several common datasets and networks, reveal potential errors in object detection models, highlighting system vulnerabilities and emphasizing the need for expanding formal verification to these new domains. This work paves the way for further research in integrating formal verification across a broader range of computer vision applications.

摘要: 深度神经网络(DNN)在实际应用中无处不在，但它们仍然容易受到错误和对手攻击。这项工作解决了应用形式化验证来确保计算机视觉模型的安全性的挑战，将验证从图像分类扩展到目标检测。我们提出了使用形式化验证来证明目标检测模型的健壮性的一般公式，并概述了与最先进的验证工具兼容的实现策略。我们的方法使得这些最初设计用于验证分类模型的工具能够应用于目标检测。我们定义了用于目标检测的各种攻击，说明了敌意输入可以损害神经网络输出的不同方式。我们在几个常见的数据集和网络上进行的实验，揭示了对象检测模型中的潜在错误，突出了系统漏洞，并强调了将正式验证扩展到这些新领域的必要性。这项工作为在更广泛的计算机视觉应用中整合形式验证的进一步研究铺平了道路。



## **21. A Survey of Attacks on Large Vision-Language Models: Resources, Advances, and Future Trends**

大型视觉语言模型攻击调查：资源、进展和未来趋势 cs.CV

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07403v1) [paper-pdf](http://arxiv.org/pdf/2407.07403v1)

**Authors**: Daizong Liu, Mingyu Yang, Xiaoye Qu, Pan Zhou, Wei Hu, Yu Cheng

**Abstract**: With the significant development of large models in recent years, Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities across a wide range of multimodal understanding and reasoning tasks. Compared to traditional Large Language Models (LLMs), LVLMs present great potential and challenges due to its closer proximity to the multi-resource real-world applications and the complexity of multi-modal processing. However, the vulnerability of LVLMs is relatively underexplored, posing potential security risks in daily usage. In this paper, we provide a comprehensive review of the various forms of existing LVLM attacks. Specifically, we first introduce the background of attacks targeting LVLMs, including the attack preliminary, attack challenges, and attack resources. Then, we systematically review the development of LVLM attack methods, such as adversarial attacks that manipulate model outputs, jailbreak attacks that exploit model vulnerabilities for unauthorized actions, prompt injection attacks that engineer the prompt type and pattern, and data poisoning that affects model training. Finally, we discuss promising research directions in the future. We believe that our survey provides insights into the current landscape of LVLM vulnerabilities, inspiring more researchers to explore and mitigate potential safety issues in LVLM developments. The latest papers on LVLM attacks are continuously collected in https://github.com/liudaizong/Awesome-LVLM-Attack.

摘要: 近年来，随着大型模型的显著发展，大型视觉语言模型在广泛的多通道理解和推理任务中表现出了卓越的能力。与传统的大语言模型相比，大语言模型因其更接近多资源的实际应用和多模式处理的复杂性而显示出巨大的潜力和挑战。然而，LVLMS的脆弱性相对较少，在日常使用中存在潜在的安全风险。在本文中，我们对现有的各种形式的LVLM攻击进行了全面的回顾。具体地说，我们首先介绍了针对LVLMS的攻击背景，包括攻击准备、攻击挑战和攻击资源。然后，我们系统地回顾了LVLM攻击方法的发展，如操纵模型输出的对抗性攻击，利用模型漏洞进行未经授权操作的越狱攻击，设计提示类型和模式的提示注入攻击，以及影响模型训练的数据中毒。最后，我们讨论了未来的研究方向。我们相信，我们的调查提供了对LVLM漏洞现状的洞察，激励更多的研究人员探索和缓解LVLM开发中的潜在安全问题。有关LVLm攻击的最新论文在https://github.com/liudaizong/Awesome-LVLM-Attack.上不断收集



## **22. Marlin: Knowledge-Driven Analysis of Provenance Graphs for Efficient and Robust Detection of Cyber Attacks**

马林：知识驱动的源源图分析，以高效、稳健地检测网络攻击 cs.CR

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2403.12541v2) [paper-pdf](http://arxiv.org/pdf/2403.12541v2)

**Authors**: Zhenyuan Li, Yangyang Wei, Xiangmin Shen, Lingzhi Wang, Yan Chen, Haitao Xu, Shouling Ji, Fan Zhang, Liang Hou, Wenmao Liu, Xuhong Zhang, Jianwei Ying

**Abstract**: Recent research in both academia and industry has validated the effectiveness of provenance graph-based detection for advanced cyber attack detection and investigation. However, analyzing large-scale provenance graphs often results in substantial overhead. To improve performance, existing detection systems implement various optimization strategies. Yet, as several recent studies suggest, these strategies could lose necessary context information and be vulnerable to evasions. Designing a detection system that is efficient and robust against adversarial attacks is an open problem. We introduce Marlin, which approaches cyber attack detection through real-time provenance graph alignment.By leveraging query graphs embedded with attack knowledge, Marlin can efficiently identify entities and events within provenance graphs, embedding targeted analysis and significantly narrowing the search space. Moreover, we incorporate our graph alignment algorithm into a tag propagation-based schema to eliminate the need for storing and reprocessing raw logs. This design significantly reduces in-memory storage requirements and minimizes data processing overhead. As a result, it enables real-time graph alignment while preserving essential context information, thereby enhancing the robustness of cyber attack detection. Moreover, Marlin allows analysts to customize attack query graphs flexibly to detect extended attacks and provide interpretable detection results. We conduct experimental evaluations on two large-scale public datasets containing 257.42 GB of logs and 12 query graphs of varying sizes, covering multiple attack techniques and scenarios. The results show that Marlin can process 137K events per second while accurately identifying 120 subgraphs with 31 confirmed attacks, along with only 1 false positive, demonstrating its efficiency and accuracy in handling massive data.

摘要: 最近学术界和工业界的研究都证实了基于起源图的检测对于高级网络攻击检测和调查的有效性。然而，分析大规模的种源图表往往会产生相当大的开销。为了提高性能，现有的检测系统采用了各种优化策略。然而，正如最近的几项研究表明的那样，这些策略可能会失去必要的背景信息，并容易受到规避。设计一个对敌方攻击高效且健壮的检测系统是一个悬而未决的问题。介绍了Marlin算法，该算法通过对源图进行实时比对来实现网络攻击的检测，利用嵌入攻击知识的查询图，能够有效地识别源图中的实体和事件，嵌入针对性的分析，大大缩小了搜索空间。此外，我们将我们的图对齐算法整合到基于标记传播的模式中，以消除存储和重新处理原始日志的需要。这种设计显著降低了内存存储需求，并最大限度地减少了数据处理开销。因此，它能够在保留基本上下文信息的同时实现实时图形对齐，从而增强网络攻击检测的健壮性。此外，Marlin允许分析人员灵活地定制攻击查询图，以检测扩展的攻击并提供可解释的检测结果。我们在两个包含257.42 GB日志和12个不同大小的查询图的大规模公共数据集上进行了实验评估，涵盖了多种攻击技术和场景。实验结果表明，Marlin能够在每秒处理137K事件的同时，准确识别出120个子图中31个已确认的攻击，并且只有1个误报，证明了其在处理海量数据时的效率和准确性。



## **23. Characterizing Encrypted Application Traffic through Cellular Radio Interface Protocol**

通过蜂窝无线电接口协议描述加密应用流量 cs.NI

9 pages, 8 figures, 2 tables. This paper has been accepted for  publication by the 21st IEEE International Conference on Mobile Ad-Hoc and  Smart Systems (MASS 2024)

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07361v1) [paper-pdf](http://arxiv.org/pdf/2407.07361v1)

**Authors**: Md Ruman Islam, Raja Hasnain Anwar, Spyridon Mastorakis, Muhammad Taqi Raza

**Abstract**: Modern applications are end-to-end encrypted to prevent data from being read or secretly modified. 5G tech nology provides ubiquitous access to these applications without compromising the application-specific performance and latency goals. In this paper, we empirically demonstrate that 5G radio communication becomes the side channel to precisely infer the user's applications in real-time. The key idea lies in observing the 5G physical and MAC layer interactions over time that reveal the application's behavior. The MAC layer receives the data from the application and requests the network to assign the radio resource blocks. The network assigns the radio resources as per application requirements, such as priority, Quality of Service (QoS) needs, amount of data to be transmitted, and buffer size. The adversary can passively observe the radio resources to fingerprint the applications. We empirically demonstrate this attack by considering four different categories of applications: online shopping, voice/video conferencing, video streaming, and Over-The-Top (OTT) media platforms. Finally, we have also demonstrated that an attacker can differentiate various types of applications in real-time within each category.

摘要: 现代应用程序是端到端加密的，以防止数据被读取或秘密修改。5G技术提供了对这些应用的无处不在的访问，而不会影响特定于应用的性能和延迟目标。在本文中，我们实证地论证了5G无线通信成为实时准确推断用户应用的辅助通道。关键思想在于观察5G物理层和MAC层随时间的交互，以揭示应用的行为。MAC层从应用程序接收数据，并请求网络分配无线电资源块。网络根据诸如优先级、服务质量(Qos)需求、要传输的数据量和缓冲区大小等应用需求来分配无线电资源。敌手可以被动地观察无线电资源来识别应用程序。我们考虑了四种不同类别的应用程序：在线购物、语音/视频会议、视频流和Over-the-Top(OTT)媒体平台，对这一攻击进行了实证演示。最后，我们还演示了攻击者可以在每个类别中实时区分各种类型的应用程序。



## **24. The Quantum Imitation Game: Reverse Engineering of Quantum Machine Learning Models**

量子模仿游戏：量子机器学习模型的反向工程 quant-ph

10 pages, 12 figures

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.07237v1) [paper-pdf](http://arxiv.org/pdf/2407.07237v1)

**Authors**: Archisman Ghosh, Swaroop Ghosh

**Abstract**: Quantum Machine Learning (QML) amalgamates quantum computing paradigms with machine learning models, providing significant prospects for solving complex problems. However, with the expansion of numerous third-party vendors in the Noisy Intermediate-Scale Quantum (NISQ) era of quantum computing, the security of QML models is of prime importance, particularly against reverse engineering, which could expose trained parameters and algorithms of the models. We assume the untrusted quantum cloud provider is an adversary having white-box access to the transpiled user-designed trained QML model during inference. Reverse engineering (RE) to extract the pre-transpiled QML circuit will enable re-transpilation and usage of the model for various hardware with completely different native gate sets and even different qubit technology. Such flexibility may not be obtained from the transpiled circuit which is tied to a particular hardware and qubit technology. The information about the number of parameters, and optimized values can allow further training of the QML model to alter the QML model, tamper with the watermark, and/or embed their own watermark or refine the model for other purposes. In this first effort to investigate the RE of QML circuits, we perform RE and compare the training accuracy of original and reverse-engineered Quantum Neural Networks (QNNs) of various sizes. We note that multi-qubit classifiers can be reverse-engineered under specific conditions with a mean error of order 1e-2 in a reasonable time. We also propose adding dummy fixed parametric gates in the QML models to increase the RE overhead for defense. For instance, adding 2 dummy qubits and 2 layers increases the overhead by ~1.76 times for a classifier with 2 qubits and 3 layers with a performance overhead of less than 9%. We note that RE is a very powerful attack model which warrants further efforts on defenses.

摘要: 量子机器学习(QML)融合了量子计算范式和机器学习模型，为解决复杂问题提供了重要的前景。然而，在喧嚣的中间尺度量子计算(NISQ)时代，随着众多第三方供应商的扩张，QML模型的安全性至关重要，特别是在对抗逆向工程时，逆向工程可能会暴露模型的训练参数和算法。我们假设不可信的量子云提供商是一个对手，在推理过程中可以通过白盒访问用户设计的经过训练的QML模型。逆向工程(RE)提取预转换的QML电路将使模型能够重新转置并用于具有完全不同的本机门设置甚至不同的量子比特技术的各种硬件。这种灵活性可能不是从绑定到特定硬件和量子比特技术的分流电路获得的。关于参数数目和最佳值的信息可以允许进一步训练QML模型以改变QML模型、篡改水印、和/或出于其他目的嵌入它们自己的水印或改进模型。在第一次研究QML电路的RE时，我们进行了RE，并比较了不同大小的原始和反向工程量子神经网络(QNN)的训练精度。我们注意到，多量子比特分类器可以在特定条件下进行逆向工程，在合理的时间内，平均误差为1e-2阶。我们还建议在QML模型中增加虚拟固定参数门，以增加防御的RE开销。例如，对于具有2个量子比特和3个层的分类器，添加2个虚拟量子比特和2个层会使开销增加~1.76倍，而性能开销不到9%。我们注意到，RE是一种非常强大的攻击模式，需要在防御上进一步努力。



## **25. Robust Neural Information Retrieval: An Adversarial and Out-of-distribution Perspective**

稳健的神经信息检索：对抗性和非分布性的角度 cs.IR

Survey paper

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06992v1) [paper-pdf](http://arxiv.org/pdf/2407.06992v1)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Recent advances in neural information retrieval (IR) models have significantly enhanced their effectiveness over various IR tasks. The robustness of these models, essential for ensuring their reliability in practice, has also garnered significant attention. With a wide array of research on robust IR being proposed, we believe it is the opportune moment to consolidate the current status, glean insights from existing methodologies, and lay the groundwork for future development. We view the robustness of IR to be a multifaceted concept, emphasizing its necessity against adversarial attacks, out-of-distribution (OOD) scenarios and performance variance. With a focus on adversarial and OOD robustness, we dissect robustness solutions for dense retrieval models (DRMs) and neural ranking models (NRMs), respectively, recognizing them as pivotal components of the neural IR pipeline. We provide an in-depth discussion of existing methods, datasets, and evaluation metrics, shedding light on challenges and future directions in the era of large language models. To the best of our knowledge, this is the first comprehensive survey on the robustness of neural IR models, and we will also be giving our first tutorial presentation at SIGIR 2024 \url{https://sigir2024-robust-information-retrieval.github.io}. Along with the organization of existing work, we introduce a Benchmark for robust IR (BestIR), a heterogeneous evaluation benchmark for robust neural information retrieval, which is publicly available at \url{https://github.com/Davion-Liu/BestIR}. We hope that this study provides useful clues for future research on the robustness of IR models and helps to develop trustworthy search engines \url{https://github.com/Davion-Liu/Awesome-Robustness-in-Information-Retrieval}.

摘要: 神经信息检索(IR)模型的最新进展显著提高了它们在各种IR任务中的有效性。这些模型的稳健性对于确保它们在实践中的可靠性至关重要，也引起了人们的极大关注。随着对稳健IR的广泛研究的提出，我们认为现在是巩固当前状况、从现有方法中收集见解并为未来发展奠定基础的好时机。我们认为信息检索的稳健性是一个多方面的概念，强调了它对对抗攻击、分布外(OOD)场景和性能差异的必要性。以对抗性和面向对象的稳健性为重点，我们分别剖析了密集检索模型(DRM)和神经排名模型(NRM)的稳健性解决方案，将它们识别为神经IR管道的关键组件。我们提供了对现有方法、数据集和评估度量的深入讨论，揭示了大型语言模型时代的挑战和未来方向。据我们所知，这是关于神经IR模型稳健性的第一次全面调查，我们还将在SIGIR2024\url{https://sigir2024-robust-information-retrieval.github.io}.上进行我们的第一次教程演示在组织现有工作的同时，我们还介绍了稳健IR基准(BSTIR)，这是一个用于稳健神经信息检索的异质评估基准，可在\url{https://github.com/Davion-Liu/BestIR}.希望本研究为今后研究信息检索模型的健壮性提供有用的线索，并为开发可信搜索引擎\url{https://github.com/Davion-Liu/Awesome-Robustness-in-Information-Retrieval}.提供帮助



## **26. Does CLIP Know My Face?**

CLIP认识我的脸吗？ cs.LG

Published in the Journal of Artificial Intelligence Research (JAIR)

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2209.07341v4) [paper-pdf](http://arxiv.org/pdf/2209.07341v4)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Manuel Brack, Felix Friedrich, Patrick Schramowski, Kristian Kersting

**Abstract**: With the rise of deep learning in various applications, privacy concerns around the protection of training data have become a critical area of research. Whereas prior studies have focused on privacy risks in single-modal models, we introduce a novel method to assess privacy for multi-modal models, specifically vision-language models like CLIP. The proposed Identity Inference Attack (IDIA) reveals whether an individual was included in the training data by querying the model with images of the same person. Letting the model choose from a wide variety of possible text labels, the model reveals whether it recognizes the person and, therefore, was used for training. Our large-scale experiments on CLIP demonstrate that individuals used for training can be identified with very high accuracy. We confirm that the model has learned to associate names with depicted individuals, implying the existence of sensitive information that can be extracted by adversaries. Our results highlight the need for stronger privacy protection in large-scale models and suggest that IDIAs can be used to prove the unauthorized use of data for training and to enforce privacy laws.

摘要: 随着深度学习在各种应用中的兴起，围绕训练数据保护的隐私问题已经成为一个关键的研究领域。鉴于以往的研究主要集中于单通道模型中的隐私风险，我们引入了一种新的方法来评估多通道模型的隐私，特别是像CLIP这样的视觉语言模型。提出的身份推断攻击(IDIA)通过用同一人的图像查询模型来揭示该人是否包括在训练数据中。让模型从各种各样的可能的文本标签中进行选择，该模型显示它是否识别出这个人，因此，它被用于训练。我们在CLIP上的大规模实验表明，用于训练的个体可以非常准确地识别。我们确认，该模型已经学会了将姓名与所描述的个人相关联，这意味着存在可被对手提取的敏感信息。我们的结果强调了在大规模模型中加强隐私保护的必要性，并建议可以使用IDIA来证明未经授权使用数据进行培训和执行隐私法。



## **27. Performance Evaluation of Knowledge Graph Embedding Approaches under Non-adversarial Attacks**

非对抗性攻击下知识图嵌入方法的性能评估 cs.LG

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06855v1) [paper-pdf](http://arxiv.org/pdf/2407.06855v1)

**Authors**: Sourabh Kapoor, Arnab Sharma, Michael Röder, Caglar Demir, Axel-Cyrille Ngonga Ngomo

**Abstract**: Knowledge Graph Embedding (KGE) transforms a discrete Knowledge Graph (KG) into a continuous vector space facilitating its use in various AI-driven applications like Semantic Search, Question Answering, or Recommenders. While KGE approaches are effective in these applications, most existing approaches assume that all information in the given KG is correct. This enables attackers to influence the output of these approaches, e.g., by perturbing the input. Consequently, the robustness of such KGE approaches has to be addressed. Recent work focused on adversarial attacks. However, non-adversarial attacks on all attack surfaces of these approaches have not been thoroughly examined. We close this gap by evaluating the impact of non-adversarial attacks on the performance of 5 state-of-the-art KGE algorithms on 5 datasets with respect to attacks on 3 attack surfaces-graph, parameter, and label perturbation. Our evaluation results suggest that label perturbation has a strong effect on the KGE performance, followed by parameter perturbation with a moderate and graph with a low effect.

摘要: 知识图嵌入(KGE)将离散的知识图(KG)转换为连续的向量空间，便于其在语义搜索、问答或推荐器等各种人工智能驱动的应用中的使用。虽然KGE方法在这些应用中是有效的，但大多数现有方法都假设给定KG中的所有信息都是正确的。这使得攻击者能够影响这些方法的输出，例如，通过干扰输入。因此，必须解决这种KGE方法的稳健性问题。最近的工作集中在对抗性攻击上。然而，这些方法的所有攻击面上的非对抗性攻击还没有得到彻底的审查。我们通过评估非对抗性攻击对5种最先进的KGE算法在5个数据集上的性能的影响来缩小这一差距，这些影响涉及3个攻击面-图、参数和标签扰动。我们的评估结果表明，标签扰动对KGE性能的影响很大，其次是参数扰动，影响中等，图的影响较小。



## **28. EvolBA: Evolutionary Boundary Attack under Hard-label Black Box condition**

EvolBA：硬标签黑匣子条件下的进化边界攻击 cs.CV

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.02248v3) [paper-pdf](http://arxiv.org/pdf/2407.02248v3)

**Authors**: Ayane Tajima, Satoshi Ono

**Abstract**: Research has shown that deep neural networks (DNNs) have vulnerabilities that can lead to the misrecognition of Adversarial Examples (AEs) with specifically designed perturbations. Various adversarial attack methods have been proposed to detect vulnerabilities under hard-label black box (HL-BB) conditions in the absence of loss gradients and confidence scores.However, these methods fall into local solutions because they search only local regions of the search space. Therefore, this study proposes an adversarial attack method named EvolBA to generate AEs using Covariance Matrix Adaptation Evolution Strategy (CMA-ES) under the HL-BB condition, where only a class label predicted by the target DNN model is available. Inspired by formula-driven supervised learning, the proposed method introduces domain-independent operators for the initialization process and a jump that enhances search exploration. Experimental results confirmed that the proposed method could determine AEs with smaller perturbations than previous methods in images where the previous methods have difficulty.

摘要: 研究表明，深度神经网络(DNN)存在漏洞，可能会导致对经过特殊设计的扰动的对抗性示例(AE)的错误识别。针对硬标签黑盒(HL-BB)环境下不存在损失梯度和置信度的漏洞检测问题，提出了多种对抗性攻击方法，但这些方法只搜索搜索空间的局部区域，容易陷入局部解.因此，本文提出了一种基于协方差矩阵自适应进化策略(CMA-ES)的对抗性攻击方法EvolBA，用于在目标DNN模型预测的类别标签不可用的HL-BB条件下生成AEs。受公式驱动的监督学习的启发，该方法在初始化过程中引入了领域无关的算子，并引入了一个跳跃来增强搜索探索。实验结果表明，该方法能够以较小的扰动确定图像中的声学效应，克服了以往方法的不足。



## **29. Learning-Based Difficulty Calibration for Enhanced Membership Inference Attacks**

基于学习的增强型成员推断攻击难度校准 cs.CR

Accepted to IEEE Euro S&P 2024

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2401.04929v3) [paper-pdf](http://arxiv.org/pdf/2401.04929v3)

**Authors**: Haonan Shi, Tu Ouyang, An Wang

**Abstract**: Machine learning models, in particular deep neural networks, are currently an integral part of various applications, from healthcare to finance. However, using sensitive data to train these models raises concerns about privacy and security. One method that has emerged to verify if the trained models are privacy-preserving is Membership Inference Attacks (MIA), which allows adversaries to determine whether a specific data point was part of a model's training dataset. While a series of MIAs have been proposed in the literature, only a few can achieve high True Positive Rates (TPR) in the low False Positive Rate (FPR) region (0.01%~1%). This is a crucial factor to consider for an MIA to be practically useful in real-world settings. In this paper, we present a novel approach to MIA that is aimed at significantly improving TPR at low FPRs. Our method, named learning-based difficulty calibration for MIA(LDC-MIA), characterizes data records by their hardness levels using a neural network classifier to determine membership. The experiment results show that LDC-MIA can improve TPR at low FPR by up to 4x compared to the other difficulty calibration based MIAs. It also has the highest Area Under ROC curve (AUC) across all datasets. Our method's cost is comparable with most of the existing MIAs, but is orders of magnitude more efficient than one of the state-of-the-art methods, LiRA, while achieving similar performance.

摘要: 机器学习模型，特别是深度神经网络，目前是从医疗保健到金融的各种应用程序的组成部分。然而，使用敏感数据来训练这些模型会引发对隐私和安全的担忧。出现的一种验证训练模型是否保护隐私的方法是成员推理攻击(MIA)，它允许对手确定特定数据点是否属于模型训练数据集的一部分。虽然文献中已经提出了一系列的MIA，但只有少数几个MIA能在低假阳性率(FPR)区域(0.01%~1%)获得高的真阳性率(TPR)。要使MIA在实际环境中发挥实际作用，这是需要考虑的关键因素。在本文中，我们提出了一种新的MIA方法，旨在显著改善低FPR下的TPR。我们的方法，称为基于学习的MIA难度校准(LDC-MIA)，使用神经网络分类器来确定成员身份，根据数据记录的硬度来表征数据记录。实验结果表明，与其他基于难度校正的MIA相比，LDC-MIA可以在较低的误码率下将TPR提高4倍。在所有数据集中，它也具有最高的ROC曲线下面积(AUC)。我们的方法的成本与大多数现有的MIA相当，但效率比最先进的方法之一LIRA高出数量级，同时实现了类似的性能。



## **30. A Hybrid Training-time and Run-time Defense Against Adversarial Attacks in Modulation Classification**

调制分类中训练时和运行时混合防御对抗攻击 cs.AI

Published in IEEE Wireless Communications Letters, vol. 11, no. 6,  pp. 1161-1165, June 2022

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06807v1) [paper-pdf](http://arxiv.org/pdf/2407.06807v1)

**Authors**: Lu Zhang, Sangarapillai Lambotharan, Gan Zheng, Guisheng Liao, Ambra Demontis, Fabio Roli

**Abstract**: Motivated by the superior performance of deep learning in many applications including computer vision and natural language processing, several recent studies have focused on applying deep neural network for devising future generations of wireless networks. However, several recent works have pointed out that imperceptible and carefully designed adversarial examples (attacks) can significantly deteriorate the classification accuracy. In this paper, we investigate a defense mechanism based on both training-time and run-time defense techniques for protecting machine learning-based radio signal (modulation) classification against adversarial attacks. The training-time defense consists of adversarial training and label smoothing, while the run-time defense employs a support vector machine-based neural rejection (NR). Considering a white-box scenario and real datasets, we demonstrate that our proposed techniques outperform existing state-of-the-art technologies.

摘要: 受深度学习在计算机视觉和自然语言处理等许多应用中的卓越性能的激励，最近的几项研究专注于应用深度神经网络来设计未来几代无线网络。然而，最近的几篇作品指出，难以察觉且精心设计的对抗性示例（攻击）可能会显着降低分类准确性。本文研究了一种基于训练时和运行时防御技术的防御机制，用于保护基于机器学习的无线电信号（调制）分类免受对抗性攻击。训练时防御由对抗训练和标签平滑组成，而运行时防御则采用基于支持载体机的神经拒绝（NR）。考虑到白盒场景和真实数据集，我们证明我们提出的技术优于现有的最先进技术。



## **31. AdaNCA: Neural Cellular Automata As Adaptors For More Robust Vision Transformer**

AdaNCA：神经元胞自动机作为更稳健的视觉Transformer的适配器 cs.CV

26 pages, 11 figures

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2406.08298v4) [paper-pdf](http://arxiv.org/pdf/2406.08298v4)

**Authors**: Yitao Xu, Tong Zhang, Sabine Süsstrunk

**Abstract**: Vision Transformers (ViTs) have demonstrated remarkable performance in image classification tasks, particularly when equipped with local information via region attention or convolutions. While such architectures improve the feature aggregation from different granularities, they often fail to contribute to the robustness of the networks. Neural Cellular Automata (NCA) enables the modeling of global cell representations through local interactions, with its training strategies and architecture design conferring strong generalization ability and robustness against noisy inputs. In this paper, we propose Adaptor Neural Cellular Automata (AdaNCA) for Vision Transformer that uses NCA as plug-in-play adaptors between ViT layers, enhancing ViT's performance and robustness against adversarial samples as well as out-of-distribution inputs. To overcome the large computational overhead of standard NCAs, we propose Dynamic Interaction for more efficient interaction learning. Furthermore, we develop an algorithm for identifying the most effective insertion points for AdaNCA based on our analysis of AdaNCA placement and robustness improvement. With less than a 3% increase in parameters, AdaNCA contributes to more than 10% absolute improvement in accuracy under adversarial attacks on the ImageNet1K benchmark. Moreover, we demonstrate with extensive evaluations across 8 robustness benchmarks and 4 ViT architectures that AdaNCA, as a plug-in-play module, consistently improves the robustness of ViTs.

摘要: 视觉变形器(VITS)在图像分类任务中表现出了显著的性能，特别是当通过区域注意或卷积来配备局部信息时。虽然这样的体系结构从不同的粒度改善了特征聚合，但它们往往无法提高网络的健壮性。神经元胞自动机(NCA)能够通过局部交互对全局细胞表示进行建模，其训练策略和结构设计具有很强的泛化能力和对噪声输入的鲁棒性。在本文中，我们提出了用于视觉转换器的适配器神经元胞自动机(AdaNCA)，它使用NCA作为VIT层之间的即插即用适配器，增强了VIT的性能和对敌意样本和分布外输入的鲁棒性。为了克服标准NCA计算开销大的缺点，我们提出了动态交互来实现更有效的交互学习。此外，基于对AdaNCA布局和健壮性改进的分析，我们提出了一种识别AdaNCA最有效插入点的算法。在参数增加不到3%的情况下，AdaNCA有助于在对ImageNet1K基准的敌意攻击下将准确率绝对提高10%以上。此外，我们通过对8个健壮性基准和4个VIT体系结构的广泛评估，证明了AdaNCA作为一个即插即用模块，持续提高了VIT的健壮性。



## **32. Countermeasures Against Adversarial Examples in Radio Signal Classification**

无线信号分类中对抗示例的对策 cs.AI

Published in IEEE Wireless Communications Letters, vol. 10, no. 8,  pp. 1830-1834, Aug. 2021

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06796v1) [paper-pdf](http://arxiv.org/pdf/2407.06796v1)

**Authors**: Lu Zhang, Sangarapillai Lambotharan, Gan Zheng, Basil AsSadhan, Fabio Roli

**Abstract**: Deep learning algorithms have been shown to be powerful in many communication network design problems, including that in automatic modulation classification. However, they are vulnerable to carefully crafted attacks called adversarial examples. Hence, the reliance of wireless networks on deep learning algorithms poses a serious threat to the security and operation of wireless networks. In this letter, we propose for the first time a countermeasure against adversarial examples in modulation classification. Our countermeasure is based on a neural rejection technique, augmented by label smoothing and Gaussian noise injection, that allows to detect and reject adversarial examples with high accuracy. Our results demonstrate that the proposed countermeasure can protect deep-learning based modulation classification systems against adversarial examples.

摘要: 深度学习算法已被证明在许多通信网络设计问题中非常强大，包括自动调制分类问题。然而，它们很容易受到精心设计的攻击，称为对抗性例子。因此，无线网络对深度学习算法的依赖对无线网络的安全和运营构成了严重威胁。在这封信中，我们首次提出了针对调制分类中对抗性示例的对策。我们的对策基于神经拒绝技术，通过标签平滑和高斯噪音注入增强，可以高准确性地检测和拒绝对抗性示例。我们的结果表明，提出的对策可以保护基于深度学习的调制分类系统免受对抗性示例的影响。



## **33. Diffusion-Based Adversarial Purification for Speaker Verification**

基于扩散的对抗净化说话人验证 eess.AS

Accepted by IEEE Signal Processing Letters

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2310.14270v3) [paper-pdf](http://arxiv.org/pdf/2310.14270v3)

**Authors**: Yibo Bai, Xiao-Lei Zhang, Xuelong Li

**Abstract**: Recently, automatic speaker verification (ASV) based on deep learning is easily contaminated by adversarial attacks, which is a new type of attack that injects imperceptible perturbations to audio signals so as to make ASV produce wrong decisions. This poses a significant threat to the security and reliability of ASV systems. To address this issue, we propose a Diffusion-Based Adversarial Purification (DAP) method that enhances the robustness of ASV systems against such adversarial attacks. Our method leverages a conditional denoising diffusion probabilistic model to effectively purify the adversarial examples and mitigate the impact of perturbations. DAP first introduces controlled noise into adversarial examples, and then performs a reverse denoising process to reconstruct clean audio. Experimental results demonstrate the efficacy of the proposed DAP in enhancing the security of ASV and meanwhile minimizing the distortion of the purified audio signals.

摘要: 近年来，基于深度学习的自动说话者验证（ASV）很容易受到对抗攻击的污染，对抗攻击是一种新型攻击，它向音频信号注入难以感知的扰动，使ASV做出错误的决策。这对ASV系统的安全性和可靠性构成了重大威胁。为了解决这个问题，我们提出了一种基于扩散的对抗性纯化（DAB）方法，该方法可以增强ASV系统针对此类对抗性攻击的鲁棒性。我们的方法利用条件去噪扩散概率模型来有效地净化对抗示例并减轻扰动的影响。DAB首先将受控噪音引入对抗性示例中，然后执行反向去噪过程以重建干净的音频。实验结果表明，所提出的DAB在增强ASV的安全性并同时最大限度地减少净化音频信号的失真方面的功效。



## **34. Improving the Transferability of Adversarial Examples by Feature Augmentation**

通过特征增强提高对抗性示例的可移植性 cs.CV

19 pages, 4 figures, 4 tables

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06714v1) [paper-pdf](http://arxiv.org/pdf/2407.06714v1)

**Authors**: Donghua Wang, Wen Yao, Tingsong Jiang, Xiaohu Zheng, Junqi Wu, Xiaoqian Chen

**Abstract**: Despite the success of input transformation-based attacks on boosting adversarial transferability, the performance is unsatisfying due to the ignorance of the discrepancy across models. In this paper, we propose a simple but effective feature augmentation attack (FAUG) method, which improves adversarial transferability without introducing extra computation costs. Specifically, we inject the random noise into the intermediate features of the model to enlarge the diversity of the attack gradient, thereby mitigating the risk of overfitting to the specific model and notably amplifying adversarial transferability. Moreover, our method can be combined with existing gradient attacks to augment their performance further. Extensive experiments conducted on the ImageNet dataset across CNN and transformer models corroborate the efficacy of our method, e.g., we achieve improvement of +26.22% and +5.57% on input transformation-based attacks and combination methods, respectively.

摘要: 尽管基于输入转换的攻击在提高对抗可移植性方面取得了成功，但由于忽视了模型之间的差异，性能并不令人满意。在本文中，我们提出了一种简单但有效的特征增强攻击（FAUG）方法，该方法在不引入额外计算成本的情况下提高了对抗性可移植性。具体来说，我们将随机噪音注入到模型的中间特征中，以扩大攻击梯度的多样性，从而降低过度适应特定模型的风险，并显着放大对抗可移植性。此外，我们的方法可以与现有的梯度攻击相结合，以进一步增强其性能。在CNN和Transformer模型上对ImageNet数据集进行的大量实验证实了我们方法的有效性，例如，我们在基于输入转换的攻击和组合方法上分别实现了+26.22%和+5.57%的改进。



## **35. Universal Multi-view Black-box Attack against Object Detectors via Layout Optimization**

通过布局优化对对象检测器进行通用多视图黑匣子攻击 cs.CV

12 pages, 13 figures, 5 tables

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06688v1) [paper-pdf](http://arxiv.org/pdf/2407.06688v1)

**Authors**: Donghua Wang, Wen Yao, Tingsong Jiang, Chao Li, Xiaoqian Chen

**Abstract**: Object detectors have demonstrated vulnerability to adversarial examples crafted by small perturbations that can deceive the object detector. Existing adversarial attacks mainly focus on white-box attacks and are merely valid at a specific viewpoint, while the universal multi-view black-box attack is less explored, limiting their generalization in practice. In this paper, we propose a novel universal multi-view black-box attack against object detectors, which optimizes a universal adversarial UV texture constructed by multiple image stickers for a 3D object via the designed layout optimization algorithm. Specifically, we treat the placement of image stickers on the UV texture as a circle-based layout optimization problem, whose objective is to find the optimal circle layout filled with image stickers so that it can deceive the object detector under the multi-view scenario. To ensure reasonable placement of image stickers, two constraints are elaborately devised. To optimize the layout, we adopt the random search algorithm enhanced by the devised important-aware selection strategy to find the most appropriate image sticker for each circle from the image sticker pools. Extensive experiments conducted on four common object detectors suggested that the detection performance decreases by a large magnitude of 74.29% on average in multi-view scenarios. Additionally, a novel evaluation tool based on the photo-realistic simulator is designed to assess the texture-based attack fairly.

摘要: 对象检测器已经证明了对由可能欺骗对象检测器的小扰动制作的敌意例子的脆弱性。现有的对抗性攻击主要集中在白盒攻击上，并且只在特定的视点有效，而通用的多视点黑盒攻击研究较少，限制了其在实践中的推广。本文提出了一种针对目标检测器的通用多视点黑盒攻击方法，通过设计的布局优化算法，优化了由多个图像贴纸构成的三维物体的通用对抗性UV纹理。具体地说，我们将图像贴纸在UV纹理上的放置视为一个基于圆的布局优化问题，其目标是在多视点场景下找到填充图像贴纸的最优圆形布局，从而欺骗对象检测器。为了确保图像贴纸的合理放置，精心设计了两个约束条件。为了优化布局，我们采用了改进的随机搜索算法，并设计了重要性感知选择策略，从图像贴纸池中为每个圆圈找到最合适的图像贴纸。在四种常见目标检测器上进行的大量实验表明，在多视角场景下，检测性能平均下降了74.29%。此外，还设计了一种基于照片真实感模拟器的评估工具来对基于纹理的攻击进行公平评估。



## **36. Attack GAN (AGAN ): A new Security Evaluation Tool for Perceptual Encryption**

Attack GAN（AGAN）：一种新的感知加密安全评估工具 cs.CV

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06570v1) [paper-pdf](http://arxiv.org/pdf/2407.06570v1)

**Authors**: Umesh Kashyap, Sudev Kumar Padhi, Sk. Subidh Ali

**Abstract**: Training state-of-the-art (SOTA) deep learning models requires a large amount of data. The visual information present in the training data can be misused, which creates a huge privacy concern. One of the prominent solutions for this issue is perceptual encryption, which converts images into an unrecognizable format to protect the sensitive visual information in the training data. This comes at the cost of a significant reduction in the accuracy of the models. Adversarial Visual Information Hiding (AV IH) overcomes this drawback to protect image privacy by attempting to create encrypted images that are unrecognizable to the human eye while keeping relevant features for the target model. In this paper, we introduce the Attack GAN (AGAN ) method, a new Generative Adversarial Network (GAN )-based attack that exposes multiple vulnerabilities in the AV IH method. To show the adaptability, the AGAN is extended to traditional perceptual encryption methods of Learnable encryption (LE) and Encryption-then-Compression (EtC). Extensive experiments were conducted on diverse image datasets and target models to validate the efficacy of our AGAN method. The results show that AGAN can successfully break perceptual encryption methods by reconstructing original images from their AV IH encrypted images. AGAN can be used as a benchmark tool to evaluate the robustness of encryption methods for privacy protection such as AV IH.

摘要: 训练最先进的(SOTA)深度学习模型需要大量数据。训练数据中的视觉信息可能会被滥用，这会造成巨大的隐私问题。针对这一问题的一个突出解决方案是感知加密，它将图像转换为无法识别的格式，以保护训练数据中的敏感视觉信息。这是以显著降低模型精度为代价的。对抗性视觉信息隐藏(AV IH)克服了这一缺点，通过尝试创建人眼无法识别的加密图像来保护图像隐私，同时保留目标模型的相关特征。本文介绍了一种新的基于生成性对抗网络(GAN)的攻击方法--攻击GAN(AGAN)方法，该方法暴露了AVIH方法中的多个漏洞。为了显示其适应性，将AGAN扩展到传统的感知加密方法，如可学习加密(LE)和加密然后压缩(ETC)。在不同的图像数据集和目标模型上进行了广泛的实验，以验证我们的AGaN方法的有效性。结果表明，AGAN能够成功地打破感知加密方法，从他们的AVIH加密图像中重建原始图像。AGAN可以作为一个基准工具来评估用于隐私保护的加密方法的健壮性，例如AV IH。



## **37. DLOVE: A new Security Evaluation Tool for Deep Learning Based Watermarking Techniques**

DLOVE：基于深度学习的水印技术的新安全评估工具 cs.CR

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06552v1) [paper-pdf](http://arxiv.org/pdf/2407.06552v1)

**Authors**: Sudev Kumar Padhi, Sk. Subidh Ali

**Abstract**: Recent developments in Deep Neural Network (DNN) based watermarking techniques have shown remarkable performance. The state-of-the-art DNN-based techniques not only surpass the robustness of classical watermarking techniques but also show their robustness against many image manipulation techniques. In this paper, we performed a detailed security analysis of different DNN-based watermarking techniques. We propose a new class of attack called the Deep Learning-based OVErwriting (DLOVE) attack, which leverages adversarial machine learning and overwrites the original embedded watermark with a targeted watermark in a watermarked image. To the best of our knowledge, this attack is the first of its kind. We have considered scenarios where watermarks are used to devise and formulate an adversarial attack in white box and black box settings. To show adaptability and efficiency, we launch our DLOVE attack analysis on seven different watermarking techniques, HiDDeN, ReDMark, PIMoG, Stegastamp, Aparecium, Distortion Agostic Deep Watermarking and Hiding Images in an Image. All these techniques use different approaches to create imperceptible watermarked images. Our attack analysis on these watermarking techniques with various constraints highlights the vulnerabilities of DNN-based watermarking. Extensive experimental results validate the capabilities of DLOVE. We propose DLOVE as a benchmark security analysis tool to test the robustness of future deep learning-based watermarking techniques.

摘要: 近年来，基于深度神经网络(DNN)的数字水印技术表现出了显著的性能。最新的基于DNN的技术不仅超越了经典水印技术的稳健性，而且表现出对许多图像篡改技术的稳健性。本文对不同的基于DNN的数字水印技术进行了详细的安全性分析。我们提出了一类新的攻击，称为基于深度学习的覆盖攻击(DLOVE)，它利用对抗性机器学习，在水印图像中使用目标水印覆盖原始嵌入的水印。据我们所知，这是此类袭击中的第一次。我们已经考虑了在白盒和黑盒环境中使用水印来设计和制定对抗性攻击的场景。为了显示我们的适应性和效率，我们对七种不同的水印技术进行了DLOVE攻击分析：HIDDEN、ReDMark、PIMoG、Stestamp、Aparecium、抗失真深度水印和图像中的隐藏图像。所有这些技术都使用不同的方法来创建不可感知的水印图像。我们对各种约束条件下的水印技术进行了攻击分析，突出了基于DNN的水印技术的脆弱性。大量的实验结果验证了DLOVE的性能。我们提出了DLOVE作为一个基准安全分析工具来测试未来基于深度学习的水印技术的稳健性。



## **38. WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs**

WildGuard：针对LLC安全风险、越狱和拒绝的开放式一站式审核工具 cs.CL

First two authors contributed equally. Third and fourth authors  contributed equally

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2406.18495v2) [paper-pdf](http://arxiv.org/pdf/2406.18495v2)

**Authors**: Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, Nouha Dziri

**Abstract**: We introduce WildGuard -- an open, light-weight moderation tool for LLM safety that achieves three goals: (1) identifying malicious intent in user prompts, (2) detecting safety risks of model responses, and (3) determining model refusal rate. Together, WildGuard serves the increasing needs for automatic safety moderation and evaluation of LLM interactions, providing a one-stop tool with enhanced accuracy and broad coverage across 13 risk categories. While existing open moderation tools such as Llama-Guard2 score reasonably well in classifying straightforward model interactions, they lag far behind a prompted GPT-4, especially in identifying adversarial jailbreaks and in evaluating models' refusals, a key measure for evaluating safety behaviors in model responses.   To address these challenges, we construct WildGuardMix, a large-scale and carefully balanced multi-task safety moderation dataset with 92K labeled examples that cover vanilla (direct) prompts and adversarial jailbreaks, paired with various refusal and compliance responses. WildGuardMix is a combination of WildGuardTrain, the training data of WildGuard, and WildGuardTest, a high-quality human-annotated moderation test set with 5K labeled items covering broad risk scenarios. Through extensive evaluations on WildGuardTest and ten existing public benchmarks, we show that WildGuard establishes state-of-the-art performance in open-source safety moderation across all the three tasks compared to ten strong existing open-source moderation models (e.g., up to 26.4% improvement on refusal detection). Importantly, WildGuard matches and sometimes exceeds GPT-4 performance (e.g., up to 3.9% improvement on prompt harmfulness identification). WildGuard serves as a highly effective safety moderator in an LLM interface, reducing the success rate of jailbreak attacks from 79.8% to 2.4%.

摘要: 我们介绍了WildGuard--一个开放的、轻量级的LLM安全防御工具，它实现了三个目标：(1)识别用户提示中的恶意意图，(2)检测模型响应的安全风险，(3)确定模型拒绝率。综合起来，WildGuard可满足日益增长的自动安全审核和评估LLM交互作用的需求，提供了一种一站式工具，具有更高的准确性和广泛的覆盖范围，涵盖13个风险类别。虽然现有的开放式审核工具，如Llama-Guard2，在对直接的模型交互进行分类方面得分相当好，但它们远远落后于GPT-4，特别是在识别对抗性越狱和评估模型拒绝方面，这是评估模型响应中安全行为的关键指标。为了应对这些挑战，我们构建了WildGuardMix，这是一个大规模的、仔细平衡的多任务安全缓和数据集，具有92K标记的示例，涵盖普通(直接)提示和对抗性越狱，并与各种拒绝和合规响应配对。WildGuardMix是WildGuard的训练数据WildGuardTrain和WildGuardTest的组合，WildGuardTest是一种高质量的人工注释适度测试集，具有覆盖广泛风险情景的5K标签项目。通过对WildGuardTest和十个现有公共基准的广泛评估，我们表明WildGuard在所有三个任务中建立了开源安全适度的最先进性能，而不是现有的十个强大的开源适度模型(例如，拒绝检测方面高达26.4%的改进)。重要的是，WildGuard的性能与GPT-4相当，有时甚至超过GPT-4(例如，在及时识别危害性方面最高提高3.9%)。WildGuard在LLM界面中充当高效的安全调节器，将越狱攻击的成功率从79.8%降低到2.4%。



## **39. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

利用剩余流激活分析防御大型语言模型免受攻击 cs.CR

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2406.03230v3) [paper-pdf](http://arxiv.org/pdf/2406.03230v3)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply a novel methodology for analyzing distinctive activation patterns in the residual streams for attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.

摘要: 大型语言模型(LLM)的广泛采用，如OpenAI的ChatGPT，使防御这些模型上的对手威胁成为当务之急。这些攻击通过引入恶意输入来操纵LLM的输出，破坏了模型的完整性和用户对其输出的信任。为了应对这一挑战，我们的论文提出了一种创新的防御策略，在白盒访问LLM的情况下，该策略利用LLM变压器层之间的剩余激活分析。我们应用了一种新的方法来分析残留流中独特的激活模式，以进行攻击提示分类。我们精选了多个数据集，以演示此分类方法如何在多种类型的攻击场景中具有高精度，包括我们新创建的攻击数据集。此外，我们通过集成LLMS的安全微调技术来增强模型的弹性，以衡量其对我们检测攻击的能力的影响。这些结果强调了我们的方法在加强对敌对输入的检测和缓解、推进LLMS运作的安全框架方面的有效性。



## **40. Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks**

保护语言模型免受越狱攻击的鲁棒即时优化 cs.LG

Code available at https://github.com/lapisrocks/rpo

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2401.17263v4) [paper-pdf](http://arxiv.org/pdf/2401.17263v4)

**Authors**: Andy Zhou, Bo Li, Haohan Wang

**Abstract**: Despite advances in AI alignment, large language models (LLMs) remain vulnerable to adversarial attacks or jailbreaking, in which adversaries can modify prompts to induce unwanted behavior. While some defenses have been proposed, they have not been adapted to newly proposed attacks and more challenging threat models. To address this, we propose an optimization-based objective for defending LLMs against jailbreaking attacks and an algorithm, Robust Prompt Optimization (RPO) to create robust system-level defenses. Our approach directly incorporates the adversary into the defensive objective and optimizes a lightweight and transferable suffix, enabling RPO to adapt to worst-case adaptive attacks. Our theoretical and experimental results show improved robustness to both jailbreaks seen during optimization and unknown jailbreaks, reducing the attack success rate (ASR) on GPT-4 to 6% and Llama-2 to 0% on JailbreakBench, setting the state-of-the-art. Code can be found at https://github.com/lapisrocks/rpo

摘要: 尽管在人工智能对齐方面取得了进展，但大型语言模型(LLM)仍然容易受到对手攻击或越狱的攻击，在这些攻击或越狱中，对手可以修改提示以诱导不想要的行为。虽然已经提出了一些防御措施，但它们还没有适应新提出的攻击和更具挑战性的威胁模型。为了解决这个问题，我们提出了一个基于优化的目标来保护LLMS免受越狱攻击，并提出了一个算法--稳健提示优化(RPO)来创建强大的系统级防御。我们的方法直接将对手合并到防御目标中，并优化了一个轻量级和可转移的后缀，使RPO能够适应最坏情况的自适应攻击。我们的理论和实验结果表明，对于优化期间看到的越狱和未知越狱，我们都提高了健壮性，将GPT-4上的攻击成功率(ASR)降低到6%，将Llama-2上的攻击成功率降低到0%，从而达到了最先进的水平。代码可在https://github.com/lapisrocks/rpo上找到



## **41. Non-Robust Features are Not Always Useful in One-Class Classification**

非稳健特征在一类分类中并不总是有用 cs.LG

CVPR Visual and Anomaly Detection (VAND) Workshop 2024

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2407.06372v1) [paper-pdf](http://arxiv.org/pdf/2407.06372v1)

**Authors**: Matthew Lau, Haoran Wang, Alec Helbling, Matthew Hul, ShengYun Peng, Martin Andreoni, Willian T. Lunardi, Wenke Lee

**Abstract**: The robustness of machine learning models has been questioned by the existence of adversarial examples. We examine the threat of adversarial examples in practical applications that require lightweight models for one-class classification. Building on Ilyas et al. (2019), we investigate the vulnerability of lightweight one-class classifiers to adversarial attacks and possible reasons for it. Our results show that lightweight one-class classifiers learn features that are not robust (e.g. texture) under stronger attacks. However, unlike in multi-class classification (Ilyas et al., 2019), these non-robust features are not always useful for the one-class task, suggesting that learning these unpredictive and non-robust features is an unwanted consequence of training.

摘要: 机器学习模型的稳健性因对抗性示例的存在而受到质疑。我们研究了实际应用中对抗性示例的威胁，这些应用需要轻量级模型进行一级分类。在Ilyas等人（2019）的基础上，我们研究了轻量级一类分类器对对抗攻击的脆弱性及其可能的原因。我们的结果表明，轻量级一类分类器在更强的攻击下学习不稳健的特征（例如纹理）。然而，与多类分类不同（Ilyas等人，2019年），这些非稳健特征并不总是对一类任务有用，这表明学习这些非预测性和非稳健特征是训练的不想要的结果。



## **42. Improving Alignment and Robustness with Circuit Breakers**

改善断路器的对准和稳健性 cs.LG

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2406.04313v3) [paper-pdf](http://arxiv.org/pdf/2406.04313v3)

**Authors**: Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, Rowan Wang, Zico Kolter, Matt Fredrikson, Dan Hendrycks

**Abstract**: AI systems can take harmful actions and are highly vulnerable to adversarial attacks. We present an approach, inspired by recent advances in representation engineering, that interrupts the models as they respond with harmful outputs with "circuit breakers." Existing techniques aimed at improving alignment, such as refusal training, are often bypassed. Techniques such as adversarial training try to plug these holes by countering specific attacks. As an alternative to refusal training and adversarial training, circuit-breaking directly controls the representations that are responsible for harmful outputs in the first place. Our technique can be applied to both text-only and multimodal language models to prevent the generation of harmful outputs without sacrificing utility -- even in the presence of powerful unseen attacks. Notably, while adversarial robustness in standalone image recognition remains an open challenge, circuit breakers allow the larger multimodal system to reliably withstand image "hijacks" that aim to produce harmful content. Finally, we extend our approach to AI agents, demonstrating considerable reductions in the rate of harmful actions when they are under attack. Our approach represents a significant step forward in the development of reliable safeguards to harmful behavior and adversarial attacks.

摘要: 人工智能系统可能采取有害行动，并且非常容易受到对抗性攻击。我们提出了一种方法，灵感来自于最近在表示工程方面的进展，该方法中断了模型，因为它们用“断路器”来响应有害的输出。旨在改善一致性的现有技术，如拒绝训练，经常被绕过。对抗性训练等技术试图通过反击特定攻击来堵塞这些漏洞。作为拒绝训练和对抗性训练的另一种选择，断路直接控制首先要对有害输出负责的陈述。我们的技术可以应用于纯文本和多模式语言模型，在不牺牲效用的情况下防止产生有害输出-即使在存在强大的看不见的攻击的情况下也是如此。值得注意的是，虽然独立图像识别中的对抗性健壮性仍然是一个开放的挑战，但断路器允许更大的多模式系统可靠地经受住旨在产生有害内容的图像“劫持”。最后，我们将我们的方法扩展到人工智能代理，表明当他们受到攻击时，有害行动的比率大大降低。我们的方法代表着在发展对有害行为和敌对攻击的可靠保障方面向前迈出了重要的一步。



## **43. Adaptive and robust watermark against model extraction attack**

抗模型提取攻击的自适应鲁棒水印 cs.CR

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2405.02365v2) [paper-pdf](http://arxiv.org/pdf/2405.02365v2)

**Authors**: Kaiyi Pang

**Abstract**: Large language models (LLMs) demonstrate general intelligence across a variety of machine learning tasks, thereby enhancing the commercial value of their intellectual property (IP). To protect this IP, model owners typically allow user access only in a black-box manner, however, adversaries can still utilize model extraction attacks to steal the model intelligence encoded in model generation. Watermarking technology offers a promising solution for defending against such attacks by embedding unique identifiers into the model-generated content. However, existing watermarking methods often compromise the quality of generated content due to heuristic alterations and lack robust mechanisms to counteract adversarial strategies, thus limiting their practicality in real-world scenarios. In this paper, we introduce an adaptive and robust watermarking method (named ModelShield) to protect the IP of LLMs. Our method incorporates a self-watermarking mechanism that allows LLMs to autonomously insert watermarks into their generated content to avoid the degradation of model content. We also propose a robust watermark detection mechanism capable of effectively identifying watermark signals under the interference of varying adversarial strategies. Besides, ModelShield is a plug-and-play method that does not require additional model training, enhancing its applicability in LLM deployments. Extensive evaluations on two real-world datasets and three LLMs demonstrate that our method surpasses existing methods in terms of defense effectiveness and robustness while significantly reducing the degradation of watermarking on the model-generated content.

摘要: 大型语言模型(LLM)在各种机器学习任务中展示了一般智能，从而提高了其知识产权(IP)的商业价值。为了保护这个IP，模型所有者通常只允许用户以黑盒方式访问，但是，攻击者仍然可以利用模型提取攻击来窃取模型生成中编码的模型情报。水印技术通过在模型生成的内容中嵌入唯一标识符，为防御此类攻击提供了一种很有前途的解决方案。然而，现有的水印方法往往会由于启发式修改而影响生成内容的质量，并且缺乏强大的机制来对抗对抗性策略，从而限制了它们在现实世界场景中的实用性。本文提出了一种自适应的稳健水印算法(ModelShield)来保护LLMS的IP地址。我们的方法结合了一种自水印机制，允许LLM自主地在其生成的内容中插入水印，以避免模型内容的降级。我们还提出了一种稳健的水印检测机制，能够在不同的对抗策略的干扰下有效地识别水印信号。此外，ModelShield是一种即插即用的方法，不需要额外的模型培训，增强了其在LLM部署中的适用性。在两个真实数据集和三个LLM上的广泛评估表明，我们的方法在防御有效性和稳健性方面优于现有方法，同时显着降低了水印对模型生成内容的退化。



## **44. Multi-View Black-Box Physical Attacks on Infrared Pedestrian Detectors Using Adversarial Infrared Grid**

使用对抗红外网格对红外行人探测器进行多视图黑匣子物理攻击 cs.CV

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2407.01168v2) [paper-pdf](http://arxiv.org/pdf/2407.01168v2)

**Authors**: Kalibinuer Tiliwalidi, Chengyin Hu, Weiwen Shi

**Abstract**: While extensive research exists on physical adversarial attacks within the visible spectrum, studies on such techniques in the infrared spectrum are limited. Infrared object detectors are vital in modern technological applications but are susceptible to adversarial attacks, posing significant security threats. Previous studies using physical perturbations like light bulb arrays and aerogels for white-box attacks, or hot and cold patches for black-box attacks, have proven impractical or limited in multi-view support. To address these issues, we propose the Adversarial Infrared Grid (AdvGrid), which models perturbations in a grid format and uses a genetic algorithm for black-box optimization. These perturbations are cyclically applied to various parts of a pedestrian's clothing to facilitate multi-view black-box physical attacks on infrared pedestrian detectors. Extensive experiments validate AdvGrid's effectiveness, stealthiness, and robustness. The method achieves attack success rates of 80.00\% in digital environments and 91.86\% in physical environments, outperforming baseline methods. Additionally, the average attack success rate exceeds 50\% against mainstream detectors, demonstrating AdvGrid's robustness. Our analyses include ablation studies, transfer attacks, and adversarial defenses, confirming the method's superiority.

摘要: 虽然在可见光光谱内对物理对抗攻击已有广泛的研究，但在红外光谱中对这类技术的研究有限。红外目标探测器在现代技术应用中至关重要，但容易受到对抗性攻击，构成重大安全威胁。以前的研究证明，使用物理扰动，如灯泡阵列和气凝胶进行白盒攻击，或使用冷热补丁进行黑盒攻击，都被证明是不切实际的，或者在多视角支持方面受到限制。为了解决这些问题，我们提出了对抗性红外网格(AdvGrid)，它以网格的形式对扰动进行建模，并使用遗传算法进行黑盒优化。这些扰动被循环应用于行人衣服的不同部分，以促进对红外行人探测器的多视角黑匣子物理攻击。大量实验验证了AdvGrid的有效性、隐蔽性和健壮性。该方法在数字环境下的攻击成功率为80.00%，在物理环境下的攻击成功率为91.86%，优于基准攻击方法。此外，对主流检测器的平均攻击成功率超过50%，显示了AdvGrid的健壮性。我们的分析包括烧蚀研究、转移攻击和对抗性防御，证实了该方法的优越性。



## **45. Malicious Agent Detection for Robust Multi-Agent Collaborative Perception**

用于鲁棒多代理协作感知的恶意代理检测 cs.CR

Accepted by IROS 2024

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2310.11901v2) [paper-pdf](http://arxiv.org/pdf/2310.11901v2)

**Authors**: Yangheng Zhao, Zhen Xiang, Sheng Yin, Xianghe Pang, Siheng Chen, Yanfeng Wang

**Abstract**: Recently, multi-agent collaborative (MAC) perception has been proposed and outperformed the traditional single-agent perception in many applications, such as autonomous driving. However, MAC perception is more vulnerable to adversarial attacks than single-agent perception due to the information exchange. The attacker can easily degrade the performance of a victim agent by sending harmful information from a malicious agent nearby. In this paper, we extend adversarial attacks to an important perception task -- MAC object detection, where generic defenses such as adversarial training are no longer effective against these attacks. More importantly, we propose Malicious Agent Detection (MADE), a reactive defense specific to MAC perception that can be deployed by each agent to accurately detect and then remove any potential malicious agent in its local collaboration network. In particular, MADE inspects each agent in the network independently using a semi-supervised anomaly detector based on a double-hypothesis test with the Benjamini-Hochberg procedure to control the false positive rate of the inference. For the two hypothesis tests, we propose a match loss statistic and a collaborative reconstruction loss statistic, respectively, both based on the consistency between the agent to be inspected and the ego agent where our detector is deployed. We conduct comprehensive evaluations on a benchmark 3D dataset V2X-sim and a real-road dataset DAIR-V2X and show that with the protection of MADE, the drops in the average precision compared with the best-case "oracle" defender against our attack are merely 1.28% and 0.34%, respectively, much lower than 8.92% and 10.00% for adversarial training, respectively.

摘要: 近年来，多智能体协作(MAC)感知被提出，并在许多应用中优于传统的单智能体感知，如自主驾驶。然而，由于信息的交换，MAC感知比单代理感知更容易受到敌意攻击。攻击者可以很容易地通过从附近的恶意代理发送有害信息来降低受害者代理的性能。在本文中，我们将对抗性攻击扩展到一项重要的感知任务--MAC对象检测，在这种情况下，对抗性训练等一般防御手段不再有效地对抗这些攻击。更重要的是，我们提出了恶意代理检测(Made)，这是一种针对MAC感知的反应性防御，可以由每个代理部署以准确检测并随后删除其本地协作网络中的任何潜在恶意代理。特别地，Made使用基于双假设检验的半监督异常检测器独立地检查网络中的每个代理，并结合Benjamini-Hochberg过程来控制推理的误检率。对于这两种假设检验，我们分别提出了一个匹配损失统计量和一个协作重建损失统计量，这两个统计量都是基于待检查代理和部署检测器的自我代理之间的一致性。我们在基准3D数据集V2X-SIM和真实道路数据集DAIR-V2X上进行了综合评估，结果表明，在Made的保护下，与最佳情况下的Oracle防御者相比，对抗我们的攻击的平均精度分别下降了1.28%和0.34%，远低于对抗性训练的8.92%和10.00%。



## **46. Exploring the Adversarial Capabilities of Large Language Models**

探索大型语言模型的对抗能力 cs.AI

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2402.09132v4) [paper-pdf](http://arxiv.org/pdf/2402.09132v4)

**Authors**: Lukas Struppek, Minh Hieu Le, Dominik Hintersdorf, Kristian Kersting

**Abstract**: The proliferation of large language models (LLMs) has sparked widespread and general interest due to their strong language generation capabilities, offering great potential for both industry and research. While previous research delved into the security and privacy issues of LLMs, the extent to which these models can exhibit adversarial behavior remains largely unexplored. Addressing this gap, we investigate whether common publicly available LLMs have inherent capabilities to perturb text samples to fool safety measures, so-called adversarial examples resp.~attacks. More specifically, we investigate whether LLMs are inherently able to craft adversarial examples out of benign samples to fool existing safe rails. Our experiments, which focus on hate speech detection, reveal that LLMs succeed in finding adversarial perturbations, effectively undermining hate speech detection systems. Our findings carry significant implications for (semi-)autonomous systems relying on LLMs, highlighting potential challenges in their interaction with existing systems and safety measures.

摘要: 大型语言模型因其强大的语言生成能力而引起了广泛的关注，为工业和研究提供了巨大的潜力。虽然之前的研究已经深入研究了LLMS的安全和隐私问题，但这些模型在多大程度上可以表现出敌对行为，仍然很大程度上还没有被探索。针对这一差距，我们调查了常见的公开可用的LLM是否具有固有的能力来扰乱文本样本以愚弄安全措施，即所谓的对抗性示例攻击。更具体地说，我们调查LLM是否天生就能够从良性样本中制作敌意示例，以愚弄现有的安全Rail。我们的实验集中在仇恨语音检测上，实验表明，LLMS成功地发现了敌意扰动，有效地破坏了仇恨语音检测系统。我们的发现对依赖LLMS的(半)自治系统具有重大影响，突显了它们与现有系统和安全措施相互作用的潜在挑战。



## **47. Improving Adversarial Transferability of Vision-Language Pre-training Models through Collaborative Multimodal Interaction**

通过协作多模式交互提高视觉语言预训练模型的对抗性可移植性 cs.CV

This work won first place in CVPR 2024 Workshop Challenge: Black-box  Adversarial Attacks on Vision Foundation Models

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2403.10883v2) [paper-pdf](http://arxiv.org/pdf/2403.10883v2)

**Authors**: Jiyuan Fu, Zhaoyu Chen, Kaixun Jiang, Haijing Guo, Jiafeng Wang, Shuyong Gao, Wenqiang Zhang

**Abstract**: Despite the substantial advancements in Vision-Language Pre-training (VLP) models, their susceptibility to adversarial attacks poses a significant challenge. Existing work rarely studies the transferability of attacks on VLP models, resulting in a substantial performance gap from white-box attacks. We observe that prior work overlooks the interaction mechanisms between modalities, which plays a crucial role in understanding the intricacies of VLP models. In response, we propose a novel attack, called Collaborative Multimodal Interaction Attack (CMI-Attack), leveraging modality interaction through embedding guidance and interaction enhancement. Specifically, attacking text at the embedding level while preserving semantics, as well as utilizing interaction image gradients to enhance constraints on perturbations of texts and images. Significantly, in the image-text retrieval task on Flickr30K dataset, CMI-Attack raises the transfer success rates from ALBEF to TCL, $\text{CLIP}_{\text{ViT}}$ and $\text{CLIP}_{\text{CNN}}$ by 8.11%-16.75% over state-of-the-art methods. Moreover, CMI-Attack also demonstrates superior performance in cross-task generalization scenarios. Our work addresses the underexplored realm of transfer attacks on VLP models, shedding light on the importance of modality interaction for enhanced adversarial robustness.

摘要: 尽管视觉语言预训练(VLP)模式有了很大的进步，但它们对对手攻击的敏感性构成了一个巨大的挑战。现有的工作很少研究攻击对VLP模型的可转移性，导致与白盒攻击相比性能有很大的差距。我们注意到，以前的工作忽略了通道之间的相互作用机制，这在理解VLP模型的复杂性方面起着至关重要的作用。对此，我们提出了一种新的攻击方法，称为协作多模式交互攻击(CMI-Attack)，通过嵌入引导和交互增强来利用通道交互。具体地说，在保留语义的同时在嵌入层攻击文本，以及利用交互图像梯度来增强对文本和图像扰动的约束。值得注意的是，在Flickr30K数据集的图文检索任务中，CMI-Attack将从ALBEF到TCL、$\Text{Clip}_{\Text{Vit}}$和$\Text{Clip}_{\Text{CNN}}$的传输成功率比最先进的方法提高了8.11%-16.75%。此外，CMI-Attack在跨任务泛化场景中也表现出了优越的性能。我们的工作解决了VLP模型上未被探索的传输攻击领域，揭示了通道交互对于增强对手健壮性的重要性。



## **48. A Survey of Fragile Model Watermarking**

脆弱模型水印综述 cs.CR

Submitted Signal Processing

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2406.04809v4) [paper-pdf](http://arxiv.org/pdf/2406.04809v4)

**Authors**: Zhenzhe Gao, Yu Cheng, Zhaoxia Yin

**Abstract**: Model fragile watermarking, inspired by both the field of adversarial attacks on neural networks and traditional multimedia fragile watermarking, has gradually emerged as a potent tool for detecting tampering, and has witnessed rapid development in recent years. Unlike robust watermarks, which are widely used for identifying model copyrights, fragile watermarks for models are designed to identify whether models have been subjected to unexpected alterations such as backdoors, poisoning, compression, among others. These alterations can pose unknown risks to model users, such as misidentifying stop signs as speed limit signs in classic autonomous driving scenarios. This paper provides an overview of the relevant work in the field of model fragile watermarking since its inception, categorizing them and revealing the developmental trajectory of the field, thus offering a comprehensive survey for future endeavors in model fragile watermarking.

摘要: 模型脆弱水印受到神经网络对抗攻击领域和传统多媒体脆弱水印的启发，逐渐成为检测篡改的有力工具，并在近年来得到了快速发展。与广泛用于识别模型版权的稳健水印不同，模型的脆弱水印旨在识别模型是否遭受了意外更改，例如后门、中毒、压缩等。这些更改可能会给模型用户带来未知的风险，例如在经典自动驾驶场景中将停车标志误识别为限速标志。本文概述了模型脆弱水印领域自诞生以来的相关工作，对其进行了分类，揭示了该领域的发展轨迹，从而为模型脆弱水印的未来工作提供了全面的综述。



## **49. To Generate or Not? Safety-Driven Unlearned Diffusion Models Are Still Easy To Generate Unsafe Images ... For Now**

生成还是不生成？安全驱动的未学习扩散模型仍然很容易生成不安全的图像.现在 cs.CV

Accepted by ECCV'24. Codes are available at  https://github.com/OPTML-Group/Diffusion-MU-Attack

**SubmitDate**: 2024-07-07    [abs](http://arxiv.org/abs/2310.11868v4) [paper-pdf](http://arxiv.org/pdf/2310.11868v4)

**Authors**: Yimeng Zhang, Jinghan Jia, Xin Chen, Aochuan Chen, Yihua Zhang, Jiancheng Liu, Ke Ding, Sijia Liu

**Abstract**: The recent advances in diffusion models (DMs) have revolutionized the generation of realistic and complex images. However, these models also introduce potential safety hazards, such as producing harmful content and infringing data copyrights. Despite the development of safety-driven unlearning techniques to counteract these challenges, doubts about their efficacy persist. To tackle this issue, we introduce an evaluation framework that leverages adversarial prompts to discern the trustworthiness of these safety-driven DMs after they have undergone the process of unlearning harmful concepts. Specifically, we investigated the adversarial robustness of DMs, assessed by adversarial prompts, when eliminating unwanted concepts, styles, and objects. We develop an effective and efficient adversarial prompt generation approach for DMs, termed UnlearnDiffAtk. This method capitalizes on the intrinsic classification abilities of DMs to simplify the creation of adversarial prompts, thereby eliminating the need for auxiliary classification or diffusion models. Through extensive benchmarking, we evaluate the robustness of widely-used safety-driven unlearned DMs (i.e., DMs after unlearning undesirable concepts, styles, or objects) across a variety of tasks. Our results demonstrate the effectiveness and efficiency merits of UnlearnDiffAtk over the state-of-the-art adversarial prompt generation method and reveal the lack of robustness of current safetydriven unlearning techniques when applied to DMs. Codes are available at https://github.com/OPTML-Group/Diffusion-MU-Attack. WARNING: There exist AI generations that may be offensive in nature.

摘要: 扩散模型的最新进展使逼真和复杂图像的生成发生了革命性的变化。然而，这些模式也带来了潜在的安全隐患，如产生有害内容和侵犯数据著作权。尽管发展了安全驱动的遗忘技术来应对这些挑战，但对其有效性的怀疑依然存在。为了解决这个问题，我们引入了一个评估框架，利用对抗性提示，在这些以安全为导向的DM经历了忘记有害概念的过程后，识别他们的可信度。具体地说，我们研究了DM在消除不需要的概念、风格和对象时，通过对抗性提示评估的对抗性健壮性。本文提出了一种高效的敌意提示生成方法，称为UnlearnDiffAtk。这种方法利用DM的内在分类能力来简化对抗性提示的创建，从而消除了对辅助分类或扩散模型的需要。通过广泛的基准测试，我们评估了广泛使用的安全驱动的未学习DM(即在忘记不需要的概念、风格或对象后的DM)在各种任务中的健壮性。实验结果证明了UnlearnDiffAtk算法相对于最新的对抗性提示生成方法的有效性和高效性，并揭示了当前安全驱动的遗忘技术在应用于决策支持系统时存在的健壮性不足。有关代码，请访问https://github.com/OPTML-Group/Diffusion-MU-Attack.警告：存在可能具有攻击性的人工智能世代。



## **50. Rethinking Targeted Adversarial Attacks For Neural Machine Translation**

重新思考神经机器翻译的有针对性的对抗攻击 cs.CL

5 pages, 2 figures, accepted by ICASSP 2024

**SubmitDate**: 2024-07-07    [abs](http://arxiv.org/abs/2407.05319v1) [paper-pdf](http://arxiv.org/pdf/2407.05319v1)

**Authors**: Junjie Wu, Lemao Liu, Wei Bi, Dit-Yan Yeung

**Abstract**: Targeted adversarial attacks are widely used to evaluate the robustness of neural machine translation systems. Unfortunately, this paper first identifies a critical issue in the existing settings of NMT targeted adversarial attacks, where their attacking results are largely overestimated. To this end, this paper presents a new setting for NMT targeted adversarial attacks that could lead to reliable attacking results. Under the new setting, it then proposes a Targeted Word Gradient adversarial Attack (TWGA) method to craft adversarial examples. Experimental results demonstrate that our proposed setting could provide faithful attacking results for targeted adversarial attacks on NMT systems, and the proposed TWGA method can effectively attack such victim NMT systems. In-depth analyses on a large-scale dataset further illustrate some valuable findings. 1 Our code and data are available at https://github.com/wujunjie1998/TWGA.

摘要: 有针对性的对抗攻击被广泛用于评估神经机器翻译系统的鲁棒性。不幸的是，本文首先指出了NMT有针对性的对抗攻击现有环境中的一个关键问题，即它们的攻击结果在很大程度上被高估了。为此，本文为NMT有针对性的对抗攻击提供了一种新设置，可以带来可靠的攻击结果。在新的设置下，它随后提出了一种有针对性的词梯度对抗攻击（TWGA）方法来制作对抗示例。实验结果表明，我们提出的设置可以为针对NMT系统的有针对性的对抗攻击提供可靠的攻击结果，并且提出的TWGA方法可以有效地攻击此类受害NMT系统。对大规模数据集的深入分析进一步说明了一些有价值的发现。1我们的代码和数据可在https://github.com/wujunjie1998/TWGA上获取。



