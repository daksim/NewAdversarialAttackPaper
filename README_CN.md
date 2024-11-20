# Latest Adversarial Attack Papers
**update at 2024-11-20 11:30:44**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Attribute Inference Attacks for Federated Regression Tasks**

针对联邦回归任务的属性推理攻击 cs.LG

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12697v1) [paper-pdf](http://arxiv.org/pdf/2411.12697v1)

**Authors**: Francesco Diana, Othmane Marfoq, Chuan Xu, Giovanni Neglia, Frédéric Giroire, Eoin Thomas

**Abstract**: Federated Learning (FL) enables multiple clients, such as mobile phones and IoT devices, to collaboratively train a global machine learning model while keeping their data localized. However, recent studies have revealed that the training phase of FL is vulnerable to reconstruction attacks, such as attribute inference attacks (AIA), where adversaries exploit exchanged messages and auxiliary public information to uncover sensitive attributes of targeted clients. While these attacks have been extensively studied in the context of classification tasks, their impact on regression tasks remains largely unexplored. In this paper, we address this gap by proposing novel model-based AIAs specifically designed for regression tasks in FL environments. Our approach considers scenarios where adversaries can either eavesdrop on exchanged messages or directly interfere with the training process. We benchmark our proposed attacks against state-of-the-art methods using real-world datasets. The results demonstrate a significant increase in reconstruction accuracy, particularly in heterogeneous client datasets, a common scenario in FL. The efficacy of our model-based AIAs makes them better candidates for empirically quantifying privacy leakage for federated regression tasks.

摘要: 联合学习(FL)使多个客户端(如移动电话和物联网设备)能够协作训练全球机器学习模型，同时保持其数据的本地化。然而，最近的研究表明，FL的训练阶段容易受到重构攻击，如属性推理攻击(AIA)，即攻击者利用交换的消息和辅助公共信息来发现目标客户的敏感属性。虽然这些攻击已经在分类任务的背景下进行了广泛的研究，但它们对回归任务的影响在很大程度上仍未被探索。在本文中，我们通过提出专门为FL环境中的回归任务设计的新的基于模型的AIAS来解决这一差距。我们的方法考虑了攻击者可以窃听交换的消息或直接干扰训练过程的场景。我们使用真实世界的数据集，根据最先进的方法对我们提出的攻击进行基准测试。结果表明，重建精度显著提高，特别是在异类客户端数据集，这是FL中的常见场景。我们基于模型的AIAS的有效性使它们更适合于经验性地量化联合回归任务的隐私泄露。



## **2. Stochastic BIQA: Median Randomized Smoothing for Certified Blind Image Quality Assessment**

随机BIQA：用于认证盲图像质量评估的随机中位数平滑 eess.IV

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12575v1) [paper-pdf](http://arxiv.org/pdf/2411.12575v1)

**Authors**: Ekaterina Shumitskaya, Mikhail Pautov, Dmitriy Vatolin, Anastasia Antsiferova

**Abstract**: Most modern No-Reference Image-Quality Assessment (NR-IQA) metrics are based on neural networks vulnerable to adversarial attacks. Attacks on such metrics lead to incorrect image/video quality predictions, which poses significant risks, especially in public benchmarks. Developers of image processing algorithms may unfairly increase the score of a target IQA metric without improving the actual quality of the adversarial image. Although some empirical defenses for IQA metrics were proposed, they do not provide theoretical guarantees and may be vulnerable to adaptive attacks. This work focuses on developing a provably robust no-reference IQA metric. Our method is based on Median Smoothing (MS) combined with an additional convolution denoiser with ranking loss to improve the SROCC and PLCC scores of the defended IQA metric. Compared with two prior methods on three datasets, our method exhibited superior SROCC and PLCC scores while maintaining comparable certified guarantees.

摘要: 大多数现代无参考图像质量评估（NR-IQA）指标都基于容易受到对抗攻击的神经网络。对此类指标的攻击会导致图像/视频质量预测错误，从而带来重大风险，尤其是在公共基准中。图像处理算法的开发人员可能会不公平地增加目标IQA指标的分数，而不提高对抗图像的实际质量。尽管提出了一些针对IQA指标的经验防御措施，但它们并不提供理论保证，并且可能容易受到自适应攻击。这项工作的重点是开发一个可证明稳健的无参考IQA指标。我们的方法基于中位数平滑（MS），结合具有排名损失的额外卷积去噪器，以提高受保护的IQA指标的SROCC和PLCC分数。与三个数据集上的两种先前方法相比，我们的方法表现出更好的SROCC和PLCC评分，同时保持了相当的认证保证。



## **3. Variational Bayesian Bow tie Neural Networks with Shrinkage**

具有收缩性的变分Bayesian领结神经网络 stat.ML

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.11132v2) [paper-pdf](http://arxiv.org/pdf/2411.11132v2)

**Authors**: Alisa Sheinkman, Sara Wade

**Abstract**: Despite the dominant role of deep models in machine learning, limitations persist, including overconfident predictions, susceptibility to adversarial attacks, and underestimation of variability in predictions. The Bayesian paradigm provides a natural framework to overcome such issues and has become the gold standard for uncertainty estimation with deep models, also providing improved accuracy and a framework for tuning critical hyperparameters. However, exact Bayesian inference is challenging, typically involving variational algorithms that impose strong independence and distributional assumptions. Moreover, existing methods are sensitive to the architectural choice of the network. We address these issues by constructing a relaxed version of the standard feed-forward rectified neural network, and employing Polya-Gamma data augmentation tricks to render a conditionally linear and Gaussian model. Additionally, we use sparsity-promoting priors on the weights of the neural network for data-driven architectural design. To approximate the posterior, we derive a variational inference algorithm that avoids distributional assumptions and independence across layers and is a faster alternative to the usual Markov Chain Monte Carlo schemes.

摘要: 尽管深度模型在机器学习中起着主导作用，但局限性依然存在，包括过度自信的预测、对对抗性攻击的敏感性以及对预测中的可变性的低估。贝叶斯范式为克服这些问题提供了一个自然的框架，并已成为深度模型不确定性估计的黄金标准，还提供了改进的精度和调整关键超参数的框架。然而，准确的贝叶斯推理是具有挑战性的，通常涉及施加强独立性和分布假设的变分算法。此外，现有的方法对网络的架构选择很敏感。我们通过构造一个松弛版本的标准前馈校正神经网络来解决这些问题，并使用Polya-Gamma数据增强技巧来呈现条件线性和高斯模型。此外，对于数据驱动的建筑设计，我们在神经网络的权值上使用了稀疏性提升的先验。为了逼近后验概率，我们推导了一种变分推理算法，它避免了分布假设和层间独立性，是通常的马尔可夫链蒙特卡罗格式的一个更快的替代方案。



## **4. NMT-Obfuscator Attack: Ignore a sentence in translation with only one word**

NMT-Obfuscator攻击：忽略翻译中只有一个单词的句子 cs.CL

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12473v1) [paper-pdf](http://arxiv.org/pdf/2411.12473v1)

**Authors**: Sahar Sadrizadeh, César Descalzo, Ljiljana Dolamic, Pascal Frossard

**Abstract**: Neural Machine Translation systems are used in diverse applications due to their impressive performance. However, recent studies have shown that these systems are vulnerable to carefully crafted small perturbations to their inputs, known as adversarial attacks. In this paper, we propose a new type of adversarial attack against NMT models. In this attack, we find a word to be added between two sentences such that the second sentence is ignored and not translated by the NMT model. The word added between the two sentences is such that the whole adversarial text is natural in the source language. This type of attack can be harmful in practical scenarios since the attacker can hide malicious information in the automatic translation made by the target NMT model. Our experiments show that different NMT models and translation tasks are vulnerable to this type of attack. Our attack can successfully force the NMT models to ignore the second part of the input in the translation for more than 50% of all cases while being able to maintain low perplexity for the whole input.

摘要: 神经机器翻译系统因其令人印象深刻的性能而被广泛应用。然而，最近的研究表明，这些系统很容易受到精心设计的对其输入的微小扰动，即所谓的对抗性攻击。本文提出了一种新型的针对NMT模型的对抗性攻击。在这种攻击中，我们发现在两个句子之间添加了一个单词，使得第二个句子被忽略，并且不被NMT模型翻译。在两个句子之间添加的单词是这样的，即整个对抗性文本在源语言中是自然的。这种类型的攻击在实际情况下可能是有害的，因为攻击者可以在目标NMT模型所做的自动翻译中隐藏恶意信息。我们的实验表明，不同的NMT模型和翻译任务都容易受到此类攻击。我们的攻击可以成功地迫使NMT模型在超过50%的情况下忽略翻译中输入的第二部分，同时能够保持整个输入的低困惑。



## **5. Efficient Verifiable Differential Privacy with Input Authenticity in the Local and Shuffle Model**

本地和洗牌模型中具有输入真实性的高效可验证差异隐私 cs.CR

21 pages, 13 figures, 2 tables; accepted for publication in the  Proceedings on the 25th Privacy Enhancing Technologies Symposium (PoPETs)  2025

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2406.18940v2) [paper-pdf](http://arxiv.org/pdf/2406.18940v2)

**Authors**: Tariq Bontekoe, Hassan Jameel Asghar, Fatih Turkmen

**Abstract**: Local differential privacy (LDP) enables the efficient release of aggregate statistics without having to trust the central server (aggregator), as in the central model of differential privacy, and simultaneously protects a client's sensitive data. The shuffle model with LDP provides an additional layer of privacy, by disconnecting the link between clients and the aggregator. However, LDP has been shown to be vulnerable to malicious clients who can perform both input and output manipulation attacks, i.e., before and after applying the LDP mechanism, to skew the aggregator's results. In this work, we show how to prevent malicious clients from compromising LDP schemes. Our only realistic assumption is that the initial raw input is authenticated; the rest of the processing pipeline, e.g., formatting the input and applying the LDP mechanism, may be under adversarial control. We give several real-world examples where this assumption is justified. Our proposed schemes for verifiable LDP (VLDP), prevent both input and output manipulation attacks against generic LDP mechanisms, requiring only one-time interaction between client and server, unlike existing alternatives [37, 43]. Most importantly, we are the first to provide an efficient scheme for VLDP in the shuffle model. We describe, and prove security of, two schemes for VLDP in the local model, and one in the shuffle model. We show that all schemes are highly practical, with client run times of less than 2 seconds, and server run times of 5-7 milliseconds per client.

摘要: 本地差异隐私(LDP)支持高效发布汇总统计数据，而不必像差异隐私的中央模型那样信任中央服务器(聚合器)，同时保护客户端的敏感数据。带有LDP的随机模式通过断开客户端和聚合器之间的链路，提供了额外的保密层。然而，LDP已被证明容易受到恶意客户端的攻击，这些客户端可以执行输入和输出操纵攻击，即在应用LDP机制之前和之后，以歪曲聚合器的结果。在这项工作中，我们展示了如何防止恶意客户端危害LDP方案。我们唯一现实的假设是初始原始输入是经过认证的；处理流水线的其余部分，例如格式化输入和应用LDP机制，可能处于敌对控制之下。我们给出了几个真实世界的例子，证明这一假设是合理的。我们提出的可验证LDP方案(VLDP)，防止了针对通用LDP机制的输入和输出操纵攻击，与现有的替代方案不同，它只需要客户端和服务器之间的一次性交互。最重要的是，我们首次在混洗模型中为VLDP提供了一种有效的方案。我们描述并证明了VLDP在局部模型下的两个方案和在置乱模型下的一个方案的安全性。我们证明了所有方案都是非常实用的，客户端运行时间不到2秒，每个客户端的服务器运行时间为5-7毫秒。



## **6. DeTrigger: A Gradient-Centric Approach to Backdoor Attack Mitigation in Federated Learning**

DeTrigger：联邦学习中以用户为中心的后门攻击缓解方法 cs.LG

14 pages

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12220v1) [paper-pdf](http://arxiv.org/pdf/2411.12220v1)

**Authors**: Kichang Lee, Yujin Shin, Jonghyuk Yun, Jun Han, JeongGil Ko

**Abstract**: Federated Learning (FL) enables collaborative model training across distributed devices while preserving local data privacy, making it ideal for mobile and embedded systems. However, the decentralized nature of FL also opens vulnerabilities to model poisoning attacks, particularly backdoor attacks, where adversaries implant trigger patterns to manipulate model predictions. In this paper, we propose DeTrigger, a scalable and efficient backdoor-robust federated learning framework that leverages insights from adversarial attack methodologies. By employing gradient analysis with temperature scaling, DeTrigger detects and isolates backdoor triggers, allowing for precise model weight pruning of backdoor activations without sacrificing benign model knowledge. Extensive evaluations across four widely used datasets demonstrate that DeTrigger achieves up to 251x faster detection than traditional methods and mitigates backdoor attacks by up to 98.9%, with minimal impact on global model accuracy. Our findings establish DeTrigger as a robust and scalable solution to protect federated learning environments against sophisticated backdoor threats.

摘要: 联合学习(FL)支持跨分布式设备进行协作模型培训，同时保护本地数据隐私，使其成为移动和嵌入式系统的理想选择。然而，FL的分散性也为建模中毒攻击打开了漏洞，特别是后门攻击，对手植入触发模式来操纵模型预测。在本文中，我们提出了DeTrigger，一个可扩展的高效后门健壮的联邦学习框架，它利用了对手攻击方法的见解。通过使用带有温度缩放的梯度分析，DeTrigger检测并隔离后门触发器，从而在不牺牲良性模型知识的情况下精确削减后门激活的模型权重。对四个广泛使用的数据集的广泛评估表明，DeTrigger的检测速度比传统方法快251倍，后门攻击减少高达98.9%，对全局模型精度的影响最小。我们的发现将DeTrigger确立为一个强大且可扩展的解决方案，可以保护联合学习环境免受复杂的后门威胁。



## **7. Architectural Patterns for Designing Quantum Artificial Intelligence Systems**

设计量子人工智能系统的架构模式 cs.SE

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.10487v2) [paper-pdf](http://arxiv.org/pdf/2411.10487v2)

**Authors**: Mykhailo Klymenko, Thong Hoang, Xiwei Xu, Zhenchang Xing, Muhammad Usman, Qinghua Lu, Liming Zhu

**Abstract**: Utilising quantum computing technology to enhance artificial intelligence systems is expected to improve training and inference times, increase robustness against noise and adversarial attacks, and reduce the number of parameters without compromising accuracy. However, moving beyond proof-of-concept or simulations to develop practical applications of these systems while ensuring high software quality faces significant challenges due to the limitations of quantum hardware and the underdeveloped knowledge base in software engineering for such systems. In this work, we have conducted a systematic mapping study to identify the challenges and solutions associated with the software architecture of quantum-enhanced artificial intelligence systems. Our review uncovered several architectural patterns that describe how quantum components can be integrated into inference engines, as well as middleware patterns that facilitate communication between classical and quantum components. These insights have been compiled into a catalog of architectural patterns. Each pattern realises a trade-off between efficiency, scalability, trainability, simplicity, portability and deployability, and other software quality attributes.

摘要: 利用量子计算技术来增强人工智能系统，预计将改善训练和推理时间，提高对噪音和对手攻击的稳健性，并在不影响准确性的情况下减少参数数量。然而，由于量子硬件的限制和此类系统的软件工程知识库的不发达，超越概念验证或模拟来开发这些系统的实际应用，同时确保高软件质量面临着重大挑战。在这项工作中，我们进行了系统的映射研究，以确定与量子增强型人工智能系统的软件体系结构相关的挑战和解决方案。我们的审查揭示了几种描述量子组件如何集成到推理引擎中的体系结构模式，以及促进经典组件和量子组件之间通信的中间件模式。这些见解已被汇编成体系结构模式的目录。每个模式都实现了效率、可伸缩性、可训练性、简单性、可移植性和可部署性以及其他软件质量属性之间的权衡。



## **8. Adversarial Multi-Agent Reinforcement Learning for Proactive False Data Injection Detection**

用于主动错误数据注入检测的对抗性多智能体强化学习 eess.SY

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12130v1) [paper-pdf](http://arxiv.org/pdf/2411.12130v1)

**Authors**: Kejun Chen, Truc Nguyen, Malik Hassanaly

**Abstract**: Smart inverters are instrumental in the integration of renewable and distributed energy resources (DERs) into the electric grid. Such inverters rely on communication layers for continuous control and monitoring, potentially exposing them to cyber-physical attacks such as false data injection attacks (FDIAs). We propose to construct a defense strategy against a priori unknown FDIAs with a multi-agent reinforcement learning (MARL) framework. The first agent is an adversary that simulates and discovers various FDIA strategies, while the second agent is a defender in charge of detecting and localizing FDIAs. This approach enables the defender to be trained against new FDIAs continuously generated by the adversary. The numerical results demonstrate that the proposed MARL defender outperforms a supervised offline defender. Additionally, we show that the detection skills of an MARL defender can be combined with that of an offline defender through a transfer learning approach.

摘要: 智能逆变器对于将可再生能源和分布式能源（BER）集成到电网中至关重要。此类逆变器依赖通信层进行持续控制和监控，可能会使它们面临虚假数据注入攻击（FDIA）等网络物理攻击。我们建议通过多智能体强化学习（MARL）框架构建针对先验未知FDIA的防御策略。第一个代理是模拟和发现各种FDIA策略的对手，而第二个代理是负责检测和定位FDIA的防御者。这种方法使防御者能够针对对手不断产生的新FDIA进行训练。数值结果表明，提出的MARL防御器优于有监督的离线防御器。此外，我们还表明，MARL防守者的检测技能可以通过迁移学习方法与离线防守者的检测技能相结合。



## **9. Theoretical Corrections and the Leveraging of Reinforcement Learning to Enhance Triangle Attack**

理论修正和利用强化学习增强三角攻击 cs.LG

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.12071v1) [paper-pdf](http://arxiv.org/pdf/2411.12071v1)

**Authors**: Nicole Meng, Caleb Manicke, David Chen, Yingjie Lao, Caiwen Ding, Pengyu Hong, Kaleel Mahmood

**Abstract**: Adversarial examples represent a serious issue for the application of machine learning models in many sensitive domains. For generating adversarial examples, decision based black-box attacks are one of the most practical techniques as they only require query access to the model. One of the most recently proposed state-of-the-art decision based black-box attacks is Triangle Attack (TA). In this paper, we offer a high-level description of TA and explain potential theoretical limitations. We then propose a new decision based black-box attack, Triangle Attack with Reinforcement Learning (TARL). Our new attack addresses the limits of TA by leveraging reinforcement learning. This creates an attack that can achieve similar, if not better, attack accuracy than TA with half as many queries on state-of-the-art classifiers and defenses across ImageNet and CIFAR-10.

摘要: 对抗性示例代表了机器学习模型在许多敏感领域的应用的一个严重问题。对于生成对抗性示例，基于决策的黑匣子攻击是最实用的技术之一，因为它们只需要对模型进行查询访问。最近提出的最先进的基于决策的黑匣子攻击之一是三角攻击（TA）。在本文中，我们对TA进行了高级描述并解释了潜在的理论局限性。然后，我们提出了一种新的基于决策的黑匣子攻击，即带强化学习的三角攻击（TARL）。我们的新攻击通过利用强化学习来解决TA的局限性。这会创建一种攻击，它可以实现与TA类似（甚至更好）的攻击准确性，只需对ImageNet和CIFAR-10中最先进的分类器和防御系统进行一半的查询。



## **10. Exploring adversarial robustness of JPEG AI: methodology, comparison and new methods**

探索JPEG AI的对抗鲁棒性：方法论、比较和新方法 eess.IV

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11795v1) [paper-pdf](http://arxiv.org/pdf/2411.11795v1)

**Authors**: Egor Kovalev, Georgii Bychkov, Khaled Abud, Aleksandr Gushchin, Anna Chistyakova, Sergey Lavrushkin, Dmitriy Vatolin, Anastasia Antsiferova

**Abstract**: Adversarial robustness of neural networks is an increasingly important area of research, combining studies on computer vision models, large language models (LLMs), and others. With the release of JPEG AI - the first standard for end-to-end neural image compression (NIC) methods - the question of its robustness has become critically significant. JPEG AI is among the first international, real-world applications of neural-network-based models to be embedded in consumer devices. However, research on NIC robustness has been limited to open-source codecs and a narrow range of attacks. This paper proposes a new methodology for measuring NIC robustness to adversarial attacks. We present the first large-scale evaluation of JPEG AI's robustness, comparing it with other NIC models. Our evaluation results and code are publicly available online (link is hidden for a blind review).

摘要: 神经网络的对抗鲁棒性是一个越来越重要的研究领域，结合了对计算机视觉模型、大型语言模型（LLM）等的研究。随着JPEG AI（端到端神经图像压缩（NIC）方法的第一个标准）的发布，其稳健性问题变得至关重要。JPEG AI是首批嵌入消费设备的基于神经网络的模型的国际现实应用之一。然而，关于NIC稳健性的研究仅限于开源编解码器和范围狭窄的攻击。本文提出了一种新的方法来衡量NIC对对抗性攻击的稳健性。我们首次对JPEG AI的稳健性进行了大规模评估，并将其与其他NIC模型进行了比较。我们的评估结果和代码可在线公开（链接已隐藏，以供盲目审查）。



## **11. Robust Subgraph Learning by Monitoring Early Training Representations**

通过监控早期训练表示进行稳健的子图学习 cs.LG

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2403.09901v2) [paper-pdf](http://arxiv.org/pdf/2403.09901v2)

**Authors**: Sepideh Neshatfar, Salimeh Yasaei Sekeh

**Abstract**: Graph neural networks (GNNs) have attracted significant attention for their outstanding performance in graph learning and node classification tasks. However, their vulnerability to adversarial attacks, particularly through susceptible nodes, poses a challenge in decision-making. The need for robust graph summarization is evident in adversarial challenges resulting from the propagation of attacks throughout the entire graph. In this paper, we address both performance and adversarial robustness in graph input by introducing the novel technique SHERD (Subgraph Learning Hale through Early Training Representation Distances). SHERD leverages information from layers of a partially trained graph convolutional network (GCN) to detect susceptible nodes during adversarial attacks using standard distance metrics. The method identifies "vulnerable (bad)" nodes and removes such nodes to form a robust subgraph while maintaining node classification performance. Through our experiments, we demonstrate the increased performance of SHERD in enhancing robustness by comparing the network's performance on original and subgraph inputs against various baselines alongside existing adversarial attacks. Our experiments across multiple datasets, including citation datasets such as Cora, Citeseer, and Pubmed, as well as microanatomical tissue structures of cell graphs in the placenta, highlight that SHERD not only achieves substantial improvement in robust performance but also outperforms several baselines in terms of node classification accuracy and computational complexity.

摘要: 图神经网络(GNN)因其在图学习和节点分类任务中的优异性能而备受关注。然而，它们对敌意攻击的脆弱性，特别是通过易受攻击的节点，对决策构成了挑战。在攻击在整个图中传播所导致的对抗性挑战中，对健壮图摘要的需求是显而易见的。在本文中，我们通过引入新的技术SHERD(子图通过早期训练表示距离学习Hale)来解决图形输入中的性能和对手健壮性。SHERD利用来自部分训练的图卷积网络(GCN)各层的信息，使用标准距离度量在敌意攻击期间检测易受攻击的节点。该方法在保持节点分类性能的同时，识别“易受攻击(坏)”的节点，并删除这些节点以形成一个健壮的子图。通过我们的实验，我们通过比较网络在原始和子图输入上的性能与不同基线的性能以及现有的对抗性攻击，证明了SHERD在增强健壮性方面的性能提高。我们在多个数据集上的实验，包括引用数据集，如Cora，Citeseer和Pubmed，以及胎盘细胞图的显微解剖组织结构，突出了Sherd不仅在健壮性性能方面取得了实质性的改进，而且在节点分类精度和计算复杂性方面也超过了几个基线。



## **12. Eidos: Efficient, Imperceptible Adversarial 3D Point Clouds**

Eidos：高效、不可感知的对抗性3D点云 cs.CV

Preprint

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2405.14210v2) [paper-pdf](http://arxiv.org/pdf/2405.14210v2)

**Authors**: Hanwei Zhang, Luo Cheng, Qisong He, Wei Huang, Renjue Li, Ronan Sicre, Xiaowei Huang, Holger Hermanns, Lijun Zhang

**Abstract**: Classification of 3D point clouds is a challenging machine learning (ML) task with important real-world applications in a spectrum from autonomous driving and robot-assisted surgery to earth observation from low orbit. As with other ML tasks, classification models are notoriously brittle in the presence of adversarial attacks. These are rooted in imperceptible changes to inputs with the effect that a seemingly well-trained model ends up misclassifying the input. This paper adds to the understanding of adversarial attacks by presenting Eidos, a framework providing Efficient Imperceptible aDversarial attacks on 3D pOint cloudS. Eidos supports a diverse set of imperceptibility metrics. It employs an iterative, two-step procedure to identify optimal adversarial examples, thereby enabling a runtime-imperceptibility trade-off. We provide empirical evidence relative to several popular 3D point cloud classification models and several established 3D attack methods, showing Eidos' superiority with respect to efficiency as well as imperceptibility.

摘要: 三维点云的分类是一项具有挑战性的机器学习(ML)任务，在从自动驾驶和机器人辅助手术到低轨道对地观测等一系列实际应用中具有重要的应用。与其他ML任务一样，分类模型在存在对抗性攻击时是出了名的脆弱。这些问题根源于对投入的潜移默化的改变，其结果是，一个看似训练有素的模型最终会错误地对投入进行分类。本文通过介绍EIDOS来加深对敌意攻击的理解，EIDOS是一种在3D点云上提供高效的隐形攻击的框架。Eidos支持一组不同的不可感知性指标。它使用迭代的两步过程来确定最佳对抗性示例，从而实现了运行时不可感知性的权衡。我们提供了与几种流行的三维点云分类模型和几种已建立的三维攻击方法相关的经验证据，表明了Eidos在效率和不可感知性方面的优势。



## **13. Bitcoin Under Volatile Block Rewards: How Mempool Statistics Can Influence Bitcoin Mining**

波动性区块奖励下的比特币：Mempool统计数据如何影响比特币采矿 cs.CR

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11702v1) [paper-pdf](http://arxiv.org/pdf/2411.11702v1)

**Authors**: Roozbeh Sarenche, Alireza Aghabagherloo, Svetla Nikova, Bart Preneel

**Abstract**: As Bitcoin experiences more halving events, the protocol reward converges to zero, making transaction fees the primary source of miner rewards. This shift in Bitcoin's incentivization mechanism, which introduces volatility into block rewards, could lead to the emergence of new security threats or intensify existing ones. Previous security analyses of Bitcoin have either considered a fixed block reward model or a highly simplified volatile model, overlooking the complexities of Bitcoin's mempool behavior.   In this paper, we present a reinforcement learning-based tool designed to analyze mining strategies under a more realistic volatile model. Our tool uses the Asynchronous Advantage Actor-Critic (A3C) algorithm to derive near-optimal mining strategies while interacting with an environment that models the complexity of the Bitcoin mempool. This tool enables the analysis of adversarial mining strategies, such as selfish mining and undercutting, both before and after difficulty adjustments, providing insights into the effects of mining attacks in both the short and long term.   Our analysis reveals that Bitcoin users' trend of offering higher fees to speed up the inclusion of their transactions in the chain can incentivize payoff-maximizing miners to deviate from the honest strategy. In the fixed reward model, a disincentive for the selfish mining attack is the initial loss period of at least two weeks, during which the attack is not profitable. However, our analysis shows that once the protocol reward diminishes to zero in the future, or even currently on days when transaction fees are comparable to the protocol reward, mining pools might be incentivized to abandon honest mining to gain an immediate profit.

摘要: 随着比特币经历更多减半事件，协议奖励趋于零，使交易费成为矿工奖励的主要来源。比特币激励机制的这种转变，在大宗奖励中引入了波动性，可能会导致新的安全威胁的出现，或者加剧现有的安全威胁。此前对比特币的安全分析要么考虑了固定的区块奖励模型，要么考虑了高度简化的波动性模型，忽视了比特币成员池行为的复杂性。在本文中，我们提出了一个基于强化学习的工具，用于在更真实的易变模型下分析挖掘策略。我们的工具使用异步优势参与者-批评者(A3C)算法来推导出接近最优的挖掘策略，同时与模拟比特币记忆池复杂性的环境交互。这一工具能够分析难度调整前后的对抗性采矿战略，如自私采矿和削价，从而深入了解采矿攻击在短期和长期的影响。我们的分析显示，比特币用户提供更高费用以加快将他们的交易纳入链中的趋势，可以激励收益最大化的矿工偏离诚实策略。在固定报酬模型中，对自私挖矿攻击的抑制是至少两周的初始损失期，在此期间攻击是不盈利的。然而，我们的分析表明，一旦协议奖励在未来减少到零，甚至目前在交易费与协议奖励相当的日子里，采矿池可能会受到激励，放弃诚实的开采，以获得直接利润。



## **14. TrojanRobot: Backdoor Attacks Against Robotic Manipulation in the Physical World**

特洛伊机器人：针对物理世界中机器人操纵的后门攻击 cs.RO

Initial version with preliminary results. We welcome any feedback or  suggestions

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11683v1) [paper-pdf](http://arxiv.org/pdf/2411.11683v1)

**Authors**: Xianlong Wang, Hewen Pan, Hangtao Zhang, Minghui Li, Shengshan Hu, Ziqi Zhou, Lulu Xue, Peijin Guo, Yichen Wang, Wei Wan, Aishan Liu, Leo Yu Zhang

**Abstract**: Robotic manipulation refers to the autonomous handling and interaction of robots with objects using advanced techniques in robotics and artificial intelligence. The advent of powerful tools such as large language models (LLMs) and large vision-language models (LVLMs) has significantly enhanced the capabilities of these robots in environmental perception and decision-making. However, the introduction of these intelligent agents has led to security threats such as jailbreak attacks and adversarial attacks.   In this research, we take a further step by proposing a backdoor attack specifically targeting robotic manipulation and, for the first time, implementing backdoor attack in the physical world. By embedding a backdoor visual language model into the visual perception module within the robotic system, we successfully mislead the robotic arm's operation in the physical world, given the presence of common items as triggers. Experimental evaluations in the physical world demonstrate the effectiveness of the proposed backdoor attack.

摘要: 机器人操纵是指使用机器人学和人工智能的先进技术，自主处理机器人与物体的交互。大型语言模型(LLM)和大型视觉语言模型(LVLM)等强大工具的出现，大大增强了这些机器人在环境感知和决策方面的能力。然而，这些智能代理的引入导致了越狱攻击和对抗性攻击等安全威胁。在这项研究中，我们进一步提出了专门针对机器人操作的后门攻击，并首次在物理世界中实现了后门攻击。通过将后门视觉语言模型嵌入机器人系统的视觉感知模块中，我们成功地误导了机械臂在物理世界中的操作，因为存在共同的物品作为触发器。物理世界中的实验评估证明了所提出的后门攻击的有效性。



## **15. Few-shot Model Extraction Attacks against Sequential Recommender Systems**

针对顺序推荐系统的少镜头模型提取攻击 cs.LG

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11677v1) [paper-pdf](http://arxiv.org/pdf/2411.11677v1)

**Authors**: Hui Zhang, Fu Liu

**Abstract**: Among adversarial attacks against sequential recommender systems, model extraction attacks represent a method to attack sequential recommendation models without prior knowledge. Existing research has primarily concentrated on the adversary's execution of black-box attacks through data-free model extraction. However, a significant gap remains in the literature concerning the development of surrogate models by adversaries with access to few-shot raw data (10\% even less). That is, the challenge of how to construct a surrogate model with high functional similarity within the context of few-shot data scenarios remains an issue that requires resolution.This study addresses this gap by introducing a novel few-shot model extraction framework against sequential recommenders, which is designed to construct a superior surrogate model with the utilization of few-shot data. The proposed few-shot model extraction framework is comprised of two components: an autoregressive augmentation generation strategy and a bidirectional repair loss-facilitated model distillation procedure. Specifically, to generate synthetic data that closely approximate the distribution of raw data, autoregressive augmentation generation strategy integrates a probabilistic interaction sampler to extract inherent dependencies and a synthesis determinant signal module to characterize user behavioral patterns. Subsequently, bidirectional repair loss, which target the discrepancies between the recommendation lists, is designed as auxiliary loss to rectify erroneous predictions from surrogate models, transferring knowledge from the victim model to the surrogate model effectively. Experiments on three datasets show that the proposed few-shot model extraction framework yields superior surrogate models.

摘要: 在针对序列推荐系统的对抗性攻击中，模型提取攻击是一种在没有先验知识的情况下攻击序列推荐模型的方法。现有的研究主要集中在对手通过无数据模型提取来执行黑盒攻击。然而，关于对手开发代理模型的文献中仍然存在着一个显著的差距，这些对手可以访问很少的原始数据(10\%甚至更少)。如何在稀疏数据场景下构建功能相似度高的代理模型是一个亟待解决的问题，本研究通过引入一种针对顺序推荐者的稀疏模型提取框架来解决这一问题，该框架旨在利用稀疏数据构建一个更优的代理模型。所提出的少镜头模型提取框架由两部分组成：自回归增广生成策略和双向修复损失促进模型精馏过程。具体地说，为了生成接近原始数据分布的合成数据，自回归增强生成策略集成了一个概率交互采样器来提取固有依赖关系和一个合成行列式信号模块来表征用户行为模式。随后，针对推荐列表之间的差异，设计了双向修复损失作为辅助损失来纠正代理模型中的错误预测，有效地将受害者模型中的知识传递到代理模型中。在三个数据集上的实验表明，所提出的少镜头模型提取框架产生了更好的代理模型。



## **16. Formal Verification of Deep Neural Networks for Object Detection**

用于对象检测的深度神经网络的形式化验证 cs.CV

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2407.01295v5) [paper-pdf](http://arxiv.org/pdf/2407.01295v5)

**Authors**: Yizhak Y. Elboher, Avraham Raviv, Yael Leibovich Weiss, Omer Cohen, Roy Assa, Guy Katz, Hillel Kugler

**Abstract**: Deep neural networks (DNNs) are widely used in real-world applications, yet they remain vulnerable to errors and adversarial attacks. Formal verification offers a systematic approach to identify and mitigate these vulnerabilities, enhancing model robustness and reliability. While most existing verification methods focus on image classification models, this work extends formal verification to the more complex domain of emph{object detection} models. We propose a formulation for verifying the robustness of such models and demonstrate how state-of-the-art verification tools, originally developed for classification, can be adapted for this purpose. Our experiments, conducted on various datasets and networks, highlight the ability of formal verification to uncover vulnerabilities in object detection models, underscoring the need to extend verification efforts to this domain. This work lays the foundation for further research into formal verification across a broader range of computer vision applications.

摘要: 深度神经网络(DNN)在实际应用中得到了广泛的应用，但它们仍然容易受到错误和敌意攻击。正式验证提供了一种系统的方法来识别和缓解这些漏洞，从而增强了模型的健壮性和可靠性。虽然现有的验证方法大多集中在图像分类模型上，但该工作将形式验证扩展到更复杂的领域，即目标检测模型。我们提出了一种验证此类模型的稳健性的公式，并演示了最初为分类而开发的最先进的验证工具如何适用于此目的。我们在各种数据集和网络上进行的实验，突出了正式验证发现对象检测模型中漏洞的能力，强调了将验证工作扩展到这一领域的必要性。这项工作为在更广泛的计算机视觉应用中进一步研究形式验证奠定了基础。



## **17. The Dark Side of Trust: Authority Citation-Driven Jailbreak Attacks on Large Language Models**

信任的阴暗面：权威引用驱动的对大型语言模型的越狱攻击 cs.LG

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11407v1) [paper-pdf](http://arxiv.org/pdf/2411.11407v1)

**Authors**: Xikang Yang, Xuehai Tang, Jizhong Han, Songlin Hu

**Abstract**: The widespread deployment of large language models (LLMs) across various domains has showcased their immense potential while exposing significant safety vulnerabilities. A major concern is ensuring that LLM-generated content aligns with human values. Existing jailbreak techniques reveal how this alignment can be compromised through specific prompts or adversarial suffixes. In this study, we introduce a new threat: LLMs' bias toward authority. While this inherent bias can improve the quality of outputs generated by LLMs, it also introduces a potential vulnerability, increasing the risk of producing harmful content. Notably, the biases in LLMs is the varying levels of trust given to different types of authoritative information in harmful queries. For example, malware development often favors trust GitHub. To better reveal the risks with LLM, we propose DarkCite, an adaptive authority citation matcher and generator designed for a black-box setting. DarkCite matches optimal citation types to specific risk types and generates authoritative citations relevant to harmful instructions, enabling more effective jailbreak attacks on aligned LLMs.Our experiments show that DarkCite achieves a higher attack success rate (e.g., LLama-2 at 76% versus 68%) than previous methods. To counter this risk, we propose an authenticity and harm verification defense strategy, raising the average defense pass rate (DPR) from 11% to 74%. More importantly, the ability to link citations to the content they encompass has become a foundational function in LLMs, amplifying the influence of LLMs' bias toward authority.

摘要: 大型语言模型(LLM)在不同领域的广泛部署展示了它们的巨大潜力，同时也暴露了重大的安全漏洞。一个主要的问题是确保LLM生成的内容符合人类的价值观。现有的越狱技术揭示了如何通过特定的提示或对抗性后缀来破坏这种对齐。在这项研究中，我们引入了一个新的威胁：LLMS对权威的偏见。虽然这种固有的偏见可以提高低成本管理产生的产出的质量，但它也引入了一个潜在的脆弱性，增加了产生有害内容的风险。值得注意的是，LLMS中的偏差是在有害查询中对不同类型的权威信息给予的不同程度的信任。例如，恶意软件开发通常偏向信任GitHub。为了更好地揭示LLM的风险，我们提出了DarkCite，这是一个为黑箱设置而设计的自适应权威引用匹配器和生成器。DarkCite将最佳引用类型与特定的风险类型相匹配，并生成与有害指令相关的权威引用，从而对对齐的LLMS进行更有效的越狱攻击。我们的实验表明，与以前的方法相比，DarkCite实现了更高的攻击成功率(例如，骆驼-2为76%，而不是68%)。为了应对这种风险，我们提出了真实性和危害性验证防御策略，将平均防御通过率(DPR)从11%提高到74%。更重要的是，将引文与它们所包含的内容相联系的能力已经成为LLMS的一项基本功能，放大了LLMS对权威的偏见的影响。



## **18. Hacking Back the AI-Hacker: Prompt Injection as a Defense Against LLM-driven Cyberattacks**

黑客攻击人工智能黑客：即时注入作为抵御LLM驱动的网络攻击的防御 cs.CR

v0.2 (evaluated on more agents)

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2410.20911v2) [paper-pdf](http://arxiv.org/pdf/2410.20911v2)

**Authors**: Dario Pasquini, Evgenios M. Kornaropoulos, Giuseppe Ateniese

**Abstract**: Large language models (LLMs) are increasingly being harnessed to automate cyberattacks, making sophisticated exploits more accessible and scalable. In response, we propose a new defense strategy tailored to counter LLM-driven cyberattacks. We introduce Mantis, a defensive framework that exploits LLMs' susceptibility to adversarial inputs to undermine malicious operations. Upon detecting an automated cyberattack, Mantis plants carefully crafted inputs into system responses, leading the attacker's LLM to disrupt their own operations (passive defense) or even compromise the attacker's machine (active defense). By deploying purposefully vulnerable decoy services to attract the attacker and using dynamic prompt injections for the attacker's LLM, Mantis can autonomously hack back the attacker. In our experiments, Mantis consistently achieved over 95% effectiveness against automated LLM-driven attacks. To foster further research and collaboration, Mantis is available as an open-source tool: https://github.com/pasquini-dario/project_mantis

摘要: 大型语言模型(LLM)越来越多地被用来自动化网络攻击，使复杂的利用更容易获得和可扩展。作为回应，我们提出了一种新的防御战略，以对抗LLM驱动的网络攻击。我们引入了Mantis，这是一个防御框架，利用LLMS对对手输入的敏感性来破坏恶意操作。在检测到自动网络攻击后，螳螂工厂会精心设计输入到系统响应中，导致攻击者的LLM扰乱自己的操作(被动防御)，甚至危害攻击者的机器(主动防御)。通过部署故意易受攻击的诱骗服务来吸引攻击者，并对攻击者的LLM使用动态提示注入，螳螂可以自主地攻击攻击者。在我们的实验中，螳螂对自动LLM驱动的攻击始终取得了95%以上的效率。为了促进进一步的研究和合作，Mantis以开源工具的形式提供：https://github.com/pasquini-dario/project_mantis



## **19. Adapting to Cyber Threats: A Phishing Evolution Network (PEN) Framework for Phishing Generation and Analyzing Evolution Patterns using Large Language Models**

适应网络威胁：用于使用大型语言模型进行网络钓鱼生成和分析进化模式的网络钓鱼进化网络（PEN）框架 cs.CR

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11389v1) [paper-pdf](http://arxiv.org/pdf/2411.11389v1)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Shuo Wang, Hongsheng Hu, Alsharif Abuadbba, Carsten Rudolph

**Abstract**: Phishing remains a pervasive cyber threat, as attackers craft deceptive emails to lure victims into revealing sensitive information. While Artificial Intelligence (AI), particularly deep learning, has become a key component in defending against phishing attacks, these approaches face critical limitations. The scarcity of publicly available, diverse, and updated data, largely due to privacy concerns, constrains their effectiveness. As phishing tactics evolve rapidly, models trained on limited, outdated data struggle to detect new, sophisticated deception strategies, leaving systems vulnerable to an ever-growing array of attacks. Addressing this gap is essential to strengthening defenses in an increasingly hostile cyber landscape. To address this gap, we propose the Phishing Evolution Network (PEN), a framework leveraging large language models (LLMs) and adversarial training mechanisms to continuously generate high quality and realistic diverse phishing samples, and analyze features of LLM-provided phishing to understand evolving phishing patterns. We evaluate the quality and diversity of phishing samples generated by PEN and find that it produces over 80% realistic phishing samples, effectively expanding phishing datasets across seven dominant types. These PEN-generated samples enhance the performance of current phishing detectors, leading to a 40% improvement in detection accuracy. Additionally, the use of PEN significantly boosts model robustness, reducing detectors' sensitivity to perturbations by up to 60%, thereby decreasing attack success rates under adversarial conditions. When we analyze the phishing patterns that are used in LLM-generated phishing, the cognitive complexity and the tone of time limitation are detected with statistically significant differences compared with existing phishing.

摘要: 网络钓鱼仍然是一个普遍存在的网络威胁，因为攻击者精心制作了欺骗性电子邮件，以引诱受害者泄露敏感信息。虽然人工智能(AI)，特别是深度学习，已经成为防御网络钓鱼攻击的关键组件，但这些方法面临着严重的限制。由于缺乏公开可用的、多样化的和更新的数据，这主要是由于隐私问题，限制了它们的有效性。随着钓鱼策略的快速发展，基于有限、过时数据的模型很难检测出新的、复杂的欺骗策略，这使得系统容易受到越来越多的攻击。在日益充满敌意的网络环境中，解决这一差距对于加强防御至关重要。为了弥补这一差距，我们提出了钓鱼进化网络(PEN)，这是一个利用大型语言模型(LLMS)和对手训练机制来持续生成高质量和真实的多样化钓鱼样本的框架，并分析LLM提供的钓鱼特征以了解不断演变的钓鱼模式。我们评估了PEN生成的网络钓鱼样本的质量和多样性，发现它产生了超过80%的真实网络钓鱼样本，有效地扩展了七种主要类型的网络钓鱼数据集。这些笔生成的样本增强了当前网络钓鱼检测器的性能，导致检测准确率提高了40%。此外，PEN的使用显著提高了模型的稳健性，将检测器对扰动的敏感度降低了高达60%，从而降低了对抗性条件下的攻击成功率。当我们分析LLM生成的网络钓鱼中使用的网络钓鱼模式时，我们检测到了认知复杂性和时间限制的基调，与现有的网络钓鱼相比具有统计学意义上的差异。



## **20. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

卡西姆·分歧：通过对抗性防御的扩散模型来启发性解纠缠 cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2410.23091v4) [paper-pdf](http://arxiv.org/pdf/2410.23091v4)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark). The code is available at \href{https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff}{https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff}

摘要: 尽管不断努力保护神经分类器免受对手攻击，但它们仍然很脆弱，特别是面对看不见的攻击。相比之下，人类很难被微妙的操纵所欺骗，因为我们只根据基本因素做出判断。受到这一观察的启发，我们试图用基本的标签原因因素来建模标签生成，并结合标签非原因因素来辅助数据生成。对于一个对抗性的例子，我们的目标是将扰动区分为非致因因素，并仅基于标签致因因素进行预测。具体地说，我们提出了一个偶然扩散模型(CausalDiff)，该模型使扩散模型适用于条件数据生成，并通过向一个新的偶然信息瓶颈目标学习来区分这两种类型的偶然因素。经验上，CausalDiff在各种隐形攻击上的表现明显优于最先进的防御方法，在CIFAR-10上获得了86.39%(+4.01%)的平均健壮性，在CIFAR-100上获得了56.25%(+3.13%)的健壮性，在GTSRB(德国交通标志识别基准)上实现了82.62%(+4.93%)的平均健壮性。代码可在\href{https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff}{https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff}上获得



## **21. Countering Backdoor Attacks in Image Recognition: A Survey and Evaluation of Mitigation Strategies**

对抗图像识别中的后门攻击：缓解策略的调查和评估 cs.CR

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2411.11200v1) [paper-pdf](http://arxiv.org/pdf/2411.11200v1)

**Authors**: Kealan Dunnett, Reza Arablouei, Dimity Miller, Volkan Dedeoglu, Raja Jurdak

**Abstract**: The widespread adoption of deep learning across various industries has introduced substantial challenges, particularly in terms of model explainability and security. The inherent complexity of deep learning models, while contributing to their effectiveness, also renders them susceptible to adversarial attacks. Among these, backdoor attacks are especially concerning, as they involve surreptitiously embedding specific triggers within training data, causing the model to exhibit aberrant behavior when presented with input containing the triggers. Such attacks often exploit vulnerabilities in outsourced processes, compromising model integrity without affecting performance on clean (trigger-free) input data. In this paper, we present a comprehensive review of existing mitigation strategies designed to counter backdoor attacks in image recognition. We provide an in-depth analysis of the theoretical foundations, practical efficacy, and limitations of these approaches. In addition, we conduct an extensive benchmarking of sixteen state-of-the-art approaches against eight distinct backdoor attacks, utilizing three datasets, four model architectures, and three poisoning ratios. Our results, derived from 122,236 individual experiments, indicate that while many approaches provide some level of protection, their performance can vary considerably. Furthermore, when compared to two seminal approaches, most newer approaches do not demonstrate substantial improvements in overall performance or consistency across diverse settings. Drawing from these findings, we propose potential directions for developing more effective and generalizable defensive mechanisms in the future.

摘要: 深度学习在各个行业的广泛采用带来了巨大的挑战，特别是在模型的可解释性和安全性方面。深度学习模型固有的复杂性，虽然有助于它们的有效性，但也使它们容易受到对手的攻击。其中，后门攻击尤其令人担忧，因为它们涉及在训练数据中秘密嵌入特定触发器，导致在输入包含触发器的输入时导致模型表现出异常行为。此类攻击通常利用外包流程中的漏洞，在不影响干净(无触发器)输入数据的性能的情况下损害模型完整性。在这篇文章中，我们提出了一个全面的审查现有的缓解策略，旨在对抗后门攻击的图像识别。我们对这些方法的理论基础、实践有效性和局限性进行了深入分析。此外，我们利用三个数据集、四个模型体系结构和三个投毒率，对针对八种不同后门攻击的16种最先进方法进行了广泛的基准测试。我们的结果来自122,236个单独的实验，表明虽然许多方法提供了一定程度的保护，但它们的性能可能会有很大的差异。此外，与两种开创性的方法相比，大多数较新的方法在总体性能或跨不同环境的一致性方面没有显示出实质性的改进。根据这些发现，我们提出了未来发展更有效和更具普遍性的防御机制的潜在方向。



## **22. Optimal Denial-of-Service Attacks Against Partially-Observable Real-Time Monitoring Systems**

针对部分可观察实时监控系统的最佳拒绝服务攻击 cs.IT

arXiv admin note: text overlap with arXiv:2403.04489

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2409.16794v2) [paper-pdf](http://arxiv.org/pdf/2409.16794v2)

**Authors**: Saad Kriouile, Mohamad Assaad, Amira Alloum, Touraj Soleymani

**Abstract**: In this paper, we investigate the impact of denial-of-service attacks on the status updating of a cyber-physical system with one or more sensors connected to a remote monitor via unreliable channels. We approach the problem from the perspective of an adversary that can strategically jam a subset of the channels. The sources are modeled as Markov chains, and the performance of status updating is measured based on the age of incorrect information at the monitor. Our objective is to derive jamming policies that strike a balance between the degradation of the system's performance and the conservation of the adversary's energy. For a single-source scenario, we formulate the problem as a partially-observable Markov decision process, and rigorously prove that the optimal jamming policy is of a threshold form. We then extend the problem to a multi-source scenario. We formulate this problem as a restless multi-armed bandit, and provide a jamming policy based on the Whittle's index. Our numerical results highlight the performance of our policies compared to baseline policies.

摘要: 在本文中，我们研究了拒绝服务攻击对一个或多个传感器通过不可靠的信道连接到远程监视器的网络物理系统状态更新的影响。我们从一个对手的角度来处理这个问题，这个对手可以战略性地堵塞部分渠道。信源被建模为马尔可夫链，状态更新的性能基于监视器处错误信息的年龄来衡量。我们的目标是制定干扰策略，在系统性能下降和保存对手能量之间取得平衡。对于单源情况，我们将问题描述为部分可观测的马尔可夫决策过程，并严格证明了最优干扰策略是门限形式的。然后，我们将问题扩展到多源场景。我们将该问题描述为一个躁动的多臂强盗问题，并基于惠特尔指标提出了一种干扰策略。我们的数字结果突出了我们的政策与基准政策相比的表现。



## **23. CLMIA: Membership Inference Attacks via Unsupervised Contrastive Learning**

CLMIA：通过无监督对比学习的成员推断攻击 cs.LG

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2411.11144v1) [paper-pdf](http://arxiv.org/pdf/2411.11144v1)

**Authors**: Depeng Chen, Xiao Liu, Jie Cui, Hong Zhong

**Abstract**: Since machine learning model is often trained on a limited data set, the model is trained multiple times on the same data sample, which causes the model to memorize most of the training set data. Membership Inference Attacks (MIAs) exploit this feature to determine whether a data sample is used for training a machine learning model. However, in realistic scenarios, it is difficult for the adversary to obtain enough qualified samples that mark accurate identity information, especially since most samples are non-members in real world applications. To address this limitation, in this paper, we propose a new attack method called CLMIA, which uses unsupervised contrastive learning to train an attack model without using extra membership status information. Meanwhile, in CLMIA, we require only a small amount of data with known membership status to fine-tune the attack model. Experimental results demonstrate that CLMIA performs better than existing attack methods for different datasets and model structures, especially with data with less marked identity information. In addition, we experimentally find that the attack performs differently for different proportions of labeled identity information for member and non-member data. More analysis proves that our attack method performs better with less labeled identity information, which applies to more realistic scenarios.

摘要: 由于机器学习模型通常是在有限的数据集上训练的，所以该模型在同一数据样本上被多次训练，这使得该模型记住了大部分训练集数据。成员资格推理攻击(MIA)利用这一特征来确定数据样本是否用于训练机器学习模型。然而，在现实场景中，攻击者很难获得足够的合格样本来标记准确的身份信息，特别是在现实世界应用中大多数样本都是非成员的情况下。针对这一局限性，本文提出了一种新的攻击方法CLMIA，该方法使用无监督对比学习来训练攻击模型，而不使用额外的成员状态信息。同时，在CLMIA中，我们只需要少量已知成员状态的数据来微调攻击模型。实验结果表明，对于不同的数据集和模型结构，CLMIA的攻击性能优于现有的攻击方法，尤其是对于身份信息标记较少的数据。此外，我们还通过实验发现，对于成员和非成员数据，对于不同比例的标签身份信息，该攻击的表现是不同的。更多的分析证明，我们的攻击方法在标签身份信息较少的情况下性能更好，适用于更真实的场景。



## **24. JailbreakLens: Interpreting Jailbreak Mechanism in the Lens of Representation and Circuit**

越狱镜头：以表象和电路的视角解读越狱机制 cs.CR

18 pages, 10 figures

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2411.11114v1) [paper-pdf](http://arxiv.org/pdf/2411.11114v1)

**Authors**: Zeqing He, Zhibo Wang, Zhixuan Chu, Huiyu Xu, Rui Zheng, Kui Ren, Chun Chen

**Abstract**: Despite the outstanding performance of Large language models (LLMs) in diverse tasks, they are vulnerable to jailbreak attacks, wherein adversarial prompts are crafted to bypass their security mechanisms and elicit unexpected responses.Although jailbreak attacks are prevalent, the understanding of their underlying mechanisms remains limited. Recent studies have explain typical jailbreaking behavior (e.g., the degree to which the model refuses to respond) of LLMs by analyzing the representation shifts in their latent space caused by jailbreak prompts or identifying key neurons that contribute to the success of these attacks. However, these studies neither explore diverse jailbreak patterns nor provide a fine-grained explanation from the failure of circuit to the changes of representational, leaving significant gaps in uncovering the jailbreak mechanism. In this paper, we propose JailbreakLens, an interpretation framework that analyzes jailbreak mechanisms from both representation (which reveals how jailbreaks alter the model's harmfulness perception) and circuit perspectives (which uncovers the causes of these deceptions by identifying key circuits contributing to the vulnerability), tracking their evolution throughout the entire response generation process. We then conduct an in-depth evaluation of jailbreak behavior on four mainstream LLMs under seven jailbreak strategies. Our evaluation finds that jailbreak prompts amplify components that reinforce affirmative responses while suppressing those that produce refusal. Although this manipulation shifts model representations toward safe clusters to deceive the LLM, leading it to provide detailed responses instead of refusals, it still produce abnormal activation which can be caught in the circuit analysis.

摘要: 尽管大型语言模型(LLM)在不同的任务中表现出色，但它们很容易受到越狱攻击，在这些攻击中，敌意提示被精心制作以绕过其安全机制并引发意外响应。尽管越狱攻击非常普遍，但对其潜在机制的了解仍然有限。最近的研究已经通过分析越狱提示引起的潜伏空间的表征变化或识别有助于这些攻击成功的关键神经元来解释LLM的典型越狱行为(例如，模型拒绝响应的程度)。然而，这些研究既没有探索多样化的越狱模式，也没有提供从电路故障到表征变化的细粒度解释，在揭示越狱机制方面留下了重大空白。在本文中，我们提出了JailBreakLens，一个解释框架，它从表示(揭示越狱如何改变模型的危害性感知)和电路角度(通过识别导致漏洞的关键电路来揭示这些欺骗的原因)来分析越狱机制，跟踪它们在整个响应生成过程中的演变。然后，我们在七种越狱策略下对四种主流的低成本移动模型的越狱行为进行了深入的评估。我们的评估发现，越狱提示放大了那些强化肯定反应的成分，同时抑制了那些产生拒绝的成分。尽管这种操作将模型表示转移到安全簇以欺骗LLM，导致它提供详细的响应而不是拒绝，但它仍然产生可以在电路分析中发现的异常激活。



## **25. Exploring the Adversarial Frontier: Quantifying Robustness via Adversarial Hypervolume**

探索对抗前沿：通过对抗超容量量化稳健性 cs.CR

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2403.05100v2) [paper-pdf](http://arxiv.org/pdf/2403.05100v2)

**Authors**: Ping Guo, Cheng Gong, Xi Lin, Zhiyuan Yang, Qingfu Zhang

**Abstract**: The escalating threat of adversarial attacks on deep learning models, particularly in security-critical fields, has underscored the need for robust deep learning systems. Conventional robustness evaluations have relied on adversarial accuracy, which measures a model's performance under a specific perturbation intensity. However, this singular metric does not fully encapsulate the overall resilience of a model against varying degrees of perturbation. To address this gap, we propose a new metric termed adversarial hypervolume, assessing the robustness of deep learning models comprehensively over a range of perturbation intensities from a multi-objective optimization standpoint. This metric allows for an in-depth comparison of defense mechanisms and recognizes the trivial improvements in robustness afforded by less potent defensive strategies. Additionally, we adopt a novel training algorithm that enhances adversarial robustness uniformly across various perturbation intensities, in contrast to methods narrowly focused on optimizing adversarial accuracy. Our extensive empirical studies validate the effectiveness of the adversarial hypervolume metric, demonstrating its ability to reveal subtle differences in robustness that adversarial accuracy overlooks. This research contributes a new measure of robustness and establishes a standard for assessing and benchmarking the resilience of current and future defensive models against adversarial threats.

摘要: 对深度学习模型的敌意攻击的威胁不断升级，特别是在安全关键领域，这突显了需要强大的深度学习系统。传统的稳健性评估依赖于对抗精度，该精度衡量模型在特定扰动强度下的性能。然而，这种单一的度量并不能完全概括模型对不同程度扰动的总体弹性。为了弥补这一差距，我们提出了一种新的度量标准，称为对抗性超体积，从多目标优化的角度全面评估深度学习模型在一系列扰动强度下的稳健性。这一指标允许对防御机制进行深入比较，并认识到较弱的防御策略在健壮性方面的微小改进。此外，我们采用了一种新的训练算法，该算法在不同的扰动强度下均匀地增强了对抗的稳健性，而不是狭隘地专注于优化对抗的准确性。我们广泛的实证研究验证了对抗性超卷度量的有效性，证明了它能够揭示对抗性准确性忽略的稳健性的细微差异。这项研究提供了一种新的稳健性衡量标准，并为评估和基准当前和未来防御模型对对手威胁的弹性建立了标准。



## **26. Game-Theoretic Neyman-Pearson Detection to Combat Strategic Evasion**

游戏理论的内曼-皮尔森检测对抗战略规避 cs.CR

**SubmitDate**: 2024-11-16    [abs](http://arxiv.org/abs/2206.05276v3) [paper-pdf](http://arxiv.org/pdf/2206.05276v3)

**Authors**: Yinan Hu, Quanyan Zhu

**Abstract**: The security in networked systems depends greatly on recognizing and identifying adversarial behaviors. Traditional detection methods focus on specific categories of attacks and have become inadequate for increasingly stealthy and deceptive attacks that are designed to bypass detection strategically. This work aims to develop a holistic theory to countermeasure such evasive attacks. We focus on extending a fundamental class of statistical-based detection methods based on Neyman-Pearson's (NP) hypothesis testing formulation. We propose game-theoretic frameworks to capture the conflicting relationship between a strategic evasive attacker and an evasion-aware NP detector. By analyzing both the equilibrium behaviors of the attacker and the NP detector, we characterize their performance using Equilibrium Receiver-Operational-Characteristic (EROC) curves. We show that the evasion-aware NP detectors outperform the passive ones in the way that the former can act strategically against the attacker's behavior and adaptively modify their decision rules based on the received messages. In addition, we extend our framework to a sequential setting where the user sends out identically distributed messages. We corroborate the analytical results with a case study of anomaly detection.

摘要: 网络系统的安全性在很大程度上取决于对敌方行为的识别和识别。传统的检测方法侧重于特定类别的攻击，已不适用于日益隐蔽和欺骗性的攻击，这些攻击旨在从战略上绕过检测。这项工作旨在开发一种整体理论来对抗这种规避攻击。基于Neyman-Pearson(NP)假设检验公式，我们重点扩展了一类基本的基于统计的检测方法。我们提出了博弈论框架来捕捉战略规避攻击者和规避感知NP检测器之间的冲突关系。通过分析攻击者和NP检测器的均衡行为，我们用均衡接收-操作-特征(EROC)曲线来表征它们的性能。我们证明了逃避感知NP检测器的性能优于被动NP检测器，前者可以针对攻击者的行为采取策略性行动，并根据收到的消息自适应地修改其决策规则。此外，我们将我们的框架扩展到顺序设置，在该设置中，用户发送相同分布的消息。我们通过一个异常检测的案例验证了分析结果。



## **27. A Survey of Graph Unlearning**

图形遗忘研究综述 cs.LG

22 page review paper on graph unlearning

**SubmitDate**: 2024-11-16    [abs](http://arxiv.org/abs/2310.02164v3) [paper-pdf](http://arxiv.org/pdf/2310.02164v3)

**Authors**: Anwar Said, Yuying Zhao, Tyler Derr, Mudassir Shabbir, Waseem Abbas, Xenofon Koutsoukos

**Abstract**: Graph unlearning emerges as a crucial advancement in the pursuit of responsible AI, providing the means to remove sensitive data traces from trained models, thereby upholding the right to be forgotten. It is evident that graph machine learning exhibits sensitivity to data privacy and adversarial attacks, necessitating the application of graph unlearning techniques to address these concerns effectively. In this comprehensive survey paper, we present the first systematic review of graph unlearning approaches, encompassing a diverse array of methodologies and offering a detailed taxonomy and up-to-date literature overview to facilitate the understanding of researchers new to this field. To ensure clarity, we provide lucid explanations of the fundamental concepts and evaluation measures used in graph unlearning, catering to a broader audience with varying levels of expertise. Delving into potential applications, we explore the versatility of graph unlearning across various domains, including but not limited to social networks, adversarial settings, recommender systems, and resource-constrained environments like the Internet of Things, illustrating its potential impact in safeguarding data privacy and enhancing AI systems' robustness. Finally, we shed light on promising research directions, encouraging further progress and innovation within the domain of graph unlearning. By laying a solid foundation and fostering continued progress, this survey seeks to inspire researchers to further advance the field of graph unlearning, thereby instilling confidence in the ethical growth of AI systems and reinforcing the responsible application of machine learning techniques in various domains.

摘要: 在追求负责任的人工智能方面，图形遗忘成为一个关键的进步，提供了从训练的模型中移除敏感数据痕迹的手段，从而维护了被遗忘的权利。显然，图机器学习表现出对数据隐私和敌意攻击的敏感性，因此有必要应用图遗忘技术来有效地解决这些问题。在这篇全面的调查论文中，我们提出了第一次系统地回顾图形遗忘方法，包括一系列不同的方法，并提供了详细的分类和最新的文献综述，以促进新进入该领域的研究人员的理解。为了确保清晰，我们对图形遗忘中使用的基本概念和评估措施进行了清晰的解释，以迎合具有不同专业水平的更广泛的受众。深入挖掘潜在的应用，我们探索了图遗忘在不同领域的多功能性，包括但不限于社交网络、对手环境、推荐系统和物联网等资源受限环境，说明了它在保护数据隐私和增强AI系统健壮性方面的潜在影响。最后，我们阐明了有前途的研究方向，鼓励在图忘却学习领域内的进一步进步和创新。通过奠定坚实的基础和促进持续进步，这项调查旨在激励研究人员进一步推进图形遗忘领域，从而灌输对人工智能系统伦理增长的信心，并加强机器学习技术在各个领域的负责任应用。



## **28. Verifiably Robust Conformal Prediction**

可验证鲁棒性保形预测 cs.LO

Accepted at NeurIPS 2024

**SubmitDate**: 2024-11-16    [abs](http://arxiv.org/abs/2405.18942v3) [paper-pdf](http://arxiv.org/pdf/2405.18942v3)

**Authors**: Linus Jeary, Tom Kuipers, Mehran Hosseini, Nicola Paoletti

**Abstract**: Conformal Prediction (CP) is a popular uncertainty quantification method that provides distribution-free, statistically valid prediction sets, assuming that training and test data are exchangeable. In such a case, CP's prediction sets are guaranteed to cover the (unknown) true test output with a user-specified probability. Nevertheless, this guarantee is violated when the data is subjected to adversarial attacks, which often result in a significant loss of coverage. Recently, several approaches have been put forward to recover CP guarantees in this setting. These approaches leverage variations of randomised smoothing to produce conservative sets which account for the effect of the adversarial perturbations. They are, however, limited in that they only support $\ell^2$-bounded perturbations and classification tasks. This paper introduces VRCP (Verifiably Robust Conformal Prediction), a new framework that leverages recent neural network verification methods to recover coverage guarantees under adversarial attacks. Our VRCP method is the first to support perturbations bounded by arbitrary norms including $\ell^1$, $\ell^2$, and $\ell^\infty$, as well as regression tasks. We evaluate and compare our approach on image classification tasks (CIFAR10, CIFAR100, and TinyImageNet) and regression tasks for deep reinforcement learning environments. In every case, VRCP achieves above nominal coverage and yields significantly more efficient and informative prediction regions than the SotA.

摘要: 保角预测是一种流行的不确定性量化方法，它假设训练和测试数据是可交换的，提供了无分布的、统计上有效的预测集。在这种情况下，CP的预测集保证以用户指定的概率覆盖(未知)真实测试输出。然而，当数据受到对抗性攻击时，这一保证就会被违反，这往往会导致覆盖范围的重大损失。最近，已经提出了几种在这种情况下恢复CP担保的方法。这些方法利用随机平滑的变化来产生保守集合，这些保守集合考虑了对抗性扰动的影响。然而，它们的局限性在于它们只支持$^2$有界的扰动和分类任务。本文介绍了一种新的框架VRCP，它利用最新的神经网络验证方法来恢复对抗性攻击下的覆盖保证。我们的VRCP方法是第一个支持以任意范数为界的扰动，包括$^1$，$^2$，$^inty$，以及回归任务。我们在深度强化学习环境下的图像分类任务(CIFAR10、CIFAR100和TinyImageNet)和回归任务上对我们的方法进行了评估和比较。在任何情况下，VRCP都达到了名义覆盖率以上，并产生了比SOTA更有效和更有信息量的预测区域。



## **29. Towards Physically-Realizable Adversarial Attacks in Embodied Vision Navigation**

视觉导航中实现物理可实现的对抗攻击 cs.CV

8 pages, 6 figures, submitted to the 2025 IEEE International  Conference on Robotics & Automation (ICRA)

**SubmitDate**: 2024-11-16    [abs](http://arxiv.org/abs/2409.10071v3) [paper-pdf](http://arxiv.org/pdf/2409.10071v3)

**Authors**: Meng Chen, Jiawei Tu, Chao Qi, Yonghao Dang, Feng Zhou, Wei Wei, Jianqin Yin

**Abstract**: The deployment of embodied navigation agents in safety-critical environments raises concerns about their vulnerability to adversarial attacks on deep neural networks. However, current attack methods often lack practicality due to challenges in transitioning from the digital to the physical world, while existing physical attacks for object detection fail to achieve both multi-view effectiveness and naturalness. To address this, we propose a practical attack method for embodied navigation by attaching adversarial patches with learnable textures and opacity to objects. Specifically, to ensure effectiveness across varying viewpoints, we employ a multi-view optimization strategy based on object-aware sampling, which uses feedback from the navigation model to optimize the patch's texture. To make the patch inconspicuous to human observers, we introduce a two-stage opacity optimization mechanism, where opacity is refined after texture optimization. Experimental results show our adversarial patches reduce navigation success rates by about 40%, outperforming previous methods in practicality, effectiveness, and naturalness. Code is available at: [https://github.com/chen37058/Physical-Attacks-in-Embodied-Navigation].

摘要: 在安全关键环境中部署具体化导航代理引起了人们对它们在深层神经网络上易受敌意攻击的担忧。然而，由于从数字世界向物理世界过渡的挑战，现有的攻击方法往往缺乏实用性，而现有的针对目标检测的物理攻击无法达到多视角的有效性和自然性。为了解决这一问题，我们提出了一种实用的具身导航攻击方法，通过将具有可学习纹理和不透明度的敌意补丁附加到对象上。具体地说，为了确保不同视点的有效性，我们采用了一种基于对象感知采样的多视点优化策略，该策略利用导航模型的反馈来优化面片的纹理。为了使面片不易被人察觉，我们引入了一种两阶段不透明度优化机制，在纹理优化后对不透明度进行细化。实验结果表明，我们的对抗性补丁使导航成功率降低了约40%，在实用性、有效性和自然性方面都优于以往的方法。代码可从以下网址获得：[https://github.com/chen37058/Physical-Attacks-in-Embodied-Navigation].



## **30. Sim-CLIP: Unsupervised Siamese Adversarial Fine-Tuning for Robust and Semantically-Rich Vision-Language Models**

Sim-CLIP：针对稳健且语义丰富的视觉语言模型的无监督Siamese对抗微调 cs.CV

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2407.14971v2) [paper-pdf](http://arxiv.org/pdf/2407.14971v2)

**Authors**: Md Zarif Hossain, Ahmed Imteaj

**Abstract**: Vision-language models (VLMs) have achieved significant strides in recent times specially in multimodal tasks, yet they remain susceptible to adversarial attacks on their vision components. To address this, we propose Sim-CLIP, an unsupervised adversarial fine-tuning method that enhances the robustness of the widely-used CLIP vision encoder against such attacks while maintaining semantic richness and specificity. By employing a Siamese architecture with cosine similarity loss, Sim-CLIP learns semantically meaningful and attack-resilient visual representations without requiring large batch sizes or momentum encoders. Our results demonstrate that VLMs enhanced with Sim-CLIP's fine-tuned CLIP encoder exhibit significantly enhanced robustness against adversarial attacks, while preserving semantic meaning of the perturbed images. Notably, Sim-CLIP does not require additional training or fine-tuning of the VLM itself; replacing the original vision encoder with our fine-tuned Sim-CLIP suffices to provide robustness. This work underscores the significance of reinforcing foundational models like CLIP to safeguard the reliability of downstream VLM applications, paving the way for more secure and effective multimodal systems.

摘要: 视觉语言模型近年来取得了长足的进步，特别是在多通道任务中，但它们仍然容易受到视觉部分的敌意攻击。为了解决这一问题，我们提出了SIM-CLIP，这是一种无监督的对抗性微调方法，它在保持语义丰富和特异性的同时，增强了广泛使用的CLIP视觉编码器对此类攻击的健壮性。通过采用具有余弦相似性损失的暹罗体系结构，Sim-Clip无需大批量或动量编码器即可学习语义上有意义的、可抵抗攻击的视觉表示。结果表明，通过Sim-Clip的精细调整的CLIP编码器增强的VLM在保持扰动图像语义的同时，显著增强了对对手攻击的稳健性。值得注意的是，SIM-Clip不需要对VLM本身进行额外的培训或微调；用我们经过微调的SIM-Clip替换原来的视觉编码器就足以提供健壮性。这项工作强调了加强像CLIP这样的基础模型对保障下游VLM应用的可靠性的重要性，为更安全和有效的多式联运系统铺平了道路。



## **31. Comparing Robustness Against Adversarial Attacks in Code Generation: LLM-Generated vs. Human-Written**

比较代码生成中对抗对抗攻击的鲁棒性：LLM生成与人类编写 cs.SE

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10565v1) [paper-pdf](http://arxiv.org/pdf/2411.10565v1)

**Authors**: Md Abdul Awal, Mrigank Rochan, Chanchal K. Roy

**Abstract**: Thanks to the widespread adoption of Large Language Models (LLMs) in software engineering research, the long-standing dream of automated code generation has become a reality on a large scale. Nowadays, LLMs such as GitHub Copilot and ChatGPT are extensively used in code generation for enterprise and open-source software development and maintenance. Despite their unprecedented successes in code generation, research indicates that codes generated by LLMs exhibit vulnerabilities and security issues. Several studies have been conducted to evaluate code generated by LLMs, considering various aspects such as security, vulnerability, code smells, and robustness. While some studies have compared the performance of LLMs with that of humans in various software engineering tasks, there's a notable gap in research: no studies have directly compared human-written and LLM-generated code for their robustness analysis. To fill this void, this paper introduces an empirical study to evaluate the adversarial robustness of Pre-trained Models of Code (PTMCs) fine-tuned on code written by humans and generated by LLMs against adversarial attacks for software clone detection. These attacks could potentially undermine software security and reliability. We consider two datasets, two state-of-the-art PTMCs, two robustness evaluation criteria, and three metrics to use in our experiments. Regarding effectiveness criteria, PTMCs fine-tuned on human-written code always demonstrate more robustness than those fine-tuned on LLMs-generated code. On the other hand, in terms of adversarial code quality, in 75% experimental combinations, PTMCs fine-tuned on the human-written code exhibit more robustness than the PTMCs fine-tuned on the LLMs-generated code.

摘要: 由于大型语言模型(LLM)在软件工程研究中的广泛采用，自动化代码生成的长期梦想已经在很大程度上成为现实。如今，GitHub Copilot和ChatGPT等LLMS被广泛用于企业和开源软件开发和维护的代码生成。尽管LLM在代码生成方面取得了前所未有的成功，但研究表明，LLM生成的代码存在漏洞和安全问题。考虑到安全性、脆弱性、代码气味和健壮性等各个方面，已经进行了几项研究来评估LLMS生成的代码。虽然一些研究已经将LLM与人类在各种软件工程任务中的性能进行了比较，但研究中存在一个明显的差距：没有研究直接比较人类编写的代码和LLM生成的代码来进行健壮性分析。为了填补这一空白，本文介绍了一项经验研究，以评估预先训练的代码模型(PTMC)对人类编写的代码进行微调并由LLMS生成的代码对软件克隆检测的恶意攻击的健壮性。这些攻击可能会潜在地破坏软件的安全性和可靠性。我们考虑了两个数据集、两个最先进的PTMC、两个健壮性评估标准和三个度量来用于我们的实验。关于有效性标准，在人类编写的代码上微调的PTMC总是比那些在LLMS生成的代码上微调的PTMC表现出更强的健壮性。另一方面，在对抗性代码质量方面，在75%的实验组合中，基于人类编写的代码微调的PTMC表现出比基于LLMS生成的代码微调的PTMC更强的稳健性。



## **32. An undetectable watermark for generative image models**

生成式图像模型的不可检测水印 cs.CR

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2410.07369v2) [paper-pdf](http://arxiv.org/pdf/2410.07369v2)

**Authors**: Sam Gunn, Xuandong Zhao, Dawn Song

**Abstract**: We present the first undetectable watermarking scheme for generative image models. Undetectability ensures that no efficient adversary can distinguish between watermarked and un-watermarked images, even after making many adaptive queries. In particular, an undetectable watermark does not degrade image quality under any efficiently computable metric. Our scheme works by selecting the initial latents of a diffusion model using a pseudorandom error-correcting code (Christ and Gunn, 2024), a strategy which guarantees undetectability and robustness. We experimentally demonstrate that our watermarks are quality-preserving and robust using Stable Diffusion 2.1. Our experiments verify that, in contrast to every prior scheme we tested, our watermark does not degrade image quality. Our experiments also demonstrate robustness: existing watermark removal attacks fail to remove our watermark from images without significantly degrading the quality of the images. Finally, we find that we can robustly encode 512 bits in our watermark, and up to 2500 bits when the images are not subjected to watermark removal attacks. Our code is available at https://github.com/XuandongZhao/PRC-Watermark.

摘要: 我们提出了第一个不可检测的生成图像模型的水印方案。不可检测性确保了即使在进行了许多自适应查询之后，有效的攻击者也无法区分加水印和未加水印的图像。特别是，不可检测的水印在任何有效计算的度量下都不会降低图像质量。我们的方案通过使用伪随机纠错码(Christian和Gunn，2024)来选择扩散模型的初始潜伏期，这是一种保证不可检测性和稳健性的策略。实验证明，利用稳定扩散2.1算法，水印具有较好的保质性和稳健性。我们的实验证明，与我们测试的每个方案相比，我们的水印不会降低图像质量。我们的实验也证明了我们的稳健性：现有的水印去除攻击不能在不显著降低图像质量的情况下去除图像中的水印。最后，我们发现我们的水印可以稳健地编码512比特，当图像没有受到水印去除攻击时，可以编码高达2500比特。我们的代码可以在https://github.com/XuandongZhao/PRC-Watermark.上找到



## **33. Llama Guard 3 Vision: Safeguarding Human-AI Image Understanding Conversations**

Lama Guard 3愿景：保护人类-人工智能图像理解对话 cs.CV

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10414v1) [paper-pdf](http://arxiv.org/pdf/2411.10414v1)

**Authors**: Jianfeng Chi, Ujjwal Karn, Hongyuan Zhan, Eric Smith, Javier Rando, Yiming Zhang, Kate Plawiak, Zacharie Delpierre Coudert, Kartikeya Upasani, Mahesh Pasupuleti

**Abstract**: We introduce Llama Guard 3 Vision, a multimodal LLM-based safeguard for human-AI conversations that involves image understanding: it can be used to safeguard content for both multimodal LLM inputs (prompt classification) and outputs (response classification). Unlike the previous text-only Llama Guard versions (Inan et al., 2023; Llama Team, 2024b,a), it is specifically designed to support image reasoning use cases and is optimized to detect harmful multimodal (text and image) prompts and text responses to these prompts. Llama Guard 3 Vision is fine-tuned on Llama 3.2-Vision and demonstrates strong performance on the internal benchmarks using the MLCommons taxonomy. We also test its robustness against adversarial attacks. We believe that Llama Guard 3 Vision serves as a good starting point to build more capable and robust content moderation tools for human-AI conversation with multimodal capabilities.

摘要: 我们引入Llama Guard 3 Vision，这是一种基于Llama的多模式LLM保护措施，用于涉及图像理解的人机对话：它可用于保护多模式LLM输入（提示分类）和输出（响应分类）的内容。与之前的纯文本Llama Guard版本不同（Inan等人，2023年; Llama Team，2024 b，a），它专门设计用于支持图像推理用例，并经过优化以检测有害的多模式（文本和图像）提示以及对这些提示的文本响应。Llama Guard 3 Vision在Llama 3.2-Vision上进行了微调，并在使用MLCommons分类法的内部基准测试上展示了强劲的性能。我们还测试了它对对抗攻击的稳健性。我们相信Llama Guard 3 Vision是为具有多模式功能的人机对话构建更强大、更强大的内容审核工具的良好起点。



## **34. Continual Adversarial Reinforcement Learning (CARL) of False Data Injection detection: forgetting and explainability**

错误数据注入检测的连续对抗强化学习（CARL）：遗忘和可解释性 cs.LG

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10367v1) [paper-pdf](http://arxiv.org/pdf/2411.10367v1)

**Authors**: Pooja Aslami, Kejun Chen, Timothy M. Hansen, Malik Hassanaly

**Abstract**: False data injection attacks (FDIAs) on smart inverters are a growing concern linked to increased renewable energy production. While data-based FDIA detection methods are also actively developed, we show that they remain vulnerable to impactful and stealthy adversarial examples that can be crafted using Reinforcement Learning (RL). We propose to include such adversarial examples in data-based detection training procedure via a continual adversarial RL (CARL) approach. This way, one can pinpoint the deficiencies of data-based detection, thereby offering explainability during their incremental improvement. We show that a continual learning implementation is subject to catastrophic forgetting, and additionally show that forgetting can be addressed by employing a joint training strategy on all generated FDIA scenarios.

摘要: 针对智能逆变器的虚假数据注入攻击（FDIA）是一个与可再生能源产量增加有关的日益令人担忧的问题。虽然基于数据的FDIA检测方法也得到了积极的开发，但我们表明它们仍然容易受到可以使用强化学习（RL）制作的有影响力且隐蔽的对抗示例的影响。我们建议通过持续对抗RL（CARL）方法将此类对抗示例包括在基于数据的检测训练过程中。这样，人们就可以找出基于数据的检测的缺陷，从而在其渐进改进期间提供解释性。我们表明，持续学习的实施会受到灾难性遗忘的影响，并且还表明，可以通过对所有生成的FDIA场景采用联合训练策略来解决遗忘。



## **35. Safe Text-to-Image Generation: Simply Sanitize the Prompt Embedding**

安全的文本到图像生成：简单地消除提示嵌入 cs.CR

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10329v1) [paper-pdf](http://arxiv.org/pdf/2411.10329v1)

**Authors**: Huming Qiu, Guanxu Chen, Mi Zhang, Min Yang

**Abstract**: In recent years, text-to-image (T2I) generation models have made significant progress in generating high-quality images that align with text descriptions. However, these models also face the risk of unsafe generation, potentially producing harmful content that violates usage policies, such as explicit material. Existing safe generation methods typically focus on suppressing inappropriate content by erasing undesired concepts from visual representations, while neglecting to sanitize the textual representation. Although these methods help mitigate the risk of misuse to certain extent, their robustness remains insufficient when dealing with adversarial attacks.   Given that semantic consistency between input text and output image is a fundamental requirement for T2I models, we identify that textual representations (i.e., prompt embeddings) are likely the primary source of unsafe generation. To this end, we propose a vision-agnostic safe generation framework, Embedding Sanitizer (ES), which focuses on erasing inappropriate concepts from prompt embeddings and uses the sanitized embeddings to guide the model for safe generation. ES is applied to the output of the text encoder as a plug-and-play module, enabling seamless integration with different T2I models as well as other safeguards. In addition, ES's unique scoring mechanism assigns a score to each token in the prompt to indicate its potential harmfulness, and dynamically adjusts the sanitization intensity to balance defensive performance and generation quality. Through extensive evaluation on five prompt benchmarks, our approach achieves state-of-the-art robustness by sanitizing the source (prompt embedding) of unsafe generation compared to nine baseline methods. It significantly outperforms existing safeguards in terms of interpretability and controllability while maintaining generation quality.

摘要: 近年来，文本到图像(T2I)生成模型在生成与文本描述一致的高质量图像方面取得了重大进展。然而，这些模型也面临着不安全生成的风险，可能会产生违反使用策略的有害内容，例如露骨的材料。现有的安全生成方法通常集中于通过从视觉表示中擦除不想要的概念来抑制不适当的内容，而忽略对文本表示进行消毒。虽然这些方法在一定程度上有助于减少误用的风险，但在处理对抗性攻击时，它们的健壮性仍然不足。鉴于输入文本和输出图像之间的语义一致性是T2I模型的基本要求，我们认为文本表示(即提示嵌入)可能是不安全生成的主要来源。为此，我们提出了一个视觉不可知的安全生成框架--嵌入消毒器(Embedding Saniizer，ES)，该框架致力于从提示嵌入中删除不合适的概念，并使用消毒化的嵌入来指导模型的安全生成。ES作为即插即用模块应用于文本编码器的输出，支持与不同的T2I模型以及其他保障措施的无缝集成。此外，ES独特的评分机制为提示中的每个令牌分配了一个分数，以指示其潜在的危害性，并动态调整消毒强度，以平衡防守性能和生成质量。通过对五个即时基准的广泛评估，与九种基准方法相比，我们的方法通过对不安全生成的来源(即时嵌入)进行消毒来实现最先进的健壮性。它在可解释性和可控性方面显著优于现有的保障措施，同时保持了发电质量。



## **36. MDHP-Net: Detecting Injection Attacks on In-vehicle Network using Multi-Dimensional Hawkes Process and Temporal Model**

MDHP-Net：使用多维Hawkes过程和时态模型检测车载网络上的注入攻击 cs.CR

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10258v1) [paper-pdf](http://arxiv.org/pdf/2411.10258v1)

**Authors**: Qi Liu, Yanchen Liu, Ruifeng Li, Chenhong Cao, Yufeng Li, Xingyu Li, Peng Wang, Runhan Feng

**Abstract**: The integration of intelligent and connected technologies in modern vehicles, while offering enhanced functionalities through Electronic Control Unit and interfaces like OBD-II and telematics, also exposes the vehicle's in-vehicle network (IVN) to potential cyberattacks. In this paper, we consider a specific type of cyberattack known as the injection attack. As demonstrated by empirical data from real-world cybersecurity adversarial competitions(available at https://mimic2024.xctf.org.cn/race/qwmimic2024 ), these injection attacks have excitation effect over time, gradually manipulating network traffic and disrupting the vehicle's normal functioning, ultimately compromising both its stability and safety. To profile the abnormal behavior of attackers, we propose a novel injection attack detector to extract long-term features of attack behavior. Specifically, we first provide a theoretical analysis of modeling the time-excitation effects of the attack using Multi-Dimensional Hawkes Process (MDHP). A gradient descent solver specifically tailored for MDHP, MDHP-GDS, is developed to accurately estimate optimal MDHP parameters. We then propose an injection attack detector, MDHP-Net, which integrates optimal MDHP parameters with MDHP-LSTM blocks to enhance temporal feature extraction. By introducing MDHP parameters, MDHP-Net captures complex temporal features that standard Long Short-Term Memory (LSTM) cannot, enriching temporal dependencies within our customized structure. Extensive evaluations demonstrate the effectiveness of our proposed detection approach.

摘要: 智能和互联技术在现代车辆中的集成，在通过电子控制单元以及OBD-II和远程信息处理等接口提供增强功能的同时，也使车辆的车载网络(IVN)面临潜在的网络攻击。在本文中，我们考虑一种特定类型的网络攻击，称为注入攻击。正如现实世界网络安全对手比赛(可在https://mimic2024.xctf.org.cn/race/qwmimic2024上获得)的经验数据所表明的那样，随着时间的推移，这些注入攻击具有兴奋效应，逐渐操纵网络流量，扰乱车辆的正常运行，最终损害其稳定性和安全性。为了刻画攻击者的异常行为，我们提出了一种新的注入攻击检测器来提取攻击行为的长期特征。具体来说，我们首先给出了利用多维霍克斯过程(MDHP)对攻击的时间激励效应进行建模的理论分析。为了准确估计MDHP的最优参数，开发了一个专为MDHP量身定制的梯度下降求解器MDHP-GDS。然后，我们提出了一种注入攻击检测器MDHP-Net，它将最优的MDHP参数与MDHP-LSTM块相结合，以增强时间特征提取。通过引入MDHP参数，MDHP-Net捕获了标准长短期记忆(LSTM)无法捕获的复杂时态特征，丰富了我们定制结构中的时态依赖关系。大量的评估证明了我们所提出的检测方法的有效性。



## **37. Fault Injection and Safe-Error Attack for Extraction of Embedded Neural Network Models**

嵌入式神经网络模型提取的故障注入和安全错误攻击 cs.CR

Accepted at SECAI Workshop, ESORICS 2023 (v2. Fix notations)

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2308.16703v2) [paper-pdf](http://arxiv.org/pdf/2308.16703v2)

**Authors**: Kevin Hector, Pierre-Alain Moellic, Mathieu Dumont, Jean-Max Dutertre

**Abstract**: Model extraction emerges as a critical security threat with attack vectors exploiting both algorithmic and implementation-based approaches. The main goal of an attacker is to steal as much information as possible about a protected victim model, so that he can mimic it with a substitute model, even with a limited access to similar training data. Recently, physical attacks such as fault injection have shown worrying efficiency against the integrity and confidentiality of embedded models. We focus on embedded deep neural network models on 32-bit microcontrollers, a widespread family of hardware platforms in IoT, and the use of a standard fault injection strategy - Safe Error Attack (SEA) - to perform a model extraction attack with an adversary having a limited access to training data. Since the attack strongly depends on the input queries, we propose a black-box approach to craft a successful attack set. For a classical convolutional neural network, we successfully recover at least 90% of the most significant bits with about 1500 crafted inputs. These information enable to efficiently train a substitute model, with only 8% of the training dataset, that reaches high fidelity and near identical accuracy level than the victim model.

摘要: 模型提取成为一种严重的安全威胁，攻击载体利用了算法和基于实现的方法。攻击者的主要目标是窃取尽可能多的关于受保护受害者模型的信息，以便他可以使用替代模型来模仿它，即使对类似训练数据的访问权限有限。最近，故障注入等物理攻击对嵌入式模型的完整性和保密性显示出令人担忧的效率。我们专注于在32位微控制器上嵌入深度神经网络模型，物联网中广泛使用的硬件平台，以及使用标准故障注入策略-安全错误攻击(SEA)-在对手访问训练数据有限的情况下执行模型提取攻击。由于攻击强烈依赖于输入查询，我们提出了一种黑盒方法来构建一个成功的攻击集。对于一个经典的卷积神经网络，我们用大约1500个精心设计的输入成功地恢复了至少90%的最高有效位。这些信息使得能够有效地训练替代模型，该替代模型仅使用训练数据集的8%，达到了与受害者模型相同的高保真度和接近相同的精度水平。



## **38. A Hard-Label Cryptanalytic Extraction of Non-Fully Connected Deep Neural Networks using Side-Channel Attacks**

使用侧通道攻击的非完全连接深度神经网络的硬标签密码分析提取 cs.CR

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10174v1) [paper-pdf](http://arxiv.org/pdf/2411.10174v1)

**Authors**: Benoit Coqueret, Mathieu Carbone, Olivier Sentieys, Gabriel Zaid

**Abstract**: During the past decade, Deep Neural Networks (DNNs) proved their value on a large variety of subjects. However despite their high value and public accessibility, the protection of the intellectual property of DNNs is still an issue and an emerging research field. Recent works have successfully extracted fully-connected DNNs using cryptanalytic methods in hard-label settings, proving that it was possible to copy a DNN with high fidelity, i.e., high similitude in the output predictions. However, the current cryptanalytic attacks cannot target complex, i.e., not fully connected, DNNs and are limited to special cases of neurons present in deep networks.   In this work, we introduce a new end-to-end attack framework designed for model extraction of embedded DNNs with high fidelity. We describe a new black-box side-channel attack which splits the DNN in several linear parts for which we can perform cryptanalytic extraction and retrieve the weights in hard-label settings. With this method, we are able to adapt cryptanalytic extraction, for the first time, to non-fully connected DNNs, while maintaining a high fidelity. We validate our contributions by targeting several architectures implemented on a microcontroller unit, including a Multi-Layer Perceptron (MLP) of 1.7 million parameters and a shortened MobileNetv1. Our framework successfully extracts all of these DNNs with high fidelity (88.4% for the MobileNetv1 and 93.2% for the MLP). Furthermore, we use the stolen model to generate adversarial examples and achieve close to white-box performance on the victim's model (95.8% and 96.7% transfer rate).

摘要: 在过去的十年里，深度神经网络(DNN)证明了它们在许多学科上的价值。然而，尽管DNN具有很高的价值和公众可获得性，但DNN的知识产权保护仍然是一个问题和一个新兴的研究领域。最近的工作已经成功地在硬标签环境下使用密码分析方法提取完全连通的DNN，证明了复制高保真的DNN是可能的，即在输出预测中具有很高的相似性。然而，目前的密码分析攻击不能针对复杂的、即非完全连通的DNN，并且仅限于深层网络中存在的神经元的特殊情况。在这项工作中，我们提出了一种新的端到端攻击框架，用于高保真的嵌入式DNN模型提取。我们描述了一种新的黑盒旁通道攻击，它将DNN分裂成几个线性部分，对这些部分进行密码分析提取，并在硬标签设置中检索权重。通过这种方法，我们能够第一次将密码分析提取适应于非完全连接的DNN，同时保持高保真。我们针对在微控制器单元上实现的几个体系结构，包括具有170万个参数的多层感知器(MLP)和缩短的MobileNetv1，验证了我们的贡献。我们的框架成功地高保真地提取了所有这些DNN(MobileNetv1和MLP分别为88.4%和93.2%)。此外，我们使用被盗模型生成对抗性实例，并且在受害者的模型上获得了接近白盒的性能(95.8%和96.7%的转移率)。



## **39. Adversarial Robustness of VAEs across Intersectional Subgroups**

跨交叉亚组VAE的对抗稳健性 cs.LG

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2407.03864v2) [paper-pdf](http://arxiv.org/pdf/2407.03864v2)

**Authors**: Chethan Krishnamurthy Ramanaik, Arjun Roy, Eirini Ntoutsi

**Abstract**: Despite advancements in Autoencoders (AEs) for tasks like dimensionality reduction, representation learning and data generation, they remain vulnerable to adversarial attacks. Variational Autoencoders (VAEs), with their probabilistic approach to disentangling latent spaces, show stronger resistance to such perturbations compared to deterministic AEs; however, their resilience against adversarial inputs is still a concern. This study evaluates the robustness of VAEs against non-targeted adversarial attacks by optimizing minimal sample-specific perturbations to cause maximal damage across diverse demographic subgroups (combinations of age and gender). We investigate two questions: whether there are robustness disparities among subgroups, and what factors contribute to these disparities, such as data scarcity and representation entanglement. Our findings reveal that robustness disparities exist but are not always correlated with the size of the subgroup. By using downstream gender and age classifiers and examining latent embeddings, we highlight the vulnerability of subgroups like older women, who are prone to misclassification due to adversarial perturbations pushing their representations toward those of other subgroups.

摘要: 尽管自动编码器(AE)在降维、表示学习和数据生成等任务中取得了进步，但它们仍然容易受到对手的攻击。变分自动编码器(VAE)以其概率方法分离潜在空间，与确定性自动编码器相比，表现出更强的抗扰动能力；然而，它们对对手输入的弹性仍然是一个令人担忧的问题。这项研究通过优化最小样本特定扰动以在不同的人口统计亚组(年龄和性别组合)上造成最大损害，来评估VAE对非目标对抗性攻击的稳健性。我们调查了两个问题：子组之间是否存在稳健性差异，以及数据稀缺和表征纠缠等因素对这些差异的影响。我们的发现表明稳健性差异是存在的，但并不总是与子组的大小相关。通过使用下游的性别和年龄分类器并检查潜在嵌入，我们强调了像老年女性这样的子组的脆弱性，由于对抗性的扰动将她们的表征推向其他子组，她们容易被错误分类。



## **40. Edge-Only Universal Adversarial Attacks in Distributed Learning**

分布式学习中的仅边通用对抗攻击 cs.CR

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10500v1) [paper-pdf](http://arxiv.org/pdf/2411.10500v1)

**Authors**: Giulio Rossolini, Tommaso Baldi, Alessandro Biondi, Giorgio Buttazzo

**Abstract**: Distributed learning frameworks, which partition neural network models across multiple computing nodes, enhance efficiency in collaborative edge-cloud systems but may also introduce new vulnerabilities. In this work, we explore the feasibility of generating universal adversarial attacks when an attacker has access to the edge part of the model only, which consists in the first network layers. Unlike traditional universal adversarial perturbations (UAPs) that require full model knowledge, our approach shows that adversaries can induce effective mispredictions in the unknown cloud part by leveraging key features on the edge side. Specifically, we train lightweight classifiers from intermediate features available at the edge, i.e., before the split point, and use them in a novel targeted optimization to craft effective UAPs. Our results on ImageNet demonstrate strong attack transferability to the unknown cloud part. Additionally, we analyze the capability of an attacker to achieve targeted adversarial effect with edge-only knowledge, revealing intriguing behaviors. By introducing the first adversarial attacks with edge-only knowledge in split inference, this work underscores the importance of addressing partial model access in adversarial robustness, encouraging further research in this area.

摘要: 分布式学习框架将神经网络模型划分到多个计算节点，提高了协作边缘云系统的效率，但也可能引入新的漏洞。在这项工作中，我们探索了当攻击者只访问模型的边缘部分时，生成通用对抗性攻击的可行性，该部分存在于第一网络层。与需要完整模型知识的传统通用对抗扰动(UAP)不同，我们的方法表明，对手可以通过利用边缘侧的关键特征在未知的云部分诱导有效的误判。具体地说，我们从边缘(即分割点之前)可用的中间特征训练轻量级分类器，并将它们用于一种新的有针对性的优化，以制作有效的UAP。我们在ImageNet上的结果表明，对未知云部分的攻击具有很强的可转移性。此外，我们还分析了攻击者利用边缘知识达到目标对抗效果的能力，揭示了有趣的行为。通过在分裂推理中引入第一个仅具有边知识的对抗性攻击，本工作强调了解决部分模型访问在对抗性稳健性中的重要性，鼓励了这一领域的进一步研究。



## **41. Prompt-Guided Environmentally Consistent Adversarial Patch**

预算引导的环境一致对抗补丁 cs.CV

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10498v1) [paper-pdf](http://arxiv.org/pdf/2411.10498v1)

**Authors**: Chaoqun Li, Huanqian Yan, Lifeng Zhou, Tairan Chen, Zhuodong Liu, Hang Su

**Abstract**: Adversarial attacks in the physical world pose a significant threat to the security of vision-based systems, such as facial recognition and autonomous driving. Existing adversarial patch methods primarily focus on improving attack performance, but they often produce patches that are easily detectable by humans and struggle to achieve environmental consistency, i.e., blending patches into the environment. This paper introduces a novel approach for generating adversarial patches, which addresses both the visual naturalness and environmental consistency of the patches. We propose Prompt-Guided Environmentally Consistent Adversarial Patch (PG-ECAP), a method that aligns the patch with the environment to ensure seamless integration into the environment. The approach leverages diffusion models to generate patches that are both environmental consistency and effective in evading detection. To further enhance the naturalness and consistency, we introduce two alignment losses: Prompt Alignment Loss and Latent Space Alignment Loss, ensuring that the generated patch maintains its adversarial properties while fitting naturally within its environment. Extensive experiments in both digital and physical domains demonstrate that PG-ECAP outperforms existing methods in attack success rate and environmental consistency.

摘要: 物理世界中的对抗性攻击对基于视觉的系统的安全构成了重大威胁，例如面部识别和自动驾驶。现有的对抗性补丁方法主要着眼于提高攻击性能，但它们往往产生容易被人类检测到的补丁，并且难以实现环境一致性，即将补丁混合到环境中。本文介绍了一种新的生成敌意补丁的方法，该方法同时考虑了补丁的视觉自然性和环境一致性。我们提出了即时引导的环境一致性对抗性补丁(PG-ECAP)，这是一种将补丁与环境对齐以确保无缝集成到环境中的方法。该方法利用扩散模型来生成既具有环境一致性又能有效躲避检测的斑块。为了进一步增强对齐的自然性和一致性，我们引入了两种对齐损失：即时对齐损失和潜在空间对齐损失，以确保生成的补丁在自然适应其环境的同时保持其对抗性。在数字和物理领域的广泛实验表明，PG-ECAP在攻击成功率和环境一致性方面优于现有方法。



## **42. Self-Defense: Optimal QIF Solutions and Application to Website Fingerprinting**

自卫：最佳QIF解决方案和网站指纹识别的应用 cs.CR

38th IEEE Computer Security Foundations Symposium, IEEE, Jun 2025,  Santa Cruz, United States

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10059v1) [paper-pdf](http://arxiv.org/pdf/2411.10059v1)

**Authors**: Andreas Athanasiou, Konstantinos Chatzikokolakis, Catuscia Palamidessi

**Abstract**: Quantitative Information Flow (QIF) provides a robust information-theoretical framework for designing secure systems with minimal information leakage. While previous research has addressed the design of such systems under hard constraints (e.g. application limitations) and soft constraints (e.g. utility), scenarios often arise where the core system's behavior is considered fixed. In such cases, the challenge is to design a new component for the existing system that minimizes leakage without altering the original system. In this work we address this problem by proposing optimal solutions for constructing a new row, in a known and unmodifiable information-theoretic channel, aiming at minimizing the leakage. We first model two types of adversaries: an exact-guessing adversary, aiming to guess the secret in one try, and a s-distinguishing one, which tries to distinguish the secret s from all the other secrets.Then, we discuss design strategies for both fixed and unknown priors by offering, for each adversary, an optimal solution under linear constraints, using Linear Programming.We apply our approach to the problem of website fingerprinting defense, considering a scenario where a site administrator can modify their own site but not others. We experimentally evaluate our proposed solutions against other natural approaches. First, we sample real-world news websites and then, for both adversaries, we demonstrate that the proposed solutions are effective in achieving the least leakage. Finally, we simulate an actual attack by training an ML classifier for the s-distinguishing adversary and show that our approach decreases the accuracy of the attacker.

摘要: 定量信息流(QIF)为设计信息泄漏最少的安全系统提供了一个健壮的信息理论框架。虽然以前的研究已经解决了在硬约束(例如，应用限制)和软约束(例如，效用)下的此类系统的设计，但经常会出现核心系统的行为被认为是固定的情况。在这种情况下，挑战是为现有系统设计一个新的组件，在不改变原有系统的情况下将泄漏降至最低。在这项工作中，我们通过提出在已知的和不可修改的信息论通道中构造新的行的最优解决方案来解决这个问题，旨在最小化泄漏。我们首先建立了两类对手的模型：一种是精确猜测的对手，目的是一次猜出秘密；另一种是S区分的对手，它试图将秘密S与所有其他秘密区分开来。然后，我们讨论了固定和未知先验的设计策略，利用线性规划为每个对手提供线性约束下的最优解。我们将我们的方法应用于网站指纹防御问题，考虑了站点管理员可以修改自己的站点而不能修改其他站点的场景。我们对照其他自然方法对我们提出的解决方案进行实验评估。首先，我们抽样真实世界的新闻网站，然后，对于两个对手，我们证明所提出的解决方案在实现最小泄密方面是有效的。最后，我们通过训练S区分的对手的ML分类器来模拟一个实际的攻击，结果表明我们的方法降低了攻击者的准确率。



## **43. EveGuard: Defeating Vibration-based Side-Channel Eavesdropping with Audio Adversarial Perturbations**

EveGuard：通过音频对抗性扰动击败基于振动的侧通道发射器丢弃 cs.CR

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10034v1) [paper-pdf](http://arxiv.org/pdf/2411.10034v1)

**Authors**: Jung-Woo Chang, Ke Sun, David Xia, Xinyu Zhang, Farinaz Koushanfar

**Abstract**: Vibrometry-based side channels pose a significant privacy risk, exploiting sensors like mmWave radars, light sensors, and accelerometers to detect vibrations from sound sources or proximate objects, enabling speech eavesdropping. Despite various proposed defenses, these involve costly hardware solutions with inherent physical limitations. This paper presents EveGuard, a software-driven defense framework that creates adversarial audio, protecting voice privacy from side channels without compromising human perception. We leverage the distinct sensing capabilities of side channels and traditional microphones where side channels capture vibrations and microphones record changes in air pressure, resulting in different frequency responses. EveGuard first proposes a perturbation generator model (PGM) that effectively suppresses sensor-based eavesdropping while maintaining high audio quality. Second, to enable end-to-end training of PGM, we introduce a new domain translation task called Eve-GAN for inferring an eavesdropped signal from a given audio. We further apply few-shot learning to mitigate the data collection overhead for Eve-GAN training. Our extensive experiments show that EveGuard achieves a protection rate of more than 97 percent from audio classifiers and significantly hinders eavesdropped audio reconstruction. We further validate the performance of EveGuard across three adaptive attack mechanisms. We have conducted a user study to verify the perceptual quality of our perturbed audio.

摘要: 基于振动测量的侧通道构成了重大的隐私风险，利用毫米波雷达、光传感器和加速计等传感器来检测声源或邻近物体的振动，从而实现语音窃听。尽管提出了各种防御措施，但这些措施涉及昂贵的硬件解决方案，具有固有的物理限制。本文介绍了EveGuard，这是一个软件驱动的防御框架，可以创建敌意音频，保护语音隐私不受旁路的影响，而不会损害人类的感知。我们利用侧通道和传统麦克风的独特传感功能，侧通道捕捉振动，麦克风记录气压变化，从而产生不同的频率响应。EveGuard首先提出了一种扰动生成器模型(PGM)，该模型在保持高音频质量的同时，有效地抑制了基于传感器的窃听。其次，为了实现PGM的端到端训练，我们引入了一个新的域转换任务Eve-GAN，用于从给定的音频中推断被窃听的信号。我们进一步应用少镜头学习来减少Eve-GAN训练的数据收集开销。我们的大量实验表明，EveGuard对音频分类器的保护率超过97%，并显著阻碍了窃听的音频重建。我们进一步验证了EveGuard在三种自适应攻击机制下的性能。我们已经进行了一项用户研究，以验证我们受干扰的音频的感知质量。



## **44. Toward Robust and Accurate Adversarial Camouflage Generation against Vehicle Detectors**

针对车辆检测器实现稳健、准确的对抗性伪装生成 cs.CV

14 pages. arXiv admin note: substantial text overlap with  arXiv:2402.15853

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.10029v1) [paper-pdf](http://arxiv.org/pdf/2411.10029v1)

**Authors**: Jiawei Zhou, Linye Lyu, Daojing He, Yu Li

**Abstract**: Adversarial camouflage is a widely used physical attack against vehicle detectors for its superiority in multi-view attack performance. One promising approach involves using differentiable neural renderers to facilitate adversarial camouflage optimization through gradient back-propagation. However, existing methods often struggle to capture environmental characteristics during the rendering process or produce adversarial textures that can precisely map to the target vehicle. Moreover, these approaches neglect diverse weather conditions, reducing the efficacy of generated camouflage across varying weather scenarios. To tackle these challenges, we propose a robust and accurate camouflage generation method, namely RAUCA. The core of RAUCA is a novel neural rendering component, End-to-End Neural Renderer Plus (E2E-NRP), which can accurately optimize and project vehicle textures and render images with environmental characteristics such as lighting and weather. In addition, we integrate a multi-weather dataset for camouflage generation, leveraging the E2E-NRP to enhance the attack robustness. Experimental results on six popular object detectors show that RAUCA-final outperforms existing methods in both simulation and real-world settings.

摘要: 对抗伪装是一种广泛使用的针对车辆探测器的物理攻击，具有多视点攻击性能的优势。一种有希望的方法包括使用可微神经呈现器通过梯度反向传播来促进对抗性伪装优化。然而，现有的方法往往难以在渲染过程中捕捉环境特征，或者生成能够精确映射到目标车辆的对抗性纹理。此外，这些方法忽略了不同的天气条件，降低了在不同天气情况下产生的伪装效果。为了应对这些挑战，我们提出了一种健壮而准确的伪装生成方法，即Ruca。Ruca的核心是一种新型的神经渲染组件-端到端神经渲染器Plus(E2E-NRP)，它可以准确地优化和投影车辆纹理，并渲染具有照明和天气等环境特征的图像。此外，我们还集成了一个用于伪装生成的多天气数据集，利用E2E-NRP来增强攻击的健壮性。在六个流行的目标检测器上的实验结果表明，Ruca-Final在模拟和真实环境中的性能都优于现有的方法。



## **45. Provably Unlearnable Data Examples**

可证明不可学习的数据示例 cs.LG

Accepted to Network and Distributed System Security (NDSS) Symposium  2025, San Diego, CA, USA. Source code is available at  https://github.com/NeuralSec/certified-data-learnability

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2405.03316v2) [paper-pdf](http://arxiv.org/pdf/2405.03316v2)

**Authors**: Derui Wang, Minhui Xue, Bo Li, Seyit Camtepe, Liming Zhu

**Abstract**: The exploitation of publicly accessible data has led to escalating concerns regarding data privacy and intellectual property (IP) breaches in the age of artificial intelligence. To safeguard both data privacy and IP-related domain knowledge, efforts have been undertaken to render shared data unlearnable for unauthorized models in the wild. Existing methods apply empirically optimized perturbations to the data in the hope of disrupting the correlation between the inputs and the corresponding labels such that the data samples are converted into Unlearnable Examples (UEs). Nevertheless, the absence of mechanisms to verify the robustness of UEs against uncertainty in unauthorized models and their training procedures engenders several under-explored challenges. First, it is hard to quantify the unlearnability of UEs against unauthorized adversaries from different runs of training, leaving the soundness of the defense in obscurity. Particularly, as a prevailing evaluation metric, empirical test accuracy faces generalization errors and may not plausibly represent the quality of UEs. This also leaves room for attackers, as there is no rigid guarantee of the maximal test accuracy achievable by attackers. Furthermore, we find that a simple recovery attack can restore the clean-task performance of the classifiers trained on UEs by slightly perturbing the learned weights. To mitigate the aforementioned problems, in this paper, we propose a mechanism for certifying the so-called $(q, \eta)$-Learnability of an unlearnable dataset via parametric smoothing. A lower certified $(q, \eta)$-Learnability indicates a more robust and effective protection over the dataset. Concretely, we 1) improve the tightness of certified $(q, \eta)$-Learnability and 2) design Provably Unlearnable Examples (PUEs) which have reduced $(q, \eta)$-Learnability.

摘要: 在人工智能时代，对公开可访问数据的利用导致了人们对数据隐私和知识产权(IP)侵犯的担忧不断升级。为了保护数据隐私和与知识产权相关的领域知识，已做出努力，使未经授权的模型无法在野外学习共享数据。现有方法将经验优化的扰动应用于数据，希望破坏输入和相应标签之间的相关性，从而将数据样本转换为不可学习的示例(UE)。然而，缺乏机制来验证UE对未经授权的模型及其培训程序中的不确定性的稳健性，这带来了几个未被充分探索的挑战。首先，很难量化UE对抗来自不同训练阶段的未经授权的对手的不可学性，使得防守的可靠性模糊不清。特别是，作为一种流行的评估指标，经验测试精度面临泛化误差，可能不能可信地代表UE的质量。这也为攻击者留下了空间，因为不能严格保证攻击者可以达到的最大测试精度。此外，我们发现，简单的恢复攻击可以通过对学习的权重进行轻微扰动来恢复在UE上训练的分类器的干净任务性能。为了缓解上述问题，在本文中，我们提出了一种通过参数平滑来证明不可学习数据集的所谓$(Q，\eta)$-可学习性的机制。较低的认证$(Q，\eta)$-可学习性表示对数据集的更健壮和有效的保护。具体地说，我们1)改进了已证明的$(Q，eta)$-可学习性的紧性；2)设计了降低了$(q，eta)$-可学习性的可证明不可学习实例(PUE)。



## **46. Confidence-aware Denoised Fine-tuning of Off-the-shelf Models for Certified Robustness**

对现成模型进行信任意识的去噪微调，以获得认证的鲁棒性 cs.CV

26 pages; TMLR 2024; Code is available at  https://github.com/suhyeok24/FT-CADIS

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.08933v2) [paper-pdf](http://arxiv.org/pdf/2411.08933v2)

**Authors**: Suhyeok Jang, Seojin Kim, Jinwoo Shin, Jongheon Jeong

**Abstract**: The remarkable advances in deep learning have led to the emergence of many off-the-shelf classifiers, e.g., large pre-trained models. However, since they are typically trained on clean data, they remain vulnerable to adversarial attacks. Despite this vulnerability, their superior performance and transferability make off-the-shelf classifiers still valuable in practice, demanding further work to provide adversarial robustness for them in a post-hoc manner. A recently proposed method, denoised smoothing, leverages a denoiser model in front of the classifier to obtain provable robustness without additional training. However, the denoiser often creates hallucination, i.e., images that have lost the semantics of their originally assigned class, leading to a drop in robustness. Furthermore, its noise-and-denoise procedure introduces a significant distribution shift from the original distribution, causing the denoised smoothing framework to achieve sub-optimal robustness. In this paper, we introduce Fine-Tuning with Confidence-Aware Denoised Image Selection (FT-CADIS), a novel fine-tuning scheme to enhance the certified robustness of off-the-shelf classifiers. FT-CADIS is inspired by the observation that the confidence of off-the-shelf classifiers can effectively identify hallucinated images during denoised smoothing. Based on this, we develop a confidence-aware training objective to handle such hallucinated images and improve the stability of fine-tuning from denoised images. In this way, the classifier can be fine-tuned using only images that are beneficial for adversarial robustness. We also find that such a fine-tuning can be done by updating a small fraction of parameters of the classifier. Extensive experiments demonstrate that FT-CADIS has established the state-of-the-art certified robustness among denoised smoothing methods across all $\ell_2$-adversary radius in various benchmarks.

摘要: 深度学习的显著进展导致了许多现成的分类器的出现，例如大型预先训练的模型。然而，由于他们通常接受的是干净数据的培训，他们仍然容易受到对手的攻击。尽管存在这个漏洞，但它们优越的性能和可转移性使得现成的分类器在实践中仍然有价值，需要进一步的工作来以后自组织的方式为它们提供对抗性的健壮性。最近提出的去噪平滑方法利用分类器前面的去噪模型来获得可证明的稳健性，而不需要额外的训练。然而，去噪通常会造成幻觉，即图像失去了最初分配的类的语义，导致健壮性下降。此外，其去噪和去噪过程引入了与原始分布显著的分布偏移，导致去噪平滑框架实现次优稳健性。在本文中，我们介绍了一种新的精调方案--基于置信度的去噪图像选择精调算法(FT-CADIS)，以增强现有分类器的稳健性。FT-CADIS的灵感来自于观察到，在去噪平滑过程中，现成分类器的信心可以有效地识别幻觉图像。在此基础上，我们开发了一种置信度感知训练目标来处理这类幻觉图像，并提高了对去噪图像进行微调的稳定性。通过这种方式，可以仅使用有利于对抗健壮性的图像来微调分类器。我们还发现，这样的微调可以通过更新分类器的一小部分参数来完成。大量的实验表明，FT-CADIS在各种基准下，在所有对手半径的去噪平滑方法中建立了最先进的经验证的稳健性。



## **47. IDEATOR: Jailbreaking Large Vision-Language Models Using Themselves**

IDEATOR：利用自己越狱大型视觉语言模型 cs.CV

**SubmitDate**: 2024-11-15    [abs](http://arxiv.org/abs/2411.00827v2) [paper-pdf](http://arxiv.org/pdf/2411.00827v2)

**Authors**: Ruofan Wang, Bo Wang, Xiaosen Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: As large Vision-Language Models (VLMs) grow in prominence, ensuring their safe deployment has become critical. Recent studies have explored VLM robustness against jailbreak attacks--techniques that exploit model vulnerabilities to elicit harmful outputs. However, the limited availability of diverse multi-modal data has led current approaches to rely heavily on adversarial or manually crafted images derived from harmful text datasets, which may lack effectiveness and diversity across different contexts. In this paper, we propose a novel jailbreak method named IDEATOR, which autonomously generates malicious image-text pairs for black-box jailbreak attacks. IDEATOR is based on the insight that VLMs themselves could serve as powerful red team models for generating multimodal jailbreak prompts. Specifically, IDEATOR uses a VLM to create targeted jailbreak texts and pairs them with jailbreak images generated by a state-of-the-art diffusion model. Our extensive experiments demonstrate IDEATOR's high effectiveness and transferability. Notably, it achieves a 94% success rate in jailbreaking MiniGPT-4 with an average of only 5.34 queries, and high success rates of 82%, 88%, and 75% when transferred to LLaVA, InstructBLIP, and Meta's Chameleon, respectively. IDEATOR uncovers specific vulnerabilities in VLMs under black-box conditions, underscoring the need for improved safety mechanisms.

摘要: 随着大型视觉语言模型(VLM)的日益突出，确保它们的安全部署变得至关重要。最近的研究探索了VLM对越狱攻击的健壮性--利用模型漏洞来获得有害输出的技术。然而，多样化的多模式数据的可获得性有限，导致目前的方法严重依赖于从有害文本数据集获得的对抗性或手动制作的图像，这可能在不同的背景下缺乏有效性和多样性。在本文中，我们提出了一种新的越狱方法IDEATOR，该方法自动生成用于黑盒越狱攻击的恶意图文对。Ideator基于这样一种见解，即VLM本身可以作为强大的红色团队模型来生成多模式越狱提示。具体地说，Ideator使用VLM创建有针对性的越狱文本，并将它们与由最先进的扩散模型生成的越狱图像配对。我们的大量实验证明了IDEADER的高效性和可移植性。值得注意的是，在平均只有5.34个查询的情况下，它在越狱MiniGPT-4上的成功率达到了94%，当转移到LLaVA、InstructBLIP和Meta‘s Chameleon时，成功率分别达到了82%、88%和75%。Ideator发现了黑箱条件下VLM中的特定漏洞，强调了改进安全机制的必要性。



## **48. Adversarial Attacks Using Differentiable Rendering: A Survey**

使用差异渲染的对抗攻击：调查 cs.LG

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2411.09749v1) [paper-pdf](http://arxiv.org/pdf/2411.09749v1)

**Authors**: Matthew Hull, Chao Zhang, Zsolt Kira, Duen Horng Chau

**Abstract**: Differentiable rendering methods have emerged as a promising means for generating photo-realistic and physically plausible adversarial attacks by manipulating 3D objects and scenes that can deceive deep neural networks (DNNs). Recently, differentiable rendering capabilities have evolved significantly into a diverse landscape of libraries, such as Mitsuba, PyTorch3D, and methods like Neural Radiance Fields and 3D Gaussian Splatting for solving inverse rendering problems that share conceptually similar properties commonly used to attack DNNs, such as back-propagation and optimization. However, the adversarial machine learning research community has not yet fully explored or understood such capabilities for generating attacks. Some key reasons are that researchers often have different attack goals, such as misclassification or misdetection, and use different tasks to accomplish these goals by manipulating different representation in a scene, such as the mesh or texture of an object. This survey adopts a task-oriented unifying framework that systematically summarizes common tasks, such as manipulating textures, altering illumination, and modifying 3D meshes to exploit vulnerabilities in DNNs. Our framework enables easy comparison of existing works, reveals research gaps and spotlights exciting future research directions in this rapidly evolving field. Through focusing on how these tasks enable attacks on various DNNs such as image classification, facial recognition, object detection, optical flow and depth estimation, our survey helps researchers and practitioners better understand the vulnerabilities of computer vision systems against photorealistic adversarial attacks that could threaten real-world applications.

摘要: 最近，可区分渲染功能已显著演变为各种库，例如Mitsuba、PyTorch3D，以及像神经辐射场和3D高斯飞溅这样的方法，用于解决在概念上共享通常用于攻击DNN的属性的逆渲染问题，例如反向传播和优化。然而，对抗性机器学习研究界还没有完全探索或理解这种产生攻击的能力。本调查采用面向任务的统一框架，系统地总结了常见任务，如操纵纹理、改变照明和修改3D网格以利用DNN中的漏洞。我们的框架可以轻松地比较现有的作品，揭示研究差距，并突出这一快速发展领域令人兴奋的未来研究方向。通过关注这些任务如何实现对各种DNN的攻击，如图像分类、面部识别、目标检测、光流和深度估计，我们的调查帮助研究人员和实践者更好地了解计算机视觉系统面对可能威胁现实应用的照片真实感对手攻击的脆弱性。



## **49. DiffPAD: Denoising Diffusion-based Adversarial Patch Decontamination**

迪夫pad：消除基于扩散的对抗性补丁净化 cs.CV

Accepted to 2025 IEEE/CVF Winter Conference on Applications of  Computer Vision (WACV)

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2410.24006v2) [paper-pdf](http://arxiv.org/pdf/2410.24006v2)

**Authors**: Jia Fu, Xiao Zhang, Sepideh Pashami, Fatemeh Rahimian, Anders Holst

**Abstract**: In the ever-evolving adversarial machine learning landscape, developing effective defenses against patch attacks has become a critical challenge, necessitating reliable solutions to safeguard real-world AI systems. Although diffusion models have shown remarkable capacity in image synthesis and have been recently utilized to counter $\ell_p$-norm bounded attacks, their potential in mitigating localized patch attacks remains largely underexplored. In this work, we propose DiffPAD, a novel framework that harnesses the power of diffusion models for adversarial patch decontamination. DiffPAD first performs super-resolution restoration on downsampled input images, then adopts binarization, dynamic thresholding scheme and sliding window for effective localization of adversarial patches. Such a design is inspired by the theoretically derived correlation between patch size and diffusion restoration error that is generalized across diverse patch attack scenarios. Finally, DiffPAD applies inpainting techniques to the original input images with the estimated patch region being masked. By integrating closed-form solutions for super-resolution restoration and image inpainting into the conditional reverse sampling process of a pre-trained diffusion model, DiffPAD obviates the need for text guidance or fine-tuning. Through comprehensive experiments, we demonstrate that DiffPAD not only achieves state-of-the-art adversarial robustness against patch attacks but also excels in recovering naturalistic images without patch remnants. The source code is available at https://github.com/JasonFu1998/DiffPAD.

摘要: 在不断发展的对抗性机器学习环境中，开发针对补丁攻击的有效防御已成为一项关键挑战，需要可靠的解决方案来保护真实世界的AI系统。虽然扩散模型在图像合成方面表现出了显著的能力，并且最近已被用于对抗$\ell_p$-范数有界攻击，但它们在缓解局部补丁攻击方面的潜力仍未被充分挖掘。在这项工作中，我们提出了DiffPAD，一个新的框架，它利用扩散模型的力量来进行对抗性补丁去污。DiffPAD首先对下采样的输入图像进行超分辨率恢复，然后采用二值化、动态阈值和滑动窗口等方法对对抗性斑块进行有效定位。这种设计的灵感来自于理论上推导出的补丁大小和扩散恢复误差之间的相关性，该相关性在不同的补丁攻击场景中得到推广。最后，DiffPAD将修复技术应用于原始输入图像，并对估计的补丁区域进行掩蔽。通过将用于超分辨率恢复和图像修复的闭合形式解决方案集成到预先训练的扩散模型的条件反向采样过程中，DiffPAD消除了对文本指导或微调的需要。通过综合实验，我们证明了DiffPAD算法不仅对补丁攻击具有最好的对抗健壮性，而且在恢复没有补丁残留的自然图像方面具有很好的性能。源代码可在https://github.com/JasonFu1998/DiffPAD.上找到



## **50. Are nuclear masks all you need for improved out-of-domain generalisation? A closer look at cancer classification in histopathology**

核口罩就是改进域外通用所需的全部吗？仔细观察组织病理学中的癌症分类 eess.IV

Poster at NeurIPS 2024

**SubmitDate**: 2024-11-14    [abs](http://arxiv.org/abs/2411.09373v1) [paper-pdf](http://arxiv.org/pdf/2411.09373v1)

**Authors**: Dhananjay Tomar, Alexander Binder, Andreas Kleppe

**Abstract**: Domain generalisation in computational histopathology is challenging because the images are substantially affected by differences among hospitals due to factors like fixation and staining of tissue and imaging equipment. We hypothesise that focusing on nuclei can improve the out-of-domain (OOD) generalisation in cancer detection. We propose a simple approach to improve OOD generalisation for cancer detection by focusing on nuclear morphology and organisation, as these are domain-invariant features critical in cancer detection. Our approach integrates original images with nuclear segmentation masks during training, encouraging the model to prioritise nuclei and their spatial arrangement. Going beyond mere data augmentation, we introduce a regularisation technique that aligns the representations of masks and original images. We show, using multiple datasets, that our method improves OOD generalisation and also leads to increased robustness to image corruptions and adversarial attacks. The source code is available at https://github.com/undercutspiky/SFL/

摘要: 计算组织病理学的领域泛化是具有挑战性的，因为由于组织和成像设备的固定和染色等因素，图像在很大程度上受到医院之间的差异的影响。我们假设，聚焦于核可以改善癌症检测中的域外(OOD)泛化。我们提出了一种简单的方法，通过关注核的形态和组织来改进癌症检测的OOD泛化，因为这些是癌症检测中关键的区域不变特征。我们的方法在训练过程中将原始图像与核分割掩模相结合，鼓励模型优先考虑核及其空间排列。除了单纯的数据增强，我们还引入了一种正则化技术，使蒙版和原始图像的表示保持一致。我们使用多个数据集显示，我们的方法改进了面向对象设计的泛化，并导致对图像损坏和敌意攻击的健壮性增强。源代码可在https://github.com/undercutspiky/SFL/上找到



