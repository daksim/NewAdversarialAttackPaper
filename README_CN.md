# Latest Adversarial Attack Papers
**update at 2025-02-28 09:51:05**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Unveiling Wireless Users' Locations via Modulation Classification-based Passive Attack**

通过基于调制分类的被动攻击揭露无线用户位置 cs.IT

7 pages, 4 figures, submitted to IEEE for possible publication

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.19341v1) [paper-pdf](http://arxiv.org/pdf/2502.19341v1)

**Authors**: Ali Hanif, Abdulrahman Katranji, Nour Kouzayha, Muhammad Mahboob Ur Rahman, Tareq Y. Al-Naffouri

**Abstract**: The broadcast nature of the wireless medium and openness of wireless standards, e.g., 3GPP releases 16-20, invite adversaries to launch various active and passive attacks on cellular and other wireless networks. This work identifies one such loose end of wireless standards and presents a novel passive attack method enabling an eavesdropper (Eve) to localize a line of sight wireless user (Bob) who is communicating with a base station or WiFi access point (Alice). The proposed attack involves two phases. In the first phase, Eve performs modulation classification by intercepting the downlink channel between Alice and Bob. This enables Eve to utilize the publicly available modulation and coding scheme (MCS) tables to do pesudo-ranging, i.e., the Eve determines the ring within which Bob is located, which drastically reduces the search space. In the second phase, Eve sniffs the uplink channel, and employs multiple strategies to further refine Bob's location within the ring. Towards the end, we present our thoughts on how this attack can be extended to non-line-of-sight scenarios, and how this attack could act as a scaffolding to construct a malicious digital twin map.

摘要: 无线介质的广播特性和无线标准的开放性(例如，3GPP版本16-20)邀请对手对蜂窝和其他无线网络发起各种主动和被动攻击。这项工作识别了一种无线标准的松散端，并提出了一种新的被动攻击方法，使窃听者(Eve)能够定位与基站或WiFi接入点(Alice)通信的视线无线用户(Bob)。拟议的攻击包括两个阶段。在第一阶段，EVE通过截获Alice和Bob之间的下行链路信道来执行调制分类。这使得EVE能够利用公共可用的调制和编码方案(MCS)表来进行伪测距，即，EVE确定Bob所在的环，这大大减少了搜索空间。在第二阶段，Eve嗅探上行链路信道，并采用多种策略进一步确定Bob在环内的位置。最后，我们提出了我们的想法，如何将这种攻击扩展到非视线场景，以及这种攻击如何充当构建恶意数字孪生地图的脚手架。



## **2. Extreme vulnerability to intruder attacks destabilizes network dynamics**

对入侵者攻击的极端脆弱性会破坏网络动态的稳定 nlin.AO

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.08552v2) [paper-pdf](http://arxiv.org/pdf/2502.08552v2)

**Authors**: Amirhossein Nazerian, Sahand Tangerami, Malbor Asllani, David Phillips, Hernan Makse, Francesco Sorrentino

**Abstract**: Consensus, synchronization, formation control, and power grid balance are all examples of virtuous dynamical states that may arise in networks. Here we focus on how such states can be destabilized from a fundamental perspective; namely, we address the question of how one or a few intruder agents within an otherwise functioning network may compromise its dynamics. We show that a single adversarial node coupled via adversarial couplings to one or more other nodes is sufficient to destabilize the entire network, which we prove to be more efficient than targeting multiple nodes. Then, we show that concentrating the attack on a single low-indegree node induces the greatest instability, challenging the common assumption that hubs are the most critical nodes. This leads to a new characterization of the vulnerability of a node, which contrasts with previous work, and identifies low-indegree nodes (as opposed to the hubs) as the most vulnerable components of a network. Our results are derived for linear systems but hold true for nonlinear networks, including those described by the Kuramoto model. Finally, we derive scaling laws showing that larger networks are less susceptible, on average, to single-node attacks. Overall, these findings highlight an intrinsic vulnerability of technological systems such as autonomous networks, sensor networks, power grids, and the internet of things, with implications also to the realm of complex social and biological networks.

摘要: 共识、同步、队形控制和电网平衡都是网络中可能出现的良性动态状态的例子。在这里，我们从根本的角度关注如何破坏这种状态的稳定；也就是，我们解决了一个或几个入侵者代理在其他功能正常的网络中如何可能危及其动态的问题。我们证明了单个敌意节点通过对抗性耦合耦合到一个或多个其他节点足以破坏整个网络的稳定，我们证明了这比针对多个节点更有效。然后，我们证明了将攻击集中在单个低度节点上会导致最大的不稳定性，挑战了集线器是最关键节点的普遍假设。这导致了对节点脆弱性的新的表征，这与以前的工作形成了对比，并将低索引度节点(而不是集线器)识别为网络中最脆弱的组件。我们的结果适用于线性系统，但也适用于非线性网络，包括用Kuramoto模型描述的网络。最后，我们推导出了标度律，表明较大的网络平均而言不太容易受到单节点攻击。总体而言，这些发现突显了自主网络、传感器网络、电网和物联网等技术系统的内在脆弱性，这也对复杂的社会和生物网络领域产生了影响。



## **3. On the Byzantine Fault Tolerance of signSGD with Majority Vote**

论多数票签名新加坡元的拜占庭式过失容忍 cs.LG

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.19170v1) [paper-pdf](http://arxiv.org/pdf/2502.19170v1)

**Authors**: Emanuele Mengoli, Luzius Moll, Virgilio Strozzi, El-Mahdi El-Mhamdi

**Abstract**: In distributed learning, sign-based compression algorithms such as signSGD with majority vote provide a lightweight alternative to SGD with an additional advantage: fault tolerance (almost) for free. However, for signSGD with majority vote, this fault tolerance has been shown to cover only the case of weaker adversaries, i.e., ones that are not omniscient or cannot collude to base their attack on common knowledge and strategy. In this work, we close this gap and provide new insights into how signSGD with majority vote can be resilient against omniscient and colluding adversaries, which craft an attack after communicating with other adversaries, thus having better information to perform the most damaging attack based on a common optimal strategy. Our core contribution is in providing a proof that begins by defining the omniscience framework and the strongest possible damage against signSGD with majority vote without imposing any restrictions on the attacker. Thanks to the filtering effect of the sign-based method, we upper-bound the space of attacks to the optimal strategy for maximizing damage by an attacker. Hence, we derive an explicit probabilistic bound in terms of incorrect aggregation without resorting to unknown constants, providing a convergence bound on signSGD with majority vote in the presence of Byzantine attackers, along with a precise convergence rate. Our findings are supported by experiments on the MNIST dataset in a distributed learning environment with adversaries of varying strength.

摘要: 在分布式学习中，基于符号的压缩算法，如多数投票的signSGD，提供了一种轻量级的SGD替代方案，具有额外的优势：几乎是免费的容错。然而，对于拥有多数票的signSGD来说，这种容错已经被证明只涵盖较弱的对手的情况，即那些不是无所不知的或不能串通以基于常识和策略的攻击的情况。在这项工作中，我们缩小了这一差距，并提供了新的见解，即拥有多数选票的signSGD如何具有抵御无所不知和串通的对手的能力，这些对手在与其他对手沟通后策划攻击，从而拥有更好的信息，基于共同的最优策略执行最具破坏性的攻击。我们的核心贡献是提供了一种证明，首先定义了无所不知的框架，并以多数票对signSGD造成了最强的破坏，而不对攻击者施加任何限制。由于基于符号的方法的过滤效果，我们将攻击空间上界到最优策略，以最大化攻击者的损害。因此，我们在不求助于未知常量的情况下，得到了不正确聚集的显式概率界，在拜占庭攻击者存在的情况下，提供了带多数投票的signSGD的收敛界，并提供了精确的收敛速度。我们的发现得到了在MNIST数据集上的实验支持，该实验在分布式学习环境中具有不同强度的对手。



## **4. XSS Adversarial Attacks Based on Deep Reinforcement Learning: A Replication and Extension Study**

基于深度强化学习的XSS对抗攻击：复制和扩展研究 cs.SE

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.19095v1) [paper-pdf](http://arxiv.org/pdf/2502.19095v1)

**Authors**: Samuele Pasini, Gianluca Maragliano, Jinhan Kim, Paolo Tonella

**Abstract**: Cross-site scripting (XSS) poses a significant threat to web application security. While Deep Learning (DL) has shown remarkable success in detecting XSS attacks, it remains vulnerable to adversarial attacks due to the discontinuous nature of its input-output mapping. These adversarial attacks employ mutation-based strategies for different components of XSS attack vectors, allowing adversarial agents to iteratively select mutations to evade detection. Our work replicates a state-of-the-art XSS adversarial attack, highlighting threats to validity in the reference work and extending it toward a more effective evaluation strategy. Moreover, we introduce an XSS Oracle to mitigate these threats. The experimental results show that our approach achieves an escape rate above 96% when the threats to validity of the replicated technique are addressed.

摘要: 跨站脚本（XSS）对Web应用程序安全构成重大威胁。虽然深度学习（DL）在检测XSS攻击方面取得了巨大成功，但由于其输入输出映射的不连续性，它仍然容易受到对抗攻击。这些对抗性攻击针对XSS攻击载体的不同组成部分采用基于突变的策略，允许对抗性代理迭代地选择突变以逃避检测。我们的工作复制了最先进的XSS对抗攻击，强调了对参考工作有效性的威胁，并将其扩展到更有效的评估策略。此外，我们还引入了XSS Oracle来缓解这些威胁。实验结果表明，当解决对复制技术有效性的威胁时，我们的方法实现了96%以上的逃脱率。



## **5. Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs**

超越表面层面模式：针对LLM越狱攻击的敏捷驱动防御框架 cs.CR

15 pages, 12 figures

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.19041v1) [paper-pdf](http://arxiv.org/pdf/2502.19041v1)

**Authors**: Shiyu Xiang, Ansen Zhang, Yanfei Cao, Yang Fan, Ronghao Chen

**Abstract**: Although Aligned Large Language Models (LLMs) are trained to refuse harmful requests, they remain vulnerable to jailbreak attacks. Unfortunately, existing methods often focus on surface-level patterns, overlooking the deeper attack essences. As a result, defenses fail when attack prompts change, even though the underlying "attack essence" remains the same. To address this issue, we introduce EDDF, an \textbf{E}ssence-\textbf{D}riven \textbf{D}efense \textbf{F}ramework Against Jailbreak Attacks in LLMs. EDDF is a plug-and-play input-filtering method and operates in two stages: 1) offline essence database construction, and 2) online adversarial query detection. The key idea behind EDDF is to extract the "attack essence" from a diverse set of known attack instances and store it in an offline vector database. Experimental results demonstrate that EDDF significantly outperforms existing methods by reducing the Attack Success Rate by at least 20\%, underscoring its superior robustness against jailbreak attacks.

摘要: 尽管统一的大型语言模型(LLM)经过了拒绝有害请求的培训，但它们仍然容易受到越狱攻击。遗憾的是，现有的方法往往只关注表层模式，而忽略了更深层次的攻击本质。其结果是，当攻击提示改变时，防御就会失败，即使潜在的“攻击本质”保持不变。为了解决这个问题，我们引入了EDDF，一个针对LLMS中越狱攻击的EDDF框架。EDDF是一种即插即用的输入过滤方法，分为两个阶段：1)离线本质数据库构建，2)在线恶意查询检测。EDDF背后的关键思想是从一组不同的已知攻击实例中提取“攻击本质”，并将其存储在脱机矢量数据库中。实验结果表明，EDDF的性能明显优于现有方法，攻击成功率降低了至少20%，突出了其对越狱攻击的卓越稳健性。



## **6. Robust Over-the-Air Computation with Type-Based Multiple Access**

具有基于类型的多路访问的稳健空中计算 eess.SP

Paper submitted to 33rd European Signal Processing Conference  (EUSIPCO 2025)

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.19014v1) [paper-pdf](http://arxiv.org/pdf/2502.19014v1)

**Authors**: Marc Martinez-Gost, Ana Pérez-Neira, Miguel Ángel Lagunas

**Abstract**: This paper utilizes the properties of type-based multiple access (TBMA) to investigate its effectiveness as a robust approach for over-the-air computation (AirComp) in the presence of Byzantine attacks, this is, adversarial strategies where malicious nodes intentionally distort their transmissions to corrupt the aggregated result. Unlike classical direct aggregation (DA) AirComp, which aggregates data in the amplitude of the signals and are highly vulnerable to attacks, TBMA distributes data over multiple radio resources, enabling the receiver to construct a histogram representation of the transmitted data. This structure allows the integration of classical robust estimators and supports the computation of diverse functions beyond the arithmetic mean, which is not feasible with DA. Through extensive simulations, we demonstrate that robust TBMA significantly outperforms DA, maintaining high accuracy even under adversarial conditions, and showcases its applicability in federated learning (FEEL) scenarios. Additionally, TBMA reduces channel state information (CSI) requirements, lowers energy consumption, and enhances resiliency by leveraging the diversity of the transmitted data. These results establish TBMA as a scalable and robust solution for AirComp, paving the way for secure and efficient aggregation in next-generation networks.

摘要: 利用基于类型的多路访问(TBMA)的特性，研究了在拜占庭攻击(即恶意节点故意扭曲其传输以破坏聚集结果)的情况下，其作为一种健壮的空中计算方法(AirComp)的有效性。与传统的直接聚合(DA)AirComp不同，直接聚合(DA)AirComp以信号的幅度聚合数据，并且极易受到攻击，而TBMA将数据分布在多个无线电资源上，使接收器能够构建传输数据的直方图表示。这种结构允许集成经典的稳健估计器，并支持超过算术平均值的各种函数的计算，而这在DA中是不可行的。通过大量的仿真实验，我们证明了稳健的TBMA算法明显优于DA算法，即使在对抗环境下也能保持较高的准确率，并展示了它在联合学习(Feel)场景中的适用性。此外，通过利用传输数据的分集，TBMA降低了对信道状态信息(CSI)的要求，降低了能耗，并增强了恢复能力。这些结果使TBMA成为AirComp的可扩展和强大的解决方案，为下一代网络中安全高效的聚合铺平了道路。



## **7. Learning atomic forces from uncertainty-calibrated adversarial attacks**

从不确定性校准的对抗攻击中学习原子力 physics.comp-ph

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.18314v2) [paper-pdf](http://arxiv.org/pdf/2502.18314v2)

**Authors**: Henrique Musseli Cezar, Tilmann Bodenstein, Henrik Andersen Sveinsson, Morten Ledum, Simen Reine, Sigbjørn Løland Bore

**Abstract**: Adversarial approaches, which intentionally challenge machine learning models by generating difficult examples, are increasingly being adopted to improve machine learning interatomic potentials (MLIPs). While already providing great practical value, little is known about the actual prediction errors of MLIPs on adversarial structures and whether these errors can be controlled. We propose the Calibrated Adversarial Geometry Optimization (CAGO) algorithm to discover adversarial structures with user-assigned errors. Through uncertainty calibration, the estimated uncertainty of MLIPs is unified with real errors. By performing geometry optimization for calibrated uncertainty, we reach adversarial structures with the user-assigned target MLIP prediction error. Integrating with active learning pipelines, we benchmark CAGO, demonstrating stable MLIPs that systematically converge structural, dynamical, and thermodynamical properties for liquid water and water adsorption in a metal-organic framework within only hundreds of training structures, where previously many thousands were typically required.

摘要: 对抗性方法通过生成困难的例子来故意挑战机器学习模型，越来越多地被用来改善机器学习的原子间势(MLIP)。虽然已经提供了很大的实用价值，但对于MLIP在对抗性结构上的实际预测误差以及这些误差是否可以控制，人们知之甚少。我们提出了校准对抗性几何优化(CAGO)算法来发现含有用户指定错误的对抗性结构。通过不确定度校正，将MLIP的估计不确定度与实际误差统一起来。通过对标定的不确定性进行几何优化，我们得到了具有用户指定的目标MLIP预测误差的对抗性结构。与主动学习管道集成，我们对CAGO进行了基准测试，展示了稳定的MLIP，这些MLIP系统地聚合了液态水的结构、动力学和热力学属性，以及金属-有机框架中的水吸附，仅在数百个培训结构中，而以前通常需要数千个。



## **8. Towards Label-Only Membership Inference Attack against Pre-trained Large Language Models**

针对预训练的大型语言模型的纯标签成员推断攻击 cs.CR

Accepted by USENIX Security 2025

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.18943v1) [paper-pdf](http://arxiv.org/pdf/2502.18943v1)

**Authors**: Yu He, Boheng Li, Liu Liu, Zhongjie Ba, Wei Dong, Yiming Li, Zhan Qin, Kui Ren, Chun Chen

**Abstract**: Membership Inference Attacks (MIAs) aim to predict whether a data sample belongs to the model's training set or not. Although prior research has extensively explored MIAs in Large Language Models (LLMs), they typically require accessing to complete output logits (\ie, \textit{logits-based attacks}), which are usually not available in practice. In this paper, we study the vulnerability of pre-trained LLMs to MIAs in the \textit{label-only setting}, where the adversary can only access generated tokens (text). We first reveal that existing label-only MIAs have minor effects in attacking pre-trained LLMs, although they are highly effective in inferring fine-tuning datasets used for personalized LLMs. We find that their failure stems from two main reasons, including better generalization and overly coarse perturbation. Specifically, due to the extensive pre-training corpora and exposing each sample only a few times, LLMs exhibit minimal robustness differences between members and non-members. This makes token-level perturbations too coarse to capture such differences.   To alleviate these problems, we propose \textbf{PETAL}: a label-only membership inference attack based on \textbf{PE}r-\textbf{T}oken sem\textbf{A}ntic simi\textbf{L}arity. Specifically, PETAL leverages token-level semantic similarity to approximate output probabilities and subsequently calculate the perplexity. It finally exposes membership based on the common assumption that members are `better' memorized and have smaller perplexity. We conduct extensive experiments on the WikiMIA benchmark and the more challenging MIMIR benchmark. Empirically, our PETAL performs better than the extensions of existing label-only attacks against personalized LLMs and even on par with other advanced logit-based attacks across all metrics on five prevalent open-source LLMs.

摘要: 成员推理攻击(MIA)的目的是预测数据样本是否属于模型的训练集。尽管先前的研究已经广泛地探索了大型语言模型(LLM)中的MIA，但它们通常需要访问以完成输出日志(即，文本{基于日志的攻击})，而这在实践中通常是不可用的。在该文中，我们研究了在仅标签设置的情况下，攻击者只能访问生成的令牌(文本)的情况下，预先训练的LLMS对MIA的脆弱性。我们首先揭示了现有的仅标签MIA在攻击预先训练的LLM方面的影响很小，尽管它们在推断用于个性化LLM的微调数据集方面非常有效。我们发现，它们的失败有两个主要原因，包括较好的泛化和过粗的扰动。具体地说，由于大量的预训练语料库和每个样本只暴露几次，LLMS在成员和非成员之间表现出最小的稳健性差异。这使得令牌级的扰动过于粗略，无法捕捉到这样的差异。为了缓解这些问题，我们提出了一种基于Textbf{PE}r-\Textbf{T}Oken Sem\Textbf{A}Ntic Simi\Textbf{L}的仅标签成员关系推理攻击。具体地说，Petal利用令牌级语义相似性来近似输出概率，并随后计算困惑。它最终基于一个共同的假设来揭示成员身份，即成员的记忆力更好，困惑程度更小。我们在WikiMIA基准和更具挑战性的Mimir基准上进行了广泛的实验。根据经验，我们的Petal在针对个性化LLM的现有仅标签攻击的扩展上性能更好，甚至在五个流行的开源LLM上的所有指标上都与其他基于Logit的高级攻击相当。



## **9. Adversarial Universal Stickers: Universal Perturbation Attacks on Traffic Sign using Stickers**

对抗通用贴纸：使用贴纸对交通标志进行通用扰动攻击 cs.CV

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.18724v1) [paper-pdf](http://arxiv.org/pdf/2502.18724v1)

**Authors**: Anthony Etim, Jakub Szefer

**Abstract**: Adversarial attacks on deep learning models have proliferated in recent years. In many cases, a different adversarial perturbation is required to be added to each image to cause the deep learning model to misclassify it. This is ineffective as each image has to be modified in a different way. Meanwhile, research on universal perturbations focuses on designing a single perturbation that can be applied to all images in a data set, and cause a deep learning model to misclassify the images. This work advances the field of universal perturbations by exploring universal perturbations in the context of traffic signs and autonomous vehicle systems. This work introduces a novel method for generating universal perturbations that visually look like simple black and white stickers, and using them to cause incorrect street sign predictions. Unlike traditional adversarial perturbations, the adversarial universal stickers are designed to be applicable to any street sign: same sticker, or stickers, can be applied in same location to any street sign and cause it to be misclassified. Further, to enable safe experimentation with adversarial images and street signs, this work presents a virtual setting that leverages Street View images of street signs, rather than the need to physically modify street signs, to test the attacks. The experiments in the virtual setting demonstrate that these stickers can consistently mislead deep learning models used commonly in street sign recognition, and achieve high attack success rates on dataset of US traffic signs. The findings highlight the practical security risks posed by simple stickers applied to traffic signs, and the ease with which adversaries can generate adversarial universal stickers that can be applied to many street signs.

摘要: 近年来，对深度学习模型的敌意攻击激增。在许多情况下，需要向每个图像添加不同的对抗性扰动，以使深度学习模型对其进行错误分类。这是无效的，因为每个图像都必须以不同的方式进行修改。同时，对普遍扰动的研究集中在设计一个单一的扰动，该单一扰动可以应用于数据集中的所有图像，并导致深度学习模型对图像进行错误分类。这项工作通过探索交通标志和自动车辆系统中的普遍扰动，推动了普遍扰动领域的发展。这项工作介绍了一种新的方法来产生普遍的扰动，视觉上看起来像简单的黑白贴纸，并使用它们来导致错误的街道标志预测。与传统的对抗性干扰不同，对抗性通用贴纸被设计为适用于任何路牌：相同的贴纸，或多个贴纸，可以在相同的位置贴到任何路标上，并导致分类错误。此外，为了实现对抗性图像和街道标志的安全试验，这项工作提供了一个虚拟环境，该环境利用街道标志的街景图像来测试攻击，而不需要物理修改街道标志。在虚拟环境中的实验表明，这些贴纸能够一致地误导路牌识别中常用的深度学习模型，并在美国交通标志数据集上取得了较高的攻击成功率。这些发现突显了应用于交通标志的简单贴纸带来的实际安全风险，以及对手可以很容易地生成可以应用于许多街道标志的对抗性通用贴纸。



## **10. Time Traveling to Defend Against Adversarial Example Attacks in Image Classification**

时间旅行以防御对抗图像分类中的示例攻击 cs.CR

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2410.08338v2) [paper-pdf](http://arxiv.org/pdf/2410.08338v2)

**Authors**: Anthony Etim, Jakub Szefer

**Abstract**: Adversarial example attacks have emerged as a critical threat to machine learning. Adversarial attacks in image classification abuse various, minor modifications to the image that confuse the image classification neural network -- while the image still remains recognizable to humans. One important domain where the attacks have been applied is in the automotive setting with traffic sign classification. Researchers have demonstrated that adding stickers, shining light, or adding shadows are all different means to make machine learning inference algorithms mis-classify the traffic signs. This can cause potentially dangerous situations as a stop sign is recognized as a speed limit sign causing vehicles to ignore it and potentially leading to accidents. To address these attacks, this work focuses on enhancing defenses against such adversarial attacks. This work shifts the advantage to the user by introducing the idea of leveraging historical images and majority voting. While the attacker modifies a traffic sign that is currently being processed by the victim's machine learning inference, the victim can gain advantage by examining past images of the same traffic sign. This work introduces the notion of ''time traveling'' and uses historical Street View images accessible to anybody to perform inference on different, past versions of the same traffic sign. In the evaluation, the proposed defense has 100% effectiveness against latest adversarial example attack on traffic sign classification algorithm.

摘要: 对抗性例子攻击已经成为机器学习的一个严重威胁。图像分类中的对抗性攻击利用了对图像的各种微小修改，这混淆了图像分类神经网络--同时图像仍然可以被人类识别。应用攻击的一个重要领域是具有交通标志分类的汽车环境。研究人员已经证明，添加贴纸、照亮灯光或添加阴影都是使机器学习推理算法错误分类交通标志的不同方法。这可能会导致潜在的危险情况，因为停车标志被识别为限速标志，导致车辆忽略它，并可能导致事故。为了应对这些攻击，这项工作的重点是加强对这种对抗性攻击的防御。这项工作通过引入利用历史图像和多数投票的想法将优势转移到用户身上。当攻击者修改当前正在由受害者的机器学习推理处理的交通标志时，受害者可以通过检查同一交通标志的过去图像来获得优势。这项工作引入了时间旅行的概念，并使用任何人都可以访问的历史街景图像来对同一交通标志的不同过去版本进行推断。在评估中，所提出的防御措施对交通标志分类算法的最新对手例攻击具有100%的有效性。



## **11. Fall Leaf Adversarial Attack on Traffic Sign Classification**

交通标志分类的落叶对抗攻击 cs.CV

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2411.18776v2) [paper-pdf](http://arxiv.org/pdf/2411.18776v2)

**Authors**: Anthony Etim, Jakub Szefer

**Abstract**: Adversarial input image perturbation attacks have emerged as a significant threat to machine learning algorithms, particularly in image classification setting. These attacks involve subtle perturbations to input images that cause neural networks to misclassify the input images, even though the images remain easily recognizable to humans. One critical area where adversarial attacks have been demonstrated is in automotive systems where traffic sign classification and recognition is critical, and where misclassified images can cause autonomous systems to take wrong actions. This work presents a new class of adversarial attacks. Unlike existing work that has focused on adversarial perturbations that leverage human-made artifacts to cause the perturbations, such as adding stickers, paint, or shining flashlights at traffic signs, this work leverages nature-made artifacts: tree leaves. By leveraging nature-made artifacts, the new class of attacks has plausible deniability: a fall leaf stuck to a street sign could come from a near-by tree, rather than be placed there by an malicious human attacker. To evaluate the new class of the adversarial input image perturbation attacks, this work analyses how fall leaves can cause misclassification in street signs. The work evaluates various leaves from different species of trees, and considers various parameters such as size, color due to tree leaf type, and rotation. The work demonstrates high success rate for misclassification. The work also explores the correlation between successful attacks and how they affect the edge detection, which is critical in many image classification algorithms.

摘要: 对抗性输入图像扰动攻击已经成为机器学习算法的一个重大威胁，特别是在图像分类设置中。这些攻击涉及对输入图像的微妙扰动，导致神经网络对输入图像进行错误分类，即使图像仍然很容易被人类识别。已演示对抗性攻击的一个关键领域是在交通标志分类和识别至关重要的汽车系统中，错误分类的图像可能会导致自动系统采取错误的操作。这项工作提出了一类新的对抗性攻击。与现有的专注于对抗性干扰的工作不同，这些工作利用人造人工制品来引起扰动，例如添加贴纸、油漆或向交通标志照射手电筒，而这项工作利用了自然制造的人工制品：树叶。通过利用自然制造的文物，这种新的攻击类别具有看似合理的抵赖性：粘在路牌上的落叶可能来自附近的树，而不是被恶意的人类攻击者放置在那里。为了评估这类新的对抗性输入图像扰动攻击，本工作分析了落叶如何导致街道标志的误分类。这项工作评估了不同树种的各种树叶，并考虑了各种参数，如大小、树叶类型造成的颜色和旋转。这项工作表明，错误分类的成功率很高。这项工作还探索了成功的攻击之间的相关性以及它们如何影响边缘检测，这在许多图像分类算法中是至关重要的。



## **12. Toward Breaking Watermarks in Distortion-free Large Language Models**

迈向无失真大型语言模型中的水印 cs.CR

5 pages, AAAI'25 Workshop on Preventing and Detecting LLM Generated  Misinformation

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.18608v1) [paper-pdf](http://arxiv.org/pdf/2502.18608v1)

**Authors**: Shayleen Reynolds, Saheed Obitayo, Niccolò Dalmasso, Dung Daniel T. Ngo, Vamsi K. Potluru, Manuela Veloso

**Abstract**: In recent years, LLM watermarking has emerged as an attractive safeguard against AI-generated content, with promising applications in many real-world domains. However, there are growing concerns that the current LLM watermarking schemes are vulnerable to expert adversaries wishing to reverse-engineer the watermarking mechanisms. Prior work in "breaking" or "stealing" LLM watermarks mainly focuses on the distribution-modifying algorithm of Kirchenbauer et al. (2023), which perturbs the logit vector before sampling. In this work, we focus on reverse-engineering the other prominent LLM watermarking scheme, distortion-free watermarking (Kuditipudi et al. 2024), which preserves the underlying token distribution by using a hidden watermarking key sequence. We demonstrate that, even under a more sophisticated watermarking scheme, it is possible to "compromise" the LLM and carry out a "spoofing" attack. Specifically, we propose a mixed integer linear programming framework that accurately estimates the secret key used for watermarking using only a few samples of the watermarked dataset. Our initial findings challenge the current theoretical claims on the robustness and usability of existing LLM watermarking techniques.

摘要: 近年来，LLM水印作为一种针对人工智能生成的内容的有吸引力的保护措施，在许多现实世界领域都有很好的应用前景。然而，越来越多的人担心当前的LLM水印方案容易受到希望对水印机制进行反向工程的专家对手的攻击。以往对LLM水印“破解”或“窃取”的研究主要集中在Kirchenbauer等人的分布修正算法上。(2023)，在采样之前对Logit向量进行扰动。在这项工作中，我们专注于对另一种著名的LLM水印方案--无失真水印(Kuditipudi等人)进行逆向工程。2024)，其通过使用隐藏的水印密钥序列来保留底层令牌分布。我们证明，即使在更复杂的水印方案下，也有可能“危害”LLM并执行“欺骗”攻击。具体地说，我们提出了一种混合整数线性规划框架，该框架仅使用几个水印数据集的样本就可以准确地估计用于水印的秘密密钥。我们的初步发现挑战了当前关于现有LLM水印技术的稳健性和可用性的理论主张。



## **13. Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models**

Top-FlipRAG：针对检索增强生成模型的面向主题的对抗性观点操纵攻击 cs.CL

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.01386v2) [paper-pdf](http://arxiv.org/pdf/2502.01386v2)

**Authors**: Yuyang Gong, Zhuo Chen, Miaokun Chen, Fengchang Yu, Wei Lu, Xiaofeng Wang, Xiaozhong Liu, Jiawei Liu

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become essential for tasks such as question answering and content generation. However, their increasing impact on public opinion and information dissemination has made them a critical focus for security research due to inherent vulnerabilities. Previous studies have predominantly addressed attacks targeting factual or single-query manipulations. In this paper, we address a more practical scenario: topic-oriented adversarial opinion manipulation attacks on RAG models, where LLMs are required to reason and synthesize multiple perspectives, rendering them particularly susceptible to systematic knowledge poisoning. Specifically, we propose Topic-FlipRAG, a two-stage manipulation attack pipeline that strategically crafts adversarial perturbations to influence opinions across related queries. This approach combines traditional adversarial ranking attack techniques and leverages the extensive internal relevant knowledge and reasoning capabilities of LLMs to execute semantic-level perturbations. Experiments show that the proposed attacks effectively shift the opinion of the model's outputs on specific topics, significantly impacting user information perception. Current mitigation methods cannot effectively defend against such attacks, highlighting the necessity for enhanced safeguards for RAG systems, and offering crucial insights for LLM security research.

摘要: 基于大型语言模型(LLMS)的检索-增强生成(RAG)系统已经成为诸如问题回答和内容生成等任务的关键。然而，由于其固有的脆弱性，它们对舆论和信息传播的影响越来越大，使其成为安全研究的关键焦点。以前的研究主要针对针对事实或单一查询操作的攻击。在本文中，我们讨论了一个更实际的场景：针对RAG模型的面向主题的对抗性意见操纵攻击，其中要求LLM推理和综合多个视角，使它们特别容易受到系统性知识中毒的影响。具体地说，我们提出了Theme-FlipRAG，这是一种两阶段操纵攻击管道，战略性地制造对抗性扰动来影响相关查询的观点。该方法结合了传统的对抗性排序攻击技术，并利用LLMS丰富的内部相关知识和推理能力来执行语义级的扰动。实验表明，提出的攻击有效地改变了模型输出对特定主题的看法，显著影响了用户对信息的感知。目前的缓解方法无法有效防御此类攻击，这突显了加强RAG系统安全保障的必要性，并为LLM安全研究提供了至关重要的见解。



## **14. CLIPure: Purification in Latent Space via CLIP for Adversarially Robust Zero-Shot Classification**

CLIPure：通过CLIP在潜空间中净化，以实现对抗鲁棒零镜头分类 cs.CV

accepted by ICLR 2025

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.18176v1) [paper-pdf](http://arxiv.org/pdf/2502.18176v1)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: In this paper, we aim to build an adversarially robust zero-shot image classifier. We ground our work on CLIP, a vision-language pre-trained encoder model that can perform zero-shot classification by matching an image with text prompts ``a photo of a <class-name>.''. Purification is the path we choose since it does not require adversarial training on specific attack types and thus can cope with any foreseen attacks. We then formulate purification risk as the KL divergence between the joint distributions of the purification process of denoising the adversarial samples and the attack process of adding perturbations to benign samples, through bidirectional Stochastic Differential Equations (SDEs). The final derived results inspire us to explore purification in the multi-modal latent space of CLIP. We propose two variants for our CLIPure approach: CLIPure-Diff which models the likelihood of images' latent vectors with the DiffusionPrior module in DaLLE-2 (modeling the generation process of CLIP's latent vectors), and CLIPure-Cos which models the likelihood with the cosine similarity between the embeddings of an image and ``a photo of a.''. As far as we know, CLIPure is the first purification method in multi-modal latent space and CLIPure-Cos is the first purification method that is not based on generative models, which substantially improves defense efficiency. We conducted extensive experiments on CIFAR-10, ImageNet, and 13 datasets that previous CLIP-based defense methods used for evaluating zero-shot classification robustness. Results show that CLIPure boosts the SOTA robustness by a large margin, e.g., from 71.7% to 91.1% on CIFAR10, from 59.6% to 72.6% on ImageNet, and 108% relative improvements of average robustness on the 13 datasets over previous SOTA. The code is available at https://github.com/TMLResearchGroup-CAS/CLIPure.

摘要: 在这篇文章中，我们的目标是建立一个对抗性稳健的零镜头图像分类器。我们的工作基于CLIP，这是一个视觉语言预先训练的编码器模型，它可以通过将图像与文本提示进行匹配来执行零镜头分类。净化是我们选择的路径，因为它不需要针对特定攻击类型的对抗性训练，因此可以应对任何可预见的攻击。然后，我们通过双向随机微分方程(SDE)将净化风险表示为对敌方样本去噪的净化过程和对良性样本添加扰动的攻击过程的联合分布之间的KL发散。最终得出的结果启发我们去探索CLIP的多峰潜伏空间中的净化。我们为我们的CLIPure方法提出了两种变体：CLIPure-Diff和CLIPure-Cos，CLIPure-Diff使用DALE-2中的DiffusionPrior模块(对剪辑的潜在向量的生成过程进行建模)来模拟图像的潜在向量的可能性，CLIPure-Cos使用图像的嵌入和“a的照片”之间的余弦相似性来建模可能性。据我们所知，CLIPure是第一个在多峰潜在空间中进行净化的方法，而CLIPure-Cos是第一个不基于产生式模型的净化方法，大大提高了防御效率。我们在CIFAR-10、ImageNet和13个数据集上进行了广泛的实验，这些数据集是以前基于剪辑的防御方法用于评估零镜头分类稳健性的。结果表明，CLIPure在很大程度上提高了SOTA的健壮性，例如，在CIFAR10上从71.7%提高到91.1%，在ImageNet上从59.6%提高到72.6%，在13个数据集上的平均健壮性比以前的SOTA提高了108%。代码可在https://github.com/TMLResearchGroup-CAS/CLIPure.上获得



## **15. Exploring the Robustness and Transferability of Patch-Based Adversarial Attacks in Quantized Neural Networks**

探索量化神经网络中基于补丁的对抗攻击的鲁棒性和可移植性 cs.CR

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2411.15246v2) [paper-pdf](http://arxiv.org/pdf/2411.15246v2)

**Authors**: Amira Guesmi, Bassem Ouni, Muhammad Shafique

**Abstract**: Quantized neural networks (QNNs) are increasingly used for efficient deployment of deep learning models on resource-constrained platforms, such as mobile devices and edge computing systems. While quantization reduces model size and computational demands, its impact on adversarial robustness-especially against patch-based attacks-remains inadequately addressed. Patch-based attacks, characterized by localized, high-visibility perturbations, pose significant security risks due to their transferability and resilience. In this study, we systematically evaluate the vulnerability of QNNs to patch-based adversarial attacks across various quantization levels and architectures, focusing on factors that contribute to the robustness of these attacks. Through experiments analyzing feature representations, quantization strength, gradient alignment, and spatial sensitivity, we find that patch attacks consistently achieve high success rates across bitwidths and architectures, demonstrating significant transferability even in heavily quantized models. Contrary to the expectation that quantization might enhance adversarial defenses, our results show that QNNs remain highly susceptible to patch attacks due to the persistence of distinct, localized features within quantized representations. These findings underscore the need for quantization-aware defenses that address the specific challenges posed by patch-based attacks. Our work contributes to a deeper understanding of adversarial robustness in QNNs and aims to guide future research in developing secure, quantization-compatible defenses for real-world applications.

摘要: 量化神经网络(QNN)越来越多地被用于在资源受限的平台上高效地部署深度学习模型，例如移动设备和边缘计算系统。虽然量化减少了模型大小和计算需求，但它对对手健壮性的影响--特别是针对基于补丁的攻击--仍然没有得到充分的解决。基于补丁的攻击，其特点是局部化、高可见性的扰动，由于其可转移性和弹性，构成了巨大的安全风险。在这项研究中，我们系统地评估了QNN在不同量化级别和体系结构上对基于补丁的攻击的脆弱性，重点讨论了影响这些攻击的健壮性的因素。通过对特征表示、量化强度、梯度对齐和空间敏感度的实验分析，我们发现补丁攻击在不同的比特和体系结构上都获得了很高的成功率，即使在高度量化的模型中也表现出了显著的可移植性。与量化可能增强敌意防御的预期相反，我们的结果表明，由于量化表示中独特的局部化特征的持久性，QNN仍然非常容易受到补丁攻击。这些发现强调了量化感知防御的必要性，以应对基于补丁的攻击带来的具体挑战。我们的工作有助于更深入地理解QNN中的对抗健壮性，并旨在指导未来的研究，为现实世界的应用开发安全的、量化兼容的防御措施。



## **16. Towards Robust and Secure Embodied AI: A Survey on Vulnerabilities and Attacks**

迈向强大和安全的人工智能：关于漏洞和攻击的调查 cs.CR

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.13175v2) [paper-pdf](http://arxiv.org/pdf/2502.13175v2)

**Authors**: Wenpeng Xing, Minghao Li, Mohan Li, Meng Han

**Abstract**: Embodied AI systems, including robots and autonomous vehicles, are increasingly integrated into real-world applications, where they encounter a range of vulnerabilities stemming from both environmental and system-level factors. These vulnerabilities manifest through sensor spoofing, adversarial attacks, and failures in task and motion planning, posing significant challenges to robustness and safety. Despite the growing body of research, existing reviews rarely focus specifically on the unique safety and security challenges of embodied AI systems. Most prior work either addresses general AI vulnerabilities or focuses on isolated aspects, lacking a dedicated and unified framework tailored to embodied AI. This survey fills this critical gap by: (1) categorizing vulnerabilities specific to embodied AI into exogenous (e.g., physical attacks, cybersecurity threats) and endogenous (e.g., sensor failures, software flaws) origins; (2) systematically analyzing adversarial attack paradigms unique to embodied AI, with a focus on their impact on perception, decision-making, and embodied interaction; (3) investigating attack vectors targeting large vision-language models (LVLMs) and large language models (LLMs) within embodied systems, such as jailbreak attacks and instruction misinterpretation; (4) evaluating robustness challenges in algorithms for embodied perception, decision-making, and task planning; and (5) proposing targeted strategies to enhance the safety and reliability of embodied AI systems. By integrating these dimensions, we provide a comprehensive framework for understanding the interplay between vulnerabilities and safety in embodied AI.

摘要: 包括机器人和自动驾驶车辆在内的具体化人工智能系统正越来越多地融入现实世界的应用程序，在这些应用程序中，它们遇到了一系列源于环境和系统层面因素的漏洞。这些漏洞表现为传感器欺骗、对抗性攻击以及任务和运动规划中的失败，对健壮性和安全性构成了重大挑战。尽管研究的主体越来越多，但现有的审查很少专门关注嵌入式人工智能系统的独特安全和安保挑战。大多数以前的工作要么解决了一般的人工智能漏洞，要么专注于孤立的方面，缺乏一个专门为体现的人工智能量身定做的统一框架。本调查通过以下方式填补这一关键空白：(1)将特定于具身人工智能的漏洞分为外源性(如物理攻击、网络安全威胁)和内源性(如传感器故障、软件缺陷)来源；(2)系统分析具身人工智能特有的对抗性攻击范式，重点关注它们对感知、决策和具身交互的影响；(3)调查针对具身系统内的大视觉语言模型(LVLM)和大语言模型(LMS)的攻击向量，如越狱攻击和指令曲解；(4)评估体现感知、决策和任务规划算法中的健壮性挑战；(5)提出有针对性的策略，以提高体现人工智能系统的安全性和可靠性。通过整合这些维度，我们提供了一个全面的框架，用于理解体现的人工智能中漏洞和安全之间的相互作用。



## **17. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

卡西姆·分歧：通过对抗性防御的扩散模型来启发性解纠缠 cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2410.23091v7) [paper-pdf](http://arxiv.org/pdf/2410.23091v7)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark). The code is available at https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff.

摘要: 尽管不断努力保护神经分类器免受对手攻击，但它们仍然很脆弱，特别是面对看不见的攻击。相比之下，人类很难被微妙的操纵所欺骗，因为我们只根据基本因素做出判断。受到这一观察的启发，我们试图用基本的标签原因因素来建模标签生成，并结合标签非原因因素来辅助数据生成。对于一个对抗性的例子，我们的目标是将扰动区分为非致因因素，并仅基于标签致因因素进行预测。具体地说，我们提出了一个偶然扩散模型(CausalDiff)，该模型使扩散模型适用于条件数据生成，并通过向一个新的偶然信息瓶颈目标学习来区分这两种类型的偶然因素。经验上，CausalDiff在各种隐形攻击上的表现明显优于最先进的防御方法，在CIFAR-10上获得了86.39%(+4.01%)的平均健壮性，在CIFAR-100上获得了56.25%(+3.13%)的健壮性，在GTSRB(德国交通标志识别基准)上实现了82.62%(+4.93%)的平均健壮性。代码可在https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff.上获得



## **18. Towards Certification of Uncertainty Calibration under Adversarial Attacks**

对抗攻击下的不确定性校准认证 cs.LG

10 pages main paper, appendix included Published at: International  Conference on Learning Representations (ICLR) 2025

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2405.13922v3) [paper-pdf](http://arxiv.org/pdf/2405.13922v3)

**Authors**: Cornelius Emde, Francesco Pinto, Thomas Lukasiewicz, Philip H. S. Torr, Adel Bibi

**Abstract**: Since neural classifiers are known to be sensitive to adversarial perturbations that alter their accuracy, \textit{certification methods} have been developed to provide provable guarantees on the insensitivity of their predictions to such perturbations. Furthermore, in safety-critical applications, the frequentist interpretation of the confidence of a classifier (also known as model calibration) can be of utmost importance. This property can be measured via the Brier score or the expected calibration error. We show that attacks can significantly harm calibration, and thus propose certified calibration as worst-case bounds on calibration under adversarial perturbations. Specifically, we produce analytic bounds for the Brier score and approximate bounds via the solution of a mixed-integer program on the expected calibration error. Finally, we propose novel calibration attacks and demonstrate how they can improve model calibration through \textit{adversarial calibration training}.

摘要: 由于众所周知，神经分类器对改变其准确性的对抗性扰动敏感，因此\textit{认证方法}的开发是为了提供可证明的保证其预测对此类扰动的不敏感性。此外，在安全关键应用中，分类器置信度的频率主义解释（也称为模型校准）可能至关重要。该属性可以通过Brier评分或预期的校准误差来测量。我们表明，攻击可能会严重损害校准，因此建议将经过认证的校准作为对抗性扰动下校准的最坏情况界限。具体来说，我们通过对预期校准误差求解混合整数程序来产生Brier分数的分析界限和近似界限。最后，我们提出了新颖的校准攻击，并演示了它们如何通过\textit{对抗校准训练}来改进模型校准。



## **19. Model-Free Adversarial Purification via Coarse-To-Fine Tensor Network Representation**

通过粗到细张量网络表示的无模型对抗净化 cs.LG

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.17972v1) [paper-pdf](http://arxiv.org/pdf/2502.17972v1)

**Authors**: Guang Lin, Duc Thien Nguyen, Zerui Tao, Konstantinos Slavakis, Toshihisa Tanaka, Qibin Zhao

**Abstract**: Deep neural networks are known to be vulnerable to well-designed adversarial attacks. Although numerous defense strategies have been proposed, many are tailored to the specific attacks or tasks and often fail to generalize across diverse scenarios. In this paper, we propose Tensor Network Purification (TNP), a novel model-free adversarial purification method by a specially designed tensor network decomposition algorithm. TNP depends neither on the pre-trained generative model nor the specific dataset, resulting in strong robustness across diverse adversarial scenarios. To this end, the key challenge lies in relaxing Gaussian-noise assumptions of classical decompositions and accommodating the unknown distribution of adversarial perturbations. Unlike the low-rank representation of classical decompositions, TNP aims to reconstruct the unobserved clean examples from an adversarial example. Specifically, TNP leverages progressive downsampling and introduces a novel adversarial optimization objective to address the challenge of minimizing reconstruction error but without inadvertently restoring adversarial perturbations. Extensive experiments conducted on CIFAR-10, CIFAR-100, and ImageNet demonstrate that our method generalizes effectively across various norm threats, attack types, and tasks, providing a versatile and promising adversarial purification technique.

摘要: 众所周知，深度神经网络很容易受到精心设计的对抗性攻击。尽管已经提出了许多防御策略，但许多都是针对特定的攻击或任务量身定做的，往往无法对不同的场景进行概括。在本文中，我们提出了张量网络净化(TNP)，这是一种新的无模型的对抗性净化方法，它通过专门设计的张量网络分解算法来实现。TNP既不依赖于预先训练的生成模型，也不依赖于特定的数据集，从而在不同的对抗场景中具有很强的稳健性。为此，关键的挑战在于放宽经典分解的高斯噪声假设，并适应对抗性扰动的未知分布。与经典分解的低阶表示不同，TNP的目标是从对抗性实例中重构未被观察到的干净实例。具体地说，TNP利用渐进式下采样，并引入了一种新的对抗性优化目标来解决最小化重建误差但不会无意中恢复对抗性扰动的挑战。在CIFAR-10、CIFAR-100和ImageNet上进行的大量实验表明，我们的方法有效地概括了各种规范威胁、攻击类型和任务，提供了一种通用的、有前景的对手净化技术。



## **20. The Hidden Risks of Large Reasoning Models: A Safety Assessment of R1**

大型推理模型的隐藏风险：R1的安全评估 cs.CY

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.12659v3) [paper-pdf](http://arxiv.org/pdf/2502.12659v3)

**Authors**: Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Shreedhar Jangam, Jayanth Srinivasa, Gaowen Liu, Dawn Song, Xin Eric Wang

**Abstract**: The rapid development of large reasoning models, such as OpenAI-o3 and DeepSeek-R1, has led to significant improvements in complex reasoning over non-reasoning large language models~(LLMs). However, their enhanced capabilities, combined with the open-source access of models like DeepSeek-R1, raise serious safety concerns, particularly regarding their potential for misuse. In this work, we present a comprehensive safety assessment of these reasoning models, leveraging established safety benchmarks to evaluate their compliance with safety regulations. Furthermore, we investigate their susceptibility to adversarial attacks, such as jailbreaking and prompt injection, to assess their robustness in real-world applications. Through our multi-faceted analysis, we uncover four key findings: (1) There is a significant safety gap between the open-source R1 models and the o3-mini model, on both safety benchmark and attack, suggesting more safety effort on R1 is needed. (2) The distilled reasoning model shows poorer safety performance compared to its safety-aligned base models. (3) The stronger the model's reasoning ability, the greater the potential harm it may cause when answering unsafe questions. (4) The thinking process in R1 models pose greater safety concerns than their final answers. Our study provides insights into the security implications of reasoning models and highlights the need for further advancements in R1 models' safety to close the gap.

摘要: 大型推理模型的快速发展，如OpenAI-03和DeepSeek-R1，使得复杂推理相对于非推理的大型语言模型有了显著的改进。然而，它们增强的能力，再加上DeepSeek-R1等型号的开源访问，引发了严重的安全问题，特别是它们可能被滥用的问题。在这项工作中，我们提出了这些推理模型的全面安全评估，利用已建立的安全基准来评估它们是否符合安全法规。此外，我们调查了它们对敌意攻击的敏感性，例如越狱和快速注入，以评估它们在现实世界应用程序中的健壮性。通过多方面的分析，我们发现了四个重要的发现：(1)无论是在安全基准上还是在攻击上，开源的R1型号和03-mini型号之间都存在着显著的安全差距，这表明需要在R1上做出更多的安全努力。(2)与安全对齐的基本模型相比，精炼推理模型的安全性能较差。(3)模型的推理能力越强，在回答不安全问题时可能造成的潜在危害就越大。(4)与最终答案相比，R1模型的思维过程带来了更大的安全顾虑。我们的研究为推理模型的安全含义提供了见解，并强调了在R1模型的安全性方面进一步改进的必要性，以缩小差距。



## **21. LiSA: Leveraging Link Recommender to Attack Graph Neural Networks via Subgraph Injection**

LiSA：利用链接推荐通过子图注入攻击图神经网络 cs.LG

PAKDD 2025

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.09271v3) [paper-pdf](http://arxiv.org/pdf/2502.09271v3)

**Authors**: Wenlun Zhang, Enyan Dai, Kentaro Yoshioka

**Abstract**: Graph Neural Networks (GNNs) have demonstrated remarkable proficiency in modeling data with graph structures, yet recent research reveals their susceptibility to adversarial attacks. Traditional attack methodologies, which rely on manipulating the original graph or adding links to artificially created nodes, often prove impractical in real-world settings. This paper introduces a novel adversarial scenario involving the injection of an isolated subgraph to deceive both the link recommender and the node classifier within a GNN system. Specifically, the link recommender is mislead to propose links between targeted victim nodes and the subgraph, encouraging users to unintentionally establish connections and that would degrade the node classification accuracy, thereby facilitating a successful attack. To address this, we present the LiSA framework, which employs a dual surrogate model and bi-level optimization to simultaneously meet two adversarial objectives. Extensive experiments on real-world datasets demonstrate the effectiveness of our method.

摘要: 图神经网络(GNN)在用图结构建模数据方面表现出了卓越的能力，但最近的研究表明，它们对对手攻击很敏感。传统的攻击方法依赖于操纵原始图形或添加到人工创建的节点的链接，在现实世界中往往被证明是不切实际的。在GNN系统中，引入一个孤立的子图来欺骗链接推荐器和节点分类器，提出了一种新的对抗性场景。具体地说，链接推荐器被误导提出目标受害节点与子图之间的链接，鼓励用户无意中建立连接，这将降低节点分类的准确性，从而促进攻击的成功。为了解决这一问题，我们提出了LISA框架，该框架采用双重代理模型和双层优化来同时满足两个对抗性目标。在真实数据集上的大量实验证明了该方法的有效性。



## **22. Relationship between Uncertainty in DNNs and Adversarial Attacks**

DNN的不确定性与对抗攻击之间的关系 cs.LG

review

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2409.13232v2) [paper-pdf](http://arxiv.org/pdf/2409.13232v2)

**Authors**: Mabel Ogonna, Abigail Adeniran, Adewale Adeyemo

**Abstract**: Deep Neural Networks (DNNs) have achieved state of the art results and even outperformed human accuracy in many challenging tasks, leading to DNNs adoption in a variety of fields including natural language processing, pattern recognition, prediction, and control optimization. However, DNNs are accompanied by uncertainty about their results, causing them to predict an outcome that is either incorrect or outside of a certain level of confidence. These uncertainties stem from model or data constraints, which could be exacerbated by adversarial attacks. Adversarial attacks aim to provide perturbed input to DNNs, causing the DNN to make incorrect predictions or increase model uncertainty. In this review, we explore the relationship between DNN uncertainty and adversarial attacks, emphasizing how adversarial attacks might raise DNN uncertainty.

摘要: 深度神经网络（DNN）已实现最先进的结果，甚至在许多具有挑战性的任务中超过了人类的准确性，导致DNN在自然语言处理、模式识别、预测和控制优化等各个领域得到采用。然而，DNN伴随着结果的不确定性，导致它们预测的结果要么不正确，要么超出一定置信水平。这些不确定性源于模型或数据限制，对抗性攻击可能会加剧这种限制。对抗性攻击旨在向DNN提供受干扰的输入，导致DNN做出错误的预测或增加模型的不确定性。在这篇评论中，我们探讨了DNN不确定性和对抗性攻击之间的关系，强调了对抗性攻击如何提高DNN不确定性。



## **23. The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence**

大型语言模型中拒绝的几何学：概念锥和表示独立性 cs.LG

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17420v1) [paper-pdf](http://arxiv.org/pdf/2502.17420v1)

**Authors**: Tom Wollschläger, Jannes Elstner, Simon Geisler, Vincent Cohen-Addad, Stephan Günnemann, Johannes Gasteiger

**Abstract**: The safety alignment of large language models (LLMs) can be circumvented through adversarially crafted inputs, yet the mechanisms by which these attacks bypass safety barriers remain poorly understood. Prior work suggests that a single refusal direction in the model's activation space determines whether an LLM refuses a request. In this study, we propose a novel gradient-based approach to representation engineering and use it to identify refusal directions. Contrary to prior work, we uncover multiple independent directions and even multi-dimensional concept cones that mediate refusal. Moreover, we show that orthogonality alone does not imply independence under intervention, motivating the notion of representational independence that accounts for both linear and non-linear effects. Using this framework, we identify mechanistically independent refusal directions. We show that refusal mechanisms in LLMs are governed by complex spatial structures and identify functionally independent directions, confirming that multiple distinct mechanisms drive refusal behavior. Our gradient-based approach uncovers these mechanisms and can further serve as a foundation for future work on understanding LLMs.

摘要: 大型语言模型(LLM)的安全一致性可以通过恶意创建的输入来规避，但这些攻击绕过安全屏障的机制仍然知之甚少。先前的工作表明，模型激活空间中的单个拒绝方向决定了LLM是否拒绝请求。在这项研究中，我们提出了一种新的基于梯度的表示工程方法，并用它来识别拒绝方向。与以前的工作相反，我们发现了多个独立的方向，甚至是调解拒绝的多维概念锥。此外，我们表明，正交性本身并不意味着干预下的独立性，这激发了既能解释线性效应又能解释非线性效应的表征独立性的概念。利用这个框架，我们确定了机械独立的拒绝方向。我们发现，LLMS中的拒绝机制受到复杂空间结构的支配，并识别出功能独立的方向，证实了多种不同的机制驱动着拒绝行为。我们的基于梯度的方法揭示了这些机制，并可以进一步作为理解LLMS的未来工作的基础。



## **24. Emoti-Attack: Zero-Perturbation Adversarial Attacks on NLP Systems via Emoji Sequences**

伪攻击：通过伪君子序列对NLP系统进行零微扰对抗攻击 cs.AI

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17392v1) [paper-pdf](http://arxiv.org/pdf/2502.17392v1)

**Authors**: Yangshijie Zhang

**Abstract**: Deep neural networks (DNNs) have achieved remarkable success in the field of natural language processing (NLP), leading to widely recognized applications such as ChatGPT. However, the vulnerability of these models to adversarial attacks remains a significant concern. Unlike continuous domains like images, text exists in a discrete space, making even minor alterations at the sentence, word, or character level easily perceptible to humans. This inherent discreteness also complicates the use of conventional optimization techniques, as text is non-differentiable. Previous research on adversarial attacks in text has focused on character-level, word-level, sentence-level, and multi-level approaches, all of which suffer from inefficiency or perceptibility issues due to the need for multiple queries or significant semantic shifts.   In this work, we introduce a novel adversarial attack method, Emoji-Attack, which leverages the manipulation of emojis to create subtle, yet effective, perturbations. Unlike character- and word-level strategies, Emoji-Attack targets emojis as a distinct layer of attack, resulting in less noticeable changes with minimal disruption to the text. This approach has been largely unexplored in previous research, which typically focuses on emoji insertion as an extension of character-level attacks. Our experiments demonstrate that Emoji-Attack achieves strong attack performance on both large and small models, making it a promising technique for enhancing adversarial robustness in NLP systems.

摘要: 深度神经网络(DNN)在自然语言处理(NLP)领域取得了显著的成功，产生了广泛认可的应用，如ChatGPT。然而，这些模型对对抗性攻击的脆弱性仍然是一个重大关切。与图像等连续领域不同，文本存在于离散的空间中，即使是句子、单词或字符级别的微小更改也很容易被人类察觉。这种固有的离散性也使传统优化技术的使用变得复杂，因为文本是不可区分的。以往对文本中敌意攻击的研究主要集中在字符级、词级、句子级和多层方法上，所有这些方法都存在效率低下或可感知性问题，原因是需要进行多个查询或显著的语义转换。在这项工作中，我们介绍了一种新的对抗性攻击方法，Emoji-Attack，它利用表情符号的操纵来制造微妙但有效的扰动。与字符和单词级别的策略不同，Emoji攻击将表情符号作为不同的攻击层，导致不太明显的变化，对文本的破坏最小。这种方法在以前的研究中基本上没有被探索过，这些研究通常专注于将表情符号插入作为字符级攻击的扩展。我们的实验表明，Emoji-Attack在大小模型上都具有很强的攻击性能，是一种在NLP系统中增强对手健壮性的很有前途的技术。



## **25. On the Vulnerability of Concept Erasure in Diffusion Models**

扩散模型中概念擦除的脆弱性 cs.LG

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17537v1) [paper-pdf](http://arxiv.org/pdf/2502.17537v1)

**Authors**: Lucas Beerens, Alex D. Richardson, Kaicheng Zhang, Dongdong Chen

**Abstract**: The proliferation of text-to-image diffusion models has raised significant privacy and security concerns, particularly regarding the generation of copyrighted or harmful images. To address these issues, research on machine unlearning has developed various concept erasure methods, which aim to remove the effect of unwanted data through post-hoc training. However, we show these erasure techniques are vulnerable, where images of supposedly erased concepts can still be generated using adversarially crafted prompts. We introduce RECORD, a coordinate-descent-based algorithm that discovers prompts capable of eliciting the generation of erased content. We demonstrate that RECORD significantly beats the attack success rate of current state-of-the-art attack methods. Furthermore, our findings reveal that models subjected to concept erasure are more susceptible to adversarial attacks than previously anticipated, highlighting the urgency for more robust unlearning approaches. We open source all our code at https://github.com/LucasBeerens/RECORD

摘要: 文本到图像传播模式的激增引起了对隐私和安全的严重关切，特别是关于产生受版权保护或有害的图像。为了解决这些问题，机器遗忘的研究已经发展出各种概念擦除方法，旨在通过后自组织训练来消除不需要的数据的影响。然而，我们发现这些擦除技术是脆弱的，在这些技术中，应该被擦除的概念的图像仍然可以使用相反的精心制作的提示来生成。我们介绍了Record，这是一种基于坐标下降的算法，它发现能够引发删除内容生成的提示。我们证明，RECORD大大超过了当前最先进的攻击方法的攻击成功率。此外，我们的发现显示，受到概念删除的模型比之前预期的更容易受到对抗性攻击，这突显了更强大的遗忘方法的紧迫性。我们在https://github.com/LucasBeerens/RECORD上开放了我们所有的代码



## **26. Order Fairness Evaluation of DAG-based ledgers**

基于DAB的分类帐的订单公平性评估 cs.CR

19 pages with 9 pages dedicated to references and appendices, 23  figures, 13 of which are in the appendices

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17270v1) [paper-pdf](http://arxiv.org/pdf/2502.17270v1)

**Authors**: Erwan Mahe, Sara Tucci-Piergiovanni

**Abstract**: Order fairness in distributed ledgers refers to properties that relate the order in which transactions are sent or received to the order in which they are eventually finalized, i.e., totally ordered. The study of such properties is relatively new and has been especially stimulated by the rise of Maximal Extractable Value (MEV) attacks in blockchain environments. Indeed, in many classical blockchain protocols, leaders are responsible for selecting the transactions to be included in blocks, which creates a clear vulnerability and opportunity for transaction order manipulation.   Unlike blockchains, DAG-based ledgers allow participants in the network to independently propose blocks, which are then arranged as vertices of a directed acyclic graph. Interestingly, leaders in DAG-based ledgers are elected only after the fact, once transactions are already part of the graph, to determine their total order. In other words, transactions are not chosen by single leaders; instead, they are collectively validated by the nodes, and leaders are only elected to establish an ordering. This approach intuitively reduces the risk of transaction manipulation and enhances fairness.   In this paper, we aim to quantify the capability of DAG-based ledgers to achieve order fairness. To this end, we define new variants of order fairness adapted to DAG-based ledgers and evaluate the impact of an adversary capable of compromising a limited number of nodes (below the one-third threshold) to reorder transactions. We analyze how often our order fairness properties are violated under different network conditions and parameterizations of the DAG algorithm, depending on the adversary's power.   Our study shows that DAG-based ledgers are still vulnerable to reordering attacks, as an adversary can coordinate a minority of Byzantine nodes to manipulate the DAG's structure.

摘要: 分布式分类账中的顺序公平性是指将发送或接收交易的顺序与最终确定的顺序(即完全有序)联系起来的属性。对这类属性的研究相对较新，尤其是区块链环境中最大可提取价值(MEV)攻击的兴起。事实上，在许多经典的区块链协议中，领导者负责选择要包含在区块中的交易，这为交易顺序操纵创造了明显的漏洞和机会。与区块链不同，基于DAG的分类账允许网络中的参与者独立提出区块，然后将这些区块排列为有向无环图的顶点。有趣的是，只有在交易已经成为图表的一部分后，才会在基于DAG的分类账中选出领导人，以确定其总顺序。换句话说，事务不是由单个领导者选择的；相反，它们由节点集体验证，而领导者只被选举来建立顺序。这种方法直观地降低了交易操纵的风险，提高了公平性。在本文中，我们的目标是量化基于DAG的分类帐实现顺序公平的能力。为此，我们定义了适用于基于DAG的分类账的顺序公平性的新变体，并评估了攻击者能够危害有限数量的节点(低于三分之一的阈值)来重新排序事务的影响。我们分析了在不同的网络条件和DAG算法的参数设置下，我们的顺序公平性被违反的频率，这取决于对手的力量。我们的研究表明，基于DAG的分类账仍然容易受到重新排序攻击，因为对手可以协调少数拜占庭节点来操纵DAG的结构。



## **27. REINFORCE Adversarial Attacks on Large Language Models: An Adaptive, Distributional, and Semantic Objective**

REINFORCE对大型语言模型的对抗攻击：自适应、分布和语义目标 cs.LG

30 pages, 6 figures, 15 tables

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17254v1) [paper-pdf](http://arxiv.org/pdf/2502.17254v1)

**Authors**: Simon Geisler, Tom Wollschläger, M. H. I. Abdalla, Vincent Cohen-Addad, Johannes Gasteiger, Stephan Günnemann

**Abstract**: To circumvent the alignment of large language models (LLMs), current optimization-based adversarial attacks usually craft adversarial prompts by maximizing the likelihood of a so-called affirmative response. An affirmative response is a manually designed start of a harmful answer to an inappropriate request. While it is often easy to craft prompts that yield a substantial likelihood for the affirmative response, the attacked model frequently does not complete the response in a harmful manner. Moreover, the affirmative objective is usually not adapted to model-specific preferences and essentially ignores the fact that LLMs output a distribution over responses. If low attack success under such an objective is taken as a measure of robustness, the true robustness might be grossly overestimated. To alleviate these flaws, we propose an adaptive and semantic optimization problem over the population of responses. We derive a generally applicable objective via the REINFORCE policy-gradient formalism and demonstrate its efficacy with the state-of-the-art jailbreak algorithms Greedy Coordinate Gradient (GCG) and Projected Gradient Descent (PGD). For example, our objective doubles the attack success rate (ASR) on Llama3 and increases the ASR from 2% to 50% with circuit breaker defense.

摘要: 为了绕过大型语言模型(LLM)的对齐，当前基于优化的对抗性攻击通常通过最大化所谓肯定响应的可能性来创建对抗性提示。肯定答复是手动设计的对不适当请求的有害答复的开始。虽然通常很容易制定提示，以产生肯定响应的很大可能性，但被攻击的模型通常不会以有害的方式完成响应。此外，肯定的目标通常不适应特定于模型的偏好，基本上忽略了LLMS输出的分布高于响应的事实。如果在这样的目标下将低攻击成功率作为稳健性的衡量标准，则可能严重高估了真正的稳健性。为了克服这些缺陷，我们提出了一种基于响应总体的自适应语义优化问题。我们通过强化策略梯度理论推导出一个普遍适用的目标，并用最先进的越狱算法贪婪坐标梯度(GCG)和投影梯度下降(PGD)来验证其有效性。例如，我们的目标是使Llama3上的攻击成功率(ASR)翻一番，并通过断路器防御将ASR从2%提高到50%。



## **28. Adversarial Training for Defense Against Label Poisoning Attacks**

防御标签中毒攻击的对抗训练 cs.LG

Accepted at the International Conference on Learning Representations  (ICLR 2025)

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17121v1) [paper-pdf](http://arxiv.org/pdf/2502.17121v1)

**Authors**: Melis Ilayda Bal, Volkan Cevher, Michael Muehlebach

**Abstract**: As machine learning models grow in complexity and increasingly rely on publicly sourced data, such as the human-annotated labels used in training large language models, they become more vulnerable to label poisoning attacks. These attacks, in which adversaries subtly alter the labels within a training dataset, can severely degrade model performance, posing significant risks in critical applications. In this paper, we propose FLORAL, a novel adversarial training defense strategy based on support vector machines (SVMs) to counter these threats. Utilizing a bilevel optimization framework, we cast the training process as a non-zero-sum Stackelberg game between an attacker, who strategically poisons critical training labels, and the model, which seeks to recover from such attacks. Our approach accommodates various model architectures and employs a projected gradient descent algorithm with kernel SVMs for adversarial training. We provide a theoretical analysis of our algorithm's convergence properties and empirically evaluate FLORAL's effectiveness across diverse classification tasks. Compared to robust baselines and foundation models such as RoBERTa, FLORAL consistently achieves higher robust accuracy under increasing attacker budgets. These results underscore the potential of FLORAL to enhance the resilience of machine learning models against label poisoning threats, thereby ensuring robust classification in adversarial settings.

摘要: 随着机器学习模型变得越来越复杂，并越来越依赖于公共来源的数据，例如用于训练大型语言模型的人类注释标签，它们变得更容易受到标签中毒攻击。在这些攻击中，攻击者巧妙地更改了训练数据集中的标签，可能会严重降低模型的性能，给关键应用程序带来重大风险。针对这些威胁，本文提出了一种新的基于支持向量机的对抗性训练防御策略FLOLAR。利用双层优化框架，我们将训练过程描述为攻击者和模型之间的非零和Stackelberg博弈，攻击者策略性地毒害关键的训练标签，而模型试图从此类攻击中恢复。我们的方法适应了不同的模型结构，并使用了一种带有核支持向量机的投影梯度下降算法进行对抗性训练。我们对算法的收敛特性进行了理论分析，并对FLORAL算法在不同分类任务上的有效性进行了实证评估。与罗伯塔等稳健的基线和基础模型相比，FLORAL在不断增加的攻击者预算下始终实现更高的稳健精度。这些结果强调了FLOLAR的潜力，以增强机器学习模型对标签中毒威胁的弹性，从而确保在对抗性环境中的稳健分类。



## **29. Improving the Transferability of Adversarial Examples by Inverse Knowledge Distillation**

通过反向知识蒸馏提高对抗示例的可移植性 cs.LG

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17003v1) [paper-pdf](http://arxiv.org/pdf/2502.17003v1)

**Authors**: Wenyuan Wu, Zheng Liu, Yong Chen, Chao Su, Dezhong Peng, Xu Wang

**Abstract**: In recent years, the rapid development of deep neural networks has brought increased attention to the security and robustness of these models. While existing adversarial attack algorithms have demonstrated success in improving adversarial transferability, their performance remains suboptimal due to a lack of consideration for the discrepancies between target and source models. To address this limitation, we propose a novel method, Inverse Knowledge Distillation (IKD), designed to enhance adversarial transferability effectively. IKD introduces a distillation-inspired loss function that seamlessly integrates with gradient-based attack methods, promoting diversity in attack gradients and mitigating overfitting to specific model architectures. By diversifying gradients, IKD enables the generation of adversarial samples with superior generalization capabilities across different models, significantly enhancing their effectiveness in black-box attack scenarios. Extensive experiments on the ImageNet dataset validate the effectiveness of our approach, demonstrating substantial improvements in the transferability and attack success rates of adversarial samples across a wide range of models.

摘要: 近年来，深度神经网络的快速发展使得这些模型的安全性和稳健性受到越来越多的关注。虽然现有的对抗性攻击算法已经证明在改善对抗性可转移性方面取得了成功，但由于没有考虑目标和源模型之间的差异，它们的性能仍然不是最优的。针对这一局限性，我们提出了一种新的方法-逆知识蒸馏(IKD)，旨在有效地增强对抗转移能力。IKD引入了一个受蒸馏启发的损失函数，该函数与基于梯度的攻击方法无缝集成，促进了攻击梯度的多样性，并缓解了对特定模型体系结构的过度适应。通过使梯度多样化，IKD能够生成跨不同模型的具有卓越泛化能力的对抗性样本，显著增强其在黑盒攻击场景中的有效性。在ImageNet数据集上的广泛实验验证了我们方法的有效性，表明在广泛的模型范围内，敌方样本的可传递性和攻击成功率都有了显著的改善。



## **30. VGFL-SA: Vertical Graph Federated Learning Structure Attack Based on Contrastive Learning**

VGFL-SA：基于对比学习的垂直图联邦学习结构攻击 cs.LG

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.16793v1) [paper-pdf](http://arxiv.org/pdf/2502.16793v1)

**Authors**: Yang Chen, Bin Zhou

**Abstract**: Graph Neural Networks (GNNs) have gained attention for their ability to learn representations from graph data. Due to privacy concerns and conflicts of interest that prevent clients from directly sharing graph data with one another, Vertical Graph Federated Learning (VGFL) frameworks have been developed. Recent studies have shown that VGFL is vulnerable to adversarial attacks that degrade performance. However, it is a common problem that client nodes are often unlabeled in the realm of VGFL. Consequently, the existing attacks, which rely on the availability of labeling information to obtain gradients, are inherently constrained in their applicability. This limitation precludes their deployment in practical, real-world environments. To address the above problems, we propose a novel graph adversarial attack against VGFL, referred to as VGFL-SA, to degrade the performance of VGFL by modifying the local clients structure without using labels. Specifically, VGFL-SA uses a contrastive learning method to complete the attack before the local clients are trained. VGFL-SA first accesses the graph structure and node feature information of the poisoned clients, and generates the contrastive views by node-degree-based edge augmentation and feature shuffling augmentation. Then, VGFL-SA uses the shared graph encoder to get the embedding of each view, and the gradients of the adjacency matrices are obtained by the contrastive function. Finally, perturbed edges are generated using gradient modification rules. We validated the performance of VGFL-SA by performing a node classification task on real-world datasets, and the results show that VGFL-SA achieves good attack effectiveness and transferability.

摘要: 图形神经网络(GNN)因其从图形数据中学习表示的能力而受到关注。由于隐私问题和利益冲突阻碍了客户之间直接共享图形数据，垂直图形联合学习(VGFL)框架已经开发出来。最近的研究表明，VGFL很容易受到降低性能的对抗性攻击。然而，在VGFL领域中，一个常见的问题是客户端节点通常是未标记的。因此，现有的攻击依赖于标记信息的可用性来获得梯度，其适用性受到固有的限制。这一限制排除了它们在实际、真实环境中的部署。针对上述问题，我们提出了一种新的针对VGFL的图对抗攻击，称为VGFL-SA，通过修改本地客户端结构而不使用标签来降低VGFL的性能。具体地说，VGFL-SA使用对比学习方法在本地客户端训练之前完成攻击。VGFL-SA首先获取中毒客户端的图结构和节点特征信息，然后通过基于节点度的边增强和特征置乱增强生成对比视图。然后，VGFL-SA使用共享图编码器得到每个视点的嵌入，并通过对比函数得到邻接矩阵的梯度。最后，使用梯度修正规则生成扰动边缘。我们通过在真实数据集上执行节点分类任务来验证VGFL-SA的性能，结果表明VGFL-SA具有良好的攻击有效性和可转移性。



## **31. Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs**

Dysca：评估LVLM感知能力的动态和可扩展基准 cs.CV

Accepted by ICLR2025

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2406.18849v4) [paper-pdf](http://arxiv.org/pdf/2406.18849v4)

**Authors**: Jie Zhang, Zhongqi Wang, Mengqi Lei, Zheng Yuan, Bei Yan, Shiguang Shan, Xilin Chen

**Abstract**: Currently many benchmarks have been proposed to evaluate the perception ability of the Large Vision-Language Models (LVLMs). However, most benchmarks conduct questions by selecting images from existing datasets, resulting in the potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic style images and clean scenarios, leaving the multi-stylized images and noisy scenarios unexplored. In response to these challenges, we propose a dynamic and scalable benchmark named Dysca for evaluating LVLMs by leveraging synthesis images. Specifically, we leverage Stable Diffusion and design a rule-based method to dynamically generate novel images, questions and the corresponding answers. We consider 51 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Corruption, Print Attacking and Adversarial Attacking) and 3 question types (i.e., Multi-choices, True-or-false and Free-form). Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks and scenarios. A total of 24 advanced open-source LVLMs and 2 close-source LVLMs are evaluated on Dysca, revealing the drawbacks of current LVLMs. The benchmark is released at https://github.com/Robin-WZQ/Dysca.

摘要: 目前，人们已经提出了许多基准来评估大型视觉语言模型的感知能力。然而，大多数基准测试通过从现有数据集中选择图像来进行问题，从而导致潜在的数据泄漏。此外，这些基准只关注真实感风格的图像和干净的场景来评估LVLMS，而对多风格化的图像和噪声场景没有进行探索。为了应对这些挑战，我们提出了一种动态的、可扩展的基准测试DYSCA，用于利用合成图像来评估LVLMS。具体地说，我们利用稳定扩散并设计了一种基于规则的方法来动态生成新的图像、问题和相应的答案。我们考虑了51种图像风格，并在20个子任务中评估了感知能力。此外，我们还在4个场景(即廉洁、腐败、打印攻击和对抗性攻击)和3个问题类型(即多项选择、对错和自由形式)下进行了评估。多亏了生成性范式，Dysca成为了一个可伸缩的基准，可以轻松添加新的子任务和场景。在Dysca上对24个先进的开源LVLMS和2个闭源LVLMS进行了评估，揭示了现有LVLMS的不足。该基准在https://github.com/Robin-WZQ/Dysca.上发布



## **32. Guardians of the Agentic System: Preventing Many Shots Jailbreak with Agentic System**

紧急系统的守护者：用紧急系统防止多次枪击越狱 cs.CR

18 pages, 7 figures

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16750v1) [paper-pdf](http://arxiv.org/pdf/2502.16750v1)

**Authors**: Saikat Barua, Mostafizur Rahman, Md Jafor Sadek, Rafiul Islam, Shehnaz Khaled, Ahmedul Kabir

**Abstract**: The autonomous AI agents using large language models can create undeniable values in all span of the society but they face security threats from adversaries that warrants immediate protective solutions because trust and safety issues arise. Considering the many-shot jailbreaking and deceptive alignment as some of the main advanced attacks, that cannot be mitigated by the static guardrails used during the supervised training, points out a crucial research priority for real world robustness. The combination of static guardrails in dynamic multi-agent system fails to defend against those attacks. We intend to enhance security for LLM-based agents through the development of new evaluation frameworks which identify and counter threats for safe operational deployment. Our work uses three examination methods to detect rogue agents through a Reverse Turing Test and analyze deceptive alignment through multi-agent simulations and develops an anti-jailbreaking system by testing it with GEMINI 1.5 pro and llama-3.3-70B, deepseek r1 models using tool-mediated adversarial scenarios. The detection capabilities are strong such as 94\% accuracy for GEMINI 1.5 pro yet the system suffers persistent vulnerabilities when under long attacks as prompt length increases attack success rates (ASR) and diversity metrics become ineffective in prediction while revealing multiple complex system faults. The findings demonstrate the necessity of adopting flexible security systems based on active monitoring that can be performed by the agents themselves together with adaptable interventions by system admin as the current models can create vulnerabilities that can lead to the unreliable and vulnerable system. So, in our work, we try to address such situations and propose a comprehensive framework to counteract the security issues.

摘要: 使用大型语言模型的自主人工智能代理可以在社会各个领域创造不可否认的价值，但他们面临来自对手的安全威胁，需要立即采取保护性解决方案，因为信任和安全问题会出现。考虑到多发越狱和欺骗性对准是一些主要的高级攻击，在监督训练期间使用的静态护栏无法减轻这些攻击，指出了现实世界健壮性的关键研究重点。动态多智能体系统中静态护栏的组合不能抵抗这些攻击。我们打算通过制定新的评估框架，确定和应对安全行动部署所面临的威胁，从而加强基于LLM的特工的安全。我们的工作使用了三种检测方法来通过反向图灵测试来检测流氓代理，并通过多代理模拟来分析欺骗性比对，并开发了一个反越狱系统，通过使用Gemini 1.5Pro和Llama-3.3-70B、使用工具中介的对抗场景来测试DeepSeek R1模型来开发反越狱系统。Gemini 1.5 PRO具有很强的检测能力，如94%的准确率，但在长时间攻击下，随着提示长度的增加，攻击成功率(ASR)增加，多样性度量在预测多个复杂系统故障时变得无效，系统存在持续漏洞。这些发现证明了采用基于主动监控的灵活安全系统的必要性，该系统可以由代理自己执行，并由系统管理员进行适应性干预，因为当前的模型可能会产生漏洞，从而导致系统不可靠和易受攻击。因此，在我们的工作中，我们试图解决这些情况，并提出一个全面的框架来对抗安全问题。



## **33. Keeping up with dynamic attackers: Certifying robustness to adaptive online data poisoning**

跟上动态攻击者：认证自适应在线数据中毒的稳健性 cs.LG

Proceedings of the 28th International Conference on Artificial  Intelligence and Statistics (AISTATS) 2025, Mai Khao, Thailand. PMLR: Volume  258

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16737v1) [paper-pdf](http://arxiv.org/pdf/2502.16737v1)

**Authors**: Avinandan Bose, Laurent Lessard, Maryam Fazel, Krishnamurthy Dj Dvijotham

**Abstract**: The rise of foundation models fine-tuned on human feedback from potentially untrusted users has increased the risk of adversarial data poisoning, necessitating the study of robustness of learning algorithms against such attacks. Existing research on provable certified robustness against data poisoning attacks primarily focuses on certifying robustness for static adversaries who modify a fraction of the dataset used to train the model before the training algorithm is applied. In practice, particularly when learning from human feedback in an online sense, adversaries can observe and react to the learning process and inject poisoned samples that optimize adversarial objectives better than when they are restricted to poisoning a static dataset once, before the learning algorithm is applied. Indeed, it has been shown in prior work that online dynamic adversaries can be significantly more powerful than static ones. We present a novel framework for computing certified bounds on the impact of dynamic poisoning, and use these certificates to design robust learning algorithms. We give an illustration of the framework for the mean estimation and binary classification problems and outline directions for extending this in further work. The code to implement our certificates and replicate our results is available at https://github.com/Avinandan22/Certified-Robustness.

摘要: 根据潜在不可信用户的人类反馈进行微调的基础模型的兴起，增加了敌意数据中毒的风险，因此有必要研究学习算法对此类攻击的稳健性。现有的针对数据中毒攻击的可证明认证稳健性的研究主要集中在认证静态攻击者的健壮性，这些静态攻击者在应用训练算法之前修改了用于训练模型的数据集的一小部分。在实践中，特别是在从在线意义上的人类反馈学习时，攻击者可以观察学习过程并对其做出反应，并注入有毒样本，以更好地优化对抗目标，而不是在应用学习算法之前限制他们一次毒化静态数据集。事实上，以前的工作已经表明，在线动态对手可能比静态对手强大得多。我们提出了一种新的框架来计算动态中毒影响的认证界，并使用这些证书来设计健壮的学习算法。我们给出了均值估计和二分类问题的框架，并概述了在进一步工作中扩展该框架的方向。实现我们的证书和复制我们的结果的代码可在https://github.com/Avinandan22/Certified-Robustness.上获得



## **34. Towards Optimal Adversarial Robust Reinforcement Learning with Infinity Measurement Error**

迈向具有无限测量误差的最佳对抗鲁棒强化学习 cs.LG

arXiv admin note: substantial text overlap with arXiv:2402.02165

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16734v1) [paper-pdf](http://arxiv.org/pdf/2502.16734v1)

**Authors**: Haoran Li, Zicheng Zhang, Wang Luo, Congying Han, Jiayu Lv, Tiande Guo, Yudong Hu

**Abstract**: Ensuring the robustness of deep reinforcement learning (DRL) agents against adversarial attacks is critical for their trustworthy deployment. Recent research highlights the challenges of achieving state-adversarial robustness and suggests that an optimal robust policy (ORP) does not always exist, complicating the enforcement of strict robustness constraints. In this paper, we further explore the concept of ORP. We first introduce the Intrinsic State-adversarial Markov Decision Process (ISA-MDP), a novel formulation where adversaries cannot fundamentally alter the intrinsic nature of state observations. ISA-MDP, supported by empirical and theoretical evidence, universally characterizes decision-making under state-adversarial paradigms. We rigorously prove that within ISA-MDP, a deterministic and stationary ORP exists, aligning with the Bellman optimal policy. Our findings theoretically reveal that improving DRL robustness does not necessarily compromise performance in natural environments. Furthermore, we demonstrate the necessity of infinity measurement error (IME) in both $Q$-function and probability spaces to achieve ORP, unveiling vulnerabilities of previous DRL algorithms that rely on $1$-measurement errors. Motivated by these insights, we develop the Consistent Adversarial Robust Reinforcement Learning (CAR-RL) framework, which optimizes surrogates of IME. We apply CAR-RL to both value-based and policy-based DRL algorithms, achieving superior performance and validating our theoretical analysis.

摘要: 确保深度强化学习(DRL)代理对对手攻击的健壮性是其可信部署的关键。最近的研究强调了实现状态对抗健壮性的挑战，并表明最优健壮性策略(ORP)并不总是存在的，这使得严格健壮性约束的实施复杂化。在本文中，我们进一步探讨了ORP的概念。我们首先介绍了本征状态-对抗马尔可夫决策过程(ISA-MDP)，这是一种新的形式，对手不能从根本上改变状态观测的内在性质。ISA-MDP得到了经验和理论证据的支持，是国家对抗范式下决策的普遍特征。我们严格地证明了在ISA-MDP中，存在与Bellman最优策略一致的确定性且平稳的ORP。我们的发现在理论上表明，提高DRL的健壮性并不一定会损害自然环境中的性能。此外，我们论证了在$q$函数和概率空间中无穷大测量误差(IME)实现ORP的必要性，揭示了以前依赖于$1$测量误差的DRL算法的弱点。受此启发，我们提出了一致对抗性稳健强化学习(CAR-RL)框架，该框架对输入法的代理进行了优化。我们将CAR-RL应用于基于值的DRL算法和基于策略的DRL算法，取得了优异的性能，验证了我们的理论分析。



## **35. Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond**

迈向LLM摆脱学习对重新学习攻击的弹性：敏锐意识的最小化视角及超越 cs.LG

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.05374v2) [paper-pdf](http://arxiv.org/pdf/2502.05374v2)

**Authors**: Chongyu Fan, Jinghan Jia, Yihua Zhang, Anil Ramakrishna, Mingyi Hong, Sijia Liu

**Abstract**: The LLM unlearning technique has recently been introduced to comply with data regulations and address the safety and ethical concerns of LLMs by removing the undesired data-model influence. However, state-of-the-art unlearning methods face a critical vulnerability: they are susceptible to ``relearning'' the removed information from a small number of forget data points, known as relearning attacks. In this paper, we systematically investigate how to make unlearned models robust against such attacks. For the first time, we establish a connection between robust unlearning and sharpness-aware minimization (SAM) through a unified robust optimization framework, in an analogy to adversarial training designed to defend against adversarial attacks. Our analysis for SAM reveals that smoothness optimization plays a pivotal role in mitigating relearning attacks. Thus, we further explore diverse smoothing strategies to enhance unlearning robustness. Extensive experiments on benchmark datasets, including WMDP and MUSE, demonstrate that SAM and other smoothness optimization approaches consistently improve the resistance of LLM unlearning to relearning attacks. Notably, smoothness-enhanced unlearning also helps defend against (input-level) jailbreaking attacks, broadening our proposal's impact in robustifying LLM unlearning. Codes are available at https://github.com/OPTML-Group/Unlearn-Smooth.

摘要: 最近引入了LLM解除学习技术，以遵守数据法规，并通过消除不希望看到的数据模型影响来解决LLM的安全和伦理问题。然而，最先进的遗忘方法面临着一个严重的漏洞：它们容易受到从少数忘记数据点移除的信息的“重新学习”，称为重新学习攻击。在本文中，我们系统地研究了如何使未学习模型对此类攻击具有健壮性。第一次，我们通过一个统一的稳健优化框架在稳健遗忘和敏锐度感知最小化(SAM)之间建立了联系，类似于旨在防御对手攻击的对抗性训练。我们对SAM的分析表明，平滑优化在减轻再学习攻击方面起着关键作用。因此，我们进一步探索不同的平滑策略来增强遗忘的稳健性。在WMDP和MUSE等基准数据集上的大量实验表明，SAM和其他平滑优化方法一致地提高了LLM遗忘对重新学习攻击的抵抗力。值得注意的是，流畅性增强的遗忘也有助于防御(输入级)越狱攻击，扩大了我们的提议在强化LLM遗忘方面的影响。有关代码，请访问https://github.com/OPTML-Group/Unlearn-Smooth.



## **36. Uncovering the Hidden Threat of Text Watermarking from Users with Cross-Lingual Knowledge**

从具有跨语言知识的用户手中发现文本水印的隐藏威胁 cs.CL

9 pages

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16699v1) [paper-pdf](http://arxiv.org/pdf/2502.16699v1)

**Authors**: Mansour Al Ghanim, Jiaqi Xue, Rochana Prih Hastuti, Mengxin Zheng, Yan Solihin, Qian Lou

**Abstract**: In this study, we delve into the hidden threats posed to text watermarking by users with cross-lingual knowledge. While most research focuses on watermarking methods for English, there is a significant gap in evaluating these methods in cross-lingual contexts. This oversight neglects critical adversary scenarios involving cross-lingual users, creating uncertainty regarding the effectiveness of cross-lingual watermarking. We assess four watermarking techniques across four linguistically rich languages, examining watermark resilience and text quality across various parameters and attacks. Our focus is on a realistic scenario featuring adversaries with cross-lingual expertise, evaluating the adequacy of current watermarking methods against such challenges.

摘要: 在这项研究中，我们深入研究了具有跨语言知识的用户对文本水印构成的隐藏威胁。虽然大多数研究都集中在英语的水印方法上，但在跨语言环境中评估这些方法存在显着差距。这种疏忽忽视了涉及跨语言用户的关键对手场景，从而产生了跨语言水印有效性的不确定性。我们评估了四种语言丰富的语言的四种水印技术，检查了各种参数和攻击的水印弹性和文本质量。我们的重点是以具有跨语言专业知识的对手为特色的现实场景，评估当前水印方法应对此类挑战的充分性。



## **37. AdverX-Ray: Ensuring X-Ray Integrity Through Frequency-Sensitive Adversarial VAEs**

DTS X射线：通过频率敏感对抗VAE确保X射线完整性 cs.CV

SPIE Medical Imaging 2025 Runner-up 2025 Robert F. Wagner  All-Conference Best Student Paper Award

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16610v1) [paper-pdf](http://arxiv.org/pdf/2502.16610v1)

**Authors**: Francisco Caetano, Christiaan Viviers, Lena Filatova, Peter H. N. de With, Fons van der Sommen

**Abstract**: Ensuring the quality and integrity of medical images is crucial for maintaining diagnostic accuracy in deep learning-based Computer-Aided Diagnosis and Computer-Aided Detection (CAD) systems. Covariate shifts are subtle variations in the data distribution caused by different imaging devices or settings and can severely degrade model performance, similar to the effects of adversarial attacks. Therefore, it is vital to have a lightweight and fast method to assess the quality of these images prior to using CAD models. AdverX-Ray addresses this need by serving as an image-quality assessment layer, designed to detect covariate shifts effectively. This Adversarial Variational Autoencoder prioritizes the discriminator's role, using the suboptimal outputs of the generator as negative samples to fine-tune the discriminator's ability to identify high-frequency artifacts. Images generated by adversarial networks often exhibit severe high-frequency artifacts, guiding the discriminator to focus excessively on these components. This makes the discriminator ideal for this approach. Trained on patches from X-ray images of specific machine models, AdverX-Ray can evaluate whether a scan matches the training distribution, or if a scan from the same machine is captured under different settings. Extensive comparisons with various OOD detection methods show that AdverX-Ray significantly outperforms existing techniques, achieving a 96.2% average AUROC using only 64 random patches from an X-ray. Its lightweight and fast architecture makes it suitable for real-time applications, enhancing the reliability of medical imaging systems. The code and pretrained models are publicly available.

摘要: 在基于深度学习的计算机辅助诊断和计算机辅助检测(CAD)系统中，确保医学图像的质量和完整性对于保持诊断的准确性至关重要。协变量漂移是由不同成像设备或设置引起的数据分布中的细微变化，可能会严重降低模型的性能，类似于对抗性攻击的影响。因此，在使用CAD模型之前，有一种轻量级且快速的方法来评估这些图像的质量是至关重要的。AdverX-Ray通过作为图像质量评估层来满足这一需求，旨在有效地检测协变量偏移。这种对抗性变分自动编码器优先考虑鉴别器的作用，使用发生器的次优输出作为负样本来微调鉴别器识别高频伪像的能力。对抗性网络生成的图像通常表现出严重的高频伪影，引导鉴别器过度关注这些分量。这使得鉴别器成为这种方法的理想选择。在特定机器型号的X射线图像的补丁上进行训练后，AdverX-Ray可以评估扫描是否与训练分布匹配，或者来自同一机器的扫描是否在不同设置下捕获。与各种面向对象检测方法的广泛比较表明，AdverX-Ray的性能明显优于现有技术，仅使用来自一条X射线的随机斑块就可以获得96.2%的平均AUROC。其轻量级和快速的体系结构使其适合实时应用，提高了医学成像系统的可靠性。代码和预先训练的模型是公开提供的。



## **38. Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images**

通过参数学习对抗图像跟踪大型视觉语言模型的版权 cs.AI

Accepted to ICLR 2025

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16593v1) [paper-pdf](http://arxiv.org/pdf/2502.16593v1)

**Authors**: Yubo Wang, Jianting Tang, Chaohu Liu, Linli Xu

**Abstract**: Large vision-language models (LVLMs) have demonstrated remarkable image understanding and dialogue capabilities, allowing them to handle a variety of visual question answering tasks. However, their widespread availability raises concerns about unauthorized usage and copyright infringement, where users or individuals can develop their own LVLMs by fine-tuning published models. In this paper, we propose a novel method called Parameter Learning Attack (PLA) for tracking the copyright of LVLMs without modifying the original model. Specifically, we construct adversarial images through targeted attacks against the original model, enabling it to generate specific outputs. To ensure these attacks remain effective on potential fine-tuned models to trigger copyright tracking, we allow the original model to learn the trigger images by updating parameters in the opposite direction during the adversarial attack process. Notably, the proposed method can be applied after the release of the original model, thus not affecting the model's performance and behavior. To simulate real-world applications, we fine-tune the original model using various strategies across diverse datasets, creating a range of models for copyright verification. Extensive experiments demonstrate that our method can more effectively identify the original copyright of fine-tuned models compared to baseline methods. Therefore, this work provides a powerful tool for tracking copyrights and detecting unlicensed usage of LVLMs.

摘要: 大型视觉语言模型(LVLM)已经显示出非凡的图像理解和对话能力，使它们能够处理各种视觉问题回答任务。然而，它们的广泛使用引发了人们对未经授权使用和侵犯版权的担忧，在这种情况下，用户或个人可以通过微调已发布的模型来开发自己的LVLM。在本文中，我们提出了一种称为参数学习攻击的新方法，该方法在不修改原始模型的情况下跟踪LVLMS的版权。具体地说，我们通过对原始模型进行有针对性的攻击来构建对抗性图像，使其能够生成特定的输出。为了确保这些攻击在触发版权跟踪的潜在微调模型上保持有效，我们允许原始模型在对抗性攻击过程中通过反向更新参数来学习触发图像。值得注意的是，所提出的方法可以在原始模型发布之后应用，因此不会影响模型的性能和行为。为了模拟真实世界的应用程序，我们使用不同的策略对原始模型进行微调，创建了一系列用于版权验证的模型。大量实验表明，与基线方法相比，该方法能更有效地识别微调模型的原始版权。因此，这项工作为跟踪版权和检测未经许可的LVLM使用提供了一个强大的工具。



## **39. Robust Kernel Hypothesis Testing under Data Corruption**

数据腐败下的鲁棒核假设测试 stat.ML

22 pages, 2 figures, 2 algorithms

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2405.19912v2) [paper-pdf](http://arxiv.org/pdf/2405.19912v2)

**Authors**: Antonin Schrab, Ilmun Kim

**Abstract**: We propose a general method for constructing robust permutation tests under data corruption. The proposed tests effectively control the non-asymptotic type I error under data corruption, and we prove their consistency in power under minimal conditions. This contributes to the practical deployment of hypothesis tests for real-world applications with potential adversarial attacks. For the two-sample and independence settings, we show that our kernel robust tests are minimax optimal, in the sense that they are guaranteed to be non-asymptotically powerful against alternatives uniformly separated from the null in the kernel MMD and HSIC metrics at some optimal rate (tight with matching lower bound). We point out that existing differentially private tests can be adapted to be robust to data corruption, and we demonstrate in experiments that our proposed tests achieve much higher power than these private tests. Finally, we provide publicly available implementations and empirically illustrate the practicality of our robust tests.

摘要: 我们提出了一种在数据损坏情况下构造稳健置换测试的通用方法。所提出的测试有效地控制了数据损坏情况下的非渐近I类错误，并在最小条件下证明了它们在功率上的一致性。这有助于为具有潜在对手攻击的真实世界应用程序实际部署假设检验。对于两样本和独立设置，我们证明了我们的核稳健测试是极小极大最优的，在某种意义上，它们对于在核MMD和HSIC度量中以某种最优率(紧与匹配下界)从零一致分离的备选方案被保证是非渐近强大的。我们指出，现有的差异私有测试可以被改造成对数据损坏具有健壮性，并且我们在实验中证明了我们提出的测试比这些私有测试获得了更高的能力。最后，我们提供了公开可用的实现，并经验地说明了我们的健壮测试的实用性。



## **40. Class-Conditional Neural Polarizer: A Lightweight and Effective Backdoor Defense by Purifying Poisoned Features**

类条件神经极化器：通过净化有毒特征来实现轻量级且有效的后门防御 cs.CR

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.18520v1) [paper-pdf](http://arxiv.org/pdf/2502.18520v1)

**Authors**: Mingli Zhu, Shaokui Wei, Hongyuan Zha, Baoyuan Wu

**Abstract**: Recent studies have highlighted the vulnerability of deep neural networks to backdoor attacks, where models are manipulated to rely on embedded triggers within poisoned samples, despite the presence of both benign and trigger information. While several defense methods have been proposed, they often struggle to balance backdoor mitigation with maintaining benign performance.In this work, inspired by the concept of optical polarizer-which allows light waves of specific polarizations to pass while filtering others-we propose a lightweight backdoor defense approach, NPD. This method integrates a neural polarizer (NP) as an intermediate layer within the compromised model, implemented as a lightweight linear transformation optimized via bi-level optimization. The learnable NP filters trigger information from poisoned samples while preserving benign content. Despite its effectiveness, we identify through empirical studies that NPD's performance degrades when the target labels (required for purification) are inaccurately estimated. To address this limitation while harnessing the potential of targeted adversarial mitigation, we propose class-conditional neural polarizer-based defense (CNPD). The key innovation is a fusion module that integrates the backdoored model's predicted label with the features to be purified. This architecture inherently mimics targeted adversarial defense mechanisms without requiring label estimation used in NPD. We propose three implementations of CNPD: the first is r-CNPD, which trains a replicated NP layer for each class and, during inference, selects the appropriate NP layer for defense based on the predicted class from the backdoored model. To efficiently handle a large number of classes, two variants are designed: e-CNPD, which embeds class information as additional features, and a-CNPD, which directs network attention using class information.

摘要: 最近的研究突显了深度神经网络在后门攻击中的脆弱性，在后门攻击中，模型被操纵，依赖于有毒样本中嵌入的触发器，尽管存在良性和触发信息。虽然已经提出了几种防御方法，但它们往往难以在后门缓解和保持良好性能之间取得平衡。在这项工作中，受光学偏振器的概念启发-允许特定偏振的光波通过而过滤其他偏振的光波-我们提出了一种轻量级后门防御方法NPD。该方法在折衷模型中集成了一个神经偏振器(NP)作为中间层，通过双层优化实现了一个轻量级线性变换。可学习的NP过滤器从有毒的样本中触发信息，同时保留良性内容。尽管NPD是有效的，但我们通过实证研究发现，当目标标记(提纯所需)被错误估计时，NPD的性能会下降。为了解决这一局限性，同时利用定向对抗缓解的潜力，我们提出了基于类条件神经极化器的防御(CNPD)。关键的创新是一个融合模块，它将回溯模型的预测标签与要提纯的特征相结合。这种体系结构本质上模仿了目标对抗性防御机制，而不需要在NPD中使用标签估计。我们提出了CNPD的三种实现：第一种是r-CNPD，它为每一类训练一个复制的NP层，并在推理过程中根据预测的类从反向模型中选择合适的NP层用于防御。为了有效地处理大量的类，设计了两个变体：E-CNPD，它将类信息作为附加特征嵌入；a-CNPD，它使用类信息来引导网络注意力。



## **41. Certified Causal Defense with Generalizable Robustness**

经过认证的因果辩护，具有可概括的稳健性 cs.LG

Submitted to AAAI

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2408.15451v2) [paper-pdf](http://arxiv.org/pdf/2408.15451v2)

**Authors**: Yiran Qiao, Yu Yin, Chen Chen, Jing Ma

**Abstract**: While machine learning models have proven effective across various scenarios, it is widely acknowledged that many models are vulnerable to adversarial attacks. Recently, there have emerged numerous efforts in adversarial defense. Among them, certified defense is well known for its theoretical guarantees against arbitrary adversarial perturbations on input within a certain range (e.g., $l_2$ ball). However, most existing works in this line struggle to generalize their certified robustness in other data domains with distribution shifts. This issue is rooted in the difficulty of eliminating the negative impact of spurious correlations on robustness in different domains. To address this problem, in this work, we propose a novel certified defense framework GLEAN, which incorporates a causal perspective into the generalization problem in certified defense. More specifically, our framework integrates a certifiable causal factor learning component to disentangle the causal relations and spurious correlations between input and label, and thereby exclude the negative effect of spurious correlations on defense. On top of that, we design a causally certified defense strategy to handle adversarial attacks on latent causal factors. In this way, our framework is not only robust against malicious noises on data in the training distribution but also can generalize its robustness across domains with distribution shifts. Extensive experiments on benchmark datasets validate the superiority of our framework in certified robustness generalization in different data domains. Code is available in the supplementary materials.

摘要: 虽然机器学习模型已被证明在各种情况下都有效，但人们普遍认为，许多模型容易受到对手攻击。最近，在对抗性防御方面出现了许多努力。其中，认证防御以其在一定范围内(例如，$L_2$球)对输入的任意对抗性扰动的理论保证而闻名。然而，这一领域的大多数现有工作都很难在具有分布偏移的其他数据域中推广其已证明的健壮性。这个问题的根源在于很难消除虚假相关性对不同领域稳健性的负面影响。针对这一问题，在本文中，我们提出了一种新的认证防御框架GLEAN，该框架将因果视角融入到认证防御的泛化问题中。更具体地说，我们的框架集成了一个可证明的因果因素学习组件，以分离输入和标签之间的因果关系和伪关联，从而排除伪关联对防御的负面影响。最重要的是，我们设计了一个因果认证的防御策略来处理对潜在因果因素的对抗性攻击。这样，我们的框架不仅对训练分布中数据的恶意噪声具有健壮性，而且可以通过分布偏移来推广其跨域的健壮性。在基准数据集上的大量实验验证了该框架在不同数据域的健壮性泛化方面的优越性。代码可在补充材料中找到。



## **42. Unified Prompt Attack Against Text-to-Image Generation Models**

针对文本到图像生成模型的统一提示攻击 cs.CV

Accepted by IEEE T-PAMI 2025

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16423v1) [paper-pdf](http://arxiv.org/pdf/2502.16423v1)

**Authors**: Duo Peng, Qiuhong Ke, Mark He Huang, Ping Hu, Jun Liu

**Abstract**: Text-to-Image (T2I) models have advanced significantly, but their growing popularity raises security concerns due to their potential to generate harmful images. To address these issues, we propose UPAM, a novel framework to evaluate the robustness of T2I models from an attack perspective. Unlike prior methods that focus solely on textual defenses, UPAM unifies the attack on both textual and visual defenses. Additionally, it enables gradient-based optimization, overcoming reliance on enumeration for improved efficiency and effectiveness. To handle cases where T2I models block image outputs due to defenses, we introduce Sphere-Probing Learning (SPL) to enable optimization even without image results. Following SPL, our model bypasses defenses, inducing the generation of harmful content. To ensure semantic alignment with attacker intent, we propose Semantic-Enhancing Learning (SEL) for precise semantic control. UPAM also prioritizes the naturalness of adversarial prompts using In-context Naturalness Enhancement (INE), making them harder for human examiners to detect. Additionally, we address the issue of iterative queries--common in prior methods and easily detectable by API defenders--by introducing Transferable Attack Learning (TAL), allowing effective attacks with minimal queries. Extensive experiments validate UPAM's superiority in effectiveness, efficiency, naturalness, and low query detection rates.

摘要: 文本到图像(T2I)模式已经有了很大的进步，但由于它们可能生成有害的图像，因此它们越来越受欢迎，引发了安全问题。为了解决这些问题，我们提出了一种从攻击角度评估T2I模型健壮性的新框架UPAM。与以前只关注文本防御的方法不同，UPAM将对文本防御和视觉防御的攻击统一起来。此外，它还支持基于梯度的优化，克服了对枚举的依赖，从而提高了效率和效果。为了处理T2I模型由于防御而阻止图像输出的情况，我们引入了球面探测学习(SPL)，即使在没有图像结果的情况下也能实现优化。在SPL之后，我们的模型绕过了防御，诱导了有害内容的生成。为了确保语义与攻击者意图的一致性，我们提出了语义增强学习(SEL)来进行精确的语义控制。UPAM还使用上下文中的自然度增强(INE)对对抗性提示的自然性进行优先排序，使人类审查员更难发现它们。此外，我们通过引入可转移攻击学习(TAL)来解决迭代查询的问题--迭代查询在以前的方法中很常见，并且很容易被API捍卫者检测到，从而允许以最少的查询进行有效的攻击。大量实验验证了UPAM在有效性、效率、自然度和较低的查询检测率方面的优势。



## **43. FedNIA: Noise-Induced Activation Analysis for Mitigating Data Poisoning in FL**

FedNIA：缓解FL数据中毒的噪音诱导激活分析 cs.LG

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16396v1) [paper-pdf](http://arxiv.org/pdf/2502.16396v1)

**Authors**: Ehsan Hallaji, Roozbeh Razavi-Far, Mehrdad Saif

**Abstract**: Federated learning systems are increasingly threatened by data poisoning attacks, where malicious clients compromise global models by contributing tampered updates. Existing defenses often rely on impractical assumptions, such as access to a central test dataset, or fail to generalize across diverse attack types, particularly those involving multiple malicious clients working collaboratively. To address this, we propose Federated Noise-Induced Activation Analysis (FedNIA), a novel defense framework to identify and exclude adversarial clients without relying on any central test dataset. FedNIA injects random noise inputs to analyze the layerwise activation patterns in client models leveraging an autoencoder that detects abnormal behaviors indicative of data poisoning. FedNIA can defend against diverse attack types, including sample poisoning, label flipping, and backdoors, even in scenarios with multiple attacking nodes. Experimental results on non-iid federated datasets demonstrate its effectiveness and robustness, underscoring its potential as a foundational approach for enhancing the security of federated learning systems.

摘要: 联邦学习系统越来越受到数据中毒攻击的威胁，恶意客户端通过提供被篡改的更新来危害全球模型。现有的防御通常依赖于不切实际的假设，例如访问中央测试数据集，或者无法概括各种攻击类型，特别是涉及多个恶意客户端协同工作的攻击类型。为了解决这一问题，我们提出了联邦噪声诱导激活分析(FedNIA)，这是一个新的防御框架，可以识别和排除恶意客户端，而不依赖于任何中央测试数据集。FedNIA利用自动编码器检测指示数据中毒的异常行为，注入随机噪声输入来分析客户端模型中的LayerWise激活模式。FedNIA可以防御多种攻击类型，包括样本中毒、标签翻转和后门，即使在具有多个攻击节点的情况下也是如此。在非IID联合数据集上的实验结果证明了该方法的有效性和稳健性，强调了它作为提高联合学习系统安全性的基础方法的潜力。



## **44. A Framework for Evaluating Vision-Language Model Safety: Building Trust in AI for Public Sector Applications**

评估视觉语言模型安全性的框架：为公共部门应用建立对人工智能的信任 cs.CY

AAAI 2025 Workshop on AI for Social Impact: Bridging Innovations in  Finance, Social Media, and Crime Prevention

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16361v1) [paper-pdf](http://arxiv.org/pdf/2502.16361v1)

**Authors**: Maisha Binte Rashid, Pablo Rivas

**Abstract**: Vision-Language Models (VLMs) are increasingly deployed in public sector missions, necessitating robust evaluation of their safety and vulnerability to adversarial attacks. This paper introduces a novel framework to quantify adversarial risks in VLMs. We analyze model performance under Gaussian, salt-and-pepper, and uniform noise, identifying misclassification thresholds and deriving composite noise patches and saliency patterns that highlight vulnerable regions. These patterns are compared against the Fast Gradient Sign Method (FGSM) to assess their adversarial effectiveness. We propose a new Vulnerability Score that combines the impact of random noise and adversarial attacks, providing a comprehensive metric for evaluating model robustness.

摘要: 视觉语言模型（VLM）越来越多地部署在公共部门任务中，因此需要对其安全性和对对抗攻击的脆弱性进行严格评估。本文引入了一种新颖的框架来量化VLM中的对抗风险。我们分析高斯、椒盐和均匀噪音下的模型性能，识别错误分类阈值并推导出突出脆弱区域的复合噪音补丁和显着模式。将这些模式与快速梯度符号法（FGSM）进行比较，以评估其对抗有效性。我们提出了一种新的漏洞分数，它结合了随机噪音和对抗攻击的影响，为评估模型稳健性提供了全面的指标。



## **45. Verification of Bit-Flip Attacks against Quantized Neural Networks**

针对量化神经网络的位翻转攻击的验证 cs.CR

37 pages, 13 figures, 14 tables

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16286v1) [paper-pdf](http://arxiv.org/pdf/2502.16286v1)

**Authors**: Yedi Zhang, Lei Huang, Pengfei Gao, Fu Song, Jun Sun, Jin Song Dong

**Abstract**: In the rapidly evolving landscape of neural network security, the resilience of neural networks against bit-flip attacks (i.e., an attacker maliciously flips an extremely small amount of bits within its parameter storage memory system to induce harmful behavior), has emerged as a relevant area of research. Existing studies suggest that quantization may serve as a viable defense against such attacks. Recognizing the documented susceptibility of real-valued neural networks to such attacks and the comparative robustness of quantized neural networks (QNNs), in this work, we introduce BFAVerifier, the first verification framework designed to formally verify the absence of bit-flip attacks or to identify all vulnerable parameters in a sound and rigorous manner. BFAVerifier comprises two integral components: an abstraction-based method and an MILP-based method. Specifically, we first conduct a reachability analysis with respect to symbolic parameters that represent the potential bit-flip attacks, based on a novel abstract domain with a sound guarantee. If the reachability analysis fails to prove the resilience of such attacks, then we encode this verification problem into an equivalent MILP problem which can be solved by off-the-shelf solvers. Therefore, BFAVerifier is sound, complete, and reasonably efficient. We conduct extensive experiments, which demonstrate its effectiveness and efficiency across various network architectures, quantization bit-widths, and adversary capabilities.

摘要: 在迅速发展的神经网络安全格局中，神经网络对位翻转攻击(即攻击者恶意翻转其参数存储存储系统中极少量的位以诱导有害行为)的弹性已成为一个相关的研究领域。现有的研究表明，量化可能是一种可行的防御此类攻击的方法。考虑到实值神经网络对此类攻击的易感性和量化神经网络(QNN)的相对稳健性，在本工作中，我们引入了BFAVerizer，这是第一个验证框架，旨在正式验证不存在比特翻转攻击或以合理和严格的方式识别所有易受攻击的参数。BFAVerator由两个完整的组件组成：基于抽象的方法和基于MILP的方法。具体地说，我们首先对代表潜在比特翻转攻击的符号参数进行了可达性分析，该分析基于一个新的具有可靠保证的抽象域。如果可达性分析不能证明这种攻击的弹性，那么我们将该验证问题编码成一个等价的MILP问题，该问题可以通过现成的求解器来解决。因此，BFAVerator是完善的、完整的，并且相当高效。我们进行了广泛的实验，这些实验证明了它在各种网络体系结构、量化位宽和对手能力方面的有效性和效率。



## **46. Your Diffusion Model is Secretly a Certifiably Robust Classifier**

您的扩散模型秘密地是一个可认证的稳健分类器 cs.LG

Accepted by NeurIPS 2024. Also named as "Diffusion Models are  Certifiably Robust Classifiers"

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2402.02316v4) [paper-pdf](http://arxiv.org/pdf/2402.02316v4)

**Authors**: Huanran Chen, Yinpeng Dong, Shitong Shao, Zhongkai Hao, Xiao Yang, Hang Su, Jun Zhu

**Abstract**: Generative learning, recognized for its effective modeling of data distributions, offers inherent advantages in handling out-of-distribution instances, especially for enhancing robustness to adversarial attacks. Among these, diffusion classifiers, utilizing powerful diffusion models, have demonstrated superior empirical robustness. However, a comprehensive theoretical understanding of their robustness is still lacking, raising concerns about their vulnerability to stronger future attacks. In this study, we prove that diffusion classifiers possess $O(1)$ Lipschitzness, and establish their certified robustness, demonstrating their inherent resilience. To achieve non-constant Lipschitzness, thereby obtaining much tighter certified robustness, we generalize diffusion classifiers to classify Gaussian-corrupted data. This involves deriving the evidence lower bounds (ELBOs) for these distributions, approximating the likelihood using the ELBO, and calculating classification probabilities via Bayes' theorem. Experimental results show the superior certified robustness of these Noised Diffusion Classifiers (NDCs). Notably, we achieve over 80% and 70% certified robustness on CIFAR-10 under adversarial perturbations with \(\ell_2\) norms less than 0.25 and 0.5, respectively, using a single off-the-shelf diffusion model without any additional data.

摘要: 生成性学习以其对数据分布的有效建模而被公认，在处理分布外实例方面提供了固有的优势，特别是在增强对对手攻击的稳健性方面。其中，扩散分类器利用了强大的扩散模型，表现出了优越的经验稳健性。然而，对它们的健壮性仍然缺乏全面的理论理解，这引发了人们对它们在未来更强大的攻击中的脆弱性的担忧。在这项研究中，我们证明了扩散分类器具有$O(1)$Lipschitz性，并建立了它们被证明的稳健性，证明了它们的内在弹性。为了实现非常数的Lipschitz性，从而获得更紧密的认证稳健性，我们推广了扩散分类器来分类受高斯污染的数据。这涉及到推导这些分布的证据下界(ELBO)，使用ELBO近似似然性，以及通过贝叶斯定理计算分类概率。实验结果表明，这些带噪扩散分类器(NDC)具有良好的鲁棒性。值得注意的是，在没有任何额外数据的情况下，我们使用单个现成的扩散模型，在对抗性扰动下分别获得了超过80%和70%的CIFAR-10的认证鲁棒性。



## **47. ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models**

ELBA-Bench：大型语言模型的高效学习后门攻击基准 cs.CR

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.18511v1) [paper-pdf](http://arxiv.org/pdf/2502.18511v1)

**Authors**: Xuxu Liu, Siyuan Liang, Mengya Han, Yong Luo, Aishan Liu, Xiantao Cai, Zheng He, Dacheng Tao

**Abstract**: Generative large language models are crucial in natural language processing, but they are vulnerable to backdoor attacks, where subtle triggers compromise their behavior. Although backdoor attacks against LLMs are constantly emerging, existing benchmarks remain limited in terms of sufficient coverage of attack, metric system integrity, backdoor attack alignment. And existing pre-trained backdoor attacks are idealized in practice due to resource access constraints. Therefore we establish $\textit{ELBA-Bench}$, a comprehensive and unified framework that allows attackers to inject backdoor through parameter efficient fine-tuning ($\textit{e.g.,}$ LoRA) or without fine-tuning techniques ($\textit{e.g.,}$ In-context-learning). $\textit{ELBA-Bench}$ provides over 1300 experiments encompassing the implementations of 12 attack methods, 18 datasets, and 12 LLMs. Extensive experiments provide new invaluable findings into the strengths and limitations of various attack strategies. For instance, PEFT attack consistently outperform without fine-tuning approaches in classification tasks while showing strong cross-dataset generalization with optimized triggers boosting robustness; Task-relevant backdoor optimization techniques or attack prompts along with clean and adversarial demonstrations can enhance backdoor attack success while preserving model performance on clean samples. Additionally, we introduce a universal toolbox designed for standardized backdoor attack research, with the goal of propelling further progress in this vital area.

摘要: 生成性大型语言模型在自然语言处理中至关重要，但它们很容易受到后门攻击，在后门攻击中，微妙的触发器会危及它们的行为。尽管针对LLM的后门攻击不断涌现，但现有基准在攻击的足够覆盖率、度量系统完整性、后门攻击对齐方面仍然有限。由于资源访问的限制，现有的预先训练的后门攻击在实践中被理想化。因此，我们建立了$\textit{elba-BENCH}$，这是一个全面而统一的框架，允许攻击者通过参数高效微调($\textit{例如，$loa)或不使用微调技术($\textit{例如，$上下文学习)来注入后门。$\textit{ELBA-BENCH}$提供了1300多个实验，包括12种攻击方法、18个数据集和12个LLM的实现。广泛的实验为各种攻击策略的优势和局限性提供了新的宝贵发现。例如，PEFT攻击在分类任务中始终在没有微调方法的情况下表现得更好，同时显示出强大的跨数据集泛化能力，优化的触发器提高了健壮性；与任务相关的后门优化技术或攻击提示以及干净和对抗性的演示可以提高后门攻击的成功率，同时保持干净样本上的模型性能。此外，我们还介绍了一个为标准化后门攻击研究设计的通用工具箱，目的是推动这一重要领域的进一步发展。



## **48. Na'vi or Knave: Jailbreaking Language Models via Metaphorical Avatars**

纳威人或恶棍：通过隐喻化身越狱语言模型 cs.CL

Our study requires further in-depth research to ensure the  comprehensiveness and adequacy of the methodology

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2412.12145v4) [paper-pdf](http://arxiv.org/pdf/2412.12145v4)

**Authors**: Yu Yan, Sheng Sun, Junqi Tong, Min Liu, Qi Li

**Abstract**: Metaphor serves as an implicit approach to convey information, while enabling the generalized comprehension of complex subjects. However, metaphor can potentially be exploited to bypass the safety alignment mechanisms of Large Language Models (LLMs), leading to the theft of harmful knowledge. In our study, we introduce a novel attack framework that exploits the imaginative capacity of LLMs to achieve jailbreaking, the J\underline{\textbf{A}}ilbreak \underline{\textbf{V}}ia \underline{\textbf{A}}dversarial Me\underline{\textbf{TA}} -pho\underline{\textbf{R}} (\textit{AVATAR}). Specifically, to elicit the harmful response, AVATAR extracts harmful entities from a given harmful target and maps them to innocuous adversarial entities based on LLM's imagination. Then, according to these metaphors, the harmful target is nested within human-like interaction for jailbreaking adaptively. Experimental results demonstrate that AVATAR can effectively and transferablly jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs. Our study exposes a security risk in LLMs from their endogenous imaginative capabilities. Furthermore, the analytical study reveals the vulnerability of LLM to adversarial metaphors and the necessity of developing defense methods against jailbreaking caused by the adversarial metaphor. \textcolor{orange}{ \textbf{Warning: This paper contains potentially harmful content from LLMs.}}

摘要: 隐喻是一种隐含的传达信息的方法，同时也使人们能够对复杂的主题进行广义的理解。然而，隐喻可能被利用来绕过大型语言模型(LLM)的安全对齐机制，从而导致有害知识的窃取。在我们的研究中，我们提出了一种新的攻击框架，该框架利用LLMS的想象能力来实现越狱，即J下划线{\extbf{A}}ilBreak\下划线{\extbf{A}}ia\下划线{\extbf{A}}dversarial Me\下划线{\extbf{TA}}-pho\下划线{\extbf{R}}(\textit{阿凡达})。具体地说，为了引发有害反应，阿凡达从给定的有害目标中提取有害实体，并基于LLM的想象力将它们映射到无害的对抗性实体。然后，根据这些隐喻，有害的目标嵌套在类似人类的互动中，以适应越狱。实验结果表明，阿凡达可以有效地、可转移地越狱LLM，并在多个高级LLM上获得最先进的攻击成功率。我们的研究揭示了LLMS的安全风险来自其内生的想象力。此外，分析研究揭示了LLM对对抗性隐喻的脆弱性，以及开发针对对抗性隐喻导致的越狱的防御方法的必要性。\extCOLOR{橙色}{\extbf{警告：此纸张包含来自LLMS的潜在有害内容。}}



## **49. REFINE: Inversion-Free Backdoor Defense via Model Reprogramming**

重新设计：通过模型重编程实现无倒置后门防御 cs.CR

This paper is accept by ICLR 2025. The first two authors contributed  equally to this work. Our code is available at BackdoorBox  (https://github.com/THUYimingLi/BackdoorBox) and Github repository  (https://github.com/WhitolfChen/REFINE). 28 pages

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.18508v1) [paper-pdf](http://arxiv.org/pdf/2502.18508v1)

**Authors**: Yukun Chen, Shuo Shao, Enhao Huang, Yiming Li, Pin-Yu Chen, Zhan Qin, Kui Ren

**Abstract**: Backdoor attacks on deep neural networks (DNNs) have emerged as a significant security threat, allowing adversaries to implant hidden malicious behaviors during the model training phase. Pre-processing-based defense, which is one of the most important defense paradigms, typically focuses on input transformations or backdoor trigger inversion (BTI) to deactivate or eliminate embedded backdoor triggers during the inference process. However, these methods suffer from inherent limitations: transformation-based defenses often fail to balance model utility and defense performance, while BTI-based defenses struggle to accurately reconstruct trigger patterns without prior knowledge. In this paper, we propose REFINE, an inversion-free backdoor defense method based on model reprogramming. REFINE consists of two key components: \textbf{(1)} an input transformation module that disrupts both benign and backdoor patterns, generating new benign features; and \textbf{(2)} an output remapping module that redefines the model's output domain to guide the input transformations effectively. By further integrating supervised contrastive loss, REFINE enhances the defense capabilities while maintaining model utility. Extensive experiments on various benchmark datasets demonstrate the effectiveness of our REFINE and its resistance to potential adaptive attacks.

摘要: 针对深度神经网络(DNN)的后门攻击已成为一种重大的安全威胁，允许攻击者在模型训练阶段植入隐藏的恶意行为。基于预处理的防御是最重要的防御范式之一，它通常侧重于输入转换或后门触发反转(BTI)，以停用或消除推理过程中嵌入的后门触发。然而，这些方法存在固有的局限性：基于变换的防御往往无法平衡模型的实用性和防御性能，而基于BTI的防御在没有先验知识的情况下难以准确地重建触发模式。本文提出了一种基于模型重编程的无倒置后门防御方法REFINE。Reine由两个关键组件组成：\extbf{(1)}一个输入转换模块，它破坏良性模式和后门模式，生成新的良性特征；以及一个输出重新映射模块，它重新定义模型的输出域，以有效地指导输入转换。通过进一步整合监督对比损失，REFINE在保持模型效用的同时增强了防御能力。在不同基准数据集上的大量实验证明了该算法的有效性和对潜在自适应攻击的抵抗能力。



## **50. Exploring Patient Data Requirements in Training Effective AI Models for MRI-based Breast Cancer Classification**

探索训练有效人工智能模型以进行基于MRI的乳腺癌分类的患者数据要求 eess.IV

Accepted for publication in MICCAI 2024 Deep Breast Workshop on AI  and Imaging for Diagnostic and Treatment Challenges in Breast Care

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.18506v1) [paper-pdf](http://arxiv.org/pdf/2502.18506v1)

**Authors**: Solha Kang, Wesley De Neve, Francois Rameau, Utku Ozbulak

**Abstract**: The past decade has witnessed a substantial increase in the number of startups and companies offering AI-based solutions for clinical decision support in medical institutions. However, the critical nature of medical decision-making raises several concerns about relying on external software. Key issues include potential variations in image modalities and the medical devices used to obtain these images, potential legal issues, and adversarial attacks. Fortunately, the open-source nature of machine learning research has made foundation models publicly available and straightforward to use for medical applications. This accessibility allows medical institutions to train their own AI-based models, thereby mitigating the aforementioned concerns. Given this context, an important question arises: how much data do medical institutions need to train effective AI models? In this study, we explore this question in relation to breast cancer detection, a particularly contested area due to the prevalence of this disease, which affects approximately 1 in every 8 women. Through large-scale experiments on various patient sizes in the training set, we show that medical institutions do not need a decade's worth of MRI images to train an AI model that performs competitively with the state-of-the-art, provided the model leverages foundation models. Furthermore, we observe that for patient counts greater than 50, the number of patients in the training set has a negligible impact on the performance of models and that simple ensembles further improve the results without additional complexity.

摘要: 在过去的十年里，为医疗机构的临床决策支持提供基于人工智能的解决方案的初创公司和公司的数量大幅增加。然而，医疗决策的关键性质引发了对依赖外部软件的几个担忧。关键问题包括成像方式和用于获取这些图像的医疗设备的潜在变化，潜在的法律问题，以及对抗性攻击。幸运的是，机器学习研究的开源性质使基础模型公开可用，并直接用于医疗应用。这种可访问性允许医疗机构培训自己的基于人工智能的模型，从而缓解了上述担忧。在这种背景下，一个重要的问题出现了：医疗机构需要多少数据来训练有效的AI模型？在这项研究中，我们探讨了与乳腺癌检测有关的这个问题，由于这种疾病的流行，这是一个特别有争议的领域，大约每8名女性中就有一人受到影响。通过对训练集中不同患者大小的大规模实验，我们表明，医疗机构不需要十年的MRI图像来训练一个与最先进的人工智能模型竞争的人工智能模型，前提是该模型利用了基础模型。此外，我们观察到，对于大于50的患者计数，训练集中的患者数量对模型的性能影响可以忽略不计，简单的集成进一步改善了结果，而不会增加复杂性。



