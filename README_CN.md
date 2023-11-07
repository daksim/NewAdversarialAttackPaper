# Latest Adversarial Attack Papers
**update at 2023-11-07 10:17:15**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. On existence, uniqueness and scalability of adversarial robustness measures for AI classifiers**

AI分类器对抗性鲁棒性测度的存在性、唯一性和可扩展性 stat.ML

16 pages, 3 figures

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2310.14421v2) [paper-pdf](http://arxiv.org/pdf/2310.14421v2)

**Authors**: Illia Horenko

**Abstract**: Simply-verifiable mathematical conditions for existence, uniqueness and explicit analytical computation of minimal adversarial paths (MAP) and minimal adversarial distances (MAD) for (locally) uniquely-invertible classifiers, for generalized linear models (GLM), and for entropic AI (EAI) are formulated and proven. Practical computation of MAP and MAD, their comparison and interpretations for various classes of AI tools (for neuronal networks, boosted random forests, GLM and EAI) are demonstrated on the common synthetic benchmarks: on a double Swiss roll spiral and its extensions, as well as on the two biomedical data problems (for the health insurance claim predictions, and for the heart attack lethality classification). On biomedical applications it is demonstrated how MAP provides unique minimal patient-specific risk-mitigating interventions in the predefined subsets of accessible control variables.

摘要: 提出并证明了(局部)唯一可逆分类器、广义线性模型(GLM)和熵人工智能(EAI)的最小对抗路径(MAP)和最小对抗距离(MAD)的存在唯一性和显式解析计算的简单可验证的数学条件.MAP和MAD的实际计算，以及它们对各种人工智能工具(用于神经元网络、增强随机森林、GLM和EAI)的比较和解释，在常见的合成基准上进行了演示：在双瑞士辊螺旋及其扩展上，以及在两个生物医学数据问题上(用于健康保险索赔预测和心脏病发作死亡分类)。在生物医学应用方面，它展示了MAP如何在可访问的控制变量的预定义子集中提供独特的、最小限度的患者特定风险缓解干预。



## **2. Preserving Privacy in GANs Against Membership Inference Attack**

防止成员推理攻击的GANS中的隐私保护 cs.LG

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2311.03172v1) [paper-pdf](http://arxiv.org/pdf/2311.03172v1)

**Authors**: Mohammadhadi Shateri, Francisco Messina, Fabrice Labeau, Pablo Piantanida

**Abstract**: Generative Adversarial Networks (GANs) have been widely used for generating synthetic data for cases where there is a limited size real-world dataset or when data holders are unwilling to share their data samples. Recent works showed that GANs, due to overfitting and memorization, might leak information regarding their training data samples. This makes GANs vulnerable to Membership Inference Attacks (MIAs). Several defense strategies have been proposed in the literature to mitigate this privacy issue. Unfortunately, defense strategies based on differential privacy are proven to reduce extensively the quality of the synthetic data points. On the other hand, more recent frameworks such as PrivGAN and PAR-GAN are not suitable for small-size training datasets. In the present work, the overfitting in GANs is studied in terms of the discriminator, and a more general measure of overfitting based on the Bhattacharyya coefficient is defined. Then, inspired by Fano's inequality, our first defense mechanism against MIAs is proposed. This framework, which requires only a simple modification in the loss function of GANs, is referred to as the maximum entropy GAN or MEGAN and significantly improves the robustness of GANs to MIAs. As a second defense strategy, a more heuristic model based on minimizing the information leaked from generated samples about the training data points is presented. This approach is referred to as mutual information minimization GAN (MIMGAN) and uses a variational representation of the mutual information to minimize the information that a synthetic sample might leak about the whole training data set. Applying the proposed frameworks to some commonly used data sets against state-of-the-art MIAs reveals that the proposed methods can reduce the accuracy of the adversaries to the level of random guessing accuracy with a small reduction in the quality of the synthetic data samples.

摘要: 生成性对抗网络(GANS)已被广泛用于生成合成数据，用于在现实世界数据集大小有限的情况下或当数据持有者不愿共享他们的数据样本的情况下。最近的研究表明，由于过度适应和记忆，Gans可能会泄露关于其训练数据样本的信息。这使得GAN容易受到成员身份推断攻击(MIA)。文献中已经提出了几种防御策略来缓解这一隐私问题。不幸的是，事实证明，基于差异隐私的防御策略会大大降低合成数据点的质量。另一方面，PrivGAN和PAR-GAN等较新的框架不适合小规模的训练数据集。本文从判别子的角度研究了Gans中的过拟合问题，定义了一种基于Bhattacharyya系数的更一般的过拟合度量。然后，受Fano不等式的启发，提出了我们针对MIA的第一种防御机制。这种框架只需要对GANS的损失函数进行简单的修改，被称为最大熵GAN或Megan，显著提高了GANS对MIA的鲁棒性。作为第二种防御策略，提出了一种更启发式的模型，该模型基于最小化关于训练数据点的生成样本所泄漏的信息。这种方法被称为互信息最小化GAN(MIMGAN)，它使用互信息的变分表示来最小化合成样本可能泄漏的关于整个训练数据集的信息。将提出的框架应用于一些常用的针对最新MIA的数据集，结果表明，所提出的方法可以在略微降低合成数据样本的质量的情况下，将对手的准确率降低到随机猜测准确率的水平。



## **3. Quantum-Error-Mitigated Detectable Byzantine Agreement with Dynamical Decoupling for Distributed Quantum Computing**

分布式量子计算中抑制量子误差的动态解耦可检测拜占庭协议 quant-ph

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2311.03097v1) [paper-pdf](http://arxiv.org/pdf/2311.03097v1)

**Authors**: Matthew Prest, Kuan-Cheng Chen

**Abstract**: In the burgeoning domain of distributed quantum computing, achieving consensus amidst adversarial settings remains a pivotal challenge. We introduce an enhancement to the Quantum Byzantine Agreement (QBA) protocol, uniquely incorporating advanced error mitigation techniques: Twirled Readout Error Extinction (T-REx) and dynamical decoupling (DD). Central to this refined approach is the utilization of a Noisy Intermediate Scale Quantum (NISQ) source device for heightened performance. Extensive tests on both simulated and real-world quantum devices, notably IBM's quantum computer, provide compelling evidence of the effectiveness of our T-REx and DD adaptations in mitigating prevalent quantum channel errors.   Subsequent to the entanglement distribution, our protocol adopts a verification method reminiscent of Quantum Key Distribution (QKD) schemes. The Commander then issues orders encoded in specific quantum states, like Retreat or Attack. In situations where received orders diverge, lieutenants engage in structured games to reconcile discrepancies. Notably, the frequency of these games is contingent upon the Commander's strategies and the overall network size. Our empirical findings underscore the enhanced resilience and effectiveness of the protocol in diverse scenarios. Nonetheless, scalability emerges as a concern with the growth of the network size. To sum up, our research illuminates the considerable potential of fortified quantum consensus systems in the NISQ era, highlighting the imperative for sustained research in bolstering quantum ecosystems.

摘要: 在蓬勃发展的分布式量子计算领域，在敌对环境中达成共识仍然是一个关键挑战。我们引入了对量子拜占庭协议(QBA)的增强，独特地结合了先进的错误缓解技术：旋转读出错误消除(T-REX)和动态解耦(DD)。这一改进方法的核心是利用噪声中尺度量子(NISQ)源设备来提高性能。对模拟和真实量子设备的广泛测试，特别是IBM的量子计算机，提供了令人信服的证据，证明我们的T-Rex和DD适配在缓解普遍存在的量子通道错误方面的有效性。在纠缠分发之后，我们的协议采用了一种类似于量子密钥分发(QKD)方案的验证方法。然后，指挥官发布以特定量子状态编码的命令，如撤退或攻击。在收到的订单不同的情况下，副手们会进行结构化的游戏，以协调差异。值得注意的是，这些游戏的频率取决于指挥官的战略和整体网络规模。我们的经验发现强调了该协议在不同情况下的增强的弹性和有效性。尽管如此，随着网络规模的增长，可扩展性成为一个令人担忧的问题。总而言之，我们的研究阐明了NISQ时代强化的量子共识系统的巨大潜力，突显了在支持量子生态系统方面进行持续研究的必要性。



## **4. SoK: Memorisation in machine learning**

SOK：机器学习中的记忆 cs.LG

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2311.03075v1) [paper-pdf](http://arxiv.org/pdf/2311.03075v1)

**Authors**: Dmitrii Usynin, Moritz Knolle, Georgios Kaissis

**Abstract**: Quantifying the impact of individual data samples on machine learning models is an open research problem. This is particularly relevant when complex and high-dimensional relationships have to be learned from a limited sample of the data generating distribution, such as in deep learning. It was previously shown that, in these cases, models rely not only on extracting patterns which are helpful for generalisation, but also seem to be required to incorporate some of the training data more or less as is, in a process often termed memorisation. This raises the question: if some memorisation is a requirement for effective learning, what are its privacy implications? In this work we unify a broad range of previous definitions and perspectives on memorisation in ML, discuss their interplay with model generalisation and their implications of these phenomena on data privacy. Moreover, we systematise methods allowing practitioners to detect the occurrence of memorisation or quantify it and contextualise our findings in a broad range of ML learning settings. Finally, we discuss memorisation in the context of privacy attacks, differential privacy (DP) and adversarial actors.

摘要: 量化单个数据样本对机器学习模型的影响是一个开放的研究问题。当必须从数据生成分布的有限样本中学习复杂和高维关系时，这尤其相关，例如在深度学习中。先前已经表明，在这些情况下，模型不仅依赖于提取有助于泛化的模式，而且似乎还需要或多或少地将一些训练数据纳入一个通常被称为记忆的过程中。这就提出了一个问题：如果记忆是有效学习的必要条件，那么它对隐私的影响是什么？在这项工作中，我们统一了以前关于ML中记忆的广泛定义和观点，讨论了它们与模型泛化的相互作用，以及它们对数据隐私的影响。此外，我们系统化的方法允许从业者检测记忆的发生或量化它，并在广泛的ML学习环境中将我们的发现与背景联系起来。最后，我们讨论了在隐私攻击、差异隐私(DP)和敌对行为的背景下的记忆。



## **5. MAWSEO: Adversarial Wiki Search Poisoning for Illicit Online Promotion**

MAWSEO：对抗性维基搜索中毒非法在线推广 cs.CR

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2304.11300v2) [paper-pdf](http://arxiv.org/pdf/2304.11300v2)

**Authors**: Zilong Lin, Zhengyi Li, Xiaojing Liao, XiaoFeng Wang, Xiaozhong Liu

**Abstract**: As a prominent instance of vandalism edits, Wiki search poisoning for illicit promotion is a cybercrime in which the adversary aims at editing Wiki articles to promote illicit businesses through Wiki search results of relevant queries. In this paper, we report a study that, for the first time, shows that such stealthy blackhat SEO on Wiki can be automated. Our technique, called MAWSEO, employs adversarial revisions to achieve real-world cybercriminal objectives, including rank boosting, vandalism detection evasion, topic relevancy, semantic consistency, user awareness (but not alarming) of promotional content, etc. Our evaluation and user study demonstrate that MAWSEO is capable of effectively and efficiently generating adversarial vandalism edits, which can bypass state-of-the-art built-in Wiki vandalism detectors, and also get promotional content through to Wiki users without triggering their alarms. In addition, we investigated potential defense, including coherence based detection and adversarial training of vandalism detection, against our attack in the Wiki ecosystem.

摘要: 作为破坏编辑的一个突出例子，非法推广的维基搜索中毒是一种网络犯罪，对手旨在编辑维基文章，通过相关查询的维基搜索结果来推广非法业务。在这篇文章中，我们报告了一项研究，首次表明维基上这种隐蔽的黑帽SEO可以自动化。我们的技术，称为MAWSEO，使用对抗性修订来实现现实世界的网络犯罪目标，包括排名提升、破坏检测规避、主题相关性、语义一致性、用户对促销内容的感知(但不令人震惊)等。我们的评估和用户研究表明，MAWSEO能够有效和高效地生成对抗性破坏编辑，这可以绕过最先进的内置维基破坏检测器，还可以在不触发维基用户警报的情况下将宣传内容传递给维基用户。此外，我们调查了针对我们在维基生态系统中的攻击的潜在防御，包括基于一致性的检测和恶意检测的对抗性培训。



## **6. Detecting Language Model Attacks with Perplexity**

基于困惑的语言模型攻击检测 cs.CL

**SubmitDate**: 2023-11-05    [abs](http://arxiv.org/abs/2308.14132v2) [paper-pdf](http://arxiv.org/pdf/2308.14132v2)

**Authors**: Gabriel Alon, Michael Kamfonas

**Abstract**: A novel hack involving Large Language Models (LLMs) has emerged, leveraging adversarial suffixes to trick models into generating perilous responses. This method has garnered considerable attention from reputable media outlets such as the New York Times and Wired, thereby influencing public perception regarding the security and safety of LLMs. In this study, we advocate the utilization of perplexity as one of the means to recognize such potential attacks. The underlying concept behind these hacks revolves around appending an unusually constructed string of text to a harmful query that would otherwise be blocked. This maneuver confuses the protective mechanisms and tricks the model into generating a forbidden response. Such scenarios could result in providing detailed instructions to a malicious user for constructing explosives or orchestrating a bank heist. Our investigation demonstrates the feasibility of employing perplexity, a prevalent natural language processing metric, to detect these adversarial tactics before generating a forbidden response. By evaluating the perplexity of queries with and without such adversarial suffixes using an open-source LLM, we discovered that nearly 90 percent were above a perplexity of 1000. This contrast underscores the efficacy of perplexity for detecting this type of exploit.

摘要: 出现了一种涉及大型语言模型(LLM)的新黑客攻击，利用敌意后缀欺骗模型生成危险的响应。这一方法引起了《纽约时报》和《连线》等知名媒体的极大关注，从而影响了公众对低地小武器安全性和安全性的看法。在这项研究中，我们主张使用困惑作为识别这种潜在攻击的手段之一。这些黑客攻击背后的基本概念围绕着在原本会被阻止的有害查询中附加一个构造异常的文本字符串。这种操作混淆了保护机制，并诱使模型产生禁止反应。这种情况可能会导致向恶意用户提供制造炸药或策划银行抢劫的详细说明。我们的研究证明了使用困惑，一种流行的自然语言处理度量，在生成禁止响应之前检测这些敌对策略的可行性。通过使用开源LLM评估带有和不带有这种敌意后缀的查询的困惑程度，我们发现近90%的查询困惑程度高于1000。这种对比突出了困惑在检测这种类型的利用方面的有效性。



## **7. CT-GAT: Cross-Task Generative Adversarial Attack based on Transferability**

CT-GAT：基于可转移性的跨任务生成性对抗攻击 cs.CL

Accepted to EMNLP 2023 main conference Corrected the header error in  Figure 3

**SubmitDate**: 2023-11-05    [abs](http://arxiv.org/abs/2310.14265v2) [paper-pdf](http://arxiv.org/pdf/2310.14265v2)

**Authors**: Minxuan Lv, Chengwei Dai, Kun Li, Wei Zhou, Songlin Hu

**Abstract**: Neural network models are vulnerable to adversarial examples, and adversarial transferability further increases the risk of adversarial attacks. Current methods based on transferability often rely on substitute models, which can be impractical and costly in real-world scenarios due to the unavailability of training data and the victim model's structural details. In this paper, we propose a novel approach that directly constructs adversarial examples by extracting transferable features across various tasks. Our key insight is that adversarial transferability can extend across different tasks. Specifically, we train a sequence-to-sequence generative model named CT-GAT using adversarial sample data collected from multiple tasks to acquire universal adversarial features and generate adversarial examples for different tasks. We conduct experiments on ten distinct datasets, and the results demonstrate that our method achieves superior attack performance with small cost.

摘要: 神经网络模型容易受到对抗性例子的影响，对抗性的可转移性进一步增加了对抗性攻击的风险。目前基于可转移性的方法往往依赖于替代模型，由于训练数据和受害者模型的结构细节的不可用，这在现实世界的场景中可能是不切实际和昂贵的。在本文中，我们提出了一种新的方法，该方法通过提取跨任务的可转移特征来直接构造对抗性实例。我们的关键洞察是，对抗性转移可以扩展到不同的任务。具体地说，我们使用从多个任务收集的对抗性样本数据来训练序列到序列生成模型CT-GAT，以获取通用的对抗性特征，并为不同的任务生成对抗性实例。我们在10个不同的数据集上进行了实验，结果表明，该方法以较小的代价获得了优越的攻击性能。



## **8. DeepZero: Scaling up Zeroth-Order Optimization for Deep Model Training**

DeepZero：深度模型训练的零阶放大优化 cs.LG

**SubmitDate**: 2023-11-05    [abs](http://arxiv.org/abs/2310.02025v2) [paper-pdf](http://arxiv.org/pdf/2310.02025v2)

**Authors**: Aochuan Chen, Yimeng Zhang, Jinghan Jia, James Diffenderfer, Jiancheng Liu, Konstantinos Parasyris, Yihua Zhang, Zheng Zhang, Bhavya Kailkhura, Sijia Liu

**Abstract**: Zeroth-order (ZO) optimization has become a popular technique for solving machine learning (ML) problems when first-order (FO) information is difficult or impossible to obtain. However, the scalability of ZO optimization remains an open problem: Its use has primarily been limited to relatively small-scale ML problems, such as sample-wise adversarial attack generation. To our best knowledge, no prior work has demonstrated the effectiveness of ZO optimization in training deep neural networks (DNNs) without a significant decrease in performance. To overcome this roadblock, we develop DeepZero, a principled ZO deep learning (DL) framework that can scale ZO optimization to DNN training from scratch through three primary innovations. First, we demonstrate the advantages of coordinate-wise gradient estimation (CGE) over randomized vector-wise gradient estimation in training accuracy and computational efficiency. Second, we propose a sparsity-induced ZO training protocol that extends the model pruning methodology using only finite differences to explore and exploit the sparse DL prior in CGE. Third, we develop the methods of feature reuse and forward parallelization to advance the practical implementations of ZO training. Our extensive experiments show that DeepZero achieves state-of-the-art (SOTA) accuracy on ResNet-20 trained on CIFAR-10, approaching FO training performance for the first time. Furthermore, we show the practical utility of DeepZero in applications of certified adversarial defense and DL-based partial differential equation error correction, achieving 10-20% improvement over SOTA. We believe our results will inspire future research on scalable ZO optimization and contribute to advancing DL with black box.

摘要: 当一阶信息难以或不可能获得时，零阶(ZO)优化已成为解决机器学习(ML)问题的一种流行技术。然而，ZO优化的可伸缩性仍然是一个悬而未决的问题：它的使用主要限于相对较小的ML问题，例如样本智慧的敌意攻击生成。据我们所知，没有任何先前的工作证明ZO优化在不显著降低性能的情况下训练深度神经网络(DNN)的有效性。为了克服这一障碍，我们开发了DeepZero，这是一个原则性的ZO深度学习(DL)框架，可以通过三项主要创新从头开始将ZO优化扩展到DNN培训。首先，我们证明了坐标梯度估计(CGE)在训练精度和计算效率上优于随机向量梯度估计。其次，我们提出了一种稀疏诱导的ZO训练协议，该协议扩展了仅使用有限差分的模型剪枝方法，以探索和利用CGE中的稀疏DL先验。第三，开发了特征重用和前向并行化的方法，推进了ZO训练的实际实现。我们的广泛实验表明，DeepZero在CIFAR-10上训练的ResNet-20上达到了最先进的精度(SOTA)，首次接近FO训练性能。此外，我们还展示了DeepZero在认证对抗防御和基于DL的偏微分方程纠错中的实际应用，比SOTA提高了10%-20%。我们相信，我们的研究结果将对未来可伸缩ZO优化的研究起到启发作用，并为进一步推进黑盒动态链接库做出贡献。



## **9. Exploring Transferability of Multimodal Adversarial Samples for Vision-Language Pre-training Models with Contrastive Learning**

基于对比学习的视觉语言预训练模型多通道对抗性样本可转换性研究 cs.MM

**SubmitDate**: 2023-11-05    [abs](http://arxiv.org/abs/2308.12636v2) [paper-pdf](http://arxiv.org/pdf/2308.12636v2)

**Authors**: Youze Wang, Wenbo Hu, Yinpeng Dong, Hanwang Zhang, Richang Hong

**Abstract**: Vision-language pre-training models (VLP) are vulnerable, especially to multimodal adversarial samples, which can be crafted by adding imperceptible perturbations on both original images and texts. However, under the black-box setting, there have been no works to explore the transferability of multimodal adversarial attacks against the VLP models. In this work, we take CLIP as the surrogate model and propose a gradient-based multimodal attack method to generate transferable adversarial samples against the VLP models. By applying the gradient to optimize the adversarial images and adversarial texts simultaneously, our method can better search for and attack the vulnerable images and text information pairs. To improve the transferability of the attack, we utilize contrastive learning including image-text contrastive learning and intra-modal contrastive learning to have a more generalized understanding of the underlying data distribution and mitigate the overfitting of the surrogate model so that the generated multimodal adversarial samples have a higher transferability for VLP models. Extensive experiments validate the effectiveness of the proposed method.

摘要: 视觉语言预训练模型(VLP)是脆弱的，特别是对多模式对抗性样本，这些样本可以通过在原始图像和文本上添加不可察觉的扰动来构建。然而，在黑箱环境下，还没有研究针对VLP模型的多通道对抗性攻击的可转移性。在本文中，我们以CLIP为代理模型，提出了一种基于梯度的多模式攻击方法来生成可传输的对抗VLP模型的样本。通过应用梯度对敌意图像和敌意文本同时进行优化，该方法能够更好地搜索和攻击易受攻击的图文信息对。为了提高攻击的可转移性，我们利用对比学习，包括图文对比学习和通道内对比学习，以更全面地了解底层数据分布，并减少代理模型的过度拟合，使生成的多通道对抗性样本对VLP模型具有更高的可转移性。大量实验验证了该方法的有效性。



## **10. Unfolding Local Growth Rate Estimates for (Almost) Perfect Adversarial Detection**

关于(几乎)完美敌意检测的局部增长率估计 cs.CV

accepted at VISAPP23

**SubmitDate**: 2023-11-04    [abs](http://arxiv.org/abs/2212.06776v3) [paper-pdf](http://arxiv.org/pdf/2212.06776v3)

**Authors**: Peter Lorenz, Margret Keuper, Janis Keuper

**Abstract**: Convolutional neural networks (CNN) define the state-of-the-art solution on many perceptual tasks. However, current CNN approaches largely remain vulnerable against adversarial perturbations of the input that have been crafted specifically to fool the system while being quasi-imperceptible to the human eye. In recent years, various approaches have been proposed to defend CNNs against such attacks, for example by model hardening or by adding explicit defence mechanisms. Thereby, a small "detector" is included in the network and trained on the binary classification task of distinguishing genuine data from data containing adversarial perturbations. In this work, we propose a simple and light-weight detector, which leverages recent findings on the relation between networks' local intrinsic dimensionality (LID) and adversarial attacks. Based on a re-interpretation of the LID measure and several simple adaptations, we surpass the state-of-the-art on adversarial detection by a significant margin and reach almost perfect results in terms of F1-score for several networks and datasets. Sources available at: https://github.com/adverML/multiLID

摘要: 卷积神经网络(CNN)定义了许多感知任务的最先进的解决方案。然而，目前的CNN方法在很大程度上仍然容易受到输入的对抗性扰动，这些扰动是专门为愚弄系统而设计的，而人眼几乎察觉不到。近年来，已经提出了各种方法来保护CNN免受此类攻击，例如通过模型硬化或通过添加显式防御机制。因此，在网络中包括一个小的“检测器”，并在区分真实数据和包含对抗性扰动的数据的二进制分类任务上进行训练。在这项工作中，我们提出了一个简单而轻量级的检测器，它利用了最近关于网络的局部固有维度(LID)与对手攻击之间关系的研究结果。基于对LID度量的重新解释和几个简单的适应，我们在对手检测方面远远超过了最先进的水平，并在几个网络和数据集的F1得分方面取得了几乎完美的结果。资料来源：https://github.com/adverML/multiLID



## **11. MTS-DVGAN: Anomaly Detection in Cyber-Physical Systems using a Dual Variational Generative Adversarial Network**

MTS-DVGAN：基于对偶变分生成对抗网络的网络物理系统异常检测 cs.CR

27 pages, 14 figures, 8 tables. Accepted by Computers & Security

**SubmitDate**: 2023-11-04    [abs](http://arxiv.org/abs/2311.02378v1) [paper-pdf](http://arxiv.org/pdf/2311.02378v1)

**Authors**: Haili Sun, Yan Huang, Lansheng Han, Cai Fu, Hongle Liu, Xiang Long

**Abstract**: Deep generative models are promising in detecting novel cyber-physical attacks, mitigating the vulnerability of Cyber-physical systems (CPSs) without relying on labeled information. Nonetheless, these generative models face challenges in identifying attack behaviors that closely resemble normal data, or deviate from the normal data distribution but are in close proximity to the manifold of the normal cluster in latent space. To tackle this problem, this article proposes a novel unsupervised dual variational generative adversarial model named MST-DVGAN, to perform anomaly detection in multivariate time series data for CPS security. The central concept is to enhance the model's discriminative capability by widening the distinction between reconstructed abnormal samples and their normal counterparts. Specifically, we propose an augmented module by imposing contrastive constraints on the reconstruction process to obtain a more compact embedding. Then, by exploiting the distribution property and modeling the normal patterns of multivariate time series, a variational autoencoder is introduced to force the generative adversarial network (GAN) to generate diverse samples. Furthermore, two augmented loss functions are designed to extract essential characteristics in a self-supervised manner through mutual guidance between the augmented samples and original samples. Finally, a specific feature center loss is introduced for the generator network to enhance its stability. Empirical experiments are conducted on three public datasets, namely SWAT, WADI and NSL_KDD. Comparing with the state-of-the-art methods, the evaluation results show that the proposed MTS-DVGAN is more stable and can achieve consistent performance improvement.

摘要: 深度生成模型在检测新的网络物理攻击、减轻网络物理系统(CPSS)的脆弱性方面很有希望，而不依赖于标签信息。然而，这些生成性模型在识别与正态数据非常相似的攻击行为，或偏离正态数据分布但在潜在空间中非常接近正态簇流形的攻击行为方面面临挑战。针对这一问题，本文提出了一种新的无监督双变分生成对抗模型MST-DVGAN，用于在多变量时间序列数据中进行异常检测，以保证CPS的安全性。其核心概念是通过扩大重建的异常样本与其正常样本之间的区别来增强模型的区分能力。具体地说，通过对重建过程施加对比约束，我们提出了一种增广模块，以获得更紧凑的嵌入。然后，通过利用多变量时间序列的分布特性和正态模式建模，引入变分自动编码器来强制生成性对抗网络(GAN)生成不同的样本。此外，还设计了两个增广损失函数，通过增广样本和原始样本之间的相互指导，以自监督的方式提取本质特征。最后，为提高发电机网络的稳定性，引入了特定的特征中心损耗。在SWAT、WADI和NSL_KDD三个公共数据集上进行了实验。评估结果表明，与现有方法相比，本文提出的MTS-DVGAN算法具有更高的稳定性，并能实现持续的性能提升。



## **12. From Trojan Horses to Castle Walls: Unveiling Bilateral Backdoor Effects in Diffusion Models**

从特洛伊木马到城堡墙：在扩散模型中揭示双边后门效应 cs.LG

10 pages, 6 figures, 7 tables

**SubmitDate**: 2023-11-04    [abs](http://arxiv.org/abs/2311.02373v1) [paper-pdf](http://arxiv.org/pdf/2311.02373v1)

**Authors**: Zhuoshi Pan, Yuguang Yao, Gaowen Liu, Bingquan Shen, H. Vicky Zhao, Ramana Rao Kompella, Sijia Liu

**Abstract**: While state-of-the-art diffusion models (DMs) excel in image generation, concerns regarding their security persist. Earlier research highlighted DMs' vulnerability to backdoor attacks, but these studies placed stricter requirements than conventional methods like 'BadNets' in image classification. This is because the former necessitates modifications to the diffusion sampling and training procedures. Unlike the prior work, we investigate whether generating backdoor attacks in DMs can be as simple as BadNets, i.e., by only contaminating the training dataset without tampering the original diffusion process. In this more realistic backdoor setting, we uncover bilateral backdoor effects that not only serve an adversarial purpose (compromising the functionality of DMs) but also offer a defensive advantage (which can be leveraged for backdoor defense). Specifically, we find that a BadNets-like backdoor attack remains effective in DMs for producing incorrect images (misaligned with the intended text conditions), and thereby yielding incorrect predictions when DMs are used as classifiers. Meanwhile, backdoored DMs exhibit an increased ratio of backdoor triggers, a phenomenon we refer to as `trigger amplification', among the generated images. We show that this latter insight can be used to enhance the detection of backdoor-poisoned training data. Even under a low backdoor poisoning ratio, studying the backdoor effects of DMs is also valuable for designing anti-backdoor image classifiers. Last but not least, we establish a meaningful linkage between backdoor attacks and the phenomenon of data replications by exploring DMs' inherent data memorization tendencies. The codes of our work are available at https://github.com/OPTML-Group/BiBadDiff.

摘要: 虽然最先进的扩散模型(DM)在图像生成方面表现出色，但对其安全性的担忧依然存在。早期的研究强调了DM对后门攻击的脆弱性，但这些研究在图像分类方面对图像分类提出了比BadNets等传统方法更严格的要求。这是因为前者需要对扩散抽样和训练程序进行修改。与以前的工作不同，我们研究了在DM中生成后门攻击是否可以像BadNets一样简单，即只污染训练数据集而不篡改原始的扩散过程。在这个更现实的后门设置中，我们揭示了双边后门效应，这些后门效应不仅服务于敌对目的(损害DM的功能)，而且提供了防御优势(可用于后门防御)。具体地说，我们发现类似BadNets的后门攻击在DM中仍然有效，因为它会产生错误的图像(与预期的文本条件不一致)，从而在DM用作分类器时产生错误的预测。与此同时，走后门的DM在生成的图像中显示出后门触发器的比例增加，我们将这种现象称为“触发放大”。我们表明，后一种见解可以用于增强对后门中毒训练数据的检测。即使在低后门投毒率的情况下，研究DM的后门效应对于反后门图像分类器的设计也是有价值的。最后但并非最不重要的是，我们通过探索DM固有的数据记忆倾向，在后门攻击和数据复制现象之间建立了有意义的联系。我们工作的代码可以在https://github.com/OPTML-Group/BiBadDiff.上找到



## **13. Secure compilation of rich smart contracts on poor UTXO blockchains**

在贫穷的UTXO区块链上安全地编译丰富的智能合同 cs.CR

**SubmitDate**: 2023-11-04    [abs](http://arxiv.org/abs/2305.09545v2) [paper-pdf](http://arxiv.org/pdf/2305.09545v2)

**Authors**: Massimo Bartoletti, Riccardo Marchesin, Roberto Zunino

**Abstract**: Most blockchain platforms from Ethereum onwards render smart contracts as stateful reactive objects that update their state and transfer crypto-assets in response to transactions. A drawback of this design is that when users submit a transaction, they cannot predict in which state it will be executed. This exposes them to transaction-ordering attacks, a widespread class of attacks where adversaries with the power to construct blocks of transactions can extract value from smart contracts (the so-called MEV attacks). The UTXO model is an alternative blockchain design that thwarts these attacks by requiring new transactions to spend past ones: since transactions have unique identifiers, reordering attacks are ineffective. Currently, the blockchains following the UTXO model either provide contracts with limited expressiveness (Bitcoin), or require complex run-time environments (Cardano). We present ILLUM, an Intermediate-Level Language for the UTXO Model. ILLUM can express real-world smart contracts, e.g. those found in Decentralized Finance. We define a compiler from ILLUM to a bare-bone UTXO blockchain with loop-free scripts. Our compilation target only requires minimal extensions to Bitcoin Script: in particular, we exploit covenants, a mechanism for preserving scripts along chains of transactions. We prove the security of our compiler: namely, any attack targeting the compiled contract is also observable at the ILLUM level. Hence, the compiler does not introduce new vulnerabilities that were not already present in the source ILLUM contract. Finally, we discuss the suitability of ILLUM as a compilation target for higher-level contract languages.

摘要: 从Etherum开始，大多数区块链平台都将智能合约呈现为有状态的反应对象，这些对象更新其状态并传输加密资产以响应交易。这种设计的一个缺点是，当用户提交事务时，他们无法预测该事务将在哪种状态下执行。这使他们面临交易顺序攻击，这是一种广泛存在的攻击类别，在这种攻击中，有能力构建交易块的对手可以从智能合约中提取价值(所谓的MEV攻击)。UTXO模型是一种替代区块链设计，通过要求新交易花费过去的交易来挫败这些攻击：由于交易具有唯一标识符，重新排序攻击是无效的。目前，遵循UTXO模式的区块链要么提供表现力有限的合同(比特币)，要么需要复杂的运行时环境(Cardano)。我们介绍了Illum，一种用于UTXO模型的中级语言。Illum可以表示现实世界中的智能合约，例如在去中心化金融中找到的那些。我们定义了一个从Illum到具有无循环脚本的基本UTXO区块链的编译器。我们的编译目标只需要对比特币脚本进行最小程度的扩展：尤其是，我们利用了契诺，这是一种在交易链上保留脚本的机制。我们证明了我们的编译器的安全性：也就是说，任何针对已编译约定的攻击也可以在Illum级别上观察到。因此，编译器不会引入源Illum协定中尚未存在的新漏洞。最后，我们讨论了Illum作为高级契约语言的编译目标的适用性。



## **14. Generative Adversarial Networks to infer velocity components in rotating turbulent flows**

生成对抗性网络用于推断旋转湍流中的速度分量 physics.flu-dyn

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2301.07541v2) [paper-pdf](http://arxiv.org/pdf/2301.07541v2)

**Authors**: Tianyi Li, Michele Buzzicotti, Luca Biferale, Fabio Bonaccorso

**Abstract**: Inference problems for two-dimensional snapshots of rotating turbulent flows are studied. We perform a systematic quantitative benchmark of point-wise and statistical reconstruction capabilities of the linear Extended Proper Orthogonal Decomposition (EPOD) method, a non-linear Convolutional Neural Network (CNN) and a Generative Adversarial Network (GAN). We attack the important task of inferring one velocity component out of the measurement of a second one, and two cases are studied: (I) both components lay in the plane orthogonal to the rotation axis and (II) one of the two is parallel to the rotation axis. We show that EPOD method works well only for the former case where both components are strongly correlated, while CNN and GAN always outperform EPOD both concerning point-wise and statistical reconstructions. For case (II), when the input and output data are weakly correlated, all methods fail to reconstruct faithfully the point-wise information. In this case, only GAN is able to reconstruct the field in a statistical sense. The analysis is performed using both standard validation tools based on $L_2$ spatial distance between the prediction and the ground truth and more sophisticated multi-scale analysis using wavelet decomposition. Statistical validation is based on standard Jensen-Shannon divergence between the probability density functions, spectral properties and multi-scale flatness.

摘要: 研究了旋转湍流二维快照的推断问题。我们对线性扩展本征正交分解(EPOD)方法、非线性卷积神经网络(CNN)和生成性对抗网络(GAN)的逐点重建和统计重建能力进行了系统的定量基准测试。我们提出了从第二个速度分量的测量中推断出一个速度分量的重要任务，并研究了两种情况：(I)两个分量都位于与旋转轴垂直的平面上；(Ii)两个分量中的一个平行于旋转轴。我们发现，EPOD方法只适用于强相关的前一种情况，而CNN和GAN在逐点重建和统计重建方面总是优于EPOD。对于情况(II)，当输入和输出数据弱相关时，所有方法都不能忠实地重建逐点信息。在这种情况下，只有GaN能够在统计意义上重建场。分析使用了基于$L_2的标准验证工具和更复杂的基于小波分解的多尺度分析。统计验证基于概率密度函数、光谱特性和多尺度平坦度之间的标准Jensen-Shannon散度。



## **15. HINT: Healthy Influential-Noise based Training to Defend against Data Poisoning Attacks**

提示：健康影响-基于噪音的培训可防御数据中毒攻击 cs.LG

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2309.08549v2) [paper-pdf](http://arxiv.org/pdf/2309.08549v2)

**Authors**: Minh-Hao Van, Alycia N. Carey, Xintao Wu

**Abstract**: While numerous defense methods have been proposed to prohibit potential poisoning attacks from untrusted data sources, most research works only defend against specific attacks, which leaves many avenues for an adversary to exploit. In this work, we propose an efficient and robust training approach to defend against data poisoning attacks based on influence functions, named Healthy Influential-Noise based Training. Using influence functions, we craft healthy noise that helps to harden the classification model against poisoning attacks without significantly affecting the generalization ability on test data. In addition, our method can perform effectively when only a subset of the training data is modified, instead of the current method of adding noise to all examples that has been used in several previous works. We conduct comprehensive evaluations over two image datasets with state-of-the-art poisoning attacks under different realistic attack scenarios. Our empirical results show that HINT can efficiently protect deep learning models against the effect of both untargeted and targeted poisoning attacks.

摘要: 虽然已经提出了许多防御方法来阻止来自不受信任的数据源的潜在中毒攻击，但大多数研究工作只防御特定的攻击，这给对手留下了许多可以利用的途径。在这项工作中，我们提出了一种基于影响函数的高效、健壮的数据中毒攻击训练方法，即基于健康影响噪声的训练方法。利用影响函数构造健康噪声，在不显著影响测试数据泛化能力的情况下，有助于加强分类模型对中毒攻击的抵抗能力。此外，我们的方法可以在只修改训练数据的子集的情况下有效地执行，而不是在以前的几个工作中使用的向所有样本添加噪声的方法。在不同的真实攻击场景下，我们对两个具有最新技术的中毒攻击的图像数据集进行了综合评估。我们的实验结果表明，提示可以有效地保护深度学习模型免受非定向和定向中毒攻击的影响。



## **16. Adaptive Data Analysis in a Balanced Adversarial Model**

均衡对抗性模型中的自适应数据分析 cs.LG

Accepted to NeurIPS 2023 (Spotlight)

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2305.15452v2) [paper-pdf](http://arxiv.org/pdf/2305.15452v2)

**Authors**: Kobbi Nissim, Uri Stemmer, Eliad Tsfadia

**Abstract**: In adaptive data analysis, a mechanism gets $n$ i.i.d. samples from an unknown distribution $D$, and is required to provide accurate estimations to a sequence of adaptively chosen statistical queries with respect to $D$. Hardt and Ullman (FOCS 2014) and Steinke and Ullman (COLT 2015) showed that in general, it is computationally hard to answer more than $\Theta(n^2)$ adaptive queries, assuming the existence of one-way functions.   However, these negative results strongly rely on an adversarial model that significantly advantages the adversarial analyst over the mechanism, as the analyst, who chooses the adaptive queries, also chooses the underlying distribution $D$. This imbalance raises questions with respect to the applicability of the obtained hardness results -- an analyst who has complete knowledge of the underlying distribution $D$ would have little need, if at all, to issue statistical queries to a mechanism which only holds a finite number of samples from $D$.   We consider more restricted adversaries, called \emph{balanced}, where each such adversary consists of two separated algorithms: The \emph{sampler} who is the entity that chooses the distribution and provides the samples to the mechanism, and the \emph{analyst} who chooses the adaptive queries, but has no prior knowledge of the underlying distribution (and hence has no a priori advantage with respect to the mechanism). We improve the quality of previous lower bounds by revisiting them using an efficient \emph{balanced} adversary, under standard public-key cryptography assumptions. We show that these stronger hardness assumptions are unavoidable in the sense that any computationally bounded \emph{balanced} adversary that has the structure of all known attacks, implies the existence of public-key cryptography.

摘要: 在适应性数据分析中，一个机制得到$n$I.I.D.来自未知分布$D$的样本，并且需要对关于$D$的适应性选择的统计查询序列提供准确的估计。Hardt和Ullman(FOCS 2014)和Steinke和Ullman(COLT 2015)表明，假设存在单向函数，通常很难回答超过$\theta(n^2)$自适应查询。然而，这些负面结果强烈依赖于对抗性模型，该模型显著地使对抗性分析师相对于该机制具有优势，因为选择自适应查询的分析师也选择基础分布$D$。这种不平衡对所获得的硬度结果的适用性提出了问题--完全了解基本分布$D$的分析员将几乎不需要向一个仅保存来自$D$的有限数量样本的机制发出统计查询。我们考虑了更受限制的对手，称为\emph{平衡}，其中每个这样的对手由两个独立的算法组成：\emph{Sampler}是选择分布并向机制提供样本的实体，以及\emph{Analyst}选择自适应查询，但对底层分布没有先验知识(因此在机制方面没有先验优势)。在标准的公钥密码学假设下，我们通过使用一个有效的、平衡的对手来重新访问以前的下界，从而提高了它们的质量。我们证明了这些更强的难度假设是不可避免的，因为任何具有所有已知攻击结构的计算有界的对手都意味着公钥密码学的存在。



## **17. The Alignment Problem in Context**

上下文中的对齐问题 cs.LG

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2311.02147v1) [paper-pdf](http://arxiv.org/pdf/2311.02147v1)

**Authors**: Raphaël Millière

**Abstract**: A core challenge in the development of increasingly capable AI systems is to make them safe and reliable by ensuring their behaviour is consistent with human values. This challenge, known as the alignment problem, does not merely apply to hypothetical future AI systems that may pose catastrophic risks; it already applies to current systems, such as large language models, whose potential for harm is rapidly increasing. In this paper, I assess whether we are on track to solve the alignment problem for large language models, and what that means for the safety of future AI systems. I argue that existing strategies for alignment are insufficient, because large language models remain vulnerable to adversarial attacks that can reliably elicit unsafe behaviour. I offer an explanation of this lingering vulnerability on which it is not simply a contingent limitation of current language models, but has deep technical ties to a crucial aspect of what makes these models useful and versatile in the first place -- namely, their remarkable aptitude to learn "in context" directly from user instructions. It follows that the alignment problem is not only unsolved for current AI systems, but may be intrinsically difficult to solve without severely undermining their capabilities. Furthermore, this assessment raises concerns about the prospect of ensuring the safety of future and more capable AI systems.

摘要: 开发能力越来越强的人工智能系统的一个核心挑战是，通过确保它们的行为符合人类价值观，使它们变得安全可靠。这一挑战被称为对齐问题，不仅适用于可能构成灾难性风险的假设的未来人工智能系统；它已经适用于当前的系统，例如大型语言模型，其危害的可能性正在迅速增加。在这篇文章中，我评估了我们是否正在解决大型语言模型的对齐问题，以及这对未来人工智能系统的安全意味着什么。我认为，现有的对齐策略是不够的，因为大型语言模型仍然容易受到敌意攻击，这些攻击可能会可靠地引发不安全的行为。我解释了这个挥之不去的漏洞，在这个漏洞上，它不仅仅是当前语言模型的偶然限制，而且与使这些模型首先有用和多功能的一个关键方面有很深的技术联系--即它们直接从用户指令中“在上下文中”学习的非凡能力。由此得出的结论是，对齐问题不仅对当前的人工智能系统没有解决，而且可能在不严重削弱其能力的情况下从本质上很难解决。此外，这项评估引发了人们对确保未来更有能力的人工智能系统安全的前景的担忧。



## **18. Bucks for Buckets (B4B): Active Defenses Against Stealing Encoders**

以桶换钱(B4B)：主动防御窃取编码器 cs.LG

Accepted at NeurIPS2023

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2310.08571v2) [paper-pdf](http://arxiv.org/pdf/2310.08571v2)

**Authors**: Jan Dubiński, Stanisław Pawlak, Franziska Boenisch, Tomasz Trzciński, Adam Dziedzic

**Abstract**: Machine Learning as a Service (MLaaS) APIs provide ready-to-use and high-utility encoders that generate vector representations for given inputs. Since these encoders are very costly to train, they become lucrative targets for model stealing attacks during which an adversary leverages query access to the API to replicate the encoder locally at a fraction of the original training costs. We propose Bucks for Buckets (B4B), the first active defense that prevents stealing while the attack is happening without degrading representation quality for legitimate API users. Our defense relies on the observation that the representations returned to adversaries who try to steal the encoder's functionality cover a significantly larger fraction of the embedding space than representations of legitimate users who utilize the encoder to solve a particular downstream task.vB4B leverages this to adaptively adjust the utility of the returned representations according to a user's coverage of the embedding space. To prevent adaptive adversaries from eluding our defense by simply creating multiple user accounts (sybils), B4B also individually transforms each user's representations. This prevents the adversary from directly aggregating representations over multiple accounts to create their stolen encoder copy. Our active defense opens a new path towards securely sharing and democratizing encoders over public APIs.

摘要: 机器学习即服务(MLaaS)API提供了现成的、高实用的编码器，可以为给定的输入生成向量表示。由于这些编码器的培训成本非常高，他们成为模型窃取攻击的有利可图的目标，在攻击期间，对手利用对API的查询访问来以原始培训成本的一小部分在本地复制编码器。我们提出了Bucks for Buckets(B4B)，这是第一种主动防御，可以在攻击发生时防止窃取，而不会降低合法API用户的表示质量。我们的辩护依赖于这样的观察，即返回给试图窃取编码器功能的对手的表示覆盖的嵌入空间比利用编码器解决特定下游任务的合法用户的表示大得多。vB4B利用这一点来根据用户对嵌入空间的覆盖自适应地调整返回的表示的效用。为了防止适应性对手通过简单地创建多个用户帐户(Sybils)来逃避我们的防御，B4B还单独转换每个用户的表示。这可以防止对手直接在多个帐户上聚合表示，以创建他们被盗的编码器副本。我们的积极防御为通过公共API安全共享和民主化编码器开辟了一条新的道路。



## **19. Efficient Black-Box Adversarial Attacks on Neural Text Detectors**

基于神经文本检测器的高效黑盒对抗攻击 cs.CL

Accepted at ICNLSP 2023

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2311.01873v1) [paper-pdf](http://arxiv.org/pdf/2311.01873v1)

**Authors**: Vitalii Fishchuk, Daniel Braun

**Abstract**: Neural text detectors are models trained to detect whether a given text was generated by a language model or written by a human. In this paper, we investigate three simple and resource-efficient strategies (parameter tweaking, prompt engineering, and character-level mutations) to alter texts generated by GPT-3.5 that are unsuspicious or unnoticeable for humans but cause misclassification by neural text detectors. The results show that especially parameter tweaking and character-level mutations are effective strategies.

摘要: 神经文本检测器是经过训练的模型，用于检测给定的文本是由语言模型生成的还是由人类编写的。在本文中，我们研究了三种简单且资源高效的策略(参数调整、即时工程和字符级突变)来更改由GPT-3.5生成的文本，这些文本对人类来说是不可疑的或不可察觉的，但会导致神经文本检测器的错误分类。结果表明，特别是参数调整和字符级突变是有效的策略。



## **20. Adversarial Attacks against Binary Similarity Systems**

二进制相似系统的对抗性攻击 cs.CR

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2303.11143v2) [paper-pdf](http://arxiv.org/pdf/2303.11143v2)

**Authors**: Gianluca Capozzi, Daniele Cono D'Elia, Giuseppe Antonio Di Luna, Leonardo Querzoni

**Abstract**: In recent years, binary analysis gained traction as a fundamental approach to inspect software and guarantee its security. Due to the exponential increase of devices running software, much research is now moving towards new autonomous solutions based on deep learning models, as they have been showing state-of-the-art performances in solving binary analysis problems. One of the hot topics in this context is binary similarity, which consists in determining if two functions in assembly code are compiled from the same source code. However, it is unclear how deep learning models for binary similarity behave in an adversarial context. In this paper, we study the resilience of binary similarity models against adversarial examples, showing that they are susceptible to both targeted and untargeted attacks (w.r.t. similarity goals) performed by black-box and white-box attackers. In more detail, we extensively test three current state-of-the-art solutions for binary similarity against two black-box greedy attacks, including a new technique that we call Spatial Greedy, and one white-box attack in which we repurpose a gradient-guided strategy used in attacks to image classifiers.

摘要: 近年来，二进制分析作为检查软件和保证其安全性的基本方法得到了越来越多的重视。由于运行软件的设备呈指数级增长，许多研究现在正在转向基于深度学习模型的新的自主解决方案，因为它们在解决二进制分析问题方面表现出了最先进的性能。这方面的一个热门话题是二进制相似性，即确定汇编代码中的两个函数是否从相同的源代码编译而来。然而，目前还不清楚二元相似性的深度学习模型在对抗性环境中的表现如何。在本文中，我们研究了二进制相似模型对敌意例子的弹性，表明它们对目标攻击和非目标攻击都敏感(w.r.t.相似性目标)由黑盒和白盒攻击者执行。更详细地，我们针对两种黑盒贪婪攻击广泛地测试了三种当前最先进的二进制相似性解决方案，其中包括一种称为空间贪婪的新技术，以及一种白盒攻击，其中我们将攻击中使用的梯度引导策略重新用于图像分类器。



## **21. Adversarial Attacks on Cooperative Multi-agent Bandits**

合作多智能体盗贼的对抗性攻击 cs.LG

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2311.01698v1) [paper-pdf](http://arxiv.org/pdf/2311.01698v1)

**Authors**: Jinhang Zuo, Zhiyao Zhang, Xuchuang Wang, Cheng Chen, Shuai Li, John C. S. Lui, Mohammad Hajiesmaili, Adam Wierman

**Abstract**: Cooperative multi-agent multi-armed bandits (CMA2B) consider the collaborative efforts of multiple agents in a shared multi-armed bandit game. We study latent vulnerabilities exposed by this collaboration and consider adversarial attacks on a few agents with the goal of influencing the decisions of the rest. More specifically, we study adversarial attacks on CMA2B in both homogeneous settings, where agents operate with the same arm set, and heterogeneous settings, where agents have distinct arm sets. In the homogeneous setting, we propose attack strategies that, by targeting just one agent, convince all agents to select a particular target arm $T-o(T)$ times while incurring $o(T)$ attack costs in $T$ rounds. In the heterogeneous setting, we prove that a target arm attack requires linear attack costs and propose attack strategies that can force a maximum number of agents to suffer linear regrets while incurring sublinear costs and only manipulating the observations of a few target agents. Numerical experiments validate the effectiveness of our proposed attack strategies.

摘要: 合作多智能体多武装土匪(CMA2B)考虑多个智能体在共享的多臂土匪博弈中的协作努力。我们研究了这种合作暴露的潜在漏洞，并考虑了对几个代理的对抗性攻击，目的是影响其余代理的决策。更具体地说，我们研究了在同构设置和异质设置下对CMA2B的对抗性攻击，在同构设置中，代理使用相同的ARM集操作，而在异质设置中，代理具有不同的ARM集。在同类环境下，我们提出了攻击策略，通过只针对一个代理，说服所有代理选择特定的目标臂$T-o(T)$次，同时在$T$轮中产生$o(T)$攻击成本。在异质环境下，我们证明了一次目标ARM攻击需要线性攻击代价，并提出了一种攻击策略，该策略可以迫使最大数量的代理遭受线性后悔，同时产生次线性代价，并且只操纵少数目标代理的观测。数值实验验证了本文提出的攻击策略的有效性。



## **22. Universal Perturbation-based Secret Key-Controlled Data Hiding**

基于普遍扰动的密钥控制数据隐藏 cs.CR

18 pages, 8 tables, 10 figures

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2311.01696v1) [paper-pdf](http://arxiv.org/pdf/2311.01696v1)

**Authors**: Donghua Wang, Wen Yao, Tingsong Jiang, Xiaoqian Chen

**Abstract**: Deep neural networks (DNNs) are demonstrated to be vulnerable to universal perturbation, a single quasi-perceptible perturbation that can deceive the DNN on most images. However, the previous works are focused on using universal perturbation to perform adversarial attacks, while the potential usability of universal perturbation as data carriers in data hiding is less explored, especially for the key-controlled data hiding method. In this paper, we propose a novel universal perturbation-based secret key-controlled data-hiding method, realizing data hiding with a single universal perturbation and data decoding with the secret key-controlled decoder. Specifically, we optimize a single universal perturbation, which serves as a data carrier that can hide multiple secret images and be added to most cover images. Then, we devise a secret key-controlled decoder to extract different secret images from the single container image constructed by the universal perturbation by using different secret keys. Moreover, a suppress loss function is proposed to prevent the secret image from leakage. Furthermore, we adopt a robust module to boost the decoder's capability against corruption. Finally, A co-joint optimization strategy is proposed to find the optimal universal perturbation and decoder. Extensive experiments are conducted on different datasets to demonstrate the effectiveness of the proposed method. Additionally, the physical test performed on platforms (e.g., WeChat and Twitter) verifies the usability of the proposed method in practice.

摘要: 深度神经网络(DNN)被证明容易受到普遍扰动的影响，这是一种单一的准可感知的扰动，可以在大多数图像上欺骗DNN。然而，前人的工作主要集中在利用普遍扰动进行对抗性攻击，而对于普遍扰动作为数据载体在数据隐藏中的潜在可用性的探讨较少，尤其是对于密钥控制的数据隐藏方法。本文提出了一种新的基于通用扰动的密钥控制数据隐藏方法，用单一的通用扰动实现数据隐藏，用密钥控制解码器实现数据解码。具体地说，我们优化了单个普遍扰动，它作为数据载体可以隐藏多个秘密图像，并被添加到大多数封面图像中。然后，设计了一个密钥控制的解码器，通过使用不同的密钥从由普适扰动构造的单个集装箱图像中提取不同的秘密图像。此外，为了防止秘密图像的泄漏，提出了一种抑制损失函数。此外，我们采用了一个健壮的模块来提高解码器的抗损坏能力。最后，提出了一种联合优化策略来寻找最优的普适扰动和解码器。在不同的数据集上进行了大量的实验，验证了该方法的有效性。此外，在平台(如微信和推特)上进行的物理测试验证了该方法在实践中的可用性。



## **23. Robust Adversarial Reinforcement Learning via Bounded Rationality Curricula**

基于有限理性课程的稳健对抗性强化学习 cs.LG

Under review

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2311.01642v1) [paper-pdf](http://arxiv.org/pdf/2311.01642v1)

**Authors**: Aryaman Reddi, Maximilian Tölle, Jan Peters, Georgia Chalvatzaki, Carlo D'Eramo

**Abstract**: Robustness against adversarial attacks and distribution shifts is a long-standing goal of Reinforcement Learning (RL). To this end, Robust Adversarial Reinforcement Learning (RARL) trains a protagonist against destabilizing forces exercised by an adversary in a competitive zero-sum Markov game, whose optimal solution, i.e., rational strategy, corresponds to a Nash equilibrium. However, finding Nash equilibria requires facing complex saddle point optimization problems, which can be prohibitive to solve, especially for high-dimensional control. In this paper, we propose a novel approach for adversarial RL based on entropy regularization to ease the complexity of the saddle point optimization problem. We show that the solution of this entropy-regularized problem corresponds to a Quantal Response Equilibrium (QRE), a generalization of Nash equilibria that accounts for bounded rationality, i.e., agents sometimes play random actions instead of optimal ones. Crucially, the connection between the entropy-regularized objective and QRE enables free modulation of the rationality of the agents by simply tuning the temperature coefficient. We leverage this insight to propose our novel algorithm, Quantal Adversarial RL (QARL), which gradually increases the rationality of the adversary in a curriculum fashion until it is fully rational, easing the complexity of the optimization problem while retaining robustness. We provide extensive evidence of QARL outperforming RARL and recent baselines across several MuJoCo locomotion and navigation problems in overall performance and robustness.

摘要: 增强学习(RL)对敌意攻击和分布变化的稳健性是一个长期的目标。为此，稳健对抗性强化学习(RARL)训练主角对抗竞争零和马尔可夫博弈中对手施加的破坏稳定的力量，其最优解，即理性策略，对应于纳什均衡。然而，寻找纳什均衡需要面对复杂的鞍点优化问题，这可能是令人望而却步的问题，特别是对于高维控制。为了降低鞍点优化问题的复杂性，本文提出了一种基于熵正则化的对抗性RL算法。我们证明了这个熵正则化问题的解对应于量子响应均衡(QRE)，这是纳什均衡的推广，它解释了有限理性，即代理人有时扮演随机行为而不是最优行为。最重要的是，通过简单地调节温度系数，熵正则化目标和QRE之间的联系使得能够自由地调节作用剂的合理性。我们利用这一见解提出了我们的新算法Quantal Adversial RL(QARL)，它以课程的方式逐渐增加对手的理性，直到它完全理性，在保持健壮性的同时缓解了优化问题的复杂性。我们提供了广泛的证据表明，QARL在总体性能和稳健性方面超过了RARL和最近几个MuJoCo移动和导航问题的基线。



## **24. Assist Is Just as Important as the Goal: Image Resurfacing to Aid Model's Robust Prediction**

辅助与目标同等重要：图像重现辅助模型的稳健预测 cs.CV

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2311.01563v1) [paper-pdf](http://arxiv.org/pdf/2311.01563v1)

**Authors**: Abhijith Sharma, Phil Munz, Apurva Narayan

**Abstract**: Adversarial patches threaten visual AI models in the real world. The number of patches in a patch attack is variable and determines the attack's potency in a specific environment. Most existing defenses assume a single patch in the scene, and the multiple patch scenarios are shown to overcome them. This paper presents a model-agnostic defense against patch attacks based on total variation for image resurfacing (TVR). The TVR is an image-cleansing method that processes images to remove probable adversarial regions. TVR can be utilized solely or augmented with a defended model, providing multi-level security for robust prediction. TVR nullifies the influence of patches in a single image scan with no prior assumption on the number of patches in the scene. We validate TVR on the ImageNet-Patch benchmark dataset and with real-world physical objects, demonstrating its ability to mitigate patch attack.

摘要: 对抗性补丁威胁到现实世界中的可视AI模型。补丁攻击中的补丁数量是可变的，并决定了攻击在特定环境中的效力。大多数现有的防御假设场景中只有一个补丁，而多个补丁场景显示可以克服它们。提出了一种基于全变分的抗补丁攻击的模型不可知防御方法(TVR)。TVR是一种图像净化方法，它处理图像以删除可能的敌对区域。TVR可以单独使用，也可以与防御模型一起使用，为稳健预测提供多级安全。TVR消除了单个图像扫描中的补丁的影响，而不需要预先假设场景中的补丁的数量。我们在ImageNet-Patch基准数据集上和真实世界的物理对象上验证了TVR，展示了其缓解补丁攻击的能力。



## **25. E(2) Equivariant Neural Networks for Robust Galaxy Morphology Classification**

基于E(2)等变神经网络的稳健星系形态分类 astro-ph.GA

10 pages, 4 figures, 3 tables, Accepted to the Machine Learning and  the Physical Sciences Workshop at NeurIPS 2023

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2311.01500v1) [paper-pdf](http://arxiv.org/pdf/2311.01500v1)

**Authors**: Sneh Pandya, Purvik Patel, Franc O, Jonathan Blazek

**Abstract**: We propose the use of group convolutional neural network architectures (GCNNs) equivariant to the 2D Euclidean group, $E(2)$, for the task of galaxy morphology classification by utilizing symmetries of the data present in galaxy images as an inductive bias in the architecture. We conduct robustness studies by introducing artificial perturbations via Poisson noise insertion and one-pixel adversarial attacks to simulate the effects of limited observational capabilities. We train, validate, and test GCNNs equivariant to discrete subgroups of $E(2)$ - the cyclic and dihedral groups of order $N$ - on the Galaxy10 DECals dataset and find that GCNNs achieve higher classification accuracy and are consistently more robust than their non-equivariant counterparts, with an architecture equivariant to the group $D_{16}$ achieving a $95.52 \pm 0.18\%$ test-set accuracy. We also find that the model loses $<6\%$ accuracy on a $50\%$-noise dataset and all GCNNs are less susceptible to one-pixel perturbations than an identically constructed CNN. Our code is publicly available at https://github.com/snehjp2/GCNNMorphology.

摘要: 我们建议使用组卷积神经网络架构（GCNN）等变的二维欧几里德组，$E（2）$，星系形态分类的任务，利用星系图像中的数据的对称性作为架构中的归纳偏差。我们通过泊松噪声插入和单像素对抗攻击引入人工扰动来模拟有限观测能力的影响，从而进行鲁棒性研究。我们在Galaxy 10 DECals数据集上训练，验证和测试GCNN等变到$E（2）$的离散子群-阶数为$N$的循环和二面体群，并发现GCNN实现了更高的分类精度，并且始终比它们的非等变对应物更鲁棒，与组$D_{16}$的结构等变实现了$95.52 \pm 0.18\%$的测试集精度。我们还发现，该模型在$50\%$噪声数据集上损失了$<6\%$的准确性，并且所有GCNN都比相同构造的CNN更不易受单像素扰动的影响。我们的代码可在https://github.com/snehjp2/GCNNMorphology上公开获取。



## **26. Like an Open Book? Read Neural Network Architecture with Simple Power Analysis on 32-bit Microcontrollers**

就像一本打开的书？在32位微控制器上实现简单功耗分析的神经网络结构 cs.CR

Accepted CARDIS 2023; ANR PICTURE PROJECT (ANR-20-CE39-0013)

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2311.01344v1) [paper-pdf](http://arxiv.org/pdf/2311.01344v1)

**Authors**: Raphael Joud, Pierre-Alain Moellic, Simon Pontie, Jean-Baptiste Rigaud

**Abstract**: Model extraction is a growing concern for the security of AI systems. For deep neural network models, the architecture is the most important information an adversary aims to recover. Being a sequence of repeated computation blocks, neural network models deployed on edge-devices will generate distinctive side-channel leakages. The latter can be exploited to extract critical information when targeted platforms are physically accessible. By combining theoretical knowledge about deep learning practices and analysis of a widespread implementation library (ARM CMSIS-NN), our purpose is to answer this critical question: how far can we extract architecture information by simply examining an EM side-channel trace? For the first time, we propose an extraction methodology for traditional MLP and CNN models running on a high-end 32-bit microcontroller (Cortex-M7) that relies only on simple pattern recognition analysis. Despite few challenging cases, we claim that, contrary to parameters extraction, the complexity of the attack is relatively low and we highlight the urgent need for practicable protections that could fit the strong memory and latency requirements of such platforms.

摘要: 模型提取是人工智能系统安全的一个日益关注的问题。对于深度神经网络模型，体系结构是对手要恢复的最重要的信息。作为一系列重复的计算模块，部署在边缘设备上的神经网络模型将产生独特的侧通道泄漏。当目标平台可物理访问时，可利用后者来提取关键信息。通过结合深度学习实践的理论知识和对广泛使用的实施库(ARM CMSIS-NN)的分析，我们的目的是回答这个关键问题：通过简单地检查EM侧通道跟踪，我们可以在多大程度上提取体系结构信息？首次针对运行在高端32位微控制器(Cortex-M7)上的传统MLP和CNN模型提出了一种仅依赖于简单模式识别分析的提取方法。尽管几乎没有具有挑战性的案例，但我们声称，与参数提取相反，攻击的复杂性相对较低，我们强调迫切需要可行的保护措施，以满足此类平台的强大内存和延迟要求。



## **27. Towards Evaluating Transfer-based Attacks Systematically, Practically, and Fairly**

对基于传输的攻击进行系统、实用和公平的评估 cs.LG

Accepted by NeurIPS 2023

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2311.01323v1) [paper-pdf](http://arxiv.org/pdf/2311.01323v1)

**Authors**: Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

**Abstract**: The adversarial vulnerability of deep neural networks (DNNs) has drawn great attention due to the security risk of applying these models in real-world applications. Based on transferability of adversarial examples, an increasing number of transfer-based methods have been developed to fool black-box DNN models whose architecture and parameters are inaccessible. Although tremendous effort has been exerted, there still lacks a standardized benchmark that could be taken advantage of to compare these methods systematically, fairly, and practically. Our investigation shows that the evaluation of some methods needs to be more reasonable and more thorough to verify their effectiveness, to avoid, for example, unfair comparison and insufficient consideration of possible substitute/victim models. Therefore, we establish a transfer-based attack benchmark (TA-Bench) which implements 30+ methods. In this paper, we evaluate and compare them comprehensively on 25 popular substitute/victim models on ImageNet. New insights about the effectiveness of these methods are gained and guidelines for future evaluations are provided. Code at: https://github.com/qizhangli/TA-Bench.

摘要: 由于深度神经网络(DNN)模型在实际应用中存在的安全风险，这些模型的敌意脆弱性引起了人们的极大关注。基于对抗性例子的可转移性，越来越多的基于转移的方法被开发来愚弄结构和参数不可访问的黑盒DNN模型。尽管已经付出了巨大的努力，但仍然缺乏一个可以利用的标准化基准来系统、公平和实际地比较这些方法。我们的调查表明，对某些方法的评价需要更加合理和彻底，以验证其有效性，避免不公平的比较和对可能的替代/受害者模型考虑不足。因此，我们建立了一个基于传输的攻击基准(TA-BENCH)，它实现了30多种方法。在本文中，我们对ImageNet上流行的25种替代/受害者模型进行了全面的评估和比较。对这些方法的有效性获得了新的见解，并为未来的评估提供了指导方针。代码：https://github.com/qizhangli/TA-Bench.



## **28. Improving Adversarial Transferability via Intermediate-level Perturbation Decay**

通过中层扰动衰减提高对手的可转换性 cs.LG

Accepted by NeurIPS 2023

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2304.13410v3) [paper-pdf](http://arxiv.org/pdf/2304.13410v3)

**Authors**: Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

**Abstract**: Intermediate-level attacks that attempt to perturb feature representations following an adversarial direction drastically have shown favorable performance in crafting transferable adversarial examples. Existing methods in this category are normally formulated with two separate stages, where a directional guide is required to be determined at first and the scalar projection of the intermediate-level perturbation onto the directional guide is enlarged thereafter. The obtained perturbation deviates from the guide inevitably in the feature space, and it is revealed in this paper that such a deviation may lead to sub-optimal attack. To address this issue, we develop a novel intermediate-level method that crafts adversarial examples within a single stage of optimization. In particular, the proposed method, named intermediate-level perturbation decay (ILPD), encourages the intermediate-level perturbation to be in an effective adversarial direction and to possess a great magnitude simultaneously. In-depth discussion verifies the effectiveness of our method. Experimental results show that it outperforms state-of-the-arts by large margins in attacking various victim models on ImageNet (+10.07% on average) and CIFAR-10 (+3.88% on average). Our code is at https://github.com/qizhangli/ILPD-attack.

摘要: 中级攻击试图按照对抗性方向彻底扰乱特征表示，在制作可转移的对抗性示例方面表现出了良好的性能。现有的这类方法通常分为两个不同的阶段，首先需要确定一个方向导轨，然后放大中层摄动在该方向导轨上的标量投影。所得到的扰动在特征空间中不可避免地偏离了导引，本文揭示了这种偏离可能导致次优攻击。为了解决这个问题，我们开发了一种新的中级方法，该方法在单个优化阶段内创建对抗性示例。特别是，所提出的方法，称为中层扰动衰变(ILPD)，它鼓励中层扰动朝着有效的对抗性方向发展，同时具有较大的幅度。通过深入讨论，验证了该方法的有效性。实验结果表明，在ImageNet(平均+10.07%)和CIFAR-10(平均+3.88%)上攻击各种受害者模型时，该算法的性能明显优于最新的攻击模型。我们的代码在https://github.com/qizhangli/ILPD-attack.



## **29. Understanding and Improving Ensemble Adversarial Defense**

认识和完善整体对抗防御 cs.LG

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2310.18477v2) [paper-pdf](http://arxiv.org/pdf/2310.18477v2)

**Authors**: Yian Deng, Tingting Mu

**Abstract**: The strategy of ensemble has become popular in adversarial defense, which trains multiple base classifiers to defend against adversarial attacks in a cooperative manner. Despite the empirical success, theoretical explanations on why an ensemble of adversarially trained classifiers is more robust than single ones remain unclear. To fill in this gap, we develop a new error theory dedicated to understanding ensemble adversarial defense, demonstrating a provable 0-1 loss reduction on challenging sample sets in an adversarial defense scenario. Guided by this theory, we propose an effective approach to improve ensemble adversarial defense, named interactive global adversarial training (iGAT). The proposal includes (1) a probabilistic distributing rule that selectively allocates to different base classifiers adversarial examples that are globally challenging to the ensemble, and (2) a regularization term to rescue the severest weaknesses of the base classifiers. Being tested over various existing ensemble adversarial defense techniques, iGAT is capable of boosting their performance by increases up to 17% evaluated using CIFAR10 and CIFAR100 datasets under both white-box and black-box attacks.

摘要: 在对抗性防御中，集成策略已经成为一种流行的策略，它训练多个基分类器以协作的方式防御对抗性攻击。尽管取得了经验上的成功，但关于为什么对抗性训练的分类器集合比单个分类器更稳健的理论解释仍然不清楚。为了填补这一空白，我们发展了一种新的错误理论，致力于理解集成对抗性防御，展示了在对抗性防御场景中挑战样本集上可证明的0-1损失减少。在此理论指导下，我们提出了一种提高集成对抗能力的有效方法，即交互式全局对抗训练(IGAT)。该方案包括(1)概率分布规则，它选择性地将对集成具有全局挑战性的对抗性实例分配给不同的基分类器；(2)正则化项以弥补基分类器最严重的弱点。在对各种现有的集成对抗防御技术进行测试后，iGAT能够将其性能提高17%，在白盒和黑盒攻击下使用CIFAR10和CIFAR100数据集进行评估。



## **30. Detection Defenses: An Empty Promise against Adversarial Patch Attacks on Optical Flow**

检测防御：对抗光流对抗性补丁攻击的空头承诺 cs.CV

Accepted to WACV 2024

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2310.17403v2) [paper-pdf](http://arxiv.org/pdf/2310.17403v2)

**Authors**: Erik Scheurer, Jenny Schmalfuss, Alexander Lis, Andrés Bruhn

**Abstract**: Adversarial patches undermine the reliability of optical flow predictions when placed in arbitrary scene locations. Therefore, they pose a realistic threat to real-world motion detection and its downstream applications. Potential remedies are defense strategies that detect and remove adversarial patches, but their influence on the underlying motion prediction has not been investigated. In this paper, we thoroughly examine the currently available detect-and-remove defenses ILP and LGS for a wide selection of state-of-the-art optical flow methods, and illuminate their side effects on the quality and robustness of the final flow predictions. In particular, we implement defense-aware attacks to investigate whether current defenses are able to withstand attacks that take the defense mechanism into account. Our experiments yield two surprising results: Detect-and-remove defenses do not only lower the optical flow quality on benign scenes, in doing so, they also harm the robustness under patch attacks for all tested optical flow methods except FlowNetC. As currently employed detect-and-remove defenses fail to deliver the promised adversarial robustness for optical flow, they evoke a false sense of security. The code is available at https://github.com/cv-stuttgart/DetectionDefenses.

摘要: 当放置在任意场景位置时，对抗性补丁破坏了光流预测的可靠性。因此，它们对真实世界的运动检测及其下游应用构成了现实的威胁。潜在的补救措施是检测和移除对抗性补丁的防御策略，但它们对潜在运动预测的影响尚未被调查。在这篇文章中，我们彻底审查了目前可用的检测和删除防御ILP和LGS的各种最先进的光流方法选择，并说明了它们对最终流动预测的质量和稳健性的副作用。特别是，我们实施防御感知攻击，以调查当前的防御是否能够抵御考虑到防御机制的攻击。我们的实验产生了两个令人惊讶的结果：检测和删除防御不仅降低了良性场景的光流质量，而且还损害了除FlowNetC之外的所有测试的光流方法在补丁攻击下的健壮性。由于目前使用的检测和删除防御系统无法为光流提供承诺的对手健壮性，它们会引起一种错误的安全感。代码可在https://github.com/cv-stuttgart/DetectionDefenses.上获得



## **31. Boosting Adversarial Transferability by Achieving Flat Local Maxima**

通过实现平坦的局部最大值来提高对手的可转移性 cs.CV

Accepted by the Neural Information Processing Systems (NeurIPS 2023)

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2306.05225v2) [paper-pdf](http://arxiv.org/pdf/2306.05225v2)

**Authors**: Zhijin Ge, Hongying Liu, Xiaosen Wang, Fanhua Shang, Yuanyuan Liu

**Abstract**: Transfer-based attack adopts the adversarial examples generated on the surrogate model to attack various models, making it applicable in the physical world and attracting increasing interest. Recently, various adversarial attacks have emerged to boost adversarial transferability from different perspectives. In this work, inspired by the observation that flat local minima are correlated with good generalization, we assume and empirically validate that adversarial examples at a flat local region tend to have good transferability by introducing a penalized gradient norm to the original loss function. Since directly optimizing the gradient regularization norm is computationally expensive and intractable for generating adversarial examples, we propose an approximation optimization method to simplify the gradient update of the objective function. Specifically, we randomly sample an example and adopt a first-order procedure to approximate the curvature of Hessian/vector product, which makes computing more efficient by interpolating two neighboring gradients. Meanwhile, in order to obtain a more stable gradient direction, we randomly sample multiple examples and average the gradients of these examples to reduce the variance due to random sampling during the iterative process. Extensive experimental results on the ImageNet-compatible dataset show that the proposed method can generate adversarial examples at flat local regions, and significantly improve the adversarial transferability on either normally trained models or adversarially trained models than the state-of-the-art attacks. Our codes are available at: https://github.com/Trustworthy-AI-Group/PGN.

摘要: 基于转移的攻击采用代理模型上生成的对抗性实例来攻击各种模型，使其适用于物理世界，引起了人们越来越多的兴趣。近年来，各种对抗性攻击层出不穷，从不同的角度提升了对抗性的可转移性。在这项工作中，受平坦局部极小值与良好泛化相关的观察结果的启发，我们假设并经验验证了平坦局部区域上的对抗性例子通过在原始损失函数中引入惩罚梯度范数而倾向于具有良好的可转移性。由于直接优化梯度正则化范数的计算量大且难以生成对抗性样本，我们提出了一种近似优化方法来简化目标函数的梯度更新。具体地说，我们随机抽样一个例子，并采用一阶过程来逼近海森/向量积的曲率，通过对相邻的两个梯度进行内插，使得计算效率更高。同时，为了得到一个更稳定的梯度方向，我们对多个样本进行随机采样，并对这些样本的梯度进行平均，以减少迭代过程中随机采样造成的方差。在ImageNet兼容的数据集上的大量实验结果表明，该方法可以在平坦的局部区域生成对抗性实例，并且无论是在正常训练的模型上还是在对抗性训练的模型上，该方法都比最新的攻击方法显著提高了对抗性可转移性。我们的代码请访问：https://github.com/Trustworthy-AI-Group/PGN.



## **32. Tensor Trust: Interpretable Prompt Injection Attacks from an Online Game**

张量信任：来自网络游戏的可解释提示注入攻击 cs.LG

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2311.01011v1) [paper-pdf](http://arxiv.org/pdf/2311.01011v1)

**Authors**: Sam Toyer, Olivia Watkins, Ethan Adrian Mendes, Justin Svegliato, Luke Bailey, Tiffany Wang, Isaac Ong, Karim Elmaaroufi, Pieter Abbeel, Trevor Darrell, Alan Ritter, Stuart Russell

**Abstract**: While Large Language Models (LLMs) are increasingly being used in real-world applications, they remain vulnerable to prompt injection attacks: malicious third party prompts that subvert the intent of the system designer. To help researchers study this problem, we present a dataset of over 126,000 prompt injection attacks and 46,000 prompt-based "defenses" against prompt injection, all created by players of an online game called Tensor Trust. To the best of our knowledge, this is currently the largest dataset of human-generated adversarial examples for instruction-following LLMs. The attacks in our dataset have a lot of easily interpretable stucture, and shed light on the weaknesses of LLMs. We also use the dataset to create a benchmark for resistance to two types of prompt injection, which we refer to as prompt extraction and prompt hijacking. Our benchmark results show that many models are vulnerable to the attack strategies in the Tensor Trust dataset. Furthermore, we show that some attack strategies from the dataset generalize to deployed LLM-based applications, even though they have a very different set of constraints to the game. We release all data and source code at https://tensortrust.ai/paper

摘要: 虽然大型语言模型(LLM)越来越多地用于现实世界的应用程序，但它们仍然容易受到提示注入攻击：破坏系统设计人员意图的恶意第三方提示。为了帮助研究人员研究这个问题，我们提供了一个超过12.6万个即时注入攻击和4.6万个基于即时注入的“防御”的数据集，所有这些都是由一款名为“张量信任”的在线游戏的玩家创建的。据我们所知，这是目前最大的人类生成的遵循指令的LLM对抗性例子的数据集。我们的数据集中的攻击具有许多易于解释的结构，并揭示了LLMS的弱点。我们还使用数据集创建了对两种类型的即时注入的抵抗力基准，我们称之为即时提取和即时劫持。我们的基准测试结果表明，许多模型都容易受到张量信任数据集中的攻击策略的影响。此外，我们还表明，来自数据集的一些攻击策略适用于部署的基于LLM的应用程序，即使它们对游戏有非常不同的约束集。我们在https://tensortrust.ai/paper上发布所有数据和源代码



## **33. Private Graph Extraction via Feature Explanations**

基于特征解释的专用图提取 cs.LG

Accepted at PETS 2023

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2206.14724v2) [paper-pdf](http://arxiv.org/pdf/2206.14724v2)

**Authors**: Iyiola E. Olatunji, Mandeep Rathee, Thorben Funke, Megha Khosla

**Abstract**: Privacy and interpretability are two important ingredients for achieving trustworthy machine learning. We study the interplay of these two aspects in graph machine learning through graph reconstruction attacks. The goal of the adversary here is to reconstruct the graph structure of the training data given access to model explanations. Based on the different kinds of auxiliary information available to the adversary, we propose several graph reconstruction attacks. We show that additional knowledge of post-hoc feature explanations substantially increases the success rate of these attacks. Further, we investigate in detail the differences between attack performance with respect to three different classes of explanation methods for graph neural networks: gradient-based, perturbation-based, and surrogate model-based methods. While gradient-based explanations reveal the most in terms of the graph structure, we find that these explanations do not always score high in utility. For the other two classes of explanations, privacy leakage increases with an increase in explanation utility. Finally, we propose a defense based on a randomized response mechanism for releasing the explanations, which substantially reduces the attack success rate. Our code is available at https://github.com/iyempissy/graph-stealing-attacks-with-explanation

摘要: 隐私和可解释性是实现可信机器学习的两个重要因素。我们通过图重构攻击来研究这两个方面在图机器学习中的相互作用。在这里，对手的目标是在获得模型解释的情况下重建训练数据的图形结构。基于敌手可获得的各种辅助信息，我们提出了几种图重构攻击。我们表明，额外的事后特征解释知识大大提高了这些攻击的成功率。此外，我们还详细研究了图神经网络的三种不同解释方法：基于梯度、基于扰动和基于代理模型的解释方法在攻击性能上的差异。虽然基于梯度的解释在图表结构方面揭示了最多，但我们发现这些解释并不总是在实用方面得分很高。对于其他两类解释，隐私泄露随着解释效用的增加而增加。最后，我们提出了一种基于随机化响应机制的防御机制来发布解释，大大降低了攻击成功率。我们的代码可以在https://github.com/iyempissy/graph-stealing-attacks-with-explanation上找到



## **34. Adversary ML Resilience in Autonomous Driving Through Human Centered Perception Mechanisms**

以人为中心的感知机制在自主驾驶中的对手ML韧性 cs.CV

15 pages, 17 figures

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2311.01478v1) [paper-pdf](http://arxiv.org/pdf/2311.01478v1)

**Authors**: Aakriti Shah

**Abstract**: Physical adversarial attacks on road signs are continuously exploiting vulnerabilities in modern day autonomous vehicles (AVs) and impeding their ability to correctly classify what type of road sign they encounter. Current models cannot generalize input data well, resulting in overfitting or underfitting. In overfitting, the model memorizes the input data but cannot generalize to new scenarios. In underfitting, the model does not learn enough of the input data to accurately classify these road signs. This paper explores the resilience of autonomous driving systems against three main physical adversarial attacks (tape, graffiti, illumination), specifically targeting object classifiers. Several machine learning models were developed and evaluated on two distinct datasets: road signs (stop signs, speed limit signs, traffic lights, and pedestrian crosswalk signs) and geometric shapes (octagons, circles, squares, and triangles). The study compared algorithm performance under different conditions, including clean and adversarial training and testing on these datasets. To build robustness against attacks, defense techniques like adversarial training and transfer learning were implemented. Results demonstrated transfer learning models played a crucial role in performance by allowing knowledge gained from shape training to improve generalizability of road sign classification, despite the datasets being completely different. The paper suggests future research directions, including human-in-the-loop validation, security analysis, real-world testing, and explainable AI for transparency. This study aims to contribute to improving security and robustness of object classifiers in autonomous vehicles and mitigating adversarial example impacts on driving systems.

摘要: 针对路标的物理对抗性攻击不断地利用现代自动驾驶车辆(AV)的漏洞，阻碍它们正确地分类它们遇到的路标类型。目前的模型不能很好地概括输入数据，导致过拟合或欠拟合。在过度拟合中，模型记住了输入数据，但不能概括为新的情景。在欠拟合的情况下，模型没有学习足够的输入数据来准确地对这些路标进行分类。本文探讨了自动驾驶系统对三种主要的物理攻击(磁带、涂鸦、照明)的弹性，特别是针对对象分类器的攻击。开发了几个机器学习模型，并在两个不同的数据集上进行了评估：道路标志(停车标志、限速标志、红绿灯和人行横道标志)和几何形状(八角形、圆形、正方形和三角形)。这项研究比较了算法在不同条件下的性能，包括在这些数据集上进行干净和对抗性的训练和测试。为了建立对攻击的健壮性，采用了对抗性训练和迁移学习等防御技术。结果表明，尽管数据集完全不同，迁移学习模型通过允许从形状训练中获得的知识来提高道路标志分类的泛化能力，从而对性能起到关键作用。论文提出了未来的研究方向，包括人在回路中的验证、安全分析、真实世界测试和透明的可解释人工智能。本研究旨在提高自动车辆目标分类器的安全性和健壮性，减少对驾驶系统的不利影响。



## **35. MIST: Defending Against Membership Inference Attacks Through Membership-Invariant Subspace Training**

MIST：通过成员关系不变子空间训练防御成员关系推理攻击 cs.CR

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2311.00919v1) [paper-pdf](http://arxiv.org/pdf/2311.00919v1)

**Authors**: Jiacheng Li, Ninghui Li, Bruno Ribeiro

**Abstract**: In Member Inference (MI) attacks, the adversary try to determine whether an instance is used to train a machine learning (ML) model. MI attacks are a major privacy concern when using private data to train ML models. Most MI attacks in the literature take advantage of the fact that ML models are trained to fit the training data well, and thus have very low loss on training instances. Most defenses against MI attacks therefore try to make the model fit the training data less well. Doing so, however, generally results in lower accuracy. We observe that training instances have different degrees of vulnerability to MI attacks. Most instances will have low loss even when not included in training. For these instances, the model can fit them well without concerns of MI attacks. An effective defense only needs to (possibly implicitly) identify instances that are vulnerable to MI attacks and avoids overfitting them. A major challenge is how to achieve such an effect in an efficient training process. Leveraging two distinct recent advancements in representation learning: counterfactually-invariant representations and subspace learning methods, we introduce a novel Membership-Invariant Subspace Training (MIST) method to defend against MI attacks. MIST avoids overfitting the vulnerable instances without significant impact on other instances. We have conducted extensive experimental studies, comparing MIST with various other state-of-the-art (SOTA) MI defenses against several SOTA MI attacks. We find that MIST outperforms other defenses while resulting in minimal reduction in testing accuracy.

摘要: 在成员推理(MI)攻击中，对手试图确定是否使用实例来训练机器学习(ML)模型。在使用私有数据训练ML模型时，MI攻击是一个主要的隐私问题。文献中的大多数MI攻击都利用了ML模型经过训练以很好地拟合训练数据的事实，因此在训练实例上的损失非常低。因此，大多数针对MI攻击的防御措施都试图使模型不太适合训练数据。然而，这样做通常会导致精度降低。我们观察到，训练实例对MI攻击具有不同程度的脆弱性。即使不包括在培训中，大多数实例的损失也很低。对于这些实例，该模型可以很好地对它们进行拟合，而无需担心MI攻击。有效的防御只需要(可能是隐式地)识别易受MI攻击的实例，并避免过度匹配它们。一个主要的挑战是如何在有效的培训过程中达到这样的效果。利用表示学习的两个不同的最新进展：反事实不变表示和子空间学习方法，我们引入了一种新的成员不变子空间训练(MIST)方法来防御MI攻击。MIST避免对易受攻击的实例过度拟合，而不会对其他实例产生重大影响。我们进行了广泛的实验研究，将MIST与其他各种最先进的(SOTA)MI防御系统进行了比较，以抵御几种SOTA MI攻击。我们发现，MIST的性能优于其他防御系统，同时对测试精度的影响也很小。



## **36. Optimal Cost Constrained Adversarial Attacks For Multiple Agent Systems**

多智能体系统的最优代价约束对抗攻击 cs.LG

Submitted to ICCASP2024

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2311.00859v1) [paper-pdf](http://arxiv.org/pdf/2311.00859v1)

**Authors**: Ziqing Lu, Guanlin Liu, Lifeng Cai, Weiyu Xu

**Abstract**: Finding optimal adversarial attack strategies is an important topic in reinforcement learning and the Markov decision process. Previous studies usually assume one all-knowing coordinator (attacker) for whom attacking different recipient (victim) agents incurs uniform costs. However, in reality, instead of using one limitless central attacker, the attacks often need to be performed by distributed attack agents. We formulate the problem of performing optimal adversarial agent-to-agent attacks using distributed attack agents, in which we impose distinct cost constraints on each different attacker-victim pair. We propose an optimal method integrating within-step static constrained attack-resource allocation optimization and between-step dynamic programming to achieve the optimal adversarial attack in a multi-agent system. Our numerical results show that the proposed attacks can significantly reduce the rewards received by the attacked agents.

摘要: 寻找最优的对抗性攻击策略是强化学习和马尔可夫决策过程中的一个重要课题。以前的研究通常假设一个无所不知的协调者(攻击者)，对其攻击不同的接收者(受害者)代理会产生统一的成本。然而，在现实中，攻击往往需要由分布式攻击代理来执行，而不是使用一个无限的中心攻击者。我们描述了使用分布式攻击代理执行最优对手代理到代理攻击的问题，其中我们对每个不同的攻击者-受害者对施加不同的代价约束。提出了一种结合步内静态受限攻击资源分配优化和步间动态规划的优化方法来实现多智能体系统中的最优敌方攻击。数值结果表明，所提出的攻击可以显著减少被攻击代理获得的奖励。



## **37. Robustness Tests for Automatic Machine Translation Metrics with Adversarial Attacks**

自动机器翻译度量在对抗性攻击下的稳健性测试 cs.CL

Accepted in Findings of EMNLP 2023

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2311.00508v1) [paper-pdf](http://arxiv.org/pdf/2311.00508v1)

**Authors**: Yichen Huang, Timothy Baldwin

**Abstract**: We investigate MT evaluation metric performance on adversarially-synthesized texts, to shed light on metric robustness. We experiment with word- and character-level attacks on three popular machine translation metrics: BERTScore, BLEURT, and COMET. Our human experiments validate that automatic metrics tend to overpenalize adversarially-degraded translations. We also identify inconsistencies in BERTScore ratings, where it judges the original sentence and the adversarially-degraded one as similar, while judging the degraded translation as notably worse than the original with respect to the reference. We identify patterns of brittleness that motivate more robust metric development.

摘要: 我们研究了机器翻译在恶意合成文本上的评估度量性能，以阐明度量的稳健性。我们在三个流行的机器翻译指标上进行了单词和字符级别的攻击：BERTScore、BLEURT和Comet。我们的人类实验证实，自动度量往往会过度惩罚对抗性降级的翻译。我们还发现了BERTScore评级中的不一致之处，即它判断原始句子和反面降级的句子相似，而判断降级的翻译在引用方面明显比原始句子差。我们确定了激励更稳健的度量开发的脆性模式。



## **38. Improving Robustness for Vision Transformer with a Simple Dynamic Scanning Augmentation**

一种简单的动态扫描增强方法提高视觉变压器的稳健性 cs.CV

Accepted in Neurocomputing

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2311.00441v1) [paper-pdf](http://arxiv.org/pdf/2311.00441v1)

**Authors**: Shashank Kotyan, Danilo Vasconcellos Vargas

**Abstract**: Vision Transformer (ViT) has demonstrated promising performance in computer vision tasks, comparable to state-of-the-art neural networks. Yet, this new type of deep neural network architecture is vulnerable to adversarial attacks limiting its capabilities in terms of robustness. This article presents a novel contribution aimed at further improving the accuracy and robustness of ViT, particularly in the face of adversarial attacks. We propose an augmentation technique called `Dynamic Scanning Augmentation' that leverages dynamic input sequences to adaptively focus on different patches, thereby maintaining performance and robustness. Our detailed investigations reveal that this adaptability to the input sequence induces significant changes in the attention mechanism of ViT, even for the same image. We introduce four variations of Dynamic Scanning Augmentation, outperforming ViT in terms of both robustness to adversarial attacks and accuracy against natural images, with one variant showing comparable results. By integrating our augmentation technique, we observe a substantial increase in ViT's robustness, improving it from $17\%$ to $92\%$ measured across different types of adversarial attacks. These findings, together with other comprehensive tests, indicate that Dynamic Scanning Augmentation enhances accuracy and robustness by promoting a more adaptive type of attention. In conclusion, this work contributes to the ongoing research on Vision Transformers by introducing Dynamic Scanning Augmentation as a technique for improving the accuracy and robustness of ViT. The observed results highlight the potential of this approach in advancing computer vision tasks and merit further exploration in future studies.

摘要: 视觉转换器(VIT)在计算机视觉任务中表现出与最先进的神经网络相媲美的良好性能。然而，这种新型的深度神经网络结构容易受到对手攻击，从而限制了其健壮性方面的能力。本文提出了一项新的贡献，旨在进一步提高VIT的准确性和稳健性，特别是在面对对手攻击的情况下。我们提出了一种称为动态扫描增强的增强技术，该技术利用动态输入序列自适应地聚焦于不同的补丁，从而保持了性能和稳健性。我们的详细研究表明，这种对输入序列的适应性导致了VIT注意机制的显著变化，即使对相同的图像也是如此。我们介绍了四种动态扫描增强算法，在对敌意攻击的稳健性和对自然图像的准确性方面都优于VIT，其中一种算法的结果与之相当。通过集成我们的增强技术，我们观察到VIT的健壮性有了很大的提高，在不同类型的对抗性攻击中测量到的VIT的健壮性从17美元提高到92美元。这些发现与其他综合性测试一起表明，动态扫描增强通过促进更适应类型的注意来提高准确性和稳健性。总之，这项工作通过引入动态扫描增强作为一种提高VIT准确性和稳健性的技术，为正在进行的视觉转换器研究做出了贡献。观察到的结果突出了这种方法在推进计算机视觉任务方面的潜力，值得在未来的研究中进一步探索。



## **39. NEO-KD: Knowledge-Distillation-Based Adversarial Training for Robust Multi-Exit Neural Networks**

NEO-KD：基于知识蒸馏的鲁棒多出口神经网络对抗训练 cs.LG

10 pages, 4 figures, accepted by 37th Conference on Neural  Information Processing Systems (NeurIPS 2023)

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2311.00428v1) [paper-pdf](http://arxiv.org/pdf/2311.00428v1)

**Authors**: Seokil Ham, Jungwuk Park, Dong-Jun Han, Jaekyun Moon

**Abstract**: While multi-exit neural networks are regarded as a promising solution for making efficient inference via early exits, combating adversarial attacks remains a challenging problem. In multi-exit networks, due to the high dependency among different submodels, an adversarial example targeting a specific exit not only degrades the performance of the target exit but also reduces the performance of all other exits concurrently. This makes multi-exit networks highly vulnerable to simple adversarial attacks. In this paper, we propose NEO-KD, a knowledge-distillation-based adversarial training strategy that tackles this fundamental challenge based on two key contributions. NEO-KD first resorts to neighbor knowledge distillation to guide the output of the adversarial examples to tend to the ensemble outputs of neighbor exits of clean data. NEO-KD also employs exit-wise orthogonal knowledge distillation for reducing adversarial transferability across different submodels. The result is a significantly improved robustness against adversarial attacks. Experimental results on various datasets/models show that our method achieves the best adversarial accuracy with reduced computation budgets, compared to the baselines relying on existing adversarial training or knowledge distillation techniques for multi-exit networks.

摘要: 虽然多出口神经网络被认为是通过早期出口进行有效推理的一种有前途的解决方案，但对抗对手攻击仍然是一个具有挑战性的问题。在多出口网络中，由于不同子模型之间的高度依赖，针对特定出口的敌意例子不仅降低了目标出口的性能，而且同时降低了所有其他出口的性能。这使得多出口网络非常容易受到简单的对抗性攻击。在本文中，我们提出了NEO-KD，一种基于知识蒸馏的对抗性训练策略，基于两个关键贡献来应对这一根本挑战。NEO-KD首先采用邻居知识提炼的方法，引导对抗性实例的输出趋向于干净数据的邻居出口的集成输出。NEO-KD还采用了出口方式的正交知识提取，以减少不同子模型之间的对抗性转移。其结果是显著提高了对对手攻击的健壮性。在不同数据集/模型上的实验结果表明，与依赖于现有的对抗性训练或多出口网络的知识提取技术的基线相比，该方法在减少计算开销的情况下获得了最好的对抗性准确率。



## **40. Adversarial Examples in the Physical World: A Survey**

物理世界中的对抗性例子：综述 cs.CV

Adversarial examples, physical-world scenarios, attacks and defenses

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2311.01473v1) [paper-pdf](http://arxiv.org/pdf/2311.01473v1)

**Authors**: Jiakai Wang, Donghua Wang, Jin Hu, Siyang Wu, Tingsong Jiang, Wen Yao, Aishan Liu, Xianglong Liu

**Abstract**: Deep neural networks (DNNs) have demonstrated high vulnerability to adversarial examples. Besides the attacks in the digital world, the practical implications of adversarial examples in the physical world present significant challenges and safety concerns. However, current research on physical adversarial examples (PAEs) lacks a comprehensive understanding of their unique characteristics, leading to limited significance and understanding. In this paper, we address this gap by thoroughly examining the characteristics of PAEs within a practical workflow encompassing training, manufacturing, and re-sampling processes. By analyzing the links between physical adversarial attacks, we identify manufacturing and re-sampling as the primary sources of distinct attributes and particularities in PAEs. Leveraging this knowledge, we develop a comprehensive analysis and classification framework for PAEs based on their specific characteristics, covering over 100 studies on physical-world adversarial examples. Furthermore, we investigate defense strategies against PAEs and identify open challenges and opportunities for future research. We aim to provide a fresh, thorough, and systematic understanding of PAEs, thereby promoting the development of robust adversarial learning and its application in open-world scenarios.

摘要: 深度神经网络(DNN)对敌意例子表现出很高的脆弱性。除了数字世界中的攻击，物理世界中敌意例子的实际影响也带来了重大挑战和安全问题。然而，目前对身体对抗例子(PAE)的研究缺乏对其独特特征的全面了解，导致其意义和理解有限。在本文中，我们通过彻底检查包括培训、制造和重新采样过程在内的实际工作流程中的PAE的特征来解决这一差距。通过分析物理对抗性攻击之间的联系，我们确定制造和重采样是PAE中不同属性和特殊性的主要来源。利用这些知识，我们根据PAE的具体特征开发了一个全面的分析和分类框架，涵盖了100多个物理世界对抗性例子的研究。此外，我们还研究了针对PAE的防御策略，并确定了未来研究的开放挑战和机会。我们的目标是对PAE提供一个新的、彻底的和系统的理解，从而促进稳健的对抗性学习的发展及其在开放世界场景中的应用。



## **41. Dynamics-aware Adversarial Attack of Adaptive Neural Networks**

自适应神经网络的动态感知敌意攻击 cs.CV

arXiv admin note: text overlap with arXiv:2112.09428

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2210.08159v3) [paper-pdf](http://arxiv.org/pdf/2210.08159v3)

**Authors**: An Tao, Yueqi Duan, Yingqi Wang, Jiwen Lu, Jie Zhou

**Abstract**: In this paper, we investigate the dynamics-aware adversarial attack problem of adaptive neural networks. Most existing adversarial attack algorithms are designed under a basic assumption -- the network architecture is fixed throughout the attack process. However, this assumption does not hold for many recently proposed adaptive neural networks, which adaptively deactivate unnecessary execution units based on inputs to improve computational efficiency. It results in a serious issue of lagged gradient, making the learned attack at the current step ineffective due to the architecture change afterward. To address this issue, we propose a Leaded Gradient Method (LGM) and show the significant effects of the lagged gradient. More specifically, we reformulate the gradients to be aware of the potential dynamic changes of network architectures, so that the learned attack better "leads" the next step than the dynamics-unaware methods when network architecture changes dynamically. Extensive experiments on representative types of adaptive neural networks for both 2D images and 3D point clouds show that our LGM achieves impressive adversarial attack performance compared with the dynamic-unaware attack methods. Code is available at https://github.com/antao97/LGM.

摘要: 本文研究了自适应神经网络的动态感知对抗攻击问题。大多数现有的对抗性攻击算法都是在一个基本假设下设计的--网络体系结构在整个攻击过程中是固定的。然而，这一假设对最近提出的许多自适应神经网络并不成立，这些自适应神经网络基于输入自适应地停用不必要的执行单元来提高计算效率。它导致了严重的梯度滞后问题，使得当前步骤的学习攻击由于之后的体系结构变化而无效。为了解决这个问题，我们提出了一种引导梯度法(LGM)，并展示了滞后梯度的显著影响。更具体地说，我们重新制定了梯度，以了解网络体系结构的潜在动态变化，以便在网络体系结构动态变化时，学习到的攻击比不知道动态变化的方法更好地“引导”下一步。在典型的自适应神经网络上对2D图像和3D点云进行的大量实验表明，与动态未知攻击方法相比，我们的LGM具有令人印象深刻的对抗性攻击性能。代码可在https://github.com/antao97/LGM.上找到



## **42. LFAA: Crafting Transferable Targeted Adversarial Examples with Low-Frequency Perturbations**

LFAA：制作具有低频扰动的可转移目标对抗性实例 cs.CV

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2310.20175v2) [paper-pdf](http://arxiv.org/pdf/2310.20175v2)

**Authors**: Kunyu Wang, Juluan Shi, Wenxuan Wang

**Abstract**: Deep neural networks are susceptible to adversarial attacks, which pose a significant threat to their security and reliability in real-world applications. The most notable adversarial attacks are transfer-based attacks, where an adversary crafts an adversarial example to fool one model, which can also fool other models. While previous research has made progress in improving the transferability of untargeted adversarial examples, the generation of targeted adversarial examples that can transfer between models remains a challenging task. In this work, we present a novel approach to generate transferable targeted adversarial examples by exploiting the vulnerability of deep neural networks to perturbations on high-frequency components of images. We observe that replacing the high-frequency component of an image with that of another image can mislead deep models, motivating us to craft perturbations containing high-frequency information to achieve targeted attacks. To this end, we propose a method called Low-Frequency Adversarial Attack (\name), which trains a conditional generator to generate targeted adversarial perturbations that are then added to the low-frequency component of the image. Extensive experiments on ImageNet demonstrate that our proposed approach significantly outperforms state-of-the-art methods, improving targeted attack success rates by a margin from 3.2\% to 15.5\%.

摘要: 深度神经网络容易受到敌意攻击，这对其在实际应用中的安全性和可靠性构成了严重威胁。最著名的对抗性攻击是基于传输的攻击，在这种攻击中，对手编造一个对抗性的例子来愚弄一个模型，这也可以愚弄其他模型。虽然以前的研究在提高非目标对抗性实例的可转移性方面取得了进展，但生成可以在模型之间转移的目标对抗性实例仍然是一项具有挑战性的任务。在这项工作中，我们提出了一种新的方法，通过利用深层神经网络对图像高频分量扰动的脆弱性来生成可转移的目标对抗性样本。我们观察到，用另一幅图像的高频分量替换另一幅图像的高频分量会误导深层模型，促使我们精心设计包含高频信息的扰动，以实现有针对性的攻击。为此，我们提出了一种称为低频对抗性攻击(\NAME)的方法，它训练一个条件生成器来生成目标对抗性扰动，然后将这些扰动添加到图像的低频分量中。在ImageNet上的大量实验表明，我们提出的方法明显优于最先进的方法，将目标攻击成功率从3.2%提高到15.5%。



## **43. Magmaw: Modality-Agnostic Adversarial Attacks on Machine Learning-Based Wireless Communication Systems**

Magmaw：基于机器学习的无线通信系统的通道无关敌意攻击 cs.CR

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2311.00207v1) [paper-pdf](http://arxiv.org/pdf/2311.00207v1)

**Authors**: Jung-Woo Chang, Ke Sun, Nasimeh Heydaribeni, Seira Hidano, Xinyu Zhang, Farinaz Koushanfar

**Abstract**: Machine Learning (ML) has been instrumental in enabling joint transceiver optimization by merging all physical layer blocks of the end-to-end wireless communication systems. Although there have been a number of adversarial attacks on ML-based wireless systems, the existing methods do not provide a comprehensive view including multi-modality of the source data, common physical layer components, and wireless domain constraints. This paper proposes Magmaw, the first black-box attack methodology capable of generating universal adversarial perturbations for any multimodal signal transmitted over a wireless channel. We further introduce new objectives for adversarial attacks on ML-based downstream applications. The resilience of the attack to the existing widely used defense methods of adversarial training and perturbation signal subtraction is experimentally verified. For proof-of-concept evaluation, we build a real-time wireless attack platform using a software-defined radio system. Experimental results demonstrate that Magmaw causes significant performance degradation even in the presence of the defense mechanisms. Surprisingly, Magmaw is also effective against encrypted communication channels and conventional communications.

摘要: 机器学习(ML)通过合并端到端无线通信系统的所有物理层块，在实现联合收发器优化方面发挥了重要作用。虽然已经有一些针对基于ML的无线系统的对抗性攻击，但现有的方法不能提供包括源数据的多模态、公共物理层组件和无线域约束在内的全面视角。本文提出了Magmaw，这是第一个能够对无线信道上传输的任何多模式信号产生通用对抗性扰动的黑盒攻击方法。我们进一步介绍了针对基于ML的下游应用程序的对抗性攻击的新目标。实验验证了该攻击对现有广泛使用的对抗性训练和扰动信号减法防御方法的抗攻击能力。对于概念验证评估，我们使用软件定义的无线电系统构建了一个实时无线攻击平台。实验结果表明，即使在存在防御机制的情况下，Magmaw也会导致性能显著下降。令人惊讶的是，Magmaw对加密通信渠道和传统通信也很有效。



## **44. Robust Safety Classifier for Large Language Models: Adversarial Prompt Shield**

面向大型语言模型的鲁棒安全分类器：对抗提示盾 cs.CL

11 pages, 2 figures

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2311.00172v1) [paper-pdf](http://arxiv.org/pdf/2311.00172v1)

**Authors**: Jinhwa Kim, Ali Derakhshan, Ian G. Harris

**Abstract**: Large Language Models' safety remains a critical concern due to their vulnerability to adversarial attacks, which can prompt these systems to produce harmful responses. In the heart of these systems lies a safety classifier, a computational model trained to discern and mitigate potentially harmful, offensive, or unethical outputs. However, contemporary safety classifiers, despite their potential, often fail when exposed to inputs infused with adversarial noise. In response, our study introduces the Adversarial Prompt Shield (APS), a lightweight model that excels in detection accuracy and demonstrates resilience against adversarial prompts. Additionally, we propose novel strategies for autonomously generating adversarial training datasets, named Bot Adversarial Noisy Dialogue (BAND) datasets. These datasets are designed to fortify the safety classifier's robustness, and we investigate the consequences of incorporating adversarial examples into the training process. Through evaluations involving Large Language Models, we demonstrate that our classifier has the potential to decrease the attack success rate resulting from adversarial attacks by up to 60%. This advancement paves the way for the next generation of more reliable and resilient conversational agents.

摘要: 大型语言模型的安全性仍然是一个关键问题，因为它们容易受到对抗性攻击，这可能会促使这些系统产生有害的响应。这些系统的核心是一个安全分类器，这是一个经过训练的计算模型，可以识别和减轻潜在的有害、攻击性或不道德的输出。然而，当代的安全分类器，尽管它们的潜力，往往失败时，暴露于输入注入对抗性噪声。作为回应，我们的研究引入了对抗性提示盾（APS），这是一种轻量级模型，在检测准确性方面表现出色，并展示了对抗性提示的弹性。此外，我们还提出了自主生成对抗训练数据集的新策略，称为Bot Adversarial Noisy Dialogue（BAND）数据集。这些数据集旨在加强安全分类器的鲁棒性，我们研究了将对抗性示例纳入训练过程的后果。通过涉及大型语言模型的评估，我们证明了我们的分类器有可能将对抗性攻击的攻击成功率降低高达60%。这一进步为下一代更可靠和更有弹性的会话代理铺平了道路。



## **45. Amoeba: Circumventing ML-supported Network Censorship via Adversarial Reinforcement Learning**

阿米巴：通过对抗性强化学习绕过ML支持的网络审查 cs.CR

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2310.20469v1) [paper-pdf](http://arxiv.org/pdf/2310.20469v1)

**Authors**: Haoyu Liu, Alec F. Diallo, Paul Patras

**Abstract**: Embedding covert streams into a cover channel is a common approach to circumventing Internet censorship, due to censors' inability to examine encrypted information in otherwise permitted protocols (Skype, HTTPS, etc.). However, recent advances in machine learning (ML) enable detecting a range of anti-censorship systems by learning distinct statistical patterns hidden in traffic flows. Therefore, designing obfuscation solutions able to generate traffic that is statistically similar to innocuous network activity, in order to deceive ML-based classifiers at line speed, is difficult.   In this paper, we formulate a practical adversarial attack strategy against flow classifiers as a method for circumventing censorship. Specifically, we cast the problem of finding adversarial flows that will be misclassified as a sequence generation task, which we solve with Amoeba, a novel reinforcement learning algorithm that we design. Amoeba works by interacting with censoring classifiers without any knowledge of their model structure, but by crafting packets and observing the classifiers' decisions, in order to guide the sequence generation process. Our experiments using data collected from two popular anti-censorship systems demonstrate that Amoeba can effectively shape adversarial flows that have on average 94% attack success rate against a range of ML algorithms. In addition, we show that these adversarial flows are robust in different network environments and possess transferability across various ML models, meaning that once trained against one, our agent can subvert other censoring classifiers without retraining.

摘要: 将隐蔽流嵌入覆盖频道是绕过互联网审查的常见方法，因为审查者无法检查以其他允许的协议(Skype、HTTPS等)加密的信息。然而，机器学习(ML)的最新进展使人们能够通过学习隐藏在交通流中的不同统计模式来检测一系列反审查系统。因此，为了在线速下欺骗基于ML的分类器，设计能够产生统计上类似于无害网络活动的流量的混淆解决方案是困难的。在本文中，我们制定了一种实用的对流分类器的对抗性攻击策略，作为一种规避审查的方法。具体地说，我们将发现被错误分类的敌意流的问题转化为序列生成任务，我们使用我们设计的一种新的强化学习算法Amoeba来解决这个问题。变形虫的工作原理是在不了解其模型结构的情况下与审查分类器交互，而是通过精心制作包并观察分类器的决策，以指导序列生成过程。我们使用从两个流行的反审查系统收集的数据进行的实验表明，变形虫能够有效地塑造敌意流，对一系列ML算法的攻击成功率平均为94%。此外，我们还证明了这些敌意流在不同的网络环境中是健壮的，并且具有跨不同ML模型的可转移性，这意味着一旦针对一个模型进行训练，我们的代理就可以颠覆其他审查分类器而不需要重新训练。



## **46. On Extracting Specialized Code Abilities from Large Language Models: A Feasibility Study**

从大型语言模型中提取专业代码能力的可行性研究 cs.SE

13 pages

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2303.03012v4) [paper-pdf](http://arxiv.org/pdf/2303.03012v4)

**Authors**: Zongjie Li, Chaozheng Wang, Pingchuan Ma, Chaowei Liu, Shuai Wang, Daoyuan Wu, Cuiyun Gao, Yang Liu

**Abstract**: Recent advances in large language models (LLMs) significantly boost their usage in software engineering. However, training a well-performing LLM demands a substantial workforce for data collection and annotation. Moreover, training datasets may be proprietary or partially open, and the process often requires a costly GPU cluster. The intellectual property value of commercial LLMs makes them attractive targets for imitation attacks, but creating an imitation model with comparable parameters still incurs high costs. This motivates us to explore a practical and novel direction: slicing commercial black-box LLMs using medium-sized backbone models. In this paper, we explore the feasibility of launching imitation attacks on LLMs to extract their specialized code abilities, such as"code synthesis" and "code translation." We systematically investigate the effectiveness of launching code ability extraction attacks under different code-related tasks with multiple query schemes, including zero-shot, in-context, and Chain-of-Thought. We also design response checks to refine the outputs, leading to an effective imitation training process. Our results show promising outcomes, demonstrating that with a reasonable number of queries, attackers can train a medium-sized backbone model to replicate specialized code behaviors similar to the target LLMs. We summarize our findings and insights to help researchers better understand the threats posed by imitation attacks, including revealing a practical attack surface for generating adversarial code examples against LLMs.

摘要: 大型语言模型(LLM)的最新进展极大地促进了它们在软件工程中的使用。然而，培训一个表现良好的LLM需要大量的数据收集和注释工作人员。此外，训练数据集可能是专有的或部分开放的，该过程通常需要昂贵的GPU集群。商业LLM的知识产权价值使其成为模仿攻击的有吸引力的目标，但创建具有可比参数的模仿模型仍然会招致高昂的成本。这促使我们探索一个实用而新颖的方向：使用中型主干模型对商业黑盒LLM进行切片。在本文中，我们探索了对LLM发动模仿攻击的可行性，以提取其专门的代码能力，如“代码合成”和“代码翻译”。我们系统地研究了在不同的代码相关任务下，使用包括零命中、上下文中和思想链在内的多种查询方案来发起代码能力提取攻击的有效性。我们还设计了响应检查来优化输出，从而实现有效的模仿培训过程。我们的结果显示了令人振奋的结果，表明通过合理数量的查询，攻击者可以训练一个中等大小的主干模型来复制类似于目标LLM的特定代码行为。我们总结了我们的发现和见解，以帮助研究人员更好地理解模仿攻击所构成的威胁，包括揭示一个实用的攻击面，用于生成针对LLM的敌意代码示例。



## **47. Robust nonparametric regression based on deep ReLU neural networks**

基于深度回归神经网络的稳健非参数回归 stat.ME

40 pages

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2310.20294v1) [paper-pdf](http://arxiv.org/pdf/2310.20294v1)

**Authors**: Juntong Chen

**Abstract**: In this paper, we consider robust nonparametric regression using deep neural networks with ReLU activation function. While several existing theoretically justified methods are geared towards robustness against identical heavy-tailed noise distributions, the rise of adversarial attacks has emphasized the importance of safeguarding estimation procedures against systematic contamination. We approach this statistical issue by shifting our focus towards estimating conditional distributions. To address it robustly, we introduce a novel estimation procedure based on $\ell$-estimation. Under a mild model assumption, we establish general non-asymptotic risk bounds for the resulting estimators, showcasing their robustness against contamination, outliers, and model misspecification. We then delve into the application of our approach using deep ReLU neural networks. When the model is well-specified and the regression function belongs to an $\alpha$-H\"older class, employing $\ell$-type estimation on suitable networks enables the resulting estimators to achieve the minimax optimal rate of convergence. Additionally, we demonstrate that deep $\ell$-type estimators can circumvent the curse of dimensionality by assuming the regression function closely resembles the composition of several H\"older functions. To attain this, new deep fully-connected ReLU neural networks have been designed to approximate this composition class. This approximation result can be of independent interest.

摘要: 本文考虑基于RELU激活函数的深度神经网络的稳健非参数回归问题。虽然现有的几种理论上合理的方法针对相同的重尾噪声分布具有稳健性，但对抗性攻击的兴起强调了保护估计过程免受系统污染的重要性。我们通过将我们的重点转移到估计条件分布来处理这个统计问题。为了更好地解决这个问题，我们引入了一种新的估计方法--估计。在温和的模型假设下，我们为所得到的估计量建立了一般的非渐近风险界，展示了它们对污染、异常值和模型错误指定的稳健性。然后，我们使用深度RELU神经网络深入研究我们的方法的应用。当模型被很好地描述并且回归函数属于$-α$-H“老类时，在适当的网络上采用$-型估计使得所得到的估计量能够达到极小极大最优收敛速度.此外，我们还证明了通过假设回归函数非常类似于几个H‘-老函数的组合，深$-型估计可以绕过维度诅咒.为了实现这一点，设计了新的深度全连接RELU神经网络来逼近这一组成类。这种近似结果可能具有独立的意义。



## **48. CFDP: Common Frequency Domain Pruning**

常见的频域修剪 cs.CV

CVPR ECV 2023 Accepted Paper

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2306.04147v2) [paper-pdf](http://arxiv.org/pdf/2306.04147v2)

**Authors**: Samir Khaki, Weihan Luo

**Abstract**: As the saying goes, sometimes less is more -- and when it comes to neural networks, that couldn't be more true. Enter pruning, the art of selectively trimming away unnecessary parts of a network to create a more streamlined, efficient architecture. In this paper, we introduce a novel end-to-end pipeline for model pruning via the frequency domain. This work aims to shed light on the interoperability of intermediate model outputs and their significance beyond the spatial domain. Our method, dubbed Common Frequency Domain Pruning (CFDP) aims to extrapolate common frequency characteristics defined over the feature maps to rank the individual channels of a layer based on their level of importance in learning the representation. By harnessing the power of CFDP, we have achieved state-of-the-art results on CIFAR-10 with GoogLeNet reaching an accuracy of 95.25%, that is, +0.2% from the original model. We also outperform all benchmarks and match the original model's performance on ImageNet, using only 55% of the trainable parameters and 60% of the FLOPs. In addition to notable performances, models produced via CFDP exhibit robustness to a variety of configurations including pruning from untrained neural architectures, and resistance to adversarial attacks. The implementation code can be found at https://github.com/Skhaki18/CFDP.

摘要: 俗话说，有时候少就是多--当谈到神经网络时，这是最正确的。修剪是有选择地修剪网络中不必要的部分，以创建更精简、更高效的架构的艺术。本文提出了一种新的基于频域的端到端模型剪枝流水线。这项工作旨在阐明中间模式输出的互操作性及其在空间领域之外的意义。我们的方法被称为公共频域修剪(CFDP)，目的是外推在特征映射上定义的公共频率特征，以基于它们在学习表示的重要性级别来对层的各个通道进行排名。通过利用CFDP的力量，我们在CIFAR-10上取得了最先进的结果，GoogLeNet达到了95.25%的准确率，即比原始模型+0.2%。我们在ImageNet上的性能也超过了所有基准测试，与原始模型的性能相当，只使用了55%的可训练参数和60%的失败。除了显著的性能外，通过CFDP产生的模型还表现出对各种配置的健壮性，包括从未经训练的神经体系结构中进行修剪，以及对对手攻击的抵抗。实现代码可在https://github.com/Skhaki18/CFDP.上找到



## **49. BERT Lost Patience Won't Be Robust to Adversarial Slowdown**

伯特失去了耐心，不会对对手的减速表现得很健壮 cs.LG

Accepted to NeurIPS 2023 [Poster]

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2310.19152v2) [paper-pdf](http://arxiv.org/pdf/2310.19152v2)

**Authors**: Zachary Coalson, Gabriel Ritter, Rakesh Bobba, Sanghyun Hong

**Abstract**: In this paper, we systematically evaluate the robustness of multi-exit language models against adversarial slowdown. To audit their robustness, we design a slowdown attack that generates natural adversarial text bypassing early-exit points. We use the resulting WAFFLE attack as a vehicle to conduct a comprehensive evaluation of three multi-exit mechanisms with the GLUE benchmark against adversarial slowdown. We then show our attack significantly reduces the computational savings provided by the three methods in both white-box and black-box settings. The more complex a mechanism is, the more vulnerable it is to adversarial slowdown. We also perform a linguistic analysis of the perturbed text inputs, identifying common perturbation patterns that our attack generates, and comparing them with standard adversarial text attacks. Moreover, we show that adversarial training is ineffective in defeating our slowdown attack, but input sanitization with a conversational model, e.g., ChatGPT, can remove perturbations effectively. This result suggests that future work is needed for developing efficient yet robust multi-exit models. Our code is available at: https://github.com/ztcoalson/WAFFLE

摘要: 在本文中，我们系统地评估了多出口语言模型对对抗减速的稳健性。为了审计它们的健壮性，我们设计了一种减速攻击，该攻击绕过提前退出点生成自然的对抗性文本。我们使用由此产生的华夫饼攻击作为工具，使用针对对抗性放缓的GLUE基准对三种多退出机制进行了全面评估。然后，我们展示了我们的攻击显著降低了白盒和黑盒设置下的三种方法所提供的计算节省。一种机制越复杂，就越容易受到对抗性放缓的影响。我们还对受干扰的文本输入执行语言分析，识别我们的攻击产生的常见扰动模式，并将它们与标准的对抗性文本攻击进行比较。此外，我们还证明了对抗性训练不能有效地抵抗我们的减速攻击，但是使用会话模型(如ChatGPT)的输入净化可以有效地去除扰动。这一结果表明，未来需要开展工作来开发高效而稳健的多退出模型。我们的代码请访问：https://github.com/ztcoalson/WAFFLE



## **50. Is Robustness Transferable across Languages in Multilingual Neural Machine Translation?**

在多语言神经机器翻译中，健壮性可以跨语言传递吗？ cs.AI

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2310.20162v1) [paper-pdf](http://arxiv.org/pdf/2310.20162v1)

**Authors**: Leiyu Pan, Supryadi, Deyi Xiong

**Abstract**: Robustness, the ability of models to maintain performance in the face of perturbations, is critical for developing reliable NLP systems. Recent studies have shown promising results in improving the robustness of models through adversarial training and data augmentation. However, in machine translation, most of these studies have focused on bilingual machine translation with a single translation direction. In this paper, we investigate the transferability of robustness across different languages in multilingual neural machine translation. We propose a robustness transfer analysis protocol and conduct a series of experiments. In particular, we use character-, word-, and multi-level noises to attack the specific translation direction of the multilingual neural machine translation model and evaluate the robustness of other translation directions. Our findings demonstrate that the robustness gained in one translation direction can indeed transfer to other translation directions. Additionally, we empirically find scenarios where robustness to character-level noise and word-level noise is more likely to transfer.

摘要: 稳健性，即模型在面临扰动时保持性能的能力，对于开发可靠的NLP系统至关重要。最近的研究表明，通过对抗性训练和数据增强，在提高模型的稳健性方面取得了有希望的结果。然而，在机器翻译中，这些研究大多集中在单一翻译方向的双语机器翻译上。本文研究了多语言神经机器翻译中健壮性在不同语言间的可转移性。我们提出了一种健壮性传输分析协议，并进行了一系列的实验。特别是，我们使用字噪声、词噪声和多层噪声来攻击多语言神经机器翻译模型的特定翻译方向，并评估其他翻译方向的稳健性。我们的发现表明，在一个翻译方向上获得的稳健性确实可以转移到其他翻译方向上。此外，我们经验地发现，对字符级噪声和词级噪声的稳健性更有可能转移。



