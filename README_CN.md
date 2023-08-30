# Latest Adversarial Attack Papers
**update at 2023-08-30 11:18:19**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Masquerade: Simple and Lightweight Transaction Reordering Mitigation in Blockchains**

伪装：区块链中简单而轻量级的事务重排序缓解 cs.CR

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15347v1) [paper-pdf](http://arxiv.org/pdf/2308.15347v1)

**Authors**: Arti Vedula, Shaileshh Bojja Venkatakrishnan, Abhishek Gupta

**Abstract**: Blockchains offer strong security gurarantees, but cannot protect users against the ordering of transactions. Players such as miners, bots and validators can reorder various transactions and reap significant profits, called the Maximal Extractable Value (MEV). In this paper, we propose an MEV aware protocol design called Masquerade, and show that it will increase user satisfaction and confidence in the system. We propose a strict per-transaction level of ordering to ensure that a transaction is committed either way even if it is revealed. In this protocol, we introduce the notion of a "token" to mitigate the actions taken by an adversary in an attack scenario. Such tokens can be purchased voluntarily by users, who can then choose to include the token numbers in their transactions. If the users include the token in their transactions, then our protocol requires the block-builder to order the transactions strictly according to token numbers. We show through extensive simulations that this reduces the probability that the adversaries can benefit from MEV transactions as compared to existing current practices.

摘要: 区块链提供了强大的安全保证，但无法保护用户免受交易订单的影响。像矿工、机器人和验证员这样的玩家可以对各种交易进行重新排序，并获得可观的利润，称为最大可提取价值(MEV)。在本文中，我们提出了一种MEV感知的协议设计，称为伪装，并表明它将增加用户对系统的满意度和信心。我们提出了严格的每个事务级别的排序，以确保事务以任何一种方式提交，即使它被揭示。在该协议中，我们引入了“令牌”的概念，以减轻对手在攻击场景中所采取的操作。这样的令牌可以由用户自愿购买，然后他们可以选择在他们的交易中包括令牌号。如果用户在其交易中包含令牌，则我们的协议要求块构建器严格按照令牌编号对交易进行排序。我们通过大量的模拟表明，与现有的实践相比，这降低了对手从MEV交易中获益的概率。



## **2. Imperceptible Adversarial Attack on Deep Neural Networks from Image Boundary**

基于图像边界的深层神经网络的潜伏性攻击 cs.LG

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15344v1) [paper-pdf](http://arxiv.org/pdf/2308.15344v1)

**Authors**: Fahad Alrasheedi, Xin Zhong

**Abstract**: Although Deep Neural Networks (DNNs), such as the convolutional neural networks (CNN) and Vision Transformers (ViTs), have been successfully applied in the field of computer vision, they are demonstrated to be vulnerable to well-sought Adversarial Examples (AEs) that can easily fool the DNNs. The research in AEs has been active, and many adversarial attacks and explanations have been proposed since they were discovered in 2014. The mystery of the AE's existence is still an open question, and many studies suggest that DNN training algorithms have blind spots. The salient objects usually do not overlap with boundaries; hence, the boundaries are not the DNN model's attention. Nevertheless, recent studies show that the boundaries can dominate the behavior of the DNN models. Hence, this study aims to look at the AEs from a different perspective and proposes an imperceptible adversarial attack that systemically attacks the input image boundary for finding the AEs. The experimental results have shown that the proposed boundary attacking method effectively attacks six CNN models and the ViT using only 32% of the input image content (from the boundaries) with an average success rate (SR) of 95.2% and an average peak signal-to-noise ratio of 41.37 dB. Correlation analyses are conducted, including the relation between the adversarial boundary's width and the SR and how the adversarial boundary changes the DNN model's attention. This paper's discoveries can potentially advance the understanding of AEs and provide a different perspective on how AEs can be constructed.

摘要: 虽然深度神经网络(DNN)，如卷积神经网络(CNN)和视觉变形器(VITS)已经成功地应用于计算机视觉领域，但它们被证明是脆弱的，可以很容易地欺骗DNN。AEs的研究一直很活跃，自2014年发现以来，提出了许多对抗性攻击和解释。声发射的存在之谜仍然是一个悬而未决的问题，许多研究表明DNN训练算法存在盲区。显著对象通常不与边界重叠；因此，边界不在DNN模型的关注范围内。然而，最近的研究表明，边界可以支配DNN模型的行为。因此，本研究旨在从不同的角度来看待特效图像，并提出了一种系统地攻击输入图像边界来寻找特效图像的隐蔽对抗性攻击方法。实验结果表明，所提出的边界攻击方法仅用32%的输入图像内容(来自边界)就能有效地攻击6个CNN模型和VIT，平均成功率为95.2%，平均峰值信噪比为41.37dB。进行了相关分析，包括对抗性边界宽度与SR的关系，以及对抗性边界如何改变DNN模型的关注度。这篇论文的发现可能会促进对企业实体的理解，并为如何构建企业实体提供一个不同的视角。



## **3. Longest-chain Attacks: Difficulty Adjustment and Timestamp Verifiability**

最长链攻击：难度调整和时间戳可验证性 cs.CR

A short version appears at MobiHoc23 as a poster

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15312v1) [paper-pdf](http://arxiv.org/pdf/2308.15312v1)

**Authors**: Tzuo Hann Law, Selman Erol, Lewis Tseng

**Abstract**: We study an adversary who attacks a Proof-of-Work (POW) blockchain by selfishly constructing an alternative longest chain. We characterize optimal strategies employed by the adversary when a difficulty adjustment rule al\`a Bitcoin applies. As time (namely the times-tamp specified in each block) in most permissionless POW blockchains is somewhat subjective, we focus on two extreme scenarios: when time is completely verifiable, and when it is completely unverifiable. We conclude that an adversary who faces a difficulty adjustment rule will find a longest-chain attack very challenging when timestamps are verifiable. POW blockchains with frequent difficulty adjustments relative to time reporting flexibility will be substantially more vulnerable to longest-chain attacks. Our main fining provides guidance on the design of difficulty adjustment rules and demonstrates the importance of timestamp verifiability.

摘要: 我们研究了一个通过自私地构建替代最长链来攻击工作证明(POW)区块链的对手。我们刻画了当难度调整规则或比特币适用时，对手所采用的最优策略。由于大多数未经许可的战俘区块链中的时间(即每个区块中指定的时间篡改)具有一定的主观性，因此我们关注两个极端场景：当时间完全可验证时，以及当它完全不可验证时。我们的结论是，当时间戳可验证时，面临难度调整规则的对手将发现最长链攻击非常具有挑战性。战俘区块链相对于时间报告灵活性具有频繁的难度调整，将大大更容易受到最长链攻击。我们的主要提炼为难度调整规则的设计提供了指导，并证明了时间戳可验证性的重要性。



## **4. A Classification-Guided Approach for Adversarial Attacks against Neural Machine Translation**

一种分类制导的神经机器翻译对抗性攻击方法 cs.CL

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15246v1) [paper-pdf](http://arxiv.org/pdf/2308.15246v1)

**Authors**: Sahar Sadrizadeh, Ljiljana Dolamic, Pascal Frossard

**Abstract**: Neural Machine Translation (NMT) models have been shown to be vulnerable to adversarial attacks, wherein carefully crafted perturbations of the input can mislead the target model. In this paper, we introduce ACT, a novel adversarial attack framework against NMT systems guided by a classifier. In our attack, the adversary aims to craft meaning-preserving adversarial examples whose translations by the NMT model belong to a different class than the original translations in the target language. Unlike previous attacks, our new approach has a more substantial effect on the translation by altering the overall meaning, which leads to a different class determined by a classifier. To evaluate the robustness of NMT models to this attack, we propose enhancements to existing black-box word-replacement-based attacks by incorporating output translations of the target NMT model and the output logits of a classifier within the attack process. Extensive experiments in various settings, including a comparison with existing untargeted attacks, demonstrate that the proposed attack is considerably more successful in altering the class of the output translation and has more effect on the translation. This new paradigm can show the vulnerabilities of NMT systems by focusing on the class of translation rather than the mere translation quality as studied traditionally.

摘要: 神经机器翻译(NMT)模型已被证明容易受到敌意攻击，其中精心设计的输入扰动可能会误导目标模型。本文介绍了一种新的基于分类器的NMT系统对抗性攻击框架ACT。在我们的攻击中，对手的目标是制作保持意义的对抗性例子，其NMT模型的翻译与目标语言的原始翻译属于不同的类别。与以前的攻击不同，我们的新方法通过改变整体意义来对翻译产生更实质性的影响，这导致了由分类器确定的不同类别。为了评估NMT模型对这种攻击的稳健性，我们通过在攻击过程中结合目标NMT模型的输出翻译和分类器的输出日志，对现有的基于黑盒单词替换的攻击进行了增强。在不同环境下的大量实验，包括与现有的非定向攻击的比较，表明所提出的攻击在改变输出翻译的类别方面取得了显著的成功，并且对翻译产生了更大的影响。这种新的范式可以通过关注翻译的类别而不是传统研究的翻译质量来揭示NMT系统的脆弱性。



## **5. Can We Rely on AI?**

我们能依靠人工智能吗？ math.NA

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15092v1) [paper-pdf](http://arxiv.org/pdf/2308.15092v1)

**Authors**: Desmond J. Higham

**Abstract**: Over the last decade, adversarial attack algorithms have revealed instabilities in deep learning tools. These algorithms raise issues regarding safety, reliability and interpretability in artificial intelligence; especially in high risk settings. From a practical perspective, there has been a war of escalation between those developing attack and defence strategies. At a more theoretical level, researchers have also studied bigger picture questions concerning the existence and computability of attacks. Here we give a brief overview of the topic, focusing on aspects that are likely to be of interest to researchers in applied and computational mathematics.

摘要: 在过去的十年里，对抗性攻击算法揭示了深度学习工具的不稳定性。这些算法提出了人工智能中的安全性、可靠性和可解释性问题；特别是在高风险环境中。从实践的角度来看，那些制定攻防战略的人之间已经发生了一场不断升级的战争。在更理论的层面上，研究人员还研究了有关攻击的存在和可计算性的更大问题。在这里，我们对这个主题做一个简要的概述，集中在应用数学和计算数学研究人员可能感兴趣的方面。



## **6. Advancing Adversarial Robustness Through Adversarial Logit Update**

通过对抗性Logit更新提高对抗性健壮性 cs.LG

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15072v1) [paper-pdf](http://arxiv.org/pdf/2308.15072v1)

**Authors**: Hao Xuan, Peican Zhu, Xingyu Li

**Abstract**: Deep Neural Networks are susceptible to adversarial perturbations. Adversarial training and adversarial purification are among the most widely recognized defense strategies. Although these methods have different underlying logic, both rely on absolute logit values to generate label predictions. In this study, we theoretically analyze the logit difference around successful adversarial attacks from a theoretical point of view and propose a new principle, namely Adversarial Logit Update (ALU), to infer adversarial sample's labels. Based on ALU, we introduce a new classification paradigm that utilizes pre- and post-purification logit differences for model's adversarial robustness boost. Without requiring adversarial or additional data for model training, our clean data synthesis model can be easily applied to various pre-trained models for both adversarial sample detection and ALU-based data classification. Extensive experiments on both CIFAR-10, CIFAR-100, and tiny-ImageNet datasets show that even with simple components, the proposed solution achieves superior robustness performance compared to state-of-the-art methods against a wide range of adversarial attacks. Our python implementation is submitted in our Supplementary document and will be published upon the paper's acceptance.

摘要: 深度神经网络容易受到对抗性扰动的影响。对抗性训练和对抗性净化是最广为人知的防御策略。尽管这两种方法具有不同的底层逻辑，但它们都依赖于绝对Logit值来生成标签预测。在本研究中，我们从理论的角度分析了成功的对抗性攻击前后的Logit差异，并提出了一种新的原理，即对抗性Logit更新(ALU)来推断对抗性样本的标签。在ALU的基础上，我们引入了一种新的分类范式，它利用净化前后的Logit差异来提高模型的对抗性健壮性。在不需要对抗性或额外数据进行模型训练的情况下，我们的清洁数据合成模型可以很容易地应用于各种预先训练的模型，用于对抗性样本检测和基于ALU的数据分类。在CIFAR-10、CIFAR-100和Tiny-ImageNet数据集上的广泛实验表明，与最先进的方法相比，所提出的解决方案即使具有简单的组件，也可以在应对广泛的对手攻击时获得卓越的健壮性性能。我们的Python实现在我们的补充文档中提交，并将在论文接受后发布。



## **7. Double Public Key Signing Function Oracle Attack on EdDSA Software Implementations**

双重公钥签名函数Oracle对EdDSA软件实现的攻击 cs.CR

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.15009v1) [paper-pdf](http://arxiv.org/pdf/2308.15009v1)

**Authors**: Sam Grierson, Konstantinos Chalkias, William J Buchanan

**Abstract**: EdDSA is a standardised elliptic curve digital signature scheme introduced to overcome some of the issues prevalent in the more established ECDSA standard. Due to the EdDSA standard specifying that the EdDSA signature be deterministic, if the signing function were to be used as a public key signing oracle for the attacker, the unforgeability notion of security of the scheme can be broken. This paper describes an attack against some of the most popular EdDSA implementations, which results in an adversary recovering the private key used during signing. With this recovered secret key, an adversary can sign arbitrary messages that would be seen as valid by the EdDSA verification function. A list of libraries with vulnerable APIs at the time of publication is provided. Furthermore, this paper provides two suggestions for securing EdDSA signing APIs against this vulnerability while it additionally discusses failed attempts to solve the issue.

摘要: EdDSA是一种标准化的椭圆曲线数字签名方案，引入该方案是为了克服在更成熟的ECDSA标准中普遍存在的一些问题。由于EdDSA标准规定EdDSA签名是确定性的，如果签名函数被用作攻击者的公钥签名预言，则方案的安全性的不可伪造性概念可能被打破。本文描述了对一些最流行的EdDSA实现的攻击，该攻击导致攻击者恢复签名期间使用的私钥。利用恢复的密钥，攻击者可以对EdDSA验证功能认为有效的任意消息进行签名。提供了在发布时具有易受攻击的API的库列表。此外，本文还提供了两条保护EdDSA签名API免受该漏洞攻击的建议，同时还讨论了解决该问题的失败尝试。



## **8. Stealthy Backdoor Attack for Code Models**

针对代码模型的隐蔽后门攻击 cs.CR

18 pages, Under review of IEEE Transactions on Software Engineering

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2301.02496v2) [paper-pdf](http://arxiv.org/pdf/2301.02496v2)

**Authors**: Zhou Yang, Bowen Xu, Jie M. Zhang, Hong Jin Kang, Jieke Shi, Junda He, David Lo

**Abstract**: Code models, such as CodeBERT and CodeT5, offer general-purpose representations of code and play a vital role in supporting downstream automated software engineering tasks. Most recently, code models were revealed to be vulnerable to backdoor attacks. A code model that is backdoor-attacked can behave normally on clean examples but will produce pre-defined malicious outputs on examples injected with triggers that activate the backdoors. Existing backdoor attacks on code models use unstealthy and easy-to-detect triggers. This paper aims to investigate the vulnerability of code models with stealthy backdoor attacks. To this end, we propose AFRAIDOOR (Adversarial Feature as Adaptive Backdoor). AFRAIDOOR achieves stealthiness by leveraging adversarial perturbations to inject adaptive triggers into different inputs. We evaluate AFRAIDOOR on three widely adopted code models (CodeBERT, PLBART and CodeT5) and two downstream tasks (code summarization and method name prediction). We find that around 85% of adaptive triggers in AFRAIDOOR bypass the detection in the defense process. By contrast, only less than 12% of the triggers from previous work bypass the defense. When the defense method is not applied, both AFRAIDOOR and baselines have almost perfect attack success rates. However, once a defense is applied, the success rates of baselines decrease dramatically to 10.47% and 12.06%, while the success rate of AFRAIDOOR are 77.05% and 92.98% on the two tasks. Our finding exposes security weaknesses in code models under stealthy backdoor attacks and shows that the state-of-the-art defense method cannot provide sufficient protection. We call for more research efforts in understanding security threats to code models and developing more effective countermeasures.

摘要: 代码模型，如CodeBERT和CodeT5，提供了代码的通用表示，并在支持下游自动化软件工程任务方面发挥了至关重要的作用。最近，代码模型被发现容易受到后门攻击。被后门攻击的代码模型可以在干净的示例上正常运行，但会在注入了激活后门的触发器的示例上生成预定义的恶意输出。现有对代码模型的后门攻击使用隐蔽且易于检测的触发器。本文旨在研究具有隐蔽后门攻击的代码模型的脆弱性。为此，我们提出了AFRAIDOOR(对抗性特征作为自适应后门)。AFRAIDOOR通过利用对抗性扰动将自适应触发器注入不同的输入来实现隐蔽性。我们在三个广泛采用的代码模型(CodeBERT、PLBART和CodeT5)和两个下游任务(代码摘要和方法名称预测)上对AFRAIDOOR进行了评估。我们发现，AFRAIDOOR中约85%的自适应触发器在防御过程中绕过了检测。相比之下，只有不到12%的以前工作中的触发因素绕过了防御。当不应用防御方法时，AFRAIDOOR和基线都具有几乎完美的攻击成功率。然而，一旦实施防御，基线的成功率急剧下降到10.47%和12.06%，而AFRAIDOOR在两个任务上的成功率分别为77.05%和92.98%。我们的发现暴露了代码模型在秘密后门攻击下的安全漏洞，并表明最先进的防御方法不能提供足够的保护。我们呼吁在了解代码模型的安全威胁和开发更有效的对策方面做出更多研究努力。



## **9. WSAM: Visual Explanations from Style Augmentation as Adversarial Attacker and Their Influence in Image Classification**

WSAM：作为对抗性攻击者的风格提升的视觉解释及其对图像分类的影响 cs.CV

8 pages, 10 figures

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2308.14995v1) [paper-pdf](http://arxiv.org/pdf/2308.14995v1)

**Authors**: Felipe Moreno-Vera, Edgar Medina, Jorge Poco

**Abstract**: Currently, style augmentation is capturing attention due to convolutional neural networks (CNN) being strongly biased toward recognizing textures rather than shapes. Most existing styling methods either perform a low-fidelity style transfer or a weak style representation in the embedding vector. This paper outlines a style augmentation algorithm using stochastic-based sampling with noise addition to improving randomization on a general linear transformation for style transfer. With our augmentation strategy, all models not only present incredible robustness against image stylizing but also outperform all previous methods and surpass the state-of-the-art performance for the STL-10 dataset. In addition, we present an analysis of the model interpretations under different style variations. At the same time, we compare comprehensive experiments demonstrating the performance when applied to deep neural architectures in training settings.

摘要: 目前，由于卷积神经网络(CNN)强烈偏向于识别纹理而不是形状，样式增强正吸引着人们的注意。大多数现有的样式设置方法要么执行低保真样式转换，要么在嵌入向量中执行弱样式表示。本文提出了一种基于随机采样和噪声的风格增强算法，改进了一般线性变换的随机性，用于风格转移。使用我们的增强策略，所有模型不仅在图像样式化方面表现出令人难以置信的健壮性，而且性能优于所有以前的方法，并超过STL-10数据集的最先进性能。此外，我们还对不同风格变化下的模型解释进行了分析。同时，我们比较了在训练环境中应用于深层神经结构时的性能的综合实验。



## **10. Randomized Line-to-Row Mapping for Low-Overhead Rowhammer Mitigations**

用于低开销Rowhammer缓解的随机化行到行映射 cs.CR

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14907v1) [paper-pdf](http://arxiv.org/pdf/2308.14907v1)

**Authors**: Anish Saxena, Saurav Mathur, Moinuddin Qureshi

**Abstract**: Modern systems mitigate Rowhammer using victim refresh, which refreshes the two neighbours of an aggressor row when it encounters a specified number of activations. Unfortunately, complex attack patterns like Half-Double break victim-refresh, rendering current systems vulnerable. Instead, recently proposed secure Rowhammer mitigations rely on performing mitigative action on the aggressor rather than the victims. Such schemes employ mitigative actions such as row-migration or access-control and include AQUA, SRS, and Blockhammer. While these schemes incur only modest slowdowns at Rowhammer thresholds of few thousand, they incur prohibitive slowdowns (15%-600%) for lower thresholds that are likely in the near future. The goal of our paper is to make secure Rowhammer mitigations practical at such low thresholds.   Our paper provides the key insights that benign application encounter thousands of hot rows (receiving more activations than the threshold) due to the memory mapping, which places spatially proximate lines in the same row to maximize row-buffer hitrate. Unfortunately, this causes row to receive activations for many frequently used lines. We propose Rubix, which breaks the spatial correlation in the line-to-row mapping by using an encrypted address to access the memory, reducing the likelihood of hot rows by 2 to 3 orders of magnitude. To aid row-buffer hits, Rubix randomizes a group of 1-4 lines. We also propose Rubix-D, which dynamically changes the line-to-row mapping. Rubix-D minimizes hot-rows and makes it much harder for an adversary to learn the spatial neighbourhood of a row. Rubix reduces the slowdown of AQUA (from 15% to 1%), SRS (from 60% to 2%), and Blockhammer (from 600% to 3%) while incurring a storage of less than 1 Kilobyte.

摘要: 现代系统使用受害者刷新来缓解Rowhammer，当遇到指定数量的激活时，受害者刷新会刷新攻击者行的两个相邻行。不幸的是，复杂的攻击模式，如半双中断受害者刷新，使当前的系统容易受到攻击。相反，最近提出的安全罗哈默减轻依赖于对侵略者而不是受害者执行减轻行动。此类方案采用行迁移或访问控制等缓解措施，包括Aqua、SRS和BlockHammer。虽然这些计划在罗哈默几千人的门槛下只会导致适度的减速，但它们会导致令人望而却步的减速(15%-600%)，因为在不久的将来可能会降低门槛。我们论文的目标是在如此低的门槛下使安全的罗哈默减刑变得切实可行。我们的论文提供了一些关键的见解，即良性应用程序会遇到数千个热行(接收的激活数超过阈值)，这是因为内存映射将空间上接近的行放置在同一行中，以最大化行缓冲区命中率。不幸的是，这会导致ROW接收许多常用行的激活。我们提出了Rubix，它通过使用加密地址访问存储器，打破了行到行映射中的空间相关性，将热行的可能性降低了2到3个数量级。为了帮助行缓冲区命中，Rubix随机化了一组1-4行。我们还提出了Rubix-D，它动态地改变行到行的映射。Rubix-D将热行最小化，并使对手更难了解行的空间邻域。Rubix降低了Aqua(从15%到1%)、SRS(从60%到2%)和块锤(从600%到3%)的速度，同时产生了不到1千字节的存储。



## **11. A Stochastic Surveillance Stackelberg Game: Co-Optimizing Defense Placement and Patrol Strategy**

随机监视Stackelberg博弈：共同优化防御部署和巡逻策略 eess.SY

8 pages, 1 figure, jointly submitted to the IEEE Control Systems  Letters and the 2024 American Control Conference

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14714v1) [paper-pdf](http://arxiv.org/pdf/2308.14714v1)

**Authors**: Yohan John, Gilberto Diaz-Garcia, Xiaoming Duan, Jason R. Marden, Francesco Bullo

**Abstract**: Stochastic patrol routing is known to be advantageous in adversarial settings; however, the optimal choice of stochastic routing strategy is dependent on a model of the adversary. Duan et al. formulated a Stackelberg game for the worst-case scenario, i.e., a surveillance agent confronted with an omniscient attacker [IEEE TCNS, 8(2), 769-80, 2021]. In this article, we extend their formulation to accommodate heterogeneous defenses at the various nodes of the graph. We derive an upper bound on the value of the game. We identify methods for computing effective patrol strategies for certain classes of graphs. Finally, we leverage the heterogeneous defense formulation to develop novel defense placement algorithms that complement the patrol strategies.

摘要: 随机巡逻路径在敌方环境中具有优势，然而，随机路径策略的最优选择取决于敌方的模型。Duan et al.为最糟糕的情况制定了Stackelberg游戏，即监视特工与无所不知的攻击者对峙[IEEE TCNs，8(2)，769-80,2021]。在这篇文章中，我们扩展了他们的公式，以适应图的不同节点上的不同防御。我们得到了博弈价值的一个上界。我们确定了计算某些图类的有效巡视策略的方法。最后，我们利用异质防御公式来开发新的防御布局算法，以补充巡逻策略。



## **12. Adversarial Attacks on Foundational Vision Models**

对基本视觉模型的对抗性攻击 cs.CV

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14597v1) [paper-pdf](http://arxiv.org/pdf/2308.14597v1)

**Authors**: Nathan Inkawhich, Gwendolyn McDonald, Ryan Luley

**Abstract**: Rapid progress is being made in developing large, pretrained, task-agnostic foundational vision models such as CLIP, ALIGN, DINOv2, etc. In fact, we are approaching the point where these models do not have to be finetuned downstream, and can simply be used in zero-shot or with a lightweight probing head. Critically, given the complexity of working at this scale, there is a bottleneck where relatively few organizations in the world are executing the training then sharing the models on centralized platforms such as HuggingFace and torch.hub. The goal of this work is to identify several key adversarial vulnerabilities of these models in an effort to make future designs more robust. Intuitively, our attacks manipulate deep feature representations to fool an out-of-distribution (OOD) detector which will be required when using these open-world-aware models to solve closed-set downstream tasks. Our methods reliably make in-distribution (ID) images (w.r.t. a downstream task) be predicted as OOD and vice versa while existing in extremely low-knowledge-assumption threat models. We show our attacks to be potent in whitebox and blackbox settings, as well as when transferred across foundational model types (e.g., attack DINOv2 with CLIP)! This work is only just the beginning of a long journey towards adversarially robust foundational vision models.

摘要: 在开发大型的、预先训练的、与任务无关的基础视觉模型方面，如CLIP、ALIGN、DINOv2等，正在取得快速进展。事实上，我们正在接近这样一个点，这些模型不必在下游进行微调，只需在零射击或带有轻型探头的情况下使用即可。关键是，考虑到在这种规模下工作的复杂性，存在一个瓶颈，即世界上执行培训并在HuggingFace和Torch.Hub等集中式平台上共享模型的组织相对较少。这项工作的目标是确定这些模型的几个关键的对抗性漏洞，以努力使未来的设计更健壮。直观地说，我们的攻击操纵深层特征表示来愚弄分布外(OOD)检测器，这将是使用这些开放世界感知模型来解决封闭集下游任务时所需的。我们的方法可靠地生成分布内(ID)图像(W.r.t.下游任务)被预测为OOD，反之亦然，而存在于极低知识假设的威胁模型中。我们展示了我们的攻击在白盒和黑盒设置中以及在跨基本模型类型传输时是有效的(例如，使用CLIP攻击DINOv2)！这项工作仅仅是迈向相反稳健的基础愿景模型的漫长旅程的开始。



## **13. ReMAV: Reward Modeling of Autonomous Vehicles for Finding Likely Failure Events**

ReMAV：自动车辆发现可能故障事件的奖励模型 cs.AI

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14550v1) [paper-pdf](http://arxiv.org/pdf/2308.14550v1)

**Authors**: Aizaz Sharif, Dusica Marijan

**Abstract**: Autonomous vehicles are advanced driving systems that are well known for being vulnerable to various adversarial attacks, compromising the vehicle's safety, and posing danger to other road users. Rather than actively training complex adversaries by interacting with the environment, there is a need to first intelligently find and reduce the search space to only those states where autonomous vehicles are found less confident. In this paper, we propose a blackbox testing framework ReMAV using offline trajectories first to analyze the existing behavior of autonomous vehicles and determine appropriate thresholds for finding the probability of failure events. Our reward modeling technique helps in creating a behavior representation that allows us to highlight regions of likely uncertain behavior even when the baseline autonomous vehicle is performing well. This approach allows for more efficient testing without the need for computational and inefficient active adversarial learning techniques. We perform our experiments in a high-fidelity urban driving environment using three different driving scenarios containing single and multi-agent interactions. Our experiment shows 35%, 23%, 48%, and 50% increase in occurrences of vehicle collision, road objects collision, pedestrian collision, and offroad steering events respectively by the autonomous vehicle under test, demonstrating a significant increase in failure events. We also perform a comparative analysis with prior testing frameworks and show that they underperform in terms of training-testing efficiency, finding total infractions, and simulation steps to identify the first failure compared to our approach. The results show that the proposed framework can be used to understand existing weaknesses of the autonomous vehicles under test in order to only attack those regions, starting with the simplistic perturbation models.

摘要: 自动驾驶汽车是一种先进的驾驶系统，众所周知，它容易受到各种对抗性攻击，危及车辆的安全，并对其他道路使用者构成危险。与其通过与环境互动来积极训练复杂的对手，不如首先智能地找到搜索空间，并将搜索空间缩小到那些自动驾驶汽车被发现信心较低的状态。在本文中，我们提出了一个基于离线轨迹的黑盒测试框架ReMAV，首先分析自动驾驶车辆的现有行为，并确定合适的阈值来发现故障事件的概率。我们的奖励建模技术有助于创建行为表示，使我们能够突出可能的不确定行为区域，即使基准自动驾驶汽车表现良好。这种方法允许更有效的测试，而不需要计算和低效的主动对抗性学习技术。我们在高保真的城市驾驶环境中使用三种不同的驾驶场景进行了实验，其中包含单代理和多代理交互。我们的实验表明，被测自动驾驶汽车的车辆碰撞、道路物体碰撞、行人碰撞和越野转向事件的发生率分别增加了35%、23%、48%和50%，表明故障事件显著增加。我们还与以前的测试框架进行了比较分析，结果表明，与我们的方法相比，它们在训练测试效率、找到总违规行为以及识别第一个故障的模拟步骤方面表现不佳。结果表明，该框架可以用来理解被测自动驾驶车辆存在的弱点，以便从简化的扰动模型开始只攻击这些区域。



## **14. Efficient Decision-based Black-box Patch Attacks on Video Recognition**

视频识别中高效的基于决策的黑盒补丁攻击 cs.CV

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2303.11917v2) [paper-pdf](http://arxiv.org/pdf/2303.11917v2)

**Authors**: Kaixun Jiang, Zhaoyu Chen, Hao Huang, Jiafeng Wang, Dingkang Yang, Bo Li, Yan Wang, Wenqiang Zhang

**Abstract**: Although Deep Neural Networks (DNNs) have demonstrated excellent performance, they are vulnerable to adversarial patches that introduce perceptible and localized perturbations to the input. Generating adversarial patches on images has received much attention, while adversarial patches on videos have not been well investigated. Further, decision-based attacks, where attackers only access the predicted hard labels by querying threat models, have not been well explored on video models either, even if they are practical in real-world video recognition scenes. The absence of such studies leads to a huge gap in the robustness assessment for video models. To bridge this gap, this work first explores decision-based patch attacks on video models. We analyze that the huge parameter space brought by videos and the minimal information returned by decision-based models both greatly increase the attack difficulty and query burden. To achieve a query-efficient attack, we propose a spatial-temporal differential evolution (STDE) framework. First, STDE introduces target videos as patch textures and only adds patches on keyframes that are adaptively selected by temporal difference. Second, STDE takes minimizing the patch area as the optimization objective and adopts spatialtemporal mutation and crossover to search for the global optimum without falling into the local optimum. Experiments show STDE has demonstrated state-of-the-art performance in terms of threat, efficiency and imperceptibility. Hence, STDE has the potential to be a powerful tool for evaluating the robustness of video recognition models.

摘要: 尽管深度神经网络(DNN)表现出了很好的性能，但它们很容易受到敌意补丁的攻击，这些补丁会给输入带来可感知的局部扰动。在图像上生成敌意补丁已经得到了很大的关注，而视频上的敌意补丁还没有得到很好的研究。此外，基于决策的攻击(攻击者仅通过查询威胁模型来访问预测的硬标签)在视频模型上也没有得到很好的探索，即使它们在现实世界的视频识别场景中是实用的。这类研究的缺乏导致了视频模型稳健性评估的巨大差距。为了弥补这一差距，这项工作首先探索了基于决策的视频模型补丁攻击。分析了视频带来的巨大参数空间和基于决策的模型返回的最小信息量都大大增加了攻击难度和查询负担。为了实现查询高效的攻击，我们提出了一种时空差异进化(STDE)框架。首先，STDE将目标视频作为补丁纹理引入，只在根据时间差异自适应选择的关键帧上添加补丁。其次，STDE算法以面片面积最小为优化目标，采用时空变异和交叉来搜索全局最优解而不陷入局部最优。实验表明，STDE在威胁、效率和不可感知性方面都表现出了最先进的性能。因此，STDE有可能成为评估视频识别模型稳健性的有力工具。



## **15. Mitigating the source-side channel vulnerability by characterization of photon statistics**

通过表征光子统计信息来缓解源端通道的脆弱性 quant-ph

Comments and suggestions are welcomed

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14402v1) [paper-pdf](http://arxiv.org/pdf/2308.14402v1)

**Authors**: Tanya Sharma, Ayan Biswas, Jayanth Ramakrishnan, Pooja Chandravanshi, Ravindra P. Singh

**Abstract**: Quantum key distribution (QKD) theoretically offers unconditional security. Unfortunately, the gap between theory and practice threatens side-channel attacks on practical QKD systems. Many well-known QKD protocols use weak coherent laser pulses to encode the quantum information. These sources differ from ideal single photon sources and follow Poisson statistics. Many protocols, such as decoy state and coincidence detection protocols, rely on monitoring the photon statistics to detect any information leakage. The accurate measurement and characterization of photon statistics enable the detection of adversarial attacks and the estimation of secure key rates, strengthening the overall security of the QKD system. We have rigorously characterized our source to estimate the mean photon number employing multiple detectors for comparison against measurements made with a single detector. Furthermore, we have also studied intensity fluctuations to help identify and mitigate any potential information leakage due to state preparation flaws. We aim to bridge the gap between theory and practice to achieve information-theoretic security.

摘要: 量子密钥分配(QKD)理论上提供了无条件的安全性。不幸的是，理论和实践之间的差距威胁着实际的量子密钥分发系统的旁路攻击。许多著名的量子密钥分发协议使用弱相干激光脉冲来编码量子信息。这些源不同于理想的单光子源，并且遵循泊松统计。许多协议，如诱饵状态和符合检测协议，依赖于监测光子统计信息来检测任何信息泄漏。光子统计的准确测量和表征使得能够检测敌意攻击和估计安全密钥率，从而加强了量子密钥分发系统的整体安全性。我们已经严格地描述了我们的源的特征，以估计使用多个探测器的平均光子数，以与使用单个探测器进行的测量进行比较。此外，我们还研究了强度波动，以帮助识别和缓解由于状态准备缺陷而导致的任何潜在信息泄漏。我们的目标是弥合理论和实践之间的差距，实现信息理论安全。



## **16. QEVSEC: Quick Electric Vehicle SEcure Charging via Dynamic Wireless Power Transfer**

QEVSEC：通过动态无线电能传输实现电动汽车快速安全充电 cs.CR

6 pages, conference

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2205.10292v3) [paper-pdf](http://arxiv.org/pdf/2205.10292v3)

**Authors**: Tommaso Bianchi, Surudhi Asokraj, Alessandro Brighente, Mauro Conti, Radha Poovendran

**Abstract**: Dynamic Wireless Power Transfer (DWPT) can be used for on-demand recharging of Electric Vehicles (EV) while driving. However, DWPT raises numerous security and privacy concerns. Recently, researchers demonstrated that DWPT systems are vulnerable to adversarial attacks. In an EV charging scenario, an attacker can prevent the authorized customer from charging, obtain a free charge by billing a victim user and track a target vehicle. State-of-the-art authentication schemes relying on centralized solutions are either vulnerable to various attacks or have high computational complexity, making them unsuitable for a dynamic scenario. In this paper, we propose Quick Electric Vehicle SEcure Charging (QEVSEC), a novel, secure, and efficient authentication protocol for the dynamic charging of EVs. Our idea for QEVSEC originates from multiple vulnerabilities we found in the state-of-the-art protocol that allows tracking of user activity and is susceptible to replay attacks. Based on these observations, the proposed protocol solves these issues and achieves lower computational complexity by using only primitive cryptographic operations in a very short message exchange. QEVSEC provides scalability and a reduced cost in each iteration, thus lowering the impact on the power needed from the grid.

摘要: 动态无线电能传输(DWPT)可用于电动汽车(EV)行驶时的按需充电。然而，DWPT带来了许多安全和隐私方面的问题。最近，研究人员证明了DWPT系统容易受到敌意攻击。在电动汽车充电场景中，攻击者可以阻止授权客户充电，通过向受害用户收费来获得免费费用，并跟踪目标车辆。依赖于集中式解决方案的最先进的身份验证方案要么容易受到各种攻击，要么具有很高的计算复杂性，不适合动态场景。本文提出了一种新颖、安全、高效的电动汽车动态充电认证协议--快速电动汽车安全充电协议。我们对QEVSEC的想法源于我们在最先进的协议中发现的多个漏洞，该协议允许跟踪用户活动，并且容易受到重播攻击。基于这些观察，提出的协议解决了这些问题，并通过在很短的消息交换中仅使用原始密码操作来实现较低的计算复杂度。QEVSEC在每次迭代中提供了可扩展性和更低的成本，从而降低了对电网所需电力的影响。



## **17. Hiding Visual Information via Obfuscating Adversarial Perturbations**

通过混淆敌意扰动隐藏视觉信息 cs.CV

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2209.15304v4) [paper-pdf](http://arxiv.org/pdf/2209.15304v4)

**Authors**: Zhigang Su, Dawei Zhou, Nannan Wangu, Decheng Li, Zhen Wang, Xinbo Gao

**Abstract**: Growing leakage and misuse of visual information raise security and privacy concerns, which promotes the development of information protection. Existing adversarial perturbations-based methods mainly focus on the de-identification against deep learning models. However, the inherent visual information of the data has not been well protected. In this work, inspired by the Type-I adversarial attack, we propose an adversarial visual information hiding method to protect the visual privacy of data. Specifically, the method generates obfuscating adversarial perturbations to obscure the visual information of the data. Meanwhile, it maintains the hidden objectives to be correctly predicted by models. In addition, our method does not modify the parameters of the applied model, which makes it flexible for different scenarios. Experimental results on the recognition and classification tasks demonstrate that the proposed method can effectively hide visual information and hardly affect the performances of models. The code is available in the supplementary material.

摘要: 日益增长的视觉信息泄露和滥用引发了人们对安全和隐私的担忧，这推动了信息保护的发展。现有的基于对抗性扰动的方法主要集中在针对深度学习模型的去识别。然而，数据固有的视觉信息并没有得到很好的保护。在这项工作中，受Type-I对抗攻击的启发，我们提出了一种对抗性视觉信息隐藏方法来保护数据的视觉隐私。具体地说，该方法产生模糊的对抗性扰动以模糊数据的可视信息。同时，保持模型对隐含目标的正确预测。此外，我们的方法不修改应用模型的参数，这使得它可以灵活地适应不同的场景。在识别和分类任务上的实验结果表明，该方法能够有效地隐藏视觉信息，且几乎不影响模型的性能。该代码可在补充材料中找到。



## **18. Detecting Language Model Attacks with Perplexity**

基于困惑的语言模型攻击检测 cs.CL

**SubmitDate**: 2023-08-27    [abs](http://arxiv.org/abs/2308.14132v1) [paper-pdf](http://arxiv.org/pdf/2308.14132v1)

**Authors**: Gabriel Alon, Michael Kamfonas

**Abstract**: A novel hack involving Large Language Models (LLMs) has emerged, leveraging adversarial suffixes to trick models into generating perilous responses. This method has garnered considerable attention from reputable media outlets such as the New York Times and Wired, thereby influencing public perception regarding the security and safety of LLMs. In this study, we advocate the utilization of perplexity as one of the means to recognize such potential attacks. The underlying concept behind these hacks revolves around appending an unusually constructed string of text to a harmful query that would otherwise be blocked. This maneuver confuses the protective mechanisms and tricks the model into generating a forbidden response. Such scenarios could result in providing detailed instructions to a malicious user for constructing explosives or orchestrating a bank heist. Our investigation demonstrates the feasibility of employing perplexity, a prevalent natural language processing metric, to detect these adversarial tactics before generating a forbidden response. By evaluating the perplexity of queries with and without such adversarial suffixes using an open-source LLM, we discovered that nearly 90 percent were above a perplexity of 1000. This contrast underscores the efficacy of perplexity for detecting this type of exploit.

摘要: 出现了一种涉及大型语言模型(LLM)的新黑客攻击，利用敌意后缀欺骗模型生成危险的响应。这一方法引起了《纽约时报》和《连线》等知名媒体的极大关注，从而影响了公众对低地小武器安全性和安全性的看法。在这项研究中，我们主张使用困惑作为识别这种潜在攻击的手段之一。这些黑客攻击背后的基本概念围绕着在原本会被阻止的有害查询中附加一个构造异常的文本字符串。这种操作混淆了保护机制，并诱使模型产生禁止反应。这种情况可能会导致向恶意用户提供制造炸药或策划银行抢劫的详细说明。我们的研究证明了使用困惑，一种流行的自然语言处理度量，在生成禁止响应之前检测这些敌对策略的可行性。通过使用开源LLM评估带有和不带有这种敌意后缀的查询的困惑程度，我们发现近90%的查询困惑程度高于1000。这种对比突出了困惑在检测这种类型的利用方面的有效性。



## **19. Fairness and Privacy in Voice Biometrics:A Study of Gender Influences Using wav2vec 2.0**

语音生物识别中的公平性和隐私性：基于Wav2vec 2.0的性别影响研究 eess.AS

7 pages

**SubmitDate**: 2023-08-27    [abs](http://arxiv.org/abs/2308.14049v1) [paper-pdf](http://arxiv.org/pdf/2308.14049v1)

**Authors**: Oubaida Chouchane, Michele Panariello, Chiara Galdi, Massimiliano Todisco, Nicholas Evans

**Abstract**: This study investigates the impact of gender information on utility, privacy, and fairness in voice biometric systems, guided by the General Data Protection Regulation (GDPR) mandates, which underscore the need for minimizing the processing and storage of private and sensitive data, and ensuring fairness in automated decision-making systems. We adopt an approach that involves the fine-tuning of the wav2vec 2.0 model for speaker verification tasks, evaluating potential gender-related privacy vulnerabilities in the process. Gender influences during the fine-tuning process were employed to enhance fairness and privacy in order to emphasise or obscure gender information within the speakers' embeddings. Results from VoxCeleb datasets indicate our adversarial model increases privacy against uninformed attacks, yet slightly diminishes speaker verification performance compared to the non-adversarial model. However, the model's efficacy reduces against informed attacks. Analysis of system performance was conducted to identify potential gender biases, thus highlighting the need for further research to understand and improve the delicate interplay between utility, privacy, and equity in voice biometric systems.

摘要: 本研究在一般数据保护法规(GDPR)的指导下，调查了性别信息对语音生物识别系统中的效用、隐私和公平性的影响，该法规强调了将私人和敏感数据的处理和存储降至最低的必要性，并确保自动化决策系统中的公平性。我们采用了一种方法，涉及对说话人验证任务的Wav2vec 2.0模型进行微调，评估过程中潜在的与性别相关的隐私漏洞。在微调过程中采用了性别影响，以加强公平和隐私，以便在发言者的嵌入中强调或模糊性别信息。来自VoxCeleb数据集的结果表明，我们的对抗性模型提高了针对不知情攻击的隐私，但与非对抗性模型相比，说话人验证性能略有下降。然而，该模型对知情攻击的有效性会降低。对系统性能进行了分析，以确定潜在的性别偏见，从而突出了进一步研究的必要性，以了解和改进语音生物识别系统中效用、隐私和公平之间的微妙相互作用。



## **20. Device-Independent Quantum Key Distribution Based on the Mermin-Peres Magic Square Game**

基于Mermin-Peres魔方博弈的设备无关量子密钥分配 quant-ph

**SubmitDate**: 2023-08-27    [abs](http://arxiv.org/abs/2308.14037v1) [paper-pdf](http://arxiv.org/pdf/2308.14037v1)

**Authors**: Yi-Zheng Zhen, Yingqiu Mao, Yu-Zhe Zhang, Feihu Xu, Barry C. Sanders

**Abstract**: Device-independent quantum key distribution (DIQKD) is information-theoretically secure against adversaries who possess a scalable quantum computer and who have supplied malicious key-establishment systems; however, the DIQKD key rate is currently too low. Consequently, we devise a DIQKD scheme based on the quantum nonlocal Mermin-Peres magic square game: our scheme asymptotically delivers DIQKD against collective attacks, even with noise. Our scheme outperforms DIQKD using the Clauser-Horne-Shimony-Holt game with respect to the number of game rounds, albeit not number of entangled pairs, provided that both state visibility and detection efficiency are high enough.

摘要: 独立于设备的量子密钥分发(DIQKD)在信息理论上是安全的，可以抵御拥有可扩展量子计算机并提供恶意密钥建立系统的攻击者；然而，DIQKD密钥率目前太低。因此，我们设计了一个基于量子非局部Mermin-Peres幻方博弈的DIQKD方案：即使在有噪声的情况下，我们的方案也能渐近地提供抗集体攻击的DIQKD。在状态可见性和检测效率都足够高的情况下，我们的方案在游戏轮数上优于使用Clauser-Horne-Shimony-Holt博弈的DIQKD，尽管不是纠缠对的数量。



## **21. A semantic backdoor attack against Graph Convolutional Networks**

一种针对图卷积网络的语义后门攻击 cs.LG

**SubmitDate**: 2023-08-26    [abs](http://arxiv.org/abs/2302.14353v4) [paper-pdf](http://arxiv.org/pdf/2302.14353v4)

**Authors**: Jiazhu Dai, Zhipeng Xiong

**Abstract**: Graph convolutional networks (GCNs) have been very effective in addressing the issue of various graph-structured related tasks. However, recent research has shown that GCNs are vulnerable to a new type of threat called a backdoor attack, where the adversary can inject a hidden backdoor into GCNs so that the attacked model performs well on benign samples, but its prediction will be maliciously changed to the attacker-specified target label if the hidden backdoor is activated by the attacker-defined trigger. A semantic backdoor attack is a new type of backdoor attack on deep neural networks (DNNs), where a naturally occurring semantic feature of samples can serve as a backdoor trigger such that the infected DNN models will misclassify testing samples containing the predefined semantic feature even without the requirement of modifying the testing samples. Since the backdoor trigger is a naturally occurring semantic feature of the samples, semantic backdoor attacks are more imperceptible and pose a new and serious threat. In this paper, we investigate whether such semantic backdoor attacks are possible for GCNs and propose a semantic backdoor attack against GCNs (SBAG) under the context of graph classification to reveal the existence of this security vulnerability in GCNs. SBAG uses a certain type of node in the samples as a backdoor trigger and injects a hidden backdoor into GCN models by poisoning training data. The backdoor will be activated, and the GCN models will give malicious classification results specified by the attacker even on unmodified samples as long as the samples contain enough trigger nodes. We evaluate SBAG on four graph datasets and the experimental results indicate that SBAG is effective.

摘要: 图卷积网络(GCNS)在解决各种与图结构相关的任务问题方面已经非常有效。然而，最近的研究表明，GCNS容易受到一种名为后门攻击的新型威胁的攻击，在这种威胁中，攻击者可以向GCNS注入隐藏的后门，以便攻击模型在良性样本上执行良好，但如果隐藏的后门被攻击者定义的触发器激活，其预测将被恶意更改为攻击者指定的目标标签。语义后门攻击是对深度神经网络(DNN)的一种新型后门攻击，其中样本的自然产生的语义特征可以作为后门触发器，使得被感染的DNN模型即使在不需要修改测试样本的情况下也会对包含预定义语义特征的测试样本进行误分类。由于后门触发器是样本中自然产生的语义特征，语义后门攻击更难以察觉，并构成新的严重威胁。本文研究了GCNS是否存在这样的语义后门攻击，并在图分类的背景下提出了一种针对GCNS的语义后门攻击(SBAG)，以揭示GCNS中这一安全漏洞的存在。SBag使用样本中的某一类型节点作为后门触发器，并通过毒化训练数据向GCN模型注入隐藏的后门。后门将被激活，GCN模型将给出攻击者指定的恶意分类结果，即使是在未经修改的样本上，只要这些样本包含足够的触发节点。我们在四个图数据集上对SAGAG进行了评估，实验结果表明SABAG是有效的。



## **22. Active learning for fast and slow modeling attacks on Arbiter PUFs**

对仲裁器PUF进行快慢建模攻击的主动学习 cs.CR

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/2308.13645v1) [paper-pdf](http://arxiv.org/pdf/2308.13645v1)

**Authors**: Vincent Dumoulin, Wenjing Rao, Natasha Devroye

**Abstract**: Modeling attacks, in which an adversary uses machine learning techniques to model a hardware-based Physically Unclonable Function (PUF) pose a great threat to the viability of these hardware security primitives. In most modeling attacks, a random subset of challenge-response-pairs (CRPs) are used as the labeled data for the machine learning algorithm. Here, for the arbiter-PUF, a delay based PUF which may be viewed as a linear threshold function with random weights (due to manufacturing imperfections), we investigate the role of active learning in Support Vector Machine (SVM) learning. We focus on challenge selection to help SVM algorithm learn ``fast'' and learn ``slow''. Our methods construct challenges rather than relying on a sample pool of challenges as in prior work. Using active learning to learn ``fast'' (less CRPs revealed, higher accuracies) may help manufacturers learn the manufactured PUFs more efficiently, or may form a more powerful attack when the attacker may query the PUF for CRPs at will. Using active learning to select challenges from which learning is ``slow'' (low accuracy despite a large number of revealed CRPs) may provide a basis for slowing down attackers who are limited to overhearing CRPs.

摘要: 建模攻击是指攻击者使用机器学习技术对基于硬件的物理不可克隆函数(PUF)进行建模，这对这些硬件安全原语的生存能力构成了极大的威胁。在大多数建模攻击中，挑战-响应对(CRP)的随机子集被用作机器学习算法的标签数据。这里，对于仲裁器-PUF，一种基于延迟的PUF，可以被视为具有随机权值的线性阈值函数(由于制造缺陷)，我们研究了主动学习在支持向量机学习中的作用。我们把重点放在挑战选择上，帮助支持向量机算法学习“快”和“慢”。我们的方法构建挑战，而不是像以前的工作那样依赖于挑战的样本池。使用主动学习来学习“快速”(揭示的CRP越少，准确率越高)可以帮助制造商更有效地学习制造的PUF，或者当攻击者可以随意向PUF查询CRP时，可能形成更强大的攻击。使用主动学习来选择学习“缓慢”的挑战(尽管发现了大量CRP，但准确率很低)可能会为减缓仅限于无意中监听CRP的攻击者提供基础。



## **23. Unveiling the Role of Message Passing in Dual-Privacy Preservation on GNNs**

揭示消息传递在GNN双重隐私保护中的作用 cs.LG

CIKM 2023

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/2308.13513v1) [paper-pdf](http://arxiv.org/pdf/2308.13513v1)

**Authors**: Tianyi Zhao, Hui Hu, Lu Cheng

**Abstract**: Graph Neural Networks (GNNs) are powerful tools for learning representations on graphs, such as social networks. However, their vulnerability to privacy inference attacks restricts their practicality, especially in high-stake domains. To address this issue, privacy-preserving GNNs have been proposed, focusing on preserving node and/or link privacy. This work takes a step back and investigates how GNNs contribute to privacy leakage. Through theoretical analysis and simulations, we identify message passing under structural bias as the core component that allows GNNs to \textit{propagate} and \textit{amplify} privacy leakage. Building upon these findings, we propose a principled privacy-preserving GNN framework that effectively safeguards both node and link privacy, referred to as dual-privacy preservation. The framework comprises three major modules: a Sensitive Information Obfuscation Module that removes sensitive information from node embeddings, a Dynamic Structure Debiasing Module that dynamically corrects the structural bias, and an Adversarial Learning Module that optimizes the privacy-utility trade-off. Experimental results on four benchmark datasets validate the effectiveness of the proposed model in protecting both node and link privacy while preserving high utility for downstream tasks, such as node classification.

摘要: 图神经网络(GNN)是学习图上表示的强大工具，如社会网络。然而，它们对隐私推理攻击的脆弱性限制了它们的实用性，特别是在高风险领域。为了解决这一问题，人们提出了隐私保护的GNN，其重点是保护节点和/或链路的隐私。这项工作退了一步，调查了GNN是如何导致隐私泄露的。通过理论分析和仿真，我们发现结构偏差下的消息传递是允许GNN传播和放大隐私泄漏的核心组件。基于这些发现，我们提出了一个原则性的隐私保护GNN框架，该框架有效地保护了节点和链路的隐私，称为双重隐私保护。该框架包括三个主要模块：从节点嵌入中移除敏感信息的敏感信息混淆模块、动态纠正结构偏差的动态结构去偏模块和优化隐私-效用权衡的对抗性学习模块。在四个基准数据集上的实验结果验证了该模型的有效性，在保护节点和链路隐私的同时，保持了节点分类等下游任务的高效用。



## **24. Overcoming Adversarial Attacks for Human-in-the-Loop Applications**

克服针对人在环中应用的敌意攻击 cs.LG

New Frontiers in Adversarial Machine Learning, ICML 2022

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/2306.05952v2) [paper-pdf](http://arxiv.org/pdf/2306.05952v2)

**Authors**: Ryan McCoppin, Marla Kennedy, Platon Lukyanenko, Sean Kennedy

**Abstract**: Including human analysis has the potential to positively affect the robustness of Deep Neural Networks and is relatively unexplored in the Adversarial Machine Learning literature. Neural network visual explanation maps have been shown to be prone to adversarial attacks. Further research is needed in order to select robust visualizations of explanations for the image analyst to evaluate a given model. These factors greatly impact Human-In-The-Loop (HITL) evaluation tools due to their reliance on adversarial images, including explanation maps and measurements of robustness. We believe models of human visual attention may improve interpretability and robustness of human-machine imagery analysis systems. Our challenge remains, how can HITL evaluation be robust in this adversarial landscape?

摘要: 包括人类分析有可能对深度神经网络的稳健性产生积极影响，在对抗性机器学习文献中相对未被探索。神经网络视觉解释地图已被证明容易受到敌意攻击。还需要进一步的研究，以便为图像分析员选择稳健的解释可视化来评估给定的模型。这些因素极大地影响了人在环(HITL)评估工具，因为它们依赖于敌方图像，包括解释地图和健壮性测量。我们相信人类视觉注意模型可以提高人机图像分析系统的可解释性和稳健性。我们的挑战仍然是，如何在这种对抗性的环境中进行HITL评估？



## **25. Defensive Few-shot Learning**

防御性少投篮学习 cs.CV

Accepted to IEEE Transactions on Pattern Analysis and Machine  Intelligence (TPAMI) 2022

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/1911.06968v2) [paper-pdf](http://arxiv.org/pdf/1911.06968v2)

**Authors**: Wenbin Li, Lei Wang, Xingxing Zhang, Lei Qi, Jing Huo, Yang Gao, Jiebo Luo

**Abstract**: This paper investigates a new challenging problem called defensive few-shot learning in order to learn a robust few-shot model against adversarial attacks. Simply applying the existing adversarial defense methods to few-shot learning cannot effectively solve this problem. This is because the commonly assumed sample-level distribution consistency between the training and test sets can no longer be met in the few-shot setting. To address this situation, we develop a general defensive few-shot learning (DFSL) framework to answer the following two key questions: (1) how to transfer adversarial defense knowledge from one sample distribution to another? (2) how to narrow the distribution gap between clean and adversarial examples under the few-shot setting? To answer the first question, we propose an episode-based adversarial training mechanism by assuming a task-level distribution consistency to better transfer the adversarial defense knowledge. As for the second question, within each few-shot task, we design two kinds of distribution consistency criteria to narrow the distribution gap between clean and adversarial examples from the feature-wise and prediction-wise perspectives, respectively. Extensive experiments demonstrate that the proposed framework can effectively make the existing few-shot models robust against adversarial attacks. Code is available at https://github.com/WenbinLee/DefensiveFSL.git.

摘要: 本文研究了一个新的具有挑战性的问题--防御性少发学习问题，目的是学习一种对敌方攻击具有鲁棒性的少发学习模型。简单地将现有的对抗性防御方法应用于少射击学习并不能有效地解决这一问题。这是因为通常假设的训练集和测试集之间的样本水平分布一致性在少镜头设置中不再能满足。针对这种情况，我们开发了一个通用防御少发学习(DFSL)框架来回答以下两个关键问题：(1)如何将对抗性防御知识从一个样本分布转移到另一个样本分布？(2)在少发情况下，如何缩小干净样本和对抗性样本之间的分布差距？为了回答第一个问题，我们提出了一种基于情节的对抗性训练机制，通过假设任务级别的分布一致性来更好地传递对抗性防御知识。对于第二个问题，在每个少镜头任务中，我们设计了两种分布一致性准则，分别从特征和预测的角度缩小了正例和对手例之间的分布差距。大量实验表明，该框架能有效地使已有的少镜头模型具有较强的抗敌意攻击能力。代码可在https://github.com/WenbinLee/DefensiveFSL.git.上找到



## **26. Feature Unlearning for Pre-trained GANs and VAEs**

针对经过预先培训的GAN和VAE的功能遗忘 cs.CV

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/2303.05699v3) [paper-pdf](http://arxiv.org/pdf/2303.05699v3)

**Authors**: Saemi Moon, Seunghyuk Cho, Dongwoo Kim

**Abstract**: We tackle the problem of feature unlearning from a pre-trained image generative model: GANs and VAEs. Unlike a common unlearning task where an unlearning target is a subset of the training set, we aim to unlearn a specific feature, such as hairstyle from facial images, from the pre-trained generative models. As the target feature is only presented in a local region of an image, unlearning the entire image from the pre-trained model may result in losing other details in the remaining region of the image. To specify which features to unlearn, we collect randomly generated images that contain the target features. We then identify a latent representation corresponding to the target feature and then use the representation to fine-tune the pre-trained model. Through experiments on MNIST and CelebA datasets, we show that target features are successfully removed while keeping the fidelity of the original models. Further experiments with an adversarial attack show that the unlearned model is more robust under the presence of malicious parties.

摘要: 我们从一个预先训练的图像生成模型GANS和VAE中解决了特征遗忘的问题。与通常的遗忘任务不同，忘记目标是训练集的一个子集，我们的目标是从预先训练的生成模型中忘记特定的特征，如面部图像中的发型。由于目标特征仅呈现在图像的局部区域中，因此从预先训练的模型中不学习整个图像可能导致丢失图像剩余区域中的其他细节。为了指定要取消学习的特征，我们收集包含目标特征的随机生成的图像。然后，我们识别对应于目标特征的潜在表示，然后使用该表示来微调预先训练的模型。通过在MNIST和CelebA数据集上的实验，我们证明了在保持原始模型保真度的情况下，目标特征被成功去除。进一步的对抗性攻击实验表明，未学习模型在恶意方存在的情况下具有更强的鲁棒性。



## **27. Why Does Little Robustness Help? Understanding and Improving Adversarial Transferability from Surrogate Training**

为什么小健壮性会有帮助？从替补训练中认识和提高对手的转换性 cs.LG

Accepted by IEEE Symposium on Security and Privacy (Oakland) 2024; 21  pages, 11 figures, 13 tables

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/2307.07873v4) [paper-pdf](http://arxiv.org/pdf/2307.07873v4)

**Authors**: Yechao Zhang, Shengshan Hu, Leo Yu Zhang, Junyu Shi, Minghui Li, Xiaogeng Liu, Wei Wan, Hai Jin

**Abstract**: Adversarial examples (AEs) for DNNs have been shown to be transferable: AEs that successfully fool white-box surrogate models can also deceive other black-box models with different architectures. Although a bunch of empirical studies have provided guidance on generating highly transferable AEs, many of these findings lack explanations and even lead to inconsistent advice. In this paper, we take a further step towards understanding adversarial transferability, with a particular focus on surrogate aspects. Starting from the intriguing little robustness phenomenon, where models adversarially trained with mildly perturbed adversarial samples can serve as better surrogates, we attribute it to a trade-off between two predominant factors: model smoothness and gradient similarity. Our investigations focus on their joint effects, rather than their separate correlations with transferability. Through a series of theoretical and empirical analyses, we conjecture that the data distribution shift in adversarial training explains the degradation of gradient similarity. Building on these insights, we explore the impacts of data augmentation and gradient regularization on transferability and identify that the trade-off generally exists in the various training mechanisms, thus building a comprehensive blueprint for the regulation mechanism behind transferability. Finally, we provide a general route for constructing better surrogates to boost transferability which optimizes both model smoothness and gradient similarity simultaneously, e.g., the combination of input gradient regularization and sharpness-aware minimization (SAM), validated by extensive experiments. In summary, we call for attention to the united impacts of these two factors for launching effective transfer attacks, rather than optimizing one while ignoring the other, and emphasize the crucial role of manipulating surrogate models.

摘要: DNN的对抗性例子(AE)已被证明是可移植的：成功欺骗白盒代理模型的AES也可以欺骗其他具有不同体系结构的黑盒模型。尽管大量的实证研究为生成高度可转移的企业实体提供了指导，但其中许多发现缺乏解释，甚至导致了不一致的建议。在这篇文章中，我们进一步了解对手的可转移性，特别关注代理方面。从有趣的小鲁棒性现象开始，我们将其归因于两个主要因素之间的权衡：模型的光滑性和梯度相似性。我们的研究重点是它们的联合影响，而不是它们与可转让性的单独关联。通过一系列的理论和实证分析，我们推测对抗性训练中的数据分布转移解释了梯度相似度的下降。基于这些见解，我们探讨了数据扩充和梯度规则化对可转移性的影响，并确定了各种培训机制中普遍存在的权衡，从而构建了可转移性背后的监管机制的全面蓝图。最后，我们提供了一条同时优化模型光滑性和梯度相似性的构造更好的代理以提高可转移性的一般路线，例如输入梯度正则化和锐度感知最小化(SAM)的组合，并通过大量的实验进行了验证。总之，我们呼吁注意这两个因素对发动有效转移攻击的联合影响，而不是优化一个而忽略另一个，并强调操纵代理模型的关键作用。



## **28. Face Encryption via Frequency-Restricted Identity-Agnostic Attacks**

通过频率受限的身份不可知攻击进行人脸加密 cs.CV

I noticed something missing in the article's description in  subsection 3.2, so I'd like to undo it and re-finalize and describe it

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/2308.05983v3) [paper-pdf](http://arxiv.org/pdf/2308.05983v3)

**Authors**: Xin Dong, Rui Wang, Siyuan Liang, Aishan Liu, Lihua Jing

**Abstract**: Billions of people are sharing their daily live images on social media everyday. However, malicious collectors use deep face recognition systems to easily steal their biometric information (e.g., faces) from these images. Some studies are being conducted to generate encrypted face photos using adversarial attacks by introducing imperceptible perturbations to reduce face information leakage. However, existing studies need stronger black-box scenario feasibility and more natural visual appearances, which challenge the feasibility of privacy protection. To address these problems, we propose a frequency-restricted identity-agnostic (FRIA) framework to encrypt face images from unauthorized face recognition without access to personal information. As for the weak black-box scenario feasibility, we obverse that representations of the average feature in multiple face recognition models are similar, thus we propose to utilize the average feature via the crawled dataset from the Internet as the target to guide the generation, which is also agnostic to identities of unknown face recognition systems; in nature, the low-frequency perturbations are more visually perceptible by the human vision system. Inspired by this, we restrict the perturbation in the low-frequency facial regions by discrete cosine transform to achieve the visual naturalness guarantee. Extensive experiments on several face recognition models demonstrate that our FRIA outperforms other state-of-the-art methods in generating more natural encrypted faces while attaining high black-box attack success rates of 96%. In addition, we validate the efficacy of FRIA using real-world black-box commercial API, which reveals the potential of FRIA in practice. Our codes can be found in https://github.com/XinDong10/FRIA.

摘要: 每天都有数十亿人在社交媒体上分享他们的日常直播图片。然而，恶意收集者使用深度人脸识别系统来轻松地从这些图像中窃取他们的生物特征信息(例如，人脸)。正在进行一些研究，通过引入不可察觉的扰动来减少面部信息的泄露，从而使用对抗性攻击来生成加密的面部照片。然而，现有的研究需要更强的黑盒场景可行性和更自然的视觉外观，这对隐私保护的可行性提出了挑战。为了解决这些问题，我们提出了一种频率受限身份不可知(FRIA)框架来加密来自未经授权的人脸识别的人脸图像，而不需要访问个人信息。对于弱黑盒场景的可行性，我们发现在多个人脸识别模型中平均特征的表示是相似的，因此我们提出利用从互联网上抓取的数据集的平均特征作为目标来指导生成，这也与未知人脸识别系统的身份无关；实际上，低频扰动更容易被人类视觉系统感知。受此启发，我们通过离散余弦变换来限制人脸低频区域的扰动，以达到视觉自然度的保证。在几个人脸识别模型上的广泛实验表明，我们的FRIA在生成更自然的加密人脸方面优于其他最先进的方法，同时获得了96%的高黑盒攻击成功率。此外，我们使用真实的黑盒商业API验证了FRIA的有效性，这揭示了FRIA在实践中的潜力。我们的代码可以在https://github.com/XinDong10/FRIA.中找到



## **29. Evaluating the Vulnerabilities in ML systems in terms of adversarial attacks**

从对抗性攻击的角度评估ML系统的脆弱性 cs.LG

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2308.12918v1) [paper-pdf](http://arxiv.org/pdf/2308.12918v1)

**Authors**: John Harshith, Mantej Singh Gill, Madhan Jothimani

**Abstract**: There have been recent adversarial attacks that are difficult to find. These new adversarial attacks methods may pose challenges to current deep learning cyber defense systems and could influence the future defense of cyberattacks. The authors focus on this domain in this research paper. They explore the consequences of vulnerabilities in AI systems. This includes discussing how they might arise, differences between randomized and adversarial examples and also potential ethical implications of vulnerabilities. Moreover, it is important to train the AI systems appropriately when they are in testing phase and getting them ready for broader use.

摘要: 最近发生了一些很难找到的对抗性攻击。这些新的对抗性攻击方法可能会对当前的深度学习网络防御系统构成挑战，并可能影响未来对网络攻击的防御。在这篇研究论文中，作者主要关注这一领域。他们探索了人工智能系统中漏洞的后果。这包括讨论它们可能出现的原因、随机例子和对抗性例子之间的差异，以及漏洞的潜在伦理影响。此外，重要的是在AI系统处于测试阶段时对其进行适当的培训，并使其为更广泛的使用做好准备。



## **30. Near Optimal Adversarial Attack on UCB Bandits**

对UCB土匪的近最优敌意攻击 cs.LG

Appeared at ICML 2023 AdvML Workshop

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2008.09312v6) [paper-pdf](http://arxiv.org/pdf/2008.09312v6)

**Authors**: Shiliang Zuo

**Abstract**: I study a stochastic multi-arm bandit problem where rewards are subject to adversarial corruption. I propose a novel attack strategy that manipulates a learner employing the UCB algorithm into pulling some non-optimal target arm $T - o(T)$ times with a cumulative cost that scales as $\widehat{O}(\sqrt{\log T})$, where $T$ is the number of rounds. I also prove the first lower bound on the cumulative attack cost. The lower bound matches the upper bound up to $O(\log \log T)$ factors, showing the proposed attack strategy to be near optimal.

摘要: 我研究了一个随机多臂强盗问题，其中报酬受到对抗性腐败的影响。提出了一种新的攻击策略，该策略利用UCB算法操纵学习者拉出一些非最优目标臂$T-o(T)$次，累积代价为$\widehat{O}(\Sqrt{\log T})$，其中$T$是轮数。我还证明了累积攻击成本的第一个下限。下界与最高可达$O(\LOG\LOG T)$因子的上界匹配，表明所提出的攻击策略接近最优。



## **31. Fast Adversarial Training with Smooth Convergence**

平滑收敛的快速对抗性训练 cs.LG

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2308.12857v1) [paper-pdf](http://arxiv.org/pdf/2308.12857v1)

**Authors**: Mengnan Zhao, Lihe Zhang, Yuqiu Kong, Baocai Yin

**Abstract**: Fast adversarial training (FAT) is beneficial for improving the adversarial robustness of neural networks. However, previous FAT work has encountered a significant issue known as catastrophic overfitting when dealing with large perturbation budgets, \ie the adversarial robustness of models declines to near zero during training.   To address this, we analyze the training process of prior FAT work and observe that catastrophic overfitting is accompanied by the appearance of loss convergence outliers.   Therefore, we argue a moderately smooth loss convergence process will be a stable FAT process that solves catastrophic overfitting.   To obtain a smooth loss convergence process, we propose a novel oscillatory constraint (dubbed ConvergeSmooth) to limit the loss difference between adjacent epochs. The convergence stride of ConvergeSmooth is introduced to balance convergence and smoothing. Likewise, we design weight centralization without introducing additional hyperparameters other than the loss balance coefficient.   Our proposed methods are attack-agnostic and thus can improve the training stability of various FAT techniques.   Extensive experiments on popular datasets show that the proposed methods efficiently avoid catastrophic overfitting and outperform all previous FAT methods. Code is available at \url{https://github.com/FAT-CS/ConvergeSmooth}.

摘要: 快速对抗训练(FAT)有利于提高神经网络的对抗健壮性。然而，以前的FAT工作在处理大扰动预算时遇到了一个重要的问题，即模型的对抗性健壮性在训练期间下降到接近于零。为了解决这个问题，我们分析了以前FAT工作的训练过程，观察到灾难性的过拟合伴随着损失收敛离群点的出现。因此，我们认为，适度平滑的损失收敛过程将是一个稳定的脂肪过程，可以解决灾难性的过拟合问题。为了获得平滑的损耗收敛过程，我们提出了一种新的振荡约束(称为收敛平滑)来限制相邻历元之间的损耗差异。为了平衡收敛和光顺，引入了收敛步长。同样，我们设计了权重集中，而不引入除损失平衡系数之外的额外超参数。我们提出的方法是攻击不可知的，因此可以提高各种FAT技术的训练稳定性。在流行的数据集上的大量实验表明，所提出的方法有效地避免了灾难性的过拟合，并且优于所有已有的FAT方法。代码可在\url{https://github.com/FAT-CS/ConvergeSmooth}.上找到



## **32. Unifying Gradients to Improve Real-world Robustness for Deep Networks**

统一梯度以提高深度网络的真实稳健性 stat.ML

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2208.06228v2) [paper-pdf](http://arxiv.org/pdf/2208.06228v2)

**Authors**: Yingwen Wu, Sizhe Chen, Kun Fang, Xiaolin Huang

**Abstract**: The wide application of deep neural networks (DNNs) demands an increasing amount of attention to their real-world robustness, i.e., whether a DNN resists black-box adversarial attacks, among which score-based query attacks (SQAs) are most threatening since they can effectively hurt a victim network with the only access to model outputs. Defending against SQAs requires a slight but artful variation of outputs due to the service purpose for users, who share the same output information with SQAs. In this paper, we propose a real-world defense by Unifying Gradients (UniG) of different data so that SQAs could only probe a much weaker attack direction that is similar for different samples. Since such universal attack perturbations have been validated as less aggressive than the input-specific perturbations, UniG protects real-world DNNs by indicating attackers a twisted and less informative attack direction. We implement UniG efficiently by a Hadamard product module which is plug-and-play. According to extensive experiments on 5 SQAs, 2 adaptive attacks and 7 defense baselines, UniG significantly improves real-world robustness without hurting clean accuracy on CIFAR10 and ImageNet. For instance, UniG maintains a model of 77.80% accuracy under 2500-query Square attack while the state-of-the-art adversarially-trained model only has 67.34% on CIFAR10. Simultaneously, UniG outperforms all compared baselines in terms of clean accuracy and achieves the smallest modification of the model output. The code is released at https://github.com/snowien/UniG-pytorch.

摘要: 深度神经网络(DNN)的广泛应用要求人们越来越多地关注它在现实世界中的健壮性，即DNN是否能够抵抗黑盒对抗攻击，其中基于分数的查询攻击(SQA)是最具威胁性的，因为它们可以有效地伤害只访问模型输出的受害者网络。由于用户的服务目的，防御SBA需要稍微但巧妙地改变输出，因为用户与SBA共享相同的输出信息。在本文中，我们提出了一种真实世界的防御方法，通过统一不同数据的梯度(Ung)，使得SQAS只能探测对不同样本相似的弱得多的攻击方向。由于这种通用的攻击扰动已被验证为不如特定于输入的扰动那么具侵略性，unig通过向攻击者指示一个扭曲的、信息量较少的攻击方向来保护现实世界的DNN。我们通过一个即插即用的Hadamard产品模块高效地实现了unig。根据对5个SQA、2个自适应攻击和7个防御基线的广泛实验，unig在不损害CIFAR10和ImageNet上的干净准确性的情况下，显著提高了真实世界的健壮性。例如，在2500-Query Square攻击下，unig保持了77.80%的准确率，而最新的对抗性训练模型在CIFAR10上只有67.34%的准确率。同时，UNIG在清洁精度方面优于所有比较的基线，并实现了对模型输出的最小修改。该代码在https://github.com/snowien/UniG-pytorch.上发布



## **33. Universal Soldier: Using Universal Adversarial Perturbations for Detecting Backdoor Attacks**

Universal Soldier：使用通用对抗性扰动检测后门攻击 cs.LG

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2302.00747v3) [paper-pdf](http://arxiv.org/pdf/2302.00747v3)

**Authors**: Xiaoyun Xu, Oguzhan Ersoy, Stjepan Picek

**Abstract**: Deep learning models achieve excellent performance in numerous machine learning tasks. Yet, they suffer from security-related issues such as adversarial examples and poisoning (backdoor) attacks. A deep learning model may be poisoned by training with backdoored data or by modifying inner network parameters. Then, a backdoored model performs as expected when receiving a clean input, but it misclassifies when receiving a backdoored input stamped with a pre-designed pattern called "trigger". Unfortunately, it is difficult to distinguish between clean and backdoored models without prior knowledge of the trigger. This paper proposes a backdoor detection method by utilizing a special type of adversarial attack, universal adversarial perturbation (UAP), and its similarities with a backdoor trigger. We observe an intuitive phenomenon: UAPs generated from backdoored models need fewer perturbations to mislead the model than UAPs from clean models. UAPs of backdoored models tend to exploit the shortcut from all classes to the target class, built by the backdoor trigger. We propose a novel method called Universal Soldier for Backdoor detection (USB) and reverse engineering potential backdoor triggers via UAPs. Experiments on 345 models trained on several datasets show that USB effectively detects the injected backdoor and provides comparable or better results than state-of-the-art methods.

摘要: 深度学习模型在众多的机器学习任务中取得了优异的性能。然而，他们面临着与安全相关的问题，如对抗性例子和中毒(后门)攻击。深度学习模型可能会因使用后备数据进行训练或通过修改内部网络参数而中毒。然后，当接收到干净的输入时，回溯模型执行预期的操作，但是当接收到带有预先设计的称为“Trigger”模式的回溯输入时，它就会错误地分类。不幸的是，在没有事先了解触发因素的情况下，很难区分干净和落后的模型。利用一种特殊类型的对抗性攻击--通用对抗性扰动(UAP)及其与后门触发器的相似性，提出了一种后门检测方法。我们观察到一个直观的现象：由回溯模型生成的UAP比从干净模型生成的UAP需要更少的扰动来误导模型。后门模型的UAP倾向于利用由后门触发器构建的从所有类到目标类的快捷方式。我们提出了一种新的方法，称为通用士兵(Universal Soldier)，用于后门检测(USB)和通过UAP反向工程潜在的后门触发器。在几个数据集上训练的345个模型上的实验表明，USB有效地检测注入的后门，并提供与最先进的方法相当或更好的结果。



## **34. Don't Look into the Sun: Adversarial Solarization Attacks on Image Classifiers**

不要看太阳：对图像分类器的对抗性日晒攻击 cs.CV

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2308.12661v1) [paper-pdf](http://arxiv.org/pdf/2308.12661v1)

**Authors**: Paul Gavrikov, Janis Keuper

**Abstract**: Assessing the robustness of deep neural networks against out-of-distribution inputs is crucial, especially in safety-critical domains like autonomous driving, but also in safety systems where malicious actors can digitally alter inputs to circumvent safety guards. However, designing effective out-of-distribution tests that encompass all possible scenarios while preserving accurate label information is a challenging task. Existing methodologies often entail a compromise between variety and constraint levels for attacks and sometimes even both. In a first step towards a more holistic robustness evaluation of image classification models, we introduce an attack method based on image solarization that is conceptually straightforward yet avoids jeopardizing the global structure of natural images independent of the intensity. Through comprehensive evaluations of multiple ImageNet models, we demonstrate the attack's capacity to degrade accuracy significantly, provided it is not integrated into the training augmentations. Interestingly, even then, no full immunity to accuracy deterioration is achieved. In other settings, the attack can often be simplified into a black-box attack with model-independent parameters. Defenses against other corruptions do not consistently extend to be effective against our specific attack.   Project website: https://github.com/paulgavrikov/adversarial_solarization

摘要: 评估深度神经网络对非分布输入的稳健性至关重要，特别是在自动驾驶等安全关键领域，但在安全系统中也是如此，在安全系统中，恶意行为者可以通过数字更改输入以绕过安全警卫。然而，设计有效的分发外测试以涵盖所有可能的场景，同时保留准确的标签信息是一项具有挑战性的任务。现有的方法往往需要在攻击的多样性和约束级别之间达成妥协，有时甚至两者兼而有之。在对图像分类模型进行更全面的稳健性评估的第一步中，我们引入了一种基于图像日晒的攻击方法，该方法在概念上简单明了，但避免了与强度无关的自然图像的全局结构。通过对多个ImageNet模型的综合评估，我们证明了攻击的能力显著降低了准确性，前提是它没有整合到训练增强中。有趣的是，即使到那时，也没有实现对精度恶化的完全免疫力。在其他设置中，攻击通常可以简化为具有与模型无关的参数的黑盒攻击。对其他腐败的防御并不总是延伸到对我们的特定攻击有效。项目网站：https://github.com/paulgavrikov/adversarial_solarization



## **35. Exploring Transferability of Multimodal Adversarial Samples for Vision-Language Pre-training Models with Contrastive Learning**

基于对比学习的视觉语言预训练模型多通道对抗性样本可转换性研究 cs.MM

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2308.12636v1) [paper-pdf](http://arxiv.org/pdf/2308.12636v1)

**Authors**: Youze Wang, Wenbo Hu, Yinpeng Dong, Richang Hong

**Abstract**: Vision-language pre-training models (VLP) are vulnerable, especially to multimodal adversarial samples, which can be crafted by adding imperceptible perturbations on both original images and texts. However, under the black-box setting, there have been no works to explore the transferability of multimodal adversarial attacks against the VLP models. In this work, we take CLIP as the surrogate model and propose a gradient-based multimodal attack method to generate transferable adversarial samples against the VLP models. By applying the gradient to optimize the adversarial images and adversarial texts simultaneously, our method can better search for and attack the vulnerable images and text information pairs. To improve the transferability of the attack, we utilize contrastive learning including image-text contrastive learning and intra-modal contrastive learning to have a more generalized understanding of the underlying data distribution and mitigate the overfitting of the surrogate model so that the generated multimodal adversarial samples have a higher transferability for VLP models. Extensive experiments validate the effectiveness of the proposed method.

摘要: 视觉语言预训练模型(VLP)是脆弱的，特别是对多模式对抗性样本，这些样本可以通过在原始图像和文本上添加不可察觉的扰动来构建。然而，在黑箱环境下，还没有研究针对VLP模型的多通道对抗性攻击的可转移性。在本文中，我们以CLIP为代理模型，提出了一种基于梯度的多模式攻击方法来生成可传输的对抗VLP模型的样本。通过应用梯度同时优化敌意图像和敌意文本，我们的方法可以更好地搜索和攻击易受攻击的图文信息对。为了提高攻击的可转移性，我们利用对比学习，包括图文对比学习和通道内对比学习，以更全面地了解底层数据分布，并减少代理模型的过度拟合，使生成的多通道对抗性样本对VLP模型具有更高的可转移性。大量实验验证了该方法的有效性。



## **36. PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts**

PromptBitch：评估大型语言模型在对抗性提示下的稳健性 cs.CL

Technical report; updated with new experiments and related work; 27  pages; code is at: https://github.com/microsoft/promptbench

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2306.04528v3) [paper-pdf](http://arxiv.org/pdf/2306.04528v3)

**Authors**: Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Neil Zhenqiang Gong, Yue Zhang, Xing Xie

**Abstract**: The increasing reliance on Large Language Models (LLMs) across academia and industry necessitates a comprehensive understanding of their robustness to prompts. In response to this vital need, we introduce PromptBench, a robustness benchmark designed to measure LLMs' resilience to adversarial prompts. This study uses a plethora of adversarial textual attacks targeting prompts across multiple levels: character, word, sentence, and semantic. These prompts are then employed in diverse tasks, such as sentiment analysis, natural language inference, reading comprehension, machine translation, and math problem-solving. Our study generates 4,032 adversarial prompts, meticulously evaluated over 8 tasks and 13 datasets, with 567,084 test samples in total. Our findings demonstrate that contemporary LLMs are vulnerable to adversarial prompts. Furthermore, we present comprehensive analysis to understand the mystery behind prompt robustness and its transferability. We then offer insightful robustness analysis and pragmatic recommendations for prompt composition, beneficial to both researchers and everyday users. We make our code, prompts, and methodologies to generate adversarial prompts publicly accessible, thereby enabling and encouraging collaborative exploration in this pivotal field: https://github.com/microsoft/promptbench.

摘要: 学术界和工业界对大型语言模型(LLM)的依赖日益增加，这就要求我们必须全面了解它们对提示的稳健性。为了响应这一关键需求，我们引入了PromptBtch，这是一个健壮性基准，旨在衡量LLMS对敌意提示的弹性。这项研究使用了过多的对抗性文本攻击，目标是多个层面的提示：字符、单词、句子和语义。这些提示随后被用于不同的任务，如情感分析、自然语言推理、阅读理解、机器翻译和数学解题。我们的研究产生了4032个对抗性提示，仔细评估了8个任务和13个数据集，总共有567,084个测试样本。我们的研究结果表明，当代的LLM容易受到对抗性提示的影响。此外，我们还提供了全面的分析，以了解即时健壮性及其可转移性背后的奥秘。然后，我们提供了有洞察力的健壮性分析和实用的即时撰写建议，这对研究人员和日常用户都是有益的。我们将生成对抗性提示的代码、提示和方法公之于众，从而支持并鼓励在这个关键领域进行协作探索：https://github.com/microsoft/promptbench.



## **37. Towards an Accurate and Secure Detector against Adversarial Perturbations**

走向准确和安全的检测器以抵御对手的扰动 cs.CV

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2305.10856v2) [paper-pdf](http://arxiv.org/pdf/2305.10856v2)

**Authors**: Chao Wang, Shuren Qi, Zhiqiu Huang, Yushu Zhang, Rushi Lan, Xiaochun Cao

**Abstract**: The vulnerability of deep neural networks to adversarial perturbations has been widely perceived in the computer vision community. From a security perspective, it poses a critical risk for modern vision systems, e.g., the popular Deep Learning as a Service (DLaaS) frameworks. For protecting off-the-shelf deep models while not modifying them, current algorithms typically detect adversarial patterns through discriminative decomposition of natural-artificial data. However, these decompositions are biased towards frequency or spatial discriminability, thus failing to capture adversarial patterns comprehensively. More seriously, successful defense-aware (secondary) adversarial attack (i.e., evading the detector as well as fooling the model) is practical under the assumption that the adversary is fully aware of the detector (i.e., the Kerckhoffs's principle). Motivated by such facts, we propose an accurate and secure adversarial example detector, relying on a spatial-frequency discriminative decomposition with secret keys. It expands the above works on two aspects: 1) the introduced Krawtchouk basis provides better spatial-frequency discriminability and thereby is more suitable for capturing adversarial patterns than the common trigonometric or wavelet basis; 2) the extensive parameters for decomposition are generated by a pseudo-random function with secret keys, hence blocking the defense-aware adversarial attack. Theoretical and numerical analysis demonstrates the increased accuracy and security of our detector with respect to a number of state-of-the-art algorithms.

摘要: 深度神经网络对对抗性扰动的脆弱性已经在计算机视觉领域得到了广泛的认识。从安全的角度来看，它对现代视觉系统构成了严重的风险，例如流行的深度学习即服务(DLaaS)框架。为了在不修改现有深度模型的同时保护它们，目前的算法通常通过对自然-人工数据的区别性分解来检测对抗性模式。然而，这些分解偏向于频率或空间可区分性，因此无法全面地捕捉对抗性模式。更严重的是，在假设对手完全知道检测器(即Kerckhoff原理)的情况下，成功的防御感知(次要)对手攻击(即，躲避检测器以及愚弄模型)是实用的。在此基础上，提出了一种基于密钥的空频判别分解的准确、安全的对抗性样本检测器。它从两个方面对上述工作进行了扩展：1)引入的Krawtchouk基提供了更好的空频分辨能力，因此比普通的三角或小波基更适合于捕获敌意模式；2)分解的广泛参数是由带有密钥的伪随机函数产生的，从而阻止了具有防御意识的敌意攻击。理论和数值分析表明，相对于一些最先进的算法，我们的检测器具有更高的准确性和安全性。



## **38. A Huber Loss Minimization Approach to Byzantine Robust Federated Learning**

拜占庭稳健联邦学习的Huber损失最小化方法 cs.LG

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2308.12581v1) [paper-pdf](http://arxiv.org/pdf/2308.12581v1)

**Authors**: Puning Zhao, Fei Yu, Zhiguo Wan

**Abstract**: Federated learning systems are susceptible to adversarial attacks. To combat this, we introduce a novel aggregator based on Huber loss minimization, and provide a comprehensive theoretical analysis. Under independent and identically distributed (i.i.d) assumption, our approach has several advantages compared to existing methods. Firstly, it has optimal dependence on $\epsilon$, which stands for the ratio of attacked clients. Secondly, our approach does not need precise knowledge of $\epsilon$. Thirdly, it allows different clients to have unequal data sizes. We then broaden our analysis to include non-i.i.d data, such that clients have slightly different distributions.

摘要: 联合学习系统很容易受到对手的攻击。针对这一问题，我们提出了一种基于Huber损失最小化的聚合器，并对其进行了全面的理论分析。在独立同分布(I.I.D)假设下，我们的方法与现有方法相比有几个优点。首先，它对代表被攻击客户比率的$\epsilon$具有最优依赖。其次，我们的方法不需要$\epsilon$的精确知识。第三，它允许不同的客户端具有不同的数据大小。然后，我们扩大分析范围，将非I.I.D数据包括在内，这样客户的分布就略有不同。



## **39. Adversarial Training Using Feedback Loops**

使用反馈环进行对抗性训练 cs.LG

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2308.11881v2) [paper-pdf](http://arxiv.org/pdf/2308.11881v2)

**Authors**: Ali Haisam Muhammad Rafid, Adrian Sandu

**Abstract**: Deep neural networks (DNN) have found wide applicability in numerous fields due to their ability to accurately learn very complex input-output relations. Despite their accuracy and extensive use, DNNs are highly susceptible to adversarial attacks due to limited generalizability. For future progress in the field, it is essential to build DNNs that are robust to any kind of perturbations to the data points. In the past, many techniques have been proposed to robustify DNNs using first-order derivative information of the network.   This paper proposes a new robustification approach based on control theory. A neural network architecture that incorporates feedback control, named Feedback Neural Networks, is proposed. The controller is itself a neural network, which is trained using regular and adversarial data such as to stabilize the system outputs. The novel adversarial training approach based on the feedback control architecture is called Feedback Looped Adversarial Training (FLAT). Numerical results on standard test problems empirically show that our FLAT method is more effective than the state-of-the-art to guard against adversarial attacks.

摘要: 深度神经网络(DNN)因其能够准确学习非常复杂的输入输出关系而在许多领域得到了广泛的应用。尽管DNN具有较高的准确性和广泛的用途，但由于泛化能力有限，DNN极易受到敌意攻击。为了在该领域取得未来的进展，建立对数据点的任何类型的扰动都是健壮的DNN是至关重要的。在过去，已经提出了许多技术来利用网络的一阶导数信息来使DNN具有健壮性。本文基于控制理论提出了一种新的鲁棒性控制方法。提出了一种包含反馈控制的神经网络体系结构--反馈神经网络。控制器本身是一个神经网络，它使用常规和对抗性数据进行训练，以稳定系统输出。基于反馈控制结构的新型对抗训练方法称为反馈环对抗训练(Flat)。在标准测试问题上的数值结果经验表明，我们的Flat方法比最新的方法更有效地防御对抗性攻击。



## **40. BadVFL: Backdoor Attacks in Vertical Federated Learning**

BadVFL：垂直联合学习中的后门攻击 cs.LG

Accepted for publication at the 45th IEEE Symposium on Security &  Privacy (S&P 2024). Please cite accordingly

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2304.08847v2) [paper-pdf](http://arxiv.org/pdf/2304.08847v2)

**Authors**: Mohammad Naseri, Yufei Han, Emiliano De Cristofaro

**Abstract**: Federated learning (FL) enables multiple parties to collaboratively train a machine learning model without sharing their data; rather, they train their own model locally and send updates to a central server for aggregation. Depending on how the data is distributed among the participants, FL can be classified into Horizontal (HFL) and Vertical (VFL). In VFL, the participants share the same set of training instances but only host a different and non-overlapping subset of the whole feature space. Whereas in HFL, each participant shares the same set of features while the training set is split into locally owned training data subsets.   VFL is increasingly used in applications like financial fraud detection; nonetheless, very little work has analyzed its security. In this paper, we focus on robustness in VFL, in particular, on backdoor attacks, whereby an adversary attempts to manipulate the aggregate model during the training process to trigger misclassifications. Performing backdoor attacks in VFL is more challenging than in HFL, as the adversary i) does not have access to the labels during training and ii) cannot change the labels as she only has access to the feature embeddings. We present a first-of-its-kind clean-label backdoor attack in VFL, which consists of two phases: a label inference and a backdoor phase. We demonstrate the effectiveness of the attack on three different datasets, investigate the factors involved in its success, and discuss countermeasures to mitigate its impact.

摘要: 联合学习(FL)使多方能够协作地训练机器学习模型，而不共享他们的数据；相反，他们在本地训练他们自己的模型，并将更新发送到中央服务器以进行聚合。根据数据在参与者之间的分布情况，外语可分为水平(HFL)和垂直(VFL)。在VFL中，参与者共享相同的训练实例集，但仅托管整个特征空间的不同且不重叠的子集。而在HFL中，每个参与者共享相同的特征集，而训练集被分成本地拥有的训练数据子集。VFL越来越多地被用于金融欺诈检测等应用中；然而，很少有工作分析它的安全性。在本文中，我们关注VFL的稳健性，特别是后门攻击，即对手试图在训练过程中操纵聚合模型以触发错误分类。在VFL中执行后门攻击比在HFL中更具挑战性，因为对手i)在训练期间无法访问标签，ii)无法更改标签，因为她只能访问特征嵌入。在VFL中，我们提出了一种首次的干净标签后门攻击，它包括两个阶段：标签推理和后门阶段。我们在三个不同的数据集上演示了攻击的有效性，调查了其成功的相关因素，并讨论了减轻其影响的对策。



## **41. BaDExpert: Extracting Backdoor Functionality for Accurate Backdoor Input Detection**

BaDExpert：提取后门功能以进行准确的后门输入检测 cs.CR

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.12439v1) [paper-pdf](http://arxiv.org/pdf/2308.12439v1)

**Authors**: Tinghao Xie, Xiangyu Qi, Ping He, Yiming Li, Jiachen T. Wang, Prateek Mittal

**Abstract**: We present a novel defense, against backdoor attacks on Deep Neural Networks (DNNs), wherein adversaries covertly implant malicious behaviors (backdoors) into DNNs. Our defense falls within the category of post-development defenses that operate independently of how the model was generated. The proposed defense is built upon a novel reverse engineering approach that can directly extract backdoor functionality of a given backdoored model to a backdoor expert model. The approach is straightforward -- finetuning the backdoored model over a small set of intentionally mislabeled clean samples, such that it unlearns the normal functionality while still preserving the backdoor functionality, and thus resulting in a model (dubbed a backdoor expert model) that can only recognize backdoor inputs. Based on the extracted backdoor expert model, we show the feasibility of devising highly accurate backdoor input detectors that filter out the backdoor inputs during model inference. Further augmented by an ensemble strategy with a finetuned auxiliary model, our defense, BaDExpert (Backdoor Input Detection with Backdoor Expert), effectively mitigates 16 SOTA backdoor attacks while minimally impacting clean utility. The effectiveness of BaDExpert has been verified on multiple datasets (CIFAR10, GTSRB and ImageNet) across various model architectures (ResNet, VGG, MobileNetV2 and Vision Transformer).

摘要: 针对深度神经网络(DNNS)的后门攻击，提出了一种新的防御方法，即攻击者秘密地在DNN中植入恶意行为(后门)。我们的防御属于开发后防御的范畴，其运作独立于模型是如何生成的。建议的防御建立在一种新的逆向工程方法之上，该方法可以直接将给定后门模型的后门功能提取到后门专家模型中。这种方法很简单--在一小部分故意错误标记的干净样本上优化后门模型，以便它在保留后门功能的同时取消学习正常功能，从而产生只能识别后门输入的模型(称为后门专家模型)。基于提取的后门专家模型，我们证明了设计高精度的后门输入检测器的可行性，该检测器在模型推理过程中过滤掉后门输入。我们的防御系统BaDExpert(带有后门专家的后门输入检测)通过精细的辅助模型进一步增强了整体策略，有效地减少了16次SOTA后门攻击，同时将对清洁实用程序的影响降至最低。BaDExpert的有效性已经在各种模型架构(ResNet、VGG、MobileNetV2和Vision Transformer)的多个数据集(CIFAR10、GTSRB和ImageNet)上得到了验证。



## **42. On-Manifold Projected Gradient Descent**

流形上投影的梯度下降 cs.LG

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.12279v1) [paper-pdf](http://arxiv.org/pdf/2308.12279v1)

**Authors**: Aaron Mahler, Tyrus Berry, Tom Stephens, Harbir Antil, Michael Merritt, Jeanie Schreiber, Ioannis Kevrekidis

**Abstract**: This work provides a computable, direct, and mathematically rigorous approximation to the differential geometry of class manifolds for high-dimensional data, along with nonlinear projections from input space onto these class manifolds. The tools are applied to the setting of neural network image classifiers, where we generate novel, on-manifold data samples, and implement a projected gradient descent algorithm for on-manifold adversarial training. The susceptibility of neural networks (NNs) to adversarial attack highlights the brittle nature of NN decision boundaries in input space. Introducing adversarial examples during training has been shown to reduce the susceptibility of NNs to adversarial attack; however, it has also been shown to reduce the accuracy of the classifier if the examples are not valid examples for that class. Realistic "on-manifold" examples have been previously generated from class manifolds in the latent of an autoencoder. Our work explores these phenomena in a geometric and computational setting that is much closer to the raw, high-dimensional input space than can be provided by VAE or other black box dimensionality reductions. We employ conformally invariant diffusion maps (CIDM) to approximate class manifolds in diffusion coordinates, and develop the Nystr\"{o}m projection to project novel points onto class manifolds in this setting. On top of the manifold approximation, we leverage the spectral exterior calculus (SEC) to determine geometric quantities such as tangent vectors of the manifold. We use these tools to obtain adversarial examples that reside on a class manifold, yet fool a classifier. These misclassifications then become explainable in terms of human-understandable manipulations within the data, by expressing the on-manifold adversary in the semantic basis on the manifold.

摘要: 这项工作为高维数据的类流形的微分几何提供了一个可计算的、直接的和数学上严格的近似，以及从输入空间到这些类流形的非线性投影。这些工具被应用于神经网络图像分类器的设置，在那里我们生成新的流形上的数据样本，并实现投影梯度下降算法来进行流形上的对抗性训练。神经网络对敌意攻击的敏感性突出了输入空间中神经网络决策边界的脆性。在训练过程中引入对抗性示例可以降低神经网络对对抗性攻击的敏感性；然而，如果这些示例不是该类的有效示例，则也会降低分类器的准确性。真实的“流形上”的例子以前已经在自动编码器的潜伏中从类流形中生成。我们的工作在几何和计算环境中探索这些现象，与VAE或其他黑盒降维方法相比，它更接近原始的高维输入空间。我们使用共形不变扩散映射(CIDM)来逼近扩散坐标下的类流形，并发展了Nystr‘{o}m投影来将新的点投影到类流形上。在流形近似的基础上，我们利用谱外演算(SEC)来确定几何量，如流形的切线向量。我们使用这些工具来获得驻留在类流形上的对抗性例子，但却欺骗了分类器。然后，通过在流形上用人类可理解的操作来表示流形上的对手，这些错误分类变得可解释。



## **43. LCANets++: Robust Audio Classification using Multi-layer Neural Networks with Lateral Competition**

LCANets++：基于横向竞争的多层神经网络稳健音频分类 cs.SD

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.12882v1) [paper-pdf](http://arxiv.org/pdf/2308.12882v1)

**Authors**: Sayanton V. Dibbo, Juston S. Moore, Garrett T. Kenyon, Michael A. Teti

**Abstract**: Audio classification aims at recognizing audio signals, including speech commands or sound events. However, current audio classifiers are susceptible to perturbations and adversarial attacks. In addition, real-world audio classification tasks often suffer from limited labeled data. To help bridge these gaps, previous work developed neuro-inspired convolutional neural networks (CNNs) with sparse coding via the Locally Competitive Algorithm (LCA) in the first layer (i.e., LCANets) for computer vision. LCANets learn in a combination of supervised and unsupervised learning, reducing dependency on labeled samples. Motivated by the fact that auditory cortex is also sparse, we extend LCANets to audio recognition tasks and introduce LCANets++, which are CNNs that perform sparse coding in multiple layers via LCA. We demonstrate that LCANets++ are more robust than standard CNNs and LCANets against perturbations, e.g., background noise, as well as black-box and white-box attacks, e.g., evasion and fast gradient sign (FGSM) attacks.

摘要: 音频分类的目的是识别音频信号，包括语音命令或声音事件。然而，当前的音频分类器容易受到扰动和对抗性攻击。此外，现实世界的音频分类任务通常会受到有限的标签数据的影响。为了弥补这些差距，以前的工作发展了神经启发卷积神经网络(CNN)，通过第一层的局部竞争算法(LCA)进行稀疏编码，用于计算机视觉。LCANet在监督和非监督学习的组合中学习，减少了对标记样本的依赖。基于听觉皮层也是稀疏的这一事实，我们将LCANets扩展到音频识别任务，并引入LCANets++，LCANets++是通过LCA在多层进行稀疏编码的CNN。我们证明了LCANet++比标准的CNN和LCANet对扰动(例如背景噪声)以及黑盒和白盒攻击(例如逃避和快速梯度符号(FGSM)攻击)具有更强的鲁棒性。



## **44. Sample Complexity of Robust Learning against Evasion Attacks**

抗逃避攻击的稳健学习的样本复杂性 cs.LG

DPhil (PhD) Thesis - University of Oxford

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.12054v1) [paper-pdf](http://arxiv.org/pdf/2308.12054v1)

**Authors**: Pascale Gourdeau

**Abstract**: It is becoming increasingly important to understand the vulnerability of machine learning models to adversarial attacks. One of the fundamental problems in adversarial machine learning is to quantify how much training data is needed in the presence of evasion attacks, where data is corrupted at test time. In this thesis, we work with the exact-in-the-ball notion of robustness and study the feasibility of adversarially robust learning from the perspective of learning theory, considering sample complexity.   We first explore the setting where the learner has access to random examples only, and show that distributional assumptions are essential. We then focus on learning problems with distributions on the input data that satisfy a Lipschitz condition and show that robustly learning monotone conjunctions has sample complexity at least exponential in the adversary's budget (the maximum number of bits it can perturb on each input). However, if the adversary is restricted to perturbing $O(\log n)$ bits, then one can robustly learn conjunctions and decision lists w.r.t. log-Lipschitz distributions.   We then study learning models where the learner is given more power. We first consider local membership queries, where the learner can query the label of points near the training sample. We show that, under the uniform distribution, the exponential dependence on the adversary's budget to robustly learn conjunctions remains inevitable. We then introduce a local equivalence query oracle, which returns whether the hypothesis and target concept agree in a given region around a point in the training sample, and a counterexample if it exists. We show that if the query radius is equal to the adversary's budget, we can develop robust empirical risk minimization algorithms in the distribution-free setting. We give general query complexity upper and lower bounds, as well as for concrete concept classes.

摘要: 理解机器学习模型在对抗攻击中的脆弱性正变得越来越重要。对抗性机器学习的基本问题之一是量化在存在逃避攻击的情况下需要多少训练数据，其中数据在测试时间被破坏。本文从学习理论的角度出发，考虑样本的复杂性，提出了对抗性鲁棒学习的可行性。我们首先探索学习者只能接触随机示例的环境，并表明分布假设是必不可少的。然后，我们将重点放在满足Lipschitz条件的输入数据分布的学习问题上，并证明了稳健学习单调合取在对手的预算(它可以在每一输入上扰动的最大比特数)中具有至少指数的样本复杂性。然而，如果对手被限制为扰乱$O(\logn)$比特，则一个人可以稳健地学习合取和决策列表。对数-李普希茨分布。然后我们研究学习模式，在这种模式下，学习者被赋予更多的权力。我们首先考虑局部隶属关系查询，其中学习者可以查询训练样本附近的点的标签。我们证明，在均匀分布下，依靠对手的预算来强健地学习合取仍然是不可避免的。然后，我们引入一个局部等价查询预言，它返回假设和目标概念在训练样本点周围的给定区域是否一致，如果存在，则返回反例。我们证明了，如果查询半径等于对手的预算，我们可以在无分布的环境下开发稳健的经验风险最小化算法。我们给出了一般查询复杂度的上下界，以及具体概念类的查询复杂度。



## **45. Phase-shifted Adversarial Training**

相移对抗性训练 cs.LG

Conference on Uncertainty in Artificial Intelligence, 2023 (UAI 2023)

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2301.04785v3) [paper-pdf](http://arxiv.org/pdf/2301.04785v3)

**Authors**: Yeachan Kim, Seongyeon Kim, Ihyeok Seo, Bonggun Shin

**Abstract**: Adversarial training has been considered an imperative component for safely deploying neural network-based applications to the real world. To achieve stronger robustness, existing methods primarily focus on how to generate strong attacks by increasing the number of update steps, regularizing the models with the smoothed loss function, and injecting the randomness into the attack. Instead, we analyze the behavior of adversarial training through the lens of response frequency. We empirically discover that adversarial training causes neural networks to have low convergence to high-frequency information, resulting in highly oscillated predictions near each data. To learn high-frequency contents efficiently and effectively, we first prove that a universal phenomenon of frequency principle, i.e., \textit{lower frequencies are learned first}, still holds in adversarial training. Based on that, we propose phase-shifted adversarial training (PhaseAT) in which the model learns high-frequency components by shifting these frequencies to the low-frequency range where the fast convergence occurs. For evaluations, we conduct the experiments on CIFAR-10 and ImageNet with the adaptive attack carefully designed for reliable evaluation. Comprehensive results show that PhaseAT significantly improves the convergence for high-frequency information. This results in improved adversarial robustness by enabling the model to have smoothed predictions near each data.

摘要: 对抗性训练一直被认为是将基于神经网络的应用安全地部署到现实世界中的一个必不可少的组成部分。为了获得更强的稳健性，现有的方法主要集中在如何通过增加更新步骤、用平滑的损失函数对模型进行正则化以及在攻击中注入随机性来产生强攻击。相反，我们通过反应频率的镜头来分析对抗性训练的行为。我们经验发现，对抗性训练导致神经网络对高频信息的收敛程度较低，导致每个数据附近的预测高度振荡。为了高效有效地学习高频内容，我们首先证明了频率原理中的一个普遍现象，即先学习较低的频率在对抗性训练中仍然成立。在此基础上，我们提出了相移对抗性训练(PhaseAT)，该模型通过将高频成分转移到发生快速收敛的低频范围来学习高频成分。在评估方面，我们在CIFAR-10和ImageNet上进行了实验，并使用精心设计的自适应攻击进行了可靠的评估。综合结果表明，PhaseAT算法显著提高了高频信息的收敛速度。这使得模型能够在每个数据附近平滑预测，从而提高了对手的稳健性。



## **46. Designing an attack-defense game: how to increase robustness of financial transaction models via a competition**

设计攻防游戏：如何通过竞争增加金融交易模型的健壮性 cs.LG

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.11406v2) [paper-pdf](http://arxiv.org/pdf/2308.11406v2)

**Authors**: Alexey Zaytsev, Alex Natekin, Evgeni Vorsin, Valerii Smirnov, Georgii Smirnov, Oleg Sidorshin, Alexander Senin, Alexander Dudin, Dmitry Berestnev

**Abstract**: Given the escalating risks of malicious attacks in the finance sector and the consequential severe damage, a thorough understanding of adversarial strategies and robust defense mechanisms for machine learning models is critical. The threat becomes even more severe with the increased adoption in banks more accurate, but potentially fragile neural networks. We aim to investigate the current state and dynamics of adversarial attacks and defenses for neural network models that use sequential financial data as the input.   To achieve this goal, we have designed a competition that allows realistic and detailed investigation of problems in modern financial transaction data. The participants compete directly against each other, so possible attacks and defenses are examined in close-to-real-life conditions. Our main contributions are the analysis of the competition dynamics that answers the questions on how important it is to conceal a model from malicious users, how long does it take to break it, and what techniques one should use to make it more robust, and introduction additional way to attack models or increase their robustness.   Our analysis continues with a meta-study on the used approaches with their power, numerical experiments, and accompanied ablations studies. We show that the developed attacks and defenses outperform existing alternatives from the literature while being practical in terms of execution, proving the validity of the competition as a tool for uncovering vulnerabilities of machine learning models and mitigating them in various domains.

摘要: 鉴于金融领域不断升级的恶意攻击风险和由此带来的严重破坏，彻底了解机器学习模型的对抗性策略和强大的防御机制至关重要。随着银行越来越多地采用更准确但可能脆弱的神经网络，这种威胁变得更加严重。我们的目标是调查当前状态和动态的对抗性攻击和防御的神经网络模型，使用连续的金融数据作为输入。为了实现这一目标，我们设计了一个竞赛，允许对现代金融交易数据中的问题进行现实和详细的调查。参与者之间直接竞争，所以可能的攻击和防御都是在接近现实的条件下进行检查的。我们的主要贡献是对竞争动态的分析，回答了以下问题：向恶意用户隐藏模型有多重要，需要多长时间才能打破它，以及应该使用什么技术使其更健壮，并引入了攻击模型或增强其健壮性的其他方法。我们的分析继续对所使用的方法及其威力、数值实验和伴随的消融研究进行元研究。我们证明了所开发的攻击和防御在执行方面优于现有的可选方案，证明了竞争作为发现机器学习模型的漏洞并在不同领域缓解它们的工具的有效性。



## **47. Does Physical Adversarial Example Really Matter to Autonomous Driving? Towards System-Level Effect of Adversarial Object Evasion Attack**

身体对抗的例子对自动驾驶真的很重要吗？论对抗性目标逃避攻击的系统级效应 cs.CR

Accepted by ICCV 2023

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.11894v1) [paper-pdf](http://arxiv.org/pdf/2308.11894v1)

**Authors**: Ningfei Wang, Yunpeng Luo, Takami Sato, Kaidi Xu, Qi Alfred Chen

**Abstract**: In autonomous driving (AD), accurate perception is indispensable to achieving safe and secure driving. Due to its safety-criticality, the security of AD perception has been widely studied. Among different attacks on AD perception, the physical adversarial object evasion attacks are especially severe. However, we find that all existing literature only evaluates their attack effect at the targeted AI component level but not at the system level, i.e., with the entire system semantics and context such as the full AD pipeline. Thereby, this raises a critical research question: can these existing researches effectively achieve system-level attack effects (e.g., traffic rule violations) in the real-world AD context? In this work, we conduct the first measurement study on whether and how effectively the existing designs can lead to system-level effects, especially for the STOP sign-evasion attacks due to their popularity and severity. Our evaluation results show that all the representative prior works cannot achieve any system-level effects. We observe two design limitations in the prior works: 1) physical model-inconsistent object size distribution in pixel sampling and 2) lack of vehicle plant model and AD system model consideration. Then, we propose SysAdv, a novel system-driven attack design in the AD context and our evaluation results show that the system-level effects can be significantly improved, i.e., the violation rate increases by around 70%.

摘要: 在自动驾驶中，准确的感知是实现安全驾驶不可缺少的。由于其安全临界性，AD感知的安全性得到了广泛的研究。在针对AD感知的各种攻击中，物理对抗性对象逃避攻击尤为严重。然而，我们发现，现有的所有文献只在目标人工智能组件级别上评估其攻击效果，而没有在系统级别上进行评估，即使用整个系统语义和上下文，如完整的AD流水线。因此，这就提出了一个关键的研究问题：这些现有的研究能否在现实的AD环境中有效地实现系统级的攻击效果(例如，违反交通规则)？在这项工作中，我们进行了第一次测量研究现有的设计是否以及如何有效地导致系统级的影响，特别是由于停车标志逃避攻击的流行和严重程度。我们的评估结果表明，所有有代表性的前期工作都不能达到任何系统级的效果。我们观察到了前人工作中的两个设计局限性：1)物理模型-像素采样中对象大小分布不一致；2)缺乏对车辆植物模型和AD系统模型的考虑。然后，我们提出了一种新的AD环境下的系统驱动攻击设计--SysAdv，我们的评估结果表明，系统级的攻击效果可以得到显著的改善，即违规率提高了70%左右。



## **48. Measuring Equality in Machine Learning Security Defenses: A Case Study in Speech Recognition**

机器学习安全防御中的平等度量：以语音识别为例 cs.LG

Accepted to AISec'23

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2302.08973v6) [paper-pdf](http://arxiv.org/pdf/2302.08973v6)

**Authors**: Luke E. Richards, Edward Raff, Cynthia Matuszek

**Abstract**: Over the past decade, the machine learning security community has developed a myriad of defenses for evasion attacks. An understudied question in that community is: for whom do these defenses defend? This work considers common approaches to defending learned systems and how security defenses result in performance inequities across different sub-populations. We outline appropriate parity metrics for analysis and begin to answer this question through empirical results of the fairness implications of machine learning security methods. We find that many methods that have been proposed can cause direct harm, like false rejection and unequal benefits from robustness training. The framework we propose for measuring defense equality can be applied to robustly trained models, preprocessing-based defenses, and rejection methods. We identify a set of datasets with a user-centered application and a reasonable computational cost suitable for case studies in measuring the equality of defenses. In our case study of speech command recognition, we show how such adversarial training and augmentation have non-equal but complex protections for social subgroups across gender, accent, and age in relation to user coverage. We present a comparison of equality between two rejection-based defenses: randomized smoothing and neural rejection, finding randomized smoothing more equitable due to the sampling mechanism for minority groups. This represents the first work examining the disparity in the adversarial robustness in the speech domain and the fairness evaluation of rejection-based defenses.

摘要: 在过去的十年里，机器学习安全社区开发了无数针对逃避攻击的防御措施。在那个社区里，一个被忽视的问题是：这些防御是为谁辩护？这项工作考虑了保护学习系统的常见方法，以及安全防御如何导致不同子群体的性能不平等。我们勾勒出合适的奇偶度量进行分析，并通过机器学习安全方法的公平性影响的实证结果开始回答这个问题。我们发现，已提出的许多方法都会造成直接危害，如错误拒绝和健壮性训练带来的不平等好处。我们提出的衡量防御平等的框架可以应用于稳健训练的模型、基于预处理的防御和拒绝方法。我们确定了一组数据集，这些数据集具有以用户为中心的应用程序，并且计算成本合理，适合用于案例研究来衡量辩护的等价性。在我们的语音命令识别案例研究中，我们展示了这种对抗性训练和增强如何根据用户覆盖范围为不同性别、口音和年龄的社会亚群提供不平等但复杂的保护。我们比较了两种基于拒绝的防御机制：随机平滑和神经排斥，发现由于少数群体的抽样机制，随机平滑更公平。这是第一个研究语音域中对抗性稳健性的差异和基于拒绝的防御的公平性评估的工作。



## **49. SEA: Shareable and Explainable Attribution for Query-based Black-box Attacks**

SEA：基于查询的黑盒攻击的可共享和可解释的属性 cs.LG

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.11845v1) [paper-pdf](http://arxiv.org/pdf/2308.11845v1)

**Authors**: Yue Gao, Ilia Shumailov, Kassem Fawaz

**Abstract**: Machine Learning (ML) systems are vulnerable to adversarial examples, particularly those from query-based black-box attacks. Despite various efforts to detect and prevent such attacks, there is a need for a more comprehensive approach to logging, analyzing, and sharing evidence of attacks. While classic security benefits from well-established forensics and intelligence sharing, Machine Learning is yet to find a way to profile its attackers and share information about them. In response, this paper introduces SEA, a novel ML security system to characterize black-box attacks on ML systems for forensic purposes and to facilitate human-explainable intelligence sharing. SEA leverages the Hidden Markov Models framework to attribute the observed query sequence to known attacks. It thus understands the attack's progression rather than just focusing on the final adversarial examples. Our evaluations reveal that SEA is effective at attack attribution, even on their second occurrence, and is robust to adaptive strategies designed to evade forensics analysis. Interestingly, SEA's explanations of the attack behavior allow us even to fingerprint specific minor implementation bugs in attack libraries. For example, we discover that the SignOPT and Square attacks implementation in ART v1.14 sends over 50% specific zero difference queries. We thoroughly evaluate SEA on a variety of settings and demonstrate that it can recognize the same attack's second occurrence with 90+% Top-1 and 95+% Top-3 accuracy.

摘要: 机器学习(ML)系统容易受到敌意例子的攻击，特别是那些来自基于查询的黑盒攻击的例子。尽管做出了各种努力来检测和防止此类攻击，但仍需要一种更全面的方法来记录、分析和共享攻击证据。虽然传统的安全技术得益于成熟的取证和情报共享，但机器学习尚未找到一种方法来分析攻击者并分享有关他们的信息。对此，本文引入了一种新的ML安全系统SEA，用于刻画针对ML系统的黑盒攻击，用于取证目的，并促进人类可解释的情报共享。SEA利用隐马尔可夫模型框架将观察到的查询序列归因于已知攻击。因此，它了解攻击的进展，而不是只关注最后的对抗性例子。我们的评估表明，SEA在攻击归属方面是有效的，即使在第二次攻击发生时也是如此，并且对于旨在逃避取证分析的自适应策略是健壮的。有趣的是，SEA对攻击行为的解释甚至允许我们确定攻击库中特定的微小实现错误。例如，我们发现ART v1.14中的SignOPT和Square攻击实现发送超过50%的特定零差查询。我们在各种不同的设置下对SEA进行了全面的评估，并证明了它能够以90%以上的Top-1和95%+%的Top-3准确率识别同一攻击的第二次发生。



## **50. Ceci n'est pas une pomme: Adversarial Illusions in Multi-Modal Embeddings**

Ceci n‘est Pas une Pomme：多模式嵌入中的对抗性错觉 cs.CR

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2308.11804v1) [paper-pdf](http://arxiv.org/pdf/2308.11804v1)

**Authors**: Eugene Bagdasaryan, Vitaly Shmatikov

**Abstract**: Multi-modal encoders map images, sounds, texts, videos, etc. into a single embedding space, aligning representations across modalities (e.g., associate an image of a dog with a barking sound). We show that multi-modal embeddings can be vulnerable to an attack we call "adversarial illusions." Given an input in any modality, an adversary can perturb it so as to make its embedding close to that of an arbitrary, adversary-chosen input in another modality. Illusions thus enable the adversary to align any image with any text, any text with any sound, etc.   Adversarial illusions exploit proximity in the embedding space and are thus agnostic to downstream tasks. Using ImageBind embeddings, we demonstrate how adversarially aligned inputs, generated without knowledge of specific downstream tasks, mislead image generation, text generation, and zero-shot classification.

摘要: 多模式编码器将图像、声音、文本、视频等映射到单个嵌入空间中，跨模式对齐表示(例如，将狗的图像与犬吠声相关联)。我们表明，多模式嵌入可能容易受到一种我们称为“对抗性错觉”的攻击。给定任何形式的输入，敌手都可以对其进行干扰，以便使其嵌入接近于任意的、对手选择的另一种形式的输入。因此，错觉使对手能够将任何图像与任何文本、任何文本与任何声音等对齐。对抗性错觉利用嵌入空间中的邻近，因此与下游任务无关。使用ImageBind嵌入，我们演示了在不知道特定下游任务的情况下生成的恶意对齐的输入如何误导图像生成、文本生成和零镜头分类。



