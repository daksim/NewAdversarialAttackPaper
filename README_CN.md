# Latest Adversarial Attack Papers
**update at 2023-10-16 15:13:22**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. A Stochastic Surveillance Stackelberg Game: Co-Optimizing Defense Placement and Patrol Strategy**

随机监视Stackelberg博弈：共同优化防御部署和巡逻策略 eess.SY

8 pages, 1 figure, jointly submitted to the IEEE Control Systems  Letters and the 2024 American Control Conference. Replaced to fix typos

**SubmitDate**: 2023-10-13    [abs](http://arxiv.org/abs/2308.14714v2) [paper-pdf](http://arxiv.org/pdf/2308.14714v2)

**Authors**: Yohan John, Gilberto Diaz-Garcia, Xiaoming Duan, Jason R. Marden, Francesco Bullo

**Abstract**: Stochastic patrol routing is known to be advantageous in adversarial settings; however, the optimal choice of stochastic routing strategy is dependent on a model of the adversary. Duan et al. formulated a Stackelberg game for the worst-case scenario, i.e., a surveillance agent confronted with an omniscient attacker [IEEE TCNS, 8(2), 769-80, 2021]. In this article, we extend their formulation to accommodate heterogeneous defenses at the various nodes of the graph. We derive an upper bound on the value of the game. We identify methods for computing effective patrol strategies for certain classes of graphs. Finally, we leverage the heterogeneous defense formulation to develop novel defense placement algorithms that complement the patrol strategies.

摘要: 随机巡逻路径在敌方环境中具有优势，然而，随机路径策略的最优选择取决于敌方的模型。Duan et al.为最糟糕的情况制定了Stackelberg游戏，即监视特工与无所不知的攻击者对峙[IEEE TCNs，8(2)，769-80,2021]。在这篇文章中，我们扩展了他们的公式，以适应图的不同节点上的不同防御。我们得到了博弈价值的一个上界。我们确定了计算某些图类的有效巡视策略的方法。最后，我们利用异质防御公式来开发新的防御布局算法，以补充巡逻策略。



## **2. SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks**

SmoothLLM：保护大型语言模型免受越狱攻击 cs.LG

**SubmitDate**: 2023-10-13    [abs](http://arxiv.org/abs/2310.03684v2) [paper-pdf](http://arxiv.org/pdf/2310.03684v2)

**Authors**: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas

**Abstract**: Despite efforts to align large language models (LLMs) with human values, widely-used LLMs such as GPT, Llama, Claude, and PaLM are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks on LLMs. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense first randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. SmoothLLM reduces the attack success rate on numerous popular LLMs to below one percentage point, avoids unnecessary conservatism, and admits provable guarantees on attack mitigation. Moreover, our defense uses exponentially fewer queries than existing attacks and is compatible with any LLM.

摘要: 尽管努力使大型语言模型(LLM)与人类价值观保持一致，但GPT、Llama、Claude和Palm等广泛使用的LLM容易受到越狱攻击，即对手欺骗目标LLM生成令人反感的内容。为了解决这一漏洞，我们提出了SmoothLLM，这是第一个旨在缓解对LLM的越狱攻击的算法。基于我们的发现，对抗性生成的提示对字符级别的变化很脆弱，我们的防御首先随机扰动给定输入提示的多个副本，然后聚合相应的预测来检测对抗性输入。SmoothLLM将许多流行的LLM的攻击成功率降低到1个百分点以下，避免了不必要的保守主义，并承认了对攻击缓解的可证明保证。此外，我们的防御使用的查询比现有攻击少得多，并且与任何LLM兼容。



## **3. Worst-Case Morphs using Wasserstein ALI and Improved MIPGAN**

使用Wasserstein Ali和改进的MIPGAN进行最坏情况的变形 cs.CV

**SubmitDate**: 2023-10-13    [abs](http://arxiv.org/abs/2310.08371v2) [paper-pdf](http://arxiv.org/pdf/2310.08371v2)

**Authors**: Una M. Kelly, Meike Nauta, Lu Liu, Luuk J. Spreeuwers, Raymond N. J. Veldhuis

**Abstract**: A morph is a combination of two separate facial images and contains identity information of two different people. When used in an identity document, both people can be authenticated by a biometric Face Recognition (FR) system. Morphs can be generated using either a landmark-based approach or approaches based on deep learning such as Generative Adversarial Networks (GAN). In a recent paper, we introduced a \emph{worst-case} upper bound on how challenging morphing attacks can be for an FR system. The closer morphs are to this upper bound, the bigger the challenge they pose to FR. We introduced an approach with which it was possible to generate morphs that approximate this upper bound for a known FR system (white box), but not for unknown (black box) FR systems.   In this paper, we introduce a morph generation method that can approximate worst-case morphs even when the FR system is not known. A key contribution is that we include the goal of generating difficult morphs \emph{during} training. Our method is based on Adversarially Learned Inference (ALI) and uses concepts from Wasserstein GANs trained with Gradient Penalty, which were introduced to stabilise the training of GANs. We include these concepts to achieve similar improvement in training stability and call the resulting method Wasserstein ALI (WALI). We finetune WALI using loss functions designed specifically to improve the ability to manipulate identity information in facial images and show how it can generate morphs that are more challenging for FR systems than landmark- or GAN-based morphs. We also show how our findings can be used to improve MIPGAN, an existing StyleGAN-based morph generator.

摘要: 变形是两个单独的面部图像的组合，包含两个不同人的身份信息。当在身份证件中使用时，两人都可以通过生物特征面部识别(FR)系统进行身份验证。可以使用基于里程碑的方法或基于深度学习的方法来生成变形，例如生成性对抗网络(GAN)。在最近的一篇论文中，我们引入了一个关于变形攻击对FR系统的挑战性的上限。变形越接近这个上限，它们对FR构成的挑战就越大。我们介绍了一种方法，利用这种方法，可以为已知的FR系统(白盒)生成近似此上限的变形，但不能为未知的(黑盒)FR系统生成此上界。在本文中，我们介绍了一种变形生成方法，即使在FR系统未知的情况下，该方法也可以近似最坏情况的变形。一个重要的贡献是，我们包含了在训练过程中生成困难变形的目标。我们的方法基于对抗性学习推理(ALI)，并使用了Wasserstein Gans中的概念，这些概念被引入以稳定Gans的训练。我们纳入这些概念是为了在训练稳定性方面实现类似的改进，并将结果方法称为Wasserstein Ali(WALI)。我们使用专门为提高处理面部图像中的身份信息的能力而设计的损失函数来优化WALI，并展示了它如何生成对FR系统来说比里程碑式或基于GaN的变形更具挑战性的变形。我们还展示了如何使用我们的发现来改进MIPGAN，这是一个现有的基于StyleGAN的变形生成器。



## **4. Attacking The Assortativity Coefficient Under A Rewiring Strategy**

重新布线策略下的吸声系数攻击 cs.SI

**SubmitDate**: 2023-10-13    [abs](http://arxiv.org/abs/2310.08924v1) [paper-pdf](http://arxiv.org/pdf/2310.08924v1)

**Authors**: Shuo Zou, Bo Zhou, Qi Xuan

**Abstract**: Degree correlation is an important characteristic of networks, which is usually quantified by the assortativity coefficient. However, concerns arise about changing the assortativity coefficient of a network when networks suffer from adversarial attacks. In this paper, we analyze the factors that affect the assortativity coefficient and study the optimization problem of maximizing or minimizing the assortativity coefficient (r) in rewired networks with $k$ pairs of edges. We propose a greedy algorithm and formulate the optimization problem using integer programming to obtain the optimal solution for this problem. Through experiments, we demonstrate the reasonableness and effectiveness of our proposed algorithm. For example, rewired edges 10% in the ER network, the assortativity coefficient improved by 60%.

摘要: 度相关性是网络的一个重要特征，通常用分类系数来量化。然而，当网络遭受敌意攻击时，人们担心改变网络的分类系数。本文分析了影响配色系数的因素，研究了具有$k$对边的重连网络中配色系数(R)最大或最小的优化问题。提出了一种贪婪算法，并用整数规划对优化问题进行了形式化描述，得到了该问题的最优解。通过实验验证了该算法的合理性和有效性。例如，在ER网络中重新布线10%的边，分类系数提高了60%。



## **5. OTJR: Optimal Transport Meets Optimal Jacobian Regularization for Adversarial Robustness**

OTJR：最优传输满足最优雅可比正则化的对抗性 cs.CV

**SubmitDate**: 2023-10-13    [abs](http://arxiv.org/abs/2303.11793v2) [paper-pdf](http://arxiv.org/pdf/2303.11793v2)

**Authors**: Binh M. Le, Shahroz Tariq, Simon S. Woo

**Abstract**: The Web, as a rich medium of diverse content, has been constantly under the threat of malicious entities exploiting its vulnerabilities, especially with the rapid proliferation of deep learning applications in various web services. One such vulnerability, crucial to the fidelity and integrity of web content, is the susceptibility of deep neural networks to adversarial perturbations, especially concerning images - a dominant form of data on the web. In light of the recent advancements in the robustness of classifiers, we delve deep into the intricacies of adversarial training (AT) and Jacobian regularization, two pivotal defenses. Our work {is the} first carefully analyzes and characterizes these two schools of approaches, both theoretically and empirically, to demonstrate how each approach impacts the robust learning of a classifier. Next, we propose our novel Optimal Transport with Jacobian regularization method, dubbed~\SystemName, jointly incorporating the input-output Jacobian regularization into the AT by leveraging the optimal transport theory. In particular, we employ the Sliced Wasserstein (SW) distance that can efficiently push the adversarial samples' representations closer to those of clean samples, regardless of the number of classes within the dataset. The SW distance provides the adversarial samples' movement directions, which are much more informative and powerful for the Jacobian regularization. Our empirical evaluations set a new standard in the domain, with our method achieving commendable accuracies of 51.41\% on the ~\CIFAR-10 and 28.49\% on the ~\CIFAR-100 datasets under the AutoAttack metric. In a real-world demonstration, we subject images sourced from the Internet to online adversarial attacks, reinforcing the efficacy and relevance of our model in defending against sophisticated web-image perturbations.

摘要: Web作为一种内容丰富的媒体，一直受到恶意实体利用其漏洞的威胁，特别是随着各种Web服务中深度学习应用的迅速激增。其中一个对网络内容的保真度和完整性至关重要的漏洞是深度神经网络对敌意干扰的敏感性，特别是关于图像--网络上的一种主要数据形式。鉴于最近在分类器稳健性方面的进展，我们深入研究了对抗性训练(AT)和雅可比正则化这两个关键防御措施的复杂性。我们的工作是第一次从理论和经验上仔细分析和表征这两种方法，以证明每种方法如何影响分类器的稳健学习。接下来，我们利用最优传输理论，将输入输出的雅可比正则化引入到AT中，提出了一种新的基于雅可比正则化的最优传输方法--系统名。特别是，我们使用了切片Wasserstein(SW)距离，该距离可以有效地将对抗性样本的表示更接近于干净样本的表示，而不管数据集中有多少类。Sw距离提供了对抗性样本的运动方向，为雅可比正则化提供了更多的信息和更强大的能力。我们的经验评估在该领域建立了一个新的标准，我们的方法在AutoAttack度量下在~CIFAR-10和~\CIFAR-100数据集上分别获得了51.41\%和28.49\%的值得称赞的精度。在现实世界的演示中，我们将来自互联网的图像置于在线敌意攻击中，加强了我们的模型在防御复杂的网络图像扰动方面的有效性和相关性。



## **6. Attacks Meet Interpretability (AmI) Evaluation and Findings**

攻击符合可解释性(AMI)评估和调查结果 cs.CR

5 pages, 4 figures

**SubmitDate**: 2023-10-13    [abs](http://arxiv.org/abs/2310.08808v1) [paper-pdf](http://arxiv.org/pdf/2310.08808v1)

**Authors**: Qian Ma, Ziping Ye, Shagufta Mehnaz

**Abstract**: To investigate the effectiveness of the model explanation in detecting adversarial examples, we reproduce the results of two papers, Attacks Meet Interpretability: Attribute-steered Detection of Adversarial Samples and Is AmI (Attacks Meet Interpretability) Robust to Adversarial Examples. And then conduct experiments and case studies to identify the limitations of both works. We find that Attacks Meet Interpretability(AmI) is highly dependent on the selection of hyperparameters. Therefore, with a different hyperparameter choice, AmI is still able to detect Nicholas Carlini's attack. Finally, we propose recommendations for future work on the evaluation of defense techniques such as AmI.

摘要: 为了考察模型解释在检测敌意实例方面的有效性，我们复制了两篇论文的结果：攻击满足解释性：对抗性样本的属性导向检测和AMI(攻击满足解释性)对对抗性实例的稳健性。然后进行实验和案例研究，找出两部作品的局限性。我们发现攻击满足可解释性(AMI)高度依赖于超参数的选择。因此，通过不同的超参数选择，阿米仍然能够检测到尼古拉斯·卡里尼的攻击。最后，我们对AMI等防御技术的未来评估工作提出了建议。



## **7. Fed-Safe: Securing Federated Learning in Healthcare Against Adversarial Attacks**

FED-SAFE：确保医疗保健领域的联合学习免受对手攻击 cs.CV

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.08681v1) [paper-pdf](http://arxiv.org/pdf/2310.08681v1)

**Authors**: Erfan Darzi, Nanna M. Sijtsema, P. M. A van Ooijen

**Abstract**: This paper explores the security aspects of federated learning applications in medical image analysis. Current robustness-oriented methods like adversarial training, secure aggregation, and homomorphic encryption often risk privacy compromises. The central aim is to defend the network against potential privacy breaches while maintaining model robustness against adversarial manipulations. We show that incorporating distributed noise, grounded in the privacy guarantees in federated settings, enables the development of a adversarially robust model that also meets federated privacy standards. We conducted comprehensive evaluations across diverse attack scenarios, parameters, and use cases in cancer imaging, concentrating on pathology, meningioma, and glioma. The results reveal that the incorporation of distributed noise allows for the attainment of security levels comparable to those of conventional adversarial training while requiring fewer retraining samples to establish a robust model.

摘要: 本文探讨了联合学习在医学图像分析中应用的安全性问题。当前面向健壮性的方法，如对抗性训练、安全聚合和同态加密，往往存在隐私泄露的风险。中心目标是保护网络免受潜在的隐私侵犯，同时保持模型对敌意操纵的健壮性。我们表明，结合分布式噪声，植根于联合环境中的隐私保障，能够开发出一种也满足联合隐私标准的对抗性健壮模型。我们在癌症成像中对不同的攻击场景、参数和用例进行了全面的评估，重点放在病理学、脑膜瘤和胶质瘤上。结果表明，引入分布噪声可以达到与传统对抗性训练相当的安全水平，同时需要更少的再训练样本来建立稳健的模型。



## **8. Unclonable Non-Interactive Zero-Knowledge**

不可克隆的非交互零知识 cs.CR

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.07118v2) [paper-pdf](http://arxiv.org/pdf/2310.07118v2)

**Authors**: Ruta Jawale, Dakshita Khurana

**Abstract**: A non-interactive ZK (NIZK) proof enables verification of NP statements without revealing secrets about them. However, an adversary that obtains a NIZK proof may be able to clone this proof and distribute arbitrarily many copies of it to various entities: this is inevitable for any proof that takes the form of a classical string. In this paper, we ask whether it is possible to rely on quantum information in order to build NIZK proof systems that are impossible to clone.   We define and construct unclonable non-interactive zero-knowledge proofs (of knowledge) for NP. Besides satisfying the zero-knowledge and proof of knowledge properties, these proofs additionally satisfy unclonability. Very roughly, this ensures that no adversary can split an honestly generated proof of membership of an instance $x$ in an NP language $\mathcal{L}$ and distribute copies to multiple entities that all obtain accepting proofs of membership of $x$ in $\mathcal{L}$. Our result has applications to unclonable signatures of knowledge, which we define and construct in this work; these non-interactively prevent replay attacks.

摘要: 非交互ZK(NIZK)证明能够在不泄露有关NP语句的秘密的情况下验证NP语句。然而，获得NIZK证明的对手可能能够克隆该证明并将其任意多个副本分发给各种实体：对于任何采用经典字符串形式的证明来说，这是不可避免的。在这篇论文中，我们问是否有可能依靠量子信息来建立不可能克隆的NIZK证明系统。我们定义并构造了NP的不可克隆、非交互的零知识证明。这些证明除了满足零知识和知识证明性质外，还满足不可克隆性。粗略地说，这确保了没有对手能够用NP语言$\数学{L}$拆分诚实地生成的实例$x$的成员资格证明，并将副本分发给多个实体，这些实体都获得了$\数学{L}$中的$x$的成员资格的接受证明。我们的结果适用于不可克隆的知识签名，我们在本工作中定义和构造了这些签名；这些非交互的签名可以防止重放攻击。



## **9. Bucks for Buckets (B4B): Active Defenses Against Stealing Encoders**

以桶换钱(B4B)：主动防御窃取编码器 cs.LG

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.08571v1) [paper-pdf](http://arxiv.org/pdf/2310.08571v1)

**Authors**: Jan Dubiński, Stanisław Pawlak, Franziska Boenisch, Tomasz Trzciński, Adam Dziedzic

**Abstract**: Machine Learning as a Service (MLaaS) APIs provide ready-to-use and high-utility encoders that generate vector representations for given inputs. Since these encoders are very costly to train, they become lucrative targets for model stealing attacks during which an adversary leverages query access to the API to replicate the encoder locally at a fraction of the original training costs. We propose Bucks for Buckets (B4B), the first active defense that prevents stealing while the attack is happening without degrading representation quality for legitimate API users. Our defense relies on the observation that the representations returned to adversaries who try to steal the encoder's functionality cover a significantly larger fraction of the embedding space than representations of legitimate users who utilize the encoder to solve a particular downstream task.vB4B leverages this to adaptively adjust the utility of the returned representations according to a user's coverage of the embedding space. To prevent adaptive adversaries from eluding our defense by simply creating multiple user accounts (sybils), B4B also individually transforms each user's representations. This prevents the adversary from directly aggregating representations over multiple accounts to create their stolen encoder copy. Our active defense opens a new path towards securely sharing and democratizing encoders over public APIs.

摘要: 机器学习即服务(MLaaS)API提供了现成的、高实用的编码器，可以为给定的输入生成向量表示。由于这些编码器的培训成本非常高，他们成为模型窃取攻击的有利可图的目标，在攻击期间，对手利用对API的查询访问来以原始培训成本的一小部分在本地复制编码器。我们提出了Bucks for Buckets(B4B)，这是第一种主动防御，可以在攻击发生时防止窃取，而不会降低合法API用户的表示质量。我们的辩护依赖于这样的观察，即返回给试图窃取编码器功能的对手的表示覆盖的嵌入空间比利用编码器解决特定下游任务的合法用户的表示大得多。vB4B利用这一点来根据用户对嵌入空间的覆盖自适应地调整返回的表示的效用。为了防止适应性对手通过简单地创建多个用户帐户(Sybils)来逃避我们的防御，B4B还单独转换每个用户的表示。这可以防止对手直接在多个帐户上聚合表示，以创建他们被盗的编码器副本。我们的积极防御为通过公共API安全共享和民主化编码器开辟了一条新的道路。



## **10. Jailbreaking Black Box Large Language Models in Twenty Queries**

20个查询中的越狱黑箱大语言模型 cs.LG

21 pages, 10 figures

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.08419v1) [paper-pdf](http://arxiv.org/pdf/2310.08419v1)

**Authors**: Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, Eric Wong

**Abstract**: There is growing interest in ensuring that large language models (LLMs) align with human values. However, the alignment of such models is vulnerable to adversarial jailbreaks, which coax LLMs into overriding their safety guardrails. The identification of these vulnerabilities is therefore instrumental in understanding inherent weaknesses and preventing future misuse. To this end, we propose Prompt Automatic Iterative Refinement (PAIR), an algorithm that generates semantic jailbreaks with only black-box access to an LLM. PAIR -- which is inspired by social engineering attacks -- uses an attacker LLM to automatically generate jailbreaks for a separate targeted LLM without human intervention. In this way, the attacker LLM iteratively queries the target LLM to update and refine a candidate jailbreak. Empirically, PAIR often requires fewer than twenty queries to produce a jailbreak, which is orders of magnitude more efficient than existing algorithms. PAIR also achieves competitive jailbreaking success rates and transferability on open and closed-source LLMs, including GPT-3.5/4, Vicuna, and PaLM-2.

摘要: 人们对确保大型语言模型(LLM)与人类价值观保持一致的兴趣与日俱增。然而，这类模型的调整很容易受到对抗性越狱的影响，这会诱使低收入国家凌驾于他们的安全护栏之上。因此，确定这些漏洞有助于了解固有的弱点并防止今后的滥用。为此，我们提出了即时自动迭代求精(Pair)，这是一种仅通过黑盒访问LLM来生成语义越狱的算法。Pair受到社会工程攻击的启发，它使用攻击者LLM自动为单独的目标LLM生成越狱，而无需人工干预。通过这种方式，攻击者LLM迭代地查询目标LLM以更新和改进候选越狱。根据经验，Pair通常只需要不到20次查询就可以产生越狱，这比现有算法的效率高出几个数量级。Pair还在开放和封闭源代码的LLM上实现了具有竞争力的越狱成功率和可转移性，包括GPT-3.5/4、维库纳和Palm-2。



## **11. Vault: Decentralized Storage Made Durable**

保险库：经久耐用的分散式存储 cs.DC

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.08403v1) [paper-pdf](http://arxiv.org/pdf/2310.08403v1)

**Authors**: Guangda Sun, Michael Hu Yiqing, Arun Fu, Akasha Zhu, Jialin Li

**Abstract**: The lack of centralized control, combined with highly dynamic adversarial behaviors, makes data durability a challenge in decentralized storage systems. In this work, we introduce a new storage system, Vault, that offers strong data durability guarantees in a fully decentralized, permission-less setting. Vault leverages the rateless property of erasure code to encode each data object into an infinite stream of encoding fragments. To ensure durability in the presence of dynamic Byzantine behaviors and targeted attacks, an infinite sequence of storage nodes are randomly selected to store encoding fragments. Encoding generation and candidate selection are fully decentralized: When necessary, Vault nodes use a gossip protocol and a publically verifiable selection proof to determine new fragments. Simulations and large-scale EC2 experiments demonstrate that Vault provides close-to-ideal mean-time-to-data-loss (MTTDL) with low storage redundancy, scales to more than 10,000 nodes, and attains performance comparable to IPFS

摘要: 缺乏集中控制，再加上高度动态的对抗性行为，使得分散存储系统中的数据持久性成为一个挑战。在这项工作中，我们介绍了一种新的存储系统，即Vault，它在完全去中心化、无权限的设置中提供强大的数据持久性保证。Vault利用擦除代码的无比率特性将每个数据对象编码为无限的编码片段流。为了确保在存在动态拜占庭行为和有针对性的攻击时的持久性，随机选择无限序列的存储节点来存储编码片段。编码生成和候选选择是完全分散的：必要时，Vault节点使用八卦协议和可公开验证的选择证据来确定新的片段。模拟和大规模EC2实验表明，Vault提供接近理想的平均数据丢失时间(MTTDL)和低存储冗余，可扩展到10,000多个节点，并获得与IPFS相当的性能



## **12. An Initial Investigation of Neural Replay Simulator for Over-the-Air Adversarial Perturbations to Automatic Speaker Verification**

用于自动说话人确认的空中对抗扰动神经重放模拟器的初步研究 cs.SD

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.05354v2) [paper-pdf](http://arxiv.org/pdf/2310.05354v2)

**Authors**: Jiaqi Li, Li Wang, Liumeng Xue, Lei Wang, Zhizheng Wu

**Abstract**: Deep Learning has advanced Automatic Speaker Verification (ASV) in the past few years. Although it is known that deep learning-based ASV systems are vulnerable to adversarial examples in digital access, there are few studies on adversarial attacks in the context of physical access, where a replay process (i.e., over the air) is involved. An over-the-air attack involves a loudspeaker, a microphone, and a replaying environment that impacts the movement of the sound wave. Our initial experiment confirms that the replay process impacts the effectiveness of the over-the-air attack performance. This study performs an initial investigation towards utilizing a neural replay simulator to improve over-the-air adversarial attack robustness. This is achieved by using a neural waveform synthesizer to simulate the replay process when estimating the adversarial perturbations. Experiments conducted on the ASVspoof2019 dataset confirm that the neural replay simulator can considerably increase the success rates of over-the-air adversarial attacks. This raises the concern for adversarial attacks on speaker verification in physical access applications.

摘要: 在过去的几年里，深度学习发展了自动说话人确认(ASV)。虽然众所周知，基于深度学习的ASV系统在数字访问中容易受到敌意攻击，但在涉及重播过程(即空中重播)的物理访问环境中，很少有关于对抗性攻击的研究。空中攻击包括扬声器、麦克风和影响声波移动的重放环境。我们的初步实验证实，重放过程会影响空中攻击性能的有效性。本研究对利用神经重放模拟器来提高空中对抗攻击的稳健性进行了初步的研究。这是通过使用神经波形合成器来模拟在估计对抗性扰动时的重播过程来实现的。在ASVspoof2019数据集上进行的实验证实，神经重放模拟器可以显著提高空中对抗性攻击的成功率。这引起了人们对物理访问应用中说话人验证的对抗性攻击的关注。



## **13. Defending Our Privacy With Backdoors**

用后门捍卫我们的隐私 cs.LG

14 pages, 4 figures

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.08320v1) [paper-pdf](http://arxiv.org/pdf/2310.08320v1)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Daniel Neider, Kristian Kersting

**Abstract**: The proliferation of large AI models trained on uncurated, often sensitive web-scraped data has raised significant privacy concerns. One of the concerns is that adversaries can extract information about the training data using privacy attacks. Unfortunately, the task of removing specific information from the models without sacrificing performance is not straightforward and has proven to be challenging. We propose a rather easy yet effective defense based on backdoor attacks to remove private information such as names of individuals from models, and focus in this work on text encoders. Specifically, through strategic insertion of backdoors, we align the embeddings of sensitive phrases with those of neutral terms-"a person" instead of the person's name. Our empirical results demonstrate the effectiveness of our backdoor-based defense on CLIP by assessing its performance using a specialized privacy attack for zero-shot classifiers. Our approach provides not only a new "dual-use" perspective on backdoor attacks, but also presents a promising avenue to enhance the privacy of individuals within models trained on uncurated web-scraped data.

摘要: 大型人工智能模型的激增引发了人们对隐私的严重担忧。这些模型针对未经管理的、往往是敏感的网络数据进行培训。其中一个令人担忧的问题是，攻击者可以使用隐私攻击来提取有关训练数据的信息。不幸的是，在不牺牲性能的情况下从模型中删除特定信息的任务并不简单，而且已被证明是具有挑战性的。我们提出了一种基于后门攻击的简单而有效的防御方法，将个人姓名等私人信息从模型中移除，并将重点放在文本编码器上。具体地说，通过策略性地插入后门，我们将敏感短语的嵌入与中性术语--“人”而不是人的名字--保持一致。我们的实验结果证明了我们的基于后门的防御在CLIP上的有效性，通过使用专门的针对零镜头分类器的隐私攻击来评估其性能。我们的方法不仅为后门攻击提供了一种新的“两用”视角，而且还提供了一种在未经管理的网络抓取数据的培训模型中增强个人隐私的有前景的途径。



## **14. Concealed Electronic Countermeasures of Radar Signal with Adversarial Examples**

雷达信号隐身电子对抗的对抗实例 eess.SP

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.08292v1) [paper-pdf](http://arxiv.org/pdf/2310.08292v1)

**Authors**: Ruinan Ma, Canjie Zhu, Mingfeng Lu, Yunjie Li, Yu-an Tan, Ruibin Zhang, Ran Tao

**Abstract**: Electronic countermeasures involving radar signals are an important aspect of modern warfare. Traditional electronic countermeasures techniques typically add large-scale interference signals to ensure interference effects, which can lead to attacks being too obvious. In recent years, AI-based attack methods have emerged that can effectively solve this problem, but the attack scenarios are currently limited to time domain radar signal classification. In this paper, we focus on the time-frequency images classification scenario of radar signals. We first propose an attack pipeline under the time-frequency images scenario and DITIMI-FGSM attack algorithm with high transferability. Then, we propose STFT-based time domain signal attack(STDS) algorithm to solve the problem of non-invertibility in time-frequency analysis, thus obtaining the time-domain representation of the interference signal. A large number of experiments show that our attack pipeline is feasible and the proposed attack method has a high success rate.

摘要: 雷达信号电子对抗是现代战争的一个重要方面。传统的电子对抗技术通常会添加大规模干扰信号来确保干扰效果，这可能会导致攻击过于明显。近年来，基于人工智能的攻击方法应运而生，可以有效地解决这一问题，但目前的攻击场景仅限于对雷达信号进行时域分类。本文主要研究雷达信号的时频图像分类方案。本文首先提出了一种时频图像场景下的攻击流水线和具有高可转移性的DITIMI-FGSM攻击算法。然后，针对时频分析中的不可逆性问题，提出了基于短时傅立叶变换的时域信号攻击(STDS)算法，从而得到了干扰信号的时域表示。大量实验表明，我们的攻击流水线是可行的，所提出的攻击方法具有较高的成功率。



## **15. Improving Fast Minimum-Norm Attacks with Hyperparameter Optimization**

利用超参数优化改进快速最小范数攻击 cs.LG

Accepted at ESANN23

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.08177v1) [paper-pdf](http://arxiv.org/pdf/2310.08177v1)

**Authors**: Giuseppe Floris, Raffaele Mura, Luca Scionis, Giorgio Piras, Maura Pintor, Ambra Demontis, Battista Biggio

**Abstract**: Evaluating the adversarial robustness of machine learning models using gradient-based attacks is challenging. In this work, we show that hyperparameter optimization can improve fast minimum-norm attacks by automating the selection of the loss function, the optimizer and the step-size scheduler, along with the corresponding hyperparameters. Our extensive evaluation involving several robust models demonstrates the improved efficacy of fast minimum-norm attacks when hyper-up with hyperparameter optimization. We release our open-source code at https://github.com/pralab/HO-FMN.

摘要: 使用基于梯度的攻击来评估机器学习模型的对抗健壮性是具有挑战性的。在这项工作中，我们证明了超参数优化可以通过自动选择损失函数、优化器和步长调度器以及相应的超参数来改善快速最小范数攻击。我们对几个稳健模型的广泛评估表明，当超参数优化超高速时，快速最小范数攻击的效率得到了提高。我们在https://github.com/pralab/HO-FMN.上发布我们的开源代码



## **16. Samples on Thin Ice: Re-Evaluating Adversarial Pruning of Neural Networks**

薄冰上的样本：重新评估神经网络的对抗性剪枝 cs.LG

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.08073v1) [paper-pdf](http://arxiv.org/pdf/2310.08073v1)

**Authors**: Giorgio Piras, Maura Pintor, Ambra Demontis, Battista Biggio

**Abstract**: Neural network pruning has shown to be an effective technique for reducing the network size, trading desirable properties like generalization and robustness to adversarial attacks for higher sparsity. Recent work has claimed that adversarial pruning methods can produce sparse networks while also preserving robustness to adversarial examples. In this work, we first re-evaluate three state-of-the-art adversarial pruning methods, showing that their robustness was indeed overestimated. We then compare pruned and dense versions of the same models, discovering that samples on thin ice, i.e., closer to the unpruned model's decision boundary, are typically misclassified after pruning. We conclude by discussing how this intuition may lead to designing more effective adversarial pruning methods in future work.

摘要: 神经网络修剪已被证明是一种有效的减小网络规模的技术，它以泛化和对敌对攻击的健壮性为代价来换取更高的稀疏性。最近的工作声称，对抗性剪枝方法可以产生稀疏网络，同时还可以保持对对抗性示例的健壮性。在这项工作中，我们首先重新评估了三种最新的对抗性剪枝方法，表明它们的健壮性确实被高估了。然后，我们比较相同模型的修剪和稠密版本，发现在修剪后，薄冰上的样本通常被错误分类，即更接近未修剪的模型的决策边界。最后，我们讨论了这种直觉如何在未来的工作中导致设计更有效的对抗性剪枝方法。



## **17. Block Coordinate Descent on Smooth Manifolds: Convergence Theory and Twenty-One Examples**

光滑流形上的块坐标下降：收敛理论及21个实例 math.OC

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2305.14744v3) [paper-pdf](http://arxiv.org/pdf/2305.14744v3)

**Authors**: Liangzu Peng, René Vidal

**Abstract**: Block coordinate descent is an optimization paradigm that iteratively updates one block of variables at a time, making it quite amenable to big data applications due to its scalability and performance. Its convergence behavior has been extensively studied in the (block-wise) convex case, but it is much less explored in the non-convex case. In this paper we analyze the convergence of block coordinate methods on non-convex sets and derive convergence rates on smooth manifolds under natural or weaker assumptions than prior work. Our analysis applies to many non-convex problems, including ones that seek low-dimensional structures (e.g., maximal coding rate reduction, neural collapse, reverse engineering adversarial attacks, generalized PCA, alternating projection); ones that seek combinatorial structures (homomorphic sensing, regression without correspondences, real phase retrieval, robust point matching); ones that seek geometric structures from visual data (e.g., essential matrix estimation, absolute pose estimation); and ones that seek inliers sparsely hidden in a large number of outliers (e.g., outlier-robust estimation via iteratively-reweighted least-squares). While our convergence theory applies to all these problems, yielding novel corollaries, it also applies to other, perhaps more familiar, problems (e.g., optimal transport, matrix factorization, Burer-Monteiro factorization), recovering previously known results.

摘要: 块坐标下降是一种每次迭代更新一个变量块的优化范例，由于其可扩展性和性能，使其非常适合大数据应用。它的收敛行为在(分块)凸的情况下已经被广泛地研究，但在非凸的情况下的研究要少得多。本文分析了块坐标方法在非凸集上的收敛，并在自然或弱于已有工作的假设下，得到了光滑流形上的收敛速度。我们的分析适用于许多非凸问题，包括寻求低维结构的问题(例如，最大编码率降低、神经崩溃、反向工程对抗性攻击、广义PCA、交替投影)；寻求组合结构的问题(同态检测、无对应回归、实相位恢复、鲁棒点匹配)；从视觉数据中寻找几何结构的问题(例如，基本矩阵估计、绝对姿态估计)；以及寻找稀疏隐藏在大量离群点中的内点的问题(例如，通过迭代重加权最小二乘估计的离群点稳健估计)。虽然我们的收敛理论适用于所有这些问题，产生了新的推论，但它也适用于其他可能更熟悉的问题(例如，最优传输、矩阵因式分解、布里-蒙泰罗因式分解)，恢复了以前已知的结果。



## **18. Why Train More? Effective and Efficient Membership Inference via Memorization**

为什么要多开火车？基于记忆的高效隶属度推理 cs.LG

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.08015v1) [paper-pdf](http://arxiv.org/pdf/2310.08015v1)

**Authors**: Jihye Choi, Shruti Tople, Varun Chandrasekaran, Somesh Jha

**Abstract**: Membership Inference Attacks (MIAs) aim to identify specific data samples within the private training dataset of machine learning models, leading to serious privacy violations and other sophisticated threats. Many practical black-box MIAs require query access to the data distribution (the same distribution where the private data is drawn) to train shadow models. By doing so, the adversary obtains models trained "with" or "without" samples drawn from the distribution, and analyzes the characteristics of the samples under consideration. The adversary is often required to train more than hundreds of shadow models to extract the signals needed for MIAs; this becomes the computational overhead of MIAs. In this paper, we propose that by strategically choosing the samples, MI adversaries can maximize their attack success while minimizing the number of shadow models. First, our motivational experiments suggest memorization as the key property explaining disparate sample vulnerability to MIAs. We formalize this through a theoretical bound that connects MI advantage with memorization. Second, we show sample complexity bounds that connect the number of shadow models needed for MIAs with memorization. Lastly, we confirm our theoretical arguments with comprehensive experiments; by utilizing samples with high memorization scores, the adversary can (a) significantly improve its efficacy regardless of the MIA used, and (b) reduce the number of shadow models by nearly two orders of magnitude compared to state-of-the-art approaches.

摘要: 成员关系推断攻击(MIA)旨在识别机器学习模型的私有训练数据集中的特定数据样本，导致严重侵犯隐私和其他复杂威胁。许多实际的黑盒MIA需要对数据分布(绘制私有数据的同一分布)的查询访问，以训练阴影模型。通过这样做，敌手获得从分布中提取的具有或不具有样本的训练模型，并分析所考虑的样本的特征。对手经常需要训练数百个以上的阴影模型来提取MIA所需的信号，这就成为MIA的计算开销。在本文中，我们提出通过策略性地选择样本，MI对手可以在最小化影子模型的数量的同时最大化他们的攻击成功。首先，我们的动机实验表明，记忆是解释不同样本对MIA脆弱性的关键属性。我们通过将MI优势与记忆联系起来的理论界限来形式化这一点。其次，我们给出了样本复杂性的界，它将MIA所需的阴影模型的数量与记忆联系起来。最后，我们通过全面的实验验证了我们的理论观点；通过使用高记忆分数的样本，对手可以(A)显著提高其有效性，而无论使用何种MIA，以及(B)与最先进的方法相比，阴影模型的数量减少了近两个数量级。



## **19. GRASP: Accelerating Shortest Path Attacks via Graph Attention**

GRAPH：通过图注意力加速最短路径攻击 cs.LG

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.07980v1) [paper-pdf](http://arxiv.org/pdf/2310.07980v1)

**Authors**: Zohair Shafi. Benjamin A. Miller, Ayan Chatterjee, Tina Eliassi-Rad, Rajmonda S. Caceres

**Abstract**: Recent advances in machine learning (ML) have shown promise in aiding and accelerating classical combinatorial optimization algorithms. ML-based speed ups that aim to learn in an end to end manner (i.e., directly output the solution) tend to trade off run time with solution quality. Therefore, solutions that are able to accelerate existing solvers while maintaining their performance guarantees, are of great interest. We consider an APX-hard problem, where an adversary aims to attack shortest paths in a graph by removing the minimum number of edges. We propose the GRASP algorithm: Graph Attention Accelerated Shortest Path Attack, an ML aided optimization algorithm that achieves run times up to 10x faster, while maintaining the quality of solution generated. GRASP uses a graph attention network to identify a smaller subgraph containing the combinatorial solution, thus effectively reducing the input problem size. Additionally, we demonstrate how careful representation of the input graph, including node features that correlate well with the optimization task, can highlight important structure in the optimization solution.

摘要: 机器学习(ML)的最新进展在辅助和加速经典组合优化算法方面显示出良好的前景。基于ML的加速旨在以端到端的方式学习(即直接输出解决方案)，往往会在运行时间和解决方案质量之间进行权衡。因此，能够在保持现有求解器性能保证的同时加速现有求解器的解决方案是非常有意义的。我们考虑APX-Hard问题，其中对手的目标是通过删除最少的边数来攻击图中的最短路径。我们提出了GRASH算法：图注意加速最短路径攻击，这是一种ML辅助优化算法，在保持所生成解的质量的情况下，运行时间最多可以提高10倍。GRASH使用图注意网络来识别包含组合解的较小的子图，从而有效地减少了输入问题的规模。此外，我们还演示了如何仔细表示输入图，包括与优化任务关联良好的节点特征，以突出优化解决方案中的重要结构。



## **20. Multi-SpacePhish: Extending the Evasion-space of Adversarial Attacks against Phishing Website Detectors using Machine Learning**

多SpacePhish：利用机器学习扩展针对钓鱼网站检测器的敌意攻击的规避空间 cs.CR

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2210.13660v3) [paper-pdf](http://arxiv.org/pdf/2210.13660v3)

**Authors**: Ying Yuan, Giovanni Apruzzese, Mauro Conti

**Abstract**: Existing literature on adversarial Machine Learning (ML) focuses either on showing attacks that break every ML model, or defenses that withstand most attacks. Unfortunately, little consideration is given to the actual feasibility of the attack or the defense. Moreover, adversarial samples are often crafted in the "feature-space", making the corresponding evaluations of questionable value. Simply put, the current situation does not allow to estimate the actual threat posed by adversarial attacks, leading to a lack of secure ML systems.   We aim to clarify such confusion in this paper. By considering the application of ML for Phishing Website Detection (PWD), we formalize the "evasion-space" in which an adversarial perturbation can be introduced to fool a ML-PWD -- demonstrating that even perturbations in the "feature-space" are useful. Then, we propose a realistic threat model describing evasion attacks against ML-PWD that are cheap to stage, and hence intrinsically more attractive for real phishers. After that, we perform the first statistically validated assessment of state-of-the-art ML-PWD against 12 evasion attacks. Our evaluation shows (i) the true efficacy of evasion attempts that are more likely to occur; and (ii) the impact of perturbations crafted in different evasion-spaces. Our realistic evasion attempts induce a statistically significant degradation (3-10% at p<0.05), and their cheap cost makes them a subtle threat. Notably, however, some ML-PWD are immune to our most realistic attacks (p=0.22).   Finally, as an additional contribution of this journal publication, we are the first to consider the intriguing case wherein an attacker introduces perturbations in multiple evasion-spaces at the same time. These new results show that simultaneously applying perturbations in the problem- and feature-space can cause a drop in the detection rate from 0.95 to 0.

摘要: 现有的关于对抗性机器学习(ML)的文献要么专注于展示打破每个ML模型的攻击，要么专注于抵御大多数攻击的防御。不幸的是，几乎没有考虑到攻击或防御的实际可行性。此外，对抗性样本往往是在“特征空间”中制作的，使得相应的评估价值值得怀疑。简单地说，目前的情况不允许估计对抗性攻击构成的实际威胁，导致缺乏安全的ML系统。我们的目的是在这篇论文中澄清这种混淆。通过考虑ML在钓鱼网站检测(PWD)中的应用，我们形式化了“规避空间”，在该空间中可以引入敌意扰动来愚弄ML-PWD--表明即使在“特征空间”中的扰动也是有用的。然后，我们提出了一个真实的威胁模型，描述了针对ML-PWD的逃避攻击，这些攻击的实施成本很低，因此本质上对真正的网络钓鱼者更具吸引力。之后，我们对最先进的ML-PWD进行了第一次统计验证评估，以对抗12次逃避攻击。我们的评估显示了(I)更有可能发生的逃避尝试的真实效果；以及(Ii)在不同的逃避空间中制造的扰动的影响。我们的现实规避尝试导致了统计上的显著下降(3%-10%，p<0.05)，而且它们的廉价成本使它们成为一个微妙的威胁。然而，值得注意的是，一些ML-PWD对我们最现实的攻击是免疫的(p=0.22)。最后，作为这本期刊出版物的另一贡献，我们第一次考虑了一个有趣的情况，其中攻击者同时在多个躲避空间中引入扰动。这些新的结果表明，同时在问题空间和特征空间中施加扰动可以导致检测率从0.95下降到0。



## **21. Deep Reinforcement Learning for Autonomous Cyber Operations: A Survey**

深度强化学习在自主网络操作中的研究进展 cs.LG

60 pages, 14 figures, 3 tables

**SubmitDate**: 2023-10-11    [abs](http://arxiv.org/abs/2310.07745v1) [paper-pdf](http://arxiv.org/pdf/2310.07745v1)

**Authors**: Gregory Palmer, Chris Parry, Daniel J. B. Harrold, Chris Willis

**Abstract**: The rapid increase in the number of cyber-attacks in recent years raises the need for principled methods for defending networks against malicious actors. Deep reinforcement learning (DRL) has emerged as a promising approach for mitigating these attacks. However, while DRL has shown much potential for cyber-defence, numerous challenges must be overcome before DRL can be applied to autonomous cyber-operations (ACO) at scale. Principled methods are required for environments that confront learners with very high-dimensional state spaces, large multi-discrete action spaces, and adversarial learning. Recent works have reported success in solving these problems individually. There have also been impressive engineering efforts towards solving all three for real-time strategy games. However, applying DRL to the full ACO problem remains an open challenge. Here, we survey the relevant DRL literature and conceptualize an idealised ACO-DRL agent. We provide: i.) A summary of the domain properties that define the ACO problem; ii.) A comprehensive evaluation of the extent to which domains used for benchmarking DRL approaches are comparable to ACO; iii.) An overview of state-of-the-art approaches for scaling DRL to domains that confront learners with the curse of dimensionality, and; iv.) A survey and critique of current methods for limiting the exploitability of agents within adversarial settings from the perspective of ACO. We conclude with open research questions that we hope will motivate future directions for researchers and practitioners working on ACO.

摘要: 近年来，网络攻击数量的迅速增加增加了对保护网络免受恶意行为侵害的原则性方法的需求。深度强化学习(DRL)已成为缓解这些攻击的一种很有前途的方法。然而，尽管DRL在网络防御方面显示出了很大的潜力，但在DRL能够大规模应用于自主网络作战(ACO)之前，必须克服许多挑战。对于学习者面对高维状态空间、大的多离散动作空间和对抗性学习的环境，需要有原则性的方法。最近的研究报告成功地单独解决了这些问题。也有令人印象深刻的工程努力，以解决所有这三个实时战略游戏。然而，将DRL应用于整个蚁群优化问题仍然是一个开放的挑战。在这里，我们回顾了相关的DRL文献，并概念化了一个理想的ACO-DRL试剂。我们提供：i.)定义ACO问题的域属性摘要；ii.)对用于基准DRL方法的领域与ACO的可比性程度进行全面评估；3.概述将DRL扩展到学习者面临维度诅咒的领域的最新方法，以及；i.)从蚁群算法的角度对当前在对抗性环境中限制代理的可利用性的方法进行了调查和评论。我们以开放的研究问题结束，我们希望这些问题将激励从事ACO工作的研究人员和从业者未来的方向。



## **22. Boosting Black-box Attack to Deep Neural Networks with Conditional Diffusion Models**

利用条件扩散模型增强对深度神经网络的黑盒攻击 cs.CV

**SubmitDate**: 2023-10-11    [abs](http://arxiv.org/abs/2310.07492v1) [paper-pdf](http://arxiv.org/pdf/2310.07492v1)

**Authors**: Renyang Liu, Wei Zhou, Tianwei Zhang, Kangjie Chen, Jun Zhao, Kwok-Yan Lam

**Abstract**: Existing black-box attacks have demonstrated promising potential in creating adversarial examples (AE) to deceive deep learning models. Most of these attacks need to handle a vast optimization space and require a large number of queries, hence exhibiting limited practical impacts in real-world scenarios. In this paper, we propose a novel black-box attack strategy, Conditional Diffusion Model Attack (CDMA), to improve the query efficiency of generating AEs under query-limited situations. The key insight of CDMA is to formulate the task of AE synthesis as a distribution transformation problem, i.e., benign examples and their corresponding AEs can be regarded as coming from two distinctive distributions and can transform from each other with a particular converter. Unlike the conventional \textit{query-and-optimization} approach, we generate eligible AEs with direct conditional transform using the aforementioned data converter, which can significantly reduce the number of queries needed. CDMA adopts the conditional Denoising Diffusion Probabilistic Model as the converter, which can learn the transformation from clean samples to AEs, and ensure the smooth development of perturbed noise resistant to various defense strategies. We demonstrate the effectiveness and efficiency of CDMA by comparing it with nine state-of-the-art black-box attacks across three benchmark datasets. On average, CDMA can reduce the query count to a handful of times; in most cases, the query count is only ONE. We also show that CDMA can obtain $>99\%$ attack success rate for untarget attacks over all datasets and targeted attack over CIFAR-10 with the noise budget of $\epsilon=16$.

摘要: 现有的黑盒攻击在创建对抗实例(AE)以欺骗深度学习模型方面显示出很好的潜力。这些攻击大多需要处理巨大的优化空间，需要大量的查询，因此在现实场景中表现出的实际影响有限。本文提出了一种新的黑盒攻击策略--条件扩散模型攻击(CDMAT)，以提高在查询受限情况下生成实体实体的查询效率。码分多址的关键思想是将声发射综合问题描述为一个分布变换问题，即良性例子及其对应的声学效应可以被认为来自两个不同的分布，并且可以通过一个特定的转换器相互转换。与传统的查询和优化方法不同，我们使用前面提到的数据转换器通过直接条件转换生成符合条件的AE，这可以显著减少所需的查询数量。码分多址采用条件去噪扩散概率模型作为转换器，可以学习从清洁样本到声学估计的转换，并确保抗各种防御策略的扰动噪声的顺利发展。我们通过在三个基准数据集上与九种最先进的黑盒攻击进行比较，证明了CDMA的有效性和高效性。平均而言，CDMA可以将查询计数减少到几倍；在大多数情况下，查询计数只有一次。我们还表明，对于所有数据集上的非目标攻击和CIFAR-10上的目标攻击，在噪声预算为$epsilon=16$的情况下，该算法可以获得>99$的攻击成功率。



## **23. Fundamental Limitations of Alignment in Large Language Models**

大型语言模型中对齐的基本限制 cs.CL

**SubmitDate**: 2023-10-11    [abs](http://arxiv.org/abs/2304.11082v4) [paper-pdf](http://arxiv.org/pdf/2304.11082v4)

**Authors**: Yotam Wolf, Noam Wies, Oshri Avnery, Yoav Levine, Amnon Shashua

**Abstract**: An important aspect in developing language models that interact with humans is aligning their behavior to be useful and unharmful for their human users. This is usually achieved by tuning the model in a way that enhances desired behaviors and inhibits undesired ones, a process referred to as alignment. In this paper, we propose a theoretical approach called Behavior Expectation Bounds (BEB) which allows us to formally investigate several inherent characteristics and limitations of alignment in large language models. Importantly, we prove that within the limits of this framework, for any behavior that has a finite probability of being exhibited by the model, there exist prompts that can trigger the model into outputting this behavior, with probability that increases with the length of the prompt. This implies that any alignment process that attenuates an undesired behavior but does not remove it altogether, is not safe against adversarial prompting attacks. Furthermore, our framework hints at the mechanism by which leading alignment approaches such as reinforcement learning from human feedback make the LLM prone to being prompted into the undesired behaviors. This theoretical result is being experimentally demonstrated in large scale by the so called contemporary "chatGPT jailbreaks", where adversarial users trick the LLM into breaking its alignment guardrails by triggering it into acting as a malicious persona. Our results expose fundamental limitations in alignment of LLMs and bring to the forefront the need to devise reliable mechanisms for ensuring AI safety.

摘要: 开发与人类交互的语言模型的一个重要方面是使他们的行为对人类用户有用而无害。这通常是通过调整模型来实现的，这种方式增强了期望的行为，抑制了不期望的行为，这一过程称为对齐。在本文中，我们提出了一种名为行为期望界限(BEB)的理论方法，它允许我们正式地研究大型语言模型中对齐的几个固有特征和限制。重要的是，我们证明了在这个框架的范围内，对于模型所表现出的任何有限概率的行为，存在可以触发模型输出该行为的提示，其概率随着提示的长度的增加而增加。这意味着，任何减弱不受欢迎的行为但不能完全消除它的对准过程，在对抗提示攻击时都是不安全的。此外，我们的框架暗示了一种机制，通过这种机制，领先的对齐方法，如来自人类反馈的强化学习，使得LLM容易被提示进入不希望看到的行为。这一理论结果正在由所谓的当代“聊天GPT越狱”大规模实验证明，在这种情况下，敌对用户通过触发LLM充当恶意角色来欺骗LLM打破其对齐护栏。我们的结果暴露了LLM对齐方面的根本限制，并将设计可靠的机制以确保人工智能安全的必要性放在了首位。



## **24. My Brother Helps Me: Node Injection Based Adversarial Attack on Social Bot Detection**

兄弟帮我：社交机器人检测中基于节点注入的对抗性攻击 cs.CR

**SubmitDate**: 2023-10-11    [abs](http://arxiv.org/abs/2310.07159v1) [paper-pdf](http://arxiv.org/pdf/2310.07159v1)

**Authors**: Lanjun Wang, Xinran Qiao, Yanwei Xie, Weizhi Nie, Yongdong Zhang, Anan Liu

**Abstract**: Social platforms such as Twitter are under siege from a multitude of fraudulent users. In response, social bot detection tasks have been developed to identify such fake users. Due to the structure of social networks, the majority of methods are based on the graph neural network(GNN), which is susceptible to attacks. In this study, we propose a node injection-based adversarial attack method designed to deceive bot detection models. Notably, neither the target bot nor the newly injected bot can be detected when a new bot is added around the target bot. This attack operates in a black-box fashion, implying that any information related to the victim model remains unknown. To our knowledge, this is the first study exploring the resilience of bot detection through graph node injection. Furthermore, we develop an attribute recovery module to revert the injected node embedding from the graph embedding space back to the original feature space, enabling the adversary to manipulate node perturbation effectively. We conduct adversarial attacks on four commonly used GNN structures for bot detection on two widely used datasets: Cresci-2015 and TwiBot-22. The attack success rate is over 73\% and the rate of newly injected nodes being detected as bots is below 13\% on these two datasets.

摘要: Twitter等社交平台正受到大量欺诈性用户的围攻。作为回应，社交机器人检测任务已经被开发出来，以识别此类虚假用户。由于社会网络结构的原因，大多数方法都是基于图神经网络(GNN)的，容易受到攻击。在本研究中，我们提出了一种基于节点注入的对抗性攻击方法，旨在欺骗BOT检测模型。值得注意的是，当在目标机器人周围添加新的机器人时，既检测不到目标机器人，也检测不到新注入的机器人。这一攻击是以黑盒方式进行的，这意味着与受害者模型有关的任何信息仍然未知。据我们所知，这是第一次通过图节点注入来探索机器人检测的弹性的研究。此外，我们开发了一个属性恢复模块，将注入的节点嵌入从图嵌入空间还原到原始特征空间，使攻击者能够有效地操纵节点扰动。我们在两个广泛使用的数据集：CRISI-2015和TwiBot-22上对四种常用的GNN结构进行了对抗性攻击，以进行机器人检测。在这两个数据集上，攻击成功率超过73%，新注入节点被检测为僵尸的比率低于13%。



## **25. No Privacy Left Outside: On the (In-)Security of TEE-Shielded DNN Partition for On-Device ML**

没有隐私留在外面：论设备上ML的TEE屏蔽DNN分区的(In-)安全性 cs.CR

Accepted by S&P'24

**SubmitDate**: 2023-10-11    [abs](http://arxiv.org/abs/2310.07152v1) [paper-pdf](http://arxiv.org/pdf/2310.07152v1)

**Authors**: Ziqi Zhang, Chen Gong, Yifeng Cai, Yuanyuan Yuan, Bingyan Liu, Ding Li, Yao Guo, Xiangqun Chen

**Abstract**: On-device ML introduces new security challenges: DNN models become white-box accessible to device users. Based on white-box information, adversaries can conduct effective model stealing (MS) and membership inference attack (MIA). Using Trusted Execution Environments (TEEs) to shield on-device DNN models aims to downgrade (easy) white-box attacks to (harder) black-box attacks. However, one major shortcoming is the sharply increased latency (up to 50X). To accelerate TEE-shield DNN computation with GPUs, researchers proposed several model partition techniques. These solutions, referred to as TEE-Shielded DNN Partition (TSDP), partition a DNN model into two parts, offloading the privacy-insensitive part to the GPU while shielding the privacy-sensitive part within the TEE. This paper benchmarks existing TSDP solutions using both MS and MIA across a variety of DNN models, datasets, and metrics. We show important findings that existing TSDP solutions are vulnerable to privacy-stealing attacks and are not as safe as commonly believed. We also unveil the inherent difficulty in deciding optimal DNN partition configurations (i.e., the highest security with minimal utility cost) for present TSDP solutions. The experiments show that such ``sweet spot'' configurations vary across datasets and models. Based on lessons harvested from the experiments, we present TEESlice, a novel TSDP method that defends against MS and MIA during DNN inference. TEESlice follows a partition-before-training strategy, which allows for accurate separation between privacy-related weights from public weights. TEESlice delivers the same security protection as shielding the entire DNN model inside TEE (the ``upper-bound'' security guarantees) with over 10X less overhead (in both experimental and real-world environments) than prior TSDP solutions and no accuracy loss.

摘要: On-Device ML带来了新的安全挑战：DNN型号成为设备用户可访问的白盒。基于白盒信息，攻击者可以进行有效的模型窃取(MS)和成员推理攻击(MIA)。使用可信执行环境(TEE)来屏蔽设备上的DNN模型旨在将(容易的)白盒攻击降级为(更难的)黑盒攻击。然而，一个主要缺点是延迟急剧增加(高达50倍)。为了在图形处理器上加速T形屏蔽DNN的计算，研究人员提出了几种模型划分技术。这些解决方案称为TEE屏蔽DNN分区(TSDP)，将DNN模型划分为两个部分，将隐私不敏感部分卸载到GPU，同时屏蔽TEE内的隐私敏感部分。本白皮书使用MS和MIA在各种DNN模型、数据集和指标上对现有TSDP解决方案进行基准测试。我们的重要发现表明，现有的TSDP解决方案容易受到隐私窃取攻击，并且不像人们通常认为的那样安全。我们还揭示了为目前的TSDP解决方案确定最优DNN分区配置(即，以最小的效用成本获得最高的安全性)的固有困难。实验表明，这样的“甜蜜点”配置在数据集和模型中有所不同。在总结实验经验的基础上，提出了一种在DNN推理过程中防御MS和MIA的TSDP方法TEESlice。TEESlice遵循先划分后训练的策略，允许将与隐私相关的权重与公共权重精确分离。TEESlice提供了与在TEE内部屏蔽整个DNN模型相同的安全保护(“上限”安全保证)，与以前的TSDP解决方案相比，开销(在实验和现实环境中)减少了10倍以上，并且没有精度损失。



## **26. Investigating the Adversarial Robustness of Density Estimation Using the Probability Flow ODE**

用概率流模型研究密度估计的对抗稳健性 cs.LG

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.07084v1) [paper-pdf](http://arxiv.org/pdf/2310.07084v1)

**Authors**: Marius Arvinte, Cory Cornelius, Jason Martin, Nageen Himayat

**Abstract**: Beyond their impressive sampling capabilities, score-based diffusion models offer a powerful analysis tool in the form of unbiased density estimation of a query sample under the training data distribution. In this work, we investigate the robustness of density estimation using the probability flow (PF) neural ordinary differential equation (ODE) model against gradient-based likelihood maximization attacks and the relation to sample complexity, where the compressed size of a sample is used as a measure of its complexity. We introduce and evaluate six gradient-based log-likelihood maximization attacks, including a novel reverse integration attack. Our experimental evaluations on CIFAR-10 show that density estimation using the PF ODE is robust against high-complexity, high-likelihood attacks, and that in some cases adversarial samples are semantically meaningful, as expected from a robust estimator.

摘要: 除了令人印象深刻的抽样能力之外，基于分数的扩散模型还提供了一种强大的分析工具，其形式是对训练数据分布下的查询样本进行无偏密度估计。在这项工作中，我们研究了概率流(PF)神经常微分方程(ODE)模型的密度估计对基于梯度的似然最大化攻击的稳健性以及与样本复杂性的关系，其中样本的压缩大小作为其复杂性的度量。我们介绍并评估了六种基于梯度的对数似然最大化攻击，其中包括一种新的反向积分攻击。我们在CIFAR-10上的实验评估表明，使用PF ODE的密度估计对高复杂性、高似然攻击具有健壮性，并且在某些情况下，对抗性样本具有语义意义，正如健壮估计器所期望的那样。



## **27. Jailbreak in pieces: Compositional Adversarial Attacks on Multi-Modal Language Models**

片断越狱：对多通道语言模型的组合对抗性攻击 cs.CR

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2307.14539v2) [paper-pdf](http://arxiv.org/pdf/2307.14539v2)

**Authors**: Erfan Shayegani, Yue Dong, Nael Abu-Ghazaleh

**Abstract**: We introduce new jailbreak attacks on vision language models (VLMs), which use aligned LLMs and are resilient to text-only jailbreak attacks. Specifically, we develop cross-modality attacks on alignment where we pair adversarial images going through the vision encoder with textual prompts to break the alignment of the language model. Our attacks employ a novel compositional strategy that combines an image, adversarially targeted towards toxic embeddings, with generic prompts to accomplish the jailbreak. Thus, the LLM draws the context to answer the generic prompt from the adversarial image. The generation of benign-appearing adversarial images leverages a novel embedding-space-based methodology, operating with no access to the LLM model. Instead, the attacks require access only to the vision encoder and utilize one of our four embedding space targeting strategies. By not requiring access to the LLM, the attacks lower the entry barrier for attackers, particularly when vision encoders such as CLIP are embedded in closed-source LLMs. The attacks achieve a high success rate across different VLMs, highlighting the risk of cross-modality alignment vulnerabilities, and the need for new alignment approaches for multi-modal models.

摘要: 我们在视觉语言模型(VLM)上引入了新的越狱攻击，该模型使用对齐的LLM，并且对纯文本越狱攻击具有弹性。具体地说，我们开发了对齐的跨通道攻击，我们将通过视觉编码器的对抗性图像与文本提示配对，以打破语言模型的对齐。我们的攻击采用了一种新颖的组合策略，将恶意针对有毒嵌入的图像与通用提示相结合，以完成越狱。因此，LLM从对抗性图像提取上下文以回答通用提示。良性的对抗性图像的生成利用了一种新的基于嵌入空间的方法，操作时不需要访问LLM模型。取而代之的是，这些攻击只需要访问视觉编码器，并利用我们的四种嵌入空间目标策略之一。通过不需要访问LLM，攻击降低了攻击者的进入门槛，特别是当视觉编码器(如CLIP)嵌入闭源LLM中时。这些攻击在不同的VLM上实现了高成功率，突显了跨通道对齐漏洞的风险，以及对多通道模型的新对齐方法的需求。



## **28. TDPP: Two-Dimensional Permutation-Based Protection of Memristive Deep Neural Networks**

TDPP：基于二维置换的记忆深度神经网络保护 cs.CR

14 pages, 11 figures

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06989v1) [paper-pdf](http://arxiv.org/pdf/2310.06989v1)

**Authors**: Minhui Zou, Zhenhua Zhu, Tzofnat Greenberg-Toledo, Orian Leitersdorf, Jiang Li, Junlong Zhou, Yu Wang, Nan Du, Shahar Kvatinsky

**Abstract**: The execution of deep neural network (DNN) algorithms suffers from significant bottlenecks due to the separation of the processing and memory units in traditional computer systems. Emerging memristive computing systems introduce an in situ approach that overcomes this bottleneck. The non-volatility of memristive devices, however, may expose the DNN weights stored in memristive crossbars to potential theft attacks. Therefore, this paper proposes a two-dimensional permutation-based protection (TDPP) method that thwarts such attacks. We first introduce the underlying concept that motivates the TDPP method: permuting both the rows and columns of the DNN weight matrices. This contrasts with previous methods, which focused solely on permuting a single dimension of the weight matrices, either the rows or columns. While it's possible for an adversary to access the matrix values, the original arrangement of rows and columns in the matrices remains concealed. As a result, the extracted DNN model from the accessed matrix values would fail to operate correctly. We consider two different memristive computing systems (designed for layer-by-layer and layer-parallel processing, respectively) and demonstrate the design of the TDPP method that could be embedded into the two systems. Finally, we present a security analysis. Our experiments demonstrate that TDPP can achieve comparable effectiveness to prior approaches, with a high level of security when appropriately parameterized. In addition, TDPP is more scalable than previous methods and results in reduced area and power overheads. The area and power are reduced by, respectively, 1218$\times$ and 2815$\times$ for the layer-by-layer system and by 178$\times$ and 203$\times$ for the layer-parallel system compared to prior works.

摘要: 在传统的计算机系统中，由于处理单元和存储单元的分离，深度神经网络(DNN)算法的执行受到了严重的瓶颈。新兴的记忆计算系统引入了一种原位方法来克服这一瓶颈。然而，记忆装置的非易失性可能会使存储在记忆交叉杆中的DNN权重暴露于潜在的盗窃攻击。因此，本文提出了一种基于二维置换的保护方法(TDPP)来阻止此类攻击。我们首先介绍了激励TDPP方法的基本概念：排列DNN权重矩阵的行和列。这与以前的方法不同，以前的方法只专注于排列单一维度的权重矩阵，要么是行，要么是列。虽然对手有可能访问矩阵值，但矩阵中行和列的原始排列仍然是隐藏的。结果，从访问的矩阵值中提取的DNN模型将不能正确操作。我们考虑了两个不同的记忆计算系统(分别设计用于逐层处理和层并行处理)，并演示了可以嵌入到这两个系统中的TDPP方法的设计。最后，我们给出了安全性分析。我们的实验表明，当适当的参数设置时，TDPP可以达到与以前的方法相当的效果，并且具有很高的安全性。此外，与以前的方法相比，TDPP具有更强的可扩展性，并且减少了面积和功率开销。与已有工作相比，逐层系统的面积和功耗分别减少了1218美元和2815美元，而层并行系统则分别减少了178美元和203美元。



## **29. Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation**

开源LLMS通过利用生成进行灾难性越狱 cs.CL

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06987v1) [paper-pdf](http://arxiv.org/pdf/2310.06987v1)

**Authors**: Yangsibo Huang, Samyak Gupta, Mengzhou Xia, Kai Li, Danqi Chen

**Abstract**: The rapid progress in open-source large language models (LLMs) is significantly advancing AI development. Extensive efforts have been made before model release to align their behavior with human values, with the primary goal of ensuring their helpfulness and harmlessness. However, even carefully aligned models can be manipulated maliciously, leading to unintended behaviors, known as "jailbreaks". These jailbreaks are typically triggered by specific text inputs, often referred to as adversarial prompts. In this work, we propose the generation exploitation attack, an extremely simple approach that disrupts model alignment by only manipulating variations of decoding methods. By exploiting different generation strategies, including varying decoding hyper-parameters and sampling methods, we increase the misalignment rate from 0% to more than 95% across 11 language models including LLaMA2, Vicuna, Falcon, and MPT families, outperforming state-of-the-art attacks with $30\times$ lower computational cost. Finally, we propose an effective alignment method that explores diverse generation strategies, which can reasonably reduce the misalignment rate under our attack. Altogether, our study underscores a major failure in current safety evaluation and alignment procedures for open-source LLMs, strongly advocating for more comprehensive red teaming and better alignment before releasing such models. Our code is available at https://github.com/Princeton-SysML/Jailbreak_LLM.

摘要: 开源大型语言模型(LLM)的快速发展极大地推动了人工智能的发展。在模型发布之前，已经做出了广泛的努力，以使它们的行为符合人类的价值观，主要目标是确保它们的帮助和无害。然而，即使是精心排列的模型也可能被恶意操纵，导致意外行为，即所谓的“越狱”。这些越狱通常由特定的文本输入触发，通常被称为对抗性提示。在这项工作中，我们提出了生成利用攻击，这是一种非常简单的方法，只需操作不同的解码方法就可以破坏模型对齐。通过使用不同的生成策略，包括不同的解码超参数和采样方法，我们将LLaMA2、Vicuna、Falcon和MPT家族等11种语言模型的错配率从0%提高到95%以上，以30倍的计算代价击败了最新的攻击。最后，我们提出了一种有效的匹配方法，该方法探索了不同的生成策略，可以合理地降低攻击下的错配率。总之，我们的研究强调了当前开源LLM安全评估和比对程序的一个重大失败，强烈主张在发布此类模型之前进行更全面的红色团队和更好的比对。我们的代码可以在https://github.com/Princeton-SysML/Jailbreak_LLM.上找到



## **30. Comparing the robustness of modern no-reference image- and video-quality metrics to adversarial attacks**

比较现代无参考图像和视频质量指标对敌方攻击的稳健性 cs.CV

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06958v1) [paper-pdf](http://arxiv.org/pdf/2310.06958v1)

**Authors**: Anastasia Antsiferova, Khaled Abud, Aleksandr Gushchin, Sergey Lavrushkin, Ekaterina Shumitskaya, Maksim Velikanov, Dmitriy Vatolin

**Abstract**: Nowadays neural-network-based image- and video-quality metrics show better performance compared to traditional methods. However, they also became more vulnerable to adversarial attacks that increase metrics' scores without improving visual quality. The existing benchmarks of quality metrics compare their performance in terms of correlation with subjective quality and calculation time. However, the adversarial robustness of image-quality metrics is also an area worth researching. In this paper, we analyse modern metrics' robustness to different adversarial attacks. We adopted adversarial attacks from computer vision tasks and compared attacks' efficiency against 15 no-reference image/video-quality metrics. Some metrics showed high resistance to adversarial attacks which makes their usage in benchmarks safer than vulnerable metrics. The benchmark accepts new metrics submissions for researchers who want to make their metrics more robust to attacks or to find such metrics for their needs. Try our benchmark using pip install robustness-benchmark.

摘要: 如今，基于神经网络的图像和视频质量度量显示出比传统方法更好的性能。然而，它们也变得更容易受到对抗性攻击，这些攻击增加了指标的分数，但没有改善视觉质量。现有的质量指标基准在与主观质量和计算时间的相关性方面比较它们的表现。然而，图像质量指标的对抗稳健性也是一个值得研究的领域。在本文中，我们分析了现代度量对不同对手攻击的稳健性。我们采用了来自计算机视觉任务的对抗性攻击，并将攻击效率与15个无参考图像/视频质量指标进行了比较。一些指标表现出对敌意攻击的高度抵抗力，这使得它们在基准中的使用比易受攻击的指标更安全。该基准接受新的指标提交给希望使其指标更具抗攻击能力或找到符合其需求的此类指标的研究人员。使用pip安装健壮性基准测试我们的基准测试。



## **31. Adversarial optimization leads to over-optimistic security-constrained dispatch, but sampling can help**

对抗性优化会导致过度乐观的安全约束调度，但抽样可以有所帮助 eess.SY

Accepted at NAPS 2023

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06956v1) [paper-pdf](http://arxiv.org/pdf/2310.06956v1)

**Authors**: Charles Dawson, Chuchu Fan

**Abstract**: To ensure safe, reliable operation of the electrical grid, we must be able to predict and mitigate likely failures. This need motivates the classic security-constrained AC optimal power flow (SCOPF) problem. SCOPF is commonly solved using adversarial optimization, where the dispatcher and an adversary take turns optimizing a robust dispatch and adversarial attack, respectively. We show that adversarial optimization is liable to severely overestimate the robustness of the optimized dispatch (when the adversary encounters a local minimum), leading the operator to falsely believe that their dispatch is secure.   To prevent this overconfidence, we develop a novel adversarial sampling approach that prioritizes diversity in the predicted attacks. We find that our method not only substantially improves the robustness of the optimized dispatch but also avoids overconfidence, accurately characterizing the likelihood of voltage collapse under a given threat model. We demonstrate a proof-of-concept on small-scale transmission systems with 14 and 57 nodes.

摘要: 为了确保电网安全、可靠地运行，我们必须能够预测和减轻可能发生的故障。这种需求激发了经典的安全约束交流最优潮流(SCOPF)问题。SCOPF通常使用对抗性优化来解决，其中调度器和对手轮流优化健壮的调度和对抗性攻击。我们表明，敌意优化容易严重高估优化调度的稳健性(当对手遇到局部极小值时)，导致运营商错误地认为他们的调度是安全的。为了防止这种过度自信，我们开发了一种新的对抗性抽样方法，该方法在预测的攻击中优先考虑多样性。我们发现，我们的方法不仅大大提高了优化调度的稳健性，而且避免了过度自信，准确地表征了给定威胁模型下电压崩溃的可能性。我们在具有14个和57个节点的小规模传输系统上演示了概念验证。



## **32. Graph-based methods coupled with specific distributional distances for adversarial attack detection**

基于图的结合特定分布距离的攻击检测方法 cs.LG

published in Neural Networks

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2306.00042v2) [paper-pdf](http://arxiv.org/pdf/2306.00042v2)

**Authors**: Dwight Nwaigwe, Lucrezia Carboni, Martial Mermillod, Sophie Achard, Michel Dojat

**Abstract**: Artificial neural networks are prone to being fooled by carefully perturbed inputs which cause an egregious misclassification. These \textit{adversarial} attacks have been the focus of extensive research. Likewise, there has been an abundance of research in ways to detect and defend against them. We introduce a novel approach of detection and interpretation of adversarial attacks from a graph perspective. For an input image, we compute an associated sparse graph using the layer-wise relevance propagation algorithm \cite{bach15}. Specifically, we only keep edges of the neural network with the highest relevance values. Three quantities are then computed from the graph which are then compared against those computed from the training set. The result of the comparison is a classification of the image as benign or adversarial. To make the comparison, two classification methods are introduced: 1) an explicit formula based on Wasserstein distance applied to the degree of node and 2) a logistic regression. Both classification methods produce strong results which lead us to believe that a graph-based interpretation of adversarial attacks is valuable.

摘要: 人工神经网络很容易被精心扰动的输入所愚弄，这会导致严重的错误分类。这些对抗性攻击一直是广泛研究的焦点。同样，在检测和防御它们的方法方面也进行了大量的研究。我们从图的角度介绍了一种新的检测和解释对抗性攻击的方法。对于输入图像，我们使用分层相关传播算法来计算关联稀疏图。具体地说，我们只保留具有最高相关值的神经网络的边缘。然后从图中计算三个量，然后将其与从训练集计算的量进行比较。比较的结果是将图像分类为良性的或敌对的。为了进行比较，引入了两种分类方法：1)基于Wasserstein距离的节点程度的显式公式；2)Logistic回归。这两种分类方法都产生了很强的结果，这使得我们相信基于图的对抗性攻击的解释是有价值的。



## **33. Privacy-oriented manipulation of speaker representations**

面向隐私的说话人表征操作 eess.AS

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06652v1) [paper-pdf](http://arxiv.org/pdf/2310.06652v1)

**Authors**: Francisco Teixeira, Alberto Abad, Bhiksha Raj, Isabel Trancoso

**Abstract**: Speaker embeddings are ubiquitous, with applications ranging from speaker recognition and diarization to speech synthesis and voice anonymisation. The amount of information held by these embeddings lends them versatility, but also raises privacy concerns. Speaker embeddings have been shown to contain information on age, sex, health and more, which speakers may want to keep private, especially when this information is not required for the target task. In this work, we propose a method for removing and manipulating private attributes from speaker embeddings that leverages a Vector-Quantized Variational Autoencoder architecture, combined with an adversarial classifier and a novel mutual information loss. We validate our model on two attributes, sex and age, and perform experiments with ignorant and fully-informed attackers, and with in-domain and out-of-domain data.

摘要: 说话人嵌入是无处不在的，应用范围从说话人识别和二值化到语音合成和语音匿名化。这些嵌入的信息量使它们具有多功能性，但也引发了隐私问题。说话人嵌入已被证明包含年龄、性别、健康等信息，说话人可能希望对这些信息保密，特别是当目标任务不需要这些信息时。在这项工作中，我们提出了一种从说话人嵌入中去除和处理私人属性的方法，该方法利用矢量量化变分自动编码器结构，结合对抗性分类器和新的互信息损失。我们在性别和年龄两个属性上验证了我们的模型，并使用无知和完全知情的攻击者以及域内和域外的数据进行了实验。



## **34. A Geometrical Approach to Evaluate the Adversarial Robustness of Deep Neural Networks**

一种评估深度神经网络对抗健壮性的几何方法 cs.CV

ACM Transactions on Multimedia Computing, Communications, and  Applications (ACM TOMM)

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06468v1) [paper-pdf](http://arxiv.org/pdf/2310.06468v1)

**Authors**: Yang Wang, Bo Dong, Ke Xu, Haiyin Piao, Yufei Ding, Baocai Yin, Xin Yang

**Abstract**: Deep Neural Networks (DNNs) are widely used for computer vision tasks. However, it has been shown that deep models are vulnerable to adversarial attacks, i.e., their performances drop when imperceptible perturbations are made to the original inputs, which may further degrade the following visual tasks or introduce new problems such as data and privacy security. Hence, metrics for evaluating the robustness of deep models against adversarial attacks are desired. However, previous metrics are mainly proposed for evaluating the adversarial robustness of shallow networks on the small-scale datasets. Although the Cross Lipschitz Extreme Value for nEtwork Robustness (CLEVER) metric has been proposed for large-scale datasets (e.g., the ImageNet dataset), it is computationally expensive and its performance relies on a tractable number of samples. In this paper, we propose the Adversarial Converging Time Score (ACTS), an attack-dependent metric that quantifies the adversarial robustness of a DNN on a specific input. Our key observation is that local neighborhoods on a DNN's output surface would have different shapes given different inputs. Hence, given different inputs, it requires different time for converging to an adversarial sample. Based on this geometry meaning, ACTS measures the converging time as an adversarial robustness metric. We validate the effectiveness and generalization of the proposed ACTS metric against different adversarial attacks on the large-scale ImageNet dataset using state-of-the-art deep networks. Extensive experiments show that our ACTS metric is an efficient and effective adversarial metric over the previous CLEVER metric.

摘要: 深度神经网络被广泛应用于计算机视觉任务中。然而，已有研究表明，深层模型容易受到敌意攻击，即当原始输入受到不知不觉的扰动时，其性能会下降，这可能会进一步降低后续的视觉任务或带来新的问题，如数据和隐私安全。因此，需要评估深度模型对敌方攻击的稳健性的度量。然而，以前的度量主要是针对小规模数据集上的浅层网络的对抗健壮性提出的。虽然已经为大规模数据集(例如，ImageNet数据集)提出了网络健壮性的交叉Lipschitz极值(Clear)度量，但它的计算代价很高，并且其性能依赖于可处理的样本数量。在本文中，我们提出了对抗收敛时间得分(ACTS)，这是一种依赖于攻击的度量，它量化了DNN在特定输入上的对抗健壮性。我们的主要观察是，在给定不同输入的情况下，DNN输出表面上的局部邻域将具有不同的形状。因此，给定不同的输入，收敛到对抗性样本所需的时间也不同。基于这一几何意义，ACTS将收敛时间作为对抗性健壮性度量。我们使用最先进的深度网络在大规模ImageNet数据集上验证了所提出的ACTS度量针对不同对手攻击的有效性和泛化能力。大量的实验表明，我们的ACTS度量是一种比以前的聪明度量更有效的对抗性度量。



## **35. Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models**

红色团队博弈：红色团队语言模型的博弈论框架 cs.CL

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.00322v2) [paper-pdf](http://arxiv.org/pdf/2310.00322v2)

**Authors**: Chengdong Ma, Ziran Yang, Minquan Gao, Hai Ci, Jun Gao, Xuehai Pan, Yaodong Yang

**Abstract**: Deployable Large Language Models (LLMs) must conform to the criterion of helpfulness and harmlessness, thereby achieving consistency between LLMs outputs and human values. Red-teaming techniques constitute a critical way towards this criterion. Existing work rely solely on manual red team designs and heuristic adversarial prompts for vulnerability detection and optimization. These approaches lack rigorous mathematical formulation, thus limiting the exploration of diverse attack strategy within quantifiable measure and optimization of LLMs under convergence guarantees. In this paper, we present Red-teaming Game (RTG), a general game-theoretic framework without manual annotation. RTG is designed for analyzing the multi-turn attack and defense interactions between Red-team language Models (RLMs) and Blue-team Language Model (BLM). Within the RTG, we propose Gamified Red-teaming Solver (GRTS) with diversity measure of the semantic space. GRTS is an automated red teaming technique to solve RTG towards Nash equilibrium through meta-game analysis, which corresponds to the theoretically guaranteed optimization direction of both RLMs and BLM. Empirical results in multi-turn attacks with RLMs show that GRTS autonomously discovered diverse attack strategies and effectively improved security of LLMs, outperforming existing heuristic red-team designs. Overall, RTG has established a foundational framework for red teaming tasks and constructed a new scalable oversight technique for alignment.

摘要: 可部署的大型语言模型(LLMS)必须符合有益和无害的标准，从而实现LLMS的输出与人的价值之间的一致性。红团队技术构成了实现这一标准的关键途径。现有的工作完全依赖于手动红色团队设计和启发式对抗性提示来进行漏洞检测和优化。这些方法缺乏严格的数学描述，从而限制了在可量化的度量范围内探索多样化的攻击策略，以及在收敛保证下对LLMS进行优化。在本文中，我们提出了一种不需要人工注释的通用博弈论框架--Red-Teaming Game(RTG)。RTG用于分析红队语言模型(RLMS)和蓝队语言模型(BLM)之间的多回合攻防交互。在RTG中，我们提出了一种具有语义空间多样性度量的Gamalized Red-Teaming Solver(GRTS)。GRTS是一种自动红队技术，通过元博弈分析解决RTG向纳什均衡的方向，这对应于理论上保证的RLMS和BLM的优化方向。在RLMS多回合攻击中的实验结果表明，GRTS自主发现多样化的攻击策略，有效地提高了LLMS的安全性，优于已有的启发式红队设计。总体而言，RTG为红色团队任务建立了一个基本框架，并构建了一种新的可扩展的协调监督技术。



## **36. Adversarial Robustness in Graph Neural Networks: A Hamiltonian Approach**

图神经网络的对抗性稳健性：哈密顿方法 cs.LG

Accepted by Advances in Neural Information Processing Systems  (NeurIPS), New Orleans, USA, Dec. 2023, spotlight

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06396v1) [paper-pdf](http://arxiv.org/pdf/2310.06396v1)

**Authors**: Kai Zhao, Qiyu Kang, Yang Song, Rui She, Sijie Wang, Wee Peng Tay

**Abstract**: Graph neural networks (GNNs) are vulnerable to adversarial perturbations, including those that affect both node features and graph topology. This paper investigates GNNs derived from diverse neural flows, concentrating on their connection to various stability notions such as BIBO stability, Lyapunov stability, structural stability, and conservative stability. We argue that Lyapunov stability, despite its common use, does not necessarily ensure adversarial robustness. Inspired by physics principles, we advocate for the use of conservative Hamiltonian neural flows to construct GNNs that are robust to adversarial attacks. The adversarial robustness of different neural flow GNNs is empirically compared on several benchmark datasets under a variety of adversarial attacks. Extensive numerical experiments demonstrate that GNNs leveraging conservative Hamiltonian flows with Lyapunov stability substantially improve robustness against adversarial perturbations. The implementation code of experiments is available at https://github.com/zknus/NeurIPS-2023-HANG-Robustness.

摘要: 图神经网络(GNN)容易受到敌意扰动的影响，包括影响节点特征和图拓扑的扰动。本文研究了来自不同神经流的GNN，集中讨论了它们与各种稳定性概念的联系，如Bibo稳定性、Lyapunov稳定性、结构稳定性和保守稳定性。我们认为，李亚普诺夫稳定性，尽管它的普遍使用，并不一定确保对手的稳健性。受物理学原理的启发，我们提倡使用保守的哈密顿神经流来构造对对手攻击具有健壮性的GNN。在几个基准数据集上对不同神经流GNN在各种对抗攻击下的对抗健壮性进行了经验比较。大量的数值实验表明，GNN利用具有Lyapunov稳定性的保守哈密顿流大大提高了对对手扰动的鲁棒性。有关实验的实现代码，请访问https://github.com/zknus/NeurIPS-2023-HANG-Robustness.



## **37. Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations**

越狱和警卫对齐的语言模型，只有很少的上下文演示 cs.LG

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06387v1) [paper-pdf](http://arxiv.org/pdf/2310.06387v1)

**Authors**: Zeming Wei, Yifei Wang, Yisen Wang

**Abstract**: Large Language Models (LLMs) have shown remarkable success in various tasks, but concerns about their safety and the potential for generating malicious content have emerged. In this paper, we explore the power of In-Context Learning (ICL) in manipulating the alignment ability of LLMs. We find that by providing just few in-context demonstrations without fine-tuning, LLMs can be manipulated to increase or decrease the probability of jailbreaking, i.e. answering malicious prompts. Based on these observations, we propose In-Context Attack (ICA) and In-Context Defense (ICD) methods for jailbreaking and guarding aligned language model purposes. ICA crafts malicious contexts to guide models in generating harmful outputs, while ICD enhances model robustness by demonstrations of rejecting to answer harmful prompts. Our experiments show the effectiveness of ICA and ICD in increasing or reducing the success rate of adversarial jailbreaking attacks. Overall, we shed light on the potential of ICL to influence LLM behavior and provide a new perspective for enhancing the safety and alignment of LLMs.

摘要: 大型语言模型(LLM)在各种任务中取得了显著的成功，但也出现了对其安全性和生成恶意内容的可能性的担忧。在这篇文章中，我们探索了情境中学习(ICL)在操纵LLMS对齐能力方面的力量。我们发现，通过提供很少的上下文演示而不进行微调，LLMS可以被操纵以增加或降低越狱的可能性，即回答恶意提示。基于这些观察，我们提出了上下文中攻击(ICA)和上下文中防御(ICD)方法，用于越狱和保护对齐语言模型。ICA制作恶意上下文来引导模型生成有害输出，而ICD通过演示拒绝回答有害提示来增强模型的稳健性。实验结果表明，ICA和ICD能够有效地提高或降低对抗性越狱攻击的成功率。总体而言，我们阐明了ICL影响LLM行为的潜力，并为提高LLM的安全性和一致性提供了一个新的视角。



## **38. Double Public Key Signing Function Oracle Attack on EdDSA Software Implementations**

双重公钥签名函数Oracle对EdDSA软件实现的攻击 cs.CR

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2308.15009v2) [paper-pdf](http://arxiv.org/pdf/2308.15009v2)

**Authors**: Sam Grierson, Konstantinos Chalkias, William J Buchanan, Leandros Maglaras

**Abstract**: EdDSA is a standardised elliptic curve digital signature scheme introduced to overcome some of the issues prevalent in the more established ECDSA standard. Due to the EdDSA standard specifying that the EdDSA signature be deterministic, if the signing function were to be used as a public key signing oracle for the attacker, the unforgeability notion of security of the scheme can be broken. This paper describes an attack against some of the most popular EdDSA implementations, which results in an adversary recovering the private key used during signing. With this recovered secret key, an adversary can sign arbitrary messages that would be seen as valid by the EdDSA verification function. A list of libraries with vulnerable APIs at the time of publication is provided. Furthermore, this paper provides two suggestions for securing EdDSA signing APIs against this vulnerability while it additionally discusses failed attempts to solve the issue.

摘要: EdDSA是一种标准化的椭圆曲线数字签名方案，引入该方案是为了克服在更成熟的ECDSA标准中普遍存在的一些问题。由于EdDSA标准规定EdDSA签名是确定性的，如果签名函数被用作攻击者的公钥签名预言，则方案的安全性的不可伪造性概念可能被打破。本文描述了对一些最流行的EdDSA实现的攻击，该攻击导致攻击者恢复签名期间使用的私钥。利用恢复的密钥，攻击者可以对EdDSA验证功能认为有效的任意消息进行签名。提供了在发布时具有易受攻击的API的库列表。此外，本文还提供了两条保护EdDSA签名API免受该漏洞攻击的建议，同时还讨论了解决该问题的失败尝试。



## **39. Exploring adversarial attacks in federated learning for medical imaging**

医学影像联合学习中的对抗性攻击研究 cs.CR

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06227v1) [paper-pdf](http://arxiv.org/pdf/2310.06227v1)

**Authors**: Erfan Darzi, Florian Dubost, N. M. Sijtsema, P. M. A van Ooijen

**Abstract**: Federated learning offers a privacy-preserving framework for medical image analysis but exposes the system to adversarial attacks. This paper aims to evaluate the vulnerabilities of federated learning networks in medical image analysis against such attacks. Employing domain-specific MRI tumor and pathology imaging datasets, we assess the effectiveness of known threat scenarios in a federated learning environment. Our tests reveal that domain-specific configurations can increase the attacker's success rate significantly. The findings emphasize the urgent need for effective defense mechanisms and suggest a critical re-evaluation of current security protocols in federated medical image analysis systems.

摘要: 联合学习为医学图像分析提供了隐私保护框架，但会使系统面临敌意攻击。本文旨在评估联合学习网络在医学图像分析中抵抗此类攻击的脆弱性。利用特定领域的MRI肿瘤和病理成像数据集，我们评估了联合学习环境中已知威胁场景的有效性。我们的测试表明，特定于域的配置可以显著提高攻击者的成功率。这些发现强调了对有效防御机制的迫切需要，并建议对联合医学图像分析系统中的当前安全协议进行关键的重新评估。



## **40. PAC-Bayesian Spectrally-Normalized Bounds for Adversarially Robust Generalization**

对抗性泛化的PAC-贝叶斯谱归一化界 cs.LG

NeurIPS 2023

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2310.06182v1) [paper-pdf](http://arxiv.org/pdf/2310.06182v1)

**Authors**: Jiancong Xiao, Ruoyu Sun, Zhi-quan Luo

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial attacks. It is found empirically that adversarially robust generalization is crucial in establishing defense algorithms against adversarial attacks. Therefore, it is interesting to study the theoretical guarantee of robust generalization. This paper focuses on norm-based complexity, based on a PAC-Bayes approach (Neyshabur et al., 2017). The main challenge lies in extending the key ingredient, which is a weight perturbation bound in standard settings, to the robust settings. Existing attempts heavily rely on additional strong assumptions, leading to loose bounds. In this paper, we address this issue and provide a spectrally-normalized robust generalization bound for DNNs. Compared to existing bounds, our bound offers two significant advantages: Firstly, it does not depend on additional assumptions. Secondly, it is considerably tighter, aligning with the bounds of standard generalization. Therefore, our result provides a different perspective on understanding robust generalization: The mismatch terms between standard and robust generalization bounds shown in previous studies do not contribute to the poor robust generalization. Instead, these disparities solely due to mathematical issues. Finally, we extend the main result to adversarial robustness against general non-$\ell_p$ attacks and other neural network architectures.

摘要: 深度神经网络(DNN)很容易受到敌意攻击。实验发现，对抗性健壮性泛化对于建立抵抗对抗性攻击的防御算法至关重要。因此，研究健壮性泛化的理论保障是很有意义的。本文基于PAC-Bayes方法(Neyshabur等人，2017年)，重点研究基于范数的复杂性。主要的挑战在于将关键成分(标准设置中的权重扰动范围)扩展到稳健设置。现有的尝试严重依赖于额外的强有力的假设，导致了宽松的界限。在这篇文章中，我们解决了这个问题，并为DNN提供了一个谱归一化的鲁棒泛化上界。与现有的边界相比，我们的边界有两个显著的优点：第一，它不依赖于额外的假设。其次，它相当紧凑，与标准泛化的界限一致。因此，我们的结果为理解健壮性概括提供了一个不同的视角：以前的研究中显示的标准和健壮性概化界限之间的不匹配项并不是导致健壮性概括较差的原因。相反，这些差异完全是由于数学问题。最后，我们将主要结果推广到对抗一般非EELL_p$攻击和其他神经网络结构的健壮性。



## **41. Lessons Learned: Defending Against Property Inference Attacks**

经验教训：防御属性推理攻击 cs.CR

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2205.08821v4) [paper-pdf](http://arxiv.org/pdf/2205.08821v4)

**Authors**: Joshua Stock, Jens Wettlaufer, Daniel Demmler, Hannes Federrath

**Abstract**: This work investigates and evaluates multiple defense strategies against property inference attacks (PIAs), a privacy attack against machine learning models. Given a trained machine learning model, PIAs aim to extract statistical properties of its underlying training data, e.g., reveal the ratio of men and women in a medical training data set. While for other privacy attacks like membership inference, a lot of research on defense mechanisms has been published, this is the first work focusing on defending against PIAs. With the primary goal of developing a generic mitigation strategy against white-box PIAs, we propose the novel approach property unlearning. Extensive experiments with property unlearning show that while it is very effective when defending target models against specific adversaries, property unlearning is not able to generalize, i.e., protect against a whole class of PIAs. To investigate the reasons behind this limitation, we present the results of experiments with the explainable AI tool LIME. They show how state-of-the-art property inference adversaries with the same objective focus on different parts of the target model. We further elaborate on this with a follow-up experiment, in which we use the visualization technique t-SNE to exhibit how severely statistical training data properties are manifested in machine learning models. Based on this, we develop the conjecture that post-training techniques like property unlearning might not suffice to provide the desirable generic protection against PIAs. As an alternative, we investigate the effects of simpler training data preprocessing methods like adding Gaussian noise to images of a training data set on the success rate of PIAs. We conclude with a discussion of the different defense approaches, summarize the lessons learned and provide directions for future work.

摘要: 本文研究和评估了针对机器学习模型的隐私攻击--属性推理攻击(PIA)的多种防御策略。给定一个经过训练的机器学习模型，PIA的目标是提取其基本训练数据的统计属性，例如，揭示医学训练数据集中的男性和女性的比例。虽然对于其他隐私攻击，如成员身份推断，已经发表了大量关于防御机制的研究，但这是第一个专注于防御PIA的工作。以开发一种针对白盒PIA的通用缓解策略为主要目标，我们提出了一种新的方法-属性遗忘。大量的属性遗忘实验表明，虽然它在防御特定对手的目标模型时非常有效，但属性遗忘不能泛化，即保护一整类PIA。为了调查这种限制背后的原因，我们给出了使用可解释的人工智能工具LIME的实验结果。它们展示了具有相同目标的最先进的属性推理对手如何专注于目标模型的不同部分。我们通过后续实验进一步阐述了这一点，在该实验中，我们使用可视化技术t-SNE来展示统计训练数据属性在机器学习模型中的表现是多么严重。在此基础上，我们提出了这样的猜测，即训练后的技术，如属性遗忘，可能不足以提供理想的通用保护，以防止PIA。作为另一种选择，我们研究了更简单的训练数据预处理方法，如向训练数据集的图像添加高斯噪声对PIA成功率的影响。最后，我们讨论了不同的防御方法，总结了经验教训，并为未来的工作提供了方向。



## **42. Universal adversarial perturbations for multiple classification tasks with quantum classifiers**

量子分类器多分类任务的普遍对抗性扰动 quant-ph

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2306.11974v2) [paper-pdf](http://arxiv.org/pdf/2306.11974v2)

**Authors**: Yun-Zhong Qiu

**Abstract**: Quantum adversarial machine learning is an emerging field that studies the vulnerability of quantum learning systems against adversarial perturbations and develops possible defense strategies. Quantum universal adversarial perturbations are small perturbations, which can make different input samples into adversarial examples that may deceive a given quantum classifier. This is a field that was rarely looked into but worthwhile investigating because universal perturbations might simplify malicious attacks to a large extent, causing unexpected devastation to quantum machine learning models. In this paper, we take a step forward and explore the quantum universal perturbations in the context of heterogeneous classification tasks. In particular, we find that quantum classifiers that achieve almost state-of-the-art accuracy on two different classification tasks can be both conclusively deceived by one carefully-crafted universal perturbation. This result is explicitly demonstrated with well-designed quantum continual learning models with elastic weight consolidation method to avoid catastrophic forgetting, as well as real-life heterogeneous datasets from hand-written digits and medical MRI images. Our results provide a simple and efficient way to generate universal perturbations on heterogeneous classification tasks and thus would provide valuable guidance for future quantum learning technologies.

摘要: 量子对抗机器学习是一个新兴的研究领域，它研究量子学习系统对对抗扰动的脆弱性，并开发可能的防御策略。量子通用对抗性扰动是一种微小的扰动，它可以使不同的输入样本变成可能欺骗给定量子分类器的对抗性例子。这是一个很少被研究但值得研究的领域，因为普遍的扰动可能会在很大程度上简化恶意攻击，给量子机器学习模型造成意想不到的破坏。在这篇文章中，我们向前迈进了一步，探索了异质分类任务背景下的量子普适微扰。特别是，我们发现，在两个不同的分类任务上获得几乎最先进的精度的量子分类器都可能最终被一个精心设计的普遍扰动所欺骗。这一结果通过设计良好的弹性权重巩固方法的量子连续学习模型来避免灾难性遗忘，以及来自手写数字和医学MRI图像的现实生活中的异质数据集得到了明确的证明。我们的结果提供了一种简单而有效的方法来产生对异类分类任务的普遍扰动，从而为未来的量子学习技术提供了有价值的指导。



## **43. RECESS Vaccine for Federated Learning: Proactive Defense Against Model Poisoning Attacks**

联邦学习的休会疫苗：对模型中毒攻击的主动防御 cs.CR

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2310.05431v1) [paper-pdf](http://arxiv.org/pdf/2310.05431v1)

**Authors**: Haonan Yan, Wenjing Zhang, Qian Chen, Xiaoguang Li, Wenhai Sun, Hui Li, Xiaodong Lin

**Abstract**: Model poisoning attacks greatly jeopardize the application of federated learning (FL). The effectiveness of existing defenses is susceptible to the latest model poisoning attacks, leading to a decrease in prediction accuracy. Besides, these defenses are intractable to distinguish benign outliers from malicious gradients, which further compromises the model generalization. In this work, we propose a novel proactive defense named RECESS against model poisoning attacks. Different from the passive analysis in previous defenses, RECESS proactively queries each participating client with a delicately constructed aggregation gradient, accompanied by the detection of malicious clients according to their responses with higher accuracy. Furthermore, RECESS uses a new trust scoring mechanism to robustly aggregate gradients. Unlike previous methods that score each iteration, RECESS considers clients' performance correlation across multiple iterations to estimate the trust score, substantially increasing fault tolerance. Finally, we extensively evaluate RECESS on typical model architectures and four datasets under various settings. We also evaluated the defensive effectiveness against other types of poisoning attacks, the sensitivity of hyperparameters, and adaptive adversarial attacks. Experimental results show the superiority of RECESS in terms of reducing accuracy loss caused by the latest model poisoning attacks over five classic and two state-of-the-art defenses.

摘要: 模型中毒攻击极大地危害了联邦学习的应用。现有防御的有效性容易受到最新模型中毒攻击的影响，导致预测准确性下降。此外，这些防御措施很难区分良性异常值和恶意梯度，这进一步损害了模型的泛化。在这项工作中，我们提出了一种新的针对模型中毒攻击的主动防御机制--休息。与以往防御中的被动分析不同，Recess通过精心构建的聚合梯度主动查询每个参与的客户端，并根据恶意客户端的响应检测恶意客户端，具有更高的准确率。此外，SESESS使用了一种新的信任评分机制来稳健地聚合梯度。与以前对每次迭代进行评分的方法不同，Recess考虑客户在多次迭代中的性能相关性来估计信任分数，从而大大提高了容错能力。最后，我们在典型的模型结构和四个不同设置下的数据集上对RESECT进行了广泛的评估。我们还评估了对其他类型的中毒攻击的防御效果、超参数的敏感度和自适应对抗性攻击。实验结果表明，在减少最新模型中毒攻击造成的精度损失方面，SESECT优于五个经典防御系统和两个最新防御系统。



## **44. AdvSV: An Over-the-Air Adversarial Attack Dataset for Speaker Verification**

AdvSV：一种用于说话人确认的空中对抗攻击数据集 cs.SD

Submitted to ICASSP2024

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2310.05369v1) [paper-pdf](http://arxiv.org/pdf/2310.05369v1)

**Authors**: Li Wang, Jiaqi Li, Yuhao Luo, Jiahao Zheng, Lei Wang, Hao Li, Ke Xu, Chengfang Fang, Jie Shi, Zhizheng Wu

**Abstract**: It is known that deep neural networks are vulnerable to adversarial attacks. Although Automatic Speaker Verification (ASV) built on top of deep neural networks exhibits robust performance in controlled scenarios, many studies confirm that ASV is vulnerable to adversarial attacks. The lack of a standard dataset is a bottleneck for further research, especially reproducible research. In this study, we developed an open-source adversarial attack dataset for speaker verification research. As an initial step, we focused on the over-the-air attack. An over-the-air adversarial attack involves a perturbation generation algorithm, a loudspeaker, a microphone, and an acoustic environment. The variations in the recording configurations make it very challenging to reproduce previous research. The AdvSV dataset is constructed using the Voxceleb1 Verification test set as its foundation. This dataset employs representative ASV models subjected to adversarial attacks and records adversarial samples to simulate over-the-air attack settings. The scope of the dataset can be easily extended to include more types of adversarial attacks. The dataset will be released to the public under the CC-BY license. In addition, we also provide a detection baseline for reproducible research.

摘要: 众所周知，深度神经网络很容易受到敌意攻击。尽管建立在深度神经网络之上的自动说话人确认(ASV)在受控场景下表现出较强的性能，但许多研究证实ASV容易受到对手攻击。缺乏标准数据集是进一步研究的瓶颈，特别是可重复性研究。在这项研究中，我们开发了一个开源的对抗性攻击数据集，用于说话人验证研究。作为第一步，我们把重点放在空中攻击上。空中对抗性攻击涉及扰动生成算法、扬声器、麦克风和声学环境。记录配置的变化使得复制先前的研究非常具有挑战性。AdvSV数据集是使用Voxeleb1验证测试集作为其基础来构建的。该数据集采用了典型的受到对抗性攻击的ASV模型，并记录了对抗性样本以模拟空中攻击设置。数据集的范围可以很容易地扩展到包括更多类型的对抗性攻击。数据集将根据CC-BY许可证向公众发布。此外，我们还为可重复性研究提供了检测基线。



## **45. GReAT: A Graph Regularized Adversarial Training Method**

GREAT：一种图规化的对抗性训练方法 cs.LG

25 pages including references. 7 figures and 4 tables

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2310.05336v1) [paper-pdf](http://arxiv.org/pdf/2310.05336v1)

**Authors**: Samet Bayram, Kenneth Barner

**Abstract**: This paper proposes a regularization method called GReAT, Graph Regularized Adversarial Training, to improve deep learning models' classification performance. Adversarial examples are a well-known challenge in machine learning, where small, purposeful perturbations to input data can mislead models. Adversarial training, a powerful and one of the most effective defense strategies, involves training models with both regular and adversarial examples. However, it often neglects the underlying structure of the data. In response, we propose GReAT, a method that leverages data graph structure to enhance model robustness. GReAT deploys the graph structure of the data into the adversarial training process, resulting in more robust models that better generalize its testing performance and defend against adversarial attacks. Through extensive evaluation on benchmark datasets, we demonstrate GReAT's effectiveness compared to state-of-the-art classification methods, highlighting its potential in improving deep learning models' classification performance.

摘要: 为了提高深度学习模型的分类性能，提出了一种称为大图正则化对抗性训练的正则化方法。对抗性的例子是机器学习中众所周知的挑战，在机器学习中，对输入数据的微小、有目的的扰动可能会误导模型。对抗性训练是一种强有力的、最有效的防御策略之一，它包括常规训练模式和对抗性训练模式。然而，它经常忽略数据的底层结构。对此，我们提出了一种利用数据图结构来增强模型稳健性的方法--GREAT。Great将数据的图形结构部署到对抗性训练过程中，从而产生更健壮的模型，更好地概括其测试性能并防御对抗性攻击。通过在基准数据集上的广泛评估，我们展示了相对于最先进的分类方法的有效性，突出了其在提高深度学习模型的分类性能方面的潜力。



## **46. On the Query Complexity of Training Data Reconstruction in Private Learning**

私学中训练数据重构的查询复杂性研究 cs.LG

Matching upper bounds, new corollaries for DP variants

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2303.16372v5) [paper-pdf](http://arxiv.org/pdf/2303.16372v5)

**Authors**: Prateeti Mukherjee, Satya Lokam

**Abstract**: We analyze the number of queries that a whitebox adversary needs to make to a private learner in order to reconstruct its training data. For $(\epsilon, \delta)$ DP learners with training data drawn from any arbitrary compact metric space, we provide the \emph{first known lower bounds on the adversary's query complexity} as a function of the learner's privacy parameters. \emph{Our results are minimax optimal for every $\epsilon \geq 0, \delta \in [0, 1]$, covering both $\epsilon$-DP and $(0, \delta)$ DP as corollaries}. Beyond this, we obtain query complexity lower bounds for $(\alpha, \epsilon)$ R\'enyi DP learners that are valid for any $\alpha > 1, \epsilon \geq 0$. Finally, we analyze data reconstruction attacks on locally compact metric spaces via the framework of Metric DP, a generalization of DP that accounts for the underlying metric structure of the data. In this setting, we provide the first known analysis of data reconstruction in unbounded, high dimensional spaces and obtain query complexity lower bounds that are nearly tight modulo logarithmic factors.

摘要: 我们分析了白盒攻击者为了重建其训练数据而需要向私人学习者进行的查询数量。对于具有来自任意紧致度量空间的训练数据的$(\epsilon，\Delta)$DP学习者，我们提供了作为学习者隐私参数的函数的\emph(对手查询复杂性的第一个已知下界)。{我们的结果对[0，1]$中的每个$\epsilon\geq0，\Delta\都是极小极大最优的，推论包括$\epsilon$-dp和$(0，\Delta)$dp}。在此基础上，我们得到了$(\α，\epsilon)$R‘Enyi DP学习者的查询复杂性下界，这些下界对任何$\α>1，\epsion\0$都是有效的。最后，我们通过度量DP框架分析了局部紧度量空间上的数据重构攻击。度量DP是DP的推广，它解释了数据的基本度量结构。在这个背景下，我们首次对无界高维空间中的数据重构进行了分析，得到了几乎紧模对数因子的查询复杂度下界。



## **47. Adversarial Attacks on Combinatorial Multi-Armed Bandits**

组合式多臂土匪的对抗性攻击 cs.LG

28 pages

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2310.05308v1) [paper-pdf](http://arxiv.org/pdf/2310.05308v1)

**Authors**: Rishab Balasubramanian, Jiawei Li, Prasad Tadepalli, Huazheng Wang, Qingyun Wu, Haoyu Zhao

**Abstract**: We study reward poisoning attacks on Combinatorial Multi-armed Bandits (CMAB). We first provide a sufficient and necessary condition for the attackability of CMAB, which depends on the intrinsic properties of the corresponding CMAB instance such as the reward distributions of super arms and outcome distributions of base arms. Additionally, we devise an attack algorithm for attackable CMAB instances. Contrary to prior understanding of multi-armed bandits, our work reveals a surprising fact that the attackability of a specific CMAB instance also depends on whether the bandit instance is known or unknown to the adversary. This finding indicates that adversarial attacks on CMAB are difficult in practice and a general attack strategy for any CMAB instance does not exist since the environment is mostly unknown to the adversary. We validate our theoretical findings via extensive experiments on real-world CMAB applications including probabilistic maximum covering problem, online minimum spanning tree, cascading bandits for online ranking, and online shortest path.

摘要: 研究了组合多臂土匪(CMAB)的悬赏中毒攻击。我们首先给出了CMAB可攻击性的充要条件，该条件依赖于相应CMAB实例的内在性质，如超臂的奖赏分布和基臂的结果分布。此外，我们还设计了一个针对可攻击CMAB实例的攻击算法。与之前对多武装强盗的理解相反，我们的工作揭示了一个令人惊讶的事实，即特定CMAB实例的可攻击性还取决于对手是否知道该强盗实例。这一发现表明，对CMAB的对抗性攻击在实践中是困难的，并且不存在针对任何CMAB实例的通用攻击策略，因为对手基本上不知道环境。通过在概率最大覆盖问题、在线最小生成树、在线排名的级联强盗和在线最短路径等实际CMAB应用上的大量实验，我们验证了我们的理论发现。



## **48. Robust Lipschitz Bandits to Adversarial Corruptions**

对抗腐败的强健Lipschitz Bandits cs.LG

Thirty-seventh Conference on Neural Information Processing Systems  (NeurIPS 2023)

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2305.18543v2) [paper-pdf](http://arxiv.org/pdf/2305.18543v2)

**Authors**: Yue Kang, Cho-Jui Hsieh, Thomas C. M. Lee

**Abstract**: Lipschitz bandit is a variant of stochastic bandits that deals with a continuous arm set defined on a metric space, where the reward function is subject to a Lipschitz constraint. In this paper, we introduce a new problem of Lipschitz bandits in the presence of adversarial corruptions where an adaptive adversary corrupts the stochastic rewards up to a total budget $C$. The budget is measured by the sum of corruption levels across the time horizon $T$. We consider both weak and strong adversaries, where the weak adversary is unaware of the current action before the attack, while the strong one can observe it. Our work presents the first line of robust Lipschitz bandit algorithms that can achieve sub-linear regret under both types of adversary, even when the total budget of corruption $C$ is unrevealed to the agent. We provide a lower bound under each type of adversary, and show that our algorithm is optimal under the strong case. Finally, we conduct experiments to illustrate the effectiveness of our algorithms against two classic kinds of attacks.

摘要: Lipschitz Bandit是随机强盗的变种，它处理定义在度量空间上的连续臂集，其中奖励函数受Lipschitz约束。在这篇文章中，我们引入了一个新的存在对抗性腐败的Lipschitz强盗问题，其中一个自适应的敌对者破坏了总预算为$C$的随机报酬。预算是通过整个时间范围内腐败程度的总和新台币来衡量的。我们既考虑弱对手，也考虑强对手，其中弱对手在攻击前不知道当前的行动，而强对手可以观察到。我们的工作提出了第一行稳健的Lipschitz强盗算法，即使在代理未透露腐败$C$的总预算时，该算法也可以在这两种类型的对手下实现次线性遗憾。在每种类型的对手下，我们给出了一个下界，并证明了我们的算法在强情形下是最优的。最后，通过实验验证了该算法对两种经典攻击的有效性。



## **49. Susceptibility of Continual Learning Against Adversarial Attacks**

持续学习对敌意攻击的敏感性 cs.LG

18 pages, 13 figures

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2207.05225v5) [paper-pdf](http://arxiv.org/pdf/2207.05225v5)

**Authors**: Hikmat Khan, Pir Masoom Shah, Syed Farhan Alam Zaidi, Saif ul Islam, Qasim Zia

**Abstract**: Recent continual learning approaches have primarily focused on mitigating catastrophic forgetting. Nevertheless, two critical areas have remained relatively unexplored: 1) evaluating the robustness of proposed methods and 2) ensuring the security of learned tasks. This paper investigates the susceptibility of continually learned tasks, including current and previously acquired tasks, to adversarial attacks. Specifically, we have observed that any class belonging to any task can be easily targeted and misclassified as the desired target class of any other task. Such susceptibility or vulnerability of learned tasks to adversarial attacks raises profound concerns regarding data integrity and privacy. To assess the robustness of continual learning approaches, we consider continual learning approaches in all three scenarios, i.e., task-incremental learning, domain-incremental learning, and class-incremental learning. In this regard, we explore the robustness of three regularization-based methods, three replay-based approaches, and one hybrid technique that combines replay and exemplar approaches. We empirically demonstrated that in any setting of continual learning, any class, whether belonging to the current or previously learned tasks, is susceptible to misclassification. Our observations identify potential limitations of continual learning approaches against adversarial attacks and highlight that current continual learning algorithms could not be suitable for deployment in real-world settings.

摘要: 最近的持续学习方法主要集中在减轻灾难性遗忘上。然而，有两个关键领域仍然相对未被探索：1)评估所提出方法的稳健性；2)确保学习任务的安全性。本文研究了持续学习任务，包括当前任务和以前获得的任务，对敌意攻击的敏感性。具体地说，我们观察到，属于任何任务的任何类都很容易成为目标，并被错误地归类为任何其他任务所需的目标类。学习任务对敌意攻击的这种易感性或脆弱性引起了人们对数据完整性和隐私的深切关注。为了评估持续学习方法的稳健性，我们考虑了三种情况下的持续学习方法，即任务增量学习、领域增量学习和班级增量学习。在这方面，我们探索了三种基于正则化的方法、三种基于回放的方法以及一种结合了回放和样本方法的混合技术的稳健性。我们的经验表明，在任何持续学习的环境中，任何班级，无论是属于当前学习的还是以前学习的任务，都容易发生错误分类。我们的观察发现了持续学习方法在对抗对手攻击时的潜在局限性，并强调了当前的持续学习算法不适合在现实世界中部署。



## **50. Transferable Availability Poisoning Attacks**

可转让可用性中毒攻击 cs.CR

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2310.05141v1) [paper-pdf](http://arxiv.org/pdf/2310.05141v1)

**Authors**: Yiyong Liu, Michael Backes, Xiao Zhang

**Abstract**: We consider availability data poisoning attacks, where an adversary aims to degrade the overall test accuracy of a machine learning model by crafting small perturbations to its training data. Existing poisoning strategies can achieve the attack goal but assume the victim to employ the same learning method as what the adversary uses to mount the attack. In this paper, we argue that this assumption is strong, since the victim may choose any learning algorithm to train the model as long as it can achieve some targeted performance on clean data. Empirically, we observe a large decrease in the effectiveness of prior poisoning attacks if the victim uses a different learning paradigm to train the model and show marked differences in frequency-level characteristics between perturbations generated with respect to different learners and attack methods. To enhance the attack transferability, we propose Transferable Poisoning, which generates high-frequency poisoning perturbations by alternately leveraging the gradient information with two specific algorithms selected from supervised and unsupervised contrastive learning paradigms. Through extensive experiments on benchmark image datasets, we show that our transferable poisoning attack can produce poisoned samples with significantly improved transferability, not only applicable to the two learners used to devise the attack but also for learning algorithms and even paradigms beyond.

摘要: 我们考虑可用性数据中毒攻击，其中对手的目标是通过对机器学习模型的训练数据进行小的扰动来降低其总体测试精度。现有的中毒策略可以达到攻击目标，但假设受害者使用与对手发动攻击相同的学习方法。在本文中，我们认为这一假设是强有力的，因为受害者可以选择任何学习算法来训练模型，只要它能够在干净的数据上达到一些目标性能。从经验上看，如果受害者使用不同的学习范式来训练模型，并显示出针对不同学习者和攻击方法产生的扰动之间的频率水平特征显著差异，我们观察到先前中毒攻击的有效性大幅下降。为了增强攻击的可转移性，我们提出了可传递中毒，通过交替利用梯度信息和从监督和非监督对比学习范例中选择的两种特定算法来产生高频中毒扰动。通过在基准图像数据集上的大量实验表明，我们的可转移中毒攻击可以产生中毒样本，并显著提高了可转移性，不仅适用于设计攻击的两个学习器，而且适用于学习算法甚至更多的范例。



