# Latest Adversarial Attack Papers
**update at 2024-09-25 10:24:22**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Order of Magnitude Speedups for LLM Membership Inference**

LLM会员推断的量级加速 cs.LG

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.14513v2) [paper-pdf](http://arxiv.org/pdf/2409.14513v2)

**Authors**: Rongting Zhang, Martin Bertran, Aaron Roth

**Abstract**: Large Language Models (LLMs) have the promise to revolutionize computing broadly, but their complexity and extensive training data also expose significant privacy vulnerabilities. One of the simplest privacy risks associated with LLMs is their susceptibility to membership inference attacks (MIAs), wherein an adversary aims to determine whether a specific data point was part of the model's training set. Although this is a known risk, state of the art methodologies for MIAs rely on training multiple computationally costly shadow models, making risk evaluation prohibitive for large models. Here we adapt a recent line of work which uses quantile regression to mount membership inference attacks; we extend this work by proposing a low-cost MIA that leverages an ensemble of small quantile regression models to determine if a document belongs to the model's training set or not. We demonstrate the effectiveness of this approach on fine-tuned LLMs of varying families (OPT, Pythia, Llama) and across multiple datasets. Across all scenarios we obtain comparable or improved accuracy compared to state of the art shadow model approaches, with as little as 6% of their computation budget. We demonstrate increased effectiveness across multi-epoch trained target models, and architecture miss-specification robustness, that is, we can mount an effective attack against a model using a different tokenizer and architecture, without requiring knowledge on the target model.

摘要: 大型语言模型(LLM)有望给计算带来革命性的变化，但它们的复杂性和大量的训练数据也暴露了严重的隐私漏洞。与LLMS相关的最简单的隐私风险之一是它们对成员关系推理攻击(MIA)的敏感性，即对手的目标是确定特定数据点是否属于模型训练集的一部分。尽管这是一个已知的风险，但MIA的最新方法依赖于训练多个计算成本高昂的阴影模型，这使得风险评估对于大型模型来说是令人望而却步的。在这里，我们采用了最近的一项工作，它使用分位数回归来发动成员关系推理攻击；我们通过提出一种低成本的MIA来扩展这一工作，该MIA利用一组小分位数回归模型来确定文档是否属于模型的训练集。我们在不同家族(OPT，Pythia，Llama)的微调LLM上展示了这种方法的有效性，并跨越了多个数据集。在所有场景中，与最先进的阴影模型方法相比，我们获得了相当或更高的精度，只需其计算预算的6%。我们证明了多时代训练的目标模型的有效性和体系结构未指定的稳健性，即，我们可以使用不同的标记器和体系结构对模型发动有效攻击，而不需要目标模型的知识。



## **2. Unclonable Non-Interactive Zero-Knowledge**

不可克隆的非互动零知识 cs.CR

Clarified definitions and included applications

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2310.07118v3) [paper-pdf](http://arxiv.org/pdf/2310.07118v3)

**Authors**: Ruta Jawale, Dakshita Khurana

**Abstract**: A non-interactive ZK (NIZK) proof enables verification of NP statements without revealing secrets about them. However, an adversary that obtains a NIZK proof may be able to clone this proof and distribute arbitrarily many copies of it to various entities: this is inevitable for any proof that takes the form of a classical string. In this paper, we ask whether it is possible to rely on quantum information in order to build NIZK proof systems that are impossible to clone.   We define and construct unclonable non-interactive zero-knowledge arguments (of knowledge) for NP, addressing a question first posed by Aaronson (CCC 2009). Besides satisfying the zero-knowledge and argument of knowledge properties, these proofs additionally satisfy unclonability. Very roughly, this ensures that no adversary can split an honestly generated proof of membership of an instance $x$ in an NP language $\mathcal{L}$ and distribute copies to multiple entities that all obtain accepting proofs of membership of $x$ in $\mathcal{L}$. Our result has applications to unclonable signatures of knowledge, which we define and construct in this work; these non-interactively prevent replay attacks.

摘要: 非交互ZK(NIZK)证明能够在不泄露有关NP语句的秘密的情况下验证NP语句。然而，获得NIZK证明的对手可能能够克隆该证明并将其任意多个副本分发给各种实体：对于任何采用经典字符串形式的证明来说，这是不可避免的。在这篇论文中，我们问是否有可能依靠量子信息来建立不可能克隆的NIZK证明系统。我们定义并构造了NP的不可克隆、非交互的零知识论点(知识)，回答了Aaronson(CCC，2009)首先提出的一个问题。这些证明除了满足知识性质的零知识和论辩外，还满足不可克隆性。粗略地说，这确保了没有对手能够用NP语言$\数学{L}$拆分诚实地生成的实例$x$的成员资格证明，并将副本分发给多个实体，这些实体都获得了$\数学{L}$中的$x$的成员资格的接受证明。我们的结果适用于不可克隆的知识签名，我们在本工作中定义和构造了这些签名；这些非交互的签名可以防止重放攻击。



## **3. Cyber Knowledge Completion Using Large Language Models**

使用大型语言模型完成网络知识 cs.CR

7 pages, 2 figures. Submitted to 2024 IEEE International Conference  on Big Data

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.16176v1) [paper-pdf](http://arxiv.org/pdf/2409.16176v1)

**Authors**: Braden K Webb, Sumit Purohit, Rounak Meyur

**Abstract**: The integration of the Internet of Things (IoT) into Cyber-Physical Systems (CPSs) has expanded their cyber-attack surface, introducing new and sophisticated threats with potential to exploit emerging vulnerabilities. Assessing the risks of CPSs is increasingly difficult due to incomplete and outdated cybersecurity knowledge. This highlights the urgent need for better-informed risk assessments and mitigation strategies. While previous efforts have relied on rule-based natural language processing (NLP) tools to map vulnerabilities, weaknesses, and attack patterns, recent advancements in Large Language Models (LLMs) present a unique opportunity to enhance cyber-attack knowledge completion through improved reasoning, inference, and summarization capabilities. We apply embedding models to encapsulate information on attack patterns and adversarial techniques, generating mappings between them using vector embeddings. Additionally, we propose a Retrieval-Augmented Generation (RAG)-based approach that leverages pre-trained models to create structured mappings between different taxonomies of threat patterns. Further, we use a small hand-labeled dataset to compare the proposed RAG-based approach to a baseline standard binary classification model. Thus, the proposed approach provides a comprehensive framework to address the challenge of cyber-attack knowledge graph completion.

摘要: 物联网(IoT)与网络物理系统(CPSS)的集成扩大了它们的网络攻击面，带来了新的复杂威胁，有可能利用新出现的漏洞。由于网络安全知识的不完整和过时，评估CPSS的风险变得越来越困难。这突出表明迫切需要更了解情况的风险评估和缓解战略。虽然以前的努力依赖于基于规则的自然语言处理(NLP)工具来绘制漏洞、弱点和攻击模式，但最近大型语言模型(LLM)的进步提供了一个独特的机会，可以通过改进的推理、推理和总结能力来增强网络攻击知识的完备性。我们使用嵌入模型来封装关于攻击模式和对抗技术的信息，并使用向量嵌入来生成它们之间的映射。此外，我们提出了一种基于检索增强生成(RAG)的方法，该方法利用预先训练的模型来创建威胁模式的不同分类之间的结构化映射。此外，我们使用一个小的手工标记的数据集来比较所提出的基于RAG的方法与基准标准的二进制分类模型。因此，提出的方法提供了一个全面的框架来解决网络攻击知识图完成的挑战。



## **4. A Strong Separation for Adversarially Robust $\ell_0$ Estimation for Linear Sketches**

线性草图的对抗稳健$\ell_0 $估计的强分离 cs.DS

FOCS 2024

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.16153v1) [paper-pdf](http://arxiv.org/pdf/2409.16153v1)

**Authors**: Elena Gribelyuk, Honghao Lin, David P. Woodruff, Huacheng Yu, Samson Zhou

**Abstract**: The majority of streaming problems are defined and analyzed in a static setting, where the data stream is any worst-case sequence of insertions and deletions that is fixed in advance. However, many real-world applications require a more flexible model, where an adaptive adversary may select future stream elements after observing the previous outputs of the algorithm. Over the last few years, there has been increased interest in proving lower bounds for natural problems in the adaptive streaming model. In this work, we give the first known adaptive attack against linear sketches for the well-studied $\ell_0$-estimation problem over turnstile, integer streams. For any linear streaming algorithm $\mathcal{A}$ that uses sketching matrix $\mathbf{A}\in \mathbb{Z}^{r \times n}$ where $n$ is the size of the universe, this attack makes $\tilde{\mathcal{O}}(r^8)$ queries and succeeds with high constant probability in breaking the sketch. We also give an adaptive attack against linear sketches for the $\ell_0$-estimation problem over finite fields $\mathbb{F}_p$, which requires a smaller number of $\tilde{\mathcal{O}}(r^3)$ queries. Finally, we provide an adaptive attack over $\mathbb{R}^n$ against linear sketches $\mathbf{A} \in \mathbb{R}^{r \times n}$ for $\ell_0$-estimation, in the setting where $\mathbf{A}$ has all nonzero subdeterminants at least $\frac{1}{\textrm{poly}(r)}$. Our results provide an exponential improvement over the previous number of queries known to break an $\ell_0$-estimation sketch.

摘要: 大多数流问题是在静态设置中定义和分析的，其中数据流是预先固定的任何最坏情况的插入和删除序列。然而，许多现实世界的应用需要更灵活的模型，在这种模型中，自适应对手可能会在观察到算法的先前输出后选择未来的流元素。在过去的几年里，人们越来越有兴趣证明自适应流模型中自然问题的下界。在这项工作中，我们给出了第一个已知的针对线性草图的自适应攻击，该攻击针对已经研究得很好的$\ell_0$-旋转门上的整数流估计问题。对于任何使用素描矩阵$\mathbf{A}\in\mathbb{Z}^{r\Times n}$的线性流算法$\mathcal{A}$，其中$n$是宇宙的大小，该攻击使$\tide{\mathcal{O}}(r^8)$查询并以很高的恒定概率破解素描。对于有限域$\mathbb{F}_p$上的$\ell_0$-估计问题，我们还给出了一个针对线性草图的自适应攻击，它需要较少的$\tilde{\mathcal{O}}(r^3)$查询。最后，我们在$\mathbb{R}^n$上对$\mathbb{R}^{r\次n}$对$\ell_0$-估计的线性勾画$\mathbf{A}\$进行自适应攻击，其中$\mathbf{A}$至少有非零子行列式$\frac{1}{\tex m{poly}(R)}$。我们的结果提供了比以前已知的打破$\ELL_0$-估计草图的查询数量的指数级改进。



## **5. Scenario of Use Scheme: Threat Model Specification for Speaker Privacy Protection in the Medical Domain**

使用场景方案：医疗领域演讲者隐私保护的威胁模型规范 eess.AS

Accepted and published at SPSC Symposium 2024 4th Symposium on  Security and Privacy in Speech Communication. Interspeech 2024

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.16106v1) [paper-pdf](http://arxiv.org/pdf/2409.16106v1)

**Authors**: Mehtab Ur Rahman, Martha Larson, Louis ten Bosch, Cristian Tejedor-García

**Abstract**: Speech recordings are being more frequently used to detect and monitor disease, leading to privacy concerns. Beyond cryptography, protection of speech can be addressed by approaches, such as perturbation, disentanglement, and re-synthesis, that eliminate sensitive information of the speaker, leaving the information necessary for medical analysis purposes. In order for such privacy protective approaches to be developed, clear and systematic specifications of assumptions concerning medical settings and the needs of medical professionals are necessary. In this paper, we propose a Scenario of Use Scheme that incorporates an Attacker Model, which characterizes the adversary against whom the speaker's privacy must be defended, and a Protector Model, which specifies the defense. We discuss the connection of the scheme with previous work on speech privacy. Finally, we present a concrete example of a specified Scenario of Use and a set of experiments about protecting speaker data against gender inference attacks while maintaining utility for Parkinson's detection.

摘要: 语音录音越来越频繁地被用于检测和监测疾病，这导致了隐私问题。除了密码学之外，语音保护还可以通过诸如扰动、解缠和重新合成等方法来解决，这些方法消除了说话人的敏感信息，留下了医学分析所需的信息。为了开发这种隐私保护方法，必须明确和系统地规定关于医疗环境和医疗专业人员需求的假设。在本文中，我们提出了一个场景使用方案，其中包括一个攻击者模型和一个保护者模型，前者描述了说话人的隐私必须被保护的对手，后者则指定了防御。我们讨论了该方案与前人在语音隐私方面的工作的联系。最后，我们给出了一个具体的使用场景的例子和一组关于保护说话人数据免受性别推理攻击的实验，同时保持了对帕金森病检测的有效性。



## **6. Adversarial Attacks on Machine Learning-Aided Visualizations**

对机器学习辅助可视化的对抗攻击 cs.CR

This version of the article has been accepted for publication, after  peer review (when applicable) but is not the Version of Record and does not  reflect post-acceptance improvements, or any corrections. The Version of  Record is available online at: http://dx.doi.org/10.1007/s12650-024-01029-2

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.02485v2) [paper-pdf](http://arxiv.org/pdf/2409.02485v2)

**Authors**: Takanori Fujiwara, Kostiantyn Kucher, Junpeng Wang, Rafael M. Martins, Andreas Kerren, Anders Ynnerman

**Abstract**: Research in ML4VIS investigates how to use machine learning (ML) techniques to generate visualizations, and the field is rapidly growing with high societal impact. However, as with any computational pipeline that employs ML processes, ML4VIS approaches are susceptible to a range of ML-specific adversarial attacks. These attacks can manipulate visualization generations, causing analysts to be tricked and their judgments to be impaired. Due to a lack of synthesis from both visualization and ML perspectives, this security aspect is largely overlooked by the current ML4VIS literature. To bridge this gap, we investigate the potential vulnerabilities of ML-aided visualizations from adversarial attacks using a holistic lens of both visualization and ML perspectives. We first identify the attack surface (i.e., attack entry points) that is unique in ML-aided visualizations. We then exemplify five different adversarial attacks. These examples highlight the range of possible attacks when considering the attack surface and multiple different adversary capabilities. Our results show that adversaries can induce various attacks, such as creating arbitrary and deceptive visualizations, by systematically identifying input attributes that are influential in ML inferences. Based on our observations of the attack surface characteristics and the attack examples, we underline the importance of comprehensive studies of security issues and defense mechanisms as a call of urgency for the ML4VIS community.

摘要: ML4VIS的研究是研究如何使用机器学习(ML)技术来生成可视化，该领域正在迅速发展，具有很高的社会影响。然而，与任何使用ML进程的计算管道一样，ML4VIS方法容易受到一系列特定于ML的对抗性攻击。这些攻击可以操纵可视化世代，导致分析师上当受骗，他们的判断受到损害。由于缺乏可视化和ML视角的综合，当前的ML4VIS文献在很大程度上忽略了这一安全方面。为了弥补这一差距，我们使用可视化和ML视角的整体视角来研究ML辅助可视化在对抗攻击中的潜在脆弱性。我们首先确定在ML辅助可视化中唯一的攻击面(即攻击入口点)。然后，我们举例说明五种不同的对抗性攻击。这些示例突出显示了在考虑攻击面和多个不同对手能力时可能的攻击范围。我们的结果表明，攻击者可以通过系统地识别对ML推理有影响的输入属性来诱导各种攻击，例如创建任意和欺骗性的可视化。根据我们对攻击表面特征和攻击实例的观察，我们强调对安全问题和防御机制进行全面研究的重要性，作为ML4VIS社区的紧急呼吁。



## **7. When Witnesses Defend: A Witness Graph Topological Layer for Adversarial Graph Learning**

当证人辩护时：对抗图学习的证人图布局层 cs.LG

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.14161v2) [paper-pdf](http://arxiv.org/pdf/2409.14161v2)

**Authors**: Naheed Anjum Arafat, Debabrota Basu, Yulia Gel, Yuzhou Chen

**Abstract**: Capitalizing on the intuitive premise that shape characteristics are more robust to perturbations, we bridge adversarial graph learning with the emerging tools from computational topology, namely, persistent homology representations of graphs. We introduce the concept of witness complex to adversarial analysis on graphs, which allows us to focus only on the salient shape characteristics of graphs, yielded by the subset of the most essential nodes (i.e., landmarks), with minimal loss of topological information on the whole graph. The remaining nodes are then used as witnesses, governing which higher-order graph substructures are incorporated into the learning process. Armed with the witness mechanism, we design Witness Graph Topological Layer (WGTL), which systematically integrates both local and global topological graph feature representations, the impact of which is, in turn, automatically controlled by the robust regularized topological loss. Given the attacker's budget, we derive the important stability guarantees of both local and global topology encodings and the associated robust topological loss. We illustrate the versatility and efficiency of WGTL by its integration with five GNNs and three existing non-topological defense mechanisms. Our extensive experiments across six datasets demonstrate that WGTL boosts the robustness of GNNs across a range of perturbations and against a range of adversarial attacks, leading to relative gains of up to 18%.

摘要: 基于形状特征对扰动的鲁棒性更强这一直观前提，我们利用计算拓扑学中的新兴工具，即图的持久同调表示，在对抗性图学习之间架起桥梁。我们将证人复合体的概念引入到图的对抗分析中，使得我们只关注图的显著形状特征，这些特征是由最重要的结点(即地标)的子集产生的，而整个图的拓扑信息损失最小。然后，剩余的节点被用作见证，控制哪些更高阶图的子结构被合并到学习过程中。结合见证机制，我们设计了见证图拓扑层(Witness Graph Topology Layer，WGTL)，该拓扑层系统地集成了局部拓扑图和全局拓扑图的特征表示，其影响由稳健的正则化拓扑损失自动控制。在给定攻击者预算的情况下，我们推导出了局部和全局拓扑编码的重要稳定性保证以及相关的稳健拓扑损失。我们通过与五个GNN和三个现有的非拓扑防御机制的集成来说明WGTL的通用性和有效性。我们在六个数据集上的广泛实验表明，WGTL提高了GNN在一系列扰动和一系列对手攻击下的健壮性，导致了高达18%的相对收益。



## **8. Adversarial Watermarking for Face Recognition**

人脸识别的对抗性水印 cs.CV

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.16056v1) [paper-pdf](http://arxiv.org/pdf/2409.16056v1)

**Authors**: Yuguang Yao, Anil Jain, Sijia Liu

**Abstract**: Watermarking is an essential technique for embedding an identifier (i.e., watermark message) within digital images to assert ownership and monitor unauthorized alterations. In face recognition systems, watermarking plays a pivotal role in ensuring data integrity and security. However, an adversary could potentially interfere with the watermarking process, significantly impairing recognition performance. We explore the interaction between watermarking and adversarial attacks on face recognition models. Our findings reveal that while watermarking or input-level perturbation alone may have a negligible effect on recognition accuracy, the combined effect of watermarking and perturbation can result in an adversarial watermarking attack, significantly degrading recognition performance. Specifically, we introduce a novel threat model, the adversarial watermarking attack, which remains stealthy in the absence of watermarking, allowing images to be correctly recognized initially. However, once watermarking is applied, the attack is activated, causing recognition failures. Our study reveals a previously unrecognized vulnerability: adversarial perturbations can exploit the watermark message to evade face recognition systems. Evaluated on the CASIA-WebFace dataset, our proposed adversarial watermarking attack reduces face matching accuracy by 67.2% with an $\ell_\infty$ norm-measured perturbation strength of ${2}/{255}$ and by 95.9% with a strength of ${4}/{255}$.

摘要: 水印是在数字图像中嵌入识别符(即水印消息)以声明所有权和监控未经授权的篡改的基本技术。在人脸识别系统中，水印在保证数据完整性和安全性方面起着举足轻重的作用。然而，敌手可能会潜在地干扰水印过程，严重影响识别性能。我们探讨了人脸识别模型中水印和敌意攻击之间的相互作用。我们的研究结果表明，虽然水印或输入级扰动单独对识别精度的影响可以忽略不计，但水印和扰动的共同作用会导致对抗性水印攻击，从而显著降低识别性能。具体地说，我们引入了一种新的威胁模型--对抗性水印攻击，该模型在没有水印的情况下仍然保持隐蔽性，使得图像能够在最初被正确识别。然而，一旦应用了水印，攻击就会被激活，从而导致识别失败。我们的研究揭示了一个以前未被认识到的漏洞：敌意扰动可以利用水印消息来逃避人脸识别系统。在CASIA-WebFace数据集上进行评估，我们提出的对抗性水印攻击在$\ell_\inty$范数测量的扰动强度为${2}/{255}$时，人脸匹配准确率降低了67.2%，而在强度为${4}/{255}$时，降低了95.9%。



## **9. Adversarial Backdoor Defense in CLIP**

CLIP中的对抗性后门防御 cs.CV

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.15968v1) [paper-pdf](http://arxiv.org/pdf/2409.15968v1)

**Authors**: Junhao Kuang, Siyuan Liang, Jiawei Liang, Kuanrong Liu, Xiaochun Cao

**Abstract**: Multimodal contrastive pretraining, exemplified by models like CLIP, has been found to be vulnerable to backdoor attacks. While current backdoor defense methods primarily employ conventional data augmentation to create augmented samples aimed at feature alignment, these methods fail to capture the distinct features of backdoor samples, resulting in suboptimal defense performance. Observations reveal that adversarial examples and backdoor samples exhibit similarities in the feature space within the compromised models. Building on this insight, we propose Adversarial Backdoor Defense (ABD), a novel data augmentation strategy that aligns features with meticulously crafted adversarial examples. This approach effectively disrupts the backdoor association. Our experiments demonstrate that ABD provides robust defense against both traditional uni-modal and multimodal backdoor attacks targeting CLIP. Compared to the current state-of-the-art defense method, CleanCLIP, ABD reduces the attack success rate by 8.66% for BadNet, 10.52% for Blended, and 53.64% for BadCLIP, while maintaining a minimal average decrease of just 1.73% in clean accuracy.

摘要: 以CLIP等模型为例的多模式对比预训练被发现容易受到后门攻击。虽然目前的后门防御方法主要使用传统的数据增强来创建针对特征对齐的扩展样本，但这些方法无法捕获后门样本的明显特征，导致防御性能不佳。观察表明，对抗性示例和后门样本在受损模型的特征空间中显示出相似之处。基于这一见解，我们提出了对抗性后门防御(ABD)，这是一种新颖的数据增强策略，将特征与精心制作的对抗性示例相结合。这种方法有效地破坏了后门关联。我们的实验表明，ABD对传统的针对CLIP的单模式和多模式后门攻击都具有很强的防御能力。与目前最先进的防御方法CleanCLIP相比，ABD使BadNet、Blend和BadCLIP的攻击成功率分别降低了8.66%、10.52%和53.64%，同时保持了仅1.73%的清洁准确率的最小平均降幅。



## **10. Can Go AIs be adversarially robust?**

Go AI能否具有对抗性强大？ cs.LG

59 pages

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2406.12843v2) [paper-pdf](http://arxiv.org/pdf/2406.12843v2)

**Authors**: Tom Tseng, Euan McLean, Kellin Pelrine, Tony T. Wang, Adam Gleave

**Abstract**: Prior work found that superhuman Go AIs can be defeated by simple adversarial strategies, especially "cyclic" attacks. In this paper, we study whether adding natural countermeasures can achieve robustness in Go, a favorable domain for robustness since it benefits from incredible average-case capability and a narrow, innately adversarial setting. We test three defenses: adversarial training on hand-constructed positions, iterated adversarial training, and changing the network architecture. We find that though some of these defenses protect against previously discovered attacks, none withstand freshly trained adversaries. Furthermore, most of the reliably effective attacks these adversaries discover are different realizations of the same overall class of cyclic attacks. Our results suggest that building robust AI systems is challenging even with extremely superhuman systems in some of the most tractable settings, and highlight two key gaps: efficient generalization in defenses, and diversity in training. For interactive examples of attacks and a link to our codebase, see https://goattack.far.ai.

摘要: 先前的工作发现，超人围棋可以通过简单的对抗性策略，特别是“循环”攻击来击败。在本文中，我们研究了添加自然对策是否可以在围棋中实现稳健性，这是一个有利于稳健性的领域，因为它受益于令人难以置信的平均情况能力和狭窄的、天生的对抗性环境。我们测试了三种防御措施：在手工搭建的阵地上进行对抗性训练，反复进行对抗性训练，以及改变网络架构。我们发现，尽管其中一些防御系统可以抵御以前发现的攻击，但没有一个能抵御新训练的对手。此外，这些攻击者发现的大多数可靠有效的攻击都是同一类循环攻击的不同实现。我们的结果表明，即使在一些最容易处理的环境中使用极其超人的系统，构建稳健的人工智能系统也是具有挑战性的，并突显了两个关键差距：防御的高效泛化和训练的多样性。有关攻击的交互式示例和我们代码库的链接，请参阅https://goattack.far.ai.



## **11. Goal-guided Generative Prompt Injection Attack on Large Language Models**

对大型语言模型的目标引导生成提示注入攻击 cs.CR

11 pages, 6 figures

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2404.07234v3) [paper-pdf](http://arxiv.org/pdf/2404.07234v3)

**Authors**: Chong Zhang, Mingyu Jin, Qinkai Yu, Chengzhi Liu, Haochen Xue, Xiaobo Jin

**Abstract**: Current large language models (LLMs) provide a strong foundation for large-scale user-oriented natural language tasks. A large number of users can easily inject adversarial text or instructions through the user interface, thus causing LLMs model security challenges. Although there is currently a large amount of research on prompt injection attacks, most of these black-box attacks use heuristic strategies. It is unclear how these heuristic strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we redefine the goal of the attack: to maximize the KL divergence between the conditional probabilities of the clean text and the adversarial text. Furthermore, we prove that maximizing the KL divergence is equivalent to maximizing the Mahalanobis distance between the embedded representation $x$ and $x'$ of the clean text and the adversarial text when the conditional probability is a Gaussian distribution and gives a quantitative relationship on $x$ and $x'$. Then we designed a simple and effective goal-guided generative prompt injection strategy (G2PIA) to find an injection text that satisfies specific constraints to achieve the optimal attack effect approximately. It is particularly noteworthy that our attack method is a query-free black-box attack method with low computational cost. Experimental results on seven LLM models and four datasets show the effectiveness of our attack method.

摘要: 现有的大型语言模型为大规模面向用户的自然语言任务提供了坚实的基础。大量用户可以很容易地通过用户界面注入敌意文本或指令，从而造成LLMS模型的安全挑战。虽然目前有大量关于即时注入攻击的研究，但这些黑盒攻击大多采用启发式策略。目前尚不清楚这些启发式策略如何与攻击成功率相关，从而有效地提高模型的稳健性。为了解决这个问题，我们重新定义了攻击的目标：最大化纯文本和敌意文本的条件概率之间的KL偏差。此外，我们证明了当条件概率为高斯分布时，最大化KL发散度等价于最大化明文和敌意文本的嵌入表示$x$和$x‘$之间的马氏距离，并给出了关于$x$和$x’$的定量关系。然后，设计了一种简单有效的目标引导生成性提示注入策略(G2PIA)，找到满足特定约束的注入文本，以近似达到最优的攻击效果。特别值得注意的是，我们的攻击方法是一种计算代价低的无查询黑盒攻击方法。在7个LLM模型和4个数据集上的实验结果表明了该攻击方法的有效性。



## **12. Efficient and Effective Model Extraction**

高效有效的模型提取 cs.CR

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.14122v2) [paper-pdf](http://arxiv.org/pdf/2409.14122v2)

**Authors**: Hongyu Zhu, Wentao Hu, Sichu Liang, Fangqi Li, Wenwen Wang, Shilin Wang

**Abstract**: Model extraction aims to create a functionally similar copy from a machine learning as a service (MLaaS) API with minimal overhead, typically for illicit profit or as a precursor to further attacks, posing a significant threat to the MLaaS ecosystem. However, recent studies have shown that model extraction is highly inefficient, particularly when the target task distribution is unavailable. In such cases, even substantially increasing the attack budget fails to produce a sufficiently similar replica, reducing the adversary's motivation to pursue extraction attacks. In this paper, we revisit the elementary design choices throughout the extraction lifecycle. We propose an embarrassingly simple yet dramatically effective algorithm, Efficient and Effective Model Extraction (E3), focusing on both query preparation and training routine. E3 achieves superior generalization compared to state-of-the-art methods while minimizing computational costs. For instance, with only 0.005 times the query budget and less than 0.2 times the runtime, E3 outperforms classical generative model based data-free model extraction by an absolute accuracy improvement of over 50% on CIFAR-10. Our findings underscore the persistent threat posed by model extraction and suggest that it could serve as a valuable benchmarking algorithm for future security evaluations.

摘要: 模型提取旨在以最低的开销从机器学习即服务(MLaaS)API创建功能相似的副本，通常用于非法利润或作为进一步攻击的前兆，对MLaaS生态系统构成重大威胁。然而，最近的研究表明，模型提取的效率非常低，特别是在目标任务分配不可用的情况下。在这种情况下，即使大幅增加攻击预算，也无法产生足够相似的副本，从而降低了对手进行提取攻击的动机。在本文中，我们回顾了整个提取生命周期中的基本设计选择。我们提出了一种简单得令人尴尬的高效算法--高效有效的模型提取算法(E3)，它同时关注查询准备和训练过程。与最先进的方法相比，E3实现了更好的泛化，同时最大限度地减少了计算成本。例如，在CIFAR-10上，E3只需要0.005倍的查询预算和不到0.2倍的运行时间，就比基于经典生成模型的无数据模型提取的准确率提高了50%以上。我们的发现强调了模型提取带来的持续威胁，并表明它可以作为未来安全评估的一个有价值的基准算法。



## **13. Toward Mixture-of-Experts Enabled Trustworthy Semantic Communication for 6G Networks**

实现6G网络中值得信赖的专家混合语义通信 cs.NI

8 pages, 3 figures

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.15695v1) [paper-pdf](http://arxiv.org/pdf/2409.15695v1)

**Authors**: Jiayi He, Xiaofeng Luo, Jiawen Kang, Hongyang Du, Zehui Xiong, Ci Chen, Dusit Niyato, Xuemin Shen

**Abstract**: Semantic Communication (SemCom) plays a pivotal role in 6G networks, offering a viable solution for future efficient communication. Deep Learning (DL)-based semantic codecs further enhance this efficiency. However, the vulnerability of DL models to security threats, such as adversarial attacks, poses significant challenges for practical applications of SemCom systems. These vulnerabilities enable attackers to tamper with messages and eavesdrop on private information, especially in wireless communication scenarios. Although existing defenses attempt to address specific threats, they often fail to simultaneously handle multiple heterogeneous attacks. To overcome this limitation, we introduce a novel Mixture-of-Experts (MoE)-based SemCom system. This system comprises a gating network and multiple experts, each specializing in different security challenges. The gating network adaptively selects suitable experts to counter heterogeneous attacks based on user-defined security requirements. Multiple experts collaborate to accomplish semantic communication tasks while meeting the security requirements of users. A case study in vehicular networks demonstrates the efficacy of the MoE-based SemCom system. Simulation results show that the proposed MoE-based SemCom system effectively mitigates concurrent heterogeneous attacks, with minimal impact on downstream task accuracy.

摘要: 语义通信(SemCom)在6G网络中起着举足轻重的作用，为未来高效的通信提供了可行的解决方案。基于深度学习的语义编解码器进一步提高了这一效率。然而，DL模型对敌意攻击等安全威胁的脆弱性给SemCom系统的实际应用带来了巨大的挑战。这些漏洞使攻击者能够篡改消息并窃听私人信息，特别是在无线通信场景中。尽管现有的防御系统试图应对特定的威胁，但它们往往无法同时处理多个不同种类的攻击。为了克服这一局限性，我们引入了一种新的基于混合专家(MOE)的SemCom系统。该系统由一个门控网络和多个专家组成，每个专家都专门研究不同的安全挑战。门控网络根据用户定义的安全需求自适应地选择合适的专家来对抗异类攻击。多专家协作完成语义通信任务，同时满足用户的安全需求。车载网络中的一个案例研究证明了基于MOE的SemCom系统的有效性。仿真结果表明，提出的基于MOE的SemCom系统在对下游任务精度影响最小的情况下，有效地缓解了并发的异质攻击。



## **14. Data Poisoning-based Backdoor Attack Framework against Supervised Learning Rules of Spiking Neural Networks**

针对尖峰神经网络监督学习规则的基于数据中毒的后门攻击框架 cs.CR

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.15670v1) [paper-pdf](http://arxiv.org/pdf/2409.15670v1)

**Authors**: Lingxin Jin, Meiyu Lin, Wei Jiang, Jinyu Zhan

**Abstract**: Spiking Neural Networks (SNNs), the third generation neural networks, are known for their low energy consumption and high robustness. SNNs are developing rapidly and can compete with Artificial Neural Networks (ANNs) in many fields. To ensure that the widespread use of SNNs does not cause serious security incidents, much research has been conducted to explore the robustness of SNNs under adversarial sample attacks. However, many other unassessed security threats exist, such as highly stealthy backdoor attacks. Therefore, to fill the research gap in this and further explore the security vulnerabilities of SNNs, this paper explores the robustness performance of SNNs trained by supervised learning rules under backdoor attacks. Specifically, the work herein includes: i) We propose a generic backdoor attack framework that can be launched against the training process of existing supervised learning rules and covers all learnable dataset types of SNNs. ii) We analyze the robustness differences between different learning rules and between SNN and ANN, which suggests that SNN no longer has inherent robustness under backdoor attacks. iii) We reveal the vulnerability of conversion-dependent learning rules caused by backdoor migration and further analyze the migration ability during the conversion process, finding that the backdoor migration rate can even exceed 99%. iv) Finally, we discuss potential countermeasures against this kind of backdoor attack and its technical challenges and point out several promising research directions.

摘要: 尖峰神经网络(SNN)是第三代神经网络，以其低能耗、高鲁棒性而著称。神经网络发展迅速，在许多领域都能与人工神经网络相抗衡。为了确保SNN的广泛使用不会导致严重的安全事件，人们对SNN在敌意样本攻击下的健壮性进行了大量的研究。然而，还有许多其他未评估的安全威胁，例如高度隐蔽的后门攻击。因此，为了填补这方面的研究空白，进一步挖掘SNN的安全漏洞，本文研究了监督学习规则训练的SNN在后门攻击下的稳健性。具体地，本文的工作包括：i)提出了一个通用的后门攻击框架，该框架可以针对现有监督学习规则的训练过程发起攻击，并且覆盖了所有可学习的SNN数据集类型。Ii)分析了不同学习规则之间以及SNN和ANN之间的健壮性差异，表明SNN在后门攻击下不再具有固有的健壮性。Iii)揭示了借壳迁移导致的依赖于转换的学习规则的脆弱性，并进一步分析了转换过程中的迁移能力，发现借壳迁移率甚至可以超过99%。最后，我们讨论了针对这种后门攻击的潜在对策及其技术挑战，并指出了几个有前途的研究方向。



## **15. Adversarial Attacks to Multi-Modal Models**

对多模式模型的对抗攻击 cs.CR

To appear in the ACM Workshop on Large AI Systems and Models with  Privacy and Safety Analysis 2024 (LAMPS '24)

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.06793v2) [paper-pdf](http://arxiv.org/pdf/2409.06793v2)

**Authors**: Zhihao Dou, Xin Hu, Haibo Yang, Zhuqing Liu, Minghong Fang

**Abstract**: Multi-modal models have gained significant attention due to their powerful capabilities. These models effectively align embeddings across diverse data modalities, showcasing superior performance in downstream tasks compared to their unimodal counterparts. Recent study showed that the attacker can manipulate an image or audio file by altering it in such a way that its embedding matches that of an attacker-chosen targeted input, thereby deceiving downstream models. However, this method often underperforms due to inherent disparities in data from different modalities. In this paper, we introduce CrossFire, an innovative approach to attack multi-modal models. CrossFire begins by transforming the targeted input chosen by the attacker into a format that matches the modality of the original image or audio file. We then formulate our attack as an optimization problem, aiming to minimize the angular deviation between the embeddings of the transformed input and the modified image or audio file. Solving this problem determines the perturbations to be added to the original media. Our extensive experiments on six real-world benchmark datasets reveal that CrossFire can significantly manipulate downstream tasks, surpassing existing attacks. Additionally, we evaluate six defensive strategies against CrossFire, finding that current defenses are insufficient to counteract our CrossFire.

摘要: 多通道模型因其强大的性能而备受关注。这些模型有效地调整了跨不同数据模式的嵌入，在下游任务中展示了与单峰对应的卓越性能。最近的研究表明，攻击者可以通过改变图像或音频文件的嵌入方式来操纵它，使其嵌入与攻击者选择的目标输入相匹配，从而欺骗下游模型。然而，由于来自不同模式的数据的内在差异，该方法常常表现不佳。在本文中，我们介绍了一种创新的攻击多通道模型的方法--CrossFire。CrossFire首先将攻击者选择的目标输入转换为与原始图像或音频文件的形态相匹配的格式。然后，我们将攻击描述为一个优化问题，旨在最小化转换后的输入和修改后的图像或音频文件的嵌入之间的角度偏差。解决此问题将确定要添加到原始介质的扰动。我们在六个真实世界基准数据集上的广泛实验表明，CrossFire可以显著操纵下游任务，超过现有的攻击。此外，我们评估了六种防御交叉火力的策略，发现目前的防御不足以对抗我们的交叉火力。



## **16. Interpretability-Guided Test-Time Adversarial Defense**

可解释性引导的测试时对抗性辩护 cs.CV

ECCV 2024. Project Page:  https://lilywenglab.github.io/Interpretability-Guided-Defense/

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2409.15190v1) [paper-pdf](http://arxiv.org/pdf/2409.15190v1)

**Authors**: Akshay Kulkarni, Tsui-Wei Weng

**Abstract**: We propose a novel and low-cost test-time adversarial defense by devising interpretability-guided neuron importance ranking methods to identify neurons important to the output classes. Our method is a training-free approach that can significantly improve the robustness-accuracy tradeoff while incurring minimal computational overhead. While being among the most efficient test-time defenses (4x faster), our method is also robust to a wide range of black-box, white-box, and adaptive attacks that break previous test-time defenses. We demonstrate the efficacy of our method for CIFAR10, CIFAR100, and ImageNet-1k on the standard RobustBench benchmark (with average gains of 2.6%, 4.9%, and 2.8% respectively). We also show improvements (average 1.5%) over the state-of-the-art test-time defenses even under strong adaptive attacks.

摘要: 我们通过设计可解释性引导的神经元重要性排名方法来识别对输出类别重要的神经元，提出了一种新颖且低成本的测试时对抗防御。我们的方法是一种免训练方法，可以显着改善稳健性与准确性的权衡，同时产生最小的计算负担。虽然我们的方法是最有效的测试时防御之一（速度快4倍），但我们的方法也对破坏之前测试时防御的各种黑匣子、白盒和自适应攻击具有鲁棒性。我们在标准RobustBench基准上证明了我们的方法对CIFAR 10、CIFAR 100和ImageNet-1 k的有效性（平均收益分别为2.6%、4.9%和2.8%）。即使在强适应性攻击下，我们还显示出与最先进的测试时防御相比的改进（平均1.5%）。



## **17. Log-normal Mutations and their Use in Detecting Surreptitious Fake Images**

对正态突变及其在检测隐秘虚假图像中的应用 cs.AI

log-normal mutations and their use in detecting surreptitious fake  images

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2409.15119v1) [paper-pdf](http://arxiv.org/pdf/2409.15119v1)

**Authors**: Ismail Labiad, Thomas Bäck, Pierre Fernandez, Laurent Najman, Tom Sanders, Furong Ye, Mariia Zameshina, Olivier Teytaud

**Abstract**: In many cases, adversarial attacks are based on specialized algorithms specifically dedicated to attacking automatic image classifiers. These algorithms perform well, thanks to an excellent ad hoc distribution of initial attacks. However, these attacks are easily detected due to their specific initial distribution. We therefore consider other black-box attacks, inspired from generic black-box optimization tools, and in particular the log-normal algorithm.   We apply the log-normal method to the attack of fake detectors, and get successful attacks: importantly, these attacks are not detected by detectors specialized on classical adversarial attacks. Then, combining these attacks and deep detection, we create improved fake detectors.

摘要: 在许多情况下，对抗性攻击基于专门致力于攻击自动图像分类器的专门算法。由于初始攻击的良好临时分布，这些算法表现良好。然而，由于这些攻击的特定初始分布，因此很容易被检测到。因此，我们考虑其他黑匣子攻击，其灵感来自通用黑匣子优化工具，特别是对log normal算法。   我们将log normal方法应用于假检测器的攻击，并获得成功的攻击：重要的是，这些攻击不会被专门研究经典对抗攻击的检测器检测到。然后，将这些攻击和深度检测结合起来，我们创建了改进的虚假检测器。



## **18. SHFL: Secure Hierarchical Federated Learning Framework for Edge Networks**

SHFL：边缘网络的安全分层联邦学习框架 cs.LG

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2409.15067v1) [paper-pdf](http://arxiv.org/pdf/2409.15067v1)

**Authors**: Omid Tavallaie, Kanchana Thilakarathna, Suranga Seneviratne, Aruna Seneviratne, Albert Y. Zomaya

**Abstract**: Federated Learning (FL) is a distributed machine learning paradigm designed for privacy-sensitive applications that run on resource-constrained devices with non-Identically and Independently Distributed (IID) data. Traditional FL frameworks adopt the client-server model with a single-level aggregation (AGR) process, where the server builds the global model by aggregating all trained local models received from client devices. However, this conventional approach encounters challenges, including susceptibility to model/data poisoning attacks. In recent years, advancements in the Internet of Things (IoT) and edge computing have enabled the development of hierarchical FL systems with a two-level AGR process running at edge and cloud servers. In this paper, we propose a Secure Hierarchical FL (SHFL) framework to address poisoning attacks in hierarchical edge networks. By aggregating trained models at the edge, SHFL employs two novel methods to address model/data poisoning attacks in the presence of client adversaries: 1) a client selection algorithm running at the edge for choosing IoT devices to participate in training, and 2) a model AGR method designed based on convex optimization theory to reduce the impact of edge models from networks with adversaries in the process of computing the global model (at the cloud level). The evaluation results reveal that compared to state-of-the-art methods, SHFL significantly increases the maximum accuracy achieved by the global model in the presence of client adversaries applying model/data poisoning attacks.

摘要: 联合学习(FL)是一种分布式机器学习范例，专为在具有非相同和独立分布(IID)数据的资源受限设备上运行的隐私敏感应用程序而设计。传统的FL框架采用具有单级聚合(AGR)过程的客户端-服务器模型，其中服务器通过聚合从客户端设备接收的所有训练的本地模型来构建全局模型。然而，这种传统的方法遇到了挑战，包括对模型/数据中毒攻击的敏感性。近年来，物联网(IoT)和边缘计算的进步使分层FL系统的开发成为可能，该系统在边缘和云服务器上运行两级AGR进程。针对层次化边缘网络中的中毒攻击，提出了一种安全的层次化FL(SHFL)框架。通过在边缘聚集训练模型，SHFL采用了两种新的方法来应对模型/数据中毒攻击：1)在边缘运行的用于选择物联网设备参与训练的客户端选择算法；2)基于凸优化理论设计的模型AGR方法，在计算全局模型的过程中(在云层)减少来自具有对手的网络的边缘模型的影响。评估结果表明，与最先进的方法相比，SHFL显著提高了在客户端对手应用模型/数据中毒攻击的情况下全局模型所实现的最大精度。



## **19. Improving Adversarial Robustness for 3D Point Cloud Recognition at Test-Time through Purified Self-Training**

通过净化自我训练提高测试时3D点云识别的对抗鲁棒性 cs.CV

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2409.14940v1) [paper-pdf](http://arxiv.org/pdf/2409.14940v1)

**Authors**: Jinpeng Lin, Xulei Yang, Tianrui Li, Xun Xu

**Abstract**: Recognizing 3D point cloud plays a pivotal role in many real-world applications. However, deploying 3D point cloud deep learning model is vulnerable to adversarial attacks. Despite many efforts into developing robust model by adversarial training, they may become less effective against emerging attacks. This limitation motivates the development of adversarial purification which employs generative model to mitigate the impact of adversarial attacks. In this work, we highlight the remaining challenges from two perspectives. First, the purification based method requires retraining the classifier on purified samples which introduces additional computation overhead. Moreover, in a more realistic scenario, testing samples arrives in a streaming fashion and adversarial samples are not isolated from clean samples. These challenges motivates us to explore dynamically update model upon observing testing samples. We proposed a test-time purified self-training strategy to achieve this objective. Adaptive thresholding and feature distribution alignment are introduced to improve the robustness of self-training. Extensive results on different adversarial attacks suggest the proposed method is complementary to purification based method in handling continually changing adversarial attacks on the testing data stream.

摘要: 识别三维点云在许多实际应用中起着举足轻重的作用。然而，部署三维点云深度学习模型容易受到敌意攻击。尽管许多人努力通过对抗性训练来开发健壮的模型，但它们对新出现的攻击可能会变得不那么有效。这种局限性促使了对抗性净化技术的发展，它采用生成性模型来减轻对抗性攻击的影响。在这项工作中，我们从两个角度强调剩余的挑战。首先，基于纯化的方法需要对纯化的样本重新训练分类器，这带来了额外的计算开销。此外，在更现实的场景中，测试样本以流的方式到达，而敌意样本并不从干净样本中分离出来。这些挑战促使我们在观察测试样本的基础上探索动态更新模型。为了实现这一目标，我们提出了一种测试时间净化自我训练策略。引入自适应阈值和特征分布对齐，提高自训练的稳健性。在不同类型的敌意攻击上的广泛结果表明，该方法在处理不断变化的对测试数据流的敌意攻击方面是对基于净化的方法的补充。



## **20. Attack Atlas: A Practitioner's Perspective on Challenges and Pitfalls in Red Teaming GenAI**

攻击地图集：从业者对红色团队GenAI挑战和陷阱的看法 cs.CR

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2409.15398v1) [paper-pdf](http://arxiv.org/pdf/2409.15398v1)

**Authors**: Ambrish Rawat, Stefan Schoepf, Giulio Zizzo, Giandomenico Cornacchia, Muhammad Zaid Hameed, Kieran Fraser, Erik Miehling, Beat Buesser, Elizabeth M. Daly, Mark Purcell, Prasanna Sattigeri, Pin-Yu Chen, Kush R. Varshney

**Abstract**: As generative AI, particularly large language models (LLMs), become increasingly integrated into production applications, new attack surfaces and vulnerabilities emerge and put a focus on adversarial threats in natural language and multi-modal systems. Red-teaming has gained importance in proactively identifying weaknesses in these systems, while blue-teaming works to protect against such adversarial attacks. Despite growing academic interest in adversarial risks for generative AI, there is limited guidance tailored for practitioners to assess and mitigate these challenges in real-world environments. To address this, our contributions include: (1) a practical examination of red- and blue-teaming strategies for securing generative AI, (2) identification of key challenges and open questions in defense development and evaluation, and (3) the Attack Atlas, an intuitive framework that brings a practical approach to analyzing single-turn input attacks, placing it at the forefront for practitioners. This work aims to bridge the gap between academic insights and practical security measures for the protection of generative AI systems.

摘要: 随着产生式人工智能，特别是大型语言模型(LLM)越来越多地集成到生产应用中，新的攻击面和漏洞出现，并将重点放在自然语言和多模式系统中的对抗性威胁上。红团队在主动识别这些系统中的弱点方面变得越来越重要，而蓝团队的工作是防止这种对手攻击。尽管学术界对生成性人工智能的对抗风险越来越感兴趣，但为实践者量身定做的指导意见有限，以评估和缓解现实世界环境中的这些挑战。为了解决这个问题，我们的贡献包括：(1)确保生成性人工智能的红色和蓝色团队战略的实践研究，(2)识别防御开发和评估中的关键挑战和开放问题，以及(3)攻击图集，这是一个直观的框架，为分析单回合输入攻击带来了实用方法，使其处于实践者的前沿。这项工作旨在弥合学术见解和保护生成性人工智能系统的实际安全措施之间的差距。



## **21. Transfer-based Adversarial Poisoning Attacks for Online (MIMO-)Deep Receviers**

基于传输的针对在线（MMO-）深度回收者的对抗中毒攻击 eess.SP

15 pages, 14 figures

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2409.02430v3) [paper-pdf](http://arxiv.org/pdf/2409.02430v3)

**Authors**: Kunze Wu, Weiheng Jiang, Dusit Niyato, Yinghuan Li, Chuang Luo

**Abstract**: Recently, the design of wireless receivers using deep neural networks (DNNs), known as deep receivers, has attracted extensive attention for ensuring reliable communication in complex channel environments. To adapt quickly to dynamic channels, online learning has been adopted to update the weights of deep receivers with over-the-air data (e.g., pilots). However, the fragility of neural models and the openness of wireless channels expose these systems to malicious attacks. To this end, understanding these attack methods is essential for robust receiver design. In this paper, we propose a transfer-based adversarial poisoning attack method for online receivers. Without knowledge of the attack target, adversarial perturbations are injected to the pilots, poisoning the online deep receiver and impairing its ability to adapt to dynamic channels and nonlinear effects. In particular, our attack method targets Deep Soft Interference Cancellation (DeepSIC)[1] using online meta-learning. As a classical model-driven deep receiver, DeepSIC incorporates wireless domain knowledge into its architecture. This integration allows it to adapt efficiently to time-varying channels with only a small number of pilots, achieving optimal performance in a multi-input and multi-output (MIMO) scenario. The deep receiver in this scenario has a number of applications in the field of wireless communication, which motivates our study of the attack methods targeting it. Specifically, we demonstrate the effectiveness of our attack in simulations on synthetic linear, synthetic nonlinear, static, and COST 2100 channels. Simulation results indicate that the proposed poisoning attack significantly reduces the performance of online receivers in rapidly changing scenarios.

摘要: 近年来，利用深度神经网络(DNN)设计无线接收器，即深度接收器，因其能在复杂的信道环境中保证可靠的通信而受到广泛关注。为了快速适应动态信道，已采用在线学习来使用空中数据(例如，导频)来更新深度接收器的权重。然而，神经模型的脆弱性和无线通道的开放性使这些系统面临恶意攻击。为此，了解这些攻击方法对于稳健的接收器设计至关重要。提出了一种针对在线接收者的基于传输的对抗性中毒攻击方法。在不知道攻击目标的情况下，对抗性扰动被注入飞行员，毒害在线深度接收器，并削弱其适应动态通道和非线性效应的能力。特别是，我们的攻击方法针对使用在线元学习的深度软干扰消除(DeepSIC)[1]。作为一种经典的模型驱动的深度接收器，DeepSIC将无线领域的知识融入到其体系结构中。这种集成使其能够有效地适应只需少量导频的时变信道，从而在多输入多输出(MIMO)场景中实现最佳性能。这种场景中的深度接收器在无线通信领域有着广泛的应用，这促使我们研究针对它的攻击方法。具体地说，我们在合成线性、合成非线性、静态和COST 2100通道上的仿真中证明了我们的攻击的有效性。仿真结果表明，所提出的中毒攻击显著降低了在线接收者在快速变化的场景中的性能。



## **22. Backtracking Improves Generation Safety**

回溯提高发电安全性 cs.LG

**SubmitDate**: 2024-09-22    [abs](http://arxiv.org/abs/2409.14586v1) [paper-pdf](http://arxiv.org/pdf/2409.14586v1)

**Authors**: Yiming Zhang, Jianfeng Chi, Hailey Nguyen, Kartikeya Upasani, Daniel M. Bikel, Jason Weston, Eric Michael Smith

**Abstract**: Text generation has a fundamental limitation almost by definition: there is no taking back tokens that have been generated, even when they are clearly problematic. In the context of language model safety, when a partial unsafe generation is produced, language models by their nature tend to happily keep on generating similarly unsafe additional text. This is in fact how safety alignment of frontier models gets circumvented in the wild, despite great efforts in improving their safety. Deviating from the paradigm of approaching safety alignment as prevention (decreasing the probability of harmful responses), we propose backtracking, a technique that allows language models to "undo" and recover from their own unsafe generation through the introduction of a special [RESET] token. Our method can be incorporated into either SFT or DPO training to optimize helpfulness and harmlessness. We show that models trained to backtrack are consistently safer than baseline models: backtracking Llama-3-8B is four times more safe than the baseline model (6.1\% $\to$ 1.5\%) in our evaluations without regression in helpfulness. Our method additionally provides protection against four adversarial attacks including an adaptive attack, despite not being trained to do so.

摘要: 几乎从定义上看，文本生成有一个基本限制：即使生成的令牌明显有问题，也不能收回它们。在语言模型安全的上下文中，当产生部分不安全的生成时，语言模型本身倾向于愉快地继续生成类似的不安全的附加文本。事实上，这就是前沿模型的安全校准在野外被绕过的方式，尽管他们付出了巨大的努力来提高它们的安全性。背离了将安全对齐作为预防的范例(降低有害响应的概率)，我们提出了回溯，这是一种允许语言模型通过引入特殊的[重置]令牌来“撤消”并从其自身的不安全生成中恢复的技术。我们的方法可以结合到SFT或DPO培训中，以优化帮助和无害。我们表明，经过回溯训练的模型始终比基线模型更安全：在我们的评估中，回溯Llama-3-8B比基线模型(6.1美元到1.5美元)安全四倍，而不会有帮助的回归。我们的方法还提供了对包括自适应攻击在内的四种对抗性攻击的保护，尽管没有接受过这样的培训。



## **23. Enhancing LLM-based Autonomous Driving Agents to Mitigate Perception Attacks**

增强基于LLM的自动驾驶代理以缓解感知攻击 cs.CR

**SubmitDate**: 2024-09-22    [abs](http://arxiv.org/abs/2409.14488v1) [paper-pdf](http://arxiv.org/pdf/2409.14488v1)

**Authors**: Ruoyu Song, Muslum Ozgur Ozmen, Hyungsub Kim, Antonio Bianchi, Z. Berkay Celik

**Abstract**: There is a growing interest in integrating Large Language Models (LLMs) with autonomous driving (AD) systems. However, AD systems are vulnerable to attacks against their object detection and tracking (ODT) functions. Unfortunately, our evaluation of four recent LLM agents against ODT attacks shows that the attacks are 63.26% successful in causing them to crash or violate traffic rules due to (1) misleading memory modules that provide past experiences for decision making, (2) limitations of prompts in identifying inconsistencies, and (3) reliance on ground truth perception data.   In this paper, we introduce Hudson, a driving reasoning agent that extends prior LLM-based driving systems to enable safer decision making during perception attacks while maintaining effectiveness under benign conditions. Hudson achieves this by first instrumenting the AD software to collect real-time perception results and contextual information from the driving scene. This data is then formalized into a domain-specific language (DSL). To guide the LLM in detecting and making safe control decisions during ODT attacks, Hudson translates the DSL into natural language, along with a list of custom attack detection instructions. Following query execution, Hudson analyzes the LLM's control decision to understand its causal reasoning process.   We evaluate the effectiveness of Hudson using a proprietary LLM (GPT-4) and two open-source LLMs (Llama and Gemma) in various adversarial driving scenarios. GPT-4, Llama, and Gemma achieve, on average, an attack detection accuracy of 83. 3%, 63. 6%, and 73. 6%. Consequently, they make safe control decisions in 86.4%, 73.9%, and 80% of the attacks. Our results, following the growing interest in integrating LLMs into AD systems, highlight the strengths of LLMs and their potential to detect and mitigate ODT attacks.

摘要: 人们对将大型语言模型(LLM)与自动驾驶(AD)系统进行集成的兴趣日益浓厚。然而，AD系统的目标检测和跟踪(ODT)功能容易受到攻击。不幸的是，我们对最近针对ODT攻击的四个LLM代理的评估显示，由于(1)为决策提供过去经验的误导性存储模块，(2)在识别不一致方面的提示限制，以及(3)对地面真相感知数据的依赖，这些攻击在导致它们崩溃或违反交通规则方面的成功率为63.26%。在本文中，我们引入了Hudson，一个驾驶推理代理，它扩展了现有的基于LLM的驾驶系统，使之能够在感知攻击时做出更安全的决策，同时在良性条件下保持有效性。哈德森首先利用广告软件从驾驶场景中收集实时感知结果和背景信息，从而实现了这一点。然后将这些数据形式化为特定于域的语言(DSL)。为了指导LLM在ODT攻击期间检测和做出安全的控制决策，Hudson将DSL翻译成自然语言，并提供了一系列自定义攻击检测指令。在查询执行之后，Hudson分析LLM的控制决策以理解其因果推理过程。我们使用一个专有的LLM(GPT-4)和两个开源的LLM(Llama和Gema)来评估Hudson在各种对抗性驾驶场景中的有效性。GPT-4、Llama和Gema的攻击检测准确率平均为83。3%，63.6%和73%。6%。因此，他们在86.4%、73.9%和80%的攻击中做出了安全控制决策。随着人们对将LLMS集成到AD系统中的兴趣与日俱增，我们的结果突显了LLMS的优势及其检测和缓解ODT攻击的潜力。



## **24. Stop Reasoning! When Multimodal LLM with Chain-of-Thought Reasoning Meets Adversarial Image**

停止推理！当具有思想链推理的多模式LLM遇到敌对形象时 cs.CV

**SubmitDate**: 2024-09-22    [abs](http://arxiv.org/abs/2402.14899v3) [paper-pdf](http://arxiv.org/pdf/2402.14899v3)

**Authors**: Zefeng Wang, Zhen Han, Shuo Chen, Fan Xue, Zifeng Ding, Xun Xiao, Volker Tresp, Philip Torr, Jindong Gu

**Abstract**: Multimodal LLMs (MLLMs) with a great ability of text and image understanding have received great attention. To achieve better reasoning with MLLMs, Chain-of-Thought (CoT) reasoning has been widely explored, which further promotes MLLMs' explainability by giving intermediate reasoning steps. Despite the strong power demonstrated by MLLMs in multimodal reasoning, recent studies show that MLLMs still suffer from adversarial images. This raises the following open questions: Does CoT also enhance the adversarial robustness of MLLMs? What do the intermediate reasoning steps of CoT entail under adversarial attacks? To answer these questions, we first generalize existing attacks to CoT-based inferences by attacking the two main components, i.e., rationale and answer. We find that CoT indeed improves MLLMs' adversarial robustness against the existing attack methods by leveraging the multi-step reasoning process, but not substantially. Based on our findings, we further propose a novel attack method, termed as stop-reasoning attack, that attacks the model while bypassing the CoT reasoning process. Experiments on three MLLMs and two visual reasoning datasets verify the effectiveness of our proposed method. We show that stop-reasoning attack can result in misled predictions and outperform baseline attacks by a significant margin.

摘要: 具有很强的文本和图像理解能力的多模式LLMS(多模式LLMS)受到了极大的关注。为了更好地实现MLLMS的推理，思想链(COT)推理得到了广泛的探索，它通过给出中间推理步骤来进一步提高MLLMS的可解释性。尽管MLLMS在多通道推理中表现出强大的能力，但最近的研究表明，MLLMS仍然受到敌意图像的影响。这提出了以下悬而未决的问题：COT是否也增强了MLLMS的对抗健壮性？在对抗性攻击下，COT的中间推理步骤意味着什么？为了回答这些问题，我们首先通过攻击两个主要组成部分，即基本原理和答案，将现有的攻击概括为基于CoT的推理。我们发现，COT确实通过利用多步推理过程提高了MLLMS对现有攻击方法的对抗健壮性，但并不显著。基于我们的发现，我们进一步提出了一种新的攻击方法，称为停止推理攻击，它在攻击模型的同时绕过了COT推理过程。在三个MLLMS和两个视觉推理数据集上的实验验证了该方法的有效性。我们表明，停止推理攻击可以导致误导预测，并显著优于基线攻击。



## **25. Cloud Adversarial Example Generation for Remote Sensing Image Classification**

用于遥感图像分类的云对抗示例生成 cs.CV

**SubmitDate**: 2024-09-21    [abs](http://arxiv.org/abs/2409.14240v1) [paper-pdf](http://arxiv.org/pdf/2409.14240v1)

**Authors**: Fei Ma, Yuqiang Feng, Fan Zhang, Yongsheng Zhou

**Abstract**: Most existing adversarial attack methods for remote sensing images merely add adversarial perturbations or patches, resulting in unnatural modifications. Clouds are common atmospheric effects in remote sensing images. Generating clouds on these images can produce adversarial examples better aligning with human perception. In this paper, we propose a Perlin noise based cloud generation attack method. Common Perlin noise based cloud generation is a random, non-optimizable process, which cannot be directly used to attack the target models. We design a Perlin Gradient Generator Network (PGGN), which takes a gradient parameter vector as input and outputs the grids of Perlin noise gradient vectors at different scales. After a series of computations based on the gradient vectors, cloud masks at corresponding scales can be produced. These cloud masks are then weighted and summed depending on a mixing coefficient vector and a scaling factor to produce the final cloud masks. The gradient vector, coefficient vector and scaling factor are collectively represented as a cloud parameter vector, transforming the cloud generation into a black-box optimization problem. The Differential Evolution (DE) algorithm is employed to solve for the optimal solution of the cloud parameter vector, achieving a query-based black-box attack. Detailed experiments confirm that this method has strong attack capabilities and achieves high query efficiency. Additionally, we analyze the transferability of the generated adversarial examples and their robustness in adversarial defense scenarios.

摘要: 现有的遥感图像对抗性攻击方法大多只是添加对抗性的扰动或补丁，造成不自然的修改。云是遥感图像中常见的大气效应。在这些图像上生成云层可以产生更符合人类感知的对抗性例子。本文提出了一种基于Perlin噪声的云生成攻击方法。普通的基于Perlin噪声的云生成是一个随机的、不可优化的过程，不能直接用于攻击目标模型。我们设计了一种Perlin梯度生成网络(PGGN)，它以一个梯度参数向量作为输入，输出不同尺度上的Perlin噪声梯度向量网格。经过一系列基于梯度向量的计算，可以生成相应尺度的云掩模。然后根据混合系数向量和比例因子对这些云遮罩进行加权和求和，以产生最终的云遮罩。将梯度向量、系数向量和比例因子统一表示为云参数向量，将云的生成问题转化为黑盒优化问题。利用差分进化(DE)算法求解云参数向量的最优解，实现了基于查询的黑盒攻击。详细的实验证明，该方法具有较强的攻击能力，查询效率较高。此外，我们还分析了生成的对抗性实例的可转移性以及它们在对抗性防御场景中的健壮性。



## **26. Adversarial Attacks on Parts of Speech: An Empirical Study in Text-to-Image Generation**

对词性的对抗性攻击：文本到图像生成中的实证研究 cs.CL

Findings of the EMNLP 2024

**SubmitDate**: 2024-09-21    [abs](http://arxiv.org/abs/2409.15381v1) [paper-pdf](http://arxiv.org/pdf/2409.15381v1)

**Authors**: G M Shahariar, Jia Chen, Jiachen Li, Yue Dong

**Abstract**: Recent studies show that text-to-image (T2I) models are vulnerable to adversarial attacks, especially with noun perturbations in text prompts. In this study, we investigate the impact of adversarial attacks on different POS tags within text prompts on the images generated by T2I models. We create a high-quality dataset for realistic POS tag token swapping and perform gradient-based attacks to find adversarial suffixes that mislead T2I models into generating images with altered tokens. Our empirical results show that the attack success rate (ASR) varies significantly among different POS tag categories, with nouns, proper nouns, and adjectives being the easiest to attack. We explore the mechanism behind the steering effect of adversarial suffixes, finding that the number of critical tokens and content fusion vary among POS tags, while features like suffix transferability are consistent across categories. We have made our implementation publicly available at - https://github.com/shahariar-shibli/Adversarial-Attack-on-POS-Tags.

摘要: 最近的研究表明，文本到图像(T2I)模型容易受到敌意攻击，特别是在文本提示中存在名词扰动的情况下。在这项研究中，我们调查了对抗性攻击对文本提示中不同的POS标签对T2I模型生成的图像的影响。我们为真实的POS标签令牌交换创建了一个高质量的数据集，并执行基于梯度的攻击来发现误导T2I模型生成具有更改的令牌的图像的敌意后缀。实验结果表明，不同的词性标签类别的攻击成功率(ASR)差异很大，其中名词、专有名词和形容词最容易受到攻击。我们探索了对抗性后缀引导效应背后的机制，发现关键标记的数量和内容融合在词性标签之间是不同的，而后缀可转移性等特征在不同类别之间是一致的。我们已经在-https://github.com/shahariar-shibli/Adversarial-Attack-on-POS-Tags.上公开了我们的实现



## **27. RAMP: Boosting Adversarial Robustness Against Multiple $l_p$ Perturbations for Universal Robustness**

RAMP：针对多重$l_p$扰动增强对抗稳健性，以实现普遍稳健性 cs.LG

**SubmitDate**: 2024-09-21    [abs](http://arxiv.org/abs/2402.06827v2) [paper-pdf](http://arxiv.org/pdf/2402.06827v2)

**Authors**: Enyi Jiang, Gagandeep Singh

**Abstract**: Most existing works focus on improving robustness against adversarial attacks bounded by a single $l_p$ norm using adversarial training (AT). However, these AT models' multiple-norm robustness (union accuracy) is still low, which is crucial since in the real-world an adversary is not necessarily bounded by a single norm. The tradeoffs among robustness against multiple $l_p$ perturbations and accuracy/robustness make obtaining good union and clean accuracy challenging. We design a logit pairing loss to improve the union accuracy by analyzing the tradeoffs from the lens of distribution shifts. We connect natural training (NT) with AT via gradient projection, to incorporate useful information from NT into AT, where we empirically and theoretically show it moderates the accuracy/robustness tradeoff. We propose a novel training framework \textbf{RAMP}, to boost the robustness against multiple $l_p$ perturbations. \textbf{RAMP} can be easily adapted for robust fine-tuning and full AT. For robust fine-tuning, \textbf{RAMP} obtains a union accuracy up to $53.3\%$ on CIFAR-10, and $29.1\%$ on ImageNet. For training from scratch, \textbf{RAMP} achieves a union accuracy of $44.6\%$ and good clean accuracy of $81.2\%$ on ResNet-18 against AutoAttack on CIFAR-10. Beyond multi-norm robustness \textbf{RAMP}-trained models achieve superior \textit{universal robustness}, effectively generalizing against a range of unseen adversaries and natural corruptions.

摘要: 现有的大部分工作都集中在利用对抗性训练(AT)来提高对单个$L_p$范数所限定的对抗性攻击的稳健性。然而，这些AT模型的多范数稳健性(联合精度)仍然很低，这一点至关重要，因为在现实世界中，对手不一定受到单一范数的约束。对多重$L_p$扰动的稳健性和精度/稳健性之间的权衡使得获得良好的并集和干净的精度具有挑战性。通过从分布移位的角度分析权衡，设计了一种Logit配对损耗来提高合并精度。我们通过梯度投影将自然训练(NT)与AT联系起来，将来自NT的有用信息结合到AT中，我们从经验和理论上证明了它缓和了准确率和稳健性之间的权衡。我们提出了一种新的训练框架\Textbf{Ramp}，以增强对多重$L_p$扰动的鲁棒性。\extbf{RAMP}可轻松调整以进行稳健的微调和完整的AT。对于稳健的微调，在CIFAR-10和ImageNet上，Textbf{Ramp}获得了高达53.3美元的联合精度和29.1美元的联合精度。对于从头开始的训练，在ResNet-18上与在CIFAR-10上的AutoAttack相比，Textbf{RAMP}实现了$44.6\$的联合精度和$81.2\$的良好清洁精度。除多范数稳健性外，经过文本bf{ramp}训练的模型实现了卓越的\textit{通用稳健性}，有效地概括了一系列看不见的对手和自然腐败。



## **28. $DA^3$: A Distribution-Aware Adversarial Attack against Language Models**

$DA ' 3 $：针对语言模型的分布感知对抗攻击 cs.CL

First two authors contribute equally; The paper has been accepted to  EMNLP2024 main conference

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2311.08598v3) [paper-pdf](http://arxiv.org/pdf/2311.08598v3)

**Authors**: Yibo Wang, Xiangjue Dong, James Caverlee, Philip S. Yu

**Abstract**: Language models can be manipulated by adversarial attacks, which introduce subtle perturbations to input data. While recent attack methods can achieve a relatively high attack success rate (ASR), we've observed that the generated adversarial examples have a different data distribution compared with the original examples. Specifically, these adversarial examples exhibit reduced confidence levels and greater divergence from the training data distribution. Consequently, they are easy to detect using straightforward detection methods, diminishing the efficacy of such attacks. To address this issue, we propose a Distribution-Aware Adversarial Attack ($DA^3$) method. $DA^3$ considers the distribution shifts of adversarial examples to improve attacks' effectiveness under detection methods. We further design a novel evaluation metric, the Non-detectable Attack Success Rate (NASR), which integrates both ASR and detectability for the attack task. We conduct experiments on four widely used datasets to validate the attack effectiveness and transferability of adversarial examples generated by $DA^3$ against both the white-box BERT-base and RoBERTa-base models and the black-box LLaMA2-7b model.

摘要: 语言模型可以通过对抗性攻击来操纵，这种攻击会给输入数据带来微妙的干扰。虽然目前的攻击方法可以达到相对较高的攻击成功率(ASR)，但我们观察到生成的敌意示例与原始示例相比具有不同的数据分布。具体地说，这些对抗性的例子表现出更低的置信度和与训练数据分布更大的背离。因此，使用直接的检测方法很容易检测到它们，从而降低了此类攻击的有效性。为了解决这个问题，我们提出了一种分布式感知的对抗性攻击方法($DA^3$)。$DA^3$考虑了对抗性实例的分布偏移，以提高检测方法下的攻击有效性。在此基础上，我们进一步设计了一种新的评价指标--不可检测攻击成功率(NASR)，它综合了攻击任务的ASR和可检测性。我们在四个广泛使用的数据集上进行了实验，以验证$DA^3$生成的对抗性实例在白盒Bert-base和Roberta-base模型以及黑盒LLaMA2-7b模型下的攻击有效性和可转移性。



## **29. Persistent Backdoor Attacks in Continual Learning**

持续学习中的持续后门攻击 cs.LG

18 pages, 15 figures, 6 tables

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13864v1) [paper-pdf](http://arxiv.org/pdf/2409.13864v1)

**Authors**: Zhen Guo, Abhinav Kumar, Reza Tourani

**Abstract**: Backdoor attacks pose a significant threat to neural networks, enabling adversaries to manipulate model outputs on specific inputs, often with devastating consequences, especially in critical applications. While backdoor attacks have been studied in various contexts, little attention has been given to their practicality and persistence in continual learning, particularly in understanding how the continual updates to model parameters, as new data distributions are learned and integrated, impact the effectiveness of these attacks over time. To address this gap, we introduce two persistent backdoor attacks-Blind Task Backdoor and Latent Task Backdoor-each leveraging minimal adversarial influence. Our blind task backdoor subtly alters the loss computation without direct control over the training process, while the latent task backdoor influences only a single task's training, with all other tasks trained benignly. We evaluate these attacks under various configurations, demonstrating their efficacy with static, dynamic, physical, and semantic triggers. Our results show that both attacks consistently achieve high success rates across different continual learning algorithms, while effectively evading state-of-the-art defenses, such as SentiNet and I-BAU.

摘要: 后门攻击对神经网络构成重大威胁，使对手能够操纵特定输入的模型输出，往往会造成毁灭性的后果，特别是在关键应用中。虽然在各种情况下对后门攻击进行了研究，但很少注意到后门攻击在持续学习中的实用性和持久性，特别是在理解随着新数据分布的学习和整合而不断更新模型参数时，如何随着时间的推移影响这些攻击的有效性。为了弥补这一差距，我们引入了两种持续的后门攻击--盲目任务后门攻击和潜在任务后门攻击--每个后门攻击都利用最小的敌意影响。我们的盲任务后门巧妙地改变了损失计算，而不直接控制训练过程，而潜在任务后门只影响单个任务的训练，所有其他任务都得到了良好的训练。我们在不同的配置下对这些攻击进行评估，通过静态、动态、物理和语义触发来演示它们的有效性。我们的结果表明，这两种攻击在不同的持续学习算法上都获得了高的成功率，同时有效地避开了最先进的防御，如Sentinet和I-BAU。



## **30. ViTGuard: Attention-aware Detection against Adversarial Examples for Vision Transformer**

ViTGuard：Vision Transformer针对对抗示例的注意力感知检测 cs.CV

To appear in the Annual Computer Security Applications Conference  (ACSAC) 2024

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13828v1) [paper-pdf](http://arxiv.org/pdf/2409.13828v1)

**Authors**: Shihua Sun, Kenechukwu Nwodo, Shridatt Sugrim, Angelos Stavrou, Haining Wang

**Abstract**: The use of transformers for vision tasks has challenged the traditional dominant role of convolutional neural networks (CNN) in computer vision (CV). For image classification tasks, Vision Transformer (ViT) effectively establishes spatial relationships between patches within images, directing attention to important areas for accurate predictions. However, similar to CNNs, ViTs are vulnerable to adversarial attacks, which mislead the image classifier into making incorrect decisions on images with carefully designed perturbations. Moreover, adversarial patch attacks, which introduce arbitrary perturbations within a small area, pose a more serious threat to ViTs. Even worse, traditional detection methods, originally designed for CNN models, are impractical or suffer significant performance degradation when applied to ViTs, and they generally overlook patch attacks.   In this paper, we propose ViTGuard as a general detection method for defending ViT models against adversarial attacks, including typical attacks where perturbations spread over the entire input and patch attacks. ViTGuard uses a Masked Autoencoder (MAE) model to recover randomly masked patches from the unmasked regions, providing a flexible image reconstruction strategy. Then, threshold-based detectors leverage distinctive ViT features, including attention maps and classification (CLS) token representations, to distinguish between normal and adversarial samples. The MAE model does not involve any adversarial samples during training, ensuring the effectiveness of our detectors against unseen attacks. ViTGuard is compared with seven existing detection methods under nine attacks across three datasets. The evaluation results show the superiority of ViTGuard over existing detectors. Finally, considering the potential detection evasion, we further demonstrate ViTGuard's robustness against adaptive attacks for evasion.

摘要: 变压器在视觉任务中的使用挑战了卷积神经网络(CNN)在计算机视觉(CV)中的传统主导地位。对于图像分类任务，视觉转换器(VIT)有效地建立图像内各块之间的空间关系，将注意力引导到重要区域以进行准确预测。然而，与CNN类似，VITS容易受到敌意攻击，这些攻击会误导图像分类器对经过精心设计的扰动的图像做出错误的决定。此外，对抗性补丁攻击在小范围内引入任意扰动，对VITS构成更严重的威胁。更糟糕的是，最初为CNN模型设计的传统检测方法在应用于VITS时不切实际或性能显著下降，而且它们通常忽略了补丁攻击。在本文中，我们提出了ViTGuard作为一种通用的检测方法来保护VIT模型免受对抗性攻击，包括典型的扰动遍及整个输入的攻击和补丁攻击。ViTGuard使用掩码自动编码器(MAE)模型从非掩码区域恢复随机掩码的补丁，提供了灵活的图像重建策略。然后，基于阈值的检测器利用独特的VIT特征，包括注意图和分类(CLS)令牌表示，来区分正常样本和敌意样本。MAE模型在训练过程中不涉及任何对抗性样本，确保了我们检测器对看不见的攻击的有效性。ViTGuard与现有的七种检测方法在三个数据集的九种攻击下进行了比较。评估结果表明，ViTGuard比现有的检测器具有更好的性能。最后，考虑到潜在的检测规避，我们进一步证明了ViTGuard对自适应攻击的鲁棒性。



## **31. Neurosymbolic Conformal Classification**

神经符号保形分类 cs.LG

10 pages, 0 figures. arXiv admin note: text overlap with  arXiv:2404.08404

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13585v1) [paper-pdf](http://arxiv.org/pdf/2409.13585v1)

**Authors**: Arthur Ledaguenel, Céline Hudelot, Mostepha Khouadjia

**Abstract**: The last decades have seen a drastic improvement of Machine Learning (ML), mainly driven by Deep Learning (DL). However, despite the resounding successes of ML in many domains, the impossibility to provide guarantees of conformity and the fragility of ML systems (faced with distribution shifts, adversarial attacks, etc.) have prevented the design of trustworthy AI systems. Several research paths have been investigated to mitigate this fragility and provide some guarantees regarding the behavior of ML systems, among which are neurosymbolic AI and conformal prediction. Neurosymbolic artificial intelligence is a growing field of research aiming to combine neural network learning capabilities with the reasoning abilities of symbolic systems. One of the objective of this hybridization can be to provide theoritical guarantees that the output of the system will comply with some prior knowledge. Conformal prediction is a set of techniques that enable to take into account the uncertainty of ML systems by transforming the unique prediction into a set of predictions, called a confidence set. Interestingly, this comes with statistical guarantees regarding the presence of the true label inside the confidence set. Both approaches are distribution-free and model-agnostic. In this paper, we see how these two approaches can complement one another. We introduce several neurosymbolic conformal prediction techniques and explore their different characteristics (size of confidence sets, computational complexity, etc.).

摘要: 在过去的几十年里，机器学习(ML)有了巨大的进步，这主要是由深度学习(DL)推动的。然而，尽管ML在许多领域取得了巨大成功，但无法保证一致性和ML系统的脆弱性(面临分布变化、对抗性攻击等)。阻碍了值得信赖的人工智能系统的设计。为了缓解这种脆弱性，并为ML系统的行为提供一些保证，人们已经研究了几条途径，其中包括神经符号人工智能和保形预测。神经符号人工智能是一个不断发展的研究领域，旨在将神经网络的学习能力与符号系统的推理能力结合起来。这种杂交的目标之一可以是提供理论上的保证，即系统的输出将符合某些先验知识。保角预测是一组技术，它通过将唯一的预测转换为一组预测，称为置信度集，从而能够考虑ML系统的不确定性。有趣的是，这伴随着关于置信度集中存在真实标签的统计保证。这两种方法都不依赖于分发，也不依赖于模型。在这篇文章中，我们看到了这两种方法是如何相互补充的。我们介绍了几种神经符号保形预测技术，并探讨了它们的不同特点(置信集的大小、计算复杂度等)。



## **32. Efficient Visualization of Neural Networks with Generative Models and Adversarial Perturbations**

具有生成模型和对抗性扰动的神经网络的有效可视化 cs.CV

4 pages, 3 figures

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13559v1) [paper-pdf](http://arxiv.org/pdf/2409.13559v1)

**Authors**: Athanasios Karagounis

**Abstract**: This paper presents a novel approach for deep visualization via a generative network, offering an improvement over existing methods. Our model simplifies the architecture by reducing the number of networks used, requiring only a generator and a discriminator, as opposed to the multiple networks traditionally involved. Additionally, our model requires less prior training knowledge and uses a non-adversarial training process, where the discriminator acts as a guide rather than a competitor to the generator. The core contribution of this work is its ability to generate detailed visualization images that align with specific class labels. Our model incorporates a unique skip-connection-inspired block design, which enhances label-directed image generation by propagating class information across multiple layers. Furthermore, we explore how these generated visualizations can be utilized as adversarial examples, effectively fooling classification networks with minimal perceptible modifications to the original images. Experimental results demonstrate that our method outperforms traditional adversarial example generation techniques in both targeted and non-targeted attacks, achieving up to a 94.5% fooling rate with minimal perturbation. This work bridges the gap between visualization methods and adversarial examples, proposing that fooling rate could serve as a quantitative measure for evaluating visualization quality. The insights from this study provide a new perspective on the interpretability of neural networks and their vulnerabilities to adversarial attacks.

摘要: 本文提出了一种新的基于产生式网络的深度可视化方法，对现有方法进行了改进。我们的模型通过减少使用的网络数量简化了体系结构，只需要一个生成器和一个鉴别器，而不是传统上涉及的多个网络。此外，我们的模型需要较少的先验训练知识，并使用非对抗性训练过程，其中鉴别器充当生成器的指南而不是竞争者。这项工作的核心贡献是它能够生成与特定类标签一致的详细可视化图像。我们的模型结合了一种独特的跳跃连接启发块设计，通过在多个层上传播类信息来增强标签导向图像的生成。此外，我们还探讨了如何利用这些生成的可视化作为对抗性的例子，有效地愚弄分类网络，对原始图像进行最小可感知的修改。实验结果表明，在目标攻击和非目标攻击中，我们的方法都优于传统的敌意样本生成技术，在最小扰动的情况下，可以达到94.5%的愚弄率。这项工作弥合了可视化方法和对抗性例子之间的差距，提出了傻瓜率可以作为评估可视化质量的量化指标。这项研究的洞察力为神经网络的可解释性及其对抗攻击的脆弱性提供了一个新的视角。



## **33. Deterministic versus stochastic dynamical classifiers: opposing random adversarial attacks with noise**

确定性与随机动态分类器：对抗带有噪音的随机对抗攻击 cs.LG

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13470v1) [paper-pdf](http://arxiv.org/pdf/2409.13470v1)

**Authors**: Lorenzo Chicchi, Duccio Fanelli, Diego Febbe, Lorenzo Buffoni, Francesca Di Patti, Lorenzo Giambagli, Raffele Marino

**Abstract**: The Continuous-Variable Firing Rate (CVFR) model, widely used in neuroscience to describe the intertangled dynamics of excitatory biological neurons, is here trained and tested as a veritable dynamically assisted classifier. To this end the model is supplied with a set of planted attractors which are self-consistently embedded in the inter-nodes coupling matrix, via its spectral decomposition. Learning to classify amounts to sculp the basin of attraction of the imposed equilibria, directing different items towards the corresponding destination target, which reflects the class of respective pertinence. A stochastic variant of the CVFR model is also studied and found to be robust to aversarial random attacks, which corrupt the items to be classified. This remarkable finding is one of the very many surprising effects which arise when noise and dynamical attributes are made to mutually resonate.

摘要: 连续可变放电率（CVFR）模型广泛用于神经科学，用于描述兴奋性生物神经元的相互交织的动力学，在这里作为名副其实的动态辅助分类器进行训练和测试。为此，该模型配备了一组种植吸引子，这些吸引子通过其谱分解自一致地嵌入到节点间耦合矩阵中。学习分类相当于雕刻强加均衡的吸引力盆地，将不同的物品引导到相应的目的地目标，这反映了各自相关性的类别。还研究了CVFR模型的一种随机变体，发现它对敌对随机攻击具有鲁棒性，这些攻击会破坏要分类的项目。这一非凡的发现是当噪音和动态属性相互共振时产生的众多令人惊讶的影响之一。



## **34. Celtibero: Robust Layered Aggregation for Federated Learning**

Celtibero：用于联邦学习的稳健分层聚合 cs.CR

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2408.14240v2) [paper-pdf](http://arxiv.org/pdf/2408.14240v2)

**Authors**: Borja Molina-Coronado

**Abstract**: Federated Learning (FL) is an innovative approach to distributed machine learning. While FL offers significant privacy advantages, it also faces security challenges, particularly from poisoning attacks where adversaries deliberately manipulate local model updates to degrade model performance or introduce hidden backdoors. Existing defenses against these attacks have been shown to be effective when the data on the nodes is identically and independently distributed (i.i.d.), but they often fail under less restrictive, non-i.i.d data conditions. To overcome these limitations, we introduce Celtibero, a novel defense mechanism that integrates layered aggregation to enhance robustness against adversarial manipulation. Through extensive experiments on the MNIST and IMDB datasets, we demonstrate that Celtibero consistently achieves high main task accuracy (MTA) while maintaining minimal attack success rates (ASR) across a range of untargeted and targeted poisoning attacks. Our results highlight the superiority of Celtibero over existing defenses such as FL-Defender, LFighter, and FLAME, establishing it as a highly effective solution for securing federated learning systems against sophisticated poisoning attacks.

摘要: 联合学习(FL)是分布式机器学习的一种创新方法。虽然FL提供了显著的隐私优势，但它也面临着安全挑战，特别是来自毒化攻击的挑战，即攻击者故意操纵本地模型更新以降低模型性能或引入隐藏后门。当节点上的数据相同且独立分布(I.I.D.)时，现有的针对这些攻击的防御已被证明是有效的，但在限制较少的非I.I.D.数据条件下，它们通常会失败。为了克服这些局限性，我们引入了Celtibero，这是一种新的防御机制，它集成了分层聚合来增强对对手操纵的健壮性。通过在MNIST和IMDB数据集上的广泛实验，我们证明了Celtibero在一系列非目标和目标中毒攻击中始终实现了高的主任务准确率(MTA)，同时保持了最低的攻击成功率(ASR)。我们的结果突出了Celtibero相对于FL-Defender、LFighter和FAME等现有防御系统的优势，使其成为保护联邦学习系统免受复杂中毒攻击的高效解决方案。



## **35. Enhancing Transferability of Adversarial Attacks with GE-AdvGAN+: A Comprehensive Framework for Gradient Editing**

利用GE-AdvGAN+增强对抗性攻击的可转移性：梯度编辑的综合框架 cs.AI

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2408.12673v3) [paper-pdf](http://arxiv.org/pdf/2408.12673v3)

**Authors**: Zhibo Jin, Jiayu Zhang, Zhiyu Zhu, Chenyu Zhang, Jiahao Huang, Jianlong Zhou, Fang Chen

**Abstract**: Transferable adversarial attacks pose significant threats to deep neural networks, particularly in black-box scenarios where internal model information is inaccessible. Studying adversarial attack methods helps advance the performance of defense mechanisms and explore model vulnerabilities. These methods can uncover and exploit weaknesses in models, promoting the development of more robust architectures. However, current methods for transferable attacks often come with substantial computational costs, limiting their deployment and application, especially in edge computing scenarios. Adversarial generative models, such as Generative Adversarial Networks (GANs), are characterized by their ability to generate samples without the need for retraining after an initial training phase. GE-AdvGAN, a recent method for transferable adversarial attacks, is based on this principle. In this paper, we propose a novel general framework for gradient editing-based transferable attacks, named GE-AdvGAN+, which integrates nearly all mainstream attack methods to enhance transferability while significantly reducing computational resource consumption. Our experiments demonstrate the compatibility and effectiveness of our framework. Compared to the baseline AdvGAN, our best-performing method, GE-AdvGAN++, achieves an average ASR improvement of 47.8. Additionally, it surpasses the latest competing algorithm, GE-AdvGAN, with an average ASR increase of 5.9. The framework also exhibits enhanced computational efficiency, achieving 2217.7 FPS, outperforming traditional methods such as BIM and MI-FGSM. The implementation code for our GE-AdvGAN+ framework is available at https://github.com/GEAdvGANP

摘要: 可转移的敌意攻击对深度神经网络构成重大威胁，特别是在内部模型信息不可访问的黑盒场景中。研究对抗性攻击方法有助于提高防御机制的性能，探索模型漏洞。这些方法可以发现和利用模型中的弱点，从而促进更健壮的体系结构的开发。然而，当前的可转移攻击方法往往伴随着巨大的计算成本，限制了它们的部署和应用，特别是在边缘计算场景中。对抗性生成模型，如生成性对抗性网络(GANS)，其特点是能够在初始训练阶段后生成样本，而不需要重新训练。GE-AdvGAN是一种新的可转移的对抗性攻击方法，它基于这一原理。本文提出了一种新颖的基于梯度编辑的可转移攻击通用框架GE-AdvGAN+，该框架集成了几乎所有的主流攻击方法，在提高可转移性的同时显著降低了计算资源消耗。我们的实验证明了该框架的兼容性和有效性。与基准的AdvGAN相比，我们性能最好的方法GE-AdvGAN++实现了平均47.8的ASR改进。此外，它还超过了最新的竞争算法GE-AdvGAN，平均ASR提高了5.9。该框架还表现出更高的计算效率，达到2217.7 FPS，优于传统的BIM和MI-FGSM方法。我们GE-AdvGan+框架的实现代码可在https://github.com/GEAdvGANP上获得



## **36. Relationship between Uncertainty in DNNs and Adversarial Attacks**

DNN的不确定性与对抗攻击之间的关系 cs.LG

review

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13232v1) [paper-pdf](http://arxiv.org/pdf/2409.13232v1)

**Authors**: Abigail Adeniran, Adewale Adeyemo

**Abstract**: Deep Neural Networks (DNNs) have achieved state of the art results and even outperformed human accuracy in many challenging tasks, leading to DNNs adoption in a variety of fields including natural language processing, pattern recognition, prediction, and control optimization. However, DNNs are accompanied by uncertainty about their results, causing them to predict an outcome that is either incorrect or outside of a certain level of confidence. These uncertainties stem from model or data constraints, which could be exacerbated by adversarial attacks. Adversarial attacks aim to provide perturbed input to DNNs, causing the DNN to make incorrect predictions or increase model uncertainty. In this review, we explore the relationship between DNN uncertainty and adversarial attacks, emphasizing how adversarial attacks might raise DNN uncertainty.

摘要: 深度神经网络（DNN）已实现最先进的结果，甚至在许多具有挑战性的任务中超过了人类的准确性，导致DNN在自然语言处理、模式识别、预测和控制优化等各个领域得到采用。然而，DNN伴随着结果的不确定性，导致它们预测的结果要么不正确，要么超出一定置信水平。这些不确定性源于模型或数据限制，对抗性攻击可能会加剧这种限制。对抗性攻击旨在向DNN提供受干扰的输入，导致DNN做出错误的预测或增加模型的不确定性。在这篇评论中，我们探讨了DNN不确定性和对抗性攻击之间的关系，强调了对抗性攻击如何提高DNN不确定性。



## **37. Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models**

面对威胁的操纵：评估端到端视觉语言动作模型中的身体脆弱性 cs.CV

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13174v1) [paper-pdf](http://arxiv.org/pdf/2409.13174v1)

**Authors**: Hao Cheng, Erjia Xiao, Chengyuan Yu, Zhao Yao, Jiahang Cao, Qiang Zhang, Jiaxu Wang, Mengshu Sun, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Recently, driven by advancements in Multimodal Large Language Models (MLLMs), Vision Language Action Models (VLAMs) are being proposed to achieve better performance in open-vocabulary scenarios for robotic manipulation tasks. Since manipulation tasks involve direct interaction with the physical world, ensuring robustness and safety during the execution of this task is always a very critical issue. In this paper, by synthesizing current safety research on MLLMs and the specific application scenarios of the manipulation task in the physical world, we comprehensively evaluate VLAMs in the face of potential physical threats. Specifically, we propose the Physical Vulnerability Evaluating Pipeline (PVEP) that can incorporate as many visual modal physical threats as possible for evaluating the physical robustness of VLAMs. The physical threats in PVEP specifically include Out-of-Distribution, Typography-based Visual Prompt, and Adversarial Patch Attacks. By comparing the performance fluctuations of VLAMs before and after being attacked, we provide generalizable \textbf{\textit{Analyses}} of how VLAMs respond to different physical security threats.

摘要: 最近，在多模式大语言模型(MLLM)的推动下，视觉语言动作模型(VLAM)被提出以在机器人操作任务的开放词汇场景中实现更好的性能。由于操作任务涉及与物理世界的直接交互，因此确保该任务执行过程中的健壮性和安全性始终是一个非常关键的问题。本文通过综合当前MLLMS的安全研究现状和物理世界中操纵任务的具体应用场景，对VLAMS在面临潜在物理威胁的情况下进行综合评估。具体地说，我们提出了物理脆弱性评估管道(PVEP)，它可以结合尽可能多的视觉通道物理威胁来评估VLAMS的物理健壮性。PVEP中的物理威胁具体包括分发外、基于排版的视觉提示和敌意补丁攻击。通过比较VLAM在受到攻击前后的性能波动，我们提供了VLAM如何应对不同的物理安全威胁的概括性的文本bf{文本{分析}}。



## **38. Hidden Activations Are Not Enough: A General Approach to Neural Network Predictions**

隐藏的激活还不够：神经网络预测的通用方法 cs.LG

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13163v1) [paper-pdf](http://arxiv.org/pdf/2409.13163v1)

**Authors**: Samuel Leblanc, Aiky Rasolomanana, Marco Armenta

**Abstract**: We introduce a novel mathematical framework for analyzing neural networks using tools from quiver representation theory. This framework enables us to quantify the similarity between a new data sample and the training data, as perceived by the neural network. By leveraging the induced quiver representation of a data sample, we capture more information than traditional hidden layer outputs. This quiver representation abstracts away the complexity of the computations of the forward pass into a single matrix, allowing us to employ simple geometric and statistical arguments in a matrix space to study neural network predictions. Our mathematical results are architecture-agnostic and task-agnostic, making them broadly applicable. As proof of concept experiments, we apply our results for the MNIST and FashionMNIST datasets on the problem of detecting adversarial examples on different MLP architectures and several adversarial attack methods. Our experiments can be reproduced with our \href{https://github.com/MarcoArmenta/Hidden-Activations-are-not-Enough}{publicly available repository}.

摘要: 我们介绍了一个新的数学框架来分析神经网络使用箭图表示理论的工具。这个框架使我们能够量化新数据样本和训练数据之间的相似性，就像神经网络所感知的那样。通过利用数据样本的诱导抖动表示，我们捕获了比传统隐含层输出更多的信息。这种箭图表示将前向传递计算的复杂性抽象到单个矩阵中，允许我们在矩阵空间中使用简单的几何和统计论点来研究神经网络预测。我们的数学结果是体系结构不可知和任务不可知的，使它们具有广泛的适用性。作为概念验证实验，我们在MNIST和FashionMNIST数据集上应用我们的结果来检测不同MLP体系结构和几种对抗性攻击方法上的对抗性实例。我们的实验可以用我们的\href{https://github.com/MarcoArmenta/Hidden-Activations-are-not-Enough}{publicly Available存储库重现。



## **39. FedAT: Federated Adversarial Training for Distributed Insider Threat Detection**

FedAT：分布式内部威胁检测的联合对抗训练 cs.CR

10 pages, 7 figures

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.13083v1) [paper-pdf](http://arxiv.org/pdf/2409.13083v1)

**Authors**: R G Gayathri, Atul Sajjanhar, Md Palash Uddin, Yong Xiang

**Abstract**: Insider threats usually occur from within the workplace, where the attacker is an entity closely associated with the organization. The sequence of actions the entities take on the resources to which they have access rights allows us to identify the insiders. Insider Threat Detection (ITD) using Machine Learning (ML)-based approaches gained attention in the last few years. However, most techniques employed centralized ML methods to perform such an ITD. Organizations operating from multiple locations cannot contribute to the centralized models as the data is generated from various locations. In particular, the user behavior data, which is the primary source of ITD, cannot be shared among the locations due to privacy concerns. Additionally, the data distributed across various locations result in extreme class imbalance due to the rarity of attacks. Federated Learning (FL), a distributed data modeling paradigm, gained much interest recently. However, FL-enabled ITD is not yet explored, and it still needs research to study the significant issues of its implementation in practical settings. As such, our work investigates an FL-enabled multiclass ITD paradigm that considers non-Independent and Identically Distributed (non-IID) data distribution to detect insider threats from different locations (clients) of an organization. Specifically, we propose a Federated Adversarial Training (FedAT) approach using a generative model to alleviate the extreme data skewness arising from the non-IID data distribution among the clients. Besides, we propose to utilize a Self-normalized Neural Network-based Multi-Layer Perceptron (SNN-MLP) model to improve ITD. We perform comprehensive experiments and compare the results with the benchmarks to manifest the enhanced performance of the proposed FedATdriven ITD scheme.

摘要: 内部威胁通常发生在工作场所内部，攻击者是与组织密切相关的实体。实体对其拥有访问权限的资源采取的操作顺序使我们能够识别内部人员。使用基于机器学习(ML)方法的内部威胁检测(ITD)在过去几年中得到了关注。然而，大多数技术使用集中式ML方法来执行这样的ITD。由于数据是从多个位置生成的，因此在多个位置运营的组织不能对集中化模型做出贡献。特别是，由于隐私问题，作为ITD的主要来源的用户行为数据不能在不同地点之间共享。此外，由于攻击的罕见，分布在不同位置的数据导致了极端的类不平衡。联邦学习(FL)是一种分布式数据建模范式，近年来受到了极大的关注。然而，启用外语的信息技术开发还没有被探索，它仍然需要研究，以研究其在实际环境中实施的重大问题。因此，我们的工作研究了启用FL的多类ITD范例，该范例考虑非独立且相同分布(Non-IID)的数据分布，以检测来自组织不同位置(客户端)的内部威胁。具体地说，我们提出了一种使用产生式模型的联合对手训练(FedAT)方法，以缓解非IID数据在客户端之间分布时产生的极端数据偏斜。此外，我们还提出了一种基于自归一化神经网络的多层感知器(SNN-MLP)模型来改进ITD。我们进行了全面的实验，并将结果与基准测试结果进行了比较，以证明所提出的FedATDriven ITD方案具有更好的性能。



## **40. Defending against Reverse Preference Attacks is Difficult**

防御反向偏好攻击很困难 cs.LG

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12914v1) [paper-pdf](http://arxiv.org/pdf/2409.12914v1)

**Authors**: Domenic Rosati, Giles Edkins, Harsh Raj, David Atanasov, Subhabrata Majumdar, Janarthanan Rajendran, Frank Rudzicz, Hassan Sajjad

**Abstract**: While there has been progress towards aligning Large Language Models (LLMs) with human values and ensuring safe behaviour at inference time, safety-aligned LLMs are known to be vulnerable to training-time attacks such as supervised fine-tuning (SFT) on harmful datasets. In this paper, we ask if LLMs are vulnerable to adversarial reinforcement learning. Motivated by this goal, we propose Reverse Preference Attacks (RPA), a class of attacks to make LLMs learn harmful behavior using adversarial reward during reinforcement learning from human feedback (RLHF). RPAs expose a critical safety gap of safety-aligned LLMs in RL settings: they easily explore the harmful text generation policies to optimize adversarial reward. To protect against RPAs, we explore a host of mitigation strategies. Leveraging Constrained Markov-Decision Processes, we adapt a number of mechanisms to defend against harmful fine-tuning attacks into the RL setting. Our experiments show that ``online" defenses that are based on the idea of minimizing the negative log likelihood of refusals -- with the defender having control of the loss function -- can effectively protect LLMs against RPAs. However, trying to defend model weights using ``offline" defenses that operate under the assumption that the defender has no control over the loss function are less effective in the face of RPAs. These findings show that attacks done using RL can be used to successfully undo safety alignment in open-weight LLMs and use them for malicious purposes.

摘要: 虽然在使大型语言模型(LLM)与人类价值观保持一致并确保推理时的安全行为方面取得了进展，但众所周知，与安全保持一致的LLM容易受到训练时的攻击，例如对有害数据集的监督微调(SFT)。在本文中，我们询问LLMS是否容易受到对抗性强化学习的影响。基于这一目标，我们提出了反向偏好攻击(RPA)，这是一类在人类反馈强化学习(RLHF)过程中利用对抗性奖励使LLM学习有害行为的攻击。RPA暴露了RL环境中安全对齐的LLM的一个关键安全漏洞：它们很容易探索有害文本生成策略，以优化对抗性奖励。为了防范RPA，我们探索了一系列缓解策略。利用受限的马尔可夫决策过程，我们采用了一些机制来防御RL设置中的有害微调攻击。我们的实验表明，基于最小化拒绝的负对数可能性的思想的“在线”防御--防御者控制着损失函数--可以有效地保护LLM免受RPA的攻击。然而，试图使用在防御者无法控制损失函数的假设下运行的“离线”防御来捍卫模型权重，在面对RPA时效果较差。这些发现表明，使用RL进行的攻击可以成功地取消开放重量LLM中的安全对齐，并将其用于恶意目的。



## **41. VCAT: Vulnerability-aware and Curiosity-driven Adversarial Training for Enhancing Autonomous Vehicle Robustness**

VCAT：脆弱性感知和好奇心驱动的对抗培训，以增强自动驾驶车辆的稳健性 cs.LG

7 pages, 5 figures, conference

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12997v1) [paper-pdf](http://arxiv.org/pdf/2409.12997v1)

**Authors**: Xuan Cai, Zhiyong Cui, Xuesong Bai, Ruimin Ke, Zhenshu Ma, Haiyang Yu, Yilong Ren

**Abstract**: Autonomous vehicles (AVs) face significant threats to their safe operation in complex traffic environments. Adversarial training has emerged as an effective method of enabling AVs to preemptively fortify their robustness against malicious attacks. Train an attacker using an adversarial policy, allowing the AV to learn robust driving through interaction with this attacker. However, adversarial policies in existing methodologies often get stuck in a loop of overexploiting established vulnerabilities, resulting in poor improvement for AVs. To overcome the limitations, we introduce a pioneering framework termed Vulnerability-aware and Curiosity-driven Adversarial Training (VCAT). Specifically, during the traffic vehicle attacker training phase, a surrogate network is employed to fit the value function of the AV victim, providing dense information about the victim's inherent vulnerabilities. Subsequently, random network distillation is used to characterize the novelty of the environment, constructing an intrinsic reward to guide the attacker in exploring unexplored territories. In the victim defense training phase, the AV is trained in critical scenarios in which the pretrained attacker is positioned around the victim to generate attack behaviors. Experimental results revealed that the training methodology provided by VCAT significantly improved the robust control capabilities of learning-based AVs, outperforming both conventional training modalities and alternative reinforcement learning counterparts, with a marked reduction in crash rates. The code is available at https://github.com/caixxuan/VCAT.

摘要: 自动驾驶汽车(AVs)在复杂的交通环境中的安全运行面临着巨大的威胁。对抗性训练已经成为一种有效的方法，使AVs能够先发制人地增强其对恶意攻击的健壮性。使用对抗性策略训练攻击者，允许反病毒通过与该攻击者的交互学习健壮的驾驶。然而，现有方法中的对抗性策略经常陷入过度利用已建立的漏洞的循环中，导致对AV的改进很差。为了克服这些限制，我们引入了一个开创性的框架，称为漏洞感知和好奇心驱动的对手训练(VCAT)。具体地说，在交通车辆攻击者训练阶段，使用代理网络来拟合反病毒受害者的价值函数，提供关于受害者固有漏洞的密集信息。随后，随机网络蒸馏被用来表征环境的新颖性，构建一种内在的奖励来引导攻击者探索未知领域。在受害者防御训练阶段，反病毒在关键场景中进行训练，在该场景中，预先训练的攻击者被定位在受害者周围以产生攻击行为。实验结果表明，VCAT提供的训练方法显著提高了基于学习的自动驾驶系统的鲁棒控制能力，表现优于传统训练模式和替代强化学习模式，并显著降低了撞车率。代码可在https://github.com/caixxuan/VCAT.上获得



## **42. Boosting Certified Robustness for Time Series Classification with Efficient Self-Ensemble**

通过高效的自我整合增强时间序列分类的认证鲁棒性 cs.LG

6 figures, 4 tables, 10 pages

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.02802v3) [paper-pdf](http://arxiv.org/pdf/2409.02802v3)

**Authors**: Chang Dong, Zhengyang Li, Liangwei Zheng, Weitong Chen, Wei Emma Zhang

**Abstract**: Recently, the issue of adversarial robustness in the time series domain has garnered significant attention. However, the available defense mechanisms remain limited, with adversarial training being the predominant approach, though it does not provide theoretical guarantees. Randomized Smoothing has emerged as a standout method due to its ability to certify a provable lower bound on robustness radius under $\ell_p$-ball attacks. Recognizing its success, research in the time series domain has started focusing on these aspects. However, existing research predominantly focuses on time series forecasting, or under the non-$\ell_p$ robustness in statistic feature augmentation for time series classification~(TSC). Our review found that Randomized Smoothing performs modestly in TSC, struggling to provide effective assurances on datasets with poor robustness. Therefore, we propose a self-ensemble method to enhance the lower bound of the probability confidence of predicted labels by reducing the variance of classification margins, thereby certifying a larger radius. This approach also addresses the computational overhead issue of Deep Ensemble~(DE) while remaining competitive and, in some cases, outperforming it in terms of robustness. Both theoretical analysis and experimental results validate the effectiveness of our method, demonstrating superior performance in robustness testing compared to baseline approaches.

摘要: 最近，时间序列域中的对抗性稳健性问题引起了人们的广泛关注。然而，现有的防御机制仍然有限，对抗性训练是主要的方法，尽管它不提供理论上的保证。由于随机化平滑方法能够证明在$ell_p$-ball攻击下的健壮性半径的一个可证明的下界，所以它已经成为一种优秀的方法。认识到它的成功，时间序列领域的研究已经开始集中在这些方面。然而，现有的研究主要集中在时间序列预测，或在统计特征增强对时间序列分类具有非埃尔p稳健性的情况下。我们的综述发现，随机平滑在TSC中表现平平，难以对稳健性较差的数据集提供有效的保证。因此，我们提出了一种自集成方法，通过减小分类裕度的方差来提高预测标签的概率置信度下界，从而证明更大的半径。这种方法还解决了深层集成~(DE)的计算开销问题，同时保持了竞争力，在某些情况下，在健壮性方面优于它。理论分析和实验结果都验证了该方法的有效性，在稳健性测试中表现出了优于基线方法的性能。



## **43. Deep generative models as an adversarial attack strategy for tabular machine learning**

深度生成模型作为表格机器学习的对抗攻击策略 cs.LG

Accepted at ICMLC 2024 (International Conference on Machine Learning  and Cybernetics)

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12642v1) [paper-pdf](http://arxiv.org/pdf/2409.12642v1)

**Authors**: Salijona Dyrmishi, Mihaela Cătălina Stoian, Eleonora Giunchiglia, Maxime Cordy

**Abstract**: Deep Generative Models (DGMs) have found application in computer vision for generating adversarial examples to test the robustness of machine learning (ML) systems. Extending these adversarial techniques to tabular ML presents unique challenges due to the distinct nature of tabular data and the necessity to preserve domain constraints in adversarial examples. In this paper, we adapt four popular tabular DGMs into adversarial DGMs (AdvDGMs) and evaluate their effectiveness in generating realistic adversarial examples that conform to domain constraints.

摘要: 深度生成模型（DGM）已在计算机视觉中应用，用于生成对抗性示例来测试机器学习（ML）系统的稳健性。由于表格数据的独特性质以及在对抗性示例中保留域约束的必要性，将这些对抗性技术扩展到表格ML带来了独特的挑战。在本文中，我们将四种流行的表格式DGM调整为对抗性DGM（AdvDGM），并评估它们在生成符合领域约束的现实对抗性示例方面的有效性。



## **44. Exploring Privacy and Fairness Risks in Sharing Diffusion Models: An Adversarial Perspective**

探索共享扩散模型中的隐私和公平风险：对抗的视角 cs.LG

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2402.18607v3) [paper-pdf](http://arxiv.org/pdf/2402.18607v3)

**Authors**: Xinjian Luo, Yangfan Jiang, Fei Wei, Yuncheng Wu, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Diffusion models have recently gained significant attention in both academia and industry due to their impressive generative performance in terms of both sampling quality and distribution coverage. Accordingly, proposals are made for sharing pre-trained diffusion models across different organizations, as a way of improving data utilization while enhancing privacy protection by avoiding sharing private data directly. However, the potential risks associated with such an approach have not been comprehensively examined.   In this paper, we take an adversarial perspective to investigate the potential privacy and fairness risks associated with the sharing of diffusion models. Specifically, we investigate the circumstances in which one party (the sharer) trains a diffusion model using private data and provides another party (the receiver) black-box access to the pre-trained model for downstream tasks. We demonstrate that the sharer can execute fairness poisoning attacks to undermine the receiver's downstream models by manipulating the training data distribution of the diffusion model. Meanwhile, the receiver can perform property inference attacks to reveal the distribution of sensitive features in the sharer's dataset. Our experiments conducted on real-world datasets demonstrate remarkable attack performance on different types of diffusion models, which highlights the critical importance of robust data auditing and privacy protection protocols in pertinent applications.

摘要: 扩散模型由于其在抽样质量和分布复盖率方面令人印象深刻的生成性能，最近在学术界和工业界都得到了极大的关注。因此，提出了在不同组织之间共享预先培训的传播模式的建议，以此作为提高数据利用率的一种方式，同时通过避免直接共享私人数据来加强隐私保护。然而，与这种方法相关的潜在风险尚未得到全面审查。在这篇文章中，我们采取对抗性的视角来调查与共享扩散模型相关的潜在的隐私和公平风险。具体地说，我们调查了一方(共享者)使用私有数据训练扩散模型，并为另一方(接收者)提供对下游任务的预训练模型的黑箱访问的情况。我们证明了共享者可以通过操纵扩散模型的训练数据分布来执行公平毒化攻击来破坏接收者的下游模型。同时，接收者可以执行属性推断攻击，以揭示共享者数据集中敏感特征的分布。我们在真实数据集上进行的实验表明，在不同类型的扩散模型上具有显著的攻击性能，这突显了健壮的数据审计和隐私保护协议在相关应用中的关键重要性。



## **45. Adversarial Attack for Explanation Robustness of Rationalization Models**

对合理化模型解释稳健性的对抗攻击 cs.CL

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2408.10795v3) [paper-pdf](http://arxiv.org/pdf/2408.10795v3)

**Authors**: Yuankai Zhang, Lingxiao Kong, Haozhao Wang, Ruixuan Li, Jun Wang, Yuhua Li, Wei Liu

**Abstract**: Rationalization models, which select a subset of input text as rationale-crucial for humans to understand and trust predictions-have recently emerged as a prominent research area in eXplainable Artificial Intelligence. However, most of previous studies mainly focus on improving the quality of the rationale, ignoring its robustness to malicious attack. Specifically, whether the rationalization models can still generate high-quality rationale under the adversarial attack remains unknown. To explore this, this paper proposes UAT2E, which aims to undermine the explainability of rationalization models without altering their predictions, thereby eliciting distrust in these models from human users. UAT2E employs the gradient-based search on triggers and then inserts them into the original input to conduct both the non-target and target attack. Experimental results on five datasets reveal the vulnerability of rationalization models in terms of explanation, where they tend to select more meaningless tokens under attacks. Based on this, we make a series of recommendations for improving rationalization models in terms of explanation.

摘要: 合理化模型选择输入文本的一个子集作为理论基础--这对人类理解和信任预测至关重要--最近已成为可解释人工智能的一个重要研究领域。然而，以往的研究大多侧重于提高理论基础的质量，而忽略了其对恶意攻击的健壮性。具体地说，在对抗性攻击下，合理化模型是否仍能产生高质量的推理仍是未知的。为了探索这一点，本文提出了UAT2E，其目的是在不改变其预测的情况下削弱合理化模型的可解释性，从而引起人类用户对这些模型的不信任。UAT2E在触发器上采用基于梯度的搜索，然后将它们插入到原始输入中，以进行非目标攻击和目标攻击。在五个数据集上的实验结果揭示了合理化模型在解释方面的脆弱性，在攻击下，它们倾向于选择更多无意义的标记。在此基础上，本文从解释的角度提出了一系列改进合理化模型的建议。



## **46. Designing an attack-defense game: how to increase robustness of financial transaction models via a competition**

设计攻守游戏：如何通过竞争提高金融交易模型的稳健性 cs.LG

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2308.11406v3) [paper-pdf](http://arxiv.org/pdf/2308.11406v3)

**Authors**: Alexey Zaytsev, Maria Kovaleva, Alex Natekin, Evgeni Vorsin, Valerii Smirnov, Georgii Smirnov, Oleg Sidorshin, Alexander Senin, Alexander Dudin, Dmitry Berestnev

**Abstract**: Banks routinely use neural networks to make decisions. While these models offer higher accuracy, they are susceptible to adversarial attacks, a risk often overlooked in the context of event sequences, particularly sequences of financial transactions, as most works consider computer vision and NLP modalities.   We propose a thorough approach to studying these risks: a novel type of competition that allows a realistic and detailed investigation of problems in financial transaction data. The participants directly oppose each other, proposing attacks and defenses -- so they are examined in close-to-real-life conditions.   The paper outlines our unique competition structure with direct opposition of participants, presents results for several different top submissions, and analyzes the competition results. We also introduce a new open dataset featuring financial transactions with credit default labels, enhancing the scope for practical research and development.

摘要: 银行经常使用神经网络来做出决策。虽然这些模型提供了更高的准确性，但它们很容易受到对抗攻击，这一风险在事件序列（尤其是金融交易序列）的背景下经常被忽视，因为大多数作品都考虑计算机视觉和NLP模式。   我们提出了一种彻底的方法来研究这些风险：一种新型竞争，可以对金融交易数据中的问题进行现实而详细的调查。参与者直接相互反对，提出攻击和防御--因此他们在接近现实生活的条件下受到审查。   该论文概述了我们独特的竞争结构，参与者直接反对，列出了几种不同的顶级提交的结果，并分析了竞争结果。我们还引入了一个新的开放数据集，以带有信用违约标签的金融交易为特色，扩大了实践研究和开发的范围。



## **47. Magmaw: Modality-Agnostic Adversarial Attacks on Machine Learning-Based Wireless Communication Systems**

Magmaw：对基于机器学习的无线通信系统的模式不可知的对抗攻击 cs.CR

Accepted at NDSS 2025

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2311.00207v2) [paper-pdf](http://arxiv.org/pdf/2311.00207v2)

**Authors**: Jung-Woo Chang, Ke Sun, Nasimeh Heydaribeni, Seira Hidano, Xinyu Zhang, Farinaz Koushanfar

**Abstract**: Machine Learning (ML) has been instrumental in enabling joint transceiver optimization by merging all physical layer blocks of the end-to-end wireless communication systems. Although there have been a number of adversarial attacks on ML-based wireless systems, the existing methods do not provide a comprehensive view including multi-modality of the source data, common physical layer protocols, and wireless domain constraints. This paper proposes Magmaw, a novel wireless attack methodology capable of generating universal adversarial perturbations for any multimodal signal transmitted over a wireless channel. We further introduce new objectives for adversarial attacks on downstream applications. We adopt the widely-used defenses to verify the resilience of Magmaw. For proof-of-concept evaluation, we build a real-time wireless attack platform using a software-defined radio system. Experimental results demonstrate that Magmaw causes significant performance degradation even in the presence of strong defense mechanisms. Furthermore, we validate the performance of Magmaw in two case studies: encrypted communication channel and channel modality-based ML model.

摘要: 机器学习(ML)通过合并端到端无线通信系统的所有物理层块，在实现联合收发器优化方面发挥了重要作用。尽管已经有一些针对基于ML的无线系统的对抗性攻击，但现有的方法不能提供包括源数据的多模态、公共物理层协议和无线域限制在内的全面视角。本文提出了一种新的无线攻击方法Magmaw，它能够对无线信道上传输的任何多模信号产生通用的对抗性扰动。我们进一步引入了针对下游应用程序的对抗性攻击的新目标。我们采用了广泛使用的防御措施来验证Magmaw的弹性。对于概念验证评估，我们使用软件定义的无线电系统构建了一个实时无线攻击平台。实验结果表明，即使在强防御机制存在的情况下，Magmaw也会导致性能显著下降。此外，我们在加密通信通道和基于通道通道的ML模型两个案例中验证了MAGMAW的性能。



## **48. TEAM: Temporal Adversarial Examples Attack Model against Network Intrusion Detection System Applied to RNN**

TEAM：应用于RNN的针对网络入侵检测系统的时间对抗示例攻击模型 cs.CR

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12472v1) [paper-pdf](http://arxiv.org/pdf/2409.12472v1)

**Authors**: Ziyi Liu, Dengpan Ye, Long Tang, Yunming Zhang, Jiacheng Deng

**Abstract**: With the development of artificial intelligence, neural networks play a key role in network intrusion detection systems (NIDS). Despite the tremendous advantages, neural networks are susceptible to adversarial attacks. To improve the reliability of NIDS, many research has been conducted and plenty of solutions have been proposed. However, the existing solutions rarely consider the adversarial attacks against recurrent neural networks (RNN) with time steps, which would greatly affect the application of NIDS in real world. Therefore, we first propose a novel RNN adversarial attack model based on feature reconstruction called \textbf{T}emporal adversarial \textbf{E}xamples \textbf{A}ttack \textbf{M}odel \textbf{(TEAM)}, which applied to time series data and reveals the potential connection between adversarial and time steps in RNN. That is, the past adversarial examples within the same time steps can trigger further attacks on current or future original examples. Moreover, TEAM leverages Time Dilation (TD) to effectively mitigates the effect of temporal among adversarial examples within the same time steps. Experimental results show that in most attack categories, TEAM improves the misjudgment rate of NIDS on both black and white boxes, making the misjudgment rate reach more than 96.68%. Meanwhile, the maximum increase in the misjudgment rate of the NIDS for subsequent original samples exceeds 95.57%.

摘要: 随着人工智能的发展，神经网络在网络入侵检测系统中发挥着关键作用。尽管神经网络具有巨大的优势，但它很容易受到对手的攻击。为了提高网络入侵检测系统的可靠性，人们进行了大量的研究，并提出了大量的解决方案。然而，现有的解决方案很少考虑对带时间步长的递归神经网络的对抗性攻击，这将极大地影响网络入侵检测系统在现实世界中的应用。为此，我们首先提出了一种新的基于特征重构的RNN对抗性攻击模型也就是说，在相同的时间步骤内的过去的对抗性例子可以触发对当前或未来的原始例子的进一步攻击。此外，团队利用时间膨胀(TD)来有效地缓解相同时间步长内的对抗性例子之间的时间效应。实验结果表明，在大多数攻击类别中，Team都提高了黑盒和白盒的误判率，使误判率达到96.68%以上。同时，网络入侵检测系统对后续原始样本的误判率最大增幅超过95.57%。



## **49. Object-fabrication Targeted Attack for Object Detection**

物体制造用于物体检测的定向攻击 cs.CV

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2212.06431v3) [paper-pdf](http://arxiv.org/pdf/2212.06431v3)

**Authors**: Xuchong Zhang, Changfeng Sun, Haoliang Han, Hongbin Sun

**Abstract**: Recent studies have demonstrated that object detection networks are usually vulnerable to adversarial examples. Generally, adversarial attacks for object detection can be categorized into targeted and untargeted attacks. Compared with untargeted attacks, targeted attacks present greater challenges and all existing targeted attack methods launch the attack by misleading detectors to mislabel the detected object as a specific wrong label. However, since these methods must depend on the presence of the detected objects within the victim image, they suffer from limitations in attack scenarios and attack success rates. In this paper, we propose a targeted feature space attack method that can mislead detectors to `fabricate' extra designated objects regardless of whether the victim image contains objects or not. Specifically, we introduce a guided image to extract coarse-grained features of the target objects and design an innovative dual attention mechanism to filter out the critical features of the target objects efficiently. The attack performance of the proposed method is evaluated on MS COCO and BDD100K datasets with FasterRCNN and YOLOv5. Evaluation results indicate that the proposed targeted feature space attack method shows significant improvements in terms of image-specific, universality, and generalization attack performance, compared with the previous targeted attack for object detection.

摘要: 最近的研究表明，目标检测网络通常容易受到敌意例子的攻击。通常，用于目标检测的对抗性攻击可以分为目标攻击和非目标攻击。与非定向攻击相比，定向攻击提出了更大的挑战，现有的所有定向攻击方法都是通过误导检测器将检测到的对象错误地标记为特定的错误标签来发起攻击。然而，由于这些方法必须依赖于受害者图像中检测到的对象的存在，它们在攻击场景和攻击成功率方面受到限制。在本文中，我们提出了一种目标特征空间攻击方法，该方法可以误导检测器‘捏造’额外的指定对象，而不管受害者图像中是否包含对象。具体地说，我们引入引导图像来提取目标对象的粗粒度特征，并设计了一种创新的双重注意机制来高效地过滤出目标对象的关键特征。使用FasterRCNN和YOLOv5对该方法在MS COCO和BDD100K数据集上的攻击性能进行了评估。评估结果表明，与已有的目标检测的目标攻击方法相比，本文提出的目标特征空间攻击方法在图像专用性、通用性和泛化性能方面都有明显的提高。



## **50. Towards Physically-Realizable Adversarial Attacks in Embodied Vision Navigation**

视觉导航中实现物理可实现的对抗攻击 cs.CV

8 pages, 6 figures, submitted to the 2025 IEEE International  Conference on Robotics & Automation (ICRA)

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.10071v2) [paper-pdf](http://arxiv.org/pdf/2409.10071v2)

**Authors**: Meng Chen, Jiawei Tu, Chao Qi, Yonghao Dang, Feng Zhou, Wei Wei, Jianqin Yin

**Abstract**: The deployment of embodied navigation agents in safety-critical environments raises concerns about their vulnerability to adversarial attacks on deep neural networks. However, current attack methods often lack practicality due to challenges in transitioning from the digital to the physical world, while existing physical attacks for object detection fail to achieve both multi-view effectiveness and naturalness. To address this, we propose a practical attack method for embodied navigation by attaching adversarial patches with learnable textures and opacity to objects. Specifically, to ensure effectiveness across varying viewpoints, we employ a multi-view optimization strategy based on object-aware sampling, which uses feedback from the navigation model to optimize the patch's texture. To make the patch inconspicuous to human observers, we introduce a two-stage opacity optimization mechanism, where opacity is refined after texture optimization. Experimental results show our adversarial patches reduce navigation success rates by about 40%, outperforming previous methods in practicality, effectiveness, and naturalness. Code is available at: [https://github.com/chen37058/Physical-Attacks-in-Embodied-Navigation].

摘要: 在安全关键环境中部署具体化导航代理引起了人们对它们在深层神经网络上易受敌意攻击的担忧。然而，由于从数字世界向物理世界过渡的挑战，现有的攻击方法往往缺乏实用性，而现有的针对目标检测的物理攻击无法达到多视角的有效性和自然性。为了解决这一问题，我们提出了一种实用的具身导航攻击方法，通过将具有可学习纹理和不透明度的敌意补丁附加到对象上。具体地说，为了确保不同视点的有效性，我们采用了一种基于对象感知采样的多视点优化策略，该策略利用导航模型的反馈来优化面片的纹理。为了使面片不易被人察觉，我们引入了一种两阶段不透明度优化机制，在纹理优化后对不透明度进行细化。实验结果表明，我们的对抗性补丁使导航成功率降低了约40%，在实用性、有效性和自然性方面都优于以往的方法。代码可从以下网址获得：[https://github.com/chen37058/Physical-Attacks-in-Embodied-Navigation].



