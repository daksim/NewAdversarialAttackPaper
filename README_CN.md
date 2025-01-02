# Latest Adversarial Attack Papers
**update at 2025-01-02 10:09:47**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adversarial Attack and Defense for LoRa Device Identification and Authentication via Deep Learning**

通过深度学习进行LoRa设备识别和认证的对抗性攻击和防御 cs.NI

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.21164v1) [paper-pdf](http://arxiv.org/pdf/2412.21164v1)

**Authors**: Yalin E. Sagduyu, Tugba Erpek

**Abstract**: LoRa provides long-range, energy-efficient communications in Internet of Things (IoT) applications that rely on Low-Power Wide-Area Network (LPWAN) capabilities. Despite these merits, concerns persist regarding the security of LoRa networks, especially in situations where device identification and authentication are imperative to secure the reliable access to the LoRa networks. This paper explores a deep learning (DL) approach to tackle these concerns, focusing on two critical tasks, namely (i) identifying LoRa devices and (ii) classifying them to legitimate and rogue devices. Deep neural networks (DNNs), encompassing both convolutional and feedforward neural networks, are trained for these tasks using actual LoRa signal data. In this setting, the adversaries may spoof rogue LoRa signals through the kernel density estimation (KDE) method based on legitimate device signals that are received by the adversaries. Two cases are considered, (i) training two separate classifiers, one for each of the two tasks, and (ii) training a multi-task classifier for both tasks. The vulnerabilities of the resulting DNNs to manipulations in input samples are studied in form of untargeted and targeted adversarial attacks using the Fast Gradient Sign Method (FGSM). Individual and common perturbations are considered against single-task and multi-task classifiers for the LoRa signal analysis. To provide resilience against such attacks, a defense approach is presented by increasing the robustness of classifiers with adversarial training. Results quantify how vulnerable LoRa signal classification tasks are to adversarial attacks and emphasize the need to fortify IoT applications against these subtle yet effective threats.

摘要: LORA在依赖低功耗广域网络(LPWAN)功能的物联网(IoT)应用中提供远程、高能效通信。尽管有这些优点，人们仍然对LoRa网络的安全感到担忧，特别是在必须进行设备识别和认证才能确保可靠访问LoRa网络的情况下。本文探索了一种深度学习的方法来解决这些问题，重点关注两个关键任务，即(I)识别LoRa设备和(Ii)将它们分类为合法设备和流氓设备。深度神经网络(DNN)包括卷积神经网络和前馈神经网络，使用实际的LORA信号数据进行训练。在这种情况下，攻击者可以通过基于攻击者接收到的合法设备信号的核密度估计(KDE)方法来欺骗恶意LORA信号。考虑了两种情况，(I)训练两个单独的分类器，两个任务中的每一个一个，以及(Ii)为两个任务训练一个多任务分类器。使用快速梯度符号方法(FGSM)，以非目标攻击和目标攻击的形式研究了所得到的DNN对输入样本中的操纵的脆弱性。在LORA信号分析中，针对单任务和多任务分类器，考虑了单个和共同的扰动。为了提供对此类攻击的恢复能力，提出了一种通过对抗性训练提高分类器的稳健性的防御方法。结果量化了LORA信号分类任务面对对手攻击的脆弱性，并强调需要加强物联网应用程序以抵御这些微妙但有效的威胁。



## **2. BridgePure: Revealing the Fragility of Black-box Data Protection**

BridgePure：揭示黑匣子数据保护的脆弱性 cs.LG

26 pages,13 figures

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.21061v1) [paper-pdf](http://arxiv.org/pdf/2412.21061v1)

**Authors**: Yihan Wang, Yiwei Lu, Xiao-Shan Gao, Gautam Kamath, Yaoliang Yu

**Abstract**: Availability attacks, or unlearnable examples, are defensive techniques that allow data owners to modify their datasets in ways that prevent unauthorized machine learning models from learning effectively while maintaining the data's intended functionality. It has led to the release of popular black-box tools for users to upload personal data and receive protected counterparts. In this work, we show such black-box protections can be substantially bypassed if a small set of unprotected in-distribution data is available. Specifically, an adversary can (1) easily acquire (unprotected, protected) pairs by querying the black-box protections with the unprotected dataset; and (2) train a diffusion bridge model to build a mapping. This mapping, termed BridgePure, can effectively remove the protection from any previously unseen data within the same distribution. Under this threat model, our method demonstrates superior purification performance on classification and style mimicry tasks, exposing critical vulnerabilities in black-box data protection.

摘要: 可用性攻击，或无法学习的例子，是一种防御性技术，允许数据所有者修改他们的数据集，以防止未经授权的机器学习模型有效学习，同时保持数据的预期功能。这导致了流行的黑盒工具的发布，用户可以上传个人数据并接收受保护的对应数据。在这项工作中，我们表明，如果有一小部分未受保护的分发中数据可用，则可以实质上绕过此类黑盒保护。具体地说，敌手可以(1)通过使用未受保护的数据集查询黑盒保护来容易地获得(未受保护的，受保护的)对；以及(2)训练扩散桥模型来建立映射。这种名为BridgePure的映射可以有效地消除对同一分发中任何以前不可见的数据的保护。在这种威胁模型下，我们的方法在分类和风格模仿任务上表现出了优越的净化性能，暴露了黑盒数据保护中的关键漏洞。



## **3. RobustBlack: Challenging Black-Box Adversarial Attacks on State-of-the-Art Defenses**

RobustBlack：对抗对最先进防御的黑匣子对抗攻击 cs.LG

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20987v1) [paper-pdf](http://arxiv.org/pdf/2412.20987v1)

**Authors**: Mohamed Djilani, Salah Ghamizi, Maxime Cordy

**Abstract**: Although adversarial robustness has been extensively studied in white-box settings, recent advances in black-box attacks (including transfer- and query-based approaches) are primarily benchmarked against weak defenses, leaving a significant gap in the evaluation of their effectiveness against more recent and moderate robust models (e.g., those featured in the Robustbench leaderboard). In this paper, we question this lack of attention from black-box attacks to robust models. We establish a framework to evaluate the effectiveness of recent black-box attacks against both top-performing and standard defense mechanisms, on the ImageNet dataset. Our empirical evaluation reveals the following key findings: (1) the most advanced black-box attacks struggle to succeed even against simple adversarially trained models; (2) robust models that are optimized to withstand strong white-box attacks, such as AutoAttack, also exhibits enhanced resilience against black-box attacks; and (3) robustness alignment between the surrogate models and the target model plays a key factor in the success rate of transfer-based attacks

摘要: 尽管在白盒环境中已经对对抗健壮性进行了广泛的研究，但黑盒攻击(包括基于转移和基于查询的方法)的最新进展主要是以弱防御为基准的，在评估其有效性方面与较新的中等健壮性模型(例如，罗布斯堡垒排行榜中的那些模型)相比存在很大差距。在这篇文章中，我们质疑这种从黑箱攻击到稳健模型的缺乏关注。我们建立了一个框架来评估最近针对ImageNet数据集上最高性能和标准防御机制的黑盒攻击的有效性。我们的经验评估揭示了以下关键发现：(1)最高级的黑盒攻击即使在简单的对抗性训练模型下也很难成功；(2)经过优化以抵抗强大的白盒攻击的健壮模型，例如AutoAttack，也表现出对黑盒攻击的更强的弹性；以及(3)代理模型和目标模型之间的健壮性对齐在基于转移的攻击的成功率中起着关键作用



## **4. GASLITEing the Retrieval: Exploring Vulnerabilities in Dense Embedding-based Search**

GASLITEING检索：探索基于密集嵌入的搜索中的漏洞 cs.CR

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20953v1) [paper-pdf](http://arxiv.org/pdf/2412.20953v1)

**Authors**: Matan Ben-Tov, Mahmood Sharif

**Abstract**: Dense embedding-based text retrieval$\unicode{x2013}$retrieval of relevant passages from corpora via deep learning encodings$\unicode{x2013}$has emerged as a powerful method attaining state-of-the-art search results and popularizing the use of Retrieval Augmented Generation (RAG). Still, like other search methods, embedding-based retrieval may be susceptible to search-engine optimization (SEO) attacks, where adversaries promote malicious content by introducing adversarial passages to corpora. To faithfully assess and gain insights into the susceptibility of such systems to SEO, this work proposes the GASLITE attack, a mathematically principled gradient-based search method for generating adversarial passages without relying on the corpus content or modifying the model. Notably, GASLITE's passages (1) carry adversary-chosen information while (2) achieving high retrieval ranking for a selected query distribution when inserted to corpora. We use GASLITE to extensively evaluate retrievers' robustness, testing nine advanced models under varied threat models, while focusing on realistic adversaries targeting queries on a specific concept (e.g., a public figure). We found GASLITE consistently outperformed baselines by $\geq$140% success rate, in all settings. Particularly, adversaries using GASLITE require minimal effort to manipulate search results$\unicode{x2013}$by injecting a negligible amount of adversarial passages ($\leq$0.0001% of the corpus), they could make them visible in the top-10 results for 61-100% of unseen concept-specific queries against most evaluated models. Inspecting variance in retrievers' robustness, we identify key factors that may contribute to models' susceptibility to SEO, including specific properties in the embedding space's geometry.

摘要: 基于密集嵌入的文本检索$\unicode{x2013}$通过深度学习编码从语料库中检索相关段落$\unicode{x2013}$已成为获得最先进的搜索结果并普及检索增强一代(RAG)的一种强大方法。尽管如此，像其他搜索方法一样，基于嵌入的检索可能容易受到搜索引擎优化(SEO)攻击，即对手通过向语料库引入敌意段落来推广恶意内容。为了忠实地评估和洞察这类系统对SEO的敏感性，本文提出了GASLITE攻击，这是一种基于数学原理的基于梯度的搜索方法，可以在不依赖语料库内容或修改模型的情况下生成对抗性段落。值得注意的是，GASLITE的段落(1)携带对手选择的信息，而(2)在插入到语料库时，对选定的查询分布实现了较高的检索排名。我们使用GASLITE来广泛地评估检索器的健壮性，在不同的威胁模型下测试了九个高级模型，同时专注于针对特定概念(例如，公众人物)的查询的现实对手。我们发现，在所有情况下，GASLITE的成功率都比基线高出140%。特别是，使用GASLITE的攻击者只需很少的努力就可以操纵搜索结果$\unicode{x2013}$通过注入微不足道的对抗性段落($\leq$0.0001%的语料库)，他们可以使它们在针对大多数评估模型的未见概念特定查询的前10个结果中可见。通过考察检索者稳健性的差异，我们确定了可能导致模型对SEO易感性的关键因素，包括嵌入空间几何结构中的特定属性。



## **5. Two Heads Are Better Than One: Averaging along Fine-Tuning to Improve Targeted Transferability**

两个头总比一个头好：利用微调来提高目标可转让性 cs.CV

9 pages, 6 figures, accepted by 2025ICASSP

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20807v1) [paper-pdf](http://arxiv.org/pdf/2412.20807v1)

**Authors**: Hui Zeng, Sanshuai Cui, Biwei Chen, Anjie Peng

**Abstract**: With much longer optimization time than that of untargeted attacks notwithstanding, the transferability of targeted attacks is still far from satisfactory. Recent studies reveal that fine-tuning an existing adversarial example (AE) in feature space can efficiently boost its targeted transferability. However, existing fine-tuning schemes only utilize the endpoint and ignore the valuable information in the fine-tuning trajectory. Noting that the vanilla fine-tuning trajectory tends to oscillate around the periphery of a flat region of the loss surface, we propose averaging over the fine-tuning trajectory to pull the crafted AE towards a more centered region. We compare the proposed method with existing fine-tuning schemes by integrating them with state-of-the-art targeted attacks in various attacking scenarios. Experimental results uphold the superiority of the proposed method in boosting targeted transferability. The code is available at github.com/zengh5/Avg_FT.

摘要: 尽管优化时间比非目标攻击长得多，但目标攻击的可转移性仍然远不能令人满意。最近的研究表明，微调特征空间中现有的对抗性示例（AE）可以有效地提高其目标可移植性。然而，现有的微调方案只利用端点而忽略了微调轨迹中的有价值信息。注意到普通微调轨迹往往会围绕损失表面平坦区域的外围振荡，我们建议对微调轨迹进行平均，以将精心制作的AE拉向更中心的区域。我们通过将所提出的方法与现有的微调方案与各种攻击场景中的最先进的定向攻击相结合，将其与现有的微调方案进行比较。实验结果证实了所提出的方法在提高目标可移植性方面的优越性。该代码可在github.com/zengh5/Avg_FT上获取。



## **6. DV-FSR: A Dual-View Target Attack Framework for Federated Sequential Recommendation**

DV-FSR：一种用于联合顺序推荐的双视图目标攻击框架 cs.CR

I am requesting the withdrawal of my paper due to identified errors  that require significant revision

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2409.07500v2) [paper-pdf](http://arxiv.org/pdf/2409.07500v2)

**Authors**: Qitao Qin, Yucong Luo, Mingyue Cheng, Qingyang Mao, Chenyi Lei

**Abstract**: Federated recommendation (FedRec) preserves user privacy by enabling decentralized training of personalized models, but this architecture is inherently vulnerable to adversarial attacks. Significant research has been conducted on targeted attacks in FedRec systems, motivated by commercial and social influence considerations. However, much of this work has largely overlooked the differential robustness of recommendation models. Moreover, our empirical findings indicate that existing targeted attack methods achieve only limited effectiveness in Federated Sequential Recommendation (FSR) tasks. Driven by these observations, we focus on investigating targeted attacks in FSR and propose a novel dualview attack framework, named DV-FSR. This attack method uniquely combines a sampling-based explicit strategy with a contrastive learning-based implicit gradient strategy to orchestrate a coordinated attack. Additionally, we introduce a specific defense mechanism tailored for targeted attacks in FSR, aiming to evaluate the mitigation effects of the attack method we proposed. Extensive experiments validate the effectiveness of our proposed approach on representative sequential models.

摘要: 联邦推荐(FedRec)通过支持个性化模型的分散训练来保护用户隐私，但这种体系结构天生就容易受到敌意攻击。出于商业和社会影响的考虑，对FedRec系统中的目标攻击进行了重要的研究。然而，这些工作在很大程度上忽略了推荐模型的差异化稳健性。此外，我们的实验结果表明，现有的定向攻击方法在联邦顺序推荐(FSR)任务中只能取得有限的效果。在此基础上，我们重点研究了FSR中的目标攻击，并提出了一种新的DualView攻击框架DV-FSR。该攻击方法独特地结合了基于采样的显式策略和基于对比学习的隐式梯度策略来协调攻击。此外，我们在FSR中引入了一种针对目标攻击的特定防御机制，旨在评估我们提出的攻击方法的缓解效果。在典型的序列模型上进行了大量的实验，验证了该方法的有效性。



## **7. Sample Correlation for Fingerprinting Deep Face Recognition**

指纹深度人脸识别的样本相关性 cs.CV

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20768v1) [paper-pdf](http://arxiv.org/pdf/2412.20768v1)

**Authors**: Jiyang Guan, Jian Liang, Yanbo Wang, Ran He

**Abstract**: Face recognition has witnessed remarkable advancements in recent years, thanks to the development of deep learning techniques.However, an off-the-shelf face recognition model as a commercial service could be stolen by model stealing attacks, posing great threats to the rights of the model owner.Model fingerprinting, as a model stealing detection method, aims to verify whether a suspect model is stolen from the victim model, gaining more and more attention nowadays.Previous methods always utilize transferable adversarial examples as the model fingerprint, but this method is known to be sensitive to adversarial defense and transfer learning techniques.To address this issue, we consider the pairwise relationship between samples instead and propose a novel yet simple model stealing detection method based on SAmple Correlation (SAC).Specifically, we present SAC-JC that selects JPEG compressed samples as model inputs and calculates the correlation matrix among their model outputs.Extensive results validate that SAC successfully defends against various model stealing attacks in deep face recognition, encompassing face verification and face emotion recognition, exhibiting the highest performance in terms of AUC, p-value and F1 score.Furthermore, we extend our evaluation of SAC-JC to object recognition datasets including Tiny-ImageNet and CIFAR10, which also demonstrates the superior performance of SAC-JC to previous methods.The code will be available at \url{https://github.com/guanjiyang/SAC_JC}.

摘要: 近年来，随着深度学习技术的发展，人脸识别技术取得了显著的进步。然而，现有的人脸识别模型作为一种商业服务，可能会被模型窃取攻击窃取，这对模型所有者的权利构成了极大的威胁。模型指纹作为一种模型窃取检测方法，旨在验证嫌疑人模型是否从受害者模型中被盗，得到了越来越多的关注。以前的方法通常使用可转移的对手样本作为模型指纹，但这种方法对攻击防御和转移学习技术非常敏感。针对这一问题，提出了一种基于模型指纹的人脸识别方法考虑了样本之间的成对关系，提出了一种新颖而简单的基于样本相关的模型窃取检测方法(SAC-JC)。具体地，我们选择JPEG压缩样本作为模型输入，并计算模型输出之间的相关矩阵。大量的实验结果验证了SAC-JC在深度人脸识别中成功地抵抗了各种模型窃取攻击，包括人脸验证和人脸情感识别，在AUC、p值和F1得分方面表现出最高的性能。此外，我们将SAC-JC的评估扩展到目标识别数据集，包括微小图像网和CIFAR10，这也展示了SAC-JC相对于以前方法的卓越性能。代码将在\url{https://github.com/guanjiyang/SAC_JC}.



## **8. Enhancing Privacy in Federated Learning through Quantum Teleportation Integration**

通过量子隐形传输集成增强联邦学习中的隐私 quant-ph

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20762v1) [paper-pdf](http://arxiv.org/pdf/2412.20762v1)

**Authors**: Koffka Khan

**Abstract**: Federated learning enables collaborative model training across multiple clients without sharing raw data, thereby enhancing privacy. However, the exchange of model updates can still expose sensitive information. Quantum teleportation, a process that transfers quantum states between distant locations without physical transmission of the particles themselves, has recently been implemented in real-world networks. This position paper explores the potential of integrating quantum teleportation into federated learning frameworks to bolster privacy. By leveraging quantum entanglement and the no-cloning theorem, quantum teleportation ensures that data remains secure during transmission, as any eavesdropping attempt would be detectable. We propose a novel architecture where quantum teleportation facilitates the secure exchange of model parameters and gradients among clients and servers. This integration aims to mitigate risks associated with data leakage and adversarial attacks inherent in classical federated learning setups. We also discuss the practical challenges of implementing such a system, including the current limitations of quantum network infrastructure and the need for hybrid quantum-classical protocols. Our analysis suggests that, despite these challenges, the convergence of quantum communication technologies and federated learning presents a promising avenue for achieving unprecedented levels of privacy in distributed machine learning.

摘要: 联合学习实现了跨多个客户的协作模型培训，而无需共享原始数据，从而增强了隐私。然而，模型更新的交换仍然可能暴露敏感信息。量子隐形传态是一种在遥远的位置之间传输量子态的过程，而不需要粒子本身的物理传输，最近已经在现实世界的网络中实现。这份立场文件探索了将量子隐形传态整合到联邦学习框架中以保护隐私的潜力。通过利用量子纠缠和不可克隆定理，量子隐形传态确保了数据在传输过程中保持安全，因为任何窃听企图都是可以检测到的。我们提出了一种新的体系结构，其中量子隐形传态促进了客户端和服务器之间模型参数和梯度的安全交换。这种集成旨在降低与传统联合学习设置中固有的数据泄露和对抗性攻击相关的风险。我们还讨论了实现这样一个系统的实际挑战，包括目前量子网络基础设施的限制以及对混合量子经典协议的需求。我们的分析表明，尽管存在这些挑战，但量子通信技术和联邦学习的融合为在分布式机器学习中实现前所未有的隐私水平提供了一条有希望的途径。



## **9. Unsupervised dense retrieval with conterfactual contrastive learning**

具有反事实对比学习的无监督密集检索 cs.IR

arXiv admin note: text overlap with arXiv:2107.07773 by other authors

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20756v1) [paper-pdf](http://arxiv.org/pdf/2412.20756v1)

**Authors**: Haitian Chen, Qingyao Ai, Xiao Wang, Yiqun Liu, Fen Lin, Qin Liu

**Abstract**: Efficiently retrieving a concise set of candidates from a large document corpus remains a pivotal challenge in Information Retrieval (IR). Neural retrieval models, particularly dense retrieval models built with transformers and pretrained language models, have been popular due to their superior performance. However, criticisms have also been raised on their lack of explainability and vulnerability to adversarial attacks. In response to these challenges, we propose to improve the robustness of dense retrieval models by enhancing their sensitivity of fine-graned relevance signals. A model achieving sensitivity in this context should exhibit high variances when documents' key passages determining their relevance to queries have been modified, while maintaining low variances for other changes in irrelevant passages. This sensitivity allows a dense retrieval model to produce robust results with respect to attacks that try to promote documents without actually increasing their relevance. It also makes it possible to analyze which part of a document is actually relevant to a query, and thus improve the explainability of the retrieval model. Motivated by causality and counterfactual analysis, we propose a series of counterfactual regularization methods based on game theory and unsupervised learning with counterfactual passages. Experiments show that, our method can extract key passages without reliance on the passage-level relevance annotations. Moreover, the regularized dense retrieval models exhibit heightened robustness against adversarial attacks, surpassing the state-of-the-art anti-attack methods.

摘要: 从大型文档语料库中高效地检索一组简明的候选对象仍然是信息检索(IR)中的一个关键挑战。神经检索模型，特别是用转换器构建的密集检索模型和预先训练的语言模型，由于其优越的性能而受到广泛的欢迎。然而，也有人批评说，它们缺乏可解释性，容易受到对手攻击。为了应对这些挑战，我们提出了通过提高密集检索模型对细粒度关联信号的敏感度来提高其稳健性。在这种情况下实现敏感性的模型应该在确定其与查询的相关性的文档的关键段落被修改时表现出高方差，同时保持对不相关段落中的其他变化的低方差。这种敏感度使得密集检索模型能够针对试图提升文档而不实际增加其相关性的攻击产生稳健的结果。它还可以分析文档的哪个部分实际上与查询相关，从而提高检索模型的可解释性。在因果关系和反事实分析的启发下，我们提出了一系列基于博弈论和带有反事实段落的无监督学习的反事实正则化方法。实验表明，我们的方法可以在不依赖于段落级关联标注的情况下提取关键段落。此外，正则化的密集检索模型表现出对对手攻击的高度稳健性，超过了最先进的反攻击方法。



## **10. Graph Neural Networks for Next-Generation-IoT: Recent Advances and Open Challenges**

下一代物联网的图形神经网络：最近的进展和开放的挑战 cs.IT

28 pages, 15 figures, and 6 tables. Submitted for publication

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20634v1) [paper-pdf](http://arxiv.org/pdf/2412.20634v1)

**Authors**: Nguyen Xuan Tung, Le Tung Giang, Bui Duc Son, Seon Geun Jeong, Trinh Van Chien, Won Joo Hwang, Lajos Hanzo

**Abstract**: Graph Neural Networks (GNNs) have emerged as a critical tool for optimizing and managing the complexities of the Internet of Things (IoT) in next-generation networks. This survey presents a comprehensive exploration of how GNNs may be harnessed in 6G IoT environments, focusing on key challenges and opportunities through a series of open questions. We commence with an exploration of GNN paradigms and the roles of node, edge, and graph-level tasks in solving wireless networking problems and highlight GNNs' ability to overcome the limitations of traditional optimization methods. This guidance enhances problem-solving efficiency across various next-generation (NG) IoT scenarios. Next, we provide a detailed discussion of the application of GNN in advanced NG enabling technologies, including massive MIMO, reconfigurable intelligent surfaces, satellites, THz, mobile edge computing (MEC), and ultra-reliable low latency communication (URLLC). We then delve into the challenges posed by adversarial attacks, offering insights into defense mechanisms to secure GNN-based NG-IoT networks. Next, we examine how GNNs can be integrated with future technologies like integrated sensing and communication (ISAC), satellite-air-ground-sea integrated networks (SAGSIN), and quantum computing. Our findings highlight the transformative potential of GNNs in improving efficiency, scalability, and security within NG-IoT systems, paving the way for future advances. Finally, we propose a set of design guidelines to facilitate the development of efficient, scalable, and secure GNN models tailored for NG IoT applications.

摘要: 图形神经网络(GNN)已经成为下一代网络中优化和管理物联网(IoT)复杂性的关键工具。这项调查全面探讨了如何在6G物联网环境中利用GNN，并通过一系列开放问题重点介绍了关键挑战和机遇。我们首先探讨GNN范例以及节点、边和图级任务在解决无线网络问题中的作用，并强调GNN克服传统优化方法局限性的能力。本指南提高了各种下一代(NG)物联网场景中的问题解决效率。接下来，我们将详细讨论GNN在先进的NG使能技术中的应用，包括大规模MIMO、可重构智能表面、卫星、太赫兹、移动边缘计算(MEC)和超可靠低延迟通信(URLLC)。然后，我们深入研究对抗性攻击带来的挑战，深入了解保护基于GNN的NG-IoT网络的防御机制。接下来，我们研究如何将GNN与未来的技术相结合，如综合传感和通信(ISAC)、卫星-空中-地面-海洋综合网络(SAGSIN)和量子计算。我们的发现突出了GNN在提高NG-IoT系统内的效率、可扩展性和安全性方面的变革潜力，为未来的发展铺平了道路。最后，我们提出了一套设计指南，以促进为下一代物联网应用定制的高效、可扩展和安全的GNN模型的开发。



## **11. ErasableMask: A Robust and Erasable Privacy Protection Scheme against Black-box Face Recognition Models**

ErasableMass：针对黑匣子人脸识别模型的稳健且可擦除的隐私保护方案 cs.CV

**SubmitDate**: 2024-12-29    [abs](http://arxiv.org/abs/2412.17038v3) [paper-pdf](http://arxiv.org/pdf/2412.17038v3)

**Authors**: Sipeng Shen, Yunming Zhang, Dengpan Ye, Xiuwen Shi, Long Tang, Haoran Duan, Jiacheng Deng, Ziyi Liu

**Abstract**: While face recognition (FR) models have brought remarkable convenience in face verification and identification, they also pose substantial privacy risks to the public. Existing facial privacy protection schemes usually adopt adversarial examples to disrupt face verification of FR models. However, these schemes often suffer from weak transferability against black-box FR models and permanently damage the identifiable information that cannot fulfill the requirements of authorized operations such as forensics and authentication. To address these limitations, we propose ErasableMask, a robust and erasable privacy protection scheme against black-box FR models. Specifically, via rethinking the inherent relationship between surrogate FR models, ErasableMask introduces a novel meta-auxiliary attack, which boosts black-box transferability by learning more general features in a stable and balancing optimization strategy. It also offers a perturbation erasion mechanism that supports the erasion of semantic perturbations in protected face without degrading image quality. To further improve performance, ErasableMask employs a curriculum learning strategy to mitigate optimization conflicts between adversarial attack and perturbation erasion. Extensive experiments on the CelebA-HQ and FFHQ datasets demonstrate that ErasableMask achieves the state-of-the-art performance in transferability, achieving over 72% confidence on average in commercial FR systems. Moreover, ErasableMask also exhibits outstanding perturbation erasion performance, achieving over 90% erasion success rate.

摘要: 虽然人脸识别(FR)模型在人脸验证和识别方面带来了显著的便利，但它们也给公众带来了巨大的隐私风险。现有的人脸隐私保护方案通常采用对抗性的例子来干扰FR模型的人脸验证。然而，这些方案往往对黑盒FR模型的可转移性较弱，并且永久性地破坏了不能满足取证和认证等授权操作要求的可识别信息。为了解决这些局限性，我们提出了一种针对黑盒FR模型的健壮且可擦除的隐私保护方案--可擦除掩码。具体地说，通过重新考虑代理FR模型之间的内在联系，ErasableMASK引入了一种新的元辅助攻击，该攻击通过学习稳定平衡的优化策略中的更多一般特征来提高黑盒的可转移性。它还提供了一种扰动消除机制，支持在不降低图像质量的情况下消除受保护人脸的语义扰动。为了进一步提高性能，ErasableMASK采用了课程学习策略来缓解对抗性攻击和扰动擦除之间的优化冲突。在CelebA-HQ和FFHQ数据集上的广泛实验表明，可擦除掩码在可转移性方面达到了最先进的性能，在商业FR系统中平均达到72%以上的置信度。此外，可擦除掩模还表现出出色的扰动擦除性能，擦除成功率达到90%以上。



## **12. Real-time Fake News from Adversarial Feedback**

来自对抗反馈的实时假新闻 cs.CL

**SubmitDate**: 2024-12-29    [abs](http://arxiv.org/abs/2410.14651v2) [paper-pdf](http://arxiv.org/pdf/2410.14651v2)

**Authors**: Sanxing Chen, Yukun Huang, Bhuwan Dhingra

**Abstract**: We show that existing evaluations for fake news detection based on conventional sources, such as claims on fact-checking websites, result in high accuracies over time for LLM-based detectors -- even after their knowledge cutoffs. This suggests that recent popular fake news from such sources can be easily detected due to pre-training and retrieval corpus contamination or increasingly salient shallow patterns. Instead, we argue that a proper fake news detection dataset should test a model's ability to reason factually about the current world by retrieving and reading related evidence. To this end, we develop a novel pipeline that leverages natural language feedback from a RAG-based detector to iteratively modify real-time news into deceptive fake news that challenges LLMs. Our iterative rewrite decreases the binary classification ROC-AUC by an absolute 17.5 percent for a strong RAG-based GPT-4o detector. Our experiments reveal the important role of RAG in both detecting and generating fake news, as retrieval-free LLM detectors are vulnerable to unseen events and adversarial attacks, while feedback from RAG detection helps discover more deceitful patterns in fake news.

摘要: 我们表明，现有的基于传统来源的假新闻检测评估，例如在事实核查网站上的声明，导致基于LLM的检测器随着时间的推移而获得高精度-即使在他们的知识中断之后。这表明，由于预训练和检索语料库的污染或日益突出的浅层模式，来自这些来源的最近流行的假新闻可以很容易地被检测出来。相反，我们认为，一个适当的假新闻检测数据集应该通过检索和阅读相关证据来测试模型对当前世界进行事实推理的能力。为此，我们开发了一种新的流水线，利用来自基于RAG的检测器的自然语言反馈来迭代地将实时新闻修改为挑战LLMS的欺骗性假新闻。对于一个强的基于RAG的GPT-40检测器，我们的迭代重写使二进制分类ROC-AUC绝对减少了17.5%。我们的实验揭示了RAG在检测和生成假新闻中的重要作用，因为免检索LLM检测器容易受到不可见事件和对手攻击的攻击，而RAG检测的反馈有助于在假新闻中发现更多的欺骗性模式。



## **13. Optimal and Feasible Contextuality-based Randomness Generation**

最佳可行的基于上下文的随机生成 quant-ph

21 pages, 8 figures

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2412.20126v1) [paper-pdf](http://arxiv.org/pdf/2412.20126v1)

**Authors**: Yuan Liu, Ravishankar Ramanathan

**Abstract**: Semi-device-independent (SDI) randomness generation protocols based on Kochen-Specker contextuality offer the attractive features of compact devices, high rates, and ease of experimental implementation over fully device-independent (DI) protocols. Here, we investigate this paradigm and derive four results to improve the state-of-art. Firstly, we introduce a family of simple, experimentally feasible orthogonality graphs (measurement compatibility structures) for which the maximum violation of the corresponding non-contextuality inequalities allows to certify the maximum amount of $\log_2 d$ bits from a qu$d$it system with projective measurements for $d \geq 3$. We analytically derive the Lovasz theta and fractional packing number for this graph family, and thereby prove their utility for optimal randomness generation in both randomness expansion and amplification tasks. Secondly, a central additional assumption in contextuality-based protocols over fully DI ones, is that the measurements are repeatable and satisfy an intended compatibility structure. We frame a relaxation of this condition in terms of $\epsilon$-orthogonality graphs for a parameter $\epsilon > 0$, and derive quantum correlations that allow to certify randomness for arbitrary relaxation $\epsilon \in [0,1)$. Thirdly, it is well known that a single qubit is non-contextual, i.e., the qubit correlations can be explained by a non-contextual hidden variable (NCHV) model. We show however that a single qubit is \textit{almost} contextual, in that there exist qubit correlations that cannot be explained by $\epsilon$-ontologically faithful NCHV models for small $\epsilon > 0$. Finally, we point out possible attacks by quantum and general consistent (non-signalling) adversaries for certain classes of contextuality tests over and above those considered in DI scenarios.

摘要: 基于Kochen-specker上下文的半设备无关(SDI)随机性生成协议具有设备紧凑、速率高、易于实验实现等特点，优于完全设备无关(DI)协议。在这里，我们研究了这一范式，并得出了四个结果来提高最新水平。首先，我们引入了一族简单的，实验上可行的正交图(测量相容结构)，对于它，对应的非上下文不等式的最大违反允许证明具有$d\geq 3$的射影测量的Qu$d$it系统的最大$\log_2 d$比特的数量。我们解析地推导了这个图族的Lovaszθ和分数填充数，从而证明了它们在随机性扩展和放大任务中的最优随机性生成方面的有效性。其次，与完全依赖注入协议相比，基于上下文的协议的一个核心附加假设是测量是可重复的，并且满足预期的兼容性结构。对于参数$\epsilon>0$，我们用$-epsilon$-正交图表示这一条件的松弛，并推导出量子关联，它允许证明[0，1]$中任意松弛的随机性。第三，众所周知，单个量子比特是非上下文相关的，即量子比特之间的关联可以用非上下文隐藏变量(NCHV)模型来解释。然而，我们证明了单个量子比特是与上下文相关的，这是因为存在着不能用$epsilon>0$的本体论忠实的NCHV模型来解释的量子比特关联。最后，我们指出了量子和一般一致(非信令)对手对某些类别的上下文测试的可能攻击，这些测试超出了DI场景中考虑的测试。



## **14. On the Validity of Traditional Vulnerability Scoring Systems for Adversarial Attacks against LLMs**

传统漏洞评分系统对LLM对抗性攻击的有效性 cs.CR

101 pages, 3 figures

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2412.20087v1) [paper-pdf](http://arxiv.org/pdf/2412.20087v1)

**Authors**: Atmane Ayoub Mansour Bahar, Ahmad Samer Wazan

**Abstract**: This research investigates the effectiveness of established vulnerability metrics, such as the Common Vulnerability Scoring System (CVSS), in evaluating attacks against Large Language Models (LLMs), with a focus on Adversarial Attacks (AAs). The study explores the influence of both general and specific metric factors in determining vulnerability scores, providing new perspectives on potential enhancements to these metrics.   This study adopts a quantitative approach, calculating and comparing the coefficient of variation of vulnerability scores across 56 adversarial attacks on LLMs. The attacks, sourced from various research papers, and obtained through online databases, were evaluated using multiple vulnerability metrics. Scores were determined by averaging the values assessed by three distinct LLMs. The results indicate that existing scoring-systems yield vulnerability scores with minimal variation across different attacks, suggesting that many of the metric factors are inadequate for assessing adversarial attacks on LLMs. This is particularly true for context-specific factors or those with predefined value sets, such as those in CVSS. These findings support the hypothesis that current vulnerability metrics, especially those with rigid values, are limited in evaluating AAs on LLMs, highlighting the need for the development of more flexible, generalized metrics tailored to such attacks.   This research offers a fresh analysis of the effectiveness and applicability of established vulnerability metrics, particularly in the context of Adversarial Attacks on Large Language Models, both of which have gained significant attention in recent years. Through extensive testing and calculations, the study underscores the limitations of these metrics and opens up new avenues for improving and refining vulnerability assessment frameworks specifically tailored for LLMs.

摘要: 这项研究考察了通用漏洞评分系统(CVSS)等已建立的漏洞度量在评估针对大型语言模型(LLMS)的攻击时的有效性，重点是对抗性攻击(AA)。这项研究探讨了一般和特定指标因素在确定脆弱性得分方面的影响，为这些指标的潜在增强提供了新的视角。本研究采用定量的方法，计算并比较了56种对抗性攻击下的LLMS脆弱性得分的变异系数。这些攻击来自各种研究论文，通过在线数据库获得，使用多种漏洞指标进行评估。得分通过三个不同的LLM评估的值的平均值来确定。结果表明，现有的评分系统产生的脆弱性分数在不同攻击之间的差异很小，这表明许多度量因素不足以评估对LLM的对抗性攻击。对于特定于上下文的因素或具有预定义值集的因素尤其如此，例如CVSS中的那些因素。这些发现支持这样一种假设，即当前的脆弱性指标，特别是那些具有刚性值的指标，在评估LLM上的AA方面是有限的，这突显了开发针对此类攻击量身定做的更灵活、更通用的指标的必要性。这项研究对已建立的脆弱性度量的有效性和适用性进行了新的分析，特别是在针对大型语言模型的对抗性攻击的背景下，这两种攻击在最近几年都得到了极大的关注。通过广泛的测试和计算，这项研究强调了这些指标的局限性，并为改进和完善专门为低土地管理定制的脆弱性评估框架开辟了新的途径。



## **15. B-AVIBench: Towards Evaluating the Robustness of Large Vision-Language Model on Black-box Adversarial Visual-Instructions**

B-AVIBench：评估黑匣子对抗视觉指令上大型视觉语言模型的鲁棒性 cs.CV

Accepted by IEEE Transactions on Information Forensics & Security

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2403.09346v2) [paper-pdf](http://arxiv.org/pdf/2403.09346v2)

**Authors**: Hao Zhang, Wenqi Shao, Hong Liu, Yongqiang Ma, Ping Luo, Yu Qiao, Nanning Zheng, Kaipeng Zhang

**Abstract**: Large Vision-Language Models (LVLMs) have shown significant progress in responding well to visual-instructions from users. However, these instructions, encompassing images and text, are susceptible to both intentional and inadvertent attacks. Despite the critical importance of LVLMs' robustness against such threats, current research in this area remains limited. To bridge this gap, we introduce B-AVIBench, a framework designed to analyze the robustness of LVLMs when facing various Black-box Adversarial Visual-Instructions (B-AVIs), including four types of image-based B-AVIs, ten types of text-based B-AVIs, and nine types of content bias B-AVIs (such as gender, violence, cultural, and racial biases, among others). We generate 316K B-AVIs encompassing five categories of multimodal capabilities (ten tasks) and content bias. We then conduct a comprehensive evaluation involving 14 open-source LVLMs to assess their performance. B-AVIBench also serves as a convenient tool for practitioners to evaluate the robustness of LVLMs against B-AVIs. Our findings and extensive experimental results shed light on the vulnerabilities of LVLMs, and highlight that inherent biases exist even in advanced closed-source LVLMs like GeminiProVision and GPT-4V. This underscores the importance of enhancing the robustness, security, and fairness of LVLMs. The source code and benchmark are available at https://github.com/zhanghao5201/B-AVIBench.

摘要: 大型视觉语言模型(LVLM)在很好地响应用户的视觉指令方面取得了重大进展。但是，这些包含图像和文本的说明很容易受到有意和无意的攻击。尽管LVLMS对这类威胁的稳健性至关重要，但目前在这一领域的研究仍然有限。为了弥补这一差距，我们引入了B-AVIB边框架，该框架旨在分析LVLMS在面对各种黑盒对抗性视觉指令(B-AVI)时的健壮性，包括四种类型的基于图像的B-AVI、10种类型的基于文本的B-AVI和九种类型的内容偏见B-AVI(如性别、暴力、文化和种族偏见等)。我们生成了316k B-AVI，包括五类多模式能力(十项任务)和内容偏见。然后，我们对14个开源LVLM进行了全面的评估，以评估它们的性能。B-AVIBtch也可作为从业者评估LVLMS对B-AVIS的稳健性的便捷工具。我们的发现和广泛的实验结果揭示了LVLMS的漏洞，并突出表明即使在GeminiProVision和GPT-4V等先进的闭源LVLM中也存在固有偏差。这凸显了增强LVLM的健壮性、安全性和公平性的重要性。源代码和基准测试可在https://github.com/zhanghao5201/B-AVIBench.上获得



## **16. A Robust Adversarial Ensemble with Causal (Feature Interaction) Interpretations for Image Classification**

具有因果（特征相互作用）解释的图像分类鲁棒对抗集成 cs.CV

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2412.20025v1) [paper-pdf](http://arxiv.org/pdf/2412.20025v1)

**Authors**: Chunheng Zhao, Pierluigi Pisu, Gurcan Comert, Negash Begashaw, Varghese Vaidyan, Nina Christine Hubig

**Abstract**: Deep learning-based discriminative classifiers, despite their remarkable success, remain vulnerable to adversarial examples that can mislead model predictions. While adversarial training can enhance robustness, it fails to address the intrinsic vulnerability stemming from the opaque nature of these black-box models. We present a deep ensemble model that combines discriminative features with generative models to achieve both high accuracy and adversarial robustness. Our approach integrates a bottom-level pre-trained discriminative network for feature extraction with a top-level generative classification network that models adversarial input distributions through a deep latent variable model. Using variational Bayes, our model achieves superior robustness against white-box adversarial attacks without adversarial training. Extensive experiments on CIFAR-10 and CIFAR-100 demonstrate our model's superior adversarial robustness. Through evaluations using counterfactual metrics and feature interaction-based metrics, we establish correlations between model interpretability and adversarial robustness. Additionally, preliminary results on Tiny-ImageNet validate our approach's scalability to more complex datasets, offering a practical solution for developing robust image classification models.

摘要: 基于深度学习的判别分类器尽管取得了显著的成功，但仍然容易受到可能误导模型预测的对抗性例子的影响。虽然对抗性训练可以增强稳健性，但它无法解决这些黑箱模型的不透明性质所产生的内在脆弱性。我们提出了一种深度集成模型，该模型结合了区分特征和生成模型，以实现高准确率和对抗健壮性。我们的方法结合了用于特征提取的底层预训练判别网络和顶层生成性分类网络，该网络通过深度潜变量模型对对抗性输入分布进行建模。使用变分贝叶斯，我们的模型在不需要对抗性训练的情况下，对白盒对抗性攻击获得了优越的稳健性。在CIFAR-10和CIFAR-100上的大量实验证明了我们的模型具有优越的对抗鲁棒性。通过使用反事实度量和基于特征交互的度量进行评估，我们建立了模型可解释性和对抗健壮性之间的关联。此外，在Tiny-ImageNet上的初步结果验证了我们的方法对更复杂的数据集的可扩展性，为开发健壮的图像分类模型提供了一个实用的解决方案。



## **17. Adversarial Robustness for Deep Learning-based Wildfire Detection Models**

基于深度学习的野火检测模型的对抗鲁棒性 cs.CV

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2412.20006v1) [paper-pdf](http://arxiv.org/pdf/2412.20006v1)

**Authors**: Ryo Ide, Lei Yang

**Abstract**: Smoke detection using Deep Neural Networks (DNNs) is an effective approach for early wildfire detection. However, because smoke is temporally and spatially anomalous, there are limitations in collecting sufficient training data. This raises overfitting and bias concerns in existing DNN-based wildfire detection models. Thus, we introduce WARP (Wildfire Adversarial Robustness Procedure), the first model-agnostic framework for evaluating the adversarial robustness of DNN-based wildfire detection models. WARP addresses limitations in smoke image diversity using global and local adversarial attack methods. The global attack method uses image-contextualized Gaussian noise, while the local attack method uses patch noise injection, tailored to address critical aspects of wildfire detection. Leveraging WARP's model-agnostic capabilities, we assess the adversarial robustness of real-time Convolutional Neural Networks (CNNs) and Transformers. The analysis revealed valuable insights into the models' limitations. Specifically, the global attack method demonstrates that the Transformer model has more than 70\% precision degradation than the CNN against global noise. In contrast, the local attack method shows that both models are susceptible to cloud image injections when detecting smoke-positive instances, suggesting a need for model improvements through data augmentation. WARP's comprehensive robustness analysis contributed to the development of wildfire-specific data augmentation strategies, marking a step toward practicality.

摘要: 基于深度神经网络的烟雾检测是野火早期检测的一种有效方法。然而，由于烟雾在时间和空间上都是反常的，收集足够的训练数据是有局限性的。这在现有的基于DNN的野火检测模型中引发了过度拟合和偏差的担忧。因此，我们引入了WARP(Wildfire对抗稳健性过程)，这是第一个模型不可知的框架，用于评估基于DNN的野火检测模型的对抗稳健性。WARP使用全球和局部对抗性攻击方法解决烟雾图像多样性方面的限制。全局攻击方法使用与图像相关的高斯噪声，而局部攻击方法使用补丁噪声注入，该方法针对野火检测的关键方面进行了量身定做。利用WARP的模型不可知能力，我们评估了实时卷积神经网络(CNN)和变形金刚的对抗健壮性。分析揭示了对模型局限性的有价值的见解。具体来说，全局攻击方法表明，Transformer模型在抗全局噪声方面比CNN模型有70%以上的精度下降。相比之下，本地攻击方法表明，当检测到烟雾阳性实例时，这两个模型都容易受到云图注入的影响，这表明需要通过数据增强来改进模型。WARP的全面稳健性分析有助于开发特定于野火的数据增强策略，标志着朝着实用化迈出了一步。



## **18. Standard-Deviation-Inspired Regularization for Improving Adversarial Robustness**

标准偏差启发的规范化提高对抗稳健性 cs.LG

**SubmitDate**: 2024-12-27    [abs](http://arxiv.org/abs/2412.19947v1) [paper-pdf](http://arxiv.org/pdf/2412.19947v1)

**Authors**: Olukorede Fakorede, Modeste Atsague, Jin Tian

**Abstract**: Adversarial Training (AT) has been demonstrated to improve the robustness of deep neural networks (DNNs) against adversarial attacks. AT is a min-max optimization procedure where in adversarial examples are generated to train a more robust DNN. The inner maximization step of AT increases the losses of inputs with respect to their actual classes. The outer minimization involves minimizing the losses on the adversarial examples obtained from the inner maximization. This work proposes a standard-deviation-inspired (SDI) regularization term to improve adversarial robustness and generalization. We argue that the inner maximization in AT is similar to minimizing a modified standard deviation of the model's output probabilities. Moreover, we suggest that maximizing this modified standard deviation can complement the outer minimization of the AT framework. To support our argument, we experimentally show that the SDI measure can be used to craft adversarial examples. Additionally, we demonstrate that combining the SDI regularization term with existing AT variants enhances the robustness of DNNs against stronger attacks, such as CW and Auto-attack, and improves generalization.

摘要: 对抗训练(AT)已被证明可以提高深度神经网络(DNN)对对抗攻击的稳健性。AT是一种最小-最大优化过程，其中在对抗性例子中生成训练更健壮的DNN。AT的内部最大化步骤增加了输入相对于其实际类别的损失。外极小化包括最小化由内极大化得到的对抗性例子的损失。该工作提出了一种标准差启发(SDI)正则化项来提高对手的稳健性和泛化能力。我们认为AT中的内极大化类似于最小化模型输出概率的修正标准差。此外，我们认为最大化这个修正的标准差可以补充AT框架的外极小化。为了支持我们的论点，我们通过实验证明SDI测量可以用来制作对抗性的例子。此外，我们还证明了将SDI正则化项与现有的AT变体相结合，增强了DNN对更强的攻击(如CW和Auto-Attack)的健壮性，并提高了泛化能力。



## **19. A High Dimensional Statistical Model for Adversarial Training: Geometry and Trade-Offs**

对抗训练的多维统计模型：几何结构和权衡 stat.ML

**SubmitDate**: 2024-12-27    [abs](http://arxiv.org/abs/2402.05674v3) [paper-pdf](http://arxiv.org/pdf/2402.05674v3)

**Authors**: Kasimir Tanner, Matteo Vilucchio, Bruno Loureiro, Florent Krzakala

**Abstract**: This work investigates adversarial training in the context of margin-based linear classifiers in the high-dimensional regime where the dimension $d$ and the number of data points $n$ diverge with a fixed ratio $\alpha = n / d$. We introduce a tractable mathematical model where the interplay between the data and adversarial attacker geometries can be studied, while capturing the core phenomenology observed in the adversarial robustness literature. Our main theoretical contribution is an exact asymptotic description of the sufficient statistics for the adversarial empirical risk minimiser, under generic convex and non-increasing losses for a Block Feature Model. Our result allow us to precisely characterise which directions in the data are associated with a higher generalisation/robustness trade-off, as defined by a robustness and a usefulness metric. We show that the the presence of multiple different feature types is crucial to the high sample complexity performances of adversarial training. In particular, we unveil the existence of directions which can be defended without penalising accuracy. Finally, we show the advantage of defending non-robust features during training, identifying a uniform protection as an inherently effective defence mechanism.

摘要: 该工作研究了高维环境下基于差值的线性分类器的对抗性训练，其中维度$d$和数据点数目$n$以固定的比率$\α=n/d$发散。我们引入了一个易于处理的数学模型，其中可以研究数据和敌意攻击者几何之间的相互作用，同时捕获在对抗性健壮性文献中观察到的核心现象学。我们的主要理论贡献是在块特征模型的一般凸和非增加损失下，给出了对抗性经验风险最小化充分统计量的精确渐近描述。我们的结果使我们能够准确地描述数据中的哪些方向与更高的泛化/稳健性权衡相关，如稳健性和有用性度量所定义的那样。结果表明，多个不同特征类型的存在对对抗性训练的高样本复杂度性能至关重要。特别是，我们揭示了方向的存在，这些方向可以在不影响准确性的情况下得到辩护。最后，我们展示了在训练过程中防御非健壮特征的优势，确定了统一保护作为一种内在有效的防御机制。



## **20. Enhancing Adversarial Robustness of Deep Neural Networks Through Supervised Contrastive Learning**

通过监督对比学习增强深度神经网络的对抗鲁棒性 cs.LG

8 pages, 11 figures

**SubmitDate**: 2024-12-27    [abs](http://arxiv.org/abs/2412.19747v1) [paper-pdf](http://arxiv.org/pdf/2412.19747v1)

**Authors**: Longwei Wang, Navid Nayyem, Abdullah Rakin

**Abstract**: Adversarial attacks exploit the vulnerabilities of convolutional neural networks by introducing imperceptible perturbations that lead to misclassifications, exposing weaknesses in feature representations and decision boundaries. This paper presents a novel framework combining supervised contrastive learning and margin-based contrastive loss to enhance adversarial robustness. Supervised contrastive learning improves the structure of the feature space by clustering embeddings of samples within the same class and separating those from different classes. Margin-based contrastive loss, inspired by support vector machines, enforces explicit constraints to create robust decision boundaries with well-defined margins. Experiments on the CIFAR-100 dataset with a ResNet-18 backbone demonstrate robustness performance improvements in adversarial accuracy under Fast Gradient Sign Method attacks.

摘要: 对抗性攻击通过引入难以感知的扰动来利用卷积神经网络的漏洞，从而导致错误分类，暴露特征表示和决策边界的弱点。本文提出了一种新颖的框架，将监督对比学习和基于边缘的对比损失相结合，以增强对抗鲁棒性。监督对比学习通过对同一类内的样本嵌入进行聚集并将来自不同类的样本嵌入分离，来改善特征空间的结构。基于利润的对比损失受支持向量机的启发，强制执行显式约束，以创建具有明确定义利润的稳健决策边界。在具有ResNet-18主干的CIFAR-100数据集上进行的实验表明，在快速梯度符号法攻击下，对抗准确性的鲁棒性性能有所提高。



## **21. Gröbner Basis Cryptanalysis of Ciminion and Hydra**

格罗布纳基础对西米尼恩和海德拉的密码分析 cs.CR

**SubmitDate**: 2024-12-27    [abs](http://arxiv.org/abs/2405.05040v3) [paper-pdf](http://arxiv.org/pdf/2405.05040v3)

**Authors**: Matthias Johann Steiner

**Abstract**: Ciminion and Hydra are two recently introduced symmetric key Pseudo-Random Functions for Multi-Party Computation applications. For efficiency both primitives utilize quadratic permutations at round level. Therefore, polynomial system solving-based attacks pose a serious threat to these primitives. For Ciminion, we construct a quadratic degree reverse lexicographic (DRL) Gr\"obner basis for the iterated polynomial model via linear transformations. With the Gr\"obner basis we can simplify cryptanalysis since we do not need to impose genericity assumptions anymore to derive complexity estimations. For Hydra, with the help of a computer algebra program like SageMath we construct a DRL Gr\"obner basis for the iterated model via linear transformations and a linear change of coordinates. In the Hydra proposal it was claimed that $r_\mathcal{H} = 31$ rounds are sufficient to provide $128$ bits of security against Gr\"obner basis attacks for an ideal adversary with $\omega = 2$. However, via our Hydra Gr\"obner basis standard term order conversion to a lexicographic (LEX) Gr\"obner basis requires just $126$ bits with $\omega = 2$. Moreover, via a dedicated polynomial system solving technique up to $r_\mathcal{H} = 33$ rounds can be attacked below $128$ bits for an ideal adversary.

摘要: Ciminion和Hydra是最近推出的两个用于多方计算应用的对称密钥伪随机函数。为了提高效率，两个基元都在循环水平上使用二次置换。因此，基于多项式系统求解的攻击对这些原语构成了严重威胁。对于Ciminion，我们通过线性变换为迭代多项式模型构造了一个二次逆词典(DRL)Gr‘obner基，利用这个Gr’obner基，我们可以简化密码分析，因为我们不再需要强加一般性假设来推导复杂性估计。对于Hydra，借助于SageMath这样的计算机代数程序，我们通过线性变换和线性坐标变化，为迭代模型构造了一个DRL Grobner基.在Hydra的方案中，声称$r_\mathcal{H}=31$轮足以为$omega=2$的理想对手提供$128$bit的Gr‘obner基攻击安全.然而，通过我们的Hydra Gr\“obner基础”将标准术语顺序转换为词典(Lex)Gr\“obner基础只需要$126$位，且$\omega=2$。此外，通过一种专门的多项式系统求解技术，高达$r_\数学{H}=33$的轮数可以被攻击到低于$128$比特的理想对手。



## **22. Attribution for Enhanced Explanation with Transferable Adversarial eXploration**

可转移对抗性探索增强解释的归因 cs.AI

**SubmitDate**: 2024-12-27    [abs](http://arxiv.org/abs/2412.19523v1) [paper-pdf](http://arxiv.org/pdf/2412.19523v1)

**Authors**: Zhiyu Zhu, Jiayu Zhang, Zhibo Jin, Huaming Chen, Jianlong Zhou, Fang Chen

**Abstract**: The interpretability of deep neural networks is crucial for understanding model decisions in various applications, including computer vision. AttEXplore++, an advanced framework built upon AttEXplore, enhances attribution by incorporating transferable adversarial attack methods such as MIG and GRA, significantly improving the accuracy and robustness of model explanations. We conduct extensive experiments on five models, including CNNs (Inception-v3, ResNet-50, VGG16) and vision transformers (MaxViT-T, ViT-B/16), using the ImageNet dataset. Our method achieves an average performance improvement of 7.57\% over AttEXplore and 32.62\% compared to other state-of-the-art interpretability algorithms. Using insertion and deletion scores as evaluation metrics, we show that adversarial transferability plays a vital role in enhancing attribution results. Furthermore, we explore the impact of randomness, perturbation rate, noise amplitude, and diversity probability on attribution performance, demonstrating that AttEXplore++ provides more stable and reliable explanations across various models. We release our code at: https://anonymous.4open.science/r/ATTEXPLOREP-8435/

摘要: 深度神经网络的可解释性对于理解包括计算机视觉在内的各种应用中的模型决策至关重要。AttEXplore++是建立在AttEXplore基础上的高级框架，通过整合可转移的对抗性攻击方法(如MIG和GRA)来增强属性，显著提高模型解释的准确性和健壮性。我们使用ImageNet数据集，在五个模型上进行了广泛的实验，包括CNN(初始-v3，ResNet-50，VGG16)和视觉转换器(MaxViT-T，Vit-B/16)。与AttEXplore算法相比，该方法的性能平均提高了7.57倍，与其他最先进的可解释性算法相比，平均性能提高了32.62倍。以插入和删除分数作为评价指标，我们发现对抗性转移对提高归因结果起着至关重要的作用。此外，我们探讨了随机性、扰动率、噪声幅度和多样性概率对归因性能的影响，证明了AttEXplore++在各种模型中提供了更稳定和可靠的解释。我们的代码发布地址为：https://anonymous.4open.science/r/ATTEXPLOREP-8435/



## **23. An Engorgio Prompt Makes Large Language Model Babble on**

Engorgio提示让大型语言模型胡言乱语 cs.CR

**SubmitDate**: 2024-12-27    [abs](http://arxiv.org/abs/2412.19394v1) [paper-pdf](http://arxiv.org/pdf/2412.19394v1)

**Authors**: Jianshuo Dong, Ziyuan Zhang, Qingjie Zhang, Han Qiu, Tianwei Zhang, Hao Wang, Hewu Li, Qi Li, Chao Zhang, Ke Xu

**Abstract**: Auto-regressive large language models (LLMs) have yielded impressive performance in many real-world tasks. However, the new paradigm of these LLMs also exposes novel threats. In this paper, we explore their vulnerability to inference cost attacks, where a malicious user crafts Engorgio prompts to intentionally increase the computation cost and latency of the inference process. We design Engorgio, a novel methodology, to efficiently generate adversarial Engorgio prompts to affect the target LLM's service availability. Engorgio has the following two technical contributions. (1) We employ a parameterized distribution to track LLMs' prediction trajectory. (2) Targeting the auto-regressive nature of LLMs' inference process, we propose novel loss functions to stably suppress the appearance of the <EOS> token, whose occurrence will interrupt the LLM's generation process. We conduct extensive experiments on 13 open-sourced LLMs with parameters ranging from 125M to 30B. The results show that Engorgio prompts can successfully induce LLMs to generate abnormally long outputs (i.e., roughly 2-13$\times$ longer to reach 90%+ of the output length limit) in a white-box scenario and our real-world experiment demonstrates Engergio's threat to LLM service with limited computing resources. The code is accessible at https://github.com/jianshuod/Engorgio-prompt.

摘要: 自回归大型语言模型(LLM)在许多实际任务中取得了令人印象深刻的性能。然而，这些LLM的新范式也暴露了新的威胁。在本文中，我们探讨了它们对推理成本攻击的脆弱性，在这种攻击中，恶意用户手工制作Engorgio会提示故意增加推理过程的计算成本和延迟。我们设计了一种新的方法Engorgio来有效地生成对抗性Engorgio提示，以影响目标LLM的服务可用性。Engorgio有以下两个技术贡献。(1)采用一种参数分布来跟踪LLMS的预测轨迹。(2)针对LLMS推理过程的自回归特性，提出了一种新的损失函数来稳定地抑制<EOS>令牌的出现，它的出现会中断LLM的生成过程。我们在13个开源的LLM上进行了广泛的实验，参数从125M到30B不等。结果表明，在白盒场景中，Engorgio提示能够成功地诱导LLMS产生异常长的输出(即大约2-13$\x$以达到输出长度限制的90%以上)，并且我们的真实世界实验证明了Engergio在有限计算资源的情况下对LLM服务的威胁。该代码可在https://github.com/jianshuod/Engorgio-prompt.上访问



## **24. Quantum-Inspired Weight-Constrained Neural Network: Reducing Variable Numbers by 100x Compared to Standard Neural Networks**

量子启发的权重约束神经网络：与标准神经网络相比将变量数减少100倍 quant-ph

13 pages, 5 figures. Comments are welcome

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19355v1) [paper-pdf](http://arxiv.org/pdf/2412.19355v1)

**Authors**: Shaozhi Li, M Sabbir Salek, Binayyak Roy, Yao Wang, Mashrur Chowdhury

**Abstract**: Although quantum machine learning has shown great promise, the practical application of quantum computers remains constrained in the noisy intermediate-scale quantum era. To take advantage of quantum machine learning, we investigate the underlying mathematical principles of these quantum models and adapt them to classical machine learning frameworks. Specifically, we develop a classical weight-constrained neural network that generates weights based on quantum-inspired insights. We find that this approach can reduce the number of variables in a classical neural network by a factor of 135 while preserving its learnability. In addition, we develop a dropout method to enhance the robustness of quantum machine learning models, which are highly susceptible to adversarial attacks. This technique can also be applied to improve the adversarial resilience of the classical weight-constrained neural network, which is essential for industry applications, such as self-driving vehicles. Our work offers a novel approach to reduce the complexity of large classical neural networks, addressing a critical challenge in machine learning.

摘要: 尽管量子机器学习显示出了巨大的前景，但在嘈杂的中等规模量子时代，量子计算机的实际应用仍然受到限制。为了利用量子机器学习的优势，我们研究了这些量子模型的基本数学原理，并将它们适应于经典的机器学习框架。具体地说，我们开发了一个经典的权重约束神经网络，它基于量子启发的见解生成权重。我们发现，这种方法可以将经典神经网络中的变量数量减少135倍，同时保持其可学习性。此外，我们开发了一种丢弃方法来增强量子机器学习模型的健壮性，这些模型对对手攻击非常敏感。该技术还可以用于提高经典的权值约束神经网络的对抗能力，这对于自动驾驶汽车等工业应用是必不可少的。我们的工作提供了一种新的方法来降低大型经典神经网络的复杂性，解决了机器学习中的一个关键挑战。



## **25. Federated Hybrid Training and Self-Adversarial Distillation: Towards Robust Edge Networks**

联合混合训练和自对抗蒸馏：迈向稳健的边缘网络 cs.CV

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19354v1) [paper-pdf](http://arxiv.org/pdf/2412.19354v1)

**Authors**: Yu Qiao, Apurba Adhikary, Kitae Kim, Eui-Nam Huh, Zhu Han, Choong Seon Hong

**Abstract**: Federated learning (FL) is a distributed training technology that enhances data privacy in mobile edge networks by allowing data owners to collaborate without transmitting raw data to the edge server. However, data heterogeneity and adversarial attacks pose challenges to develop an unbiased and robust global model for edge deployment. To address this, we propose Federated hyBrid Adversarial training and self-adversarial disTillation (FedBAT), a new framework designed to improve both robustness and generalization of the global model. FedBAT seamlessly integrates hybrid adversarial training and self-adversarial distillation into the conventional FL framework from data augmentation and feature distillation perspectives. From a data augmentation perspective, we propose hybrid adversarial training to defend against adversarial attacks by balancing accuracy and robustness through a weighted combination of standard and adversarial training. From a feature distillation perspective, we introduce a novel augmentation-invariant adversarial distillation method that aligns local adversarial features of augmented images with their corresponding unbiased global clean features. This alignment can effectively mitigate bias from data heterogeneity while enhancing both the robustness and generalization of the global model. Extensive experimental results across multiple datasets demonstrate that FedBAT yields comparable or superior performance gains in improving robustness while maintaining accuracy compared to several baselines.

摘要: 联合学习(FL)是一种分布式训练技术，它允许数据所有者在不向边缘服务器传输原始数据的情况下进行协作，从而增强移动边缘网络中的数据隐私。然而，数据异构性和对抗性攻击给开发无偏见和健壮的全球EDGE部署模型带来了挑战。为了解决这一问题，我们提出了联邦混合对抗训练和自我对抗蒸馏(FedBAT)，这是一个新的框架，旨在提高全局模型的健壮性和泛化能力。FedBAT从数据增强和特征提取的角度，将混合对抗性训练和自我对抗性提炼无缝地集成到传统的FL框架中。从数据增强的角度，我们提出了混合对抗性训练，通过标准训练和对抗性训练的加权组合来平衡精确度和稳健性来防御对抗性攻击。从特征提取的角度，我们提出了一种新的增强不变对抗提取方法，该方法将增强图像的局部对抗特征与其相应的无偏全局清洁特征对齐。这种对齐可以有效地减少数据异质性带来的偏差，同时增强全局模型的稳健性和泛化能力。在多个数据集上的广泛实验结果表明，与几个基线相比，FedBAT在提高稳健性同时保持准确性方面获得了类似或更好的性能收益。



## **26. xSRL: Safety-Aware Explainable Reinforcement Learning -- Safety as a Product of Explainability**

xSRL：安全意识的可解释强化学习--安全作为可解释性的产物 cs.AI

Accepted to 24th International Conference on Autonomous Agents and  Multiagent Systems (AAMAS 2025)

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19311v1) [paper-pdf](http://arxiv.org/pdf/2412.19311v1)

**Authors**: Risal Shahriar Shefin, Md Asifur Rahman, Thai Le, Sarra Alqahtani

**Abstract**: Reinforcement learning (RL) has shown great promise in simulated environments, such as games, where failures have minimal consequences. However, the deployment of RL agents in real-world systems such as autonomous vehicles, robotics, UAVs, and medical devices demands a higher level of safety and transparency, particularly when facing adversarial threats. Safe RL algorithms have been developed to address these concerns by optimizing both task performance and safety constraints. However, errors are inevitable, and when they occur, it is essential that the RL agents can also explain their actions to human operators. This makes trust in the safety mechanisms of RL systems crucial for effective deployment. Explainability plays a key role in building this trust by providing clear, actionable insights into the agent's decision-making process, ensuring that safety-critical decisions are well understood. While machine learning (ML) has seen significant advances in interpretability and visualization, explainability methods for RL remain limited. Current tools fail to address the dynamic, sequential nature of RL and its needs to balance task performance with safety constraints over time. The re-purposing of traditional ML methods, such as saliency maps, is inadequate for safety-critical RL applications where mistakes can result in severe consequences. To bridge this gap, we propose xSRL, a framework that integrates both local and global explanations to provide a comprehensive understanding of RL agents' behavior. xSRL also enables developers to identify policy vulnerabilities through adversarial attacks, offering tools to debug and patch agents without retraining. Our experiments and user studies demonstrate xSRL's effectiveness in increasing safety in RL systems, making them more reliable and trustworthy for real-world deployment. Code is available at https://github.com/risal-shefin/xSRL.

摘要: 强化学习(RL)在模拟环境中显示了巨大的前景，例如游戏，在这些环境中，失败的后果最小。然而，在自动驾驶车辆、机器人、无人机和医疗设备等现实世界系统中部署RL代理需要更高水平的安全性和透明度，特别是在面临对手威胁的情况下。安全RL算法已经被开发出来，通过优化任务性能和安全约束来解决这些问题。然而，错误是不可避免的，当它们发生时，RL代理也可以向人类操作员解释他们的行为是至关重要的。这使得对RL系统安全机制的信任对有效部署至关重要。可解释性在建立这种信任方面发挥了关键作用，它为代理人的决策过程提供了清晰、可操作的见解，确保了对安全至关重要的决策得到很好的理解。虽然机器学习(ML)在可解释性和可视化方面取得了重大进展，但用于RL的可解释性方法仍然有限。目前的工具不能解决RL的动态、连续的性质，以及它需要随着时间的推移平衡任务性能和安全约束。传统ML方法的再利用，如显著图，对于安全关键的RL应用是不够的，因为错误可能会导致严重的后果。为了弥合这一差距，我们提出了xSRL，一个整合了局部和全局解释的框架，以提供对RL代理行为的全面理解。XSRL还使开发人员能够通过对抗性攻击识别策略漏洞，提供工具来调试和修补代理，而无需重新培训。我们的实验和用户研究证明了xSRL在提高RL系统安全性方面的有效性，使它们在现实世界的部署中更加可靠和值得信赖。代码可在https://github.com/risal-shefin/xSRL.上找到



## **27. Game-Theoretically Secure Distributed Protocols for Fair Allocation in Coalitional Games**

联盟游戏中公平分配的游戏理论安全分布式协议 cs.GT

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19192v1) [paper-pdf](http://arxiv.org/pdf/2412.19192v1)

**Authors**: T-H. Hubert Chan, Qipeng Kuang, Quan Xue

**Abstract**: We consider game-theoretically secure distributed protocols for coalition games that approximate the Shapley value with small multiplicative error. Since all known existing approximation algorithms for the Shapley value are randomized, it is a challenge to design efficient distributed protocols among mutually distrusted players when there is no central authority to generate unbiased randomness. The game-theoretic notion of maximin security has been proposed to offer guarantees to an honest player's reward even if all other players are susceptible to an adversary.   Permutation sampling is often used in approximation algorithms for the Shapley value. A previous work in 1994 by Zlotkin et al. proposed a simple constant-round distributed permutation generation protocol based on commitment scheme, but it is vulnerable to rushing attacks. The protocol, however, can detect such attacks.   In this work, we model the limited resources of an adversary by a violation budget that determines how many times it can perform such detectable attacks. Therefore, by repeating the number of permutation samples, an honest player's reward can be guaranteed to be close to its Shapley value. We explore both high probability and expected maximin security. We obtain an upper bound on the number of permutation samples for high probability maximin security, even with an unknown violation budget. Furthermore, we establish a matching lower bound for the weaker notion of expected maximin security in specific permutation generation protocols. We have also performed experiments on both synthetic and real data to empirically verify our results.

摘要: 我们考虑在小乘法误差下近似Shapley值的联盟博弈的博弈论安全分布式协议。由于所有已知的Shapley值的近似算法都是随机化的，在没有中央权威机构来产生无偏随机性的情况下，在相互不信任的参与者之间设计有效的分布式协议是一个挑战。博弈论的最大限度安全的概念被提出，以保证诚实的玩家的回报，即使所有其他玩家都容易受到对手的影响。在Shapley值的近似算法中，通常使用置换采样。Zlotkin等人在1994年进行的前一项工作。提出了一种简单的基于承诺方案的恒轮分布式置换生成协议，但该协议容易受到冲刺攻击。然而，该协议可以检测到此类攻击。在这项工作中，我们通过违规预算来模拟对手的有限资源，该预算决定了对手可以执行这种可检测到的攻击的次数。因此，通过重复排列样本的数量，可以保证诚实玩家的奖励接近其Shapley值。我们探讨了高概率安全性和期望最大安全性。我们得到了高概率最大化安全性的置换样本数目的上界，即使在未知的违规预算下也是如此。此外，对于特定置换生成协议中较弱的期望最大安全性概念，我们建立了匹配的下界。我们还在合成数据和真实数据上进行了实验，以经验地验证我们的结果。



## **28. TSCheater: Generating High-Quality Tibetan Adversarial Texts via Visual Similarity**

TSCheater：通过视觉相似性生成高质量的西藏对抗文本 cs.CL

Camera-Ready Version; Accepted at ICASSP 2025

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.02371v3) [paper-pdf](http://arxiv.org/pdf/2412.02371v3)

**Authors**: Xi Cao, Quzong Gesang, Yuan Sun, Nuo Qun, Tashi Nyima

**Abstract**: Language models based on deep neural networks are vulnerable to textual adversarial attacks. While rich-resource languages like English are receiving focused attention, Tibetan, a cross-border language, is gradually being studied due to its abundant ancient literature and critical language strategy. Currently, there are several Tibetan adversarial text generation methods, but they do not fully consider the textual features of Tibetan script and overestimate the quality of generated adversarial texts. To address this issue, we propose a novel Tibetan adversarial text generation method called TSCheater, which considers the characteristic of Tibetan encoding and the feature that visually similar syllables have similar semantics. This method can also be transferred to other abugidas, such as Devanagari script. We utilize a self-constructed Tibetan syllable visual similarity database called TSVSDB to generate substitution candidates and adopt a greedy algorithm-based scoring mechanism to determine substitution order. After that, we conduct the method on eight victim language models. Experimentally, TSCheater outperforms existing methods in attack effectiveness, perturbation magnitude, semantic similarity, visual similarity, and human acceptance. Finally, we construct the first Tibetan adversarial robustness evaluation benchmark called AdvTS, which is generated by existing methods and proofread by humans.

摘要: 基于深度神经网络的语言模型容易受到文本攻击。在英语等资源丰富的语言受到关注的同时，藏语这一跨境语言也因其丰富的古代文献和批评的语言策略而逐渐被研究。目前，有几种藏文对抗性文本生成方法，但它们没有充分考虑藏文的文本特征，高估了生成的对抗性文本的质量。针对这一问题，我们提出了一种新的藏文对抗性文本生成方法TSCheater，该方法考虑了藏文编码的特点和视觉上相似音节具有相似语义的特点。这种方法也可以移植到其他ABUGIDAS，如天成文书。利用自行构建的藏文音节视觉相似度数据库TSVSDB生成替换候选，并采用基于贪婪算法的评分机制确定替换顺序。之后，我们在八个受害者语言模型上进行了该方法。实验结果表明，TSCheater在攻击效果、扰动幅度、语义相似度、视觉相似度和人类接受度等方面均优于现有方法。最后，我们构建了第一个藏文对手健壮性评估基准ADVTS，该基准由现有方法生成并由人工校对。



## **29. DiffPatch: Generating Customizable Adversarial Patches using Diffusion Model**

迪夫补丁：使用扩散模型生成可定制的对抗补丁 cs.CV

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.01440v2) [paper-pdf](http://arxiv.org/pdf/2412.01440v2)

**Authors**: Zhixiang Wang, Guangnan Ye, Xiaosen Wang, Siheng Chen, Zhibo Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Physical adversarial patches printed on clothing can easily allow individuals to evade person detectors. However, most existing adversarial patch generation methods prioritize attack effectiveness over stealthiness, resulting in patches that are aesthetically unpleasing. Although existing methods using generative adversarial networks or diffusion models can produce more natural-looking patches, they often struggle to balance stealthiness with attack effectiveness and lack flexibility for user customization. To address these challenges, we propose a novel diffusion-based customizable patch generation framework termed DiffPatch, specifically tailored for creating naturalistic and customizable adversarial patches. Our approach enables users to utilize a reference image as the source, rather than starting from random noise, and incorporates masks to craft naturalistic patches of various shapes, not limited to squares. To prevent the original semantics from being lost during the diffusion process, we employ Null-text inversion to map random noise samples to a single input image and generate patches through Incomplete Diffusion Optimization (IDO). Notably, while maintaining a natural appearance, our method achieves a comparable attack performance to state-of-the-art non-naturalistic patches when using similarly sized attacks. Using DiffPatch, we have created a physical adversarial T-shirt dataset, AdvPatch-1K, specifically targeting YOLOv5s. This dataset includes over a thousand images across diverse scenarios, validating the effectiveness of our attack in real-world environments. Moreover, it provides a valuable resource for future research.

摘要: 衣服上印有敌意的物理补丁可以很容易地让个人躲避个人探测器。然而，大多数现有的对抗性补丁生成方法将攻击效率置于隐蔽性之上，导致生成的补丁在美学上令人不快。虽然现有的方法使用生成性对抗网络或扩散模型可以产生看起来更自然的补丁，但它们往往难以平衡隐蔽性和攻击有效性，并且缺乏用户定制的灵活性。为了应对这些挑战，我们提出了一种新的基于扩散的可定制补丁生成框架DiffPatch，该框架专门用于创建自然的和可定制的对抗性补丁。我们的方法使用户能够利用参考图像作为源，而不是从随机噪声开始，并结合蒙版来制作各种形状的自然斑块，而不限于正方形。为了避免在扩散过程中丢失原始语义，我们使用空文本反转将随机噪声样本映射到单一输入图像，并通过不完全扩散优化(IDO)生成斑块。值得注意的是，在保持自然外观的同时，我们的方法在使用类似大小的攻击时，实现了与最先进的非自然主义补丁相当的攻击性能。使用DiffPatch，我们已经创建了一个物理对手T恤数据集AdvPatch-1K，专门针对YOLOv5。该数据集包括1000多张不同场景的图像，验证了我们的攻击在真实环境中的有效性。此外，它还为今后的研究提供了宝贵的资源。



## **30. Provable Robust Saliency-based Explanations**

可证明的稳健基于显着性的解释 cs.LG

Accepted to NeurIPS 2024

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2212.14106v4) [paper-pdf](http://arxiv.org/pdf/2212.14106v4)

**Authors**: Chao Chen, Chenghua Guo, Rufeng Chen, Guixiang Ma, Ming Zeng, Xiangwen Liao, Xi Zhang, Sihong Xie

**Abstract**: To foster trust in machine learning models, explanations must be faithful and stable for consistent insights. Existing relevant works rely on the $\ell_p$ distance for stability assessment, which diverges from human perception. Besides, existing adversarial training (AT) associated with intensive computations may lead to an arms race. To address these challenges, we introduce a novel metric to assess the stability of top-$k$ salient features. We introduce R2ET which trains for stable explanation by efficient and effective regularizer, and analyze R2ET by multi-objective optimization to prove numerical and statistical stability of explanations. Moreover, theoretical connections between R2ET and certified robustness justify R2ET's stability in all attacks. Extensive experiments across various data modalities and model architectures show that R2ET achieves superior stability against stealthy attacks, and generalizes effectively across different explanation methods.

摘要: 为了促进对机器学习模型的信任，解释必须忠实且稳定，以获得一致的见解。现有的相关作品依赖于$\ell_p$距离进行稳定性评估，这与人类的感知存在分歧。此外，与密集计算相关的现有对抗训练（AT）可能会导致军备竞赛。为了应对这些挑战，我们引入了一种新颖的指标来评估顶级$k$显着特征的稳定性。我们引入R2 ET，通过高效且有效的正规化器训练稳定的解释，并通过多目标优化分析R2 ET，以证明解释的数字和统计稳定性。此外，R2 ET和认证稳健性之间的理论联系证明了R2 ET在所有攻击中的稳定性。跨各种数据模式和模型架构的广泛实验表明，R2 ET针对隐形攻击实现了卓越的稳定性，并在不同的解释方法中有效推广。



## **31. Imperceptible Adversarial Attacks on Point Clouds Guided by Point-to-Surface Field**

点到表面场引导下的点云不可感知的对抗攻击 cs.CV

Accepted by ICASSP 2025

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19015v1) [paper-pdf](http://arxiv.org/pdf/2412.19015v1)

**Authors**: Keke Tang, Weiyao Ke, Weilong Peng, Xiaofei Wang, Ziyong Du, Zhize Wu, Peican Zhu, Zhihong Tian

**Abstract**: Adversarial attacks on point clouds are crucial for assessing and improving the adversarial robustness of 3D deep learning models. Traditional solutions strictly limit point displacement during attacks, making it challenging to balance imperceptibility with adversarial effectiveness. In this paper, we attribute the inadequate imperceptibility of adversarial attacks on point clouds to deviations from the underlying surface. To address this, we introduce a novel point-to-surface (P2S) field that adjusts adversarial perturbation directions by dragging points back to their original underlying surface. Specifically, we use a denoising network to learn the gradient field of the logarithmic density function encoding the shape's surface, and apply a distance-aware adjustment to perturbation directions during attacks, thereby enhancing imperceptibility. Extensive experiments show that adversarial attacks guided by our P2S field are more imperceptible, outperforming state-of-the-art methods.

摘要: 对点云的对抗攻击对于评估和提高3D深度学习模型的对抗稳健性至关重要。传统的解决方案严格限制攻击期间的点位移，使得平衡不可感知性与对抗有效性变得具有挑战性。在本文中，我们将点云对抗攻击的不可感知性不足归因于与底层表面的偏差。为了解决这个问题，我们引入了一种新型的点到面（P2 S）场，该场通过将点拖回其原始底层表面来调整对抗扰动方向。具体来说，我们使用去噪网络来学习编码形状表面的log密度函数的梯度场，并在攻击期间对扰动方向进行距离感知调整，从而增强不可感知性。大量实验表明，由我们的P2S领域引导的对抗攻击更难以察觉，性能优于最先进的方法。



## **32. Bridging Interpretability and Robustness Using LIME-Guided Model Refinement**

使用LIME引导的模型细化来弥合可解释性和鲁棒性 cs.LG

10 pages, 15 figures

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18952v1) [paper-pdf](http://arxiv.org/pdf/2412.18952v1)

**Authors**: Navid Nayyem, Abdullah Rakin, Longwei Wang

**Abstract**: This paper explores the intricate relationship between interpretability and robustness in deep learning models. Despite their remarkable performance across various tasks, deep learning models often exhibit critical vulnerabilities, including susceptibility to adversarial attacks, over-reliance on spurious correlations, and a lack of transparency in their decision-making processes. To address these limitations, we propose a novel framework that leverages Local Interpretable Model-Agnostic Explanations (LIME) to systematically enhance model robustness. By identifying and mitigating the influence of irrelevant or misleading features, our approach iteratively refines the model, penalizing reliance on these features during training. Empirical evaluations on multiple benchmark datasets demonstrate that LIME-guided refinement not only improves interpretability but also significantly enhances resistance to adversarial perturbations and generalization to out-of-distribution data.

摘要: 本文探讨了深度学习模型中可解释性和稳健性之间的复杂关系。尽管深度学习模型在各种任务中表现出色，但它们往往表现出严重的漏洞，包括容易受到对抗攻击、过度依赖虚假相关性以及决策过程缺乏透明度。为了解决这些限制，我们提出了一种新颖的框架，该框架利用本地可解释模型不可知解释（LIME）来系统性地增强模型稳健性。通过识别和减轻不相关或误导性特征的影响，我们的方法迭代地完善模型，惩罚训练期间对这些特征的依赖。对多个基准数据集的经验评估表明，LIME引导的细化不仅提高了可解释性，而且显着增强了对对抗性扰动的抵抗力和对非分布数据的概括。



## **33. Improving Integrated Gradient-based Transferable Adversarial Examples by Refining the Integration Path**

通过完善集成路径改进基于集成对象的可转移对抗示例 cs.CR

Accepted by AAAI 2025

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18844v1) [paper-pdf](http://arxiv.org/pdf/2412.18844v1)

**Authors**: Yuchen Ren, Zhengyu Zhao, Chenhao Lin, Bo Yang, Lu Zhou, Zhe Liu, Chao Shen

**Abstract**: Transferable adversarial examples are known to cause threats in practical, black-box attack scenarios. A notable approach to improving transferability is using integrated gradients (IG), originally developed for model interpretability. In this paper, we find that existing IG-based attacks have limited transferability due to their naive adoption of IG in model interpretability. To address this limitation, we focus on the IG integration path and refine it in three aspects: multiplicity, monotonicity, and diversity, supported by theoretical analyses. We propose the Multiple Monotonic Diversified Integrated Gradients (MuMoDIG) attack, which can generate highly transferable adversarial examples on different CNN and ViT models and defenses. Experiments validate that MuMoDIG outperforms the latest IG-based attack by up to 37.3\% and other state-of-the-art attacks by 8.4\%. In general, our study reveals that migrating established techniques to improve transferability may require non-trivial efforts. Code is available at \url{https://github.com/RYC-98/MuMoDIG}.

摘要: 众所周知，可转移的对抗性示例在实际的黑盒攻击场景中会造成威胁。提高可转移性的一个值得注意的方法是使用集成梯度(IG)，它最初是为模型的可解释性而开发的。在本文中，我们发现现有的基于IG的攻击由于在模型可解释性方面对IG的天真采用而具有有限的可转移性。为了解决这一局限性，我们聚焦于IG整合路径，并在理论分析的支持下，从多样性、单调性和多样性三个方面对其进行了提炼。提出了多重单调多元集成梯度(MuMoDIG)攻击，该攻击可以在不同的CNN和VIT模型和防御上产生高度可移植的敌意实例。实验证明，MuMoDIG的性能比最新的基于IG的攻击高出37.3\%，比其他最新的攻击高出8.4\%。总体而言，我们的研究表明，移植已有的技术以提高可转移性可能需要付出巨大的努力。代码位于\url{https://github.com/RYC-98/MuMoDIG}.



## **34. Distortion-Aware Adversarial Attacks on Bounding Boxes of Object Detectors**

对物体检测器边界盒的失真感知对抗攻击 cs.CV

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18815v1) [paper-pdf](http://arxiv.org/pdf/2412.18815v1)

**Authors**: Pham Phuc, Son Vuong, Khang Nguyen, Tuan Dang

**Abstract**: Deep learning-based object detection has become ubiquitous in the last decade due to its high accuracy in many real-world applications. With this growing trend, these models are interested in being attacked by adversaries, with most of the results being on classifiers, which do not match the context of practical object detection. In this work, we propose a novel method to fool object detectors, expose the vulnerability of state-of-the-art detectors, and promote later works to build more robust detectors to adversarial examples. Our method aims to generate adversarial images by perturbing object confidence scores during training, which is crucial in predicting confidence for each class in the testing phase. Herein, we provide a more intuitive technique to embed additive noises based on detected objects' masks and the training loss with distortion control over the original image by leveraging the gradient of iterative images. To verify the proposed method, we perform adversarial attacks against different object detectors, including the most recent state-of-the-art models like YOLOv8, Faster R-CNN, RetinaNet, and Swin Transformer. We also evaluate our technique on MS COCO 2017 and PASCAL VOC 2012 datasets and analyze the trade-off between success attack rate and image distortion. Our experiments show that the achievable success attack rate is up to $100$\% and up to $98$\% when performing white-box and black-box attacks, respectively. The source code and relevant documentation for this work are available at the following link: https://github.com/anonymous20210106/attack_detector

摘要: 在过去的十年里，基于深度学习的目标检测已经变得无处不在，因为它在许多实际应用中都具有很高的准确率。随着这一趋势的发展，这些模型对受到对手的攻击感兴趣，大多数结果都是基于分类器的，这与实际目标检测的上下文不匹配。在这项工作中，我们提出了一种新的方法来愚弄对象检测器，揭露现有检测器的脆弱性，并将后来的工作推广到构建更健壮的检测器。我们的方法旨在通过在训练过程中扰动对象置信度分数来生成对抗性图像，这对于在测试阶段预测每个类别的置信度是至关重要的。在这里，我们提供了一种更直观的技术，通过利用迭代图像的梯度来嵌入基于检测对象的掩模和训练损失的加性噪声，并对原始图像进行失真控制。为了验证所提出的方法，我们对不同的对象检测器进行了对抗性攻击，包括最新的模型，如YOLOv8，FASTER R-CNN，RetinaNet和Swin Transformer。我们还在MS Coco 2017和Pascal VOC 2012数据集上对我们的技术进行了评估，并分析了攻击成功率和图像失真之间的权衡。实验表明，白盒攻击和黑盒攻击的成功率分别达到100美元和98美元。这项工作的源代码和相关文档可在以下链接中找到：https://github.com/anonymous20210106/attack_detector



## **35. Protective Perturbations against Unauthorized Data Usage in Diffusion-based Image Generation**

基于扩散的图像生成中针对未经授权的数据使用的保护性扰动 cs.CV

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18791v1) [paper-pdf](http://arxiv.org/pdf/2412.18791v1)

**Authors**: Sen Peng, Jijia Yang, Mingyue Wang, Jianfei He, Xiaohua Jia

**Abstract**: Diffusion-based text-to-image models have shown immense potential for various image-related tasks. However, despite their prominence and popularity, customizing these models using unauthorized data also brings serious privacy and intellectual property issues. Existing methods introduce protective perturbations based on adversarial attacks, which are applied to the customization samples. In this systematization of knowledge, we present a comprehensive survey of protective perturbation methods designed to prevent unauthorized data usage in diffusion-based image generation. We establish the threat model and categorize the downstream tasks relevant to these methods, providing a detailed analysis of their designs. We also propose a completed evaluation framework for these perturbation techniques, aiming to advance research in this field.

摘要: 基于扩散的文本到图像模型在各种图像相关任务中表现出了巨大的潜力。然而，尽管它们引人注目且受欢迎，但使用未经授权的数据定制这些模型也带来了严重的隐私和知识产权问题。现有方法引入基于对抗攻击的保护性扰动，并应用于定制样本。在知识的系统化中，我们对旨在防止在基于扩散的图像生成中未经授权使用数据的保护性扰动方法进行了全面调查。我们建立威胁模型并对与这些方法相关的下游任务进行分类，并对其设计进行详细分析。我们还为这些扰动技术提出了一个完整的评估框架，旨在推进该领域的研究。



## **36. Attack-in-the-Chain: Bootstrapping Large Language Models for Attacks Against Black-box Neural Ranking Models**

链中攻击：引导大型语言模型来攻击黑匣子神经排名模型 cs.IR

Accepted by AAAI25

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18770v1) [paper-pdf](http://arxiv.org/pdf/2412.18770v1)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Neural ranking models (NRMs) have been shown to be highly effective in terms of retrieval performance. Unfortunately, they have also displayed a higher degree of sensitivity to attacks than previous generation models. To help expose and address this lack of robustness, we introduce a novel ranking attack framework named Attack-in-the-Chain, which tracks interactions between large language models (LLMs) and NRMs based on chain-of-thought (CoT) prompting to generate adversarial examples under black-box settings. Our approach starts by identifying anchor documents with higher ranking positions than the target document as nodes in the reasoning chain. We then dynamically assign the number of perturbation words to each node and prompt LLMs to execute attacks. Finally, we verify the attack performance of all nodes at each reasoning step and proceed to generate the next reasoning step. Empirical results on two web search benchmarks show the effectiveness of our method.

摘要: 神经排名模型（NRM）已被证明在检索性能方面非常有效。不幸的是，它们还表现出比前一代模型更高的攻击敏感性。为了帮助揭露和解决这种缺乏稳健性的问题，我们引入了一种名为Chain Attack-in-the-Chain的新型排名攻击框架，该框架基于思想链（CoT）来跟踪大型语言模型（LLM）和NRM之间的交互，以在黑匣子设置下生成对抗性示例。我们的方法首先将排名位置高于目标文档的锚文档识别为推理链中的节点。然后，我们动态地为每个节点分配扰动字的数量，并提示LLM执行攻击。最后，我们在每个推理步骤中验证所有节点的攻击性能，并继续生成下一个推理步骤。两个网络搜索基准的经验结果表明了我们方法的有效性。



## **37. Token Highlighter: Inspecting and Mitigating Jailbreak Prompts for Large Language Models**

Token Highliter：检查和缓解大型语言模型的越狱承诺 cs.CR

Accepted by AAAI 2025. Project page:  https://huggingface.co/spaces/TrustSafeAI/Token-Highlighter

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18171v2) [paper-pdf](http://arxiv.org/pdf/2412.18171v2)

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Large Language Models (LLMs) are increasingly being integrated into services such as ChatGPT to provide responses to user queries. To mitigate potential harm and prevent misuse, there have been concerted efforts to align the LLMs with human values and legal compliance by incorporating various techniques, such as Reinforcement Learning from Human Feedback (RLHF), into the training of the LLMs. However, recent research has exposed that even aligned LLMs are susceptible to adversarial manipulations known as Jailbreak Attacks. To address this challenge, this paper proposes a method called Token Highlighter to inspect and mitigate the potential jailbreak threats in the user query. Token Highlighter introduced a concept called Affirmation Loss to measure the LLM's willingness to answer the user query. It then uses the gradient of Affirmation Loss for each token in the user query to locate the jailbreak-critical tokens. Further, Token Highlighter exploits our proposed Soft Removal technique to mitigate the jailbreak effects of critical tokens via shrinking their token embeddings. Experimental results on two aligned LLMs (LLaMA-2 and Vicuna-V1.5) demonstrate that the proposed method can effectively defend against a variety of Jailbreak Attacks while maintaining competent performance on benign questions of the AlpacaEval benchmark. In addition, Token Highlighter is a cost-effective and interpretable defense because it only needs to query the protected LLM once to compute the Affirmation Loss and can highlight the critical tokens upon refusal.

摘要: 大型语言模型(LLM)越来越多地被集成到ChatGPT等服务中，以提供对用户查询的响应。为减少潜在危害和防止滥用，已作出协调一致的努力，通过将从人类反馈中强化学习(RLHF)等各种技术纳入LLMS的培训，使LLMS与人的价值观和法律合规保持一致。然而，最近的研究表明，即使是对准的LLM也容易受到称为越狱攻击的对抗性操纵的影响。为了应对这一挑战，本文提出了一种称为令牌荧光的方法来检测和缓解用户查询中潜在的越狱威胁。令牌亮点引入了一个名为肯定损失的概念，以衡量LLM回答用户问题的意愿。然后，它使用用户查询中每个令牌的确认损失梯度来定位越狱关键令牌。此外，令牌荧光利用我们提出的软删除技术，通过缩小关键令牌的令牌嵌入来缓解关键令牌的越狱影响。在两个对齐的LLMS(Llama-2和Vicuna-V1.5)上的实验结果表明，该方法可以有效地防御各种越狱攻击，同时保持在AlpacaEval基准测试的良性问题上的良好性能。此外，令牌加亮器是一种经济高效且可解释的防御方案，因为它只需查询受保护的LLM一次即可计算肯定损失，并且可以在拒绝时突出显示关键令牌。



## **38. Evaluating the Adversarial Robustness of Detection Transformers**

评估检测转换器的对抗鲁棒性 cs.CV

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18718v1) [paper-pdf](http://arxiv.org/pdf/2412.18718v1)

**Authors**: Amirhossein Nazeri, Chunheng Zhao, Pierluigi Pisu

**Abstract**: Robust object detection is critical for autonomous driving and mobile robotics, where accurate detection of vehicles, pedestrians, and obstacles is essential for ensuring safety. Despite the advancements in object detection transformers (DETRs), their robustness against adversarial attacks remains underexplored. This paper presents a comprehensive evaluation of DETR model and its variants under both white-box and black-box adversarial attacks, using the MS-COCO and KITTI datasets to cover general and autonomous driving scenarios. We extend prominent white-box attack methods (FGSM, PGD, and CW) to assess DETR vulnerability, demonstrating that DETR models are significantly susceptible to adversarial attacks, similar to traditional CNN-based detectors. Our extensive transferability analysis reveals high intra-network transferability among DETR variants, but limited cross-network transferability to CNN-based models. Additionally, we propose a novel untargeted attack designed specifically for DETR, exploiting its intermediate loss functions to induce misclassification with minimal perturbations. Visualizations of self-attention feature maps provide insights into how adversarial attacks affect the internal representations of DETR models. These findings reveal critical vulnerabilities in detection transformers under standard adversarial attacks, emphasizing the need for future research to enhance the robustness of transformer-based object detectors in safety-critical applications.

摘要: 稳健的目标检测对于自动驾驶和移动机器人至关重要，在这些领域，对车辆、行人和障碍物的准确检测对于确保安全至关重要。尽管目标检测转换器(DETR)有了很大的进步，但它们对对手攻击的健壮性仍然没有得到充分的研究。本文使用MS-COCO和KITTI数据集对白盒和黑盒对抗攻击下的DETR模型及其变体进行了综合评估，以涵盖一般和自动驾驶场景。我们扩展了著名的白盒攻击方法(FGSM、PGD和CW)来评估DETR漏洞，表明DETR模型与传统的基于CNN的检测器相似，非常容易受到对抗性攻击。我们广泛的可转移性分析表明，DETR变体之间的网络内可转移性很高，但对基于CNN的模型的跨网络可转移性有限。此外，我们还提出了一种新的针对DETR的非目标攻击，利用其中间损失函数以最小的扰动来诱导错误分类。自我注意特征图的可视化提供了对对抗性攻击如何影响DETR模型的内部表示的洞察。这些发现揭示了标准对抗性攻击下检测变压器的关键漏洞，强调了未来研究的必要性，以增强基于变压器的对象检测器在安全关键应用中的稳健性。



## **39. SurvAttack: Black-Box Attack On Survival Models through Ontology-Informed EHR Perturbation**

SurvAttack：通过基于实体的EHR扰动对生存模型进行黑匣子攻击 cs.LG

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18706v1) [paper-pdf](http://arxiv.org/pdf/2412.18706v1)

**Authors**: Mohsen Nayebi Kerdabadi, Arya Hadizadeh Moghaddam, Bin Liu, Mei Liu, Zijun Yao

**Abstract**: Survival analysis (SA) models have been widely studied in mining electronic health records (EHRs), particularly in forecasting the risk of critical conditions for prioritizing high-risk patients. However, their vulnerability to adversarial attacks is much less explored in the literature. Developing black-box perturbation algorithms and evaluating their impact on state-of-the-art survival models brings two benefits to medical applications. First, it can effectively evaluate the robustness of models in pre-deployment testing. Also, exploring how subtle perturbations would result in significantly different outcomes can provide counterfactual insights into the clinical interpretation of model prediction. In this work, we introduce SurvAttack, a novel black-box adversarial attack framework leveraging subtle clinically compatible, and semantically consistent perturbations on longitudinal EHRs to degrade survival models' predictive performance. We specifically develop a greedy algorithm to manipulate medical codes with various adversarial actions throughout a patient's medical history. Then, these adversarial actions are prioritized using a composite scoring strategy based on multi-aspect perturbation quality, including saliency, perturbation stealthiness, and clinical meaningfulness. The proposed adversarial EHR perturbation algorithm is then used in an efficient SA-specific strategy to attack a survival model when estimating the temporal ranking of survival urgency for patients. To demonstrate the significance of our work, we conduct extensive experiments, including baseline comparisons, explainability analysis, and case studies. The experimental results affirm our research's effectiveness in illustrating the vulnerabilities of patient survival models, model interpretation, and ultimately contributing to healthcare quality.

摘要: 生存分析(SA)模型在挖掘电子健康记录(EHR)中得到了广泛的研究，特别是在预测危重疾病的风险以优先处理高危患者方面。然而，它们在对抗性攻击中的脆弱性在文献中很少被探讨。开发黑盒扰动算法并评估它们对最先进的生存模型的影响为医学应用带来了两个好处。首先，它可以在部署前测试中有效地评估模型的稳健性。此外，探索细微的扰动如何导致显著不同的结果，可以为模型预测的临床解释提供反事实的见解。在这项工作中，我们引入了SurvAttack，一个新的黑盒对抗性攻击框架，利用对纵向EHR的微妙的临床兼容和语义一致的扰动来降低生存模型的预测性能。我们专门开发了一种贪婪的算法来操纵医疗代码，在患者的病史上采取各种敌对行动。然后，使用基于多方面扰动质量(包括显著程度、扰动隐蔽性和临床意义)的综合评分策略对这些对抗性动作进行优先排序。然后，将所提出的对抗性EHR扰动算法用于有效的SA特定策略中，在估计患者生存紧迫性的时间排序时攻击生存模型。为了证明我们工作的重要性，我们进行了广泛的实验，包括基线比较、可解释性分析和案例研究。实验结果肯定了我们的研究在阐明患者生存模型的脆弱性、模型解释以及最终对医疗质量做出贡献方面的有效性。



## **40. Adversarial Attack Against Images Classification based on Generative Adversarial Networks**

基于生成对抗网络的图像分类对抗攻击 cs.CV

7 pages, 6 figures

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.16662v2) [paper-pdf](http://arxiv.org/pdf/2412.16662v2)

**Authors**: Yahe Yang

**Abstract**: Adversarial attacks on image classification systems have always been an important problem in the field of machine learning, and generative adversarial networks (GANs), as popular models in the field of image generation, have been widely used in various novel scenarios due to their powerful generative capabilities. However, with the popularity of generative adversarial networks, the misuse of fake image technology has raised a series of security problems, such as malicious tampering with other people's photos and videos, and invasion of personal privacy. Inspired by the generative adversarial networks, this work proposes a novel adversarial attack method, aiming to gain insight into the weaknesses of the image classification system and improve its anti-attack ability. Specifically, the generative adversarial networks are used to generate adversarial samples with small perturbations but enough to affect the decision-making of the classifier, and the adversarial samples are generated through the adversarial learning of the training generator and the classifier. From extensive experiment analysis, we evaluate the effectiveness of the method on a classical image classification dataset, and the results show that our model successfully deceives a variety of advanced classifiers while maintaining the naturalness of adversarial samples.

摘要: 针对图像分类系统的对抗性攻击一直是机器学习领域的一个重要问题，而生成性对抗性网络(GANS)作为图像生成领域的热门模型，由于其强大的生成能力而被广泛应用于各种新颖的场景中。然而，随着生成性对抗网络的流行，虚假图像技术的滥用引发了一系列安全问题，如恶意篡改他人照片和视频、侵犯个人隐私等。受生成式对抗性网络的启发，本文提出了一种新颖的对抗性攻击方法，旨在洞察图像分类系统的弱点，提高其抗攻击能力。具体地说，生成式对抗性网络用于生成扰动较小但足以影响分类器决策的对抗性样本，并通过训练器和分类器的对抗性学习来生成对抗性样本。通过大量的实验分析，我们在一个经典的图像分类数据集上对该方法的有效性进行了评估，结果表明，我们的模型成功地欺骗了各种高级分类器，同时保持了对抗性样本的自然性。



## **41. An Empirical Analysis of Federated Learning Models Subject to Label-Flipping Adversarial Attack**

受标签翻转对抗攻击的联邦学习模型的实证分析 cs.LG

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18507v1) [paper-pdf](http://arxiv.org/pdf/2412.18507v1)

**Authors**: Kunal Bhatnagar, Sagana Chattanathan, Angela Dang, Bhargav Eranki, Ronnit Rana, Charan Sridhar, Siddharth Vedam, Angie Yao, Mark Stamp

**Abstract**: In this paper, we empirically analyze adversarial attacks on selected federated learning models. The specific learning models considered are Multinominal Logistic Regression (MLR), Support Vector Classifier (SVC), Multilayer Perceptron (MLP), Convolution Neural Network (CNN), %Recurrent Neural Network (RNN), Random Forest, XGBoost, and Long Short-Term Memory (LSTM). For each model, we simulate label-flipping attacks, experimenting extensively with 10 federated clients and 100 federated clients. We vary the percentage of adversarial clients from 10% to 100% and, simultaneously, the percentage of labels flipped by each adversarial client is also varied from 10% to 100%. Among other results, we find that models differ in their inherent robustness to the two vectors in our label-flipping attack, i.e., the percentage of adversarial clients, and the percentage of labels flipped by each adversarial client. We discuss the potential practical implications of our results.

摘要: 在本文中，我们实证分析了对选定联邦学习模型的对抗攻击。考虑的具体学习模型是多项逻辑回归（MLR）、支持载体分类器（SRC）、多层感知器（MLP）、卷积神经网络（CNN）、%回归神经网络（RNN）、随机森林、XGboost和长短期记忆（LSTM）。对于每个模型，我们模拟标签翻转攻击，对10个联邦客户端和100个联邦客户端进行了广泛实验。我们将敌对客户的百分比从10%到100%不等，同时，每个敌对客户翻转的标签百分比也从10%到100%不等。除其他结果外，我们发现模型对标签翻转攻击中的两个载体的固有鲁棒性有所不同，即敌对客户的百分比，以及每个敌对客户翻转的标签百分比。我们讨论了结果的潜在实际影响。



## **42. Prompted Contextual Vectors for Spear-Phishing Detection**

用于鱼叉钓鱼检测的预定上下文载体 cs.LG

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2402.08309v3) [paper-pdf](http://arxiv.org/pdf/2402.08309v3)

**Authors**: Daniel Nahmias, Gal Engelberg, Dan Klein, Asaf Shabtai

**Abstract**: Spear-phishing attacks present a significant security challenge, with large language models (LLMs) escalating the threat by generating convincing emails and facilitating target reconnaissance. To address this, we propose a detection approach based on a novel document vectorization method that utilizes an ensemble of LLMs to create representation vectors. By prompting LLMs to reason and respond to human-crafted questions, we quantify the presence of common persuasion principles in the email's content, producing prompted contextual document vectors for a downstream supervised machine learning model. We evaluate our method using a unique dataset generated by a proprietary system that automates target reconnaissance and spear-phishing email creation. Our method achieves a 91\% F1 score in identifying LLM-generated spear-phishing emails, with the training set comprising only traditional phishing and benign emails. Key contributions include a novel document vectorization method utilizing LLM reasoning, a publicly available dataset of high-quality spear-phishing emails, and the demonstrated effectiveness of our method in detecting such emails. This methodology can be utilized for various document classification tasks, particularly in adversarial problem domains.

摘要: 鱼叉式网络钓鱼攻击是一个重大的安全挑战，大型语言模型(LLM)通过生成令人信服的电子邮件和促进目标侦察来升级威胁。针对这一问题，我们提出了一种基于一种新的文档矢量化方法的检测方法，该方法利用一组LLM来创建表示向量。通过促使LLM对人类提出的问题进行推理和回应，我们量化了电子邮件内容中常见说服原则的存在，为下游有监督的机器学习模型生成了提示的上下文文档向量。我们使用由专有系统生成的唯一数据集来评估我们的方法，该系统自动执行目标侦察和鱼叉式网络钓鱼电子邮件创建。我们的方法在识别LLM生成的鱼叉式钓鱼邮件方面取得了91%的F1分数，训练集仅包括传统钓鱼邮件和良性电子邮件。主要贡献包括一种利用LLM推理的新的文档矢量化方法，一个公开可用的高质量鱼叉式钓鱼电子邮件数据集，以及我们的方法在检测此类电子邮件方面的有效性。这种方法可用于各种文档分类任务，特别是在对抗性问题领域。



## **43. Unveiling the Threat of Fraud Gangs to Graph Neural Networks: Multi-Target Graph Injection Attacks against GNN-Based Fraud Detectors**

揭露欺诈团伙对图神经网络的威胁：针对基于GNN的欺诈检测器的多目标图注入攻击 cs.LG

19 pages, 5 figures, 12 tables, The 39th AAAI Conference on  Artificial Intelligence (AAAI 2025)

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18370v1) [paper-pdf](http://arxiv.org/pdf/2412.18370v1)

**Authors**: Jinhyeok Choi, Heehyeon Kim, Joyce Jiyoung Whang

**Abstract**: Graph neural networks (GNNs) have emerged as an effective tool for fraud detection, identifying fraudulent users, and uncovering malicious behaviors. However, attacks against GNN-based fraud detectors and their risks have rarely been studied, thereby leaving potential threats unaddressed. Recent findings suggest that frauds are increasingly organized as gangs or groups. In this work, we design attack scenarios where fraud gangs aim to make their fraud nodes misclassified as benign by camouflaging their illicit activities in collusion. Based on these scenarios, we study adversarial attacks against GNN-based fraud detectors by simulating attacks of fraud gangs in three real-world fraud cases: spam reviews, fake news, and medical insurance frauds. We define these attacks as multi-target graph injection attacks and propose MonTi, a transformer-based Multi-target one-Time graph injection attack model. MonTi simultaneously generates attributes and edges of all attack nodes with a transformer encoder, capturing interdependencies between attributes and edges more effectively than most existing graph injection attack methods that generate these elements sequentially. Additionally, MonTi adaptively allocates the degree budget for each attack node to explore diverse injection structures involving target, candidate, and attack nodes, unlike existing methods that fix the degree budget across all attack nodes. Experiments show that MonTi outperforms the state-of-the-art graph injection attack methods on five real-world graphs.

摘要: 图神经网络(GNN)已经成为检测欺诈、识别欺诈用户和揭露恶意行为的有效工具。然而，对基于GNN的欺诈探测器的攻击及其风险很少被研究，从而使潜在的威胁得不到解决。最近的发现表明，诈骗越来越多地被组织成帮派或团体。在这项工作中，我们设计了攻击场景，其中欺诈团伙的目标是通过伪装他们在串通中的非法活动来使他们的欺诈节点错误地被归类为良性的。基于这些场景，我们通过模拟三个真实世界的欺诈案例：垃圾邮件评论、假新闻和医疗保险欺诈，研究了针对基于GNN的欺诈检测器的对抗性攻击。我们将这些攻击定义为多目标图注入攻击，并提出了一种基于变压器的多目标一次性图注入攻击模型MONTI。Monti使用转换器编码器同时生成所有攻击节点的属性和边，比大多数现有的按顺序生成这些元素的图注入攻击方法更有效地捕获属性和边之间的相互依赖关系。此外，与现有方法固定所有攻击节点的度预算不同，Monti自适应地为每个攻击节点分配度预算，以探索涉及目标、候选和攻击节点的不同注入结构。实验表明，Monti在五个真实图上的性能优于目前最先进的图注入攻击方法。



## **44. Hypergraph Attacks via Injecting Homogeneous Nodes into Elite Hyperedges**

通过将同质节点注入精英超文本攻击 cs.LG

9 pages, The 39th Annual AAAI Conference on Artificial  Intelligence(2025)

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18365v1) [paper-pdf](http://arxiv.org/pdf/2412.18365v1)

**Authors**: Meixia He, Peican Zhu, Keke Tang, Yangming Guo

**Abstract**: Recent studies have shown that Hypergraph Neural Networks (HGNNs) are vulnerable to adversarial attacks. Existing approaches focus on hypergraph modification attacks guided by gradients, overlooking node spanning in the hypergraph and the group identity of hyperedges, thereby resulting in limited attack performance and detectable attacks. In this manuscript, we present a novel framework, i.e., Hypergraph Attacks via Injecting Homogeneous Nodes into Elite Hyperedges (IE-Attack), to tackle these challenges. Initially, utilizing the node spanning in the hypergraph, we propose the elite hyperedges sampler to identify hyperedges to be injected. Subsequently, a node generator utilizing Kernel Density Estimation (KDE) is proposed to generate the homogeneous node with the group identity of hyperedges. Finally, by injecting the homogeneous node into elite hyperedges, IE-Attack improves the attack performance and enhances the imperceptibility of attacks. Extensive experiments are conducted on five authentic datasets to validate the effectiveness of IE-Attack and the corresponding superiority to state-of-the-art methods.

摘要: 最近的研究表明，超图神经网络(HGNN)容易受到敌意攻击。现有的攻击方法主要关注梯度引导的超图修改攻击，忽略了超图中节点的生成和超边的群标识，从而导致攻击性能有限和攻击可检测。在这篇文章中，我们提出了一种新的框架，即通过向精英超边注入同质节点的超图攻击(IE-Attack)来应对这些挑战。首先，利用超图中的节点生成特性，提出了精英超边采样器来识别待注入的超边。在此基础上，提出了一种基于核密度估计(KDE)的节点生成器来生成具有超边群同一性的同质节点。最后，IE-Attack通过在精英超边中注入同质节点，改善了攻击性能，增强了攻击的隐蔽性。在五个真实的数据集上进行了大量的实验，以验证IE攻击的有效性及其相对于最新方法的优越性。



## **45. Level Up with ML Vulnerability Identification: Leveraging Domain Constraints in Feature Space for Robust Android Malware Detection**

利用ML漏洞识别升级：利用特征空间中的域约束进行稳健的Android恶意软件检测 cs.LG

The paper was accepted by ACM Transactions on Privacy and Security on  2 December 2024

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2205.15128v4) [paper-pdf](http://arxiv.org/pdf/2205.15128v4)

**Authors**: Hamid Bostani, Zhengyu Zhao, Zhuoran Liu, Veelasha Moonsamy

**Abstract**: Machine Learning (ML) promises to enhance the efficacy of Android Malware Detection (AMD); however, ML models are vulnerable to realistic evasion attacks--crafting realizable Adversarial Examples (AEs) that satisfy Android malware domain constraints. To eliminate ML vulnerabilities, defenders aim to identify susceptible regions in the feature space where ML models are prone to deception. The primary approach to identifying vulnerable regions involves investigating realizable AEs, but generating these feasible apps poses a challenge. For instance, previous work has relied on generating either feature-space norm-bounded AEs or problem-space realizable AEs in adversarial hardening. The former is efficient but lacks full coverage of vulnerable regions while the latter can uncover these regions by satisfying domain constraints but is known to be time-consuming. To address these limitations, we propose an approach to facilitate the identification of vulnerable regions. Specifically, we introduce a new interpretation of Android domain constraints in the feature space, followed by a novel technique that learns them. Our empirical evaluations across various evasion attacks indicate effective detection of AEs using learned domain constraints, with an average of 89.6%. Furthermore, extensive experiments on different Android malware detectors demonstrate that utilizing our learned domain constraints in Adversarial Training (AT) outperforms other AT-based defenses that rely on norm-bounded AEs or state-of-the-art non-uniform perturbations. Finally, we show that retraining a malware detector with a wide variety of feature-space realizable AEs results in a 77.9% robustness improvement against realizable AEs generated by unknown problem-space transformations, with up to 70x faster training than using problem-space realizable AEs.

摘要: 机器学习(ML)有望提高Android恶意软件检测(AMD)的效率；然而，ML模型容易受到现实的逃避攻击--制作满足Android恶意软件领域约束的可实现的对手示例(AE)。为了消除ML漏洞，防御者的目标是识别特征空间中ML模型容易被欺骗的敏感区域。识别易受攻击地区的主要方法包括调查可实现的企业实体，但生成这些可行的应用程序会带来挑战。例如，以前的工作依赖于在对抗性强化中产生特征空间范数有界的实体或问题空间可实现的实体。前者是有效的，但缺乏对脆弱区域的完全覆盖，而后者可以通过满足域约束来发现这些区域，但众所周知是耗时的。为了解决这些限制，我们提出了一种便于识别脆弱区域的方法。具体地说，我们在特征空间中引入了对Android域约束的新解释，随后采用了一种新的技术来学习它们。我们对各种规避攻击的实验评估表明，使用学习的域约束可以有效地检测到AEs，平均检测准确率为89.6%。此外，在不同的Android恶意软件检测器上的大量实验表明，在对抗训练(AT)中利用我们学习的域约束的性能优于其他基于AT的防御系统，这些防御系统依赖于范数有界的AE或最新的非均匀扰动。最后，我们展示了用各种各样的特征空间可实现的AE来重新训练恶意软件检测器，对于未知问题空间变换产生的可实现的AE，健壮性提高了77.9%，训练速度比使用问题空间可实现的AE快70倍。



## **46. Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?**

大型语言模型能否提高图神经网络的对抗鲁棒性？ cs.LG

accepted by KDD 2025

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2408.08685v3) [paper-pdf](http://arxiv.org/pdf/2408.08685v3)

**Authors**: Zhongjian Zhang, Xiao Wang, Huichi Zhou, Yue Yu, Mengmei Zhang, Cheng Yang, Chuan Shi

**Abstract**: Graph neural networks (GNNs) are vulnerable to adversarial attacks, especially for topology perturbations, and many methods that improve the robustness of GNNs have received considerable attention. Recently, we have witnessed the significant success of large language models (LLMs), leading many to explore the great potential of LLMs on GNNs. However, they mainly focus on improving the performance of GNNs by utilizing LLMs to enhance the node features. Therefore, we ask: Will the robustness of GNNs also be enhanced with the powerful understanding and inference capabilities of LLMs? By presenting the empirical results, we find that despite that LLMs can improve the robustness of GNNs, there is still an average decrease of 23.1% in accuracy, implying that the GNNs remain extremely vulnerable against topology attacks. Therefore, another question is how to extend the capabilities of LLMs on graph adversarial robustness. In this paper, we propose an LLM-based robust graph structure inference framework, LLM4RGNN, which distills the inference capabilities of GPT-4 into a local LLM for identifying malicious edges and an LM-based edge predictor for finding missing important edges, so as to recover a robust graph structure. Extensive experiments demonstrate that LLM4RGNN consistently improves the robustness across various GNNs. Even in some cases where the perturbation ratio increases to 40%, the accuracy of GNNs is still better than that on the clean graph. The source code can be found in https://github.com/zhongjian-zhang/LLM4RGNN.

摘要: 图神经网络(GNN)容易受到敌意攻击，尤其是对拓扑扰动的攻击，许多提高GNN健壮性的方法受到了广泛的关注。最近，我们目睹了大型语言模型(LLM)的巨大成功，这导致许多人探索LLM在GNN上的巨大潜力。然而，它们主要集中在通过利用LLMS来增强节点特征来提高GNN的性能。因此，我们问：GNN的健壮性是否也会随着LLMS强大的理解和推理能力而得到增强？通过给出实验结果，我们发现，尽管LLMS可以提高GNN的健壮性，但其准确率仍然平均下降23.1%，这意味着GNN仍然非常容易受到拓扑攻击。因此，另一个问题是如何扩展LLMS在图对抗健壮性方面的能力。本文提出了一种基于LLM的稳健图结构推理框架LLM4RGNN，该框架将GPT-4的推理能力抽象为用于识别恶意边的局部LLM和用于发现丢失重要边的基于LLM的边预测器，以恢复稳健的图结构。大量的实验表明，LLM4RGNN在不同的GNN上一致地提高了健壮性。即使在某些扰动比增加到40%的情况下，GNN的精度仍然好于干净图形上的精度。源代码可以在https://github.com/zhongjian-zhang/LLM4RGNN.中找到



## **47. On the Effectiveness of Adversarial Training on Malware Classifiers**

恶意软件分类器对抗训练的有效性 cs.LG

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18218v1) [paper-pdf](http://arxiv.org/pdf/2412.18218v1)

**Authors**: Hamid Bostani, Jacopo Cortellazzi, Daniel Arp, Fabio Pierazzi, Veelasha Moonsamy, Lorenzo Cavallaro

**Abstract**: Adversarial Training (AT) has been widely applied to harden learning-based classifiers against adversarial evasive attacks. However, its effectiveness in identifying and strengthening vulnerable areas of the model's decision space while maintaining high performance on clean data of malware classifiers remains an under-explored area. In this context, the robustness that AT achieves has often been assessed against unrealistic or weak adversarial attacks, which negatively affect performance on clean data and are arguably no longer threats. Previous work seems to suggest robustness is a task-dependent property of AT. We instead argue it is a more complex problem that requires exploring AT and the intertwined roles played by certain factors within data, feature representations, classifiers, and robust optimization settings, as well as proper evaluation factors, such as the realism of evasion attacks, to gain a true sense of AT's effectiveness. In our paper, we address this gap by systematically exploring the role such factors have in hardening malware classifiers through AT. Contrary to recent prior work, a key observation of our research and extensive experiments confirm the hypotheses that all such factors influence the actual effectiveness of AT, as demonstrated by the varying degrees of success from our empirical analysis. We identify five evaluation pitfalls that affect state-of-the-art studies and summarize our insights in ten takeaways to draw promising research directions toward better understanding the factors' settings under which adversarial training works at best.

摘要: 对抗性训练(AT)已被广泛应用于强化基于学习的分类器抵抗对抗性回避攻击。然而，它在识别和加强模型决策空间的易受攻击区域方面的有效性，同时在恶意软件分类器的干净数据上保持高性能，仍然是一个探索不足的领域。在这种情况下，AT实现的健壮性经常被评估以对抗不现实或弱的对手攻击，这些攻击对干净数据的性能产生负面影响，并且可以说不再是威胁。前人的研究似乎表明稳健性是任务依赖的AT特性。相反，我们认为这是一个更复杂的问题，需要探索AT以及数据、特征表示、分类器和稳健优化设置中的某些因素所扮演的相互交织的角色，以及适当的评估因素，如规避攻击的真实性，以获得对AT有效性的真实感觉。在我们的论文中，我们通过系统地探索这些因素在通过AT强化恶意软件分类器中所起的作用来解决这一差距。与最近的工作相反，我们对研究的关键观察和广泛的实验证实了这样的假设，即所有这些因素都会影响AT的实际有效性，正如我们的实证分析所显示的不同程度的成功所证明的那样。我们找出了影响最先进研究的五个评估陷阱，并总结了我们在十个方面的见解，以得出有希望的研究方向，以更好地理解对抗性训练最好发挥作用的因素设置。



## **48. Robustness-aware Automatic Prompt Optimization**

具有鲁棒性的自动提示优化 cs.CL

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18196v1) [paper-pdf](http://arxiv.org/pdf/2412.18196v1)

**Authors**: Zeru Shi, Zhenting Wang, Yongye Su, Weidi Luo, Fan Yang, Yongfeng Zhang

**Abstract**: The performance of Large Language Models (LLMs) is based on the quality of the prompts and the semantic and structural integrity information of the input data. However, current prompt generation methods primarily focus on generating prompts for clean input data, often overlooking the impact of perturbed inputs on prompt performance. To address this limitation, we propose BATprompt (By Adversarial Training prompt), a novel method for prompt generation designed to withstand input perturbations (such as typos in the input). Inspired by adversarial training techniques, BATprompt demonstrates strong performance on a variety of perturbed tasks through a two-step process: adversarial perturbation and iterative optimization on unperturbed input via LLM. Unlike conventional adversarial attack methods, BATprompt avoids reliance on real gradients or model parameters. Instead, it leverages the advanced reasoning, language understanding and self reflection capabilities of LLMs to simulate gradients, guiding the generation of adversarial perturbations and optimizing prompt performance. In our experiments, we evaluate BATprompt on multiple datasets across both language understanding and generation tasks. The results indicate that BATprompt outperforms existing prompt generation methods, delivering superior robustness and performance under diverse perturbation scenarios.

摘要: 大语言模型的性能取决于提示的质量以及输入数据的语义和结构完整性信息。然而，目前的提示生成方法主要集中于为干净的输入数据生成提示，往往忽略了输入干扰对提示性能的影响。为了解决这一局限性，我们提出了一种新的提示生成方法BATprint(通过对抗性训练提示)，该方法旨在抵抗输入扰动(如输入中的打字错误)。受到对抗性训练技术的启发，通过两步过程：对抗性扰动和通过LLM对不受扰动的输入进行迭代优化，BATprint在各种扰动任务上表现出了强大的性能。与传统的对抗性攻击方法不同，BATprint避免了对真实梯度或模型参数的依赖。相反，它利用LLMS的高级推理、语言理解和自我反思能力来模拟梯度，指导生成对抗性扰动并优化提示性能。在我们的实验中，我们在语言理解和生成任务的多个数据集上对BATprint进行了评估。结果表明，BATprint的性能优于现有的提示生成方法，在不同的扰动场景下都具有较好的健壮性和性能。



## **49. Sparse-PGD: A Unified Framework for Sparse Adversarial Perturbations Generation**

稀疏对抗扰动生成的统一框架 cs.LG

Extended version. Codes are available at  https://github.com/CityU-MLO/sPGD

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2405.05075v3) [paper-pdf](http://arxiv.org/pdf/2405.05075v3)

**Authors**: Xuyang Zhong, Chen Liu

**Abstract**: This work studies sparse adversarial perturbations, including both unstructured and structured ones. We propose a framework based on a white-box PGD-like attack method named Sparse-PGD to effectively and efficiently generate such perturbations. Furthermore, we combine Sparse-PGD with a black-box attack to comprehensively and more reliably evaluate the models' robustness against unstructured and structured sparse adversarial perturbations. Moreover, the efficiency of Sparse-PGD enables us to conduct adversarial training to build robust models against various sparse perturbations. Extensive experiments demonstrate that our proposed attack algorithm exhibits strong performance in different scenarios. More importantly, compared with other robust models, our adversarially trained model demonstrates state-of-the-art robustness against various sparse attacks.

摘要: 这项工作研究了稀疏的对抗性扰动，包括非结构化和结构化的扰动。我们提出了一个基于类似白盒PGD攻击方法的框架，名为Sparse-PVD，以有效且高效地生成此类扰动。此外，我们将Sparse-PGDD与黑匣子攻击相结合，以全面、更可靠地评估模型对非结构化和结构化稀疏对抗扰动的鲁棒性。此外，Sparse-PVD的效率使我们能够进行对抗训练，以针对各种稀疏扰动构建稳健的模型。大量实验表明，我们提出的攻击算法在不同场景下表现出强大的性能。更重要的是，与其他稳健模型相比，我们的对抗训练模型表现出了针对各种稀疏攻击的最新稳健性。



## **50. AEIOU: A Unified Defense Framework against NSFW Prompts in Text-to-Image Models**

AEIOU：针对文本到图像模型中NSFW格式的统一防御框架 cs.CR

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18123v1) [paper-pdf](http://arxiv.org/pdf/2412.18123v1)

**Authors**: Yiming Wang, Jiahao Chen, Qingming Li, Xing Yang, Shouling Ji

**Abstract**: As text-to-image (T2I) models continue to advance and gain widespread adoption, their associated safety issues are becoming increasingly prominent. Malicious users often exploit these models to generate Not-Safe-for-Work (NSFW) images using harmful or adversarial prompts, highlighting the critical need for robust safeguards to ensure the integrity and compliance of model outputs. Current internal safeguards frequently degrade image quality, while external detection methods often suffer from low accuracy and inefficiency.   In this paper, we introduce AEIOU, a defense framework that is Adaptable, Efficient, Interpretable, Optimizable, and Unified against NSFW prompts in T2I models. AEIOU extracts NSFW features from the hidden states of the model's text encoder, utilizing the separable nature of these features to detect NSFW prompts. The detection process is efficient, requiring minimal inference time. AEIOU also offers real-time interpretation of results and supports optimization through data augmentation techniques. The framework is versatile, accommodating various T2I architectures. Our extensive experiments show that AEIOU significantly outperforms both commercial and open-source moderation tools, achieving over 95% accuracy across all datasets and improving efficiency by at least tenfold. It effectively counters adaptive attacks and excels in few-shot and multi-label scenarios.

摘要: 随着文本到图像(T2I)模型的不断发展和广泛采用，其相关的安全问题也变得越来越突出。恶意用户经常利用这些模型，使用有害或敌对的提示生成不安全工作(NSFW)图像，突显出迫切需要强有力的保障措施，以确保模型输出的完整性和合规性。目前的内部保护措施经常会降低图像质量，而外部检测方法往往存在精度低和效率低的问题。在本文中，我们介绍了一种针对T2I模型中NSFW提示的适应性、高效、可解释、可优化和统一的防御框架AEIOU。AEIOU从模型的文本编码器的隐藏状态中提取NSFW特征，利用这些特征的可分离性来检测NSFW提示。检测过程是高效的，所需的推理时间最短。AEIOU还提供对结果的实时解释，并通过数据增强技术支持优化。该框架是通用的，可以容纳各种T2I架构。我们的广泛实验表明，AEIOU的性能明显优于商业和开源审核工具，在所有数据集上实现了95%以上的准确率，并将效率提高了至少10倍。它有效地对抗自适应攻击，并在少镜头和多标签场景中表现出色。



