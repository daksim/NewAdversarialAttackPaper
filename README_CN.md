# Latest Adversarial Attack Papers
**update at 2024-08-13 18:58:44**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. On Effects of Steering Latent Representation for Large Language Model Unlearning**

论引导潜在表示对大型语言模型取消学习的影响 cs.CL

15 pages, 5 figures, 8 tables

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.06223v1) [paper-pdf](http://arxiv.org/pdf/2408.06223v1)

**Authors**: Dang Huu-Tien, Trung-Tin Pham, Hoang Thanh-Tung, Naoya Inoue

**Abstract**: Representation Misdirection for Unlearning (RMU), which steers model representation in the intermediate layer to a target random representation, is an effective method for large language model (LLM) unlearning. Despite its high performance, the underlying cause and explanation remain underexplored. In this paper, we first theoretically demonstrate that steering forget representations in the intermediate layer reduces token confidence, causing LLMs to generate wrong or nonsense responses. Second, we investigate how the coefficient influences the alignment of forget-sample representations with the random direction and hint at the optimal coefficient values for effective unlearning across different network layers. Third, we show that RMU unlearned models are robust against adversarial jailbreak attacks. Last, our empirical analysis shows that RMU is less effective when applied to the middle and later layers in LLMs. To resolve this drawback, we propose Adaptive RMU -- a simple yet effective alternative method that makes unlearning effective with most layers. Extensive experiments demonstrate that Adaptive RMU significantly improves the unlearning performance compared to prior art while incurring no additional computational cost.

摘要: 遗忘表征误导(RMU)是一种有效的大语言模型遗忘方法，它将中间层的模型表征引导到目标随机表征。尽管其表现良好，但其根本原因和解释仍未得到充分研究。在本文中，我们首先从理论上证明，中间层中的转向遗忘表征会降低令牌置信度，从而导致LLM生成错误或无意义的响应。其次，我们研究了系数如何影响遗忘样本表示与随机方向的对齐，并提示了跨不同网络层有效遗忘的最优系数值。第三，我们证明了RMU未学习模型对敌意越狱攻击是健壮的。最后，我们的实证分析表明，当RMU应用于LLMS的中后期时，其有效性较差。为了解决这一缺陷，我们提出了自适应RMU--一种简单但有效的替代方法，使遗忘在大多数层都有效。大量实验表明，与现有技术相比，自适应RMU在不增加额外计算代价的情况下，显著改善了遗忘性能。



## **2. Lancelot: Towards Efficient and Privacy-Preserving Byzantine-Robust Federated Learning within Fully Homomorphic Encryption**

Lancelot：在完全同质加密中实现高效且保护隐私的拜占庭鲁棒联邦学习 cs.CR

26 pages

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.06197v1) [paper-pdf](http://arxiv.org/pdf/2408.06197v1)

**Authors**: Siyang Jiang, Hao Yang, Qipeng Xie, Chuan Ma, Sen Wang, Guoliang Xing

**Abstract**: In sectors such as finance and healthcare, where data governance is subject to rigorous regulatory requirements, the exchange and utilization of data are particularly challenging. Federated Learning (FL) has risen as a pioneering distributed machine learning paradigm that enables collaborative model training across multiple institutions while maintaining data decentralization. Despite its advantages, FL is vulnerable to adversarial threats, particularly poisoning attacks during model aggregation, a process typically managed by a central server. However, in these systems, neural network models still possess the capacity to inadvertently memorize and potentially expose individual training instances. This presents a significant privacy risk, as attackers could reconstruct private data by leveraging the information contained in the model itself. Existing solutions fall short of providing a viable, privacy-preserving BRFL system that is both completely secure against information leakage and computationally efficient. To address these concerns, we propose Lancelot, an innovative and computationally efficient BRFL framework that employs fully homomorphic encryption (FHE) to safeguard against malicious client activities while preserving data privacy. Our extensive testing, which includes medical imaging diagnostics and widely-used public image datasets, demonstrates that Lancelot significantly outperforms existing methods, offering more than a twenty-fold increase in processing speed, all while maintaining data privacy.

摘要: 在金融和医疗等数据治理受到严格监管要求的行业，数据的交换和利用尤其具有挑战性。联合学习(FL)已经成为一种开创性的分布式机器学习范例，它支持跨多个机构的协作模型训练，同时保持数据去中心化。尽管FL具有优势，但它很容易受到对手威胁，特别是在模型聚合期间的中毒攻击，这一过程通常由中央服务器管理。然而，在这些系统中，神经网络模型仍然具有无意中记忆和潜在地暴露个别训练实例的能力。这带来了很大的隐私风险，因为攻击者可以通过利用模型本身中包含的信息来重建私人数据。现有的解决方案不能提供一个可行的、保护隐私的BRFL系统，该系统既完全安全地防止信息泄漏，又具有计算效率。为了解决这些问题，我们提出了Lancelot，这是一个创新的、计算高效的BRFL框架，它使用完全同态加密(FHE)来防止恶意客户端活动，同时保护数据隐私。我们的广泛测试，包括医学成像诊断和广泛使用的公共图像数据集，表明Lancelot的性能显著优于现有方法，在保持数据隐私的同时，处理速度提高了20倍以上。



## **3. Towards Adversarial Robustness via Debiased High-Confidence Logit Alignment**

通过去偏高置信Logit对齐实现对抗鲁棒性 cs.CV

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.06079v1) [paper-pdf](http://arxiv.org/pdf/2408.06079v1)

**Authors**: Kejia Zhang, Juanjuan Weng, Zhiming Luo, Shaozi Li

**Abstract**: Despite the significant advances that deep neural networks (DNNs) have achieved in various visual tasks, they still exhibit vulnerability to adversarial examples, leading to serious security concerns. Recent adversarial training techniques have utilized inverse adversarial attacks to generate high-confidence examples, aiming to align the distributions of adversarial examples with the high-confidence regions of their corresponding classes. However, in this paper, our investigation reveals that high-confidence outputs under inverse adversarial attacks are correlated with biased feature activation. Specifically, training with inverse adversarial examples causes the model's attention to shift towards background features, introducing a spurious correlation bias. To address this bias, we propose Debiased High-Confidence Adversarial Training (DHAT), a novel approach that not only aligns the logits of adversarial examples with debiased high-confidence logits obtained from inverse adversarial examples, but also restores the model's attention to its normal state by enhancing foreground logit orthogonality. Extensive experiments demonstrate that DHAT achieves state-of-the-art performance and exhibits robust generalization capabilities across various vision datasets. Additionally, DHAT can seamlessly integrate with existing advanced adversarial training techniques for improving the performance.

摘要: 尽管深度神经网络(DNN)在各种视觉任务中取得了显著的进展，但它们仍然表现出对敌意例子的脆弱性，导致了严重的安全问题。最近的对抗性训练技术利用反向对抗性攻击来生成高置信度样本，旨在将对抗性样本的分布与其对应类别的高置信度区域对齐。然而，在本文中，我们的研究表明，反向对抗攻击下的高置信度输出与有偏的特征激活相关。具体地说，用反向对抗性例子进行训练会导致模型的注意力转移到背景特征上，从而引入虚假的相关偏差。为了解决这一偏差，我们提出了去偏高置信度对抗性训练(DHAT)，这是一种新的方法，它不仅将对抗性实例的逻辑与从反向对抗性实例获得的无偏高置信度对齐，而且通过增强前景对数正交性来恢复模型对其正常状态的关注。大量的实验表明，DHAT具有最先进的性能，并且在不同的视觉数据集上表现出强大的泛化能力。此外，DHAT可以与现有的高级对抗性训练技术无缝集成，以提高性能。



## **4. Multimodal Large Language Models for Phishing Webpage Detection and Identification**

用于网络钓鱼网页检测和识别的多模式大语言模型 cs.CR

To appear in eCrime 2024

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.05941v1) [paper-pdf](http://arxiv.org/pdf/2408.05941v1)

**Authors**: Jehyun Lee, Peiyuan Lim, Bryan Hooi, Dinil Mon Divakaran

**Abstract**: To address the challenging problem of detecting phishing webpages, researchers have developed numerous solutions, in particular those based on machine learning (ML) algorithms. Among these, brand-based phishing detection that uses models from Computer Vision to detect if a given webpage is imitating a well-known brand has received widespread attention. However, such models are costly and difficult to maintain, as they need to be retrained with labeled dataset that has to be regularly and continuously collected. Besides, they also need to maintain a good reference list of well-known websites and related meta-data for effective performance.   In this work, we take steps to study the efficacy of large language models (LLMs), in particular the multimodal LLMs, in detecting phishing webpages. Given that the LLMs are pretrained on a large corpus of data, we aim to make use of their understanding of different aspects of a webpage (logo, theme, favicon, etc.) to identify the brand of a given webpage and compare the identified brand with the domain name in the URL to detect a phishing attack. We propose a two-phase system employing LLMs in both phases: the first phase focuses on brand identification, while the second verifies the domain. We carry out comprehensive evaluations on a newly collected dataset. Our experiments show that the LLM-based system achieves a high detection rate at high precision; importantly, it also provides interpretable evidence for the decisions. Our system also performs significantly better than a state-of-the-art brand-based phishing detection system while demonstrating robustness against two known adversarial attacks.

摘要: 为了解决检测钓鱼网页这一具有挑战性的问题，研究人员开发了许多解决方案，特别是基于机器学习(ML)算法的解决方案。其中，基于品牌的钓鱼检测利用计算机视觉的模型来检测给定的网页是否在模仿知名品牌，受到了广泛的关注。然而，这种模型成本很高，很难维护，因为它们需要用必须定期和连续收集的标记数据集进行再训练。此外，他们还需要维护一个良好的参考名单的知名网站和相关的元数据，以有效的表现。在这项工作中，我们采取步骤研究大语言模型，特别是多模式大语言模型在检测钓鱼网页方面的有效性。鉴于LLM是在大型数据语料库上预先培训的，我们的目标是利用他们对网页的不同方面(徽标、主题、图标等)的理解。识别给定网页的品牌，并将识别的品牌与URL中的域名进行比较，以检测网络钓鱼攻击。我们提出了一个在两个阶段都使用LLMS的两阶段系统：第一阶段专注于品牌识别，第二阶段验证领域。我们对新收集的数据集进行了全面的评估。我们的实验表明，基于LLM的系统在高精度的情况下实现了高检测率，重要的是它还为决策提供了可解释的证据。我们的系统也比最先进的基于品牌的钓鱼检测系统性能要好得多，同时对两种已知的对手攻击表现出了健壮性。



## **5. Classifier Guidance Enhances Diffusion-based Adversarial Purification by Preserving Predictive Information**

分类器引导通过保留预测信息来增强基于扩散的对抗净化 cs.CV

Accepted by ECAI 2024

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.05900v1) [paper-pdf](http://arxiv.org/pdf/2408.05900v1)

**Authors**: Mingkun Zhang, Jianing Li, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Adversarial purification is one of the promising approaches to defend neural networks against adversarial attacks. Recently, methods utilizing diffusion probabilistic models have achieved great success for adversarial purification in image classification tasks. However, such methods fall into the dilemma of balancing the needs for noise removal and information preservation. This paper points out that existing adversarial purification methods based on diffusion models gradually lose sample information during the core denoising process, causing occasional label shift in subsequent classification tasks. As a remedy, we suggest to suppress such information loss by introducing guidance from the classifier confidence. Specifically, we propose Classifier-cOnfidence gUided Purification (COUP) algorithm, which purifies adversarial examples while keeping away from the classifier decision boundary. Experimental results show that COUP can achieve better adversarial robustness under strong attack methods.

摘要: 对抗净化是神经网络抵御对抗攻击的一种很有前途的方法。近年来，利用扩散概率模型的方法在图像分类任务中的对抗性净化方面取得了很大的成功。然而，这样的方法陷入了平衡去噪和信息保存需求的两难境地。本文指出，现有的基于扩散模型的对抗性净化方法在核心去噪过程中逐渐丢失样本信息，导致后续分类任务中偶尔出现标签漂移。作为一种补救措施，我们建议通过引入分类器置信度的指导来抑制这种信息损失。具体地说，我们提出了分类器置信度引导的净化(COUP)算法，该算法在净化对抗性实例的同时避免了分类器的决策边界。实验结果表明，在强攻击方式下，COUP算法具有较好的抗攻击能力。



## **6. Using Retriever Augmented Large Language Models for Attack Graph Generation**

使用检索器增强大型语言模型生成攻击图 cs.CR

**SubmitDate**: 2024-08-11    [abs](http://arxiv.org/abs/2408.05855v1) [paper-pdf](http://arxiv.org/pdf/2408.05855v1)

**Authors**: Renascence Tarafder Prapty, Ashish Kundu, Arun Iyengar

**Abstract**: As the complexity of modern systems increases, so does the importance of assessing their security posture through effective vulnerability management and threat modeling techniques. One powerful tool in the arsenal of cybersecurity professionals is the attack graph, a representation of all potential attack paths within a system that an adversary might exploit to achieve a certain objective. Traditional methods of generating attack graphs involve expert knowledge, manual curation, and computational algorithms that might not cover the entire threat landscape due to the ever-evolving nature of vulnerabilities and exploits. This paper explores the approach of leveraging large language models (LLMs), such as ChatGPT, to automate the generation of attack graphs by intelligently chaining Common Vulnerabilities and Exposures (CVEs) based on their preconditions and effects. It also shows how to utilize LLMs to create attack graphs from threat reports.

摘要: 随着现代系统复杂性的增加，通过有效的漏洞管理和威胁建模技术评估其安全态势的重要性也随之增加。网络安全专业人员武器库中的一个强大工具是攻击图，它代表了系统内对手可能利用的所有潜在攻击路径来实现特定目标。生成攻击图的传统方法涉及专家知识、手动策划和计算算法，由于漏洞和漏洞利用的不断变化的性质，这些方法可能无法覆盖整个威胁格局。本文探讨了利用ChatGPT等大型语言模型（LLM）来自动生成攻击图的方法，通过根据其先决条件和效果智能链接常见漏洞和暴露（CVE）。它还展示了如何利用LLM根据威胁报告创建攻击图。



## **7. A Diamond Model Analysis on Twitter's Biggest Hack**

Twitter最大黑客行为的钻石模型分析 cs.CR

**SubmitDate**: 2024-08-11    [abs](http://arxiv.org/abs/2306.15878v3) [paper-pdf](http://arxiv.org/pdf/2306.15878v3)

**Authors**: Chaitanya Rahalkar

**Abstract**: Cyberattacks have prominently increased over the past few years now, and have targeted actors from a wide variety of domains. Understanding the motivation, infrastructure, attack vectors, etc. behind such attacks is vital to proactively work against preventing such attacks in the future and also to analyze the economic and social impact of such attacks. In this paper, we leverage the diamond model to perform an intrusion analysis case study of the 2020 Twitter account hijacking Cyberattack. We follow this standardized incident response model to map the adversary, capability, infrastructure, and victim and perform a comprehensive analysis of the attack, and the impact posed by the attack from a Cybersecurity policy standpoint.

摘要: 过去几年，网络攻击显着增加，目标是来自各个领域的行为者。了解此类攻击背后的动机、基础设施、攻击载体等对于积极预防未来此类攻击以及分析此类攻击的经济和社会影响至关重要。在本文中，我们利用钻石模型对2020年Twitter帐户劫持网络攻击进行入侵分析案例研究。我们遵循这个标准化事件响应模型来绘制对手、能力、基础设施和受害者，并从网络安全政策的角度对攻击以及攻击造成的影响进行全面分析。



## **8. Improving Adversarial Transferability with Neighbourhood Gradient Information**

利用邻居梯度信息改善对抗可移植性 cs.CV

**SubmitDate**: 2024-08-11    [abs](http://arxiv.org/abs/2408.05745v1) [paper-pdf](http://arxiv.org/pdf/2408.05745v1)

**Authors**: Haijing Guo, Jiafeng Wang, Zhaoyu Chen, Kaixun Jiang, Lingyi Hong, Pinxue Guo, Jinglun Li, Wenqiang Zhang

**Abstract**: Deep neural networks (DNNs) are known to be susceptible to adversarial examples, leading to significant performance degradation. In black-box attack scenarios, a considerable attack performance gap between the surrogate model and the target model persists. This work focuses on enhancing the transferability of adversarial examples to narrow this performance gap. We observe that the gradient information around the clean image, i.e. Neighbourhood Gradient Information, can offer high transferability. Leveraging this, we propose the NGI-Attack, which incorporates Example Backtracking and Multiplex Mask strategies, to use this gradient information and enhance transferability fully. Specifically, we first adopt Example Backtracking to accumulate Neighbourhood Gradient Information as the initial momentum term. Multiplex Mask, which forms a multi-way attack strategy, aims to force the network to focus on non-discriminative regions, which can obtain richer gradient information during only a few iterations. Extensive experiments demonstrate that our approach significantly enhances adversarial transferability. Especially, when attacking numerous defense models, we achieve an average attack success rate of 95.8%. Notably, our method can plugin with any off-the-shelf algorithm to improve their attack performance without additional time cost.

摘要: 众所周知，深度神经网络(DNN)容易受到敌意示例的影响，导致性能显著下降。在黑盒攻击场景中，代理模型和目标模型之间存在着相当大的攻击性能差距。这项工作的重点是提高对抗性例子的可转移性，以缩小这一性能差距。我们观察到清洁图像周围的梯度信息，即邻域梯度信息，可以提供很高的可转移性。利用这一点，我们提出了NGI攻击，它结合了示例回溯和多重掩码策略，利用这种梯度信息，充分提高了可转移性。具体地说，我们首先采用示例回溯累积邻域梯度信息作为初始动量项。多重掩码是一种多途径攻击策略，旨在迫使网络聚焦于非歧视区域，使其只需几次迭代即可获得更丰富的梯度信息。大量的实验表明，我们的方法显著提高了对抗性转移能力。特别是对众多防御模型的攻击，平均攻击成功率达到95.8%。值得注意的是，我们的方法可以插入任何现成的算法来提高他们的攻击性能，而不需要额外的时间成本。



## **9. Graph Agent Network: Empowering Nodes with Decentralized Communications Capabilities for Adversarial Resilience**

图代理网络：赋予节点去中心化通信能力，以实现对抗复原力 cs.LG

**SubmitDate**: 2024-08-11    [abs](http://arxiv.org/abs/2306.06909v2) [paper-pdf](http://arxiv.org/pdf/2306.06909v2)

**Authors**: Ao Liu, Wenshan Li, Tao Li, Beibei Li, Guangquan Xu, Pan Zhou, Wengang Ma, Hanyuan Huang

**Abstract**: End-to-end training with global optimization have popularized graph neural networks (GNNs) for node classification, yet inadvertently introduced vulnerabilities to adversarial edge-perturbing attacks. Adversaries can exploit the inherent opened interfaces of GNNs' input and output, perturbing critical edges and thus manipulating the classification results. Current defenses, due to their persistent utilization of global-optimization-based end-to-end training schemes, inherently encapsulate the vulnerabilities of GNNs. This is specifically evidenced in their inability to defend against targeted secondary attacks. In this paper, we propose the Graph Agent Network (GAgN) to address the aforementioned vulnerabilities of GNNs. GAgN is a graph-structured agent network in which each node is designed as an 1-hop-view agent. Through the decentralized interactions between agents, they can learn to infer global perceptions to perform tasks including inferring embeddings, degrees and neighbor relationships for given nodes. This empowers nodes to filtering adversarial edges while carrying out classification tasks. Furthermore, agents' limited view prevents malicious messages from propagating globally in GAgN, thereby resisting global-optimization-based secondary attacks. We prove that single-hidden-layer multilayer perceptrons (MLPs) are theoretically sufficient to achieve these functionalities. Experimental results show that GAgN effectively implements all its intended capabilities and, compared to state-of-the-art defenses, achieves optimal classification accuracy on the perturbed datasets.

摘要: 具有全局优化的端到端训练普及了图神经网络(GNN)用于节点分类，但无意中引入了对敌意边缘扰动攻击的脆弱性。攻击者可以利用GNN输入和输出固有的开放接口，扰乱关键边缘，从而操纵分类结果。目前的防御措施由于持续使用基于全局优化的端到端培训方案，固有地封装了GNN的脆弱性。这一点具体表现在他们无法防御有针对性的二次攻击。在本文中，我们提出了图代理网络(GagN)来解决GNN的上述漏洞。GAGN是一个图结构的代理网络，其中每个节点被设计为一个1跳视图代理。通过代理之间的分散交互，它们可以学习推断全局感知来执行任务，包括推断给定节点的嵌入度、度数和邻居关系。这使节点能够在执行分类任务时过滤敌意边缘。此外，代理的有限视点防止恶意消息在GAGN中全局传播，从而抵抗基于全局优化的二次攻击。我们证明了单隐层多层感知器(MLP)理论上足以实现这些功能。实验结果表明，GAGN有效地实现了其预期的所有功能，并且与现有的防御措施相比，在扰动数据集上获得了最优的分类精度。



## **10. StealthDiffusion: Towards Evading Diffusion Forensic Detection through Diffusion Model**

StealthDistance：通过扩散模型避免扩散法医检测 cs.CV

**SubmitDate**: 2024-08-11    [abs](http://arxiv.org/abs/2408.05669v1) [paper-pdf](http://arxiv.org/pdf/2408.05669v1)

**Authors**: Ziyin Zhou, Ke Sun, Zhongxi Chen, Huafeng Kuang, Xiaoshuai Sun, Rongrong Ji

**Abstract**: The rapid progress in generative models has given rise to the critical task of AI-Generated Content Stealth (AIGC-S), which aims to create AI-generated images that can evade both forensic detectors and human inspection. This task is crucial for understanding the vulnerabilities of existing detection methods and developing more robust techniques. However, current adversarial attacks often introduce visible noise, have poor transferability, and fail to address spectral differences between AI-generated and genuine images. To address this, we propose StealthDiffusion, a framework based on stable diffusion that modifies AI-generated images into high-quality, imperceptible adversarial examples capable of evading state-of-the-art forensic detectors. StealthDiffusion comprises two main components: Latent Adversarial Optimization, which generates adversarial perturbations in the latent space of stable diffusion, and Control-VAE, a module that reduces spectral differences between the generated adversarial images and genuine images without affecting the original diffusion model's generation process. Extensive experiments show that StealthDiffusion is effective in both white-box and black-box settings, transforming AI-generated images into high-quality adversarial forgeries with frequency spectra similar to genuine images. These forgeries are classified as genuine by advanced forensic classifiers and are difficult for humans to distinguish.

摘要: 生成模型的快速发展催生了人工智能生成内容隐身(AIGC-S)的关键任务，该任务旨在创建能够躲避法医检测器和人类检查的人工智能生成图像。这项任务对于了解现有检测方法的漏洞和开发更健壮的技术至关重要。然而，目前的对抗性攻击经常引入可见噪声，可转移性较差，并且无法解决人工智能生成的图像和真实图像之间的光谱差异。为了解决这个问题，我们提出了StealthDiffation，这是一个基于稳定扩散的框架，它将人工智能生成的图像修改为能够躲避最先进的法医检测器的高质量、不可察觉的对抗性样本。隐写扩散包括两个主要组成部分：潜在对抗性优化，它在稳定扩散的潜在空间中产生对抗性扰动；以及控制-VAE，它在不影响原始扩散模型的生成过程的情况下，减少生成的对抗性图像和真实图像之间的光谱差异。大量的实验表明，该算法在白盒和黑盒环境下都是有效的，可以将人工智能生成的图像转换成频谱与真实图像相似的高质量对抗性伪造图像。这些赝品被高级法医分类器归类为真品，人类很难区分。



## **11. Utilizing Large Language Models to Optimize the Detection and Explainability of Phishing Websites**

利用大型语言模型优化网络钓鱼网站的检测和解释性 cs.CR

**SubmitDate**: 2024-08-11    [abs](http://arxiv.org/abs/2408.05667v1) [paper-pdf](http://arxiv.org/pdf/2408.05667v1)

**Authors**: Sayak Saha Roy, Shirin Nilizadeh

**Abstract**: In this paper, we introduce PhishLang, an open-source, lightweight Large Language Model (LLM) specifically designed for phishing website detection through contextual analysis of the website. Unlike traditional heuristic or machine learning models that rely on static features and struggle to adapt to new threats and deep learning models that are computationally intensive, our model utilizes the advanced language processing capabilities of LLMs to learn granular features that are characteristic of phishing attacks. Furthermore, PhishLang operates with minimal data preprocessing and offers performance comparable to leading deep learning tools, while being significantly faster and less resource-intensive. Over a 3.5-month testing period, PhishLang successfully identified approximately 26K phishing URLs, many of which were undetected by popular antiphishing blocklists, thus demonstrating its potential to aid current detection measures. We also evaluate PhishLang against several realistic adversarial attacks and develop six patches that make it very robust against such threats. Furthermore, we integrate PhishLang with GPT-3.5 Turbo to create \textit{explainable blocklisting} - warnings that provide users with contextual information about different features that led to a website being marked as phishing. Finally, we have open-sourced the PhishLang framework and developed a Chromium-based browser extension and URL scanner website, which implement explainable warnings for end-users.

摘要: 在本文中，我们介绍了PhishLang，一个开源的，轻量级的大型语言模型(LLM)，专门为钓鱼网站检测而设计的，通过对网站的上下文分析。与传统的启发式或机器学习模型依赖静态特征并难以适应计算密集型的新威胁和深度学习模型不同，我们的模型利用LLMS的高级语言处理能力来学习钓鱼攻击的细粒度特征。此外，PhishLang的操作只需最少的数据预处理，并提供可与领先的深度学习工具相媲美的性能，同时速度更快，资源消耗更少。在3.5个月的测试期内，PhishLang成功识别了大约26K个钓鱼URL，其中许多都没有被流行的反钓鱼阻止列表检测到，从而展示了其帮助当前检测措施的潜力。我们还评估了Phishlang对几种现实对手攻击的抵抗力，并开发了六个补丁，使其对此类威胁非常健壮。此外，我们将PhishLang与GPT-3.5 Turbo集成，以创建\Text{可解释的阻止列表}-警告，向用户提供有关导致网站被标记为网络钓鱼的不同功能的上下文信息。最后，我们开源了PhishLang框架，并开发了一个基于Chromium的浏览器扩展和URL扫描器网站，为最终用户实现了可解释的警告。



## **12. Cooperative Abnormal Node Detection with Adversary Resistance**

具有成瘾抵抗力的协作异常节点检测 eess.SY

12 pages, 12 figures

**SubmitDate**: 2024-08-10    [abs](http://arxiv.org/abs/2311.16661v2) [paper-pdf](http://arxiv.org/pdf/2311.16661v2)

**Authors**: Yingying Huangfu, Tian Bai

**Abstract**: This paper presents a novel probabilistic detection scheme called Cooperative Statistical Detection~(CSD) for abnormal node detection while defending against adversarial attacks in cluster-tree wireless sensor networks. The CSD performs a two-phase process: 1) designing a likelihood ratio test~(LRT) for a non-root node at its children from the perspective of packet loss; 2) making an overall decision at the root node based on the aggregated detection data of the nodes over tree branches. In most adversarial scenarios, malicious children knowing the detection policy can generate falsified data to protect the abnormal parent from being detected. To resolve this issue, a mechanism is presented in the CSD to remove untrustworthy information. Through theoretical analysis, we show that the LRT-based method achieves the perfect detection. Furthermore, the optimal removal threshold is derived for falsifications with uncertain strategies and guarantees perfect detection of the CSD. As our simulation results shown, the CSD approach is robust to falsifications and can rapidly reach $99\%$ detection accuracy, even in existing adversarial scenarios, which outperforms the state-of-the-art technology.

摘要: 提出了一种新的概率检测方案--协同统计检测(CSD)，用于检测簇树无线传感器网络中的异常节点，同时防止恶意攻击。CSD分两个阶段进行：1)从丢包的角度为子节点上的非根节点设计似然比检验(LRT)；2)在根节点上根据树枝上节点的聚合检测数据做出总体决策。在大多数对抗性场景中，知道检测策略的恶意儿童可以生成伪造数据，以保护异常父母不被检测到。为了解决这个问题，CSD中提出了一种删除不可信信息的机制。通过理论分析表明，基于LRT的检测方法达到了较好的检测效果。此外，对于具有不确定策略的篡改，给出了最优去除阈值，并保证了对CSD的完美检测。仿真结果表明，CSD方法对篡改具有较强的鲁棒性，即使在现有的对抗性场景中，也能快速达到99美元的检测精度，其性能优于最先进的技术。



## **13. ReToMe-VA: Recursive Token Merging for Video Diffusion-based Unrestricted Adversarial Attack**

ReToMe-VA：针对基于视频扩散的无限制对抗攻击的回归令牌合并 cs.CV

**SubmitDate**: 2024-08-10    [abs](http://arxiv.org/abs/2408.05479v1) [paper-pdf](http://arxiv.org/pdf/2408.05479v1)

**Authors**: Ziyi Gao, Kai Chen, Zhipeng Wei, Tingshu Mou, Jingjing Chen, Zhiyu Tan, Hao Li, Yu-Gang Jiang

**Abstract**: Recent diffusion-based unrestricted attacks generate imperceptible adversarial examples with high transferability compared to previous unrestricted attacks and restricted attacks. However, existing works on diffusion-based unrestricted attacks are mostly focused on images yet are seldom explored in videos. In this paper, we propose the Recursive Token Merging for Video Diffusion-based Unrestricted Adversarial Attack (ReToMe-VA), which is the first framework to generate imperceptible adversarial video clips with higher transferability. Specifically, to achieve spatial imperceptibility, ReToMe-VA adopts a Timestep-wise Adversarial Latent Optimization (TALO) strategy that optimizes perturbations in diffusion models' latent space at each denoising step. TALO offers iterative and accurate updates to generate more powerful adversarial frames. TALO can further reduce memory consumption in gradient computation. Moreover, to achieve temporal imperceptibility, ReToMe-VA introduces a Recursive Token Merging (ReToMe) mechanism by matching and merging tokens across video frames in the self-attention module, resulting in temporally consistent adversarial videos. ReToMe concurrently facilitates inter-frame interactions into the attack process, inducing more diverse and robust gradients, thus leading to better adversarial transferability. Extensive experiments demonstrate the efficacy of ReToMe-VA, particularly in surpassing state-of-the-art attacks in adversarial transferability by more than 14.16% on average.

摘要: 最近的基于扩散的无限制攻击产生了不可察觉的对抗性例子，与以前的无限制攻击和受限攻击相比，具有很高的可转移性。然而，现有的基于扩散的无限制攻击的研究大多集中在图像上，而很少在视频中进行研究。本文提出了基于视频扩散的递归令牌合并算法(RetoMe-VA)，这是第一个生成具有较高可转移性的不可感知的恶意视频片段的框架。具体地说，为了实现空间不可见性，ReToMe-VA采用了一种时间步长的对抗性潜在优化(TALO)策略，该策略在每个去噪步骤优化扩散模型潜在空间的扰动。Talo提供迭代和准确的更新，以生成更强大的对抗性帧。TALO可以进一步减少梯度计算中的内存消耗。此外，为了实现时间不可感知性，ReToMe-VA引入了递归令牌合并(ReToMe)机制，通过在自我注意模块中跨视频帧匹配和合并令牌，从而产生时间一致的对抗性视频。ReToMe同时促进了攻击过程中的帧间交互，导致了更多样化和更健壮的梯度，从而导致更好的对手可转移性。广泛的实验证明了ReToMe-VA的有效性，特别是在对抗性可转移性方面，平均超过了14.16%的最新攻击。



## **14. Towards Resilient and Efficient LLMs: A Comparative Study of Efficiency, Performance, and Adversarial Robustness**

迈向弹性和高效的法学硕士：效率、绩效和对抗稳健性的比较研究 cs.CL

**SubmitDate**: 2024-08-09    [abs](http://arxiv.org/abs/2408.04585v2) [paper-pdf](http://arxiv.org/pdf/2408.04585v2)

**Authors**: Xiaojing Fan, Chunliang Tao

**Abstract**: With the increasing demand for practical applications of Large Language Models (LLMs), many attention-efficient models have been developed to balance performance and computational cost. However, the adversarial robustness of these models remains under-explored. In this work, we design a framework to investigate the trade-off between efficiency, performance, and adversarial robustness of LLMs by comparing three prominent models with varying levels of complexity and efficiency -- Transformer++, Gated Linear Attention (GLA) Transformer, and MatMul-Free LM -- utilizing the GLUE and AdvGLUE datasets. The AdvGLUE dataset extends the GLUE dataset with adversarial samples designed to challenge model robustness. Our results show that while the GLA Transformer and MatMul-Free LM achieve slightly lower accuracy on GLUE tasks, they demonstrate higher efficiency and either superior or comparative robustness on AdvGLUE tasks compared to Transformer++ across different attack levels. These findings highlight the potential of simplified architectures to achieve a compelling balance between efficiency, performance, and adversarial robustness, offering valuable insights for applications where resource constraints and resilience to adversarial attacks are critical.

摘要: 随着大型语言模型的实际应用需求的增加，人们已经开发了许多注意力高效的模型来平衡性能和计算成本。然而，这些模型的对抗性稳健性仍然没有得到充分的研究。在这项工作中，我们设计了一个框架来研究LLMS的效率、性能和对抗健壮性之间的权衡，方法是利用GLUE和AdvGLUE数据集比较三种不同复杂度和效率的重要模型--Transformer++、GLA Transformer和Matmul-Free LM。AdvGLUE数据集使用旨在挑战模型稳健性的对抗性样本扩展了GLUE数据集。我们的结果表明，虽然GLA Transformer和MatMul-Free LM在粘合任务上的准确率略低，但在不同攻击级别上，它们在AdvGLUE任务上表现出比Transformer++更高的效率和更好的健壮性或相对较高的稳健性。这些发现突出了简化体系结构在效率、性能和对手攻击健壮性之间实现引人注目的平衡的潜力，为资源约束和对抗攻击的弹性至关重要的应用程序提供了宝贵的见解。



## **15. Modeling Electromagnetic Signal Injection Attacks on Camera-based Smart Systems: Applications and Mitigation**

对基于相机的智能系统进行电磁信号注入攻击建模：应用和缓解 cs.CR

13 pages, 10 figures, 4 tables

**SubmitDate**: 2024-08-09    [abs](http://arxiv.org/abs/2408.05124v1) [paper-pdf](http://arxiv.org/pdf/2408.05124v1)

**Authors**: Youqian Zhang, Michael Cheung, Chunxi Yang, Xinwei Zhai, Zitong Shen, Xinyu Ji, Eugene Y. Fu, Sze-Yiu Chau, Xiapu Luo

**Abstract**: Numerous safety- or security-critical systems depend on cameras to perceive their surroundings, further allowing artificial intelligence (AI) to analyze the captured images to make important decisions. However, a concerning attack vector has emerged, namely, electromagnetic waves, which pose a threat to the integrity of these systems. Such attacks enable attackers to manipulate the images remotely, leading to incorrect AI decisions, e.g., autonomous vehicles missing detecting obstacles ahead resulting in collisions. The lack of understanding regarding how different systems react to such attacks poses a significant security risk. Furthermore, no effective solutions have been demonstrated to mitigate this threat.   To address these gaps, we modeled the attacks and developed a simulation method for generating adversarial images. Through rigorous analysis, we confirmed that the effects of the simulated adversarial images are indistinguishable from those from real attacks. This method enables researchers and engineers to rapidly assess the susceptibility of various AI vision applications to these attacks, without the need for constructing complicated attack devices. In our experiments, most of the models demonstrated vulnerabilities to these attacks, emphasizing the need to enhance their robustness. Fortunately, our modeling and simulation method serves as a stepping stone toward developing more resilient models. We present a pilot study on adversarial training to improve their robustness against attacks, and our results demonstrate a significant improvement by recovering up to 91% performance, offering a promising direction for mitigating this threat.

摘要: 许多安全或安全关键系统依赖摄像头来感知周围环境，进一步允许人工智能(AI)分析捕获的图像以做出重要决策。然而，出现了一个令人担忧的攻击载体，即电磁波，它对这些系统的完整性构成了威胁。这类攻击使攻击者能够远程操纵图像，导致不正确的人工智能决策，例如，自动驾驶车辆错过了检测到前方障碍物导致碰撞的情况。缺乏对不同系统如何应对此类攻击的了解构成了重大的安全风险。此外，没有任何有效的解决方案被证明可以减轻这一威胁。为了弥补这些差距，我们对攻击进行了建模，并开发了一种生成对抗性图像的模拟方法。通过严格的分析，我们证实了模拟对抗性图像的效果与真实攻击的效果是难以区分的。这种方法使研究人员和工程师能够快速评估各种人工智能视觉应用程序对这些攻击的敏感度，而不需要构建复杂的攻击设备。在我们的实验中，大多数模型对这些攻击都表现出了脆弱性，强调了增强其健壮性的必要性。幸运的是，我们的建模和仿真方法是开发更具弹性的模型的垫脚石。我们给出了一个对抗性训练的初步研究，以提高他们对攻击的健壮性，我们的结果显示出显著的改进，恢复了高达91%的性能，为缓解这种威胁提供了一个有希望的方向。



## **16. PriPHiT: Privacy-Preserving Hierarchical Training of Deep Neural Networks**

PriPhiT：深度神经网络的隐私保护分层训练 cs.CV

16 pages, 16 figures, 6 tables

**SubmitDate**: 2024-08-09    [abs](http://arxiv.org/abs/2408.05092v1) [paper-pdf](http://arxiv.org/pdf/2408.05092v1)

**Authors**: Yamin Sepehri, Pedram Pad, Pascal Frossard, L. Andrea Dunbar

**Abstract**: The training phase of deep neural networks requires substantial resources and as such is often performed on cloud servers. However, this raises privacy concerns when the training dataset contains sensitive content, e.g., face images. In this work, we propose a method to perform the training phase of a deep learning model on both an edge device and a cloud server that prevents sensitive content being transmitted to the cloud while retaining the desired information. The proposed privacy-preserving method uses adversarial early exits to suppress the sensitive content at the edge and transmits the task-relevant information to the cloud. This approach incorporates noise addition during the training phase to provide a differential privacy guarantee. We extensively test our method on different facial datasets with diverse face attributes using various deep learning architectures, showcasing its outstanding performance. We also demonstrate the effectiveness of privacy preservation through successful defenses against different white-box and deep reconstruction attacks.

摘要: 深度神经网络的训练阶段需要大量资源，因此通常在云服务器上执行。然而，当训练数据集包含敏感内容(例如，面部图像)时，这会引起隐私问题。在这项工作中，我们提出了一种方法，在边缘设备和云服务器上执行深度学习模型的训练阶段，以防止敏感数据被传输到云中，同时保留所需的信息。提出的隐私保护方法使用对抗性的提前退出来抑制边缘敏感内容，并将与任务相关的信息传输到云中。这种方法在训练阶段加入了噪声，以提供不同的隐私保证。我们使用不同的深度学习结构在具有不同人脸属性的不同人脸数据集上对我们的方法进行了广泛的测试，展示了其出色的性能。我们还通过成功地防御不同的白盒攻击和深度重构攻击来展示隐私保护的有效性。



## **17. XNN: Paradigm Shift in Mitigating Identity Leakage within Cloud-Enabled Deep Learning**

XNN：减轻云深度学习中身份泄露的范式转变 cs.CR

**SubmitDate**: 2024-08-09    [abs](http://arxiv.org/abs/2408.04974v1) [paper-pdf](http://arxiv.org/pdf/2408.04974v1)

**Authors**: Kaixin Liu, Huixin Xiong, Bingyu Duan, Zexuan Cheng, Xinyu Zhou, Wanqian Zhang, Xiangyu Zhang

**Abstract**: In the domain of cloud-based deep learning, the imperative for external computational resources coexists with acute privacy concerns, particularly identity leakage. To address this challenge, we introduce XNN and XNN-d, pioneering methodologies that infuse neural network features with randomized perturbations, striking a harmonious balance between utility and privacy. XNN, designed for the training phase, ingeniously blends random permutation with matrix multiplication techniques to obfuscate feature maps, effectively shielding private data from potential breaches without compromising training integrity. Concurrently, XNN-d, devised for the inference phase, employs adversarial training to integrate generative adversarial noise. This technique effectively counters black-box access attacks aimed at identity extraction, while a distilled face recognition network adeptly processes the perturbed features, ensuring accurate identification. Our evaluation demonstrates XNN's effectiveness, significantly outperforming existing methods in reducing identity leakage while maintaining a high model accuracy.

摘要: 在基于云的深度学习领域，对外部计算资源的迫切需求与严重的隐私问题共存，尤其是身份泄露。为了应对这一挑战，我们引入了XNN和XNN-d，这两种开创性的方法将神经网络的特征注入随机扰动，在效用和隐私之间取得了和谐的平衡。XNN是为训练阶段设计的，它巧妙地将随机置换与矩阵乘法技术相结合，以混淆特征地图，有效地保护私人数据免受潜在入侵，而不会影响训练完整性。同时，XNN-d是为推理阶段设计的，它使用对抗性训练来整合生成性对抗性噪声。这项技术有效地对抗了旨在提取身份的黑匣子访问攻击，而提取的人脸识别网络熟练地处理了扰动的特征，确保了准确的识别。我们的评估表明了XNN的有效性，在保持较高的模型精度的同时，在减少身份泄漏方面显著优于现有方法。



## **18. LiD-FL: Towards List-Decodable Federated Learning**

LiD-FL：迈向列表可解码联邦学习 cs.LG

17 pages, 5 figures

**SubmitDate**: 2024-08-09    [abs](http://arxiv.org/abs/2408.04963v1) [paper-pdf](http://arxiv.org/pdf/2408.04963v1)

**Authors**: Hong Liu, Liren Shan, Han Bao, Ronghui You, Yuhao Yi, Jiancheng Lv

**Abstract**: Federated learning is often used in environments with many unverified participants. Therefore, federated learning under adversarial attacks receives significant attention. This paper proposes an algorithmic framework for list-decodable federated learning, where a central server maintains a list of models, with at least one guaranteed to perform well. The framework has no strict restriction on the fraction of honest workers, extending the applicability of Byzantine federated learning to the scenario with more than half adversaries. Under proper assumptions on the loss function, we prove a convergence theorem for our method. Experimental results, including image classification tasks with both convex and non-convex losses, demonstrate that the proposed algorithm can withstand the malicious majority under various attacks.

摘要: 联邦学习通常用于有许多未经验证的参与者的环境中。因此，对抗攻击下的联邦学习受到了高度关注。本文提出了一种用于列表可解码联邦学习的算法框架，其中中央服务器维护一个模型列表，至少有一个模型保证表现良好。该框架对诚实工人的比例没有严格限制，将拜占庭联邦学习的适用性扩展到有一半以上对手的场景。在对损失函数的适当假设下，我们证明了我们方法的收敛性定理。实验结果（包括具有凸损失和非凸损失的图像分类任务）表明，所提出的算法可以抵御各种攻击下的恶意多数。



## **19. Adversarially Robust Industrial Anomaly Detection Through Diffusion Model**

通过扩散模型进行反向稳健的工业异常检测 cs.LG

**SubmitDate**: 2024-08-09    [abs](http://arxiv.org/abs/2408.04839v1) [paper-pdf](http://arxiv.org/pdf/2408.04839v1)

**Authors**: Yuanpu Cao, Lu Lin, Jinghui Chen

**Abstract**: Deep learning-based industrial anomaly detection models have achieved remarkably high accuracy on commonly used benchmark datasets. However, the robustness of those models may not be satisfactory due to the existence of adversarial examples, which pose significant threats to the practical deployment of deep anomaly detectors. Recently, it has been shown that diffusion models can be used to purify the adversarial noises and thus build a robust classifier against adversarial attacks. Unfortunately, we found that naively applying this strategy in anomaly detection (i.e., placing a purifier before an anomaly detector) will suffer from a high anomaly miss rate since the purifying process can easily remove both the anomaly signal and the adversarial perturbations, causing the later anomaly detector failed to detect anomalies. To tackle this issue, we explore the possibility of performing anomaly detection and adversarial purification simultaneously. We propose a simple yet effective adversarially robust anomaly detection method, \textit{AdvRAD}, that allows the diffusion model to act both as an anomaly detector and adversarial purifier. We also extend our proposed method for certified robustness to $l_2$ norm bounded perturbations. Through extensive experiments, we show that our proposed method exhibits outstanding (certified) adversarial robustness while also maintaining equally strong anomaly detection performance on par with the state-of-the-art methods on industrial anomaly detection benchmark datasets.

摘要: 基于深度学习的工业异常检测模型在常用的基准数据集上取得了非常高的准确率。然而，由于对抗性例子的存在，这些模型的稳健性可能并不令人满意，这对深度异常检测器的实际部署构成了巨大的威胁。最近的研究表明，扩散模型可以用来净化敌方噪声，从而建立一个针对敌方攻击的稳健的分类器。不幸的是，我们发现，在异常检测中幼稚地应用这一策略(即在异常检测器之前放置净化器)将遭受高异常错失率，因为净化过程可以很容易地去除异常信号和敌对扰动，导致后来的异常检测器无法检测到异常。为了解决这个问题，我们探索了同时执行异常检测和恶意净化的可能性。我们提出了一种简单而有效的敌意鲁棒异常检测方法，它允许扩散模型既充当异常检测器又充当敌意净化器。我们还将我们提出的证明鲁棒性的方法推广到$L_2$范数有界扰动。通过大量的实验表明，我们提出的方法在保持与工业异常检测基准数据集相同的强大异常检测性能的同时，表现出了出色的(经过认证的)对手攻击的健壮性。



## **20. h4rm3l: A Dynamic Benchmark of Composable Jailbreak Attacks for LLM Safety Assessment**

h4 rm3l：LLM安全评估的可组合越狱攻击的动态基准 cs.CR

**SubmitDate**: 2024-08-09    [abs](http://arxiv.org/abs/2408.04811v1) [paper-pdf](http://arxiv.org/pdf/2408.04811v1)

**Authors**: Moussa Koulako Bala Doumbouya, Ananjan Nandi, Gabriel Poesia, Davide Ghilardi, Anna Goldie, Federico Bianchi, Dan Jurafsky, Christopher D. Manning

**Abstract**: The safety of Large Language Models (LLMs) remains a critical concern due to a lack of adequate benchmarks for systematically evaluating their ability to resist generating harmful content. Previous efforts towards automated red teaming involve static or templated sets of illicit requests and adversarial prompts which have limited utility given jailbreak attacks' evolving and composable nature. We propose a novel dynamic benchmark of composable jailbreak attacks to move beyond static datasets and taxonomies of attacks and harms. Our approach consists of three components collectively called h4rm3l: (1) a domain-specific language that formally expresses jailbreak attacks as compositions of parameterized prompt transformation primitives, (2) bandit-based few-shot program synthesis algorithms that generate novel attacks optimized to penetrate the safety filters of a target black box LLM, and (3) open-source automated red-teaming software employing the previous two components. We use h4rm3l to generate a dataset of 2656 successful novel jailbreak attacks targeting 6 state-of-the-art (SOTA) open-source and proprietary LLMs. Several of our synthesized attacks are more effective than previously reported ones, with Attack Success Rates exceeding 90% on SOTA closed language models such as claude-3-haiku and GPT4-o. By generating datasets of jailbreak attacks in a unified formal representation, h4rm3l enables reproducible benchmarking and automated red-teaming, contributes to understanding LLM safety limitations, and supports the development of robust defenses in an increasingly LLM-integrated world.   Warning: This paper and related research artifacts contain offensive and potentially disturbing prompts and model-generated content.

摘要: 大型语言模型(LLM)的安全性仍然是一个严重的问题，因为缺乏系统地评估它们抵抗产生有害内容的能力的适当基准。以前的自动化红色团队的努力包括静态的或模板化的非法请求集和对抗性提示，鉴于越狱攻击不断演变和可组合的性质，这些提示的效用有限。我们提出了一种新的可组合越狱攻击的动态基准，以超越静态数据集和攻击和危害的分类。我们的方法由三个组件组成，统称为h4rm3l：(1)特定于领域的语言，它将越狱攻击形式化地表达为参数化提示转换原语的组合；(2)基于盗贼的少发程序合成算法，它生成经过优化的新型攻击，以穿透目标黑盒LLM的安全过滤器；以及(3)使用前两个组件的开源自动红队软件。我们使用h4rm3l生成了一个2656个成功的新型越狱攻击的数据集，目标是6个最先进的开源和专有LLM。我们的几个合成攻击比以前报道的更有效，在Claude-3-haiku和GPT4-o等Sota封闭语言模型上的攻击成功率超过90%。通过以统一的形式表示生成越狱攻击的数据集，h4rm3l实现了可重现的基准测试和自动化的红团队，有助于了解LLM的安全限制，并支持在日益集成LLM的世界中开发强大的防御措施。警告：本文和相关研究文章包含冒犯性和潜在令人不安的提示和模型生成的内容。



## **21. Tamper-Resistant Safeguards for Open-Weight LLMs**

开放重量LLM的防篡改保障措施 cs.LG

Website: https://www.tamper-resistant-safeguards.com

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2408.00761v2) [paper-pdf](http://arxiv.org/pdf/2408.00761v2)

**Authors**: Rishub Tamirisa, Bhrugu Bharathi, Long Phan, Andy Zhou, Alice Gatti, Tarun Suresh, Maxwell Lin, Justin Wang, Rowan Wang, Ron Arel, Andy Zou, Dawn Song, Bo Li, Dan Hendrycks, Mantas Mazeika

**Abstract**: Rapid advances in the capabilities of large language models (LLMs) have raised widespread concerns regarding their potential for malicious use. Open-weight LLMs present unique challenges, as existing safeguards lack robustness to tampering attacks that modify model weights. For example, recent works have demonstrated that refusal and unlearning safeguards can be trivially removed with a few steps of fine-tuning. These vulnerabilities necessitate new approaches for enabling the safe release of open-weight LLMs. We develop a method, called TAR, for building tamper-resistant safeguards into open-weight LLMs such that adversaries cannot remove the safeguards even after thousands of steps of fine-tuning. In extensive evaluations and red teaming analyses, we find that our method greatly improves tamper-resistance while preserving benign capabilities. Our results demonstrate that tamper-resistance is a tractable problem, opening up a promising new avenue to improve the safety and security of open-weight LLMs.

摘要: 大型语言模型(LLM)功能的快速发展引起了人们对其潜在恶意使用的广泛关注。开放重量LLM提出了独特的挑战，因为现有的保障措施缺乏对篡改模型权重的篡改攻击的稳健性。例如，最近的研究表明，通过几个步骤的微调，就可以很容易地消除拒绝和遗忘的保障措施。这些漏洞需要新的方法来实现安全释放未加重量的低密度脂蛋白。我们开发了一种名为TAR的方法，用于在开放重量的LLM中构建防篡改保护措施，以便对手即使在数千个步骤的微调之后也无法移除这些保护措施。在广泛的评估和红团队分析中，我们发现我们的方法在保持良性性能的同时大大提高了防篡改能力。我们的结果表明，防篡改是一个容易解决的问题，为提高开重LLMS的安全性开辟了一条很有前途的新途径。



## **22. Improving Network Interpretability via Explanation Consistency Evaluation**

通过解释一致性评估提高网络可解释性 cs.CV

To appear in IEEE Transactions on Multimedia

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2408.04600v1) [paper-pdf](http://arxiv.org/pdf/2408.04600v1)

**Authors**: Hefeng Wu, Hao Jiang, Keze Wang, Ziyi Tang, Xianghuan He, Liang Lin

**Abstract**: While deep neural networks have achieved remarkable performance, they tend to lack transparency in prediction. The pursuit of greater interpretability in neural networks often results in a degradation of their original performance. Some works strive to improve both interpretability and performance, but they primarily depend on meticulously imposed conditions. In this paper, we propose a simple yet effective framework that acquires more explainable activation heatmaps and simultaneously increase the model performance, without the need for any extra supervision. Specifically, our concise framework introduces a new metric, i.e., explanation consistency, to reweight the training samples adaptively in model learning. The explanation consistency metric is utilized to measure the similarity between the model's visual explanations of the original samples and those of semantic-preserved adversarial samples, whose background regions are perturbed by using image adversarial attack techniques. Our framework then promotes the model learning by paying closer attention to those training samples with a high difference in explanations (i.e., low explanation consistency), for which the current model cannot provide robust interpretations. Comprehensive experimental results on various benchmarks demonstrate the superiority of our framework in multiple aspects, including higher recognition accuracy, greater data debiasing capability, stronger network robustness, and more precise localization ability on both regular networks and interpretable networks. We also provide extensive ablation studies and qualitative analyses to unveil the detailed contribution of each component.

摘要: 尽管深度神经网络取得了显著的性能，但它们在预测方面往往缺乏透明度。在神经网络中追求更高的可解释性往往会导致其原始性能的下降。一些作品努力提高可解释性和表现力，但它们主要取决于精心设定的条件。在本文中，我们提出了一个简单而有效的框架，它获得了更多可解释的激活热图，同时提高了模型的性能，而不需要任何额外的监督。具体地说，我们的简明框架引入了一种新的度量，即解释一致性，以在模型学习中自适应地重新加权训练样本。使用解释一致性度量来度量模型对原始样本的视觉解释与使用图像对抗攻击技术扰动背景区域的语义保留的对抗性样本的视觉解释之间的相似性。然后，我们的框架通过更密切地关注那些解释差异较大(即解释一致性较低)的训练样本来促进模型学习，而当前的模型无法提供稳健的解释。在不同基准上的综合实验结果表明，该框架在多个方面具有优势，包括更高的识别准确率、更强的数据去偏能力、更强的网络稳健性以及在规则网络和可解释网络上更精确的定位能力。我们还提供了广泛的烧蚀研究和定性分析，以揭示每个组件的详细贡献。



## **23. Ensemble everything everywhere: Multi-scale aggregation for adversarial robustness**

包容无处不在的一切：多规模聚合以实现对抗稳健性 cs.CV

34 pages, 25 figures, appendix

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2408.05446v1) [paper-pdf](http://arxiv.org/pdf/2408.05446v1)

**Authors**: Stanislav Fort, Balaji Lakshminarayanan

**Abstract**: Adversarial examples pose a significant challenge to the robustness, reliability and alignment of deep neural networks. We propose a novel, easy-to-use approach to achieving high-quality representations that lead to adversarial robustness through the use of multi-resolution input representations and dynamic self-ensembling of intermediate layer predictions. We demonstrate that intermediate layer predictions exhibit inherent robustness to adversarial attacks crafted to fool the full classifier, and propose a robust aggregation mechanism based on Vickrey auction that we call \textit{CrossMax} to dynamically ensemble them. By combining multi-resolution inputs and robust ensembling, we achieve significant adversarial robustness on CIFAR-10 and CIFAR-100 datasets without any adversarial training or extra data, reaching an adversarial accuracy of $\approx$72% (CIFAR-10) and $\approx$48% (CIFAR-100) on the RobustBench AutoAttack suite ($L_\infty=8/255)$ with a finetuned ImageNet-pretrained ResNet152. This represents a result comparable with the top three models on CIFAR-10 and a +5 % gain compared to the best current dedicated approach on CIFAR-100. Adding simple adversarial training on top, we get $\approx$78% on CIFAR-10 and $\approx$51% on CIFAR-100, improving SOTA by 5 % and 9 % respectively and seeing greater gains on the harder dataset. We validate our approach through extensive experiments and provide insights into the interplay between adversarial robustness, and the hierarchical nature of deep representations. We show that simple gradient-based attacks against our model lead to human-interpretable images of the target classes as well as interpretable image changes. As a byproduct, using our multi-resolution prior, we turn pre-trained classifiers and CLIP models into controllable image generators and develop successful transferable attacks on large vision language models.

摘要: 对抗性的例子对深度神经网络的稳健性、可靠性和对齐提出了巨大的挑战。我们提出了一种新颖的、易于使用的方法，通过使用多分辨率输入表示和中间层预测的动态自集成来获得高质量的表示，从而导致对抗性健壮性。我们证明了中间层预测对于欺骗完整分类器的敌意攻击表现出固有的稳健性，并提出了一种基于Vickrey拍卖的健壮聚集机制，我们称之为\textit{CRossmax}来动态集成它们。通过结合多分辨率输入和稳健集成，我们在CIFAR-10和CIFAR-100数据集上实现了显著的对抗稳健性，而无需任何对抗性训练或额外数据，在RobustBack AutoAttack套件($L_\INFTY=8/255)$上达到了约$\\72%(CIFAR-10)和$\\约48%(CIFAR-100)的对抗准确率。这代表了一个可以与CIFAR-10上的前三个型号相媲美的结果，并且与CIFAR-100上当前最好的专用方法相比，增加了5%。加上简单的对抗性训练，我们在CIFAR-10上获得了约78%的收益，在CIFAR-100上获得了约51%的收益，分别将SOTA提高了5%和9%，并在较难的数据集上看到了更大的收益。我们通过广泛的实验验证了我们的方法，并对对手健壮性和深层表示的层次性之间的相互作用提供了见解。我们表明，对我们的模型的简单的基于梯度的攻击会导致目标类的人类可解释的图像以及可解释的图像变化。作为一个副产品，我们利用我们的多分辨率先验知识，将预先训练的分类器和剪辑模型转化为可控的图像生成器，并成功地开发了对大型视觉语言模型的可转移攻击。



## **24. Understanding the Security Benefits and Overheads of Emerging Industry Solutions to DRAM Read Disturbance**

了解新兴行业解决方案的安全优势和管理费用针对内存读取干扰 cs.CR

To appear in DRAMSec 2024

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2406.19094v3) [paper-pdf](http://arxiv.org/pdf/2406.19094v3)

**Authors**: Oğuzhan Canpolat, A. Giray Yağlıkçı, Geraldo F. Oliveira, Ataberk Olgun, Oğuz Ergin, Onur Mutlu

**Abstract**: We present the first rigorous security, performance, energy, and cost analyses of the state-of-the-art on-DRAM-die read disturbance mitigation method, Per Row Activation Counting (PRAC), described in JEDEC DDR5 specification's April 2024 update. Unlike prior state-of-the-art that advises the memory controller to periodically issue refresh management (RFM) commands, which provides the DRAM chip with time to perform refreshes, PRAC introduces a new back-off signal. PRAC's back-off signal propagates from the DRAM chip to the memory controller and forces the memory controller to 1) stop serving requests and 2) issue RFM commands. As a result, RFM commands are issued when needed as opposed to periodically, reducing RFM's overheads. We analyze PRAC in four steps. First, we define an adversarial access pattern that represents the worst-case for PRAC's security. Second, we investigate PRAC's configurations and security implications. Our analyses show that PRAC can be configured for secure operation as long as no bitflip occurs before accessing a memory location 10 times. Third, we evaluate the performance impact of PRAC and compare it against prior works using Ramulator 2.0. Our analysis shows that while PRAC incurs less than 13% performance overhead for today's DRAM chips, its performance overheads can reach up to 94% for future DRAM chips that are more vulnerable to read disturbance bitflips. Fourth, we define an availability adversarial access pattern that exacerbates PRAC's performance overhead to perform a memory performance attack, demonstrating that such an adversarial pattern can hog up to 94% of DRAM throughput and degrade system throughput by up to 95%. We discuss PRAC's implications on future systems and foreshadow future research directions. To aid future research, we open-source our implementations and scripts at https://github.com/CMU-SAFARI/ramulator2.

摘要: 我们首次对JEDEC DDR5规范2024年4月更新中描述的最先进的片上DRAM读取干扰缓解方法-每行激活计数(PRAC)-进行了严格的安全、性能、能量和成本分析。与建议存储器控制器定期发出刷新管理(RFM)命令(为DRAM芯片提供执行刷新的时间)的现有技术不同，PRAC引入了新的退避信号。PRAC的退避信号从DRAM芯片传播到存储器控制器，并迫使存储器控制器1)停止服务请求和2)发出RFM命令。因此，RFM命令在需要时发出，而不是定期发出，从而减少了RFM的管理费用。我们分四个步骤对PRAC进行分析。首先，我们定义了一种对抗性访问模式，它代表了对PRAC安全的最坏情况。其次，我们调查了PRAC的配置和安全影响。我们的分析表明，只要在访问一个存储单元10次之前没有发生位翻转，就可以将PRAC配置为安全操作。第三，我们评估了PRAC对性能的影响，并将其与使用Ramuler2.0的前人工作进行了比较。我们的分析表明，虽然PRAC对今天的DRAM芯片产生的性能开销不到13%，但对于更容易受到读取干扰位翻转的未来DRAM芯片，其性能开销可能高达94%。第四，我们定义了一种可用性对抗性访问模式，它加剧了PRAC执行内存性能攻击的性能开销，证明了这种对抗性模式可以占用高达94%的DRAM吞吐量，并使系统吞吐量降低高达95%。我们讨论了PRAC对未来系统的影响，并预示了未来的研究方向。为了帮助未来的研究，我们在https://github.com/CMU-SAFARI/ramulator2.上开放了我们的实现和脚本



## **25. Constructing Adversarial Examples for Vertical Federated Learning: Optimal Client Corruption through Multi-Armed Bandit**

构建垂直联邦学习的对抗性示例：通过多臂强盗实现最佳客户腐败 cs.LG

Published on ICLR2024

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2408.04310v1) [paper-pdf](http://arxiv.org/pdf/2408.04310v1)

**Authors**: Duanyi Yao, Songze Li, Ye Xue, Jin Liu

**Abstract**: Vertical federated learning (VFL), where each participating client holds a subset of data features, has found numerous applications in finance, healthcare, and IoT systems. However, adversarial attacks, particularly through the injection of adversarial examples (AEs), pose serious challenges to the security of VFL models. In this paper, we investigate such vulnerabilities through developing a novel attack to disrupt the VFL inference process, under a practical scenario where the adversary is able to adaptively corrupt a subset of clients. We formulate the problem of finding optimal attack strategies as an online optimization problem, which is decomposed into an inner problem of adversarial example generation (AEG) and an outer problem of corruption pattern selection (CPS). Specifically, we establish the equivalence between the formulated CPS problem and a multi-armed bandit (MAB) problem, and propose the Thompson sampling with Empirical maximum reward (E-TS) algorithm for the adversary to efficiently identify the optimal subset of clients for corruption. The key idea of E-TS is to introduce an estimation of the expected maximum reward for each arm, which helps to specify a small set of competitive arms, on which the exploration for the optimal arm is performed. This significantly reduces the exploration space, which otherwise can quickly become prohibitively large as the number of clients increases. We analytically characterize the regret bound of E-TS, and empirically demonstrate its capability of efficiently revealing the optimal corruption pattern with the highest attack success rate, under various datasets of popular VFL tasks.

摘要: 垂直联合学习(VFL)已在金融、医疗保健和物联网系统中发现了大量应用，其中每个参与客户端都拥有数据功能的子集。然而，对抗性攻击，特别是通过注入对抗性例子(AE)，对VFL模型的安全性构成了严重的挑战。在本文中，我们通过开发一种新的攻击来扰乱VFL推理过程，在攻击者能够自适应地破坏客户端子集的实际场景下，研究了这种漏洞。我们将寻找最优攻击策略的问题描述为一个在线优化问题，并将其分解为对抗性实例生成(AEG)和外部腐败模式选择(CPS)问题。具体地说，我们建立了所提出的CPS问题与多臂盗贼(MAB)问题的等价性，并针对敌手提出了带经验最大回报的Thompson抽样(E-TS)算法来有效地识别最优客户子集。E-TS的核心思想是引入对每个手臂的期望最大回报的估计，这有助于指定一个小的竞争手臂集合，在这些集合上进行最优手臂的探索。这大大减少了探索空间，否则，随着客户端数量的增加，探索空间可能会迅速变得令人望而却步。我们对E-TS的遗憾界进行了分析刻画，并在各种流行的VFL任务数据集上实验证明了它能够以最高的攻击成功率有效地揭示最优的破坏模式。



## **26. Eliminating Backdoors in Neural Code Models via Trigger Inversion**

通过触发器倒置消除神经代码模型中的后门 cs.CR

Under review

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2408.04683v1) [paper-pdf](http://arxiv.org/pdf/2408.04683v1)

**Authors**: Weisong Sun, Yuchen Chen, Chunrong Fang, Yebo Feng, Yuan Xiao, An Guo, Quanjun Zhang, Yang Liu, Baowen Xu, Zhenyu Chen

**Abstract**: Neural code models (NCMs) have been widely used for addressing various code understanding tasks, such as defect detection and clone detection. However, numerous recent studies reveal that such models are vulnerable to backdoor attacks. Backdoored NCMs function normally on normal code snippets, but exhibit adversary-expected behavior on poisoned code snippets injected with the adversary-crafted trigger. It poses a significant security threat. For example, a backdoored defect detection model may misclassify user-submitted defective code as non-defective. If this insecure code is then integrated into critical systems, like autonomous driving systems, it could lead to life safety. However, there is an urgent need for effective defenses against backdoor attacks targeting NCMs.   To address this issue, in this paper, we innovatively propose a backdoor defense technique based on trigger inversion, called EliBadCode. EliBadCode first filters the model vocabulary for trigger tokens to reduce the search space for trigger inversion, thereby enhancing the efficiency of the trigger inversion. Then, EliBadCode introduces a sample-specific trigger position identification method, which can reduce the interference of adversarial perturbations for subsequent trigger inversion, thereby producing effective inverted triggers efficiently. Subsequently, EliBadCode employs a Greedy Coordinate Gradient algorithm to optimize the inverted trigger and designs a trigger anchoring method to purify the inverted trigger. Finally, EliBadCode eliminates backdoors through model unlearning. We evaluate the effectiveness of EliBadCode in eliminating backdoor attacks against multiple NCMs used for three safety-critical code understanding tasks. The results demonstrate that EliBadCode can effectively eliminate backdoors while having minimal adverse effects on the normal functionality of the model.

摘要: 神经代码模型(NCM)已被广泛用于解决各种代码理解任务，如缺陷检测和克隆检测。然而，最近的大量研究表明，这种模型很容易受到后门攻击。反向NCMS在正常代码片段上正常工作，但在注入了对手定制的触发器的有毒代码片段上表现出对手预期的行为。它构成了重大的安全威胁。例如，回溯的缺陷检测模型可能将用户提交的缺陷代码错误分类为无缺陷。如果这种不安全的代码随后被集成到关键系统中，比如自动驾驶系统，它可能会导致生命安全。然而，迫切需要有效防御针对新农合的后门攻击。针对这一问题，本文创新性地提出了一种基于触发器反转的后门防御技术，称为EliBadCode。EliBadCode首先过滤触发器令牌的模型词汇，以减少触发器反转的搜索空间，从而提高触发器反转的效率。然后，EliBadCode引入了一种特定样本的触发位置识别方法，该方法可以减少对抗性扰动对后续触发倒置的干扰，从而高效地产生有效的倒置触发。随后，EliBadCode使用贪婪坐标梯度算法对倒置触发器进行优化，并设计了触发器锚定方法对倒置触发器进行净化。最后，EliBadCode通过模型遗忘消除了后门。我们评估了EliBadCode在消除针对用于三个安全关键代码理解任务的多个NCM的后门攻击方面的有效性。结果表明，EliBadCode可以有效地消除后门，同时对模型的正常功能产生的负面影响最小。



## **27. Unveiling Hidden Visual Information: A Reconstruction Attack Against Adversarial Visual Information Hiding**

揭开隐藏的视觉信息：针对对抗性视觉信息隐藏的重建攻击 cs.CV

12 pages

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2408.04261v1) [paper-pdf](http://arxiv.org/pdf/2408.04261v1)

**Authors**: Jonggyu Jang, Hyeonsu Lyu, Seongjin Hwang, Hyun Jong Yang

**Abstract**: This paper investigates the security vulnerabilities of adversarial-example-based image encryption by executing data reconstruction (DR) attacks on encrypted images. A representative image encryption method is the adversarial visual information hiding (AVIH), which uses type-I adversarial example training to protect gallery datasets used in image recognition tasks. In the AVIH method, the type-I adversarial example approach creates images that appear completely different but are still recognized by machines as the original ones. Additionally, the AVIH method can restore encrypted images to their original forms using a predefined private key generative model. For the best security, assigning a unique key to each image is recommended; however, storage limitations may necessitate some images sharing the same key model. This raises a crucial security question for AVIH: How many images can safely share the same key model without being compromised by a DR attack? To address this question, we introduce a dual-strategy DR attack against the AVIH encryption method by incorporating (1) generative-adversarial loss and (2) augmented identity loss, which prevent DR from overfitting -- an issue akin to that in machine learning. Our numerical results validate this approach through image recognition and re-identification benchmarks, demonstrating that our strategy can significantly enhance the quality of reconstructed images, thereby requiring fewer key-sharing encrypted images. Our source code to reproduce our results will be available soon.

摘要: 通过对加密图像进行数据重构(DR)攻击，研究了基于对抗性实例的图像加密的安全漏洞。一种具有代表性的图像加密方法是对抗性视觉信息隐藏(AVIH)，它使用I类对抗性示例训练来保护图像识别任务中使用的图库数据集。在AVIH方法中，第一类对抗性示例方法创建的图像看起来完全不同，但仍然被机器识别为原始图像。此外，AVIH方法可以使用预定义的私钥生成模型将加密图像恢复到其原始形式。为获得最佳安全性，建议为每个映像分配唯一的密钥；但是，存储限制可能需要某些映像共享相同的密钥模型。这对AVIH提出了一个关键的安全问题：有多少映像可以安全地共享相同的密钥模型，而不会受到灾难恢复攻击？为了解决这个问题，我们引入了一种针对AVIH加密方法的双重策略DR攻击，通过结合(1)生成-对手丢失和(2)增强身份丢失来防止DR过度匹配--这是一个类似于机器学习中的问题。我们的数值结果通过图像识别和重新识别基准来验证该方法，表明我们的策略可以显著提高重建图像的质量，从而需要更少的密钥共享加密图像。我们复制结果的源代码很快就会推出。



## **28. Artificial Intelligence based Approach for Identification and Mitigation of Cyber-Attacks in Wide-Area Control of Power Systems**

基于人工智能的电力系统广域控制中网络攻击识别和缓解方法 eess.SY

AAAI 2023 workshop

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2408.04189v1) [paper-pdf](http://arxiv.org/pdf/2408.04189v1)

**Authors**: Jishnudeep Kar, Aranya Chakrabortty

**Abstract**: We propose a generative adversarial network (GAN) based deep learning method that serves the dual role of both identification and mitigation of cyber-attacks in wide-area damping control loops of power systems. Two specific types of attacks considered are false data injection and denial-of-service (DoS). Unlike existing methods, which are either model-based or model-free and yet require two separate learning modules for detection and mitigation leading to longer response times before clearing an attack, our deep learner incorporate both goals within the same integrated framework. A Long Short-Term Memory (LSTM) encoder-decoder based GAN is proposed that captures the temporal dynamics of the power system significantly more accurately than fully-connected GANs, thereby providing better accuracy and faster response for both goals. The method is validated using the IEEE 68-bus power system model.

摘要: 我们提出了一种基于生成对抗网络（GAN）的深度学习方法，该方法具有识别和缓解电力系统广域衰减控制环中网络攻击的双重作用。考虑的两种特定类型的攻击是虚假数据注入和拒绝服务（DPS）。与现有方法不同，现有方法要么基于模型，要么无模型，但需要两个独立的学习模块来检测和缓解，从而在清除攻击之前获得更长的响应时间，我们的深度学习器将这两个目标整合在同一个集成框架中。提出了一种基于长短期存储器（LSTM）编码器-解码器的GAN，它比完全连接的GAN更准确地捕捉电力系统的时间动态，从而为这两个目标提供更好的准确性和更快的响应。使用IEEE 68节点电力系统模型对该方法进行了验证。



## **29. EdgeShield: A Universal and Efficient Edge Computing Framework for Robust AI**

EdgeShield：用于稳健人工智能的通用高效边缘计算框架 cs.CR

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2408.04181v1) [paper-pdf](http://arxiv.org/pdf/2408.04181v1)

**Authors**: Duo Zhong, Bojing Li, Xiang Chen, Chenchen Liu

**Abstract**: The increasing prevalence of adversarial attacks on Artificial Intelligence (AI) systems has created a need for innovative security measures. However, the current methods of defending against these attacks often come with a high computing cost and require back-end processing, making real-time defense challenging. Fortunately, there have been remarkable advancements in edge-computing, which make it easier to deploy neural networks on edge devices. Building upon these advancements, we propose an edge framework design to enable universal and efficient detection of adversarial attacks. This framework incorporates an attention-based adversarial detection methodology and a lightweight detection network formation, making it suitable for a wide range of neural networks and can be deployed on edge devices. To assess the effectiveness of our proposed framework, we conducted evaluations on five neural networks. The results indicate an impressive 97.43% F-score can be achieved, demonstrating the framework's proficiency in detecting adversarial attacks. Moreover, our proposed framework also exhibits significantly reduced computing complexity and cost in comparison to previous detection methods. This aspect is particularly beneficial as it ensures that the defense mechanism can be efficiently implemented in real-time on-edge devices.

摘要: 对人工智能(AI)系统的敌意攻击日益普遍，这就产生了对创新安全措施的需求。然而，当前防御这些攻击的方法往往伴随着较高的计算成本，并且需要后端处理，这使得实时防御具有挑战性。幸运的是，边缘计算已经有了显著的进步，这使得在边缘设备上部署神经网络变得更容易。在这些进展的基础上，我们提出了一种EDGE框架设计，以实现对对手攻击的通用和高效检测。该框架结合了基于注意力的敌意检测方法和轻量级检测网络的形成，使其适用于广泛的神经网络，并可部署在边缘设备上。为了评估我们提出的框架的有效性，我们对五个神经网络进行了评估。结果表明，该框架能够达到令人印象深刻的97.43%的F-Score，证明了该框架在检测对抗性攻击方面的熟练程度。此外，与以前的检测方法相比，我们提出的框架还显著降低了计算复杂度和成本。这方面尤其有益，因为它确保了防御机制可以在实时边缘设备中有效地实施。



## **30. Investigating Adversarial Attacks in Software Analytics via Machine Learning Explainability**

通过机器学习解释性调查软件分析中的对抗性攻击 cs.SE

This is paper is under review

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2408.04124v1) [paper-pdf](http://arxiv.org/pdf/2408.04124v1)

**Authors**: MD Abdul Awal, Mrigank Rochan, Chanchal K. Roy

**Abstract**: With the recent advancements in machine learning (ML), numerous ML-based approaches have been extensively applied in software analytics tasks to streamline software development and maintenance processes. Nevertheless, studies indicate that despite their potential usefulness, ML models are vulnerable to adversarial attacks, which may result in significant monetary losses in these processes. As a result, the ML models' robustness against adversarial attacks must be assessed before they are deployed in software analytics tasks. Despite several techniques being available for adversarial attacks in software analytics tasks, exploring adversarial attacks using ML explainability is largely unexplored. Therefore, this study aims to investigate the relationship between ML explainability and adversarial attacks to measure the robustness of ML models in software analytics tasks. In addition, unlike most existing attacks that directly perturb input-space, our attack approach focuses on perturbing feature-space. Our extensive experiments, involving six datasets, three ML explainability techniques, and seven ML models, demonstrate that ML explainability can be used to conduct successful adversarial attacks on ML models in software analytics tasks. This is achieved by modifying only the top 1-3 important features identified by ML explainability techniques. Consequently, the ML models under attack fail to accurately predict up to 86.6% of instances that were correctly predicted before adversarial attacks, indicating the models' low robustness against such attacks. Finally, our proposed technique demonstrates promising results compared to four state-of-the-art adversarial attack techniques targeting tabular data.

摘要: 随着机器学习的最新进展，许多基于机器学习的方法被广泛应用于软件分析任务中，以简化软件开发和维护过程。然而，研究表明，尽管ML模型具有潜在的实用性，但它很容易受到对抗性攻击，这可能会在这些过程中导致重大的金钱损失。因此，在软件分析任务中部署ML模型之前，必须评估它们对对手攻击的健壮性。尽管有几种技术可用于软件分析任务中的对抗性攻击，但使用ML可解释性探索对抗性攻击在很大程度上是未被探索的。因此，本研究旨在探讨ML可解释性与敌意攻击之间的关系，以衡量ML模型在软件分析任务中的稳健性。此外，与大多数现有的直接扰动输入空间的攻击不同，我们的攻击方法侧重于扰动特征空间。我们的实验涉及六个数据集、三个ML可解释性技术和七个ML模型，证明了ML可解释性可用于在软件分析任务中对ML模型进行成功的对抗性攻击。这是通过只修改ML可解释性技术确定的前1-3个重要特性来实现的。因此，受到攻击的ML模型无法准确预测在对抗性攻击之前正确预测的实例的86.6%，表明该模型对此类攻击的稳健性较低。最后，与四种最先进的针对表格数据的对抗性攻击技术相比，我们提出的技术显示了有希望的结果。



## **31. Effective Prompt Extraction from Language Models**

从语言模型中有效的提示提取 cs.CL

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2307.06865v3) [paper-pdf](http://arxiv.org/pdf/2307.06865v3)

**Authors**: Yiming Zhang, Nicholas Carlini, Daphne Ippolito

**Abstract**: The text generated by large language models is commonly controlled by prompting, where a prompt prepended to a user's query guides the model's output. The prompts used by companies to guide their models are often treated as secrets, to be hidden from the user making the query. They have even been treated as commodities to be bought and sold on marketplaces. However, anecdotal reports have shown adversarial users employing prompt extraction attacks to recover these prompts. In this paper, we present a framework for systematically measuring the effectiveness of these attacks. In experiments with 3 different sources of prompts and 11 underlying large language models, we find that simple text-based attacks can in fact reveal prompts with high probability. Our framework determines with high precision whether an extracted prompt is the actual secret prompt, rather than a model hallucination. Prompt extraction from real systems such as Claude 3 and ChatGPT further suggest that system prompts can be revealed by an adversary despite existing defenses in place.

摘要: 大型语言模型生成的文本通常通过提示进行控制，其中用户查询前的提示将指导模型的输出。公司用来指导其模型的提示通常被视为秘密，对进行查询的用户隐藏。它们甚至被视为可以在市场上买卖的商品。然而，坊间报道显示，敌意用户使用提示提取攻击来恢复这些提示。在本文中，我们提出了一个系统地衡量这些攻击的有效性的框架。在对3种不同的提示源和11个基本的大型语言模型进行的实验中，我们发现简单的基于文本的攻击实际上可以高概率地揭示提示。我们的框架高精度地确定提取的提示是否是实际的秘密提示，而不是模型幻觉。从Claude 3和ChatGPT等真实系统中提取提示进一步表明，尽管已有防御措施，但系统提示仍可被对手泄露。



## **32. LaFA: Latent Feature Attacks on Non-negative Matrix Factorization**

LaFA：对非负矩阵分解的潜在特征攻击 cs.LG

LA-UR-24-26951

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2408.03909v1) [paper-pdf](http://arxiv.org/pdf/2408.03909v1)

**Authors**: Minh Vu, Ben Nebgen, Erik Skau, Geigh Zollicoffer, Juan Castorena, Kim Rasmussen, Boian Alexandrov, Manish Bhattarai

**Abstract**: As Machine Learning (ML) applications rapidly grow, concerns about adversarial attacks compromising their reliability have gained significant attention. One unsupervised ML method known for its resilience to such attacks is Non-negative Matrix Factorization (NMF), an algorithm that decomposes input data into lower-dimensional latent features. However, the introduction of powerful computational tools such as Pytorch enables the computation of gradients of the latent features with respect to the original data, raising concerns about NMF's reliability. Interestingly, naively deriving the adversarial loss for NMF as in the case of ML would result in the reconstruction loss, which can be shown theoretically to be an ineffective attacking objective. In this work, we introduce a novel class of attacks in NMF termed Latent Feature Attacks (LaFA), which aim to manipulate the latent features produced by the NMF process. Our method utilizes the Feature Error (FE) loss directly on the latent features. By employing FE loss, we generate perturbations in the original data that significantly affect the extracted latent features, revealing vulnerabilities akin to those found in other ML techniques. To handle large peak-memory overhead from gradient back-propagation in FE attacks, we develop a method based on implicit differentiation which enables their scaling to larger datasets. We validate NMF vulnerabilities and FE attacks effectiveness through extensive experiments on synthetic and real-world data.

摘要: 随着机器学习(ML)应用的快速发展，人们对威胁其可靠性的对抗性攻击的担忧日益受到关注。一种以其对此类攻击的弹性而闻名的无监督最大似然方法是非负矩阵分解(NMF)，这是一种将输入数据分解为低维潜在特征的算法。然而，强大的计算工具的引入，如火炬，使潜在特征的梯度相对于原始数据的计算成为可能，这引发了人们对NMF可靠性的担忧。有趣的是，天真地推导出NMF在ML的情况下的对抗性损失将导致重建损失，从理论上可以证明这是一个无效的攻击目标。在这项工作中，我们引入了一类新的攻击，称为潜在特征攻击(LAFA)，其目的是操纵NMF过程产生的潜在特征。我们的方法直接利用潜在特征的特征误差(FE)损失。通过使用有限元损失，我们在原始数据中产生了显著影响提取的潜在特征的扰动，揭示了类似于其他ML技术中发现的漏洞。为了处理FE攻击中梯度反向传播带来的巨大峰值内存开销，我们提出了一种基于隐式微分的方法，使其能够扩展到更大的数据集。通过对人工数据和真实数据的大量实验，我们验证了NMF漏洞和FE攻击的有效性。



## **33. Decoding Biases: Automated Methods and LLM Judges for Gender Bias Detection in Language Models**

解码偏见：语言模型中性别偏见检测的自动方法和LLM法官 cs.CL

6 pages paper content, 17 pages of appendix

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2408.03907v1) [paper-pdf](http://arxiv.org/pdf/2408.03907v1)

**Authors**: Shachi H Kumar, Saurav Sahay, Sahisnu Mazumder, Eda Okur, Ramesh Manuvinakurike, Nicole Beckage, Hsuan Su, Hung-yi Lee, Lama Nachman

**Abstract**: Large Language Models (LLMs) have excelled at language understanding and generating human-level text. However, even with supervised training and human alignment, these LLMs are susceptible to adversarial attacks where malicious users can prompt the model to generate undesirable text. LLMs also inherently encode potential biases that can cause various harmful effects during interactions. Bias evaluation metrics lack standards as well as consensus and existing methods often rely on human-generated templates and annotations which are expensive and labor intensive. In this work, we train models to automatically create adversarial prompts to elicit biased responses from target LLMs. We present LLM- based bias evaluation metrics and also analyze several existing automatic evaluation methods and metrics. We analyze the various nuances of model responses, identify the strengths and weaknesses of model families, and assess where evaluation methods fall short. We compare these metrics to human evaluation and validate that the LLM-as-a-Judge metric aligns with human judgement on bias in response generation.

摘要: 大型语言模型(LLM)在语言理解和生成人类级别的文本方面表现出色。然而，即使在有监督的训练和人类对齐的情况下，这些LLM也容易受到敌意攻击，恶意用户可以提示模型生成不想要的文本。LLM还固有地编码潜在的偏见，这些偏见可能在相互作用期间造成各种有害影响。偏差评估指标缺乏标准和共识，现有的方法往往依赖于人工生成的模板和注释，这些模板和注释昂贵且劳动密集型。在这项工作中，我们训练模型自动创建对抗性提示，以引起目标LLM的偏见反应。提出了基于LLM的偏差评价指标，并分析了现有的几种自动评价方法和指标。我们分析模型响应的各种细微差别，确定模型家庭的优点和缺点，并评估评估方法的不足之处。我们将这些指标与人类评估进行比较，并验证LLM作为法官的指标与人类对响应生成中的偏差的判断一致。



## **34. EnJa: Ensemble Jailbreak on Large Language Models**

EnJa：大型语言模型上的越狱 cs.CR

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2408.03603v1) [paper-pdf](http://arxiv.org/pdf/2408.03603v1)

**Authors**: Jiahao Zhang, Zilong Wang, Ruofan Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: As Large Language Models (LLMs) are increasingly being deployed in safety-critical applications, their vulnerability to potential jailbreaks -- malicious prompts that can disable the safety mechanism of LLMs -- has attracted growing research attention. While alignment methods have been proposed to protect LLMs from jailbreaks, many have found that aligned LLMs can still be jailbroken by carefully crafted malicious prompts, producing content that violates policy regulations. Existing jailbreak attacks on LLMs can be categorized into prompt-level methods which make up stories/logic to circumvent safety alignment and token-level attack methods which leverage gradient methods to find adversarial tokens. In this work, we introduce the concept of Ensemble Jailbreak and explore methods that can integrate prompt-level and token-level jailbreak into a more powerful hybrid jailbreak attack. Specifically, we propose a novel EnJa attack to hide harmful instructions using prompt-level jailbreak, boost the attack success rate using a gradient-based attack, and connect the two types of jailbreak attacks via a template-based connector. We evaluate the effectiveness of EnJa on several aligned models and show that it achieves a state-of-the-art attack success rate with fewer queries and is much stronger than any individual jailbreak.

摘要: 随着大型语言模型(LLM)越来越多地被部署在安全关键型应用程序中，它们对潜在越狱的脆弱性--可以禁用LLM安全机制的恶意提示--引起了越来越多的研究关注。虽然已经提出了一些方法来保护LLM免受越狱之苦，但许多人发现，通过精心设计的恶意提示，仍然可以通过精心设计的恶意提示来越狱，从而产生违反政策规定的内容。现有的针对LLMS的越狱攻击可以分为两种：一种是编造故事/逻辑来规避安全对齐的提示级攻击方法，另一种是利用梯度方法来寻找对抗性令牌的令牌级攻击方法。在这项工作中，我们引入了集成越狱的概念，并探索了可以将提示级和令牌级越狱集成到更强大的混合越狱攻击中的方法。具体地说，我们提出了一种新的Enja攻击，利用提示级越狱隐藏有害指令，使用基于梯度的攻击提高攻击成功率，并通过基于模板的连接器将两种类型的越狱攻击连接起来。我们在几个对齐的模型上对Enja的有效性进行了评估，结果表明，它以更少的查询获得了最先进的攻击成功率，并且比任何单个越狱都要强大得多。



## **35. Enhancing Output Diversity Improves Conjugate Gradient-based Adversarial Attacks**

增强输出多样性改善基于结合对象的对抗攻击 cs.LG

ICPRAI2024

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2408.03972v1) [paper-pdf](http://arxiv.org/pdf/2408.03972v1)

**Authors**: Keiichiro Yamamura, Issa Oe, Hiroki Ishikura, Katsuki Fujisawa

**Abstract**: Deep neural networks are vulnerable to adversarial examples, and adversarial attacks that generate adversarial examples have been studied in this context. Existing studies imply that increasing the diversity of model outputs contributes to improving the attack performance. This study focuses on the Auto Conjugate Gradient (ACG) attack, which is inspired by the conjugate gradient method and has a high diversification performance. We hypothesized that increasing the distance between two consecutive search points would enhance the output diversity. To test our hypothesis, we propose Rescaling-ACG (ReACG), which automatically modifies the two components that significantly affect the distance between two consecutive search points, including the search direction and step size. ReACG showed higher attack performance than that of ACG, and is particularly effective for ImageNet models with several classification classes. Experimental results show that the distance between two consecutive search points enhances the output diversity and may help develop new potent attacks. The code is available at \url{https://github.com/yamamura-k/ReACG}

摘要: 深层神经网络很容易受到对抗性例子的攻击，而产生对抗性例子的对抗性攻击就是在这种背景下研究的。现有研究表明，增加模型输出的多样性有助于提高攻击性能。本文研究的是自共轭梯度攻击，该攻击受到共轭梯度方法的启发，具有较高的多样化性能。我们假设增加两个连续搜索点之间的距离会增强输出多样性。为了验证我们的假设，我们提出了Rescaling-ACG(ReacG)算法，它自动修改对两个连续搜索点之间的距离有显著影响的两个分量，包括搜索方向和步长。ReACG表现出比ACG更高的攻击性能，对于具有多个分类类别的ImageNet模型尤其有效。实验结果表明，两个连续搜索点之间的距离增强了输出的多样性，有助于开发新的有效攻击。代码可在\url{https://github.com/yamamura-k/ReACG}



## **36. Spatial-Frequency Discriminability for Revealing Adversarial Perturbations**

揭示对抗性扰动的空间频率辨别性 cs.CV

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2305.10856v3) [paper-pdf](http://arxiv.org/pdf/2305.10856v3)

**Authors**: Chao Wang, Shuren Qi, Zhiqiu Huang, Yushu Zhang, Rushi Lan, Xiaochun Cao, Feng-Lei Fan

**Abstract**: The vulnerability of deep neural networks to adversarial perturbations has been widely perceived in the computer vision community. From a security perspective, it poses a critical risk for modern vision systems, e.g., the popular Deep Learning as a Service (DLaaS) frameworks. For protecting deep models while not modifying them, current algorithms typically detect adversarial patterns through discriminative decomposition for natural and adversarial data. However, these decompositions are either biased towards frequency resolution or spatial resolution, thus failing to capture adversarial patterns comprehensively. Also, when the detector relies on few fixed features, it is practical for an adversary to fool the model while evading the detector (i.e., defense-aware attack). Motivated by such facts, we propose a discriminative detector relying on a spatial-frequency Krawtchouk decomposition. It expands the above works from two aspects: 1) the introduced Krawtchouk basis provides better spatial-frequency discriminability, capturing the differences between natural and adversarial data comprehensively in both spatial and frequency distributions, w.r.t. the common trigonometric or wavelet basis; 2) the extensive features formed by the Krawtchouk decomposition allows for adaptive feature selection and secrecy mechanism, significantly increasing the difficulty of the defense-aware attack, w.r.t. the detector with few fixed features. Theoretical and numerical analyses demonstrate the uniqueness and usefulness of our detector, exhibiting competitive scores on several deep models and image sets against a variety of adversarial attacks.

摘要: 深度神经网络对对抗性扰动的脆弱性已经在计算机视觉领域得到了广泛的认识。从安全的角度来看，它对现代视觉系统构成了严重的风险，例如流行的深度学习即服务(DLaaS)框架。为了在不修改深层模型的同时保护深层模型，目前的算法通常通过对自然数据和对抗性数据进行区分分解来检测对抗性模式。然而，这些分解要么偏向于频率分辨率，要么偏向于空间分辨率，因此不能全面地捕捉到对抗性模式。此外，当检测器依赖于较少的固定特征时，对手在逃避检测器的同时愚弄模型是可行的(即，防御感知攻击)。受此启发，我们提出了一种基于空间频率Krawtchouk分解的鉴别检测器。它从两个方面对上述工作进行了扩展：1)引入的Krawtchouk基提供了更好的空间-频率可区分性，全面地捕捉了自然数据和对抗性数据在空间和频率分布上的差异。2)Krawtchouk分解形成的广泛特征允许自适应的特征选择和保密机制，显著增加了防御感知攻击的难度。探测器几乎没有固定的特征。理论和数值分析证明了我们的检测器的独特性和实用性，在几个深层模型和图像集上展示了对抗各种敌意攻击的竞争性分数。



## **37. A Study on Prompt Injection Attack Against LLM-Integrated Mobile Robotic Systems**

针对LLM集成移动机器人系统的即时注入攻击研究 cs.RO

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2408.03515v1) [paper-pdf](http://arxiv.org/pdf/2408.03515v1)

**Authors**: Wenxiao Zhang, Xiangrui Kong, Conan Dewitt, Thomas Braunl, Jin B. Hong

**Abstract**: The integration of Large Language Models (LLMs) like GPT-4o into robotic systems represents a significant advancement in embodied artificial intelligence. These models can process multi-modal prompts, enabling them to generate more context-aware responses. However, this integration is not without challenges. One of the primary concerns is the potential security risks associated with using LLMs in robotic navigation tasks. These tasks require precise and reliable responses to ensure safe and effective operation. Multi-modal prompts, while enhancing the robot's understanding, also introduce complexities that can be exploited maliciously. For instance, adversarial inputs designed to mislead the model can lead to incorrect or dangerous navigational decisions. This study investigates the impact of prompt injections on mobile robot performance in LLM-integrated systems and explores secure prompt strategies to mitigate these risks. Our findings demonstrate a substantial overall improvement of approximately 30.8% in both attack detection and system performance with the implementation of robust defence mechanisms, highlighting their critical role in enhancing security and reliability in mission-oriented tasks.

摘要: 将像GPT-40这样的大型语言模型(LLM)集成到机器人系统中，代表着体现的人工智能的重大进步。这些模型可以处理多模式提示，使它们能够生成更多情景感知响应。然而，这种整合并不是没有挑战。其中一个主要问题是在机器人导航任务中使用LLMS存在潜在的安全风险。这些任务需要准确可靠的反应，以确保安全有效的运行。多模式提示在增强机器人理解能力的同时，也引入了可能被恶意利用的复杂性。例如，旨在误导模型的对抗性输入可能导致错误或危险的导航决策。这项研究调查了快速注射对LLM集成系统中移动机器人性能的影响，并探索了安全的提示策略来缓解这些风险。我们的研究结果表明，随着强大的防御机制的实施，攻击检测和系统性能都有了大约30.8%的大幅整体改进，突出了它们在增强面向任务的任务的安全性和可靠性方面的关键作用。



## **38. Simple Perturbations Subvert Ethereum Phishing Transactions Detection: An Empirical Analysis**

简单扰动颠覆以太坊网络钓鱼交易检测：实证分析 cs.CR

12 pages, 1 figure, 5 tables, accepted for presentation at WISA 2024

**SubmitDate**: 2024-08-06    [abs](http://arxiv.org/abs/2408.03441v1) [paper-pdf](http://arxiv.org/pdf/2408.03441v1)

**Authors**: Ahod Alghureid, David Mohaisen

**Abstract**: This paper explores the vulnerability of machine learning models, specifically Random Forest, Decision Tree, and K-Nearest Neighbors, to very simple single-feature adversarial attacks in the context of Ethereum fraudulent transaction detection. Through comprehensive experimentation, we investigate the impact of various adversarial attack strategies on model performance metrics, such as accuracy, precision, recall, and F1-score. Our findings, highlighting how prone those techniques are to simple attacks, are alarming, and the inconsistency in the attacks' effect on different algorithms promises ways for attack mitigation. We examine the effectiveness of different mitigation strategies, including adversarial training and enhanced feature selection, in enhancing model robustness.

摘要: 本文探讨了机器学习模型（特别是随机森林、决策树和K-最近邻居）在以太坊欺诈交易检测背景下对非常简单的单特征对抗攻击的脆弱性。通过全面的实验，我们研究了各种对抗攻击策略对模型性能指标的影响，例如准确性、精确性、召回率和F1得分。我们的研究结果强调了这些技术容易受到简单攻击的影响，这令人震惊，而且攻击对不同算法的影响的不一致性为缓解攻击提供了方法。我们研究了不同缓解策略（包括对抗训练和增强的特征选择）在增强模型稳健性方面的有效性。



## **39. Attacks and Defenses for Generative Diffusion Models: A Comprehensive Survey**

生成扩散模型的攻击和防御：全面调查 cs.CR

**SubmitDate**: 2024-08-06    [abs](http://arxiv.org/abs/2408.03400v1) [paper-pdf](http://arxiv.org/pdf/2408.03400v1)

**Authors**: Vu Tuan Truong, Luan Ba Dang, Long Bao Le

**Abstract**: Diffusion models (DMs) have achieved state-of-the-art performance on various generative tasks such as image synthesis, text-to-image, and text-guided image-to-image generation. However, the more powerful the DMs, the more harmful they potentially are. Recent studies have shown that DMs are prone to a wide range of attacks, including adversarial attacks, membership inference, backdoor injection, and various multi-modal threats. Since numerous pre-trained DMs are published widely on the Internet, potential threats from these attacks are especially detrimental to the society, making DM-related security a worth investigating topic. Therefore, in this paper, we conduct a comprehensive survey on the security aspect of DMs, focusing on various attack and defense methods for DMs. First, we present crucial knowledge of DMs with five main types of DMs, including denoising diffusion probabilistic models, denoising diffusion implicit models, noise conditioned score networks, stochastic differential equations, and multi-modal conditional DMs. We further survey a variety of recent studies investigating different types of attacks that exploit the vulnerabilities of DMs. Then, we thoroughly review potential countermeasures to mitigate each of the presented threats. Finally, we discuss open challenges of DM-related security and envision certain research directions for this topic.

摘要: 扩散模型(DM)在图像合成、文本到图像和文本引导的图像到图像生成等各种生成任务中取得了最先进的性能。然而，DM越强大，它们潜在的危害性就越大。最近的研究表明，DM容易受到广泛的攻击，包括对抗性攻击、成员关系推断、后门注入和各种多模式威胁。由于大量预先训练的DM被广泛发布在互联网上，这些攻击的潜在威胁对社会尤其有害，使DM相关的安全成为一个值得研究的话题。因此，在本文中，我们对DMS的安全方面进行了全面的调查，重点研究了DMS的各种攻击和防御方法。首先，我们介绍了数据挖掘的五种主要类型的数据挖掘的关键知识，包括去噪扩散概率模型、去噪扩散隐式模型、噪声条件得分网络、随机微分方程和多模式条件数据挖掘。我们进一步调查了最近的各种研究，调查了利用DM漏洞的不同类型的攻击。然后，我们将彻底审查缓解每一种威胁的潜在对策。最后，我们讨论了数据挖掘相关安全的开放挑战，并展望了该课题的研究方向。



## **40. Self-Evaluation as a Defense Against Adversarial Attacks on LLMs**

自我评估作为对LLM的对抗攻击的防御 cs.LG

8 pages, 7 figures

**SubmitDate**: 2024-08-06    [abs](http://arxiv.org/abs/2407.03234v3) [paper-pdf](http://arxiv.org/pdf/2407.03234v3)

**Authors**: Hannah Brown, Leon Lin, Kenji Kawaguchi, Michael Shieh

**Abstract**: We introduce a defense against adversarial attacks on LLMs utilizing self-evaluation. Our method requires no model fine-tuning, instead using pre-trained models to evaluate the inputs and outputs of a generator model, significantly reducing the cost of implementation in comparison to other, finetuning-based methods. Our method can significantly reduce the attack success rate of attacks on both open and closed-source LLMs, beyond the reductions demonstrated by Llama-Guard2 and commonly used content moderation APIs. We present an analysis of the effectiveness of our method, including attempts to attack the evaluator in various settings, demonstrating that it is also more resilient to attacks than existing methods. Code and data will be made available at https://github.com/Linlt-leon/self-eval.

摘要: 我们利用自我评估引入针对LLM的对抗攻击的防御措施。我们的方法不需要模型微调，而是使用预先训练的模型来评估生成器模型的输入和输出，与其他基于微调的方法相比，显着降低了实施成本。我们的方法可以显着降低对开源和闭源LLM的攻击成功率，超出Llama-Guard 2和常用内容审核API所展示的降低程度。我们对我们方法的有效性进行了分析，包括在各种环境下攻击评估者的尝试，证明它比现有方法对攻击的弹性更强。代码和数据将在https://github.com/Linlt-leon/self-eval上提供。



## **41. Sample-agnostic Adversarial Perturbation for Vision-Language Pre-training Models**

视觉语言预训练模型的样本不可知对抗扰动 cs.CV

13 pages, 8 figures, published in ACMMM2024

**SubmitDate**: 2024-08-06    [abs](http://arxiv.org/abs/2408.02980v1) [paper-pdf](http://arxiv.org/pdf/2408.02980v1)

**Authors**: Haonan Zheng, Wen Jiang, Xinyang Deng, Wenrui Li

**Abstract**: Recent studies on AI security have highlighted the vulnerability of Vision-Language Pre-training (VLP) models to subtle yet intentionally designed perturbations in images and texts. Investigating multimodal systems' robustness via adversarial attacks is crucial in this field. Most multimodal attacks are sample-specific, generating a unique perturbation for each sample to construct adversarial samples. To the best of our knowledge, it is the first work through multimodal decision boundaries to explore the creation of a universal, sample-agnostic perturbation that applies to any image. Initially, we explore strategies to move sample points beyond the decision boundaries of linear classifiers, refining the algorithm to ensure successful attacks under the top $k$ accuracy metric. Based on this foundation, in visual-language tasks, we treat visual and textual modalities as reciprocal sample points and decision hyperplanes, guiding image embeddings to traverse text-constructed decision boundaries, and vice versa. This iterative process consistently refines a universal perturbation, ultimately identifying a singular direction within the input space which is exploitable to impair the retrieval performance of VLP models. The proposed algorithms support the creation of global perturbations or adversarial patches. Comprehensive experiments validate the effectiveness of our method, showcasing its data, task, and model transferability across various VLP models and datasets. Code: https://github.com/LibertazZ/MUAP

摘要: 最近关于人工智能安全的研究突显了视觉语言预训练(VLP)模型对图像和文本中细微但故意设计的扰动的脆弱性。研究多通道系统在对抗攻击下的健壮性是这一领域的关键。大多数多模式攻击是特定于样本的，为每个样本生成唯一的扰动以构建对抗性样本。据我们所知，这是第一个通过多模式决策边界来探索创建适用于任何图像的普遍的、样本不可知的扰动的工作。首先，我们探索了将采样点移动到线性分类器的决策边界之外的策略，改进了算法以确保在最高$k$精度度量下成功攻击。在此基础上，在视觉语言任务中，我们将视觉通道和文本通道视为相互的样本点和决策超平面，引导图像嵌入遍历文本构建的决策边界，反之亦然。这种迭代过程一致地改进了普遍的扰动，最终确定了输入空间内的奇异方向，该方向可被利用来损害VLP模型的检索性能。所提出的算法支持创建全局扰动或对抗性补丁。全面的实验验证了我们方法的有效性，展示了它的数据、任务和模型在各种VLP模型和数据集上的可转移性。代码：https://github.com/LibertazZ/MUAP



## **42. Sharpness-Aware Cross-Domain Recommendation to Cold-Start Users**

向冷启动用户提供具有敏锐意识的跨域推荐 cs.IR

**SubmitDate**: 2024-08-06    [abs](http://arxiv.org/abs/2408.01931v2) [paper-pdf](http://arxiv.org/pdf/2408.01931v2)

**Authors**: Guohang Zeng, Qian Zhang, Guangquan Zhang, Jie Lu

**Abstract**: Cross-Domain Recommendation (CDR) is a promising paradigm inspired by transfer learning to solve the cold-start problem in recommender systems. Existing state-of-the-art CDR methods train an explicit mapping function to transfer the cold-start users from a data-rich source domain to a target domain. However, a limitation of these methods is that the mapping function is trained on overlapping users across domains, while only a small number of overlapping users are available for training. By visualizing the loss landscape of the existing CDR model, we find that training on a small number of overlapping users causes the model to converge to sharp minima, leading to poor generalization. Based on this observation, we leverage loss-geometry-based machine learning approach and propose a novel CDR method called Sharpness-Aware CDR (SCDR). Our proposed method simultaneously optimizes recommendation loss and loss sharpness, leading to better generalization with theoretical guarantees. Empirical studies on real-world datasets demonstrate that SCDR significantly outperforms the other CDR models for cold-start recommendation tasks, while concurrently enhancing the model's robustness to adversarial attacks.

摘要: 跨域推荐(CDR)是一种受迁移学习启发的解决推荐系统冷启动问题的有效方法。现有最先进的CDR方法训练显式映射函数以将冷启动用户从数据丰富的源域转移到目标域。然而，这些方法的局限性在于映射函数是针对跨域的重叠用户进行训练的，而仅有少量重叠用户可供训练。通过可视化现有CDR模型的损失情况，我们发现，在少量重叠用户上进行训练会导致模型收敛到尖锐的最小值，导致泛化能力较差。基于这一观察，我们利用基于损失几何的机器学习方法，提出了一种新的CDR方法，称为清晰度感知CDR(SCDR)。我们提出的方法同时优化了推荐损失和损失锐度，在理论上保证了更好的泛化能力。在真实数据集上的实证研究表明，SCDR在冷启动推荐任务上显著优于其他CDR模型，同时增强了模型对对手攻击的稳健性。



## **43. Adversarial Robustness of Open-source Text Classification Models and Fine-Tuning Chains**

开源文本分类模型和微调链的对抗鲁棒性 cs.SE

**SubmitDate**: 2024-08-06    [abs](http://arxiv.org/abs/2408.02963v1) [paper-pdf](http://arxiv.org/pdf/2408.02963v1)

**Authors**: Hao Qin, Mingyang Li, Junjie Wang, Qing Wang

**Abstract**: Context:With the advancement of artificial intelligence (AI) technology and applications, numerous AI models have been developed, leading to the emergence of open-source model hosting platforms like Hugging Face (HF). Thanks to these platforms, individuals can directly download and use models, as well as fine-tune them to construct more domain-specific models. However, just like traditional software supply chains face security risks, AI models and fine-tuning chains also encounter new security risks, such as adversarial attacks. Therefore, the adversarial robustness of these models has garnered attention, potentially influencing people's choices regarding open-source models. Objective:This paper aims to explore the adversarial robustness of open-source AI models and their chains formed by the upstream-downstream relationships via fine-tuning to provide insights into the potential adversarial risks. Method:We collect text classification models on HF and construct the fine-tuning chains.Then, we conduct an empirical analysis of model reuse and associated robustness risks under existing adversarial attacks from two aspects, i.e., models and their fine-tuning chains. Results:Despite the models' widespread downloading and reuse, they are generally susceptible to adversarial attack risks, with an average of 52.70% attack success rate. Moreover, fine-tuning typically exacerbates this risk, resulting in an average 12.60% increase in attack success rates. We also delve into the influence of factors such as attack techniques, datasets, and model architectures on the success rate, as well as the transitivity along the model chains.

摘要: 背景：随着人工智能(AI)技术和应用的进步，大量的AI模型被开发出来，导致了像拥抱脸(HF)这样的开源模型托管平台的出现。多亏了这些平台，个人可以直接下载和使用模型，以及对它们进行微调以构建更多特定于领域的模型。然而，就像传统的软件供应链面临安全风险一样，AI模型和微调链也遇到了新的安全风险，比如对抗性攻击。因此，这些模型的对抗性健壮性引起了人们的注意，潜在地影响了人们对开源模型的选择。目的：通过微调研究开源人工智能模型及其上下游关系形成的链的对抗稳健性，为潜在的对抗风险提供洞察。方法：收集高频下的文本分类模型，构建精调链，从模型及其精调链两个方面对现有敌意攻击下的模型重用及相关健壮性风险进行实证分析。结果：尽管模型被广泛下载和重用，但它们普遍容易受到对抗性攻击风险的影响，平均攻击成功率为52.70%。此外，微调通常会加剧这种风险，导致攻击成功率平均增加12.60%。我们还深入研究了攻击技术、数据集和模型体系结构等因素对成功率的影响，以及模型链上的传递性。



## **44. Adv3D: Generating 3D Adversarial Examples for 3D Object Detection in Driving Scenarios with NeRF**

Adv3D：使用NeRF为驾驶场景中的3D物体检测生成3D对抗示例 cs.CV

**SubmitDate**: 2024-08-06    [abs](http://arxiv.org/abs/2309.01351v2) [paper-pdf](http://arxiv.org/pdf/2309.01351v2)

**Authors**: Leheng Li, Qing Lian, Ying-Cong Chen

**Abstract**: Deep neural networks (DNNs) have been proven extremely susceptible to adversarial examples, which raises special safety-critical concerns for DNN-based autonomous driving stacks (i.e., 3D object detection). Although there are extensive works on image-level attacks, most are restricted to 2D pixel spaces, and such attacks are not always physically realistic in our 3D world. Here we present Adv3D, the first exploration of modeling adversarial examples as Neural Radiance Fields (NeRFs). Advances in NeRF provide photorealistic appearances and 3D accurate generation, yielding a more realistic and realizable adversarial example. We train our adversarial NeRF by minimizing the surrounding objects' confidence predicted by 3D detectors on the training set. Then we evaluate Adv3D on the unseen validation set and show that it can cause a large performance reduction when rendering NeRF in any sampled pose. To generate physically realizable adversarial examples, we propose primitive-aware sampling and semantic-guided regularization that enable 3D patch attacks with camouflage adversarial texture. Experimental results demonstrate that the trained adversarial NeRF generalizes well to different poses, scenes, and 3D detectors. Finally, we provide a defense method to our attacks that involves adversarial training through data augmentation. Project page: https://len-li.github.io/adv3d-web

摘要: 深度神经网络(DNN)已被证明极易受到敌意例子的影响，这对基于DNN的自动驾驶堆栈(即3D对象检测)提出了特殊的安全关键问题。虽然有大量关于图像级攻击的工作，但大多数局限于2D像素空间，并且在我们的3D世界中，此类攻击并不总是物理上真实的。在这里，我们介绍了Adv3D，这是将对抗性例子建模为神经辐射场(NERF)的第一次探索。NERF的进步提供了照片逼真的外观和3D准确的生成，产生了一个更真实和可实现的对抗性例子。我们通过最小化3D检测器在训练集上预测的周围对象的置信度来训练我们的对手NERF。然后，我们在不可见的验证集上对Adv3D进行了评估，结果表明，在任何采样姿势下绘制NERF时，它都会导致性能大幅下降。为了生成物理上可实现的对抗性示例，我们提出了基元感知采样和语义引导正则化，使得3D补丁攻击具有伪装对抗性纹理。实验结果表明，训练好的对抗性神经网络对不同的姿态、场景和3D检测器都具有很好的泛化能力。最后，我们提供了一种通过数据增强进行对抗性训练的攻击防御方法。项目页面：https://len-li.github.io/adv3d-web



## **45. A Survey and Evaluation of Adversarial Attacks for Object Detection**

对象检测的对抗性攻击调查与评估 cs.CV

14 pages

**SubmitDate**: 2024-08-06    [abs](http://arxiv.org/abs/2408.01934v2) [paper-pdf](http://arxiv.org/pdf/2408.01934v2)

**Authors**: Khoi Nguyen Tiet Nguyen, Wenyu Zhang, Kangkang Lu, Yuhuan Wu, Xingjian Zheng, Hui Li Tan, Liangli Zhen

**Abstract**: Deep learning models excel in various computer vision tasks but are susceptible to adversarial examples-subtle perturbations in input data that lead to incorrect predictions. This vulnerability poses significant risks in safety-critical applications such as autonomous vehicles, security surveillance, and aircraft health monitoring. While numerous surveys focus on adversarial attacks in image classification, the literature on such attacks in object detection is limited. This paper offers a comprehensive taxonomy of adversarial attacks specific to object detection, reviews existing adversarial robustness evaluation metrics, and systematically assesses open-source attack methods and model robustness. Key observations are provided to enhance the understanding of attack effectiveness and corresponding countermeasures. Additionally, we identify crucial research challenges to guide future efforts in securing automated object detection systems.

摘要: 深度学习模型在各种计算机视觉任务中表现出色，但容易受到对抗性示例的影响--输入数据中的微妙扰动，导致错误的预测。该漏洞对自动驾驶汽车、安全监控和飞机健康监控等安全关键应用构成重大风险。虽然大量调查关注图像分类中的对抗攻击，但关于对象检测中此类攻击的文献有限。本文提供了针对对象检测的对抗性攻击的全面分类，回顾了现有的对抗性稳健性评估指标，并系统性评估开源攻击方法和模型稳健性。提供了关键观察，以增强对攻击有效性和相应对策的理解。此外，我们还确定了关键的研究挑战，以指导未来保护自动化物体检测系统的工作。



## **46. Compromising Embodied Agents with Contextual Backdoor Attacks**

通过上下文后门攻击损害被授权的代理 cs.AI

**SubmitDate**: 2024-08-06    [abs](http://arxiv.org/abs/2408.02882v1) [paper-pdf](http://arxiv.org/pdf/2408.02882v1)

**Authors**: Aishan Liu, Yuguang Zhou, Xianglong Liu, Tianyuan Zhang, Siyuan Liang, Jiakai Wang, Yanjun Pu, Tianlin Li, Junqi Zhang, Wenbo Zhou, Qing Guo, Dacheng Tao

**Abstract**: Large language models (LLMs) have transformed the development of embodied intelligence. By providing a few contextual demonstrations, developers can utilize the extensive internal knowledge of LLMs to effortlessly translate complex tasks described in abstract language into sequences of code snippets, which will serve as the execution logic for embodied agents. However, this paper uncovers a significant backdoor security threat within this process and introduces a novel method called \method{}. By poisoning just a few contextual demonstrations, attackers can covertly compromise the contextual environment of a black-box LLM, prompting it to generate programs with context-dependent defects. These programs appear logically sound but contain defects that can activate and induce unintended behaviors when the operational agent encounters specific triggers in its interactive environment. To compromise the LLM's contextual environment, we employ adversarial in-context generation to optimize poisoned demonstrations, where an LLM judge evaluates these poisoned prompts, reporting to an additional LLM that iteratively optimizes the demonstration in a two-player adversarial game using chain-of-thought reasoning. To enable context-dependent behaviors in downstream agents, we implement a dual-modality activation strategy that controls both the generation and execution of program defects through textual and visual triggers. We expand the scope of our attack by developing five program defect modes that compromise key aspects of confidentiality, integrity, and availability in embodied agents. To validate the effectiveness of our approach, we conducted extensive experiments across various tasks, including robot planning, robot manipulation, and compositional visual reasoning. Additionally, we demonstrate the potential impact of our approach by successfully attacking real-world autonomous driving systems.

摘要: 大型语言模型(LLM)改变了体验式智能的发展。通过提供一些上下文演示，开发人员可以利用LLM的丰富内部知识，毫不费力地将以抽象语言描述的复杂任务转换为代码片段序列，这些代码片段将用作具体化代理的执行逻辑。然而，本文发现了该过程中存在的一个严重的后门安全威胁，并介绍了一种名为\方法{}的新方法。只要毒化几个上下文演示，攻击者就可以秘密地危害黑盒LLM的上下文环境，促使它生成具有上下文相关缺陷的程序。这些程序在逻辑上看起来是合理的，但存在缺陷，当操作代理在其交互环境中遇到特定触发器时，这些缺陷可能会激活和诱导意外行为。为了折衷LLM的上下文环境，我们使用对抗性的上下文生成来优化有毒演示，其中LLM法官评估这些有毒的提示，向额外的LLM报告，该LLM使用思想链推理在两人对抗性游戏中迭代优化演示。为了在下游代理中实现上下文相关的行为，我们实现了一种双通道激活策略，该策略通过文本和视觉触发来控制程序缺陷的生成和执行。我们通过开发五种程序缺陷模式来扩大我们的攻击范围，这些模式损害了具体化代理中的机密性、完整性和可用性的关键方面。为了验证我们方法的有效性，我们在各种任务中进行了广泛的实验，包括机器人规划、机器人操作和组合视觉推理。此外，我们通过成功攻击真实世界的自动驾驶系统来演示我们的方法的潜在影响。



## **47. Pre-trained Encoder Inference: Revealing Upstream Encoders In Downstream Machine Learning Services**

预训练的编码器推理：揭示下游机器学习服务中的上游编码器 cs.LG

**SubmitDate**: 2024-08-05    [abs](http://arxiv.org/abs/2408.02814v1) [paper-pdf](http://arxiv.org/pdf/2408.02814v1)

**Authors**: Shaopeng Fu, Xuexue Sun, Ke Qing, Tianhang Zheng, Di Wang

**Abstract**: Though pre-trained encoders can be easily accessed online to build downstream machine learning (ML) services quickly, various attacks have been designed to compromise the security and privacy of these encoders. While most attacks target encoders on the upstream side, it remains unknown how an encoder could be threatened when deployed in a downstream ML service. This paper unveils a new vulnerability: the Pre-trained Encoder Inference (PEI) attack, which posts privacy threats toward encoders hidden behind downstream ML services. By only providing API accesses to a targeted downstream service and a set of candidate encoders, the PEI attack can infer which encoder is secretly used by the targeted service based on candidate ones. We evaluate the attack performance of PEI against real-world encoders on three downstream tasks: image classification, text classification, and text-to-image generation. Experiments show that the PEI attack succeeds in revealing the hidden encoder in most cases and seldom makes mistakes even when the hidden encoder is not in the candidate set. We also conducted a case study on one of the most recent vision-language models, LLaVA, to illustrate that the PEI attack is useful in assisting other ML attacks such as adversarial attacks. The code is available at https://github.com/fshp971/encoder-inference.

摘要: 虽然可以很容易地在线访问预先训练的编码器以快速构建下游机器学习(ML)服务，但是已经设计了各种攻击来危害这些编码器的安全和隐私。虽然大多数攻击的目标是上游端的编码器，但编码器在部署到下游ML服务中时会受到怎样的威胁仍是个未知数。本文揭示了一个新的漏洞：预先训练的编码者推理(PEI)攻击，它向隐藏在下游ML服务后面的编码者发布隐私威胁。通过只提供对目标下游服务和一组候选编码器的API访问，PEI攻击可以根据候选编码器来推断目标服务秘密使用了哪个编码器。我们在三个下游任务：图像分类、文本分类和文本到图像的生成上评估了PEI对真实世界编码者的攻击性能。实验表明，PEI攻击在大多数情况下都能成功地发现隐藏的编码者，即使隐藏的编码者不在候选集合中，也很少出错。我们还对最新的视觉语言模型之一LLaVA进行了案例研究，以说明PEI攻击在辅助其他ML攻击方面也是有用的，例如对抗性攻击。代码可在https://github.com/fshp971/encoder-inference.上获得



## **48. SEAS: Self-Evolving Adversarial Safety Optimization for Large Language Models**

SEAS：大型语言模型的自进化对抗安全优化 cs.CL

**SubmitDate**: 2024-08-05    [abs](http://arxiv.org/abs/2408.02632v1) [paper-pdf](http://arxiv.org/pdf/2408.02632v1)

**Authors**: Muxi Diao, Rumei Li, Shiyang Liu, Guogang Liao, Jingang Wang, Xunliang Cai, Weiran Xu

**Abstract**: As large language models (LLMs) continue to advance in capability and influence, ensuring their security and preventing harmful outputs has become crucial. A promising approach to address these concerns involves training models to automatically generate adversarial prompts for red teaming. However, the evolving subtlety of vulnerabilities in LLMs challenges the effectiveness of current adversarial methods, which struggle to specifically target and explore the weaknesses of these models. To tackle these challenges, we introduce the $\mathbf{S}\text{elf-}\mathbf{E}\text{volving }\mathbf{A}\text{dversarial }\mathbf{S}\text{afety }\mathbf{(SEAS)}$ optimization framework, which enhances security by leveraging data generated by the model itself. SEAS operates through three iterative stages: Initialization, Attack, and Adversarial Optimization, refining both the Red Team and Target models to improve robustness and safety. This framework reduces reliance on manual testing and significantly enhances the security capabilities of LLMs. Our contributions include a novel adversarial framework, a comprehensive safety dataset, and after three iterations, the Target model achieves a security level comparable to GPT-4, while the Red Team model shows a marked increase in attack success rate (ASR) against advanced models.

摘要: 随着大型语言模型在能力和影响力方面的不断进步，确保它们的安全和防止有害输出变得至关重要。解决这些担忧的一个有希望的方法是建立训练模型，为红色团队自动生成对抗性提示。然而，LLMS中不断演变的漏洞的微妙之处挑战了当前对抗性方法的有效性，这些方法难以具体针对和探索这些模型的弱点。为了应对这些挑战，我们引入了$\mathbf{S}\Text{ELF-}\mathbf{E}\Text{volving}\mathbf{A}\Text{dversarial}\mathbf{S}\Text{afty}\mathbf{(SEA)}$优化框架，该框架通过利用模型本身生成的数据来增强安全性。SEA经历了三个迭代阶段：初始化、攻击和对抗性优化，完善了Red Team和Target模型，以提高健壮性和安全性。该框架减少了对手动测试的依赖，显著增强了LLMS的安全能力。我们的贡献包括一个新的对抗性框架，一个全面的安全数据集，经过三次迭代，Target模型达到了与GPT-4相当的安全级别，而Red Team模型显示出相对于高级模型在攻击成功率(ASR)方面的显著提高。



## **49. SSAP: A Shape-Sensitive Adversarial Patch for Comprehensive Disruption of Monocular Depth Estimation in Autonomous Navigation Applications**

SSAP：一种形状敏感对抗补丁，用于全面破坏自主导航应用中单目深度估计 cs.CV

arXiv admin note: text overlap with arXiv:2303.01351

**SubmitDate**: 2024-08-05    [abs](http://arxiv.org/abs/2403.11515v2) [paper-pdf](http://arxiv.org/pdf/2403.11515v2)

**Authors**: Amira Guesmi, Muhammad Abdullah Hanif, Ihsen Alouani, Bassem Ouni, Muhammad Shafique

**Abstract**: Monocular depth estimation (MDE) has advanced significantly, primarily through the integration of convolutional neural networks (CNNs) and more recently, Transformers. However, concerns about their susceptibility to adversarial attacks have emerged, especially in safety-critical domains like autonomous driving and robotic navigation. Existing approaches for assessing CNN-based depth prediction methods have fallen short in inducing comprehensive disruptions to the vision system, often limited to specific local areas. In this paper, we introduce SSAP (Shape-Sensitive Adversarial Patch), a novel approach designed to comprehensively disrupt monocular depth estimation (MDE) in autonomous navigation applications. Our patch is crafted to selectively undermine MDE in two distinct ways: by distorting estimated distances or by creating the illusion of an object disappearing from the system's perspective. Notably, our patch is shape-sensitive, meaning it considers the specific shape and scale of the target object, thereby extending its influence beyond immediate proximity. Furthermore, our patch is trained to effectively address different scales and distances from the camera. Experimental results demonstrate that our approach induces a mean depth estimation error surpassing 0.5, impacting up to 99% of the targeted region for CNN-based MDE models. Additionally, we investigate the vulnerability of Transformer-based MDE models to patch-based attacks, revealing that SSAP yields a significant error of 0.59 and exerts substantial influence over 99% of the target region on these models.

摘要: 单眼深度估计(MDE)已经有了显著的进步，主要是通过卷积神经网络(CNN)的集成，以及最近的Transformers。然而，对它们易受对手攻击的担忧已经出现，特别是在自动驾驶和机器人导航等安全关键领域。现有的评估基于CNN的深度预测方法的方法在导致视觉系统全面中断方面做得不够，通常仅限于特定的局部地区。在本文中，我们介绍了形状敏感对抗性补丁(SSAP)，一种新的方法，旨在全面扰乱单眼深度估计(MDE)在自主导航应用中。我们的补丁是为了有选择地以两种不同的方式削弱MDE：通过扭曲估计的距离或通过创造物体从系统的角度消失的错觉。值得注意的是，我们的面片是形状敏感的，这意味着它考虑目标对象的特定形状和比例，从而将其影响扩展到直接邻近之外。此外，我们的补丁经过训练，可以有效地处理不同比例和距离相机的问题。实验结果表明，对于基于CNN的MDE模型，该方法的平均深度估计误差超过0.5，影响高达99%的目标区域。此外，我们研究了基于Transformer的MDE模型对基于补丁的攻击的脆弱性，发现SSAP产生了0.59的显著误差，并且对这些模型上99%的目标区域产生了重大影响。



## **50. APARATE: Adaptive Adversarial Patch for CNN-based Monocular Depth Estimation for Autonomous Navigation**

APARATE：用于自主导航基于CNN的单目深度估计的自适应对抗补丁 cs.CV

**SubmitDate**: 2024-08-05    [abs](http://arxiv.org/abs/2303.01351v3) [paper-pdf](http://arxiv.org/pdf/2303.01351v3)

**Authors**: Amira Guesmi, Muhammad Abdullah Hanif, Ihsen Alouani, Muhammad Shafique

**Abstract**: In recent times, monocular depth estimation (MDE) has experienced significant advancements in performance, largely attributed to the integration of innovative architectures, i.e., convolutional neural networks (CNNs) and Transformers. Nevertheless, the susceptibility of these models to adversarial attacks has emerged as a noteworthy concern, especially in domains where safety and security are paramount. This concern holds particular weight for MDE due to its critical role in applications like autonomous driving and robotic navigation, where accurate scene understanding is pivotal. To assess the vulnerability of CNN-based depth prediction methods, recent work tries to design adversarial patches against MDE. However, the existing approaches fall short of inducing a comprehensive and substantially disruptive impact on the vision system. Instead, their influence is partial and confined to specific local areas. These methods lead to erroneous depth predictions only within the overlapping region with the input image, without considering the characteristics of the target object, such as its size, shape, and position. In this paper, we introduce a novel adversarial patch named APARATE. This patch possesses the ability to selectively undermine MDE in two distinct ways: by distorting the estimated distances or by creating the illusion of an object disappearing from the perspective of the autonomous system. Notably, APARATE is designed to be sensitive to the shape and scale of the target object, and its influence extends beyond immediate proximity. APARATE, results in a mean depth estimation error surpassing $0.5$, significantly impacting as much as $99\%$ of the targeted region when applied to CNN-based MDE models. Furthermore, it yields a significant error of $0.34$ and exerts substantial influence over $94\%$ of the target region in the context of Transformer-based MDE.

摘要: 近年来，单目深度估计(MDE)在性能上取得了显著的进步，这在很大程度上归功于卷积神经网络(CNN)和变压器等创新体系结构的集成。然而，这些模型对对抗性攻击的易感性已经成为一个值得关注的问题，特别是在安全和安保至上的领域。这一担忧对MDE来说尤为重要，因为它在自动驾驶和机器人导航等应用中扮演着关键角色，在这些应用中，准确的场景理解至关重要。为了评估基于CNN的深度预测方法的脆弱性，最近的工作试图设计对抗MDE的对抗性补丁。然而，现有的方法不能对视觉系统造成全面和实质性的颠覆性影响。相反，他们的影响是部分的，仅限于特定的当地地区。这些方法只在与输入图像重叠的区域内导致错误的深度预测，而没有考虑目标对象的特征，例如其大小、形状和位置。在本文中，我们介绍了一种新的对抗性补丁APARATE。这个补丁能够以两种不同的方式选择性地削弱MDE：通过扭曲估计的距离或通过从自主系统的角度创造物体消失的错觉。值得注意的是，APARATE被设计为对目标对象的形状和比例敏感，其影响超出了直接接近的范围。APARATE，导致平均深度估计误差超过$0.5$，当应用于基于CNN的MDE模型时，显著影响目标区域的$99\$。此外，在基于变压器的MDE的背景下，它产生了$0.34$的显著误差，并对目标区域的$94\$产生了重大影响。



