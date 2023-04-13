# Latest Adversarial Attack Papers
**update at 2023-04-13 11:04:04**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Multi-Glimpse Network: A Robust and Efficient Classification Architecture based on Recurrent Downsampled Attention**

多瞥网络：一种基于循环下采样注意的稳健高效分类体系结构 cs.CV

Accepted at BMVC 2021

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2111.02018v2) [paper-pdf](http://arxiv.org/pdf/2111.02018v2)

**Authors**: Sia Huat Tan, Runpei Dong, Kaisheng Ma

**Abstract**: Most feedforward convolutional neural networks spend roughly the same efforts for each pixel. Yet human visual recognition is an interaction between eye movements and spatial attention, which we will have several glimpses of an object in different regions. Inspired by this observation, we propose an end-to-end trainable Multi-Glimpse Network (MGNet) which aims to tackle the challenges of high computation and the lack of robustness based on recurrent downsampled attention mechanism. Specifically, MGNet sequentially selects task-relevant regions of an image to focus on and then adaptively combines all collected information for the final prediction. MGNet expresses strong resistance against adversarial attacks and common corruptions with less computation. Also, MGNet is inherently more interpretable as it explicitly informs us where it focuses during each iteration. Our experiments on ImageNet100 demonstrate the potential of recurrent downsampled attention mechanisms to improve a single feedforward manner. For example, MGNet improves 4.76% accuracy on average in common corruptions with only 36.9% computational cost. Moreover, while the baseline incurs an accuracy drop to 7.6%, MGNet manages to maintain 44.2% accuracy in the same PGD attack strength with ResNet-50 backbone. Our code is available at https://github.com/siahuat0727/MGNet.

摘要: 大多数前馈卷积神经网络对于每个像素花费大致相同的努力。然而，人类的视觉识别是眼球运动和空间注意力之间的相互作用，我们会对不同地区的一个物体有几次瞥见。受此启发，我们提出了一种端到端可训练的多掠影网络(MGNet)，旨在解决基于循环下采样注意机制的高计算量和健壮性不足的挑战。具体地说，MGNet按顺序选择图像中与任务相关的区域进行聚焦，然后自适应地组合所有收集的信息以进行最终预测。MGNet以较少的运算量对敌意攻击和常见的腐败表现出很强的抵抗力。此外，MGNet天生更易于解释，因为它在每次迭代中明确地通知我们它关注的地方。我们在ImageNet100上的实验证明了循环下采样注意机制改善单一前馈方式的潜力。例如，MGNet在常见的腐败问题上平均提高了4.76%的准确率，而计算代价仅为36.9%。此外，当基线导致准确率下降到7.6%时，MGNet在与ResNet-50主干相同的PGD攻击强度下设法保持44.2%的准确率。我们的代码可以在https://github.com/siahuat0727/MGNet.上找到



## **2. Identification of Systematic Errors of Image Classifiers on Rare Subgroups**

稀有子群上图像分类器系统误差的辨识 cs.CV

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2303.05072v2) [paper-pdf](http://arxiv.org/pdf/2303.05072v2)

**Authors**: Jan Hendrik Metzen, Robin Hutmacher, N. Grace Hua, Valentyn Boreiko, Dan Zhang

**Abstract**: Despite excellent average-case performance of many image classifiers, their performance can substantially deteriorate on semantically coherent subgroups of the data that were under-represented in the training data. These systematic errors can impact both fairness for demographic minority groups as well as robustness and safety under domain shift. A major challenge is to identify such subgroups with subpar performance when the subgroups are not annotated and their occurrence is very rare. We leverage recent advances in text-to-image models and search in the space of textual descriptions of subgroups ("prompts") for subgroups where the target model has low performance on the prompt-conditioned synthesized data. To tackle the exponentially growing number of subgroups, we employ combinatorial testing. We denote this procedure as PromptAttack as it can be interpreted as an adversarial attack in a prompt space. We study subgroup coverage and identifiability with PromptAttack in a controlled setting and find that it identifies systematic errors with high accuracy. Thereupon, we apply PromptAttack to ImageNet classifiers and identify novel systematic errors on rare subgroups.

摘要: 尽管许多图像分类器的平均情况性能很好，但在训练数据中表示不足的数据的语义连贯子组上，它们的性能可能会显著恶化。这些系统性错误既会影响人口少数群体的公平性，也会影响领域转移下的稳健性和安全性。一个主要的挑战是在子组没有被注释并且它们的出现非常罕见的情况下，识别这样的子组具有低于标准的性能。我们利用文本到图像模型中的最新进展，并在目标模型对提示条件合成数据的性能较低的子组的子组(提示)的文本描述空间中进行搜索。为了解决指数级增长的子组数量，我们使用了组合测试。我们将这个过程表示为PromptAttack，因为它可以解释为提示空间中的对抗性攻击。我们在受控环境下研究了PromptAttack的子组复盖率和可识别性，发现它识别系统错误的准确率很高。于是，我们将PromptAttack应用于ImageNet分类器，并在稀有子群上识别出新的系统误差。



## **3. Optimal Detector Placement in Networked Control Systems under Cyber-attacks with Applications to Power Networks**

网络攻击下网络控制系统检测器的最优配置及其在电网中的应用 eess.SY

7 pages, 4 figures, accepted to IFAC 2023

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2304.05710v1) [paper-pdf](http://arxiv.org/pdf/2304.05710v1)

**Authors**: Anh Tung Nguyen, Sribalaji C. Anand, André M. H. Teixeira, Alexander Medvedev

**Abstract**: This paper proposes a game-theoretic method to address the problem of optimal detector placement in a networked control system under cyber-attacks. The networked control system is composed of interconnected agents where each agent is regulated by its local controller over unprotected communication, which leaves the system vulnerable to malicious cyber-attacks. To guarantee a given local performance, the defender optimally selects a single agent on which to place a detector at its local controller with the purpose of detecting cyber-attacks. On the other hand, an adversary optimally chooses a single agent on which to conduct a cyber-attack on its input with the aim of maximally worsening the local performance while remaining stealthy to the defender. First, we present a necessary and sufficient condition to ensure that the maximal attack impact on the local performance is bounded, which restricts the possible actions of the defender to a subset of available agents. Then, by considering the maximal attack impact on the local performance as a game payoff, we cast the problem of finding optimal actions of the defender and the adversary as a zero-sum game. Finally, with the possible action sets of the defender and the adversary, an algorithm is devoted to determining the Nash equilibria of the zero-sum game that yield the optimal detector placement. The proposed method is illustrated on an IEEE benchmark for power systems.

摘要: 针对网络控制系统中检测器的最优配置问题，提出了一种基于博弈论的方法。网络控制系统由相互连接的代理组成，其中每个代理由其本地控制器通过不受保护的通信进行管理，这使得系统容易受到恶意网络攻击。为了保证给定的本地性能，防御者最优化地选择单个代理，在其本地控制器上放置检测器，以检测网络攻击。另一方面，对手最优地选择一个代理对其输入进行网络攻击，目的是最大限度地恶化本地性能，同时保持对防御者的隐身。首先，我们给出了确保攻击对局部性能的最大影响是有界的充要条件，该条件将防御者可能的行为限制在可用代理的子集上。然后，通过考虑攻击对局部性能的最大影响作为博弈收益，将寻找防御者和对手的最优行动的问题转化为零和博弈。最后，利用防御者和对手可能的行动集，给出了一个算法来确定零和博弈中产生最优检测器配置的纳什均衡。以IEEE电力系统基准为例，说明了该方法的有效性。



## **4. SoK: Certified Robustness for Deep Neural Networks**

SOK：深度神经网络认证的健壮性 cs.LG

To appear at 2023 IEEE Symposium on Security and Privacy (SP)  (Version 8); include recent progress till Apr 2023 in Version 9; 14 pages for  the main text; benchmark & tool website:  http://sokcertifiedrobustness.github.io/

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2009.04131v9) [paper-pdf](http://arxiv.org/pdf/2009.04131v9)

**Authors**: Linyi Li, Tao Xie, Bo Li

**Abstract**: Great advances in deep neural networks (DNNs) have led to state-of-the-art performance on a wide range of tasks. However, recent studies have shown that DNNs are vulnerable to adversarial attacks, which have brought great concerns when deploying these models to safety-critical applications such as autonomous driving. Different defense approaches have been proposed against adversarial attacks, including: a) empirical defenses, which can usually be adaptively attacked again without providing robustness certification; and b) certifiably robust approaches, which consist of robustness verification providing the lower bound of robust accuracy against any attacks under certain conditions and corresponding robust training approaches. In this paper, we systematize certifiably robust approaches and related practical and theoretical implications and findings. We also provide the first comprehensive benchmark on existing robustness verification and training approaches on different datasets. In particular, we 1) provide a taxonomy for the robustness verification and training approaches, as well as summarize the methodologies for representative algorithms, 2) reveal the characteristics, strengths, limitations, and fundamental connections among these approaches, 3) discuss current research progresses, theoretical barriers, main challenges, and future directions for certifiably robust approaches for DNNs, and 4) provide an open-sourced unified platform to evaluate 20+ representative certifiably robust approaches.

摘要: 深度神经网络(DNN)的巨大进步导致了在各种任务中最先进的性能。然而，最近的研究表明，DNN很容易受到对手攻击，这在将这些模型部署到自动驾驶等安全关键型应用时带来了极大的担忧。针对敌意攻击已经提出了不同的防御方法，包括：a)经验防御，通常可以在不提供健壮性证明的情况下自适应地再次攻击；b)可证明的健壮性方法，包括在一定条件下提供对任何攻击的健壮性精度下界的健壮性验证和相应的健壮训练方法。在这篇文章中，我们系统化了可证明的稳健方法以及相关的实践和理论意义和发现。我们还提供了关于不同数据集上现有稳健性验证和训练方法的第一个全面基准。具体地说，我们1)提供了健壮性验证和训练方法的分类，并总结了典型算法的方法论；2)揭示了这些方法的特点、优势、局限性和基本联系；3)讨论了当前的研究进展、理论障碍、主要挑战和未来的发展方向；4)提供了一个开源的统一平台来评估20多个具有代表性的可证健壮方法。



## **5. Generative Adversarial Networks-Driven Cyber Threat Intelligence Detection Framework for Securing Internet of Things**

产生式对抗网络驱动的物联网安全网络威胁情报检测框架 cs.CR

The paper is accepted and will be published in the IEEE DCOSS-IoT  2023 Conference Proceedings

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2304.05644v1) [paper-pdf](http://arxiv.org/pdf/2304.05644v1)

**Authors**: Mohamed Amine Ferrag, Djallel Hamouda, Merouane Debbah, Leandros Maglaras, Abderrahmane Lakas

**Abstract**: While the benefits of 6G-enabled Internet of Things (IoT) are numerous, providing high-speed, low-latency communication that brings new opportunities for innovation and forms the foundation for continued growth in the IoT industry, it is also important to consider the security challenges and risks associated with the technology. In this paper, we propose a two-stage intrusion detection framework for securing IoTs, which is based on two detectors. In the first stage, we propose an adversarial training approach using generative adversarial networks (GAN) to help the first detector train on robust features by supplying it with adversarial examples as validation sets. Consequently, the classifier would perform very well against adversarial attacks. Then, we propose a deep learning (DL) model for the second detector to identify intrusions. We evaluated the proposed approach's efficiency in terms of detection accuracy and robustness against adversarial attacks. Experiment results with a new cyber security dataset demonstrate the effectiveness of the proposed methodology in detecting both intrusions and persistent adversarial examples with a weighted avg of 96%, 95%, 95%, and 95% for precision, recall, f1-score, and accuracy, respectively.

摘要: 虽然支持6G的物联网(IoT)带来了许多好处，提供了高速、低延迟的通信，为创新带来了新的机遇，并为物联网行业的持续增长奠定了基础，但考虑与该技术相关的安全挑战和风险也很重要。本文提出了一种基于两个检测器的两阶段物联网安全入侵检测框架。在第一阶段，我们提出了一种使用生成性对抗性网络(GAN)的对抗性训练方法，通过向第一个检测器提供对抗性实例作为验证集来帮助它训练健壮特征。因此，分类器将在对抗对手攻击时表现得非常好。然后，我们提出了一种用于第二个检测器的深度学习模型来识别入侵。我们从检测准确率和对抗攻击的健壮性两个方面对该方法的效率进行了评估。在一个新的网络安全数据集上的实验结果表明，该方法在检测入侵和持续敌意示例方面的有效性，其准确率、召回率、F1得分和准确率的加权平均值分别为96%、95%、95%和95%。



## **6. Overload: Latency Attacks on Object Detection for Edge Devices**

过载：边缘设备对象检测的延迟攻击 cs.CV

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2304.05370v2) [paper-pdf](http://arxiv.org/pdf/2304.05370v2)

**Authors**: Erh-Chung Chen, Pin-Yu Chen, I-Hsin Chung, Che-rung Lee

**Abstract**: Nowadays, the deployment of deep learning based applications on edge devices is an essential task owing to the increasing demands on intelligent services. However, the limited computing resources on edge nodes make the models vulnerable to attacks, such that the predictions made by models are unreliable. In this paper, we investigate latency attacks on deep learning applications. Unlike common adversarial attacks for misclassification, the goal of latency attacks is to increase the inference time, which may stop applications from responding to the requests within a reasonable time. This kind of attack is ubiquitous for various applications, and we use object detection to demonstrate how such kind of attacks work. We also design a framework named Overload to generate latency attacks at scale. Our method is based on a newly formulated optimization problem and a novel technique, called spatial attention, to increase the inference time of object detection. We have conducted experiments using YOLOv5 models on Nvidia NX. The experimental results show that with latency attacks, the inference time of a single image can be increased ten times longer in reference to the normal setting. Moreover, comparing to existing methods, our attacking method is simpler and more effective.

摘要: 随着人们对智能服务需求的不断增长，在边缘设备上部署基于深度学习的应用是一项必不可少的任务。然而，边缘节点有限的计算资源使得模型容易受到攻击，使得模型做出的预测是不可靠的。本文研究了深度学习应用程序中的延迟攻击。与常见的误分类对抗性攻击不同，延迟攻击的目标是增加推理时间，这可能会使应用程序在合理的时间内停止对请求的响应。这种攻击在各种应用中普遍存在，我们使用对象检测来演示这种攻击是如何工作的。我们还设计了一个名为OverLoad的框架来生成大规模的延迟攻击。我们的方法基于一个新的优化问题和一种新的技术，称为空间注意，以增加目标检测的推理时间。我们已经在NVIDIA NX上使用YOLOv5模型进行了实验。实验结果表明，在延迟攻击的情况下，单幅图像的推理时间可以比正常设置增加十倍。此外，与现有的攻击方法相比，我们的攻击方法更简单有效。



## **7. On the Adversarial Inversion of Deep Biometric Representations**

关于深层生物特征表示的对抗性反转 cs.CV

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2304.05561v1) [paper-pdf](http://arxiv.org/pdf/2304.05561v1)

**Authors**: Gioacchino Tangari, Shreesh Keskar, Hassan Jameel Asghar, Dali Kaafar

**Abstract**: Biometric authentication service providers often claim that it is not possible to reverse-engineer a user's raw biometric sample, such as a fingerprint or a face image, from its mathematical (feature-space) representation. In this paper, we investigate this claim on the specific example of deep neural network (DNN) embeddings. Inversion of DNN embeddings has been investigated for explaining deep image representations or synthesizing normalized images. Existing studies leverage full access to all layers of the original model, as well as all possible information on the original dataset. For the biometric authentication use case, we need to investigate this under adversarial settings where an attacker has access to a feature-space representation but no direct access to the exact original dataset nor the original learned model. Instead, we assume varying degree of attacker's background knowledge about the distribution of the dataset as well as the original learned model (architecture and training process). In these cases, we show that the attacker can exploit off-the-shelf DNN models and public datasets, to mimic the behaviour of the original learned model to varying degrees of success, based only on the obtained representation and attacker's prior knowledge. We propose a two-pronged attack that first infers the original DNN by exploiting the model footprint on the embedding, and then reconstructs the raw data by using the inferred model. We show the practicality of the attack on popular DNNs trained for two prominent biometric modalities, face and fingerprint recognition. The attack can effectively infer the original recognition model (mean accuracy 83\% for faces, 86\% for fingerprints), and can craft effective biometric reconstructions that are successfully authenticated with 1-vs-1 authentication accuracy of up to 92\% for some models.

摘要: 生物特征认证服务提供商经常声称，不可能从其数学(特征空间)表示中逆向工程用户的原始生物特征样本，例如指纹或面部图像。在本文中，我们以深度神经网络(DNN)嵌入的具体例子来研究这一论断。DNN嵌入的逆已被用于解释深层图像表示或合成归一化图像。现有研究充分利用了对原始模型的所有层的访问权限，以及原始数据集的所有可能信息。对于生物认证用例，我们需要在敌意设置下调查这一点，在这种情况下，攻击者可以访问特征空间表示，但不能直接访问确切的原始数据集或原始学习模型。相反，我们假设攻击者对数据集的分布以及原始学习模型(体系结构和训练过程)有不同程度的背景知识。在这些情况下，我们表明攻击者可以利用现成的DNN模型和公共数据集，仅基于获得的表示和攻击者的先验知识来模仿原始学习模型的行为，以获得不同程度的成功。我们提出了一种双管齐下的攻击方法，首先利用嵌入的模型足迹推断出原始的DNN，然后利用推断出的模型重构原始数据。我们展示了该攻击对两种主要的生物识别模式--人脸和指纹识别--训练的流行DNN的实用性。该攻击可以有效地推断出原始的识别模型(人脸的平均准确率为83%，指纹的平均准确率为86%)，并且可以成功地构造出有效的生物特征重建，对于某些模型，1-vs-1的认证准确率高达92%。



## **8. Unfooling Perturbation-Based Post Hoc Explainers**

基于非愚弄扰动的帖子随机解说器 cs.AI

Accepted to AAAI-23. See the companion blog post at  https://medium.com/@craymichael/noncompliance-in-algorithmic-audits-and-defending-auditors-5b9fbdab2615.  9 pages (not including references and supplemental)

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2205.14772v3) [paper-pdf](http://arxiv.org/pdf/2205.14772v3)

**Authors**: Zachariah Carmichael, Walter J Scheirer

**Abstract**: Monumental advancements in artificial intelligence (AI) have lured the interest of doctors, lenders, judges, and other professionals. While these high-stakes decision-makers are optimistic about the technology, those familiar with AI systems are wary about the lack of transparency of its decision-making processes. Perturbation-based post hoc explainers offer a model agnostic means of interpreting these systems while only requiring query-level access. However, recent work demonstrates that these explainers can be fooled adversarially. This discovery has adverse implications for auditors, regulators, and other sentinels. With this in mind, several natural questions arise - how can we audit these black box systems? And how can we ascertain that the auditee is complying with the audit in good faith? In this work, we rigorously formalize this problem and devise a defense against adversarial attacks on perturbation-based explainers. We propose algorithms for the detection (CAD-Detect) and defense (CAD-Defend) of these attacks, which are aided by our novel conditional anomaly detection approach, KNN-CAD. We demonstrate that our approach successfully detects whether a black box system adversarially conceals its decision-making process and mitigates the adversarial attack on real-world data for the prevalent explainers, LIME and SHAP.

摘要: 人工智能(AI)的巨大进步吸引了医生、贷款人、法官和其他专业人士的兴趣。尽管这些事关重大的决策者对这项技术持乐观态度，但那些熟悉人工智能系统的人对其决策过程缺乏透明度持谨慎态度。基于扰动的后自组织解释器提供了一种模型不可知的方法来解释这些系统，而只需要查询级别的访问。然而，最近的研究表明，这些解释程序可能会被相反的人愚弄。这一发现对审计师、监管者和其他哨兵产生了不利影响。考虑到这一点，几个自然的问题就产生了--我们如何审计这些黑匣子系统？我们如何确定被审计人是真诚地遵守审计的？在这项工作中，我们严格地形式化了这个问题，并设计了一个防御对基于扰动的解释器的敌意攻击。在新的条件异常检测方法KNN-CAD的辅助下，我们提出了针对这些攻击的检测(CAD-检测)和防御(CAD-防御)算法。我们证明，我们的方法成功地检测到黑盒系统是否恶意地隐藏了其决策过程，并缓解了流行的解释程序LIME和Shap对真实数据的恶意攻击。



## **9. Existence and Minimax Theorems for Adversarial Surrogate Risks in Binary Classification**

二元分类中对抗性代理风险的存在性和极大极小定理 cs.LG

37 pages. version 2: corrects several errors and employs a  significantly different proof technique. version 3: modifies the arXiv author  list but has no other changes

**SubmitDate**: 2023-04-11    [abs](http://arxiv.org/abs/2206.09098v3) [paper-pdf](http://arxiv.org/pdf/2206.09098v3)

**Authors**: Natalie S. Frank, Jonathan Niles-Weed

**Abstract**: Adversarial training is one of the most popular methods for training methods robust to adversarial attacks, however, it is not well-understood from a theoretical perspective. We prove and existence, regularity, and minimax theorems for adversarial surrogate risks. Our results explain some empirical observations on adversarial robustness from prior work and suggest new directions in algorithm development. Furthermore, our results extend previously known existence and minimax theorems for the adversarial classification risk to surrogate risks.

摘要: 对抗性训练是对抗攻击能力最强的训练方法之一，但从理论上对它的理解还不够深入。我们证明了对抗性代理风险的存在性、正则性和极大极小定理。我们的结果解释了以前工作中关于对手稳健性的一些经验观察，并为算法开发提供了新的方向。此外，我们的结果推广了已知的对抗性分类风险到代理风险的存在性和极大极小定理。



## **10. Fiat-Shamir for Proofs Lacks a Proof Even in the Presence of Shared Entanglement**

菲亚特-沙米尔的证据缺乏证据，即使在存在共同纠缠的情况下 quant-ph

63 pages, 2 figures

**SubmitDate**: 2023-04-11    [abs](http://arxiv.org/abs/2204.02265v3) [paper-pdf](http://arxiv.org/pdf/2204.02265v3)

**Authors**: Frédéric Dupuis, Philippe Lamontagne, Louis Salvail

**Abstract**: We explore the cryptographic power of arbitrary shared physical resources. The most general such resource is access to a fresh entangled quantum state at the outset of each protocol execution. We call this the Common Reference Quantum State (CRQS) model, in analogy to the well-known Common Reference String (CRS). The CRQS model is a natural generalization of the CRS model but appears to be more powerful: in the two-party setting, a CRQS can sometimes exhibit properties associated with a Random Oracle queried once by measuring a maximally entangled state in one of many mutually unbiased bases. We formalize this notion as a Weak One-Time Random Oracle (WOTRO), where we only ask of the $m$--bit output to have some randomness when conditioned on the $n$--bit input.   We show that when $n-m\in\omega(\lg n)$, any protocol for WOTRO in the CRQS model can be attacked by an (inefficient) adversary. Moreover, our adversary is efficiently simulatable, which rules out the possibility of proving the computational security of a scheme by a black-box reduction to a cryptographic game assumption. On the other hand, we introduce a non-game quantum assumption for hash functions that implies WOTRO in the CRQ\$ model (where the CRQS consists only of EPR pairs). We first build a statistically secure WOTRO protocol where $m=n$, then hash the output.   The impossibility of WOTRO has the following consequences. First, we show the black-box impossibility of a quantum Fiat-Shamir transform, extending the impossibility result of Bitansky et al. (TCC '13) to the CRQS model. Second, we show a black-box impossibility result for a strenghtened version of quantum lightning (Zhandry, Eurocrypt '19) where quantum bolts have an additional parameter that cannot be changed without generating new bolts.

摘要: 我们探索任意共享物理资源的加密能力。最常见的这类资源是在每个协议执行开始时访问新的纠缠量子态。我们称之为公共参考量子态(CRQS)模型，类似于众所周知的公共参考弦(CRS)。CRQS模型是CRS模型的自然推广，但似乎更强大：在两方设置中，CRQS有时可以通过测量许多相互无偏的碱基之一中的最大纠缠态来展示与查询一次的随机Oracle相关联的属性。我们将这个概念形式化为弱一次性随机Oracle(WOTRO)，其中我们只要求$m$位的输出在以$n$位输入为条件时具有一定的随机性。我们证明了当$n-m\in\omega(\lg n)$时，CRQS模型中用于WOTRO的任何协议都可以被(低效的)攻击者攻击。此外，我们的对手是高效可模拟的，这排除了通过将黑盒简化为密码博弈假设来证明方案的计算安全性的可能性。另一方面，我们引入了散列函数的非博弈量子假设，在CRQ模型(其中CRQS只由EPR对组成)中隐含了WOTRO。我们首先构建一个统计安全的WOTRO协议，其中$m=n$，然后对输出进行散列。WOTRO的不可能性会产生以下后果。首先，我们证明了量子Fiat-Shamir变换的黑盒不可能性，推广了Bitansky等人的不可能结果。其次，我们给出了一个加强版量子闪电(Zhandry，Eurocrypt‘19)的黑箱不可能结果，其中量子闪电有一个额外的参数，如果不产生新的闪电，这个参数就不能改变。



## **11. MENLI: Robust Evaluation Metrics from Natural Language Inference**

MENLI：自然语言推理中的稳健评价指标 cs.CL

TACL 2023 Camera-ready; github link fixed+Fig.3 legend fixed

**SubmitDate**: 2023-04-11    [abs](http://arxiv.org/abs/2208.07316v4) [paper-pdf](http://arxiv.org/pdf/2208.07316v4)

**Authors**: Yanran Chen, Steffen Eger

**Abstract**: Recently proposed BERT-based evaluation metrics for text generation perform well on standard benchmarks but are vulnerable to adversarial attacks, e.g., relating to information correctness. We argue that this stems (in part) from the fact that they are models of semantic similarity. In contrast, we develop evaluation metrics based on Natural Language Inference (NLI), which we deem a more appropriate modeling. We design a preference-based adversarial attack framework and show that our NLI based metrics are much more robust to the attacks than the recent BERT-based metrics. On standard benchmarks, our NLI based metrics outperform existing summarization metrics, but perform below SOTA MT metrics. However, when combining existing metrics with our NLI metrics, we obtain both higher adversarial robustness (15%-30%) and higher quality metrics as measured on standard benchmarks (+5% to 30%).

摘要: 最近提出的基于BERT的文本生成评估指标在标准基准上表现良好，但容易受到敌意攻击，例如与信息正确性有关的攻击。我们认为，这(部分)源于这样一个事实：它们是语义相似性的模型。相比之下，我们基于自然语言推理(NLI)开发评估指标，我们认为这是更合适的建模。我们设计了一个基于偏好的对抗性攻击框架，并表明我们的基于NLI的度量比最近的基于BERT的度量具有更强的抗攻击能力。在标准基准测试中，我们基于NLI的指标优于现有的摘要指标，但低于SOTA MT指标。然而，当将现有指标与我们的NLI指标相结合时，我们获得了更高的对手健壮性(15%-30%)和标准基准测试的更高质量指标(+5%到30%)。



## **12. Visually Adversarial Attacks and Defenses in the Physical World: A Survey**

物理世界中的视觉对抗性攻击和防御：综述 cs.CV

**SubmitDate**: 2023-04-11    [abs](http://arxiv.org/abs/2211.01671v3) [paper-pdf](http://arxiv.org/pdf/2211.01671v3)

**Authors**: Xingxing Wei, Bangzheng Pu, Jiefan Lu, Baoyuan Wu

**Abstract**: Although Deep Neural Networks (DNNs) have been widely applied in various real-world scenarios, they are vulnerable to adversarial examples. The current adversarial attacks in computer vision can be divided into digital attacks and physical attacks according to their different attack forms. Compared with digital attacks, which generate perturbations in the digital pixels, physical attacks are more practical in the real world. Owing to the serious security problem caused by physically adversarial examples, many works have been proposed to evaluate the physically adversarial robustness of DNNs in the past years. In this paper, we summarize a survey versus the current physically adversarial attacks and physically adversarial defenses in computer vision. To establish a taxonomy, we organize the current physical attacks from attack tasks, attack forms, and attack methods, respectively. Thus, readers can have a systematic knowledge of this topic from different aspects. For the physical defenses, we establish the taxonomy from pre-processing, in-processing, and post-processing for the DNN models to achieve full coverage of the adversarial defenses. Based on the above survey, we finally discuss the challenges of this research field and further outlook on the future direction.

摘要: 尽管深度神经网络(DNN)已被广泛应用于各种现实场景中，但它们很容易受到对手例子的影响。根据攻击形式的不同，目前计算机视觉中的对抗性攻击可分为数字攻击和物理攻击。与在数字像素中产生扰动的数字攻击相比，物理攻击在现实世界中更实用。由于物理对抗实例带来了严重的安全问题，在过去的几年里，人们已经提出了许多工作来评估DNN的物理对抗健壮性。本文对当前计算机视觉中的身体对抗攻击和身体对抗防御进行了综述。为了建立分类，我们分别从攻击任务、攻击形式和攻击方法三个方面对当前的物理攻击进行了组织。因此，读者可以从不同的方面对这一主题有一个系统的了解。对于物理防御，我们从DNN模型的前处理、内处理和后处理三个方面建立了分类，以实现对抗性防御的全覆盖。在上述调查的基础上，我们最后讨论了该研究领域面临的挑战和对未来方向的进一步展望。



## **13. A Game-theoretic Framework for Federated Learning**

一种联合学习的博弈论框架 cs.LG

**SubmitDate**: 2023-04-11    [abs](http://arxiv.org/abs/2304.05836v1) [paper-pdf](http://arxiv.org/pdf/2304.05836v1)

**Authors**: Xiaojin Zhang, Lixin Fan, Siwei Wang, Wenjie Li, Kai Chen, Qiang Yang

**Abstract**: In federated learning, benign participants aim to optimize a global model collaboratively. However, the risk of \textit{privacy leakage} cannot be ignored in the presence of \textit{semi-honest} adversaries. Existing research has focused either on designing protection mechanisms or on inventing attacking mechanisms. While the battle between defenders and attackers seems never-ending, we are concerned with one critical question: is it possible to prevent potential attacks in advance? To address this, we propose the first game-theoretic framework that considers both FL defenders and attackers in terms of their respective payoffs, which include computational costs, FL model utilities, and privacy leakage risks. We name this game the Federated Learning Security Game (FLSG), in which neither defenders nor attackers are aware of all participants' payoffs.   To handle the \textit{incomplete information} inherent in this situation, we propose associating the FLSG with an \textit{oracle} that has two primary responsibilities. First, the oracle provides lower and upper bounds of the payoffs for the players. Second, the oracle acts as a correlation device, privately providing suggested actions to each player. With this novel framework, we analyze the optimal strategies of defenders and attackers. Furthermore, we derive and demonstrate conditions under which the attacker, as a rational decision-maker, should always follow the oracle's suggestion \textit{not to attack}.

摘要: 在联合学习中，良性参与者的目标是协作优化全球模型。然而，在存在半诚实的对手的情况下，隐私泄露的风险是不容忽视的。现有的研究要么集中在设计保护机制上，要么集中在发明攻击机制上。虽然防御者和攻击者之间的战斗似乎永无止境，但我们关心的是一个关键问题：是否有可能提前防止潜在的攻击？为了解决这一问题，我们提出了第一个博弈论框架，该框架考虑了FL防御者和攻击者各自的收益，其中包括计算成本、FL模型效用和隐私泄露风险。我们将这款游戏命名为联邦学习安全游戏(FLSG)，在该游戏中，防御者和攻击者都不知道所有参与者的收益。为了处理这种情况下固有的不完整信息，我们建议将FLSG与具有两个主要职责的\textit{Oracle}相关联。首先，先知为玩家提供了收益的上下限。其次，先知充当了关联设备，私下向每个玩家提供建议的动作。在此框架下，我们分析了防御者和攻击者的最优策略。此外，我们还推导并证明了攻击者作为理性决策者应始终遵循神谕的建议的条件。



## **14. RecUP-FL: Reconciling Utility and Privacy in Federated Learning via User-configurable Privacy Defense**

RECOP-FL：通过用户可配置的隐私防御在联合学习中协调效用和隐私 cs.LG

**SubmitDate**: 2023-04-11    [abs](http://arxiv.org/abs/2304.05135v1) [paper-pdf](http://arxiv.org/pdf/2304.05135v1)

**Authors**: Yue Cui, Syed Irfan Ali Meerza, Zhuohang Li, Luyang Liu, Jiaxin Zhang, Jian Liu

**Abstract**: Federated learning (FL) provides a variety of privacy advantages by allowing clients to collaboratively train a model without sharing their private data. However, recent studies have shown that private information can still be leaked through shared gradients. To further minimize the risk of privacy leakage, existing defenses usually require clients to locally modify their gradients (e.g., differential privacy) prior to sharing with the server. While these approaches are effective in certain cases, they regard the entire data as a single entity to protect, which usually comes at a large cost in model utility. In this paper, we seek to reconcile utility and privacy in FL by proposing a user-configurable privacy defense, RecUP-FL, that can better focus on the user-specified sensitive attributes while obtaining significant improvements in utility over traditional defenses. Moreover, we observe that existing inference attacks often rely on a machine learning model to extract the private information (e.g., attributes). We thus formulate such a privacy defense as an adversarial learning problem, where RecUP-FL generates slight perturbations that can be added to the gradients before sharing to fool adversary models. To improve the transferability to un-queryable black-box adversary models, inspired by the idea of meta-learning, RecUP-FL forms a model zoo containing a set of substitute models and iteratively alternates between simulations of the white-box and the black-box adversarial attack scenarios to generate perturbations. Extensive experiments on four datasets under various adversarial settings (both attribute inference attack and data reconstruction attack) show that RecUP-FL can meet user-specified privacy constraints over the sensitive attributes while significantly improving the model utility compared with state-of-the-art privacy defenses.

摘要: 联合学习(FL)通过允许客户在不共享他们的私人数据的情况下协作地训练模型来提供各种隐私优势。然而，最近的研究表明，私人信息仍然可以通过共享的梯度泄露。为了进一步最小化隐私泄露的风险，现有的防御措施通常要求客户端在与服务器共享之前本地修改其梯度(例如，差异隐私)。虽然这些方法在某些情况下是有效的，但它们将整个数据视为要保护的单个实体，这在模型实用程序中通常会带来很大的成本。在本文中，我们试图在FL中调和效用和隐私，提出了一种用户可配置的隐私防御机制Recup-FL，它可以更好地关注用户指定的敏感属性，同时在效用方面比传统防御机制有显著的提高。此外，我们观察到现有的推理攻击往往依赖于机器学习模型来提取私有信息(例如属性)。因此，我们将这样的隐私防御描述为对抗性学习问题，其中Recup-FL产生轻微的扰动，这些扰动可以在分享之前添加到梯度中，以愚弄对手模型。为了提高对不可查询的黑盒攻击模型的可转换性，受元学习思想的启发，Recup-FL形成一个包含一组替代模型的模型动物园，并在模拟白盒和黑盒攻击场景之间迭代交替产生扰动。在四个数据集上的大量实验表明，在不同的敌意环境下(包括属性推理攻击和数据重构攻击)，Recup-FL在满足用户对敏感属性的隐私约束的同时，与最新的隐私防御相比，显著提高了模型的实用性。



## **15. EGC: Image Generation and Classification via a Single Energy-Based Model**

EGC：基于单一能量模型的图像生成和分类 cs.CV

**SubmitDate**: 2023-04-11    [abs](http://arxiv.org/abs/2304.02012v2) [paper-pdf](http://arxiv.org/pdf/2304.02012v2)

**Authors**: Qiushan Guo, Chuofan Ma, Yi Jiang, Zehuan Yuan, Yizhou Yu, Ping Luo

**Abstract**: Learning image classification and image generation using the same set of network parameters is a challenging problem. Recent advanced approaches perform well in one task often exhibit poor performance in the other. This work introduces an energy-based classifier and generator, namely EGC, which can achieve superior performance in both tasks using a single neural network. Unlike a conventional classifier that outputs a label given an image (i.e., a conditional distribution $p(y|\mathbf{x})$), the forward pass in EGC is a classifier that outputs a joint distribution $p(\mathbf{x},y)$, enabling an image generator in its backward pass by marginalizing out the label $y$. This is done by estimating the energy and classification probability given a noisy image in the forward pass, while denoising it using the score function estimated in the backward pass. EGC achieves competitive generation results compared with state-of-the-art approaches on ImageNet-1k, CelebA-HQ and LSUN Church, while achieving superior classification accuracy and robustness against adversarial attacks on CIFAR-10. This work represents the first successful attempt to simultaneously excel in both tasks using a single set of network parameters. We believe that EGC bridges the gap between discriminative and generative learning.

摘要: 使用相同的网络参数学习图像分类和图像生成是一个具有挑战性的问题。最近的高级方法在一项任务中表现良好，但在另一项任务中往往表现不佳。这项工作介绍了一种基于能量的分类器和生成器，即EGC，它可以使用单个神经网络在这两个任务中获得优越的性能。与输出给定图像的标签(即，条件分布$p(y|\mathbf{x})$)的传统分类器不同，EGC中的前向通道是输出联合分布$p(\mathbf{x}，y)$的分类器，从而使图像生成器在其后向通道中通过边缘化标签$y$来实现。这是通过在前传中给出噪声图像的情况下估计能量和分类概率来完成的，同时使用在后传中估计的得分函数来对其进行去噪。EGC在ImageNet-1k、CelebA-HQ和LSUN Church上实现了与最先进的方法相比具有竞争力的生成结果，同时在CIFAR-10上获得了卓越的分类准确性和对对手攻击的稳健性。这项工作代表了首次成功尝试使用一组网络参数同时在两项任务中脱颖而出。我们认为，EGC弥合了歧视性学习和生成性学习之间的差距。



## **16. Non-Asymptotic Lower Bounds For Training Data Reconstruction**

训练数据重构的非渐近下界 cs.LG

Corrected minor typos

**SubmitDate**: 2023-04-11    [abs](http://arxiv.org/abs/2303.16372v3) [paper-pdf](http://arxiv.org/pdf/2303.16372v3)

**Authors**: Prateeti Mukherjee, Satya Lokam

**Abstract**: We investigate semantic guarantees of private learning algorithms for their resilience to training Data Reconstruction Attacks (DRAs) by informed adversaries. To this end, we derive non-asymptotic minimax lower bounds on the adversary's reconstruction error against learners that satisfy differential privacy (DP) and metric differential privacy (mDP). Furthermore, we demonstrate that our lower bound analysis for the latter also covers the high dimensional regime, wherein, the input data dimensionality may be larger than the adversary's query budget. Motivated by the theoretical improvements conferred by metric DP, we extend the privacy analysis of popular deep learning algorithms such as DP-SGD and Projected Noisy SGD to cover the broader notion of metric differential privacy.

摘要: 我们研究了私有学习算法的语义保证，因为它们对来自知情对手的训练数据重建攻击(DRA)具有韧性。为此，我们得到了满足差分隐私(DP)和度量差分隐私(MDP)的敌手重构误差的非渐近极大下界。此外，我们证明了对后者的下界分析也涵盖了高维机制，其中，输入数据的维度可能大于对手的查询预算。在度量DP的理论改进的推动下，我们扩展了DP-SGD和Projected Noise SGD等流行深度学习算法的隐私分析，以涵盖更广泛的度量差异隐私的概念。



## **17. Benchmarking the Physical-world Adversarial Robustness of Vehicle Detection**

对车辆检测的物理世界对抗健壮性进行基准测试 cs.CV

CVPR 2023 workshop

**SubmitDate**: 2023-04-11    [abs](http://arxiv.org/abs/2304.05098v1) [paper-pdf](http://arxiv.org/pdf/2304.05098v1)

**Authors**: Tianyuan Zhang, Yisong Xiao, Xiaoya Zhang, Hao Li, Lu Wang

**Abstract**: Adversarial attacks in the physical world can harm the robustness of detection models. Evaluating the robustness of detection models in the physical world can be challenging due to the time-consuming and labor-intensive nature of many experiments. Thus, virtual simulation experiments can provide a solution to this challenge. However, there is no unified detection benchmark based on virtual simulation environment. To address this challenge, we proposed an instant-level data generation pipeline based on the CARLA simulator. Using this pipeline, we generated the DCI dataset and conducted extensive experiments on three detection models and three physical adversarial attacks. The dataset covers 7 continuous and 1 discrete scenes, with over 40 angles, 20 distances, and 20,000 positions. The results indicate that Yolo v6 had strongest resistance, with only a 6.59% average AP drop, and ASA was the most effective attack algorithm with a 14.51% average AP reduction, twice that of other algorithms. Static scenes had higher recognition AP, and results under different weather conditions were similar. Adversarial attack algorithm improvement may be approaching its 'limitation'.

摘要: 物理世界中的敌意攻击可能会损害检测模型的健壮性。由于许多实验的耗时和劳动密集性，评估物理世界中检测模型的健壮性可能是具有挑战性的。因此，虚拟仿真实验可以为这一挑战提供解决方案。然而，目前还没有一个统一的基于虚拟仿真环境的检测基准。为了应对这一挑战，我们提出了一种基于CALA模拟器的即时级数据生成流水线。使用这条管道，我们生成了DCI数据集，并在三个检测模型和三个物理对手攻击上进行了广泛的实验。数据集涵盖了7个连续和1个离散的场景，超过40个角度，20个距离，20,000个位置。结果表明，Yolo V6算法抵抗能力最强，平均AP降幅仅为6.59%，而ASA算法是最有效的攻击算法，平均AP降幅为14.51%，是其他算法的两倍。静态场景具有较高的识别率，不同天气条件下的识别结果相似。对抗性攻击算法的改进可能正在接近其“极限”。



## **18. Simultaneous Adversarial Attacks On Multiple Face Recognition System Components**

对多个人脸识别系统组件的同时敌意攻击 cs.CV

**SubmitDate**: 2023-04-11    [abs](http://arxiv.org/abs/2304.05048v1) [paper-pdf](http://arxiv.org/pdf/2304.05048v1)

**Authors**: Inderjeet Singh, Kazuya Kakizaki, Toshinori Araki

**Abstract**: In this work, we investigate the potential threat of adversarial examples to the security of face recognition systems. Although previous research has explored the adversarial risk to individual components of FRSs, our study presents an initial exploration of an adversary simultaneously fooling multiple components: the face detector and feature extractor in an FRS pipeline. We propose three multi-objective attacks on FRSs and demonstrate their effectiveness through a preliminary experimental analysis on a target system. Our attacks achieved up to 100% Attack Success Rates against both the face detector and feature extractor and were able to manipulate the face detection probability by up to 50% depending on the adversarial objective. This research identifies and examines novel attack vectors against FRSs and suggests possible ways to augment the robustness by leveraging the attack vector's knowledge during training of an FRS's components.

摘要: 在这项工作中，我们研究了敌意例子对人脸识别系统安全的潜在威胁。虽然以前的研究已经探索了FRS的单个组件的对抗风险，但我们的研究提供了一个初步的探索，即敌手同时愚弄多个组件：FRS管道中的人脸检测器和特征提取器。我们提出了三种针对FRS的多目标攻击，并通过对一个目标系统的初步实验分析，证明了它们的有效性。我们的攻击对人脸检测器和特征提取器的攻击成功率都达到了100%，并且能够根据对手的目标将人脸检测概率操纵高达50%。本研究识别和检查了针对FRS的新攻击矢量，并提出了在FRS组件的训练过程中利用攻击矢量的知识来增强稳健性的可能方法。



## **19. How many dimensions are required to find an adversarial example?**

需要多少维度才能找到对抗性的例子？ cs.LG

Comments welcome! V2: minor edits for clarity

**SubmitDate**: 2023-04-11    [abs](http://arxiv.org/abs/2303.14173v2) [paper-pdf](http://arxiv.org/pdf/2303.14173v2)

**Authors**: Charles Godfrey, Henry Kvinge, Elise Bishoff, Myles Mckay, Davis Brown, Tim Doster, Eleanor Byler

**Abstract**: Past work exploring adversarial vulnerability have focused on situations where an adversary can perturb all dimensions of model input. On the other hand, a range of recent works consider the case where either (i) an adversary can perturb a limited number of input parameters or (ii) a subset of modalities in a multimodal problem. In both of these cases, adversarial examples are effectively constrained to a subspace $V$ in the ambient input space $\mathcal{X}$. Motivated by this, in this work we investigate how adversarial vulnerability depends on $\dim(V)$. In particular, we show that the adversarial success of standard PGD attacks with $\ell^p$ norm constraints behaves like a monotonically increasing function of $\epsilon (\frac{\dim(V)}{\dim \mathcal{X}})^{\frac{1}{q}}$ where $\epsilon$ is the perturbation budget and $\frac{1}{p} + \frac{1}{q} =1$, provided $p > 1$ (the case $p=1$ presents additional subtleties which we analyze in some detail). This functional form can be easily derived from a simple toy linear model, and as such our results land further credence to arguments that adversarial examples are endemic to locally linear models on high dimensional spaces.

摘要: 过去探索对手脆弱性的工作主要集中在对手可以扰乱模型输入的所有维度的情况。另一方面，最近的一系列工作考虑了这样的情况：(I)对手可以扰动有限数量的输入参数或(Ii)多通道问题中的一组通道。在这两种情况下，敌意示例都被有效地约束到环境输入空间$\mathcal{X}$中的子空间$V$。受此启发，在本工作中，我们研究了对手脆弱性是如何依赖于$\dim(V)$的。特别地，我们证明了具有$^p$范数约束的标准PGD攻击的对抗成功表现为$\epsilon(\frac{\dim(V)}{\dim\mathcal{X}})^{\frac{1}{q}}$的单调递增函数，其中$\epsilon$是扰动预算，而$\frac{1}{p}+\frac{1}{q}=1$，假设$p>1$($p=1$给出了更多的细节，我们进行了一些详细的分析)。这种函数形式可以很容易地从一个简单的玩具线性模型中得到，因此我们的结果进一步证明了高维空间上的对抗性例子是局部线性模型特有的。



## **20. Gradient-based Uncertainty Attribution for Explainable Bayesian Deep Learning**

基于梯度的可解释贝叶斯深度学习不确定性属性 cs.LG

Accepted to CVPR 2023

**SubmitDate**: 2023-04-10    [abs](http://arxiv.org/abs/2304.04824v1) [paper-pdf](http://arxiv.org/pdf/2304.04824v1)

**Authors**: Hanjing Wang, Dhiraj Joshi, Shiqiang Wang, Qiang Ji

**Abstract**: Predictions made by deep learning models are prone to data perturbations, adversarial attacks, and out-of-distribution inputs. To build a trusted AI system, it is therefore critical to accurately quantify the prediction uncertainties. While current efforts focus on improving uncertainty quantification accuracy and efficiency, there is a need to identify uncertainty sources and take actions to mitigate their effects on predictions. Therefore, we propose to develop explainable and actionable Bayesian deep learning methods to not only perform accurate uncertainty quantification but also explain the uncertainties, identify their sources, and propose strategies to mitigate the uncertainty impacts. Specifically, we introduce a gradient-based uncertainty attribution method to identify the most problematic regions of the input that contribute to the prediction uncertainty. Compared to existing methods, the proposed UA-Backprop has competitive accuracy, relaxed assumptions, and high efficiency. Moreover, we propose an uncertainty mitigation strategy that leverages the attribution results as attention to further improve the model performance. Both qualitative and quantitative evaluations are conducted to demonstrate the effectiveness of our proposed methods.

摘要: 深度学习模型所做的预测容易受到数据扰动、对抗性攻击和分布外输入的影响。因此，为了建立一个可信的人工智能系统，准确地量化预测不确定性是至关重要的。虽然目前的努力侧重于提高不确定性量化的准确性和效率，但仍有必要确定不确定性来源并采取行动减轻其对预测的影响。因此，我们建议开发可解释和可操作的贝叶斯深度学习方法，不仅可以执行准确的不确定性量化，还可以解释不确定性，识别其来源，并提出缓解不确定性影响的策略。具体地说，我们引入了一种基于梯度的不确定性归因方法来识别导致预测不确定性的输入中最有问题的区域。与现有方法相比，提出的UA-Backprop具有相当的精度、宽松的假设和较高的效率。此外，我们提出了一种不确定性缓解策略，该策略利用属性结果作为关注点来进一步提高模型的性能。我们进行了定性和定量的评估，以证明我们提出的方法的有效性。



## **21. Language-Driven Anchors for Zero-Shot Adversarial Robustness**

语言驱动的零射击对抗稳健性锚 cs.CV

11 pages

**SubmitDate**: 2023-04-10    [abs](http://arxiv.org/abs/2301.13096v2) [paper-pdf](http://arxiv.org/pdf/2301.13096v2)

**Authors**: Xiao Li, Wei Zhang, Yining Liu, Zhanhao Hu, Bo Zhang, Xiaolin Hu

**Abstract**: Deep neural networks are known to be susceptible to adversarial attacks. In this work, we focus on improving adversarial robustness in the challenging zero-shot image classification setting. To address this issue, we propose LAAT, a novel Language-driven, Anchor-based Adversarial Training strategy. LAAT utilizes a text encoder to generate fixed anchors (normalized feature embeddings) for each category and then uses these anchors for adversarial training. By leveraging the semantic consistency of the text encoders, LAAT can enhance the adversarial robustness of the image model on novel categories without additional examples. We identify the large cosine similarity problem of recent text encoders and design several effective techniques to address it. The experimental results demonstrate that LAAT significantly improves zero-shot adversarial performance, outperforming previous state-of-the-art adversarially robust one-shot methods. Moreover, our method produces substantial zero-shot adversarial robustness when models are trained on large datasets such as ImageNet-1K and applied to several downstream datasets.

摘要: 众所周知，深度神经网络容易受到对抗性攻击。在这项工作中，我们专注于在具有挑战性的零镜头图像分类环境下提高对手的稳健性。为了解决这一问题，我们提出了一种新颖的语言驱动的、基于锚的对抗性训练策略LAAT。LAAT利用文本编码器为每个类别生成固定锚(归一化特征嵌入)，然后使用这些锚进行对抗性训练。通过利用文本编码器的语义一致性，LAAT可以在不增加额外示例的情况下增强图像模型在新类别上的对抗性健壮性。针对目前文本编码器存在的大余弦相似性问题，设计了几种有效的技术来解决这一问题。实验结果表明，LAAT显著提高了零射击对抗的性能，优于以往最先进的对抗健壮的一次射击方法。此外，当模型在大型数据集(如ImageNet-1K)上训练并应用于多个下游数据集时，我们的方法产生了相当大的零命中率对抗健壮性。



## **22. Reinforcement Learning-Based Black-Box Model Inversion Attacks**

基于强化学习的黑盒模型反转攻击 cs.LG

CVPR 2023, Accepted

**SubmitDate**: 2023-04-10    [abs](http://arxiv.org/abs/2304.04625v1) [paper-pdf](http://arxiv.org/pdf/2304.04625v1)

**Authors**: Gyojin Han, Jaehyun Choi, Haeil Lee, Junmo Kim

**Abstract**: Model inversion attacks are a type of privacy attack that reconstructs private data used to train a machine learning model, solely by accessing the model. Recently, white-box model inversion attacks leveraging Generative Adversarial Networks (GANs) to distill knowledge from public datasets have been receiving great attention because of their excellent attack performance. On the other hand, current black-box model inversion attacks that utilize GANs suffer from issues such as being unable to guarantee the completion of the attack process within a predetermined number of query accesses or achieve the same level of performance as white-box attacks. To overcome these limitations, we propose a reinforcement learning-based black-box model inversion attack. We formulate the latent space search as a Markov Decision Process (MDP) problem and solve it with reinforcement learning. Our method utilizes the confidence scores of the generated images to provide rewards to an agent. Finally, the private data can be reconstructed using the latent vectors found by the agent trained in the MDP. The experiment results on various datasets and models demonstrate that our attack successfully recovers the private information of the target model by achieving state-of-the-art attack performance. We emphasize the importance of studies on privacy-preserving machine learning by proposing a more advanced black-box model inversion attack.

摘要: 模型反转攻击是一种隐私攻击，它仅通过访问模型来重建用于训练机器学习模型的私人数据。最近，利用生成性对抗网络(GANS)从公开数据集中提取知识的白盒模型反转攻击因其良好的攻击性能而受到极大的关注。另一方面，当前利用Gans的黑盒模型反转攻击存在无法保证在预定的查询访问次数内完成攻击过程或无法达到与白盒攻击相同的性能水平等问题。为了克服这些局限性，我们提出了一种基于强化学习的黑盒模型反转攻击。将潜在空间搜索问题转化为马尔可夫决策过程(MDP)问题，并用强化学习方法进行求解。我们的方法利用生成的图像的置信度分数来向代理提供奖励。最后，利用在MDP中训练的代理所找到的潜在向量来重建私有数据。在不同的数据集和模型上的实验结果表明，我们的攻击成功地恢复了目标模型的隐私信息，达到了最好的攻击性能。通过提出一种更高级的黑盒模型反转攻击，强调了隐私保护机器学习研究的重要性。



## **23. Defense-Prefix for Preventing Typographic Attacks on CLIP**

防御-用于防止对剪辑进行排版攻击的前缀 cs.CV

Under review

**SubmitDate**: 2023-04-10    [abs](http://arxiv.org/abs/2304.04512v1) [paper-pdf](http://arxiv.org/pdf/2304.04512v1)

**Authors**: Hiroki Azuma, Yusuke Matsui

**Abstract**: Vision-language pre-training models (VLPs) have exhibited revolutionary improvements in various vision-language tasks. In VLP, some adversarial attacks fool a model into false or absurd classifications. Previous studies addressed these attacks by fine-tuning the model or changing its architecture. However, these methods risk losing the original model's performance and are difficult to apply to downstream tasks. In particular, their applicability to other tasks has not been considered. In this study, we addressed the reduction of the impact of typographic attacks on CLIP without changing the model parameters. To achieve this, we expand the idea of ``prefix learning'' and introduce our simple yet effective method: Defense-Prefix (DP), which inserts the DP token before a class name to make words ``robust'' against typographic attacks. Our method can be easily applied to downstream tasks, such as object detection, because the proposed method is independent of the model parameters. Our method significantly improves the accuracy of classification tasks for typographic attack datasets, while maintaining the zero-shot capabilities of the model. In addition, we leverage our proposed method for object detection, demonstrating its high applicability and effectiveness. The codes and datasets will be publicly available.

摘要: 视觉语言预训练模型(VLP)在各种视觉语言任务中表现出革命性的改进。在VLP中，一些对抗性攻击欺骗模型进行错误或荒谬的分类。以前的研究通过微调模型或更改其体系结构来解决这些攻击。然而，这些方法可能会失去原始模型的性能，并且很难应用于下游任务。特别是，它们对其他任务的适用性没有得到考虑。在这项研究中，我们解决了在不改变模型参数的情况下减少排版攻击对CLIP的影响。为了实现这一点，我们扩展了前缀学习的概念，并引入了我们简单但有效的方法：防御前缀(DP)，它将DP标记插入到类名之前，使单词对排版攻击具有健壮性。我们的方法可以很容易地应用于下游任务，如目标检测，因为所提出的方法与模型参数无关。我们的方法显著地提高了排版攻击数据集的分类任务的准确性，同时保持了模型的零命中能力。此外，我们利用我们提出的方法进行目标检测，证明了其高度的适用性和有效性。代码和数据集将公开提供。



## **24. Robust Neural Architecture Search**

稳健的神经结构搜索 cs.LG

**SubmitDate**: 2023-04-10    [abs](http://arxiv.org/abs/2304.02845v2) [paper-pdf](http://arxiv.org/pdf/2304.02845v2)

**Authors**: Xunyu Zhu, Jian Li, Yong Liu, Weiping Wang

**Abstract**: Neural Architectures Search (NAS) becomes more and more popular over these years. However, NAS-generated models tends to suffer greater vulnerability to various malicious attacks. Lots of robust NAS methods leverage adversarial training to enhance the robustness of NAS-generated models, however, they neglected the nature accuracy of NAS-generated models. In our paper, we propose a novel NAS method, Robust Neural Architecture Search (RNAS). To design a regularization term to balance accuracy and robustness, RNAS generates architectures with both high accuracy and good robustness. To reduce search cost, we further propose to use noise examples instead adversarial examples as input to search architectures. Extensive experiments show that RNAS achieves state-of-the-art (SOTA) performance on both image classification and adversarial attacks, which illustrates the proposed RNAS achieves a good tradeoff between robustness and accuracy.

摘要: 近年来，神经结构搜索(NAS)变得越来越流行。然而，NAS生成的模型往往更容易受到各种恶意攻击。许多健壮的NAS方法利用对抗性训练来增强NAS生成的模型的健壮性，然而，它们忽略了NAS生成的模型的自然准确性。在本文中，我们提出了一种新的NAS方法--稳健神经结构搜索(RNAS)。为了设计一个正则化项来平衡精度和健壮性，RNAS生成具有高精度和良好健壮性的体系结构。为了降低搜索成本，我们进一步建议使用噪声示例而不是对抗性示例作为搜索架构的输入。大量的实验表明，RNAS在图像分类和敌意攻击方面都达到了最好的性能，说明了RNAS在稳健性和准确性之间取得了很好的折衷。



## **25. Generating Adversarial Attacks in the Latent Space**

在潜在空间中生成对抗性攻击 cs.LG

**SubmitDate**: 2023-04-10    [abs](http://arxiv.org/abs/2304.04386v1) [paper-pdf](http://arxiv.org/pdf/2304.04386v1)

**Authors**: Nitish Shukla, Sudipta Banerjee

**Abstract**: Adversarial attacks in the input (pixel) space typically incorporate noise margins such as $L_1$ or $L_{\infty}$-norm to produce imperceptibly perturbed data that confound deep learning networks. Such noise margins confine the magnitude of permissible noise. In this work, we propose injecting adversarial perturbations in the latent (feature) space using a generative adversarial network, removing the need for margin-based priors. Experiments on MNIST, CIFAR10, Fashion-MNIST, CIFAR100 and Stanford Dogs datasets support the effectiveness of the proposed method in generating adversarial attacks in the latent space while ensuring a high degree of visual realism with respect to pixel-based adversarial attack methods.

摘要: 输入(像素)空间中的对抗性攻击通常包含诸如$L_1$或$L_{\inty}$-NORMAL之类的噪声裕度，以产生难以察觉的扰动数据，从而混淆深度学习网络。这样的噪声裕度限制了允许的噪声的大小。在这项工作中，我们建议使用生成性对抗性网络在潜在(特征)空间中注入对抗性扰动，消除了对基于差值的先验的需要。在MNIST、CIFAR10、Fashion-MNIST、CIFAR100和Stanford Dogs数据集上的实验表明，该方法在生成潜在空间中的对抗性攻击的同时，相对于基于像素的对抗性攻击方法具有高度的视觉真实感。



## **26. Certifiable Black-Box Attack: Ensuring Provably Successful Attack for Adversarial Examples**

可证明的黑盒攻击：确保对抗性例子的可证明的成功攻击 cs.LG

**SubmitDate**: 2023-04-10    [abs](http://arxiv.org/abs/2304.04343v1) [paper-pdf](http://arxiv.org/pdf/2304.04343v1)

**Authors**: Hanbin Hong, Yuan Hong

**Abstract**: Black-box adversarial attacks have shown strong potential to subvert machine learning models. Existing black-box adversarial attacks craft the adversarial examples by iteratively querying the target model and/or leveraging the transferability of a local surrogate model. Whether such attack can succeed remains unknown to the adversary when empirically designing the attack. In this paper, to our best knowledge, we take the first step to study a new paradigm of adversarial attacks -- certifiable black-box attack that can guarantee the attack success rate of the crafted adversarial examples. Specifically, we revise the randomized smoothing to establish novel theories for ensuring the attack success rate of the adversarial examples. To craft the adversarial examples with the certifiable attack success rate (CASR) guarantee, we design several novel techniques, including a randomized query method to query the target model, an initialization method with smoothed self-supervised perturbation to derive certifiable adversarial examples, and a geometric shifting method to reduce the perturbation size of the certifiable adversarial examples for better imperceptibility. We have comprehensively evaluated the performance of the certifiable black-box attack on CIFAR10 and ImageNet datasets against different levels of defenses. Both theoretical and experimental results have validated the effectiveness of the proposed certifiable attack.

摘要: 黑盒对抗性攻击已经显示出极大的潜力来颠覆机器学习模型。现有的黑盒对抗性攻击通过迭代查询目标模型和/或利用本地代理模型的可转移性来制作对抗性实例。在经验性地设计攻击时，对手仍不知道这种攻击是否能成功。在本文中，据我们所知，我们首先研究了一种新的对抗性攻击范例--可证明黑盒攻击，它可以保证精心制作的对抗性范例的攻击成功率。具体地说，我们对随机平滑进行了修正，以建立新的理论来确保对抗性例子的攻击成功率。为了在保证可证明攻击成功率(CaSR)的前提下生成可证明的对抗实例，我们设计了几种新的技术，包括用于查询目标模型的随机查询方法、用于获得可证明的对抗实例的平滑自监督扰动的初始化方法、以及用于减小可证明的对抗实例的扰动大小以获得更好不可见性的几何平移方法。我们综合评估了针对CIFAR10和ImageNet数据集的可认证黑盒攻击在不同防御级别下的性能。理论和实验结果都验证了该可证明攻击的有效性。



## **27. Unsupervised Multi-Criteria Adversarial Detection in Deep Image Retrieval**

深度图像检索中的无监督多准则攻击检测 cs.CV

**SubmitDate**: 2023-04-09    [abs](http://arxiv.org/abs/2304.04228v1) [paper-pdf](http://arxiv.org/pdf/2304.04228v1)

**Authors**: Yanru Xiao, Cong Wang, Xing Gao

**Abstract**: The vulnerability in the algorithm supply chain of deep learning has imposed new challenges to image retrieval systems in the downstream. Among a variety of techniques, deep hashing is gaining popularity. As it inherits the algorithmic backend from deep learning, a handful of attacks are recently proposed to disrupt normal image retrieval. Unfortunately, the defense strategies in softmax classification are not readily available to be applied in the image retrieval domain. In this paper, we propose an efficient and unsupervised scheme to identify unique adversarial behaviors in the hamming space. In particular, we design three criteria from the perspectives of hamming distance, quantization loss and denoising to defend against both untargeted and targeted attacks, which collectively limit the adversarial space. The extensive experiments on four datasets demonstrate 2-23% improvements of detection rates with minimum computational overhead for real-time image queries.

摘要: 深度学习算法供应链中的漏洞给下游的图像检索系统带来了新的挑战。在各种技术中，深度散列越来越受欢迎。由于它继承了深度学习的算法后端，最近提出了一些攻击来扰乱正常的图像检索。遗憾的是，Softmax分类中的防御策略并不容易应用于图像检索领域。在本文中，我们提出了一种有效的无监督方案来识别Hamming空间中唯一的敌对行为。特别是，我们从汉明距离、量化损失和去噪的角度设计了三个准则来防御非目标攻击和目标攻击，这三个准则共同限制了对抗空间。在四个数据集上的大量实验表明，在最小计算开销的情况下，实时图像查询的检测率提高了2%-23%。



## **28. Adversarially Robust Neural Architecture Search for Graph Neural Networks**

图神经网络的逆鲁棒神经结构搜索 cs.LG

Accepted as a conference paper at CVPR 2023

**SubmitDate**: 2023-04-09    [abs](http://arxiv.org/abs/2304.04168v1) [paper-pdf](http://arxiv.org/pdf/2304.04168v1)

**Authors**: Beini Xie, Heng Chang, Ziwei Zhang, Xin Wang, Daixin Wang, Zhiqiang Zhang, Rex Ying, Wenwu Zhu

**Abstract**: Graph Neural Networks (GNNs) obtain tremendous success in modeling relational data. Still, they are prone to adversarial attacks, which are massive threats to applying GNNs to risk-sensitive domains. Existing defensive methods neither guarantee performance facing new data/tasks or adversarial attacks nor provide insights to understand GNN robustness from an architectural perspective. Neural Architecture Search (NAS) has the potential to solve this problem by automating GNN architecture designs. Nevertheless, current graph NAS approaches lack robust design and are vulnerable to adversarial attacks. To tackle these challenges, we propose a novel Robust Neural Architecture search framework for GNNs (G-RNA). Specifically, we design a robust search space for the message-passing mechanism by adding graph structure mask operations into the search space, which comprises various defensive operation candidates and allows us to search for defensive GNNs. Furthermore, we define a robustness metric to guide the search procedure, which helps to filter robust architectures. In this way, G-RNA helps understand GNN robustness from an architectural perspective and effectively searches for optimal adversarial robust GNNs. Extensive experimental results on benchmark datasets show that G-RNA significantly outperforms manually designed robust GNNs and vanilla graph NAS baselines by 12.1% to 23.4% under adversarial attacks.

摘要: 图神经网络(GNN)在关系数据建模方面取得了巨大的成功。尽管如此，它们仍然容易受到对抗性攻击，这是将GNN应用于风险敏感领域的巨大威胁。现有的防御方法既不能保证在面对新数据/任务或敌意攻击时的性能，也不能提供从体系结构角度理解GNN健壮性的见解。神经体系结构搜索(NAS)通过自动化GNN体系结构设计来解决这一问题。然而，目前的图NAS方法缺乏健壮的设计，并且容易受到对手攻击。为了应对这些挑战，我们提出了一种新的面向GNN的健壮神经结构搜索框架(G-RNA)。具体地说，我们通过在搜索空间中加入图结构掩码操作，为消息传递机制设计了一个健壮的搜索空间，该空间包含了各种防御操作候选者，允许我们搜索防御性GNN。此外，我们定义了一个健壮性度量来指导搜索过程，这有助于过滤健壮的体系结构。通过这种方式，G-RNA有助于从体系结构的角度理解GNN的健壮性，并有效地搜索最优的对抗性健壮GNN。在基准数据集上的大量实验结果表明，G-RNA在对抗攻击下的性能明显优于人工设计的健壮GNN和Vanilla图NAS基线，分别为12.1%和23.4%。



## **29. Exploring the Connection between Robust and Generative Models**

探索健壮性模型和生成性模型之间的联系 cs.LG

technical report, 6 pages, 6 figures

**SubmitDate**: 2023-04-08    [abs](http://arxiv.org/abs/2304.04033v1) [paper-pdf](http://arxiv.org/pdf/2304.04033v1)

**Authors**: Senad Beadini, Iacopo Masi

**Abstract**: We offer a study that connects robust discriminative classifiers trained with adversarial training (AT) with generative modeling in the form of Energy-based Models (EBM). We do so by decomposing the loss of a discriminative classifier and showing that the discriminative model is also aware of the input data density. Though a common assumption is that adversarial points leave the manifold of the input data, our study finds out that, surprisingly, untargeted adversarial points in the input space are very likely under the generative model hidden inside the discriminative classifier -- have low energy in the EBM. We present two evidence: untargeted attacks are even more likely than the natural data and their likelihood increases as the attack strength increases. This allows us to easily detect them and craft a novel attack called High-Energy PGD that fools the classifier yet has energy similar to the data set.

摘要: 我们提供了一项研究，将经过对抗性训练(AT)训练的稳健区分分类器与基于能量的模型(EBM)形式的生成性建模相结合。我们通过分解判别分类器的损失来做到这一点，并表明判别模型也知道输入数据的密度。虽然一个普遍的假设是敌对点离开了输入数据的流形，但我们的研究发现，令人惊讶的是，在隐藏在判别分类器中的生成模型下，输入空间中的非目标对抗性点很可能在EBM中具有低能量。我们提出了两个证据：非目标攻击的可能性甚至比自然数据更高，并且随着攻击强度的增加，它们的可能性也会增加。这使我们能够轻松地检测到它们，并创建一种名为高能PGD的新型攻击，它愚弄了分类器，但具有与数据集相似的能量。



## **30. On anti-stochastic properties of unlabeled graphs**

关于无标号图的反随机性 cs.DM

**SubmitDate**: 2023-04-08    [abs](http://arxiv.org/abs/2112.04395v4) [paper-pdf](http://arxiv.org/pdf/2112.04395v4)

**Authors**: Sergei Kiselev, Andrey Kupavskii, Oleg Verbitsky, Maksim Zhukovskii

**Abstract**: We study vulnerability of a uniformly distributed random graph to an attack by an adversary who aims for a global change of the distribution while being able to make only a local change in the graph. We call a graph property $A$ anti-stochastic if the probability that a random graph $G$ satisfies $A$ is small but, with high probability, there is a small perturbation transforming $G$ into a graph satisfying $A$. While for labeled graphs such properties are easy to obtain from binary covering codes, the existence of anti-stochastic properties for unlabeled graphs is not so evident. If an admissible perturbation is either the addition or the deletion of one edge, we exhibit an anti-stochastic property that is satisfied by a random unlabeled graph of order $n$ with probability $(2+o(1))/n^2$, which is as small as possible. We also express another anti-stochastic property in terms of the degree sequence of a graph. This property has probability $(2+o(1))/(n\ln n)$, which is optimal up to factor of 2.

摘要: 我们研究了一个均匀分布的随机图对敌手攻击的脆弱性，该对手的目标是改变分布的全局，但只能对图进行局部改变。如果一个随机图$G$满足$A$的概率很小，但在很高的概率下，存在一个将$G$转换成满足$A$的图的小扰动，我们称图的性质$A$是反随机的。而对于有标号的图，这样的性质很容易从二元覆盖码中获得，而无标号图的反随机性的存在就不那么明显了。如果一个允许的扰动是增加或删除一条边，我们展示了一个反随机性质，它由一个概率为$(2+o(1))/n^2$的n阶随机无标号图所满足，它尽可能小。我们还用图的度序列来表示另一个反随机性质。该性质的概率为$(2+o(1))/(n\ln)$，最优为2倍。



## **31. RobCaps: Evaluating the Robustness of Capsule Networks against Affine Transformations and Adversarial Attacks**

RobCaps：评估胶囊网络对仿射变换和对手攻击的稳健性 cs.LG

To appear at the 2023 International Joint Conference on Neural  Networks (IJCNN), Queensland, Australia, June 2023

**SubmitDate**: 2023-04-08    [abs](http://arxiv.org/abs/2304.03973v1) [paper-pdf](http://arxiv.org/pdf/2304.03973v1)

**Authors**: Alberto Marchisio, Antonio De Marco, Alessio Colucci, Maurizio Martina, Muhammad Shafique

**Abstract**: Capsule Networks (CapsNets) are able to hierarchically preserve the pose relationships between multiple objects for image classification tasks. Other than achieving high accuracy, another relevant factor in deploying CapsNets in safety-critical applications is the robustness against input transformations and malicious adversarial attacks.   In this paper, we systematically analyze and evaluate different factors affecting the robustness of CapsNets, compared to traditional Convolutional Neural Networks (CNNs). Towards a comprehensive comparison, we test two CapsNet models and two CNN models on the MNIST, GTSRB, and CIFAR10 datasets, as well as on the affine-transformed versions of such datasets. With a thorough analysis, we show which properties of these architectures better contribute to increasing the robustness and their limitations. Overall, CapsNets achieve better robustness against adversarial examples and affine transformations, compared to a traditional CNN with a similar number of parameters. Similar conclusions have been derived for deeper versions of CapsNets and CNNs. Moreover, our results unleash a key finding that the dynamic routing does not contribute much to improving the CapsNets' robustness. Indeed, the main generalization contribution is due to the hierarchical feature learning through capsules.

摘要: 胶囊网络(CapsNets)能够在图像分类任务中分层地保持多个对象之间的姿势关系。在安全关键型应用程序中部署CapsNet的另一个相关因素是对输入转换和恶意对手攻击的稳健性。本文系统地分析和评价了影响CapsNets健壮性的各种因素，并与传统卷积神经网络(CNN)进行了比较。为了进行全面的比较，我们在MNIST、GTSRB和CIFAR10数据集以及这些数据集的仿射变换版本上测试了两个CapsNet模型和两个CNN模型。通过深入的分析，我们展示了这些体系结构的哪些属性更有助于提高健壮性及其局限性。总体而言，与具有类似参数的传统CNN相比，CapsNets在对抗对手示例和仿射变换方面实现了更好的健壮性。对于CapsNet和CNN的更深版本，也得出了类似的结论。此外，我们的结果揭示了一个关键发现，即动态路由对提高CapsNet的健壮性没有太大帮助。事实上，主要的泛化贡献是由于通过胶囊进行的分层特征学习。



## **32. TSFool: Crafting Highly-imperceptible Adversarial Time Series through Multi-objective Black-box Attack to Fool RNN Classifiers**

TSFool：通过多目标黑盒攻击对愚弄RNN分类器制作高度不可察觉的对抗性时间序列 cs.LG

9 pages, 7 figures

**SubmitDate**: 2023-04-08    [abs](http://arxiv.org/abs/2209.06388v2) [paper-pdf](http://arxiv.org/pdf/2209.06388v2)

**Authors**: Yanyun Wang, Dehui Du, Yuanhao Liu

**Abstract**: Neural network (NN) classifiers are vulnerable to adversarial attacks. Although the existing gradient-based attacks achieve state-of-the-art performance in feed-forward NNs and image recognition tasks, they do not perform as well on time series classification with recurrent neural network (RNN) models. This is because the cyclical structure of RNN prevents direct model differentiation and the visual sensitivity of time series data to perturbations challenges the traditional local optimization objective of the adversarial attack. In this paper, a black-box method called TSFool is proposed to efficiently craft highly-imperceptible adversarial time series for RNN classifiers. We propose a novel global optimization objective named Camouflage Coefficient to consider the imperceptibility of adversarial samples from the perspective of class distribution, and accordingly refine the adversarial attack as a multi-objective optimization problem to enhance the perturbation quality. To get rid of the dependence on gradient information, we also propose a new idea that introduces a representation model for RNN to capture deeply embedded vulnerable samples having otherness between their features and latent manifold, based on which the optimization solution can be heuristically approximated. Experiments on 10 UCR datasets are conducted to confirm that TSFool averagely outperforms existing methods with a 46.3% higher attack success rate, 87.4% smaller perturbation and 25.6% better Camouflage Coefficient at a similar time cost.

摘要: 神经网络(NN)分类器容易受到敌意攻击。尽管现有的基于梯度的攻击在前馈神经网络和图像识别任务中取得了最好的性能，但它们在使用递归神经网络(RNN)模型进行时间序列分类时表现不佳。这是因为RNN的周期性结构阻止了直接的模型区分，并且时间序列数据对扰动的视觉敏感性挑战了传统的对抗性攻击的局部优化目标。本文提出了一种称为TSFool的黑盒方法来有效地为RNN分类器构造高度不可察觉的对抗性时间序列。提出了一种新的全局优化目标伪装系数，从类别分布的角度考虑敌方样本的隐蔽性，从而将敌方攻击细化为多目标优化问题，以提高扰动质量。为了摆脱对梯度信息的依赖，我们还提出了一种新的思想，引入了一种RNN的表示模型来捕捉深度嵌入的特征与潜在流形不同的易受攻击样本，基于该模型可以启发式地逼近最优解。在10个UCR数据集上的实验结果表明，在相同的时间代价下，TSFool的攻击成功率提高了46.3%，扰动减少了87.4%，伪装系数提高了25.6%。



## **33. Benchmarking the Robustness of Quantized Models**

量化模型稳健性的基准测试 cs.LG

Workshop at IEEE Conference on Computer Vision and Pattern  Recognition 2023

**SubmitDate**: 2023-04-08    [abs](http://arxiv.org/abs/2304.03968v1) [paper-pdf](http://arxiv.org/pdf/2304.03968v1)

**Authors**: Yisong Xiao, Tianyuan Zhang, Shunchang Liu, Haotong Qin

**Abstract**: Quantization has emerged as an essential technique for deploying deep neural networks (DNNs) on devices with limited resources. However, quantized models exhibit vulnerabilities when exposed to various noises in real-world applications. Despite the importance of evaluating the impact of quantization on robustness, existing research on this topic is limited and often disregards established principles of robustness evaluation, resulting in incomplete and inconclusive findings. To address this gap, we thoroughly evaluated the robustness of quantized models against various noises (adversarial attacks, natural corruptions, and systematic noises) on ImageNet. Extensive experiments demonstrate that lower-bit quantization is more resilient to adversarial attacks but is more susceptible to natural corruptions and systematic noises. Notably, our investigation reveals that impulse noise (in natural corruptions) and the nearest neighbor interpolation (in systematic noises) have the most significant impact on quantized models. Our research contributes to advancing the robust quantization of models and their deployment in real-world scenarios.

摘要: 量化已经成为在资源有限的设备上部署深度神经网络(DNN)的一项基本技术。然而，在实际应用中，当量化模型暴露在各种噪声中时，会显示出脆弱性。尽管评估量化对稳健性的影响很重要，但现有的关于这一主题的研究是有限的，而且往往忽视了稳健性评估的既定原则，导致结果不完整和不确定。为了弥补这一差距，我们彻底评估了量化模型对ImageNet上的各种噪声(敌意攻击、自然破坏和系统噪声)的稳健性。广泛的实验表明，低位量化对敌意攻击更具弹性，但更容易受到自然破坏和系统噪声的影响。值得注意的是，我们的调查显示脉冲噪声(在自然破坏中)和最近邻内插(在系统噪声中)对量化模型有最显著的影响。我们的研究有助于推进模型的稳健量化及其在现实世界场景中的部署。



## **34. Robust Deep Learning Models Against Semantic-Preserving Adversarial Attack**

抵抗语义保持的敌意攻击的稳健深度学习模型 cs.LG

Paper accepted by the 2023 International Joint Conference on Neural  Networks (IJCNN 2023)

**SubmitDate**: 2023-04-08    [abs](http://arxiv.org/abs/2304.03955v1) [paper-pdf](http://arxiv.org/pdf/2304.03955v1)

**Authors**: Dashan Gao, Yunce Zhao, Yinghua Yao, Zeqi Zhang, Bifei Mao, Xin Yao

**Abstract**: Deep learning models can be fooled by small $l_p$-norm adversarial perturbations and natural perturbations in terms of attributes. Although the robustness against each perturbation has been explored, it remains a challenge to address the robustness against joint perturbations effectively. In this paper, we study the robustness of deep learning models against joint perturbations by proposing a novel attack mechanism named Semantic-Preserving Adversarial (SPA) attack, which can then be used to enhance adversarial training. Specifically, we introduce an attribute manipulator to generate natural and human-comprehensible perturbations and a noise generator to generate diverse adversarial noises. Based on such combined noises, we optimize both the attribute value and the diversity variable to generate jointly-perturbed samples. For robust training, we adversarially train the deep learning model against the generated joint perturbations. Empirical results on four benchmarks show that the SPA attack causes a larger performance decline with small $l_{\infty}$ norm-ball constraints compared to existing approaches. Furthermore, our SPA-enhanced training outperforms existing defense methods against such joint perturbations.

摘要: 深度学习模型可以被小的$l_p$-范数对抗性扰动和自然属性扰动所愚弄。虽然已经研究了对每个扰动的稳健性，但有效地解决针对联合扰动的稳健性仍然是一个挑战。本文研究了深度学习模型对联合扰动的健壮性，提出了一种新的攻击机制--语义保持对抗性攻击(SPA)，并将其用于增强对抗性训练。具体地说，我们引入了一个属性操纵器来产生自然和人类可以理解的扰动，以及一个噪声生成器来产生不同的对抗性噪声。基于这种组合噪声，我们对属性值和多样性变量进行了优化，以产生联合扰动样本。对于稳健的训练，我们相反地训练深度学习模型来对抗产生的联合扰动。在四个基准测试上的实验结果表明，与现有方法相比，SPA攻击在较小的$l_(Inty)$范数球约束下导致了更大的性能下降。此外，我们的SPA增强训练在对抗此类联合扰动方面优于现有的防御方法。



## **35. Discrete Point-wise Attack Is Not Enough: Generalized Manifold Adversarial Attack for Face Recognition**

离散逐点攻击是不够的：面向人脸识别的广义流形对抗攻击 cs.CV

Accepted by CVPR2023

**SubmitDate**: 2023-04-08    [abs](http://arxiv.org/abs/2301.06083v2) [paper-pdf](http://arxiv.org/pdf/2301.06083v2)

**Authors**: Qian Li, Yuxiao Hu, Ye Liu, Dongxiao Zhang, Xin Jin, Yuntian Chen

**Abstract**: Classical adversarial attacks for Face Recognition (FR) models typically generate discrete examples for target identity with a single state image. However, such paradigm of point-wise attack exhibits poor generalization against numerous unknown states of identity and can be easily defended. In this paper, by rethinking the inherent relationship between the face of target identity and its variants, we introduce a new pipeline of Generalized Manifold Adversarial Attack (GMAA) to achieve a better attack performance by expanding the attack range. Specifically, this expansion lies on two aspects - GMAA not only expands the target to be attacked from one to many to encourage a good generalization ability for the generated adversarial examples, but it also expands the latter from discrete points to manifold by leveraging the domain knowledge that face expression change can be continuous, which enhances the attack effect as a data augmentation mechanism did. Moreover, we further design a dual supervision with local and global constraints as a minor contribution to improve the visual quality of the generated adversarial examples. We demonstrate the effectiveness of our method based on extensive experiments, and reveal that GMAA promises a semantic continuous adversarial space with a higher generalization ability and visual quality

摘要: 针对人脸识别(FR)模型的经典对抗性攻击通常生成具有单一状态图像的目标身份的离散示例。然而，这种点式攻击对许多未知的身份状态表现出很差的泛化，并且可以很容易地进行防御。本文通过重新考虑目标身份人脸及其变体之间的内在关系，提出了一种新的广义流形对抗攻击(GMAA)流水线，通过扩大攻击范围来达到更好的攻击性能。具体地说，这种扩展体现在两个方面--GMAA不仅将攻击目标从一个扩展到多个，鼓励对生成的对抗性实例具有良好的泛化能力，而且利用人脸表情变化可以连续的领域知识，将后者从离散的点扩展到流形，从而增强了攻击效果，就像一种数据增强机制一样。此外，我们进一步设计了一种局部约束和全局约束的双重监督机制，以提高生成的对抗性实例的视觉质量。我们在大量实验的基础上证明了我们的方法的有效性，并表明GMAA保证了一个具有更高泛化能力和视觉质量的语义连续对抗空间



## **36. SoK: Decentralized Finance (DeFi) Attacks**

SOK：去中心化金融(Defi)攻击 cs.CR

**SubmitDate**: 2023-04-07    [abs](http://arxiv.org/abs/2208.13035v3) [paper-pdf](http://arxiv.org/pdf/2208.13035v3)

**Authors**: Liyi Zhou, Xihan Xiong, Jens Ernstberger, Stefanos Chaliasos, Zhipeng Wang, Ye Wang, Kaihua Qin, Roger Wattenhofer, Dawn Song, Arthur Gervais

**Abstract**: Within just four years, the blockchain-based Decentralized Finance (DeFi) ecosystem has accumulated a peak total value locked (TVL) of more than 253 billion USD. This surge in DeFi's popularity has, unfortunately, been accompanied by many impactful incidents. According to our data, users, liquidity providers, speculators, and protocol operators suffered a total loss of at least 3.24 billion USD from Apr 30, 2018 to Apr 30, 2022. Given the blockchain's transparency and increasing incident frequency, two questions arise: How can we systematically measure, evaluate, and compare DeFi incidents? How can we learn from past attacks to strengthen DeFi security?   In this paper, we introduce a common reference frame to systematically evaluate and compare DeFi incidents, including both attacks and accidents. We investigate 77 academic papers, 30 audit reports, and 181 real-world incidents. Our data reveals several gaps between academia and the practitioners' community. For example, few academic papers address "price oracle attacks" and "permissonless interactions", while our data suggests that they are the two most frequent incident types (15% and 10.5% correspondingly). We also investigate potential defenses, and find that: (i) 103 (56%) of the attacks are not executed atomically, granting a rescue time frame for defenders; (ii) SoTA bytecode similarity analysis can at least detect 31 vulnerable/23 adversarial contracts; and (iii) 33 (15.3%) of the adversaries leak potentially identifiable information by interacting with centralized exchanges.

摘要: 短短四年时间，基于区块链的去中心化金融(DEFI)生态系统已经积累了超过2530亿美元的峰值总价值锁定(TVL)。不幸的是，Defi人气的飙升伴随着许多有影响力的事件。根据我们的数据，从2018年4月30日到2022年4月30日，用户、流动性提供商、投机者和协议运营商总共遭受了至少32.4亿美元的损失。鉴于区块链的透明度和不断增加的事件频率，出现了两个问题：我们如何系统地衡量、评估和比较Defi事件？我们如何从过去的袭击中吸取教训，以加强Defi安全？在这篇文章中，我们引入了一个通用的参照系来系统地评估和比较DEFI事件，包括攻击和事故。我们调查了77篇学术论文，30份审计报告和181起真实世界的事件。我们的数据揭示了学术界和从业者社区之间的几个差距。举例来说，很少有学术论文涉及“价格先知攻击”和“不允许的相互作用”，而我们的数据显示，它们是最常见的两种事件类型(分别为15%和10.5%)。我们还调查了潜在的防御措施，发现：(I)103(56%)的攻击不是自动执行的，这为防御者提供了救援时间框架；(Ii)Sota字节码相似性分析至少可以检测到31个VULNERABLE/23个对手合同；以及(Iii)33个(15.3%)的对手通过与中央交易所的交互泄露了潜在的可识别信息。



## **37. AMS-DRL: Learning Multi-Pursuit Evasion for Safe Targeted Navigation of Drones**

AMS-DRL：用于无人机安全定向导航的学习多目标规避 cs.RO

**SubmitDate**: 2023-04-07    [abs](http://arxiv.org/abs/2304.03443v1) [paper-pdf](http://arxiv.org/pdf/2304.03443v1)

**Authors**: Jiaping Xiao, Mir Feroskhan

**Abstract**: Safe navigation of drones in the presence of adversarial physical attacks from multiple pursuers is a challenging task. This paper proposes a novel approach, asynchronous multi-stage deep reinforcement learning (AMS-DRL), to train an adversarial neural network that can learn from the actions of multiple pursuers and adapt quickly to their behavior, enabling the drone to avoid attacks and reach its target. Our approach guarantees convergence by ensuring Nash Equilibrium among agents from the game-theory analysis. We evaluate our method in extensive simulations and show that it outperforms baselines with higher navigation success rates. We also analyze how parameters such as the relative maximum speed affect navigation performance. Furthermore, we have conducted physical experiments and validated the effectiveness of the trained policies in real-time flights. A success rate heatmap is introduced to elucidate how spatial geometry influences navigation outcomes. Project website: https://github.com/NTU-UAVG/AMS-DRL-for-Pursuit-Evasion.

摘要: 在多个追踪者的敌对物理攻击下，无人机的安全导航是一项具有挑战性的任务。提出了一种新的方法--异步多阶段深度强化学习(AMS-DRL)，用于训练对抗神经网络，该网络能够从多个追赶者的行为中学习并快速适应他们的行为，使无人机能够躲避攻击并到达目标。我们的方法通过从博弈论分析确保代理之间的纳什均衡来确保收敛。我们在大量的仿真中对我们的方法进行了评估，结果表明，它的导航成功率比基线更高。我们还分析了相对最大速度等参数对导航性能的影响。此外，我们进行了物理实验，并在实时飞行中验证了训练后的策略的有效性。介绍了一个成功率热图，以阐明空间几何形状如何影响导航结果。项目网站：https://github.com/NTU-UAVG/AMS-DRL-for-Pursuit-Evasion.



## **38. LP-BFGS attack: An adversarial attack based on the Hessian with limited pixels**

LP-BFGS攻击：一种基于有限像素黑森的对抗性攻击 cs.CR

15 pages, 7 figures

**SubmitDate**: 2023-04-07    [abs](http://arxiv.org/abs/2210.15446v2) [paper-pdf](http://arxiv.org/pdf/2210.15446v2)

**Authors**: Jiebao Zhang, Wenhua Qian, Rencan Nie, Jinde Cao, Dan Xu

**Abstract**: Deep neural networks are vulnerable to adversarial attacks. Most $L_{0}$-norm based white-box attacks craft perturbations by the gradient of models to the input. Since the computation cost and memory limitation of calculating the Hessian matrix, the application of Hessian or approximate Hessian in white-box attacks is gradually shelved. In this work, we note that the sparsity requirement on perturbations naturally lends itself to the usage of Hessian information. We study the attack performance and computation cost of the attack method based on the Hessian with a limited number of perturbation pixels. Specifically, we propose the Limited Pixel BFGS (LP-BFGS) attack method by incorporating the perturbation pixel selection strategy and the BFGS algorithm. Pixels with top-k attribution scores calculated by the Integrated Gradient method are regarded as optimization variables of the LP-BFGS attack. Experimental results across different networks and datasets demonstrate that our approach has comparable attack ability with reasonable computation in different numbers of perturbation pixels compared with existing solutions.

摘要: 深度神经网络很容易受到敌意攻击。大多数基于$L_{0}$范数的白盒通过模型到输入的梯度来攻击手工扰动。由于计算海森矩阵的计算量和内存的限制，海森矩阵或近似海森矩阵在白盒攻击中的应用逐渐被搁置。在这项工作中，我们注意到对扰动的稀疏性要求自然地适合于使用Hessian信息。研究了基于有限扰动像素的Hessian攻击方法的攻击性能和计算代价。将扰动像素选择策略与有限像素BFGS算法相结合，提出了有限像素BFGS(LP-BFGS)攻击方法。将积分梯度法计算出的top-k属性得分的像素作为LP-BFGS攻击的优化变量。在不同网络和数据集上的实验结果表明，该方法在不同扰动像素数下具有与已有方案相当的攻击能力，并具有合理的计算能力。



## **39. EZClone: Improving DNN Model Extraction Attack via Shape Distillation from GPU Execution Profiles**

EZClone：基于形状提取的改进DNN模型提取攻击 cs.LG

11 pages, 6 tables, 4 figures

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2304.03388v1) [paper-pdf](http://arxiv.org/pdf/2304.03388v1)

**Authors**: Jonah O'Brien Weiss, Tiago Alves, Sandip Kundu

**Abstract**: Deep Neural Networks (DNNs) have become ubiquitous due to their performance on prediction and classification problems. However, they face a variety of threats as their usage spreads. Model extraction attacks, which steal DNNs, endanger intellectual property, data privacy, and security. Previous research has shown that system-level side-channels can be used to leak the architecture of a victim DNN, exacerbating these risks. We propose two DNN architecture extraction techniques catering to various threat models. The first technique uses a malicious, dynamically linked version of PyTorch to expose a victim DNN architecture through the PyTorch profiler. The second, called EZClone, exploits aggregate (rather than time-series) GPU profiles as a side-channel to predict DNN architecture, employing a simple approach and assuming little adversary capability as compared to previous work. We investigate the effectiveness of EZClone when minimizing the complexity of the attack, when applied to pruned models, and when applied across GPUs. We find that EZClone correctly predicts DNN architectures for the entire set of PyTorch vision architectures with 100% accuracy. No other work has shown this degree of architecture prediction accuracy with the same adversarial constraints or using aggregate side-channel information. Prior work has shown that, once a DNN has been successfully cloned, further attacks such as model evasion or model inversion can be accelerated significantly.

摘要: 深度神经网络(DNN)因其在预测和分类问题上的性能而变得无处不在。然而，随着它们的使用普及，它们面临着各种威胁。模型提取攻击窃取DNN，危害知识产权、数据隐私和安全。先前的研究表明，系统级旁路可用于泄漏受害者DNN的体系结构，从而加剧这些风险。针对不同的威胁模型，我们提出了两种DNN结构提取技术。第一种技术使用恶意的动态链接版本的PyTorch，通过PyTorch分析器暴露受攻击的DNN体系结构。第二种称为EZClone，它利用聚合的(而不是时间序列的)GPU配置文件作为侧通道来预测DNN体系结构，采用了一种简单的方法，与以前的工作相比，假设的对手能力很小。我们研究了EZClone在将攻击的复杂性降至最低时的有效性，当应用于修剪的模型时，以及当应用于GPU时。我们发现，EZClone能够100%准确地预测整套PyTorch VISION体系结构的DNN体系结构。没有其他工作表明，在相同的对抗性约束下或使用聚合的旁路信息，体系结构预测的准确性达到了这种程度。以前的工作表明，一旦DNN被成功克隆，进一步的攻击，如模型逃避或模型反转，可以显著加速。



## **40. Reliable Learning for Test-time Attacks and Distribution Shift**

针对测试时间攻击和分布转移的可靠学习 cs.LG

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2304.03370v1) [paper-pdf](http://arxiv.org/pdf/2304.03370v1)

**Authors**: Maria-Florina Balcan, Steve Hanneke, Rattana Pukdee, Dravyansh Sharma

**Abstract**: Machine learning algorithms are often used in environments which are not captured accurately even by the most carefully obtained training data, either due to the possibility of `adversarial' test-time attacks, or on account of `natural' distribution shift. For test-time attacks, we introduce and analyze a novel robust reliability guarantee, which requires a learner to output predictions along with a reliability radius $\eta$, with the meaning that its prediction is guaranteed to be correct as long as the adversary has not perturbed the test point farther than a distance $\eta$. We provide learners that are optimal in the sense that they always output the best possible reliability radius on any test point, and we characterize the reliable region, i.e. the set of points where a given reliability radius is attainable. We additionally analyze reliable learners under distribution shift, where the test points may come from an arbitrary distribution Q different from the training distribution P. For both cases, we bound the probability mass of the reliable region for several interesting examples, for linear separators under nearly log-concave and s-concave distributions, as well as for smooth boundary classifiers under smooth probability distributions.

摘要: 机器学习算法通常用于即使是通过最仔细地获得的训练数据也不能准确捕获的环境中，这要么是因为可能发生对抗性的测试时间攻击，要么是由于“自然的”分布偏移。对于测试时间攻击，我们引入并分析了一种新的稳健可靠性保证，它要求学习者输出预测和可靠性半径，即只要对手没有干扰测试点超过一段距离，它的预测就保证是正确的。我们提供的学习器在某种意义上是最优的，即他们总是在任何测试点上输出最佳可能的可靠性半径，并且我们刻画了可靠区域，即在给定可靠性半径可达到的点集。此外，我们还分析了分布漂移下的可靠学习器，其中测试点可能来自与训练分布P不同的任意分布Q。对于这两种情况，我们对几个有趣的例子、近对数凹分布和s凹分布下的线性分离器以及光滑概率分布下的光滑边界分类器，给出了可靠区域的概率质量。



## **41. Improving Visual Question Answering Models through Robustness Analysis and In-Context Learning with a Chain of Basic Questions**

通过稳健性分析和带基本问题链的情境学习改进视觉问答模型 cs.CV

28 pages

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2304.03147v1) [paper-pdf](http://arxiv.org/pdf/2304.03147v1)

**Authors**: Jia-Hong Huang, Modar Alfadly, Bernard Ghanem, Marcel Worring

**Abstract**: Deep neural networks have been critical in the task of Visual Question Answering (VQA), with research traditionally focused on improving model accuracy. Recently, however, there has been a trend towards evaluating the robustness of these models against adversarial attacks. This involves assessing the accuracy of VQA models under increasing levels of noise in the input, which can target either the image or the proposed query question, dubbed the main question. However, there is currently a lack of proper analysis of this aspect of VQA. This work proposes a new method that utilizes semantically related questions, referred to as basic questions, acting as noise to evaluate the robustness of VQA models. It is hypothesized that as the similarity of a basic question to the main question decreases, the level of noise increases. To generate a reasonable noise level for a given main question, a pool of basic questions is ranked based on their similarity to the main question, and this ranking problem is cast as a LASSO optimization problem. Additionally, this work proposes a novel robustness measure, R_score, and two basic question datasets to standardize the analysis of VQA model robustness. The experimental results demonstrate that the proposed evaluation method effectively analyzes the robustness of VQA models. Moreover, the experiments show that in-context learning with a chain of basic questions can enhance model accuracy.

摘要: 深度神经网络在视觉问答(VQA)任务中一直是至关重要的，传统上的研究集中在提高模型精度上。然而，最近有一种趋势是评估这些模型对对手攻击的稳健性。这涉及在输入噪声水平不断增加的情况下评估VQA模型的准确性，这可能针对图像或拟议的查询问题，称为主要问题。然而，目前还缺乏对VQA这一方面的适当分析。本文提出了一种新的方法，利用语义相关的问题，即基本问题，作为噪声来评估VQA模型的稳健性。假设基本问题与主要问题的相似度越低，噪音水平就越高。为了为给定的主问题生成合理的噪声水平，根据基本问题池与主问题的相似度对其进行排序，并将该排序问题转换为套索优化问题。此外，本文还提出了一种新的健壮性度量R_Score和两个基本问题数据集来规范VQA模型的健壮性分析。实验结果表明，该评价方法有效地分析了VQA模型的稳健性。此外，实验表明，带有一系列基本问题的情境学习可以提高模型的准确性。



## **42. Public Key Encryption with Secure Key Leasing**

使用安全密钥租赁的公钥加密 quant-ph

68 pages, 4 figures. added related works and a comparison with a  concurrent work (2023-04-07)

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2302.11663v2) [paper-pdf](http://arxiv.org/pdf/2302.11663v2)

**Authors**: Shweta Agrawal, Fuyuki Kitagawa, Ryo Nishimaki, Shota Yamada, Takashi Yamakawa

**Abstract**: We introduce the notion of public key encryption with secure key leasing (PKE-SKL). Our notion supports the leasing of decryption keys so that a leased key achieves the decryption functionality but comes with the guarantee that if the quantum decryption key returned by a user passes a validity test, then the user has lost the ability to decrypt. Our notion is similar in spirit to the notion of secure software leasing (SSL) introduced by Ananth and La Placa (Eurocrypt 2021) but captures significantly more general adversarial strategies. In more detail, our adversary is not restricted to use an honest evaluation algorithm to run pirated software. Our results can be summarized as follows:   1. Definitions: We introduce the definition of PKE with secure key leasing and formalize security notions.   2. Constructing PKE with Secure Key Leasing: We provide a construction of PKE-SKL by leveraging a PKE scheme that satisfies a new security notion that we call consistent or inconsistent security against key leasing attacks (CoIC-KLA security). We then construct a CoIC-KLA secure PKE scheme using 1-key Ciphertext-Policy Functional Encryption (CPFE) that in turn can be based on any IND-CPA secure PKE scheme.   3. Identity Based Encryption, Attribute Based Encryption and Functional Encryption with Secure Key Leasing: We provide definitions of secure key leasing in the context of advanced encryption schemes such as identity based encryption (IBE), attribute-based encryption (ABE) and functional encryption (FE). Then we provide constructions by combining the above PKE-SKL with standard IBE, ABE and FE schemes.

摘要: 我们引入了公钥加密和安全密钥租赁(PKE-SKL)的概念。我们的想法支持租赁解密密钥，以便租赁的密钥实现解密功能，但同时也保证，如果用户返回的量子解密密钥通过有效性测试，则用户已失去解密能力。我们的概念在精神上类似于Ananth和La Placa(Eurocrypt 2021)提出的安全软件租赁(SSL)概念，但捕获了更一般的对抗策略。更详细地说，我们的对手并不局限于使用诚实的评估算法来运行盗版软件。定义：引入了具有安全密钥租赁的PKE的定义，并对安全概念进行了形式化描述。2.使用安全密钥租赁构建PKE：利用一种新的PKE方案来构造PKE-SKL，该方案满足一种新的安全概念，即针对密钥租赁攻击的一致或不一致安全(COIC-KLA安全)。然后，我们使用1密钥密文策略函数加密(CPFE)构造了COIC-KLA安全PKE方案，而CPFE又可以基于任何IND-CPA安全PKE方案。3.基于身份的加密、基于属性的加密和基于安全密钥租赁的功能加密：我们在基于身份的加密(IBE)、基于属性的加密(ABE)和功能加密(FE)等高级加密方案的背景下定义了安全密钥租赁。然后，我们将上述PKE-SKL与标准的IBE、ABE和FE格式相结合，给出了构造方法。



## **43. StratDef: Strategic Defense Against Adversarial Attacks in ML-based Malware Detection**

StratDef：基于ML的恶意软件检测中对抗攻击的战略防御 cs.LG

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2202.07568v5) [paper-pdf](http://arxiv.org/pdf/2202.07568v5)

**Authors**: Aqib Rashid, Jose Such

**Abstract**: Over the years, most research towards defenses against adversarial attacks on machine learning models has been in the image recognition domain. The malware detection domain has received less attention despite its importance. Moreover, most work exploring these defenses has focused on several methods but with no strategy when applying them. In this paper, we introduce StratDef, which is a strategic defense system based on a moving target defense approach. We overcome challenges related to the systematic construction, selection, and strategic use of models to maximize adversarial robustness. StratDef dynamically and strategically chooses the best models to increase the uncertainty for the attacker while minimizing critical aspects in the adversarial ML domain, like attack transferability. We provide the first comprehensive evaluation of defenses against adversarial attacks on machine learning for malware detection, where our threat model explores different levels of threat, attacker knowledge, capabilities, and attack intensities. We show that StratDef performs better than other defenses even when facing the peak adversarial threat. We also show that, of the existing defenses, only a few adversarially-trained models provide substantially better protection than just using vanilla models but are still outperformed by StratDef.

摘要: 多年来，针对机器学习模型的敌意攻击防御的研究大多集中在图像识别领域。恶意软件检测领域尽管很重要，但受到的关注较少。此外，大多数探索这些防御措施的工作都集中在几种方法上，但在应用这些方法时没有策略。本文介绍了一种基于移动目标防御方法的战略防御系统StratDef。我们克服了与模型的系统构建、选择和战略使用相关的挑战，以最大限度地提高对手的稳健性。StratDef动态和战略性地选择最佳模型，以增加攻击者的不确定性，同时最小化敌对ML领域的关键方面，如攻击可转移性。我们提供了针对恶意软件检测的机器学习的首次全面防御评估，其中我们的威胁模型探索了不同级别的威胁、攻击者知识、能力和攻击强度。我们表明，即使在面临最大的对手威胁时，StratDef也比其他防御系统表现得更好。我们还表明，在现有的防御系统中，只有少数经过对抗性训练的模型提供了比仅仅使用普通模型更好的保护，但仍然优于StratDef。



## **44. PAD: Towards Principled Adversarial Malware Detection Against Evasion Attacks**

PAD：针对逃避攻击的原则性恶意软件检测 cs.CR

Accepted by IEEE Transactions on Dependable and Secure Computing; To  appear

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2302.11328v2) [paper-pdf](http://arxiv.org/pdf/2302.11328v2)

**Authors**: Deqiang Li, Shicheng Cui, Yun Li, Jia Xu, Fu Xiao, Shouhuai Xu

**Abstract**: Machine Learning (ML) techniques can facilitate the automation of malicious software (malware for short) detection, but suffer from evasion attacks. Many studies counter such attacks in heuristic manners, lacking theoretical guarantees and defense effectiveness. In this paper, we propose a new adversarial training framework, termed Principled Adversarial Malware Detection (PAD), which offers convergence guarantees for robust optimization methods. PAD lays on a learnable convex measurement that quantifies distribution-wise discrete perturbations to protect malware detectors from adversaries, whereby for smooth detectors, adversarial training can be performed with theoretical treatments. To promote defense effectiveness, we propose a new mixture of attacks to instantiate PAD to enhance deep neural network-based measurements and malware detectors. Experimental results on two Android malware datasets demonstrate: (i) the proposed method significantly outperforms the state-of-the-art defenses; (ii) it can harden ML-based malware detection against 27 evasion attacks with detection accuracies greater than 83.45%, at the price of suffering an accuracy decrease smaller than 2.16% in the absence of attacks; (iii) it matches or outperforms many anti-malware scanners in VirusTotal against realistic adversarial malware.

摘要: 机器学习(ML)技术可以促进恶意软件(简称恶意软件)检测的自动化，但受到逃避攻击。许多研究以启发式的方式对抗这种攻击，缺乏理论保障和防御有效性。本文提出了一种新的对抗性训练框架，称为原则性对抗性恶意软件检测(PAD)，它为稳健优化方法提供了收敛保证。PAD建立在可学习的凸度量上，该度量量化了分布方向的离散扰动，以保护恶意软件检测器免受对手的攻击，从而对于平滑的检测器，可以通过理论处理来执行对抗性训练。为了提高防御效果，我们提出了一种新的混合攻击来实例化PAD，以增强基于深度神经网络的测量和恶意软件检测。在两个Android恶意软件数据集上的实验结果表明：(I)该方法的性能明显优于最新的防御方法；(Ii)它可以强化基于ML的恶意软件检测，对27次逃避攻击的检测准确率超过83.45%，而在没有攻击的情况下，检测准确率下降小于2.16%；(Iii)它与VirusTotal中的许多反恶意软件扫描器相匹配或优于对现实恶意软件的检测。



## **45. Robust Upper Bounds for Adversarial Training**

对抗性训练的稳健上界 cs.LG

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2112.09279v2) [paper-pdf](http://arxiv.org/pdf/2112.09279v2)

**Authors**: Dimitris Bertsimas, Xavier Boix, Kimberly Villalobos Carballo, Dick den Hertog

**Abstract**: Many state-of-the-art adversarial training methods for deep learning leverage upper bounds of the adversarial loss to provide security guarantees against adversarial attacks. Yet, these methods rely on convex relaxations to propagate lower and upper bounds for intermediate layers, which affect the tightness of the bound at the output layer. We introduce a new approach to adversarial training by minimizing an upper bound of the adversarial loss that is based on a holistic expansion of the network instead of separate bounds for each layer. This bound is facilitated by state-of-the-art tools from Robust Optimization; it has closed-form and can be effectively trained using backpropagation. We derive two new methods with the proposed approach. The first method (Approximated Robust Upper Bound or aRUB) uses the first order approximation of the network as well as basic tools from Linear Robust Optimization to obtain an empirical upper bound of the adversarial loss that can be easily implemented. The second method (Robust Upper Bound or RUB), computes a provable upper bound of the adversarial loss. Across a variety of tabular and vision data sets we demonstrate the effectiveness of our approach -- RUB is substantially more robust than state-of-the-art methods for larger perturbations, while aRUB matches the performance of state-of-the-art methods for small perturbations.

摘要: 许多先进的深度学习对抗性训练方法利用对抗性损失的上限来提供针对对抗性攻击的安全保证。然而，这些方法依赖于凸松弛来传播中间层的下界和上界，这影响了输出层上界的紧密性。我们引入了一种新的对抗性训练方法，通过最小化对抗性损失的上界，该上界基于网络的整体扩展，而不是针对每一层单独的界。这一界限是由稳健优化的最先进工具促成的；它具有封闭的形式，可以使用反向传播进行有效的训练。利用提出的方法，我们得到了两种新的方法。第一种方法(近似稳健上界或ARUB)利用网络的一阶近似以及线性稳健优化的基本工具来获得易于实现的对手损失的经验上界。第二种方法(稳健上界或RUB)，计算对手损失的可证明上界。在各种表格和视觉数据集上，我们证明了我们方法的有效性--对于较大的扰动，RUB比最先进的方法更健壮，而对于较小的扰动，ABRUB的性能与最先进的方法相当。



## **46. Improving Fast Adversarial Training with Prior-Guided Knowledge**

利用先验指导知识改进快速对抗训练 cs.LG

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2304.00202v2) [paper-pdf](http://arxiv.org/pdf/2304.00202v2)

**Authors**: Xiaojun Jia, Yong Zhang, Xingxing Wei, Baoyuan Wu, Ke Ma, Jue Wang, Xiaochun Cao

**Abstract**: Fast adversarial training (FAT) is an efficient method to improve robustness. However, the original FAT suffers from catastrophic overfitting, which dramatically and suddenly reduces robustness after a few training epochs. Although various FAT variants have been proposed to prevent overfitting, they require high training costs. In this paper, we investigate the relationship between adversarial example quality and catastrophic overfitting by comparing the training processes of standard adversarial training and FAT. We find that catastrophic overfitting occurs when the attack success rate of adversarial examples becomes worse. Based on this observation, we propose a positive prior-guided adversarial initialization to prevent overfitting by improving adversarial example quality without extra training costs. This initialization is generated by using high-quality adversarial perturbations from the historical training process. We provide theoretical analysis for the proposed initialization and propose a prior-guided regularization method that boosts the smoothness of the loss function. Additionally, we design a prior-guided ensemble FAT method that averages the different model weights of historical models using different decay rates. Our proposed method, called FGSM-PGK, assembles the prior-guided knowledge, i.e., the prior-guided initialization and model weights, acquired during the historical training process. Evaluations of four datasets demonstrate the superiority of the proposed method.

摘要: 快速对抗训练(FAT)是一种提高鲁棒性的有效方法。然而，原始的脂肪会遭受灾难性的过度拟合，这会在几个训练时期后急剧而突然地降低健壮性。虽然已经提出了各种脂肪变种来防止过度适应，但它们需要很高的培训成本。本文通过比较标准对抗性训练和FAT的训练过程，考察了对抗性样本质量与灾难性过拟合的关系。我们发现，当对抗性例子的攻击成功率变差时，就会发生灾难性的过拟合。基于这一观察结果，我们提出了一种积极的先验指导的对抗性初始化方法，通过在不增加额外训练成本的情况下提高对抗性实例的质量来防止过度拟合。这种初始化是通过使用来自历史训练过程的高质量对抗性扰动来生成的。我们对所提出的初始化方法进行了理论分析，并提出了一种先验引导的正则化方法，提高了损失函数的光滑性。此外，我们设计了一种先验指导的集成FAT方法，该方法使用不同的衰减率来平均历史模型的不同模型权重。我们提出的方法称为FGSM-PGK，它集合了历史训练过程中获得的先验指导知识，即先验指导的初始化和模型权重。对四个数据集的评价表明了该方法的优越性。



## **47. UNICORN: A Unified Backdoor Trigger Inversion Framework**

独角兽：一种统一的后门触发器反转框架 cs.LG

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2304.02786v1) [paper-pdf](http://arxiv.org/pdf/2304.02786v1)

**Authors**: Zhenting Wang, Kai Mei, Juan Zhai, Shiqing Ma

**Abstract**: The backdoor attack, where the adversary uses inputs stamped with triggers (e.g., a patch) to activate pre-planted malicious behaviors, is a severe threat to Deep Neural Network (DNN) models. Trigger inversion is an effective way of identifying backdoor models and understanding embedded adversarial behaviors. A challenge of trigger inversion is that there are many ways of constructing the trigger. Existing methods cannot generalize to various types of triggers by making certain assumptions or attack-specific constraints. The fundamental reason is that existing work does not consider the trigger's design space in their formulation of the inversion problem. This work formally defines and analyzes the triggers injected in different spaces and the inversion problem. Then, it proposes a unified framework to invert backdoor triggers based on the formalization of triggers and the identified inner behaviors of backdoor models from our analysis. Our prototype UNICORN is general and effective in inverting backdoor triggers in DNNs. The code can be found at https://github.com/RU-System-Software-and-Security/UNICORN.

摘要: 后门攻击是对深度神经网络(DNN)模型的严重威胁，攻击者使用带有触发器(例如补丁)的输入来激活预先植入的恶意行为。触发反转是识别后门模型和理解嵌入的敌对行为的有效方法。触发器反转的一个挑战是有许多构造触发器的方法。现有方法不能通过做出某些假设或特定于攻击的约束来对各种类型的触发器进行泛化。其根本原因是现有的工作没有考虑触发器的设计空间在他们的反问题的公式。这项工作形式化地定义和分析了不同空间中注入的触发器和反演问题。然后，基于触发器的形式化和从分析中识别出的后门模型的内部行为，提出了一个倒置后门触发器的统一框架。我们的原型独角兽在倒置DNN中的后门触发器方面是通用的和有效的。代码可在https://github.com/RU-System-Software-and-Security/UNICORN.上找到



## **48. Planning for Attacker Entrapment in Adversarial Settings**

对抗性环境下攻击者诱捕的计划 cs.AI

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2303.00822v2) [paper-pdf](http://arxiv.org/pdf/2303.00822v2)

**Authors**: Brittany Cates, Anagha Kulkarni, Sarath Sreedharan

**Abstract**: In this paper, we propose a planning framework to generate a defense strategy against an attacker who is working in an environment where a defender can operate without the attacker's knowledge. The objective of the defender is to covertly guide the attacker to a trap state from which the attacker cannot achieve their goal. Further, the defender is constrained to achieve its goal within K number of steps, where K is calculated as a pessimistic lower bound within which the attacker is unlikely to suspect a threat in the environment. Such a defense strategy is highly useful in real world systems like honeypots or honeynets, where an unsuspecting attacker interacts with a simulated production system while assuming it is the actual production system. Typically, the interaction between an attacker and a defender is captured using game theoretic frameworks. Our problem formulation allows us to capture it as a much simpler infinite horizon discounted MDP, in which the optimal policy for the MDP gives the defender's strategy against the actions of the attacker. Through empirical evaluation, we show the merits of our problem formulation.

摘要: 在本文中，我们提出了一个计划框架来生成针对攻击者的防御策略，在这种环境中，防御者可以在攻击者不知情的情况下操作。防御者的目标是秘密地引导攻击者进入陷阱状态，使攻击者无法实现他们的目标。此外，防御者被限制在K个步骤内实现其目标，其中K被计算为悲观的下限，在该下限内攻击者不太可能怀疑环境中的威胁。这样的防御策略在蜜罐或蜜网等现实世界系统中非常有用，在这些系统中，毫无戒心的攻击者与模拟生产系统交互，同时假设它是实际的生产系统。通常，攻击者和防御者之间的交互是使用博弈论框架来捕捉的。我们的问题公式允许我们将其捕获为一个更简单的无限地平线折扣MDP，其中MDP的最优策略给出了防御者针对攻击者的行为的策略。通过实证评估，我们展示了我们的问题描述的优点。



## **49. Domain Generalization with Adversarial Intensity Attack for Medical Image Segmentation**

基于对抗性强度攻击的医学图像分割领域泛化 eess.IV

Code is available upon publication

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2304.02720v1) [paper-pdf](http://arxiv.org/pdf/2304.02720v1)

**Authors**: Zheyuan Zhang, Bin Wang, Lanhong Yao, Ugur Demir, Debesh Jha, Ismail Baris Turkbey, Boqing Gong, Ulas Bagci

**Abstract**: Most statistical learning algorithms rely on an over-simplified assumption, that is, the train and test data are independent and identically distributed. In real-world scenarios, however, it is common for models to encounter data from new and different domains to which they were not exposed to during training. This is often the case in medical imaging applications due to differences in acquisition devices, imaging protocols, and patient characteristics. To address this problem, domain generalization (DG) is a promising direction as it enables models to handle data from previously unseen domains by learning domain-invariant features robust to variations across different domains. To this end, we introduce a novel DG method called Adversarial Intensity Attack (AdverIN), which leverages adversarial training to generate training data with an infinite number of styles and increase data diversity while preserving essential content information. We conduct extensive evaluation experiments on various multi-domain segmentation datasets, including 2D retinal fundus optic disc/cup and 3D prostate MRI. Our results demonstrate that AdverIN significantly improves the generalization ability of the segmentation models, achieving significant improvement on these challenging datasets. Code is available upon publication.

摘要: 大多数统计学习算法依赖于一个过于简化的假设，即训练数据和测试数据是独立的、同分布的。然而，在现实世界的场景中，模型经常遇到来自新的不同领域的数据，而它们在培训期间没有接触到这些数据。由于采集设备、成像协议和患者特征的不同，在医学成像应用中通常会出现这种情况。为了解决这个问题，领域泛化(DG)是一个很有前途的方向，因为它通过学习对不同领域之间的变化具有鲁棒性的领域不变特征，使模型能够处理来自以前未见过的领域的数据。为此，我们引入了一种称为对抗性强度攻击(AdverIN)的DG方法，它利用对抗性训练来生成具有无限样式的训练数据，并在保留基本内容信息的同时增加数据多样性。我们在各种多域分割数据集上进行了广泛的评估实验，包括2D视网膜眼底视盘/视杯和3D前列腺MRI。我们的结果表明，AdverIN显著提高了分割模型的泛化能力，在这些具有挑战性的数据集上取得了显著的改善。代码在发布后即可使用。



## **50. A Certified Radius-Guided Attack Framework to Image Segmentation Models**

一种用于图像分割模型的半径制导攻击认证框架 cs.CV

Accepted by EuroSP 2023

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2304.02693v1) [paper-pdf](http://arxiv.org/pdf/2304.02693v1)

**Authors**: Wenjie Qu, Youqi Li, Binghui Wang

**Abstract**: Image segmentation is an important problem in many safety-critical applications. Recent studies show that modern image segmentation models are vulnerable to adversarial perturbations, while existing attack methods mainly follow the idea of attacking image classification models. We argue that image segmentation and classification have inherent differences, and design an attack framework specially for image segmentation models. Our attack framework is inspired by certified radius, which was originally used by defenders to defend against adversarial perturbations to classification models. We are the first, from the attacker perspective, to leverage the properties of certified radius and propose a certified radius guided attack framework against image segmentation models. Specifically, we first adapt randomized smoothing, the state-of-the-art certification method for classification models, to derive the pixel's certified radius. We then focus more on disrupting pixels with relatively smaller certified radii and design a pixel-wise certified radius guided loss, when plugged into any existing white-box attack, yields our certified radius-guided white-box attack. Next, we propose the first black-box attack to image segmentation models via bandit. We design a novel gradient estimator, based on bandit feedback, which is query-efficient and provably unbiased and stable. We use this gradient estimator to design a projected bandit gradient descent (PBGD) attack, as well as a certified radius-guided PBGD (CR-PBGD) attack. We prove our PBGD and CR-PBGD attacks can achieve asymptotically optimal attack performance with an optimal rate. We evaluate our certified-radius guided white-box and black-box attacks on multiple modern image segmentation models and datasets. Our results validate the effectiveness of our certified radius-guided attack framework.

摘要: 图像分割是许多安全关键应用中的一个重要问题。最近的研究表明，现代图像分割模型容易受到对抗性扰动，而现有的攻击方法主要遵循攻击图像分类模型的思想。我们认为图像分割和分类有着内在的区别，并设计了一个专门针对图像分割模型的攻击框架。我们的攻击框架的灵感来自认证的RADIUS，它最初是防御者用来防御分类模型的对抗性扰动的。从攻击者的角度来看，我们是第一个利用认证RADIUS的属性，并提出了针对图像分割模型的认证RADIUS制导攻击框架。具体地说，我们首先采用随机平滑，这是目前最先进的分类模型认证方法，来推导像素的认证半径。然后，我们将重点放在具有相对较小认证半径的像素上，并设计一个像素级认证半径制导损耗，当插入任何现有的白盒攻击时，就会产生我们认证的半径制导白盒攻击。接下来，我们提出了利用盗贼对图像分割模型进行第一次黑盒攻击。我们设计了一种新的基于强盗反馈的梯度估计器，该估计器具有查询效率高、可证明无偏和稳定的特点。我们使用这个梯度估计器设计了一个投影的强盗梯度下降(PBGD)攻击，以及一个认证的半径制导的PBGD(CR-PBGD)攻击。我们证明了我们的PBGD和CR-PBGD攻击能够以最优率获得渐近最优的攻击性能。我们在多种现代图像分割模型和数据集上评估了我们的认证半径制导的白盒和黑盒攻击。我们的结果验证了我们认证的半径制导攻击框架的有效性。



