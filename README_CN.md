# Latest Adversarial Attack Papers
**update at 2022-11-26 11:39:11**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Query Efficient Cross-Dataset Transferable Black-Box Attack on Action Recognition**

基于动作识别的查询高效跨数据集可转移黑盒攻击 cs.CV

**SubmitDate**: 2022-11-23    [abs](http://arxiv.org/abs/2211.13171v1) [paper-pdf](http://arxiv.org/pdf/2211.13171v1)

**Authors**: Rohit Gupta, Naveed Akhtar, Gaurav Kumar Nayak, Ajmal Mian, Mubarak Shah

**Abstract**: Black-box adversarial attacks present a realistic threat to action recognition systems. Existing black-box attacks follow either a query-based approach where an attack is optimized by querying the target model, or a transfer-based approach where attacks are generated using a substitute model. While these methods can achieve decent fooling rates, the former tends to be highly query-inefficient while the latter assumes extensive knowledge of the black-box model's training data. In this paper, we propose a new attack on action recognition that addresses these shortcomings by generating perturbations to disrupt the features learned by a pre-trained substitute model to reduce the number of queries. By using a nearly disjoint dataset to train the substitute model, our method removes the requirement that the substitute model be trained using the same dataset as the target model, and leverages queries to the target model to retain the fooling rate benefits provided by query-based methods. This ultimately results in attacks which are more transferable than conventional black-box attacks. Through extensive experiments, we demonstrate highly query-efficient black-box attacks with the proposed framework. Our method achieves 8% and 12% higher deception rates compared to state-of-the-art query-based and transfer-based attacks, respectively.

摘要: 黑盒对抗性攻击对动作识别系统构成了现实的威胁。现有的黑盒攻击要么遵循基于查询的方法，其中通过查询目标模型来优化攻击，要么遵循基于转移的方法，其中使用替代模型来生成攻击。虽然这些方法可以获得不错的愚弄率，但前者往往查询效率很低，而后者则假设对黑盒模型的训练数据有广泛的了解。在本文中，我们提出了一种新的动作识别攻击，通过产生扰动来扰乱预先训练的替代模型学习的特征来减少查询数量，从而解决了这些缺点。通过使用几乎不相交的数据集来训练替换模型，我们的方法消除了使用与目标模型相同的数据集来训练替换模型的要求，并利用对目标模型的查询来保留基于查询的方法所提供的欺骗率优势。这最终导致了比传统的黑盒攻击更具转移性的攻击。通过大量的实验，我们展示了基于该框架的高效查询黑盒攻击。与最先进的基于查询和基于传输的攻击相比，我们的方法分别提高了8%和12%的欺骗率。



## **2. Adversarial Attacks are a Surprisingly Strong Baseline for Poisoning Few-Shot Meta-Learners**

对抗性攻击是毒害少数元学习者的一个令人惊讶的强大基线 cs.LG

Accepted at I Can't Believe It's Not Better Workshop, Neurips 2022

**SubmitDate**: 2022-11-23    [abs](http://arxiv.org/abs/2211.12990v1) [paper-pdf](http://arxiv.org/pdf/2211.12990v1)

**Authors**: Elre T. Oldewage, John Bronskill, Richard E. Turner

**Abstract**: This paper examines the robustness of deployed few-shot meta-learning systems when they are fed an imperceptibly perturbed few-shot dataset. We attack amortized meta-learners, which allows us to craft colluding sets of inputs that are tailored to fool the system's learning algorithm when used as training data. Jointly crafted adversarial inputs might be expected to synergistically manipulate a classifier, allowing for very strong data-poisoning attacks that would be hard to detect. We show that in a white box setting, these attacks are very successful and can cause the target model's predictions to become worse than chance. However, in opposition to the well-known transferability of adversarial examples in general, the colluding sets do not transfer well to different classifiers. We explore two hypotheses to explain this: 'overfitting' by the attack, and mismatch between the model on which the attack is generated and that to which the attack is transferred. Regardless of the mitigation strategies suggested by these hypotheses, the colluding inputs transfer no better than adversarial inputs that are generated independently in the usual way.

摘要: 本文考察了当部署的少镜头元学习系统被提供给一个不可察觉的扰动的少镜头数据集时，它们的稳健性。我们攻击分期元学习者，它允许我们制作合谋的输入集，当用作训练数据时，这些输入集被量身定做，以愚弄系统的学习算法。联合编制的敌意输入可能会协同操作分类器，从而允许很难检测到的非常强大的数据中毒攻击。我们表明，在白盒设置，这些攻击是非常成功的，可以导致目标模型的预测变得比机会更糟糕。然而，与众所周知的对抗性例子的可转移性相反，合谋集合不能很好地迁移到不同的量词上。我们探索了两个假设来解释这一点：攻击的“过度匹配”，以及生成攻击的模型和攻击转移到的模型之间的不匹配。不管这些假设建议的缓解策略是什么，串通输入的转移效果并不比以通常方式独立生成的对抗性输入好。



## **3. Visual Information Hiding Based on Obfuscating Adversarial Perturbations**

基于混淆对抗性扰动的视觉信息隐藏 cs.CV

**SubmitDate**: 2022-11-23    [abs](http://arxiv.org/abs/2209.15304v2) [paper-pdf](http://arxiv.org/pdf/2209.15304v2)

**Authors**: Zhigang Su, Dawei Zhou, Decheng Liu, Nannan Wang, Zhen Wang, Xinbo Gao

**Abstract**: Growing leakage and misuse of visual information raise security and privacy concerns, which promotes the development of information protection. Existing adversarial perturbations-based methods mainly focus on the de-identification against deep learning models. However, the inherent visual information of the data has not been well protected. In this work, inspired by the Type-I adversarial attack, we propose an adversarial visual information hiding method to protect the visual privacy of data. Specifically, the method generates obfuscating adversarial perturbations to obscure the visual information of the data. Meanwhile, it maintains the hidden objectives to be correctly predicted by models. In addition, our method does not modify the parameters of the applied model, which makes it flexible for different scenarios. Experimental results on the recognition and classification tasks demonstrate that the proposed method can effectively hide visual information and hardly affect the performances of models. The code is available in the supplementary material.

摘要: 日益增长的视觉信息泄露和滥用引发了人们对安全和隐私的担忧，这推动了信息保护的发展。现有的基于对抗性扰动的方法主要集中在针对深度学习模型的去识别。然而，数据固有的视觉信息并没有得到很好的保护。在这项工作中，受Type-I对抗攻击的启发，我们提出了一种对抗性视觉信息隐藏方法来保护数据的视觉隐私。具体地说，该方法产生模糊的对抗性扰动以模糊数据的可视信息。同时，保持模型对隐含目标的正确预测。此外，我们的方法不修改应用模型的参数，这使得它可以灵活地适应不同的场景。在识别和分类任务上的实验结果表明，该方法能够有效地隐藏视觉信息，且几乎不影响模型的性能。该代码可在补充材料中找到。



## **4. Privacy-Enhancing Optical Embeddings for Lensless Classification**

用于无透镜分类的隐私增强光学嵌入技术 cs.CV

30 pages, 27 figures, under review, for code, see  https://github.com/ebezzam/LenslessClassification. arXiv admin note:  substantial text overlap with arXiv:2206.01429

**SubmitDate**: 2022-11-23    [abs](http://arxiv.org/abs/2211.12864v1) [paper-pdf](http://arxiv.org/pdf/2211.12864v1)

**Authors**: Eric Bezzam, Martin Vetterli, Matthieu Simeoni

**Abstract**: Lensless imaging can provide visual privacy due to the highly multiplexed characteristic of its measurements. However, this alone is a weak form of security, as various adversarial attacks can be designed to invert the one-to-many scene mapping of such cameras. In this work, we enhance the privacy provided by lensless imaging by (1) downsampling at the sensor and (2) using a programmable mask with variable patterns as our optical encoder. We build a prototype from a low-cost LCD and Raspberry Pi components, for a total cost of around 100 USD. This very low price point allows our system to be deployed and leveraged in a broad range of applications. In our experiments, we first demonstrate the viability and reconfigurability of our system by applying it to various classification tasks: MNIST, CelebA (face attributes), and CIFAR10. By jointly optimizing the mask pattern and a digital classifier in an end-to-end fashion, low-dimensional, privacy-enhancing embeddings are learned directly at the sensor. Secondly, we show how the proposed system, through variable mask patterns, can thwart adversaries that attempt to invert the system (1) via plaintext attacks or (2) in the event of camera parameters leaks. We demonstrate the defense of our system to both risks, with 55% and 26% drops in image quality metrics for attacks based on model-based convex optimization and generative neural networks respectively. We open-source a wave propagation and camera simulator needed for end-to-end optimization, the training software, and a library for interfacing with the camera.

摘要: 由于其测量的高度多路特性，无透镜成像可以提供视觉隐私。然而，这本身就是一种薄弱的安全形式，因为可以设计各种对抗性攻击来反转此类摄像机的一对多场景映射。在这项工作中，我们通过(1)在传感器处下采样和(2)使用具有可变图案的可编程掩模作为我们的光学编码器来增强无透镜成像所提供的隐私。我们用低成本的LCD和树莓PI组件构建了一个原型，总成本约为100美元。这一非常低的价位允许我们的系统在广泛的应用中进行部署和利用。在我们的实验中，我们首先将我们的系统应用于各种分类任务：MNIST、CelebA(人脸属性)和CIFAR10，从而证明了该系统的可行性和可重构性。通过以端到端的方式联合优化掩模图案和数字分类器，可以直接在传感器上学习低维、隐私增强的嵌入。其次，我们展示了所提出的系统如何通过可变掩码模式来挫败试图(1)通过明文攻击或(2)在摄像机参数泄露的情况下反转系统的攻击者。我们展示了我们的系统对这两种风险的防御，基于模型的凸优化和基于生成神经网络的攻击的图像质量度量分别下降了55%和26%。我们开源了端到端优化所需的波传播和摄像机模拟器、培训软件和用于与摄像机接口的库。



## **5. Byzantine Multiple Access Channels -- Part I: Reliable Communication**

拜占庭式多址接入信道--第一部分：可靠通信 cs.IT

This supercedes Part I of arxiv:1904.11925

**SubmitDate**: 2022-11-23    [abs](http://arxiv.org/abs/2211.12769v1) [paper-pdf](http://arxiv.org/pdf/2211.12769v1)

**Authors**: Neha Sangwan, Mayank Bakshi, Bikash Kumar Dey, Vinod M. Prabhakaran

**Abstract**: We study communication over a Multiple Access Channel (MAC) where users can possibly be adversarial. When all users are non-adversarial, we want their messages to be decoded reliably. When a user behaves adversarially, we require that the honest users' messages be decoded reliably. An adversarial user can mount an attack by sending any input into the channel rather than following the protocol. It turns out that the $2$-user MAC capacity region follows from the point-to-point Arbitrarily Varying Channel (AVC) capacity. For the $3$-user MAC in which at most one user may be malicious, we characterize the capacity region for deterministic codes and randomized codes (where each user shares an independent random secret key with the receiver). These results are then generalized for the $k$-user MAC where the adversary may control all users in one out of a collection of given subsets.

摘要: 我们研究了多路访问信道(MAC)上的通信，其中用户可能是对抗性的。当所有用户都是非对抗性的时，我们希望他们的消息被可靠地解码。当用户做出恶意行为时，我们要求可靠地解码诚实用户的消息。敌意用户可以通过向通道发送任何输入而不是遵循协议来发动攻击。事实证明，$2$-用户MAC容量区域紧随点对点任意变化信道(AVC)容量。对于最多一个用户可能是恶意用户的$3$-用户MAC，我们刻画了确定码和随机码的容量域(其中每个用户与接收方共享一个独立的随机密钥)。然后将这些结果推广到$k$-用户MAC，其中对手可以控制给定子集集合中的一个用户。



## **6. Reliable Robustness Evaluation via Automatically Constructed Attack Ensembles**

基于自动构建攻击集成的可靠健壮性评估 cs.LG

**SubmitDate**: 2022-11-23    [abs](http://arxiv.org/abs/2211.12713v1) [paper-pdf](http://arxiv.org/pdf/2211.12713v1)

**Authors**: Shengcai Liu, Fu Peng, Ke Tang

**Abstract**: Attack Ensemble (AE), which combines multiple attacks together, provides a reliable way to evaluate adversarial robustness. In practice, AEs are often constructed and tuned by human experts, which however tends to be sub-optimal and time-consuming. In this work, we present AutoAE, a conceptually simple approach for automatically constructing AEs. In brief, AutoAE repeatedly adds the attack and its iteration steps to the ensemble that maximizes ensemble improvement per additional iteration consumed. We show theoretically that AutoAE yields AEs provably within a constant factor of the optimal for a given defense. We then use AutoAE to construct two AEs for $l_{\infty}$ and $l_2$ attacks, and apply them without any tuning or adaptation to 45 top adversarial defenses on the RobustBench leaderboard. In all except one cases we achieve equal or better (often the latter) robustness evaluation than existing AEs, and notably, in 29 cases we achieve better robustness evaluation than the best known one. Such performance of AutoAE shows itself as a reliable evaluation protocol for adversarial robustness, which further indicates the huge potential of automatic AE construction. Code is available at \url{https://github.com/LeegerPENG/AutoAE}.

摘要: 攻击集成(AE)将多种攻击结合在一起，为评估对手的健壮性提供了一种可靠的方法。在实践中，工程实体通常是由人类专家构建和调整的，然而，这往往是次优的和耗时的。在这项工作中，我们提出了AutoAE，一种概念上简单的自动构建AE的方法。简而言之，AutoAE反复将攻击及其迭代步骤添加到合奏中，从而在每次消耗的额外迭代中最大化合奏改进。我们从理论上证明，对于给定的防御，AutoAE可以在最优值的恒定因子内产生AEs。然后，我们使用AutoAE为$l_{inty}$和$l_2$攻击构造两个AE，并且在没有任何调整或调整的情况下将它们应用于罗布斯班奇排行榜上的45个顶级对手防御。除了一种情况外，在所有情况下，我们都实现了与现有AE相同或更好的健壮性评估(通常是后者)，值得注意的是，在29种情况下，我们实现了比最知名的健壮性评估更好的健壮性评估。AutoAE的这种性能表明它是一种可靠的对抗健壮性评估协议，这进一步表明了自动构建AE的巨大潜力。代码位于\url{https://github.com/LeegerPENG/AutoAE}.



## **7. Benchmarking Adversarially Robust Quantum Machine Learning at Scale**

在规模上对相反稳健的量子机器学习进行基准测试 quant-ph

10 pages, 5 Figures

**SubmitDate**: 2022-11-23    [abs](http://arxiv.org/abs/2211.12681v1) [paper-pdf](http://arxiv.org/pdf/2211.12681v1)

**Authors**: Maxwell T. West, Sarah M. Erfani, Christopher Leckie, Martin Sevior, Lloyd C. L. Hollenberg, Muhammad Usman

**Abstract**: Machine learning (ML) methods such as artificial neural networks are rapidly becoming ubiquitous in modern science, technology and industry. Despite their accuracy and sophistication, neural networks can be easily fooled by carefully designed malicious inputs known as adversarial attacks. While such vulnerabilities remain a serious challenge for classical neural networks, the extent of their existence is not fully understood in the quantum ML setting. In this work, we benchmark the robustness of quantum ML networks, such as quantum variational classifiers (QVC), at scale by performing rigorous training for both simple and complex image datasets and through a variety of high-end adversarial attacks. Our results show that QVCs offer a notably enhanced robustness against classical adversarial attacks by learning features which are not detected by the classical neural networks, indicating a possible quantum advantage for ML tasks. Contrarily, and remarkably, the converse is not true, with attacks on quantum networks also capable of deceiving classical neural networks. By combining quantum and classical network outcomes, we propose a novel adversarial attack detection technology. Traditionally quantum advantage in ML systems has been sought through increased accuracy or algorithmic speed-up, but our work has revealed the potential for a new kind of quantum advantage through superior robustness of ML models, whose practical realisation will address serious security concerns and reliability issues of ML algorithms employed in a myriad of applications including autonomous vehicles, cybersecurity, and surveillance robotic systems.

摘要: 人工神经网络等机器学习(ML)方法正迅速在现代科学、技术和工业中变得无处不在。尽管神经网络的准确性和复杂性很高，但它们很容易被精心设计的恶意输入所愚弄，这种恶意输入被称为对抗性攻击。虽然这些漏洞对经典神经网络来说仍然是一个严重的挑战，但在量子ML环境中，它们的存在程度并没有得到充分的了解。在这项工作中，我们通过对简单和复杂的图像数据集进行严格的训练，以及通过各种高端的对抗性攻击，在规模上对量子ML网络(如量子变分分类器)的健壮性进行基准测试。我们的结果表明，QVC通过学习经典神经网络没有检测到的特征，对经典对手攻击提供了显著增强的稳健性，这表明对于ML任务可能具有量子优势。相反，值得注意的是，反之亦然，对量子网络的攻击也能够欺骗经典神经网络。结合量子网络和经典网络的研究成果，提出了一种新的敌意攻击检测技术。传统上，ML系统中的量子优势是通过提高精确度或算法加速来寻求的，但我们的工作揭示了通过ML模型的卓越健壮性来实现新型量子优势的潜力，其实际实现将解决ML算法在包括自动驾驶汽车、网络安全和监控机器人系统在内的众多应用中使用的严重安全问题和可靠性问题。



## **8. Membership Inference Attacks via Adversarial Examples**

基于对抗性例子的成员关系推理攻击 cs.LG

Trustworthy and Socially Responsible Machine Learning (TSRML 2022)  co-located with NeurIPS 2022

**SubmitDate**: 2022-11-23    [abs](http://arxiv.org/abs/2207.13572v2) [paper-pdf](http://arxiv.org/pdf/2207.13572v2)

**Authors**: Hamid Jalalzai, Elie Kadoche, Rémi Leluc, Vincent Plassier

**Abstract**: The raise of machine learning and deep learning led to significant improvement in several domains. This change is supported by both the dramatic rise in computation power and the collection of large datasets. Such massive datasets often include personal data which can represent a threat to privacy. Membership inference attacks are a novel direction of research which aims at recovering training data used by a learning algorithm. In this paper, we develop a mean to measure the leakage of training data leveraging a quantity appearing as a proxy of the total variation of a trained model near its training samples. We extend our work by providing a novel defense mechanism. Our contributions are supported by empirical evidence through convincing numerical experiments.

摘要: 机器学习和深度学习的兴起导致了几个领域的显著改善。计算能力的戏剧性增长和大型数据集的收集都支持这种变化。如此庞大的数据集通常包括可能对隐私构成威胁的个人数据。隶属度推理攻击是一个新的研究方向，其目的是恢复学习算法所使用的训练数据。在本文中，我们开发了一种方法来衡量训练数据的泄漏，该方法利用一个量来衡量训练模型在其训练样本附近的总变异。我们通过提供一种新颖的防御机制来扩展我们的工作。通过令人信服的数值实验，我们的贡献得到了经验证据的支持。



## **9. Improving Robust Generalization by Direct PAC-Bayesian Bound Minimization**

基于直接PAC-贝叶斯界最小化的鲁棒泛化 cs.LG

**SubmitDate**: 2022-11-22    [abs](http://arxiv.org/abs/2211.12624v1) [paper-pdf](http://arxiv.org/pdf/2211.12624v1)

**Authors**: Zifan Wang, Nan Ding, Tomer Levinboim, Xi Chen, Radu Soricut

**Abstract**: Recent research in robust optimization has shown an overfitting-like phenomenon in which models trained against adversarial attacks exhibit higher robustness on the training set compared to the test set. Although previous work provided theoretical explanations for this phenomenon using a robust PAC-Bayesian bound over the adversarial test error, related algorithmic derivations are at best only loosely connected to this bound, which implies that there is still a gap between their empirical success and our understanding of adversarial robustness theory. To close this gap, in this paper we consider a different form of the robust PAC-Bayesian bound and directly minimize it with respect to the model posterior. The derivation of the optimal solution connects PAC-Bayesian learning to the geometry of the robust loss surface through a Trace of Hessian (TrH) regularizer that measures the surface flatness. In practice, we restrict the TrH regularizer to the top layer only, which results in an analytical solution to the bound whose computational cost does not depend on the network depth. Finally, we evaluate our TrH regularization approach over CIFAR-10/100 and ImageNet using Vision Transformers (ViT) and compare against baseline adversarial robustness algorithms. Experimental results show that TrH regularization leads to improved ViT robustness that either matches or surpasses previous state-of-the-art approaches while at the same time requires less memory and computational cost.

摘要: 最近在稳健优化方面的研究表明，与测试集相比，针对对手攻击而训练的模型在训练集上表现出更强的稳健性。尽管以前的工作使用了对抗测试误差上的稳健PAC-贝叶斯界来解释这一现象，但相关的算法推导充其量只能松散地联系到这个界，这意味着他们的经验成功与我们对对抗稳健性理论的理解之间仍然存在差距。为了缩小这一差距，在本文中，我们考虑了一种不同形式的稳健PAC-贝叶斯上界，并直接将其关于模型后验概率最小化。最优解的推导通过测量曲面平坦度的跟踪黑森(TRH)正则化将PAC-贝叶斯学习与稳健损失曲面的几何联系起来。在实际应用中，我们将TRH正则化仅限于顶层，从而得到了边界的解析解，其计算成本与网络深度无关。最后，我们使用视觉转换器(VIT)在CIFAR-10/100和ImageNet上评估了我们的TRH正则化方法，并与基线对抗健壮性算法进行了比较。实验结果表明，TRH正则化提高了VIT的稳健性，达到或超过了以往的最新方法，同时需要更少的内存和计算代价。



## **10. Diagnostics for Deep Neural Networks with Automated Copy/Paste Attacks**

具有自动复制/粘贴攻击的深度神经网络的诊断 cs.LG

**SubmitDate**: 2022-11-22    [abs](http://arxiv.org/abs/2211.10024v2) [paper-pdf](http://arxiv.org/pdf/2211.10024v2)

**Authors**: Stephen Casper, Kaivalya Hariharan, Dylan Hadfield-Menell

**Abstract**: Deep neural networks (DNNs) are powerful, but they can make mistakes that pose significant risks. A model performing well on a test set does not imply safety in deployment, so it is important to have additional tools to understand its flaws. Adversarial examples can help reveal weaknesses, but they are often difficult for a human to interpret or draw generalizable, actionable conclusions from. Some previous works have addressed this by studying human-interpretable attacks. We build on these with three contributions. First, we introduce a method termed Search for Natural Adversarial Features Using Embeddings (SNAFUE) which offers a fully-automated method for finding "copy/paste" attacks in which one natural image can be pasted into another in order to induce an unrelated misclassification. Second, we use this to red team an ImageNet classifier and identify hundreds of easily-describable sets of vulnerabilities. Third, we compare this approach with other interpretability tools by attempting to rediscover trojans. Our results suggest that SNAFUE can be useful for interpreting DNNs and generating adversarial data for them. Code is available at https://github.com/thestephencasper/snafue

摘要: 深度神经网络(DNN)功能强大，但它们可能会犯下构成重大风险的错误。一个在测试集上表现良好的模型并不意味着部署的安全性，因此有更多的工具来了解它的缺陷是很重要的。对抗性的例子可以帮助揭示弱点，但对人类来说，它们往往很难解释，也很难得出可概括的、可操作的结论。以前的一些工作通过研究人类可解释的攻击来解决这个问题。我们在这些基础上做出了三项贡献。首先，我们介绍了一种称为使用嵌入搜索自然对抗性特征(SNAFUE)的方法，它提供了一种全自动的方法来发现复制/粘贴攻击，在该攻击中，一幅自然图像可以粘贴到另一幅图像中，从而导致不相关的错误分类。其次，我们使用它对ImageNet分类器进行分组，并识别数百组易于描述的漏洞。第三，我们通过尝试重新发现特洛伊木马，将此方法与其他可解释性工具进行了比较。我们的结果表明，SNAFUE可以用于解释DNN并为它们生成对抗性数据。代码可在https://github.com/thestephencasper/snafue上找到



## **11. Attacking Image Splicing Detection and Localization Algorithms Using Synthetic Traces**

基于合成痕迹的攻击图像拼接检测与定位算法 eess.IV

**SubmitDate**: 2022-11-22    [abs](http://arxiv.org/abs/2211.12314v1) [paper-pdf](http://arxiv.org/pdf/2211.12314v1)

**Authors**: Shengbang Fang, Matthew C Stamm

**Abstract**: Recent advances in deep learning have enabled forensics researchers to develop a new class of image splicing detection and localization algorithms. These algorithms identify spliced content by detecting localized inconsistencies in forensic traces using Siamese neural networks, either explicitly during analysis or implicitly during training. At the same time, deep learning has enabled new forms of anti-forensic attacks, such as adversarial examples and generative adversarial network (GAN) based attacks. Thus far, however, no anti-forensic attack has been demonstrated against image splicing detection and localization algorithms. In this paper, we propose a new GAN-based anti-forensic attack that is able to fool state-of-the-art splicing detection and localization algorithms such as EXIF-Net, Noiseprint, and Forensic Similarity Graphs. This attack operates by adversarially training an anti-forensic generator against a set of Siamese neural networks so that it is able to create synthetic forensic traces. Under analysis, these synthetic traces appear authentic and are self-consistent throughout an image. Through a series of experiments, we demonstrate that our attack is capable of fooling forensic splicing detection and localization algorithms without introducing visually detectable artifacts into an attacked image. Additionally, we demonstrate that our attack outperforms existing alternative attack approaches. %

摘要: 深度学习的最新进展使法医研究人员能够开发出一类新的图像拼接检测和定位算法。这些算法通过使用暹罗神经网络检测法医痕迹中的局部不一致来识别剪接内容，无论是在分析期间显式地还是在训练期间隐式地。与此同时，深度学习使反取证攻击的新形式成为可能，例如对抗性例子和基于生成性对抗性网络(GAN)的攻击。然而，到目前为止，还没有针对图像拼接检测和定位算法的反取证攻击。在本文中，我们提出了一种新的基于GAN的反取证攻击，它能够欺骗最新的剪接检测和定位算法，如EXIF-Net、Noiseprint和法医相似图。这种攻击通过恶意训练反法医生成器来对抗一组暹罗神经网络，以便它能够创建合成的法医痕迹。在分析之下，这些人工合成的痕迹看起来是真实的，并且在整个图像中都是自我一致的。通过一系列的实验，我们证明了我们的攻击能够欺骗法医剪接检测和定位算法，而不会在攻击图像中引入视觉上可检测的伪影。此外，我们还证明了我们的攻击优于现有的替代攻击方法。百分比



## **12. Jointly Attacking Graph Neural Network and its Explanations**

联合攻击图神经网络及其解释 cs.LG

Accepted by ICDE 2023 (39th IEEE International Conference on Data  Engineering)

**SubmitDate**: 2022-11-22    [abs](http://arxiv.org/abs/2108.03388v2) [paper-pdf](http://arxiv.org/pdf/2108.03388v2)

**Authors**: Wenqi Fan, Wei Jin, Xiaorui Liu, Han Xu, Xianfeng Tang, Suhang Wang, Qing Li, Jiliang Tang, Jianping Wang, Charu Aggarwal

**Abstract**: Graph Neural Networks (GNNs) have boosted the performance for many graph-related tasks. Despite the great success, recent studies have shown that GNNs are highly vulnerable to adversarial attacks, where adversaries can mislead the GNNs' prediction by modifying graphs. On the other hand, the explanation of GNNs (GNNExplainer) provides a better understanding of a trained GNN model by generating a small subgraph and features that are most influential for its prediction. In this paper, we first perform empirical studies to validate that GNNExplainer can act as an inspection tool and have the potential to detect the adversarial perturbations for graphs. This finding motivates us to further initiate a new problem investigation: Whether a graph neural network and its explanations can be jointly attacked by modifying graphs with malicious desires? It is challenging to answer this question since the goals of adversarial attacks and bypassing the GNNExplainer essentially contradict each other. In this work, we give a confirmative answer to this question by proposing a novel attack framework (GEAttack), which can attack both a GNN model and its explanations by simultaneously exploiting their vulnerabilities. Extensive experiments on two explainers (GNNExplainer and PGExplainer) under various real-world datasets demonstrate the effectiveness of the proposed method.

摘要: 图神经网络(GNN)提高了许多与图相关的任务的性能。尽管取得了巨大的成功，但最近的研究表明，GNN非常容易受到对手攻击，对手可以通过修改图来误导GNN的预测。另一方面，对GNN的解释(GNNExplainer)通过生成一个小的子图和对其预测最有影响的特征来更好地理解训练的GNN模型。在本文中，我们首先进行了实证研究，以验证GNNExplainer可以作为一种检测工具，并具有检测图的敌意扰动的潜力。这一发现促使我们进一步发起一个新的问题研究：图神经网络及其解释是否可以通过恶意修改图来联合攻击？回答这个问题很有挑战性，因为敌意攻击和绕过GNNExplainer的目标本质上是相互矛盾的。在这项工作中，我们提出了一个新的攻击框架(GEAttack)，它可以通过同时利用GNN模型及其解释的漏洞来攻击它们，从而给出了一个肯定的答案。在两个解释器(GNNExplainer和PGExplainer)上的大量实验证明了该方法的有效性。



## **13. PointCA: Evaluating the Robustness of 3D Point Cloud Completion Models Against Adversarial Examples**

PointCA：评估3D点云补全模型对敌方示例的稳健性 cs.CV

Accepted by the 37th AAAI Conference on Artificial Intelligence  (AAAI-23)

**SubmitDate**: 2022-11-22    [abs](http://arxiv.org/abs/2211.12294v1) [paper-pdf](http://arxiv.org/pdf/2211.12294v1)

**Authors**: Shengshan Hu, Junwei Zhang, Wei Liu, Junhui Hou, Minghui Li, Leo Yu Zhang, Hai Jin, Lichao Sun

**Abstract**: Point cloud completion, as the upstream procedure of 3D recognition and segmentation, has become an essential part of many tasks such as navigation and scene understanding. While various point cloud completion models have demonstrated their powerful capabilities, their robustness against adversarial attacks, which have been proven to be fatally malicious towards deep neural networks, remains unknown. In addition, existing attack approaches towards point cloud classifiers cannot be applied to the completion models due to different output forms and attack purposes. In order to evaluate the robustness of the completion models, we propose PointCA, the first adversarial attack against 3D point cloud completion models. PointCA can generate adversarial point clouds that maintain high similarity with the original ones, while being completed as another object with totally different semantic information. Specifically, we minimize the representation discrepancy between the adversarial example and the target point set to jointly explore the adversarial point clouds in the geometry space and the feature space. Furthermore, to launch a stealthier attack, we innovatively employ the neighbourhood density information to tailor the perturbation constraint, leading to geometry-aware and distribution-adaptive modifications for each point. Extensive experiments against different premier point cloud completion networks show that PointCA can cause a performance degradation from 77.9% to 16.7%, with the structure chamfer distance kept below 0.01. We conclude that existing completion models are severely vulnerable to adversarial examples, and state-of-the-art defenses for point cloud classification will be partially invalid when applied to incomplete and uneven point cloud data.

摘要: 点云补全作为三维识别和分割的上游步骤，已经成为导航、场景理解等许多任务的重要组成部分。虽然各种点云补全模型已经展示了它们强大的能力，但它们对对手攻击的健壮性仍然未知，这些攻击已被证明是对深度神经网络的致命恶意攻击。此外，由于不同的输出形式和攻击目的，现有的针对点云分类器的攻击方法不能应用于完成模型。为了评估补全模型的健壮性，我们提出了针对三维点云补全模型的第一次对抗性攻击PointCA。PointCA可以生成与原始点云保持高度相似的对抗性点云，同时作为另一个对象完成，具有完全不同的语义信息。具体地说，我们最小化对抗性样本和目标点集之间的表示差异，共同探索几何空间和特征空间中的对抗性点云。此外，为了发动更隐蔽的攻击，我们创新性地使用邻域密度信息来定制扰动约束，导致对每个点的几何感知和分布自适应修改。在不同初始点云完成网络上的大量实验表明，在结构倒角距离保持在0.01以下的情况下，PointCA可以导致性能从77.9%下降到16.7%。我们得出的结论是，现有的完备化模型非常容易受到对手例子的攻击，并且最新的点云分类方法在应用于不完整和不均匀的点云数据时将部分无效。



## **14. SoK: Inference Attacks and Defenses in Human-Centered Wireless Sensing**

SOK：以人为中心的无线传感中的推理攻击和防御 cs.CR

**SubmitDate**: 2022-11-22    [abs](http://arxiv.org/abs/2211.12087v1) [paper-pdf](http://arxiv.org/pdf/2211.12087v1)

**Authors**: Wei Sun, Tingjun Chen, Neil Gong

**Abstract**: Human-centered wireless sensing aims to understand the fine-grained environment and activities of a human using the diverse wireless signals around her. The wireless sensing community has demonstrated the superiority of such techniques in many applications such as smart homes, human-computer interactions, and smart cities. Like many other technologies, wireless sensing is also a double-edged sword. While the sensed information about a human can be used for many good purposes such as enhancing life quality, an adversary can also abuse it to steal private information about the human (e.g., location, living habits, and behavioral biometric characteristics). However, the literature lacks a systematic understanding of the privacy vulnerabilities of wireless sensing and the defenses against them.   In this work, we aim to bridge this gap. First, we propose a framework to systematize wireless sensing-based inference attacks. Our framework consists of three key steps: deploying a sniffing device, sniffing wireless signals, and inferring private information. Our framework can be used to guide the design of new inference attacks since different attacks can instantiate these three steps differently. Second, we propose a defense-in-depth framework to systematize defenses against such inference attacks. The prevention component of our framework aims to prevent inference attacks via obfuscating the wireless signals around a human, while the detection component aims to detect and respond to attacks. Third, based on our attack and defense frameworks, we identify gaps in the existing literature and discuss future research directions.

摘要: 以人为中心的无线传感旨在利用周围不同的无线信号来了解人类的细粒度环境和活动。无线传感社区已经在智能家居、人机交互和智能城市等许多应用中展示了此类技术的优越性。与许多其他技术一样，无线传感也是一把双刃剑。虽然感知到的关于人类的信息可以用于许多良好的目的，如提高生活质量，但攻击者也可以滥用这些信息来窃取关于人类的私人信息(例如，位置、生活习惯和行为生物特征)。然而，这些文献缺乏对无线传感的隐私漏洞及其防御措施的系统了解。在这项工作中，我们的目标是弥合这一差距。首先，我们提出了一个对基于无线感知的推理攻击进行系统化的框架。我们的框架由三个关键步骤组成：部署嗅探设备、嗅探无线信号和推断私人信息。我们的框架可以用来指导新的推理攻击的设计，因为不同的攻击可以不同地实例化这三个步骤。其次，我们提出了一个深度防御框架来系统化对这类推理攻击的防御。该框架的预防部分旨在通过混淆人周围的无线信号来防止推理攻击，而检测部分则旨在检测和响应攻击。第三，基于我们的攻击和防御框架，我们找出了现有文献中的差距，并讨论了未来的研究方向。



## **15. How Fraudster Detection Contributes to Robust Recommendation**

欺诈者检测如何为强大的推荐做出贡献 cs.IR

**SubmitDate**: 2022-11-22    [abs](http://arxiv.org/abs/2211.11534v2) [paper-pdf](http://arxiv.org/pdf/2211.11534v2)

**Authors**: Yuni Lai, Kai Zhou

**Abstract**: The adversarial robustness of recommendation systems under node injection attacks has received considerable research attention. Recently, a robust recommendation system GraphRfi was proposed, and it was shown that GraphRfi could successfully mitigate the effects of injected fake users in the system. Unfortunately, we demonstrate that GraphRfi is still vulnerable to attacks due to the supervised nature of its fraudster detection component. Specifically, we propose a new attack metaC against GraphRfi, and further analyze why GraphRfi fails under such an attack. Based on the insights we obtained from the vulnerability analysis, we build a new robust recommendation system PDR by re-designing the fraudster detection component. Comprehensive experiments show that our defense approach outperforms other benchmark methods under attacks. Overall, our research demonstrates an effective framework of integrating fraudster detection into recommendation to achieve adversarial robustness.

摘要: 推荐系统在节点注入攻击下的对抗健壮性受到了广泛的研究。最近，一个健壮的推荐系统GraphRfi被提出，结果表明GraphRfi可以成功地缓解系统中注入的虚假用户的影响。不幸的是，我们证明，由于其欺诈者检测组件的监督性质，GraphRfi仍然容易受到攻击。具体地说，我们提出了一种新的针对GraphRfi的攻击Metac，并进一步分析了GraphRfi在这种攻击下失败的原因。在漏洞分析的基础上，通过重新设计欺诈者检测组件，构建了一个新的健壮推荐系统PDR。综合实验表明，我们的防御方法在攻击下的性能优于其他基准方法。总体而言，我们的研究证明了一个有效的框架，将欺诈者检测集成到推荐中，以实现对手健壮性。



## **16. Modeling Resources in Permissionless Longest-chain Total-order Broadcast**

无许可最长链全序广播资源建模 cs.CR

**SubmitDate**: 2022-11-22    [abs](http://arxiv.org/abs/2211.12050v1) [paper-pdf](http://arxiv.org/pdf/2211.12050v1)

**Authors**: Sarah Azouvi, Christian Cachin, Duc V. Le, Marko Vukolic, Luca Zanolini

**Abstract**: Blockchain protocols implement total-order broadcast in a permissionless setting, where processes can freely join and leave. In such a setting, to safeguard against Sybil attacks, correct processes rely on cryptographic proofs tied to a particular type of resource to make them eligible to order transactions. For example, in the case of Proof-of-Work (PoW), this resource is computation, and the proof is a solution to a computationally hard puzzle. Conversely, in Proof-of-Stake (PoS), the resource corresponds to the number of coins that every process in the system owns, and a secure lottery selects a process for participation proportionally to its coin holdings.   Although many resource-based blockchain protocols are formally proven secure in the literature, the existing security proofs fail to demonstrate why particular types of resources cause the blockchain protocols to be vulnerable to distinct classes of attacks. For instance, PoS systems are more vulnerable to long-range attacks, where an adversary corrupts past processes to re-write the history, than Proof-of-Work and Proof-of-Storage systems. Proof-of-Storage-based and Proof-of-Stake-based protocols are both more susceptible to private double-spending attacks than Proof-of-Work-based protocols; in this case, an adversary mines its chain in secret without sharing its blocks with the rest of the processes until the end of the attack.   In this paper, we formally characterize the properties of resources through an abstraction called resource allocator and give a framework for understanding longest-chain consensus protocols based on different underlying resources. In addition, we use this resource allocator to demonstrate security trade-offs between various resources focusing on well-known attacks (e.g., the long-range attack and nothing-at-stake attacks).

摘要: 区块链协议在未经许可的设置下实现全序广播，进程可以自由加入和离开。在这种情况下，为了防止Sybil攻击，正确的流程依赖于绑定到特定类型资源的加密证明，以使它们有资格订购交易。例如，在工作证明(PoW)的情况下，该资源是计算，而证明是计算困难难题的解决方案。相反，在赌注证明(POS)中，资源对应于系统中每个进程拥有的硬币数量，安全彩票根据其硬币持有量比例选择参与的进程。虽然许多基于资源的区块链协议在文献中被正式证明是安全的，但现有的安全证明无法证明为什么特定类型的资源会导致区块链协议容易受到不同类别的攻击。例如，POS系统比工作证明和存储证明系统更容易受到远程攻击，在远程攻击中，对手会破坏过去的流程以重写历史。基于存储证明和基于风险证明的协议都比基于工作证明的协议更容易受到私人双重支出攻击；在这种情况下，对手秘密挖掘其链，而不与其余进程共享其块，直到攻击结束。在本文中，我们通过一个称为资源分配器的抽象来形式化地刻画资源的属性，并给出了一个基于不同底层资源的理解最长链一致性协议的框架。此外，我们使用此资源分配器来演示专注于众所周知的攻击(例如，远程攻击和无风险攻击)的各种资源之间的安全权衡。



## **17. QueryNet: Attack by Multi-Identity Surrogates**

QueryNet：多身份代理的攻击 cs.LG

QueryNet reduces queries by about an order of magnitude against SOTA  black-box attacks

**SubmitDate**: 2022-11-22    [abs](http://arxiv.org/abs/2105.15010v4) [paper-pdf](http://arxiv.org/pdf/2105.15010v4)

**Authors**: Sizhe Chen, Zhehao Huang, Qinghua Tao, Xiaolin Huang

**Abstract**: Deep Neural Networks (DNNs) are acknowledged as vulnerable to adversarial attacks, while the existing black-box attacks require extensive queries on the victim DNN to achieve high success rates. For query-efficiency, surrogate models of the victim are used to generate transferable Adversarial Examples (AEs) because of their Gradient Similarity (GS), i.e., surrogates' attack gradients are similar to the victim's ones. However, it is generally neglected to exploit their similarity on outputs, namely the Prediction Similarity (PS), to filter out inefficient queries by surrogates without querying the victim. To jointly utilize and also optimize surrogates' GS and PS, we develop QueryNet, a unified attack framework that can significantly reduce queries. QueryNet creatively attacks by multi-identity surrogates, i.e., crafts several AEs for one sample by different surrogates, and also uses surrogates to decide on the most promising AE for the query. After that, the victim's query feedback is accumulated to optimize not only surrogates' parameters but also their architectures, enhancing both the GS and the PS. Although QueryNet has no access to pre-trained surrogates' prior, it reduces queries by averagely about an order of magnitude compared to alternatives within an acceptable time, according to our comprehensive experiments: 11 victims (including two commercial models) on MNIST/CIFAR10/ImageNet, allowing only 8-bit image queries, and no access to the victim's training data. The code is available at https://github.com/Sizhe-Chen/QueryNet.

摘要: 深度神经网络(DNN)被认为容易受到对抗性攻击，而现有的黑盒攻击需要对受害者DNN进行广泛的查询才能获得高的成功率。为了提高查询效率，受害者的代理模型被用来生成可转移的对抗实例，因为它们具有梯度相似性，即代理的攻击梯度与受害者的攻击梯度相似。然而，通常忽略了利用它们在输出上的相似性，即预测相似度(PS)来过滤代理在不查询受害者的情况下的低效查询。为了联合利用并优化代理的GS和PS，我们开发了QueryNet，这是一个可以显著减少查询的统一攻击框架。QueryNet创造性地利用多身份代理进行攻击，即通过不同的代理为一个样本构造多个代理实体，并使用代理为查询选择最有希望的代理实体。之后，受害者的查询反馈被累积，不仅优化了代理的参数，还优化了它们的体系结构，提高了GS和PS。虽然QueryNet无法访问预先训练的代理人的先前，但根据我们的综合实验：11名受害者(包括两个商业模型)在MNIST/CIFAR10/ImageNet上仅允许8位图像查询，并且无法访问受害者的训练数据，与替代方案相比，它在可接受的时间内平均减少了一个数量级的查询。代码可在https://github.com/Sizhe-Chen/QueryNet.上获得



## **18. Addressing Mistake Severity in Neural Networks with Semantic Knowledge**

利用语义知识解决神经网络中的错误严重性问题 cs.LG

**SubmitDate**: 2022-11-21    [abs](http://arxiv.org/abs/2211.11880v1) [paper-pdf](http://arxiv.org/pdf/2211.11880v1)

**Authors**: Natalie Abreu, Nathan Vaska, Victoria Helus

**Abstract**: Robustness in deep neural networks and machine learning algorithms in general is an open research challenge. In particular, it is difficult to ensure algorithmic performance is maintained on out-of-distribution inputs or anomalous instances that cannot be anticipated at training time. Embodied agents will be deployed in these conditions, and are likely to make incorrect predictions. An agent will be viewed as untrustworthy unless it can maintain its performance in dynamic environments. Most robust training techniques aim to improve model accuracy on perturbed inputs; as an alternate form of robustness, we aim to reduce the severity of mistakes made by neural networks in challenging conditions. We leverage current adversarial training methods to generate targeted adversarial attacks during the training process in order to increase the semantic similarity between a model's predictions and true labels of misclassified instances. Results demonstrate that our approach performs better with respect to mistake severity compared to standard and adversarially trained models. We also find an intriguing role that non-robust features play with regards to semantic similarity.

摘要: 一般情况下，深度神经网络和机器学习算法的稳健性是一个开放的研究挑战。特别是，很难确保在非分布输入或在训练时无法预测的异常情况下保持算法性能。具体化代理将在这些条件下部署，并可能做出错误的预测。除非一个代理能够在动态环境中保持其性能，否则它将被视为不可信任。大多数稳健的训练技术旨在提高扰动输入的模型精度；作为稳健性的另一种形式，我们的目标是降低神经网络在具有挑战性的条件下所犯错误的严重性。我们利用现有的对抗性训练方法在训练过程中产生有针对性的对抗性攻击，以增加模型预测和错误分类实例的真实标签之间的语义相似度。结果表明，与标准模型和反向训练模型相比，我们的方法在错误严重性方面表现得更好。我们还发现，非健壮特征在语义相似性方面扮演了一个有趣的角色。



## **19. Voice Spoofing Countermeasures: Taxonomy, State-of-the-art, experimental analysis of generalizability, open challenges, and the way forward**

语音欺骗对策：分类、最新技术、可概括性实验分析、开放挑战和前进方向 eess.AS

**SubmitDate**: 2022-11-21    [abs](http://arxiv.org/abs/2210.00417v2) [paper-pdf](http://arxiv.org/pdf/2210.00417v2)

**Authors**: Awais Khan, Khalid Mahmood Malik, James Ryan, Mikul Saravanan

**Abstract**: Malicious actors may seek to use different voice-spoofing attacks to fool ASV systems and even use them for spreading misinformation. Various countermeasures have been proposed to detect these spoofing attacks. Due to the extensive work done on spoofing detection in automated speaker verification (ASV) systems in the last 6-7 years, there is a need to classify the research and perform qualitative and quantitative comparisons on state-of-the-art countermeasures. Additionally, no existing survey paper has reviewed integrated solutions to voice spoofing evaluation and speaker verification, adversarial/antiforensics attacks on spoofing countermeasures, and ASV itself, or unified solutions to detect multiple attacks using a single model. Further, no work has been done to provide an apples-to-apples comparison of published countermeasures in order to assess their generalizability by evaluating them across corpora. In this work, we conduct a review of the literature on spoofing detection using hand-crafted features, deep learning, end-to-end, and universal spoofing countermeasure solutions to detect speech synthesis (SS), voice conversion (VC), and replay attacks. Additionally, we also review integrated solutions to voice spoofing evaluation and speaker verification, adversarial and anti-forensics attacks on voice countermeasures, and ASV. The limitations and challenges of the existing spoofing countermeasures are also presented. We report the performance of these countermeasures on several datasets and evaluate them across corpora. For the experiments, we employ the ASVspoof2019 and VSDC datasets along with GMM, SVM, CNN, and CNN-GRU classifiers. (For reproduceability of the results, the code of the test bed can be found in our GitHub Repository.

摘要: 恶意攻击者可能会试图使用不同的语音欺骗攻击来欺骗ASV系统，甚至利用它们来传播错误信息。已经提出了各种对策来检测这些欺骗攻击。由于过去6-7年在自动说话人验证(ASV)系统中对欺骗检测所做的大量工作，有必要对这些研究进行分类，并对最新的对策进行定性和定量的比较。此外，现有的调查论文没有审查语音欺骗评估和说话人验证的集成解决方案、对欺骗对策的对抗性/反取证攻击，以及ASV本身，或者使用单一模型检测多个攻击的统一解决方案。此外，还没有做任何工作来对已发表的对策进行逐一比较，以便通过在语料库中对其进行评估来评估其普遍性。在这项工作中，我们对使用手工特征、深度学习、端到端和通用欺骗对策解决方案来检测语音合成(SS)、语音转换(VC)和重放攻击的欺骗检测的文献进行了回顾。此外，我们还回顾了语音欺骗评估和说话人验证、针对语音对策的对抗性和反取证攻击以及ASV的集成解决方案。文中还指出了现有欺骗对策的局限性和挑战。我们报告了这些对策在几个数据集上的性能，并在语料库中对它们进行了评估。在实验中，我们使用了ASVspoof2019和VSDC数据集，以及GMM、SVM、CNN和CNN-GRU分类器。(对于结果的重现性，可以在我们的GitHub存储库中找到试验台的代码。



## **20. Backdoor Attacks on Multiagent Collaborative Systems**

对多智能体协作系统的后门攻击 cs.MA

11 pages

**SubmitDate**: 2022-11-21    [abs](http://arxiv.org/abs/2211.11455v1) [paper-pdf](http://arxiv.org/pdf/2211.11455v1)

**Authors**: Shuo Chen, Yue Qiu, Jie Zhang

**Abstract**: Backdoor attacks on reinforcement learning implant a backdoor in a victim agent's policy. Once the victim observes the trigger signal, it will switch to the abnormal mode and fail its task. Most of the attacks assume the adversary can arbitrarily modify the victim's observations, which may not be practical. One work proposes to let one adversary agent use its actions to affect its opponent in two-agent competitive games, so that the opponent quickly fails after observing certain trigger actions. However, in multiagent collaborative systems, agents may not always be able to observe others. When and how much the adversary agent can affect others are uncertain, and we want the adversary agent to trigger others for as few times as possible. To solve this problem, we first design a novel training framework to produce auxiliary rewards that measure the extent to which the other agents'observations being affected. Then we use the auxiliary rewards to train a trigger policy which enables the adversary agent to efficiently affect the others' observations. Given these affected observations, we further train the other agents to perform abnormally. Extensive experiments demonstrate that the proposed method enables the adversary agent to lure the others into the abnormal mode with only a few actions.

摘要: 对强化学习的后门攻击在受害者代理的策略中植入了一个后门。一旦受害者观察到触发信号，它就会切换到异常模式，任务失败。大多数攻击都假设对手可以任意修改受害者的观察结果，这可能是不现实的。一项工作提出在两个智能体的竞争博弈中，让一个对手智能体用自己的动作来影响对手，使对手在观察到某些触发动作后迅速失败。然而，在多智能体协作系统中，智能体可能并不总是能够观察到其他人。敌方代理何时以及在多大程度上可以影响他人是不确定的，我们希望敌方代理尽可能少地触发他人。为了解决这个问题，我们首先设计了一个新的训练框架来产生辅助奖励，以衡量其他代理的观察受到影响的程度。然后，我们使用辅助奖励来训练触发策略，使对手代理能够有效地影响他人的观察。鉴于这些受影响的观察结果，我们进一步训练其他代理执行异常操作。大量实验表明，该方法能够使敌方代理只需少量动作就能引诱其他代理进入异常模式。



## **21. SPIN: Simulated Poisoning and Inversion Network for Federated Learning-Based 6G Vehicular Networks**

SPIN：基于联邦学习的6G车载网络模拟投毒反转网络 cs.LG

6 pages, 4 figures

**SubmitDate**: 2022-11-21    [abs](http://arxiv.org/abs/2211.11321v1) [paper-pdf](http://arxiv.org/pdf/2211.11321v1)

**Authors**: Sunder Ali Khowaja, Parus Khuwaja, Kapal Dev, Angelos Antonopoulos

**Abstract**: The applications concerning vehicular networks benefit from the vision of beyond 5G and 6G technologies such as ultra-dense network topologies, low latency, and high data rates. Vehicular networks have always faced data privacy preservation concerns, which lead to the advent of distributed learning techniques such as federated learning. Although federated learning has solved data privacy preservation issues to some extent, the technique is quite vulnerable to model inversion and model poisoning attacks. We assume that the design of defense mechanism and attacks are two sides of the same coin. Designing a method to reduce vulnerability requires the attack to be effective and challenging with real-world implications. In this work, we propose simulated poisoning and inversion network (SPIN) that leverages the optimization approach for reconstructing data from a differential model trained by a vehicular node and intercepted when transmitted to roadside unit (RSU). We then train a generative adversarial network (GAN) to improve the generation of data with each passing round and global update from the RSU, accordingly. Evaluation results show the qualitative and quantitative effectiveness of the proposed approach. The attack initiated by SPIN can reduce up to 22% accuracy on publicly available datasets while just using a single attacker. We assume that revealing the simulation of such attacks would help us find its defense mechanism in an effective manner.

摘要: 车载网络的应用得益于超密集网络拓扑、低延迟和高数据速率等Beyond 5G和6G技术的愿景。车载网络一直面临着数据隐私保护的问题，这导致了联邦学习等分布式学习技术的出现。尽管联邦学习在一定程度上解决了数据隐私保护问题，但该技术很容易受到模型反转和模型中毒攻击。我们假设防御机制的设计和攻击是一枚硬币的两面。设计一种降低脆弱性的方法要求攻击有效并具有挑战性，并具有现实世界的影响。在这项工作中，我们提出了模拟毒化和反转网络(SPIN)，它利用优化方法从车辆节点训练的差异模型中重建数据，并在传输到路边单元(RSU)时截获数据。然后，我们训练一个生成性对抗网络(GAN)，以相应地改进每一轮数据的生成和来自RSU的全局更新。评价结果表明了该方法的定性和定量有效性。SPIN发起的攻击在仅使用单个攻击者的情况下，可以在公开可用的数据集上降低高达22%的准确率。我们认为，揭示此类攻击的模拟将有助于我们有效地发现其防御机制。



## **22. Understanding the Vulnerability of Skeleton-based Human Activity Recognition via Black-box Attack**

通过黑盒攻击理解基于骨架的人类活动识别的脆弱性 cs.CV

arXiv admin note: substantial text overlap with arXiv:2103.05266

**SubmitDate**: 2022-11-21    [abs](http://arxiv.org/abs/2211.11312v1) [paper-pdf](http://arxiv.org/pdf/2211.11312v1)

**Authors**: Yunfeng Diao, He Wang, Tianjia Shao, Yong-Liang Yang, Kun Zhou, David Hogg

**Abstract**: Human Activity Recognition (HAR) has been employed in a wide range of applications, e.g. self-driving cars, where safety and lives are at stake. Recently, the robustness of existing skeleton-based HAR methods has been questioned due to their vulnerability to adversarial attacks, which causes concerns considering the scale of the implication. However, the proposed attacks require the full-knowledge of the attacked classifier, which is overly restrictive. In this paper, we show such threats indeed exist, even when the attacker only has access to the input/output of the model. To this end, we propose the very first black-box adversarial attack approach in skeleton-based HAR called BASAR. BASAR explores the interplay between the classification boundary and the natural motion manifold. To our best knowledge, this is the first time data manifold is introduced in adversarial attacks on time series. Via BASAR, we find on-manifold adversarial samples are extremely deceitful and rather common in skeletal motions, in contrast to the common belief that adversarial samples only exist off-manifold. Through exhaustive evaluation, we show that BASAR can deliver successful attacks across classifiers, datasets, and attack modes. By attack, BASAR helps identify the potential causes of the model vulnerability and provides insights on possible improvements. Finally, to mitigate the newly identified threat, we propose a new adversarial training approach by leveraging the sophisticated distributions of on/off-manifold adversarial samples, called mixed manifold-based adversarial training (MMAT). MMAT can successfully help defend against adversarial attacks without compromising classification accuracy.

摘要: 人类活动识别(HAR)已被广泛应用于安全和生命受到威胁的自动驾驶汽车等领域。最近，现有的基于骨架的HAR方法的健壮性受到了质疑，因为它们对对手攻击的脆弱性，考虑到其蕴含的规模，这引起了人们的担忧。然而，所提出的攻击需要被攻击分类器的完全知识，这是过度限制的。在这篇文章中，我们证明了这样的威胁确实存在，即使攻击者只有权访问模型的输入/输出。为此，我们在基于骨架的HAR中提出了第一种黑盒对抗攻击方法BASAR。巴萨探索了分类边界和自然运动流形之间的相互作用。据我们所知，这是首次将数据流形引入时间序列的对抗性攻击中。通过BASAR，我们发现流形上的对抗性样本具有极大的欺骗性，并且在骨骼运动中相当常见，而不是通常认为对抗性样本只存在于流形外。通过详尽的评估，我们证明了Basar可以跨分类器、数据集和攻击模式进行成功的攻击。通过攻击，Basar帮助识别模型漏洞的潜在原因，并提供可能改进的见解。最后，为了缓解新识别的威胁，我们提出了一种新的对抗训练方法，即基于混合流形的对抗训练(MMAT)。MMAT可以在不影响分类准确性的情况下成功地帮助防御对手攻击。



## **23. Boosting the Transferability of Adversarial Attacks with Global Momentum Initialization**

利用全局动量初始化提高对抗性攻击的可转移性 cs.CV

**SubmitDate**: 2022-11-21    [abs](http://arxiv.org/abs/2211.11236v1) [paper-pdf](http://arxiv.org/pdf/2211.11236v1)

**Authors**: Jiafeng Wang, Zhaoyu Chen, Kaixun Jiang, Dingkang Yang, Lingyi Hong, Yan Wang, Wenqiang Zhang

**Abstract**: Deep neural networks are vulnerable to adversarial examples, which attach human invisible perturbations to benign inputs. Simultaneously, adversarial examples exhibit transferability under different models, which makes practical black-box attacks feasible. However, existing methods are still incapable of achieving desired transfer attack performance. In this work, from the perspective of gradient optimization and consistency, we analyze and discover the gradient elimination phenomenon as well as the local momentum optimum dilemma. To tackle these issues, we propose Global Momentum Initialization (GI) to suppress gradient elimination and help search for the global optimum. Specifically, we perform gradient pre-convergence before the attack and carry out a global search during the pre-convergence stage. Our method can be easily combined with almost all existing transfer methods, and we improve the success rate of transfer attacks significantly by an average of 6.4% under various advanced defense mechanisms compared to state-of-the-art methods. Eventually, we achieve an attack success rate of 95.4%, fully illustrating the insecurity of existing defense mechanisms.

摘要: 深层神经网络很容易受到敌意例子的影响，这些例子将人类看不见的扰动附加到良性输入上。同时，对抗性例子在不同的模型下表现出可转移性，这使得实际的黑盒攻击是可行的。然而，现有的方法仍然不能达到期望的传输攻击性能。在这项工作中，我们从梯度优化和一致性的角度，分析和发现了梯度消除现象以及局部动量最优困境。为了解决这些问题，我们提出了全局动量初始化(GI)来抑制梯度消除，并帮助搜索全局最优解。具体地说，我们在攻击前进行梯度预收敛，并在预收敛阶段进行全局搜索。我们的方法可以很容易地与几乎所有现有的传输方法相结合，在各种先进的防御机制下，与最先进的方法相比，我们的传输攻击成功率平均提高了6.4%。最终达到了95.4%的攻击成功率，充分说明了现有防御机制的不安全性。



## **24. Deep Composite Face Image Attacks: Generation, Vulnerability and Detection**

深度复合人脸图像攻击：产生、漏洞和检测 cs.CV

**SubmitDate**: 2022-11-20    [abs](http://arxiv.org/abs/2211.11039v1) [paper-pdf](http://arxiv.org/pdf/2211.11039v1)

**Authors**: Jag Mohan Singh, Raghavendra Ramachandra

**Abstract**: Face manipulation attacks have drawn the attention of biometric researchers because of their vulnerability to Face Recognition Systems (FRS). This paper proposes a novel scheme to generate Composite Face Image Attacks (CFIA) based on the Generative Adversarial Networks (GANs). Given the face images from contributory data subjects, the proposed CFIA method will independently generate the segmented facial attributes, then blend them using transparent masks to generate the CFIA samples. { The primary motivation for CFIA is to utilize deep learning to generate facial attribute-based composite attacks, which has been explored relatively less in the current literature.} We generate $14$ different combinations of facial attributes resulting in $14$ unique CFIA samples for each pair of contributory data subjects. Extensive experiments are carried out on our newly generated CFIA dataset consisting of 1000 unique identities with 2000 bona fide samples and 14000 CFIA samples, thus resulting in an overall 16000 face image samples. We perform a sequence of experiments to benchmark the vulnerability of CFIA to automatic FRS (based on both deep-learning and commercial-off-the-shelf (COTS). We introduced a new metric named Generalized Morphing Attack Potential (GMAP) to benchmark the vulnerability effectively. Additional experiments are performed to compute the perceptual quality of the generated CFIA samples. Finally, the CFIA detection performance is presented using three different Face Morphing Attack Detection (MAD) algorithms. The proposed CFIA method indicates good perceptual quality based on the obtained results. Further, { FRS is vulnerable to CFIA} (much higher than SOTA), making it difficult to detect by human observers and automatic detection algorithms. Lastly, we performed experiments to detect the CFIA samples using three different detection techniques automatically.

摘要: 人脸操纵攻击因其易受人脸识别系统(FRS)攻击而受到生物特征识别研究人员的关注。提出了一种新的基于生成性对抗网络(GANS)的合成人脸图像攻击生成方案。给定有贡献的数据对象的人脸图像，所提出的CFIA方法将独立地生成分割的人脸属性，然后使用透明掩膜对它们进行混合以生成CFIA样本。{CFIA的主要动机是利用深度学习来生成基于面部属性的复合攻击，这在当前的文献中相对较少探索。}我们生成$14$不同的面部属性组合，从而为每对有贡献的数据对象生成$14$唯一的CFIA样本。在我们新生成的包含1,000个唯一身份的CFIA数据集上进行了大量的实验，其中包含2,000个真实样本和14000个CFIA样本，从而得到总共16000个人脸图像样本。我们进行了一系列实验，以基准CFIA对自动FRS(基于深度学习和商业现成(COTS))的脆弱性。我们引入了一种名为广义变形攻击潜力(GMAP)的新度量来有效地对该漏洞进行基准测试。另外还进行了实验，以计算生成的CFIA样本的感知质量。最后，给出了三种不同人脸变形攻击检测(MAD)算法的CFIA检测性能。基于所获得的结果，所提出的CFIA方法显示出良好的感知质量。此外，{FRS易受CFIA攻击}(比SOTA高得多)，很难被人类观察者和自动检测算法检测到。最后，利用三种不同的检测技术对CFIA样本进行了自动检测实验。



## **25. Adversarial Cheap Talk**

对抗性的低级谈资 cs.LG

**SubmitDate**: 2022-11-20    [abs](http://arxiv.org/abs/2211.11030v1) [paper-pdf](http://arxiv.org/pdf/2211.11030v1)

**Authors**: Chris Lu, Timon Willi, Alistair Letcher, Jakob Foerster

**Abstract**: Adversarial attacks in reinforcement learning (RL) often assume highly-privileged access to the victim's parameters, environment, or data. Instead, this paper proposes a novel adversarial setting called a Cheap Talk MDP in which an Adversary can merely append deterministic messages to the Victim's observation, resulting in a minimal range of influence. The Adversary cannot occlude ground truth, influence underlying environment dynamics or reward signals, introduce non-stationarity, add stochasticity, see the Victim's actions, or access their parameters. Additionally, we present a simple meta-learning algorithm called Adversarial Cheap Talk (ACT) to train Adversaries in this setting. We demonstrate that an Adversary trained with ACT can still significantly influence the Victim's training and testing performance, despite the highly constrained setting. Affecting train-time performance reveals a new attack vector and provides insight into the success and failure modes of existing RL algorithms. More specifically, we show that an ACT Adversary is capable of harming performance by interfering with the learner's function approximation, or instead helping the Victim's performance by outputting useful features. Finally, we show that an ACT Adversary can manipulate messages during train-time to directly and arbitrarily control the Victim at test-time.

摘要: 强化学习(RL)中的对抗性攻击通常假定具有访问受害者参数、环境或数据的高度特权。相反，本文提出了一种新的对抗性环境，称为廉价谈话MDP，在该环境中，对手只需将确定性消息附加到受害者的观察中，从而产生最小的影响范围。敌手不能掩盖基本事实、影响潜在环境动态或奖励信号、引入非平稳性、增加随机性、看到受害者的行为或获取他们的参数。此外，我们还提出了一个简单的元学习算法，称为对抗性廉价谈话(ACT)，以在这种情况下训练对手。我们证明，尽管在高度受限的环境下，接受过ACT训练的对手仍然可以显著影响受害者的训练和测试表现。影响训练时间性能揭示了新的攻击向量，并提供了对现有RL算法的成功和失败模式的洞察。更具体地说，我们证明了ACT对手能够通过干扰学习者的函数逼近来损害性能，或者相反地通过输出有用的特征来帮助受害者的性能。最后，我们证明了ACT攻击者可以在训练时间内操纵消息，从而在测试时间直接任意控制受害者。



## **26. Towards Robust Neural Networks via Orthogonal Diversity**

基于正交分集的稳健神经网络研究 cs.CV

**SubmitDate**: 2022-11-20    [abs](http://arxiv.org/abs/2010.12190v4) [paper-pdf](http://arxiv.org/pdf/2010.12190v4)

**Authors**: Kun Fang, Qinghua Tao, Yingwen Wu, Tao Li, Jia Cai, Feipeng Cai, Xiaolin Huang, Jie Yang

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to invisible perturbations on the images generated by adversarial attacks, which raises researches on the adversarial robustness of DNNs. A series of methods represented by the adversarial training and its variants have proven as one of the most effective techniques in enhancing the DNN robustness. Generally, adversarial training focuses on enriching the training data by involving perturbed data. Despite of the efficiency in defending specific attacks, adversarial training is benefited from the data augmentation, which does not contribute to the robustness of DNN itself and usually suffers from accuracy drop on clean data as well as inefficiency in unknown attacks. Towards the robustness of DNN itself, we propose a novel defense that aims at augmenting the model in order to learn features adaptive to diverse inputs, including adversarial examples. Specifically, we introduce multiple paths to augment the network, and impose orthogonality constraints on these paths. In addition, a margin-maximization loss is designed to further boost DIversity via Orthogonality (DIO). Extensive empirical results on various data sets, architectures, and attacks demonstrate the adversarial robustness of the proposed DIO.

摘要: 深度神经网络(DNN)易受敌意攻击产生的图像不可见扰动的影响，这就引发了对DNN对抗鲁棒性的研究。以对抗性训练及其变体为代表的一系列方法已被证明是增强DNN鲁棒性的最有效技术之一。一般来说，对抗性训练的重点是通过使用扰动数据来丰富训练数据。尽管DNN在防御特定攻击方面效率很高，但对抗性训练得益于数据增强，这并不有助于DNN本身的健壮性，而且通常会导致对干净数据的准确率下降，以及对未知攻击的效率低下。针对DNN本身的健壮性，我们提出了一种新的防御方法，旨在增强模型以学习适应不同输入的特征，包括对抗性例子。具体地说，我们引入多条路径来增强网络，并对这些路径施加正交性约束。此外，利润率最大化损失旨在通过正交性(DIO)进一步提高多样性。在各种数据集、体系结构和攻击上的广泛实验结果证明了所提出的DIO的对抗性健壮性。



## **27. Invisible Backdoor Attack with Dynamic Triggers against Person Re-identification**

利用动态触发器对个人重新身份进行隐形后门攻击 cs.CV

**SubmitDate**: 2022-11-20    [abs](http://arxiv.org/abs/2211.10933v1) [paper-pdf](http://arxiv.org/pdf/2211.10933v1)

**Authors**: Wenli Sun, Xinyang Jiang, Shuguang Dou, Dongsheng Li, Duoqian Miao, Cheng Deng, Cairong Zhao

**Abstract**: In recent years, person Re-identification (ReID) has rapidly progressed with wide real-world applications, but also poses significant risks of adversarial attacks. In this paper, we focus on the backdoor attack on deep ReID models. Existing backdoor attack methods follow an all-to-one/all attack scenario, where all the target classes in the test set have already been seen in the training set. However, ReID is a much more complex fine-grained open-set recognition problem, where the identities in the test set are not contained in the training set. Thus, previous backdoor attack methods for classification are not applicable for ReID. To ameliorate this issue, we propose a novel backdoor attack on deep ReID under a new all-to-unknown scenario, called Dynamic Triggers Invisible Backdoor Attack (DT-IBA). Instead of learning fixed triggers for the target classes from the training set, DT-IBA can dynamically generate new triggers for any unknown identities. Specifically, an identity hashing network is proposed to first extract target identity information from a reference image, which is then injected into the benign images by image steganography. We extensively validate the effectiveness and stealthiness of the proposed attack on benchmark datasets, and evaluate the effectiveness of several defense methods against our attack.

摘要: 近年来，身份识别技术发展迅速，在实际应用中得到了广泛的应用，但同时也带来了巨大的对抗性攻击风险。本文主要研究对深度Reid模型的后门攻击。现有的后门攻击方法遵循All-to-One/All-to-One攻击场景，其中测试集中的所有目标类都已经出现在训练集中。然而，REID是一个更复杂的细粒度开集识别问题，其中测试集中的身份不包含在训练集中。因此，以前用于分类的后门攻击方法不适用于REID。为了改善这一问题，我们提出了一种新的全未知场景下对深度Reid的后门攻击，称为动态触发器不可见后门攻击(DT-IBA)。DT-IBA不需要从训练集中学习目标类的固定触发器，而是可以为任何未知身份动态生成新的触发器。具体地说，提出了一种身份散列网络，首先从参考图像中提取目标身份信息，然后通过图像隐写将这些身份信息注入到良性图像中。我们在基准数据集上广泛验证了提出的攻击的有效性和隐蔽性，并评估了几种防御方法对我们的攻击的有效性。



## **28. Spectral Adversarial Training for Robust Graph Neural Network**

稳健图神经网络的谱对抗训练 cs.LG

Accepted by TKDE. Code availiable at  https://github.com/EdisonLeeeee/SAT

**SubmitDate**: 2022-11-20    [abs](http://arxiv.org/abs/2211.10896v1) [paper-pdf](http://arxiv.org/pdf/2211.10896v1)

**Authors**: Jintang Li, Jiaying Peng, Liang Chen, Zibin Zheng, Tingting Liang, Qing Ling

**Abstract**: Recent studies demonstrate that Graph Neural Networks (GNNs) are vulnerable to slight but adversarially designed perturbations, known as adversarial examples. To address this issue, robust training methods against adversarial examples have received considerable attention in the literature. \emph{Adversarial Training (AT)} is a successful approach to learning a robust model using adversarially perturbed training samples. Existing AT methods on GNNs typically construct adversarial perturbations in terms of graph structures or node features. However, they are less effective and fraught with challenges on graph data due to the discreteness of graph structure and the relationships between connected examples. In this work, we seek to address these challenges and propose Spectral Adversarial Training (SAT), a simple yet effective adversarial training approach for GNNs. SAT first adopts a low-rank approximation of the graph structure based on spectral decomposition, and then constructs adversarial perturbations in the spectral domain rather than directly manipulating the original graph structure. To investigate its effectiveness, we employ SAT on three widely used GNNs. Experimental results on four public graph datasets demonstrate that SAT significantly improves the robustness of GNNs against adversarial attacks without sacrificing classification accuracy and training efficiency.

摘要: 最近的研究表明，图神经网络(GNN)容易受到轻微但相反设计的扰动的影响，这些扰动称为对抗性例子。为了解决这一问题，针对对抗性例子的稳健训练方法在文献中得到了相当大的关注。对抗训练(AT)是利用对抗扰动训练样本学习稳健模型的一种成功方法。现有的基于GNN的AT方法通常根据图的结构或节点特征来构造对抗性扰动。然而，由于图结构的离散性和连通实例之间的关系，它们对图数据的处理效率较低且充满挑战。在这项工作中，我们试图解决这些挑战，并提出频谱对抗训练(SAT)，一种简单但有效的GNN对抗训练方法。SAT首先采用基于谱分解的图结构的低阶近似，然后在谱域中构造对抗性扰动，而不是直接操纵原始的图结构。为了考察其有效性，我们在三个广泛使用的GNN上使用了SAT。在四个公共图数据集上的实验结果表明，SAT在不牺牲分类精度和训练效率的情况下，显著提高了GNN对敌意攻击的健壮性。



## **29. Towards Compositional Adversarial Robustness: Generalizing Adversarial Training to Composite Semantic Perturbations**

走向成分对抗稳健性：将对抗训练推广到复合语义扰动 cs.CV

**SubmitDate**: 2022-11-20    [abs](http://arxiv.org/abs/2202.04235v2) [paper-pdf](http://arxiv.org/pdf/2202.04235v2)

**Authors**: Lei Hsiung, Yun-Yun Tsai, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Model robustness against adversarial examples of single perturbation type such as the $\ell_{p}$-norm has been widely studied, yet its generalization to more realistic scenarios involving multiple semantic perturbations and their composition remains largely unexplored. In this paper, we first propose a novel method for generating composite adversarial examples. Our method can find the optimal attack composition by utilizing component-wise projected gradient descent and automatic attack-order scheduling. We then propose generalized adversarial training (GAT) to extend model robustness from $\ell_{p}$-ball to composite semantic perturbations, such as the combination of Hue, Saturation, Brightness, Contrast, and Rotation. Results obtained using ImageNet and CIFAR-10 datasets indicate that GAT can be robust not only to all the tested types of a single attack, but also to any combination of such attacks. GAT also outperforms baseline $\ell_{\infty}$-norm bounded adversarial training approaches by a significant margin.

摘要: 针对单一扰动类型的对抗性实例，如$ellp-范数，模型的稳健性已经得到了广泛的研究，但它对涉及多个语义扰动及其组成的更现实场景的推广仍在很大程度上有待探索。在本文中，我们首先提出了一种生成复合对抗性实例的新方法。该方法利用基于组件的投影梯度下降和自动攻击顺序调度来寻找最优的攻击组合。然后，我们提出了广义对抗性训练(GAT)来扩展模型的稳健性，将模型的稳健性从球状扩展到复合语义扰动，如色调、饱和度、亮度、对比度和旋转的组合。使用ImageNet和CIFAR-10数据集获得的结果表明，GAT不仅对所有测试类型的单一攻击，而且对此类攻击的任何组合都具有健壮性。GAT的性能也大大超过了基准范数有界的对抗性训练方法。



## **30. Let Graph be the Go Board: Gradient-free Node Injection Attack for Graph Neural Networks via Reinforcement Learning**

图为棋盘：基于强化学习的图神经网络无梯度节点注入攻击 cs.LG

AAAI 2023. arXiv admin note: substantial text overlap with  arXiv:2202.09389

**SubmitDate**: 2022-11-19    [abs](http://arxiv.org/abs/2211.10782v1) [paper-pdf](http://arxiv.org/pdf/2211.10782v1)

**Authors**: Mingxuan Ju, Yujie Fan, Chuxu Zhang, Yanfang Ye

**Abstract**: Graph Neural Networks (GNNs) have drawn significant attentions over the years and been broadly applied to essential applications requiring solid robustness or vigorous security standards, such as product recommendation and user behavior modeling. Under these scenarios, exploiting GNN's vulnerabilities and further downgrading its performance become extremely incentive for adversaries. Previous attackers mainly focus on structural perturbations or node injections to the existing graphs, guided by gradients from the surrogate models. Although they deliver promising results, several limitations still exist. For the structural perturbation attack, to launch a proposed attack, adversaries need to manipulate the existing graph topology, which is impractical in most circumstances. Whereas for the node injection attack, though being more practical, current approaches require training surrogate models to simulate a white-box setting, which results in significant performance downgrade when the surrogate architecture diverges from the actual victim model. To bridge these gaps, in this paper, we study the problem of black-box node injection attack, without training a potentially misleading surrogate model. Specifically, we model the node injection attack as a Markov decision process and propose Gradient-free Graph Advantage Actor Critic, namely G2A2C, a reinforcement learning framework in the fashion of advantage actor critic. By directly querying the victim model, G2A2C learns to inject highly malicious nodes with extremely limited attacking budgets, while maintaining a similar node feature distribution. Through our comprehensive experiments over eight acknowledged benchmark datasets with different characteristics, we demonstrate the superior performance of our proposed G2A2C over the existing state-of-the-art attackers. Source code is publicly available at: https://github.com/jumxglhf/G2A2C}.

摘要: 多年来，图神经网络(GNN)引起了人们的广泛关注，并被广泛应用于需要可靠的健壮性或严格的安全标准的重要应用，如产品推荐和用户行为建模。在这些场景下，利用GNN的漏洞并进一步降低其性能成为对手的极大诱因。以前的攻击者主要集中在结构扰动或对现有图的节点注入上，由代理模型的梯度引导。尽管它们带来了令人振奋的结果，但仍然存在一些限制。对于结构扰动攻击，要发起拟议的攻击，攻击者需要操纵现有的图拓扑，这在大多数情况下是不切实际的。而对于节点注入攻击，目前的方法虽然更加实用，但需要训练代理模型来模拟白盒设置，当代理体系结构偏离实际受害者模型时，这会导致性能显著下降。为了弥补这些差距，在本文中，我们研究了黑盒节点注入攻击问题，而不需要训练一个潜在的误导性代理模型。具体地说，我们将节点注入攻击建模为马尔可夫决策过程，提出了无梯度图Advantage Actor Critic，即G2A2C，一种基于Advantage Actor Critic的强化学习框架。通过直接查询受害者模型，G2A2C学习以极其有限的攻击预算注入高度恶意的节点，同时保持类似的节点特征分布。通过我们在八个不同特征的公认基准数据集上的综合实验，我们证明了我们提出的G2A2C比现有的最先进的攻击者具有更好的性能。源代码可在以下网址公开获得：https://github.com/jumxglhf/G2A2C}.



## **31. Robust Smart Home Face Recognition under Starving Federated Data**

联邦数据饥饿下的稳健智能家居人脸识别 cs.LG

11 pages, 12 figures, 7 tables, accepted as a conference paper at  IEEE UV 2022, Boston, USA

**SubmitDate**: 2022-11-19    [abs](http://arxiv.org/abs/2211.05410v2) [paper-pdf](http://arxiv.org/pdf/2211.05410v2)

**Authors**: Jaechul Roh, Yajun Fang

**Abstract**: Over the past few years, the field of adversarial attack received numerous attention from various researchers with the help of successful attack success rate against well-known deep neural networks that were acknowledged to achieve high classification ability in various tasks. However, majority of the experiments were completed under a single model, which we believe it may not be an ideal case in a real-life situation. In this paper, we introduce a novel federated adversarial training method for smart home face recognition, named FLATS, where we observed some interesting findings that may not be easily noticed in a traditional adversarial attack to federated learning experiments. By applying different variations to the hyperparameters, we have spotted that our method can make the global model to be robust given a starving federated environment. Our code can be found on https://github.com/jcroh0508/FLATS.

摘要: 在过去的几年里，借助对公认在各种任务中具有高分类能力的知名深度神经网络的攻击成功率，对抗性攻击领域受到了众多研究人员的关注。然而，大多数实验都是在单一模型下完成的，我们认为这在现实生活中可能不是理想的情况。本文介绍了一种新的用于智能家居人脸识别的联合对抗性训练方法--Flats，我们在该方法中观察到了一些有趣的发现，这些发现在传统的对抗性攻击联合学习实验中可能不容易被注意到。通过对超参数应用不同的变化，我们已经发现，我们的方法可以使全局模型在饥饿的联邦环境下具有健壮性。我们的代码可以在https://github.com/jcroh0508/FLATS.上找到



## **32. A privacy-preserving data storage and service framework based on deep learning and blockchain for construction workers' wearable IoT sensors**

基于深度学习和区块链的建筑工人可穿戴物联网传感器隐私保护数据存储与服务框架 cs.CR

**SubmitDate**: 2022-11-19    [abs](http://arxiv.org/abs/2211.10713v1) [paper-pdf](http://arxiv.org/pdf/2211.10713v1)

**Authors**: Xiaoshan Zhou, Pin-Chao Liao

**Abstract**: Classifying brain signals collected by wearable Internet of Things (IoT) sensors, especially brain-computer interfaces (BCIs), is one of the fastest-growing areas of research. However, research has mostly ignored the secure storage and privacy protection issues of collected personal neurophysiological data. Therefore, in this article, we try to bridge this gap and propose a secure privacy-preserving protocol for implementing BCI applications. We first transformed brain signals into images and used generative adversarial network to generate synthetic signals to protect data privacy. Subsequently, we applied the paradigm of transfer learning for signal classification. The proposed method was evaluated by a case study and results indicate that real electroencephalogram data augmented with artificially generated samples provide superior classification performance. In addition, we proposed a blockchain-based scheme and developed a prototype on Ethereum, which aims to make storing, querying and sharing personal neurophysiological data and analysis reports secure and privacy-aware. The rights of three main transaction bodies - construction workers, BCI service providers and project managers - are described and the advantages of the proposed system are discussed. We believe this paper provides a well-rounded solution to safeguard private data against cyber-attacks, level the playing field for BCI application developers, and to the end improve professional well-being in the industry.

摘要: 对可穿戴物联网(IoT)传感器收集的大脑信号进行分类，特别是脑机接口(BCI)，是增长最快的研究领域之一。然而，研究大多忽略了收集的个人神经生理数据的安全存储和隐私保护问题。因此，在本文中，我们试图弥合这一差距，并提出一种安全的隐私保护协议来实现脑机接口应用。我们首先将大脑信号转换为图像，并使用生成性对抗网络生成合成信号来保护数据隐私。随后，我们将迁移学习范式应用于信号分类。通过实例对该方法进行了评估，结果表明，在人工生成样本的基础上增加真实的脑电数据可以获得更好的分类效果。此外，我们提出了一种基于区块链的方案，并在Etherum上开发了一个原型，旨在实现个人神经生理数据和分析报告的安全和隐私感知的存储、查询和共享。描述了三个主要交易主体--建筑工人、BCI服务提供商和项目经理的权利，并讨论了拟议系统的优点。我们相信，这份白皮书提供了一个全面的解决方案，以保护私人数据免受网络攻击，为BCI应用程序开发人员提供公平的竞争环境，并最终提高行业的职业福祉。



## **33. A Survey on Differential Privacy with Machine Learning and Future Outlook**

基于机器学习的差分隐私研究综述及未来展望 cs.LG

12 pages, 3 figures

**SubmitDate**: 2022-11-19    [abs](http://arxiv.org/abs/2211.10708v1) [paper-pdf](http://arxiv.org/pdf/2211.10708v1)

**Authors**: Samah Baraheem, Zhongmei Yao

**Abstract**: Nowadays, machine learning models and applications have become increasingly pervasive. With this rapid increase in the development and employment of machine learning models, a concern regarding privacy has risen. Thus, there is a legitimate need to protect the data from leaking and from any attacks. One of the strongest and most prevalent privacy models that can be used to protect machine learning models from any attacks and vulnerabilities is differential privacy (DP). DP is strict and rigid definition of privacy, where it can guarantee that an adversary is not capable to reliably predict if a specific participant is included in the dataset or not. It works by injecting a noise to the data whether to the inputs, the outputs, the ground truth labels, the objective functions, or even to the gradients to alleviate the privacy issue and protect the data. To this end, this survey paper presents different differentially private machine learning algorithms categorized into two main categories (traditional machine learning models vs. deep learning models). Moreover, future research directions for differential privacy with machine learning algorithms are outlined.

摘要: 如今，机器学习模型和应用已经变得越来越普遍。随着机器学习模型的开发和应用的迅速增加，人们对隐私的担忧也上升了。因此，有必要保护数据不受泄露和任何攻击。差异隐私(DP)是可用于保护机器学习模型免受任何攻击和漏洞攻击的最强大和最流行的隐私模型之一。DP是对隐私的严格和严格的定义，它可以保证对手不能可靠地预测特定参与者是否包括在数据集中。它的工作原理是向数据注入噪声，无论是输入、输出、基本事实标签、目标函数，甚至是梯度，以缓解隐私问题并保护数据。为此，本文提出了两类不同的不同的私有机器学习算法(传统机器学习模型和深度学习模型)。此外，还展望了利用机器学习算法研究差分隐私的未来发展方向。



## **34. Phonemic Adversarial Attack against Audio Recognition in Real World**

现实世界中针对音频识别的音素敌意攻击 cs.SD

**SubmitDate**: 2022-11-19    [abs](http://arxiv.org/abs/2211.10661v1) [paper-pdf](http://arxiv.org/pdf/2211.10661v1)

**Authors**: Jiakai Wang, Zhendong Chen, Zixin Yin, Qinghong Yang, Xianglong Liu

**Abstract**: Recently, adversarial attacks for audio recognition have attracted much attention. However, most of the existing studies mainly rely on the coarse-grain audio features at the instance level to generate adversarial noises, which leads to expensive generation time costs and weak universal attacking ability. Motivated by the observations that all audio speech consists of fundamental phonemes, this paper proposes a phonemic adversarial tack (PAT) paradigm, which attacks the fine-grain audio features at the phoneme level commonly shared across audio instances, to generate phonemic adversarial noises, enjoying the more general attacking ability with fast generation speed. Specifically, for accelerating the generation, a phoneme density balanced sampling strategy is introduced to sample quantity less but phonemic features abundant audio instances as the training data via estimating the phoneme density, which substantially alleviates the heavy dependency on the large training dataset. Moreover, for promoting universal attacking ability, the phonemic noise is optimized in an asynchronous way with a sliding window, which enhances the phoneme diversity and thus well captures the critical fundamental phonemic patterns. By conducting extensive experiments, we comprehensively investigate the proposed PAT framework and demonstrate that it outperforms the SOTA baselines by large margins (i.e., at least 11X speed up and 78% attacking ability improvement).

摘要: 近年来，针对音频识别的敌意攻击引起了人们的广泛关注。然而，现有的研究大多依赖于实例级的粗粒度音频特征来生成对抗性噪声，导致生成时间开销较大，通用攻击能力较弱。基于所有音频语音都是由基本音素组成的这一观察结果，提出了一种音素对抗Tack(PAT)范式，它在音频实例之间共享的音素级别攻击细粒度音频特征，以生成音素对抗噪声，具有更普遍的攻击能力和快速的生成速度。具体地说，为了加快生成速度，引入了音素密度均衡采样策略，通过估计音素密度，将采样量较少但音素特征丰富的音频实例作为训练数据，大大缓解了对大训练数据集的严重依赖。此外，为了提高通用攻击能力，采用滑动窗口对音素噪声进行了异步优化，增强了音素多样性，很好地捕捉到了关键的基本音素模式。通过大量的实验，我们对所提出的PAT框架进行了全面的研究，并证明了它的性能大大超过了SOTA基线(即，速度至少提高了11倍，攻击能力提高了78%)。



## **35. Scale-free and Task-agnostic Attack: Generating Photo-realistic Adversarial Patterns with Patch Quilting Generator**

无标度和任务不可知攻击：使用补丁缝合生成器生成照片级真实感对抗性图案 cs.CV

**SubmitDate**: 2022-11-19    [abs](http://arxiv.org/abs/2208.06222v2) [paper-pdf](http://arxiv.org/pdf/2208.06222v2)

**Authors**: Xiangbo Gao, Cheng Luo, Qinliang Lin, Weicheng Xie, Minmin Liu, Linlin Shen, Keerthy Kusumam, Siyang Song

**Abstract**: \noindent Traditional L_p norm-restricted image attack algorithms suffer from poor transferability to black box scenarios and poor robustness to defense algorithms. Recent CNN generator-based attack approaches can synthesize unrestricted and semantically meaningful entities to the image, which is shown to be transferable and robust. However, such methods attack images by either synthesizing local adversarial entities, which are only suitable for attacking specific contents or performing global attacks, which are only applicable to a specific image scale. In this paper, we propose a novel Patch Quilting Generative Adversarial Networks (PQ-GAN) to learn the first scale-free CNN generator that can be applied to attack images with arbitrary scales for various computer vision tasks. The principal investigation on transferability of the generated adversarial examples, robustness to defense frameworks, and visual quality assessment show that the proposed PQG-based attack framework outperforms the other nine state-of-the-art adversarial attack approaches when attacking the neural networks trained on two standard evaluation datasets (i.e., ImageNet and CityScapes).

摘要: 传统的L_p范数受限图像攻击算法对黑盒场景的可移植性差，对防御算法的健壮性差。最近的基于CNN生成器的攻击方法可以对图像合成不受限制的、具有语义意义的实体，被证明是可传输的和健壮的。然而，这些方法要么通过合成仅适用于攻击特定内容的局部敌对实体来攻击图像，要么通过执行全局攻击来攻击图像，而全局攻击仅适用于特定图像规模。本文提出了一种新的Patch Quilting生成对抗网络(PQ-GAN)来学习第一个无标度的CNN生成器，该生成器可以用于各种计算机视觉任务的任意规模的攻击图像。对生成的对抗性样本的可转移性、对防御框架的健壮性和视觉质量评估的主要研究表明，所提出的基于PQG的攻击框架在攻击基于两个标准评估数据集(即ImageNet和Citycapes)训练的神经网络时的性能优于其他9种最新的对抗性攻击方法。



## **36. Investigating the Security of EV Charging Mobile Applications As an Attack Surface**

以电动汽车充电移动应用为攻击面的安全性研究 cs.CR

**SubmitDate**: 2022-11-19    [abs](http://arxiv.org/abs/2211.10603v1) [paper-pdf](http://arxiv.org/pdf/2211.10603v1)

**Authors**: K. Sarieddine, M. A. Sayed, S. Torabi, R. Atallah, C. Assi

**Abstract**: The adoption rate of EVs has witnessed a significant increase in recent years driven by multiple factors, chief among which is the increased flexibility and ease of access to charging infrastructure. To improve user experience, increase system flexibility and commercialize the charging process, mobile applications have been incorporated into the EV charging ecosystem. EV charging mobile applications allow consumers to remotely trigger actions on charging stations and use functionalities such as start/stop charging sessions, pay for usage, and locate charging stations, to name a few. In this paper, we study the security posture of the EV charging ecosystem against remote attacks, which exploit the insecurity of the EV charging mobile applications as an attack surface. We leverage a combination of static and dynamic analysis techniques to analyze the security of widely used EV charging mobile applications. Our analysis of 31 widely used mobile applications and their interactions with various components such as the cloud management systems indicate the lack of user/vehicle verification and improper authorization for critical functions, which lead to remote (dis)charging session hijacking and Denial of Service (DoS) attacks against the EV charging station. Indeed, we discuss specific remote attack scenarios and their impact on the EV users. More importantly, our analysis results demonstrate the feasibility of leveraging existing vulnerabilities across various EV charging mobile applications to perform wide-scale coordinated remote charging/discharging attacks against the connected critical infrastructure (e.g., power grid), with significant undesired economical and operational implications. Finally, we propose counter measures to secure the infrastructure and impede adversaries from performing reconnaissance and launching remote attacks using compromised accounts.

摘要: 近年来，在多种因素的推动下，电动汽车的采用率大幅上升，其中最主要的是充电基础设施的灵活性和便利性的提高。为了改善用户体验、增加系统灵活性并将充电过程商业化，移动应用程序已被纳入电动汽车充电生态系统。电动汽车充电移动应用程序允许消费者远程触发充电站上的操作，并使用诸如开始/停止充电会话、付费使用和定位充电站等功能。本文研究了电动汽车充电生态系统对远程攻击的安全态势，利用电动汽车充电移动应用的不安全性作为攻击面。我们利用静态和动态分析技术的组合来分析广泛使用的电动汽车充电移动应用程序的安全性。我们对31个广泛使用的移动应用程序及其与云管理系统等各种组件的交互分析表明，缺乏用户/车辆验证和关键功能授权不当，导致对电动汽车充电站的远程(DIS)充电会话劫持和拒绝服务(DoS)攻击。事实上，我们讨论了特定的远程攻击场景及其对电动汽车用户的影响。更重要的是，我们的分析结果证明了利用各种电动汽车充电移动应用程序中的现有漏洞对连接的关键基础设施(如电网)执行大规模协同远程充放电攻击的可行性，这将带来严重的不良经济和运营影响。最后，我们提出了保护基础设施并阻止对手使用受攻击帐户执行侦察和发起远程攻击的对策。



## **37. Person Text-Image Matching via Text-Feature Interpretability Embedding and External Attack Node Implantation**

基于文本特征可解释性嵌入和外部攻击节点植入的人文本图像匹配 cs.CV

**SubmitDate**: 2022-11-19    [abs](http://arxiv.org/abs/2211.08657v2) [paper-pdf](http://arxiv.org/pdf/2211.08657v2)

**Authors**: Fan Li, Hang Zhou, Huafeng Li, Yafei Zhang, Zhengtao Yu

**Abstract**: Person text-image matching, also known as text based person search, aims to retrieve images of specific pedestrians using text descriptions. Although person text-image matching has made great research progress, existing methods still face two challenges. First, the lack of interpretability of text features makes it challenging to effectively align them with their corresponding image features. Second, the same pedestrian image often corresponds to multiple different text descriptions, and a single text description can correspond to multiple different images of the same identity. The diversity of text descriptions and images makes it difficult for a network to extract robust features that match the two modalities. To address these problems, we propose a person text-image matching method by embedding text-feature interpretability and an external attack node. Specifically, we improve the interpretability of text features by providing them with consistent semantic information with image features to achieve the alignment of text and describe image region features.To address the challenges posed by the diversity of text and the corresponding person images, we treat the variation caused by diversity to features as caused by perturbation information and propose a novel adversarial attack and defense method to solve it. In the model design, graph convolution is used as the basic framework for feature representation and the adversarial attacks caused by text and image diversity on feature extraction is simulated by implanting an additional attack node in the graph convolution layer to improve the robustness of the model against text and image diversity. Extensive experiments demonstrate the effectiveness and superiority of text-pedestrian image matching over existing methods. The source code of the method is published at

摘要: 人的文本-图像匹配，也称为基于文本的人搜索，旨在利用文本描述检索特定行人的图像。虽然人的文本-图像匹配已经取得了很大的研究进展，但现有的方法仍然面临着两个方面的挑战。首先，文本特征缺乏可解释性，这使得有效地将它们与其对应的图像特征对齐具有挑战性。其次，相同的行人图像往往对应于多个不同的文本描述，并且单个文本描述可以对应于相同身份的多个不同图像。文本描述和图像的多样性使得网络很难提取匹配这两种模式的稳健特征。针对这些问题，我们提出了一种嵌入文本特征可解释性和外部攻击节点的人文本图像匹配方法。具体而言，通过为文本特征提供与图像特征一致的语义信息来提高文本特征的可解释性，从而实现文本对齐和描述图像区域特征；针对文本和对应人物图像的多样性带来的挑战，我们将特征多样性引起的变异看作是扰动信息引起的，并提出了一种新的对抗性攻防方法。在模型设计中，使用图卷积作为特征表示的基本框架，通过在图卷积层中增加一个攻击节点来模拟文本和图像多样性对特征提取的敌意攻击，以提高模型对文本和图像多样性的鲁棒性。大量的实验证明了文本-行人图像匹配方法的有效性和优越性。该方法的源代码发布在



## **38. Adversarial Detection by Approximation of Ensemble Boundary**

基于集合边界逼近的对抗性检测 cs.LG

8 pages, 8 figures, 8 tables

**SubmitDate**: 2022-11-18    [abs](http://arxiv.org/abs/2211.10227v1) [paper-pdf](http://arxiv.org/pdf/2211.10227v1)

**Authors**: T. Windeatt

**Abstract**: A spectral approximation of a Boolean function is proposed for approximating the decision boundary of an ensemble of Deep Neural Networks (DNNs) solving two-class pattern recognition problems. The Walsh combination of relatively weak DNN classifiers is shown experimentally to be capable of detecting adversarial attacks. By observing the difference in Walsh coefficient approximation between clean and adversarial images, it appears that transferability of attack may be used for detection. Approximating the decision boundary may also aid in understanding the learning and transferability properties of DNNs. While the experiments here use images, the proposed approach of modelling two-class ensemble decision boundaries could in principle be applied to any application area.

摘要: 提出一种布尔函数的谱逼近方法，用于逼近求解两类模式识别问题的深度神经网络(DNN)集成的决策边界。实验表明，相对较弱的DNN分类器的Walsh组合能够检测到对抗性攻击。通过观察干净图像和敌意图像在沃尔什系数逼近上的差异，可以看出攻击的可转移性可以用于检测。近似决策边界也有助于理解DNN的学习和可转移性。虽然这里的实验使用的是图像，但所提出的建模两类集合决策边界的方法原则上可以应用于任何应用领域。



## **39. Adv-Attribute: Inconspicuous and Transferable Adversarial Attack on Face Recognition**

ADV-ATTRIBUTE：对人脸识别的隐蔽且可转移的敌意攻击 cs.CV

Accepted by NeurIPS2022

**SubmitDate**: 2022-11-18    [abs](http://arxiv.org/abs/2210.06871v2) [paper-pdf](http://arxiv.org/pdf/2210.06871v2)

**Authors**: Shuai Jia, Bangjie Yin, Taiping Yao, Shouhong Ding, Chunhua Shen, Xiaokang Yang, Chao Ma

**Abstract**: Deep learning models have shown their vulnerability when dealing with adversarial attacks. Existing attacks almost perform on low-level instances, such as pixels and super-pixels, and rarely exploit semantic clues. For face recognition attacks, existing methods typically generate the l_p-norm perturbations on pixels, however, resulting in low attack transferability and high vulnerability to denoising defense models. In this work, instead of performing perturbations on the low-level pixels, we propose to generate attacks through perturbing on the high-level semantics to improve attack transferability. Specifically, a unified flexible framework, Adversarial Attributes (Adv-Attribute), is designed to generate inconspicuous and transferable attacks on face recognition, which crafts the adversarial noise and adds it into different attributes based on the guidance of the difference in face recognition features from the target. Moreover, the importance-aware attribute selection and the multi-objective optimization strategy are introduced to further ensure the balance of stealthiness and attacking strength. Extensive experiments on the FFHQ and CelebA-HQ datasets show that the proposed Adv-Attribute method achieves the state-of-the-art attacking success rates while maintaining better visual effects against recent attack methods.

摘要: 深度学习模型在处理对抗性攻击时显示出了它们的脆弱性。现有的攻击几乎是在低层实例上执行的，例如像素和超像素，很少利用语义线索。对于人脸识别攻击，现有的方法通常会产生像素上的l_p范数扰动，导致攻击可传递性低，对去噪防御模型的脆弱性高。在这项工作中，我们不是对低层像素进行扰动，而是通过对高层语义的扰动来产生攻击，以提高攻击的可转移性。具体地说，设计了一个统一的灵活框架--对抗性属性(ADV-ATTRIBUTE)，用于产生对人脸识别的隐蔽性和可转移性攻击，该框架根据人脸识别特征与目标的差异指导生成对抗性噪声并将其添加到不同的属性中。此外，引入了重要性感知的属性选择和多目标优化策略，进一步保证了隐蔽性和攻击力的平衡。在FFHQ和CelebA-HQ数据集上的大量实验表明，所提出的ADV属性方法达到了最先进的攻击成功率，同时对最近的攻击方法保持了更好的视觉效果。



## **40. Leveraging Algorithmic Fairness to Mitigate Blackbox Attribute Inference Attacks**

利用算法公平性缓解黑盒属性推理攻击 cs.LG

arXiv admin note: text overlap with arXiv:2202.02242

**SubmitDate**: 2022-11-18    [abs](http://arxiv.org/abs/2211.10209v1) [paper-pdf](http://arxiv.org/pdf/2211.10209v1)

**Authors**: Jan Aalmoes, Vasisht Duddu, Antoine Boutet

**Abstract**: Machine learning (ML) models have been deployed for high-stakes applications, e.g., healthcare and criminal justice. Prior work has shown that ML models are vulnerable to attribute inference attacks where an adversary, with some background knowledge, trains an ML attack model to infer sensitive attributes by exploiting distinguishable model predictions. However, some prior attribute inference attacks have strong assumptions about adversary's background knowledge (e.g., marginal distribution of sensitive attribute) and pose no more privacy risk than statistical inference. Moreover, none of the prior attacks account for class imbalance of sensitive attribute in datasets coming from real-world applications (e.g., Race and Sex). In this paper, we propose an practical and effective attribute inference attack that accounts for this imbalance using an adaptive threshold over the attack model's predictions. We exhaustively evaluate our proposed attack on multiple datasets and show that the adaptive threshold over the model's predictions drastically improves the attack accuracy over prior work. Finally, current literature lacks an effective defence against attribute inference attacks. We investigate the impact of fairness constraints (i.e., designed to mitigate unfairness in model predictions) during model training on our attribute inference attack. We show that constraint based fairness algorithms which enforces equalized odds acts as an effective defense against attribute inference attacks without impacting the model utility. Hence, the objective of algorithmic fairness and sensitive attribute privacy are aligned.

摘要: 机器学习(ML)模型已被部署用于高风险应用，例如医疗保健和刑事司法。前人的工作表明，ML模型容易受到属性推理攻击，在这种攻击中，具有一定背景知识的对手训练ML攻击模型，通过利用可区分的模型预测来推断敏感属性。然而，一些先验属性推理攻击对对手的背景知识(如敏感属性的边际分布)有很强的假设，并不会比统计推理带来更多的隐私风险。此外，以前的攻击都没有考虑到来自真实世界应用程序(如种族和性别)的数据集中敏感属性的类别不平衡。在本文中，我们提出了一种实用而有效的属性推理攻击，它通过对攻击模型的预测使用自适应阈值来解释这种不平衡。我们在多个数据集上对我们提出的攻击进行了详尽的评估，结果表明，在模型预测之上的自适应阈值大大提高了攻击的准确性。最后，现有文献缺乏对属性推理攻击的有效防御。我们研究了模型训练过程中公平性约束(即，旨在缓解模型预测中的不公平性)对属性推理攻击的影响。我们证明了基于约束的公平算法可以在不影响模型效用的情况下有效地防御属性推理攻击。因此，算法公平性和敏感属性私密性的目标是一致的。



## **41. Cheating Automatic Short Answer Grading: On the Adversarial Usage of Adjectives and Adverbs**

作弊自动简答评分：形容词和副词的对抗性用法 cs.CL

**SubmitDate**: 2022-11-18    [abs](http://arxiv.org/abs/2201.08318v2) [paper-pdf](http://arxiv.org/pdf/2201.08318v2)

**Authors**: Anna Filighera, Sebastian Ochs, Tim Steuer, Thomas Tregel

**Abstract**: Automatic grading models are valued for the time and effort saved during the instruction of large student bodies. Especially with the increasing digitization of education and interest in large-scale standardized testing, the popularity of automatic grading has risen to the point where commercial solutions are widely available and used. However, for short answer formats, automatic grading is challenging due to natural language ambiguity and versatility. While automatic short answer grading models are beginning to compare to human performance on some datasets, their robustness, especially to adversarially manipulated data, is questionable. Exploitable vulnerabilities in grading models can have far-reaching consequences ranging from cheating students receiving undeserved credit to undermining automatic grading altogether - even when most predictions are valid. In this paper, we devise a black-box adversarial attack tailored to the educational short answer grading scenario to investigate the grading models' robustness. In our attack, we insert adjectives and adverbs into natural places of incorrect student answers, fooling the model into predicting them as correct. We observed a loss of prediction accuracy between 10 and 22 percentage points using the state-of-the-art models BERT and T5. While our attack made answers appear less natural to humans in our experiments, it did not significantly increase the graders' suspicions of cheating. Based on our experiments, we provide recommendations for utilizing automatic grading systems more safely in practice.

摘要: 自动评分模型的价值在于在教学过程中节省了大量学生的时间和精力。尤其是随着教育的日益数字化和大规模标准化考试的兴趣，自动评分的普及已经上升到商业解决方案被广泛获得和使用的地步。然而，对于简答题格式，由于自然语言的歧义性和多功能性，自动评分是具有挑战性的。虽然自动简答评分模型开始与人类在某些数据集上的表现进行比较，但它们的稳健性，特别是对相反操纵的数据，是值得怀疑的。评分模型中可利用的漏洞可能会产生深远的后果，从作弊的学生获得不应得的学分，到完全破坏自动评分--即使大多数预测是正确的。在本文中，我们设计了一个针对教育简答题评分场景的黑盒对抗性攻击，以考察评分模型的稳健性。在我们的攻击中，我们在错误的学生答案的自然位置插入形容词和副词，愚弄模型预测它们是正确的。我们观察到，使用最先进的模型BERT和T5预测精度损失了10到22个百分点。虽然我们的攻击使答案在我们的实验中对人类来说不那么自然，但它并没有显著增加评分员作弊的怀疑。在实验的基础上，我们提出了在实践中更安全地使用自动评分系统的建议。



## **42. Data-Adaptive Discriminative Feature Localization with Statistically Guaranteed Interpretation**

基于统计保证解释的数据自适应判别特征定位 stat.ML

27 pages, 11 figures

**SubmitDate**: 2022-11-18    [abs](http://arxiv.org/abs/2211.10061v1) [paper-pdf](http://arxiv.org/pdf/2211.10061v1)

**Authors**: Ben Dai, Xiaotong Shen, Lin Yee Chen, Chunlin Li, Wei Pan

**Abstract**: In explainable artificial intelligence, discriminative feature localization is critical to reveal a blackbox model's decision-making process from raw data to prediction. In this article, we use two real datasets, the MNIST handwritten digits and MIT-BIH Electrocardiogram (ECG) signals, to motivate key characteristics of discriminative features, namely adaptiveness, predictive importance and effectiveness. Then, we develop a localization framework based on adversarial attacks to effectively localize discriminative features. In contrast to existing heuristic methods, we also provide a statistically guaranteed interpretability of the localized features by measuring a generalized partial $R^2$. We apply the proposed method to the MNIST dataset and the MIT-BIH dataset with a convolutional auto-encoder. In the first, the compact image regions localized by the proposed method are visually appealing. Similarly, in the second, the identified ECG features are biologically plausible and consistent with cardiac electrophysiological principles while locating subtle anomalies in a QRS complex that may not be discernible by the naked eye. Overall, the proposed method compares favorably with state-of-the-art competitors. Accompanying this paper is a Python library dnn-locate (https://dnn-locate.readthedocs.io/en/latest/) that implements the proposed approach.

摘要: 在可解释人工智能中，区分特征定位是揭示黑盒模型从原始数据到预测的决策过程的关键。在本文中，我们使用两个真实的数据集，MNIST手写数字和MIT-BIH心电信号，来激发区分特征的关键特征，即适应性、预测重要性和有效性。然后，我们开发了一个基于对抗性攻击的定位框架，以有效地定位区分特征。与现有的启发式方法不同，我们还通过度量广义部分$R^2$来提供局部特征的统计保证的可解释性。我们将所提出的方法应用于MNIST数据集和具有卷积自动编码器的MIT-BIH数据集。首先，该方法定位出的紧凑图像区域具有良好的视觉吸引力。同样，在第二种方法中，所识别的心电特征在生物学上是可信的，并且与心脏电生理原理一致，同时定位QRS复合波中可能用肉眼无法识别的细微异常。总体而言，所提出的方法与最先进的竞争对手相比是有利的。本文附带了一个实现该方法的Python库dnn-Locate(https://dnn-locate.readthedocs.io/en/latest/)。



## **43. Adversarial Stimuli: Attacking Brain-Computer Interfaces via Perturbed Sensory Events**

对抗性刺激：通过扰乱的感觉事件攻击脑-计算机接口 cs.CR

**SubmitDate**: 2022-11-18    [abs](http://arxiv.org/abs/2211.10033v1) [paper-pdf](http://arxiv.org/pdf/2211.10033v1)

**Authors**: Bibek Upadhayay, Vahid Behzadan

**Abstract**: Machine learning models are known to be vulnerable to adversarial perturbations in the input domain, causing incorrect predictions. Inspired by this phenomenon, we explore the feasibility of manipulating EEG-based Motor Imagery (MI) Brain Computer Interfaces (BCIs) via perturbations in sensory stimuli. Similar to adversarial examples, these \emph{adversarial stimuli} aim to exploit the limitations of the integrated brain-sensor-processing components of the BCI system in handling shifts in participants' response to changes in sensory stimuli. This paper proposes adversarial stimuli as an attack vector against BCIs, and reports the findings of preliminary experiments on the impact of visual adversarial stimuli on the integrity of EEG-based MI BCIs. Our findings suggest that minor adversarial stimuli can significantly deteriorate the performance of MI BCIs across all participants (p=0.0003). Additionally, our results indicate that such attacks are more effective in conditions with induced stress.

摘要: 众所周知，机器学习模型容易受到输入域中对抗性扰动的影响，从而导致错误的预测。受这一现象的启发，我们探索了通过感觉刺激的扰动来操纵基于EEG的运动想象(MI)脑计算机接口(BCI)的可行性。与对抗性的例子类似，这些对抗性刺激旨在利用脑-机接口系统集成的大脑传感器处理组件在处理参与者对感觉刺激变化的反应的变化方面的局限性。本文提出用对抗性刺激作为攻击BCI的载体，并报道了视觉对抗性刺激对基于脑电信号的MI BCI完整性影响的初步实验结果。我们的发现表明，轻微的对抗性刺激可以显著降低所有参与者的MI BCI的表现(p=0.0003)。此外，我们的结果表明，这种攻击在有诱导应激的条件下更有效。



## **44. Adaptive Test-Time Defense with the Manifold Hypothesis**

流形假设下的自适应测试时间防御 cs.LG

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2210.14404v3) [paper-pdf](http://arxiv.org/pdf/2210.14404v3)

**Authors**: Zhaoyuan Yang, Zhiwei Xu, Jing Zhang, Richard Hartley, Peter Tu

**Abstract**: In this work, we formulate a novel framework of adversarial robustness using the manifold hypothesis. Our framework provides sufficient conditions for defending against adversarial examples. We develop a test-time defense method with variational inference and our formulation. The developed approach combines manifold learning with variational inference to provide adversarial robustness without the need for adversarial training. We show that our approach can provide adversarial robustness even if attackers are aware of the existence of test-time defense. In addition, our approach can also serve as a test-time defense mechanism for variational autoencoders.

摘要: 在这项工作中，我们使用流形假设建立了一个新的对抗健壮性框架。我们的框架为防御对抗性例子提供了充分的条件。我们提出了一种带有变分推理的测试时间防御方法和我们的公式。该方法将流形学习和变分推理相结合，在不需要对抗性训练的情况下提供对抗性稳健性。我们表明，即使攻击者知道测试时间防御的存在，我们的方法也可以提供对抗健壮性。此外，我们的方法还可以作为可变自动编码器的测试时间防御机制。



## **45. UPTON: Unattributable Authorship Text via Data Poisoning**

Upton：数据中毒导致的不明作者身份文本 cs.CY

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2211.09717v1) [paper-pdf](http://arxiv.org/pdf/2211.09717v1)

**Authors**: Ziyao Wang, Thai Le, Dongwon Lee

**Abstract**: In online medium such as opinion column in Bloomberg, The Guardian and Western Journal, aspiring writers post their writings for various reasons with their names often proudly open. However, it may occur that such a writer wants to write in other venues anonymously or under a pseudonym (e.g., activist, whistle-blower). However, if an attacker has already built an accurate authorship attribution (AA) model based off of the writings from such platforms, attributing an anonymous writing to the known authorship is possible. Therefore, in this work, we ask a question "can one make the writings and texts, T, in the open spaces such as opinion sharing platforms unattributable so that AA models trained from T cannot attribute authorship well?" Toward this question, we present a novel solution, UPTON, that exploits textual data poisoning method to disturb the training process of AA models. UPTON uses data poisoning to destroy the authorship feature only in training samples by perturbing them, and try to make released textual data unlearnable on deep neuron networks. It is different from previous obfuscation works, that use adversarial attack to modify the test samples and mislead an AA model, and also the backdoor works, which use trigger words both in test and training samples and only change the model output when trigger words occur. Using four authorship datasets (e.g., IMDb10, IMDb64, Enron and WJO), then, we present empirical validation where: (1)UPTON is able to downgrade the test accuracy to about 30% with carefully designed target-selection methods. (2)UPTON poisoning is able to preserve most of the original semantics. The BERTSCORE between the clean and UPTON poisoned texts are higher than 0.95. The number is very closed to 1.00, which means no sematic change. (3)UPTON is also robust towards spelling correction systems.

摘要: 在彭博社、《卫报》、《西部日报》的观点专栏等网络媒体上，有抱负的作家出于各种原因发布自己的作品，经常自豪地打开自己的名字。然而，可能会发生这样的作者想要在其他场所匿名或以化名(例如，活动家、告密者)写作的情况。然而，如果攻击者已经基于来自这些平台的作品构建了准确的作者归属(AA)模型，则可以将匿名作品归因于已知的作者。因此，在这项工作中，我们提出了一个问题：是否可以让意见分享平台等开放空间中的文字和文本T无法归因于T，从而使从T训练的AA模型无法很好地归属作者？针对这一问题，我们提出了一种新的解决方案Upton，它利用文本数据毒化方法来干扰AA模型的训练过程。厄普顿使用数据中毒来破坏只有在训练样本中才有的作者特征，并试图使已发布的文本数据在深层神经元网络上无法学习。不同于以往的混淆工作，它使用对抗性攻击来修改测试样本，误导AA模型；后门工作，在测试和训练样本中都使用触发词，只有在触发词出现时才改变模型输出。然后，使用四个作者的数据集(例如，IMDb10，IMDb64，Enron和WJO)，我们提供了经验验证，其中：(1)Upton能够通过精心设计的目标选择方法将测试准确率降低到约30%。(2)Upton中毒能够保留大部分原始语义。CLEAN文本和Upton中毒文本之间的BERTSCORE均大于0.95。这个数字非常接近1.00，这意味着没有语义变化。(3)厄普顿对拼写纠正系统也很感兴趣。



## **46. An efficient combination of quantum error correction and authentication**

一种量子纠错和认证的有效组合 quant-ph

30 pages, 10 figures

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2211.09686v1) [paper-pdf](http://arxiv.org/pdf/2211.09686v1)

**Authors**: Yfke Dulek, Garazi Muguruza, Florian Speelman

**Abstract**: When sending quantum information over a channel, we want to ensure that the message remains intact. Quantum error correction and quantum authentication both aim to protect (quantum) information, but approach this task from two very different directions: error-correcting codes protect against probabilistic channel noise and are meant to be very robust against small errors, while authentication codes prevent adversarial attacks and are designed to be very sensitive against any error, including small ones.   In practice, when sending an authenticated state over a noisy channel, one would have to wrap it in an error-correcting code to counterbalance the sensitivity of the underlying authentication scheme. We study the question of whether this can be done more efficiently by combining the two functionalities in a single code. To illustrate the potential of such a combination, we design the threshold code, a modification of the trap authentication code which preserves that code's authentication properties, but which is naturally robust against depolarizing channel noise. We show that the threshold code needs polylogarithmically fewer qubits to achieve the same level of security and robustness, compared to the naive composition of the trap code with any concatenated CSS code. We believe our analysis opens the door to combining more general error-correction and authentication codes, which could improve the practicality of the resulting scheme.

摘要: 在通过通道发送量子信息时，我们希望确保消息保持不变。量子纠错和量子认证都旨在保护(量子)信息，但从两个非常不同的方向来处理这项任务：纠错码保护不受概率信道噪声的影响，并且对小错误非常健壮，而认证码防止对手攻击，并且对任何错误(包括小错误)非常敏感。在实践中，当通过噪声信道发送认证状态时，必须将其包装在纠错码中以平衡基础认证方案的敏感性。我们研究的问题是，是否可以通过在单个代码中组合这两个功能来更有效地完成这项工作。为了说明这种组合的可能性，我们设计了门限码，这是对陷阱认证码的修改，它保留了该码的认证属性，但对去极化信道噪声具有天然的健壮性。我们证明，与任何级联的CSS码相比，门限码需要更少的多对数量子比特来实现相同级别的安全性和稳健性。我们相信，我们的分析为结合更一般的纠错和认证码打开了大门，这可能会提高所得到的方案的实用性。



## **47. Towards Good Practices in Evaluating Transfer Adversarial Attacks**

在评估转会对抗性攻击方面的良好做法 cs.CR

Our code and a list of categorized attacks are publicly available at  https://github.com/ZhengyuZhao/TransferAttackEval

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2211.09565v1) [paper-pdf](http://arxiv.org/pdf/2211.09565v1)

**Authors**: Zhengyu Zhao, Hanwei Zhang, Renjue Li, Ronan Sicre, Laurent Amsaleg, Michael Backes

**Abstract**: Transfer adversarial attacks raise critical security concerns in real-world, black-box scenarios. However, the actual progress of attack methods is difficult to assess due to two main limitations in existing evaluations. First, existing evaluations are unsystematic and sometimes unfair since new methods are often directly added to old ones without complete comparisons to similar methods. Second, existing evaluations mainly focus on transferability but overlook another key attack property: stealthiness. In this work, we design good practices to address these limitations. We first introduce a new attack categorization, which enables our systematic analyses of similar attacks in each specific category. Our analyses lead to new findings that complement or even challenge existing knowledge. Furthermore, we comprehensively evaluate 23 representative attacks against 9 defenses on ImageNet. We pay particular attention to stealthiness, by adopting diverse imperceptibility metrics and looking into new, finer-grained characteristics. Our evaluation reveals new important insights: 1) Transferability is highly contextual, and some white-box defenses may give a false sense of security since they are actually vulnerable to (black-box) transfer attacks; 2) All transfer attacks are less stealthy, and their stealthiness can vary dramatically under the same $L_{\infty}$ bound.

摘要: 在现实世界的黑盒场景中，传输敌意攻击会引发严重的安全问题。然而，由于现有评估中的两个主要限制，攻击方法的实际进展很难评估。首先，现有的评价是不系统的，有时是不公平的，因为新的方法往往直接添加到旧的方法中，而没有与类似的方法进行完全的比较。其次，现有的评估主要集中在可转移性上，而忽略了另一个关键的攻击属性：隐蔽性。在这项工作中，我们设计了良好的实践来解决这些限制。我们首先介绍了一种新的攻击分类，它使我们能够对每个特定类别中的类似攻击进行系统分析。我们的分析导致了补充甚至挑战现有知识的新发现。此外，我们还综合评估了ImageNet上针对9个防御的23种代表性攻击。我们特别关注隐蔽性，采用了不同的隐蔽性度量标准，并研究了新的、更细粒度的特征。我们的评估揭示了新的重要见解：1)可转移性与上下文高度相关，一些白盒防御可能会给人一种错误的安全感，因为它们实际上容易受到(黑盒)传输攻击；2)所有传输攻击的隐蔽性都较低，并且它们的隐蔽性在相同的$L_(\infty)$界限下可能会有很大的变化。



## **48. Ignore Previous Prompt: Attack Techniques For Language Models**

忽略前面的提示：语言模型的攻击技巧 cs.CL

ML Safety Workshop NeurIPS 2022

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2211.09527v1) [paper-pdf](http://arxiv.org/pdf/2211.09527v1)

**Authors**: Fábio Perez, Ian Ribeiro

**Abstract**: Transformer-based large language models (LLMs) provide a powerful foundation for natural language tasks in large-scale customer-facing applications. However, studies that explore their vulnerabilities emerging from malicious user interaction are scarce. By proposing PromptInject, a prosaic alignment framework for mask-based iterative adversarial prompt composition, we examine how GPT-3, the most widely deployed language model in production, can be easily misaligned by simple handcrafted inputs. In particular, we investigate two types of attacks -- goal hijacking and prompt leaking -- and demonstrate that even low-aptitude, but sufficiently ill-intentioned agents, can easily exploit GPT-3's stochastic nature, creating long-tail risks. The code for PromptInject is available at https://github.com/agencyenterprise/PromptInject.

摘要: 基于转换器的大型语言模型(LLM)为大规模面向客户应用中的自然语言任务提供了强大的基础。然而，探索恶意用户交互中出现的漏洞的研究很少。通过提出PromptInject，一个基于掩码的迭代对抗性提示合成的平淡无奇的对齐框架，我们研究了GPT-3，生产中应用最广泛的语言模型，如何通过简单的手工制作的输入很容易地错位。特别是，我们调查了两种类型的攻击--目标劫持和快速泄漏--并证明即使是低能力但足够恶意的代理也可以很容易地利用GPT-3的随机性质，创造长尾风险。PromptInject的代码可在https://github.com/agencyenterprise/PromptInject.上获得



## **49. Look Closer to Your Enemy: Learning to Attack via Teacher-student Mimicking**

走近你的敌人：通过师生模仿学习攻击 cs.CV

13 pages, 8 figures, NDSS

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2207.13381v3) [paper-pdf](http://arxiv.org/pdf/2207.13381v3)

**Authors**: Mingjie Wang, Zhiqing Tang, Sirui Li, Dingwen Xiao

**Abstract**: This paper aims to generate realistic attack samples of person re-identification, ReID, by reading the enemy's mind (VM). In this paper, we propose a novel inconspicuous and controllable ReID attack baseline, LCYE, to generate adversarial query images. Concretely, LCYE first distills VM's knowledge via teacher-student memory mimicking in the proxy task. Then this knowledge prior acts as an explicit cipher conveying what is essential and realistic, believed by VM, for accurate adversarial misleading. Besides, benefiting from the multiple opposing task framework of LCYE, we further investigate the interpretability and generalization of ReID models from the view of the adversarial attack, including cross-domain adaption, cross-model consensus, and online learning process. Extensive experiments on four ReID benchmarks show that our method outperforms other state-of-the-art attackers with a large margin in white-box, black-box, and target attacks. Our code is now available at https://gitfront.io/r/user-3704489/mKXusqDT4ffr/LCYE/.

摘要: 本文旨在通过读取敌人的心理(Vm)来生成真实的人重新识别的攻击样本，Reid。本文提出了一种新的隐蔽可控的Reid攻击基线--LCYE，用于生成敌意查询图像。具体来说，LCYE首先通过模仿代理任务中的师生记忆来提取VM的知识。然后，这种先验知识就像一个明确的密码，传达了被VM认为是必要和现实的东西，以实现准确的对抗性误导。此外，得益于LCYE的多重对立任务框架，我们从对抗性攻击的角度进一步考察了Reid模型的可解释性和泛化，包括跨域适应、跨模型共识和在线学习过程。在四个Reid基准测试上的大量实验表明，我们的方法在白盒、黑盒和目标攻击中的性能远远优于其他最先进的攻击者。我们的代码现已在https://gitfront.io/r/user-3704489/mKXusqDT4ffr/LCYE/.上提供



## **50. Phantom Sponges: Exploiting Non-Maximum Suppression to Attack Deep Object Detectors**

幻影海绵：利用非最大抑制攻击深度对象探测器 cs.CV

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2205.13618v3) [paper-pdf](http://arxiv.org/pdf/2205.13618v3)

**Authors**: Avishag Shapira, Alon Zolfi, Luca Demetrio, Battista Biggio, Asaf Shabtai

**Abstract**: Adversarial attacks against deep learning-based object detectors have been studied extensively in the past few years. Most of the attacks proposed have targeted the model's integrity (i.e., caused the model to make incorrect predictions), while adversarial attacks targeting the model's availability, a critical aspect in safety-critical domains such as autonomous driving, have not yet been explored by the machine learning research community. In this paper, we propose a novel attack that negatively affects the decision latency of an end-to-end object detection pipeline. We craft a universal adversarial perturbation (UAP) that targets a widely used technique integrated in many object detector pipelines -- non-maximum suppression (NMS). Our experiments demonstrate the proposed UAP's ability to increase the processing time of individual frames by adding "phantom" objects that overload the NMS algorithm while preserving the detection of the original objects which allows the attack to go undetected for a longer period of time.

摘要: 针对基于深度学习的目标检测器的对抗性攻击在过去的几年中得到了广泛的研究。大多数提出的攻击都是针对模型的完整性(即导致模型做出错误的预测)，而针对模型可用性的对抗性攻击(自动驾驶等安全关键领域的一个关键方面)尚未被机器学习研究社区探索。在本文中，我们提出了一种新的攻击，它对端到端对象检测流水线的决策延迟产生负面影响。我们设计了一种通用对抗摄动(UAP)，目标是集成在许多对象探测器流水线中的一种广泛使用的技术--非最大抑制(NMS)。我们的实验证明了所提出的UAP能够通过添加重载NMS算法的“幻影”对象来增加单个帧的处理时间，同时保持对原始对象的检测，从而允许攻击在更长的时间内被检测到。



