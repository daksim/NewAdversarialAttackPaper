# Latest Adversarial Attack Papers
**update at 2022-06-24 06:31:30**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Making Generated Images Hard To Spot: A Transferable Attack On Synthetic Image Detectors**

使生成的图像难以识别：对合成图像检测器的可转移攻击 cs.CV

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2104.12069v2)

**Authors**: Xinwei Zhao, Matthew C. Stamm

**Abstracts**: Visually realistic GAN-generated images have recently emerged as an important misinformation threat. Research has shown that these synthetic images contain forensic traces that are readily identifiable by forensic detectors. Unfortunately, these detectors are built upon neural networks, which are vulnerable to recently developed adversarial attacks. In this paper, we propose a new anti-forensic attack capable of fooling GAN-generated image detectors. Our attack uses an adversarially trained generator to synthesize traces that these detectors associate with real images. Furthermore, we propose a technique to train our attack so that it can achieve transferability, i.e. it can fool unknown CNNs that it was not explicitly trained against. We evaluate our attack through an extensive set of experiments, where we show that our attack can fool eight state-of-the-art detection CNNs with synthetic images created using seven different GANs, and outperform other alternative attacks.

摘要: 视觉逼真的GaN生成的图像最近已经成为一种重要的错误信息威胁。研究表明，这些合成图像包含法医探测器可以很容易识别的法医痕迹。不幸的是，这些探测器是建立在神经网络的基础上的，而神经网络很容易受到最近发展起来的对抗性攻击。在本文中，我们提出了一种新的反取证攻击，能够欺骗GaN生成的图像检测器。我们的攻击使用一个经过敌意训练的生成器来合成这些检测器与真实图像相关联的痕迹。此外，我们还提出了一种技术来训练我们的攻击，以便它能够实现可转移性，即它可以欺骗它没有明确训练针对的未知CNN。我们通过一组广泛的实验来评估我们的攻击，其中我们表明我们的攻击可以通过使用7个不同的GAN创建的合成图像来欺骗8个最先进的检测CNN，并且性能优于其他替代攻击。



## **2. AdvSmo: Black-box Adversarial Attack by Smoothing Linear Structure of Texture**

AdvSmo：平滑纹理线性结构的黑盒对抗性攻击 cs.CV

6 pages,3 figures

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2206.10988v1)

**Authors**: Hui Xia, Rui Zhang, Shuliang Jiang, Zi Kang

**Abstracts**: Black-box attacks usually face two problems: poor transferability and the inability to evade the adversarial defense. To overcome these shortcomings, we create an original approach to generate adversarial examples by smoothing the linear structure of the texture in the benign image, called AdvSmo. We construct the adversarial examples without relying on any internal information to the target model and design the imperceptible-high attack success rate constraint to guide the Gabor filter to select appropriate angles and scales to smooth the linear texture from the input images to generate adversarial examples. Benefiting from the above design concept, AdvSmo will generate adversarial examples with strong transferability and solid evasiveness. Finally, compared to the four advanced black-box adversarial attack methods, for the eight target models, the results show that AdvSmo improves the average attack success rate by 9% on the CIFAR-10 and 16% on the Tiny-ImageNet dataset compared to the best of these attack methods.

摘要: 黑盒攻击通常面临两个问题：可转移性差和无法躲避对手的防御。为了克服这些缺点，我们创建了一种新颖的方法，通过平滑良性图像中纹理的线性结构来生成对抗性示例，称为AdvSmo。我们在不依赖目标模型任何内部信息的情况下构造敌意样本，并设计了不可察觉的高攻击成功率约束来指导Gabor滤波器选择合适的角度和尺度来平滑输入图像中的线性纹理来生成敌意样本。受益于上述设计理念，AdvSmo将生成具有很强的可转移性和坚实的规避能力的对抗性范例。最后，与四种先进的黑盒对抗攻击方法相比，对于8个目标模型，结果表明，AdvSmo在CIFAR-10上的平均攻击成功率比这些攻击方法中最好的方法提高了9%，在Tiny-ImageNet数据集上的平均攻击成功率提高了16%。



## **3. Adversarial Reconfigurable Intelligent Surface Against Physical Layer Key Generation**

对抗物理层密钥生成的对抗性可重构智能表面 eess.SP

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2206.10955v1)

**Authors**: Zhuangkun Wei, Bin Li, Weisi Guo

**Abstracts**: The development of reconfigurable intelligent surface (RIS) has recently advanced the research of physical layer security (PLS). Beneficial impact of RIS includes but is not limited to offering a new domain of freedom (DoF) for key-less PLS optimization, and increasing channel randomness for physical layer secret key generation (PL-SKG). However, there is a lack of research studying how adversarial RIS can be used to damage the communication confidentiality. In this work, we show how a Eve controlled adversarial RIS (Eve-RIS) can be used to reconstruct the shared PLS secret key between legitimate users (Alice and Bob). This is achieved by Eve-RIS overlaying the legitimate channel with an artificial random and reciprocal channel. The resulting Eve-RIS corrupted channel enable Eve to successfully attack the PL-SKG process. To operationalize this novel concept, we design Eve-RIS schemes against two PL-SKG techniques used: (i) the channel estimation based PL-SKG, and (ii) the two-way cross multiplication based PL-SKG. Our results show a high key match rate between the designed Eve-RIS and the legitimate users. We also present theoretical key match rate between Eve-RIS and legitimate users. Our novel scheme is different from the existing spoofing-Eve, in that the latter can be easily detected by comparing the channel estimation results of the legitimate users. Indeed, our proposed Eve-RIS can maintain the legitimate channel reciprocity, which makes detection challenging. This means the novel Eve-RIS provides a new eavesdropping threat on PL-SKG, which can spur new research areas to counter adversarial RIS attacks.

摘要: 近年来，可重构智能表面(RIS)的发展推动了物理层安全(PLS)的研究。RIS的有益影响包括但不限于提供用于无密钥的PLS优化的新的自由域(DoF)，以及增加用于物理层秘密密钥生成(PL-SKG)的信道随机性。然而，缺乏研究如何利用敌意RIS来破坏通信机密性。在这项工作中，我们展示了如何使用Eve控制的对抗RIS(Eve-RIS)来重构合法用户(Alice和Bob)之间共享的PLS秘密密钥。这是通过EVE-RIS用人工随机和互惠的信道覆盖合法信道来实现的。由此产生的Eve-RIS损坏的通道使Eve能够成功攻击PL-SKG进程。为了实现这一新概念，我们针对使用的两种PL-SKG技术设计了Eve-RIS方案：(I)基于信道估计的PL-SKG和(Ii)基于双向交叉乘法的PL-SKG。结果表明，所设计的Eve-RIS与合法用户之间具有较高的密钥匹配率。我们还给出了Eve-RIS与合法用户之间的理论密钥匹配率。我们的新方案不同于现有的欺骗-EVE方案，后者可以通过比较合法用户的信道估计结果来容易地检测到。事实上，我们提出的Eve-RIS能够保持合法的信道互惠，这使得检测具有挑战性。这意味着新的Eve-RIS为PL-SKG提供了一种新的窃听威胁，可以刺激新的研究领域来对抗敌意RIS攻击。



## **4. Introduction to Machine Learning for the Sciences**

面向科学的机器学习导论 physics.comp-ph

84 pages, 37 figures. The content of these lecture notes together  with exercises is available under http://www.ml-lectures.org. A shorter  German version of the lecture notes is published in the Springer essential  series, ISBN 978-3-658-32268-7, doi:10.1007/978-3-658-32268-7

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2102.04883v2)

**Authors**: Titus Neupert, Mark H Fischer, Eliska Greplova, Kenny Choo, M. Michael Denner

**Abstracts**: This is an introductory machine-learning course specifically developed with STEM students in mind. Our goal is to provide the interested reader with the basics to employ machine learning in their own projects and to familiarize themself with the terminology as a foundation for further reading of the relevant literature. In these lecture notes, we discuss supervised, unsupervised, and reinforcement learning. The notes start with an exposition of machine learning methods without neural networks, such as principle component analysis, t-SNE, clustering, as well as linear regression and linear classifiers. We continue with an introduction to both basic and advanced neural-network structures such as dense feed-forward and conventional neural networks, recurrent neural networks, restricted Boltzmann machines, (variational) autoencoders, generative adversarial networks. Questions of interpretability are discussed for latent-space representations and using the examples of dreaming and adversarial attacks. The final section is dedicated to reinforcement learning, where we introduce basic notions of value functions and policy learning.

摘要: 这是一门专门为STEM学生开发的机器学习入门课程。我们的目标是为感兴趣的读者提供在他们自己的项目中使用机器学习的基础知识，并熟悉这些术语作为进一步阅读相关文献的基础。在这些课堂讲稿中，我们讨论有监督、无监督和强化学习。这些笔记首先阐述了没有神经网络的机器学习方法，如主成分分析、t-SNE、聚类以及线性回归和线性分类器。我们继续介绍基本和高级神经网络结构，如密集前馈和常规神经网络、递归神经网络、受限Boltzmann机器、(变分)自动编码器、生成性对手网络。讨论了潜在空间表示的可解释性问题，并使用了梦和对抗性攻击的例子。最后一节致力于强化学习，在那里我们介绍价值函数和策略学习的基本概念。



## **5. Guided Diffusion Model for Adversarial Purification from Random Noise**

随机噪声中对抗性净化的引导扩散模型 cs.LG

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2206.10875v1)

**Authors**: Quanlin Wu, Hang Ye, Yuntian Gu

**Abstracts**: In this paper, we propose a novel guided diffusion purification approach to provide a strong defense against adversarial attacks. Our model achieves 89.62% robust accuracy under PGD-L_inf attack (eps = 8/255) on the CIFAR-10 dataset. We first explore the essential correlations between unguided diffusion models and randomized smoothing, enabling us to apply the models to certified robustness. The empirical results show that our models outperform randomized smoothing by 5% when the certified L2 radius r is larger than 0.5.

摘要: 在本文中，我们提出了一种新的引导扩散净化方法，以提供对对手攻击的强大防御。在Pgd-L_inf攻击(EPS=8/255)下，我们的模型在CIFAR-10数据集上达到了89.62%的稳健准确率。我们首先探讨了无引导扩散模型和随机平滑之间的本质关联，使我们能够将这些模型应用于已证明的稳健性。实证结果表明，当认证的L2半径r大于0.5时，我们的模型比随机平滑的性能高出5%。



## **6. Robust Universal Adversarial Perturbations**

稳健的普遍对抗性摄动 cs.LG

16 pages, 3 figures

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2206.10858v1)

**Authors**: Changming Xu, Gagandeep Singh

**Abstracts**: Universal Adversarial Perturbations (UAPs) are imperceptible, image-agnostic vectors that cause deep neural networks (DNNs) to misclassify inputs from a data distribution with high probability. Existing methods do not create UAPs robust to transformations, thereby limiting their applicability as a real-world attacks. In this work, we introduce a new concept and formulation of robust universal adversarial perturbations. Based on our formulation, we build a novel, iterative algorithm that leverages probabilistic robustness bounds for generating UAPs robust against transformations generated by composing arbitrary sub-differentiable transformation functions. We perform an extensive evaluation on the popular CIFAR-10 and ILSVRC 2012 datasets measuring robustness under human-interpretable semantic transformations, such as rotation, contrast changes, etc, that are common in the real-world. Our results show that our generated UAPs are significantly more robust than those from baselines.

摘要: 通用对抗性扰动(UAP)是一种不可察觉的、与图像无关的向量，它会导致深度神经网络(DNN)高概率地对来自数据分布的输入进行错误分类。现有方法不能创建对变换具有健壮性的UAP，从而限制了它们作为现实世界攻击的适用性。在这项工作中，我们引入了一个新的概念和形式，稳健的泛对抗摄动。基于我们的公式，我们构建了一种新颖的迭代算法，该算法利用概率鲁棒界来生成对通过合成任意次可微变换函数而产生的变换具有鲁棒性的UAP。我们在流行的CIFAR-10和ILSVRC 2012数据集上进行了广泛的评估，测量了在人类可解释的语义转换下的健壮性，例如旋转、对比度变化等，这些转换在现实世界中很常见。我们的结果表明，我们生成的UAP明显比基线生成的UAP更健壮。



## **7. Adaptive Adversarial Training to Improve Adversarial Robustness of DNNs for Medical Image Segmentation and Detection**

自适应对抗训练提高DNN在医学图像分割和检测中的对抗鲁棒性 eess.IV

17 pages

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2206.01736v2)

**Authors**: Linhai Ma, Liang Liang

**Abstracts**: It is known that Deep Neural networks (DNNs) are vulnerable to adversarial attacks, and the adversarial robustness of DNNs could be improved by adding adversarial noises to training data (e.g., the standard adversarial training (SAT)). However, inappropriate noises added to training data may reduce a model's performance, which is termed the trade-off between accuracy and robustness. This problem has been sufficiently studied for the classification of whole images but has rarely been explored for image analysis tasks in the medical application domain, including image segmentation, landmark detection, and object detection tasks. In this study, we show that, for those medical image analysis tasks, the SAT method has a severe issue that limits its practical use: it generates a fixed and unified level of noise for all training samples for robust DNN training. A high noise level may lead to a large reduction in model performance and a low noise level may not be effective in improving robustness. To resolve this issue, we design an adaptive-margin adversarial training (AMAT) method that generates sample-wise adaptive adversarial noises for robust DNN training. In contrast to the existing, classification-oriented adversarial training methods, our AMAT method uses a loss-defined-margin strategy so that it can be applied to different tasks as long as the loss functions are well-defined. We successfully apply our AMAT method to state-of-the-art DNNs, using five publicly available datasets. The experimental results demonstrate that: (1) our AMAT method can be applied to the three seemingly different tasks in the medical image application domain; (2) AMAT outperforms the SAT method in adversarial robustness; (3) AMAT has a minimal reduction in prediction accuracy on clean data, compared with the SAT method; and (4) AMAT has almost the same training time cost as SAT.

摘要: 众所周知，深度神经网络(DNN)容易受到对抗性攻击，通过在训练数据(例如标准对抗性训练(SAT))中添加对抗性噪声可以提高DNN的对抗性健壮性。然而，在训练数据中添加不适当的噪声可能会降低模型的性能，这被称为精度和稳健性之间的权衡。对于整个图像的分类，这个问题已经得到了充分的研究，但在医学应用领域的图像分析任务中，包括图像分割、地标检测和目标检测任务，却很少被探索。在这项研究中，我们发现，对于这些医学图像分析任务，SAT方法有一个严重的问题限制了它的实际应用：它为所有训练样本生成固定和统一的噪声水平，以便进行稳健的DNN训练。较高的噪声水平可能会导致模型性能的大幅降低，而较低的噪声水平可能不能有效地提高鲁棒性。为了解决这一问题，我们设计了一种自适应差值对抗性训练(AMAT)方法，该方法产生样本级自适应对抗性噪声，用于稳健的DNN训练。与现有的面向分类的对抗性训练方法相比，我们的AMAT方法使用了损失定义边际策略，因此只要损失函数定义得很好，它就可以应用于不同的任务。我们成功地将我们的AMAT方法应用于最先进的DNN，使用了五个公开可用的数据集。实验结果表明：(1)我们的AMAT方法可以应用于医学图像应用领域中三个看似不同的任务；(2)AMAT方法在对抗健壮性方面优于SAT方法；(3)与SAT方法相比，AMAT方法对干净数据的预测精度有很小的降低；(4)AMAT方法的训练时间开销与SAT方法几乎相同。



## **8. SSMI: How to Make Objects of Interest Disappear without Accessing Object Detectors?**

SSMI：如何在不访问对象探测器的情况下使感兴趣的对象消失？ cs.CV

6 pages, 2 figures

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2206.10809v1)

**Authors**: Hui Xia, Rui Zhang, Zi Kang, Shuliang Jiang

**Abstracts**: Most black-box adversarial attack schemes for object detectors mainly face two shortcomings: requiring access to the target model and generating inefficient adversarial examples (failing to make objects disappear in large numbers). To overcome these shortcomings, we propose a black-box adversarial attack scheme based on semantic segmentation and model inversion (SSMI). We first locate the position of the target object using semantic segmentation techniques. Next, we design a neighborhood background pixel replacement to replace the target region pixels with background pixels to ensure that the pixel modifications are not easily detected by human vision. Finally, we reconstruct a machine-recognizable example and use the mask matrix to select pixels in the reconstructed example to modify the benign image to generate an adversarial example. Detailed experimental results show that SSMI can generate efficient adversarial examples to evade human-eye perception and make objects of interest disappear. And more importantly, SSMI outperforms existing same kinds of attacks. The maximum increase in new and disappearing labels is 16%, and the maximum decrease in mAP metrics for object detection is 36%.

摘要: 大多数针对目标探测器的黑盒对抗性攻击方案主要面临两个缺点：需要访问目标模型和生成低效的对抗性实例(无法使对象大量消失)。为了克服这些不足，我们提出了一种基于语义分割和模型反转的黑盒对抗攻击方案。我们首先使用语义分割技术定位目标对象的位置。接下来，我们设计了一种邻域背景像素替换算法，将目标区域的像素替换为背景像素，以保证像素的变化不易被人的视觉检测到。最后，我们重建一个机器可识别的样本，并使用掩码矩阵来选择重建样本中的像素来修正良性图像以生成对抗性样本。详细的实验结果表明，SSMI能够生成有效的对抗性实例，避开人眼的感知，使感兴趣的对象消失。更重要的是，SSMI的性能优于现有的同类攻击。新的和消失的标签的最大增幅为16%，用于目标检测的MAP指标的最大降幅为36%。



## **9. Secure and Efficient Query Processing in Outsourced Databases**

外包数据库中安全高效的查询处理 cs.CR

Ph.D. thesis

**SubmitDate**: 2022-06-21    [paper-pdf](http://arxiv.org/pdf/2206.10753v1)

**Authors**: Dmytro Bogatov

**Abstracts**: Various cryptographic techniques are used in outsourced database systems to ensure data privacy while allowing for efficient querying. This work proposes a definition and components of a new secure and efficient outsourced database system, which answers various types of queries, with different privacy guarantees in different security models. This work starts with the survey of five order-revealing encryption schemes that can be used directly in many database indices and five range query protocols with various security / efficiency tradeoffs. The survey systematizes the state-of-the-art range query solutions in a snapshot adversary setting and offers some non-obvious observations regarding the efficiency of the constructions. In $\mathcal{E}\text{psolute}$, a secure range query engine, security is achieved in a setting with a much stronger adversary where she can continuously observe everything on the server, and leaking even the result size can enable a reconstruction attack. $\mathcal{E}\text{psolute}$ proposes a definition, construction, analysis, and experimental evaluation of a system that provably hides both access pattern and communication volume while remaining efficient. The work concludes with $k\text{-a}n\text{o}n$ -- a secure similarity search engine in a snapshot adversary model. The work presents a construction in which the security of $k\text{NN}$ queries is achieved similarly to OPE / ORE solutions -- encrypting the input with an approximate Distance Comparison Preserving Encryption scheme so that the inputs, the points in a hyperspace, are perturbed, but the query algorithm still produces accurate results. We use TREC datasets and queries for the search, and track the rank quality metrics such as MRR and nDCG. For the attacks, we build an LSTM model that trains on the correlation between a sentence and its embedding and then predicts words from the embedding.

摘要: 在外包数据库系统中使用了各种加密技术，以确保数据隐私，同时允许高效查询。提出了一种新的安全高效的外包数据库系统的定义和组成，该系统可以回答不同类型的查询，在不同的安全模型下具有不同的隐私保障。这项工作首先调查了五种可以直接用于许多数据库索引的顺序揭示加密方案和五种具有各种安全/效率权衡的范围查询协议。该调查将快照对手环境中最先进的范围查询解决方案系统化，并提供了一些关于构造效率的不明显的观察。在安全范围查询引擎$\Mathcal{E}\Text{psolte}$中，安全是在一个更强大的对手的设置下实现的，在这种设置中，她可以连续观察服务器上的一切，即使泄漏结果大小也可能导致重建攻击。$\Mathcal{E}\Text{psolte}$提出了一种系统的定义、构造、分析和实验评估，该系统可以证明在保持效率的同时隐藏了访问模式和通信量。这项工作以$k\Text{-a}n\Text{o}n$结束--快照对手模型中的安全相似性搜索引擎。该工作提出了一种结构，其中$k\Text{NN}$查询的安全性类似于OPE/ORE解决方案--使用一种保持近似距离比较的加密方案对输入进行加密，使得输入，即超空间中的点，被扰动，但查询算法仍然产生准确的结果。我们使用TREC数据集和查询进行搜索，并跟踪排名质量度量，如MRR和nDCG。对于攻击，我们建立了一个LSTM模型，该模型根据句子与其嵌入之间的相关性进行训练，然后从嵌入中预测单词。



## **10. FlashSyn: Flash Loan Attack Synthesis via Counter Example Driven Approximation**

FlashSyn：基于反例驱动近似的闪贷攻击合成 cs.PL

29 pages, 8 figures, technical report

**SubmitDate**: 2022-06-21    [paper-pdf](http://arxiv.org/pdf/2206.10708v1)

**Authors**: Zhiyang Chen, Sidi Mohamed Beillahi, Fan Long

**Abstracts**: In decentralized finance (DeFi) ecosystem, lenders can offer flash loans to borrowers, i.e., loans that are only valid within a blockchain transaction and must be repaid with some fees by the end of that transaction. Unlike normal loans, flash loans allow borrowers to borrow a large amount of assets without upfront collaterals deposits. Malicious adversaries can use flash loans to gather large amount of assets to launch costly exploitations targeting DeFi protocols. In this paper, we introduce a new framework for automated synthesis of adversarial contracts that exploit DeFi protocols using flash loans. To bypass the complexity of a DeFi protocol, we propose a new technique to approximate DeFi protocol functional behaviors using numerical methods. Then, we propose a novel algorithm to find an adversarial attack which constitutes of a sequence of invocations of functions in a DeFi protocol with the optimized parameters for profits. We implemented our framework in a tool called FlashSyn. We run FlashSyn on 5 DeFi protocols that were victims to flash loan attacks and DeFi protocols from Damn Vulnerable DeFi challenges. FlashSyn automatically synthesizes an adversarial attack for each one of them.

摘要: 在去中心化金融(Defi)生态系统中，贷款人可以向借款人提供闪存贷款，即仅在区块链交易中有效且必须在该交易结束前支付一定费用的贷款。与正常贷款不同，闪付贷款允许借款人借入大量资产，而无需预付抵押金。恶意攻击者可以使用闪存贷款来收集大量资产，以发起针对Defi协议的代价高昂的攻击。在这篇文章中，我们介绍了一个新的框架，用于自动合成利用闪存贷款的Defi协议的对抗性合同。为了绕过DEFI协议的复杂性，我们提出了一种利用数值方法来近似DEFI协议功能行为的新技术。然后，我们提出了一种新的算法来发现敌意攻击，该攻击由Defi协议中的一系列函数调用组成，并通过优化参数来获利。我们在一个名为FlashSyn的工具中实现了我们的框架。我们在5个Defi协议上运行FlashSyn，这些协议是闪电贷款攻击的受害者，并且是来自该死的脆弱Defi挑战的Defi协议。FlashSyn会自动为它们中的每一个合成一次对抗性攻击。



## **11. Using EBGAN for Anomaly Intrusion Detection**

利用EBGAN进行异常入侵检测 cs.CR

**SubmitDate**: 2022-06-21    [paper-pdf](http://arxiv.org/pdf/2206.10400v1)

**Authors**: Yi Cui, Wenfeng Shen, Jian Zhang, Weijia Lu, Chuang Liu, Lin Sun, Si Chen

**Abstracts**: As an active network security protection scheme, intrusion detection system (IDS) undertakes the important responsibility of detecting network attacks in the form of malicious network traffic. Intrusion detection technology is an important part of IDS. At present, many scholars have carried out extensive research on intrusion detection technology. However, developing an efficient intrusion detection method for massive network traffic data is still difficult. Since Generative Adversarial Networks (GANs) have powerful modeling capabilities for complex high-dimensional data, they provide new ideas for addressing this problem. In this paper, we put forward an EBGAN-based intrusion detection method, IDS-EBGAN, that classifies network records as normal traffic or malicious traffic. The generator in IDS-EBGAN is responsible for converting the original malicious network traffic in the training set into adversarial malicious examples. This is because we want to use adversarial learning to improve the ability of discriminator to detect malicious traffic. At the same time, the discriminator adopts Autoencoder model. During testing, IDS-EBGAN uses reconstruction error of discriminator to classify traffic records.

摘要: 入侵检测系统作为一种主动的网络安全防护方案，担负着检测以恶意网络流量为形式的网络攻击的重要责任。入侵检测技术是入侵检测系统的重要组成部分。目前，许多学者对入侵检测技术进行了广泛的研究。然而，开发一种高效的针对海量网络流量数据的入侵检测方法仍然是一个难点。由于生成性对抗网络(GAN)对复杂的高维数据具有强大的建模能力，它们为解决这一问题提供了新的思路。本文提出了一种基于EBGAN的入侵检测方法--IDS-EBGAN，将网络记录分为正常流量和恶意流量。入侵检测系统EBGAN中的生成器负责将训练集中的原始恶意网络流量转换为对抗性恶意实例。这是因为我们希望利用对抗性学习来提高鉴别器检测恶意流量的能力。同时，该鉴别器采用自动编码器模型。在测试过程中，IDS-EBGAN利用识别器的重构误差对流量记录进行分类。



## **12. Problem-Space Evasion Attacks in the Android OS: a Survey**

Android操作系统中的问题空间逃避攻击：综述 cs.CR

**SubmitDate**: 2022-06-21    [paper-pdf](http://arxiv.org/pdf/2205.14576v2)

**Authors**: Harel Berger, Chen Hajaj, Amit Dvir

**Abstracts**: Android is the most popular OS worldwide. Therefore, it is a target for various kinds of malware. As a countermeasure, the security community works day and night to develop appropriate Android malware detection systems, with ML-based or DL-based systems considered as some of the most common types. Against these detection systems, intelligent adversaries develop a wide set of evasion attacks, in which an attacker slightly modifies a malware sample to evade its target detection system. In this survey, we address problem-space evasion attacks in the Android OS, where attackers manipulate actual APKs, rather than their extracted feature vector. We aim to explore this kind of attacks, frequently overlooked by the research community due to a lack of knowledge of the Android domain, or due to focusing on general mathematical evasion attacks - i.e., feature-space evasion attacks. We discuss the different aspects of problem-space evasion attacks, using a new taxonomy, which focuses on key ingredients of each problem-space attack, such as the attacker model, the attacker's mode of operation, and the functional assessment of post-attack applications.

摘要: 安卓是全球最受欢迎的操作系统。因此，它是各种恶意软件的目标。作为对策，安全社区夜以继日地开发合适的Android恶意软件检测系统，基于ML或基于DL的系统被认为是一些最常见的类型。针对这些检测系统，智能攻击者开发了一系列广泛的逃避攻击，攻击者略微修改恶意软件样本以逃避其目标检测系统。在这篇调查中，我们讨论了Android操作系统中的问题空间逃避攻击，即攻击者操纵实际的APK，而不是他们提取的特征向量。我们的目标是探索这类攻击，由于缺乏Android领域的知识，或者由于专注于一般的数学逃避攻击，即特征空间逃避攻击，经常被研究界忽视。我们讨论了问题空间逃避攻击的不同方面，使用了一种新的分类方法，重点讨论了每种问题空间攻击的关键要素，如攻击者的模型、攻击者的操作模式以及攻击后应用程序的功能评估。



## **13. Certifiably Robust Policy Learning against Adversarial Communication in Multi-agent Systems**

多智能体系统中对抗敌意通信的可证明稳健策略学习 cs.LG

**SubmitDate**: 2022-06-21    [paper-pdf](http://arxiv.org/pdf/2206.10158v1)

**Authors**: Yanchao Sun, Ruijie Zheng, Parisa Hassanzadeh, Yongyuan Liang, Soheil Feizi, Sumitra Ganesh, Furong Huang

**Abstracts**: Communication is important in many multi-agent reinforcement learning (MARL) problems for agents to share information and make good decisions. However, when deploying trained communicative agents in a real-world application where noise and potential attackers exist, the safety of communication-based policies becomes a severe issue that is underexplored. Specifically, if communication messages are manipulated by malicious attackers, agents relying on untrustworthy communication may take unsafe actions that lead to catastrophic consequences. Therefore, it is crucial to ensure that agents will not be misled by corrupted communication, while still benefiting from benign communication. In this work, we consider an environment with $N$ agents, where the attacker may arbitrarily change the communication from any $C<\frac{N-1}{2}$ agents to a victim agent. For this strong threat model, we propose a certifiable defense by constructing a message-ensemble policy that aggregates multiple randomly ablated message sets. Theoretical analysis shows that this message-ensemble policy can utilize benign communication while being certifiably robust to adversarial communication, regardless of the attacking algorithm. Experiments in multiple environments verify that our defense significantly improves the robustness of trained policies against various types of attacks.

摘要: 在许多多智能体强化学习(MAIL)问题中，通信对于智能体共享信息和做出正确的决策是非常重要的。然而，当在存在噪声和潜在攻击者的真实世界应用中部署训练有素的通信代理时，基于通信的策略的安全性成为一个未被探索的严重问题。具体地说，如果通信消息被恶意攻击者操纵，依赖于不可信通信的代理可能会采取不安全的行为，导致灾难性的后果。因此，确保代理不会被损坏的通信误导，同时仍受益于良性通信是至关重要的。在这项工作中，我们考虑了一个具有$N$代理的环境，在该环境中，攻击者可以任意地将通信从任何$C<\frac{N-1}{2}$代理更改为受害者代理。对于这种强威胁模型，我们通过构造聚合多个随机消融消息集的消息集成策略来提出一种可证明的防御。理论分析表明，无论采用何种攻击算法，该消息集成策略都能充分利用良性通信，同时对敌意通信具有较强的鲁棒性。在多个环境中的实验证明，我们的防御显著提高了经过训练的策略对各种类型攻击的健壮性。



## **14. ProML: A Decentralised Platform for Provenance Management of Machine Learning Software Systems**

ProML：一种用于机器学习软件系统来源管理的分布式平台 cs.SE

Accepted as full paper in ECSA 2022 conference. To be presented

**SubmitDate**: 2022-06-21    [paper-pdf](http://arxiv.org/pdf/2206.10110v1)

**Authors**: Nguyen Khoi Tran, Bushra Sabir, M. Ali Babar, Nini Cui, Mehran Abolhasan, Justin Lipman

**Abstracts**: Large-scale Machine Learning (ML) based Software Systems are increasingly developed by distributed teams situated in different trust domains. Insider threats can launch attacks from any domain to compromise ML assets (models and datasets). Therefore, practitioners require information about how and by whom ML assets were developed to assess their quality attributes such as security, safety, and fairness. Unfortunately, it is challenging for ML teams to access and reconstruct such historical information of ML assets (ML provenance) because it is generally fragmented across distributed ML teams and threatened by the same adversaries that attack ML assets. This paper proposes ProML, a decentralised platform that leverages blockchain and smart contracts to empower distributed ML teams to jointly manage a single source of truth about circulated ML assets' provenance without relying on a third party, which is vulnerable to insider threats and presents a single point of failure. We propose a novel architectural approach called Artefact-as-a-State-Machine to leverage blockchain transactions and smart contracts for managing ML provenance information and introduce a user-driven provenance capturing mechanism to integrate existing scripts and tools to ProML without compromising participants' control over their assets and toolchains. We evaluate the performance and overheads of ProML by benchmarking a proof-of-concept system on a global blockchain. Furthermore, we assessed ProML's security against a threat model of a distributed ML workflow.

摘要: 基于大规模机器学习(ML)的软件系统越来越多地由分布在不同信任域的团队开发。内部威胁可以从任何域发起攻击，以危害ML资产(模型和数据集)。因此，实践者需要有关ML资产是如何以及由谁开发的信息，以评估其质量属性，如安全性、安全性和公平性。不幸的是，ML团队访问和重建这种ML资产的历史信息(ML起源)是具有挑战性的，因为这些信息通常分散在分散的ML团队中，并且受到攻击ML资产的相同对手的威胁。本文提出了ProML，这是一个分散的平台，利用区块链和智能合同来授权分布式ML团队联合管理关于流通的ML资产来源的单一真相来源，而不依赖于第三方，这容易受到内部威胁，并提供单点故障。我们提出了一种称为Arteact-as-State-Machine的新颖架构方法来利用区块链事务和智能合约来管理ML起源信息，并引入了用户驱动的起源捕获机制来将现有脚本和工具集成到ProML中，而不会损害参与者对其资产和工具链的控制。我们通过在全球区块链上对概念验证系统进行基准测试来评估ProML的性能和开销。此外，我们根据分布式ML工作流的威胁模型评估了ProML的安全性。



## **15. Make Some Noise: Reliable and Efficient Single-Step Adversarial Training**

制造一些噪音：可靠而高效的单步对抗性训练 cs.LG

**SubmitDate**: 2022-06-20    [paper-pdf](http://arxiv.org/pdf/2202.01181v2)

**Authors**: Pau de Jorge, Adel Bibi, Riccardo Volpi, Amartya Sanyal, Philip H. S. Torr, Grégory Rogez, Puneet K. Dokania

**Abstracts**: Recently, Wong et al. showed that adversarial training with single-step FGSM leads to a characteristic failure mode named catastrophic overfitting (CO), in which a model becomes suddenly vulnerable to multi-step attacks. They showed that adding a random perturbation prior to FGSM (RS-FGSM) seemed to be sufficient to prevent CO. However, Andriushchenko and Flammarion observed that RS-FGSM still leads to CO for larger perturbations, and proposed an expensive regularizer (GradAlign) to avoid CO. In this work, we methodically revisit the role of noise and clipping in single-step adversarial training. Contrary to previous intuitions, we find that using a stronger noise around the clean sample combined with not clipping is highly effective in avoiding CO for large perturbation radii. Based on these observations, we then propose Noise-FGSM (N-FGSM) that, while providing the benefits of single-step adversarial training, does not suffer from CO. Empirical analyses on a large suite of experiments show that N-FGSM is able to match or surpass the performance of previous single-step methods while achieving a 3$\times$ speed-up. Code can be found in https://github.com/pdejorge/N-FGSM

摘要: 最近，Wong et al.研究表明，采用单步FGSM的对抗性训练会导致一种称为灾难性过匹配(CO)的特征故障模式，在这种模式下，模型突然变得容易受到多步攻击。他们表明，在FGSM(RS-FGSM)之前增加随机扰动似乎足以防止CO。然而，Andriushchenko和Flammarion观察到，对于较大的扰动，RS-FGSM仍然会导致CO，并提出了一种昂贵的正则化(GradAlign)来避免CO。在这项工作中，我们有条不紊地重新审视噪声和剪辑在单步对抗性训练中的作用。与以前的直觉相反，我们发现在清洁样本周围使用更强的噪声结合不削波在大扰动半径下避免CO是非常有效的。基于这些观察，我们提出了Noise-FGSM(N-FGSM)，它在提供单步对抗性训练的好处的同时，不会受到CO的影响。大量实验的实验结果表明，N-FGSM在性能上达到或超过了以往单步算法的性能，同时获得了3倍于3倍的加速。代码可在https://github.com/pdejorge/N-FGSM中找到



## **16. Diversified Adversarial Attacks based on Conjugate Gradient Method**

基于共轭梯度法的多样化对抗性攻击 cs.LG

**SubmitDate**: 2022-06-20    [paper-pdf](http://arxiv.org/pdf/2206.09628v1)

**Authors**: Keiichiro Yamamura, Haruki Sato, Nariaki Tateiwa, Nozomi Hata, Toru Mitsutake, Issa Oe, Hiroki Ishikura, Katsuki Fujisawa

**Abstracts**: Deep learning models are vulnerable to adversarial examples, and adversarial attacks used to generate such examples have attracted considerable research interest. Although existing methods based on the steepest descent have achieved high attack success rates, ill-conditioned problems occasionally reduce their performance. To address this limitation, we utilize the conjugate gradient (CG) method, which is effective for this type of problem, and propose a novel attack algorithm inspired by the CG method, named the Auto Conjugate Gradient (ACG) attack. The results of large-scale evaluation experiments conducted on the latest robust models show that, for most models, ACG was able to find more adversarial examples with fewer iterations than the existing SOTA algorithm Auto-PGD (APGD). We investigated the difference in search performance between ACG and APGD in terms of diversification and intensification, and define a measure called Diversity Index (DI) to quantify the degree of diversity. From the analysis of the diversity using this index, we show that the more diverse search of the proposed method remarkably improves its attack success rate.

摘要: 深度学习模型容易受到对抗性实例的影响，而用于生成此类实例的对抗性攻击已经引起了相当大的研究兴趣。虽然现有的基于最陡下降的方法已经取得了很高的攻击成功率，但条件恶劣的问题有时会降低它们的性能。针对这一局限性，我们利用对这类问题有效的共轭梯度(CG)方法，并在CG方法的启发下提出了一种新的攻击算法，称为自动共轭梯度(ACG)攻击。在最新的稳健模型上进行的大规模评估实验结果表明，对于大多数模型，ACG能够以更少的迭代发现更多的对抗性实例，而不是现有的SOTA算法Auto-PGD(APGD)。我们研究了ACG和APGD在多样化和集约化方面的搜索性能差异，并定义了一个称为多样性指数(DI)的度量来量化多样性程度。从该指标的多样性分析可以看出，该方法搜索的多样性显著提高了其攻击成功率。



## **17. On the Limitations of Stochastic Pre-processing Defenses**

论随机前处理防御的局限性 cs.LG

**SubmitDate**: 2022-06-19    [paper-pdf](http://arxiv.org/pdf/2206.09491v1)

**Authors**: Yue Gao, Ilia Shumailov, Kassem Fawaz, Nicolas Papernot

**Abstracts**: Defending against adversarial examples remains an open problem. A common belief is that randomness at inference increases the cost of finding adversarial inputs. An example of such a defense is to apply a random transformation to inputs prior to feeding them to the model. In this paper, we empirically and theoretically investigate such stochastic pre-processing defenses and demonstrate that they are flawed. First, we show that most stochastic defenses are weaker than previously thought; they lack sufficient randomness to withstand even standard attacks like projected gradient descent. This casts doubt on a long-held assumption that stochastic defenses invalidate attacks designed to evade deterministic defenses and force attackers to integrate the Expectation over Transformation (EOT) concept. Second, we show that stochastic defenses confront a trade-off between adversarial robustness and model invariance; they become less effective as the defended model acquires more invariance to their randomization. Future work will need to decouple these two effects. Our code is available in the supplementary material.

摘要: 抵御敌意的例子仍然是一个悬而未决的问题。一种普遍的看法是，推理的随机性增加了寻找敌对输入的成本。这种防御的一个例子是在将输入提供给模型之前对它们应用随机转换。在本文中，我们从经验和理论上研究了这种随机预处理防御机制，并证明了它们是有缺陷的。首先，我们证明了大多数随机防御比之前认为的要弱；它们缺乏足够的随机性，即使是像投影梯度下降这样的标准攻击也是如此。这让人对一个长期持有的假设产生了怀疑，即随机防御使旨在逃避确定性防御的攻击无效，并迫使攻击者整合期望过转换(EOT)概念。其次，我们证明了随机防御面临着对抗稳健性和模型不变性之间的权衡；随着被防御模型对其随机化获得更多的不变性，它们变得不那么有效。未来的工作将需要将这两种影响脱钩。我们的代码可以在补充材料中找到。



## **18. A Universal Adversarial Policy for Text Classifiers**

一种适用于文本分类器的通用对抗策略 cs.LG

Accepted for publication in Neural Networks (2022), see  https://doi.org/10.1016/j.neunet.2022.06.018

**SubmitDate**: 2022-06-19    [paper-pdf](http://arxiv.org/pdf/2206.09458v1)

**Authors**: Gallil Maimon, Lior Rokach

**Abstracts**: Discovering the existence of universal adversarial perturbations had large theoretical and practical impacts on the field of adversarial learning. In the text domain, most universal studies focused on adversarial prefixes which are added to all texts. However, unlike the vision domain, adding the same perturbation to different inputs results in noticeably unnatural inputs. Therefore, we introduce a new universal adversarial setup - a universal adversarial policy, which has many advantages of other universal attacks but also results in valid texts - thus making it relevant in practice. We achieve this by learning a single search policy over a predefined set of semantics preserving text alterations, on many texts. This formulation is universal in that the policy is successful in finding adversarial examples on new texts efficiently. Our approach uses text perturbations which were extensively shown to produce natural attacks in the non-universal setup (specific synonym replacements). We suggest a strong baseline approach for this formulation which uses reinforcement learning. It's ability to generalise (from as few as 500 training texts) shows that universal adversarial patterns exist in the text domain as well.

摘要: 发现普遍存在的对抗性扰动对对抗性学习领域有很大的理论和实践影响。在语篇领域，大多数普遍的研究集中于添加到所有语篇中的对抗性前缀。然而，与视觉领域不同的是，将相同的扰动添加到不同的输入会导致明显不自然的输入。因此，我们引入了一种新的通用对抗设置-通用对抗策略，它具有其他通用攻击的许多优点，但也产生了有效的文本-从而使其在实践中具有相关性。我们通过在预定义的一组语义上学习单个搜索策略来实现这一点，该语义集保留了对许多文本的文本更改。这一提法具有普遍性，因为该政策成功地在新文本上有效地找到了对抗性的例子。我们的方法使用文本扰动，这被广泛显示为在非通用设置(特定同义词替换)中产生自然攻击。我们为这种使用强化学习的公式建议了一个强大的基线方法。它的泛化能力(从短短500个训练文本)表明，普遍的对抗性模式也存在于文本领域。



## **19. JPEG Compression-Resistant Low-Mid Adversarial Perturbation against Unauthorized Face Recognition System**

抗JPEG压缩的非授权人脸识别系统的中低端对抗性扰动 cs.CV

**SubmitDate**: 2022-06-19    [paper-pdf](http://arxiv.org/pdf/2206.09410v1)

**Authors**: Jiaming Zhang, Qi Yi, Jitao Sang

**Abstracts**: It has been observed that the unauthorized use of face recognition system raises privacy problems. Using adversarial perturbations provides one possible solution to address this issue. A critical issue to exploit adversarial perturbation against unauthorized face recognition system is that: The images uploaded to the web need to be processed by JPEG compression, which weakens the effectiveness of adversarial perturbation. Existing JPEG compression-resistant methods fails to achieve a balance among compression resistance, transferability, and attack effectiveness. To this end, we propose a more natural solution called low frequency adversarial perturbation (LFAP). Instead of restricting the adversarial perturbations, we turn to regularize the source model to employing more low-frequency features by adversarial training. Moreover, to better influence model in different frequency components, we proposed the refined low-mid frequency adversarial perturbation (LMFAP) considering the mid frequency components as the productive complement. We designed a variety of settings in this study to simulate the real-world application scenario, including cross backbones, supervisory heads, training datasets and testing datasets. Quantitative and qualitative experimental results validate the effectivenss of proposed solutions.

摘要: 据观察，未经授权使用人脸识别系统会带来隐私问题。使用对抗性扰动为解决这一问题提供了一种可能的解决方案。利用敌意扰动对抗未经授权的人脸识别系统的一个关键问题是：上传到网络的图像需要进行JPEG压缩处理，这削弱了对抗扰动的有效性。现有的JPEG抗压缩方法不能在抗压缩、可转移性和攻击有效性之间取得平衡。为此，我们提出了一种更自然的解决方案，称为低频对抗性扰动(LFAP)。我们没有限制对抗性扰动，而是通过对抗性训练来规则化信源模型以使用更多的低频特征。此外，为了更好地影响模型对不同频率成分的影响，我们提出了以中频成分作为产生性补充的精化中低频对抗扰动(LMFAP)。在本研究中，我们设计了多种设置来模拟真实世界的应用场景，包括交叉骨干、主管、训练数据集和测试数据集。定量和定性实验结果验证了所提出的解决方案的有效性。



## **20. Towards Adversarial Attack on Vision-Language Pre-training Models**

视觉语言预训练模型的对抗性攻击 cs.LG

**SubmitDate**: 2022-06-19    [paper-pdf](http://arxiv.org/pdf/2206.09391v1)

**Authors**: Jiaming Zhang, Qi Yi, Jitao Sang

**Abstracts**: While vision-language pre-training model (VLP) has shown revolutionary improvements on various vision-language (V+L) tasks, the studies regarding its adversarial robustness remain largely unexplored. This paper studied the adversarial attack on popular VLP models and V+L tasks. First, we analyzed the performance of adversarial attacks under different settings. By examining the influence of different perturbed objects and attack targets, we concluded some key observations as guidance on both designing strong multimodal adversarial attack and constructing robust VLP models. Second, we proposed a novel multimodal attack method on the VLP models called Collaborative Multimodal Adversarial Attack (Co-Attack), which collectively carries out the attacks on the image modality and the text modality. Experimental results demonstrated that the proposed method achieves improved attack performances on different V+L downstream tasks and VLP models. The analysis observations and novel attack method hopefully provide new understanding into the adversarial robustness of VLP models, so as to contribute their safe and reliable deployment in more real-world scenarios.

摘要: 虽然视觉-语言预训练模型(VLP)在各种视觉-语言(V+L)任务上有了革命性的改进，但关于其对抗健壮性的研究仍然很少。研究了对流行的VLP模型和V+L任务的对抗性攻击。首先，我们分析了不同环境下对抗性攻击的性能。通过考察不同扰动对象和攻击目标的影响，我们总结了一些关键的观察结果，作为设计强多通道对抗性攻击和构建稳健VLP模型的指导。其次，我们提出了一种新的针对VLP模型的多模式攻击方法，称为协作式多模式对抗攻击(Co-Attack)，它共同对图像通道和文本通道进行攻击。实验结果表明，该方法在不同的V+L下游任务和VLP模型下均能获得较好的攻击性能。分析观察和新颖的攻击方法有望对VLP模型的对抗健壮性提供新的理解，从而有助于在更真实的场景中安全可靠地部署VLP模型。



## **21. Adversarially trained neural representations may already be as robust as corresponding biological neural representations**

反向训练的神经表征可能已经和相应的生物神经表征一样健壮 q-bio.NC

10 pages, 6 figures, ICML2022

**SubmitDate**: 2022-06-19    [paper-pdf](http://arxiv.org/pdf/2206.11228v1)

**Authors**: Chong Guo, Michael J. Lee, Guillaume Leclerc, Joel Dapello, Yug Rao, Aleksander Madry, James J. DiCarlo

**Abstracts**: Visual systems of primates are the gold standard of robust perception. There is thus a general belief that mimicking the neural representations that underlie those systems will yield artificial visual systems that are adversarially robust. In this work, we develop a method for performing adversarial visual attacks directly on primate brain activity. We then leverage this method to demonstrate that the above-mentioned belief might not be well founded. Specifically, we report that the biological neurons that make up visual systems of primates exhibit susceptibility to adversarial perturbations that is comparable in magnitude to existing (robustly trained) artificial neural networks.

摘要: 灵长类动物的视觉系统是强健感知的黄金标准。因此，人们普遍认为，模仿构成这些系统的神经表示将产生相反的健壮的人工视觉系统。在这项工作中，我们开发了一种直接对灵长类大脑活动进行对抗性视觉攻击的方法。然后，我们利用这种方法来证明上述信念可能没有很好的依据。具体地说，我们报告了组成灵长类视觉系统的生物神经元对对抗性扰动的敏感性，其大小与现有的(稳健训练的)人工神经网络相当。



## **22. Efficient and Transferable Adversarial Examples from Bayesian Neural Networks**

贝叶斯神经网络中高效且可移植的对抗性实例 cs.LG

Accepted at UAI 2022

**SubmitDate**: 2022-06-18    [paper-pdf](http://arxiv.org/pdf/2011.05074v4)

**Authors**: Martin Gubri, Maxime Cordy, Mike Papadakis, Yves Le Traon, Koushik Sen

**Abstracts**: An established way to improve the transferability of black-box evasion attacks is to craft the adversarial examples on an ensemble-based surrogate to increase diversity. We argue that transferability is fundamentally related to uncertainty. Based on a state-of-the-art Bayesian Deep Learning technique, we propose a new method to efficiently build a surrogate by sampling approximately from the posterior distribution of neural network weights, which represents the belief about the value of each parameter. Our extensive experiments on ImageNet, CIFAR-10 and MNIST show that our approach improves the success rates of four state-of-the-art attacks significantly (up to 83.2 percentage points), in both intra-architecture and inter-architecture transferability. On ImageNet, our approach can reach 94% of success rate while reducing training computations from 11.6 to 2.4 exaflops, compared to an ensemble of independently trained DNNs. Our vanilla surrogate achieves 87.5% of the time higher transferability than three test-time techniques designed for this purpose. Our work demonstrates that the way to train a surrogate has been overlooked, although it is an important element of transfer-based attacks. We are, therefore, the first to review the effectiveness of several training methods in increasing transferability. We provide new directions to better understand the transferability phenomenon and offer a simple but strong baseline for future work.

摘要: 提高黑盒逃避攻击可转移性的一种既定方法是在基于集成的代理上精心制作对抗性示例，以增加多样性。我们认为，可转让性从根本上与不确定性有关。基于最新的贝叶斯深度学习技术，我们提出了一种新的方法，通过对神经网络权值的后验分布进行近似采样来有效地构建代理，该后验分布代表了对每个参数的值的信念。我们在ImageNet、CIFAR-10和MNIST上的广泛实验表明，我们的方法显著提高了四种最先进攻击的成功率(高达83.2个百分点)，在架构内和架构间的可转移性方面都是如此。在ImageNet上，与独立训练的DNN集成相比，我们的方法可以达到94%的成功率，同时将训练计算量从11.6exaflop减少到2.4exaflop。我们的香草代理在87.5%的时间内实现了比为此目的而设计的三种测试时间技术更高的可转移性。我们的工作表明，训练代理的方法被忽视了，尽管它是基于传输的攻击的一个重要元素。因此，我们第一次审查了几种培训方法在提高可转移性方面的有效性。我们为更好地理解可转移性现象提供了新的方向，并为未来的工作提供了一个简单但强有力的基线。



## **23. DECK: Model Hardening for Defending Pervasive Backdoors**

甲板：用于防御无处不在的后门的模型强化 cs.CR

**SubmitDate**: 2022-06-18    [paper-pdf](http://arxiv.org/pdf/2206.09272v1)

**Authors**: Guanhong Tao, Yingqi Liu, Siyuan Cheng, Shengwei An, Zhuo Zhang, Qiuling Xu, Guangyu Shen, Xiangyu Zhang

**Abstracts**: Pervasive backdoors are triggered by dynamic and pervasive input perturbations. They can be intentionally injected by attackers or naturally exist in normally trained models. They have a different nature from the traditional static and localized backdoors that can be triggered by perturbing a small input area with some fixed pattern, e.g., a patch with solid color. Existing defense techniques are highly effective for traditional backdoors. However, they may not work well for pervasive backdoors, especially regarding backdoor removal and model hardening. In this paper, we propose a novel model hardening technique against pervasive backdoors, including both natural and injected backdoors. We develop a general pervasive attack based on an encoder-decoder architecture enhanced with a special transformation layer. The attack can model a wide range of existing pervasive backdoor attacks and quantify them by class distances. As such, using the samples derived from our attack in adversarial training can harden a model against these backdoor vulnerabilities. Our evaluation on 9 datasets with 15 model structures shows that our technique can enlarge class distances by 59.65% on average with less than 1% accuracy degradation and no robustness loss, outperforming five hardening techniques such as adversarial training, universal adversarial training, MOTH, etc. It can reduce the attack success rate of six pervasive backdoor attacks from 99.06% to 1.94%, surpassing seven state-of-the-art backdoor removal techniques.

摘要: 无处不在的后门由动态和无处不在的输入扰动触发。它们可能是攻击者故意注入的，也可能是正常训练的模型中自然存在的。它们与传统的静态和本地化后门不同，后者可以通过以某种固定图案干扰小输入区域来触发，例如，具有纯色的补丁。现有的防御技术对传统的后门非常有效。然而，它们可能不适用于普遍存在的后门，特别是在后门删除和模型强化方面。在这篇文章中，我们提出了一种新的模型硬化技术，针对普遍存在的后门，包括自然后门和注入后门。我们开发了一种通用的普适攻击，该攻击基于一种编解码器体系结构，并通过特殊的转换层进行了增强。该攻击可以模拟一系列现有的普遍存在的后门攻击，并根据类别距离对它们进行量化。因此，在对抗性训练中使用从我们的攻击中获得的样本可以加强针对这些后门漏洞的模型。在15个模型结构的9个数据集上的测试结果表明，该技术在准确率和健壮性没有损失的情况下，类距离平均扩展59.65%，优于对抗性训练、万能对抗性训练、MOST等5种强化技术，将6种普遍存在的后门攻击的攻击成功率从99.06%降低到1.94%，超过了7种最先进的后门移除技术。



## **24. On the Role of Generalization in Transferability of Adversarial Examples**

论概括在对抗性例句可转移性中的作用 cs.LG

**SubmitDate**: 2022-06-18    [paper-pdf](http://arxiv.org/pdf/2206.09238v1)

**Authors**: Yilin Wang, Farzan Farnia

**Abstracts**: Black-box adversarial attacks designing adversarial examples for unseen neural networks (NNs) have received great attention over the past years. While several successful black-box attack schemes have been proposed in the literature, the underlying factors driving the transferability of black-box adversarial examples still lack a thorough understanding. In this paper, we aim to demonstrate the role of the generalization properties of the substitute classifier used for generating adversarial examples in the transferability of the attack scheme to unobserved NN classifiers. To do this, we apply the max-min adversarial example game framework and show the importance of the generalization properties of the substitute NN in the success of the black-box attack scheme in application to different NN classifiers. We prove theoretical generalization bounds on the difference between the attack transferability rates on training and test samples. Our bounds suggest that a substitute NN with better generalization behavior could result in more transferable adversarial examples. In addition, we show that standard operator norm-based regularization methods could improve the transferability of the designed adversarial examples. We support our theoretical results by performing several numerical experiments showing the role of the substitute network's generalization in generating transferable adversarial examples. Our empirical results indicate the power of Lipschitz regularization methods in improving the transferability of adversarial examples.

摘要: 针对看不见的神经网络设计对抗性实例的黑盒对抗性攻击在过去的几年里受到了极大的关注。虽然文献中已经提出了几个成功的黑盒攻击方案，但驱动黑盒对抗性例子可转换性的潜在因素仍然缺乏深入的了解。在本文中，我们的目的是证明用于生成对抗性示例的替换分类器的泛化性质在攻击方案向不可观测的NN分类器的可转移性中所起的作用。为此，我们应用了最大-最小对抗性例子博弈框架，并展示了替代神经网络的泛化性质在黑盒攻击方案应用于不同的神经网络分类器中的重要性。我们证明了训练样本和测试样本上的攻击可转移率之间的差异的理论概括界。我们的界限表明，具有更好泛化行为的替代NN可以产生更多可转移的对抗性实例。此外，我们还证明了基于标准算子范数的正则化方法可以提高所设计的对抗性实例的可转移性。我们通过几个数值实验来支持我们的理论结果，这些实验显示了替代网络的泛化在生成可转移的对抗性例子中的作用。我们的实证结果表明，Lipschitz正则化方法在提高对抗性例子的可转移性方面是有效的。



## **25. Measuring Lower Bounds of Local Differential Privacy via Adversary Instantiations in Federated Learning**

联合学习中通过敌意实例化测量局部差分隐私的下界 cs.CR

15 pages, 7 figures

**SubmitDate**: 2022-06-18    [paper-pdf](http://arxiv.org/pdf/2206.09122v1)

**Authors**: Marin Matsumoto, Tsubasa Takahashi, Seng Pei Liew, Masato Oguchi

**Abstracts**: Local differential privacy (LDP) gives a strong privacy guarantee to be used in a distributed setting like federated learning (FL). LDP mechanisms in FL protect a client's gradient by randomizing it on the client; however, how can we interpret the privacy level given by the randomization? Moreover, what types of attacks can we mitigate in practice? To answer these questions, we introduce an empirical privacy test by measuring the lower bounds of LDP. The privacy test estimates how an adversary predicts if a reported randomized gradient was crafted from a raw gradient $g_1$ or $g_2$. We then instantiate six adversaries in FL under LDP to measure empirical LDP at various attack surfaces, including a worst-case attack that reaches the theoretical upper bound of LDP. The empirical privacy test with the adversary instantiations enables us to interpret LDP more intuitively and discuss relaxation of the privacy parameter until a particular instantiated attack surfaces. We also demonstrate numerical observations of the measured privacy in these adversarial settings, and the worst-case attack is not realistic in FL. In the end, we also discuss the possible relaxation of privacy levels in FL under LDP.

摘要: 局部差异隐私(LDP)为联邦学习(FL)等分布式环境下的应用提供了强有力的隐私保障。FL中的LDP机制通过将客户端上的梯度随机化来保护客户端的梯度；然而，我们如何解释随机化所提供的隐私级别？此外，我们在实践中可以减轻哪些类型的攻击？为了回答这些问题，我们引入了一项经验隐私测试，通过测量自民党的下限。隐私测试估计对手如何预测所报告的随机梯度是从原始梯度$g_1$还是$g_2$创建的。然后，我们在LDP下实例化了FL中的六个对手，以测量不同攻击面上的经验LDP，包括达到LDP理论上限的最坏情况攻击。使用对手实例的经验隐私测试使我们能够更直观地解释LDP，并讨论隐私参数的放松，直到特定的实例化攻击浮出水面。我们还展示了在这些对抗性设置下测量的隐私的数值观测，并且最坏情况下的攻击在FL中是不现实的。最后，我们还讨论了LDP下FL隐私级别放宽的可能性。



## **26. Existence and Minimax Theorems for Adversarial Surrogate Risks in Binary Classification**

二元分类中对抗性代理风险的存在性和极大极小定理 cs.LG

37 pages, 1 Figure

**SubmitDate**: 2022-06-18    [paper-pdf](http://arxiv.org/pdf/2206.09098v1)

**Authors**: Natalie S. Frank

**Abstracts**: Adversarial training is one of the most popular methods for training methods robust to adversarial attacks, however, it is not well-understood from a theoretical perspective. We prove and existence, regularity, and minimax theorems for adversarial surrogate risks. Our results explain some empirical observations on adversarial robustness from prior work and suggest new directions in algorithm development. Furthermore, our results extend previously known existence and minimax theorems for the adversarial classification risk to surrogate risks.

摘要: 对抗性训练是对抗攻击能力最强的训练方法之一，但从理论上对它的理解还不够深入。我们证明了对抗性代理风险的存在性、正则性和极大极小定理。我们的结果解释了以前工作中关于对手稳健性的一些经验观察，并为算法开发提供了新的方向。此外，我们的结果推广了已知的对抗性分类风险到代理风险的存在性和极大极小定理。



## **27. Comment on Transferability and Input Transformation with Additive Noise**

加性噪声条件下的可转移性与输入变换 cs.LG

**SubmitDate**: 2022-06-18    [paper-pdf](http://arxiv.org/pdf/2206.09075v1)

**Authors**: Hoki Kim, Jinseong Park, Jaewook Lee

**Abstracts**: Adversarial attacks have verified the existence of the vulnerability of neural networks. By adding small perturbations to a benign example, adversarial attacks successfully generate adversarial examples that lead misclassification of deep learning models. More importantly, an adversarial example generated from a specific model can also deceive other models without modification. We call this phenomenon ``transferability". Here, we analyze the relationship between transferability and input transformation with additive noise by mathematically proving that the modified optimization can produce more transferable adversarial examples.

摘要: 对抗性攻击验证了神经网络脆弱性的存在。通过在良性示例中添加小的扰动，对抗性攻击成功地生成了导致深度学习模型错误分类的对抗性示例。更重要的是，从特定模型生成的对抗性示例也可以在不修改的情况下欺骗其他模型。我们称这种现象为可转移性。在这里，我们通过数学证明改进的最优化方法可以产生更多可转移性的对抗性例子，分析了可转移性与加性噪声的输入变换之间的关系。



## **28. Learning Generative Deception Strategies in Combinatorial Masking Games**

在组合掩饰博弈中学习生成性欺骗策略 cs.GT

GameSec 2021

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2109.11637v2)

**Authors**: Junlin Wu, Charles Kamhoua, Murat Kantarcioglu, Yevgeniy Vorobeychik

**Abstracts**: Deception is a crucial tool in the cyberdefence repertoire, enabling defenders to leverage their informational advantage to reduce the likelihood of successful attacks. One way deception can be employed is through obscuring, or masking, some of the information about how systems are configured, increasing attacker's uncertainty about their targets. We present a novel game-theoretic model of the resulting defender-attacker interaction, where the defender chooses a subset of attributes to mask, while the attacker responds by choosing an exploit to execute. The strategies of both players have combinatorial structure with complex informational dependencies, and therefore even representing these strategies is not trivial. First, we show that the problem of computing an equilibrium of the resulting zero-sum defender-attacker game can be represented as a linear program with a combinatorial number of system configuration variables and constraints, and develop a constraint generation approach for solving this problem. Next, we present a novel highly scalable approach for approximately solving such games by representing the strategies of both players as neural networks. The key idea is to represent the defender's mixed strategy using a deep neural network generator, and then using alternating gradient-descent-ascent algorithm, analogous to the training of Generative Adversarial Networks. Our experiments, as well as a case study, demonstrate the efficacy of the proposed approach.

摘要: 欺骗是网络防御体系中的一个重要工具，使防御者能够利用他们的信息优势来降低攻击成功的可能性。欺骗的一种方式是通过模糊或掩盖有关系统配置的一些信息，增加攻击者对目标的不确定性。我们提出了一个新的博弈论模型来描述由此产生的防御者和攻击者的相互作用，其中防御者选择掩蔽属性的子集，而攻击者通过选择一个利用漏洞来执行。两个参与者的策略都具有组合结构，具有复杂的信息依赖关系，因此即使是表示这些策略也不是微不足道的。首先，我们证明了由此产生的零和防御者-攻击者博弈的均衡计算问题可以表示为具有多个系统配置变量和约束的组合数量的线性规划，并给出了一种求解该问题的约束生成方法。接下来，我们提出了一种新的高度可扩展的方法，通过将双方的策略表示为神经网络来近似求解这类对策。其核心思想是用一个深度神经网络生成器来表示防御者的混合策略，然后使用交替的梯度-下降-上升算法，类似于生成式对手网络的训练。我们的实验和一个案例研究证明了该方法的有效性。



## **29. RetrievalGuard: Provably Robust 1-Nearest Neighbor Image Retrieval**

RetrievalGuard：可证明健壮的1-近邻图像检索 cs.IR

accepted by ICML 2022

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2206.11225v1)

**Authors**: Yihan Wu, Hongyang Zhang, Heng Huang

**Abstracts**: Recent research works have shown that image retrieval models are vulnerable to adversarial attacks, where slightly modified test inputs could lead to problematic retrieval results. In this paper, we aim to design a provably robust image retrieval model which keeps the most important evaluation metric Recall@1 invariant to adversarial perturbation. We propose the first 1-nearest neighbor (NN) image retrieval algorithm, RetrievalGuard, which is provably robust against adversarial perturbations within an $\ell_2$ ball of calculable radius. The challenge is to design a provably robust algorithm that takes into consideration the 1-NN search and the high-dimensional nature of the embedding space. Algorithmically, given a base retrieval model and a query sample, we build a smoothed retrieval model by carefully analyzing the 1-NN search procedure in the high-dimensional embedding space. We show that the smoothed retrieval model has bounded Lipschitz constant and thus the retrieval score is invariant to $\ell_2$ adversarial perturbations. Experiments on image retrieval tasks validate the robustness of our RetrievalGuard method.

摘要: 最近的研究工作表明，图像检索模型容易受到对抗性攻击，其中稍加修改的测试输入可能会导致有问题的检索结果。在本文中，我们的目标是设计一个可证明稳健的图像检索模型，使最重要的评价指标recall@1对对手的扰动保持不变。我们提出了第一个1-近邻(NN)图像检索算法RetrivalGuard，该算法在半径可计算的$ell2$球内对敌意扰动具有证明的健壮性。挑战在于设计一种可证明稳健的算法，该算法考虑到1-NN搜索和嵌入空间的高维性质。在算法上，给出了一个基本的检索模型和一个查询样本，通过仔细分析高维嵌入空间中的1-NN搜索过程，建立了一个平滑的检索模型。我们证明了平滑的检索模型具有有界的Lipschitz常数，因此检索分数对于$\ell_2$对抗性扰动是不变的。在图像检索任务上的实验验证了该方法的健壮性。



## **30. Is Multi-Modal Necessarily Better? Robustness Evaluation of Multi-modal Fake News Detection**

多式联运一定会更好吗？多模式假新闻检测的稳健性评价 cs.AI

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2206.08788v1)

**Authors**: Jinyin Chen, Chengyu Jia, Haibin Zheng, Ruoxi Chen, Chenbo Fu

**Abstracts**: The proliferation of fake news and its serious negative social influence push fake news detection methods to become necessary tools for web managers. Meanwhile, the multi-media nature of social media makes multi-modal fake news detection popular for its ability to capture more modal features than uni-modal detection methods. However, current literature on multi-modal detection is more likely to pursue the detection accuracy but ignore the robustness of the detector. To address this problem, we propose a comprehensive robustness evaluation of multi-modal fake news detectors. In this work, we simulate the attack methods of malicious users and developers, i.e., posting fake news and injecting backdoors. Specifically, we evaluate multi-modal detectors with five adversarial and two backdoor attack methods. Experiment results imply that: (1) The detection performance of the state-of-the-art detectors degrades significantly under adversarial attacks, even worse than general detectors; (2) Most multi-modal detectors are more vulnerable when subjected to attacks on visual modality than textual modality; (3) Popular events' images will cause significant degradation to the detectors when they are subjected to backdoor attacks; (4) The performance of these detectors under multi-modal attacks is worse than under uni-modal attacks; (5) Defensive methods will improve the robustness of the multi-modal detectors.

摘要: 假新闻的泛滥及其严重的负面社会影响，促使假新闻检测方法成为网络管理者的必备工具。同时，社交媒体的多媒体特性使得多模式假新闻检测因其能够捕捉到比单模式检测方法更多的模式特征而广受欢迎。然而，目前关于多模式检测的文献更倾向于追求检测精度，而忽略了检测器的稳健性。针对这一问题，我们提出了一种多模式假新闻检测器的综合稳健性评估方法。在这项工作中，我们模拟了恶意用户和开发者的攻击方法，即发布假新闻和注入后门。具体地说，我们使用五种对抗性攻击方法和两种后门攻击方法来评估多模式检测器。实验结果表明：(1)现有检测器在对抗性攻击下检测性能显著下降，甚至比一般检测器更差；(2)大多数多模式检测器在受到视觉通道攻击时比文本通道更容易受到攻击；(3)热门事件的图像在受到后门攻击时会导致检测器性能显著下降；(4)多模式检测器在多模式攻击下的性能比单模式攻击下的性能差；(5)防御方法将提高多模式检测器的健壮性。



## **31. Detecting Adversarial Examples in Batches -- a geometrical approach**

批量检测对抗性样本--一种几何方法 cs.LG

Submitted to AdvML workshop at ICML2022

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2206.08738v1)

**Authors**: Danush Kumar Venkatesh, Peter Steinbach

**Abstracts**: Many deep learning methods have successfully solved complex tasks in computer vision and speech recognition applications. Nonetheless, the robustness of these models has been found to be vulnerable to perturbed inputs or adversarial examples, which are imperceptible to the human eye, but lead the model to erroneous output decisions. In this study, we adapt and introduce two geometric metrics, density and coverage, and evaluate their use in detecting adversarial samples in batches of unseen data. We empirically study these metrics using MNIST and two real-world biomedical datasets from MedMNIST, subjected to two different adversarial attacks. Our experiments show promising results for both metrics to detect adversarial examples. We believe that his work can lay the ground for further study on these metrics' use in deployed machine learning systems to monitor for possible attacks by adversarial examples or related pathologies such as dataset shift.

摘要: 许多深度学习方法已经成功地解决了计算机视觉和语音识别应用中的复杂任务。尽管如此，这些模型的稳健性已被发现容易受到扰动输入或对抗性例子的影响，这些输入或对抗性例子是人眼无法察觉的，但会导致模型做出错误的输出决定。在这项研究中，我们采用并引入了密度和覆盖率这两个几何度量，并评估了它们在批量未见数据中检测敌意样本的应用。我们使用MNIST和来自MedMNIST的两个真实世界生物医学数据集对这些度量进行了经验性研究，这些数据集受到两种不同的对抗性攻击。我们的实验表明，这两个度量在检测敌意示例方面都取得了很好的结果。我们相信，他的工作可以为进一步研究这些度量在已部署的机器学习系统中的使用奠定基础，以监控来自对抗性示例或相关病理(如数据集移动)的可能攻击。



## **32. Minimum Noticeable Difference based Adversarial Privacy Preserving Image Generation**

基于最小显著差的对抗性隐私保护图像生成 cs.CV

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2206.08638v1)

**Authors**: Wen Sun, Jian Jin, Weisi Lin

**Abstracts**: Deep learning models are found to be vulnerable to adversarial examples, as wrong predictions can be caused by small perturbation in input for deep learning models. Most of the existing works of adversarial image generation try to achieve attacks for most models, while few of them make efforts on guaranteeing the perceptual quality of the adversarial examples. High quality adversarial examples matter for many applications, especially for the privacy preserving. In this work, we develop a framework based on the Minimum Noticeable Difference (MND) concept to generate adversarial privacy preserving images that have minimum perceptual difference from the clean ones but are able to attack deep learning models. To achieve this, an adversarial loss is firstly proposed to make the deep learning models attacked by the adversarial images successfully. Then, a perceptual quality-preserving loss is developed by taking the magnitude of perturbation and perturbation-caused structural and gradient changes into account, which aims to preserve high perceptual quality for adversarial image generation. To the best of our knowledge, this is the first work on exploring quality-preserving adversarial image generation based on the MND concept for privacy preserving. To evaluate its performance in terms of perceptual quality, the deep models on image classification and face recognition are tested with the proposed method and several anchor methods in this work. Extensive experimental results demonstrate that the proposed MND framework is capable of generating adversarial images with remarkably improved performance metrics (e.g., PSNR, SSIM, and MOS) than that generated with the anchor methods.

摘要: 深度学习模型被发现容易受到敌意例子的影响，因为深度学习模型的输入中的微小扰动可能会导致错误的预测。现有的对抗性图像生成工作大多试图实现对大多数模型的攻击，而很少在保证对抗性实例的感知质量上下功夫。高质量的对抗性例子对许多应用都很重要，特别是对于隐私保护。在这项工作中，我们开发了一个基于最小显著差异(MND)概念的框架，以生成与干净图像具有最小感知差异但能够攻击深度学习模型的对抗性隐私保护图像。为此，首先提出了对抗性损失的概念，使得深度学习模型能够成功地受到对抗性图像的攻击。然后，通过考虑扰动的大小和扰动引起的结构和梯度变化的大小，提出了一种保持感知质量的损失，目的是为了在对抗性图像生成中保持高感知质量。据我们所知，这是探索基于MND概念的隐私保护的质量保持的敌意图像生成的第一个工作。为了从感知质量的角度评价其性能，本文用该方法和几种锚定方法对图像分类和人脸识别中的深层模型进行了测试。大量的实验结果表明，MND框架生成的敌意图像的性能指标(如PSNR、SSIM和MOS)明显优于锚定方法生成的图像。



## **33. Query-Efficient and Scalable Black-Box Adversarial Attacks on Discrete Sequential Data via Bayesian Optimization**

基于贝叶斯优化的查询高效、可扩展的离散序列数据黑盒攻击 cs.LG

ICML 2022; Codes at  https://github.com/snu-mllab/DiscreteBlockBayesAttack

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2206.08575v1)

**Authors**: Deokjae Lee, Seungyong Moon, Junhyeok Lee, Hyun Oh Song

**Abstracts**: We focus on the problem of adversarial attacks against models on discrete sequential data in the black-box setting where the attacker aims to craft adversarial examples with limited query access to the victim model. Existing black-box attacks, mostly based on greedy algorithms, find adversarial examples using pre-computed key positions to perturb, which severely limits the search space and might result in suboptimal solutions. To this end, we propose a query-efficient black-box attack using Bayesian optimization, which dynamically computes important positions using an automatic relevance determination (ARD) categorical kernel. We introduce block decomposition and history subsampling techniques to improve the scalability of Bayesian optimization when an input sequence becomes long. Moreover, we develop a post-optimization algorithm that finds adversarial examples with smaller perturbation size. Experiments on natural language and protein classification tasks demonstrate that our method consistently achieves higher attack success rate with significant reduction in query count and modification rate compared to the previous state-of-the-art methods.

摘要: 我们主要研究了在黑盒环境下对离散序列数据上的模型进行对抗性攻击的问题，其中攻击者的目标是在对受害者模型的查询访问受限的情况下创建对抗性示例。现有的黑盒攻击大多基于贪婪算法，使用预先计算的关键位置来扰动寻找敌意实例，这严重限制了搜索空间，并可能导致次优解。为此，我们提出了一种基于贝叶斯优化的查询高效黑盒攻击，该攻击使用自动相关性确定(ARD)分类核来动态计算重要位置。我们引入了块分解和历史次抽样技术来提高贝叶斯优化在输入序列变长时的可伸缩性。此外，我们还开发了一种后优化算法，该算法可以找到扰动规模较小的对抗性实例。在自然语言和蛋白质分类任务上的实验表明，与以前的方法相比，该方法在查询次数和修改率方面都有明显的降低，并获得了更高的攻击成功率。



## **34. Adversarial Attack and Defense for Non-Parametric Two-Sample Tests**

非参数两样本检验的对抗性攻防 cs.LG

Accepted by ICML 2022

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2202.03077v2)

**Authors**: Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**Abstracts**: Non-parametric two-sample tests (TSTs) that judge whether two sets of samples are drawn from the same distribution, have been widely used in the analysis of critical data. People tend to employ TSTs as trusted basic tools and rarely have any doubt about their reliability. This paper systematically uncovers the failure mode of non-parametric TSTs through adversarial attacks and then proposes corresponding defense strategies. First, we theoretically show that an adversary can upper-bound the distributional shift which guarantees the attack's invisibility. Furthermore, we theoretically find that the adversary can also degrade the lower bound of a TST's test power, which enables us to iteratively minimize the test criterion in order to search for adversarial pairs. To enable TST-agnostic attacks, we propose an ensemble attack (EA) framework that jointly minimizes the different types of test criteria. Second, to robustify TSTs, we propose a max-min optimization that iteratively generates adversarial pairs to train the deep kernels. Extensive experiments on both simulated and real-world datasets validate the adversarial vulnerabilities of non-parametric TSTs and the effectiveness of our proposed defense. Source code is available at https://github.com/GodXuxilie/Robust-TST.git.

摘要: 非参数两样本检验(TSTs)是判断两组样本是否来自同一分布的检验方法，在关键数据的分析中得到了广泛的应用。人们倾向于使用TST作为受信任的基本工具，并且很少对其可靠性有任何怀疑。系统地揭示了非参数TSTs通过对抗性攻击的失效模式，并提出了相应的防御策略。首先，我们从理论上证明了敌手可以在保证攻击不可见性的分布移位上界。此外，我们从理论上发现，敌手也可以降低TST测试功率的下界，这使得我们能够迭代最小化测试标准，以搜索对手对。为了支持与TST无关的攻击，我们提出了一个联合最小化不同类型测试标准的集成攻击(EA)框架。其次，为了增强TSTs的健壮性，我们提出了一种最大-最小优化算法，该算法迭代地生成敌意对来训练深层核。在模拟数据集和真实数据集上的大量实验验证了非参数TST的对抗性漏洞和我们所提出的防御的有效性。源代码可在https://github.com/GodXuxilie/Robust-TST.git.上获得



## **35. A Unified Evaluation of Textual Backdoor Learning: Frameworks and Benchmarks**

文本后门学习的统一评价：框架和基准 cs.LG

19 pages

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2206.08514v1)

**Authors**: Ganqu Cui, Lifan Yuan, Bingxiang He, Yangyi Chen, Zhiyuan Liu, Maosong Sun

**Abstracts**: Textual backdoor attacks are a kind of practical threat to NLP systems. By injecting a backdoor in the training phase, the adversary could control model predictions via predefined triggers. As various attack and defense models have been proposed, it is of great significance to perform rigorous evaluations. However, we highlight two issues in previous backdoor learning evaluations: (1) The differences between real-world scenarios (e.g. releasing poisoned datasets or models) are neglected, and we argue that each scenario has its own constraints and concerns, thus requires specific evaluation protocols; (2) The evaluation metrics only consider whether the attacks could flip the models' predictions on poisoned samples and retain performances on benign samples, but ignore that poisoned samples should also be stealthy and semantic-preserving. To address these issues, we categorize existing works into three practical scenarios in which attackers release datasets, pre-trained models, and fine-tuned models respectively, then discuss their unique evaluation methodologies. On metrics, to completely evaluate poisoned samples, we use grammar error increase and perplexity difference for stealthiness, along with text similarity for validity. After formalizing the frameworks, we develop an open-source toolkit OpenBackdoor to foster the implementations and evaluations of textual backdoor learning. With this toolkit, we perform extensive experiments to benchmark attack and defense models under the suggested paradigm. To facilitate the underexplored defenses against poisoned datasets, we further propose CUBE, a simple yet strong clustering-based defense baseline. We hope that our frameworks and benchmarks could serve as the cornerstones for future model development and evaluations.

摘要: 文本后门攻击是对NLP系统的一种实际威胁。通过在训练阶段插入后门，对手可以通过预定义的触发器控制模型预测。随着各种攻防模型的提出，进行严格的评估具有重要意义。然而，我们强调了以往的后门学习评估中的两个问题：(1)忽略了现实世界场景(如发布有毒数据集或模型)之间的差异，我们认为每个场景都有自己的约束和关注点，因此需要特定的评估协议；(2)评估指标只考虑攻击是否会颠覆模型对有毒样本的预测，并保持对良性样本的性能，而忽略了有毒样本也应该是隐蔽的和保持语义的。为了解决这些问题，我们将现有的工作分为三个实际场景，攻击者分别发布数据集、预先训练的模型和微调的模型，然后讨论他们独特的评估方法。在度量上，为了全面评估中毒样本，我们使用语法错误增加和困惑差异来隐蔽性，并使用文本相似性来验证有效性。在形式化框架之后，我们开发了一个开源工具包OpenBackdoor来促进文本后门学习的实现和评估。利用这个工具包，我们进行了大量的实验，在建议的范式下对攻击和防御模型进行基准测试。为了方便对有毒数据集的未充分挖掘的防御，我们进一步提出了CUBE，一种简单但强大的基于聚类的防御基线。我们希望我们的框架和基准能够成为未来模式发展和评价的基石。



## **36. I Know What You Trained Last Summer: A Survey on Stealing Machine Learning Models and Defences**

我知道你去年夏天培训了什么：关于窃取机器学习模型和防御的调查 cs.LG

Under review at ACM Computing Surveys

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2206.08451v1)

**Authors**: Daryna Oliynyk, Rudolf Mayer, Andreas Rauber

**Abstracts**: Machine Learning-as-a-Service (MLaaS) has become a widespread paradigm, making even the most complex machine learning models available for clients via e.g. a pay-per-query principle. This allows users to avoid time-consuming processes of data collection, hyperparameter tuning, and model training. However, by giving their customers access to the (predictions of their) models, MLaaS providers endanger their intellectual property, such as sensitive training data, optimised hyperparameters, or learned model parameters. Adversaries can create a copy of the model with (almost) identical behavior using the the prediction labels only. While many variants of this attack have been described, only scattered defence strategies have been proposed, addressing isolated threats. This raises the necessity for a thorough systematisation of the field of model stealing, to arrive at a comprehensive understanding why these attacks are successful, and how they could be holistically defended against. We address this by categorising and comparing model stealing attacks, assessing their performance, and exploring corresponding defence techniques in different settings. We propose a taxonomy for attack and defence approaches, and provide guidelines on how to select the right attack or defence strategy based on the goal and available resources. Finally, we analyse which defences are rendered less effective by current attack strategies.

摘要: 机器学习即服务(MLaaS)已经成为一种广泛的范例，即使是最复杂的机器学习模型也可以通过例如按查询付费的原则提供给客户。这使用户可以避免耗时的数据收集、超参数调整和模型训练过程。然而，MLaaS提供商允许他们的客户访问(他们的)模型的预测，从而危及他们的知识产权，例如敏感的培训数据、优化的超参数或学习的模型参数。攻击者仅使用预测标签就可以创建具有(几乎)相同行为的模型副本。虽然已经描述了这种攻击的许多变体，但只提出了分散的防御战略，以应对孤立的威胁。这就提出了对模型窃取领域进行彻底系统化的必要性，以全面了解这些攻击为什么成功，以及如何从整体上防御它们。我们通过对模型窃取攻击进行分类和比较，评估它们的性能，并在不同的环境中探索相应的防御技术来解决这个问题。我们提出了攻击和防御方法的分类，并提供了关于如何根据目标和可用资源选择正确的攻击或防御策略的指导方针。最后，我们分析了当前的攻击策略降低了哪些防御的有效性。



## **37. Boosting the Adversarial Transferability of Surrogate Model with Dark Knowledge**

利用暗知识提高代理模型的对抗性转移能力 cs.LG

26 pages, 5 figures

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2206.08316v1)

**Authors**: Dingcheng Yang, Zihao Xiao, Wenjian Yu

**Abstracts**: Deep neural networks (DNNs) for image classification are known to be vulnerable to adversarial examples. And, the adversarial examples have transferability, which means an adversarial example for a DNN model can fool another black-box model with a non-trivial probability. This gave birth of the transfer-based adversarial attack where the adversarial examples generated by a pretrained or known model (called surrogate model) are used to conduct black-box attack. There are some work on how to generate the adversarial examples from a given surrogate model to achieve better transferability. However, training a special surrogate model to generate adversarial examples with better transferability is relatively under-explored. In this paper, we propose a method of training a surrogate model with abundant dark knowledge to boost the adversarial transferability of the adversarial examples generated by the surrogate model. This trained surrogate model is named dark surrogate model (DSM), and the proposed method to train DSM consists of two key components: a teacher model extracting dark knowledge and providing soft labels, and the mixing augmentation skill which enhances the dark knowledge of training data. Extensive experiments have been conducted to show that the proposed method can substantially improve the adversarial transferability of surrogate model across different architectures of surrogate model and optimizers for generating adversarial examples. We also show that the proposed method can be applied to other scenarios of transfer-based attack that contain dark knowledge, like face verification.

摘要: 用于图像分类的深度神经网络(DNN)很容易受到敌意例子的影响。而且，对抗性例子具有可转移性，这意味着一个DNN模型的对抗性例子可以以非平凡的概率愚弄另一个黑盒模型。这就产生了基于迁移的对抗性攻击，利用预先训练或已知模型(称为代理模型)生成的对抗性样本进行黑箱攻击。关于如何从给定的代理模型生成对抗性实例以实现更好的可转移性，已经有一些工作。然而，训练一种特殊的代理模型来生成具有更好可转移性的对抗性实例的研究相对较少。在本文中，我们提出了一种训练具有丰富暗知识的代理模型的方法，以提高由代理模型生成的对抗性实例的对抗性可转移性。训练后的代理模型被称为暗代理模型(DSM)，提出的训练DSM的方法由两个关键部分组成：提取暗知识并提供软标签的教师模型和增强训练数据的暗知识的混合扩充技巧。大量的实验表明，该方法可以显著提高代理模型在不同体系结构下的对抗性可转移性，并优化生成对抗性实例的优化器。我们还表明，该方法也可以应用于其他基于转移的攻击场景，如人脸验证等。



## **38. Adversarial Patch Attacks and Defences in Vision-Based Tasks: A Survey**

视觉任务中对抗性补丁攻击与防御的研究进展 cs.CV

A. Sharma and Y. Bian share equal contribution

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2206.08304v1)

**Authors**: Abhijith Sharma, Yijun Bian, Phil Munz, Apurva Narayan

**Abstracts**: Adversarial attacks in deep learning models, especially for safety-critical systems, are gaining more and more attention in recent years, due to the lack of trust in the security and robustness of AI models. Yet the more primitive adversarial attacks might be physically infeasible or require some resources that are hard to access like the training data, which motivated the emergence of patch attacks. In this survey, we provide a comprehensive overview to cover existing techniques of adversarial patch attacks, aiming to help interested researchers quickly catch up with the progress in this field. We also discuss existing techniques for developing detection and defences against adversarial patches, aiming to help the community better understand this field and its applications in the real world.

摘要: 由于对人工智能模型的安全性和健壮性缺乏信任，深度学习模型中的对抗性攻击，特别是对安全关键系统的攻击，近年来得到了越来越多的关注。然而，更原始的对抗性攻击可能在物理上是不可行的，或者需要一些难以获取的资源，比如促使补丁攻击出现的训练数据。在这次调查中，我们提供了一个全面的概述，涵盖了现有的对抗性补丁攻击技术，旨在帮助感兴趣的研究人员快速跟上该领域的进展。我们还讨论了开发针对恶意补丁的检测和防御的现有技术，旨在帮助社区更好地了解这一领域及其在现实世界中的应用。



## **39. Adversarial Robustness of Graph-based Anomaly Detection**

基于图的异常检测的攻击健壮性 cs.CR

arXiv admin note: substantial text overlap with arXiv:2106.09989

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2206.08260v1)

**Authors**: Yulin Zhu, Yuni Lai, Kaifa Zhao, Xiapu Luo, Mingquan Yuan, Jian Ren, Kai Zhou

**Abstracts**: Graph-based anomaly detection is becoming prevalent due to the powerful representation abilities of graphs as well as recent advances in graph mining techniques. These GAD tools, however, expose a new attacking surface, ironically due to their unique advantage of being able to exploit the relations among data. That is, attackers now can manipulate those relations (i.e., the structure of the graph) to allow target nodes to evade detection or degenerate the classification performance of the detection. In this paper, we exploit this vulnerability by designing the structural poisoning attacks to a FeXtra-based GAD system termed OddBall as well as the black box attacks against GCN-based GAD systems by attacking the imbalanced lienarized GCN ( LGCN ). Specifically, we formulate the attack against OddBall and LGCN as a one-level optimization problem by incorporating different regression techniques, where the key technical challenge is to efficiently solve the problem in a discrete domain. We propose a novel attack method termed BinarizedAttack based on gradient descent. Comparing to prior arts, BinarizedAttack can better use the gradient information, making it particularly suitable for solving discrete optimization problems, thus opening the door to studying a new type of attack against security analytic tools that rely on graph data.

摘要: 由于图的强大的表示能力以及图挖掘技术的最新进展，基于图的异常检测正在变得流行起来。然而，具有讽刺意味的是，这些GAD工具暴露了一个新的攻击面，因为它们能够利用数据之间的关系这一独特优势。也就是说，攻击者现在可以操纵这些关系(即图的结构)，以允许目标节点逃避检测或降低检测的分类性能。在本文中，我们通过设计对基于FeXtra的GAD系统的结构毒化攻击(称为odball)以及通过攻击不平衡连续化的GCN(LGCN)来对基于GCN的GAD系统进行黑盒攻击来利用这一漏洞。具体地说，通过结合不同的回归技术，我们将对Odball和LGCN的攻击描述为一个一层优化问题，其中关键的技术挑战是在离散域中有效地求解该问题。提出了一种新的基于梯度下降的攻击方法BinarizedAttack。与现有技术相比，BinarizedAttack能够更好地利用梯度信息，使其特别适合于求解离散优化问题，从而为研究依赖于图数据的安全分析工具的新型攻击打开了大门。



## **40. Adversarial attacks on voter model dynamics in complex networks**

复杂网络中选民模型动态的对抗性攻击 physics.soc-ph

7 pages, 5 figures

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2111.09561v2)

**Authors**: Katsumi Chiyomaru, Kazuhiro Takemoto

**Abstracts**: This study investigates adversarial attacks conducted to distort voter model dynamics in complex networks. Specifically, a simple adversarial attack method is proposed to hold the state of opinions of an individual closer to the target state in the voter model dynamics. This indicates that even when one opinion is the majority, the vote outcome can be inverted (i.e., the outcome can lean toward the other opinion) by adding extremely small (hard-to-detect) perturbations strategically generated in social networks. Adversarial attacks are relatively more effective in complex (large and dense) networks. These results indicate that opinion dynamics can be unknowingly distorted.

摘要: 这项研究调查了复杂网络中为扭曲选民模型动态而进行的对抗性攻击。具体地说，提出了一种简单的对抗性攻击方法，使个体的意见状态更接近于选民模型动态中的目标状态。这表明，即使当一种意见占多数时，投票结果也可以通过添加在社交网络中策略性地产生的极小(难以检测)的扰动来颠倒(即结果可能倾向于另一种意见)。对抗性攻击在复杂(大型和密集)网络中相对更有效。这些结果表明，意见动态可能在不知不觉中被扭曲。



## **41. Adversarial Privacy Protection on Speech Enhancement**

语音增强中的对抗性隐私保护 cs.SD

5 pages, 6 figures

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2206.08170v1)

**Authors**: Mingyu Dong, Diqun Yan, Rangding Wang

**Abstracts**: Speech is easily leaked imperceptibly, such as being recorded by mobile phones in different situations. Private content in speech may be maliciously extracted through speech enhancement technology. Speech enhancement technology has developed rapidly along with deep neural networks (DNNs), but adversarial examples can cause DNNs to fail. In this work, we propose an adversarial method to degrade speech enhancement systems. Experimental results show that generated adversarial examples can erase most content information in original examples or replace it with target speech content through speech enhancement. The word error rate (WER) between an enhanced original example and enhanced adversarial example recognition result can reach 89.0%. WER of target attack between enhanced adversarial example and target example is low to 33.75% . Adversarial perturbation can bring the rate of change to the original example to more than 1.4430. This work can prevent the malicious extraction of speech.

摘要: 语音很容易在不知不觉中泄露，比如在不同的情况下被手机录音。语音中的私人内容可能会通过语音增强技术被恶意提取。语音增强技术随着深度神经网络(DNN)的发展而迅速发展，但敌意的例子会导致DNN的失败。在这项工作中，我们提出了一种对抗性方法来降低语音增强系统的性能。实验结果表明，生成的对抗性实例可以消除原始实例中的大部分内容信息，或者通过语音增强将其替换为目标语音内容。增强的原始实例和增强的对抗性实例识别结果的误词率(WER)可以达到89.0%。增强对抗性范例与目标范例之间的目标攻击准确率低至33.75%。对抗性扰动可以使原始例子的变化率达到1.4430以上。这项工作可以防止语音的恶意提取。



## **42. Detecting Adversarial Examples Is (Nearly) As Hard As Classifying Them**

发现敌意的例子(几乎)和分类一样难。 cs.LG

ICML 2022 (Long Talk)

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2107.11630v2)

**Authors**: Florian Tramèr

**Abstracts**: Making classifiers robust to adversarial examples is hard. Thus, many defenses tackle the seemingly easier task of detecting perturbed inputs. We show a barrier towards this goal. We prove a general hardness reduction between detection and classification of adversarial examples: given a robust detector for attacks at distance {\epsilon} (in some metric), we can build a similarly robust (but inefficient) classifier for attacks at distance {\epsilon}/2. Our reduction is computationally inefficient, and thus cannot be used to build practical classifiers. Instead, it is a useful sanity check to test whether empirical detection results imply something much stronger than the authors presumably anticipated. To illustrate, we revisit 13 detector defenses. For 11/13 cases, we show that the claimed detection results would imply an inefficient classifier with robustness far beyond the state-of-the-art.

摘要: 要让分类器对对抗性例子具有健壮性是很困难的。因此，许多防御措施处理的似乎更容易的任务是检测受干扰的输入。我们展示了实现这一目标的障碍。我们证明了敌意例子的检测和分类之间的一般难度降低：给定一个对于距离为{\epsilon}的攻击的稳健检测器(在某些度量中)，我们可以为距离为{\epsilon}/2的攻击构建一个类似的稳健(但效率低下)的分类器。我们的简化在计算上是低效的，因此不能用于构建实用的分类器。相反，它是一种有用的理智检查，以测试经验检测结果是否意味着比作者推测的预期更强烈的东西。为了说明这一点，我们回顾了13个探测器防御系统。对于11/13的情况，我们表明，声称的检测结果将意味着一个低效的分类器，其稳健性远远超过最先进的水平。



## **43. Adversarial Attacks on Gaussian Process Bandits**

对高斯过程环的对抗性攻击 stat.ML

Accepted to ICML 2022

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2110.08449v3)

**Authors**: Eric Han, Jonathan Scarlett

**Abstracts**: Gaussian processes (GP) are a widely-adopted tool used to sequentially optimize black-box functions, where evaluations are costly and potentially noisy. Recent works on GP bandits have proposed to move beyond random noise and devise algorithms robust to adversarial attacks. This paper studies this problem from the attacker's perspective, proposing various adversarial attack methods with differing assumptions on the attacker's strength and prior information. Our goal is to understand adversarial attacks on GP bandits from theoretical and practical perspectives. We focus primarily on targeted attacks on the popular GP-UCB algorithm and a related elimination-based algorithm, based on adversarially perturbing the function $f$ to produce another function $\tilde{f}$ whose optima are in some target region $\mathcal{R}_{\rm target}$. Based on our theoretical analysis, we devise both white-box attacks (known $f$) and black-box attacks (unknown $f$), with the former including a Subtraction attack and Clipping attack, and the latter including an Aggressive subtraction attack. We demonstrate that adversarial attacks on GP bandits can succeed in forcing the algorithm towards $\mathcal{R}_{\rm target}$ even with a low attack budget, and we test our attacks' effectiveness on a diverse range of objective functions.

摘要: 高斯过程(GP)是一种广泛采用的工具，用于顺序优化黑盒函数，其中评估成本较高，并且可能存在噪声。最近关于GP盗贼的研究已经提出超越随机噪声，设计出对对手攻击强大的算法。本文从攻击者的角度研究这一问题，提出了各种对抗性攻击方法，并对攻击者的强度和先验信息进行了不同的假设。我们的目标是从理论和实践的角度来理解对GP土匪的敌意攻击。我们主要关注对流行的GP-UCB算法和相关的基于消元的算法的定向攻击，该算法基于对函数$f$的恶意扰动来产生另一个函数$\tide{f}$，其最优值位于某个目标区域$\数学{R}_{\rm目标}$。在理论分析的基础上，我们设计了白盒攻击(已知$f$)和黑盒攻击(未知$f$)，前者包括减法攻击和剪裁攻击，后者包括侵略性减法攻击。我们证明了对GP盗贼的敌意攻击即使在较低的攻击预算下也能成功地迫使算法向数学上的{R}_{\Rm目标}$逼近，并在不同的目标函数上测试了我们的攻击的有效性。



## **44. Analysis and Extensions of Adversarial Training for Video Classification**

对抗性训练在视频分类中的分析与扩展 cs.CV

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2206.07953v1)

**Authors**: Kaleab A. Kinfu, René Vidal

**Abstracts**: Adversarial training (AT) is a simple yet effective defense against adversarial attacks to image classification systems, which is based on augmenting the training set with attacks that maximize the loss. However, the effectiveness of AT as a defense for video classification has not been thoroughly studied. Our first contribution is to show that generating optimal attacks for video requires carefully tuning the attack parameters, especially the step size. Notably, we show that the optimal step size varies linearly with the attack budget. Our second contribution is to show that using a smaller (sub-optimal) attack budget at training time leads to a more robust performance at test time. Based on these findings, we propose three defenses against attacks with variable attack budgets. The first one, Adaptive AT, is a technique where the attack budget is drawn from a distribution that is adapted as training iterations proceed. The second, Curriculum AT, is a technique where the attack budget is increased as training iterations proceed. The third, Generative AT, further couples AT with a denoising generative adversarial network to boost robust performance. Experiments on the UCF101 dataset demonstrate that the proposed methods improve adversarial robustness against multiple attack types.

摘要: 对抗性训练(AT)是一种针对图像分类系统的对抗性攻击的简单而有效的防御方法，其基础是用最大化损失的攻击来扩充训练集。然而，AT作为视频分类的防御手段的有效性还没有得到深入的研究。我们的第一个贡献是表明为视频生成最优攻击需要仔细调整攻击参数，特别是步长。值得注意的是，我们发现最优步长与攻击预算呈线性关系。我们的第二个贡献是表明在训练时使用较小的(次优)攻击预算可以在测试时获得更稳健的性能。基于这些发现，我们提出了三种不同攻击预算的攻击防御方案。第一种是自适应AT，这种技术的攻击预算是从随着训练迭代进行而调整的分布中提取的。第二种，课程AT，是一种攻击预算随着训练迭代的进行而增加的技术。第三，生成性AT，进一步将AT与去噪生成性对抗网络相结合，以提高稳健的性能。在UCF101数据集上的实验表明，所提出的方法提高了对多种攻击类型的鲁棒性。



## **45. A Comprehensive Test Pattern Generation Approach Exploiting SAT Attack for Logic Locking**

一种利用SAT攻击进行逻辑锁定的综合测试码生成方法 cs.CR

10 pages, 8 figures

**SubmitDate**: 2022-06-16    [paper-pdf](http://arxiv.org/pdf/2204.11307v2)

**Authors**: Yadi Zhong, Ujjwal Guin

**Abstracts**: The need for reducing manufacturing defect escape in today's safety-critical applications requires increased fault coverage. However, generating a test set using commercial automatic test pattern generation (ATPG) tools that lead to zero-defect escape is still an open problem. It is challenging to detect all stuck-at faults to reach 100% fault coverage. In parallel, the hardware security community has been actively involved in developing solutions for logic locking to prevent IP piracy. Locks (e.g., XOR gates) are inserted in different locations of the netlist so that an adversary cannot determine the secret key. Unfortunately, the Boolean satisfiability (SAT) based attack, introduced in [1], can break different logic locking schemes in minutes. In this paper, we propose a novel test pattern generation approach using the powerful SAT attack on logic locking. A stuck-at fault is modeled as a locked gate with a secret key. Our modeling of stuck-at faults preserves the property of fault activation and propagation. We show that the input pattern that determines the key is a test for the stuck-at fault. We propose two different approaches for test pattern generation. First, a single stuck-at fault is targeted, and a corresponding locked circuit with one key bit is created. This approach generates one test pattern per fault. Second, we consider a group of faults and convert the circuit to its locked version with multiple key bits. The inputs obtained from the SAT tool are the test set for detecting this group of faults. Our approach is able to find test patterns for hard-to-detect faults that were previously failed in commercial ATPG tools. The proposed test pattern generation approach can efficiently detect redundant faults present in a circuit. We demonstrate the effectiveness of the approach on ITC'99 benchmarks. The results show that we can achieve a perfect fault coverage reaching 100%.

摘要: 在当今的安全关键应用中，减少制造缺陷逃逸的需要需要增加故障覆盖率。然而，使用商业自动测试模式生成(ATPG)工具生成测试集以实现零缺陷逃逸仍然是一个未解决的问题。要检测所有固定故障以达到100%的故障覆盖率是具有挑战性的。与此同时，硬件安全界一直积极参与开发逻辑锁定解决方案，以防止知识产权盗版。锁(例如，异或门)被插入网表的不同位置，使得对手不能确定密钥。不幸的是，在[1]中引入的基于布尔可满足性(SAT)的攻击可以在几分钟内破解不同的逻辑锁定方案。在本文中，我们提出了一种新的测试模式生成方法，该方法利用了对逻辑锁的强大SAT攻击。一个顽固的错误被建模为一扇锁着的门和一把密钥。我们对固定故障的建模保留了故障激活和传播的性质。我们证明了决定关键字的输入模式是对固定错误的测试。我们提出了两种不同的测试模式生成方法。首先，针对单个固定故障，创建具有一个密钥位的相应锁定电路。该方法为每个故障生成一个测试模式。其次，我们考虑一组故障，并将电路转换为具有多个密钥位的锁定版本。从SAT工具获得的输入是用于检测这组故障的测试集。我们的方法能够为以前在商业ATPG工具中失败的难以检测的故障找到测试模式。提出的测试码生成方法可以有效地检测电路中存在的冗余故障。我们在ITC‘99基准上证明了该方法的有效性。结果表明，我们可以达到100%的完美故障覆盖率。



## **46. Architectural Backdoors in Neural Networks**

神经网络中的体系结构后门 cs.LG

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.07840v1)

**Authors**: Mikel Bober-Irizar, Ilia Shumailov, Yiren Zhao, Robert Mullins, Nicolas Papernot

**Abstracts**: Machine learning is vulnerable to adversarial manipulation. Previous literature has demonstrated that at the training stage attackers can manipulate data and data sampling procedures to control model behaviour. A common attack goal is to plant backdoors i.e. force the victim model to learn to recognise a trigger known only by the adversary. In this paper, we introduce a new class of backdoor attacks that hide inside model architectures i.e. in the inductive bias of the functions used to train. These backdoors are simple to implement, for instance by publishing open-source code for a backdoored model architecture that others will reuse unknowingly. We demonstrate that model architectural backdoors represent a real threat and, unlike other approaches, can survive a complete re-training from scratch. We formalise the main construction principles behind architectural backdoors, such as a link between the input and the output, and describe some possible protections against them. We evaluate our attacks on computer vision benchmarks of different scales and demonstrate the underlying vulnerability is pervasive in a variety of training settings.

摘要: 机器学习很容易受到敌意操纵。以前的文献已经证明，在训练阶段，攻击者可以操纵数据和数据采样程序来控制模型行为。一个常见的攻击目标是植入后门，即迫使受害者模型学习识别只有对手知道的触发器。在本文中，我们引入了一类新的隐藏在模型体系结构中的后门攻击，即用于训练的函数的归纳偏差。这些后门很容易实现，例如，通过发布后端模型体系结构的开放源代码，其他人将在不知情的情况下重用这些代码。我们演示了模型体系结构后门是一种真正的威胁，并且与其他方法不同，它可以从头开始接受完整的重新培训。我们形式化了建筑后门背后的主要构造原则，例如输入和输出之间的链接，并描述了一些可能的保护措施。我们评估了我们对不同规模的计算机视觉基准的攻击，并证明了潜在的漏洞在各种训练环境中普遍存在。



## **47. Search-Based Testing Approach for Deep Reinforcement Learning Agents**

一种基于搜索的深度强化学习代理测试方法 cs.SE

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.07813v1)

**Authors**: Amirhossein Zolfagharian, Manel Abdellatif, Lionel Briand, Mojtaba Bagherzadeh, Ramesh S

**Abstracts**: Deep Reinforcement Learning (DRL) algorithms have been increasingly employed during the last decade to solve various decision-making problems such as autonomous driving and robotics. However, these algorithms have faced great challenges when deployed in safety-critical environments since they often exhibit erroneous behaviors that can lead to potentially critical errors. One way to assess the safety of DRL agents is to test them to detect possible faults leading to critical failures during their execution. This raises the question of how we can efficiently test DRL policies to ensure their correctness and adherence to safety requirements. Most existing works on testing DRL agents use adversarial attacks that perturb states or actions of the agent. However, such attacks often lead to unrealistic states of the environment. Their main goal is to test the robustness of DRL agents rather than testing the compliance of agents' policies with respect to requirements. Due to the huge state space of DRL environments, the high cost of test execution, and the black-box nature of DRL algorithms, the exhaustive testing of DRL agents is impossible. In this paper, we propose a Search-based Testing Approach of Reinforcement Learning Agents (STARLA) to test the policy of a DRL agent by effectively searching for failing executions of the agent within a limited testing budget. We use machine learning models and a dedicated genetic algorithm to narrow the search towards faulty episodes. We apply STARLA on a Deep-Q-Learning agent which is widely used as a benchmark and show that it significantly outperforms Random Testing by detecting more faults related to the agent's policy. We also investigate how to extract rules that characterize faulty episodes of the DRL agent using our search results. Such rules can be used to understand the conditions under which the agent fails and thus assess its deployment risks.

摘要: 在过去的十年中，深度强化学习(DRL)算法被越来越多地用于解决各种决策问题，如自动驾驶和机器人技术。然而，当这些算法部署在安全关键环境中时，它们面临着巨大的挑战，因为它们经常表现出可能导致潜在关键错误的错误行为。评估DRL代理安全性的一种方法是对其进行测试，以检测在其执行期间可能导致严重故障的故障。这提出了一个问题，即我们如何有效地测试DRL政策，以确保它们的正确性和对安全要求的遵守。大多数现有的测试DRL代理的工作使用对抗性攻击，扰乱代理的状态或动作。然而，这样的攻击往往会导致不切实际的环境状况。他们的主要目标是测试DRL代理的健壮性，而不是测试代理策略与需求的符合性。由于DRL环境的状态空间巨大、测试执行成本高以及DRL算法的黑箱性质，对DRL代理进行穷举测试是不可能的。在本文中，我们提出了一种基于搜索的强化学习代理测试方法(STARLA)，通过在有限的测试预算内有效地搜索代理的失败执行来测试DRL代理的策略。我们使用机器学习模型和专门的遗传算法将搜索范围缩小到故障剧集。我们将Starla应用于一个被广泛用作基准测试的Deep-Q-Learning代理上，结果表明它通过检测更多与代理策略相关的错误而显著优于随机测试。我们还研究了如何使用搜索结果提取表征DRL代理故障情节的规则。此类规则可用于了解代理发生故障的条件，从而评估其部署风险。



## **48. Poison Forensics: Traceback of Data Poisoning Attacks in Neural Networks**

毒物取证：神经网络中数据中毒攻击的追溯 cs.CR

18 pages

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2110.06904v2)

**Authors**: Shawn Shan, Arjun Nitin Bhagoji, Haitao Zheng, Ben Y. Zhao

**Abstracts**: In adversarial machine learning, new defenses against attacks on deep learning systems are routinely broken soon after their release by more powerful attacks. In this context, forensic tools can offer a valuable complement to existing defenses, by tracing back a successful attack to its root cause, and offering a path forward for mitigation to prevent similar attacks in the future.   In this paper, we describe our efforts in developing a forensic traceback tool for poison attacks on deep neural networks. We propose a novel iterative clustering and pruning solution that trims "innocent" training samples, until all that remains is the set of poisoned data responsible for the attack. Our method clusters training samples based on their impact on model parameters, then uses an efficient data unlearning method to prune innocent clusters. We empirically demonstrate the efficacy of our system on three types of dirty-label (backdoor) poison attacks and three types of clean-label poison attacks, across domains of computer vision and malware classification. Our system achieves over 98.4% precision and 96.8% recall across all attacks. We also show that our system is robust against four anti-forensics measures specifically designed to attack it.

摘要: 在对抗性机器学习中，针对深度学习系统攻击的新防御措施在发布后不久通常会被更强大的攻击打破。在这方面，法医工具可以通过追溯一次成功的攻击的根本原因，并为缓解提供一条前进的道路，以防止未来发生类似的攻击，从而对现有的防御措施提供宝贵的补充。在这篇文章中，我们描述了我们在开发一个针对深度神经网络的毒物攻击的法医追踪工具方面所做的努力。我们提出了一种新的迭代聚类和剪枝解决方案，该方案修剪“无辜”的训练样本，直到剩下的只是导致攻击的有毒数据集。我们的方法根据训练样本对模型参数的影响对训练样本进行分类，然后使用一种有效的数据忘却学习方法来剪除无辜的聚类。我们在计算机视觉和恶意软件分类领域对三种类型的脏标签(后门)毒物攻击和三种类型的干净标签毒物攻击的有效性进行了经验验证。我们的系统在所有攻击中都达到了98.4%以上的准确率和96.8%的召回率。我们还表明，我们的系统对四种专门为攻击它而设计的反取证措施是健壮的。



## **49. Learn to Adapt: Robust Drift Detection in Security Domain**

学会适应：安全域中的稳健漂移检测 cs.CR

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.07581v1)

**Authors**: Aditya Kuppa, Nhien-An Le-Khac

**Abstracts**: Deploying robust machine learning models has to account for concept drifts arising due to the dynamically changing and non-stationary nature of data. Addressing drifts is particularly imperative in the security domain due to the ever-evolving threat landscape and lack of sufficiently labeled training data at the deployment time leading to performance degradation. Recently proposed concept drift detection methods in literature tackle this problem by identifying the changes in feature/data distributions and periodically retraining the models to learn new concepts. While these types of strategies should absolutely be conducted when possible, they are not robust towards attacker-induced drifts and suffer from a delay in detecting new attacks. We aim to address these shortcomings in this work. we propose a robust drift detector that not only identifies drifted samples but also discovers new classes as they arrive in an on-line fashion. We evaluate the proposed method with two security-relevant data sets -- network intrusion data set released in 2018 and APT Command and Control dataset combined with web categorization data. Our evaluation shows that our drifting detection method is not only highly accurate but also robust towards adversarial drifts and discovers new classes from drifted samples.

摘要: 部署健壮的机器学习模型必须考虑到由于数据的动态变化和非静态性质而产生的概念漂移。在安全领域，解决漂移问题尤为迫切，因为威胁形势不断演变，而且在部署时缺乏标记充分的训练数据，从而导致性能下降。最近文献中提出的概念漂移检测方法通过识别特征/数据分布的变化并周期性地重新训练模型来学习新概念来解决这一问题。虽然这些类型的策略绝对应该在可能的情况下进行，但它们对攻击者引发的漂移并不健壮，并且在检测新攻击方面存在延迟。我们的目标是解决这项工作中的这些缺点。我们提出了一种稳健的漂移检测器，它不仅识别漂移的样本，而且当它们以在线方式到达时发现新的类别。我们用2018年发布的网络入侵数据集和APT指挥控制数据集结合网络分类数据对该方法进行了评估。我们的评估表明，我们的漂移检测方法不仅具有很高的准确率，而且对敌意漂移具有很强的鲁棒性，并从漂移样本中发现新的类别。



## **50. Creating a Secure Underlay for the Internet**

创建Internet的安全参考底图 cs.NI

Usenix Security 2022

**SubmitDate**: 2022-06-15    [paper-pdf](http://arxiv.org/pdf/2206.06879v2)

**Authors**: Henry Birge-Lee, Joel Wanner, Grace Cimaszewski, Jonghoon Kwon, Liang Wang, Francois Wirz, Prateek Mittal, Adrian Perrig, Yixin Sun

**Abstracts**: Adversaries can exploit inter-domain routing vulnerabilities to intercept communication and compromise the security of critical Internet applications. Meanwhile the deployment of secure routing solutions such as Border Gateway Protocol Security (BGPsec) and Scalability, Control and Isolation On Next-generation networks (SCION) are still limited. How can we leverage emerging secure routing backbones and extend their security properties to the broader Internet?   We design and deploy an architecture to bootstrap secure routing. Our key insight is to abstract the secure routing backbone as a virtual Autonomous System (AS), called Secure Backbone AS (SBAS). While SBAS appears as one AS to the Internet, it is a federated network where routes are exchanged between participants using a secure backbone. SBAS makes BGP announcements for its customers' IP prefixes at multiple locations (referred to as Points of Presence or PoPs) allowing traffic from non-participating hosts to be routed to a nearby SBAS PoP (where it is then routed over the secure backbone to the true prefix owner). In this manner, we are the first to integrate a federated secure non-BGP routing backbone with the BGP-speaking Internet.   We present a real-world deployment of our architecture that uses SCIONLab to emulate the secure backbone and the PEERING framework to make BGP announcements to the Internet. A combination of real-world attacks and Internet-scale simulations shows that SBAS substantially reduces the threat of routing attacks. Finally, we survey network operators to better understand optimal governance and incentive models.

摘要: 攻击者可以利用域间路由漏洞来拦截通信并危及关键Internet应用程序的安全。同时，边界网关协议安全(BGPSEC)和下一代网络的可扩展性、控制和隔离(SCION)等安全路由解决方案的部署仍然有限。我们如何利用新兴的安全路由主干并将其安全属性扩展到更广泛的互联网？我们设计并部署了一个用于引导安全路由的体系结构。我们的主要见解是将安全路由主干抽象为一个虚拟自治系统(AS)，称为安全主干AS(SBAS)。虽然SBAS在Internet上看起来像一个整体，但它是一个联合网络，参与者之间使用安全的主干交换路由。SBAS在多个位置(称为入网点或POP)为其客户的IP前缀发布BGP公告，允许将来自非参与主机的流量路由到附近的SBAS POP(然后通过安全主干将其路由到真正的前缀所有者)。通过这种方式，我们是第一个将联合安全的非BGP路由骨干网与BGP语音互联网集成在一起的公司。我们提供了我们的架构的实际部署，该架构使用SCIONLab模拟安全主干，并使用对等框架向互联网发布BGP公告。现实世界的攻击和互联网规模的模拟相结合表明，SBAS大大降低了路由攻击的威胁。最后，我们对网络运营商进行了调查，以更好地了解最优治理和激励模式。



