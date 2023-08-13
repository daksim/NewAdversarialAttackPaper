# Latest Adversarial Attack Papers
**update at 2023-08-13 21:30:44**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. RobustPdM: Designing Robust Predictive Maintenance against Adversarial Attacks**

RobustPdM：针对敌方攻击设计稳健的预测性维护 cs.CR

**SubmitDate**: 2023-08-10    [abs](http://arxiv.org/abs/2301.10822v2) [paper-pdf](http://arxiv.org/pdf/2301.10822v2)

**Authors**: Ayesha Siddique, Ripan Kumar Kundu, Gautam Raj Mode, Khaza Anuarul Hoque

**Abstract**: The state-of-the-art predictive maintenance (PdM) techniques have shown great success in reducing maintenance costs and downtime of complicated machines while increasing overall productivity through extensive utilization of Internet-of-Things (IoT) and Deep Learning (DL). Unfortunately, IoT sensors and DL algorithms are both prone to cyber-attacks. For instance, DL algorithms are known for their susceptibility to adversarial examples. Such adversarial attacks are vastly under-explored in the PdM domain. This is because the adversarial attacks in the computer vision domain for classification tasks cannot be directly applied to the PdM domain for multivariate time series (MTS) regression tasks. In this work, we propose an end-to-end methodology to design adversarially robust PdM systems by extensively analyzing the effect of different types of adversarial attacks and proposing a novel adversarial defense technique for DL-enabled PdM models. First, we propose novel MTS Projected Gradient Descent (PGD) and MTS PGD with random restarts (PGD_r) attacks. Then, we evaluate the impact of MTS PGD and PGD_r along with MTS Fast Gradient Sign Method (FGSM) and MTS Basic Iterative Method (BIM) on Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), Convolutional Neural Network (CNN), and Bi-directional LSTM based PdM system. Our results using NASA's turbofan engine dataset show that adversarial attacks can cause a severe defect (up to 11X) in the RUL prediction, outperforming the effectiveness of the state-of-the-art PdM attacks by 3X. Furthermore, we present a novel approximate adversarial training method to defend against adversarial attacks. We observe that approximate adversarial training can significantly improve the robustness of PdM models (up to 54X) and outperforms the state-of-the-art PdM defense methods by offering 3X more robustness.

摘要: 最先进的预测性维护(PDM)技术通过广泛利用物联网(IoT)和深度学习(DL)，在降低复杂机器的维护成本和停机时间方面取得了巨大成功，同时提高了整体生产率。不幸的是，物联网传感器和数字图书馆算法都容易受到网络攻击。例如，DL算法以其对对抗性例子的敏感性而闻名。这种对抗性攻击在产品数据管理领域被极大地忽视了。这是因为用于分类任务的计算机视觉领域中的对抗性攻击不能直接应用于用于多变量时间序列(MTS)回归任务的PDM域。在这项工作中，我们通过广泛分析不同类型的对抗性攻击的影响，提出了一种端到端的方法来设计对抗性健壮的产品数据管理系统，并提出了一种新的针对DL使能的产品数据管理模型的对抗性防御技术。首先，我们提出了新的MTS投影梯度下降(PGD)攻击和带随机重启的MTS PGD(PGD_R)攻击。然后，结合MTS快速梯度符号法(FGSM)和MTS基本迭代法(BIM)，评价了MTS、PGD和PGD_r对基于长短期记忆(LSTM)、门控递归单元(GRU)、卷积神经网络(CNN)和双向LSTM的产品数据管理系统的影响。我们使用NASA的涡扇发动机数据集的结果表明，敌意攻击可以导致RUL预测中的严重缺陷(高达11倍)，比最先进的PDM攻击的有效性高出3倍。此外，我们还提出了一种新的近似对抗性训练方法来防御对抗性攻击。我们观察到，近似对抗性训练可以显著提高PDM模型的健壮性(高达54倍)，并通过提供3倍以上的健壮性来超越最新的PDM防御方法。



## **2. Diffusion Denoised Smoothing for Certified and Adversarial Robust Out-Of-Distribution Detection**

基于扩散去噪平滑的认证和对抗稳健失配检测 cs.LG

**SubmitDate**: 2023-08-10    [abs](http://arxiv.org/abs/2303.14961v3) [paper-pdf](http://arxiv.org/pdf/2303.14961v3)

**Authors**: Nicola Franco, Daniel Korth, Jeanette Miriam Lorenz, Karsten Roscher, Stephan Guennemann

**Abstract**: As the use of machine learning continues to expand, the importance of ensuring its safety cannot be overstated. A key concern in this regard is the ability to identify whether a given sample is from the training distribution, or is an "Out-Of-Distribution" (OOD) sample. In addition, adversaries can manipulate OOD samples in ways that lead a classifier to make a confident prediction. In this study, we present a novel approach for certifying the robustness of OOD detection within a $\ell_2$-norm around the input, regardless of network architecture and without the need for specific components or additional training. Further, we improve current techniques for detecting adversarial attacks on OOD samples, while providing high levels of certified and adversarial robustness on in-distribution samples. The average of all OOD detection metrics on CIFAR10/100 shows an increase of $\sim 13 \% / 5\%$ relative to previous approaches.

摘要: 随着机器学习的使用不断扩大，确保其安全性的重要性怎么强调都不为过。这方面的一个关键问题是能否识别给定样本是来自训练分布，还是“超出分布”(OOD)样本。此外，攻击者还可以操纵OOD样本，从而使分类器做出可靠的预测。在这项研究中，我们提出了一种新的方法来证明OOD检测的稳健性在输入周围的$\ell_2$-范数内，而与网络体系结构无关，并且不需要特定的组件或额外的训练。此外，我们改进了当前检测OOD样本上的对抗性攻击的技术，同时在分发内样本上提供了高水平的认证和对抗性健壮性。与以前的方法相比，CIFAR10/100上所有OOD检测指标的平均值增加了$\sim 13/5$。



## **3. Hard No-Box Adversarial Attack on Skeleton-Based Human Action Recognition with Skeleton-Motion-Informed Gradient**

基于骨架运动信息梯度的基于骨架的人体动作识别的硬非盒对抗攻击 cs.CV

**SubmitDate**: 2023-08-10    [abs](http://arxiv.org/abs/2308.05681v1) [paper-pdf](http://arxiv.org/pdf/2308.05681v1)

**Authors**: Zhengzhi Lu, He Wang, Ziyi Chang, Guoan Yang, Hubert P. H. Shum

**Abstract**: Recently, methods for skeleton-based human activity recognition have been shown to be vulnerable to adversarial attacks. However, these attack methods require either the full knowledge of the victim (i.e. white-box attacks), access to training data (i.e. transfer-based attacks) or frequent model queries (i.e. black-box attacks). All their requirements are highly restrictive, raising the question of how detrimental the vulnerability is. In this paper, we show that the vulnerability indeed exists. To this end, we consider a new attack task: the attacker has no access to the victim model or the training data or labels, where we coin the term hard no-box attack. Specifically, we first learn a motion manifold where we define an adversarial loss to compute a new gradient for the attack, named skeleton-motion-informed (SMI) gradient. Our gradient contains information of the motion dynamics, which is different from existing gradient-based attack methods that compute the loss gradient assuming each dimension in the data is independent. The SMI gradient can augment many gradient-based attack methods, leading to a new family of no-box attack methods. Extensive evaluation and comparison show that our method imposes a real threat to existing classifiers. They also show that the SMI gradient improves the transferability and imperceptibility of adversarial samples in both no-box and transfer-based black-box settings.

摘要: 最近，基于骨架的人类活动识别方法被证明容易受到对手的攻击。然而，这些攻击方法要么需要受害者的全部知识(即白盒攻击)，要么需要访问训练数据(即基于传输的攻击)，或者需要频繁的模型查询(即黑盒攻击)。他们的所有要求都是高度限制性的，这引发了漏洞的危害性有多大的问题。在本文中，我们证明了该漏洞确实存在。为此，我们考虑了一个新的攻击任务：攻击者无权访问受害者模型、训练数据或标签，其中我们创造了术语硬无盒攻击。具体地说，我们首先学习一个运动流形，其中我们定义了一个对手损失来计算攻击的一个新的梯度，称为骨架-运动信息(SMI)梯度。我们的梯度包含了运动动力学的信息，这不同于现有的基于梯度的攻击方法，该方法假设数据中的每个维度都是独立的，来计算损失梯度。SMI梯度可以扩展许多基于梯度的攻击方法，从而产生一类新的非盒子攻击方法。广泛的评估和比较表明，我们的方法对现有的分类器构成了真正的威胁。它们还表明，SMI梯度在无盒和基于传输的黑盒设置下都改善了对抗性样本的可转移性和不可见性。



## **4. Symmetry Defense Against XGBoost Adversarial Perturbation Attacks**

XGBoost敌意扰动攻击的对称性防御 cs.LG

16 pages

**SubmitDate**: 2023-08-10    [abs](http://arxiv.org/abs/2308.05575v1) [paper-pdf](http://arxiv.org/pdf/2308.05575v1)

**Authors**: Blerta Lindqvist

**Abstract**: We examine whether symmetry can be used to defend tree-based ensemble classifiers such as gradient-boosting decision trees (GBDTs) against adversarial perturbation attacks. The idea is based on a recent symmetry defense for convolutional neural network classifiers (CNNs) that utilizes CNNs' lack of invariance with respect to symmetries. CNNs lack invariance because they can classify a symmetric sample, such as a horizontally flipped image, differently from the original sample. CNNs' lack of invariance also means that CNNs can classify symmetric adversarial samples differently from the incorrect classification of adversarial samples. Using CNNs' lack of invariance, the recent CNN symmetry defense has shown that the classification of symmetric adversarial samples reverts to the correct sample classification. In order to apply the same symmetry defense to GBDTs, we examine GBDT invariance and are the first to show that GBDTs also lack invariance with respect to symmetries. We apply and evaluate the GBDT symmetry defense for nine datasets against six perturbation attacks with a threat model that ranges from zero-knowledge to perfect-knowledge adversaries. Using the feature inversion symmetry against zero-knowledge adversaries, we achieve up to 100% accuracy on adversarial samples even when default and robust classifiers have 0% accuracy. Using the feature inversion and horizontal flip symmetries against perfect-knowledge adversaries, we achieve up to over 95% accuracy on adversarial samples for the GBDT classifier of the F-MNIST dataset even when default and robust classifiers have 0% accuracy.

摘要: 我们研究了对称性是否可以用来保护基于树的集成分类器，如梯度提升决策树(GBDT)，以抵御对抗性扰动攻击。这个想法是基于最近卷积神经网络分类器(CNN)的对称性防御，它利用了CNN关于对称性的不变性。CNN缺乏不变性，因为它们可以对对称样本(例如水平翻转的图像)进行不同于原始样本的分类。CNN的缺乏不变性也意味着CNN可以对对称的对抗性样本进行不同于对对抗性样本的错误分类。利用CNN的不变性，最近的CNN对称性防御表明，对称对抗性样本的分类恢复到正确的样本分类。为了将同样的对称防御应用于GBDT，我们考察了GBDT不变性，并首次证明GBDT也缺乏关于对称性的不变性。在一个从零知识到完全知识对手的威胁模型下，我们对9个数据集上的GBDT对称性防御进行了应用和评估。使用特征倒置对称性对抗零知识对手，即使在默认和稳健的分类器准确率为0%的情况下，我们在敌意样本上也达到了100%的准确率。对于F-MNIST数据集的GBDT分类器，使用特征倒置和水平翻转对称来对抗完美知识对手，即使在默认和健壮的分类器准确率为0%的情况下，我们也可以在对手样本上获得95%以上的准确率。



## **5. Symmetry Defense Against CNN Adversarial Perturbation Attacks**

CNN对抗扰动攻击的对称性防御 cs.LG

19 pages

**SubmitDate**: 2023-08-10    [abs](http://arxiv.org/abs/2210.04087v3) [paper-pdf](http://arxiv.org/pdf/2210.04087v3)

**Authors**: Blerta Lindqvist

**Abstract**: This paper uses symmetry to make Convolutional Neural Network classifiers (CNNs) robust against adversarial perturbation attacks. Such attacks add perturbation to original images to generate adversarial images that fool classifiers such as road sign classifiers of autonomous vehicles. Although symmetry is a pervasive aspect of the natural world, CNNs are unable to handle symmetry well. For example, a CNN can classify an image differently from its mirror image. For an adversarial image that misclassifies with a wrong label $l_w$, CNN inability to handle symmetry means that a symmetric adversarial image can classify differently from the wrong label $l_w$. Further than that, we find that the classification of a symmetric adversarial image reverts to the correct label. To classify an image when adversaries are unaware of the defense, we apply symmetry to the image and use the classification label of the symmetric image. To classify an image when adversaries are aware of the defense, we use mirror symmetry and pixel inversion symmetry to form a symmetry group. We apply all the group symmetries to the image and decide on the output label based on the agreement of any two of the classification labels of the symmetry images. Adaptive attacks fail because they need to rely on loss functions that use conflicting CNN output values for symmetric images. Without attack knowledge, the proposed symmetry defense succeeds against both gradient-based and random-search attacks, with up to near-default accuracies for ImageNet. The defense even improves the classification accuracy of original images.

摘要: 利用对称性使卷积神经网络分类器(CNN)对敌意扰动攻击具有较强的鲁棒性。这种攻击向原始图像添加扰动，以生成欺骗分类器(如自动车辆的路标分类器)的对抗性图像。尽管对称性是自然界普遍存在的一个方面，但CNN无法很好地处理对称性。例如，CNN可以将图像与其镜像图像进行不同的分类。对于带有错误标签$L_w$的对抗性图像，CNN无法处理对称意味着对称对抗性图像可以与错误标签$L_w$进行不同的分类。此外，我们还发现对称对抗性图像的分类还原为正确的标签。为了在攻击者没有意识到防御的情况下对图像进行分类，我们对图像进行对称，并使用对称图像的分类标签。为了在对手意识到防御的情况下对图像进行分类，我们使用镜像对称和像素反转对称来形成对称群。我们将所有的群对称应用于图像，并根据对称图像的任意两个分类标记的一致性来确定输出标记。自适应攻击之所以失败，是因为它们需要依赖对对称图像使用冲突的CNN输出值的损失函数。在没有攻击知识的情况下，所提出的对称防御能够成功地抵抗基于梯度和随机搜索的攻击，对ImageNet的准确率高达接近默认的精度。该防御措施甚至提高了原始图像的分类精度。



## **6. Complex Network Effects on the Robustness of Graph Convolutional Networks**

复杂网络对图卷积网络稳健性的影响 cs.SI

39 pages, 8 figures. arXiv admin note: text overlap with  arXiv:2003.05822

**SubmitDate**: 2023-08-10    [abs](http://arxiv.org/abs/2308.05498v1) [paper-pdf](http://arxiv.org/pdf/2308.05498v1)

**Authors**: Benjamin A. Miller, Kevin Chan, Tina Eliassi-Rad

**Abstract**: Vertex classification -- the problem of identifying the class labels of nodes in a graph -- has applicability in a wide variety of domains. Examples include classifying subject areas of papers in citation networks or roles of machines in a computer network. Vertex classification using graph convolutional networks is susceptible to targeted poisoning attacks, in which both graph structure and node attributes can be changed in an attempt to misclassify a target node. This vulnerability decreases users' confidence in the learning method and can prevent adoption in high-stakes contexts. Defenses have also been proposed, focused on filtering edges before creating the model or aggregating information from neighbors more robustly. This paper considers an alternative: we leverage network characteristics in the training data selection process to improve robustness of vertex classifiers.   We propose two alternative methods of selecting training data: (1) to select the highest-degree nodes and (2) to iteratively select the node with the most neighbors minimally connected to the training set. In the datasets on which the original attack was demonstrated, we show that changing the training set can make the network much harder to attack. To maintain a given probability of attack success, the adversary must use far more perturbations; often a factor of 2--4 over the random training baseline. These training set selection methods often work in conjunction with the best recently published defenses to provide even greater robustness. While increasing the amount of randomly selected training data sometimes results in a more robust classifier, the proposed methods increase robustness substantially more. We also run a simulation study in which we demonstrate conditions under which each of the two methods outperforms the other, controlling for the graph topology, homophily of the labels, and node attributes.

摘要: 顶点分类--识别图中节点的类别标签的问题--在广泛的领域中具有适用性。例如，对引文网络中论文的主题领域或计算机网络中机器的角色进行分类。使用图卷积网络的顶点分类容易受到有针对性的中毒攻击，图结构和节点属性都可能被更改，以尝试对目标节点进行错误分类。此漏洞降低了用户对学习方法的信心，并可能阻止在高风险环境中采用该方法。也有人提出了防御措施，重点是在创建模型之前过滤边缘，或者更有力地从邻居那里聚合信息。本文考虑了一种替代方案：在训练数据选择过程中利用网络特性来提高顶点分类器的稳健性。我们提出了两种选择训练数据的方法：(1)选择次数最高的节点；(2)迭代地选择与训练集连接最少的邻居最多的节点。在演示了原始攻击的数据集中，我们表明更改训练集可以使网络更难攻击。为了保持给定的攻击成功概率，对手必须使用更多的干扰；通常是随机训练基线上的2-4倍。这些训练集选择方法通常与最近发布的最好的防御方法一起工作，以提供更好的健壮性。虽然增加随机选择的训练数据量有时会导致更稳健的分类器，但所提出的方法显著地提高了稳健性。我们还运行了一个模拟研究，其中我们演示了两种方法中的每一种都优于另一种方法的条件，控制了图的拓扑、标签的同质性和节点属性。



## **7. Multi-metrics adaptively identifies backdoors in Federated learning**

多指标自适应地识别联合学习中的后门 cs.CR

14 pages, 8 figures and 7 tables; 2023 IEEE/CVF International  Conference on Computer Vision (ICCV)

**SubmitDate**: 2023-08-10    [abs](http://arxiv.org/abs/2303.06601v2) [paper-pdf](http://arxiv.org/pdf/2303.06601v2)

**Authors**: Siquan Huang, Yijiang Li, Chong Chen, Leyu Shi, Ying Gao

**Abstract**: The decentralized and privacy-preserving nature of federated learning (FL) makes it vulnerable to backdoor attacks aiming to manipulate the behavior of the resulting model on specific adversary-chosen inputs. However, most existing defenses based on statistical differences take effect only against specific attacks, especially when the malicious gradients are similar to benign ones or the data are highly non-independent and identically distributed (non-IID). In this paper, we revisit the distance-based defense methods and discover that i) Euclidean distance becomes meaningless in high dimensions and ii) malicious gradients with diverse characteristics cannot be identified by a single metric. To this end, we present a simple yet effective defense strategy with multi-metrics and dynamic weighting to identify backdoors adaptively. Furthermore, our novel defense has no reliance on predefined assumptions over attack settings or data distributions and little impact on benign performance. To evaluate the effectiveness of our approach, we conduct comprehensive experiments on different datasets under various attack settings, where our method achieves the best defensive performance. For instance, we achieve the lowest backdoor accuracy of 3.06% under the difficult Edge-case PGD, showing significant superiority over previous defenses. The results also demonstrate that our method can be well-adapted to a wide range of non-IID degrees without sacrificing the benign performance.

摘要: 联邦学习(FL)的去中心化和隐私保护特性使其容易受到后门攻击，目的是在特定对手选择的输入上操纵结果模型的行为。然而，现有的大多数基于统计差异的防御措施只对特定的攻击有效，特别是当恶意梯度类似于良性梯度或数据具有高度非独立和同分布(Non-IID)时。在本文中，我们回顾了基于距离的防御方法，发现i)欧氏距离在高维中变得没有意义，ii)具有不同特征的恶意梯度不能用单一的度量来识别。为此，我们提出了一种简单而有效的防御策略，采用多指标和动态加权来自适应地识别后门。此外，我们的新型防御不依赖于对攻击设置或数据分布的预定义假设，并且对良性性能几乎没有影响。为了评估该方法的有效性，我们在不同的攻击环境下对不同的数据集进行了全面的实验，其中我们的方法取得了最好的防御性能。例如，在困难的Edge-Case PGD下，我们实现了3.06%的最低后门精度，显示出明显优于以前的防御。结果还表明，我们的方法可以很好地适应广泛的非IID程度，而不牺牲良好的性能。



## **8. Adv-Inpainting: Generating Natural and Transferable Adversarial Patch via Attention-guided Feature Fusion**

基于注意力引导的特征融合生成自然的、可转移的对抗性补丁 cs.CV

**SubmitDate**: 2023-08-10    [abs](http://arxiv.org/abs/2308.05320v1) [paper-pdf](http://arxiv.org/pdf/2308.05320v1)

**Authors**: Yanjie Li, Mingxing Duan, Bin Xiao

**Abstract**: The rudimentary adversarial attacks utilize additive noise to attack facial recognition (FR) models. However, because manipulating the total face is impractical in the physical setting, most real-world FR attacks are based on adversarial patches, which limit perturbations to a small area. Previous adversarial patch attacks often resulted in unnatural patterns and clear boundaries that were easily noticeable. In this paper, we argue that generating adversarial patches with plausible content can result in stronger transferability than using additive noise or directly sampling from the latent space. To generate natural-looking and highly transferable adversarial patches, we propose an innovative two-stage coarse-to-fine attack framework called Adv-Inpainting. In the first stage, we propose an attention-guided StyleGAN (Att-StyleGAN) that adaptively combines texture and identity features based on the attention map to generate high-transferable and natural adversarial patches. In the second stage, we design a refinement network with a new boundary variance loss to further improve the coherence between the patch and its surrounding area. Experiment results demonstrate that Adv-Inpainting is stealthy and can produce adversarial patches with stronger transferability and improved visual quality than previous adversarial patch attacks.

摘要: 基本的对抗性攻击利用加性噪声攻击面部识别(FR)模型。然而，因为操纵整个人脸在物理环境中是不切实际的，所以大多数现实世界中的FR攻击都是基于对抗性补丁的，这将扰动限制在一个小区域。以前的对抗性补丁攻击通常会导致不自然的模式和清晰的边界，很容易被注意到。在本文中，我们认为，与使用加性噪声或直接从潜在空间采样相比，生成含有可信内容的敌意补丁可以产生更强的可转移性。为了生成看起来自然且高度可转移的敌意补丁，我们提出了一种创新的两阶段由粗到精的攻击框架ADV-INPING。在第一阶段，我们提出了一种注意力引导的StyleGAN(Att-StyleGAN)，它基于注意力图自适应地结合纹理和身份特征来生成高可传递性和自然对抗性的补丁。在第二阶段，我们设计了一个具有新的边界方差损失的细化网络，以进一步提高面片与其周围区域之间的一致性。实验结果表明，该算法具有较好的隐蔽性，生成的对抗性补丁比以往的对抗性补丁攻击具有更强的可传递性和更好的视觉质量。



## **9. Analyzing Privacy Leakage in Machine Learning via Multiple Hypothesis Testing: A Lesson From Fano**

用多重假设检验分析机器学习中的隐私泄露--来自Fano的经验 cs.LG

**SubmitDate**: 2023-08-10    [abs](http://arxiv.org/abs/2210.13662v2) [paper-pdf](http://arxiv.org/pdf/2210.13662v2)

**Authors**: Chuan Guo, Alexandre Sablayrolles, Maziar Sanjabi

**Abstract**: Differential privacy (DP) is by far the most widely accepted framework for mitigating privacy risks in machine learning. However, exactly how small the privacy parameter $\epsilon$ needs to be to protect against certain privacy risks in practice is still not well-understood. In this work, we study data reconstruction attacks for discrete data and analyze it under the framework of multiple hypothesis testing. We utilize different variants of the celebrated Fano's inequality to derive upper bounds on the inferential power of a data reconstruction adversary when the model is trained differentially privately. Importantly, we show that if the underlying private data takes values from a set of size $M$, then the target privacy parameter $\epsilon$ can be $O(\log M)$ before the adversary gains significant inferential power. Our analysis offers theoretical evidence for the empirical effectiveness of DP against data reconstruction attacks even at relatively large values of $\epsilon$.

摘要: 差异隐私(DP)是迄今为止被最广泛接受的减轻机器学习中隐私风险的框架。然而，在实践中，隐私参数$\epsilon$需要多小才能防止某些隐私风险仍然没有得到很好的理解。在本工作中，我们研究了离散数据的数据重构攻击，并在多重假设检验的框架下对其进行了分析。我们利用著名的Fano不等式的不同变体来推导数据重建对手在模型被私人差分训练时的推理能力的上界。重要的是，我们证明了如果底层私有数据取自一组大小为$M$的值，则在对手获得显著的推理能力之前，目标隐私参数$\epsilon$可以是$O(\log M)$。我们的分析为DP抵抗数据重建攻击的经验有效性提供了理论证据，即使在相对较大的$\epsilon$的情况下也是如此。



## **10. Benchmarking and Analyzing Robust Point Cloud Recognition: Bag of Tricks for Defending Adversarial Examples**

基准化和分析稳健的点云识别：防御敌方例子的一袋诡计 cs.CV

8 pages 6 figures

**SubmitDate**: 2023-08-10    [abs](http://arxiv.org/abs/2307.16361v2) [paper-pdf](http://arxiv.org/pdf/2307.16361v2)

**Authors**: Qiufan Ji, Lin Wang, Cong Shi, Shengshan Hu, Yingying Chen, Lichao Sun

**Abstract**: Deep Neural Networks (DNNs) for 3D point cloud recognition are vulnerable to adversarial examples, threatening their practical deployment. Despite the many research endeavors have been made to tackle this issue in recent years, the diversity of adversarial examples on 3D point clouds makes them more challenging to defend against than those on 2D images. For examples, attackers can generate adversarial examples by adding, shifting, or removing points. Consequently, existing defense strategies are hard to counter unseen point cloud adversarial examples. In this paper, we first establish a comprehensive, and rigorous point cloud adversarial robustness benchmark to evaluate adversarial robustness, which can provide a detailed understanding of the effects of the defense and attack methods. We then collect existing defense tricks in point cloud adversarial defenses and then perform extensive and systematic experiments to identify an effective combination of these tricks. Furthermore, we propose a hybrid training augmentation methods that consider various types of point cloud adversarial examples to adversarial training, significantly improving the adversarial robustness. By combining these tricks, we construct a more robust defense framework achieving an average accuracy of 83.45\% against various attacks, demonstrating its capability to enabling robust learners. Our codebase are open-sourced on: \url{https://github.com/qiufan319/benchmark_pc_attack.git}.

摘要: 用于三维点云识别的深度神经网络(DNN)容易受到敌意例子的攻击，威胁到其实际应用。尽管近年来已经进行了许多研究工作来解决这个问题，但3D点云上的对抗性例子的多样性使得它们比2D图像上的更具挑战性。例如，攻击者可以通过添加、移动或删除点来生成对抗性示例。因此，现有的防御策略很难对抗看不见的点云对抗性例子。本文首先建立了一个全面、严谨的点云对抗健壮性基准来评估对抗健壮性，它可以提供对防御和攻击方法效果的详细了解。然后，我们收集现有的点云对抗防御中的防御技巧，然后进行广泛和系统的实验，以确定这些技巧的有效组合。此外，我们还提出了一种混合训练增强方法，将各种类型的点云对抗性样本考虑到对抗性训练中，显著提高了对抗性训练的健壮性。通过将这些技巧结合起来，我们构建了一个更健壮的防御框架，对各种攻击的平均准确率达到了83.45\%，展示了它对健壮学习者的能力。我们的代码库在\url{https://github.com/qiufan319/benchmark_pc_attack.git}.上是开源的



## **11. Byzantine-Robust Decentralized Stochastic Optimization with Stochastic Gradient Noise-Independent Learning Error**

学习误差与随机梯度噪声无关的拜占庭分散随机优化 cs.LG

**SubmitDate**: 2023-08-10    [abs](http://arxiv.org/abs/2308.05292v1) [paper-pdf](http://arxiv.org/pdf/2308.05292v1)

**Authors**: Jie Peng, Weiyu Li, Qing Ling

**Abstract**: This paper studies Byzantine-robust stochastic optimization over a decentralized network, where every agent periodically communicates with its neighbors to exchange local models, and then updates its own local model by stochastic gradient descent (SGD). The performance of such a method is affected by an unknown number of Byzantine agents, which conduct adversarially during the optimization process. To the best of our knowledge, there is no existing work that simultaneously achieves a linear convergence speed and a small learning error. We observe that the learning error is largely dependent on the intrinsic stochastic gradient noise. Motivated by this observation, we introduce two variance reduction methods, stochastic average gradient algorithm (SAGA) and loopless stochastic variance-reduced gradient (LSVRG), to Byzantine-robust decentralized stochastic optimization for eliminating the negative effect of the stochastic gradient noise. The two resulting methods, BRAVO-SAGA and BRAVO-LSVRG, enjoy both linear convergence speeds and stochastic gradient noise-independent learning errors. Such learning errors are optimal for a class of methods based on total variation (TV)-norm regularization and stochastic subgradient update. We conduct extensive numerical experiments to demonstrate their effectiveness under various Byzantine attacks.

摘要: 研究了分散网络上的拜占庭-稳健随机优化问题，其中每个代理周期性地与其邻居通信以交换局部模型，然后利用随机梯度下降(SGD)来更新自己的局部模型。这种方法的性能受到未知数量的拜占庭代理的影响，这些拜占庭代理在优化过程中进行相反的操作。就我们所知，目前还没有同时达到线性收敛速度和小学习误差的现有工作。我们观察到学习误差在很大程度上依赖于固有的随机梯度噪声。受此启发，我们将随机平均梯度算法(SAGA)和随机减方差梯度算法(LSVRG)引入拜占庭-稳健分散随机优化中，以消除随机梯度噪声的负面影响。得到的两种方法，Bravo-SAGA和Bravo-LSVRG，既具有线性收敛速度，又具有与噪声无关的随机梯度学习误差。对于一类基于全变分(TV)范数正则化和随机次梯度更新的方法，这样的学习误差是最优的。我们进行了大量的数值实验，以验证它们在各种拜占庭攻击下的有效性。



## **12. Risk-based Security Measure Allocation Against Injection Attacks on Actuators**

基于风险的执行器注入攻击安全措施分配 eess.SY

Accepted to IEEE Open Journal of Control Systems (OJ-CSYS)

**SubmitDate**: 2023-08-09    [abs](http://arxiv.org/abs/2304.02055v2) [paper-pdf](http://arxiv.org/pdf/2304.02055v2)

**Authors**: Sribalaji C. Anand, André M. H. Teixeira

**Abstract**: This article considers the problem of risk-optimal allocation of security measures when the actuators of an uncertain control system are under attack. We consider an adversary injecting false data into the actuator channels. The attack impact is characterized by the maximum performance loss caused by a stealthy adversary with bounded energy. Since the impact is a random variable, due to system uncertainty, we use Conditional Value-at-Risk (CVaR) to characterize the risk associated with the attack. We then consider the problem of allocating the security measures which minimize the risk. We assume that there are only a limited number of security measures available. Under this constraint, we observe that the allocation problem is a mixed-integer optimization problem. Thus we use relaxation techniques to approximate the security allocation problem into a Semi-Definite Program (SDP). We also compare our allocation method $(i)$ across different risk measures: the worst-case measure, the average (nominal) measure, and $(ii)$ across different search algorithms: the exhaustive and the greedy search algorithms. We depict the efficacy of our approach through numerical examples.

摘要: 研究了不确定控制系统执行器受到攻击时，安全措施的风险最优分配问题。我们认为对手将错误数据注入致动器通道。攻击影响的特征是具有有限能量的隐形对手所造成的最大性能损失。由于影响是一个随机变量，由于系统的不确定性，我们使用条件风险值(CVAR)来表征与攻击相关的风险。然后，我们考虑分配将风险降至最低的安全措施的问题。我们假设只有有限数量的安全措施可用。在此约束下，我们观察到分配问题是一个混合整数优化问题。因此，我们使用松弛技术将安全分配问题近似为半定规划(SDP)。我们还比较了我们的分配方法$(I)$跨不同的风险度量：最坏情况度量、平均(名义)度量，以及$(Ii)$跨不同的搜索算法：穷举搜索算法和贪婪搜索算法。我们通过数值例子描述了我们方法的有效性。



## **13. Do Perceptually Aligned Gradients Imply Adversarial Robustness?**

感知上对齐的梯度是否意味着对抗的健壮性？ cs.CV

**SubmitDate**: 2023-08-09    [abs](http://arxiv.org/abs/2207.11378v3) [paper-pdf](http://arxiv.org/pdf/2207.11378v3)

**Authors**: Roy Ganz, Bahjat Kawar, Michael Elad

**Abstract**: Adversarially robust classifiers possess a trait that non-robust models do not -- Perceptually Aligned Gradients (PAG). Their gradients with respect to the input align well with human perception. Several works have identified PAG as a byproduct of robust training, but none have considered it as a standalone phenomenon nor studied its own implications. In this work, we focus on this trait and test whether \emph{Perceptually Aligned Gradients imply Robustness}. To this end, we develop a novel objective to directly promote PAG in training classifiers and examine whether models with such gradients are more robust to adversarial attacks. Extensive experiments on multiple datasets and architectures validate that models with aligned gradients exhibit significant robustness, exposing the surprising bidirectional connection between PAG and robustness. Lastly, we show that better gradient alignment leads to increased robustness and harness this observation to boost the robustness of existing adversarial training techniques.

摘要: 相反，稳健的分类器具有非稳健模型所没有的特征--感知对齐梯度(PAG)。它们相对于输入的梯度很好地符合人类的感知。一些研究认为PAG是强健训练的副产品，但没有一篇认为它是一个独立的现象，也没有研究它本身的影响。在这项工作中，我们专注于这一特征，并测试\emph{感知对齐的梯度是否意味着稳健性}。为此，我们提出了一个新的目标，即在训练分类器时直接推广PAG，并检验具有这种梯度的模型是否对对手攻击更健壮。在多个数据集和体系结构上的广泛实验验证了具有对齐梯度的模型表现出显著的稳健性，揭示了PAG和稳健性之间令人惊讶的双向联系。最后，我们证明了更好的梯度对齐可以提高健壮性，并利用这一观察结果来提高现有对抗性训练技术的健壮性。



## **14. Adversarial ModSecurity: Countering Adversarial SQL Injections with Robust Machine Learning**

敌意模式安全：用健壮的机器学习来对抗敌意SQL注入 cs.LG

**SubmitDate**: 2023-08-09    [abs](http://arxiv.org/abs/2308.04964v1) [paper-pdf](http://arxiv.org/pdf/2308.04964v1)

**Authors**: Biagio Montaruli, Luca Demetrio, Andrea Valenza, Battista Biggio, Luca Compagna, Davide Balzarotti, Davide Ariu, Luca Piras

**Abstract**: ModSecurity is widely recognized as the standard open-source Web Application Firewall (WAF), maintained by the OWASP Foundation. It detects malicious requests by matching them against the Core Rule Set, identifying well-known attack patterns. Each rule in the CRS is manually assigned a weight, based on the severity of the corresponding attack, and a request is detected as malicious if the sum of the weights of the firing rules exceeds a given threshold. In this work, we show that this simple strategy is largely ineffective for detecting SQL injection (SQLi) attacks, as it tends to block many legitimate requests, while also being vulnerable to adversarial SQLi attacks, i.e., attacks intentionally manipulated to evade detection. To overcome these issues, we design a robust machine learning model, named AdvModSec, which uses the CRS rules as input features, and it is trained to detect adversarial SQLi attacks. Our experiments show that AdvModSec, being trained on the traffic directed towards the protected web services, achieves a better trade-off between detection and false positive rates, improving the detection rate of the vanilla version of ModSecurity with CRS by 21%. Moreover, our approach is able to improve its adversarial robustness against adversarial SQLi attacks by 42%, thereby taking a step forward towards building more robust and trustworthy WAFs.

摘要: ModSecurity被广泛认为是标准的开源Web应用程序防火墙(WAF)，由OWASP基金会维护。它通过将恶意请求与核心规则集进行匹配来检测恶意请求，识别众所周知的攻击模式。CRS中的每个规则都根据相应攻击的严重性被手动分配一个权重，如果触发规则的权重之和超过给定阈值，则检测到请求是恶意的。在这项工作中，我们证明了这种简单的策略对于检测SQL注入(SQLI)攻击是非常无效的，因为它倾向于阻止许多合法的请求，同时也容易受到敌意的SQLI攻击，即被故意操纵以逃避检测的攻击。为了克服这些问题，我们设计了一个健壮的机器学习模型AdvMoSec，该模型使用CRS规则作为输入特征，并被训练来检测敌意SQLI攻击。我们的实验表明，通过对指向受保护Web服务的流量进行训练，AdvmodSec在检测和误检率之间实现了更好的权衡，将带有CRS的普通版本的ModSecurity的检测率提高了21%。此外，我们的方法能够将其对敌对SQLI攻击的对抗健壮性提高42%，从而朝着建立更健壮和可信的WAFs迈进了一步。



## **15. Adversarial Deep Reinforcement Learning for Cyber Security in Software Defined Networks**

软件定义网络中网络安全的对抗性深度强化学习 cs.CR

**SubmitDate**: 2023-08-09    [abs](http://arxiv.org/abs/2308.04909v1) [paper-pdf](http://arxiv.org/pdf/2308.04909v1)

**Authors**: Luke Borchjes, Clement Nyirenda, Louise Leenen

**Abstract**: This paper focuses on the impact of leveraging autonomous offensive approaches in Deep Reinforcement Learning (DRL) to train more robust agents by exploring the impact of applying adversarial learning to DRL for autonomous security in Software Defined Networks (SDN). Two algorithms, Double Deep Q-Networks (DDQN) and Neural Episodic Control to Deep Q-Network (NEC2DQN or N2D), are compared. NEC2DQN was proposed in 2018 and is a new member of the deep q-network (DQN) family of algorithms. The attacker has full observability of the environment and access to a causative attack that uses state manipulation in an attempt to poison the learning process. The implementation of the attack is done under a white-box setting, in which the attacker has access to the defender's model and experiences. Two games are played; in the first game, DDQN is a defender and N2D is an attacker, and in second game, the roles are reversed. The games are played twice; first, without an active causative attack and secondly, with an active causative attack. For execution, three sets of game results are recorded in which a single set consists of 10 game runs. The before and after results are then compared in order to see if there was actually an improvement or degradation. The results show that with minute parameter changes made to the algorithms, there was growth in the attacker's role, since it is able to win games. Implementation of the adversarial learning by the introduction of the causative attack showed the algorithms are still able to defend the network according to their strengths.

摘要: 通过探讨将对抗性学习应用于深度强化学习(DRL)对软件定义网络(SDN)中自主安全的影响，重点研究了在深度强化学习(DRL)中利用自主进攻方法来训练更健壮的代理的影响。对双深度Q网络(DDQN)和神经情节控制深度Q网络(NEC2DQN或N2D)两种算法进行了比较。NEC2DQN于2018年提出，是深度Q网络(DQN)算法家族的新成员。攻击者对环境具有完全的可观察性，并且可以访问使用状态操纵来试图毒化学习过程的原因攻击。攻击的实施是在白盒设置下完成的，在白盒设置中，攻击者可以访问防御者的模型和经验。在第一场比赛中，DDQN是防守者，N2D是攻击手，在第二场比赛中，角色互换。游戏进行两次；第一次，没有主动的因果攻击，第二次，主动的因果攻击。对于执行，记录三组游戏结果，其中一组由10个游戏运行组成。然后将前后的结果进行比较，以确定是否确实有所改善或下降。结果表明，只要对算法进行微小的参数更改，攻击者的角色就会增加，因为它能够赢得比赛。通过引入致因攻击来实现对抗性学习，表明算法仍然能够根据自身的优势对网络进行防御。



## **16. Rickrolling the Artist: Injecting Backdoors into Text Encoders for Text-to-Image Synthesis**

点击艺术家：将后门注入文本编码器以进行文本到图像的合成 cs.LG

Published as a conference paper at ICCV 2023

**SubmitDate**: 2023-08-09    [abs](http://arxiv.org/abs/2211.02408v3) [paper-pdf](http://arxiv.org/pdf/2211.02408v3)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Kristian Kersting

**Abstract**: While text-to-image synthesis currently enjoys great popularity among researchers and the general public, the security of these models has been neglected so far. Many text-guided image generation models rely on pre-trained text encoders from external sources, and their users trust that the retrieved models will behave as promised. Unfortunately, this might not be the case. We introduce backdoor attacks against text-guided generative models and demonstrate that their text encoders pose a major tampering risk. Our attacks only slightly alter an encoder so that no suspicious model behavior is apparent for image generations with clean prompts. By then inserting a single character trigger into the prompt, e.g., a non-Latin character or emoji, the adversary can trigger the model to either generate images with pre-defined attributes or images following a hidden, potentially malicious description. We empirically demonstrate the high effectiveness of our attacks on Stable Diffusion and highlight that the injection process of a single backdoor takes less than two minutes. Besides phrasing our approach solely as an attack, it can also force an encoder to forget phrases related to certain concepts, such as nudity or violence, and help to make image generation safer.

摘要: 虽然文本到图像的合成目前在研究人员和普通大众中很受欢迎，但到目前为止，这些模型的安全性一直被忽视。许多文本制导的图像生成模型依赖于来自外部来源的预先训练的文本编码器，它们的用户相信检索到的模型将如承诺的那样运行。不幸的是，情况可能并非如此。我们引入了针对文本引导的生成模型的后门攻击，并证明了它们的文本编码器构成了主要的篡改风险。我们的攻击只略微改变了编码器，因此对于具有干净提示的图像生成来说，没有明显的可疑模型行为。然后，通过将单个字符触发器(例如，非拉丁字符或表情符号)插入到提示中，对手可以触发模型以生成具有预定义属性的图像或者在隐藏的潜在恶意描述之后生成图像。我们从经验上证明了我们对稳定扩散攻击的高效性，并强调了单个后门的注入过程只需不到两分钟。除了将我们的方法仅作为一种攻击来表述外，它还可以迫使编码者忘记与某些概念相关的短语，如裸体或暴力，并有助于使图像生成更安全。



## **17. Data-Free Model Extraction Attacks in the Context of Object Detection**

目标检测环境下的无数据模型提取攻击 cs.CR

Submitted to The 14th International Conference on Computer Vision  Systems (ICVS 2023), to be published in Springer, Lecture Notes in Computer  Science

**SubmitDate**: 2023-08-09    [abs](http://arxiv.org/abs/2308.05127v1) [paper-pdf](http://arxiv.org/pdf/2308.05127v1)

**Authors**: Harshit Shah, Aravindhan G, Pavan Kulkarni, Yuvaraj Govidarajulu, Manojkumar Parmar

**Abstract**: A significant number of machine learning models are vulnerable to model extraction attacks, which focus on stealing the models by using specially curated queries against the target model. This task is well accomplished by using part of the training data or a surrogate dataset to train a new model that mimics a target model in a white-box environment. In pragmatic situations, however, the target models are trained on private datasets that are inaccessible to the adversary. The data-free model extraction technique replaces this problem when it comes to using queries artificially curated by a generator similar to that used in Generative Adversarial Nets. We propose for the first time, to the best of our knowledge, an adversary black box attack extending to a regression problem for predicting bounding box coordinates in object detection. As part of our study, we found that defining a loss function and using a novel generator setup is one of the key aspects in extracting the target model. We find that the proposed model extraction method achieves significant results by using reasonable queries. The discovery of this object detection vulnerability will support future prospects for securing such models.

摘要: 相当多的机器学习模型容易受到模型提取攻击，这些攻击集中于通过对目标模型使用经过特殊策划的查询来窃取模型。通过使用部分训练数据或代理数据集来训练模仿白盒环境中的目标模型的新模型，可以很好地完成这一任务。然而，在实际情况下，目标模型是在对手无法访问的私有数据集上进行训练的。当涉及到使用由生成器人工管理的查询时，无数据模型提取技术取代了这个问题，该生成器类似于生成性对抗网中使用的生成器。据我们所知，我们首次提出了一种对手黑盒攻击，扩展到目标检测中预测包围盒坐标的回归问题。作为我们研究的一部分，我们发现定义损失函数和使用新的发生器设置是提取目标模型的关键方面之一。我们发现，通过使用合理的查询，该模型提取方法取得了显著的效果。该对象检测漏洞的发现将支持保护此类模型的未来前景。



## **18. GIFD: A Generative Gradient Inversion Method with Feature Domain Optimization**

GIFD：一种基于特征域优化的产生式梯度反演方法 cs.CV

ICCV 2023

**SubmitDate**: 2023-08-09    [abs](http://arxiv.org/abs/2308.04699v1) [paper-pdf](http://arxiv.org/pdf/2308.04699v1)

**Authors**: Hao Fang, Bin Chen, Xuan Wang, Zhi Wang, Shu-Tao Xia

**Abstract**: Federated Learning (FL) has recently emerged as a promising distributed machine learning framework to preserve clients' privacy, by allowing multiple clients to upload the gradients calculated from their local data to a central server. Recent studies find that the exchanged gradients also take the risk of privacy leakage, e.g., an attacker can invert the shared gradients and recover sensitive data against an FL system by leveraging pre-trained generative adversarial networks (GAN) as prior knowledge. However, performing gradient inversion attacks in the latent space of the GAN model limits their expression ability and generalizability. To tackle these challenges, we propose \textbf{G}radient \textbf{I}nversion over \textbf{F}eature \textbf{D}omains (GIFD), which disassembles the GAN model and searches the feature domains of the intermediate layers. Instead of optimizing only over the initial latent code, we progressively change the optimized layer, from the initial latent space to intermediate layers closer to the output images. In addition, we design a regularizer to avoid unreal image generation by adding a small ${l_1}$ ball constraint to the searching range. We also extend GIFD to the out-of-distribution (OOD) setting, which weakens the assumption that the training sets of GANs and FL tasks obey the same data distribution. Extensive experiments demonstrate that our method can achieve pixel-level reconstruction and is superior to the existing methods. Notably, GIFD also shows great generalizability under different defense strategy settings and batch sizes.

摘要: 联合学习(FL)是最近出现的一种很有前途的分布式机器学习框架，通过允许多个客户端将从他们的本地数据计算出的梯度上传到中央服务器，来保护客户的隐私。最近的研究发现，交换的梯度也存在隐私泄露的风险，例如，攻击者可以利用预先训练的生成性对抗网络(GAN)作为先验知识来反转共享的梯度，并针对FL系统恢复敏感数据。然而，在GaN模型的潜在空间中进行梯度反转攻击限制了其表达能力和泛化能力。为了解决这些问题，我们提出了一种新的GIFD算法，即通过分解GaN模型并搜索中间层的特征域来实现对Textbf{F}eature\Textbf{D}omains的转换。我们不是只对初始的潜在代码进行优化，而是逐步地将优化的层从初始的潜在空间更改为更接近输出图像的中间层。此外，通过在搜索范围中添加一个较小的${L_1}$球约束，我们设计了一个正则化算法来避免产生虚幻图像。我们还将GIFD扩展到超出分布(OOD)环境，削弱了GANS和FL任务的训练集服从相同数据分布的假设。大量实验表明，该方法能够实现像素级重建，且优于现有方法。值得注意的是，GIFD在不同的防御策略设置和批量大小下也显示出很好的通用性。



## **19. SSL-Auth: An Authentication Framework by Fragile Watermarking for Pre-trained Encoders in Self-supervised Learning**

SSL-AUTH：一种基于脆弱水印的自监督学习编码认证框架 cs.CR

**SubmitDate**: 2023-08-09    [abs](http://arxiv.org/abs/2308.04673v1) [paper-pdf](http://arxiv.org/pdf/2308.04673v1)

**Authors**: Xiaobei Li, Changchun Yin, Liming Fang, Run Wang, Chenhao Lin

**Abstract**: Self-supervised learning (SSL) which leverages unlabeled datasets for pre-training powerful encoders has achieved significant success in recent years. These encoders are commonly used as feature extractors for various downstream tasks, requiring substantial data and computing resources for their training process. With the deployment of pre-trained encoders in commercial use, protecting the intellectual property of model owners and ensuring the trustworthiness of the models becomes crucial. Recent research has shown that encoders are threatened by backdoor attacks, adversarial attacks, etc. Therefore, a scheme to verify the integrity of pre-trained encoders is needed to protect users. In this paper, we propose SSL-Auth, the first fragile watermarking scheme for verifying the integrity of encoders without compromising model performance. Our method utilizes selected key samples as watermark information and trains a verification network to reconstruct the watermark information, thereby verifying the integrity of the encoder. By comparing the reconstruction results of the key samples, malicious modifications can be effectively detected, as altered models should not exhibit similar reconstruction performance as the original models. Extensive evaluations on various models and diverse datasets demonstrate the effectiveness and fragility of our proposed SSL-Auth.

摘要: 自监督学习(SSL)利用未标记的数据集对强大的编码器进行预训练，近年来取得了显着的成功。这些编码器通常用作各种下游任务的特征提取器，需要大量数据和计算资源用于其训练过程。随着预先培训的编码器在商业使用中的部署，保护模型所有者的知识产权和确保模型的可信性变得至关重要。最近的研究表明，编码器受到后门攻击、对抗性攻击等的威胁。因此，需要一种方案来验证预先训练的编码器的完整性来保护用户。本文提出了第一个在不影响模型性能的情况下验证编码器完整性的脆弱数字水印方案--SSL-Auth。该方法利用选取的关键样本作为水印信息，训练验证网络重构水印信息，从而验证编码器的完整性。通过比较关键样本的重建结果，可以有效地检测恶意修改，因为改变的模型不应该表现出与原始模型相似的重建性能。在不同模型和不同数据集上的广泛评估证明了我们提出的SSL-Auth的有效性和脆弱性。



## **20. Pelta: Shielding Transformers to Mitigate Evasion Attacks in Federated Learning**

Pelta：在联合学习中屏蔽变形金刚以减少逃避攻击 cs.LG

**SubmitDate**: 2023-08-08    [abs](http://arxiv.org/abs/2308.04373v1) [paper-pdf](http://arxiv.org/pdf/2308.04373v1)

**Authors**: Simon Queyrut, Yérom-David Bromberg, Valerio Schiavoni

**Abstract**: The main premise of federated learning is that machine learning model updates are computed locally, in particular to preserve user data privacy, as those never leave the perimeter of their device. This mechanism supposes the general model, once aggregated, to be broadcast to collaborating and non malicious nodes. However, without proper defenses, compromised clients can easily probe the model inside their local memory in search of adversarial examples. For instance, considering image-based applications, adversarial examples consist of imperceptibly perturbed images (to the human eye) misclassified by the local model, which can be later presented to a victim node's counterpart model to replicate the attack. To mitigate such malicious probing, we introduce Pelta, a novel shielding mechanism leveraging trusted hardware. By harnessing the capabilities of Trusted Execution Environments (TEEs), Pelta masks part of the back-propagation chain rule, otherwise typically exploited by attackers for the design of malicious samples. We evaluate Pelta on a state of the art ensemble model and demonstrate its effectiveness against the Self Attention Gradient adversarial Attack.

摘要: 联合学习的主要前提是机器学习模型更新是在本地计算的，特别是为了保护用户数据隐私，因为那些人永远不会离开他们的设备。该机制假定通用模型一旦聚合，将被广播到协作和非恶意节点。然而，如果没有适当的防御措施，受攻击的客户可以很容易地在他们的本地内存中探测模型，以搜索对抗性的例子。例如，考虑到基于图像的应用，敌意的例子包括被本地模型错误分类的(对人眼)不可察觉的扰动图像，这些图像稍后可以被呈现给受害者节点的对应模型以复制攻击。为了缓解这种恶意探测，我们引入了PELTA，一种利用可信硬件的新型屏蔽机制。通过利用可信执行环境(TEE)的功能，Pelta屏蔽了反向传播链规则的一部分，否则通常会被攻击者用来设计恶意样本。我们在一个最新的集成模型上对Pelta进行了评估，并证明了它对自我注意梯度对手攻击的有效性。



## **21. Accurate, Explainable, and Private Models: Providing Recourse While Minimizing Training Data Leakage**

准确、可解释和私有的模型：在提供资源的同时最大限度地减少训练数据泄露 cs.LG

Proceedings of The Second Workshop on New Frontiers in Adversarial  Machine Learning (AdvML-Frontiers @ ICML 2023)

**SubmitDate**: 2023-08-08    [abs](http://arxiv.org/abs/2308.04341v1) [paper-pdf](http://arxiv.org/pdf/2308.04341v1)

**Authors**: Catherine Huang, Chelse Swoopes, Christina Xiao, Jiaqi Ma, Himabindu Lakkaraju

**Abstract**: Machine learning models are increasingly utilized across impactful domains to predict individual outcomes. As such, many models provide algorithmic recourse to individuals who receive negative outcomes. However, recourse can be leveraged by adversaries to disclose private information. This work presents the first attempt at mitigating such attacks. We present two novel methods to generate differentially private recourse: Differentially Private Model (DPM) and Laplace Recourse (LR). Using logistic regression classifiers and real world and synthetic datasets, we find that DPM and LR perform well in reducing what an adversary can infer, especially at low FPR. When training dataset size is large enough, we find particular success in preventing privacy leakage while maintaining model and recourse accuracy with our novel LR method.

摘要: 机器学习模型越来越多地应用于有影响的领域来预测个人结果。因此，许多模型为收到负面结果的个人提供了算法求助。然而，对手可以利用追索权来泄露私人信息。这项工作是减轻此类攻击的第一次尝试。我们提出了两种产生差异私有资源的新方法：差异私有模型(DPM)和拉普拉斯资源(LR)。使用Logistic回归分类器以及真实世界和合成数据集，我们发现DPM和LR在减少对手可以推断的东西方面表现良好，特别是在低FPR的情况下。当训练数据集足够大时，我们发现新的LR方法在防止隐私泄露方面取得了特别成功的效果，同时保持了模型和资源的准确性。



## **22. Why Does Little Robustness Help? Understanding Adversarial Transferability From Surrogate Training**

为什么小健壮性会有帮助？从替补训练看对手的转换性 cs.LG

Accepted by IEEE Symposium on Security and Privacy (Oakland) 2024; 21  pages, 11 figures, 13 tables

**SubmitDate**: 2023-08-08    [abs](http://arxiv.org/abs/2307.07873v3) [paper-pdf](http://arxiv.org/pdf/2307.07873v3)

**Authors**: Yechao Zhang, Shengshan Hu, Leo Yu Zhang, Junyu Shi, Minghui Li, Xiaogeng Liu, Wei Wan, Hai Jin

**Abstract**: Adversarial examples (AEs) for DNNs have been shown to be transferable: AEs that successfully fool white-box surrogate models can also deceive other black-box models with different architectures. Although a bunch of empirical studies have provided guidance on generating highly transferable AEs, many of these findings lack explanations and even lead to inconsistent advice. In this paper, we take a further step towards understanding adversarial transferability, with a particular focus on surrogate aspects. Starting from the intriguing little robustness phenomenon, where models adversarially trained with mildly perturbed adversarial samples can serve as better surrogates, we attribute it to a trade-off between two predominant factors: model smoothness and gradient similarity. Our investigations focus on their joint effects, rather than their separate correlations with transferability. Through a series of theoretical and empirical analyses, we conjecture that the data distribution shift in adversarial training explains the degradation of gradient similarity. Building on these insights, we explore the impacts of data augmentation and gradient regularization on transferability and identify that the trade-off generally exists in the various training mechanisms, thus building a comprehensive blueprint for the regulation mechanism behind transferability. Finally, we provide a general route for constructing better surrogates to boost transferability which optimizes both model smoothness and gradient similarity simultaneously, e.g., the combination of input gradient regularization and sharpness-aware minimization (SAM), validated by extensive experiments. In summary, we call for attention to the united impacts of these two factors for launching effective transfer attacks, rather than optimizing one while ignoring the other, and emphasize the crucial role of manipulating surrogate models.

摘要: DNN的对抗性例子(AE)已被证明是可移植的：成功欺骗白盒代理模型的AES也可以欺骗其他具有不同体系结构的黑盒模型。尽管大量的实证研究为生成高度可转移的企业实体提供了指导，但其中许多发现缺乏解释，甚至导致了不一致的建议。在这篇文章中，我们进一步了解对手的可转移性，特别关注代理方面。从有趣的小鲁棒性现象开始，我们将其归因于两个主要因素之间的权衡：模型的光滑性和梯度相似性。我们的研究重点是它们的联合影响，而不是它们与可转让性的单独关联。通过一系列的理论和实证分析，我们推测对抗性训练中的数据分布转移解释了梯度相似度的下降。基于这些见解，我们探讨了数据扩充和梯度规则化对可转移性的影响，并确定了各种培训机制中普遍存在的权衡，从而构建了可转移性背后的监管机制的全面蓝图。最后，我们提供了一条同时优化模型光滑性和梯度相似性的构造更好的代理以提高可转移性的一般路线，例如输入梯度正则化和锐度感知最小化(SAM)的组合，并通过大量的实验进行了验证。总之，我们呼吁注意这两个因素对发动有效转移攻击的联合影响，而不是优化一个而忽略另一个，并强调操纵代理模型的关键作用。



## **23. FLIRT: Feedback Loop In-context Red Teaming**

调情：反馈环路情景中的红色团队 cs.AI

**SubmitDate**: 2023-08-08    [abs](http://arxiv.org/abs/2308.04265v1) [paper-pdf](http://arxiv.org/pdf/2308.04265v1)

**Authors**: Ninareh Mehrabi, Palash Goyal, Christophe Dupuy, Qian Hu, Shalini Ghosh, Richard Zemel, Kai-Wei Chang, Aram Galstyan, Rahul Gupta

**Abstract**: Warning: this paper contains content that may be inappropriate or offensive.   As generative models become available for public use in various applications, testing and analyzing vulnerabilities of these models has become a priority. Here we propose an automatic red teaming framework that evaluates a given model and exposes its vulnerabilities against unsafe and inappropriate content generation. Our framework uses in-context learning in a feedback loop to red team models and trigger them into unsafe content generation. We propose different in-context attack strategies to automatically learn effective and diverse adversarial prompts for text-to-image models. Our experiments demonstrate that compared to baseline approaches, our proposed strategy is significantly more effective in exposing vulnerabilities in Stable Diffusion (SD) model, even when the latter is enhanced with safety features. Furthermore, we demonstrate that the proposed framework is effective for red teaming text-to-text models, resulting in significantly higher toxic response generation rate compared to previously reported numbers.

摘要: 警告：本文包含可能不恰当或冒犯性的内容。随着产生式模型在各种应用程序中的公共使用，测试和分析这些模型的漏洞已成为当务之急。在这里，我们提出了一个自动的红色团队框架，它评估一个给定的模型，并暴露其针对不安全和不适当的内容生成的漏洞。我们的框架在RED团队模型的反馈循环中使用情境学习，并触发它们生成不安全的内容。我们提出了不同的上下文攻击策略，以自动学习有效和多样化的文本到图像模型的对抗性提示。我们的实验表明，与基线方法相比，我们提出的策略在暴露稳定扩散(SD)模型中的漏洞方面明显更有效，即使后者被增强了安全特征。此外，我们证明了所提出的框架对于红色团队文本到文本模型是有效的，与先前报道的数字相比，导致了显著更高的毒性反应生成率。



## **24. A semantic backdoor attack against Graph Convolutional Networks**

一种针对图卷积网络的语义后门攻击 cs.LG

**SubmitDate**: 2023-08-08    [abs](http://arxiv.org/abs/2302.14353v3) [paper-pdf](http://arxiv.org/pdf/2302.14353v3)

**Authors**: Jiazhu Dai, Zhipeng Xiong

**Abstract**: Graph convolutional networks (GCNs) have been very effective in addressing the issue of various graph-structured related tasks, such as node classification and graph classification. However, recent research has shown that GCNs are vulnerable to a new type of threat called a backdoor attack, where the adversary can inject a hidden backdoor into GCNs so that the attacked model performs well on benign samples, but its prediction will be maliciously changed to the attacker-specified target label if the hidden backdoor is activated by the attacker-defined trigger. In this paper, we investigate whether such semantic backdoor attacks are possible for GCNs and propose a semantic backdoor attack against GCNs (SBAG) under the context of graph classification to reveal the existence of this security vulnerability in GCNs. SBAG uses a certain type of node in the samples as a backdoor trigger and injects a hidden backdoor into GCN models by poisoning training data. The backdoor will be activated, and the GCN models will give malicious classification results specified by the attacker even on unmodified samples as long as the samples contain enough trigger nodes. We evaluate SBAG on four graph datasets. The experimental results indicate that SBAG can achieve attack success rates of approximately 99.9% and over 82% for two kinds of attack samples, respectively, with poisoning rates of less than 5%.

摘要: 图卷积网络(GCNS)在解决各种与图结构相关的任务，如节点分类和图分类方面已经非常有效。然而，最近的研究表明，GCNS容易受到一种名为后门攻击的新型威胁的攻击，在这种威胁中，攻击者可以向GCNS注入隐藏的后门，以便攻击模型在良性样本上执行良好，但如果隐藏的后门被攻击者定义的触发器激活，其预测将被恶意更改为攻击者指定的目标标签。本文研究了GCNS是否存在这样的语义后门攻击，并在图分类的背景下提出了一种针对GCNS的语义后门攻击(SBAG)，以揭示GCNS中这一安全漏洞的存在。SBag使用样本中的某一类型节点作为后门触发器，并通过毒化训练数据向GCN模型注入隐藏的后门。后门将被激活，GCN模型将给出攻击者指定的恶意分类结果，即使是在未经修改的样本上，只要这些样本包含足够的触发节点。我们在四个图数据集上对SBAG进行了评估。实验结果表明，该算法对两种攻击样本的攻击成功率分别达到99.9%和82%以上，中毒率低于5%。



## **25. Security of a Continuous-Variable based Quantum Position Verification Protocol**

一种基于连续变量的量子位置验证协议的安全性 quant-ph

17 pages, 2 figures

**SubmitDate**: 2023-08-08    [abs](http://arxiv.org/abs/2308.04166v1) [paper-pdf](http://arxiv.org/pdf/2308.04166v1)

**Authors**: Rene Allerstorfer, Llorenç Escolà-Farràs, Arpan Akash Ray, Boris Škorić, Florian Speelman, Philip Verduyn Lunel

**Abstract**: In this work we study quantum position verification with continuous-variable quantum states. In contrast to existing discrete protocols, we present and analyze a protocol that utilizes coherent states and its properties. Compared to discrete-variable photonic states, coherent states offer practical advantages since they can be efficiently prepared and manipulated with current technology. We prove security of the protocol against any unentangled attackers via entropic uncertainty relations, showing that the adversary has more uncertainty than the honest prover about the correct response as long as the noise in the quantum channel is below a certain threshold. Additionally, we show that attackers who pre-share one continuous-variable EPR pair can break the protocol.

摘要: 在这项工作中，我们研究了连续变量量子态的量子位置验证。与现有的离散协议不同，我们提出并分析了一种利用相干态的协议及其性质。与离散变量光子态相比，相干态具有实用的优势，因为它们可以用当前的技术有效地制备和操纵。我们通过熵不确定关系证明了协议对任何非纠缠攻击者的安全性，表明只要量子信道中的噪声低于一定的阈值，敌手就比诚实的证明者对正确的响应具有更大的不确定性。此外，我们还证明了预先共享一个连续变量EPR对的攻击者可以破坏协议。



## **26. UMD: Unsupervised Model Detection for X2X Backdoor Attacks**

UMD：X2X后门攻击的无监督模型检测 cs.LG

Proceedings of the 40th International Conference on Machine Learning

**SubmitDate**: 2023-08-08    [abs](http://arxiv.org/abs/2305.18651v3) [paper-pdf](http://arxiv.org/pdf/2305.18651v3)

**Authors**: Zhen Xiang, Zidi Xiong, Bo Li

**Abstract**: Backdoor (Trojan) attack is a common threat to deep neural networks, where samples from one or more source classes embedded with a backdoor trigger will be misclassified to adversarial target classes. Existing methods for detecting whether a classifier is backdoor attacked are mostly designed for attacks with a single adversarial target (e.g., all-to-one attack). To the best of our knowledge, without supervision, no existing methods can effectively address the more general X2X attack with an arbitrary number of source classes, each paired with an arbitrary target class. In this paper, we propose UMD, the first Unsupervised Model Detection method that effectively detects X2X backdoor attacks via a joint inference of the adversarial (source, target) class pairs. In particular, we first define a novel transferability statistic to measure and select a subset of putative backdoor class pairs based on a proposed clustering approach. Then, these selected class pairs are jointly assessed based on an aggregation of their reverse-engineered trigger size for detection inference, using a robust and unsupervised anomaly detector we proposed. We conduct comprehensive evaluations on CIFAR-10, GTSRB, and Imagenette dataset, and show that our unsupervised UMD outperforms SOTA detectors (even with supervision) by 17%, 4%, and 8%, respectively, in terms of the detection accuracy against diverse X2X attacks. We also show the strong detection performance of UMD against several strong adaptive attacks.

摘要: 后门(特洛伊木马)攻击是深度神经网络的常见威胁，来自嵌入后门触发器的一个或多个源类的样本将被错误分类为对抗性目标类。现有的检测分类器是否被后门攻击的方法大多是针对单个敌对目标的攻击而设计的(例如，All-to-One攻击)。就我们所知，在没有监督的情况下，没有任何现有方法可以有效地应对具有任意数量的源类的更通用的X2X攻击，每个源类都与任意的目标类配对。在本文中，我们提出了第一种无监督模型检测方法UMD，它通过联合推理对手(源、目标)类对来有效地检测X2X后门攻击。特别是，我们首先定义了一种新的可转移性统计量来度量和选择基于所提出的聚类方法的假定的后门类对的子集。然后，使用我们提出的健壮和无监督的异常检测器，基于它们的反向工程触发大小的聚集来联合评估这些选择的类对以用于检测推理。我们在CIFAR-10、GTSRB和Imagenette数据集上进行了综合评估，结果表明，在对各种X2X攻击的检测准确率方面，我们的无监督UMD分别比SOTA检测器(即使有监督)提高了17%、4%和8%。我们还展示了UMD对几种强自适应攻击的强检测性能。



## **27. Federated Zeroth-Order Optimization using Trajectory-Informed Surrogate Gradients**

基于轨迹信息代理梯度的联邦零阶优化 cs.LG

**SubmitDate**: 2023-08-08    [abs](http://arxiv.org/abs/2308.04077v1) [paper-pdf](http://arxiv.org/pdf/2308.04077v1)

**Authors**: Yao Shu, Xiaoqiang Lin, Zhongxiang Dai, Bryan Kian Hsiang Low

**Abstract**: Federated optimization, an emerging paradigm which finds wide real-world applications such as federated learning, enables multiple clients (e.g., edge devices) to collaboratively optimize a global function. The clients do not share their local datasets and typically only share their local gradients. However, the gradient information is not available in many applications of federated optimization, which hence gives rise to the paradigm of federated zeroth-order optimization (ZOO). Existing federated ZOO algorithms suffer from the limitations of query and communication inefficiency, which can be attributed to (a) their reliance on a substantial number of function queries for gradient estimation and (b) the significant disparity between their realized local updates and the intended global updates. To this end, we (a) introduce trajectory-informed gradient surrogates which is able to use the history of function queries during optimization for accurate and query-efficient gradient estimation, and (b) develop the technique of adaptive gradient correction using these gradient surrogates to mitigate the aforementioned disparity. Based on these, we propose the federated zeroth-order optimization using trajectory-informed surrogate gradients (FZooS) algorithm for query- and communication-efficient federated ZOO. Our FZooS achieves theoretical improvements over the existing approaches, which is supported by our real-world experiments such as federated black-box adversarial attack and federated non-differentiable metric optimization.

摘要: 联合优化是一种新兴的范例，它可以找到广泛的现实应用，如联合学习，使多个客户端(例如边缘设备)能够协作优化一个全局功能。客户端不共享其本地数据集，并且通常仅共享其本地渐变。然而，在联邦优化的许多应用中，梯度信息是不可用的，这就产生了联邦零阶优化(ZOO)的范例。现有的联合ZOO算法受到查询和通信效率低下的限制，这可以归因于(A)它们依赖于大量的函数查询来进行梯度估计，以及(B)它们实现的局部更新与预期的全局更新之间的显著差异。为此，我们(A)引入了轨迹信息梯度代理，它能够在优化过程中使用函数查询的历史来进行准确和高效的梯度估计，以及(B)开发使用这些梯度代理的自适应梯度校正技术来缓解上述差异。在此基础上，针对查询和通信高效的联邦动物园，提出了基于轨迹信息代理梯度的联邦零阶优化算法(FZooS)。我们的FZooS实现了对现有方法的理论改进，并得到了联邦黑盒对抗攻击和联邦不可微度量优化等真实世界实验的支持。



## **28. Adversarial Coreset Selection for Efficient Robust Training**

用于高效稳健训练的对抗性同位重置选择 cs.LG

Accepted to the International Journal of Computer Vision (IJCV).  Extended version of the ECCV2022 paper: arXiv:2112.00378. arXiv admin note:  substantial text overlap with arXiv:2112.00378

**SubmitDate**: 2023-08-08    [abs](http://arxiv.org/abs/2209.05785v2) [paper-pdf](http://arxiv.org/pdf/2209.05785v2)

**Authors**: Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstract**: Neural networks are vulnerable to adversarial attacks: adding well-crafted, imperceptible perturbations to their input can modify their output. Adversarial training is one of the most effective approaches to training robust models against such attacks. Unfortunately, this method is much slower than vanilla training of neural networks since it needs to construct adversarial examples for the entire training data at every iteration. By leveraging the theory of coreset selection, we show how selecting a small subset of training data provides a principled approach to reducing the time complexity of robust training. To this end, we first provide convergence guarantees for adversarial coreset selection. In particular, we show that the convergence bound is directly related to how well our coresets can approximate the gradient computed over the entire training data. Motivated by our theoretical analysis, we propose using this gradient approximation error as our adversarial coreset selection objective to reduce the training set size effectively. Once built, we run adversarial training over this subset of the training data. Unlike existing methods, our approach can be adapted to a wide variety of training objectives, including TRADES, $\ell_p$-PGD, and Perceptual Adversarial Training. We conduct extensive experiments to demonstrate that our approach speeds up adversarial training by 2-3 times while experiencing a slight degradation in the clean and robust accuracy.

摘要: 神经网络很容易受到敌意攻击：在它们的输入中添加精心设计的、不可察觉的扰动可以修改它们的输出。对抗性训练是训练抵抗此类攻击的稳健模型的最有效方法之一。遗憾的是，这种方法比普通的神经网络训练要慢得多，因为它需要在每次迭代中为整个训练数据构造对抗性样本。通过利用核心选择理论，我们展示了如何选择一小部分训练数据提供了一种原则性的方法来降低稳健训练的时间复杂性。为此，我们首先为对抗性核心重置选择提供收敛保证。特别是，我们证明了收敛界与我们的核集在整个训练数据上计算的梯度的逼近程度直接相关。在理论分析的基础上，我们提出了利用这一梯度逼近误差作为对抗性核心集选择的目标，以有效地减少训练集的规模。一旦构建，我们就对训练数据的这个子集进行对抗性训练。与现有方法不同，我们的方法可以适应广泛的训练目标，包括交易、$\ell_p$-PGD和感知对手训练。我们进行了大量的实验，证明了我们的方法将对手训练速度提高了2-3倍，同时经历了干净和健壮的准确性的轻微下降。



## **29. Improving Performance of Semi-Supervised Learning by Adversarial Attacks**

利用对抗性攻击提高半监督学习的性能 cs.LG

4 pages

**SubmitDate**: 2023-08-08    [abs](http://arxiv.org/abs/2308.04018v1) [paper-pdf](http://arxiv.org/pdf/2308.04018v1)

**Authors**: Dongyoon Yang, Kunwoong Kim, Yongdai Kim

**Abstract**: Semi-supervised learning (SSL) algorithm is a setup built upon a realistic assumption that access to a large amount of labeled data is tough. In this study, we present a generalized framework, named SCAR, standing for Selecting Clean samples with Adversarial Robustness, for improving the performance of recent SSL algorithms. By adversarially attacking pre-trained models with semi-supervision, our framework shows substantial advances in classifying images. We introduce how adversarial attacks successfully select high-confident unlabeled data to be labeled with current predictions. On CIFAR10, three recent SSL algorithms with SCAR result in significantly improved image classification.

摘要: 半监督学习(半监督学习)算法是建立在一个现实的假设之上的，即访问大量的标记数据是困难的。在这项研究中，我们提出了一个通用的框架，称为SCAR，代表选择具有对抗健壮性的清洁样本，以改进现有的SSL算法的性能。通过对半监督的预先训练模型进行恶意攻击，我们的框架在图像分类方面取得了实质性的进步。我们介绍了对抗性攻击如何成功地选择高置信度的未标记数据来标记当前预测。在CIFAR10上，最近的三种带有SCAR的SSL算法显著改善了图像分类。



## **30. PAIF: Perception-Aware Infrared-Visible Image Fusion for Attack-Tolerant Semantic Segmentation**

PAIF：感知红外-可见光图像融合的攻击容忍语义分割 cs.CV

Accepted by ACM MM'2023;The source codes are available at  https://github.com/LiuZhu-CV/PAIF

**SubmitDate**: 2023-08-08    [abs](http://arxiv.org/abs/2308.03979v1) [paper-pdf](http://arxiv.org/pdf/2308.03979v1)

**Authors**: Zhu Liu, Jinyuan Liu, Benzhuang Zhang, Long Ma, Xin Fan, Risheng Liu

**Abstract**: Infrared and visible image fusion is a powerful technique that combines complementary information from different modalities for downstream semantic perception tasks. Existing learning-based methods show remarkable performance, but are suffering from the inherent vulnerability of adversarial attacks, causing a significant decrease in accuracy. In this work, a perception-aware fusion framework is proposed to promote segmentation robustness in adversarial scenes. We first conduct systematic analyses about the components of image fusion, investigating the correlation with segmentation robustness under adversarial perturbations. Based on these analyses, we propose a harmonized architecture search with a decomposition-based structure to balance standard accuracy and robustness. We also propose an adaptive learning strategy to improve the parameter robustness of image fusion, which can learn effective feature extraction under diverse adversarial perturbations. Thus, the goals of image fusion (\textit{i.e.,} extracting complementary features from source modalities and defending attack) can be realized from the perspectives of architectural and learning strategies. Extensive experimental results demonstrate that our scheme substantially enhances the robustness, with gains of 15.3% mIOU of segmentation in the adversarial scene, compared with advanced competitors. The source codes are available at https://github.com/LiuZhu-CV/PAIF.

摘要: 红外和可见光图像融合是一种强大的技术，它结合了来自不同模式的互补信息，用于下游的语义感知任务。现有的基于学习的方法表现出显著的性能，但存在对抗性攻击的固有脆弱性，导致准确率显著下降。在这项工作中，提出了一种感知融合框架，以提高对抗性场景下的分割健壮性。我们首先对图像融合的各个组成部分进行了系统的分析，研究了在对抗性扰动下图像融合与分割稳健性的相关性。基于这些分析，我们提出了一种基于分解的协调结构搜索，以平衡标准的准确性和健壮性。我们还提出了一种自适应学习策略来提高图像融合的参数稳健性，该学习策略可以在不同的对抗性扰动下学习有效的特征提取。因此，图像融合的目标(即从源模式中提取互补特征和防御攻击)可以从体系结构和学习策略的角度来实现。大量的实验结果表明，我们的方案大大增强了算法的鲁棒性，在对抗性场景下，与先进的竞争对手相比，分割效率提高了15.3%。源代码可在https://github.com/LiuZhu-CV/PAIF.上找到



## **31. Fixed Inter-Neuron Covariability Induces Adversarial Robustness**

固定神经元间协变性诱导对抗健壮性 cs.LG

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2308.03956v1) [paper-pdf](http://arxiv.org/pdf/2308.03956v1)

**Authors**: Muhammad Ahmed Shah, Bhiksha Raj

**Abstract**: The vulnerability to adversarial perturbations is a major flaw of Deep Neural Networks (DNNs) that raises question about their reliability when in real-world scenarios. On the other hand, human perception, which DNNs are supposed to emulate, is highly robust to such perturbations, indicating that there may be certain features of the human perception that make it robust but are not represented in the current class of DNNs. One such feature is that the activity of biological neurons is correlated and the structure of this correlation tends to be rather rigid over long spans of times, even if it hampers performance and learning. We hypothesize that integrating such constraints on the activations of a DNN would improve its adversarial robustness, and, to test this hypothesis, we have developed the Self-Consistent Activation (SCA) layer, which comprises of neurons whose activations are consistent with each other, as they conform to a fixed, but learned, covariability pattern. When evaluated on image and sound recognition tasks, the models with a SCA layer achieved high accuracy, and exhibited significantly greater robustness than multi-layer perceptron models to state-of-the-art Auto-PGD adversarial attacks \textit{without being trained on adversarially perturbed data

摘要: 深度神经网络(DNN)对敌意扰动的脆弱性是其在现实世界场景中的一大缺陷，引发了人们对其可靠性的质疑。另一方面，DNN应该模仿的人类感知对这种扰动具有高度的健壮性，这表明人类感知的某些特征可能使其健壮，但在当前的DNN类别中没有表现出来。其中一个特征是，生物神经元的活动是相关的，这种相关性的结构在很长一段时间内往往是相当僵硬的，即使它阻碍了表现和学习。我们假设，将这些限制整合到DNN的激活上将提高其对抗健壮性，为了验证这一假设，我们开发了自洽激活(SCA)层，它由激活彼此一致的神经元组成，因为它们符合固定的、但学习的协变性模式。当在图像和声音识别任务上进行评估时，具有SCA层的模型获得了高精度，并且表现出比多层感知器模型对最先进的Auto-PGD对手攻击的显著更强的稳健性



## **32. "Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models**

《Do Anything Now》：在大型语言模型上描述和评估野外越狱提示 cs.CR

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2308.03825v1) [paper-pdf](http://arxiv.org/pdf/2308.03825v1)

**Authors**: Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, Yang Zhang

**Abstract**: The misuse of large language models (LLMs) has garnered significant attention from the general public and LLM vendors. In response, efforts have been made to align LLMs with human values and intent use. However, a particular type of adversarial prompts, known as jailbreak prompt, has emerged and continuously evolved to bypass the safeguards and elicit harmful content from LLMs. In this paper, we conduct the first measurement study on jailbreak prompts in the wild, with 6,387 prompts collected from four platforms over six months. Leveraging natural language processing technologies and graph-based community detection methods, we discover unique characteristics of jailbreak prompts and their major attack strategies, such as prompt injection and privilege escalation. We also observe that jailbreak prompts increasingly shift from public platforms to private ones, posing new challenges for LLM vendors in proactive detection. To assess the potential harm caused by jailbreak prompts, we create a question set comprising 46,800 samples across 13 forbidden scenarios. Our experiments show that current LLMs and safeguards cannot adequately defend jailbreak prompts in all scenarios. Particularly, we identify two highly effective jailbreak prompts which achieve 0.99 attack success rates on ChatGPT (GPT-3.5) and GPT-4, and they have persisted online for over 100 days. Our work sheds light on the severe and evolving threat landscape of jailbreak prompts. We hope our study can facilitate the research community and LLM vendors in promoting safer and regulated LLMs.

摘要: 大型语言模型(LLM)的误用引起了公众和LLM供应商的极大关注。为此，已作出努力使低土地利用方式与人的价值和意图用途相一致。然而，一种称为越狱提示的特定类型的对抗性提示已经出现，并不断演变，以绕过安全措施，从LLMS引出有害内容。本文首次在野外对越狱提示进行了测量研究，在六个月的时间里，从四个平台收集了6387条提示。利用自然语言处理技术和基于图的社区检测方法，我们发现了越狱提示的独特特征及其主要攻击策略，如提示注入和权限提升。我们还观察到，越狱提示越来越多地从公共平台转向私人平台，这给LLM供应商在主动检测方面提出了新的挑战。为了评估越狱提示造成的潜在危害，我们创建了一个包含13个禁止场景的46,800个样本的问题集。我们的实验表明，现有的LLM和安全措施不能在所有场景下充分防御越狱提示。特别是，我们识别了两个高效的越狱提示，它们在ChatGPT(GPT-3.5)和GPT-4上的攻击成功率达到了0.99，并且在线持续了100多天。我们的工作揭示了越狱提示的严重和不断变化的威胁图景。我们希望我们的研究能够促进研究界和LLM供应商推广更安全和规范的LLM。



## **33. Assessing Adversarial Replay and Deep Learning-Driven Attacks on Specific Emitter Identification-based Security Approaches**

评估基于特定发射器识别的安全方法上的对抗性重放和深度学习驱动攻击 eess.SP

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2308.03579v1) [paper-pdf](http://arxiv.org/pdf/2308.03579v1)

**Authors**: Joshua H. Tyler, Mohamed K. M. Fadul, Matthew R. Hilling, Donald R. Reising, T. Daniel Loveless

**Abstract**: Specific Emitter Identification (SEI) detects, characterizes, and identifies emitters by exploiting distinct, inherent, and unintentional features in their transmitted signals. Since its introduction, a significant amount of work has been conducted; however, most assume the emitters are passive and that their identifying signal features are immutable and challenging to mimic. Suggesting the emitters are reluctant and incapable of developing and implementing effective SEI countermeasures; however, Deep Learning (DL) has been shown capable of learning emitter-specific features directly from their raw in-phase and quadrature signal samples, and Software-Defined Radios (SDRs) can manipulate them. Based on these capabilities, it is fair to question the ease at which an emitter can effectively mimic the SEI features of another or manipulate its own to hinder or defeat SEI. This work considers SEI mimicry using three signal features mimicking countermeasures; off-the-self DL; two SDRs of different sizes, weights, power, and cost (SWaP-C); handcrafted and DL-based SEI processes, and a coffee shop deployment. Our results show off-the-shelf DL algorithms, and SDR enables SEI mimicry; however, adversary success is hindered by: the use of decoy emitter preambles, the use of a denoising autoencoder, and SDR SWaP-C constraints.

摘要: 特定发射体识别(SEI)通过利用发射体发射信号中不同的、固有的和无意的特征来检测、表征和识别辐射体。自从它被引入以来，已经进行了大量的工作；然而，大多数人假设发射器是被动的，并且它们的识别信号特征是不变的，并且具有模拟的挑战性。这表明发射器不愿意也没有能力开发和实施有效的SEI对策；然而，深度学习(DL)已被证明能够直接从其原始的同相和正交信号样本中学习发射器特定的特征，而软件定义无线电(SDR)可以操纵它们。基于这些能力，可以很容易地质疑发射器是否能够有效地模仿另一个发射器的SEI特征，或者操纵自己的SEI来阻碍或击败SEI。这项工作考虑了使用三个模仿对策的信号功能的SEI模拟；离线自定义DL；两个不同大小、重量、功率和成本的SDR(SWAP-C)；手工制作的基于DL的SEI流程，以及咖啡馆部署。我们的结果显示了现成的DL算法，SDR使SEI模仿成为可能；然而，对手的成功受到以下因素的阻碍：使用诱饵发射器前导，使用去噪自动编码器，以及SDR SWAP-C约束。



## **34. Mondrian: Prompt Abstraction Attack Against Large Language Models for Cheaper API Pricing**

Mondrian：针对大型语言模型的即时抽象攻击，以获得更低的API定价 cs.CR

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2308.03558v1) [paper-pdf](http://arxiv.org/pdf/2308.03558v1)

**Authors**: Wai Man Si, Michael Backes, Yang Zhang

**Abstract**: The Machine Learning as a Service (MLaaS) market is rapidly expanding and becoming more mature. For example, OpenAI's ChatGPT is an advanced large language model (LLM) that generates responses for various queries with associated fees. Although these models can deliver satisfactory performance, they are far from perfect. Researchers have long studied the vulnerabilities and limitations of LLMs, such as adversarial attacks and model toxicity. Inevitably, commercial ML models are also not exempt from such issues, which can be problematic as MLaaS continues to grow. In this paper, we discover a new attack strategy against LLM APIs, namely the prompt abstraction attack. Specifically, we propose Mondrian, a simple and straightforward method that abstracts sentences, which can lower the cost of using LLM APIs. In this approach, the adversary first creates a pseudo API (with a lower established price) to serve as the proxy of the target API (with a higher established price). Next, the pseudo API leverages Mondrian to modify the user query, obtain the abstracted response from the target API, and forward it back to the end user. Our results show that Mondrian successfully reduces user queries' token length ranging from 13% to 23% across various tasks, including text classification, generation, and question answering. Meanwhile, these abstracted queries do not significantly affect the utility of task-specific and general language models like ChatGPT. Mondrian also reduces instruction prompts' token length by at least 11% without compromising output quality. As a result, the prompt abstraction attack enables the adversary to profit without bearing the cost of API development and deployment.

摘要: 机器学习即服务(MLaaS)市场正在迅速扩大并日趋成熟。例如，OpenAI的ChatGPT是一个高级的大型语言模型(LLM)，它可以为各种查询生成响应，并收取相关费用。尽管这些车型可以提供令人满意的性能，但它们远不是完美的。长期以来，研究人员一直在研究LLMS的脆弱性和局限性，如对抗性攻击和模型毒性。不可避免的是，商业ML模型也不能幸免于此类问题，随着MLaaS的持续增长，这些问题可能会成为问题。在本文中，我们发现了一种针对LLMAPI的新攻击策略，即即时抽象攻击。具体地说，我们提出了Mondrian，这是一种简单明了的抽象句子的方法，可以降低使用LLMAPI的成本。在这种方法中，对手首先创建一个伪API(具有较低的既定价格)来充当目标API的代理(具有较高的既定价格)。接下来，伪API利用Mondrian修改用户查询，从目标API获取抽象的响应，并将其转发回最终用户。我们的结果表明，Mondrian成功地将用户查询的令牌长度在包括文本分类、生成和问答在内的各种任务中减少了13%到23%。同时，这些抽象的查询不会显著影响特定任务和通用语言模型(如ChatGPT)的效用。Mondrian还在不影响输出质量的情况下将指令提示符的标记长度减少了至少11%。因此，及时的抽象攻击使对手能够在不承担API开发和部署成本的情况下获利。



## **35. Exploring the Physical World Adversarial Robustness of Vehicle Detection**

探索车辆检测的物理世界对抗稳健性 cs.CV

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2308.03476v1) [paper-pdf](http://arxiv.org/pdf/2308.03476v1)

**Authors**: Wei Jiang, Tianyuan Zhang, Shuangcheng Liu, Weiyu Ji, Zichao Zhang, Gang Xiao

**Abstract**: Adversarial attacks can compromise the robustness of real-world detection models. However, evaluating these models under real-world conditions poses challenges due to resource-intensive experiments. Virtual simulations offer an alternative, but the absence of standardized benchmarks hampers progress. Addressing this, we propose an innovative instant-level data generation pipeline using the CARLA simulator. Through this pipeline, we establish the Discrete and Continuous Instant-level (DCI) dataset, enabling comprehensive experiments involving three detection models and three physical adversarial attacks. Our findings highlight diverse model performances under adversarial conditions. Yolo v6 demonstrates remarkable resilience, experiencing just a marginal 6.59% average drop in average precision (AP). In contrast, the ASA attack yields a substantial 14.51% average AP reduction, twice the effect of other algorithms. We also note that static scenes yield higher recognition AP values, and outcomes remain relatively consistent across varying weather conditions. Intriguingly, our study suggests that advancements in adversarial attack algorithms may be approaching its ``limitation''.In summary, our work underscores the significance of adversarial attacks in real-world contexts and introduces the DCI dataset as a versatile benchmark. Our findings provide valuable insights for enhancing the robustness of detection models and offer guidance for future research endeavors in the realm of adversarial attacks.

摘要: 敌意攻击可能会损害真实世界检测模型的健壮性。然而，由于资源密集型实验，在现实世界条件下评估这些模型会带来挑战。虚拟模拟提供了另一种选择，但缺乏标准化的基准阻碍了进展。为了解决这一问题，我们提出了一种使用CALA模拟器的创新的即时级数据生成管道。通过这条管道，我们建立了离散和连续的即时级别(DCI)数据集，使涉及三个检测模型和三个物理对抗性攻击的全面实验成为可能。我们的发现突出了模型在对抗性条件下的不同表现。YOLO V6表现出非凡的弹性，平均精度(AP)仅略有6.59%的下降。相比之下，ASA攻击的平均AP减少了14.51%，是其他算法的两倍。我们还注意到，静态场景产生更高的识别AP值，并且结果在不同的天气条件下保持相对一致。有趣的是，我们的研究表明，对抗性攻击算法的进步可能正在接近其“局限性”。总而言之，我们的工作强调了对抗性攻击在现实世界中的重要性，并引入了DCI数据集作为通用基准。我们的发现为增强检测模型的稳健性提供了有价值的见解，并为未来在对抗性攻击领域的研究工作提供了指导。



## **36. Imperceptible Physical Attack against Face Recognition Systems via LED Illumination Modulation**

利用LED照明调制对人脸识别系统的隐形物理攻击 cs.CV

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2307.13294v2) [paper-pdf](http://arxiv.org/pdf/2307.13294v2)

**Authors**: Junbin Fang, Canjian Jiang, You Jiang, Puxi Lin, Zhaojie Chen, Yujing Sun, Siu-Ming Yiu, Zoe L. Jiang

**Abstract**: Although face recognition starts to play an important role in our daily life, we need to pay attention that data-driven face recognition vision systems are vulnerable to adversarial attacks. However, the current two categories of adversarial attacks, namely digital attacks and physical attacks both have drawbacks, with the former ones impractical and the latter one conspicuous, high-computational and inexecutable. To address the issues, we propose a practical, executable, inconspicuous and low computational adversarial attack based on LED illumination modulation. To fool the systems, the proposed attack generates imperceptible luminance changes to human eyes through fast intensity modulation of scene LED illumination and uses the rolling shutter effect of CMOS image sensors in face recognition systems to implant luminance information perturbation to the captured face images. In summary,we present a denial-of-service (DoS) attack for face detection and a dodging attack for face verification. We also evaluate their effectiveness against well-known face detection models, Dlib, MTCNN and RetinaFace , and face verification models, Dlib, FaceNet,and ArcFace.The extensive experiments show that the success rates of DoS attacks against face detection models reach 97.67%, 100%, and 100%, respectively, and the success rates of dodging attacks against all face verification models reach 100%.

摘要: 虽然人脸识别开始在我们的日常生活中发挥重要作用，但我们需要注意的是，数据驱动的人脸识别视觉系统很容易受到对手的攻击。然而，目前的两类对抗性攻击，即数字攻击和物理攻击都有缺点，前者不切实际，后者突出，计算量大，无法执行。针对这些问题，我们提出了一种实用的、可执行的、隐蔽的、低计算量的基于LED照明调制的对抗性攻击。为了欺骗系统，该攻击通过场景LED照明的快速强度调制来产生人眼不可感知的亮度变化，并利用人脸识别系统中CMOS图像传感器的滚动快门效应在捕获的人脸图像中植入亮度信息扰动。综上所述，我们提出了一种用于人脸检测的拒绝服务攻击和一种用于人脸验证的躲避攻击。实验表明，该算法对人脸检测模型的DoS攻击成功率分别达到97.67%、100%和100%，对所有人脸验证模型的躲避攻击成功率均达到100%。



## **37. A reading survey on adversarial machine learning: Adversarial attacks and their understanding**

对抗性机器学习的阅读调查：对抗性攻击及其理解 cs.LG

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2308.03363v1) [paper-pdf](http://arxiv.org/pdf/2308.03363v1)

**Authors**: Shashank Kotyan

**Abstract**: Deep Learning has empowered us to train neural networks for complex data with high performance. However, with the growing research, several vulnerabilities in neural networks have been exposed. A particular branch of research, Adversarial Machine Learning, exploits and understands some of the vulnerabilities that cause the neural networks to misclassify for near original input. A class of algorithms called adversarial attacks is proposed to make the neural networks misclassify for various tasks in different domains. With the extensive and growing research in adversarial attacks, it is crucial to understand the classification of adversarial attacks. This will help us understand the vulnerabilities in a systematic order and help us to mitigate the effects of adversarial attacks. This article provides a survey of existing adversarial attacks and their understanding based on different perspectives. We also provide a brief overview of existing adversarial defences and their limitations in mitigating the effect of adversarial attacks. Further, we conclude with a discussion on the future research directions in the field of adversarial machine learning.

摘要: 深度学习使我们能够对复杂数据进行高性能的神经网络训练。然而，随着研究的不断深入，神经网络中的一些漏洞也被暴露出来。对抗性机器学习是研究的一个特定分支，它利用和理解一些漏洞，这些漏洞导致神经网络对接近原始的输入进行错误分类。为了使神经网络对不同领域的不同任务进行误分类，提出了一种称为对抗攻击的算法。随着对抗性攻击研究的不断深入，理解对抗性攻击的分类变得至关重要。这将有助于我们系统地了解漏洞，并帮助我们减轻对抗性攻击的影响。本文从不同的角度对现有的对抗性攻击及其理解进行了综述。我们还简要概述了现有的对抗性防御及其在减轻对抗性攻击影响方面的局限性。最后，我们对对抗性机器学习领域未来的研究方向进行了讨论。



## **38. Membership Inference Attacks against Language Models via Neighbourhood Comparison**

基于邻域比较的语言模型隶属度推理攻击 cs.CL

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2305.18462v2) [paper-pdf](http://arxiv.org/pdf/2305.18462v2)

**Authors**: Justus Mattern, Fatemehsadat Mireshghallah, Zhijing Jin, Bernhard Schölkopf, Mrinmaya Sachan, Taylor Berg-Kirkpatrick

**Abstract**: Membership Inference attacks (MIAs) aim to predict whether a data sample was present in the training data of a machine learning model or not, and are widely used for assessing the privacy risks of language models. Most existing attacks rely on the observation that models tend to assign higher probabilities to their training samples than non-training points. However, simple thresholding of the model score in isolation tends to lead to high false-positive rates as it does not account for the intrinsic complexity of a sample. Recent work has demonstrated that reference-based attacks which compare model scores to those obtained from a reference model trained on similar data can substantially improve the performance of MIAs. However, in order to train reference models, attacks of this kind make the strong and arguably unrealistic assumption that an adversary has access to samples closely resembling the original training data. Therefore, we investigate their performance in more realistic scenarios and find that they are highly fragile in relation to the data distribution used to train reference models. To investigate whether this fragility provides a layer of safety, we propose and evaluate neighbourhood attacks, which compare model scores for a given sample to scores of synthetically generated neighbour texts and therefore eliminate the need for access to the training data distribution. We show that, in addition to being competitive with reference-based attacks that have perfect knowledge about the training data distribution, our attack clearly outperforms existing reference-free attacks as well as reference-based attacks with imperfect knowledge, which demonstrates the need for a reevaluation of the threat model of adversarial attacks.

摘要: 成员关系推理攻击(MIA)旨在预测数据样本是否存在于机器学习模型的训练数据中，被广泛用于评估语言模型的隐私风险。大多数现有的攻击都依赖于这样的观察，即模型倾向于为其训练样本分配比非训练点更高的概率。然而，孤立地对模型分数进行简单的阈值处理往往会导致高的假阳性率，因为它没有考虑到样本的内在复杂性。最近的工作表明，基于参考的攻击将模型得分与根据相似数据训练的参考模型获得的得分进行比较，可以显著提高MIA的性能。然而，为了训练参考模型，这类攻击做出了强有力的、可以说是不切实际的假设，即对手可以获得与原始训练数据非常相似的样本。因此，我们在更现实的场景中调查了它们的性能，发现它们相对于用于训练参考模型的数据分布来说是非常脆弱的。为了调查这种脆弱性是否提供了一层安全，我们提出并评估了邻居攻击，该攻击将给定样本的模型分数与数十个合成生成的邻居文本进行比较，从而消除了访问训练数据分布的需要。我们表明，除了与对训练数据分布有完善了解的基于引用的攻击相比，该攻击的性能明显优于现有的无引用攻击和具有不完全知识的基于引用的攻击，这表明需要对对抗性攻击的威胁模型进行重新评估。



## **39. Token-Modification Adversarial Attacks for Natural Language Processing: A Survey**

自然语言处理中的标记修改攻击：综述 cs.CL

Version 2: updated

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2103.00676v2) [paper-pdf](http://arxiv.org/pdf/2103.00676v2)

**Authors**: Tom Roth, Yansong Gao, Alsharif Abuadbba, Surya Nepal, Wei Liu

**Abstract**: There are now many adversarial attacks for natural language processing systems. Of these, a vast majority achieve success by modifying individual document tokens, which we call here a token-modification attack. Each token-modification attack is defined by a specific combination of fundamental components, such as a constraint on the adversary or a particular search algorithm. Motivated by this observation, we survey existing token-modification attacks and extract the components of each. We use an attack-independent framework to structure our survey which results in an effective categorisation of the field and an easy comparison of components. This survey aims to guide new researchers to this field and spark further research into individual attack components.

摘要: 现在对自然语言处理系统的敌意攻击很多。在这些攻击中，绝大多数是通过修改单个文档令牌来获得成功的，我们在这里称之为令牌修改攻击。每个令牌修改攻击都是由基本组件的特定组合定义的，例如对对手的约束或特定的搜索算法。在这一观察的基础上，我们调查了现有的令牌修改攻击，并提取了每个攻击的组件。我们使用独立于攻击的框架来组织我们的调查，这导致了对字段的有效分类和对组件的轻松比较。这项调查旨在引导新的研究人员进入这一领域，并引发对个别攻击组件的进一步研究。



## **40. APBench: A Unified Benchmark for Availability Poisoning Attacks and Defenses**

APBtch：可用性、毒化攻击和防御的统一基准 cs.CV

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2308.03258v1) [paper-pdf](http://arxiv.org/pdf/2308.03258v1)

**Authors**: Tianrui Qin, Xitong Gao, Juanjuan Zhao, Kejiang Ye, Cheng-Zhong Xu

**Abstract**: The efficacy of availability poisoning, a method of poisoning data by injecting imperceptible perturbations to prevent its use in model training, has been a hot subject of investigation. Previous research suggested that it was difficult to effectively counteract such poisoning attacks. However, the introduction of various defense methods has challenged this notion. Due to the rapid progress in this field, the performance of different novel methods cannot be accurately validated due to variations in experimental setups. To further evaluate the attack and defense capabilities of these poisoning methods, we have developed a benchmark -- APBench for assessing the efficacy of adversarial poisoning. APBench consists of 9 state-of-the-art availability poisoning attacks, 8 defense algorithms, and 4 conventional data augmentation techniques. We also have set up experiments with varying different poisoning ratios, and evaluated the attacks on multiple datasets and their transferability across model architectures. We further conducted a comprehensive evaluation of 2 additional attacks specifically targeting unsupervised models. Our results reveal the glaring inadequacy of existing attacks in safeguarding individual privacy. APBench is open source and available to the deep learning community: https://github.com/lafeat/apbench.

摘要: 可用性中毒是一种通过注入难以察觉的扰动来毒化数据以防止其在模型训练中使用的方法，其有效性一直是研究的热点。此前的研究表明，很难有效地对抗这种中毒攻击。然而，各种防御方法的引入挑战了这一概念。由于这一领域的快速发展，由于实验设置的不同，不同的新方法的性能无法得到准确的验证。为了进一步评估这些中毒方法的攻防能力，我们开发了一个评估对抗性中毒效果的基准--APBENCH。APBtch包括9个最先进的可用性中毒攻击、8个防御算法和4个常规数据增强技术。我们还建立了不同投毒率的实验，并评估了对多个数据集的攻击及其跨模型体系结构的可传输性。我们进一步对另外两个专门针对非监督模型的攻击进行了全面评估。我们的结果揭示了现有攻击在保护个人隐私方面的明显不足。Https://github.com/lafeat/apbench.是开放源码的，可供深度学习社区使用



## **41. Unsupervised Adversarial Detection without Extra Model: Training Loss Should Change**

无额外模型的无监督对手检测：训练损失应该改变 cs.LG

AdvML in ICML 2023  code:https://github.com/CycleBooster/Unsupervised-adversarial-detection-without-extra-model

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2308.03243v1) [paper-pdf](http://arxiv.org/pdf/2308.03243v1)

**Authors**: Chien Cheng Chyou, Hung-Ting Su, Winston H. Hsu

**Abstract**: Adversarial robustness poses a critical challenge in the deployment of deep learning models for real-world applications. Traditional approaches to adversarial training and supervised detection rely on prior knowledge of attack types and access to labeled training data, which is often impractical. Existing unsupervised adversarial detection methods identify whether the target model works properly, but they suffer from bad accuracies owing to the use of common cross-entropy training loss, which relies on unnecessary features and strengthens adversarial attacks. We propose new training losses to reduce useless features and the corresponding detection method without prior knowledge of adversarial attacks. The detection rate (true positive rate) against all given white-box attacks is above 93.9% except for attacks without limits (DF($\infty$)), while the false positive rate is barely 2.5%. The proposed method works well in all tested attack types and the false positive rates are even better than the methods good at certain types.

摘要: 对抗性稳健性是为现实世界应用部署深度学习模型的一个关键挑战。传统的对抗性训练和监督检测方法依赖于攻击类型的先验知识和对标记训练数据的访问，这往往是不切实际的。现有的无监督敌意检测方法识别目标模型是否正常工作，但由于使用了共同的交叉熵训练损失，依赖于不必要的特征，从而加强了对抗性攻击，因此准确率较低。我们提出了一种新的训练损失来减少无用特征，并提出了相应的检测方法。对所有给定白盒攻击的检测率(真阳性率)除无限制攻击(DF($\infty$))外均在93.9%以上，而假阳性率仅为2.5%。该方法在测试的所有攻击类型中都表现良好，其误检率甚至优于某些类型的方法。



## **42. CGBA: Curvature-aware Geometric Black-box Attack**

CGBA：曲率感知几何黑盒攻击 cs.CV

This paper is accepted to publish in ICCV

**SubmitDate**: 2023-08-06    [abs](http://arxiv.org/abs/2308.03163v1) [paper-pdf](http://arxiv.org/pdf/2308.03163v1)

**Authors**: Md Farhamdur Reza, Ali Rahmati, Tianfu Wu, Huaiyu Dai

**Abstract**: Decision-based black-box attacks often necessitate a large number of queries to craft an adversarial example. Moreover, decision-based attacks based on querying boundary points in the estimated normal vector direction often suffer from inefficiency and convergence issues. In this paper, we propose a novel query-efficient curvature-aware geometric decision-based black-box attack (CGBA) that conducts boundary search along a semicircular path on a restricted 2D plane to ensure finding a boundary point successfully irrespective of the boundary curvature. While the proposed CGBA attack can work effectively for an arbitrary decision boundary, it is particularly efficient in exploiting the low curvature to craft high-quality adversarial examples, which is widely seen and experimentally verified in commonly used classifiers under non-targeted attacks. In contrast, the decision boundaries often exhibit higher curvature under targeted attacks. Thus, we develop a new query-efficient variant, CGBA-H, that is adapted for the targeted attack. In addition, we further design an algorithm to obtain a better initial boundary point at the expense of some extra queries, which considerably enhances the performance of the targeted attack. Extensive experiments are conducted to evaluate the performance of our proposed methods against some well-known classifiers on the ImageNet and CIFAR10 datasets, demonstrating the superiority of CGBA and CGBA-H over state-of-the-art non-targeted and targeted attacks, respectively. The source code is available at https://github.com/Farhamdur/CGBA.

摘要: 基于决策的黑盒攻击通常需要进行大量查询才能创建一个对抗性的例子。此外，在估计的法向向量方向上查询边界点的基于决策的攻击往往存在效率低下和收敛问题。本文提出了一种新的查询效率高的基于曲率感知几何决策的黑盒攻击(CGBA)，该攻击在受限制的2D平面上沿半圆路径进行边界搜索，以确保无论边界曲率如何都能成功地找到边界点。虽然所提出的CGBA攻击可以在任意决策边界下有效地工作，但它在利用低曲率来构造高质量的对抗性样本方面特别有效，这在非目标攻击下的常用分类器中得到了广泛的看到和实验验证。相比之下，在有针对性的攻击下，决策边界往往表现出更高的曲率。因此，我们开发了一种新的查询高效的变体CGBA-H，该变体适合于定向攻击。此外，我们进一步设计了一种算法，以牺牲一些额外的查询来获得更好的初始边界点，从而大大提高了定向攻击的性能。在ImageNet和CIFAR10数据集上对我们提出的方法进行了大量的测试，证明了CGBA和CGBA-H分别比最新的非目标攻击和目标攻击的性能。源代码可在https://github.com/Farhamdur/CGBA.上找到



## **43. MM-BD: Post-Training Detection of Backdoor Attacks with Arbitrary Backdoor Pattern Types Using a Maximum Margin Statistic**

MM-BD：基于最大余量统计的任意类型后门攻击的训练后检测 cs.LG

**SubmitDate**: 2023-08-06    [abs](http://arxiv.org/abs/2205.06900v2) [paper-pdf](http://arxiv.org/pdf/2205.06900v2)

**Authors**: Hang Wang, Zhen Xiang, David J. Miller, George Kesidis

**Abstract**: Backdoor attacks are an important type of adversarial threat against deep neural network classifiers, wherein test samples from one or more source classes will be (mis)classified to the attacker's target class when a backdoor pattern is embedded. In this paper, we focus on the post-training backdoor defense scenario commonly considered in the literature, where the defender aims to detect whether a trained classifier was backdoor-attacked without any access to the training set. Many post-training detectors are designed to detect attacks that use either one or a few specific backdoor embedding functions (e.g., patch-replacement or additive attacks). These detectors may fail when the backdoor embedding function used by the attacker (unknown to the defender) is different from the backdoor embedding function assumed by the defender. In contrast, we propose a post-training defense that detects backdoor attacks with arbitrary types of backdoor embeddings, without making any assumptions about the backdoor embedding type. Our detector leverages the influence of the backdoor attack, independent of the backdoor embedding mechanism, on the landscape of the classifier's outputs prior to the softmax layer. For each class, a maximum margin statistic is estimated. Detection inference is then performed by applying an unsupervised anomaly detector to these statistics. Thus, our detector does not need any legitimate clean samples, and can efficiently detect backdoor attacks with arbitrary numbers of source classes. These advantages over several state-of-the-art methods are demonstrated on four datasets, for three different types of backdoor patterns, and for a variety of attack configurations. Finally, we propose a novel, general approach for backdoor mitigation once a detection is made. The mitigation approach was the runner-up at the first IEEE Trojan Removal Competition. The code is online available.

摘要: 后门攻击是对深度神经网络分类器的一种重要的敌意威胁，当嵌入后门模式时，来自一个或多个源类的测试样本将被(错误地)分类到攻击者的目标类。在本文中，我们关注文献中通常考虑的训练后后门防御场景，其中防御者的目标是在不访问训练集的情况下检测训练的分类器是否受到后门攻击。许多训练后检测器被设计为检测使用一个或几个特定后门嵌入函数的攻击(例如，补丁替换或添加攻击)。当攻击者使用的后门嵌入函数(防御者未知)不同于防御者采用的后门嵌入函数时，这些检测器可能会失败。相反，我们提出了一种训练后防御，该防御检测具有任意类型的后门嵌入的后门攻击，而不对后门嵌入类型做出任何假设。我们的检测器利用后门攻击的影响，独立于后门嵌入机制，在Softmax层之前对分类器输出的景观进行影响。对于每一类，估计最大边际统计。然后，通过对这些统计量应用非监督异常检测器来执行检测推理。因此，我们的检测器不需要任何合法的干净样本，并且可以有效地检测具有任意数量的源类的后门攻击。在四个数据集上，对于三种不同类型的后门模式和各种攻击配置，这些相对于几种最先进方法的优势得到了演示。最后，我们提出了一种新颖的、通用的方法，用于在检测到后进行后门缓解。缓解方法在第一届IEEE特洛伊木马删除大赛中获得亚军。该代码可在网上获得。



## **44. Phase-shifted Adversarial Training**

相移对抗性训练 cs.LG

Proceedings of Uncertainty in Artificial Intelligence, 2023 (UAI  2023)

**SubmitDate**: 2023-08-06    [abs](http://arxiv.org/abs/2301.04785v2) [paper-pdf](http://arxiv.org/pdf/2301.04785v2)

**Authors**: Yeachan Kim, Seongyeon Kim, Ihyeok Seo, Bonggun Shin

**Abstract**: Adversarial training has been considered an imperative component for safely deploying neural network-based applications to the real world. To achieve stronger robustness, existing methods primarily focus on how to generate strong attacks by increasing the number of update steps, regularizing the models with the smoothed loss function, and injecting the randomness into the attack. Instead, we analyze the behavior of adversarial training through the lens of response frequency. We empirically discover that adversarial training causes neural networks to have low convergence to high-frequency information, resulting in highly oscillated predictions near each data. To learn high-frequency contents efficiently and effectively, we first prove that a universal phenomenon of frequency principle, i.e., \textit{lower frequencies are learned first}, still holds in adversarial training. Based on that, we propose phase-shifted adversarial training (PhaseAT) in which the model learns high-frequency components by shifting these frequencies to the low-frequency range where the fast convergence occurs. For evaluations, we conduct the experiments on CIFAR-10 and ImageNet with the adaptive attack carefully designed for reliable evaluation. Comprehensive results show that PhaseAT significantly improves the convergence for high-frequency information. This results in improved adversarial robustness by enabling the model to have smoothed predictions near each data.

摘要: 对抗性训练一直被认为是将基于神经网络的应用安全地部署到现实世界中的一个必不可少的组成部分。为了获得更强的稳健性，现有的方法主要集中在如何通过增加更新步骤、用平滑的损失函数对模型进行正则化以及在攻击中注入随机性来产生强攻击。相反，我们通过反应频率的镜头来分析对抗性训练的行为。我们经验发现，对抗性训练导致神经网络对高频信息的收敛程度较低，导致每个数据附近的预测高度振荡。为了高效有效地学习高频内容，我们首先证明了频率原理中的一个普遍现象，即先学习较低的频率在对抗性训练中仍然成立。在此基础上，我们提出了相移对抗性训练(PhaseAT)，该模型通过将高频成分转移到发生快速收敛的低频范围来学习高频成分。在评估方面，我们在CIFAR-10和ImageNet上进行了实验，并使用精心设计的自适应攻击进行了可靠的评估。综合结果表明，PhaseAT算法显著提高了高频信息的收敛速度。这使得模型能够在每个数据附近平滑预测，从而提高了对手的稳健性。



## **45. WASMixer: Binary Obfuscation for WebAssembly**

WASMixer：WebAssembly的二进制混淆 cs.CR

**SubmitDate**: 2023-08-06    [abs](http://arxiv.org/abs/2308.03123v1) [paper-pdf](http://arxiv.org/pdf/2308.03123v1)

**Authors**: Shangtong Cao, Ningyu He, Yao Guo, Haoyu Wang

**Abstract**: WebAssembly (Wasm) is an emerging binary format that draws great attention from our community. However, Wasm binaries are weakly protected, as they can be read, edited, and manipulated by adversaries using either the officially provided readable text format (i.e., wat) or some advanced binary analysis tools. Reverse engineering of Wasm binaries is often used for nefarious intentions, e.g., identifying and exploiting both classic vulnerabilities and Wasm specific vulnerabilities exposed in the binaries. However, no Wasm-specific obfuscator is available in our community to secure the Wasm binaries. To fill the gap, in this paper, we present WASMixer, the first general-purpose Wasm binary obfuscator, enforcing data-level (string literals and function names) and code-level (control flow and instructions) obfuscation for Wasm binaries. We propose a series of key techniques to overcome challenges during Wasm binary rewriting, including an on-demand decryption method to minimize the impact brought by decrypting the data in memory area, and code splitting/reconstructing algorithms to handle structured control flow in Wasm. Extensive experiments demonstrate the correctness, effectiveness and efficiency of WASMixer. Our research has shed light on the promising direction of Wasm binary research, including Wasm code protection, Wasm binary diversification, and the attack-defense arm race of Wasm binaries.

摘要: WebAssembly(WASM)是一种新兴的二进制格式，引起了我们社区的极大关注。然而，WASM二进制文件受到的保护较弱，因为攻击者可以使用官方提供的可读文本格式(即WAT)或一些高级二进制分析工具来读取、编辑和操纵它们。WASM二进制文件的反向工程经常被用于邪恶的目的，例如识别和利用二进制文件中暴露的传统漏洞和WASM特定漏洞。然而，在我们的社区中没有特定于WASM的模糊处理程序可用来保护WASM二进制文件。为了填补这一空白，在本文中，我们提出了第一个通用的WASM二进制混淆程序WASMixer，它对WASM二进制文件实施了数据级(字符串文字和函数名)和代码级(控制流和指令)的混淆。我们提出了一系列关键技术来克服WASM二进制重写过程中的挑战，包括按需解密方法以最大限度地减少解密内存区数据所带来的影响，以及代码拆分/重构算法来处理WASM中的结构化控制流。大量实验证明了WASMixer的正确性、有效性和高效性。我们的研究揭示了WASM二进制研究的前景，包括WASM代码保护、WASM二进制多样化以及WASM二进制的攻防手臂竞赛。



## **46. SAAM: Stealthy Adversarial Attack on Monoculor Depth Estimation**

SAAM：对单色深度估计的隐身对抗攻击 cs.CV

**SubmitDate**: 2023-08-06    [abs](http://arxiv.org/abs/2308.03108v1) [paper-pdf](http://arxiv.org/pdf/2308.03108v1)

**Authors**: Amira Guesmi, Muhammad Abdullah Hanif, Bassem Ouni, Muhammad Shafique

**Abstract**: In this paper, we investigate the vulnerability of MDE to adversarial patches. We propose a novel \underline{S}tealthy \underline{A}dversarial \underline{A}ttacks on \underline{M}DE (SAAM) that compromises MDE by either corrupting the estimated distance or causing an object to seamlessly blend into its surroundings. Our experiments, demonstrate that the designed stealthy patch successfully causes a DNN-based MDE to misestimate the depth of objects. In fact, our proposed adversarial patch achieves a significant 60\% depth error with 99\% ratio of the affected region. Importantly, despite its adversarial nature, the patch maintains a naturalistic appearance, making it inconspicuous to human observers. We believe that this work sheds light on the threat of adversarial attacks in the context of MDE on edge devices. We hope it raises awareness within the community about the potential real-life harm of such attacks and encourages further research into developing more robust and adaptive defense mechanisms.

摘要: 在本文中，我们研究了MDE对敌意补丁的脆弱性。我们提出了一种新的基于{M}DE(SAAM)的{A}大头针，它破坏了估计的距离或使物体无缝地融入其周围，从而折衷了MDE。我们的实验表明，所设计的隐身补丁成功地导致了基于DNN的MDE错误估计目标的深度。事实上，我们提出的对抗性补丁获得了显著的60%的深度误差，其受影响区域的比率为99%。重要的是，尽管它具有对抗性，但它保持了一种自然主义的外观，使它对人类观察者来说并不引人注目。我们相信，这项工作有助于揭示边缘设备上的MDE环境中的对抗性攻击威胁。我们希望它能提高社区对此类攻击的潜在现实危害的认识，并鼓励进一步研究开发更强大和适应性更强的防御机制。



## **47. Using Overlapping Methods to Counter Adversaries in Community Detection**

利用重叠方法对抗社区发现中的敌手 cs.SI

28 pages, 10 figures

**SubmitDate**: 2023-08-06    [abs](http://arxiv.org/abs/2308.03081v1) [paper-pdf](http://arxiv.org/pdf/2308.03081v1)

**Authors**: Benjamin A. Miller, Kevin Chan, Tina Eliassi-Rad

**Abstract**: When dealing with large graphs, community detection is a useful data triage tool that can identify subsets of the network that a data analyst should investigate. In an adversarial scenario, the graph may be manipulated to avoid scrutiny of certain nodes by the analyst. Robustness to such behavior is an important consideration for data analysts in high-stakes scenarios such as cyber defense and counterterrorism. In this paper, we evaluate the use of overlapping community detection methods in the presence of adversarial attacks aimed at lowering the priority of a specific vertex. We formulate the data analyst's choice as a Stackelberg game in which the analyst chooses a community detection method and the attacker chooses an attack strategy in response. Applying various attacks from the literature to seven real network datasets, we find that, when the attacker has a sufficient budget, overlapping community detection methods outperform non-overlapping methods, often overwhelmingly so. This is the case when the attacker can only add edges that connect to the target and when the capability is added to add edges between neighbors of the target. We also analyze the tradeoff between robustness in the presence of an attack and performance when there is no attack. Our extensible analytic framework enables network data analysts to take these considerations into account and incorporate new attacks and community detection methods as they are developed.

摘要: 在处理大型图表时，社区检测是一个有用的数据分类工具，它可以识别数据分析师应该调查的网络子集。在对抗性场景中，该图可能被操纵以避免分析师对某些节点的仔细检查。对于网络防御和反恐等高风险场景中的数据分析师来说，对此类行为的稳健性是一个重要考虑因素。在本文中，我们评估了重叠社区检测方法在存在对抗性攻击的情况下的使用，旨在降低特定顶点的优先级。我们将数据分析师的选择描述为Stackelberg博弈，其中分析师选择社区检测方法，攻击者选择攻击策略作为响应。将文献中的各种攻击应用于七个真实的网络数据集，我们发现，当攻击者有足够的预算时，重叠的社区检测方法的性能优于非重叠的方法，通常是压倒性的。当攻击者只能添加连接到目标的边，并且添加了在目标的邻居之间添加边的功能时，就会出现这种情况。我们还分析了在存在攻击时的健壮性和在没有攻击时的性能之间的权衡。我们的可扩展分析框架使网络数据分析人员能够将这些考虑因素考虑在内，并在开发新的攻击和社区检测方法时纳入这些方法。



## **48. D4: Detection of Adversarial Diffusion Deepfakes Using Disjoint Ensembles**

D4：利用不相交集成检测对抗性扩散深度假象 cs.LG

**SubmitDate**: 2023-08-06    [abs](http://arxiv.org/abs/2202.05687v3) [paper-pdf](http://arxiv.org/pdf/2202.05687v3)

**Authors**: Ashish Hooda, Neal Mangaokar, Ryan Feng, Kassem Fawaz, Somesh Jha, Atul Prakash

**Abstract**: Detecting diffusion-generated deepfake images remains an open problem. Current detection methods fail against an adversary who adds imperceptible adversarial perturbations to the deepfake to evade detection. In this work, we propose Disjoint Diffusion Deepfake Detection (D4), a deepfake detector designed to improve black-box adversarial robustness beyond de facto solutions such as adversarial training. D4 uses an ensemble of models over disjoint subsets of the frequency spectrum to significantly improve adversarial robustness. Our key insight is to leverage a redundancy in the frequency domain and apply a saliency partitioning technique to disjointly distribute frequency components across multiple models. We formally prove that these disjoint ensembles lead to a reduction in the dimensionality of the input subspace where adversarial deepfakes lie, thereby making adversarial deepfakes harder to find for black-box attacks. We then empirically validate the D4 method against several black-box attacks and find that D4 significantly outperforms existing state-of-the-art defenses applied to diffusion-generated deepfake detection. We also demonstrate that D4 provides robustness against adversarial deepfakes from unseen data distributions as well as unseen generative techniques.

摘要: 检测扩散产生的深度伪像仍然是一个悬而未决的问题。当前的检测方法无法对抗将不可察觉的对抗性扰动添加到深度伪码以逃避检测的对手。在这项工作中，我们提出了不相交扩散深伪检测(D4)，这是一种深伪检测器，旨在提高黑盒对抗解决方案(如对抗训练)之外的稳健性。D4使用频谱的不相交子集上的模型集合来显著提高对手的稳健性。我们的主要见解是利用频域中的冗余，并应用显著分区技术来在多个模型之间分散分布频率分量。我们正式证明了这些不相交的集成导致对抗性深伪所在的输入子空间的维度降低，从而使得对抗性深伪更难为黑盒攻击找到。然后，我们通过实验验证了D4方法对几种黑盒攻击的抵抗能力，发现D4方法的性能明显优于现有的应用于扩散生成的深度伪检测的最先进防御方法。我们还证明了D4对来自不可见数据分布的敌意深度假冒以及不可见的生成技术具有健壮性。



## **49. FLAME: Taming Backdoors in Federated Learning (Extended Version 1)**

火焰：联合学习中的驯服后门(扩展版本1) cs.CR

This extended version incorporates a novel section (Section 10) that  provides a comprehensive analysis of recent proposed attacks, notably "3DFed:  Adaptive and extensible framework for covert backdoor attack in federated  learning" by Li et al. This new section addresses flawed assertions made in  the papers that aim to bypass FLAME or misinterpreted its fundamental design  principles

**SubmitDate**: 2023-08-05    [abs](http://arxiv.org/abs/2101.02281v5) [paper-pdf](http://arxiv.org/pdf/2101.02281v5)

**Authors**: Thien Duc Nguyen, Phillip Rieger, Huili Chen, Hossein Yalame, Helen Möllering, Hossein Fereidooni, Samuel Marchal, Markus Miettinen, Azalia Mirhoseini, Shaza Zeitouni, Farinaz Koushanfar, Ahmad-Reza Sadeghi, Thomas Schneider

**Abstract**: Federated Learning (FL) is a collaborative machine learning approach allowing participants to jointly train a model without having to share their private, potentially sensitive local datasets with others. Despite its benefits, FL is vulnerable to backdoor attacks, in which an adversary injects manipulated model updates into the model aggregation process so that the resulting model will provide targeted false predictions for specific adversary-chosen inputs. Proposed defenses against backdoor attacks based on detecting and filtering out malicious model updates consider only very specific and limited attacker models, whereas defenses based on differential privacy-inspired noise injection significantly deteriorate the benign performance of the aggregated model. To address these deficiencies, we introduce FLAME, a defense framework that estimates the sufficient amount of noise to be injected to ensure the elimination of backdoors while maintaining the model performance. To minimize the required amount of noise, FLAME uses a model clustering and weight clipping approach. Our evaluation of FLAME on several datasets stemming from application areas including image classification, word prediction, and IoT intrusion detection demonstrates that FLAME removes backdoors effectively with a negligible impact on the benign performance of the models. Furthermore, following the considerable attention that our research has received after its presentation at USENIX SEC 2022, FLAME has become the subject of numerous investigations proposing diverse attack methodologies in an attempt to circumvent it. As a response to these endeavors, we provide a comprehensive analysis of these attempts. Our findings show that these papers (e.g., 3DFed [36]) have not fully comprehended nor correctly employed the fundamental principles underlying FLAME, i.e., our defense mechanism effectively repels these attempted attacks.

摘要: 联合学习(FL)是一种协作式机器学习方法，允许参与者联合训练模型，而不必与其他人共享他们私有的、潜在敏感的本地数据集。尽管FL有好处，但它很容易受到后门攻击，在后门攻击中，对手将操纵的模型更新注入模型聚合过程，从而生成的模型将为对手选择的特定输入提供有针对性的错误预测。提出的基于检测和过滤恶意模型更新的后门攻击防御方案只考虑非常具体和有限的攻击者模型，而基于差异隐私激发噪声注入的防御方案会显著降低聚合模型的良性性能。为了解决这些不足，我们引入了FLAME，这是一个防御框架，可以估计要注入的足够数量的噪音，以确保在保持模型性能的同时消除后门。为了最大限度地减少所需的噪声量，FLAME使用了模型聚类和权重裁剪方法。我们在几个来自图像分类、词语预测和物联网入侵检测等应用领域的数据集上的评估表明，FLAME有效地移除了后门，而对模型的良性性能的影响可以忽略不计。此外，在我们的研究在USENIX SEC 2022上展示后受到了相当大的关注，FLAME已经成为许多调查的主题，这些调查提出了各种攻击方法，试图绕过它。作为对这些努力的回应，我们对这些尝试进行了全面的分析。我们的发现表明，这些论文(例如，3DFed[36])没有完全理解或正确使用火焰背后的基本原理，即我们的防御机制有效地击退了这些企图攻击。



## **50. An AI-Enabled Framework to Defend Ingenious MDT-based Attacks on the Emerging Zero Touch Cellular Networks**

一种支持人工智能的框架，用于防御对新兴的零接触蜂窝网络的基于MDT的巧妙攻击 cs.LG

15 pages, 5 figures, 1 table

**SubmitDate**: 2023-08-05    [abs](http://arxiv.org/abs/2308.02923v1) [paper-pdf](http://arxiv.org/pdf/2308.02923v1)

**Authors**: Aneeqa Ijaz, Waseem Raza, Hasan Farooq, Marvin Manalastas, Ali Imran

**Abstract**: Deep automation provided by self-organizing network (SON) features and their emerging variants such as zero touch automation solutions is a key enabler for increasingly dense wireless networks and pervasive Internet of Things (IoT). To realize their objectives, most automation functionalities rely on the Minimization of Drive Test (MDT) reports. The MDT reports are used to generate inferences about network state and performance, thus dynamically change network parameters accordingly. However, the collection of MDT reports from commodity user devices, particularly low cost IoT devices, make them a vulnerable entry point to launch an adversarial attack on emerging deeply automated wireless networks. This adds a new dimension to the security threats in the IoT and cellular networks. Existing literature on IoT, SON, or zero touch automation does not address this important problem. In this paper, we investigate an impactful, first of its kind adversarial attack that can be launched by exploiting the malicious MDT reports from the compromised user equipment (UE). We highlight the detrimental repercussions of this attack on the performance of common network automation functions. We also propose a novel Malicious MDT Reports Identification framework (MRIF) as a countermeasure to detect and eliminate the malicious MDT reports using Machine Learning and verify it through a use-case. Thus, the defense mechanism can provide the resilience and robustness for zero touch automation SON engines against the adversarial MDT attacks

摘要: 自组织网络(SON)功能及其新兴变体(如零接触自动化解决方案)提供的深度自动化是日益密集的无线网络和无处不在的物联网(IoT)的关键推动因素。为了实现他们的目标，大多数自动化功能依赖于最小化驾驶测试(MDT)报告。MDT报告用于生成有关网络状态和性能的推论，从而相应地动态更改网络参数。然而，从商用用户设备，特别是低成本物联网设备收集的MDT报告，使它们成为对新兴的高度自动化的无线网络发起敌意攻击的脆弱切入点。这为物联网和蜂窝网络中的安全威胁增加了一个新的维度。关于物联网、SON或零接触自动化的现有文献没有解决这一重要问题。在本文中，我们研究了一种通过利用来自受攻击的用户设备(UE)的恶意MDT报告来发起的、具有影响力的、第一种类型的对抗性攻击。我们重点介绍了这种攻击对常见网络自动化功能性能的不利影响。我们还提出了一种新的恶意MDT报告识别框架(MRIF)，作为利用机器学习检测和消除恶意MDT报告的对策，并通过用例进行了验证。因此，该防御机制可以为零接触自动化SON引擎提供对恶意MDT攻击的弹性和健壮性



