# Latest Adversarial Attack Papers
**update at 2022-03-10 06:31:41**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. RAPTEE: Leveraging trusted execution environments for Byzantine-tolerant peer sampling services**

RAPTEE：为拜占庭容忍的对等采样服务利用可信执行环境 cs.DC

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2203.04258v1)

**Authors**: Matthieu Pigaglio, Joachim Bruneau-Queyreix, David Bromberg, Davide Frey, Etienne Rivière, Laurent Réveillère

**Abstracts**: Peer sampling is a first-class abstraction used in distributed systems for overlay management and information dissemination. The goal of peer sampling is to continuously build and refresh a partial and local view of the full membership of a dynamic, large-scale distributed system. Malicious nodes under the control of an adversary may aim at being over-represented in the views of correct nodes, increasing their impact on the proper operation of protocols built over peer sampling. State-of-the-art Byzantine resilient peer sampling protocols reduce this bias as long as Byzantines are not overly present. This paper studies the benefits brought to the resilience of peer sampling services when considering that a small portion of trusted nodes can run code whose authenticity and integrity can be assessed within a trusted execution environment, and specifically Intel's software guard extensions technology (SGX). We present RAPTEE, a protocol that builds and leverages trusted gossip-based communications to hamper an adversary's ability to increase its system-wide representation in the views of all nodes. We apply RAPTEE to BRAHMS, the most resilient peer sampling protocol to date. Experiments with 10,000 nodes show that with only 1% of SGX-capable devices, RAPTEE can reduce the proportion of Byzantine IDs in the view of honest nodes by up to 17% when the system contains 10% of Byzantine nodes. In addition, the security guarantees of RAPTEE hold even in the presence of a powerful attacker attempting to identify trusted nodes and injecting view-poisoned trusted nodes.

摘要: 对等抽样是分布式系统中用于覆盖管理和信息分发的一级抽象。对等抽样的目标是持续构建和刷新动态、大规模分布式系统的完整成员的局部和局部视图。在对手控制下的恶意节点可能旨在在正确节点的视图中被过度表示，从而增加它们对建立在对等采样之上的协议的正确操作的影响。只要拜占庭人不过度存在，最先进的拜占庭弹性对等采样协议就会减少这种偏见。本文研究了当考虑到一小部分可信节点可以在可信执行环境(特别是Intel的软件保护扩展技术(SGX))中运行其真实性和完整性可以被评估的代码时，给对等采样服务的弹性带来的好处。我们提出了RAPTEE，这是一种协议，它建立并利用基于可信八卦的通信来阻碍对手在所有节点的视图中增加其系统范围表示的能力。我们将RAPTEE应用于Brahms，这是迄今为止最具弹性的对等采样协议。在10000个节点上的实验表明，当系统包含10%的拜占庭节点时，RAPTEE可以在仅使用1%的SGX功能的设备的情况下，将拜占庭ID在诚实节点中的比例降低高达17%。此外，即使强大的攻击者试图识别受信任节点并注入视图中毒的受信任节点，RAPTEE的安全保证仍然有效。



## **2. Towards Effective and Robust Neural Trojan Defenses via Input Filtering**

通过输入过滤实现高效、健壮的神经木马防御 cs.CR

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2202.12154v3)

**Authors**: Kien Do, Haripriya Harikumar, Hung Le, Dung Nguyen, Truyen Tran, Santu Rana, Dang Nguyen, Willy Susilo, Svetha Venkatesh

**Abstracts**: Trojan attacks on deep neural networks are both dangerous and surreptitious. Over the past few years, Trojan attacks have advanced from using only a single input-agnostic trigger and targeting only one class to using multiple, input-specific triggers and targeting multiple classes. However, Trojan defenses have not caught up with this development. Most defense methods still make out-of-date assumptions about Trojan triggers and target classes, thus, can be easily circumvented by modern Trojan attacks. To deal with this problem, we propose two novel "filtering" defenses called Variational Input Filtering (VIF) and Adversarial Input Filtering (AIF) which leverage lossy data compression and adversarial learning respectively to effectively purify all potential Trojan triggers in the input at run time without making assumptions about the number of triggers/target classes or the input dependence property of triggers. In addition, we introduce a new defense mechanism called "Filtering-then-Contrasting" (FtC) which helps avoid the drop in classification accuracy on clean data caused by "filtering", and combine it with VIF/AIF to derive new defenses of this kind. Extensive experimental results and ablation studies show that our proposed defenses significantly outperform well-known baseline defenses in mitigating five advanced Trojan attacks including two recent state-of-the-art while being quite robust to small amounts of training data and large-norm triggers.

摘要: 特洛伊木马对深层神经网络的攻击既危险又隐蔽。在过去的几年中，特洛伊木马攻击已经从只使用一个与输入无关的触发器和只针对一个类发展到使用多个特定于输入的触发器和以多个类为目标。然而，特洛伊木马防御并没有跟上这一发展。大多数防御方法仍然对木马触发器和目标类做出过时的假设，因此很容易被现代木马攻击所规避。为了解决这一问题，我们提出了两种新的“过滤”防御机制，称为变量输入过滤(VIF)和对抗性输入过滤(AIF)，它们分别利用有损数据压缩和对抗性学习在运行时有效地净化输入中所有潜在的木马触发器，而不需要假设触发器/目标类的数量或触发器的输入依赖属性。此外，我们引入了一种新的防御机制，称为“过滤-然后-对比”(FTC)，它有助于避免“过滤”导致的对干净数据分类精度的下降，并将其与VIF/AIF相结合来派生出新的防御机制。广泛的实验结果和烧蚀研究表明，我们提出的防御方案在缓解五种高级木马攻击(包括最近的两种)方面明显优于众所周知的基线防御方案，同时对少量训练数据和大范数触发事件具有相当的鲁棒性。



## **3. Adaptative Perturbation Patterns: Realistic Adversarial Learning for Robust NIDS**

自适应扰动模式：鲁棒NIDS的现实对抗性学习 cs.CR

16 pages, 6 tables, 8 figures, Future Internet journal

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2203.04234v1)

**Authors**: João Vitorino, Nuno Oliveira, Isabel Praça

**Abstracts**: Adversarial attacks pose a major threat to machine learning and to the systems that rely on it. Nonetheless, adversarial examples cannot be freely generated for domains with tabular data, such as cybersecurity. This work establishes the fundamental constraint levels required to achieve realism and introduces the Adaptative Perturbation Pattern Method (A2PM) to fulfill these constraints in a gray-box setting. A2PM relies on pattern sequences that are independently adapted to the characteristics of each class to create valid and coherent data perturbations. The developed method was evaluated in a cybersecurity case study with two scenarios: Enterprise and Internet of Things (IoT) networks. Multilayer Perceptron (MLP) and Random Forest (RF) classifiers were created with regular and adversarial training, using the CIC-IDS2017 and IoT-23 datasets. In each scenario, targeted and untargeted attacks were performed against the classifiers, and the generated examples were compared with the original network traffic flows to assess their realism. The obtained results demonstrate that A2PM provides a time efficient generation of realistic adversarial examples, which can be advantageous for both adversarial training and attacks.

摘要: 对抗性攻击对机器学习和依赖机器学习的系统构成了重大威胁。尽管如此，对于具有表格数据的域(如网络安全)，不能免费生成敌意示例。这项工作建立了达到真实感所需的基本约束水平，并引入了自适应扰动模式方法(A2PM)来满足灰箱设置中的这些约束。A2PM依赖于独立适应每个类别的特征的模式序列来创建有效且一致的数据扰动。开发的方法在两个场景的网络安全案例研究中进行了评估：企业和物联网(IoT)网络。使用CIC-IDS2017和IoT-23数据集，通过定期和对抗性训练创建了多层感知器(MLP)和随机森林(RF)分类器。在每个场景中，对分类器执行目标攻击和非目标攻击，并将生成的示例与原始网络流量进行比较，以评估其真实性。所获得的结果表明，A2 PM提供了一种时间效率高的真实对抗性示例的生成，这对于对抗性训练和攻击都是有利的。



## **4. Robustly-reliable learners under poisoning attacks**

中毒攻击下健壮可靠的学习者 cs.LG

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2203.04160v1)

**Authors**: Maria-Florina Balcan, Avrim Blum, Steve Hanneke, Dravyansh Sharma

**Abstracts**: Data poisoning attacks, in which an adversary corrupts a training set with the goal of inducing specific desired mistakes, have raised substantial concern: even just the possibility of such an attack can make a user no longer trust the results of a learning system. In this work, we show how to achieve strong robustness guarantees in the face of such attacks across multiple axes.   We provide robustly-reliable predictions, in which the predicted label is guaranteed to be correct so long as the adversary has not exceeded a given corruption budget, even in the presence of instance targeted attacks, where the adversary knows the test example in advance and aims to cause a specific failure on that example. Our guarantees are substantially stronger than those in prior approaches, which were only able to provide certificates that the prediction of the learning algorithm does not change, as opposed to certifying that the prediction is correct, as we are able to achieve in our work. Remarkably, we provide a complete characterization of learnability in this setting, in particular, nearly-tight matching upper and lower bounds on the region that can be certified, as well as efficient algorithms for computing this region given an ERM oracle. Moreover, for the case of linear separators over logconcave distributions, we provide efficient truly polynomial time algorithms (i.e., non-oracle algorithms) for such robustly-reliable predictions.   We also extend these results to the active setting where the algorithm adaptively asks for labels of specific informative examples, and the difficulty is that the adversary might even be adaptive to this interaction, as well as to the agnostic learning setting where there is no perfect classifier even over the uncorrupted data.

摘要: 数据中毒攻击，即对手破坏训练集，目的是诱导特定的预期错误，已经引起了极大的担忧：即使是这种攻击的可能性也会让用户不再信任学习系统的结果。在这项工作中，我们展示了如何在面对这种跨越多个轴的攻击时实现强鲁棒性保证。我们提供鲁棒可靠的预测，其中只要对手没有超过给定的腐败预算，即使在存在实例目标攻击的情况下，预测的标签也被保证是正确的，其中对手提前知道测试示例并旨在导致该示例上的特定失败。我们的保证比以前的方法要强得多，以前的方法只能提供学习算法的预测不变的证书，而不是像我们在工作中能够实现的那样，证明预测是正确的。值得注意的是，在这种情况下，我们给出了可学习性的完整刻画，特别是在可证明的区域的上下界几乎紧匹配的情况下，以及在给定ERM预言的情况下计算该区域的高效算法。此外，对于对数凹分布上线性分隔符的情况，我们为这种鲁棒可靠的预测提供了有效的真多项式时间算法(即非Oracle算法)。我们还将这些结果扩展到主动设置，其中算法自适应地要求特定信息示例的标签，困难在于对手甚至可能适应这种交互，以及即使在未被破坏的数据上也没有完美分类器的不可知学习设置。



## **5. Adversarial Texture for Fooling Person Detectors in the Physical World**

物理世界中愚人探测器的对抗性纹理 cs.CV

Accepted by CVPR 2022

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2203.03373v2)

**Authors**: Zhanhao Hu, Siyuan Huang, Xiaopei Zhu, Xiaolin Hu, Fuchun Sun, Bo Zhang

**Abstracts**: Nowadays, cameras equipped with AI systems can capture and analyze images to detect people automatically. However, the AI system can make mistakes when receiving deliberately designed patterns in the real world, i.e., physical adversarial examples. Prior works have shown that it is possible to print adversarial patches on clothes to evade DNN-based person detectors. However, these adversarial examples could have catastrophic drops in the attack success rate when the viewing angle (i.e., the camera's angle towards the object) changes. To perform a multi-angle attack, we propose Adversarial Texture (AdvTexture). AdvTexture can cover clothes with arbitrary shapes so that people wearing such clothes can hide from person detectors from different viewing angles. We propose a generative method, named Toroidal-Cropping-based Expandable Generative Attack (TC-EGA), to craft AdvTexture with repetitive structures. We printed several pieces of cloth with AdvTexure and then made T-shirts, skirts, and dresses in the physical world. Experiments showed that these clothes could fool person detectors in the physical world.

摘要: 如今，配备人工智能系统的摄像头可以捕捉和分析图像，自动检测人。然而，当接收到现实世界中故意设计的模式时，人工智能系统可能会出错，即物理对抗性示例。以前的工作已经表明，可以在衣服上打印敌意补丁来躲避基于DNN的人检测器。然而，当视角(即相机朝向对象的角度)改变时，这些对抗性的例子可能会使攻击成功率灾难性地下降。为了进行多角度攻击，我们提出了对抗性纹理(AdvTexture)。AdvTexture可以覆盖任意形状的衣服，这样穿着这种衣服的人就可以从不同的视角躲避人的探测器。提出了一种基于环形裁剪的可扩展生成攻击方法(TC-EGA)来制作具有重复结构的AdvTexture。我们用AdvTexure打印了几块布，然后在现实世界中制作了T恤、裙子和连衣裙。实验表明，这些衣服可以愚弄物理世界中的人体探测器。



## **6. Shape-invariant 3D Adversarial Point Clouds**

形状不变的三维对抗性点云 cs.CV

Accepted at CVPR 2022

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2203.04041v1)

**Authors**: Qidong Huang, Xiaoyi Dong, Dongdong Chen, Hang Zhou, Weiming Zhang, Nenghai Yu

**Abstracts**: Adversary and invisibility are two fundamental but conflict characters of adversarial perturbations. Previous adversarial attacks on 3D point cloud recognition have often been criticized for their noticeable point outliers, since they just involve an "implicit constrain" like global distance loss in the time-consuming optimization to limit the generated noise. While point cloud is a highly structured data format, it is hard to metric and constrain its perturbation with a simple loss properly. In this paper, we propose a novel Point-Cloud Sensitivity Map to boost both the efficiency and imperceptibility of point perturbations. This map reveals the vulnerability of point cloud recognition models when encountering shape-invariant adversarial noises. These noises are designed along the shape surface with an "explicit constrain" instead of extra distance loss. Specifically, we first apply a reversible coordinate transformation on each point of the point cloud input, to reduce one degree of point freedom and limit its movement on the tangent plane. Then we calculate the best attacking direction with the gradients of the transformed point cloud obtained on the white-box model. Finally we assign each point with a non-negative score to construct the sensitivity map, which benefits both white-box adversarial invisibility and black-box query-efficiency extended in our work. Extensive evaluations prove that our method can achieve the superior performance on various point cloud recognition models, with its satisfying adversarial imperceptibility and strong resistance to different point cloud defense settings. Our code is available at: https://github.com/shikiw/SI-Adv.

摘要: 对抗性和隐蔽性是对抗性扰动的两个基本但又相互冲突的特征。以前针对3D点云识别的敌意攻击经常因为其明显的点离群值而受到批评，因为它们只是在耗时的优化过程中涉及诸如全局距离损失这样的“隐式约束”，以限制生成的噪声。虽然点云是一种高度结构化的数据格式，但是很难用简单的损失来度量和约束它的扰动。在本文中，我们提出了一种新的点云敏感度图，以提高点扰动的效率和隐蔽性。这张地图揭示了点云识别模型在遇到形状不变的对抗性噪声时的脆弱性。这些噪波是沿着形状表面设计的，带有“显式约束”，而不是额外的距离损失。具体地说，我们首先对点云输入的每个点应用可逆坐标变换，以减少一个点自由度并限制其在切面上的移动。然后利用白盒模型得到的变换后的点云梯度计算最佳攻击方向。最后，我们给每个点分配一个非负分数来构造敏感度图，这样既有利于白盒对抗不可见性，也有利于提高黑盒查询效率。广泛的评测表明，该方法在各种点云识别模型上都能取得较好的性能，具有令人满意的对抗性和对不同点云防御设置的较强抵抗力。我们的代码可从以下网址获得：https://github.com/shikiw/SI-Adv.



## **7. ART-Point: Improving Rotation Robustness of Point Cloud Classifiers via Adversarial Rotation**

ART-Point：通过对抗性轮换提高点云分类器的旋转稳健性 cs.CV

CVPR 2022

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2203.03888v1)

**Authors**: Robin Wang, Yibo Yang, Dacheng Tao

**Abstracts**: Point cloud classifiers with rotation robustness have been widely discussed in the 3D deep learning community. Most proposed methods either use rotation invariant descriptors as inputs or try to design rotation equivariant networks. However, robust models generated by these methods have limited performance under clean aligned datasets due to modifications on the original classifiers or input space. In this study, for the first time, we show that the rotation robustness of point cloud classifiers can also be acquired via adversarial training with better performance on both rotated and clean datasets. Specifically, our proposed framework named ART-Point regards the rotation of the point cloud as an attack and improves rotation robustness by training the classifier on inputs with Adversarial RoTations. We contribute an axis-wise rotation attack that uses back-propagated gradients of the pre-trained model to effectively find the adversarial rotations. To avoid model over-fitting on adversarial inputs, we construct rotation pools that leverage the transferability of adversarial rotations among samples to increase the diversity of training data. Moreover, we propose a fast one-step optimization to efficiently reach the final robust model. Experiments show that our proposed rotation attack achieves a high success rate and ART-Point can be used on most existing classifiers to improve the rotation robustness while showing better performance on clean datasets than state-of-the-art methods.

摘要: 具有旋转鲁棒性的点云分类器在三维深度学习领域得到了广泛的讨论。大多数提出的方法要么使用旋转不变描述符作为输入，要么尝试设计旋转等变网络。然而，由于对原始分类器或输入空间的修改，这些方法生成的鲁棒模型在干净的对齐数据集上的性能有限。在这项研究中，我们首次表明，点云分类器的旋转鲁棒性也可以通过对抗性训练获得，在旋转数据集和清洁数据集上都具有更好的性能。具体地说，我们提出的ART-Point框架将点云的旋转视为一种攻击，并通过对具有对抗性旋转的输入训练分类器来提高旋转的鲁棒性。我们提出了一种轴向旋转攻击，它使用预训练模型的反向传播梯度来有效地找到对抗性旋转。为了避免对抗性输入的模型过拟合，我们构建了轮转池，利用对抗性轮换在样本之间的可传递性来增加训练数据的多样性。此外，我们还提出了一种快速的一步优化方法来高效地得到最终的鲁棒模型。实验表明，我们提出的旋转攻击取得了很高的成功率，ART-Point可以在现有的大多数分类器上使用，以提高旋转的鲁棒性，同时在干净的数据集上表现出比现有方法更好的性能。



## **8. Submodularity-based False Data Injection Attack Scheme in Multi-agent Dynamical Systems**

多智能体动态系统中基于子模块性的虚假数据注入攻击方案 math.DS

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2201.06017v2)

**Authors**: Xiaoyu Luo, Chengcheng Zhao, Chongrong Fang, Jianping He

**Abstracts**: Consensus in multi-agent dynamical systems is prone to be sabotaged by the adversary, which has attracted much attention due to its key role in broad applications. In this paper, we study a new false data injection (FDI) attack design problem, where the adversary with limited capability aims to select a subset of agents and manipulate their local multi-dimensional states to maximize the consensus convergence error. We first formulate the FDI attack design problem as a combinatorial optimization problem and prove it is NP-hard. Then, based on the submodularity optimization theory, we show the convergence error is a submodular function of the set of the compromised agents, which satisfies the property of diminishing marginal returns. In other words, the benefit of adding an extra agent to the compromised set decreases as that set becomes larger. With this property, we exploit the greedy scheme to find the optimal compromised agent set that can produce the maximum convergence error when adding one extra agent to that set each time. Thus, the FDI attack set selection algorithms are developed to obtain the near-optimal subset of the compromised agents. Furthermore, we derive the analytical suboptimality bounds and the worst-case running time under the proposed algorithms. Extensive simulation results are conducted to show the effectiveness of the proposed algorithm.

摘要: 多智能体动态系统中的共识容易受到对手的破坏，因其在广泛应用中的关键作用而备受关注。本文研究了一类新的虚假数据注入(FDI)攻击设计问题，其中能力有限的敌手的目标是选择一个Agent子集并操纵其局部多维状态以最大化共识收敛误差。我们首先将FDI攻击设计问题描述为一个组合优化问题，并证明了它是NP难的。然后，基于子模优化理论，证明了收敛误差是折衷智能体集合的子模函数，满足边际收益递减的性质。换句话说，向受危害的集合添加额外代理的好处随着该集合变得更大而降低。利用这一性质，我们利用贪婪方案来寻找最优的折衷智能体集合，该集合在每次额外增加一个智能体的情况下可以产生最大的收敛误差。因此，开发了FDI攻击集选择算法，以获得受攻击代理的近优子集。此外，我们还给出了所提出算法的分析次优界和最坏情况下的运行时间。大量的仿真结果表明了该算法的有效性。



## **9. Shadows can be Dangerous: Stealthy and Effective Physical-world Adversarial Attack by Natural Phenomenon**

阴影可能是危险的：自然现象对物理世界的隐秘而有效的对抗性攻击 cs.CV

This paper has been accepted by CVPR2022

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2203.03818v1)

**Authors**: Yiqi Zhong, Xianming Liu, Deming Zhai, Junjun Jiang, Xiangyang Ji

**Abstracts**: Estimating the risk level of adversarial examples is essential for safely deploying machine learning models in the real world. One popular approach for physical-world attacks is to adopt the "sticker-pasting" strategy, which however suffers from some limitations, including difficulties in access to the target or printing by valid colors. A new type of non-invasive attacks emerged recently, which attempt to cast perturbation onto the target by optics based tools, such as laser beam and projector. However, the added optical patterns are artificial but not natural. Thus, they are still conspicuous and attention-grabbed, and can be easily noticed by humans. In this paper, we study a new type of optical adversarial examples, in which the perturbations are generated by a very common natural phenomenon, shadow, to achieve naturalistic and stealthy physical-world adversarial attack under the black-box setting. We extensively evaluate the effectiveness of this new attack on both simulated and real-world environments. Experimental results on traffic sign recognition demonstrate that our algorithm can generate adversarial examples effectively, reaching 98.23% and 90.47% success rates on LISA and GTSRB test sets respectively, while continuously misleading a moving camera over 95% of the time in real-world scenarios. We also offer discussions about the limitations and the defense mechanism of this attack.

摘要: 估计对抗性示例的风险水平对于在现实世界中安全地部署机器学习模型是至关重要的。物理世界攻击的一种流行方法是采用“粘贴”策略，但该策略受到一些限制，包括难以接近目标或以有效颜色打印。最近出现了一种新型的非侵入性攻击，它试图通过激光束和投影仪等基于光学的工具对目标进行摄动。然而，添加的光学图案是人造的，但不是自然的。因此，它们仍然是引人注目和引人注目的，很容易被人类注意到。本文研究了一种新的光学对抗实例，其中的扰动是由一种非常常见的自然现象--阴影产生的，从而在黑盒环境下实现了自然主义的、隐身的物理世界对抗攻击。我们广泛评估了这种新攻击在模拟和真实环境中的有效性。在交通标志识别上的实验结果表明，该算法能够有效地生成对抗性样本，在LISA和GTSRB测试集上的成功率分别达到98.23%和90.47%，而在真实场景中，95%以上的时间都能连续误导移动的摄像机。我们还讨论了这种攻击的局限性和防御机制。



## **10. Adversarial Attacks in Cooperative AI**

协作式人工智能中的对抗性攻击 cs.LG

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2111.14833v3)

**Authors**: Ted Fujimoto, Arthur Paul Pedersen

**Abstracts**: Single-agent reinforcement learning algorithms in a multi-agent environment are inadequate for fostering cooperation. If intelligent agents are to interact and work together to solve complex problems, methods that counter non-cooperative behavior are needed to facilitate the training of multiple agents. This is the goal of cooperative AI. Recent research in adversarial machine learning, however, shows that models (e.g., image classifiers) can be easily deceived into making inferior decisions. Meanwhile, an important line of research in cooperative AI has focused on introducing algorithmic improvements that accelerate learning of optimally cooperative behavior. We argue that prominent methods of cooperative AI are exposed to weaknesses analogous to those studied in prior machine learning research. More specifically, we show that three algorithms inspired by human-like social intelligence are, in principle, vulnerable to attacks that exploit weaknesses introduced by cooperative AI's algorithmic improvements and report experimental findings that illustrate how these vulnerabilities can be exploited in practice.

摘要: 多智能体环境中的单智能体强化学习算法不能很好地促进协作。如果智能Agent要交互并共同工作来解决复杂问题，就需要针对不合作行为的方法，以便于多个Agent的训练。这是合作AI的目标。然而，最近在对抗性机器学习方面的研究表明，模型(例如，图像分类器)很容易被欺骗，从而做出较差的决策。同时，合作人工智能的一条重要研究方向是引入算法改进，以加速最优合作行为的学习。我们认为，合作人工智能的突出方法暴露了与先前机器学习研究中所研究的类似的弱点。更具体地说，我们证明了三种受类人类社会智能启发的算法原则上容易受到攻击，这些攻击利用合作AI的算法改进引入的弱点，并报告了实验结果，说明了如何在实践中利用这些漏洞。



## **11. Taxonomy of Machine Learning Safety: A Survey and Primer**

机器学习安全分类学：综述与入门读本 cs.LG

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2106.04823v2)

**Authors**: Sina Mohseni, Haotao Wang, Zhiding Yu, Chaowei Xiao, Zhangyang Wang, Jay Yadawa

**Abstracts**: The open-world deployment of Machine Learning (ML) algorithms in safety-critical applications such as autonomous vehicles needs to address a variety of ML vulnerabilities such as interpretability, verifiability, and performance limitations. Research explores different approaches to improve ML dependability by proposing new models and training techniques to reduce generalization error, achieve domain adaptation, and detect outlier examples and adversarial attacks. However, there is a missing connection between ongoing ML research and well-established safety principles. In this paper, we present a structured and comprehensive review of ML techniques to improve the dependability of ML algorithms in uncontrolled open-world settings. From this review, we propose the Taxonomy of ML Safety that maps state-of-the-art ML techniques to key engineering safety strategies. Our taxonomy of ML safety presents a safety-oriented categorization of ML techniques to provide guidance for improving dependability of the ML design and development. The proposed taxonomy can serve as a safety checklist to aid designers in improving coverage and diversity of safety strategies employed in any given ML system.

摘要: 机器学习(ML)算法在自动驾驶汽车等安全关键型应用中的开放世界部署需要解决各种ML漏洞，如可解释性、可验证性和性能限制。研究探索了通过提出新的模型和训练技术来提高ML可靠性的不同方法，以减少泛化错误，实现领域自适应，并检测离群点示例和敌意攻击。然而，正在进行的ML研究和公认的安全原则之间缺少联系。本文对ML技术进行了系统全面的综述，以提高ML算法在非受控开放环境下的可靠性。从这篇综述中，我们提出了ML安全分类法，它将最先进的ML技术映射到关键的工程安全策略。我们的ML安全分类法对ML技术进行了面向安全的分类，为提高ML设计和开发的可靠性提供指导。建议的分类可以作为安全检查表，帮助设计者提高在任何给定ML系统中采用的安全策略的复盖率和多样性。



## **12. Defending Graph Convolutional Networks against Dynamic Graph Perturbations via Bayesian Self-supervision**

基于贝叶斯自监督的图卷积网络抗动态图扰动 cs.LG

The paper is accepted by AAAI 2022

**SubmitDate**: 2022-03-07    [paper-pdf](http://arxiv.org/pdf/2203.03762v1)

**Authors**: Jun Zhuang, Mohammad Al Hasan

**Abstracts**: In recent years, plentiful evidence illustrates that Graph Convolutional Networks (GCNs) achieve extraordinary accomplishments on the node classification task. However, GCNs may be vulnerable to adversarial attacks on label-scarce dynamic graphs. Many existing works aim to strengthen the robustness of GCNs; for instance, adversarial training is used to shield GCNs against malicious perturbations. However, these works fail on dynamic graphs for which label scarcity is a pressing issue. To overcome label scarcity, self-training attempts to iteratively assign pseudo-labels to highly confident unlabeled nodes but such attempts may suffer serious degradation under dynamic graph perturbations. In this paper, we generalize noisy supervision as a kind of self-supervised learning method and then propose a novel Bayesian self-supervision model, namely GraphSS, to address the issue. Extensive experiments demonstrate that GraphSS can not only affirmatively alert the perturbations on dynamic graphs but also effectively recover the prediction of a node classifier when the graph is under such perturbations. These two advantages prove to be generalized over three classic GCNs across five public graph datasets.

摘要: 近年来，大量的证据表明，图卷积网络(GCNS)在节点分类任务中取得了非凡的成就。然而，在标签稀缺的动态图上，GCNS可能容易受到敌意攻击。许多现有的工作都是为了增强GCNS的健壮性，例如，使用对抗性训练来保护GCNS免受恶意干扰。然而，这些工作在动态图上失败了，因为对于动态图来说，标签稀缺是一个紧迫的问题。为了克服标签稀缺性，自训练试图迭代地将伪标签分配给高度自信的无标签节点，但是这种尝试在动态图扰动下可能遭受严重降级。本文将噪声监督推广为一种自监督学习方法，并提出了一种新的贝叶斯自监督模型GraphSS来解决这一问题。大量实验表明，GraphSS不仅能肯定地告警动态图上的扰动，而且当图受到扰动时，还能有效地恢复节点分类器的预测。事实证明，这两个优势在五个公共图数据集上的三个经典GCN上得到了推广。



## **13. Uncertify: Attacks Against Neural Network Certification**

未认证：针对神经网络认证的攻击 cs.LG

**SubmitDate**: 2022-03-07    [paper-pdf](http://arxiv.org/pdf/2108.11299v2)

**Authors**: Tobias Lorenz, Marta Kwiatkowska, Mario Fritz

**Abstracts**: Certifiers for neural networks have made great progress towards provable robustness guarantees against evasion attacks using adversarial examples. However, introducing certifiers into deep learning systems also opens up new attack vectors, which need to be considered before deployment. In this work, we conduct the first systematic analysis of training-time attacks against certifiers in practical application pipelines, identifying new threat vectors that can be exploited to degrade the overall system. Using these insights, we design two backdoor attacks against network certifiers, which can drastically reduce certified robustness. For example, adding 1% poisoned data points during training is sufficient to reduce certified robustness by up to 95 percentage points, effectively rendering the certifier useless. We analyze how such novel attacks can compromise the overall system's integrity or availability. Our extensive experiments across multiple datasets, model architectures, and certifiers demonstrate the wide applicability of these attacks. A first investigation into potential defenses shows that current approaches are insufficient to mitigate the issue, highlighting the need for new, more specific solutions.

摘要: 神经网络的认证器已经取得了很大的进展，可以用敌意的例子来证明对规避攻击的鲁棒性保证。然而，将认证器引入深度学习系统也会打开新的攻击向量，在部署之前需要考虑这些攻击向量。在这项工作中，我们首次对实际应用管道中针对认证器的训练时间攻击进行了系统分析，识别出可以用来降低整个系统性能的新威胁向量。利用这些见解，我们设计了两种针对网络认证器的后门攻击，这两种攻击会极大地降低认证的健壮性。例如，在训练期间添加1%的有毒数据点就足以将认证的健壮性降低高达95个百分点，从而有效地使认证器无用。我们分析了这种新颖的攻击如何危害整个系统的完整性或可用性。我们在多个数据集、模型架构和认证器上的广泛实验证明了这些攻击的广泛适用性。对潜在防御措施的首次调查显示，目前的方法不足以缓解这一问题，这突显了需要新的、更具体的解决方案。



## **14. The Dangerous Combo: Fileless Malware and Cryptojacking**

危险的组合：无文件恶意软件和密码劫持 cs.CR

9 Pages - Accepted to be published in SoutheastCon 2022 IEEE Region 3  Technical, Professional, and Student Conference. Mobile, Alabama, USA. Mar  31st to Apr 03rd 2022. https://ieeesoutheastcon.org/

**SubmitDate**: 2022-03-07    [paper-pdf](http://arxiv.org/pdf/2203.03175v1)

**Authors**: Said Varlioglu, Nelly Elsayed, Zag ElSayed, Murat Ozer

**Abstracts**: Fileless malware and cryptojacking attacks have appeared independently as the new alarming threats in 2017. After 2020, fileless attacks have been devastating for victim organizations with low-observable characteristics. Also, the amount of unauthorized cryptocurrency mining has increased after 2019. Adversaries have started to merge these two different cyberattacks to gain more invisibility and profit under "Fileless Cryptojacking." This paper aims to provide a literature review in academic papers and industry reports for this new threat. Additionally, we present a new threat hunting-oriented DFIR approach with the best practices derived from field experience as well as the literature. Last, this paper reviews the fundamentals of the fileless threat that can also help ransomware researchers examine similar patterns.

摘要: 无文件恶意软件和密码劫持攻击已经独立出现，成为2017年新的令人担忧的威胁。2020年后，无文件攻击对具有低可察觉特征的受害者组织来说是毁灭性的。此外，2019年之后，未经授权的加密货币挖掘量有所增加。对手已经开始合并这两种不同的网络攻击，以在“无文件密码劫持”下获得更多的隐蔽性和利润。本文旨在对有关这一新威胁的学术论文和行业报告中的文献进行综述。此外，我们提出了一种新的面向威胁追捕的DFIR方法，该方法结合了来自现场经验和文献的最佳实践。最后，本文回顾了无文件威胁的基本原理，它也可以帮助勒索软件研究人员检查类似的模式。



## **15. Searching for Robust Neural Architectures via Comprehensive and Reliable Evaluation**

通过综合可靠的评估寻找健壮的神经结构 cs.LG

**SubmitDate**: 2022-03-07    [paper-pdf](http://arxiv.org/pdf/2203.03128v1)

**Authors**: Jialiang Sun, Tingsong Jiang, Chao Li, Weien Zhou, Xiaoya Zhang, Wen Yao, Xiaoqian Chen

**Abstracts**: Neural architecture search (NAS) could help search for robust network architectures, where defining robustness evaluation metrics is the important procedure. However, current robustness evaluations in NAS are not sufficiently comprehensive and reliable. In particular, the common practice only considers adversarial noise and quantified metrics such as the Jacobian matrix, whereas, some studies indicated that the models are also vulnerable to other types of noises such as natural noise. In addition, existing methods taking adversarial noise as the evaluation just use the robust accuracy of the FGSM or PGD, but these adversarial attacks could not provide the adequately reliable evaluation, leading to the vulnerability of the models under stronger attacks. To alleviate the above problems, we propose a novel framework, called Auto Adversarial Attack and Defense (AAAD), where we employ neural architecture search methods, and four types of robustness evaluations are considered, including adversarial noise, natural noise, system noise and quantified metrics, thereby assisting in finding more robust architectures. Also, among the adversarial noise, we use the composite adversarial attack obtained by random search as the new metric to evaluate the robustness of the model architectures. The empirical results on the CIFAR10 dataset show that the searched efficient attack could help find more robust architectures.

摘要: 神经体系结构搜索(NAS)可以帮助搜索健壮的网络结构，其中定义健壮性评估度量是重要的步骤。然而，目前NAS中的健壮性评估还不够全面和可靠。特别是，通常的做法只考虑对抗性噪声和量化的度量，如雅可比矩阵，而一些研究表明，模型也容易受到其他类型的噪声，如自然噪声的影响。另外，现有的以对抗性噪声为评价指标的方法仅仅使用了FGSM或PGD的鲁棒准确度，但这些对抗性攻击不能提供足够可靠的评价，导致模型在较强攻击下的脆弱性。为了缓解上述问题，我们提出了一种新的框架，称为自动对抗攻击和防御(AAAD)，其中我们使用了神经结构搜索方法，并考虑了四种类型的健壮性评估，包括对抗噪声、自然噪声、系统噪声和量化度量，从而帮助发现更健壮的体系结构。另外，在敌意噪声中，我们使用随机搜索得到的复合敌意攻击作为新的度量来评估模型体系的健壮性。在CIFAR10数据集上的实验结果表明，搜索到的高效攻击可以帮助发现更健壮的体系结构。



## **16. Protecting Facial Privacy: Generating Adversarial Identity Masks via Style-robust Makeup Transfer**

保护面部隐私：通过风格稳健的化妆传输生成敌意身份面具 cs.CV

Accepted by CVPR2022, NOT the camera-ready version

**SubmitDate**: 2022-03-07    [paper-pdf](http://arxiv.org/pdf/2203.03121v1)

**Authors**: Shengshan Hu, Xiaogeng Liu, Yechao Zhang, Minghui Li, Leo Yu Zhang, Hai Jin, Libing Wu

**Abstracts**: While deep face recognition (FR) systems have shown amazing performance in identification and verification, they also arouse privacy concerns for their excessive surveillance on users, especially for public face images widely spread on social networks. Recently, some studies adopt adversarial examples to protect photos from being identified by unauthorized face recognition systems. However, existing methods of generating adversarial face images suffer from many limitations, such as awkward visual, white-box setting, weak transferability, making them difficult to be applied to protect face privacy in reality. In this paper, we propose adversarial makeup transfer GAN (AMT-GAN), a novel face protection method aiming at constructing adversarial face images that preserve stronger black-box transferability and better visual quality simultaneously. AMT-GAN leverages generative adversarial networks (GAN) to synthesize adversarial face images with makeup transferred from reference images. In particular, we introduce a new regularization module along with a joint training strategy to reconcile the conflicts between the adversarial noises and the cycle consistence loss in makeup transfer, achieving a desirable balance between the attack strength and visual changes. Extensive experiments verify that compared with state of the arts, AMT-GAN can not only preserve a comfortable visual quality, but also achieve a higher attack success rate over commercial FR APIs, including Face++, Aliyun, and Microsoft.

摘要: 深度人脸识别(FR)系统在身份识别和验证方面表现出惊人性能的同时，也因其对用户的过度监控而引起隐私问题，特别是对社交网络上广泛传播的公共人脸图像。近年来，一些研究采用对抗性例子来保护照片不被未经授权的人脸识别系统识别。然而，现有的生成敌意人脸图像的方法存在着视觉效果不佳、白盒设置、可移植性差等诸多局限性，难以应用于现实中的人脸隐私保护。本文提出了一种新的人脸保护方法--对抗性化妆转移GAN(AMT-GAN)，其目的是在构建对抗性人脸图像的同时保持较强的黑盒可传递性和较好的视觉质量。AMT-GAN利用生成性对抗性网络(GAN)来合成带有参考图像化妆的对抗性人脸图像。特别是，我们引入了一种新的正则化模型和联合训练策略来协调化妆转移中对抗性噪声和循环一致性损失之间的冲突，实现了攻击强度和视觉变化之间的理想平衡。大量实验证明，与现有技术相比，AMT-GAN不仅能保持舒适的视觉质量，而且比Face++、阿里云、微软等商用FR API具有更高的攻击成功率。



## **17. Can You Hear It? Backdoor Attacks via Ultrasonic Triggers**

你能听到吗？通过超声波触发器进行后门攻击 cs.CR

**SubmitDate**: 2022-03-06    [paper-pdf](http://arxiv.org/pdf/2107.14569v3)

**Authors**: Stefanos Koffas, Jing Xu, Mauro Conti, Stjepan Picek

**Abstracts**: This work explores backdoor attacks for automatic speech recognition systems where we inject inaudible triggers. By doing so, we make the backdoor attack challenging to detect for legitimate users, and thus, potentially more dangerous. We conduct experiments on two versions of a speech dataset and three neural networks and explore the performance of our attack concerning the duration, position, and type of the trigger. Our results indicate that less than 1% of poisoned data is sufficient to deploy a backdoor attack and reach a 100% attack success rate. We observed that short, non-continuous triggers result in highly successful attacks. However, since our trigger is inaudible, it can be as long as possible without raising any suspicions making the attack more effective. Finally, we conducted our attack in actual hardware and saw that an adversary could manipulate inference in an Android application by playing the inaudible trigger over the air.

摘要: 这项工作探索了自动语音识别系统的后门攻击，在这些系统中，我们注入了可听得见的触发器。通过这样做，我们使后门攻击对于合法用户来说更难检测，因此，潜在地更危险。我们在两个版本的语音数据集和三个神经网络上进行了实验，并探讨了我们的攻击在触发持续时间、位置和类型方面的性能。我们的结果表明，只有不到1%的有毒数据足以部署后门攻击，并且达到100%的攻击成功率。我们观察到短的、非连续的触发器会导致非常成功的攻击。然而，由于我们的扳机是听不见的，它可以尽可能长，而不会引起任何怀疑，从而使攻击更有效。最后，我们在实际的硬件上进行了攻击，看到对手可以通过空中播放听不见的触发器来操纵Android应用程序中的推理。



## **18. An Improved Genetic Algorithm and Its Application in Neural Network Adversarial Attack**

一种改进的遗传算法及其在神经网络攻击中的应用 cs.NE

18 pages, 9 figures, 9 tables and 23 References

**SubmitDate**: 2022-03-06    [paper-pdf](http://arxiv.org/pdf/2110.01818v5)

**Authors**: Dingming Yang, Zeyu Yu, Hongqiang Yuan, Yanrong Cui

**Abstracts**: The choice of crossover and mutation strategies plays a crucial role in the searchability, convergence efficiency and precision of genetic algorithms. In this paper, a novel improved genetic algorithm is proposed by improving the crossover and mutation operation of the simple genetic algorithm, and it is verified by 15 test functions. The qualitative results show that, compared with three other mainstream swarm intelligence optimization algorithms, the algorithm can not only improve the global search ability, convergence efficiency and precision, but also increase the success rate of convergence to the optimal value under the same experimental conditions. The quantitative results show that the algorithm performs superiorly in 13 of the 15 tested functions. The Wilcoxon rank-sum test was used for statistical evaluation, showing the significant advantage of the algorithm at $95\%$ confidence intervals. Finally, the algorithm is applied to neural network adversarial attacks. The applied results show that the method does not need the structure and parameter information inside the neural network model, and it can obtain the adversarial samples with high confidence in a brief time just by the classification and confidence information output from the neural network.

摘要: 交叉和变异策略的选择对遗传算法的可搜索性、收敛效率和精度起着至关重要的作用。通过对简单遗传算法交叉和变异操作的改进，提出了一种新的改进遗传算法，并通过15个测试函数进行了验证。定性结果表明，与其他三种主流群体智能优化算法相比，该算法在相同的实验条件下，不仅提高了全局搜索能力、收敛效率和精度，而且提高了收敛到最优值的成功率。定量结果表明，该算法在15个测试函数中有13个表现较好。采用Wilcoxon秩和检验进行统计评价，结果表明该算法在95美元置信区间内具有显著优势。最后，将该算法应用于神经网络对抗性攻击。应用结果表明，该方法不需要神经网络模型内部的结构和参数信息，仅根据神经网络输出的分类和置信度信息，即可在短时间内获得高置信度的对抗性样本。



## **19. Finding Dynamics Preserving Adversarial Winning Tickets**

寻找动态保存的对抗性中奖彩票 cs.LG

Accepted by AISTATS2022

**SubmitDate**: 2022-03-06    [paper-pdf](http://arxiv.org/pdf/2202.06488v3)

**Authors**: Xupeng Shi, Pengfei Zheng, A. Adam Ding, Yuan Gao, Weizhong Zhang

**Abstracts**: Modern deep neural networks (DNNs) are vulnerable to adversarial attacks and adversarial training has been shown to be a promising method for improving the adversarial robustness of DNNs. Pruning methods have been considered in adversarial context to reduce model capacity and improve adversarial robustness simultaneously in training. Existing adversarial pruning methods generally mimic the classical pruning methods for natural training, which follow the three-stage 'training-pruning-fine-tuning' pipelines. We observe that such pruning methods do not necessarily preserve the dynamics of dense networks, making it potentially hard to be fine-tuned to compensate the accuracy degradation in pruning. Based on recent works of \textit{Neural Tangent Kernel} (NTK), we systematically study the dynamics of adversarial training and prove the existence of trainable sparse sub-network at initialization which can be trained to be adversarial robust from scratch. This theoretically verifies the \textit{lottery ticket hypothesis} in adversarial context and we refer such sub-network structure as \textit{Adversarial Winning Ticket} (AWT). We also show empirical evidences that AWT preserves the dynamics of adversarial training and achieve equal performance as dense adversarial training.

摘要: 现代深层神经网络(DNNs)容易受到敌意攻击，对抗性训练已被证明是提高DNN对抗性鲁棒性的一种很有前途的方法。在训练过程中，考虑了对抗性环境下的剪枝方法，在减少模型容量的同时提高对抗性鲁棒性。现有的对抗性剪枝方法一般是模仿经典的自然训练剪枝方法，遵循“训练-剪枝-微调”三阶段的流水线。我们观察到，这样的剪枝方法并不一定保持密集网络的动态，使得它可能很难被微调来补偿剪枝过程中的精度下降。基于神经切核(NTK)的最新工作，系统地研究了对抗性训练的动力学，证明了在初始化时存在可训练的稀疏子网络，它可以从头开始训练为对抗性健壮性网络。这从理论上验证了对抗性环境下的\text{彩票假设}，我们将这种子网络结构称为\text{对抗性中票}(AWT)。我们还展示了经验证据，AWT保持了对抗性训练的动态性，并获得了与密集对抗性训练相同的性能。



## **20. aaeCAPTCHA: The Design and Implementation of Audio Adversarial CAPTCHA**

AaeCAPTCHA：音频对抗性验证码的设计与实现 cs.CR

Accepted at 7th IEEE European Symposium on Security and Privacy  (EuroS&P 2022)

**SubmitDate**: 2022-03-05    [paper-pdf](http://arxiv.org/pdf/2203.02735v1)

**Authors**: Md Imran Hossen, Xiali Hei

**Abstracts**: CAPTCHAs are designed to prevent malicious bot programs from abusing websites. Most online service providers deploy audio CAPTCHAs as an alternative to text and image CAPTCHAs for visually impaired users. However, prior research investigating the security of audio CAPTCHAs found them highly vulnerable to automated attacks using Automatic Speech Recognition (ASR) systems. To improve the robustness of audio CAPTCHAs against automated abuses, we present the design and implementation of an audio adversarial CAPTCHA (aaeCAPTCHA) system in this paper. The aaeCAPTCHA system exploits audio adversarial examples as CAPTCHAs to prevent the ASR systems from automatically solving them. Furthermore, we conducted a rigorous security evaluation of our new audio CAPTCHA design against five state-of-the-art DNN-based ASR systems and three commercial Speech-to-Text (STT) services. Our experimental evaluations demonstrate that aaeCAPTCHA is highly secure against these speech recognition technologies, even when the attacker has complete knowledge of the current attacks against audio adversarial examples. We also conducted a usability evaluation of the proof-of-concept implementation of the aaeCAPTCHA scheme. Our results show that it achieves high robustness at a moderate usability cost compared to normal audio CAPTCHAs. Finally, our extensive analysis highlights that aaeCAPTCHA can significantly enhance the security and robustness of traditional audio CAPTCHA systems while maintaining similar usability.

摘要: 验证码旨在防止恶意的僵尸程序滥用网站。大多数在线服务提供商为视障用户部署音频验证码作为文本和图像验证码的替代方案。然而，先前调查音频验证码安全性的研究发现，它们非常容易受到使用自动语音识别(ASR)系统的自动攻击。为了提高音频验证码对自动化滥用的健壮性，本文设计并实现了一个音频对抗性验证码系统(AaeCAPTCHA)。aaeCAPTCHA系统利用音频对抗性示例作为验证码来防止ASR系统自动求解它们。此外，我们针对五个基于DNN的最先进的ASR系统和三个商业语音到文本(STT)服务对我们的新音频验证码设计进行了严格的安全评估。我们的实验评估表明，即使攻击者完全了解当前针对音频对抗性示例的攻击，aaeCAPTCHA对这些语音识别技术也是高度安全的。我们还对aaeCAPTCHA方案的概念验证实现进行了可用性评估。实验结果表明，与普通的音频验证码相比，该算法以适中的可用性代价实现了较高的鲁棒性。最后，我们的广泛分析强调，aaeCAPTCHA可以显著增强传统音频验证码系统的安全性和健壮性，同时保持类似的可用性。



## **21. Generating Out of Distribution Adversarial Attack using Latent Space Poisoning**

利用潜在空间毒化产生分布外的敌意攻击 cs.CV

IEEE SPL 2021

**SubmitDate**: 2022-03-05    [paper-pdf](http://arxiv.org/pdf/2012.05027v2)

**Authors**: Ujjwal Upadhyay, Prerana Mukherjee

**Abstracts**: Traditional adversarial attacks rely upon the perturbations generated by gradients from the network which are generally safeguarded by gradient guided search to provide an adversarial counterpart to the network. In this paper, we propose a novel mechanism of generating adversarial examples where the actual image is not corrupted rather its latent space representation is utilized to tamper with the inherent structure of the image while maintaining the perceptual quality intact and to act as legitimate data samples. As opposed to gradient-based attacks, the latent space poisoning exploits the inclination of classifiers to model the independent and identical distribution of the training dataset and tricks it by producing out of distribution samples. We train a disentangled variational autoencoder (beta-VAE) to model the data in latent space and then we add noise perturbations using a class-conditioned distribution function to the latent space under the constraint that it is misclassified to the target label. Our empirical results on MNIST, SVHN, and CelebA dataset validate that the generated adversarial examples can easily fool robust l_0, l_2, l_inf norm classifiers designed using provably robust defense mechanisms.

摘要: 传统的对抗性攻击依赖于由来自网络的梯度产生的扰动，这些扰动通常由梯度引导搜索来保护，以提供网络的对应物。在本文中，我们提出了一种新的生成敌意示例的机制，其中实际图像没有被破坏，而是利用其潜在空间表示来篡改图像的内在结构，同时保持感知质量不变，并作为合法的数据样本。与基于梯度的攻击不同，潜在空间中毒利用分类器的倾向性来建模训练数据集的独立且相同的分布，并通过产生分布外的样本来欺骗它。我们训练一个解缠变分自动编码器(β-VAE)来对潜在空间中的数据建模，然后在误分类为目标标签的约束下，使用分类条件分布函数向潜在空间添加噪声扰动。我们在MNIST、SVHN和CelebA数据集上的实验结果验证了所生成的敌意示例可以很容易地欺骗使用可证明鲁棒防御机制设计的鲁棒l_0、l_2、l_inf范数分类器。



## **22. Adversarial samples for deep monocular 6D object pose estimation**

用于深部单目6维目标姿态估计的对抗性样本 cs.CV

15 pages

**SubmitDate**: 2022-03-05    [paper-pdf](http://arxiv.org/pdf/2203.00302v2)

**Authors**: Jinlai Zhang, Weiming Li, Shuang Liang, Hao Wang, Jihong Zhu

**Abstracts**: Estimating 6D object pose from an RGB image is important for many real-world applications such as autonomous driving and robotic grasping. Recent deep learning models have achieved significant progress on this task but their robustness received little research attention. In this work, for the first time, we study adversarial samples that can fool deep learning models with imperceptible perturbations to input image. In particular, we propose a Unified 6D pose estimation Attack, namely U6DA, which can successfully attack several state-of-the-art (SOTA) deep learning models for 6D pose estimation. The key idea of our U6DA is to fool the models to predict wrong results for object instance localization and shape that are essential for correct 6D pose estimation. Specifically, we explore a transfer-based black-box attack to 6D pose estimation. We design the U6DA loss to guide the generation of adversarial examples, the loss aims to shift the segmentation attention map away from its original position. We show that the generated adversarial samples are not only effective for direct 6D pose estimation models, but also are able to attack two-stage models regardless of their robust RANSAC modules. Extensive experiments were conducted to demonstrate the effectiveness, transferability, and anti-defense capability of our U6DA on large-scale public benchmarks. We also introduce a new U6DA-Linemod dataset for robustness study of the 6D pose estimation task. Our codes and dataset will be available at \url{https://github.com/cuge1995/U6DA}.

摘要: 从RGB图像估计6D物体姿态对于许多现实世界的应用非常重要，例如自动驾驶和机器人抓取。最近的深度学习模型在这方面已经取得了显着的进展，但是它们的鲁棒性却没有得到足够的研究。在这项工作中，我们首次研究了可以欺骗深度学习模型的对抗性样本，并对输入图像进行了潜移默化的扰动。特别地，我们提出了一种统一的6D位姿估计攻击，即U6DA，它可以成功地攻击几种用于6D位姿估计的SOTA深度学习模型。我们的U6DA的关键思想是愚弄模型来预测错误的对象实例定位和形状结果，这是正确的6D姿势估计所必需的。具体地说，我们探索了一种基于传输的黑盒攻击来进行6D位姿估计。我们设计了U6DA丢失来指导对抗性示例的生成，该丢失的目的是将分割注意图从原来的位置移开。结果表明，生成的对抗性样本不仅对直接6D姿态估计模型有效，而且无论其RANSAC模型是否具有鲁棒性，都能够攻击两阶段模型。在大型公共基准上进行了广泛的实验，以验证我们的U6DA的有效性、可转移性和反防御能力。我们还介绍了一个新的U6DA-Linemod数据集，用于6D位姿估计任务的鲁棒性研究。我们的代码和数据集将在\url{https://github.com/cuge1995/U6DA}.



## **23. Training privacy-preserving video analytics pipelines by suppressing features that reveal information about private attributes**

通过抑制泄露关于私有属性的信息的特征来训练保护隐私的视频分析管道 cs.CV

**SubmitDate**: 2022-03-05    [paper-pdf](http://arxiv.org/pdf/2203.02635v1)

**Authors**: Chau Yi Li, Andrea Cavallaro

**Abstracts**: Deep neural networks are increasingly deployed for scene analytics, including to evaluate the attention and reaction of people exposed to out-of-home advertisements. However, the features extracted by a deep neural network that was trained to predict a specific, consensual attribute (e.g. emotion) may also encode and thus reveal information about private, protected attributes (e.g. age or gender). In this work, we focus on such leakage of private information at inference time. We consider an adversary with access to the features extracted by the layers of a deployed neural network and use these features to predict private attributes. To prevent the success of such an attack, we modify the training of the network using a confusion loss that encourages the extraction of features that make it difficult for the adversary to accurately predict private attributes. We validate this training approach on image-based tasks using a publicly available dataset. Results show that, compared to the original network, the proposed PrivateNet can reduce the leakage of private information of a state-of-the-art emotion recognition classifier by 2.88% for gender and by 13.06% for age group, with a minimal effect on task accuracy.

摘要: 深度神经网络越来越多地用于场景分析，包括评估接触户外广告的人的注意力和反应。然而，由被训练来预测特定的、一致的属性(例如，情感)的深度神经网络提取的特征也可以编码并且因此揭示关于私人的、受保护的属性(例如，年龄或性别)的信息。在这项工作中，我们关注的是推理时隐私信息的泄露。我们考虑一个可以访问由部署的神经网络的各层提取的特征的对手，并使用这些特征来预测私有属性。为了防止此类攻击成功，我们使用念力损失修改了网络的训练，该损失鼓励提取使对手难以准确预测私人属性的特征。我们使用公开可用的数据集在基于图像的任务上验证了这种训练方法。实验结果表明，与原网络相比，本文提出的PrivateNet能够将最先进的情感识别分类器的隐私信息泄露减少2.88%(性别)和13.06%(年龄组)，并且对任务准确率的影响最小。



## **24. Optimal Clock Synchronization with Signatures**

带签名的最佳时钟同步 cs.DC

**SubmitDate**: 2022-03-04    [paper-pdf](http://arxiv.org/pdf/2203.02553v1)

**Authors**: Christoph Lenzen, Julian Loss

**Abstracts**: Cryptographic signatures can be used to increase the resilience of distributed systems against adversarial attacks, by increasing the number of faulty parties that can be tolerated. While this is well-studied for consensus, it has been underexplored in the context of fault-tolerant clock synchronization, even in fully connected systems. Here, the honest parties of an $n$-node system are required to compute output clocks of small skew (i.e., maximum phase offset) despite local clock rates varying between $1$ and $\vartheta>1$, end-to-end communication delays varying between $d-u$ and $d$, and the interference from malicious parties. So far, it is only known that clock pulses of skew $d$ can be generated with (trivially optimal) resilience of $\lceil n/2\rceil-1$ (PODC `19), improving over the tight bound of $\lceil n/3\rceil-1$ holding without signatures for \emph{any} skew bound (STOC `84, PODC `85). Since typically $d\gg u$ and $\vartheta-1\ll 1$, this is far from the lower bound of $u+(\vartheta-1)d$ that applies even in the fault-free case (IPL `01).   We prove matching upper and lower bounds of $\Theta(u+(\vartheta-1)d)$ on the skew for the resilience range from $\lceil n/3\rceil$ to $\lceil n/2\rceil-1$. The algorithm showing the upper bound is, under the assumption that the adversary cannot forge signatures, deterministic. The lower bound holds even if clocks are initially perfectly synchronized, message delays between honest nodes are known, $\vartheta$ is arbitrarily close to one, and the synchronization algorithm is randomized. This has crucial implications for network designers that seek to leverage signatures for providing more robust time. In contrast to the setting without signatures, they must ensure that an attacker cannot easily bypass the lower bound on the delay on links with a faulty endpoint.

摘要: 通过增加可容忍的故障方的数量，可以使用密码签名来提高分布式系统对敌意攻击的恢复能力。虽然这是为了达成共识而研究得很好的，但在容错时钟同步的背景下，甚至在完全连接的系统中，这一点也没有得到充分的探索。这里，尽管本地时钟速率在$1$和$\vartheta>1$之间变化，端到端通信延迟在$d-u$和$d$之间变化，以及来自恶意方的干扰，但$n$节点系统的诚实方被要求计算小偏差(即最大相位偏移)的输出时钟。到目前为止，只知道歪斜$d$的时钟脉冲可以产生$\lceil n/2\rceil$(PODC`19)的弹性(最优)，改进了没有签名的$\lceil n/3\rceil$保持的紧凑界限(STEC`84，PODC‘85)。由于通常为$d\gg u$和$\vartheta-1\ll 1$，即使在无故障的情况下(IPL‘01)，这也远不是适用于$u+(\vartheta-1)d$的下限。我们证明了$theta(u+(vartheta-1)d)$在$lceil n/3\rceil$到$lceil n/2\rceil-1$的斜斜度上的上下界是匹配的。在假设对手不能伪造签名的情况下，给出上限的算法是确定性的。即使时钟最初是完全同步的，诚实节点之间的消息延迟是已知的，$\vartheta$任意接近于1，并且同步算法是随机的，这个下限也是成立的。这对于寻求利用签名来提供更可靠时间的网络设计人员具有重要意义。与没有签名的设置相比，它们必须确保攻击者不能轻松绕过具有故障端点的链路的延迟下限。



## **25. Medical Aegis: Robust adversarial protectors for medical images**

医学宙斯盾：强大的医学图像对抗性保护者 cs.CV

**SubmitDate**: 2022-03-04    [paper-pdf](http://arxiv.org/pdf/2111.10969v4)

**Authors**: Qingsong Yao, Zecheng He, S. Kevin Zhou

**Abstracts**: Deep neural network based medical image systems are vulnerable to adversarial examples. Many defense mechanisms have been proposed in the literature, however, the existing defenses assume a passive attacker who knows little about the defense system and does not change the attack strategy according to the defense. Recent works have shown that a strong adaptive attack, where an attacker is assumed to have full knowledge about the defense system, can easily bypass the existing defenses. In this paper, we propose a novel adversarial example defense system called Medical Aegis. To the best of our knowledge, Medical Aegis is the first defense in the literature that successfully addresses the strong adaptive adversarial example attacks to medical images. Medical Aegis boasts two-tier protectors: The first tier of Cushion weakens the adversarial manipulation capability of an attack by removing its high-frequency components, yet posing a minimal effect on classification performance of the original image; the second tier of Shield learns a set of per-class DNN models to predict the logits of the protected model. Deviation from the Shield's prediction indicates adversarial examples. Shield is inspired by the observations in our stress tests that there exist robust trails in the shallow layers of a DNN model, which the adaptive attacks can hardly destruct. Experimental results show that the proposed defense accurately detects adaptive attacks, with negligible overhead for model inference.

摘要: 基于深度神经网络的医学图像系统容易受到敌意例子的攻击。文献中提出了很多防御机制，但现有的防御机制都假设攻击者是被动的，对防御系统知之甚少，不会根据防御情况改变攻击策略。最近的研究表明，强自适应攻击(假设攻击者完全了解防御系统)可以很容易地绕过现有的防御系统。在本文中，我们提出了一种新的对抗性实例防御系统--医学宙斯盾(Medical Aegis)。据我们所知，医学宙斯盾是文献中第一个成功解决了对医学图像的强适应性敌意攻击的防御方案。医学宙斯盾拥有两层保护器：第一层缓冲通过移除攻击的高频分量来削弱攻击的敌意操纵能力，但对原始图像的分类性能影响最小；第二层Shield学习一组按类的DNN模型来预测受保护模型的逻辑。与盾牌的预测背道而驰表明了敌对的例子。Shield的灵感来自于我们在压力测试中观察到的DNN模型的浅层存在健壮的踪迹，而自适应攻击很难破坏这些踪迹。实验结果表明，该防御方法能够准确检测出自适应攻击，而模型推理开销可以忽略不计。



## **26. Adversarial Patterns: Building Robust Android Malware Classifiers**

对抗性模式：构建健壮的Android恶意软件分类器 cs.CR

**SubmitDate**: 2022-03-04    [paper-pdf](http://arxiv.org/pdf/2203.02121v1)

**Authors**: Dipkamal Bhusal, Nidhi Rastogi

**Abstracts**: Deep learning-based classifiers have substantially improved recognition of malware samples. However, these classifiers can be vulnerable to adversarial input perturbations. Any vulnerability in malware classifiers poses significant threats to the platforms they defend. Therefore, to create stronger defense models against malware, we must understand the patterns in input perturbations caused by an adversary. This survey paper presents a comprehensive study on adversarial machine learning for android malware classifiers. We first present an extensive background in building a machine learning classifier for android malware, covering both image-based and text-based feature extraction approaches. Then, we examine the pattern and advancements in the state-of-the-art research in evasion attacks and defenses. Finally, we present guidelines for designing robust malware classifiers and enlist research directions for the future.

摘要: 基于深度学习的分类器大大提高了恶意软件样本的识别能力。然而，这些分类器可能容易受到对抗性输入扰动的影响。恶意软件分类器中的任何漏洞都会对其防御的平台构成重大威胁。因此，要创建针对恶意软件的更强大的防御模型，我们必须了解由对手造成的输入扰动的模式。本文对Android恶意软件分类器的对抗性机器学习进行了全面的研究。我们首先介绍了构建Android恶意软件机器学习分类器的广泛背景，包括基于图像和基于文本的特征提取方法。然后，我们考察了在躲避攻击和防御方面的最新研究模式和进展。最后，我们提出了设计健壮的恶意软件分类器的指导原则，并提出了未来的研究方向。



## **27. Label Leakage and Protection from Forward Embedding in Vertical Federated Learning**

垂直联合学习中的标签泄漏与前向嵌入保护 cs.LG

**SubmitDate**: 2022-03-04    [paper-pdf](http://arxiv.org/pdf/2203.01451v2)

**Authors**: Jiankai Sun, Xin Yang, Yuanshun Yao, Chong Wang

**Abstracts**: Vertical federated learning (vFL) has gained much attention and been deployed to solve machine learning problems with data privacy concerns in recent years. However, some recent work demonstrated that vFL is vulnerable to privacy leakage even though only the forward intermediate embedding (rather than raw features) and backpropagated gradients (rather than raw labels) are communicated between the involved participants. As the raw labels often contain highly sensitive information, some recent work has been proposed to prevent the label leakage from the backpropagated gradients effectively in vFL. However, these work only identified and defended the threat of label leakage from the backpropagated gradients. None of these work has paid attention to the problem of label leakage from the intermediate embedding. In this paper, we propose a practical label inference method which can steal private labels effectively from the shared intermediate embedding even though some existing protection methods such as label differential privacy and gradients perturbation are applied. The effectiveness of the label attack is inseparable from the correlation between the intermediate embedding and corresponding private labels. To mitigate the issue of label leakage from the forward embedding, we add an additional optimization goal at the label party to limit the label stealing ability of the adversary by minimizing the distance correlation between the intermediate embedding and corresponding private labels. We conducted massive experiments to demonstrate the effectiveness of our proposed protection methods.

摘要: 垂直联合学习(VFL)近年来得到了广泛的关注，并被应用于解决数据隐私问题中的机器学习问题。然而，最近的一些工作表明，即使参与者之间只传递前向中间嵌入(而不是原始特征)和反向传播梯度(而不是原始标签)，VFL也容易受到隐私泄露的影响。由于原始标签往往包含高度敏感的信息，最近已有一些工作被提出以有效地防止VFL中反向传播梯度引起的标签泄漏。然而，这些工作仅仅识别和防御了反向传播梯度带来的标签泄漏威胁。这些工作都没有注意到中间嵌入带来的标签泄漏问题。本文提出了一种实用的标签推理方法，即使采用了标签差分隐私、梯度扰动等保护方法，也能有效地从共享中间嵌入中窃取私有标签。标签攻击的有效性离不开中间嵌入与对应的私有标签之间的关联。为了缓解前向嵌入带来的标签泄漏问题，我们在标签方增加了一个额外的优化目标，通过最小化中间嵌入与相应私有标签之间的距离相关性来限制敌手的标签窃取能力。我们进行了大量的实验来验证我们提出的保护方法的有效性。



## **28. Differentially Private Label Protection in Split Learning**

分裂学习中的差异化私有标签保护 cs.LG

**SubmitDate**: 2022-03-04    [paper-pdf](http://arxiv.org/pdf/2203.02073v1)

**Authors**: Xin Yang, Jiankai Sun, Yuanshun Yao, Junyuan Xie, Chong Wang

**Abstracts**: Split learning is a distributed training framework that allows multiple parties to jointly train a machine learning model over vertically partitioned data (partitioned by attributes). The idea is that only intermediate computation results, rather than private features and labels, are shared between parties so that raw training data remains private. Nevertheless, recent works showed that the plaintext implementation of split learning suffers from severe privacy risks that a semi-honest adversary can easily reconstruct labels. In this work, we propose \textsf{TPSL} (Transcript Private Split Learning), a generic gradient perturbation based split learning framework that provides provable differential privacy guarantee. Differential privacy is enforced on not only the model weights, but also the communicated messages in the distributed computation setting. Our experiments on large-scale real-world datasets demonstrate the robustness and effectiveness of \textsf{TPSL} against label leakage attacks. We also find that \textsf{TPSL} have a better utility-privacy trade-off than baselines.

摘要: 分裂学习是一种分布式训练框架，允许多方在垂直划分的数据(按属性划分)上联合训练机器学习模型。其想法是，各方之间只共享中间计算结果，而不是私有特征和标签，因此原始训练数据保持私有。然而，最近的研究表明，分裂学习的明文实现存在严重的隐私风险，半诚实的对手可以很容易地重构标签。在这项工作中，我们提出了一个通用的基于梯度扰动的分裂学习框架--textsf{TPSL}(Transcript Private Split Learning)，它提供了可证明的差分隐私保证。在分布式计算环境中，不仅对模型权重，而且对通信消息实施差分隐私。我们在大规模真实数据集上的实验证明了Textsf{TPSL}对标签泄漏攻击的鲁棒性和有效性。我们还发现，\textsf{tpsl}比基线有更好的效用-隐私权衡。



## **29. Can Authoritative Governments Abuse the Right to Access?**

权威政府会滥用访问权吗？ cs.CR

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2203.02068v1)

**Authors**: Cédric Lauradoux

**Abstracts**: The right to access is a great tool provided by the GDPR to empower data subjects with their data. However, it needs to be implemented properly otherwise it could turn subject access requests against the subjects privacy. Indeed, recent works have shown that it is possible to abuse the right to access using impersonation attacks. We propose to extend those impersonation attacks by considering that the adversary has an access to governmental resources. In this case, the adversary can forge official documents or exploit copy of them. Our attack affects more people than one may expect. To defeat the attacks from this kind of adversary, several solutions are available like multi-factors or proof of aliveness. Our attacks highlight the need for strong procedures to authenticate subject access requests.

摘要: 访问权是GDPR提供的一个很好的工具，用来赋予数据主体数据权力。但是，它需要正确实现，否则可能会使主体访问请求与主体隐私背道而驰。事实上，最近的研究表明，使用冒充攻击来滥用访问权限是可能的。我们建议扩展这些冒充攻击，因为考虑到对手可以访问政府资源。在这种情况下，对手可以伪造官方文件或利用其副本。我们的袭击影响的人比人们预料的要多。要击败这类对手的攻击，有几种解决方案可用，比如多因素或活性证明。我们的攻击突显了需要强大的程序来验证主体访问请求。



## **30. Autonomous and Resilient Control for Optimal LEO Satellite Constellation Coverage Against Space Threats**

空间威胁下LEO卫星星座最优覆盖的自主弹性控制 eess.SY

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2203.02050v1)

**Authors**: Yuhan Zhao, Quanyan Zhu

**Abstracts**: LEO satellite constellation coverage has served as the base platform for various space applications. However, the rapidly evolving security environment such as orbit debris and adversarial space threats are greatly endangering the security of satellite constellation and integrity of the satellite constellation coverage. As on-orbit repairs are challenging, a distributed and autonomous protection mechanism is necessary to ensure the adaptation and self-healing of the satellite constellation coverage from different attacks. To this end, we establish an integrative and distributed framework to enable resilient satellite constellation coverage planning and control in a single orbit. Each satellite can make decisions individually to recover from adversarial and non-adversarial attacks and keep providing coverage service. We first provide models and methodologies to measure the coverage performance. Then, we formulate the joint resilient coverage planning-control problem as a two-stage problem. A coverage game is proposed to find the equilibrium constellation deployment for resilient coverage planning and an agent-based algorithm is developed to compute the equilibrium. The multi-waypoint Model Predictive Control (MPC) methodology is adopted to achieve autonomous self-healing control. Finally, we use a typical LEO satellite constellation as a case study to corroborate the results.

摘要: 低轨卫星星座复盖已成为各种空间应用的基础平台。然而，快速发展的轨道碎片和对抗性空间威胁等安全环境极大地威胁着卫星星座的安全性和卫星星座覆盖的完整性。由于在轨修复具有挑战性，需要一种分布式、自主的保护机制来确保卫星星座覆盖在不同攻击下的适应性和自愈性。为此，我们建立了一个一体化和分布式的框架，使弹性卫星星座覆盖规划和控制能够在单一轨道上进行。每颗卫星都可以单独做出决定，从对抗性和非对抗性攻击中恢复，并继续提供覆盖服务。我们首先提供了衡量覆盖性能的模型和方法。然后，我们将联合弹性覆盖规划-控制问题描述为一个两阶段问题。提出了一种基于覆盖博弈的弹性覆盖规划均衡星座部署方法，并提出了一种基于Agent的均衡算法。采用多路点模型预测控制(MPC)方法实现自主自愈控制。最后，以一个典型的低轨卫星星座为例进行了验证。



## **31. Why adversarial training can hurt robust accuracy**

为什么对抗性训练会损害稳健的准确性 cs.LG

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2203.02006v1)

**Authors**: Jacob Clarysse, Julia Hörmann, Fanny Yang

**Abstracts**: Machine learning classifiers with high test accuracy often perform poorly under adversarial attacks. It is commonly believed that adversarial training alleviates this issue. In this paper, we demonstrate that, surprisingly, the opposite may be true -- Even though adversarial training helps when enough data is available, it may hurt robust generalization in the small sample size regime. We first prove this phenomenon for a high-dimensional linear classification setting with noiseless observations. Our proof provides explanatory insights that may also transfer to feature learning models. Further, we observe in experiments on standard image datasets that the same behavior occurs for perceptible attacks that effectively reduce class information such as mask attacks and object corruptions.

摘要: 测试精度高的机器学习分类器在敌意攻击下往往表现不佳。一般认为对抗性训练可以缓解这个问题。在这篇文章中，我们证明，令人惊讶的是，相反的情况可能是真的--尽管对抗性训练在足够的数据可用时有所帮助，但它可能会损害小样本制度下的鲁棒泛化。我们首先在高维线性分类环境中用无声观测证明了这一现象。我们的证明提供了解释性的见解，也可以转移到特征学习模型中。此外，我们在标准图像数据集上的实验中观察到，对于可感知的攻击，也会出现同样的行为，这些攻击有效地减少了掩码攻击和对象损坏等类别信息。



## **32. Dynamic Backdoor Attacks Against Machine Learning Models**

针对机器学习模型的动态后门攻击 cs.CR

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2003.03675v2)

**Authors**: Ahmed Salem, Rui Wen, Michael Backes, Shiqing Ma, Yang Zhang

**Abstracts**: Machine learning (ML) has made tremendous progress during the past decade and is being adopted in various critical real-world applications. However, recent research has shown that ML models are vulnerable to multiple security and privacy attacks. In particular, backdoor attacks against ML models have recently raised a lot of awareness. A successful backdoor attack can cause severe consequences, such as allowing an adversary to bypass critical authentication systems.   Current backdooring techniques rely on adding static triggers (with fixed patterns and locations) on ML model inputs which are prone to detection by the current backdoor detection mechanisms. In this paper, we propose the first class of dynamic backdooring techniques against deep neural networks (DNN), namely Random Backdoor, Backdoor Generating Network (BaN), and conditional Backdoor Generating Network (c-BaN). Triggers generated by our techniques can have random patterns and locations, which reduce the efficacy of the current backdoor detection mechanisms. In particular, BaN and c-BaN based on a novel generative network are the first two schemes that algorithmically generate triggers. Moreover, c-BaN is the first conditional backdooring technique that given a target label, it can generate a target-specific trigger. Both BaN and c-BaN are essentially a general framework which renders the adversary the flexibility for further customizing backdoor attacks.   We extensively evaluate our techniques on three benchmark datasets: MNIST, CelebA, and CIFAR-10. Our techniques achieve almost perfect attack performance on backdoored data with a negligible utility loss. We further show that our techniques can bypass current state-of-the-art defense mechanisms against backdoor attacks, including ABS, Februus, MNTD, Neural Cleanse, and STRIP.

摘要: 机器学习(ML)在过去的十年中取得了巨大的进步，并被应用于各种关键的现实世界应用中。然而，最近的研究表明，ML模型容易受到多种安全和隐私攻击。特别值得一提的是，针对ML模型的后门攻击最近引起了很多关注。成功的后门攻击可能导致严重后果，例如允许攻击者绕过关键身份验证系统。当前的回溯技术依赖于在ML模型输入上添加静电触发器(具有固定的模式和位置)，这容易被当前的后门检测机制检测到。本文提出了针对深度神经网络(DNN)的第一类动态回溯技术，即随机后门、后门生成网络(BAN)和条件后门生成网络(C-BAN)。我们的技术生成的触发器可能具有随机的模式和位置，这降低了当前后门检测机制的效率。特别地，基于新型产生式网络的BAN和C-BAN是算法上生成触发器的前两个方案。此外，C-BAN是第一种在给定目标标签的情况下可以生成特定于目标的触发器的条件回溯技术。BAND和C-BAN本质上都是一个通用框架，为对手提供了进一步定制后门攻击的灵活性。我们在三个基准数据集上广泛评估了我们的技术：MNIST、CelebA和CIFAR-10。我们的技术在几乎可以忽略效用损失的情况下实现了对后置数据的近乎完美的攻击性能。我们进一步表明，我们的技术可以绕过当前最先进的后门攻击防御机制，包括ABS、Februus、MNTD、NeuroCleanse和STRINE。



## **33. Assessing the Robustness of Visual Question Answering Models**

视觉问答模型的稳健性评估 cs.CV

24 pages, 13 figures, International Journal of Computer Vision (IJCV)  [under review]. arXiv admin note: substantial text overlap with  arXiv:1711.06232, arXiv:1709.04625

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/1912.01452v2)

**Authors**: Jia-Hong Huang, Modar Alfadly, Bernard Ghanem, Marcel Worring

**Abstracts**: Deep neural networks have been playing an essential role in the task of Visual Question Answering (VQA). Until recently, their accuracy has been the main focus of research. Now there is a trend toward assessing the robustness of these models against adversarial attacks by evaluating the accuracy of these models under increasing levels of noisiness in the inputs of VQA models. In VQA, the attack can target the image and/or the proposed query question, dubbed main question, and yet there is a lack of proper analysis of this aspect of VQA. In this work, we propose a new method that uses semantically related questions, dubbed basic questions, acting as noise to evaluate the robustness of VQA models. We hypothesize that as the similarity of a basic question to the main question decreases, the level of noise increases. To generate a reasonable noise level for a given main question, we rank a pool of basic questions based on their similarity with this main question. We cast this ranking problem as a LASSO optimization problem. We also propose a novel robustness measure Rscore and two large-scale basic question datasets in order to standardize robustness analysis of VQA models. The experimental results demonstrate that the proposed evaluation method is able to effectively analyze the robustness of VQA models. To foster the VQA research, we will publish our proposed datasets.

摘要: 深度神经网络在视觉问答(VQA)中起着至关重要的作用。直到最近，它们的准确性一直是研究的主要焦点。现在有一种趋势是通过评估在VQA模型输入的噪声水平增加时这些模型的准确性来评估这些模型对敌方攻击的鲁棒性。在VQA中，攻击可以以图像和/或建议的查询问题(称为主问题)为目标，但缺乏对VQA这一方面的适当分析。在这项工作中，我们提出了一种新的方法，使用语义相关的问题，称为基本问题，作为噪声来评估VQA模型的稳健性。我们假设，随着基本问题与主要问题的相似度降低，噪音水平就会增加。为了为给定的主要问题生成合理的噪声水平，我们根据基本问题与该主要问题的相似性对基本问题池进行排名。我们把这个排序问题归结为套索优化问题。为了规范VQA模型的健壮性分析，我们还提出了一种新的健壮性度量RSCORE和两个大规模的基本问题数据集。实验结果表明，该评估方法能够有效地分析VQA模型的鲁棒性。为了促进VQA研究，我们将公布我们建议的数据集。



## **34. Detection of Word Adversarial Examples in Text Classification: Benchmark and Baseline via Robust Density Estimation**

文本分类中词语对抗性实例的检测：基于鲁棒密度估计的基准和基线 cs.CL

Findings of ACL 2022

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2203.01677v1)

**Authors**: KiYoon Yoo, Jangho Kim, Jiho Jang, Nojun Kwak

**Abstracts**: Word-level adversarial attacks have shown success in NLP models, drastically decreasing the performance of transformer-based models in recent years. As a countermeasure, adversarial defense has been explored, but relatively few efforts have been made to detect adversarial examples. However, detecting adversarial examples may be crucial for automated tasks (e.g. review sentiment analysis) that wish to amass information about a certain population and additionally be a step towards a robust defense system. To this end, we release a dataset for four popular attack methods on four datasets and four models to encourage further research in this field. Along with it, we propose a competitive baseline based on density estimation that has the highest AUC on 29 out of 30 dataset-attack-model combinations. Source code is available in https://github.com/anoymous92874838/text-adv-detection.

摘要: 词级敌意攻击在NLP模型中已显示出成功，近年来极大地降低了基于变压器的模型的性能。作为一种对策，对抗性防御已经被探索，但对发现对抗性例子的努力相对较少。然而，检测敌意的例子对于自动化任务(例如，评论情绪分析)可能是至关重要的，这些自动化任务希望收集关于特定人群的信息，并且另外是迈向健壮防御系统的一步。为此，我们在四个数据集和四个模型上发布了四种流行攻击方法的数据集，以鼓励该领域的进一步研究。在此基础上，我们提出了一个基于密度估计的好胜基线，该基线在30个数据集攻击模型组合中的29个组合上具有最高的AUC值。源代码在https://github.com/anoymous92874838/text-adv-detection.中提供



## **35. On Improving Adversarial Transferability of Vision Transformers**

关于提高视觉变形金刚对抗性转换性的探讨 cs.CV

ICLR'22 (Spotlight), the first two authors contributed equally. Code:  https://t.ly/hBbW

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2106.04169v3)

**Authors**: Muzammal Naseer, Kanchana Ranasinghe, Salman Khan, Fahad Shahbaz Khan, Fatih Porikli

**Abstracts**: Vision transformers (ViTs) process input images as sequences of patches via self-attention; a radically different architecture than convolutional neural networks (CNNs). This makes it interesting to study the adversarial feature space of ViT models and their transferability. In particular, we observe that adversarial patterns found via conventional adversarial attacks show very \emph{low} black-box transferability even for large ViT models. We show that this phenomenon is only due to the sub-optimal attack procedures that do not leverage the true representation potential of ViTs. A deep ViT is composed of multiple blocks, with a consistent architecture comprising of self-attention and feed-forward layers, where each block is capable of independently producing a class token. Formulating an attack using only the last class token (conventional approach) does not directly leverage the discriminative information stored in the earlier tokens, leading to poor adversarial transferability of ViTs. Using the compositional nature of ViT models, we enhance transferability of existing attacks by introducing two novel strategies specific to the architecture of ViT models. (i) Self-Ensemble: We propose a method to find multiple discriminative pathways by dissecting a single ViT model into an ensemble of networks. This allows explicitly utilizing class-specific information at each ViT block. (ii) Token Refinement: We then propose to refine the tokens to further enhance the discriminative capacity at each block of ViT. Our token refinement systematically combines the class tokens with structural information preserved within the patch tokens.

摘要: 视觉转换器(VITS)通过自我注意将输入图像处理为补丁序列；这是一种与卷积神经网络(CNN)完全不同的体系结构。这使得研究VIT模型的对抗性特征空间及其可移植性变得很有意义。特别地，我们观察到，即使对于大型VIT模型，通过传统的对抗性攻击发现的对抗性模式也表现出非常低的黑箱可转移性。我们表明，这种现象仅仅是由于次优攻击过程没有充分利用VITS的真实表现潜力所致。深度VIT由多个块组成，具有一致的架构，由自观层和前馈层组成，每个挡路可以独立生成一个类Token。仅使用最后一个类令牌(传统方法)来制定攻击没有直接利用存储在较早令牌中的区别性信息，从而导致VITS的对抗性差的可转移性。利用VIT模型的组合特性，通过引入两种针对VIT模型体系结构的新策略，增强了现有攻击的可转移性。(I)自集成：我们提出了一种通过将单个VIT模型分解成一个网络集成来寻找多条区分路径的方法。这允许在每个VIT挡路上显式地利用特定于类的信息。(Ii)优化代币：然后，我们建议优化代币，以进一步增强每个挡路的识别能力。我们的令牌精化将类令牌与保存在补丁令牌中的结构信息系统地结合在一起。



## **36. On Robustness of Neural Ordinary Differential Equations**

关于神经常微分方程的稳健性 cs.LG

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/1910.05513v4)

**Authors**: Hanshu Yan, Jiawei Du, Vincent Y. F. Tan, Jiashi Feng

**Abstracts**: Neural ordinary differential equations (ODEs) have been attracting increasing attention in various research domains recently. There have been some works studying optimization issues and approximation capabilities of neural ODEs, but their robustness is still yet unclear. In this work, we fill this important gap by exploring robustness properties of neural ODEs both empirically and theoretically. We first present an empirical study on the robustness of the neural ODE-based networks (ODENets) by exposing them to inputs with various types of perturbations and subsequently investigating the changes of the corresponding outputs. In contrast to conventional convolutional neural networks (CNNs), we find that the ODENets are more robust against both random Gaussian perturbations and adversarial attack examples. We then provide an insightful understanding of this phenomenon by exploiting a certain desirable property of the flow of a continuous-time ODE, namely that integral curves are non-intersecting. Our work suggests that, due to their intrinsic robustness, it is promising to use neural ODEs as a basic block for building robust deep network models. To further enhance the robustness of vanilla neural ODEs, we propose the time-invariant steady neural ODE (TisODE), which regularizes the flow on perturbed data via the time-invariant property and the imposition of a steady-state constraint. We show that the TisODE method outperforms vanilla neural ODEs and also can work in conjunction with other state-of-the-art architectural methods to build more robust deep networks.

摘要: 近年来，神经常微分方程(ODE)在各个研究领域受到越来越多的关注。已有一些研究神经常微分方程的优化问题和逼近能力的工作，但其鲁棒性尚不清楚。在这项工作中，我们通过从经验和理论上探索神经ODE的稳健性来填补这一重要空白。我们首先对基于ODENET的神经网络(ODENet)的鲁棒性进行了实证研究，方法是将ODENet暴露在具有各种类型扰动的输入中，然后研究相应输出的变化。与传统的卷积神经网络(CNNs)相比，我们发现ODENet对随机高斯扰动和敌意攻击示例都具有更强的鲁棒性。然后，我们通过利用连续时间颂歌的流的某些理想性质，即积分曲线是不相交的，来提供对这一现象的深刻理解。我们的工作表明，由于其固有的鲁棒性，使用神经ODE作为构建鲁棒深层网络模型的基础挡路是很有前途的。为了进一步增强香草神经微分方程组的鲁棒性，我们提出了时不变稳态神经微分方程组(TisODE)，它通过时不变性和施加稳态约束来规则化扰动数据上的流动。我们表明，TisODE方法的性能优于香草神经ODE方法，并且还可以与其他最先进的体系结构方法相结合来构建更健壮的深层网络。



## **37. Authentication Attacks on Projection-based Cancelable Biometric Schemes**

对基于投影的可取消生物特征识别方案的认证攻击 cs.CR

arXiv admin note: text overlap with arXiv:1910.01389 by other authors

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2110.15163v2)

**Authors**: Axel Durbet, Pascal Lafourcade, Denis Migdal, Kevin Thiry-Atighehchi, Paul-Marie Grollemund

**Abstracts**: Cancelable biometric schemes aim at generating secure biometric templates by combining user specific tokens, such as password, stored secret or salt, along with biometric data. This type of transformation is constructed as a composition of a biometric transformation with a feature extraction algorithm. The security requirements of cancelable biometric schemes concern the irreversibility, unlinkability and revocability of templates, without losing in accuracy of comparison. While several schemes were recently attacked regarding these requirements, full reversibility of such a composition in order to produce colliding biometric characteristics, and specifically presentation attacks, were never demonstrated to the best of our knowledge. In this paper, we formalize these attacks for a traditional cancelable scheme with the help of integer linear programming (ILP) and quadratically constrained quadratic programming (QCQP). Solving these optimization problems allows an adversary to slightly alter its fingerprint image in order to impersonate any individual. Moreover, in an even more severe scenario, it is possible to simultaneously impersonate several individuals.

摘要: 可取消生物测定方案旨在通过将诸如密码、存储的秘密或盐等用户特定令牌与生物测定数据相结合来生成安全的生物测定模板。这种类型的变换被构造为生物测定变换与特征提取算法的组合。可撤销生物特征识别方案的安全性要求涉及模板的不可逆性、不可链接性和可撤销性，而不损失比较的准确性。虽然最近几个方案在这些要求方面受到攻击，但据我们所知，这种组合物的完全可逆性以产生碰撞的生物测定特征，特别是呈现攻击，从未被证明。本文利用整数线性规划(ILP)和二次约束二次规划(QCQP)对传统的可取消方案进行了形式化描述。解决这些优化问题允许对手稍微更改其指纹图像，以便冒充任何个人。此外，在更严重的情况下，可以同时冒充几个人。



## **38. Ad2Attack: Adaptive Adversarial Attack on Real-Time UAV Tracking**

Ad2Attack：无人机实时跟踪的自适应对抗攻击 cs.CV

7 pages, 7 figures, accepted by ICRA 2022

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2203.01516v1)

**Authors**: Changhong Fu, Sihang Li, Xinnan Yuan, Junjie Ye, Ziang Cao, Fangqiang Ding

**Abstracts**: Visual tracking is adopted to extensive unmanned aerial vehicle (UAV)-related applications, which leads to a highly demanding requirement on the robustness of UAV trackers. However, adding imperceptible perturbations can easily fool the tracker and cause tracking failures. This risk is often overlooked and rarely researched at present. Therefore, to help increase awareness of the potential risk and the robustness of UAV tracking, this work proposes a novel adaptive adversarial attack approach, i.e., Ad$^2$Attack, against UAV object tracking. Specifically, adversarial examples are generated online during the resampling of the search patch image, which leads trackers to lose the target in the following frames. Ad$^2$Attack is composed of a direct downsampling module and a super-resolution upsampling module with adaptive stages. A novel optimization function is proposed for balancing the imperceptibility and efficiency of the attack. Comprehensive experiments on several well-known benchmarks and real-world conditions show the effectiveness of our attack method, which dramatically reduces the performance of the most advanced Siamese trackers.

摘要: 无人机相关应用广泛采用视觉跟踪，这对无人机跟踪器的健壮性提出了很高的要求。然而，添加不可察觉的扰动很容易欺骗跟踪器并导致跟踪失败。目前，这一风险往往被忽视，研究甚少。因此，为了提高对无人机跟踪潜在风险和鲁棒性的认识，本文提出了一种新的针对无人机目标跟踪的自适应对抗性攻击方法，即Ad$^2$攻击。具体地说，在搜索补丁图像的重采样过程中，在线生成对抗性示例，这会导致跟踪者在随后的帧中丢失目标。AD$^2$攻击由直接下采样模块和带自适应阶段的超分辨率上采样模块组成。为了平衡攻击的隐蔽性和效率，提出了一种新的优化函数。在几个著名的基准和真实世界条件下的综合实验表明，我们的攻击方法是有效的，这大大降低了最先进的暹罗跟踪器的性能。



## **39. Two Attacks On Proof-of-Stake GHOST/Ethereum**

对证明鬼/以太的两次攻击 cs.CR

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2203.01315v1)

**Authors**: Joachim Neu, Ertem Nusret Tas, David Tse

**Abstracts**: We present two attacks targeting the Proof-of-Stake (PoS) Ethereum consensus protocol. The first attack suggests a fundamental conceptual incompatibility between PoS and the Greedy Heaviest-Observed Sub-Tree (GHOST) fork choice paradigm employed by PoS Ethereum. In a nutshell, PoS allows an adversary with a vanishing amount of stake to produce an unlimited number of equivocating blocks. While most equivocating blocks will be orphaned, such orphaned `uncle blocks' still influence fork choice under the GHOST paradigm, bestowing upon the adversary devastating control over the canonical chain. While the Latest Message Driven (LMD) aspect of current PoS Ethereum prevents a straightforward application of this attack, our second attack shows how LMD specifically can be exploited to obtain a new variant of the balancing attack that overcomes a recent protocol addition that was intended to mitigate balancing-type attacks. Thus, in its current form, PoS Ethereum without and with LMD is vulnerable to our first and second attack, respectively.

摘要: 我们提出了两种针对以太共识协议的攻击。第一个攻击表明，pos和pos Etherum采用的贪婪最重的子树(GHOST)分叉选择范例之间存在根本的概念上的不兼容。简而言之，POS允许赌注逐渐消失的对手产生无限数量的模棱两可的块。虽然大多数模棱两可的块将是孤立的，但这种孤立的“叔叔块”仍然影响着幽灵范式下的叉子选择，赋予对手对正则链的毁灭性控制。虽然当前PoS Etherum的最新消息驱动(LMD)方面阻止了此攻击的直接应用，但我们的第二个攻击显示了如何专门利用LMD来获得平衡攻击的新变体，该变体克服了最近添加的旨在缓解平衡型攻击的协议。因此，在目前的形式下，没有LMD和有LMD的POS Etherum分别容易受到我们的第一次和第二次攻击。



## **40. Detecting Adversarial Perturbations in Multi-Task Perception**

多任务感知中的敌意扰动检测 cs.CV

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2203.01177v1)

**Authors**: Marvin Klingner, Varun Ravi Kumar, Senthil Yogamani, Andreas Bär, Tim Fingscheidt

**Abstracts**: While deep neural networks (DNNs) achieve impressive performance on environment perception tasks, their sensitivity to adversarial perturbations limits their use in practical applications. In this paper, we (i) propose a novel adversarial perturbation detection scheme based on multi-task perception of complex vision tasks (i.e., depth estimation and semantic segmentation). Specifically, adversarial perturbations are detected by inconsistencies between extracted edges of the input image, the depth output, and the segmentation output. To further improve this technique, we (ii) develop a novel edge consistency loss between all three modalities, thereby improving their initial consistency which in turn supports our detection scheme. We verify our detection scheme's effectiveness by employing various known attacks and image noises. In addition, we (iii) develop a multi-task adversarial attack, aiming at fooling both tasks as well as our detection scheme. Experimental evaluation on the Cityscapes and KITTI datasets shows that under an assumption of a 5% false positive rate up to 100% of images are correctly detected as adversarially perturbed, depending on the strength of the perturbation. Code will be available on github. A short video at https://youtu.be/KKa6gOyWmH4 provides qualitative results.

摘要: 虽然深度神经网络(DNNs)在环境感知任务中取得了令人印象深刻的性能，但其对对抗性扰动的敏感性限制了其在实际应用中的应用。本文(I)提出了一种新的基于复杂视觉任务多任务感知(即深度估计和语义分割)的对抗性扰动检测方案。具体地说，通过所提取的输入图像的边缘、深度输出和分割输出之间的不一致来检测对抗性扰动。为了进一步改进这一技术，我们(Ii)在所有三种模态之间开发了一种新的边缘一致性损失，从而提高了它们的初始一致性，这反过来又支持我们的检测方案。我们通过使用各种已知攻击和图像噪声来验证我们的检测方案的有效性。此外，我们(Iii)开发了一种多任务对抗性攻击，旨在欺骗任务和我们的检测方案。在CITYSCAPES和KITTI数据集上的实验评估表明，在假阳性率为5%的假设下，高达100%的图像被正确检测为恶意扰动，这取决于扰动的强度。代码将在GitHub上提供。https://youtu.be/KKa6gOyWmH4上的一段简短视频提供了定性结果。



## **41. How to Inject Backdoors with Better Consistency: Logit Anchoring on Clean Data**

如何以更好的一致性注入后门：基于干净数据的Logit锚定 cs.LG

Accepted by ICLR 2022

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2109.01300v2)

**Authors**: Zhiyuan Zhang, Lingjuan Lyu, Weiqiang Wang, Lichao Sun, Xu Sun

**Abstracts**: Since training a large-scale backdoored model from scratch requires a large training dataset, several recent attacks have considered to inject backdoors into a trained clean model without altering model behaviors on the clean data. Previous work finds that backdoors can be injected into a trained clean model with Adversarial Weight Perturbation (AWP). Here AWPs refers to the variations of parameters that are small in backdoor learning. In this work, we observe an interesting phenomenon that the variations of parameters are always AWPs when tuning the trained clean model to inject backdoors. We further provide theoretical analysis to explain this phenomenon. We formulate the behavior of maintaining accuracy on clean data as the consistency of backdoored models, which includes both global consistency and instance-wise consistency. We extensively analyze the effects of AWPs on the consistency of backdoored models. In order to achieve better consistency, we propose a novel anchoring loss to anchor or freeze the model behaviors on the clean data, with a theoretical guarantee. Both the analytical and the empirical results validate the effectiveness of the anchoring loss in improving the consistency, especially the instance-wise consistency.

摘要: 由于从零开始训练大规模回溯模型需要大量的训练数据集，最近的几次攻击已经考虑在不改变干净数据上的模型行为的情况下向训练过的干净模型注入后门。以前的工作发现，后门可以被注入到具有对抗性权重扰动(AWP)的训练有素的干净模型中。这里的AWP指的是在后门学习中较小的参数变化。在这项工作中，我们观察到一个有趣的现象，即在调整训练好的干净模型进行后门注入时，参数的变化总是AWP。我们进一步对这一现象进行了理论分析。我们将在干净数据上保持准确性的行为表述为回溯模型的一致性，包括全局一致性和实例一致性。我们广泛地分析了AWP对回溯模型一致性的影响。为了达到更好的一致性，我们提出了一种新的锚定损失来锚定或冻结干净数据上的模型行为，并提供了理论上的保证。分析和实验结果都验证了锚定损失对提高一致性，特别是实例一致性的有效性。



## **42. Video is All You Need: Attacking PPG-based Biometric Authentication**

视频就是您需要的一切：攻击基于PPG的生物识别身份验证 cs.CR

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2203.00928v1)

**Authors**: Lin Li, Chao Chen, Lei Pan, Jun Zhang, Yang Xiang

**Abstracts**: Unobservable physiological signals enhance biometric authentication systems. Photoplethysmography (PPG) signals are convenient owning to its ease of measurement and are usually well protected against remote adversaries in authentication. Any leaked PPG signals help adversaries compromise the biometric authentication systems, and the advent of remote PPG (rPPG) enables adversaries to acquire PPG signals through restoration. While potentially dangerous, rPPG-based attacks are overlooked because existing methods require the victim's PPG signals. This paper proposes a novel spoofing attack approach that uses the waveforms of rPPG signals extracted from video clips to fool the PPG-based biometric authentication. We develop a new PPG restoration model that does not require leaked PPG signals for adversarial attacks. Test results on state-of-art PPG-based biometric authentication show that the signals recovered through rPPG pose a severe threat to PPG-based biometric authentication.

摘要: 不可观测的生理信号增强了生物特征认证系统。光体积描记(PPG)信号由于其易于测量而非常方便，并且通常在认证时能够很好地防止远程攻击。任何泄漏的PPG信号都会帮助攻击者危害生物特征认证系统，而远程PPG(RPPG)的出现使攻击者能够通过恢复来获取PPG信号。虽然存在潜在危险，但基于rPPG的攻击被忽略了，因为现有方法需要受害者的PPG信号。提出了一种利用从视频片段中提取的rPPG信号波形来欺骗基于PPG的生物特征认证的欺骗攻击方法。我们开发了一种新的PPG恢复模型，该模型不需要泄漏的PPG信号来进行对抗性攻击。对现有的基于PPG的生物特征认证的测试结果表明，通过rPPG恢复的信号对基于PPG的生物特征认证构成了严重的威胁。



## **43. Canonical foliations of neural networks: application to robustness**

神经网络的标准叶：在鲁棒性方面的应用 stat.ML

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2203.00922v1)

**Authors**: Eliot Tron, Nicolas Couellan, Stéphane Puechmorel

**Abstracts**: Adversarial attack is an emerging threat to the trustability of machine learning. Understanding these attacks is becoming a crucial task. We propose a new vision on neural network robustness using Riemannian geometry and foliation theory, and create a new adversarial attack by taking into account the curvature of the data space. This new adversarial attack called the "dog-leg attack" is a two-step approximation of a geodesic in the data space. The data space is treated as a (pseudo) Riemannian manifold equipped with the pullback of the Fisher Information Metric (FIM) of the neural network. In most cases, this metric is only semi-definite and its kernel becomes a central object to study. A canonical foliation is derived from this kernel. The curvature of the foliation's leaves gives the appropriate correction to get a two-step approximation of the geodesic and hence a new efficient adversarial attack. Our attack is tested on a toy example, a neural network trained to mimic the $\texttt{Xor}$ function, and demonstrates better results that the state of the art attack presented by Zhao et al. (2019).

摘要: 对抗性攻击是对机器学习可信性的一种新兴威胁。了解这些攻击正在成为一项至关重要的任务。我们利用黎曼几何和分层理论提出了一种新的神经网络鲁棒性的观点，并通过考虑数据空间的曲率来创建一种新的对抗性攻击。这种新的对抗性攻击被称为“狗腿攻击”，它是数据空间中测地线的两步近似。将数据空间处理为带有神经网络Fisher信息度量(FIM)回撤的(伪)黎曼流形。在大多数情况下，该度量只是半定的，其内核成为研究的中心对象。一个典型的叶理就是从这个核中衍生出来的。叶面的曲率给出了适当的修正，以得到测地线的两步近似，从而得到一种新的有效的对抗性攻击。我们的攻击在一个玩具示例上进行了测试，该神经网络被训练成模仿$\texttt{XOR}$函数，并显示了比赵等人提出的最新攻击更好的结果。(2019年)。



## **44. MIAShield: Defending Membership Inference Attacks via Preemptive Exclusion of Members**

MIAShield：通过抢占排除成员来防御成员推断攻击 cs.CR

21 pages, 17 figures, 10 tables

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2203.00915v1)

**Authors**: Ismat Jarin, Birhanu Eshete

**Abstracts**: In membership inference attacks (MIAs), an adversary observes the predictions of a model to determine whether a sample is part of the model's training data. Existing MIA defenses conceal the presence of a target sample through strong regularization, knowledge distillation, confidence masking, or differential privacy.   We propose MIAShield, a new MIA defense based on preemptive exclusion of member samples instead of masking the presence of a member. The key insight in MIAShield is weakening the strong membership signal that stems from the presence of a target sample by preemptively excluding it at prediction time without compromising model utility. To that end, we design and evaluate a suite of preemptive exclusion oracles leveraging model-confidence, exact or approximate sample signature, and learning-based exclusion of member data points. To be practical, MIAShield splits a training data into disjoint subsets and trains each subset to build an ensemble of models. The disjointedness of subsets ensures that a target sample belongs to only one subset, which isolates the sample to facilitate the preemptive exclusion goal.   We evaluate MIAShield on three benchmark image classification datasets. We show that MIAShield effectively mitigates membership inference (near random guess) for a wide range of MIAs, achieves far better privacy-utility trade-off compared with state-of-the-art defenses, and remains resilient against an adaptive adversary.

摘要: 在成员关系推断攻击(MIA)中，对手通过观察模型的预测来确定样本是否为模型训练数据的一部分。现有的MIA防御通过强正则化、知识提炼、置信度掩蔽或差分隐私来隐藏目标样本的存在。我们提出了MIAShield，一种新的基于抢占排除成员样本的MIA防御，而不是掩盖成员的存在。MIAShield的关键洞察力是在不影响模型效用的情况下，通过在预测时先发制人地排除目标样本，来削弱由于目标样本的存在而产生的强烈成员资格信号。为此，我们设计并评估了一套抢占式排除预言，利用模型置信度、精确或近似样本签名以及基于学习的成员数据点排除。实际上，MIAShield将训练数据分割成不相交的子集，并训练每个子集来构建模型集成。子集的不相交性保证了目标样本只属于一个子集，从而隔离了样本，有利于抢占排除目标的实现。我们在三个基准图像分类数据集上对MIAShield进行了评估。我们表明，MIAShield有效地缓解了大范围MIA的成员推断(近乎随机猜测)，与最先进的防御措施相比，实现了更好的隐私效用权衡，并且对自适应对手保持了弹性。



## **45. ECG-ATK-GAN: Robustness against Adversarial Attacks on ECGs using Conditional Generative Adversarial Networks**

ECG-ATK-GAN：基于条件生成对抗网络的心电对抗攻击鲁棒性 eess.SP

10 pages, 3 figures, 3 tables

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2110.09983v2)

**Authors**: Khondker Fariha Hossain, Sharif Amit Kamran, Alireza Tavakkoli, Xingjun Ma

**Abstracts**: Automating arrhythmia detection from ECG requires a robust and trusted system that retains high accuracy under electrical disturbances. Many machine learning approaches have reached human-level performance in classifying arrhythmia from ECGs. However, these architectures are vulnerable to adversarial attacks, which can misclassify ECG signals by decreasing the model's accuracy. Adversarial attacks are small crafted perturbations injected in the original data which manifest the out-of-distribution shifts in signal to misclassify the correct class. Thus, security concerns arise for false hospitalization and insurance fraud abusing these perturbations. To mitigate this problem, we introduce a novel Conditional Generative Adversarial Network (GAN), robust against adversarial attacked ECG signals and retaining high accuracy. Our architecture integrates a new class-weighted objective function for adversarial perturbation identification and two novel blocks for discerning and combining out-of-distribution shifts in signals in the learning process for accurately classifying various arrhythmia types. Furthermore, we benchmark our architecture on six different white and black-box attacks and compare with other recently proposed arrhythmia classification models on two publicly available ECG arrhythmia datasets. The experiment confirms that our model is more robust against such adversarial attacks for classifying arrhythmia with high accuracy.

摘要: 从ECG中自动检测心律失常需要一个健壮可靠的系统，该系统在电干扰下仍能保持高精度。许多机器学习方法在从心电图中分类心律失常方面已经达到了人类的水平。然而，这些体系结构容易受到敌意攻击，这些攻击会降低模型的准确性，从而导致心电信号的误分类。对抗性攻击是注入到原始数据中的小的、精心设计的扰动，它显示了信号的非分布转移，以误分类正确的类别。因此，滥用这些扰动的虚假住院和保险欺诈引起了安全担忧。为了缓解这一问题，我们引入了一种新的条件生成对抗网络(GAN)，该网络对敌意攻击的心电信号具有较强的鲁棒性，并保持了较高的准确率。我们的体系结构集成了一个新的类别加权目标函数来识别对抗性扰动，以及两个新的块来识别和组合学习过程中信号的非分布偏移，以准确地分类各种心律失常类型。此外，我们在六种不同的白盒和黑盒攻击上对我们的体系结构进行了基准测试，并在两个公开可用的ECG心律失常数据集上与其他最近提出的心律失常分类模型进行了比较。实验证明，该模型对心律失常的分类准确率较高，对这种对抗性攻击具有较强的鲁棒性。



## **46. Proceedings of the Artificial Intelligence for Cyber Security (AICS) Workshop at AAAI 2022**

2022年AAAI 2022年网络安全人工智能(AICS)研讨会论文集 cs.CR

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2202.14010v2)

**Authors**: James Holt, Edward Raff, Ahmad Ridley, Dennis Ross, Arunesh Sinha, Diane Staheli, William Streilen, Milind Tambe, Yevgeniy Vorobeychik, Allan Wollaber

**Abstracts**: The workshop will focus on the application of AI to problems in cyber security. Cyber systems generate large volumes of data, utilizing this effectively is beyond human capabilities. Additionally, adversaries continue to develop new attacks. Hence, AI methods are required to understand and protect the cyber domain. These challenges are widely studied in enterprise networks, but there are many gaps in research and practice as well as novel problems in other domains.   In general, AI techniques are still not widely adopted in the real world. Reasons include: (1) a lack of certification of AI for security, (2) a lack of formal study of the implications of practical constraints (e.g., power, memory, storage) for AI systems in the cyber domain, (3) known vulnerabilities such as evasion, poisoning attacks, (4) lack of meaningful explanations for security analysts, and (5) lack of analyst trust in AI solutions. There is a need for the research community to develop novel solutions for these practical issues.

摘要: 研讨会将重点讨论人工智能在网络安全问题上的应用。网络系统产生了大量的数据，有效地利用这些数据超出了人类的能力范围。此外，对手还在继续开发新的攻击。因此，需要人工智能方法来理解和保护网络领域。这些挑战在企业网络中得到了广泛的研究，但在研究和实践中还存在许多差距，在其他领域也出现了一些新的问题。总的来说，人工智能技术在现实世界中仍然没有被广泛采用。原因包括：(1)缺乏对人工智能安全的认证，(2)缺乏对网络领域中实际限制(例如，电力、内存、存储)对人工智能系统的影响的正式研究，(3)已知的漏洞，如逃避、中毒攻击，(4)缺乏对安全分析师的有意义的解释，以及(5)分析师对人工智能解决方案缺乏信任。研究界需要为这些实际问题开发新的解决方案。



## **47. Beyond Gradients: Exploiting Adversarial Priors in Model Inversion Attacks**

超越梯度：在模型反转攻击中利用对抗性先验 cs.LG

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2203.00481v1)

**Authors**: Dmitrii Usynin, Daniel Rueckert, Georgios Kaissis

**Abstracts**: Collaborative machine learning settings like federated learning can be susceptible to adversarial interference and attacks. One class of such attacks is termed model inversion attacks, characterised by the adversary reverse-engineering the model to extract representations and thus disclose the training data. Prior implementations of this attack typically only rely on the captured data (i.e. the shared gradients) and do not exploit the data the adversary themselves control as part of the training consortium. In this work, we propose a novel model inversion framework that builds on the foundations of gradient-based model inversion attacks, but additionally relies on matching the features and the style of the reconstructed image to data that is controlled by an adversary. Our technique outperforms existing gradient-based approaches both qualitatively and quantitatively, while still maintaining the same honest-but-curious threat model, allowing the adversary to obtain enhanced reconstructions while remaining concealed.

摘要: 协作式机器学习环境(如联合学习)很容易受到敌意干扰和攻击。一类这样的攻击被称为模型反转攻击，其特征是对手对模型进行逆向工程以提取表示，从而泄露训练数据。该攻击的先前实现通常仅依赖于捕获的数据(即共享梯度)，并且不利用对手自己控制的数据作为训练联盟的一部分。在这项工作中，我们提出了一种新的模型反演框架，它建立在基于梯度的模型反演攻击的基础上，但另外还依赖于将重建图像的特征和样式与对手控制的数据进行匹配。我们的技术在质量和数量上都优于现有的基于梯度的方法，同时仍然保持相同的诚实但好奇的威胁模型，允许攻击者在保持隐蔽的情况下获得增强的重建。



## **48. RAB: Provable Robustness Against Backdoor Attacks**

RAB：针对后门攻击的可证明的健壮性 cs.LG

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2003.08904v6)

**Authors**: Maurice Weber, Xiaojun Xu, Bojan Karlaš, Ce Zhang, Bo Li

**Abstracts**: Recent studies have shown that deep neural networks are vulnerable to adversarial attacks, including evasion and backdoor (poisoning) attacks. On the defense side, there have been intensive efforts on improving both empirical and provable robustness against evasion attacks; however, provable robustness against backdoor attacks still remains largely unexplored. In this paper, we focus on certifying the machine learning model robustness against general threat models, especially backdoor attacks. We first provide a unified framework via randomized smoothing techniques and show how it can be instantiated to certify the robustness against both evasion and backdoor attacks. We then propose the first robust training process, RAB, to smooth the trained model and certify its robustness against backdoor attacks. We theoretically prove the robustness bound for machine learning models trained with RAB, and prove that our robustness bound is tight. We derive the robustness conditions for different smoothing distributions including Gaussian and uniform distributions. In addition, we theoretically show that it is possible to train the robust smoothed models efficiently for simple models such as K-nearest neighbor classifiers, and we propose an exact smooth-training algorithm which eliminates the need to sample from a noise distribution for such models. Empirically, we conduct comprehensive experiments for different machine learning models such as DNNs and K-NN models on MNIST, CIFAR-10, and ImageNette datasets and provide the first benchmark for certified robustness against backdoor attacks. In addition, we evaluate K-NN models on a spambase tabular dataset to demonstrate the advantages of the proposed exact algorithm. Both the theoretic analysis and the comprehensive evaluation on diverse ML models and datasets shed lights on further robust learning strategies against general training time attacks.

摘要: 最近的研究表明，深层神经网络容易受到敌意攻击，包括逃避和后门(中毒)攻击。在防御方面，已经进行了密集的努力来提高针对规避攻击的经验性和可证明的健壮性；然而，针对后门攻击的可证明的健壮性在很大程度上仍未得到探索。在本文中，我们重点验证机器学习模型对一般威胁模型，特别是后门攻击的鲁棒性。我们首先通过随机平滑技术提供了一个统一的框架，并展示了如何将其实例化来证明对规避和后门攻击的鲁棒性。然后，我们提出了第一个鲁棒训练过程RAB，以平滑训练的模型并证明其对后门攻击的鲁棒性。从理论上证明了RAB训练的机器学习模型的稳健界，并证明了我们的稳健界是紧的。我们推导了不同平滑分布(包括高斯分布和均匀分布)的稳健性条件。此外，我们从理论上证明了对于K近邻分类器等简单模型，可以有效地训练鲁棒平滑模型，并提出了一种精确的平滑训练算法，该算法消除了对此类模型从噪声分布中采样的需要。经验上，我们在MNIST、CIFAR-10和ImageNette数据集上对不同的机器学习模型(如DNNS和K-NN模型)进行了全面的实验，并提供了第一个经验证的针对后门攻击的健壮性基准。此外，我们在垃圾邮件库表格数据集上对K-NN模型进行了评估，以展示所提出的精确算法的优势。理论分析和对不同ML模型和数据集的综合评价，为进一步研究抗一般训练时间攻击的鲁棒学习策略提供了理论依据。



## **49. Towards Robust Stacked Capsule Autoencoder with Hybrid Adversarial Training**

基于混合对抗训练的鲁棒堆叠式胶囊自动编码器 cs.CV

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2202.13755v2)

**Authors**: Jiazhu Dai, Siwei Xiong

**Abstracts**: Capsule networks (CapsNets) are new neural networks that classify images based on the spatial relationships of features. By analyzing the pose of features and their relative positions, it is more capable to recognize images after affine transformation. The stacked capsule autoencoder (SCAE) is a state-of-the-art CapsNet, and achieved unsupervised classification of CapsNets for the first time. However, the security vulnerabilities and the robustness of the SCAE has rarely been explored. In this paper, we propose an evasion attack against SCAE, where the attacker can generate adversarial perturbations based on reducing the contribution of the object capsules in SCAE related to the original category of the image. The adversarial perturbations are then applied to the original images, and the perturbed images will be misclassified. Furthermore, we propose a defense method called Hybrid Adversarial Training (HAT) against such evasion attacks. HAT makes use of adversarial training and adversarial distillation to achieve better robustness and stability. We evaluate the defense method and the experimental results show that the refined SCAE model can achieve 82.14% classification accuracy under evasion attack. The source code is available at https://github.com/FrostbiteXSW/SCAE_Defense.

摘要: 胶囊网络(CapsNets)是一种基于特征空间关系对图像进行分类的新型神经网络。通过分析特征的姿态及其相对位置，使仿射变换后的图像具有更强的识别能力。堆叠式胶囊自动编码器(SCAE)是一种先进的CapsNet，首次实现了CapsNet的无监督分类。然而，SCAE的安全漏洞和健壮性很少被研究。本文提出了一种针对SCAE的规避攻击，攻击者可以通过减少SCAE中对象胶囊相对于图像原始类别的贡献来产生敌意扰动。然后将对抗性扰动应用于原始图像，并且扰动图像将被错误分类。此外，针对此类逃避攻击，我们提出了一种称为混合对抗训练(HAT)的防御方法。HAT利用对抗性训练和对抗性蒸馏来实现更好的健壮性和稳定性。实验结果表明，改进后的SCAE模型在规避攻击下可以达到82.14%的分类正确率。源代码可以在https://github.com/FrostbiteXSW/SCAE_Defense.上找到



## **50. Adversarial Attack Framework on Graph Embedding Models with Limited Knowledge**

基于有限知识的图嵌入模型的对抗性攻击框架 cs.LG

Journal extension of GF-Attack, accepted by TKDE

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2105.12419v2)

**Authors**: Heng Chang, Yu Rong, Tingyang Xu, Wenbing Huang, Honglei Zhang, Peng Cui, Xin Wang, Wenwu Zhu, Junzhou Huang

**Abstracts**: With the success of the graph embedding model in both academic and industry areas, the robustness of graph embedding against adversarial attack inevitably becomes a crucial problem in graph learning. Existing works usually perform the attack in a white-box fashion: they need to access the predictions/labels to construct their adversarial loss. However, the inaccessibility of predictions/labels makes the white-box attack impractical to a real graph learning system. This paper promotes current frameworks in a more general and flexible sense -- we demand to attack various kinds of graph embedding models with black-box driven. We investigate the theoretical connections between graph signal processing and graph embedding models and formulate the graph embedding model as a general graph signal process with a corresponding graph filter. Therefore, we design a generalized adversarial attacker: GF-Attack. Without accessing any labels and model predictions, GF-Attack can perform the attack directly on the graph filter in a black-box fashion. We further prove that GF-Attack can perform an effective attack without knowing the number of layers of graph embedding models. To validate the generalization of GF-Attack, we construct the attacker on four popular graph embedding models. Extensive experiments validate the effectiveness of GF-Attack on several benchmark datasets.

摘要: 随着图嵌入模型在学术界和工业界的成功应用，图嵌入对敌意攻击的鲁棒性不可避免地成为图学习中的一个关键问题。现有的作品通常以白盒方式进行攻击：它们需要访问预测/标签来构建它们的对抗性损失。然而，预测/标签的不可访问性使得白盒攻击对于真实的图学习系统来说是不切实际的。本文在更一般、更灵活的意义上提升了现有的框架--我们要求攻击各种黑盒驱动的图嵌入模型。我们研究了图信号处理和图嵌入模型之间的理论联系，并将图嵌入模型表示为具有相应图过滤的一般图信号过程。因此，我们设计了一种广义对抗性攻击者：GF-攻击。GF-Attack在不访问任何标签和模型预测的情况下，可以黑盒方式直接对图过滤进行攻击。进一步证明了GF-攻击可以在不知道图嵌入模型层数的情况下进行有效的攻击。为了验证GF-攻击的泛化能力，我们在四种流行的图嵌入模型上构造了攻击者。在几个基准数据集上的大量实验验证了GF-攻击的有效性。



