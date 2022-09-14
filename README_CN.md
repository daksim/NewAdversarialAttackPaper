# Latest Adversarial Attack Papers
**update at 2022-09-15 06:31:28**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Certified Defences Against Adversarial Patch Attacks on Semantic Segmentation**

基于语义分割的对抗性补丁攻击认证防御 cs.CV

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05980v1)

**Authors**: Maksym Yatsura, Kaspar Sakmann, N. Grace Hua, Matthias Hein, Jan Hendrik Metzen

**Abstracts**: Adversarial patch attacks are an emerging security threat for real world deep learning applications. We present Demasked Smoothing, the first approach (up to our knowledge) to certify the robustness of semantic segmentation models against this threat model. Previous work on certifiably defending against patch attacks has mostly focused on image classification task and often required changes in the model architecture and additional training which is undesirable and computationally expensive. In Demasked Smoothing, any segmentation model can be applied without particular training, fine-tuning, or restriction of the architecture. Using different masking strategies, Demasked Smoothing can be applied both for certified detection and certified recovery. In extensive experiments we show that Demasked Smoothing can on average certify 64% of the pixel predictions for a 1% patch in the detection task and 48% against a 0.5% patch for the recovery task on the ADE20K dataset.

摘要: 对抗性补丁攻击是现实世界深度学习应用面临的一种新的安全威胁。我们提出了去任务平滑，这是第一种(据我们所知)来证明语义分割模型对这种威胁模型的稳健性的方法。以前关于可证明防御补丁攻击的工作主要集中在图像分类任务上，并且经常需要改变模型体系结构和额外的训练，这是不受欢迎的，并且计算代价高昂。在去任务平滑中，任何分割模型都可以在没有特定训练、微调或体系结构限制的情况下应用。使用不同的掩码策略，去掩码平滑可以应用于认证检测和认证恢复。在ADE20K数据集上的大量实验中，对于检测任务中1%的块，去任务平滑平均可以保证64%的像素预测，对于恢复任务，对于0.5%的块，去任务平滑平均可以保证48%的像素预测。



## **2. Adversarial Inter-Group Link Injection Degrades the Fairness of Graph Neural Networks**

对抗性组间链路注入降低了图神经网络的公平性 cs.LG

A shorter version of this work has been accepted by IEEE ICDM 2022

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05957v1)

**Authors**: Hussain Hussain, Meng Cao, Sandipan Sikdar, Denis Helic, Elisabeth Lex, Markus Strohmaier, Roman Kern

**Abstracts**: We present evidence for the existence and effectiveness of adversarial attacks on graph neural networks (GNNs) that aim to degrade fairness. These attacks can disadvantage a particular subgroup of nodes in GNN-based node classification, where nodes of the underlying network have sensitive attributes, such as race or gender. We conduct qualitative and experimental analyses explaining how adversarial link injection impairs the fairness of GNN predictions. For example, an attacker can compromise the fairness of GNN-based node classification by injecting adversarial links between nodes belonging to opposite subgroups and opposite class labels. Our experiments on empirical datasets demonstrate that adversarial fairness attacks can significantly degrade the fairness of GNN predictions (attacks are effective) with a low perturbation rate (attacks are efficient) and without a significant drop in accuracy (attacks are deceptive). This work demonstrates the vulnerability of GNN models to adversarial fairness attacks. We hope our findings raise awareness about this issue in our community and lay a foundation for the future development of GNN models that are more robust to such attacks.

摘要: 我们提出了针对图神经网络(GNN)的对抗性攻击的存在和有效性的证据，这些攻击旨在降低公平性。这些攻击可能使基于GNN的节点分类中的特定节点子组处于不利地位，其中底层网络的节点具有敏感属性，如种族或性别。我们进行了定性和实验分析，解释了敌意链接注入如何损害GNN预测的公平性。例如，攻击者可以通过在属于相反子组和相反类标签的节点之间注入敌对链接来损害基于GNN的节点分类的公平性。我们在经验数据集上的实验表明，对抗性公平攻击能够以较低的扰动率(攻击是有效的)显著降低GNN预测的公平性(攻击是有效的)，并且不会显著降低准确率(攻击是欺骗性的)。这项工作证明了GNN模型对敌意公平攻击的脆弱性。我们希望我们的发现提高我们社区对这个问题的认识，并为未来开发更稳健地抵御此类攻击的GNN模型奠定基础。



## **3. An Evolutionary, Gradient-Free, Query-Efficient, Black-Box Algorithm for Generating Adversarial Instances in Deep Networks**

一种进化、无梯度、查询高效的深层网络对抗性实例生成黑盒算法 cs.CV

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2208.08297v2)

**Authors**: Raz Lapid, Zvika Haramaty, Moshe Sipper

**Abstracts**: Deep neural networks (DNNs) are sensitive to adversarial data in a variety of scenarios, including the black-box scenario, where the attacker is only allowed to query the trained model and receive an output. Existing black-box methods for creating adversarial instances are costly, often using gradient estimation or training a replacement network. This paper introduces \textbf{Qu}ery-Efficient \textbf{E}volutiona\textbf{ry} \textbf{Attack}, \textit{QuEry Attack}, an untargeted, score-based, black-box attack. QuEry Attack is based on a novel objective function that can be used in gradient-free optimization problems. The attack only requires access to the output logits of the classifier and is thus not affected by gradient masking. No additional information is needed, rendering our method more suitable to real-life situations. We test its performance with three different state-of-the-art models -- Inception-v3, ResNet-50, and VGG-16-BN -- against three benchmark datasets: MNIST, CIFAR10 and ImageNet. Furthermore, we evaluate QuEry Attack's performance on non-differential transformation defenses and state-of-the-art robust models. Our results demonstrate the superior performance of QuEry Attack, both in terms of accuracy score and query efficiency.

摘要: 深度神经网络(DNN)对各种场景中的敌意数据很敏感，包括黑盒场景，在这种场景中，攻击者只被允许查询训练的模型并接收输出。现有的创建对抗性实例的黑盒方法成本很高，通常使用梯度估计或训练替换网络。本文介绍了一种非常有效的无目标、基于分数的黑盒攻击查询攻击基于一种新的目标函数，可用于无梯度优化问题。攻击只需要访问分类器的输出逻辑，因此不受梯度掩蔽的影响。不需要额外的信息，使我们的方法更适合实际情况。我们使用三个不同的最先进的模型--先启-v3、ResNet-50和VGG-16-BN--针对三个基准数据集：MNIST、CIFAR10和ImageNet测试其性能。此外，我们还评估了查询攻击在非差分变换防御和现有健壮性模型上的性能。实验结果表明，查询攻击在准确率和查询效率方面都具有较好的性能。



## **4. Bayesian Pseudo Labels: Expectation Maximization for Robust and Efficient Semi-Supervised Segmentation**

贝叶斯伪标签：稳健有效的半监督分割的期望最大化 cs.CV

MICCAI 2022 (Early accept, Student Travel Award)

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2208.04435v3)

**Authors**: Mou-Cheng Xu, Yukun Zhou, Chen Jin, Marius de Groot, Daniel C. Alexander, Neil P. Oxtoby, Yipeng Hu, Joseph Jacob

**Abstracts**: This paper concerns pseudo labelling in segmentation. Our contribution is fourfold. Firstly, we present a new formulation of pseudo-labelling as an Expectation-Maximization (EM) algorithm for clear statistical interpretation. Secondly, we propose a semi-supervised medical image segmentation method purely based on the original pseudo labelling, namely SegPL. We demonstrate SegPL is a competitive approach against state-of-the-art consistency regularisation based methods on semi-supervised segmentation on a 2D multi-class MRI brain tumour segmentation task and a 3D binary CT lung vessel segmentation task. The simplicity of SegPL allows less computational cost comparing to prior methods. Thirdly, we demonstrate that the effectiveness of SegPL may originate from its robustness against out-of-distribution noises and adversarial attacks. Lastly, under the EM framework, we introduce a probabilistic generalisation of SegPL via variational inference, which learns a dynamic threshold for pseudo labelling during the training. We show that SegPL with variational inference can perform uncertainty estimation on par with the gold-standard method Deep Ensemble.

摘要: 本文研究的是分割中的伪标注问题。我们的贡献是四倍的。首先，我们提出了一种新的伪标记公式，作为一种用于清晰统计解释的期望最大化(EM)算法。其次，提出了一种完全基于原始伪标记的半监督医学图像分割方法--SegPL。在2D多类MRI脑肿瘤分割任务和3D二值CT肺血管分割任务中，我们证明了SegPL是一种与最先进的基于一致性正则化的半监督分割方法相竞争的方法。与以前的方法相比，SegPL的简单性允许更少的计算成本。第三，我们证明了SegPL的有效性可能源于它对分布外噪声和对手攻击的健壮性。最后，在EM框架下，我们通过变分推理对SegPL进行概率推广，在训练过程中学习伪标签的动态阈值。我们证明了带变分推理的SegPL方法可以与金标准方法深层集成一样进行不确定度估计。



## **5. Adversarial Coreset Selection for Efficient Robust Training**

用于高效稳健训练的对抗性同位重置选择 cs.LG

Extended version of the ECCV2022 paper: arXiv:2112.00378. arXiv admin  note: substantial text overlap with arXiv:2112.00378

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05785v1)

**Authors**: Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstracts**: Neural networks are vulnerable to adversarial attacks: adding well-crafted, imperceptible perturbations to their input can modify their output. Adversarial training is one of the most effective approaches to training robust models against such attacks. Unfortunately, this method is much slower than vanilla training of neural networks since it needs to construct adversarial examples for the entire training data at every iteration. By leveraging the theory of coreset selection, we show how selecting a small subset of training data provides a principled approach to reducing the time complexity of robust training. To this end, we first provide convergence guarantees for adversarial coreset selection. In particular, we show that the convergence bound is directly related to how well our coresets can approximate the gradient computed over the entire training data. Motivated by our theoretical analysis, we propose using this gradient approximation error as our adversarial coreset selection objective to reduce the training set size effectively. Once built, we run adversarial training over this subset of the training data. Unlike existing methods, our approach can be adapted to a wide variety of training objectives, including TRADES, $\ell_p$-PGD, and Perceptual Adversarial Training. We conduct extensive experiments to demonstrate that our approach speeds up adversarial training by 2-3 times while experiencing a slight degradation in the clean and robust accuracy.

摘要: 神经网络很容易受到敌意攻击：在它们的输入中添加精心设计的、不可察觉的扰动可以修改它们的输出。对抗性训练是训练抵抗此类攻击的稳健模型的最有效方法之一。遗憾的是，这种方法比普通的神经网络训练要慢得多，因为它需要在每次迭代中为整个训练数据构造对抗性样本。通过利用核心选择理论，我们展示了如何选择一小部分训练数据提供了一种原则性的方法来降低稳健训练的时间复杂性。为此，我们首先为对抗性核心重置选择提供收敛保证。特别是，我们证明了收敛界与我们的核集在整个训练数据上计算的梯度的逼近程度直接相关。在理论分析的基础上，我们提出了利用这一梯度逼近误差作为对抗性核心集选择的目标，以有效地减少训练集的规模。一旦构建，我们就对训练数据的这个子集进行对抗性训练。与现有方法不同，我们的方法可以适应广泛的训练目标，包括交易、$\ell_p$-PGD和感知对手训练。我们进行了大量的实验，证明了我们的方法将对手训练速度提高了2-3倍，同时经历了干净和健壮的准确性的轻微下降。



## **6. Adaptive Perturbation Generation for Multiple Backdoors Detection**

多后门检测的自适应扰动生成 cs.CV

7 pages, 5 figures

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05244v2)

**Authors**: Yuhang Wang, Huafeng Shi, Rui Min, Ruijia Wu, Siyuan Liang, Yichao Wu, Ding Liang, Aishan Liu

**Abstracts**: Extensive evidence has demonstrated that deep neural networks (DNNs) are vulnerable to backdoor attacks, which motivates the development of backdoor detection methods. Existing backdoor detection methods are typically tailored for backdoor attacks with individual specific types (e.g., patch-based or perturbation-based). However, adversaries are likely to generate multiple types of backdoor attacks in practice, which challenges the current detection strategies. Based on the fact that adversarial perturbations are highly correlated with trigger patterns, this paper proposes the Adaptive Perturbation Generation (APG) framework to detect multiple types of backdoor attacks by adaptively injecting adversarial perturbations. Since different trigger patterns turn out to show highly diverse behaviors under the same adversarial perturbations, we first design the global-to-local strategy to fit the multiple types of backdoor triggers via adjusting the region and budget of attacks. To further increase the efficiency of perturbation injection, we introduce a gradient-guided mask generation strategy to search for the optimal regions for adversarial attacks. Extensive experiments conducted on multiple datasets (CIFAR-10, GTSRB, Tiny-ImageNet) demonstrate that our method outperforms state-of-the-art baselines by large margins(+12%).

摘要: 大量证据表明，深度神经网络(DNN)很容易受到后门攻击，这促使了后门检测方法的发展。现有的后门检测方法通常是为个别特定类型的后门攻击量身定做的(例如，基于补丁或基于扰动)。然而，攻击者在实践中可能会产生多种类型的后门攻击，这对现有的检测策略提出了挑战。基于敌意扰动与触发模式高度相关的事实，提出了自适应扰动生成(APG)框架，通过自适应注入敌意扰动来检测多种类型的后门攻击。由于不同的触发模式在相同的对抗性扰动下表现出高度不同的行为，我们首先设计了全局到局部的策略，通过调整攻击的区域和预算来适应多种类型的后门触发。为了进一步提高扰动注入的效率，我们引入了一种梯度引导的掩码生成策略来搜索对抗性攻击的最优区域。在多个数据集(CIFAR-10，GTSRB，Tiny-ImageNet)上进行的大量实验表明，我们的方法比最先进的基线方法有很大的优势(+12%)。



## **7. A Tale of HodgeRank and Spectral Method: Target Attack Against Rank Aggregation Is the Fixed Point of Adversarial Game**

HodgeRank和谱方法的故事：针对等级聚集的目标攻击是对抗性游戏的固定点 cs.LG

33 pages,  https://github.com/alphaprime/Target_Attack_Rank_Aggregation

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05742v1)

**Authors**: Ke Ma, Qianqian Xu, Jinshan Zeng, Guorong Li, Xiaochun Cao, Qingming Huang

**Abstracts**: Rank aggregation with pairwise comparisons has shown promising results in elections, sports competitions, recommendations, and information retrieval. However, little attention has been paid to the security issue of such algorithms, in contrast to numerous research work on the computational and statistical characteristics. Driven by huge profits, the potential adversary has strong motivation and incentives to manipulate the ranking list. Meanwhile, the intrinsic vulnerability of the rank aggregation methods is not well studied in the literature. To fully understand the possible risks, we focus on the purposeful adversary who desires to designate the aggregated results by modifying the pairwise data in this paper. From the perspective of the dynamical system, the attack behavior with a target ranking list is a fixed point belonging to the composition of the adversary and the victim. To perform the targeted attack, we formulate the interaction between the adversary and the victim as a game-theoretic framework consisting of two continuous operators while Nash equilibrium is established. Then two procedures against HodgeRank and RankCentrality are constructed to produce the modification of the original data. Furthermore, we prove that the victims will produce the target ranking list once the adversary masters the complete information. It is noteworthy that the proposed methods allow the adversary only to hold incomplete information or imperfect feedback and perform the purposeful attack. The effectiveness of the suggested target attack strategies is demonstrated by a series of toy simulations and several real-world data experiments. These experimental results show that the proposed methods could achieve the attacker's goal in the sense that the leading candidate of the perturbed ranking list is the designated one by the adversary.

摘要: 在选举、体育竞赛、推荐和信息检索等领域，采用配对比较的排名聚合方法已显示出良好的效果。然而，与大量关于算法的计算和统计特性的研究工作相比，对这类算法的安全问题关注较少。在巨额利润的驱动下，潜在对手操纵排行榜的动机和动机很强。同时，文献中对秩聚类方法的内在脆弱性的研究还不够深入。为了充分理解可能的风险，我们将重点放在有目的的对手身上，他们希望通过修改成对数据来指定聚合结果。从动力系统的角度来看，具有目标排行榜的攻击行为是属于对手和受害者组成的固定点。为了进行有针对性的攻击，我们将对手和受害者之间的相互作用描述为一个由两个连续算子组成的博弈论框架，同时建立了纳什均衡。然后构造了针对HodgeRank和RankCentrality的两个过程来产生对原始数据的修改。此外，我们证明了一旦对手掌握了完整的信息，受害者就会产生目标排名表。值得注意的是，所提出的方法只允许对手持有不完全信息或不完全反馈，并执行有目的的攻击。通过一系列玩具仿真和几个真实世界的数据实验，验证了所提出的目标攻击策略的有效性。实验结果表明，所提出的方法能够达到攻击者的目的，即扰动排序列表的领先候选者就是对手指定的候选者。



## **8. Sample Complexity of an Adversarial Attack on UCB-based Best-arm Identification Policy**

基于UCB的最佳武器识别策略下对抗性攻击的样本复杂性 cs.LG

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05692v1)

**Authors**: Varsha Pendyala

**Abstracts**: In this work I study the problem of adversarial perturbations to rewards, in a Multi-armed bandit (MAB) setting. Specifically, I focus on an adversarial attack to a UCB type best-arm identification policy applied to a stochastic MAB. The UCB attack presented in [1] results in pulling a target arm K very often. I used the attack model of [1] to derive the sample complexity required for selecting target arm K as the best arm. I have proved that the stopping condition of UCB based best-arm identification algorithm given in [2], can be achieved by the target arm K in T rounds, where T depends only on the total number of arms and $\sigma$ parameter of $\sigma^2-$ sub-Gaussian random rewards of the arms.

摘要: 在这项工作中，我研究了多臂强盗(MAB)环境下的对抗性报酬摄动问题。具体地说，我将重点放在对应用于随机MAB的UCB类型的最佳ARM识别策略的对抗性攻击上。文[1]中提出的UCB攻击导致经常拉动目标手臂K。我使用了[1]的攻击模型来推导出选择目标手臂K作为最佳手臂所需的样本复杂度。证明了文[2]中给出的基于UCB的最佳手臂识别算法的停止条件可以由目标手臂K在T轮中实现，其中T仅取决于手臂的总数和手臂的$\sigma^2-$亚高斯随机奖励的$\sigma参数。



## **9. Replay-based Recovery for Autonomous Robotic Vehicles from Sensor Deception Attacks**

基于重放的自主机器人对传感器欺骗攻击的恢复 cs.RO

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.04554v2)

**Authors**: Pritam Dash, Guanpeng Li, Mehdi Karimibiuki, Karthik Pattabiraman

**Abstracts**: Sensors are crucial for autonomous operation in robotic vehicles (RV). Physical attacks on sensors such as sensor tampering or spoofing can feed erroneous values to RVs through physical channels, which results in mission failures. In this paper, we present DeLorean, a comprehensive diagnosis and recovery framework for securing autonomous RVs from physical attacks. We consider a strong form of physical attack called sensor deception attacks (SDAs), in which the adversary targets multiple sensors of different types simultaneously (even including all sensors). Under SDAs, DeLorean inspects the attack induced errors, identifies the targeted sensors, and prevents the erroneous sensor inputs from being used in RV's feedback control loop. DeLorean replays historic state information in the feedback control loop and recovers the RV from attacks. Our evaluation on four real and two simulated RVs shows that DeLorean can recover RVs from different attacks, and ensure mission success in 94% of the cases (on average), without any crashes. DeLorean incurs low performance, memory and battery overheads.

摘要: 传感器对于机器人车辆(RV)的自主操作至关重要。对传感器的物理攻击，如传感器篡改或欺骗，可能会通过物理通道向房车提供错误的值，从而导致任务失败。在本文中，我们提出了DeLorean，一个全面的诊断和恢复框架，用于保护自主房车免受物理攻击。我们考虑了一种称为传感器欺骗攻击(SDA)的强物理攻击形式，在这种攻击中，对手同时针对不同类型的多个传感器(甚至包括所有传感器)。在SDAS下，DeLorean检查攻击导致的错误，识别目标传感器，并防止错误的传感器输入用于房车的反馈控制回路。DeLorean在反馈控制环路中重放历史状态信息，并恢复RV免受攻击。我们对四辆真实房车和两辆模拟房车的评估表明，DeLorean可以从不同的攻击中恢复房车，并确保94%的任务成功(平均而言)，而不会发生任何崩溃。DeLorean的性能、内存和电池开销都很低。



## **10. Boosting Robustness Verification of Semantic Feature Neighborhoods**

增强语义特征邻域的健壮性验证 cs.LG

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.05446v1)

**Authors**: Anan Kabaha, Dana Drachsler-Cohen

**Abstracts**: Deep neural networks have been shown to be vulnerable to adversarial attacks that perturb inputs based on semantic features. Existing robustness analyzers can reason about semantic feature neighborhoods to increase the networks' reliability. However, despite the significant progress in these techniques, they still struggle to scale to deep networks and large neighborhoods. In this work, we introduce VeeP, an active learning approach that splits the verification process into a series of smaller verification steps, each is submitted to an existing robustness analyzer. The key idea is to build on prior steps to predict the next optimal step. The optimal step is predicted by estimating the certification velocity and sensitivity via parametric regression. We evaluate VeeP on MNIST, Fashion-MNIST, CIFAR-10 and ImageNet and show that it can analyze neighborhoods of various features: brightness, contrast, hue, saturation, and lightness. We show that, on average, given a 90 minute timeout, VeeP verifies 96% of the maximally certifiable neighborhoods within 29 minutes, while existing splitting approaches verify, on average, 73% of the maximally certifiable neighborhoods within 58 minutes.

摘要: 深度神经网络已被证明容易受到基于语义特征的输入扰乱的对抗性攻击。现有的健壮性分析器可以对语义特征邻域进行推理，以增加网络的可靠性。然而，尽管这些技术取得了重大进展，它们仍难以扩展到深度网络和大型社区。在这项工作中，我们引入了VeEP，这是一种主动学习方法，它将验证过程划分为一系列较小的验证步骤，每个步骤都提交给现有的健壮性分析器。其关键思想是建立在先前步骤的基础上，以预测下一个最佳步骤。通过参数回归估计认证速度和灵敏度，预测最优步骤。我们在MNIST、Fashion-MNIST、CIFAR-10和ImageNet上对Veep进行了评估，结果表明，它可以分析各种特征的社区：亮度、对比度、色调、饱和度和亮度。我们发现，在平均90分钟的超时时间内，Veep在29分钟内验证了96%的最大可证明邻域，而现有的分割方法平均在58分钟内验证了73%的最大可证明邻域。



## **11. Bilevel Optimization with a Lower-level Contraction: Optimal Sample Complexity without Warm-Start**

低水平收缩的两层优化：无热启动的最优样本复杂性 stat.ML

35 pages, 2 figures. Code at  https://github.com/CSML-IIT-UCL/bioptexps

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2202.03397v2)

**Authors**: Riccardo Grazzi, Massimiliano Pontil, Saverio Salzo

**Abstracts**: We analyze a general class of bilevel problems, in which the upper-level problem consists in the minimization of a smooth objective function and the lower-level problem is to find the fixed point of a smooth contraction map. This type of problems include instances of meta-learning, equilibrium models, hyperparameter optimization and data poisoning adversarial attacks. Several recent works have proposed algorithms which warm-start the lower level problem, i.e. they use the previous lower-level approximate solution as a staring point for the lower-level solver. This warm-start procedure allows one to improve the sample complexity in both the stochastic and deterministic settings, achieving in some cases the order-wise optimal sample complexity. However, there are situations, e.g., meta learning and equilibrium models, in which the warm-start procedure is not well-suited or ineffective. In this work we show that without warm-start, it is still possible to achieve order-wise optimal or near-optimal sample complexity. In particular, we propose a simple method which uses stochastic fixed point iterations at the lower-level and projected inexact gradient descent at the upper-level, that reaches an $\epsilon$-stationary point using $O(\epsilon^{-2})$ and $\tilde{O}(\epsilon^{-1})$ samples for the stochastic and the deterministic setting, respectively. Finally, compared to methods using warm-start, our approach yields a simpler analysis that does not need to study the coupled interactions between the upper-level and lower-level iterates

摘要: 我们分析了一类一般的两层问题，其中上层问题在于光滑目标函数的最小化，下层问题是寻找光滑压缩映射的不动点。这类问题包括元学习、均衡模型、超参数优化和数据中毒攻击等实例。最近的一些工作提出了暖启动下层问题的算法，即使用先前的下层近似解作为下层求解器的起始点。这种热启动过程允许人们在随机和确定性设置下改善样本复杂性，在某些情况下实现顺序最优的样本复杂性。然而，在一些情况下，例如元学习和平衡模型，热启动程序不是很适合或无效的。在这项工作中，我们证明了在没有热启动的情况下，仍然有可能获得按顺序最优或接近最优的样本复杂度。特别地，我们提出了一种简单的方法，它在低层使用随机不动点迭代，在上层使用投影的不精确梯度下降，分别在随机和确定环境下使用$O(epsilon^{-2})$和$tide{O}(epsilon^{-1})$样本达到$epsilon$-驻点。最后，与使用热启动的方法相比，我们的方法产生了更简单的分析，不需要研究上层和下层迭代之间的耦合作用



## **12. Fiat-Shamir for Proofs Lacks a Proof Even in the Presence of Shared Entanglement**

菲亚特-沙米尔的证据缺乏证据，即使在存在共同纠缠的情况下 quant-ph

62 pages, 2 figures

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2204.02265v2)

**Authors**: Frédéric Dupuis, Philippe Lamontagne, Louis Salvail

**Abstracts**: We explore the cryptographic power of arbitrary shared physical resources. The most general such resource is access to a fresh entangled quantum state at the outset of each protocol execution. We call this the Common Reference Quantum State (CRQS) model, in analogy to the well-known Common Reference String (CRS). The CRQS model is a natural generalization of the CRS model but appears to be more powerful: in the two-party setting, a CRQS can sometimes exhibit properties associated with a Random Oracle queried once by measuring a maximally entangled state in one of many mutually unbiased bases. We formalize this notion as a Weak One-Time Random Oracle (WOTRO), where we only ask of the $m$--bit output to have some randomness when conditioned on the $n$--bit input.   We show that when $n-m\in\omega(\lg n)$, any protocol for WOTRO in the CRQS model can be attacked by an (inefficient) adversary. Moreover, our adversary is efficiently simulatable, which rules out the possibility of proving the computational security of a scheme by a black-box reduction to a cryptographic game assumption. On the other hand, we introduce a non-game quantum assumption for hash functions that implies WOTRO in the CRQ\$ model (where the CRQS consists only of EPR pairs). We first build a statistically secure WOTRO protocol where $m=n$, then hash the output.   The impossibility of WOTRO has the following consequences. First, we show the black-box impossibility of a quantum Fiat-Shamir transform, extending the impossibility result of Bitansky et al. (TCC '13) to the CRQS model. Second, we show a black-box impossibility result for a strenghtened version of quantum lightning (Zhandry, Eurocrypt '19) where quantum bolts have an additional parameter that cannot be changed without generating new bolts.

摘要: 我们探索任意共享物理资源的加密能力。最常见的这类资源是在每个协议执行开始时访问新的纠缠量子态。我们称之为公共参考量子态(CRQS)模型，类似于众所周知的公共参考弦(CRS)。CRQS模型是CRS模型的自然推广，但似乎更强大：在两方设置中，CRQS有时可以通过测量许多相互无偏的碱基之一中的最大纠缠态来展示与查询一次的随机Oracle相关联的属性。我们将这个概念形式化为弱一次性随机Oracle(WOTRO)，其中我们只要求$m$位的输出在以$n$位输入为条件时具有一定的随机性。我们证明了当$n-m\in\omega(\lg n)$时，CRQS模型中用于WOTRO的任何协议都可以被(低效的)攻击者攻击。此外，我们的对手是高效可模拟的，这排除了通过将黑盒简化为密码博弈假设来证明方案的计算安全性的可能性。另一方面，我们引入了散列函数的非博弈量子假设，在CRQ模型(其中CRQS只由EPR对组成)中隐含了WOTRO。我们首先构建一个统计安全的WOTRO协议，其中$m=n$，然后对输出进行散列。WOTRO的不可能性会产生以下后果。首先，我们证明了量子Fiat-Shamir变换的黑盒不可能性，推广了Bitansky等人的不可能结果。其次，我们给出了一个加强版量子闪电(Zhandry，Eurocrypt‘19)的黑箱不可能结果，其中量子闪电有一个额外的参数，如果不产生新的闪电，这个参数就不能改变。



## **13. A Survey of Machine Unlearning**

机器遗忘研究综述 cs.LG

fixed overlaps

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.02299v4)

**Authors**: Thanh Tam Nguyen, Thanh Trung Huynh, Phi Le Nguyen, Alan Wee-Chung Liew, Hongzhi Yin, Quoc Viet Hung Nguyen

**Abstracts**: Computer systems hold a large amount of personal data over decades. On the one hand, such data abundance allows breakthroughs in artificial intelligence (AI), especially machine learning (ML) models. On the other hand, it can threaten the privacy of users and weaken the trust between humans and AI. Recent regulations require that private information about a user can be removed from computer systems in general and from ML models in particular upon request (e.g. the "right to be forgotten"). While removing data from back-end databases should be straightforward, it is not sufficient in the AI context as ML models often "remember" the old data. Existing adversarial attacks proved that we can learn private membership or attributes of the training data from the trained models. This phenomenon calls for a new paradigm, namely machine unlearning, to make ML models forget about particular data. It turns out that recent works on machine unlearning have not been able to solve the problem completely due to the lack of common frameworks and resources. In this survey paper, we seek to provide a thorough investigation of machine unlearning in its definitions, scenarios, mechanisms, and applications. Specifically, as a categorical collection of state-of-the-art research, we hope to provide a broad reference for those seeking a primer on machine unlearning and its various formulations, design requirements, removal requests, algorithms, and uses in a variety of ML applications. Furthermore, we hope to outline key findings and trends in the paradigm as well as highlight new areas of research that have yet to see the application of machine unlearning, but could nonetheless benefit immensely. We hope this survey provides a valuable reference for ML researchers as well as those seeking to innovate privacy technologies. Our resources are at https://github.com/tamlhp/awesome-machine-unlearning.

摘要: 几十年来，计算机系统保存着大量的个人数据。一方面，这样的数据丰富使人工智能(AI)，特别是机器学习(ML)模型取得了突破。另一方面，它会威胁用户的隐私，削弱人类与AI之间的信任。最近的法规要求，一般情况下，可以从计算机系统中删除关于用户的私人信息，特别是在请求时可以从ML模型中删除用户的私人信息(例如，“被遗忘权”)。虽然从后端数据库中删除数据应该很简单，但在人工智能环境中这是不够的，因为ML模型经常“记住”旧数据。现有的对抗性攻击证明，我们可以从训练好的模型中学习训练数据的私人成员或属性。这种现象呼唤一种新的范式，即机器遗忘，以使ML模型忘记特定的数据。事实证明，由于缺乏通用的框架和资源，最近关于机器遗忘的研究并不能完全解决这个问题。在这篇调查论文中，我们试图对机器遗忘的定义、场景、机制和应用进行全面的调查。具体地说，作为最新研究的分类集合，我们希望为那些寻求机器遗忘及其各种公式、设计要求、移除请求、算法和在各种ML应用中使用的入门知识的人提供广泛的参考。此外，我们希望概述该范式中的主要发现和趋势，并强调尚未看到机器遗忘应用的新研究领域，但仍可能受益匪浅。我们希望这项调查为ML研究人员以及那些寻求创新隐私技术的人提供有价值的参考。我们的资源在https://github.com/tamlhp/awesome-machine-unlearning.



## **14. GRNN: Generative Regression Neural Network -- A Data Leakage Attack for Federated Learning**

GRNN：生成回归神经网络--一种面向联邦学习的数据泄漏攻击 cs.LG

The source code can be found at: https://github.com/Rand2AI/GRNN

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2105.00529v3)

**Authors**: Hanchi Ren, Jingjing Deng, Xianghua Xie

**Abstracts**: Data privacy has become an increasingly important issue in Machine Learning (ML), where many approaches have been developed to tackle this challenge, e.g. cryptography (Homomorphic Encryption (HE), Differential Privacy (DP), etc.) and collaborative training (Secure Multi-Party Computation (MPC), Distributed Learning and Federated Learning (FL)). These techniques have a particular focus on data encryption or secure local computation. They transfer the intermediate information to the third party to compute the final result. Gradient exchanging is commonly considered to be a secure way of training a robust model collaboratively in Deep Learning (DL). However, recent researches have demonstrated that sensitive information can be recovered from the shared gradient. Generative Adversarial Network (GAN), in particular, has shown to be effective in recovering such information. However, GAN based techniques require additional information, such as class labels which are generally unavailable for privacy-preserved learning. In this paper, we show that, in the FL system, image-based privacy data can be easily recovered in full from the shared gradient only via our proposed Generative Regression Neural Network (GRNN). We formulate the attack to be a regression problem and optimize two branches of the generative model by minimizing the distance between gradients. We evaluate our method on several image classification tasks. The results illustrate that our proposed GRNN outperforms state-of-the-art methods with better stability, stronger robustness, and higher accuracy. It also has no convergence requirement to the global FL model. Moreover, we demonstrate information leakage using face re-identification. Some defense strategies are also discussed in this work.

摘要: 数据隐私已经成为机器学习(ML)中一个日益重要的问题，人们已经开发了许多方法来应对这一挑战，例如密码学(同态加密(HE)、差分隐私(DP)等)。和协作培训(安全多方计算(MPC)、分布式学习和联合学习(FL))。这些技术特别关注数据加密或安全本地计算。他们将中间信息传递给第三方来计算最终结果。梯度交换通常被认为是深度学习中协作训练健壮模型的一种安全方式。然而，最近的研究表明，敏感信息可以从共享梯度中恢复出来。尤其是生成性对抗网络(GAN)在恢复这类信息方面是有效的。然而，基于GaN的技术需要额外的信息，例如类别标签，这些信息通常不能用于隐私保护学习。在本文中，我们证明，在FL系统中，仅通过我们提出的生成回归神经网络(GRNN)就可以很容易地从共享梯度中完全恢复基于图像的隐私数据。我们将攻击描述为一个回归问题，并通过最小化梯度之间的距离来优化生成模型的两个分支。我们在几个图像分类任务上对我们的方法进行了评估。实验结果表明，本文提出的GRNN方法在稳定性、鲁棒性和准确率等方面均优于现有的方法。它对全局FL模型也没有收敛要求。此外，我们还使用人脸重新识别来演示信息泄漏。文中还讨论了一些防御策略。



## **15. Semantic-Preserving Adversarial Code Comprehension**

保留语义的对抗性代码理解 cs.CL

Accepted by COLING 2022

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.05130v1)

**Authors**: Yiyang Li, Hongqiu Wu, Hai Zhao

**Abstracts**: Based on the tremendous success of pre-trained language models (PrLMs) for source code comprehension tasks, current literature studies either ways to further improve the performance (generalization) of PrLMs, or their robustness against adversarial attacks. However, they have to compromise on the trade-off between the two aspects and none of them consider improving both sides in an effective and practical way. To fill this gap, we propose Semantic-Preserving Adversarial Code Embeddings (SPACE) to find the worst-case semantic-preserving attacks while forcing the model to predict the correct labels under these worst cases. Experiments and analysis demonstrate that SPACE can stay robust against state-of-the-art attacks while boosting the performance of PrLMs for code.

摘要: 基于预先训练的语言模型在源代码理解任务中的巨大成功，目前的文献研究要么是进一步提高预先训练的语言模型的性能(泛化)，要么是研究它们对对手攻击的健壮性。然而，他们不得不在这两个方面的权衡上妥协，没有一个人考虑以有效和实际的方式改善双方。为了填补这一空白，我们提出了保持语义的对抗性代码嵌入(SPACE)，以发现最坏情况下保持语义的攻击，同时迫使模型在这些最坏情况下预测正确的标签。实验和分析表明，SPACE在提高PrLMS代码性能的同时，可以保持对最先进攻击的健壮性。



## **16. Passive Triangulation Attack on ORide**

ORIDE上的被动三角剖分攻击 cs.CR

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2208.12216v2)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstracts**: Privacy preservation in Ride Hailing Services is intended to protect privacy of drivers and riders. ORide is one of the early RHS proposals published at USENIX Security Symposium 2017. In the ORide protocol, riders and drivers, operating in a zone, encrypt their locations using a Somewhat Homomorphic Encryption scheme (SHE) and forward them to the Service Provider (SP). SP homomorphically computes the squared Euclidean distance between riders and available drivers. Rider receives the encrypted distances and selects the optimal rider after decryption. In order to prevent a triangulation attack, SP randomly permutes the distances before sending them to the rider. In this work, we use propose a passive attack that uses triangulation to determine coordinates of all participating drivers whose permuted distances are available from the points of view of multiple honest-but-curious adversary riders. An attack on ORide was published at SAC 2021. The same paper proposes a countermeasure using noisy Euclidean distances to thwart their attack. We extend our attack to determine locations of drivers when given their permuted and noisy Euclidean distances from multiple points of reference, where the noise perturbation comes from a uniform distribution. We conduct experiments with different number of drivers and for different perturbation values. Our experiments show that we can determine locations of all drivers participating in the ORide protocol. For the perturbed distance version of the ORide protocol, our algorithm reveals locations of about 25% to 50% of participating drivers. Our algorithm runs in time polynomial in number of drivers.

摘要: 网约车服务中的隐私保护旨在保护司机和乘客的隐私。ORIDE是USENIX安全研讨会2017上发布的早期RHS提案之一。在ORIDE协议中，在区域中操作的乘客和司机使用某种同态加密方案(SHE)加密他们的位置，并将其转发给服务提供商(SP)。SP同态计算乘客和可用司机之间的平方欧几里得距离。骑手收到加密的距离，解密后选择最优的骑手。为了防止三角测量攻击，SP在将距离发送给骑手之前随机排列距离。在这项工作中，我们使用了一种被动攻击，该攻击使用三角测量来确定所有参与的司机的坐标，这些司机的置换距离是从多个诚实但好奇的对手车手的角度出发的。对ORide的攻击在SAC 2021上发表。同时提出了一种利用噪声欧几里德距离来阻止他们攻击的对策。当给定司机与多个参考点的置换和噪声欧几里德距离时，我们将我们的攻击扩展到确定司机的位置，其中噪声扰动来自均匀分布。我们对不同数量的驱动器和不同的摄动值进行了实验。我们的实验表明，我们可以确定所有参与ORIDE协议的司机的位置。对于受干扰的距离版本的ORide协议，我们的算法显示了大约25%到50%的参与司机的位置。我们的算法以时间多项式的形式运行在驱动器的数量上。



## **17. CARE: Certifiably Robust Learning with Reasoning via Variational Inference**

注意：通过变分推理进行推理的可证明稳健学习 cs.LG

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.05055v1)

**Authors**: Jiawei Zhang, Linyi Li, Ce Zhang, Bo Li

**Abstracts**: Despite great recent advances achieved by deep neural networks (DNNs), they are often vulnerable to adversarial attacks. Intensive research efforts have been made to improve the robustness of DNNs; however, most empirical defenses can be adaptively attacked again, and the theoretically certified robustness is limited, especially on large-scale datasets. One potential root cause of such vulnerabilities for DNNs is that although they have demonstrated powerful expressiveness, they lack the reasoning ability to make robust and reliable predictions. In this paper, we aim to integrate domain knowledge to enable robust learning with the reasoning paradigm. In particular, we propose a certifiably robust learning with reasoning pipeline (CARE), which consists of a learning component and a reasoning component. Concretely, we use a set of standard DNNs to serve as the learning component to make semantic predictions, and we leverage the probabilistic graphical models, such as Markov logic networks (MLN), to serve as the reasoning component to enable knowledge/logic reasoning. However, it is known that the exact inference of MLN (reasoning) is #P-complete, which limits the scalability of the pipeline. To this end, we propose to approximate the MLN inference via variational inference based on an efficient expectation maximization algorithm. In particular, we leverage graph convolutional networks (GCNs) to encode the posterior distribution during variational inference and update the parameters of GCNs (E-step) and the weights of knowledge rules in MLN (M-step) iteratively. We conduct extensive experiments on different datasets and show that CARE achieves significantly higher certified robustness compared with the state-of-the-art baselines. We additionally conducted different ablation studies to demonstrate the empirical robustness of CARE and the effectiveness of different knowledge integration.

摘要: 尽管深度神经网络(DNN)最近取得了很大的进展，但它们往往容易受到对手的攻击。人们已经进行了大量的研究来提高DNN的稳健性，然而，大多数经验防御都可以再次自适应攻击，理论上证明的健壮性是有限的，特别是在大规模数据集上。DNN这种漏洞的一个潜在根本原因是，尽管它们表现出强大的表现力，但它们缺乏做出稳健和可靠预测的推理能力。在本文中，我们的目标是将领域知识集成到推理范式中，以实现稳健的学习。特别地，我们提出了一种带推理的可证明稳健学习流水线(CARE)，该流水线由学习组件和推理组件组成。具体地说，我们使用一组标准的DNN作为学习组件进行语义预测，并利用马尔可夫逻辑网络(MLN)等概率图形模型作为推理组件来实现知识/逻辑推理。然而，众所周知，MLN(推理)的精确推理是#P-完全的，这限制了流水线的可扩展性。为此，我们提出了基于一种有效的期望最大化算法的变分推理来逼近最大似然推理。特别是，我们利用图卷积网络(GCNS)对变分推理过程中的后验分布进行编码，并迭代地更新GCNS的参数(E步)和MLN中知识规则的权值(M步)。我们在不同的数据集上进行了广泛的实验，并表明与最先进的基线相比，CARE实现了显著更高的认证稳健性。此外，我们还进行了不同的消融研究，以证明CARE的经验稳健性和不同知识整合的有效性。



## **18. GFCL: A GRU-based Federated Continual Learning Framework against Data Poisoning Attacks in IoV**

GFCL：一种基于GRU的联合持续学习框架对抗IoV中的数据中毒攻击 cs.LG

11 pages, 12 figures, 3 tables; This work has been submitted to the  IEEE Transactions on Vehicular Technology for possible publication. Copyright  may be transferred without notice, after which this version may no longer be  accessible

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2204.11010v2)

**Authors**: Anum Talpur, Mohan Gurusamy

**Abstracts**: Integration of machine learning (ML) in 5G-based Internet of Vehicles (IoV) networks has enabled intelligent transportation and smart traffic management. Nonetheless, the security against adversarial poisoning attacks is also increasingly becoming a challenging task. Specifically, Deep Reinforcement Learning (DRL) is one of the widely used ML designs in IoV applications. The standard ML security techniques are not effective in DRL where the algorithm learns to solve sequential decision-making through continuous interaction with the environment, and the environment is time-varying, dynamic, and mobile. In this paper, we propose a Gated Recurrent Unit (GRU)-based federated continual learning (GFCL) anomaly detection framework against Sybil-based data poisoning attacks in IoV. The objective is to present a lightweight and scalable framework that learns and detects the illegitimate behavior without having a-priori training dataset consisting of attack samples. We use GRU to predict a future data sequence to analyze and detect illegitimate behavior from vehicles in a federated learning-based distributed manner. We investigate the performance of our framework using real-world vehicle mobility traces. The results demonstrate the effectiveness of our proposed solution in terms of different performance metrics.

摘要: 将机器学习(ML)集成到基于5G的车联网(IoV)网络中，实现了智能交通和智能交通管理。尽管如此，防御对抗性中毒攻击的安全也日益成为一项具有挑战性的任务。其中，深度强化学习(DRL)是IoV应用中广泛使用的ML设计之一。标准的ML安全技术在DRL中并不有效，在DRL中，算法通过与环境的持续交互来学习解决顺序决策，并且环境是时变的、动态的和移动的。针对物联网中基于Sybil的数据中毒攻击，提出了一种基于门控递归单元(GRU)的联合连续学习(GFCL)异常检测框架。其目的是提供一个轻量级和可扩展的框架，在没有包含攻击样本的先验训练数据集的情况下学习和检测非法行为。我们使用GRU来预测未来的数据序列，以基于联合学习的分布式方式来分析和检测车辆的非法行为。我们使用真实世界的车辆移动轨迹来研究我们的框架的性能。结果表明，在不同的性能指标下，我们提出的解决方案是有效的。



## **19. Generate novel and robust samples from data: accessible sharing without privacy concerns**

从数据中生成新颖且可靠的样本：无隐私问题的可访问共享 cs.LG

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.06113v1)

**Authors**: David Banh, Alan Huang

**Abstracts**: Generating new samples from data sets can mitigate extra expensive operations, increased invasive procedures, and mitigate privacy issues. These novel samples that are statistically robust can be used as a temporary and intermediate replacement when privacy is a concern. This method can enable better data sharing practices without problems relating to identification issues or biases that are flaws for an adversarial attack.

摘要: 从数据集生成新样本可以减少额外昂贵的操作、增加侵入性程序，并缓解隐私问题。当隐私受到关注时，这些在统计上稳健的新样本可以用作临时和中间替代。这种方法可以实现更好的数据共享实践，而不会出现与识别问题或作为对抗性攻击缺陷的偏见有关的问题。



## **20. Resisting Deep Learning Models Against Adversarial Attack Transferability via Feature Randomization**

基于特征随机化的抗敌意攻击传递的深度学习模型 cs.CR

**SubmitDate**: 2022-09-11    [paper-pdf](http://arxiv.org/pdf/2209.04930v1)

**Authors**: Ehsan Nowroozi, Mohammadreza Mohammadi, Pargol Golmohammadi, Yassine Mekdad, Mauro Conti, Selcuk Uluagac

**Abstracts**: In the past decades, the rise of artificial intelligence has given us the capabilities to solve the most challenging problems in our day-to-day lives, such as cancer prediction and autonomous navigation. However, these applications might not be reliable if not secured against adversarial attacks. In addition, recent works demonstrated that some adversarial examples are transferable across different models. Therefore, it is crucial to avoid such transferability via robust models that resist adversarial manipulations. In this paper, we propose a feature randomization-based approach that resists eight adversarial attacks targeting deep learning models in the testing phase. Our novel approach consists of changing the training strategy in the target network classifier and selecting random feature samples. We consider the attacker with a Limited-Knowledge and Semi-Knowledge conditions to undertake the most prevalent types of adversarial attacks. We evaluate the robustness of our approach using the well-known UNSW-NB15 datasets that include realistic and synthetic attacks. Afterward, we demonstrate that our strategy outperforms the existing state-of-the-art approach, such as the Most Powerful Attack, which consists of fine-tuning the network model against specific adversarial attacks. Finally, our experimental results show that our methodology can secure the target network and resists adversarial attack transferability by over 60%.

摘要: 在过去的几十年里，人工智能的崛起给了我们解决日常生活中最具挑战性的问题的能力，比如癌症预测和自主导航。但是，如果不确保这些应用程序不能抵御敌意攻击，则这些应用程序可能不可靠。此外，最近的工作表明，一些对抗性例子可以在不同的模型之间转移。因此，通过稳健的模型抵抗敌意操纵来避免这种可转移性是至关重要的。在本文中，我们提出了一种基于特征随机化的方法，该方法在测试阶段抵抗了八种针对深度学习模型的对抗性攻击。我们的新方法包括改变目标网络分类器中的训练策略和随机选择特征样本。我们认为攻击者在有限知识和半知识条件下可以进行最常见的对抗性攻击类型。我们使用著名的UNSW-NB15数据集评估了我们方法的健壮性，其中包括现实攻击和合成攻击。之后，我们证明了我们的策略优于现有的最先进的方法，例如最强大的攻击，它包括针对特定的对手攻击对网络模型进行微调。实验结果表明，该方法能够保证目标网络的安全，并能抵抗60%以上的敌意攻击可转移性。



## **21. Detecting Adversarial Perturbations in Multi-Task Perception**

多任务感知中的对抗性扰动检测 cs.CV

Accepted at IROS 2022

**SubmitDate**: 2022-09-11    [paper-pdf](http://arxiv.org/pdf/2203.01177v2)

**Authors**: Marvin Klingner, Varun Ravi Kumar, Senthil Yogamani, Andreas Bär, Tim Fingscheidt

**Abstracts**: While deep neural networks (DNNs) achieve impressive performance on environment perception tasks, their sensitivity to adversarial perturbations limits their use in practical applications. In this paper, we (i) propose a novel adversarial perturbation detection scheme based on multi-task perception of complex vision tasks (i.e., depth estimation and semantic segmentation). Specifically, adversarial perturbations are detected by inconsistencies between extracted edges of the input image, the depth output, and the segmentation output. To further improve this technique, we (ii) develop a novel edge consistency loss between all three modalities, thereby improving their initial consistency which in turn supports our detection scheme. We verify our detection scheme's effectiveness by employing various known attacks and image noises. In addition, we (iii) develop a multi-task adversarial attack, aiming at fooling both tasks as well as our detection scheme. Experimental evaluation on the Cityscapes and KITTI datasets shows that under an assumption of a 5% false positive rate up to 100% of images are correctly detected as adversarially perturbed, depending on the strength of the perturbation. Code is available at https://github.com/ifnspaml/AdvAttackDet. A short video at https://youtu.be/KKa6gOyWmH4 provides qualitative results.

摘要: 虽然深度神经网络(DNN)在环境感知任务中取得了令人印象深刻的性能，但它们对对抗性扰动的敏感性限制了它们在实际应用中的应用。本文(I)提出了一种基于复杂视觉任务多任务感知(深度估计和语义分割)的对抗性扰动检测方法。具体地，通过所提取的输入图像的边缘、深度输出和分割输出之间的不一致来检测对抗性扰动。为了进一步改进这一技术，我们(Ii)在所有三种模式之间开发了一种新的边缘一致性损失，从而改善了它们的初始一致性，这反过来又支持我们的检测方案。通过使用各种已知攻击和图像噪声来验证我们的检测方案的有效性。此外，我们(Iii)开发了一种多任务对抗性攻击，旨在愚弄两个任务和我们的检测方案。在CITYSCAPES和KITTI数据集上的实验评估表明，在假阳性率为5%的假设下，高达100%的图像被正确检测为恶意扰动，这取决于扰动的强度。代码可在https://github.com/ifnspaml/AdvAttackDet.上找到Https://youtu.be/KKa6gOyWmH4上的一段简短视频提供了定性的结果。



## **22. Robustness through Cognitive Dissociation Mitigation in Contrastive Adversarial Training**

对比性对抗训练中认知分离缓解的稳健性 cs.LG

Accepted to ICMLC 2022

**SubmitDate**: 2022-09-11    [paper-pdf](http://arxiv.org/pdf/2203.08959v3)

**Authors**: Adir Rahamim, Itay Naeh

**Abstracts**: In this paper, we introduce a novel neural network training framework that increases model's adversarial robustness to adversarial attacks while maintaining high clean accuracy by combining contrastive learning (CL) with adversarial training (AT). We propose to improve model robustness to adversarial attacks by learning feature representations that are consistent under both data augmentations and adversarial perturbations. We leverage contrastive learning to improve adversarial robustness by considering an adversarial example as another positive example, and aim to maximize the similarity between random augmentations of data samples and their adversarial example, while constantly updating the classification head in order to avoid a cognitive dissociation between the classification head and the embedding space. This dissociation is caused by the fact that CL updates the network up to the embedding space, while freezing the classification head which is used to generate new positive adversarial examples. We validate our method, Contrastive Learning with Adversarial Features(CLAF), on the CIFAR-10 dataset on which it outperforms both robust accuracy and clean accuracy over alternative supervised and self-supervised adversarial learning methods.

摘要: 本文提出了一种新的神经网络训练框架，通过将对比学习(CL)和对抗训练(AT)相结合，在保持较高精度的同时，提高了模型对对手攻击的鲁棒性。我们提出通过学习在数据扩充和对抗性扰动下都是一致的特征表示来提高模型对对抗性攻击的稳健性。我们利用对比学习来提高对抗性样本的稳健性，将一个对抗性样本作为另一个正例，目标是最大化随机增加的数据样本与其对抗性样本之间的相似度，同时不断更新分类头，以避免分类头与嵌入空间之间的认知分离。这种分离是由于CL将网络更新到嵌入空间，同时冻结用于生成新的正面对抗性实例的分类头。我们在CIFAR-10数据集上验证了我们的方法，即带有对抗性特征的对比学习(CLAF)，在CIFAR-10数据集上，它的性能优于其他监督和自我监督对抗性学习方法的稳健准确率和干净准确率。



## **23. Scattering Model Guided Adversarial Examples for SAR Target Recognition: Attack and Defense**

散射模型制导的SAR目标识别对抗实例：攻防 cs.CV

**SubmitDate**: 2022-09-11    [paper-pdf](http://arxiv.org/pdf/2209.04779v1)

**Authors**: Bowen Peng, Bo Peng, Jie Zhou, Jianyue Xie, Li Liu

**Abstracts**: Deep Neural Networks (DNNs) based Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR) systems have shown to be highly vulnerable to adversarial perturbations that are deliberately designed yet almost imperceptible but can bias DNN inference when added to targeted objects. This leads to serious safety concerns when applying DNNs to high-stake SAR ATR applications. Therefore, enhancing the adversarial robustness of DNNs is essential for implementing DNNs to modern real-world SAR ATR systems. Toward building more robust DNN-based SAR ATR models, this article explores the domain knowledge of SAR imaging process and proposes a novel Scattering Model Guided Adversarial Attack (SMGAA) algorithm which can generate adversarial perturbations in the form of electromagnetic scattering response (called adversarial scatterers). The proposed SMGAA consists of two parts: 1) a parametric scattering model and corresponding imaging method and 2) a customized gradient-based optimization algorithm. First, we introduce the effective Attributed Scattering Center Model (ASCM) and a general imaging method to describe the scattering behavior of typical geometric structures in the SAR imaging process. By further devising several strategies to take the domain knowledge of SAR target images into account and relax the greedy search procedure, the proposed method does not need to be prudentially finetuned, but can efficiently to find the effective ASCM parameters to fool the SAR classifiers and facilitate the robust model training. Comprehensive evaluations on the MSTAR dataset show that the adversarial scatterers generated by SMGAA are more robust to perturbations and transformations in the SAR processing chain than the currently studied attacks, and are effective to construct a defensive model against the malicious scatterers.

摘要: 基于深度神经网络(DNN)的合成孔径雷达(SAR)自动目标识别(ATR)系统被证明是非常容易受到敌意扰动的，这些扰动是故意设计的，但几乎不可察觉，但当添加到目标对象时，会使DNN推理产生偏差。这导致在将DNN应用于高风险的SAR ATR应用时存在严重的安全问题。因此，增强DNN的对抗健壮性对于将DNN应用于现代真实的SARATR系统是至关重要的。为了建立更稳健的基于离散神经网络的合成孔径雷达ATR模型，本文深入研究了合成孔径雷达成像过程中的领域知识，提出了一种新的散射模型制导对抗攻击算法(SMGAA)，该算法可以产生电磁散射响应形式的对抗扰动(称为对抗散射者)。SMGAA由两部分组成：1)参数散射模型和相应的成像方法；2)定制的基于梯度的优化算法。首先，我们引入了有效属性散射中心模型(ASCM)和一种通用的成像方法来描述合成孔径雷达成像过程中典型几何结构的散射行为。通过进一步设计几种策略来考虑SAR目标图像的领域知识，并放松贪婪搜索过程，该方法不需要谨慎地精调，但可以有效地找到有效的ASCM参数来愚弄SAR分类器，便于稳健的模型训练。在MStar数据集上的综合评估表明，SMGAA生成的敌意散射体对SAR处理链中的扰动和变换具有更强的鲁棒性，并且能够有效地构建针对恶意散射体的防御模型。



## **24. Atomic cross-chain exchanges of shared assets**

共享资产的原子跨链交换 cs.CR

**SubmitDate**: 2022-09-10    [paper-pdf](http://arxiv.org/pdf/2202.12855v3)

**Authors**: Krishnasuri Narayanam, Venkatraman Ramakrishna, Dhinakaran Vinayagamurthy, Sandeep Nishad

**Abstracts**: A core enabler for blockchain or DLT interoperability is the ability to atomically exchange assets held by mutually untrusting owners on different ledgers. This atomic swap problem has been well-studied, with the Hash Time Locked Contract (HTLC) emerging as a canonical solution. HTLC ensures atomicity of exchange, albeit with caveats for node failure and timeliness of claims. But a bigger limitation of HTLC is that it only applies to a model consisting of two adversarial parties having sole ownership of a single asset in each ledger. Realistic extensions of the model in which assets may be jointly owned by multiple parties, all of whose consents are required for exchanges, or where multiple assets must be exchanged for one, are susceptible to collusion attacks and hence cannot be handled by HTLC. In this paper, we generalize the model of asset exchanges across DLT networks and present a taxonomy of use cases, describe the threat model, and propose MPHTLC, an augmented HTLC protocol for atomic multi-owner-and-asset exchanges. We analyze the correctness, safety, and application scope of MPHTLC. As proof-of-concept, we show how MPHTLC primitives can be implemented in networks built on Hyperledger Fabric and Corda, and how MPHTLC can be implemented in the Hyperledger Labs Weaver framework by augmenting its existing HTLC protocol.

摘要: 区块链或DLT互操作性的核心推动因素是能够自动交换不同分类账上相互不信任的所有者持有的资产。这个原子交换问题已经得到了很好的研究，哈希时间锁定合同(HTLC)成为一种规范的解决方案。HTLC确保了交换的原子性，但对节点故障和索赔的及时性提出了警告。但HTLC的一个更大限制是，它只适用于由两个对立方单独拥有每个分类账中的一项资产的模型。资产可能由多方共同拥有的模型的现实扩展，其中交易需要所有各方的同意，或者必须用多个资产交换一个资产，这容易受到共谋攻击，因此无法由HTLC处理。本文对跨DLT网络的资产交换模型进行了推广，给出了用例的分类，描述了威胁模型，并提出了一种用于原子多所有者和资产交换的扩展HTLC协议MPHTLC。分析了MPHTLC的正确性、安全性和适用范围。作为概念验证，我们展示了如何在建立在Hyperledger Fabric和Corda上的网络中实现MPHTLC原语，以及如何通过增强Hyperledger Labs Weaver框架的现有HTLC协议来实现MPHTLC。



## **25. Phantom Sponges: Exploiting Non-Maximum Suppression to Attack Deep Object Detectors**

幻影海绵：利用非最大抑制攻击深度对象探测器 cs.CV

**SubmitDate**: 2022-09-10    [paper-pdf](http://arxiv.org/pdf/2205.13618v2)

**Authors**: Avishag Shapira, Alon Zolfi, Luca Demetrio, Battista Biggio, Asaf Shabtai

**Abstracts**: Adversarial attacks against deep learning-based object detectors have been studied extensively in the past few years. Most of the attacks proposed have targeted the model's integrity (i.e., caused the model to make incorrect predictions), while adversarial attacks targeting the model's availability, a critical aspect in safety-critical domains such as autonomous driving, have not yet been explored by the machine learning research community. In this paper, we propose a novel attack that negatively affects the decision latency of an end-to-end object detection pipeline. We craft a universal adversarial perturbation (UAP) that targets a widely used technique integrated in many object detector pipelines -- non-maximum suppression (NMS). Our experiments demonstrate the proposed UAP's ability to increase the processing time of individual frames by adding "phantom" objects that overload the NMS algorithm while preserving the detection of the original objects (which allows the attack to go undetected for a longer period of time).

摘要: 针对基于深度学习的目标检测器的对抗性攻击在过去的几年中得到了广泛的研究。大多数提出的攻击都是针对模型的完整性(即导致模型做出错误的预测)，而针对模型可用性的对抗性攻击(自动驾驶等安全关键领域的一个关键方面)尚未被机器学习研究社区探索。在本文中，我们提出了一种新的攻击，它对端到端对象检测流水线的决策延迟产生负面影响。我们设计了一种通用对抗摄动(UAP)，目标是集成在许多对象探测器流水线中的一种广泛使用的技术--非最大抑制(NMS)。我们的实验证明了所提出的UAP能够通过添加超载NMS算法的“幻影”对象来增加单个帧的处理时间，同时保持对原始对象的检测(这使得攻击在更长的时间内不被检测到)。



## **26. Bankrupting DoS Attackers Despite Uncertainty**

尽管存在不确定性，但仍使DoS攻击者破产 cs.CR

**SubmitDate**: 2022-09-10    [paper-pdf](http://arxiv.org/pdf/2205.08287v2)

**Authors**: Trisha Chakraborty, Abir Islam, Valerie King, Daniel Rayborn, Jared Saia, Maxwell Young

**Abstracts**: On-demand provisioning in the cloud allows for services to remain available despite massive denial-of-service (DoS) attacks. Unfortunately, on-demand provisioning is expensive and must be weighed against the costs incurred by an adversary. This leads to a recent threat known as {\it economic denial-of-sustainability (EDoS)}, where the cost for defending a service is higher than that of attacking.   A natural tool for combating EDoS is to impose costs via resource burning (RB). Here, a client must verifiably consume resources -- for example, by solving a computational challenge -- before service is rendered. However, prior RB-based defenses with security guarantees do not account for the cost of on-demand provisioning.   Another common approach is the use of heuristics -- such as a client's reputation score or the geographical location -- to identify and discard spurious job requests. However, these heuristics may err and existing approaches do not provide security guarantees when this occurs.   Here, we propose an EDoS defense, LCharge, that uses resource burning while accounting for on-demand provisioning. LCharge leverages an estimate of the number of job requests from honest clients (i.e., good jobs) in any set $S$ of requests to within an $O(\alpha)$-factor, for any unknown $\alpha>0$, but retains a strong security guarantee despite the uncertainty of this estimate. Specifically, against an adversary that expends $B$ resources to attack, the total cost for defending is $O( \alpha^{5/2}\sqrt{B\,(g+1)} + \alpha^3(g+\alpha))$ where $g$ is the number of good jobs. Notably, for large $B$ relative to $g$ and $\alpha$, the adversary has higher cost, implying that the algorithm has an economic advantage. Finally, we prove a lower bound for our problem of $\Omega(\sqrt{\alpha B g})$, showing that the cost of LCharge is asymptotically tight for $\alpha=\Theta(1)$.

摘要: 云中的按需配置允许服务在遭受大规模拒绝服务(DoS)攻击时保持可用。不幸的是，按需配置的成本很高，必须权衡对手所产生的成本。这导致了最近的一种称为{\it经济拒绝可持续性(EDOS)}的威胁，在这种威胁下，防御服务的成本高于攻击。对抗EDO的一个自然工具是通过资源燃烧(RB)来施加成本。在这里，在提供服务之前，客户端必须可验证地消耗资源--例如，通过解决计算挑战。然而，以前的基于RB的安全保证防御不会考虑按需配置的成本。另一种常见的方法是使用试探法--例如客户的声誉分数或地理位置--来识别和丢弃虚假的工作请求。然而，这些启发式方法可能会出错，并且现有方法在发生这种情况时不提供安全保证。在这里，我们提出了EDOS防御LCharge，它在考虑按需配置的同时使用资源消耗。LCharge利用任何$S$请求集合中来自诚实客户(即好工作)的工作请求数量的估计，对于任何未知的$\α>0$，在$O(\Alpha)$因子内，但尽管该估计存在不确定性，LCharge仍保持强大的安全保证。具体地说，针对花费$B$资源进行攻击的对手，防御的总成本为$O(\alpha^{5/2}\sqrt{B\，(g+1)}+\alpha^3(g+\pha))$，其中$g$是好工作的数量。值得注意的是，对于较大的$B$相对于$g$和$\α$，对手具有更高的成本，这意味着该算法具有经济优势。最后，我们证明了问题$Omega(\Sqrt{\Alpha Bg})$的一个下界，证明了LCharge的代价对于$\α=\Theta(1)$是渐近紧的。



## **27. The Space of Adversarial Strategies**

对抗性策略的空间 cs.CR

Accepted to the 32nd USENIX Security Symposium

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2209.04521v1)

**Authors**: Ryan Sheatsley, Blaine Hoak, Eric Pauley, Patrick McDaniel

**Abstracts**: Adversarial examples, inputs designed to induce worst-case behavior in machine learning models, have been extensively studied over the past decade. Yet, our understanding of this phenomenon stems from a rather fragmented pool of knowledge; at present, there are a handful of attacks, each with disparate assumptions in threat models and incomparable definitions of optimality. In this paper, we propose a systematic approach to characterize worst-case (i.e., optimal) adversaries. We first introduce an extensible decomposition of attacks in adversarial machine learning by atomizing attack components into surfaces and travelers. With our decomposition, we enumerate over components to create 576 attacks (568 of which were previously unexplored). Next, we propose the Pareto Ensemble Attack (PEA): a theoretical attack that upper-bounds attack performance. With our new attacks, we measure performance relative to the PEA on: both robust and non-robust models, seven datasets, and three extended lp-based threat models incorporating compute costs, formalizing the Space of Adversarial Strategies. From our evaluation we find that attack performance to be highly contextual: the domain, model robustness, and threat model can have a profound influence on attack efficacy. Our investigation suggests that future studies measuring the security of machine learning should: (1) be contextualized to the domain & threat models, and (2) go beyond the handful of known attacks used today.

摘要: 对抗性例子是机器学习模型中旨在诱导最坏情况行为的输入，在过去的十年里得到了广泛的研究。然而，我们对这一现象的理解源于相当零散的知识池；目前，有几种攻击，每一种攻击在威胁模型中都有不同的假设，对最优的定义也是无与伦比的。在本文中，我们提出了一种系统的方法来刻画最坏情况(即最佳)对手的特征。我们首先介绍了对抗性机器学习中攻击的一种可扩展分解，将攻击组件原子化到表面和旅行者中。通过我们的分解，我们列举了组件以创建576次攻击(其中568次是以前未曾探索过的)。接下来，我们提出了Pareto系综攻击(PEA)：一种上界攻击性能的理论攻击。在我们的新攻击中，我们在以下方面衡量相对于PEA的性能：稳健和非稳健模型、七个数据集和三个包含计算成本的扩展的基于LP的威胁模型，正式确定了对手战略空间。从我们的评估中，我们发现攻击性能与上下文高度相关：域、模型健壮性和威胁模型可以对攻击效率产生深远的影响。我们的调查表明，未来衡量机器学习安全性的研究应该：(1)从域和威胁模型出发，(2)超越目前使用的少数已知攻击。



## **28. SoK: Certified Robustness for Deep Neural Networks**

SOK：深度神经网络认证的健壮性 cs.LG

To appear at 2023 IEEE Symposium on Security and Privacy (SP); 14  pages for the main text; benchmark & tool website:  http://sokcertifiedrobustness.github.io/

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2009.04131v8)

**Authors**: Linyi Li, Tao Xie, Bo Li

**Abstracts**: Great advances in deep neural networks (DNNs) have led to state-of-the-art performance on a wide range of tasks. However, recent studies have shown that DNNs are vulnerable to adversarial attacks, which have brought great concerns when deploying these models to safety-critical applications such as autonomous driving. Different defense approaches have been proposed against adversarial attacks, including: a) empirical defenses, which can usually be adaptively attacked again without providing robustness certification; and b) certifiably robust approaches, which consist of robustness verification providing the lower bound of robust accuracy against any attacks under certain conditions and corresponding robust training approaches. In this paper, we systematize certifiably robust approaches and related practical and theoretical implications and findings. We also provide the first comprehensive benchmark on existing robustness verification and training approaches on different datasets. In particular, we 1) provide a taxonomy for the robustness verification and training approaches, as well as summarize the methodologies for representative algorithms, 2) reveal the characteristics, strengths, limitations, and fundamental connections among these approaches, 3) discuss current research progresses, theoretical barriers, main challenges, and future directions for certifiably robust approaches for DNNs, and 4) provide an open-sourced unified platform to evaluate 20+ representative certifiably robust approaches.

摘要: 深度神经网络(DNN)的巨大进步导致了在各种任务中最先进的性能。然而，最近的研究表明，DNN很容易受到对手攻击，这在将这些模型部署到自动驾驶等安全关键型应用时带来了极大的担忧。针对敌意攻击已经提出了不同的防御方法，包括：a)经验防御，通常可以在不提供健壮性证明的情况下自适应地再次攻击；b)可证明的健壮性方法，包括在一定条件下提供对任何攻击的健壮性精度下界的健壮性验证和相应的健壮训练方法。在这篇文章中，我们系统化了可证明的稳健方法以及相关的实践和理论意义和发现。我们还提供了关于不同数据集上现有稳健性验证和训练方法的第一个全面基准。具体地说，我们1)提供了健壮性验证和训练方法的分类，并总结了典型算法的方法论；2)揭示了这些方法的特点、优势、局限性和基本联系；3)讨论了当前的研究进展、理论障碍、主要挑战和未来的发展方向；4)提供了一个开源的统一平台来评估20多个具有代表性的可证健壮方法。



## **29. Adversarial Examples in Constrained Domains**

受限领域中的对抗性例子 cs.CR

Accepted to IOS Press Journal of Computer Security

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2011.01183v3)

**Authors**: Ryan Sheatsley, Nicolas Papernot, Michael Weisman, Gunjan Verma, Patrick McDaniel

**Abstracts**: Machine learning algorithms have been shown to be vulnerable to adversarial manipulation through systematic modification of inputs (e.g., adversarial examples) in domains such as image recognition. Under the default threat model, the adversary exploits the unconstrained nature of images; each feature (pixel) is fully under control of the adversary. However, it is not clear how these attacks translate to constrained domains that limit which and how features can be modified by the adversary (e.g., network intrusion detection). In this paper, we explore whether constrained domains are less vulnerable than unconstrained domains to adversarial example generation algorithms. We create an algorithm for generating adversarial sketches: targeted universal perturbation vectors which encode feature saliency within the envelope of domain constraints. To assess how these algorithms perform, we evaluate them in constrained (e.g., network intrusion detection) and unconstrained (e.g., image recognition) domains. The results demonstrate that our approaches generate misclassification rates in constrained domains that were comparable to those of unconstrained domains (greater than 95%). Our investigation shows that the narrow attack surface exposed by constrained domains is still sufficiently large to craft successful adversarial examples; and thus, constraints do not appear to make a domain robust. Indeed, with as little as five randomly selected features, one can still generate adversarial examples.

摘要: 已经证明，机器学习算法通过系统地修改诸如图像识别等领域中的输入(例如，对抗性例子)而容易受到对抗性操纵。在默认威胁模型下，对手利用图像的不受限制的性质；每个特征(像素)都完全在对手的控制之下。然而，目前尚不清楚这些攻击如何转化为受限的域，从而限制攻击者可以修改哪些功能以及如何修改(例如，网络入侵检测)。在本文中，我们探讨了受限域是否比非约束域更不容易受到敌意示例生成算法的影响。我们创建了一种生成对抗性草图的算法：目标通用扰动向量，它在领域约束的包络内编码特征显著。为了评估这些算法的性能，我们在受限(例如，网络入侵检测)和非受限(例如，图像识别)域中对它们进行评估。结果表明，我们的方法在受限领域产生的错误分类率与非约束领域相当(大于95%)。我们的调查表明，受约束域暴露的狭窄攻击面仍然足够大，足以制作成功的敌意示例；因此，约束似乎不会使域变得健壮。事实上，只需随机选择五个特征，就仍然可以生成对抗性的例子。



## **30. Robust-by-Design Classification via Unitary-Gradient Neural Networks**

基于么正梯度神经网络的稳健设计分类 cs.LG

Under review

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2209.04293v1)

**Authors**: Fabio Brau, Giulio Rossolini, Alessandro Biondi, Giorgio Buttazzo

**Abstracts**: The use of neural networks in safety-critical systems requires safe and robust models, due to the existence of adversarial attacks. Knowing the minimal adversarial perturbation of any input x, or, equivalently, knowing the distance of x from the classification boundary, allows evaluating the classification robustness, providing certifiable predictions. Unfortunately, state-of-the-art techniques for computing such a distance are computationally expensive and hence not suited for online applications. This work proposes a novel family of classifiers, namely Signed Distance Classifiers (SDCs), that, from a theoretical perspective, directly output the exact distance of x from the classification boundary, rather than a probability score (e.g., SoftMax). SDCs represent a family of robust-by-design classifiers. To practically address the theoretical requirements of a SDC, a novel network architecture named Unitary-Gradient Neural Network is presented. Experimental results show that the proposed architecture approximates a signed distance classifier, hence allowing an online certifiable classification of x at the cost of a single inference.

摘要: 由于存在对抗性攻击，在安全关键系统中使用神经网络需要安全和健壮的模型。知道任何输入x的最小对抗性扰动，或者，等价地，知道x到分类边界的距离，允许评估分类稳健性，提供可证明的预测。不幸的是，用于计算这种距离的最先进的技术计算成本很高，因此不适合在线应用。这项工作提出了一类新的分类器，即符号距离分类器(SDCS)，从理论上讲，它直接输出x到分类边界的准确距离，而不是概率分数(例如SoftMax)。SDC代表了一系列稳健的按设计分类的分类器。为了满足SDC的理论要求，提出了一种新的网络体系结构--酉梯度神经网络。实验结果表明，该体系结构接近于符号距离分类器，从而允许以单一推理为代价对x进行在线可证明分类。



## **31. Improving Out-of-Distribution Detection via Epistemic Uncertainty Adversarial Training**

通过认知不确定性对抗性训练改进失配检测 cs.LG

8 pages, 5 figures

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2209.03148v2)

**Authors**: Derek Everett, Andre T. Nguyen, Luke E. Richards, Edward Raff

**Abstracts**: The quantification of uncertainty is important for the adoption of machine learning, especially to reject out-of-distribution (OOD) data back to human experts for review. Yet progress has been slow, as a balance must be struck between computational efficiency and the quality of uncertainty estimates. For this reason many use deep ensembles of neural networks or Monte Carlo dropout for reasonable uncertainty estimates at relatively minimal compute and memory. Surprisingly, when we focus on the real-world applicable constraint of $\leq 1\%$ false positive rate (FPR), prior methods fail to reliably detect OOD samples as such. Notably, even Gaussian random noise fails to trigger these popular OOD techniques. We help to alleviate this problem by devising a simple adversarial training scheme that incorporates an attack of the epistemic uncertainty predicted by the dropout ensemble. We demonstrate this method improves OOD detection performance on standard data (i.e., not adversarially crafted), and improves the standardized partial AUC from near-random guessing performance to $\geq 0.75$.

摘要: 不确定性的量化对于机器学习的采用非常重要，特别是对于拒绝将分布外(OOD)数据返回给人类专家进行审查。然而，进展缓慢，因为必须在计算效率和不确定性估计的质量之间取得平衡。出于这个原因，许多人使用神经网络的深度集成或蒙特卡罗退学来在相对最小的计算和内存下进行合理的不确定性估计。令人惊讶的是，当我们关注现实世界中可应用的假阳性率(FPR)约束时，现有方法无法可靠地检测出OOD样本。值得注意的是，即使是高斯随机噪声也无法触发这些流行的OOD技术。我们通过设计一个简单的对抗性训练方案来帮助缓解这个问题，该方案结合了对辍学生群体预测的认知不确定性的攻击。我们证明了该方法提高了对标准数据的OOD检测性能(即，不是恶意定制的)，并将标准化的部分AUC从近乎随机猜测的性能提高到0.75美元。



## **32. Harnessing Perceptual Adversarial Patches for Crowd Counting**

利用感知对抗性斑块进行人群计数 cs.CV

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2109.07986v2)

**Authors**: Shunchang Liu, Jiakai Wang, Aishan Liu, Yingwei Li, Yijie Gao, Xianglong Liu, Dacheng Tao

**Abstracts**: Crowd counting, which has been widely adopted for estimating the number of people in safety-critical scenes, is shown to be vulnerable to adversarial examples in the physical world (e.g., adversarial patches). Though harmful, adversarial examples are also valuable for evaluating and better understanding model robustness. However, existing adversarial example generation methods for crowd counting lack strong transferability among different black-box models, which limits their practicability for real-world systems. Motivated by the fact that attacking transferability is positively correlated to the model-invariant characteristics, this paper proposes the Perceptual Adversarial Patch (PAP) generation framework to tailor the adversarial perturbations for crowd counting scenes using the model-shared perceptual features. Specifically, we handcraft an adaptive crowd density weighting approach to capture the invariant scale perception features across various models and utilize the density guided attention to capture the model-shared position perception. Both of them are demonstrated to improve the attacking transferability of our adversarial patches. Extensive experiments show that our PAP could achieve state-of-the-art attacking performance in both the digital and physical world, and outperform previous proposals by large margins (at most +685.7 MAE and +699.5 MSE). Besides, we empirically demonstrate that adversarial training with our PAP can benefit the performance of vanilla models in alleviating several practical challenges in crowd counting scenarios, including generalization across datasets (up to -376.0 MAE and -354.9 MSE) and robustness towards complex backgrounds (up to -10.3 MAE and -16.4 MSE).

摘要: 人群计数被广泛用于估计安全关键场景中的人数，但在现实世界中，它很容易受到对抗性例子的影响(例如，对抗性补丁)。对抗性例子虽然有害，但对于评估和更好地理解模型的健壮性也是有价值的。然而，现有的人群计数对抗性实例生成方法在不同的黑盒模型之间缺乏很强的可移植性，这限制了它们在现实系统中的实用性。基于攻击的可转移性与模型不变特性正相关这一事实，提出了感知对抗性补丁(PAP)生成框架，利用模型共享的感知特征来定制人群计数场景中的对抗性扰动。具体地说，我们手工设计了一种自适应的人群密度加权方法来捕捉各种模型上的不变尺度感知特征，并利用密度引导注意力来捕捉模型共享的位置感知。它们都被证明可以提高我们对手补丁的攻击可转移性。大量的实验表明，我们的PAP在数字和物理世界都可以达到最先进的攻击性能，并且比以前的方案有很大的优势(最多+685.7 MAE和+699.5 MSE)。此外，我们的经验证明，使用我们的PAP进行对抗性训练可以帮助Vanilla模型在缓解人群计数场景中的几个实际挑战方面的性能，包括跨数据集的泛化(高达-376.0 MAE和-354.9 MSE)以及对复杂背景的稳健性(高达-10.3MAE和-16.4MSE)。



## **33. Uncovering the Connection Between Differential Privacy and Certified Robustness of Federated Learning against Poisoning Attacks**

揭示差分隐私与联合学习对中毒攻击的认证健壮性之间的联系 cs.CR

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2209.04030v1)

**Authors**: Chulin Xie, Yunhui Long, Pin-Yu Chen, Bo Li

**Abstracts**: Federated learning (FL) provides an efficient paradigm to jointly train a global model leveraging data from distributed users. As the local training data come from different users who may not be trustworthy, several studies have shown that FL is vulnerable to poisoning attacks. Meanwhile, to protect the privacy of local users, FL is always trained in a differentially private way (DPFL). Thus, in this paper, we ask: Can we leverage the innate privacy property of DPFL to provide certified robustness against poisoning attacks? Can we further improve the privacy of FL to improve such certification? We first investigate both user-level and instance-level privacy of FL and propose novel mechanisms to achieve improved instance-level privacy. We then provide two robustness certification criteria: certified prediction and certified attack cost for DPFL on both levels. Theoretically, we prove the certified robustness of DPFL under a bounded number of adversarial users or instances. Empirically, we conduct extensive experiments to verify our theories under a range of attacks on different datasets. We show that DPFL with a tighter privacy guarantee always provides stronger robustness certification in terms of certified attack cost, but the optimal certified prediction is achieved under a proper balance between privacy protection and utility loss.

摘要: 联合学习(FL)提供了一种有效的范例来联合训练利用来自分布式用户的数据的全局模型。由于本地训练数据来自可能不可信的不同用户，多项研究表明FL很容易受到中毒攻击。同时，为了保护本地用户的隐私，FL总是以一种不同的私人方式进行培训(DPFL)。因此，在这篇文章中，我们问：我们能否利用DPFL固有的隐私属性来提供经过认证的针对中毒攻击的健壮性？我们能否进一步改善FL的隐私，以提高此类认证？我们首先研究了FL的用户级隐私和实例级隐私，并提出了新的机制来实现改进的实例级隐私。然后，我们提供了两个健壮性认证标准：DPFL在两个级别上的认证预测和认证攻击成本。理论上，我们证明了DPFL在有限数量的敌意用户或实例下的证明的健壮性。在经验上，我们在不同数据集的一系列攻击下进行了广泛的实验来验证我们的理论。我们发现，在认证攻击代价方面，具有更严格隐私保障的DPFL总是提供更强的健壮性认证，但最优认证预测是在隐私保护和效用损失之间取得适当平衡的情况下实现的。



## **34. Evaluating the Security of Aircraft Systems**

评估飞机系统的安全性 cs.CR

38 pages,

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2209.04028v1)

**Authors**: Edan Habler, Ron Bitton, Asaf Shabtai

**Abstracts**: The sophistication and complexity of cyber attacks and the variety of targeted platforms have been growing in recent years. Various adversaries are abusing an increasing range of platforms, e.g., enterprise platforms, mobile phones, PCs, transportation systems, and industrial control systems. In recent years, we have witnessed various cyber attacks on transportation systems, including attacks on ports, airports, and trains. It is only a matter of time before transportation systems become a more common target of cyber attackers. Due to the enormous potential damage inherent in attacking vehicles carrying many passengers and the lack of security measures applied in traditional airborne systems, the vulnerability of aircraft systems is one of the most concerning topics in the vehicle security domain. This paper provides a comprehensive review of aircraft systems and components and their various networks, emphasizing the cyber threats they are exposed to and the impact of a cyber attack on these components and networks and the essential capabilities of the aircraft. In addition, we present a comprehensive and in-depth taxonomy that standardizes the knowledge and understanding of cyber security in the avionics field from an adversary's perspective. The taxonomy divides techniques into relevant categories (tactics) reflecting the various phases of the adversarial attack lifecycle and maps existing attacks according to the MITRE ATT&CK methodology. Furthermore, we analyze the security risks among the various systems according to the potential threat actors and categorize the threats based on STRIDE threat model. Future work directions are presented as guidelines for industry and academia.

摘要: 近年来，网络攻击的复杂性和复杂性以及目标平台的多样性一直在增长。各种对手正在滥用越来越多的平台，例如企业平台、移动电话、PC、交通系统和工业控制系统。近年来，我们目睹了针对交通系统的各种网络攻击，包括对港口、机场和火车的攻击。交通系统成为网络攻击者更常见的目标只是个时间问题。由于攻击载客车辆固有的巨大潜在危害，以及传统机载系统缺乏安全措施，飞机系统的脆弱性是车辆安全领域最受关注的话题之一。本文对飞机系统和部件及其各种网络进行了全面的回顾，强调了它们所面临的网络威胁，以及网络攻击对这些部件和网络以及飞机的基本能力的影响。此外，我们提出了一个全面和深入的分类，从对手的角度标准化了对航空电子领域网络安全的知识和理解。该分类将技术划分为相关类别(战术)，反映对抗性攻击生命周期的不同阶段，并根据MITRE ATT&CK方法映射现有攻击。在此基础上，根据潜在威胁主体分析了各个系统之间的安全风险，并基于STRIDE威胁模型对威胁进行了分类。提出了未来的工作方向，作为产业界和学术界的指导方针。



## **35. SafeNet: The Unreasonable Effectiveness of Ensembles in Private Collaborative Learning**

SafeNet：私人合作学习中合奏的不合理有效性 cs.CR

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2205.09986v2)

**Authors**: Harsh Chaudhari, Matthew Jagielski, Alina Oprea

**Abstracts**: Secure multiparty computation (MPC) has been proposed to allow multiple mutually distrustful data owners to jointly train machine learning (ML) models on their combined data. However, by design, MPC protocols faithfully compute the training functionality, which the adversarial ML community has shown to leak private information and can be tampered with in poisoning attacks. In this work, we argue that model ensembles, implemented in our framework called SafeNet, are a highly MPC-amenable way to avoid many adversarial ML attacks. The natural partitioning of data amongst owners in MPC training allows this approach to be highly scalable at training time, provide provable protection from poisoning attacks, and provably defense against a number of privacy attacks. We demonstrate SafeNet's efficiency, accuracy, and resilience to poisoning on several machine learning datasets and models trained in end-to-end and transfer learning scenarios. For instance, SafeNet reduces backdoor attack success significantly, while achieving $39\times$ faster training and $36 \times$ less communication than the four-party MPC framework of Dalskov et al. Our experiments show that ensembling retains these benefits even in many non-iid settings. The simplicity, cheap setup, and robustness properties of ensembling make it a strong first choice for training ML models privately in MPC.

摘要: 安全多方计算(MPC)已被提出，以允许多个相互不信任的数据所有者联合训练机器学习(ML)模型。然而，通过设计，MPC协议忠实地计算训练功能，敌对的ML社区已经证明这些功能会泄露私人信息，并且可以在中毒攻击中篡改。在这项工作中，我们认为在我们的框架SafeNet中实现的模型集成是一种高度兼容MPC的方法，可以避免许多对抗性的ML攻击。在MPC培训中，所有者之间的数据自然分区允许此方法在培训期间高度可扩展，提供针对中毒攻击的可证明保护，并可证明针对多个隐私攻击的防御。我们在几个在端到端和转移学习场景中训练的机器学习数据集和模型上展示了SafeNet的效率、准确性和对中毒的弹性。例如，SafeNet显著降低了后门攻击的成功率，同时实现了比Dalskov等人的四方MPC框架快39倍的培训和36倍的通信。我们的实验表明，即使在许多非IID环境中，集合也保留了这些好处。集成的简单性、设置成本和健壮性使其成为在MPC中私下训练ML模型的首选方法。



## **36. Incorporating Locality of Images to Generate Targeted Transferable Adversarial Examples**

结合图像的局部性生成目标可转移的对抗性实例 cs.CV

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2209.03716v1)

**Authors**: Zhipeng Wei, Jingjing Chen, Zuxuan Wu, Yu-Gang Jiang

**Abstracts**: Despite that leveraging the transferability of adversarial examples can attain a fairly high attack success rate for non-targeted attacks, it does not work well in targeted attacks since the gradient directions from a source image to a targeted class are usually different in different DNNs. To increase the transferability of target attacks, recent studies make efforts in aligning the feature of the generated adversarial example with the feature distributions of the targeted class learned from an auxiliary network or a generative adversarial network. However, these works assume that the training dataset is available and require a lot of time to train networks, which makes it hard to apply to real-world scenarios. In this paper, we revisit adversarial examples with targeted transferability from the perspective of universality and find that highly universal adversarial perturbations tend to be more transferable. Based on this observation, we propose the Locality of Images (LI) attack to improve targeted transferability. Specifically, instead of using the classification loss only, LI introduces a feature similarity loss between intermediate features from adversarial perturbed original images and randomly cropped images, which makes the features from adversarial perturbations to be more dominant than that of benign images, hence improving targeted transferability. Through incorporating locality of images into optimizing perturbations, the LI attack emphasizes that targeted perturbations should be universal to diverse input patterns, even local image patches. Extensive experiments demonstrate that LI can achieve high success rates for transfer-based targeted attacks. On attacking the ImageNet-compatible dataset, LI yields an improvement of 12\% compared with existing state-of-the-art methods.

摘要: 尽管利用对抗性例子的可转移性可以在非目标攻击中获得相当高的攻击成功率，但它在目标攻击中不能很好地工作，因为在不同的DNN中，从源图像到目标类别的梯度方向通常是不同的。为了提高目标攻击的可转移性，最近的研究致力于将生成的对抗性实例的特征与从辅助网络或生成性对抗性网络学习的目标类的特征分布相匹配。然而，这些工作假设训练数据集是可用的，并且需要大量的时间来训练网络，这使得很难将其应用于现实世界的场景。在本文中，我们从普遍性的角度重新考察了具有针对性可转移性的对抗性例子，发现具有高度普遍性的对抗性扰动往往更具可转移性。基于这一观察结果，我们提出了图像局部性(LI)攻击来提高目标可转移性。具体地说，Li不是只使用分类损失，而是在来自对抗性扰动的原始图像和随机裁剪图像的中间特征之间引入了特征相似性损失，使得来自对抗性扰动的特征比良性图像的特征更具优势，从而提高了目标可转移性。通过将图像的局部性引入优化扰动，LI攻击强调目标扰动对于不同的输入模式应该是通用的，甚至对局部图像块也是如此。广泛的实验证明，李灿对基于转会的靶向攻击取得了很高的成功率。在攻击与ImageNet兼容的数据集方面，与现有的最先进方法相比，LI的性能提高了12%。



## **37. Exploring Adversarial Attacks and Defenses in Vision Transformers trained with DINO**

探索与恐龙一起训练的视觉变形金刚的对抗性攻击和防御 cs.CV

ICML 2022 Workshop paper accepted at AdvML Frontiers

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2206.06761v4)

**Authors**: Javier Rando, Nasib Naimi, Thomas Baumann, Max Mathys

**Abstracts**: This work conducts the first analysis on the robustness against adversarial attacks on self-supervised Vision Transformers trained using DINO. First, we evaluate whether features learned through self-supervision are more robust to adversarial attacks than those emerging from supervised learning. Then, we present properties arising for attacks in the latent space. Finally, we evaluate whether three well-known defense strategies can increase adversarial robustness in downstream tasks by only fine-tuning the classification head to provide robustness even in view of limited compute resources. These defense strategies are: Adversarial Training, Ensemble Adversarial Training and Ensemble of Specialized Networks.

摘要: 本文首次对使用Dino训练的自监督视觉转换器的抗敌意攻击能力进行了分析。首先，我们评估通过自我监督学习的特征是否比通过监督学习获得的特征对对手攻击更健壮。然后，我们给出了潜在空间中攻击产生的性质。最后，我们评估了三种著名的防御策略是否能够在下游任务中通过微调分类头来提高对手的健壮性，即使在计算资源有限的情况下也是如此。这些防御策略是：对抗性训练、系列性对抗性训练和专业网络系列化。



## **38. Feature Importance Guided Attack: A Model Agnostic Adversarial Attack**

特征重要性制导攻击：一种不可知的对抗性攻击模型 cs.LG

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2106.14815v2)

**Authors**: Gilad Gressel, Niranjan Hegde, Archana Sreekumar, Rishikumar Radhakrishnan, Kalyani Harikumar, Anjali S., Michael Darling

**Abstracts**: Research in adversarial learning has primarily focused on homogeneous unstructured datasets, which often map into the problem space naturally. Inverting a feature space attack on heterogeneous datasets into the problem space is much more challenging, particularly the task of finding the perturbation to perform. This work presents a formal search strategy: the `Feature Importance Guided Attack' (FIGA), which finds perturbations in the feature space of heterogeneous tabular datasets to produce evasion attacks. We first demonstrate FIGA in the feature space and then in the problem space. FIGA assumes no prior knowledge of the defending model's learning algorithm and does not require any gradient information. FIGA assumes knowledge of the feature representation and the mean feature values of defending model's dataset. FIGA leverages feature importance rankings by perturbing the most important features of the input in the direction of the target class. While FIGA is conceptually similar to other work which uses feature selection processes (e.g., mimicry attacks), we formalize an attack algorithm with three tunable parameters and investigate the strength of FIGA on tabular datasets. We demonstrate the effectiveness of FIGA by evading phishing detection models trained on four different tabular phishing datasets and one financial dataset with an average success rate of 94%. We extend FIGA to the phishing problem space by limiting the possible perturbations to be valid and feasible in the phishing domain. We generate valid adversarial phishing sites that are visually identical to their unperturbed counterpart and use them to attack six tabular ML models achieving a 13.05% average success rate.

摘要: 对抗性学习的研究主要集中在同质的非结构化数据集上，这些数据集往往自然地映射到问题空间。将异类数据集上的特征空间攻击转化到问题空间中的挑战要大得多，特别是找到要执行的扰动的任务。该工作提出了一种形式化的搜索策略：特征重要性制导攻击(FIGA)，它在异类表格数据集的特征空间中发现扰动，从而产生规避攻击。我们首先在特征空间中证明FIGA，然后在问题空间中证明FIGA。Figa不假定防御模型的学习算法的先验知识，也不需要任何梯度信息。FigA假设已知防御模型数据集的特征表示和平均特征值。FigA通过在目标类的方向上干扰输入的最重要特征来利用特征重要性排名。虽然FIGA在概念上类似于其他使用特征选择过程的工作(例如，模仿攻击)，但我们使用三个可调参数来形式化攻击算法，并研究了FIGA在表格数据集上的优势。我们通过在四个不同的表格钓鱼数据集和一个金融数据集上训练的钓鱼检测模型，证明了FIGA的有效性，平均成功率为94%。通过限制可能的扰动在钓鱼领域是有效和可行的，我们将FIGA扩展到钓鱼问题空间。我们生成了有效的敌意钓鱼网站，这些网站在视觉上与未受干扰的网站相同，并使用它们攻击六个表格ML模型，平均成功率为13.05%。



## **39. AdaptOver: Adaptive Overshadowing Attacks in Cellular Networks**

AdaptOver：蜂窝网络中的自适应遮蔽攻击 cs.CR

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2106.05039v3)

**Authors**: Simon Erni, Martin Kotuliak, Patrick Leu, Marc Roeschlin, Srdjan Capkun

**Abstracts**: In cellular networks, attacks on the communication link between a mobile device and the core network significantly impact privacy and availability. Up until now, fake base stations have been required to execute such attacks. Since they require a continuously high output power to attract victims, they are limited in range and can be easily detected both by operators and dedicated apps on users' smartphones.   This paper introduces AdaptOver - a MITM attack system designed for cellular networks, specifically for LTE and 5G-NSA. AdaptOver allows an adversary to decode, overshadow (replace) and inject arbitrary messages over the air in either direction between the network and the mobile device. Using overshadowing, AdaptOver can cause a persistent ($\geq$ 12h) DoS or a privacy leak by triggering a UE to transmit its persistent identifier (IMSI) in plain text. These attacks can be launched against all users within a cell or specifically target a victim based on its phone number.   We implement AdaptOver using a software-defined radio and a low-cost amplification setup. We demonstrate the effects and practicality of the attacks on a live operational LTE and 5G-NSA network with a wide range of smartphones. Our experiments show that AdaptOver can launch an attack on a victim more than 3.8km away from the attacker. Given its practicability and efficiency, AdaptOver shows that existing countermeasures that are focused on fake base stations are no longer sufficient, marking a paradigm shift for designing security mechanisms in cellular networks.

摘要: 在蜂窝网络中，对移动设备和核心网络之间的通信链路的攻击会严重影响隐私和可用性。到目前为止，伪基站已经被要求执行这样的攻击。由于它们需要持续高的输出功率来吸引受害者，因此它们的射程有限，运营商和用户智能手机上的专用应用程序都很容易检测到它们。介绍了一种专为LTE和5G-NSA蜂窝网络设计的MITM攻击系统--AdaptOver。AdaptOver允许对手在网络和移动设备之间的任一方向上通过空中解码、掩盖(替换)和注入任意消息。使用遮蔽，AdaptOver可以触发UE以纯文本形式传输其永久标识符(IMSI)，从而导致持续($\geq$12h)DoS或隐私泄露。这些攻击可以针对一个小区内的所有用户，也可以根据受害者的电话号码专门针对受害者。我们使用软件定义的无线电和低成本的放大设置来实现AdaptOver。我们使用各种智能手机演示了这些攻击对实时运行的LTE和5G-NSA网络的影响和实用性。我们的实验表明，AdaptOver可以对距离攻击者3.8公里以上的受害者发动攻击。考虑到其实用性和效率，AdaptOver表明，专注于伪基站的现有对策不再足够，标志着蜂窝网络安全机制设计的范式转变。



## **40. Combing for Credentials: Active Pattern Extraction from Smart Reply**

梳理凭据：从智能回复中提取活动模式 cs.CR

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2207.10802v2)

**Authors**: Bargav Jayaraman, Esha Ghosh, Melissa Chase, Sambuddha Roy, Huseyin Inan, Wei Dai, David Evans

**Abstracts**: With the wide availability of large pre-trained language models such as GPT-2 and BERT, the recent trend has been to fine-tune a pre-trained model to achieve state-of-the-art performance on a downstream task. One natural example is the "Smart Reply" application where a pre-trained model is tuned to provide suggested responses for a given query message. Since these models are often tuned using sensitive data such as emails or chat transcripts, it is important to understand and mitigate the risk that the model leaks its tuning data. We investigate potential information leakage vulnerabilities in a typical Smart Reply pipeline and introduce a new type of active extraction attack that exploits canonical patterns in text containing sensitive data. We show experimentally that it is possible for an adversary to extract sensitive user information present in the training data. We explore potential mitigation strategies and demonstrate empirically how differential privacy appears to be an effective defense mechanism to such pattern extraction attacks.

摘要: 随着GPT-2和BERT等大型预训练语言模型的广泛使用，最近的趋势是微调预训练模型，以在下游任务中实现最先进的性能。一个自然的例子是“智能回复”应用程序，其中预先训练的模型被调优以提供对给定查询消息的建议响应。由于这些模型通常使用电子邮件或聊天记录等敏感数据进行调整，因此了解并降低模型泄露其调整数据的风险非常重要。我们调查了一个典型的智能回复管道中潜在的信息泄漏漏洞，并引入了一种新型的主动提取攻击，该攻击利用了包含敏感数据的文本中的规范模式。我们的实验表明，对手有可能提取训练数据中存在的敏感用户信息。我们探索了潜在的缓解策略，并经验地证明了差异隐私似乎是应对此类模式提取攻击的一种有效防御机制。



## **41. Inferring Sensitive Attributes from Model Explanations**

从模型解释中推断敏感属性 cs.CR

ACM CIKM 2022

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2208.09967v2)

**Authors**: Vasisht Duddu, Antoine Boutet

**Abstracts**: Model explanations provide transparency into a trained machine learning model's blackbox behavior to a model builder. They indicate the influence of different input attributes to its corresponding model prediction. The dependency of explanations on input raises privacy concerns for sensitive user data. However, current literature has limited discussion on privacy risks of model explanations.   We focus on the specific privacy risk of attribute inference attack wherein an adversary infers sensitive attributes of an input (e.g., race and sex) given its model explanations. We design the first attribute inference attack against model explanations in two threat models where model builder either (a) includes the sensitive attributes in training data and input or (b) censors the sensitive attributes by not including them in the training data and input.   We evaluate our proposed attack on four benchmark datasets and four state-of-the-art algorithms. We show that an adversary can successfully infer the value of sensitive attributes from explanations in both the threat models accurately. Moreover, the attack is successful even by exploiting only the explanations corresponding to sensitive attributes. These suggest that our attack is effective against explanations and poses a practical threat to data privacy.   On combining the model predictions (an attack surface exploited by prior attacks) with explanations, we note that the attack success does not improve. Additionally, the attack success on exploiting model explanations is better compared to exploiting only model predictions. These suggest that model explanations are a strong attack surface to exploit for an adversary.

摘要: 模型解释为模型构建者提供了对经过训练的机器学习模型的黑箱行为的透明性。它们表明了不同的输入属性对其相应模型预测的影响。解释对输入的依赖引发了对敏感用户数据的隐私问题。然而，目前的文献对模型解释的隐私风险的讨论有限。我们专注于属性推理攻击的特定隐私风险，其中对手根据输入的模型解释推断输入的敏感属性(例如，种族和性别)。我们针对两个威胁模型中的模型解释设计了第一个属性推理攻击，在这两个模型中，建模者或者(A)在训练数据和输入中包括敏感属性，或者(B)通过在训练数据和输入中不包括敏感属性来审查敏感属性。我们在四个基准数据集和四个最先进的算法上评估了我们提出的攻击。我们表明，攻击者可以从两种威胁模型中的解释中准确地推断出敏感属性的值。此外，即使只利用与敏感属性对应的解释，攻击也是成功的。这些都表明，我们的攻击针对解释是有效的，并对数据隐私构成了实际威胁。在将模型预测(先前攻击所利用的攻击面)与解释相结合时，我们注意到攻击成功率并没有提高。此外，与仅利用模型预测相比，利用模型解释的攻击成功更好。这些都表明，模型解释是对手可以利用的强大攻击面。



## **42. Securing the Spike: On the Transferabilty and Security of Spiking Neural Networks to Adversarial Examples**

保护尖峰：尖峰神经网络对对抗性例子的可传递性和安全性 cs.NE

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2209.03358v1)

**Authors**: Nuo Xu, Kaleel Mahmood, Haowen Fang, Ethan Rathbun, Caiwen Ding, Wujie Wen

**Abstracts**: Spiking neural networks (SNNs) have attracted much attention for their high energy efficiency and for recent advances in their classification performance. However, unlike traditional deep learning approaches, the analysis and study of the robustness of SNNs to adversarial examples remains relatively underdeveloped. In this work we advance the field of adversarial machine learning through experimentation and analyses of three important SNN security attributes. First, we show that successful white-box adversarial attacks on SNNs are highly dependent on the underlying surrogate gradient technique. Second, we analyze the transferability of adversarial examples generated by SNNs and other state-of-the-art architectures like Vision Transformers and Big Transfer CNNs. We demonstrate that SNNs are not often deceived by adversarial examples generated by Vision Transformers and certain types of CNNs. Lastly, we develop a novel white-box attack that generates adversarial examples capable of fooling both SNN models and non-SNN models simultaneously. Our experiments and analyses are broad and rigorous covering two datasets (CIFAR-10 and CIFAR-100), five different white-box attacks and twelve different classifier models.

摘要: 尖峰神经网络(SNN)因其高能量效率和分类性能的最新进展而备受关注。然而，与传统的深度学习方法不同的是，对SNN对敌意例子的稳健性的分析和研究还相对较不发达。在这项工作中，我们通过实验和分析三个重要的SNN安全属性来推进对抗性机器学习领域。首先，我们证明了针对SNN的成功的白盒对抗攻击高度依赖于潜在的代理梯度技术。其次，我们分析了SNN和其他最先进的架构，如Vision Transformers和Big Transfer CNN生成的对抗性例子的可转移性。我们证明了SNN不会经常被Vision Transformers和某些类型的CNN生成的敌意例子所欺骗。最后，我们开发了一种新的白盒攻击，它可以生成能够同时愚弄SNN模型和非SNN模型的对抗性示例。我们的实验和分析涵盖了两个数据集(CIFAR-10和CIFAR-100)、五种不同的白盒攻击和12种不同的分类器模型。



## **43. Minotaur: Multi-Resource Blockchain Consensus**

牛头人：多资源区块链共识 cs.CR

To appear in ACM CCS 2022

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2201.11780v2)

**Authors**: Matthias Fitzi, Xuechao Wang, Sreeram Kannan, Aggelos Kiayias, Nikos Leonardos, Pramod Viswanath, Gerui Wang

**Abstracts**: Resource-based consensus is the backbone of permissionless distributed ledger systems. The security of such protocols relies fundamentally on the level of resources actively engaged in the system. The variety of different resources (and related proof protocols, some times referred to as PoX in the literature) raises the fundamental question whether it is possible to utilize many of them in tandem and build multi-resource consensus protocols. The challenge in combining different resources is to achieve fungibility between them, in the sense that security would hold as long as the cumulative adversarial power across all resources is bounded.   In this work, we put forth Minotaur, a multi-resource blockchain consensus protocol that combines proof-of-work (PoW) and proof-of-stake (PoS), and we prove it optimally fungible. At the core of our design, Minotaur operates in epochs while continuously sampling the active computational power to provide a fair exchange between the two resources, work and stake. Further, we demonstrate the ability of Minotaur to handle a higher degree of work fluctuation as compared to the Bitcoin blockchain; we also generalize Minotaur to any number of resources.   We demonstrate the simplicity of Minotaur via implementing a full stack client in Rust (available open source). We use the client to test the robustness of Minotaur to variable mining power and combined work/stake attacks and demonstrate concrete empirical evidence towards the suitability of Minotaur to serve as the consensus layer of a real-world blockchain.

摘要: 基于资源的共识是未经许可的分布式分类账系统的支柱。这类协议的安全性从根本上取决于系统中活跃的资源水平。不同资源的多样性(以及相关的证明协议，在文献中有时被称为POX)提出了一个基本的问题，即是否有可能同时利用其中的许多资源并建立多资源共识协议。组合不同资源的挑战是实现它们之间的互换性，从这个意义上说，只要所有资源的累积对抗能力是有限度的，安全就会保持。在这项工作中，我们提出了Minotaur，一个结合了工作证明(PoW)和风险证明(POS)的多资源区块链共识协议，并证明了它的最优可替换性。在我们设计的核心，Minotaur在不断采样活跃的计算能力的同时，在工作和赌注这两种资源之间提供公平的交换。此外，我们还展示了与比特币区块链相比，Minotaur能够处理更高程度的工作波动；我们还将Minotaur推广到任何数量的资源。我们通过在Rust(开放源码可用)中实现一个完整的堆栈客户端来演示Minotaur的简单性。我们使用客户端来测试Minotaur对可变挖掘功率和组合工作/桩攻击的健壮性，并展示了具体的经验证据，证明Minotaur适合作为现实世界区块链的共识层。



## **44. Distributed Adversarial Training to Robustify Deep Neural Networks at Scale**

深度神经网络规模化的分布式对抗性训练 cs.LG

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2206.06257v2)

**Authors**: Gaoyuan Zhang, Songtao Lu, Yihua Zhang, Xiangyi Chen, Pin-Yu Chen, Quanfu Fan, Lee Martie, Lior Horesh, Mingyi Hong, Sijia Liu

**Abstracts**: Current deep neural networks (DNNs) are vulnerable to adversarial attacks, where adversarial perturbations to the inputs can change or manipulate classification. To defend against such attacks, an effective and popular approach, known as adversarial training (AT), has been shown to mitigate the negative impact of adversarial attacks by virtue of a min-max robust training method. While effective, it remains unclear whether it can successfully be adapted to the distributed learning context. The power of distributed optimization over multiple machines enables us to scale up robust training over large models and datasets. Spurred by that, we propose distributed adversarial training (DAT), a large-batch adversarial training framework implemented over multiple machines. We show that DAT is general, which supports training over labeled and unlabeled data, multiple types of attack generation methods, and gradient compression operations favored for distributed optimization. Theoretically, we provide, under standard conditions in the optimization theory, the convergence rate of DAT to the first-order stationary points in general non-convex settings. Empirically, we demonstrate that DAT either matches or outperforms state-of-the-art robust accuracies and achieves a graceful training speedup (e.g., on ResNet-50 under ImageNet). Codes are available at https://github.com/dat-2022/dat.

摘要: 当前的深度神经网络(DNN)很容易受到敌意攻击，对输入的敌意扰动可以改变或操纵分类。为了防御这种攻击，一种被称为对抗性训练(AT)的有效和流行的方法已经被证明通过最小-最大稳健训练方法来减轻对抗性攻击的负面影响。虽然有效，但它是否能成功地适应分布式学习环境仍不清楚。在多台机器上进行分布式优化的能力使我们能够在大型模型和数据集上扩大健壮的训练。受此启发，我们提出了分布式对抗训练(DAT)，这是一种在多台机器上实现的大批量对抗训练框架。我们证明DAT是通用的，它支持对有标签和无标签数据的训练，支持多种类型的攻击生成方法，以及有利于分布式优化的梯度压缩操作。理论上，在最优化理论的标准条件下，我们给出了一般非凸集上DAT收敛到一阶驻点的收敛速度。在实验上，我们证明了DAT匹配或超过了最先进的稳健精度，并实现了优雅的训练加速比(例如，在ImageNet下的ResNet-50上)。有关代码，请访问https://github.com/dat-2022/dat.



## **45. Privacy Against Inference Attacks in Vertical Federated Learning**

垂直联合学习中抵抗推理攻击的隐私保护 cs.LG

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2207.11788v3)

**Authors**: Borzoo Rassouli, Morteza Varasteh, Deniz Gunduz

**Abstracts**: Vertical federated learning is considered, where an active party, having access to true class labels, wishes to build a classification model by utilizing more features from a passive party, which has no access to the labels, to improve the model accuracy. In the prediction phase, with logistic regression as the classification model, several inference attack techniques are proposed that the adversary, i.e., the active party, can employ to reconstruct the passive party's features, regarded as sensitive information. These attacks, which are mainly based on a classical notion of the center of a set, i.e., the Chebyshev center, are shown to be superior to those proposed in the literature. Moreover, several theoretical performance guarantees are provided for the aforementioned attacks. Subsequently, we consider the minimum amount of information that the adversary needs to fully reconstruct the passive party's features. In particular, it is shown that when the passive party holds one feature, and the adversary is only aware of the signs of the parameters involved, it can perfectly reconstruct that feature when the number of predictions is large enough. Next, as a defense mechanism, a privacy-preserving scheme is proposed that worsen the adversary's reconstruction attacks, while preserving the full benefits that VFL brings to the active party. Finally, experimental results demonstrate the effectiveness of the proposed attacks and the privacy-preserving scheme.

摘要: 考虑垂直联合学习，其中可以访问真实类别标签的主动方希望通过利用来自被动方的更多特征来构建分类模型，而被动方不能访问标签，以提高模型的精度。在预测阶段，以Logistic回归为分类模型，提出了几种推理攻击技术，对手即主动方可以用来重构被动方的特征，并将其视为敏感信息。这些攻击主要基于经典的集合中心概念，即切比雪夫中心，被证明优于文献中提出的攻击。此外，还为上述攻击提供了几个理论上的性能保证。随后，我们考虑了对手完全重建被动方特征所需的最小信息量。特别地，当被动方持有一个特征，并且对手只知道所涉及的参数的符号时，当预测次数足够大时，它可以完美地重构该特征。接下来，作为一种防御机制，提出了一种隐私保护方案，该方案在保留VFL给主动方带来的全部好处的同时，恶化了对手的重构攻击。最后，实验结果验证了所提出的攻击和隐私保护方案的有效性。



## **46. Fact-Saboteurs: A Taxonomy of Evidence Manipulation Attacks against Fact-Verification Systems**

事实破坏者：针对事实核查系统的证据操纵攻击的分类 cs.CR

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2209.03755v1)

**Authors**: Sahar Abdelnabi, Mario Fritz

**Abstracts**: Mis- and disinformation are now a substantial global threat to our security and safety. To cope with the scale of online misinformation, one viable solution is to automate the fact-checking of claims by retrieving and verifying against relevant evidence. While major recent advances have been achieved in pushing forward the automatic fact-verification, a comprehensive evaluation of the possible attack vectors against such systems is still lacking. Particularly, the automated fact-verification process might be vulnerable to the exact disinformation campaigns it is trying to combat. In this work, we assume an adversary that automatically tampers with the online evidence in order to disrupt the fact-checking model via camouflaging the relevant evidence, or planting a misleading one. We first propose an exploratory taxonomy that spans these two targets and the different threat model dimensions. Guided by this, we design and propose several potential attack methods. We show that it is possible to subtly modify claim-salient snippets in the evidence, in addition to generating diverse and claim-aligned evidence. As a result, we highly degrade the fact-checking performance under many different permutations of the taxonomy's dimensions. The attacks are also robust against post-hoc modifications of the claim. Our analysis further hints at potential limitations in models' inference when faced with contradicting evidence. We emphasize that these attacks can have harmful implications on the inspectable and human-in-the-loop usage scenarios of such models, and we conclude by discussing challenges and directions for future defenses.

摘要: 错误和虚假信息现在是对我们的安全和安全的重大全球威胁。为了应对网上虚假信息的规模，一个可行的解决方案是通过检索和核实相关证据来自动化索赔的事实核查。虽然最近在推动自动事实核查方面取得了重大进展，但仍然缺乏对针对这类系统的可能攻击媒介的全面评估。特别是，自动化的事实核查过程可能容易受到它试图打击的虚假信息运动的影响。在这项工作中，我们假设一个对手自动篡改在线证据，以便通过伪装相关证据或植入误导性证据来扰乱事实核查模型。我们首先提出了一种探索性分类，该分类跨越这两个目标和不同的威胁模型维度。在此指导下，我们设计并提出了几种潜在的攻击方法。我们表明，除了生成多样化的和与索赔一致的证据外，还可以微妙地修改证据中突出索赔的片段。因此，在分类维度的许多不同排列下，我们会极大地降低事实检查性能。这些攻击也对索赔的事后修改具有很强的抵御能力。我们的分析进一步暗示，在面对相互矛盾的证据时，模型的推理可能存在局限性。我们强调，这些攻击可能会对此类模型的可检查和人在环中使用场景产生有害影响，我们最后讨论了未来防御的挑战和方向。



## **47. State of Security Awareness in the AM Industry: 2020 Survey**

AM行业的安全意识状况：2020年调查 cs.CR

The material was presented at ASTM ICAM 2021 and a publication was  accepted for publication as a Selected Technical Papers (STP)

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2209.03073v1)

**Authors**: Mark Yampolskiy, Paul Bates, Mohsen Seifi, Nima Shamsaei

**Abstracts**: Security of Additive Manufacturing (AM) gets increased attention due to the growing proliferation and adoption of AM in a variety of applications and business models. However, there is a significant disconnect between AM community focused on manufacturing and AM Security community focused on securing this highly computerized manufacturing technology. To bridge this gap, we surveyed the America Makes AM community, asking in total eleven AM security-related questions aiming to discover the existing concerns, posture, and expectations. The first set of questions aimed to discover how many of these organizations use AM, outsource AM, or provide AM as a service. Then we asked about biggest security concerns as well as about assessment of who the potential adversaries might be and their motivation for attack. We then proceeded with questions on any experienced security incidents, if any security risk assessment was conducted, and if the participants' organizations were partnering with external experts to secure AM. Lastly, we asked whether security measures are implemented at all and, if yes, whether they fall under the general cyber-security category. Out of 69 participants affiliated with commercial industry, agencies, and academia, 53 have completed the entire survey. This paper presents the results of this survey, as well as provides our assessment of the AM Security posture. The answers are a mixture of what we could label as expected, "shocking but not surprising," and completely unexpected. Assuming that the provided answers are somewhat representative to the current state of the AM industry, we conclude that the industry is not ready to prevent or detect AM-specific attacks that have been demonstrated in the research literature.

摘要: 由于添加制造(AM)在各种应用和商业模式中的普及和采用，AM的安全性受到越来越多的关注。然而，在专注于制造的AM社区和专注于保护这种高度计算机化的制造技术的AM Security社区之间存在着严重的脱节。为了弥合这一差距，我们对美国制造AM社区进行了调查，总共询问了11个与AM安全相关的问题，旨在发现现有的担忧、状况和期望。第一组问题旨在发现这些组织中有多少使用AM、外包AM或将AM作为服务提供。然后，我们询问了最大的安全担忧，以及对潜在对手可能是谁及其攻击动机的评估。然后，我们继续就任何有经验的安全事件、是否进行了任何安全风险评估以及参与者的组织是否与外部专家合作确保AM的安全进行了提问最后，我们询问是否实施了安全措施，如果是，这些措施是否属于一般网络安全类别。在69名与商业、机构和学术界有关联的参与者中，有53人完成了整个调查。本文介绍了本次调查的结果，并提供了我们对AM安全态势的评估。答案既有我们所期待的，也有完全出乎意料的，既令人震惊，又不令人惊讶。假设所提供的答案在某种程度上代表了AM行业的当前状态，我们得出的结论是，该行业还没有准备好预防或检测研究文献中已经证明的AM特定攻击。



## **48. On the Transferability of Adversarial Examples between Encrypted Models**

关于对抗性例子在加密模型之间的可转移性 cs.CV

to be appear in ISPACS 2022

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2209.02997v1)

**Authors**: Miki Tanaka, Isao Echizen, Hitoshi Kiya

**Abstracts**: Deep neural networks (DNNs) are well known to be vulnerable to adversarial examples (AEs). In addition, AEs have adversarial transferability, namely, AEs generated for a source model fool other (target) models. In this paper, we investigate the transferability of models encrypted for adversarially robust defense for the first time. To objectively verify the property of transferability, the robustness of models is evaluated by using a benchmark attack method, called AutoAttack. In an image-classification experiment, the use of encrypted models is confirmed not only to be robust against AEs but to also reduce the influence of AEs in terms of the transferability of models.

摘要: 众所周知，深度神经网络(DNN)很容易受到敌意例子(AEs)的攻击。此外，AEs具有对抗性可转移性，即，为源模型生成的AEs欺骗其他(目标)模型。在这篇文章中，我们首次研究了用于对抗健壮防御的加密模型的可转移性。为了客观地验证模型的可转移性，使用一种称为AutoAttack的基准攻击方法对模型的稳健性进行了评估。在图像分类实验中，加密模型的使用被证实不仅对AEs具有健壮性，而且在模型的可转移性方面也减少了AEs的影响。



## **49. Adversarial Mask: Real-World Universal Adversarial Attack on Face Recognition Model**

对抗性面具：现实世界中人脸识别模型的通用对抗性攻击 cs.CV

16 pages, 9 figures

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2111.10759v3)

**Authors**: Alon Zolfi, Shai Avidan, Yuval Elovici, Asaf Shabtai

**Abstracts**: Deep learning-based facial recognition (FR) models have demonstrated state-of-the-art performance in the past few years, even when wearing protective medical face masks became commonplace during the COVID-19 pandemic. Given the outstanding performance of these models, the machine learning research community has shown increasing interest in challenging their robustness. Initially, researchers presented adversarial attacks in the digital domain, and later the attacks were transferred to the physical domain. However, in many cases, attacks in the physical domain are conspicuous, and thus may raise suspicion in real-world environments (e.g., airports). In this paper, we propose Adversarial Mask, a physical universal adversarial perturbation (UAP) against state-of-the-art FR models that is applied on face masks in the form of a carefully crafted pattern. In our experiments, we examined the transferability of our adversarial mask to a wide range of FR model architectures and datasets. In addition, we validated our adversarial mask's effectiveness in real-world experiments (CCTV use case) by printing the adversarial pattern on a fabric face mask. In these experiments, the FR system was only able to identify 3.34% of the participants wearing the mask (compared to a minimum of 83.34% with other evaluated masks). A demo of our experiments can be found at: https://youtu.be/_TXkDO5z11w.

摘要: 基于深度学习的面部识别(FR)模型在过去几年展示了最先进的性能，即使在新冠肺炎大流行期间戴防护性医用口罩变得司空见惯。鉴于这些模型的出色性能，机器学习研究界对挑战它们的稳健性表现出越来越大的兴趣。最初，研究人员在数字领域提出了对抗性攻击，后来攻击被转移到物理领域。然而，在许多情况下，物理域中的攻击很明显，因此可能会在现实环境(例如机场)中引起怀疑。在本文中，我们提出了对抗面具，一种针对最先进的FR模型的物理通用对抗扰动(UAP)，它以精心制作的模式的形式应用于人脸面具上。在我们的实验中，我们检查了我们的对手面具在广泛的FR模型体系结构和数据集上的可转移性。此外，我们在真实世界的实验中(CCTV用例)验证了我们的对抗面具的有效性，通过将对抗图案打印在织物面膜上。在这些实验中，FR系统只能识别3.34%的戴口罩的参与者(相比之下，其他评估的口罩的最低识别率为83.34%)。我们的实验演示可在以下网址找到：https://youtu.be/_TXkDO5z11w.



## **50. Facial De-morphing: Extracting Component Faces from a Single Morph**

面部去变形：从单个变形中提取组件人脸 cs.CV

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2209.02933v1)

**Authors**: Sudipta Banerjee, Prateek Jaiswal, Arun Ross

**Abstracts**: A face morph is created by strategically combining two or more face images corresponding to multiple identities. The intention is for the morphed image to match with multiple identities. Current morph attack detection strategies can detect morphs but cannot recover the images or identities used in creating them. The task of deducing the individual face images from a morphed face image is known as \textit{de-morphing}. Existing work in de-morphing assume the availability of a reference image pertaining to one identity in order to recover the image of the accomplice - i.e., the other identity. In this work, we propose a novel de-morphing method that can recover images of both identities simultaneously from a single morphed face image without needing a reference image or prior information about the morphing process. We propose a generative adversarial network that achieves single image-based de-morphing with a surprisingly high degree of visual realism and biometric similarity with the original face images. We demonstrate the performance of our method on landmark-based morphs and generative model-based morphs with promising results.

摘要: 面部变形是通过策略性地组合对应于多个身份的两个或多个面部图像来创建的。其目的是使变形后的图像与多个身份匹配。当前的变形攻击检测策略可以检测变形，但无法恢复创建变形时使用的图像或身份。从变形后的人脸图像中推断出单个人脸图像的任务称为纹理{去变形}。现有的去变形工作假定存在与一个身份有关的参考图像，以便恢复共犯--即另一个身份--的图像。在这项工作中，我们提出了一种新的去变形方法，它可以从一幅变形后的人脸图像中同时恢复出两种身份的图像，而不需要参考图像或关于变形过程的先验信息。我们提出了一种产生式对抗性网络，它实现了基于单一图像的去变形，具有与原始人脸图像惊人的高度视觉真实感和生物特征相似性。我们展示了我们的方法在基于里程碑的变形和基于生成性模型的变形上的性能，并取得了令人满意的结果。



