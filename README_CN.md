# Latest Adversarial Attack Papers
**update at 2021-12-03 23:56:46**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. FedRAD: Federated Robust Adaptive Distillation**

FedRAD：联合鲁棒自适应精馏 cs.LG

Accepted for 1st NeurIPS Workshop on New Frontiers in Federated  Learning (NFFL 2021), Virtual Meeting

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2112.01405v1)

**Authors**: Stefán Páll Sturluson, Samuel Trew, Luis Muñoz-González, Matei Grama, Jonathan Passerat-Palmbach, Daniel Rueckert, Amir Alansary

**Abstracts**: The robustness of federated learning (FL) is vital for the distributed training of an accurate global model that is shared among large number of clients. The collaborative learning framework by typically aggregating model updates is vulnerable to model poisoning attacks from adversarial clients. Since the shared information between the global server and participants are only limited to model parameters, it is challenging to detect bad model updates. Moreover, real-world datasets are usually heterogeneous and not independent and identically distributed (Non-IID) among participants, which makes the design of such robust FL pipeline more difficult. In this work, we propose a novel robust aggregation method, Federated Robust Adaptive Distillation (FedRAD), to detect adversaries and robustly aggregate local models based on properties of the median statistic, and then performing an adapted version of ensemble Knowledge Distillation. We run extensive experiments to evaluate the proposed method against recently published works. The results show that FedRAD outperforms all other aggregators in the presence of adversaries, as well as in heterogeneous data distributions.

摘要: 联邦学习(FL)的健壮性对于分布式训练大量客户端共享的精确全局模型至关重要。典型地聚合模型更新的协作学习框架容易受到来自敌对客户端的模型中毒攻击。由于全局服务器和参与者之间的共享信息仅限于模型参数，因此检测错误的模型更新是具有挑战性的。此外，现实世界的数据集通常是异构的，参与者之间并不是独立且相同分布的(非IID)，这使得设计这样健壮的FL流水线变得更加困难。在这项工作中，我们提出了一种新的健壮聚合方法，联邦健壮自适应蒸馏(FedRAD)，根据中值统计特性检测对手并健壮聚合局部模型，然后执行改进版本的集成知识蒸馏。我们进行了大量的实验，以评估所提出的方法与最近发表的作品。结果表明，FedRAD在存在对手的情况下，以及在异构数据分布的情况下，性能优于所有其他聚合器。



## **2. A Unified Framework for Adversarial Attack and Defense in Constrained Feature Space**

受限特征空间中的对抗性攻防统一框架 cs.AI

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2112.01156v1)

**Authors**: Thibault Simonetto, Salijona Dyrmishi, Salah Ghamizi, Maxime Cordy, Yves Le Traon

**Abstracts**: The generation of feasible adversarial examples is necessary for properly assessing models that work on constrained feature space. However, it remains a challenging task to enforce constraints into attacks that were designed for computer vision. We propose a unified framework to generate feasible adversarial examples that satisfy given domain constraints. Our framework supports the use cases reported in the literature and can handle both linear and non-linear constraints. We instantiate our framework into two algorithms: a gradient-based attack that introduces constraints in the loss function to maximize, and a multi-objective search algorithm that aims for misclassification, perturbation minimization, and constraint satisfaction. We show that our approach is effective on two datasets from different domains, with a success rate of up to 100%, where state-of-the-art attacks fail to generate a single feasible example. In addition to adversarial retraining, we propose to introduce engineered non-convex constraints to improve model adversarial robustness. We demonstrate that this new defense is as effective as adversarial retraining. Our framework forms the starting point for research on constrained adversarial attacks and provides relevant baselines and datasets that future research can exploit.

摘要: 生成可行的对抗性示例对于正确评估工作在受限特征空间上的模型是必要的。然而，对专为计算机视觉设计的攻击实施约束仍然是一项具有挑战性的任务。我们提出了一个统一的框架来生成满足给定领域约束的可行对抗性示例。我们的框架支持文献中报告的用例，并且可以处理线性和非线性约束。我们将我们的框架实例化为两种算法：一种是在损失函数中引入约束以最大化的基于梯度的攻击算法，另一种是以误分类、扰动最小化和约束满足为目标的多目标搜索算法。我们在两个来自不同领域的数据集上证明了我们的方法是有效的，成功率高达100%，其中最先进的攻击没有产生一个可行的例子。除了对抗性再训练之外，我们还建议引入工程非凸约束来提高模型对抗性的稳健性。我们证明了这种新的防御和对抗性的再训练一样有效。我们的框架构成了受限对抗攻击研究的起点，并为未来的研究提供了相关的基线和数据集。



## **3. Adversarial Robustness of Deep Reinforcement Learning based Dynamic Recommender Systems**

基于深度强化学习的动态推荐系统的对抗鲁棒性 cs.LG

arXiv admin note: text overlap with arXiv:2006.07934

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2112.00973v1)

**Authors**: Siyu Wang, Yuanjiang Cao, Xiaocong Chen, Lina Yao, Xianzhi Wang, Quan Z. Sheng

**Abstracts**: Adversarial attacks, e.g., adversarial perturbations of the input and adversarial samples, pose significant challenges to machine learning and deep learning techniques, including interactive recommendation systems. The latent embedding space of those techniques makes adversarial attacks difficult to detect at an early stage. Recent advance in causality shows that counterfactual can also be considered one of ways to generate the adversarial samples drawn from different distribution as the training samples. We propose to explore adversarial examples and attack agnostic detection on reinforcement learning-based interactive recommendation systems. We first craft different types of adversarial examples by adding perturbations to the input and intervening on the casual factors. Then, we augment recommendation systems by detecting potential attacks with a deep learning-based classifier based on the crafted data. Finally, we study the attack strength and frequency of adversarial examples and evaluate our model on standard datasets with multiple crafting methods. Our extensive experiments show that most adversarial attacks are effective, and both attack strength and attack frequency impact the attack performance. The strategically-timed attack achieves comparative attack performance with only 1/3 to 1/2 attack frequency. Besides, our black-box detector trained with one crafting method has the generalization ability over several other crafting methods.

摘要: 对抗性攻击，例如输入和对抗性样本的对抗性扰动，给机器学习和深度学习技术(包括交互式推荐系统)带来了重大挑战。这些技术的潜在嵌入空间使得敌意攻击很难在早期阶段被发现。最近因果关系的进展表明，反事实也可以被认为是生成来自不同分布的对抗性样本作为训练样本的方法之一。我们提出在基于强化学习的交互式推荐系统上探索敌意示例和攻击不可知检测。我们首先制作不同类型的对抗性例子，通过在输入中添加扰动和对偶然因素进行干预来创建不同类型的对抗性例子。然后，我们利用基于深度学习的分类器基于精心制作的数据来检测潜在的攻击，从而增强推荐系统。最后，我们研究了敌意实例的攻击强度和攻击频率，并在标准数据集上采用多种制作方法对我们的模型进行了评估。我们的大量实验表明，大多数对抗性攻击都是有效的，攻击强度和攻击频率都会影响攻击性能。战略计时攻击仅用1/3到1/2的攻击频率就达到了比较的攻击性能。此外，用一种工艺方法训练的黑盒检测器比其他几种工艺方法具有更好的泛化能力。



## **4. Learning Task-aware Robust Deep Learning Systems**

学习任务感知的鲁棒深度学习系统 cs.LG

9 Pages

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2010.05125v2)

**Authors**: Keji Han, Yun Li, Xianzhong Long, Yao Ge

**Abstracts**: Many works demonstrate that deep learning system is vulnerable to adversarial attack. A deep learning system consists of two parts: the deep learning task and the deep model. Nowadays, most existing works investigate the impact of the deep model on robustness of deep learning systems, ignoring the impact of the learning task. In this paper, we adopt the binary and interval label encoding strategy to redefine the classification task and design corresponding loss to improve robustness of the deep learning system. Our method can be viewed as improving the robustness of deep learning systems from both the learning task and deep model. Experimental results demonstrate that our learning task-aware method is much more robust than traditional classification while retaining the accuracy.

摘要: 大量研究表明，深度学习系统容易受到敌意攻击。深度学习系统由两部分组成：深度学习任务和深度模型。目前，已有的工作大多研究深度模型对深度学习系统鲁棒性的影响，忽略了学习任务的影响。本文采用二进制和区间标签编码策略重新定义分类任务，并设计相应的损失来提高深度学习系统的鲁棒性。我们的方法可以看作是从学习任务和深度模型两个方面提高深度学习系统的鲁棒性。实验结果表明，我们的学习任务感知方法在保持分类准确率的同时，比传统的分类方法具有更强的鲁棒性。



## **5. They See Me Rollin': Inherent Vulnerability of the Rolling Shutter in CMOS Image Sensors**

他们看到我在滚动：CMOS图像传感器中滚动快门的固有弱点 cs.CV

15 pages, 15 figures

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2101.10011v2)

**Authors**: Sebastian Köhler, Giulio Lovisotto, Simon Birnbach, Richard Baker, Ivan Martinovic

**Abstracts**: In this paper, we describe how the electronic rolling shutter in CMOS image sensors can be exploited using a bright, modulated light source (e.g., an inexpensive, off-the-shelf laser), to inject fine-grained image disruptions. We demonstrate the attack on seven different CMOS cameras, ranging from cheap IoT to semi-professional surveillance cameras, to highlight the wide applicability of the rolling shutter attack. We model the fundamental factors affecting a rolling shutter attack in an uncontrolled setting. We then perform an exhaustive evaluation of the attack's effect on the task of object detection, investigating the effect of attack parameters. We validate our model against empirical data collected on two separate cameras, showing that by simply using information from the camera's datasheet the adversary can accurately predict the injected distortion size and optimize their attack accordingly. We find that an adversary can hide up to 75% of objects perceived by state-of-the-art detectors by selecting appropriate attack parameters. We also investigate the stealthiness of the attack in comparison to a na\"{i}ve camera blinding attack, showing that common image distortion metrics can not detect the attack presence. Therefore, we present a new, accurate and lightweight enhancement to the backbone network of an object detector to recognize rolling shutter attacks. Overall, our results indicate that rolling shutter attacks can substantially reduce the performance and reliability of vision-based intelligent systems.

摘要: 在这篇文章中，我们描述了如何利用CMOS图像传感器中的电子滚动快门，使用明亮的调制光源(例如，廉价的现成激光器)来注入细粒度的图像干扰。我们演示了对七种不同CMOS摄像头的攻击，从廉价的物联网到半专业的监控摄像头，以突出滚动快门攻击的广泛适用性。我们模拟了在不受控制的环境下影响滚动快门攻击的基本因素。然后，我们对攻击对目标检测任务的影响进行了详尽的评估，考察了攻击参数的影响。我们通过在两个不同的摄像机上收集的经验数据验证了我们的模型，结果表明，通过简单地使用摄像机数据表中的信息，对手可以准确地预测注入失真的大小，并相应地优化他们的攻击。我们发现，通过选择合适的攻击参数，敌手可以隐藏最新检测器感知到的高达75%的对象。与单纯的相机盲攻击相比，我们还研究了该攻击的隐蔽性，发现普通的图像失真度量不能检测到攻击的存在，因此，我们提出了一种新的、准确的、轻量级的对象检测器主干网络增强方法来识别滚动快门攻击，结果表明，滚动快门攻击会大大降低基于视觉的智能系统的性能和可靠性。



## **6. Certified Adversarial Defenses Meet Out-of-Distribution Corruptions: Benchmarking Robustness and Simple Baselines**

认证的对抗性防御遇到分布外的腐败：基准、健壮性和简单的基线 cs.LG

21 pages, 15 figures, and 9 tables

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2112.00659v1)

**Authors**: Jiachen Sun, Akshay Mehra, Bhavya Kailkhura, Pin-Yu Chen, Dan Hendrycks, Jihun Hamm, Z. Morley Mao

**Abstracts**: Certified robustness guarantee gauges a model's robustness to test-time attacks and can assess the model's readiness for deployment in the real world. In this work, we critically examine how the adversarial robustness guarantees from randomized smoothing-based certification methods change when state-of-the-art certifiably robust models encounter out-of-distribution (OOD) data. Our analysis demonstrates a previously unknown vulnerability of these models to low-frequency OOD data such as weather-related corruptions, rendering these models unfit for deployment in the wild. To alleviate this issue, we propose a novel data augmentation scheme, FourierMix, that produces augmentations to improve the spectral coverage of the training data. Furthermore, we propose a new regularizer that encourages consistent predictions on noise perturbations of the augmented data to improve the quality of the smoothed models. We find that FourierMix augmentations help eliminate the spectral bias of certifiably robust models enabling them to achieve significantly better robustness guarantees on a range of OOD benchmarks. Our evaluation also uncovers the inability of current OOD benchmarks at highlighting the spectral biases of the models. To this end, we propose a comprehensive benchmarking suite that contains corruptions from different regions in the spectral domain. Evaluation of models trained with popular augmentation methods on the proposed suite highlights their spectral biases and establishes the superiority of FourierMix trained models at achieving better-certified robustness guarantees under OOD shifts over the entire frequency spectrum.

摘要: 认证的健壮性保证衡量模型对测试时间攻击的健壮性，并可以评估模型在现实世界中部署的准备情况。在这项工作中，我们批判性地研究了当最新的可证明鲁棒性模型遇到分布外(OOD)数据时，基于随机平滑的认证方法所保证的敌意鲁棒性是如何改变的。我们的分析表明，这些模型对低频OOD数据(如与天气相关的损坏)存在以前未知的脆弱性，使得这些模型不适合在野外部署。为了缓解这一问题，我们提出了一种新的数据增强方案FURIERMIX，该方案通过产生增强来提高训练数据的频谱覆盖率。此外，我们还提出了一种新的正则化方法，它鼓励对增强数据的噪声扰动进行一致的预测，以提高平滑模型的质量。我们发现，傅立叶混合增强有助于消除可证明的健壮性模型的频谱偏差，使它们能够在一系列面向对象设计基准上获得显着更好的健壮性保证。我们的评估还揭示了当前OOD基准在突出模型的光谱偏差方面的不足。为此，我们提出了一个全面的基准测试套件，该套件包含来自谱域中不同区域的腐败。在建议的套件上用流行的增强方法训练的模型的评估突出了它们的频谱偏差，并确立了傅里叶混合训练的模型在整个频谱上的OOD漂移下实现更好的认证鲁棒性保证的优势。



## **7. Well-classified Examples are Underestimated in Classification with Deep Neural Networks**

深度神经网络在分类中低估分类好的样本 cs.LG

Accepted by AAAI 2022; 16 pages, 11 figures, 13 tables

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2110.06537v3)

**Authors**: Guangxiang Zhao, Wenkai Yang, Xuancheng Ren, Lei Li, Xu Sun

**Abstracts**: The conventional wisdom behind learning deep classification models is to focus on bad-classified examples and ignore well-classified examples that are far from the decision boundary. For instance, when training with cross-entropy loss, examples with higher likelihoods (i.e., well-classified examples) contribute smaller gradients in back-propagation. However, we theoretically show that this common practice hinders representation learning, energy optimization, and the growth of margin. To counteract this deficiency, we propose to reward well-classified examples with additive bonuses to revive their contribution to learning. This counterexample theoretically addresses these three issues. We empirically support this claim by directly verify the theoretical results or through the significant performance improvement with our counterexample on diverse tasks, including image classification, graph classification, and machine translation. Furthermore, this paper shows that because our idea can solve these three issues, we can deal with complex scenarios, such as imbalanced classification, OOD detection, and applications under adversarial attacks. Code is available at: https://github.com/lancopku/well-classified-examples-are-underestimated.

摘要: 学习深度分类模型背后的传统智慧是专注于分类不良的示例，而忽略远离决策边界的分类良好的示例。例如，当使用交叉熵损失进行训练时，具有较高似然的示例(即，分类良好的示例)在反向传播中贡献较小的梯度。然而，我们从理论上表明，这种普遍的做法阻碍了表征学习、能量优化和边际增长。为了弥补这一不足，我们建议向分类良好的例子发放额外奖金，以恢复他们对学习的贡献。这个反例从理论上解决了这三个问题。我们通过直接验证理论结果或通过我们的反例在不同任务(包括图像分类、图形分类和机器翻译)上的显著性能改进来实证支持这一主张。此外，本文还表明，由于我们的思想可以解决这三个问题，所以我们可以处理复杂的场景，如不平衡分类、面向对象的检测以及在对抗性攻击下的应用。代码可在以下网址获得：https://github.com/lancopku/well-classified-examples-are-underestimated.



## **8. Understanding Adversarial Attacks on Observations in Deep Reinforcement Learning**

理解深度强化学习中对观测的敌意攻击 cs.LG

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2106.15860v2)

**Authors**: You Qiaoben, Chengyang Ying, Xinning Zhou, Hang Su, Jun Zhu, Bo Zhang

**Abstracts**: Deep reinforcement learning models are vulnerable to adversarial attacks that can decrease a victim's cumulative expected reward by manipulating the victim's observations. Despite the efficiency of previous optimization-based methods for generating adversarial noise in supervised learning, such methods might not be able to achieve the lowest cumulative reward since they do not explore the environmental dynamics in general. In this paper, we provide a framework to better understand the existing methods by reformulating the problem of adversarial attacks on reinforcement learning in the function space. Our reformulation generates an optimal adversary in the function space of the targeted attacks, repelling them via a generic two-stage framework. In the first stage, we train a deceptive policy by hacking the environment, and discover a set of trajectories routing to the lowest reward or the worst-case performance. Next, the adversary misleads the victim to imitate the deceptive policy by perturbing the observations. Compared to existing approaches, we theoretically show that our adversary is stronger under an appropriate noise level. Extensive experiments demonstrate our method's superiority in terms of efficiency and effectiveness, achieving the state-of-the-art performance in both Atari and MuJoCo environments.

摘要: 深度强化学习模型很容易受到敌意攻击，这种攻击会通过操纵受害者的观察结果来降低受害者的累积预期回报。尽管以前的基于优化的方法在监督学习中产生对抗性噪声是有效的，但是这些方法可能不能获得最低的累积奖励，因为它们通常不探索环境动态。本文通过在函数空间中重新表述强化学习的对抗性攻击问题，为更好地理解现有方法提供了一个框架。我们的重构在目标攻击的函数空间中生成一个最优对手，通过一个通用的两阶段框架击退它们。在第一阶段，我们通过黑客攻击环境来训练欺骗性策略，并发现一组通往最低回报或最坏情况表现的轨迹。接下来，对手通过扰乱观察来误导受害者模仿欺骗性的政策。与现有的方法相比，我们从理论上证明了在适当的噪声水平下，我们的对手更强。大量的实验证明了我们的方法在效率和有效性方面的优越性，在Atari和MuJoCo环境中都实现了最先进的性能。



## **9. Adv-4-Adv: Thwarting Changing Adversarial Perturbations via Adversarial Domain Adaptation**

ADV-4-ADV：通过对抗性领域适应挫败不断变化的对抗性扰动 cs.CV

9 pages

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2112.00428v1)

**Authors**: Tianyue Zheng, Zhe Chen, Shuya Ding, Chao Cai, Jun Luo

**Abstracts**: Whereas adversarial training can be useful against specific adversarial perturbations, they have also proven ineffective in generalizing towards attacks deviating from those used for training. However, we observe that this ineffectiveness is intrinsically connected to domain adaptability, another crucial issue in deep learning for which adversarial domain adaptation appears to be a promising solution. Consequently, we proposed Adv-4-Adv as a novel adversarial training method that aims to retain robustness against unseen adversarial perturbations. Essentially, Adv-4-Adv treats attacks incurring different perturbations as distinct domains, and by leveraging the power of adversarial domain adaptation, it aims to remove the domain/attack-specific features. This forces a trained model to learn a robust domain-invariant representation, which in turn enhances its generalization ability. Extensive evaluations on Fashion-MNIST, SVHN, CIFAR-10, and CIFAR-100 demonstrate that a model trained by Adv-4-Adv based on samples crafted by simple attacks (e.g., FGSM) can be generalized to more advanced attacks (e.g., PGD), and the performance exceeds state-of-the-art proposals on these datasets.

摘要: 虽然对抗性训练对对抗特定的对抗性干扰是有用的，但事实证明，它们也不能有效地概括出与用于训练的攻击不同的攻击。然而，我们观察到这种低效与领域适应性有内在的联系，这是深度学习中的另一个关键问题，对抗性领域适应似乎是一个有希望的解决方案。因此，我们提出了ADV-4-ADV作为一种新的对抗性训练方法，旨在保持对不可见的对抗性扰动的鲁棒性。从本质上讲，ADV-4-ADV将遭受不同扰动的攻击视为不同的域，并利用敌对域自适应的能力，旨在去除域/攻击特定的特征。这迫使训练后的模型学习健壮的领域不变表示，进而增强其泛化能力。在Fashion-MNIST、SVHN、CIFAR-10和CIFAR-100上的广泛评估表明，由ADV-4-ADV基于简单攻击(例如FGSM)构造的样本训练的模型可以推广到更高级的攻击(例如PGD)，并且性能超过了在这些数据集上的最新建议。



## **10. $\ell_\infty$-Robustness and Beyond: Unleashing Efficient Adversarial Training**

$\ell_\infty$-健壮性和超越：释放高效的对抗性训练 cs.LG

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2112.00378v1)

**Authors**: Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstracts**: Neural networks are vulnerable to adversarial attacks: adding well-crafted, imperceptible perturbations to their input can modify their output. Adversarial training is one of the most effective approaches in training robust models against such attacks. However, it is much slower than vanilla training of neural networks since it needs to construct adversarial examples for the entire training data at every iteration, which has hampered its effectiveness. Recently, Fast Adversarial Training was proposed that can obtain robust models efficiently. However, the reasons behind its success are not fully understood, and more importantly, it can only train robust models for $\ell_\infty$-bounded attacks as it uses FGSM during training. In this paper, by leveraging the theory of coreset selection we show how selecting a small subset of training data provides a more principled approach towards reducing the time complexity of robust training. Unlike existing methods, our approach can be adapted to a wide variety of training objectives, including TRADES, $\ell_p$-PGD, and Perceptual Adversarial Training. Our experimental results indicate that our approach speeds up adversarial training by 2-3 times, while experiencing a small reduction in the clean and robust accuracy.

摘要: 神经网络很容易受到敌意攻击：在它们的输入中添加精心设计的、不可察觉的扰动可以修改它们的输出。对抗性训练是训练鲁棒模型对抗此类攻击的最有效方法之一。但是，由于它在每次迭代时都需要为整个训练数据构造对抗性样本，因此比神经网络的香草训练慢得多，这就阻碍了它的有效性。最近，人们提出了一种快速对抗性训练方法，可以有效地获得稳健的模型。然而，其成功背后的原因还没有被完全理解，更重要的是，由于它在训练期间使用FGSM，所以它只能训练健壮的模型来应对$\ell_\$有界攻击。在本文中，通过利用核心选择理论，我们展示了如何选择一小部分训练数据提供了一种更有原则的方法来降低鲁棒训练的时间复杂度。与现有方法不同，我们的方法可以适应广泛的训练目标，包括行业、$\ell_p$-PGD和知觉对抗性训练。我们的实验结果表明，我们的方法将对抗性训练的速度提高了2-3倍，同时经历了干净和健壮的准确率的小幅下降。



## **11. Designing a Location Trace Anonymization Contest**

设计一个位置跟踪匿名化竞赛 cs.CR

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2107.10407v2)

**Authors**: Takao Murakami, Hiromi Arai, Koki Hamada, Takuma Hatano, Makoto Iguchi, Hiroaki Kikuchi, Atsushi Kuromasa, Hiroshi Nakagawa, Yuichi Nakamura, Kenshiro Nishiyama, Ryo Nojima, Hidenobu Oguri, Chiemi Watanabe, Akira Yamada, Takayasu Yamaguchi, Yuji Yamaoka

**Abstracts**: For a better understanding of anonymization methods for location traces, we have designed and held a location trace anonymization contest. Our contest deals with a long trace (400 events per user) and fine-grained locations (1024 regions). In our contest, each team anonymizes her original traces, and then the other teams perform privacy attacks against the anonymized traces in a partial-knowledge attacker model where the adversary does not know the original traces. To realize such a contest, we propose a location synthesizer that has diversity and utility; the synthesizer generates different synthetic traces for each team while preserving various statistical features of real traces. We also show that re-identification alone is insufficient as a privacy risk and that trace inference should be added as an additional risk. Specifically, we show an example of anonymization that is perfectly secure against re-identification and is not secure against trace inference. Based on this, our contest evaluates both the re-identification risk and trace inference risk and analyzes their relationship. Through our contest, we show several findings in a situation where both defense and attack compete together. In particular, we show that an anonymization method secure against trace inference is also secure against re-identification under the presence of appropriate pseudonymization.

摘要: 为了更好地了解位置踪迹的匿名化方法，我们设计并举办了位置踪迹匿名化大赛。我们的竞赛涉及长跟踪(每个用户400个事件)和细粒度位置(1024个区域)。在我们的比赛中，每个团队匿名她的原始痕迹，然后其他团队在部分知识攻击者模型中对匿名的痕迹进行隐私攻击，其中对手不知道原始痕迹。为了实现这样的竞赛，我们提出了一种具有多样性和实用性的位置合成器，该合成器在保留真实轨迹的各种统计特征的同时，为每个团队生成不同的合成轨迹。我们还表明，仅重新识别作为隐私风险是不够的，应该添加跟踪推断作为附加风险。具体地说，我们展示了一个匿名化的例子，它对于重新识别是完全安全的，而对于跟踪推理是不安全的。在此基础上，对再识别风险和痕迹推理风险进行了评估，并分析了它们之间的关系。通过我们的比赛，我们展示了在防守和进攻同时竞争的情况下的几个发现。特别地，我们证明了在存在适当的假名的情况下，一个安全的抗踪迹推理的匿名化方法也是安全的。



## **12. Push Stricter to Decide Better: A Class-Conditional Feature Adaptive Framework for Improving Adversarial Robustness**

越严越优：一种提高对手健壮性的类条件特征自适应框架 cs.CV

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2112.00323v1)

**Authors**: Jia-Li Yin, Lehui Xie, Wanqing Zhu, Ximeng Liu, Bo-Hao Chen

**Abstracts**: In response to the threat of adversarial examples, adversarial training provides an attractive option for enhancing the model robustness by training models on online-augmented adversarial examples. However, most of the existing adversarial training methods focus on improving the robust accuracy by strengthening the adversarial examples but neglecting the increasing shift between natural data and adversarial examples, leading to a dramatic decrease in natural accuracy. To maintain the trade-off between natural and robust accuracy, we alleviate the shift from the perspective of feature adaption and propose a Feature Adaptive Adversarial Training (FAAT) optimizing the class-conditional feature adaption across natural data and adversarial examples. Specifically, we propose to incorporate a class-conditional discriminator to encourage the features become (1) class-discriminative and (2) invariant to the change of adversarial attacks. The novel FAAT framework enables the trade-off between natural and robust accuracy by generating features with similar distribution across natural and adversarial data, and achieve higher overall robustness benefited from the class-discriminative feature characteristics. Experiments on various datasets demonstrate that FAAT produces more discriminative features and performs favorably against state-of-the-art methods. Codes are available at https://github.com/VisionFlow/FAAT.

摘要: 为了应对对抗性示例的威胁，对抗性训练通过训练在线扩充的对抗性示例模型，为增强模型的稳健性提供了一种有吸引力的选择。然而，现有的对抗性训练方法大多侧重于通过加强对抗性实例来提高鲁棒准确率，而忽略了自然数据与对抗性实例之间不断增加的偏移，导致自然精确度急剧下降。为了保持自然和鲁棒精度之间的折衷，我们从特征自适应的角度缓解了这一转变，并提出了一种特征自适应对抗训练(FAAT)，优化了跨自然数据和对抗性示例的类条件特征自适应。具体地说，我们建议加入类条件鉴别器，以鼓励特征成为(1)类可分辨的和(2)对敌方攻击变化不变的特征。新的FAAT框架通过在自然数据和对抗性数据上生成分布相似的特征，能够在自然和鲁棒精度之间进行权衡，并得益于类区分特征特性而获得更高的整体鲁棒性。在不同的数据集上的实验表明，FAAT产生了更具区分性的特征，并且与最先进的方法相比表现出了良好的性能。有关代码，请访问https://github.com/VisionFlow/FAAT.。



## **13. Adversarial Attacks Against Deep Generative Models on Data: A Survey**

针对数据深层生成模型的对抗性攻击：综述 cs.CR

To be published in IEEE Transactions on Knowledge and Data  Engineering

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2112.00247v1)

**Authors**: Hui Sun, Tianqing Zhu, Zhiqiu Zhang, Dawei Jin. Ping Xiong, Wanlei Zhou

**Abstracts**: Deep generative models have gained much attention given their ability to generate data for applications as varied as healthcare to financial technology to surveillance, and many more - the most popular models being generative adversarial networks and variational auto-encoders. Yet, as with all machine learning models, ever is the concern over security breaches and privacy leaks and deep generative models are no exception. These models have advanced so rapidly in recent years that work on their security is still in its infancy. In an attempt to audit the current and future threats against these models, and to provide a roadmap for defense preparations in the short term, we prepared this comprehensive and specialized survey on the security and privacy preservation of GANs and VAEs. Our focus is on the inner connection between attacks and model architectures and, more specifically, on five components of deep generative models: the training data, the latent code, the generators/decoders of GANs/ VAEs, the discriminators/encoders of GANs/ VAEs, and the generated data. For each model, component and attack, we review the current research progress and identify the key challenges. The paper concludes with a discussion of possible future attacks and research directions in the field.

摘要: 深度生成模型因其能够为从医疗保健到金融技术再到监控等各种应用程序生成数据而备受关注-最受欢迎的模型是生成性对抗性网络和变化式自动编码器。然而，与所有机器学习模型一样，人们一直担心安全漏洞和隐私泄露，深度生成模型也不例外。近年来，这些模式发展如此之快，其安全方面的工作仍处于初级阶段。为了审计这些模式当前和未来的威胁，并为短期内的防御准备提供路线图，我们准备了这项关于GAN和VAE的安全和隐私保护的全面而专业的调查。我们的重点是攻击和模型体系结构之间的内在联系，更具体地说，是深入生成模型的五个组成部分：训练数据、潜在代码、GANS/VAE的生成器/解码器、GANS/VAE的鉴别器/编码器和生成的数据。对于每个模型、组件和攻击，我们回顾了当前的研究进展，并确定了关键挑战。最后，对未来可能的攻击和该领域的研究方向进行了讨论。



## **14. Model Extraction Attacks on Graph Neural Networks: Taxonomy and Realization**

对图神经网络的模型提取攻击：分类与实现 cs.LG

This paper has been published in the 17th ACM ASIA Conference on  Computer and Communications Security (ACM ASIACCS 2022)

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2010.12751v2)

**Authors**: Bang Wu, Xiangwen Yang, Shirui Pan, Xingliang Yuan

**Abstracts**: Machine learning models are shown to face a severe threat from Model Extraction Attacks, where a well-trained private model owned by a service provider can be stolen by an attacker pretending as a client. Unfortunately, prior works focus on the models trained over the Euclidean space, e.g., images and texts, while how to extract a GNN model that contains a graph structure and node features is yet to be explored. In this paper, for the first time, we comprehensively investigate and develop model extraction attacks against GNN models. We first systematically formalise the threat modelling in the context of GNN model extraction and classify the adversarial threats into seven categories by considering different background knowledge of the attacker, e.g., attributes and/or neighbour connections of the nodes obtained by the attacker. Then we present detailed methods which utilise the accessible knowledge in each threat to implement the attacks. By evaluating over three real-world datasets, our attacks are shown to extract duplicated models effectively, i.e., 84% - 89% of the inputs in the target domain have the same output predictions as the victim model.

摘要: 机器学习模型面临着模型提取攻击的严重威胁，在这种攻击中，服务提供商拥有的训练有素的私有模型可能会被冒充客户端的攻击者窃取。遗憾的是，以前的工作主要集中在欧氏空间上训练的模型，例如图像和文本，而如何提取包含图结构和节点特征的GNN模型还有待探索。本文首次全面研究并开发了针对GNN模型的模型提取攻击。我们首先在GNN模型提取的背景下系统地形式化威胁建模，并通过考虑攻击者的不同背景知识(例如攻击者获取的节点的属性和/或邻居连接)将敌意威胁分类为七类。然后，我们给出了利用每个威胁中可访问的知识来实施攻击的详细方法。通过对三个真实数据集的评估，我们的攻击可以有效地提取重复模型，即目标领域中84%-89%的输入与受害者模型具有相同的输出预测。



## **15. Robust Multiple-Path Orienteering Problem: Securing Against Adversarial Attacks**

鲁棒多路径定向问题：抵抗敌方攻击的安全 cs.RO

submitted to TRO

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2003.13896v3)

**Authors**: Guangyao Shi, Lifeng Zhou, Pratap Tokekar

**Abstracts**: The multiple-path orienteering problem asks for paths for a team of robots that maximize the total reward collected while satisfying budget constraints on the path length. This problem models many multi-robot routing tasks such as exploring unknown environments and information gathering for environmental monitoring. In this paper, we focus on how to make the robot team robust to failures when operating in adversarial environments. We introduce the Robust Multiple-path Orienteering Problem (RMOP) where we seek worst-case guarantees against an adversary that is capable of attacking at most $\alpha$ robots. We consider two versions of this problem: RMOP offline and RMOP online. In the offline version, there is no communication or replanning when robots execute their plans and our main contribution is a general approximation scheme with a bounded approximation guarantee that depends on $\alpha$ and the approximation factor for single robot orienteering. In particular, we show that the algorithm yields a (i) constant-factor approximation when the cost function is modular; (ii) $\log$ factor approximation when the cost function is submodular; and (iii) constant-factor approximation when the cost function is submodular but the robots are allowed to exceed their path budgets by a bounded amount. In the online version, RMOP is modeled as a two-player sequential game and solved adaptively in a receding horizon fashion based on Monte Carlo Tree Search (MCTS). In addition to theoretical analysis, we perform simulation studies for ocean monitoring and tunnel information-gathering applications to demonstrate the efficacy of our approach.

摘要: 多路径定向问题要求一组机器人在满足路径长度的预算约束的同时最大化所收集的总奖励的路径。该问题模拟了许多多机器人路由任务，如探索未知环境和为环境监测收集信息。在本文中，我们重点研究如何使机器人团队在对抗性环境中工作时对故障具有健壮性。我们引入了鲁棒多路径定向问题(RMOP)，其中我们寻求对最多能够攻击$\α$机器人的对手的最坏情况保证。我们考虑此问题的两个版本：RMOP离线和RMOP在线。在离线版本中，机器人在执行其计划时不会进行通信或重新规划，我们的主要贡献是提供了一种一般的近似方案，它具有有界的逼近保证，它依赖于$\α和单个机器人定向的逼近因子。特别地，我们证明了当代价函数是模数时，该算法产生了常数因子近似；当代价函数是子模时，算法产生了$\log$因子逼近；以及(Iii)当代价函数是子模的，但允许机器人超出路径预算有限量时，算法得到了常数因子逼近。在在线版本中，RMOP被建模为一个两人序列博弈，并基于蒙特卡罗树搜索(MCTS)以滚动时域的方式自适应求解。除了理论分析外，我们还对海洋监测和隧道信息收集应用进行了仿真研究，以证明该方法的有效性。



## **16. Trustworthy Medical Segmentation with Uncertainty Estimation**

基于不确定性估计的可信医学分割 eess.IV

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.05978v2)

**Authors**: Giuseppina Carannante, Dimah Dera, Nidhal C. Bouaynaya, Ghulam Rasool, Hassan M. Fathallah-Shaykh

**Abstracts**: Deep Learning (DL) holds great promise in reshaping the healthcare systems given its precision, efficiency, and objectivity. However, the brittleness of DL models to noisy and out-of-distribution inputs is ailing their deployment in the clinic. Most systems produce point estimates without further information about model uncertainty or confidence. This paper introduces a new Bayesian deep learning framework for uncertainty quantification in segmentation neural networks, specifically encoder-decoder architectures. The proposed framework uses the first-order Taylor series approximation to propagate and learn the first two moments (mean and covariance) of the distribution of the model parameters given the training data by maximizing the evidence lower bound. The output consists of two maps: the segmented image and the uncertainty map of the segmentation. The uncertainty in the segmentation decisions is captured by the covariance matrix of the predictive distribution. We evaluate the proposed framework on medical image segmentation data from Magnetic Resonances Imaging and Computed Tomography scans. Our experiments on multiple benchmark datasets demonstrate that the proposed framework is more robust to noise and adversarial attacks as compared to state-of-the-art segmentation models. Moreover, the uncertainty map of the proposed framework associates low confidence (or equivalently high uncertainty) to patches in the test input images that are corrupted with noise, artifacts or adversarial attacks. Thus, the model can self-assess its segmentation decisions when it makes an erroneous prediction or misses part of the segmentation structures, e.g., tumor, by presenting higher values in the uncertainty map.

摘要: 深度学习(DL)由于其精确性、效率和客观性，在重塑医疗系统方面有着巨大的希望。然而，DL模型对噪声和非分布输入的脆性阻碍了它们在临床上的部署。大多数系统在没有关于模型不确定性或置信度的进一步信息的情况下产生点估计。本文介绍了一种新的贝叶斯深度学习框架，用于分段神经网络中的不确定性量化，特别是编解码器的体系结构。该框架使用一阶泰勒级数近似，通过最大化证据下界来传播和学习给定训练数据的模型参数分布的前两个矩(均值和协方差)。输出由两幅图组成：分割后的图像和分割的不确定性图。通过预测分布的协方差矩阵来捕捉分割决策中的不确定性。我们在磁共振成像和计算机断层扫描的医学图像分割数据上对所提出的框架进行了评估。我们在多个基准数据集上的实验表明，与现有的分割模型相比，该框架对噪声和敌意攻击具有更强的鲁棒性。此外，该框架的不确定性图将低置信度(或相当于高不确定性)与测试输入图像中被噪声、伪影或敌意攻击破坏的补丁相关联。因此，当模型做出错误的预测或通过在不确定性图中呈现更高的值来错过部分分割结构(例如，肿瘤)时，该模型可以自我评估其分割决策。



## **17. Defending Against Adversarial Denial-of-Service Data Poisoning Attacks**

防御敌意的拒绝服务数据中毒攻击 cs.CR

Published at ACSAC DYNAMICS 2020

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2104.06744v3)

**Authors**: Nicolas M. Müller, Simon Roschmann, Konstantin Böttinger

**Abstracts**: Data poisoning is one of the most relevant security threats against machine learning and data-driven technologies. Since many applications rely on untrusted training data, an attacker can easily craft malicious samples and inject them into the training dataset to degrade the performance of machine learning models. As recent work has shown, such Denial-of-Service (DoS) data poisoning attacks are highly effective. To mitigate this threat, we propose a new approach of detecting DoS poisoned instances. In comparison to related work, we deviate from clustering and anomaly detection based approaches, which often suffer from the curse of dimensionality and arbitrary anomaly threshold selection. Rather, our defence is based on extracting information from the training data in such a generalized manner that we can identify poisoned samples based on the information present in the unpoisoned portion of the data. We evaluate our defence against two DoS poisoning attacks and seven datasets, and find that it reliably identifies poisoned instances. In comparison to related work, our defence improves false positive / false negative rates by at least 50%, often more.

摘要: 数据中毒是机器学习和数据驱动技术面临的最相关的安全威胁之一。由于许多应用程序依赖于不可信的训练数据，攻击者可以很容易地手工制作恶意样本并将其注入到训练数据集中，从而降低机器学习模型的性能。最近的研究表明，这种拒绝服务(DoS)数据中毒攻击非常有效。为了缓解这种威胁，我们提出了一种检测DoS中毒实例的新方法。与相关工作相比，我们偏离了基于聚类和异常检测的方法，这些方法经常受到维数灾难和任意选择异常阈值的影响。相反，我们的辩护是基于以一种普遍的方式从训练数据中提取信息，以便我们可以基于数据的未中毒部分中存在的信息来识别中毒样本。我们评估了我们对两个DoS中毒攻击和七个数据集的防御，发现它可以可靠地识别中毒实例。与相关工作相比，我们的辩护将假阳性/假阴性率提高了至少50%，往往更高。



## **18. FROB: Few-shot ROBust Model for Classification and Out-of-Distribution Detection**

FROB：一种用于分类和越界检测的少射鲁棒模型 cs.LG

Paper, 22 pages, Figures, Tables

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.15487v1)

**Authors**: Nikolaos Dionelis

**Abstracts**: Nowadays, classification and Out-of-Distribution (OoD) detection in the few-shot setting remain challenging aims due to rarity and the limited samples in the few-shot setting, and because of adversarial attacks. Accomplishing these aims is important for critical systems in safety, security, and defence. In parallel, OoD detection is challenging since deep neural network classifiers set high confidence to OoD samples away from the training data. To address such limitations, we propose the Few-shot ROBust (FROB) model for classification and few-shot OoD detection. We devise FROB for improved robustness and reliable confidence prediction for few-shot OoD detection. We generate the support boundary of the normal class distribution and combine it with few-shot Outlier Exposure (OE). We propose a self-supervised learning few-shot confidence boundary methodology based on generative and discriminative models. The contribution of FROB is the combination of the generated boundary in a self-supervised learning manner and the imposition of low confidence at this learned boundary. FROB implicitly generates strong adversarial samples on the boundary and forces samples from OoD, including our boundary, to be less confident by the classifier. FROB achieves generalization to unseen OoD with applicability to unknown, in the wild, test sets that do not correlate to the training datasets. To improve robustness, FROB redesigns OE to work even for zero-shots. By including our boundary, FROB reduces the threshold linked to the model's few-shot robustness; it maintains the OoD performance approximately independent of the number of few-shots. The few-shot robustness analysis evaluation of FROB on different sets and on One-Class Classification (OCC) data shows that FROB achieves competitive performance and outperforms benchmarks in terms of robustness to the outlier few-shot sample population and variability.

摘要: 目前，少射环境下的分类和失配(OOD)检测仍然是一个极具挑战性的课题，因为少射环境下的稀有性和有限的样本，以及敌方攻击的存在。实现这些目标对于安全、安保和防御方面的关键系统非常重要。同时，由于深度神经网络分类器对远离训练数据的OOD样本设置了很高的置信度，因此OOD检测是具有挑战性的。为了解决这些局限性，我们提出了用于分类和少射OOD检测的少射鲁棒(FROB)模型。我们设计了FROB来提高少射OOD检测的鲁棒性和可靠的置信度预测。我们生成正态类分布的支持边界，并将其与少镜头离群点曝光(OE)相结合。提出了一种基于产生式模型和判别式模型的自监督学习小概率置信边界方法。FROB的贡献是将以自监督学习方式生成的边界与在该学习边界上施加的低置信度相结合。FROB隐式地在边界上生成强对抗性样本，并迫使来自OOD(包括我们的边界)的样本被分类器降低信心。FROB实现了对看不见的OOD的泛化，适用于与训练数据集不相关的未知的野外测试集。为了提高健壮性，FROB重新设计了OE，使其即使在零炮情况下也能工作。通过包括我们的边界，FROB降低了与模型的少镜头稳健性相关的阈值；它保持了OOD性能与少镜头数量大致无关。在不同集合和一类分类数据上的少镜头稳健性分析评估表明，在对离群点、少镜头样本总体和变异性的鲁棒性方面，FROB达到了好胜的性能，并优于基准测试结果。结果表明，FROB达到了好胜的性能，并且在对异常点、少镜头样本总体和变异性的鲁棒性方面优于基准。



## **19. A Face Recognition System's Worst Morph Nightmare, Theoretically**

从理论上讲，人脸识别系统最糟糕的梦魇 cs.CV

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.15416v1)

**Authors**: Una M. Kelly, Raymond Veldhuis, Luuk Spreeuwers

**Abstracts**: It has been shown that Face Recognition Systems (FRSs) are vulnerable to morphing attacks, but most research focusses on landmark-based morphs. A second method for generating morphs uses Generative Adversarial Networks, which results in convincingly real facial images that can be almost as challenging for FRSs as landmark-based attacks. We propose a method to create a third, different type of morph, that has the advantage of being easier to train. We introduce the theoretical concept of \textit{worst-case morphs}, which are those morphs that are most challenging for a fixed FRS. For a set of images and corresponding embeddings in an FRS's latent space, we generate images that approximate these worst-case morphs using a mapping from embedding space back to image space. While the resulting images are not yet as challenging as other morphs, they can provide valuable information in future research on Morphing Attack Detection (MAD) methods and on weaknesses of FRSs. Methods for MAD need to be validated on more varied morph databases. Our proposed method contributes to achieving such variation.

摘要: 已有研究表明，人脸识别系统(FRS)容易受到变形攻击，但大多数研究都集中在基于标志性的变形攻击上。第二种生成变形的方法使用生成性对抗网络，它产生令人信服的真实面部图像，这对FRS来说几乎和基于里程碑的攻击一样具有挑战性。我们提出了一种方法来创建第三种不同类型的变形，其优点是更容易训练。我们引入了最坏情况变形的理论概念，它们是对固定FRS最具挑战性的变形。对于FRS的潜在空间中的一组图像和相应的嵌入，我们使用从嵌入空间返回到图像空间的映射来生成近似这些最坏情况下的变形的图像。虽然生成的图像还不像其他变形图像那样具有挑战性，但它们可以为未来变形攻击检测(MAD)方法的研究和FRS的弱点提供有价值的信息。MAD的方法需要在更多不同的变形数据库上进行验证。我们提出的方法有助于实现这种变化。



## **20. Medical Aegis: Robust adversarial protectors for medical images**

医学宙斯盾：强大的医学图像对抗性保护者 cs.CV

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.10969v2)

**Authors**: Qingsong Yao, Zecheng He, S. Kevin Zhou

**Abstracts**: Deep neural network based medical image systems are vulnerable to adversarial examples. Many defense mechanisms have been proposed in the literature, however, the existing defenses assume a passive attacker who knows little about the defense system and does not change the attack strategy according to the defense. Recent works have shown that a strong adaptive attack, where an attacker is assumed to have full knowledge about the defense system, can easily bypass the existing defenses. In this paper, we propose a novel adversarial example defense system called Medical Aegis. To the best of our knowledge, Medical Aegis is the first defense in the literature that successfully addresses the strong adaptive adversarial example attacks to medical images. Medical Aegis boasts two-tier protectors: The first tier of Cushion weakens the adversarial manipulation capability of an attack by removing its high-frequency components, yet posing a minimal effect on classification performance of the original image; the second tier of Shield learns a set of per-class DNN models to predict the logits of the protected model. Deviation from the Shield's prediction indicates adversarial examples. Shield is inspired by the observations in our stress tests that there exist robust trails in the shallow layers of a DNN model, which the adaptive attacks can hardly destruct. Experimental results show that the proposed defense accurately detects adaptive attacks, with negligible overhead for model inference.

摘要: 基于深度神经网络的医学图像系统容易受到敌意例子的攻击。文献中提出了很多防御机制，但现有的防御机制都假设攻击者是被动的，对防御系统知之甚少，不会根据防御情况改变攻击策略。最近的研究表明，强自适应攻击(假设攻击者完全了解防御系统)可以很容易地绕过现有的防御系统。在本文中，我们提出了一种新的对抗性实例防御系统--医学宙斯盾(Medical Aegis)。据我们所知，医学宙斯盾是文献中第一个成功解决了对医学图像的强适应性敌意攻击的防御方案。医学宙斯盾拥有两层保护器：第一层缓冲通过移除攻击的高频分量来削弱攻击的敌意操纵能力，但对原始图像的分类性能影响最小；第二层Shield学习一组按类的DNN模型来预测受保护模型的逻辑。与盾牌的预测背道而驰表明了敌对的例子。Shield的灵感来自于我们在压力测试中观察到的DNN模型的浅层存在健壮的踪迹，而自适应攻击很难破坏这些踪迹。实验结果表明，该防御方法能够准确检测出自适应攻击，而模型推理开销可以忽略不计。



## **21. COREATTACK: Breaking Up the Core Structure of Graphs**

COREATTACK：打破图的核心结构 cs.SI

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.15276v1)

**Authors**: Bo Zhou, Yuqian Lv, Jinhuan Wang, Jian Zhang, Qi Xuan

**Abstracts**: The concept of k-core in complex networks plays a key role in many applications, e.g., understanding the global structure, or identifying central/critical nodes, of a network. A malicious attacker with jamming ability can exploit the vulnerability of the k-core structure to attack the network and invalidate the network analysis methods, e.g., reducing the k-shell values of nodes can deceive graph algorithms, leading to the wrong decisions. In this paper, we investigate the robustness of the k-core structure under adversarial attacks by deleting edges, for the first time. Firstly, we give the general definition of targeted k-core attack, map it to the set cover problem which is NP-hard, and further introduce a series of evaluation metrics to measure the performance of attack methods. Then, we propose $Q$ index theoretically as the probability that the terminal node of an edge does not belong to the innermost core, which is further used to guide the design of our heuristic attack methods, namely COREATTACK and GreedyCOREATTACK. The experiments on a variety of real-world networks demonstrate that our methods behave much better than a series of baselines, in terms of much smaller Edge Change Rate (ECR) and False Attack Rate (FAR), achieving state-of-the-art attack performance. More impressively, for certain real-world networks, only deleting one edge from the k-core may lead to the collapse of the innermost core, even if this core contains dozens of nodes. Such a phenomenon indicates that the k-core structure could be extremely vulnerable under adversarial attacks, and its robustness thus should be carefully addressed to ensure the security of many graph algorithms.

摘要: 在复杂网络中，k核的概念在许多应用中起着关键作用，例如，理解网络的全局结构或识别网络的中心/关键节点。具有干扰能力的恶意攻击者可以利用k-core结构的漏洞攻击网络，使网络分析方法失效，例如降低节点的k-shell值可以欺骗图算法，导致错误的决策。本文首次通过删除边的方法研究了k-core结构在敌意攻击下的健壮性。首先，给出了目标k-核攻击的一般定义，将其映射到NP-hard的集合覆盖问题，并进一步引入了一系列评价指标来衡量攻击方法的性能。然后，从理论上提出了$q$指标作为边的末端节点不属于最内核节点的概率，并进一步用它来指导我们的启发式攻击方法COREATTACK和GreedyCOREATTACK的设计。在各种真实网络上的实验表明，我们的方法在边缘变化率(ECR)和错误攻击率(FAR)方面明显优于一系列基线，达到了最先进的攻击性能。更令人印象深刻的是，对于某些现实世界的网络，只从k核中删除一条边可能会导致最里面的核崩溃，即使这个核包含几十个节点。这种现象表明k-core结构在敌意攻击下极易受到攻击，因此其健壮性是保证许多图算法安全的关键。



## **22. Black-box Adversarial Attacks on Commercial Speech Platforms with Minimal Information**

基于最小信息的商业语音平台黑盒对抗性攻击 cs.CR

A version of this paper appears in the proceedings of the 28th ACM  Conference on Computer and Communications Security (CCS 2021). The notes in  Tables 1 and 4 have been updated

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2110.09714v2)

**Authors**: Baolin Zheng, Peipei Jiang, Qian Wang, Qi Li, Chao Shen, Cong Wang, Yunjie Ge, Qingyang Teng, Shenyi Zhang

**Abstracts**: Adversarial attacks against commercial black-box speech platforms, including cloud speech APIs and voice control devices, have received little attention until recent years. The current "black-box" attacks all heavily rely on the knowledge of prediction/confidence scores to craft effective adversarial examples, which can be intuitively defended by service providers without returning these messages. In this paper, we propose two novel adversarial attacks in more practical and rigorous scenarios. For commercial cloud speech APIs, we propose Occam, a decision-only black-box adversarial attack, where only final decisions are available to the adversary. In Occam, we formulate the decision-only AE generation as a discontinuous large-scale global optimization problem, and solve it by adaptively decomposing this complicated problem into a set of sub-problems and cooperatively optimizing each one. Our Occam is a one-size-fits-all approach, which achieves 100% success rates of attacks with an average SNR of 14.23dB, on a wide range of popular speech and speaker recognition APIs, including Google, Alibaba, Microsoft, Tencent, iFlytek, and Jingdong, outperforming the state-of-the-art black-box attacks. For commercial voice control devices, we propose NI-Occam, the first non-interactive physical adversarial attack, where the adversary does not need to query the oracle and has no access to its internal information and training data. We combine adversarial attacks with model inversion attacks, and thus generate the physically-effective audio AEs with high transferability without any interaction with target devices. Our experimental results show that NI-Occam can successfully fool Apple Siri, Microsoft Cortana, Google Assistant, iFlytek and Amazon Echo with an average SRoA of 52% and SNR of 9.65dB, shedding light on non-interactive physical attacks against voice control devices.

摘要: 针对商业黑盒语音平台的对抗性攻击，包括云语音API和语音控制设备，直到最近几年才受到很少关注。目前的“黑匣子”攻击都严重依赖预测/置信分数的知识来制作有效的对抗性例子，服务提供商无需返回这些消息就可以直观地进行防御。在这篇文章中，我们提出了两个新的对抗性攻击，在更实际和更严格的情况下。对于商用云语音API，我们提出了OCCAM，这是一种仅限决策的黑盒对抗攻击，只有最终决策才能提供给对手。在OCCAM中，我们将只有决策的AE生成问题描述为一个不连续的大规模全局优化问题，并将这个复杂问题自适应地分解成一组子问题并对每个子问题进行协同优化来求解。我们的OCCAM是一种一刀切的方法，在谷歌、阿里巴巴、微软、腾讯、iFLYTEK、京东等各种流行的语音和说话人识别API上，实现了100%的攻击成功率，平均SNR为14.23dB，表现优于最先进的黑匣子攻击。对于商用语音控制设备，我们提出了NI-OCCAM，这是第一种非交互式的物理对抗攻击，对手不需要查询先知，也不能访问它的内部信息和训练数据。我们将对抗性攻击和模型反转攻击相结合，在不与目标设备交互的情况下生成物理上有效、可移植性高的音频AEs。我们的实验结果表明，NI-Occam能够成功欺骗Apple Siri、Microsoft Cortana、Google Assistant、iFLYTEK和Amazon Echo，平均SRoA达到52%，信噪比达到9.65dB，为针对语音控制设备的非交互式物理攻击提供了线索。



## **23. Mitigating Adversarial Attacks by Distributing Different Copies to Different Users**

通过将不同的副本分发给不同的用户来缓解敌意攻击 cs.CR

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.15160v1)

**Authors**: Jiyi Zhang, Wesley Joon-Wie Tann, Ee-Chien Chang

**Abstracts**: Machine learning models are vulnerable to adversarial attacks. In this paper, we consider the scenario where a model is to be distributed to multiple users, among which a malicious user attempts to attack another user. The malicious user probes its copy of the model to search for adversarial samples and then presents the found samples to the victim's model in order to replicate the attack. We point out that by distributing different copies of the model to different users, we can mitigate the attack such that adversarial samples found on one copy would not work on another copy. We first observed that training a model with different randomness indeed mitigates such replication to certain degree. However, there is no guarantee and retraining is computationally expensive. Next, we propose a flexible parameter rewriting method that directly modifies the model's parameters. This method does not require additional training and is able to induce different sets of adversarial samples in different copies in a more controllable manner. Experimentation studies show that our approach can significantly mitigate the attacks while retaining high classification accuracy. From this study, we believe that there are many further directions worth exploring.

摘要: 机器学习模型容易受到敌意攻击。在本文中，我们考虑将一个模型分发给多个用户的场景，其中一个恶意用户试图攻击另一个用户。恶意用户探测其模型副本以搜索敌意样本，然后将找到的样本呈现给受害者的模型，以便复制攻击。我们指出，通过将模型的不同副本分发给不同的用户，我们可以减轻攻击，使得在一个副本上发现的敌意样本在另一个副本上不起作用。我们首先观察到，训练具有不同随机性的模型确实在一定程度上减轻了这种复制。然而，这是没有保证的，再培训在计算上是昂贵的。接下来，我们提出了一种灵活的参数重写方法，可以直接修改模型的参数。这种方法不需要额外的训练，并且能够以更可控的方式在不同的副本中诱导不同的对抗性样本集。实验研究表明，该方法在保持较高分类准确率的同时，能显着缓解攻击。从本次研究来看，我们认为还有很多值得进一步探索的方向。



## **24. Adversarial Robustness of Deep Code Comment Generation**

深层代码注释生成的对抗健壮性 cs.SE

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2108.00213v3)

**Authors**: Yu Zhou, Xiaoqing Zhang, Juanjuan Shen, Tingting Han, Taolue Chen, Harald Gall

**Abstracts**: Deep neural networks (DNNs) have shown remarkable performance in a variety of domains such as computer vision, speech recognition, or natural language processing. Recently they also have been applied to various software engineering tasks, typically involving processing source code. DNNs are well-known to be vulnerable to adversarial examples, i.e., fabricated inputs that could lead to various misbehaviors of the DNN model while being perceived as benign by humans. In this paper, we focus on the code comment generation task in software engineering and study the robustness issue of the DNNs when they are applied to this task. We propose ACCENT, an identifier substitution approach to craft adversarial code snippets, which are syntactically correct and semantically close to the original code snippet, but may mislead the DNNs to produce completely irrelevant code comments. In order to improve the robustness, ACCENT also incorporates a novel training method, which can be applied to existing code comment generation models. We conduct comprehensive experiments to evaluate our approach by attacking the mainstream encoder-decoder architectures on two large-scale publicly available datasets. The results show that ACCENT efficiently produces stable attacks with functionality-preserving adversarial examples, and the generated examples have better transferability compared with baselines. We also confirm, via experiments, the effectiveness in improving model robustness with our training method.

摘要: 深度神经网络(DNNs)在计算机视觉、语音识别、自然语言处理等领域表现出显著的性能。最近，它们还被应用于各种软件工程任务，通常涉及处理源代码。众所周知，DNN很容易受到敌意示例的攻击，即在人类认为DNN模型是良性的情况下，可能会导致DNN模型的各种错误行为的捏造输入。本文针对软件工程中的代码注释生成任务，研究了DNN应用于该任务时的健壮性问题。我们提出了一种标识符替换方法Accent来制作敌意代码片段，这些代码片段在语法上是正确的，在语义上也接近于原始代码片段，但可能会误导DNN生成完全不相关的代码注释。为了提高鲁棒性，Accent还引入了一种新的训练方法，该方法可以应用于现有的代码注释生成模型。我们在两个大规模公开可用的数据集上进行了全面的实验，通过攻击主流的编解码器架构来评估我们的方法。实验结果表明，重音算法能有效地产生稳定的攻击，且生成的实例与基线相比具有更好的可移植性。通过实验，我们也证实了我们的训练方法在提高模型鲁棒性方面的有效性。



## **25. Living-Off-The-Land Command Detection Using Active Learning**

基于主动学习的陆上生活指挥检测 cs.CR

14 pages, published in RAID 2021

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.15039v1)

**Authors**: Talha Ongun, Jack W. Stokes, Jonathan Bar Or, Ke Tian, Farid Tajaddodianfar, Joshua Neil, Christian Seifert, Alina Oprea, John C. Platt

**Abstracts**: In recent years, enterprises have been targeted by advanced adversaries who leverage creative ways to infiltrate their systems and move laterally to gain access to critical data. One increasingly common evasive method is to hide the malicious activity behind a benign program by using tools that are already installed on user computers. These programs are usually part of the operating system distribution or another user-installed binary, therefore this type of attack is called "Living-Off-The-Land". Detecting these attacks is challenging, as adversaries may not create malicious files on the victim computers and anti-virus scans fail to detect them. We propose the design of an Active Learning framework called LOLAL for detecting Living-Off-the-Land attacks that iteratively selects a set of uncertain and anomalous samples for labeling by a human analyst. LOLAL is specifically designed to work well when a limited number of labeled samples are available for training machine learning models to detect attacks. We investigate methods to represent command-line text using word-embedding techniques, and design ensemble boosting classifiers to distinguish malicious and benign samples based on the embedding representation. We leverage a large, anonymized dataset collected by an endpoint security product and demonstrate that our ensemble classifiers achieve an average F1 score of 0.96 at classifying different attack classes. We show that our active learning method consistently improves the classifier performance, as more training data is labeled, and converges in less than 30 iterations when starting with a small number of labeled instances.

摘要: 近年来，企业一直是高级对手的目标，他们利用创造性的方式渗透到他们的系统中，并横向移动以获取关键数据。一种越来越常见的规避方法是通过使用已安装在用户计算机上的工具将恶意活动隐藏在良性程序后面。这些程序通常是操作系统发行版或其他用户安装的二进制文件的一部分，因此这种类型的攻击被称为“生活在陆地上”(Living-Off-Land)。检测这些攻击具有挑战性，因为攻击者可能不会在受攻击的计算机上创建恶意文件，并且防病毒扫描无法检测到它们。我们提出了一个称为LOLAL的主动学习框架来检测生活在陆地上的攻击，该框架迭代地选择一组不确定和异常的样本供人类分析员进行标记。LOLAL专门设计用于在有限数量的标签样本可用于训练机器学习模型以检测攻击时很好地工作。我们研究了使用词嵌入技术来表示命令行文本的方法，并设计了基于嵌入表示的集成Boosting分类器来区分恶意样本和良性样本。我们利用终端安全产品收集的大型匿名数据集，展示了我们的集成分类器在分类不同攻击类别时的平均F1得分为0.96。我们表明，随着更多的训练数据被标注，我们的主动学习方法持续地提高了分类器的性能，并且当从少量的标注实例开始时，在不到30次的迭代中就收敛了。



## **26. Understanding and Increasing Efficiency of Frank-Wolfe Adversarial Training**

对弗兰克-沃尔夫对抗性训练的认识与提高效率 cs.LG

Accepted to ICML 2021 Adversarial Machine Learning Workshop. Under  review

**SubmitDate**: 2021-11-29    [paper-pdf](http://arxiv.org/pdf/2012.12368v4)

**Authors**: Theodoros Tsiligkaridis, Jay Roberts

**Abstracts**: Deep neural networks are easily fooled by small perturbations known as adversarial attacks. Adversarial Training (AT) is a technique that approximately solves a robust optimization problem to minimize the worst-case loss and is widely regarded as the most effective defense against such attacks. Due to the high computation time for generating strong adversarial examples for AT, single-step approaches have been proposed to reduce training time. However, these methods suffer from catastrophic overfitting where adversarial accuracy drops during training. Although improvements have been proposed, they increase training time and robustness is far from that of multi-step AT. We develop a theoretical framework for adversarial training with FW optimization (FW-AT) that reveals a geometric connection between the loss landscape and the $\ell_2$ distortion of $\ell_\infty$ FW attacks. We analytically show that high distortion of FW attacks is equivalent to small gradient variation along the attack path. It is then experimentally demonstrated on various deep neural network architectures that $\ell_\infty$ attacks against robust models achieve near maximal $\ell_2$ distortion, while standard networks have lower distortion. Furthermore, it is experimentally shown that catastrophic overfitting is strongly correlated with low distortion of FW attacks. To demonstrate the utility of our theoretical framework we develop FW-AT-Adapt, a novel adversarial training algorithm which uses a simple distortion measure to adapt the number of attack steps to increase efficiency without compromising robustness. FW-AT-Adapt provides training times on par with single-step fast AT methods and improves closing the gap between fast AT methods and multi-step PGD-AT with minimal loss in adversarial accuracy in white-box and black-box settings.

摘要: 深度神经网络很容易被称为对抗性攻击的小扰动所愚弄。对抗性训练(AT)是一种近似地解决鲁棒优化问题以最小化最坏情况下的损失的技术，被广泛认为是对抗此类攻击的最有效的防御方法。由于AT生成强对抗性样本的计算时间较长，因此提出了一种单步方法来减少训练时间。然而，这些方法存在灾难性的过度拟合问题，在训练过程中对抗性准确率会下降。虽然已经提出了一些改进措施，但它们增加了训练时间，鲁棒性远不及多步AT。我们建立了一个基于FW优化的对抗性训练理论框架(FW-AT)，该框架揭示了损失情况与$ELL_INFTY$FW攻击的$\ELL_2$失真之间的几何关系。分析表明，FW攻击的高失真等价于攻击路径上的小梯度变化。在不同的深度神经网络结构上实验证明，对鲁棒模型的$ellinty$攻击获得了接近最大的$ell2$失真，而标准网络具有较低的失真。此外，实验还表明，灾难性过拟合与FW攻击的低失真有很强的相关性。为了证明我们的理论框架的有效性，我们开发了一种新的对抗性训练算法FW-AT-Adapt，它使用一个简单的失真度量来调整攻击步骤的数量，从而在不影响鲁棒性的情况下提高效率。FW-AT-Adapt提供与单步快速AT方法相当的训练时间，并改善了快速AT方法与多步PGD-AT之间的差距，同时最大限度地降低了白盒和黑盒设置中的对抗精度损失。



## **27. MedRDF: A Robust and Retrain-Less Diagnostic Framework for Medical Pretrained Models Against Adversarial Attack**

MedRDF：一种健壮且无需再训练的医学预训练模型对抗攻击诊断框架 cs.CV

TMI under review

**SubmitDate**: 2021-11-29    [paper-pdf](http://arxiv.org/pdf/2111.14564v1)

**Authors**: Mengting Xu, Tao Zhang, Daoqiang Zhang

**Abstracts**: Deep neural networks are discovered to be non-robust when attacked by imperceptible adversarial examples, which is dangerous for it applied into medical diagnostic system that requires high reliability. However, the defense methods that have good effect in natural images may not be suitable for medical diagnostic tasks. The preprocessing methods (e.g., random resizing, compression) may lead to the loss of the small lesions feature in the medical image. Retraining the network on the augmented data set is also not practical for medical models that have already been deployed online. Accordingly, it is necessary to design an easy-to-deploy and effective defense framework for medical diagnostic tasks. In this paper, we propose a Robust and Retrain-Less Diagnostic Framework for Medical pretrained models against adversarial attack (i.e., MedRDF). It acts on the inference time of the pertained medical model. Specifically, for each test image, MedRDF firstly creates a large number of noisy copies of it, and obtains the output labels of these copies from the pretrained medical diagnostic model. Then, based on the labels of these copies, MedRDF outputs the final robust diagnostic result by majority voting. In addition to the diagnostic result, MedRDF produces the Robust Metric (RM) as the confidence of the result. Therefore, it is convenient and reliable to utilize MedRDF to convert pre-trained non-robust diagnostic models into robust ones. The experimental results on COVID-19 and DermaMNIST datasets verify the effectiveness of our MedRDF in improving the robustness of medical diagnostic models.

摘要: 由于深层神经网络应用于可靠性要求高的医疗诊断系统，在受到潜伏的敌意攻击时表现出非稳健性，这是很危险的。然而，在自然图像中效果较好的防御方法可能不适用于医学诊断任务。预处理方法(例如，随机调整大小、压缩)可能会导致医学图像中的小病变特征丢失。对于已经在线部署的医疗模型来说，在增强的数据集上重新培训网络也是不切实际的。因此，有必要为医疗诊断任务设计一个易于部署和有效的防御框架。在本文中，我们提出了一个健壮且无需再训练的医学预训练模型抗攻击诊断框架(MedRDF)。它作用于相关医学模型的推理时间。具体地说，对于每幅测试图像，MedRDF首先为其创建大量的噪声副本，并从预先训练的医疗诊断模型中获得这些副本的输出标签。然后，基于这些副本的标签，MedRDF通过多数投票输出最终的鲁棒诊断结果。除了诊断结果之外，MedRDF还生成稳健度量(RM)作为结果的置信度。因此，利用MedRDF将预先训练好的非稳健诊断模型转换为稳健的诊断模型是方便可靠的。在冠状病毒和DermaMNIST数据集上的实验结果验证了我们的MedRDF在提高医疗诊断模型鲁棒性方面的有效性。



## **28. Reliably fast adversarial training via latent adversarial perturbation**

通过潜在的对抗性扰动进行可靠的快速对抗性训练 cs.LG

ICCV 2021 (Oral)

**SubmitDate**: 2021-11-29    [paper-pdf](http://arxiv.org/pdf/2104.01575v2)

**Authors**: Geon Yeong Park, Sang Wan Lee

**Abstracts**: While multi-step adversarial training is widely popular as an effective defense method against strong adversarial attacks, its computational cost is notoriously expensive, compared to standard training. Several single-step adversarial training methods have been proposed to mitigate the above-mentioned overhead cost; however, their performance is not sufficiently reliable depending on the optimization setting. To overcome such limitations, we deviate from the existing input-space-based adversarial training regime and propose a single-step latent adversarial training method (SLAT), which leverages the gradients of latent representation as the latent adversarial perturbation. We demonstrate that the L1 norm of feature gradients is implicitly regularized through the adopted latent perturbation, thereby recovering local linearity and ensuring reliable performance, compared to the existing single-step adversarial training methods. Because latent perturbation is based on the gradients of the latent representations which can be obtained for free in the process of input gradients computation, the proposed method costs roughly the same time as the fast gradient sign method. Experiment results demonstrate that the proposed method, despite its structural simplicity, outperforms state-of-the-art accelerated adversarial training methods.

摘要: 虽然多步对抗性训练作为一种对抗强对抗性攻击的有效防御方法得到了广泛的欢迎，但与标准训练相比，其计算成本是出了名的昂贵。为了降低上述开销，已经提出了几种单步对抗性训练方法，但它们的性能并不完全可靠，这取决于优化设置。为了克服这些局限性，我们偏离了现有的基于输入空间的对抗性训练方法，提出了一种利用潜在表征梯度作为潜在对抗性扰动的单步潜在对抗性训练方法(SLAT)。与现有的单步对抗性训练方法相比，我们证明了特征梯度的L1范数通过所采用的潜在扰动被隐式正则化，从而恢复了局部线性并确保了可靠的性能。由于潜在摄动是基于输入梯度计算过程中可以免费获得的潜在表示的梯度，因此该方法的计算时间与快速梯度符号方法大致相同。实验结果表明，尽管该方法结构简单，但性能优于目前最先进的加速对抗性训练方法。



## **29. Adversarial Attacks in Cooperative AI**

协作式人工智能中的对抗性攻击 cs.LG

**SubmitDate**: 2021-11-29    [paper-pdf](http://arxiv.org/pdf/2111.14833v1)

**Authors**: Ted Fujimoto, Arthur Paul Pedersen

**Abstracts**: Single-agent reinforcement learning algorithms in a multi-agent environment are inadequate for fostering cooperation. If intelligent agents are to interact and work together to solve complex problems, methods that counter non-cooperative behavior are needed to facilitate the training of multiple agents. This is the goal of cooperative AI. Recent work in adversarial machine learning, however, shows that models (e.g., image classifiers) can be easily deceived into making incorrect decisions. In addition, some past research in cooperative AI has relied on new notions of representations, like public beliefs, to accelerate the learning of optimally cooperative behavior. Hence, cooperative AI might introduce new weaknesses not investigated in previous machine learning research. In this paper, our contributions include: (1) arguing that three algorithms inspired by human-like social intelligence introduce new vulnerabilities, unique to cooperative AI, that adversaries can exploit, and (2) an experiment showing that simple, adversarial perturbations on the agents' beliefs can negatively impact performance. This evidence points to the possibility that formal representations of social behavior are vulnerable to adversarial attacks.

摘要: 多智能体环境中的单智能体强化学习算法不能很好地促进协作。如果智能Agent要交互并共同工作来解决复杂问题，就需要针对不合作行为的方法，以便于多个Agent的训练。这是合作AI的目标。然而，最近在对抗性机器学习方面的工作表明，模型(例如，图像分类器)很容易被欺骗，从而做出不正确的决定。此外，过去对合作人工智能的一些研究依赖于新的表征概念，如公众信仰，以加速最佳合作行为的学习。因此，合作人工智能可能会引入以前的机器学习研究中没有研究的新弱点。在本文中，我们的贡献包括：(1)论证了三种受类人类社会智能启发的算法引入了新的漏洞，这些漏洞是合作人工智能所特有的，攻击者可以利用这些漏洞；(2)实验表明，对Agent信念的简单对抗性扰动可能会对性能产生负面影响。这一证据表明，社交行为的正式表述很容易受到敌意攻击。



## **30. Feature-Filter: Detecting Adversarial Examples through Filtering off Recessive Features**

功能-过滤：通过过滤隐性特征来检测敌意实例 cs.LG

**SubmitDate**: 2021-11-29    [paper-pdf](http://arxiv.org/pdf/2107.09502v2)

**Authors**: Hui Liu, Bo Zhao, Minzhi Ji, Yuefeng Peng, Jiabao Guo, Peng Liu

**Abstracts**: Deep neural networks (DNNs) are under threat from adversarial example attacks. The adversary can easily change the outputs of DNNs by adding small well-designed perturbations to inputs. Adversarial example detection is a fundamental work for robust DNNs-based service. Adversarial examples show the difference between humans and DNNs in image recognition. From a human-centric perspective, image features could be divided into dominant features that are comprehensible to humans, and recessive features that are incomprehensible to humans, yet are exploited by DNNs. In this paper, we reveal that imperceptible adversarial examples are the product of recessive features misleading neural networks, and an adversarial attack is essentially a kind of method to enrich these recessive features in the image. The imperceptibility of the adversarial examples indicates that the perturbations enrich recessive features, yet hardly affect dominant features. Therefore, adversarial examples are sensitive to filtering off recessive features, while benign examples are immune to such operation. Inspired by this idea, we propose a label-only adversarial detection approach that is referred to as feature-filter. Feature-filter utilizes discrete cosine transform to approximately separate recessive features from dominant features, and gets a mutant image that is filtered off recessive features. By only comparing DNN's prediction labels on the input and its mutant, feature-filter can real-time detect imperceptible adversarial examples at high accuracy and few false positives.

摘要: 深度神经网络(DNNs)正受到敌意示例攻击的威胁。敌手可以通过向输入添加小的精心设计的扰动来很容易地改变DNN的输出。敌意实例检测是健壮的基于DNNs的服务的一项基础性工作。对抗性的例子显示了人类和DNN在图像识别方面的差异。从以人为中心的角度来看，图像特征可以分为人类可以理解的显性特征和人类无法理解但被DNNs利用的隐性特征。本文揭示了潜伏的对抗性例子是隐性特征误导神经网络的产物，而对抗性攻击本质上是丰富图像中这些隐性特征的一种方法。对抗性例子的隐蔽性表明，扰动丰富了隐性特征，但对显性特征影响不大。因此，敌意的例子对过滤隐性特征很敏感，而良性的例子则不受这种操作的影响。受此启发，我们提出了一种基于标签的对抗性检测方法，称为特征过滤。特征-过滤利用离散余弦变换对隐性特征和显性特征进行近似分离，得到从隐性特征中滤除的突变图像。特征过滤只需要比较输入样本和变异样本上的预测标签，就可以实时检测出不易察觉的敌意样本，准确率高，误报率低。



## **31. GreedyFool: Multi-Factor Imperceptibility and Its Application to Designing a Black-box Adversarial Attack**

GreedyFool：多因素隐蔽性及其在设计黑盒对抗攻击中的应用 cs.LG

**SubmitDate**: 2021-11-29    [paper-pdf](http://arxiv.org/pdf/2010.06855v4)

**Authors**: Hui Liu, Bo Zhao, Minzhi Ji, Peng Liu

**Abstracts**: Adversarial examples are well-designed input samples, in which perturbations are imperceptible to the human eyes, but easily mislead the output of deep neural networks (DNNs). Existing works synthesize adversarial examples by leveraging simple metrics to penalize perturbations, that lack sufficient consideration of the human visual system (HVS), which produces noticeable artifacts. To explore why the perturbations are visible, this paper summarizes four primary factors affecting the perceptibility of human eyes. Based on this investigation, we design a multi-factor metric MulFactorLoss for measuring the perceptual loss between benign examples and adversarial ones. In order to test the imperceptibility of the multi-factor metric, we propose a novel black-box adversarial attack that is referred to as GreedyFool. GreedyFool applies differential evolution to evaluate the effects of perturbed pixels on the confidence of a target DNN, and introduces greedy approximation to automatically generate adversarial perturbations. We conduct extensive experiments on the ImageNet and CIFRA-10 datasets and a comprehensive user study with 60 participants. The experimental results demonstrate that MulFactorLoss is a more imperceptible metric than the existing pixelwise metrics, and GreedyFool achieves a 100% success rate in a black-box manner.

摘要: 对抗性的例子是设计良好的输入样本，其中的扰动对人眼是不可察觉的，但容易误导深度神经网络(DNNs)的输出。现有的作品通过利用简单的度量来惩罚扰动，从而合成了对抗性的例子，这些扰动缺乏对人类视觉系统(HVS)的充分考虑，而人类视觉系统(HVS)会产生明显的人工产物。为了探索为什么扰动是可见的，本文总结了影响人眼感知的四个主要因素。在此基础上，我们设计了一个多因素度量MulFactorLoss来度量良性样本和敌意样本之间的知觉损失。为了测试多因子度量的不可见性，我们提出了一种新的黑盒对抗攻击，称为GreedyFool。GreedyFool应用差分进化来评估扰动像素对目标DNN置信度的影响，并引入贪婪近似来自动生成对抗性扰动。我们在ImageNet和CIFRA-10数据集上进行了广泛的实验，并对60名参与者进行了全面的用户研究。实验结果表明，MulFactorLoss是一种比现有的像素化度量更隐蔽的度量，GreedyFool以黑盒的方式达到了100%的成功率。



## **32. Detecting Adversaries, yet Faltering to Noise? Leveraging Conditional Variational AutoEncoders for Adversary Detection in the Presence of Noisy Images**

侦测到对手，却对噪音犹豫不决？利用条件变分自动编码器在噪声图像中进行敌意检测 cs.LG

**SubmitDate**: 2021-11-28    [paper-pdf](http://arxiv.org/pdf/2111.15518v1)

**Authors**: Dvij Kalaria, Aritra Hazra, Partha Pratim Chakrabarti

**Abstracts**: With the rapid advancement and increased use of deep learning models in image identification, security becomes a major concern to their deployment in safety-critical systems. Since the accuracy and robustness of deep learning models are primarily attributed from the purity of the training samples, therefore the deep learning architectures are often susceptible to adversarial attacks. Adversarial attacks are often obtained by making subtle perturbations to normal images, which are mostly imperceptible to humans, but can seriously confuse the state-of-the-art machine learning models. What is so special in the slightest intelligent perturbations or noise additions over normal images that it leads to catastrophic classifications by the deep neural networks? Using statistical hypothesis testing, we find that Conditional Variational AutoEncoders (CVAE) are surprisingly good at detecting imperceptible image perturbations. In this paper, we show how CVAEs can be effectively used to detect adversarial attacks on image classification networks. We demonstrate our results over MNIST, CIFAR-10 dataset and show how our method gives comparable performance to the state-of-the-art methods in detecting adversaries while not getting confused with noisy images, where most of the existing methods falter.

摘要: 随着深度学习模型在图像识别中的快速发展和越来越多的使用，安全性成为它们在安全关键系统中部署的主要考虑因素。由于深度学习模型的准确性和鲁棒性主要取决于训练样本的纯度，因此深度学习结构往往容易受到敌意攻击。对抗性攻击通常是通过对正常图像进行微妙的扰动来获得的，这对人类来说大多是不可察觉的，但会严重混淆最先进的机器学习模型。在正常图像上，最轻微的智能扰动或噪声添加有什么特别之处，以至于导致深层神经网络进行灾难性的分类？利用统计假设检验，我们发现条件变分自动编码器(CVAE)在检测不可察觉的图像扰动方面表现出惊人的优势。在本文中，我们展示了如何有效地使用CVAE来检测对图像分类网络的敌意攻击。我们在MNIST，CIFAR-10数据集上演示了我们的结果，并展示了我们的方法如何在检测对手方面提供与最先进的方法相当的性能，同时又不会与大多数现有方法步履蹒跚的噪声图像混淆。



## **33. MALIGN: Adversarially Robust Malware Family Detection using Sequence Alignment**

恶意：使用序列比对检测恶意软件系列 cs.CR

**SubmitDate**: 2021-11-28    [paper-pdf](http://arxiv.org/pdf/2111.14185v1)

**Authors**: Shoumik Saha, Sadia Afroz, Atif Rahman

**Abstracts**: We propose MALIGN, a novel malware family detection approach inspired by genome sequence alignment. MALIGN encodes malware using four nucleotides and then uses genome sequence alignment approaches to create a signature of a malware family based on the code fragments conserved in the family making it robust to evasion by modification and addition of content. Moreover, unlike previous approaches based on sequence alignment, our method uses a multiple whole-genome alignment tool that protects against adversarial attacks such as code insertion, deletion or modification. Our approach outperforms state-of-the-art machine learning based malware detectors and demonstrates robustness against trivial adversarial attacks. MALIGN also helps identify the techniques malware authors use to evade detection.

摘要: 我们提出了一种受基因组序列比对启发的新型恶意软件家族检测方法MARIGN。Malign使用四个核苷酸对恶意软件进行编码，然后使用基因组序列比对方法根据家族中保存的代码片段创建恶意软件家族的签名，从而使其对通过修改和添加内容进行规避具有很强的鲁棒性。此外，与以前基于序列比对的方法不同，我们的方法使用了多个全基因组比对工具，该工具可以防止代码插入、删除或修改等敌意攻击。我们的方法比最先进的基于机器学习的恶意软件检测器性能更好，并且对琐碎的对手攻击表现出健壮性。恶意还有助于识别恶意软件作者用来逃避检测的技术。



## **34. Statically Detecting Adversarial Malware through Randomised Chaining**

通过随机链静态检测敌意恶意软件 cs.CR

**SubmitDate**: 2021-11-28    [paper-pdf](http://arxiv.org/pdf/2111.14037v1)

**Authors**: Matthew Crawford, Wei Wang, Ruoxi Sun, Minhui Xue

**Abstracts**: With the rapid growth of malware attacks, more antivirus developers consider deploying machine learning technologies into their productions. Researchers and developers published various machine learning-based detectors with high precision on malware detection in recent years. Although numerous machine learning-based malware detectors are available, they face various machine learning-targeted attacks, including evasion and adversarial attacks. This project explores how and why adversarial examples evade malware detectors, then proposes a randomised chaining method to defend against adversarial malware statically. This research is crucial for working towards combating the pertinent malware cybercrime.

摘要: 随着恶意软件攻击的快速增长，越来越多的反病毒开发人员考虑将机器学习技术部署到他们的产品中。近年来，研究人员和开发人员发布了各种基于机器学习的恶意软件检测高精度检测器。虽然有许多基于机器学习的恶意软件检测器可用，但它们面临着各种机器学习目标攻击，包括逃避和敌意攻击。该项目探讨了敌意实例如何以及为什么躲避恶意软件检测器，然后提出了一种随机链接的方法来静态防御敌意恶意软件。这项研究对于打击相关的恶意软件网络犯罪至关重要。



## **35. Increasing-Margin Adversarial (IMA) Training to Improve Adversarial Robustness of Neural Networks**

提高神经网络对抗鲁棒性的增量对抗性(IMA)训练 cs.CV

11 pages, 8 figures, 10 tables

**SubmitDate**: 2021-11-27    [paper-pdf](http://arxiv.org/pdf/2005.09147v5)

**Authors**: Linhai Ma, Liang Liang

**Abstracts**: Convolutional neural network (CNN) has surpassed traditional methods for medical image classification. However, CNN is vulnerable to adversarial attacks which may lead to disastrous consequences in medical applications. Although adversarial noises are usually generated by attack algorithms, white-noise-induced adversarial samples can exist, and therefore the threats are real. In this study, we propose a novel training method, named IMA, to improve the robust-ness of CNN against adversarial noises. During training, the IMA method increases the margins of training samples in the input space, i.e., moving CNN decision boundaries far away from the training samples to improve robustness. The IMA method is evaluated on publicly available datasets under strong 100-PGD white-box adversarial attacks, and the results show that the proposed method significantly improved CNN classification and segmentation accuracy on noisy data while keeping a high accuracy on clean data. We hope our approach may facilitate the development of robust applications in medical field.

摘要: 卷积神经网络(CNN)已经超越了传统的医学图像分类方法。然而，CNN很容易受到对抗性攻击，这可能会导致医疗应用中的灾难性后果。虽然攻击算法通常会产生对抗性噪声，但白噪声诱导的对抗性样本可能存在，因此威胁是真实存在的。在这项研究中，我们提出了一种新的训练方法，称为IMA，以提高CNN对对抗性噪声的鲁棒性。在训练过程中，IMA方法增加了输入空间中训练样本的边际，即使CNN决策边界远离训练样本，以提高鲁棒性。在100-PGD强白盒攻击下，在公开数据集上对IMA方法进行了评估，结果表明，该方法在保持对干净数据较高精度的同时，显著提高了对含噪声数据的CNN分类和分割的准确率。我们希望我们的方法可以促进医学领域健壮应用的发展。



## **36. Adaptive Image Transformations for Transfer-based Adversarial Attack**

基于传输的对抗性攻击的自适应图像变换 cs.CV

20 pages, 6 figures, 8 tables

**SubmitDate**: 2021-11-27    [paper-pdf](http://arxiv.org/pdf/2111.13844v1)

**Authors**: Zheng Yuan, Jie Zhang, Shiguang Shan

**Abstracts**: Adversarial attacks provide a good way to study the robustness of deep learning models. One category of methods in transfer-based black-box attack utilizes several image transformation operations to improve the transferability of adversarial examples, which is effective, but fails to take the specific characteristic of the input image into consideration. In this work, we propose a novel architecture, called Adaptive Image Transformation Learner (AITL), which incorporates different image transformation operations into a unified framework to further improve the transferability of adversarial examples. Unlike the fixed combinational transformations used in existing works, our elaborately designed transformation learner adaptively selects the most effective combination of image transformations specific to the input image. Extensive experiments on ImageNet demonstrate that our method significantly improves the attack success rates on both normally trained models and defense models under various settings.

摘要: 对抗性攻击为深入学习模型的鲁棒性研究提供了一种很好的途径。基于传输的黑盒攻击方法中有一类是利用多种图像变换操作来提高对抗性实例的可传递性，这种方法是有效的，但没有考虑到输入图像的具体特性。在这项工作中，我们提出了一种新的体系结构，称为自适应图像变换学习器(AITL)，它将不同的图像变换操作合并到一个统一的框架中，以进一步提高对抗性示例的可移植性。与现有工作中使用的固定组合变换不同，我们精心设计的变换学习器能够自适应地选择针对输入图像的最有效的图像变换组合。在ImageNet上的大量实验表明，在不同的设置下，我们的方法都能显著提高正常训练模型和防御模型的攻击成功率。



## **37. Adaptive Perturbation for Adversarial Attack**

对抗性攻击的自适应摄动 cs.CV

11 pages, 3 figures, 8 tables

**SubmitDate**: 2021-11-27    [paper-pdf](http://arxiv.org/pdf/2111.13841v1)

**Authors**: Zheng Yuan, Jie Zhang, Shiguang Shan

**Abstracts**: In recent years, the security of deep learning models achieves more and more attentions with the rapid development of neural networks, which are vulnerable to adversarial examples. Almost all existing gradient-based attack methods use the sign function in the generation to meet the requirement of perturbation budget on $L_\infty$ norm. However, we find that the sign function may be improper for generating adversarial examples since it modifies the exact gradient direction. We propose to remove the sign function and directly utilize the exact gradient direction with a scaling factor for generating adversarial perturbations, which improves the attack success rates of adversarial examples even with fewer perturbations. Moreover, considering that the best scaling factor varies across different images, we propose an adaptive scaling factor generator to seek an appropriate scaling factor for each image, which avoids the computational cost for manually searching the scaling factor. Our method can be integrated with almost all existing gradient-based attack methods to further improve the attack success rates. Extensive experiments on the CIFAR10 and ImageNet datasets show that our method exhibits higher transferability and outperforms the state-of-the-art methods.

摘要: 近年来，随着神经网络的快速发展，深度学习模型的安全性受到越来越多的关注，因为神经网络容易受到敌意例子的攻击。现有的基于梯度的攻击方法几乎都是在生成过程中使用符号函数，以满足$L\infty$范数上的扰动预算要求。然而，我们发现符号函数可能不适合生成对抗性示例，因为它修改了精确的梯度方向。我们建议去掉符号函数，直接利用带有比例因子的精确梯度方向来产生对抗性扰动，从而在扰动较少的情况下提高了对抗性示例的攻击成功率。此外，考虑到不同图像的最佳比例因子是不同的，我们提出了一种自适应比例因子生成器来为每幅图像寻找合适的比例因子，从而避免了人工搜索比例因子的计算代价。该方法可以与几乎所有现有的基于梯度的攻击方法相结合，进一步提高攻击成功率。在CIFAR10和ImageNet数据集上的大量实验表明，我们的方法表现出更高的可移植性，并且性能优于最先进的方法。



## **38. Benchmarking Shadow Removal for Facial Landmark Detection and Beyond**

人脸标志点检测的基准阴影去除及进一步研究 cs.CV

**SubmitDate**: 2021-11-27    [paper-pdf](http://arxiv.org/pdf/2111.13790v1)

**Authors**: Lan Fu, Qing Guo, Felix Juefei-Xu, Hongkai Yu, Wei Feng, Yang Liu, Song Wang

**Abstracts**: Facial landmark detection is a very fundamental and significant vision task with many important applications. In practice, facial landmark detection can be affected by a lot of natural degradations. One of the most common and important degradations is the shadow caused by light source blocking. While many advanced shadow removal methods have been proposed to recover the image quality in recent years, their effects to facial landmark detection are not well studied. For example, it remains unclear whether shadow removal could enhance the robustness of facial landmark detection to diverse shadow patterns or not. In this work, for the first attempt, we construct a novel benchmark to link two independent but related tasks (i.e., shadow removal and facial landmark detection). In particular, the proposed benchmark covers diverse face shadows with different intensities, sizes, shapes, and locations. Moreover, to mine hard shadow patterns against facial landmark detection, we propose a novel method (i.e., adversarial shadow attack), which allows us to construct a challenging subset of the benchmark for a comprehensive analysis. With the constructed benchmark, we conduct extensive analysis on three state-of-the-art shadow removal methods and three landmark detectors. The observation of this work motivates us to design a novel detection-aware shadow removal framework, which empowers shadow removal to achieve higher restoration quality and enhance the shadow robustness of deployed facial landmark detectors.

摘要: 人脸标志点检测是一项非常基础和重要的视觉任务，有着许多重要的应用。在实际应用中，人脸地标检测会受到许多自然退化的影响。光源遮挡造成的阴影是最常见也是最重要的退化之一。虽然近年来提出了许多先进的阴影去除方法来恢复图像质量，但它们对人脸标志点检测的影响还没有得到很好的研究。例如，阴影去除是否可以增强人脸标志点检测对不同阴影模式的鲁棒性，目前还不清楚。在这项工作中，作为第一次尝试，我们构建了一个新的基准来连接两个独立但又相关的任务(即阴影去除和人脸地标检测)。特别是，建议的基准涵盖了不同强度、大小、形状和位置的不同脸部阴影。此外，为了针对人脸标志点检测挖掘硬阴影模式，我们提出了一种新的方法(即对抗性阴影攻击)，它允许我们构造一个具有挑战性的基准子集来进行全面的分析。在构建的基准测试中，我们对三种最新的阴影去除方法和三种标志点检测器进行了广泛的分析。这项工作的观察促使我们设计了一种新颖的检测感知阴影去除框架，使得阴影去除能够达到更高的恢复质量，并增强了部署的人脸地标检测器的阴影鲁棒性。



## **39. Resilient Nash Equilibrium Seeking in the Partial Information Setting**

部分信息环境下弹性纳什均衡搜索 math.OC

**SubmitDate**: 2021-11-26    [paper-pdf](http://arxiv.org/pdf/2111.13735v1)

**Authors**: Dian Gadjov, Lacra Pavel

**Abstracts**: Current research in distributed Nash equilibrium (NE) seeking in the partial information setting assumes that information is exchanged between agents that are "truthful". However, in general noncooperative games agents may consider sending misinformation to neighboring agents with the goal of further reducing their cost. Additionally, communication networks are vulnerable to attacks from agents outside the game as well as communication failures. In this paper, we propose a distributed NE seeking algorithm that is robust against adversarial agents that transmit noise, random signals, constant singles, deceitful messages, as well as being resilient to external factors such as dropped communication, jammed signals, and man in the middle attacks. The core issue that makes the problem challenging is that agents have no means of verifying if the information they receive is correct, i.e. there is no "ground truth". To address this problem, we use an observation graph, that gives truthful action information, in conjunction with a communication graph, that gives (potentially incorrect) information. By filtering information obtained from these two graphs, we show that our algorithm is resilient against adversarial agents and converges to the Nash equilibrium.

摘要: 目前在部分信息环境下寻求分布式纳什均衡(NE)的研究假设信息在“真实”的Agent之间交换。然而，通常情况下，不合作的游戏代理可能会考虑向邻近的代理发送错误信息，目标是进一步降低他们的成本。此外，通信网络容易受到来自游戏外代理的攻击以及通信故障。本文提出了一种分布式网元搜索算法，该算法对传输噪声、随机信号、恒定信号、欺骗性消息等敌意代理具有较强的鲁棒性，并对通信中断、信号阻塞、中间人攻击等外部因素具有较强的适应性。使问题具有挑战性的核心问题是，代理无法验证他们收到的信息是否正确，即没有“地面真相”。为了解决这个问题，我们使用观察图(提供真实的操作信息)和通信图(提供(可能不正确的)信息)。通过对从这两个图中获得的信息进行过滤，我们证明了我们的算法对敌意代理是有弹性的，并且收敛到纳什均衡。



## **40. The Geometry of Adversarial Training in Binary Classification**

二元分类中对抗性训练的几何问题 cs.LG

**SubmitDate**: 2021-11-26    [paper-pdf](http://arxiv.org/pdf/2111.13613v1)

**Authors**: Leon Bungert, Nicolás García Trillos, Ryan Murray

**Abstracts**: We establish an equivalence between a family of adversarial training problems for non-parametric binary classification and a family of regularized risk minimization problems where the regularizer is a nonlocal perimeter functional. The resulting regularized risk minimization problems admit exact convex relaxations of the type $L^1+$ (nonlocal) $\operatorname{TV}$, a form frequently studied in image analysis and graph-based learning. A rich geometric structure is revealed by this reformulation which in turn allows us to establish a series of properties of optimal solutions of the original problem, including the existence of minimal and maximal solutions (interpreted in a suitable sense), and the existence of regular solutions (also interpreted in a suitable sense). In addition, we highlight how the connection between adversarial training and perimeter minimization problems provides a novel, directly interpretable, statistical motivation for a family of regularized risk minimization problems involving perimeter/total variation. The majority of our theoretical results are independent of the distance used to define adversarial attacks.

摘要: 我们建立了一类非参数二元分类的对抗性训练问题和一类正则化风险最小化问题之间的等价性，其中正则化子是非局部周长泛函。由此产生的正则化风险最小化问题允许类型为$L^1+$(非局部)$\操作符名称{TV}$的精确凸松弛，这是图像分析和基于图的学习中经常研究的一种形式。这种改写揭示了一个丰富的几何结构，进而允许我们建立原问题最优解的一系列性质，包括最小解和最大解(在适当意义下解释)的存在性，以及正则解(也在适当意义上解释)的存在性。此外，我们强调了对抗性训练和周长最小化问题之间的联系如何为一类涉及周长/总变异的正则化风险最小化问题提供了一种新颖的、直接可解释的统计动机。我们的大部分理论结果与用于定义对抗性攻击的距离无关。



## **41. Explainability-Aware One Point Attack for Point Cloud Neural Networks**

基于可解释性的点云神经网络单点攻击 cs.CV

**SubmitDate**: 2021-11-26    [paper-pdf](http://arxiv.org/pdf/2110.04158v2)

**Authors**: Hanxiao Tan, Helena Kotthaus

**Abstracts**: With the proposition of neural networks for point clouds, deep learning has started to shine in the field of 3D object recognition while researchers have shown an increased interest to investigate the reliability of point cloud networks by adversarial attacks. However, most of the existing studies aim to deceive humans or defense algorithms, while the few that address the operation principles of the models themselves remain flawed in terms of critical point selection. In this work, we propose two adversarial methods: One Point Attack (OPA) and Critical Traversal Attack (CTA), which incorporate the explainability technologies and aim to explore the intrinsic operating principle of point cloud networks and their sensitivity against critical points perturbations. Our results show that popular point cloud networks can be deceived with almost $100\%$ success rate by shifting only one point from the input instance. In addition, we show the interesting impact of different point attribution distributions on the adversarial robustness of point cloud networks. Finally, we discuss how our approaches facilitate the explainability study for point cloud networks. To the best of our knowledge, this is the first point-cloud-based adversarial approach concerning explainability. Our code is available at https://github.com/Explain3D/Exp-One-Point-Atk-PC.

摘要: 随着神经网络在点云领域的提出，深度学习开始在三维物体识别领域闪耀光芒，而研究人员对利用对抗性攻击来研究点云网络的可靠性也表现出越来越大的兴趣。然而，现有的大多数研究都是为了欺骗人类或防御算法，而少数针对模型本身操作原理的研究在临界点选择方面仍然存在缺陷。在这项工作中，我们提出了两种对抗方法：单点攻击(OPA)和临界遍历攻击(CTA)，它们融合了可解释性技术，旨在探索点云网络的内在工作原理及其对临界点扰动的敏感性。我们的结果表明，只要从输入实例中移动一个点，流行的点云网络就可以被欺骗，成功率接近100美元。此外，我们还展示了不同的点属性分布对点云网络对抗健壮性的影响。最后，我们讨论了我们的方法如何促进点云网络的可解释性研究。据我们所知，这是第一种基于点云的对抗性解释方法。我们的代码可在https://github.com/Explain3D/Exp-One-Point-Atk-PC.获得



## **42. Privacy-Preserving Synthetic Smart Meters Data**

保护隐私的合成智能电表数据 eess.SP

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2012.04475v2)

**Authors**: Ganesh Del Grosso, Georg Pichler, Pablo Piantanida

**Abstracts**: Power consumption data is very useful as it allows to optimize power grids, detect anomalies and prevent failures, on top of being useful for diverse research purposes. However, the use of power consumption data raises significant privacy concerns, as this data usually belongs to clients of a power company. As a solution, we propose a method to generate synthetic power consumption samples that faithfully imitate the originals, but are detached from the clients and their identities. Our method is based on Generative Adversarial Networks (GANs). Our contribution is twofold. First, we focus on the quality of the generated data, which is not a trivial task as no standard evaluation methods are available. Then, we study the privacy guarantees provided to members of the training set of our neural network. As a minimum requirement for privacy, we demand our neural network to be robust to membership inference attacks, as these provide a gateway for further attacks in addition to presenting a privacy threat on their own. We find that there is a compromise to be made between the privacy and the performance provided by the algorithm.

摘要: 电力消耗数据非常有用，因为它可以优化电网，检测异常和防止故障，此外还可以用于不同的研究目的。然而，用电数据的使用引起了很大的隐私问题，因为这些数据通常属于电力公司的客户。作为解决方案，我们提出了一种生成合成功耗样本的方法，该方法忠实地模仿原始功耗样本，但与客户及其身份分离。我们的方法是基于产生式对抗性网络(GANS)的。我们的贡献是双重的。首先，我们将重点放在生成数据的质量上，这不是一项微不足道的任务，因为没有标准的评估方法可用。然后，我们研究了提供给神经网络训练集成员的隐私保证。作为隐私的最低要求，我们要求我们的神经网络对成员关系推断攻击具有健壮性，因为这些攻击除了本身构成隐私威胁之外，还为进一步的攻击提供了一个网关。我们发现需要在隐私和算法提供的性能之间进行折衷。



## **43. Real-Time Privacy-Preserving Data Release for Smart Meters**

智能电表的实时隐私保护数据发布 eess.SP

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/1906.06427v4)

**Authors**: Mohammadhadi Shateri, Francisco Messina, Pablo Piantanida, Fabrice Labeau

**Abstracts**: Smart Meters (SMs) are able to share the power consumption of users with utility providers almost in real-time. These fine-grained signals carry sensitive information about users, which has raised serious concerns from the privacy viewpoint. In this paper, we focus on real-time privacy threats, i.e., potential attackers that try to infer sensitive information from SMs data in an online fashion. We adopt an information-theoretic privacy measure and show that it effectively limits the performance of any attacker. Then, we propose a general formulation to design a privatization mechanism that can provide a target level of privacy by adding a minimal amount of distortion to the SMs measurements. On the other hand, to cope with different applications, a flexible distortion measure is considered. This formulation leads to a general loss function, which is optimized using a deep learning adversarial framework, where two neural networks -- referred to as the releaser and the adversary -- are trained with opposite goals. An exhaustive empirical study is then performed to validate the performance of the proposed approach and compare it with state-of-the-art methods for the occupancy detection privacy problem. Finally, we also investigate the impact of data mismatch between the releaser and the attacker.

摘要: 智能电表(SMS)能够几乎实时地与公用事业提供商共享用户的电能消耗。这些细粒度的信号携带着用户的敏感信息，从隐私的角度来看，这引起了严重的担忧。在本文中，我们关注的是实时隐私威胁，即试图以在线方式从短信数据中推断敏感信息的潜在攻击者。我们采用了一种信息论的隐私措施，并表明它有效地限制了任何攻击者的表现。然后，我们提出了一个通用的公式来设计一种私有化机制，该机制可以通过对短信测量添加最小的失真来提供目标级别的隐私。另一方面，为了适应不同的应用，考虑了一种灵活的失真度量。这一公式导致了一般的损失函数，该函数使用深度学习对手框架进行优化，其中两个神经网络--称为释放者和对手--被训练成具有相反的目标。在此基础上，进行了详尽的实证研究，验证了该方法的性能，并将其与现有的占有率检测隐私问题的方法进行了比较。最后，我们还调查了发布者和攻击者之间数据不匹配的影响。



## **44. Simple Post-Training Robustness Using Test Time Augmentations and Random Forest**

使用测试时间增加和随机森林的简单训练后鲁棒性 cs.CV

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2109.08191v2)

**Authors**: Gilad Cohen, Raja Giryes

**Abstracts**: Although Deep Neural Networks (DNNs) achieve excellent performance on many real-world tasks, they are highly vulnerable to adversarial attacks. A leading defense against such attacks is adversarial training, a technique in which a DNN is trained to be robust to adversarial attacks by introducing adversarial noise to its input. This procedure is effective but must be done during the training phase. In this work, we propose Augmented Random Forest (ARF), a simple and easy-to-use strategy for robustifying an existing pretrained DNN without modifying its weights. For every image, we generate randomized test time augmentations by applying diverse color, blur, noise, and geometric transforms. Then we use the DNN's logits output to train a simple random forest to predict the real class label. Our method achieves state-of-the-art adversarial robustness on a diversity of white and black box attacks with minimal compromise on the natural images' classification. We test ARF also against numerous adaptive white-box attacks and it shows excellent results when combined with adversarial training. Code is available at https://github.com/giladcohen/ARF.

摘要: 尽管深度神经网络(DNNs)在许多现实世界的任务中取得了优异的性能，但它们非常容易受到对手的攻击。对抗这种攻击的主要防御是对抗性训练，这是一种通过在输入中引入对抗性噪声来训练DNN对对抗性攻击具有健壮性的技术。此程序是有效的，但必须在培训阶段进行。在这项工作中，我们提出了增广随机森林(ARF)，这是一种简单易用的策略，可以在不修改权值的情况下对现有的预先训练的DNN进行鲁棒性处理。对于每个图像，我们通过应用不同的颜色、模糊、噪声和几何变换来生成随机的测试时间增量。然后利用DNN的Logits输出训练一个简单的随机森林来预测真实的类标签。我们的方法在对自然图像分类影响最小的情况下，实现了对各种白盒和黑盒攻击的最好的对抗鲁棒性。我们还测试了ARF对许多自适应白盒攻击的攻击，当与对抗性训练相结合时，它显示出极好的结果。代码可在https://github.com/giladcohen/ARF.上获得



## **45. Gradient Inversion Attack: Leaking Private Labels in Two-Party Split Learning**

梯度反转攻击：两方分裂学习中的私有标签泄漏 cs.CR

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2112.01299v1)

**Authors**: Sanjay Kariyappa, Moinuddin K Qureshi

**Abstracts**: Split learning is a popular technique used to perform vertical federated learning, where the goal is to jointly train a model on the private input and label data held by two parties. To preserve privacy of the input and label data, this technique uses a split model and only requires the exchange of intermediate representations (IR) of the inputs and gradients of the IR between the two parties during the learning process. In this paper, we propose Gradient Inversion Attack (GIA), a label leakage attack that allows an adversarial input owner to learn the label owner's private labels by exploiting the gradient information obtained during split learning. GIA frames the label leakage attack as a supervised learning problem by developing a novel loss function using certain key properties of the dataset and models. Our attack can uncover the private label data on several multi-class image classification problems and a binary conversion prediction task with near-perfect accuracy (97.01% - 99.96%), demonstrating that split learning provides negligible privacy benefits to the label owner. Furthermore, we evaluate the use of gradient noise to defend against GIA. While this technique is effective for simpler datasets, it significantly degrades utility for datasets with higher input dimensionality. Our findings underscore the need for better privacy-preserving training techniques for vertically split data.

摘要: 分裂学习是一种用于执行垂直联合学习的流行技术，其目标是联合训练一个关于双方持有的私有输入和标签数据的模型。为了保护输入和标签数据的私密性，该技术使用分割模型，并且在学习过程中只需要在双方之间交换输入的IR和IR的梯度的中间表示(IR)。本文提出了梯度反转攻击(GIA)，这是一种标签泄漏攻击，允许敌意输入所有者利用分裂学习过程中获得的梯度信息来学习标签所有者的私有标签。GIA通过利用数据集和模型的某些关键属性开发了一种新的损失函数，将标签泄漏攻击框架化为一个有监督学习问题。我们的攻击能够以近乎完美的准确率(97.01%-99.96%)发现多类图像分类问题和二进制转换预测任务上的私有标签数据，表明分裂学习可以为标签所有者提供可以忽略不计的隐私收益。此外，我们评估了使用梯度噪声来防御GIA。虽然该技术对于较简单的数据集是有效的，但它显著降低了具有较高输入维数的数据集的效用。我们的发现强调了垂直分割数据需要更好的隐私保护训练技术。



## **46. EAD: an ensemble approach to detect adversarial examples from the hidden features of deep neural networks**

EAD：一种从深层神经网络隐含特征中检测敌意实例的集成方法 cs.CV

Corrected Figure 4

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2111.12631v2)

**Authors**: Francesco Craighero, Fabrizio Angaroni, Fabio Stella, Chiara Damiani, Marco Antoniotti, Alex Graudenzi

**Abstracts**: One of the key challenges in Deep Learning is the definition of effective strategies for the detection of adversarial examples. To this end, we propose a novel approach named Ensemble Adversarial Detector (EAD) for the identification of adversarial examples, in a standard multiclass classification scenario. EAD combines multiple detectors that exploit distinct properties of the input instances in the internal representation of a pre-trained Deep Neural Network (DNN). Specifically, EAD integrates the state-of-the-art detectors based on Mahalanobis distance and on Local Intrinsic Dimensionality (LID) with a newly introduced method based on One-class Support Vector Machines (OSVMs). Although all constituting methods assume that the greater the distance of a test instance from the set of correctly classified training instances, the higher its probability to be an adversarial example, they differ in the way such distance is computed. In order to exploit the effectiveness of the different methods in capturing distinct properties of data distributions and, accordingly, efficiently tackle the trade-off between generalization and overfitting, EAD employs detector-specific distance scores as features of a logistic regression classifier, after independent hyperparameters optimization. We evaluated the EAD approach on distinct datasets (CIFAR-10, CIFAR-100 and SVHN) and models (ResNet and DenseNet) and with regard to four adversarial attacks (FGSM, BIM, DeepFool and CW), also by comparing with competing approaches. Overall, we show that EAD achieves the best AUROC and AUPR in the large majority of the settings and comparable performance in the others. The improvement over the state-of-the-art, and the possibility to easily extend EAD to include any arbitrary set of detectors, pave the way to a widespread adoption of ensemble approaches in the broad field of adversarial example detection.

摘要: 深度学习的关键挑战之一是定义有效的策略来检测敌意示例。为此，我们提出了一种新的方法，称为集成对抗性检测器(EAD)，用于在标准的多类分类场景中识别对抗性实例。EAD结合了多个检测器，这些检测器利用预先训练的深度神经网络(DNN)的内部表示中的输入实例的不同属性。具体地说，EAD将基于马氏距离和基于局部本征维数(LID)的最新检测器与一种新的基于单类支持向量机(OSVMs)的方法相结合。尽管所有的构成方法都假设测试实例与正确分类的训练实例集合的距离越大，其成为对抗性示例的概率就越高，但是它们在计算该距离的方式上是不同的。为了利用不同方法在捕捉数据分布的不同特性方面的有效性，从而有效地撞击泛化和过拟合之间的权衡，经过独立的超参数优化，EAD使用特定于检测器的距离得分作为Logistic回归分类器的特征。我们在不同的数据集(CIFAR-10、CIFAR-100和SVHN)和模型(ResNet和DenseNet)以及四种对手攻击(FGSM、BIM、DeepFool和CW)上对EAD方法进行了评估，并与其他方法进行了比较。总体而言，我们显示EAD在绝大多数设置中实现了最好的AUROC和AUPR，而在其他设置中实现了相当的性能。对现有技术的改进，以及轻松扩展EAD以包括任意一组检测器的可能性，为在广泛的对抗性示例检测领域广泛采用集成方法铺平了道路。



## **47. Can You Spot the Chameleon? Adversarially Camouflaging Images from Co-Salient Object Detection**

你能认出变色龙吗？协同显著目标检测中的对抗性伪装图像 cs.CV

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2009.09258v3)

**Authors**: Ruijun Gao, Qing Guo, Felix Juefei-Xu, Hongkai Yu, Huazhu Fu, Wei Feng, Yang Liu, Song Wang

**Abstracts**: Co-salient object detection (CoSOD) has recently achieved significant progress and played a key role in retrieval-related tasks. However, it inevitably poses an entirely new safety and security issue, \ie, highly personal and sensitive content can potentially be extracting by powerful CoSOD methods. In this paper, we address this problem from the perspective of adversarial attacks and identify a novel task: adversarial co-saliency attack. Specially, given an image selected from a group of images containing some common and salient objects, we aim to generate an adversarial version that can mislead CoSOD methods to predict incorrect co-salient regions. Note that, compared with general white-box adversarial attacks for classification, this new task faces two additional challenges: (1) low success rate due to the diverse appearance of images in the group; (2) low transferability across CoSOD methods due to the considerable difference between CoSOD pipelines. To address these challenges, we propose the very first black-box joint adversarial exposure and noise attack (Jadena), where we jointly and locally tune the exposure and additive perturbations of the image according to a newly designed high-feature-level contrast-sensitive loss function. Our method, without any information on the state-of-the-art CoSOD methods, leads to significant performance degradation on various co-saliency detection datasets and makes the co-salient objects undetectable. This can have strong practical benefits in properly securing the large number of personal photos currently shared on the internet. Moreover, our method is potential to be utilized as a metric for evaluating the robustness of CoSOD methods.

摘要: 共显著对象检测(CoSOD)近年来取得了重大进展，在检索相关任务中发挥了关键作用。然而，这不可避免地带来了一个全新的安全问题，即可以通过强大的CoSOD方法潜在地提取高度个人化和敏感的内容。在本文中，我们从对抗性攻击的角度来研究这一问题，并提出了一种新的任务：对抗性共显攻击。特别地，给定一幅从一组包含一些常见和显著对象的图像中选择的图像，我们的目标是生成一个可能误导CoSOD方法预测错误共显著区域的对抗性版本。值得注意的是，与一般的白盒对抗性分类攻击相比，这个新任务面临着两个额外的挑战：(1)由于组内图像的多样性，成功率较低；(2)由于CoSOD管道之间的巨大差异，CoSOD方法之间的可移植性较低。为了应对这些挑战，我们提出了第一个黑盒联合对抗性曝光和噪声攻击(JADENA)，其中我们根据新设计的高特征级对比度敏感损失函数来联合和局部地调整图像的曝光和加性扰动。我们的方法没有任何关于最新CoSOD方法的信息，导致在各种共显性检测数据集上的性能显著下降，并且使得共显性对象不可检测。这对妥善保护目前在互联网上共享的大量个人照片有很大的实际好处。此外，我们的方法有可能被用作评估CoSOD方法稳健性的一个度量。



## **48. AdvBokeh: Learning to Adversarially Defocus Blur**

AdvBokeh：学会对抗性散焦模糊 cs.CV

13 pages

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2111.12971v1)

**Authors**: Yihao Huang, Felix Juefei-Xu, Qing Guo, Weikai Miao, Yang Liu, Geguang Pu

**Abstracts**: Bokeh effect is a natural shallow depth-of-field phenomenon that blurs the out-of-focus part in photography. In pursuit of aesthetically pleasing photos, people usually regard the bokeh effect as an indispensable part of the photo. Due to its natural advantage and universality, as well as the fact that many visual recognition tasks can already be negatively affected by the `natural bokeh' phenomenon, in this work, we systematically study the bokeh effect from a new angle, i.e., adversarial bokeh attack (AdvBokeh) that aims to embed calculated deceptive information into the bokeh generation and produce a natural adversarial example without any human-noticeable noise artifacts. To this end, we first propose a Depth-guided Bokeh Synthesis Network (DebsNet) that is able to flexibly synthesis, refocus, and adjust the level of bokeh of the image, with a one-stage training procedure. The DebsNet allows us to tap into the bokeh generation process and attack the depth map that is needed for generating realistic bokeh (i.e., adversarially tuning the depth map) based on subsequent visual tasks. To further improve the realisticity of the adversarial bokeh, we propose depth-guided gradient-based attack to regularize the gradient.We validate the proposed method on a popular adversarial image classification dataset, i.e., NeurIPS-2017 DEV, and show that the proposed method can penetrate four state-of-the-art (SOTA) image classification networks i.e., ResNet50, VGG, DenseNet, and MobileNetV2 with a high success rate as well as high image quality. The adversarial examples obtained by AdvBokeh also exhibit high level of transferability under black-box settings. Moreover, the adversarially generated defocus blur images from the AdvBokeh can actually be capitalized to enhance the performance of SOTA defocus deblurring system, i.e., IFAN.

摘要: 波克效应是一种自然的浅景深现象，它模糊了摄影中的失焦部分。在美观照片的追打中，人们通常将波克效应视为照片中不可或缺的一部分。由于其天然的优势和广泛性，以及许多视觉识别任务已经受到“自然波克”现象的负面影响，本文从一个新的角度系统地研究了波克效应，即对抗性波克攻击(AdvBokeh)，其目的是将计算出的欺骗性信息嵌入到波克生成中，并在没有任何人类可察觉的噪声伪影的情况下产生一个自然的对抗性例子。在这项工作中，我们从一个新的角度对波克效应进行了系统的研究，即对抗性波克攻击(AdvBokeh Attack，AdvBokeh)。为此，我们首先提出了一种深度引导的Bokeh合成网络(DebsNet)，它能够灵活地合成、重新聚焦和调整图像的Bokeh级别，并且只需一步训练过程。DebsNet允许我们利用bokeh生成过程，并基于后续视觉任务攻击生成真实bokeh所需的深度图(即相反地调整深度图)。为了进一步提高对抗性图像分类的真实性，提出了基于深度引导的梯度正则化攻击方法，并在一个流行的对抗性图像分类数据集NeurIPS-2017 DEV上进行了验证，结果表明该方法能够穿透ResNet50、VGG、DenseNet和MobileNetV2四种最新的图像分类网络，具有较高的分类成功率和图像质量。AdvBokeh获得的对抗性例子在黑盒设置下也表现出很高的可转移性。此外，来自AdvBokeh的相反产生的散焦模糊图像实际上可以被资本化以增强SOTA散焦模糊系统(即IFAN)的性能。



## **49. Normal vs. Adversarial: Salience-based Analysis of Adversarial Samples for Relation Extraction**

正常VS对抗性：基于显著性的对抗性样本关系提取分析 cs.CL

IJCKG 2021

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2104.00312v4)

**Authors**: Luoqiu Li, Xiang Chen, Zhen Bi, Xin Xie, Shumin Deng, Ningyu Zhang, Chuanqi Tan, Mosha Chen, Huajun Chen

**Abstracts**: Recent neural-based relation extraction approaches, though achieving promising improvement on benchmark datasets, have reported their vulnerability towards adversarial attacks. Thus far, efforts mostly focused on generating adversarial samples or defending adversarial attacks, but little is known about the difference between normal and adversarial samples. In this work, we take the first step to leverage the salience-based method to analyze those adversarial samples. We observe that salience tokens have a direct correlation with adversarial perturbations. We further find the adversarial perturbations are either those tokens not existing in the training set or superficial cues associated with relation labels. To some extent, our approach unveils the characters against adversarial samples. We release an open-source testbed, "DiagnoseAdv" in https://github.com/zjunlp/DiagnoseAdv.

摘要: 最近的基于神经的关系提取方法，虽然在基准数据集上取得了有希望的改进，但已经报告了它们对对手攻击的脆弱性。到目前为止，努力主要集中在生成对抗性样本或防御对抗性攻击，但对正常样本和对抗性样本之间的区别知之甚少。在这项工作中，我们迈出了第一步，利用基于显著性的方法来分析这些敌意样本。我们观察到显著标记与对抗性扰动有直接关系。我们进一步发现，对抗性扰动要么是那些不存在于训练集中的标记，要么是与关系标签相关联的表面线索。在某种程度上，我们的方法针对对手样本揭开了人物的面纱。我们在https://github.com/zjunlp/DiagnoseAdv.中发布了一个开源的测试平台“DiagnoseAdv



## **50. Towards Practical Deployment-Stage Backdoor Attack on Deep Neural Networks**

走向实用化部署--深度神经网络的阶段后门攻击 cs.CR

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2111.12965v1)

**Authors**: Xiangyu Qi, Tinghao Xie, Ruizhe Pan, Jifeng Zhu, Yong Yang, Kai Bu

**Abstracts**: One major goal of the AI security community is to securely and reliably produce and deploy deep learning models for real-world applications. To this end, data poisoning based backdoor attacks on deep neural networks (DNNs) in the production stage (or training stage) and corresponding defenses are extensively explored in recent years. Ironically, backdoor attacks in the deployment stage, which can often happen in unprofessional users' devices and are thus arguably far more threatening in real-world scenarios, draw much less attention of the community. We attribute this imbalance of vigilance to the weak practicality of existing deployment-stage backdoor attack algorithms and the insufficiency of real-world attack demonstrations. To fill the blank, in this work, we study the realistic threat of deployment-stage backdoor attacks on DNNs. We base our study on a commonly used deployment-stage attack paradigm -- adversarial weight attack, where adversaries selectively modify model weights to embed backdoor into deployed DNNs. To approach realistic practicality, we propose the first gray-box and physically realizable weights attack algorithm for backdoor injection, namely subnet replacement attack (SRA), which only requires architecture information of the victim model and can support physical triggers in the real world. Extensive experimental simulations and system-level real-world attack demonstrations are conducted. Our results not only suggest the effectiveness and practicality of the proposed attack algorithm, but also reveal the practical risk of a novel type of computer virus that may widely spread and stealthily inject backdoor into DNN models in user devices. By our study, we call for more attention to the vulnerability of DNNs in the deployment stage.

摘要: AI安全社区的一个主要目标是安全可靠地为现实世界的应用程序生成和部署深度学习模型。为此，基于数据中毒的深度神经网络(DNNs)在生产阶段(或训练阶段)的后门攻击以及相应的防御措施近年来得到了广泛的研究。具有讽刺意味的是，部署阶段的后门攻击通常会发生在非专业用户的设备上，因此在现实场景中可以说威胁要大得多，但社区对此的关注要少得多。我们将这种警惕性的不平衡归因于现有部署阶段后门攻击算法的实用性较弱，以及现实世界攻击演示的不足。为了填补这一空白，在这项工作中，我们研究了部署阶段后门攻击对DNNs的现实威胁。我们的研究基于一种常用的部署阶段攻击范例--对抗性权重攻击，在这种攻击中，攻击者有选择地修改模型权重，以将后门嵌入到部署的DNN中。为了更接近实际应用，我们提出了第一种灰盒物理可实现权重的后门注入攻击算法，即子网替换攻击算法(SRA)，该算法只需要受害者模型的体系结构信息，能够支持现实世界中的物理触发器。进行了广泛的实验模拟和系统级的真实世界攻击演示。我们的结果不仅表明了提出的攻击算法的有效性和实用性，而且揭示了一种新型计算机病毒的实际风险，这种病毒可能会广泛传播并偷偷地向用户设备中的DNN模型注入后门。通过我们的研究，我们呼吁更多地关注DNNs在部署阶段的脆弱性。



