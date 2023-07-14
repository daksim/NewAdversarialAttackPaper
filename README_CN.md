# Latest Adversarial Attack Papers
**update at 2023-07-14 15:34:03**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Visually Adversarial Attacks and Defenses in the Physical World: A Survey**

物理世界中的视觉对抗性攻击和防御：综述 cs.CV

**SubmitDate**: 2023-07-13    [abs](http://arxiv.org/abs/2211.01671v5) [paper-pdf](http://arxiv.org/pdf/2211.01671v5)

**Authors**: Xingxing Wei, Bangzheng Pu, Jiefan Lu, Baoyuan Wu

**Abstract**: Although Deep Neural Networks (DNNs) have been widely applied in various real-world scenarios, they are vulnerable to adversarial examples. The current adversarial attacks in computer vision can be divided into digital attacks and physical attacks according to their different attack forms. Compared with digital attacks, which generate perturbations in the digital pixels, physical attacks are more practical in the real world. Owing to the serious security problem caused by physically adversarial examples, many works have been proposed to evaluate the physically adversarial robustness of DNNs in the past years. In this paper, we summarize a survey versus the current physically adversarial attacks and physically adversarial defenses in computer vision. To establish a taxonomy, we organize the current physical attacks from attack tasks, attack forms, and attack methods, respectively. Thus, readers can have a systematic knowledge of this topic from different aspects. For the physical defenses, we establish the taxonomy from pre-processing, in-processing, and post-processing for the DNN models to achieve full coverage of the adversarial defenses. Based on the above survey, we finally discuss the challenges of this research field and further outlook on the future direction.

摘要: 尽管深度神经网络(DNN)已被广泛应用于各种现实场景中，但它们很容易受到对手例子的影响。根据攻击形式的不同，目前计算机视觉中的对抗性攻击可分为数字攻击和物理攻击。与在数字像素中产生扰动的数字攻击相比，物理攻击在现实世界中更实用。由于物理对抗实例带来了严重的安全问题，在过去的几年里，人们已经提出了许多工作来评估DNN的物理对抗健壮性。本文对当前计算机视觉中的身体对抗攻击和身体对抗防御进行了综述。为了建立分类，我们分别从攻击任务、攻击形式和攻击方法三个方面对当前的物理攻击进行了组织。因此，读者可以从不同的方面对这一主题有一个系统的了解。对于物理防御，我们从DNN模型的前处理、内处理和后处理三个方面建立了分类，以实现对抗性防御的全覆盖。在上述调查的基础上，我们最后讨论了该研究领域面临的挑战和对未来方向的进一步展望。



## **2. Introducing Foundation Models as Surrogate Models: Advancing Towards More Practical Adversarial Attacks**

引入基础模型作为代理模型：向更实用的对抗性攻击迈进 cs.LG

**SubmitDate**: 2023-07-13    [abs](http://arxiv.org/abs/2307.06608v1) [paper-pdf](http://arxiv.org/pdf/2307.06608v1)

**Authors**: Jiaming Zhang, Jitao Sang, Qi Yi

**Abstract**: Recently, the no-box adversarial attack, in which the attacker lacks access to the model's architecture, weights, and training data, become the most practical and challenging attack setup. However, there is an unawareness of the potential and flexibility inherent in the surrogate model selection process on no-box setting. Inspired by the burgeoning interest in utilizing foundational models to address downstream tasks, this paper adopts an innovative idea that 1) recasting adversarial attack as a downstream task. Specifically, image noise generation to meet the emerging trend and 2) introducing foundational models as surrogate models. Harnessing the concept of non-robust features, we elaborate on two guiding principles for surrogate model selection to explain why the foundational model is an optimal choice for this role. However, paradoxically, we observe that these foundational models underperform. Analyzing this unexpected behavior within the feature space, we attribute the lackluster performance of foundational models (e.g., CLIP) to their significant representational capacity and, conversely, their lack of discriminative prowess. To mitigate this issue, we propose the use of a margin-based loss strategy for the fine-tuning of foundational models on target images. The experimental results verify that our approach, which employs the basic Fast Gradient Sign Method (FGSM) attack algorithm, outstrips the performance of other, more convoluted algorithms. We conclude by advocating for the research community to consider surrogate models as crucial determinants in the effectiveness of adversarial attacks in no-box settings. The implications of our work bear relevance for improving the efficacy of such adversarial attacks and the overall robustness of AI systems.

摘要: 最近，攻击者无法访问模型的体系结构、权重和训练数据的非盒子对抗性攻击成为最实用和最具挑战性的攻击设置。然而，人们没有意识到在非盒子设置的代理模型选择过程中固有的潜力和灵活性。受利用基础模型处理下游任务的兴趣日益浓厚的启发，本文采用了一种创新的思想：1)将对抗性攻击重塑为下游任务。具体地说，图像噪声生成以适应新兴趋势，以及2)引入基础模型作为代理模型。利用非稳健特征的概念，我们详细阐述了代理模型选择的两个指导原则，以解释为什么基础模型是这一角色的最佳选择。然而，矛盾的是，我们观察到这些基本模型表现不佳。分析特征空间中的这种意外行为，我们将基础模型(如CLIP)表现平平的原因归因于它们显著的表征能力，反之，它们缺乏区分能力。为了缓解这一问题，我们建议使用基于边际的损失策略来微调目标图像上的基础模型。实验结果表明，该方法采用了基本的快速梯度符号算法(FGSM)，其性能优于其他卷积算法。最后，我们倡导研究界将代理模型视为非盒子环境下对抗性攻击有效性的关键决定因素。我们的工作对于提高这种对抗性攻击的效率和人工智能系统的整体稳健性具有重要意义。



## **3. Adversarial Policies Beat Superhuman Go AIs**

对抗性政策击败了超人围棋 cs.LG

Accepted to ICML 2023, see paper for changelog

**SubmitDate**: 2023-07-13    [abs](http://arxiv.org/abs/2211.00241v4) [paper-pdf](http://arxiv.org/pdf/2211.00241v4)

**Authors**: Tony T. Wang, Adam Gleave, Tom Tseng, Kellin Pelrine, Nora Belrose, Joseph Miller, Michael D. Dennis, Yawen Duan, Viktor Pogrebniak, Sergey Levine, Stuart Russell

**Abstract**: We attack the state-of-the-art Go-playing AI system KataGo by training adversarial policies against it, achieving a >97% win rate against KataGo running at superhuman settings. Our adversaries do not win by playing Go well. Instead, they trick KataGo into making serious blunders. Our attack transfers zero-shot to other superhuman Go-playing AIs, and is comprehensible to the extent that human experts can implement it without algorithmic assistance to consistently beat superhuman AIs. The core vulnerability uncovered by our attack persists even in KataGo agents adversarially trained to defend against our attack. Our results demonstrate that even superhuman AI systems may harbor surprising failure modes. Example games are available https://goattack.far.ai/.

摘要: 我们通过训练对抗对手的策略来攻击最先进的围棋人工智能系统KataGo，在超人设置下对KataGo的胜率达到了97%以上。我们的对手不是靠下好围棋来取胜的。相反，他们欺骗KataGo犯下严重的错误。我们的攻击将零射转移到其他超人围棋人工智能，并且可以理解到人类专家可以在没有算法辅助的情况下实现它，以一致地击败超人人工智能。我们的攻击发现的核心漏洞仍然存在，即使在接受过敌意训练以防御我们的攻击的KataGo代理中也是如此。我们的结果表明，即使是超人人工智能系统也可能存在令人惊讶的故障模式。Https://goattack.far.ai/.提供了示例游戏



## **4. Multi-objective Evolutionary Search of Variable-length Composite Semantic Perturbations**

变长复合语义扰动的多目标进化搜索 cs.CV

**SubmitDate**: 2023-07-13    [abs](http://arxiv.org/abs/2307.06548v1) [paper-pdf](http://arxiv.org/pdf/2307.06548v1)

**Authors**: Jialiang Suna, Wen Yao, Tingsong Jianga, Xiaoqian Chena

**Abstract**: Deep neural networks have proven to be vulnerable to adversarial attacks in the form of adding specific perturbations on images to make wrong outputs. Designing stronger adversarial attack methods can help more reliably evaluate the robustness of DNN models. To release the harbor burden and improve the attack performance, auto machine learning (AutoML) has recently emerged as one successful technique to help automatically find the near-optimal adversarial attack strategy. However, existing works about AutoML for adversarial attacks only focus on $L_{\infty}$-norm-based perturbations. In fact, semantic perturbations attract increasing attention due to their naturalnesses and physical realizability. To bridge the gap between AutoML and semantic adversarial attacks, we propose a novel method called multi-objective evolutionary search of variable-length composite semantic perturbations (MES-VCSP). Specifically, we construct the mathematical model of variable-length composite semantic perturbations, which provides five gradient-based semantic attack methods. The same type of perturbation in an attack sequence is allowed to be performed multiple times. Besides, we introduce the multi-objective evolutionary search consisting of NSGA-II and neighborhood search to find near-optimal variable-length attack sequences. Experimental results on CIFAR10 and ImageNet datasets show that compared with existing methods, MES-VCSP can obtain adversarial examples with a higher attack success rate, more naturalness, and less time cost.

摘要: 深度神经网络已被证明容易受到敌意攻击，其形式是在图像上添加特定的扰动以做出错误的输出。设计更强的对抗性攻击方法有助于更可靠地评估DNN模型的稳健性。为了减轻港口负担，提高攻击性能，自动机器学习(AutoML)近年来成为一种成功的技术，可以帮助自动找到接近最优的对抗性攻击策略。然而，现有的关于AutoML用于对抗性攻击的工作仅集中在基于$L_{\inty}$-范数的扰动上。事实上，语义扰动由于其自然性和物理可实现性而受到越来越多的关注。为了在AutoML和语义攻击之间架起一座桥梁，我们提出了一种新的方法-可变长度复合语义扰动的多目标进化搜索(MES-VCSP)。具体而言，我们构建了变长复合语义扰动的数学模型，给出了五种基于梯度的语义攻击方法。允许多次执行攻击序列中相同类型的扰动。此外，我们还引入了由NSGA-II和邻域搜索组成的多目标进化搜索来寻找接近最优的变长攻击序列。在CIFAR10和ImageNet数据集上的实验结果表明，与现有方法相比，MES-VCSP能够获得攻击成功率更高、更自然、时间开销更少的对抗性实例。



## **5. Hiding in Plain Sight: Differential Privacy Noise Exploitation for Evasion-resilient Localized Poisoning Attacks in Multiagent Reinforcement Learning**

隐藏在明显的视线中：在多智能体强化学习中利用差分隐私噪声进行逃避弹性局部中毒攻击 cs.LG

6 pages, 4 figures, Published in the proceeding of the ICMLC 2023,  9-11 July 2023, The University of Adelaide, Adelaide, Australia

**SubmitDate**: 2023-07-13    [abs](http://arxiv.org/abs/2307.00268v2) [paper-pdf](http://arxiv.org/pdf/2307.00268v2)

**Authors**: Md Tamjid Hossain, Hung La

**Abstract**: Lately, differential privacy (DP) has been introduced in cooperative multiagent reinforcement learning (CMARL) to safeguard the agents' privacy against adversarial inference during knowledge sharing. Nevertheless, we argue that the noise introduced by DP mechanisms may inadvertently give rise to a novel poisoning threat, specifically in the context of private knowledge sharing during CMARL, which remains unexplored in the literature. To address this shortcoming, we present an adaptive, privacy-exploiting, and evasion-resilient localized poisoning attack (PeLPA) that capitalizes on the inherent DP-noise to circumvent anomaly detection systems and hinder the optimal convergence of the CMARL model. We rigorously evaluate our proposed PeLPA attack in diverse environments, encompassing both non-adversarial and multiple-adversarial contexts. Our findings reveal that, in a medium-scale environment, the PeLPA attack with attacker ratios of 20% and 40% can lead to an increase in average steps to goal by 50.69% and 64.41%, respectively. Furthermore, under similar conditions, PeLPA can result in a 1.4x and 1.6x computational time increase in optimal reward attainment and a 1.18x and 1.38x slower convergence for attacker ratios of 20% and 40%, respectively.

摘要: 最近，在协作多智能体强化学习(CMARL)中引入了差异隐私(DP)，以保护智能体在知识共享过程中的隐私不受对手推理的影响。然而，我们认为DP机制引入的噪声可能无意中引起一种新的中毒威胁，特别是在CMARL期间的私人知识共享的背景下，这在文献中仍未被探索。针对这一缺陷，我们提出了一种自适应的、利用隐私攻击和逃避弹性的局部中毒攻击(PeLPA)，该攻击利用固有的DP噪声来绕过异常检测系统并阻碍CMARL模型的最优收敛。我们在不同的环境中严格评估我们提出的PeLPA攻击，包括非对抗性和多对抗性环境。我们的研究结果表明，在中等规模的环境中，攻击者比率分别为20%和40%的PeLPA攻击可以使到达目标的平均步数分别增加50.69%和64.41%。此外，在类似的条件下，PeLPA可以使最优奖励获得的计算时间分别增加1.4倍和1.6倍，对于攻击比率分别为20%和40%的攻击者，收敛速度分别慢1.18倍和1.38倍。



## **6. The Butterfly Effect in AI Fairness and Bias**

人工智能公平与偏见中的蝴蝶效应 cs.CY

Working Draft

**SubmitDate**: 2023-07-13    [abs](http://arxiv.org/abs/2307.05842v2) [paper-pdf](http://arxiv.org/pdf/2307.05842v2)

**Authors**: Emilio Ferrara

**Abstract**: The Butterfly Effect, a concept originating from chaos theory, underscores how small changes can have significant and unpredictable impacts on complex systems. In the context of AI fairness and bias, the Butterfly Effect can stem from a variety of sources, such as small biases or skewed data inputs during algorithm development, saddle points in training, or distribution shifts in data between training and testing phases. These seemingly minor alterations can lead to unexpected and substantial unfair outcomes, disproportionately affecting underrepresented individuals or groups and perpetuating pre-existing inequalities. Moreover, the Butterfly Effect can amplify inherent biases within data or algorithms, exacerbate feedback loops, and create vulnerabilities for adversarial attacks. Given the intricate nature of AI systems and their societal implications, it is crucial to thoroughly examine any changes to algorithms or input data for potential unintended consequences. In this paper, we envision both algorithmic and empirical strategies to detect, quantify, and mitigate the Butterfly Effect in AI systems, emphasizing the importance of addressing these challenges to promote fairness and ensure responsible AI development.

摘要: 蝴蝶效应是一个起源于混沌理论的概念，它强调了微小的变化如何能够对复杂系统产生重大和不可预测的影响。在人工智能公平和偏见的背景下，蝴蝶效应可能源于各种来源，例如算法开发期间的小偏差或倾斜的数据输入，训练中的鞍点，或训练和测试阶段之间数据的分布偏移。这些看似微小的改变可能会导致意想不到的重大不公平结果，对代表性不足的个人或群体造成不成比例的影响，并使先前存在的不平等永久化。此外，蝴蝶效应会放大数据或算法中的固有偏见，加剧反馈循环，并为对手攻击创造漏洞。鉴于人工智能系统的错综复杂的性质及其社会影响，彻底检查算法或输入数据的任何变化是否有潜在的意外后果是至关重要的。在本文中，我们设想了算法和经验策略来检测、量化和缓解人工智能系统中的蝴蝶效应，强调了应对这些挑战的重要性，以促进公平和确保负责任的人工智能发展。



## **7. Microbial Genetic Algorithm-based Black-box Attack against Interpretable Deep Learning Systems**

基于微生物遗传算法的可解释深度学习系统黑盒攻击 cs.CV

**SubmitDate**: 2023-07-13    [abs](http://arxiv.org/abs/2307.06496v1) [paper-pdf](http://arxiv.org/pdf/2307.06496v1)

**Authors**: Eldor Abdukhamidov, Mohammed Abuhamad, Simon S. Woo, Eric Chan-Tin, Tamer Abuhmed

**Abstract**: Deep learning models are susceptible to adversarial samples in white and black-box environments. Although previous studies have shown high attack success rates, coupling DNN models with interpretation models could offer a sense of security when a human expert is involved, who can identify whether a given sample is benign or malicious. However, in white-box environments, interpretable deep learning systems (IDLSes) have been shown to be vulnerable to malicious manipulations. In black-box settings, as access to the components of IDLSes is limited, it becomes more challenging for the adversary to fool the system. In this work, we propose a Query-efficient Score-based black-box attack against IDLSes, QuScore, which requires no knowledge of the target model and its coupled interpretation model. QuScore is based on transfer-based and score-based methods by employing an effective microbial genetic algorithm. Our method is designed to reduce the number of queries necessary to carry out successful attacks, resulting in a more efficient process. By continuously refining the adversarial samples created based on feedback scores from the IDLS, our approach effectively navigates the search space to identify perturbations that can fool the system. We evaluate the attack's effectiveness on four CNN models (Inception, ResNet, VGG, DenseNet) and two interpretation models (CAM, Grad), using both ImageNet and CIFAR datasets. Our results show that the proposed approach is query-efficient with a high attack success rate that can reach between 95% and 100% and transferability with an average success rate of 69% in the ImageNet and CIFAR datasets. Our attack method generates adversarial examples with attribution maps that resemble benign samples. We have also demonstrated that our attack is resilient against various preprocessing defense techniques and can easily be transferred to different DNN models.

摘要: 深度学习模型在白盒和黑盒环境中容易受到对抗性样本的影响。尽管之前的研究表明攻击成功率很高，但当涉及到人类专家时，将DNN模型与解释模型结合可以提供一种安全感，专家可以识别给定样本是良性的还是恶意的。然而，在白盒环境中，可解释深度学习系统(IDLS)已被证明容易受到恶意操纵。在黑盒设置中，由于对IDLS组件的访问受到限制，因此对手欺骗系统变得更具挑战性。在这项工作中，我们提出了一种针对入侵检测系统的查询高效的基于分数的黑盒攻击，QuScore，它不需要知道目标模型及其耦合解释模型。QuScore基于基于转移和基于分数的方法，采用了有效的微生物遗传算法。我们的方法旨在减少执行成功攻击所需的查询数量，从而实现更高效的过程。通过不断改进基于IDL反馈分数创建的敌意样本，我们的方法有效地导航搜索空间以识别可以愚弄系统的扰动。我们使用ImageNet和CIFAR数据集，在四个CNN模型(初始、ResNet、VGG、DenseNet)和两个解释模型(CAM、Grad)上评估了攻击的有效性。实验结果表明，该方法具有较高的查询效率，在ImageNet和CIFAR数据集中具有较高的攻击成功率(95%~100%)和可移植性(平均成功率为69%)。我们的攻击方法生成具有类似于良性样本的属性图的对抗性示例。我们还证明了我们的攻击对各种预处理防御技术具有弹性，并且可以很容易地转移到不同的DNN模型上。



## **8. Single-Class Target-Specific Attack against Interpretable Deep Learning Systems**

针对可解释深度学习系统的单类特定目标攻击 cs.CV

13 pages

**SubmitDate**: 2023-07-12    [abs](http://arxiv.org/abs/2307.06484v1) [paper-pdf](http://arxiv.org/pdf/2307.06484v1)

**Authors**: Eldor Abdukhamidov, Mohammed Abuhamad, George K. Thiruvathukal, Hyoungshick Kim, Tamer Abuhmed

**Abstract**: In this paper, we present a novel Single-class target-specific Adversarial attack called SingleADV. The goal of SingleADV is to generate a universal perturbation that deceives the target model into confusing a specific category of objects with a target category while ensuring highly relevant and accurate interpretations. The universal perturbation is stochastically and iteratively optimized by minimizing the adversarial loss that is designed to consider both the classifier and interpreter costs in targeted and non-targeted categories. In this optimization framework, ruled by the first- and second-moment estimations, the desired loss surface promotes high confidence and interpretation score of adversarial samples. By avoiding unintended misclassification of samples from other categories, SingleADV enables more effective targeted attacks on interpretable deep learning systems in both white-box and black-box scenarios. To evaluate the effectiveness of SingleADV, we conduct experiments using four different model architectures (ResNet-50, VGG-16, DenseNet-169, and Inception-V3) coupled with three interpretation models (CAM, Grad, and MASK). Through extensive empirical evaluation, we demonstrate that SingleADV effectively deceives the target deep learning models and their associated interpreters under various conditions and settings. Our experimental results show that the performance of SingleADV is effective, with an average fooling ratio of 0.74 and an adversarial confidence level of 0.78 in generating deceptive adversarial samples. Furthermore, we discuss several countermeasures against SingleADV, including a transfer-based learning approach and existing preprocessing defenses.

摘要: 在本文中，我们提出了一种新的单类目标特定的对抗性攻击，称为SingleADV。SingleADV的目标是产生一种普遍的扰动，欺骗目标模型，使其混淆特定类别的对象和目标类别，同时确保高度相关和准确的解释。通过最小化对抗性损失来随机和迭代地优化普遍扰动，所述对手损失被设计为在目标和非目标类别中同时考虑分类器和解释器成本。在这个优化框架中，以一阶和二阶矩估计为准则，期望的损失面提高了敌方样本的置信度和解释得分。通过避免其他类别样本的意外错误分类，SingleADV能够在白盒和黑盒场景中对可解释的深度学习系统进行更有效的针对性攻击。为了评估SingleADV的有效性，我们使用四种不同的模型体系结构(ResNet-50、VGG-16、DenseNet-169和Inception-V3)以及三种解释模型(CAM、Grad和MASK)进行了实验。通过广泛的实证评估，我们证明了SingleADV在不同的条件和背景下有效地欺骗了目标深度学习模型及其相关的解释人员。实验结果表明，SingleADV的性能是有效的，在生成欺骗性对手样本时，平均愚弄率为0.74，对手置信度为0.78。此外，我们讨论了针对SingleADV的几种对策，包括基于迁移的学习方法和现有的预处理防御措施。



## **9. Rational Neural Network Controllers**

有理神经网络控制器 eess.SY

20 Pages, 12 Figures

**SubmitDate**: 2023-07-12    [abs](http://arxiv.org/abs/2307.06287v1) [paper-pdf](http://arxiv.org/pdf/2307.06287v1)

**Authors**: Matthew Newton, Antonis Papachristodoulou

**Abstract**: Neural networks have shown great success in many machine learning related tasks, due to their ability to act as general function approximators. Recent work has demonstrated the effectiveness of neural networks in control systems (known as neural feedback loops), most notably by using a neural network as a controller. However, one of the big challenges of this approach is that neural networks have been shown to be sensitive to adversarial attacks. This means that, unless they are designed properly, they are not an ideal candidate for controllers due to issues with robustness and uncertainty, which are pivotal aspects of control systems. There has been initial work on robustness to both analyse and design dynamical systems with neural network controllers. However, one prominent issue with these methods is that they use existing neural network architectures tailored for traditional machine learning tasks. These structures may not be appropriate for neural network controllers and it is important to consider alternative architectures. This paper considers rational neural networks and presents novel rational activation functions, which can be used effectively in robustness problems for neural feedback loops. Rational activation functions are replaced by a general rational neural network structure, which is convex in the neural network's parameters. A method is proposed to recover a stabilising controller from a Sum of Squares feasibility test. This approach is then applied to a refined rational neural network which is more compatible with Sum of Squares programming. Numerical examples show that this method can successfully recover stabilising rational neural network controllers for neural feedback loops with non-linear plants with noise and parametric uncertainty.

摘要: 神经网络在许多与机器学习相关的任务中表现出了巨大的成功，这归功于它作为通用函数逼近器的能力。最近的工作证明了神经网络在控制系统(称为神经反馈回路)中的有效性，最显著的是使用神经网络作为控制器。然而，这种方法的一大挑战是，神经网络已被证明对对手攻击很敏感。这意味着，除非它们设计得当，否则由于控制系统的关键方面--鲁棒性和不确定性问题，它们不是理想的控制器候选者。在分析和设计带有神经网络控制器的动态系统时，已经有了关于鲁棒性的初步工作。然而，这些方法的一个突出问题是它们使用了为传统机器学习任务量身定做的现有神经网络结构。这些结构可能不适合神经网络控制器，因此重要的是要考虑替代体系结构。本文考虑了有理神经网络，提出了新的有理激活函数，有效地解决了神经反馈环路的鲁棒性问题。有理激活函数被一种一般的有理神经网络结构代替，该结构在神经网络参数中是凸的。提出了一种从平方和可行性检验中恢复镇定控制器的方法。然后将这种方法应用于改进的有理神经网络，该网络与平方和编程更兼容。数值算例表明，对于含有噪声和参数不确定性的非线性对象，该方法能成功地恢复稳定的有理神经网络控制器。



## **10. I See Dead People: Gray-Box Adversarial Attack on Image-To-Text Models**

我看到死人：对图像到文本模型的灰箱对抗性攻击 cs.CV

**SubmitDate**: 2023-07-12    [abs](http://arxiv.org/abs/2306.07591v2) [paper-pdf](http://arxiv.org/pdf/2306.07591v2)

**Authors**: Raz Lapid, Moshe Sipper

**Abstract**: Modern image-to-text systems typically adopt the encoder-decoder framework, which comprises two main components: an image encoder, responsible for extracting image features, and a transformer-based decoder, used for generating captions. Taking inspiration from the analysis of neural networks' robustness against adversarial perturbations, we propose a novel gray-box algorithm for creating adversarial examples in image-to-text models. Unlike image classification tasks that have a finite set of class labels, finding visually similar adversarial examples in an image-to-text task poses greater challenges because the captioning system allows for a virtually infinite space of possible captions. In this paper, we present a gray-box adversarial attack on image-to-text, both untargeted and targeted. We formulate the process of discovering adversarial perturbations as an optimization problem that uses only the image-encoder component, meaning the proposed attack is language-model agnostic. Through experiments conducted on the ViT-GPT2 model, which is the most-used image-to-text model in Hugging Face, and the Flickr30k dataset, we demonstrate that our proposed attack successfully generates visually similar adversarial examples, both with untargeted and targeted captions. Notably, our attack operates in a gray-box manner, requiring no knowledge about the decoder module. We also show that our attacks fool the popular open-source platform Hugging Face.

摘要: 现代图像到文本系统通常采用编解码器框架，该框架包括两个主要组件：负责提取图像特征的图像编码器和用于生成字幕的基于转换器的解码器。从神经网络对对抗性扰动的鲁棒性分析中得到启发，我们提出了一种新的灰盒算法，用于在图像到文本模型中创建对抗性示例。与具有有限类别标签集的图像分类任务不同，在图像到文本的任务中找到视觉上相似的对抗性例子带来了更大的挑战，因为字幕系统允许可能的字幕的几乎无限空间。在本文中，我们提出了一种针对图像到文本的灰盒对抗性攻击，包括无目标攻击和目标攻击。我们将发现敌意扰动的过程描述为一个只使用图像编码器组件的优化问题，这意味着所提出的攻击是语言模型不可知的。通过在拥抱脸中最常用的图文转换模型VIT-GPT2模型和Flickr30k数据集上的实验，我们证明了我们的攻击成功地生成了视觉上相似的对抗性例子，无论是无目标字幕还是有目标字幕。值得注意的是，我们的攻击以灰盒方式运行，不需要了解解码器模块。我们还表明，我们的攻击愚弄了流行的开源平台拥抱脸。



## **11. Random-Set Convolutional Neural Network (RS-CNN) for Epistemic Deep Learning**

认知深度学习的随机集卷积神经网络(RS-CNN) cs.LG

**SubmitDate**: 2023-07-11    [abs](http://arxiv.org/abs/2307.05772v1) [paper-pdf](http://arxiv.org/pdf/2307.05772v1)

**Authors**: Shireen Kudukkil Manchingal, Muhammad Mubashar, Kaizheng Wang, Keivan Shariatmadar, Fabio Cuzzolin

**Abstract**: Machine learning is increasingly deployed in safety-critical domains where robustness against adversarial attacks is crucial and erroneous predictions could lead to potentially catastrophic consequences. This highlights the need for learning systems to be equipped with the means to determine a model's confidence in its prediction and the epistemic uncertainty associated with it, 'to know when a model does not know'. In this paper, we propose a novel Random-Set Convolutional Neural Network (RS-CNN) for classification which predicts belief functions rather than probability vectors over the set of classes, using the mathematics of random sets, i.e., distributions over the power set of the sample space. Based on the epistemic deep learning approach, random-set models are capable of representing the 'epistemic' uncertainty induced in machine learning by limited training sets. We estimate epistemic uncertainty by approximating the size of credal sets associated with the predicted belief functions, and experimentally demonstrate how our approach outperforms competing uncertainty-aware approaches in a classical evaluation setting. The performance of RS-CNN is best demonstrated on OOD samples where it manages to capture the true prediction while standard CNNs fail.

摘要: 机器学习越来越多地部署在安全关键领域，在这些领域，对对手攻击的健壮性至关重要，错误的预测可能会导致潜在的灾难性后果。这突显了学习系统需要配备这样的手段，以确定模型对其预测的信心以及与之相关的认知不确定性，即“知道模型何时不知道”。本文提出了一种新的随机集卷积神经网络(RS-CNN)用于分类，它利用随机集的数学知识，即样本空间功率集上的分布，来预测类集合上的信任函数而不是概率向量。在认知深度学习方法的基础上，随机集模型能够用有限的训练集来表示机器学习中的认知不确定性。我们通过近似与预测的信念函数相关联的信任集的大小来估计认知不确定性，并通过实验演示了我们的方法如何在经典评估环境中优于竞争的不确定性感知方法。RS-CNN的性能在OOD样本上得到了最好的演示，在标准CNN失败的情况下，它设法捕捉到了真实的预测。



## **12. In and Out-of-Domain Text Adversarial Robustness via Label Smoothing**

通过标签平滑实现域内和域外文本对抗健壮性 cs.CL

**SubmitDate**: 2023-07-11    [abs](http://arxiv.org/abs/2212.10258v2) [paper-pdf](http://arxiv.org/pdf/2212.10258v2)

**Authors**: Yahan Yang, Soham Dan, Dan Roth, Insup Lee

**Abstract**: Recently it has been shown that state-of-the-art NLP models are vulnerable to adversarial attacks, where the predictions of a model can be drastically altered by slight modifications to the input (such as synonym substitutions). While several defense techniques have been proposed, and adapted, to the discrete nature of text adversarial attacks, the benefits of general-purpose regularization methods such as label smoothing for language models, have not been studied. In this paper, we study the adversarial robustness provided by various label smoothing strategies in foundational models for diverse NLP tasks in both in-domain and out-of-domain settings. Our experiments show that label smoothing significantly improves adversarial robustness in pre-trained models like BERT, against various popular attacks. We also analyze the relationship between prediction confidence and robustness, showing that label smoothing reduces over-confident errors on adversarial examples.

摘要: 最近有研究表明，最新的自然语言处理模型容易受到敌意攻击，模型的预测可以通过对输入的轻微修改(如同义词替换)来显著改变。虽然已经提出了几种防御技术，并对其进行了调整，以适应文本对抗性攻击的离散性质，但还没有研究通用正则化方法的好处，例如语言模型的标签平滑。在本文中，我们研究了不同的标签平滑策略在基本模型中对不同的NLP任务在域内和域外环境下提供的对抗健壮性。我们的实验表明，在像BERT这样的预先训练的模型中，标签平滑显著提高了对抗各种流行攻击的健壮性。我们还分析了预测置信度和稳健性之间的关系，表明标签平滑减少了对抗性例子中的过度自信错误。



## **13. Zeroth-order Optimization with Weak Dimension Dependency**

具有弱维相关性的零阶优化 math.OC

to be published in COLT 2023

**SubmitDate**: 2023-07-11    [abs](http://arxiv.org/abs/2307.05753v1) [paper-pdf](http://arxiv.org/pdf/2307.05753v1)

**Authors**: Pengyun Yue, Long Yang, Cong Fang, Zhouchen Lin

**Abstract**: Zeroth-order optimization is a fundamental research topic that has been a focus of various learning tasks, such as black-box adversarial attacks, bandits, and reinforcement learning. However, in theory, most complexity results assert a linear dependency on the dimension of optimization variable, which implies paralyzations of zeroth-order algorithms for high-dimensional problems and cannot explain their effectiveness in practice. In this paper, we present a novel zeroth-order optimization theory characterized by complexities that exhibit weak dependencies on dimensionality. The key contribution lies in the introduction of a new factor, denoted as $\mathrm{ED}_{\alpha}=\sup_{x\in \mathbb{R}^d}\sum_{i=1}^d\sigma_i^\alpha(\nabla^2 f(x))$ ($\alpha>0$, $\sigma_i(\cdot)$ is the $i$-th singular value in non-increasing order), which effectively functions as a measure of dimensionality. The algorithms we propose demonstrate significantly reduced complexities when measured in terms of the factor $\mathrm{ED}_{\alpha}$. Specifically, we first study a well-known zeroth-order algorithm from Nesterov and Spokoiny (2017) on quadratic objectives and show a complexity of $\mathcal{O}\left(\frac{\mathrm{ED}_1}{\sigma_d}\log(1/\epsilon)\right)$ for the strongly convex setting. Furthermore, we introduce novel algorithms that leverages the Heavy-ball mechanism. Our proposed algorithm exhibits a complexity of $\mathcal{O}\left(\frac{\mathrm{ED}_{1/2}}{\sqrt{\sigma_d}}\cdot\log{\frac{L}{\mu}}\cdot\log(1/\epsilon)\right)$. We further expand the scope of the method to encompass generic smooth optimization problems under an additional Hessian-smooth condition. The resultant algorithms demonstrate remarkable complexities which improve by an order in $d$ under appropriate conditions. Our analysis lays the foundation for zeroth-order optimization methods for smooth functions within high-dimensional settings.

摘要: 零阶优化是一个基础性的研究课题，一直是各种学习任务的焦点，如黑盒对抗性攻击、强盗和强化学习。然而，在理论上，大多数复杂性结果与优化变量的维度呈线性关系，这意味着高维问题的零阶算法陷入瘫痪，无法解释其在实践中的有效性。在这篇文章中，我们提出了一种新的零阶优化理论，其特征是复杂性表现出对维度的弱依赖性。其关键贡献在于引入了一个新的因子，记为$\mathm{Ed}_{\α}=\sup_{x\in\mathbb{R}^d}\sum_{i=1}^d\sigma_i^\alpha(\nabla^2 f(X)$($\α>0$，$\sigma_i(\cdot)$是非升序的第$i$奇异值)，它是一种有效的维度度量。当以因子$\mathm{ed}_{\pha}$来衡量时，我们提出的算法的复杂性显著降低。具体地说，我们首先研究了内斯特夫和斯波科尼(2017年)关于二次目标的一个著名的零阶算法，并证明了在强凸设置下的$\mathcal{O}\left(\frac{\mathrm{ED}_1}{\sigma_d}\log(1/\epsilon)\right)$的复杂性。此外，我们引入了利用重球机制的新算法。我们提出的算法具有$\mathcal{O}\left(\frac{\mathrm{ED}_{1/2}}{\sqrt{\sigma_d}}\cdot\log{\frac{L}{\mu}}\cdot\log(1/\epsilon)\right)$.的复杂性我们进一步扩展了该方法的适用范围，使其包含了附加海森光滑条件下的一般光滑优化问题。所得到的算法表现出显著的复杂性，在适当的条件下提高了$d$的数量级。我们的分析为高维光滑函数的零阶优化方法奠定了基础。



## **14. Adversarial Cheap Talk**

对抗性的低级谈资 cs.LG

To be published at ICML 2023. Project video and code are available at  https://sites.google.com/view/adversarial-cheap-talk

**SubmitDate**: 2023-07-11    [abs](http://arxiv.org/abs/2211.11030v4) [paper-pdf](http://arxiv.org/pdf/2211.11030v4)

**Authors**: Chris Lu, Timon Willi, Alistair Letcher, Jakob Foerster

**Abstract**: Adversarial attacks in reinforcement learning (RL) often assume highly-privileged access to the victim's parameters, environment, or data. Instead, this paper proposes a novel adversarial setting called a Cheap Talk MDP in which an Adversary can merely append deterministic messages to the Victim's observation, resulting in a minimal range of influence. The Adversary cannot occlude ground truth, influence underlying environment dynamics or reward signals, introduce non-stationarity, add stochasticity, see the Victim's actions, or access their parameters. Additionally, we present a simple meta-learning algorithm called Adversarial Cheap Talk (ACT) to train Adversaries in this setting. We demonstrate that an Adversary trained with ACT still significantly influences the Victim's training and testing performance, despite the highly constrained setting. Affecting train-time performance reveals a new attack vector and provides insight into the success and failure modes of existing RL algorithms. More specifically, we show that an ACT Adversary is capable of harming performance by interfering with the learner's function approximation, or instead helping the Victim's performance by outputting useful features. Finally, we show that an ACT Adversary can manipulate messages during train-time to directly and arbitrarily control the Victim at test-time. Project video and code are available at https://sites.google.com/view/adversarial-cheap-talk

摘要: 强化学习(RL)中的对抗性攻击通常假定具有访问受害者参数、环境或数据的高度特权。相反，本文提出了一种新的对抗性环境，称为廉价谈话MDP，在该环境中，对手只需将确定性消息附加到受害者的观察中，从而产生最小的影响范围。敌手不能掩盖基本事实、影响潜在环境动态或奖励信号、引入非平稳性、增加随机性、看到受害者的行为或获取他们的参数。此外，我们还提出了一个简单的元学习算法，称为对抗性廉价谈话(ACT)，以在这种情况下训练对手。我们证明，尽管在高度受限的环境下，接受过ACT训练的对手仍然会显著影响受害者的训练和测试表现。影响训练时间性能揭示了新的攻击向量，并提供了对现有RL算法的成功和失败模式的洞察。更具体地说，我们证明了ACT对手能够通过干扰学习者的函数逼近来损害性能，或者相反地通过输出有用的特征来帮助受害者的性能。最后，我们证明了ACT攻击者可以在训练时间内操纵消息，从而在测试时间直接任意控制受害者。项目视频和代码可在https://sites.google.com/view/adversarial-cheap-talk上获得



## **15. Hyper-parameter Tuning for Adversarially Robust Models**

逆稳健模型的超参数整定 cs.LG

**SubmitDate**: 2023-07-11    [abs](http://arxiv.org/abs/2304.02497v2) [paper-pdf](http://arxiv.org/pdf/2304.02497v2)

**Authors**: Pedro Mendes, Paolo Romano, David Garlan

**Abstract**: This work focuses on the problem of hyper-parameter tuning (HPT) for robust (i.e., adversarially trained) models, shedding light on the new challenges and opportunities arising during the HPT process for robust models. To this end, we conduct an extensive experimental study based on 3 popular deep models, in which we explore exhaustively 9 (discretized) HPs, 2 fidelity dimensions, and 2 attack bounds, for a total of 19208 configurations (corresponding to 50 thousand GPU hours). Through this study, we show that the complexity of the HPT problem is further exacerbated in adversarial settings due to the need to independently tune the HPs used during standard and adversarial training: succeeding in doing so (i.e., adopting different HP settings in both phases) can lead to a reduction of up to 80% and 43% of the error for clean and adversarial inputs, respectively. On the other hand, we also identify new opportunities to reduce the cost of HPT for robust models. Specifically, we propose to leverage cheap adversarial training methods to obtain inexpensive, yet highly correlated, estimations of the quality achievable using state-of-the-art methods. We show that, by exploiting this novel idea in conjunction with a recent multi-fidelity optimizer (taKG), the efficiency of the HPT process can be enhanced by up to 2.1x.

摘要: 这项工作集中于稳健(即反向训练)模型的超参数调节(HPT)问题，揭示了稳健模型HPT过程中出现的新挑战和新机遇。为此，我们基于3个流行的深度模型进行了广泛的实验研究，其中我们详尽地探索了9个(离散化的)HPS，2个保真维度，2个攻击界限，总共19208个配置(对应于5万个GPU小时)。通过这项研究，我们表明，由于需要独立调整标准和对抗性训练中使用的HP，HPT问题的复杂性在对抗性环境中进一步加剧：成功做到这一点(即在两个阶段采用不同的HP设置)可以使干净输入和对抗性输入的错误分别减少80%和43%。另一方面，我们也发现了新的机会，以降低稳健模型的HPT成本。具体地说，我们建议利用廉价的对抗性训练方法来获得对使用最先进方法可实现的质量的廉价但高度相关的估计。我们证明，通过利用这一新的想法与最近的多保真优化器(TaKG)相结合，HPT过程的效率可以提高高达2.1倍。



## **16. Revisiting the Trade-off between Accuracy and Robustness via Weight Distribution of Filters**

从滤波器权重分布看精度与稳健性的权衡 cs.CV

**SubmitDate**: 2023-07-11    [abs](http://arxiv.org/abs/2306.03430v2) [paper-pdf](http://arxiv.org/pdf/2306.03430v2)

**Authors**: Xingxing Wei, Shiji Zhao

**Abstract**: Adversarial attacks have been proven to be potential threats to Deep Neural Networks (DNNs), and many methods are proposed to defend against adversarial attacks. However, while enhancing the robustness, the clean accuracy will decline to a certain extent, implying a trade-off existed between the accuracy and robustness. In this paper, we firstly empirically find an obvious distinction between standard and robust models in the filters' weight distribution of the same architecture, and then theoretically explain this phenomenon in terms of the gradient regularization, which shows this difference is an intrinsic property for DNNs, and thus a static network architecture is difficult to improve the accuracy and robustness at the same time. Secondly, based on this observation, we propose a sample-wise dynamic network architecture named Adversarial Weight-Varied Network (AW-Net), which focuses on dealing with clean and adversarial examples with a ``divide and rule" weight strategy. The AW-Net dynamically adjusts network's weights based on regulation signals generated by an adversarial detector, which is directly influenced by the input sample. Benefiting from the dynamic network architecture, clean and adversarial examples can be processed with different network weights, which provides the potentiality to enhance the accuracy and robustness simultaneously. A series of experiments demonstrate that our AW-Net is architecture-friendly to handle both clean and adversarial examples and can achieve better trade-off performance than state-of-the-art robust models.

摘要: 对抗性攻击已被证明是深度神经网络(DNNS)的潜在威胁，并提出了许多方法来防御对抗性攻击。然而，在增强鲁棒性的同时，清洁精度会有一定程度的下降，这意味着在精度和稳健性之间存在着权衡。本文首先通过实验发现，在同一结构的过滤器的权值分布中，标准模型和稳健模型存在明显的差异，然后用梯度正则化的方法从理论上解释了这种差异，说明这种差异是DNN的固有特性，因此静态的网络结构很难同时提高精度和健壮性。其次，基于这一观察结果，我们提出了一种基于样本的动态网络体系结构，称为对抗性变权重网络(AW-Net)，其重点是采用“分而治之”的权重策略来处理干净和对抗性的例子。AW-Net根据对抗性检测器产生的规则信号动态调整网络的权值，该信号直接受输入样本的影响。得益于动态的网络结构，可以用不同的网络权重处理干净的和对抗性的例子，这为同时提高准确性和稳健性提供了可能性。一系列的实验表明，我们的AW-Net是体系结构友好的，可以处理干净和对抗性的例子，并且可以获得比最新的健壮模型更好的折衷性能。



## **17. Mitigating the Accuracy-Robustness Trade-off via Multi-Teacher Adversarial Distillation**

通过多教师对抗性蒸馏缓解精度与稳健性的权衡 cs.LG

**SubmitDate**: 2023-07-11    [abs](http://arxiv.org/abs/2306.16170v2) [paper-pdf](http://arxiv.org/pdf/2306.16170v2)

**Authors**: Shiji Zhao, Xizhe Wang, Xingxing Wei

**Abstract**: Adversarial training is a practical approach for improving the robustness of deep neural networks against adversarial attacks. Although bringing reliable robustness, the performance toward clean examples is negatively affected after adversarial training, which means a trade-off exists between accuracy and robustness. Recently, some studies have tried to use knowledge distillation methods in adversarial training, achieving competitive performance in improving the robustness but the accuracy for clean samples is still limited. In this paper, to mitigate the accuracy-robustness trade-off, we introduce the Multi-Teacher Adversarial Robustness Distillation (MTARD) to guide the model's adversarial training process by applying a strong clean teacher and a strong robust teacher to handle the clean examples and adversarial examples, respectively. During the optimization process, to ensure that different teachers show similar knowledge scales, we design the Entropy-Based Balance algorithm to adjust the teacher's temperature and keep the teachers' information entropy consistent. Besides, to ensure that the student has a relatively consistent learning speed from multiple teachers, we propose the Normalization Loss Balance algorithm to adjust the learning weights of different types of knowledge. A series of experiments conducted on public datasets demonstrate that MTARD outperforms the state-of-the-art adversarial training and distillation methods against various adversarial attacks.

摘要: 对抗性训练是提高深层神经网络抗敌意攻击能力的一种实用方法。虽然带来了可靠的稳健性，但经过对抗性训练后，对干净样本的性能会受到负面影响，这意味着在准确性和稳健性之间存在权衡。近年来，一些研究尝试将知识提取方法应用于对抗性训练，在提高鲁棒性方面取得了较好的性能，但对清洁样本的准确率仍然有限。为了缓解准确性和稳健性之间的权衡，我们引入了多教师对抗稳健性蒸馏(MTARD)来指导模型的对抗训练过程，分别采用强清洁教师和强稳健教师来处理干净实例和对抗性实例。在优化过程中，为了保证不同教师表现出相似的知识尺度，设计了基于熵的均衡算法来调整教师的温度，保持教师信息熵的一致性。此外，为了确保学生从多个老师那里获得相对一致的学习速度，我们提出了归一化损失平衡算法来调整不同类型知识的学习权重。在公开数据集上进行的一系列实验表明，MTARD在对抗各种对抗性攻击方面优于最先进的对抗性训练和蒸馏方法。



## **18. Membership Inference Attacks on DNNs using Adversarial Perturbations**

基于对抗性扰动的DNN成员推理攻击 cs.LG

**SubmitDate**: 2023-07-11    [abs](http://arxiv.org/abs/2307.05193v1) [paper-pdf](http://arxiv.org/pdf/2307.05193v1)

**Authors**: Hassan Ali, Adnan Qayyum, Ala Al-Fuqaha, Junaid Qadir

**Abstract**: Several membership inference (MI) attacks have been proposed to audit a target DNN. Given a set of subjects, MI attacks tell which subjects the target DNN has seen during training. This work focuses on the post-training MI attacks emphasizing high confidence membership detection -- True Positive Rates (TPR) at low False Positive Rates (FPR). Current works in this category -- likelihood ratio attack (LiRA) and enhanced MI attack (EMIA) -- only perform well on complex datasets (e.g., CIFAR-10 and Imagenet) where the target DNN overfits its train set, but perform poorly on simpler datasets (0% TPR by both attacks on Fashion-MNIST, 2% and 0% TPR respectively by LiRA and EMIA on MNIST at 1% FPR). To address this, firstly, we unify current MI attacks by presenting a framework divided into three stages -- preparation, indication and decision. Secondly, we utilize the framework to propose two novel attacks: (1) Adversarial Membership Inference Attack (AMIA) efficiently utilizes the membership and the non-membership information of the subjects while adversarially minimizing a novel loss function, achieving 6% TPR on both Fashion-MNIST and MNIST datasets; and (2) Enhanced AMIA (E-AMIA) combines EMIA and AMIA to achieve 8% and 4% TPRs on Fashion-MNIST and MNIST datasets respectively, at 1% FPR. Thirdly, we introduce two novel augmented indicators that positively leverage the loss information in the Gaussian neighborhood of a subject. This improves TPR of all four attacks on average by 2.5% and 0.25% respectively on Fashion-MNIST and MNIST datasets at 1% FPR. Finally, we propose simple, yet novel, evaluation metric, the running TPR average (RTA) at a given FPR, that better distinguishes different MI attacks in the low FPR region. We also show that AMIA and E-AMIA are more transferable to the unknown DNNs (other than the target DNN) and are more robust to DP-SGD training as compared to LiRA and EMIA.

摘要: 已经提出了几种成员关系推理(MI)攻击来审计目标DNN。给定一组受试者，MI攻击告诉目标DNN在训练期间见过哪些受试者。这项工作主要针对训练后的MI攻击，强调高置信度成员检测--低假阳性率(FPR)下的真阳性率(TPR)。目前在这一类别中的工作--似然比攻击(LIRA)和增强型MI攻击(MIA)--仅在目标DNN超出其训练集的复杂数据集(例如，CIFAR-10和Imagenet)上表现良好，但在较简单的数据集上表现较差(对Fashion-MNIST的两次攻击均为0%TPR，在MNIST上分别为2%和0%的TPR，FFP为1%)。为了解决这一问题，首先，我们统一了当前的MI攻击，提出了一个分为三个阶段的框架--准备、指示和决策。其次，我们利用该框架提出了两个新的攻击：(1)对抗成员关系推理攻击(AMIA)有效地利用了受试者的成员和非成员信息，同时相反地最小化了一个新的损失函数，在Fashion-MNIST和MNIST数据集上都获得了6%的TPR；(2)增强型AMIA(E-AMIA)结合了AMIA和AMIA，在1%的FPR下分别在Fashion-MNIST和MNIST数据集上获得了8%和4%的TPR。第三，我们引入了两个新的增广指标，它们积极地利用了对象的高斯邻域中的损失信息。这在FFP为1%的Fashion-MNIST和MNIST数据集上，将所有四种攻击的TPR分别平均提高了2.5%和0.25%。最后，我们提出了一种简单而新颖的评估指标，即给定FPR下的运行TPR平均值(RTA)，它可以更好地区分低FPR区域内的不同MI攻击。我们还表明，AMIA和E-AMIA对未知DNN(而不是目标DNN)的迁移能力更强，对DP-SGD训练的健壮性也比LIRA和AMIA更强。



## **19. ATWM: Defense against adversarial malware based on adversarial training**

ATWM：基于对抗性训练的恶意软件防御 cs.CR

**SubmitDate**: 2023-07-11    [abs](http://arxiv.org/abs/2307.05095v1) [paper-pdf](http://arxiv.org/pdf/2307.05095v1)

**Authors**: Kun Li, Fan Zhang, Wei Guo

**Abstract**: Deep learning technology has made great achievements in the field of image. In order to defend against malware attacks, researchers have proposed many Windows malware detection models based on deep learning. However, deep learning models are vulnerable to adversarial example attacks. Malware can generate adversarial malware with the same malicious function to attack the malware detection model and evade detection of the model. Currently, many adversarial defense studies have been proposed, but existing adversarial defense studies are based on image sample and cannot be directly applied to malware sample. Therefore, this paper proposes an adversarial malware defense method based on adversarial training. This method uses preprocessing to defend simple adversarial examples to reduce the difficulty of adversarial training. Moreover, this method improves the adversarial defense capability of the model through adversarial training. We experimented with three attack methods in two sets of datasets, and the results show that the method in this paper can improve the adversarial defense capability of the model without reducing the accuracy of the model.

摘要: 深度学习技术在图像领域取得了巨大的成就。为了防御恶意软件攻击，研究人员提出了许多基于深度学习的Windows恶意软件检测模型。然而，深度学习模型很容易受到对抗性范例的攻击。恶意软件可以生成具有相同恶意功能的对抗性恶意软件，以攻击恶意软件检测模型并逃避该模型的检测。目前，已经提出了很多对抗性防御研究，但现有的对抗性防御研究都是基于图像样本，不能直接应用于恶意软件样本。因此，本文提出了一种基于对抗性训练的对抗性恶意软件防御方法。该方法通过对简单的对抗性实例进行预处理，降低了对抗性训练的难度。此外，该方法通过对抗性训练提高了模型的对抗性防御能力。在两组数据集上对三种攻击方法进行了实验，结果表明，本文提出的方法在不降低模型精度的前提下，提高了模型的对抗性防御能力。



## **20. Categorical composable cryptography: extended version**

范畴可合成密码学：扩展版本 cs.CR

Extended version of arXiv:2105.05949 which appeared in FoSSaCS 2022

**SubmitDate**: 2023-07-10    [abs](http://arxiv.org/abs/2208.13232v2) [paper-pdf](http://arxiv.org/pdf/2208.13232v2)

**Authors**: Anne Broadbent, Martti Karvonen

**Abstract**: We formalize the simulation paradigm of cryptography in terms of category theory and show that protocols secure against abstract attacks form a symmetric monoidal category, thus giving an abstract model of composable security definitions in cryptography. Our model is able to incorporate computational security, set-up assumptions and various attack models such as colluding or independently acting subsets of adversaries in a modular, flexible fashion. We conclude by using string diagrams to rederive the security of the one-time pad, correctness of Diffie-Hellman key exchange and no-go results concerning the limits of bipartite and tripartite cryptography, ruling out e.g., composable commitments and broadcasting. On the way, we exhibit two categorical constructions of resource theories that might be of independent interest: one capturing resources shared among multiple parties and one capturing resource conversions that succeed asymptotically.

摘要: 我们用范畴理论形式化了密码学的模拟范型，证明了对抽象攻击安全的协议形成了对称的么半范畴，从而给出了密码学中可组合安全定义的抽象模型。我们的模型能够以模块化、灵活的方式结合计算安全性、设置假设和各种攻击模型，例如串通或独立行动的对手子集。最后，我们使用字符串图重新推导了一次性密钥的安全性，Diffie-Hellman密钥交换的正确性，以及关于二方和三方密码术限制的不可行结果，排除了例如可组合承诺和广播。在此过程中，我们展示了两种可能独立感兴趣的资源理论范畴结构：一种是捕获多方共享的资源，另一种是捕获渐近成功的资源转换。



## **21. On the Robustness of Bayesian Neural Networks to Adversarial Attacks**

贝叶斯神经网络对敌方攻击的稳健性研究 cs.LG

arXiv admin note: text overlap with arXiv:2002.04359

**SubmitDate**: 2023-07-10    [abs](http://arxiv.org/abs/2207.06154v2) [paper-pdf](http://arxiv.org/pdf/2207.06154v2)

**Authors**: Luca Bortolussi, Ginevra Carbone, Luca Laurenti, Andrea Patane, Guido Sanguinetti, Matthew Wicker

**Abstract**: Vulnerability to adversarial attacks is one of the principal hurdles to the adoption of deep learning in safety-critical applications. Despite significant efforts, both practical and theoretical, training deep learning models robust to adversarial attacks is still an open problem. In this paper, we analyse the geometry of adversarial attacks in the large-data, overparameterized limit for Bayesian Neural Networks (BNNs). We show that, in the limit, vulnerability to gradient-based attacks arises as a result of degeneracy in the data distribution, i.e., when the data lies on a lower-dimensional submanifold of the ambient space. As a direct consequence, we demonstrate that in this limit BNN posteriors are robust to gradient-based adversarial attacks. Crucially, we prove that the expected gradient of the loss with respect to the BNN posterior distribution is vanishing, even when each neural network sampled from the posterior is vulnerable to gradient-based attacks. Experimental results on the MNIST, Fashion MNIST, and half moons datasets, representing the finite data regime, with BNNs trained with Hamiltonian Monte Carlo and Variational Inference, support this line of arguments, showing that BNNs can display both high accuracy on clean data and robustness to both gradient-based and gradient-free based adversarial attacks.

摘要: 对敌意攻击的脆弱性是在安全关键应用中采用深度学习的主要障碍之一。尽管在实践和理论上都做了大量的努力，但训练对对手攻击稳健的深度学习模型仍然是一个悬而未决的问题。本文分析了贝叶斯神经网络(BNN)在大数据、过参数限制下的攻击几何。我们证明，在极限情况下，由于数据分布的退化，即当数据位于环境空间的低维子流形上时，对基于梯度的攻击的脆弱性出现。作为一个直接的推论，我们证明了在这个极限下，BNN后验网络对基于梯度的敌意攻击是稳健的。重要的是，我们证明了损失相对于BNN后验分布的期望梯度是零的，即使从后验采样的每个神经网络都容易受到基于梯度的攻击。在代表有限数据区的MNIST、Fashion MNIST和半月数据集上的实验结果支持这一论点，BNN采用哈密顿蒙特卡罗和变分推理进行训练，表明BNN在干净数据上具有很高的准确率，并且对基于梯度和基于无梯度的敌意攻击都具有很好的鲁棒性。



## **22. Enhancing Adversarial Robustness via Score-Based Optimization**

通过基于分数的优化增强对手的健壮性 cs.LG

**SubmitDate**: 2023-07-10    [abs](http://arxiv.org/abs/2307.04333v1) [paper-pdf](http://arxiv.org/pdf/2307.04333v1)

**Authors**: Boya Zhang, Weijian Luo, Zhihua Zhang

**Abstract**: Adversarial attacks have the potential to mislead deep neural network classifiers by introducing slight perturbations. Developing algorithms that can mitigate the effects of these attacks is crucial for ensuring the safe use of artificial intelligence. Recent studies have suggested that score-based diffusion models are effective in adversarial defenses. However, existing diffusion-based defenses rely on the sequential simulation of the reversed stochastic differential equations of diffusion models, which are computationally inefficient and yield suboptimal results. In this paper, we introduce a novel adversarial defense scheme named ScoreOpt, which optimizes adversarial samples at test-time, towards original clean data in the direction guided by score-based priors. We conduct comprehensive experiments on multiple datasets, including CIFAR10, CIFAR100 and ImageNet. Our experimental results demonstrate that our approach outperforms existing adversarial defenses in terms of both robustness performance and inference speed.

摘要: 对抗性攻击有可能通过引入轻微的扰动来误导深度神经网络分类器。开发能够缓解这些攻击影响的算法，对于确保人工智能的安全使用至关重要。最近的研究表明，基于分数的扩散模型在对抗防御中是有效的。然而，现有的基于扩散的防御依赖于对扩散模型的逆随机微分方程的顺序模拟，这在计算上效率低下，并且产生次优结果。在本文中，我们提出了一种新的对抗防御方案ScoreOpt，该方案在测试时优化对手样本，在基于分数的先验的指导下，朝着原始干净数据的方向进行优化。我们在包括CIFAR10、CIFAR100和ImageNet在内的多个数据集上进行了全面的实验。实验结果表明，该方法在稳健性和推理速度上均优于现有的对抗性防御方法。



## **23. Probabilistic and Semantic Descriptions of Image Manifolds and Their Applications**

图像流形的概率和语义描述及其应用 cs.CV

24 pages, 17 figures, 1 table

**SubmitDate**: 2023-07-10    [abs](http://arxiv.org/abs/2307.02881v2) [paper-pdf](http://arxiv.org/pdf/2307.02881v2)

**Authors**: Peter Tu, Zhaoyuan Yang, Richard Hartley, Zhiwei Xu, Jing Zhang, Dylan Campbell, Jaskirat Singh, Tianyu Wang

**Abstract**: This paper begins with a description of methods for estimating probability density functions for images that reflects the observation that such data is usually constrained to lie in restricted regions of the high-dimensional image space - not every pattern of pixels is an image. It is common to say that images lie on a lower-dimensional manifold in the high-dimensional space. However, although images may lie on such lower-dimensional manifolds, it is not the case that all points on the manifold have an equal probability of being images. Images are unevenly distributed on the manifold, and our task is to devise ways to model this distribution as a probability distribution. In pursuing this goal, we consider generative models that are popular in AI and computer vision community. For our purposes, generative/probabilistic models should have the properties of 1) sample generation: it should be possible to sample from this distribution according to the modelled density function, and 2) probability computation: given a previously unseen sample from the dataset of interest, one should be able to compute the probability of the sample, at least up to a normalising constant. To this end, we investigate the use of methods such as normalising flow and diffusion models. We then show that such probabilistic descriptions can be used to construct defences against adversarial attacks. In addition to describing the manifold in terms of density, we also consider how semantic interpretations can be used to describe points on the manifold. To this end, we consider an emergent language framework which makes use of variational encoders to produce a disentangled representation of points that reside on a given manifold. Trajectories between points on a manifold can then be described in terms of evolving semantic descriptions.

摘要: 本文首先描述了用于估计图像的概率密度函数的方法，该方法反映了这样的观察，即这种数据通常被限制在高维图像空间的受限区域--并不是每一种像素模式都是图像。人们常说，图像位于高维空间中的低维流形上。然而，尽管图像可能位于这样的低维流形上，但流形上的所有点成为图像的概率并不相等。图像在流形上是不均匀分布的，我们的任务是设计出将这种分布建模为概率分布的方法。在追求这一目标的过程中，我们考虑了人工智能和计算机视觉领域中流行的生成性模型。就我们的目的而言，生成/概率模型应该具有以下特性：1)样本生成：应该能够根据建模的密度函数从该分布中进行样本；以及2)概率计算：给定感兴趣的数据集中以前未见过的样本，应该能够计算该样本的概率，至少达到归一化常数。为此，我们研究了流和扩散模型等方法的使用。然后，我们证明了这种概率描述可以用来构建对抗攻击的防御。除了用密度来描述流形之外，我们还考虑了如何使用语义解释来描述流形上的点。为此，我们考虑了一种新的语言框架，它利用变分编码器来产生驻留在给定流形上的点的无纠缠表示。然后，流形上的点之间的轨迹可以通过不断演变的语义描述来描述。



## **24. Testing Robustness Against Unforeseen Adversaries**

测试针对不可预见的对手的健壮性 cs.LG

Datasets available at  https://github.com/centerforaisafety/adversarial-corruptions

**SubmitDate**: 2023-07-09    [abs](http://arxiv.org/abs/1908.08016v3) [paper-pdf](http://arxiv.org/pdf/1908.08016v3)

**Authors**: Max Kaufmann, Daniel Kang, Yi Sun, Steven Basart, Xuwang Yin, Mantas Mazeika, Akul Arora, Adam Dziedzic, Franziska Boenisch, Tom Brown, Jacob Steinhardt, Dan Hendrycks

**Abstract**: When considering real-world adversarial settings, defenders are unlikely to have access to the full range of deployment-time adversaries during training, and adversaries are likely to use realistic adversarial distortions that will not be limited to small L_p-constrained perturbations. To narrow in on this discrepancy between research and reality we introduce eighteen novel adversarial attacks, which we use to create ImageNet-UA, a new benchmark for evaluating model robustness against a wide range of unforeseen adversaries. We make use of our benchmark to identify a range of defense strategies which can help overcome this generalization gap, finding a rich space of techniques which can improve unforeseen robustness. We hope the greater variety and realism of ImageNet-UA will make it a useful tool for those working on real-world worst-case robustness, enabling development of more robust defenses which can generalize beyond attacks seen during training.

摘要: 在考虑真实世界的对抗性设置时，防守者不太可能在训练期间接触到部署时间的所有对手，并且对手可能会使用现实的对抗性扭曲，这不会局限于L_p约束的小扰动。为了缩小研究和现实之间的差距，我们引入了18种新的对手攻击，我们使用它们来创建ImageNet-UA，这是一个新的基准，用于评估模型对广泛不可预见的对手的健壮性。我们利用我们的基准来确定一系列防御策略，这些策略可以帮助克服这一普遍差距，找到丰富的技术空间，可以提高不可预见的健壮性。我们希望ImageNet-UA的更多多样性和现实性将使其成为那些致力于研究现实世界最坏情况下的健壮性的有用工具，使其能够开发出更健壮的防御，可以概括出训练中看到的攻击之外的攻击。



## **25. GNP Attack: Transferable Adversarial Examples via Gradient Norm Penalty**

GNP攻击：通过梯度范数惩罚的可转移对抗性例子 cs.LG

30th IEEE International Conference on Image Processing (ICIP),  October 2023

**SubmitDate**: 2023-07-09    [abs](http://arxiv.org/abs/2307.04099v1) [paper-pdf](http://arxiv.org/pdf/2307.04099v1)

**Authors**: Tao Wu, Tie Luo, Donald C. Wunsch

**Abstract**: Adversarial examples (AE) with good transferability enable practical black-box attacks on diverse target models, where insider knowledge about the target models is not required. Previous methods often generate AE with no or very limited transferability; that is, they easily overfit to the particular architecture and feature representation of the source, white-box model and the generated AE barely work for target, black-box models. In this paper, we propose a novel approach to enhance AE transferability using Gradient Norm Penalty (GNP). It drives the loss function optimization procedure to converge to a flat region of local optima in the loss landscape. By attacking 11 state-of-the-art (SOTA) deep learning models and 6 advanced defense methods, we empirically show that GNP is very effective in generating AE with high transferability. We also demonstrate that it is very flexible in that it can be easily integrated with other gradient based methods for stronger transfer-based attacks.

摘要: 对抗性例子(AE)具有良好的可转移性，使得对不同目标模型的实用黑盒攻击成为可能，其中不需要关于目标模型的内部知识。以前的方法往往生成没有可移植性或可移植性非常有限的AE，即它们很容易过度适应源、白盒模型的特定结构和特征表示，而生成的AE对目标、黑盒模型几乎不起作用。在本文中，我们提出了一种使用梯度范数惩罚(GNP)来增强声发射可转移性的新方法。它驱动损失函数优化过程收敛到损失前景中局部最优的平坦区域。通过攻击11种最新的SOTA深度学习模型和6种先进的防御方法，我们的经验表明GNP在生成具有高可转移性的AE方面是非常有效的。我们还证明了它是非常灵活的，因为它可以很容易地与其他基于梯度的方法相集成，以实现更强的基于传输的攻击。



## **26. Random Position Adversarial Patch for Vision Transformers**

视觉变形金刚的随机位置对抗性补丁 cs.CV

**SubmitDate**: 2023-07-09    [abs](http://arxiv.org/abs/2307.04066v1) [paper-pdf](http://arxiv.org/pdf/2307.04066v1)

**Authors**: Mingzhen Shao

**Abstract**: Previous studies have shown the vulnerability of vision transformers to adversarial patches, but these studies all rely on a critical assumption: the attack patches must be perfectly aligned with the patches used for linear projection in vision transformers. Due to this stringent requirement, deploying adversarial patches for vision transformers in the physical world becomes impractical, unlike their effectiveness on CNNs. This paper proposes a novel method for generating an adversarial patch (G-Patch) that overcomes the alignment constraint, allowing the patch to launch a targeted attack at any position within the field of view. Specifically, instead of directly optimizing the patch using gradients, we employ a GAN-like structure to generate the adversarial patch. Our experiments show the effectiveness of the adversarial patch in achieving universal attacks on vision transformers, both in digital and physical-world scenarios. Additionally, further analysis reveals that the generated adversarial patch exhibits robustness to brightness restriction, color transfer, and random noise. Real-world attack experiments validate the effectiveness of the G-Patch to launch robust attacks even under some very challenging conditions.

摘要: 以前的研究表明视觉转换器对对抗性补丁的脆弱性，但这些研究都依赖于一个关键的假设：攻击补丁必须与视觉转换器中用于线性投影的补丁完全对齐。由于这一严格的要求，在物理世界中为视觉转换器部署对抗性补丁变得不切实际，不像它们在CNN上的有效性。提出了一种新的生成敌意补丁(G-Patch)的方法，该方法克服了对齐限制，允许该补丁在视场内的任何位置发起有针对性的攻击。具体地说，我们没有直接使用梯度来优化补丁，而是使用了一种类似GAN的结构来生成对抗性补丁。我们的实验表明，对抗性补丁在实现对视觉转换器的通用攻击方面是有效的，无论是在数字场景还是在物理世界场景中。此外，进一步的分析表明，生成的恶意补丁对亮度限制、颜色传递和随机噪声具有健壮性。真实世界的攻击实验验证了G-Patch的有效性，即使在一些非常具有挑战性的条件下，G-Patch也能发起强大的攻击。



## **27. Robust Ranking Explanations**

稳健的排名解释 cs.LG

Accepted to IMLH (Interpretable ML in Healthcare) workshop at ICML  2023. arXiv admin note: substantial text overlap with arXiv:2212.14106

**SubmitDate**: 2023-07-08    [abs](http://arxiv.org/abs/2307.04024v1) [paper-pdf](http://arxiv.org/pdf/2307.04024v1)

**Authors**: Chao Chen, Chenghua Guo, Guixiang Ma, Ming Zeng, Xi Zhang, Sihong Xie

**Abstract**: Robust explanations of machine learning models are critical to establish human trust in the models. Due to limited cognition capability, most humans can only interpret the top few salient features. It is critical to make top salient features robust to adversarial attacks, especially those against the more vulnerable gradient-based explanations. Existing defense measures robustness using $\ell_p$-norms, which have weaker protection power. We define explanation thickness for measuring salient features ranking stability, and derive tractable surrogate bounds of the thickness to design the \textit{R2ET} algorithm to efficiently maximize the thickness and anchor top salient features. Theoretically, we prove a connection between R2ET and adversarial training. Experiments with a wide spectrum of network architectures and data modalities, including brain networks, demonstrate that R2ET attains higher explanation robustness under stealthy attacks while retaining accuracy.

摘要: 机器学习模型的可靠解释对于建立人类对模型的信任至关重要。由于认知能力有限，大多数人只能解释最突出的几个特征。关键是使最显著的特征对对抗性攻击具有健壮性，特别是针对更脆弱的基于梯度的解释的攻击。现有的防御使用$\ell_p$-范数来衡量稳健性，这些范数的保护能力较弱。我们定义了用于度量显著特征排序稳定性的解释厚度，并推导了该厚度的易于处理的替代界，以设计有效最大化厚度和锚定顶部显著特征的文本{R2ET}算法。从理论上讲，我们证明了R2ET和对抗性训练之间的联系。对包括脑网络在内的各种网络架构和数据模式的实验表明，R2ET在保持准确性的同时，在隐蔽攻击下获得了更高的解释稳健性。



## **28. Provable Robust Saliency-based Explanations**

基于显著程度的可证明的稳健解释 cs.LG

**SubmitDate**: 2023-07-08    [abs](http://arxiv.org/abs/2212.14106v3) [paper-pdf](http://arxiv.org/pdf/2212.14106v3)

**Authors**: Chao Chen, Chenghua Guo, Guixiang Ma, Ming Zeng, Xi Zhang, Sihong Xie

**Abstract**: Robust explanations of machine learning models are critical to establishing human trust in the models. The top-$k$ intersection is widely used to evaluate the robustness of explanations. However, most existing attacking and defense strategies are based on $\ell_p$ norms, thus creating a mismatch between the evaluation and optimization objectives. To this end, we define explanation thickness for measuring top-$k$ salient features ranking stability, and design the \textit{R2ET} algorithm based on a novel tractable surrogate to maximize the thickness and stabilize the top salient features efficiently. Theoretically, we prove a connection between R2ET and adversarial training; using a novel multi-objective optimization formulation and a generalization error bound, we further prove that the surrogate objective can improve both the numerical and statistical stability of the explanations. Experiments with a wide spectrum of network architectures and data modalities demonstrate that R2ET attains higher explanation robustness under stealthy attacks while retaining model accuracy.

摘要: 机器学习模型的可靠解释对于建立人类对模型的信任至关重要。Top-$k$交集被广泛用于评估解释的健壮性。然而，现有的大多数攻防策略都是基于$\ell_p$规范的，从而造成了评估和优化目标之间的不匹配。为此，我们定义了用于度量前k个显著特征排序稳定性的解释厚度，并设计了基于一种新的易处理代理的文本{R2ET}算法，以最大化厚度并有效地稳定顶部显著特征。在理论上，我们证明了R2ET和对抗性训练之间的联系；利用一个新的多目标优化公式和推广误差界，我们进一步证明了替代目标可以提高解释的数值稳定性和统计稳定性。在广泛的网络架构和数据模式下的实验表明，R2ET在保持模型准确性的同时，在隐蔽攻击下获得了更高的解释稳健性。



## **29. Adversarial Self-Attack Defense and Spatial-Temporal Relation Mining for Visible-Infrared Video Person Re-Identification**

基于对抗性自攻击防御和时空关系挖掘的可见光-红外视频人员再识别 cs.CV

11 pages,8 figures

**SubmitDate**: 2023-07-08    [abs](http://arxiv.org/abs/2307.03903v1) [paper-pdf](http://arxiv.org/pdf/2307.03903v1)

**Authors**: Huafeng Li, Le Xu, Yafei Zhang, Dapeng Tao, Zhengtao Yu

**Abstract**: In visible-infrared video person re-identification (re-ID), extracting features not affected by complex scenes (such as modality, camera views, pedestrian pose, background, etc.) changes, and mining and utilizing motion information are the keys to solving cross-modal pedestrian identity matching. To this end, the paper proposes a new visible-infrared video person re-ID method from a novel perspective, i.e., adversarial self-attack defense and spatial-temporal relation mining. In this work, the changes of views, posture, background and modal discrepancy are considered as the main factors that cause the perturbations of person identity features. Such interference information contained in the training samples is used as an adversarial perturbation. It performs adversarial attacks on the re-ID model during the training to make the model more robust to these unfavorable factors. The attack from the adversarial perturbation is introduced by activating the interference information contained in the input samples without generating adversarial samples, and it can be thus called adversarial self-attack. This design allows adversarial attack and defense to be integrated into one framework. This paper further proposes a spatial-temporal information-guided feature representation network to use the information in video sequences. The network cannot only extract the information contained in the video-frame sequences but also use the relation of the local information in space to guide the network to extract more robust features. The proposed method exhibits compelling performance on large-scale cross-modality video datasets. The source code of the proposed method will be released at https://github.com/lhf12278/xxx.

摘要: 在可见光-红外视频人重新识别(Re-ID)中，提取不受复杂场景(如通道、摄像机视角、行人姿势、背景等)影响的特征。变化，以及运动信息的挖掘和利用是解决跨模式行人身份匹配的关键。为此，本文从一个新的角度提出了一种新的可见光-红外视频人身份识别方法，即对抗性自攻击防御和时空关系挖掘。在本工作中，视角、姿态、背景和模式差异的变化被认为是导致身份特征扰动的主要因素。包含在训练样本中的这种干扰信息被用作对抗性扰动。在训练过程中对Re-ID模型进行对抗性攻击，使模型对这些不利因素具有更强的鲁棒性。来自对抗性扰动的攻击是通过激活输入样本中包含的干扰信息而不产生对抗性样本来引入的，因此可以称为对抗性自攻击。这种设计允许将对抗性攻击和防御集成到一个框架中。本文进一步提出了一种时空信息制导的特征表示网络来利用视频序列中的信息。该网络不仅可以提取视频帧序列中包含的信息，而且可以利用局部信息在空间上的关系来指导网络提取更鲁棒的特征。该方法在大规模跨通道视频数据集上表现出了令人信服的性能。建议的方法的源代码将在https://github.com/lhf12278/xxx.上发布



## **30. On Pseudolinear Codes for Correcting Adversarial Errors**

关于用于纠错的伪线性码 cs.IT

**SubmitDate**: 2023-07-07    [abs](http://arxiv.org/abs/2307.05528v1) [paper-pdf](http://arxiv.org/pdf/2307.05528v1)

**Authors**: Eric Ruzomberka, Homa Nikbakht, Christopher G. Brinton, H. Vincent Poor

**Abstract**: We consider error-correction coding schemes for adversarial wiretap channels (AWTCs) in which the channel can a) read a fraction of the codeword bits up to a bound $r$ and b) flip a fraction of the bits up to a bound $p$. The channel can freely choose the locations of the bit reads and bit flips via a process with unbounded computational power. Codes for the AWTC are of broad interest in the area of information security, as they can provide data resiliency in settings where an attacker has limited access to a storage or transmission medium.   We investigate a family of non-linear codes known as pseudolinear codes, which were first proposed by Guruswami and Indyk (FOCS 2001) for constructing list-decodable codes independent of the AWTC setting. Unlike general non-linear codes, pseudolinear codes admit efficient encoders and have succinct representations. We focus on unique decoding and show that random pseudolinear codes can achieve rates up to the binary symmetric channel (BSC) capacity $1-H_2(p)$ for any $p,r$ in the less noisy region: $p<1/2$ and $r<1-H_2(p)$ where $H_2(\cdot)$ is the binary entropy function. Thus, pseudolinear codes are the first known optimal-rate binary code family for the less noisy AWTC that admit efficient encoders. The above result can be viewed as a derandomization result of random general codes in the AWTC setting, which in turn opens new avenues for applying derandomization techniques to randomized constructions of AWTC codes. Our proof applies a novel concentration inequality for sums of random variables with limited independence which may be of interest as an analysis tool more generally.

摘要: 我们考虑用于对抗窃听信道(AWTC)的纠错编码方案，其中信道可以a)读取直到界限$r$的码字比特的一小部分，并且b)翻转比特的一部分直到界限$p$。通道可以通过具有无限计算能力的过程自由地选择比特读取和比特翻转的位置。AWTC的代码在信息安全领域具有广泛的意义，因为它们可以在攻击者对存储或传输介质的访问权限有限的情况下提供数据弹性。我们研究了由Guruswami和Indyk(FOCS 2001)首次提出的一类被称为伪线性码的非线性码，用于构造与AWTC设置无关的列表可译码。与一般的非线性码不同，伪线性码具有高效的编码者和简洁的表示形式。研究了随机伪线性码的唯一译码问题，证明了在噪声较小的区域：$p<1/2$和$r<1-H_2(P)$时，随机伪线性码可以达到二进制对称信道容量$1-H_2(P)$，其中$H_2(\CDOT)$是二进制熵函数。因此，伪线性码是第一个已知的用于噪声较低的AWTC的最优速率二进制码系列，其允许高效编码器。上述结果可以被视为AWTC设置中的随机一般码的去随机化结果，这进而为将去随机化技术应用于AWTC码的随机化构造开辟了新的途径。我们的证明适用于具有有限独立性的随机变量和的一个新的集中不等式，作为一种更广泛的分析工具可能是有意义的。



## **31. A Theoretical Perspective on Subnetwork Contributions to Adversarial Robustness**

子网络对对抗健壮性贡献的理论视角 cs.LG

3 figures, 3 tables, 17 pages, has appendices

**SubmitDate**: 2023-07-07    [abs](http://arxiv.org/abs/2307.03803v1) [paper-pdf](http://arxiv.org/pdf/2307.03803v1)

**Authors**: Jovon Craig, Josh Andle, Theodore S. Nowak, Salimeh Yasaei Sekeh

**Abstract**: The robustness of deep neural networks (DNNs) against adversarial attacks has been studied extensively in hopes of both better understanding how deep learning models converge and in order to ensure the security of these models in safety-critical applications. Adversarial training is one approach to strengthening DNNs against adversarial attacks, and has been shown to offer a means for doing so at the cost of applying computationally expensive training methods to the entire model. To better understand these attacks and facilitate more efficient adversarial training, in this paper we develop a novel theoretical framework that investigates how the adversarial robustness of a subnetwork contributes to the robustness of the entire network. To do so we first introduce the concept of semirobustness, which is a measure of the adversarial robustness of a subnetwork. Building on this concept, we then provide a theoretical analysis to show that if a subnetwork is semirobust and there is a sufficient dependency between it and each subsequent layer in the network, then the remaining layers are also guaranteed to be robust. We validate these findings empirically across multiple DNN architectures, datasets, and adversarial attacks. Experiments show the ability of a robust subnetwork to promote full-network robustness, and investigate the layer-wise dependencies required for this full-network robustness to be achieved.

摘要: 深度神经网络(DNN)对敌意攻击的稳健性已被广泛研究，以期更好地理解深度学习模型是如何收敛的，并确保这些模型在安全关键应用中的安全性。对抗性训练是加强DNN抵御对抗性攻击的一种方法，已被证明提供了一种这样做的手段，代价是将计算代价高昂的训练方法应用于整个模型。为了更好地理解这些攻击，促进更有效的对抗训练，本文提出了一个新的理论框架，研究了子网络的对抗健壮性如何对整个网络的健壮性做出贡献。为此，我们首先引入半健壮性的概念，它是衡量一个子网络的对抗健壮性的一种度量。在这个概念的基础上，我们给出了一个理论分析，证明了如果一个子网络是半鲁棒的，并且它与网络中的每个后续层之间有足够的依赖关系，那么其余的层也保证是健壮的。我们在多个DNN架构、数据集和对抗性攻击中对这些发现进行了经验验证。实验表明，健壮子网络能够促进全网络健壮性，并研究实现这种全网络健壮性所需的分层依赖关系。



## **32. When and How to Fool Explainable Models (and Humans) with Adversarial Examples**

什么时候以及如何用对抗性的例子愚弄可解释的模型(和人类) cs.LG

Updated version. 43 pages, 9 figures, 4 tables

**SubmitDate**: 2023-07-07    [abs](http://arxiv.org/abs/2107.01943v2) [paper-pdf](http://arxiv.org/pdf/2107.01943v2)

**Authors**: Jon Vadillo, Roberto Santana, Jose A. Lozano

**Abstract**: Reliable deployment of machine learning models such as neural networks continues to be challenging due to several limitations. Some of the main shortcomings are the lack of interpretability and the lack of robustness against adversarial examples or out-of-distribution inputs. In this exploratory review, we explore the possibilities and limits of adversarial attacks for explainable machine learning models. First, we extend the notion of adversarial examples to fit in explainable machine learning scenarios, in which the inputs, the output classifications and the explanations of the model's decisions are assessed by humans. Next, we propose a comprehensive framework to study whether (and how) adversarial examples can be generated for explainable models under human assessment, introducing and illustrating novel attack paradigms. In particular, our framework considers a wide range of relevant yet often ignored factors such as the type of problem, the user expertise or the objective of the explanations, in order to identify the attack strategies that should be adopted in each scenario to successfully deceive the model (and the human). The intention of these contributions is to serve as a basis for a more rigorous and realistic study of adversarial examples in the field of explainable machine learning.

摘要: 由于几个限制，机器学习模型(如神经网络)的可靠部署仍然具有挑战性。其中一些主要缺点是缺乏可解释性，对敌意例子或分配外的投入缺乏稳健性。在这篇探索性综述中，我们探索了针对可解释机器学习模型的对抗性攻击的可能性和局限性。首先，我们将对抗性例子的概念扩展到适合可解释的机器学习场景，在这种场景中，模型的输入、输出分类和解释都是由人类评估的。接下来，我们提出了一个全面的框架来研究是否(以及如何)在人类评估下可以为可解释的模型生成对抗性实例，引入并说明了新的攻击范例。特别是，我们的框架考虑了广泛的相关但经常被忽略的因素，如问题类型、用户专业知识或解释的目标，以便确定在每个场景中应该采用的攻击策略，以成功欺骗模型(和人)。这些贡献的目的是作为对可解释机器学习领域中的对抗性例子进行更严格和现实的研究的基础。



## **33. Enhancing Adversarial Training via Reweighting Optimization Trajectory**

通过重新加权优化轨迹加强对抗性训练 cs.LG

Accepted by ECML 2023

**SubmitDate**: 2023-07-07    [abs](http://arxiv.org/abs/2306.14275v3) [paper-pdf](http://arxiv.org/pdf/2306.14275v3)

**Authors**: Tianjin Huang, Shiwei Liu, Tianlong Chen, Meng Fang, Li Shen, Vlaod Menkovski, Lu Yin, Yulong Pei, Mykola Pechenizkiy

**Abstract**: Despite the fact that adversarial training has become the de facto method for improving the robustness of deep neural networks, it is well-known that vanilla adversarial training suffers from daunting robust overfitting, resulting in unsatisfactory robust generalization. A number of approaches have been proposed to address these drawbacks such as extra regularization, adversarial weights perturbation, and training with more data over the last few years. However, the robust generalization improvement is yet far from satisfactory. In this paper, we approach this challenge with a brand new perspective -- refining historical optimization trajectories. We propose a new method named \textbf{Weighted Optimization Trajectories (WOT)} that leverages the optimization trajectories of adversarial training in time. We have conducted extensive experiments to demonstrate the effectiveness of WOT under various state-of-the-art adversarial attacks. Our results show that WOT integrates seamlessly with the existing adversarial training methods and consistently overcomes the robust overfitting issue, resulting in better adversarial robustness. For example, WOT boosts the robust accuracy of AT-PGD under AA-$L_{\infty}$ attack by 1.53\% $\sim$ 6.11\% and meanwhile increases the clean accuracy by 0.55\%$\sim$5.47\% across SVHN, CIFAR-10, CIFAR-100, and Tiny-ImageNet datasets.

摘要: 尽管对抗性训练已经成为提高深度神经网络鲁棒性的事实上的方法，但众所周知，对抗性训练存在令人望而生畏的健壮性过拟合问题，导致不能令人满意的健壮泛化。在过去的几年里，已经提出了一些方法来解决这些缺点，例如额外的正则化、对抗性权重扰动和使用更多数据进行训练。然而，健壮的泛化改进还远远不能令人满意。在本文中，我们以一种全新的视角来应对这一挑战--提炼历史优化轨迹。我们提出了一种新的方法我们已经进行了大量的实验，以证明WOT在各种最先进的对抗性攻击下的有效性。实验结果表明，WOT算法与现有的对抗性训练方法无缝结合，始终克服了健壮性超调的问题，具有更好的对抗性。例如，WOT将AT-PGD在AA-L攻击下的稳健准确率提高了1.53$\sim$6.11\%，同时在SVHN、CIFAR-10、CIFAR-100和微型ImageNet数据集中将CLEAN准确率提高了0.55$\sim$5.47\%。



## **34. Evaluating Similitude and Robustness of Deep Image Denoising Models via Adversarial Attack**

对抗性攻击下深度图像去噪模型的相似性和稳健性评价 cs.CV

**SubmitDate**: 2023-07-07    [abs](http://arxiv.org/abs/2306.16050v2) [paper-pdf](http://arxiv.org/pdf/2306.16050v2)

**Authors**: Jie Ning, Jiebao Sun, Yao Li, Zhichang Guo, Wangmeng Zuo

**Abstract**: Deep neural networks (DNNs) have shown superior performance comparing to traditional image denoising algorithms. However, DNNs are inevitably vulnerable while facing adversarial attacks. In this paper, we propose an adversarial attack method named denoising-PGD which can successfully attack all the current deep denoising models while keep the noise distribution almost unchanged. We surprisingly find that the current mainstream non-blind denoising models (DnCNN, FFDNet, ECNDNet, BRDNet), blind denoising models (DnCNN-B, Noise2Noise, RDDCNN-B, FAN), plug-and-play (DPIR, CurvPnP) and unfolding denoising models (DeamNet) almost share the same adversarial sample set on both grayscale and color images, respectively. Shared adversarial sample set indicates that all these models are similar in term of local behaviors at the neighborhood of all the test samples. Thus, we further propose an indicator to measure the local similarity of models, called robustness similitude. Non-blind denoising models are found to have high robustness similitude across each other, while hybrid-driven models are also found to have high robustness similitude with pure data-driven non-blind denoising models. According to our robustness assessment, data-driven non-blind denoising models are the most robust. We use adversarial training to complement the vulnerability to adversarial attacks. Moreover, the model-driven image denoising BM3D shows resistance on adversarial attacks.

摘要: 与传统的图像去噪算法相比，深度神经网络(DNN)表现出了更好的性能。然而，DNN在面临敌意攻击时不可避免地容易受到攻击。在本文中，我们提出了一种对抗性攻击方法-去噪-PGD，它可以在保持噪声分布几乎不变的情况下成功地攻击所有现有的深度去噪模型。我们惊奇地发现，当前主流的非盲去噪模型(DnCNN、FFDNet、ECNDNet、BRDNet)、盲去噪模型(DnCNN-B、Noise2Noise、RDDCNN-B、FAN)、即插即用模型(DPIR、CurvPnP)和展开去噪模型(DeamNet)分别在灰度和彩色图像上几乎共享相同的对抗性样本集。共享对抗性样本集表明，所有这些模型在所有测试样本的邻域内的局部行为是相似的。因此，我们进一步提出了一个度量模型局部相似性的指标，称为稳健性相似度。非盲去噪模型之间具有较高的鲁棒性相似性，而混合驱动模型与纯数据驱动的非盲去噪模型也具有较高的鲁棒性相似性。根据我们的稳健性评估，数据驱动的非盲去噪模型是最健壮的。我们使用对抗性训练来弥补对对抗性攻击的脆弱性。此外，模型驱动的图像去噪算法BM3D表现出了对敌意攻击的抵抗能力。



## **35. A Vulnerability of Attribution Methods Using Pre-Softmax Scores**

使用Pre-Softmax分数的归因方法的漏洞 cs.LG

7 pages, 5 figures,

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.03305v1) [paper-pdf](http://arxiv.org/pdf/2307.03305v1)

**Authors**: Miguel Lerma, Mirtha Lucas

**Abstract**: We discuss a vulnerability involving a category of attribution methods used to provide explanations for the outputs of convolutional neural networks working as classifiers. It is known that this type of networks are vulnerable to adversarial attacks, in which imperceptible perturbations of the input may alter the outputs of the model. In contrast, here we focus on effects that small modifications in the model may cause on the attribution method without altering the model outputs.

摘要: 我们讨论了一个漏洞，涉及一类属性方法，用于解释用作分类器的卷积神经网络的输出。众所周知，这种类型的网络容易受到敌意攻击，在这种攻击中，输入的不知不觉的扰动可能会改变模型的输出。相反，这里我们关注的是在不改变模型输出的情况下，模型中的微小修改可能会对归因方法造成的影响。



## **36. Quantum Solutions to the Privacy vs. Utility Tradeoff**

隐私与效用权衡的量子解 quant-ph

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.03118v1) [paper-pdf](http://arxiv.org/pdf/2307.03118v1)

**Authors**: Sagnik Chatterjee, Vyacheslav Kungurtsev

**Abstract**: In this work, we propose a novel architecture (and several variants thereof) based on quantum cryptographic primitives with provable privacy and security guarantees regarding membership inference attacks on generative models. Our architecture can be used on top of any existing classical or quantum generative models. We argue that the use of quantum gates associated with unitary operators provides inherent advantages compared to standard Differential Privacy based techniques for establishing guaranteed security from all polynomial-time adversaries.

摘要: 在这项工作中，我们提出了一个新的体系结构(及其几个变种)，基于量子密码原语，具有可证明的私密性和对生成模型的成员推理攻击的安全性保证。我们的体系结构可以在任何现有的经典或量子生成模型上使用。我们认为，与基于标准差分隐私的技术相比，使用与酉运算符相关的量子门提供了固有的优势，以建立针对所有多项式时间对手的保证安全。



## **37. On Distribution-Preserving Mitigation Strategies for Communication under Cognitive Adversaries**

认知对手环境下通信的分布保持缓解策略研究 cs.IT

Presented at IEEE ISIT 2023

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.03105v1) [paper-pdf](http://arxiv.org/pdf/2307.03105v1)

**Authors**: Soumita Hazra, J. Harshan

**Abstract**: In wireless security, cognitive adversaries are known to inject jamming energy on the victim's frequency band and monitor the same band for countermeasures thereby trapping the victim. Under the class of cognitive adversaries, we propose a new threat model wherein the adversary, upon executing the jamming attack, measures the long-term statistic of Kullback-Leibler Divergence (KLD) between its observations over each of the network frequencies before and after the jamming attack. To mitigate this adversary, we propose a new cooperative strategy wherein the victim takes the assistance for a helper node in the network to reliably communicate its message to the destination. The underlying idea is to appropriately split their energy and time resources such that their messages are reliably communicated without disturbing the statistical distribution of the samples in the network. We present rigorous analyses on the reliability and the covertness metrics at the destination and the adversary, respectively, and then synthesize tractable algorithms to obtain near-optimal division of resources between the victim and the helper. Finally, we show that the obtained near-optimal division of energy facilitates in deceiving the adversary with a KLD estimator.

摘要: 在无线安全中，已知认知对手在受害者的频段上注入干扰能量，并监控同一频段以采取对策，从而诱捕受害者。在认知对手的情况下，我们提出了一个新的威胁模型，在该模型中，对手在实施干扰攻击时，测量干扰攻击前后其在每个网络频率上的观测之间的Kullback-Leibler散度(KLD)的长期统计量。为了缓解这种敌意，我们提出了一种新的合作策略，在该策略中，受害者接受网络中帮助节点的帮助，以可靠地将其消息传递到目的地。其基本思想是适当地分割它们的能量和时间资源，以便在不干扰网络中样本的统计分布的情况下可靠地传递它们的消息。我们分别对目标和对手的可靠性和隐蔽性度量进行了严格的分析，然后综合易处理的算法来获得受害者和帮助者之间的近最优资源分配。最后，我们证明了所得到的接近最优的能量分配便于利用KLD估计来欺骗对手。



## **38. NatLogAttack: A Framework for Attacking Natural Language Inference Models with Natural Logic**

NatLogAttack：一个用自然逻辑攻击自然语言推理模型的框架 cs.CL

Published as a conference paper at ACL 2023

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.02849v1) [paper-pdf](http://arxiv.org/pdf/2307.02849v1)

**Authors**: Zi'ou Zheng, Xiaodan Zhu

**Abstract**: Reasoning has been a central topic in artificial intelligence from the beginning. The recent progress made on distributed representation and neural networks continues to improve the state-of-the-art performance of natural language inference. However, it remains an open question whether the models perform real reasoning to reach their conclusions or rely on spurious correlations. Adversarial attacks have proven to be an important tool to help evaluate the Achilles' heel of the victim models. In this study, we explore the fundamental problem of developing attack models based on logic formalism. We propose NatLogAttack to perform systematic attacks centring around natural logic, a classical logic formalism that is traceable back to Aristotle's syllogism and has been closely developed for natural language inference. The proposed framework renders both label-preserving and label-flipping attacks. We show that compared to the existing attack models, NatLogAttack generates better adversarial examples with fewer visits to the victim models. The victim models are found to be more vulnerable under the label-flipping setting. NatLogAttack provides a tool to probe the existing and future NLI models' capacity from a key viewpoint and we hope more logic-based attacks will be further explored for understanding the desired property of reasoning.

摘要: 从一开始，推理就是人工智能的中心话题。最近在分布式表示和神经网络方面取得的进展继续提高了自然语言推理的最新性能。然而，这些模型是进行真正的推理来得出结论，还是依赖于虚假的相关性，这仍然是一个悬而未决的问题。对抗性攻击已被证明是帮助评估受害者模型的致命弱点的重要工具。在本研究中，我们探讨了建立基于逻辑形式主义的攻击模型的基本问题。我们建议NatLogAttack以自然逻辑为中心执行系统攻击，自然逻辑是一种经典的逻辑形式主义，可以追溯到亚里士多德的三段论，并为自然语言推理而密切发展。该框架同时提供了标签保留攻击和标签翻转攻击。结果表明，与已有的攻击模型相比，NatLogAttack能够以较少的访问受害者模型生成更好的对抗性实例。受害者模特被发现在标签翻转的设置下更容易受到攻击。NatLogAttack提供了一个工具，可以从一个关键的角度来探索现有和未来的NLI模型的能力，我们希望进一步探索更多基于逻辑的攻击，以理解所需的推理属性。



## **39. Sampling-based Fast Gradient Rescaling Method for Highly Transferable Adversarial Attacks**

一种基于采样的高可转移对抗性攻击快速梯度重缩放方法 cs.CV

10 pages, 6 figures, 7 tables. arXiv admin note: substantial text  overlap with arXiv:2204.02887

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.02828v1) [paper-pdf](http://arxiv.org/pdf/2307.02828v1)

**Authors**: Xu Han, Anmin Liu, Chenxuan Yao, Yanbo Fan, Kun He

**Abstract**: Deep neural networks are known to be vulnerable to adversarial examples crafted by adding human-imperceptible perturbations to the benign input. After achieving nearly 100% attack success rates in white-box setting, more focus is shifted to black-box attacks, of which the transferability of adversarial examples has gained significant attention. In either case, the common gradient-based methods generally use the sign function to generate perturbations on the gradient update, that offers a roughly correct direction and has gained great success. But little work pays attention to its possible limitation. In this work, we observe that the deviation between the original gradient and the generated noise may lead to inaccurate gradient update estimation and suboptimal solutions for adversarial transferability. To this end, we propose a Sampling-based Fast Gradient Rescaling Method (S-FGRM). Specifically, we use data rescaling to substitute the sign function without extra computational cost. We further propose a Depth First Sampling method to eliminate the fluctuation of rescaling and stabilize the gradient update. Our method could be used in any gradient-based attacks and is extensible to be integrated with various input transformation or ensemble methods to further improve the adversarial transferability. Extensive experiments on the standard ImageNet dataset show that our method could significantly boost the transferability of gradient-based attacks and outperform the state-of-the-art baselines.

摘要: 众所周知，深度神经网络很容易受到敌意例子的攻击，这些例子是通过在良性输入中添加人类无法察觉的扰动来构建的。在白盒环境下达到近100%的攻击成功率后，更多的注意力转移到黑盒攻击上，其中对抗性例子的可转移性得到了显著的关注。在这两种情况下，常用的基于梯度的方法一般都使用符号函数对梯度更新产生扰动，这提供了一个大致正确的方向，并取得了很大的成功。但很少有人注意到它可能存在的局限性。在这项工作中，我们观察到原始梯度和产生的噪声之间的偏差可能导致不准确的梯度更新估计和对抗性可转移性的次最优解。为此，我们提出了一种基于采样的快速梯度重缩放方法(S-FGRM)。具体地说，我们使用数据重缩放来代替符号函数，而不需要额外的计算代价。在此基础上，提出了深度优先的采样方法，消除了重缩放的波动，稳定了梯度的更新。我们的方法可以用于任何基于梯度的攻击，并且可以扩展到与各种输入变换或集成方法相集成，以进一步提高对抗转移的能力。在标准的ImageNet数据集上的大量实验表明，我们的方法可以显著提高基于梯度的攻击的可转移性，并优于最新的基线。



## **40. A Testbed To Study Adversarial Cyber-Attack Strategies in Enterprise Networks**

企业网络中对抗性网络攻击策略研究的试验台 cs.CR

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.02794v1) [paper-pdf](http://arxiv.org/pdf/2307.02794v1)

**Authors**: Ayush Kumar, David K. Yau

**Abstract**: In this work, we propose a testbed environment to capture the attack strategies of an adversary carrying out a cyber-attack on an enterprise network. The testbed contains nodes with known security vulnerabilities which can be exploited by hackers. Participants can be invited to play the role of a hacker (e.g., black-hat, hacktivist) and attack the testbed. The testbed is designed such that there are multiple attack pathways available to hackers. We describe the working of the testbed components and discuss its implementation on a VMware ESXi server. Finally, we subject our testbed implementation to a few well-known cyber-attack strategies, collect data during the process and present our analysis of the data.

摘要: 在这项工作中，我们提出了一个试验台环境来捕获对企业网络进行网络攻击的对手的攻击策略。测试床包含具有已知安全漏洞的节点，黑客可以利用这些漏洞进行攻击。参与者可以被邀请扮演黑客(例如，黑帽、黑客活动家)的角色并攻击试验床。试验台的设计使得黑客可以使用多种攻击路径。我们将介绍试验床组件的工作原理，并讨论其在VMware ESXi服务器上的实施。最后，我们对我们的试验台实施进行了一些众所周知的网络攻击策略，收集了过程中的数据，并给出了我们对数据的分析。



## **41. Chaos Theory and Adversarial Robustness**

混沌理论与对抗稳健性 cs.LG

14 pages, 6 figures

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2210.13235v2) [paper-pdf](http://arxiv.org/pdf/2210.13235v2)

**Authors**: Jonathan S. Kent

**Abstract**: Neural networks, being susceptible to adversarial attacks, should face a strict level of scrutiny before being deployed in critical or adversarial applications. This paper uses ideas from Chaos Theory to explain, analyze, and quantify the degree to which neural networks are susceptible to or robust against adversarial attacks. To this end, we present a new metric, the "susceptibility ratio," given by $\hat \Psi(h, \theta)$, which captures how greatly a model's output will be changed by perturbations to a given input.   Our results show that susceptibility to attack grows significantly with the depth of the model, which has safety implications for the design of neural networks for production environments. We provide experimental evidence of the relationship between $\hat \Psi$ and the post-attack accuracy of classification models, as well as a discussion of its application to tasks lacking hard decision boundaries. We also demonstrate how to quickly and easily approximate the certified robustness radii for extremely large models, which until now has been computationally infeasible to calculate directly.

摘要: 神经网络容易受到对抗性攻击，在部署到关键或对抗性应用程序之前，应该面临严格的审查。本文利用混沌理论的思想来解释、分析和量化神经网络对敌意攻击的敏感程度或稳健程度。为此，我们提出了一个新的度量，由$\HAT\Psi(h，\theta)$给出的“敏感度比”，它捕捉到模型的输出将因给定输入的扰动而发生多大变化。我们的结果表明，随着模型深度的增加，对攻击的敏感性显著增加，这对生产环境下的神经网络设计具有安全意义。我们提供了分类模型的攻击后精度与$HAT\PSI之间关系的实验证据，并讨论了它在缺乏硬决策边界的任务中的应用。我们还演示了如何快速、轻松地近似极大模型的认证稳健性半径，到目前为止，这在计算上是不可行的直接计算。



## **42. GIT: Detecting Uncertainty, Out-Of-Distribution and Adversarial Samples using Gradients and Invariance Transformations**

GIT：使用梯度和不变性变换检测不确定性、非分布和对抗性样本 cs.LG

Accepted at IJCNN 2023

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2307.02672v1) [paper-pdf](http://arxiv.org/pdf/2307.02672v1)

**Authors**: Julia Lust, Alexandru P. Condurache

**Abstract**: Deep neural networks tend to make overconfident predictions and often require additional detectors for misclassifications, particularly for safety-critical applications. Existing detection methods usually only focus on adversarial attacks or out-of-distribution samples as reasons for false predictions. However, generalization errors occur due to diverse reasons often related to poorly learning relevant invariances. We therefore propose GIT, a holistic approach for the detection of generalization errors that combines the usage of gradient information and invariance transformations. The invariance transformations are designed to shift misclassified samples back into the generalization area of the neural network, while the gradient information measures the contradiction between the initial prediction and the corresponding inherent computations of the neural network using the transformed sample. Our experiments demonstrate the superior performance of GIT compared to the state-of-the-art on a variety of network architectures, problem setups and perturbation types.

摘要: 深度神经网络往往会做出过于自信的预测，并且经常需要额外的检测器来进行错误分类，特别是对于安全关键应用。现有的检测方法通常只关注对抗性攻击或分布不正确的样本作为错误预测的原因。然而，泛化错误是由于各种原因造成的，通常与学习不好的相关不变性有关。因此，我们提出了GIT，一种结合使用梯度信息和不变性变换来检测泛化误差的整体方法。不变性变换的目的是将错误分类的样本移回神经网络的泛化区域，而梯度信息则衡量了初始预测与使用变换后的样本进行相应的神经网络固有计算之间的矛盾。我们的实验表明，与最先进的网络架构、问题设置和扰动类型相比，GIT具有更高的性能。



## **43. Jailbroken: How Does LLM Safety Training Fail?**

越狱：LLM安全培训是如何失败的？ cs.LG

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2307.02483v1) [paper-pdf](http://arxiv.org/pdf/2307.02483v1)

**Authors**: Alexander Wei, Nika Haghtalab, Jacob Steinhardt

**Abstract**: Large language models trained for safety and harmlessness remain susceptible to adversarial misuse, as evidenced by the prevalence of "jailbreak" attacks on early releases of ChatGPT that elicit undesired behavior. Going beyond recognition of the issue, we investigate why such attacks succeed and how they can be created. We hypothesize two failure modes of safety training: competing objectives and mismatched generalization. Competing objectives arise when a model's capabilities and safety goals conflict, while mismatched generalization occurs when safety training fails to generalize to a domain for which capabilities exist. We use these failure modes to guide jailbreak design and then evaluate state-of-the-art models, including OpenAI's GPT-4 and Anthropic's Claude v1.3, against both existing and newly designed attacks. We find that vulnerabilities persist despite the extensive red-teaming and safety-training efforts behind these models. Notably, new attacks utilizing our failure modes succeed on every prompt in a collection of unsafe requests from the models' red-teaming evaluation sets and outperform existing ad hoc jailbreaks. Our analysis emphasizes the need for safety-capability parity -- that safety mechanisms should be as sophisticated as the underlying model -- and argues against the idea that scaling alone can resolve these safety failure modes.

摘要: 经过安全和无害培训的大型语言模型仍然容易受到对手滥用的影响，对早期版本的ChatGPT进行“越狱”攻击的盛行就证明了这一点，这引发了不受欢迎的行为。除了认识到这个问题，我们还调查了此类攻击成功的原因以及如何创建这些攻击。我们假设了安全培训的两种失败模式：目标竞争和不匹配的概括。当模型的能力和安全目标冲突时，就会出现相互竞争的目标，而当安全培训未能概括到存在能力的领域时，就会出现不匹配的泛化。我们使用这些失败模式来指导越狱设计，然后评估最先进的模型，包括OpenAI的GPT-4和Anthropic的Claude v1.3，针对现有的和新设计的攻击。我们发现，尽管在这些模型背后进行了广泛的红色团队和安全培训努力，但漏洞仍然存在。值得注意的是，利用我们的失败模式的新攻击在模型的红团队评估集的不安全请求集合中的每一个提示下都会成功，并且表现优于现有的临时越狱。我们的分析强调了安全能力对等的必要性--安全机制应该与基础模型一样复杂--并反对仅靠扩展就能解决这些安全故障模式的想法。



## **44. Defense against Adversarial Cloud Attack on Remote Sensing Salient Object Detection**

遥感显著目标检测中对敌云攻击的防御 cs.CV

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2306.17431v2) [paper-pdf](http://arxiv.org/pdf/2306.17431v2)

**Authors**: Huiming Sun, Lan Fu, Jinlong Li, Qing Guo, Zibo Meng, Tianyun Zhang, Yuewei Lin, Hongkai Yu

**Abstract**: Detecting the salient objects in a remote sensing image has wide applications for the interdisciplinary research. Many existing deep learning methods have been proposed for Salient Object Detection (SOD) in remote sensing images and get remarkable results. However, the recent adversarial attack examples, generated by changing a few pixel values on the original remote sensing image, could result in a collapse for the well-trained deep learning based SOD model. Different with existing methods adding perturbation to original images, we propose to jointly tune adversarial exposure and additive perturbation for attack and constrain image close to cloudy image as Adversarial Cloud. Cloud is natural and common in remote sensing images, however, camouflaging cloud based adversarial attack and defense for remote sensing images are not well studied before. Furthermore, we design DefenseNet as a learn-able pre-processing to the adversarial cloudy images so as to preserve the performance of the deep learning based remote sensing SOD model, without tuning the already deployed deep SOD model. By considering both regular and generalized adversarial examples, the proposed DefenseNet can defend the proposed Adversarial Cloud in white-box setting and other attack methods in black-box setting. Experimental results on a synthesized benchmark from the public remote sensing SOD dataset (EORSSD) show the promising defense against adversarial cloud attacks.

摘要: 遥感图像中显著目标的检测在多学科交叉研究中有着广泛的应用。已有的许多深度学习方法被提出用于遥感图像中的显著目标检测，并取得了显著的效果。然而，最近通过改变原始遥感图像上的几个像素值而生成的对抗性攻击实例，可能会导致基于深度学习的训练有素的SOD模型崩溃。与已有的在原始图像上添加扰动的方法不同，我们提出了联合调整攻击的对抗性曝光和加性扰动，并将接近云图的图像约束为对抗性云。云层是遥感图像中常见的自然现象，但基于云层伪装的遥感图像对抗攻防研究较少。此外，我们将DefenseNet设计为对敌意云图进行可学习的预处理，以保持基于深度学习的遥感SOD模型的性能，而不需要调整已经部署的深度SOD模型。通过考虑常规和广义的对抗性实例，所提出的防御网络可以在白盒环境下防御所提出的对抗性云，并在黑盒环境下防御其他攻击方法。在一个基于公共遥感数据集(EORSSD)的合成基准上的实验结果表明，该方法能够有效地防御敌意云攻击。



## **45. On the Adversarial Robustness of Generative Autoencoders in the Latent Space**

生成式自动编码器在潜在空间中的对抗健壮性 cs.LG

18 pages, 12 figures

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2307.02202v1) [paper-pdf](http://arxiv.org/pdf/2307.02202v1)

**Authors**: Mingfei Lu, Badong Chen

**Abstract**: The generative autoencoders, such as the variational autoencoders or the adversarial autoencoders, have achieved great success in lots of real-world applications, including image generation, and signal communication.   However, little concern has been devoted to their robustness during practical deployment.   Due to the probabilistic latent structure, variational autoencoders (VAEs) may confront problems such as a mismatch between the posterior distribution of the latent and real data manifold, or discontinuity in the posterior distribution of the latent.   This leaves a back door for malicious attackers to collapse VAEs from the latent space, especially in scenarios where the encoder and decoder are used separately, such as communication and compressed sensing.   In this work, we provide the first study on the adversarial robustness of generative autoencoders in the latent space.   Specifically, we empirically demonstrate the latent vulnerability of popular generative autoencoders through attacks in the latent space.   We also evaluate the difference between variational autoencoders and their deterministic variants and observe that the latter performs better in latent robustness.   Meanwhile, we identify a potential trade-off between the adversarial robustness and the degree of the disentanglement of the latent codes.   Additionally, we also verify the feasibility of improvement for the latent robustness of VAEs through adversarial training.   In summary, we suggest concerning the adversarial latent robustness of the generative autoencoders, analyze several robustness-relative issues, and give some insights into a series of key challenges.

摘要: 产生式自动编码器，如变分自动编码器或对抗性自动编码器，已经在图像生成、信号通信等实际应用中取得了巨大的成功。然而，很少有人关注它们在实际部署过程中的健壮性。由于概率潜在结构，变分自动编码器可能会遇到潜在数据流形和真实数据流形的后验分布不匹配或后验分布不连续等问题。这为恶意攻击者从潜在空间崩溃VAE留下了后门，特别是在单独使用编码器和解码器的场景中，例如通信和压缩传感。在这项工作中，我们首次研究了生成式自动编码器在潜在空间中的对抗健壮性。具体地说，我们通过对潜在空间的攻击，经验地证明了流行的生成式自动编码器的潜在脆弱性。我们还评估了变分自动编码器和它们的确定性变体之间的差异，并观察到后者在潜在稳健性方面表现得更好。同时，我们确定了潜在代码的对抗健壮性和解缠程度之间的潜在权衡。此外，我们还验证了通过对抗性训练来提高VAE的潜在健壮性的可行性。综上所述，我们建议关注生成式自动编码器的对抗性潜在健壮性，分析了几个与健壮性相关的问题，并对一系列关键挑战给出了一些见解。



## **46. Boosting Adversarial Transferability via Fusing Logits of Top-1 Decomposed Feature**

融合Top-1分解特征的Logit提高对手的可转移性 cs.CV

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2305.01361v3) [paper-pdf](http://arxiv.org/pdf/2305.01361v3)

**Authors**: Juanjuan Weng, Zhiming Luo, Dazhen Lin, Shaozi Li, Zhun Zhong

**Abstract**: Recent research has shown that Deep Neural Networks (DNNs) are highly vulnerable to adversarial samples, which are highly transferable and can be used to attack other unknown black-box models. To improve the transferability of adversarial samples, several feature-based adversarial attack methods have been proposed to disrupt neuron activation in the middle layers. However, current state-of-the-art feature-based attack methods typically require additional computation costs for estimating the importance of neurons. To address this challenge, we propose a Singular Value Decomposition (SVD)-based feature-level attack method. Our approach is inspired by the discovery that eigenvectors associated with the larger singular values decomposed from the middle layer features exhibit superior generalization and attention properties. Specifically, we conduct the attack by retaining the decomposed Top-1 singular value-associated feature for computing the output logits, which are then combined with the original logits to optimize adversarial examples. Our extensive experimental results verify the effectiveness of our proposed method, which can be easily integrated into various baselines to significantly enhance the transferability of adversarial samples for disturbing normally trained CNNs and advanced defense strategies. The source code of this study is available at https://github.com/WJJLL/SVD-SSA

摘要: 最近的研究表明，深度神经网络非常容易受到敌意样本的攻击，这些样本具有很高的可传递性，可以用来攻击其他未知的黑盒模型。为了提高对抗性样本的可转移性，已经提出了几种基于特征的对抗性攻击方法来破坏中间层神经元的激活。然而，当前最先进的基于特征的攻击方法通常需要额外的计算成本来估计神经元的重要性。为了应对这一挑战，我们提出了一种基于奇异值分解(SVD)的特征级攻击方法。我们的方法是受到这样的发现的启发，即与从中间层特征分解的较大奇异值相关的特征向量具有更好的泛化和注意特性。具体地说，我们通过保留分解后的Top-1奇异值关联特征来计算输出逻辑，然后将其与原始逻辑相结合来优化对抗性实例，从而进行攻击。大量的实验结果验证了该方法的有效性，该方法可以很容易地集成到不同的基线中，显著提高对手样本干扰正常训练的CNN和高级防御策略的可转移性。这项研究的源代码可在https://github.com/WJJLL/SVD-SSA上获得



## **47. Adversarial Attacks on Image Classification Models: FGSM and Patch Attacks and their Impact**

对图像分类模型的敌意攻击：FGSM和Patch攻击及其影响 cs.CV

This is the preprint of the chapter titled "Adversarial Attacks on  Image Classification Models: FGSM and Patch Attacks and their Impact" which  will be published in the volume titled "Information Security and Privacy in  the Digital World - Some Selected Cases", edited by Jaydip Sen. The book will  be published by IntechOpen, London, UK, in 2023. This is not the final  version of the chapter

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2307.02055v1) [paper-pdf](http://arxiv.org/pdf/2307.02055v1)

**Authors**: Jaydip Sen, Subhasis Dasgupta

**Abstract**: This chapter introduces the concept of adversarial attacks on image classification models built on convolutional neural networks (CNN). CNNs are very popular deep-learning models which are used in image classification tasks. However, very powerful and pre-trained CNN models working very accurately on image datasets for image classification tasks may perform disastrously when the networks are under adversarial attacks. In this work, two very well-known adversarial attacks are discussed and their impact on the performance of image classifiers is analyzed. These two adversarial attacks are the fast gradient sign method (FGSM) and adversarial patch attack. These attacks are launched on three powerful pre-trained image classifier architectures, ResNet-34, GoogleNet, and DenseNet-161. The classification accuracy of the models in the absence and presence of the two attacks are computed on images from the publicly accessible ImageNet dataset. The results are analyzed to evaluate the impact of the attacks on the image classification task.

摘要: 本章介绍了基于卷积神经网络(CNN)的图像分类模型的对抗性攻击的概念。CNN是一种非常流行的深度学习模型，用于图像分类任务。然而，非常强大和预先训练的CNN模型在图像数据集上非常准确地工作以执行图像分类任务，当网络受到敌意攻击时，可能会灾难性地执行。在这项工作中，讨论了两个非常著名的对抗性攻击，并分析了它们对图像分类器性能的影响。这两种对抗性攻击是快速梯度符号方法(FGSM)和对抗性补丁攻击。这些攻击是在三个强大的预先训练的图像分类器架构上发起的，ResNet-34、GoogLeNet和DenseNet-161。对来自公众可访问的ImageNet数据集中的图像计算在两种攻击不存在的情况下模型的分类精度。分析结果以评估攻击对图像分类任务的影响。



## **48. Complex Graph Laplacian Regularizer for Inferencing Grid States**

用于网格状态推断的复图拉普拉斯正则化算法 eess.SP

**SubmitDate**: 2023-07-04    [abs](http://arxiv.org/abs/2307.01906v1) [paper-pdf](http://arxiv.org/pdf/2307.01906v1)

**Authors**: Chinthaka Dinesh, Junfei Wang, Gene Cheung, Pirathayini Srikantha

**Abstract**: In order to maintain stable grid operations, system monitoring and control processes require the computation of grid states (e.g. voltage magnitude and angles) at high granularity. It is necessary to infer these grid states from measurements generated by a limited number of sensors like phasor measurement units (PMUs) that can be subjected to delays and losses due to channel artefacts, and/or adversarial attacks (e.g. denial of service, jamming, etc.). We propose a novel graph signal processing (GSP) based algorithm to interpolate states of the entire grid from observations of a small number of grid measurements. It is a two-stage process, where first an underlying Hermitian graph is learnt empirically from existing grid datasets. Then, the graph is used to interpolate missing grid signal samples in linear time. With our proposal, we can effectively reconstruct grid signals with significantly smaller number of observations when compared to existing traditional approaches (e.g. state estimation). In contrast to existing GSP approaches, we do not require knowledge of the underlying grid structure and parameters and are able to guarantee fast spectral optimization. We demonstrate the computational efficacy and accuracy of our proposal via practical studies conducted on the IEEE 118 bus system.

摘要: 为了维持稳定的电网运行，系统监测和控制过程需要计算高粒度的电网状态(如电压幅值和角度)。有必要从有限数量的传感器(如相量测量单元(PMU))生成的测量结果中推断这些网格状态，这些传感器可能由于信道伪影和/或对抗性攻击(例如拒绝服务、干扰等)而受到延迟和损失。我们提出了一种基于图信号处理(GSP)的新算法，该算法根据少量网格测量的观测值来内插整个网格的状态。这是一个分两个阶段的过程，首先从现有的网格数据集中经验地学习潜在的厄米特图。然后，使用该图在线性时间内对缺失的网格信号样本进行内插。与现有的传统方法(例如状态估计)相比，我们的建议可以用明显更少的观测值来有效地重建网格信号。与现有的GSP方法相比，我们不需要底层网格结构和参数的知识，并且能够保证快速的频谱优化。通过在IEEE118节点系统上进行的实际研究，证明了该方法的计算效率和准确性。



## **49. Physically Realizable Natural-Looking Clothing Textures Evade Person Detectors via 3D Modeling**

物理上可实现的自然外观服装纹理通过3D建模躲避人的检测 cs.CV

Accepted by CVPR 2023

**SubmitDate**: 2023-07-04    [abs](http://arxiv.org/abs/2307.01778v1) [paper-pdf](http://arxiv.org/pdf/2307.01778v1)

**Authors**: Zhanhao Hu, Wenda Chu, Xiaopei Zhu, Hui Zhang, Bo Zhang, Xiaolin Hu

**Abstract**: Recent works have proposed to craft adversarial clothes for evading person detectors, while they are either only effective at limited viewing angles or very conspicuous to humans. We aim to craft adversarial texture for clothes based on 3D modeling, an idea that has been used to craft rigid adversarial objects such as a 3D-printed turtle. Unlike rigid objects, humans and clothes are non-rigid, leading to difficulties in physical realization. In order to craft natural-looking adversarial clothes that can evade person detectors at multiple viewing angles, we propose adversarial camouflage textures (AdvCaT) that resemble one kind of the typical textures of daily clothes, camouflage textures. We leverage the Voronoi diagram and Gumbel-softmax trick to parameterize the camouflage textures and optimize the parameters via 3D modeling. Moreover, we propose an efficient augmentation pipeline on 3D meshes combining topologically plausible projection (TopoProj) and Thin Plate Spline (TPS) to narrow the gap between digital and real-world objects. We printed the developed 3D texture pieces on fabric materials and tailored them into T-shirts and trousers. Experiments show high attack success rates of these clothes against multiple detectors.

摘要: 最近的研究提出了为躲避人体探测器而制作对抗服装，而这些服装要么只在有限的视角下有效，要么对人类来说非常显眼。我们的目标是基于3D建模为衣服制作对抗性纹理，这一想法已被用于制作刚性对抗性对象，如3D打印的乌龟。与刚性物体不同，人和衣服是非刚性的，这导致了物理实现的困难。为了制作出看起来自然的、能够在多个视角下躲避人体探测的对抗性服装，我们提出了一种类似于日常服装的典型纹理--伪装纹理的对抗性伪装纹理(AdvCaT)。我们利用Voronoi图和Gumbel-Softmax技巧对伪装纹理进行参数化，并通过3D建模优化参数。此外，我们还提出了一种结合拓扑似然投影(TOPO Proj)和薄板样条线(TPS)的三维网格增强流水线，以缩小数字对象和真实对象之间的差距。我们将开发的3D纹理块打印在面料上，并将它们裁剪成T恤和裤子。实验表明，这些衣服对多个探测器的攻击成功率很高。



## **50. vWitness: Certifying Web Page Interactions with Computer Vision**

VWitness：使用计算机视觉验证网页交互 cs.CR

**SubmitDate**: 2023-07-04    [abs](http://arxiv.org/abs/2007.15805v2) [paper-pdf](http://arxiv.org/pdf/2007.15805v2)

**Authors**: He Shuang, Lianying Zhao, David Lie

**Abstract**: Web servers service client requests, some of which might cause the web server to perform security-sensitive operations (e.g. money transfer, voting). An attacker may thus forge or maliciously manipulate such requests by compromising a web client. Unfortunately, a web server has no way of knowing whether the client from which it receives a request has been compromised or not -- current "best practice" defenses such as user authentication or network encryption cannot aid a server as they all assume web client integrity. To address this shortcoming, we propose vWitness, which "witnesses" the interactions of a user with a web page and certifies whether they match a specification provided by the web server, enabling the web server to know that the web request is user-intended. The main challenge that vWitness overcomes is that even benign clients introduce unpredictable variations in the way they render web pages. vWitness differentiates between these benign variations and malicious manipulation using computer vision, allowing it to certify to the web server that 1) the web page user interface is properly displayed 2) observed user interactions are used to construct the web request. Our vWitness prototype achieves compatibility with modern web pages, is resilient to adversarial example attacks and is accurate and performant -- vWitness achieves 99.97% accuracy and adds 197ms of overhead to the entire interaction session in the average case.

摘要: Web服务器为客户端请求提供服务，其中一些请求可能会导致Web服务器执行安全敏感操作(例如，转账、投票)。因此，攻击者可以通过危害Web客户端来伪造或恶意操纵此类请求。不幸的是，Web服务器无法知道它从其接收请求的客户端是否已被破坏--当前的“最佳实践”防御，如用户身份验证或网络加密，无法帮助服务器，因为它们都假定Web客户端的完整性。为了解决这一缺点，我们提出了vWitness，它“见证”用户与网页的交互，并验证它们是否符合Web服务器提供的规范，使Web服务器知道Web请求是用户预期的。VWitness克服的主要挑战是，即使是良性的客户端，也会在呈现网页的方式上引入不可预测的变化。VWitness使用计算机视觉区分这些良性变化和恶意操纵，允许它向Web服务器证明1)网页用户界面被正确显示2)使用观察到的用户交互来构造Web请求。我们的vWitness原型实现了与现代网页的兼容性，对敌意示例攻击具有弹性，并且是准确和高性能的--vWitness达到99.97%的准确率，在平均情况下，整个交互会话增加了197ms的开销。



