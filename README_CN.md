# Latest Adversarial Attack Papers
**update at 2022-07-08 06:31:21**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Enhancing Adversarial Attacks on Single-Layer NVM Crossbar-Based Neural Networks with Power Consumption Information**

利用功耗信息增强单层NVM Crosbar神经网络的敌意攻击 cs.LG

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2207.02764v1)

**Authors**: Cory Merkel

**Abstracts**: Adversarial attacks on state-of-the-art machine learning models pose a significant threat to the safety and security of mission-critical autonomous systems. This paper considers the additional vulnerability of machine learning models when attackers can measure the power consumption of their underlying hardware platform. In particular, we explore the utility of power consumption information for adversarial attacks on non-volatile memory crossbar-based single-layer neural networks. Our results from experiments with MNIST and CIFAR-10 datasets show that power consumption can reveal important information about the neural network's weight matrix, such as the 1-norm of its columns. That information can be used to infer the sensitivity of the network's loss with respect to different inputs. We also find that surrogate-based black box attacks that utilize crossbar power information can lead to improved attack efficiency.

摘要: 对最先进的机器学习模型的对抗性攻击对任务关键型自主系统的安全构成了严重威胁。本文考虑了当攻击者可以测量其底层硬件平台的功耗时，机器学习模型的附加漏洞。特别是，我们探索了功耗信息在基于非易失性记忆交叉开关的单层神经网络上的对抗性攻击中的效用。我们在MNIST和CIFAR-10数据集上的实验结果表明，功耗可以揭示有关神经网络权重矩阵的重要信息，例如其列的1范数。该信息可用于推断网络损耗对不同输入的敏感性。我们还发现，利用纵横制能量信息的基于代理的黑盒攻击可以提高攻击效率。



## **2. LADDER: Latent Boundary-guided Adversarial Training**

梯子：潜在边界制导的对抗性训练 cs.LG

To appear in Machine Learning

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2206.03717v2)

**Authors**: Xiaowei Zhou, Ivor W. Tsang, Jie Yin

**Abstracts**: Deep Neural Networks (DNNs) have recently achieved great success in many classification tasks. Unfortunately, they are vulnerable to adversarial attacks that generate adversarial examples with a small perturbation to fool DNN models, especially in model sharing scenarios. Adversarial training is proved to be the most effective strategy that injects adversarial examples into model training to improve the robustness of DNN models against adversarial attacks. However, adversarial training based on the existing adversarial examples fails to generalize well to standard, unperturbed test data. To achieve a better trade-off between standard accuracy and adversarial robustness, we propose a novel adversarial training framework called LAtent bounDary-guided aDvErsarial tRaining (LADDER) that adversarially trains DNN models on latent boundary-guided adversarial examples. As opposed to most of the existing methods that generate adversarial examples in the input space, LADDER generates a myriad of high-quality adversarial examples through adding perturbations to latent features. The perturbations are made along the normal of the decision boundary constructed by an SVM with an attention mechanism. We analyze the merits of our generated boundary-guided adversarial examples from a boundary field perspective and visualization view. Extensive experiments and detailed analysis on MNIST, SVHN, CelebA, and CIFAR-10 validate the effectiveness of LADDER in achieving a better trade-off between standard accuracy and adversarial robustness as compared with vanilla DNNs and competitive baselines.

摘要: 近年来，深度神经网络(DNN)在许多分类任务中取得了巨大的成功。不幸的是，它们很容易受到对抗性攻击，生成带有微小扰动的对抗性示例来愚弄DNN模型，特别是在模型共享场景中。对抗性训练被证明是将对抗性样本注入模型训练以提高DNN模型对对抗性攻击的鲁棒性的最有效的策略。然而，基于现有对抗性实例的对抗性训练不能很好地推广到标准的、不受干扰的测试数据。为了在标准准确率和对手健壮性之间实现更好的折衷，我们提出了一种新的对手训练框架，称为潜在边界制导的对抗性训练(LIDA)，该框架针对潜在的边界制导的对抗性实例对DNN模型进行对抗性训练。与大多数现有的在输入空间生成对抗性实例的方法不同，梯形图通过对潜在特征添加扰动来生成大量高质量的对抗性实例。扰动沿具有注意力机制的支持向量机构造的决策边界的法线进行。我们从边界场的角度和可视化的角度分析了我们生成的边界制导的对抗性例子的优点。在MNIST、SVHN、CelebA和CIFAR-10上的大量实验和详细分析验证了梯形算法在标准准确率和对手健壮性之间取得了比普通DNN和竞争基线更好的折衷。



## **3. Adversarial Robustness of Visual Dialog**

可视化对话框的对抗性健壮性 cs.CV

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2207.02639v1)

**Authors**: Lu Yu, Verena Rieser

**Abstracts**: Adversarial robustness evaluates the worst-case performance scenario of a machine learning model to ensure its safety and reliability. This study is the first to investigate the robustness of visually grounded dialog models towards textual attacks. These attacks represent a worst-case scenario where the input question contains a synonym which causes the previously correct model to return a wrong answer. Using this scenario, we first aim to understand how multimodal input components contribute to model robustness. Our results show that models which encode dialog history are more robust, and when launching an attack on history, model prediction becomes more uncertain. This is in contrast to prior work which finds that dialog history is negligible for model performance on this task. We also evaluate how to generate adversarial test examples which successfully fool the model but remain undetected by the user/software designer. We find that the textual, as well as the visual context are important to generate plausible worst-case scenarios.

摘要: 对抗健壮性评估机器学习模型的最差性能场景，以确保其安全性和可靠性。本研究首次研究了基于视觉的对话模型对文本攻击的稳健性。这些攻击代表了最坏的情况，其中输入问题包含同义词，这会导致先前正确的模型返回错误的答案。使用这个场景，我们首先要了解多模式输入组件如何有助于模型的健壮性。我们的结果表明，编码对话历史的模型更健壮，当对历史发起攻击时，模型预测变得更不确定。这与以前的工作形成对比，以前的工作发现对话历史对于此任务的模型性能可以忽略不计。我们还评估了如何生成敌意测试用例，这些测试用例成功地愚弄了模型，但仍然没有被用户/软件设计者发现。我们发现，文本环境和视觉环境对于生成看似合理的最坏情况都很重要。



## **4. Adversarial Mask: Real-World Universal Adversarial Attack on Face Recognition Models**

对抗性面具：对人脸识别模型的真实世界通用对抗性攻击 cs.CV

16

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2111.10759v2)

**Authors**: Alon Zolfi, Shai Avidan, Yuval Elovici, Asaf Shabtai

**Abstracts**: Deep learning-based facial recognition (FR) models have demonstrated state-of-the-art performance in the past few years, even when wearing protective medical face masks became commonplace during the COVID-19 pandemic. Given the outstanding performance of these models, the machine learning research community has shown increasing interest in challenging their robustness. Initially, researchers presented adversarial attacks in the digital domain, and later the attacks were transferred to the physical domain. However, in many cases, attacks in the physical domain are conspicuous, and thus may raise suspicion in real-world environments (e.g., airports). In this paper, we propose Adversarial Mask, a physical universal adversarial perturbation (UAP) against state-of-the-art FR models that is applied on face masks in the form of a carefully crafted pattern. In our experiments, we examined the transferability of our adversarial mask to a wide range of FR model architectures and datasets. In addition, we validated our adversarial mask's effectiveness in real-world experiments (CCTV use case) by printing the adversarial pattern on a fabric face mask. In these experiments, the FR system was only able to identify 3.34% of the participants wearing the mask (compared to a minimum of 83.34% with other evaluated masks). A demo of our experiments can be found at: https://youtu.be/_TXkDO5z11w.

摘要: 基于深度学习的面部识别(FR)模型在过去几年展示了最先进的性能，即使在新冠肺炎大流行期间戴防护性医用口罩变得司空见惯。鉴于这些模型的出色性能，机器学习研究界对挑战它们的稳健性表现出越来越大的兴趣。最初，研究人员在数字领域提出了对抗性攻击，后来攻击被转移到物理领域。然而，在许多情况下，物理域中的攻击很明显，因此可能会在现实环境(例如机场)中引起怀疑。在本文中，我们提出了对抗面具，一种针对最先进的FR模型的物理通用对抗扰动(UAP)，它以精心制作的模式的形式应用于人脸面具上。在我们的实验中，我们检查了我们的对手面具在广泛的FR模型体系结构和数据集上的可转移性。此外，我们在真实世界的实验中(CCTV用例)验证了我们的对抗面具的有效性，通过将对抗图案打印在织物面膜上。在这些实验中，FR系统只能识别3.34%的戴口罩的参与者(相比之下，其他评估的口罩的最低识别率为83.34%)。我们的实验演示可在以下网址找到：https://youtu.be/_TXkDO5z11w.



## **5. FIDO2 With Two Displays-Or How to Protect Security-Critical Web Transactions Against Malware Attacks**

带两个显示屏的FIDO2-或如何保护安全关键型Web交易免受恶意软件攻击 cs.CR

**SubmitDate**: 2022-07-06    [paper-pdf](http://arxiv.org/pdf/2206.13358v3)

**Authors**: Timon Hackenjos, Benedikt Wagner, Julian Herr, Jochen Rill, Marek Wehmer, Niklas Goerke, Ingmar Baumgart

**Abstracts**: With the rise of attacks on online accounts in the past years, more and more services offer two-factor authentication for their users. Having factors out of two of the three categories something you know, something you have and something you are should ensure that an attacker cannot compromise two of them at once. Thus, an adversary should not be able to maliciously interact with one's account. However, this is only true if one considers a weak adversary. In particular, since most current solutions only authenticate a session and not individual transactions, they are noneffective if one's device is infected with malware. For online banking, the banking industry has long since identified the need for authenticating transactions. However, specifications of such authentication schemes are not public and implementation details vary wildly from bank to bank with most still being unable to protect against malware. In this work, we present a generic approach to tackle the problem of malicious account takeovers, even in the presence of malware. To this end, we define a new paradigm to improve two-factor authentication that involves the concepts of one-out-of-two security and transaction authentication. Web authentication schemes following this paradigm can protect security-critical transactions against manipulation, even if one of the factors is completely compromised. Analyzing existing authentication schemes, we find that they do not realize one-out-of-two security. We give a blueprint of how to design secure web authentication schemes in general. Based on this blueprint we propose FIDO2 With Two Displays (FIDO2D), a new web authentication scheme based on the FIDO2 standard and prove its security using Tamarin. We hope that our work inspires a new wave of more secure web authentication schemes, which protect security-critical transactions even against attacks with malware.

摘要: 随着过去几年针对在线账户的攻击事件的增加，越来越多的服务为其用户提供双因素身份验证。拥有三个类别中的两个因素，你知道的，你拥有的和你是的，应该确保攻击者不能同时危害其中的两个。因此，对手不应该能够恶意地与自己的帐户交互。然而，只有当一个人考虑到一个弱小的对手时，这才是正确的。特别是，由于大多数当前的解决方案只对会话进行身份验证，而不是对单个事务进行身份验证，因此如果设备感染了恶意软件，这些解决方案就会无效。对于网上银行，银行业早就认识到了对交易进行身份验证的必要性。然而，此类身份验证方案的规范并未公开，各银行的实施细节也存在很大差异，大多数银行仍无法防范恶意软件。在这项工作中，我们提出了一种通用的方法来解决恶意帐户接管问题，即使在存在恶意软件的情况下也是如此。为此，我们定义了一个新的范例来改进双因素身份验证，它涉及二选一安全和事务身份验证的概念。遵循此范例的Web身份验证方案可以保护安全关键型交易免受操纵，即使其中一个因素完全受损。分析现有的认证方案，发现它们并没有实现二选一的安全性。我们给出了一个总体上如何设计安全的Web认证方案的蓝图。在此基础上，我们提出了一种新的基于FIDO2标准的网络认证方案FIDO2 with Two Display(FIDO2D)，并用Tamarin对其安全性进行了证明。我们希望我们的工作激发出新一波更安全的网络身份验证方案，这些方案甚至可以保护安全关键交易免受恶意软件的攻击。



## **6. Learning to Accelerate Approximate Methods for Solving Integer Programming via Early Fixing**

通过早期确定学习加速求解整数规划的近似方法 cs.DM

16 pages, 11 figures, 6 tables

**SubmitDate**: 2022-07-05    [paper-pdf](http://arxiv.org/pdf/2207.02087v1)

**Authors**: Longkang Li, Baoyuan Wu

**Abstracts**: Integer programming (IP) is an important and challenging problem. Approximate methods have shown promising performance on both effectiveness and efficiency for solving the IP problem. However, we observed that a large fraction of variables solved by some iterative approximate methods fluctuate around their final converged discrete states in very long iterations. Inspired by this observation, we aim to accelerate these approximate methods by early fixing these fluctuated variables to their converged states while not significantly harming the solution accuracy. To this end, we propose an early fixing framework along with the approximate method. We formulate the whole early fixing process as a Markov decision process, and train it using imitation learning. A policy network will evaluate the posterior probability of each free variable concerning its discrete candidate states in each block of iterations. Specifically, we adopt the powerful multi-headed attention mechanism in the policy network. Extensive experiments on our proposed early fixing framework are conducted to three different IP applications: constrained linear programming, MRF energy minimization and sparse adversarial attack. The former one is linear IP problem, while the latter two are quadratic IP problems. We extend the problem scale from regular size to significantly large size. The extensive experiments reveal the competitiveness of our early fixing framework: the runtime speeds up significantly, while the solution quality does not degrade much, even in some cases it is available to obtain better solutions. Our proposed early fixing framework can be regarded as an acceleration extension of ADMM methods for solving integer programming. The source codes are available at \url{https://github.com/SCLBD/Accelerated-Lpbox-ADMM}.

摘要: 整数规划(IP)是一个重要且具有挑战性的问题。近似方法在解决IP问题的有效性和效率方面都表现出了良好的性能。然而，我们观察到，一些迭代近似方法求解的大部分变量在很长的迭代中围绕其最终收敛的离散状态波动。受这一观察结果的启发，我们的目标是在不显著损害解精度的情况下，通过及早将这些波动变量固定到它们的收敛状态来加速这些近似方法。为此，我们提出了一个早期固定框架和近似方法。我们将整个早期定位过程描述为一个马尔可夫决策过程，并利用模仿学习对其进行训练。策略网络将评估每个自由变量在每个迭代块中关于其离散候选状态的后验概率。具体地说，我们在政策网络中采用了强大的多头注意机制。针对三种不同的IP应用：约束线性规划、马尔可夫随机场能量最小化和稀疏敌意攻击，对我们提出的早期修复框架进行了大量的实验。前一个是线性IP问题，后两个是二次IP问题。我们将问题的规模从常规规模扩展到显著的大型规模。广泛的实验表明了我们早期修复框架的竞争力：运行时间显著加快，而解决方案质量并没有太大下降，甚至在某些情况下可以获得更好的解决方案。我们提出的早期固定框架可以看作是求解整数规划的ADMM方法的加速扩展。源代码可在\url{https://github.com/SCLBD/Accelerated-Lpbox-ADMM}.上找到



## **7. Benign Adversarial Attack: Tricking Models for Goodness**

良性对抗性攻击：欺骗模型向善 cs.AI

ACM MM2022 Brave New Idea

**SubmitDate**: 2022-07-05    [paper-pdf](http://arxiv.org/pdf/2107.11986v2)

**Authors**: Jitao Sang, Xian Zhao, Jiaming Zhang, Zhiyu Lin

**Abstracts**: In spite of the successful application in many fields, machine learning models today suffer from notorious problems like vulnerability to adversarial examples. Beyond falling into the cat-and-mouse game between adversarial attack and defense, this paper provides alternative perspective to consider adversarial example and explore whether we can exploit it in benign applications. We first attribute adversarial example to the human-model disparity on employing non-semantic features. While largely ignored in classical machine learning mechanisms, non-semantic feature enjoys three interesting characteristics as (1) exclusive to model, (2) critical to affect inference, and (3) utilizable as features. Inspired by this, we present brave new idea of benign adversarial attack to exploit adversarial examples for goodness in three directions: (1) adversarial Turing test, (2) rejecting malicious model application, and (3) adversarial data augmentation. Each direction is positioned with motivation elaboration, justification analysis and prototype applications to showcase its potential.

摘要: 尽管机器学习模型在许多领域都得到了成功的应用，但它仍然存在一些臭名昭著的问题，比如易受敌意例子的攻击。除了陷入对抗性攻击和防御之间的猫鼠游戏之外，本文还提供了另一种视角来考虑对抗性例子，并探索是否可以在良性应用中利用它。我们首先将对抗性例子归因于人类模型在使用非语义特征上的差异。在经典的机器学习机制中，非语义特征被忽略了，但它具有三个有趣的特征：(1)模型独有的，(2)对影响推理的关键的，(3)可用作特征的。受此启发，我们提出了良性对抗性攻击的大胆新思想，从三个方向挖掘对抗性实例：(1)对抗性图灵测试，(2)拒绝恶意模型应用，(3)对抗性数据扩充。每个方向都定位于动机阐述、理由分析和原型应用，以展示其潜力。



## **8. Formalizing and Estimating Distribution Inference Risks**

分布推断风险的形式化和估计 cs.LG

Update: Accepted at PETS 2022

**SubmitDate**: 2022-07-05    [paper-pdf](http://arxiv.org/pdf/2109.06024v6)

**Authors**: Anshuman Suri, David Evans

**Abstracts**: Distribution inference, sometimes called property inference, infers statistical properties about a training set from access to a model trained on that data. Distribution inference attacks can pose serious risks when models are trained on private data, but are difficult to distinguish from the intrinsic purpose of statistical machine learning -- namely, to produce models that capture statistical properties about a distribution. Motivated by Yeom et al.'s membership inference framework, we propose a formal definition of distribution inference attacks that is general enough to describe a broad class of attacks distinguishing between possible training distributions. We show how our definition captures previous ratio-based property inference attacks as well as new kinds of attack including revealing the average node degree or clustering coefficient of a training graph. To understand distribution inference risks, we introduce a metric that quantifies observed leakage by relating it to the leakage that would occur if samples from the training distribution were provided directly to the adversary. We report on a series of experiments across a range of different distributions using both novel black-box attacks and improved versions of the state-of-the-art white-box attacks. Our results show that inexpensive attacks are often as effective as expensive meta-classifier attacks, and that there are surprising asymmetries in the effectiveness of attacks. Code is available at https://github.com/iamgroot42/FormEstDistRisks

摘要: 分布推理，有时被称为属性推理，从对基于该数据训练的模型的访问中推断出关于训练集的统计属性。当模型根据私有数据进行训练时，分布推断攻击可能会带来严重的风险，但很难与统计机器学习的内在目的区分开来--即产生捕获有关分布的统计属性的模型。受Yeom等人的成员关系推理框架的启发，我们提出了分布推理攻击的形式化定义，该定义足够通用，以描述区分可能的训练分布的广泛的攻击类别。我们展示了我们的定义如何捕获以前的基于比率的属性推理攻击，以及新的攻击类型，包括揭示训练图的平均结点度或聚类系数。为了了解分布推断风险，我们引入了一个度量，该度量通过将观察到的泄漏与训练分布的样本直接提供给对手时将发生的泄漏联系起来，来量化观察到的泄漏。我们报告了使用新的黑盒攻击和最先进的白盒攻击的改进版本在一系列不同的发行版上进行的一系列实验。我们的结果表明，廉价的攻击通常与昂贵的元分类器攻击一样有效，并且攻击的有效性存在惊人的不对称性。代码可在https://github.com/iamgroot42/FormEstDistRisks上找到



## **9. The StarCraft Multi-Agent Challenges+ : Learning of Multi-Stage Tasks and Environmental Factors without Precise Reward Functions**

星际争霸多智能体挑战+：没有精确奖励函数的多阶段任务和环境因素的学习 cs.LG

ICML Workshop: AI for Agent Based Modeling 2022 Spotlight

**SubmitDate**: 2022-07-05    [paper-pdf](http://arxiv.org/pdf/2207.02007v1)

**Authors**: Mingyu Kim, Jihwan Oh, Yongsik Lee, Joonkee Kim, Seonghwan Kim, Song Chong, Se-Young Yun

**Abstracts**: In this paper, we propose a novel benchmark called the StarCraft Multi-Agent Challenges+, where agents learn to perform multi-stage tasks and to use environmental factors without precise reward functions. The previous challenges (SMAC) recognized as a standard benchmark of Multi-Agent Reinforcement Learning are mainly concerned with ensuring that all agents cooperatively eliminate approaching adversaries only through fine manipulation with obvious reward functions.   This challenge, on the other hand, is interested in the exploration capability of MARL algorithms to efficiently learn implicit multi-stage tasks and environmental factors as well as micro-control. This study covers both offensive and defensive scenarios.   In the offensive scenarios, agents must learn to first find opponents and then eliminate them. The defensive scenarios require agents to use topographic features. For example, agents need to position themselves behind protective structures to make it harder for enemies to attack. We investigate MARL algorithms under SMAC+ and observe that recent approaches work well in similar settings to the previous challenges, but misbehave in offensive scenarios.   Additionally, we observe that an enhanced exploration approach has a positive effect on performance but is not able to completely solve all scenarios. This study proposes new directions for future research.

摘要: 在本文中，我们提出了一个新的基准测试称为星际争霸多代理挑战+，其中代理学习执行多阶段任务和使用环境因素，而不是精确的奖励函数。以前的挑战(SMAC)被公认为多智能体强化学习的标准基准，主要涉及确保所有智能体只有通过具有明显奖励功能的精细操纵才能合作地消除逼近的对手。另一方面，这一挑战与Marl算法高效学习隐式多阶段任务和环境因素以及微控制的探索能力有关。这项研究涵盖了进攻和防守两种情况。在进攻场景中，代理人必须学会首先找到对手，然后消灭他们。防御场景要求特工使用地形特征。例如，特工需要将自己部署在保护性建筑后面，以使敌人更难发动攻击。我们研究了SMAC+下的Marl算法，观察到最近的方法在类似于之前的挑战的环境下工作得很好，但在攻击性场景中表现不佳。此外，我们观察到，增强的探索方法对性能有积极影响，但并不能完全解决所有情况。本研究为今后的研究提出了新的方向。



## **10. Query-Efficient Adversarial Attack Based on Latin Hypercube Sampling**

基于拉丁超立方体采样的查询高效对抗攻击 cs.CV

**SubmitDate**: 2022-07-05    [paper-pdf](http://arxiv.org/pdf/2207.02391v1)

**Authors**: Dan Wang, Jiayu Lin, Yuan-Gen Wang

**Abstracts**: In order to be applicable in real-world scenario, Boundary Attacks (BAs) were proposed and ensured one hundred percent attack success rate with only decision information. However, existing BA methods craft adversarial examples by leveraging a simple random sampling (SRS) to estimate the gradient, consuming a large number of model queries. To overcome the drawback of SRS, this paper proposes a Latin Hypercube Sampling based Boundary Attack (LHS-BA) to save query budget. Compared with SRS, LHS has better uniformity under the same limited number of random samples. Therefore, the average on these random samples is closer to the true gradient than that estimated by SRS. Various experiments are conducted on benchmark datasets including MNIST, CIFAR, and ImageNet-1K. Experimental results demonstrate the superiority of the proposed LHS-BA over the state-of-the-art BA methods in terms of query efficiency. The source codes are publicly available at https://github.com/GZHU-DVL/LHS-BA.

摘要: 边界攻击(BAS)是为了更好地应用于实际场景而提出的，只需要决策信息就能保证100%的攻击成功率。然而，现有的BA方法通过利用简单随机抽样(SRS)来估计梯度来制作对抗性例子，消耗了大量的模型查询。针对SRS算法的不足，提出了一种基于拉丁超立方体采样的边界攻击算法(LHS-BA)，以节省查询开销。与SRS相比，在相同的有限随机样本数下，LHS具有更好的一致性。因此，这些随机样本的平均值比SRS估计的梯度更接近真实梯度。在包括MNIST、CIFAR和ImageNet-1K在内的基准数据集上进行了各种实验。实验结果表明，LHS-BA在查询效率方面优于现有的BA方法。源代码可在https://github.com/GZHU-DVL/LHS-BA.上公开获取



## **11. Task-agnostic Defense against Adversarial Patch Attacks**

对抗对抗性补丁攻击的任务不可知防御 cs.CV

**SubmitDate**: 2022-07-05    [paper-pdf](http://arxiv.org/pdf/2207.01795v1)

**Authors**: Ke Xu, Yao Xiao, Zhaoheng Zheng, Kaijie Cai, Ram Nevatia

**Abstracts**: Adversarial patch attacks mislead neural networks by injecting adversarial pixels within a designated local region. Patch attacks can be highly effective in a variety of tasks and physically realizable via attachment (e.g. a sticker) to the real-world objects. Despite the diversity in attack patterns, adversarial patches tend to be highly textured and different in appearance from natural images. We exploit this property and present PatchZero, a task-agnostic defense against white-box adversarial patches. Specifically, our defense detects the adversarial pixels and "zeros out" the patch region by repainting with mean pixel values. We formulate the patch detection problem as a semantic segmentation task such that our model can generalize to patches of any size and shape. We further design a two-stage adversarial training scheme to defend against the stronger adaptive attacks. We thoroughly evaluate PatchZero on the image classification (ImageNet, RESISC45), object detection (PASCAL VOC), and video classification (UCF101) datasets. Our method achieves SOTA robust accuracy without any degradation in the benign performance.

摘要: 对抗性补丁攻击通过在指定的局部区域内注入对抗性像素来误导神经网络。补丁攻击可以在各种任务中非常有效，并且可以通过附着(例如贴纸)到真实世界的对象来物理实现。尽管攻击模式多种多样，但敌方补丁往往纹理丰富，外观与自然图像不同。我们利用这一性质，提出了PatchZero，一种针对白盒对手补丁的与任务无关的防御。具体地说，我们的防御检测到对抗性像素，并通过使用平均像素值重新绘制补丁区域来“清零”。我们将斑块检测问题描述为一个语义分割任务，使得我们的模型可以推广到任何大小和形状的斑块。我们进一步设计了一种两阶段对抗性训练方案，以抵御更强的适应性攻击。我们在图像分类(ImageNet，RESISC45)、目标检测(Pascal VOC)和视频分类(UCF101)数据集上对PatchZero进行了全面的评估。我们的方法在不降低良好性能的情况下达到了SOTA的稳健精度。



## **12. Transferable Graph Backdoor Attack**

可转移图后门攻击 cs.CR

Accepted by the 25th International Symposium on Research in Attacks,  Intrusions, and Defenses

**SubmitDate**: 2022-07-05    [paper-pdf](http://arxiv.org/pdf/2207.00425v3)

**Authors**: Shuiqiao Yang, Bao Gia Doan, Paul Montague, Olivier De Vel, Tamas Abraham, Seyit Camtepe, Damith C. Ranasinghe, Salil S. Kanhere

**Abstracts**: Graph Neural Networks (GNNs) have achieved tremendous success in many graph mining tasks benefitting from the message passing strategy that fuses the local structure and node features for better graph representation learning. Despite the success of GNNs, and similar to other types of deep neural networks, GNNs are found to be vulnerable to unnoticeable perturbations on both graph structure and node features. Many adversarial attacks have been proposed to disclose the fragility of GNNs under different perturbation strategies to create adversarial examples. However, vulnerability of GNNs to successful backdoor attacks was only shown recently. In this paper, we disclose the TRAP attack, a Transferable GRAPh backdoor attack. The core attack principle is to poison the training dataset with perturbation-based triggers that can lead to an effective and transferable backdoor attack. The perturbation trigger for a graph is generated by performing the perturbation actions on the graph structure via a gradient based score matrix from a surrogate model. Compared with prior works, TRAP attack is different in several ways: i) it exploits a surrogate Graph Convolutional Network (GCN) model to generate perturbation triggers for a blackbox based backdoor attack; ii) it generates sample-specific perturbation triggers which do not have a fixed pattern; and iii) the attack transfers, for the first time in the context of GNNs, to different GNN models when trained with the forged poisoned training dataset. Through extensive evaluations on four real-world datasets, we demonstrate the effectiveness of the TRAP attack to build transferable backdoors in four different popular GNNs using four real-world datasets.

摘要: 图神经网络在许多图挖掘任务中取得了巨大的成功，得益于融合了局部结构和节点特征的消息传递策略，以实现更好的图表示学习。尽管GNN取得了成功，但与其他类型的深度神经网络类似，GNN被发现容易受到图结构和节点特征的不可察觉的扰动。已经提出了许多对抗性攻击，以揭示GNN在不同扰动策略下的脆弱性，以创建对抗性示例。然而，GNN对成功的后门攻击的脆弱性直到最近才显示出来。在本文中，我们揭示了陷阱攻击，一种可转移的图后门攻击。核心攻击原理是使用基于扰动的触发来毒化训练数据集，从而导致有效且可转移的后门攻击。通过经由来自代理模型的基于梯度的分数矩阵对图结构执行扰动动作来生成用于图的扰动触发器。与以往的工作相比，陷阱攻击在以下几个方面有所不同：i)它利用代理图卷积网络(GCN)模型为基于黑盒的后门攻击生成扰动触发器；ii)它生成的样本特定的扰动触发器没有固定的模式；iii)攻击首次在GNN的上下文中传输到不同的GNN模型，当使用伪造的有毒训练数据集进行训练时。通过对四个真实世界数据集的广泛评估，我们使用四个真实世界数据集演示了陷阱攻击在四个不同流行的GNN中建立可转移后门的有效性。



## **13. Wild Networks: Exposure of 5G Network Infrastructures to Adversarial Examples**

狂野网络：5G网络基础设施暴露在敌意例子面前 cs.CR

**SubmitDate**: 2022-07-04    [paper-pdf](http://arxiv.org/pdf/2207.01531v1)

**Authors**: Giovanni Apruzzese, Rodion Vladimirov, Aliya Tastemirova, Pavel Laskov

**Abstracts**: Fifth Generation (5G) networks must support billions of heterogeneous devices while guaranteeing optimal Quality of Service (QoS). Such requirements are impossible to meet with human effort alone, and Machine Learning (ML) represents a core asset in 5G. ML, however, is known to be vulnerable to adversarial examples; moreover, as our paper will show, the 5G context is exposed to a yet another type of adversarial ML attacks that cannot be formalized with existing threat models. Proactive assessment of such risks is also challenging due to the lack of ML-powered 5G equipment available for adversarial ML research.   To tackle these problems, we propose a novel adversarial ML threat model that is particularly suited to 5G scenarios, and is agnostic to the precise function solved by ML. In contrast to existing ML threat models, our attacks do not require any compromise of the target 5G system while still being viable due to the QoS guarantees and the open nature of 5G networks. Furthermore, we propose an original framework for realistic ML security assessments based on public data. We proactively evaluate our threat model on 6 applications of ML envisioned in 5G. Our attacks affect both the training and the inference stages, can degrade the performance of state-of-the-art ML systems, and have a lower entry barrier than previous attacks.

摘要: 第五代(5G)网络必须支持数十亿台异类设备，同时保证最佳服务质量(Qos)。仅靠人的努力是不可能满足这些要求的，而机器学习(ML)是5G中的一项核心资产。然而，众所周知，ML容易受到对抗性例子的攻击；此外，正如我们的论文所示，5G环境面临着另一种类型的对抗性ML攻击，而这种攻击无法用现有的威胁模型形式化。由于缺乏支持ML的5G设备用于对抗性ML研究，因此对此类风险的主动评估也具有挑战性。为了解决这些问题，我们提出了一种新的对抗性ML威胁模型，该模型特别适合于5G场景，并且与ML解决的精确函数无关。与现有的ML威胁模型相比，我们的攻击不需要对目标5G系统进行任何妥协，同时由于5G网络的服务质量保证和开放性质，我们的攻击仍然可行。此外，我们还提出了一个基于公共数据的真实ML安全评估框架。我们在5G中设想的6个ML应用上主动评估了我们的威胁模型。我们的攻击同时影响训练和推理阶段，会降低最新的ML系统的性能，并且比以前的攻击具有更低的进入门槛。



## **14. Adversarial Ensemble Training by Jointly Learning Label Dependencies and Member Models**

联合学习标签依赖关系和成员模型的对抗性集成训练 cs.LG

**SubmitDate**: 2022-07-04    [paper-pdf](http://arxiv.org/pdf/2206.14477v2)

**Authors**: Lele Wang, Bin Liu

**Abstracts**: Training an ensemble of different sub-models has empirically proven to be an effective strategy to improve deep neural networks' adversarial robustness. Current ensemble training methods for image recognition usually encode the image labels by one-hot vectors, which neglect dependency relationships between the labels. Here we propose a novel adversarial ensemble training approach to jointly learn the label dependencies and the member models. Our approach adaptively exploits the learned label dependencies to promote the diversity of the member models. We test our approach on widely used datasets MNIST, FasionMNIST, and CIFAR-10. Results show that our approach is more robust against black-box attacks compared with the state-of-the-art methods. Our code is available at https://github.com/ZJLAB-AMMI/LSD.

摘要: 实验证明，训练不同子模型的集成是提高深度神经网络对抗健壮性的有效策略。目前用于图像识别的集成训练方法通常将图像标签编码为单热点向量，忽略了标签之间的依赖关系。在这里，我们提出了一种新的对抗性集成训练方法来联合学习标签依赖和成员模型。我们的方法自适应地利用学习到的标签依赖关系来促进成员模型的多样性。我们在广泛使用的数据集MNIST、FasionMNIST和CIFAR-10上测试了我们的方法。结果表明，与现有的方法相比，该方法对黑盒攻击具有更强的鲁棒性。我们的代码可以在https://github.com/ZJLAB-AMMI/LSD.上找到



## **15. Hessian-Free Second-Order Adversarial Examples for Adversarial Learning**

用于对抗性学习的Hessian-Free二阶对抗性实例 cs.LG

**SubmitDate**: 2022-07-04    [paper-pdf](http://arxiv.org/pdf/2207.01396v1)

**Authors**: Yaguan Qian, Yuqi Wang, Bin Wang, Zhaoquan Gu, Yuhan Guo, Wassim Swaileh

**Abstracts**: Recent studies show deep neural networks (DNNs) are extremely vulnerable to the elaborately designed adversarial examples. Adversarial learning with those adversarial examples has been proved as one of the most effective methods to defend against such an attack. At present, most existing adversarial examples generation methods are based on first-order gradients, which can hardly further improve models' robustness, especially when facing second-order adversarial attacks. Compared with first-order gradients, second-order gradients provide a more accurate approximation of the loss landscape with respect to natural examples. Inspired by this, our work crafts second-order adversarial examples and uses them to train DNNs. Nevertheless, second-order optimization involves time-consuming calculation for Hessian-inverse. We propose an approximation method through transforming the problem into an optimization in the Krylov subspace, which remarkably reduce the computational complexity to speed up the training procedure. Extensive experiments conducted on the MINIST and CIFAR-10 datasets show that our adversarial learning with second-order adversarial examples outperforms other fisrt-order methods, which can improve the model robustness against a wide range of attacks.

摘要: 最近的研究表明，深度神经网络(DNN)非常容易受到精心设计的敌意例子的影响。用这些对抗性例子进行对抗性学习已被证明是防御此类攻击的最有效方法之一。目前，已有的对抗性样本生成方法大多是基于一阶梯度的，这很难进一步提高模型的稳健性，尤其是在面对二阶对抗性攻击时。与一阶梯度相比，二阶梯度相对于自然例子提供了更准确的损失情况的近似。受此启发，我们的工作制作了二阶对抗性例子，并用它们来训练DNN。然而，二阶优化涉及耗时的海森逆计算。通过将问题转化为Krylov子空间中的优化问题，提出了一种近似方法，大大降低了计算复杂度，加快了训练过程。在MINIST和CIFAR-10数据集上进行的大量实验表明，本文提出的基于二阶对抗性实例的对抗性学习方法优于其他一阶方法，可以提高模型对大范围攻击的稳健性。



## **16. Training strategy for a lightweight countermeasure model for automatic speaker verification**

一种轻量级说话人自动确认对策模型的训练策略 cs.SD

Need to finish experiment

**SubmitDate**: 2022-07-04    [paper-pdf](http://arxiv.org/pdf/2203.17031v4)

**Authors**: Yen-Lun Liao, Xuanjun Chen, Chung-Che Wang, Jyh-Shing Roger Jang

**Abstracts**: The countermeasure (CM) model is developed to protect Automatic Speaker Verification (ASV) systems from spoof attacks and prevent resulting personal information leakage. Based on practicality and security considerations, the CM model is usually deployed on edge devices, which have more limited computing resources and storage space than cloud-based systems. This work proposes training strategies for a lightweight CM model for ASV, using generalized end-to-end (GE2E) pre-training and adversarial fine-tuning to improve performance, and applying knowledge distillation (KD) to reduce the size of the CM model. In the evaluation phase of the ASVspoof 2021 Logical Access task, the lightweight ResNetSE model reaches min t-DCF 0.2695 and EER 3.54%. Compared to the teacher model, the lightweight student model only uses 22.5% of parameters and 21.1% of multiply and accumulate operands of the teacher model.

摘要: 对策(CM)模型是为了保护自动说话人验证(ASV)系统免受欺骗攻击并防止由此导致的个人信息泄露而开发的。基于实用性和安全性考虑，CM模型通常部署在边缘设备上，与基于云的系统相比，边缘设备的计算资源和存储空间更有限。提出了一种面向ASV的轻量级CM模型的训练策略，使用通用端到端(GE2E)预训练和对抗性微调来提高性能，并应用知识蒸馏(KD)来减小CM模型的规模。在ASVspoof2021逻辑访问任务的评估阶段，轻量级ResNetSE模型达到了最小t-DCF值0.2695和EER3.54%.与教师模型相比，轻量级学生模型仅使用了教师模型22.5%的参数和21.1%的乘加操作数。



## **17. BadHash: Invisible Backdoor Attacks against Deep Hashing with Clean Label**

BadHash：使用Clean Label对深度哈希进行隐形后门攻击 cs.CV

conference

**SubmitDate**: 2022-07-04    [paper-pdf](http://arxiv.org/pdf/2207.00278v2)

**Authors**: Shengshan Hu, Ziqi Zhou, Yechao Zhang, Leo Yu Zhang, Yifeng Zheng, Yuanyuan HE, Hai Jin

**Abstracts**: Due to its powerful feature learning capability and high efficiency, deep hashing has achieved great success in large-scale image retrieval. Meanwhile, extensive works have demonstrated that deep neural networks (DNNs) are susceptible to adversarial examples, and exploring adversarial attack against deep hashing has attracted many research efforts. Nevertheless, backdoor attack, another famous threat to DNNs, has not been studied for deep hashing yet. Although various backdoor attacks have been proposed in the field of image classification, existing approaches failed to realize a truly imperceptive backdoor attack that enjoys invisible triggers and clean label setting simultaneously, and they also cannot meet the intrinsic demand of image retrieval backdoor.   In this paper, we propose BadHash, the first generative-based imperceptible backdoor attack against deep hashing, which can effectively generate invisible and input-specific poisoned images with clean label. Specifically, we first propose a new conditional generative adversarial network (cGAN) pipeline to effectively generate poisoned samples. For any given benign image, it seeks to generate a natural-looking poisoned counterpart with a unique invisible trigger. In order to improve the attack effectiveness, we introduce a label-based contrastive learning network LabCLN to exploit the semantic characteristics of different labels, which are subsequently used for confusing and misleading the target model to learn the embedded trigger. We finally explore the mechanism of backdoor attacks on image retrieval in the hash space. Extensive experiments on multiple benchmark datasets verify that BadHash can generate imperceptible poisoned samples with strong attack ability and transferability over state-of-the-art deep hashing schemes.

摘要: 深度哈希法由于其强大的特征学习能力和高效的检索效率，在大规模图像检索中取得了巨大的成功。同时，大量的研究表明，深度神经网络(DNN)容易受到敌意例子的影响，探索针对深度散列的敌意攻击吸引了许多研究努力。然而，DNNS的另一个著名威胁--后门攻击，还没有被研究过深度散列。虽然在图像分类领域已经提出了各种各样的后门攻击，但现有的方法未能实现真正的隐蔽的、同时具有不可见触发器和干净标签设置的后门攻击，也不能满足图像检索的内在需求。本文提出了BadHash，这是第一个基于生成性的针对深度哈希的不可察觉的后门攻击，它可以有效地生成标签清晰的不可见和输入特定的有毒图像。具体地说，我们首先提出了一种新的条件生成对抗网络(CGAN)管道来有效地生成有毒样本。对于任何给定的良性形象，它都试图生成一个看起来自然、有毒的形象，并带有独特的无形触发器。为了提高攻击的有效性，我们引入了一个基于标签的对比学习网络LabCLN来利用不同标签的语义特征，这些语义特征被用来混淆和误导目标模型学习嵌入的触发器。最后，我们探讨了哈希空间中后门攻击对图像检索的影响机制。在多个基准数据集上的大量实验证明，BadHash可以生成不可察觉的有毒样本，具有很强的攻击能力和可转移性，优于最先进的深度哈希方案。



## **18. Removing Batch Normalization Boosts Adversarial Training**

取消批次归一化加强对抗性训练 cs.LG

ICML 2022

**SubmitDate**: 2022-07-04    [paper-pdf](http://arxiv.org/pdf/2207.01156v1)

**Authors**: Haotao Wang, Aston Zhang, Shuai Zheng, Xingjian Shi, Mu Li, Zhangyang Wang

**Abstracts**: Adversarial training (AT) defends deep neural networks against adversarial attacks. One challenge that limits its practical application is the performance degradation on clean samples. A major bottleneck identified by previous works is the widely used batch normalization (BN), which struggles to model the different statistics of clean and adversarial training samples in AT. Although the dominant approach is to extend BN to capture this mixture of distribution, we propose to completely eliminate this bottleneck by removing all BN layers in AT. Our normalizer-free robust training (NoFrost) method extends recent advances in normalizer-free networks to AT for its unexplored advantage on handling the mixture distribution challenge. We show that NoFrost achieves adversarial robustness with only a minor sacrifice on clean sample accuracy. On ImageNet with ResNet50, NoFrost achieves $74.06\%$ clean accuracy, which drops merely $2.00\%$ from standard training. In contrast, BN-based AT obtains $59.28\%$ clean accuracy, suffering a significant $16.78\%$ drop from standard training. In addition, NoFrost achieves a $23.56\%$ adversarial robustness against PGD attack, which improves the $13.57\%$ robustness in BN-based AT. We observe better model smoothness and larger decision margins from NoFrost, which make the models less sensitive to input perturbations and thus more robust. Moreover, when incorporating more data augmentations into NoFrost, it achieves comprehensive robustness against multiple distribution shifts. Code and pre-trained models are public at https://github.com/amazon-research/normalizer-free-robust-training.

摘要: 对抗性训练(AT)保护深层神经网络免受对抗性攻击。限制其实际应用的一个挑战是清洁样品的性能下降。前人工作发现的一个主要瓶颈是广泛使用的批处理归一化(BN)，它难以对AT中的干净训练样本和对抗性训练样本的不同统计进行建模。虽然主要的方法是扩展BN来捕获这种混合分布，但我们建议通过移除AT中的所有BN层来完全消除这一瓶颈。我们的无规格化稳健训练(NoFrost)方法将无规格化网络的最新进展扩展到AT，因为它在处理混合分布挑战方面具有尚未开发的优势。我们证明了NoFrost在仅牺牲干净样本精度的情况下实现了对抗性稳健性。在带有ResNet50的ImageNet上，NoFrost达到了$74.06\$CLEAN精度，仅比标准训练降低了$2.00\$。相比之下，基于BN的AT获得了59.28美元的CLEAN准确率，与标准训练相比下降了16.78美元。此外，NoFrost在抵抗PGD攻击时达到了23.56美元的健壮性，提高了基于BN的AT的13.57美元的健壮性。我们观察到NoFrost模型具有更好的光滑性和更大的决策裕度，这使得模型对输入扰动不那么敏感，从而具有更强的鲁棒性。此外，当在NoFrost中加入更多的数据增强时，它实现了对多个分布偏移的综合稳健性。代码和预先培训的模型在https://github.com/amazon-research/normalizer-free-robust-training.上公开



## **19. RAF: Recursive Adversarial Attacks on Face Recognition Using Extremely Limited Queries**

RAF：使用极其有限的查询对人脸识别进行递归对抗性攻击 cs.CV

**SubmitDate**: 2022-07-04    [paper-pdf](http://arxiv.org/pdf/2207.01149v1)

**Authors**: Keshav Kasichainula, Hadi Mansourifar, Weidong Shi

**Abstracts**: Recent successful adversarial attacks on face recognition show that, despite the remarkable progress of face recognition models, they are still far behind the human intelligence for perception and recognition. It reveals the vulnerability of deep convolutional neural networks (CNNs) as state-of-the-art building block for face recognition models against adversarial examples, which can cause certain consequences for secure systems. Gradient-based adversarial attacks are widely studied before and proved to be successful against face recognition models. However, finding the optimized perturbation per each face needs to submitting the significant number of queries to the target model. In this paper, we propose recursive adversarial attack on face recognition using automatic face warping which needs extremely limited number of queries to fool the target model. Instead of a random face warping procedure, the warping functions are applied on specific detected regions of face like eyebrows, nose, lips, etc. We evaluate the robustness of proposed method in the decision-based black-box attack setting, where the attackers have no access to the model parameters and gradients, but hard-label predictions and confidence scores are provided by the target model.

摘要: 最近成功的人脸识别对抗性攻击表明，尽管人脸识别模型取得了显著的进步，但它们仍然远远落后于人类的感知和识别智能。它揭示了深度卷积神经网络(CNN)作为最先进的人脸识别模型构建块在对抗敌意示例时的脆弱性，这可能会对安全系统造成一定的后果。基于梯度的敌意攻击已被广泛研究，并被证明在人脸识别模型上是成功的。然而，找到每个人脸的优化扰动需要向目标模型提交大量的查询。在本文中，我们提出了基于自动人脸变形的递归对抗性人脸识别攻击，这种攻击只需要极其有限的查询次数就可以愚弄目标模型。在基于决策的黑盒攻击环境中，我们评估了该方法的稳健性，其中攻击者无法获得模型参数和梯度，但硬标签预测和置信度分数由目标模型提供。



## **20. Generating gender-ambiguous voices for privacy-preserving speech recognition**

为保护隐私的语音识别生成性别模糊的声音 cs.SD

5 pages, 4 figures, submitted to INTERSPEECH

**SubmitDate**: 2022-07-03    [paper-pdf](http://arxiv.org/pdf/2207.01052v1)

**Authors**: Dimitrios Stoidis, Andrea Cavallaro

**Abstracts**: Our voice encodes a uniquely identifiable pattern which can be used to infer private attributes, such as gender or identity, that an individual might wish not to reveal when using a speech recognition service. To prevent attribute inference attacks alongside speech recognition tasks, we present a generative adversarial network, GenGAN, that synthesises voices that conceal the gender or identity of a speaker. The proposed network includes a generator with a U-Net architecture that learns to fool a discriminator. We condition the generator only on gender information and use an adversarial loss between signal distortion and privacy preservation. We show that GenGAN improves the trade-off between privacy and utility compared to privacy-preserving representation learning methods that consider gender information as a sensitive attribute to protect.

摘要: 我们的声音编码了一个唯一可识别的模式，该模式可用于推断个人在使用语音识别服务时可能不希望透露的私人属性，如性别或身份。为了防止语音识别任务中的属性推理攻击，我们提出了一个生成性对抗网络GenGAN，它合成的语音隐藏了说话人的性别或身份。提议的网络包括一个具有U-Net架构的生成器，它可以学习愚弄鉴别者。我们仅根据性别信息来设置生成器的条件，并在信号失真和隐私保护之间使用对抗性损失。与将性别信息视为需要保护的敏感属性的隐私保护表示学习方法相比，GenGAN改进了隐私和效用之间的权衡。



## **21. Adversarial Attack and Defense of YOLO Detectors in Autonomous Driving Scenarios**

自主驾驶场景中YOLO检测器的对抗性攻击与防御 cs.CV

Accepted by 2022 IEEE Intelligent Vehicles Symposium (IV 2022)

**SubmitDate**: 2022-07-03    [paper-pdf](http://arxiv.org/pdf/2202.04781v2)

**Authors**: Jung Im Choi, Qing Tian

**Abstracts**: Visual detection is a key task in autonomous driving, and it serves as a crucial foundation for self-driving planning and control. Deep neural networks have achieved promising results in various visual tasks, but they are known to be vulnerable to adversarial attacks. A comprehensive understanding of deep visual detectors' vulnerability is required before people can improve their robustness. However, only a few adversarial attack/defense works have focused on object detection, and most of them employed only classification and/or localization losses, ignoring the objectness aspect. In this paper, we identify a serious objectness-related adversarial vulnerability in YOLO detectors and present an effective attack strategy targeting the objectness aspect of visual detection in autonomous vehicles. Furthermore, to address such vulnerability, we propose a new objectness-aware adversarial training approach for visual detection. Experiments show that the proposed attack targeting the objectness aspect is 45.17% and 43.50% more effective than those generated from classification and/or localization losses on the KITTI and COCO traffic datasets, respectively. Also, the proposed adversarial defense approach can improve the detectors' robustness against objectness-oriented attacks by up to 21% and 12% mAP on KITTI and COCO traffic, respectively.

摘要: 视觉检测是自动驾驶中的一项关键任务，是自动驾驶规划和控制的重要基础。深度神经网络在各种视觉任务中取得了令人振奋的结果，但它们很容易受到对手的攻击。在人们提高其稳健性之前，需要对深度视觉检测器的脆弱性进行全面的了解。然而，只有少数对抗性攻防研究集中在目标检测方面，而且大多只采用分类和/或定位损失，而忽略了客观性方面。本文针对自主车辆视觉检测的客观性方面，针对YOLO检测器存在的一个严重的客观性攻击漏洞，提出了一种有效的攻击策略。此外，为了解决这种脆弱性，我们提出了一种新的基于客观性感知的对抗性视觉检测训练方法。实验表明，针对客观性方面的攻击，在Kitti和CoCo流量数据集上分别比分类和/或定位损失产生的攻击有效45.17%和43.50%。此外，在Kitti和CoCo流量上，提出的对抗性防御方法可以将检测器对面向对象攻击的健壮性分别提高21%和12%。



## **22. AGIC: Approximate Gradient Inversion Attack on Federated Learning**

AGIC：联邦学习中的近似梯度反转攻击 cs.LG

This paper is accepted at the 41st International Symposium on  Reliable Distributed Systems (SRDS 2022)

**SubmitDate**: 2022-07-03    [paper-pdf](http://arxiv.org/pdf/2204.13784v2)

**Authors**: Jin Xu, Chi Hong, Jiyue Huang, Lydia Y. Chen, Jérémie Decouchant

**Abstracts**: Federated learning is a private-by-design distributed learning paradigm where clients train local models on their own data before a central server aggregates their local updates to compute a global model. Depending on the aggregation method used, the local updates are either the gradients or the weights of local learning models. Recent reconstruction attacks apply a gradient inversion optimization on the gradient update of a single minibatch to reconstruct the private data used by clients during training. As the state-of-the-art reconstruction attacks solely focus on single update, realistic adversarial scenarios are overlooked, such as observation across multiple updates and updates trained from multiple mini-batches. A few studies consider a more challenging adversarial scenario where only model updates based on multiple mini-batches are observable, and resort to computationally expensive simulation to untangle the underlying samples for each local step. In this paper, we propose AGIC, a novel Approximate Gradient Inversion Attack that efficiently and effectively reconstructs images from both model or gradient updates, and across multiple epochs. In a nutshell, AGIC (i) approximates gradient updates of used training samples from model updates to avoid costly simulation procedures, (ii) leverages gradient/model updates collected from multiple epochs, and (iii) assigns increasing weights to layers with respect to the neural network structure for reconstruction quality. We extensively evaluate AGIC on three datasets, CIFAR-10, CIFAR-100 and ImageNet. Our results show that AGIC increases the peak signal-to-noise ratio (PSNR) by up to 50% compared to two representative state-of-the-art gradient inversion attacks. Furthermore, AGIC is faster than the state-of-the-art simulation based attack, e.g., it is 5x faster when attacking FedAvg with 8 local steps in between model updates.

摘要: 联合学习是一种私人设计的分布式学习范例，其中客户端在中央服务器聚合其本地更新以计算全局模型之前，根据自己的数据训练本地模型。根据所使用的聚合方法，局部更新要么是局部学习模型的梯度，要么是局部学习模型的权重。最近的重建攻击将梯度倒置优化应用于单个小批量的梯度更新，以重建客户在训练期间使用的私有数据。由于最新的重建攻击只关注单个更新，因此忽略了现实的对抗性场景，例如跨多个更新的观察和从多个小批次训练的更新。一些研究考虑了一种更具挑战性的对抗性场景，其中只能观察到基于多个小批次的模型更新，并求助于计算代价高昂的模拟来解开每个局部步骤的潜在样本。在本文中，我们提出了AGIC，一种新的近似梯度反转攻击，它可以高效地从模型或梯度更新中重建图像，并跨越多个历元。简而言之，AGIC(I)根据模型更新近似使用的训练样本的梯度更新以避免昂贵的模拟过程，(Ii)利用从多个历元收集的梯度/模型更新，以及(Iii)为重建质量向层分配相对于神经网络结构的不断增加的权重。我们在三个数据集CIFAR-10、CIFAR-100和ImageNet上对AGIC进行了广泛的评估。实验结果表明，与两种典型的梯度反转攻击相比，AGIC的峰值信噪比(PSNR)提高了50%。此外，AGIC比最先进的基于模拟的攻击速度更快，例如，在模型更新之间有8个本地步骤的情况下，攻击FedAvg的速度要快5倍。



## **23. Tricking the Hashing Trick: A Tight Lower Bound on the Robustness of CountSketch to Adaptive Inputs**

欺骗散列技巧：CountSketch对自适应输入的稳健性的紧致下界 cs.DS

**SubmitDate**: 2022-07-03    [paper-pdf](http://arxiv.org/pdf/2207.00956v1)

**Authors**: Edith Cohen, Jelani Nelson, Tamás Sarlós, Uri Stemmer

**Abstracts**: CountSketch and Feature Hashing (the "hashing trick") are popular randomized dimensionality reduction methods that support recovery of $\ell_2$-heavy hitters (keys $i$ where $v_i^2 > \epsilon \|\boldsymbol{v}\|_2^2$) and approximate inner products. When the inputs are {\em not adaptive} (do not depend on prior outputs), classic estimators applied to a sketch of size $O(\ell/\epsilon)$ are accurate for a number of queries that is exponential in $\ell$. When inputs are adaptive, however, an adversarial input can be constructed after $O(\ell)$ queries with the classic estimator and the best known robust estimator only supports $\tilde{O}(\ell^2)$ queries. In this work we show that this quadratic dependence is in a sense inherent: We design an attack that after $O(\ell^2)$ queries produces an adversarial input vector whose sketch is highly biased. Our attack uses "natural" non-adaptive inputs (only the final adversarial input is chosen adaptively) and universally applies with any correct estimator, including one that is unknown to the attacker. In that, we expose inherent vulnerability of this fundamental method.

摘要: CountSketch和Feature Hash(散列技巧)是流行的随机降维方法，支持恢复$\ell_2$-重打击者(key$i$where$v_i^2>\epsilon\boldsign{v}\_2^2$)和近似内积。当输入是{\em不自适应的}(不依赖于先前的输出)时，应用于大小为$O(\ell/\epsilon)$的草图的经典估计器对于许多以$\ell$为指数的查询是准确的。然而，当输入是自适应的时，可以用经典估计在$O(\ell)$查询之后构造敌意输入，而最著名的稳健估计只支持$T{O}(\ell^2)$查询。在这项工作中，我们证明了这种二次依赖在某种意义上是固有的：我们设计了一个攻击，在$O(\ell^2)$查询后产生一个高度有偏的敌对输入向量。我们的攻击使用“自然的”非自适应输入(只有最终的对抗性输入是自适应选择的)，并且普遍适用于任何正确的估计器，包括攻击者未知的估计器。在这一点上，我们暴露了这一基本方法的固有弱点。



## **24. When Are Linear Stochastic Bandits Attackable?**

线性随机土匪什么时候可以攻击？ cs.LG

27 pages, 3 figures, ICML 2022

**SubmitDate**: 2022-07-03    [paper-pdf](http://arxiv.org/pdf/2110.09008v2)

**Authors**: Huazheng Wang, Haifeng Xu, Hongning Wang

**Abstracts**: We study adversarial attacks on linear stochastic bandits: by manipulating the rewards, an adversary aims to control the behaviour of the bandit algorithm. Perhaps surprisingly, we first show that some attack goals can never be achieved. This is in sharp contrast to context-free stochastic bandits, and is intrinsically due to the correlation among arms in linear stochastic bandits. Motivated by this finding, this paper studies the attackability of a $k$-armed linear bandit environment. We first provide a complete necessity and sufficiency characterization of attackability based on the geometry of the arms' context vectors. We then propose a two-stage attack method against LinUCB and Robust Phase Elimination. The method first asserts whether the given environment is attackable; and if yes, it poisons the rewards to force the algorithm to pull a target arm linear times using only a sublinear cost. Numerical experiments further validate the effectiveness and cost-efficiency of the proposed attack method.

摘要: 我们研究对线性随机强盗的敌意攻击：通过操纵报酬，敌手的目标是控制强盗算法的行为。或许令人惊讶的是，我们首先展示了一些进攻目标永远无法实现。这与上下文无关的随机土匪形成了鲜明的对比，本质上是由于线性随机土匪中武器之间的相关性。受这一发现的启发，本文研究了$k$-武装线性盗贼环境的可攻性。我们首先给出了基于武器上下文向量几何的可攻击性的完全必要性和充分性刻画。然后，我们提出了一种针对LinUCB的两阶段攻击方法和稳健阶段消除方法。该方法首先断言给定环境是否是可攻击的；如果是，则毒化奖励以迫使算法仅使用次线性代价线性地拉动目标手臂。数值实验进一步验证了该攻击方法的有效性和成本效益。



## **25. Certifiably Robust Policy Learning against Adversarial Communication in Multi-agent Systems**

多智能体系统中对抗敌意通信的可证明稳健策略学习 cs.LG

**SubmitDate**: 2022-07-02    [paper-pdf](http://arxiv.org/pdf/2206.10158v2)

**Authors**: Yanchao Sun, Ruijie Zheng, Parisa Hassanzadeh, Yongyuan Liang, Soheil Feizi, Sumitra Ganesh, Furong Huang

**Abstracts**: Communication is important in many multi-agent reinforcement learning (MARL) problems for agents to share information and make good decisions. However, when deploying trained communicative agents in a real-world application where noise and potential attackers exist, the safety of communication-based policies becomes a severe issue that is underexplored. Specifically, if communication messages are manipulated by malicious attackers, agents relying on untrustworthy communication may take unsafe actions that lead to catastrophic consequences. Therefore, it is crucial to ensure that agents will not be misled by corrupted communication, while still benefiting from benign communication. In this work, we consider an environment with $N$ agents, where the attacker may arbitrarily change the communication from any $C<\frac{N-1}{2}$ agents to a victim agent. For this strong threat model, we propose a certifiable defense by constructing a message-ensemble policy that aggregates multiple randomly ablated message sets. Theoretical analysis shows that this message-ensemble policy can utilize benign communication while being certifiably robust to adversarial communication, regardless of the attacking algorithm. Experiments in multiple environments verify that our defense significantly improves the robustness of trained policies against various types of attacks.

摘要: 在许多多智能体强化学习(MAIL)问题中，通信对于智能体共享信息和做出正确的决策是非常重要的。然而，当在存在噪声和潜在攻击者的真实世界应用中部署训练有素的通信代理时，基于通信的策略的安全性成为一个未被探索的严重问题。具体地说，如果通信消息被恶意攻击者操纵，依赖于不可信通信的代理可能会采取不安全的行为，导致灾难性的后果。因此，确保代理不会被损坏的通信误导，同时仍受益于良性通信是至关重要的。在这项工作中，我们考虑了一个具有$N$代理的环境，在该环境中，攻击者可以任意地将通信从任何$C<\frac{N-1}{2}$代理更改为受害者代理。对于这种强威胁模型，我们通过构造聚合多个随机消融消息集的消息集成策略来提出一种可证明的防御。理论分析表明，无论采用何种攻击算法，该消息集成策略都能充分利用良性通信，同时对敌意通信具有较强的鲁棒性。在多个环境中的实验证明，我们的防御显著提高了经过训练的策略对各种类型攻击的健壮性。



## **26. Backdoor Attack is A Devil in Federated GAN-based Medical Image Synthesis**

后门攻击是联邦GAN医学图像合成中的一大难题 cs.CV

13 pages, 4 figures

**SubmitDate**: 2022-07-02    [paper-pdf](http://arxiv.org/pdf/2207.00762v1)

**Authors**: Ruinan Jin, Xiaoxiao Li

**Abstracts**: Deep Learning-based image synthesis techniques have been applied in healthcare research for generating medical images to support open research. Training generative adversarial neural networks (GAN) usually requires large amounts of training data. Federated learning (FL) provides a way of training a central model using distributed data from different medical institutions while keeping raw data locally. However, FL is vulnerable to backdoor attack, an adversarial by poisoning training data, given the central server cannot access the original data directly. Most backdoor attack strategies focus on classification models and centralized domains. In this study, we propose a way of attacking federated GAN (FedGAN) by treating the discriminator with a commonly used data poisoning strategy in backdoor attack classification models. We demonstrate that adding a small trigger with size less than 0.5 percent of the original image size can corrupt the FL-GAN model. Based on the proposed attack, we provide two effective defense strategies: global malicious detection and local training regularization. We show that combining the two defense strategies yields a robust medical image generation.

摘要: 基于深度学习的图像合成技术已被应用于医疗保健研究，以生成支持开放研究的医学图像。生成对抗神经网络(GAN)的训练通常需要大量的训练数据。联合学习(FL)提供了一种使用来自不同医疗机构的分布式数据训练中央模型的方法，同时保持本地的原始数据。然而，由于中央服务器不能直接访问原始数据，FL很容易受到后门攻击，这是通过毒化训练数据而产生的敌意。大多数后门攻击策略侧重于分类模型和集中域。在这项研究中，我们提出了一种利用后门攻击分类模型中常用的数据中毒策略来处理鉴别器来攻击联邦GAN(FedGAN)的方法。我们证明，添加一个尺寸小于原始图像尺寸0.5%的小触发器可以破坏FL-GaN模型。基于提出的攻击，我们提出了两种有效的防御策略：全局恶意检测和局部训练正则化。我们表明，结合这两种防御策略可以产生稳健的医学图像生成。



## **27. Efficient Adversarial Training With Data Pruning**

使用数据剪枝进行高效的对抗性训练 cs.LG

**SubmitDate**: 2022-07-01    [paper-pdf](http://arxiv.org/pdf/2207.00694v1)

**Authors**: Maximilian Kaufmann, Yiren Zhao, Ilia Shumailov, Robert Mullins, Nicolas Papernot

**Abstracts**: Neural networks are susceptible to adversarial examples-small input perturbations that cause models to fail. Adversarial training is one of the solutions that stops adversarial examples; models are exposed to attacks during training and learn to be resilient to them. Yet, such a procedure is currently expensive-it takes a long time to produce and train models with adversarial samples, and, what is worse, it occasionally fails. In this paper we demonstrate data pruning-a method for increasing adversarial training efficiency through data sub-sampling.We empirically show that data pruning leads to improvements in convergence and reliability of adversarial training, albeit with different levels of utility degradation. For example, we observe that using random sub-sampling of CIFAR10 to drop 40% of data, we lose 8% adversarial accuracy against the strongest attackers, while by using only 20% of data we lose 14% adversarial accuracy and reduce runtime by a factor of 3. Interestingly, we discover that in some settings data pruning brings benefits from both worlds-it both improves adversarial accuracy and training time.

摘要: 神经网络很容易受到对抗性例子的影响--微小的输入扰动会导致模型失败。对抗性训练是阻止对抗性例子的解决方案之一；模型在训练过程中暴露于攻击中，并学会对攻击具有弹性。然而，这样的程序目前成本很高--用对抗性样本生产和训练模型需要很长时间，更糟糕的是，它偶尔会失败。在本文中，我们展示了一种通过数据次采样来提高对抗性训练效率的方法--数据剪枝。我们的经验表明，数据剪枝可以提高对抗性训练的收敛和可靠性，尽管其效用降级程度不同。例如，我们观察到，使用CIFAR10的随机子采样来丢弃40%的数据，我们对最强的攻击者损失了8%的对抗准确率，而仅使用20%的数据，我们损失了14%的对抗准确率，并将运行时间减少了3倍。有趣的是，我们发现在某些情况下，数据剪枝带来了两方面的好处--它既提高了对抗准确率，又提高了训练时间。



## **28. Watermarking Graph Neural Networks based on Backdoor Attacks**

基于后门攻击的数字水印图神经网络 cs.LG

13 pages, 9 figures

**SubmitDate**: 2022-07-01    [paper-pdf](http://arxiv.org/pdf/2110.11024v3)

**Authors**: Jing Xu, Stjepan Picek

**Abstracts**: Graph Neural Networks (GNNs) have achieved promising performance in various real-world applications. Building a powerful GNN model is not a trivial task, as it requires a large amount of training data, powerful computing resources, and human expertise in fine-tuning the model. What is more, with the development of adversarial attacks, e.g., model stealing attacks, GNNs raise challenges to model authentication. To avoid copyright infringement on GNNs, it is necessary to verify the ownership of the GNN models.   In this paper, we present a watermarking framework for GNNs for both graph and node classification tasks. We 1) design two strategies to generate watermarked data for the graph classification task and one for the node classification task, 2) embed the watermark into the host model through training to obtain the watermarked GNN model, and 3) verify the ownership of the suspicious model in a black-box setting. The experiments show that our framework can verify the ownership of GNN models with a very high probability (around $95\%$) for both tasks. Finally, we experimentally show that our watermarking approach is robust against two model modifications and an input reformation defense against backdoor attacks.

摘要: 图神经网络(GNN)在各种实际应用中取得了良好的性能。构建一个强大的GNN模型不是一项简单的任务，因为它需要大量的训练数据、强大的计算资源和微调模型的人力专业知识。更重要的是，随着敌意攻击的发展，例如模型窃取攻击，GNN对模型认证提出了挑战。为了避免在GNN上侵犯版权，有必要核实GNN模型的所有权。在这篇文章中，我们提出了一个适用于图和节点分类任务的GNN数字水印框架。我们设计了两种策略来为图分类任务和节点分类任务生成水印数据，2)通过训练将水印嵌入到宿主模型中，得到带水印的GNN模型，3)在黑盒环境下验证可疑模型的所有权。实验表明，我们的框架能够以很高的概率(约95美元)验证这两个任务的GNN模型的所有权。最后，我们的实验表明，我们的水印方法对两次模型修改和一次输入重构防御后门攻击都是健壮的。



## **29. More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks**

越多越好(多数)：联邦图神经网络中的后门攻击 cs.CR

15 pages, 15 figures

**SubmitDate**: 2022-07-01    [paper-pdf](http://arxiv.org/pdf/2202.03195v3)

**Authors**: Jing Xu, Rui Wang, Stefanos Koffas, Kaitai Liang, Stjepan Picek

**Abstracts**: Graph Neural Networks (GNNs) are a class of deep learning-based methods for processing graph domain information. GNNs have recently become a widely used graph analysis method due to their superior ability to learn representations for complex graph data. However, due to privacy concerns and regulation restrictions, centralized GNNs can be difficult to apply to data-sensitive scenarios. Federated learning (FL) is an emerging technology developed for privacy-preserving settings when several parties need to train a shared global model collaboratively. Although several research works have applied FL to train GNNs (Federated GNNs), there is no research on their robustness to backdoor attacks.   This paper bridges this gap by conducting two types of backdoor attacks in Federated GNNs: centralized backdoor attacks (CBA) and distributed backdoor attacks (DBA). Our experiments show that the DBA attack success rate is higher than CBA in almost all evaluated cases. For CBA, the attack success rate of all local triggers is similar to the global trigger even if the training set of the adversarial party is embedded with the global trigger. To further explore the properties of two backdoor attacks in Federated GNNs, we evaluate the attack performance for a different number of clients, trigger sizes, poisoning intensities, and trigger densities. Moreover, we explore the robustness of DBA and CBA against two state-of-the-art defenses. We find that both attacks are robust against the investigated defenses, necessitating the need to consider backdoor attacks in Federated GNNs as a novel threat that requires custom defenses.

摘要: 图神经网络是一类基于深度学习的图域信息处理方法。由于其优越的学习复杂图形数据表示的能力，GNN最近已成为一种广泛使用的图形分析方法。然而，由于隐私问题和监管限制，集中式GNN可能很难适用于数据敏感的情况。联合学习(FL)是一种新兴的技术，是为保护隐私而开发的，当多个参与方需要协作训练共享的全球模型时。虽然一些研究工作已经将FL用于训练GNN(Federated GNN)，但还没有关于其对后门攻击的健壮性的研究。本文通过在联邦GNN中实施两种类型的后门攻击来弥合这一差距：集中式后门攻击(CBA)和分布式后门攻击(DBA)。我们的实验表明，DBA攻击的成功率几乎在所有评估案例中都高于CBA。对于CBA，即使敌方的训练集嵌入了全局触发器，所有局部触发器的攻击成功率也与全局触发器相似。为了进一步探索联邦GNN中两种后门攻击的特性，我们评估了不同客户端数量、触发器大小、中毒强度和触发器密度下的攻击性能。此外，我们还探讨了DBA和CBA对两种最先进的防御措施的健壮性。我们发现，这两种攻击对所调查的防御都是健壮的，因此有必要将联邦GNN中的后门攻击视为一种需要自定义防御的新威胁。



## **30. Effect of Homomorphic Encryption on the Performance of Training Federated Learning Generative Adversarial Networks**

同态加密对训练联合学习产生式对抗网络性能的影响 cs.CR

**SubmitDate**: 2022-07-01    [paper-pdf](http://arxiv.org/pdf/2207.00263v1)

**Authors**: Ignjat Pejic, Rui Wang, Kaitai Liang

**Abstracts**: A Generative Adversarial Network (GAN) is a deep-learning generative model in the field of Machine Learning (ML) that involves training two Neural Networks (NN) using a sizable data set. In certain fields, such as medicine, the training data may be hospital patient records that are stored across different hospitals. The classic centralized approach would involve sending the data to a centralized server where the model would be trained. However, that would involve breaching the privacy and confidentiality of the patients and their data, which would be unacceptable. Therefore, Federated Learning (FL), an ML technique that trains ML models in a distributed setting without data ever leaving the host device, would be a better alternative to the centralized option. In this ML technique, only parameters and certain metadata would be communicated. In spite of that, there still exist attacks that can infer user data using the parameters and metadata. A fully privacy-preserving solution involves homomorphically encrypting (HE) the data communicated. This paper will focus on the performance loss of training an FL-GAN with three different types of Homomorphic Encryption: Partial Homomorphic Encryption (PHE), Somewhat Homomorphic Encryption (SHE), and Fully Homomorphic Encryption (FHE). We will also test the performance loss of Multi-Party Computations (MPC), as it has homomorphic properties. The performances will be compared to the performance of training an FL-GAN without encryption as well. Our experiments show that the more complex the encryption method is, the longer it takes, with the extra time taken for HE is quite significant in comparison to the base case of FL.

摘要: 产生式对抗性网络(GAN)是机器学习(ML)领域中的一种深度学习产生式模型，它涉及到使用相当大的数据集来训练两个神经网络(NN)。在某些领域，例如医学，训练数据可以是跨不同医院存储的医院患者记录。经典的集中式方法将涉及将数据发送到集中式服务器，在那里将训练模型。然而，这将涉及侵犯患者及其数据的隐私和机密性，这是不可接受的。因此，联合学习(FL)是一种ML技术，它在分布式环境中训练ML模型，而无需数据离开主机设备，将是集中式选项的更好替代方案。在这种ML技术中，只会传递参数和特定的元数据。尽管如此，仍然存在使用参数和元数据推断用户数据的攻击。完全保护隐私的解决方案涉及对通信的数据进行同态加密(HE)。本文将重点研究用三种不同类型的同态加密来训练FL-GaN的性能损失：部分同态加密(PHE)、部分同态加密(SHE)和完全同态加密(FHE)。我们还将测试多方计算(MPC)的性能损失，因为它具有同态性质。这些性能也将与训练未加密的FL-GaN的性能进行比较。我们的实验表明，加密方法越复杂，所花费的时间就越长，与FL的基本情况相比，HE所需的额外时间是相当显著的。



## **31. Threat Assessment in Machine Learning based Systems**

基于机器学习系统中的威胁评估 cs.CR

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2207.00091v1)

**Authors**: Lionel Nganyewou Tidjon, Foutse Khomh

**Abstracts**: Machine learning is a field of artificial intelligence (AI) that is becoming essential for several critical systems, making it a good target for threat actors. Threat actors exploit different Tactics, Techniques, and Procedures (TTPs) against the confidentiality, integrity, and availability of Machine Learning (ML) systems. During the ML cycle, they exploit adversarial TTPs to poison data and fool ML-based systems. In recent years, multiple security practices have been proposed for traditional systems but they are not enough to cope with the nature of ML-based systems. In this paper, we conduct an empirical study of threats reported against ML-based systems with the aim to understand and characterize the nature of ML threats and identify common mitigation strategies. The study is based on 89 real-world ML attack scenarios from the MITRE's ATLAS database, the AI Incident Database, and the literature; 854 ML repositories from the GitHub search and the Python Packaging Advisory database, selected based on their reputation. Attacks from the AI Incident Database and the literature are used to identify vulnerabilities and new types of threats that were not documented in ATLAS. Results show that convolutional neural networks were one of the most targeted models among the attack scenarios. ML repositories with the largest vulnerability prominence include TensorFlow, OpenCV, and Notebook. In this paper, we also report the most frequent vulnerabilities in the studied ML repositories, the most targeted ML phases and models, the most used TTPs in ML phases and attack scenarios. This information is particularly important for red/blue teams to better conduct attacks/defenses, for practitioners to prevent threats during ML development, and for researchers to develop efficient defense mechanisms.

摘要: 机器学习是人工智能(AI)的一个领域，对于几个关键系统来说正变得至关重要，使其成为威胁参与者的良好目标。威胁参与者利用不同的策略、技术和过程(TTP)来攻击机器学习(ML)系统的机密性、完整性和可用性。在ML周期中，他们利用敌意TTP来毒化数据并愚弄基于ML的系统。近年来，针对传统系统提出了多种安全措施，但它们不足以应对基于ML的系统的本质。在本文中，我们对基于ML的系统报告的威胁进行了实证研究，目的是了解和表征ML威胁的性质，并确定常见的缓解策略。这项研究基于MITRE的ATLAS数据库、AI事件数据库和文献中的89个真实世界ML攻击场景；GitHub搜索和Python打包咨询数据库中的854个ML存储库，根据它们的声誉进行选择。来自AI事件数据库和文献的攻击被用来识别ATLAS中没有记录的漏洞和新类型的威胁。结果表明，卷积神经网络是攻击场景中最具针对性的模型之一。漏洞最突出的ML存储库包括TensorFlow、OpenCV和Notebook。在本文中，我们还报告了所研究的ML库中最常见的漏洞、最有针对性的ML阶段和模型、ML阶段中最常用的TTP以及攻击场景。这些信息对于红/蓝团队更好地进行攻击/防御，对于实践者在ML开发过程中防止威胁，对于研究人员开发高效的防御机制尤为重要。



## **32. MEAD: A Multi-Armed Approach for Evaluation of Adversarial Examples Detectors**

Mead：一种评估对抗性范例检测器的多臂方法 cs.CV

This paper has been accepted to appear in the Proceedings of the 2022  European Conference on Machine Learning and Data Mining (ECML-PKDD), 19th to  the 23rd of September, Grenoble, France

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2206.15415v1)

**Authors**: Federica Granese, Marine Picot, Marco Romanelli, Francisco Messina, Pablo Piantanida

**Abstracts**: Detection of adversarial examples has been a hot topic in the last years due to its importance for safely deploying machine learning algorithms in critical applications. However, the detection methods are generally validated by assuming a single implicitly known attack strategy, which does not necessarily account for real-life threats. Indeed, this can lead to an overoptimistic assessment of the detectors' performance and may induce some bias in the comparison between competing detection schemes. We propose a novel multi-armed framework, called MEAD, for evaluating detectors based on several attack strategies to overcome this limitation. Among them, we make use of three new objectives to generate attacks. The proposed performance metric is based on the worst-case scenario: detection is successful if and only if all different attacks are correctly recognized. Empirically, we show the effectiveness of our approach. Moreover, the poor performance obtained for state-of-the-art detectors opens a new exciting line of research.

摘要: 对抗性样本的检测是近年来的一个热门话题，因为它对于在关键应用中安全地部署机器学习算法具有重要意义。然而，检测方法通常是通过假设单个隐式已知的攻击策略来验证的，这不一定考虑现实生活中的威胁。事实上，这可能会导致对检测器性能的过度乐观评估，并可能在相互竞争的检测方案之间的比较中导致一些偏见。为了克服这一局限性，我们提出了一种新的多臂框架，称为MEAD，用于基于几种攻击策略来评估检测器。其中，我们利用三个新的目标来产生攻击。建议的性能指标基于最坏的情况：当且仅当正确识别所有不同的攻击时，检测才成功。在经验上，我们展示了我们方法的有效性。此外，最先进的探测器获得的糟糕性能开启了一条新的令人兴奋的研究路线。



## **33. The Topological BERT: Transforming Attention into Topology for Natural Language Processing**

拓扑学BERT：将注意力转化为自然语言处理的拓扑学 cs.CL

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2206.15195v1)

**Authors**: Ilan Perez, Raphael Reinauer

**Abstracts**: In recent years, the introduction of the Transformer models sparked a revolution in natural language processing (NLP). BERT was one of the first text encoders using only the attention mechanism without any recurrent parts to achieve state-of-the-art results on many NLP tasks.   This paper introduces a text classifier using topological data analysis. We use BERT's attention maps transformed into attention graphs as the only input to that classifier. The model can solve tasks such as distinguishing spam from ham messages, recognizing whether a sentence is grammatically correct, or evaluating a movie review as negative or positive. It performs comparably to the BERT baseline and outperforms it on some tasks.   Additionally, we propose a new method to reduce the number of BERT's attention heads considered by the topological classifier, which allows us to prune the number of heads from 144 down to as few as ten with no reduction in performance. Our work also shows that the topological model displays higher robustness against adversarial attacks than the original BERT model, which is maintained during the pruning process. To the best of our knowledge, this work is the first to confront topological-based models with adversarial attacks in the context of NLP.

摘要: 近年来，Transformer模型的引入引发了自然语言处理(NLP)的革命。Bert是第一批只使用注意力机制而不使用任何重复部分的文本编码者之一，以在许多NLP任务中获得最先进的结果。本文介绍了一种基于拓扑数据分析的文本分类器。我们使用Bert转换为注意图的注意图作为该分类器的唯一输入。该模型可以解决一些任务，比如区分垃圾邮件和垃圾邮件，识别句子的语法是否正确，或者评估电影评论是负面的还是正面的。它的表现与BERT基线相当，并在某些任务上超过它。此外，我们还提出了一种新的方法来减少拓扑分类器所考虑的BERT注意头数，该方法允许我们在不降低性能的情况下将注意头数从144个减少到10个。我们的工作还表明，与在剪枝过程中保持的原始BERT模型相比，该拓扑模型对敌意攻击表现出更高的稳健性。据我们所知，这是第一个在NLP环境下对抗基于拓扑模型的攻击的工作。



## **34. Reliable Representations Make A Stronger Defender: Unsupervised Structure Refinement for Robust GNN**

可靠的表示使防御者更强大：健壮GNN的无监督结构求精 cs.LG

Accepted in KDD2022

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2207.00012v1)

**Authors**: Kuan Li, Yang Liu, Xiang Ao, Jianfeng Chi, Jinghua Feng, Hao Yang, Qing He

**Abstracts**: Benefiting from the message passing mechanism, Graph Neural Networks (GNNs) have been successful on flourish tasks over graph data. However, recent studies have shown that attackers can catastrophically degrade the performance of GNNs by maliciously modifying the graph structure. A straightforward solution to remedy this issue is to model the edge weights by learning a metric function between pairwise representations of two end nodes, which attempts to assign low weights to adversarial edges. The existing methods use either raw features or representations learned by supervised GNNs to model the edge weights. However, both strategies are faced with some immediate problems: raw features cannot represent various properties of nodes (e.g., structure information), and representations learned by supervised GNN may suffer from the poor performance of the classifier on the poisoned graph. We need representations that carry both feature information and as mush correct structure information as possible and are insensitive to structural perturbations. To this end, we propose an unsupervised pipeline, named STABLE, to optimize the graph structure. Finally, we input the well-refined graph into a downstream classifier. For this part, we design an advanced GCN that significantly enhances the robustness of vanilla GCN without increasing the time complexity. Extensive experiments on four real-world graph benchmarks demonstrate that STABLE outperforms the state-of-the-art methods and successfully defends against various attacks.

摘要: 得益于消息传递机制，图神经网络(GNN)已经成功地处理了大量的图数据任务。然而，最近的研究表明，攻击者可以通过恶意修改图结构来灾难性地降低GNN的性能。解决这一问题的一个直接解决方案是通过学习两个末端节点的成对表示之间的度量函数来对边权重进行建模，该度量函数试图为对抗性边分配较低的权重。现有的方法要么使用原始特征，要么使用由监督GNN学习的表示来对边权重进行建模。然而，这两种策略都面临着一些迫在眉睫的问题：原始特征不能表示节点的各种属性(例如结构信息)，而有监督GNN学习的表示可能会受到有毒图上分类器性能较差的影响。我们需要既携带特征信息又尽可能正确的结构信息并对结构扰动不敏感的表示法。为此，我们提出了一种名为STRATE的无监督流水线来优化图的结构。最后，我们将精化后的图输入到下游分类器中。对于这一部分，我们设计了一种改进的GCN，它在不增加时间复杂度的情况下显著增强了普通GCN的健壮性。在四个真实图形基准上的大量实验表明，STRATE的性能优于最先进的方法，并成功地防御了各种攻击。



## **35. An Intermediate-level Attack Framework on The Basis of Linear Regression**

一种基于线性回归的中级攻击框架 cs.CV

Accepted by TPAMI; Code is available at  https://github.com/qizhangli/ila-plus-plus-lr

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2203.10723v2)

**Authors**: Yiwen Guo, Qizhang Li, Wangmeng Zuo, Hao Chen

**Abstracts**: This paper substantially extends our work published at ECCV, in which an intermediate-level attack was proposed to improve the transferability of some baseline adversarial examples. Specifically, we advocate a framework in which a direct linear mapping from the intermediate-level discrepancies (between adversarial features and benign features) to prediction loss of the adversarial example is established. By delving deep into the core components of such a framework, we show that 1) a variety of linear regression models can all be considered in order to establish the mapping, 2) the magnitude of the finally obtained intermediate-level adversarial discrepancy is correlated with the transferability, 3) further boost of the performance can be achieved by performing multiple runs of the baseline attack with random initialization. In addition, by leveraging these findings, we achieve new state-of-the-arts on transfer-based $\ell_\infty$ and $\ell_2$ attacks. Our code is publicly available at https://github.com/qizhangli/ila-plus-plus-lr.

摘要: 本文大大扩展了我们在ECCV上发表的工作，在该工作中，提出了一种中级攻击来提高一些基线对手例子的可转移性。具体地说，我们主张建立一个框架，在这个框架中，建立从对抗性例子的中间级差异(对抗性特征和良性特征之间)到预测损失的直接线性映射。通过深入研究该框架的核心部分，我们发现：1)为了建立映射，可以考虑多种线性回归模型；2)最终获得的中级敌方差异的大小与可转移性相关；3)通过随机初始化执行多次基线攻击，可以进一步提高性能。此外，通过利用这些发现，我们实现了针对基于传输的$\ell_\inty$和$\ell_2$攻击的新技术。我们的代码在https://github.com/qizhangli/ila-plus-plus-lr.上公开提供



## **36. On the Challenges of Detecting Side-Channel Attacks in SGX**

关于在SGX中检测旁路攻击的挑战 cs.CR

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2011.14599v2)

**Authors**: Jianyu Jiang, Claudio Soriente, Ghassan Karame

**Abstracts**: Existing tools to detect side-channel attacks on Intel SGX are grounded on the observation that attacks affect the performance of the victim application. As such, all detection tools monitor the potential victim and raise an alarm if the witnessed performance (in terms of runtime, enclave interruptions, cache misses, etc.) is out of the ordinary.   In this paper, we show that monitoring the performance of enclaves to detect side-channel attacks may not be effective. Our core intuition is that all monitoring tools are geared towards an adversary that interferes with the victim's execution in order to extract the most number of secret bits (e.g., the entire secret) in one or few runs. They cannot, however, detect an adversary that leaks smaller portions of the secret - as small as a single bit - at each execution of the victim. In particular, by minimizing the information leaked at each run, the impact of any side-channel attack on the application's performance is significantly lowered - ensuring that the detection tool does not detect an attack. By repeating the attack multiple times, each time on a different part of the secret, the adversary can recover the whole secret and remain undetected. Based on this intuition, we adapt known attacks leveraging page-tables and L3 cache to bypass existing detection mechanisms. We show experimentally how an attacker can successfully exfiltrate the secret key used in an enclave running various cryptographic routines of libgcrypt. Beyond cryptographic libraries, we also show how to compromise the predictions of enclaves running decision-tree routines of OpenCV. Our evaluation results suggest that performance-based detection tools do not deter side-channel attacks on SGX enclaves and that effective detection mechanisms are yet to be designed.

摘要: 现有工具用于检测针对Intel SGX的旁路攻击，其基础是观察到攻击会影响受攻击应用程序的性能。因此，所有检测工具都会监视潜在受害者，并在发现性能(在运行时、飞地中断、缓存未命中等方面)时发出警报是不寻常的。在本文中，我们表明，通过监控Enclaves的性能来检测旁路攻击可能并不有效。我们的核心直觉是，所有监控工具都是针对干扰受害者执行的对手，以便在一次或几次运行中提取最多数量的秘密比特(例如，整个秘密)。然而，他们无法检测到在每次处决受害者时泄露较小部分秘密的对手。特别是，通过最大限度地减少每次运行时泄漏的信息，任何侧通道攻击对应用程序性能的影响都会显著降低，从而确保检测工具不会检测到攻击。通过多次重复攻击，每次对秘密的不同部分进行攻击，攻击者可以恢复整个秘密并保持不被发现。基于这一直觉，我们采用了利用页表和L3缓存的已知攻击来绕过现有的检测机制。我们通过实验展示了攻击者如何成功地渗出在运行各种libgcrypt加密例程的飞地中使用的秘密密钥。除了密码库之外，我们还展示了如何折衷运行OpenCV决策树例程的Enclaves的预测。我们的评估结果表明，基于性能的检测工具不能阻止对SGX飞地的旁路攻击，并且还没有设计有效的检测机制。



## **37. Depth-2 Neural Networks Under a Data-Poisoning Attack**

数据中毒攻击下的深度-2神经网络 cs.LG

32 page, 7 figures

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2005.01699v3)

**Authors**: Sayar Karmakar, Anirbit Mukherjee, Theodore Papamarkou

**Abstracts**: In this work, we study the possibility of defending against data-poisoning attacks while training a shallow neural network in a regression setup. We focus on doing supervised learning for a class of depth-2 finite-width neural networks, which includes single-filter convolutional networks. In this class of networks, we attempt to learn the network weights in the presence of a malicious oracle doing stochastic, bounded and additive adversarial distortions on the true output during training. For the non-gradient stochastic algorithm that we construct, we prove worst-case near-optimal trade-offs among the magnitude of the adversarial attack, the weight approximation accuracy, and the confidence achieved by the proposed algorithm. As our algorithm uses mini-batching, we analyze how the mini-batch size affects convergence. We also show how to utilize the scaling of the outer layer weights to counter output-poisoning attacks depending on the probability of attack. Lastly, we give experimental evidence demonstrating how our algorithm outperforms stochastic gradient descent under different input data distributions, including instances of heavy-tailed distributions.

摘要: 在这项工作中，我们研究了在回归设置中训练浅层神经网络的同时防御数据中毒攻击的可能性。重点研究了一类深度为2的有限宽度神经网络的监督学习问题，其中包括单滤波卷积网络。在这类网络中，我们试图在恶意预言存在的情况下学习网络权重，该预言在训练期间对真实输出进行随机的、有界的和相加的对抗性扭曲。对于我们构造的非梯度随机算法，我们证明了该算法在对抗性攻击的强度、权重逼近精度和所获得的置信度之间的最坏情况下的近优折衷。由于我们的算法使用了小批量，我们分析了小批量大小对收敛的影响。我们还展示了如何利用外层权重的比例来对抗依赖于攻击概率的输出中毒攻击。最后，我们给出了实验证据，展示了在不同的输入数据分布下，包括重尾分布的情况下，我们的算法的性能如何优于随机梯度下降。



## **38. IBP Regularization for Verified Adversarial Robustness via Branch-and-Bound**

基于分枝定界的IBP正则化算法 cs.LG

ICML 2022 Workshop on Formal Verification of Machine Learning

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2206.14772v1)

**Authors**: Alessandro De Palma, Rudy Bunel, Krishnamurthy Dvijotham, M. Pawan Kumar, Robert Stanforth

**Abstracts**: Recent works have tried to increase the verifiability of adversarially trained networks by running the attacks over domains larger than the original perturbations and adding various regularization terms to the objective. However, these algorithms either underperform or require complex and expensive stage-wise training procedures, hindering their practical applicability. We present IBP-R, a novel verified training algorithm that is both simple and effective. IBP-R induces network verifiability by coupling adversarial attacks on enlarged domains with a regularization term, based on inexpensive interval bound propagation, that minimizes the gap between the non-convex verification problem and its approximations. By leveraging recent branch-and-bound frameworks, we show that IBP-R obtains state-of-the-art verified robustness-accuracy trade-offs for small perturbations on CIFAR-10 while training significantly faster than relevant previous work. Additionally, we present UPB, a novel branching strategy that, relying on a simple heuristic based on $\beta$-CROWN, reduces the cost of state-of-the-art branching algorithms while yielding splits of comparable quality.

摘要: 最近的工作试图通过在比原始扰动更大的域上运行攻击并在目标中添加各种正则化项来增加恶意训练网络的可验证性。然而，这些算法要么表现不佳，要么需要复杂而昂贵的阶段性训练过程，从而阻碍了它们的实际适用性。提出了一种简单有效的新的验证训练算法IBP-R。IBP-R通过将扩展域上的敌意攻击与基于廉价区间界传播的正则化项相结合来诱导网络可验证性，从而最小化非凸验证问题与其近似问题之间的差距。通过利用最近的分支定界框架，我们表明IBP-R在CIFAR-10上的小扰动下获得了经过验证的最先进的健壮性和准确性折衷，同时训练速度比相关以前的工作快得多。此外，我们提出了一种新的分支策略UPB，它依赖于基于$\beta$-Crown的简单启发式算法，在产生类似质量的分裂的同时，降低了最先进的分支算法的成本。



## **39. longhorns at DADC 2022: How many linguists does it take to fool a Question Answering model? A systematic approach to adversarial attacks**

DADC 2022上的长角人：需要多少语言学家才能愚弄一个问题回答模型？应对对抗性攻击的系统方法 cs.CL

Accepted at DADC2022

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2206.14729v1)

**Authors**: Venelin Kovatchev, Trina Chatterjee, Venkata S Govindarajan, Jifan Chen, Eunsol Choi, Gabriella Chronis, Anubrata Das, Katrin Erk, Matthew Lease, Junyi Jessy Li, Yating Wu, Kyle Mahowald

**Abstracts**: Developing methods to adversarially challenge NLP systems is a promising avenue for improving both model performance and interpretability. Here, we describe the approach of the team "longhorns" on Task 1 of the The First Workshop on Dynamic Adversarial Data Collection (DADC), which asked teams to manually fool a model on an Extractive Question Answering task. Our team finished first, with a model error rate of 62%. We advocate for a systematic, linguistically informed approach to formulating adversarial questions, and we describe the results of our pilot experiments, as well as our official submission.

摘要: 开发反挑战NLP系统的方法是提高模型性能和可解释性的一条很有前途的途径。在这里，我们描述了“长角人”团队在第一次动态对手数据收集(DADC)研讨会(DADC)的任务1上的方法，该方法要求团队在提取问答任务中手动愚弄模型。我们的团队以62%的模型错误率获得第一名。我们提倡一种系统的、在语言上知情的方法来提出对抗性的问题，我们描述了我们的试点实验的结果，以及我们的正式提交。



## **40. Private Graph Extraction via Feature Explanations**

基于特征解释的专用图提取 cs.LG

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2206.14724v1)

**Authors**: Iyiola E. Olatunji, Mandeep Rathee, Thorben Funke, Megha Khosla

**Abstracts**: Privacy and interpretability are two of the important ingredients for achieving trustworthy machine learning. We study the interplay of these two aspects in graph machine learning through graph reconstruction attacks. The goal of the adversary here is to reconstruct the graph structure of the training data given access to model explanations. Based on the different kinds of auxiliary information available to the adversary, we propose several graph reconstruction attacks. We show that additional knowledge of post-hoc feature explanations substantially increases the success rate of these attacks. Further, we investigate in detail the differences between attack performance with respect to three different classes of explanation methods for graph neural networks: gradient-based, perturbation-based, and surrogate model-based methods. While gradient-based explanations reveal the most in terms of the graph structure, we find that these explanations do not always score high in utility. For the other two classes of explanations, privacy leakage increases with an increase in explanation utility. Finally, we propose a defense based on a randomized response mechanism for releasing the explanations which substantially reduces the attack success rate. Our anonymized code is available.

摘要: 隐私和可解释性是实现可信机器学习的两个重要因素。我们通过图重构攻击来研究这两个方面在图机器学习中的相互作用。在这里，对手的目标是在获得模型解释的情况下重建训练数据的图形结构。基于敌手可获得的各种辅助信息，我们提出了几种图重构攻击。我们表明，额外的事后特征解释知识大大提高了这些攻击的成功率。此外，我们还详细研究了图神经网络的三种不同解释方法：基于梯度、基于扰动和基于代理模型的解释方法在攻击性能上的差异。虽然基于梯度的解释在图表结构方面揭示了最多，但我们发现这些解释并不总是在实用方面得分很高。对于其他两类解释，隐私泄露随着解释效用的增加而增加。最后，我们提出了一种基于随机化响应机制的防御机制，用于发布解释，大大降低了攻击成功率。我们的匿名码是可用的。



## **41. Enhancing Security of Memristor Computing System Through Secure Weight Mapping**

通过安全权重映射提高忆阻器计算系统的安全性 cs.ET

6 pages, 4 figures, accepted by IEEE ISVLSI 2022

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2206.14498v1)

**Authors**: Minhui Zou, Junlong Zhou, Xiaotong Cui, Wei Wang, Shahar Kvatinsky

**Abstracts**: Emerging memristor computing systems have demonstrated great promise in improving the energy efficiency of neural network (NN) algorithms. The NN weights stored in memristor crossbars, however, may face potential theft attacks due to the nonvolatility of the memristor devices. In this paper, we propose to protect the NN weights by mapping selected columns of them in the form of 1's complements and leaving the other columns in their original form, preventing the adversary from knowing the exact representation of each weight. The results show that compared with prior work, our method achieves effectiveness comparable to the best of them and reduces the hardware overhead by more than 18X.

摘要: 新兴的忆阻器计算系统在提高神经网络(NN)算法的能量效率方面显示出巨大的前景。然而，由于忆阻器器件的非易失性，存储在忆阻器纵横杆中的NN权重可能面临潜在的盗窃攻击。在本文中，我们建议通过以1的补码的形式映射选定的列来保护NN权重，而将其他列保持其原始形式，以防止对手知道每个权重的准确表示。实验结果表明，与前人的工作相比，我们的方法取得了与之相当的效果，硬件开销减少了18倍以上。



## **42. Guided Diffusion Model for Adversarial Purification**

对抗性净化中的引导扩散模型 cs.CV

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2205.14969v3)

**Authors**: Jinyi Wang, Zhaoyang Lyu, Dahua Lin, Bo Dai, Hongfei Fu

**Abstracts**: With wider application of deep neural networks (DNNs) in various algorithms and frameworks, security threats have become one of the concerns. Adversarial attacks disturb DNN-based image classifiers, in which attackers can intentionally add imperceptible adversarial perturbations on input images to fool the classifiers. In this paper, we propose a novel purification approach, referred to as guided diffusion model for purification (GDMP), to help protect classifiers from adversarial attacks. The core of our approach is to embed purification into the diffusion denoising process of a Denoised Diffusion Probabilistic Model (DDPM), so that its diffusion process could submerge the adversarial perturbations with gradually added Gaussian noises, and both of these noises can be simultaneously removed following a guided denoising process. On our comprehensive experiments across various datasets, the proposed GDMP is shown to reduce the perturbations raised by adversarial attacks to a shallow range, thereby significantly improving the correctness of classification. GDMP improves the robust accuracy by 5%, obtaining 90.1% under PGD attack on the CIFAR10 dataset. Moreover, GDMP achieves 70.94% robustness on the challenging ImageNet dataset.

摘要: 随着深度神经网络(DNN)在各种算法和框架中的广泛应用，安全威胁已成为人们关注的问题之一。对抗性攻击干扰了基于DNN的图像分类器，攻击者可以故意在输入图像上添加不可察觉的对抗性扰动来愚弄分类器。在本文中，我们提出了一种新的净化方法，称为引导扩散净化模型(GDMP)，以帮助保护分类器免受对手攻击。该方法的核心是将净化嵌入到去噪扩散概率模型(DDPM)的扩散去噪过程中，使其扩散过程能够淹没带有逐渐增加的高斯噪声的对抗性扰动，并在引导去噪过程后同时去除这两种噪声。在不同数据集上的综合实验表明，所提出的GDMP将对抗性攻击引起的扰动减少到较小的范围，从而显著提高了分类的正确性。GDMP在CIFAR10数据集上的稳健准确率提高了5%，在PGD攻击下达到了90.1%。此外，GDMP在具有挑战性的ImageNet数据集上获得了70.94%的健壮性。



## **43. A Deep Learning Approach to Create DNS Amplification Attacks**

一种创建域名系统放大攻击的深度学习方法 cs.CR

12 pages, 6 figures, Conference: to 2022 4th International Conference  on Management Science and Industrial Engineering (MSIE) (MSIE 2022), DOI:  https://doi.org/10.1145/3535782.3535838, accepted to conference above, not  yet published

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2206.14346v1)

**Authors**: Jared Mathews, Prosenjit Chatterjee, Shankar Banik, Cory Nance

**Abstracts**: In recent years, deep learning has shown itself to be an incredibly valuable tool in cybersecurity as it helps network intrusion detection systems to classify attacks and detect new ones. Adversarial learning is the process of utilizing machine learning to generate a perturbed set of inputs to then feed to the neural network to misclassify it. Much of the current work in the field of adversarial learning has been conducted in image processing and natural language processing with a wide variety of algorithms. Two algorithms of interest are the Elastic-Net Attack on Deep Neural Networks and TextAttack. In our experiment the EAD and TextAttack algorithms are applied to a Domain Name System amplification classifier. The algorithms are used to generate malicious Distributed Denial of Service adversarial examples to then feed as inputs to the network intrusion detection systems neural network to classify as valid traffic. We show in this work that both image processing and natural language processing adversarial learning algorithms can be applied against a network intrusion detection neural network.

摘要: 近年来，深度学习已被证明是网络安全中一个极其有价值的工具，因为它有助于网络入侵检测系统对攻击进行分类并检测新的攻击。对抗性学习是利用机器学习生成一组扰动的输入，然后馈送到神经网络进行错误分类的过程。目前在对抗性学习领域的许多工作都是在图像处理和自然语言处理方面进行的，并使用了各种算法。两个有趣的算法是对深度神经网络的弹性网络攻击和TextAttack。在我们的实验中，我们将EAD和TextAttack算法应用于域名系统放大分类器。这些算法被用来生成恶意的分布式拒绝服务攻击实例，然后将其作为输入输入到网络入侵检测系统的神经网络中，以分类为有效流量。在这项工作中，我们证明了图像处理和自然语言处理对抗性学习算法都可以应用于网络入侵检测神经网络。



## **44. Linear Model Against Malicious Adversaries with Local Differential Privacy**

基于局部差分隐私的对抗恶意攻击的线性模型 cs.CR

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2202.02448v2)

**Authors**: Guanhong Miao, A. Adam Ding, Samuel S. Wu

**Abstracts**: Scientific collaborations benefit from collaborative learning of distributed sources, but remain difficult to achieve when data are sensitive. In recent years, privacy preserving techniques have been widely studied to analyze distributed data across different agencies while protecting sensitive information. Most existing privacy preserving techniques are designed to resist semi-honest adversaries and require intense computation to perform data analysis. Secure collaborative learning is significantly difficult with the presence of malicious adversaries who may deviates from the secure protocol. Another challenge is to maintain high computation efficiency with privacy protection. In this paper, matrix encryption is applied to encrypt data such that the secure schemes are against malicious adversaries, including chosen plaintext attack, known plaintext attack, and collusion attack. The encryption scheme also achieves local differential privacy. Moreover, cross validation is studied to prevent overfitting without additional communication cost. Empirical experiments on real-world datasets demonstrate that the proposed schemes are computationally efficient compared to existing techniques against malicious adversary and semi-honest model.

摘要: 科学协作受益于分布式来源的协作学习，但在数据敏感时仍难以实现。近年来，隐私保护技术被广泛研究，以在保护敏感信息的同时分析跨不同机构的分布式数据。大多数现有的隐私保护技术都是为了抵抗半诚实的对手而设计的，并且需要大量的计算来执行数据分析。在存在可能偏离安全协议的恶意攻击者的情况下，安全协作学习非常困难。另一个挑战是在隐私保护的情况下保持高计算效率。本文采用矩阵加密的方法对数据进行加密，使安全方案能够抵抗选择明文攻击、已知明文攻击和合谋攻击等恶意攻击。该加密方案还实现了局部差分保密。此外，为了在不增加通信成本的情况下防止过拟合，还研究了交叉验证。在真实数据集上的实验表明，与已有的对抗恶意攻击和半诚实模型的技术相比，所提出的方案具有较高的计算效率。



## **45. An Empirical Study of Challenges in Converting Deep Learning Models**

深度学习模式转换挑战的实证研究 cs.LG

Accepted for publication in ICSME 2022

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2206.14322v1)

**Authors**: Moses Openja, Amin Nikanjam, Ahmed Haj Yahmed, Foutse Khomh, Zhen Ming, Jiang

**Abstracts**: There is an increase in deploying Deep Learning (DL)-based software systems in real-world applications. Usually DL models are developed and trained using DL frameworks that have their own internal mechanisms/formats to represent and train DL models, and usually those formats cannot be recognized by other frameworks. Moreover, trained models are usually deployed in environments different from where they were developed. To solve the interoperability issue and make DL models compatible with different frameworks/environments, some exchange formats are introduced for DL models, like ONNX and CoreML. However, ONNX and CoreML were never empirically evaluated by the community to reveal their prediction accuracy, performance, and robustness after conversion. Poor accuracy or non-robust behavior of converted models may lead to poor quality of deployed DL-based software systems. We conduct, in this paper, the first empirical study to assess ONNX and CoreML for converting trained DL models. In our systematic approach, two popular DL frameworks, Keras and PyTorch, are used to train five widely used DL models on three popular datasets. The trained models are then converted to ONNX and CoreML and transferred to two runtime environments designated for such formats, to be evaluated. We investigate the prediction accuracy before and after conversion. Our results unveil that the prediction accuracy of converted models are at the same level of originals. The performance (time cost and memory consumption) of converted models are studied as well. The size of models are reduced after conversion, which can result in optimized DL-based software deployment. Converted models are generally assessed as robust at the same level of originals. However, obtained results show that CoreML models are more vulnerable to adversarial attacks compared to ONNX.

摘要: 在现实世界的应用程序中部署基于深度学习(DL)的软件系统的情况越来越多。通常，使用具有自己的内部机制/格式来表示和训练DL模型的DL框架来开发和训练DL模型，并且通常这些格式不能被其他框架识别。此外，经过训练的模型通常部署在与开发环境不同的环境中。为了解决互操作问题，使DL模型与不同的框架/环境兼容，引入了一些用于DL模型的交换格式，如ONNX和CoreML。然而，ONNX和CoreML从未得到社区的经验性评估，以揭示它们在转换后的预测准确性、性能和稳健性。转换后模型的准确性或非健壮性可能会导致已部署的基于DL的软件系统的质量较差。在本文中，我们进行了第一次实证研究，以评估ONNX和CoreML用于转换训练的DL模型的能力。在我们的系统方法中，两个流行的DL框架Kera和PyTorch被用来在三个流行的数据集上训练五个广泛使用的DL模型。然后，训练的模型被转换为ONNX和CoreML，并被传输到为这些格式指定的两个运行时环境，以进行评估。我们考察了转换前后的预测精度。我们的结果表明，转换后的模型的预测精度与原始模型相同。并对转换后模型的性能(时间开销和内存消耗)进行了研究。转换后模型的大小会减小，从而可以优化基于DL的软件部署。转换后的模型通常被评估为与原始模型具有相同水平的健壮性。然而，研究结果表明，与ONNX相比，CoreML模型更容易受到敌意攻击。



## **46. Collecting high-quality adversarial data for machine reading comprehension tasks with humans and models in the loop**

为机器阅读理解任务收集高质量的对抗性数据，其中人和模型处于循环中 cs.CL

8 pages, 3 figures, for more information about the shared task please  go to https://dadcworkshop.github.io/

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2206.14272v1)

**Authors**: Damian Y. Romero Diaz, Magdalena Anioł, John Culnan

**Abstracts**: We present our experience as annotators in the creation of high-quality, adversarial machine-reading-comprehension data for extractive QA for Task 1 of the First Workshop on Dynamic Adversarial Data Collection (DADC). DADC is an emergent data collection paradigm with both models and humans in the loop. We set up a quasi-experimental annotation design and perform quantitative analyses across groups with different numbers of annotators focusing on successful adversarial attacks, cost analysis, and annotator confidence correlation. We further perform a qualitative analysis of our perceived difficulty of the task given the different topics of the passages in our dataset and conclude with recommendations and suggestions that might be of value to people working on future DADC tasks and related annotation interfaces.

摘要: 我们介绍了我们作为注释员在创建高质量的对抗性机器阅读理解数据方面的经验，这些数据用于第一次动态对抗性数据收集(DADC)研讨会的任务1的摘录QA。DADC是一种模型和人类都在循环中的紧急数据收集范例。我们建立了一个准实验性的标注设计，并对不同数量的注释者进行了量化分析，重点是成功的对抗性攻击、代价分析和注释者的置信度相关性。在给定数据集中段落的不同主题的情况下，我们进一步对任务的感知难度进行了定性分析，并提出了可能对从事未来DADC任务和相关注释接口工作的人有价值的建议和建议。



## **47. How to Steer Your Adversary: Targeted and Efficient Model Stealing Defenses with Gradient Redirection**

如何引导你的对手：有针对性和高效的模型窃取防御和渐变重定向 cs.LG

ICML 2022

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2206.14157v1)

**Authors**: Mantas Mazeika, Bo Li, David Forsyth

**Abstracts**: Model stealing attacks present a dilemma for public machine learning APIs. To protect financial investments, companies may be forced to withhold important information about their models that could facilitate theft, including uncertainty estimates and prediction explanations. This compromise is harmful not only to users but also to external transparency. Model stealing defenses seek to resolve this dilemma by making models harder to steal while preserving utility for benign users. However, existing defenses have poor performance in practice, either requiring enormous computational overheads or severe utility trade-offs. To meet these challenges, we present a new approach to model stealing defenses called gradient redirection. At the core of our approach is a provably optimal, efficient algorithm for steering an adversary's training updates in a targeted manner. Combined with improvements to surrogate networks and a novel coordinated defense strategy, our gradient redirection defense, called GRAD${}^2$, achieves small utility trade-offs and low computational overhead, outperforming the best prior defenses. Moreover, we demonstrate how gradient redirection enables reprogramming the adversary with arbitrary behavior, which we hope will foster work on new avenues of defense.

摘要: 模型窃取攻击给公共机器学习API带来了两难境地。为了保护金融投资，公司可能会被迫隐瞒有关其模型的重要信息，这些信息可能会为盗窃提供便利，包括不确定性估计和预测解释。这种妥协不仅损害了用户，也损害了外部透明度。模型窃取防御试图通过使模型更难被窃取，同时保护良性用户的实用性来解决这一困境。然而，现有的防御在实践中表现不佳，要么需要巨大的计算开销，要么需要严重的实用权衡。为了应对这些挑战，我们提出了一种新的方法来模拟窃取防御，称为梯度重定向。我们方法的核心是一种可证明是最优的、高效的算法，用于以有针对性的方式引导对手的训练更新。结合对代理网络的改进和一种新的协调防御策略，我们的梯度重定向防御，称为Grad$^2$，实现了小的效用权衡和较低的计算开销，性能优于最好的先前防御。此外，我们还演示了梯度重定向如何使用任意行为对对手进行重新编程，我们希望这将促进新防御途径的工作。



## **48. Debiasing Learning for Membership Inference Attacks Against Recommender Systems**

推荐系统成员关系推理攻击的去偏学习 cs.IR

Accepted by KDD 2022

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2206.12401v2)

**Authors**: Zihan Wang, Na Huang, Fei Sun, Pengjie Ren, Zhumin Chen, Hengliang Luo, Maarten de Rijke, Zhaochun Ren

**Abstracts**: Learned recommender systems may inadvertently leak information about their training data, leading to privacy violations. We investigate privacy threats faced by recommender systems through the lens of membership inference. In such attacks, an adversary aims to infer whether a user's data is used to train the target recommender. To achieve this, previous work has used a shadow recommender to derive training data for the attack model, and then predicts the membership by calculating difference vectors between users' historical interactions and recommended items. State-of-the-art methods face two challenging problems: (1) training data for the attack model is biased due to the gap between shadow and target recommenders, and (2) hidden states in recommenders are not observational, resulting in inaccurate estimations of difference vectors. To address the above limitations, we propose a Debiasing Learning for Membership Inference Attacks against recommender systems (DL-MIA) framework that has four main components: (1) a difference vector generator, (2) a disentangled encoder, (3) a weight estimator, and (4) an attack model. To mitigate the gap between recommenders, a variational auto-encoder (VAE) based disentangled encoder is devised to identify recommender invariant and specific features. To reduce the estimation bias, we design a weight estimator, assigning a truth-level score for each difference vector to indicate estimation accuracy. We evaluate DL-MIA against both general recommenders and sequential recommenders on three real-world datasets. Experimental results show that DL-MIA effectively alleviates training and estimation biases simultaneously, and achieves state-of-the-art attack performance.

摘要: 学习推荐系统可能会无意中泄露有关其训练数据的信息，导致侵犯隐私。我们通过成员关系推理的视角来研究推荐系统所面临的隐私威胁。在这类攻击中，对手的目标是推断用户的数据是否被用来训练目标推荐者。为此，以前的工作使用影子推荐器来获取攻击模型的训练数据，然后通过计算用户历史交互与推荐项目之间的差异向量来预测成员资格。最新的方法面临两个具有挑战性的问题：(1)由于阴影和目标推荐器之间的差距，攻击模型的训练数据存在偏差；(2)推荐器中的隐藏状态不是可观测的，导致对差异向量的估计不准确。针对上述局限性，我们提出了一个针对推荐系统成员推理攻击的去偏学习框架(DL-MIA)，该框架包括四个主要部分：(1)差分向量生成器，(2)解缠编码器，(3)权重估计器，(4)攻击模型。为了缩小推荐者之间的差距，设计了一种基于变分自动编码器(VAE)的解缠编码器来识别推荐者的不变性和特定特征。为了减少估计偏差，我们设计了一个权重估计器，为每个差异向量分配一个真实度分数来表示估计的准确性。在三个真实数据集上，我们对比了一般推荐器和顺序推荐器对DL-MIA进行了评估。实验结果表明，DL-MIA有效地同时缓解了训练偏差和估计偏差，达到了最好的攻击性能。



## **49. On the amplification of security and privacy risks by post-hoc explanations in machine learning models**

机器学习模型中事后解释对安全和隐私风险的放大 cs.LG

9 pages, appendix: 2 pages

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2206.14004v1)

**Authors**: Pengrui Quan, Supriyo Chakraborty, Jeya Vikranth Jeyakumar, Mani Srivastava

**Abstracts**: A variety of explanation methods have been proposed in recent years to help users gain insights into the results returned by neural networks, which are otherwise complex and opaque black-boxes. However, explanations give rise to potential side-channels that can be leveraged by an adversary for mounting attacks on the system. In particular, post-hoc explanation methods that highlight input dimensions according to their importance or relevance to the result also leak information that weakens security and privacy. In this work, we perform the first systematic characterization of the privacy and security risks arising from various popular explanation techniques. First, we propose novel explanation-guided black-box evasion attacks that lead to 10 times reduction in query count for the same success rate. We show that the adversarial advantage from explanations can be quantified as a reduction in the total variance of the estimated gradient. Second, we revisit the membership information leaked by common explanations. Contrary to observations in prior studies, via our modified attacks we show significant leakage of membership information (above 100% improvement over prior results), even in a much stricter black-box setting. Finally, we study explanation-guided model extraction attacks and demonstrate adversarial gains through a large reduction in query count.

摘要: 近年来，人们提出了各种解释方法，以帮助用户深入了解神经网络返回的结果，否则这些结果就是复杂和不透明的黑匣子。然而，解释会导致潜在的旁路，对手可以利用这些旁路对系统进行攻击。特别是，根据输入维度的重要性或与结果的相关性来强调输入维度的事后解释方法也会泄露削弱安全和隐私的信息。在这项工作中，我们首次系统地描述了各种流行的解释技术所产生的隐私和安全风险。首先，我们提出了一种新的解释引导的黑盒逃避攻击，在相同的成功率下使查询次数减少了10倍。我们表明，从解释中获得的对手优势可以量化为估计梯度的总方差的减少。其次，我们重新审视了由常见解释泄露的成员信息。与先前研究中的观察相反，通过我们修改的攻击，我们显示了显著的成员信息泄露(比先前的结果提高了100%以上)，即使在更严格的黑盒设置中也是如此。最后，我们研究了解释引导的模型提取攻击，并通过大幅减少查询次数展示了敌意收益。



## **50. Increasing Confidence in Adversarial Robustness Evaluations**

增加对对手健壮性评估的信心 cs.LG

Oral at CVPR 2022 Workshop (Art of Robustness). Project website  https://zimmerrol.github.io/active-tests/

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2206.13991v1)

**Authors**: Roland S. Zimmermann, Wieland Brendel, Florian Tramer, Nicholas Carlini

**Abstracts**: Hundreds of defenses have been proposed to make deep neural networks robust against minimal (adversarial) input perturbations. However, only a handful of these defenses held up their claims because correctly evaluating robustness is extremely challenging: Weak attacks often fail to find adversarial examples even if they unknowingly exist, thereby making a vulnerable network look robust. In this paper, we propose a test to identify weak attacks, and thus weak defense evaluations. Our test slightly modifies a neural network to guarantee the existence of an adversarial example for every sample. Consequentially, any correct attack must succeed in breaking this modified network. For eleven out of thirteen previously-published defenses, the original evaluation of the defense fails our test, while stronger attacks that break these defenses pass it. We hope that attack unit tests - such as ours - will be a major component in future robustness evaluations and increase confidence in an empirical field that is currently riddled with skepticism.

摘要: 已经提出了数百种防御措施，以使深度神经网络对最小(对抗性)输入扰动具有健壮性。然而，这些防御中只有少数几个站得住脚，因为正确评估健壮性极具挑战性：弱攻击往往无法找到对抗性示例，即使它们在不知情的情况下存在，从而使易受攻击的网络看起来很健壮。在本文中，我们提出了一种测试来识别弱攻击，从而对弱防御进行评估。我们的测试略微修改了神经网络，以确保每个样本都存在对抗性示例。因此，任何正确的攻击都必须成功地破坏这个修改后的网络。在之前公布的13个防御措施中，有11个没有通过我们的测试，而打破这些防御措施的更强大的攻击通过了我们的测试。我们希望攻击单元测试--比如我们的测试--将成为未来健壮性评估的主要组成部分，并增加对目前充满怀疑的经验领域的信心。



