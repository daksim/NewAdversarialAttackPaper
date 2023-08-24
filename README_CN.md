# Latest Adversarial Attack Papers
**update at 2023-08-24 13:07:38**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. On-Manifold Projected Gradient Descent**

流形上投影的梯度下降 cs.LG

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.12279v1) [paper-pdf](http://arxiv.org/pdf/2308.12279v1)

**Authors**: Aaron Mahler, Tyrus Berry, Tom Stephens, Harbir Antil, Michael Merritt, Jeanie Schreiber, Ioannis Kevrekidis

**Abstract**: This work provides a computable, direct, and mathematically rigorous approximation to the differential geometry of class manifolds for high-dimensional data, along with nonlinear projections from input space onto these class manifolds. The tools are applied to the setting of neural network image classifiers, where we generate novel, on-manifold data samples, and implement a projected gradient descent algorithm for on-manifold adversarial training. The susceptibility of neural networks (NNs) to adversarial attack highlights the brittle nature of NN decision boundaries in input space. Introducing adversarial examples during training has been shown to reduce the susceptibility of NNs to adversarial attack; however, it has also been shown to reduce the accuracy of the classifier if the examples are not valid examples for that class. Realistic "on-manifold" examples have been previously generated from class manifolds in the latent of an autoencoder. Our work explores these phenomena in a geometric and computational setting that is much closer to the raw, high-dimensional input space than can be provided by VAE or other black box dimensionality reductions. We employ conformally invariant diffusion maps (CIDM) to approximate class manifolds in diffusion coordinates, and develop the Nystr\"{o}m projection to project novel points onto class manifolds in this setting. On top of the manifold approximation, we leverage the spectral exterior calculus (SEC) to determine geometric quantities such as tangent vectors of the manifold. We use these tools to obtain adversarial examples that reside on a class manifold, yet fool a classifier. These misclassifications then become explainable in terms of human-understandable manipulations within the data, by expressing the on-manifold adversary in the semantic basis on the manifold.

摘要: 这项工作为高维数据的类流形的微分几何提供了一个可计算的、直接的和数学上严格的近似，以及从输入空间到这些类流形的非线性投影。这些工具被应用于神经网络图像分类器的设置，在那里我们生成新的流形上的数据样本，并实现投影梯度下降算法来进行流形上的对抗性训练。神经网络对敌意攻击的敏感性突出了输入空间中神经网络决策边界的脆性。在训练过程中引入对抗性示例可以降低神经网络对对抗性攻击的敏感性；然而，如果这些示例不是该类的有效示例，则也会降低分类器的准确性。真实的“流形上”的例子以前已经在自动编码器的潜伏中从类流形中生成。我们的工作在几何和计算环境中探索这些现象，与VAE或其他黑盒降维方法相比，它更接近原始的高维输入空间。我们使用共形不变扩散映射(CIDM)来逼近扩散坐标下的类流形，并发展了Nystr‘{o}m投影来将新的点投影到类流形上。在流形近似的基础上，我们利用谱外演算(SEC)来确定几何量，如流形的切线向量。我们使用这些工具来获得驻留在类流形上的对抗性例子，但却欺骗了分类器。然后，通过在流形上用人类可理解的操作来表示流形上的对手，这些错误分类变得可解释。



## **2. Sample Complexity of Robust Learning against Evasion Attacks**

抗逃避攻击的稳健学习的样本复杂性 cs.LG

DPhil (PhD) Thesis - University of Oxford

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.12054v1) [paper-pdf](http://arxiv.org/pdf/2308.12054v1)

**Authors**: Pascale Gourdeau

**Abstract**: It is becoming increasingly important to understand the vulnerability of machine learning models to adversarial attacks. One of the fundamental problems in adversarial machine learning is to quantify how much training data is needed in the presence of evasion attacks, where data is corrupted at test time. In this thesis, we work with the exact-in-the-ball notion of robustness and study the feasibility of adversarially robust learning from the perspective of learning theory, considering sample complexity.   We first explore the setting where the learner has access to random examples only, and show that distributional assumptions are essential. We then focus on learning problems with distributions on the input data that satisfy a Lipschitz condition and show that robustly learning monotone conjunctions has sample complexity at least exponential in the adversary's budget (the maximum number of bits it can perturb on each input). However, if the adversary is restricted to perturbing $O(\log n)$ bits, then one can robustly learn conjunctions and decision lists w.r.t. log-Lipschitz distributions.   We then study learning models where the learner is given more power. We first consider local membership queries, where the learner can query the label of points near the training sample. We show that, under the uniform distribution, the exponential dependence on the adversary's budget to robustly learn conjunctions remains inevitable. We then introduce a local equivalence query oracle, which returns whether the hypothesis and target concept agree in a given region around a point in the training sample, and a counterexample if it exists. We show that if the query radius is equal to the adversary's budget, we can develop robust empirical risk minimization algorithms in the distribution-free setting. We give general query complexity upper and lower bounds, as well as for concrete concept classes.

摘要: 理解机器学习模型在对抗攻击中的脆弱性正变得越来越重要。对抗性机器学习的基本问题之一是量化在存在逃避攻击的情况下需要多少训练数据，其中数据在测试时间被破坏。本文从学习理论的角度出发，考虑样本的复杂性，提出了对抗性鲁棒学习的可行性。我们首先探索学习者只能接触随机示例的环境，并表明分布假设是必不可少的。然后，我们将重点放在满足Lipschitz条件的输入数据分布的学习问题上，并证明了稳健学习单调合取在对手的预算(它可以在每一输入上扰动的最大比特数)中具有至少指数的样本复杂性。然而，如果对手被限制为扰乱$O(\logn)$比特，则一个人可以稳健地学习合取和决策列表。对数-李普希茨分布。然后我们研究学习模式，在这种模式下，学习者被赋予更多的权力。我们首先考虑局部隶属关系查询，其中学习者可以查询训练样本附近的点的标签。我们证明，在均匀分布下，依靠对手的预算来强健地学习合取仍然是不可避免的。然后，我们引入一个局部等价查询预言，它返回假设和目标概念在训练样本点周围的给定区域是否一致，如果存在，则返回反例。我们证明了，如果查询半径等于对手的预算，我们可以在无分布的环境下开发稳健的经验风险最小化算法。我们给出了一般查询复杂度的上下界，以及具体概念类的查询复杂度。



## **3. Phase-shifted Adversarial Training**

相移对抗性训练 cs.LG

Conference on Uncertainty in Artificial Intelligence, 2023 (UAI 2023)

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2301.04785v3) [paper-pdf](http://arxiv.org/pdf/2301.04785v3)

**Authors**: Yeachan Kim, Seongyeon Kim, Ihyeok Seo, Bonggun Shin

**Abstract**: Adversarial training has been considered an imperative component for safely deploying neural network-based applications to the real world. To achieve stronger robustness, existing methods primarily focus on how to generate strong attacks by increasing the number of update steps, regularizing the models with the smoothed loss function, and injecting the randomness into the attack. Instead, we analyze the behavior of adversarial training through the lens of response frequency. We empirically discover that adversarial training causes neural networks to have low convergence to high-frequency information, resulting in highly oscillated predictions near each data. To learn high-frequency contents efficiently and effectively, we first prove that a universal phenomenon of frequency principle, i.e., \textit{lower frequencies are learned first}, still holds in adversarial training. Based on that, we propose phase-shifted adversarial training (PhaseAT) in which the model learns high-frequency components by shifting these frequencies to the low-frequency range where the fast convergence occurs. For evaluations, we conduct the experiments on CIFAR-10 and ImageNet with the adaptive attack carefully designed for reliable evaluation. Comprehensive results show that PhaseAT significantly improves the convergence for high-frequency information. This results in improved adversarial robustness by enabling the model to have smoothed predictions near each data.

摘要: 对抗性训练一直被认为是将基于神经网络的应用安全地部署到现实世界中的一个必不可少的组成部分。为了获得更强的稳健性，现有的方法主要集中在如何通过增加更新步骤、用平滑的损失函数对模型进行正则化以及在攻击中注入随机性来产生强攻击。相反，我们通过反应频率的镜头来分析对抗性训练的行为。我们经验发现，对抗性训练导致神经网络对高频信息的收敛程度较低，导致每个数据附近的预测高度振荡。为了高效有效地学习高频内容，我们首先证明了频率原理中的一个普遍现象，即先学习较低的频率在对抗性训练中仍然成立。在此基础上，我们提出了相移对抗性训练(PhaseAT)，该模型通过将高频成分转移到发生快速收敛的低频范围来学习高频成分。在评估方面，我们在CIFAR-10和ImageNet上进行了实验，并使用精心设计的自适应攻击进行了可靠的评估。综合结果表明，PhaseAT算法显著提高了高频信息的收敛速度。这使得模型能够在每个数据附近平滑预测，从而提高了对手的稳健性。



## **4. Designing an attack-defense game: how to increase robustness of financial transaction models via a competition**

设计攻防游戏：如何通过竞争增加金融交易模型的健壮性 cs.LG

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.11406v2) [paper-pdf](http://arxiv.org/pdf/2308.11406v2)

**Authors**: Alexey Zaytsev, Alex Natekin, Evgeni Vorsin, Valerii Smirnov, Georgii Smirnov, Oleg Sidorshin, Alexander Senin, Alexander Dudin, Dmitry Berestnev

**Abstract**: Given the escalating risks of malicious attacks in the finance sector and the consequential severe damage, a thorough understanding of adversarial strategies and robust defense mechanisms for machine learning models is critical. The threat becomes even more severe with the increased adoption in banks more accurate, but potentially fragile neural networks. We aim to investigate the current state and dynamics of adversarial attacks and defenses for neural network models that use sequential financial data as the input.   To achieve this goal, we have designed a competition that allows realistic and detailed investigation of problems in modern financial transaction data. The participants compete directly against each other, so possible attacks and defenses are examined in close-to-real-life conditions. Our main contributions are the analysis of the competition dynamics that answers the questions on how important it is to conceal a model from malicious users, how long does it take to break it, and what techniques one should use to make it more robust, and introduction additional way to attack models or increase their robustness.   Our analysis continues with a meta-study on the used approaches with their power, numerical experiments, and accompanied ablations studies. We show that the developed attacks and defenses outperform existing alternatives from the literature while being practical in terms of execution, proving the validity of the competition as a tool for uncovering vulnerabilities of machine learning models and mitigating them in various domains.

摘要: 鉴于金融领域不断升级的恶意攻击风险和由此带来的严重破坏，彻底了解机器学习模型的对抗性策略和强大的防御机制至关重要。随着银行越来越多地采用更准确但可能脆弱的神经网络，这种威胁变得更加严重。我们的目标是调查当前状态和动态的对抗性攻击和防御的神经网络模型，使用连续的金融数据作为输入。为了实现这一目标，我们设计了一个竞赛，允许对现代金融交易数据中的问题进行现实和详细的调查。参与者之间直接竞争，所以可能的攻击和防御都是在接近现实的条件下进行检查的。我们的主要贡献是对竞争动态的分析，回答了以下问题：向恶意用户隐藏模型有多重要，需要多长时间才能打破它，以及应该使用什么技术使其更健壮，并引入了攻击模型或增强其健壮性的其他方法。我们的分析继续对所使用的方法及其威力、数值实验和伴随的消融研究进行元研究。我们证明了所开发的攻击和防御在执行方面优于现有的可选方案，证明了竞争作为发现机器学习模型的漏洞并在不同领域缓解它们的工具的有效性。



## **5. Does Physical Adversarial Example Really Matter to Autonomous Driving? Towards System-Level Effect of Adversarial Object Evasion Attack**

身体对抗的例子对自动驾驶真的很重要吗？论对抗性目标逃避攻击的系统级效应 cs.CR

Accepted by ICCV 2023

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.11894v1) [paper-pdf](http://arxiv.org/pdf/2308.11894v1)

**Authors**: Ningfei Wang, Yunpeng Luo, Takami Sato, Kaidi Xu, Qi Alfred Chen

**Abstract**: In autonomous driving (AD), accurate perception is indispensable to achieving safe and secure driving. Due to its safety-criticality, the security of AD perception has been widely studied. Among different attacks on AD perception, the physical adversarial object evasion attacks are especially severe. However, we find that all existing literature only evaluates their attack effect at the targeted AI component level but not at the system level, i.e., with the entire system semantics and context such as the full AD pipeline. Thereby, this raises a critical research question: can these existing researches effectively achieve system-level attack effects (e.g., traffic rule violations) in the real-world AD context? In this work, we conduct the first measurement study on whether and how effectively the existing designs can lead to system-level effects, especially for the STOP sign-evasion attacks due to their popularity and severity. Our evaluation results show that all the representative prior works cannot achieve any system-level effects. We observe two design limitations in the prior works: 1) physical model-inconsistent object size distribution in pixel sampling and 2) lack of vehicle plant model and AD system model consideration. Then, we propose SysAdv, a novel system-driven attack design in the AD context and our evaluation results show that the system-level effects can be significantly improved, i.e., the violation rate increases by around 70%.

摘要: 在自动驾驶中，准确的感知是实现安全驾驶不可缺少的。由于其安全临界性，AD感知的安全性得到了广泛的研究。在针对AD感知的各种攻击中，物理对抗性对象逃避攻击尤为严重。然而，我们发现，现有的所有文献只在目标人工智能组件级别上评估其攻击效果，而没有在系统级别上进行评估，即使用整个系统语义和上下文，如完整的AD流水线。因此，这就提出了一个关键的研究问题：这些现有的研究能否在现实的AD环境中有效地实现系统级的攻击效果(例如，违反交通规则)？在这项工作中，我们进行了第一次测量研究现有的设计是否以及如何有效地导致系统级的影响，特别是由于停车标志逃避攻击的流行和严重程度。我们的评估结果表明，所有有代表性的前期工作都不能达到任何系统级的效果。我们观察到了前人工作中的两个设计局限性：1)物理模型-像素采样中对象大小分布不一致；2)缺乏对车辆植物模型和AD系统模型的考虑。然后，我们提出了一种新的AD环境下的系统驱动攻击设计--SysAdv，我们的评估结果表明，系统级的攻击效果可以得到显著的改善，即违规率提高了70%左右。



## **6. Adversarial Training Using Feedback Loops**

使用反馈环进行对抗性训练 cs.LG

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.11881v1) [paper-pdf](http://arxiv.org/pdf/2308.11881v1)

**Authors**: Ali Haisam Muhammad Rafid, Adrian Sandu

**Abstract**: Deep neural networks (DNN) have found wide applicability in numerous fields due to their ability to accurately learn very complex input-output relations. Despite their accuracy and extensive use, DNNs are highly susceptible to adversarial attacks due to limited generalizability. For future progress in the field, it is essential to build DNNs that are robust to any kind of perturbations to the data points. In the past, many techniques have been proposed to robustify DNNs using first-order derivative information of the network.   This paper proposes a new robustification approach based on control theory. A neural network architecture that incorporates feedback control, named Feedback Neural Networks, is proposed. The controller is itself a neural network, which is trained using regular and adversarial data such as to stabilize the system outputs. The novel adversarial training approach based on the feedback control architecture is called Feedback Looped Adversarial Training (FLAT). Numerical results on standard test problems empirically show that our FLAT method is more effective than the state-of-the-art to guard against adversarial attacks.

摘要: 深度神经网络(DNN)因其能够准确学习非常复杂的输入输出关系而在许多领域得到了广泛的应用。尽管DNN具有较高的准确性和广泛的用途，但由于泛化能力有限，DNN极易受到敌意攻击。为了在该领域取得未来的进展，建立对数据点的任何类型的扰动都是健壮的DNN是至关重要的。在过去，已经提出了许多技术来利用网络的一阶导数信息来使DNN具有健壮性。本文基于控制理论提出了一种新的鲁棒性控制方法。提出了一种包含反馈控制的神经网络体系结构--反馈神经网络。控制器本身是一个神经网络，它使用常规和对抗性数据进行训练，以稳定系统输出。基于反馈控制结构的新型对抗训练方法称为反馈环对抗训练(Flat)。在标准测试问题上的数值结果经验表明，我们的Flat方法比最新的方法更有效地防御对抗性攻击。



## **7. Measuring Equality in Machine Learning Security Defenses: A Case Study in Speech Recognition**

机器学习安全防御中的平等度量：以语音识别为例 cs.LG

Accepted to AISec'23

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2302.08973v6) [paper-pdf](http://arxiv.org/pdf/2302.08973v6)

**Authors**: Luke E. Richards, Edward Raff, Cynthia Matuszek

**Abstract**: Over the past decade, the machine learning security community has developed a myriad of defenses for evasion attacks. An understudied question in that community is: for whom do these defenses defend? This work considers common approaches to defending learned systems and how security defenses result in performance inequities across different sub-populations. We outline appropriate parity metrics for analysis and begin to answer this question through empirical results of the fairness implications of machine learning security methods. We find that many methods that have been proposed can cause direct harm, like false rejection and unequal benefits from robustness training. The framework we propose for measuring defense equality can be applied to robustly trained models, preprocessing-based defenses, and rejection methods. We identify a set of datasets with a user-centered application and a reasonable computational cost suitable for case studies in measuring the equality of defenses. In our case study of speech command recognition, we show how such adversarial training and augmentation have non-equal but complex protections for social subgroups across gender, accent, and age in relation to user coverage. We present a comparison of equality between two rejection-based defenses: randomized smoothing and neural rejection, finding randomized smoothing more equitable due to the sampling mechanism for minority groups. This represents the first work examining the disparity in the adversarial robustness in the speech domain and the fairness evaluation of rejection-based defenses.

摘要: 在过去的十年里，机器学习安全社区开发了无数针对逃避攻击的防御措施。在那个社区里，一个被忽视的问题是：这些防御是为谁辩护？这项工作考虑了保护学习系统的常见方法，以及安全防御如何导致不同子群体的性能不平等。我们勾勒出合适的奇偶度量进行分析，并通过机器学习安全方法的公平性影响的实证结果开始回答这个问题。我们发现，已提出的许多方法都会造成直接危害，如错误拒绝和健壮性训练带来的不平等好处。我们提出的衡量防御平等的框架可以应用于稳健训练的模型、基于预处理的防御和拒绝方法。我们确定了一组数据集，这些数据集具有以用户为中心的应用程序，并且计算成本合理，适合用于案例研究来衡量辩护的等价性。在我们的语音命令识别案例研究中，我们展示了这种对抗性训练和增强如何根据用户覆盖范围为不同性别、口音和年龄的社会亚群提供不平等但复杂的保护。我们比较了两种基于拒绝的防御机制：随机平滑和神经排斥，发现由于少数群体的抽样机制，随机平滑更公平。这是第一个研究语音域中对抗性稳健性的差异和基于拒绝的防御的公平性评估的工作。



## **8. SEA: Shareable and Explainable Attribution for Query-based Black-box Attacks**

SEA：基于查询的黑盒攻击的可共享和可解释的属性 cs.LG

**SubmitDate**: 2023-08-23    [abs](http://arxiv.org/abs/2308.11845v1) [paper-pdf](http://arxiv.org/pdf/2308.11845v1)

**Authors**: Yue Gao, Ilia Shumailov, Kassem Fawaz

**Abstract**: Machine Learning (ML) systems are vulnerable to adversarial examples, particularly those from query-based black-box attacks. Despite various efforts to detect and prevent such attacks, there is a need for a more comprehensive approach to logging, analyzing, and sharing evidence of attacks. While classic security benefits from well-established forensics and intelligence sharing, Machine Learning is yet to find a way to profile its attackers and share information about them. In response, this paper introduces SEA, a novel ML security system to characterize black-box attacks on ML systems for forensic purposes and to facilitate human-explainable intelligence sharing. SEA leverages the Hidden Markov Models framework to attribute the observed query sequence to known attacks. It thus understands the attack's progression rather than just focusing on the final adversarial examples. Our evaluations reveal that SEA is effective at attack attribution, even on their second occurrence, and is robust to adaptive strategies designed to evade forensics analysis. Interestingly, SEA's explanations of the attack behavior allow us even to fingerprint specific minor implementation bugs in attack libraries. For example, we discover that the SignOPT and Square attacks implementation in ART v1.14 sends over 50% specific zero difference queries. We thoroughly evaluate SEA on a variety of settings and demonstrate that it can recognize the same attack's second occurrence with 90+% Top-1 and 95+% Top-3 accuracy.

摘要: 机器学习(ML)系统容易受到敌意例子的攻击，特别是那些来自基于查询的黑盒攻击的例子。尽管做出了各种努力来检测和防止此类攻击，但仍需要一种更全面的方法来记录、分析和共享攻击证据。虽然传统的安全技术得益于成熟的取证和情报共享，但机器学习尚未找到一种方法来分析攻击者并分享有关他们的信息。对此，本文引入了一种新的ML安全系统SEA，用于刻画针对ML系统的黑盒攻击，用于取证目的，并促进人类可解释的情报共享。SEA利用隐马尔可夫模型框架将观察到的查询序列归因于已知攻击。因此，它了解攻击的进展，而不是只关注最后的对抗性例子。我们的评估表明，SEA在攻击归属方面是有效的，即使在第二次攻击发生时也是如此，并且对于旨在逃避取证分析的自适应策略是健壮的。有趣的是，SEA对攻击行为的解释甚至允许我们确定攻击库中特定的微小实现错误。例如，我们发现ART v1.14中的SignOPT和Square攻击实现发送超过50%的特定零差查询。我们在各种不同的设置下对SEA进行了全面的评估，并证明了它能够以90%以上的Top-1和95%+%的Top-3准确率识别同一攻击的第二次发生。



## **9. Ceci n'est pas une pomme: Adversarial Illusions in Multi-Modal Embeddings**

Ceci n‘est Pas une Pomme：多模式嵌入中的对抗性错觉 cs.CR

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2308.11804v1) [paper-pdf](http://arxiv.org/pdf/2308.11804v1)

**Authors**: Eugene Bagdasaryan, Vitaly Shmatikov

**Abstract**: Multi-modal encoders map images, sounds, texts, videos, etc. into a single embedding space, aligning representations across modalities (e.g., associate an image of a dog with a barking sound). We show that multi-modal embeddings can be vulnerable to an attack we call "adversarial illusions." Given an input in any modality, an adversary can perturb it so as to make its embedding close to that of an arbitrary, adversary-chosen input in another modality. Illusions thus enable the adversary to align any image with any text, any text with any sound, etc.   Adversarial illusions exploit proximity in the embedding space and are thus agnostic to downstream tasks. Using ImageBind embeddings, we demonstrate how adversarially aligned inputs, generated without knowledge of specific downstream tasks, mislead image generation, text generation, and zero-shot classification.

摘要: 多模式编码器将图像、声音、文本、视频等映射到单个嵌入空间中，跨模式对齐表示(例如，将狗的图像与犬吠声相关联)。我们表明，多模式嵌入可能容易受到一种我们称为“对抗性错觉”的攻击。给定任何形式的输入，敌手都可以对其进行干扰，以便使其嵌入接近于任意的、对手选择的另一种形式的输入。因此，错觉使对手能够将任何图像与任何文本、任何文本与任何声音等对齐。对抗性错觉利用嵌入空间中的邻近，因此与下游任务无关。使用ImageBind嵌入，我们演示了在不知道特定下游任务的情况下生成的恶意对齐的输入如何误导图像生成、文本生成和零镜头分类。



## **10. Multi-Instance Adversarial Attack on GNN-Based Malicious Domain Detection**

基于GNN的恶意域名检测中的多实例对抗性攻击 cs.CR

To Appear in the 45th IEEE Symposium on Security and Privacy (IEEE  S\&P 2024), May 20-23, 2024

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2308.11754v1) [paper-pdf](http://arxiv.org/pdf/2308.11754v1)

**Authors**: Mahmoud Nazzal, Issa Khalil, Abdallah Khreishah, NhatHai Phan, Yao Ma

**Abstract**: Malicious domain detection (MDD) is an open security challenge that aims to detect if an Internet domain is associated with cyber-attacks. Among many approaches to this problem, graph neural networks (GNNs) are deemed highly effective. GNN-based MDD uses DNS logs to represent Internet domains as nodes in a maliciousness graph (DMG) and trains a GNN to infer their maliciousness by leveraging identified malicious domains. Since this method relies on accessible DNS logs to construct DMGs, it exposes a vulnerability for adversaries to manipulate their domain nodes' features and connections within DMGs. Existing research mainly concentrates on threat models that manipulate individual attacker nodes. However, adversaries commonly generate multiple domains to achieve their goals economically and avoid detection. Their objective is to evade discovery across as many domains as feasible. In this work, we call the attack that manipulates several nodes in the DMG concurrently a multi-instance evasion attack. We present theoretical and empirical evidence that the existing single-instance evasion techniques for are inadequate to launch multi-instance evasion attacks against GNN-based MDDs. Therefore, we introduce MintA, an inference-time multi-instance adversarial attack on GNN-based MDDs. MintA enhances node and neighborhood evasiveness through optimized perturbations and operates successfully with only black-box access to the target model, eliminating the need for knowledge about the model's specifics or non-adversary nodes. We formulate an optimization challenge for MintA, achieving an approximate solution. Evaluating MintA on a leading GNN-based MDD technique with real-world data showcases an attack success rate exceeding 80%. These findings act as a warning for security experts, underscoring GNN-based MDDs' susceptibility to practical attacks that can undermine their effectiveness and benefits.

摘要: 恶意域检测(MDD)是一种开放的安全挑战，旨在检测Internet域是否与网络攻击相关联。在解决这一问题的众多方法中，图神经网络被认为是非常有效的。基于GNN的MDD使用DNS日志将Internet域表示为恶意图(DMG)中的节点，并训练GNN通过利用已识别的恶意域来推断它们的恶意。由于此方法依赖于可访问的DNS日志来构建DMG，因此它暴露了一个漏洞，使得攻击者能够操纵其域节点在DMG中的功能和连接。现有的研究主要集中在操纵单个攻击者节点的威胁模型上。然而，攻击者通常会生成多个域以经济地实现其目标并避免被检测到。他们的目标是在尽可能多的可行领域中逃避发现。在本文中，我们将同时操纵DMG中多个节点的攻击称为多实例逃避攻击。我们给出了理论和经验证据，证明了现有的单实例规避技术不足以对基于GNN的MDDS发起多实例规避攻击。因此，我们引入了Minta，一种基于GNN的MDDS的推理时多实例对抗性攻击。Minta通过优化的扰动增强了节点和邻域的规避能力，并且只需通过黑盒访问目标模型即可成功运行，无需了解模型的细节或非敌对节点。我们为Minta制定了一个优化挑战，得到了一个近似解。用真实世界的数据评估Minta在基于GNN的领先MDD技术上的攻击成功率超过80%。这些发现对安全专家来说是一个警告，突显了基于GNN的MDDS容易受到实际攻击，这些攻击可能会破坏其有效性和好处。



## **11. Security Analysis of the Consumer Remote SIM Provisioning Protocol**

消费者远程SIM配置协议的安全性分析 cs.CR

35 pages, 9 figures, Associated ProVerif model files located at  https://github.com/peltona/rsp_model

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2211.15323v2) [paper-pdf](http://arxiv.org/pdf/2211.15323v2)

**Authors**: Abu Shohel Ahmed, Aleksi Peltonen, Mohit Sethi, Tuomas Aura

**Abstract**: Remote SIM provisioning (RSP) for consumer devices is the protocol specified by the GSM Association for downloading SIM profiles into a secure element in a mobile device. The process is commonly known as eSIM, and it is expected to replace removable SIM cards. The security of the protocol is critical because the profile includes the credentials with which the mobile device will authenticate to the mobile network. In this paper, we present a formal security analysis of the consumer RSP protocol. We model the multi-party protocol in applied pi calculus, define formal security goals, and verify them in ProVerif. The analysis shows that the consumer RSP protocol protects against a network adversary when all the intended participants are honest. However, we also model the protocol in realistic partial compromise scenarios where the adversary controls a legitimate participant or communication channel. The security failures in the partial compromise scenarios reveal weaknesses in the protocol design. The most important observation is that the security of RSP depends unnecessarily on it being encapsulated in a TLS tunnel. Also, the lack of pre-established identifiers means that a compromised download server anywhere in the world or a compromised secure element can be used for attacks against RSP between honest participants. Additionally, the lack of reliable methods for verifying user intent can lead to serious security failures. Based on the findings, we recommend practical improvements to RSP implementations, future versions of the specification, and mobile operator processes to increase the robustness of eSIM security.

摘要: 用于消费者设备的远程SIM供应(RSP)是由GSM协会指定的用于将SIM简档下载到移动设备中的安全元件的协议。这一过程通常被称为eSIM卡，预计将取代可拆卸的SIM卡。协议的安全性是至关重要的，因为简档包括移动设备将用来向移动网络进行认证的凭证。本文对消费者RSP协议进行了形式化的安全性分析。我们用pi演算对多方协议进行了建模，定义了形式化的安全目标，并在ProVerif中进行了验证。分析表明，当所有预期参与者都是诚实的时，消费者RSP协议可以防御网络对手。然而，我们也在现实的部分妥协场景中对协议进行建模，其中对手控制合法的参与者或通信通道。部分妥协场景中的安全故障揭示了协议设计中的弱点。最重要的观察是，RSP的安全性不必要地依赖于它被封装在TLS隧道中。此外，缺乏预先建立的标识符意味着世界上任何地方的受攻击的下载服务器或受攻击的安全元素都可以用于在诚实的参与者之间对RSP进行攻击。此外，缺乏可靠的方法来验证用户意图可能会导致严重的安全故障。基于这些发现，我们建议对RSP实施、规范的未来版本和移动运营商流程进行实际改进，以增加eSIM安全的健壮性。



## **12. Evading Watermark based Detection of AI-Generated Content**

基于规避水印的人工智能生成内容检测 cs.LG

To appear in ACM Conference on Computer and Communications Security  (CCS), 2023

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2305.03807v3) [paper-pdf](http://arxiv.org/pdf/2305.03807v3)

**Authors**: Zhengyuan Jiang, Jinghuai Zhang, Neil Zhenqiang Gong

**Abstract**: A generative AI model can generate extremely realistic-looking content, posing growing challenges to the authenticity of information. To address the challenges, watermark has been leveraged to detect AI-generated content. Specifically, a watermark is embedded into an AI-generated content before it is released. A content is detected as AI-generated if a similar watermark can be decoded from it. In this work, we perform a systematic study on the robustness of such watermark-based AI-generated content detection. We focus on AI-generated images. Our work shows that an attacker can post-process a watermarked image via adding a small, human-imperceptible perturbation to it, such that the post-processed image evades detection while maintaining its visual quality. We show the effectiveness of our attack both theoretically and empirically. Moreover, to evade detection, our adversarial post-processing method adds much smaller perturbations to AI-generated images and thus better maintain their visual quality than existing popular post-processing methods such as JPEG compression, Gaussian blur, and Brightness/Contrast. Our work shows the insufficiency of existing watermark-based detection of AI-generated content, highlighting the urgent needs of new methods. Our code is publicly available: https://github.com/zhengyuan-jiang/WEvade.

摘要: 生成性人工智能模型可以生成极其逼真的内容，对信息的真实性提出了越来越大的挑战。为了应对这些挑战，水印被用来检测人工智能生成的内容。具体地说，水印在发布之前被嵌入到人工智能生成的内容中。如果可以从内容中解码类似的水印，则该内容被检测为人工智能生成的内容。在这项工作中，我们对这种基于水印的人工智能内容检测的稳健性进行了系统的研究。我们专注于人工智能生成的图像。我们的工作表明，攻击者可以通过在水印图像上添加一个人类无法察觉的小扰动来对其进行后处理，从而在保持其视觉质量的同时逃避检测。我们从理论和经验上证明了我们的攻击的有效性。此外，为了逃避检测，我们的对抗性后处理方法向人工智能生成的图像添加了更小的扰动，从而比现有的流行的后处理方法，如JPEG压缩、高斯模糊和亮度/对比度，更好地保持了它们的视觉质量。我们的工作显示了现有基于水印的人工智能生成内容检测的不足，突出了对新方法的迫切需求。我们的代码是公开提供的：https://github.com/zhengyuan-jiang/WEvade.



## **13. Robustness of SAM: Segment Anything Under Corruptions and Beyond**

SAM的健壮性：在腐败和其他方面分割任何东西 cs.CV

The first work evaluates the robustness of SAM under various  corruptions such as style transfer, local occlusion, and adversarial patch  attack

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2306.07713v2) [paper-pdf](http://arxiv.org/pdf/2306.07713v2)

**Authors**: Yu Qiao, Chaoning Zhang, Taegoo Kang, Donghun Kim, Shehbaz Tariq, Chenshuang Zhang, Choong Seon Hong

**Abstract**: Segment anything model (SAM), as the name suggests, is claimed to be capable of cutting out any object and demonstrates impressive zero-shot transfer performance with the guidance of a prompt. However, there is currently a lack of comprehensive evaluation regarding its robustness under various corruptions. Understanding SAM's robustness across different corruption scenarios is crucial for its real-world deployment. Prior works show that SAM is biased towards texture (style) rather than shape, motivated by which we start by investigating SAM's robustness against style transfer, which is synthetic corruption. Following the interpretation of the corruption's effect as style change, we proceed to conduct a comprehensive evaluation of the SAM for its robustness against 15 types of common corruption. These corruptions mainly fall into categories such as digital, noise, weather, and blur. Within each of these corruption categories, we explore 5 severity levels to simulate real-world corruption scenarios. Beyond the corruptions, we further assess its robustness regarding local occlusion and local adversarial patch attacks in images. To the best of our knowledge, our work is the first of its kind to evaluate the robustness of SAM under style change, local occlusion, and local adversarial patch attacks. Considering that patch attacks visible to human eyes are easily detectable, we also assess SAM's robustness against adversarial perturbations that are imperceptible to human eyes. Overall, this work provides a comprehensive empirical study on SAM's robustness, evaluating its performance under various corruptions and extending the assessment to critical aspects like local occlusion, local patch attacks, and imperceptible adversarial perturbations, which yields valuable insights into SAM's practical applicability and effectiveness in addressing real-world challenges.

摘要: 片断任何模型(SAM)，顾名思义，声称能够裁剪出任何对象，并在提示的指导下展示了令人印象深刻的零射传输性能。然而，目前还缺乏对其在各种腐败情况下的稳健性的全面评估。了解SAM在不同腐败场景中的健壮性对于其实际部署至关重要。以前的工作表明，SAM偏向于纹理(风格)而不是形状，受此启发，我们从研究SAM对风格转移的健壮性开始，这是一种合成腐败。在将腐败的影响解释为风格变化之后，我们接着对SAM进行了全面的评估，以评估其对15种常见腐败的稳健性。这些损坏主要分为数字、噪声、天气和模糊等类别。在这些腐败类别中的每一个中，我们探索了5个严重程度来模拟真实世界的腐败场景。此外，我们还进一步评估了该算法对图像局部遮挡和局部对抗性补丁攻击的稳健性。据我们所知，我们的工作是第一次评估SAM在风格变化、局部遮挡和局部敌意补丁攻击下的稳健性。考虑到人眼可见的补丁攻击很容易被检测到，我们还评估了SAM对人眼不可察觉的敌意扰动的鲁棒性。总体而言，这项工作对SAM的健壮性进行了全面的实证研究，评估了其在各种腐败情况下的性能，并将评估扩展到关键方面，如局部遮挡、局部补丁攻击和不可察觉的对抗性扰动，这对SAM在应对现实世界挑战方面的实际适用性和有效性产生了有价值的见解。



## **14. CrowdGuard: Federated Backdoor Detection in Federated Learning**

CrowdGuard：联邦学习中的联邦后门检测 cs.CR

To appear in the Network and Distributed System Security (NDSS)  Symposium 2024. Phillip Rieger and Torsten Krau{\ss} contributed equally to  this contribution. 19 pages, 8 figures, 5 tables, 4 algorithms, 5 equations

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2210.07714v3) [paper-pdf](http://arxiv.org/pdf/2210.07714v3)

**Authors**: Phillip Rieger, Torsten Krauß, Markus Miettinen, Alexandra Dmitrienko, Ahmad-Reza Sadeghi

**Abstract**: Federated Learning (FL) is a promising approach enabling multiple clients to train Deep Neural Networks (DNNs) collaboratively without sharing their local training data. However, FL is susceptible to backdoor (or targeted poisoning) attacks. These attacks are initiated by malicious clients who seek to compromise the learning process by introducing specific behaviors into the learned model that can be triggered by carefully crafted inputs. Existing FL safeguards have various limitations: They are restricted to specific data distributions or reduce the global model accuracy due to excluding benign models or adding noise, are vulnerable to adaptive defense-aware adversaries, or require the server to access local models, allowing data inference attacks.   This paper presents a novel defense mechanism, CrowdGuard, that effectively mitigates backdoor attacks in FL and overcomes the deficiencies of existing techniques. It leverages clients' feedback on individual models, analyzes the behavior of neurons in hidden layers, and eliminates poisoned models through an iterative pruning scheme. CrowdGuard employs a server-located stacked clustering scheme to enhance its resilience to rogue client feedback. The evaluation results demonstrate that CrowdGuard achieves a 100% True-Positive-Rate and True-Negative-Rate across various scenarios, including IID and non-IID data distributions. Additionally, CrowdGuard withstands adaptive adversaries while preserving the original performance of protected models. To ensure confidentiality, CrowdGuard uses a secure and privacy-preserving architecture leveraging Trusted Execution Environments (TEEs) on both client and server sides.

摘要: 联合学习(FL)是一种很有前途的方法，可以使多个客户在不共享本地训练数据的情况下协作训练深度神经网络(DNN)。然而，FL很容易受到后门(或定向中毒)攻击。这些攻击是由恶意客户端发起的，这些客户端试图通过在学习模型中引入特定行为来危害学习过程，这些行为可以由精心编制的输入触发。现有的FL安全机制有各种局限性：它们限制于特定的数据分布，或者由于排除良性模型或添加噪声而降低全局模型的准确性，容易受到自适应防御感知的对手的攻击，或者要求服务器访问本地模型，从而允许数据推理攻击。提出了一种新的防御机制CrowdGuard，有效地缓解了FL中的后门攻击，克服了现有技术的不足。它利用客户对单个模型的反馈，分析隐藏层中神经元的行为，并通过迭代剪枝方案消除有毒模型。CrowdGuard采用位于服务器的堆叠群集方案，以增强其对恶意客户端反馈的恢复能力。评估结果表明，CrowdGuard在各种场景下，包括IID和非IID数据分布，都达到了100%的正确率和正反率。此外，CrowdGuard在保持受保护模型的原始性能的同时，还能抵御适应性攻击。为了确保机密性，CrowdGuard使用安全和隐私保护的架构，利用客户端和服务器端的可信执行环境(TEE)。



## **15. LDP-Feat: Image Features with Local Differential Privacy**

LDP-FEAT：具有局部差分隐私的图像特征 cs.CV

11 pages, 4 figures, to be published in International Conference on  Computer Vision (ICCV) 2023

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2308.11223v1) [paper-pdf](http://arxiv.org/pdf/2308.11223v1)

**Authors**: Francesco Pittaluga, Bingbing Zhuang

**Abstract**: Modern computer vision services often require users to share raw feature descriptors with an untrusted server. This presents an inherent privacy risk, as raw descriptors may be used to recover the source images from which they were extracted. To address this issue, researchers recently proposed privatizing image features by embedding them within an affine subspace containing the original feature as well as adversarial feature samples. In this paper, we propose two novel inversion attacks to show that it is possible to (approximately) recover the original image features from these embeddings, allowing us to recover privacy-critical image content. In light of such successes and the lack of theoretical privacy guarantees afforded by existing visual privacy methods, we further propose the first method to privatize image features via local differential privacy, which, unlike prior approaches, provides a guaranteed bound for privacy leakage regardless of the strength of the attacks. In addition, our method yields strong performance in visual localization as a downstream task while enjoying the privacy guarantee.

摘要: 现代计算机视觉服务通常要求用户与不可信的服务器共享原始特征描述符。这存在固有的隐私风险，因为原始描述符可能被用来恢复从中提取它们的源图像。为了解决这个问题，研究人员最近提出了通过将图像特征嵌入到包含原始特征和对抗性特征样本的仿射子空间中来私有化图像特征。在本文中，我们提出了两种新的反转攻击，以表明从这些嵌入中(近似地)恢复原始图像特征是可能的，从而允许我们恢复隐私关键的图像内容。鉴于这些研究的成功和现有视觉隐私保护方法缺乏理论上的隐私保障，我们进一步提出了第一种通过局部差分隐私保护图像特征的方法，与以往的方法不同，该方法提供了隐私泄露的保证范围，而不考虑攻击的强度。此外，我们的方法在享受隐私保障的同时，作为一项下游任务在视觉定位方面取得了很好的性能。



## **16. Boosting Adversarial Transferability by Block Shuffle and Rotation**

通过区块洗牌和轮换来提高对手的转移能力 cs.CV

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2308.10299v2) [paper-pdf](http://arxiv.org/pdf/2308.10299v2)

**Authors**: Kunyu Wang, Xuanran He, Wenxuan Wang, Xiaosen Wang

**Abstract**: Adversarial examples mislead deep neural networks with imperceptible perturbations and have brought significant threats to deep learning. An important aspect is their transferability, which refers to their ability to deceive other models, thus enabling attacks in the black-box setting. Though various methods have been proposed to boost transferability, the performance still falls short compared with white-box attacks. In this work, we observe that existing input transformation based attacks, one of the mainstream transfer-based attacks, result in different attention heatmaps on various models, which might limit the transferability. We also find that breaking the intrinsic relation of the image can disrupt the attention heatmap of the original image. Based on this finding, we propose a novel input transformation based attack called block shuffle and rotation (BSR). Specifically, BSR splits the input image into several blocks, then randomly shuffles and rotates these blocks to construct a set of new images for gradient calculation. Empirical evaluations on the ImageNet dataset demonstrate that BSR could achieve significantly better transferability than the existing input transformation based methods under single-model and ensemble-model settings. Combining BSR with the current input transformation method can further improve the transferability, which significantly outperforms the state-of-the-art methods.

摘要: 对抗性例子用潜移默化的扰动误导了深度神经网络，给深度学习带来了重大威胁。一个重要的方面是它们的可转移性，这指的是它们欺骗其他模型的能力，从而使攻击能够在黑盒环境中进行。虽然已经提出了各种方法来提高可转移性，但与白盒攻击相比，性能仍然不足。在这项工作中，我们观察到现有的基于输入变换的攻击是基于转移的主流攻击之一，在不同的模型上会导致不同的注意力热图，这可能会限制可转移性。我们还发现，打破图像的内在联系会扰乱原始图像的注意热图。基于这一发现，我们提出了一种新的基于输入变换的攻击方法，称为块置乱和旋转攻击(BSR)。具体地说，BSR将输入图像分成几个块，然后随机地对这些块进行洗牌和旋转，以构建一组新的图像用于梯度计算。在ImageNet数据集上的实证评估表明，在单模型和集成模型的设置下，BSR可以获得比现有的基于输入变换的方法更好的可转移性。将BSR与当前的输入变换方法相结合，可以进一步提高可转移性，显著优于最先进的方法。



## **17. Adversarial Attacks on Code Models with Discriminative Graph Patterns**

具有区分图模式的代码模型的对抗性攻击 cs.SE

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2308.11161v1) [paper-pdf](http://arxiv.org/pdf/2308.11161v1)

**Authors**: Thanh-Dat Nguyen, Yang Zhou, Xuan Bach D. Le, Patanamon, Thongtanunam, David Lo

**Abstract**: Pre-trained language models of code are now widely used in various software engineering tasks such as code generation, code completion, vulnerability detection, etc. This, in turn, poses security and reliability risks to these models. One of the important threats is \textit{adversarial attacks}, which can lead to erroneous predictions and largely affect model performance on downstream tasks. Current adversarial attacks on code models usually adopt fixed sets of program transformations, such as variable renaming and dead code insertion, leading to limited attack effectiveness. To address the aforementioned challenges, we propose a novel adversarial attack framework, GraphCodeAttack, to better evaluate the robustness of code models. Given a target code model, GraphCodeAttack automatically mines important code patterns, which can influence the model's decisions, to perturb the structure of input code to the model. To do so, GraphCodeAttack uses a set of input source codes to probe the model's outputs and identifies the \textit{discriminative} ASTs patterns that can influence the model decisions. GraphCodeAttack then selects appropriate AST patterns, concretizes the selected patterns as attacks, and inserts them as dead code into the model's input program. To effectively synthesize attacks from AST patterns, GraphCodeAttack uses a separate pre-trained code model to fill in the ASTs with concrete code snippets. We evaluate the robustness of two popular code models (e.g., CodeBERT and GraphCodeBERT) against our proposed approach on three tasks: Authorship Attribution, Vulnerability Prediction, and Clone Detection. The experimental results suggest that our proposed approach significantly outperforms state-of-the-art approaches in attacking code models such as CARROT and ALERT.

摘要: 预先训练的代码语言模型现在被广泛用于各种软件工程任务，如代码生成、代码完成、漏洞检测等。这反过来又给这些模型带来了安全和可靠性风险。其中一个重要的威胁是对抗性攻击，它会导致错误的预测，并在很大程度上影响模型在下游任务上的性能。当前针对代码模型的对抗性攻击通常采用固定的程序转换集，如变量重命名和死代码插入，导致攻击效果有限。为了应对上述挑战，我们提出了一种新的对抗性攻击框架GraphCodeAttack，以更好地评估代码模型的健壮性。在给定目标代码模型的情况下，GraphCodeAttack自动挖掘可能影响模型决策的重要代码模式，以扰乱模型的输入代码结构。为此，GraphCodeAttack使用一组输入源代码来探测模型的输出，并识别可能影响模型决策的\textit{鉴别性}ASTS模式。然后，GraphCodeAttack选择适当的AST模式，将所选模式具体化为攻击，并将它们作为死代码插入到模型的输入程序中。为了有效地从AST模式合成攻击，GraphCodeAttack使用单独的预先训练的代码模型来用具体的代码片段填充AST。我们评估了两个流行的代码模型(例如，CodeBERT和GraphCodeBERT)在作者属性、漏洞预测和克隆检测三个任务上的健壮性。实验结果表明，我们提出的方法在攻击胡萝卜和ALERT等代码模型方面明显优于最先进的方法。



## **18. Distributed Black-box Attack against Image Classification Cloud Services**

针对图像分类云服务的分布式黑盒攻击 cs.LG

8 pages, 10 figures

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2210.16371v3) [paper-pdf](http://arxiv.org/pdf/2210.16371v3)

**Authors**: Han Wu, Sareh Rowlands, Johan Wahlstrom

**Abstract**: Black-box adversarial attacks can fool image classifiers into misclassifying images without requiring access to model structure and weights. Recent studies have reported attack success rates of over 95% with less than 1,000 queries. The question then arises of whether black-box attacks have become a real threat against IoT devices that rely on cloud APIs to achieve image classification. To shed some light on this, note that prior research has primarily focused on increasing the success rate and reducing the number of queries. However, another crucial factor for black-box attacks against cloud APIs is the time required to perform the attack. This paper applies black-box attacks directly to cloud APIs rather than to local models, thereby avoiding mistakes made in prior research that applied the perturbation before image encoding and pre-processing. Further, we exploit load balancing to enable distributed black-box attacks that can reduce the attack time by a factor of about five for both local search and gradient estimation methods.

摘要: 黑盒对抗性攻击可以欺骗图像分类器对图像进行错误分类，而不需要访问模型结构和权重。最近的研究报告说，在不到1,000个查询的情况下，攻击成功率超过95%。随之而来的问题是，黑盒攻击是否已经成为对依赖云API实现图像分类的物联网设备的真正威胁。为了阐明这一点，请注意，以前的研究主要集中在提高成功率和减少查询数量上。然而，针对云API的黑盒攻击的另一个关键因素是执行攻击所需的时间。本文将黑盒攻击直接应用于云API而不是本地模型，从而避免了以往研究中在图像编码和预处理之前应用扰动的错误。此外，我们利用负载平衡来实现分布式黑盒攻击，对于局部搜索和梯度估计方法，可以将攻击时间减少约5倍。



## **19. A Man-in-the-Middle Attack against Object Detection Systems**

一种针对目标检测系统的中间人攻击 cs.RO

6 pages, 7 figures

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2208.07174v3) [paper-pdf](http://arxiv.org/pdf/2208.07174v3)

**Authors**: Han Wu, Sareh Rowlands, Johan Wahlstrom

**Abstract**: Object detection systems using deep learning models have become increasingly popular in robotics thanks to the rising power of CPUs and GPUs in embedded systems. However, these models are susceptible to adversarial attacks. While some attacks are limited by strict assumptions on access to the detection system, we propose a novel hardware attack inspired by Man-in-the-Middle attacks in cryptography. This attack generates an Universal Adversarial Perturbation (UAP) and then inject the perturbation between the USB camera and the detection system via a hardware attack. Besides, prior research is misled by an evaluation metric that measures the model accuracy rather than the attack performance. In combination with our proposed evaluation metrics, we significantly increases the strength of adversarial perturbations. These findings raise serious concerns for applications of deep learning models in safety-critical systems, such as autonomous driving.

摘要: 由于嵌入式系统中CPU和GPU的能力不断增强，使用深度学习模型的目标检测系统在机器人领域变得越来越受欢迎。然而，这些模型容易受到对抗性攻击。虽然一些攻击受到对检测系统访问权限的严格假设的限制，但我们受到密码学中中间人攻击的启发，提出了一种新的硬件攻击。该攻击产生通用对抗扰动(UAP)，然后通过硬件攻击在USB摄像头和检测系统之间注入扰动。此外，以前的研究被衡量模型准确性而不是攻击性能的评估指标所误导。与我们提出的评估度量相结合，我们显著增加了对抗性扰动的强度。这些发现引发了人们对深度学习模型在自动驾驶等安全关键系统中应用的严重担忧。



## **20. Spear and Shield: Adversarial Attacks and Defense Methods for Model-Based Link Prediction on Continuous-Time Dynamic Graphs**

矛盾：连续时间动态图上基于模型的链接预测的对抗性攻击和防御方法 cs.LG

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2308.10779v1) [paper-pdf](http://arxiv.org/pdf/2308.10779v1)

**Authors**: Dongjin Lee, Juho Lee, Kijung Shin

**Abstract**: Real-world graphs are dynamic, constantly evolving with new interactions, such as financial transactions in financial networks. Temporal Graph Neural Networks (TGNNs) have been developed to effectively capture the evolving patterns in dynamic graphs. While these models have demonstrated their superiority, being widely adopted in various important fields, their vulnerabilities against adversarial attacks remain largely unexplored. In this paper, we propose T-SPEAR, a simple and effective adversarial attack method for link prediction on continuous-time dynamic graphs, focusing on investigating the vulnerabilities of TGNNs. Specifically, before the training procedure of a victim model, which is a TGNN for link prediction, we inject edge perturbations to the data that are unnoticeable in terms of the four constraints we propose, and yet effective enough to cause malfunction of the victim model. Moreover, we propose a robust training approach T-SHIELD to mitigate the impact of adversarial attacks. By using edge filtering and enforcing temporal smoothness to node embeddings, we enhance the robustness of the victim model. Our experimental study shows that T-SPEAR significantly degrades the victim model's performance on link prediction tasks, and even more, our attacks are transferable to other TGNNs, which differ from the victim model assumed by the attacker. Moreover, we demonstrate that T-SHIELD effectively filters out adversarial edges and exhibits robustness against adversarial attacks, surpassing the link prediction performance of the naive TGNN by up to 11.2% under T-SPEAR.

摘要: 真实世界的图表是动态的，随着新的交互而不断发展，例如金融网络中的金融交易。时态图神经网络(TGNN)被用来有效地捕捉动态图中的演化模式。虽然这些模型已经显示出它们的优越性，在各个重要领域被广泛采用，但它们对对手攻击的脆弱性在很大程度上仍未被探索。在本文中，我们提出了一种简单有效的针对连续时间动态图的链接预测的对抗性攻击方法T-SPEAR，重点研究了TGNN的脆弱性。具体地说，在受害者模型(用于链接预测的TGNN)的训练过程之前，我们向数据注入边缘扰动，这些扰动在我们提出的四个约束条件下是不明显的，但足够有效地导致受害者模型故障。此外，我们提出了一种健壮的训练方法T-Shield来减轻对抗性攻击的影响。通过使用边缘滤波和对节点嵌入进行时间平滑，增强了受害者模型的稳健性。我们的实验研究表明，T-SPEAR显著降低了受害者模型在链接预测任务中的性能，而且我们的攻击可以转移到其他不同于攻击者假设的受害者模型的TGNN上。此外，我们还证明了T-Shield能够有效地过滤掉敌意边缘，并表现出对敌意攻击的健壮性，在T-Spear环境下比朴素的TGNN的链路预测性能高出11.2%。



## **21. Adversarial Attacks and Defenses for Semantic Communication in Vehicular Metaverses**

车载变形器中语义通信的对抗性攻击与防御 cs.CR

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2306.03528v2) [paper-pdf](http://arxiv.org/pdf/2306.03528v2)

**Authors**: Jiawen Kang, Jiayi He, Hongyang Du, Zehui Xiong, Zhaohui Yang, Xumin Huang, Shengli Xie

**Abstract**: For vehicular metaverses, one of the ultimate user-centric goals is to optimize the immersive experience and Quality of Service (QoS) for users on board. Semantic Communication (SemCom) has been introduced as a revolutionary paradigm that significantly eases communication resource pressure for vehicular metaverse applications to achieve this goal. SemCom enables high-quality and ultra-efficient vehicular communication, even with explosively increasing data traffic among vehicles. In this article, we propose a hierarchical SemCom-enabled vehicular metaverses framework consisting of the global metaverse, local metaverses, SemCom module, and resource pool. The global and local metaverses are brand-new concepts from the metaverse's distribution standpoint. Considering the QoS of users, this article explores the potential security vulnerabilities of the proposed framework. To that purpose, this study highlights a specific security risk to the framework's SemCom module and offers a viable defense solution, so encouraging community researchers to focus more on vehicular metaverse security. Finally, we provide an overview of the open issues of secure SemCom in the vehicular metaverses, notably pointing out potential future research directions.

摘要: 对于车载虚拟现实，以用户为中心的最终目标之一是优化车载用户的身临其境体验和服务质量(Qos)。语义通信(SemCom)作为一种革命性的范式被引入，它显著缓解了车载虚拟现实应用为实现这一目标而带来的通信资源压力。SemCom实现了高质量和超高效的车辆通信，即使车辆之间的数据流量呈爆炸性增长。在本文中，我们提出了一个支持SemCom的层次化车载虚拟世界框架，该框架由全局虚拟世界、局部虚拟世界、SemCom模块和资源池组成。从虚拟世界的分布角度来看，全球虚拟世界和局部虚拟世界是一个全新的概念。考虑到用户的服务质量，本文探讨了该框架潜在的安全漏洞。为此，这项研究强调了框架的SemCom模块的特定安全风险，并提供了可行的防御解决方案，因此鼓励社区研究人员更多地关注车辆虚拟现实安全。最后，对车载虚拟现实中的安全SemCom存在的问题进行了总结，指出了未来可能的研究方向。



## **22. Boosting Adversarial Attack with Similar Target**

用相似的目标加强对抗性攻击 cs.CV

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2308.10743v1) [paper-pdf](http://arxiv.org/pdf/2308.10743v1)

**Authors**: Shuo Zhang, Ziruo Wang, Zikai Zhou, Huanran Chen

**Abstract**: Deep neural networks are vulnerable to adversarial examples, posing a threat to the models' applications and raising security concerns. An intriguing property of adversarial examples is their strong transferability. Several methods have been proposed to enhance transferability, including ensemble attacks which have demonstrated their efficacy. However, prior approaches simply average logits, probabilities, or losses for model ensembling, lacking a comprehensive analysis of how and why model ensembling significantly improves transferability. In this paper, we propose a similar targeted attack method named Similar Target~(ST). By promoting cosine similarity between the gradients of each model, our method regularizes the optimization direction to simultaneously attack all surrogate models. This strategy has been proven to enhance generalization ability. Experimental results on ImageNet validate the effectiveness of our approach in improving adversarial transferability. Our method outperforms state-of-the-art attackers on 18 discriminative classifiers and adversarially trained models.

摘要: 深度神经网络很容易受到敌意例子的攻击，这对模型的应用构成了威胁，并引发了安全担忧。对抗性例子的一个耐人寻味的特点是它们具有很强的可转移性。已经提出了几种提高可转移性的方法，包括已经证明其有效性的集合攻击。然而，以前的方法只是对模型集成的对数、概率或损失进行平均，缺乏对模型集成如何以及为什么显著提高可转移性的全面分析。本文提出了一种类似的目标攻击方法--相似目标~(ST)。通过提高每个模型梯度之间的余弦相似度，我们的方法将优化方向正则化以同时攻击所有代理模型。实践证明，该策略提高了泛化能力。在ImageNet上的实验结果验证了该方法在提高对手可转移性方面的有效性。我们的方法在18个区分分类器和对抗性训练的模型上优于最先进的攻击者。



## **23. SALSy: Security-Aware Layout Synthesis**

SALSy：安全感知的版图合成 cs.CR

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2308.06201v2) [paper-pdf](http://arxiv.org/pdf/2308.06201v2)

**Authors**: Mohammad Eslami, Tiago Perez, Samuel Pagliarini

**Abstract**: Integrated Circuits (ICs) are the target of diverse attacks during their lifetime. Fabrication-time attacks, such as the insertion of Hardware Trojans, can give an adversary access to privileged data and/or the means to corrupt the IC's internal computation. Post-fabrication attacks, where the end-user takes a malicious role, also attempt to obtain privileged information through means such as fault injection and probing. Taking these threats into account and at the same time, this paper proposes a methodology for Security-Aware Layout Synthesis (SALSy), such that ICs can be designed with security in mind in the same manner as power-performance-area (PPA) metrics are considered today, a concept known as security closure. Furthermore, the trade-offs between PPA and security are considered and a chip is fabricated in a 65nm CMOS commercial technology for validation purposes - a feature not seen in previous research on security closure. Measurements on the fabricated ICs indicate that SALSy promotes a modest increase in power in order to achieve significantly improved security metrics.

摘要: 集成电路(IC)在其生命周期内是各种攻击的目标。制造时间攻击，如插入硬件特洛伊木马，可让对手访问特权数据和/或破坏IC内部计算的手段。伪造后攻击，其中终端用户扮演恶意角色，还试图通过故障注入和探测等手段获取特权信息。考虑到这些威胁，本文提出了一种安全感知版图综合(SALSy)的方法，使得IC的设计可以像当今考虑功率-性能-面积(PPA)度量的方式一样考虑安全，这一概念被称为安全闭包。此外，考虑了PPA和安全性之间的权衡，为了验证的目的，芯片以65 nm的商业工艺制造--这是以前关于安全闭合的研究中没有看到的特征。对制造的IC的测量表明，SALSy促进了功率的适度增加，以实现显著改善的安全指标。



## **24. On the Adversarial Robustness of Multi-Modal Foundation Models**

多通道基础模型的对抗稳健性研究 cs.LG

ICCV AROW 2023

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2308.10741v1) [paper-pdf](http://arxiv.org/pdf/2308.10741v1)

**Authors**: Christian Schlarmann, Matthias Hein

**Abstract**: Multi-modal foundation models combining vision and language models such as Flamingo or GPT-4 have recently gained enormous interest. Alignment of foundation models is used to prevent models from providing toxic or harmful output. While malicious users have successfully tried to jailbreak foundation models, an equally important question is if honest users could be harmed by malicious third-party content. In this paper we show that imperceivable attacks on images in order to change the caption output of a multi-modal foundation model can be used by malicious content providers to harm honest users e.g. by guiding them to malicious websites or broadcast fake information. This indicates that countermeasures to adversarial attacks should be used by any deployed multi-modal foundation model.

摘要: 将视觉和语言模型相结合的多模式基础模型，如Flamingo或GPT-4，最近获得了巨大的兴趣。基础模型的对齐用于防止模型提供有毒或有害的输出。虽然恶意用户已经成功地试图越狱Foundation Model，但同样重要的问题是，诚实的用户是否会受到恶意第三方内容的伤害。在本文中，我们证明了恶意内容提供商可以利用对图像的不可见攻击来改变多模式基础模型的字幕输出，以伤害诚实的用户，例如通过将他们引导到恶意网站或传播虚假信息。这表明，任何部署的多模式基础模型都应该使用对抗攻击的对策。



## **25. Model-free False Data Injection Attack in Networked Control Systems: A Feedback Optimization Approach**

网络控制系统中的无模型虚假数据注入攻击：一种反馈优化方法 math.OC

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2212.07633v2) [paper-pdf](http://arxiv.org/pdf/2212.07633v2)

**Authors**: Xiaoyu Luo, Chongrong Fang, Jianping He, Chengcheng Zhao, Dario Paccagnan

**Abstract**: Security issues have gathered growing interest within the control systems community, as physical components and communication networks are increasingly vulnerable to cyber attacks. In this context, recent literature has studied increasingly sophisticated \emph{false data injection} attacks, with the aim to design mitigative measures that improve the systems' security. Notably, data-driven attack strategies -- whereby the system dynamics is oblivious to the adversary -- have received increasing attention. However, many of the existing works on the topic rely on the implicit assumption of linear system dynamics, significantly limiting their scope. Contrary to that, in this work we design and analyze \emph{truly} model-free false data injection attack that applies to general linear and nonlinear systems. More specifically, we aim at designing an injected signal that steers the output of the system toward a (maliciously chosen) trajectory. We do so by designing a zeroth-order feedback optimization policy and jointly use probing signals for real-time measurements. We then characterize the quality of the proposed model-free attack through its optimality gap, which is affected by the dimensions of the attack signal, the number of iterations performed, and the convergence rate of the system. Finally, we extend the proposed attack scheme to the systems with internal noise. Extensive simulations show the effectiveness of the proposed attack scheme.

摘要: 随着物理组件和通信网络越来越容易受到网络攻击，安全问题在控制系统社区内引起了越来越大的兴趣。在这种背景下，最近的文献研究了越来越复杂的\emph{虚假数据注入}攻击，目的是设计缓解措施来提高系统的安全性。值得注意的是，数据驱动的攻击策略--系统动态对对手视而不见--受到了越来越多的关注。然而，现有的许多关于这一主题的工作依赖于线性系统动力学的隐含假设，这大大限制了它们的范围。与此相反，本文设计并分析了适用于一般线性和非线性系统的无模型伪数据注入攻击。更具体地说，我们的目标是设计一种注入信号，将系统的输出引导到(恶意选择的)轨迹。为此，我们设计了一种零阶反馈优化策略，并联合使用探测信号进行实时测量。然后，我们通过最优性间隙来表征所提出的无模型攻击的质量，该最优性间隙受攻击信号的维度、迭代次数和系统的收敛速度的影响。最后，我们将所提出的攻击方案扩展到具有内部噪声的系统。大量的仿真实验表明了该攻击方案的有效性。



## **26. Measuring the Effect of Causal Disentanglement on the Adversarial Robustness of Neural Network Models**

因果解缠对神经网络模型对抗稳健性的影响 cs.LG

12 pages, 3 figures

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2308.10708v1) [paper-pdf](http://arxiv.org/pdf/2308.10708v1)

**Authors**: Preben M. Ness, Dusica Marijan, Sunanda Bose

**Abstract**: Causal Neural Network models have shown high levels of robustness to adversarial attacks as well as an increased capacity for generalisation tasks such as few-shot learning and rare-context classification compared to traditional Neural Networks. This robustness is argued to stem from the disentanglement of causal and confounder input signals. However, no quantitative study has yet measured the level of disentanglement achieved by these types of causal models or assessed how this relates to their adversarial robustness.   Existing causal disentanglement metrics are not applicable to deterministic models trained on real-world datasets. We, therefore, utilise metrics of content/style disentanglement from the field of Computer Vision to measure different aspects of the causal disentanglement for four state-of-the-art causal Neural Network models. By re-implementing these models with a common ResNet18 architecture we are able to fairly measure their adversarial robustness on three standard image classification benchmarking datasets under seven common white-box attacks. We find a strong association (r=0.820, p=0.001) between the degree to which models decorrelate causal and confounder signals and their adversarial robustness. Additionally, we find a moderate negative association between the pixel-level information content of the confounder signal and adversarial robustness (r=-0.597, p=0.040).

摘要: 与传统神经网络相比，因果神经网络模型表现出对敌意攻击的高度稳健性，以及更强的泛化任务能力，如少机会学习和稀有上下文分类。这种稳健性被认为源于因果和混淆输入信号的分离。然而，还没有一项定量研究衡量这些类型的因果模型所实现的解缠水平，或者评估这与它们的对抗健壮性之间的关系。现有的因果解缠度量不适用于在真实世界数据集上训练的确定性模型。因此，我们利用计算机视觉领域的内容/样式解缠度量来测量四种最先进的因果神经网络模型的因果解缠的不同方面。通过使用通用的ResNet18架构重新实现这些模型，我们能够在七种常见的白盒攻击下，在三个标准图像分类基准数据集上公平地衡量它们的对抗健壮性。我们发现模型去相关因果信号和混杂信号的程度与它们的对抗稳健性之间存在很强的关联(r=0.820，p=0.001)。此外，我们发现混杂信号的像素级信息含量与对抗稳健性之间存在中等程度的负相关(r=-0.597，p=0.040)。



## **27. Transferable Attack for Semantic Segmentation**

语义分词的可转移攻击 cs.CV

Source code is available at: https://github.com/anucvers/TASS

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2307.16572v2) [paper-pdf](http://arxiv.org/pdf/2307.16572v2)

**Authors**: Mengqi He, Jing Zhang, Zhaoyuan Yang, Mingyi He, Nick Barnes, Yuchao Dai

**Abstract**: We analysis performance of semantic segmentation models wrt. adversarial attacks, and observe that the adversarial examples generated from a source model fail to attack the target models. i.e The conventional attack methods, such as PGD and FGSM, do not transfer well to target models, making it necessary to study the transferable attacks, especially transferable attacks for semantic segmentation. We find two main factors to achieve transferable attack. Firstly, the attack should come with effective data augmentation and translation-invariant features to deal with unseen models. Secondly, stabilized optimization strategies are needed to find the optimal attack direction. Based on the above observations, we propose an ensemble attack for semantic segmentation to achieve more effective attacks with higher transferability. The source code and experimental results are publicly available via our project page: https://github.com/anucvers/TASS.

摘要: 分析了语义切分模型WRT的性能。对抗性攻击，并观察到从源模型生成的对抗性示例无法攻击目标模型。也就是说，传统的攻击方法，如PGD和FGSM，不能很好地向目标模型转化，因此有必要研究可转移攻击，特别是用于语义分割的可转移攻击。我们发现了实现可转移进攻的两个主要因素。首先，攻击应该具有有效的数据增强和平移不变特征来处理不可见的模型。其次，需要稳定的优化策略来寻找最优的攻击方向。基于以上观察结果，我们提出了一种语义分割的集成攻击，以实现更有效的攻击，并具有更高的可转移性。源代码和实验结果可通过我们的项目页面公开获得：https://github.com/anucvers/TASS.



## **28. FLARE: Fingerprinting Deep Reinforcement Learning Agents using Universal Adversarial Masks**

FLARE：使用通用对抗面具对深度强化学习代理进行指纹识别 cs.LG

Will appear in the proceedings of ACSAC 2023; 13 pages, 5 figures, 7  tables

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2307.14751v2) [paper-pdf](http://arxiv.org/pdf/2307.14751v2)

**Authors**: Buse G. A. Tekgul, N. Asokan

**Abstract**: We propose FLARE, the first fingerprinting mechanism to verify whether a suspected Deep Reinforcement Learning (DRL) policy is an illegitimate copy of another (victim) policy. We first show that it is possible to find non-transferable, universal adversarial masks, i.e., perturbations, to generate adversarial examples that can successfully transfer from a victim policy to its modified versions but not to independently trained policies. FLARE employs these masks as fingerprints to verify the true ownership of stolen DRL policies by measuring an action agreement value over states perturbed via such masks. Our empirical evaluations show that FLARE is effective (100% action agreement on stolen copies) and does not falsely accuse independent policies (no false positives). FLARE is also robust to model modification attacks and cannot be easily evaded by more informed adversaries without negatively impacting agent performance. We also show that not all universal adversarial masks are suitable candidates for fingerprints due to the inherent characteristics of DRL policies. The spatio-temporal dynamics of DRL problems and sequential decision-making process make characterizing the decision boundary of DRL policies more difficult, as well as searching for universal masks that capture the geometry of it.

摘要: 我们提出了FLARE，这是第一个验证可疑的深度强化学习(DRL)策略是否是另一个(受害者)策略的非法副本的指纹识别机制。我们首先证明了可以找到不可转移的通用对抗掩码，即扰动，以生成能够成功地从受害者策略转移到其修改版本但不能转移到独立训练策略的对抗例子。FLARE使用这些掩码作为指纹，通过测量通过此类掩码扰乱的州的行动协议值来验证被盗DRL保单的真实所有权。我们的经验评估表明，FLARE是有效的(100%对被盗副本的操作协议)，并且不会错误地指责独立政策(没有假阳性)。FLARE对模型修改攻击也很强大，在不对代理性能产生负面影响的情况下，更知情的对手很难躲避。我们还表明，由于DRL策略的固有特征，并不是所有的通用对抗面具都适合指纹候选。DRL问题的时空动态和顺序决策过程使得刻画DRL策略的决策边界变得更加困难，同时也使得寻找捕捉其几何形状的通用掩码变得更加困难。



## **29. Improving the Transferability of Adversarial Examples with Arbitrary Style Transfer**

用任意风格转移提高对抗性例句的可转移性 cs.CV

10 pages, 2 figures, accepted by the 31st ACM International  Conference on Multimedia (MM '23)

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2308.10601v1) [paper-pdf](http://arxiv.org/pdf/2308.10601v1)

**Authors**: Zhijin Ge, Fanhua Shang, Hongying Liu, Yuanyuan Liu, Liang Wan, Wei Feng, Xiaosen Wang

**Abstract**: Deep neural networks are vulnerable to adversarial examples crafted by applying human-imperceptible perturbations on clean inputs. Although many attack methods can achieve high success rates in the white-box setting, they also exhibit weak transferability in the black-box setting. Recently, various methods have been proposed to improve adversarial transferability, in which the input transformation is one of the most effective methods. In this work, we notice that existing input transformation-based works mainly adopt the transformed data in the same domain for augmentation. Inspired by domain generalization, we aim to further improve the transferability using the data augmented from different domains. Specifically, a style transfer network can alter the distribution of low-level visual features in an image while preserving semantic content for humans. Hence, we propose a novel attack method named Style Transfer Method (STM) that utilizes a proposed arbitrary style transfer network to transform the images into different domains. To avoid inconsistent semantic information of stylized images for the classification network, we fine-tune the style transfer network and mix up the generated images added by random noise with the original images to maintain semantic consistency and boost input diversity. Extensive experimental results on the ImageNet-compatible dataset show that our proposed method can significantly improve the adversarial transferability on either normally trained models or adversarially trained models than state-of-the-art input transformation-based attacks. Code is available at: https://github.com/Zhijin-Ge/STM.

摘要: 深度神经网络很容易受到敌意例子的攻击，这些例子是通过对干净的输入应用人类无法察觉的扰动来制作的。虽然许多攻击方法在白盒环境下可以达到很高的成功率，但在黑盒环境下也表现出较弱的可移植性。近年来，人们提出了各种方法来提高对抗性转移能力，其中输入变换是最有效的方法之一。在这项工作中，我们注意到，现有的基于输入转换的工作主要是采用同一领域中的转换后的数据进行扩充。受领域泛化的启发，我们的目标是利用来自不同领域的扩展数据来进一步提高可转移性。具体地说，风格传递网络可以改变图像中低级视觉特征的分布，同时为人类保留语义内容。因此，我们提出了一种新的攻击方法，称为样式传递方法(STM)，它利用提出的任意样式传递网络将图像变换到不同的域。为了避免分类网络中风格化图像的语义信息不一致，我们对样式传递网络进行了微调，并将添加了随机噪声的生成图像与原始图像混合在一起，以保持语义的一致性，提高输入的多样性。在ImageNet兼容的数据集上的大量实验结果表明，无论是在正常训练的模型上还是在对抗性训练的模型上，我们提出的方法都能显著提高基于输入变换的攻击的对抗性。代码可从以下网址获得：https://github.com/Zhijin-Ge/STM.



## **30. Single-User Injection for Invisible Shilling Attack against Recommender Systems**

针对推荐系统的隐形先令攻击的单用户注入 cs.IR

CIKM 2023. 10 pages, 5 figures

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2308.10467v1) [paper-pdf](http://arxiv.org/pdf/2308.10467v1)

**Authors**: Chengzhi Huang, Hui Li

**Abstract**: Recommendation systems (RS) are crucial for alleviating the information overload problem. Due to its pivotal role in guiding users to make decisions, unscrupulous parties are lured to launch attacks against RS to affect the decisions of normal users and gain illegal profits. Among various types of attacks, shilling attack is one of the most subsistent and profitable attacks. In shilling attack, an adversarial party injects a number of well-designed fake user profiles into the system to mislead RS so that the attack goal can be achieved. Although existing shilling attack methods have achieved promising results, they all adopt the attack paradigm of multi-user injection, where some fake user profiles are required. This paper provides the first study of shilling attack in an extremely limited scenario: only one fake user profile is injected into the victim RS to launch shilling attacks (i.e., single-user injection). We propose a novel single-user injection method SUI-Attack for invisible shilling attack. SUI-Attack is a graph based attack method that models shilling attack as a node generation task over the user-item bipartite graph of the victim RS, and it constructs the fake user profile by generating user features and edges that link the fake user to items. Extensive experiments demonstrate that SUI-Attack can achieve promising attack results in single-user injection. In addition to its attack power, SUI-Attack increases the stealthiness of shilling attack and reduces the risk of being detected. We provide our implementation at: https://github.com/KDEGroup/SUI-Attack.

摘要: 推荐系统是缓解信息过载问题的关键。由于RS在引导用户决策方面起着举足轻重的作用，引诱不法分子对RS发起攻击，影响正常用户的决策，获取非法利润。在各种类型的攻击中，先令攻击是最存在和最有利可图的攻击之一。在先令攻击中，敌意方向系统注入大量精心设计的虚假用户档案，误导RS，从而达到攻击目的。虽然现有的先令攻击方法已经取得了很好的效果，但它们都采用了多用户注入的攻击范式，需要一些虚假的用户模板。本文首次研究了在一个极其有限的场景下的先令攻击：仅向受害者RS注入一个伪用户配置文件来发起先令攻击(即单用户注入)。针对隐形先令攻击，提出了一种新的单用户注入方法SUI-Attack。Sui-Attack是一种基于图的攻击方法，它将先令攻击建模为受害者RS的用户-项目二部图上的节点生成任务，并通过生成将虚假用户链接到项目的用户特征和边来构建虚假用户配置文件。大量的实验表明，SUI-Attack能够在单用户注入中获得良好的攻击效果。除了攻击能力外，SUI-Attack还增加了先令攻击的隐蔽性，降低了被发现的风险。我们通过以下网址提供实施方案：https://github.com/KDEGroup/SUI-Attack.



## **31. Adaptive Local Steps Federated Learning with Differential Privacy Driven by Convergence Analysis**

收敛分析驱动的差分隐私自适应局部步长联合学习 cs.LG

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2308.10457v1) [paper-pdf](http://arxiv.org/pdf/2308.10457v1)

**Authors**: Xinpeng Ling, Jie Fu, Zhili Chen

**Abstract**: Federated Learning (FL) is a distributed machine learning technique that allows model training among multiple devices or organizations without sharing data. However, while FL ensures that the raw data is not directly accessible to external adversaries, adversaries can still obtain some statistical information about the data through differential attacks. Differential Privacy (DP) has been proposed, which adds noise to the model or gradients to prevent adversaries from inferring private information from the transmitted parameters. We reconsider the framework of differential privacy federated learning in resource-constrained scenarios (privacy budget and communication resources). We analyze the convergence of federated learning with differential privacy (DPFL) on resource-constrained scenarios and propose an Adaptive Local Steps Differential Privacy Federated Learning (ALS-DPFL) algorithm. We experiment our algorithm on the FashionMNIST and Cifar-10 datasets and achieve quite good performance relative to previous work.

摘要: 联合学习(FL)是一种分布式机器学习技术，允许在不共享数据的情况下在多个设备或组织之间进行模型训练。然而，虽然FL确保外部攻击者不能直接访问原始数据，但攻击者仍然可以通过差异攻击获得有关数据的一些统计信息。差分隐私(DP)被提出，它在模型中添加噪声或梯度，以防止攻击者从传输的参数中推断出私人信息。我们重新考虑了资源受限场景(隐私预算和通信资源)下的差异隐私联合学习框架。分析了具有差异隐私的联邦学习(DPFL)在资源受限场景下的收敛问题，提出了一种自适应局部步长差分隐私联邦学习(ALS-DPFL)算法。我们在FashionMNIST和CIFAR-10数据集上进行了实验，取得了较好的性能。



## **32. Quantum Query Lower Bounds for Key Recovery Attacks on the Even-Mansour Cipher**

Even-Mansour密码密钥恢复攻击的量子查询下界 quant-ph

**SubmitDate**: 2023-08-21    [abs](http://arxiv.org/abs/2308.10418v1) [paper-pdf](http://arxiv.org/pdf/2308.10418v1)

**Authors**: Akinori Kawachi, Yuki Naito

**Abstract**: The Even-Mansour (EM) cipher is one of the famous constructions for a block cipher. Kuwakado and Morii demonstrated that a quantum adversary can recover its $n$-bit secret keys only with $O(n)$ nonadaptive quantum queries. While the security of the EM cipher and its variants is well-understood for classical adversaries, very little is currently known of their quantum security. Towards a better understanding of the quantum security, or the limits of quantum adversaries for the EM cipher, we study the quantum query complexity for the key recovery of the EM cipher and prove every quantum algorithm requires $\Omega(n)$ quantum queries for the key recovery even if it is allowed to make adaptive queries. Therefore, the quantum attack of Kuwakado and Morii has the optimal query complexity up to a constant factor, and we cannot asymptotically improve it even with adaptive quantum queries.

摘要: 偶-曼苏尔(EM)密码是分组密码的著名构造之一。Kukkado和Morii证明了量子敌手只能用$O(N)$非自适应量子查询恢复其$n$比特密钥。虽然EM密码及其变体的安全性对于经典对手来说是众所周知的，但目前对它们的量子安全性知之甚少。为了更好地理解EM密码的量子安全性或量子对手对EM密码的限制，我们研究了EM密码密钥恢复的量子查询复杂性，证明了每个量子算法都需要$Omega(N)$量子查询来恢复密钥，即使它被允许进行自适应查询。因此，Kukkado和Morii的量子攻击具有高达常数因子的最优查询复杂度，即使使用自适应量子查询也不能渐近改善它。



## **33. HoSNN: Adversarially-Robust Homeostatic Spiking Neural Networks with Adaptive Firing Thresholds**

HoSNN：具有自适应放电阈值的逆鲁棒自适应稳态尖峰神经网络 cs.NE

**SubmitDate**: 2023-08-20    [abs](http://arxiv.org/abs/2308.10373v1) [paper-pdf](http://arxiv.org/pdf/2308.10373v1)

**Authors**: Hejia Geng, Peng Li

**Abstract**: Spiking neural networks (SNNs) offer promise for efficient and powerful neurally inspired computation. Common to other types of neural networks, however, SNNs face the severe issue of vulnerability to adversarial attacks. We present the first study that draws inspiration from neural homeostasis to develop a bio-inspired solution that counters the susceptibilities of SNNs to adversarial onslaughts. At the heart of our approach is a novel threshold-adapting leaky integrate-and-fire (TA-LIF) neuron model, which we adopt to construct the proposed adversarially robust homeostatic SNN (HoSNN). Distinct from traditional LIF models, our TA-LIF model incorporates a self-stabilizing dynamic thresholding mechanism, curtailing adversarial noise propagation and safeguarding the robustness of HoSNNs in an unsupervised manner. Theoretical analysis is presented to shed light on the stability and convergence properties of the TA-LIF neurons, underscoring their superior dynamic robustness under input distributional shifts over traditional LIF neurons. Remarkably, without explicit adversarial training, our HoSNNs demonstrate inherent robustness on CIFAR-10, with accuracy improvements to 72.6% and 54.19% against FGSM and PGD attacks, up from 20.97% and 0.6%, respectively. Furthermore, with minimal FGSM adversarial training, our HoSNNs surpass previous models by 29.99% under FGSM and 47.83% under PGD attacks on CIFAR-10. Our findings offer a new perspective on harnessing biological principles for bolstering SNNs adversarial robustness and defense, paving the way to more resilient neuromorphic computing.

摘要: 尖峰神经网络(SNN)为高效和强大的神经启发计算提供了希望。然而，与其他类型的神经网络一样，SNN面临着易受对手攻击的严重问题。我们介绍了第一项从神经动态平衡中获得灵感的研究，以开发一种生物启发的解决方案，以对抗SNN对对手攻击的敏感性。我们方法的核心是一种新的阈值自适应泄漏积分与点火(TA-LIF)神经元模型，我们采用该模型来构造所提出的对抗性鲁棒自稳态SNN(HoSNN)。与传统的LIF模型不同，TA-LIF模型引入了一种自稳定的动态阈值机制，在无监督的情况下抑制了敌对噪声的传播，保护了HoSNN的健壮性。理论分析揭示了TA-LIF神经元的稳定性和收敛特性，强调了其在输入分布漂移下优于传统LIF神经元的动态鲁棒性。值得注意的是，在没有明确的对抗性训练的情况下，我们的HoSNN对CIFAR-10表现出固有的鲁棒性，对FGSM和PGD攻击的准确率分别从20.97%和0.6%提高到72.6%和54.19%。此外，在最少的FGSM对抗训练下，我们的HoSNN在FGSM下超过了以前的模型29.99%，在PGD攻击CIFAR-10下超过了47.83%。我们的发现为利用生物学原理加强SNN对抗的稳健性和防御提供了一个新的视角，为更具弹性的神经形态计算铺平了道路。



## **34. Hiding Backdoors within Event Sequence Data via Poisoning Attacks**

通过中毒攻击在事件序列数据中隐藏后门 cs.LG

**SubmitDate**: 2023-08-20    [abs](http://arxiv.org/abs/2308.10201v1) [paper-pdf](http://arxiv.org/pdf/2308.10201v1)

**Authors**: Elizaveta Kovtun, Alina Ermilova, Dmitry Berestnev, Alexey Zaytsev

**Abstract**: The financial industry relies on deep learning models for making important decisions. This adoption brings new danger, as deep black-box models are known to be vulnerable to adversarial attacks. In computer vision, one can shape the output during inference by performing an adversarial attack called poisoning via introducing a backdoor into the model during training. For sequences of financial transactions of a customer, insertion of a backdoor is harder to perform, as models operate over a more complex discrete space of sequences, and systematic checks for insecurities occur. We provide a method to introduce concealed backdoors, creating vulnerabilities without altering their functionality for uncontaminated data. To achieve this, we replace a clean model with a poisoned one that is aware of the availability of a backdoor and utilize this knowledge. Our most difficult for uncovering attacks include either additional supervised detection step of poisoned data activated during the test or well-hidden model weight modifications. The experimental study provides insights into how these effects vary across different datasets, architectures, and model components. Alternative methods and baselines, such as distillation-type regularization, are also explored but found to be less efficient. Conducted on three open transaction datasets and architectures, including LSTM, CNN, and Transformer, our findings not only illuminate the vulnerabilities in contemporary models but also can drive the construction of more robust systems.

摘要: 金融业依靠深度学习模型来做出重要决策。这种采用带来了新的危险，因为众所周知，深黑盒模型很容易受到对手的攻击。在计算机视觉中，人们可以在推理过程中通过在训练期间将后门引入模型来执行称为中毒的对抗性攻击来塑造输出。对于客户的金融交易序列，插入后门更难执行，因为模型在更复杂的离散序列空间上操作，并对不安全性进行系统检查。我们提供了一种方法来引入隐藏的后门，在不改变其针对未受污染数据的功能的情况下创建漏洞。为了实现这一点，我们用知道后门可用的有毒模型替换干净的模型，并利用这一知识。我们最难发现的攻击包括在测试期间激活的有毒数据的额外监督检测步骤，或者隐藏良好的模型权重修改。该实验研究深入了解了这些影响在不同数据集、体系结构和模型组件中的差异。还探索了其他方法和基线，如蒸馏型正则化，但发现效率较低。我们在包括LSTM、CNN和Transformer在内的三个开放事务数据集和体系结构上进行了实验，我们的发现不仅揭示了当代模型中的漏洞，而且可以推动构建更健壮的系统。



## **35. Intriguing Properties of Text-guided Diffusion Models**

文本引导扩散模型的有趣性质 cs.CV

Project page: https://sage-diffusion.github.io/

**SubmitDate**: 2023-08-19    [abs](http://arxiv.org/abs/2306.00974v4) [paper-pdf](http://arxiv.org/pdf/2306.00974v4)

**Authors**: Qihao Liu, Adam Kortylewski, Yutong Bai, Song Bai, Alan Yuille

**Abstract**: Text-guided diffusion models (TDMs) are widely applied but can fail unexpectedly. Common failures include: (i) natural-looking text prompts generating images with the wrong content, or (ii) different random samples of the latent variables that generate vastly different, and even unrelated, outputs despite being conditioned on the same text prompt. In this work, we aim to study and understand the failure modes of TDMs in more detail. To achieve this, we propose SAGE, an adversarial attack on TDMs that uses image classifiers as surrogate loss functions, to search over the discrete prompt space and the high-dimensional latent space of TDMs to automatically discover unexpected behaviors and failure cases in the image generation. We make several technical contributions to ensure that SAGE finds failure cases of the diffusion model, rather than the classifier, and verify this in a human study. Our study reveals four intriguing properties of TDMs that have not been systematically studied before: (1) We find a variety of natural text prompts producing images that fail to capture the semantics of input texts. We categorize these failures into ten distinct types based on the underlying causes. (2) We find samples in the latent space (which are not outliers) that lead to distorted images independent of the text prompt, suggesting that parts of the latent space are not well-structured. (3) We also find latent samples that lead to natural-looking images which are unrelated to the text prompt, implying a potential misalignment between the latent and prompt spaces. (4) By appending a single adversarial token embedding to an input prompt we can generate a variety of specified target objects, while only minimally affecting the CLIP score. This demonstrates the fragility of language representations and raises potential safety concerns. Project page: https://sage-diffusion.github.io/

摘要: 文本引导扩散模型(TDM)被广泛应用，但可能会意外失败。常见的故障包括：(I)看起来自然的文本提示生成具有错误内容的图像，或(Ii)潜在变量的不同随机样本，尽管以相同的文本提示为条件，但生成的输出却截然不同，甚至是无关的。在这项工作中，我们旨在更详细地研究和理解TDM的故障模式。为此，我们提出了一种针对TDMS的对抗性攻击方法SAGE，它使用图像分类器作为代理损失函数，在TDMS的离散提示空间和高维潜在空间中进行搜索，自动发现图像生成中的意外行为和失败案例。我们做出了几项技术贡献，以确保SAGE找到扩散模型的故障案例，而不是分类器，并在人体研究中验证这一点。我们的研究揭示了TDM的四个以前没有被系统研究的有趣的性质：(1)我们发现各种自然的文本提示产生的图像无法捕捉到输入文本的语义。根据根本原因，我们将这些故障分为十种不同的类型。(2)我们在潜在空间(不是离群点)中发现了与文本提示无关的导致失真图像的样本，这表明潜在空间的部分结构不是良好的。(3)我们还发现潜在样本导致了与文本提示无关的看起来自然的图像，这意味着潜在空间和提示空间之间存在潜在的错位。(4)通过在输入提示中添加单个对抗性令牌，我们可以生成各种指定的目标对象，而对片段得分的影响最小。这表明了语言表达的脆弱性，并引发了潜在的安全问题。项目页面：https://sage-diffusion.github.io/



## **36. A Comparison of Adversarial Learning Techniques for Malware Detection**

恶意软件检测中对抗性学习技术的比较 cs.CR

**SubmitDate**: 2023-08-19    [abs](http://arxiv.org/abs/2308.09958v1) [paper-pdf](http://arxiv.org/pdf/2308.09958v1)

**Authors**: Pavla Louthánová, Matouš Kozák, Martin Jureček, Mark Stamp

**Abstract**: Machine learning has proven to be a useful tool for automated malware detection, but machine learning models have also been shown to be vulnerable to adversarial attacks. This article addresses the problem of generating adversarial malware samples, specifically malicious Windows Portable Executable files. We summarize and compare work that has focused on adversarial machine learning for malware detection. We use gradient-based, evolutionary algorithm-based, and reinforcement-based methods to generate adversarial samples, and then test the generated samples against selected antivirus products. We compare the selected methods in terms of accuracy and practical applicability. The results show that applying optimized modifications to previously detected malware can lead to incorrect classification of the file as benign. It is also known that generated malware samples can be successfully used against detection models other than those used to generate them and that using combinations of generators can create new samples that evade detection. Experiments show that the Gym-malware generator, which uses a reinforcement learning approach, has the greatest practical potential. This generator achieved an average sample generation time of 5.73 seconds and the highest average evasion rate of 44.11%. Using the Gym-malware generator in combination with itself improved the evasion rate to 58.35%.

摘要: 机器学习已被证明是自动恶意软件检测的有用工具，但机器学习模型也被证明容易受到对手攻击。本文解决了生成恶意软件示例的问题，特别是恶意Windows可移植可执行文件。我们总结和比较了专注于恶意软件检测的对抗性机器学习的工作。我们使用基于梯度、基于进化算法和基于强化的方法来生成对抗性样本，然后将生成的样本与选定的反病毒产品进行测试。我们从准确性和实用性两个方面对所选方法进行了比较。结果表明，对以前检测到的恶意软件应用优化修改可能会导致将文件错误分类为良性文件。还已知，生成的恶意软件样本可以成功地用于除了用于生成恶意软件样本的检测模型之外的检测模型，并且使用生成器的组合可以创建逃避检测的新样本。实验表明，使用强化学习方法的GYM恶意软件生成器具有最大的实用潜力。该生成器的平均样本生成时间为5.73秒，最高平均逃避率为44.11%。将GYM恶意软件生成器与GYM恶意软件生成器结合使用，可以将逃避率提高到58.35%。



## **37. Black-box Adversarial Attacks against Dense Retrieval Models: A Multi-view Contrastive Learning Method**

针对稠密检索模型的黑盒对抗性攻击：一种多视点对比学习方法 cs.IR

Accept by CIKM2023, 10 pages

**SubmitDate**: 2023-08-19    [abs](http://arxiv.org/abs/2308.09861v1) [paper-pdf](http://arxiv.org/pdf/2308.09861v1)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Wei Chen, Yixing Fan, Xueqi Cheng

**Abstract**: Neural ranking models (NRMs) and dense retrieval (DR) models have given rise to substantial improvements in overall retrieval performance. In addition to their effectiveness, and motivated by the proven lack of robustness of deep learning-based approaches in other areas, there is growing interest in the robustness of deep learning-based approaches to the core retrieval problem. Adversarial attack methods that have so far been developed mainly focus on attacking NRMs, with very little attention being paid to the robustness of DR models. In this paper, we introduce the adversarial retrieval attack (AREA) task. The AREA task is meant to trick DR models into retrieving a target document that is outside the initial set of candidate documents retrieved by the DR model in response to a query. We consider the decision-based black-box adversarial setting, which is realistic in real-world search engines. To address the AREA task, we first employ existing adversarial attack methods designed for NRMs. We find that the promising results that have previously been reported on attacking NRMs, do not generalize to DR models: these methods underperform a simple term spamming method. We attribute the observed lack of generalizability to the interaction-focused architecture of NRMs, which emphasizes fine-grained relevance matching. DR models follow a different representation-focused architecture that prioritizes coarse-grained representations. We propose to formalize attacks on DR models as a contrastive learning problem in a multi-view representation space. The core idea is to encourage the consistency between each view representation of the target document and its corresponding viewer via view-wise supervision signals. Experimental results demonstrate that the proposed method can significantly outperform existing attack strategies in misleading the DR model with small indiscernible text perturbations.

摘要: 神经排序模型(NRM)和密集检索(DR)模型在整体检索性能上有了很大的提高。除了它们的有效性，而且由于基于深度学习的方法在其他领域被证明缺乏稳健性，人们对基于深度学习的方法解决核心检索问题的稳健性越来越感兴趣。到目前为止，已开发的对抗性攻击方法主要集中在攻击NRM上，对DR模型的稳健性关注很少。本文介绍了对抗性检索攻击(区域)任务。区域任务旨在诱使DR模型检索在DR模型响应于查询而检索的初始候选文档集合之外的目标文档。我们考虑了基于决策的黑盒对抗性设置，这在现实世界的搜索引擎中是现实的。为了解决区域任务，我们首先使用为NRM设计的现有对抗性攻击方法。我们发现，以前报道的关于攻击NRM的有希望的结果并不能推广到DR模型：这些方法不如简单的术语垃圾邮件方法。我们将观察到的缺乏泛化能力归因于NRMS以交互为中心的体系结构，该体系结构强调细粒度的关联匹配。灾难恢复模型遵循不同的以表示为中心的体系结构，该体系结构优先考虑粗粒度的表示。我们提出将对DR模型的攻击形式化为多视点表示空间中的对比学习问题。其核心思想是通过基于视图的监督信号来鼓励目标文档的每个视图表示与其对应的查看器之间的一致性。实验结果表明，在文本扰动较小的情况下，该方法在误导DR模型方面明显优于已有的攻击策略。



## **38. RAIN: RegulArization on Input and Network for Black-Box Domain Adaptation**

RAIN：黑箱域自适应的输入和网络规则化 cs.CV

Accepted by IJCAI 2023

**SubmitDate**: 2023-08-19    [abs](http://arxiv.org/abs/2208.10531v4) [paper-pdf](http://arxiv.org/pdf/2208.10531v4)

**Authors**: Qucheng Peng, Zhengming Ding, Lingjuan Lyu, Lichao Sun, Chen Chen

**Abstract**: Source-Free domain adaptation transits the source-trained model towards target domain without exposing the source data, trying to dispel these concerns about data privacy and security. However, this paradigm is still at risk of data leakage due to adversarial attacks on the source model. Hence, the Black-Box setting only allows to use the outputs of source model, but still suffers from overfitting on the source domain more severely due to source model's unseen weights. In this paper, we propose a novel approach named RAIN (RegulArization on Input and Network) for Black-Box domain adaptation from both input-level and network-level regularization. For the input-level, we design a new data augmentation technique as Phase MixUp, which highlights task-relevant objects in the interpolations, thus enhancing input-level regularization and class consistency for target models. For network-level, we develop a Subnetwork Distillation mechanism to transfer knowledge from the target subnetwork to the full target network via knowledge distillation, which thus alleviates overfitting on the source domain by learning diverse target representations. Extensive experiments show that our method achieves state-of-the-art performance on several cross-domain benchmarks under both single- and multi-source black-box domain adaptation.

摘要: 无源域自适应在不暴露源数据的情况下将源训练模型过渡到目标域，试图消除这些对数据隐私和安全的担忧。然而，由于对源模型的对抗性攻击，该范例仍然面临数据泄露的风险。因此，黑盒设置只允许使用源模型的输出，但由于源模型的不可见权重，仍然受到源域上的过度拟合的更严重影响。本文从输入级正则化和网络级正则化两个方面提出了一种新的黑箱域自适应方法RAIN。对于输入层，我们设计了一种新的数据增强技术--阶段混合，它在内插中突出与任务相关的对象，从而增强了输入层的正则性和目标模型的类一致性。对于网络级，我们提出了一种子网络精馏机制，通过知识精馏将知识从目标子网络传递到整个目标网络，从而通过学习不同的目标表示来缓解源域的过度匹配。大量的实验表明，在单源和多源黑盒领域自适应的情况下，我们的方法在多个跨域基准测试上都达到了最好的性能。



## **39. Backdoor Mitigation by Correcting the Distribution of Neural Activations**

纠正神经激活分布的后门缓解 cs.LG

**SubmitDate**: 2023-08-18    [abs](http://arxiv.org/abs/2308.09850v1) [paper-pdf](http://arxiv.org/pdf/2308.09850v1)

**Authors**: Xi Li, Zhen Xiang, David J. Miller, George Kesidis

**Abstract**: Backdoor (Trojan) attacks are an important type of adversarial exploit against deep neural networks (DNNs), wherein a test instance is (mis)classified to the attacker's target class whenever the attacker's backdoor trigger is present. In this paper, we reveal and analyze an important property of backdoor attacks: a successful attack causes an alteration in the distribution of internal layer activations for backdoor-trigger instances, compared to that for clean instances. Even more importantly, we find that instances with the backdoor trigger will be correctly classified to their original source classes if this distribution alteration is corrected. Based on our observations, we propose an efficient and effective method that achieves post-training backdoor mitigation by correcting the distribution alteration using reverse-engineered triggers. Notably, our method does not change any trainable parameters of the DNN, but achieves generally better mitigation performance than existing methods that do require intensive DNN parameter tuning. It also efficiently detects test instances with the trigger, which may help to catch adversarial entities in the act of exploiting the backdoor.

摘要: 后门(特洛伊木马)攻击是针对深度神经网络(DNN)的一种重要的对抗性攻击，只要攻击者的后门触发器存在，测试实例就会(错误地)分类到攻击者的目标类。在本文中，我们揭示和分析了后门攻击的一个重要性质：与干净实例相比，成功的攻击会导致后门触发器实例的内层激活分布发生变化。更重要的是，我们发现，如果这种分布更改得到纠正，带有后门触发器的实例将被正确地分类为它们的原始源类。基于我们的观察，我们提出了一种高效有效的方法，通过使用反向工程触发器纠正分布变化来实现训练后后门缓解。值得注意的是，我们的方法不改变DNN的任何可训练参数，但总体上获得了比现有方法更好的缓解性能，这些方法确实需要密集的DNN参数调整。它还使用触发器高效地检测测试实例，这可能有助于在利用后门的行为中捕获敌对实体。



## **40. Hard No-Box Adversarial Attack on Skeleton-Based Human Action Recognition with Skeleton-Motion-Informed Gradient**

基于骨架运动信息梯度的基于骨架的人体动作识别的硬非盒对抗攻击 cs.CV

Camera-ready version for ICCV 2023

**SubmitDate**: 2023-08-18    [abs](http://arxiv.org/abs/2308.05681v2) [paper-pdf](http://arxiv.org/pdf/2308.05681v2)

**Authors**: Zhengzhi Lu, He Wang, Ziyi Chang, Guoan Yang, Hubert P. H. Shum

**Abstract**: Recently, methods for skeleton-based human activity recognition have been shown to be vulnerable to adversarial attacks. However, these attack methods require either the full knowledge of the victim (i.e. white-box attacks), access to training data (i.e. transfer-based attacks) or frequent model queries (i.e. black-box attacks). All their requirements are highly restrictive, raising the question of how detrimental the vulnerability is. In this paper, we show that the vulnerability indeed exists. To this end, we consider a new attack task: the attacker has no access to the victim model or the training data or labels, where we coin the term hard no-box attack. Specifically, we first learn a motion manifold where we define an adversarial loss to compute a new gradient for the attack, named skeleton-motion-informed (SMI) gradient. Our gradient contains information of the motion dynamics, which is different from existing gradient-based attack methods that compute the loss gradient assuming each dimension in the data is independent. The SMI gradient can augment many gradient-based attack methods, leading to a new family of no-box attack methods. Extensive evaluation and comparison show that our method imposes a real threat to existing classifiers. They also show that the SMI gradient improves the transferability and imperceptibility of adversarial samples in both no-box and transfer-based black-box settings.

摘要: 最近，基于骨架的人类活动识别方法被证明容易受到对手的攻击。然而，这些攻击方法要么需要受害者的全部知识(即白盒攻击)，要么需要访问训练数据(即基于传输的攻击)，或者需要频繁的模型查询(即黑盒攻击)。他们的所有要求都是高度限制性的，这引发了漏洞的危害性有多大的问题。在本文中，我们证明了该漏洞确实存在。为此，我们考虑了一个新的攻击任务：攻击者无权访问受害者模型、训练数据或标签，其中我们创造了术语硬无盒攻击。具体地说，我们首先学习一个运动流形，其中我们定义了一个对手损失来计算攻击的一个新的梯度，称为骨架-运动信息(SMI)梯度。我们的梯度包含了运动动力学的信息，这不同于现有的基于梯度的攻击方法，该方法假设数据中的每个维度都是独立的，来计算损失梯度。SMI梯度可以扩展许多基于梯度的攻击方法，从而产生一类新的非盒子攻击方法。广泛的评估和比较表明，我们的方法对现有的分类器构成了真正的威胁。它们还表明，SMI梯度在无盒和基于传输的黑盒设置下都改善了对抗性样本的可转移性和不可见性。



## **41. MONA: An Efficient and Scalable Strategy for Targeted k-Nodes Collapse**

MONA：一种高效可扩展的目标k节点崩溃策略 cs.SI

5 pages, 6 figures, 1 table, 5 algorithms

**SubmitDate**: 2023-08-18    [abs](http://arxiv.org/abs/2308.09601v1) [paper-pdf](http://arxiv.org/pdf/2308.09601v1)

**Authors**: Yuqian Lv, Bo Zhou, Jinhuan Wang, Shanqing Yu, Qi Xuan

**Abstract**: The concept of k-core plays an important role in measuring the cohesiveness and engagement of a network. And recent studies have shown the vulnerability of k-core under adversarial attacks. However, there are few researchers concentrating on the vulnerability of individual nodes within k-core. Therefore, in this paper, we attempt to study Targeted k-Nodes Collapse Problem (TNsCP), which focuses on removing a minimal size set of edges to make multiple target k-nodes collapse. For this purpose, we first propose a novel algorithm named MOD for candidate reduction. Then we introduce an efficient strategy named MONA, based on MOD, to address TNsCP. Extensive experiments validate the effectiveness and scalability of MONA compared to several baselines. An open-source implementation is available at https://github.com/Yocenly/MONA.

摘要: K-core的概念在衡量网络的凝聚力和参与度方面发挥着重要作用。最近的研究表明，k-core在对抗性攻击下的脆弱性。然而，很少有研究人员专注于k-core中单个节点的脆弱性。因此，在本文中，我们试图研究目标k节点崩溃问题(TNsCP)，该问题的重点是去除最小尺寸的边集以使多个目标k节点崩溃。为此，我们首先提出了一种新的候选约简算法MOD。然后，我们在MOD的基础上提出了一种有效的解决TNsCP的策略--MONA。大量实验验证了MONA的有效性和可扩展性，并与几条基线进行了比较。Https://github.com/Yocenly/MONA.上提供了一个开放源码实现



## **42. Compensating Removed Frequency Components: Thwarting Voice Spectrum Reduction Attacks**

补偿去除的频率分量：挫败语音频谱降低攻击 cs.CR

Accepted by 2024 Network and Distributed System Security Symposium  (NDSS'24)

**SubmitDate**: 2023-08-18    [abs](http://arxiv.org/abs/2308.09546v1) [paper-pdf](http://arxiv.org/pdf/2308.09546v1)

**Authors**: Shu Wang, Kun Sun, Qi Li

**Abstract**: Automatic speech recognition (ASR) provides diverse audio-to-text services for humans to communicate with machines. However, recent research reveals ASR systems are vulnerable to various malicious audio attacks. In particular, by removing the non-essential frequency components, a new spectrum reduction attack can generate adversarial audios that can be perceived by humans but cannot be correctly interpreted by ASR systems. It raises a new challenge for content moderation solutions to detect harmful content in audio and video available on social media platforms. In this paper, we propose an acoustic compensation system named ACE to counter the spectrum reduction attacks over ASR systems. Our system design is based on two observations, namely, frequency component dependencies and perturbation sensitivity. First, since the Discrete Fourier Transform computation inevitably introduces spectral leakage and aliasing effects to the audio frequency spectrum, the frequency components with similar frequencies will have a high correlation. Thus, considering the intrinsic dependencies between neighboring frequency components, it is possible to recover more of the original audio by compensating for the removed components based on the remaining ones. Second, since the removed components in the spectrum reduction attacks can be regarded as an inverse of adversarial noise, the attack success rate will decrease when the adversarial audio is replayed in an over-the-air scenario. Hence, we can model the acoustic propagation process to add over-the-air perturbations into the attacked audio. We implement a prototype of ACE and the experiments show ACE can effectively reduce up to 87.9% of ASR inference errors caused by spectrum reduction attacks. Also, by analyzing residual errors, we summarize six general types of ASR inference errors and investigate the error causes and potential mitigation solutions.

摘要: 自动语音识别(ASR)为人类与机器交流提供了多样化的音频到文本服务。然而，最近的研究表明，ASR系统容易受到各种恶意音频攻击。特别是，通过删除不必要的频率分量，新的频谱减少攻击可以生成人类可以感知但无法被ASR系统正确解释的敌意音频。这对内容审核解决方案提出了新的挑战，即检测社交媒体平台上可用音频和视频中的有害内容。本文针对ASR系统中的频谱缩减攻击，提出了一种声学补偿系统ACE。我们的系统设计基于两个观察，即频率分量相关性和扰动敏感度。首先，由于离散傅里叶变换的计算不可避免地会在音频频谱中引入频谱泄漏和混叠效应，因此具有相似频率的频率分量将具有较高的相关性。因此，考虑到相邻频率分量之间的内在相关性，可以通过基于剩余分量补偿去除的分量来恢复更多的原始音频。其次，由于频谱缩减攻击中被移除的分量可以被视为对抗性噪声的逆，因此当对抗性音频在空中场景中重放时，攻击成功率会降低。因此，我们可以对声传播过程进行建模，以向被攻击的音频中添加空中扰动。我们实现了一个ACE的原型，实验表明，ACE可以有效地减少高达87.9%的ASR推理错误。通过对剩余误差的分析，总结了ASR推理误差的六种常见类型，并探讨了误差产生的原因和可能的缓解方法。



## **43. Robust Evaluation of Diffusion-Based Adversarial Purification**

基于扩散的对抗净化算法的稳健性评价 cs.CV

Accepted by ICCV 2023, Oral presentation

**SubmitDate**: 2023-08-18    [abs](http://arxiv.org/abs/2303.09051v2) [paper-pdf](http://arxiv.org/pdf/2303.09051v2)

**Authors**: Minjong Lee, Dongwoo Kim

**Abstract**: We question the current evaluation practice on diffusion-based purification methods. Diffusion-based purification methods aim to remove adversarial effects from an input data point at test time. The approach gains increasing attention as an alternative to adversarial training due to the disentangling between training and testing. Well-known white-box attacks are often employed to measure the robustness of the purification. However, it is unknown whether these attacks are the most effective for the diffusion-based purification since the attacks are often tailored for adversarial training. We analyze the current practices and provide a new guideline for measuring the robustness of purification methods against adversarial attacks. Based on our analysis, we further propose a new purification strategy improving robustness compared to the current diffusion-based purification methods.

摘要: 我们对目前基于扩散的纯化方法的评估实践提出了质疑。基于扩散的净化方法旨在消除测试时输入数据点的对抗性影响。由于训练和测试之间的分离，这种方法作为对抗性训练的替代方法受到越来越多的关注。通常使用众所周知的白盒攻击来衡量净化的健壮性。然而，目前尚不清楚这些攻击对于基于扩散的净化是否最有效，因为这些攻击通常是为对抗性训练量身定做的。我们分析了目前的实践，并为衡量净化方法对对手攻击的健壮性提供了新的指导方针。在分析的基础上，进一步提出了一种新的纯化策略，与现有的基于扩散的纯化方法相比，提高了算法的稳健性。



## **44. REAP: A Large-Scale Realistic Adversarial Patch Benchmark**

REAP：一个大规模的现实对抗性补丁基准 cs.CV

ICCV 2023. Code and benchmark can be found at  https://github.com/wagner-group/reap-benchmark

**SubmitDate**: 2023-08-18    [abs](http://arxiv.org/abs/2212.05680v2) [paper-pdf](http://arxiv.org/pdf/2212.05680v2)

**Authors**: Nabeel Hingun, Chawin Sitawarin, Jerry Li, David Wagner

**Abstract**: Machine learning models are known to be susceptible to adversarial perturbation. One famous attack is the adversarial patch, a sticker with a particularly crafted pattern that makes the model incorrectly predict the object it is placed on. This attack presents a critical threat to cyber-physical systems that rely on cameras such as autonomous cars. Despite the significance of the problem, conducting research in this setting has been difficult; evaluating attacks and defenses in the real world is exceptionally costly while synthetic data are unrealistic. In this work, we propose the REAP (REalistic Adversarial Patch) benchmark, a digital benchmark that allows the user to evaluate patch attacks on real images, and under real-world conditions. Built on top of the Mapillary Vistas dataset, our benchmark contains over 14,000 traffic signs. Each sign is augmented with a pair of geometric and lighting transformations, which can be used to apply a digitally generated patch realistically onto the sign. Using our benchmark, we perform the first large-scale assessments of adversarial patch attacks under realistic conditions. Our experiments suggest that adversarial patch attacks may present a smaller threat than previously believed and that the success rate of an attack on simpler digital simulations is not predictive of its actual effectiveness in practice. We release our benchmark publicly at https://github.com/wagner-group/reap-benchmark.

摘要: 众所周知，机器学习模型容易受到对抗性扰动的影响。一种著名的攻击是对抗性补丁，这是一种带有特别精心制作的图案的贴纸，使模型无法正确预测它所放置的对象。这种攻击对自动驾驶汽车等依赖摄像头的网络物理系统构成了严重威胁。尽管这个问题很重要，但在这种情况下进行研究一直很困难；评估现实世界中的攻击和防御成本异常高昂，而合成数据是不切实际的。在这项工作中，我们提出了REAP(现实对抗补丁)基准，这是一个数字基准，允许用户在真实世界的条件下评估对真实图像的补丁攻击。我们的基准建立在Mapillary Vistas数据集的基础上，包含超过14,000个交通标志。每个标志都增加了一对几何和照明变换，可以用来将数字生成的补丁逼真地应用到标志上。使用我们的基准，我们在现实条件下执行了第一次大规模的对抗性补丁攻击评估。我们的实验表明，对抗性补丁攻击可能比之前认为的威胁更小，并且在更简单的数字模拟上的攻击成功率并不能预测其在实践中的实际有效性。我们在https://github.com/wagner-group/reap-benchmark.上公开发布我们的基准



## **45. Attacking logo-based phishing website detectors with adversarial perturbations**

利用对抗性扰动攻击基于徽标的钓鱼网站检测器 cs.CR

To appear in ESORICS 2023

**SubmitDate**: 2023-08-18    [abs](http://arxiv.org/abs/2308.09392v1) [paper-pdf](http://arxiv.org/pdf/2308.09392v1)

**Authors**: Jehyun Lee, Zhe Xin, Melanie Ng Pei See, Kanav Sabharwal, Giovanni Apruzzese, Dinil Mon Divakaran

**Abstract**: Recent times have witnessed the rise of anti-phishing schemes powered by deep learning (DL). In particular, logo-based phishing detectors rely on DL models from Computer Vision to identify logos of well-known brands on webpages, to detect malicious webpages that imitate a given brand. For instance, Siamese networks have demonstrated notable performance for these tasks, enabling the corresponding anti-phishing solutions to detect even "zero-day" phishing webpages. In this work, we take the next step of studying the robustness of logo-based phishing detectors against adversarial ML attacks. We propose a novel attack exploiting generative adversarial perturbations to craft "adversarial logos" that evade phishing detectors. We evaluate our attacks through: (i) experiments on datasets containing real logos, to evaluate the robustness of state-of-the-art phishing detectors; and (ii) user studies to gauge whether our adversarial logos can deceive human eyes. The results show that our proposed attack is capable of crafting perturbed logos subtle enough to evade various DL models-achieving an evasion rate of up to 95%. Moreover, users are not able to spot significant differences between generated adversarial logos and original ones.

摘要: 最近见证了由深度学习(DL)提供动力的反网络钓鱼计划的兴起。特别是，基于徽标的钓鱼检测器依赖于计算机视觉的DL模型来识别网页上知名品牌的徽标，以检测模仿给定品牌的恶意网页。例如，暹罗网络在这些任务中表现出了显著的性能，使相应的反网络钓鱼解决方案能够检测到甚至是“零日”网络钓鱼网页。在这项工作中，我们下一步研究了基于标识的钓鱼检测器对恶意ML攻击的稳健性。我们提出了一种新的攻击，利用生成性敌意扰动来创建逃避网络钓鱼检测器的“对抗性标识”。我们通过以下方式评估我们的攻击：(I)在包含真实标识的数据集上进行实验，以评估最先进的网络钓鱼检测器的健壮性；以及(Ii)用户研究，以衡量我们的对手标识是否可以欺骗人眼。结果表明，我们提出的攻击能够巧妙地制作足够微妙的扰动徽标来规避各种DL模型-实现高达95%的逃避率。此外，用户无法发现生成的敌意徽标与原始徽标之间的显著差异。



## **46. Gradient-Based Word Substitution for Obstinate Adversarial Examples Generation in Language Models**

语言模型中基于梯度的词语替换生成顽固对抗性实例 cs.CL

19 pages

**SubmitDate**: 2023-08-18    [abs](http://arxiv.org/abs/2307.12507v2) [paper-pdf](http://arxiv.org/pdf/2307.12507v2)

**Authors**: Yimu Wang, Peng Shi, Hongyang Zhang

**Abstract**: In this paper, we study the problem of generating obstinate (over-stability) adversarial examples by word substitution in NLP, where input text is meaningfully changed but the model's prediction does not, even though it should. Previous word substitution approaches have predominantly focused on manually designed antonym-based strategies for generating obstinate adversarial examples, which hinders its application as these strategies can only find a subset of obstinate adversarial examples and require human efforts. To address this issue, in this paper, we introduce a novel word substitution method named GradObstinate, a gradient-based approach that automatically generates obstinate adversarial examples without any constraints on the search space or the need for manual design principles. To empirically evaluate the efficacy of GradObstinate, we conduct comprehensive experiments on five representative models (Electra, ALBERT, Roberta, DistillBERT, and CLIP) finetuned on four NLP benchmarks (SST-2, MRPC, SNLI, and SQuAD) and a language-grounding benchmark (MSCOCO). Extensive experiments show that our proposed GradObstinate generates more powerful obstinate adversarial examples, exhibiting a higher attack success rate compared to antonym-based methods. Furthermore, to show the transferability of obstinate word substitutions found by GradObstinate, we replace the words in four representative NLP benchmarks with their obstinate substitutions. Notably, obstinate substitutions exhibit a high success rate when transferred to other models in black-box settings, including even GPT-3 and ChatGPT. Examples of obstinate adversarial examples found by GradObstinate are available at https://huggingface.co/spaces/anonauthors/SecretLanguage.

摘要: 在本文中，我们研究了在自然语言处理中通过单词替换生成顽固(过稳定)对抗性例子的问题，其中输入文本被有意义地改变，但模型的预测没有，尽管它应该被改变。以往的单词替换方法主要集中在人工设计的基于反义词的策略来生成顽固的对抗性实例，这阻碍了其应用，因为这些策略只能找到顽固的反义词实例的子集，并且需要人工努力。为了解决这一问题，本文提出了一种新的词替换方法GradObstate，该方法基于梯度自动生成顽固的对抗性实例，不需要任何搜索空间的限制，也不需要手动设计原则。为了经验性地评估GradObstate的有效性，我们在四个NLP基准(SST-2、MRPC、SNLI和TEAND)和一个语言基础基准(MSCOCO)上对五个有代表性的模型(ELECTRA、Albert、Roberta、DistillBERT和CLIP)进行了全面的测试。大量的实验表明，与基于反义词的方法相比，我们提出的GradObstate方法生成了更强大的顽固对抗性实例，表现出更高的攻击成功率。此外，为了显示GradObstate发现的顽固单词替换的可转移性，我们将四个有代表性的NLP基准中的单词替换为它们的顽固替换。值得注意的是，当在黑盒设置中转移到其他型号时，顽固的替代显示出高成功率，甚至包括GPT-3和ChatGPT。GradObstate发现的顽固敌意例子可以在https://huggingface.co/spaces/anonauthors/SecretLanguage.上找到



## **47. Among Us: Adversarially Robust Collaborative Perception by Consensus**

在我们中间：基于共识的相反的强健协作感知 cs.RO

Accepted by ICCV 2023

**SubmitDate**: 2023-08-18    [abs](http://arxiv.org/abs/2303.09495v3) [paper-pdf](http://arxiv.org/pdf/2303.09495v3)

**Authors**: Yiming Li, Qi Fang, Jiamu Bai, Siheng Chen, Felix Juefei-Xu, Chen Feng

**Abstract**: Multiple robots could perceive a scene (e.g., detect objects) collaboratively better than individuals, although easily suffer from adversarial attacks when using deep learning. This could be addressed by the adversarial defense, but its training requires the often-unknown attacking mechanism. Differently, we propose ROBOSAC, a novel sampling-based defense strategy generalizable to unseen attackers. Our key idea is that collaborative perception should lead to consensus rather than dissensus in results compared to individual perception. This leads to our hypothesize-and-verify framework: perception results with and without collaboration from a random subset of teammates are compared until reaching a consensus. In such a framework, more teammates in the sampled subset often entail better perception performance but require longer sampling time to reject potential attackers. Thus, we derive how many sampling trials are needed to ensure the desired size of an attacker-free subset, or equivalently, the maximum size of such a subset that we can successfully sample within a given number of trials. We validate our method on the task of collaborative 3D object detection in autonomous driving scenarios.

摘要: 多个机器人可以比个人更好地协作感知场景(例如，检测对象)，尽管在使用深度学习时很容易受到对抗性攻击。这可以通过对抗性防守来解决，但它的训练需要往往未知的攻击机制。不同的是，我们提出了ROBOSAC，一种新的基于采样的防御策略，可以推广到看不见的攻击者。我们的关键思想是，与个人感知相比，合作感知应该在结果中导致共识，而不是分歧。这导致了我们的假设和验证框架：对随机的队友子集进行协作和不协作的感知结果进行比较，直到达成共识。在这样的框架中，采样子集中更多的队友通常会带来更好的感知性能，但需要更长的采样时间来拒绝潜在的攻击者。因此，我们推导出需要多少次抽样试验才能确保没有攻击者的子集的期望大小，或者等价地，在给定的试验次数内可以成功抽样的子集的最大大小。我们在自主驾驶场景下的协同3D目标检测任务中验证了我们的方法。



## **48. Targeted Adversarial Attacks on Wind Power Forecasts**

对风电预测的有针对性的对抗性攻击 cs.LG

21 pages, including appendix, 12 figures

**SubmitDate**: 2023-08-17    [abs](http://arxiv.org/abs/2303.16633v2) [paper-pdf](http://arxiv.org/pdf/2303.16633v2)

**Authors**: René Heinrich, Christoph Scholz, Stephan Vogt, Malte Lehna

**Abstract**: In recent years, researchers proposed a variety of deep learning models for wind power forecasting. These models predict the wind power generation of wind farms or entire regions more accurately than traditional machine learning algorithms or physical models. However, latest research has shown that deep learning models can often be manipulated by adversarial attacks. Since wind power forecasts are essential for the stability of modern power systems, it is important to protect them from this threat. In this work, we investigate the vulnerability of two different forecasting models to targeted, semi-targeted, and untargeted adversarial attacks. We consider a Long Short-Term Memory (LSTM) network for predicting the power generation of individual wind farms and a Convolutional Neural Network (CNN) for forecasting the wind power generation throughout Germany. Moreover, we propose the Total Adversarial Robustness Score (TARS), an evaluation metric for quantifying the robustness of regression models to targeted and semi-targeted adversarial attacks. It assesses the impact of attacks on the model's performance, as well as the extent to which the attacker's goal was achieved, by assigning a score between 0 (very vulnerable) and 1 (very robust). In our experiments, the LSTM forecasting model was fairly robust and achieved a TARS value of over 0.78 for all adversarial attacks investigated. The CNN forecasting model only achieved TARS values below 0.10 when trained ordinarily, and was thus very vulnerable. Yet, its robustness could be significantly improved by adversarial training, which always resulted in a TARS above 0.46.

摘要: 近年来，研究人员提出了各种用于风电功率预测的深度学习模型。这些模型比传统的机器学习算法或物理模型更准确地预测风电场或整个地区的风力发电量。然而，最新的研究表明，深度学习模型经常会被对抗性攻击所操纵。由于风力发电预测对现代电力系统的稳定性至关重要，因此保护它们免受这种威胁是很重要的。在这项工作中，我们调查了两种不同的预测模型对定向、半定向和非定向对手攻击的脆弱性。我们考虑了用于预测单个风电场发电量的长短期记忆(LSTM)网络和用于预测整个德国风电场发电量的卷积神经网络(CNN)。此外，我们还提出了总对抗稳健性分数(TARS)，这是一个量化回归模型对定向和半定向对抗攻击的稳健性的评估指标。它通过在0(非常脆弱)和1(非常健壮)之间分配分数来评估攻击对模型性能的影响，以及攻击者目标的实现程度。在我们的实验中，LSTM预测模型是相当健壮的，对于所有被调查的对抗性攻击，TARS值都超过了0.78。CNN预测模型在正常训练时只能达到0.10以下的TARS值，因此非常容易受到攻击。然而，它的稳健性可以通过对抗性训练显著提高，这总是导致TARS高于0.46。



## **49. Recent Latest Message Driven GHOST: Balancing Dynamic Availability With Asynchrony Resilience**

最近最新的消息驱动幽灵：平衡动态可用性和异步弹性 cs.DC

**SubmitDate**: 2023-08-17    [abs](http://arxiv.org/abs/2302.11326v3) [paper-pdf](http://arxiv.org/pdf/2302.11326v3)

**Authors**: Francesco D'Amato, Luca Zanolini

**Abstract**: Dynamic participation has recently become a crucial requirement for devising permissionless consensus protocols. This notion, originally formalized by Pass and Shi (ASIACRYPT 2017) through their "sleepy model", captures the essence of a system's ability to handle participants joining or leaving during a protocol execution. A dynamically available consensus protocol preserves safety and liveness while allowing dynamic participation. Blockchain protocols, such as Bitcoin's consensus protocol, have implicitly adopted this concept. In the context of Ethereum's consensus protocol, Gasper, Neu, Tas, and Tse (S&P 2021) presented an attack against LMD-GHOST -- the component of Gasper designed to ensure dynamic availability. Consequently, LMD-GHOST results unable to fulfill its intended function of providing dynamic availability for the protocol. Despite attempts to mitigate this issue, the modified protocol still does not achieve dynamic availability, highlighting the need for more secure dynamically available protocols. In this work, we present RLMD-GHOST, a synchronous consensus protocol that not only ensures dynamic availability but also maintains safety during bounded periods of asynchrony. This protocol is particularly appealing for practical systems where strict synchrony assumptions may not always hold, contrary to general assumptions in standard synchronous protocols. Additionally, we present the "generalized sleepy model", within which our results are proven. Building upon the original sleepy model proposed by Pass and Shi, our model extends it with more generalized and stronger constraints on the corruption and sleepiness power of the adversary. This approach allows us to explore a wide range of dynamic participation regimes, spanning from complete dynamic participation to no dynamic participation, i.e., with every participant online.

摘要: 最近，动态参与已成为设计未经许可的协商一致协议的关键要求。这个概念最初是由Pass和Shih(ASIACRYPT 2017)通过他们的“休眠模型”形式化的，它捕捉到了系统在协议执行期间处理参与者加入或离开的能力的本质。动态可用的协商一致协议在允许动态参与的同时保持安全性和活跃性。区块链协议，如比特币的共识协议，隐含地采用了这一概念。在以太的共识协议的背景下，Gasper、Neu、Tas和Tse(S&P2021)提出了对LMD-Ghost的攻击-Gasper的组件旨在确保动态可用性。因此，LMD-GHOST导致不能实现其为协议提供动态可用性的预期功能。尽管试图缓解这一问题，但修改后的协议仍然无法实现动态可用性，这突显了需要更安全的动态可用的协议。在这项工作中，我们提出了一种同步一致性协议RLMD-GHOST，它不仅保证了动态可用性，而且在有界的异步期内保持了安全性。与标准同步协议中的一般假设相反，该协议对于严格的同步假设可能不总是成立的实际系统特别有吸引力。此外，我们还提出了“广义嗜睡模型”，在该模型中，我们的结果得到了证明。该模型在Pass和Shih提出的嗜睡模型的基础上，对其进行了扩展，对对手的腐败和嗜睡权力进行了更普遍和更强的约束。这一方法使我们能够探索广泛的动态参与机制，从完全动态参与到非动态参与，即每个参与者都在线。



## **50. A Survey on Malware Detection with Graph Representation Learning**

基于图表示学习的恶意软件检测综述 cs.CR

Preprint, submitted to ACM Computing Surveys on March 2023. For any  suggestions or improvements, please contact me directly by e-mail

**SubmitDate**: 2023-08-17    [abs](http://arxiv.org/abs/2303.16004v2) [paper-pdf](http://arxiv.org/pdf/2303.16004v2)

**Authors**: Tristan Bilot, Nour El Madhoun, Khaldoun Al Agha, Anis Zouaoui

**Abstract**: Malware detection has become a major concern due to the increasing number and complexity of malware. Traditional detection methods based on signatures and heuristics are used for malware detection, but unfortunately, they suffer from poor generalization to unknown attacks and can be easily circumvented using obfuscation techniques. In recent years, Machine Learning (ML) and notably Deep Learning (DL) achieved impressive results in malware detection by learning useful representations from data and have become a solution preferred over traditional methods. More recently, the application of such techniques on graph-structured data has achieved state-of-the-art performance in various domains and demonstrates promising results in learning more robust representations from malware. Yet, no literature review focusing on graph-based deep learning for malware detection exists. In this survey, we provide an in-depth literature review to summarize and unify existing works under the common approaches and architectures. We notably demonstrate that Graph Neural Networks (GNNs) reach competitive results in learning robust embeddings from malware represented as expressive graph structures, leading to an efficient detection by downstream classifiers. This paper also reviews adversarial attacks that are utilized to fool graph-based detection methods. Challenges and future research directions are discussed at the end of the paper.

摘要: 由于恶意软件的数量和复杂性不断增加，恶意软件检测已成为一个主要问题。传统的基于签名和启发式的检测方法被用于恶意软件检测，但遗憾的是，它们对未知攻击的泛化能力较差，可以通过混淆技术轻松地绕过。近年来，机器学习(ML)和深度学习(DL)通过从数据中学习有用的表示，在恶意软件检测方面取得了令人印象深刻的结果，并成为一种比传统方法更受欢迎的解决方案。最近，这种技术在图结构数据上的应用已经在各个领域取得了最先进的性能，并在从恶意软件中学习更健壮的表示方面展示了良好的结果。然而，目前还没有关于基于图的深度学习用于恶意软件检测的文献综述。在这次调查中，我们提供了深入的文献回顾，以总结和统一在共同的方法和架构下的现有工作。值得注意的是，图神经网络(GNN)在学习表示为可表达图结构的恶意软件的健壮嵌入方面取得了具有竞争力的结果，从而导致了下游分类器的有效检测。本文还回顾了用于欺骗基于图的检测方法的对抗性攻击。在文章的最后，讨论了挑战和未来的研究方向。



