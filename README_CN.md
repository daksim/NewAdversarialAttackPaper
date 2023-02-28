# Latest Adversarial Attack Papers
**update at 2023-02-28 19:31:06**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Understanding Adversarial Attacks on Observations in Deep Reinforcement Learning**

理解深度强化学习中对观测数据的敌意攻击 cs.LG

**SubmitDate**: 2023-02-27    [abs](http://arxiv.org/abs/2106.15860v3) [paper-pdf](http://arxiv.org/pdf/2106.15860v3)

**Authors**: You Qiaoben, Chengyang Ying, Xinning Zhou, Hang Su, Jun Zhu, Bo Zhang

**Abstract**: Deep reinforcement learning models are vulnerable to adversarial attacks that can decrease a victim's cumulative expected reward by manipulating the victim's observations. Despite the efficiency of previous optimization-based methods for generating adversarial noise in supervised learning, such methods might not be able to achieve the lowest cumulative reward since they do not explore the environmental dynamics in general. In this paper, we provide a framework to better understand the existing methods by reformulating the problem of adversarial attacks on reinforcement learning in the function space. Our reformulation generates an optimal adversary in the function space of the targeted attacks, repelling them via a generic two-stage framework. In the first stage, we train a deceptive policy by hacking the environment, and discover a set of trajectories routing to the lowest reward or the worst-case performance. Next, the adversary misleads the victim to imitate the deceptive policy by perturbing the observations. Compared to existing approaches, we theoretically show that our adversary is stronger under an appropriate noise level. Extensive experiments demonstrate our method's superiority in terms of efficiency and effectiveness, achieving the state-of-the-art performance in both Atari and MuJoCo environments.

摘要: 深度强化学习模型容易受到对抗性攻击，这些攻击可以通过操纵受害者的观察来降低受害者的累积预期回报。尽管以前的基于优化的方法在监督学习中产生对抗性噪声是有效的，但这些方法可能不能获得最低的累积奖励，因为它们通常不探索环境动态。在本文中，我们提供了一个框架，通过在函数空间中重新描述对抗性攻击对强化学习的问题来更好地理解现有的方法。我们的重构在目标攻击的函数空间中生成一个最优对手，通过一个通用的两阶段框架击退它们。在第一阶段，我们通过黑客攻击环境来训练欺骗性策略，并发现一组通往最低回报或最差表现的轨迹。接下来，对手通过扰乱观察来误导受害者模仿欺骗性的政策。与现有的方法相比，我们从理论上证明了在适当的噪声水平下，我们的对手更强。大量的实验证明了我们的方法在效率和有效性方面的优越性，在雅达利和MuJoCo环境中都实现了最先进的性能。



## **2. Implicit Poisoning Attacks in Two-Agent Reinforcement Learning: Adversarial Policies for Training-Time Attacks**

两智能体强化学习中的隐性中毒攻击：训练时间攻击的对抗性策略 cs.LG

**SubmitDate**: 2023-02-27    [abs](http://arxiv.org/abs/2302.13851v1) [paper-pdf](http://arxiv.org/pdf/2302.13851v1)

**Authors**: Mohammad Mohammadi, Jonathan Nöther, Debmalya Mandal, Adish Singla, Goran Radanovic

**Abstract**: In targeted poisoning attacks, an attacker manipulates an agent-environment interaction to force the agent into adopting a policy of interest, called target policy. Prior work has primarily focused on attacks that modify standard MDP primitives, such as rewards or transitions. In this paper, we study targeted poisoning attacks in a two-agent setting where an attacker implicitly poisons the effective environment of one of the agents by modifying the policy of its peer. We develop an optimization framework for designing optimal attacks, where the cost of the attack measures how much the solution deviates from the assumed default policy of the peer agent. We further study the computational properties of this optimization framework. Focusing on a tabular setting, we show that in contrast to poisoning attacks based on MDP primitives (transitions and (unbounded) rewards), which are always feasible, it is NP-hard to determine the feasibility of implicit poisoning attacks. We provide characterization results that establish sufficient conditions for the feasibility of the attack problem, as well as an upper and a lower bound on the optimal cost of the attack. We propose two algorithmic approaches for finding an optimal adversarial policy: a model-based approach with tabular policies and a model-free approach with parametric/neural policies. We showcase the efficacy of the proposed algorithms through experiments.

摘要: 在定向投毒攻击中，攻击者操纵代理与环境的交互，迫使代理采用感兴趣的策略，称为目标策略。以前的工作主要集中在修改标准MDP原语的攻击上，例如奖励或转换。在这篇文章中，我们研究了两个智能体环境中的定向中毒攻击，其中攻击者通过修改其中一个智能体的策略来隐含地毒化其中一个智能体的有效环境。我们开发了一个优化框架来设计最优攻击，其中攻击的代价衡量了解决方案与假设的对等代理的默认策略的偏差程度。我们进一步研究了该优化框架的计算性质。重点在表格设置下，我们证明了与基于MDP原语(转移和(无界)回报)的中毒攻击相比，隐式中毒攻击的可行性是NP-困难的。我们的刻画结果为攻击问题的可行性建立了充分条件，并给出了攻击的最优代价的上界和下界。我们提出了两种寻找最优对抗策略的算法方法：基于模型的列表策略方法和参数/神经策略的无模型方法。通过实验验证了所提算法的有效性。



## **3. Locality-Sensitive Hashing Does Not Guarantee Privacy! Attacks on Google's FLoC and the MinHash Hierarchy System**

位置敏感散列不能保证隐私！对Google的Floc和MinHash层次结构系统的攻击 cs.CR

14 pages, 9 figures submitted to PETS 2023

**SubmitDate**: 2023-02-27    [abs](http://arxiv.org/abs/2302.13635v1) [paper-pdf](http://arxiv.org/pdf/2302.13635v1)

**Authors**: Florian Turati, Carlos Cotrini, Karel Kubicek, David Basin

**Abstract**: Recently proposed systems aim at achieving privacy using locality-sensitive hashing. We show how these approaches fail by presenting attacks against two such systems: Google's FLoC proposal for privacy-preserving targeted advertising and the MinHash Hierarchy, a system for processing mobile users' traffic behavior in a privacy-preserving way. Our attacks refute the pre-image resistance, anonymity, and privacy guarantees claimed for these systems.   In the case of FLoC, we show how to deanonymize users using Sybil attacks and to reconstruct 10% or more of the browsing history for 30% of its users using Generative Adversarial Networks. We achieve this only analyzing the hashes used by FLoC. For MinHash, we precisely identify the movement of a subset of individuals and, on average, we can limit users' movement to just 10% of the possible geographic area, again using just the hashes. In addition, we refute their differential privacy claims.

摘要: 最近提出的系统旨在使用位置敏感散列来实现隐私。我们通过对两个这样的系统的攻击来展示这些方法是如何失败的：谷歌针对隐私保护定向广告的Floc提案，以及MinHash Hierarchy，一个以保护隐私的方式处理移动用户流量行为的系统。我们的攻击驳斥了这些系统声称的前映像抵抗、匿名性和隐私保障。在Floc的例子中，我们展示了如何使用Sybil攻击使用户去匿名化，并使用生成性对手网络为30%的用户重建10%或更多的浏览历史。我们只需要分析Floc使用的散列就可以做到这一点。对于MinHash，我们精确地识别一部分人的移动，平均而言，我们可以将用户的移动限制在可能地理区域的10%以内，同样只使用哈希。此外，我们驳斥了他们不同的隐私主张。



## **4. Online Black-Box Confidence Estimation of Deep Neural Networks**

深度神经网络的在线黑箱置信度估计 cs.CV

**SubmitDate**: 2023-02-27    [abs](http://arxiv.org/abs/2302.13578v1) [paper-pdf](http://arxiv.org/pdf/2302.13578v1)

**Authors**: Fabian Woitschek, Georg Schneider

**Abstract**: Autonomous driving (AD) and advanced driver assistance systems (ADAS) increasingly utilize deep neural networks (DNNs) for improved perception or planning. Nevertheless, DNNs are quite brittle when the data distribution during inference deviates from the data distribution during training. This represents a challenge when deploying in partly unknown environments like in the case of ADAS. At the same time, the standard confidence of DNNs remains high even if the classification reliability decreases. This is problematic since following motion control algorithms consider the apparently confident prediction as reliable even though it might be considerably wrong. To reduce this problem real-time capable confidence estimation is required that better aligns with the actual reliability of the DNN classification. Additionally, the need exists for black-box confidence estimation to enable the homogeneous inclusion of externally developed components to an entire system. In this work we explore this use case and introduce the neighborhood confidence (NHC) which estimates the confidence of an arbitrary DNN for classification. The metric can be used for black-box systems since only the top-1 class output is required and does not need access to the gradients, the training dataset or a hold-out validation dataset. Evaluation on different data distributions, including small in-domain distribution shifts, out-of-domain data or adversarial attacks, shows that the NHC performs better or on par with a comparable method for online white-box confidence estimation in low data regimes which is required for real-time capable AD/ADAS.

摘要: 自动驾驶(AD)和高级驾驶辅助系统(ADA)越来越多地利用深度神经网络(DNN)来改善感知或规划。然而，当推理过程中的数据分布偏离训练过程中的数据分布时，DNN是相当脆弱的。当在部分未知的环境中部署时，这是一个挑战，比如ADAS。同时，在分类可靠性下降的情况下，DNN的标准置信度仍然很高。这是有问题的，因为后续的运动控制算法认为明显有信心的预测是可靠的，即使它可能是相当错误的。为了减少这一问题，需要更好地符合DNN分类的实际可靠性的实时有能力的置信度估计。此外，还需要黑盒置信度估计，以便能够将外部开发的组件均匀地包含到整个系统中。在这项工作中，我们探索这一用例，并引入邻域置信度(NHC)来估计任意DNN用于分类的置信度。该度量可以用于黑盒系统，因为只需要前1类输出，而不需要访问梯度、训练数据集或坚持验证数据集。对不同的数据分布，包括域内分布漂移、域外数据和敌意攻击的评估表明，NHC在低数据区域的在线白盒置信度估计方法的性能比实时AD/ADAS所需的在线白盒置信度估计方法更好或相当。



## **5. Physical Adversarial Attacks on Deep Neural Networks for Traffic Sign Recognition: A Feasibility Study**

基于深度神经网络的物理对抗攻击用于交通标志识别的可行性研究 cs.CV

**SubmitDate**: 2023-02-27    [abs](http://arxiv.org/abs/2302.13570v1) [paper-pdf](http://arxiv.org/pdf/2302.13570v1)

**Authors**: Fabian Woitschek, Georg Schneider

**Abstract**: Deep Neural Networks (DNNs) are increasingly applied in the real world in safety critical applications like advanced driver assistance systems. An example for such use case is represented by traffic sign recognition systems. At the same time, it is known that current DNNs can be fooled by adversarial attacks, which raises safety concerns if those attacks can be applied under realistic conditions. In this work we apply different black-box attack methods to generate perturbations that are applied in the physical environment and can be used to fool systems under different environmental conditions. To the best of our knowledge we are the first to combine a general framework for physical attacks with different black-box attack methods and study the impact of the different methods on the success rate of the attack under the same setting. We show that reliable physical adversarial attacks can be performed with different methods and that it is also possible to reduce the perceptibility of the resulting perturbations. The findings highlight the need for viable defenses of a DNN even in the black-box case, but at the same time form the basis for securing a DNN with methods like adversarial training which utilizes adversarial attacks to augment the original training data.

摘要: 深度神经网络(DNN)越来越多地应用于现实世界中的安全关键应用，如先进的驾驶员辅助系统。交通标志识别系统代表了这种用例的一个例子。同时，众所周知，当前的DNN可能会被对抗性攻击所愚弄，如果这些攻击能够在现实条件下应用，这就引起了安全问题。在这项工作中，我们应用不同的黑盒攻击方法来产生应用于物理环境的扰动，并在不同的环境条件下使用这些扰动来愚弄系统。据我们所知，我们首次将物理攻击的一般框架与不同的黑盒攻击方法相结合，并在相同的设置下研究了不同方法对攻击成功率的影响。我们证明了可靠的物理对抗性攻击可以用不同的方法来执行，并且还可以降低由此产生的扰动的可感知性。这些发现强调了即使在黑盒情况下也需要对DNN进行可行的防御，但同时也构成了使用对抗性训练等方法确保DNN安全的基础，对抗性训练利用对抗性攻击来增强原始训练数据。



## **6. Aegis: Mitigating Targeted Bit-flip Attacks against Deep Neural Networks**

Aegis：减轻针对深度神经网络的定向比特翻转攻击 cs.CR

**SubmitDate**: 2023-02-27    [abs](http://arxiv.org/abs/2302.13520v1) [paper-pdf](http://arxiv.org/pdf/2302.13520v1)

**Authors**: Jialai Wang, Ziyuan Zhang, Meiqi Wang, Han Qiu, Tianwei Zhang, Qi Li, Zongpeng Li, Tao Wei, Chao Zhang

**Abstract**: Bit-flip attacks (BFAs) have attracted substantial attention recently, in which an adversary could tamper with a small number of model parameter bits to break the integrity of DNNs. To mitigate such threats, a batch of defense methods are proposed, focusing on the untargeted scenarios. Unfortunately, they either require extra trustworthy applications or make models more vulnerable to targeted BFAs. Countermeasures against targeted BFAs, stealthier and more purposeful by nature, are far from well established.   In this work, we propose Aegis, a novel defense method to mitigate targeted BFAs. The core observation is that existing targeted attacks focus on flipping critical bits in certain important layers. Thus, we design a dynamic-exit mechanism to attach extra internal classifiers (ICs) to hidden layers. This mechanism enables input samples to early-exit from different layers, which effectively upsets the adversary's attack plans. Moreover, the dynamic-exit mechanism randomly selects ICs for predictions during each inference to significantly increase the attack cost for the adaptive attacks where all defense mechanisms are transparent to the adversary. We further propose a robustness training strategy to adapt ICs to the attack scenarios by simulating BFAs during the IC training phase, to increase model robustness. Extensive evaluations over four well-known datasets and two popular DNN structures reveal that Aegis could effectively mitigate different state-of-the-art targeted attacks, reducing attack success rate by 5-10$\times$, significantly outperforming existing defense methods.

摘要: 比特翻转攻击(BFA)近年来引起了广泛的关注，在这种攻击中，攻击者可以篡改少量的模型参数比特来破坏DNN的完整性。为了缓解这种威胁，人们提出了一批针对非目标场景的防御方法。不幸的是，它们要么需要额外的值得信赖的应用程序，要么使模型更容易受到目标BFA的攻击。针对定向BFA的对策本质上更隐蔽、更有目的，但远未得到很好的确立。在这项工作中，我们提出了一种新的防御方法Aegis来缓解目标BFA。核心观察是，现有的有针对性的攻击集中在翻转某些重要层的关键比特。因此，我们设计了一种动态退出机制，将额外的内部分类器(IC)附加到隐层。这种机制使得输入样本能够从不同的层提前退出，从而有效地扰乱了对手的攻击计划。此外，动态退出机制在每次推理过程中随机选择IC进行预测，从而显著增加了所有防御机制对对手透明的自适应攻击的攻击成本。在此基础上，提出了一种健壮性训练策略，通过在ICS训练阶段模拟BFA来使ICS适应攻击场景，从而增强模型的健壮性。对四个著名的数据集和两个流行的DNN结构的广泛评估表明，Aegis可以有效地缓解各种最先进的针对性攻击，使攻击成功率降低5-10美元\x$，显著优于现有的防御方法。



## **7. CBA: Contextual Background Attack against Optical Aerial Detection in the Physical World**

CBA：物理世界中对光学空中探测的背景攻击 cs.CV

**SubmitDate**: 2023-02-27    [abs](http://arxiv.org/abs/2302.13519v1) [paper-pdf](http://arxiv.org/pdf/2302.13519v1)

**Authors**: Jiawei Lian, Xiaofei Wang, Yuru Su, Mingyang Ma, Shaohui Mei

**Abstract**: Patch-based physical attacks have increasingly aroused concerns.   However, most existing methods focus on obscuring targets captured on the ground, and some of these methods are simply extended to deceive aerial detectors.   They smear the targeted objects in the physical world with the elaborated adversarial patches, which can only slightly sway the aerial detectors' prediction and with weak attack transferability.   To address the above issues, we propose to perform Contextual Background Attack (CBA), a novel physical attack framework against aerial detection, which can achieve strong attack efficacy and transferability in the physical world even without smudging the interested objects at all.   Specifically, the targets of interest, i.e. the aircraft in aerial images, are adopted to mask adversarial patches.   The pixels outside the mask area are optimized to make the generated adversarial patches closely cover the critical contextual background area for detection, which contributes to gifting adversarial patches with more robust and transferable attack potency in the real world.   To further strengthen the attack performance, the adversarial patches are forced to be outside targets during training, by which the detected objects of interest, both on and outside patches, benefit the accumulation of attack efficacy.   Consequently, the sophisticatedly designed patches are gifted with solid fooling efficacy against objects both on and outside the adversarial patches simultaneously.   Extensive proportionally scaled experiments are performed in physical scenarios, demonstrating the superiority and potential of the proposed framework for physical attacks.   We expect that the proposed physical attack method will serve as a benchmark for assessing the adversarial robustness of diverse aerial detectors and defense methods.

摘要: 基于补丁的物理攻击越来越引起人们的关注。然而，现有的大多数方法都集中在遮挡地面捕获的目标上，其中一些方法只是简单地扩展到欺骗航空探测器。他们用精心制作的对抗性补丁涂抹物理世界中的目标对象，这只能轻微动摇航空探测器的预测，攻击可转移性较弱。为了解决上述问题，我们提出了一种新的针对空中探测的物理攻击框架--上下文背景攻击(CBA)，该框架即使在不玷污感兴趣对象的情况下也可以在物理世界中实现强大的攻击效能和可转移性。具体地说，采用感兴趣的目标，即航空图像中的飞机来掩盖敌方补丁。对掩码区域外的像素进行了优化，使生成的对抗性补丁紧密覆盖关键背景区域进行检测，有助于在现实世界中赋予对抗性补丁更健壮和可转移的攻击能力。为了进一步增强攻击性能，在训练过程中将对抗性补丁强制为外部目标，这样无论是在补丁上还是在补丁外，检测到的感兴趣对象都有利于攻击效能的积累。因此，复杂设计的补丁被赋予了对敌方补丁内外的对象同时具有可靠的愚弄效果。在物理场景中进行了广泛的按比例扩展的实验，展示了所提出的框架在物理攻击方面的优势和潜力。我们期望所提出的物理攻击方法将作为评估不同空中探测器和防御方法的对抗健壮性的基准。



## **8. PolyScope: Multi-Policy Access Control Analysis to Triage Android Scoped Storage**

PolyScope：对Android范围存储进行分类的多策略访问控制分析 cs.CR

14 pages, 5 figures, submitted to IEEE TDSC. arXiv admin note:  substantial text overlap with arXiv:2008.03593

**SubmitDate**: 2023-02-27    [abs](http://arxiv.org/abs/2302.13506v1) [paper-pdf](http://arxiv.org/pdf/2302.13506v1)

**Authors**: Yu-Tsung Lee, Haining Chen, William Enck, Hayawardh Vijayakumar, Ninghui Li, Zhiyun Qian, Giuseppe Petracca

**Abstract**: Android's filesystem access control is a crucial aspect of its system integrity. It utilizes a combination of mandatory access controls, such as SELinux, and discretionary access controls, like Unix permissions, along with specialized access controls such as Android permissions to safeguard OEM and Android services from third-party applications. However, when OEMs introduce differentiating features, they often create vulnerabilities due to their inability to properly reconfigure this complex policy combination. To address this, we introduce the POLYSCOPE tool, which triages Android filesystem access control policies to identify attack operations - authorized operations that may be exploited by adversaries to elevate their privileges. POLYSCOPE has three significant advantages over prior analyses: it allows for the independent extension and analysis of individual policy models, understands the flexibility untrusted parties have in modifying access control policies, and can identify attack operations that system configurations permit. We demonstrate the effectiveness of POLYSCOPE by examining the impact of Scoped Storage on Android, revealing that it reduces the number of attack operations possible on external storage resources by over 50%. However, because OEMs only partially adopt Scoped Storage, we also uncover two previously unknown vulnerabilities, demonstrating how POLYSCOPE can assess an ideal scenario where all apps comply with Scoped Storage, which can reduce the number of untrusted parties accessing attack operations by over 65% on OEM systems. POLYSCOPE thus helps Android OEMs evaluate complex access control policies to pinpoint the attack operations that require further examination.

摘要: Android的文件系统访问控制是其系统完整性的关键方面。它利用强制访问控制(如SELinux)和自主访问控制(如Unix权限)以及专门的访问控制(如Android权限)相结合，以保护OEM和Android服务不受第三方应用程序的影响。然而，当OEM引入差异化功能时，由于无法正确重新配置这一复杂的策略组合，他们往往会产生漏洞。为了解决这个问题，我们引入了PolyScope工具，该工具对Android文件系统访问控制策略进行分类，以识别攻击操作--攻击者可能利用这些授权操作来提升其权限。与以前的分析相比，PolyScope有三个显著的优势：它允许独立扩展和分析单个策略模型，了解不可信任方在修改访问控制策略方面的灵活性，以及可以识别系统配置允许的攻击操作。我们通过检查范围存储对Android的影响来展示PolyScope的有效性，发现它将可能对外部存储资源的攻击操作数量减少了50%以上。然而，由于OEM仅部分采用范围存储，我们还发现了两个以前未知的漏洞，展示了PolyScope如何评估所有应用程序都符合范围存储的理想场景，这可以将访问OEM系统上的攻击操作的不可信方数量减少65%以上。因此，PolyScope可以帮助Android OEM评估复杂的访问控制策略，以查明需要进一步检查的攻击操作。



## **9. Contextual adversarial attack against aerial detection in the physical world**

物理世界中针对空中探测的上下文对抗性攻击 cs.CV

**SubmitDate**: 2023-02-27    [abs](http://arxiv.org/abs/2302.13487v1) [paper-pdf](http://arxiv.org/pdf/2302.13487v1)

**Authors**: Jiawei Lian, Xiaofei Wang, Yuru Su, Mingyang Ma, Shaohui Mei

**Abstract**: Deep Neural Networks (DNNs) have been extensively utilized in aerial detection. However, DNNs' sensitivity and vulnerability to maliciously elaborated adversarial examples have progressively garnered attention. Recently, physical attacks have gradually become a hot issue due to they are more practical in the real world, which poses great threats to some security-critical applications. In this paper, we take the first attempt to perform physical attacks in contextual form against aerial detection in the physical world. We propose an innovative contextual attack method against aerial detection in real scenarios, which achieves powerful attack performance and transfers well between various aerial object detectors without smearing or blocking the interested objects to hide. Based on the findings that the targets' contextual information plays an important role in aerial detection by observing the detectors' attention maps, we propose to make full use of the contextual area of the interested targets to elaborate contextual perturbations for the uncovered attacks in real scenarios. Extensive proportionally scaled experiments are conducted to evaluate the effectiveness of the proposed contextual attack method, which demonstrates the proposed method's superiority in both attack efficacy and physical practicality.

摘要: 深度神经网络在航空探测中得到了广泛的应用。然而，DNN对恶意阐述的敌意例子的敏感性和脆弱性逐渐引起了人们的注意。近年来，物理攻击由于在现实世界中更加实用而逐渐成为一个热点问题，这对一些安全关键型应用程序构成了巨大的威胁。在本文中，我们首次尝试对物理世界中的空中探测执行上下文形式的物理攻击。我们提出了一种针对真实场景中空中检测的上下文攻击方法，该方法实现了强大的攻击性能，并且可以在各种空中目标检测器之间很好地传输，而不会涂抹或遮挡感兴趣的对象进行隐藏。基于目标的上下文信息通过观察检测器的注意图在空中探测中的重要作用这一发现，我们提出了充分利用感兴趣目标的上下文区域来阐述对真实场景中未被发现的攻击的上下文扰动。通过大量的按比例扩展的实验对所提出的上下文攻击方法的有效性进行了评估，证明了该方法在攻击效率和物理实用性方面的优越性。



## **10. Randomness in ML Defenses Helps Persistent Attackers and Hinders Evaluators**

ML防御中的随机性有助于持续攻击者，阻碍评估者 cs.LG

**SubmitDate**: 2023-02-27    [abs](http://arxiv.org/abs/2302.13464v1) [paper-pdf](http://arxiv.org/pdf/2302.13464v1)

**Authors**: Keane Lucas, Matthew Jagielski, Florian Tramèr, Lujo Bauer, Nicholas Carlini

**Abstract**: It is becoming increasingly imperative to design robust ML defenses. However, recent work has found that many defenses that initially resist state-of-the-art attacks can be broken by an adaptive adversary. In this work we take steps to simplify the design of defenses and argue that white-box defenses should eschew randomness when possible. We begin by illustrating a new issue with the deployment of randomized defenses that reduces their security compared to their deterministic counterparts. We then provide evidence that making defenses deterministic simplifies robustness evaluation, without reducing the effectiveness of a truly robust defense. Finally, we introduce a new defense evaluation framework that leverages a defense's deterministic nature to better evaluate its adversarial robustness.

摘要: 设计强大的ML防御系统正变得越来越迫切。然而，最近的研究发现，许多最初抵抗最先进攻击的防御系统可以被适应性强的对手打破。在这项工作中，我们采取措施简化防御设计，并主张白盒防御应尽可能避免随机性。我们首先说明部署随机化防御的一个新问题，与确定性防御相比，随机化防御降低了它们的安全性。然后，我们提供证据表明，使防御具有确定性可以简化健壮性评估，而不会降低真正健壮的防御的有效性。最后，我们引入了一种新的防御评估框架，该框架利用防御的确定性性质来更好地评估其对抗健壮性。



## **11. Investigating the Security of EV Charging Mobile Applications As an Attack Surface**

以电动汽车充电移动应用为攻击面的安全性研究 cs.CR

**SubmitDate**: 2023-02-26    [abs](http://arxiv.org/abs/2211.10603v2) [paper-pdf](http://arxiv.org/pdf/2211.10603v2)

**Authors**: K. Sarieddine, M. A. Sayed, S. Torabi, R. Atallah, C. Assi

**Abstract**: In this paper, we study the security posture of the EV charging ecosystem against a new type of remote that exploits vulnerabilities in the EV charging mobile applications as an attack surface. We leverage a combination of static and dynamic analysis techniques to analyze the security of widely used EV charging mobile applications. Our analysis was performed on 31 of the most widely used mobile applications including their interactions with various components such as cloud management systems. The attack, scenarios that exploit these vulnerabilities were verified on a real-time co-simulation test bed. Our discoveries indicate the lack of user/vehicle verification and improper authorization for critical functions, which allow adversaries to remotely hijack charging sessions and launch attacks against the connected critical infrastructure. The attacks were demonstrated using the EVCS mobile applications showing the feasibility and the applicability of our attacks. Indeed, we discuss specific remote attack scenarios and their impact on EV users. More importantly, our analysis results demonstrate the feasibility of leveraging existing vulnerabilities across various EV charging mobile applications to perform wide-scale coordinated remote charging/discharging attacks against the connected critical infrastructure (e.g., power grid), with significant economical and operational implications. Finally, we propose countermeasures to secure the infrastructure and impede adversaries from performing reconnaissance and launching remote attacks using compromised accounts.

摘要: 本文研究了电动汽车充电生态系统对一种新型遥控器的安全态势，该遥控器利用电动汽车充电移动应用中的漏洞作为攻击面。我们利用静态和动态分析技术的组合来分析广泛使用的电动汽车充电移动应用程序的安全性。我们对31个使用最广泛的移动应用进行了分析，包括它们与云管理系统等各种组件的交互。利用这些漏洞的攻击场景在实时联合模拟试验台上进行了验证。我们的发现表明，缺乏用户/车辆验证和对关键功能的不当授权，这使得攻击者能够远程劫持充电会话，并对连接的关键基础设施发动攻击。这些攻击是使用EVCS移动应用程序演示的，表明了我们攻击的可行性和适用性。事实上，我们讨论了特定的远程攻击场景及其对电动汽车用户的影响。更重要的是，我们的分析结果证明了利用各种电动汽车充电移动应用程序中的现有漏洞对连接的关键基础设施(如电网)执行大规模协同远程充放电攻击的可行性，具有重大的经济和运营影响。最后，我们提出了保护基础设施的对策，并阻止对手使用受攻击的帐户执行侦察和发动远程攻击。



## **12. Adversarial Path Planning for Optimal Camera Positioning**

摄像机最优定位的对抗性路径规划 cs.CG

**SubmitDate**: 2023-02-26    [abs](http://arxiv.org/abs/2302.07051v2) [paper-pdf](http://arxiv.org/pdf/2302.07051v2)

**Authors**: Gaia Carenini, Alexandre Duplessis

**Abstract**: The use of visual sensors is flourishing, driven among others by the several applications in detection and prevention of crimes or dangerous events. While the problem of optimal camera placement for total coverage has been solved for a decade or so, that of the arrangement of cameras maximizing the recognition of objects "in-transit" is still open. The objective of this paper is to attack this problem by providing an adversarial method of proven optimality based on the resolution of Hamilton-Jacobi equations. The problem is attacked by first assuming the perspective of an adversary, i.e. computing explicitly the path minimizing the probability of detection and the quality of reconstruction. Building on this result, we introduce an optimality measure for camera configurations and perform a simulated annealing algorithm to find the optimal camera placement.

摘要: 视觉传感器的使用正在蓬勃发展，其中包括在侦测和预防犯罪或危险事件方面的几种应用。虽然总覆盖范围的最佳摄像机布置问题已经解决了十年左右，但如何布置摄像机以最大限度地识别“过境”物体的问题仍然悬而未决。本文的目的是通过提供一种基于Hamilton-Jacobi方程的解的证明最优性的对抗性方法来解决这一问题。该问题首先从敌方的角度出发，即显式计算最小化检测概率和重构质量的路径。在此基础上，我们引入了摄像机配置的最优性度量，并用模拟退火法来寻找摄像机的最优布置。



## **13. Empowering Graph Representation Learning with Test-Time Graph Transformation**

利用测试时间图变换增强图表示学习能力 cs.LG

ICLR 2023

**SubmitDate**: 2023-02-26    [abs](http://arxiv.org/abs/2210.03561v2) [paper-pdf](http://arxiv.org/pdf/2210.03561v2)

**Authors**: Wei Jin, Tong Zhao, Jiayuan Ding, Yozen Liu, Jiliang Tang, Neil Shah

**Abstract**: As powerful tools for representation learning on graphs, graph neural networks (GNNs) have facilitated various applications from drug discovery to recommender systems. Nevertheless, the effectiveness of GNNs is immensely challenged by issues related to data quality, such as distribution shift, abnormal features and adversarial attacks. Recent efforts have been made on tackling these issues from a modeling perspective which requires additional cost of changing model architectures or re-training model parameters. In this work, we provide a data-centric view to tackle these issues and propose a graph transformation framework named GTrans which adapts and refines graph data at test time to achieve better performance. We provide theoretical analysis on the design of the framework and discuss why adapting graph data works better than adapting the model. Extensive experiments have demonstrated the effectiveness of GTrans on three distinct scenarios for eight benchmark datasets where suboptimal data is presented. Remarkably, GTrans performs the best in most cases with improvements up to 2.8%, 8.2% and 3.8% over the best baselines on three experimental settings. Code is released at https://github.com/ChandlerBang/GTrans.

摘要: 作为图上表示学习的强大工具，图神经网络(GNN)促进了从药物发现到推荐系统的各种应用。然而，GNN的有效性受到与数据质量有关的问题的巨大挑战，例如分布偏移、异常特征和对抗性攻击。最近在从建模的角度解决这些问题方面做出了努力，这需要更改模型体系结构或重新培训模型参数的额外成本。在这项工作中，我们提供了一个以数据为中心的视图来解决这些问题，并提出了一个名为GTrans的图转换框架，它在测试时对图数据进行调整和提炼，以获得更好的性能。我们对框架的设计进行了理论分析，并讨论了为什么采用图形数据比采用模型效果更好。广泛的实验已经证明了GTrans在三种不同的场景中的有效性，这些场景针对八个基准数据集，其中呈现的数据不是最优的。值得注意的是，在大多数情况下，GTrans的性能最好，在三个实验设置中，GTrans的性能比最佳基线分别提高了2.8%、8.2%和3.8%。代码在https://github.com/ChandlerBang/GTrans.上发布



## **14. Deep Learning-based Multi-Organ CT Segmentation with Adversarial Data Augmentation**

基于深度学习的对抗性数据增强多器官CT分割 eess.IV

Accepted at SPIE Medical Imaging 2023

**SubmitDate**: 2023-02-25    [abs](http://arxiv.org/abs/2302.13172v1) [paper-pdf](http://arxiv.org/pdf/2302.13172v1)

**Authors**: Shaoyan Pan, Shao-Yuan Lo, Min Huang, Chaoqiong Ma, Jacob Wynne, Tonghe Wang, Tian Liu, Xiaofeng Yang

**Abstract**: In this work, we propose an adversarial attack-based data augmentation method to improve the deep-learning-based segmentation algorithm for the delineation of Organs-At-Risk (OAR) in abdominal Computed Tomography (CT) to facilitate radiation therapy. We introduce Adversarial Feature Attack for Medical Image (AFA-MI) augmentation, which forces the segmentation network to learn out-of-distribution statistics and improve generalization and robustness to noises. AFA-MI augmentation consists of three steps: 1) generate adversarial noises by Fast Gradient Sign Method (FGSM) on the intermediate features of the segmentation network's encoder; 2) inject the generated adversarial noises into the network, intentionally compromising performance; 3) optimize the network with both clean and adversarial features. Experiments are conducted segmenting the heart, left and right kidney, liver, left and right lung, spinal cord, and stomach. We first evaluate the AFA-MI augmentation using nnUnet and TT-Vnet on the test data from a public abdominal dataset and an institutional dataset. In addition, we validate how AFA-MI affects the networks' robustness to the noisy data by evaluating the networks with added Gaussian noises of varying magnitudes to the institutional dataset. Network performance is quantitatively evaluated using Dice Similarity Coefficient (DSC) for volume-based accuracy. Also, Hausdorff Distance (HD) is applied for surface-based accuracy. On the public dataset, nnUnet with AFA-MI achieves DSC = 0.85 and HD = 6.16 millimeters (mm); and TT-Vnet achieves DSC = 0.86 and HD = 5.62 mm. AFA-MI augmentation further improves all contour accuracies up to 0.217 DSC score when tested on images with Gaussian noises. AFA-MI augmentation is therefore demonstrated to improve segmentation performance and robustness in CT multi-organ segmentation.

摘要: 在这项工作中，我们提出了一种基于对抗性攻击的数据增强方法，以改进基于深度学习的分割算法，在腹部计算机断层扫描(CT)中勾画危险器官(OAR)，以便于放射治疗。我们引入了对抗性特征攻击(AFA-MI)来增强医学图像，迫使分割网络学习非分布统计，提高了泛化能力和抗噪声能力。AFA-MI增强包括三个步骤：1)在分段网络编码器的中间特征上使用快速梯度符号方法(FGSM)产生对抗性噪声；2)将产生的对抗性噪声注入网络，故意损害网络性能；3)同时使用干净和对抗性特征来优化网络。实验对心脏、左肾、右肾、肝、左肺、右肺、脊髓和胃进行分割。我们首先使用nnUnet和TT-vNet对AFA-MI增强进行评估，这些数据来自公共腹部数据集和机构数据集。此外，我们通过评估在机构数据集中添加了不同幅度的高斯噪声的网络来验证AFA-MI如何影响网络对噪声数据的稳健性。网络性能使用Dice相似性系数(DSC)进行定量评估，以获得基于体积的准确性。此外，Hausdorff距离(HD)用于基于曲面的精度。在公共数据集上，使用AFA-MI的nnUnet的DSC=0.85，HD=6.16 mm；TT-vNet的DSC=0.86，HD=5.62 mm。当在含有高斯噪声的图像上测试时，AFA-MI增强进一步提高了所有轮廓精度，最高可达0.217 DSC分数。因此，在CT多器官分割中，AFA-MI增强被证明能够提高分割性能和稳健性。



## **15. Chaotic Variational Auto encoder-based Adversarial Machine Learning**

基于混沌变分自动编码器的对抗性机器学习 cs.LG

24 pages, 6 figures and 5 tables

**SubmitDate**: 2023-02-25    [abs](http://arxiv.org/abs/2302.12959v1) [paper-pdf](http://arxiv.org/pdf/2302.12959v1)

**Authors**: Pavan Venkata Sainadh Reddy, Yelleti Vivek, Gopi Pranay, Vadlamani Ravi

**Abstract**: Machine Learning (ML) has become the new contrivance in almost every field. This makes them a target of fraudsters by various adversary attacks, thereby hindering the performance of ML models. Evasion and Data-Poison-based attacks are well acclaimed, especially in finance, healthcare, etc. This motivated us to propose a novel computationally less expensive attack mechanism based on the adversarial sample generation by Variational Auto Encoder (VAE). It is well known that Wavelet Neural Network (WNN) is considered computationally efficient in solving image and audio processing, speech recognition, and time-series forecasting. This paper proposed VAE-Deep-Wavelet Neural Network (VAE-Deep-WNN), where Encoder and Decoder employ WNN networks. Further, we proposed chaotic variants of both VAE with Multi-layer perceptron (MLP) and Deep-WNN and named them C-VAE-MLP and C-VAE-Deep-WNN, respectively. Here, we employed a Logistic map to generate random noise in the latent space. In this paper, we performed VAE-based adversary sample generation and applied it to various problems related to finance and cybersecurity domain-related problems such as loan default, credit card fraud, and churn modelling, etc., We performed both Evasion and Data-Poison attacks on Logistic Regression (LR) and Decision Tree (DT) models. The results indicated that VAE-Deep-WNN outperformed the rest in the majority of the datasets and models. However, its chaotic variant C-VAE-Deep-WNN performed almost similarly to VAE-Deep-WNN in the majority of the datasets.

摘要: 机器学习(ML)已成为几乎所有领域的新工具。这使得它们成为各种对手攻击的诈骗者的目标，从而阻碍了ML模型的性能。基于逃避和基于数据毒物的攻击受到了广泛的好评，特别是在金融、医疗等领域。这促使我们提出了一种新的计算成本较低的攻击机制，该机制基于可变自动编码器(VAE)生成的对抗性样本。众所周知，小波神经网络(WNN)在解决图像和音频处理、语音识别和时间序列预测方面具有计算效率。本文提出了VAE-Deep-Wavelet神经网络(VAE-Deep-WNN)，其中编解码器采用WNN网络。进一步，我们提出了多层感知器(MLP)的VAE和Deep-WNN的混沌变体，并分别命名为C-VAE-MLP和C-VAE-Deep-WNN。在这里，我们使用Logistic映射在潜在空间中产生随机噪声。在本文中，我们进行了基于VAE的对手样本生成，并将其应用于各种与金融和网络安全领域相关的问题，如贷款违约、信用卡欺诈和流失建模等。我们在Logistic回归(LR)和决策树(DT)模型上进行了逃避攻击和数据毒害攻击。结果表明，VAE-Deep-WNN在大多数数据集和模型中的表现优于其他方法。然而，它的混沌变体C-VAE-Deep-WNN在大多数数据集上的性能几乎与VAE-Deep-WNN相似。



## **16. Edge-Based Detection and Localization of Adversarial Oscillatory Load Attacks Orchestrated By Compromised EV Charging Stations**

基于边缘的检测和定位受危害电动汽车充电站策划的对抗性振荡负载攻击 cs.CR

**SubmitDate**: 2023-02-24    [abs](http://arxiv.org/abs/2302.12890v1) [paper-pdf](http://arxiv.org/pdf/2302.12890v1)

**Authors**: Khaled Sarieddine, Mohammad Ali Sayed, Sadegh Torabi, Ribal Atallah, Chadi Assi

**Abstract**: In this paper, we investigate an edge-based approach for the detection and localization of coordinated oscillatory load attacks initiated by exploited EV charging stations against the power grid. We rely on the behavioral characteristics of the power grid in the presence of interconnected EVCS while combining cyber and physical layer features to implement deep learning algorithms for the effective detection of oscillatory load attacks at the EVCS. We evaluate the proposed detection approach by building a real-time test bed to synthesize benign and malicious data, which was generated by analyzing real-life EV charging data collected during recent years. The results demonstrate the effectiveness of the implemented approach with the Convolutional Long-Short Term Memory model producing optimal classification accuracy (99.4\%). Moreover, our analysis results shed light on the impact of such detection mechanisms towards building resiliency into different levels of the EV charging ecosystem while allowing power grid operators to localize attacks and take further mitigation measures. Specifically, we managed to decentralize the detection mechanism of oscillatory load attacks and create an effective alternative for operator-centric mechanisms to mitigate multi-operator and MitM oscillatory load attacks against the power grid. Finally, we leverage the created test bed to evaluate a distributed mitigation technique, which can be deployed on public/private charging stations to average out the impact of oscillatory load attacks while allowing the power system to recover smoothly within 1 second with minimal overhead.

摘要: 本文研究了一种基于边缘的方法来检测和定位被利用的电动汽车充电站对电网发起的协同振荡负载攻击。利用互联EVCS下电网的行为特征，结合网络层和物理层特征，实现深度学习算法，有效检测EVCS上的振荡负载攻击。我们构建了一个实时测试床来综合良性和恶意数据，这些数据是通过分析近年来收集的真实电动汽车充电数据生成的，从而对提出的检测方法进行了评估。实验结果表明，卷积长短时记忆模型的分类精度最高(99.4)，是有效的。此外，我们的分析结果揭示了此类检测机制对在不同级别的电动汽车充电生态系统中构建弹性的影响，同时允许电网运营商定位攻击并采取进一步的缓解措施。具体地说，我们成功地分散了振荡负载攻击的检测机制，并为以操作员为中心的机制创建了一个有效的替代方案，以缓解针对电网的多操作员和MITM振荡负载攻击。最后，我们利用创建的试验台来评估分布式缓解技术，该技术可以部署在公共/私人充电站上，以平均负载振荡攻击的影响，同时允许电力系统以最小的开销在1秒内平稳恢复。



## **17. Take Me Home: Reversing Distribution Shifts using Reinforcement Learning**

带我回家：使用强化学习逆转分布变化 cs.LG

preprint (under submission)

**SubmitDate**: 2023-02-24    [abs](http://arxiv.org/abs/2302.10341v2) [paper-pdf](http://arxiv.org/pdf/2302.10341v2)

**Authors**: Vivian Lin, Kuk Jin Jang, Souradeep Dutta, Michele Caprio, Oleg Sokolsky, Insup Lee

**Abstract**: Deep neural networks have repeatedly been shown to be non-robust to the uncertainties of the real world. Even subtle adversarial attacks and naturally occurring distribution shifts wreak havoc on systems relying on deep neural networks. In response to this, current state-of-the-art techniques use data-augmentation to enrich the training distribution of the model and consequently improve robustness to natural distribution shifts. We propose an alternative approach that allows the system to recover from distribution shifts online. Specifically, our method applies a sequence of semantic-preserving transformations to bring the shifted data closer in distribution to the training set, as measured by the Wasserstein distance. We formulate the problem of sequence selection as an MDP, which we solve using reinforcement learning. To aid in our estimates of Wasserstein distance, we employ dimensionality reduction through orthonormal projection. We provide both theoretical and empirical evidence that orthonormal projection preserves characteristics of the data at the distributional level. Finally, we apply our distribution shift recovery approach to the ImageNet-C benchmark for distribution shifts, targeting shifts due to additive noise and image histogram modifications. We demonstrate an improvement in average accuracy up to 14.21% across a variety of state-of-the-art ImageNet classifiers.

摘要: 深度神经网络已多次被证明对现实世界的不确定性是不稳健的。即使是微妙的对抗性攻击和自然发生的分布变化，也会对依赖深度神经网络的系统造成严重破坏。作为对此的响应，当前最先进的技术使用数据扩充来丰富模型的训练分布，从而提高对自然分布变化的稳健性。我们提出了一种替代方法，允许系统从在线分配班次中恢复。具体地说，我们的方法应用了一系列保持语义的转换，以使移动的数据在分布上更接近训练集，这是通过Wasserstein距离来衡量的。我们将序列选择问题描述为一个MDP，并使用强化学习来解决该问题。为了帮助我们估计Wasserstein距离，我们使用了通过正交化投影进行降维的方法。我们提供了理论和经验证据，证明了正交化投影在分布水平上保留了数据的特征。最后，我们将我们的分布移位恢复方法应用到ImageNet-C基准测试中，用于计算由于加性噪声和图像直方图修改而导致的目标移位。我们证明，在各种最先进的ImageNet分类器上，平均准确率提高了14.21%。



## **18. Adversarial Robustness for Tabular Data through Cost and Utility Awareness**

通过成本和效用意识实现表格数据的对抗稳健性 cs.LG

The first two authors contributed equally. To appear in the  proceedings of NDSS 2023

**SubmitDate**: 2023-02-24    [abs](http://arxiv.org/abs/2208.13058v2) [paper-pdf](http://arxiv.org/pdf/2208.13058v2)

**Authors**: Klim Kireev, Bogdan Kulynych, Carmela Troncoso

**Abstract**: Many safety-critical applications of machine learning, such as fraud or abuse detection, use data in tabular domains. Adversarial examples can be particularly damaging for these applications. Yet, existing works on adversarial robustness primarily focus on machine-learning models in image and text domains. We argue that, due to the differences between tabular data and images or text, existing threat models are not suitable for tabular domains. These models do not capture that the costs of an attack could be more significant than imperceptibility, or that the adversary could assign different values to the utility obtained from deploying different adversarial examples. We demonstrate that, due to these differences, the attack and defense methods used for images and text cannot be directly applied to tabular settings. We address these issues by proposing new cost and utility-aware threat models that are tailored to the adversarial capabilities and constraints of attackers targeting tabular domains. We introduce a framework that enables us to design attack and defense mechanisms that result in models protected against cost and utility-aware adversaries, for example, adversaries constrained by a certain financial budget. We show that our approach is effective on three datasets corresponding to applications for which adversarial examples can have economic and social implications.

摘要: 机器学习的许多安全关键应用程序，如欺诈或滥用检测，都使用表格域中的数据。对抗性的例子可能对这些应用程序特别有害。然而，现有的关于对抗稳健性的工作主要集中在图像和文本领域的机器学习模型上。我们认为，由于表格数据与图像或文本之间的差异，现有的威胁模型不适用于表格领域。这些模型没有考虑到攻击的成本可能比不可察觉更重要，或者对手可以为通过部署不同的对手例子而获得的效用分配不同的值。我们证明，由于这些差异，用于图像和文本的攻击和防御方法不能直接应用于表格设置。我们通过提出新的成本和效用感知威胁模型来解决这些问题，该模型针对针对表格域的攻击者的对抗能力和约束而量身定做。我们引入了一个框架，使我们能够设计攻击和防御机制，从而产生针对成本和效用意识的对手的模型，例如，受特定财务预算约束的对手。我们表明，我们的方法在三个数据集上是有效的，这些数据集对应于应用程序，对于这些应用程序，对抗性例子可能具有经济和社会影响。



## **19. Defending Against Backdoor Attacks by Layer-wise Feature Analysis**

基于分层特征分析的后门攻击防御 cs.CR

This paper is accepted by PAKDD 2023

**SubmitDate**: 2023-02-24    [abs](http://arxiv.org/abs/2302.12758v1) [paper-pdf](http://arxiv.org/pdf/2302.12758v1)

**Authors**: Najeeb Moharram Jebreel, Josep Domingo-Ferrer, Yiming Li

**Abstract**: Training deep neural networks (DNNs) usually requires massive training data and computational resources. Users who cannot afford this may prefer to outsource training to a third party or resort to publicly available pre-trained models. Unfortunately, doing so facilitates a new training-time attack (i.e., backdoor attack) against DNNs. This attack aims to induce misclassification of input samples containing adversary-specified trigger patterns. In this paper, we first conduct a layer-wise feature analysis of poisoned and benign samples from the target class. We find out that the feature difference between benign and poisoned samples tends to be maximum at a critical layer, which is not always the one typically used in existing defenses, namely the layer before fully-connected layers. We also demonstrate how to locate this critical layer based on the behaviors of benign samples. We then propose a simple yet effective method to filter poisoned samples by analyzing the feature differences between suspicious and benign samples at the critical layer. We conduct extensive experiments on two benchmark datasets, which confirm the effectiveness of our defense.

摘要: 训练深度神经网络(DNN)通常需要大量的训练数据和计算资源。负担不起这种费用的用户可能更愿意将培训外包给第三方，或者求助于公开提供的预先培训的模型。不幸的是，这样做为针对DNN的新的训练时间攻击(即后门攻击)提供了便利。该攻击旨在对包含对手指定的触发模式的输入样本进行错误分类。在本文中，我们首先对目标类中的有毒和良性样本进行分层特征分析。我们发现，良性和有毒样本之间的特征差异往往在关键层达到最大，这并不总是现有防御措施中通常使用的层，即完全连通层之前的层。我们还演示了如何根据良性样本的行为来定位这一临界层。在此基础上，通过分析可疑样本和良性样本在关键层的特征差异，提出了一种简单有效的有毒样本过滤方法。我们在两个基准数据集上进行了大量的实验，验证了我们的防御方法的有效性。



## **20. Harnessing the Speed and Accuracy of Machine Learning to Advance Cybersecurity**

利用机器学习的速度和准确性提高网络安全 cs.CR

**SubmitDate**: 2023-02-24    [abs](http://arxiv.org/abs/2302.12415v1) [paper-pdf](http://arxiv.org/pdf/2302.12415v1)

**Authors**: Khatoon Mohammed

**Abstract**: As cyber attacks continue to increase in frequency and sophistication, detecting malware has become a critical task for maintaining the security of computer systems. Traditional signature-based methods of malware detection have limitations in detecting complex and evolving threats. In recent years, machine learning (ML) has emerged as a promising solution to detect malware effectively. ML algorithms are capable of analyzing large datasets and identifying patterns that are difficult for humans to identify. This paper presents a comprehensive review of the state-of-the-art ML techniques used in malware detection, including supervised and unsupervised learning, deep learning, and reinforcement learning. We also examine the challenges and limitations of ML-based malware detection, such as the potential for adversarial attacks and the need for large amounts of labeled data. Furthermore, we discuss future directions in ML-based malware detection, including the integration of multiple ML algorithms and the use of explainable AI techniques to enhance the interpret ability of ML-based detection systems. Our research highlights the potential of ML-based techniques to improve the speed and accuracy of malware detection, and contribute to enhancing cybersecurity

摘要: 随着网络攻击的频率和复杂性不断增加，检测恶意软件已成为维护计算机系统安全的关键任务。传统的基于特征码的恶意软件检测方法在检测复杂和不断变化的威胁方面存在局限性。近年来，机器学习作为一种有效检测恶意软件的解决方案应运而生。ML算法能够分析大型数据集，并识别人类难以识别的模式。本文对恶意软件检测中使用的最大似然学习技术进行了全面的综述，包括监督学习和非监督学习、深度学习和强化学习。我们还研究了基于ML的恶意软件检测的挑战和局限性，例如潜在的对抗性攻击和对大量标记数据的需求。此外，我们还讨论了基于ML的恶意软件检测的未来发展方向，包括集成多种ML算法和使用可解释人工智能技术来增强基于ML的检测系统的解释能力。我们的研究突出了基于ML的技术在提高恶意软件检测的速度和准确性方面的潜力，并有助于增强网络安全



## **21. Principled Data-Driven Decision Support for Cyber-Forensic Investigations**

面向网络取证调查的原则性数据驱动决策支持 cs.CR

**SubmitDate**: 2023-02-24    [abs](http://arxiv.org/abs/2211.13345v2) [paper-pdf](http://arxiv.org/pdf/2211.13345v2)

**Authors**: Soodeh Atefi, Sakshyam Panda, Manos Panaousis, Aron Laszka

**Abstract**: In the wake of a cybersecurity incident, it is crucial to promptly discover how the threat actors breached security in order to assess the impact of the incident and to develop and deploy countermeasures that can protect against further attacks. To this end, defenders can launch a cyber-forensic investigation, which discovers the techniques that the threat actors used in the incident. A fundamental challenge in such an investigation is prioritizing the investigation of particular techniques since the investigation of each technique requires time and effort, but forensic analysts cannot know which ones were actually used before investigating them. To ensure prompt discovery, it is imperative to provide decision support that can help forensic analysts with this prioritization. A recent study demonstrated that data-driven decision support, based on a dataset of prior incidents, can provide state-of-the-art prioritization. However, this data-driven approach, called DISCLOSE, is based on a heuristic that utilizes only a subset of the available information and does not approximate optimal decisions. To improve upon this heuristic, we introduce a principled approach for data-driven decision support for cyber-forensic investigations. We formulate the decision-support problem using a Markov decision process, whose states represent the states of a forensic investigation. To solve the decision problem, we propose a Monte Carlo tree search based method, which relies on a k-NN regression over prior incidents to estimate state-transition probabilities. We evaluate our proposed approach on multiple versions of the MITRE ATT&CK dataset, which is a knowledge base of adversarial techniques and tactics based on real-world cyber incidents, and demonstrate that our approach outperforms DISCLOSE in terms of techniques discovered per effort spent.

摘要: 在网络安全事件发生后，至关重要的是迅速发现威胁行为者如何破坏安全，以便评估事件的影响，并制定和部署能够防止进一步攻击的对策。为此，防御者可以发起网络取证调查，发现威胁行为者在事件中使用的技术。这种调查的一个根本挑战是优先调查特定的技术，因为每种技术的调查都需要时间和精力，但法医分析人员在调查之前无法知道实际使用了哪些技术。为了确保及时发现，必须提供决策支持，帮助法医分析员确定优先顺序。最近的一项研究表明，数据驱动的决策支持基于先前事件的数据集，可以提供最先进的优先级。然而，这种数据驱动的方法称为公开，它基于仅利用可用信息的子集的启发式方法，而不是近似最优决策。为了改进这个启发式方法，我们引入了一种原则性的方法，用于网络取证调查的数据驱动决策支持。我们使用马尔可夫决策过程来描述决策支持问题，其状态表示法医调查的状态。为了解决这一决策问题，我们提出了一种基于蒙特卡罗树搜索的方法，该方法依赖于对先前事件的k-NN回归来估计状态转移概率。我们在MITRE ATT&CK数据集的多个版本上对我们提出的方法进行了评估，该数据集是基于真实网络事件的对抗性技术和战术的知识库，并证明了我们的方法在每花费一次努力发现的技术方面优于公开的。



## **22. HyperAttack: Multi-Gradient-Guided White-box Adversarial Structure Attack of Hypergraph Neural Networks**

HyperAttack：多梯度引导的超图神经网络白盒对抗结构攻击 cs.LG

10+2pages,9figures

**SubmitDate**: 2023-02-24    [abs](http://arxiv.org/abs/2302.12407v1) [paper-pdf](http://arxiv.org/pdf/2302.12407v1)

**Authors**: Chao Hu, Ruishi Yu, Binqi Zeng, Yu Zhan, Ying Fu, Quan Zhang, Rongkai Liu, Heyuan Shi

**Abstract**: Hypergraph neural networks (HGNN) have shown superior performance in various deep learning tasks, leveraging the high-order representation ability to formulate complex correlations among data by connecting two or more nodes through hyperedge modeling. Despite the well-studied adversarial attacks on Graph Neural Networks (GNN), there is few study on adversarial attacks against HGNN, which leads to a threat to the safety of HGNN applications. In this paper, we introduce HyperAttack, the first white-box adversarial attack framework against hypergraph neural networks. HyperAttack conducts a white-box structure attack by perturbing hyperedge link status towards the target node with the guidance of both gradients and integrated gradients. We evaluate HyperAttack on the widely-used Cora and PubMed datasets and three hypergraph neural networks with typical hypergraph modeling techniques. Compared to state-of-the-art white-box structural attack methods for GNN, HyperAttack achieves a 10-20X improvement in time efficiency while also increasing attack success rates by 1.3%-3.7%. The results show that HyperAttack can achieve efficient adversarial attacks that balance effectiveness and time costs.

摘要: 超图神经网络(HGNN)在各种深度学习任务中表现出了优异的性能，它利用高阶表示能力通过超边建模连接两个或多个节点来表达数据之间的复杂相关性。尽管针对图神经网络的敌意攻击已经得到了很好的研究，但针对HGNN的对抗性攻击的研究却很少，这对HGNN应用的安全构成了威胁。本文介绍了第一个针对超图神经网络的白盒对抗攻击框架HyperAttack。HyperAttack在梯度和积分梯度的指导下，通过向目标节点扰动超边缘链路状态来进行白盒结构攻击。我们在广泛使用的Cora和PubMed数据集和三个超图神经网络上使用典型的超图建模技术对HyperAttack进行了评估。与最先进的针对GNN的白盒结构攻击方法相比，HyperAttack在时间效率上提高了10-20倍，同时攻击成功率也提高了1.3%-3.7%。结果表明，HyperAttack能够实现高效的对抗性攻击，平衡了攻击的有效性和时间开销。



## **23. On the Hardness of Robustness Transfer: A Perspective from Rademacher Complexity over Symmetric Difference Hypothesis Space**

关于稳健性迁移的难度：基于对称差分假设空间的Rademacher复杂性 cs.LG

**SubmitDate**: 2023-02-23    [abs](http://arxiv.org/abs/2302.12351v1) [paper-pdf](http://arxiv.org/pdf/2302.12351v1)

**Authors**: Yuyang Deng, Nidham Gazagnadou, Junyuan Hong, Mehrdad Mahdavi, Lingjuan Lyu

**Abstract**: Recent studies demonstrated that the adversarially robust learning under $\ell_\infty$ attack is harder to generalize to different domains than standard domain adaptation. How to transfer robustness across different domains has been a key question in domain adaptation field. To investigate the fundamental difficulty behind adversarially robust domain adaptation (or robustness transfer), we propose to analyze a key complexity measure that controls the cross-domain generalization: the adversarial Rademacher complexity over {\em symmetric difference hypothesis space} $\mathcal{H} \Delta \mathcal{H}$. For linear models, we show that adversarial version of this complexity is always greater than the non-adversarial one, which reveals the intrinsic hardness of adversarially robust domain adaptation. We also establish upper bounds on this complexity measure. Then we extend them to the ReLU neural network class by upper bounding the adversarial Rademacher complexity in the binary classification setting. Finally, even though the robust domain adaptation is provably harder, we do find positive relation between robust learning and standard domain adaptation. We explain \emph{how adversarial training helps domain adaptation in terms of standard risk}. We believe our results initiate the study of the generalization theory of adversarially robust domain adaptation, and could shed lights on distributed adversarially robust learning from heterogeneous sources, e.g., federated learning scenario.

摘要: 最近的研究表明，与标准领域适应相比，在攻击下的对抗性稳健学习更难推广到不同的领域。如何在不同领域之间传递健壮性一直是领域自适应领域的一个关键问题。为了研究对抗性健壮域适应(或健壮性转移)背后的根本困难，我们提出了一种控制跨域泛化的关键复杂性度量：对称差分假设空间上的对抗性Rademacher复杂性\数学{H}\Delta\数学{H}。对于线性模型，我们证明了这种复杂性的对抗性版本总是大于非对抗性版本，这揭示了对抗性稳健领域适应的内在难度。我们还建立了这个复杂性度量的上界。然后，我们通过在二分类环境下的对抗性Rademacher复杂度的上界将它们扩展到RELU神经网络类。最后，即使稳健的领域适应被证明是困难的，我们确实发现稳健学习和标准领域适应之间存在正相关关系。我们解释了[对抗性训练如何在标准风险方面帮助领域适应]。我们相信，我们的结果开启了对抗性鲁棒领域自适应泛化理论的研究，并可能为从异质来源(如联合学习场景)进行分布式对抗性稳健学习提供帮助。



## **24. Characterizing Internal Evasion Attacks in Federated Learning**

联合学习中内部逃避攻击的特征 cs.LG

16 pages, 8 figures (14 images if counting sub-figures separately),  Camera ready version for AISTATS 2023, longer version of paper submitted to  CrossFL 2022 poster workshop, code available at  (https://github.com/tj-kim/pFedDef_v1)

**SubmitDate**: 2023-02-23    [abs](http://arxiv.org/abs/2209.08412v2) [paper-pdf](http://arxiv.org/pdf/2209.08412v2)

**Authors**: Taejin Kim, Shubhranshu Singh, Nikhil Madaan, Carlee Joe-Wong

**Abstract**: Federated learning allows for clients in a distributed system to jointly train a machine learning model. However, clients' models are vulnerable to attacks during the training and testing phases. In this paper, we address the issue of adversarial clients performing "internal evasion attacks": crafting evasion attacks at test time to deceive other clients. For example, adversaries may aim to deceive spam filters and recommendation systems trained with federated learning for monetary gain. The adversarial clients have extensive information about the victim model in a federated learning setting, as weight information is shared amongst clients. We are the first to characterize the transferability of such internal evasion attacks for different learning methods and analyze the trade-off between model accuracy and robustness depending on the degree of similarities in client data. We show that adversarial training defenses in the federated learning setting only display limited improvements against internal attacks. However, combining adversarial training with personalized federated learning frameworks increases relative internal attack robustness by 60% compared to federated adversarial training and performs well under limited system resources.

摘要: 联合学习允许分布式系统中的客户端联合训练机器学习模型。然而，客户的模型在培训和测试阶段很容易受到攻击。在本文中，我们讨论了敌意客户执行“内部逃避攻击”的问题：在测试时精心设计逃避攻击以欺骗其他客户。例如，对手的目标可能是欺骗经过联合学习培训的垃圾邮件过滤器和推荐系统，以换取金钱利益。当权重信息在客户端之间共享时，敌意客户端在联合学习环境中具有关于受害者模型的大量信息。我们首次针对不同的学习方法刻画了这种内部规避攻击的可转移性，并根据客户数据的相似程度分析了模型精度和稳健性之间的权衡。我们表明，在联合学习环境中，对抗性训练防御对内部攻击的改善有限。然而，将对抗性训练与个性化的联合学习框架相结合，与联合对抗性训练相比，相对内部攻击健壮性提高了60%，并且在有限的系统资源下表现良好。



## **25. Boosting Adversarial Transferability using Dynamic Cues**

利用动态线索提高对抗性转移能力 cs.CV

International Conference on Learning Representations (ICLR'23),  Code:https://bit.ly/3Xd9gRQ

**SubmitDate**: 2023-02-23    [abs](http://arxiv.org/abs/2302.12252v1) [paper-pdf](http://arxiv.org/pdf/2302.12252v1)

**Authors**: Muzammal Naseer, Ahmad Mahmood, Salman Khan, Fahad Khan

**Abstract**: The transferability of adversarial perturbations between image models has been extensively studied. In this case, an attack is generated from a known surrogate \eg, the ImageNet trained model, and transferred to change the decision of an unknown (black-box) model trained on an image dataset. However, attacks generated from image models do not capture the dynamic nature of a moving object or a changing scene due to a lack of temporal cues within image models. This leads to reduced transferability of adversarial attacks from representation-enriched \emph{image} models such as Supervised Vision Transformers (ViTs), Self-supervised ViTs (\eg, DINO), and Vision-language models (\eg, CLIP) to black-box \emph{video} models. In this work, we induce dynamic cues within the image models without sacrificing their original performance on images. To this end, we optimize \emph{temporal prompts} through frozen image models to capture motion dynamics. Our temporal prompts are the result of a learnable transformation that allows optimizing for temporal gradients during an adversarial attack to fool the motion dynamics. Specifically, we introduce spatial (image) and temporal (video) cues within the same source model through task-specific prompts. Attacking such prompts maximizes the adversarial transferability from image-to-video and image-to-image models using the attacks designed for image models. Our attack results indicate that the attacker does not need specialized architectures, \eg, divided space-time attention, 3D convolutions, or multi-view convolution networks for different data modalities. Image models are effective surrogates to optimize an adversarial attack to fool black-box models in a changing environment over time. Code is available at https://bit.ly/3Xd9gRQ

摘要: 对抗性扰动在图像模型之间的可转移性已被广泛研究。在这种情况下，从已知的代理(例如，ImageNet训练的模型)生成攻击，并将其传输以改变在图像数据集上训练的未知(黑盒)模型的决策。然而，由于图像模型中缺乏时间线索，从图像模型生成的攻击不能捕捉到运动对象或变化的场景的动态性质。这导致对抗性攻击从表示丰富的{图像}模型，例如监督视觉转换器(VITS)、自我监督VITS(例如，DINO)和视觉语言模型(例如，CLIP)转移到黑盒\EMPH{VIDEO}模型。在这项工作中，我们在不牺牲图像的原始性能的情况下，在图像模型中诱导动态线索。为此，我们通过冻结图像模型来优化时间提示，以捕捉运动动力学。我们的时间提示是一种可学习转换的结果，这种转换允许在敌方攻击期间优化时间梯度，以愚弄运动动力学。具体地说，我们通过特定于任务的提示在同一来源模型中引入空间(图像)和时间(视频)线索。利用为图像模型设计的攻击，攻击此类提示最大限度地提高了图像到视频和图像到图像模型之间的对抗性。我们的攻击结果表明，攻击者不需要专门的体系结构，例如，分割时空注意力、3D卷积或针对不同数据形态的多视点卷积网络。图像模型是优化对抗性攻击的有效替代品，可以在随时间变化的环境中愚弄黑盒模型。代码可在https://bit.ly/3Xd9gRQ上找到



## **26. More than you've asked for: A Comprehensive Analysis of Novel Prompt Injection Threats to Application-Integrated Large Language Models**

比您要求的更多：对应用程序集成的大型语言模型的新型即时注入威胁的全面分析 cs.CR

**SubmitDate**: 2023-02-23    [abs](http://arxiv.org/abs/2302.12173v1) [paper-pdf](http://arxiv.org/pdf/2302.12173v1)

**Authors**: Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, Mario Fritz

**Abstract**: We are currently witnessing dramatic advances in the capabilities of Large Language Models (LLMs). They are already being adopted in practice and integrated into many systems, including integrated development environments (IDEs) and search engines. The functionalities of current LLMs can be modulated via natural language prompts, while their exact internal functionality remains implicit and unassessable. This property, which makes them adaptable to even unseen tasks, might also make them susceptible to targeted adversarial prompting. Recently, several ways to misalign LLMs using Prompt Injection (PI) attacks have been introduced. In such attacks, an adversary can prompt the LLM to produce malicious content or override the original instructions and the employed filtering schemes. Recent work showed that these attacks are hard to mitigate, as state-of-the-art LLMs are instruction-following. So far, these attacks assumed that the adversary is directly prompting the LLM.   In this work, we show that augmenting LLMs with retrieval and API calling capabilities (so-called Application-Integrated LLMs) induces a whole new set of attack vectors. These LLMs might process poisoned content retrieved from the Web that contains malicious prompts pre-injected and selected by adversaries. We demonstrate that an attacker can indirectly perform such PI attacks. Based on this key insight, we systematically analyze the resulting threat landscape of Application-Integrated LLMs and discuss a variety of new attack vectors. To demonstrate the practical viability of our attacks, we implemented specific demonstrations of the proposed attacks within synthetic applications. In summary, our work calls for an urgent evaluation of current mitigation techniques and an investigation of whether new techniques are needed to defend LLMs against these threats.

摘要: 我们目前正在见证大型语言模型(LLM)能力的巨大进步。它们已经在实践中被采用，并集成到许多系统中，包括集成开发环境(IDE)和搜索引擎。当前LLMS的功能可以通过自然语言提示进行调整，而其确切的内部功能仍然是隐含的和不可评估的。这一特性使他们能够适应甚至是看不见的任务，也可能使他们容易受到有针对性的对抗性提示。最近，已经引入了几种使用快速注入(PI)攻击来错位LLM的方法。在此类攻击中，攻击者可以提示LLM生成恶意内容或覆盖原始指令和采用的过滤方案。最近的研究表明，这些攻击很难缓解，因为最先进的LLM是遵循指令的。到目前为止，这些攻击假设对手是直接提示LLM的。在这项工作中，我们证明了增加LLMS的检索和API调用能力(即所谓的应用集成LLMS)会导致一组全新的攻击矢量。这些LLM可能会处理从Web检索的有毒内容，其中包含预先注入并由对手选择的恶意提示。我们证明攻击者可以间接地执行此类PI攻击。基于这一关键见解，我们系统地分析了应用集成的LLMS所产生的威胁环境，并讨论了各种新的攻击载体。为了证明我们的攻击的实际可行性，我们在合成应用程序中实现了拟议攻击的具体演示。总之，我们的工作需要对当前的缓解技术进行紧急评估，并调查是否需要新技术来保护小岛屿发展中国家免受这些威胁。



## **27. A Plot is Worth a Thousand Words: Model Information Stealing Attacks via Scientific Plots**

一个阴谋胜过千言万语：通过科学阴谋建立信息窃取攻击模型 cs.CR

To appear in the 32nd USENIX Security Symposium, August 2023,  Anaheim, CA, USA

**SubmitDate**: 2023-02-23    [abs](http://arxiv.org/abs/2302.11982v1) [paper-pdf](http://arxiv.org/pdf/2302.11982v1)

**Authors**: Boyang Zhang, Xinlei He, Yun Shen, Tianhao Wang, Yang Zhang

**Abstract**: Building advanced machine learning (ML) models requires expert knowledge and many trials to discover the best architecture and hyperparameter settings. Previous work demonstrates that model information can be leveraged to assist other attacks, such as membership inference, generating adversarial examples. Therefore, such information, e.g., hyperparameters, should be kept confidential. It is well known that an adversary can leverage a target ML model's output to steal the model's information. In this paper, we discover a new side channel for model information stealing attacks, i.e., models' scientific plots which are extensively used to demonstrate model performance and are easily accessible. Our attack is simple and straightforward. We leverage the shadow model training techniques to generate training data for the attack model which is essentially an image classifier. Extensive evaluation on three benchmark datasets shows that our proposed attack can effectively infer the architecture/hyperparameters of image classifiers based on convolutional neural network (CNN) given the scientific plot generated from it. We also reveal that the attack's success is mainly caused by the shape of the scientific plots, and further demonstrate that the attacks are robust in various scenarios. Given the simplicity and effectiveness of the attack method, our study indicates scientific plots indeed constitute a valid side channel for model information stealing attacks. To mitigate the attacks, we propose several defense mechanisms that can reduce the original attacks' accuracy while maintaining the plot utility. However, such defenses can still be bypassed by adaptive attacks.

摘要: 构建高级机器学习(ML)模型需要专业知识和多次试验来发现最佳架构和超参数设置。以前的工作表明，模型信息可以被用来辅助其他攻击，如成员推理，生成对抗性示例。因此，这类信息，例如超级参数，应该保密。众所周知，敌手可以利用目标ML模型的输出来窃取该模型的信息。在本文中，我们发现了模型信息窃取攻击的一种新的侧通道，即模型的科学情节，它被广泛用于演示模型的性能，并且易于访问。我们的攻击简单明了。我们利用阴影模型训练技术为攻击模型生成训练数据，该模型本质上是一个图像分类器。在三个基准数据集上的广泛评估表明，我们提出的攻击可以有效地推断基于卷积神经网络(CNN)的图像分类器的结构/超参数，从而生成科学的情节。我们还揭示了攻击的成功主要是由科学阴谋的形状造成的，并进一步证明了攻击在各种场景下都是健壮的。鉴于攻击方法的简单性和有效性，我们的研究表明，科学阴谋确实构成了模型信息窃取攻击的有效侧通道。为了缓解攻击，我们提出了几种防御机制，可以在保持情节效用的同时降低原始攻击的准确性。然而，这种防御仍然可以被适应性攻击绕过。



## **28. CalFAT: Calibrated Federated Adversarial Training with Label Skewness**

卡尔法特：带有标签偏斜度的校准联合对抗性训练 cs.LG

Accepted to the Conference on the Advances in Neural Information  Processing Systems (NeurIPS) 2022

**SubmitDate**: 2023-02-23    [abs](http://arxiv.org/abs/2205.14926v3) [paper-pdf](http://arxiv.org/pdf/2205.14926v3)

**Authors**: Chen Chen, Yuchen Liu, Xingjun Ma, Lingjuan Lyu

**Abstract**: Recent studies have shown that, like traditional machine learning, federated learning (FL) is also vulnerable to adversarial attacks. To improve the adversarial robustness of FL, federated adversarial training (FAT) methods have been proposed to apply adversarial training locally before global aggregation. Although these methods demonstrate promising results on independent identically distributed (IID) data, they suffer from training instability on non-IID data with label skewness, resulting in degraded natural accuracy. This tends to hinder the application of FAT in real-world applications where the label distribution across the clients is often skewed. In this paper, we study the problem of FAT under label skewness, and reveal one root cause of the training instability and natural accuracy degradation issues: skewed labels lead to non-identical class probabilities and heterogeneous local models. We then propose a Calibrated FAT (CalFAT) approach to tackle the instability issue by calibrating the logits adaptively to balance the classes. We show both theoretically and empirically that the optimization of CalFAT leads to homogeneous local models across the clients and better convergence points.

摘要: 最近的研究表明，与传统的机器学习一样，联邦学习(FL)也容易受到对手攻击。为了提高FL的对抗健壮性，联合对抗训练(FAT)方法被提出在全局聚集之前局部应用对抗训练。虽然这些方法在独立同分布(IID)数据上显示了良好的结果，但它们在具有标签偏斜的非IID数据上存在训练不稳定性，导致自然精度降低。这往往会阻碍FAT在实际应用中的应用，在现实应用中，跨客户端的标签分布通常是不对称的。本文研究了标签倾斜下的FAT问题，揭示了训练不稳定和自然精度下降的一个根本原因：倾斜的标签会导致类别概率不同和局部模型的异构性。然后，我们提出了一种校准FAT(CALFAT)方法来解决不稳定性问题，方法是自适应地校准逻辑以平衡类别。我们从理论和经验两个方面证明了CALFAT算法的优化可以得到跨客户的同质局部模型和更好的收敛点。



## **29. Adversarial Contrastive Distillation with Adaptive Denoising**

对抗性对比蒸馏与自适应去噪 cs.CV

accepted for ICASSP 2023

**SubmitDate**: 2023-02-23    [abs](http://arxiv.org/abs/2302.08764v2) [paper-pdf](http://arxiv.org/pdf/2302.08764v2)

**Authors**: Yuzheng Wang, Zhaoyu Chen, Dingkang Yang, Yang Liu, Siao Liu, Wenqiang Zhang, Lizhe Qi

**Abstract**: Adversarial Robustness Distillation (ARD) is a novel method to boost the robustness of small models. Unlike general adversarial training, its robust knowledge transfer can be less easily restricted by the model capacity. However, the teacher model that provides the robustness of knowledge does not always make correct predictions, interfering with the student's robust performances. Besides, in the previous ARD methods, the robustness comes entirely from one-to-one imitation, ignoring the relationship between examples. To this end, we propose a novel structured ARD method called Contrastive Relationship DeNoise Distillation (CRDND). We design an adaptive compensation module to model the instability of the teacher. Moreover, we utilize the contrastive relationship to explore implicit robustness knowledge among multiple examples. Experimental results on multiple attack benchmarks show CRDND can transfer robust knowledge efficiently and achieves state-of-the-art performances.

摘要: 对抗稳健性蒸馏(ARD)是一种提高小模型稳健性的新方法。与一般的对抗性训练不同，其稳健的知识传递不太容易受到模型容量的限制。然而，提供知识稳健性的教师模型并不总是做出正确的预测，干扰了学生的稳健表现。此外，在以往的ARD方法中，鲁棒性完全来自一对一的模仿，忽略了样本之间的关系。为此，我们提出了一种新的结构化ARD方法，称为对比关系降噪蒸馏(CRDND)。我们设计了一个自适应补偿模块来模拟教师的不稳定性。此外，我们利用这种对比关系来探索多个实例之间的隐含稳健性知识。在多个攻击基准上的实验结果表明，CRDND能够有效地传递健壮的知识，并达到最先进的性能。



## **30. Mitigating Adversarial Attacks in Deepfake Detection: An Exploration of Perturbation and AI Techniques**

深度伪码检测中对抗攻击的缓解：扰动和人工智能技术的探讨 cs.LG

**SubmitDate**: 2023-02-22    [abs](http://arxiv.org/abs/2302.11704v1) [paper-pdf](http://arxiv.org/pdf/2302.11704v1)

**Authors**: Saminder Dhesi, Laura Fontes, Pedro Machado, Isibor Kennedy Ihianle, Farhad Fassihi Tash, David Ada Adama

**Abstract**: Deep learning is a crucial aspect of machine learning, but it also makes these techniques vulnerable to adversarial examples, which can be seen in a variety of applications. These examples can even be targeted at humans, leading to the creation of false media, such as deepfakes, which are often used to shape public opinion and damage the reputation of public figures. This article will explore the concept of adversarial examples, which are comprised of perturbations added to clean images or videos, and their ability to deceive DL algorithms. The proposed approach achieved a precision value of accuracy of 76.2% on the DFDC dataset.

摘要: 深度学习是机器学习的一个关键方面，但它也使这些技术容易受到敌意示例的攻击，这在各种应用中都可以看到。这些例子甚至可以针对人类，导致制造虚假媒体，如深度假，这些媒体经常被用来塑造舆论，损害公众人物的声誉。本文将探讨对抗性例子的概念，它由添加到干净图像或视频的扰动组成，以及它们欺骗DL算法的能力。该方法在DFDC数据集上取得了76.2%的准确率。



## **31. Decorrelative Network Architecture for Robust Electrocardiogram Classification**

用于稳健心电分类的解相关网络结构 cs.LG

16 pages, 6 figures

**SubmitDate**: 2023-02-22    [abs](http://arxiv.org/abs/2207.09031v3) [paper-pdf](http://arxiv.org/pdf/2207.09031v3)

**Authors**: Christopher Wiedeman, Ge Wang

**Abstract**: Artificial intelligence has made great progress in medical data analysis, but the lack of robustness and trustworthiness has kept these methods from being widely deployed. As it is not possible to train networks that are accurate in all situations, models must recognize situations where they cannot operate confidently. Bayesian deep learning methods sample the model parameter space to estimate uncertainty, but these parameters are often subject to the same vulnerabilities, which can be exploited by adversarial attacks. We propose a novel ensemble approach based on feature decorrelation and Fourier partitioning for teaching networks diverse complementary features, reducing the chance of perturbation-based fooling. We test our approach on electrocardiogram classification, demonstrating superior accuracy confidence measurement, on a variety of adversarial attacks. For example, on our ensemble trained with both decorrelation and Fourier partitioning scored a 50.18% inference accuracy and 48.01% uncertainty accuracy (area under the curve) on {\epsilon} = 50 projected gradient descent attacks, while a conventionally trained ensemble scored 21.1% and 30.31% on these metrics respectively. Our approach does not require expensive optimization with adversarial samples and can be scaled to large problems. These methods can easily be applied to other tasks for more robust and trustworthy models.

摘要: 人工智能在医疗数据分析方面取得了很大进展，但缺乏健壮性和可信性，阻碍了这些方法的广泛部署。由于不可能训练出在所有情况下都准确的网络，模型必须认识到它们不能自信地运行的情况。贝叶斯深度学习方法对模型参数空间进行采样以估计不确定性，但这些参数经常受到相同的漏洞的影响，这可能被对抗性攻击所利用。我们提出了一种新的基于特征去相关和傅立叶划分的集成方法，用于训练网络中不同的互补特征，减少了基于扰动的愚弄的机会。我们在心电图分类上测试了我们的方法，在各种对抗性攻击上展示了优越的准确性和置信度测量。例如，在我们用去相关和傅立叶划分训练的集成上，在{\epsilon}=50个投影梯度下降攻击上，推理准确率和不确定性准确率(曲线下面积)分别为50.18%和48.01%，而常规训练的集成在这些指标上的得分分别为21.1%和30.31%。我们的方法不需要使用对抗性样本进行昂贵的优化，并且可以扩展到大型问题。这些方法可以很容易地应用于其他任务，以获得更健壮和可靠的模型。



## **32. Disrupting Adversarial Transferability in Deep Neural Networks**

深度神经网络中破坏对抗性转移的方法 cs.LG

20 pages, 13 figures

**SubmitDate**: 2023-02-22    [abs](http://arxiv.org/abs/2108.12492v3) [paper-pdf](http://arxiv.org/pdf/2108.12492v3)

**Authors**: Christopher Wiedeman, Ge Wang

**Abstract**: Adversarial attack transferability is well-recognized in deep learning. Prior work has partially explained transferability by recognizing common adversarial subspaces and correlations between decision boundaries, but little is known beyond this. We propose that transferability between seemingly different models is due to a high linear correlation between the feature sets that different networks extract. In other words, two models trained on the same task that are distant in the parameter space likely extract features in the same fashion, just with trivial affine transformations between the latent spaces. Furthermore, we show how applying a feature correlation loss, which decorrelates the extracted features in a latent space, can reduce the transferability of adversarial attacks between models, suggesting that the models complete tasks in semantically different ways. Finally, we propose a Dual Neck Autoencoder (DNA), which leverages this feature correlation loss to create two meaningfully different encodings of input information with reduced transferability.

摘要: 对抗性攻击的可转移性在深度学习中得到了很好的认可。以前的工作已经通过识别共同的敌对子空间和决策边界之间的相关性来部分解释可转移性，但除此之外知之甚少。我们认为，在看似不同的模型之间的可转移性是由于不同网络提取的特征集之间的高度线性相关性。换句话说，在参数空间中相距较远的两个在同一任务上训练的模型可能以相同的方式提取特征，只是在潜在空间之间进行了简单的仿射变换。此外，我们还展示了如何应用特征相关性损失来降低模型之间的对抗性攻击的可转移性，这表明两个模型以不同的语义方式完成任务。最后，我们提出了一种双颈自动编码器(DNA)，它利用这种特征相关性损失来创建两种有意义的不同的输入信息编码，降低了可传输性。



## **33. Public Key Encryption with Secure Key Leasing**

使用安全密钥租赁的公钥加密 quant-ph

67 pages, 4 figures

**SubmitDate**: 2023-02-22    [abs](http://arxiv.org/abs/2302.11663v1) [paper-pdf](http://arxiv.org/pdf/2302.11663v1)

**Authors**: Shweta Agrawal, Fuyuki Kitagawa, Ryo Nishimaki, Shota Yamada, Takashi Yamakawa

**Abstract**: We introduce the notion of public key encryption with secure key leasing (PKE-SKL). Our notion supports the leasing of decryption keys so that a leased key achieves the decryption functionality but comes with the guarantee that if the quantum decryption key returned by a user passes a validity test, then the user has lost the ability to decrypt. Our notion is similar in spirit to the notion of secure software leasing (SSL) introduced by Ananth and La Placa (Eurocrypt 2021) but captures significantly more general adversarial strategies. In more detail, our adversary is not restricted to use an honest evaluation algorithm to run pirated software. Our results can be summarized as follows:   1. Definitions: We introduce the definition of PKE with secure key leasing and formalize security notions.   2. Constructing PKE with Secure Key Leasing: We provide a construction of PKE-SKL by leveraging a PKE scheme that satisfies a new security notion that we call consistent or inconsistent security against key leasing attacks (CoIC-KLA security). We then construct a CoIC-KLA secure PKE scheme using 1-key Ciphertext-Policy Functional Encryption (CPFE) that in turn can be based on any IND-CPA secure PKE scheme.   3. Identity Based Encryption, Attribute Based Encryption and Functional Encryption with Secure Key Leasing: We provide definitions of secure key leasing in the context of advanced encryption schemes such as identity based encryption (IBE), attribute-based encryption (ABE) and functional encryption (FE). Then we provide constructions by combining the above PKE-SKL with standard IBE, ABE and FE schemes.

摘要: 我们引入了公钥加密和安全密钥租赁(PKE-SKL)的概念。我们的想法支持租赁解密密钥，以便租赁的密钥实现解密功能，但同时也保证，如果用户返回的量子解密密钥通过有效性测试，则用户已失去解密能力。我们的概念在精神上类似于Ananth和La Placa(Eurocrypt 2021)提出的安全软件租赁(SSL)概念，但捕获了更一般的对抗策略。更详细地说，我们的对手并不局限于使用诚实的评估算法来运行盗版软件。定义：引入了具有安全密钥租赁的PKE的定义，并对安全概念进行了形式化描述。2.使用安全密钥租赁构建PKE：利用一种新的PKE方案来构造PKE-SKL，该方案满足一种新的安全概念，即针对密钥租赁攻击的一致或不一致安全(COIC-KLA安全)。然后，我们使用1密钥密文策略函数加密(CPFE)构造了COIC-KLA安全PKE方案，而CPFE又可以基于任何IND-CPA安全PKE方案。3.基于身份的加密、基于属性的加密和基于安全密钥租赁的功能加密：我们在基于身份的加密(IBE)、基于属性的加密(ABE)和功能加密(FE)等高级加密方案的背景下定义了安全密钥租赁。然后，我们将上述PKE-SKL与标准的IBE、ABE和FE格式相结合，给出了构造方法。



## **34. Designing a Visual Cryptography Curriculum for K-12 Education**

面向K-12教育的可视密码学课程设计 cs.CR

**SubmitDate**: 2023-02-22    [abs](http://arxiv.org/abs/2302.11655v1) [paper-pdf](http://arxiv.org/pdf/2302.11655v1)

**Authors**: Pranathi Rayavaram, Sreekriti Sista, Ashwin Jagadeesha, Justin Marwad, Nathan Percival, Sashank Narain, Claire Seungeun Lee

**Abstract**: We have designed and developed a simple, visual, and narrative K-12 cybersecurity curriculum leveraging the Scratch programming platform to demonstrate and teach fundamental cybersecurity concepts such as confidentiality, integrity protection, and authentication. The visual curriculum simulates a real-world scenario of a user and a bank performing a bank transaction and an adversary attempting to attack the transaction.We have designed six visual scenarios, the curriculum first introduces students to three visual scenarios demonstrating attacks that exist when systems do not integrate concepts such as confidentiality, integrity protection, and authentication. Then, it introduces them to three visual scenarios that build on the attacks to demonstrate and teach how these fundamental concepts can be used to defend against them. We conducted an evaluation of our curriculum through a study with 18 middle and high school students. To evaluate the student's comprehension of these concepts we distributed a technical survey, where overall average of students answering these questions related to the demonstrated concepts is 9.28 out of 10. Furthermore, the survey results revealed that 66.7% found the system extremely easy and the remaining 27.8% found it easy to use and understand.

摘要: 我们利用Scratch编程平台设计和开发了一个简单、直观和叙述性的K-12网络安全课程，以演示和教授基本的网络安全概念，如机密性、完整性保护和身份验证。可视化课程模拟用户和银行执行银行交易以及对手试图攻击交易的真实场景。我们设计了六个可视化场景，该课程首先向学生介绍了三个可视化场景，演示了当系统没有集成机密性、完整性保护和身份验证等概念时存在的攻击。然后，它向他们介绍了三个基于攻击的可视场景，以演示和教授如何使用这些基本概念来防御攻击。通过对18名初中生的研究，我们对我们的课程进行了评估。为了评估学生对这些概念的理解程度，我们进行了一项技术调查，回答这些与演示概念相关的问题的学生的平均得分为9.28分(满分10分)。此外，调查结果显示，66.7%的学生认为该系统非常容易，其余27.8%的学生认为它易于使用和理解。



## **35. Feature Partition Aggregation: A Fast Certified Defense Against a Union of Sparse Adversarial Attacks**

特征分区聚合：针对稀疏对抗性攻击联盟的快速认证防御 cs.LG

**SubmitDate**: 2023-02-22    [abs](http://arxiv.org/abs/2302.11628v1) [paper-pdf](http://arxiv.org/pdf/2302.11628v1)

**Authors**: Zayd Hammoudeh, Daniel Lowd

**Abstract**: Deep networks are susceptible to numerous types of adversarial attacks. Certified defenses provide guarantees on a model's robustness, but most of these defenses are restricted to a single attack type. In contrast, this paper proposes feature partition aggregation (FPA) - a certified defense against a union of attack types, namely evasion, backdoor, and poisoning attacks. We specifically consider an $\ell_0$ or sparse attacker that arbitrarily controls an unknown subset of the training and test features - even across all instances. FPA generates robustness guarantees via an ensemble whose submodels are trained on disjoint feature sets. Following existing certified sparse defenses, we generalize FPA's guarantees to top-$k$ predictions. FPA significantly outperforms state-of-the-art sparse defenses providing larger and stronger robustness guarantees, while simultaneously being up to 5,000${\times}$ faster.

摘要: 深层网络容易受到多种类型的对抗性攻击。经过认证的防御为模型的健壮性提供了保证，但大多数此类防御仅限于单一攻击类型。相反，本文提出了特征分区聚合(FPA)，这是一种针对多种攻击类型的认证防御，即逃避攻击、后门攻击和中毒攻击。我们特别考虑$\ELL_0$或稀疏攻击者，它可以任意控制训练和测试功能的未知子集--甚至跨越所有实例。FPA通过在不相交的特征集上训练子模型的集成来生成健壮性保证。根据已有的认证稀疏防御，我们将FPA的保证推广到TOP-$K$预测。FPA的性能远远超过最先进的稀疏防御，提供更大和更强大的健壮性保证，同时速度快达5,000美元。



## **36. PAD: Towards Principled Adversarial Malware Detection Against Evasion Attacks**

PAD：针对逃避攻击的原则性恶意软件检测 cs.CR

20 pages; In submission

**SubmitDate**: 2023-02-22    [abs](http://arxiv.org/abs/2302.11328v1) [paper-pdf](http://arxiv.org/pdf/2302.11328v1)

**Authors**: Deqiang Li, Shicheng Cui, Yun Li, Jia Xu, Fu Xiao, Shouhuai Xu

**Abstract**: Machine Learning (ML) techniques facilitate automating malicious software (malware for short) detection, but suffer from evasion attacks. Many researchers counter such attacks in heuristic manners short of both theoretical guarantees and defense effectiveness. We hence propose a new adversarial training framework, termed Principled Adversarial Malware Detection (PAD), which encourages convergence guarantees for robust optimization methods. PAD lays on a learnable convex measurement that quantifies distribution-wise discrete perturbations and protects the malware detector from adversaries, by which for smooth detectors, adversarial training can be performed heuristically with theoretical treatments. To promote defense effectiveness, we propose a new mixture of attacks to instantiate PAD for enhancing the deep neural network-based measurement and malware detector. Experimental results on two Android malware datasets demonstrate: (i) the proposed method significantly outperforms the state-of-the-art defenses; (ii) it can harden the ML-based malware detection against 27 evasion attacks with detection accuracies greater than 83.45%, while suffering an accuracy decrease smaller than 2.16% in the absence of attacks; (iii) it matches or outperforms many anti-malware scanners in VirusTotal service against realistic adversarial malware.

摘要: 机器学习(ML)技术有助于自动检测恶意软件(简称恶意软件)，但受到逃避攻击。许多研究人员以缺乏理论保证和防御有效性的启发式方式反击此类攻击。因此，我们提出了一种新的对抗性训练框架，称为原则性对抗性恶意软件检测(PAD)，它鼓励健壮优化方法的收敛保证。PAD建立在可学习的凸度量上，该度量量化分布方向的离散扰动并保护恶意软件检测器免受攻击者的攻击，通过该度量，对于平滑的检测器，可以通过理论处理来启发式地执行敌意训练。为了提高防御效果，我们提出了一种新的混合攻击来实例化PAD，以增强基于深度神经网络的测量和恶意软件检测。在两个Android恶意软件数据集上的实验结果表明：(I)该方法的性能明显优于最新的防御方法；(Ii)对于27次逃避攻击，该方法可以强化基于ML的恶意软件检测，检测准确率高于83.45%，而在没有攻击的情况下，准确率下降小于2.16%；(Iii)它与VirusTotal服务中的许多反恶意软件扫描器相匹配或优于对现实恶意软件的检测。



## **37. A Hitting Time Analysis for Stochastic Time-Varying Functions with Applications to Adversarial Attacks on Computation of Markov Decision Processes**

随机时变函数的命中时间分析及其在马尔可夫决策过程计算对抗性攻击中的应用 math.OC

**SubmitDate**: 2023-02-22    [abs](http://arxiv.org/abs/2302.11190v1) [paper-pdf](http://arxiv.org/pdf/2302.11190v1)

**Authors**: Ali Yekkehkhany, Han Feng, Donghao Ying, Javad Lavaei

**Abstract**: Stochastic time-varying optimization is an integral part of learning in which the shape of the function changes over time in a non-deterministic manner. This paper considers multiple models of stochastic time variation and analyzes the corresponding notion of hitting time for each model, i.e., the period after which optimizing the stochastic time-varying function reveals informative statistics on the optimization of the target function. The studied models of time variation are motivated by adversarial attacks on the computation of value iteration in Markov decision processes. In this application, the hitting time quantifies the extent that the computation is robust to adversarial disturbance. We develop upper bounds on the hitting time by analyzing the contraction-expansion transformation appeared in the time-variation models. We prove that the hitting time of the value function in the value iteration with a probabilistic contraction-expansion transformation is logarithmic in terms of the inverse of a desired precision. In addition, the hitting time is analyzed for optimization of unknown continuous or discrete time-varying functions whose noisy evaluations are revealed over time. The upper bound for a continuous function is super-quadratic (but sub-cubic) in terms of the inverse of a desired precision and the upper bound for a discrete function is logarithmic in terms of the cardinality of the function domain. Improved bounds for convex functions are obtained and we show that such functions are learned faster than non-convex functions. Finally, we study a time-varying linear model with additive noise, where hitting time is bounded with the notion of shape dominance.

摘要: 随机时变优化是学习的一个组成部分，其中函数的形状随时间以不确定的方式变化。本文考虑了多个随机时变模型，并分析了每个模型对应的命中时间的概念，即在优化随机时变函数后的一段时间内，揭示了目标函数优化的信息统计。所研究的时变模型的动机是对马尔可夫决策过程中的值迭代计算的敌意攻击。在这种应用中，命中时间量化了计算对对手干扰的健壮性程度。通过分析时变模型中出现的收缩-扩张变换，给出了命中时间的上界。我们证明了值函数在概率收缩-扩张变换的值迭代中的击中时间是关于期望精度的倒数的对数。此外，对于未知的连续或离散时变函数的优化，分析了其命中时间，这些函数的噪声评估随着时间的推移而被揭示。连续函数的上界关于期望精度的倒数是超二次(但次三次)的，而离散函数的上界是关于函数域的基数的对数。得到了凸函数的改进界，并证明了这种函数的学习速度比非凸函数快。最后，我们研究了一类带加性噪声的时变线性模型，其中击中时间用形状优势的概念来定义。



## **38. MultiRobustBench: Benchmarking Robustness Against Multiple Attacks**

MultiRobustBch：针对多个攻击的健壮性基准 cs.LG

**SubmitDate**: 2023-02-21    [abs](http://arxiv.org/abs/2302.10980v1) [paper-pdf](http://arxiv.org/pdf/2302.10980v1)

**Authors**: Sihui Dai, Saeed Mahloujifar, Chong Xiang, Vikash Sehwag, Pin-Yu Chen, Prateek Mittal

**Abstract**: The bulk of existing research in defending against adversarial examples focuses on defending against a single (typically bounded Lp-norm) attack, but for a practical setting, machine learning (ML) models should be robust to a wide variety of attacks. In this paper, we present the first unified framework for considering multiple attacks against ML models. Our framework is able to model different levels of learner's knowledge about the test-time adversary, allowing us to model robustness against unforeseen attacks and robustness against unions of attacks. Using our framework, we present the first leaderboard, MultiRobustBench, for benchmarking multiattack evaluation which captures performance across attack types and attack strengths. We evaluate the performance of 16 defended models for robustness against a set of 9 different attack types, including Lp-based threat models, spatial transformations, and color changes, at 20 different attack strengths (180 attacks total). Additionally, we analyze the state of current defenses against multiple attacks. Our analysis shows that while existing defenses have made progress in terms of average robustness across the set of attacks used, robustness against the worst-case attack is still a big open problem as all existing models perform worse than random guessing.

摘要: 现有的大量研究集中于防御单一(通常有界的Lp范数)攻击，但对于实际环境，机器学习(ML)模型应该对各种攻击具有健壮性。在这篇文章中，我们提出了第一个考虑针对ML模型的多重攻击的统一框架。我们的框架能够对学习者关于测试时间对手的不同级别的知识进行建模，使我们能够建模对意外攻击的健壮性和对攻击组合的健壮性。使用我们的框架，我们提出了第一个排行榜，MultiRobustBch，用于对多攻击进行基准评估，该评估捕获了攻击类型和攻击强度的性能。我们评估了16种防御模型在20种不同攻击强度(总共180次攻击)下对9种不同攻击类型的健壮性，包括基于LP的威胁模型、空间变换和颜色变化。此外，我们还分析了当前对多种攻击的防御状态。我们的分析表明，尽管现有防御在使用的一组攻击的平均健壮性方面取得了进展，但对最坏情况攻击的健壮性仍然是一个巨大的开放问题，因为所有现有模型的表现都不如随机猜测。



## **39. Attacking Fake News Detectors via Manipulating News Social Engagement**

通过操纵新闻社会参与打击假新闻检测器 cs.SI

In Proceedings of the ACM Web Conference 2023 (WWW'23)

**SubmitDate**: 2023-02-21    [abs](http://arxiv.org/abs/2302.07363v2) [paper-pdf](http://arxiv.org/pdf/2302.07363v2)

**Authors**: Haoran Wang, Yingtong Dou, Canyu Chen, Lichao Sun, Philip S. Yu, Kai Shu

**Abstract**: Social media is one of the main sources for news consumption, especially among the younger generation. With the increasing popularity of news consumption on various social media platforms, there has been a surge of misinformation which includes false information or unfounded claims. As various text- and social context-based fake news detectors are proposed to detect misinformation on social media, recent works start to focus on the vulnerabilities of fake news detectors. In this paper, we present the first adversarial attack framework against Graph Neural Network (GNN)-based fake news detectors to probe their robustness. Specifically, we leverage a multi-agent reinforcement learning (MARL) framework to simulate the adversarial behavior of fraudsters on social media. Research has shown that in real-world settings, fraudsters coordinate with each other to share different news in order to evade the detection of fake news detectors. Therefore, we modeled our MARL framework as a Markov Game with bot, cyborg, and crowd worker agents, which have their own distinctive cost, budget, and influence. We then use deep Q-learning to search for the optimal policy that maximizes the rewards. Extensive experimental results on two real-world fake news propagation datasets demonstrate that our proposed framework can effectively sabotage the GNN-based fake news detector performance. We hope this paper can provide insights for future research on fake news detection.

摘要: 社交媒体是新闻消费的主要来源之一，尤其是在年轻一代中。随着新闻消费在各种社交媒体平台上的日益流行，错误信息激增，其中包括虚假信息或毫无根据的说法。随着各种基于文本和社会语境的假新闻检测器被提出来检测社交媒体上的错误信息，最近的研究开始关注假新闻检测器的脆弱性。本文提出了第一个针对基于图神经网络(GNN)的假新闻检测器的对抗性攻击框架，以探讨其健壮性。具体地说，我们利用多智能体强化学习(MAIL)框架来模拟社交媒体上欺诈者的对抗行为。研究表明，在现实世界中，欺诈者相互协调，分享不同的新闻，以躲避假新闻检测器的检测。因此，我们将我们的Marl框架建模为一个包含BOT、半机械人和群工代理的马尔可夫博弈，这些代理都有自己独特的成本、预算和影响。然后，我们使用深度Q-学习来搜索最大化回报的最优策略。在两个真实假新闻传播数据集上的大量实验结果表明，我们提出的框架可以有效地破坏基于GNN的假新闻检测器的性能。希望本文能为今后的假新闻检测研究提供一些启示。



## **40. MalProtect: Stateful Defense Against Adversarial Query Attacks in ML-based Malware Detection**

MalProtect：基于ML的恶意软件检测中对抗恶意查询攻击的状态防御 cs.LG

**SubmitDate**: 2023-02-21    [abs](http://arxiv.org/abs/2302.10739v1) [paper-pdf](http://arxiv.org/pdf/2302.10739v1)

**Authors**: Aqib Rashid, Jose Such

**Abstract**: ML models are known to be vulnerable to adversarial query attacks. In these attacks, queries are iteratively perturbed towards a particular class without any knowledge of the target model besides its output. The prevalence of remotely-hosted ML classification models and Machine-Learning-as-a-Service platforms means that query attacks pose a real threat to the security of these systems. To deal with this, stateful defenses have been proposed to detect query attacks and prevent the generation of adversarial examples by monitoring and analyzing the sequence of queries received by the system. Several stateful defenses have been proposed in recent years. However, these defenses rely solely on similarity or out-of-distribution detection methods that may be effective in other domains. In the malware detection domain, the methods to generate adversarial examples are inherently different, and therefore we find that such detection mechanisms are significantly less effective. Hence, in this paper, we present MalProtect, which is a stateful defense against query attacks in the malware detection domain. MalProtect uses several threat indicators to detect attacks. Our results show that it reduces the evasion rate of adversarial query attacks by 80+\% in Android and Windows malware, across a range of attacker scenarios. In the first evaluation of its kind, we show that MalProtect outperforms prior stateful defenses, especially under the peak adversarial threat.

摘要: 众所周知，ML模型容易受到敌意查询攻击。在这些攻击中，查询被迭代地扰动到特定的类，除了其输出之外，不知道目标模型。远程托管的ML分类模型和机器学习即服务平台的流行意味着查询攻击对这些系统的安全构成了真正的威胁。为了解决这个问题，已经提出了状态防御来检测查询攻击，并通过监控和分析系统接收到的查询序列来防止敌对实例的生成。近年来，有人提出了几项有状态的辩护。然而，这些防御完全依赖于可能在其他领域有效的相似性或分布外检测方法。在恶意软件检测领域，生成恶意示例的方法本质上是不同的，因此我们发现这种检测机制的有效性显著降低。因此，在本文中，我们提出了MalProtect，它是恶意软件检测领域中针对查询攻击的一种状态防御。MalProtect使用多个威胁指示器来检测攻击。我们的结果表明，在各种攻击场景下，该算法将Android和Windows恶意软件中恶意查询攻击的逃避率降低了80%+\%。在该类型的第一次评估中，我们表明MalProtect的性能优于先前的状态防御，特别是在峰值敌意威胁下。



## **41. Characterizing the Optimal 0-1 Loss for Multi-class Classification with a Test-time Attacker**

具有测试时间攻击的多类分类最优0-1损失的刻画 cs.LG

**SubmitDate**: 2023-02-21    [abs](http://arxiv.org/abs/2302.10722v1) [paper-pdf](http://arxiv.org/pdf/2302.10722v1)

**Authors**: Sihui Dai, Wenxin Ding, Arjun Nitin Bhagoji, Daniel Cullina, Ben Y. Zhao, Haitao Zheng, Prateek Mittal

**Abstract**: Finding classifiers robust to adversarial examples is critical for their safe deployment. Determining the robustness of the best possible classifier under a given threat model for a given data distribution and comparing it to that achieved by state-of-the-art training methods is thus an important diagnostic tool. In this paper, we find achievable information-theoretic lower bounds on loss in the presence of a test-time attacker for multi-class classifiers on any discrete dataset. We provide a general framework for finding the optimal 0-1 loss that revolves around the construction of a conflict hypergraph from the data and adversarial constraints. We further define other variants of the attacker-classifier game that determine the range of the optimal loss more efficiently than the full-fledged hypergraph construction. Our evaluation shows, for the first time, an analysis of the gap to optimal robustness for classifiers in the multi-class setting on benchmark datasets.

摘要: 寻找对敌意例子具有健壮性的分类器对于它们的安全部署至关重要。因此，确定在给定数据分布的给定威胁模型下的最佳可能分类器的稳健性，并将其与最先进的训练方法所实现的稳健性进行比较，是一种重要的诊断工具。在这篇文章中，我们找到了在任意离散数据集上的多类分类器在测试时间攻击者存在的情况下可获得的信息论损失下界。我们提供了一个寻找最优0-1损失的一般框架，该框架围绕着从数据和对抗性约束构造冲突超图。我们进一步定义了攻击者-分类器博弈的其他变体，它们比完整的超图构造更有效地确定最优损失的范围。我们的评估首次分析了在基准数据集上的多类设置下分类器的最优稳健性的差距。



## **42. Interpretable Spectrum Transformation Attacks to Speaker Recognition**

对说话人识别的可解释谱变换攻击 cs.SD

**SubmitDate**: 2023-02-21    [abs](http://arxiv.org/abs/2302.10686v1) [paper-pdf](http://arxiv.org/pdf/2302.10686v1)

**Authors**: Jiadi Yao, Hong Luo, Xiao-Lei Zhang

**Abstract**: The success of adversarial attacks to speaker recognition is mainly in white-box scenarios. When applying the adversarial voices that are generated by attacking white-box surrogate models to black-box victim models, i.e. \textit{transfer-based} black-box attacks, the transferability of the adversarial voices is not only far from satisfactory, but also lacks interpretable basis. To address these issues, in this paper, we propose a general framework, named spectral transformation attack based on modified discrete cosine transform (STA-MDCT), to improve the transferability of the adversarial voices to a black-box victim model. Specifically, we first apply MDCT to the input voice. Then, we slightly modify the energy of different frequency bands for capturing the salient regions of the adversarial noise in the time-frequency domain that are critical to a successful attack. Unlike existing approaches that operate voices in the time domain, the proposed framework operates voices in the time-frequency domain, which improves the interpretability, transferability, and imperceptibility of the attack. Moreover, it can be implemented with any gradient-based attackers. To utilize the advantage of model ensembling, we not only implement STA-MDCT with a single white-box surrogate model, but also with an ensemble of surrogate models. Finally, we visualize the saliency maps of adversarial voices by the class activation maps (CAM), which offers an interpretable basis to transfer-based attacks in speaker recognition for the first time. Extensive comparison results with five representative attackers show that the CAM visualization clearly explains the effectiveness of STA-MDCT, and the weaknesses of the comparison methods; the proposed method outperforms the comparison methods by a large margin.

摘要: 对抗性攻击对说话人识别的成功主要是在白盒场景中。在将攻击白盒代理模型产生的对抗性声音应用于黑盒受害者模型，即基于文本的黑盒攻击时，对抗性声音的可转移性不仅不理想，而且缺乏可解释的基础。针对这些问题，本文提出了一种基于修正离散余弦变换的谱变换攻击(STA-MDCT)框架，以提高敌方声音到黑箱受害者模型的可转换性。具体地说，我们首先对输入语音应用MDCT。然后，我们略微修改了不同频段的能量，以捕获时频域中对攻击成功至关重要的对抗性噪声的显著区域。与现有的在时间域中操作语音的方法不同，该框架在时频域中操作语音，从而提高了攻击的可解释性、可转移性和不可感知性。此外，它可以与任何基于梯度的攻击者一起实现。为了利用模型集成的优势，我们不仅用单个白盒代理模型实现了STA-MDCT，而且还用代理模型集成实现了STA-MDCT。最后，我们利用类激活映射(CAM)来可视化对抗性语音的显著图，这首次为说话人识别中基于转移的攻击提供了可解释的基础。与5个具有代表性的攻击者的广泛比较结果表明，CAM可视化清楚地解释了STA-MDCT的有效性，以及比较方法的弱点；所提出的方法的性能明显优于比较方法。



## **43. Adversarial Deep Reinforcement Learning for Improving the Robustness of Multi-agent Autonomous Driving Policies**

对抗性深度强化学习提高多智能体自主驾驶策略的稳健性 cs.AI

**SubmitDate**: 2023-02-21    [abs](http://arxiv.org/abs/2112.11937v3) [paper-pdf](http://arxiv.org/pdf/2112.11937v3)

**Authors**: Aizaz Sharif, Dusica Marijan

**Abstract**: Autonomous cars are well known for being vulnerable to adversarial attacks that can compromise the safety of the car and pose danger to other road users. To effectively defend against adversaries, it is required to not only test autonomous cars for finding driving errors but to improve the robustness of the cars to these errors. To this end, in this paper, we propose a two-step methodology for autonomous cars that consists of (i) finding failure states in autonomous cars by training the adversarial driving agent, and (ii) improving the robustness of autonomous cars by retraining them with effective adversarial inputs. Our methodology supports testing autonomous cars in a multi-agent environment, where we train and compare adversarial car policy on two custom reward functions to test the driving control decision of autonomous cars. We run experiments in a vision-based high-fidelity urban driving simulated environment. Our results show that adversarial testing can be used for finding erroneous autonomous driving behavior, followed by adversarial training for improving the robustness of deep reinforcement learning-based autonomous driving policies. We demonstrate that the autonomous cars retrained using the effective adversarial inputs noticeably increase the performance of their driving policies in terms of reduced collision and offroad steering errors.

摘要: 众所周知，自动驾驶汽车容易受到对抗性攻击，这些攻击可能会危及汽车的安全，并对其他道路使用者构成危险。为了有效地防御对手，不仅需要测试自动驾驶汽车是否发现驾驶错误，还需要提高汽车对这些错误的稳健性。为此，在本文中，我们提出了一种自动驾驶汽车的两步方法，包括(I)通过训练对抗性驾驶主体来发现自动驾驶汽车中的故障状态，(Ii)通过对自动驾驶汽车进行有效的对抗性输入来重新训练它们来提高自动驾驶汽车的稳健性。我们的方法支持在多智能体环境中测试自动驾驶汽车，在这种环境中，我们训练并比较两个定制奖励函数上的对抗性汽车策略，以测试自动驾驶汽车的驾驶控制决策。我们在基于视觉的高保真城市驾驶模拟环境中进行了实验。结果表明，对抗性测试可以用来发现错误的自主驾驶行为，然后通过对抗性训练来提高基于深度强化学习的自主驾驶策略的稳健性。我们证明，使用有效的对抗性输入进行再培训的自动驾驶汽车在减少碰撞和越野转向错误方面显著提高了其驾驶策略的性能。



## **44. CatchBackdoor: Backdoor Testing by Critical Trojan Neural Path Identification via Differential Fuzzing**

CatchBackdoor：基于差分模糊的关键木马神经路径识别的后门测试 cs.CR

There are some problems in the experiment so we need to withdraw this  paper. We will upload the new version after revision

**SubmitDate**: 2023-02-21    [abs](http://arxiv.org/abs/2112.13064v2) [paper-pdf](http://arxiv.org/pdf/2112.13064v2)

**Authors**: Haibo Jin, Ruoxi Chen, Jinyin Chen, Yao Cheng, Chong Fu, Ting Wang, Yue Yu, Zhaoyan Ming

**Abstract**: The success of deep neural networks (DNNs) in real-world applications has benefited from abundant pre-trained models. However, the backdoored pre-trained models can pose a significant trojan threat to the deployment of downstream DNNs. Existing DNN testing methods are mainly designed to find incorrect corner case behaviors in adversarial settings but fail to discover the backdoors crafted by strong trojan attacks. Observing the trojan network behaviors shows that they are not just reflected by a single compromised neuron as proposed by previous work but attributed to the critical neural paths in the activation intensity and frequency of multiple neurons. This work formulates the DNN backdoor testing and proposes the CatchBackdoor framework. Via differential fuzzing of critical neurons from a small number of benign examples, we identify the trojan paths and particularly the critical ones, and generate backdoor testing examples by simulating the critical neurons in the identified paths. Extensive experiments demonstrate the superiority of CatchBackdoor, with higher detection performance than existing methods. CatchBackdoor works better on detecting backdoors by stealthy blending and adaptive attacks, which existing methods fail to detect. Moreover, our experiments show that CatchBackdoor may reveal the potential backdoors of models in Model Zoo.

摘要: 深度神经网络(DNN)在实际应用中的成功得益于丰富的预训练模型。然而，后退的预先训练的模型可能会对下游DNN的部署构成重大的特洛伊木马威胁。现有的DNN测试方法主要是为了发现对抗性环境中不正确的角例行为，而无法发现由强木马攻击构建的后门。对木马网络行为的观察表明，木马网络行为并不像以前的工作那样只反映在单个受损神经元上，而是归因于多个神经元激活强度和频率中的关键神经路径。本文对DNN后门测试进行了阐述，提出了CatchBackdoor框架。通过对少数良性样本中的关键神经元进行差分模糊，识别木马路径，特别是关键路径，并通过模拟识别路径中的关键神经元生成后门测试用例。大量实验证明了CatchBackdoor的优越性，其检测性能高于现有方法。CatchBackdoor通过秘密混合和自适应攻击来更好地检测后门程序，而现有方法无法检测到这些攻击。此外，我们的实验表明，CatchBackdoor可能会揭示模型动物园中模型的潜在后门。



## **45. A Survey of Trustworthy Federated Learning with Perspectives on Security, Robustness, and Privacy**

基于安全性、健壮性和隐私性的可信联邦学习综述 cs.LG

**SubmitDate**: 2023-02-21    [abs](http://arxiv.org/abs/2302.10637v1) [paper-pdf](http://arxiv.org/pdf/2302.10637v1)

**Authors**: Yifei Zhang, Dun Zeng, Jinglong Luo, Zenglin Xu, Irwin King

**Abstract**: Trustworthy artificial intelligence (AI) technology has revolutionized daily life and greatly benefited human society. Among various AI technologies, Federated Learning (FL) stands out as a promising solution for diverse real-world scenarios, ranging from risk evaluation systems in finance to cutting-edge technologies like drug discovery in life sciences. However, challenges around data isolation and privacy threaten the trustworthiness of FL systems. Adversarial attacks against data privacy, learning algorithm stability, and system confidentiality are particularly concerning in the context of distributed training in federated learning. Therefore, it is crucial to develop FL in a trustworthy manner, with a focus on security, robustness, and privacy. In this survey, we propose a comprehensive roadmap for developing trustworthy FL systems and summarize existing efforts from three key aspects: security, robustness, and privacy. We outline the threats that pose vulnerabilities to trustworthy federated learning across different stages of development, including data processing, model training, and deployment. To guide the selection of the most appropriate defense methods, we discuss specific technical solutions for realizing each aspect of Trustworthy FL (TFL). Our approach differs from previous work that primarily discusses TFL from a legal perspective or presents FL from a high-level, non-technical viewpoint.

摘要: 值得信赖的人工智能(AI)技术彻底改变了日常生活，极大地造福了人类社会。在各种人工智能技术中，联合学习(FL)作为一种适用于各种现实世界场景的有前途的解决方案而脱颖而出，范围从金融领域的风险评估系统到生命科学中的药物发现等尖端技术。然而，围绕数据隔离和隐私的挑战威胁到FL系统的可信性。在联合学习的分布式训练环境中，针对数据隐私、学习算法稳定性和系统机密性的对抗性攻击尤其令人担忧。因此，以一种值得信赖的方式发展FL，并将重点放在安全性、健壮性和隐私上，这是至关重要的。在这项调查中，我们提出了开发可信FL系统的全面路线图，并从三个关键方面总结了现有的努力：安全性、健壮性和隐私。我们概述了在开发的不同阶段(包括数据处理、模型培训和部署)对可信任的联合学习构成漏洞的威胁。为了指导选择最合适的防御方法，我们讨论了实现可信FL(TFL)各个方面的具体技术方案。我们的研究方法不同于以往的研究，这些研究主要从法律的角度来讨论外语，或者从高层次、非技术性的角度来呈现外语。



## **46. Generalization Bounds for Adversarial Contrastive Learning**

对抗性对比学习的泛化界 cs.LG

**SubmitDate**: 2023-02-21    [abs](http://arxiv.org/abs/2302.10633v1) [paper-pdf](http://arxiv.org/pdf/2302.10633v1)

**Authors**: Xin Zou, Weiwei Liu

**Abstract**: Deep networks are well-known to be fragile to adversarial attacks, and adversarial training is one of the most popular methods used to train a robust model. To take advantage of unlabeled data, recent works have applied adversarial training to contrastive learning (Adversarial Contrastive Learning; ACL for short) and obtain promising robust performance. However, the theory of ACL is not well understood. To fill this gap, we leverage the Rademacher complexity to analyze the generalization performance of ACL, with a particular focus on linear models and multi-layer neural networks under $\ell_p$ attack ($p \ge 1$). Our theory shows that the average adversarial risk of the downstream tasks can be upper bounded by the adversarial unsupervised risk of the upstream task. The experimental results validate our theory.

摘要: 众所周知，深度网络对敌意攻击很脆弱，对抗性训练是训练健壮模型最常用的方法之一。为了利用未标记的数据，最近的工作将对抗性训练应用到对比学习(对抗性对比学习；简称ACL)中，并获得了有希望的稳健性能。然而，前交叉韧带的理论还没有被很好地理解。为了填补这一空白，我们利用Rademacher复杂性分析了ACL的泛化性能，重点分析了线性模型和多层神经网络在$\ell_p$攻击($p\ge 1$)下的性能。我们的理论表明，下游任务的平均对抗性风险可以由上游任务的对抗性无监督风险上界。实验结果验证了我们的理论。



## **47. Certified Defences Against Adversarial Patch Attacks on Semantic Segmentation**

基于语义分割的对抗性补丁攻击认证防御 cs.CV

accepted at ICLR 2023

**SubmitDate**: 2023-02-21    [abs](http://arxiv.org/abs/2209.05980v2) [paper-pdf](http://arxiv.org/pdf/2209.05980v2)

**Authors**: Maksym Yatsura, Kaspar Sakmann, N. Grace Hua, Matthias Hein, Jan Hendrik Metzen

**Abstract**: Adversarial patch attacks are an emerging security threat for real world deep learning applications. We present Demasked Smoothing, the first approach (up to our knowledge) to certify the robustness of semantic segmentation models against this threat model. Previous work on certifiably defending against patch attacks has mostly focused on image classification task and often required changes in the model architecture and additional training which is undesirable and computationally expensive. In Demasked Smoothing, any segmentation model can be applied without particular training, fine-tuning, or restriction of the architecture. Using different masking strategies, Demasked Smoothing can be applied both for certified detection and certified recovery. In extensive experiments we show that Demasked Smoothing can on average certify 64% of the pixel predictions for a 1% patch in the detection task and 48% against a 0.5% patch for the recovery task on the ADE20K dataset.

摘要: 对抗性补丁攻击是现实世界深度学习应用面临的一种新的安全威胁。我们提出了去任务平滑，这是第一种(据我们所知)来证明语义分割模型对这种威胁模型的稳健性的方法。以前关于可证明防御补丁攻击的工作主要集中在图像分类任务上，并且经常需要改变模型体系结构和额外的训练，这是不受欢迎的，并且计算代价高昂。在去任务平滑中，任何分割模型都可以在没有特定训练、微调或体系结构限制的情况下应用。使用不同的掩码策略，去掩码平滑可以应用于认证检测和认证恢复。在ADE20K数据集上的大量实验中，对于检测任务中1%的块，去任务平滑平均可以保证64%的像素预测，对于恢复任务，对于0.5%的块，去任务平滑平均可以保证48%的像素预测。



## **48. Internal Wasserstein Distance for Adversarial Attack and Defense**

对抗性攻防的瓦瑟斯坦内部距离 cs.LG

**SubmitDate**: 2023-02-21    [abs](http://arxiv.org/abs/2103.07598v4) [paper-pdf](http://arxiv.org/pdf/2103.07598v4)

**Authors**: Qicheng Wang, Shuhai Zhang, Jiezhang Cao, Jincheng Li, Mingkui Tan, Yang Xiang

**Abstract**: Deep neural networks (DNNs) are known to be vulnerable to adversarial attacks that would trigger misclassification of DNNs but may be imperceptible to human perception. Adversarial defense has been an important way to improve the robustness of DNNs. Existing attack methods often construct adversarial examples relying on some metrics like the $\ell_p$ distance to perturb samples. However, these metrics can be insufficient to conduct adversarial attacks due to their limited perturbations. In this paper, we propose a new internal Wasserstein distance (IWD) to capture the semantic similarity of two samples, and thus it helps to obtain larger perturbations than currently used metrics such as the $\ell_p$ distance. We then apply the internal Wasserstein distance to perform adversarial attack and defense. In particular, we develop a novel attack method relying on IWD to calculate the similarities between an image and its adversarial examples. In this way, we can generate diverse and semantically similar adversarial examples that are more difficult to defend by existing defense methods. Moreover, we devise a new defense method relying on IWD to learn robust models against unseen adversarial examples. We provide both thorough theoretical and empirical evidence to support our methods.

摘要: 深度神经网络(DNN)容易受到敌意攻击，这种攻击可能会导致DNN的错误分类，但可能无法被人类感知到。对抗性防御已成为提高DNN健壮性的重要途径。现有的攻击方法通常依赖于诸如$\ell_p$距离之类的度量来构建敌意示例来扰动样本。然而，由于其有限的扰动，这些指标可能不足以进行对抗性攻击。本文提出了一种新的内部Wasserstein距离(IWD)来刻画两个样本之间的语义相似性，从而有助于获得比目前使用的$\ell_p$距离等度量更大的扰动。然后，我们应用内部Wasserstein距离进行对抗性攻击和防御。特别是，我们开发了一种新的攻击方法，利用IWD来计算图像与其对手示例之间的相似度。通过这种方式，我们可以生成不同的和语义相似的对抗性例子，这些例子用现有的防御方法更难防御。此外，我们设计了一种新的防御方法，依赖于IWD来学习针对未知对手实例的稳健模型。我们提供了充分的理论和经验证据来支持我们的方法。



## **49. Model-based feature selection for neural networks: A mixed-integer programming approach**

基于模型的神经网络特征选择：一种混合整数规划方法 math.OC

15 pages, 3 figures, 5 tables

**SubmitDate**: 2023-02-20    [abs](http://arxiv.org/abs/2302.10344v1) [paper-pdf](http://arxiv.org/pdf/2302.10344v1)

**Authors**: Shudian Zhao, Calvin Tsay, Jan Kronqvist

**Abstract**: In this work, we develop a novel input feature selection framework for ReLU-based deep neural networks (DNNs), which builds upon a mixed-integer optimization approach. While the method is generally applicable to various classification tasks, we focus on finding input features for image classification for clarity of presentation. The idea is to use a trained DNN, or an ensemble of trained DNNs, to identify the salient input features. The input feature selection is formulated as a sequence of mixed-integer linear programming (MILP) problems that find sets of sparse inputs that maximize the classification confidence of each category. These ''inverse'' problems are regularized by the number of inputs selected for each category and by distribution constraints. Numerical results on the well-known MNIST and FashionMNIST datasets show that the proposed input feature selection allows us to drastically reduce the size of the input to $\sim$15\% while maintaining a good classification accuracy. This allows us to design DNNs with significantly fewer connections, reducing computational effort and producing DNNs that are more robust towards adversarial attacks.

摘要: 在这项工作中，我们开发了一种新的输入特征选择框架，用于基于REU的深度神经网络(DNN)，该框架建立在混合整数优化方法的基础上。虽然该方法一般适用于各种分类任务，但为了表达的清晰度，我们专注于为图像分类寻找输入特征。其思想是使用训练的DNN或训练的DNN的集合来识别显著的输入特征。输入特征选择被表示为混合整数线性规划(MILP)问题的序列，该问题找到最大化每个类别的分类置信度的稀疏输入集合。这些“逆”问题通过为每个类别选择的输入数量和分布约束来正规化。在著名的MNIST和FashionMNIST数据集上的数值结果表明，所提出的输入特征选择允许我们在保持良好的分类精度的同时将输入的大小大幅减少到$\sim$15\%。这使我们能够设计具有显著较少连接的DNN，减少计算工作量，并产生对对手攻击更健壮的DNN。



## **50. Robust Fair Clustering: A Novel Fairness Attack and Defense Framework**

稳健公平聚类：一种新的公平攻防框架 cs.LG

Accepted to the 11th International Conference on Learning  Representations (ICLR 2023)

**SubmitDate**: 2023-02-20    [abs](http://arxiv.org/abs/2210.01953v3) [paper-pdf](http://arxiv.org/pdf/2210.01953v3)

**Authors**: Anshuman Chhabra, Peizhao Li, Prasant Mohapatra, Hongfu Liu

**Abstract**: Clustering algorithms are widely used in many societal resource allocation applications, such as loan approvals and candidate recruitment, among others, and hence, biased or unfair model outputs can adversely impact individuals that rely on these applications. To this end, many fair clustering approaches have been recently proposed to counteract this issue. Due to the potential for significant harm, it is essential to ensure that fair clustering algorithms provide consistently fair outputs even under adversarial influence. However, fair clustering algorithms have not been studied from an adversarial attack perspective. In contrast to previous research, we seek to bridge this gap and conduct a robustness analysis against fair clustering by proposing a novel black-box fairness attack. Through comprehensive experiments, we find that state-of-the-art models are highly susceptible to our attack as it can reduce their fairness performance significantly. Finally, we propose Consensus Fair Clustering (CFC), the first robust fair clustering approach that transforms consensus clustering into a fair graph partitioning problem, and iteratively learns to generate fair cluster outputs. Experimentally, we observe that CFC is highly robust to the proposed attack and is thus a truly robust fair clustering alternative.

摘要: 聚类算法广泛应用于许多社会资源分配应用中，如贷款审批和候选人招聘等，因此，有偏见或不公平的模型输出可能会对依赖这些应用的个人产生不利影响。为此，最近提出了许多公平的聚类方法来解决这个问题。由于潜在的重大危害，确保公平的聚类算法即使在敌意影响下也能提供一致的公平输出是至关重要的。然而，公平分簇算法还没有从对抗攻击的角度进行研究。与以往的研究不同，我们试图弥补这一差距，并通过提出一种新的黑盒公平攻击来进行针对公平聚类的健壮性分析。通过全面的实验，我们发现最新的模型非常容易受到我们的攻击，因为它会显著降低它们的公平性能。最后，我们提出了共识公平聚类(CFC)，这是第一种将共识聚类转化为公平图划分问题的稳健公平聚类方法，并迭代地学习生成公平聚类输出。在实验上，我们观察到cfc对所提出的攻击具有高度的健壮性，因此是一种真正健壮的公平集群替代方案。



